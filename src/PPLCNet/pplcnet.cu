#include "PPLCNet/pplcnet.hpp"
#include "infer/infer.hpp"
#include "common/check.hpp"
#include "common/logger.hpp"
#include "common/image.hpp"
#include "common/dim.hpp"
#include "common/timer.hpp"
#include "memory/memory.hpp"
#include "preprocess/preprocess.hpp"


namespace pplcnet
{

using namespace std;



class InferImpl : public Infer 
{
public:
    shared_ptr<trt::Infer> trt_;
    string engine_file_;
    vector<shared_ptr<trt::Memory<unsigned char>>> preprocess_buffers_;
    trt::Memory<float> input_buffer_, output_array_;
    trt::Memory<int> classes_indices_;
    int network_input_width_, network_input_height_;
    pre::Norm normalize_;
    int num_classes_ = 0;
    bool isdynamic_model_ = false;

    virtual ~InferImpl() = default;

    void adjust_memory(int batch_size) 
    {
        // the inference batch_size
        size_t input_numel = network_input_width_ * network_input_height_ * 3;
        input_buffer_.gpu(batch_size * input_numel);
        output_array_.gpu(batch_size * num_classes_);
        output_array_.cpu(batch_size * num_classes_);
        classes_indices_.gpu(batch_size);
        classes_indices_.cpu(batch_size);


        if ((int)preprocess_buffers_.size() < batch_size) {
        for (int i = preprocess_buffers_.size(); i < batch_size; ++i)
            preprocess_buffers_.push_back(make_shared<trt::Memory<unsigned char>>());
        }
    }

    void preprocess(int ibatch, const trt::Image &image,
                    shared_ptr<trt::Memory<unsigned char>> preprocess_buffer,
                    void *stream = nullptr) {
        pre::ResizeShortCenterCropMatrix affine;
        affine.compute(make_tuple(image.width, image.height),
                    make_tuple(network_input_width_, network_input_height_), 256);

        size_t input_numel = network_input_width_ * network_input_height_ * 3;
        float *input_device = input_buffer_.gpu() + ibatch * input_numel;
        size_t size_image = image.width * image.height * 3;
        size_t size_matrix = upbound(sizeof(affine.d2i), 32);
        uint8_t *gpu_workspace = preprocess_buffer->gpu(size_matrix + size_image);
        float *affine_matrix_device = (float *)gpu_workspace;
        uint8_t *image_device = gpu_workspace + size_matrix;

        uint8_t *cpu_workspace = preprocess_buffer->cpu(size_matrix + size_image);
        float *affine_matrix_host = (float *)cpu_workspace;
        uint8_t *image_host = cpu_workspace + size_matrix;

        // speed up
        cudaStream_t stream_ = (cudaStream_t)stream;
        memcpy(image_host, image.bgrptr, size_image);
        memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
        checkRuntime(
            cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));
        checkRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i),
                                    cudaMemcpyHostToDevice, stream_));

        pre::warp_affine_bilinear_and_normalize_plane(image_device, image.width * 3, image.width,
                                                image.height, input_device, network_input_width_,
                                                network_input_height_, affine_matrix_device, 114,
                                                normalize_, stream_);
    }

    bool load(const string &engine_file) 
    {
        trt_ = trt::load(engine_file);
        if (trt_ == nullptr) return false;

        trt_->print();

        auto input_dim = trt_->static_dims(0);

        network_input_width_ = input_dim[3];
        network_input_height_ = input_dim[2];
        isdynamic_model_ = trt_->has_dynamic_dim();

        // normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
        // [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        float mean[3] = {0.485, 0.456, 0.406}; 
        float std[3] = {0.229, 0.224, 0.225};
        normalize_ = pre::Norm::mean_std(mean, std, 1/255.0,  pre::ChannelType::SwapRB);
        num_classes_ = trt_->static_dims(1)[1];
        return true;
    }

    virtual Attribute forward(const trt::Image &image, void *stream = nullptr) override 
    {
        auto output = forwards({image}, stream);
        if (output.empty()) return {};
        return output[0];
    }

    virtual vector<Attribute> forwards(const vector<trt::Image> &images, void *stream = nullptr) override 
    {
        int num_image = images.size();
        if (num_image == 0) return {};

        auto input_dims = trt_->static_dims(0);
        int infer_batch_size = input_dims[0];
        if (infer_batch_size != num_image) 
        {
            if (isdynamic_model_) 
            {
                infer_batch_size = num_image;
                input_dims[0] = num_image;
                if (!trt_->set_run_dims(0, input_dims)) return {};
            } 
            else 
            {
                if (infer_batch_size < num_image) 
                {
                    INFO(
                        "When using static shape model, number of images[%d] must be "
                        "less than or equal to the maximum batch[%d].",
                        num_image, infer_batch_size);
                    return {};
                }
            }
        }
        adjust_memory(infer_batch_size);

        cudaStream_t stream_ = (cudaStream_t)stream;
        for (int i = 0; i < num_image; ++i)
        {
            preprocess(i, images[i], preprocess_buffers_[i], stream);
        }
            

        float *output_array_device = output_array_.gpu();
        vector<void *> bindings{input_buffer_.gpu(), output_array_device};

        if (!trt_->forward(bindings, stream)) 
        {
            INFO("Failed to tensorRT forward.");
            return {};
        }

        checkRuntime(cudaMemcpyAsync(output_array_.cpu(), output_array_.gpu(),
                                    output_array_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_));
        checkRuntime(cudaMemcpyAsync(classes_indices_.cpu(), classes_indices_.gpu(),
                                    classes_indices_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_));
        checkRuntime(cudaStreamSynchronize(stream_));

        vector<Attribute> arrout;
        arrout.reserve(num_image);

        for (int ib = 0; ib < num_image; ++ib) 
        {
            float *output_array_cpu = output_array_.cpu() + ib * num_classes_;
            float max_score = -1;
            int max_index = -1;
            for (int cls_index = 0; cls_index < num_classes_; cls_index++)
            {
                if (output_array_cpu[cls_index] > max_score)
                {
                    max_score = output_array_cpu[cls_index];
                    max_index = cls_index;
                }
            }
            arrout.emplace_back(max_score, max_index);
        }
        return arrout;
    }
};

Infer *loadraw(const std::string &engine_file) 
{
    InferImpl *impl = new InferImpl();
    if (!impl->load(engine_file)) 
    {
        delete impl;
        impl = nullptr;
    }
    return impl;
}

shared_ptr<Infer> load(const string &engine_file, int gpu_id) 
{
    checkRuntime(cudaSetDevice(gpu_id));
    return std::shared_ptr<InferImpl>((InferImpl *)loadraw(engine_file));
}


} // namespace resnet
