#include "resnet/resnet.hpp"
#include "infer/infer.hpp"
#include "common/check.hpp"
#include "common/logger.hpp"
#include "common/image.hpp"
#include "common/dim.hpp"
#include "common/timer.hpp"
#include "memory/memory.hpp"
#include "preprocess/preprocess.hpp"


namespace resnet
{

using namespace std;


static __global__ void softmax(float *predict, int length, int *max_index) 
{
    extern __shared__ float shared_data[];
    float *shared_max_vals = shared_data;
    int *shared_max_indices = (int*)&shared_max_vals[blockDim.x];
    
    int tid = threadIdx.x;

    // 1. 找到最大值和最大值的下标，存储在共享内存中
    float max_val = -FLT_MAX;
    int max_idx = -1;
    for (int i = tid; i < length; i += blockDim.x) 
    {
        if (predict[i] > max_val) 
        {
            max_val = predict[i];
            max_idx = i;
        }
    }
    shared_max_vals[tid] = max_val;
    shared_max_indices[tid] = max_idx;
    __syncthreads();

    // 在所有线程间找到全局最大值和对应的下标
    if (tid == 0) 
    {
        for (int i = 1; i < blockDim.x; i++)
        {
            if (shared_max_vals[i] > shared_max_vals[0]) 
            {
                shared_max_vals[0] = shared_max_vals[i];
                shared_max_indices[0] = shared_max_indices[i];
            }
        }
        *max_index = shared_max_indices[0];
    }
    __syncthreads();

    max_val = shared_max_vals[0];

    // 2. 计算指数并求和
    float sum_exp = 0.0f;
    for (int i = tid; i < length; i += blockDim.x) 
    {
        predict[i] = expf(predict[i] - max_val);
        sum_exp += predict[i];
    }
    shared_max_vals[tid] = sum_exp;
    __syncthreads();

    // 汇总所有线程的指数和
    if (tid == 0) 
    {
        for (int i = 1; i < blockDim.x; i++) 
        {
            shared_max_vals[0] += shared_max_vals[i];
        }
    }
    __syncthreads();
    float total_sum = shared_max_vals[0];

    // 3. 每个元素除以总和，得到 softmax 值
    for (int i = tid; i < length; i += blockDim.x) 
    {
        predict[i] /= total_sum;
    }
}

static void classfier_softmax(float *predict, int length, int *max_index, cudaStream_t stream) 
{
    int block_size = 256;
    checkKernel(softmax<<<1, block_size, block_size * sizeof(float), stream>>>(predict, length, max_index));
}


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
        pre::ResizeMatrix affine;
        affine.compute(make_tuple(image.width, image.height),
                    make_tuple(network_input_width_, network_input_height_));

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

        warp_affine_bilinear_and_normalize_plane(image_device, image.width * 3, image.width,
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

        for (int ib = 0; ib < num_image; ++ib) 
        {
            float *output_array_device = output_array_.gpu() + ib * num_classes_;
            int *classes_indices_device = classes_indices_.gpu() + ib;
            classfier_softmax(output_array_device, num_classes_, classes_indices_device, stream_);
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
            int *max_index = classes_indices_.cpu() + ib;
            int index = *max_index;
            float max_score = output_array_cpu[index];
            arrout.emplace_back(max_score, index);
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

shared_ptr<Infer> load(const string &engine_file) 
{
    return std::shared_ptr<InferImpl>((InferImpl *)loadraw(engine_file));
}


} // namespace resnet
