#include "yolo/yolov11pose.hpp"
#include "infer/infer.hpp"
#include "common/check.hpp"
#include "common/logger.hpp"
#include "common/image.hpp"
#include "common/dim.hpp"
#include "common/timer.hpp"
#include "memory/memory.hpp"
#include "preprocess/preprocess.hpp"

namespace yolov11pose
{

using namespace std;

const int KEY_POINT_NUM = 17; // 关键点数量
const int NUM_BOX_ELEMENT = 8 + KEY_POINT_NUM * 3;  // left, top, right, bottom, confidence, class, keepflag, row_index(output), (KEY_POINT_NUMx3) key points
const int MAX_IMAGE_BOXES = 1024;


static __host__ __device__ void affine_project(float *matrix, float x, float y, float *ox, float *oy) 
{
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

static __global__ void decode_kernel_v11pose(float *predict, int num_bboxes, int num_classes,
                                              int output_cdim, float confidence_threshold,
                                              float *invert_affine_matrix, float *parray,
                                              int MAX_IMAGE_BOXES) 
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;

    float *pitem = predict + output_cdim * position;
    float *class_confidence = pitem + 4;
    float *key_points = pitem + 5;
    float confidence = *class_confidence++;
    int label = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence) 
    {
        if (*class_confidence > confidence) 
        {
            confidence = *class_confidence;
            label = i;
        }
    }
    if (confidence < confidence_threshold) return;

    int index = atomicAdd(parray, 1);
    if (index >= MAX_IMAGE_BOXES) return;

    float cx = *pitem++;
    float cy = *pitem++;
    float width = *pitem++;
    float height = *pitem++;
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1;  // 1 = keep, 0 = ignore
    *pout_item++ = position;
    for (int i = 0; i < KEY_POINT_NUM; i++)
    {
        float x = *key_points++;
        float y = *key_points++;
        affine_project(invert_affine_matrix, x, y, &x, &y);
        float score = *key_points++;
        *pout_item++ = x;
        *pout_item++ = y;
        *pout_item++ = score;
    }
}

static __device__ float box_iou(float aleft, float atop, float aright, float abottom, float bleft,
                                float btop, float bright, float bbottom) \
{
    float cleft = max(aleft, bleft);
    float ctop = max(atop, btop);
    float cright = min(aright, bright);
    float cbottom = min(abottom, bbottom);

    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f) return 0.0f;

    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void fast_nms_kernel(float *bboxes, int MAX_IMAGE_BOXES, float threshold) 
{
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = min((int)*bboxes, MAX_IMAGE_BOXES);
    if (position >= count) return;

    // left, top, right, bottom, confidence, class, keepflag
    float *pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
    for (int i = 0; i < count; ++i) 
    {
        float *pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        if (i == position || pcurrent[5] != pitem[5]) continue;

        if (pitem[4] >= pcurrent[4]) 
        {
            if (pitem[4] == pcurrent[4] && i < position) continue;

            float iou = box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pitem[0], pitem[1],
                                pitem[2], pitem[3]);

            if (iou > threshold) 
            {
                pcurrent[6] = 0;  // 1=keep, 0=ignore
                return;
            }
        }
    }
}

static void decode_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                  float confidence_threshold, float nms_threshold,
                                  float *invert_affine_matrix, float *parray, int MAX_IMAGE_BOXES,
                                cudaStream_t stream) 
{
    auto grid = grid_dims(num_bboxes);
    auto block = block_dims(num_bboxes);

    checkKernel(decode_kernel_v11pose<<<grid, block, 0, stream>>>(
            predict, num_bboxes, num_classes, output_cdim, confidence_threshold, invert_affine_matrix,
            parray, MAX_IMAGE_BOXES));

    grid = grid_dims(MAX_IMAGE_BOXES);
    block = block_dims(MAX_IMAGE_BOXES);
    checkKernel(fast_nms_kernel<<<grid, block, 0, stream>>>(parray, MAX_IMAGE_BOXES, nms_threshold));
}


class InferImpl : public Infer 
{

public:
    shared_ptr<trt::Infer> trt_;
    string engine_file_;
    float confidence_threshold_;
    float nms_threshold_;
    vector<shared_ptr<trt::Memory<unsigned char>>> preprocess_buffers_;
    trt::Memory<float> input_buffer_, bbox_predict_, output_boxarray_;
    int network_input_width_, network_input_height_;
    pre::Norm normalize_;
    vector<int> bbox_head_dims_;
    int num_classes_ = 0;
    bool isdynamic_model_ = false;

    virtual ~InferImpl() = default;

    void adjust_memory(int batch_size) 
    {
        // the inference batch_size
        size_t input_numel = network_input_width_ * network_input_height_ * 3;
        input_buffer_.gpu(batch_size * input_numel);
        bbox_predict_.gpu(batch_size * bbox_head_dims_[1] * bbox_head_dims_[2]);
        output_boxarray_.gpu(batch_size * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));
        output_boxarray_.cpu(batch_size * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));

        if ((int)preprocess_buffers_.size() < batch_size) 
        {
            for (int i = preprocess_buffers_.size(); i < batch_size; ++i)
            {
                preprocess_buffers_.push_back(make_shared<trt::Memory<unsigned char>>());
            }
        }
    }

    void preprocess(int ibatch, const trt::Image &image,
                    shared_ptr<trt::Memory<unsigned char>> preprocess_buffer, pre::AffineMatrix &affine,
                    void *stream = nullptr)
    {
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

    bool load(const string &engine_file, float confidence_threshold, float nms_threshold) 
    {
        trt_ = trt::load(engine_file);
        if (trt_ == nullptr) return false;

        trt_->print();

        this->confidence_threshold_ = confidence_threshold;
        this->nms_threshold_ = nms_threshold;

        auto input_dim = trt_->static_dims(0);
        bbox_head_dims_ = trt_->static_dims(1);
        network_input_width_ = input_dim[3];
        network_input_height_ = input_dim[2];
        isdynamic_model_ = trt_->has_dynamic_dim();

        normalize_ = pre::Norm::alpha_beta(1 / 255.0f, 0.0f, pre::ChannelType::SwapRB);
        num_classes_ = 1;
        return true;
    }

    virtual BoxArray forward(const trt::Image &image, void *stream = nullptr) override 
    {
        auto output = forwards({image}, stream);
        if (output.empty()) return {};
        return output[0];
    }

    virtual vector<BoxArray> forwards(const vector<trt::Image> &images, void *stream = nullptr) override 
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

        vector<pre::AffineMatrix> affine_matrixs(num_image);
        cudaStream_t stream_ = (cudaStream_t)stream;
        for (int i = 0; i < num_image; ++i)
            preprocess(i, images[i], preprocess_buffers_[i], affine_matrixs[i], stream);

        float *bbox_output_device = bbox_predict_.gpu();
        vector<void *> bindings{input_buffer_.gpu(), bbox_output_device};

        if (!trt_->forward(bindings, stream)) 
        {
            INFO("Failed to tensorRT forward.");
            return {};
        }

        for (int ib = 0; ib < num_image; ++ib) 
        {
            float *boxarray_device =
                output_boxarray_.gpu() + ib * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
            float *affine_matrix_device = (float *)preprocess_buffers_[ib]->gpu();
            float *image_based_bbox_output =
                bbox_output_device + ib * (bbox_head_dims_[1] * bbox_head_dims_[2]);
            checkRuntime(cudaMemsetAsync(boxarray_device, 0, sizeof(int), stream_));
            decode_kernel_invoker(image_based_bbox_output, bbox_head_dims_[1], num_classes_,
                                    bbox_head_dims_[2], confidence_threshold_, nms_threshold_,
                                    affine_matrix_device, boxarray_device, MAX_IMAGE_BOXES, stream_);
        }
        checkRuntime(cudaMemcpyAsync(output_boxarray_.cpu(), output_boxarray_.gpu(),
                                    output_boxarray_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_));
        checkRuntime(cudaStreamSynchronize(stream_));

        vector<BoxArray> arrout(num_image);
        int imemory = 0;
        for (int ib = 0; ib < num_image; ++ib) 
        {
            float *parray = output_boxarray_.cpu() + ib * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
            int count = min(MAX_IMAGE_BOXES, (int)*parray);
            BoxArray &output = arrout[ib];
            output.reserve(count);
            for (int i = 0; i < count; ++i) 
            {
                float *pbox = parray + 1 + i * NUM_BOX_ELEMENT;
                int label = pbox[5];
                int keepflag = pbox[6];
                if (keepflag == 1) {
                    Box result_object_box(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label);
                    result_object_box.pose.reserve(KEY_POINT_NUM);
                    for (int i = 0; i < KEY_POINT_NUM; i++)
                    {
                        result_object_box.pose.emplace_back(pbox[8+i*3], pbox[8+i*3+1], pbox[8+i*3+2]);
                    }
                    output.emplace_back(result_object_box);
                }
            }
        }
        return arrout;
    }
};

Infer *loadraw(const std::string &engine_file, float confidence_threshold,
               float nms_threshold) 
{
    InferImpl *impl = new InferImpl();
    if (!impl->load(engine_file, confidence_threshold, nms_threshold)) 
    {
        delete impl;
        impl = nullptr;
    }
    return impl;
}

shared_ptr<Infer> load(const string &engine_file, float confidence_threshold,
                       float nms_threshold) 
{
    return std::shared_ptr<InferImpl>(
        (InferImpl *)loadraw(engine_file, confidence_threshold, nms_threshold));
}

}