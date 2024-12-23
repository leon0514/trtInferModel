#ifndef PREPROCESS_HPP__
#define PREPROCESS_HPP__
#include <memory>
#include <cuda_runtime.h>

namespace pre
{

enum class NormType : int { None = 0, MeanStd = 1, AlphaBeta = 2 };

enum class ChannelType : int { None = 0, SwapRB = 1 };

struct Norm 
{
    float mean[3];
    float std[3];
    float alpha, beta;
    NormType type = NormType::None;
    ChannelType channel_type = ChannelType::None;

    // out = (x * alpha - mean) / std
    static Norm mean_std(const float mean[3], const float std[3], float alpha = 1 / 255.0f,
                        ChannelType channel_type = ChannelType::None);

    // out = x * alpha + beta
    static Norm alpha_beta(float alpha, float beta = 0, ChannelType channel_type = ChannelType::None);

    // None
    static Norm None();
};

struct ResizeMatrix 
{
    float i2d[6];  // image to dst(network), 2x3 matrix
    float d2i[6];  // dst to image, 2x3 matrix

    void compute(const std::tuple<int, int> &from, const std::tuple<int, int> &to) 
    {
        float scale_x = std::get<0>(to) / (float)std::get<0>(from);
        float scale_y = std::get<1>(to) / (float)std::get<1>(from);
        float scale = std::min(scale_x, scale_y);

        // letter box
        // i2d[0] = scale;
        // i2d[1] = 0;
        // i2d[2] = -scale * get<0>(from) * 0.5 + get<0>(to) * 0.5 + scale * 0.5 - 0.5;
        // i2d[3] = 0;
        // i2d[4] = scale;
        // i2d[5] = -scale * get<1>(from) * 0.5 + get<1>(to) * 0.5 + scale * 0.5 - 0.5;

        // resize 
        i2d[0] = scale;
        i2d[1] = 0;
        i2d[2] = 0;
        i2d[3] = 0;
        i2d[4] = scale;
        i2d[5] = 0;


        double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
        D = D != 0. ? double(1.) / D : double(0.);
        double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D, A21 = -i2d[3] * D;
        double b1 = -A11 * i2d[2] - A12 * i2d[5];
        double b2 = -A21 * i2d[2] - A22 * i2d[5];

        d2i[0] = A11;
        d2i[1] = A12;
        d2i[2] = b1;
        d2i[3] = A21;
        d2i[4] = A22;
        d2i[5] = b2;
    }
};

struct ResizeRandomCropMatrix 
{
    float i2d[6];  // image to dst(network), 2x3 matrix
    float d2i[6];  // dst to image, 2x3 matrix

    void compute(const std::tuple<int, int> &from, const std::tuple<int, int> &to, int resize_w, int resize_h) 
    {        
        int h = std::get<0>(from);
        int w = std::get<1>(from);

        int sub_img_w = std::get<0>(to);
        int sub_img_h = std::get<1>(to);

        // 计算缩放比例
        bool flag = false;
        if ((w - h) * (resize_w - resize_h) < 0) 
        {
            std::swap(resize_w, resize_h);
            std::swap(sub_img_w, sub_img_h);
            flag = true;
        }
        float scale = max(static_cast<float>(resize_h) / h, static_cast<float>(resize_w) / w);

        // 计算放射矩阵
        float cx = w / 2.0f;
        float cy = h / 2.0f;
        float tx = (resize_w - sub_img_w) / 2.0f;
        float ty = (resize_h - sub_img_h) / 2.0f;

        i2d[0] = scale;
        i2d[1] = 0;
        i2d[2] = -scale * cx + tx;
        i2d[3] = 0;
        i2d[4] = scale;
        i2d[5] = -scale * cy + ty;

        // 如果需要旋转90度
        if (flag) 
        {
            std::swap(i2d[0], i2d[3]);
            std::swap(i2d[1], i2d[4]);
        }

        double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
        D = D != 0. ? double(1.) / D : double(0.);
        double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D, A21 = -i2d[3] * D;
        double b1 = -A11 * i2d[2] - A12 * i2d[5];
        double b2 = -A21 * i2d[2] - A22 * i2d[5];

        d2i[0] = A11;
        d2i[1] = A12;
        d2i[2] = b1;
        d2i[3] = A21;
        d2i[4] = A22;
        d2i[5] = b2;
    }
};

struct ResizeShortCenterCropMatrix 
{
    float i2d[6];  // image to dst(network), 2x3 matrix
    float d2i[6];  // dst to image, 2x3 matrix

    void compute(const std::tuple<int, int> &from, const std::tuple<int, int> &to, int resize_short) 
    {   
        int h = std::get<0>(from);
        int w = std::get<1>(from);

        int sub_img_w = std::get<0>(to);
        int sub_img_h = std::get<1>(to);

        // 计算缩放比例
        float scale = static_cast<float>(resize_short) / min(w, h);

        int resize_w = static_cast<int>(scale * w);
        int resize_h = static_cast<int>(scale * h);

        // 计算放射矩阵
        float cx = w / 2.0f;
        float cy = h / 2.0f;
        float tx = (resize_w - sub_img_w) / 2.0f;
        float ty = (resize_h - sub_img_h) / 2.0f;

        i2d[0] = scale;
        i2d[1] = 0;
        i2d[2] = -scale * cx + tx;
        i2d[3] = 0;
        i2d[4] = scale;
        i2d[5] = -scale * cy + ty;

        double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
        D = D != 0. ? double(1.) / D : double(0.);
        double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D, A21 = -i2d[3] * D;
        double b1 = -A11 * i2d[2] - A12 * i2d[5];
        double b2 = -A21 * i2d[2] - A22 * i2d[5];

        d2i[0] = A11;
        d2i[1] = A12;
        d2i[2] = b1;
        d2i[3] = A21;
        d2i[4] = A22;
        d2i[5] = b2;
    }
};


// 用于目标检测模型检测出区域后，截取出该区域并做resize到目标大小
struct ResizeCropMatrix 
{
    float i2d[6];  // image to dst(network), 2x3 matrix
    float d2i[6];  // dst to image, 2x3 matrix

    // 1 0 -x     sx 0  -x*sx
    // 0 1 -y  -> 0  sy -y*sy
    // 0 0 1      0  0  1
    void compute(const std::tuple<int, int> &to, 
                const std::tuple<int, int> &start, const std::tuple<int, int> &end) 
    {
        int start_x = std::get<0>(start);
        int start_y = std::get<1>(start);
        
        int end_x = std::get<0>(end);
        int end_y = std::get<1>(end);


        int dst_w = std::get<0>(to);
        int dst_h = std::get<1>(to);

        float scale_x = 1.0f * (end_x - start_x) / dst_w;
        float scale_y = 1.0f * (end_y - start_y) / dst_h;

        i2d[0] = scale_x;
        i2d[1] = 0;
        i2d[2] = -start_x * scale_x;
        i2d[3] = 0;
        i2d[4] = scale_y;
        i2d[5] = -start_y * scale_y;

        double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
        D = D != 0. ? double(1.) / D : double(0.);
        double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D, A21 = -i2d[3] * D;
        double b1 = -A11 * i2d[2] - A12 * i2d[5];
        double b2 = -A21 * i2d[2] - A22 * i2d[5];

        d2i[0] = A11;
        d2i[1] = A12;
        d2i[2] = b1;
        d2i[3] = A21;
        d2i[4] = A22;
        d2i[5] = b2;
    }
};


struct AffineMatrix 
{
    float i2d[6];  // image to dst(network), 2x3 matrix
    float d2i[6];  // dst to image, 2x3 matrix

    void compute(const std::tuple<int, int> &from, const std::tuple<int, int> &to) 
    {
        float scale_x = std::get<0>(to) / (float)std::get<0>(from);
        float scale_y = std::get<1>(to) / (float)std::get<1>(from);
        float scale = std::min(scale_x, scale_y);

        // letter box
        i2d[0] = scale;
        i2d[1] = 0;
        i2d[2] = -scale * std::get<0>(from) * 0.5 + std::get<0>(to) * 0.5 + scale * 0.5 - 0.5;
        i2d[3] = 0;
        i2d[4] = scale;
        i2d[5] = -scale * std::get<1>(from) * 0.5 + std::get<1>(to) * 0.5 + scale * 0.5 - 0.5;

        // resize 
        // i2d[0] = scale;
        // i2d[1] = 0;
        // i2d[2] = 0;
        // i2d[3] = 0;
        // i2d[4] = scale;
        // i2d[5] = 0;


        double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
        D = D != 0. ? double(1.) / D : double(0.);
        double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D, A21 = -i2d[3] * D;
        double b1 = -A11 * i2d[2] - A12 * i2d[5];
        double b2 = -A21 * i2d[2] - A22 * i2d[5];

        d2i[0] = A11;
        d2i[1] = A12;
        d2i[2] = b1;
        d2i[3] = A21;
        d2i[4] = A22;
        d2i[5] = b2;
    }
};


void warp_affine_bilinear_and_normalize_plane(uint8_t *src, int src_line_size, int src_width,
                                                int src_height, float *dst, int dst_width,
                                                int dst_height, float *matrix_2_3,
                                                uint8_t const_value, const Norm &norm,
                                                cudaStream_t stream);

}


#endif
