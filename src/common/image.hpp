#ifndef IMAGE_HPP__
#define IMAGE_HPP__
#include "opencv2/opencv.hpp"

namespace trt
{

struct Image {
    const void *bgrptr = nullptr;
    int width = 0, height = 0;

    Image() = default;
    Image(const void *bgrptr, int width, int height) : bgrptr(bgrptr), width(width), height(height) {}
};

Image cvimg(const cv::Mat &image);

}

#endif
