#include "common/image.hpp"

trt::Image cvimg(const cv::Mat &image) { return trt::Image(image.data, image.cols, image.rows); }