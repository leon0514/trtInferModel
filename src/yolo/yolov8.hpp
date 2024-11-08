#ifndef YOLOV8_HPP__
#define YOLOV8_HPP__
#include <vector>
#include <string>
#include "common/image.hpp"
#include "common/data.hpp"

namespace yolov8
{

using Box       = data::Box;

using BoxArray  = data::BoxArray;

class Infer {
public:
    virtual BoxArray forward(const trt::Image &image, void *stream = nullptr) = 0;
    virtual std::vector<BoxArray> forwards(const std::vector<trt::Image> &images,
                                            void *stream = nullptr) = 0;
};

std::shared_ptr<Infer> load(const std::string &engine_file, int gpu_id = 0, float confidence_threshold=0.5f, float nms_threshold=0.45f);

}

#endif