#ifndef RESNET_HPP__
#define RESNET_HPP__
#include <vector>
#include <string>
#include "common/image.hpp"
#include "common/data.hpp"

namespace resnet
{

using Attribute = data::Attribute

class Infer {
public:
    virtual Attribute forward(const trt::Image &image, void *stream = nullptr) = 0;
    virtual std::vector<Attribute> forwards(const std::vector<trt::Image> &images,
                                            void *stream = nullptr) = 0;
};

std::shared_ptr<Infer> load(const std::string &engine_file, int gpu_id=0);

}

#endif