#ifndef RESNET_HPP__
#define RESNET_HPP__
#include <vector>
#include <string>
#include "common/image.hpp"

namespace resnet
{

struct Attribute 
{
    float confidence;
    int class_label;

    Attribute() = default;
    Attribute(float confidence, int class_label) : confidence(confidence), class_label(class_label) {}
};

class Infer {
public:
    virtual Attribute forward(const trt::Image &image, void *stream = nullptr) = 0;
    virtual std::vector<Attribute> forwards(const std::vector<trt::Image> &images,
                                            void *stream = nullptr) = 0;
};

std::shared_ptr<Infer> load(const std::string &engine_file);

}

#endif