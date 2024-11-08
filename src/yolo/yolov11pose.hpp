#ifndef YOLOV11POSE_HPP__
#define YOLOV11POSE_HPP__

#include <vector>
#include <string>
#include "common/image.hpp"


namespace yolov11pose
{

struct PosePoint
{
    float x, y,confidence;
    PosePoint() = default;
    PosePoint(float x, float y, float confidence) :
        x(x), y(y), confidence(confidence) {}
};

struct Box 
{
    float left, top, right, bottom, confidence;
    int class_label;
    std::vector<PosePoint> pose;

    Box() = default;
    Box(float left, float top, float right, float bottom, float confidence, int class_label)
        : left(left),
            top(top),
            right(right),
            bottom(bottom),
            confidence(confidence),
            class_label(class_label) {}
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