#ifndef DATA_HPP__
#define DATA_HPP__
#include <vector>
#include <string>

namespace data
{

struct Attribute 
{
    float confidence;
    int class_label;

    Attribute() = default;
    Attribute(float confidence, int class_label) : confidence(confidence), class_label(class_label) {}
};


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

using BoxArray = std::vector<Box>;

}





#endif 