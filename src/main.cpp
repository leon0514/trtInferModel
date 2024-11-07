#include "resnet/resnet.hpp"
#include "opencv2/opencv.hpp"
#include "common/timer.hpp"

void resnet()
{
    cv::Mat image = cv::imread("inference/car.jpg");
    auto resnet = resnet::load("resnet.engine");
    if (resnet == nullptr) return;
    trt::Timer timer;
    for (int i = 0; i < 100; i++)
    {
        timer.start();
        auto attr = resnet->forward(cvimg(image));
        printf("score : %lf, label : %d\n", attr.confidence, attr.class_label);
        timer.stop("BATCH1");
    }
}

int main()
{
    resnet();
}