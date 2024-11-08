#include "resnet/resnet.hpp"
#include "resnet/yolov11pose.hpp"
#include "opencv2/opencv.hpp"
#include "common/timer.hpp"
#include "common/image.hpp"

std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v) 
{
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f * s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i) 
    {
        case 0:
            r = v, g = t, b = p;
            break;
        case 1:
            r = q, g = v, b = p;
            break;
        case 2:
            r = p, g = v, b = t;
            break;
        case 3:
            r = p, g = q, b = v;
            break;
        case 4:
            r = t, g = p, b = v;
            break;
        case 5:
            r = v, g = p, b = q;
            break;
        default:
            r = 1, g = 1, b = 1;
            break;
    }
    return std::make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255),
                        static_cast<uint8_t>(r * 255));
}

std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id) 
{
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

void resnetInfer()
{
    cv::Mat image = cv::imread("inference/car.jpg");
    auto resnet = resnet::load("resnet.engine");
    if (resnet == nullptr) return;
    auto attr = resnet->forward(trt::cvimg(image));
    printf("score : %lf, label : %d\n", attr.confidence, attr.class_label);
}

void yolov11poseInfer()
{
    const std::vector<std::pair<int, int>> coco_pairs = {
        {0, 1}, {0, 2}, {0, 11}, {0, 12}, {1, 3}, {2, 4},
        {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},
        {11, 12}, {5, 11}, {6, 12},
        {11, 13}, {13, 15}, {12, 14}, {14, 16}
    };

    const std::string cocolabels[] = { "person" };

    cv::Mat image = cv::imread("inference/gril.jpg");
    auto yolov11pose = yolov11pose::load("resnet.engine");
    if (yolov11pose == nullptr) return;
    auto objs = yolov11pose->forward(trt::cvimg(image));
    for (auto &obj : objs) 
    {
        uint8_t b, g, r;
        std::tie(b, g, r) = random_color(obj.class_label);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                    cv::Scalar(b, g, r), 5);

        auto name = cocolabels[obj.class_label];
        auto caption = cv::format("%s %.2f", name, obj.confidence);
        int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
                    cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
        cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);

        for (const auto& point : obj.pose)
        {
            int x = (int)point.x;
            int y = (int)point.y;
            cv::circle(image, cv::Point(x, y), 6, cv::Scalar(b, g, r), -1);
        }
        for (const auto& pair : coco_pairs) 
        {
            int startIdx = pair.first;
            int endIdx = pair.second;

            if (startIdx < obj.pose.size() && endIdx < obj.pose.size()) 
            {
                int x1 = (int)obj.pose[startIdx].x;
                int y1 = (int)obj.pose[startIdx].y;
                int x2 = (int)obj.pose[endIdx].x;
                int y2 = (int)obj.pose[endIdx].y;

                cv::line(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 2);
            }
        }
        printf("Save result to Result.jpg, %d objects\n", (int)objs.size());
        cv::imwrite("Yolov11-pose-result.jpg", image);
    }

}

int main()
{
    // resnetInfer();
    yolov11poseInfer();
}
