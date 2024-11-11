#include <sstream>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "opencv2/opencv.hpp"
#include "infer/infer.hpp"
#include "resnet/resnet.hpp"
#include "CenseoQoE/qoe.hpp"
#include "PPLCNet/pplcnet.hpp"
#include "yolo/yolov11pose.hpp"
#include "yolo/yolov8.hpp"
#include "common/image.hpp" 
#include "common/data.hpp" 

#define UNUSED(expr) do { (void)(expr); } while (0)

using namespace std;

namespace py=pybind11;

namespace pybind11 { namespace detail{
template<>
struct type_caster<cv::Mat>{
public:
    PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

    //! 1. cast numpy.ndarray to cv::Mat
    bool load(handle obj, bool){
        array b = reinterpret_borrow<array>(obj);
        buffer_info info = b.request();

        //const int ndims = (int)info.ndim;
        int nh = 1;
        int nw = 1;
        int nc = 1;
        int ndims = info.ndim;
        if(ndims == 2){
            nh = info.shape[0];
            nw = info.shape[1];
        } else if(ndims == 3){
            nh = info.shape[0];
            nw = info.shape[1];
            nc = info.shape[2];
        }else{
            char msg[64];
            std::sprintf(msg, "Unsupported dim %d, only support 2d, or 3-d", ndims);
            throw std::logic_error(msg);
            return false;
        }

        int dtype;
        if(info.format == format_descriptor<unsigned char>::format()){
            dtype = CV_8UC(nc);
        }else if (info.format == format_descriptor<int>::format()){
            dtype = CV_32SC(nc);
        }else if (info.format == format_descriptor<float>::format()){
            dtype = CV_32FC(nc);
        }else{
            throw std::logic_error("Unsupported type, only support uchar, int32, float");
            return false;
        }

        value = cv::Mat(nh, nw, dtype, info.ptr);
        return true;
    }

    //! 2. cast cv::Mat to numpy.ndarray
    static handle cast(const cv::Mat& mat, return_value_policy, handle defval){
        UNUSED(defval);

        std::string format = format_descriptor<unsigned char>::format();
        size_t elemsize = sizeof(unsigned char);
        int nw = mat.cols;
        int nh = mat.rows;
        int nc = mat.channels();
        int depth = mat.depth();
        int type = mat.type();
        int dim = (depth == type)? 2 : 3;

        if(depth == CV_8U){
            format = format_descriptor<unsigned char>::format();
            elemsize = sizeof(unsigned char);
        }else if(depth == CV_32S){
            format = format_descriptor<int>::format();
            elemsize = sizeof(int);
        }else if(depth == CV_32F){
            format = format_descriptor<float>::format();
            elemsize = sizeof(float);
        }else{
            throw std::logic_error("Unsupport type, only support uchar, int32, float");
        }

        std::vector<size_t> bufferdim;
        std::vector<size_t> strides;
        if (dim == 2) {
            bufferdim = {(size_t) nh, (size_t) nw};
            strides = {elemsize * (size_t) nw, elemsize};
        } else if (dim == 3) {
            bufferdim = {(size_t) nh, (size_t) nw, (size_t) nc};
            strides = {(size_t) elemsize * nw * nc, (size_t) elemsize * nc, (size_t) elemsize};
        }
        return array(buffer_info( mat.data,  elemsize,  format, dim, bufferdim, strides )).release();
    }
};
}}//! end namespace pybind11::detail

class TrtQoEInfer{
public:
    TrtQoEInfer(std::string model_path, int gpu_id = 0)
    {
        instance_ = qoe::load(model_path, gpu_id);
    }


    data::Attribute forward(const cv::Mat& image)
    {
        return instance_->forward(trt::cvimg(image));
    }

    data::Attribute forward_path(const std::string& image_path)
    {
        cv::Mat image = cv::imread(image_path);
        return instance_->forward(trt::cvimg(image));
    }


    bool valid(){
		return instance_ != nullptr;
	}

private:
    std::shared_ptr<qoe::Infer> instance_;

};

class TrtResnetInfer{
public:
    TrtResnetInfer(std::string model_path, int gpu_id = 0)
    {
        instance_ = resnet::load(model_path, gpu_id);
    }


    data::Attribute forward(const cv::Mat& image)
    {
        return instance_->forward(trt::cvimg(image));
    }

    data::Attribute forward_path(const std::string& image_path)
    {
        cv::Mat image = cv::imread(image_path);
        return instance_->forward(trt::cvimg(image));
    }


    bool valid(){
		return instance_ != nullptr;
	}

private:
    std::shared_ptr<resnet::Infer> instance_;

};

class TrtPPLCNetInfer{
public:
    TrtPPLCNetInfer(std::string model_path, int gpu_id = 0)
    {
        instance_ = resnet::load(model_path, gpu_id);
    }


    data::Attribute forward(const cv::Mat& image)
    {
        return instance_->forward(trt::cvimg(image));
    }

    data::Attribute forward_path(const std::string& image_path)
    {
        cv::Mat image = cv::imread(image_path);
        return instance_->forward(trt::cvimg(image));
    }


    bool valid(){
		return instance_ != nullptr;
	}

private:
    std::shared_ptr<pplcnet::Infer> instance_;

};


class TrtYolov11poseInfer{
public:
    TrtYolov11poseInfer(std::string model_path, int gpu_id = 0, float confidence_threshold=0.5f, float nms_threshold=0.45f)
    {
        instance_ = yolov11pose::load(model_path, gpu_id, confidence_threshold, nms_threshold);
    }

    data::BoxArray forward(const cv::Mat& image)
    {
        return instance_->forward(trt::cvimg(image));
    }

    data::BoxArray forward_path(const std::string& image_path)
    {
        cv::Mat image = cv::imread(image_path);
        return instance_->forward(trt::cvimg(image));
    }


    bool valid(){
		return instance_ != nullptr;
	}

private:
    std::shared_ptr<yolov11pose::Infer> instance_;

};

class TrtYolov8Infer{
public:
    TrtYolov8Infer(std::string model_path, int gpu_id = 0, float confidence_threshold=0.5f, float nms_threshold=0.45f)
    {
        instance_ = yolov8::load(model_path, gpu_id, confidence_threshold, nms_threshold);
    }

    data::BoxArray forward(const cv::Mat& image)
    {
        return instance_->forward(trt::cvimg(image));
    }

    data::BoxArray forward_path(const std::string& image_path)
    {
        cv::Mat image = cv::imread(image_path);
        return instance_->forward(trt::cvimg(image));
    }


    bool valid(){
		return instance_ != nullptr;
	}

private:
    std::shared_ptr<yolov8::Infer> instance_;

};


PYBIND11_MODULE(trtinfer, m){
    py::class_<data::Attribute>(m, "Attribute")
        .def_readwrite("confidence", &data::Attribute::confidence)
        .def_readwrite("class_label", &data::Attribute::class_label)
        .def("__repr__", [](const data::Attribute &attr) {
            std::ostringstream oss;
            oss << "Attribute(class_label: " << attr.class_label << ", confidence: " << attr.confidence << ")";
            return oss.str();
        });

    py::class_<data::PosePoint>(m, "PosePoint")
        .def_readwrite("x", &data::PosePoint::x)
        .def_readwrite("y", &data::PosePoint::x)
        .def_readwrite("confidence", &data::PosePoint::confidence)
        .def("__repr__", [](const data::PosePoint &point) {
            std::ostringstream oss;
            oss << "PosePoint(x: " << point.x << ", y: " << point.y << ", confidence: " << point.confidence <<")";
            return oss.str();
        });
    
    py::class_<data::Box>(m, "Box")
        .def_readwrite("left", &data::Box::left)
        .def_readwrite("top", &data::Box::top)
        .def_readwrite("right", &data::Box::right)
        .def_readwrite("bottom", &data::Box::bottom)
        .def_readwrite("confidence", &data::Box::confidence)
        .def_readwrite("class_label", &data::Box::class_label)
        .def_readwrite("pose", &data::Box::pose)
        .def("__repr__", [](const data::Box &box) {
            std::ostringstream oss;
            oss << "Box(class_label: " << box.class_label
                << ", left: " << box.left
                << ", top: " << box.top
                << ", right: " << box.right
                << ", bottom: " << box.bottom
                << ", confidence: " << box.confidence
                << ")";
            if (box.pose.size() != 0)
            {
                oss << ", PosePoint[";
                for (size_t i = 0; i < box.pose.size(); ++i) 
                {
                    oss << py::str(py::cast(box.pose[i]));
                    if (i < box.pose.size() - 1) {
                        oss << ", ";
                    }
                }
                oss << "]";
            }
            
            return oss.str();
        });

    py::class_<TrtYolov11poseInfer>(m, "TrtYolov11poseInfer")
		.def(py::init<string, int, float, float>(), py::arg("model_path"), py::arg("gpu_id"), py::arg("confidence_threshold"), py::arg("nms_threshold"))
		.def_property_readonly("valid", &TrtYolov11poseInfer::valid)
        .def("forward_path", &TrtYolov11poseInfer::forward_path, py::arg("image_path"))
		.def("forward", &TrtYolov11poseInfer::forward, py::arg("image"));

    py::class_<TrtYolov8Infer>(m, "TrtYolov8Infer")
		.def(py::init<string, int, float, float>(), py::arg("model_path"), py::arg("gpu_id"), py::arg("confidence_threshold"), py::arg("nms_threshold"))
		.def_property_readonly("valid", &TrtYolov8Infer::valid)
        .def("forward_path", &TrtYolov8Infer::forward_path, py::arg("image_path"))
		.def("forward", &TrtYolov8Infer::forward, py::arg("image"));

	py::class_<TrtResnetInfer>(m, "TrtResnetInfer")
		.def(py::init<string, int>(), py::arg("model_path"), py::arg("gpu_id"))
		.def_property_readonly("valid", &TrtResnetInfer::valid)
        .def("forward_path", &TrtResnetInfer::forward_path, py::arg("image_path"))
		.def("forward", &TrtResnetInfer::forward, py::arg("image"));

    py::class_<TrtPPLCNetInfer>(m, "TrtPPLCNetInfer")
		.def(py::init<string, int>(), py::arg("model_path"), py::arg("gpu_id"))
		.def_property_readonly("valid", &TrtPPLCNetInfer::valid)
        .def("forward_path", &TrtPPLCNetInfer::forward_path, py::arg("image_path"))
		.def("forward", &TrtPPLCNetInfer::forward, py::arg("image"));
    
    py::class_<TrtQoEInfer>(m, "TrtQoEInfer")
		.def(py::init<string, int>(), py::arg("model_path"), py::arg("gpu_id"))
		.def_property_readonly("valid", &TrtQoEInfer::valid)
        .def("forward_path", &TrtQoEInfer::forward_path, py::arg("image_path"))
		.def("forward", &TrtQoEInfer::forward, py::arg("image"));
};