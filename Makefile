cc        := g++
name      := trtinfer.so
workdir   := workspace
srcdir    := src
objdir    := objs
stdcpp    := c++11
cuda_home := /usr/local/cuda-12
cuda_arch := 8.6
nvcc      := $(cuda_home)/bin/nvcc -ccbin=$(cc)


project_include_path := src
opencv_include_path  := /usr/local/include/opencv4
trt_include_path     := /opt/nvidia/tensorrt/TensorRT-8.6.1.6/include/ 
cuda_include_path    := $(cuda_home)/include
ffmpeg_include_path  := 

python_include_path  := /usr/include/python3.8/


include_paths        := $(project_include_path) \
						$(opencv_include_path) \
						$(trt_include_path) \
						$(cuda_include_path) \
						$(python_include_path)


opencv_library_path  := /usr/local/lib/
trt_library_path     := /opt/nvidia/tensorrt/TensorRT-8.6.1.6/lib/
cuda_library_path    := $(cuda_home)/lib64/
python_library_path  := 

library_paths        := $(opencv_library_path) \
						$(trt_library_path) \
						$(cuda_library_path) \
						$(cuda_library_path) \
						$(python_library_path)

link_ffmpeg       := avcodec avformat swresample swscale avutil
link_opencv       := opencv_core opencv_imgproc opencv_videoio opencv_imgcodecs
link_trt          := nvinfer nvinfer_plugin nvonnxparser
link_cuda         := cuda cublas cudart cudnn
link_sys          := stdc++ dl

link_librarys     := $(link_ffmpeg) $(link_opencv) $(link_trt) $(link_cuda) $(link_sys)


empty := 
library_path_export := $(subst $(empty) $(empty),:,$(library_paths))

# 把库路径和头文件路径拼接起来成一个，批量自动加-I、-L、-l
run_paths     := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))

cpp_compile_flags := -std=$(stdcpp) -w -g -O0 -m64 -fPIC -fopenmp -pthread $(include_paths)
cu_compile_flags  := -Xcompiler "$(cpp_compile_flags)"
link_flags        := -pthread -fopenmp -Wl,-rpath='$$ORIGIN' $(library_paths) $(link_librarys)

cpp_srcs := $(shell find $(srcdir) -name "*.cpp")
cpp_objs := $(cpp_srcs:.cpp=.cpp.o)
cpp_objs := $(cpp_objs:$(srcdir)/%=$(objdir)/%)
cpp_mk   := $(cpp_objs:.cpp.o=.cpp.mk)

# 定义cu文件的路径查找和依赖项mk文件
cu_srcs := $(shell find $(srcdir) -name "*.cu")
cu_objs := $(cu_srcs:.cu=.cu.o)
cu_objs := $(cu_objs:$(srcdir)/%=$(objdir)/%)
cu_mk   := $(cu_objs:.cu.o=.cu.mk)

pro_cpp_objs := $(filter-out interface.o, $(cpp_objs))

# 所有的头文件依赖产生的makefile文件，进行include
ifneq ($(MAKECMDGOALS), clean)
include $(mks)
endif


$(name)   : $(workdir)/$(name)

all       : $(name)
run       : $(name)
	@cd $(workdir) && python test.py

runhdd    : $(name)
	@cd $(workdir) && python test_hard_decode_yolov5.py

pro       : $(workdir)/pro
runpro    : pro
	@cd $(workdir) && ./pro

$(workdir)/$(name) : $(cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@$(cc) -shared $^ -o $@ $(link_flags)

$(workdir)/pro : $(pro_cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@$(cc) $^ -o $@ $(link_flags)

$(objdir)/%.cpp.o : $(srcdir)/%.cpp
	@echo Compile CXX $<
	@mkdir -p $(dir $@)
	@$(cc) -c $< -o $@ $(cpp_compile_flags)

$(objdir)/%.cu.o : $(srcdir)/%.cu
	@echo Compile CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -c $< -o $@ $(cu_compile_flags)

# 编译cpp依赖项，生成mk文件
$(objdir)/%.cpp.mk : $(srcdir)/%.cpp
	@echo Compile depends C++ $<
	@mkdir -p $(dir $@)
	@$(cc) -M $< -MF $@ -MT $(@:.cpp.mk=.cpp.o) $(cpp_compile_flags)
    
# 编译cu文件的依赖项，生成cumk文件
$(objdir)/%.cu.mk : $(srcdir)/%.cu
	@echo Compile depends CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -M $< -MF $@ -MT $(@:.cu.mk=.cu.o) $(cu_compile_flags)

# 定义清理指令
clean :
	@rm -rf $(objdir) $(workdir)/$(name) $(workdir)/pro $(workdir)/*.trtmodel $(workdir)/imgs

# 防止符号被当做文件
.PHONY : clean run $(name)

# 导出依赖库路径，使得能够运行起来
export LD_LIBRARY_PATH:=$(library_path_export)