# trtInferModel
基于杜佬的infer框架修改
添加使用tensorrt 推理 resnet 模型、yolov11-pose等模型
持续更新中

## Resnet
```
[infer.cu:251]: Infer 0x55fe06de6b70 [StaticShape]
[infer.cu:264]: Inputs: 1
[infer.cu:269]: 	0.input.1 : shape {1x3x224x224}
[infer.cu:272]: Outputs: 1
[infer.cu:277]: 	0.343 : shape {1x3}
score : 0.997001, label : 2
```

## yolov11 pose
![](./workspace/result/Yolov11-pose-result.jpg)

## yolov8 yolov11
![](./workspace/result/Yolov8-result.jpg)

## yolo系列输出shape转换
coco 数据集 yolov8、yolov11的模型输出为 `1x84x8400`     
在处理这个输出的时候需要将其转换为`1x8400x84`，可以利用[infer](https://github.com/shouxieai/infer)仓库中的v8trans.py脚本进行转换

## 图像质量评估
该模型来自于腾讯的 `CenseoQoE`
- 前处理    
通过调整图像的大小并从中心进行裁剪，确保输出图像符合目标尺寸的要求。如果原始图像的宽高比与目标尺寸不匹配，还会进行必要的旋转以保持图像内容的正确方向
    ``` json
    {
        "input_process": {
            "read_mode": "resize_random_crop",
            "sub_img_dim": [ 1280, 720 ],
            "resize_dim": [ 1920, 1080 ]
        }
    }
    ```

- 输出
    ``` shell
    [infer.cu:251]: Infer 0x301c600 [StaticShape]
    [infer.cu:264]: Inputs: 1
    [infer.cu:269]: 	0.test_input : shape {1x3x1280x720}
    [infer.cu:272]: Outputs: 1
    [infer.cu:277]: 	0.test_output : shape {1x1}
    Attribute(class_label: 0, confidence: 0.609141)
    ```


# pybind11 封装
- 编译
    ```shell
    make all
    ```
- 使用
    ```python
    import trtinfer
    model  = trtinfer.TrtYolov11poseInfer("yolov11s-pose.transd.engine", 0, 0.5, 0.45)
    result = model.forward_path("inference/gril.jpg")
    print(result)
    ```


# Reference
- [🌻infer](https://github.com/shouxieai/infer)
- [🌻CenseoQoE](https://github.com/Tencent/CenseoQoE)