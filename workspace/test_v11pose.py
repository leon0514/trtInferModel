import trtinfer
model  = trtinfer.TrtYolov11poseInfer("yolov11s-pose.transd.engine", 0, 0.5, 0.45)
result = model.forward_path("inference/gril.jpg")
print(result)
