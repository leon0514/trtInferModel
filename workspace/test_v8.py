import trtinfer
model  = trtinfer.TrtYolov8Infer("yolov8.engine", 0, 0.5, 0.45)
result = model.forward_path("inference/gril.jpg")
print(result)
