import trtinfer
model  = trtinfer.TrtQoEInfer("model.engine", 0)
result = model.forward_path("inference/b.jpg")
print(result)
