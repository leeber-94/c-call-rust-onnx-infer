import torch
import torch.onnx
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.dense = nn.Linear(8, 1)

    def forward(self, x):
        float_tensor = x.to(dtype=torch.float32)
        float_tensor = self.dense(float_tensor)
        int_tensor = float_tensor.to(dtype=torch.int32)
        return int_tensor

inputData = [[79, 30, 73, 65, 69, 51, 57, 67]]

# 创建torch模型
model = SimpleModel()
dummy_input = torch.tensor(inputData, dtype=torch.int32)
output = model(dummy_input)
print("torch Input:", dummy_input)
print("torch Output:", output)

# 导出onnx模型
onnx_filename = "simple_model.onnx"
torch.onnx.export(model, dummy_input, onnx_filename, verbose=False, input_names=['input'], output_names=['output'])

# 测试onnx模型
import onnxruntime
import numpy as np
ort_session = onnxruntime.InferenceSession(onnx_filename)
output = ort_session.run(['output'], {'input': np.array(inputData,dtype=np.int32)})
print("onnx Output:", output)