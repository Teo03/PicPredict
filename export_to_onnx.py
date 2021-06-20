import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import InferenceNet


model = InferenceNet()
model.eval()

torch.onnx.export(
    model,
    torch.zeros(280 * 280 * 4),
    './docs/onnx_model.onnx',
    verbose=True
)