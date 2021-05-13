import os
from config import MODELS_DIR
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import resnet50

class InferenceResNet50(nn.Module):
    def __init__(self):
        super(InferenceResNet50, self).__init__()

        self.model = resnet50()
        self.model.load_state_dict(torch.load('./models/model.pth'))

    # modified forward pass
    def forward(self, img):
        # we get (280, 280, 4) from JS canvas
        img = img.reshape(280, 280, 4)
        img = torch.narrow(img, dim=2, start=3, length=1)
        img = img.reshape(1, 1, 280, 280)
        img = F.avg_pool2d(img, 10, stride=10)
        img = img / 255

        img = self.model(img)
        
        # use softmax to get probs in range [0,1]
        output = F.softmax(img, dim=1)
        return output


model = InferenceResNet50()
model.eval()

torch.onnx.export(
    model,
    torch.zeros(280 * 280 * 4),
    os.path.join(MODELS_DIR, 'onnx_model.onnx'),
    verbose=True
)