from config import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # conv layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4)

        # fully-connected dense layers
        self.fc1 = nn.Linear(in_features=32, out_features=320)
        self.fc2 = nn.Linear(in_features=320, out_features=640)
        self.out = nn.Linear(in_features=640, out_features=10)

    def forward(self, x):
        # input layer
        x = x

        # conv layers
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv3(x)
        x = F.relu(x)

        # flatten the conv output 
        x = x.flatten(start_dim=1)

        # fully-connected dense layers
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        # output layer
        x = self.out(x)

        return x

class InferenceNet(nn.Module):
    def __init__(self):
        super(InferenceNet, self).__init__()

        self.model = Net()
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


#def resnet50():
#    model = models.resnet50(pretrained=False)
#    
#    conv1_out_channels = model.conv1.out_channels
#    model.conv1 = nn.Conv2d(1, conv1_out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#    model.maxpool = nn.MaxPool2d(kernel_size=2)
#    fc_features = model.fc.in_features
#    model.fc = nn.Linear(fc_features, len(CLASSES))
#    
#    return model