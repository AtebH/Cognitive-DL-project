# model.py
import torch
from torch import nn
from torchvision.models import mobilenet_v3_large, resnet18, resnet50, resnet101, resnet152, alexnet, vgg16
from torchvision.models import MobileNet_V3_Large_Weights, ResNet18_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights, AlexNet_Weights, VGG16_Weights

class MobileNetV3Binary(nn.Module):
    def __init__(self):
        super(MobileNetV3Binary, self).__init__()
        self.mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.mobilenet.classifier[3] = nn.Linear(self.mobilenet.classifier[3].in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.mobilenet(x)
        x = self.sigmoid(x)
        return x

class ResNetBinary(nn.Module):
    def __init__(self, resnet_version='resnet152'):
        super(ResNetBinary, self).__init__()
        if resnet_version == 'resnet18':
            self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif resnet_version == 'resnet50':
            self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif resnet_version == 'resnet101':
            self.resnet = resnet101(weights=ResNet101_Weights.DEFAULT)
        elif resnet_version == 'resnet152':
            self.resnet = resnet152(weights=ResNet152_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported ResNet version: {resnet_version}")
        
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.sigmoid(x)
        return x

class AlexNetBinary(nn.Module):
    def __init__(self):
        super(AlexNetBinary, self).__init__()
        self.alexnet = alexnet(weights=AlexNet_Weights.DEFAULT)
        self.alexnet.classifier[6] = nn.Linear(self.alexnet.classifier[6].in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.alexnet(x)
        x = self.sigmoid(x)
        return x

class VGG16Binary(nn.Module):
    def __init__(self):
        super(VGG16Binary, self).__init__()
        self.vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        self.vgg.classifier[6] = nn.Linear(self.vgg.classifier[6].in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.vgg(x)
        x = self.sigmoid(x)
        return x

def get_model(config):
    model_name = config['model']['name']
    if model_name == 'mobilenet':
        return MobileNetV3Binary()
    elif model_name.startswith('resnet'):
        return ResNetBinary(resnet_version=model_name)
    elif model_name == 'alexnet':
        return AlexNetBinary()
    elif model_name == 'vgg':
        return VGG16Binary()
    else:
        raise ValueError(f"Model {model_name} is not supported")