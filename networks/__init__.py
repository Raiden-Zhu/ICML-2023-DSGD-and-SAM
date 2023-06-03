
import torch
import torch.nn as nn
from torchvision import models

from .alexnet_micro import alexnet as alexnet_m
from .resnet_micro import resnet18 as resnet18_m
from .resnet_micro import resnet34 as resnet34_m
from .densenet_micro import densenet121 as densenet121_m

def load_model(name, outputsize, pretrained=None):

    if pretrained:
        pretrained = True
    else:
        pretrained = False

    if name.lower() in 'alexnet_micro':
        model = alexnet_m(pretrained=pretrained)
        model.classifier = nn.Linear(256, outputsize)
    elif name.lower() in 'resnet18_micro':
        model = resnet18_m(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, outputsize)
    elif name.lower() in 'resnet34_micro':
        model = resnet34_m(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, outputsize)
    elif name.lower() in 'densenet121_micro':
        model = densenet121_m(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, outputsize)

    return model



