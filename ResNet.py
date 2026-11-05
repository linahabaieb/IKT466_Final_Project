import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
from  torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(SCRIPT_PATH,"spectrograms")

dataset = datasets.ImageFolder(data_dir,transform=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()]))
labels = dataset.targets
indicies = list(range(len(dataset)))

train_indexes, temp_indexes, train_labels, temp_labels = train_test_split(indicies, labels,train_size=0.7,random_state=42,stratify=labels)
test_indexes, validate_indexes, test_labels, validate_labels = train_test_split(temp_indexes, temp_labels,train_size=0.67,random_state=42,stratify=labels)

train_data = Subset(dataset, train_indexes)
test_data = Subset(dataset, test_indexes)
validate_data = Subset(dataset, validate_indexes)

loaded_training_data = DataLoader(train_data,batch_size=32,shuffle=True)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample = None, stride = 1):
        super(Block,self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1)
        self.batch_norm_layer1 = nn.BatchNorm2d(out_channels)
        self.conv_layer2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.batch_norm_layer2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self,x):
        identity = x

        x = self.conv_layer1(x)
        x = self.batch_norm_layer1(x)
        x = self.conv_layer2(x)
        x = self.batch_norm_layer2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        activated = self.relu(x)
        return activated

class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes): 
        super(ResNet,self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels,64,kernel_size=7,stride=2,padding=3)
        self.batch_norm_layer1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1 = self._make_layer(block,layers[0],64,1)
        self.layer2 = self._make_layer(block,layers[1],128,2)
        self.layer3 = self._make_layer(block,layers[2],256,2)
        self.layer4 = self._make_layer(block,layers[3],512,2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, layer_size, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels,out_channels, kernel_size=1, stride = stride), nn.BatchNorm2d(out_channels))

        layers.append(block(self.in_channels,out_channels, identity_downsample, stride))
        self.in_channels = out_channels

        for i in range(layer_size-1):
            layers.append(block(self.in_channels,out_channels))

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = self.batch_norm_layer1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    resNet34 = ResNet(Block,[3,4,6,3],3,192)
    x = torch.randn(1,3,224,224)
    y = resNet34(x)
    print(y.shape)
    