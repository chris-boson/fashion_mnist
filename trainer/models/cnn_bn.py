# Inspired from here: https://gist.github.com/Noumanmufc1/60f00e434f0ce42b6f4826029737490a
# Added seperable convolution module as drop in replacement for conv2d which saves a lot of parameters and ops
# Replaced ReLU with ReLU6 to make the model more resilient under quantization

import torch.nn as nn

class SeparableConv2d(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, stride=stride, padding=padding, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class CNNwithBN(nn.Module):
    def __init__(self, num_classes, conv_type='regular'):
        super().__init__()
        if conv_type == 'regular':
            conv = nn.Conv2d
        elif conv_type == 'separable':
            conv = SeparableConv2d

        self.layer1 = nn.Sequential(
            conv(1, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU6(),
            nn.BatchNorm2d(32))
        self.layer2 = nn.Sequential(
            conv(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU6(),
            nn.BatchNorm2d(64))
        self.layer3 = nn.Sequential(
            conv(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU6(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            conv(128, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU6(),
            nn.BatchNorm2d(256))
        self.layer5 = nn.Sequential(
            conv(256, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU6(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=8, stride=1))
        self.fc = nn.Linear(512, num_classes)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)

        return out
