''' This code is taken and modified from https://github.com/mundher/local-global. '''

import torch.nn as nn
import torch
import torch as T

def matrix_mult_to_conv(matrix1, matrix2):
    '''
    matrix1: dimensions (bzs, channel, width, height)
    matrix2: dimensions (bzs, channel, width, height)
    '''
    assert len(matrix1.shape)==4
    assert len(matrix1.shape) == len(matrix2.shape)
    assert matrix1.shape[3] == matrix2.shape[2]
    assert matrix1.shape[0] == matrix2.shape[0]
    assert matrix1.shape[0] == 1
    assert matrix1.shape[1] == matrix2.shape[1]

    b,c,w,h,= matrix1.shape
    b1,c1,w1,h1 = matrix2.shape
    A_o = matrix1.permute(0,1,3,2).reshape(b, h*c, w)[None].permute(1,2,3,0) # bsz, h*c, w, 1
    B_o = matrix2.permute(1,3,2,0).reshape(c1*h1, w1, b1)[None].permute(1,2,3,0) # c*h, w, bsz(1), 1

    conv_out = torch.nn.functional.conv2d(A_o, B_o,groups=c).reshape(b,c1,h1,w).permute(0,1,3,2)

    return conv_out

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicResnet(nn.Module):
    def __init__(self, norm_module=False, mean=None, std=None):
        super().__init__()
        if norm_module:
            self.resnet = nn.Sequential(
                Norm_Module(mean, std),
                BasicBlock(1, 32),
                nn.Dropout(0.1),
                BasicBlock(32, 32),
                nn.Dropout(0.2),
                nn.AdaptiveAvgPool2d((1,1))
            )
        else:
            self.resnet = nn.Sequential(
                BasicBlock(1, 32),
                nn.Dropout(0.1),
                BasicBlock(32, 32),
                nn.Dropout(0.2),
                nn.AdaptiveAvgPool2d((1,1))
            )
        self.fc1 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, x.size()[1])
        x = self.fc1(x)
        #x = T.sigmoid(x)
        return x

class Norm_Module(nn.Module):
    def __init__(self, mean=None, std=None):
        super().__init__()
        self.mean = mean
        self.std = std
    def forward(self, x):
        eps=1e-3
        if x.std()> eps:
            xo = (x - x.mean([1,2,3])[:,None,None,None]) / (x.std([1,2,3])[:,None,None,None])
        else:
            xo = (x - x.mean([1,2,3])[:,None,None,None])
        return xo
        
