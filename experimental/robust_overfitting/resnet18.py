'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet1(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_planes=64):
        super(ResNet1, self).__init__()
        self.in_planes = in_planes

        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_planes * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes * 4, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, in_planes * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(in_planes * 4 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # print("conv1", out.shape)
        out = self.layer1(out)
        # print("layer1", out.shape)
        out = self.layer2(out)
        # print("layer2", out.shape)
        out = self.layer3(out)
        # print("layer3", out.shape)
        # out = self.layer4(out)
        # print("layer4", out.shape)
        out = F.avg_pool2d(out, 4)
        # print("avg", out.shape)
        out = out.view(out.size(0), -1)
        # print("view", out.shape)
        out = self.linear(out)
        # print(out.shape)
        return out

class ResNet2(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_planes=64):
        super(ResNet2, self).__init__()
        self.in_planes = in_planes

        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_planes * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, in_planes * 4, num_blocks[3], stride=2)
        self.linear = nn.Linear(in_planes * 4 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # print("conv1", out.shape)
        out = self.layer1(out)
        # print("layer1", out.shape)
        out = self.layer2(out)
        # print("layer2", out.shape)
        out = self.layer3(out)
        # print("layer3", out.shape)
        out = self.layer4(out)
        # print("layer4", out.shape)
        out = F.avg_pool2d(out, 2)
        # print("avg", out.shape)
        out = out.view(out.size(0), -1)
        # print("view", out.shape)
        out = self.linear(out)
        # print(out.shape)
        return out

class ResNet3(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_planes=64):
        super(ResNet3, self).__init__()
        self.in_planes = in_planes

        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=2, padding=1, bias=False)
        # self.conv1 = nn.Conv2d(3, 16, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, in_planes , num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, in_planes * 2, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, in_planes * 4, num_blocks[3], stride=2)
        self.linear = nn.Linear(in_planes * 4 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # print("conv1", out.shape)
        out = self.layer1(out)
        # print("layer1", out.shape)
        out = self.layer2(out)
        # print("layer2", out.shape)
        # out = self.layer3(out)
        # print("layer3", out.shape)
        # out = self.layer4(out)
        # print("layer4", out.shape)
        out = F.avg_pool2d(out, 2)
        # print("avg", out.shape)
        out = out.view(out.size(0), -1)
        # print("view", out.shape)
        out = self.linear(out)
        # print(out.shape)
        return out

class ResNet4(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_planes=64):
        super(ResNet4, self).__init__()
        self.in_planes = in_planes

        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=2, padding=1, bias=False)
        # self.conv1 = nn.Conv2d(3, 16, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes*2, num_blocks[0], stride=2)
        # self.layer2 = self._make_layer(block, in_planes , num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, in_planes * 2, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, in_planes * 4, num_blocks[3], stride=2)
        self.linear = nn.Linear(in_planes * 8 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # print("conv1", out.shape)
        out = self.layer1(out)
        # print("layer1", out.shape)
        # out = self.layer2(out)
        # print("layer2", out.shape)
        # out = self.layer3(out)
        # print("layer3", out.shape)
        # out = self.layer4(out)
        # print("layer4", out.shape)
        out = F.avg_pool2d(out, 4)
        # print("avg", out.shape)
        out = out.view(out.size(0), -1)
        # print("view", out.shape)
        out = self.linear(out)
        # print(out.shape)
        return out


def ResNet18(in_planes=2):
    # return ResNet1(BasicBlock, [2, 2, 2, 2], in_planes=in_planes)
    # return ResNet2(BasicBlock, [2, 2, 2, 2], in_planes=in_planes)
    # return ResNet3(BasicBlock, [2, 2, 2, 2], in_planes=8)
    # return ResNet4(BasicBlock, [2, 2, 2, 2], in_planes=16)
    return ResNet4(BasicBlock, [2, 2, 2, 2], in_planes=8)