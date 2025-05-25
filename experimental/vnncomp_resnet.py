### preprocessor-hint: private-file

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import arguments

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True, kernel=3):
        super(BasicBlock, self).__init__()
        self.bn = bn
        if kernel == 3:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=(not self.bn))
        elif kernel == 2:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=2, stride=stride, padding=1, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=2,
                                   stride=1, padding=0, bias=(not self.bn))
        elif kernel == 1:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
                                   stride=1, padding=0, bias=(not self.bn))
        else:
            exit("kernel not supported!")

        if self.bn:
            self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if self.bn:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=(not self.bn)),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=(not self.bn)),
                )

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True, kernel=3):
        super(BasicBlock2, self).__init__()
        self.bn = bn
        if kernel == 3:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=(not self.bn))
        elif kernel == 2:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=2, stride=stride, padding=1, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=2,
                                   stride=1, padding=0, bias=(not self.bn))
        elif kernel == 1:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=(not self.bn))
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
                                   stride=1, padding=0, bias=(not self.bn))
        else:
            exit("kernel not supported!")

        if self.bn:
            self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if self.bn:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=(not self.bn)),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=(not self.bn)),
                )

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
        out += self.shortcut(x)
        return out

class ResNet9_v4(nn.Module):
	def __init__(self, block, num_blocks=2, num_classes=10, in_planes=64, bn=True, last_layer="avg"):
		super(ResNet9_v4, self).__init__()
		self.in_planes = in_planes
		self.bn = bn
		self.last_layer = last_layer
		self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
							   stride=2, padding=0, bias=not self.bn)
		if self.bn:
			self.bn1 = nn.BatchNorm2d(in_planes)
		self.layer1 = self._make_layer(block, in_planes, num_blocks, stride=2, bn=bn, kernel=3)
		self.layer2 = self._make_layer(block, in_planes*2, num_blocks, stride=2, bn=bn, kernel=3)
		self.layer3 = self._make_layer(block, in_planes*2, num_blocks, stride=2, bn=bn, kernel=3)
		if self.last_layer == "avg":
			self.avg2d = nn.AvgPool2d(4)
			self.linear = nn.Linear(in_planes * 2 * block.expansion, num_classes)
		elif self.last_layer == "dense":
			self.linear1 = nn.Linear(in_planes * 2 * block.expansion * 16 // 4, 100)
			self.linear2 = nn.Linear(100, num_classes)
		else:
			exit("last_layer type not supported!")

	def _make_layer(self, block, planes, num_blocks, stride, bn, kernel):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride, bn, kernel))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):
		if self.bn:
			out = F.relu(self.bn1(self.conv1(x)))
		else:
			out = F.relu(self.conv1(x))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		if self.last_layer == "avg":
			out = self.avg2d(out)
			out = out.view(out.size(0), -1)
			out = self.linear(out)
		elif self.last_layer == "dense":
			out = torch.flatten(out, 1)
			out = F.relu(self.linear1(out))
			out = self.linear2(out)
		return out

class ResNet5(nn.Module):
    def __init__(self, block, num_blocks=2, num_classes=10, in_planes=64, bn=True, last_layer="avg"):
        super(ResNet5, self).__init__()
        self.in_planes = in_planes
        self.bn = bn
        self.last_layer = last_layer
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=2, padding=1, bias=not self.bn)
        if self.bn: self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes*2, num_blocks, stride=2, bn=bn, kernel=3)
        if self.last_layer == "avg":
            self.avg2d = nn.AvgPool2d(4)
            self.linear = nn.Linear(in_planes * 8 * block.expansion, num_classes)
        elif self.last_layer == "dense":
            self.linear1 = nn.Linear(in_planes * 8 * block.expansion * 16, 100)
            self.linear2 = nn.Linear(100, num_classes)
        else:
            exit("last_layer type not supported!")

    def _make_layer(self, block, planes, num_blocks, stride, bn, kernel):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn, kernel))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        out = self.layer1(out)
        if self.last_layer == "avg":
            out = self.avg2d(out)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        elif self.last_layer == "dense":
            out = torch.flatten(out, 1)
            out = F.relu(self.linear1(out))
            out = self.linear2(out)
        return out


class ResNet9(nn.Module):
    def __init__(self, block, num_blocks=2, num_classes=10, in_planes=64, bn=True, last_layer="avg"):
        super(ResNet9, self).__init__()
        self.in_planes = in_planes
        self.bn = bn
        self.last_layer = last_layer
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=2, padding=1, bias=not self.bn)
        if self.bn: self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes*2, num_blocks, stride=2, bn=bn, kernel=3)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks, stride=2, bn=bn, kernel=3)
        if self.last_layer == "avg":
            self.avg2d = nn.AvgPool2d(4)
            self.linear = nn.Linear(in_planes * 2 * block.expansion, num_classes)
        elif self.last_layer == "dense":
            self.linear1 = nn.Linear(in_planes * 2 * block.expansion * 16, 100)
            self.linear2 = nn.Linear(100, num_classes)
        else:
            exit("last_layer type not supported!")

    def _make_layer(self, block, planes, num_blocks, stride, bn, kernel):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn, kernel))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        if self.last_layer == "avg":
            out = self.avg2d(out)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        elif self.last_layer == "dense":
            out = torch.flatten(out, 1)
            out = F.relu(self.linear1(out))
            out = self.linear2(out)
        return out

class ResNet9_v1(nn.Module):
    def __init__(self, block, num_blocks=2, num_classes=10, in_planes=64, bn=True, last_layer="avg"):
        super(ResNet9_v1, self).__init__()
        self.in_planes = in_planes
        self.bn = bn
        self.last_layer = last_layer
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=2, padding=0, bias=not self.bn)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes*2, num_blocks, stride=2, bn=bn, kernel=3)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks, stride=2, bn=bn, kernel=3)
        if self.last_layer == "avg":
            self.avg2d = nn.AvgPool2d(4)
            self.linear = nn.Linear(in_planes * 2 * block.expansion, num_classes)
        elif self.last_layer == "dense":
            self.linear1 = nn.Linear(in_planes * 2 * block.expansion * 16, 100)
            self.linear2 = nn.Linear(100, num_classes)
        else:
            exit("last_layer type not supported!")

    def _make_layer(self, block, planes, num_blocks, stride, bn, kernel):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn, kernel))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        if self.last_layer == "avg":
            out = self.avg2d(out)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        elif self.last_layer == "dense":
            out = torch.flatten(out, 1)
            out = F.relu(self.linear1(out))
            out = self.linear2(out)
        return out

class ResNet9_v2(nn.Module):
    def __init__(self, block, num_blocks=2, num_classes=10, in_planes=64, bn=True, last_layer="avg"):
        super(ResNet9_v2, self).__init__()
        self.in_planes = in_planes
        self.bn = bn
        self.last_layer = last_layer
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=2, padding=0, bias=not self.bn)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes*2, num_blocks, stride=2, bn=bn, kernel=3)
        self.layer2 = self._make_layer(block, in_planes*4, num_blocks, stride=2, bn=bn, kernel=3)
        if self.last_layer == "avg":
            self.avg2d = nn.AvgPool2d(4)
            self.linear = nn.Linear(in_planes * 2 * block.expansion, num_classes)
        elif self.last_layer == "dense":
            self.linear1 = nn.Linear(in_planes * 2 * block.expansion * 16 * 2, 100)
            self.linear2 = nn.Linear(100, num_classes)
        else:
            exit("last_layer type not supported!")

    def _make_layer(self, block, planes, num_blocks, stride, bn, kernel):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn, kernel))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        if self.last_layer == "avg":
            out = self.avg2d(out)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        elif self.last_layer == "dense":
            out = torch.flatten(out, 1)
            out = F.relu(self.linear1(out))
            out = self.linear2(out)
        return out

class ResNet18_v2(nn.Module):
    def __init__(self, block, num_blocks=2, num_classes=10, in_planes=64, bn=True):
        super(ResNet18_v2, self).__init__()
        self.in_planes = in_planes
        self.bn = bn
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=1, padding=1, bias=not self.bn)
        if (self.bn):
            self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes, num_blocks, stride=1, bn=bn, kernel=3)
        self.layer2 = self._make_layer(block, in_planes * 2, num_blocks, stride=2, bn=bn, kernel=3)
        self.layer3 = self._make_layer(block, in_planes * 2, num_blocks, stride=2, bn=bn, kernel=3)
        self.layer4 = self._make_layer(block, in_planes * 4, num_blocks, stride=2, bn=bn, kernel=3)
        self.linear1 = nn.Linear(in_planes * 8 * block.expansion * 8, 100)
        self.linear2 = nn.Linear(100, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, bn, kernel):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn, kernel))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if (self.bn):
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.flatten(out, 1)
        out = F.relu(self.linear1(out))
        out = self.linear2(out)
        return out
class ResNet18_v4(nn.Module):
    def __init__(self, block, num_blocks=2, num_classes=10, in_planes=64, bn=True):
        super(ResNet18_v4, self).__init__()
        self.in_planes = in_planes
        self.bn = bn
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=1, padding=1, bias=not self.bn)
        if (self.bn):
            self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes, num_blocks, stride=1, bn=bn, kernel=3)
        self.layer2 = self._make_layer(block, in_planes, num_blocks, stride=2, bn=bn, kernel=3)
        self.layer3 = self._make_layer(block, in_planes * 2, num_blocks, stride=2, bn=bn, kernel=3)
        self.layer4 = self._make_layer(block, in_planes * 2, num_blocks, stride=2, bn=bn, kernel=3)
        self.linear1 = nn.Linear(in_planes * 8 * block.expansion * 4, 100)
        self.linear2 = nn.Linear(100, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, bn, kernel):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn, kernel))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if (self.bn):
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.flatten(out, 1)
        out = F.relu(self.linear1(out))
        out = self.linear2(out)
        return out


def conv3x3(in_planes, out_planes, bn=True, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=not bn)

def conv_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		init.xavier_uniform_(m.weight, gain=np.sqrt(2))
		init.constant_(m.bias, 0)
	elif classname.find('BatchNorm') != -1:
		init.constant_(m.weight, 1)
		init.constant_(m.bias, 0)

class wide_basic(nn.Module):
	def __init__(self, in_planes, planes, dropout_rate, bn=True, stride=1):
		super(wide_basic, self).__init__()
		self.bn = bn
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=not self.bn)
		if (self.bn): self.bn1 = nn.BatchNorm2d(planes)
		# self.dropout = nn.Dropout(p=dropout_rate)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=not self.bn)
		if (self.bn): self.bn2 = nn.BatchNorm2d(planes)
		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != planes:
			if (self.bn):
				self.shortcut = nn.Sequential(
					nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
					nn.BatchNorm2d(planes)
				)
			else:
				self.shortcut = nn.Sequential(
					nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True)
				)


	def forward(self, x):
		# out = self.dropout(self.conv1(F.relu(self.bn1(x))))
		if (self.bn):
			out = F.relu(self.bn1(self.conv1(x)))
			out = F.relu(self.bn2(self.conv2(out)))
		else:
			out = F.relu(self.conv1(x))
			out = F.relu(self.conv2(out))
		out += self.shortcut(x)

		return out

class Wide_ResNet(nn.Module):
	def __init__(self, depth, widen_factor, dropout_rate, num_classes,
			in_planes=16, in_dim=56, bn=True):
		super(Wide_ResNet, self).__init__()
		self.in_planes = in_planes
		self.bn = bn
		assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
		n = (depth-4)/6
		k = widen_factor

		print('| Wide-Resnet %dx%d' %(depth, k))
		nStages = [in_planes, in_planes*k, in_planes*k]#, in_planes*4*k]

		self.conv1 = conv3x3(3,nStages[0],bn=self.bn)
		if (self.bn):
			self.bn1 = nn.BatchNorm2d(nStages[0])
		self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=2, bn=bn)
		self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2, bn=bn)
		#self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
		#self.bn1 = nn.BatchNorm2d(nStages[2], momentum=0.1)
		self.linear1 = nn.Linear(nStages[2] * (in_dim//4)**2, 200)
		self.linear2 = nn.Linear(200, num_classes)

	def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, bn):
		strides = [stride] + [1]*(int(num_blocks)-1)
		layers = []

		for stride in strides:
			layers.append(block(self.in_planes, planes, dropout_rate, bn=bn, stride=stride))
			self.in_planes = planes

		return nn.Sequential(*layers)

	def forward(self, x):
		if (self.bn): out = F.relu(self.bn1(self.conv1(x)))
		else: out = F.relu(self.conv1(x))

		out = self.layer1(out)
		out = self.layer2(out)
		#out = self.layer3(out)
		#out = F.relu(self.bn1(out))
		out = torch.flatten(out, 1)
		out = F.relu(self.linear1(out))
		out = self.linear2(out)
		return out

class ImageNet_ResNet9_v1(nn.Module):
    def __init__(self, block, num_blocks=2, num_classes=10, in_planes=64, bn=True, last_layer="avg"):
        super(ImageNet_ResNet9_v1, self).__init__()
        self.in_planes = in_planes
        self.bn = bn
        self.last_layer = last_layer
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=2, padding=0, bias=not self.bn)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes*2, num_blocks, stride=2, bn=bn, kernel=3)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks, stride=2, bn=bn, kernel=3)
        if self.last_layer == "avg":
            self.avg2d = nn.AvgPool2d(4)
            self.linear = nn.Linear(in_planes * 2 * block.expansion, num_classes)
        elif self.last_layer == "dense":
            self.linear1 = nn.Linear(98 * in_planes, 200)
            self.linear2 = nn.Linear(200, 200)
        else:
            exit("last_layer type not supported!")

    def _make_layer(self, block, planes, num_blocks, stride, bn, kernel):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bn, kernel))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        if self.last_layer == "avg":
            out = self.avg2d(out)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        elif self.last_layer == "dense":
            out = torch.flatten(out, 1)
            out = F.relu(self.linear1(out))
            out = self.linear2(out)
        return out


class ImageNet_ResNet9_v4(nn.Module):
	def __init__(self, block, num_blocks=2, num_classes=10, in_planes=64, bn=True, last_layer="avg"):
		super(ImageNet_ResNet9_v4, self).__init__()
		self.in_planes = in_planes
		self.bn = bn
		self.last_layer = last_layer
		self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
							   stride=2, padding=0, bias=not self.bn)
		if self.bn:
			self.bn1 = nn.BatchNorm2d(in_planes)
		self.layer1 = self._make_layer(block, in_planes, num_blocks, stride=2, bn=bn, kernel=3)
		self.layer2 = self._make_layer(block, in_planes*2, num_blocks, stride=2, bn=bn, kernel=3)
		self.layer3 = self._make_layer(block, in_planes*2, num_blocks, stride=2, bn=bn, kernel=3)
		if self.last_layer == "avg":
			self.avg2d = nn.AvgPool2d(4)
			self.linear = nn.Linear(in_planes * 2 * block.expansion, num_classes)
		elif self.last_layer == "dense":
			self.linear1 = nn.Linear(32 * in_planes, 200)
			self.linear2 = nn.Linear(200, 200)
		else:
			exit("last_layer type not supported!")

	def _make_layer(self, block, planes, num_blocks, stride, bn, kernel):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride, bn, kernel))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):
		if self.bn:
			out = F.relu(self.bn1(self.conv1(x)))
		else:
			out = F.relu(self.conv1(x))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		if self.last_layer == "avg":
			out = self.avg2d(out)
			out = out.view(out.size(0), -1)
			out = self.linear(out)
		elif self.last_layer == "dense":
			out = torch.flatten(out, 1)
			out = F.relu(self.linear1(out))
			out = self.linear2(out)
		return out


def resnet2b(num_classes):
    return ResNet5(BasicBlock, num_blocks=2, num_classes=num_classes, in_planes=8, bn=False, last_layer="dense")

def resnet4b(num_classes):
    return ResNet9(BasicBlock, num_blocks=2, num_classes=num_classes, in_planes=16, bn=True, last_layer="dense")

def resnet_v1(num_classes, bn):
    return ResNet9_v1(BasicBlock, num_blocks=2, num_classes=num_classes, in_planes=32, bn=bn, last_layer="dense")

def resnet_v2(num_classes, bn):
    return ResNet9_v1(BasicBlock, num_blocks=2, num_classes=num_classes, in_planes=64, bn=bn, last_layer="dense")

def resnet_v3(num_classes, bn):
    return ResNet9_v1(BasicBlock, num_blocks=3, num_classes=num_classes, in_planes=32, bn=bn, last_layer="dense")

def resnet_v4(num_classes, bn):
    return ResNet9_v1(BasicBlock, num_blocks=3, num_classes=num_classes, in_planes=16, bn=bn, last_layer="dense")

def resnet_v5(num_classes, bn):
    return ResNet9_v1(BasicBlock2, num_blocks=3, num_classes=num_classes, in_planes=32, bn=bn, last_layer="dense")

def resnet_v6(num_classes, bn):
    return ResNet9_v2(BasicBlock, num_blocks=3, num_classes=num_classes, in_planes=16, bn=bn, last_layer="dense")
def resnet_v12(num_classes, bn):
    return ResNet9_v4(BasicBlock2, num_blocks=3, num_classes=num_classes, in_planes=128, bn=bn, last_layer="dense")
def resnet18_v2(num_classes, bn):
    return ResNet18_v2(BasicBlock2, num_blocks=2, num_classes=num_classes, in_planes=64, bn=bn)
def resnet18_v4(num_classes, bn):
    return ResNet18_v4(BasicBlock2, num_blocks=2, num_classes=num_classes, in_planes=64, bn=bn)
def resnet18_v5(num_classes, bn):
    return ResNet18_v4(BasicBlock2, num_blocks=2, num_classes=num_classes, in_planes=32, bn=bn)

def resnet_v11(num_classes, bn):
    return ResNet9_v1(BasicBlock2, num_blocks=4, num_classes=num_classes, in_planes=64, bn=bn, last_layer="dense")

def resnet_p11(num_classes=200, bn=False):
    return ImageNet_ResNet9_v1(BasicBlock2, num_blocks=4, num_classes=num_classes, in_planes=64, bn=bn, last_layer="dense")

def resnet_p12(num_classes=200, bn=False):
    return ImageNet_ResNet9_v4(BasicBlock2, num_blocks=3, num_classes=num_classes, in_planes=128, bn=bn, last_layer="dense")

def wide_resnet_imagenet64(in_ch=3, in_dim=56, in_planes=16, widen_factor=10, bn=True):
	return Wide_ResNet(10, widen_factor, 0.3, 200, in_dim=in_dim, in_planes=in_planes, bn=bn)


# Definition of data.
def load_sampled_cifar100(eps, seed=111, size=200):
    database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    normalize = transforms.Normalize(mean=arguments.Config["data"]["mean"], std=arguments.Config["data"]["std"])
    loader = datasets.CIFAR100
    test_data = loader(database_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_data.mean = torch.tensor(arguments.Config["data"]["mean"])
    test_data.std = torch.tensor(arguments.Config["data"]["std"])
    testloader = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=False, num_workers=4)
    X, labels = next(iter(testloader))
    # Rescale epsilon.
    eps = torch.reshape(eps / torch.tensor(arguments.Config["data"]["std"], dtype=torch.get_default_dtype()), (1, -1, 1, 1))
    # set data_max and data_min to be None if no clip
    data_max = torch.reshape((1. - test_data.mean) / test_data.std, (1, -1, 1, 1))
    data_min = torch.reshape((0. - test_data.mean) / test_data.std, (1, -1, 1, 1))

    np.random.seed(seed)
    perm = np.random.permutation(X.shape[0])[:size]
    X, labels = X[perm], labels[perm]
    print(f'data checksum: {torch.sum(torch.abs(X.reshape(-1)))}')

    return X, labels, data_max, data_min, eps

def load_sampled_tinyimagenet(eps, seed=111, size=200):
    database_path = "../data/tinyImageNet/tiny-imagenet-200/val"
    #os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')

    normalize = transforms.Normalize(mean=arguments.Config["data"]["mean"], std=arguments.Config["data"]["std"])
    test_data = datasets.ImageFolder(database_path,
                                     transform=transforms.Compose([
                                         # transforms.RandomResizedCrop(64, scale=(0.875, 0.875), ratio=(1., 1.)),
                                         transforms.CenterCrop(56),
                                         transforms.ToTensor(),
                                         normalize]))

    test_data.mean = torch.tensor(arguments.Config["data"]["mean"])
    test_data.std = torch.tensor(arguments.Config["data"]["std"])
    testloader = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=False, num_workers=4)
    X, labels = next(iter(testloader))
    # Rescale epsilon.
    eps = torch.reshape(eps / torch.tensor(arguments.Config["data"]["std"], dtype=torch.get_default_dtype()), (1, -1, 1, 1))
    # set data_max and data_min to be None if no clip
    data_max = torch.reshape((1. - test_data.mean) / test_data.std, (1, -1, 1, 1))
    data_min = torch.reshape((0. - test_data.mean) / test_data.std, (1, -1, 1, 1))

    np.random.seed(seed)
    perm = np.random.permutation(X.shape[0])[:size]
    X, labels = X[perm], labels[perm]
    print(f'data checksum: {torch.sum(torch.abs(X.reshape(-1)))}')

    return X, labels, data_max, data_min, eps



if __name__ == '__main__':
    model_ori = resnet_v5(100, bn=True)
    model_ori.load_state_dict(torch.load("vnn2022_benchmarks/resnet6b+_no_blockrelu_0.20_0.80_ori.pt"))
    model_fusion = resnet_v5(100, bn=False)
    model_fusion.load_state_dict(torch.load("vnn2022_benchmarks/resnet6b+_no_blockrelu_0.20_0.80_fusion.pt"))
    model_ori.eval()
    model_fusion.eval()

    with torch.no_grad():
        input = torch.zeros((1, 3, 32, 32))
        output = model_ori(input)
        output2 = model_fusion(input)
        print(torch.sum((output - output2).reshape(-1)))
