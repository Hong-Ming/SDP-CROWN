from numpy.lib.arraysetops import isin
import torch
import random
import numpy as np
import torch.nn as nn
from torch.nn import parameter
import torch.nn.functional as F
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from auto_LiRPA.bound_ops import BoundMaxPool
import onnx
from onnx2pytorch import ConvertModel
# This auto_LiRPA release supports a new mode for computing bounds for
# convolutional layers. The new "patches" mode implementation makes full
# backward bounds (CROWN) for convolutional layers significantly faster by
# using more efficient GPU operators, but it is currently stil under beta test
# and may not support any architecutre.  The convolution mode can be set by
# the "conv_mode" key in the bound_opts parameter when constructing your
# BoundeModule object.  In this test we show the difference between Patches
# mode and Matrix mode in memory consumption.

device = 'cpu'
conv_mode = 'matrix' # conv_mode can be set as 'matrix' or 'patches'

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=1)
        self.conv = nn.Conv2d(1, 1, 2, stride=2, padding=1)
        self.linear = nn.Linear(8*8*1, 10)
    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        # x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
## Step 1: Define the model
# model_ori = ResNet(BasicBlock, num_blocks=2, in_planes=8, bn=False, last_layer="dense")
onnx_model = onnx.load('small_mnist.onnx')
model_ori = ConvertModel(onnx_model).to(device)
# model_ori = Model()
# print(model_ori)
# model_ori = models.model_resnet(width=1, mult=4).cuda()
# model_ori = models.ResNet18(in_planes=2).cuda()
# model_ori.load_state_dict(torch.load("data/cifar_base_kw.pth")['state_dict'][0])

## Step 2: Prepare dataset as usual
test_data = torchvision.datasets.MNIST("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())
# print(len(test_data.targets))
# normalize = torchvision.transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010])
# test_data = torchvision.datasets.CIFAR10("./data", train=False, download=True,
#                 transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize]))
# For illustration we only use 1 image from dataset
image_num = 10000
batch_size = 100
n_classes = 10

robust_correct = 0
correct = 0

for i in range(image_num//batch_size):
    image = test_data.data[batch_size*i: batch_size*(i+1)].view(batch_size,1,28,28)
    true_label = test_data.targets[batch_size*i: batch_size*(i+1)]
    # Convert to float
    image = image.to(torch.float32) / 255.0
    image = torch.div(image, 1e-8, rounding_mode='trunc') * 1e-8
    image = image.to(device)

    print("ground truth label: {}".format(true_label))
    # Convert to float
    # image = image.to(torch.float32) / 255.0
    # image = torch.rand((N, 1, 28, 28)).to(torch.float32)
    if device == 'cuda':
        image = image.cuda()

    ## Step 3: wrap model with auto_LiRPA
    # The second parameter is for constructing the trace of the computational graph, and its content is not important.
    # The new "patches" conv_mode provides an more efficient implementation for convolutional neural networks.

    model = BoundedModule(model_ori, image[0].unsqueeze(0), bound_opts={"conv_mode": conv_mode, 'maxpool': 'optimized'}, device=device)

    ## Step 4: Compute bounds using LiRPA given a perturbation
    eps = 0.004
    norm = np.inf

    # generate specifications
    num_class = 10
    c = torch.eye(num_class).type_as(image)[true_label].unsqueeze(1) - torch.eye(num_class).type_as(image).unsqueeze(0)
    # remove specifications to self
    I = (~(true_label.data.unsqueeze(1) == torch.arange(num_class).type_as(true_label.data).unsqueeze(0)))
    c = (c[I].view(image.size(0), num_class - 1, num_class))

    data_max = torch.reshape(torch.tensor((1.)), (1, -1, 1, 1)).to(device)
    data_min = torch.reshape(torch.tensor((0.)), (1, -1, 1, 1)).to(device)
    data_ub = torch.min(image + eps, data_max)
    data_lb = torch.max(image - eps, data_min)

    ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
    image = BoundedTensor(image, ptb)
    # Get model prediction as usual
    pred = model(image)
    # print(pred)

    # Compute bounds
    torch.cuda.empty_cache()
    print('Using {} mode to compute convolution.'.format(conv_mode))
    # lb, ub = model.compute_bounds(IBP=False, C=c, method='backward')

    if False:
        lb, ub = model.compute_bounds(x=(image,), C=c, method='backward')
        lr = 1e-1
        parameters = []
        for l in model._modules.values():
            if isinstance(l, BoundMaxPool):
                l.opt_start()
                l.init_opt_parameters(image, None)
                parameters.append(l.alpha)

        opt = torch.optim.Adam(parameters, lr=lr)
        for i in range(20):
            lb, ub = model.compute_bounds(x=(image,), C=c, IBP=False, method='backward')
            opt.zero_grad()
            (-lb).sum().backward()
            # print(parameters[0].grad)
            opt.step()
            print((-lb).sum())
    else:
        lb, ub = model.compute_bounds(x=(image,), IBP=False, C=c, method='CROWN-Optimized')
    ## Step 5: Final output
    # pred = pred.detach().cpu().numpy()
    lb = lb.detach().cpu().numpy()
    ub = ub.detach().cpu().numpy()

    for i in range(batch_size):
        correct += (pred[i][true_label[i]] >= pred[i]).all()
        robust_correct += (lb[i] >= 0).all()

print(robust_correct, correct)