import sys 
import os
sys.path.append('../../complete_verifier/')
from enum import auto
import enum
# from complete_verifier.model_defs import cifar_model_wide, cnn_4layer_b, mnist_cnn_4layer
from auto_LiRPA import BoundedModule, BoundedTensor
import torch
import torchvision
# from complete_verifier.auto_attack.autoattack import AutoAttack
#from auto_attack.autoattack import AutoAttack
import random
# from complete_verifier.attack_pgd import attack_pgd
from attack_pgd import attack_pgd
from plnn.advmnist_models.adv_models import *
import numpy as np
import matplotlib.pyplot as plt
import csv
import torchvision.transforms as transforms
import multiprocessing
import torch.nn as nn
# from complete_verifier.model_defs import *
from model_defs import *
from torch.autograd import Variable
import torch
import time
device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "gpu not supported!"


class Normalization(nn.Module):
    def __init__(self, mean, std, model):
        super(Normalization, self).__init__()
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)
        self.model = model
    
    def forward(self, x):
        return self.model((x - self.mean)/self.std)


def clamp(X, lower_limit=None, upper_limit=None):
    if lower_limit is None and upper_limit is None:
        return X
    if lower_limit is not None:
        return torch.max(X, lower_limit)
    if upper_limit is not None:
        return torch.min(X, upper_limit)
    return torch.max(torch.min(X, upper_limit), lower_limit)

    
def pairwise_dist (A, B):
    D = A.pow(2).sum(1, keepdim = True) + B.pow(2).sum(1, keepdim = True).t() - 2 * torch.mm(A, B.t())
    return torch.clamp(D, 0.0)


def svgd_kernel(Theta):
    Theta_shape = list(Theta.size()) 
    theta = Theta.view(-1, Theta_shape[1]*Theta_shape[2]*Theta_shape[3])

    pairwise_dists = pairwise_dist(theta, theta)
    theta_shape = list(theta.size())
    h = torch.median(pairwise_dists)
    h_square = 0.5*torch.div(h, torch.log(torch.tensor([float(theta_shape[0])]).cuda()))
    Kxy = torch.exp(-0.5*torch.div(pairwise_dists, h_square))
    
    dxkxy = -torch.mm(Kxy, theta)
    sumkxy = torch.sum(Kxy, dim=1, keepdim=True)
    dxkxy = dxkxy + theta*sumkxy.expand((theta_shape[0], theta_shape[1]))
    dxkxy = torch.div(dxkxy, h_square)
    
    return (Kxy, dxkxy)


def distributional_attack(model, X, y, epsilon, niters=100, alpha=0.01): 
    out = model(X)
    ce = nn.CrossEntropyLoss()(out, y)
    err = (out.data.max(1)[1] != y.data).float().sum()  / X.size(0)

    X_pgd = Variable(X.data, requires_grad=True)
    for i in range(niters): 
        opt = torch.optim.Adam([X_pgd], lr=1e-3)
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        
        grad = X_pgd.grad.data
        Kxy, dxkxy = svgd_kernel(X_pgd.data)
        
        X_shape = list(X_pgd.data.size())
        svgd = -(torch.mm(Kxy, -grad.view(-1, X_shape[1]*X_shape[2]*X_shape[3])) + dxkxy)/float(X_shape[0])
        
        eta = alpha*(0.05*svgd.view(X_shape[0], X_shape[1], X_shape[2], X_shape[3])+grad).sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        X_pgd = torch.clamp(X_pgd.data, 0., 1.)
        X_pgd = Variable(torch.max(torch.min(X_pgd, X + epsilon), X - epsilon), requires_grad=True)
        
        # import pdb; pdb.set_trace()
        assert ((X_pgd - X).max() <= epsilon + 1e-3).all()
        assert ((X_pgd - X).min() >= -epsilon - 1e-3).all()
        assert X_pgd.min() >= 0.
        assert X_pgd.max() <= 1.
    
    return X_pgd


def auto_attack(model_ori, epsilon, images, labels, num_example=100, lower_limit=0.0, upper_limit=1.0):
    adversary = AutoAttack(model_ori, norm='Linf', eps=epsilon, version='standard')

    ans = []
    for i in range(num_example):
        adversary.seed = random.randint(0,10000)
        x_adv = adversary.run_standard_evaluation(images, labels, bs=256)
        ans.append(x_adv)

    return torch.cat(ans, dim=1)

def diversity_pgd(model_ori, epsilon, images, labels, num_example=100, lower_limit=0.0, upper_limit=1.0):
    ans = []
    for i in range(num_example):
        best_delta = attack_pgd(model_ori, images, labels, epsilon, torch.max(epsilon).item(), 50, 5, OSI_init_X=False, lower_limit=lower_limit, upper_limit=upper_limit)
        ans.append(best_delta + images)
    return torch.cat(ans, dim=1)
    
def read_idx(txt_file):
    out = []
    f = open(txt_file, "r")
    for line in f.readlines():
        if len(line) > 1:
            out.append(int(line[:-1]))
    print(txt_file, len(out))
    f.close()
    return out


# model_name = "cnn_4_layer"
# model_name = "cifar_cnn_a_mix"
model_name = sys.argv[1]
task = "generate"
unknown_filter = f"../../complete_verifier/exp_configs/bab_attack/attack_idx/{model_name}/mip_unknown_idx.txt"
unsafe_filter = f"../../complete_verifier/exp_configs/bab_attack/attack_idx/{model_name}/mip_unsafe_idx.txt"
filter = []
if os.path.exists(unknown_filter):
    print("unknown filters exist, load filters")
    filter += read_idx(unknown_filter)
if os.path.exists(unsafe_filter):
    print("unsafe filters exist, load filters")
    filter += read_idx(unsafe_filter)
if __name__ == "__main__":
    random.seed(100)
    eps = None
    if model_name == "mnist_cnn_a_adv":
        model = mnist_cnn_4layer()
        # state_dict = torch.load("plnn/sdp_models/mnist_cnn_a_adv.model")
        state_dict = torch.load("../../complete_verifier/models/sdp/mnist_cnn_a_adv.model")
        dataset = "mnist"
    elif model_name == "mnist_MadryCNN_no_maxpool_tiny":
        model = MadryCNN_no_maxpool_tiny()
        state_dict = torch.load("plnn/advmnist_models/MadryCNN_no_maxpool_tiny.pt")
        dataset = "mnist"
    elif model_name == "mnist_conv_small":
        model = mnist_conv_small()
        state_dict = torch.load("../../complete_verifier/models/eran/mnist_conv_small_nat.pth")["state_dict"][0]
        dataset = "mnist"
        mean = [0.1307]
        std = [0.3081]
        eps = torch.Tensor([0.12]).view((1,1,1,1)).to(device)
    elif model_name == "cifar_cnn_b_adv":
        model = cnn_4layer_b()
        # state_dict = torch.load("plnn/sdp_models/cifar_cnn_b_adv.model")
        state_dict = torch.load("../../complete_verifier/models/sdp/cifar_cnn_b_adv.model")
        dataset = "cifar"
        mean = [125.3/255, 123.0/255, 113.9/255]
        std = [63.0/255, 62.1/255, 66.7/255]
    elif model_name == "cifar_cnn_a_adv":
        model = cnn_4layer_adv()
        # state_dict = torch.load("plnn/sdp_models/cifar_cnn_b_adv.model")
        state_dict = torch.load("../../complete_verifier/models/sdp/cifar_cnn_a_adv.model")
        dataset = "cifar"
        mean = [125.3/255, 123.0/255, 113.9/255]
        std = [63.0/255, 62.1/255, 66.7/255]
        filter = read_idx(unknown_filter) +  read_idx(unsafe_filter)
    elif model_name == "cifar_cnn_a_adv_alt":
        model = cnn_4layer_adv()
        # state_dict = torch.load("plnn/sdp_models/cifar_cnn_b_adv.model")
        state_dict = torch.load("../../complete_verifier/models/sdp/cifar_cnn_a_adv_alt.model")
        dataset = "cifar"
        mean = [125.3/255, 123.0/255, 113.9/255]
        std = [63.0/255, 62.1/255, 66.7/255]
        filter = read_idx(unknown_filter) +  read_idx(unsafe_filter)
    elif model_name == "cifar_cnn_a_mix":
        model = cnn_4layer_mix4()
        # state_dict = torch.load("plnn/sdp_models/cifar_cnn_b_adv.model")
        state_dict = torch.load("../../complete_verifier/models/sdp/cifar_cnn_a_mix.model")
        dataset = "cifar"
        mean = [125.3/255, 123.0/255, 113.9/255]
        std = [63.0/255, 62.1/255, 66.7/255]
    elif model_name == "cifar_lpd_cnn_a":
        # cifar_lpd_cnn_a
        model = cifar_model_wide()
        # state_dict = torch.load("./cifar_small_2px.pth")["state_dict"][0]
        state_dict = torch.load("../../complete_verifier/models/bab_attack/cifar_small_2px.pth")["state_dict"][0]
        dataset = "cifar"
        mean=[0.485, 0.456, 0.406]
        std=[0.225, 0.225, 0.225]
    elif model_name == "model_resnet":
        model = model_resnet()
        state_dict = torch.load("../../complete_verifier/models/eran/cifar_resnet_8px.pth")["state_dict"][0]
        dataset = "cifar"
        mean=[0.485, 0.456, 0.406]
        std=[0.225, 0.225, 0.225]
        eps = torch.Tensor([8./255]*3).view((1,3,1,1)).to(device)
    else:
        print("model name not supported!")
        exit()
        
    model.load_state_dict(state_dict)
    model = model.to(device)

    # data
    if dataset == "mnist":
        test_data = torchvision.datasets.MNIST("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())
        if filter:
            test_data = torch.utils.data.Subset(test_data, filter)
        test_data = torch.utils.data.DataLoader(test_data, batch_size=256, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),4))
        mean = torch.Tensor(mean).view((1,1,1,1)).to(device)
        std = torch.Tensor(std).view((1,1,1,1)).to(device)
        model = Normalization(mean, std, model)

    else:
        test_data = torchvision.datasets.CIFAR10("./data", train=False, download=True, 
                transform=transforms.Compose([transforms.ToTensor()]))
        if filter:
            test_data = torch.utils.data.Subset(test_data, filter)
        test_data = torch.utils.data.DataLoader(test_data, batch_size=256, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),4))

        mean = torch.Tensor(mean).view((1,3,1,1)).to(device)
        std = torch.Tensor(std).view((1,3,1,1)).to(device)
        model = Normalization(mean, std, model)

    preds = []
    margin = []
    clean_pred = []
    acc = 0
    attack_acc = 0
    total = 0
    start_time = time.time()
    for (data, label) in test_data:
        image = data
        true_label = label
        image = image.to(torch.float32)
        image = image.to(device)
        total += image.shape[0]
        
        true_label = true_label.to(device)

        # lirpa_model = BoundedModule(model, torch.empty_like(image), device=device)
        if eps is None:
            if dataset == 'mnist':
                eps = torch.Tensor([0.3]).view((1,1,1,1)).to(device)
            else:
                eps = torch.Tensor([2./255]*3).view((1,3,1,1)).to(device)
        print(eps)

        # adv_data = auto_attack(model, eps, image, true_label, 1)
        adv_data = distributional_attack(model, image, true_label, eps)
        assert (adv_data >= 0.0).all()
        assert (adv_data <= 1.0).all()
        if (adv_data-image).abs().max() <= eps.max():
            print((adv_data-image).abs().max())
        # assert (attack_image-x).abs().max() <= eps_temp.max(), f"{(attack_image-x).abs().max()} <= {eps_temp.max()}"

        l = model(adv_data)
        
        pred = torch.argmax(l, -1)
        preds.append(pred)
        attack_acc += (l.argmax(1) == true_label).sum().item()

        clean = model(image)
        clean_pred.append(clean)
        acc += (clean.argmax(1) == true_label).sum().item()

        l = l.detach().cpu().numpy()
        for idx, logit in enumerate(l):
            l[idx]=logit[label[idx]]-l[idx]
        margin.append(l)
    
    pred = torch.cat(preds)
    clean_pred = torch.cat(clean_pred, axis=0)
    margin = np.concatenate(margin, axis=0)
    # print(margin[4024])
    # print(clean_pred[4024])
    print(f"clean acc: {acc / total}[{total}]; attack success: {1 - attack_acc / total}[{total - attack_acc}/{total}]; time: {(time.time() - start_time) / total}[{total}]")

    # torch.save(pred, "my_data/{}_new_auto_attack_result.torch".format(model_name))

    # # save the data
    # torch.save(adv_data, "my_data/{}_new_auto_attack_adv_data.torch".format(model_name))
    # torch.save(margin, "my_data/{}_new_auto_attack_margin.torch".format(model_name))
    # torch.save(clean_pred, "my_data/{}_new_auto_attack_clean_pred.torch".format(model_name))
