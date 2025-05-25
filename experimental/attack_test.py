from enum import auto
import enum
from pdb import set_trace
import sys
sys.path.append("../../")
from complete_verifier.model_defs import cifar_model_wide, cnn_4layer_b, mnist_cnn_4layer, cnn_4layer_adv, cifar_conv_small
from auto_LiRPA import BoundedModule, BoundedTensor
import torch
import torchvision
from autoattack import AutoAttack
import random
from complete_verifier.attack.attack_pgd import attack_pgd
from plnn.advmnist_models.adv_models import *
import numpy as np
import matplotlib.pyplot as plt
import csv
import torchvision.transforms as transforms
import multiprocessing
import torch.nn as nn
import pandas as pd
from complete_verifier.model_defs import *
from mip_log import analyze_log, recal_time
import pickle as pkl

class Normalization(nn.Module):
    def __init__(self, mean, std, model):
        super(Normalization, self).__init__()
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)
        self.model = model

    def forward(self, x):
        return self.model((x - self.mean)/self.std)





def auto_attack(model_ori, epsilon, images, labels, num_example=100, lower_limit=0.0, upper_limit=1.0):
    adversary = AutoAttack(model_ori, norm='Linf', eps=epsilon, version='standard')

    ans = []
    for i in range(num_example):
        adversary.seed = random.randint(0,10000)
        x_adv = adversary.run_standard_evaluation(images, labels, bs=1)
        ans.append(x_adv)

    return torch.cat(ans, dim=1)

def diversed_pgd_attack(model_ori, epsilon, images, labels, num_example=100, lower_limit=0.0, upper_limit=1.0):
    ans = []
    for i in range(num_example):
        best_delta, _ = attack_pgd(model_ori, images, labels, epsilon, torch.max(epsilon).item(), 1000, 500, initialization="osi", lower_limit=lower_limit, upper_limit=upper_limit)
        ans.append(best_delta + images)
    return torch.cat(ans, dim=1)

def pgd_attack(model_ori, epsilon, images, labels, num_example=100, lower_limit=0.0, upper_limit=1.0):
    ans = []
    for i in range(num_example):
        best_delta, _ = attack_pgd(model_ori, images, labels, epsilon, torch.max(epsilon).item(), 1000, 500, initialization="uniform", lower_limit=lower_limit, upper_limit=upper_limit)
        ans.append(best_delta + images)
    return torch.cat(ans, dim=1)




def main(model_name, method, task):
    random.seed(100)
    pkl_name = "data/attack_filters/filtered_{}.pkl".format(model_name)
    if model_name == "mnist_cnn_a_adv":
        model = mnist_cnn_4layer()
        state_dict = torch.load("../../complete_verifier/models/sdp/mnist_cnn_a_adv.model")
        dataset = "mnist"
        timeout = 3600
        mean = [0.0]
        std = [1.0]
    # elif model_name == "robust_test":
    #     from robustbench import load_model
    #     model = load_model(model_name='Carmon2019Unlabeled', dataset='cifar10', threat_model='Linf')
    #     dataset = "cifar"
    #     std = [1.0, 1.0, 1.0]
    #     mean = [0.0, 0.0, 0.0]
    elif model_name == "mnist_small_adv":
        model = MadryCNN_no_maxpool_tiny()
        state_dict = torch.load("plnn/advmnist_models/MadryCNN_no_maxpool_tiny.pt")
        pkl_name = "data/attack_filters/filtered_Madry_no_maxpool_tiny.pkl".format(model_name)
        dataset = "mnist"
        timeout = 3600
        mean = [0.0]
        std = [1.0]
    elif model_name == "cifar_cnn_b_adv":
        model = cnn_4layer_b()
        state_dict = torch.load("../../complete_verifier/models/sdp/cifar_cnn_b_adv.model")
        dataset = "cifar"
        mean = [125.3/255, 123.0/255, 113.9/255]
        std = [63.0/255, 62.1/255, 66.7/255]
        timeout = 3600
    elif model_name == "cifar_lpd_cnn_a":
        model = cifar_model_wide()
        state_dict = torch.load("./cifar_small_2px.pth")["state_dict"][0]
        pkl_name = "data/attack_filters/filtered_cifar_small_2px.pkl".format(model_name)
        dataset = "cifar"
        mean=[0.485, 0.456, 0.406]
        std=[0.225, 0.225, 0.225]
        timeout = 3600
    # elif model_name == "model_resnet":
    #     model = model_resnet()
    #     state_dict = torch.load("data/cifar_resnet_8px.pth")["state_dict"][0]
    #     dataset = "cifar"
    #     mean=[0.485, 0.456, 0.406]
    #     std=[0.225, 0.225, 0.225]
    elif model_name == "cifar_cnn_a_adv":
        model = cnn_4layer_adv()
        state_dict = torch.load("../../complete_verifier/models/sdp/cifar_cnn_a_adv.model")
        std=[0.2471, 0.2435, 0.2616]
        mean=[0.4914, 0.4824, 0.4467]
        dataset = "cifar"
        timeout = 3600
    elif model_name == "cifar_cnn_a_mix":
        model = cnn_4layer()
        state_dict = torch.load("../../complete_verifier/models/sdp/cifar_cnn_a_mix.model")
        std=[0.2471, 0.2435, 0.2616]
        mean=[0.4914, 0.4824, 0.4467]
        timeout = 3600
        dataset = "cifar"
    # elif model_name == "cifar_conv_small":
    #     model = cifar_conv_small()
    #     state_dict = torch.load("../../complete_verifier/models/eran/cifar_conv_small_pgd.pth")["state_dict"][0]
    #     mean = [0.4914, 0.4822, 0.4465]
    #     std = [0.2023, 0.1994, 0.201]
    #     pkl_name = "data/attack_filters/filtered_cifar_conv_small.pkl"
    #     dataset = "cifar"
    # elif model_name=="oval_base":
    #     model = cifar_model_base()
    #     state_dict = torch.load("../../complete_verifier/models/oval/cifar_base.pth")["state_dict"][0]
    #     std = [0.225, 0.225, 0.225]
    #     mean = [0.485, 0.456, 0.406]
    #     pkl_name = "data/attack_filters/filtered_oval_base.pkl"
    #     dataset = "cifar"
    # elif model_name=="oval_deep":
    #     model = cifar_model_deep()
    #     state_dict = torch.load("../../complete_verifier/models/oval/cifar_deep.pth")["state_dict"][0]
    #     std = [0.225, 0.225, 0.225]
    #     mean = [0.485, 0.456, 0.406]
    #     pkl_name = "data/attack_filters/filtered_oval_deep.pkl"
    #     dataset = "cifar"
    # elif model_name=="oval_wide":
    #     model = cifar_model_wide()
    #     state_dict = torch.load("../../complete_verifier/models/oval/cifar_wide.pth")["state_dict"][0]
    #     std = [0.225, 0.225, 0.225]
    #     mean = [0.485, 0.456, 0.406]
    #     dataset = "cifar"
    #     pkl_name = "data/attack_filters/filtered_oval_wide.pkl"
    elif model_name=="mnist_conv_small":
        model = mnist_conv_small()
        state_dict = torch.load("../../complete_verifier/models/eran/mnist_conv_small_nat.pth")["state_dict"][0]
        dataset = "mnist"
        timeout = 3600
        mean = [0.1307]
        std = [0.3081]
    elif model_name == "cifar_cnn_a_adv_alt":
        model = cnn_4layer_adv()
        state_dict = torch.load("../../complete_verifier/models/sdp/cifar_cnn_a_adv_alt.model")
        dataset = "cifar"
        pkl_name = "data/attack_filters/filtered_cifar_cnn_a_adv_alt.pkl"
        timeout = 3600
        std = [0.2471, 0.2435, 0.2616]
        mean = [0.4914, 0.4824, 0.4467]
    elif model_name == "cifar_marabou_small":
        model = cifar_marabou_small()
        state_dict = torch.load("../../complete_verifier/models/marabou_cifar10/cifar_marabou_small.pth")
        dataset = "cifar"
        std = [1.0, 1.0, 1.0]
        mean = [0.0, 0.0, 0.0]
        timeout = 3600
    # elif model_name == "cifar_marabou_medium":
    #     model = cifar_marabou_medium()
    #     state_dict = torch.load("../../complete_verifier/models/marabou_cifar10/cifar_marabou_medium.pth")
    #     dataset = "cifar"
    #     pkl_name = "data/attack_filters/filtered_cifar_marabou_medium.pkl"
    #     std = [1.0, 1.0, 1.0]
    #     mean = [0.0, 0.0, 0.0]


    model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"
    model = model.to(device)
    model.eval()

    # data
    if dataset == "mnist":
        test_data = torchvision.datasets.MNIST("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())
        test_data = torch.utils.data.DataLoader(test_data, batch_size=1, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),4))

        mean = torch.Tensor(mean).view((1,1,1,1)).to(device)
        std = torch.Tensor(std).view((1,1,1,1)).to(device)
        model = model.to(device)

        model = Normalization(mean, std, model)

    else:
        test_data = torchvision.datasets.CIFAR10("./data", train=False, download=True,
                transform=transforms.Compose([transforms.ToTensor()]))

        test_data = torch.utils.data.DataLoader(test_data, batch_size=1, pin_memory=True, num_workers=min(multiprocessing.cpu_count(),4))

        mean = torch.Tensor(mean).view((1,3,1,1)).to(device)
        std = torch.Tensor(std).view((1,3,1,1)).to(device)
        model = model.to(device)

        model = Normalization(mean, std, model)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    pkl_data = pkl.load(open(pkl_name, "rb"))

    if task == "test":
        true_label = []
        input_data = []
        for (data, label) in test_data:
            true_label.append(label)
            input_data.append(data)
        true_label = torch.cat(true_label)
        data = torch.cat(input_data, dim=0)
        diversed_pgd_attack = torch.load("my_data/{}_diversed_pgd_attack_result.torch".format(model_name))
        pgd_attack = torch.load("my_data/{}_pgd_attack_result.torch".format(model_name))
        auto_attack = torch.load("my_data/{}_auto_attack_result.torch".format(model_name))

        diversed_pgd_margin = torch.load("my_data/{}_diversed_pgd_attack_margin.torch".format(model_name))
        pgd_margin = torch.load("my_data/{}_pgd_attack_margin.torch".format(model_name))
        auto_margin = torch.load("my_data/{}_auto_attack_margin.torch".format(model_name))

        pkl_data = pkl.load(open(pkl_name, "rb"))

        print(((true_label.cpu().numpy() != diversed_pgd_attack.cpu().numpy())).sum())
        print(((true_label.cpu().numpy() != pgd_attack.cpu().numpy())).sum())
        print(((true_label.cpu().numpy() != auto_attack.cpu().numpy())).sum())
        print(((true_label.cpu() != model(data.to("cuda")).cpu().argmax(-1))).sum())

        error = ((true_label.cpu() != auto_attack.cpu()) & (true_label.cpu() == pgd_attack.cpu())).nonzero()
        for idx in error:
            print(pgd_margin[idx], auto_margin[idx], true_label[idx], pgd_attack[idx], auto_attack[idx], pkl_data["bab_verified"].iloc[idx])
        print(pgd_margin[3], auto_margin[3], true_label[3])

        diversed_result = true_label.cpu() == diversed_pgd_attack.cpu()
        pgd_result = true_label.cpu() == pgd_attack.cpu()
        auto_attack_result = true_label.cpu() == auto_attack.cpu()
        clean_result = (true_label.cpu() == model(data.to("cuda")).cpu().argmax(-1))

        pkl_data["classify_correct"] = 0
        pkl_data["classify_correct"][np.flatnonzero((clean_result==True).detach().cpu().numpy())] = 1
        pkl_data["dpgd_success"] = 0
        pkl_data["dpgd_success"][np.flatnonzero(((diversed_result==False) & (clean_result==True)).detach().cpu().numpy())]=1

        pkl_data["pgd_success"] = 0
        pkl_data["pgd_success"][np.flatnonzero(((pgd_result==False) & (clean_result==True)).detach().cpu().numpy())]=1

        pkl_data["auto_success"] = 0
        pkl_data["auto_success"][np.flatnonzero(((auto_attack_result==False) & (clean_result==True)).detach().cpu().numpy())]=1

        # pkl_data["bab_verified"] = 0
        # with open("verifier_log_{}/verify_success.txt".format(model_name)) as f:
        #     line = f.readline()
        #     while line != "":
        #         line = line.strip()
        #         if line != "":
        #             pkl_data["bab_verified"][int(line)] = 1
        #         line = f.readline()

        # pkl_data["pgd_in_bab_success"] = 0
        # with open("verifier_log_{}/verify_attack.txt".format(model_name)) as f:
        #     line = f.readline()
        #     while line != "":
        #         line = line.strip()
        #         if line != "":
        #             pkl_data["pgd_in_bab_success"][int(line)] = 1
        #         line = f.readline()

        print(pkl_data.columns)
        attack_verify = pkl_data["classify_correct"] & (pkl_data["dpgd_success"]| pkl_data["pgd_success"] | pkl_data["auto_success"] | pkl_data["bab_verified"])
        #pkl_data["pgd_in_bab_success"] |
        remain = pkl_data["classify_correct"] & (~attack_verify)
        print(remain.sum())

        # pkl_data["need_attack"] = 0
        # pkl_data["need_attack"][np.flatnonzero(remain.to_numpy())] = 1
        # pkl_data.drop(columns=["need_attack"])

        with open("data/attack_ids/{}.txt".format(model_name), "w+") as f:
            for k in np.flatnonzero(remain.to_numpy()):
                f.write(str(k)+'\n')

        mip_safe, mip_unknown, mip_unsafe, mip_time, label_time, attack_margin = analyze_log("mip_log_{}/".format(model_name))

        pkl_data["mip_status"] = -1
        pkl_data["mip_status"][mip_safe] = 1
        pkl_data["mip_status"][mip_unsafe] = 2
        pkl_data["mip_status"][mip_unknown] = 3


        # mip_time = list(mip_time.items())

        mip_time = list(recal_time(label_time, attack_margin, timeout).items())
        pkl_data["mip_time"] = 0
        pkl_data["mip_time"][[a[0] for a in mip_time]] = [a[1] for a in mip_time]
        pkl_data.to_pickle(pkl_name)

        # import pdb; pdb.set_trace()

        print(pkl_data.iloc[65])


        margin = torch.load("my_data/{}_diversed_pgd_attack_margin.torch".format(model_name))
        margin[margin == 0.] = float('inf')
        margin = margin.min(-1)
        margin_sort = np.argsort(margin)

        with open("../../complete_verifier/exp_configs/bab_attack/attack_idx/{}/mip_unsafe_idx.txt".format(model_name), "w+") as f:
            for i in margin_sort:
                if i in mip_unsafe and remain[i]:
                    f.write(str(i) + "\n")

        with open("../../complete_verifier/exp_configs/bab_attack/attack_idx/{}/mip_unknown_idx.txt".format(model_name), "w+") as f:
            for i in margin_sort:
                if i in mip_unknown and remain[i]:
                    f.write(str(i) + "\n")


        pkl.dump(label_time, open(pkl_name[:-4] + ".dict", "wb+"))
    else:

        preds = []
        margin = []
        clean_pred = []
        robust_acc = 0
        standard_acc = 0
        total = 0
        for i, (data, label) in enumerate(test_data):
            if i % 100 == 0:
                print(i)
            image = data
            true_label = label
            image = image.to(torch.float32)
            image = image.to(device)

            if pkl_data["bab_verified"].iloc[i] == 1:
                # print("bab_verification success.")
                standard_acc += 1
                robust_acc += 1

                l = model(image)

                pred = torch.argmax(l, -1)
                preds.append(pred.detach().cpu())

                clean = model(image)
                clean_pred.append(clean.detach().cpu())

                l = l.detach().cpu().numpy()
                for idx, logit in enumerate(l):
                    l[idx]=logit[label[idx]]-l[idx]
                margin.append(l)
                continue


            true_label = true_label.to(device)

            # lirpa_model = BoundedModule(model, torch.empty_like(image), device=device)

            if dataset == 'mnist':
                eps = torch.Tensor([0.12]).view((1,1,1,1)).to(device)
            else:
                eps = torch.Tensor([2./255]*3).view((1,3,1,1)).to(device)

            adv_data = image
            adv_data = eval(method)(model, eps.max(), image, true_label, 1)
            assert (adv_data >= 0.0).all()
            assert (adv_data <= 1.0).all()

            l = model(adv_data)

            pred = torch.argmax(l, -1)
            preds.append(pred.detach().cpu())
            assert (adv_data-image).abs().max() <= eps.max() + 0.0001, f"{(adv_data-image).abs().max()} <= {eps.max()}"

            clean = model(image)
            clean_pred.append(clean.detach().cpu())

            l = l.detach().cpu().numpy()
            for idx, logit in enumerate(l):
                l[idx]=logit[label[idx]]-l[idx]
            margin.append(l)
            total += true_label.detach().cpu().shape[0]
            robust_acc += (true_label == pred).detach().cpu().int().sum()
            standard_acc += (true_label == clean.argmax(-1)).detach().cpu().int().sum()

        print("robust acc: ", robust_acc/total)
        print("clean acc: ", standard_acc/total)

        pred = torch.cat(preds)
        clean_pred = torch.cat(clean_pred, axis=0)
        margin = np.concatenate(margin, axis=0)
        print(margin[4024])
        print(clean_pred[4024])

        torch.save(pred, "my_data/{}_{}_result.torch".format(model_name, method))

        # save the data
        torch.save(adv_data, "my_data/{}_{}_adv_data.torch".format(model_name, method))
        torch.save(margin, "my_data/{}_{}_margin.torch".format(model_name, method))
        torch.save(clean_pred, "my_data/{}_{}_clean_pred.torch".format(model_name, method))


if __name__ == "__main__":
    models = ["mnist_cnn_a_adv"]
    # "mnist_cnn_a_adv"

    # "mnist_small_adv"

    # "cifar_cnn_b_adv"

    # "cifar_cnn_a_mix"

    # "cifar_marabou_small"

    # "cifar_cnn_a_adv_alt"

    # "cifar_cnn_a_adv"

    # "mnist_conv_small"

    # "cifar_lpd_cnn_a"
    for m in models:
        model_name = m
        method = "auto_attack"
        main(model_name, method, "test")
        print(model_name + " {} done".format(method))