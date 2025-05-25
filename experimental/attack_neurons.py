from src.model_defs import mnist_cnn_4layer
from auto_LiRPA import BoundedModule, BoundedTensor
import torch
import torchvision
from autoattack import AutoAttack
import random
from src.attack_pgd import attack_pgd
from plnn.advmnist_models.adv_models import *
import numpy as np
import matplotlib.pyplot as plt


def auto_attack(model_ori, epsilon, images, labels):
    adversary = AutoAttack(model_ori, norm='Linf', eps=epsilon, version='standard')

    ans = []
    for i in range(100):
        adversary.seed = random.randint(0,10000)
        x_adv = adversary.run_standard_evaluation(images, labels, bs=16)
        ans.append(x_adv)

    return torch.cat(ans, dim=1)

def diversity_pgd(model_ori, epsilon, images, labels):
    ans = []
    for i in range(100):
        best_delta, _ = attack_pgd(model_ori, images, labels, epsilon, epsilon/2, 50, 5, OSI_init_X=True)
        ans.append(best_delta + images)
    return torch.cat(ans, dim=1)
    

model = "cnn_4_layer"
if __name__ == "__main__":
    random.seed(100)
    if model == "cnn_4_layer":
        model = mnist_cnn_4layer()
        state_dict = torch.load("plnn/sdp_models/mnist_0.3_cnn_a_adv.model")
    elif model == "madrycnn_no_maxpool_tiny":
        model = MadryCNN_no_maxpool_tiny()
        state_dict = torch.load("plnn/advmnist_models/MadryCNN_no_maxpool_tiny.pt")


    model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"
    model = model.to(device)

    # data
    test_data = torchvision.datasets.MNIST("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())
    N = 100
    indices = []
    for i in range(N):
        indices.append(random.randint(0,len(test_data)))

    image = test_data.data[indices].view(N,1,28,28)
    true_label = test_data.targets[:N]
    image = image.to(torch.float32) / 255.0
    image = image.to(device)
    true_label = true_label.to(device)

    lirpa_model = BoundedModule(model, torch.empty_like(image), device=device)

    adv_data = diversity_pgd(model, 0.3, image, true_label)

    ## save the data
    # torch.save(adv_data, "adv_data.torch")
    

    N = 100
    for n in range(1):
        relus = []
        for m in model.modules():
            if isinstance(m, torch.nn.ReLU):
                relus.append([None,None,None])

        x_adv = adv_data[n].unsqueeze(1)
        x_adv.to(device)

        x = x_adv
        
        relu_idx = 0
        for module in model.children():
            if isinstance(module, torch.nn.ReLU):
                if relus[relu_idx][0] is None:
                    relus[relu_idx] = [torch.zeros_like(x_adv.flatten()), torch.zeros_like(x_adv.flatten()), torch.zeros_like(x_adv.flatten())]
                relus[relu_idx][0] += x_adv.flatten() > 0
                relus[relu_idx][1] += x_adv.flatten() == 0
                relus[relu_idx][2] += x_adv.flatten() < 0

                relu_idx += 1
            x_adv = module(x_adv)

        for i in range(len(relus)):
            relu_sum = sum(relus[i])
            relus[i] = [list((r/relu_sum).detach().cpu().numpy()) for r in relus[i]]


        for i, relu in enumerate(relus):
            plt.hist(relu[0])
            plt.savefig("layer{}.png".format(i))
            plt.clf()