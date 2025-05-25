import random, sys, time, multiprocessing
import argparse
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from auto_LiRPA import BoundedModule, BoundedTensor, BoundedParameter
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import MultiAverageMeter
from auto_LiRPA.eps_scheduler import LinearScheduler, AdaptiveScheduler, SmoothedScheduler, FixedScheduler
from datasets import loaders
import torch.nn.functional as F

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
import copy

## Step 1: Initial original model as usual, see model details in models/sample_models.py
class mlp_MNIST(nn.Module):
    def __init__(self, in_ch=1, in_dim=28, width=1):
        super(mlp_MNIST, self).__init__()
        self.fc1 = nn.Linear(in_ch * in_dim * in_dim, 64 * width)
        self.fc2 = nn.Linear(64 * width, 64 * width)
        self.fc3 = nn.Linear(64 * width, 10)

        eps = 0.01
        norm = 2
        global ptb
        ptb = PerturbationLpNorm(norm=norm, eps=eps)
        self.fc1.weight = BoundedParameter(self.fc1.weight.data, ptb)
        self.fc2.weight = BoundedParameter(self.fc2.weight.data, ptb)
        self.fc3.weight = BoundedParameter(self.fc3.weight.data, ptb)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

parser = argparse.ArgumentParser()

parser.add_argument("--verify", action="store_true", help='verification mode, do not train')
parser.add_argument("--load", type=str, default="", help='Load pretrained model')
parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help='use cpu or cuda')
parser.add_argument("--data", type=str, default="MNIST", choices=["MNIST", "FashionMNIST"], help='dataset')
parser.add_argument("--ratio", type=float, default=None, help='percent of training used, None means whole training data')
parser.add_argument("--seed", type=int, default=150, help='random seed')
parser.add_argument("--eps", type=float, default=0.05, help='epsilon perturbation on weights')
parser.add_argument("--norm", type=float, default='inf', help='p norm for epsilon perturbation on weights')
parser.add_argument("--bound_type", type=str, default="CROWN-IBP",
                    choices=["IBP", "CROWN-IBP", "CROWN"], help='method of bound analysis')
parser.add_argument("--num_epochs", type=int, default=150, help='number of total epochs')
parser.add_argument("--batch_size", type=int, default=256, help='batch size')
parser.add_argument("--lr", type=float, default=5e-4, help='learning rate')
parser.add_argument("--weight_decay", type=float, default=0.01, help='L2 penalty of weights')
parser.add_argument("--scheduler_name", type=str, default="LinearScheduler",
                    choices=["LinearScheduler", "AdaptiveScheduler", "SmoothedScheduler"], help='epsilon scheduler')
parser.add_argument("--scheduler_opts", type=str, default="start=5,length=120", help='options for epsilon scheduler')
parser.add_argument("--bound_opts", type=str, default=None, choices=["same-slope", "zero-lb", "one-lb"],
                    help='bound options')

args = parser.parse_args()

def main():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ## Step 1: Initial original model as usual; note that this model has BoundedParameter as its weight parameters
    model = mlp_MNIST()
    if args.load:
        state_dict = torch.load(args.load)['state_dict']
        model.load_state_dict(state_dict)

    model.train()
    ## Step 2: Prepare dataset as usual
    dummy_input = torch.randn(1, 1, 28, 28)
    train_data, test_data = loaders[args.data](batch_size=args.batch_size, shuffle_train=True, ratio=args.ratio)

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    test_input = []
    test_labels = []
    
    for i, (data, labels) in enumerate(train_data):
        test_input.append(data)
        test_labels.append(labels)
    
    test_input = torch.cat(test_input, 0)
    test_labels = torch.cat(test_labels, 0)

    opt.zero_grad()
    output = model(test_input)

    regular_ce = CrossEntropyLoss()(output, test_labels)
    regular_ce.backward()

    parameters = model.parameters()

    para_grad = []
    parameters = []
    for p in model.parameters():
        parameters.append(copy.deepcopy(p.data))
        para_grad.append(p.grad)
    grad_sum = torch.cat([p.view(-1) for p in para_grad], 0).view(-1)
    grad_norm = torch.norm(grad_sum)

    grad_direct = [w/grad_norm for w in para_grad]
    random_direct = []

    for p in parameters:
        random_direct.append(torch.rand(p.shape))
    random_norm = torch.norm(torch.cat([p.view(-1) for p in random_direct]).view(-1))
    random_direct = [p/random_norm for p in random_direct]

    X = np.linspace(-1, 1, 50)
    Y = np.linspace(-1, 1, 50)

    X,Y = np.meshgrid(X,Y)
    Z = np.zeros(X.shape)
    for i in range(len(X)):
        for j in range(len(X[0])):
            x = X[i][j]
            y = Y[i][j]

            for ip, p in enumerate(model.parameters()):
                p.data = parameters[ip] + torch.Tensor(x * grad_direct[ip]) + torch.Tensor(y * random_direct[ip])

            output = model(test_input)
            regular_ce = CrossEntropyLoss()(output, test_labels)

            Z[i][j] += regular_ce.item()

    ticks = np.arange(0, 5, 1)
    xticks = np.arange(-1, 1.1, 0.5)

    pdf = PdfPages('test.pdf')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z)
    ax.set_zticks(ticks)
    ax.set_xticks(xticks)
    ax.set_yticks(xticks)

    ax.set_xlabel('gradient direction', fontsize = 17)
    ax.set_ylabel('random direction', fontsize = 17)
    ax.set_zlabel('Loss', fontsize = 17)

    pdf.savefig()
    plt.close()
    pdf.close()

if __name__ == "__main__":
    main()
