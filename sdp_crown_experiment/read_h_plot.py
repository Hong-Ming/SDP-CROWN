import os
import sys
import torch
import time
import argparse
import ast
sys.path.append("../")

import numpy as np
from models import *
from utils import *

def read_log_file(log_file):
    file_dict = {}
    if os.path.exists(log_file):
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()

                # convert to native types
                if key in ("sample_idx", "true_label"):
                    file_dict[key] = int(val)
                elif key == "elapsed_time":
                    file_dict[key] = float(val)
                elif key in ("margins", "sdp_offset", "alpha_offset"):
                    # file_dict[key] = np.array(ast.literal_eval(val))   # turns "[...]" into a Python list
                    file_dict[key] = ast.literal_eval(val)   # turns "[...]" into a Python list
                else:
                    file_dict[key] = val
    else:
        file_dict["margins"] = [-np.inf]
    return file_dict

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--radius', default=1, type=parse_float_or_fraction, help='L2_norm range')
parser.add_argument('--lr_alpha', default=0.5, type=float, help='alpha learning rate')
parser.add_argument('--lr_lambda', default=0.05, type=float, help='lmabda learning rate')
parser.add_argument('--start', default=0, type=int, help='start_index')
parser.add_argument('--end', default=200, type=int, help='end_index')
args = parser.parse_args()


# args.model = 'mnist_mlp'
# args.model = 'mnist_convsmall'
# args.model = 'mnist_convlarge'
# num_sample = 10
# radii = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# args.model = 'cifar10_cnn_a'
# args.model = 'cifar10_convsmall'
args.model = 'cifar10_convlarge'
num_sample = 10
radii = [0.0, 0.025, 0.05, 0.075, 0.1, 
              0.125, 0.15, 0.175, 0.2, 
              0.225, 0.25, 0.275, 0.3]


pgd_data = []
sdp_crown_data = []
alpha_crown_data = []
lipnaive_data = []
lp_all_data = []

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model, dataset, labels, radius_rescale, classes = load_model_and_dataset(args, device)

# Run original model for clean accuracy.
with torch.no_grad():
    labels_tensor = labels.to(device)
    dataset_tensor = dataset.to(device)
    output = model(dataset_tensor)
    clean_output = torch.sum((output.max(1)[1] == labels_tensor).float()).cpu()
    predictions = output.argmax(dim=1)
    correct_indices = (predictions == labels_tensor).nonzero(as_tuple=True)[0]

args.end = correct_indices[num_sample]
correct_indices = correct_indices[:num_sample]
N = num_sample*(classes-1)
for radius in radii:
    args.radius = radius
    model, dataset, labels, radius_rescale, classes = load_model_and_dataset(args, device)
    
    sdp_crown_sum = 0
    alpha_crown_sum = 0
    for sample_idx in correct_indices:
        sdp_crown_sum += sum(read_log_file(f'./logs/sdp_crown/{args.model.lower()}/{args.radius}/sample_{sample_idx}.log')['sdp_offset'])
        alpha_crown_sum += sum(read_log_file(f'./logs/sdp_crown/{args.model.lower()}/{args.radius}/sample_{sample_idx}.log')['alpha_offset'])


    sdp_crown_data.append(sdp_crown_sum/N)
    alpha_crown_data.append(alpha_crown_sum/N)
sdp_crown_data[0] = 0.0
print(f'radius = {radii};')
print(f'{args.model}_sdp = {sdp_crown_data};')
print(f'{args.model}_alpha = {alpha_crown_data};')