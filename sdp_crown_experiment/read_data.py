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
# from pgd import *
# from sdp_crown import *
# from alpha_crown import *
# from lipnaive import *
# from lp_all import *

def read_log_file(log_file):
    file_dict = {}
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
            elif key == "margins":
                # file_dict[key] = np.array(ast.literal_eval(val))   # turns "[...]" into a Python list
                file_dict[key] = ast.literal_eval(val)   # turns "[...]" into a Python list
            else:
                file_dict[key] = val
    return file_dict

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--radius', default=1, type=parse_float_or_fraction, help='L2_norm range')
parser.add_argument('--lr_alpha', default=0.5, type=float, help='alpha learning rate')
parser.add_argument('--lr_lambda', default=0.05, type=float, help='lmabda learning rate')
parser.add_argument('--start', default=0, type=int, help='start_index')
parser.add_argument('--end', default=200, type=int, help='end_index')
args = parser.parse_args()


args.model = 'mnist_mlp'
num_sample = 10
radii = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
         1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]


pgd = []
sdp_crown = []
alpha_crown = []
lipnaive = []
lp_all = []

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
print(f'perturbation: {radius_rescale}')
print(f'The clean output for the {args.end-args.start} samples is {clean_output/(args.end-args.start)*100}%')

args.end = correct_indices[num_sample]
correct_indices = correct_indices[:num_sample]
for radius in radii:
    args.radius = radius
    model, dataset, labels, radius_rescale, classes = load_model_and_dataset(args, device)
    
    pgd_sum = 0
    sdp_crown_sum = 0
    alpha_crown_sum = 0
    lipnaive_sum = 0
    lp_all_sum = 0
    for sample_idx in correct_indices:
        pgd_file += read_log_file(f'./logs/pgd/{args.model.lower()}/{args.radius}/sample_{sample_idx}.log')
        sdp_crown_file += read_log_file(f'./logs/sdp_crown/{args.model.lower()}/{args.radius}/sample_{sample_idx}.log')
        alpha_crown_file += read_log_file(f'./logs/alpha_crown/{args.model.lower()}/{args.radius}/sample_{sample_idx}.log')
        lipnaive_file += read_log_file(f'./logs/lipnaive/{args.model.lower()}/{args.radius}/sample_{sample_idx}.log')
        lp_all_file += read_log_file(f'./logs/lp_all/{args.model.lower()}/{args.radius}/sample_{sample_idx}.log')