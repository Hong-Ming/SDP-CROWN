import torch
import argparse

from models import *
from utils import *
from pgd import *
from sdp_crown import *
from alpha_crown import *
from lipnaive import *
from lp_all import *

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--radius', default=1, type=parse_float_or_fraction, help='L2_norm range')
parser.add_argument('--lr_alpha', default=0.5, type=float, help='alpha learning rate')
parser.add_argument('--lr_lambda', default=0.05, type=float, help='lmabda learning rate')
parser.add_argument('--start', default=0, type=int, help='start_index')
parser.add_argument('--end', default=200, type=int, help='end_index')
parser.add_argument('--sr', default=0, type=int, help='end_index')
args = parser.parse_args()




args.model = 'cifar10_convsmall'
num_sample = 10
# radii = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
radii = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95]
radii = radii[args.sr:]



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
    attack_pgd(dataset = dataset, labels = labels, model = model, radius = radius_rescale, clean_output = correct_indices, device = device, classes = classes, args = args)  
    verified_sdp_crown(dataset = dataset, labels = labels, model = model, radius = radius_rescale, clean_output = correct_indices, device = device, classes = classes, args = args)  
    verified_alpha_crown(dataset = dataset, labels = labels, model = model, radius = radius_rescale, clean_output = correct_indices, device = device, classes = classes, args = args)
    verified_lipnaive(dataset = dataset, labels = labels, model = model, radius = radius_rescale, clean_output = correct_indices, device = device, classes = classes, args = args)
    # verified_lp_all(dataset = dataset, labels = labels, model = model, radius = radius_rescale, clean_output = correct_indices, device = device, classes = classes, args = args)
