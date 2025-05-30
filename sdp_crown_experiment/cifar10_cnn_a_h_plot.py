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




args.model = 'cifar10_cnn_a'
num_sample = 10
radii = [0.0, 0.025, 0.05, 0.075, 0.1, 
              0.125, 0.15, 0.175, 0.2, 
              0.225, 0.25, 0.275, 0.3]
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
    verified_sdp_crown(dataset = dataset, labels = labels, model = model, radius = radius_rescale, clean_output = correct_indices, device = device, classes = classes, args = args)  
