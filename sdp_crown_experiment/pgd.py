import os
import sys
import torch
import time
import argparse
sys.path.append("../")

from models import *
from utils import *
    
def attack_pgd(dataset, labels, model, radius, clean_output, device, classes, args):
    samples = dataset.shape[0]
    verification_fail = samples - len(clean_output)
    verification_fail_idx = []
    total_time = 0
    log_dir = f'./logs/pgd/{args.model.lower()}'
    os.makedirs(log_dir, exist_ok=True)

    for idx, (image, label) in enumerate(zip(dataset, labels)):
        if idx not in clean_output:
            continue
        
        sample_idx = args.start + idx
        verifiction_status = "Success"
        image = image.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)
        C = build_C(label, classes)
        x_L, x_U = None, None
        if "mnist" in args.model.lower():
            x_U = torch.ones_like(image)
            x_L = torch.zeros_like(image)

        # Run pgd
        num_iter = 300
        pgd_ub = []
        start_time = time.time()
        for c in C.unbind(dim=1):
            X = nn.Parameter(torch.zeros_like(image).uniform_(-radius, radius)).to(device)
            optimizer = torch.optim.SGD([X], lr=0.01, momentum=0.9, nesterov=True)
            for _ in range(num_iter):
                optimizer.zero_grad()
                loss = torch.sum(c*model(X))   
                loss.backward()
                optimizer.step()
                # projection
                with torch.no_grad():
                    delta = X-image
                    d_norm = delta.reshape(-1).norm()
                    if d_norm > radius:
                        X.copy_(image + delta*radius/d_norm)
                    if x_U is not None:
                        X.clamp_(max=x_U)
                    if x_L is not None:
                        X.clamp_(min=x_L)
            pgd_ub.append(loss.item())     
        end_time = time.time()

        with torch.no_grad():
            if any(ub < 0 for ub in pgd_ub):
                verification_fail += 1
                verifiction_status = "Fail"
                verification_fail_idx.append(sample_idx)
            elapsed_time = end_time - start_time
            total_time += elapsed_time
            sample_log = {
                'sample_idx': sample_idx,
                'true_label': label.item() if isinstance(label, torch.Tensor) else label,
                'margins': pgd_ub,       
                'verifiction_status': verifiction_status,
                'elapsed_time': elapsed_time,
            }
            with open(f'{log_dir}/sample_{sample_idx}.log', "w", encoding='utf-8') as f:
                for key, val in sample_log.items():
                    f.write(f"{key}: {val}\n") 
            print(f'Sample {sample_idx}, verifiction_status: {verifiction_status}, elapsed_time: {elapsed_time}s')
    
    verified_accuracy = (samples-verification_fail)/samples*100
    average_time =  total_time/len(clean_output)
    final_log = {
        'verification_fail_idx': verification_fail_idx,
        'verification_fail': verification_fail,
        'verified_accuracy': verified_accuracy,
        'average_time': average_time,
    }
    with open(f'{log_dir}/final_results.log', "w", encoding='utf-8') as f:
        for key, val in final_log.items():
            f.write(f"{key}: {val}\n")         
    print(f'Total Verification Fail: {verification_fail}, verified_accuracy: {(samples-verification_fail)/samples*100}%, average_time: {average_time}s')
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--radius', default=1, type=parse_float_or_fraction, help='L2_norm range')
    parser.add_argument('--start', default=0, type=int, help='start_index')
    parser.add_argument('--end', default=200, type=int, help='end_index')
    parser.add_argument('--model', default='mnist_mlp',
    choices=[
        'mnist_mlp',
        'mnist_convsmall',
        'mnist_convlarge',
        'cifar10_cnn_a',
        'cifar10_cnn_b',
        'cifar10_cnn_c',
        'cifar10_convsmall',
        'cifar10_convdeep',
        'cifar10_convlarge',
        ])
    args = parser.parse_args()
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
    
    attack_pgd(
        dataset = dataset, 
        labels = labels, 
        model = model, 
        radius = radius_rescale, 
        clean_output = correct_indices, 
        device = device, 
        classes = classes, 
        args = args
        )
