import os
import sys
import torch
import time
import argparse
sys.path.append("../")

from models import *
from utils import *
    
def verified_lipnaive(dataset, labels, model, radius, clean_output, device, classes, args):
    samples = dataset.shape[0]
    verification_fail = samples - len(clean_output)
    verification_fail_idx = []
    log_dir = f'./logs/lipnaive/{args.model.lower()}/{args.radius}'
    os.makedirs(log_dir, exist_ok=True)

    # Estimate global Lipschitz constant with respect to each class
    start_time = time.time()
    lip_class = torch.empty(classes, 1, classes-1).to(device)
    global_input = dataset[0].unsqueeze(0).to(device)
    current_lipschitz_constant = 1
    for idx, layer in enumerate(model.modules()):
        if isinstance(layer, nn.Linear):
            weight = layer.weight.data
            if idx == len(list(model.modules())) -1:
                for i in range(classes):
                    C = build_C(torch.tensor([i]).to(device), classes)
                    lip_class[i] = current_lipschitz_constant * C.matmul(weight).norm(dim=2)
                    print(f'Lipschitz constant bound with respect to class {i}: {lip_class[i]}')
            else:
                _, S, _ = torch.svd(weight)
                current_lipschitz_constant *= S[0].item()
                print(f'Lipschitz constant bound up to the {layer}: {current_lipschitz_constant}')
        
        elif isinstance(layer, nn.Conv2d):
            current_lipschitz_constant *= power_iteration(layer, global_input.shape)
            global_input = layer(global_input)
            print(f'Lipschitz constant bound up to the {layer}: {current_lipschitz_constant}')
    end_time = time.time()
    elapsed_time = end_time - start_time

    for idx, (image, label) in enumerate(zip(dataset, labels)):
        if idx not in clean_output:
            continue
        
        sample_idx = args.start + idx
        verifiction_status = "Success"
        image = image.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)
        C = build_C(label, classes)
        lip_lb = (C*model(image)).sum(dim=2) - radius*lip_class[label.squeeze(0)]

        with torch.no_grad():
            if torch.any(lip_lb < 0):
                verification_fail += 1
                verifiction_status = "Fail"
                verification_fail_idx.append(sample_idx)
            sample_log = {
                'sample_idx': sample_idx,
                'true_label': label.item() if isinstance(label, torch.Tensor) else label,
                'margins': lip_lb.cpu().tolist()[0],       
                'verifiction_status': verifiction_status,
                'elapsed_time': elapsed_time,
            }
            with open(f'{log_dir}/sample_{sample_idx}.log', "w", encoding='utf-8') as f:
                for key, val in sample_log.items():
                    f.write(f"{key}: {val}\n") 
            print(f'Sample {sample_idx}, verifiction_status: {verifiction_status}, elapsed_time: {elapsed_time}s')
    
    verified_accuracy = (samples-verification_fail)/samples*100
    final_log = {
        'verification_fail_idx': verification_fail_idx,
        'verification_fail': verification_fail,
        'verified_accuracy': verified_accuracy,
        'elapsed_time': elapsed_time,
    }
    with open(f'{log_dir}/final_results.log', "w", encoding='utf-8') as f:
        for key, val in final_log.items():
            f.write(f"{key}: {val}\n")         
    print(f'Total Verification Fail: {verification_fail}, verified_accuracy: {(samples-verification_fail)/samples*100}%, elapsed_time: {elapsed_time}s')
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
    
    verified_lipnaive(
        dataset = dataset, 
        labels = labels, 
        model = model, 
        radius = radius_rescale, 
        clean_output = correct_indices, 
        device = device, 
        classes = classes, 
        args = args
        )
