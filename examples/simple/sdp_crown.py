import sys
# sys.path.append("/Users/hongming/Documents/Git/Verifier_Development")
sys.path.append("../")
import torch
import torch.nn as nn
import numpy as np
from models import *
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import time
import argparse

# Help function to generate C matrix for calculate
# the margins.
def build_C(label, classes):
    """
    label: shape (B,). Each label[b] in [0..classes-1].
    Return:
        C: shape (B, classes-1, classes).
        For each sample b, each row is a “negative class” among [0..classes-1]\{label[b]}.
        Puts +1 at column=label[b], -1 at each negative class column.
    """
    device = label.device
    batch_size = label.size(0)
    
    # 1) Initialize
    C = torch.zeros((batch_size, classes-1, classes), device=device)
    
    # 2) All class indices
    # shape: (1, K) -> (B, K)
    all_cls = torch.arange(classes, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # 3) Negative classes only, shape (B, K-1)
    # mask out the ground-truth
    mask = all_cls != label.unsqueeze(1)
    neg_cls = all_cls[mask].view(batch_size, -1)
    
    # 4) Scatter +1 at each sample’s ground-truth label
    #    shape needed: (B, K-1, 1)
    pos_idx = label.unsqueeze(1).expand(-1, classes-1).unsqueeze(-1)
    C.scatter_(dim=2, index=pos_idx, value=1.0)
    
    # 5) Scatter -1 at each row’s negative label
    #    We have (B, K-1) negative labels. For row j in each sample b, neg_cls[b, j] is that row’s negative label
    row_idx = torch.arange(classes-1, device=device).unsqueeze(0).expand(batch_size, -1)
    # shape: (B, K-1)
    
    # We can do advanced indexing:
    C[torch.arange(batch_size).unsqueeze(1), row_idx, neg_cls] = -1.0
    
    return C
    
def verified_sdp_crown(dataset, labels, model, radius, clean_output, device, classes, lr_alpha, lr_lambda):
    samples = dataset.shape[0]
    attack_success = samples - len(clean_output)
    total_time = 0
    margin_avg = 0
    final_log = {
        'attack_success_samples': {},
        'average_time': 0,
    }
    for idx, (x_test, label) in enumerate(zip(dataset, labels)):
        if idx not in clean_output:
            continue
        sample_log = {
            'sample_idx': idx,
            'true_label': label.item() if isinstance(label, torch.Tensor) else label,
            'margins': [],       
            'attack_success': False,
            'elapsed_time': None,
        }
        start_time = time.time()
        x_test = x_test.unsqueeze(0).to(device)
        label = label.to(device).unsqueeze(0)
        norm = 2.0
        ptb = PerturbationLpNorm(norm=norm, eps=radius)
        image = BoundedTensor(x_test, ptb)
        method = 'CROWN-Optimized'
        lirpa_model = BoundedModule(model, image, device=image.device,
                                verbose=0)
        C = build_C(label, classes)
        lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 300, 'lr_alpha': lr_alpha, 'early_stop_patience': 20, 'fix_interm_bounds': False, 'enable_opt_interm_bounds':True, 'enable_SDP_crown': True, 'lr_lambda_out': lr_lambda}})
        crown_lb, _ = lirpa_model.compute_bounds(x=(image,), method=method.split()[0], C=C, bound_lower=True, bound_upper=False)
        with torch.no_grad():
            margin_avg += torch.mean(crown_lb).item()
            if torch.any(crown_lb < 0):
                attack_success += 1
                sample_log['attack_success'] = True
                final_log['attack_success_samples'][idx] = (torch.min(crown_lb))
            end_time = time.time()
            total_time += end_time - start_time
            print(f'Sample {idx}, verify_time: {end_time - start_time} s')
            sample_log['margins'] = crown_lb.cpu().tolist()  
            sample_log['elapsed_time'] = end_time - start_time
            total_time += end_time - start_time
            with open(f'./logs/MLP_MNIST/sample_{idx}_alpha_{lr_alpha}_lambda_{lr_lambda}.log', "w", encoding='utf-8') as f:
                for key, val in sample_log.items():
                    f.write(f"{key}: {val}\n") 
    final_log['average_time'] = total_time / len(clean_output)
    with open(f'./logs/MLP_MNIST/alpha_{lr_alpha}_lambda_{lr_lambda}.log', "w", encoding='utf-8') as f:
        for key, val in final_log.items():
            f.write(f"{key}: {val}\n")         
    print(f'Total Attack Success: {attack_success}, verified_accuracy: {(samples-attack_success)/samples*100}%')
    return 

# You can create you own models here.
def MLP():
	net = nn.Sequential(
		nn.Flatten(),
		nn.Linear(784, 100),
		nn.ReLU(),
		nn.Linear(100, 100),
		nn.ReLU(),
		nn.Linear(100, 10)
	)
	return net

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--radius', default=1, type=float, help='L2_norm range')
    parser.add_argument('--lr_alpha', default=0.5, type=float, help='alpha learning rate')
    parser.add_argument('--lr_lambda', default=0.05, type=float, help='lmabda learning rate')
    parser.add_argument('--start', default=0, type=int, help='start_index')
    parser.add_argument('--end', default=200, type=int, help='end_index')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Load model.
    model = MLP().to(device)
    checkpoint = torch.load('./experiments_weight/mlp.pth',map_location=device)

    model.load_state_dict(checkpoint)
    model.eval()
    
    # Load data.
    dataset = np.load('../../complete_verifier/datasets/sdp/mnist/X_sdp.npy')
    labels = np.load('../../complete_verifier/datasets/sdp/mnist/y_sdp.npy')
    dataset = torch.from_numpy(dataset).permute(0,3,1,2)
    labels = torch.from_numpy(labels)
    range_ = args.radius
    classes = 10

    batch_size = 1
    start = args.start
    end = args.end

    clean_output = 0
    correct = 0 
    dataset = dataset[start:end]
    labels = labels[start:end]
    samples = dataset.shape[0]

    # Run original model for clean accuracy.
    with torch.no_grad():
        labels_tensor = labels.to(device)
        dataset_tensor = dataset.to(device)
        output = model(dataset_tensor)
        clean_output = torch.sum((output.max(1)[1] == labels_tensor).float()).cpu()
        predictions = output.argmax(dim=1)
        correct_indices = (predictions == labels_tensor).nonzero(as_tuple=True)[0]
        print(correct_indices)
    print(f'perturbation: {range_}')
    print(f'The clean output for the {end-start} samples is {clean_output/samples*100}%')
    
    verified_sdp_crown(dataset=dataset, labels=labels, model=model, radius=range_, clean_output=correct_indices, device=device, classes=classes, lr_alpha=args.lr_alpha, lr_lambda=args.lr_lambda)
