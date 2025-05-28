import os
import sys
import torch
import time
import argparse
import gurobipy as grb
sys.path.append("../")

from models import *
from utils import *
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.operators import BoundLinear, BoundConv
    
def verified_lp_all(dataset, labels, model, radius, clean_output, device, classes, args):
    samples = dataset.shape[0]
    verification_fail = samples - len(clean_output)
    verification_fail_idx = []
    total_time = 0
    log_dir = f'./logs/lp_all/{args.model.lower()}/{args.radius}'
    os.makedirs(log_dir, exist_ok=True)

    for idx, (image, label) in enumerate(zip(dataset, labels)):
        if idx not in clean_output:
            continue
        sample_idx = args.start + idx

        verifiction_status = "Success"
        image = image.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)
        norm = 2.0
        C = build_C(label, classes)
        x_L, x_U = None, None
        if "mnist" in args.model.lower():
            x_U = torch.ones_like(image)
            x_L = torch.zeros_like(image)
          
        ptb = PerturbationLpNorm(norm=norm, eps=radius, x_U=x_U, x_L=x_L)
        image = BoundedTensor(image, ptb)
        interm_bounds = {}
        lirpa_model = BoundedModule(model, image, device=image.device,verbose=0)
        # Store the output shape for each layer first
        for node in lirpa_model.nodes():
            # For each intermediate layers, we first set their bound to be infinity as placeholder.
            if hasattr(node, 'output_shape'):
                interm_lb = torch.full(node.output_shape, -float('inf'))
                interm_ub = torch.full(node.output_shape, float('inf'))
                interm_bounds[node.name] = [interm_lb, interm_ub]

        # Here we assume that the last node is the model output, and we start from intermdiate layers first.
        # Technically, here we need a topological sort of all model nodes if the computation graph is general.
        start_time = time.time()
        for node in lirpa_model.nodes():
            # For simplicity, we assume the model contains linear, conv, and ReLU layers.
            # We need to calculate the preactivation bounds before each ReLU layer, which are the bounds for linear of conv layers.
            if isinstance(node, (BoundLinear, BoundConv)):
                interm_lb = torch.full(node.output_shape, -float('inf'))
                interm_ub = torch.full(node.output_shape, float('inf'))
                if node.is_final_node:
                    print(f'Solving LPs for final layer bounds...')
                    # Last node, all intermediate layer bounds have been obtained.
                    # For last node, we need to use the specification matrix C to calculate the bounds on groundtruth - target labels.
                    solver_vars = lirpa_model.build_solver_module(model_type='lp', x=(image,), final_node_name=node.name, interm_bounds=interm_bounds, C=C)
                    lirpa_model.solver_model.setParam('OutputFlag', 0)
                    final_lb = torch.empty(classes-1)
                    # final_ub = torch.empty(classes-1)
                    for i in range(classes-1):
                        # print(f'Solving class {i}...')
                        # Now you can define objectives based on the variables on the output layer.
                        # And then solve them using gurobi. Here we just output the lower and upper
                        # bounds for each output neuron.
                        # Solve upper bound.
                        # lirpa_model.solver_model.setObjective(solver_vars[i], grb.GRB.MAXIMIZE)
                        # lirpa_model.solver_model.optimize()
                        # # If the solver does not terminate, you will get a NaN.
                        # if lirpa_model.solver_model.status == grb.GRB.Status.OPTIMAL:
                        #     final_ub[i] = lirpa_model.solver_model.objVal
                        # Solve lower bound.
                        lirpa_model.solver_model.setObjective(solver_vars[i], grb.GRB.MINIMIZE)
                        lirpa_model.solver_model.optimize()
                        if lirpa_model.solver_model.status == grb.GRB.Status.OPTIMAL:
                            final_lb[i] = lirpa_model.solver_model.objVal
                else:
                    print(f'Solving LPs for layer {node.name} intermediate layer bounds...')
                    # Solve intermediate layer bounds, one by one.
                    solver_vars = lirpa_model.build_solver_module(model_type='lp', x=(image,), final_node_name=node.name, interm_bounds=interm_bounds)
                    lirpa_model.solver_model.setParam('OutputFlag', 0)
                    # For linear layer, the solver_vars shape is: (neurons).
                    if isinstance(node, BoundLinear):
                        for i, var in enumerate(solver_vars):
                            lirpa_model.solver_model.setObjective(var, grb.GRB.MAXIMIZE)
                            lirpa_model.solver_model.optimize()
                            if lirpa_model.solver_model.status == grb.GRB.Status.OPTIMAL:
                                interm_ub[0][i] = lirpa_model.solver_model.objVal
                            # Solve lower bound.
                            lirpa_model.solver_model.setObjective(var, grb.GRB.MINIMIZE)
                            lirpa_model.solver_model.optimize()
                            if lirpa_model.solver_model.status == grb.GRB.Status.OPTIMAL:
                                interm_lb[0][i] = lirpa_model.solver_model.objVal
                        print(f'Finished solving layer {node.name} with {i+1} neurons')
                    # For convolutional layer, the solver_vars shape is (channel, out_w, out_h).
                    elif isinstance(node, BoundConv):
                        for i,channel in enumerate(solver_vars):
                            for j, row in enumerate(channel):
                                for k, var in enumerate(row):
                                    lirpa_model.solver_model.setObjective(var, grb.GRB.MAXIMIZE)
                                    lirpa_model.solver_model.optimize()
                                    if lirpa_model.solver_model.status == grb.GRB.Status.OPTIMAL:
                                        interm_ub[0][i][j][k] = lirpa_model.solver_model.objVal
                                    # Solve lower bound.
                                    lirpa_model.solver_model.setObjective(var, grb.GRB.MINIMIZE)
                                    lirpa_model.solver_model.optimize()
                                    if lirpa_model.solver_model.status == grb.GRB.Status.OPTIMAL:
                                        interm_lb[0][i][j][k] = lirpa_model.solver_model.objVal
                        print(f'Finished solving layer {node.name} with {(i+1)*(j+1)*(k+1)} neurons')
                    interm_bounds[node.name] = [interm_lb, interm_ub]
        end_time = time.time()
        with torch.no_grad():
            if torch.any(final_lb < 0):
                verification_fail += 1
                verifiction_status = "Fail"
                verification_fail_idx.append(sample_idx)
            elapsed_time = end_time - start_time
            total_time += elapsed_time
            sample_log = {
                'sample_idx': sample_idx,
                'true_label': label.item() if isinstance(label, torch.Tensor) else label,
                'margins': final_lb.cpu().tolist(),       
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
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
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
    
    verified_lp_all(
        dataset = dataset, 
        labels = labels, 
        model = model, 
        radius = radius_rescale, 
        clean_output = correct_indices, 
        device = device, 
        classes = classes, 
        args = args
        )
