import sys
sys.path.append('../complete_verifier')

import argparse
import os
import torch
import numpy as np
from abcrown import ABCROWN

parser = argparse.ArgumentParser()
parser.add_argument('--example', choices=['quadrotor2d_output', 'acasxu'],
                    type=str, default='quadrotor2d_output')
args = parser.parse_args()

args_abcrown = []

args_abcrown = {
    'batch_size': 10000,
    'select_instance': 0,
    'override_timeout': 5,
    'sort_domain_interval': 1,
    'input_split_update_rhs_with_attack': True,
}

if args.example == 'quadrotor2d_output':
    path_lyapunov_training = os.path.expanduser('~/neural_lyapunov_training')
    args_abcrown.update({
        'config': f'{path_lyapunov_training}/quadrotor2d_output_feedback_lyapunov_in_levelset.yaml',
        'root_path': path_lyapunov_training,
    })
else:
    args_abcrown.update({
        'config': 'exp_configs/vnncomp23/acasxu.yaml'
    })

verifier = ABCROWN(**args_abcrown)
# Run it with a small timeout to initialize the verifier
verifier.main()

# For this example, reuse BaB arguments saved in the initial `.main()` call.
# Please inspect the format of these arguements and set them when actually using the API.
data = verifier.data
targets = verifier.targets
data_lb = verifier.data_lb
data_ub = verifier.data_ub
vnnlib = verifier.vnnlib
c = verifier.c
rhs = verifier.rhs

lower_bound, x_L, x_U = verifier.bab(
    data_lb, data_ub, c, rhs, data=data, targets=targets,
    max_iterations=2, timeout=1e8,
    create_model=False,
    return_domains=-1,  # Return all domains
)
print((lower_bound - rhs.cpu()).amin(dim=-1))

# ret = verifier.bab(
#    data_lb, data_ub, c, rhs, data=data, targets=targets, timeout=timeout,
#    create_model=False, return_domains=True,
# )
# domains = ret[3]
# print(len(domains))
