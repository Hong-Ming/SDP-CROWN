"""Generate a wrapper script for running `run_all_categories.sh`
with different output directories."""

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('output_dir', type=str)
parser.add_argument('categories', type=str, nargs='+')
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

commands = []
for category in args.categories:
    print('Running category:', category)
    category_dir = f'{args.output_dir}/{category}'
    if not os.path.exists(category_dir):
        os.makedirs(category_dir)
    command = (
        'cd ~/vnncomp2024_benchmarks; ./run_all_categories.sh v1 '
        f'{os.path.dirname(os.path.dirname(__file__))}/vnncomp_scripts . '
        f'{category_dir}/out.csv '
        f'{category_dir}/counterexamples '
        f'"{category}" all 2>&1 | tee {category_dir}/log.txt'
    )
    print(command)
    commands.append(command)

# Need to save the commands into a shell script, as this python can be killed
# when running the VNNCOMP scripts.
path_script = f'{args.output_dir}/run.sh'
with open(path_script, 'w') as file:
    for command in commands:
        file.write(f'{command}\n')
print('Please run bash script:', path_script)
