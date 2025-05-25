import os
import subprocess
import argparse


"""
This script saves all the sat instances to a new folder and prints out a command to run these instances.

"""
parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', type=str, required=True, help='path to the result csv file')
parser.add_argument('--benchmark_folder', type=str, required=False, default='../../vnncomp2023_benchmarks', help='path to the benchmark folder')
parser.add_argument('--benchmark_folder_satonly', type=str, required=False, default='../../vnncomp2023_benchmarks_satonly', help='path to the benchmark folder')
args = parser.parse_args()

categories = set()

if os.path.exists(args.benchmark_folder_satonly):
    print('INFO: Removing old directory for sat instances.')
    os.system('rm -r {}'.format(args.benchmark_folder_satonly))

with open(args.path) as f:
    lines = f.readlines()

for line in lines:
    # skip if the line is invalid
    try:
        benchmark, onnx, vnnlib, time1, result, time2 = line.strip().split(',')
    except:
        print('WARNING: Invalid line: {}'.format(line))
        continue

    if result == 'sat':
        old_instance_file = os.path.join(args.benchmark_folder, 'benchmarks', benchmark, 'instances.csv')
        try:
            with open(old_instance_file) as f:
                instances = f.readlines()
        except:
            print('WARNING: Original instance file not found: {}'.format(old_instance_file))
            continue

        new_dir = os.path.join(args.benchmark_folder_satonly, 'benchmarks', benchmark)
        new_onnx = '/'.join(onnx.split('/')[-2:])
        new_vnnlib = '/'.join(vnnlib.split('/')[-2:])

        timeout = 0
        for instance in instances:
            old_onnx, old_vnnlib, old_timeout = instance.strip().split(',')
            if old_onnx == new_onnx and old_vnnlib == new_vnnlib:
                timeout = old_timeout
                break
        if timeout == 0:
            print('WARNING: Instance not found in the original benchmark folder!')
            continue

        os.makedirs(new_dir, exist_ok=True)
        new_instance_file = os.path.join(new_dir, 'instances.csv')
        with open(new_instance_file, 'a') as f:
            f.write(','.join([new_onnx, new_vnnlib, timeout]) + '\n')

        old_onnx_path = os.path.join(args.benchmark_folder, 'benchmarks', benchmark, old_onnx)
        old_vnnlib_path = os.path.join(args.benchmark_folder, 'benchmarks', benchmark, old_vnnlib)

        new_onnx_folder_path = os.path.join(new_dir, 'onnx')
        new_vnnlib_folder_path = os.path.join(new_dir, 'vnnlib')

        os.makedirs(new_onnx_folder_path, exist_ok=True)
        os.makedirs(new_vnnlib_folder_path, exist_ok=True)
        os.system('cp {} {}'.format(old_onnx_path, new_onnx_folder_path))
        os.system('cp {} {}'.format(old_vnnlib_path, new_vnnlib_folder_path))

        categories.add(benchmark)


os.system('cp {} {}'.format(os.path.join(args.benchmark_folder, 'run_all_categories.sh'), args.benchmark_folder_satonly))
os.system('cp {} {}'.format(os.path.join(args.benchmark_folder, 'run_single_instance.sh'), args.benchmark_folder_satonly))


cmd = './run_all_categories_nopgd.sh v1 ../vnncomp_scripts {} all_satonly.csv {} "{}" all 2>&1'.format(args.benchmark_folder_satonly, os.path.join(args.benchmark_folder_satonly, 'counterexamples'), ' '.join(list(categories)))
print('Please run the following command for evaluation:')
print(cmd)

