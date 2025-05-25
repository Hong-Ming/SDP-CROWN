"""Independently verify saved counterexamples."""

import argparse
import os
import csv
from counterexamples import is_correct_counterexample

parser = argparse.ArgumentParser()
parser.add_argument('folder_cex', type=str)
parser.add_argument('folder_benchmark', type=str)
args = parser.parse_args()

wrong_results = []

for benchmark in os.listdir(args.folder_cex):
    print('Checking benchmark:', benchmark)
    instance_path = os.path.join(
        args.folder_benchmark,
        'benchmarks',
        benchmark, 'instances.csv'
    )
    with open(instance_path) as csvfile:
        reader = csv.reader(csvfile)
        instances = [row for row in reader]
    result_path = os.path.join(
        args.folder_cex,
        benchmark,
        'out.csv'
    )
    with open(result_path) as csvfile:
        reader = csv.reader(csvfile)
        # Ignore the last item which is `test_nano`
        results = [row for row in reader][:-1]

    if len(instances) != len(results):
        print('Error: Results are incomplete. '
              f'{len(instances)} instances. {len(results)} results.')

    # safenlp has duplicate filenames
    duplicate_files = {}

    for i in range(len(results)):
        if results[i][4] == 'sat':
            print(f'Checking instance {i}')
            onnx_filename = os.path.join(args.folder_benchmark, results[i][1])
            vnnlib_filename = os.path.join(args.folder_benchmark, results[i][2])

            onnx_in_cex = '.'.join(os.path.basename(onnx_filename).split('.')[:-1])
            vnnlib_in_cex = '.'.join(os.path.basename(vnnlib_filename).split('.')[:-1])

            if (onnx_in_cex, vnnlib_in_cex) in duplicate_files:
                print('Duplicate files!', onnx_in_cex, vnnlib_in_cex)
            duplicate_files[(onnx_in_cex, vnnlib_in_cex)] = True

            cex_path = os.path.join(
                args.folder_cex,
                benchmark,
                'counterexamples',
                benchmark,
                f'{onnx_in_cex}_{vnnlib_in_cex}.counterexample'
            )
            print(cex_path)
            print(os.path.exists(cex_path))
            ret = is_correct_counterexample(
                onnx_filename, vnnlib_filename, cex_path)
            if ret[0] != 'correct':
                print('Failed!!!!!!!!!!!!!!!!!!!')
                wrong_results.append(ret)
            print(ret)
            print()

print('Wrong results:')
print(wrong_results)
