import os

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import yaml

from vnnlib import get_io_nodes, read_vnnlib_simple


class CounterexampleResult:
    """Enum for return value of is_correct_counterexample"""
    CORRECT = "correct"
    NO_CE = "no_ce"
    EXEC_DOESNT_MATCH = "exec_doesnt_match"
    SPEC_NOT_VIOLATED = "spec_not_violated"


def predict_with_onnxruntime(model_def, *inputs):
    'Run an ONNX model'
    sess = ort.InferenceSession(model_def.SerializeToString())
    names = [i.name for i in sess.get_inputs()]
    inp = dict(zip(names, inputs))
    res = sess.run(None, inp)
    return res[0]


def read_ce_file(ce_path):
    """Get file contents"""
    with open(ce_path, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace('\n', ' ').strip()
    return content


def get_ce_diff(onnx_filename, vnnlib_filename, ce_path, abs_tol, rel_tol):
    """Get difference in execution"""
    content = read_ce_file(ce_path)
    if len(content) < 2:
        return CounterexampleResult.NO_CE, f"Note: no counter example provided in {ce_path}"
    assert content[0] == '(' and content[-1] == ')'
    content = content[1:-1]
    x_list = []
    y_list = []
    parts = content.split(')')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        while "  " in part:
            part = part.replace("  ", " ")
        assert part[0] == '('
        part = part[1:]
        name, num = part.split(' ')
        assert name[0:2] in ['X_', 'Y_']
        if name[0:2] == 'X_':
            assert int(name[2:]) == len(x_list)
            x_list.append(float(num))
        else:
            assert int(name[2:]) == len(y_list)
            y_list.append(float(num))
    onnx_model = onnx.load(onnx_filename)
    inp, _out, input_dtype = get_io_nodes(onnx_model)
    input_shape = tuple(d.dim_value if d.dim_value != 0 else 1
                        for d in inp.type.tensor_type.shape.dim)
    x_in = np.array(x_list, dtype=input_dtype)
    flatten_order = 'C'
    x_in = x_in.reshape(input_shape, order=flatten_order)
    output = predict_with_onnxruntime(onnx_model, x_in)
    flat_out = output.flatten(flatten_order)
    expected_y = np.array(y_list)
    extra_msg = ""
    try:
        diff = np.linalg.norm(flat_out - expected_y, ord=np.inf)
        norm = np.linalg.norm(expected_y, ord=np.inf)
        if norm < 1e-6:
            rel_error = 0
        else:
            rel_error = diff / norm
    except ValueError as e:
        diff = 9999
        rel_error = 9999
        extra_msg = f" ERROR: {e}"
    msg = f"L-inf norm difference between onnx execution and CE file output: {diff} (rel error: {rel_error}); (rel_limit: {rel_tol})" + extra_msg
    rv = CounterexampleResult.CORRECT
    if rel_error > rel_tol:
        rv = CounterexampleResult.EXEC_DOESNT_MATCH
    else:
        is_vio, msg2 = is_specification_vio(onnx_filename, vnnlib_filename,
                                            tuple(x_list), tuple(y_list),
                                            abs_tol)
        msg += "\n" + msg2
        if not is_vio:
            msg += "\nNote: counterexample in file did not violate the specification and so was invalid!"
            rv = CounterexampleResult.SPEC_NOT_VIOLATED
    return rv, msg


def is_specification_vio(onnx_filename, vnnlib_filename, x_list, expected_y,
                         tol):
    """Check that the spec file was obeyed"""

    msg = "Checking if spec was actually violated"
    onnx_model = onnx.load(onnx_filename)

    inp, out, _ = get_io_nodes(onnx_model)

    inp_shape = tuple(d.dim_value if d.dim_value != 0 else 1
                      for d in inp.type.tensor_type.shape.dim)
    out_shape = tuple(d.dim_value if d.dim_value != 0 else 1
                      for d in out.type.tensor_type.shape.dim)

    num_inputs = 1
    num_outputs = 1

    for n in inp_shape:
        num_inputs *= n

    for n in out_shape:
        num_outputs *= n

    box_spec_list = read_vnnlib_simple(vnnlib_filename, num_inputs,
                                       num_outputs)

    rv = False

    for i, box_spec in enumerate(box_spec_list):
        input_box, spec_list = box_spec
        assert len(input_box) == len(
            x_list
        ), f"input box len: {len(input_box)}, x_in len: {len(x_list)}"

        inside_input_box = True

        input_box_array = np.array(input_box)
        lb_array, ub_array = input_box_array[:, 0], input_box_array[:, 1]

        # Check if x_list is inside the input box using numpy operations
        inside_input_box = np.all((x_list >= lb_array - tol)
                                  & (x_list <= ub_array + tol))

        if inside_input_box:
            msg += f"\nCE input X was inside box #{i}"

            # check spec
            violated = False

            for j, (prop_mat, prop_rhs) in enumerate(spec_list):
                vec = prop_mat.dot(expected_y)
                sat = np.all(vec <= prop_rhs + tol)

                if sat:
                    msg += f"\nprop #{j} violated:\n{vec - prop_rhs}"
                    violated = True
                    break

            if violated:
                rv = True
                break

    return rv, msg


def check_counterexample(directory):
    """
    Args:
        directory: dir to store config.yaml/tests.yaml of test sets.
    """
    results = []
    config_path = os.path.join(directory, 'config.yaml')
    if not os.path.exists(config_path):
        print(f'Configuration file not found in {directory}')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    tests_path = os.path.join(directory, 'tests.yaml')
    if not os.path.exists(config_path):
        print(f'Tests file not found in {directory}')
    with open(tests_path, 'r') as file:
        tests = yaml.safe_load(file)

    instance = pd.read_csv(os.path.join(directory,
                                        config['general']['root_path'],
                                        config['general']['csv_name']),
                           header=None,
                           names=['onnx', 'vnnlib', 'timeout'])

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.counterexample'):
                ce_path = os.path.join(root, file)
                index = os.path.splitext(os.path.basename(file))[0]

                # Construct file paths using the index
                onnx_filename = os.path.join(
                    directory, config['general']['root_path'],
                    instance['onnx'][tests['tests'][int(index)]['idx']])
                vnnlib_filename = os.path.join(
                    directory, config['general']['root_path'],
                    instance['vnnlib'][tests['tests'][int(index)]['idx']])

                res, msg = get_ce_diff(onnx_filename, vnnlib_filename, ce_path,
                                       0.001, 0.0001)
                results.append((ce_path, res))
    error_counterexamples = [item[0] for item in results if item[1] == 'error']
    if len(error_counterexamples) == 0:
        print('All counterexamples are valid.')
        return True
    for counterexample in error_counterexamples:
        print('Error counterexample(s):', counterexample)
    return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process counterexamples.')
    parser.add_argument('directory',
                        type=str,
                        help='Directory to scan for counterexamples')
    args = parser.parse_args()

    results = check_counterexample(args.directory)

    for ce_path, res in results:
        print(f"{ce_path}: {res}")


if __name__ == "__main__":
    main()
