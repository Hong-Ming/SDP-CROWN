
import os
import sys
import pickle
import numpy as np
import pandas as pd
import csv
import re

import ast


def parse_vnncomp_log(file_name):
    # print(f'Processing file {file_name}')
    final_ret = []
    with open(file_name) as f:
        all_lines = f.readlines()
        for i, line in enumerate(all_lines):
            if "%%%%%% idx: " in line:
                m = re.match(r'.*img ID: (\d+) %%%%%%%%%%%%%%%%%', line)
                tmp_ret = []
                tmp_ret.append(int(m.group(1)))
                continue
            if line.startswith('attack margin '):
                next_line = all_lines[i+1]
                full_line = line.strip() + next_line.strip()
                start = full_line.find('[')
                end = full_line.find(']')
                tensor = full_line[start : end + 1]
                tensor = tensor.replace("inf", "9999999")
                parsed = ast.literal_eval(tensor)
                tmp_ret.append(np.array(parsed))
                continue
            if "Result:" in line:
                if 'prediction is incorrect,' in line:
                    tmp_ret += [None, 'prediction wrong']
                elif 'verification success' in line:
                    tmp_ret.append('verified success')
                elif 'attack success!' in line:
                    tmp_ret.append('attack success')
                elif 'verification failure' in line:
                    tmp_ret.append('verified fail')
                else:
                    print(line)
                    raise NotImplementedError

                final_ret.append(tmp_ret)
                continue
    return final_ret


if __name__ == "__main__":
    # model_name = 'cifar_conv_small'
    # root_path = '/home/kx46@drexel.edu/Downloads/verifier_log_cifar_conv_small/'

    # model_name = 'cifar_cnn_a_mix'
    # root_path = '/home/kx46@drexel.edu/Downloads/verifier_log_cnn_a_mix_part1/'

    # model_name = 'cifar_cnn_a_adv'
    # root_path = '/home/kx46@drexel.edu/Downloads/verifier_log_cifar_cnn_a_adv/'

    # model_name = 'cifar_cnn_b_adv'
    # root_path = '/home/kx46@drexel.edu/Downloads/verifier_log_merge_cifar_cnn_b_adv'

    # model_name = 'oval_base'
    # root_path = '/home/kx46@drexel.edu/Downloads/verifier_log_cifar_oval_base'

    # model_name = 'oval_wide'
    # root_path = '/home/kx46@drexel.edu/Downloads/verifier_log_cifar_oval_wide'

    # model_name = 'oval_deep'
    # root_path = '/home/kx46@drexel.edu/Downloads/verifier_log_cifar_oval_deep'

    # model_name = 'mnist_conv_small'
    # root_path = '/home/kx46@drexel.edu/Downloads/verifier_log_mnist_conv_small'

    # model_name = 'cifar_cnn_a_adv_alt'
    # root_path = '/home/kx46@drexel.edu/Downloads/verifier_log_cifar_cnn_a_adv_alt'

    # model_name = 'cifar_marabou_small'
    # root_path = '/home/kx46@drexel.edu/Downloads/verifier_log_cifar_marabou_small'

    model_name = 'resnet2b'
    root_path = '/home/kx46@drexel.edu/Downloads/verifier_log_resnet_2b'

    # model_name = 'cifar_marabou_medium'
    # root_path = '/home/kx46@drexel.edu/Downloads/verifier_log_cifar_marabou_medium'

    # model_name = 'Madry_no_maxpool_tiny'
    # root_path = '/home/kx46@drexel.edu/Downloads/verifier_log_MadryCNN_no_maxpool_tiny'

    # model_name = 'mnist_cnn_a_adv'
    # root_path = '/home/kx46@drexel.edu/Downloads/verifier_logs_mnist_cnn_a_adv'

    ret = []
    for log_file in sorted(os.listdir(root_path)):
        if log_file.endswith('.log'):
            parse_log = parse_vnncomp_log(os.path.join(root_path, log_file))
            ret += parse_log

    print(len(ret))
    final_ret = []

    c = 0
    for i in ret:
        assert c == i[0]
        c += 1
        # 'Idx', 'classify_correct', 'pgd_in_bab_success', 'dpgd_success', 'pgd_success', 'auto_success', 'bab_verified', 'pgd_margin'
        if i[-1] == 'prediction wrong':
            final_ret.append([i[0], 0, 0, 0, 0, 0, 0, None])
        elif i[-1] == 'verified success':
            final_ret.append([i[0], 1, 0, 0, 0, 0, 1, i[1]])
        elif i[-1] == 'attack success':
            final_ret.append([i[0], 1, 1, None, None, None, 0, i[1]])
        elif i[-1] == 'verified fail':
            final_ret.append([i[0], 1, 0, None, None, None, 0, i[1]])
        else:
            print(i)
            raise NotImplementedError

    df = pd.DataFrame(final_ret,
                      columns=['Idx', 'classify_correct', 'pgd_in_bab_success', 'dpgd_success', 'pgd_success', 'auto_success', 'bab_verified',
                               'pgd_margin'])

    print('# of classify_correct:', sum(df['classify_correct'].to_list()))
    print('# of pgd_in_bab_success:', sum(df['pgd_in_bab_success'].to_list()))
    print('# of bab_verified:', sum(df['bab_verified'].to_list()))

    df.to_pickle('filtered_' + model_name + '.pkl', )  # save to pkl

    print(df.columns)





