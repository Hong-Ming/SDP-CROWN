### preprocessor-hint: private-file

import os
import glob
import sys
import ast
import numpy as np

def cnt_success(log):
    success = False
    time_flag = "mean time [cnt:1] (including attack success): "
    ub_flag = "Global ub: "
    f = open(log, "r")
    time = 0
    ub_min = float('inf')
    all_lines = f.readlines()
    for i, line in enumerate(all_lines):
        if "img ID: " in line:
            start_idx = line.find("img ID: ") + len("img ID: ")
            end_idx = line[start_idx:].find(" %") + start_idx
            imgidx = int(line[start_idx: end_idx])
        # if "Attack success during bab!!!!!" in line:
        if "attack_success rate: 1.0" in line:
            success = True
        if ub_flag in line:
            idx = line.find(ub_flag) + len(ub_flag)
            end_idx = line.find(",")
            ub = float(line[idx:end_idx])
            ub_min = min(ub_min, ub)
        if "Current lb:" in line and "ub:" in line:
            idx = line.find("ub:") + len("ub:")
            ub = float(line[idx:])
            ub_min = min(ub_min, ub)
        if "attack margin" in line:
            # import pdb; pdb.set_trace()
            full_line = line.strip() + all_lines[i+1].strip() + all_lines[i+2].strip()
            start = full_line.find('[')
            end = full_line.find(']')
            tensor = full_line[start : end + 1]
            tensor = tensor.replace("inf", "9999999")
            tensor = ast.literal_eval(tensor)
            tensor = np.array(tensor)
            ub_min = min(ub_min, np.min(tensor))
            # print(tensor, ub_min)
            # exit()
        if time_flag in line:
            idx = line.find(time_flag) + len(time_flag)
            time = float(line[idx:])
    f.close()
    return imgidx, success, time, ub_min


def overwrite_ubs(file_name, ubs):
    if not os.path.exists(file_name):
        f = open(file_name, "w")
        for ub in ubs:
            f.write(f"{ub},\n")
        f.close()
        return
    # if already exists ub files, only overwrite better ubs
    old_ubs = []
    f = open(file_name, "r")
    for line in f.readlines():
        if len(line) > 1:
            old_ubs.append(float(line[:-2]))
    f.close()
    if len(old_ubs) != len(ubs):
        print("Warning: ubs length not the same! Please either remove the old one or make sure the order is the same")
    # assert len(old_ubs) == len(ubs)
    f = open(file_name, "w")
    for i, (oub, ub) in enumerate(zip(old_ubs, ubs)):
        if oub > ub:
            print(f"sample {i} updated from {oub} to {ub}")
        f.write(f"{min(oub, ub)},\n")


if __name__ == "__main__":

    # model = "mnist_madry"
    # model = "mnist_a_adv"
    # model = "cifar_cnn_a_adv"
    model = sys.argv[1]
    task_set = ["unsafe", "unknown", "all"]

    success_time = {}
    for ts in task_set:
        folder = f"{model}_{ts}/"
        logs = glob.glob(folder+"job_*.log")
        logs.sort()
        num_logs = len(logs)

        cnt = 0
        avg_time = 0.
        failed = []
        ubs = []
        max_idx = -1
        log_idx = 0
        for log in logs:
            idx_start = log.find("job_")+len("job_")
            idx_end = log[idx_start:].find("_") + idx_start
            idx = int(log[idx_start: idx_end])
            assert idx > max_idx
            max_idx = idx
            if idx > log_idx:
                ubs += [float('inf')]*(idx-log_idx)
                log_idx = idx
            # print(log_idx, idx)
            imgidx, success, time, ub = cnt_success(log)
            ubs.append(ub)
            if success:
                cnt += 1
                avg_time += time
                success_time[imgidx] = time
            else:
                failed.append(idx)
            log_idx += 1
        avg_time = avg_time/cnt if cnt != 0 else "inf"
        print(f"{folder}, {ts}: success {cnt} total {len(logs)}, avg time: {avg_time}")
        print("failure:", failed)
        if len(logs) == 0:
           continue

        # write ubs
        # ub_folder = "exp_configs/bab_attack/attack_ubs/"
        # if not os.path.exists(ub_folder):
        #     print(f"create folder {ub_folder}")
        #     os.system(f"mkdir {ub_folder}")

        # file_name = f"{ub_folder}{folder[:-1]}_ub.txt"
        # overwrite_ubs(file_name, ubs)

    # print(f"{model} success time [{len(success_time)}]:", success_time)
    print(f"{model} success time [{len(success_time)}]:", ["{}: {:.2f}".format(key, success_time[key]) for key in sorted(success_time.keys())])




