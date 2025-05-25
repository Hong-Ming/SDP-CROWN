"""Test linear programming when all nodes are split"""
import os
import numpy as np
from auto_LiRPA.utils import logger

# True for SAT and False for UNSAT
expected_output = [True, True, True, True, True, False, True, True, False, True,
                    False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                    False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True]

# constraints of output for each vnnlib and its result:
# Y_0 >= Y_1 for all_node_split_0.vnnlib SAT
# Y_0 <= 0 for all_node_split_1.vnnlib SAT
# (Y_0 <= 0 and Y_1 <= 0.5) for all_node_split_2.vnnlib SAT
# (Y_0 >= 1000 or Y_1 >= 0) for all_node_split_3.vnnlib SAT
# ((Y_0 >= 1000 or Y_1 >= 0) and (Y_0 <= 0 or Y_1 <= 1)) for all_node_split_4.vnnlib SAT
# ((Y_0 >= 10 or Y_1 >= 20) and (Y_0 <= 0 or Y_1 <= 1)) for all_node_split_5.vnnlib UNSAT
# ((Y_0 >= 1000 or Y_1 >= 0) and (Y_0 <= 0 or Y_1 <= 1)) and (Y_0 <= -10 or Y_0 >= 1) for all_node_split_6.vnnlib SAT
# (Y_0 >= Y_1) and (Y_0 >= 1 or Y_1 >= 0) and (Y_0 <= 3 or Y_1 <= 0.5) for all_node_split_7.vnnlib SAT
# (Y_0 >= Y_1) and (Y_0 >= 10 or Y_1 >= 0) and (Y_0 <= -10 or Y_1 <= -10) for all_node_split_8.vnnlib UNSAT
# (Y_0 <= Y_1) and (Y_0 >= 0 or Y_1 >= 0) and (Y_0 <= 1 or Y_1 <= -10) for all_node_split_9.vnnlib SAT

# (Y_0 >= 10 or Y_0 <= -10 or Y_1 >= 10 or Y_1 <= -10) for all_node_split_10.vnnlib UNSAT (All False)
# (Y_0 >= 0 or Y_0 <= -10 or Y_1 >= 10 or Y_1 <= -10) for all_node_split_11.vnnlib SAT
# (Y_0 >= 10 or Y_0 <= 3 or Y_1 >= 10 or Y_1 <= -10) for all_node_split_12.vnnlib SAT
# (Y_0 >= 0 or Y_0 <= 3 or Y_1 >= 10 or Y_1 <= -10) for all_node_split_13.vnnlib SAT
# (Y_0 >= 10 or Y_0 <= -10 or Y_1 >= 0 or Y_1 <= -10) for all_node_split_14.vnnlib SAT
# (Y_0 >= 0 or Y_0 <= -10 or Y_1 >= 0 or Y_1 <= -10) for all_node_split_15.vnnlib SAT
# (Y_0 >= 10 or Y_0 <= 3 or Y_1 >= 0 or Y_1 <= -10) for all_node_split_16.vnnlib SAT
# (Y_0 >= 0 or Y_0 <= 3 or Y_1 >= 0 or Y_1 <= -10) for all_node_split_17.vnnlib SAT
# (Y_0 >= 10 or Y_0 <= -10 or Y_1 >= 10 or Y_1 <= 3) for all_node_split_18.vnnlib SAT
# (Y_0 >= 0 or Y_0 <= -10 or Y_1 >= 10 or Y_1 <= 3) for all_node_split_19.vnnlib SAT
# (Y_0 >= 10 or Y_0 <= 3 or Y_1 >= 10 or Y_1 <= 3) for all_node_split_20.vnnlib SAT
# (Y_0 >= 0 or Y_0 <= 3 or Y_1 >= 10 or Y_1 <= 3) for all_node_split_21.vnnlib SAT
# (Y_0 >= 10 or Y_0 <= -10 or Y_1 >= 0 or Y_1 <= 3) for all_node_split_22.vnnlib SAT
# (Y_0 >= 0 or Y_0 <= -10 or Y_1 >= 0 or Y_1 <= 3) for all_node_split_23.vnnlib SAT
# (Y_0 >= 10 or Y_0 <= 3 or Y_1 >= 0 or Y_1 <= 3) for all_node_split_24.vnnlib SAT
# (Y_0 >= 0 or Y_0 <= 3 or Y_1 >= 0 or Y_1 <= 3) for all_node_split_25.vnnlib SAT

# (Y_0 >= 10 and Y_0 <= -10 and Y_1 >= 10 and Y_1 <= -10) for all_node_split_26.vnnlib UNSAT
# (Y_0 >= 0 and Y_0 <= -10 and Y_1 >= 10 and Y_1 <= -10) for all_node_split_27.vnnlib UNSAT
# (Y_0 >= 10 and Y_0 <= 3 and Y_1 >= 10 and Y_1 <= -10) for all_node_split_28.vnnlib UNSAT
# (Y_0 >= 0 and Y_0 <= 3 and Y_1 >= 10 and Y_1 <= -10) for all_node_split_29.vnnlib UNSAT
# (Y_0 >= 10 and Y_0 <= -10 and Y_1 >= 0 and Y_1 <= -10) for all_node_split_30.vnnlib UNSAT
# (Y_0 >= 0 and Y_0 <= -10 and Y_1 >= 0 and Y_1 <= -10) for all_node_split_31.vnnlib UNSAT
# (Y_0 >= 10 and Y_0 <= 3 and Y_1 >= 0 and Y_1 <= -10) for all_node_split_32.vnnlib UNSAT
# (Y_0 >= 0 and Y_0 <= 3 and Y_1 >= 0 and Y_1 <= -10) for all_node_split_33.vnnlib UNSAT
# (Y_0 >= 10 and Y_0 <= -10 and Y_1 >= 10 and Y_1 <= 3) for all_node_split_34.vnnlib UNSAT
# (Y_0 >= 0 and Y_0 <= -10 and Y_1 >= 10 and Y_1 <= 3) for all_node_split_35.vnnlib UNSAT
# (Y_0 >= 10 and Y_0 <= 3 and Y_1 >= 10 and Y_1 <= 3) for all_node_split_36.vnnlib UNSAT
# (Y_0 >= 0 and Y_0 <= 3 and Y_1 >= 10 and Y_1 <= 3) for all_node_split_37.vnnlib UNSAT
# (Y_0 >= 10 and Y_0 <= -10 and Y_1 >= 0 and Y_1 <= 3) for all_node_split_38.vnnlib UNSAT
# (Y_0 >= 0 and Y_0 <= -10 and Y_1 >= 0 and Y_1 <= 3) for all_node_split_39.vnnlib UNSAT
# (Y_0 >= 10 and Y_0 <= 3 and Y_1 >= 0 and Y_1 <= 3) for all_node_split_40.vnnlib UNSAT
# (Y_0 >= 0 and Y_0 <= 3 and Y_1 >= 0 and Y_1 <= 3) for all_node_split_41.vnnlib SAT (All True)




cmd_all_node_split_test = [
    f'cd ../complete_verifier; \
    python abcrown.py --config exp_configs/tutorial_examples/custom_box_data_all_node_split.yaml --enable_all_node_split_LP --save_adv_example \
    --vnnlib_path ../tests/gpu_tests/beta_crown/all_node_split/all_node_split_{i}.vnnlib'
    for i in range(42)
]

def run(cmd):
    # If there exists test_cex.txt, delete it. And then implement the abcrown.
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    if os.path.exists("../complete_verifier/test_cex.txt"):
        os.system("rm -rf ../complete_verifier/test_cex.txt")
        logger.info('there exists test_adv.txt and has been removed')
    assert os.path.exists("../complete_verifier/test_cex.txt") == False, "the 'test_adv.txt' file already exists before running!"
    os.system(cmd)


def check(cmd, expected):
    # Ensure that the counterexample and its verified result are correct. If it is unsafe, we should get a file with the counterexample.
    assert os.path.exists("../complete_verifier/test_cex.txt") == expected, "The {expected} exsistence of the 'adv_example.txt' file is unexpected for {cmd}!"

def test():
    assert len(cmd_all_node_split_test) == len(expected_output), "One command should be related to one expected output"
    for c,e in zip(cmd_all_node_split_test, expected_output):
        run(c)
        check(c,e)
    logger.info("all node split test done")

if __name__ == '__main__':
    test()