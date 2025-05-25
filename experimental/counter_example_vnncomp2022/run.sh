#!/bin/bash

# Repeat multiple times with different random seed and see if we can produce a counter example.
# The command runs on the VNN-COMP 2022 version base, not the master.

while :;
do
    seed=$RANDOM
    echo "Using seed $seed"
    python bab_verification_general.py --config exp_configs/vnncomp22/collins-rul-cnn.yaml --start 20 --end 21 --save_adv_example --seed $seed
done
