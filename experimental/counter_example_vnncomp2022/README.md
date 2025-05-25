## Counter example in VNN-COMP 2022

Saving all information regarding the missing counterexample in VNN-COMP 2022
where we got penalized.

The counter examples are saved in `.pth` files in the `saved_counter_examples`
folder. For over 300 runs of our code in the vnncomp22 commit, about 16 can
produce counterexamples. When counterexamples are produced, they are fast
(within 20 seconds).  Log files are saved in the `logs` folder.

To double check the counter examples, run the following:

```bash
# Convert the saved pth file to VNN-COMP required format.
python pth2cex.py saved_counter_examples/attack_image_1663677588.pth saved_counter_examples/attack_output_1663677588.pth out.txt <<< 20
BENCHMARK_DIR=../../../vnncomp2022_benchmarks
python check_counterexample.py ${BENCHMARK_DIR}/benchmarks/collins_rul_cnn/onnx/NN_rul_small_window_20.onnx ${BENCHMARK_DIR}/benchmarks/collins_rul_cnn/vnnlib/if_then_7levels_w20.vnnlib out.txt
```


