# MNIST
python lipnaive.py --model mnist_mlp --radius 1.0
python lipnaive.py --model mnist_convsmall --radius 0.3
python lipnaive.py --model mnist_convlarge --radius 0.3

# CIRAR10
python lipnaive.py --model cifar10_cnn_a --radius 24/255
python lipnaive.py --model cifar10_cnn_b --radius 24/255
python lipnaive.py --model cifar10_cnn_c --radius 24/255
python lipnaive.py --model cifar10_convsmall --radius 24/255
python lipnaive.py --model cifar10_convdeep --radius 24/255
python lipnaive.py --model cifar10_convlarge --radius 8/255