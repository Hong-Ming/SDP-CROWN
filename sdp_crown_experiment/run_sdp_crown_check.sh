# MNIST
# python sdp_crown.py --model mnist_mlp --radius 1.0 --end 10
# python sdp_crown.py --model mnist_convsmall --radius 0.3 --end 2
# python sdp_crown.py --model mnist_convlarge --radius 0.3

# CIRAR10
# python sdp_crown.py --model cifar10_cnn_a --radius 24/255
python sdp_crown.py --model cifar10_cnn_b --radius 24/255 --end 3
# python sdp_crown.py --model cifar10_cnn_c --radius 24/255 --end 10
# python sdp_crown.py --model cifar10_convbase --radius 24/255
# python sdp_crown.py --model cifar10_convdeep --radius 24/255
# python sdp_crown.py --model cifar10_convlarge --radius 8/255