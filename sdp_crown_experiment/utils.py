import torch
import numpy as np

from models import *
from fractions import Fraction

def parse_float_or_fraction(x: str) -> float:
    try:
        return float(x)
    except ValueError:
        return float(Fraction(x))

# Help function to generate C matrix for calculate the margins.
def build_C(label, classes):
    """
    label: shape (B,). Each label[b] in [0..classes-1].
    Return:
        C: shape (B, classes-1, classes).
        For each sample b, each row is a “negative class” among [0..classes-1]\{label[b]}.
        Puts +1 at column=label[b], -1 at each negative class column.
    """
    device = label.device
    batch_size = label.size(0)
    
    # 1) Initialize
    C = torch.zeros((batch_size, classes-1, classes), device=device)
    
    # 2) All class indices
    # shape: (1, K) -> (B, K)
    all_cls = torch.arange(classes, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # 3) Negative classes only, shape (B, K-1)
    # mask out the ground-truth
    mask = all_cls != label.unsqueeze(1)
    neg_cls = all_cls[mask].view(batch_size, -1)
    
    # 4) Scatter +1 at each sample’s ground-truth label
    #    shape needed: (B, K-1, 1)
    pos_idx = label.unsqueeze(1).expand(-1, classes-1).unsqueeze(-1)
    C.scatter_(dim=2, index=pos_idx, value=1.0)
    
    # 5) Scatter -1 at each row’s negative label
    #    We have (B, K-1) negative labels. For row j in each sample b, neg_cls[b, j] is that row’s negative label
    row_idx = torch.arange(classes-1, device=device).unsqueeze(0).expand(batch_size, -1)
    # shape: (B, K-1)
    
    # We can do advanced indexing:
    C[torch.arange(batch_size).unsqueeze(1), row_idx, neg_cls] = -1.0
    
    return C

def preprocess_cifar(image, inception_preprocess=False, perturbation=False):
    """
    Preprocess images and perturbations.Preprocessing used by the SDP paper.
    """
    MEANS = np.array([125.3, 123.0, 113.9], dtype=np.float32)/255
    STD = np.array([0.225, 0.225, 0.225], dtype=np.float32)
    if inception_preprocess:
        # Use 2x - 1 to get [-1, 1]-scaled images
        rescaled_devs = 0.5
        rescaled_means = 0.5
    else:
        rescaled_means = MEANS
        rescaled_devs = STD
    if perturbation:
        return image / rescaled_devs
    else:
        return (image - rescaled_means) / rescaled_devs
    
def load_model_and_dataset(args, device):
    match args.model.lower():
        case "mnist_mlp":
            model = MNIST_MLP().to(device)
            checkpoint = torch.load('./models/mnist_mlp.pth',map_location=device)
            args.dataset = "mnist"
        case "mnist_convsmall":
            model = MNIST_ConvSmall().to(device)
            checkpoint = torch.load('./models/mnist_convsmall.pth',map_location=device)
            args.dataset = "mnist"
        case "mnist_convlarge":
            model = MNIST_ConvLarge().to(device)
            checkpoint = torch.load('./models/mnist_convlarge.pth',map_location=device)
            args.dataset = "mnist"
        case "cifar10_cnn_a":
            model = CIFAR10_CNN_A().to(device)
            checkpoint = torch.load('./models/cifar10_cnn_a.pth',map_location=device)
            args.dataset = "cifar10"
        case "cifar10_cnn_b":
            model = CIFAR10_CNN_B().to(device)
            checkpoint = torch.load('./models/cifar10_cnn_b.pth',map_location=device)
            args.dataset = "cifar10"
        case "cifar10_cnn_c":
            model = CIFAR10_CNN_C().to(device)
            checkpoint = torch.load('./models/cifar10_cnn_c.pth',map_location=device)
            args.dataset = "cifar10"
        case "cifar10_convsmall":
            model = CIFAR10_ConvSmall().to(device)
            checkpoint = torch.load('./models/cifar10_convsmall.pth',map_location=device)
            args.dataset = "cifar10"
        case "cifar10_convdeep":
            model = CIFAR10_ConvDeep().to(device)
            checkpoint = torch.load('./models/cifar10_convdeep.pth',map_location=device)
            args.dataset = "cifar10"
        case "cifar10_convlarge":
            model = CIFAR10_ConvLarge().to(device)
            checkpoint = torch.load('./models/cifar10_convlarge.pth',map_location=device)
            args.dataset = "cifar10"
        case _:
            raise ValueError(f"Unexpected model: {args.model}")
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Load dataset
    if "mnist" in args.model.lower():
        dataset = np.load('./data/sdp/mnist/X_sdp.npy')
        labels = np.load('./data/sdp/mnist/y_sdp.npy')
        dataset = torch.from_numpy(dataset).permute(0,3,1,2)
        labels = torch.from_numpy(labels)
        range = args.radius
        classes = 10
    elif "cifar10" in args.model.lower():
        dataset = np.load('./data/sdp/cifar/X_sdp.npy')
        labels = np.load('./data/sdp/cifar/y_sdp.npy')
        dataset = preprocess_cifar(dataset)
        dataset = torch.from_numpy(dataset).permute(0,3,1,2)
        labels = torch.from_numpy(labels)
        range = args.radius/0.225
        classes = 10
    else:
        raise ValueError(f"Unexpected model: {args.model}")
    
    dataset = dataset[args.start:args.end]
    labels = labels[args.start:args.end]
    return model, dataset, labels, range, classes
