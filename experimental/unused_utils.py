import torch


def reshape_bounds(lower_bounds, upper_bounds, y, global_lb=None):
    with torch.no_grad():
        last_lower_bounds = torch.zeros(size=(1, lower_bounds[-1].size(1)+1), dtype=lower_bounds[-1].dtype, device=lower_bounds[-1].device)
        last_upper_bounds = torch.zeros(size=(1, upper_bounds[-1].size(1)+1), dtype=upper_bounds[-1].dtype, device=upper_bounds[-1].device)
        last_lower_bounds[:, :y] = lower_bounds[-1][:, :y]
        last_lower_bounds[:, y+1:] = lower_bounds[-1][:, y:]
        last_upper_bounds[:, :y] = upper_bounds[-1][:, :y]
        last_upper_bounds[:, y+1:] = upper_bounds[-1][:, y:]
        lower_bounds[-1] = last_lower_bounds
        upper_bounds[-1] = last_upper_bounds
        if global_lb is not None:
            last_global_lb = torch.zeros(size=(1, global_lb.size(1)+1), dtype=global_lb.dtype, device=global_lb.device)
            last_global_lb[:, :y] = global_lb[:, :y]
            last_global_lb[:, y+1:] = global_lb[:, y:]
            global_lb = last_global_lb
    return lower_bounds, upper_bounds, global_lb


def convert_mlp_model(model, dummy_input):
    model.eval()
    feature_maps = {}

    def get_feature_map(name):
        def hook(model, input, output):
            feature_maps[name] = output.detach()

        return hook

    def conv_to_dense(conv, inputs):
        b, n, w, h = inputs.shape
        kernel = conv.weight
        bias = conv.bias
        I = torch.eye(n * w * h).view(n * w * h, n, w, h)
        W = F.conv2d(I, kernel, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
        # input_flat = inputs.view(b, -1)
        b1, n1, w1, h1 = W.shape
        # out = torch.matmul(input_flat, W.view(b1, -1)).view(b, n1, w1, h1)
        new_bias = bias.view(1, n1, 1, 1).repeat(1, 1, w1, h1)

        dense_w = W.view(b1, -1).transpose(1, 0)
        dense_bias = new_bias.view(-1)

        new_m = nn.Linear(in_features=dense_w.shape[1], out_features=dense_w.shape[0], bias=m.bias is not None)
        new_m.weight.data.copy_(dense_w)
        new_m.bias.data.copy_(dense_bias)

        return new_m

    new_modules = []
    modules = list(model.named_modules())[1:]
    for mi, (name, m) in enumerate(modules):

        if mi+1 < len(modules) and isinstance(modules[mi+1][-1], nn.Conv2d):
            m.register_forward_hook(get_feature_map(name))
            model(dummy_input)
            pre_conv_input = feature_maps[name]
        elif mi == 0 and isinstance(m, nn.Conv2d):
            pre_conv_input = dummy_input

        if isinstance(m, nn.Linear):
            new_m = nn.Linear(in_features=m.in_features, out_features=m.out_features, bias=m.bias is not None)
            new_m.weight.data.copy_(m.weight.data)
            new_m.bias.data.copy_(m.bias)
            new_modules.append(new_m)
        elif isinstance(m, nn.ReLU):
            new_modules.append(nn.ReLU())
        elif isinstance(m, nn.Flatten):
            pass
            # will flatten at the first layer
            # new_modules.append(nn.Flatten())
        elif isinstance(m, nn.Conv2d):
            new_modules.append(conv_to_dense(m, pre_conv_input))
        else:
            print(m, 'not support in convert_mlp_model')
            raise NotImplementedError

    #  add flatten at the beginning
    new_modules.insert(0, nn.Flatten())
    seq_model = nn.Sequential(*new_modules)

    return seq_model


def get_pgd_acc(model, X, labels, eps, data_min, data_max, batch_size):
    start = arguments.Config["data"]["start"]
    total = arguments.Config["data"]["end"]
    clean_correct = 0
    robust_correct = 0
    model = model.to(device=arguments.Config["general"]["device"])
    X = X.to(device=arguments.Config["general"]["device"])
    labels = labels.to(device=arguments.Config["general"]["device"])
    if isinstance(data_min, torch.Tensor):
        data_min = data_min.to(device=arguments.Config["general"]["device"])
    if isinstance(data_max, torch.Tensor):
        data_max = data_max.to(device=arguments.Config["general"]["device"])
    if isinstance(eps, torch.Tensor):
        eps = eps.to(device=arguments.Config["general"]["device"])
    if arguments.Config["attack"]["pgd_alpha"] == 'auto':
        alpha = eps.mean() / 4 if isinstance(eps, torch.Tensor) else eps / 4
    else:
        alpha = float(arguments.Config["attack"]["pgd_alpha"])
    while start < total:
        end = min(start + batch_size, total)
        batch_X = X[start:end]
        batch_labels = labels[start:end]
        if arguments.Config["specification"]["type"] == "lp":
            # Linf norm only so far.
            data_ub = torch.min(batch_X + eps, data_max)
            data_lb = torch.max(batch_X - eps, data_min)
        else:
            # Per-example, per-element lower and upper bounds.
            data_ub = data_max[start:end]
            data_lb = data_min[start:end]
        clean_output = model(batch_X)

        best_deltas, last_deltas = attack_pgd(model, X=batch_X, y=batch_labels, epsilon=float("inf"), alpha=alpha,
                num_classes=arguments.Config["data"]["num_outputs"],
                attack_iters=arguments.Config["attack"]["pgd_steps"], num_restarts=arguments.Config["attack"]["pgd_restarts"],
                upper_limit=data_ub, lower_limit=data_lb, multi_targeted=True, lr_decay=arguments.Config["attack"]["pgd_lr_decay"],
                target=None, early_stop=arguments.Config["attack"]["pgd_early_stop"])
        attack_images = torch.max(torch.min(batch_X + best_deltas, data_ub), data_lb)
        attack_output = model(attack_images)
        clean_labels = clean_output.argmax(1)
        attack_labels = attack_output.argmax(1)
        batch_clean_correct = (clean_labels == batch_labels).sum().item()
        batch_robust_correct = (attack_labels == batch_labels).sum().item()
        if start == 0:
            print("Clean prediction for first a few examples:")
            print(clean_output[:10].detach().cpu().numpy())
            print("PGD prediction for first a few examples:")
            print(attack_output[:10].detach().cpu().numpy())
        print(f'batch start {start}, batch size {end - start}, clean correct {batch_clean_correct}, robust correct {batch_robust_correct}')
        clean_correct += batch_clean_correct
        robust_correct += batch_robust_correct
        start += batch_size
        del clean_output, best_deltas, last_deltas, attack_images, attack_output
    print(f'data start {arguments.Config["data"]["start"]} end {total}, clean correct {clean_correct}, robust correct {robust_correct}')
    return clean_correct, robust_correct


class Normalization(nn.Module):
    def __init__(self, mean, std, model):
        super(Normalization, self).__init__()
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)
        self.model = model

    def forward(self, x):
        return self.model((x - self.mean)/self.std)


def get_test_acc(model, input_shape=None, X=None, labels=None, batch_size=256):
    device = arguments.Config["general"]["device"]
    if X is None and labels is None:
        # Load MNIST or CIFAR, used for quickly debugging.
        database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
        mean = torch.tensor(arguments.Config["data"]["mean"])
        std = torch.tensor(arguments.Config["data"]["std"])
        normalize = transforms.Normalize(mean=mean, std=std)
        if input_shape == (3, 32, 32):
            testset = torchvision.datasets.CIFAR10(
                root=database_path, train=False, download=True,
                transform=transforms.Compose([transforms.ToTensor(), normalize]))
        elif input_shape == (1, 28, 28):
            testset = torchvision.datasets.MNIST(
                root=database_path, train=False, download=True,
                transform=transforms.Compose([transforms.ToTensor(), normalize]))
        else:
            raise RuntimeError("Unable to determine dataset for test accuracy.")
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2)
    else:
        testloader = [(X, labels)]
    total = 0
    correct = 0
    if device != 'cpu':
        model = model.to(device)
    print_first_batch = True
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if device != 'cpu':
                images = images.to(device)
                labels = labels.to(device)
            if arguments.Config["model"]["convert_model_to_NCHW"]:
                images = images.permute(0, 2, 3, 1)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if print_first_batch:
                print_first_batch = False
                for i in range(min(outputs.size(0), 10)):
                    print(f"Image {i} norm {images[i].abs().sum().item()} label {labels[i].item()} correct {labels[i].item() == outputs[i].argmax().item()}\nprediction {outputs[i].cpu().numpy()}")
    print(f'correct {correct} of {total}')
