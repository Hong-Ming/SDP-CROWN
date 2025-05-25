"""A copy of the transfer_to_cpu function which is originally a part of
beta_CROWN_solver."""

def transfer_to_cpu(self, net, non_blocking=False, opt_intermediate_beta=False, transfer_items="all"):
    """
    Trasfer all necessary tensors to CPU in a batch.
    WARNING: non_blocking transfer can be tricky becasue you may use a tensor
    before it is fully transffer. An explicit synchronization is needed.
    """
    cpu_net = SimpleNamespace()
    cpu_net.layers_requiring_bounds = [None] * len (net.layers_requiring_bounds)
    for i in range(len(cpu_net.layers_requiring_bounds)):
        cpu_net.layers_requiring_bounds[i] = SimpleNamespace()
        cpu_net.layers_requiring_bounds[i].inputs = [SimpleNamespace()]
        cpu_net.layers_requiring_bounds[i].name = net.layers_requiring_bounds[i].name
    cpu_net.perturbed_optimizable_activations = [None] * len(net.perturbed_optimizable_activations)
    for i in range(len(cpu_net.perturbed_optimizable_activations)):
        cpu_net.perturbed_optimizable_activations[i] = SimpleNamespace()
        cpu_net.perturbed_optimizable_activations[i].inputs = [SimpleNamespace()]
        cpu_net.perturbed_optimizable_activations[i].name = net.perturbed_optimizable_activations[i].name

    transfer_size = defaultdict(int)
    # Transfer data structures for each neuron.
    # For get_candidate_parallel().
    if transfer_items == "all" or transfer_items == "intermediate_bounds":
        if self.interm_transfer:
            for cpu_layer, layer in zip(cpu_net.layers_requiring_bounds, net.layers_requiring_bounds):
                # For get_candidate_parallel.
                cpu_layer.lower = layer.lower.to(device='cpu', non_blocking=non_blocking)
                cpu_layer.upper = layer.upper.to(device='cpu', non_blocking=non_blocking)
                transfer_size['pre'] += layer.lower.numel() * 2
        # For get_lA_parallel().
        for cpu_layer, layer in zip(cpu_net.perturbed_optimizable_activations, net.perturbed_optimizable_activations):
            cpu_layer.lA = layer.lA.to(device='cpu', non_blocking=non_blocking)
            transfer_size['lA'] += layer.lA.numel()
    # For get_slope().
    if transfer_items == "all" or transfer_items == "slopes":
        for cpu_layer, layer in zip(cpu_net.perturbed_optimizable_activations, net.perturbed_optimizable_activations):
            # Per-neuron alpha.
            cpu_layer.alpha = OrderedDict()
            for spec_name, alpha in layer.alpha.items():
                cpu_layer.alpha[spec_name] = alpha.half().to(device='cpu', non_blocking=non_blocking)
                transfer_size['alpha'] += alpha.numel()
    # For get_beta().
    if transfer_items == "all":
        for cpu_layer, layer in zip(cpu_net.perturbed_optimizable_activations, net.perturbed_optimizable_activations):
            if hasattr(layer, 'sparse_beta') and layer.sparse_beta is not None:
                if arguments.Config["solver"]["beta-crown"]['enable_opt_interm_bounds']:
                    cpu_layer.sparse_beta = OrderedDict()
                    for key in layer.sparse_beta.keys():
                        cpu_layer.sparse_beta[key] = layer.sparse_beta[key].to(device='cpu', non_blocking=non_blocking)
                        transfer_size['beta'] += layer.sparse_beta[key].numel()
                else:
                    cpu_layer.sparse_beta = layer.sparse_beta.to(device='cpu', non_blocking=non_blocking)
                    transfer_size['beta'] += layer.sparse_beta.numel()
        # For intermediate beta.
        if opt_intermediate_beta and net.best_intermediate_betas is not None:
            cpu_net.best_intermediate_betas = OrderedDict()
            for split_layer, all_int_betas_this_layer in net.best_intermediate_betas.items():
                # Single neuron split so far.
                assert 'single' in all_int_betas_this_layer
                assert 'history' not in all_int_betas_this_layer
                assert 'split' not in all_int_betas_this_layer
                cpu_net.best_intermediate_betas[split_layer] = {'single': defaultdict(dict)}
                for intermediate_layer, this_layer_intermediate_betas in all_int_betas_this_layer['single'].items():
                    cpu_net.best_intermediate_betas[split_layer]['single'][intermediate_layer]['lb'] = this_layer_intermediate_betas['lb'].to(device='cpu', non_blocking=non_blocking)
                    cpu_net.best_intermediate_betas[split_layer]['single'][intermediate_layer]['ub'] = this_layer_intermediate_betas['ub'].to(device='cpu', non_blocking=non_blocking)
                    transfer_size['itermediate_beta'] += this_layer_intermediate_betas['lb'].numel() * 2
    print(f'Tensors transferred: {" ".join("{}={:.4f}M".format(k, v / 1048576) for (k, v) in transfer_size.items())}')
    return cpu_net
