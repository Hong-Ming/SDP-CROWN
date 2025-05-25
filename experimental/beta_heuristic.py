"""Beta heuristic for nonlinear split.

Last working master: 7d58719bd432cc9a0c937ca8769e016760971146
"""
import torch
import torch.nn as nn
from heuristics.nonlinear.utils import set_roots


class BetaHeuristic:
    def __init__(self, net, backward_propagation_func, num_branches, iterations):
        self.net = net
        self.model = net.net
        self.roots = self.model.roots()
        self.backward_propagation_func = backward_propagation_func
        self.num_branches = num_branches
        self.iterations = iterations

    def get_bound(self, domains, node, lAs, lb, ub, A_before, bound_before,
                  start_nodes, branch_idx, num_branches):
        device = A_before.device
        beta_params = [
            nn.Parameter(torch.zeros(A_before.shape[1:3], device=device)),
            nn.Parameter(torch.zeros(A_before.shape[1:3], device=device))]
        for idx, history in enumerate(domains['history']):
            history = history[node.name]
            if history[0] != []:
                past_beta = domains['betas'][idx][node.name].to(device)
                beta_params.append(torch.nn.Parameter(past_beta))

        if num_branches == 0 and len(beta_params) == 2:
            return bound_before

        optimizer = torch.optim.Adam(beta_params, lr=0.05)
        for _ in range(self.iterations):
            output = self.compute_bounds_with_beta(
                lAs, A_before, bound_before, lb, ub, node, start_nodes,
                beta_info=(domains['history'], beta_params, branch_idx, num_branches))
            loss = output.neg()
            optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            print(f'loss at iter {_+1}: {loss:.4f}')
            for param in beta_params:
                param.data = (param >= 0) * param.data
        print()

        with torch.no_grad():
            bound = self.compute_bounds_with_beta(
                lAs, A_before, bound_before, lb, ub, node, start_nodes,
                beta_info=(domains['history'], beta_params, branch_idx, num_branches))

        return bound

    def compute_bounds_with_beta(self, lAs, A_before, bound_before, lb, ub,
                                 node, start_nodes, beta_info):
        batch_size = bound_before.shape[0]
        A, bound, _ = self.backward_propagation_func(
            lAs, lb, ub, node, start_nodes,
            include_beta=beta_info)

        diff_A = A - A_before
        A_ = A_before.sum(dim=2, keepdim=True) + diff_A
        self.roots[0].lA = A_.transpose(1, 2).reshape(-1, batch_size, A.shape[-1])

        bound = bound.transpose(1, 2).reshape(
            -1, batch_size).transpose(0, 1)
        bound = self.model.concretize(bound.shape[0], bound.shape[1])[0]

        shape = bound_before.transpose(1, 2).shape
        bound = bound.reshape(shape).transpose(1, 2)
        return bound

    def apply_beta(self, A_, bound_, beta_info, node_pre, lb, ub):
        history, beta_params, idx_branch, num_branches = beta_info
        beta_1 = beta_params[0]
        beta_2 = beta_params[1]
        device = beta_1.device

        if idx_branch + 1 < num_branches:
            A_ = A_ + beta_1.unsqueeze(0)
            bound_ = bound_ - beta_1 * ub[node_pre.name]
        if idx_branch > 0:
            A_ = A_ - beta_2.unsqueeze(0)
            bound_ = bound_ + beta_2 * lb[node_pre.name]

        beta_idx = 2
        for idx, hist in enumerate(history):
            hist = hist[node_pre.name]
            if not isinstance(hist[0], list):
                beta_value = hist[1].to(device) * beta_params[beta_idx]
                beta_idx += 1
                A_[0, idx] = node_pre.non_deter_scatter_add(
                    A_[0, idx], dim=0, index=hist[0].to(device),
                    src=beta_value.neg())
                beta_bias = beta_value * hist[2].to(device)
                bound_[0, idx] = node_pre.non_deter_scatter_add(
                    bound_[0, idx], dim=0, index=hist[0].to(device),
                    src=beta_bias)

        return A_, bound_


class BetaHeuristicSimple(BetaHeuristic):
    """A relatively simple version of the beta heuristic.

    It only estimates a single bound with beta,
    without bound_before, bound_after, diff_A, etc.

    It's used for deciding branching points, not neurons.
    """
    def __init__(self, net, backward_propagation_func, num_branches, iterations):
        super().__init__(net, backward_propagation_func, num_branches, iterations)

    def get_bound(self, domains, mask, decisions, node):
        domains = {
            'lower_bounds': {
                k: v[mask]
                for k, v in domains['lower_bounds'].items()
            },
            'upper_bounds': {
                k: v[mask]
                for k, v in domains['upper_bounds'].items()
            },
            'alphas': {
                k: {kk: vv[:, :, mask]
                    for kk, vv in v.items()}
                for k, v in domains['alphas'].items()
            },
            'cs': domains['cs'][mask],
            'thresholds': domains['thresholds'][mask],
            'betas': [
                domains['betas'][k]
                for k in range(mask.shape[0]) if mask[k]
            ],
            'history': [
                domains['history'][k]
                for k in range(mask.shape[0]) if mask[k]
            ],
            'lAs': {
                k: v[mask]
                for k, v in domains['lAs'].items()
            },
        }
        branching_decision, branching_points, _ = decisions
        split = {
            'decision': branching_decision,
            'points': branching_points,
        }
        self.net.build_history_and_set_bounds(domains, split, mode='breath')
        splits_per_example = self.net.set_beta(domains, bias=True)
        domains['betas'] = [
            {
                k: self.net.net[k].sparse_betas[0].val[
                    i, :splits_per_example[i][k]]
                for k in splits_per_example[i]
            }
            for i in range(len(splits_per_example))
        ]

        name = node.name
        lAs, lb, ub = domains['lAs'], domains['lower_bounds'], domains['upper_bounds']
        start_nodes = [act[0] for act in self.net.split_activations[name]]

        beta_params = []
        for idx, history in enumerate(domains['history']):
            history = history[name]
            if history[0] != []:
                beta_params.append(torch.nn.Parameter(
                    domains['betas'][idx][name]))

        optimizer = torch.optim.Adam(beta_params, lr=0.05)
        for i in range(self.iterations):
            output = self.compute_bounds_with_beta(
                lAs, lb, ub, node, start_nodes,
                beta_info=(domains['history'], beta_params))
            loss = output.neg()
            optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            print(f'loss at iter {i+1}: {loss:.4f}')
            for param in beta_params:
                param.data = (param >= 0) * param.data
        print()

        with torch.no_grad():
            bound = self.compute_bounds_with_beta(
                lAs, lb, ub, node, start_nodes,
                beta_info=(domains['history'], beta_params))

        return bound

    def compute_bounds_with_beta(self, lAs, lb, ub, node, start_nodes, beta_info):
        A, bound, _ = self.backward_propagation_func(
            lAs, lb, ub, node, start_nodes, include_beta=beta_info)
        dim_output, batch_size = bound.shape[:2]
        x_new = self.net.expand_x(batch_size)
        set_roots(self.roots, x_new, A)
        # (batch_size, dim_output)
        bound_from_A = self.model.concretize(
            batch_size, dim_output,
            torch.zeros((batch_size, dim_output), device=bound.device))[0]
        # (batch_size, dim_output)
        bound = bound_from_A + bound.permute(1, 2, 0).sum(dim=1)
        return bound

    def apply_beta(self, A_, bound_, beta_info, node_pre):
        history, beta_params = beta_info
        device = A_.device
        beta_idx = 0
        for idx, hist in enumerate(history):
            hist = hist[node_pre.name]
            if hist[0] != []:
                beta_value = hist[1].to(device) * beta_params[beta_idx]
                beta_idx += 1
                A_[0, idx] = node_pre.non_deter_scatter_add(
                    A_[0, idx], dim=0, index=hist[0].to(device),
                    src=beta_value.neg())
                beta_bias = beta_value * hist[2].to(device)
                bound_[0, idx] = node_pre.non_deter_scatter_add(
                    bound_[0, idx], dim=0, index=hist[0].to(device),
                    src=beta_bias)
        return A_, bound_
