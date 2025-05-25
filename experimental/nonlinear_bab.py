"""Experimental code for nonlinear bab."""

# Debugging the choice of branching points (in the branching heuristic)
lb_ret = torch.empty(
    (layers.shape[0], layers.shape[1]), device=scores.device)
ub_ret = torch.empty(
    (layers.shape[0], layers.shape[1]), device=scores.device)
# TODO randomly generate some potential branching points and compare
assert branching_candidates == 1
num_domains = lb_ret.shape[0]
results_debug = [[] for i in range(num_domains)]
import tqdm
for t in tqdm.tqdm(range(100)):
    if t == 0:
        points_ = points
    else:
        points_ratio = torch.sort(torch.rand_like(points), dim=-1)[0]
        points_ = (lb_ret.unsqueeze(-1) * (1 - points_ratio) +
                ub_ret.unsqueeze(-1) * points_ratio)
    branching_decision, branching_points, _ = self.format_decisions(
        layers, indices, points_)
    split = {
        'decision': branching_decision,
        'points': branching_points
    }
    args_update_bounds_ = copy.deepcopy(args_update_bounds)
    self.net.build_history_and_set_bounds(
        args_update_bounds_, split, mode='breath')
    ret_lbs = self.net.update_bounds(
        args_update_bounds_,
        shortcut=True, beta=self.filter_beta)
    ret_lbs = ret_lbs.reshape(
        branching_candidates, self.num_branches, -1)
    for i in range(num_domains):
        results_debug[i].append({
            'points': points_[i, 0],
            'ret': ret_lbs[0, :, i],
        })
for i in range(num_domains):
    print(lb_ret[i, 0], ub_ret[i, 0])
    for j, r in enumerate(results_debug[i]):
        worst = r['ret'].min()
        if j == 0 or worst > results_debug[i][0]['ret'].min()+0.05:
            print('  ', float(worst),
                r['ret'], r['points'])
        if (r['ret'].min() < 0 and
                r['ret'].min() > results_debug[i][0]['ret'].min()
                + 0.5):
            print('Good improvement??????')
print()

# Refine branching points and naive branching scores
if self.branching_point_refinement:
    act = self.net.split_activations[name][0][0]
    assert lAs[act.name].size(1) == 1
    lAs_ = torch.gather(
        lAs[act.name].squeeze(1), dim=-1, index=indices_
    ).view(-1)[mask]
    bias_ = get_preact_params(act)[indices_].view(-1)[mask]
    points_refined = self.refine_branching_points(
        act, lb_, ub_, lAs_, bias_, points_)
    points_ret_refined.view(-1, num_branching_points)[mask, :] = points_refined

def refine_branching_points(self, act, lower, upper, lA, bias, points_ref=None):
    # Not using optimizable bounds in the branching score computation
    opt_stage = act.opt_stage
    act.opt_stage = None

    score_best = torch.full_like(lower, fill_value=torch.inf)
    points_left = points_ref[:, 0]
    points_right = points_ref[:, 1]
    n = 10
    for i in range(1, n):
        for j in range(i + 1, n):
            points_left_ = lower + (upper-lower) * i*1./n
            points_right_ = lower + (upper-lower) * j*1./n
            score_1 = self._eval_branching_point(act, lower, points_left_, lA, bias, 'area')
            score_2 = self._eval_branching_point(act, points_left_, points_right_, lA, bias, 'area')
            score_3 = self._eval_branching_point(act, points_right_, upper, lA, bias, 'area')
            # sum up
            # score = score_1 + score_2 + score_3
            # # worst segment
            score = torch.max(torch.max(score_1, score_2), score_3)
            good = score < score_best
            score_best = torch.where(good, score, score_best)
            points_left = torch.where(good, points_left_, points_left)
            points_right = torch.where(good, points_right_, points_right)

    points = torch.concat([points_left.unsqueeze(-1),
                            points_right.unsqueeze(-1)], dim=-1)

    # Restore opt_stage
    act.opt_stage = opt_stage
    # Invalidate the saved relaxation that is for branching only
    act.relaxed = False

    return points

def _naive_branching(self, lb, ub, split_masks):
    scores = (torch.rand_like(lb) + 1e-10) * split_masks
    ratio = torch.arange(0, 1, step=1./self.num_branches)[1:].to(lb)
    points = (lb.unsqueeze(-1) * (1 - ratio)
                    + ub.unsqueeze(-1) * ratio)
    scores = scores.flatten(1)
    points = points.flatten(1, -2)
    return scores, points


# _eval_branching_point with beta

def _set_A_options(self, bab=False):
    input_and_act = arguments.Config['bab']['branching'][
        'branching_input_and_activation']
    # For beta_heuristic, A matrices are needed in `build` with full-CROWN
    # rather than build_with_refined_bounds.
    beta_heuristic = not bab and arguments.Config['bab']['branching'][
        'nonlinear_split']['beta_heuristic']
    get_upper_bound = bab and arguments.Config['bab']['get_upper_bound']
    if get_upper_bound or input_and_act:
        self.needed_A_dict = defaultdict(set)
        self.needed_A_dict[self.net.output_name[0]].add(
            self.net.input_name[0])
    elif beta_heuristic:
        self.needed_A_dict = defaultdict(set)
        self.net.get_split_layers()
        for node in self.net.split_layers:
            self.needed_A_dict[node.name].add(self.net.input_name[0])
    # FIXME just use "self.needed_A_dict is not None" without the extra "self.return_A"
    if self.needed_A_dict is not None:
        self.return_A = True

if self.beta_heuristic:
    num_iterations = 100
    lr = 0.01
    debug = True

    beta_l = torch.nn.Parameter(torch.zeros_like(lb))
    beta_u = torch.nn.Parameter(torch.zeros_like(lb))
    opt = torch.optim.Adam([beta_l, beta_u], lr=lr)
    best_bound = None

    # Experimental new heuristic
    A_pos, A_neg = lA.clamp(min=0), lA.clamp(max=0)
    bound = A_pos * lb + A_neg * ub
    A_next = A_pos * lw + A_neg * uw
    bound_ref = bound + A_next * bias  # Standard babsr

    for t in range(num_iterations):
        # Add beta term
        # A_next * z
        # maximize (A_next * z - beta_l * (z-lz) + beta_u * (z-uz))
        bound_ = bound + beta_l * lower - beta_u * upper
        A_next_ = A_next - beta_l + beta_u
        A_next_pos = A_next_.clamp(min=0)
        A_next_neg = A_next_.clamp(max=0)
        A_saved = self.net.A_saved[act.inputs[0].name][self.root_name]
        bound_ += A_next_ * bias
        # bound_ += (A_next_pos * A_saved['lbias']
        #         + A_next_neg * A_saved['ubias'])
        A_next_ = (
            A_next_pos.unsqueeze(-1) * A_saved['lA'].view(
                A_saved['lA'].shape[0], A_saved['lA'].shape[1], -1)
            + A_next_neg.unsqueeze(-1) * A_saved['uA'].view(
                A_saved['uA'].shape[0], A_saved['uA'].shape[1], -1)
        )
        bound_ += self.net.x.ptb.concretize(
            self.net.x, A_next_.transpose(0, 1), sign=-1
        ).transpose(0, 1)
        if best_bound is None:
            best_bound = bound_
        else:
            best_bound = torch.max(bound_, best_bound)

        loss = -bound_.sum()
        if debug and (t + 1) % 10 == 0:
            print(f'iteration {t+1}: loss {loss:.3f}'
                    f' diff {(bound_-bound_ref).abs().mean()}')
        loss.backward()
        opt.step()
        beta_l.data = beta_l.clamp(min=0)
        beta_u.data = beta_u.clamp(min=0)

    if debug:
        print()

    # return bound_ref
    return best_bound



# Double check the consistency of bounds (2/27)
use_betas = True
args_update_bounds = {
    'lower_bounds': d['lower_bounds'],
    'upper_bounds': d['upper_bounds'],
    'slopes': d['slopes'], 'cs': d['cs']
}
if use_betas:
    args_update_bounds.update({
        'betas': d['betas'],
        'history': d['history']
    })
import copy
# Before branching
print('==============================================')
print('Compare bounds with previously computed ones:')
print('reference', d['lower_bounds'][-1])
ret_lbs_base = net.update_bounds(
    copy.deepcopy(args_update_bounds),
    fix_intermediate_layer_bounds=True,
    shortcut=True, beta=use_betas,
    shortcut_method='CROWN-optimized' if use_betas else 'backward',
    beta_bias=True)
print(ret_lbs_base.squeeze(-1))
print('\n')
print('before split_domain...')
import pdb; pdb.set_trace()


# Update alpha when branching (2/23)
def build_history_and_set_bounds(self, d, split, mode='depth', impl_params=None):
    """
    d: Domains
    split: Split decisions
    mode ('depth' or 'breath'): For multiple candidate decisions, whether to
    stack them in the depth direction (to apply all the decisions) or
    breath direction (to try different decisions separately).
    """

    num_domain = d['lower_bounds'][0].shape[0]
    num_split = len(split['decision']) // num_domain
    num_layer = len(d['lower_bounds']) - 1
    device = d['lower_bounds'][0].device
    non_zero_updates = split.get('points', None) is not None
    if non_zero_updates and split['points'].ndim == 2:
        num_branches = split['points'].shape[1] + 1
        multi_branch = True
    else:
        num_branches = 2
        multi_branch = False

    def as_tensor(t):
        return torch.as_tensor(t).to(device=device, non_blocking=True)

    use_simple_upd = num_split == 1 and impl_params is None and not multi_branch
    if use_simple_upd:
        upd_domain = [[] for _ in range(num_layer)]
        upd_idx = [[] for _ in range(num_layer)]
        upd_val = [[] for _ in range(num_layer)]
        upd = [upd_domain, upd_idx, upd_val]
    else:
        upd_hist_l = [[] for _ in range(num_layer)]
        upd_domain_l = [[] for _ in range(num_layer)]
        upd_idx_l = [[] for _ in range(num_layer)]
        upd_hist_u = [[] for _ in range(num_layer)]
        upd_domain_u = [[] for _ in range(num_layer)]
        upd_idx_u = [[] for _ in range(num_layer)]
        upd_val_l = [[] for _ in range(num_layer)]
        upd_val_u = [[] for _ in range(num_layer)]
        upd = [upd_hist_l, upd_hist_u, upd_domain_l, upd_domain_u,
                upd_idx_l, upd_idx_u, upd_val_l, upd_val_u]

    if mode == 'depth':
        # TODO some branching points may be invalid and thus the actual
        # number of branches may be fewer (to allow some neurons to have
        # fewer branching points).
        num_copy = num_branches**num_split # TODO support multiple branches
    else:
        num_copy = num_branches * num_split

    d['lower_bounds'] = [repeat(lb, num_copy, unsqueeze=True) for lb in d['lower_bounds']]
    d['upper_bounds'] = [repeat(ub, num_copy, unsqueeze=True) for ub in d['upper_bounds']]
    history = d.get('history', None)
    new_history = []

    # TODO transpose the first two dimensions, potentially for effiency
    # TODO Maybe slow. Need to speed up.
    if impl_params is not None:
        assert mode == 'depth'
        branch_bounds_implied(
            d, split, num_copy, num_domain, num_split, self.split_indices,
            impl_params, new_history,
            upd_hist_l, upd_hist_u, upd_domain_l, upd_domain_u,
            upd_idx_l, upd_idx_u, upd_val_l, upd_val_u)
    else:
        if history is not None:
            for i in range(num_copy):
                for j in range(num_domain):
                    new_history.append(fast_hist_copy_single(history[j]))
        for i in range(num_domain):
            cycle = 1
            for cur_split in range(num_split):
                # FIXME Inconsistent layer index for new_history (split_indices)
                # and elsewhere.
                layer, idx = split['decision'][cur_split*num_domain+i]
                layer_ = self.split_indices[layer]
                if non_zero_updates:
                    points = split['points'][cur_split*num_domain+i]
                    # TODO Allow some branching points to be invalid
                    num_branches = points.numel() + 1
                else:
                    points = 0.
                    num_branches = 2

                if mode == 'depth':
                    j_iter = range(num_copy)
                else:
                    j_iter = range(cur_split*num_branches,
                                    (cur_split+1)*num_branches)

                branch_idx = 0
                count = 0
                for j in j_iter:
                    history_idx = (
                        -num_copy * num_domain + j * num_domain + i)
                    if use_simple_upd:
                        val = points
                        upd_domain[self.split_indices[layer]].append(i)
                        upd_idx[self.split_indices[layer]].append(idx)
                        upd_val[self.split_indices[layer]].append(val)
                        if history is not None:
                            new_history[history_idx][layer][0].append(idx)
                            new_history[history_idx][layer][1].append(
                                1 - branch_idx * 2)
                            new_history[history_idx][layer][2].append(val)
                    else:
                        if branch_idx + 1 < num_branches:
                            val = points[branch_idx] if multi_branch else points
                            if history is not None:
                                new_history[history_idx][layer][0].append(idx)
                                new_history[history_idx][layer][1].append(-1)
                                new_history[history_idx][layer][2].append(val)
                            upd_hist_u[layer_].append(j)
                            upd_domain_u[layer_].append(i)
                            upd_idx_u[layer_].append(idx)
                            upd_val_u[layer_].append(val)
                        if branch_idx > 0:
                            val = points[branch_idx - 1] if multi_branch else points
                            if history is not None:
                                new_history[history_idx][layer][0].append(idx)
                                new_history[history_idx][layer][1].append(1)
                                new_history[history_idx][layer][2].append(val)
                            upd_hist_l[layer_].append(j)
                            upd_domain_l[layer_].append(i)
                            upd_idx_l[layer_].append(idx)
                            upd_val_l[layer_].append(val)
                    if mode == 'depth':
                        count += 1
                        if count == cycle:
                            branch_idx = (branch_idx + 1) % num_branches
                            count = 0
                    else:
                        branch_idx += 1
                if mode == 'depth':
                    cycle *= num_branches

    if history is not None:
        d['history'] = new_history
    if 'depths' in d:
        if mode == 'depth':
            d['depths'] = [depth + num_split for depth in d['depths']]
        else:
            d['depths'] = [depth + 1 for depth in d['depths']]
        d['depths'] = d['depths'] * num_copy
    if 'slopes' in d:
        new_slopes = defaultdict(dict)
        for k, v in d['slopes'].items():
            new_slopes[k] = {kk: torch.cat([vv] * num_copy, dim=2)
                for kk, vv in v.items()}
        d['slopes'] = new_slopes
    for k in ['split_history', 'cs', 'betas', 'intermediate_betas',
            'thresholds', 'x_Ls', 'x_Us']:
        if k in d:
            d[k] = repeat(d[k], num_copy)
    for k in split:
        if isinstance(split[k], list):
            split[k] = split[k][-num_domain:] * num_copy
        elif isinstance(split[k], torch.Tensor):
            split[k] = split[k][-num_domain:].repeat(
                num_copy, *[1]*(split[k].ndim - 1))

    for upd_list in upd:
        for i in range(len(upd_list)):
            upd_list[i] = as_tensor(upd_list[i])
    # Experimental: update alpha for tanh
    from auto_LiRPA.bound_ops import BoundTanh
    if not use_simple_upd:
        for i in range(num_layer):
            if len(upd_hist_u[i]):
                d['upper_bounds'][i].view(num_copy, num_domain, -1)[
                    upd_hist_u[i], upd_domain_u[i], upd_idx_u[i]
                ] = upd_val_u[i]
                # Experimental
                if isinstance(self.split_activations[i], BoundTanh):
                    raise NotImplementedError
            if len(upd_hist_l[i]):
                d['lower_bounds'][i].view(num_copy, num_domain, -1)[
                    upd_hist_l[i], upd_domain_l[i], upd_idx_l[i]
                ] = upd_val_l[i]
                # Experimental
                if isinstance(self.split_activations[i], BoundTanh):
                    raise NotImplementedError
    else:
        for i in range(num_layer):
            if len(upd_domain[i]):
                assert d['lower_bounds'][i].shape[0] == 2
                # act = self.split_activations[i]
                # if isinstance(act, BoundTanh):
                #     mask_both = torch.logical_and(
                #         d['lower_bounds'][i][0].view(num_domain, -1)[
                #             upd_domain[i], upd_idx[i]] < 0,
                #         d['upper_bounds'][i][0].view(num_domain, -1)[
                #             upd_domain[i], upd_idx[i]] > 0
                #     )
                d['lower_bounds'][i][0].view(num_domain, -1)[
                    upd_domain[i], upd_idx[i]] = upd_val[i]
                d['upper_bounds'][i][1].view(num_domain, -1)[
                    upd_domain[i], upd_idx[i]] = upd_val[i]
                # if isinstance(act, BoundTanh):
                #     print(f'Updated alpha for {act}')
                #     for slopes in d['slopes'][act.name].values():
                #         assert slopes.shape[1] == 1
                #         # For neurons whose pre-activation bounds have certain signs,
                #         # copy alpha from the case where pre-activation bounds have uncertain signs.
                #         # TODO only copy when the sign really changes
                #         slopes[0:2, 0].data.view(2, 2, num_domain, -1)[:, :, upd_domain[i], upd_idx[i]][:, :, mask_both] = (
                #             slopes[6:8, 0].view(2, 2, num_domain, -1)[:, :, upd_domain[i], upd_idx[i]][:, :, mask_both])
                #         slopes[2:4, 0].data.view(2, 2, num_domain, -1)[:, :, upd_domain[i], upd_idx[i]][:, :, mask_both] = (
                #             slopes[4:6, 0].view(2, 2, num_domain, -1)[:, :, upd_domain[i], upd_idx[i]][:, :, mask_both])
                #         import pdb; pdb.set_trace()

    d['lower_bounds'] = [lb.view(-1, *lb.shape[2:]) for lb in d['lower_bounds']]
    d['upper_bounds'] = [ub.view(-1, *ub.shape[2:]) for ub in d['upper_bounds']]




# Branching score with beta. Wrong. Do not use.
from auto_LiRPA.bound_ops import BoundLinear
A_next = lA.clamp(min=0) * lw + lA.clamp(max=0) * uw
assert isinstance(self.inputs[0], BoundLinear)
linear = self.inputs[0]
weight = linear.inputs[1].param
scores =  (lA.clamp(min=0) * lb + lA.clamp(max=0) * ub
    + A_next * bias)
if isinstance(getattr(linear.inputs[0].inputs[0], 'lower', None),
                torch.Tensor):
    lower_ = linear.inputs[0].inputs[0].lower
    upper_ = linear.inputs[0].inputs[0].upper
    assert isinstance(linear.inputs[0], BoundTanh)
    # This gives the input bounds to the linear layer
    lower_ = linear.inputs[0](lower_)
    upper_ = linear.inputs[0](upper_)

    A_next = A_next.matmul(weight)
    approx = A_next.clamp(min=0) * lower_ + A_next.clamp(max=0) * upper_

    beta_l = torch.nn.Parameter(torch.zeros_like(lower))
    beta_u = torch.nn.Parameter(torch.zeros_like(upper))
    num_iterations = 10
    opt = torch.optim.Adam([beta_l, beta_u], lr=self.lr_heuristic_beta)
    with torch.enable_grad():
        for i in range(num_iterations):
            A_next_ = (
                A_next.detach()
                - beta_l.matmul(weight.detach())
                + beta_u.matmul(weight.detach()))
            bias_ = (-beta_l * (bias - lower).detach()
                        + beta_u * (bias - upper).detach())
            approx_ = (
                A_next_.clamp(min=0) * lower_
                + A_next_.clamp(max=0) * upper_
                + bias_)
            loss = (-approx_).sum()
            print(f'{loss:.3f}', end='\t')
            opt.zero_grad()
            loss.backward()
            opt.step()
            beta_l.data = beta_l.clamp(min=0)
            beta_u.data = beta_u.clamp(min=0)
        print()
        print('imp:', (approx_-approx).max(),
                (approx_-approx).mean())
        if (approx_-approx).max()>10:
            import pdb; pdb.set_trace()
        approx = approx_

    scores += approx


return scores

# An alternative to the BaBSR-like branching score
# (A_pos*lw+A_neg*uw)_pos * l + (A_pos*lw+A_neg*uw)_neg * u + A_pos*lb + A_neg*ub
A0 = A_pos * lw_0 + A_neg * uw_0
score0 = (A0.clamp(min=0) * lower + A0.clamp(max=0) * upper
        + A_pos * lb_0 + A_neg * ub_0)
A1 = A_pos * lw_1 + A_neg * uw_1
score1 = (A1.clamp(min=0) * lower + A1.clamp(max=0) * points
        + A_pos * lb_1 + A_neg * ub_1)
A2 = A_pos * lw_2 + A_neg * uw_2
score2 = (A2.clamp(min=0) * points + A2.clamp(max=0) * upper
        + A_pos * lb_2 + A_neg * ub_2)
score = torch.min(score1, score2) - score0

# Debugging: use beta-crown to check the potential improvement (02/06)
# DEBUG
import os
if os.environ.get('DEBUG', None):
    if True and split_depth == 1:
        def _get_bounds(points, use_betas=False):
            branching_decision, branching_points, split_depth = (
                self.format_decisions(layers, indices, points))
            args_update_bounds = {
                'lower_bounds': lb, 'upper_bounds': ub,
                'slopes': domains['slopes'], 'cs': domains['cs']
            }
            if use_betas:
                args_update_bounds.update({
                    'betas': domains['betas'],
                    'history': domains['history']
                })
            split = {
                'decision': branching_decision,
                'points': branching_points
            }
            self.net.build_history_and_set_bounds(args_update_bounds, split)
            ret_lbs = self.net.update_bounds(
                args_update_bounds, fix_intermediate_layer_bounds=True,
                shortcut=True, beta=use_betas,
                shortcut_method='CROWN-optimized' if use_betas else 'backward')
            return ret_lbs, args_update_bounds
        ret_lbs, args_update_bounds = _get_bounds(points_ret)
        ret_lbs_refined, args_update_bounds_refined = _get_bounds(points_ret_refined)
        print('Improvement:')
        if lb[-1].numel() <= 5:
            print('  From:', lb[-1])
            print('  To:', ret_lbs,
                ret_lbs.view(3, -1).min(dim=0).values)
            print('  To (refined):', ret_lbs_refined,
                ret_lbs_refined.view(3, -1).min(dim=0).values)

        points_ = []
        n = 15
        for i in range(1, n):
            for j in range(i+1, n):
                points_.append(torch.concat([
                    (i/n*lb_ret+(1-i/n)*ub_ret).unsqueeze(-1),
                    (j/n*lb_ret+(1-j/n)*ub_ret).unsqueeze(-1),
                ], dim=-1))
        points_ = torch.concat(points_, dim=1)

        import tqdm
        ret_best = torch.full((points_.shape[0],), -np.inf).to(ret_lbs)
        for i in tqdm.tqdm(range(points_.shape[1])):
            ret_lbs_, args_update_bounds_ = _get_bounds(points_[:, i:i+1])
            ret_lbs_ = ret_lbs_.view(3, -1).min(dim=0).values.view(-1)
            ret_best = torch.max(ret_best, ret_lbs_)
        print('Potentially best:', ret_best)
        ret_best = ret_best.clamp(max=0)
        ret_lbs = ret_lbs.view(3, -1).min(dim=0).values.clamp(max=0)
        ret_lbs_refined = ret_lbs_refined.view(3, -1).min(dim=0).values.clamp(max=0)
        print('improvement compared to ret_lbs',
                ret_best-ret_lbs)
        print('improvement compared to ret_lbs_refined',
                ret_best-ret_lbs_refined)
        print('improvement compared to ret_lbs_refined (min)',
                torch.min(ret_best-ret_lbs, ret_best-ret_lbs_refined))
        # import pdb; pdb.set_trace()
        print()



# Debugging branching heuristic (2/5)
import os
if os.environ.get('DEBUG', None):
    if True and split_depth == 1:
        def _get_bounds(points, use_betas=True):
            branching_decision, branching_points, split_depth = (
                self.format_decisions(layers, indices, points))
            args_update_bounds = {
                'lower_bounds': lb, 'upper_bounds': ub,
                'slopes': domains['slopes'], 'cs': domains['cs']
            }
            if use_betas:
                args_update_bounds.update({
                    'betas': domains['betas'],
                    'history': domains['history']
                })
            split = {
                'decision': branching_decision,
                'points': branching_points
            }
            self.net.build_history_and_set_bounds(args_update_bounds, split)
            ret_lbs = self.net.update_bounds(
                args_update_bounds, beta=use_betas,
                fix_intermediate_layer_bounds=True, shortcut=True)
            return ret_lbs, args_update_bounds
        # ret_lbs, args_update_bounds = _get_bounds(points_ret)
        # print(points_ret_refined)
        # ret_lbs_refined, args_update_bounds_refined = _get_bounds(points_ret_refined)
        # print('Improvement:')
        # print('  From:', lb[-1])
        # print('  To:', ret_lbs)
        # print('  To (refined):', ret_lbs_refined)

        ratio = torch.arange(0,1,0.5).to(lb_ret)
        points_3 = ((ub_ret-lb_ret)*ratio+lb_ret).unsqueeze(0)
        ret_lbs_3, args_update_bounds_3 = _get_bounds(points_3)
        # ret_lbs_3, args_update_bounds_3 = _get_bounds(points_3, use_betas=False)
        relax3 = (
            self.net.net['/45'].lw[:, :, -1:].clone(),
            self.net.net['/45'].lb[:, :, -1:].clone(),
            self.net.net['/45'].uw[:, :, -1:].clone(),
            self.net.net['/45'].ub[:, :, -1:].clone(),
        )
        print(self.net.net['/45'].lw[:, :, -1, 113])
        print(self.net.net['/45'].lb[:, :, -1, 113])
        print(self.net.net['/45'].uw[:, :, -1, 113])
        print(self.net.net['/45'].ub[:, :, -1, 113])
        print(self.net.net['/45'].inputs[0].lower[:, 113])
        print(self.net.net['/45'].inputs[0].upper[:, 113])
        print(ret_lbs_3)

        # Search
        ratio = torch.arange(0,1,0.01).to(lb_ret)
        points_1 = ((ub_ret-lb_ret)*ratio+lb_ret).unsqueeze(0)
        ret_lbs_1, args_update_bounds_1 = _get_bounds(points_1)
        relax1 = (
            self.net.net['/45'].lw[:, :, -1:].clone(),
            self.net.net['/45'].lb[:, :, -1:].clone(),
            self.net.net['/45'].uw[:, :, -1:].clone(),
            self.net.net['/45'].ub[:, :, -1:].clone(),
        )
        print(self.net.net['/45'].lw[:, :, -1, 113])
        print(self.net.net['/45'].lb[:, :, -1, 113])
        print(self.net.net['/45'].uw[:, :, -1, 113])
        print(self.net.net['/45'].ub[:, :, -1, 113])
        print(self.net.net['/45'].inputs[0].lower[:, 113])
        print(self.net.net['/45'].inputs[0].upper[:, 113])
        print()

        # ratio = torch.arange(0,1,0.05).to(lb_ret)
        # points_2 = ((ub_ret-lb_ret)*ratio+lb_ret).unsqueeze(0)
        # ret_lbs_2, args_update_bounds_2 = _get_bounds(points_2)

        ret_lbs_1 = ret_lbs_1.reshape(-1, points_1.shape[1])
        ret_lbs_3 = ret_lbs_3.reshape(-1, points_3.shape[1])
        print(ret_lbs_1, ret_lbs_1.min())
        print(ret_lbs_3, ret_lbs_3.min())

        import pdb; pdb.set_trace()



# Refine points by binary search
def refine_branching_points(self, lower, upper, lA, bias, points_ref=None):
    # Not using optimizable bounds in the branching score computation
    opt_stage = self.opt_stage
    self.opt_stage = None

    before = self._eval_branching_point(lower, upper, lA, bias)
    points_left = lower * 2./3 + upper * 1./3
    points_right = lower * 1./3 + upper * 2./3
    points = torch.concat([points_left.unsqueeze(-1),
                            points_right.unsqueeze(-1)], dim=-1)
    before = self._eval_branching_point(lower, upper, lA, bias)
    score_1 = self._eval_branching_point(lower, points_left, lA, bias)
    score_2 = self._eval_branching_point(points_left, points_right, lA, bias)
    score_3 = self._eval_branching_point(points_right, upper, lA, bias)
    score = torch.min(torch.min(score_1, score_2), score_3)

    score_best = score_base = score
    points_left_best = points_left
    points_right_best = points_right

    left = score
    right = score + 0.05
    # right = score+(score-before)*4 # try to boost the improvement
    if (score_base-before).max()<0.1:
        num_iterations = 0
    else:
        num_iterations = 5
    for _ in range(num_iterations):
        mid = (left + right) / 2
        points_left = left
        length = upper - lower
        for k in range(1, 6):
            # 1./(2^k)
            points_left_ = points_left + length / (2**k)
            score_ = self._eval_branching_point(lower, points_left_, lA, bias)
            points_left = torch.where(score_ >= mid, points_left_, points_left)
        length = upper - points_left
        points_right = points_left
        for k in range(1, 6):
            # 1./(2^k)
            points_right_ = points_right + length / (2**k)
            score_ = self._eval_branching_point(points_left, points_right_, lA, bias)
            points_right = torch.where(score_ >= mid, points_right_, points_right)
        score_ = torch.min(
            torch.min(
                self._eval_branching_point(lower, points_left, lA, bias),
                self._eval_branching_point(points_right, upper, lA, bias),
            ),
            self._eval_branching_point(points_left, points_right, lA, bias),
        )
        good = score_ >= mid
        left = torch.where(good, mid, left)
        right = torch.where(good, right, mid)
        score_best = torch.where(good, score_, score_best)
        points_left_best = torch.where(good, points_left, points_left_best)
        points_right_best = torch.where(good, points_right, points_right_best)

    points_left = points_left_best
    points_right = points_right_best
    score = score_best - before
    points = torch.concat([points_left.unsqueeze(-1),
                            points_right.unsqueeze(-1)], dim=-1)

    if True:
        # TODO not every layer needs a sophisticated method for finding
        # branching points!
        print((score_best-score_base).mean(),
            (score_base-before).mean())
        print((score_best-score_base).max(),
            (score_base-before).max())
        print(num_iterations)
        print()
        # import pdb; pdb.set_trace()

    # Restore opt_stage
    self.opt_stage = opt_stage
    # Invalidate the saved relaxation that is for branching only
    self.relaxed = False

    return points


# Iteratively update the worst/best segment
elif branching_point_method == 'ternary':
    before = self._eval_branching_point(lower, upper, lA, bias)

    num_iterations = 10
    l1, r1 = lower, torch.max(lower, upper - 1e-2)
    l2, r2 = torch.min(lower + 1e-2, upper), upper

    for _ in range(num_iterations):
        m1 = (l1 + r1) / 2
        m2 = (l2 + r2) / 2
        s1 = self._eval_branching_point(lower, m1, lA, bias)
        s2 = self._eval_branching_point(m1, m2, lA, bias)
        s3 = self._eval_branching_point(m2, upper, lA, bias)
        s1_worst = torch.logical_and(s1 < s2, s1 < s3)
        s1_best = torch.logical_and(s1 > s2, s1 > s3)
        s3_worst = torch.logical_and(s3 < s1, s3 < s2)
        s3_best = torch.logical_and(s3 > s1, s3 > s2)
        s2_worst = torch.logical_and(
            torch.logical_not(s1_worst), torch.logical_not(s3_worst))
        l1 = torch.where(torch.logical_and(s2_worst, s1_best), m1, l1)
        r1 = torch.where(s1_worst, m1, r1)
        l2 = torch.where(s3_worst, m2, l2)
        r2 = torch.where(torch.logical_and(s2_worst, s3_best), m2, r2)

        # points_left = mid = (left + right) / 2
        # score_1 = self._eval_branching_point(
        #     lower, points_left, lA, bias)

        # left_2 = mid
        # right_2 = upper
        # num_iterations_2 = 5
        # for _  in range(num_iterations_2):
        #     points_right = mid_2 = (left_2 + right_2) / 2
        #     score_2 = self._eval_branching_point(
        #         points_left, mid_2, lA, bias)
        #     right_2 = torch.where(score_2 > score_1, right_2, mid_2)
        #     left_2 = torch.where(score_2 > score_1, mid_2, left_2)

        # score_3 = self._eval_branching_point(
        #     points_right, upper, lA, bias)
        # left = torch.where(score_3 > score_1, left, mid)
        # right = torch.where(score_3 > score_1, mid, right)
    points_left = m1
    points_right = m2
    print('!', (points_left>points_right).any())
    score_1 = self._eval_branching_point(lower, points_left, lA, bias)
    score_2 = self._eval_branching_point(points_left, points_right, lA, bias)
    score_3 = self._eval_branching_point(points_right, upper, lA, bias)
    score = torch.min(torch.min(score_1, score_2), score_3) - before

    if score.max() >= 0.05:
        for i in range(score.shape[1]):
            if score[0,i]<0.05:
                continue
            print(f'{i} {lower[0,i]:.3f} {upper[0,i]:.3f} '
                f'{points_left[0,i]:.3f} {points_right[0, i]:.3f} ')
            print(f'  {score[0,i]:.3f} {score_1[0,i]:.3f} '
                f'{score_2[0, i]:.3f} {score_3[0, i]:.3f}')
        print(score.max())
        import pdb; pdb.set_trace()
    # score_1 = self._eval_branching_point_2(lower, points_left, lA, bias)
    # score_2 = self._eval_branching_point_2(points_left, points_right, lA, bias)
    # score_3 = self._eval_branching_point_2(points_right, upper, lA, bias)
    # for i in range(10):
    #     print(f'{i} {lower[0,i]:.3f} {upper[0,i]:.3f} '
    #         f'{points_left[0,i]:.3f} {points_right[0, i]:.3f} ')
    #     print(f'  {score_1[0,i]:.3f} {score_2[0,i]:.3f} {score_3[0,i]:.3f}')
    # import pdb; pdb.set_trace()

    points = torch.concat([points_left.unsqueeze(-1),
                            points_right.unsqueeze(-1)], dim=-1)


# Two binary search levels to determine branching points
num_iterations = 10
for _ in range(num_iterations):
    points_left = mid = (left + right) / 2
    score_1 = self._eval_branching_point(
        lower, points_left, lA, bias)

    left_2 = mid
    right_2 = upper
    num_iterations_2 = 5
    for _  in range(num_iterations_2):
        points_right = mid_2 = (left_2 + right_2) / 2
        score_2 = self._eval_branching_point(
            points_left, mid_2, lA, bias)
        right_2 = torch.where(score_2 > score_1, right_2, mid_2)
        left_2 = torch.where(score_2 > score_1, mid_2, left_2)

    score_3 = self._eval_branching_point(
        points_right, upper, lA, bias)
    left = torch.where(score_3 > score_1, left, mid)
    right = torch.where(score_3 > score_1, mid, right)


# Bad methods for branching points.
# These are worse than branching in the middle.
elif branching_point_method == 'ternary_search':
    points_lo, points_hi = lower, upper
    num_iterations = 10
    for _ in range(num_iterations):
        points_m1 = 2./3 * points_lo + 1./3 * points_hi
        points_m2 = 1./3 * points_lo + 2./3 * points_hi
        scores_1 = self._eval_branching_point(
            lower, upper, lA, bias, points_m1)
        scores_2 = self._eval_branching_point(
            lower, upper, lA, bias, points_m2)
        points_lo = torch.where(
            scores_1 < scores_2, points_m1, points_lo)
        points_hi = torch.where(
            scores_1 < scores_2, points_hi, points_m2)
    points = points_lo
elif branching_point_method == 'binary_search':
    # Run a binary search to balance the left branch and the right branch
    points_lo, points_hi = lower, upper
    num_iterations = 10
    for _ in range(num_iterations):
        points_mid = (points_lo + points_hi) / 2
        gap_left = self._get_gap(points_lo, points_mid)
        gap_right = self._get_gap(points_mid, points_hi)
        points_lo = torch.where(gap_left < gap_right, points_mid, points_lo)
        points_hi = torch.where(gap_left < gap_right, points_hi, points_mid)
    points = points_lo
elif branching_point_method == 'smart':
    # babsr-like

    # experimental: search for now
    for i in range(0, 6):
        points = (upper - lower) * i / 5. + lower
        after = self._eval_branching_point(
            lower, upper, lA, bias, points)
        if i == 0:
            points_best = points
            score_best = after
        else:
            points_best = torch.where(after > score_best, points, points_best)
            score_best = torch.max(score_best, after)

    points = points_best


# Branching heuristic via integral
def _eval_branching_point_2(self, lower, upper, lA, bias):
    # Compute area
    # from l to u
    # int_{l}^u w*x+b = w*x^2/2 + b*x
    # uw*upper^2/2 + ub*upper - uw*lower^2/2 + ub*lower

    lw, lb, uw, ub = self._get_relaxation(lower, upper)
    int_u = uw*(upper**2-lower**2)/2 + ub*(upper-lower)
    int_l = lw*(upper**2-lower**2)/2 + lb*(upper-lower)
    return int_u - int_l

    # return (lA.clamp(min=0) * (lb + lw * bias)
    #         + lA.clamp(max=0) * (ub + uw * bias))


def refine_branching_points():
    """In branching_heuristics.py"""
    assert lAs[idx].size(1) == 1
    lb_ = torch.gather(lb[idx], dim=-1, index=indices).view(-1)[mask]
    ub_ = torch.gather(ub[idx], dim=-1, index=indices).view(-1)[mask]
    lA_ = torch.gather(
        lAs[idx][:, 0], dim=-1, index=indices).view(-1)[mask]
    if isinstance(act.inputs[0], BoundLinear):
        bias = act.inputs[0].inputs[2].param
    else:
        raise NotImplementedError
    bias_ = torch.gather(bias, dim=-1, index=indices.view(-1))[mask]
    points_ret.view(-1)[mask] = act.refine_branching_points(
        lb_, ub_, lA_, bias_)

def optimize_for_points():
    points = torch.nn.Parameter(points)
    opt = torch.optim.SGD([points], lr=1e-4)
    for i in range(10):
        scores = self._eval_branching_point(lower, upper, lA, bias, points)
        scores = scores.sum()
        opt.zero_grad()
        import pdb; pdb.set_trace()
        (-scores).backward()
        opt.step()
        points.data = torch.clamp(points, lower, upper)
        print(f'iteration {i+1}, scores {scores.sum():.5f}')
    import pdb; pdb.set_trace()


def search_for_points():
    points_best = points
    score_best = score_init

    for i in range(0, 20):
        points_ = (upper - lower) * i / 20. + lower
        scores_ = self._eval_branching_point(
            lower, upper, lA, bias, points_)
        points_best = torch.where(scores_ > score_best, points_, points_best)
        score_best = torch.max(score_best, scores_)
        print(i, scores_)
    print('init', score_init)
    print('best', score_best)

