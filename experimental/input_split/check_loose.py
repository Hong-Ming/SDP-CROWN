# DEBUG
check_loose = arguments.Config['bab']['branching']['input_split']['check_loose']
if check_loose and num_iter >= check_loose:
    assert new_dm_lb.shape[0] == dm_lb.shape[0] * 2
    before = (dm_lb - thresholds[:dm_lb.shape[0]]).amax(-1)
    objective = (new_dm_lb.cuda()-thresholds).amax(dim=-1)
    after = objective.reshape(2, dm_lb.shape[0]).amin(dim=0)
    # print('Improvement in this batch')
    # print('before', before)
    # print('after', after)
    print('before worst', before.min())
    print('after worst', after.min())

    good_output_spec = (new_dm_lb.cuda()-thresholds).argmax(dim=-1)
    print('easier output spec:',
            '0:', (good_output_spec == 0).sum(),
            '1:', (good_output_spec == 1).sum())
    verified_output_spec = (new_dm_lb.cuda()-thresholds) > 0
    print('verified output spec:',
            '0:', (verified_output_spec[:, 0]).sum(),
            '1:', (verified_output_spec[:, 1]).sum())
    # ptb = PerturbationLpNorm(x_L=new_dm_l_all, x_U=new_dm_U_all)
    # new_x = BoundedTensor(dm_l, ptb)  # the value of new_x doesn't matter, only pdb matters
    assert new_dm_l_all.ndim == 2
    idx = objective.argmin().item()
    idx = idx % dm_lb.shape[0]
    x_L = dm_l_all[idx]
    x_U = dm_u_all[idx]
    x_M = (x_L + x_U) / 2
    input_dim = dm_l_all.shape[1]
    x_L_debug = x_L.expand(2*input_dim, -1).clone()
    x_U_debug = x_U.expand(2*input_dim, -1).clone()
    for i in range(input_dim):
        x_U_debug[i*2, i] = x_M[i]
        x_L_debug[i*2+1, i] = x_M[i]
    x_debug = BoundedTensor(
        dm_l_all[idx:idx+1].expand(2*input_dim, -1),
        ptb=PerturbationLpNorm(
            x_L=x_L_debug, x_U=x_U_debug))
    reference_interm_bounds = {}
    lb_ibp = net.net.compute_bounds(
        x=(x_debug,), C=cs[:1].expand(2*input_dim, -1, -1),
        method='ibp',
        bound_upper=False, return_A=False)[0]
    for node in net.net.nodes():
        if (node.perturbed
            and isinstance(getattr(node, 'lower', None), torch.Tensor)
            and isinstance(getattr(node, 'upper', None), torch.Tensor)):
            reference_interm_bounds[node.name] = (node.lower, node.upper)
    lb_debug = net.net.compute_bounds(
        x=(x_debug,), C=cs[:1].expand(2*input_dim, -1, -1),
        method=bounding_method, bound_upper=False,
        reference_bounds=reference_interm_bounds
    )[0]
    # print('Before', (dm_lb[idx:idx+1] - thresholds[:1]).amax(dim=-1))
    # print('After',torch.min(
    #     (new_dm_lb.cuda()[idx:idx+1] - thresholds[:1]).amax(dim=-1),
    #     (new_dm_lb.cuda()[idx+dm_lb.shape[0]:idx+dm_lb.shape[0]+1]
    #      - thresholds[:1]).amax(dim=-1)))
    print('Before', (dm_lb[idx:idx+1] - thresholds[:1]))
    print('After left', (new_dm_lb.cuda()[idx:idx+1] - thresholds[:1]))
    print('After right',
            (new_dm_lb.cuda()[idx+dm_lb.shape[0]:idx+dm_lb.shape[0]+1]
            - thresholds[:1]))
    after_ = torch.max(lb_ibp, lb_debug) - thresholds[:1]
    # lb_debug may be looser than lb_ibp
    for i in range(input_dim):
        print(f'Branching {i}')
        print(f'  left {after_[i*2]}')
        print(f'  right {after_[i*2+1]}')
        # left = after_[i*2].amax(dim=-1)
        # right = after_[i*2+1].amax(dim=-1)
        # print(f'Branching {i}', torch.min(left, right))
    print('Selected', split_idx[idx])
    # lA = lA.view(lA.shape[0], lA.shape[1], -1)
    # perturb = (x_U - x_L).unsqueeze(-2)
    # branching_args = arguments.Config['bab']['branching']
    # input_split_args = branching_args['input_split']
    # lA_clamping_thresh = branching_args['sb_coeff_thresh']
    # sb_margin_weight = input_split_args['sb_margin_weight']
    # sb_primary_spec = input_split_args['sb_primary_spec']
    # sb_primary_spec_iter = input_split_args['sb_primary_spec_iter']
    # sb_sum = input_split_args['sb_sum']
    # score1 = (lA[:, sb_primary_spec].abs().clamp(min=lA_clamping_thresh)
    #             * perturb.squeeze(1) / 2
    #         + (lb[:, sb_primary_spec].to(lA.device).unsqueeze(-1)
    #             - thresholds[:, sb_primary_spec].unsqueeze(-1))
    #         * sb_margin_weight)
    # score2 = (lA.abs().clamp(min=lA_clamping_thresh) * perturb / 2
    #         + (lb.to(lA.device).unsqueeze(-1)
    #             - thresholds.unsqueeze(-1)) * sb_margin_weight)
    # score2 = score2.amax(dim=-2)
    import pdb; pdb.set_trace()
