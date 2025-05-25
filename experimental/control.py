"""Experimental code for the control project."""

# For debugging the heuristic
# if num_iter > 120 and too_bad[worst_idx]:
#     torch.save((x_L[worst_idx], x_U[worst_idx], thresholds[worst_idx]), '/home/zhouxingshi/nfs/experiments/control/0909/debug.pth')

import os
if os.environ.get('DEBUG_TIGHTNESS', 0):
    from auto_LiRPA.bound_ops import BoundRelu, BoundLeakyRelu
    # for node in self.net.nodes():
    #     if node.perturbed and isinstance(node, (BoundRelu, BoundLeakyRelu)):
    #         unstable = torch.logical_and(node.inputs[0].lower<0,
    #                                     node.inputs[0].upper>0)
    #         print(node, f'{unstable.int().sum()*1./unstable.numel()}')
    # print()
    for node in self.net.nodes():
        if (isinstance(getattr(node, 'lower', None), torch.Tensor)
                and isinstance(getattr(node, 'upper', None), torch.Tensor)
                and node.perturbed):
            center = (node.lower + node.upper) / 2
            print(node)
            diff = node.upper - node.lower
            print('  ', float(diff.mean()), end=' ')
            print(float((diff / center.abs().clamp(min=1e-8)).mean()))
            # print('  ', float((node.upper - node.lower).mean() / center.std().clamp(min=1e-8)))
    print()
if os.environ.get('DEV_Q', 0):
    # Check if the second output is monotonic
    q_cho = self.net['/266/q_cholesky'].param
    q = q_cho.matmul(q_cho.t())
    assert (self.net['/259'].inputs[1].value == 0).all()
    # check the sign of x.matmul(q)
    grad_L = (x_unverified.ptb.x_L.matmul(q.clamp(min=0))
            + x_unverified.ptb.x_U.matmul(q.clamp(max=0)))
    grad_U = (x_unverified.ptb.x_U.matmul(q.clamp(min=0))
            + x_unverified.ptb.x_L.matmul(q.clamp(max=0)))
    stable_grad = torch.logical_or(grad_L>=0, grad_U<=0).all(dim=-1)
    print('stable grad for Q:', stable_grad.float().sum()/stable_grad.numel())

    if stable_grad.any():
        x_q = (grad_L>=0) * x_unverified.ptb.x_L + (grad_U<=0) * x_unverified.ptb.x_U
        xqx = (x_q*x_q.matmul(q)).sum(dim=-1)
        print('Improve the Q term?')
        print(lb_crown[stable_grad, 1][:10] - thresholds[0, 1])
        print(xqx[stable_grad][:10] - thresholds[0, 1])
        lb_crown[stable_grad, 1] = torch.max(
            lb_crown[stable_grad, 1], xqx[stable_grad])

