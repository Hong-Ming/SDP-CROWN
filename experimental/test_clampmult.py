import torch
import time
from typing import Optional, Tuple
from torch import Tensor
from auto_LiRPA.operators.clampmult import multiply_by_A_signs


def _reference_multiply_by_A_signs(A: Tensor, d_pos: Tensor, d_neg: Tensor,
        b_pos: Optional[Tensor], b_neg: Optional[Tensor], patches_mode: bool) -> Tuple[Tensor, Tensor]:
    """Reference implementation."""
    A_pos = A.clamp(min=0)
    A_neg = A.clamp(max=0)
    A_new = d_pos * A_pos + d_neg * A_neg
    bias_pos = bias_neg = torch.tensor(0.)
    if b_pos is not None:
        if patches_mode:
            bias_pos = torch.einsum('sb...chw,sb...chw->sb...', A_pos, b_pos)
        else:
            bias_pos = torch.einsum('sb...,sb...->sb', A_pos, b_pos)
    if b_neg is not None:
        if patches_mode:
            bias_neg = torch.einsum('sb...chw,sb...chw->sb...', A_neg, b_neg)
        else:
            bias_neg = torch.einsum('sb...,sb...->sb', A_neg, b_neg)
    return A_new, bias_pos + bias_neg


def _speed_test(A, d_pos, d_neg, b_pos, b_neg, patches_mode=False, n_test=20, warmup=3):
    """Benchmarking function."""
    print(f'patches_mode = {patches_mode}, b_pos is {type(b_pos)}, b_neg is {type(b_neg)}')
    total_ref = 0.
    total_new = 0.
    run = ['ref', 'new']
    for i in range(n_test):
        ref_time = new_time = 0.

        if 'ref' in run:
            torch.cuda.synchronize()
            start = time.time()
            ref_A, ref_bias = _reference_multiply_by_A_signs(A, d_pos, d_neg, b_pos, b_neg, patches_mode)
            ref_loss = ref_A.sum() + ref_bias.sum()
            ref_loss.backward()
            torch.cuda.synchronize()
            ref_time = time.time() - start
            ref_gA = A.grad.detach().clone()
            ref_gd_pos = d_pos.grad.detach().clone()
            ref_gd_neg = d_neg.grad.detach().clone()
            ref_gb_pos = b_pos.grad.detach().clone() if b_pos is not None else torch.tensor(0.)
            ref_gb_neg = b_neg.grad.detach().clone() if b_neg is not None else torch.tensor(0.)
            A.grad = d_pos.grad = d_neg.grad = None
            if b_pos is not None:
                b_pos.grad = None
            if b_neg is not None:
                b_neg.grad = None
            del ref_loss

        if 'new' in run:
            torch.cuda.synchronize()
            start = time.time()
            new_A, new_bias = multiply_by_A_signs(A, d_pos, d_neg, b_pos, b_neg, patches_mode)
            new_loss = new_A.sum() + new_bias.sum()
            new_loss.backward()
            torch.cuda.synchronize()
            new_time = time.time() - start
            new_gA = A.grad.detach().clone()
            new_gd_pos = d_pos.grad.detach().clone()
            new_gd_neg = d_neg.grad.detach().clone()
            new_gb_pos = b_pos.grad.detach().clone() if b_pos is not None else torch.tensor(0.)
            new_gb_neg = b_neg.grad.detach().clone() if b_neg is not None else torch.tensor(0.)
            A.grad = d_pos.grad = d_neg.grad = None
            if b_pos is not None:
                b_pos.grad = None
            if b_neg is not None:
                b_neg.grad = None
            del new_loss

        print(f'Loop {i:3d} {"(warmup)" if i < warmup else "        "} time ref {ref_time:.5f} new {new_time:.6f} speedup {ref_time / new_time if i >= warmup else float("nan"):.3f}')
        if i >= warmup:
            total_ref += ref_time
            total_new += new_time

        if 'ref' in run and 'new' in run:
            A_diff = (ref_A - new_A).abs().sum().item() / ref_A.abs().sum().item()
            gA_diff = (ref_gA - new_gA).abs().sum().item() / ref_gA.abs().sum().item()
            bias_diff = (ref_bias - new_bias).abs().sum().item() / (ref_bias.abs().sum().item() + 1e-10)
            gd_pos_diff = (ref_gd_pos - new_gd_pos).abs().sum().item() / ref_gd_pos.abs().sum().item()
            gd_neg_diff = (ref_gd_neg - new_gd_neg).abs().sum().item() / ref_gd_neg.abs().sum().item()
            gb_pos_diff = (ref_gb_pos - new_gb_pos).abs().sum().item() / (ref_gb_pos.abs().sum().item() + 1e-10)
            gb_neg_diff = (ref_gb_neg - new_gb_neg).abs().sum().item() / (ref_gb_neg.abs().sum().item() + 1e-10)
            print(f'                  diff {A_diff} {gA_diff} {bias_diff} {gd_pos_diff} {gd_neg_diff} {gb_pos_diff} {gb_neg_diff}')
            assert A_diff < 1e-6 and bias_diff < 1e-6 and gA_diff < 1e-6 and gd_pos_diff < 1e-6 and gd_neg_diff < 1e-6
            assert gb_pos_diff < 1e-6 and gb_neg_diff < 1e-6


    avg_ref_time = total_ref / (n_test - warmup)
    avg_new_time = total_new / (n_test - warmup)
    print(f'Avg. time: reference {avg_ref_time:.5f} new {avg_new_time:.6f} speedup {avg_ref_time / avg_new_time:.3f}')


if __name__ == '__main__':
    for patches_mode in [True, False]:
        if patches_mode:
            shape = (256, 8, 8, 8, 16, 32)
        else:
            shape = (256, 8, 128, 256)
        A = torch.randn(shape, device='cuda', requires_grad=True)
        d_pos = torch.randn(shape, device='cuda', requires_grad=True)
        d_neg = torch.randn(shape, device='cuda', requires_grad=True)
        b_pos = torch.randn(shape, device='cuda', requires_grad=True)
        b_neg = torch.randn(shape, device='cuda', requires_grad=True)
        _speed_test(A, d_pos, d_neg, None, None, patches_mode=patches_mode)
        _speed_test(A, d_pos, d_neg, None, b_neg, patches_mode=patches_mode)
        _speed_test(A, d_pos, d_neg, b_pos, None, patches_mode=patches_mode)
        _speed_test(A, d_pos, d_neg, b_pos, b_neg, patches_mode=patches_mode)
        print('Press Enter key to continue.')
        input()
        del A, d_pos, d_neg, b_pos, b_neg
