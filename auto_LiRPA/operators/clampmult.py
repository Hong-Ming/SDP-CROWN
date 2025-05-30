#########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2025 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##                     Zhouxing Shi <zshi@cs.ucla.edu> (UCLA)          ##
##                     Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""Element multiplication with the A matrix based on its sign."""
import torch
from typing import Optional, Tuple
from torch import Tensor
from ..patches import Patches


torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)


class ClampedMultiplication(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    @torch.jit.script
    def clamp_mutiply_forward(A: Tensor, d_pos: Tensor, d_neg: Tensor,
            b_pos: Optional[Tensor], b_neg: Optional[Tensor], patches_mode: bool,
            reduce_bias: bool = False, same_slope: bool = False
        ) -> Tuple[Tensor, Tensor]:
        """Forward operations; actually the same as the reference implementation."""
        A_pos = A.clamp(min=0)
        A_neg = A.clamp(max=0)    
        if same_slope:
            # "same-slope" option is enabled; lower and upper bounds use the same A.
            A_new = d_pos * A          
        else:  
            A_new = d_pos * A_pos + d_neg * A_neg
        
        bias_pos = bias_neg = torch.zeros(
                (), dtype=A_new.dtype, device=A_new.device)
        if b_pos is not None:
            if not reduce_bias:
                bias_pos = A_pos * b_pos
            else:
                if patches_mode:
                    bias_pos = torch.einsum('sb...chw,sb...chw->sb...', A_pos, b_pos)
                else:
                    bias_pos = torch.einsum('sb...,sb...->sb', A_pos, b_pos)
        if b_neg is not None:
            if not reduce_bias:
                bias_neg = A_neg * b_neg
            else:
                if patches_mode:
                    bias_neg = torch.einsum('sb...chw,sb...chw->sb...', A_neg, b_neg)
                else:
                    bias_neg = torch.einsum('sb...,sb...->sb', A_neg, b_neg)
        return A_new, bias_pos + bias_neg

    @staticmethod
    @torch.no_grad()
    @torch.jit.script
    def clamp_mutiply_backward(A: Tensor, d_pos: Tensor, d_neg: Tensor,
            b_pos: Optional[Tensor], b_neg: Optional[Tensor],
            grad_output_A: Tensor, grad_output_bias: Optional[Tensor], same_slope: bool = False
        ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor],
                   None, None, None]:
        """Improved backward operation. This should be better than the backward
        function generated by Pytorch."""
        if grad_output_bias is not None:
            extension_dim = len(A.shape) - len(grad_output_bias.shape)
            grad_output_bias = grad_output_bias.view(
                grad_output_bias.shape + (1, ) * extension_dim)
        A_pos_mask = (A >= 0).to(dtype=grad_output_A.dtype)
        A_neg_mask = 1. - A_pos_mask
        A_pos_grad_output_A = A_pos_mask * grad_output_A
        A_neg_grad_output_A = A_neg_mask * grad_output_A
        # Although d_pos is d_neg, we still need to get gd_pos and gd_neg separately.
        gd_pos = A * A_pos_grad_output_A
        gd_neg = A * A_neg_grad_output_A
        if b_pos is not None and b_neg is not None and grad_output_bias is not None:
            A_pos_grad_output_bias = A_pos_mask * grad_output_bias
            A_neg_grad_output_bias = A_neg_mask * grad_output_bias
            gb_neg = A * A_neg_grad_output_bias
            gb_pos = A * A_pos_grad_output_bias
            if same_slope:
                gA = (d_pos * grad_output_A
                      + b_pos * A_pos_grad_output_bias
                      + b_neg * A_neg_grad_output_bias)
            else:
                gA = (d_pos * A_pos_grad_output_A
                    + d_neg * A_neg_grad_output_A
                    + b_pos * A_pos_grad_output_bias
                    + b_neg * A_neg_grad_output_bias)
        elif b_neg is not None and grad_output_bias is not None:
            A_neg_grad_output_bias = A_neg_mask * grad_output_bias
            gb_neg = A * A_neg_grad_output_bias
            gb_pos = None
            if same_slope:
                gA = (d_pos * grad_output_A
                      + b_neg * A_neg_grad_output_bias)
            else:
                gA = (d_pos * A_pos_grad_output_A
                    + d_neg * A_neg_grad_output_A
                    + b_neg * A_neg_grad_output_bias)
        elif b_pos is not None and grad_output_bias is not None:
            A_pos_grad_output_bias = A_pos_mask * grad_output_bias
            gb_pos = A * A_pos_grad_output_bias
            gb_neg = None
            if same_slope:
                gA = (d_pos * grad_output_A
                      + b_pos * A_pos_grad_output_bias)
            else:
                gA = (d_pos * A_pos_grad_output_A + d_neg * A_neg_grad_output_A
                    + b_pos * A_pos_grad_output_bias)
        else:
            if same_slope:
                gA = d_pos * grad_output_A
            else:
                gA = d_pos * A_pos_grad_output_A + d_neg * A_neg_grad_output_A
            gb_pos = gb_neg = None
        return gA, gd_pos, gd_neg, gb_pos, gb_neg, None, None, None

    @staticmethod
    def forward(ctx, A, d_pos, d_neg, b_pos, b_neg, patches_mode, reduce_bias=True, same_slope=False):
        # No need to save the intermediate A_pos, A_neg as they have been fused into the computation.
        ctx.save_for_backward(A, d_pos, d_neg, b_pos, b_neg)
        ctx.patches_mode = patches_mode
        ctx.reduce_bias = reduce_bias
        ctx.same_slope = same_slope
        return ClampedMultiplication.clamp_mutiply_forward(
            A, d_pos, d_neg, b_pos, b_neg, patches_mode, reduce_bias, same_slope)

    @staticmethod
    def backward(ctx, grad_output_A, grad_output_bias):
        A, d_pos, d_neg, b_pos, b_neg = ctx.saved_tensors
        assert ctx.reduce_bias
        return ClampedMultiplication.clamp_mutiply_backward(
            A, d_pos, d_neg, b_pos, b_neg,
            grad_output_A, grad_output_bias, ctx.same_slope)


def multiply_by_A_signs(A, d_pos, d_neg, b_pos, b_neg, contiguous='auto',
                        reduce_bias=True, same_slope=False):
    if isinstance(A, Tensor):
        if contiguous is True or contiguous == 'auto':
            # For dense mode, convert d_pos and d_neg to contiguous tensor by default.
            d_pos = d_pos.contiguous()
            d_neg = d_neg.contiguous()
        if d_pos.ndim == 1:
            # Special case for LSTM, the bias term is 1-dimension. (FIXME)
            assert d_neg.ndim == 1 and b_pos.ndim == 1 and b_neg.ndim == 1
            new_A = A.clamp(min=0) * d_pos + A.clamp(max=0) * d_neg
            new_bias = A.clamp(min=0) * b_pos + A.clamp(max=0) * b_neg
            return new_A, new_bias
        return ClampedMultiplication.apply(
            A, d_pos, d_neg, b_pos, b_neg, False, reduce_bias, same_slope)
    elif isinstance(A, Patches):
        if contiguous:
            # For patches mode, do not convert d_pos and d_neg to contiguous tensor by default.
            d_pos = d_pos.contiguous()
            d_neg = d_neg.contiguous()
        assert A.identity == 0  # TODO: handle the A.identity = 1 case. Currently not used.
        patches = A.patches
        patches_shape = patches.shape
        # patches shape: [out_c, batch_size, out_h, out_w, in_c, H, W]. Here out_c is the spec dimension.
        # or (unstable_size, batch_size, in_c, H, W) when it is sparse.
        if len(patches_shape) == 6:
            patches = patches.view(*patches_shape[:2], -1, *patches_shape[-2:])
            d_pos = d_pos.view(*patches_shape[:2], -1, *patches_shape[-2:]) if d_pos is not None else None
            d_neg = d_neg.view(*patches_shape[:2], -1, *patches_shape[-2:]) if d_neg is not None else None
            b_pos = b_pos.view(*patches_shape[:2], -1, *patches_shape[-2:]) if b_pos is not None else None
            b_neg = b_neg.view(*patches_shape[:2], -1, *patches_shape[-2:]) if b_neg is not None else None
        # Apply the multiplication based on signs.
        A_prod, bias = ClampedMultiplication.apply(
            patches, d_pos, d_neg, b_pos, b_neg, True, reduce_bias, same_slope)
        # prod has shape [out_c, batch_size, out_h, out_w, in_c, H, W] or (unstable_size, batch_size, in_c, H, W) when it is sparse.
        # For sparse patches the return bias size is (unstable_size, batch).
        # For regular patches the return bias size is (spec, batch, out_h, out_w).
        if len(patches_shape) == 6:
            A_prod = A_prod.view(*patches_shape)
        return A.create_similar(A_prod), bias

