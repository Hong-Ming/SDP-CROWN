"""Concretization for PerturbationLpNorm with "extra_constr".

It does not seem to be used anywhere and has been cleaned in the main code.
"""

def concretize_matrix(self, x, A, sign, extra_constr):
    # If A is an identity matrix, we will handle specially.
    if not isinstance(A, eyeC):
        # A has (Batch, spec, *input_size). For intermediate neurons, spec is *neuron_size.
        A = A.reshape(A.shape[0], A.shape[1], -1)

        if extra_constr is not None:
            # For each neuron, we have a beta, so beta size is (Batch, *neuron_size, n_beta) (in A, spec is *neuron_size).
            # For intermediate layer neurons, A has *neuron_size specifications.
            beta = extra_constr['beta']
            beta = beta.view(beta.size(0), -1, beta.size(-1))
            # coeffs are linear relationships between split neurons and x. They have size (batch, n_beta, *input_size), and unreated to neuron_size.
            beta_coeffs = extra_constr['coeffs']
            beta_coeffs = beta_coeffs.view(beta_coeffs.size(0), beta_coeffs.size(1), -1)
            # biases are added for each batch each spec, size is (batch, n_beta), and unrelated to neuron_size.
            beta_bias = extra_constr['bias']
            # Merge beta into extra A and bias. Extra A has size (batch, spec, *input_size). For intermediate neurons, spec is *neuron_size.
            extra_A = torch.einsum('ijk,ikl->ijl', beta, beta_coeffs)
            # Merge beta into the bias term. Output has size (batch, spec).
            extra_bias = torch.einsum('ijk,ik->ij', beta, beta_bias)

    if self.norm == np.inf:
        # For Linfinity distortion, when an upper and lower bound is given, we use them instead of eps.
        x_L, x_U = self.get_input_bounds(x, A)
        x_ub = x_U.reshape(x_U.shape[0], -1, 1)
        x_lb = x_L.reshape(x_L.shape[0], -1, 1)
        # Find the uppwer and lower bound similarly to IBP.
        center = (x_ub + x_lb) / 2.0
        diff = (x_ub - x_lb) / 2.0
        if not isinstance(A, eyeC):
            if extra_constr is not None:
                # Extra linear and bias terms from constraints.
                print(
                    f'A extra: {(sign * extra_A).abs().sum().item()}, '
                    f'b extra: {(sign * extra_bias).abs().sum().item()}')
                A = A - sign * extra_A
                bound = A.matmul(center) - sign * extra_bias.unsqueeze(-1) + sign * A.abs().matmul(diff)
            else:
                bound = A.matmul(center) + sign * A.abs().matmul(diff)
        else:
            assert extra_constr is None
            # A is an identity matrix. No need to do this matmul.
            bound = center + sign * diff
    else:
        assert extra_constr is None
        x = x.reshape(x.shape[0], -1, 1)
        if not isinstance(A, eyeC):
            # Find the upper and lower bounds via dual norm.
            deviation = A.norm(self.dual_norm, -1) * self.eps
            bound = A.matmul(x) + sign * deviation.unsqueeze(-1)
        else:
            # A is an identity matrix. Its norm is all 1.
            bound = x + sign * self.eps
    bound = bound.squeeze(-1)
    return bound

def concretize_patches(self, x, A, sign, extra_constr):
    if self.norm == np.inf:
        x_L, x_U = self.get_input_bounds(x, A)

        # Here we should not reshape
        # Find the uppwer and lower bound similarly to IBP.
        center = (x_U + x_L) / 2.0
        diff = (x_U - x_L) / 2.0

        if not A.identity == 1:
            bound = A.matmul(center)
            bound_diff = A.matmul(diff, patch_abs=True)

            if sign == 1:
                bound += bound_diff
            elif sign == -1:
                bound -= bound_diff
            else:
                raise ValueError("Unsupported Sign")

            # The extra bias term from beta term.
            if extra_constr is not None:
                bound += extra_constr
        else:
            assert extra_constr is None
            # A is an identity matrix. No need to do this matmul.
            bound = center + sign * diff
        return bound
    else:  # Lp norm
        input_shape = x.shape
        if not A.identity:
            # Find the upper and lower bounds via dual norm.
            # matrix has shape (batch_size, out_c * out_h * out_w, input_c, input_h, input_w) or (batch_size, unstable_size, input_c, input_h, input_w)
            matrix = patches_to_matrix(A.patches, input_shape, A.stride, A.padding, A.output_shape, A.unstable_idx)
            # Note that we should avoid reshape the matrix. Due to padding, matrix cannot be reshaped without copying.
            deviation = matrix.norm(p=self.dual_norm, dim=(-3,-2,-1)) * self.eps
            # Bound has shape (batch, out_c * out_h * out_w) or (batch, unstable_size).
            bound = torch.einsum('bschw,bchw->bs', matrix, x) + sign * deviation
            if A.unstable_idx is None:
                # Reshape to (batch, out_c, out_h, out_w).
                bound = bound.view(matrix.size(0), A.patches.size(0), A.patches.size(2), A.patches.size(3))
        else:
            # A is an identity matrix. Its norm is all 1.
            bound = x + sign * self.eps
        return bound

def concretize(self, x, A, sign=-1, aux=None, extra_constr=None):
    """Given an variable x and its bound matrix A, compute worst case bound according to Lp norm."""
    if A is None:
        return None
    if isinstance(A, eyeC) or isinstance(A, torch.Tensor):
        return self.concretize_matrix(x, A, sign, extra_constr)
    elif isinstance(A, Patches):
        return self.concretize_patches(x, A, sign, extra_constr)
    else:
        raise NotImplementedError()