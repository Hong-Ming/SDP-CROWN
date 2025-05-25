"""Unused code below.
Note that these functions do not work now.
Add missing variables (self, start_nodes, device, etc.) if you still want to use them.
"""

# Generalized Beta CROWN with new multiple neuron split constraints
def _multi_neuron_splits_split_beta(A, uA, lA, unstable_idx, beta_for_intermediate_layers):
    split_convert_time = time.time()
    if self.split_coeffs["dense"] is None:
        assert not hasattr(self, 'split_intermediate_betas')  # intermediate beta split must use the dense mode.
        ##### we can use repeat to further save the conversion time
        # since the new split constraint coeffs can be optimized, we can just save the index and assign optimized coeffs value to the sparse matrix
        self.new_split_coeffs = torch.zeros(self.split_c.size(0), self.flattened_nodes,
                                            dtype=torch.get_default_dtype(), device=device)
        # assign coeffs value to the first half batch
        self.new_split_coeffs[
            (self.split_coeffs["nonzero"][:, 0], self.split_coeffs["nonzero"][:, 1])] = \
            self.split_coeffs["coeffs"]
        # # assign coeffs value to the rest half batch with the same values since split constraint shared the same coeffs for >0/<0
        self.new_split_coeffs[(self.split_coeffs["nonzero"][:, 0] + int(self.split_c.size(0) / 2),
                                self.split_coeffs["nonzero"][:, 1])] = self.split_coeffs["coeffs"]
    else:
        # batch = int(self.split_c.size(0)/2)
        # assign coeffs value to the first half batch and the second half batch
        self.new_split_coeffs = self.split_coeffs["dense"].repeat(2, 1)
    split_convert_time = time.time() - split_convert_time
    split_compute_time = time.time()
    if beta_for_intermediate_layers:
        assert hasattr(self, 'split_intermediate_betas')
        # print(f'split intermediate beta for {start_node.name} with beta shape {self.split_intermediate_betas[start_node.name]["ub"].size()}')
        if uA is not None:
            # upper bound betas for this set of intermediate neurons.
            # Make an extra spec dimension. Now new_split_coeffs has size (batch, specs, #nodes). Specs is the number of intermediate neurons of start node. The same split will be applied to all specs in a batch element.
            # masked_beta_upper has shape (batch, spec, #nodes)
            split_intermediate_betas = self.split_intermediate_betas[start_node.name]['ub']
            split_intermediate_betas = split_intermediate_betas.view(split_intermediate_betas.size(0), -1, split_intermediate_betas.size(-1))
            if unstable_idx is not None:
                # Only unstable neurons of the start_node neurons are used.
                split_intermediate_betas = self.non_deter_index_select(split_intermediate_betas, index=unstable_idx, dim=1)
            self.split_masked_beta_upper = split_intermediate_betas * (
                    self.new_split_coeffs * self.split_c).unsqueeze(1)
        if lA is not None:
            split_intermediate_betas = self.split_intermediate_betas[start_node.name]['lb']
            split_intermediate_betas = split_intermediate_betas.view(split_intermediate_betas.size(0), -1, split_intermediate_betas.size(-1))
            if unstable_idx is not None:
                # Only unstable neurons of the start_node neurons are used.
                split_intermediate_betas = self.non_deter_index_select(split_intermediate_betas, index=unstable_idx, dim=1)
            self.split_masked_beta_lower = split_intermediate_betas * (
                    self.new_split_coeffs * self.split_c).unsqueeze(1)
    else:
        # beta for final objective only. TODO: distinguish between lb and ub.
        self.split_masked_beta_upper = self.split_masked_beta_lower = self.new_split_coeffs * (
                self.split_beta * self.split_c)
    # add the new split constraint beta to the masked_beta
    if self.masked_beta_upper is None:
        self.masked_beta_upper = self.split_masked_beta_upper
    else:
        self.masked_beta_upper = self.masked_beta_upper + self.split_masked_beta_upper

    if self.masked_beta_lower is None:
        self.masked_beta_lower = self.split_masked_beta_lower
    else:
        self.masked_beta_lower = self.masked_beta_lower + self.split_masked_beta_lower
    # For backwards compatibility - we originally only have one beta.
    self.masked_beta = self.masked_beta_lower
    split_compute_time = time.time() - split_compute_time

    return lA, uA


# Beta CROWN with multiple neuron split or cuts
def _beta_crown_multi_neuron_splits(x, A, uA, lA, unstable_idx, start_node=None):
    print("beta crown multi neuron splits function should not be triggered in the current version!")
    exit()
    # A = uA if uA is not None else lA
    if type(A) == Tensor:
        device = A.device
    else:
        device = A.patches.device
    print_time = False

    # There are three types of beta used
    # single beta: with constraint only has single relu neuron
    # split beta: with constraint have multiple relu neurons
    # history beta: history constraints for multiple neuron splits
    if self.single_beta_used or self.split_beta_used or self.history_beta_used:
        start_time = time.time()
        history_compute_time, split_compute_time, split_convert_time = 0, 0, 0
        history_compute_time1, history_compute_time2 = 0, 0
        # assert len(self.split_beta) > 0, "split_beta_used or history_beta_used is True means there have to be one relu in one batch is used in split constraints"
        if self.single_beta_used:
            lA, uA = _multi_neuron_splits_single_beta(A, uA, lA, unstable_idx, beta_for_intermediate_layers)

        ############################
        # sparse_coo version for history coeffs
        if self.history_beta_used:
            lA, uA = _multi_neuron_splits_history_beta(A, uA, lA, unstable_idx, beta_for_intermediate_layers)

        # new split constraint
        if self.split_beta_used:
            lA, uA = _multi_neuron_splits_split_beta(A, uA, lA, unstable_idx, beta_for_intermediate_layers)

        A = last_uA if last_uA is not None else last_lA
        if type(A) is Patches:
            assert not hasattr(self, 'split_intermediate_betas')
            assert not hasattr(self, 'single_intermediate_betas')
            A_patches = A.patches
            # Reshape beta to image size.
            self.masked_beta = self.masked_beta.view(self.masked_beta.size(0), *ub_r.size()[1:])
            # unfold the beta as patches, size (batch, out_h, out_w, in_c, H, W)
            masked_beta_unfolded = inplace_unfold(self.masked_beta, kernel_size=A_patches.shape[-2:], padding=A.padding, stride=A.stride, inserted_zeros=A.inserted_zeros, output_padding=A.output_padding)
            if A.unstable_idx is not None:
                masked_beta_unfolded = masked_beta_unfolded.permute(1, 2, 0, 3, 4)
                # After selection, the shape is (unstable_size, batch, in_c, H, W).
                masked_beta_unfolded = masked_beta_unfolded[A.unstable_idx[1], A.unstable_idx[2]]
            else:
                # Add the spec (out_c) dimension.
                masked_beta_unfolded = masked_beta_unfolded.unsqueeze(0)
            if uA is not None:
                uA = uA.create_similar(uA.patches + masked_beta_unfolded)
            if lA is not None:
                lA = lA.create_similar(lA.patches - masked_beta_unfolded)
        elif type(A) is Tensor:
            if uA is not None:
                # print("uA", uA.shape, self.masked_beta.shape)
                # uA/lA has shape (spec, batch, *nodes)
                if beta_for_intermediate_layers:
                    if not self.single_beta_used:
                        # masked_beta_upper has shape (batch, spec, #nodes)
                        self.masked_beta_upper = self.masked_beta_upper.transpose(0, 1)
                        self.masked_beta_upper = self.masked_beta_upper.view(self.masked_beta_upper.size(0),
                                                                                self.masked_beta_upper.size(1),
                                                                                *uA.shape[2:])
                else:
                    # masked_beta_upper has shape (batch, #nodes)
                    self.masked_beta_upper = self.masked_beta_upper.reshape(uA[0].shape).unsqueeze(0)
                if not self.single_beta_used or not beta_for_intermediate_layers:
                    # For intermediate layer betas witn single node split, uA has been modified above.
                    uA = uA + self.masked_beta_upper
            if lA is not None:
                # print("lA", lA.shape, self.masked_beta.shape)
                if beta_for_intermediate_layers:
                    if not self.single_beta_used:
                        # masked_beta_upper has shape (batch, spec, #nodes)
                        self.masked_beta_lower = self.masked_beta_lower.transpose(0, 1)
                        self.masked_beta_lower = self.masked_beta_lower.view(self.masked_beta_lower.size(0),
                                                                                self.masked_beta_lower.size(1),
                                                                                *lA.shape[2:])
                else:
                    # masked_beta_upper has shape (batch, #nodes)
                    self.masked_beta_lower = self.masked_beta_lower.reshape(lA[0].shape).unsqueeze(0)
                if not self.single_beta_used or not beta_for_intermediate_layers:
                    # For intermediate layer betas witn single node split, lA has been modified above.
                    lA = lA - self.masked_beta_lower
        else:
            raise RuntimeError(f"Unknown type {type(A)} for A")
        # print("total:", time.time()-start_time, history_compute_time1, history_compute_time2, split_convert_time, split_compute_time)

    return lA, uA


# Generalized Beta CROWN with single neuron split constraint
def _multi_neuron_splits_single_beta(A, uA, lA, unstable_idx, beta_for_intermediate_layers):
    if beta_for_intermediate_layers:
        # We handle the refinement of intermediate layer after this split layer here. (the refinement for intermediate layers before the split is handled in compute_bounds().
        # print(f'single node beta for {start_node.name} with beta shape {self.single_intermediate_betas[start_node.name]["ub"].size()}')
        assert not self.history_beta_used
        assert type(A) is not Patches
        if uA is not None:
            # The beta for start_node has shape ([batch, prod(start_node.shape), n_max_history_beta])
            single_intermediate_beta = self.single_intermediate_betas[start_node.name]['ub']
            single_intermediate_beta = single_intermediate_beta.view(
                single_intermediate_beta.size(0), -1, single_intermediate_beta.size(-1))
            if unstable_idx is not None:
                # Only unstable neurons of the start_node neurons are used.
                single_intermediate_beta = self.non_deter_index_select(single_intermediate_beta, index=unstable_idx, dim=1)
            # This is the sign.
            single_intermediate_beta = single_intermediate_beta * self.single_beta_sign.unsqueeze(1)
            # We now generate a large matrix in shape (batch, prod(start_node.shape), prod(nodes)) which is the same size as uA and lA.
            prev_size = uA.size()
            # self.single_beta_loc has shape [batch, max_single_split]. Need to expand at the specs dimension.
            indices = self.single_beta_loc.unsqueeze(0).expand(uA.size(0), -1, -1)
            # We update uA here directly using sparse operation. Note the spec dimension is at the first!
            if self.alpha_beta_update_mask is not None:
                indices = indices[:, self.alpha_beta_update_mask]
                single_intermediate_beta = single_intermediate_beta[:, self.alpha_beta_update_mask]
            uA = self.non_deter_scatter_add(uA.view(uA.size(0), uA.size(1), -1), dim=2, index=indices, src=single_intermediate_beta.transpose(0,1))
            uA = uA.view(prev_size)
        if lA is not None:
            # The beta for start_node has shape ([batch, prod(start_node.shape), n_max_history_beta])
            single_intermediate_beta = self.single_intermediate_betas[start_node.name]['lb']
            single_intermediate_beta = single_intermediate_beta.view(
                single_intermediate_beta.size(0), -1, single_intermediate_beta.size(-1))
            if unstable_idx is not None:
                # Only unstable neurons of the start_node neurons are used.
                single_intermediate_beta = self.non_deter_index_select(single_intermediate_beta, index=unstable_idx, dim=1)
            # This is the sign, for lower bound we need to negate.
            single_intermediate_beta = single_intermediate_beta * ( - self.single_beta_sign.unsqueeze(1))
            # We now generate a large matrix in shape (batch, prod(start_node.shape), prod(nodes)) which is the same size as uA and lA.
            prev_size = lA.size()
            # self.single_beta_loc has shape [batch, max_single_split]. Need to expand at the specs dimension.
            indices = self.single_beta_loc.unsqueeze(0).expand(lA.size(0), -1, -1)
            # We update lA here directly using sparse operation. Note the spec dimension is at the first!
            if self.alpha_beta_update_mask is not None:
                indices = indices[:, self.alpha_beta_update_mask]
                single_intermediate_beta = single_intermediate_beta[:, self.alpha_beta_update_mask]
            lA = self.non_deter_scatter_add(lA.view(lA.size(0), lA.size(1), -1), dim=2, index=indices, src=single_intermediate_beta.transpose(0,1))
            lA = lA.view(prev_size)
    else:
        self.masked_beta_lower = self.masked_beta_upper = self.masked_beta = self.beta * self.beta_mask
    return lA, uA


# Generalized Beta CROWN with history multiple neuron split constraints
def _multi_neuron_splits_history_beta(A, uA, lA, unstable_idx, beta_for_intermediate_layers):
    # history_compute_time = time.time()
    if beta_for_intermediate_layers:
        # print(f'history intermediate beta for {start_node.name} with beta shape {self.history_intermediate_betas[start_node.name]["ub"].size()}')
        if uA is not None:
            # The beta for start_node has shape ([batch, prod(start_node.shape), n_max_history_beta])
            history_intermediate_beta = self.history_intermediate_betas[start_node.name]['ub']
            history_intermediate_beta = history_intermediate_beta.view(
                history_intermediate_beta.size(0), -1, history_intermediate_beta.size(-1))
            if unstable_idx is not None:
                # Only unstable neurons of the start_node neurons are used.
                history_intermediate_beta = self.non_deter_index_select(history_intermediate_beta, index=unstable_idx, dim=1)
            # new_history_coeffs has shape (batch, prod(nodes), n_max_history_beta)
            # new_history_c has shape (batch, n_max_history_beta)
            # This can generate a quite large matrix in shape (batch, prod(start_node.shape), prod(nodes)) which is the same size as uA and lA.
            self.masked_beta_upper = torch.bmm(history_intermediate_beta, (
                    self.new_history_coeffs * self.new_history_c.unsqueeze(1)).transpose(-1,
                                                                                            -2))
        if lA is not None:
            history_intermediate_beta = self.history_intermediate_betas[start_node.name]['lb']
            history_intermediate_beta = history_intermediate_beta.view(
                history_intermediate_beta.size(0), -1, history_intermediate_beta.size(-1))
            if unstable_idx is not None:
                # Only unstable neurons of the start_node neurons are used.
                history_intermediate_beta = self.non_deter_index_select(history_intermediate_beta, index=unstable_idx, dim=1)
            self.masked_beta_lower = torch.bmm(history_intermediate_beta, (
                    self.new_history_coeffs * self.new_history_c.unsqueeze(1)).transpose(-1,
                                                                                            -2))
    else:
        # new_history_coeffs has shape (batch, prod(nodes), n_max_history_beta)
        # new_history_beta has shape (batch, m_max_history_beta)
        self.masked_beta_lower = self.masked_beta_upper = torch.bmm(self.new_history_coeffs, (
                self.new_history_beta * self.new_history_c).unsqueeze(-1)).squeeze(-1)
    return lA, uA
