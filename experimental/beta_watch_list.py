beta_watch_list = get_beta_watch_list(self.intermediate_constr, all_nodes_before)


### preprocessor-hint: private-section-start
# Check if we have any beta split from a layer after the starting node.
if l.name in beta_watch_list:
    # Finished adding beta splits from all layers. Now merge them into the A matrix of this layer.
    # Our A has spec dimension at the front, so a transpose is needed.
    intermediate_beta_lA, intermediate_beta_uA, intermediate_beta_lb, intermediate_beta_ub = \
        self._get_intermediate_beta_bounds(l, beta_watch_list, unstable_idx)
    A[0] = (
        A[0][0] + intermediate_beta_lA.transpose(0, 1),
        A[0][1] + intermediate_beta_uA.transpose(0, 1))
    lb += intermediate_beta_lb.transpose(0, 1)
    ub += intermediate_beta_ub.transpose(0, 1)
    # Only need to add the first encountered. Set the watch list to empty.
    beta_watch_list = {}
### preprocessor-hint: private-section-end


def get_beta_watch_list(intermediate_constr, all_nodes_before):
    beta_watch_list = defaultdict(dict)
    if intermediate_constr is not None:
        # Intermediate layer betas are handled in two cases.
        # First, if the beta split is before this node, we don't need to do anything special;
        # it will done in BoundRelu.
        # Second, if the beta split after this node, we need to modify the A matrix
        # during bound propagation to reflect beta after this layer.
        for k in intermediate_constr:
            if k not in all_nodes_before:
                # The second case needs special care: we add all such splits in a watch list.
                # However, after first occurance of a layer in the watchlist,
                # beta_watch_list will be deleted and the A matrix from split constraints
                # has been added and will be propagated to later layers.
                for kk, vv in intermediate_constr[k].items():
                    beta_watch_list[kk][k] = vv
    return beta_watch_list