"""A copy of related functions from batch_branch_and_bound.py before
cleaning intermediate_refinement that is not actually used anywhere."""


def _get_intermediate_beta_bounds(self, l, beta_watch_list, unstable_idx):
    # Contribution to the constant term.
    intermediate_beta_lb = intermediate_beta_ub = 0.0
    # Contribution to the linear coefficients.
    intermediate_beta_lA = intermediate_beta_uA = 0.0
    for split_layer_name, split_coeffs in beta_watch_list[l.name].items():
        # We may have multiple splits after this layer, and they are enumerated here.
        # Find the corresponding beta variable, which is in a later layer.
        # split_layer_name should be the pre-act layer name, and Relu is after it.
        split_layer = self._modules[self._modules[split_layer_name].output_name[0]]
        # Concat betas from history split and current split.
        all_betas_lb = all_betas_ub = None
        if split_layer.history_beta_used:
            # The beta has size [batch, *node_shape, n_beta] (i.e., each element, each neuron, has a different beta)
            all_betas_lb = split_layer.history_intermediate_betas[l.name]["lb"]
            all_betas_lb = all_betas_lb.view(all_betas_lb.size(0), -1, all_betas_lb.size(-1))
            all_betas_ub = split_layer.history_intermediate_betas[l.name]["ub"]
            all_betas_ub = all_betas_ub.view(all_betas_ub.size(0), -1, all_betas_ub.size(-1))
            if unstable_idx is not None:
                # Only unstable neuron is considered.
                all_betas_lb = self.non_deter_index_select(all_betas_lb, index=unstable_idx, dim=1)
                all_betas_ub = self.non_deter_index_select(all_betas_ub, index=unstable_idx, dim=1)
        if split_layer.split_beta_used:
            # Note: we must keep split_intermediate_betas at the last because this is the way we build split_coeffs.
            split_intermediate_betas_lb = split_layer.split_intermediate_betas[l.name]["lb"]
            split_intermediate_betas_ub = split_layer.split_intermediate_betas[l.name]["ub"]
            split_intermediate_betas_lb = split_intermediate_betas_lb.view(split_intermediate_betas_lb.size(0), -1, split_intermediate_betas_lb.size(-1))
            split_intermediate_betas_ub = split_intermediate_betas_ub.view(split_intermediate_betas_ub.size(0), -1, split_intermediate_betas_ub.size(-1))
            if unstable_idx is not None:
                split_intermediate_betas_lb = self.non_deter_index_select(split_intermediate_betas_lb, index=unstable_idx, dim=1)
                split_intermediate_betas_ub = self.non_deter_index_select(split_intermediate_betas_ub, index=unstable_idx, dim=1)
            all_betas_lb = torch.cat((all_betas_lb, split_intermediate_betas_lb), dim=-1) if all_betas_lb is not None else split_intermediate_betas_lb
            all_betas_ub = torch.cat((all_betas_ub, split_intermediate_betas_ub), dim=-1) if all_betas_ub is not None else split_intermediate_betas_ub
        if split_layer.single_beta_used:
            single_intermediate_betas_lb = split_layer.single_intermediate_betas[l.name]["lb"]
            single_intermediate_betas_ub = split_layer.single_intermediate_betas[l.name]["ub"]
            single_intermediate_betas_lb = single_intermediate_betas_lb.view(single_intermediate_betas_lb.size(0), -1, single_intermediate_betas_lb.size(-1))
            single_intermediate_betas_ub = single_intermediate_betas_ub.view(single_intermediate_betas_ub.size(0), -1, single_intermediate_betas_ub.size(-1))
            if unstable_idx is not None:
                single_intermediate_betas_lb = self.non_deter_index_select(single_intermediate_betas_lb, index=unstable_idx, dim=1)
                single_intermediate_betas_ub = self.non_deter_index_select(single_intermediate_betas_ub, index=unstable_idx, dim=1)
            all_betas_lb = torch.cat((all_betas_lb, single_intermediate_betas_lb), dim=-1) if all_betas_lb is not None else single_intermediate_betas_lb
            all_betas_ub = torch.cat((all_betas_ub, single_intermediate_betas_ub), dim=-1) if all_betas_ub is not None else single_intermediate_betas_ub
        # beta has been reshaped to to [batch, prod(node_shape), n_beta] , prod(node_shape) is the spec dimension for A.
        # print(f'Add beta to {l.name} with shape {all_betas_lb.size()}, A shape {A[0][0].size()}, split_coeffs A {split_coeffs["lA"].size()}, split_coeffs bias {split_coeffs["lbias"].size()}')
        # Constant terms from beta related Lagrangian. split_coeffs['lbias'] is in shape [batch, n_beta].
        # We got shape [batch, *node_shape], which corresponds to the bias terms caused by beta per batch element per intermediate neuron.
        intermediate_beta_lb = intermediate_beta_lb + torch.einsum('ijb,ib->ij', all_betas_lb,
                                                                   split_coeffs['lbias'])
        intermediate_beta_ub = intermediate_beta_ub + torch.einsum('ijb,ib->ij', all_betas_ub,
                                                                   split_coeffs['ubias'])
        # A coefficients from beta related Lagrangian. split_coeffs['lA'] is in shape [batch, n_beta, *preact_shape].
        # We got shape [batch, prod(node_shape), *preact_shape].
        # print(f'BEFORE {node.name} split layer {split_layer.name} l {l.name} {torch.tensor(intermediate_beta_lA).abs().sum()} {torch.tensor(intermediate_beta_uA).abs().sum()} \t {all_betas_lb.abs().sum()} {all_betas_lb.size()} {split_coeffs["lA"].abs().sum()} {split_coeffs["lA"].size()}')
        intermediate_beta_lA = intermediate_beta_lA + torch.einsum('ijb,ib...->ij...', all_betas_lb,
                                                                   split_coeffs['lA'])
        intermediate_beta_uA = intermediate_beta_uA + torch.einsum('ijb,ib...->ij...', all_betas_ub,
                                                                   split_coeffs['uA'])
        # print(f'AFTER  {node.name} split layer {split_layer.name} l {l.name} {torch.tensor(intermediate_beta_lA).abs().sum()} {torch.tensor(intermediate_beta_uA).abs().sum()} \t {all_betas_ub.abs().sum()} {all_betas_lb.size()} {split_coeffs["uA"].abs().sum()} {split_coeffs["uA"].size()}')
        return intermediate_beta_lA, intermediate_beta_uA, intermediate_beta_lb, intermediate_beta_ub



def split_domain(
        net, domains, d, batch,
        fix_intermediate_layer_bounds, stop_func, multi_spec_keep_func, timer):
    global Visited

    solver_args = arguments.Config['solver']
    bab_args = arguments.Config['bab']
    opt_intermediate_beta = False
    ### preprocessor-hint: private-section-start
    opt_intermediate_beta = solver_args['intermediate_refinement']['enabled']
    opt_coeffs = solver_args['intermediate_refinement']['opt_coeffs']
    opt_bias = solver_args['intermediate_refinement']['opt_bias']
    ### preprocessor-hint: private-section-end
    branching_method = bab_args['branching']['method']
    branching_reduceop = bab_args['branching']['reduceop']
    branching_candidates = bab_args['branching']['candidates']

    min_batch_size = min(
        solver_args['min_batch_size_ratio'] * solver_args['batch_size'],
        batch)
    split_depth = get_split_depth(d['lower_bounds'][0].shape[0], min_batch_size)
    print("batch: ", d['lower_bounds'][0].shape, "pre split depth: ", split_depth)

    timer.start('decision')
    branching_obj, branching_func = get_branching_heuristic(net)
    if branching_obj is not None:
        branching_decision, branching_points, split_depth = (
            branching_obj.get_branching_decisions(d, split_depth))
    else:
        # Increase the maximum number of candidates for fsb and kfsb if there are more splits needed.
        branching_candidates = max(branching_candidates, split_depth)
        branching_decision, split_depth = branching_func(
            d, net, net.split_indices,
            branching_candidates=branching_candidates,
            branching_reduceop=branching_reduceop,
            split_depth=split_depth, method=branching_method)
        branching_points = None

    print("post split depth: ", split_depth)

    if len(branching_decision) < len(d['mask'][0]):
        print('all nodes are split!!')
        print(f'{Visited} domains visited')
        global all_node_split
        all_node_split = True
        if not solver_args['beta-crown']['all_node_split_LP']:
            global_lb = d['global_lb'][0] - d['thresholds'][0]
            for i in range(1, len(d['global_lb'])):
                if max(d['global_lb'][i] - d['thresholds'][i]) <= max(global_lb):
                    global_lb = d['global_lb'][i] - d['thresholds'][i]
            return global_lb, np.inf

    split = {}
    # split['decision']: selected domains (next batch/2)->node list->node: [layer, idx]
    split['decision'] = branching_decision
    split['points'] = branching_points

    print_splitting_decisions(d, split_depth, split)
    timer.add('decision')
    timer.start('solve')

    single_node_split = True
    ### preprocessor-hint: private-section-start
    single_node_split = not opt_coeffs and not opt_bias and not opt_intermediate_beta
    ### preprocessor-hint: private-section-end
    # copy the original lbs

    num_copy = (2**(split_depth-1))
    copy_domains(d, split, split_depth, net.split_indices)

    # Caution: we use "all" predicate to keep the domain when multiple specs are present: all lbs should be <= threshold, otherwise pruned
    # maybe other "keeping" criterion needs to be passed here
    ret = net.get_lower_bound(
        d, split,
        fix_intermediate_layer_bounds=fix_intermediate_layer_bounds,
        single_node_split=single_node_split,
        stop_func=stop_func(torch.cat([d['thresholds'], d['thresholds']])),
        multi_spec_keep_func=multi_spec_keep_func)
    timer.add('solve')
    timer.start('add')
    batch = len(split['decision'])
    depths = [d + split_depth - 1 for d in d['depths']] * num_copy * 2

    old_d_len = len(domains)
    if solver_args['beta-crown']['all_node_split_LP']:
        ret_lp = batch_verification_all_node_split_LP(
            net, depths, ret['lower_bounds'][-1], d['thresholds'],
            ret['lower_bounds'], ret['upper_bounds'])
        if ret_lp is not None:
            # lp_status == "unsafe"
            # unsafe cases still needed to be handled! set to be unknown for now!
            all_node_split = True
            return ret_lp, np.inf

    # If intermediate layers are not refined or updated, we do not need to check
    # infeasibility when adding new domains.
    check_infeasibility = not (single_node_split and fix_intermediate_layer_bounds)
    domains.add(ret, d['history'], depths, split, d['thresholds'],
                check_infeasibility)

    # FIXME ???
    Visited += (len(d['depths']) * num_copy) * 2 - (len(domains) - old_d_len)
    domains.print()
    timer.add('add')

    return ret


def act_bab(
        net, domain, x, refined_lower_bounds=None,
        refined_upper_bounds=None, activation_opt_params=None,
        reference_slopes=None, reference_lA=None, attack_images=None,
        timeout=None, refined_betas=None, rhs=0):
    # the crown_lower/upper_bounds are present for initializing the unstable indx when constructing bounded module
    # it is ok to not pass them here, but then we need to go through a CROWN process again which is slightly slower
    start = time.time()
    # All supported arguments.
    global Visited, all_node_split

    solver_args = arguments.Config['solver']
    bab_args = arguments.Config['bab']
    branch_args = bab_args['branching']
    timeout = timeout or bab_args['timeout']
    max_domains = bab_args['max_domains']
    batch = solver_args['batch_size']
    opt_intermediate_beta = False
    ### preprocessor-hint: private-section-start
    opt_intermediate_beta = solver_args['intermediate_refinement']['enabled']
    intermediate_refinement_layers = solver_args['intermediate_refinement']['layers']
    ### preprocessor-hint: private-section-end
    get_upper_bound = bab_args['get_upper_bound']
    use_bab_attack = bab_args['attack']['enabled']
    cut_enabled = bab_args['cut']['enabled']
    lp_cut_enabled = bab_args['cut']['lp_cut']
    branching_input_and_activation = branch_args['branching_input_and_activation']

    if branching_input_and_activation and not bab_args['interm_transfer']:
        raise ValueError("Branching input and activation must be used when interm_transfer is True")

    if not isinstance(rhs, torch.Tensor):
        rhs = torch.tensor(rhs)

    stop_criterion = stop_criterion_batch_any(rhs)

    Visited = 0
    betas = None
    if solver_args['alpha-crown']['no_joint_opt']:
        global_ub, global_lb, updated_mask, lA, lower_bounds, upper_bounds, slope, history, betas = net.build_the_model_with_refined_bounds(
            domain, x, stop_criterion_func=stop_criterion, cutter=net.cutter, decision_thresh=rhs)
    elif refined_lower_bounds is None or refined_upper_bounds is None:
        assert arguments.Config['general']['enable_incomplete_verification'] is False
        global_lb, ret = net.build_the_model(
            domain, x, stop_criterion_func=stop_criterion, decision_thresh=rhs)
        updated_mask, lA, lower_bounds, upper_bounds, slope, history, input_split_idx = (
            ret['mask'], ret['lA'], ret['lower_bounds'], ret['upper_bounds'],
            ret['slope'], ret['history'], ret['input_split_idx'])
        global_ub = global_lb + np.inf
    else:
        global_ub, global_lb, updated_mask, lA, lower_bounds, upper_bounds, slope, history, betas, input_split_idx = net.build_the_model_with_refined_bounds(
            domain, x, refined_lower_bounds, refined_upper_bounds, activation_opt_params, reference_lA=reference_lA,
            reference_slopes=reference_slopes, stop_criterion_func=stop_criterion,
            cutter=net.cutter, refined_betas=refined_betas, decision_thresh=rhs)
        # release some storage to save memory
        if activation_opt_params is not None: del activation_opt_params
        torch.cuda.empty_cache()

    if solver_args['beta-crown']['all_node_split_LP']:
        timeout = bab_args['timeout']
        net.build_solver_model(timeout, model_type="lp")
    if use_bab_attack:
        # Beam search based BaB enabled. We need to construct the MIP model.
        print('Building MIP for beam search...')
        net.build_solver_model(
            timeout=bab_args['attack']['mip_timeout'],
            mip_multi_proc=solver_args['mip']['parallel_solvers'],
            mip_threads=solver_args['mip']['solver_threads'], model_type="mip")

    all_label_global_lb = torch.min(global_lb - rhs).item()
    all_label_global_ub = torch.max(global_ub - rhs).item()

    if arguments.Config['debug']['lp_test'] in ['LP", "MIP']:
        return all_label_global_lb, all_label_global_ub, 0, 'unknown'

    if stop_criterion(global_lb).all():
        return all_label_global_lb, all_label_global_ub, 0, 'safe'

    if not opt_intermediate_beta:
        # If we are not optimizing intermediate layer bounds, we do not need to
        # save all the intermediate alpha.
        # We only keep the alpha for the last layer.
        if not solver_args['beta-crown'].get('enable_opt_interm_bounds', False):
            kept_layer_names = [net.net.final_name]
            kept_layer_names.extend(
                filter(lambda x: len(x.strip()) > 0,
                       bab_args['optimized_intermediate_layers'].split(",")))
            print(f'Keeping slopes for these layers: {kept_layer_names}')
            # new_slope shape: [dict[relu_layer_name, {final_layer: torch.tensor storing alpha}] for each sample in batch]
            new_slope = prune_slopes(slope, kept_layer_names)
        else:
            new_slope = slope
    ### preprocessor-hint: private-section-start
    else:
        # FIXME hard-coded for ReLU only
        # We keep only these alphas for layers that will be optimized.
        # First get the layer names for refinement.
        refinement_layer_names = [net.net.relus[il].input_name[0] for il in intermediate_refinement_layers]
        # Output layer alpha is always included.
        refinement_layer_names.append(net.net.final_name)
        # new_slope shape: [dict[relu_layer_name, dict[dest_layer_name, torch.tensor storing alpha]] for each sample in batch]
        new_slope = prune_slopes(slope, refinement_layer_names)
    ### preprocessor-hint: private-section-end

    if use_bab_attack:
        DomainClass = SortedReLUDomainList
    else:
        DomainClass = BatchedReLUDomainList

    # This is the first (initial) domain.
    num_initial_domains = net.c.shape[0]
    domains = DomainClass(
        lA, global_lb, global_ub, lower_bounds, upper_bounds, new_slope,
        copy.deepcopy(history), [0] * num_initial_domains, net.c, # "[0] * num_initial_domains" corresponds to initial domain depth
        rhs, betas, num_initial_domains,
        interm_transfer=bab_args['interm_transfer'],
        x_Ls=x.ptb.x_L if branching_input_and_activation else None,
        x_Us=x.ptb.x_U if branching_input_and_activation else None,
        input_split_idx=input_split_idx if branching_input_and_activation else None,)

    if use_bab_attack:
        # BaB-attack code still uses a legacy sorted domain list.
        domains = to_sorted_list(domains)

    # tell the AutoLiRPA class not to transfer intermediate bounds to save time
    net.interm_transfer = bab_args['interm_transfer']

    # after domains are added, we replace global_lb, global_ub with the multile targets "real" global lb and ub to make them scalars
    global_lb, global_ub = all_label_global_lb, all_label_global_ub

    tot_ambi_nodes = 0
    # only pick the first copy from possible multiple x
    updated_mask = [mask[0:1] for mask in updated_mask]
    for i, layer_mask in enumerate(updated_mask):
        n_unstable = int(torch.sum(layer_mask).item())
        print(f'layer {i} name {net.net.layers_requiring_bounds[i]} size {layer_mask.shape[1:]} unstable {n_unstable}')
        tot_ambi_nodes += n_unstable
    print(f'-----------------\n# of unstable neurons: {tot_ambi_nodes}\n-----------------\n')
    net.tot_ambi_nodes = tot_ambi_nodes

    if cut_enabled:
        cut_verification(net, domains, lp_cut_enabled=lp_cut_enabled,
                         cplex_cuts=bab_args['cut']['cplex_cuts'],
                         cplex_cuts_wait=bab_args['cut']['cplex_cuts_wait'])
    if use_bab_attack:
        adv_pool = init_bab_attack(net, updated_mask, attack_images)
        global_ub = min(all_label_global_ub, adv_pool.adv_pool[0].obj)

    run_condition = len(domains) > 0
    split_round = 0
    sort_domain_iter = bab_args['sort_domain_interval']
    relu_split_iterations = branch_args['branching_relu_iterations']
    input_split_iterations = branch_args['branching_input_iterations']
    split_order = branch_args['branching_input_and_activation_order']
    reseting_round = relu_split_iterations + input_split_iterations
    split_condition = lambda r: r < input_split_iterations if split_order[0] == 'input' else r >= relu_split_iterations
    total_round = 0

    def _terminate(net, domains):
        del domains
        clean_net_mps_process(net)

    timer = Timer()
    while run_condition:
        total_round += 1
        global_lb = None
        ### preprocessor-hint: private-section-start
        if opt_intermediate_beta:
            if len(domains) > solver_args['intermediate_refinement']['max_domains']:
                # reach refinement max domains, disable refinement and set back to regular batch size
                solver_args['intermediate_refinement']['layers'] = []
                batch = solver_args['batch_size']
            else:
                batch = solver_args['intermediate_refinement']['batch_size']
        ### preprocessor-hint: private-section-end

        if use_bab_attack:
            global_lb, batch_ub, domains = bab_attack(
                domains, net, batch, net.split_indices, tot_ambi_nodes,
                adv_pool=adv_pool)
        else:
            if branching_input_and_activation and split_condition(split_round):
                print(f'Round {split_round}, using input split.')
                global_lb, batch_ub = input_split_on_relu_domains(
                    domains, wrapped_net=net, batch_size=batch)
            else:
                if branching_input_and_activation:
                    print(f'Round {split_round}, using activation split.')
                if bab_args['cut']['enabled'] and bab_args['cut']['cplex_cuts']:
                    fetch_cut_from_cplex(net)
                global_lb, batch_ub = act_split_round(
                    domains, net, batch,
                    fix_intermediate_layer_bounds=not opt_intermediate_beta,
                    timer=timer)
            if sort_domain_iter > 0 and total_round % sort_domain_iter == 0:
                domains.sort()

            split_round += 1
            if split_round >= reseting_round:
                split_round = 0

        if get_upper_bound:
            print(f"Global ub: {global_ub}, batch ub: {batch_ub}")
        global_ub = min(global_ub, batch_ub)
        run_condition = len(domains) > 0

        if isinstance(global_lb, torch.Tensor):
            global_lb = global_lb.max().item()
        if isinstance(global_ub, torch.Tensor):
            global_ub = global_ub.min().item()

        if all_node_split:
            all_node_split = False
            _terminate(net, domains)
            return global_lb, global_ub, Visited, 'unknown'

        if len(domains) > max_domains:
            print("Maximum number of visited domains has reached.")
            _terminate(net, domains)
            return global_lb, global_ub, Visited, 'unknown'

        if (get_upper_bound or use_bab_attack) and global_ub < rhs:
            print("Attack success during branch and bound.")
            # Terminate MIP if it has been started.
            if use_bab_attack and beam_mip_attack.started:
                print('Terminating MIP processes...')
                net.pool_termination_flag.value = 1
            _terminate(net, domains)
            return global_lb, global_ub, Visited, 'unsafe'

        if time.time() - start > timeout:
            print('Time out!!!!!!!!')
            if use_bab_attack and beam_mip_attack.started:
                print('Terminating MIP processes...')
                net.pool_termination_flag.value = 1
            _terminate(net, domains)
            return global_lb, global_ub, Visited, 'unknown'

        print(f'Cumulative time: {time.time() - start}\n')

    _terminate(net, domains)
    if use_bab_attack:
        # No domains left and no ub < 0 found.
        return global_lb, global_ub, Visited, 'unknown'
    else:
        # No domains left and not timed out.
        return global_lb, global_ub, Visited, 'safe'
