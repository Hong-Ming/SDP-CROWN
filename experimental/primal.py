"""Primal related code that is not used anywhere from beta_CROWN_solver.py"""

def get_neuron_primal(self, input_primal, lb, ub, slope_opt=None):
    # calculate the primal values for intermediate neurons
    # slope_opt is a list, each element has the dict for slopes of each batch

    if slope_opt is None:
        slope_opt = self.get_slope(self.net)

    batch_size = input_primal.shape[0]
    primal_values = [input_primal]
    # save the integer primal values in MIP constructions
    integer_primals = []
    primal = input_primal
    relu_idx = 0
    keys = list(slope_opt[0].keys())
    output_key = list(slope_opt[0][keys[0]].keys())[-1]
    # load saved primals from gurobi lp for debug
    # gurobi_primals = None
    # gurobi_primals = [np.load(f"gurobi_primals/{i}.npy") for i in range(10)]
    # gurobi_integer_primals = [np.load(f"gurobi_primals/z_relu{relu_idx}.npy") for relu_idx in range(5)]

    dual_values = torch.zeros((batch_size, 1), device=primal.device)

    for layer in self.layers:
        if not isinstance(layer, nn.ReLU):
            # just propagate the primal value if linear function or flatten layer
            primal = layer(primal)
        elif layer.perturbed:
            # only store input, pre_relu primal values, and output primals
            primal_values.append(primal.clone().detach())

            # handling nonlinear relus for primal propagations
            # we can use the lA from get_mask_lA_parallel but relu.lA is more straightforward
            # lA = lAs[0][relu_idx]
            lA = self.net.splittable_activations[relu_idx].lA.squeeze(0)

            # primal on lower boundary: lA<=0 & unstable
            u, l = ub[relu_idx].to(primal.device), lb[relu_idx].to(primal.device)
            unstable = (u > 0).logical_and(l < 0)

            # slope = slope_opt[which batch][keys[relu_idx]][output_key][0, 0]
            slope = self.net.splittable_activations[relu_idx].alpha[output_key][0, 0].to(primal.device)
            primal_l = primal * slope
            z_l =  primal / u
            z_l[z_l < 0] = 0

            # primal on upper boundary: lA>0 & unstable
            slope = (u / (u-l))
            bias = (-u * l / (u - l))
            primal_u = (primal * slope + bias).detach()
            z_u = (primal - l) / (u - l)
            # calculate z integer first, using last linear layer node primal values
            z = z_u
            z[(lA>0).logical_and(unstable)] = z_l[(lA>0).logical_and(unstable)]

            primal[(lA<=0).logical_and(unstable)] = primal_u[(lA<=0).logical_and(unstable)].detach()
            primal[(lA>0).logical_and(unstable)] = primal_l[(lA>0).logical_and(unstable)].detach()
            primal[(u<0)] = 0

            if self.net.splittable_activations[relu_idx].sparse_beta is not None and self.net.splittable_activations[relu_idx].sparse_beta.nelement() != 0:
                beta_loc = self.net.splittable_activations[relu_idx].sparse_beta_loc
                sparse_beta = self.net.splittable_activations[relu_idx].sparse_beta * self.net.splittable_activations[relu_idx].sparse_beta_sign

                # we select split neuron from primal with tuple index
                beta_loc_tuple = (torch.ones(beta_loc.shape).nonzero(as_tuple=True)[0], beta_loc.view(-1))
                # we get the pre relu primal values for each split node
                selected_primals = primal.view(batch_size, -1).gather(dim=1, index=beta_loc)
                # we will add beta * split node pre relu primal to the eventual primal output obj
                dual_values = dual_values + (sparse_beta * selected_primals).sum(1, keepdim=True).detach()
                # for split node, we need to force choice 1 to be pre relu primal and 0 for choice 0
                beta_c = (self.net.splittable_activations[relu_idx].sparse_beta_sign.view(-1) + 1) / 2
                primal.view(batch_size, -1)[beta_loc_tuple] = primal_values[-1].view(batch_size, -1)[beta_loc_tuple] * beta_c
                # force mixed integer z to be 1 and 0 for split nodes
                z.view(batch_size, -1)[beta_loc_tuple] = beta_c

            # store the primal values of mixed integers
            # if z[unstable].view(-1).shape[0] % batch_size !=0:
            #     import pdb; pdb.set_trace()
            ip = torch.ones(z.shape, device=z.device) * (-1.)
            ip[unstable] = z[unstable]
            integer_primals.append(ip.view(batch_size, -1))

            # We should not force primal to be larger than 0, otherwise not correct !!!
            # primal = layer(primal)
            relu_idx += 1

    primal_values.append(primal.clone().detach())
    primal_values[-1] = primal_values[-1] - dual_values

    integer_primals = [iv.to(device='cpu', non_blocking=True) for iv in integer_primals]
    primal_values = [pv.to(device='cpu', non_blocking=True) for pv in primal_values]

    return primal_values, integer_primals


def layer_wise_primals(self, primals):
    # originally layer -> batch,
    # now need to be a list with batch elements
    neuron_primals, integer_primals = primals["p"], primals["z"]
    ret_p = []
    for bi in range(neuron_primals[0].size(0)):
        pv, iv = [], []
        for layer_idx in range(len(neuron_primals)):
            pv.append(neuron_primals[layer_idx][bi:bi + 1])
        for relu_idx in range(len(integer_primals)):
            iv.append(integer_primals[relu_idx][bi:bi + 1])
        ret_p.append({"p": pv, "z": iv})
    return ret_p
