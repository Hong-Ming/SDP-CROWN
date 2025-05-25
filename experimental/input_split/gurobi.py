import gurobipy as gp
from gurobipy import GRB
from tqdm import tqdm
from multiprocessing import Pool


gurobi_enhancement = arguments.Config['bab']['branching']['input_split']['gurobi_enhancement']
if gurobi_enhancement:
    # Experimental. Hard-coded for the output feedback model.

    unverified = torch.logical_not(
        stop_criterion_func(thresholds)(lb).any(dim=-1))

    weight = self.net['/257'].value.cpu()
    rhs = thresholds[0][1].cpu()
    cnt = 0

    indices = unverified.nonzero().squeeze(-1).tolist()

    args_gurobi = [
        (new_x.ptb.x_L[i].cpu(), new_x.ptb.x_U[i].cpu(), weight, rhs)
        for i in indices
    ]

    with Pool(4) as pool:
        lb_gurobi = pool.starmap(_solve_qp, args_gurobi)

    for i in range(len(indices)):
        if lb_gurobi[i] > 0:
            lb[indices[i], 1] = lb_gurobi[i]
            cnt += 1

    print('verified with gurobi:', cnt)


def _solve_qp(x_L, x_U, weights, thresholds, verbose=0):
    with gp.Env(empty=True) as env:
        # env.setParam("WLSAccessID", str)
        # env.setParam("WLSSECRET", str)
        # env.setParam("LICENSEID", int)
        env.setParam("OutputFlag", 0)
        env.start()

        # To read the model from a file, you can use:
        # with gp.read("glass4.mps", env) as model:

        with gp.Model(env=env) as m:
            m.setParam('Threads', 1)
            m.setParam('OutputFlag', 0)

            n = x_L.shape[-1]
            x = []
            for i in range(n):
                x.append(m.addVar(lb=x_L[i], ub=x_U[i], name=f'x_{i}'))

            obj = None
            for i in range(n):
                for j in range(n):
                    obj_ = x[i] * x[j] * weights[j][i]
                    if obj is None:
                        obj = obj_
                    else:
                        obj = obj + obj_

            m.setObjective(obj)
            m.optimize()
            # print('status', m.status)
            result = obj.getValue() - thresholds

    return result
