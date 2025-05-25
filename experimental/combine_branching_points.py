import torch

ret_sigmoid = torch.load('/home/zhouxingshi/nfs/bab/branching_points/sigmoid_dynamic.pt')
ret_sin = torch.load('/home/zhouxingshi/nfs/bab/branching_points/sin_dynamic.3_branches.pt')

ret = {
    'BoundSin': ret_sin,
    'BoundSigmoid': ret_sigmoid,
}

torch.save(ret, '/home/zhouxingshi/nfs/bab/branching_points/all.pt')
