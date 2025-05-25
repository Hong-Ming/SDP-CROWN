""" Test inputs of general shapes (especially for matmul)"""
import torch
import torch.nn as nn
import numpy as np

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from auto_LiRPA.operators import BoundMatMul
from testcase import TestCase

BATCH_SIZE = 2

class GeneralShapeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_1 = torch.randn(3, 4)
        self.weight_2 = torch.randn(4, 3)
        self.weight_3 = torch.randn(3, 4)
        self.weight_4 = torch.randn(4, 4, 3)
        self.weight_5 = torch.randn(6, 3, 4)
        self.weight_6 = torch.randn(3, 5)
        self.relu = nn.ReLU()
    def forward(self, x):
        # Basic MatMul (B, 3) @ (3, 4) -> (B, 4)
        y1 = x.matmul(self.weight_1)

        # BoundUnsqueeze and BoundTile
        y2 = self.relu(y1)
        y2 = y2.unsqueeze(1).repeat(1, 5, 1)   # (B, 5, 4)
        y2 = y2.matmul(self.weight_2)   # (B, 5, 4) @ (4, 3) -> (B, 5, 3)

        # More dimensions on x
        y3 = self.relu(y2)
        y3 = y3.unsqueeze(1).repeat(1, 4, 1, 1)     # (B, 4, 5, 3)
        y3 = y3.matmul(self.weight_3)   # (B, 4, 5, 3) @ (3, 4) -> (B, 4, 5, 4)

        # More dimensions on weight
        y4 = self.relu(y3)
        y4 = y4.matmul(self.weight_4)   # (B, 4, 5, 4) @ (4, 4, 3) -> (B, 4, 5, 3)

        # Automatically broadcast x
        y5 = self.relu(y4)
        y5 = y5.unsqueeze(2)   # (B, 4, 1, 5, 3)
        y5 = y5.matmul(self.weight_5)   # (B, 4, 1, 5, 3) @ (6, 3, 4) -> (B, 4, 6, 5, 4)

        # Swap x and weight
        y6 = self.relu(y5)
        y6 = self.weight_6.matmul(y6)   # (3, 5) @ (B, 4, 6, 5, 4) -> (B, 4, 6, 3, 4)

        return y6

class TestGeneralShape(TestCase):
    def __init__(self, methodName='runTest', seed=1, ref_path=None, generate=False):
        ref_path = "data/test_general_shape_data"
        super().__init__(methodName, seed, ref_path, generate)

    def test(self):
        model = GeneralShapeModel()
        input = torch.rand((BATCH_SIZE, 3))
        eps = 100
        ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
        x = BoundedTensor(input, ptb)
        lirpa_model = BoundedModule(model, x)

        lb, ub = lirpa_model.compute_bounds(x, method="backward")

        # Test by sampling
        # sample_ptb = torch.rand(10000, *input.shape[1:]) * 2 * eps - eps
        # sample_inputs = input[0] + sample_ptb
        # sample_output = model(sample_inputs)
        # assert (sample_output <= ub[0]).all()
        # assert (sample_output >= lb[0]).all()

        self.result = []
        for node in lirpa_model.nodes():
            if type(node) == BoundMatMul:
                self.result.append((node.lower, node.upper))
        self.result.append((lb, ub))

        self.check()

if __name__ == '__main__':
    testcase = TestGeneralShape(generate=False)
    testcase.setUp()
    testcase.test()