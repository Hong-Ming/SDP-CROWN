"""For debugging: save the state when calling a compute_bounds.

https://colab.research.google.com/drive/17DwP80kQZY0ZgFbywgJU1dtGB8irX5Fo
"""

#  ~/output/0427/debug/compute_bounds
self._count_compute_bounds = getattr(self, '_count_compute_bounds', 0) + 1
save_path = f'/home/zhouxingshi/output/0427/debug/compute_bounds/{self._count_compute_bounds}.pkl'
state = {'alphas': {}, 'betas': {}, 'interm_bounds': interm_bounds}
for node in self.nodes():
    if getattr(node, 'alpha', None) is not None:
        state['alphas'][node.name] = node.alpha
    if getattr(node, 'sparse_betas', None) is not None:
        state['betas'][node.name] = {
            'val': node.sparse_betas[0].val,
            'sign': node.sparse_betas[0].sign,
            'loc': node.sparse_betas[0].loc,
            'bias': node.sparse_betas[0].bias}
print(f'Saving state to {save_path}')
torch.save(state, save_path)
