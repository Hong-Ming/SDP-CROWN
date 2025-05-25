import numpy as np
from mayavi import mlab
from auto_LiRPA.bound_ops import BoundMin
from types import SimpleNamespace
import torch
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

xl0=-1.
xu0=1.
yl0=-1.
yu0=1.

module = BoundMin()
x_n = SimpleNamespace()
x_n.lower = torch.tensor(xl0)
x_n.upper = torch.tensor(xu0)
y_n = SimpleNamespace()
y_n.lower = torch.tensor(yl0)
y_n.upper = torch.tensor(yu0)
alpha_u, beta_u, gamma_u, alpha_l, beta_l, gamma_l = module._backward_relaxation(None, None, x_n, y_n)

alpha_l = alpha_l.numpy()
beta_l = beta_l.numpy()
gamma_l = gamma_l.numpy()
alpha_u = alpha_u.numpy()
beta_u = beta_u.numpy()
gamma_u = gamma_u.numpy()

def f(x, y):
    return np.min([x, y], axis=0)

def fu(x, y):
    return alpha_l * x + beta_l * y + gamma_l

def fl(x, y):
    return alpha_u * x + beta_u * y + gamma_u

range_min = -1.
range_max = 1.
# Draw 4 points for a rectangular area
mlab.points3d(xl0, yl0, 0, scale_factor=0.2)
mlab.points3d(xu0, yu0, 0, scale_factor=0.2)
mlab.points3d(xl0, yu0, 0, scale_factor=0.2)
mlab.points3d(xu0, yl0, 0, scale_factor=0.2)
x, y = np.mgrid[range_min:range_max:0.01, range_min:range_max:0.01]
# Multiplication function
mlab.surf(x, y, f, color=(0.0, 0.5, 0.))
# Upper bound.
su = mlab.surf(x, y, fu, color=(0.0, 0.5, 0.5), opacity=0.5)
# # Lower bound
sl = mlab.surf(x, y, fl, color=(0.5, 0.0, 0.5), opacity=0.5)
# z=0 plane
# mlab.surf(x, y, lambda x, y: 0 * x, color=(0.0, 0.0, 0.5), opacity=0.2)
# 4 lines for the rectangular area
mlab.plot3d([xl0, xl0], [yl0, yl0], [2 * range_min, 2 * range_max], tube_radius=0.025, tube_sides=8)
mlab.plot3d([xu0, xu0], [yl0, yl0], [2 * range_min, 2 * range_max], tube_radius=0.025, tube_sides=8)
mlab.plot3d([xu0, xu0], [yu0, yu0], [2 * range_min, 2 * range_max], tube_radius=0.025, tube_sides=8)
mlab.plot3d([xl0, xl0], [yu0, yu0], [2 * range_min, 2 * range_max], tube_radius=0.025, tube_sides=8)

def change(val):
    module.opt_stage = 'opt'
    module.init_opt_parameters([])
    module.alpha['/0'] = torch.ones(2, *x_n.lower.shape)*val
    node = SimpleNamespace()
    node.name = '/0'
    module.clip_alpha()
    alpha_u, beta_u, gamma_u, alpha_l, beta_l, gamma_l = module._backward_relaxation(None, None, x_n, y_n, node)
    alpha_l = alpha_l.numpy()
    beta_l = beta_l.numpy()
    gamma_l = gamma_l.numpy()
    alpha_u = alpha_u.numpy()
    beta_u = beta_u.numpy()
    gamma_u = gamma_u.numpy()
    sl.mlab_source.scalars = alpha_l * x + beta_l * y + gamma_l
    su.mlab_source.scalars = alpha_u * x + beta_u * y + gamma_u

plt.figure()
ax = plt.subplot(1, 1, 1)
slider = Slider(ax, valmin=0., valmax=1., label='test')
slider.on_changed(change)
plt.show()
mlab.show()

