import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import numpy as np
import torch
import matplotlib as mpl
mpl.use('TkAgg')

from auto_LiRPA.bound_ops import (BoundMul, BoundTanh, BoundSigmoid, BoundSin,
                                  BoundCos, BoundAtan, BoundTan, BoundPow, BoundGelu, BoundHardTanh, BoundConstant)

function = 'sin'
range_l = -5.
range_u = 5.
x_l = -math.pi
x_u = math.pi
clip_min = -1
clip_max = 1
hardtanh_chosen = "same-slope"

if function == 'hardtanh':
    bound = BoundHardTanh({}, [], 0, {})
    min_val = BoundConstant(attr={'value': torch.tensor(clip_min)})  # change constant value here for min & max
    max_val = BoundConstant(attr={'value': torch.tensor(clip_max)})  # change constant value here for min & max

    def get_bound(x_l, x_u, min_val, max_val):
        class Input(object):
            def __init__(self, x_l, x_u):
                self.lower = torch.tensor([[x_l]], dtype=torch.float32)
                self.upper = torch.tensor([[x_u]], dtype=torch.float32)

            # Create a fake input object with lower and upper bounds.
        i = Input(x_l, x_u)
        bound.hardtanh_options = hardtanh_chosen.lower()
        bound.bound_relax(i, min_val, max_val)
        return bound.lw.item(), bound.lb.item(), bound.uw.item(), bound.ub.item()
    def f(x, min_val, max_val):
        return torch.nn.functional.hardtanh(torch.tensor(x), min_val, max_val)

elif function == 'x^2':
    def get_bound(x_l, x_u):
        a_l, _, b_l, a_u, _, b_u = BoundMul.get_bound_square(torch.tensor(x_l), torch.tensor(x_u))
        return a_l, b_l, a_u, b_u

    def f(x):
        return x * x

elif function in ['sin', 'cos']:
    if function == 'sin':
        range_l = -3. * math.pi
        range_u = 3. * math.pi
        bound = BoundSin({}, [], 0, {})
        def f(x):
            return np.sin(x)
    elif function == 'cos':
        range_l = -4. * math.pi
        range_u = 4. * math.pi
        bound = BoundCos({}, [], 0, {})
        def f(x):
            return np.cos(x)


    def get_bound(x_l, x_u, ratio, use_alpha=False):
        class Input(object):
            def __init__(self, x_l, x_u):
                self.lower = torch.tensor([[x_l]], dtype=torch.float32)
                self.upper = torch.tensor([[x_u]], dtype=torch.float32)
        # Create a fake input object with lower and upper bounds.
        i = Input(x_l, x_u)
        bound.inputs = [i]
        bound.opt_init()
        bound.opt_stage = 'init'
        bound._start = '/0'
        bound.bound_relax(i, init=True)
        bound.opt_stage = 'opt'
        bound.init_opt_parameters([['/0', 1]])

        if use_alpha:
            lb, ub = i.lower, i.upper
            step = 2 * torch.pi / (BoundSin.n_table_entries - 1)
            lb_cycles = torch.floor(lb / (2 * torch.pi)) * (2 * torch.pi)
            lb = torch.clamp(lb - lb_cycles, min=0, max=2 * torch.pi)
            ub_cycles = torch.floor(ub / (2 * torch.pi)) * (2 * torch.pi)
            ub = torch.clamp(ub - ub_cycles, min=0, max=2 * torch.pi)
            indices_lb = lb.div(step).long()
            indices_ub = ub.div(step).long()
            tangent_left_lower = BoundSin.xl_lower_tb[0][indices_lb]
            tangent_left_upper = BoundSin.xl_upper_tb[0][indices_lb]
            tangent_right_lower = BoundSin.xu_lower_tb[0][indices_ub]
            tangent_right_upper = BoundSin.xu_upper_tb[0][indices_ub]

            bound.alpha['/0'].data[0:2, :] = (i.upper - i.lower) * ratio + i.lower
            bound.alpha['/0'].data[2:4, :] = (i.upper - i.lower) * ratio + i.lower
            bound.alpha['/0'].data[4:6, :] = (
                bound.convex_concave * ((bound.alpha['/0'][4:6, :] - i.lower) * ratio + i.lower)
                + bound.convex_concave * ((bound.alpha['/0'][4:6, :] - i.upper) * ratio + i.upper))
            bound.alpha['/0'].data[6:8, :] = (
                    bound.convex_concave * ((bound.alpha['/0'][6:8, :] - i.upper) * ratio + i.upper)
                    + bound.convex_concave * ((bound.alpha['/0'][6:8, :] - i.lower) * ratio + i.lower))
            bound.alpha['/0'].data[8:10, :] = (tangent_right_lower
                    - tangent_left_lower) * ratio + tangent_left_lower
            bound.alpha['/0'].data[10:12, :] = (tangent_right_upper
                    - tangent_left_upper) * ratio + tangent_left_upper
        bound.bound_relax(i, init=True, dim_opt=1)
        return bound.lw[0].item(), bound.lb[0].item(), bound.uw[1].item(), bound.ub[1].item()
else:
    if function == 'tanh':
        bound = BoundTanh({}, [], 0, {})
        def f(x):
            return np.tanh(x)
    elif function == 'gelu':
        bound = BoundGelu({}, [], 0, {})
        def f(x):
            return torch.nn.functional.gelu(torch.tensor(x))
    elif function == 'sigmoid':
        bound = BoundSigmoid({}, [], 0, {})
        def f(x):
            return 1 / (1 + np.exp(-x))
    elif function == 'arctan':
        range_l = - math.pi
        range_u = math.pi
        bound = BoundAtan({}, [], 0, {})
        def f(x):
            return np.arctan(x)
    elif function == 'pow3':
        range_l = -5.
        range_u = 5.
        bound = BoundPow({}, [], 0, {})
        bound.exponent = 3
        bound.precompute_relaxation(bound.act_func, bound.d_act_func)
        def f(x):
            return x**3
    elif function == 'tan':
        period = -1 * torch.pi
        range_l = -0.5 * torch.pi + 0.1 + period
        range_u = 0.5 * torch.pi - 0.1 + period
        x_l = -1. + period
        x_u = 1. + period
        bound = BoundTan({}, [], 0, {})
        def f(x):
            return np.tan(x)
    else:
        raise NotImplementedError(function)

    def get_bound(x_l, x_u):
        class Input(object):
            def __init__(self, x_l, x_u):
                self.lower = torch.tensor([[x_l]], dtype=torch.float32)
                self.upper = torch.tensor([[x_u]], dtype=torch.float32)
        # Create a fake input object with lower and upper bounds.
        i = Input(x_l, x_u)
        bound.bound_relax(i, init=True)
        return bound.lw.item(), bound.lb.item(), bound.uw.item(), bound.ub.item()

def fu(x, a, b):
    return a * x + b

def fl(x, a, b):
    return a * x + b

# Get initial values.
ratio = 0.
if function in ['sin', 'cos']:
    a_l, b_l, a_u, b_u = get_bound(x_l, x_u, ratio)
elif function in ['hardtanh']:
    a_l, b_l, a_u, b_u = get_bound(x_l, x_u, min_val, max_val)
else:
    a_l, b_l, a_u, b_u = get_bound(x_l, x_u)
fig = plt.figure()
# Leave some space below for sliders.
plt.subplots_adjust(bottom=0.25)
ax = fig.gca()
x = np.linspace(range_l, range_u, 1001)
# Plot main function.
if function in ['hardtanh']:
    mp, = plt.plot(x, f(x, clip_min, clip_max), color='skyblue', linewidth=1)
else:
    mp, = plt.plot(x, f(x), color='skyblue', linewidth=1)
y_l, y_u = ax.get_ylim()
# Plot upper and lower bounds.
l_p, = plt.plot(x, fl(x, a_l, b_l), color='olive', linewidth=1, label="lb")
u_p, = plt.plot(x, fu(x, a_u, b_u), color='red', linewidth=1, label="ub")
# Plot two straight lines for ub and lb.
l_pl, = plt.plot(x_l * np.ones_like(x), np.linspace(y_l, y_u, 1001), color='blue', linestyle='dashed', linewidth=1)
u_pl, = plt.plot(x_u * np.ones_like(x), np.linspace(y_l, y_u, 1001), color='blue', linestyle='dashed', linewidth=1)
plt.ylim(y_l, y_u)
plt.legend()

# Create sliders.
axcolor = 'lightgoldenrodyellow'
ax_xl = plt.axes([0.25, 0.00, 0.65, 0.03], facecolor=axcolor)
ax_xu = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
s_xl = Slider(ax_xl, 'lb', range_l, range_u, valinit=x_l)
s_xu = Slider(ax_xu, 'ub', range_l, range_u, valinit=x_u)
if function in ['sin', 'cos']:
    ax_al = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
    alpha = Slider(ax_al, 'alpha', 0, 1, valinit=ratio)
elif function == 'hardtanh':
    ax_al = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
    ax_al2 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    clip_min_slider = Slider(ax_al, 'clip_min', range_l, range_u, valinit=clip_min)
    clip_max_slider = Slider(ax_al2, 'clip_max', range_l, range_u, valinit=clip_max)
    ax_radio = plt.axes([0.90, 0.70, 0.1, 0.1], facecolor=axcolor)
    radio = RadioButtons(ax_radio, ('Same-Slope', 'Adaptive'), label_props={'color': 'cmy', 'fontsize': [8, 8, 16]})


def update_xu(val):
    # Update upper bound value, and update figure.
    global x_u
    x_u = val
    if x_u < x_l:
        print("x_u < x_l")
        return
    if function in ['sin', 'cos']:
        a_l, b_l, a_u, b_u = get_bound(x_l, x_u, ratio)
    elif function in ['hardtanh']:
        global clip_max, clip_min
        max_val = BoundConstant(attr={'value': torch.tensor(clip_max)})  # change constant value here for min & max
        min_val = BoundConstant(attr={'value': torch.tensor(clip_min)})
        a_l, b_l, a_u, b_u = get_bound(x_l, x_u, min_val, max_val)
    else:
        a_l, b_l, a_u, b_u = get_bound(x_l, x_u)
    u_p.set_ydata(fu(x, a_u, b_u))
    l_p.set_ydata(fl(x, a_l, b_l))
    u_pl.set_xdata(x_u * np.ones_like(x))
    fig.canvas.draw_idle()

def update_xl(val):
    # Update lower bound value, and update figure.
    global x_l
    x_l = val
    if x_u < x_l:
        print("x_u < x_l")
        return
    if function in ['sin', 'cos']:
        a_l, b_l, a_u, b_u = get_bound(x_l, x_u, ratio)
    elif function in ['hardtanh']:
        global clip_max, clip_min
        max_val = BoundConstant(attr={'value': torch.tensor(clip_max)})  # change constant value here for min & max
        min_val = BoundConstant(attr={'value': torch.tensor(clip_min)})
        a_l, b_l, a_u, b_u = get_bound(x_l, x_u, min_val, max_val)
    else:
        a_l, b_l, a_u, b_u = get_bound(x_l, x_u)
    u_p.set_ydata(fu(x, a_u, b_u))
    l_p.set_ydata(fl(x, a_l, b_l))
    l_pl.set_xdata(x_l * np.ones_like(x))
    fig.canvas.draw_idle()

def update_alpha(val):
    # Update lower bound value, and update figure.
    global ratio
    ratio = val
    a_l, b_l, a_u, b_u = get_bound(x_l, x_u, ratio, use_alpha=True)
    u_p.set_ydata(fu(x, a_u, b_u))
    l_p.set_ydata(fl(x, a_l, b_l))
    l_pl.set_xdata(x_l * np.ones_like(x))
    fig.canvas.draw_idle()

def update_min(val):
    global clip_min
    clip_min = val
    if clip_max < clip_min:
        print("clip_max is less than clip_min")
        return
    max_val = BoundConstant(attr={'value': torch.tensor(clip_max)})
    min_val = BoundConstant(attr={'value': torch.tensor(clip_min)})
    a_l, b_l, a_u, b_u = get_bound(x_l, x_u, min_val, max_val)
    mp.set_ydata(f(x, clip_min, clip_max))
    u_p.set_ydata(fu(x, a_u, b_u))
    l_p.set_ydata(fl(x, a_l, b_l))
    l_pl.set_xdata(x_l * np.ones_like(x))
    fig.canvas.draw_idle()

def update_hardtanh(val):
    global hardtanh_chosen
    hardtanh_chosen = val
    max_val = BoundConstant(attr={'value': torch.tensor(clip_max)})
    min_val = BoundConstant(attr={'value': torch.tensor(clip_min)})
    a_l, b_l, a_u, b_u = get_bound(x_l, x_u, min_val, max_val)
    mp.set_ydata(f(x, clip_min, clip_max))
    u_p.set_ydata(fu(x, a_u, b_u))
    l_p.set_ydata(fl(x, a_l, b_l))
    l_pl.set_xdata(x_l * np.ones_like(x))
    fig.canvas.draw_idle()

def update_max(val):
    global clip_max
    clip_max = val
    if clip_max < clip_min:
        print("clip_max is less than clip_min")
        return
    max_val = BoundConstant(attr={'value': torch.tensor(clip_max)})
    min_val = BoundConstant(attr={'value': torch.tensor(clip_min)})
    a_l, b_l, a_u, b_u = get_bound(x_l, x_u, min_val, max_val)
    mp.set_ydata(f(x, clip_min, clip_max))
    u_p.set_ydata(fu(x, a_u, b_u))
    l_p.set_ydata(fl(x, a_l, b_l))
    l_pl.set_xdata(x_l * np.ones_like(x))
    fig.canvas.draw_idle()

s_xl.on_changed(update_xl)
s_xu.on_changed(update_xu)
if function in ['sin', 'cos']:
    alpha.on_changed(update_alpha)

if function == "hardtanh":
    clip_min_slider.on_changed(update_min)
    clip_max_slider.on_changed(update_max)
    radio.on_clicked(update_hardtanh)

plt.show()
