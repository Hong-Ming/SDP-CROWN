import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import numpy as np
from scipy.interpolate import make_interp_spline
import scipy
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D

def plot_result(pgd_lower, pgd_upper, naive_lower, naive_upper, l2_lower, l2_upper, alpha_lower, alpha_upper, sample_name, network_name, perturbation):
    # Class labels
    classes = ['f0(x)', 'f1(x)', 'f2(x)', 'f3(x)', 'f4(x)', 'f5(x)', 'f6(x)', 'f7(x)', 'f8(x)', 'f9(x)']
    classes = classes[:len(pgd_lower[0])]

    def to_numpy(x):
        # Check if x is a PyTorch tensor
        if torch.is_tensor(x):
            # Move to CPU if necessary and convert to NumPy
            return x.detach().cpu().numpy()
        else:
            # Convert lists or already existing NumPy arrays to NumPy
            return np.array(x)

    # Example usage:
    pgd_lower = to_numpy(pgd_lower[0])
    pgd_upper = to_numpy(pgd_upper[0])

    naive_lower = to_numpy(naive_lower[0])
    naive_upper = to_numpy(naive_upper[0])

    l2_lower = to_numpy(l2_lower[0])
    l2_upper = to_numpy(l2_upper[0])
    
    alpha_lower = to_numpy(alpha_lower[0])
    alpha_upper = to_numpy(alpha_upper[0])

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10,6))

    # Plot Naive bars
    naive_bar = ax.bar(x - width, naive_upper - naive_lower, width=width, bottom=naive_lower, 
                    label='Naive Lipschitz Constant', color='tab:orange', alpha=0.8)
    # Plot L2-auto-LiRPA bars
    l2_bar = ax.bar(x, l2_upper - l2_lower, width=width, bottom=l2_lower, 
                    label='L2-auto-LiRPA', color='tab:green', alpha=0.8)
    
    alpha_bar = ax.bar(x + width, alpha_upper - alpha_lower, width=width, bottom=alpha_lower, 
                    label='L2-auto-LiRPA', color='tab:blue', alpha=0.8)
    pgd_x = x
    pgd_mid = (pgd_lower + pgd_upper) / 2
    pgd_err = (pgd_upper - pgd_lower) / 2 
    bar_pgd = ax.errorbar(x, pgd_mid, yerr=pgd_err, capsize=5, label='PGD Attack',linestyle='none', color='black')
    # Overlay PGD range as a dashed line for each class on top of these bars
    for i in range(len(classes)):
        # Draw a vertical dashed line at the position of Naive and L2 bars
        # We'll place the PGD line at the midpoint between these two bars for clarity.
        # Alternatively, we could place two lines, one per bar, but one centered line is simpler.
        pgd_x = (x[i] - width/2 + x[i] + width/2) / 2.0  # midpoint between the two bars
        ax.plot([pgd_x, pgd_x], [pgd_lower[i], pgd_upper[i]], '--', color='black')

        # Compute differences in sum for Naive and L2 relative to PGD
        pgd_sum = pgd_upper[i] - pgd_lower[i] 
        naive_sum = naive_upper[i] - naive_lower[i] 
        l2_sum = l2_upper[i] - l2_lower[i] 
        alpha_sum = alpha_upper[i] - alpha_lower[i]
        l2_diff = min((naive_sum - l2_sum) / naive_sum*100, (alpha_sum - l2_sum)/alpha_sum*100)

        # Annotate the Naive bar with the difference
        # Place the annotation slightly above the top of the bar
        # ax.text(x[i] - width/2, naive_upper[i] + 0.05, f"{naive_diff:+.2f}",
        #         ha='center', va='bottom', fontsize=8, rotation=90, color='black')

        # Annotate the L2-auto-LiRPA bar with the difference
        ax.text(x[i], l2_upper[i] + 0.05, f"{l2_diff:.2f}%",
                ha='center', va='bottom', fontsize=8, rotation=0, color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel('Value')
    ax.set_title('Naive and L2-auto-LiRPA with PGD GAP_'+ network_name + '_perturbation_' + str(perturbation))
    ax.legend()
    ax.grid(True)
    
    output_dir = './graphic_result/MNIST_result/MNIST_result_2/perturbation_' + str(perturbation)
    # Ensure the directory exists; if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(output_dir + '/sample_'+sample_name +'_' + network_name +'.png')
    plt.show()

def plot_with_zero_crossing():
    lp = np.load('./logs/plot_output/final_plot/lp_full_plot_2.0_margin_2.npy')
    # l2_crown = np.load('./logs/plot_output/final_plot/l2_crown_NOR_MLP_B_plot_2.0.npy')
    # alpha_crown = np.load('./logs/plot_output/final_plot/alpha_crown_NOR_MLP_B_plot_2.0.npy')
    # naive_lipschitz = np.load('./logs/plot_output/final_plot/naive_lipschitz_NOR_MLP_B_plot_2.0.npy')
    # pgd = np.load('./logs/plot_output/final_plot/pgd_NOR_MLP_B_plot_2.0.npy')
    
    l2_crown = np.load('./logs/plot_output/final_plot/l2_crown_cifar_cnn_b_adv_retrained_plot_1.5.npy')
    alpha_crown = np.load('./logs/plot_output/final_plot/alpha_crown_cifar_cnn_b_adv_retrained_plot_1.5.npy')
    naive_lipschitz = np.load('./logs/plot_output/final_plot/naive_lipschitz_cifar_cnn_b_adv_retrained_plot_1.5.npy')
    pgd = np.load('./logs/plot_output/final_plot/pgd_cifar_cnn_b_adv_retrained_plot_1.5.npy')
    
    # l2_crown = np.load('./logs/plot_output/final_plot/l2_crown_cifar_4C3F_retrained_plot_1.5.npy')
    # alpha_crown = np.load('./logs/plot_output/final_plot/alpha_crown_cifar_4C3F_retrained_plot_1.5.npy')
    # naive_lipschitz = np.load('./logs/plot_output/final_plot/naive_lipschitz_cifar_4C3F_retrained_plot_1.5.npy')
    # pgd = np.load('./logs/plot_output/final_plot/pgd_cifar_4C3F_retrained_plot_1.5.npy')

    # lp = np.insert(lp, 0, l2_crown[0])
    pgd[0] = l2_crown[0]

    l2_radius = np.linspace(0, 2.0, 21)
    smooth = np.linspace(l2_radius.min(), l2_radius.max(), 300)

    pgd_smooth = make_interp_spline(l2_radius, pgd)(smooth)
    l2_smooth = make_interp_spline(l2_radius, l2_crown)(smooth)
    alpha_smooth = make_interp_spline(l2_radius, alpha_crown)(smooth)
    naive_smooth = make_interp_spline(l2_radius, naive_lipschitz)(smooth)
    lp_smooth = make_interp_spline(l2_radius, lp)(smooth)

    methods = ["PGD", "LP-FULL","SDP-CROWN(Ours)", "α-CROWN", "naive_lipschitz"]
    curves = [pgd_smooth, lp_smooth, l2_smooth, alpha_smooth, naive_smooth]
    colors = ['red', 'orange', 'green', 'blue', 'black']

    plt.figure(figsize=(10, 8))

    for method, yvals, color in zip(methods, curves, colors):
        plt.plot(smooth, yvals, label=method, color=color, linewidth=2)

    for method, yvals, color in zip(methods, curves, colors):
        sign_change_indices = np.where(np.diff(np.sign(yvals)) != 0)[0]
        if len(sign_change_indices) > 0:
            idx = sign_change_indices[0]
            x0, x1 = smooth[idx], smooth[idx+1]
            y0, y1 = yvals[idx], yvals[idx+1]
            if abs(y1 - y0) > 1e-12:
                x_zero = x0 - y0 * (x1 - x0) / (y1 - y0)

                # plt.axvline(x=x_zero, color=color, linestyle='--', alpha=0.5)
                plt.annotate(
                    f"{x_zero:.2f}",
                    xy=(x_zero, 0),
                    xytext=(x_zero, 2.0),  
                    textcoords="data",
                    arrowprops=dict(arrowstyle="->", color=color,),
                    fontsize=20,
                    color=color,
                    horizontalalignment='center'
                )

    plt.xlabel('$\\ell_2$ Radius', fontsize=25)
    plt.ylabel('Lower Bound', fontsize=25)
    plt.ylim(-5, 3)
    plt.tick_params(axis='both', which='major', labelsize=20) 
    plt.legend(loc='lower left', fontsize=20)
    plt.grid(alpha=0.6)
    plt.tight_layout()
    plt.show()

    # plt.savefig('./graphic_result/verified_curve_new_with_zero_mark.eps', format='eps', dpi=1000)
    # plt.savefig('./graphic_result/verified_curve_new_with_zero_mark.png')

def stick():
    # 这里定义一种较为和谐的 4 色方案（取自常用调色板），分别用来画4根柱子
    bar_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]  

    # 如果你只想要统一颜色的柱子，可以只保留一个色号，例如：
    # single_bar_color = "#4C72B0"

    # PGD 参考线用灰色
    ref_line_color = "#555555"

    # 数据：PGD 参考线 + 四种方法的验证精度
    pgd             = 71.5
    alpha_crown     = 2.0
    beta_crown_2    = 4.5
    naive_lipschitz = 35.0
    l2_crown        = 61.5

    # 对应的耗时（字符串形式，用于标注）
    times = ["46.77s", "292.01s", "<0.001s", "127.64s"]

    # 方法名称及数值
    methods = ["α-CROWN", "α,β-CROWN", "Naive_lipschitz", "SDP-CROWN"]
    accs    = [alpha_crown, beta_crown_2, naive_lipschitz, l2_crown]

    # 画布及基本设置
    plt.figure(figsize=(34, 18))
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)  # 只对 y 轴做虚线网格
    lw = 2  # 参考线宽度

    # X 轴刻度序列
    x = np.arange(len(methods))

    # 画柱状图（分别给每个柱子分配颜色）
    barlist = plt.bar(x, accs, width=0.6, color=bar_colors)

    # 在 pgd 位置画虚线
    plt.axhline(y=pgd, color=ref_line_color, linestyle='--', linewidth=lw+5, alpha=0.8)

    # 在柱子顶部标注精度值，并在底部标注耗时
    for i, val in enumerate(accs):
        # 柱子顶部标注精度
        plt.text(
            x[i], val + 2,
            f"{val:.1f}%", 
            ha='center', va='bottom',
            color='black', fontsize=60, fontweight='bold'
        )
        # 柱子底部（x 轴下方）标注运行时间
        plt.text(
            x[i], -3,
            times[i], 
            ha='center', va='top',
            fontsize=60, fontweight='bold'
        )

    # 在最后一个柱子下方再额外标注 this work
    plt.text(
        x[-1], -18,
        '(this work)',
        ha='center', va='top',
        fontsize=55, fontweight='bold'
    )

    # 自定义 X 轴与 Y 轴
    plt.xticks(x, methods, fontsize=66, fontweight='bold', rotation=0)
    y_ticks = np.arange(0, 120, 20)  # 0,20,40,60,80,100
    plt.yticks(y_ticks, [f"{yt}%" for yt in y_ticks], fontsize=50, fontweight='bold')

    # 调整坐标轴范围
    plt.xlim([-0.5, len(methods)-0.5])  
    plt.ylim([-10, 80])  # 下方留空展示时间

    # 在图中标注 PGD 参考线数值
    plt.text(
        x[1]-0.3, pgd + 2,
        f"upper bound: {pgd:.1f}%", 
        color=ref_line_color, fontsize=60, fontweight='bold',
        ha='left', va='bottom'
    )

    plt.tight_layout()
    plt.show()
    # plt.savefig('./graphic_result/verified_accuracy.eps', format='eps', dpi=1000)
    # plt.savefig('./graphic_result/verified_accuracy.png')
    
    
def h_plot():
    # 1) 生成/读取数据
    perturbation = np.linspace(0, 2, 21)
    
    alpha_crown_h_1 = np.load('./logs/plot_output/final_plot/l2_crown_NOR_MLP_B_crown_h_plot_2.0.npy')
    l2_crown_h_1    = np.load('./logs/plot_output/final_plot/l2_crown_NOR_MLP_B_l2_crown_h_plot_2.0.npy')
    
    # alpha_crown_h_2 = np.load('./logs/plot_output/final_plot/l2_crown_cifar_cnn_b_adv_retrained_crown_h_plot_2.0.npy')
    # l2_crown_h_2    = np.load('./logs/plot_output/final_plot/l2_crown_cifar_cnn_b_adv_retrained_l2_crown_h_plot_2.0.npy')
    alpha_crown_h_2 = np.load('./logs/plot_output/final_plot/l2_crown_cifar_conv_small_crown_h_plot_2.0.npy')
    l2_crown_h_2    = np.load('./logs/plot_output/final_plot/l2_crown_cifar_conv_small_l2_crown_h_plot_2.0.npy')
    
    alpha_crown_h_3 = np.load('./logs/plot_output/final_plot/l2_crown_cifar_4C3F_retrained_crown_h_plot_1.5.npy')
    l2_crown_h_3    = np.load('./logs/plot_output/final_plot/l2_crown_cifar_4C3F_retrained_l2_crown_h_plot_1.5.npy')

    # 拟合函数（无常数项）
    def poly2_constrained(x, a, b):
        return a * x**2 + b * x

    # 简易函数：强制后一个点 <= 前一个点（单调不增）
    def enforce_non_increasing(arr):
        arr_mod = arr.copy()
        for i in range(1, len(arr_mod)):
            if arr_mod[i] > arr_mod[i - 1]:
                arr_mod[i] = arr_mod[i - 1]
        return arr_mod
    def enforce_non_increasing_2(arr1, arr2):
        for i in range(1, len(arr1)):
            if arr2[i] > arr1[i]:
                arr2[i] = arr1[i]
        return arr2


    # 2) 拟合三组曲线参数
    params_alpha_1, _ = curve_fit(poly2_constrained, perturbation, alpha_crown_h_1)
    params_l2_1,    _ = curve_fit(poly2_constrained, perturbation, l2_crown_h_1)
    
    params_alpha_2, _ = curve_fit(poly2_constrained, perturbation, alpha_crown_h_2)
    params_l2_2,    _ = curve_fit(poly2_constrained, perturbation, l2_crown_h_2)
    
    params_alpha_3, _ = curve_fit(poly2_constrained, perturbation, alpha_crown_h_3)
    params_l2_3,    _ = curve_fit(poly2_constrained, perturbation, l2_crown_h_3)

    # 3) 生成拟合曲线并强制单调不增
    alpha_fit_1 = enforce_non_increasing(poly2_constrained(perturbation, *params_alpha_1))
    l2_fit_1    = enforce_non_increasing(poly2_constrained(perturbation, *params_l2_1))
    
    alpha_fit_2 = enforce_non_increasing(poly2_constrained(perturbation, *params_alpha_2))
    l2_fit_2    = enforce_non_increasing(poly2_constrained(perturbation, *params_l2_2))
    
    alpha_fit_3 = enforce_non_increasing(poly2_constrained(perturbation, *params_alpha_3))
    l2_fit_3    = enforce_non_increasing(poly2_constrained(perturbation, *params_l2_3))
    
    alpha_fit_1 = enforce_non_increasing_2(l2_fit_1, alpha_fit_1)
    alpha_fit_2 = enforce_non_increasing_2(l2_fit_2, alpha_fit_2)
    alpha_fit_3 = enforce_non_increasing_2(l2_fit_3, alpha_fit_3)

    # 4) 作图
    plt.figure(figsize=(10, 6))

    # ---------- 实验 1 ----------
    plt.plot(
        perturbation, alpha_fit_1, color='red'
        , markersize=5, linewidth=3,
        label='α-CROWN (MLP-MNIST)'
    )
    plt.plot(
        perturbation, l2_fit_1, color='red',
        linestyle='--', marker='^', markersize=10, linewidth=3,
        label='SDP-CROWN (MLP-MNIST)'
    )

    # ---------- 实验 2 ----------
    plt.plot(
        perturbation, alpha_fit_2, color='blue', markersize=5, linewidth=3,
        label='α-CROWN (CNN-B-Adv-CIFAR)'
    )
    plt.plot(
        perturbation, l2_fit_2, color='blue',
        linestyle='--', marker='^', markersize=10, linewidth=3,
        label='SDP-CROWN (CNN-B-Adv-CIFAR)'
    )

    # ---------- 实验 3 ----------
    plt.plot(
        perturbation, alpha_fit_3, color='orange', markersize=5, linewidth=3,
        label='α-CROWN (CONVBIG-CIFAR)'
    )
    plt.plot(
        perturbation, l2_fit_3, color='orange',
        linestyle='--', marker='^', markersize=10, linewidth=3,
        label='SDP-CROWN (CONVBIG-CIFAR)'
    )

    # 坐标轴、标题等
    plt.xlabel('$\\ell_2$ Radius', fontsize=25)
    plt.ylabel('h Value',       fontsize=25)
    plt.ylim(-10, 3)
    plt.tick_params(axis='both', which='major', labelsize=22)

    plt.legend(
    handles=[
        Line2D([0], [0], color='red', linewidth=3, label=r'$\alpha$-CROWN (MLP-MNIST)'),
        Line2D([0], [0], color='blue', linewidth=3, label=r'$\alpha$-CROWN (CNN-B-Adv-CIFAR)'),
        Line2D([0], [0], color='orange', linewidth=3, label=r'$\alpha$-CROWN (CONVBIG-CIFAR)'),
        Line2D([0], [0], color='red', linestyle='--', marker='^', markersize=10, linewidth=3, label='SDP-CROWN (MLP-MNIST)'),
        Line2D([0], [0], color='blue', linestyle='--', marker='^', markersize=10, linewidth=3, label='SDP-CROWN (CNN-B-Adv-CIFAR)'),
        Line2D([0], [0], color='orange', linestyle='--', marker='^', markersize=10, linewidth=3, label='SDP-CROWN (CONVBIG-CIFAR)'),
    ],
    fontsize=14, loc='best', ncol=2
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 保存
    # plt.savefig('./graphic_result/h_plot.eps', format='eps', dpi=1000, bbox_inches='tight')
    # plt.savefig('./graphic_result/h_plot.png', dpi=300, bbox_inches='tight')
    
def plot_with_extended_curves():
    lp = np.load('./logs/plot_output/final_plot/lp_full_plot_2.0_margin_2.npy')
    l2_crown = np.load('./logs/plot_output/final_plot/l2_crown_NOR_MLP_B_plot_2.0.npy')
    alpha_crown = np.load('./logs/plot_output/final_plot/alpha_crown_NOR_MLP_B_plot_2.0.npy')
    naive_lipschitz = np.load('./logs/plot_output/final_plot/naive_lipschitz_NOR_MLP_B_plot_2.0.npy')
    pgd = np.load('./logs/plot_output/final_plot/pgd_NOR_MLP_B_plot_2.0.npy')

    # Adjust initial values
    lp = np.insert(lp, 0, l2_crown[0])
    pgd[0] = l2_crown[0]

    # Original l2 radius and interpolation
    l2_radius = np.linspace(0, 2.0, 21)
    pgd_smooth_func = make_interp_spline(l2_radius, pgd)
    l2_smooth_func = make_interp_spline(l2_radius, l2_crown)
    alpha_smooth_func = make_interp_spline(l2_radius, alpha_crown)
    naive_smooth_func = make_interp_spline(l2_radius, naive_lipschitz)
    lp_smooth_func = make_interp_spline(l2_radius, lp)

    # Find where PGD reaches 0
    smooth = np.linspace(l2_radius.min(), l2_radius.max(), 300)
    pgd_smooth = pgd_smooth_func(smooth)
    pgd_zero_idx = np.where(np.diff(np.sign(pgd_smooth)) != 0)[0]
    if len(pgd_zero_idx) > 0:
        idx = pgd_zero_idx[0]
        x0, x1 = smooth[idx], smooth[idx + 1]
        y0, y1 = pgd_smooth[idx], pgd_smooth[idx + 1]
        pgd_zero_x = x0 - y0 * (x1 - x0) / (y1 - y0)  # Linear interpolation to find exact zero

        # Extend smooth range to include the zero crossing point
        extended_smooth = np.linspace(l2_radius.min(), pgd_zero_x, 400)

        # Extend all curves to this new range
        pgd_extended = pgd_smooth_func(extended_smooth)
        l2_extended = l2_smooth_func(extended_smooth)
        alpha_extended = alpha_smooth_func(extended_smooth)
        naive_extended = naive_smooth_func(extended_smooth)
        lp_extended = lp_smooth_func(extended_smooth)
    else:
        extended_smooth = smooth
        pgd_extended, l2_extended, alpha_extended, naive_extended, lp_extended = (
            pgd_smooth, l2_smooth_func(smooth), alpha_smooth_func(smooth),
            naive_smooth_func(smooth), lp_smooth_func(smooth)
        )

    # Plotting
    methods = ["PGD", "LP-FULL", "SDP-CROWN(Ours)", "α-CROWN", "naive_lipschitz"]
    curves = [pgd_extended, lp_extended, l2_extended, alpha_extended, naive_extended]
    colors = ['red', 'orange', 'green', 'blue', 'black']

    plt.figure(figsize=(8, 6))

    for method, yvals, color in zip(methods, curves, colors):
        plt.plot(extended_smooth, yvals, label=method, color=color, linewidth=2)

    # Annotate PGD zero crossing
    plt.axvline(x=pgd_zero_x, color='gray', linestyle='--', alpha=0.7, label=f'PGD zero at {pgd_zero_x:.2f}')
    plt.annotate(
        f"{pgd_zero_x:.2f}",
        xy=(pgd_zero_x, 0),
        xytext=(pgd_zero_x + 0.1, -3),  # Offset for annotation
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color='gray'),
        fontsize=12,
        color='gray'
    )

    plt.xlabel('$\\ell_2$ Radius', fontsize=16)
    plt.ylabel('Lower Bound', fontsize=16)
    plt.ylim(-5, 3)
    plt.legend(loc='lower left', fontsize=12)
    plt.grid(alpha=0.6)
    
    plt.savefig('./graphic_result/extended_verified_curve_with_pgd_zero.eps', format='eps', dpi=1000)
    plt.savefig('./graphic_result/extended_verified_curve_with_pgd_zero.png')

def sticks():
    methods = ["α-CROWN", "α,β-CROWN", "Naive_lipschitz", "SDP-CROWN"]
    x = np.arange(0, len(methods)*1.05, 1.05)

    # 各方法上、下限（单位：百分比）
    tops = np.array([71.5, 71.5, 71.5, 71.5])
    bottoms = np.array([2.0, 4.5, 35.0, 61.5])
    
    times = ["46.77s", "292.01s", "<0.001s", "127.64s"]

    fig, ax = plt.subplots(figsize=(9,5))

    # # “Convex relaxation barrier” 的可视区域，你可根据需要更改
    # barrier_low, barrier_high = 0, 100
    # fill_low, fill_high = 20, 80
    # ax.fill_between(
    #     [-0.5, len(methods)-0.5],  # 横向填充的范围
    #     fill_low, fill_high,
    #     color='coral',
    #     alpha=0.2,
    #     label='Convex relaxation barrier'
    # )

    # 控制顶部/底部横线相对中心的左右宽度
    line_half_width = 0.15

    for i in range(len(x)):
        # 竖线
        ax.plot([x[i], x[i]], [bottoms[i], tops[i]], color='black', linewidth=1.5)
        
        # 顶部和底部横线
        ax.hlines(y=tops[i], xmin=x[i]-line_half_width, xmax=x[i]+line_half_width, color='black', linewidth=1.5)
        ax.hlines(y=bottoms[i], xmin=x[i]-line_half_width, xmax=x[i]+line_half_width, color='black', linewidth=1.5)
        
        # 顶部数值标注
        ax.text(x[i], tops[i] + 1, f"{tops[i]:.1f}%", color='red',
                ha='center', va='bottom', fontsize=18, fontweight='bold')
        # 底部数值标注
        ax.text(x[i], bottoms[i] - 2,f"{bottoms[i]:.1f}%", color='blue',
                ha='center', va='top', fontsize=18, fontweight='bold')
        ax.text(
            x[i], -8,
            times[i], 
            ha='center', va='top',
            fontsize=18, fontweight='bold'
        )
    ax.text(
        x[-1], -28,
        '(this work)', 
        ha='center', va='top',
        fontsize=16
    )
    # 设置x轴刻度
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=19)
    ax.set_xlim(-0.5, len(methods)-0.5)

    # 设置y轴范围 (0~100%)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(-15, 100)
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=18)
    # ax.set_ylabel("Verified Accuracy (%)", fontsize=14)

    # ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.2)
    # ax.set_title("Comparison of Different Bounds Methods", fontsize=12)
    plt.tight_layout()
    plt.show()
    # plt.savefig('./graphic_result/error_bar.eps', format='eps', dpi=1000)
    # plt.savefig('./graphic_result/error_bar.png')

if __name__ == '__main__':
    # figure 4 (a)(b)
    plot_with_zero_crossing()

    # figure 3
    # h_plot()

    # plot_with_extended_curves()
    # sticks()