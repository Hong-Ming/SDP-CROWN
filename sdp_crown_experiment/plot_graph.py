import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, PchipInterpolator

def plot_one_subplot(
    ax,
    l2_radius,
    pgd,
    l2_crown,
    alpha_crown,
    naive_lipschitz,
    lp=None,
    title="",
    xlim=(0,2.0),
    ylim=(-5,3),
    alpha=0.5,
    move=False,
):
    if lp is not None:
        lp = np.insert(lp, 0, l2_crown[0])
        pgd[0] = l2_crown[0]

    # 为了平滑画图，需要更细的 x 坐标
    smooth_x = np.linspace(l2_radius.min(), l2_radius.max(), 800)

    # 分别对各个方法的数据做插值
    pgd_smooth = make_interp_spline(l2_radius, pgd)(smooth_x)
    l2_smooth = make_interp_spline(l2_radius, l2_crown, k=1)(smooth_x)
    # pchip_func = PchipInterpolator(l2_radius, l2_crown)
    # l2_smooth = pchip_func(smooth_x)
    # alpha_smooth = make_interp_spline(l2_radius, alpha_crown, k=3)(smooth_x)
    pchip_func = PchipInterpolator(l2_radius, alpha_crown)
    alpha_smooth = pchip_func(smooth_x)
    naive_smooth = make_interp_spline(l2_radius, naive_lipschitz)(smooth_x)

    # 准备方法名字和对应曲线、颜色；若没有lp数据就不画
    methods = ["PGD", "SDP-CROWN(Ours)", "α-CROWN", "Naive_Lipschitz"]
    curves = [pgd_smooth, l2_smooth, alpha_smooth, naive_smooth]
    colors = ['red', 'green', 'blue', 'black']

    if lp is not None:
        lp_smooth = make_interp_spline(l2_radius, lp)(smooth_x)
        methods.insert(1, "LP-FULL")      # 在 PGD 后插入
        curves.insert(1, lp_smooth)
        colors.insert(1, 'orange')

    # 画曲线
    for method, yvals, color in zip(methods, curves, colors):
        ax.plot(smooth_x, yvals, label=method, color=color, linewidth=2)

    # 对每一条曲线查找零交叉点并标注
    offset = 1.5
    if move:
        offset = 1.25
    for method, yvals, color in zip(methods, curves, colors):
        sign_change_indices = np.where(np.diff(np.sign(yvals)) != 0)[0]
        if len(sign_change_indices) > 0:
            idx = sign_change_indices[0]
            x0, x1 = smooth_x[idx], smooth_x[idx+1]
            y0, y1 = yvals[idx], yvals[idx+1]
            if abs(y1 - y0) > 1e-12:
                # 线性插值找交点
                x_zero = x0 - y0 * (x1 - x0) / (y1 - y0)
                ax.annotate(
                    f"{x_zero:.2f}",
                    xy=(x_zero, 0),
                    xytext=(x_zero, offset),
                    textcoords="data",
                    arrowprops=dict(arrowstyle="->", color=color),
                    fontsize=22,
                    color=color,
                    horizontalalignment='center'
                )

    ax.set_title(title, fontsize=20)
    ax.set_xticks(np.arange(0, xlim[-1], (xlim[-1]-0.1)/5))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('$\\ell_2$ Radius', fontsize=28)
    ax.set_ylabel('Lower Bound', fontsize=28)
    ax.grid(alpha=alpha)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.legend(loc='lower left', fontsize=20)

def plot_three_figures_together():
    # ========== 第一组数据 (示例: NOR_MLP_B_plot_2.0) ==========
    lp_1 = np.load('/home/haoc8/Robust_verification_of_L2-norm/logs/plot_output/final_plot/lp_full_plot_2.0_margin_2.npy')
    l2_crown_1 = np.load('./logs/plot_output/final_plot/l2_crown_NOR_MLP_B_plot_2.0.npy')
    alpha_crown_1 = np.load('./logs/plot_output/final_plot/alpha_crown_NOR_MLP_B_plot_2.0.npy')
    naive_lipschitz_1 = np.load('./logs/plot_output/final_plot/naive_lipschitz_NOR_MLP_B_plot_2.0.npy')
    pgd_1 = np.load('./logs/plot_output/final_plot/pgd_NOR_MLP_B_plot_2.0.npy')

    # 第一组的横坐标（从 0 到 2.0, 共 21 个点）
    radius_1 = np.linspace(0, 2.0, 21)

    # ========== 第二组数据 (示例: cifar_cnn_b_adv_retrained_plot_1.5) ==========
    l2_crown_2 = np.load('./logs/plot_output/final_plot/l2_crown_cifar_cnn_b_adv_retrained_plot_2.5.npy')
    alpha_crown_2 = np.load('./logs/plot_output/final_plot/alpha_crown_cifar_cnn_b_adv_retrained_plot_2.5.npy')
    naive_lipschitz_2 = np.load('./logs/plot_output/final_plot/naive_lipschitz_cifar_cnn_b_adv_retrained_plot_2.5.npy')
    pgd_2 = np.load('./logs/plot_output/final_plot/pgd_cifar_cnn_b_adv_retrained_plot_2.5.npy')

    # 假设第二组目前没有 LP 数据，lp_2 = None
    lp_2 = None
    radius_2 = np.linspace(0, 2.5, 21)

    # ========== 第三组数据 (示例: cifar_4C3F_retrained_plot_1.5) ==========
    l2_crown_3 = np.load('./logs/plot_output/final_plot/l2_crown_cifar_4C3F_retrained_plot_1.5.npy')
    alpha_crown_3 = np.load('./logs/plot_output/final_plot/alpha_crown_cifar_4C3F_retrained_plot_1.5.npy')
    naive_lipschitz_3 = np.load('./logs/plot_output/final_plot/naive_lipschitz_cifar_4C3F_retrained_plot_1.5.npy')
    pgd_3 = np.load('./logs/plot_output/final_plot/pgd_cifar_4C3F_retrained_plot_1.5.npy')

    # 同理，第三组如果没有 LP 数据，可以设为 None
    lp_3 = None
    radius_3 = np.linspace(0, 1.5, 21)
    pgd_1[0] = l2_crown_1[0]
    pgd_2[0] = l2_crown_2[0]
    pgd_3[0] = l2_crown_3[0]
    # ========== 开始绘图：三个子图 ========== 
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))

    # 第一个子图
    plot_one_subplot(
        ax=axs[0],
        l2_radius=radius_1,
        pgd=pgd_1,
        l2_crown=l2_crown_1,
        alpha_crown=alpha_crown_1,
        naive_lipschitz=naive_lipschitz_1,
        lp=lp_1,  # 只有第一组才有 LP
        title="MLP-MNIST",
        xlim=(-0.10,2.10),
        ylim=(-5,3),
    )

    # 第二个子图
    plot_one_subplot(
        ax=axs[1],
        l2_radius=radius_2,
        pgd=pgd_2,
        l2_crown=l2_crown_2,
        alpha_crown=alpha_crown_2,
        naive_lipschitz=naive_lipschitz_2,
        lp=lp_2,  # 无 LP，就传 None
        title="CNN-B-Adv-CIFAR",
        xlim=(-0.10,2.60),
        ylim=(-8,4),
        move=True,
    )

    # 第三个子图
    plot_one_subplot(
        ax=axs[2],
        l2_radius=radius_3,
        pgd=pgd_3,
        l2_crown=l2_crown_3,
        alpha_crown=alpha_crown_3,
        naive_lipschitz=naive_lipschitz_3,
        lp=lp_3,  # 无 LP，就传 None
        title="CONVBIG-CIFAR",
        xlim=(-0.10,1.60),
        alpha=0.5,
        ylim=(-8,5),
    )

    plt.tight_layout()
    # 如果需要保存
    plt.savefig('./graphic_result/three_subplots.png', dpi=300)
    plt.savefig('./graphic_result/three_subplots.eps', format='eps', dpi=1000)
    # 显示
    plt.show()


if __name__ == "__main__":
    plot_three_figures_together()
