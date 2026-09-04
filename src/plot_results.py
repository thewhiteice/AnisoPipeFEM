import matplotlib.pyplot as plt


def plot_sigma_results(r_vals, s_vals, s_filename=None, coord_system="cyl"):
    """
    绘制应力分布图

    Parameters:
    -----------
    r_vals : array_like
        径向坐标
    s_vals : ndarray, shape (N, 6)
        应力数据，六列的顺序取决于 coord_system
    s_filename : str, optional
        图表标题
    coord_system : str, {'cyl', 'mat'}
        指定 s_vals 的坐标系：
        - 'cyl' : 柱坐标系，顺序 [σ_r, σ_θ, σ_z, τ_θz, τ_rz, τ_rθ]
        - 'mat' : 材料主方向，顺序 [σ_1, σ_2, σ_3, τ_23, τ_13, τ_12]
    """
    if s_vals.shape[-1] != 6:
        raise ValueError("s_vals 最后一维必须为 6")

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(s_filename or "Stress Distributions")
    ax = axes.flatten()

    if coord_system == "cyl":
        # 柱坐标系标签
        labels = [
            (r"Radial stress $\sigma_r$ (MPa)", r"Radius $r$ (m)"),
            (r"Hoop stress $\sigma_\theta$ (MPa)", r"Radius $r$ (m)"),
            (r"Axial stress $\sigma_z$ (MPa)", r"Radius $r$ (m)"),
            (r"Shear stress $\sigma_{\theta z}$ (MPa)", r"Radius $r$ (m)"),
            (r"Shear stress $\sigma_{z r}$ (MPa)", r"Radius $r$ (m)"),
            (r"Shear stress $\sigma_{r \theta}$ (MPa)", r"Radius $r$ (m)"),
        ]
        # 数据顺序直接使用
        for i in range(6):
            ax[i].plot(r_vals, s_vals[:, i] / 1e6)
            ax[i].set_xlabel(labels[i][1])
            ax[i].set_ylabel(labels[i][0])
            ax[i].set_title(labels[i][0].split("(")[0])  # 简短标题

    elif coord_system == "mat":
        # 材料坐标系标签（纤维方向为1，横向为2，径向为3）
        labels = [
            (r"Fiber stress $\sigma_1$ (MPa)", r"Radius $r$ (m)"),
            (r"Transverse stress $\sigma_2$ (MPa)", r"Radius $r$ (m)"),
            (r"Radial stress $\sigma_3$ (MPa)", r"Radius $r$ (m)"),
            (r"Shear stress $\tau_{23}$ (MPa)", r"Radius $r$ (m)"),
            (r"Shear stress $\tau_{13}$ (MPa)", r"Radius $r$ (m)"),
            (r"Shear stress $\tau_{12}$ (MPa)", r"Radius $r$ (m)"),
        ]
        for i in range(6):
            ax[i].plot(r_vals, s_vals[:, i] / 1e6)
            ax[i].set_xlabel(labels[i][1])
            ax[i].set_ylabel(labels[i][0])
            ax[i].set_title(labels[i][0].split("(")[0])
    else:
        raise ValueError("coord_system 必须是 'cyl' 或 'mat'")

    fig.tight_layout()
    return fig, axes
