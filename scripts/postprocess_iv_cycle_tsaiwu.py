import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.paths import RESULT_DIR

# 1. 统一指向数据所在的文件夹
out_dir = RESULT_DIR / "defaultIV_cycle_tsai_wu"

# 2. 读取数据
data = np.load(out_dir / "history.npz")
p_hist = data["p"]
u_hist = data["u"]
d_ft = data["d_ft"]
d_fc = data["d_fc"]

# 读取之前保存的 JSON 结果
with open(out_dir / "defaultIV_solution.json", "r", encoding="utf-8") as f:
    sol_data = json.load(f)
    p_burst = sol_data[0]["p_burst_MPa"]

# 3. 画 P-U (压力-位移) 曲线
fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(u_hist * 1e3, p_hist / 1e6, "b-")
ax1.axhline(y=p_burst, color="r", linestyle="--", label=f"Burst: {p_burst:.1f} MPa")
ax1.set_xlabel("Radial Displacement (mm)")
ax1.set_ylabel("Pressure (MPa)")
ax1.set_title("Pressure-Displacement Curve")
ax1.grid(True)
# 保存到同一个 out_dir
fig1.savefig(out_dir / "p_u_curve.png", dpi=300, bbox_inches="tight")


def plot_damage_evolution(d_ft_history, d_fc_history=None, p_history=None):
    fig, axes = plt.subplots(figsize=(8, 8), sharex=True)
    cmap = ListedColormap(["#F2F2F2", "#D62728"])  # 浅灰 -> 深红
    # cmap = ListedColormap(["black", "red"])

    for ax, data, title in zip(
        [axes],
        [d_ft_history],
        ["Fiber Tension (d_ft)"],
    ):
        mask = (data > 0).astype(int)
        # y 轴直接用步数索引，不强制映射实际压力值，避免不等距失真
        ax.imshow(mask, aspect="auto", cmap=cmap, origin="lower")
        # ax.pcolormesh(mask, cmap=cmap, edgecolors='gray', linewidth=0.1)
        ax.set_ylabel("Step Index")
        ax.set_title(title)

        # 使用图例（Legend）而非 Colorbar
        legend_elements = [
            Patch(facecolor="#D62728", label="Damaged"),
            Patch(facecolor="#F2F2F2", label="No Damage"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

    axes.set_xlabel("Element Index (Inner -> Outer)")
    fig.tight_layout()
    return fig, axes


# 4. 画损伤演化图
fig2, axes2 = plot_damage_evolution(
    d_ft_history=d_ft, d_fc_history=d_fc, p_history=p_hist
)
# 保存到同一个 out_dir
fig2.savefig(out_dir / "damage_evolution.png", dpi=300, bbox_inches="tight")
