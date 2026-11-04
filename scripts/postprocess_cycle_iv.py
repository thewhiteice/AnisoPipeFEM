import json

import matplotlib.pyplot as plt
import numpy as np

from src.paths import RESULT_DIR
from src.plot_results import plot_damage_evolution

# 1. 统一指向数据所在的文件夹
out_dir = RESULT_DIR / "defaultIV_cycle"

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

# 4. 画损伤演化图
fig2, axes2 = plot_damage_evolution(
    d_ft_history=d_ft, d_fc_history=d_fc, p_history=p_hist
)
# 保存到同一个 out_dir
fig2.savefig(out_dir / "damage_evolution.png", dpi=300, bbox_inches="tight")
