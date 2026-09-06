from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.stress_solver import (
    solve_aniso_pipe,
    solve_iso_pipe,
    solve_lame_stress,
    solver_aniso_layered_pipe,
)


def main():
    r_in = 0.1
    r_out = 0.2
    E = 210.09
    nu = 0.3
    p_out = 0.1e6  # 外壁压强
    p_in = 70.0e6  # 内壁压强

    r_vals, _, s_vals_list = solve_iso_pipe(r_in, r_out, E, nu, p_in, p_out)
    s_r_iso = s_vals_list[0][0][0]

    s_lame_vals = solve_lame_stress(r_vals, r_in, r_out, p_in, p_out)
    s_r_lame = s_lame_vals[0]

    # ===== 可视化并保存结果
    script_dir = Path(__file__).parent.parent / "results" / "stress_comparetion_iso_pipe"
    script_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 7))
    plt.plot(r_vals, s_r_lame / 1.0e6, label="lame")
    plt.plot(r_vals, s_r_iso / 1.0e6, "--", label="iso FEM")
    plt.legend()
    plt.grid()
    plt.xlabel("r (m)")
    plt.ylabel(r"$\sigma_r$ (Mpa)")
    plt.savefig(script_dir / "stress_iso_pipe.png", dpi=150)

    plt.figure(figsize=(7, 7))
    plt.plot(r_vals, (s_r_iso - s_r_lame) / 1.0e6, label="iso FEM")
    plt.legend()
    plt.grid()
    plt.xlabel("r (m)")
    plt.ylabel(r"$\sigma_r$ (Mpa)")
    plt.savefig(script_dir / "stress_diff_iso_pipe.png", dpi=150)


if __name__ == "__main__":
    main()
