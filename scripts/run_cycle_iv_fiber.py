import copy

import numpy as np
from tqdm import tqdm

from src.burst_pressure_prediction import (
    init_damagestate,
    init_layers,
    init_stiffness,
    setup_domain,
    solve_step,
    update_damage,
    update_stiffness,
)
from src.failure_criteria import hashin
from src.io_utils import load_config, make_result_dir, save_json
from src.paths import CONFIG_DIR, RESULT_DIR


def analyze_failure(
    C,
    r_i_list,
    theta,
    failure,
    p_o,
    p_i_lim,
    d_limit=0.9,
    nx=100,
    dp=1e6,
    dp_min=1e5,
    tol=1e-4,
    max_iter=50,
    debug=False,
):

    domain, ds = setup_domain(r_i_list, nx)

    layers = init_layers(r_i_list, C, theta, failure, nx)
    criteria = {
        "Xt": layers.XT,
        "Xc": layers.XC,
        "Yt": layers.YT,
        "Yc": layers.YC,
        "S12": layers.S12,
        "S13": layers.S13,
        "S23": layers.S23,
    }

    state = init_damagestate(layers)

    Q, C_field = init_stiffness(domain, C, theta, r_i_list)

    # 压力循环
    pbar = tqdm(total=p_i_lim / 1e6, desc="内压加载", unit="MPa")
    p_i = 0.0
    is_unloading = False
    history = []
    while p_i < p_i_lim:
        p_i += dp
        if is_unloading and np.isclose(p_i, 0.0):
            p_i = 0.0
        converged = False

        pbar.n = p_i / 1e6
        pbar.refresh()

        state_backup = copy.deepcopy(state)

        for it in range(max_iter):
            # 用当前损伤状态更新刚度场
            update_stiffness(state, layers, Q, C_field)

            # 求解当前压力下的应力和应变
            u_vals, sigma_cyl, _, sigma_mat, _ = solve_step(
                domain,
                ds,
                C_field,
                Q,
                p_i,
                p_o,
                r_i_list[0],
                r_i_list[-1],
                layers.angle,
            )

            # 计算 Hashin 失效触发
            hashin_trigger = hashin(sigma_mat, criteria)  # 输出 (4, nx) 0/1

            # 保存旧损伤用于收敛判断
            old_d = (
                state.d_ft.copy(),
                state.d_fc.copy(),
                state.d_mt.copy(),
                state.d_mc.copy(),
            )

            # 更新损伤
            state = update_damage(state, hashin_trigger, d_limit)

            if debug:
                tqdm.write(f"p={p_i / 1e6:.2f} MPa")
                tqdm.write(
                    f"  sigma_cyl max   = [{np.max(sigma_cyl[:, 0]) / 1e6:.2f}, "
                    f"{np.max(sigma_cyl[:, 1]) / 1e6:.2f}, {np.max(sigma_cyl[:, 2]) / 1e6:.2f}, "
                    f"{np.max(sigma_cyl[:, 3]) / 1e6:.2f}, {np.max(sigma_cyl[:, 4]) / 1e6:.2f}, "
                    f"{np.max(sigma_cyl[:, 5]) / 1e6:.2f}] MPa"
                )
                tqdm.write(
                    f"  sigma_mat [0]   = [{np.max(sigma_mat[0, 0]) / 1e6:.2f}, "
                    f"{sigma_mat[0, 1] / 1e6:.2f}, {sigma_mat[0, 2] / 1e6:.2f}, "
                    f"{sigma_mat[0, 3] / 1e6:.2f}, {sigma_mat[0, 4] / 1e6:.2f}, "
                    f"{sigma_mat[0, 5] / 1e6:.2f}] MPa"
                )
                tqdm.write(
                    f"  sigma_mat [-1]  = [{sigma_mat[-1, 0] / 1e6:.2f}, "
                    f"{sigma_mat[-1, 1] / 1e6:.2f}, {sigma_mat[-1, 2] / 1e6:.2f}, "
                    f"{sigma_mat[-1, 3] / 1e6:.2f}, {sigma_mat[-1, 4] / 1e6:.2f}, "
                    f"{sigma_mat[-1, 5] / 1e6:.2f}] MPa"
                )
                tqdm.write(
                    f"  Hashin trigger count = {np.sum(hashin_trigger, axis=1).tolist()}"
                )
                tqdm.write(
                    f"p={p_i / 1e6:.2f} MPa, d_mt[0] ={state.d_mt[0]:.3f}, "
                    f"d_mt[-1]={state.d_mt[-1]:.3f}, "
                    f"d_mt_max={np.max(state.d_mt):.3f}, d_mt_min={np.min(state.d_mt):.3f}, "
                )
                tqdm.write(
                    f"p={p_i / 1e6:.2f} MPa, d_ft[0] ={state.d_ft[0]:.3f}, "
                    f"d_ft[-1]={state.d_ft[-1]:.3f}, "
                    f"d_ft_max={np.max(state.d_ft):.3f}, d_ft_min={np.min(state.d_ft):.3f}"
                )
                tqdm.write(f"Num of d_ft >= d_limit: {np.sum(state.d_ft >= d_limit)}")

            # 计算最大损伤变化
            delta = max(
                np.max(np.abs(state.d_ft - old_d[0])),
                np.max(np.abs(state.d_fc - old_d[1])),
                np.max(np.abs(state.d_mt - old_d[2])),
                np.max(np.abs(state.d_mc - old_d[3])),
            )
            if delta < tol:
                converged = True
                """这里是不必要的"""
                # state_backup = copy.deepcopy(state)
                break

        if not converged:
            p_i -= dp  # 回退到上一步压力
            dp = np.sign(dp) * max(abs(dp) / 2, dp_min)
            state = copy.deepcopy(state_backup)
            if abs(dp) <= dp_min:
                if debug:
                    tqdm.write(f"压力 {p_i / 1e6:.2e} MPa处无法收敛，停止")
                pbar.close()
                break
            if debug:
                tqdm.write(f"未收敛，减小增量至 {dp / 1e6:.2e} MPa并重试")
            continue

        # 检查纤维贯通失效
        history.append(
            (
                p_i,
                u_vals[0],
                int(np.sum(state.d_ft >= d_limit)),
                state.d_ft.copy(),
                state.d_fc.copy(),
            )
        )

        if not is_unloading:
            # 检查是否出现了新损伤（纤维或基体损伤总量增加）
            n_damaged_prev = np.sum(state_backup.d_f > 0)
            n_damaged_curr = np.sum(state.d_f > 0)

            if n_damaged_curr > n_damaged_prev:
                is_unloading = True
                dp = -dp  # 翻转步长，开始卸载
                continue

        # --- 卸载完成，准备重新加载 ---
        if is_unloading and p_i == 0.0:
            is_unloading = False
            dp = -dp  # 翻转为正，重新加载
            continue

    pbar.close()
    return _, history


def find_knee(history):
    """p-u 拐点：只在纤维失效单元数增加的步⾥，找增量柔度 du/dp 跳变最⼤的⼀步"""
    p = np.array([h[0] for h in history])
    u = np.array([h[1] for h in history])
    n = np.array([h[2] for h in history])

    dp = np.diff(p)
    du = np.diff(u)
    dn = np.diff(n)

    # 只用加压段，卸载段设为 nan 自动排除
    c = np.full_like(dp, np.nan)
    loading = dp > 0
    c[loading] = du[loading] / dp[loading]  # c[k]: p[k] -> p[k+1] 的增量柔度

    jump = np.full_like(c, -np.inf)
    jump[1:] = c[1:] / c[:-1]

    # 排除没有新损伤的步
    jump[dn <= 0] = -np.inf  # 排除基体开裂等引起的柔度变化

    k = int(np.argmax(jump))
    if not np.isfinite(jump[k]):
        return None, None
    return p[k + 1], jump[k]  # 损伤雪崩发⽣的压⼒（精度 ±dp）及柔度跳变倍数


def predict_burst_pressure(config_path, d_limit=0.9, p_i_lim=300.0e6, **overrides):
    """
    运行单次求解，返回数值结果。
    config_path: YAML 路径
    overrides: 覆盖材料失效参数，如 S12=1.0e8
    """
    C_list, r_i_list, theta_list, failure, name, slug = load_config(config_path)

    # 覆盖 failure 参数
    for f in failure:
        for k, v in overrides.items():
            if k in f:
                f[k] = v

    n_layers = len(r_i_list) - 1
    nx = max(n_layers * 3, 100)

    _, history = analyze_failure(
        C_list,
        r_i_list,
        theta_list,
        failure,
        p_o=0.0,
        p_i_lim=p_i_lim,
        d_limit=d_limit,
        nx=nx,
        dp=5e6,
        debug=False,
    )

    p_burst, ratio = find_knee(history)

    return {
        "name": name,
        "slug": slug,
        "p_burst_MPa": float(p_burst) / 1e6 if p_burst is not None else None,
        "ratio": float(ratio) if ratio is not None else None,
        "overrides": dict(overrides),
        "history": history,
    }


def main():
    path = CONFIG_DIR / "defaultIV.yaml"
    slug = path.stem + "_cycle_fiber"
    out_dir = make_result_dir(RESULT_DIR, slug, timestamp=False)

    r = predict_burst_pressure(path, d_limit=0.95)
    history = r["history"]

    # 保存标量结果用 JSON
    save_json(
        out_dir,
        [{"p_burst_MPa": r["p_burst_MPa"], "ratio": r["ratio"]}],
        filename="defaultIV_solution.json",
    )

    # 保存数组历史用 npz (必须用 numpy 专属格式)
    np.savez(
        out_dir / "history.npz",
        p=np.array([h[0] for h in history]),
        u=np.array([h[1] for h in history]),
        n=np.array([h[2] for h in history]),
        d_ft=np.array([h[3] for h in history]),
        d_fc=np.array([h[4] for h in history]),
    )
    print(f"  saved to: {out_dir}")


if __name__ == "__main__":
    main()
