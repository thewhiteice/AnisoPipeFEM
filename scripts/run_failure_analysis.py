import copy
from dataclasses import dataclass

import basix
import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities_boundary, meshtags
from mpi4py import MPI
from tqdm import tqdm

from config.config import setup_config1, setup_config2, setup_config3
from src.failure_criteria import hashin
from src.material_utils import (
    bond_transform,
    build_stiffness,
    cyl2mat_strain_v,
    cyl2mat_stress_v,
    stiffness_to_properties,
)


@dataclass
class Layers:
    """所有字段都是长度 nx 的数组，每个单元一个值"""

    # 几何
    h: np.ndarray  # 单元厚度
    angle: np.ndarray  # 缠绕角 (rad)
    # 原始工程常数
    E1: np.ndarray
    E2: np.ndarray
    E3: np.ndarray
    G12: np.ndarray
    G13: np.ndarray
    G23: np.ndarray
    nu12: np.ndarray
    nu13: np.ndarray
    nu23: np.ndarray
    # 强度
    XT: np.ndarray  # 纤维拉伸强度
    XC: np.ndarray  # 纤维压缩强度
    YT: np.ndarray  # 基体拉伸强度
    YC: np.ndarray  # 基体压缩强度
    S12: np.ndarray  # 面内剪切强度
    S13: np.ndarray
    S23: np.ndarray
    # 损伤起始应变（材料常数，按单元广播）
    eps0_ft: np.ndarray
    eps0_fc: np.ndarray
    eps0_mt: np.ndarray
    eps0_mc: np.ndarray
    # 断裂能
    Gft: np.ndarray  # 纤维拉伸断裂能
    Gfc: np.ndarray  # 纤维压缩断裂能
    Gmt: np.ndarray  # 基体拉伸断裂能
    Gmc: np.ndarray  # 基体压缩断裂能


@dataclass
class Damagestate:
    """所有字段都是长度 nx 的数组，存损伤变量和折减后模量"""

    # 损伤变量
    d_ft: np.ndarray
    d_fc: np.ndarray
    d_mt: np.ndarray
    d_mc: np.ndarray
    # 折减后的工程常数
    E1: np.ndarray
    E2: np.ndarray
    E3: np.ndarray
    G12: np.ndarray
    G13: np.ndarray
    G23: np.ndarray
    nu12: np.ndarray
    nu13: np.ndarray
    nu23: np.ndarray


def setup_domain(r_i_list: np.ndarray, nx: int = 100):
    lengths = np.diff(r_i_list)
    nx_per_layer = (nx * lengths / lengths.sum()).astype(int)
    nx_per_layer[-1] += nx - nx_per_layer.sum()  # 修正整数舍入

    # 逐层生成节点，去除重复界面点
    pts_list = []
    for i in range(len(r_i_list) - 1):
        r0, r1 = r_i_list[i], r_i_list[i + 1]
        n = nx_per_layer[i]
        layer_pts = np.linspace(r0, r1, n + 1)
        if i > 0:
            layer_pts = layer_pts[1:]  # 去掉与上一层重复的界面节点
        pts_list.append(layer_pts)
    points = np.hstack(pts_list)

    # 构造拓扑
    cells = np.array([[i, i + 1] for i in range(len(points) - 1)], dtype=np.int64)
    e = basix.ufl.element("Lagrange", "interval", 1, shape=(1,))
    geometry = points.reshape(-1, 1)

    # 构造网格
    domain = mesh.create_mesh(MPI.COMM_WORLD, cells, e, geometry)

    # 边界标记与积分
    fdim = domain.topology.dim - 1
    left = locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[0], r_i_list[0])
    )
    right = locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[0], r_i_list[-1])
    )
    indices = np.hstack([left, right])
    values = np.hstack([np.full_like(left, 1), np.full_like(right, 2)])
    sorted_order = np.argsort(indices)
    mt = meshtags(domain, fdim, indices[sorted_order], values[sorted_order])
    ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)

    return domain, ds


def init_layers(r_interface_list, C_basic_list, theta_rad_list, failure_list, nx=100):
    # ---------- 计算每个单元所属层索引和单元厚度 ----------
    lengths = np.diff(r_interface_list)
    nx_per_layer = (nx * lengths / lengths.sum()).astype(int)
    nx_per_layer[-1] += nx - nx_per_layer.sum()  # 修正舍入

    # 生成所有节点坐标（确保层界面为节点）
    pts_list = []
    for i in range(len(r_interface_list) - 1):
        r0, r1 = r_interface_list[i], r_interface_list[i + 1]
        n = nx_per_layer[i]
        layer_pts = np.linspace(r0, r1, n + 1)
        if i > 0:
            layer_pts = layer_pts[1:]  # 去掉与上一层重复的界面节点
        pts_list.append(layer_pts)
    points = np.hstack(pts_list)

    h_cell = np.diff(points)  # 每个单元的厚度
    r_vals = (points[:-1] + points[1:]) / 2  # 单元中心半径

    # 判断每个单元属于哪一层
    layer_idx = np.zeros(nx, dtype=int)
    for i in range(len(r_interface_list) - 1):
        r_in, r_out = r_interface_list[i], r_interface_list[i + 1]
        if i < len(r_interface_list) - 2:
            mask = (r_vals >= r_in) & (r_vals < r_out)
        else:
            mask = (r_vals >= r_in) & (r_vals <= r_out)
        layer_idx[mask] = i

    # ---------- 提取每层的工程常数（从刚度矩阵） ----------
    n_layers = len(C_basic_list)
    E1_l = np.zeros(n_layers)
    E2_l = np.zeros(n_layers)
    E3_l = np.zeros(n_layers)
    G12_l = np.zeros(n_layers)
    G13_l = np.zeros(n_layers)
    G23_l = np.zeros(n_layers)
    nu12_l = np.zeros(n_layers)
    nu13_l = np.zeros(n_layers)
    nu23_l = np.zeros(n_layers)
    for i, C in enumerate(C_basic_list):
        E_list, nu_list, G_list = stiffness_to_properties(C)
        E1_l[i], E2_l[i], E3_l[i] = E_list
        nu12_l[i], nu13_l[i], nu23_l[i] = nu_list
        G12_l[i], G13_l[i], G23_l[i] = G_list

    # ---------- 强度与断裂能提取 ----------
    XT_l = np.array([f["Xt"] for f in failure_list])
    XC_l = np.array([f["Xc"] for f in failure_list])
    YT_l = np.array([f["Yt"] for f in failure_list])
    YC_l = np.array([f["Yc"] for f in failure_list])
    S12_l = np.array([f["S12"] for f in failure_list])
    S13_l = np.array([f["S13"] for f in failure_list])
    S23_l = np.array([f["S23"] for f in failure_list])
    Gft_l = np.array([f.get("Gft", 90e3) for f in failure_list])  # 默认值
    Gfc_l = np.array([f.get("Gfc", 50e3) for f in failure_list])
    Gmt_l = np.array([f.get("Gmt", 0.3e3) for f in failure_list])
    Gmc_l = np.array([f.get("Gmc", 0.8e3) for f in failure_list])

    # ---------- 计算每层损伤起始应变 ----------
    eps0_ft_l = XT_l / E1_l
    eps0_fc_l = XC_l / E1_l
    eps0_mt_l = YT_l / E2_l
    eps0_mc_l = YC_l / E2_l

    # ---------- 广播到每个单元 ----------
    idx = layer_idx  # 形状 (nx,)
    return Layers(
        h=h_cell,
        angle=np.array(theta_rad_list)[idx],
        E1=E1_l[idx],
        E2=E2_l[idx],
        E3=E3_l[idx],
        G12=G12_l[idx],
        G13=G13_l[idx],
        G23=G23_l[idx],
        nu12=nu12_l[idx],
        nu13=nu13_l[idx],
        nu23=nu23_l[idx],
        XT=XT_l[idx],
        XC=XC_l[idx],
        YT=YT_l[idx],
        YC=YC_l[idx],
        eps0_ft=eps0_ft_l[idx],
        eps0_fc=eps0_fc_l[idx],
        eps0_mt=eps0_mt_l[idx],
        eps0_mc=eps0_mc_l[idx],
        S12=S12_l[idx],
        S13=S13_l[idx],
        S23=S23_l[idx],
        Gft=Gft_l[idx],
        Gfc=Gfc_l[idx],
        Gmt=Gmt_l[idx],
        Gmc=Gmc_l[idx],
    )


def init_damagestate(layers: Layers) -> Damagestate:
    nx = len(layers.h)
    return Damagestate(
        d_ft=np.zeros(nx),
        d_fc=np.zeros(nx),
        d_mt=np.zeros(nx),
        d_mc=np.zeros(nx),
        E1=layers.E1.copy(),
        E2=layers.E2.copy(),
        E3=layers.E3.copy(),
        G12=layers.G12.copy(),
        G13=layers.G13.copy(),
        G23=layers.G23.copy(),
        nu12=layers.nu12.copy(),
        nu13=layers.nu13.copy(),
        nu23=layers.nu23.copy(),
    )


def init_stiffness(domain, C0_list, theta_list, r_i_list):
    # ===== 定义材料参数 ======
    Q_list, C_list = [], []
    for C_basic, theta in zip(C0_list, theta_list):
        C_mat = bond_transform(C_basic, theta)
        Q_full = C_mat - np.outer(C_mat[:, 4], C_mat[:, 4]) / C_mat[4, 4]
        Q_full = np.ascontiguousarray(
            Q_full[:3, :3], dtype=np.float64
        )  # 强制转换为 C 连续
        Q_list.append(Q_full)
        C_list.append(C_mat)

    V_Q = fem.functionspace(domain, ("DG", 0, (3, 3)))
    V_C = fem.functionspace(domain, ("DG", 0, (6, 6)))

    Q = fem.Function(V_Q)
    C = fem.Function(V_C)

    def _piecewise(r_list, mats):
        assert len(r_list) == len(mats) + 1, (
            f"r_list 应有 {len(mats) + 1} 个元素，实际 {len(r_list)} 个"
        )

        def evaluator(x):
            r = x[0]
            out = np.zeros((mats[0].size, r.shape[0]))
            for i, mat in enumerate(mats):
                r_in, r_out = r_list[i], r_list[i + 1]
                mask = (
                    (r >= r_in) & (r < r_out)
                    if i < len(mats) - 1
                    else (r >= r_in) & (r <= r_out)
                )
                out[:, mask] = mat.flatten()[:, None]
            return out

        return evaluator

    Q.interpolate(_piecewise(r_i_list, Q_list))
    C.interpolate(_piecewise(r_i_list, C_list))

    return Q, C


def solve_case0(domain, ds, r, u, v, Q, p_i, p_o, r_i, r_o):
    # 求解平面应变状态, ε_z=0
    eps_r = ufl.grad(u)[0]
    eps_theta = u / r
    eps_vec_0 = ufl.as_vector([eps_r, eps_theta, 0.0])

    delta_eps_r = ufl.grad(v)[0]
    delta_eps_theta = v / r

    sigma_vec_0 = ufl.dot(Q, eps_vec_0)
    sigma_r_0 = sigma_vec_0[0]
    sigma_theta_0 = sigma_vec_0[1]
    sigma_z_0 = sigma_vec_0[2]

    LHS_0 = (2 * ufl.pi * r * sigma_r_0 * delta_eps_r) * ufl.dx + (
        2 * ufl.pi * r * sigma_theta_0 * delta_eps_theta
    ) * ufl.dx

    # 右边：外力虚功（仅内外压，去掉轴向力项）
    p_a = fem.Constant(domain, p_i)  # 内壁压强(Pa)
    p_b = fem.Constant(domain, p_o)  # 外壁压强(Pa)
    F_total = np.pi * (r_i**2 * p_a.value - r_o**2 * p_b.value)  # 总轴向拉力(N)

    RHS_0 = 2 * ufl.pi * r_i * p_a * v * ds(1) - 2 * ufl.pi * r_o * p_b * v * ds(2)

    # ===========================
    # Case 0 求解平面应变位移场
    problem_0 = LinearProblem(
        LHS_0,
        RHS_0,
        bcs=[],
        petsc_options_prefix="solve_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )

    return problem_0, sigma_z_0, F_total


def solve_case1(r, u, v, Q):
    # 求解纯轴向拉力状态, ε_z=1
    eps_r = ufl.grad(u)[0]
    eps_theta = u / r
    eps_z_1 = 1.0
    eps_vec_1 = ufl.as_vector([eps_r, eps_theta, eps_z_1])

    delta_eps_r = ufl.grad(v)[0]
    delta_eps_theta = v / r

    sigma_vec_1 = ufl.dot(Q, eps_vec_1)
    sigma_r_1 = sigma_vec_1[0]
    sigma_theta_1 = sigma_vec_1[1]
    sigma_z_1 = sigma_vec_1[2]

    # ===========================
    # Case 1 定义弱形式（仅径向平衡，去掉轴向约束项）
    LHS_1 = (2 * ufl.pi * r * sigma_r_1 * delta_eps_r) * ufl.dx + (
        2 * ufl.pi * r * sigma_theta_1 * delta_eps_theta
    ) * ufl.dx

    RHS_1 = (
        -(2 * ufl.pi * r * (Q[0, 2] * eps_z_1) * delta_eps_r) * ufl.dx
        - (2 * ufl.pi * r * (Q[1, 2] * eps_z_1) * delta_eps_theta) * ufl.dx
    )

    # ===========================
    # Case 1 求解轴向拉伸位移场
    problem_1 = LinearProblem(
        LHS_1,
        RHS_1,
        bcs=[],
        petsc_options_prefix="solve_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )

    return problem_1, sigma_z_1


def solve_step(domain, ds, C, Q, p_i, p_o, r_i, r_o, angles):
    # 径向坐标
    r = ufl.SpatialCoordinate(domain)[0]

    # 定义试函数、测试函数空间
    V_u = fem.functionspace(domain, ("Lagrange", 2))
    u = ufl.TrialFunction(V_u)
    v = ufl.TestFunction(V_u)

    # 平面应变问题
    prob0, s_z_0, F_tol = solve_case0(domain, ds, r, u, v, Q, p_i, p_o, r_i, r_o)
    u_0_sol = prob0.solve()
    s_z_0_sol = ufl.replace(s_z_0, {u: u_0_sol})
    F0_form = 2 * ufl.pi * s_z_0_sol * r * ufl.dx
    F0 = fem.assemble_scalar(fem.form(F0_form))

    # 径向拉伸问题
    prob1, s_z_1 = solve_case1(r, u, v, Q)
    u_1_sol = prob1.solve()
    s_z_1_sol = ufl.replace(s_z_1, {u: u_1_sol})
    F1_form = 2 * ufl.pi * s_z_1_sol * r * ufl.dx
    F1 = fem.assemble_scalar(fem.form(F1_form))

    eps_z_sol = (F_tol - F0) / F1

    # 叠加位移场并合成总应力
    u_total = u_0_sol + eps_z_sol * u_1_sol
    eps_r_sol = ufl.grad(u_total)[0]
    eps_theta_sol = u_total / r

    gamma_rz_sol = (
        -(C[0, 4] * eps_r_sol + C[1, 4] * eps_theta_sol + C[2, 4] * eps_z_sol) / C[4, 4]
    )

    eps_vec_total = ufl.as_vector(
        [
            eps_r_sol,  # 索引0: ε_r
            eps_theta_sol,  # 索引1: ε_θ
            eps_z_sol,  # 索引2: ε_z
            0.0,  # 索引3: 0（对应 γ_θz 或 2ε_θz）
            gamma_rz_sol,  # 索引4: γ_rz（非零）
            0.0,  # 索引5: 0（对应 γ_rθ 或 2ε_rθ）
        ]
    )

    sigma_vec_total = ufl.dot(C, eps_vec_total)

    V_dg0 = fem.functionspace(domain, ("DG", 0))
    ip = V_dg0.element.interpolation_points

    u_expr = fem.Expression(u_total, ip)
    u_sol = fem.Function(V_dg0)
    u_sol.interpolate(u_expr)

    # ===========================
    # 插值应力到 DG0
    V_dg0_vec = fem.functionspace(domain, ("DG", 0, (6,)))
    ip_vec = V_dg0.element.interpolation_points

    sigma_vec_expr = fem.Expression(sigma_vec_total, ip_vec)
    sigma_vec_sol = fem.Function(V_dg0_vec)
    sigma_vec_sol.interpolate(sigma_vec_expr)

    # ===========================
    # 插值应变到 DG0（新增，损伤需要）
    eps_vec_expr = fem.Expression(eps_vec_total, ip_vec)
    eps_vec_sol = fem.Function(V_dg0_vec)
    eps_vec_sol.interpolate(eps_vec_expr)

    # ===========================
    # 提取为数组
    u_vals = u_sol.x.array  # (nx,)
    sigma_cyl = sigma_vec_sol.x.array.reshape(-1, 6)  # (nx, 6)
    eps_cyl = eps_vec_sol.x.array.reshape(-1, 6)  # (nx, 6)

    # ===========================
    # 转换到材料坐标系
    sigma_mat = cyl2mat_stress_v(sigma_cyl, angles)  # 材料坐标系应力
    eps_mat = cyl2mat_strain_v(eps_cyl, angles)  # 材料坐标系应变

    return u_vals, sigma_cyl, eps_cyl, sigma_mat, eps_mat


def update_damage(state, layers, hashin_trigger, eps_mat):
    """
    state: Damagestate, 原地更新
    layers: Layers, 原始材料参数
    hashin_trigger: (4, nx) 0/1 数组，顺序 [ft, fc, mt, mc]
    eps_mat: (nx, 6) 材料坐标系应变，顺序 [ε1, ε2, ε3, ε23, ε13, ε12]
    """

    """    
    eps1 = eps_mat[:, 0]
    eps2 = eps_mat[:, 1]

    eq_ft = np.maximum(eps1, 0.0)
    eq_fc = np.abs(np.minimum(eps1, 0.0))
    eq_mt = np.maximum(eps2, 0.0)
    eq_mc = np.abs(np.minimum(eps2, 0.0))

    # 固定损伤起始应变
    eps0_ft = layers.eps0_ft
    eps0_fc = layers.eps0_fc
    eps0_mt = layers.eps0_mt
    eps0_mc = layers.eps0_mc

    epsf_ft = 2.0 * layers.Gft / (layers.XT * layers.h)
    epsf_fc = 2.0 * layers.Gfc / (layers.XC * layers.h)
    epsf_mt = 2.0 * layers.Gmt / (layers.YT * layers.h)
    epsf_mc = 2.0 * layers.Gmc / (layers.YC * layers.h)

    active_ft = (state.d_ft > 0) | (hashin_trigger[0] == 1)
    active_fc = (state.d_fc > 0) | (hashin_trigger[1] == 1)
    active_mt = (state.d_mt > 0) | (hashin_trigger[2] == 1)
    active_mc = (state.d_mc > 0) | (hashin_trigger[3] == 1)

    d_new_ft = (eq_ft - eps0_ft) / (epsf_ft - eps0_ft + 1e-30)
    d_new_fc = (eq_fc - eps0_fc) / (epsf_fc - eps0_fc + 1e-30)
    d_new_mt = (eq_mt - eps0_mt) / (epsf_mt - eps0_mt + 1e-30)
    d_new_mc = (eq_mc - eps0_mc) / (epsf_mc - eps0_mc + 1e-30)

    state.d_ft = np.minimum(
        0.999, np.maximum(state.d_ft, np.where(active_ft, d_new_ft, 0.0))
    )
    state.d_fc = np.minimum(
        0.999, np.maximum(state.d_fc, np.where(active_fc, d_new_fc, 0.0))
    )
    state.d_mt = np.minimum(
        0.999, np.maximum(state.d_mt, np.where(active_mt, d_new_mt, 0.0))
    )
    state.d_mc = np.minimum(
        0.999, np.maximum(state.d_mc, np.where(active_mc, d_new_mc, 0.0))
    )
    """

    # 纤维拉伸触发
    state.d_ft = np.where(hashin_trigger[0] == 1, 0.90, state.d_ft)
    # 纤维压缩触发
    state.d_fc = np.where(hashin_trigger[1] == 1, 0.90, state.d_fc)
    # 基体拉伸触发
    state.d_mt = np.where(hashin_trigger[2] == 1, 0.90, state.d_mt)
    # 基体压缩触发
    state.d_mc = np.where(hashin_trigger[3] == 1, 0.90, state.d_mc)

    d_f = np.maximum(state.d_ft, state.d_fc)
    d_m = np.maximum(state.d_mt, state.d_mc)
    d_s = np.maximum(d_f, d_m)

    state.E1 = layers.E1 * (1.0 - d_f)
    state.E2 = layers.E2 * (1.0 - d_m)
    state.E3 = layers.E3 * (1.0 - d_m)
    state.G12 = layers.G12 * (1.0 - d_s)
    state.G13 = layers.G13 * (1.0 - d_s)
    state.G23 = layers.G23 * (1.0 - d_s)

    # 损伤不可逆已经由 np.maximum 保证
    return state


def update_stiffness(state, layers, Q, C):
    """根据 state 中折减后的工程常数，更新 Q 和 C 两个 DG0 函数"""
    Q_arr = np.zeros((len(layers.h), 3, 3))
    C_arr = np.zeros((len(layers.h), 6, 6))

    for i in range(len(layers.h)):
        E_list = [state.E1[i], state.E2[i], state.E3[i]]
        nu_list = [state.nu12[i], state.nu13[i], state.nu23[i]]
        G_list = [state.G12[i], state.G13[i], state.G23[i]]
        C_local = build_stiffness(E_list, nu_list, G_list)

        C_global = bond_transform(C_local, layers.angle[i])
        C_arr[i] = C_global
        Q_arr[i] = (
            C_global[:3, :3]
            - np.outer(C_global[:3, 4], C_global[:3, 4]) / C_global[4, 4]
        )

    Q_arr = np.ascontiguousarray(Q_arr, dtype=np.float64)
    C_arr = np.ascontiguousarray(C_arr, dtype=np.float64)

    Q.x.array[:] = Q_arr.flatten()
    C.x.array[:] = C_arr.flatten()
    # 不需要return, 已执行原地更新


def analyze_failure(
    C,
    r_i_list,
    theta,
    failure,
    p_o,
    p_i_lim,
    nx=100,
    dp=1e6,
    dp_min=1e5,
    tol=1e-4,
    max_iter=50,
):
    # 如果 failure 中没有断裂能，补充默认值
    failure.setdefault("Gft", 60e3)  # J/m^2 133e3
    failure.setdefault("Gfc", 40e3)  # 8e3
    failure.setdefault("Gmt", 0.6e3)
    failure.setdefault("Gmc", 2.1e3)

    domain, ds = setup_domain(r_i_list)

    layers = init_layers(r_i_list, [C], [theta], [failure], nx)

    state = init_damagestate(layers)

    Q, C_field = init_stiffness(domain, [C], [theta], r_i_list)

    # 压力循环
    pbar = tqdm(total=p_i_lim / 1e6, desc="内压加载", unit="MPa")
    p_i = 0.0
    while p_i < p_i_lim:
        p_i += dp
        converged = False

        pbar.n = p_i / 1e6
        pbar.refresh()

        state_backup = copy.deepcopy(state)

        for it in range(max_iter):
            # 用当前损伤状态更新刚度场
            update_stiffness(state, layers, Q, C_field)

            # 求解当前压力下的应力和应变
            u_vals, sigma_cyl, eps_cyl, sigma_mat, eps_mat = solve_step(
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
            hashin_trigger = hashin(
                sigma_mat, failure
            )  # 现在应输出 (4, nx) 0/1, 原来是nx, 4

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
            tqdm.write(f"  Hashin trigger count = {np.sum(hashin_trigger, axis=1).tolist()}")

            # 保存旧损伤用于收敛判断
            old_d = (
                state.d_ft.copy(),
                state.d_fc.copy(),
                state.d_mt.copy(),
                state.d_mc.copy(),
            )

            # 更新损伤
            state = update_damage(state, layers, hashin_trigger, eps_mat)

            tqdm.write(
                f"p={p_i / 1e6:.2f} MPa, d_mt[0] ={state.d_mt[0]:.3f}, "
                f"d_mt[-1]={state.d_mt[0]:.3f}, "
                f"d_mt_max={np.max(state.d_mt):.3f}, d_mt_min={np.min(state.d_mt):.3f}, "
            )

            tqdm.write(
                f"p={p_i / 1e6:.2f} MPa, d_ft[0] ={state.d_ft[0]:.3f}, "
                f"d_ft[-1]={state.d_ft[-1]:.3f}, "
                f"d_ft_max={np.max(state.d_ft):.3f}, d_ft_min={np.min(state.d_ft):.3f}"
            )

            # 计算最大损伤变化
            delta = max(
                np.max(np.abs(state.d_ft - old_d[0])),
                np.max(np.abs(state.d_fc - old_d[1])),
                np.max(np.abs(state.d_mt - old_d[2])),
                np.max(np.abs(state.d_mc - old_d[3])),
            )
            if delta < tol:
                converged = True
                state_backup = copy.deepcopy(state)
                break

        if not converged:
            p_i -= dp  # 回退到上一步压力
            dp = max(dp / 2, dp_min)
            state = copy.deepcopy(state_backup)
            if dp <= dp_min:
                print(f"压力 {p_i / 1e6:.2e} MPa处无法收敛，停止")
                pbar.close()
                break
            print(f"未收敛，减小增量至 {dp / 1e6:.2e} MPa并重试")
            continue

        # 检查纤维贯通失效
        d_fiber = np.maximum(state.d_ft, state.d_fc)
        d_matrix = np.maximum(state.d_mt, state.d_mc)
        if np.all(state.d_ft >= 0.90):  # 0.99 or np.all(d_matrix >= 0.99)
            tqdm.write(f"[DEBUG] p_i={p_i/1e6:.2f} MPa, d_fiber_max={np.max(d_fiber):.4f}, "
                       f"d_fiber_min={np.min(d_fiber):.4f}")
            print(f"爆破压力 = {p_i / 1e6:.3e} MPa")
            pbar.close()
            return p_i

    pbar.close()
    print("达到压力上限，未发生爆破")
    return p_i


def main():
    C, _, r_i_list, theta, failure, name = setup_config2()
    E_list, nu_list, G_list = stiffness_to_properties(C)
    assert np.allclose(C, build_stiffness(E_list, nu_list, G_list)), (
        "计算刚度矩阵不可逆"
    )

    p_o = 0.0
    p_i_lim = 300.0e6
    # theta = np.deg2rad(0)

    p_failure = analyze_failure(C, r_i_list, theta, failure, p_o, p_i_lim, dp=5e6)
    print(f"p_failure = {p_failure / 1e6:.2f}Mpa")


if __name__ == "__main__":
    main()
