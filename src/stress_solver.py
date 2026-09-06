"""
单一工况应力求解文件
    solve_iso_pipe()            单层各向同性圆管
    solve_aniso_pipe()          单层正交各向异性圆管
    solve_aniso_layered_pipe()  多层正交各向异性圆管
    solve_lame_stress()         厚壁均匀圆管闭口状态
"""

import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem, assemble_matrix, assemble_vector
from dolfinx.mesh import locate_entities_boundary, meshtags
from mpi4py import MPI
from petsc4py import PETSc
from tqdm import tqdm


def solve_iso_pipe(r_in, r_out, E, nu, p_in_list, p_out_list, nx=100):
    """
    各向同性圆管径向应力分布求解器

    求解径向一维广义平面应变问题的应力分量分布
    已与 lame 解析解结果成功对照

    输入:
        r_in:   内壁半径 m
        r_out:  外壁半径 m
        E:      各向同性杨氏模量 Pa
        nu:     各向同性泊松比
        p_in:   内壁压强 Pa
        p_out:  外壁压强 Pa
        nx:     径向划分单元数 int 默认100
    输出:
        r_vals: 径向插值坐标 (nx,)
        u_vals: 插值径向位移 (nx,)
        sigma_vec_vals: 柱坐标系下插值应力分量 (nx,3)
            - 索引0: sigma_r
            - 索引1: sigma_θ
            - 索引2: sigma_z
    """
    # 修改类型
    p_in_list = np.atleast_1d(p_in_list)
    p_out_list = np.atleast_1d(p_out_list)

    # ===== 定义几何与网格 ======
    domain = mesh.create_interval(MPI.COMM_WORLD, nx, [r_in, r_out])
    r = ufl.SpatialCoordinate(domain)[0]  # 径向坐标

    # ===== 定义函数空间 (径向位移场) =====
    V_u = fem.functionspace(domain, ("Lagrange", 2))
    u = ufl.TrialFunction(V_u)
    v = ufl.TestFunction(V_u)

    # ===== 定义材料参数 =====
    E = fem.Constant(domain, E)  # 弹性模量(Pa)
    nu = fem.Constant(domain, nu)  # 泊松比
    G = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

    # ===== 定义应力与应变 (平面应变状态, ε_z=0)
    eps_r = ufl.grad(u)[0]
    eps_theta = u / r
    # 平面应变下的应力（ε_z=0）
    sigma_r_0 = lmbda * (eps_r + eps_theta) + 2 * G * eps_r
    sigma_theta_0 = lmbda * (eps_r + eps_theta) + 2 * G * eps_theta
    sigma_z_0 = lmbda * (eps_r + eps_theta) + 2 * G * 0  # ε_z=0

    # ===== 定义弱形式（仅径向平衡，去掉轴向约束项）=====
    # 左边：内力虚功（平面应变）
    LHS = (2 * ufl.pi * r * sigma_r_0 * ufl.grad(v)[0]) * ufl.dx + (
        2 * ufl.pi * sigma_theta_0 * v
    ) * ufl.dx

    # 右边：外力虚功（仅内外压，去掉轴向力项）
    p_a = fem.Constant(domain, p_in_list[0])  # 内壁压强(Pa)
    p_b = fem.Constant(domain, p_out_list[0])  # 外壁压强(Pa)
    F_total = np.pi * (r_in**2 * p_a.value - r_out**2 * p_b.value)  # 总轴向拉力(N)

    # 边界标记与积分
    fdim = domain.topology.dim - 1
    left = locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], r_in))
    right = locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], r_out))
    indices = np.hstack([left, right])
    values = np.hstack([np.full_like(left, 1), np.full_like(right, 2)])
    sorted_order = np.argsort(indices)
    mt = meshtags(domain, fdim, indices[sorted_order], values[sorted_order])
    ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)

    RHS = 2 * ufl.pi * r_in * p_a * v * ds(1) - 2 * ufl.pi * r_out * p_b * v * ds(2)
    rhs_form = fem.form(RHS)

    # ===== 求解平面应变位移场 =====
    u_0_sol = fem.Function(V_u)

    # 组装矩阵
    A = assemble_matrix(fem.form(LHS), bcs=[])
    A.assemble()

    # 创建求解器
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType("preonly")
    solver.getPC().setType("lu")
    """
    problem = LinearProblem(
        LHS,
        RHS,
        bcs=[],
        u=u_0_sol,
        petsc_options_prefix="solve_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    """

    # ===== 定义输出 DG0 空间 =====
    V_dg0 = fem.functionspace(domain, ("DG", 0))
    ip = V_dg0.element.interpolation_points

    sigma_r_sol = fem.Function(V_dg0)
    sigma_theta_sol = fem.Function(V_dg0)
    sigma_z_sol = fem.Function(V_dg0)
    u_total_sol = fem.Function(V_dg0)

    L1 = len(p_out_list)
    L2 = len(p_in_list)
    h = (r_out - r_in) / nx
    r_vals = r_in + (np.arange(nx) + 0.5) * h  # 单元中心坐标
    u_vals_list = np.zeros((L1, L2, nx))
    sigma_vals_list = np.zeros((L1, L2, 6, nx))

    # ===== 循环求解 =====
    total_tqdm = L1 * L2
    pbar = tqdm(total=total_tqdm)
    for i, p_o in enumerate(p_out_list):
        for j, p_i in enumerate(p_in_list):
            # ===== 更新参数 组装矩阵并求解 =====
            p_a.value = p_i
            p_b.value = p_o
            b = assemble_vector(rhs_form)
            # b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            # u_0_sol = problem.solve()
            solver.solve(b, u_0_sol.x.petsc_vec)
            u_0_sol.x.scatter_forward()

            # ===== 叠加法计算广义平面应变总解 =====
            # 1. 计算平面应变下的总轴向力F0
            sigma_z_0_sol = ufl.replace(sigma_z_0, {u: u_0_sol})
            F0_form = 2 * ufl.pi * sigma_z_0_sol * r * ufl.dx
            F0 = fem.assemble_scalar(fem.form(F0_form))

            # 2. 计算轴向力增量与均匀轴向应变
            A = np.pi * (r_out**2 - r_in**2)  # 横截面积
            delta_F = F_total - F0
            delta_sigma_z = delta_F / A
            eps_z_sol = delta_sigma_z / E.value  # 总轴向应变（全局常数）

            # 3. 计算总应力场
            eps_r_sol = ufl.grad(u_0_sol)[0]
            eps_theta_sol = u_0_sol / r
            sigma_r_val = lmbda * (eps_r_sol + eps_theta_sol) + 2 * G * eps_r_sol
            sigma_theta_val = (
                lmbda * (eps_r_sol + eps_theta_sol) + 2 * G * eps_theta_sol
            )
            sigma_z_val = lmbda * (eps_r_sol + eps_theta_sol) + delta_sigma_z

            sigma_r_expr = fem.Expression(sigma_r_val, ip)
            sigma_theta_expr = fem.Expression(sigma_theta_val, ip)
            sigma_z_expr = fem.Expression(sigma_z_val, ip)

            sigma_r_sol.interpolate(sigma_r_expr)
            sigma_theta_sol.interpolate(sigma_theta_expr)
            sigma_z_sol.interpolate(sigma_z_expr)

            # 4. 计算总径向位移（叠加泊松效应引起的径向位移）
            u_total_expr = fem.Expression(u_0_sol - nu * eps_z_sol * r, ip)
            u_total_sol.interpolate(u_total_expr)

            # ===== 赋值 =====
            # DG0 函数的 x.array 按单元顺序存储，直接取用即可
            u_vals = u_total_sol.x.array.copy()
            u_vals_list[i][j] = u_vals

            sigma_r_vals = sigma_r_sol.x.array.copy()
            sigma_theta_vals = sigma_theta_sol.x.array.copy()
            sigma_z_vals = sigma_z_sol.x.array.copy()
            sigma_vals = np.stack(
                [sigma_r_vals, sigma_theta_vals, sigma_z_vals], axis=0
            )
            sigma_vals_list[i][j][:3] = sigma_vals

            # 更新 tqdm
            pbar.update(1)
            tqdm.write(f"u_0_sol.x.norm() = {np.linalg.norm(u_0_sol.x.array)}")

    pbar.close()
    return r_vals, u_vals_list, sigma_vals_list


def solve_aniso_pipe(
    r_interface_list, C_basic, theta_list, p_in_list, p_out_list, nx=100
):
    """
    各向异性圆管径向应力分布求解器

    求解径向一维广义平面应变问题的应力分量分布
    不支持多铺层嵌套圆管, 不支持获得材料坐标系下应力分量

    输入:
        r_interface_list: 界面半径数组 (layers_num+1,)
        C_basic:    材料坐标系下原始刚度矩阵数组 (6, 6)
        theta_list: 铺层角度数组 rad (layers_num,)
        p_i_list:   内壁压强数组 Pa (...,)
        p_o_list:   外壁压强数组 Pa (...,)
        nx:         网格单元数及插值输出数 默认=100 int
    输出:
        r_vals: 径向插值坐标 (nx,)
        u_vals: 插值径向位移 (nx,)
        sigma_vec_vals: 柱坐标系下插值应力分量 (nx,6)
            - 索引0: sigma_r
            - 索引1: sigma_θ
            - 索引2: sigma_z
            - 索引3: sigma_θz
            - 索引4: sigma_rz
            - 索引5: sigma_rθ
    """
    pass


def solver_aniso_layered_pipe():
    pass


def solve_lame_stress(r, r_in, r_out, p_in, p_out):
    """
    本函数给出厚壁均匀圆管(闭口状态)在受内外压力下的lame应力分布解析解
    解遵循 A ± B/r**2 的简洁形式
    """

    p1 = p_in
    p2 = p_out
    r1 = r_in
    r2 = r_out

    A = (r1**2 * p1 - r2**2 * p2) / (r2**2 - r1**2)
    B = (r1**2 * r2**2 * (p1 - p2)) / (r2**2 - r1**2)

    s_r = A - B / (r**2)
    s_t = A + B / (r**2)
    s_z = A

    return s_r, s_t, s_z


def main():
    r_in = 0.1
    r_out = 0.2
    E = 210.09
    nu = 0.3
    p_out_list = np.linspace(0.0, 0.1e6, 3)
    p_in_list = np.linspace(1.0e6, 5.0e6, 5)

    r_vals, u_vals_list, s_vals_list = solve_iso_pipe(
        r_in, r_out, E, nu, p_in_list, p_out_list
    )

    print("solve_iso_pipe solution:")
    print(f"r_vals.shape = {r_vals.shape}")
    print(f"u_vals_list.shape = {u_vals_list.shape}")
    print(f"s_vals_list.shape = {s_vals_list.shape}")
    print("\n")

    print()


if __name__ == "__main__":
    main()
