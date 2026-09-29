"""
本文件实现广义平面应变问题下的混合函数空间构建及牛顿非线性求解器调用。

柱坐标系下一维径向求解域。
介质为各向同性钢材: E=200GPa, nu=0.3
边界条件:
    径向受内外壁压力, 轴向受闭口状态带来的拉力
"""

# %%
import basix
import numpy as np
import ufl
from basix.ufl import real_element
from dolfinx import fem, mesh
from dolfinx.mesh import locate_entities_boundary, meshtags
from mpi4py import MPI
from scifem import NewtonSolver

# %%
# 1. 几何与网格
r_i = 191.0e-3  # 内半径 (m)
r_o = 221.0e-3  # 外半径 (m)，例如 30mm 厚
nx = 50  # 单元数

# 构造一维径向网格
points = np.linspace(r_i, r_o, nx + 1)
cells = np.array([[i, i + 1] for i in range(nx)], dtype=np.int64)
e = basix.ufl.element("Lagrange", "interval", 1, shape=(1,))
domain = mesh.create_mesh(MPI.COMM_WORLD, cells, e, points.reshape(-1, 1))

# 2. 混合函数空间：位移 u (标量，Lagrange) + 轴向应变 eps_z (标量常数)
V_u = fem.functionspace(domain, ("Lagrange", 2))
real_elem = real_element(domain.basix_cell())
R = fem.functionspace(domain, real_elem)
V = ufl.MixedFunctionSpace(V_u, R)

du, deps = ufl.TrialFunctions(V)  # 试探函数，给雅可比用
v, q = ufl.TestFunctions(V)  # 测试函数
u = fem.Function(V_u)  # 解函数
eps_z = fem.Function(R)  # 解函数

# %%
# 3. 材料参数（线弹性，柱坐标下的正交各向异性）
# 应力顺序: [sigma_r, sigma_theta, sigma_z, tau_thetaz, tau_rz, tau_rtheta]
# 这里简单设置一个各向同性材料（钢材）
E = 200e9
nu = 0.3
lam = E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame 常数
mu = E / (2 * (1 + nu))

# 柱坐标下的各向同性刚度矩阵
C_cyl = np.array(
    [
        [lam + 2 * mu, lam, lam, 0, 0, 0],
        [lam, lam + 2 * mu, lam, 0, 0, 0],
        [lam, lam, lam + 2 * mu, 0, 0, 0],
        [0, 0, 0, mu, 0, 0],
        [0, 0, 0, 0, mu, 0],
        [0, 0, 0, 0, 0, mu],
    ],
    dtype=np.float64,
)

# 作为 DG0 场
V_DG0_t = fem.functionspace(domain, ("DG", 0, (6, 6)))
C = fem.Function(V_DG0_t)
num_cells = domain.topology.index_map(domain.topology.dim).size_local
C.x.array[:] = np.tile(C_cyl.flatten(), num_cells)

# 4. 载荷
p_i = 100e6  # 内压 (Pa)
p_o = 0.0  # 外压
r = ufl.SpatialCoordinate(domain)[0]

# 应变分量
eps_r = u.dx(0)
eps_theta = u / r
# 广义平面应变下，剪应变为0
eps_vec = ufl.as_vector([eps_r, eps_theta, eps_z, 0.0, 0.0, 0.0])

# 应力
sigma_vec = ufl.dot(C, eps_vec)
sigma_r = sigma_vec[0]
sigma_theta = sigma_vec[1]
sigma_z = sigma_vec[2]

# 5. 残差方程
# 径向平衡弱形式
ds = ufl.Measure("ds", domain=domain)

fdim = domain.topology.dim - 1
left = locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], r_i))
right = locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], r_o))
indices = np.hstack([left, right])
values = np.hstack([np.full_like(left, 1), np.full_like(right, 2)])
sorted_order = np.argsort(indices)
mt = meshtags(domain, fdim, indices[sorted_order], values[sorted_order])
ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)

# 径向平衡残差
R_u = (
    2 * ufl.pi * r * (sigma_r * v.dx(0) + sigma_theta * v / r) * ufl.dx
    - 2 * ufl.pi * r_i * p_i * v * ds(1)
    + 2 * ufl.pi * r_o * p_o * v * ds(2)
)

# 轴向力平衡残差
volume = np.pi * (r_o**2 - r_i**2)
F_target = np.pi * (r_i**2 * p_i - r_o**2 * p_o)
F_target_density = F_target / volume
R_z = (2 * ufl.pi * r * sigma_z - F_target_density) * q * ufl.dx

# 总残差
F = R_u + R_z

# %%
# 6. 定义非线性问题
R = [R_u, R_z]  # 构造残差列表 R

# 用 ufl.derivative 构造雅可比，方向是 du 和 deps
K = [
    [ufl.derivative(R_u, u, du), ufl.derivative(R_u, eps_z, deps)],
    [ufl.derivative(R_z, u, du), ufl.derivative(R_z, eps_z, deps)],
]

# 5. 直接用 scifem.NewtonSolver 求解
problem = NewtonSolver(
    R,
    K,
    [u, eps_z],
    max_iterations=25,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
)

# %%
# 7. 求解器设置
problem.convergence_criterion = "residual"
problem.rtol = 1e-10
problem.atol = 1e-12
problem.max_it = 25
problem.report = True

# %%
"""
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "residual"
solver.rtol = 1e-10
solver.atol = 1e-12
solver.max_it = 25
solver.report = True
"""

# %%
# 求解
try:
    n = problem.solve()
    print(f"Newton iterations: {n}")
except RuntimeError as e:
    print(f"Solver failed: {e}")

# %%
# 8. 输出结果
print(f"eps_z = {eps_z.x.array[0]:.6e}")
print(f"u(r_i) = {u.x.array[0]:.6e} m")
print(f"u(r_o) = {u.x.array[-1]:.6e} m")
