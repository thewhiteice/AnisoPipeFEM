import ufl
from dolfinx import mesh, fem
from basix.ufl import real_element
from mpi4py import MPI

domain = mesh.create_unit_interval(MPI.COMM_WORLD, 10)

# 创建常规函数空间
V_u = fem.functionspace(domain, ("Lagrange", 2))

# 用 real_element 创建 Real 元素，再手动构造函数空间
real_elem = real_element(domain.basix_cell())
R = fem.functionspace(domain, real_elem)

# 用 ufl.MixedFunctionSpace 组合两个空间
V = ufl.MixedFunctionSpace(V_u, R)

u, eps = ufl.TrialFunctions(V)
v, q = ufl.TestFunctions(V)
print("Success! Mixed space created.")