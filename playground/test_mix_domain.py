import ufl
from dolfinx import mesh, fem
from scifem import create_real_functionspace
from mpi4py import MPI


domain = mesh.create_unit_interval(MPI.COMM_WORLD, 10)

V_u = fem.functionspace(domain, ("Lagrange", 2))

R = create_real_functionspace(domain)
V = ufl.MixedFunctionSpace(V_u, R)

u, eps = ufl.TrialFunctions(V)
v, q = ufl.TestFunctions(V)
print("Success! Mixed space created.")