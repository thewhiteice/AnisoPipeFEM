#
# Copyright (C) 2020-2024 by The DOLFINx developers
#
# This file is part of DOLFINx.
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
# ruff: noqa: E402

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import ufl
from basix.ufl import element
from dolfinx import fem, has_adios2, mesh
from dolfinx.fem.petsc import discrete_gradient, interpolation_matrix
from dolfinx.mesh import CellType, create_unit_square

# Solution scalar (e.g., float32, complex128) and geometry (float32/64)
# types
dtype = PETSc.ScalarType
xdtype = PETSc.RealType

msh = create_unit_square(MPI.COMM_WORLD, 96, 96, CellType.triangle, dtype=xdtype)
gdim = msh.geometry.dim
fdim = msh.topology.dim - 1
facets_top = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
facets_bottom = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
cells_top = mesh.compute_incident_entities(msh.topology, facets_top, fdim, fdim + 1)
cells_bottom = mesh.compute_incident_entities(msh.topology, facets_bottom, fdim, fdim + 1)
has_hypre = PETSc.Sys().hasExternalPackage("hypre")
hypre_ams_compatible = not np.issubdtype(dtype, np.complexfloating)


def solve(k: int, use_hypre: bool) -> tuple[fem.Function, fem.Function]:
    """Solve the mixed Poisson problem with Raviart-Thomas degree ``k``.

    Args:
        k: Raviart-Thomas element degree.
        use_hypre: Whether to use Hypre AMS rather than LU.
    """
    if k < 1:
        raise ValueError("Element degree must be at least 1.")
    if use_hypre and not has_hypre:
        raise RuntimeError("PETSc is not configured with Hypre.")
    if use_hypre and not hypre_ams_compatible:
        raise RuntimeError("Hypre AMS does not support complex scalar types.")

    V = fem.functionspace(msh, element("RT", msh.basix_cell(), k, dtype=xdtype))
    W = fem.functionspace(msh, element("DG", msh.basix_cell(), k - 1, dtype=xdtype))
    Q = ufl.MixedFunctionSpace(V, W)
    sigma_trial, u_trial = ufl.TrialFunctions(Q)
    tau, v = ufl.TestFunctions(Q)

    x = ufl.SpatialCoordinate(msh)
    f = 10 * ufl.exp(-((x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5)) / 0.02)
    dx = ufl.Measure("dx", msh)
    a = ufl.extract_blocks(
        ufl.inner(sigma_trial, tau) * dx
        + ufl.inner(u_trial, ufl.div(tau)) * dx
        + ufl.inner(ufl.div(sigma_trial), v) * dx
    )
    L = [ufl.ZeroBaseForm((tau,)), -ufl.inner(f, v) * dx]
    a_p = ufl.extract_blocks(
        ufl.inner(sigma_trial, tau) * dx
        + ufl.inner(ufl.div(sigma_trial), ufl.div(tau)) * dx
        + ufl.inner(u_trial, v) * dx
    )

    dofs_top = fem.locate_dofs_topological(V, fdim, facets_top)
    dofs_bottom = fem.locate_dofs_topological(V, fdim, facets_bottom)
    g = fem.Function(V, dtype=dtype)
    g.interpolate(lambda x: np.vstack((np.zeros_like(x[0]), np.sin(5 * x[0]))), cells0=cells_top)
    g.interpolate(
        lambda x: np.vstack((np.zeros_like(x[0]), -np.sin(5 * x[0]))), cells0=cells_bottom
    )
    bcs = [fem.dirichletbc(g, dofs_top), fem.dirichletbc(g, dofs_bottom)]

    sigma = fem.Function(V, name="sigma", dtype=dtype)
    u = fem.Function(W, name="u", dtype=dtype)
    problem = fem.petsc.LinearProblem(
        a,
        L,
        u=[sigma, u],
        P=a_p,
        kind="nest",
        bcs=bcs,
        petsc_options_prefix=f"demo_mixed_poisson_{k}_",
        petsc_options={
            "ksp_type": "minres",
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "additive",
            "ksp_rtol": 1e-5 if np.finfo(dtype).bits == 32 else 1e-7,
            "ksp_error_if_not_converged": True,
        },
    )
    ksp = problem.solver
    solver_label = f"k={k} ({'Hypre AMS' if use_hypre else 'LU'})"
    ksp.setMonitor(
        lambda _, its, rnorm: PETSc.Sys.Print(
            f"{solver_label}: iteration {its:>4d}, residual: {rnorm:.3e}"
        )
    )

    ksp_sigma, ksp_u = ksp.getPC().getFieldSplitSubKSP()
    ksp_u.getPC().setType("jacobi")
    ksp_u.setFromOptions()
    pc_sigma = ksp_sigma.getPC()

    if use_hypre:
        pc_sigma.setType("hypre")
        pc_sigma.setHYPREType("ams")

        opts = PETSc.Options()
        opts[f"{ksp_sigma.prefix}pc_hypre_ams_cycle_type"] = 7  # type: ignore[index]
        opts[f"{ksp_sigma.prefix}pc_hypre_ams_relax_times"] = 3  # type: ignore[index]

        V_H1 = fem.functionspace(msh, element("Lagrange", msh.basix_cell(), k, dtype=xdtype))
        V_curl = fem.functionspace(msh, element("N1curl", msh.basix_cell(), k, dtype=xdtype))
        G = discrete_gradient(V_H1, V_curl)
        G.assemble()
        pc_sigma.setHYPREDiscreteGradient(G)

        if k == 1:
            cvec0, cvec1 = fem.Function(V), fem.Function(V)
            cvec0.interpolate(lambda x: np.vstack((np.ones_like(x[0]), np.zeros_like(x[1]))))
            cvec1.interpolate(lambda x: np.vstack((np.zeros_like(x[0]), np.ones_like(x[1]))))
            pc_sigma.setHYPRESetEdgeConstantVectors(cvec0.x.petsc_vec, cvec1.x.petsc_vec, None)
        else:
            V_H1d = fem.functionspace(msh, ("Lagrange", k, (msh.geometry.dim,)))
            Pi = interpolation_matrix(V_H1d, V)
            Pi.assemble()
            pc_sigma.setHYPRESetInterpolations(msh.geometry.dim, None, None, Pi, None)
            opts[f"{ksp_sigma.prefix}pc_hypre_ams_tol"] = 1e-12  # type: ignore[index]
            opts[f"{ksp_sigma.prefix}pc_hypre_ams_max_iter"] = 3  # type: ignore[index]

    else:
        pc_sigma.setType("lu")
        use_superlu = PETSc.IntType == np.int64
        if PETSc.Sys().hasExternalPackage("mumps") and not use_superlu:
            pc_sigma.setFactorSolverType("mumps")
        elif PETSc.Sys().hasExternalPackage("superlu_dist"):
            pc_sigma.setFactorSolverType("superlu_dist")

    ksp_sigma.setFromOptions()

    problem.solve()
    return sigma, u


# Solve and save the flux and scalar solutions for the lowest-order and
# next-order cases.
if has_adios2:
    from dolfinx.io import VTXWriter


use_hypre = has_hypre and hypre_ams_compatible
for k in (1, 2):
    sigma, u = solve(k, use_hypre)
    if has_adios2:
        # VTX supports (discontinuous) Lagrange functions, so
        # interpolate the flux
        V_sigma = fem.functionspace(
            msh,
            element("DG", msh.basix_cell(), k, shape=(gdim,), dtype=xdtype),
        )
        sigma_output = fem.Function(V_sigma, name="sigma", dtype=dtype)
        sigma_output.interpolate(sigma)
        with VTXWriter(msh.comm, f"output_mixed_poisson_sigma_{k}.bp", sigma_output) as f:
            f.write(0.0)
        with VTXWriter(msh.comm, f"output_mixed_poisson_{k}.bp", u) as f:
            f.write(0.0)

if not has_adios2:
    print("ADIOS2 required for VTX output.")