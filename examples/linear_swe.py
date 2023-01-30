import firedrake as fd
from petsc4py import PETSc
import complex_proxy as cpx

import numpy as np
from copy import deepcopy

# asQ utils module
from utils import units
from utils.planets import earth
from utils import shallow_water as swe
import utils.shallow_water.gravity_bumps as case

mesh = swe.create_mg_globe_mesh(ref_level=3, coords_degree=1)
x = fd.SpatialCoordinate(mesh)

W = swe.default_function_space(mesh)

H = case.H
g = earth.Gravity
f = case.coriolis_expression(*x)

dt = 1.36
dt = dt*units.hour

theta = 0.5
Theta = fd.Constant(theta)

Dt_r = fd.Constant(1/(dt*theta))

winit = fd.Function(W)
uinit, hinit = winit.split()

uinit.project(case.velocity_expression(*x))
hinit.project(case.depth_expression(*x))

w0 = fd.Function(W).assign(winit)

wtrial = fd.TrialFunction(W)
wtest = fd.TestFunction(W)

def form_mass(u, h, v, q):
    return swe.linear.form_mass(mesh, u, h, v, q)

def form_function(u, h, v, q):
    return swe.linear.form_function(mesh, g, H, f, u, h, v, q)

u, h = fd.split(wtrial)
v, q = fd.split(wtest)

M = form_mass(u, h, v, q)
K = form_function(u, h, v, q)

A = Dt_r*M + K

L = fd.action(Dt_r*M - ((1 - Theta)/Theta)*K, w0)

patch_parameters = {
    'pc_patch': {
        'save_operators': True,
        'partition_of_unity': True,
        'sub_mat_type': 'seqdense',
        'construct_dim': 0,
        'construct_type': 'vanka',
        'local_type': 'additive',
        'precompute_element_tensors': True,
        'symmetrise_sweep': False
    },
    'sub': {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_shift_type': 'nonzero'
    }
}

mg_parameters = {
    'levels': {
        'ksp_type': 'gmres',
        'ksp_max_it': 5,
        'pc_type': 'python',
        'pc_python_type': 'firedrake.PatchPC',
        'patch': patch_parameters
    },
    'coarse': {
        'pc_type': 'python',
        'pc_python_type': 'firedrake.AssembledPC',
        'assembled_pc_type': 'lu',
        'assembled_pc_factor_mat_solver_type': 'mumps'
    }
}

solver_parameters = {
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'monitor': None,
        'converged_reason': None,
        'atol': 1e-0,
        'rtol': 1e-10,
    },
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'multiplicative',
    'mg': mg_parameters
}

PETSc.Sys.Print("\nReal-valued solve\n")

w1 = fd.Function(W)
problem = fd.LinearVariationalProblem(A, L, w1)
solver = fd.LinearVariationalSolver(problem, solver_parameters=solver_parameters)

ofile = fd.File("swe.pvd")
ofile.write(*w0.split(), time=0)

solver.solve()
w0.assign(w1)
ofile.write(*w0.split(), time=dt/units.hour)


# complex problem

Wc = cpx.FunctionSpace(W)

wc0 = fd.Function(Wc)
wc1 = fd.Function(Wc)

cpx.set_real(wc0, winit)
cpx.set_imag(wc0, winit)

from math import sin, cos, pi

# complex block

#phi = deg*(2*pi)/360
phi = (pi/2)*(10/90)
eta = 1.0
z = eta*(cos(phi) + sin(phi)*1j)

Mc = cpx.BilinearForm(Wc, z, form_mass)
Kc = cpx.BilinearForm(Wc, 1, form_function)

Ac = Dt_r*Mc + Kc

Lc = fd.action(Dt_r*Mc - ((1-Theta)/Theta)*Kc, wc0)

PETSc.Sys.Print(f"\nComplex-valued solve:")
PETSc.Sys.Print(f"z = {z}\n")

solver_parameters_c = solver_parameters

wc1 = fd.Function(Wc)
problem_c = fd.LinearVariationalProblem(Ac, Lc, wc1)
solver_c = fd.LinearVariationalSolver(problem_c, solver_parameters=solver_parameters_c)

solver_c.solve()

# shift preconditioner

#phi_p = (pi/2)*(0/90)
phi_p = 1.0*phi
eta_p = 1.0*eta
z_p = eta_p*(cos(phi_p) + sin(phi_p)*1j)

Mc_p = cpx.BilinearForm(Wc, z_p, form_mass)
Kc_p = cpx.BilinearForm(Wc, 1, form_function)

Ac_p = Dt_r*Mc_p + Kc_p

class ShiftPC(fd.AuxiliaryOperatorPC):
    def form(self, pc, *trials_and_tests):
        return (Ac_p, None)

solver_parameters_cp = {
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'monitor': None,
        'converged_reason': None,
        'atol': 1e-0,
        'rtol': 1e-10,
    },
    'pc_type': 'python',
    'pc_python_type': __name__+'.ShiftPC',
    'aux_pc_type': 'ksp',
    'aux_ksp': {
        'monitor': None,
        'converged_reason': None,
        'mat_type': 'matfree',
        'ksp_type': 'fgmres',
        'ksp_rtol': 1e-5,
        'pc_type': 'mg',
        'pc_mg_cycle_type': 'v',
        'pc_mg_type': 'multiplicative',
        'mg': mg_parameters
    }
}

PETSc.Sys.Print(f"\nShift preconditioned complex-valued solve:")
PETSc.Sys.Print(f"zp = {z_p}\n")

wcp1 = fd.Function(Wc)
problem_cp = fd.LinearVariationalProblem(Ac, Lc, wcp1)
solver_cp = fd.LinearVariationalSolver(problem_cp, solver_parameters=solver_parameters_cp)

solver_cp.solve()
