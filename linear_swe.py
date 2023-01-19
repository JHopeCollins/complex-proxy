import firedrake as fd
import complex_proxy as cpx

import numpy as np

# asQ utils module
from utils import units
from utils.planets import earth
from utils import shallow_water as swe
import utils.shallow_water.gravity_bumps as case

#mesh = swe.create_mg_globe_mesh(ref_level=3, coords_degree=1)
mesh = fd.IcosahedralSphereMesh(radius=earth.radius,
                                refinement_level=3,
                                degree=1)
x = fd.SpatialCoordinate(mesh)
mesh.init_cell_orientations(x)

W = swe.default_function_space(mesh)

H = case.H
g = earth.Gravity
f = case.coriolis_expression(*x)

dt = 1.36
dt = dt*units.hour
Dt1 = fd.Constant(1/dt)
Dt = fd.Constant(dt)

theta = 0.5
Theta = fd.Constant(theta)

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
A = Dt1*M + Theta*K

aP = M + Theta*K

L = fd.action(Dt1*M - (1 - Theta)*K, w0)

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
    #'ksp_view': None,
    'mat_type': 'matfree',
    'ksp_type': 'fgmres',
    'ksp': {
        'monitor': None,
        'converged_reason': None,
        'atol': 1e-0,
        'rtol': 1e-16,
    },
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'multiplicative',
    'mg': mg_parameters
}

w1 = fd.Function(W)
problem = fd.LinearVariationalProblem(A, L, w1, aP=aP)
solver = fd.LinearVariationalSolver(problem, solver_parameters=solver_parameters)

ofile = fd.File("swe.pvd")
ofile.write(*w0.split(), time=0)

solver.solve()
w0.assign(w1)
ofile.write(*w0.split(), time=dt/units.hour)

from sys import exit
exit()


Wc = cpx.FunctionSpace(W)

wc0 = fd.Function(Wc)
wc1 = fd.Function(Wc)

cpx.set_real(wc0, winit)
cpx.set_imag(wc0, winit)

from math import sin, cos, pi

# complex block

#phi = deg*(2*pi)/360
phi = 0.5*(pi/2)
z = cos(phi) + sin(phi)*1j

Mc = cpx.BilinearForm(Wc, z, form_mass)
Kc = cpx.BilinearForm(Wc, 1, form_function)

Ac = Dt1*Mc + Theta*Kc

Lc = fd.action(Dt1*Mc - (1-Theta)*Kc, wc0)

# shift preconditioner

phi_p = 0.5*(pi/2)
z_p = cos(phi) + sin(phi)*1j

Mc_p = cpx.BilinearForm(Wc, z_p, form_mass)
Kc_p = cpx.BilinearForm(Wc, 1, form_function)

Ac_p = Dt1*Mc_p + Theta*Kc_p
#aP = Ac_p
#aP = None
aP = fd.assemble(Ac_p)

fd.solve(Ac == Lc, w, solver_parameters=solver_parameters, Jp=Ac_p)

#problem_c = fd.LinearVariationalProblem(Ac, Lc, wc1, aP=aP)
#solver_c = fd.LinearVariationalSolver(problem_c, solver_parameters=solver_parameters)
#
#solver_c.solve()
