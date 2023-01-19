import firedrake as fd
import complex_proxy as cpx

import numpy as np

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

dt = 1
dt = dt*units.hour
Dt1 = fd.Constant(1/dt)
Dt = fd.Constant(dt)

theta = 0.5
Theta = fd.Constant(theta)

winit = fd.Function(W)
uinit, hinit = winit.split()

uinit.project(case.velocity_expression(*x))
hinit.project(case.depth_expression(*x))

wtrial = fd.TrialFunction(W)
wtest = fd.TestFunction(W)

u, h = fd.split(wtrial)
v, q = fd.split(wtest)

def form_mass(u, h, v, q):
    return swe.linear.form_mass(mesh, u, h, v, q)

def form_function(u, h, v, q):
    return swe.linear.form_function(mesh, g, H, f, u, h, v, q)

M = form_mass(u, h, v, q)
K = form_function(u, h, v, q)
A = Dt1*M + Theta*K

r = fd.Function(W)
#fd.assemble(fd.action(Dt1*M - (1-Theta)*K, winit), tensor=r)
rhs = fd.assemble(fd.action(Dt1*M - (1-Theta)*K, winit), tensor=r)

#ofile = fd.File("swe.pvd")
##ofile.write(uinit, hinit)
#ofile.write(*r.split())
#from sys import exit
#exit()

#def rhs(v, q):
#    r = fd.Function(W)
#    np.random.seed(97469)
#    for s in r.split():
#        s.dat.data[:] = np.random.randn(*(s.dat.data.shape))
#    rs = r.split()
#    return (fd.inner(rs[0], v) + fd.inner(rs[1], q))*fd.dx

#L = rhs(v, q)
L = fd.inner(rhs, fd.TestFunction(W))*fd.dx

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
        'rtol': 1e-14,
    },
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'full',
    'mg': mg_parameters
}

w = fd.Function(W)
problem = fd.LinearVariationalProblem(A, L, w)
solver = fd.LinearVariationalSolver(problem, solver_parameters=solver_parameters)

solver.solve()

ofile = fd.File("swe.pvd")
ofile.write(*winit.split())

#Wc = cpx.FunctionSpace(W)
#
#Mc = cpx.BilinearForm(Wc, 1+1j, form_mass)
#Kc = cpx.BilinearForm(Wc, 1, form_function)
#
#Ac = Dt1*Mc + Theta*Kc
#
#Lc = cpx.LinearForm(
