import firedrake as fd
from petsc4py import PETSc
import complex_proxy as cpx

Print = PETSc.Sys.Print

import numpy as np
from copy import deepcopy

# asQ utils module
from utils import units
from utils.planets import earth
from utils import shallow_water as swe
import utils.shallow_water.gravity_bumps as case

from sys import exit

import argparse
parser = argparse.ArgumentParser(
    description='Complex-valued gravity wave testcase using fully implicit linear SWE solver.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--ref_level', type=int, default=3, help='Refinement level of icosahedral grid.')
parser.add_argument('--dt', type=float, default=1.36, help='Timestep in hours.')
parser.add_argument('--filename', type=str, default='gravity_waves', help='Name of output vtk files.')
parser.add_argument('--degs', type=float, default=45, help='Angle of complex coefficient on mass matrix.')
parser.add_argument('--patch_type', type=str, default='vanka', help='Patch type for multigrid smoother.')
parser.add_argument('--nscales', type=int, default=6, help='Number of times to half complex coefficient.')
parser.add_argument('--show_args', action='store_true', default=True, help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)
# set up real case

mesh = swe.create_mg_globe_mesh(ref_level=args.ref_level, coords_degree=1)
x = fd.SpatialCoordinate(mesh)

W = swe.default_function_space(mesh)

H = case.H
g = earth.Gravity
f = case.coriolis_expression(*x)

dt = args.dt
dt = dt*units.hour

theta = 0.5
Theta = fd.Constant(theta)

# coefficient on stiffness matrix is 1
Dt_r = fd.Constant(1/(dt*theta))

winit = fd.Function(W)
uinit, hinit = winit.subfunctions

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

# vertex patch smoother

patch_parameters = {
    'pc_patch': {
        'save_operators': True,
        'partition_of_unity': True,
        'sub_mat_type': 'seqdense',
        'construct_dim': 0,
        'construct_type': args.patch_type,
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
        #'monitor': None,
        #'converged_reason': None,
        'atol': 1e-0,
        'rtol': 1e-10,
    },
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'multiplicative',
    'mg': deepcopy(mg_parameters)
}

w1 = fd.Function(W).assign(0)
problem = fd.LinearVariationalProblem(A, L, w1)
solver = fd.LinearVariationalSolver(problem, solver_parameters=solver_parameters)

ofile = fd.File(f"output/{args.filename}.pvd")
ofile.write(*w0.subfunctions, time=0)

Print("\nReal-valued solve")

solver.solve()
w0.assign(w1)
ofile.write(*w0.subfunctions, time=dt/units.hour)

real_its = solver.snes.getLinearSolveIterations()

Print(f"Real iteration count: {real_its}")

# complex problem

Wc = cpx.FunctionSpace(W)

wc0 = fd.Function(Wc)
wc1 = fd.Function(Wc)

cpx.set_real(wc0, winit)
cpx.set_imag(wc0, winit)

from math import sin, cos, pi

# complex block

phi = (pi/2)*(args.degs/90)
eta = 1.0
z = eta*(cos(phi) + sin(phi)*1j)

Mc, zr, zi = cpx.BilinearForm(Wc, z, form_mass, return_z=True)
Kc = cpx.BilinearForm(Wc, 1, form_function)

zr.assign(1.5*z.real)
zi.assign(1.5*z.imag)

Ac = Dt_r*Mc + Kc

Lc = fd.action(Dt_r*Mc - ((1-Theta)/Theta)*Kc, wc0)

solver_parameters_c = solver_parameters

wc1 = fd.Function(Wc).assign(0)
problem_c = fd.LinearVariationalProblem(Ac, Lc, wc1)
solver_c = fd.LinearVariationalSolver(problem_c, solver_parameters=solver_parameters_c)

Print("\nComplex-valued solves\n")
Print(f"Complex angle: {args.degs}")
Print(f"Patch type: {args.patch_type}\n")

for n in range(args.nscales):
    scale = 1/pow(2,n)

    zr.assign(scale*z.real)
    zi.assign(scale*z.imag)
    
    solver_c.solve()
    complex_its = solver_c.snes.getLinearSolveIterations()

    Print(f"scale: {str(scale).ljust(10, ' ')} | niterations: {str(complex_its).rjust(2, ' ')}")

    wc1.assign(0)
