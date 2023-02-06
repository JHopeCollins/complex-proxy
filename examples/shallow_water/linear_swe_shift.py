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
    description='Complex-valued gravity wave testcase using fully implicit linear SWE solver and shift preconditioning.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--ref_level', type=int, default=3, help='Refinement level of icosahedral grid.')
parser.add_argument('--dt', type=float, default=1.36, help='Timestep in hours.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for implicit theta-method.')
parser.add_argument('--filename', type=str, default='gravity_waves', help='Name of output vtk files.')
parser.add_argument('--degs', type=float, default=45, help='Angle of complex coefficient on mass matrix.')
parser.add_argument('--shift_degs', type=float, default=45, help='Angle of complex coefficient on mass matrix in shift preconditioner.')
parser.add_argument('--patch_type', type=str, default='vanka', help='Patch type for multigrid smoother.')
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

theta = args.theta
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
        'monitor': None,
        'converged_reason': None,
        'atol': 1e-0,
        'rtol': 1e-10,
    },
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'multiplicative',
    'mg': deepcopy(mg_parameters)
}

Print("\nReal-valued solve\n")

w1 = fd.Function(W)
problem = fd.LinearVariationalProblem(A, L, w1)
solver = fd.LinearVariationalSolver(problem, solver_parameters=solver_parameters)

ofile = fd.File(f"output/{args.filename}.pvd")
ofile.write(*w0.subfunctions, time=0)

solver.solve()
w0.assign(w1)
ofile.write(*w0.subfunctions, time=dt/units.hour)

# complex problem

Wc = cpx.FunctionSpace(W)

wc0 = fd.Function(Wc)
wc1 = fd.Function(Wc)

cpx.set_real(wc0, winit)
cpx.set_imag(wc0, winit)

from math import sin, cos, pi

# complex block

phi = (pi/2)*(args.degs/90)
eta = 0.025
z = eta*(cos(phi) + sin(phi)*1j)

Mc = cpx.BilinearForm(Wc, z, form_mass)
Kc = cpx.BilinearForm(Wc, 1, form_function)

Ac = Dt_r*Mc + Kc

Lc = fd.action(Dt_r*Mc - ((1-Theta)/Theta)*Kc, wc0)

Print(f"\nComplex-valued solve:")
Print(f"z = {z}\n")

solver_parameters_c = solver_parameters

wc1 = fd.Function(Wc)
problem_c = fd.LinearVariationalProblem(Ac, Lc, wc1)
solver_c = fd.LinearVariationalSolver(problem_c, solver_parameters=solver_parameters_c)

solver_c.solve()

# shift preconditioner

phi_p = (pi/2)*(args.shift_degs/90)
eta_p = 1.0*eta
z_p = eta_p*(cos(phi_p) + sin(phi_p)*1j)

Mc_p = cpx.BilinearForm(Wc, z_p, form_mass)
Kc_p = cpx.BilinearForm(Wc, 1, form_function)

Ac_p = Dt_r*Mc_p + Kc_p

class ShiftPC(fd.AuxiliaryOperatorPC):
    def form(self, pc, *trials_and_tests):
        return self.get_appctx(pc).get("cpx_form")

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
        'mat_type': 'matfree',
        'ksp_type': 'fgmres',
        'ksp': {
            'monitor': None,
            'converged_reason': None,
            'rtol': 1e-5,
        },
        'pc_type': 'mg',
        'pc_mg_cycle_type': 'v',
        'pc_mg_type': 'multiplicative',
        'mg': deepcopy(mg_parameters)
    }
}

Print(f"\nShift preconditioned complex-valued solve:")
Print(f"zp = {z_p}\n")

appctx = {'cpx_form': (Ac_p, None)}

wcp1 = fd.Function(Wc)
problem_cp = fd.LinearVariationalProblem(Ac, Lc, wcp1)
solver_cp = fd.LinearVariationalSolver(problem_cp,
                                       solver_parameters=solver_parameters_cp,
                                       appctx=appctx)

solver_cp.solve()
