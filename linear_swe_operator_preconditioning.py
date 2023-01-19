import firedrake as fd

import utils.shallow_water.gravity_bumps as case
from utils.planets import earth
from utils import shallow_water as swe

from functools import partial

mesh = swe.create_mg_globe_mesh(ref_level=3, coords_degree=1)
x = fd.SpatialCoordinate(mesh)

g = earth.Gravity
H = case.H
f = case.coriolis_expression(*x)
dt = fd.Constant(1.36)

form_mass = partial(swe.linear.form_mass,
                    mesh)

form_function = partial(swe.linear.form_function,
                        mesh, g, H, f)

W = swe.default_function_space(mesh)

winit = fd.Function(W)
uinit, hinit = winit.split()

uinit.project(case.velocity_expression(*x))
hinit.project(case.depth_expression(*x))

w0 = fd.Function(W).assign(winit)
w1 = fd.Function(W).assign(winit)

wtrial = fd.TrialFunction(W)
wtest = fd.TestFunction(W)

wtrials = fd.split(wtrial)
wtests = fd.split(wtest)

M = form_mass(*wtrials, *wtests)
K = form_function(*wtrials, *wtests)

a = (1/dt)*M + K

aP = M + K

L = fd.action((1/dt)*M, w0)

solver_parameters = {
    #'ksp_view': None,
    'ksp_monitor': None,
    'ksp_converged_reason': None,
    'ksp_type': 'gmres',
    #'pc_type': 'ilu',
    #'pc_ksp_monitor': None,
    #'pc_ksp_converged_reason': None,
    #'pc_ksp_type': 'gmres',
    #'pc_ksp_pc_type': 'ilu'
}

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
        'rtol': 1e-12,
    },
    'pc_type': 'mg',
    'pc_mg_cycle_type': 'v',
    'pc_mg_type': 'multiplicative',
    'mg': mg_parameters
}

problem = fd.LinearVariationalProblem(a, L, w1, aP=aP)
solver = fd.LinearVariationalSolver(problem, solver_parameters=solver_parameters)
solver.solve()

fd.File("swe_aP.pvd").write(*w1.split())
