
import firedrake as fd

nx = 10

mesh = fd.UnitSquareMesh(nx, nx)
x, y = fd.SpatialCoordinate(mesh)

V = fd.FunctionSpace(mesh, "CG", 1)

u = fd.TrialFunction(V)
v = fd.TestFunction(V)

# right hand side function
f = (1+8*fd.pi*fd.pi)*fd.cos(x*fd.pi*2)*fd.cos(y*fd.pi*2)

L = fd.inner(f, v)*fd.dx

# left hand side

a = (fd.inner(fd.grad(u), fd.grad(v)) + fd.inner(u, v))*fd.dx

#aP = (fd.inner(fd.grad(u), fd.grad(v)) + fd.inner(u, v))*fd.dx
aP = fd.inner(fd.grad(u), fd.grad(v))*fd.dx
#aP = fd.inner(u, v)*fd.dx
#aP = None

solver_parameters = {
    'ksp_monitor': None,
    'ksp_converged_reason': None,
    'ksp_type': 'cg',
    'pc_type': 'icc',
}

w = fd.Function(V)

lvp = fd.LinearVariationalProblem(a, L, w, aP=aP)
lvs = fd.LinearVariationalSolver(lvp, solver_parameters=solver_parameters)

lvs.solve()
