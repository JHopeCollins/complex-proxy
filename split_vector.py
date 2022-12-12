
import firedrake as fd
import complex_proxy as cpx

mesh = fd.UnitSquareMesh(4, 4)

# V = fd.FunctionSpace(mesh, "DG", 1)  # works
# V = fd.FunctionSpace(mesh, "BDM", 1)  # breaks
# V = fd.VectorFunctionSpace(mesh, "CG", 1)  # breaks

V = fd.MixedFunctionSpace((fd.FunctionSpace(mesh, "DG", 1),
                           fd.FunctionSpace(mesh, "BDM", 1)))  # works!!

W = cpx.FunctionSpace(V)

w = fd.Function(W)

print("\n")
print(f"w.ufl_element is a {w.ufl_element().family()} element")
print(f"w.ufl_shape = {w.ufl_shape}")
print("\n")

ws = fd.split(w)
