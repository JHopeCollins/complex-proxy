
import firedrake as fd
import complex_proxy as cpx


def test_function_space():
    """
    Test that the complex FunctionSpace is correctly constructed for a scalar real FunctionSpace
    """
    nx = 10
    mesh = fd.UnitIntervalMesh(nx)

    V = fd.FunctionSpace(mesh, "CG", 1)

    W = cpx.FunctionSpace(V)

    assert len(W.split()) == len(V.split())

    assert W.sub(0).ufl_element() == V.ufl_element()
    assert W.sub(1).ufl_element() == V.ufl_element()


def test_mixed_function_space():
    """
    Test that the complex FunctionSpace is correctly constructed for a mixed real FunctionSpace
    """
    nx = 10
    mesh = fd.UnitSquareMesh(nx, nx)

    V1 = fd.FunctionSpace(mesh, "DG", 1)
    V2 = fd.FunctionSpace(mesh, "BDM", 2)
    V3 = fd.FunctionSpace(mesh, "CG", 3)

    V = V1*V2*V3

    W = cpx.FunctionSpace(V)

    assert len(W.split()) == len(V.split())

    for Q, P in zip(V.split(), W.split()):
        assert P.sub(0).ufl_element() == Q.ufl_element()
        assert P.sub(1).ufl_element() == Q.ufl_element()
