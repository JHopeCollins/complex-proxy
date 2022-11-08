
import firedrake as fd
import complex_proxy as cpx

import pytest


@pytest.fixture
def nx():
    return 10


@pytest.fixture
def mesh(nx):
    return fd.UnitSquareMesh(nx, nx)


@pytest.mark.parametrize("family", ["CG", "BDM"])
def test_function_space(mesh, family):
    """
    Test that the complex FunctionSpace is correctly constructed for a scalar real FunctionSpace
    """
    V = fd.FunctionSpace(mesh, family, 1)

    W = cpx.FunctionSpace(V)

    assert len(W.split()) == len(V.split())

    assert W.sub(0).ufl_element() == V.ufl_element()
    assert W.sub(1).ufl_element() == V.ufl_element()


@pytest.mark.parametrize("dim", [None, 3])
def test_vector_function_space(mesh, dim):
    """
    Test that the complex FunctionSpace is correctly constructed for a real VectorFunctionSpace
    """
    V = fd.VectorFunctionSpace(mesh, "CG", 1, dim=dim)

    W = cpx.FunctionSpace(V)

    assert len(W.split()) == len(V.split())

    shape_check = (2, V.ufl_element().num_sub_elements())
    assert W.ufl_element()._shape == shape_check

    for elem in W.ufl_element().sub_elements():
        assert elem == V.ufl_element().sub_elements()[0]


@pytest.mark.parametrize("shape", [None, (3, 3), (2, 3, 4)])
def test_tensor_function_space(mesh, shape):
    """
    Test that the complex FunctionSpace is correctly constructed for a scalar real TensorFunctionSpace
    """
    V = fd.TensorFunctionSpace(mesh, "CG", 1, shape=shape)

    W = cpx.FunctionSpace(V)

    assert len(W.split()) == len(V.split())

    shape_check = (2,) + V.ufl_element()._shape
    assert W.ufl_element()._shape == shape_check

    for elem in W.ufl_element().sub_elements():
        assert elem == V.ufl_element().sub_elements()[0]


def test_mixed_function_space(mesh):
    """
    Test that the complex FunctionSpace is correctly constructed for a mixed real FunctionSpace
    """
    V0 = fd.FunctionSpace(mesh, "DG", 1)
    V1 = fd.FunctionSpace(mesh, "BDM", 2)
    V2 = fd.VectorFunctionSpace(mesh, "CG", 3)
    V3 = fd.TensorFunctionSpace(mesh, "Lagrange", 2, shape=(1,2,3,4))

    V = V0*V1*V2*V3

    W0 = cpx.FunctionSpace(V0)
    W1 = cpx.FunctionSpace(V1)
    W2 = cpx.FunctionSpace(V2)
    W3 = cpx.FunctionSpace(V3)

    W = cpx.FunctionSpace(V)

    assert len(W.split()) == len(V.split())

    assert W.split()[0] == W0
    assert W.split()[1] == W1
    assert W.split()[2] == W2
    assert W.split()[3] == W3
