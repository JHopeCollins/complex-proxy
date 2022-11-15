
import firedrake as fd
import complex_proxy as cpx

import pytest


@pytest.fixture
def nx():
    return 10


@pytest.fixture
def mesh(nx):
    return fd.UnitSquareMesh(nx, nx)


cell = fd.Cell('triangle')

scalar_elements = [
    fd.FiniteElement("CG", cell, 1),
    fd.FiniteElement("BDM", cell, 2),
]

vector_elements = [
    fd.VectorElement("DG", cell, 1),
    fd.VectorElement("DG", cell, 1, dim=3),
]

tensor_elements = [
    fd.TensorElement("Lagrange", cell, 1),
    fd.TensorElement("Lagrange", cell, 1, shape=(2, 3, 4))
]

elements = scalar_elements + vector_elements + tensor_elements


@pytest.mark.parametrize("elem", scalar_elements)
def test_finite_element(elem):
    """
    Test that the complex proxy FiniteElement is constructed correctly from a real FiniteElement.
    """
    celem = cpx.FiniteElement(elem)

    assert celem.num_sub_elements() == 2

    for ce in celem.sub_elements():
        assert ce == elem


@pytest.mark.parametrize("elem", vector_elements)
def test_vector_element(elem):
    """
    Test that the complex proxy FiniteElement is constructed correctly from a real VectorElement.
    """
    celem = cpx.FiniteElement(elem)

    assert celem.num_sub_elements() == 2*elem.num_sub_elements()

    assert celem._shape == (2, elem.num_sub_elements())

    for ce in celem.sub_elements():
        assert ce == elem.sub_elements()[0]


@pytest.mark.parametrize("elem", tensor_elements)
def test_tensor_element(elem):
    """
    Test that the complex proxy FiniteElement is constructed correctly from a real TensorElement.
    """
    celem = cpx.FiniteElement(elem)

    assert celem.num_sub_elements() == 2*elem.num_sub_elements()

    assert celem._shape == (2,) + elem._shape

    for ce in celem.sub_elements():
        assert ce == elem.sub_elements()[0]


def test_mixed_element():
    """
    Test that the complex proxy FiniteElement is constructed correctly from a real MixedElement.
    """
    mixed_elem = fd.MixedElement(elements)
    celem = cpx.FiniteElement(mixed_elem)

    assert celem.num_sub_elements() == mixed_elem.num_sub_elements()

    for csub, msub in zip(celem.sub_elements(), mixed_elem.sub_elements()):
        assert csub == cpx.FiniteElement(msub)


@pytest.mark.parametrize("elem", elements)
def test_function_space(mesh, elem):
    """
    Test that the proxy complex FunctionSpace is correctly constructed for a scalar real FunctionSpace
    """
    V = fd.FunctionSpace(mesh, elem)
    W = cpx.FunctionSpace(V)

    assert W.ufl_element() == cpx.FiniteElement(elem)


def test_mixed_function_space(mesh):
    """
    Test that the proxy complex FunctionSpace is correctly constructed for a mixed real FunctionSpace
    """
    mixed_elem = fd.MixedElement(elements)

    V = fd.FunctionSpace(mesh, mixed_elem)
    W = cpx.FunctionSpace(V)

    assert len(W.split()) == len(V.split())

    for wcpt, vcpt in zip(W.split(), V.split()):
        assert wcpt == cpx.FunctionSpace(vcpt)


def test_set_get_part_scalar_scalar(mesh):
    """
    Test that the real and imaginary parts are set and get correctly for scalar real FunctionSpace
    """
    x, y = fd.SpatialCoordinate(mesh)

    V = fd.FunctionSpace(mesh, "CG", 1)
    W = cpx.FunctionSpace(V)

    u0 = fd.Function(V)
    u1 = fd.Function(V)
    w = fd.Function(W).assign(0)

    cpx.get_real(w, u0)
    cpx.get_imag(w, u1)

    assert fd.norm(u0) < 1e12
    assert fd.norm(u1) < 1e12

    u0.project(x)

    cpx.set_real(w, u0)
    cpx.get_real(w, u1)

    assert fd.errornorm(u0, u1) < 1e12

    u0.project(-2*y)

    cpx.set_real(w, u0)
    cpx.get_real(w, u1)

    assert fd.errornorm(u0, u1) < 1e12


def test_set_get_part_scalar_vector(mesh):
    """
    Test that the real and imaginary parts are set and get correctly for scalar real FunctionSpace
    """
    x, y = fd.SpatialCoordinate(mesh)

    V = fd.FunctionSpace(mesh, "BDM", 1)
    W = cpx.FunctionSpace(V)

    u0 = fd.Function(V)
    u1 = fd.Function(V)
    w = fd.Function(W).assign(0)

    cpx.get_real(w, u0)
    cpx.get_imag(w, u1)

    assert fd.norm(u0) < 1e12
    assert fd.norm(u1) < 1e12

    u0.project(fd.as_vector([x, y]))

    cpx.set_real(w, u0)
    cpx.get_real(w, u1)

    assert fd.errornorm(u0, u1) < 1e12

    u0.project(fd.as_vector([2*y, -0.5*x]))

    cpx.set_real(w, u0)
    cpx.get_real(w, u1)

    assert fd.errornorm(u0, u1) < 1e12
