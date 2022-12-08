
import firedrake as fd
import complex_proxy as cpx

import pytest


cell = fd.Cell('triangle')

scalar_elements = [
    pytest.param(fd.FiniteElement("CG", cell, 1), id="CG1"),
    pytest.param(fd.FiniteElement("BDM", cell, 2), id="BDM1")
]

vector_elements = [
    pytest.param(fd.VectorElement("DG", cell, 1), id="VectorDG1"),
    pytest.param(fd.VectorElement("DG", cell, 1, dim=3), id="VectorDG1_3D")
]

tensor_elements = [
    pytest.param(fd.TensorElement("Lagrange", cell, 1), id="TensorL1"),
    pytest.param(fd.TensorElement("Lagrange", cell, 1, shape=(2, 3, 4)), id="TensorL1_234D")
]

elements = scalar_elements + vector_elements + tensor_elements


@pytest.fixture
def nx():
    return 10


@pytest.fixture
def mesh(nx):
    return fd.UnitSquareMesh(nx, nx)


@pytest.fixture
def mixed_element():
    return fd.MixedElement([param.values[0] for param in elements])


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


def test_mixed_element(mixed_element):
    """
    Test that the complex proxy FiniteElement is constructed correctly from a real MixedElement.
    """

    celem = cpx.FiniteElement(mixed_element)

    assert celem.num_sub_elements() == mixed_element.num_sub_elements()

    for csub, msub in zip(celem.sub_elements(), mixed_element.sub_elements()):
        assert csub == cpx.FiniteElement(msub)


@pytest.mark.parametrize("elem", elements)
def test_function_space(mesh, elem):
    """
    Test that the proxy complex FunctionSpace is correctly constructed for a scalar real FunctionSpace
    """
    V = fd.FunctionSpace(mesh, elem)
    W = cpx.FunctionSpace(V)

    assert W.ufl_element() == cpx.FiniteElement(elem)


def test_mixed_function_space(mesh, mixed_element):
    """
    Test that the proxy complex FunctionSpace is correctly constructed for a mixed real FunctionSpace
    """

    V = fd.FunctionSpace(mesh, mixed_element)
    W = cpx.FunctionSpace(V)

    assert len(W.split()) == len(V.split())

    for wcpt, vcpt in zip(W.split(), V.split()):
        assert wcpt == cpx.FunctionSpace(vcpt)


@pytest.mark.parametrize("elem", scalar_elements)
def test_set_get_part_scalar(mesh, elem):
    """
    Test that the real and imaginary parts are set and get correctly for scalar real FunctionSpace
    """
    x, y = fd.SpatialCoordinate(mesh)

    if elem.reference_value_shape() != ():
        dim = elem.reference_value_shape()[0]
        expr0 = fd.as_vector([x*i for i in range(dim)])
        expr1 = fd.as_vector([-y*i for i in range(dim)])
        assert elem.family() != fd.FiniteElement("CG", cell, 1)
    else:
        expr0 = x
        expr1 = -2*y

    V = fd.FunctionSpace(mesh, elem)
    W = cpx.FunctionSpace(V)

    u0 = fd.Function(V).project(expr0)
    u1 = fd.Function(V).project(expr1)
    ur = fd.Function(V).assign(1)
    ui = fd.Function(V).assign(1)
    w = fd.Function(W).assign(0)

    cpx.get_real(w, ur)
    cpx.get_imag(w, ui)

    assert fd.norm(ur) < 1e12
    assert fd.norm(ui) < 1e12

    cpx.set_real(w, u0)

    cpx.get_real(w, ur)
    cpx.get_imag(w, ui)

    assert fd.errornorm(u0, ur) < 1e12
    assert fd.norm(ui) < 1e12

    cpx.set_imag(w, u1)

    cpx.get_real(w, ur)
    cpx.get_imag(w, ui)

    assert fd.errornorm(u0, ur) < 1e12
    assert fd.errornorm(u1, ui) < 1e12


@pytest.mark.parametrize("elem", vector_elements)
def test_set_get_part_vector(mesh, elem):
    """
    Test that the real and imaginary parts are set and get correctly for real VectorFunctionSpace
    """
    x, y = fd.SpatialCoordinate(mesh)

    V = fd.FunctionSpace(mesh, elem)
    W = cpx.FunctionSpace(V)

    vec0 = fd.as_vector([x*i for i in range(elem.num_sub_elements())])
    vec1 = fd.as_vector([-y*i for i in range(elem.num_sub_elements())])

    u0 = fd.Function(V).project(vec0)
    u1 = fd.Function(V).project(vec1)
    ur = fd.Function(V).assign(1)
    ui = fd.Function(V).assign(1)
    w = fd.Function(W).assign(0)

    cpx.get_real(w, ur)
    cpx.get_imag(w, ui)

    assert fd.norm(ur) < 1e12
    assert fd.norm(ui) < 1e12

    cpx.set_real(w, u0)

    cpx.get_real(w, ur)
    cpx.get_imag(w, ui)

    assert fd.errornorm(u0, ur) < 1e12
    assert fd.norm(ui) < 1e12

    cpx.set_imag(w, u1)

    cpx.get_real(w, ur)
    cpx.get_imag(w, ui)

    assert fd.errornorm(u0, ur) < 1e12
    assert fd.errornorm(u1, ui) < 1e12


@pytest.mark.skip
def test_bilinear_form(mesh):
    """
    Test that the bilinear form is constructed correctly
    """
    V = fd.FunctionSpace(mesh, "CG", 1)
    W = cpx.FunctionSpace(V)

    def form_function(u, v):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

    # non-zero only on diagonal blocks: real and imag parts independent
    re = 1+0j

    K = cpx.BilinearForm(W, re, form_function)

    # make rhs same for both

    # solve

    # check both have same correct answer

    # different rhs for real and imaginary

    # check both have different correct answer
    # non-zero only on off-diagonal blocks: real and imag parts independent and use rhs of opposite part
    im = 1+0j

    K = cpx.BilinearForm(W, im, form_function)

    # make rhs same for both

    # solve

    # check both have same correct answer

    # different rhs for real and imaginary

    # check both have different correct answer

    # check both have different correct answer

    # non-zero on all blocks: solution should be linear combination of solutions of two previous problems
    z = 1+1j

    K = cpx.BilinearForm(W, z, form_function)

    # make rhs same for both

    # solve

    # check both have correct answer

    # different rhs for real and imaginary

    # check both have correct answer
