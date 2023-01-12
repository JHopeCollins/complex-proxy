
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


complex_numbers = [2+0j, 0+3j, 3+2j]


@pytest.fixture
def nx():
    return 10


@pytest.fixture
def mesh(nx):
    return fd.UnitSquareMesh(nx, nx)


@pytest.fixture
def mixed_element():
    return fd.MixedElement([param.values[0] for param in elements])


@pytest.mark.parametrize("elem", elements)
def test_finite_element(elem):
    """
    Test that the complex proxy FiniteElement is constructed correctly from a real FiniteElement.
    """
    celem = cpx.FiniteElement(elem)

    assert celem.num_sub_elements() == 2

    for ce in celem.sub_elements():
        assert ce == elem


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

    assert len(W.split()) == 2*len(V.split())

    for i in range(V.ufl_element().num_sub_elements()):
        idx_real = 2*i+0
        idx_imag = 2*i+1

        real_elem = W.split()[idx_real].ufl_element()
        imag_elem = W.split()[idx_imag].ufl_element()
        orig_elem = V.split()[i].ufl_element()

        assert real_elem == orig_elem
        assert imag_elem == orig_elem


@pytest.mark.parametrize("elem", scalar_elements+vector_elements)
def test_set_get_part(mesh, elem):
    """
    Test that the real and imaginary parts are set and get correctly from/to real FunctionSpace

    TODO: add tests for tensor_elements
    """
    eps = 1e-12

    x, y = fd.SpatialCoordinate(mesh)

    if elem.reference_value_shape() != ():
        dim = elem.reference_value_shape()[0]
        expr0 = fd.as_vector([x*i for i in range(dim)])
        expr1 = fd.as_vector([-y*i for i in range(dim)])
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

    assert fd.norm(ur) < eps
    assert fd.norm(ui) < eps

    cpx.set_real(w, u0)

    cpx.get_real(w, ur)
    cpx.get_imag(w, ui)

    assert fd.errornorm(u0, ur) < eps
    assert fd.norm(ui) < eps

    cpx.set_imag(w, u1)

    cpx.get_real(w, ur)
    cpx.get_imag(w, ui)

    assert fd.errornorm(u0, ur) < eps
    assert fd.errornorm(u1, ui) < eps


@pytest.mark.parametrize("elem", scalar_elements+vector_elements)
@pytest.mark.parametrize("z", complex_numbers)
def test_linear_form(mesh, elem, z):
    """
    Test that the linear Form is constructed correctly
    """
    eps = 1e-12

    V = fd.FunctionSpace(mesh, elem)
    W = cpx.FunctionSpace(V)

    x, y = fd.SpatialCoordinate(mesh)

    f = x*x-y
    if elem.reference_value_shape() != ():
        vec_expr = [x*x-y, y+x, -y-0.5*x]
        dim = elem.reference_value_shape()[0]
        f = fd.as_vector(vec_expr[:dim])
    else:
        f = x*x-y

    def L(v):
        return fd.inner(f, v)*fd.dx

    v = fd.TestFunction(V)
    rhs = fd.assemble(L(v))

    ur = fd.Function(V)
    ui = fd.Function(V)
    w = fd.Function(W)

    w = fd.assemble(cpx.LinearForm(W, z, L))

    cpx.get_real(w, ur)
    cpx.get_imag(w, ui)

    zr = z.real
    zi = z.imag
    assert fd.errornorm(zr*rhs, ur) < eps
    assert fd.errornorm(zi*rhs, ui) < eps


def test_bilinear_form(mesh):
    """
    Test that the bilinear form is constructed correctly
    """
    V = fd.FunctionSpace(mesh, "CG", 1)
    W = cpx.FunctionSpace(V)

    x, y = fd.SpatialCoordinate(mesh)
    f = fd.Function(V).interpolate(x*x-y)

    def form_function(u, v):
        return fd.inner(fd.grad(u), fd.grad(v))*fd.dx

    u = fd.TrialFunction(V)
    v = fd.TestFunction(V)

    a = form_function(u, v)

    # the real value
    b = fd.assemble(fd.action(a, f))

    # complex vector to multiply
    g = fd.Function(W)

    cpx.set_real(g, f)
    f.assign(2*f)
    cpx.set_imag(g, f)

    # real/imag parts of mat-vec product
    br = fd.Function(V)
    bi = fd.Function(V)

    # non-zero only on diagonal blocks: real and imag parts independent
    zr = 3+0j
    wr = fd.Function(W)

    K = cpx.BilinearForm(W, zr, form_function)
    fd.assemble(fd.action(K, g), tensor=wr)

    cpx.get_real(wr, br)
    cpx.get_imag(wr, bi)

    assert fd.errornorm(3*1*b, br) < 1e-12
    assert fd.errornorm(3*2*b, bi) < 1e-12

    # non-zero only on off-diagonal blocks: real and imag parts independent
    zi = 0+4j

    wi = fd.Function(W)

    K = cpx.BilinearForm(W, zi, form_function)
    fd.assemble(fd.action(K, g), tensor=wi)

    cpx.get_real(wi, br)
    cpx.get_imag(wi, bi)

    assert fd.errornorm(-4*2*b, br) < 1e-12
    assert fd.errornorm(4*1*b, bi) < 1e-12

    # non-zero in all blocks:
    z = zr + zi

    wz = fd.Function(W)

    K = cpx.BilinearForm(W, z, form_function)
    fd.assemble(fd.action(K, g), tensor=wz)

    cpx.get_real(wz, br)
    cpx.get_imag(wz, bi)

    # mat-vec multiplication should be linear
    br_check = fd.Function(V)
    bi_check = fd.Function(V)

    wz.assign(wr + wi)

    cpx.get_real(wz, br_check)
    cpx.get_imag(wz, bi_check)

    assert fd.errornorm(br_check, br) < 1e-12
    assert fd.errornorm(bi_check, bi) < 1e-12
