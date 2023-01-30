
import firedrake as fd

from enum import IntEnum

__all__ = ["FiniteElement", "FunctionSpace", "DirichletBC",
           "split", "subfunctions", "get_real", "get_imag", "set_real", "set_imag",
           "LinearForm", "BilinearForm", "derivative",
           "Part", "re", "im"]

# flags for real and imaginary parts
Part = IntEnum("Part", (("Real", 0), ("Imag", 1)))
re = Part.Real
im = Part.Imag


def _flatten_tree(root, is_leaf, get_children, container=tuple):
    if is_leaf(root):
        return container((root,))
    else:
        return container((leaf
                          for child in get_children(root)
                          for leaf in _flatten_tree(child, is_leaf, get_children)))


def _duplicate_elements(orig, dup=2):
    return type(orig)(dup_elem
                      for elem in orig
                      for dup_elem in (elem for _ in range(dup)))


def FiniteElement(elem):
    """
    Return a UFL FiniteElement which proxies a complex version of the real UFL FiniteElement elem.

    The returned complex-valued element has as many components as the real-valued element, but each component has a 'real' and 'imaginary' part eg:
    Scalar real elements become 2-vector complex elements.
    n-vector real elements become 2xn-tensor complex elements
    (shape)-tensor real elements become (2,shape)-tensor complex elements

    :arg elem: the UFL FiniteElement to be proxied
    """
    flat_elem = _flatten_tree(elem,
                              is_leaf=lambda e: type(e) is not fd.MixedElement,
                              get_children=lambda e: e.sub_elements())

    return fd.MixedElement(_duplicate_elements(flat_elem, 2))


def compatible_ufl_elements(elemc, elemr):
    """
    Return whether the ufl element elemc is a complex proxy for real ufl element elemr

    :arg elemc: complex proxy ufl element
    :arg elemr: real ufl element
    """
    return elemc == FiniteElement(elemr)


def FunctionSpace(V):
    """
    Return a FunctionSpace which proxies a complex version of the real FunctionSpace V.

    The returned complex-valued function space has as many components as the real-valued function space, but each component has a 'real' and 'imaginary' part eg:
    Scalar components of the real-valued function space become 2-vector components of the complex-valued space.
    n-vector components of the real-valued function space become 2xn-tensor components of the complex-valued space.
    (shape)-tensor components of the real-valued function space become (2,shape)-tensor components of the complex-valued space.

    :arg V: the real-valued function space.
    """
    return fd.FunctionSpace(V.mesh(), FiniteElement(V.ufl_element()))


def DirichletBC(W, V, bc):
    """
    Return a DirichletBC on the complex FunctionSpace W that is equivalent to the DirichletBC bc on the real FunctionSpace V that W was constructed from.

    :arg W: the complex FunctionSpace.
    :arg V: the real FunctionSpace that W was constructed from.
    :arg bc: a DirichletBC on the real FunctionSpace that W was constructed from.
    """
    if type(V.ufl_element()) is fd.MixedElement:
        off = 2*bc.function_space().index
    else:
        off = 0

    return tuple((fd.DirichletBC(W.sub(off+i), bc.function_arg, bc.sub_domain)
                  for i in range(2)))


def _component_elements(us, i):
    """
    Return a tuple of the real or imaginary elements of the iterable us

    :arg us: an iterable having the same number of elements as the complex function space
                i.e. twice the number of components as the real function space.
    :arg i: the index of the components, Part.Real for real or Part.Imag for imaginary.
    """
    if not isinstance(i, Part):
        raise TypeError("i must be a Part enum")
    return tuple((us[2*j+i] for j in range(len(us)//2)))


def split(u, i):
    """
    If u is a Coefficient or Argument in the complex FunctionSpace,
        returns a tuple with the function components corresponding
        to the real or imaginary subelements, indexed appropriately.
        Analogous to firedrake.split(u)

    :arg u: a Coefficient or Argument in the complex FunctionSpace
    :arg i: Part.Real for real subelements, Part.Imag for imaginary elements
    """
    return _component_elements(fd.split(u), i)


def subfunctions(u, i):
    """
    Return a tuple of the real or imaginary components of the complex function u.
    Analogous to u.split() (which will be renamed u.subfunctions() at some point).

    :arg u: a complex Function.
    :arg i: the index of the components, Part.Real for real or Part.Imag for imaginary.
    """
    return _component_elements(u.split(), i)


def _get_part(u, vout, i):
    """
    Get the real or imaginary part of the complex Function u and copy it to real Function vout.

    :arg u: a complex Function.
    :arg vout: a real Function.
    :arg i: the index of the components, Part.Real for real or Part.Imag for imaginary.
    """
    if not compatible_ufl_elements(u.ufl_element(), vout.ufl_element()):
        raise ValueError("u and vout must be Functions from the complex and real FunctionSpaces")

    for q, p in zip(subfunctions(u, i), vout.split()):
        p.assign(q)

    return vout


def _set_part(u, vnew, i):
    """
    Set the real or imaginary part of the complex Function u to the value of the real Function vnew.

    :arg u: a complex Function.
    :arg vnew: a real Function.
    :arg i: the index of the components, Part.Real for real or Part.Imag for imaginary.
    """
    if not compatible_ufl_elements(u.ufl_element(), vnew.ufl_element()):
        raise ValueError("u and vnew must be Functions from the complex and real FunctionSpaces")

    for q, p in zip(subfunctions(u, i), vnew.split()):
        q.assign(p)


def get_real(u, vout, name=None):
    """
    Return a real Function equal to the real component of the complex Function u.

    :arg u: a complex Function.
    :arg vout: If a real Function then real component of u is placed here. NotImplementedError(If None then a new Function is returned.)
    :arg name: If vout is None, the name of the new Function. Ignored if vout is not none.
    """
    if vout is None:
        raise NotImplementedError("Inferring real FunctionSpace from complex FunctionSpace not implemented yet")
    return _get_part(u, vout, Part.Real)


def get_imag(u, vout, name=None):
    """
    Return a real Function equal to the imaginary component of the complex Function u.

    :arg u: a complex Function.
    :arg vout: If a real Function then the imaginary component of u is placed here. NotImplementedError(If None then a new Function is returned.)
    :arg name: If vout is None, the name of the new Function. Ignored if uout is not none.
    """
    if vout is None:
        raise NotImplementedError("Inferring real FunctionSpace from complex FunctionSpace not implemented yet")
    return _get_part(u, vout, Part.Imag)


def set_real(u, vnew):
    """
    Set the real component of the complex Function u to the value of the real Function vnew.

    :arg u: a complex Function.
    :arg vnew: A real Function.
    """
    _set_part(u, vnew, Part.Real)


def set_imag(u, vnew):
    """
    Set the imaginary component of the complex Function u to the value of the real Function vnew.

    :arg u: a complex Function.
    :arg vnew: A real Function.
    """
    _set_part(u, vnew, Part.Imag)


def LinearForm(W, z, f):
    """
    Return a Linear Form on the complex FunctionSpace W equal to a complex multiple of a linear Form on the real FunctionSpace.
    If z = zr + i*zi is a complex number, v = vr + i*vi is a complex TestFunction, we want to construct the Form:
    <zr*vr,f> + i<zi*vi,f>

    :arg W: the complex-proxy FunctionSpace
    :arg z: a complex number.
    :arg f: a generator function for a linear Form on the real FunctionSpace, callable as f(*v) where v are TestFunctions on the real FunctionSpace.
    """
    v = fd.TestFunction(W)
    vr = split(v, Part.Real)
    vi = split(v, Part.Imag)

    zr = fd.Constant(z.real)
    zi = fd.Constant(z.imag)

    fr = zr*f(*vr)
    fi = zi*f(*vi)
    return fr + fi


def BilinearForm(W, z, A):
    """
    Return a bilinear Form on the complex FunctionSpace W equal to a complex multiple of a bilinear Form on the real FunctionSpace.
    If z = zr + i*zi is a complex number, u = ur + i*ui is a complex Function, and b = br + i*bi is a complex linear Form, we want to construct a Form such that (zA)u=b

    (zA)u = (zr*A + i*zi*A)(ur + i*ui)
          = (zr*A*ur - zi*A*ui) + i*(zr*A*ui + zi*A*ur)

          = | zr*A   -zi*A | | ur | = | br |
            |              | |    |   |    |
            | zi*A    zr*A | | ui | = | bi |

    :arg W: the complex-proxy FunctionSpace
    :arg z: a complex number.
    :arg A: a generator function for a bilinear Form on the real FunctionSpace, callable as A(*u, *v) where u and v are TrialFunctions and TestFunctions on the real FunctionSpace.
    """
    u = fd.TrialFunction(W)
    v = fd.TestFunction(W)

    ur = split(u, Part.Real)
    ui = split(u, Part.Imag)

    vr = split(v, Part.Real)
    vi = split(v, Part.Imag)

    zr = fd.Constant(z.real)
    zi = fd.Constant(z.imag)

    A11 = zr*A(*ur, *vr)
    A12 = -zi*A(*ui, *vr)
    A21 = zi*A(*ur, *vi)
    A22 = zr*A(*ui, *vi)
    return A11 + A12 + A21 + A22


def derivative(z, F, u):
    """
    Return a bilinear Form equivalent to z*J where z is a complex number, J = dF/dw, F is a nonlinear Form on the real-valued space, and w is a function in the real-valued space. The real and imaginary components of the complex function u most both be equal to w for this operation to be valid.

    If z = zr + i*zi is a complex number, x = xr + i*xi is a complex Function, b = br + i*bi is a complex linear Form, J is the bilinear Form dF/dw, we want to construct a Form such that (zJ)x=b

    (zJ)x = (zr*J + i*zi*J)(xr + i*xi)
          = (zr*J*xr - zi*J*xi) + i*(zr*A*xi + zi*A*xr)

          = | zr*J   -zi*J | | xr | = | br |
            |              | |    |   |    |
            | zi*J    zr*J | | xi | = | bi |

    :arg z: a complex number.
    :arg F: a generator function for a nonlinear Form on the real FunctionSpace, callable as F(*u, *v) where u and v are Functions and TestFunctions on the real FunctionSpace.
    :arg u: the Function to differentiate F with respect to
    """
    W = u.function_space()
    v = fd.TestFunction(W)

    ur = split(u, Part.Real)
    ui = split(u, Part.Imag)

    vr = split(v, Part.Real)
    vi = split(v, Part.Imag)

    Frr = F(*ur, *vr)
    Fri = F(*ur, *vi)
    Fir = F(*ui, *vr)
    Fii = F(*ui, *vi)

    Jrr = fd.derivative(Frr, u)
    Jri = fd.derivative(Fri, u)
    Jir = fd.derivative(Fir, u)
    Jii = fd.derivative(Fii, u)

    zr = fd.Constant(z.real)
    zi = fd.Constant(z.imag)

    A11 = zr*Jrr
    A12 = -zi*Jir
    A21 = zi*Jri
    A22 = zr*Jii

    return A11 + A12 + A21 + A22