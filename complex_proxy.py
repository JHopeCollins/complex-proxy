
import firedrake as fd

from ufl.classes import MultiIndex, FixedIndex, Indexed

from enum import IntEnum

# flags for real and imaginary parts
Part = IntEnum("Part", (("Real", 0), ("Imag", 1)))
re = Part.Real
im = Part.Imag


def compatible_ufl_elements(elemc, elemr):
    """
    Return whether the ufl element elemc is a complex proxy for real ufl element elemr

    :arg elemc: complex proxy ufl element
    :arg elemr: real ufl element
    """
    return elemc == FiniteElement(elemr)


def FiniteElement(elem):
    """
    Return a UFL FiniteElement which proxies a complex version of the real UFL FiniteElement elem.

    The returned complex-valued element has as many components as the real-valued element, but each component has a 'real' and 'imaginary' part eg:
    Scalar real elements become 2-vector complex elements.
    n-vector real elements become 2xn-tensor complex elements
    (shape)-tensor real elements become (2,shape)-tensor complex elements

    :arg elem: the UFL FiniteElement to be proxied
    """
    if isinstance(elem, fd.TensorElement):  # TensorElement is MixedElement so test first
        return fd.MixedElement([elem, elem])

    elif isinstance(elem, fd.VectorElement):  # VectorElement is MixedElement so test first
        return fd.MixedElement([elem, elem])

    elif isinstance(elem, fd.MixedElement):  # recurse
        return fd.MixedElement([FiniteElement(e) for e in elem.sub_elements()])

    else:
        return fd.MixedElement([elem, elem])


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
    pass


def split(u, i):
    """
    If u is a Coefficient or Argument in the complex FunctionSpace,
        returns a tuple with the function components corresponding
        to the real or imaginary subelements, indexed appropriately.

    :arg u: a Coefficient or Argument in the complex FunctionSpace
    :arg i: Part.Real for real subelements, Part.Imag for imaginary elements
    """
    if not isinstance(i, Part):
        raise ValueError("i must be a Part enum")

    us = fd.split(u)
    vlen = len(us)//2
    return tuple((us[2*j+i] for j in range(vlen)))

    #ncomponents = len(u.function_space().split())

    #if ncomponents == 1:
    #    return tuple((us[i],))

    #def get_sub_element(cpt, i):
    #    part = us[cpt]
    #    idxs = fd.indices(len(part.ufl_shape) - 1)
    #    return fd.as_tensor(Indexed(part, MultiIndex((FixedIndex(i), *idxs))), idxs)

    #return tuple(get_sub_element(cpt, i) for cpt in range(ncomponents))


def subfunctions(u, i):
    """
    Return a tuple of the real or imaginary components of the complex function u.
    Analogous to u.split() (which will be renamed u.subfunctions() at some point).

    :arg u: a complex Function.
    :arg i: the index of the components, Part.Real for real or Part.Imag for imaginary.
    """
    if not isinstance(i, Part):
        raise ValueError("i must be a Part enum")

    us = u.split()
    vlen = len(us)//2
    return tuple((us[2*j+i] for j in range(vlen)))


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


def _get_part(u, vout, i):
    """
    Get the real or imaginary part of the complex Function u and copy it to real Function vout.

    :arg u: a complex Function.
    :arg vout: a real Function.
    :arg i: the index of the components, Part.Real for real or Part.Imag for imaginary.
    """
    if not isinstance(i, Part):
        raise ValueError("i must be a Part enum")

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
    if not isinstance(i, Part):
        raise ValueError("i must be a Part enum")

    if not compatible_ufl_elements(u.ufl_element(), vnew.ufl_element()):
        raise ValueError("u and vnew must be Functions from the complex and real FunctionSpaces")

    for q, p in zip(subfunctions(u, i), vnew.split()):
        q.assign(p)


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

    fr = z.real*f(*vr)
    fi = z.imag*f(*vi)
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

    A11 = z.real*A(*ur, *vr)
    A12 = -z.imag*A(*ui, *vr)
    A21 = z.imag*A(*ur, *vi)
    A22 = z.real*A(*ui, *vi)
    return A11 + A12 + A21 + A22


def NonlinearForm(z, F):
    """
    Return a nonlinear Form on the complex FunctionSpace equal to a complex multiple of a nonlinear Form on the real FunctionSpace

    If z = zr + i*zi and u = ur + i*ui, we want to construct a Form z*F(u)

    z*F(u) = (zr + i*zi)*(F(ur) + i*F(ui))
           = (zr*F(ur) - zi*F(ui)) + i*(zr*F(ui) + zi*F(ur))

    :arg z: a complex number.
    :arg F: a generator function for a nonlinear Form on the real FunctionSpace, callable as F(*u, *v) where u and v are Functions and TestFunctions on the real FunctionSpace.
    """
    raise NotImplementedError("TODO: Is NonlinearForm valid for anything higher than quadratic nonlinearities?")


def derivative(z, F, u):
    """
    Return a bilinear Form equivalent to z*J if J=dF/du is the derivative of the nonlinear Form F with respect to the complex Function u

    :arg z: a complex number.
    :arg F: a generator function for a nonlinear Form on the real FunctionSpace, callable as F(*u, *v) where u and v are Functions and TestFunctions on the real FunctionSpace.
    :arg u: the Function to differentiate F with respect to
    """
    pass
