
# import firedrake as fd


def FunctionSpace(V):
    """
    Return a FunctionSpace which proxies a complex version of the real FunctionSpace V.

    The returned complex-valued function space has as many components as the real-valued function space, but each component has a 'real' and 'imaginary' component eg:
    Scalar components of the real-valued function space become 2-vector components of the complex-valued space.
    n-vector components of the real-valued function space become 2xn-tensor components of the complex-valued space.

    :arg V: the real-valued function space.
    """
    pass


def DirichletBC(W, bc):
    """
    Return a DirichletBC on the complex FunctionSpace W that is equivalent to the DirichletBC bc on the real FunctionSpace that W was constructed from.

    :arg W: the complex FunctionSpace.
    :arg bc: a DirichletBC on the real FunctionSpace that W was constructed from.
    """
    pass


def real_components(u):
    """
    Return a tuple of the real components of the complex function u.

    :arg u: indexable of function-like objects from the complex FunctionSpace e.g. TestFunctions, TrialFunctions, split(Function).
    """
    pass


def imag_components(u):
    """
    Return a tuple of the imaginary components of the function u.

    :arg u: indexable of function-like objects from the complex FunctionSpace e.g. TestFunctions, TrialFunctions, split(Function).
    """
    pass


def get_real(u, vout=None, name=None):
    """
    Return a real Function equal to the real component of the complex Function u.

    :arg u: a complex Function.
    :arg vout: If a real Function then real component of u is placed here. If None then a new Function is returned.
    :arg name: If vout is None, the name of the new Function. Ignored if uout is not none.
    """
    pass


def get_imag(u, vout=None, name=None):
    """
    Return a real Function equal to the imaginary component of the complex Function u.

    :arg u: a complex Function.
    :arg vout: If a real Function then the imaginary component of u is placed here. If None then a new Function is returned.
    :arg name: If vout is None, the name of the new Function. Ignored if uout is not none.
    """
    pass


def set_real(u, vnew):
    """
    Set the real component of the complex Function u to the value of the real Function v.

    :arg u: a complex Function.
    :arg uout: A real Function.
    """
    pass


def set_imag(u, vnew):
    """
    Set the imaginary component of the complex Function u to the value of the real Function v.

    :arg u: a complex Function.
    :arg uout: A real Function.
    """
    pass


def _get_components(u, i):
    """
    Return a tuple of the real or imaginary components of the function u.

    :arg u: indexable of function-like objects from the complex FunctionSpace e.g. TestFunctions, TrialFunctions, split(Function).
    :arg i: the index of the components, 0 for real or 1 for imaginary.
    """
    pass


def _set_components(u, vnew, i):
    """
    Set the real or imaginary components of the complex Function u to the value of the real Function vnew.

    :arg u: a complex Function.
    :arg v: a real Function.
    :arg i: the index of the components, 0 for real or 1 for imaginary.
    """
    pass


def LinearForm(z, A):
    """
    Return a linear Form on the complex FunctionSpace equal to a complex multiple of a linear Form on the real FunctionSpace.
    If z = zr + i*zi, u = ur + i*ui, and b = br + i*bi, we want to construct a Form such that (zA)u=b

    (zA)u = (zr*A + i*zi*A)(ur + i*ui)
          = (zr*A*ur - zi*A*ui) + i*(zr*A*ui + zi*A*ur)

          = | zr*A   -zi*A | | ur | = | br |
            |              | |    |   |    |
            | zi*A    zr*A | | ui | = | bi |

    :arg z: a complex number.
    :arg A: a generator function for a linear Form on the real FunctionSpace, callable as A(*u, *v) where u and v are TrialFunctions and TestFunctions on the real FunctionSpace.
    """


def NonLinearForm(z, F):
    """
    TODO: Is this valid for anything higher than quadratic nonlinearities?

    Return a nonlinear Form on the complex FunctionSpace equal to a complex multiple of a nonlinear Form on the real FunctionSpace

    If z = zr + i*zi and u = ur + i*ui, we want to construct a Form z*F(u)

    z*F(u) = (zr + i*zi)*(F(ur) + i*F(ui))
           = (zr*F(ur) - zi*F(ui)) + i*(zr*F(ui) + zi*F(ur))

    :arg z: a complex number.
    :arg F: a generator function for a nonlinear Form on the real FunctionSpace, callable as F(*u, *v) where u and v are Functions and TestFunctions on the real FunctionSpace.
    """


def derivative(z, F, u):
    """
    Return a Form equivalent to z*J if J=dF/du is the derivative of the nonlinear Form F with respect to the complex Function u

    :arg z: a complex number.
    :arg F: a generator function for a nonlinear Form on the real FunctionSpace, callable as F(*u, *v) where u and v are Functions and TestFunctions on the real FunctionSpace.
    :arg u: the Function to differentiate F with respect to
    """
