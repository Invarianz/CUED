import numpy as np
import sympy as sp

from numba import njit
from sympy.utilities.lambdify import lambdify

class conditional_njit():
    """
    njit execution only with double precision
    """
    def __init__(self, precision, **kwargs):
        self.precision = precision
        self.kwargs = kwargs

    def __call__(self, func):
        if self.precision in (np.longdouble, np.longcomplex):
            return func
        return njit(func, **self.kwargs)

def matrix_to_njit_functions(sf, hsymbols, dtype=np.complex128, kpflag=False):
    """
    Converts a sympy matrix into a matrix of functions
    """
    shp = sf.shape
    jitmat = [[to_njit_function(sf[j, i], hsymbols, dtype, kpflag=kpflag)
               for i in range(shp[0])] for j in range(shp[1])]
    return jitmat


def list_to_njit_functions(sf, hsymbols, dtype=np.complex128, kpflag=False):
    """
    Converts a list of sympy functions/matrices to a list of numpy
    callable functions/matrices
    """
    return [to_njit_function(sfn, hsymbols, dtype, kpflag) for sfn in sf]


def to_njit_function(sf, hsymbols, dtype=np.complex128, kpflag=False):
    """
    Converts a simple sympy function to a function callable by numpy
    """

    # Standard k variables
    kx, ky = sp.symbols('kx ky', real=True)

    # Decide wheter we need to use the kp version of the program
    if kpflag:
        kxp, kyp = sp.symbols('kxp kyp', real=True)
        return __to_njit_function_kp(sf, hsymbols, kx, ky, kxp, kyp, dtype=dtype)

    return __to_njit_function_k(sf, hsymbols, kx, ky, dtype=dtype)


def __to_njit_function_k(sf, hsymbols, kx, ky, dtype=np.complex128):
    kset = {kx, ky}
    # Check wheter k is contained in the free symbols
    contains_k = bool(sf.free_symbols.intersection(kset))
    if contains_k:
        # All free Hamiltonian symbols get function parameters
        if dtype == np.longcomplex:
            return lambdify(list(hsymbols), sf, np)
        return njit(lambdify(list(hsymbols), sf, np))
    # Here we have non k variables in sf. Expand sf by 0*kx*ky
    sf = sf + kx*ky*sp.UnevaluatedExpr(0)
    if dtype == np.longcomplex:
        return lambdify(list(hsymbols), sf, np)
    return njit(lambdify(list(hsymbols), sf, np))


def __to_njit_function_kp(sf, hsymbols, kx, ky, kxp, kyp, dtype=np.complex128):
    kset = {kx, ky, kxp, kyp}
    hsymbols = hsymbols.union({kxp, kyp})
    # Check wheter k is contained in the free symbols
    contains_k = bool(sf.free_symbols.intersection(kset))
    if contains_k:
        # All free Hamiltonian symbols get function parameters
        if dtype == np.longcomplex:
            return lambdify(list(hsymbols), sf, np)
        return njit(lambdify(list(hsymbols), sf, np))

    sf = sf + kx*ky*kxp*kyp*sp.UnevaluatedExpr(0)
    if dtype == np.longcomplex:
        return lambdify(list(hsymbols), sf, np)
    return njit(lambdify(list(hsymbols), sf, np))


def evaluate_njit_matrix(mjit, kx=np.empty(1), ky=np.empty(1), dtype=np.complex128, **fkwargs):
    shp = np.shape(mjit)
    numpy_matrix = np.empty((np.size(kx),) + shp, dtype=dtype)

    for r in range(shp[0]):
        for c in range(shp[1]):
            numpy_matrix[:, r, c] = mjit[r][c](kx=kx, ky=ky, **fkwargs)

    return numpy_matrix
