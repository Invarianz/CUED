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
        if self.precision in (np.longdouble, np.clongdouble):
            return func
        return njit(func, **self.kwargs)

def matrix_to_njit_functions(
    sf: sp.Matrix,
    hsymbols,
    dtype: type = np.cdouble
):
    """
    Converts a sympy matrix into a matrix of functions
    """
    shp = sf.shape
    jitmat = [[to_njit_function(sf[j, i], hsymbols, dtype)
               for i in range(shp[0])] for j in range(shp[1])]
    return jitmat


def list_to_njit_functions(
    sf,
    hsymbols,
    dtype: type = np.cdouble
):
    """
    Converts a list of sympy functions/matrices to a list of numpy
    callable functions/matrices
    """
    return [to_njit_function(sfn, hsymbols, dtype) for sfn in sf]


def to_njit_function(
    sf,
    hsymbols,
    dtype: type = np.cdouble
):
    """
    Converts a simple sympy function to a function callable by numpy
    """
    # Standard k variables
    kx, ky = sp.symbols('kx ky', real=True)

    return __to_njit_function_k(sf, hsymbols, kx, ky, dtype=dtype)


def __to_njit_function_k(
    sf,
    hsymbols,
    kx: sp.Symbol,
    ky: sp.Symbol,
    dtype: type = np.cdouble
):
    kset = {kx, ky}
    # Check wheter k is contained in the free symbols
    contains_k = bool(sf.free_symbols.intersection(kset))
    if contains_k:
        # All free Hamiltonian symbols get function parameters
        if dtype == np.clongdouble:
            return lambdify(list(hsymbols), sf, np)
        return njit(lambdify(list(hsymbols), sf, np))
    # Here we have missing kx, or ky in sf. Expand sf by 0*kx*ky
    sf = sf + sp.Mul(kx, ky, sp.UnevaluatedExpr(0))
    if dtype == np.longcomplex:
        return lambdify(list(hsymbols), sf, np)
    return njit(lambdify(list(hsymbols), sf, np))

def evaluate_njit_list(
    ljit, 
    kx: np.ndarray = np.empty(1),
    ky: np.ndarray = np.empty(1),
    dtype: type = np.cdouble,
    **fkwargs
):
    n = len(ljit)
    numpy_arr = np.empty((np.size(kx), n), dtype=dtype)

    for i in range(n):
        numpy_arr[:, i] = ljit[i](kx=kx, ky=ky, **fkwargs)

    return numpy_arr

def evaluate_njit_matrix(
    mjit, 
    kx: np.ndarray = np.empty(1),
    ky: np.ndarray = np.empty(1),
    dtype: type = np.cdouble,
    **fkwargs
):
    shp = np.shape(mjit)
    numpy_matrix = np.empty((np.size(kx),) + shp, dtype=dtype)

    for r in range(shp[0]):
        for c in range(shp[1]):
            numpy_matrix[:, r, c] = mjit[r][c](kx=kx, ky=ky, **fkwargs)

    return numpy_matrix