import numpy as np
import sympy as sp

from typing import cast, List, Optional, Tuple, Union

from cued.utility.njit import (list_to_njit_functions, matrix_to_njit_functions,
                               evaluate_njit_list, evaluate_njit_matrix)

class TwoBandHamiltonianSystem():
    so = sp.Matrix([[1, 0], [0, 1]])
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.Matrix([[1, 0], [0, -1]])

    kx = sp.Symbol('kx', real=True)
    ky = sp.Symbol('ky', real=True)

    m_zee_x = sp.Symbol('m_zee_x', real=True)
    m_zee_y = sp.Symbol('m_zee_y', real=True)
    m_zee_z = sp.Symbol('m_zee_z', real=True)

    def __init__(
        self,
        ho: Union[int, float, sp.Symbol],
        hx: Union[int, float, sp.Symbol],
        hy: Union[int, float, sp.Symbol],
        hz: Union[int, float, sp.Symbol]
    ):
        """
        Generates the symbolic Hamiltonian, wave functions and energies.
        """
        self.bands = 2

        self.ho = ho
        self.hx = hx
        self.hy = hy
        self.hz = hz

        # Hamiltonian & Hamiltonian derivatives
        self.h = sp.Add(sp.Mul(self.ho, self.so),
                        sp.Mul(self.hx, self.sx),
                        sp.Mul(self.hy, self.sy),
                        sp.Mul(self.hz, self.sz))
        self.hderiv = [sp.diff(self.h, self.kx), sp.diff(self.h, self.ky)]

        # Energies (e_v, e_c) & Energy derivatives (de_v/dkx, de_v/dky, de_c/dkx, de_c/dky)
        self.e_soc =\
            sp.sqrt(sp.Add(sp.Pow(self.hx, 2), sp.Pow(self.hy, 2), sp.Pow(self.hz, 2)))
        self.e = [sp.Add(self.ho, -self.e_soc), sp.Add(self.ho, self.e_soc)]
        self.ederiv = []
        for e in self.e:
            self.ederiv.append(sp.diff(e, self.kx))
            self.ederiv.append(sp.diff(e, self.ky))

        # Get set when eigensystem is called (gauge needed)
        self.U = None             # Normalised eigenstates
        self.U_h = None           # Hermitian conjugate
        self.U_no_norm = None     # Unnormalised eigenstates
        self.U_h_no_norm = None   # Hermitian conjugate
        self.__eigensystem_called = False

        # Jit functions for calculating energies and energy derivatives
        self.e_jit = None
        self.ederiv_jit = None

        self.U_jit = None
        self.U_h_jit = None

        self.Ax_jit = None
        self.Ay_jit = None
        
        self.B_jit = None

        self.e_in_path = None   #set when eigensystem_dipole_path is called

        self.dipole_path_x = None
        self.dipole_path_y = None

        self.dipole_in_path = None
        self.dipole_ortho = None

        self.dipole_derivative = None
        self.dipole_derivative_jit = None
        self.dipole_derivative_in_path = None

    def make_eigensystem(
        self,
        gidx: Optional[Union[int, float]] = 1
    ):
        """
        Creates (symbolic) wave functions, berry connection and berry curvature.
        """

        def up_down_gauge():
            wfv = sp.Matrix([sp.Add(-self.hx, sp.Mul(sp.I, self.hy)),
                             sp.Add(self.hz, self.e_soc)])
            wfc = sp.Matrix([sp.Add(self.hz, self.e_soc),
                             sp.Add(self.hx, sp.Mul(sp.I, self.hy))])
            wfv_h = sp.Matrix([sp.Add(-self.hx, -sp.Mul(sp.I, self.hy)),
                               sp.Add(self.hz, self.e_soc)])
            wfc_h = sp.Matrix([sp.Add(self.hz, self.e_soc),
                               sp.Add(self.hx, -sp.Mul(sp.I, self.hy))])
            # Analytical norms
            # Equals normc_sq
            normv_sq = sp.Mul(2, sp.Add(self.e_soc, self.hz), self.e_soc) 

            normv = sp.sqrt(normv_sq)
            normc = sp.sqrt(normv_sq)

            # SymPy calculated norms
            # normv_sq = wfv_h.dot(wfv)
            # normc_sq = wfc_h.dot(wfc)

            # normv = sp.sqrt(normv_sq)
            # normc = sp.sqrt(normc_sq)

            return wfv, wfc, wfv_h, wfc_h, normv, normc

        def up_gauge():
            wfv = sp.Matrix([sp.Add(self.hz, -self.e_soc),
                             sp.Add(self.hx, sp.Mul(sp.I, self.hy))])
            wfc = sp.Matrix([sp.Add(self.hz, self.e_soc),
                             sp.Add(self.hx, sp.Mul(sp.I, self.hy))])
            wfv_h = sp.Matrix([sp.Add(self.hz, -self.e_soc),
                               sp.Add(self.hx, -sp.Mul(sp.I, self.hy))])
            wfc_h = sp.Matrix([sp.Add(self.hz, self.e_soc),
                               sp.Add(self.hx, -sp.Mul(sp.I, self.hy))])
            # Analytical norms
            # normv_sq = sp.Mul(2, sp.Add(self.e_soc, -self.hz), self.e_soc)
            # normc_sq = sp.Mul(2, sp.Add(self.e_soc, self.hz), self.e_soc)

            # normv = sp.sqrt(normv_sq)
            # normc = sp.sqrt(normc_sq)

            # SymPy calculated norms
            normv_sq = wfv_h.dot(wfv)
            normc_sq = wfc_h.dot(wfc)

            normv = sp.sqrt(normv_sq)
            normc = sp.sqrt(normc_sq)

            return wfv, wfc, wfv_h, wfc_h, normv, normc

        def down_gauge():
            wfv = sp.Matrix([sp.Add(-self.hx, sp.Mul(sp.I, self.hy)),
                             sp.Add(self.hz, self.e_soc)])
            wfc = sp.Matrix([sp.Add(-self.hx, sp.Mul(sp.I, self.hy)),
                             sp.Add(self.hz, -self.e_soc)])
            wfv_h = sp.Matrix([sp.Add(-self.hx, -sp.Mul(sp.I, self.hy)),
                               sp.Add(self.hz, self.e_soc)])
            wfc_h = sp.Matrix([sp.Add(-self.hx, -sp.Mul(sp.I, self.hy)),
                               sp.Add(self.hz, -self.e_soc)])
            # Analytical norms
            # normv_sq = sp.Mul(2, sp.Add(self.e_soc, self.hz), self.e_soc)
            # normc_sq = sp.Mul(2, sp.Add(self.e_soc, -self.hz), self.e_soc)

            # normv = sp.sqrt(sp.Mul(2, sp.Add(self.e_soc, self.hz), self.e_soc))
            # normc = sp.sqrt(sp.Mul(2, sp.Add(self.e_soc, -self.hz), self.e_soc))

            # SymPy calculated norms
            normv_sq = wfv_h.dot(wfv)
            normc_sq = wfc_h.dot(wfc)

            normv = sp.sqrt(normv_sq)
            normc = sp.sqrt(normc_sq)

            return wfv, wfc, wfv_h, wfc_h, normv, normc

        def mixed_gauge(gidx):
            wfv_up, wfc_up, wfv_up_h, wfc_up_h, _, _ = up_gauge()
            wfv_do, wfc_do, wfv_do_h, wfc_do_h, _, _ = down_gauge()

            wfv = (1-gidx)*wfv_up + gidx*wfv_do
            wfc = (1-gidx)*wfc_up + gidx*wfc_do
            wfv_h = (1-gidx)*wfv_up_h + gidx*wfv_do_h
            wfc_h = (1-gidx)*wfc_up_h + gidx*wfc_do_h
            normv_sq = wfv_h.dot(wfv)
            normc_sq = wfc_h.dot(wfc)
            normv = sp.sqrt(normv_sq)
            normc = sp.sqrt(normc_sq)

            return wfv, wfc, wfv_h, wfc_h, normv, normc

        if gidx is None:
            wfv, wfc, wfv_h, wfc_h, normv, normc = up_down_gauge()
        elif gidx == 1:
            wfv, wfc, wfv_h, wfc_h, normv, normc = down_gauge()
        elif gidx == 0:
            wfv, wfc, wfv_h, wfc_h, normv, normc = up_gauge()
        elif 0 < gidx < 1:
            wfv, wfc, wfv_h, wfc_h, normv, normc = mixed_gauge(gidx)
        else:
            raise RuntimeError("gidx needs to be between 0 and 1 or None")

        self.U = (wfv/normv).row_join(wfc/normc)
        self.U_h = (wfv_h/normv).T.col_join((wfc_h/normc).T)

        # Create Berry connection
        # Minus sign is the charge
        self.Ax = -sp.I * self.U_h * sp.diff(self.U, self.kx)
        self.Ay = -sp.I * self.U_h * sp.diff(self.U, self.ky)

        # Create Berry curvature
        self.B = sp.Add(sp.diff(self.Ax, self.ky), -sp.diff(self.Ay, self.kx))
        self.__eigensystem_called = True

    def make_eigensystem_jit(
        self,
        dtype: type = np.cdouble
    ):
        """
        Create callable compiled ("jit'ed") functions for all symbols i.e.
        Hamiltonian, Energies, Energy derivatives, Berry Connection & Curvature
        """
        if not self.__eigensystem_called:
            # If this is called we can assume the following symbols to be set
            # i.e. we cast them to the correct type
            raise RuntimeError("Eigensystem method needs to be called first."
                               "Wave function gauge needs manual setting.")

        # Assume Matrix and Symbols since eigensystem was called
        self.h = cast(sp.Matrix, self.h)
        self.hderiv = cast(List[sp.Matrix], self.hderiv)
        # Jitted Hamiltonian and Hamiltonian derivatives
        self.h_jit =\
            matrix_to_njit_functions(self.h, self.h.free_symbols, dtype=dtype)
        self.hderiv_jit =\
            [matrix_to_njit_functions(hd, self.h.free_symbols, dtype=dtype)
             for hd in self.hderiv]

        # Assume Symbols since eigensystem was called
        self.e = cast(List[sp.Symbol], self.e)
        self.ederiv = cast(List[sp.Symbol], self.ederiv)
        # Jitted Energies and Energy derivatives
        self.e_jit =\
            list_to_njit_functions(self.e, self.h.free_symbols, dtype=dtype)
        self.ederiv_jit =\
            list_to_njit_functions(self.ederiv, self.h.free_symbols, dtype=dtype)

        # Assume Matrix since eigensystem was called
        self.U = cast(sp.Matrix, self.U)
        self.U_h = cast(sp.Matrix, self.U_h)
        # Jitted Wave functions and Wave function derivatives
        self.U_jit =\
            matrix_to_njit_functions(self.U, self.h.free_symbols, dtype=dtype)
        self.U_h_jit =\
            matrix_to_njit_functions(self.U_h, self.h.free_symbols, dtype=dtype)

        # Assume Matrix since eigensystem was called
        self.Ax = cast(sp.Matrix, self.Ax)
        self.Ay = cast(sp.Matrix, self.Ay)
        # Jitted Berry Connection
        self.Ax_jit =\
            matrix_to_njit_functions(self.Ax, self.h.free_symbols, dtype=dtype)
        self.Ay_jit =\
            matrix_to_njit_functions(self.Ay, self.h.free_symbols, dtype=dtype)

        # Assume Matrix since eigensystem was called
        self.B = cast(sp.Matrix, self.B)
        # Jitted Berry Curvature
        self.B_jit =\
            matrix_to_njit_functions(self.B, self.h.free_symbols, dtype=dtype)

    def evaluate_energy(
        self,
        kx: np.ndarray,
        ky: np.ndarray,
        dtype=np.double,
        **fkwargs
    ) -> np.ndarray:
        '''
        Evaluate energy bands at all give k-points
        '''
        return evaluate_njit_list(self.e_jit, kx=kx, ky=ky, dtype=dtype, **fkwargs)

    def evaluate_ederivative(
        self,
        kx: np.ndarray,
        ky: np.ndarray,
        dtype=np.double,
        **fkwargs
    ) -> np.ndarray:
        return evaluate_njit_list(self.ederiv_jit, kx=kx, ky=ky, dtype=dtype, **fkwargs)

    def evaluate_wavefunction(
        self,
        kx: np.ndarray,
        ky: np.ndarray,
        dtype=np.cdouble,
        **fkwargs
    ) -> Tuple[np.ndarray, np.ndarray]:

        U_eval = evaluate_njit_matrix(self.U_jit, kx, ky, dtype=dtype, **fkwargs)
        U_h_eval = evaluate_njit_matrix(self.U_h_jit, kx, ky, dtype=dtype, **fkwargs)
        return U_eval, U_h_eval

    def evaluate_dipole(
        self,
        kx: np.ndarray,
        ky: np.ndarray,
        dtype=np.cdouble,
        **fkwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Evaluate all kpoints without BZ
        Ax_eval = evaluate_njit_matrix(self.Ax_jit, kx, ky, dtype=dtype, **fkwargs)
        Ay_eval = evaluate_njit_matrix(self.Ay_jit, kx, ky, dtype=dtype, **fkwargs)
        return Ax_eval, Ay_eval

    def evaluate_curvature(
        self,
        kx: np.ndarray,
        ky: np.ndarray,
        dtype=np.cdouble,
        **fkwargs
    ) -> np.ndarray:
        # Evaluate all kpoints without BZ
        return evaluate_njit_matrix(self.B_jit, kx, ky, dtype=dtype, **fkwargs)

## Calling method deprecated
#     def eigensystem_dipole_path(
#         self,
#         path,
#         E_dir,
#         E_ort,
#         bands,
#         dynamics_method,
#         dtype_real,
#         dtype_complex
#     ):
# 
#         # Retrieve the set of k-points for the current path
#         kx_in_path = path[:, 0]
#         ky_in_path = path[:, 1]
#         pathlen = path[:, 0].size
# 
#         if dynamics_method == 'semiclassics':
#             self.dipole_path_x = np.zeros([bands, bands, pathlen], dtype=dtype_complex)
#             self.dipole_path_y = np.zeros([bands, bands, pathlen], dtype=dtype_complex)
#             self.Ax_path = evaluate_njit_matrix(self.Ax_jit, kx=kx_in_path, ky=ky_in_path, dtype=dtype_complex)
#             self.Ay_path = evaluate_njit_matrix(self.Ay_jit, kx=kx_in_path, ky=ky_in_path, dtype=dtype_complex)
#             self.Bcurv = evaluate_njit_matrix(self.B_jit, kx=kx_in_path, ky=ky_in_path, dtype=dtype_complex)
#         else:
#             # Calculate the dipole components along the path
#             self.dipole_path_x = evaluate_njit_matrix(self.Ax_jit, kx=kx_in_path, ky=ky_in_path, dtype=dtype_complex)
#             self.dipole_path_y = evaluate_njit_matrix(self.Ay_jit, kx=kx_in_path, ky=ky_in_path, dtype=dtype_complex)
# 
#         # Evaluate energies for each band
#         self.e_in_path = evaluate_njit_list(self.e_jit, kx=kx_in_path, ky=ky_in_path, dtype=dtype_real)
# 
#         self.dipole_in_path = E_dir[0]*self.dipole_path_x + E_dir[1]*self.dipole_path_y
#         self.dipole_ortho = E_ort[0]*self.dipole_path_x + E_ort[1]*self.dipole_path_y
# 
#         if dynamics_method == 'EEA':
#             self.dipole_derivative_in_path = evaluate_njit_matrix(self.dipole_derivative_jit, kx=kx_in_path, ky=ky_in_path, dtype=dtype_complex)
# 
# #        if P.dm_dynamics_method == 'EEA':
# #            self.dipole_derivative = P.E_dir[0] * P.E_dir[0] * sp.diff(self.Ax, self.kx) \
# #                + P.E_dir[0] * P.E_dir[1] * (sp.diff(self.Ax, self.ky) + sp.diff(self.Ay, self.kx)) \
# #                + P.E_dir[1] * P.E_dir[1] * sp.diff(self.Ay, self.ky)
# #            self.dipole_derivative_jit = matrix_to_njit_functions(self.dipole_derivative, self.h.free_symbols, dtype=P.type_complex_np)
# 