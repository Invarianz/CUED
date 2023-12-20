import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from typing import List, Union

from cued.utility.njit import (list_to_njit_functions, matrix_to_njit_functions,
                               evaluate_njit_matrix)

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
        self.h = self.ho*self.so + self.hx*self.sx + self.hy*self.sy + self.hz*self.sz
        self.hderiv = [sp.diff(self.h, self.kx), sp.diff(self.h, self.ky)]

        # Energies (e_v, e_c) & Energy derivatives (de_v/dkx, de_v/dky, de_c/dkx, de_c/dky)
        self.e_soc = sp.sqrt(self.hx**2 + self.hy**2 + self.hz**2) # type: ignore
        self.e = [self.ho - self.e_soc, self.ho + self.e_soc] # type: ignore
        self.ederiv = []
        for e in self.e:
            self.ederiv.append(sp.diff(e, self.kx))
            self.ederiv.append(sp.diff(e, self.ky))

        # Get set when eigensystem is called (gauge needed)
        self.U = None             # Normalised eigenstates
        self.U_h = None           # Hermitian conjugate
        self.U_no_norm = None     # Unnormalised eigenstates
        self.U_h_no_norm = None   # Hermitian conjugate

        # Jit functions for calculating energies and energy derivatives
        self.e_jit = None
        self.ederiv_jit = None

        self.U_jit = None
        self.U_h_jit = None

        # Evaluated fields
        self.Ax_eval = None
        self.Ay_eval = None

        self.B_eval = None

        # Assign when evaluate_energy is called
        self.e_eval = None
        self.ederiv_eval = None

        self.e_in_path = None   #set when eigensystem_dipole_path is called

        self.dipole_path_x = None
        self.dipole_path_y = None

        self.dipole_in_path = None
        self.dipole_ortho = None

        self.dipole_derivative = None
        self.dipole_derivative_jit = None
        self.dipole_derivative_in_path = None

    def eigensystem_dipole_path(self, path, P):

        # Set eigenfunction first time eigensystem_dipole_path is called
        if self.e is None:
            self.make_eigensystem_dipole(P)

        # Retrieve the set of k-points for the current path
        kx_in_path = path[:, 0]
        ky_in_path = path[:, 1]
        pathlen = path[:,0].size
        self.e_in_path = np.zeros([pathlen, P.bands], dtype=P.type_real_np)

        if P.dm_dynamics_method == 'semiclassics':
            self.dipole_path_x = np.zeros([pathlen, P.bands, P.bands], dtype=P.type_complex_np)
            self.dipole_path_y = np.zeros([pathlen, P.bands, P.bands], dtype=P.type_complex_np)
            self.Ax_path = evaluate_njit_matrix(self.Axfjit, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)
            self.Ay_path = evaluate_njit_matrix(self.Ayfjit, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)
            self.Bcurv = evaluate_njit_matrix(self.Bfjit, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)

        else:
            # Calculate the dipole components along the path
            self.dipole_path_x = evaluate_njit_matrix(self.Axfjit, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)
            self.dipole_path_y = evaluate_njit_matrix(self.Ayfjit, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)

        # Evaluate energies for each band
        for n, e_jit in enumerate(self.e_jit):
            self.e_in_path[:, n] = e_jit(kx=kx_in_path, ky=ky_in_path)

        self.dipole_in_path = P.E_dir[0]*self.dipole_path_x + P.E_dir[1]*self.dipole_path_y
        self.dipole_ortho = P.E_ort[0]*self.dipole_path_x + P.E_ort[1]*self.dipole_path_y

        if P.dm_dynamics_method == 'EEA':
            self.dipole_derivative_in_path = evaluate_njit_matrix(self.dipole_derivative_jit, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)


    def make_eigensystem_jit(
        self,
        dtype: np.dtype
    ):
        """
        Create callable compiled ("jit'ed") functions for all symbols i.e.
        Hamiltonian, Energies, Energy derivatives, Berry Connection & Curvature
        """
        # Jitted Hamiltonian and energies
        self.h_jit =\
            matrix_to_njit_functions(self.h, self.h.free_symbols, dtype=dtype)
        self.hderiv_jit =\
            [matrix_to_njit_functions(hd, self.h.free_symbols, dtype=dtype)
             for hd in self.hderiv]

        self.e_jit =\
            list_to_njit_functions(self.e, self.h.free_symbols, dtype=dtype)
        self.ederiv_jit =\
            list_to_njit_functions(self.ederiv, self.h.free_symbols, dtype=dtype)

        # Njit function and function arguments
        self.Ax_jit =\
            matrix_to_njit_functions(self.Ax, self.h.free_symbols, dtype=dtype)
        self.Ay_jit =\
            matrix_to_njit_functions(self.Ay, self.h.free_symbols, dtype=dtype)

        # Curvature
        self.B_jit =\
            matrix_to_njit_functions(self.B, self.h.free_symbols, dtype=dtype)

#        if P.dm_dynamics_method == 'EEA':
#            self.dipole_derivative = P.E_dir[0] * P.E_dir[0] * sp.diff(self.Ax, self.kx) \
#                + P.E_dir[0] * P.E_dir[1] * (sp.diff(self.Ax, self.ky) + sp.diff(self.Ay, self.kx)) \
#                + P.E_dir[1] * P.E_dir[1] * sp.diff(self.Ay, self.ky)
#            self.dipole_derivative_jit = matrix_to_njit_functions(self.dipole_derivative, self.h.free_symbols, dtype=P.type_complex_np)

    def eigensystem(
        self,
        gidx: Union[int, float]
    ):
        """
        Generic form of Hamiltonian, energies and wave functions in a two band
        Hamiltonian.
        Creates (symbolic) wave functions, berry connection and berry curvature.
        """

        if gidx is None:
            wfv = sp.Matrix([-self.hx + sp.I*self.hy, self.hz + self.e_soc])
            wfc = sp.Matrix([self.hz + self.e_soc, self.hx + sp.I*self.hy])
            wfv_h = sp.Matrix([-self.hx - sp.I*self.hy, self.hz + self.e_soc])
            wfc_h = sp.Matrix([self.hz + self.e_soc, self.hx - sp.I*self.hy])
            normv = sp.sqrt(2*(self.e_soc + self.hz)*self.e_soc)
            normc = sp.sqrt(2*(self.e_soc + self.hz)*self.e_soc)
        elif 0 <= gidx <= 1:
            wfv_up = sp.Matrix([self.hz - self.e_soc, (self.hx+sp.I*self.hy)])
            wfc_up = sp.Matrix([self.hz + self.e_soc, (self.hx+sp.I*self.hy)])
            wfv_up_h = sp.Matrix([self.hz-self.e_soc, (self.hx-sp.I*self.hy)])
            wfc_up_h = sp.Matrix([self.hz+self.e_soc, (self.hx-sp.I*self.hy)])

            wfv_do = sp.Matrix([-self.hx+sp.I*self.hy, self.hz+self.e_soc])
            wfc_do = sp.Matrix([-self.hx+sp.I*self.hy, self.hz-self.e_soc])
            wfv_do_h = sp.Matrix([-self.hx-sp.I*self.hy, self.hz+self.e_soc])
            wfc_do_h = sp.Matrix([-self.hx-sp.I*self.hy, self.hz-self.e_soc])

            wfv = (1-gidx)*wfv_up + gidx*wfv_do
            wfc = (1-gidx)*wfc_up + gidx*wfc_do
            wfv_h = (1-gidx)*wfv_up_h + gidx*wfv_do_h
            wfc_h = (1-gidx)*wfc_up_h + gidx*wfc_do_h
            normv = sp.sqrt(wfv_h.dot(wfv))
            normc = sp.sqrt(wfc_h.dot(wfc))
        else:
            raise RuntimeError("gidx needs to be between 0 and 1 or None")

        self.U = (wfv/normv).row_join(wfc/normc)
        self.U_h = (wfv_h/normv).T.col_join((wfc_h/normc).T)

        self.U_no_norm = (wfv).row_join(wfc)
        self.U_h_no_norm = (wfv_h).T.col_join(wfc_h.T)

        # Create Berry connection
        # Minus sign is the charge
        self.Ax = -sp.I * self.U_h * sp.diff(self.U, self.kx)
        self.Ay = -sp.I * self.U_h * sp.diff(self.U, self.ky)

        # Create Berry curvature
        self.B = sp.diff(self.Ax, self.ky) - sp.diff(self.Ay, self.kx)

    def evaluate_energy(
        self,
        kx: np.ndarray,
        ky: np.ndarray,
        **fkwargs
    ) -> List:
        '''
        Evaluate energy bands at all give k-points
        '''
        self.e_eval = []

        for e_jit in self.e_jit:
            self.e_eval.append(e_jit(kx=kx, ky=ky, **fkwargs))
        return self.e_eval

    def evaluate_ederivative(
        self,
        kx: np.ndarray,
        ky: np.ndarray,
        **fkwargs
    ) -> List:
        '''
        
        '''
        self.ederiv_eval = []
        # Evaluate all kpoints without BZ
        for ederiv_jit in self.ederiv_jit:
            self.ederiv_eval.append(ederiv_jit(kx=kx, ky=ky, **fkwargs))
        return self.ederiv_eval

    def evaluate_dipole(self, kx, ky, **fkwargs):
        """
        Transforms the symbolic expression for the
        berry connection/dipole moment matrix to an expression
        that is numerically evaluated.

        Parameters
        ----------
        kx, ky : np.ndarray
            array of all point combinations
        fkwargs :
            keyword arguments passed to the symbolic expression
        """
        # Evaluate all kpoints without BZ
        self.Ax_eval = evaluate_njit_matrix(self.Axfjit, kx, ky, **fkwargs)
        self.Ay_eval = evaluate_njit_matrix(self.Ayfjit, kx, ky, **fkwargs)
        return self.Ax_eval, self.Ay_eval

    def evaluate_curvature(self, kx, ky, **fkwargs):
        # Evaluate all kpoints without BZ

        self.B_eval = evaluate_njit_matrix(self.Bfjit, kx, ky, **fkwargs)

        return self.B_eval