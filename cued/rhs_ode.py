import numpy as np

from scipy.integrate import ode
from typing import Callable, Tuple, Union

from cued.utility.njit import conditional_njit

class seriesSolver():
    """"
    This class imitates methods of a scipy.integrate.ode solver.
    It only implements the methods needed for the solver loop in CUED.
    To be specific: This is a series expansion
    """
    def __init__(self, dt):
        self.dt = dt

class rk4Solver():
    """
    This class imitates methods of a scipy.integrate.ode solver.
    It only implements the methods needed for the solver loop in CUED.
    To be specific: This is a Runge-Kutta 4 solver.
    """
    def __init__(self, rhs_ode, dt):
        self.rhs_ode = rhs_ode
        self.dt = dt

    def set_initial_value(self, y0, t0):
        # Initial
        self.y0 = y0
        # Updated
        self.y = y0
        # Initial
        self.t0 = t0
        # Updated
        self.t = t0

        return self

    def set_f_params(self, kpath, dipole_in_path, e_in_path, y0, dk):
        self.kpath = kpath
        self.dipole_in_path = dipole_in_path
        self.e_in_path = e_in_path
        self.y0 = y0
        self.dk = dk

    def integrate(self, t):
        self.t = t
        k1 = self.rhs_ode(
                self.t - self.dt, self.y, self.kpath,
                self.dipole_in_path, self.e_in_path, self.y0, self.dk
             )
        k2 = self.rhs_ode(
                self.t - 0.5*self.dt, self.y + 0.5*k1, self.kpath,
                self.dipole_in_path, self.e_in_path, self.y0, self.dk
             )
        k3 = self.rhs_ode(
                self.t - 0.5*self.dt, self.y + 0.5*k2, self.kpath,
                self.dipole_in_path, self.e_in_path, self.y0, self.dk
             )
        k4 = self.rhs_ode(
                self.t, self.y + k3, self.kpath, 
                self.dipole_in_path, self.e_in_path, self.y0, self.dk
             )

        self.y = self.y + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    def successful(self):
        return True


def make_rhs_ode_2_band(
    sys,
    electric_field,
    P
) -> Callable:
    """
        Initialization of the solver for the sbe ( eq. (39/47/80) in https://arxiv.org/abs/2008.03177)

        Author:
        Additional Contact: Jan Wilhelm (jan.wilhelm@ur.de)
    """
    gamma1 = P.gamma1
    gamma2 = P.gamma2
    type_complex_np = P.type_complex_np
    dk_order = P.dk_order
    dm_dynamics_method = P.dm_dynamics_method
    E_dir = P.E_dir
    gauge = P.gauge

    # Wire the energies
    evf = sys.e_jit[0]
    ecf = sys.e_jit[1]

    # Wire the dipoles
    # kx-parameter
    di_00xf = sys.Ax_jit[0][0]
    di_01xf = sys.Ax_jit[0][1]
    di_11xf = sys.Ax_jit[1][1]

    # ky-parameter
    di_00yf = sys.Ay_jit[0][0]
    di_01yf = sys.Ay_jit[0][1]
    di_11yf = sys.Ay_jit[1][1]

    @conditional_njit(P.type_complex_np)
    def flength(t, y, kpath, dipole_in_path, e_in_path, y0, dk):
        """
        Length gauge doesn't need recalculation of energies and dipoles.
        The length gauge is evaluated on a constant pre-defined k-grid.
        """
        # x != y(t+dt)
        x = np.empty(np.shape(y), dtype=type_complex_np)

        # Gradient term coefficient
        electric_f = electric_field(t)
        D = electric_f/dk

        # Update the solution vector
        Nk_path = kpath.shape[0]
        for k in range(Nk_path):
            i = 4*k
            right4 = 4*(k+4)
            right3 = 4*(k+3)
            right2 = 4*(k+2)
            right  = 4*(k+1)
            left   = 4*(k-1)
            left2  = 4*(k-2)
            left3  = 4*(k-3)
            left4  = 4*(k-4)
            if k == 0:
                left   = 4*(Nk_path-1)
                left2  = 4*(Nk_path-2)
                left3  = 4*(Nk_path-3)
                left4  = 4*(Nk_path-4)
            elif k == 1 and dk_order >= 4:
                left2  = 4*(Nk_path-1)
                left3  = 4*(Nk_path-2)
                left4  = 4*(Nk_path-3)
            elif k == 2 and dk_order >= 6:
                left3  = 4*(Nk_path-1)
                left4  = 4*(Nk_path-2)
            elif k == 3 and dk_order >= 8:
                left4  = 4*(Nk_path-1)
            elif k == Nk_path-1:
                right4 = 4*3
                right3 = 4*2
                right2 = 4*1
                right  = 4*0
            elif k == Nk_path-2 and dk_order >= 4:
                right4 = 4*2
                right3 = 4*1
                right2 = 4*0
            elif k == Nk_path-3 and dk_order >= 6:
                right4 = 4*1
                right3 = 4*0
            elif k == Nk_path-4 and dk_order >= 8:
                right4 = 4*0

            # Energy gap e_2(k) - e_1(k) >= 0 at point k
            ecv = e_in_path[k, 1] - e_in_path[k, 0]

            # Berry connection
            A_in_path = dipole_in_path[k, 0, 0] - dipole_in_path[k, 1, 1]

            # Rabi frequency: w_R = q*d_12(k)*E(t)
            # Rabi frequency conjugate: w_R_c = q*d_21(k)*E(t)
            wr = dipole_in_path[k, 0, 1]*electric_f
            wr_c = wr.conjugate()

            # Rabi frequency: w_R = q*(d_11(k) - d_22(k))*E(t)
            wr_d_diag = A_in_path*electric_f

            # Update each component of the solution vector
            # i = f_v, i+1 = p_vc, i+2 = p_cv, i+3 = f_c
            x[i]   = 2*(y[i+1]*wr_c).imag - gamma1*(y[i]-y0[i])

            x[i+1] = (1j*ecv - gamma2 + 1j*wr_d_diag)*y[i+1] - 1j*wr*(y[i]-y[i+3])

            x[i+3] = -2*(y[i+1]*wr_c).imag - gamma1*(y[i+3]-y0[i+3])

            # compute drift term via k-derivative
            if dk_order == 2:
                x[i]   += D*( y[right]/2   - y[left]/2  )
                x[i+1] += D*( y[right+1]/2 - y[left+1]/2 )
                x[i+3] += D*( y[right+3]/2 - y[left+3]/2 )
            elif dk_order == 4:
                x[i]   += D*(- y[right2]/12   + 2/3*y[right]   - 2/3*y[left]   + y[left2]/12 )
                x[i+1] += D*(- y[right2+1]/12 + 2/3*y[right+1] - 2/3*y[left+1] + y[left2+1]/12 )
                x[i+3] += D*(- y[right2+3]/12 + 2/3*y[right+3] - 2/3*y[left+3] + y[left2+3]/12 )
            elif dk_order == 6:
                x[i]   += D*(  y[right3]/60   - 3/20*y[right2]   + 3/4*y[right] \
                             - y[left3]/60    + 3/20*y[left2]    - 3/4*y[left] )
                x[i+1] += D*(  y[right3+1]/60 - 3/20*y[right2+1] + 3/4*y[right+1] \
                             - y[left3+1]/60  + 3/20*y[left2+1]  - 3/4*y[left+1] )
                x[i+3] += D*(  y[right3+3]/60 - 3/20*y[right2+3] + 3/4*y[right+3] \
                             - y[left3+3]/60  + 3/20*y[left2+3]  - 3/4*y[left+3] )
            elif dk_order == 8:
                x[i]   += D*(- y[right4]/280   + 4/105*y[right3]   - 1/5*y[right2]   + 4/5*y[right] \
                             + y[left4] /280   - 4/105*y[left3]    + 1/5*y[left2]    - 4/5*y[left] )
                x[i+1] += D*(- y[right4+1]/280 + 4/105*y[right3+1] - 1/5*y[right2+1] + 4/5*y[right+1] \
                             + y[left4+1] /280 - 4/105*y[left3+1]  + 1/5*y[left2+1]  - 4/5*y[left+1] )
                x[i+3] += D*(- y[right4+3]/280 + 4/105*y[right3+3] - 1/5*y[right2+3] + 4/5*y[right+3] \
                             + y[left4+3] /280 - 4/105*y[left3+3]  + 1/5*y[left2+3]  - 4/5*y[left+3] )

            x[i+2] = x[i+1].conjugate()

        x[-1] = -electric_f
        return x

    @conditional_njit(P.type_complex_np)
    def pre_velocity(kpath, k_shift):
        # First round k_shift is zero, consequently we just recalculate
        # the original data ecv_in_path, dipole_in_path, A_in_path
        kx = kpath[:, 0] + E_dir[0]*k_shift
        ky = kpath[:, 1] + E_dir[1]*k_shift

        ecv_in_path = ecf(kx=kx, ky=ky) - evf(kx=kx, ky=ky)

        if dm_dynamics_method == 'semiclassics':
            zero_arr = np.zeros(kx.size, dtype=type_complex_np)
            dipole_in_path = zero_arr
            A_in_path = zero_arr
        else:
            di_00x = di_00xf(kx=kx, ky=ky)
            di_01x = di_01xf(kx=kx, ky=ky)
            di_11x = di_11xf(kx=kx, ky=ky)
            di_00y = di_00yf(kx=kx, ky=ky)
            di_01y = di_01yf(kx=kx, ky=ky)
            di_11y = di_11yf(kx=kx, ky=ky)

            dipole_in_path = E_dir[0]*di_01x + E_dir[1]*di_01y
            A_in_path = E_dir[0]*di_00x + E_dir[1]*di_00y \
                - (E_dir[0]*di_11x + E_dir[1]*di_11y)

        return ecv_in_path, dipole_in_path, A_in_path

    @conditional_njit(P.type_complex_np)
    def fvelocity(t, y, kpath, _dipole_in_path, _e_in_path, y0, _dk):
        """
        Velocity gauge needs a recalculation of energies and dipoles as k
        is shifted according to the vector potential A
        """

        ecv_in_path, dipole_in_path, A_in_path = pre_velocity(kpath, y[-1].real)
        # x != y(t+dt)
        x = np.empty(np.shape(y), dtype=type_complex_np)

        electric_f = electric_field(t)

        # Update the solution vector
        Nk_path = kpath.shape[0]
        for k in range(Nk_path):
            i = 4*k
            # Energy term eband(i,k) the energy of band i at point k
            ecv = ecv_in_path[k]

            # Rabi frequency: w_R = d_12(k).E(t)
            # Rabi frequency conjugate
            wr = dipole_in_path[k]*electric_f
            wr_c = wr.conjugate()

            # Rabi frequency: w_R = (d_11(k) - d_22(k))*E(t)
            # wr_d_diag   = A_in_path[k]*D
            wr_d_diag = A_in_path[k]*electric_f

            # Update each component of the solution vector
            # i = f_v, i+1 = p_vc, i+2 = p_cv, i+3 = f_c
            x[i] = 2*(y[i+1]*wr_c).imag - gamma1*(y[i]-y0[i])

            x[i+1] = (1j*ecv - gamma2 + 1j*wr_d_diag)*y[i+1] - 1j*wr*(y[i]-y[i+3])

            x[i+2] = x[i+1].conjugate()

            x[i+3] = -2*(y[i+1]*wr_c).imag - gamma1*(y[i+3]-y0[i+3])

        x[-1] = -electric_f

        return x

    freturn = None
    if gauge == 'length':
        print("Using length gauge")
        freturn = flength
    elif gauge == 'velocity':
        print("Using velocity gauge")
        freturn = fvelocity
    else:
        raise AttributeError("You have to either assign velocity or length gauge")

    # The python solver does not directly accept jitted functions so we wrap it
    def f(t, y, kpath, dipole_in_path, e_in_path, y0, dk):
        return freturn(t, y, kpath, dipole_in_path, e_in_path, y0, dk)

    return f

def dispatch_rhs_ode_and_solver(
    P,
    T,
    sys
) -> Tuple[Union[Callable, int], Union[ode, rk4Solver, int]]:

    if P.dm_dynamics_method in ('sbe', 'semiclassics'):
        rhs_ode = make_rhs_ode_2_band(sys, T.electric_field, P)

        if P.solver_method in ('bdf', 'adams'):
            solver = ode(rhs_ode, jac=None)\
                .set_integrator('zvode', method=P.solver_method, max_step=P.dt)
        elif P.solver_method == 'rk4':
            solver = rk4Solver(rhs_ode, dt=P.dt)
        else:
            raise AttributeError("You have to either assign bdf, adams or rk4 as solver method")
    else:
        rhs_ode = 0
        solver = 0

    return rhs_ode, solver