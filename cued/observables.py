import numpy as np
from cued.utility.njit import conditional_njit, evaluate_njit_matrix

########################################
# Observables for the 2-band code
########################################


########################################
# Observables working with density matrices that contain NO time data; only path
########################################
def make_polarization_path(path, P, sys):
    """
    Function that calculates the polarization for the current path

    Parameters:
    -----------
    dipole : Symbolic Dipole
    pathlen : int
        Length of one path
    n_time_steps : int
        Number of time steps
    E_dir : np.ndarray [type_real_np]
        Direction of the electric field
    A_field : np.ndarray [type_real_np]
        Vector potential integrated from electric field
    gauge : string
        System gauge 'length' or 'velocity'

    Returns:
    --------
    P_E_dir : np.ndarray [type_real_np]
        Polarization in E-field direction
    P_ortho : np.ndarray [type_real_np]
        Polarization orthogonal to E-field direction
    """
    di_01xf = sys.Axfjit[0][1]
    di_01yf = sys.Ayfjit[0][1]

    E_dir = P.E_dir
    E_ort = P.E_ort

    kx_in_path_before_shift = path[:, 0]
    ky_in_path_before_shift = path[:, 1]
    pathlen = kx_in_path_before_shift.size

    type_complex_np = P.type_complex_np
    gauge = P.gauge
    @conditional_njit(type_complex_np)
    def polarization_path(solution, E_field, A_field):

        # Dipole container
        rho_cv = solution[:, 1, 0]

        d_01x = np.empty(pathlen, dtype=type_complex_np)
        d_01y = np.empty(pathlen, dtype=type_complex_np)

        d_E_dir = np.empty(pathlen, dtype=type_complex_np)
        d_ortho = np.empty(pathlen, dtype=type_complex_np)

        if gauge == 'length':
            kx_shift = 0
            ky_shift = 0
        if gauge == 'velocity':
            kx_shift = A_field*E_dir[0]
            ky_shift = A_field*E_dir[1]

        kx_in_path = kx_in_path_before_shift + kx_shift
        ky_in_path = ky_in_path_before_shift + ky_shift

        d_01x[:] = di_01xf(kx=kx_in_path, ky=ky_in_path)
        d_01y[:] = di_01yf(kx=kx_in_path, ky=ky_in_path)

        d_E_dir[:] = d_01x * E_dir[0] + d_01y * E_dir[1]
        d_ortho[:] = d_01x * E_ort[0] + d_01y * E_ort[1]

        P_E_dir = 2*np.real(np.sum(d_E_dir * rho_cv))
        P_ortho = 2*np.real(np.sum(d_ortho * rho_cv))

        return P_E_dir, P_ortho

    return polarization_path


def make_current_path(path, P, sys):
    '''
    Calculates the intraband current as: J(t) = sum_k sum_n [j_n(k)f_n(k,t)]
    where j_n(k) != (d/dk) E_n(k)

    Parameters
    ----------
    sys : TwoBandSystem
        Hamiltonian and related functions
    pathlen : int
        Length of one path
    n_time_steps : int
        Number of time steps
    E_dir : np.ndarray [type_real_np]
        Direction of the electric field
    A_field : np.ndarray [type_real_np]
        Vector potential integrated from electric field
    gauge : string
        System gauge 'length' or 'velocity'

    Returns
    -------
    J_E_dir : np.ndarray [type_real_np]
        intraband current j_intra in E-field direction
    J_ortho : np.ndarray [type_real_np]
        intraband current j_intra orthogonal to E-field direction
    '''


    edxjit_v = sys.ederivfjit[0]
    edyjit_v = sys.ederivfjit[1]
    edxjit_c = sys.ederivfjit[2]
    edyjit_c = sys.ederivfjit[3]

    if P.save_anom:
        Bcurv_00 = sys.Bfjit[0][0]
        Bcurv_11 = sys.Bfjit[1][1]

    E_dir = P.E_dir
    E_ort = P.E_ort
    kx_in_path_before_shift = path[:, 0]
    ky_in_path_before_shift = path[:, 1]
    pathlen = kx_in_path_before_shift.size

    type_real_np = P.type_real_np
    type_complex_np = P.type_complex_np
    gauge = P.gauge
    save_anom = P.save_anom
    @conditional_njit(type_complex_np)
    def current_path(solution, E_field, A_field):

        rho_vv = solution[:, 0, 0]
        rho_cc = solution[:, 1, 1]

        # E derivative container
        edx_v = np.empty(pathlen, dtype=type_real_np)
        edy_v = np.empty(pathlen, dtype=type_real_np)
        edx_c = np.empty(pathlen, dtype=type_real_np)
        edy_c = np.empty(pathlen, dtype=type_real_np)

        e_deriv_E_dir_v = np.empty(pathlen, dtype=type_real_np)
        e_deriv_ortho_v = np.empty(pathlen, dtype=type_real_np)
        e_deriv_E_dir_c = np.empty(pathlen, dtype=type_real_np)
        e_deriv_ortho_c = np.empty(pathlen, dtype=type_real_np)

        if gauge == 'length':
            kx_shift = 0
            ky_shift = 0
            rho_vv_subs = 0
        if gauge == 'velocity':
            kx_shift = A_field*E_dir[0]
            ky_shift = A_field*E_dir[1]
            rho_vv_subs = 1

        kx_in_path = kx_in_path_before_shift + kx_shift
        ky_in_path = ky_in_path_before_shift + ky_shift

        edx_v[:] = edxjit_v(kx=kx_in_path, ky=ky_in_path)
        edy_v[:] = edyjit_v(kx=kx_in_path, ky=ky_in_path)
        edx_c[:] = edxjit_c(kx=kx_in_path, ky=ky_in_path)
        edy_c[:] = edyjit_c(kx=kx_in_path, ky=ky_in_path)


        e_deriv_E_dir_v[:] = edx_v * E_dir[0] + edy_v * E_dir[1]
        e_deriv_ortho_v[:] = edx_v * E_ort[0] + edy_v * E_ort[1]
        e_deriv_E_dir_c[:] = edx_c * E_dir[0] + edy_c * E_dir[1]
        e_deriv_ortho_c[:] = edx_c * E_ort[0] + edy_c * E_ort[1]

        J_E_dir = - np.sum(e_deriv_E_dir_v * (rho_vv.real - rho_vv_subs)) \
                  - np.sum(e_deriv_E_dir_c * rho_cc.real)
        J_ortho = - np.sum(e_deriv_ortho_v * (rho_vv.real - rho_vv_subs)) \
                  - np.sum(e_deriv_ortho_c * rho_cc.real)

        if save_anom:

            J_anom_ortho = np.zeros(2, dtype=type_real_np)

            Bcurv_v = np.empty(pathlen, dtype=type_complex_np)
            Bcurv_c = np.empty(pathlen, dtype=type_complex_np)

            Bcurv_v = Bcurv_00(kx=kx_in_path, ky=ky_in_path)
            Bcurv_c = Bcurv_11(kx=kx_in_path, ky=ky_in_path)

            J_anom_ortho[0] = -E_field * np.sum(Bcurv_v.real * rho_vv.real)
            J_anom_ortho[1] += -E_field * np.sum(Bcurv_c.real * rho_cc.real)
        else:
            J_anom_ortho = 0

        return J_E_dir, J_ortho, J_anom_ortho

    return current_path


def make_current_exact_path_velocity(path, P, sys):
    """
    Construct a function that calculates the current for the system solution per path
    Works for velocity gauge.

    Parameters
    ----------
    sys : TwoBandSystem
        Hamiltonian and related functions
    path : np.ndarray [type_real_np]
        kx and ky components of path
    E_dir : np.ndarray [type_real_np]
        Direction of the electric field
    do_semicl : bool
        if semiclassical calculation should be done
    curvature : SymbolicCurvature
        Curvature is only needed for semiclassical calculation

    Returns
    -------
    current_kernel : function
        Calculates per timestep current of a path
    """
    E_dir = P.E_dir

    hderivx = sys.hderivfjit[0]
    hdx_00 = hderivx[0][0]
    hdx_01 = hderivx[0][1]
    hdx_10 = hderivx[1][0]
    hdx_11 = hderivx[1][1]

    hderivy = sys.hderivfjit[1]
    hdy_00 = hderivy[0][0]
    hdy_01 = hderivy[0][1]
    hdy_10 = hderivy[1][0]
    hdy_11 = hderivy[1][1]

    Ujit = sys.Ujit
    U_00 = Ujit[0][0]
    U_01 = Ujit[0][1]
    U_10 = Ujit[1][0]
    U_11 = Ujit[1][1]

    Ujit_h = sys.Ujit_h
    U_h_00 = Ujit_h[0][0]
    U_h_01 = Ujit_h[0][1]
    U_h_10 = Ujit_h[1][0]
    U_h_11 = Ujit_h[1][1]

    if P.dm_dynamics_method == 'semiclassics':
        Bcurv_00 = sys.Bfjit[0][0]
        Bcurv_11 = sys.Bfjit[1][1]

    E_ort = np.array([E_dir[1], -E_dir[0]])

    kx_in_path_before_shift = path[:, 0]
    ky_in_path_before_shift = path[:, 1]
    pathlen = kx_in_path_before_shift.size

    type_complex_np = P.type_complex_np
    symmetric_insulator = P.symmetric_insulator
    dm_dynamics_method = P.dm_dynamics_method
    @conditional_njit(type_complex_np)
    def current_exact_path_velocity(solution, E_field, A_field):
        '''
        Calculates current from the system density matrix

        Parameters:
        -----------
        solution : np.ndarray [type_complex_np]
            Per timestep solution, idx 0 is k; idx 1 is fv, pvc, pcv, fc
        E_field : type_real_np
            Per timestep E_field
        A_field : type_real_np
            In the velocity gauge this determines the k-shift

        Returns:
        --------
        I_E_dir : type_real_np
            Parallel to electric field component of current
        I_ortho : type_real_np
            Orthogonal to electric field component of current
        '''
        solution = solution.reshape(pathlen, 4)

        # H derivative container
        h_deriv_x = np.empty((pathlen, 2, 2), dtype=type_complex_np)
        h_deriv_y = np.empty((pathlen, 2, 2), dtype=type_complex_np)
        h_deriv_E_dir = np.empty((pathlen, 2, 2), dtype=type_complex_np)
        h_deriv_ortho = np.empty((pathlen, 2, 2), dtype=type_complex_np)

        # Wave function container
        U = np.empty((pathlen, 2, 2), dtype=type_complex_np)
        U_h = np.empty((pathlen, 2, 2), dtype=type_complex_np)

        # Berry curvature container
        if dm_dynamics_method == 'semiclassics':
            Bcurv = np.empty((pathlen, 2), dtype=type_complex_np)

        kx_in_path = kx_in_path_before_shift + A_field*E_dir[0]
        ky_in_path = ky_in_path_before_shift + A_field*E_dir[1]

        h_deriv_x[:, 0, 0] = hdx_00(kx=kx_in_path, ky=ky_in_path)
        h_deriv_x[:, 0, 1] = hdx_01(kx=kx_in_path, ky=ky_in_path)
        h_deriv_x[:, 1, 0] = hdx_10(kx=kx_in_path, ky=ky_in_path)
        h_deriv_x[:, 1, 1] = hdx_11(kx=kx_in_path, ky=ky_in_path)

        h_deriv_y[:, 0, 0] = hdy_00(kx=kx_in_path, ky=ky_in_path)
        h_deriv_y[:, 0, 1] = hdy_01(kx=kx_in_path, ky=ky_in_path)
        h_deriv_y[:, 1, 0] = hdy_10(kx=kx_in_path, ky=ky_in_path)
        h_deriv_y[:, 1, 1] = hdy_11(kx=kx_in_path, ky=ky_in_path)

        h_deriv_E_dir[:, :, :] = h_deriv_x*E_dir[0] + h_deriv_y*E_dir[1]
        h_deriv_ortho[:, :, :] = h_deriv_x*E_ort[0] + h_deriv_y*E_ort[1]

        U[:, 0, 0] = U_00(kx=kx_in_path, ky=ky_in_path)
        U[:, 0, 1] = U_01(kx=kx_in_path, ky=ky_in_path)
        U[:, 1, 0] = U_10(kx=kx_in_path, ky=ky_in_path)
        U[:, 1, 1] = U_11(kx=kx_in_path, ky=ky_in_path)

        U_h[:, 0, 0] = U_h_00(kx=kx_in_path, ky=ky_in_path)
        U_h[:, 0, 1] = U_h_01(kx=kx_in_path, ky=ky_in_path)
        U_h[:, 1, 0] = U_h_10(kx=kx_in_path, ky=ky_in_path)
        U_h[:, 1, 1] = U_h_11(kx=kx_in_path, ky=ky_in_path)

        if dm_dynamics_method == 'semiclassics':
            Bcurv[:, 0] = Bcurv_00(kx=kx_in_path, ky=ky_in_path)
            Bcurv[:, 1] = Bcurv_11(kx=kx_in_path, ky=ky_in_path)

        I_E_dir = 0
        I_ortho = 0

        rho_vv = solution[:, 0]
        # rho_vc = solution[:, 1]
        rho_cv = solution[:, 2]
        rho_cc = solution[:, 3]

        if symmetric_insulator:
            rho_vv = -rho_cc + 1

        for i_k in range(pathlen):

            dH_U_E_dir = h_deriv_E_dir[i_k] @ U[i_k]
            U_h_H_U_E_dir = U_h[i_k] @ dH_U_E_dir

            dH_U_ortho = h_deriv_ortho[i_k] @ U[i_k]
            U_h_H_U_ortho = U_h[i_k] @ dH_U_ortho

            I_E_dir += - U_h_H_U_E_dir[0, 0].real * (rho_vv[i_k].real - 1)
            I_E_dir += - U_h_H_U_E_dir[1, 1].real * rho_cc[i_k].real
            I_E_dir += - 2*np.real(U_h_H_U_E_dir[0, 1] * rho_cv[i_k])

            I_ortho += - U_h_H_U_ortho[0, 0].real * (rho_vv[i_k].real - 1)
            I_ortho += - U_h_H_U_ortho[1, 1].real * rho_cc[i_k].real
            I_ortho += - 2*np.real(U_h_H_U_ortho[0, 1] * rho_cv[i_k])

            if dm_dynamics_method == 'semiclassics':
                # '-' because there is q^2 compared to q only at the SBE current
                I_ortho += -E_field * Bcurv[i_k, 0].real * rho_vv[i_k].real
                I_ortho += -E_field * Bcurv[i_k, 1].real * rho_cc[i_k].real

        return I_E_dir, I_ortho

    return current_exact_path_velocity


def make_current_exact_path_length(path, P, sys):
    """
    Construct a function that calculates the current for the system solution per path.
    Works for length gauge.

    Parameters
    ----------
    sys : TwoBandSystem
        Hamiltonian and related functions
    path : np.ndarray [type_real_np]
        kx and ky components of path
    E_dir : np.ndarray [type_real_np]
        Direction of the electric field
    do_semicl : bool
        if semiclassical calculation should be done
    curvature : SymbolicCurvature
        Curvature is only needed for semiclassical calculation

    Returns:
    --------
    current_kernel : function
        Calculates per timestep current of a path
    """
    E_dir = P.E_dir
    E_ort = P.E_ort

    kx_in_path = path[:, 0]
    ky_in_path = path[:, 1]
    pathlen = kx_in_path.size

    if P.dm_dynamics_method == 'semiclassics':
        # Berry curvature container
        Bcurv = np.empty((pathlen, 2), dtype=P.type_complex_np)

    h_deriv_x = evaluate_njit_matrix(sys.hderivfjit[0], kx=kx_in_path, ky=ky_in_path,
                                     dtype=P.type_complex_np)
    h_deriv_y = evaluate_njit_matrix(sys.hderivfjit[1], kx=kx_in_path, ky=ky_in_path,
                                     dtype=P.type_complex_np)

    h_deriv_E_dir= h_deriv_x*E_dir[0] + h_deriv_y*E_dir[1]
    h_deriv_ortho = h_deriv_x*E_ort[0] + h_deriv_y*E_ort[1]

    U = evaluate_njit_matrix(sys.Ujit, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)
    U_h = evaluate_njit_matrix(sys.Ujit_h, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)

    if P.dm_dynamics_method == 'semiclassics':
        Bcurv[:, 0] = sys.Bfjit[0][0](kx=kx_in_path, ky=ky_in_path)
        Bcurv[:, 1] = sys.Bfjit[1][1](kx=kx_in_path, ky=ky_in_path)

    symmetric_insulator = P.symmetric_insulator
    dm_dynamics_method = P.dm_dynamics_method
    @conditional_njit(P.type_complex_np)
    def current_exact_path_length(solution, E_field, A_field):
        '''
        Parameters:
        -----------
        solution : np.ndarray [type_complex_np]
            Per timestep solution, idx 0 is k; idx 1 is fv, pvc, pcv, fc
        E_field : type_real_np
            Per timestep E_field
        _A_field : dummy
            In the length gauge this is just a dummy variable
        j_k_E_dir_path


        Returns:
        --------
        I_E_dir : type_real_np
            Parallel to electric field component of current
        I_ortho : type_real_np
            Orthogonal to electric field component of current
        '''
        solution = solution.reshape(pathlen, 4)
        I_E_dir = 0
        I_ortho = 0

        rho_vv = solution[:, 0]
        rho_vc = solution[:, 1]
        rho_cv = solution[:, 2]
        rho_cc = solution[:, 3]

        if symmetric_insulator:
            rho_vv = -rho_cc

        for i_k in range(pathlen):
            U_h_H_U_E_dir = U_h[i_k] @ (h_deriv_E_dir[i_k] @ U[i_k])
            U_h_H_U_ortho = U_h[i_k] @ (h_deriv_ortho[i_k] @ U[i_k])

            I_E_dir -= U_h_H_U_E_dir[0, 0].real * rho_vv[i_k].real \
                     + U_h_H_U_E_dir[1, 1].real * rho_cc[i_k].real \
                     + 2*np.real(U_h_H_U_E_dir[0, 1] * rho_cv[i_k])

            I_ortho -= U_h_H_U_ortho[0, 0].real * rho_vv[i_k].real \
                     + U_h_H_U_ortho[1, 1].real * rho_cc[i_k].real \
                     + 2*np.real(U_h_H_U_ortho[0, 1] * rho_cv[i_k])

            if dm_dynamics_method == 'semiclassics':
                # '-' because there is q^2 compared to q only at the SBE current
                I_ortho -= E_field * Bcurv[i_k, 0].real * rho_vv[i_k].real \
                         + E_field * Bcurv[i_k, 1].real * rho_vc[i_k].real

        return I_E_dir, I_ortho

    if P.save_full:
        @conditional_njit(P.type_complex_np)
        def current_exact_path_length(solution, E_field, A_field, j_k_E_dir_path, j_k_ortho_path):
            '''
            Parameters:
            -----------
            solution : np.ndarray [type_complex_np]
                    Per timestep solution, idx 0 is k; idx 1 is fv, pvc, pcv, fc
            E_field : type_real_np
                    Per timestep E_field
            _A_field : dummy
                    In the length gauge this is just a dummy variable

            Returns:
            --------
            I_E_dir : type_real_np
                    Parallel to electric field component of current
            I_ortho : type_real_np
                    Orthogonal to electric field component of current
            '''
            solution = solution.reshape(pathlen, 4)
            I_E_dir = 0
            I_ortho = 0

            rho_vv = solution[:, 0]
            rho_vc = solution[:, 1]
            rho_cv = solution[:, 2]
            rho_cc = solution[:, 3]

            if symmetric_insulator:
                rho_vv = -rho_cc

            for i_k in range(pathlen):

                U_h_H_U_E_dir = U_h[i_k] @ (h_deriv_E_dir[i_k] @ U[i_k])
                U_h_H_U_ortho = U_h[i_k] @ (h_deriv_ortho[i_k] @ U[i_k])

                j_k_E_dir_path[i_k] = - U_h_H_U_E_dir[0, 0].real * rho_vv[i_k].real \
                                      - U_h_H_U_E_dir[1, 1].real * rho_cc[i_k].real \
                                      - 2*np.real(U_h_H_U_E_dir[0, 1] * rho_cv[i_k])

                j_k_ortho_path[i_k] = - U_h_H_U_ortho[0, 0].real * rho_vv[i_k].real \
                                      - U_h_H_U_ortho[1, 1].real * rho_cc[i_k].real \
                                      - 2*np.real(U_h_H_U_ortho[0, 1] * rho_cv[i_k])

                if dm_dynamics_method == 'semiclassics':
                    # '-' because there is q^2 compared to q only at the SBE current
                    j_k_ortho_path[i_k] = - E_field * Bcurv[i_k, 0].real * rho_vv[i_k].real \
                                          - E_field * Bcurv[i_k, 1].real * rho_vc[i_k].real

            return np.sum(j_k_E_dir_path), np.sum(j_k_ortho_path)

    return current_exact_path_length

########################################
# Observables from given bandstructures
########################################
def make_current_exact_bandstructure_velocity(path, P, sys):

    Nk1 = P.Nk1
    n = P.bands
    E_dir = P.E_dir
    E_ort = np.array([E_dir[1], -E_dir[0]])

    #wire matrix elements
    mel00x = sys.melxjit[0][0]
    mel01x = sys.melxjit[0][1]
    mel10x = sys.melxjit[1][0]
    mel11x = sys.melxjit[1][1]

    mel00y = sys.melyjit[0][0]
    mel01y = sys.melyjit[0][1]
    mel10y = sys.melyjit[1][0]
    mel11y = sys.melyjit[1][1]

    kx_in_path_before_shift = path[:, 0]
    ky_in_path_before_shift = path[:, 1]
    pathlen = kx_in_path_before_shift.size

    type_complex_np = P.type_complex_np

    # mel_x = evaluate_njit_matrix(sys.melxjit, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)
    # mel_y = evaluate_njit_matrix(sys.melyjit, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)

    # mel_in_path = P.E_dir[0]*mel_x + P.E_dir[1]*mel_y
    # mel_ortho = P.E_ort[0]*mel_x + P.E_ort[1]*mel_y

    @conditional_njit(P.type_complex_np)
    def current_exact_path(solution, E_field=0, A_field=0):

        kx_in_path = kx_in_path_before_shift + A_field*E_dir[0]
        ky_in_path = ky_in_path_before_shift + A_field*E_dir[1]

        mel_x = np.empty((pathlen, 2, 2), dtype=type_complex_np)
        mel_y = np.empty((pathlen, 2, 2), dtype=type_complex_np)

        mel_x[:, 0, 0] = mel00x(kx=kx_in_path, ky=ky_in_path)
        mel_x[:, 0, 1] = mel01x(kx=kx_in_path, ky=ky_in_path)
        mel_x[:, 1, 0] = mel10x(kx=kx_in_path, ky=ky_in_path)
        mel_x[:, 1, 1] = mel11x(kx=kx_in_path, ky=ky_in_path)

        mel_y[:, 0, 0] = mel00y(kx=kx_in_path, ky=ky_in_path)
        mel_y[:, 0, 1] = mel01y(kx=kx_in_path, ky=ky_in_path)
        mel_y[:, 1, 0] = mel10y(kx=kx_in_path, ky=ky_in_path)
        mel_y[:, 1, 1] = mel11y(kx=kx_in_path, ky=ky_in_path)

        mel_in_path = E_dir[0]*mel_x + E_dir[1]*mel_y
        mel_ortho = E_ort[0]*mel_x + E_ort[1]*mel_y

        J_exact_E_dir = 0
        J_exact_ortho = 0

        rho_vv = solution[:, 0, 0]
        # rho_vc = solution[:, 1]
        rho_cv = solution[:, 1, 0]
        rho_cc = solution[:, 1, 1]

        for i_k in range(pathlen):
            J_exact_E_dir += - mel_in_path[i_k, 0, 0].real * (rho_vv[i_k].real - 1)
            J_exact_E_dir += - mel_in_path[i_k, 1, 1].real * rho_cc[i_k].real
            J_exact_E_dir += - 2*np.real( mel_in_path[i_k, 0, 1] * rho_cv[i_k] )

            J_exact_ortho += - mel_ortho[i_k, 0, 0].real * (rho_vv[i_k].real - 1)
            J_exact_ortho += - mel_ortho[i_k, 1, 1].real * rho_cc[i_k].real
            J_exact_ortho += - 2*np.real( mel_ortho[i_k, 0, 1] * rho_cv[i_k] )
        return J_exact_E_dir, J_exact_ortho
    return current_exact_path

def make_current_exact_bandstructure(path, P, sys):

    Nk1 = P.Nk1
    n = P.bands

    kx_in_path = path[:, 0]
    ky_in_path = path[:, 1]

    mel_x = evaluate_njit_matrix(sys.melxjit, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)
    mel_y = evaluate_njit_matrix(sys.melyjit, kx=kx_in_path, ky=ky_in_path, dtype=P.type_complex_np)

    mel_in_path = P.E_dir[0]*mel_x + P.E_dir[1]*mel_y
    mel_ortho = P.E_ort[0]*mel_x + P.E_ort[1]*mel_y 

    @conditional_njit(P.type_complex_np)
    def current_exact_path(solution, _E_field=0, _A_field=0):


        J_exact_E_dir = 0
        J_exact_ortho = 0

        for i_k in range(Nk1):
            for i in range(n):
                J_exact_E_dir -=  mel_in_path[i_k, i, i].real * solution[i_k, i, i].real
                J_exact_ortho -=  mel_ortho[i_k, i, i].real * solution[i_k, i, i].real
                for j in range(n):
                    if i != j:
                        J_exact_E_dir -= np.real(mel_in_path[i_k, i, j] * solution[i_k, j, i])
                        J_exact_ortho -= np.real(mel_ortho[i_k, i, j] * solution[i_k, j, i])

        return J_exact_E_dir, J_exact_ortho

    return current_exact_path

def make_intraband_current_bandstructure_velocity(path, P, sys):
    """
        Function that calculates the intraband current from eq. (76 and 77) with or without the
            anomalous contribution via the Berry curvature
    """

    Nk1 = P.Nk1
    n = P.bands
    save_anom = P.save_anom
    E_dir = P.E_dir
    E_ort = np.array([E_dir[1], -E_dir[0]])
    #wire ederivative

    ederiv0x = sys.dkxejit[0]
    ederiv1x = sys.dkxejit[1]
    ederiv0y = sys.dkyejit[0]
    ederiv1y = sys.dkyejit[1]

    kx_in_path_before_shift = path[:, 0]
    ky_in_path_before_shift = path[:, 1]
    pathlen = kx_in_path_before_shift.size

    type_complex_np = P.type_complex_np


    @conditional_njit(P.type_complex_np)
    def current_intra_path(solution, E_field, A_field):

        kx_in_path = kx_in_path_before_shift + A_field*E_dir[0]
        ky_in_path = ky_in_path_before_shift + A_field*E_dir[1]

        ederivx = np.empty((pathlen, 2), dtype=type_complex_np)
        ederivy = np.empty((pathlen, 2), dtype=type_complex_np)

        ederivx[:, 0] = ederiv0x(kx=kx_in_path, ky=ky_in_path)
        ederivx[:, 1] = ederiv1x(kx=kx_in_path, ky=ky_in_path)
        ederivy[:, 0] = ederiv0y(kx=kx_in_path, ky=ky_in_path)
        ederivy[:, 1] = ederiv1y(kx=kx_in_path, ky=ky_in_path)

        ederiv_in_path = E_dir[0]*ederivx + E_dir[1]*ederivy
        ederiv_ortho = E_ort[0]*ederivx + E_ort[1]*ederivy

        J_intra_E_dir = 0
        J_intra_ortho = 0
        J_anom_ortho = 0

        for k in range(Nk1):
            for i in range(n):
                if i == 0:
                    J_intra_E_dir -= ederiv_in_path[k, i] * (solution[k, i, i].real - 1)
                    J_intra_ortho -= ederiv_ortho[k, i] * (solution[k, i, i].real - 1)
                else:
                    J_intra_E_dir -= ederiv_in_path[k, i] * solution[k, i, i].real
                    J_intra_ortho -= ederiv_ortho[k, i] * solution[k, i, i].real

                if save_anom:
                    J_anom_ortho += 0
                    print('J_anom not implemented')

        return J_intra_E_dir, J_intra_ortho, J_anom_ortho
    return current_intra_path


def make_intraband_current_bandstructure(path, P, sys):
    """
        Function that calculates the intraband current from eq. (76 and 77) with or without the
            anomalous contribution via the Berry curvature
    """

    Nk1 = P.Nk1
    n = P.bands
    save_anom = P.save_anom

    kx_in_path = path[:, 0]
    ky_in_path = path[:, 1]

    ederivx = np.zeros([P.Nk1, P.bands], dtype=P.type_real_np)
    ederivy = np.zeros([P.Nk1, P.bands], dtype=P.type_real_np)

    for i in range(P.bands):
        ederivx[:, i] = sys.dkxejit[i](kx=kx_in_path, ky=ky_in_path)
        ederivy[:, i] = sys.dkyejit[i](kx=kx_in_path, ky=ky_in_path)

    ederiv_in_path = P.E_dir[0]*ederivx + P.E_dir[1]*ederivy
    ederiv_ortho = P.E_ort[0]*ederivx + P.E_ort[1]*ederivy

    @conditional_njit(P.type_complex_np)
    def current_intra_path(solution, E_field, A_field):

        J_intra_E_dir = 0
        J_intra_ortho = 0
        J_anom_ortho = 0

        for k in range(Nk1):
            for i in range(n):
                J_intra_E_dir -= ederiv_in_path[k, i] * solution[k, i, i].real
                J_intra_ortho -= ederiv_ortho[k, i] * solution[k, i, i].real

                if save_anom:
                    J_anom_ortho += 0
                    print('J_anom not implemented')

        return J_intra_E_dir, J_intra_ortho, J_anom_ortho
    return current_intra_path

def make_polarization_inter_bandstructure_velocity(path, P, sys):
    """
        Function that calculates the interband polarization from eq. (74)
    """
    dipole_in_path = sys.dipole_in_path
    dipole_ortho = sys.dipole_ortho

    Nk1 = P.Nk1
    n = P.bands
    save_anom = P.save_anom
    E_dir = P.E_dir
    E_ort = np.array([E_dir[1], -E_dir[0]])
    #wire dipoles

    dipole00x = sys.dipole_xfjit[0][0]
    dipole01x = sys.dipole_xfjit[0][1]
    dipole10x = sys.dipole_xfjit[1][0]
    dipole11x = sys.dipole_xfjit[1][1]
    dipole00y = sys.dipole_yfjit[0][0]
    dipole01y = sys.dipole_yfjit[0][1]
    dipole10y = sys.dipole_yfjit[1][0]
    dipole11y = sys.dipole_yfjit[1][1]      

    kx_in_path_before_shift = path[:, 0]
    ky_in_path_before_shift = path[:, 1]
    pathlen = kx_in_path_before_shift.size

    type_complex_np = P.type_complex_np


    @conditional_njit(type_complex_np)
    def polarization_inter_path(solution, E_field, A_field):

        kx_in_path = kx_in_path_before_shift + A_field*E_dir[0]
        ky_in_path = ky_in_path_before_shift + A_field*E_dir[1]

        dipole_x = np.empty((pathlen, 2, 2), dtype=type_complex_np)
        dipole_y = np.empty((pathlen, 2, 2), dtype=type_complex_np)

        dipole_x[:, 0, 0] = dipole00x(kx=kx_in_path, ky=ky_in_path)
        dipole_x[:, 0, 1] = dipole01x(kx=kx_in_path, ky=ky_in_path)
        dipole_x[:, 1, 0] = dipole10x(kx=kx_in_path, ky=ky_in_path)
        dipole_x[:, 1, 1] = dipole11x(kx=kx_in_path, ky=ky_in_path)

        dipole_y[:, 0, 0] = dipole00y(kx=kx_in_path, ky=ky_in_path)
        dipole_y[:, 0, 1] = dipole01y(kx=kx_in_path, ky=ky_in_path)
        dipole_y[:, 1, 0] = dipole10y(kx=kx_in_path, ky=ky_in_path)
        dipole_y[:, 1, 1] = dipole11y(kx=kx_in_path, ky=ky_in_path)

        dipole_in_path = E_dir[0]*dipole_x + E_dir[1]*dipole_y
        dipole_ortho = E_ort[0]*dipole_x + E_ort[1]*dipole_y

        P_inter_E_dir = 0
        P_inter_ortho = 0

        for k in range(Nk1):
            for i in range(n):
                for j in range(n):
                    if i > j:
                        P_inter_E_dir += 2*np.real(dipole_in_path[k, i, j]*solution[k, j, i])
                        P_inter_ortho += 2*np.real(dipole_ortho[k, i, j]*solution[k, j, i])
        return P_inter_E_dir, P_inter_ortho
    return polarization_inter_path


def make_polarization_inter_bandstructure(P, sys):
    """
        Function that calculates the interband polarization from eq. (74)
    """
    dipole_in_path = sys.dipole_in_path
    dipole_ortho = sys.dipole_ortho
    n = P.bands
    Nk1 = P.Nk1
    type_complex_np = P.type_complex_np

    @conditional_njit(type_complex_np)
    def polarization_inter_path(solution, E_field, A_field):

        P_inter_E_dir = 0
        P_inter_ortho = 0

        for k in range(Nk1):
            for i in range(n):
                for j in range(n):
                    if i > j:
                        P_inter_E_dir += 2*np.real(dipole_in_path[k, i, j]*solution[k, j, i])
                        P_inter_ortho += 2*np.real(dipole_ortho[k, i, j]*solution[k, j, i])
        return P_inter_E_dir, P_inter_ortho
    return polarization_inter_path
