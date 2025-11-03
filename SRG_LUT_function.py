from joblib import Parallel, delayed
import numpy as np
import os
from SRG_RCWA_functions import func_ER_srg

def rcwa_worker_joblib(idx, theta_flat, phi_flat, lam0, te, tm, n1, n2, Lx, M, ER, Lz):
    from SRG_RCWA_functions import func_rcwa_1d_isotropic
    th = theta_flat[idx]
    ph = phi_flat[idx]
    E_te_ref, E_tm_ref, E_te_trn, E_tm_trn, R, T= func_rcwa_1d_isotropic(lam0, th, ph, te, tm, n1, n2, Lx, M, ER, Lz)
    return [
        E_te_ref[M - 2][0], E_te_ref[M - 1][0], E_te_ref[M][0], E_te_ref[M + 1][0], E_te_ref[M + 2][0], 
        E_tm_ref[M - 2][0], E_tm_ref[M - 1][0], E_tm_ref[M][0], E_tm_ref[M + 1][0], E_tm_ref[M + 2][0], 
        E_te_trn[M - 2][0], E_te_trn[M - 1][0], E_te_trn[M][0], E_te_trn[M + 1][0], E_te_trn[M + 2][0], 
        E_tm_trn[M - 2][0], E_tm_trn[M - 1][0], E_tm_trn[M][0], E_tm_trn[M + 1][0], E_tm_trn[M + 2][0]]

def rcwa_worker_joblib1(idx, theta_flat, phi_flat, lam0, te, tm, n1, n2, Lx, M, ER, Lz):
    from SRG_RCWA_functions import func_rcwa_1d_isotropic
    th = theta_flat[idx]
    ph = phi_flat[idx]
    E_te_ref, E_tm_ref, E_te_trn, E_tm_trn, R, T= func_rcwa_1d_isotropic(lam0, th, ph, te, tm, n1, n2, Lx, M, ER, Lz)
    return [
        E_te_ref[M - 1][0], E_te_ref[M][0], E_te_ref[M + 1][0],
        E_tm_ref[M - 1][0], E_tm_ref[M][0], E_tm_ref[M + 1][0],
        E_te_trn[M - 1][0], E_te_trn[M][0], E_te_trn[M + 1][0],
        E_tm_trn[M - 1][0], E_tm_trn[M][0], E_tm_trn[M + 1][0]]

def LUT_srg_rcwa_parallel_joblib(
    n_ridge,
    top,
    bottom,
    front,
    back,
    d,
    Lx,
    fill_factor,
    lam0,
    theta,
    phi,
    n1,
    n2,
    te,
    tm,
    n_jobs=-1  # use all available cores by default
):
    um = 1e-6
    deg = np.pi / 180

    n_groove = 1.0
    Nx = 2 ** 10
    Nz = int(np.ceil(d / um / 0.02))
    M = 3

    ER, Lz = func_ER_srg(n_groove, n_ridge, top, bottom, front, back, d, Lx, fill_factor, Nx, Nz)

    theta_flat = theta.flatten() # flatten the angular matrix for parallel calculation
    phi_flat = phi.flatten()
    N = len(theta_flat) # number of calculation angles

    results_list = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(rcwa_worker_joblib)(
            idx, theta_flat, phi_flat, lam0, te, tm, n1, n2, Lx, M, ER, Lz
        ) for idx in range(N)
    )

    results_array = np.array(results_list)
    results_array = results_array.reshape(*theta.shape, 20)
    return results_array


def LUT_srg_rcwa_parallel_joblib1(
    n_ridge,
    top,
    bottom,
    front,
    back,
    d,
    Lx,
    fill_factor,
    lam0,
    theta,
    phi,
    n1,
    n2,
    te,
    tm,
    n_jobs=-1  # use all available cores by default
):
    um = 1e-6
    deg = np.pi / 180

    n_groove = 1.0
    Nx = 2 ** 10
    Nz = int(np.ceil(d / um / 0.02))
    M = 3

    ER, Lz = func_ER_srg(n_groove, n_ridge, top, bottom, front, back, d, Lx, fill_factor, Nx, Nz)

    theta_flat = theta.flatten() # flatten the angular matrix for parallel calculation
    phi_flat = phi.flatten()
    N = len(theta_flat) # number of calculation angles

    results_list = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(rcwa_worker_joblib1)(
            idx, theta_flat, phi_flat, lam0, te, tm, n1, n2, Lx, M, ER, Lz
        ) for idx in range(N)
    )

    results_array = np.array(results_list)
    results_array = results_array.reshape(*theta.shape, 12)
    return results_array
    