import numpy as np
from scipy.linalg import expm, eig, inv


def func_ER_srg(n_groove, n_ridge, top, bottom, front, back, d, Lx, fill_factor, Nx, Nz):
    """
    Numerical model of a slanted grating in rectangular coordinates.

    Parameters:
    - n_groove: Refractive index of the groove
    - n_ridge: Refractive index of the ridge
    - top: Slant angle of the top (alpha_3) [in radians]
    - bottom: Slant angle of the bottom (alpha_4) [in radians]
    - front: Slant angle of the front (alpha_1) [in radians]
    - back: Slant angle of the back (alpha_2) [in radians]
    - d: Depth of the trapezoidal region
    - Lx: Period of the grating
    - fill_factor: Fill factor of the trapezoidal region at the bottom
    - Nx: Number of points along x in the real-space grid
    - Nz: Number of points in the trapezoidal region along z

    Returns:
    - ER: Permittivity of the grating [Nx, Nz]
    - Lz: Thickness of each layer [Nz]
    """
    er_ridge = n_ridge ** 2
    er_groove = n_groove ** 2

    # Critical points
    z1 = 0
    z3 = d
    z4 = d
    x1 = 0
    x2 = x1 + fill_factor * Lx
    x4 = z4 * np.tan(back) + x2
    z5 = (np.tan(top) * x4 + z4) / (1 + np.tan(top) * np.tan(front))
    z6 = (np.tan(bottom) * (Lx - x2)) / (1 + np.tan(back) * np.tan(bottom))

    # Coordinates
    x = np.linspace(-3 * Lx, 4 * Lx, 7 * Nx)
    Nz1 = Nz
    dz1 = z3 / Nz1
    Nz2 = int(round((z5 - z3) / dz1)) if dz1 > 0 else 0
    dz2 = (z5 - z3) / Nz2 if Nz2 > 0 else 0
    Nz = Nz1 + Nz2
    Lz = np.concatenate((np.full(Nz1, dz1), np.full(Nz2, dz2)))

    # Construct grating
    ER = np.zeros((Nz, 7 * Nx))

    # Trapezoid region
    for ii in range(Nz1):
        z_prime = z1 + ii * dz1
        x1_prime = z_prime * np.tan(front) + x1
        x2_prime = z_prime * np.tan(back) + x2
        ER[ii, (x >= x1_prime) & (x <= x2_prime)] = 1

    # Top triangle
    for ii in range(Nz2):
        z_prime = z3 + ii * dz2
        x1_prime = z_prime * np.tan(front) + x1
        x2_prime = x4 - (z_prime - z4) / np.tan(top)
        ER[Nz1 + ii, (x >= x1_prime) & (x <= x2_prime)] = 1

    # Side triangle
    Nz3 = int(np.floor(z6 / dz1))
    for ii in range(Nz3):
        z_prime = z1 + ii * dz1
        x1_prime = z_prime * np.tan(back) + x2
        x2_prime = Lx - z_prime / np.tan(bottom)
        ER[ii, (x >= x1_prime) & (x <= x2_prime)] = 1

    # Combine and assign permittivity
    ER = sum(ER[:, i * Nx : (i + 1) * Nx] for i in range(7))
    ER[ER > 0] = 1
    ER[ER == 1] = er_ridge
    ER[ER == 0] = er_groove

    # Transpose to match output shape [Nx, Nz]
    ER = ER.T

    return ER, Lz


def func_star(SA, SB):
    """
    Redheffer Star Product of two scattering matrices SA and SB.

    Parameters:
    - SA: dict with keys 'S11', 'S12', 'S21', 'S22'
    - SB: dict with keys 'S11', 'S12', 'S21', 'S22'

    Returns:
    - S: dict with combined scattering matrix entries
    """
    I = np.eye(SA['S11'].shape[0])

    inv1 = np.linalg.inv(I - SB['S11'] @ SA['S22'])
    inv2 = np.linalg.inv(I - SA['S22'] @ SB['S11'])

    S11 = SA['S11'] + SA['S12'] @ inv1 @ SB['S11'] @ SA['S21']
    S12 = SA['S12'] @ inv1 @ SB['S12']
    S21 = SB['S21'] @ inv2 @ SA['S21']
    S22 = SB['S22'] + SB['S21'] @ inv2 @ SA['S22'] @ SB['S12']

    return {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}


def func_convmat(A, P, Q=1, R=1):
    """
    Constructs a rectangular convolution matrix from a real-space grid A.

    Parameters:
    - A : 1D, 2D, or 3D numpy array representing the real-space permittivity or similar quantity.
    - P : Number of harmonics in x-dimension (T1)
    - Q : Number of harmonics in y-dimension (T2), default = 1
    - R : Number of harmonics in z-dimension (T3), default = 1

    Returns:
    - C : Convolution matrix of shape (NH, NH), where NH = P * Q * R
    """
    Nx, Ny, Nz = A.shape
    NH = P * Q * R

    # Spatial harmonic indices
    p = np.arange(-P // 2, P // 2 + 1)
    q = np.arange(-Q // 2, Q // 2 + 1)
    r = np.arange(-R // 2, R // 2 + 1)

    # Ensure correct length if P, Q, R are even (mimicking MATLAB's floor behavior)
    p = p[:P]
    q = q[:Q]
    r = r[:R]

    # Compute FFT and shift
    A_fft = np.fft.fftshift(np.fft.fftn(A)) / (Nx * Ny * Nz)

    # Center harmonic indices
    p0 = Nx // 2
    q0 = Ny // 2
    r0 = Nz // 2

    # Initialize convolution matrix
    C = np.zeros((NH, NH), dtype=complex)

    # Map 3D index (prow, qrow, rrow) to row and (pcol, qcol, rcol) to col
    for rrow in range(R):
        for qrow in range(Q):
            for prow in range(P):
                row = rrow * Q * P + qrow * P + prow
                for rcol in range(R):
                    for qcol in range(Q):
                        for pcol in range(P):
                            col = rcol * Q * P + qcol * P + pcol
                            pfft = p[prow] - p[pcol]
                            qfft = q[qrow] - q[qcol]
                            rfft = r[rrow] - r[rcol]
                            idx_p = p0 + pfft
                            idx_q = q0 + qfft
                            idx_r = r0 + rfft
                            # Only include if within bounds
                            if 0 <= idx_p < Nx and 0 <= idx_q < Ny and 0 <= idx_r < Nz:
                                C[row, col] = A_fft[idx_p, idx_q, idx_r]
    return C


def func_rcwa_1d_isotropic(lam0, theta, phi, te, tm, n1, n2, Lx, M, ER, Lz):

    # STEP 1: DASHBOARD
    Ly = 1
    PQ = (2 * M + 1, 1)
    te = te / np.sqrt(abs(te)**2 + abs(tm)**2)
    tm = tm / np.sqrt(abs(te)**2 + abs(tm)**2)

    # STEP 2: BUILD DEVICE LAYERS ON HIGH RESOLUTION GRID
    NLAY = len(Lz)

    # STEP 3: COMPUTE CONVOLUTION MATRICES OF EACH LAYER OF DEVICE
    NH = PQ[0] * PQ[1]
    ERC = np.zeros((NH, NH, NLAY), dtype=complex)
    for n in range(NLAY):
        ERC[:, :, n] = func_convmat(ER[:, n].reshape(-1, 1, 1), PQ[0])

    # STEP 4: COMPUTE WAVE VECTOR EXPANSIONS
    ur1 = ur2 = 1
    er1 = n1**2
    er2 = n2**2
    I = np.eye(NH, dtype=complex)
    Z = np.zeros((NH, NH), dtype=complex)
    k0 = 2 * np.pi / lam0
    kinc = n1 * np.array([np.sin(theta) * np.cos(phi),
                          np.sin(theta) * np.sin(phi),
                          np.cos(theta)])
    m = np.arange(-M, M + 1)
    n = np.array([0])
    kx = kinc[0] - 2 * np.pi * m / (k0 * Lx)
    ky = kinc[1] - 2 * np.pi * n / (k0 * Ly)
    kx[kx == 0] = 1e-7
    ky[ky == 0] = 1e-7
    Ky, Kx = np.meshgrid(ky, kx)
    Kx = np.diag(Kx.flatten())
    Ky = np.diag(Ky.flatten())
    Kzref = -np.conj(np.sqrt(np.conj(er1 * ur1) * I - Kx @ Kx - Ky @ Ky))
    Kztrn = np.conj(np.sqrt(np.conj(er2 * ur2) * I - Kx @ Kx - Ky @ Ky))

    # STEP 5: COMPUTE EIGEN MODES OF FREE SPACE
    Q = np.block([[Kx @ Ky, I + Ky @ Ky], [-(I + Kx @ Kx), -Kx @ Ky]])
    W0 = np.block([[I, Z], [Z, I]])
    V0 = -1j * Q

    # STEP 6: INITIALIZE GLOBAL SCATTERING MATRIX
    II = np.eye(2 * NH, dtype=complex)
    ZZ = np.zeros((2 * NH, 2 * NH), dtype=complex)
    SG = {'S11': ZZ.copy(), 'S12': II.copy(), 'S21': II.copy(), 'S22': ZZ.copy()}

    # STEP 7: MAIN LOOP FOR S-MATRICES THROUGH LAYERS
    for n in range(NLAY):
        ERinv = inv(ERC[:, :, n])
        P = np.block([
            [Kx @ ERinv @ Ky, I - Kx @ ERinv @ Kx],
            [Ky @ ERinv @ Ky - I, -Ky @ ERinv @ Kx]
        ])
        Qm = np.block([
            [Kx @ Ky, ERC[:, :, n] - Kx @ Kx],
            [Ky @ Ky - ERC[:, :, n], -Ky @ Kx]
        ])
        OMEGA2 = P @ Qm
        LAM, W = eig(OMEGA2)
        LAM = np.diag(np.sqrt(LAM + 0j))
        V = Qm @ W @ inv(LAM)
        aa = inv(W) @ W0
        bb = inv(V) @ V0
        A = aa + bb
        B = aa - bb
        X = expm(-LAM * k0 * Lz[n])
        Ainv = inv(A)
        S = {
            'S11': inv(A - X @ B @ Ainv @ X @ B) @ (X @ B @ Ainv @ X @ A - B),
            'S12': inv(A - X @ B @ Ainv @ X @ B) @ X @ (A - B @ Ainv @ B)
        }
        S['S21'] = S['S12']
        S['S22'] = S['S11']
        SG = func_star(SG, S)

    # STEP 8: COMPUTE RELECTION SIDE CONNECTION S-MATRIX
    Qr = (1 / ur1) * np.block([
        [Kx @ Ky, er1 * I - Kx @ Kx],
        [Ky @ Ky - er1 * I, -Ky @ Kx]
    ])
    Wref = np.block([[I, Z], [Z, I]])
    LAMr = np.block([[-1j * Kzref, Z], [Z, -1j * Kzref]])
    Vref = Qr @ inv(LAMr)
    aa = inv(W0) @ Wref
    bb = inv(V0) @ Vref
    A = aa + bb
    B = aa - bb
    Ainv = inv(A)
    SR = {
        'S11': -Ainv @ B,
        'S12': 2 * II @ Ainv,
        'S21': 0.5 * (A - B @ Ainv @ B),
        'S22': B @ Ainv
    }

    # STEP 9: COMPUTE TRANSMISSION SIDE CONNECTION S-MATRIX
    Qt = (1 / ur2) * np.block([
        [Kx @ Ky, er2 * I - Kx @ Kx],
        [Ky @ Ky - er2 * I, -Ky @ Kx]
    ])
    Wtrn = np.block([[I, Z], [Z, I]])
    LAMt = np.block([[1j * Kztrn, Z], [Z, 1j * Kztrn]])
    Vtrn = Qt @ inv(LAMt)
    aa = inv(W0) @ Wtrn
    bb = inv(V0) @ Vtrn
    A = aa + bb
    B = aa - bb
    Ainv = inv(A)
    ST = {
        'S11': B @ Ainv,
        'S12': 0.5 * (A - B @ Ainv @ B),
        'S21': 2 * II @ Ainv,
        'S22': -Ainv @ B
    }

    # STEP 10: UPDATE GLOBAL SCATTERING MATRIX
    SG = func_star(SR, SG)
    SG = func_star(SG, ST)

    # STEP 11: COMPUTE REFLECTION AND TRANSMITTED FIELDS
    delta = np.zeros((NH, 1), dtype=complex)
    delta[NH // 2, 0] = 1
    n_hat = np.array([0, 0, 1])
    if theta == 0:
        ate = np.array([0, 1, 0])
    else:
        ate = np.cross(kinc, n_hat)
        ate /= np.linalg.norm(ate)
    atm = np.cross(ate, kinc)
    atm /= np.linalg.norm(atm)
    EP = te * ate + tm * atm

    esrc = np.zeros((2 * NH, 1), dtype=complex)
    esrc[:NH] = EP[0] * delta
    esrc[NH:] = EP[1] * delta
    csrc = inv(Wref) @ esrc
    cref = SG['S11'] @ csrc
    ctrn = SG['S21'] @ csrc
    eref = Wref @ cref
    etrn = Wtrn @ ctrn

    rx, ry = eref[:NH], eref[NH:]
    rz = -inv(Kzref) @ (Kx @ rx + Ky @ ry)
    tx, ty = etrn[:NH], etrn[NH:]
    tz = -inv(Kztrn) @ (Kx @ tx + Ky @ ty)

    # STEP 12: COMPUTE DIFFRACTION EFFICIENCIES
    r2 = np.abs(rx)**2 + np.abs(ry)**2 + np.abs(rz)**2
    t2 = np.abs(tx)**2 + np.abs(ty)**2 + np.abs(tz)**2

    # Reflection and transmission of all orders
    R = np.real(np.diag(-Kzref)) / np.real(kinc[2]) * r2.flatten()
    T = np.real(np.diag(Kztrn)) / np.real(kinc[2]) * t2.flatten()
    R = R.reshape(PQ)
    T = T.reshape(PQ)

    # STEP 13: Compute Polarization of R & T
    Kref = np.vstack([np.diag(Kx), np.diag(Ky), np.diag(Kzref)])  # shape (3, N)
    Ktrn = np.vstack([np.diag(Kx), np.diag(Ky), np.diag(Kztrn)])
    # Polarization of reflection light
    n_hat_neg = -n_hat.reshape(3, 1)
    teref_hat = np.cross(Kref.T, np.tile(n_hat_neg.T, (Kref.shape[1], 1))).T
    teref_hat /= np.linalg.norm(teref_hat, axis=0, keepdims=True)
    ind = np.logical_and(kx == 1e-7, ky == 1e-7)
    if np.any(ind):
        teref_hat[:, ind] = np.array([[0], [1], [0]])
    tmref_hat = np.cross(teref_hat.T, Kref.T).T
    tmref_hat /= np.linalg.norm(tmref_hat, axis=0, keepdims=True)
    Eref = np.vstack([rx.T, ry.T, rz.T])
    E_te_ref = np.sum(np.conj(Eref) * teref_hat, axis=0)
    E_tm_ref = np.sum(np.conj(Eref) * tmref_hat, axis=0)
    # Polarization of transmission light
    tetrn_hat = np.cross(Kref.T, np.tile(n_hat.reshape(1, 3), (Kref.shape[1], 1))).T
    tetrn_hat /= np.linalg.norm(tetrn_hat, axis=0, keepdims=True)
    if np.any(ind):
        tetrn_hat[:, ind] = np.array([[0], [1], [0]])
    tmtrn_hat = np.cross(tetrn_hat.T, Ktrn.T).T
    tmtrn_hat /= np.linalg.norm(tmtrn_hat, axis=0, keepdims=True)
    Etrn = np.vstack([tx.T, ty.T, tz.T])
    E_te_trn = np.sum(np.conj(Etrn) * tetrn_hat, axis=0)
    E_tm_trn = np.sum(np.conj(Etrn) * tmtrn_hat, axis=0)

    # STEP 14: Compute Diffraction Efficiencies of TE and TM Light
    # REF
    r2_te = np.abs(E_te_ref) ** 2
    r2_tm = np.abs(E_tm_ref) ** 2
    R_te = np.real(-np.diag(Kzref) / ur1) / np.real(kinc[2] / ur1) * r2_te
    R_tm = np.real(-np.diag(Kzref) / ur1) / np.real(kinc[2] / ur1) * r2_tm
    R_te = R_te.reshape(PQ)
    R_tm = R_tm.reshape(PQ)
    # TRN
    t2_te = np.abs(E_te_trn) ** 2
    t2_tm = np.abs(E_tm_trn) ** 2
    T_te = np.real(np.diag(Kztrn) / ur1) / np.real(kinc[2] / ur1) * t2_te
    T_tm = np.real(np.diag(Kztrn) / ur1) / np.real(kinc[2] / ur1) * t2_tm
    T_te = T_te.reshape(PQ)
    T_tm = T_tm.reshape(PQ)

    return E_te_ref.reshape(PQ), E_tm_ref.reshape(PQ), E_te_trn.reshape(PQ), E_tm_trn.reshape(PQ), R, T