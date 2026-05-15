# -*- coding: utf-8 -*-
"""
Functions to calculate the optical and optomechanical properties for cavity with Two-Fano-mode mirror.

| Variable | Notations                   |
| -------- | -------------------------------------------------------- |
|     Plas | P_\text{las}                                             |
|    thlas | \theta_\text{las} (laser phase, irrelevant global phase) |
|       wL | \omega_\text{las}                                        |
|        w | \omega                                                   |
| -------- | -------------------------------------------------------- |
|       wa | \tilde{\omega}_a = \tilde{\Delta}_a - \omega_\text{las}  |
|      wd1 | \tilde{\omega}_{d_1} = \tilde{\Delta}_d_{1} - \omega_\text{las}  |
|      wd2 | \tilde{\omega}_{d_2} = \tilde{\Delta}_d_{2} - \omega_\text{las}  |
|       Da | \tilde{\Delta}_a                                         |
|      Dd1 | \tilde{\Delta}_{d_1}                                         |
|      Dd2 | \tilde{\Delta}_{d_2}                                         |
|       ka | \tilde{\kappa}_a                                         |
|      kd1 | \tilde{\kappa}_{d_1}                                         |
|      kd2 | \tilde{\kappa}_{d_2}                                         |
|      ka0 | {\kappa}_a                                               |
|     kd10 | {\kappa}_{d_1}                                               |
|     kd20 | {\kappa}_{d_1}                                               |
|       ga | \gamma_a                                                 |
|     lbd1 | \lambda_{1}                                                  |
|     lbd2 | \lambda_{2}                                                  |
|       G1 | G_1 = \lambda_{1} - j \sqrt(\kappa_a \kappa_{d_1})           |
|       G2 | G_2 = \lambda_{2} - j \sqrt(\kappa_a \kappa_{d_2})           |
|      K12 | K12 = \sqrt(\kappa_{d_1} \kappa_{d_2})
| -------- | -------------------------------------------------------- |
|       Om | \Omega_\text{mec}                                        |
|      gam | \Gamma_\text{mec}                                        |
|      Nth | \bar{n}_\text{mec}                                       |
| -------- | -------------------------------------------------------- |
|     gwa0 | g^\omega_a                                               |
|    gwd10 | g^\omega_{d_1}                                               |
|    gwd20 | g^\omega_{d_2}                                               |
|     gka0 | g^\kappa_a                                               |
|    gkd10| g^\kappa_{d_1}                                               |
|    gkd20| g^\kappa_{d_2}                                               |
|     gad1p | g^{\kappa,sym}_{1}                                           |
|     gad2p | g^{\kappa,sym}_{2}                                           |
|     gd1d2p | g^{\kappa,sym}_{12}                                           |
|     gad1m | g^{\kappa,asym}_{1}                                          |
|     gad2m | g^{\kappa,asym}_{2}                                          |
|     gd1d2m | g^{\kappa,asym}_{12}                                          |
|     tga  | \tilde{g}_a                                              |
|     tgd1  | \tilde{g}_{d_1}                                              |
|     tgd2  | \tilde{g}_{d_2}                                              |
|     tgma | \tilde{g}_{\text{mec}, a}                                |
|     tgmd1 | \tilde{g}_{\text{mec}, d_1}                                |
|     tgmd2 | \tilde{g}_{\text{mec}, d_2}                                |
| -------- | -------------------------------------------------------- |
|      ass | \bar{a}                                                  |
|      d1ss | \bar{d}_1                                                  |
|      d2ss | \bar{d}_2                                                  |
|      qss | \bar{q}                                                  |
| -------- | -------------------------------------------------------- |

"""

import numpy as np
from scipy import linalg as scilin
from numpy import exp, pi, abs, conjugate, sqrt, arctan, real, imag
from multiprocessing import Pool
from itertools import permutations
atan = arctan
re = real
im = imag

# ------------------------- physical constants --------------------------------

e = 1.602e-19             # elementary charge
h = 6.62606957e-34/(2*pi) # \hbar
kB = 1.3806488e-23        # Boltzmann constant
c = 299792458             # speed of light


# --------------------- bare optical two-mode analysis -------------------------



def scan_device_vs_Ddbar_kdbar(
    sys,
    Ddbar_min_THz,
    Ddbar_max_THz,
    kdbar_min_THz,
    kdbar_max_THz,
    npts=501,
):
    """
    Scan bare optical eigenvalues in H_aBD.

    Upper-panel scan:
        Re(Omega_i)/2pi versus Delta_dbar/2pi.

    Lower-panel scan:
        kappa_i/2pi = -Im(Omega_i)/2pi versus kappa_dbar/2pi.

    Inputs are in ordinary-frequency THz.
    """
    TWOPI = 2*np.pi
    unit_THz = TWOPI * sys.w_scale * 1e12

    # fixed offsets
    dDd = 0.5*(sys.wd1 - sys.wd2)
    dkd = 0.5*(sys.kd1 - sys.kd2)

    # ----------------------------
    # Delta_dbar scan
    # ----------------------------
    Ddbar_vals = np.linspace(
        Ddbar_min_THz*unit_THz,
        Ddbar_max_THz*unit_THz,
        npts,
    )

    eigvals_D_raw = []

    for Ddbar in Ddbar_vals:
        wd1 = Ddbar + dDd
        wd2 = Ddbar - dDd

        H = Hbare_aBdD(
            sys.ga, sys.ka, sys.kd1, sys.kd2,
            sys.lbd1, sys.lbd2,
            sys.wa, wd1, wd2,
        )

        eigvals_D_raw.append(np.linalg.eigvals(H))

    eigs_D = track_eigs_continuously(eigvals_D_raw)

    xD_THz = Ddbar_vals/unit_THz
    Delta_THz = eigs_D.real/unit_THz

    # ----------------------------
    # kappa_dbar scan
    # ----------------------------
    kdbar_vals = np.linspace(
        kdbar_min_THz*unit_THz,
        kdbar_max_THz*unit_THz,
        npts,
    )

    eigvals_k_raw = []

    for kdbar in kdbar_vals:
        kd1 = kdbar + dkd
        kd2 = kdbar - dkd

        H = Hbare_aBdD(
            sys.ga, sys.ka, kd1, kd2,
            sys.lbd1, sys.lbd2,
            sys.wa, sys.wd1, sys.wd2,
        )

        eigvals_k_raw.append(np.linalg.eigvals(H))

    eigs_k = track_eigs_continuously(eigvals_k_raw)

    xk_THz = kdbar_vals/unit_THz
    kappa_THz = (-eigs_k.imag)/unit_THz

    return xD_THz, Delta_THz, eigs_D, xk_THz, kappa_THz, eigs_k

def optical_G1(ka, kd1, lbd1):
    """Bare optical coupling G1 = lbd1 - 1j*sqrt(ka*kd1)."""
    return lbd1 - 1j*sqrt(ka)*sqrt(kd1)


def optical_G2(ka, kd2, lbd2):
    """Bare optical coupling G2 = lbd2 - 1j*sqrt(ka*kd2)."""
    return lbd2 - 1j*sqrt(ka)*sqrt(kd2)


def optical_K12(kd1, kd2):
    """Bare mirror-mode dissipative coupling K12 = sqrt(kd1*kd2)."""
    return sqrt(kd1)*sqrt(kd2)


def optical_kbar(kd1, kd2):
    """Average mirror linewidth kbar = (kd1 + kd2)/2."""
    return (kd1 + kd2)/2


def optical_lbdbar(lbd1, lbd2):
    """Average coherent coupling lbdbar = (lbd1 + lbd2)/2."""
    return (lbd1 + lbd2)/2


def optical_bare_parameters(ka, kd1, kd2, lbd1, lbd2):
    """
    Return useful bare optical parameters:

        G1 = lbd1 - 1j*sqrt(ka*kd1)
        G2 = lbd2 - 1j*sqrt(ka*kd2)
        K12 = sqrt(kd1*kd2)
        kbar = (kd1 + kd2)/2
        lbdbar = (lbd1 + lbd2)/2
    """
    G1 = optical_G1(ka, kd1, lbd1)
    G2 = optical_G2(ka, kd2, lbd2)
    K12 = optical_K12(kd1, kd2)
    kbar = optical_kbar(kd1, kd2)
    lbdbar = optical_lbdbar(lbd1, lbd2)

    return G1, G2, K12, kbar, lbdbar


def U_BD():
    """
    Basis transformation from (a,d1,d2) to (a,dB,dD).

        dB = (d1 + d2)/sqrt(2)
        dD = (d1 - d2)/sqrt(2)

    Therefore

        [a, dB, dD]^T = U_BD [a, d1, d2]^T.
    """
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1/sqrt(2), 1/sqrt(2)],
        [0.0, 1/sqrt(2), -1/sqrt(2)]
    ], dtype=complex)


def Hbare_ad1d2(ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2):
    """
    Bare optical non-Hermitian matrix in the (a,d1,d2) basis.

        H =
        [[wa  - 1j*(ka + ga),  G1,            G2],
         [G1,                   wd1 - 1j*kd1, -1j*K12],
         [G2,                  -1j*K12,       wd2 - 1j*kd2]]

    with

        G1  = lbd1 - 1j*sqrt(ka*kd1)
        G2  = lbd2 - 1j*sqrt(ka*kd2)
        K12 = sqrt(kd1*kd2).
    """
    G1 = optical_G1(ka, kd1, lbd1)
    G2 = optical_G2(ka, kd2, lbd2)
    K12 = optical_K12(kd1, kd2)

    return np.array([
        [wa - 1j*(ka + ga), G1, G2],
        [G1, wd1 - 1j*kd1, -1j*K12],
        [G2, -1j*K12, wd2 - 1j*kd2]
    ], dtype=complex)


def Hbare_aBdD(ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2):
    """
    Bare optical non-Hermitian matrix in the (a,dB,dD) basis.

        H_aBdD = U_BD H_ad1d2 U_BD.T

    Since U_BD is real orthogonal, U_BD.T is its inverse.
    """
    U = U_BD()
    H = Hbare_ad1d2(ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2)

    return U @ H @ U.T


def sorted_eigvals(vals):
    """Sort complex eigenvalues by real part and then imaginary part."""
    return np.array(sorted(vals, key=lambda z: (np.real(z), np.imag(z))), dtype=complex)


def eigs_ad1d2(ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2):
    """Eigenvalues of Hbare_ad1d2 sorted by real part."""
    return sorted_eigvals(
        np.linalg.eigvals(Hbare_ad1d2(ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2))
    )


def eigs_aBdD(ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2):
    """Eigenvalues of Hbare_aBdD sorted by real part."""
    return sorted_eigvals(
        np.linalg.eigvals(Hbare_aBdD(ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2))
    )


def check_bare_basis_exactness(
    ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2,
    rtol=1e-10, atol=1e-12
):
    """
    Check that the bare optical eigenvalues are identical in the
    (a,d1,d2) and (a,dB,dD) bases.

    Returns:
        ok, eigs_ad1d2, eigs_aBdD, max_err
    """
    vals_ad1d2 = eigs_ad1d2(ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2)
    vals_aBdD = eigs_aBdD(ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2)

    ok = np.allclose(vals_ad1d2, vals_aBdD, rtol=rtol, atol=atol)
    max_err = np.max(np.abs(vals_ad1d2 - vals_aBdD))

    return ok, vals_ad1d2, vals_aBdD, max_err


def set_bare_scan_parameter(
    xname, x,
    ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2
):
    """
    Set one bare optical scan parameter.

    Supported xname values:

        "ga", "ka", "kd1", "kd2", "lbd1", "lbd2",
        "wa", "wd1", "wd2",
        "kbar", "kdbar", "lbdbar", "K12"

    Conventions:

        kbar  = kdbar = (kd1 + kd2)/2
        lbdbar = (lbd1 + lbd2)/2
        K12 = sqrt(kd1*kd2)

    For xname == "kbar" or "kdbar", the difference kd1-kd2 is kept fixed.

    For xname == "lbdbar", the difference lbd1-lbd2 is kept fixed.

    For xname == "K12", the ratio kd1/kd2 is kept fixed.
    """
    if xname == "ga":
        ga = x
    elif xname == "ka":
        ka = x
    elif xname == "kd1":
        kd1 = x
    elif xname == "kd2":
        kd2 = x
    elif xname == "lbd1":
        lbd1 = x
    elif xname == "lbd2":
        lbd2 = x
    elif xname == "wa":
        wa = x
    elif xname == "wd1":
        wd1 = x
    elif xname == "wd2":
        wd2 = x
    elif xname in ["kbar", "kdbar"]:
        dkd = (kd1 - kd2)/2
        kd1 = x + dkd
        kd2 = x - dkd
    elif xname == "lbdbar":
        dlbd = (lbd1 - lbd2)/2
        lbd1 = x + dlbd
        lbd2 = x - dlbd
    elif xname == "K12":
        ratio = kd1/kd2
        sr = sqrt(ratio)
        kd1 = x*sr
        kd2 = x/sr
    else:
        raise ValueError("Unsupported xname = " + str(xname))

    return ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2


def scan_bare_eigs(
    xvals, xname,
    ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2,
    basis="ad1d2"
):
    """
    Scan the three bare optical eigenvalues versus one parameter.

    basis can be:
        "ad1d2" for the (a,d1,d2) basis
        "aBdD" for the (a,dB,dD) basis

    Returns:
        eigs, shape = (len(xvals), 3)
    """
    vals = np.zeros((len(xvals), 3), dtype=complex)

    for i, x in enumerate(xvals):
        pars = set_bare_scan_parameter(
            xname, x,
            ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2
        )

        if basis == "ad1d2":
            vals[i] = eigs_ad1d2(*pars)
        elif basis == "aBdD":
            vals[i] = eigs_aBdD(*pars)
        else:
            raise ValueError("basis must be 'ad1d2' or 'aBdD'")

    return vals


def scan_bare_basis_exactness(
    xvals, xname,
    ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2,
    rtol=1e-10, atol=1e-12
):
    """
    Scan eigenvalues in both bases and check exactness over the full scan.

    Returns:
        ok, eigs_ad1d2_scan, eigs_aBdD_scan, max_err
    """
    vals_ad1d2 = scan_bare_eigs(
        xvals, xname,
        ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2,
        basis="ad1d2"
    )

    vals_aBdD = scan_bare_eigs(
        xvals, xname,
        ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2,
        basis="aBdD"
    )

    ok = np.allclose(vals_ad1d2, vals_aBdD, rtol=rtol, atol=atol)
    max_err = np.max(np.abs(vals_ad1d2 - vals_aBdD))

    return ok, vals_ad1d2, vals_aBdD, max_err


def kappa_from_bare_eigs(eigs):
    """
    Convert complex bare optical eigenvalues to linewidths:

        k_i = -Im(Omega_i).
    """
    return -np.imag(eigs)


def bare_eig_real_parts(eigs):
    """Return Re(Omega_i) from complex bare optical eigenvalues."""
    return np.real(eigs)


def bare_eig_imag_parts(eigs):
    """Return Im(Omega_i) from complex bare optical eigenvalues."""
    return np.imag(eigs)


def bare_kappa_minima(xvals, eigs):
    """
    Find the minimum of each linewidth branch

        k_i(x) = -Im(Omega_i(x)).

    Returns a dictionary with branch minima and the global minimum.
    """
    kappas = kappa_from_bare_eigs(eigs)
    out = {}

    for i in range(3):
        j = np.argmin(kappas[:, i])
        out["k" + str(i + 1) + "_min"] = kappas[j, i]
        out["x_at_k" + str(i + 1) + "_min"] = xvals[j]
        out["Omega" + str(i + 1) + "_at_k" + str(i + 1) + "_min"] = eigs[j, i]

    jflat = np.argmin(kappas)
    ix, imode = np.unravel_index(jflat, kappas.shape)

    out["k_global_min"] = kappas[ix, imode]
    out["x_at_k_global_min"] = xvals[ix]
    out["mode_of_k_global_min"] = imode + 1
    out["Omega_at_k_global_min"] = eigs[ix, imode]

    return out


def scan_bare_kappa_minima(
    xvals, xname,
    ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2,
    basis="ad1d2"
):
    """
    Scan bare optical eigenvalues and return the linewidth minima.

    Returns:
        eigs, minima
    """
    eigs = scan_bare_eigs(
        xvals, xname,
        ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2,
        basis=basis
    )

    minima = bare_kappa_minima(xvals, eigs)

    return eigs, minima


# --------------------------------- optics ------------------------------------

def t_CM(ga, ka, kd1, kd2, lbd1, lbd2, wL, wa, wd1, wd2):
    """Two-mode amplitude transmission <b_out,R>/<b_in,L>."""

    G1 = lbd1 - 1j*sqrt(ka)*sqrt(kd1)
    G2 = lbd2 - 1j*sqrt(ka)*sqrt(kd2)

    A  = ka + ga - 1j*(wL - wa)
    D1 = kd1     - 1j*(wL - wd1)
    D2 = kd2     - 1j*(wL - wd2)

    K12 = sqrt(kd1)*sqrt(kd2)

    Den = (
        A*(D1*D2 - K12**2)
        + G1**2*D2
        + G2**2*D1
        - 2*G1*G2*K12
    )

    Num = (
        -sqrt(ka)*(wL - wd1)*(wL - wd2)
        - lbd1*sqrt(kd1)*(wL - wd2)
        - lbd2*sqrt(kd2)*(wL - wd1)
    )

    return -2*sqrt(ga)*Num/Den

def r_CM(ga, ka, kd1, kd2, lbd1, lbd2, wL, wa, wd1, wd2):
    """
    Two-mode amplitude reflection.

    Convention:
        r_CM = 1 - (sqrt(2*ka)*abar
                    + sqrt(2*kd1)*d1bar
                    + sqrt(2*kd2)*d2bar)/alpha_las

        G_i = lbd_i - 1j*sqrt(ka*kd_i)
        K12 = sqrt(kd1*kd2)
    """

    Da  = wa  - wL
    Dd1 = wd1 - wL
    Dd2 = wd2 - wL

    G1 = lbd1 - 1j*sqrt(ka)*sqrt(kd1)
    G2 = lbd2 - 1j*sqrt(ka)*sqrt(kd2)

    A  = 1j*Da  + ka  + ga
    D1 = 1j*Dd1 + kd1
    D2 = 1j*Dd2 + kd2

    K12 = sqrt(kd1)*sqrt(kd2)

    Den = (
        A*(D1*D2 - K12**2)
        + G1**2*D2
        + G2**2*D1
        - 2*G1*G2*K12
    )

    # These are defined by
    # abar  = sqrt(2)*alpha_las*Na/Den
    # d1bar = sqrt(2)*alpha_las*Nd1/Den
    # d2bar = sqrt(2)*alpha_las*Nd2/Den

    Na = (
        sqrt(ka)*(D1*D2 - K12**2)
        - 1j*G1*(sqrt(kd1)*D2 - K12*sqrt(kd2))
        + 1j*G2*(sqrt(kd1)*K12 - D1*sqrt(kd2))
    )

    Nd1 = (
        A*(sqrt(kd1)*D2 - K12*sqrt(kd2))
        - 1j*sqrt(ka)*(G1*D2 - K12*G2)
        + G2**2*sqrt(kd1)
        - G1*G2*sqrt(kd2)
    )

    Nd2 = (
        A*(D1*sqrt(kd2) - K12*sqrt(kd1))
        + 1j*sqrt(ka)*(G1*K12 - D1*G2)
        + G1**2*sqrt(kd2)
        - G1*G2*sqrt(kd1)
    )

    return 1 - 2*(sqrt(ka)*Na + sqrt(kd1)*Nd1 + sqrt(kd2)*Nd2)/Den

def dTs(
    ga, ka, kd1, kd2, lbd1, lbd2, wL, wa, wd1, wd2,
    eps=1e-6
):
    """
    Numerical derivatives of the two-mode transmission T = |t_CM_2mode|^2.

    Returns:
        dTdDa, dTdDd1, dTdDd2, dTdka, dTdkd1, dTdkd2, T

    Since
        Da  = wa  - wL
        Dd1 = wd1 - wL
        Dd2 = wd2 - wL

    at fixed wL:
        d/dDa  = d/dwa
        d/dDd1 = d/dwd1
        d/dDd2 = d/dwd2
    """

    def T_val(ga_, ka_, kd1_, kd2_, lbd1_, lbd2_, wL_, wa_, wd1_, wd2_):
        t = t_CM_2mode(
            ga_, ka_, kd1_, kd2_,
            lbd1_, lbd2_,
            wL_, wa_, wd1_, wd2_
        )
        return abs(t)**2

    T = T_val(ga, ka, kd1, kd2, lbd1, lbd2, wL, wa, wd1, wd2)

    # Derivative with respect to Da = wa - wL
    dTdDa = (
        T_val(ga, ka, kd1, kd2, lbd1, lbd2, wL, wa + eps, wd1, wd2)
        - T_val(ga, ka, kd1, kd2, lbd1, lbd2, wL, wa - eps, wd1, wd2)
    )/(2*eps)

    # Derivative with respect to Dd1 = wd1 - wL
    dTdDd1 = (
        T_val(ga, ka, kd1, kd2, lbd1, lbd2, wL, wa, wd1 + eps, wd2)
        - T_val(ga, ka, kd1, kd2, lbd1, lbd2, wL, wa, wd1 - eps, wd2)
    )/(2*eps)

    # Derivative with respect to Dd2 = wd2 - wL
    dTdDd2 = (
        T_val(ga, ka, kd1, kd2, lbd1, lbd2, wL, wa, wd1, wd2 + eps)
        - T_val(ga, ka, kd1, kd2, lbd1, lbd2, wL, wa, wd1, wd2 - eps)
    )/(2*eps)

    # Derivative with respect to ka
    dTdka = (
        T_val(ga, ka + eps, kd1, kd2, lbd1, lbd2, wL, wa, wd1, wd2)
        - T_val(ga, ka - eps, kd1, kd2, lbd1, lbd2, wL, wa, wd1, wd2)
    )/(2*eps)

    # Derivative with respect to kd1
    dTdkd1 = (
        T_val(ga, ka, kd1 + eps, kd2, lbd1, lbd2, wL, wa, wd1, wd2)
        - T_val(ga, ka, kd1 - eps, kd2, lbd1, lbd2, wL, wa, wd1, wd2)
    )/(2*eps)

    # Derivative with respect to kd2
    dTdkd2 = (
        T_val(ga, ka, kd1, kd2 + eps, lbd1, lbd2, wL, wa, wd1, wd2)
        - T_val(ga, ka, kd1, kd2 - eps, lbd1, lbd2, wL, wa, wd1, wd2)
    )/(2*eps)

    return dTdDa, dTdDd1, dTdDd2, dTdka, dTdkd1, dTdkd2, T

def tw_all(ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2):
    """
    Eigenvalues of the two-mode non-Hermitian matrix

        H = [[Da - 1j*(ka + ga),  G1,   G2 ],
             [G1,                 Dd1 - 1j*kd1,  -1j*K12],
             [G2,                 -1j*K12,       Dd2 - 1j*kd2]]

    Here:
        Da  = wa
        Dd1 = wd1
        Dd2 = wd2

        G1  = lbd1 - 1j*sqrt(ka*kd1)
        G2  = lbd2 - 1j*sqrt(ka*kd2)
        K12 = sqrt(kd1*kd2)

    Returns eigenvalues sorted by real part.
    """

    A  = wa  - 1j*(ka + ga)
    D1 = wd1 - 1j*kd1
    D2 = wd2 - 1j*kd2

    G1 = lbd1 - 1j*sqrt(ka)*sqrt(kd1)
    G2 = lbd2 - 1j*sqrt(ka)*sqrt(kd2)

    K12 = sqrt(kd1)*sqrt(kd2)

    H = np.array([
        [A,   G1,        G2],
        [G1,  D1, -1j*K12],
        [G2, -1j*K12,    D2]
    ], dtype=complex)

    vals = np.linalg.eigvals(H)

    # fixed ordering: lowest real part to highest real part
    vals = sorted(vals, key=lambda z: (np.real(z), np.imag(z)))

    return vals

def tw_1(ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2):
    """First effective complex eigenfrequency."""
    return tw_all(ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2)[0]

def tw_2(ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2):
    """Second effective complex eigenfrequency."""
    return tw_all(ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2)[1]

def tw_3(ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2):
    """Third effective complex eigenfrequency."""
    return tw_all(ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2)[2]

def g10(
    ga,
    gka0, gkd10, gkd20,
    gwa0, gwd10, gwd20,
    ka, kd1, kd2,
    lbd1, lbd2,
    wa, wd1, wd2,
    dq=1e-7
):
    """
    g10 = -1/sqrt(2) * d tw_1 / dq at q = 0.

    Assumes tw_1 is already defined.
    """

    rt2 = sqrt(2)

    ka_p  = ka  + rt2*gka0*dq
    kd1_p = kd1 + rt2*gkd10*dq
    kd2_p = kd2 + rt2*gkd20*dq
    wa_p  = wa  - rt2*gwa0*dq
    wd1_p = wd1 - rt2*gwd10*dq
    wd2_p = wd2 - rt2*gwd20*dq

    ka_m  = ka  - rt2*gka0*dq
    kd1_m = kd1 - rt2*gkd10*dq
    kd2_m = kd2 - rt2*gkd20*dq
    wa_m  = wa  + rt2*gwa0*dq
    wd1_m = wd1 + rt2*gwd10*dq
    wd2_m = wd2 + rt2*gwd20*dq

    Omega_1_p = tw_1(ga, ka_p, kd1_p, kd2_p, lbd1, lbd2, wa_p, wd1_p, wd2_p)
    Omega_1_m = tw_1(ga, ka_m, kd1_m, kd2_m, lbd1, lbd2, wa_m, wd1_m, wd2_m)

    return -(Omega_1_p - Omega_1_m)/(2*dq*rt2)

def g20(
    ga,
    gka0, gkd10, gkd20,
    gwa0, gwd10, gwd20,
    ka, kd1, kd2,
    lbd1, lbd2,
    wa, wd1, wd2,
    dq=1e-7
):
    """
    g20 = -1/sqrt(2) * d tw_2 / dq at q = 0.

    Assumes tw_2 is already defined.
    """

    rt2 = sqrt(2)

    ka_p  = ka  + rt2*gka0*dq
    kd1_p = kd1 + rt2*gkd10*dq
    kd2_p = kd2 + rt2*gkd20*dq
    wa_p  = wa  - rt2*gwa0*dq
    wd1_p = wd1 - rt2*gwd10*dq
    wd2_p = wd2 - rt2*gwd20*dq

    ka_m  = ka  - rt2*gka0*dq
    kd1_m = kd1 - rt2*gkd10*dq
    kd2_m = kd2 - rt2*gkd20*dq
    wa_m  = wa  + rt2*gwa0*dq
    wd1_m = wd1 + rt2*gwd10*dq
    wd2_m = wd2 + rt2*gwd20*dq

    Omega_2_p = tw_2(ga, ka_p, kd1_p, kd2_p, lbd1, lbd2, wa_p, wd1_p, wd2_p)
    Omega_2_m = tw_2(ga, ka_m, kd1_m, kd2_m, lbd1, lbd2, wa_m, wd1_m, wd2_m)

    return -(Omega_2_p - Omega_2_m)/(2*dq*rt2)

def g30(
    ga,
    gka0, gkd10, gkd20,
    gwa0, gwd10, gwd20,
    ka, kd1, kd2,
    lbd1, lbd2,
    wa, wd1, wd2,
    dq=1e-7
):
    """
    g30 = -1/sqrt(2) * d tw_3 / dq at q = 0.

    Assumes tw_3 is already defined.
    """

    rt2 = sqrt(2)

    ka_p  = ka  + rt2*gka0*dq
    kd1_p = kd1 + rt2*gkd10*dq
    kd2_p = kd2 + rt2*gkd20*dq
    wa_p  = wa  - rt2*gwa0*dq
    wd1_p = wd1 - rt2*gwd10*dq
    wd2_p = wd2 - rt2*gwd20*dq

    ka_m  = ka  - rt2*gka0*dq
    kd1_m = kd1 - rt2*gkd10*dq
    kd2_m = kd2 - rt2*gkd20*dq
    wa_m  = wa  + rt2*gwa0*dq
    wd1_m = wd1 + rt2*gwd10*dq
    wd2_m = wd2 + rt2*gwd20*dq

    Omega_3_p = tw_3(ga, ka_p, kd1_p, kd2_p, lbd1, lbd2, wa_p, wd1_p, wd2_p)
    Omega_3_m = tw_3(ga, ka_m, kd1_m, kd2_m, lbd1, lbd2, wa_m, wd1_m, wd2_m)

    return -(Omega_3_p - Omega_3_m)/(2*dq*rt2)

# ------------------------------ steady state ---------------------------------

def Na_ss(Plas, ga, ka, kd1, kd2, lbd1, lbd2, wL, wa, wd1, wd2):
    """
    Two-mode steady-state cavity photon number:

        Na_ss = |abar|^2

    with

        abar = sqrt(2)*alpha_las*Num/Den

    and

        |alpha_las|^2 = Plas/(h*wL)

    Assumes h is defined globally.
    """

    Da  = wa  - wL
    Dd1 = wd1 - wL
    Dd2 = wd2 - wL

    G1 = lbd1 - 1j*sqrt(ka)*sqrt(kd1)
    G2 = lbd2 - 1j*sqrt(ka)*sqrt(kd2)

    A  = 1j*Da  + ka  + ga
    D1 = 1j*Dd1 + kd1
    D2 = 1j*Dd2 + kd2

    K12 = sqrt(kd1)*sqrt(kd2)

    Den = (
        A*(D1*D2 - K12**2)
        + G1**2*D2
        + G2**2*D1
        - 2*G1*G2*K12
    )

    Num = (
        -sqrt(ka)*Dd1*Dd2
        + lbd1*sqrt(kd1)*Dd2
        + lbd2*sqrt(kd2)*Dd1
    )

    return 2*Plas/(h*wL) * abs(Num/Den)**2

def Nd1_ss(Plas, ga, ka, kd1, kd2, lbd1, lbd2, wL, wa, wd1, wd2):
    """
    Two-mode steady-state photon number in d1 mode:

        Nd1_ss = |d1bar|^2

    Assumes h is defined globally.
    """

    Da  = wa  - wL
    Dd1 = wd1 - wL
    Dd2 = wd2 - wL

    G1 = lbd1 - 1j*sqrt(ka)*sqrt(kd1)
    G2 = lbd2 - 1j*sqrt(ka)*sqrt(kd2)

    A  = 1j*Da  + ka  + ga
    D1 = 1j*Dd1 + kd1
    D2 = 1j*Dd2 + kd2

    K12 = sqrt(kd1)*sqrt(kd2)

    Den = (
        A*(D1*D2 - K12**2)
        + G1**2*D2
        + G2**2*D1
        - 2*G1*G2*K12
    )

    Num_d1 = (
        A*(sqrt(kd1)*D2 - K12*sqrt(kd2))
        - 1j*sqrt(ka)*(G1*D2 - K12*G2)
        + G2**2*sqrt(kd1)
        - G1*G2*sqrt(kd2)
    )

    return 2*Plas/(h*wL) * abs(Num_d1/Den)**2

def Nd2_ss(Plas, ga, ka, kd1, kd2, lbd1, lbd2, wL, wa, wd1, wd2):
    """
    Two-mode steady-state photon number in d2 mode:

        Nd2_ss = |d2bar|^2

    Assumes h is defined globally.
    """

    Da  = wa  - wL
    Dd1 = wd1 - wL
    Dd2 = wd2 - wL

    G1 = lbd1 - 1j*sqrt(ka)*sqrt(kd1)
    G2 = lbd2 - 1j*sqrt(ka)*sqrt(kd2)

    A  = 1j*Da  + ka  + ga
    D1 = 1j*Dd1 + kd1
    D2 = 1j*Dd2 + kd2

    K12 = sqrt(kd1)*sqrt(kd2)

    Den = (
        A*(D1*D2 - K12**2)
        + G1**2*D2
        + G2**2*D1
        - 2*G1*G2*K12
    )

    Num_d2 = (
        A*(D1*sqrt(kd2) - K12*sqrt(kd1))
        + 1j*sqrt(ka)*(G1*K12 - D1*G2)
        + G1**2*sqrt(kd2)
        - G1*G2*sqrt(kd1)
    )

    return 2*Plas/(h*wL) * abs(Num_d2/Den)**2

def a_ss(Plas, ga, ka, kd1, kd2, lbd1, lbd2, thlas, wL, wa, wd1, wd2):
    """
    Two-mode steady-state cavity field amplitude abar.

        abar = sqrt(2)*alpha_las*Num_a/Den

    with

        alpha_las = sqrt(Plas/(h*wL))*exp(1j*thlas)

    Assumes h is defined globally.
    """

    Da  = wa  - wL
    Dd1 = wd1 - wL
    Dd2 = wd2 - wL

    alpha_las = sqrt(Plas/(h*wL))*exp(1j*thlas)

    G1 = lbd1 - 1j*sqrt(ka)*sqrt(kd1)
    G2 = lbd2 - 1j*sqrt(ka)*sqrt(kd2)

    A  = 1j*Da  + ka  + ga
    D1 = 1j*Dd1 + kd1
    D2 = 1j*Dd2 + kd2

    K12 = sqrt(kd1)*sqrt(kd2)

    Den = (
        A*(D1*D2 - K12**2)
        + G1**2*D2
        + G2**2*D1
        - 2*G1*G2*K12
    )

    Num_a = (
        -sqrt(ka)*Dd1*Dd2
        + lbd1*sqrt(kd1)*Dd2
        + lbd2*sqrt(kd2)*Dd1
    )

    return sqrt(2)*alpha_las*Num_a/Den

def d1_ss(Plas, ga, ka, kd1, kd2, lbd1, lbd2, thlas, wL, wa, wd1, wd2):
    """
    Two-mode steady-state d1 field amplitude d1bar.

        d1bar = sqrt(2)*alpha_las*Num_d1/Den

    Assumes h is defined globally.
    """

    Da  = wa  - wL
    Dd1 = wd1 - wL
    Dd2 = wd2 - wL

    alpha_las = sqrt(Plas/(h*wL))*exp(1j*thlas)

    G1 = lbd1 - 1j*sqrt(ka)*sqrt(kd1)
    G2 = lbd2 - 1j*sqrt(ka)*sqrt(kd2)

    A  = 1j*Da  + ka  + ga
    D1 = 1j*Dd1 + kd1
    D2 = 1j*Dd2 + kd2

    K12 = sqrt(kd1)*sqrt(kd2)

    Den = (
        A*(D1*D2 - K12**2)
        + G1**2*D2
        + G2**2*D1
        - 2*G1*G2*K12
    )

    Num_d1 = (
        A*(sqrt(kd1)*D2 - K12*sqrt(kd2))
        - 1j*sqrt(ka)*(G1*D2 - K12*G2)
        + G2**2*sqrt(kd1)
        - G1*G2*sqrt(kd2)
    )

    return sqrt(2)*alpha_las*Num_d1/Den

def d2_ss(Plas, ga, ka, kd1, kd2, lbd1, lbd2, thlas, wL, wa, wd1, wd2):
    """
    Two-mode steady-state d2 field amplitude d2bar.

        d2bar = sqrt(2)*alpha_las*Num_d2/Den

    Assumes h is defined globally.
    """

    Da  = wa  - wL
    Dd1 = wd1 - wL
    Dd2 = wd2 - wL

    alpha_las = sqrt(Plas/(h*wL))*exp(1j*thlas)

    G1 = lbd1 - 1j*sqrt(ka)*sqrt(kd1)
    G2 = lbd2 - 1j*sqrt(ka)*sqrt(kd2)

    A  = 1j*Da  + ka  + ga
    D1 = 1j*Dd1 + kd1
    D2 = 1j*Dd2 + kd2

    K12 = sqrt(kd1)*sqrt(kd2)

    Den = (
        A*(D1*D2 - K12**2)
        + G1**2*D2
        + G2**2*D1
        - 2*G1*G2*K12
    )

    Num_d2 = (
        A*(D1*sqrt(kd2) - K12*sqrt(kd1))
        + 1j*sqrt(ka)*(G1*K12 - D1*G2)
        + G1**2*sqrt(kd2)
        - G1*G2*sqrt(kd1)
    )

    return sqrt(2)*alpha_las*Num_d2/Den

def q_ss(
    Om, Plas,
    ga,
    gka0, gkd10, gkd20,
    gwa0, gwd10, gwd20,
    ka, kd1, kd2,
    lbd1, lbd2,
    thlas, wL, wa, wd1, wd2
):
    """
    Zeroth-order two-mode steady-state displacement qbar0.

    Optical amplitudes are evaluated at the effective parameters
    (ka, kd1, kd2, wa, wd1, wd2), and the explicit bare-linewidth
    prefactors are frozen at ka0->ka, kd10->kd1, kd20->kd2.
    """

    # ------------------------------------------------------------
    # optical steady-state building blocks
    # ------------------------------------------------------------
    Xa  = wL - wa
    Xd1 = wL - wd1
    Xd2 = wL - wd2

    K = sqrt(kd1 * kd2)

    Den = (
        -(ga + ka) * Xd1 * Xd2
        - kd1 * Xd2 * Xa
        - kd2 * Xd1 * Xa
        + kd1 * lbd2**2
        + kd2 * lbd1**2
        - 2 * lbd1 * lbd2 * K
        - 2 * lbd1 * sqrt(ka * kd1) * Xd2
        - 2 * lbd2 * sqrt(ka * kd2) * Xd1
        + 1j * (
            Xa * Xd1 * Xd2
            - ga * kd1 * Xd2
            - ga * kd2 * Xd1
            - lbd1**2 * Xd2
            - lbd2**2 * Xd1
        )
    )

    Na = (
        sqrt(ka) * Xd1 * Xd2
        + sqrt(kd1) * lbd1 * Xd2
        + sqrt(kd2) * lbd2 * Xd1
    )

    Nd1 = (
        ga * sqrt(kd1) * Xd2
        + sqrt(kd1) * ka * Xd2
        - sqrt(ka) * lbd1 * Xd2
        - sqrt(kd1) * lbd2**2
        + sqrt(kd2) * lbd1 * lbd2
        + 1j * (
            sqrt(kd1) * Xa * Xd2
            - sqrt(kd2) * K * Xa
            - ga * sqrt(kd2) * K
            + sqrt(ka) * K * lbd2
        )
    )

    Nd2 = (
        ga * sqrt(kd2) * Xd1
        + sqrt(kd2) * ka * Xd1
        - sqrt(ka) * lbd2 * Xd1
        - sqrt(kd2) * lbd1**2
        + sqrt(kd1) * lbd1 * lbd2
        + 1j * (
            sqrt(kd2) * Xa * Xd1
            - sqrt(kd1) * K * Xa
            - ga * sqrt(kd1) * K
            + sqrt(ka) * K * lbd1
        )
    )

    # optical numerators appearing in the fields:
    # ass  = sqrt(2) alpha_las * Ca  / Den
    # d1ss = sqrt(2) alpha_las * C1  / Den
    # d2ss = sqrt(2) alpha_las * C2  / Den
    Ca = -1j * Na
    C1 = Nd1
    C2 = Nd2

    # ------------------------------------------------------------
    # antisymmetric couplings at zeroth order
    # ------------------------------------------------------------
    gad1m0 = sqrt(ka * kd1) * (gka0 / ka - gkd10 / kd1) / 2.0
    gad2m0 = sqrt(ka * kd2) * (gka0 / ka - gkd20 / kd2) / 2.0
    gd1d2m0 = sqrt(kd1 * kd2) * (gkd10 / kd1 - gkd20 / kd2) / 2.0

    # ------------------------------------------------------------
    # grouped real quantities
    # ------------------------------------------------------------
    Sa  = imag(Ca * conjugate(Den))
    Sd1 = imag(C1 * conjugate(Den))
    Sd2 = imag(C2 * conjugate(Den))

    Ia1 = imag(conjugate(Ca) * C1)
    Ia2 = imag(conjugate(Ca) * C2)
    I12 = imag(conjugate(C1) * C2)

    # ------------------------------------------------------------
    # Q0 block
    # ------------------------------------------------------------
    Q0 = (
        sqrt(2) * (
            gwa0 * abs(Na)**2
            + gwd10 * abs(Nd1)**2
            + gwd20 * abs(Nd2)**2
        )
        - (gka0 / sqrt(ka)) * Sa
        - (gkd10 / sqrt(kd1)) * Sd1
        - (gkd20 / sqrt(kd2)) * Sd2
        - 2 * sqrt(2) * (
            gad1m0 * Ia1
            + gad2m0 * Ia2
            + gd1d2m0 * I12
        )
    )

    q0 = 2 * Plas * Q0 / (h * wL * Om * abs(Den)**2)
    return np.real_if_close(q0)

def q_ss_two_mode_lin(
    Om, Plas,
    ga,
    gka0, gkd10, gkd20,
    gwa0, gwd10, gwd20,
    ka, kd1, kd2,
    lbd1, lbd2,
    thlas, wL, wa, wd1, wd2
):
    """
    Linearized two-mode steady-state displacement qbar_lin.

    Approximation used:
      1) ass, d1ss, d2ss are evaluated at the effective optical parameters
         (ka, kd1, kd2, wa, wd1, wd2),
      2) only the explicit dependence on the bare linewidth prefactors
         ka0, kd10, kd20 is linearized in the q-equation.

    The global laser phase thlas does not affect qbar_lin and is not used.
    """

    # ------------------------------------------------------------
    # optical steady-state building blocks
    # ------------------------------------------------------------
    Xa  = wL - wa
    Xd1 = wL - wd1
    Xd2 = wL - wd2

    K = sqrt(kd1 * kd2)

    Den = (
        -(ga + ka) * Xd1 * Xd2
        - kd1 * Xd2 * Xa
        - kd2 * Xd1 * Xa
        + kd1 * lbd2**2
        + kd2 * lbd1**2
        - 2 * lbd1 * lbd2 * K
        - 2 * lbd1 * sqrt(ka * kd1) * Xd2
        - 2 * lbd2 * sqrt(ka * kd2) * Xd1
        + 1j * (
            Xa * Xd1 * Xd2
            - ga * kd1 * Xd2
            - ga * kd2 * Xd1
            - lbd1**2 * Xd2
            - lbd2**2 * Xd1
        )
    )

    Na = (
        sqrt(ka) * Xd1 * Xd2
        + sqrt(kd1) * lbd1 * Xd2
        + sqrt(kd2) * lbd2 * Xd1
    )

    Nd1 = (
        ga * sqrt(kd1) * Xd2
        + sqrt(kd1) * ka * Xd2
        - sqrt(ka) * lbd1 * Xd2
        - sqrt(kd1) * lbd2**2
        + sqrt(kd2) * lbd1 * lbd2
        + 1j * (
            sqrt(kd1) * Xa * Xd2
            - sqrt(kd2) * K * Xa
            - ga * sqrt(kd2) * K
            + sqrt(ka) * K * lbd2
        )
    )

    Nd2 = (
        ga * sqrt(kd2) * Xd1
        + sqrt(kd2) * ka * Xd1
        - sqrt(ka) * lbd2 * Xd1
        - sqrt(kd2) * lbd1**2
        + sqrt(kd1) * lbd1 * lbd2
        + 1j * (
            sqrt(kd2) * Xa * Xd1
            - sqrt(kd1) * K * Xa
            - ga * sqrt(kd1) * K
            + sqrt(ka) * K * lbd1
        )
    )

    Ca = -1j * Na
    C1 = Nd1
    C2 = Nd2

    # ------------------------------------------------------------
    # zero-th order antisymmetric and symmetric couplings
    # ------------------------------------------------------------
    gad1m0 = sqrt(ka * kd1) * (gka0 / ka - gkd10 / kd1) / 2.0
    gad2m0 = sqrt(ka * kd2) * (gka0 / ka - gkd20 / kd2) / 2.0
    gd1d2m0 = sqrt(kd1 * kd2) * (gkd10 / kd1 - gkd20 / kd2) / 2.0

    gad1p0 = sqrt(ka * kd1) * (gka0 / ka + gkd10 / kd1) / 2.0
    gad2p0 = sqrt(ka * kd2) * (gka0 / ka + gkd20 / kd2) / 2.0
    gd1d2p0 = sqrt(kd1 * kd2) * (gkd10 / kd1 + gkd20 / kd2) / 2.0

    # ------------------------------------------------------------
    # grouped real quantities
    # ------------------------------------------------------------
    Sa  = imag(Ca * conjugate(Den))
    Sd1 = imag(C1 * conjugate(Den))
    Sd2 = imag(C2 * conjugate(Den))

    Ia1 = imag(conjugate(Ca) * C1)
    Ia2 = imag(conjugate(Ca) * C2)
    I12 = imag(conjugate(C1) * C2)

    # ------------------------------------------------------------
    # Q0 block (direct substitution with ka0->ka, kd10->kd1, kd20->kd2)
    # ------------------------------------------------------------
    Q0 = (
        sqrt(2) * (
            gwa0 * abs(Na)**2
            + gwd10 * abs(Nd1)**2
            + gwd20 * abs(Nd2)**2
        )
        - (gka0 / sqrt(ka)) * Sa
        - (gkd10 / sqrt(kd1)) * Sd1
        - (gkd20 / sqrt(kd2)) * Sd2
        - 2 * sqrt(2) * (
            gad1m0 * Ia1
            + gad2m0 * Ia2
            + gd1d2m0 * I12
        )
    )

    # ------------------------------------------------------------
    # L block (linearized correction)
    # ------------------------------------------------------------
    L = (
        (gka0**2 / (sqrt(2) * ka**(3/2))) * Sa
        + (gkd10**2 / (sqrt(2) * kd1**(3/2))) * Sd1
        + (gkd20**2 / (sqrt(2) * kd2**(3/2))) * Sd2
        + 2 * gad1p0 * (gka0 / ka - gkd10 / kd1) * Ia1
        + 2 * gad2p0 * (gka0 / ka - gkd20 / kd2) * Ia2
        + 2 * gd1d2p0 * (gkd10 / kd1 - gkd20 / kd2) * I12
    )

    qlin = 2 * Plas * Q0 / (h * wL * Om * abs(Den)**2 + 2 * Plas * L)
    return np.real_if_close(qlin)

# --------------------- effective mechanical properties -----------------------

def g_effs(
    Om,
    Plas,
    ga,
    gka0, gkd10, gkd20,
    gwa0, gwd10, gwd20,
    ka, kd1, kd2,
    lbd1, lbd2,
    thlas,
    wL, wa, wd1, wd2
):
    r"""
    Two-mode effective OM couplings.

    Returns:
        tgma, tgmd1, tgmd2, tga, tgd1, tgd2

    Notation:
        tgma  = \tilde{g}_{mec,a}
        tgmd1 = \tilde{g}_{mec,d_1}
        tgmd2 = \tilde{g}_{mec,d_2}

        tga   = \tilde{g}_a
        tgd1  = \tilde{g}_{d_1}
        tgd2  = \tilde{g}_{d_2}

    Assumes h is defined globally.

    Also assumes q_ss, a_ss_2mode, d1_ss_2mode, d2_ss_2mode
    are already defined.
    """

    # Laser amplitude
    a_las = sqrt(Plas/(h*wL))*exp(1j*thlas)

    # Steady-state fields at the effective parameters
    ass = a_ss_2mode(
        Plas,
        ga,
        ka, kd1, kd2,
        lbd1, lbd2,
        thlas,
        wL, wa, wd1, wd2
    )

    d1ss = d1_ss_2mode(
        Plas,
        ga,
        ka, kd1, kd2,
        lbd1, lbd2,
        thlas,
        wL, wa, wd1, wd2
    )

    d2ss = d2_ss_2mode(
        Plas,
        ga,
        ka, kd1, kd2,
        lbd1, lbd2,
        thlas,
        wL, wa, wd1, wd2
    )

    # Linearized static displacement
    qss = q_ss(
        Om,
        Plas,
        ga,
        gka0, gkd10, gkd20,
        gwa0, gwd10, gwd20,
        ka, kd1, kd2,
        lbd1, lbd2,
        thlas,
        wL, wa, wd1, wd2
    )

    # Bare linewidths from effective linewidths:
    #
    #   ka  = ka0  + sqrt(2)*gka0*qss
    #   kd1 = kd10 + sqrt(2)*gkd10*qss
    #   kd2 = kd20 + sqrt(2)*gkd20*qss
    #
    # Therefore:
    #
    #   ka0  = ka  - sqrt(2)*gka0*qss
    #   kd10 = kd1 - sqrt(2)*gkd10*qss
    #   kd20 = kd2 - sqrt(2)*gkd20*qss

    ka0  = ka  - sqrt(2)*gka0*qss
    kd10 = kd1 - sqrt(2)*gkd10*qss
    kd20 = kd2 - sqrt(2)*gkd20*qss

    # Symmetric/asymmetric dissipative couplings
    gad1p = sqrt(ka0)*sqrt(kd10)*(gka0/ka0 + gkd10/kd10)/2
    gad2p = sqrt(ka0)*sqrt(kd20)*(gka0/ka0 + gkd20/kd20)/2
    gd1d2p = sqrt(kd10)*sqrt(kd20)*(gkd10/kd10 + gkd20/kd20)/2

    gad1m = sqrt(ka0)*sqrt(kd10)*(gka0/ka0 - gkd10/kd10)/2
    gad2m = sqrt(ka0)*sqrt(kd20)*(gka0/ka0 - gkd20/kd20)/2
    gd1d2m = sqrt(kd10)*sqrt(kd20)*(gkd10/kd10 - gkd20/kd20)/2

    # ------------------------------------------------------------
    # Mechanical-force-type effective couplings
    # ------------------------------------------------------------

    tgma = (
        -sqrt(2)*1j*a_las*gka0/(2*sqrt(ka0))
        + ass*gwa0
        + 1j*d1ss*gad1m
        + 1j*d2ss*gad2m
    )

    tgmd1 = (
        -sqrt(2)*1j*a_las*gkd10/(2*sqrt(kd10))
        - 1j*ass*gad1m
        + d1ss*gwd10
        + 1j*d2ss*gd1d2m
    )

    tgmd2 = (
        -sqrt(2)*1j*a_las*gkd20/(2*sqrt(kd20))
        - 1j*ass*gad2m
        - 1j*d1ss*gd1d2m
        + d2ss*gwd20
    )

    # ------------------------------------------------------------
    # Optical-fluctuation effective couplings
    # ------------------------------------------------------------

    tga = (
        ass*gwa0
        + 1j*d1ss*gad1p
        + 1j*d2ss*gad2p
        + 1j*gka0*(-sqrt(2)*a_las/sqrt(ka0) + 2*ass)/2
    )

    tgd1 = (
        1j*ass*gad1p
        + d1ss*gwd10
        + 1j*d2ss*gd1d2p
        + 1j*gkd10*(-sqrt(2)*a_las/sqrt(kd10) + 2*d1ss)/2
    )

    tgd2 = (
        1j*ass*gad2p
        + 1j*d1ss*gd1d2p
        + d2ss*gwd20
        + 1j*gkd20*(-sqrt(2)*a_las/sqrt(kd20) + 2*d2ss)/2
    )

    return tgma, tgmd1, tgmd2, tga, tgd1, tgd2

def Xmeff(
    Nth,
    Om,
    Plas,
    ga,
    gam,
    gka0, gkd10, gkd20,
    gwa0, gwd10, gwd20,
    ka, kd1, kd2,
    lbd1, lbd2,
    thlas,
    w,
    wL, wa, wd1, wd2
):
    """Two-mode effective mechanical susceptibility."""

    # ----------------------------
    # Detunings
    # ----------------------------
    Da  = -wL + wa
    Dd1 = -wL + wd1
    Dd2 = -wL + wd2

    # ----------------------------
    # Couplings appearing in fluctuation matrix
    # cG_i = i G_i = sqrt(ka*kdi) + i lbd_i
    # ----------------------------
    cG1 = sqrt(ka)*sqrt(kd1) + 1j*lbd1
    cG2 = sqrt(ka)*sqrt(kd2) + 1j*lbd2
    K12 = sqrt(kd1)*sqrt(kd2)

    # ----------------------------
    # Laser amplitude
    # ----------------------------
    a_las = sqrt(Plas/h/wL)*exp(1j*thlas)

    # ----------------------------
    # Optical denominators at +w
    # ----------------------------
    chi_a_inv  = ka  + ga + 1j*(Da  - w)
    chi_d1_inv = kd1      + 1j*(Dd1 - w)
    chi_d2_inv = kd2      + 1j*(Dd2 - w)

    Dw = (
        chi_a_inv*(chi_d1_inv*chi_d2_inv - K12**2)
        - cG1**2*chi_d2_inv
        - cG2**2*chi_d1_inv
        + 2*cG1*cG2*K12
    )

    # ----------------------------
    # Optical denominators at -w
    # ----------------------------
    chi_a_inv_m  = ka  + ga + 1j*(Da  + w)
    chi_d1_inv_m = kd1      + 1j*(Dd1 + w)
    chi_d2_inv_m = kd2      + 1j*(Dd2 + w)

    Dmw = (
        chi_a_inv_m*(chi_d1_inv_m*chi_d2_inv_m - K12**2)
        - cG1**2*chi_d2_inv_m
        - cG2**2*chi_d1_inv_m
        + 2*cG1*cG2*K12
    )

    # ----------------------------
    # Steady-state fields
    # ----------------------------
    ass = a_ss_2mode(
        Plas, ga,
        ka, kd1, kd2,
        lbd1, lbd2,
        thlas,
        wL, wa, wd1, wd2
    )

    d1ss = d1_ss_2mode(
        Plas, ga,
        ka, kd1, kd2,
        lbd1, lbd2,
        thlas,
        wL, wa, wd1, wd2
    )

    d2ss = d2_ss_2mode(
        Plas, ga,
        ka, kd1, kd2,
        lbd1, lbd2,
        thlas,
        wL, wa, wd1, wd2
    )

    # ----------------------------
    # Static displacement
    # ----------------------------
    qss = q_ss(
        Om,
        Plas,
        ga,
        gka0, gkd10, gkd20,
        gwa0, gwd10, gwd20,
        ka, kd1, kd2,
        lbd1, lbd2,
        thlas,
        wL, wa, wd1, wd2
    )

    # ----------------------------
    # Bare linewidths reconstructed from effective linewidths
    # ka = ka0 + sqrt(2)*gka0*qss
    # ----------------------------
    ka0  = -sqrt(2)*gka0*qss  + ka
    kd10 = -sqrt(2)*gkd10*qss + kd1
    kd20 = -sqrt(2)*gkd20*qss + kd2

    # ----------------------------
    # Symmetric/asymmetric dissipative couplings
    # ----------------------------
    gad1m = sqrt(ka0)*sqrt(kd10)*(gka0/ka0 - gkd10/kd10)/2
    gad2m = sqrt(ka0)*sqrt(kd20)*(gka0/ka0 - gkd20/kd20)/2
    gd1d2m = sqrt(kd10)*sqrt(kd20)*(gkd10/kd10 - gkd20/kd20)/2

    gad1p = sqrt(ka0)*sqrt(kd10)*(gka0/ka0 + gkd10/kd10)/2
    gad2p = sqrt(ka0)*sqrt(kd20)*(gka0/ka0 + gkd20/kd20)/2
    gd1d2p = sqrt(kd10)*sqrt(kd20)*(gkd10/kd10 + gkd20/kd20)/2

    # ----------------------------
    # Mechanical-force effective couplings
    # Same structure as your one-mode tgma, tgmd
    # ----------------------------
    tgma = (
        -sqrt(2)*1j*a_las*gka0/(2*sqrt(ka0))
        + ass*gwa0
        + 1j*d1ss*gad1m
        + 1j*d2ss*gad2m
    )

    tgmd1 = (
        -sqrt(2)*1j*a_las*gkd10/(2*sqrt(kd10))
        - 1j*ass*gad1m
        + d1ss*gwd10
        + 1j*d2ss*gd1d2m
    )

    tgmd2 = (
        -sqrt(2)*1j*a_las*gkd20/(2*sqrt(kd20))
        - 1j*ass*gad2m
        - 1j*d1ss*gd1d2m
        + d2ss*gwd20
    )

    # ----------------------------
    # Optical-fluctuation effective couplings
    # Same structure as your one-mode tga, tgd
    # ----------------------------
    tga = (
        ass*gwa0
        + 1j*d1ss*gad1p
        + 1j*d2ss*gad2p
        + 1j*gka0*(-sqrt(2)*a_las/sqrt(ka0) + 2*ass)/2
    )

    tgd1 = (
        1j*ass*gad1p
        + d1ss*gwd10
        + 1j*d2ss*gd1d2p
        + 1j*gkd10*(-sqrt(2)*a_las/sqrt(kd10) + 2*d1ss)/2
    )

    tgd2 = (
        1j*ass*gad2p
        + 1j*d1ss*gd1d2p
        + d2ss*gwd20
        + 1j*gkd20*(-sqrt(2)*a_las/sqrt(kd20) + 2*d2ss)/2
    )

    # ----------------------------
    # Conjugates
    # ----------------------------
    tgma_cc  = tgma.conjugate()
    tgmd1_cc = tgmd1.conjugate()
    tgmd2_cc = tgmd2.conjugate()

    # ----------------------------
    # C_q coefficients at +w
    # C_q = i M^{-1} t
    # ----------------------------
    Ca_q = 1j*(
        (chi_d1_inv*chi_d2_inv - K12**2)*tga
        + (cG2*K12 - cG1*chi_d2_inv)*tgd1
        + (cG1*K12 - cG2*chi_d1_inv)*tgd2
    )/Dw

    Cd1_q = 1j*(
        (cG2*K12 - cG1*chi_d2_inv)*tga
        + (chi_a_inv*chi_d2_inv - cG2**2)*tgd1
        + (cG1*cG2 - chi_a_inv*K12)*tgd2
    )/Dw

    Cd2_q = 1j*(
        (cG1*K12 - cG2*chi_d1_inv)*tga
        + (cG1*cG2 - chi_a_inv*K12)*tgd1
        + (chi_a_inv*chi_d1_inv - cG1**2)*tgd2
    )/Dw

    # ----------------------------
    # C_q^*(-w) coefficients
    # ----------------------------
    Ca_qm_cc = (
        1j*(
            (chi_d1_inv_m*chi_d2_inv_m - K12**2)*tga
            + (cG2*K12 - cG1*chi_d2_inv_m)*tgd1
            + (cG1*K12 - cG2*chi_d1_inv_m)*tgd2
        )/Dmw
    ).conjugate()

    Cd1_qm_cc = (
        1j*(
            (cG2*K12 - cG1*chi_d2_inv_m)*tga
            + (chi_a_inv_m*chi_d2_inv_m - cG2**2)*tgd1
            + (cG1*cG2 - chi_a_inv_m*K12)*tgd2
        )/Dmw
    ).conjugate()

    Cd2_qm_cc = (
        1j*(
            (cG1*K12 - cG2*chi_d1_inv_m)*tga
            + (cG1*cG2 - chi_a_inv_m*K12)*tgd1
            + (chi_a_inv_m*chi_d1_inv_m - cG1**2)*tgd2
        )/Dmw
    ).conjugate()

    # ----------------------------
    # Optical self-energy
    # ----------------------------
    Xopti = (
        -2*(tgma_cc*Ca_q   + tgma*Ca_qm_cc)
        -2*(tgmd1_cc*Cd1_q + tgmd1*Cd1_qm_cc)
        -2*(tgmd2_cc*Cd2_q + tgmd2*Cd2_qm_cc)
    )

    # ----------------------------
    # Effective mechanical susceptibility
    # ----------------------------
    return 1/(Xopti + (Om**2 - 1j*gam*w - w**2)/Om)

def Xopt_inv_components(
    Nth,
    Om,
    Plas,
    ga,
    gam,
    gka0, gkd10, gkd20,
    gwa0, gwd10, gwd20,
    ka, kd1, kd2,
    lbd1, lbd2,
    thlas,
    w,
    wL, wa, wd1, wd2
):
    """
    Components of two-mode chi_opt^{-1}.

    Returns:
        Xopti_a,
        Xopti_d1,
        Xopti_d2,
        Xopti_ad1,
        Xopti_ad2,
        Xopti_d1d2

    Assumes h, q_ss, a_ss_2mode, d1_ss_2mode, d2_ss_2mode are defined.
    """

    # ----------------------------
    # Detunings
    # ----------------------------
    Da  = -wL + wa
    Dd1 = -wL + wd1
    Dd2 = -wL + wd2

    # ----------------------------
    # Couplings in fluctuation matrix
    # cG_i = i G_i = sqrt(ka*kdi) + i*lbd_i
    # ----------------------------
    cG1 = sqrt(ka)*sqrt(kd1) + 1j*lbd1
    cG2 = sqrt(ka)*sqrt(kd2) + 1j*lbd2
    K12 = sqrt(kd1)*sqrt(kd2)

    # ----------------------------
    # Laser amplitude
    # ----------------------------
    a_las = sqrt(Plas/h/wL)*exp(1j*thlas)

    # ----------------------------
    # Susceptibility entries at +w
    # ----------------------------
    chi_a_inv  = ka  + ga + 1j*(Da  - w)
    chi_d1_inv = kd1      + 1j*(Dd1 - w)
    chi_d2_inv = kd2      + 1j*(Dd2 - w)

    Dw = (
        chi_a_inv*(chi_d1_inv*chi_d2_inv - K12**2)
        - cG1**2*chi_d2_inv
        - cG2**2*chi_d1_inv
        + 2*cG1*cG2*K12
    )

    # ----------------------------
    # Susceptibility entries at -w
    # ----------------------------
    chi_a_inv_m  = ka  + ga + 1j*(Da  + w)
    chi_d1_inv_m = kd1      + 1j*(Dd1 + w)
    chi_d2_inv_m = kd2      + 1j*(Dd2 + w)

    Dmw = (
        chi_a_inv_m*(chi_d1_inv_m*chi_d2_inv_m - K12**2)
        - cG1**2*chi_d2_inv_m
        - cG2**2*chi_d1_inv_m
        + 2*cG1*cG2*K12
    )

    # ----------------------------
    # Steady-state fields
    # ----------------------------
    ass = a_ss_2mode(
        Plas, ga,
        ka, kd1, kd2,
        lbd1, lbd2,
        thlas,
        wL, wa, wd1, wd2
    )

    d1ss = d1_ss_2mode(
        Plas, ga,
        ka, kd1, kd2,
        lbd1, lbd2,
        thlas,
        wL, wa, wd1, wd2
    )

    d2ss = d2_ss_2mode(
        Plas, ga,
        ka, kd1, kd2,
        lbd1, lbd2,
        thlas,
        wL, wa, wd1, wd2
    )

    # ----------------------------
    # Static displacement
    # ----------------------------
    qss = q_ss(
        Om,
        Plas,
        ga,
        gka0, gkd10, gkd20,
        gwa0, gwd10, gwd20,
        ka, kd1, kd2,
        lbd1, lbd2,
        thlas,
        wL, wa, wd1, wd2
    )

    # ----------------------------
    # Bare linewidths reconstructed from effective linewidths
    # ka = ka0 + sqrt(2)*gka0*qss
    # ----------------------------
    ka0  = -sqrt(2)*gka0*qss  + ka
    kd10 = -sqrt(2)*gkd10*qss + kd1
    kd20 = -sqrt(2)*gkd20*qss + kd2

    # ----------------------------
    # Symmetric/asymmetric dissipative couplings
    # ----------------------------
    gad1m = sqrt(ka0)*sqrt(kd10)*(gka0/ka0 - gkd10/kd10)/2
    gad2m = sqrt(ka0)*sqrt(kd20)*(gka0/ka0 - gkd20/kd20)/2
    gd1d2m = sqrt(kd10)*sqrt(kd20)*(gkd10/kd10 - gkd20/kd20)/2

    gad1p = sqrt(ka0)*sqrt(kd10)*(gka0/ka0 + gkd10/kd10)/2
    gad2p = sqrt(ka0)*sqrt(kd20)*(gka0/ka0 + gkd20/kd20)/2
    gd1d2p = sqrt(kd10)*sqrt(kd20)*(gkd10/kd10 + gkd20/kd20)/2

    # ----------------------------
    # Mechanical-force couplings
    # m_a, m_1, m_2
    # ----------------------------
    tgma = (
        -sqrt(2)*1j*a_las*gka0/(2*sqrt(ka0))
        + ass*gwa0
        + 1j*d1ss*gad1m
        + 1j*d2ss*gad2m
    )

    tgmd1 = (
        -sqrt(2)*1j*a_las*gkd10/(2*sqrt(kd10))
        - 1j*ass*gad1m
        + d1ss*gwd10
        + 1j*d2ss*gd1d2m
    )

    tgmd2 = (
        -sqrt(2)*1j*a_las*gkd20/(2*sqrt(kd20))
        - 1j*ass*gad2m
        - 1j*d1ss*gd1d2m
        + d2ss*gwd20
    )

    # ----------------------------
    # Optical-fluctuation couplings
    # t_a, t_1, t_2
    # ----------------------------
    tga = (
        ass*gwa0
        + 1j*d1ss*gad1p
        + 1j*d2ss*gad2p
        + 1j*gka0*(-sqrt(2)*a_las/sqrt(ka0) + 2*ass)/2
    )

    tgd1 = (
        1j*ass*gad1p
        + d1ss*gwd10
        + 1j*d2ss*gd1d2p
        + 1j*gkd10*(-sqrt(2)*a_las/sqrt(kd10) + 2*d1ss)/2
    )

    tgd2 = (
        1j*ass*gad2p
        + 1j*d1ss*gd1d2p
        + d2ss*gwd20
        + 1j*gkd20*(-sqrt(2)*a_las/sqrt(kd20) + 2*d2ss)/2
    )

    # ----------------------------
    # Conjugates
    # ----------------------------
    tgma_cc  = tgma.conjugate()
    tgmd1_cc = tgmd1.conjugate()
    tgmd2_cc = tgmd2.conjugate()

    # ----------------------------
    # Cofactors at +w
    # ----------------------------
    Saa = chi_d1_inv*chi_d2_inv - K12**2
    S11 = chi_a_inv*chi_d2_inv - cG2**2
    S22 = chi_a_inv*chi_d1_inv - cG1**2

    Sa1 = cG2*K12 - cG1*chi_d2_inv
    Sa2 = cG1*K12 - cG2*chi_d1_inv
    S12 = cG1*cG2 - chi_a_inv*K12

    # ----------------------------
    # Cofactors at -w
    # ----------------------------
    Saa_m = chi_d1_inv_m*chi_d2_inv_m - K12**2
    S11_m = chi_a_inv_m*chi_d2_inv_m - cG2**2
    S22_m = chi_a_inv_m*chi_d1_inv_m - cG1**2

    Sa1_m = cG2*K12 - cG1*chi_d2_inv_m
    Sa2_m = cG1*K12 - cG2*chi_d1_inv_m
    S12_m = cG1*cG2 - chi_a_inv_m*K12

    # ----------------------------
    # Components of chi_opt^{-1}
    #
    # chi_opt^{-1} =
    #   Xopti_a + Xopti_d1 + Xopti_d2
    # + Xopti_ad1 + Xopti_ad2 + Xopti_d1d2
    # ----------------------------

    Xopti_a = (
        -2j*(tgma_cc*tga*Saa/Dw)
        + (-2j*(tgma_cc*tga*Saa_m/Dmw)).conjugate()
    )

    Xopti_d1 = (
        -2j*(tgmd1_cc*tgd1*S11/Dw)
        + (-2j*(tgmd1_cc*tgd1*S11_m/Dmw)).conjugate()
    )

    Xopti_d2 = (
        -2j*(tgmd2_cc*tgd2*S22/Dw)
        + (-2j*(tgmd2_cc*tgd2*S22_m/Dmw)).conjugate()
    )

    Xopti_ad1 = (
        -2j*((tgma_cc*tgd1 + tgmd1_cc*tga)*Sa1/Dw)
        + (-2j*((tgma_cc*tgd1 + tgmd1_cc*tga)*Sa1_m/Dmw)).conjugate()
    )

    Xopti_ad2 = (
        -2j*((tgma_cc*tgd2 + tgmd2_cc*tga)*Sa2/Dw)
        + (-2j*((tgma_cc*tgd2 + tgmd2_cc*tga)*Sa2_m/Dmw)).conjugate()
    )

    Xopti_d1d2 = (
        -2j*((tgmd1_cc*tgd2 + tgmd2_cc*tgd1)*S12/Dw)
        + (-2j*((tgmd1_cc*tgd2 + tgmd2_cc*tgd1)*S12_m/Dmw)).conjugate()
    )

    return Xopti_a, Xopti_d1, Xopti_d2, Xopti_ad1, Xopti_ad2, Xopti_d1d2

def Xopt_inv(
    Om,
    Plas,
    ga,
    gka0, gkd10, gkd20,
    gwa0, gwd10, gwd20,
    ka, kd1, kd2,
    lbd1, lbd2,
    thlas,
    w,
    wL, wa, wd1, wd2
):
    """
    Optical contribution to the inverse effective mechanical susceptibility
    for the two-mode case.

        Xopt_inv = chi_opt^{-1}

    Assumes h, q_ss, a_ss_2mode, d1_ss_2mode, d2_ss_2mode are already defined.
    """

    # ------------------------------------------------------------
    # Detunings
    # ------------------------------------------------------------
    Da  = -wL + wa
    Dd1 = -wL + wd1
    Dd2 = -wL + wd2

    # ------------------------------------------------------------
    # Couplings in optical fluctuation matrix
    # cG_i = i G_i = sqrt(ka*kdi) + i*lbd_i
    # ------------------------------------------------------------
    cG1 = sqrt(ka)*sqrt(kd1) + 1j*lbd1
    cG2 = sqrt(ka)*sqrt(kd2) + 1j*lbd2
    K12 = sqrt(kd1)*sqrt(kd2)

    # ------------------------------------------------------------
    # Laser amplitude
    # ------------------------------------------------------------
    a_las = sqrt(Plas/(h*wL))*exp(1j*thlas)

    # ------------------------------------------------------------
    # Optical susceptibilities at +w
    # ------------------------------------------------------------
    chi_a_inv  = ka  + ga + 1j*(Da  - w)
    chi_d1_inv = kd1      + 1j*(Dd1 - w)
    chi_d2_inv = kd2      + 1j*(Dd2 - w)

    Dw = (
        chi_a_inv*(chi_d1_inv*chi_d2_inv - K12**2)
        - cG1**2*chi_d2_inv
        - cG2**2*chi_d1_inv
        + 2*cG1*cG2*K12
    )

    # ------------------------------------------------------------
    # Optical susceptibilities at -w
    # ------------------------------------------------------------
    chi_a_inv_m  = ka  + ga + 1j*(Da  + w)
    chi_d1_inv_m = kd1      + 1j*(Dd1 + w)
    chi_d2_inv_m = kd2      + 1j*(Dd2 + w)

    Dmw = (
        chi_a_inv_m*(chi_d1_inv_m*chi_d2_inv_m - K12**2)
        - cG1**2*chi_d2_inv_m
        - cG2**2*chi_d1_inv_m
        + 2*cG1*cG2*K12
    )

    # ------------------------------------------------------------
    # Steady-state fields
    # ------------------------------------------------------------
    ass = a_ss_2mode(
        Plas, ga,
        ka, kd1, kd2,
        lbd1, lbd2,
        thlas,
        wL, wa, wd1, wd2
    )

    d1ss = d1_ss_2mode(
        Plas, ga,
        ka, kd1, kd2,
        lbd1, lbd2,
        thlas,
        wL, wa, wd1, wd2
    )

    d2ss = d2_ss_2mode(
        Plas, ga,
        ka, kd1, kd2,
        lbd1, lbd2,
        thlas,
        wL, wa, wd1, wd2
    )

    # ------------------------------------------------------------
    # Static displacement
    # ------------------------------------------------------------
    qss = q_ss(
        Om,
        Plas,
        ga,
        gka0, gkd10, gkd20,
        gwa0, gwd10, gwd20,
        ka, kd1, kd2,
        lbd1, lbd2,
        thlas,
        wL, wa, wd1, wd2
    )

    # ------------------------------------------------------------
    # Bare linewidths reconstructed from effective linewidths
    #
    # ka  = ka0  + sqrt(2)*gka0*qss
    # kd1 = kd10 + sqrt(2)*gkd10*qss
    # kd2 = kd20 + sqrt(2)*gkd20*qss
    # ------------------------------------------------------------
    ka0  = -sqrt(2)*gka0*qss  + ka
    kd10 = -sqrt(2)*gkd10*qss + kd1
    kd20 = -sqrt(2)*gkd20*qss + kd2

    # ------------------------------------------------------------
    # Symmetric/asymmetric dissipative couplings
    # ------------------------------------------------------------
    gad1m = sqrt(ka0)*sqrt(kd10)*(gka0/ka0 - gkd10/kd10)/2
    gad2m = sqrt(ka0)*sqrt(kd20)*(gka0/ka0 - gkd20/kd20)/2
    gd1d2m = sqrt(kd10)*sqrt(kd20)*(gkd10/kd10 - gkd20/kd20)/2

    gad1p = sqrt(ka0)*sqrt(kd10)*(gka0/ka0 + gkd10/kd10)/2
    gad2p = sqrt(ka0)*sqrt(kd20)*(gka0/ka0 + gkd20/kd20)/2
    gd1d2p = sqrt(kd10)*sqrt(kd20)*(gkd10/kd10 + gkd20/kd20)/2

    # ------------------------------------------------------------
    # Mechanical-force couplings:
    # tgma, tgmd1, tgmd2
    # ------------------------------------------------------------
    tgma = (
        -sqrt(2)*1j*a_las*gka0/(2*sqrt(ka0))
        + ass*gwa0
        + 1j*d1ss*gad1m
        + 1j*d2ss*gad2m
    )

    tgmd1 = (
        -sqrt(2)*1j*a_las*gkd10/(2*sqrt(kd10))
        - 1j*ass*gad1m
        + d1ss*gwd10
        + 1j*d2ss*gd1d2m
    )

    tgmd2 = (
        -sqrt(2)*1j*a_las*gkd20/(2*sqrt(kd20))
        - 1j*ass*gad2m
        - 1j*d1ss*gd1d2m
        + d2ss*gwd20
    )

    # ------------------------------------------------------------
    # Optical-fluctuation couplings:
    # tga, tgd1, tgd2
    # ------------------------------------------------------------
    tga = (
        ass*gwa0
        + 1j*d1ss*gad1p
        + 1j*d2ss*gad2p
        + 1j*gka0*(-sqrt(2)*a_las/sqrt(ka0) + 2*ass)/2
    )

    tgd1 = (
        1j*ass*gad1p
        + d1ss*gwd10
        + 1j*d2ss*gd1d2p
        + 1j*gkd10*(-sqrt(2)*a_las/sqrt(kd10) + 2*d1ss)/2
    )

    tgd2 = (
        1j*ass*gad2p
        + 1j*d1ss*gd1d2p
        + d2ss*gwd20
        + 1j*gkd20*(-sqrt(2)*a_las/sqrt(kd20) + 2*d2ss)/2
    )

    # ------------------------------------------------------------
    # Cofactors at +w
    # ------------------------------------------------------------
    Saa = chi_d1_inv*chi_d2_inv - K12**2
    Sa1 = cG2*K12 - cG1*chi_d2_inv
    Sa2 = cG1*K12 - cG2*chi_d1_inv

    S11 = chi_a_inv*chi_d2_inv - cG2**2
    S12 = cG1*cG2 - chi_a_inv*K12
    S22 = chi_a_inv*chi_d1_inv - cG1**2

    # ------------------------------------------------------------
    # C_q coefficients at +w
    # C_q = i M^{-1} t
    # ------------------------------------------------------------
    Ca_q = 1j*(Saa*tga + Sa1*tgd1 + Sa2*tgd2)/Dw

    Cd1_q = 1j*(Sa1*tga + S11*tgd1 + S12*tgd2)/Dw

    Cd2_q = 1j*(Sa2*tga + S12*tgd1 + S22*tgd2)/Dw

    # ------------------------------------------------------------
    # Cofactors at -w
    # ------------------------------------------------------------
    Saa_m = chi_d1_inv_m*chi_d2_inv_m - K12**2
    Sa1_m = cG2*K12 - cG1*chi_d2_inv_m
    Sa2_m = cG1*K12 - cG2*chi_d1_inv_m

    S11_m = chi_a_inv_m*chi_d2_inv_m - cG2**2
    S12_m = cG1*cG2 - chi_a_inv_m*K12
    S22_m = chi_a_inv_m*chi_d1_inv_m - cG1**2

    # ------------------------------------------------------------
    # C_q^*(-w) coefficients
    # ------------------------------------------------------------
    Ca_qm_cc = (
        1j*(Saa_m*tga + Sa1_m*tgd1 + Sa2_m*tgd2)/Dmw
    ).conjugate()

    Cd1_qm_cc = (
        1j*(Sa1_m*tga + S11_m*tgd1 + S12_m*tgd2)/Dmw
    ).conjugate()

    Cd2_qm_cc = (
        1j*(Sa2_m*tga + S12_m*tgd1 + S22_m*tgd2)/Dmw
    ).conjugate()

    # ------------------------------------------------------------
    # Optical inverse contribution
    #
    # Xopt_inv =
    # -2 sum_c [
    #     tgmc^* C_q^c(w)
    #   + tgmc C_q^{c*}(-w)
    # ]
    # ------------------------------------------------------------
    Xopti = (
        -2*(tgma.conjugate()*Ca_q + tgma*Ca_qm_cc)
        -2*(tgmd1.conjugate()*Cd1_q + tgmd1*Cd1_qm_cc)
        -2*(tgmd2.conjugate()*Cd2_q + tgmd2*Cd2_qm_cc)
    )

    return Xopti

def dOm_Gopt(
    Om,
    Plas,
    ga,
    gka0, gkd10, gkd20,
    gwa0, gwd10, gwd20,
    ka, kd1, kd2,
    lbd1, lbd2,
    thlas,
    w,
    wL, wa, wd1, wd2
):
    """
    Two-mode mechanical frequency shift and optical damping.

        dOm  = Re[Xopt_inv]/2
        Gopt = -Om Im[Xopt_inv]/w
    """

    Xopti = Xopt_inv_2mode(
        Om,
        Plas,
        ga,
        gka0, gkd10, gkd20,
        gwa0, gwd10, gwd20,
        ka, kd1, kd2,
        lbd1, lbd2,
        thlas,
        w,
        wL, wa, wd1, wd2
    )

    return re(Xopti)/2, -Om*im(Xopti)/w

# -------------------------- Lyapunov equations  ------------------------------

def A_Lyapunov(
    Om,
    Plas,
    ga,
    gam,
    gka0, gkd10, gkd20,
    gwa0, gwd10, gwd20,
    ka, kd1, kd2,
    lbd1, lbd2,
    thlas,
    wL, wa, wd1, wd2
):
    """
    A matrix in the two-mode Lyapunov equation

        dV/dt = A V + V A^T + B

    Quadrature ordering:

        Y = (
            delta X_a,
            delta P_a,
            delta X_d1,
            delta P_d1,
            delta X_d2,
            delta P_d2,
            delta q,
            delta p
        )

    Assumes h, q_ss, a_ss_2mode, d1_ss_2mode, d2_ss_2mode are defined.
    """

    # ------------------------------------------------------------
    # Detunings
    # ------------------------------------------------------------
    Da  = -wL + wa
    Dd1 = -wL + wd1
    Dd2 = -wL + wd2

    # ------------------------------------------------------------
    # Steady-state fields
    # ------------------------------------------------------------
    ass = a_ss_2mode(
        Plas,
        ga,
        ka, kd1, kd2,
        lbd1, lbd2,
        thlas,
        wL, wa, wd1, wd2
    )

    d1ss = d1_ss_2mode(
        Plas,
        ga,
        ka, kd1, kd2,
        lbd1, lbd2,
        thlas,
        wL, wa, wd1, wd2
    )

    d2ss = d2_ss_2mode(
        Plas,
        ga,
        ka, kd1, kd2,
        lbd1, lbd2,
        thlas,
        wL, wa, wd1, wd2
    )

    # ------------------------------------------------------------
    # Static displacement
    # ------------------------------------------------------------
    qss = q_ss(
        Om,
        Plas,
        ga,
        gka0, gkd10, gkd20,
        gwa0, gwd10, gwd20,
        ka, kd1, kd2,
        lbd1, lbd2,
        thlas,
        wL, wa, wd1, wd2
    )

    # ------------------------------------------------------------
    # Bare linewidths reconstructed from effective linewidths
    #
    # ka  = ka0  + sqrt(2)*gka0*qss
    # kd1 = kd10 + sqrt(2)*gkd10*qss
    # kd2 = kd20 + sqrt(2)*gkd20*qss
    # ------------------------------------------------------------
    ka0  = -sqrt(2)*gka0*qss  + ka
    kd10 = -sqrt(2)*gkd10*qss + kd1
    kd20 = -sqrt(2)*gkd20*qss + kd2

    # ------------------------------------------------------------
    # Laser amplitude
    # ------------------------------------------------------------
    a_las = sqrt(Plas/(h*wL))*exp(1j*thlas)

    # ------------------------------------------------------------
    # Dissipative symmetric/asymmetric couplings
    # ------------------------------------------------------------
    gad1m = sqrt(ka0)*sqrt(kd10)*(gka0/ka0 - gkd10/kd10)/2
    gad2m = sqrt(ka0)*sqrt(kd20)*(gka0/ka0 - gkd20/kd20)/2
    gd1d2m = sqrt(kd10)*sqrt(kd20)*(gkd10/kd10 - gkd20/kd20)/2

    gad1p = sqrt(ka0)*sqrt(kd10)*(gka0/ka0 + gkd10/kd10)/2
    gad2p = sqrt(ka0)*sqrt(kd20)*(gka0/ka0 + gkd20/kd20)/2
    gd1d2p = sqrt(kd10)*sqrt(kd20)*(gkd10/kd10 + gkd20/kd20)/2

    # ------------------------------------------------------------
    # Mechanical-force couplings:
    #
    #   tgma  = gtilde_mec,a
    #   tgmd1 = gtilde_mec,d1
    #   tgmd2 = gtilde_mec,d2
    # ------------------------------------------------------------
    tgma = (
        -sqrt(2)*1j*a_las*gka0/(2*sqrt(ka0))
        + ass*gwa0
        + 1j*d1ss*gad1m
        + 1j*d2ss*gad2m
    )

    tgmd1 = (
        -sqrt(2)*1j*a_las*gkd10/(2*sqrt(kd10))
        - 1j*ass*gad1m
        + d1ss*gwd10
        + 1j*d2ss*gd1d2m
    )

    tgmd2 = (
        -sqrt(2)*1j*a_las*gkd20/(2*sqrt(kd20))
        - 1j*ass*gad2m
        - 1j*d1ss*gd1d2m
        + d2ss*gwd20
    )

    # ------------------------------------------------------------
    # Optical-fluctuation couplings:
    #
    #   tga  = gtilde_a
    #   tgd1 = gtilde_d1
    #   tgd2 = gtilde_d2
    # ------------------------------------------------------------
    tga = (
        ass*gwa0
        + 1j*d1ss*gad1p
        + 1j*d2ss*gad2p
        + 1j*gka0*(-sqrt(2)*a_las/sqrt(ka0) + 2*ass)/2
    )

    tgd1 = (
        1j*ass*gad1p
        + d1ss*gwd10
        + 1j*d2ss*gd1d2p
        + 1j*gkd10*(-sqrt(2)*a_las/sqrt(kd10) + 2*d1ss)/2
    )

    tgd2 = (
        1j*ass*gad2p
        + 1j*d1ss*gd1d2p
        + d2ss*gwd20
        + 1j*gkd20*(-sqrt(2)*a_las/sqrt(kd20) + 2*d2ss)/2
    )

    # ------------------------------------------------------------
    # Coherent + dissipative optical couplings
    #
    # G_i = lbd_i - 1j*sqrt(ka*kdi)
    # ------------------------------------------------------------
    G1 = lbd1 - 1j*sqrt(ka)*sqrt(kd1)
    G2 = lbd2 - 1j*sqrt(ka)*sqrt(kd2)

    G1R = np.real(G1)
    G1I = np.imag(G1)

    G2R = np.real(G2)
    G2I = np.imag(G2)

    K12 = sqrt(kd1)*sqrt(kd2)

    # ------------------------------------------------------------
    # Short names for real/imaginary parts
    # ------------------------------------------------------------
    tgaR  = np.real(tga)
    tgaI  = np.imag(tga)

    tgd1R = np.real(tgd1)
    tgd1I = np.imag(tgd1)

    tgd2R = np.real(tgd2)
    tgd2I = np.imag(tgd2)

    tgmaR  = np.real(tgma)
    tgmaI  = np.imag(tgma)

    tgmd1R = np.real(tgmd1)
    tgmd1I = np.imag(tgmd1)

    tgmd2R = np.real(tgmd2)
    tgmd2I = np.imag(tgmd2)

    # ------------------------------------------------------------
    # Drift matrix A
    # ------------------------------------------------------------
    A = np.array([
        [
            -ga - ka,
            Da,
            G1I,
            G1R,
            G2I,
            G2R,
            -2*tgaI,
            0
        ],
        [
            -Da,
            -ga - ka,
            -G1R,
            G1I,
            -G2R,
            G2I,
            2*tgaR,
            0
        ],
        [
            G1I,
            G1R,
            -kd1,
            Dd1,
            -K12,
            0,
            -2*tgd1I,
            0
        ],
        [
            -G1R,
            G1I,
            -Dd1,
            -kd1,
            0,
            -K12,
            2*tgd1R,
            0
        ],
        [
            G2I,
            G2R,
            -K12,
            0,
            -kd2,
            Dd2,
            -2*tgd2I,
            0
        ],
        [
            -G2R,
            G2I,
            0,
            -K12,
            -Dd2,
            -kd2,
            2*tgd2R,
            0
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            Om
        ],
        [
            2*tgmaR,
            2*tgmaI,
            2*tgmd1R,
            2*tgmd1I,
            2*tgmd2R,
            2*tgmd2I,
            -Om,
            -gam
        ]
    ], dtype=complex)

    return np.real_if_close(A)

def B_Lyapunov(
    Nth,
    Om,
    Plas,
    ga,
    gam,
    gka0, gkd10, gkd20,
    gwa0, gwd10, gwd20,
    ka, kd1, kd2,
    lbd1, lbd2,
    thlas,
    wL, wa, wd1, wd2
):
    """
    B matrix in the two-mode Lyapunov equation

        dV/dt = A V + V A^T + B

    Quadrature ordering:

        Y = (
            delta X_a,
            delta P_a,
            delta X_d1,
            delta P_d1,
            delta X_d2,
            delta P_d2,
            delta q,
            delta p
        )

    Assumes h, q_ss, a_ss_2mode, d1_ss_2mode, d2_ss_2mode are defined.
    """

    # Steady-state fields
    ass = a_ss_2mode(
        Plas,
        ga,
        ka, kd1, kd2,
        lbd1, lbd2,
        thlas,
        wL, wa, wd1, wd2
    )

    d1ss = d1_ss_2mode(
        Plas,
        ga,
        ka, kd1, kd2,
        lbd1, lbd2,
        thlas,
        wL, wa, wd1, wd2
    )

    d2ss = d2_ss_2mode(
        Plas,
        ga,
        ka, kd1, kd2,
        lbd1, lbd2,
        thlas,
        wL, wa, wd1, wd2
    )

    # Static displacement
    qss = q_ss(
        Om,
        Plas,
        ga,
        gka0, gkd10, gkd20,
        gwa0, gwd10, gwd20,
        ka, kd1, kd2,
        lbd1, lbd2,
        thlas,
        wL, wa, wd1, wd2
    )

    # Bare linewidths reconstructed from effective linewidths
    ka0  = -sqrt(2)*gka0*qss  + ka
    kd10 = -sqrt(2)*gkd10*qss + kd1
    kd20 = -sqrt(2)*gkd20*qss + kd2

    # Mechanical dissipative-noise coefficients
    #
    # n_p contains:
    #
    #   sqrt(2)*cX*X_in,L + sqrt(2)*cP*P_in,L
    #
    # with
    #
    #   cX = - sum_c gk_c/sqrt(kappa_c0) Im(cbar)
    #   cP = + sum_c gk_c/sqrt(kappa_c0) Re(cbar)

    cX = -(
        gka0*np.imag(ass)/sqrt(ka0)
        + gkd10*np.imag(d1ss)/sqrt(kd10)
        + gkd20*np.imag(d2ss)/sqrt(kd20)
    )

    cP = (
        gka0*np.real(ass)/sqrt(ka0)
        + gkd10*np.real(d1ss)/sqrt(kd10)
        + gkd20*np.real(d2ss)/sqrt(kd20)
    )

    B = np.array([
        [
            ga + ka,
            0,
            sqrt(ka)*sqrt(kd1),
            0,
            sqrt(ka)*sqrt(kd2),
            0,
            0,
            sqrt(ka)*cX
        ],
        [
            0,
            ga + ka,
            0,
            sqrt(ka)*sqrt(kd1),
            0,
            sqrt(ka)*sqrt(kd2),
            0,
            sqrt(ka)*cP
        ],
        [
            sqrt(ka)*sqrt(kd1),
            0,
            kd1,
            0,
            sqrt(kd1)*sqrt(kd2),
            0,
            0,
            sqrt(kd1)*cX
        ],
        [
            0,
            sqrt(ka)*sqrt(kd1),
            0,
            kd1,
            0,
            sqrt(kd1)*sqrt(kd2),
            0,
            sqrt(kd1)*cP
        ],
        [
            sqrt(ka)*sqrt(kd2),
            0,
            sqrt(kd1)*sqrt(kd2),
            0,
            kd2,
            0,
            0,
            sqrt(kd2)*cX
        ],
        [
            0,
            sqrt(ka)*sqrt(kd2),
            0,
            sqrt(kd1)*sqrt(kd2),
            0,
            kd2,
            0,
            sqrt(kd2)*cP
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        ],
        [
            sqrt(ka)*cX,
            sqrt(ka)*cP,
            sqrt(kd1)*cX,
            sqrt(kd1)*cP,
            sqrt(kd2)*cX,
            sqrt(kd2)*cP,
            0,
            gam*(2*Nth + 1) + cX**2 + cP**2
        ]
    ], dtype=complex)

    return np.real_if_close(B)

def neff_2d(
    Nth,
    Om,
    Plas,
    ga,
    gam,
    gka0, gkd10, gkd20,
    gwa0, gwd10, gwd20,
    ka, kd1, kd2,
    lbd1, lbd2,
    thlas,
    wL,
    wa, wd1, wd2,
    threads=8
):
    """
    Two-mode phonon number in the fluctuations.

    Parallel calculation of neff_Lyapunov_linsys_2mode()[2]
    for 2D arrays wL and Plas.

    Inputs:
        wL   : 2D array of laser frequencies
        Plas : 2D array of laser powers

    Returns:
        neff : 2D array with same shape as wL and Plas
    """

    neff = np.zeros_like(wL, dtype=float)
    ps = []

    with Pool(threads) as pool:
        for i in range(neff.shape[0]):
            ps.append([])
            for j in range(neff.shape[1]):
                ps[i].append(
                    pool.apply_async(
                        neff_Lyapunov_linsys_2mode,
                        (
                            Nth,
                            Om,
                            Plas[i, j],
                            ga,
                            gam,
                            gka0, gkd10, gkd20,
                            gwa0, gwd10, gwd20,
                            ka, kd1, kd2,
                            lbd1, lbd2,
                            thlas,
                            wL[i, j],
                            wa, wd1, wd2
                        )
                    )
                )

        for i in range(neff.shape[0]):
            for j in range(neff.shape[1]):
                neff[i, j] = ps[i][j].get()[2][0]

    return neff



# --------------------- bare optical two-mode analysis -------------------------

def optical_G1(ka, kd1, lbd1):
    """Bare optical coupling G1 = lbd1 - 1j*sqrt(ka*kd1)."""
    return lbd1 - 1j*sqrt(ka)*sqrt(kd1)


def optical_G2(ka, kd2, lbd2):
    """Bare optical coupling G2 = lbd2 - 1j*sqrt(ka*kd2)."""
    return lbd2 - 1j*sqrt(ka)*sqrt(kd2)


def optical_K12(kd1, kd2):
    """Bare mirror-mode dissipative coupling K12 = sqrt(kd1*kd2)."""
    return sqrt(kd1)*sqrt(kd2)


def optical_kbar(kd1, kd2):
    """Average mirror linewidth kbar = (kd1 + kd2)/2."""
    return (kd1 + kd2)/2


def optical_lbdbar(lbd1, lbd2):
    """Average coherent coupling lbdbar = (lbd1 + lbd2)/2."""
    return (lbd1 + lbd2)/2


def optical_bare_parameters(ka, kd1, kd2, lbd1, lbd2):
    """
    Return useful bare optical parameters:

        G1 = lbd1 - 1j*sqrt(ka*kd1)
        G2 = lbd2 - 1j*sqrt(ka*kd2)
        K12 = sqrt(kd1*kd2)
        kbar = (kd1 + kd2)/2
        lbdbar = (lbd1 + lbd2)/2
    """
    G1 = optical_G1(ka, kd1, lbd1)
    G2 = optical_G2(ka, kd2, lbd2)
    K12 = optical_K12(kd1, kd2)
    kbar = optical_kbar(kd1, kd2)
    lbdbar = optical_lbdbar(lbd1, lbd2)

    return G1, G2, K12, kbar, lbdbar


def U_BD():
    """
    Basis transformation from (a,d1,d2) to (a,dB,dD).

        dB = (d1 + d2)/sqrt(2)
        dD = (d1 - d2)/sqrt(2)

    Therefore

        [a, dB, dD]^T = U_BD [a, d1, d2]^T.
    """
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1/sqrt(2), 1/sqrt(2)],
        [0.0, 1/sqrt(2), -1/sqrt(2)]
    ], dtype=complex)


def Hbare_ad1d2(ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2):
    """
    Bare optical non-Hermitian matrix in the (a,d1,d2) basis.

        H =
        [[wa  - 1j*(ka + ga),  G1,            G2],
         [G1,                   wd1 - 1j*kd1, -1j*K12],
         [G2,                  -1j*K12,       wd2 - 1j*kd2]]

    with

        G1  = lbd1 - 1j*sqrt(ka*kd1)
        G2  = lbd2 - 1j*sqrt(ka*kd2)
        K12 = sqrt(kd1*kd2).
    """
    G1 = optical_G1(ka, kd1, lbd1)
    G2 = optical_G2(ka, kd2, lbd2)
    K12 = optical_K12(kd1, kd2)

    return np.array([
        [wa - 1j*(ka + ga), G1, G2],
        [G1, wd1 - 1j*kd1, -1j*K12],
        [G2, -1j*K12, wd2 - 1j*kd2]
    ], dtype=complex)


def Hbare_aBdD(ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2):
    """
    Bare optical non-Hermitian matrix in the (a,dB,dD) basis.

        H_aBdD = U_BD H_ad1d2 U_BD.T

    Since U_BD is real orthogonal, U_BD.T is its inverse.
    """
    U = U_BD()
    H = Hbare_ad1d2(ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2)

    return U @ H @ U.T


def sorted_eigvals(vals):
    """Sort complex eigenvalues by real part and then imaginary part."""
    return np.array(sorted(vals, key=lambda z: (np.real(z), np.imag(z))), dtype=complex)


def eigs_ad1d2(ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2):
    """Eigenvalues of Hbare_ad1d2 sorted by real part."""
    return sorted_eigvals(
        np.linalg.eigvals(Hbare_ad1d2(ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2))
    )


def eigs_aBdD(ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2):
    """Eigenvalues of Hbare_aBdD sorted by real part."""
    return sorted_eigvals(
        np.linalg.eigvals(Hbare_aBdD(ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2))
    )


def check_bare_basis_exactness(
    ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2,
    rtol=1e-10, atol=1e-12
):
    """
    Check that the bare optical eigenvalues are identical in the
    (a,d1,d2) and (a,dB,dD) bases.

    Returns:
        ok, eigs_ad1d2, eigs_aBdD, max_err
    """
    vals_ad1d2 = eigs_ad1d2(ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2)
    vals_aBdD = eigs_aBdD(ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2)

    ok = np.allclose(vals_ad1d2, vals_aBdD, rtol=rtol, atol=atol)
    max_err = np.max(np.abs(vals_ad1d2 - vals_aBdD))

    return ok, vals_ad1d2, vals_aBdD, max_err


def set_bare_scan_parameter(
    xname, x,
    ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2
):
    """
    Set one bare optical scan parameter.

    Supported xname values:

        "gamma_a", "ga"
        "kappa_a", "ka"
        "kappa_dbar", "kdbar"
        "kbar", "kappabar"
        "Delta_dbar", "wdbar"
        "Deltabar"
        "lambdabar", "lbdbar"
        "K", "K12"

    Conventions:

        kappa_dbar = (kd1 + kd2)/2
        kbar = kappabar = (ka + ga + kd1 + kd2)/3
        Delta_dbar = (wd1 + wd2)/2
        Deltabar = (wa + wd1 + wd2)/3
        lambdabar = (lbd1 + lbd2)/2
        K = K12 = sqrt(kd1*kd2)

    For kappa_dbar, Delta_dbar, and lambdabar, the corresponding
    difference between the two mirror modes is kept fixed.

    For kbar, ka, kd1, and kd2 are shifted together while ga is kept fixed.

    For Deltabar, wa, wd1, and wd2 are shifted together.

    For K, the ratio kd1/kd2 is kept fixed.
    """
    if xname in ["gamma_a", "ga"]:
        ga = x

    elif xname in ["kappa_a", "ka"]:
        ka = x

    elif xname in ["kappa_dbar", "kdbar"]:
        dkd = (kd1 - kd2)/2
        kd1 = x + dkd
        kd2 = x - dkd

    elif xname in ["kbar", "kappabar"]:
        kbar0 = optical_kappabar_full(ga, ka, kd1, kd2)
        dk = x - kbar0
        ka = ka + dk
        kd1 = kd1 + dk
        kd2 = kd2 + dk

    elif xname in ["Delta_dbar", "wdbar"]:
        dwd = (wd1 - wd2)/2
        wd1 = x + dwd
        wd2 = x - dwd

    elif xname == "Deltabar":
        Deltabar0 = optical_Deltabar(wa, wd1, wd2)
        dD = x - Deltabar0
        wa = wa + dD
        wd1 = wd1 + dD
        wd2 = wd2 + dD

    elif xname in ["lambdabar", "lbdbar"]:
        dlbd = (lbd1 - lbd2)/2
        lbd1 = x + dlbd
        lbd2 = x - dlbd

    elif xname in ["K", "K12"]:
        ratio = kd1/kd2
        sr = sqrt(ratio)
        kd1 = x*sr
        kd2 = x/sr

    else:
        raise ValueError("Unsupported xname = " + str(xname))

    return ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2


def scan_bare_eigs(
    xvals, xname,
    ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2,
    basis="ad1d2"
):
    """
    Scan the three bare optical eigenvalues versus one parameter.

    basis can be:
        "ad1d2" for the (a,d1,d2) basis
        "aBdD" for the (a,dB,dD) basis

    Returns:
        eigs, shape = (len(xvals), 3)
    """
    vals = np.zeros((len(xvals), 3), dtype=complex)

    for i, x in enumerate(xvals):
        pars = set_bare_scan_parameter(
            xname, x,
            ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2
        )

        if basis == "ad1d2":
            vals[i] = eigs_ad1d2(*pars)
        elif basis == "aBdD":
            vals[i] = eigs_aBdD(*pars)
        else:
            raise ValueError("basis must be 'ad1d2' or 'aBdD'")

    return vals


def scan_bare_basis_exactness(
    xvals, xname,
    ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2,
    rtol=1e-10, atol=1e-12
):
    """
    Scan eigenvalues in both bases and check exactness over the full scan.

    Returns:
        ok, eigs_ad1d2_scan, eigs_aBdD_scan, max_err
    """
    vals_ad1d2 = scan_bare_eigs(
        xvals, xname,
        ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2,
        basis="ad1d2"
    )

    vals_aBdD = scan_bare_eigs(
        xvals, xname,
        ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2,
        basis="aBdD"
    )

    ok = np.allclose(vals_ad1d2, vals_aBdD, rtol=rtol, atol=atol)
    max_err = np.max(np.abs(vals_ad1d2 - vals_aBdD))

    return ok, vals_ad1d2, vals_aBdD, max_err


def kappa_from_bare_eigs(eigs):
    """
    Convert complex bare optical eigenvalues to linewidths:

        k_i = -Im(Omega_i).
    """
    return -np.imag(eigs)


def bare_eig_real_parts(eigs):
    """Return Re(Omega_i) from complex bare optical eigenvalues."""
    return np.real(eigs)


def bare_eig_imag_parts(eigs):
    """Return Im(Omega_i) from complex bare optical eigenvalues."""
    return np.imag(eigs)


def bare_kappa_minima(xvals, eigs):
    """
    Find the minimum of each linewidth branch

        k_i(x) = -Im(Omega_i(x)).

    Returns a dictionary with branch minima and the global minimum.
    """
    kappas = kappa_from_bare_eigs(eigs)
    out = {}

    for i in range(3):
        j = np.argmin(kappas[:, i])
        out["k" + str(i + 1) + "_min"] = kappas[j, i]
        out["x_at_k" + str(i + 1) + "_min"] = xvals[j]
        out["Omega" + str(i + 1) + "_at_k" + str(i + 1) + "_min"] = eigs[j, i]

    jflat = np.argmin(kappas)
    ix, imode = np.unravel_index(jflat, kappas.shape)

    out["k_global_min"] = kappas[ix, imode]
    out["x_at_k_global_min"] = xvals[ix]
    out["mode_of_k_global_min"] = imode + 1
    out["Omega_at_k_global_min"] = eigs[ix, imode]

    return out


def scan_bare_kappa_minima(
    xvals, xname,
    ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2,
    basis="ad1d2"
):
    """
    Scan bare optical eigenvalues and return the linewidth minima.

    Returns:
        eigs, minima
    """
    eigs = scan_bare_eigs(
        xvals, xname,
        ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2,
        basis=basis
    )

    minima = bare_kappa_minima(xvals, eigs)

    return eigs, minima


# -------- centered bare optical eigenvalue scans for figure files -------------

def optical_wdbar(wd1, wd2):
    """Average mirror-mode detuning/frequency wdbar = (wd1 + wd2)/2."""
    return (wd1 + wd2)/2


def optical_Deltabar(wa, wd1, wd2):
    """
    Average optical detuning/frequency

        Deltabar = (wa + 2*wdbar)/3
                 = (wa + wd1 + wd2)/3.
    """
    return (wa + wd1 + wd2)/3


def optical_kappabar_full(ga, ka, kd1, kd2):
    """
    Average optical linewidth

        kappabar = (ka + ga + 2*kdbar)/3
                 = (ka + ga + kd1 + kd2)/3.
    """
    return (ka + ga + kd1 + kd2)/3


def set_centered_bare_scan_parameter(
    xname, x,
    ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2
):
    """
    Set one scan parameter for centered bare optical eigenvalue plots.

    Supported xname values:

        gamma_a, ga
        kappa_a, ka
        kappa_dbar, kdbar
        kbar, kappabar
        Delta_dbar, wdbar
        Deltabar
        lambdabar, lbdbar
        K, K12

    Conventions:

        kappa_dbar = (kd1 + kd2)/2
        kbar = kappabar = (ka + ga + kd1 + kd2)/3
        Delta_dbar = (wd1 + wd2)/2
        Deltabar = (wa + wd1 + wd2)/3
        lambdabar = (lbd1 + lbd2)/2
        K = K12 = sqrt(kd1*kd2)

    For kappa_dbar, wdbar, and lambdabar, the corresponding difference
    between the two mirror modes is kept fixed.

    For kbar, ka, kd1, and kd2 are shifted together while ga is kept fixed.

    For Deltabar, wa, wd1, and wd2 are shifted together.

    For K, the ratio kd1/kd2 is kept fixed.
    """
    if xname in ["gamma_a", "ga"]:
        ga = x

    elif xname in ["kappa_a", "ka"]:
        ka = x

    elif xname in ["kappa_dbar", "kdbar"]:
        dkd = (kd1 - kd2)/2
        kd1 = x + dkd
        kd2 = x - dkd

    elif xname in ["kbar", "kappabar"]:
        kbar0 = optical_kappabar_full(ga, ka, kd1, kd2)
        dk = x - kbar0
        ka = ka + dk
        kd1 = kd1 + dk
        kd2 = kd2 + dk

    elif xname in ["Delta_dbar", "wdbar"]:
        dwd = (wd1 - wd2)/2
        wd1 = x + dwd
        wd2 = x - dwd

    elif xname in ["Deltabar"]:
        Deltabar0 = optical_Deltabar(wa, wd1, wd2)
        dD = x - Deltabar0
        wa = wa + dD
        wd1 = wd1 + dD
        wd2 = wd2 + dD

    elif xname in ["lambdabar", "lbdbar"]:
        dlbd = (lbd1 - lbd2)/2
        lbd1 = x + dlbd
        lbd2 = x - dlbd

    elif xname in ["K", "K12"]:
        ratio = kd1/kd2
        sr = sqrt(ratio)
        kd1 = x*sr
        kd2 = x/sr

    else:
        raise ValueError("Unsupported xname = " + str(xname))

    return ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2


def scan_centered_bare_eigs(
    xvals, xname,
    ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2,
    basis="ad1d2"
):
    """
    Scan bare optical eigenvalues and return centered real and imaginary parts.

    The raw eigenvalues are Omega_i.

    The centered quantities are

        Re_centered_i = Re(Omega_i) - Deltabar

    and

        Im_centered_i = Im(Omega_i) + kappabar,

    because for a bare optical eigenvalue

        Omega_i = Re(Omega_i) - 1j*kappa_i,

    and therefore Im(Omega_i) = -kappa_i.

    Returns:
        eigs          : complex array, shape (len(xvals), 3)
        real_centered : real array, shape (len(xvals), 3)
        imag_centered : real array, shape (len(xvals), 3)
        Deltabar      : real array, shape (len(xvals),)
        kappabar      : real array, shape (len(xvals),)
    """
    eigs = np.zeros((len(xvals), 3), dtype=complex)
    Deltabar = np.zeros(len(xvals), dtype=float)
    kappabar = np.zeros(len(xvals), dtype=float)

    for i, x in enumerate(xvals):
        pars = set_centered_bare_scan_parameter(
            xname, x,
            ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2
        )

        ga_i, ka_i, kd1_i, kd2_i, lbd1_i, lbd2_i, wa_i, wd1_i, wd2_i = pars

        if basis == "ad1d2":
            eigs[i] = eigs_ad1d2(*pars)
        elif basis == "aBdD":
            eigs[i] = eigs_aBdD(*pars)
        else:
            raise ValueError("basis must be 'ad1d2' or 'aBdD'")

        Deltabar[i] = optical_Deltabar(wa_i, wd1_i, wd2_i)
        kappabar[i] = optical_kappabar_full(ga_i, ka_i, kd1_i, kd2_i)

    real_centered = np.real(eigs) - Deltabar[:, None]
    imag_centered = np.imag(eigs) + kappabar[:, None]

    return eigs, real_centered, imag_centered, Deltabar, kappabar


def scan_centered_bare_eigs_both_bases(
    xvals, xname,
    ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2,
    rtol=1e-10, atol=1e-12
):
    """
    Scan centered bare optical eigenvalues in both bases.

    Returns:
        ok
        max_err
        eigs_ad1d2_scan
        eigs_aBdD_scan
        real_centered
        imag_centered
        Deltabar
        kappabar

    The centered real/imaginary arrays are computed from the ad1d2 basis.
    """
    eigs1, real_centered, imag_centered, Deltabar, kappabar = scan_centered_bare_eigs(
        xvals, xname,
        ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2,
        basis="ad1d2"
    )

    eigs2 = scan_centered_bare_eigs(
        xvals, xname,
        ga, ka, kd1, kd2, lbd1, lbd2, wa, wd1, wd2,
        basis="aBdD"
    )[0]

    ok = np.allclose(eigs1, eigs2, rtol=rtol, atol=atol)
    max_err = np.max(np.abs(eigs1 - eigs2))

    return ok, max_err, eigs1, eigs2, real_centered, imag_centered, Deltabar, kappabar


def centered_kappa_from_bare_eigs(eigs, kappabar):
    """
    Return centered linewidths

        kappa_i - kappabar,

    where kappa_i = -Im(Omega_i).
    """
    return -np.imag(eigs) - np.asarray(kappabar)[:, None]


def centered_bare_kappa_minima(xvals, eigs, kappabar):
    """
    Find minima of the absolute linewidths kappa_i = -Im(Omega_i)
    and also return centered linewidths kappa_i - kappabar.
    """
    kappas = -np.imag(eigs)
    kappas_centered = kappas - np.asarray(kappabar)[:, None]

    out = {}

    for i in range(3):
        j = np.argmin(kappas[:, i])
        out["k" + str(i + 1) + "_min"] = kappas[j, i]
        out["k" + str(i + 1) + "_min_centered"] = kappas_centered[j, i]
        out["x_at_k" + str(i + 1) + "_min"] = xvals[j]
        out["Omega" + str(i + 1) + "_at_k" + str(i + 1) + "_min"] = eigs[j, i]

    jflat = np.argmin(kappas)
    ix, imode = np.unravel_index(jflat, kappas.shape)

    out["k_global_min"] = kappas[ix, imode]
    out["k_global_min_centered"] = kappas_centered[ix, imode]
    out["x_at_k_global_min"] = xvals[ix]
    out["mode_of_k_global_min"] = imode + 1
    out["Omega_at_k_global_min"] = eigs[ix, imode]

    return out


TWOPI = 2*np.pi


def bare_lbdbar(sys):
    """lambda_bar = (lambda_1 + lambda_2)/2"""
    return 0.5*(sys.lbd1 + sys.lbd2)


def bare_dlbd(sys):
    """delta_lambda = (lambda_1 - lambda_2)/2"""
    return 0.5*(sys.lbd1 - sys.lbd2)


def track_eigs_HaBD_vs_lbdbar(sys, lbdbar_vals, module):
    """
    Continuity-tracked eigenvalues of Hbare_aBdD versus lambda_bar.
    """
    dlbd = bare_dlbd(sys)

    mats = []
    for lbdbar in lbdbar_vals:
        lbd1 = lbdbar + dlbd
        lbd2 = lbdbar - dlbd
        mats.append(
            module.Hbare_aBdD(
                sys.ga, sys.ka, sys.kd1, sys.kd2,
                lbd1, lbd2,
                sys.wa, sys.wd1, sys.wd2
            )
        )

    tracked = np.zeros((len(mats), 3), dtype=complex)

    vals0 = np.linalg.eigvals(mats[0])
    tracked[0] = vals0[np.argsort(vals0.real)]

    perms = list(permutations(range(3)))

    for k in range(1, len(mats)):
        vals = np.linalg.eigvals(mats[k])

        best_perm = None
        best_cost = np.inf
        for p in perms:
            cost = sum(abs(tracked[k-1, i] - vals[p[i]]) for i in range(3))
            if cost < best_cost:
                best_cost = cost
                best_perm = p

        tracked[k] = vals[list(best_perm)]

    return tracked


def scan_HaBD_eigs_vs_lbdbar(
    sys,
    module,
    npts=401,
    scan_frac=0.8,
    lbdbar_min_THz=None,
    lbdbar_max_THz=None,
):
    """
    Scan Re(Omega_i) and Im(Omega_i) of H_aBD versus lambda_bar.

    Returns
    -------
    x_THz : array
        lambda_bar / 2pi in THz
    re_THz : array, shape (npts, 3)
        Re(Omega_i) / 2pi in THz
    kappa_MHz : array, shape (npts, 3)
        kappa_i / 2pi = -Im(Omega_i)/2pi in MHz
    eigs : array, shape (npts, 3)
        complex eigenvalues
    """
    lbdbar0 = bare_lbdbar(sys)

    if lbdbar_min_THz is None or lbdbar_max_THz is None:
        lo = (1.0 - scan_frac)*lbdbar0
        hi = (1.0 + scan_frac)*lbdbar0

        # avoid accidental sign crossing if lambda_bar0 > 0
        if np.real(lbdbar0) > 0 and lo <= 0:
            lo = 1e-6*lbdbar0
    else:
        lo = TWOPI*sys.w_scale*1e12*lbdbar_min_THz
        hi = TWOPI*sys.w_scale*1e12*lbdbar_max_THz

    lbdbar_vals = np.linspace(lo, hi, npts)
    eigs = track_eigs_HaBD_vs_lbdbar(sys, lbdbar_vals, module)

    x_THz = lbdbar_vals / (TWOPI*sys.w_scale*1e12)
    re_THz = eigs.real / (TWOPI*sys.w_scale*1e12)
    kappa_MHz = (-eigs.imag) / (TWOPI*sys.w_scale*1e6)

    return x_THz, re_THz, kappa_MHz, eigs


def plot_HaBD_eigs_vs_lbdbar_panels(
    devices,
    module,
    npts=401,
    scan_frac=0.8,
    lbdbar_overrides=None,
    yfloor_re_THz=1e-12,
    yfloor_kappa_MHz=1e-6,
    figsize=None,
):
    """
    Plot Re(Omega_i) and kappa_i = -Im(Omega_i) versus lambda_bar
    for several devices in separate panels.

    Parameters
    ----------
    devices : dict
        Example: {"exp": sys_exp, "1": sys1, "2": sys2}
    lbdbar_overrides : dict or None
        Optional per-device scan ranges in THz:
        {"2": (0.0, 0.35), "exp": (1.0, 10.0)}
    """
    ndev = len(devices)

    if figsize is None:
        figsize = (11.0, 3.4*ndev)

    fig, axs = plt.subplots(ndev, 2, figsize=figsize, squeeze=False)

    colors = ["C0", "C1", "C2"]
    labels = [r"$\Omega_1$", r"$\Omega_2$", r"$\Omega_3$"]

    for r, (name, sys) in enumerate(devices.items()):
        if lbdbar_overrides is not None and name in lbdbar_overrides:
            lmin, lmax = lbdbar_overrides[name]
        else:
            lmin, lmax = None, None

        x_THz, re_THz, kappa_MHz, eigs = scan_HaBD_eigs_vs_lbdbar(
            sys,
            module,
            npts=npts,
            scan_frac=scan_frac,
            lbdbar_min_THz=lmin,
            lbdbar_max_THz=lmax,
        )

        axL = axs[r, 0]
        axR = axs[r, 1]

        for i in range(3):
            axL.plot(
                x_THz,
                np.clip(re_THz[:, i], yfloor_re_THz, None),
                color=colors[i],
                lw=2,
                label=labels[i],
            )
            axR.plot(
                x_THz,
                np.clip(kappa_MHz[:, i], yfloor_kappa_MHz, None),
                color=colors[i],
                lw=2,
                label=labels[i],
            )

        axL.set_yscale("log")
        axR.set_yscale("log")

        axL.set_title(fr"Device {name}: $\mathrm{{Re}}(\Omega_i)$")
        axR.set_title(fr"Device {name}: $\kappa_i=-\mathrm{{Im}}(\Omega_i)$")

        axL.set_xlabel(r"$\bar{\lambda}/2\pi$ (THz)")
        axR.set_xlabel(r"$\bar{\lambda}/2\pi$ (THz)")

        axL.set_ylabel(r"$\mathrm{Re}(\Omega_i)/2\pi$ (THz)")
        axR.set_ylabel(r"$\kappa_i/2\pi$ (MHz)")

        axL.grid(alpha=0.3, which="both")
        axR.grid(alpha=0.3, which="both")

        if r == 0:
            axL.legend(frameon=False, loc="best")
            axR.legend(frameon=False, loc="best")

    fig.tight_layout()
    return fig, axs


# =============================================================================
# Bare optical eigenvalue scans for plotting
# =============================================================================


def track_eigs_continuously(vals_list):
    """
    Continuity-based tracking of 3 eigenvalue branches.
    """
    tracked = np.zeros((len(vals_list), 3), dtype=complex)
    tracked[0] = vals_list[0][np.argsort(vals_list[0].real)]

    perms = list(permutations(range(3)))

    for k in range(1, len(vals_list)):
        vals = vals_list[k]

        best_cost = np.inf
        best_perm = None

        for p in perms:
            cost = sum(abs(tracked[k - 1, i] - vals[p[i]]) for i in range(3))
            if cost < best_cost:
                best_cost = cost
                best_perm = p

        tracked[k] = vals[list(best_perm)]

    return tracked


def scan_device_vs_lbdbar(sys, lbdbar_min_THz, lbdbar_max_THz, npts=501):
    """
    Scan H_aBD eigenvalues versus lambda_bar.

    Inputs:
        sys:
            System object containing ga, ka, kd1, kd2, lbd1, lbd2,
            wa, wd1, wd2, and w_scale.

        lbdbar_min_THz, lbdbar_max_THz:
            Scan range for lambda_bar/2pi in THz.

    Returns:
        x_THz:
            lambda_bar/2pi in THz.

        Delta_THz:
            Re(Omega_i)/2pi in THz.

        kappa_THz:
            k_i/2pi = -Im(Omega_i)/2pi in THz.

        eigs:
            tracked complex eigenvalues.
    """
    TWOPI = 2*np.pi
    unit_THz = TWOPI * sys.w_scale * 1e12

    dlbd = 0.5*(sys.lbd1 - sys.lbd2)

    lbdbar_vals = np.linspace(
        lbdbar_min_THz * unit_THz,
        lbdbar_max_THz * unit_THz,
        npts,
    )

    eigvals_raw = []

    for lbdbar in lbdbar_vals:
        lbd1 = lbdbar + dlbd
        lbd2 = lbdbar - dlbd

        H = Hbare_aBdD(
            sys.ga, sys.ka, sys.kd1, sys.kd2,
            lbd1, lbd2,
            sys.wa, sys.wd1, sys.wd2,
        )

        eigvals_raw.append(np.linalg.eigvals(H))

    eigs = track_eigs_continuously(eigvals_raw)

    x_THz = lbdbar_vals / unit_THz
    Delta_THz = eigs.real / unit_THz
    kappa_THz = (-eigs.imag) / unit_THz

    return x_THz, Delta_THz, kappa_THz, eigs


def used_parameter_row(sys, name):
    """
    Table-1-style summary of only the quantities used for the
    lambda_bar scan plot.

    All optical-frequency-like quantities are shown as ordinary
    frequencies in THz, i.e. quantity / 2pi.
    """
    TWOPI = 2*np.pi
    unit_THz = TWOPI * sys.w_scale * 1e12

    lbdbar = 0.5*(sys.lbd1 + sys.lbd2) / unit_THz
    dlbd = 0.5*(sys.lbd1 - sys.lbd2) / unit_THz

    kdbar = 0.5*(sys.kd1 + sys.kd2) / unit_THz
    dkd = 0.5*(sys.kd1 - sys.kd2) / unit_THz

    Ddbar = 0.5*(sys.wd1 + sys.wd2) / unit_THz
    dDd = 0.5*(sys.wd1 - sys.wd2) / unit_THz

    return {
        "Device": name,
        r"$\gamma_a/2\pi$ (THz)": sys.ga / unit_THz,
        r"$\kappa_a/2\pi$ (THz)": sys.ka / unit_THz,
        r"$\bar{\kappa}_d/2\pi$ (THz)": kdbar,
        r"$\delta_{\kappa_d}/2\pi$ (THz)": dkd,
        r"$\bar{\lambda}/2\pi$ (THz)": lbdbar,
        r"$\delta_{\lambda}/2\pi$ (THz)": dlbd,
        r"$\bar{\Delta}_d/2\pi$ (THz)": Ddbar,
        r"$\delta_{\Delta_d}/2\pi$ (THz)": dDd,
    }







# =============================================================================
# Refactored notebook support layer
# =============================================================================
# This support layer keeps figures.ipynb short.  It treats wa, wd1, wd2 as the
# real diagonal entries of the rotating-frame optical matrix, i.e. Delta_c at
# the chosen reference laser frequency whenever omega_las is not the scan
# variable.  When omega_las is scanned in r_CM/t_CM/g_effs, the same quantities
# are used consistently by the underlying functions through Delta_c = w_c - wL.

import inspect as _inspect
import pandas as _pd
import matplotlib.pyplot as _plt
from matplotlib.lines import Line2D as _Line2D

# ------------------------- compatibility aliases ----------------------------
# Some internal functions call *_2mode names.  The current file defines the
# clean names.  The aliases make old and new code paths both work.
try:
    t_CM_2mode
except NameError:
    t_CM_2mode = t_CM
try:
    r_CM_2mode
except NameError:
    r_CM_2mode = r_CM
try:
    Xopt_inv_2mode
except NameError:
    try:
        Xopt_inv_2mode = Xopt_inv
    except NameError:
        pass
try:
    a_ss_2mode
except NameError:
    try:
        a_ss_2mode = a_ss
    except NameError:
        pass
try:
    d1_ss_2mode
except NameError:
    try:
        d1_ss_2mode = d1_ss
    except NameError:
        pass
try:
    d2_ss_2mode
except NameError:
    try:
        d2_ss_2mode = d2_ss
    except NameError:
        pass

# ------------------------------- switches -----------------------------------
USE_TWO_FANO_MODES_DEFAULT = True
D2_EPS_DEFAULT = 1e-30

# ---------------------------- unit conversion --------------------------------
def internal_to_THz(x, sys):
    return np.asarray(x) / (2*np.pi*sys.w_scale*1e12)

def internal_to_GHz(x, sys):
    return np.asarray(x) / (2*np.pi*sys.w_scale*1e9)

def internal_to_MHz(x, sys):
    return np.asarray(x) / (2*np.pi*sys.w_scale*1e6)

def internal_to_Hz(x, sys):
    return np.asarray(x) / (2*np.pi*sys.w_scale)

def THz_to_raw_angular(x_THz):
    return 2*np.pi*np.asarray(x_THz)*1e12

def THz_to_internal(x_THz, sys):
    return 2*np.pi*np.asarray(x_THz)*sys.w_scale*1e12

def _fmt_value(val, digits=4):
    if isinstance(val, str):
        return val
    try:
        val = float(val)
    except Exception:
        return str(val)
    if val == 0:
        return "0"
    if abs(val) >= 1e4 or abs(val) < 1e-3:
        return f"{val:.{digits}e}"
    return f"{val:.{digits}f}"

# ----------------------- one-/two-Fano-mode switch ---------------------------
def apply_mode_switch(pars, use_two_fano_modes=True, d2_eps=D2_EPS_DEFAULT):
    """
    If use_two_fano_modes=False, suppress d2 while keeping the two-mode code
    numerically safe.  This is the single switch that lets the same notebook
    reproduce an effective one-Fano-mode limit.
    """
    pars = pars.copy()
    if not use_two_fano_modes:
        pars["kd2"] = d2_eps
        pars["lbd2"] = d2_eps
        pars["gkd20"] = 0.0
        pars["gwd20"] = 0.0
        ref = pars.get("wa", 0.0)
        pars["wd2"] = ref + 2*np.pi*1e18
    return pars

# -------------------------------- System -------------------------------------
class System:
    """
    Convenience container for parameters.  All angular-frequency-like inputs are
    rescaled by w_scale.  The real diagonal entries wa, wd1, wd2 are the optical
    diagonal quantities used by the Hamiltonian.  If omega_las is not scanned,
    they should be understood as fixed Delta_c values at the reference laser.
    """
    def __init__(self, w_scale=1e-9, **kwargs):
        self.__dict__.update(kwargs)
        self._init_kwargs = kwargs.copy()
        self.w_scale = w_scale

        # Wavelength-like diagnostics use the raw input values before rescaling.
        try:
            self.lam_a = 2*np.pi*c/(self.wa)*1e9
            self.lam_d1 = 2*np.pi*c/(self.wd1)*1e9
            self.lam_d2 = 2*np.pi*c/(self.wd2)*1e9
        except Exception:
            pass

        if "Qm" in kwargs and "gam" not in kwargs:
            self.gam = self.Om/self.Qm

        try:
            self.Nth = 1/(exp(h*self.Om/kB/self.Tmec) - 1)
        except Exception:
            self.Nth = np.nan

        for freq in [
            "wa", "wd1", "wd2", "ka", "kd1", "kd2", "ga", "Om", "gam",
            "lbd1", "lbd2", "gka0", "gkd10", "gkd20", "gwa0", "gwd10", "gwd20",
        ]:
            if freq in self.__dict__:
                self.__dict__[freq] *= w_scale

        if "Plas" in kwargs:
            self.Plas *= w_scale**2
        if "thlas" not in kwargs:
            self.thlas = 0.0
        if "gam" not in self.__dict__ and "Qm" in self.__dict__:
            self.gam = self.Om/self.Qm

        self.refresh_derived()

    def refresh_derived(self):
        self.G1 = optical_G1(self.ka, self.kd1, self.lbd1)
        self.G2 = optical_G2(self.ka, self.kd2, self.lbd2)
        self.K12 = optical_K12(self.kd1, self.kd2)
        self.kbar = 0.5*(self.kd1 + self.kd2)
        self.dk = 0.5*(self.kd1 - self.kd2)
        self.lbdbar = 0.5*(self.lbd1 + self.lbd2)
        self.dlbd = 0.5*(self.lbd1 - self.lbd2)
        self.Ddbar = 0.5*(self.wd1 + self.wd2)
        self.dDd = 0.5*(self.wd1 - self.wd2)
        self.tw1, self.tw2, self.tw3 = [complex(z) for z in tw_all(
            self.ga, self.ka, self.kd1, self.kd2,
            self.lbd1, self.lbd2, self.wa, self.wd1, self.wd2
        )]
        self.w1, self.w2, self.w3 = self.tw1.real, self.tw2.real, self.tw3.real
        self.k1, self.k2, self.k3 = -self.tw1.imag, -self.tw2.imag, -self.tw3.imag
        tws = np.array([self.tw1, self.tw2, self.tw3], dtype=complex)
        ks = -np.imag(tws)
        self.lowest_mode_index = int(np.argmin(ks)) + 1
        self.tw_min = tws[self.lowest_mode_index-1]
        self.w_min = self.tw_min.real
        self.k_min = -self.tw_min.imag
        # Old one-mode-style aliases: use the lowest-loss branch.
        self.wm = self.w_min
        self.km = self.k_min

    def __repr__(self):
        keys = ["wa", "ka", "ga", "wd1", "wd2", "kd1", "kd2", "lbd1", "lbd2"]
        small = {k: getattr(self, k, None) for k in keys if hasattr(self, k)}
        return f"System({small})"

    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getattr__(self, attr):
        g = globals()
        if attr not in g or not callable(g[attr]):
            raise AttributeError(attr)
        func = g[attr]
        args = _inspect.getfullargspec(func).args
        def func2(**kw):
            for arg in args:
                if arg in self.__dict__:
                    kw.setdefault(arg, self.__dict__[arg])
            return func(**kw)
        return func2

    def get(self, names):
        return tuple(self.__dict__[n.strip()] for n in names.split(','))

    def get_pars(self):
        return self._init_kwargs.copy()

    def clone_with_updates(self, **updates_raw):
        pars = self.get_pars()
        pars.update(updates_raw)
        return System(w_scale=self.w_scale, **pars)

# ---------------------------- basic summaries --------------------------------
def device_modes(sys):
    return np.array([sys.tw1, sys.tw2, sys.tw3], dtype=complex)

def choose_mode(sys, mode_index=None, active_modes=3):
    tws = device_modes(sys)[:active_modes]
    ks = -np.imag(tws)
    if mode_index is None:
        idx = int(np.argmin(ks))
    else:
        idx = int(mode_index) - 1
    if idx < 0 or idx >= active_modes:
        raise ValueError("mode_index must be in 1..active_modes")
    return idx + 1, tws[idx], tws[idx].real, -tws[idx].imag

def make_eigenvalue_table(devices, active_modes=3):
    rows = []
    for name, sys in devices.items():
        for i, Om_i in enumerate(device_modes(sys)[:active_modes], start=1):
            rows.append({
                "Device": str(name),
                "Mode": rf"$\Omega_{{{i}}}$",
                r"$\mathrm{Re}(\Omega_i)/2\pi$ (THz)": float(internal_to_THz(Om_i.real, sys)),
                r"$\kappa_i/2\pi$ (THz)": float(internal_to_THz(-Om_i.imag, sys)),
                r"$\kappa_i/2\pi$ (MHz)": float(internal_to_MHz(-Om_i.imag, sys)),
            })
    return _pd.DataFrame(rows)

def bare_parameter_table(devices):
    rows = []
    for name, sys in devices.items():
        rows.append({
            "Device": str(name),
            r"$\gamma_a/2\pi$ (THz)": float(internal_to_THz(sys.ga, sys)),
            r"$\kappa_a/2\pi$ (THz)": float(internal_to_THz(sys.ka, sys)),
            r"$\kappa_{d_1}/2\pi$ (THz)": float(internal_to_THz(sys.kd1, sys)),
            r"$\kappa_{d_2}/2\pi$ (THz)": float(internal_to_THz(sys.kd2, sys)),
            r"$\Delta_a/2\pi$ (THz)": float(internal_to_THz(sys.wa, sys)),
            r"$\Delta_{d_1}/2\pi$ (THz)": float(internal_to_THz(sys.wd1, sys)),
            r"$\Delta_{d_2}/2\pi$ (THz)": float(internal_to_THz(sys.wd2, sys)),
            r"$\lambda_1/2\pi$ (THz)": float(internal_to_THz(sys.lbd1, sys)),
            r"$\lambda_2/2\pi$ (THz)": float(internal_to_THz(sys.lbd2, sys)),
            r"$\Omega_{\rm mec}/2\pi$ (Hz)": float(internal_to_Hz(sys.Om, sys)),
        })
    df = _pd.DataFrame(rows)
    quantities = [c for c in df.columns if c != "Device"]
    out = _pd.DataFrame({"Quantity": quantities})
    for name in df["Device"]:
        row = df[df["Device"] == name].iloc[0]
        out[name] = [row[q] for q in quantities]
    return out

# ------------------------- generic optimization ------------------------------
def _parameter_scan_values(sys, variable, x_THz):
    unit = 2*np.pi*sys.w_scale*1e12
    x = x_THz*unit
    pars = dict(ga=sys.ga, ka=sys.ka, kd1=sys.kd1, kd2=sys.kd2,
                lbd1=sys.lbd1, lbd2=sys.lbd2, wa=sys.wa, wd1=sys.wd1, wd2=sys.wd2)
    if variable == "lbdbar":
        dlbd = 0.5*(sys.lbd1 - sys.lbd2)
        pars["lbd1"] = x + dlbd
        pars["lbd2"] = x - dlbd
    elif variable in ["Ddbar", "Dbar", "Omegadbar"]:
        dDd = 0.5*(sys.wd1 - sys.wd2)
        pars["wd1"] = x + dDd
        pars["wd2"] = x - dDd
    elif variable in ["delta_Dd", "dDd", "deltaOmegad"]:
        Dbar = 0.5*(sys.wd1 + sys.wd2)
        pars["wd1"] = Dbar + x
        pars["wd2"] = Dbar - x
    elif variable in ["kdbar", "kbar"]:
        dkd = 0.5*(sys.kd1 - sys.kd2)
        pars["kd1"] = x + dkd
        pars["kd2"] = x - dkd
    elif variable in ["delta_kd", "dkd"]:
        kbar = 0.5*(sys.kd1 + sys.kd2)
        pars["kd1"] = kbar + x
        pars["kd2"] = kbar - x
    else:
        raise ValueError("Unsupported optimization variable: " + str(variable))
    return pars

def scan_device_vs_variable(sys, variable, xmin_THz, xmax_THz, npts=801, basis="aBdD"):
    x_THz = np.linspace(xmin_THz, xmax_THz, npts)
    raw = []
    for x in x_THz:
        p = _parameter_scan_values(sys, variable, x)
        H = Hbare_aBdD(**p) if basis == "aBdD" else Hbare_ad1d2(**p)
        raw.append(np.linalg.eigvals(H))
    eigs = track_eigs_continuously(raw)
    unit = 2*np.pi*sys.w_scale*1e12
    return x_THz, eigs.real/unit, (-eigs.imag)/unit, eigs

def _default_scan_range(sys, variable, scan_frac=1.2, min_half_width_THz=0.5):
    unit = 2*np.pi*sys.w_scale*1e12
    if variable == "lbdbar":
        x0 = 0.5*(sys.lbd1 + sys.lbd2)/unit
        hw = max(abs(x0)*scan_frac, min_half_width_THz)
    elif variable in ["Ddbar", "Dbar", "Omegadbar"]:
        x0 = 0.5*(sys.wd1 + sys.wd2)/unit
        hw = max(0.04*abs(x0), min_half_width_THz)
    elif variable in ["delta_Dd", "dDd", "deltaOmegad"]:
        x0 = 0.5*(sys.wd1 - sys.wd2)/unit
        hw = max(abs(x0)*scan_frac, min_half_width_THz)
    elif variable in ["kdbar", "kbar"]:
        x0 = 0.5*(sys.kd1 + sys.kd2)/unit
        hw = max(abs(x0)*scan_frac, min_half_width_THz)
    elif variable in ["delta_kd", "dkd"]:
        x0 = 0.5*(sys.kd1 - sys.kd2)/unit
        kbar = 0.5*(sys.kd1 + sys.kd2)/unit
        hw = max(min(abs(kbar)*0.95, abs(x0)*scan_frac + min_half_width_THz), min_half_width_THz)
    else:
        raise ValueError(variable)
    return x0-hw, x0+hw

def update_system_from_optimum(sys, variable, xopt_THz):
    p_scaled = _parameter_scan_values(sys, variable, xopt_THz)
    # Convert scaled internal values back to raw angular for System constructor.
    pars = sys.get_pars().copy()
    for key in ["kd1", "kd2", "lbd1", "lbd2", "wa", "wd1", "wd2", "ka", "ga"]:
        if key in p_scaled:
            pars[key] = p_scaled[key] / sys.w_scale
    return System(w_scale=sys.w_scale, **pars)

def run_bare_optimization(devices, variable="lbdbar", npts=801, scan_ranges_THz=None,
                          scan_frac=1.2, active_modes=3, basis="aBdD"):
    scan_ranges_THz = scan_ranges_THz or {}
    scan_data = {}
    minima_rows = []
    updated_devices = {}
    update_rows = []
    for name, sys in devices.items():
        if scan_ranges_THz.get(name) is None:
            xmin, xmax = _default_scan_range(sys, variable, scan_frac=scan_frac)
        else:
            xmin, xmax = scan_ranges_THz[name]
        x, Delta, kappa, eigs = scan_device_vs_variable(sys, variable, xmin, xmax, npts=npts, basis=basis)
        scan_data[name] = dict(x=x, Delta_THz=Delta, kappa_THz=kappa, eigs=eigs, range=(xmin, xmax))
        # minima per active mode and global minimum
        sub_mins = []
        for im in range(active_modes):
            j = int(np.argmin(kappa[:, im]))
            row = {
                "Device": str(name), "Mode": rf"$\Omega_{{{im+1}}}$",
                "x_min_THz": float(x[j]),
                "kappa_min_THz": float(kappa[j, im]),
                "kappa_min_MHz": float(kappa[j, im]*1e6),
                "ReOmega_at_min_THz": float(Delta[j, im]),
            }
            sub_mins.append(row)
            minima_rows.append(row)
        best = min(sub_mins, key=lambda r: r["kappa_min_THz"])
        updated = update_system_from_optimum(sys, variable, best["x_min_THz"])
        updated_devices[name] = updated
        update_rows.append({
            "Device": str(name), "chosen mode": best["Mode"],
            "optimized variable": variable, "x_opt_THz": best["x_min_THz"],
            "kappa_min_MHz": best["kappa_min_MHz"],
            r"$\lambda_1/2\pi$ (THz)": float(internal_to_THz(updated.lbd1, updated)),
            r"$\lambda_2/2\pi$ (THz)": float(internal_to_THz(updated.lbd2, updated)),
            r"$\Delta_{d_1}/2\pi$ (THz)": float(internal_to_THz(updated.wd1, updated)),
            r"$\Delta_{d_2}/2\pi$ (THz)": float(internal_to_THz(updated.wd2, updated)),
        })
    return {
        "variable": variable,
        "scan_data": scan_data,
        "minima_table": _pd.DataFrame(minima_rows),
        "update_table": _pd.DataFrame(update_rows),
        "initial_devices": devices,
        "updated_devices": updated_devices,
        "updated_parameter_table": bare_parameter_table(updated_devices),
    }

def run_lbdbar_optimization(devices, **kwargs):
    return run_bare_optimization(devices, variable="lbdbar", **kwargs)

# ---------------------------- scan builders ----------------------------------
def build_wL_scan(sys, x, mode_index=None, xscale="kappa", active_modes=3):
    im, tw, wi, ki = choose_mode(sys, mode_index=mode_index, active_modes=active_modes)
    if xscale in ["kappa", "k"]:
        scale = ki
        xlabel = r"$\Delta_i/\kappa_i$"
    elif xscale in ["Omega", "Om", "Omega_mec"]:
        scale = sys.Om
        xlabel = r"$\Delta_i/\Omega_{\mathrm{mec}}$"
    else:
        raise ValueError("xscale must be 'kappa' or 'Omega'")
    wL = wi - x*scale
    return wL, im, tw, wi, ki, xlabel

def _auto_wL_range_from_modes(sys, reference_sys=None, margin_THz=1.0):
    vals = [z.real for z in device_modes(sys)]
    if reference_sys is not None:
        vals += [z.real for z in device_modes(reference_sys)]
    vals_THz = internal_to_THz(np.array(vals), sys)
    return float(np.nanmin(vals_THz)-margin_THz), float(np.nanmax(vals_THz)+margin_THz)

# ------------------------------ plot helpers ---------------------------------
def _new_figure_grid(figsize=(13.8, 10.4), wspace=0.25, hspace=0.26):
    fig = _plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, wspace=wspace, hspace=hspace)
    return fig, gs, {"exp": gs[0,0], "1": gs[0,1], "2": gs[1,0], "info": gs[1,1]}

def _panel_title(ax, label, title, mode_index=None):
    mode = "" if mode_index is None else rf"  ($\Omega_{{{mode_index}}}$)"
    ax.text(0.98, 1.02, f"{label}  {title}{mode}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10, clip_on=False)

def _make_info_table(ax, param_table, bbox=(0.02, 0.02, 0.96, 0.72), fontsize=7.2):
    cols = list(param_table.columns)
    text = []
    for _, row in param_table.iterrows():
        text.append([row[c] if c == "Quantity" else _fmt_value(row[c]) for c in cols])
    tbl = ax.table(cellText=text, colLabels=cols, cellLoc="center", colLoc="center",
                   colWidths=[0.42] + [0.17]*(len(cols)-1), bbox=bbox)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    for (r,c), cell in tbl.get_celld().items():
        cell.set_edgecolor("0.20"); cell.set_linewidth(0.75); cell.PAD=0.026
        if r == 0:
            cell.set_facecolor("#ececec"); cell.set_text_props(weight="bold", fontsize=fontsize+0.8)
        else:
            cell.set_facecolor("white" if r % 2 else "#fafafa")
    return tbl

# ----------------------------- plotting functions ----------------------------
def plot_optimization_result(result, x_label=None, title=None, active_modes=3):
    fig, gs, specs = _new_figure_grid()
    colors = ["C0", "C1", "C2"]
    labels = {"exp": "(a)", "1": "(b)", "2": "(c)"}
    titles = {"exp": "Device exp", "1": "Device 1", "2": "Device 2"}
    for name in ["exp", "1", "2"]:
        if name not in result["scan_data"]:
            continue
        subgs = specs[name].subgridspec(2,1,hspace=0.45)
        ax1 = fig.add_subplot(subgs[0,0]); ax2 = fig.add_subplot(subgs[1,0])
        d = result["scan_data"][name]
        x = d["x"]
        for i in range(active_modes):
            ax1.plot(x, d["Delta_THz"][:,i], color=colors[i], lw=2)
            ax2.plot(x, np.clip(d["kappa_THz"][:,i], 1e-18, None), color=colors[i], lw=2)
        ax2.set_yscale("log")
        ax1.set_ylabel(r"$\mathrm{Re}(\Omega_i)/2\pi$ (THz)")
        ax2.set_ylabel(r"$\kappa_i/2\pi$ (THz)")
        xl = x_label or result["variable"] + r"$/2\pi$ (THz)"
        ax1.set_xlabel(xl); ax2.set_xlabel(xl)
        for ax in [ax1, ax2]:
            ax.grid(False); ax.margins(x=0); ax.set_xlim(x[0], x[-1])
            for sp in ax.spines.values(): sp.set_linewidth(1.2)
        _panel_title(ax1, labels[name], titles[name])
    ax = fig.add_subplot(specs["info"]); ax.axis("off")
    handles = [_Line2D([0],[0], color=colors[i], lw=2.5, label=rf"$\Omega_{{{i+1}}}$") for i in range(active_modes)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5,0.97), ncol=active_modes, frameon=True)
    ax.text(0.03,0.97,"(d)", transform=ax.transAxes, va="top")
    _make_info_table(ax, result["updated_parameter_table"], bbox=(0.02,0.02,0.96,0.72), fontsize=7.0)
    _plt.show()
    return {"fig": fig, "parameter_table": result["updated_parameter_table"], "minima_table": result["minima_table"]}

def plot_reflection_devices(devices, reference_devices=None, scan_ranges_THz=None, npts=2001, margin_THz=1.0):
    scan_ranges_THz = scan_ranges_THz or {}
    fig, gs, specs = _new_figure_grid()
    rows = []
    labels = {"exp":"(a)", "1":"(b)", "2":"(c)"}; titles={"exp":"Device exp","1":"Device 1","2":"Device 2"}
    for name, sys in devices.items():
        ax = fig.add_subplot(specs[name])
        ref = reference_devices.get(name) if reference_devices else None
        if scan_ranges_THz.get(name) is None:
            lo, hi = _auto_wL_range_from_modes(sys, ref, margin_THz=margin_THz)
        else:
            lo, hi = scan_ranges_THz[name]
        wL_THz = np.linspace(lo, hi, npts)
        wL = THz_to_internal(wL_THz, sys)
        r = r_CM(sys.ga, sys.ka, sys.kd1, sys.kd2, sys.lbd1, sys.lbd2, wL, sys.wa, sys.wd1, sys.wd2)
        ax.plot(wL_THz, np.abs(r)**2, lw=2.2)
        ax.set_xlim(lo, hi); ax.margins(x=0); ax.grid(False)
        ax.set_xlabel(r"$\omega_{\rm las}/2\pi$ (THz)"); ax.set_ylabel(r"$R=|r|^2$")
        _panel_title(ax, labels[name], titles[name])
        rows.append({"Device": name, "scan range": f"[{lo:.3g}, {hi:.3g}]"})
    ax = fig.add_subplot(specs["info"]); ax.axis("off"); ax.text(0.03,0.97,"(d)", transform=ax.transAxes, va="top")
    table = bare_parameter_table(devices)
    _make_info_table(ax, table, bbox=(0.02,0.02,0.96,0.84), fontsize=7.1)
    _plt.show()
    return {"fig": fig, "parameter_table": table}

def plot_RT_devices(devices, reference_devices=None, scan_ranges_THz=None, npts=2001, margin_THz=1.0):
    scan_ranges_THz = scan_ranges_THz or {}
    fig, gs, specs = _new_figure_grid()
    labels={"exp":"(a)","1":"(b)","2":"(c)"}; titles={"exp":"Device exp","1":"Device 1","2":"Device 2"}
    for name, sys in devices.items():
        ax = fig.add_subplot(specs[name])
        ref = reference_devices.get(name) if reference_devices else None
        lo, hi = scan_ranges_THz.get(name, _auto_wL_range_from_modes(sys, ref, margin_THz=margin_THz))
        wL_THz = np.linspace(lo, hi, npts); wL = THz_to_internal(wL_THz, sys)
        R = np.abs(r_CM(sys.ga, sys.ka, sys.kd1, sys.kd2, sys.lbd1, sys.lbd2, wL, sys.wa, sys.wd1, sys.wd2))**2
        T = np.abs(t_CM(sys.ga, sys.ka, sys.kd1, sys.kd2, sys.lbd1, sys.lbd2, wL, sys.wa, sys.wd1, sys.wd2))**2
        ax.plot(wL_THz, R, lw=2, label=r"$R$")
        ax.plot(wL_THz, T, lw=2, label=r"$T$")
        ax.plot(wL_THz, R+T, lw=2, label=r"$R+T$")
        ax.set_ylim(-0.02, 1.05*max(1, np.nanmax(R+T))); ax.set_xlim(lo, hi); ax.margins(x=0); ax.grid(False)
        ax.set_xlabel(r"$\omega_{\rm las}/2\pi$ (THz)"); ax.set_ylabel(r"$R,T,R+T$")
        _panel_title(ax, labels[name], titles[name])
    ax = fig.add_subplot(specs["info"]); ax.axis("off"); ax.text(0.03,0.97,"(d)", transform=ax.transAxes, va="top")
    handles = [_Line2D([0],[0], lw=2, color=f"C{i}", label=lab) for i, lab in enumerate([r"$R$", r"$T$", r"$R+T$"])]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5,0.97), ncol=3, frameon=True)
    table = bare_parameter_table(devices)
    _make_info_table(ax, table, bbox=(0.02,0.02,0.96,0.78), fontsize=7.0)
    _plt.show()
    return {"fig": fig, "parameter_table": table}

# ------------------------- standard devices ----------------------------------
def make_standard_devices_from_lowest_loss(devices, gwa0_std_by_device=None, Plas_factor=1.0, d_eps=D2_EPS_DEFAULT):
    gwa0_std_by_device = gwa0_std_by_device or {}
    std_devices = {}; rows=[]; std_rows=[]
    for name, sys in devices.items():
        im, tw, wi, ki = choose_mode(sys)
        gwa0_raw = gwa0_std_by_device.get(name, sys.gwa0/sys.w_scale)
        pars = dict(
            kd1=d_eps, wd1=wi/sys.w_scale + 2*np.pi*1e18,
            kd2=d_eps, wd2=wi/sys.w_scale + 4*np.pi*1e18,
            Om=sys.Om/sys.w_scale, Plas=sys.Plas/(sys.w_scale**2)*Plas_factor,
            thlas=sys.thlas, Qm=getattr(sys, "Qm", sys.Om/sys.gam), Tmec=sys.Tmec,
            wa=wi/sys.w_scale, ka=(ki/2)/sys.w_scale, ga=(ki/2)/sys.w_scale,
            lbd1=d_eps, lbd2=d_eps,
            gka0=0.0, gkd10=0.0, gkd20=0.0,
            gwa0=gwa0_raw, gwd10=0.0, gwd20=0.0,
        )
        s = System(w_scale=sys.w_scale, **pars)
        std_devices[name] = s
        rows.append({"Device":name, "mode":rf"$\Omega_{{{im}}}$", "ReOmega_THz":float(internal_to_THz(wi, sys)), "kappa_THz":float(internal_to_THz(ki, sys)), "kappa_MHz":float(internal_to_MHz(ki, sys))})
        std_rows.append({"Device":name, "wa_THz":float(internal_to_THz(s.wa, s)), "ka_THz":float(internal_to_THz(s.ka, s)), "ga_THz":float(internal_to_THz(s.ga, s)), "gwa0_Hz":float(internal_to_Hz(s.gwa0, s))})
    return {"std_devices": std_devices, "lowest_loss_table": _pd.DataFrame(rows), "std_parameter_table": _pd.DataFrame(std_rows)}

# ------------------------------ g_eff plots ----------------------------------
def _g_labels(active_modes=3):
    labs = [r"$\tilde g_{\mathrm{mec},a}$", r"$\tilde g_{\mathrm{mec},d_1}$", r"$\tilde g_{\mathrm{mec},d_2}$",
            r"$\tilde g_a$", r"$\tilde g_{d_1}$", r"$\tilde g_{d_2}$"]
    if active_modes == 2:
        return [labs[0], labs[1], labs[3], labs[4]]
    return labs

def _g_indices(active_modes=3):
    return [0,1,2,3,4,5] if active_modes == 3 else [0,1,3,4]

def plot_geff_devices(devices, std_devices=None, xscale="kappa", x_ranges=None, npts=1001, active_modes=3):
    x_ranges = x_ranges or {"exp":(-2,2), "1":(-2,2), "2":(-2,2)}
    fig, gs, specs = _new_figure_grid()
    colors=["C0","C2","C1","C3","C4","C5"]; styles=["-","-","-","--","--","--"]
    labels={"exp":"(a)","1":"(b)","2":"(c)"}; titles={"exp":"Device exp","1":"Device 1","2":"Device 2"}
    data={}; rows=[]; inds=_g_indices(active_modes); labs=_g_labels(active_modes)
    for name, sys in devices.items():
        subgs = specs[name].subgridspec(2,1,hspace=0.0)
        ax1=fig.add_subplot(subgs[0,0]); ax2=fig.add_subplot(subgs[1,0], sharex=ax1)
        x=np.linspace(*x_ranges[name], npts)
        wL, im, tw, wi, ki, xlabel = build_wL_scan(sys, x, xscale=xscale, active_modes=active_modes)
        g=np.array(sys.g_effs(wL=wL), dtype=complex)
        for j,idx in enumerate(inds):
            ax1.plot(x, np.abs(g[idx])/sys.Om, color=colors[idx], ls=styles[idx], lw=2)
            ax2.plot(x, np.angle(g[idx]), color=colors[idx], ls=styles[idx], lw=2)
        if std_devices and name in std_devices:
            s=std_devices[name]
            # use same dimensionless x but standard scan around its lowest mode
            wLs, *_ = build_wL_scan(s, x, xscale=xscale, active_modes=active_modes)
            try:
                gs0=np.array(s.g_effs(wL=wLs), dtype=complex)[0]
                ax1.plot(x, np.abs(gs0)/s.Om, color="k", lw=2, ls=(0,(1,3)))
                ax2.plot(x, np.angle(gs0), color="k", lw=2, ls=(0,(1,3)))
            except Exception:
                pass
        ax2.set_ylim(-np.pi, np.pi); ax2.set_yticks([-np.pi,0,np.pi]); ax2.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
        ax1.set_ylabel(r"$|\tilde g|/\Omega_{\rm mec}$"); ax2.set_ylabel(r"$\phi$ (rad)"); ax2.set_xlabel(xlabel)
        ax1.tick_params(axis="x", labelbottom=False)
        for ax in [ax1,ax2]:
            ax.grid(False); ax.margins(x=0); ax.set_xlim(x[0],x[-1])
            for sp in ax.spines.values(): sp.set_linewidth(1.2)
        _panel_title(ax1, labels[name], titles[name], im)
        rows.append({"Device":name, "mode used":rf"$\Omega_{{{im}}}$", "x-axis":xlabel, "range":str(x_ranges[name]), r"$\kappa_i/2\pi$ (MHz)":float(internal_to_MHz(ki, sys))})
        data[name]=dict(x=x,wL=wL,g=g)
    ax=fig.add_subplot(specs["info"]); ax.axis("off"); ax.text(0.03,0.97,"(d)", transform=ax.transAxes, va="top")
    handles=[_Line2D([0],[0], color=colors[idx], ls=styles[idx], lw=2, label=lab) for idx, lab in zip(inds,labs)]
    if std_devices: handles.append(_Line2D([0],[0], color="k", ls=(0,(1,3)), lw=2, label=r"$\tilde g^{\rm std}$"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5,0.97), ncol=2, frameon=True, fontsize=8.5)
    ptab=bare_parameter_table(devices)
    _make_info_table(ax, ptab, bbox=(0.02,0.02,0.96,0.72), fontsize=7.0)
    _plt.show()
    return {"fig":fig,"raw_data":data,"parameter_table":ptab,"mode_table":_pd.DataFrame(rows)}

def fit_circle_complex(z):
    z=np.asarray(z, dtype=complex); x=z.real; y=z.imag
    m=np.isfinite(x)&np.isfinite(y); x=x[m]; y=y[m]
    M=np.column_stack([x,y,np.ones_like(x)]); rhs=-(x*x+y*y)
    A,B,C=np.linalg.lstsq(M,rhs,rcond=None)[0]
    xc=-A/2; yc=-B/2; R2=xc*xc+yc*yc-C
    R=np.sqrt(R2) if R2>0 else np.nan
    zc=xc+1j*yc
    rel=np.sqrt(np.mean((np.abs(x+1j*y-zc)-R)**2))/R if np.isfinite(R) and R!=0 else np.nan
    return zc,R,rel

def plot_geff_complex_plane(devices, xscale="kappa", x_ranges=None, npts=3001, active_modes=3):
    x_ranges=x_ranges or {"exp":(-100,100),"1":(-20,20),"2":(-80,80)}
    fig, gs, specs = _new_figure_grid()
    colors=["C0","C2","C1","C3","C4","C5"]; styles=["-","-","-","--","--","--"]
    labels={"exp":"(a)","1":"(b)","2":"(c)"}; titles={"exp":"Device exp","1":"Device 1","2":"Device 2"}
    inds=_g_indices(active_modes); labs=_g_labels(active_modes); circle_rows=[]
    for name, sys in devices.items():
        ax=fig.add_subplot(specs[name]); x=np.linspace(*x_ranges[name], npts)
        wL, im, tw, wi, ki, xlabel=build_wL_scan(sys,x,xscale=xscale,active_modes=active_modes)
        g=np.array(sys.g_effs(wL=wL), dtype=complex)
        xs=[]; ys=[]
        for idx in inds:
            z=g[idx]/sys.Om; xs.append(z.real); ys.append(z.imag)
            ax.plot(z.real, z.imag, color=colors[idx], ls=styles[idx], lw=2)
            ax.plot(z.real[0], z.imag[0], "o", color=colors[idx], mfc="w", ms=4)
            zc,R,rel=fit_circle_complex(z)
            circle_rows.append({"Device":name,"curve":idx,"center":str(zc),"R":R,"rel_error":rel})
        allx=np.concatenate(xs); ally=np.concatenate(ys); m=np.isfinite(allx)&np.isfinite(ally)
        allx=allx[m]; ally=ally[m]
        if len(allx):
            xm=(allx.min()+allx.max())/2; ym=(ally.min()+ally.max())/2; span=max(allx.max()-allx.min(), ally.max()-ally.min())*0.58
            if span<=0: span=1e-12
            ax.set_xlim(xm-span,xm+span); ax.set_ylim(ym-span,ym+span)
        ax.set_aspect("equal", adjustable="box"); ax.grid(True, alpha=0.35)
        ax.set_xlabel(r"$\Re(\tilde g)/\Omega_{\rm mec}$"); ax.set_ylabel(r"$\Im(\tilde g)/\Omega_{\rm mec}$")
        _panel_title(ax, labels[name], titles[name], im)
    ax=fig.add_subplot(specs["info"]); ax.axis("off"); ax.text(0.03,0.97,"(d)", transform=ax.transAxes, va="top")
    handles=[_Line2D([0],[0], color=colors[idx], ls=styles[idx], lw=2, label=lab) for idx,lab in zip(inds,labs)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5,0.97), ncol=2, frameon=True, fontsize=8.5)
    ptab=bare_parameter_table(devices); _make_info_table(ax, ptab, bbox=(0.02,0.02,0.96,0.72), fontsize=7.0)
    _plt.show()
    return {"fig":fig,"circle_table":_pd.DataFrame(circle_rows),"parameter_table":ptab}

# ----------------------------- dT/dx plots -----------------------------------
def plot_dTdx_devices(devices, xscale="kappa", x_ranges=None, npts=800, include_instability=False, active_modes=3):
    x_ranges=x_ranges or {"exp":(-1.5,1.5),"1":(-1.5,1.5),"2":(-1.5,1.5)}
    fig, gs, specs = _new_figure_grid()
    colors=["C0","C2","C1","C3","C4","C5"]; styles=["-","-.",(0,(4,2,1,2,1,2)),"--",":",(0,(6,2))]
    clabs=[r"$x=\tilde\Delta_a$",r"$x=\tilde\Delta_{d_1}$",r"$x=\tilde\Delta_{d_2}$",r"$x=\tilde\kappa_a$",r"$x=\tilde\kappa_{d_1}$",r"$x=\tilde\kappa_{d_2}$"]
    labels={"exp":"(a)","1":"(b)","2":"(c)"}; titles={"exp":"Device exp","1":"Device 1","2":"Device 2"}
    rows=[]; raw={}
    for name, sys in devices.items():
        ax=fig.add_subplot(specs[name]); x=np.linspace(*x_ranges[name], npts)
        wL, im, tw, wi, ki, xlabel=build_wL_scan(sys,x,xscale=xscale,active_modes=active_modes)
        dts=sys.dTs(wL=wL); T=dts[-1]; Tmax=np.nanmax(T)
        for i,dT in enumerate(dts[:-1]): ax.plot(x, dT/Tmax, color=colors[i], ls=styles[i], lw=1.9)
        ax.set_xlim(x[0],x[-1]); ax.margins(x=0); ax.grid(False)
        ax.set_xlabel(xlabel); ax.set_ylabel(r"$\partial_xT/T_{\max}$")
        _panel_title(ax, labels[name], titles[name], im)
        rows.append({"Device":name,"mode used":rf"$\Omega_{{{im}}}$","x-axis":xlabel,"range":str(x_ranges[name]),r"$T_{\max}$":float(Tmax)})
        raw[name]=dict(x=x,wL=wL,dTs=dts,Tmax=Tmax)
    ax=fig.add_subplot(specs["info"]); ax.axis("off"); ax.text(0.03,0.97,"(d)", transform=ax.transAxes, va="top")
    handles=[_Line2D([0],[0], color=colors[i], ls=styles[i], lw=2, label=clabs[i]) for i in range(6)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5,0.97), ncol=2, frameon=True, fontsize=8.5)
    ptab=bare_parameter_table(devices); _make_info_table(ax, ptab, bbox=(0.02,0.02,0.96,0.72), fontsize=7.0)
    _plt.show()
    return {"fig":fig,"raw_data":raw,"parameter_table":ptab,"mode_table":_pd.DataFrame(rows)}

# ------------------------- optical spring/damping ----------------------------
def plot_optical_spring_damping_devices(devices, std_devices=None, xscale="kappa", x_ranges=None, npts=1001, active_modes=3, w_eval=None):
    x_ranges=x_ranges or {"exp":(-2,2),"1":(-2,2),"2":(-2,2)}
    fig, gs, specs = _new_figure_grid()
    labels={"exp":"(a)","1":"(b)","2":"(c)"}; titles={"exp":"Device exp","1":"Device 1","2":"Device 2"}
    raw={}; rows=[]
    for name, sys in devices.items():
        subgs=specs[name].subgridspec(2,1,hspace=0.0); ax1=fig.add_subplot(subgs[0,0]); ax2=fig.add_subplot(subgs[1,0], sharex=ax1)
        x=np.linspace(*x_ranges[name], npts); wL, im, tw, wi, ki, xlabel=build_wL_scan(sys,x,xscale=xscale,active_modes=active_modes)
        w = sys.Om if w_eval is None else w_eval
        dOm, Gopt = sys.dOm_Gopt(w=w, wL=wL)
        ax1.plot(x, internal_to_Hz(dOm, sys), color="C0", lw=2, label="device")
        ax2.plot(x, Gopt/sys.gam, color="C0", lw=2)
        if std_devices and name in std_devices:
            s=std_devices[name]; wLs,*_=build_wL_scan(s,x,xscale=xscale,active_modes=active_modes)
            try:
                dO,stdG=s.dOm_Gopt(w=s.Om, wL=wLs)
                ax1.plot(x, internal_to_Hz(dO,s), color="k", lw=2, ls=(0,(1,3)), label="std")
                ax2.plot(x, stdG/s.gam, color="k", lw=2, ls=(0,(1,3)))
            except Exception:
                pass
        ax1.set_ylabel(r"$\delta\Omega_{\rm mec}/2\pi$ (Hz)"); ax2.set_ylabel(r"$\Gamma_{\rm opt}/\Gamma_{\rm mec}$"); ax2.set_xlabel(xlabel)
        ax1.tick_params(axis="x", labelbottom=False)
        for ax in [ax1,ax2]:
            ax.grid(False); ax.margins(x=0); ax.set_xlim(x[0],x[-1])
        _panel_title(ax1, labels[name], titles[name], im)
        raw[name]=dict(x=x,wL=wL,dOm=dOm,Gopt=Gopt)
        rows.append({"Device":name,"mode used":rf"$\Omega_{{{im}}}$","x-axis":xlabel,"range":str(x_ranges[name])})
    ax=fig.add_subplot(specs["info"]); ax.axis("off"); ax.text(0.03,0.97,"(d)", transform=ax.transAxes, va="top")
    handles=[_Line2D([0],[0], color="C0", lw=2, label="two-Fano"), _Line2D([0],[0], color="k", lw=2, ls=(0,(1,3)), label="std")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5,0.97), ncol=2, frameon=True)
    ptab=bare_parameter_table(devices); _make_info_table(ax, ptab, bbox=(0.02,0.02,0.96,0.72), fontsize=7.0)
    _plt.show()
    return {"fig":fig,"raw_data":raw,"parameter_table":ptab,"mode_table":_pd.DataFrame(rows)}
