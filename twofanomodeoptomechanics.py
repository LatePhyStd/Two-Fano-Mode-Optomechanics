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
atan = arctan
re = real
im = imag

# ------------------------- physical constants --------------------------------

e = 1.602e-19             # elementary charge
h = 6.62606957e-34/(2*pi) # \hbar
kB = 1.3806488e-23        # Boltzmann constant
c = 299792458             # speed of light


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

    Returns:
        x_THz       = lambda_bar/2pi in THz
        Delta_THz   = Re(Omega_i)/2pi in THz
        kappa_THz   = k_i/2pi = -Im(Omega_i)/2pi in THz
        eigs        = tracked complex eigenvalues
    """
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

        H = module.Hbare_aBdD(
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
    Table-1-style summary of only the quantities used for this plot.
    All optical-frequency-like quantities are shown as ordinary frequencies in THz.
    """
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




