# -*- coding: utf-8 -*-
"""
Functions to calculate the optical and optomechanical properties based on the results from

J. Monsel, A. Ciers, S. K. Manjeshwar, W. Wieczorek, J. Splettstoesser: Dissipative and dispersive cavity optomechanics with a frequency-dependent mirror. arXiv:2311.15311 (2023)
https://arxiv.org/abs/2311.15311

Please cite the article if you use this code.

| Variable | Notations in the article (see Table I)                   |
| -------- | -------------------------------------------------------- |
|     Plas | P_\text{las}                                             |
|    thlas | \theta_\text{las} (laser phase, irrelevant global phase) |
|       wL | \omega_\text{las}                                        |
|        w | \omega                                                   |
| -------- | -------------------------------------------------------- |
|       wa | \tilde{\omega}_a = \tilde{\Delta}_a - \omega_\text{las}  |
|       wd | \tilde{\omega}_d = \tilde{\Delta}_d - \omega_\text{las}  |
|       Da | \tilde{\Delta}_a                                         |
|       Dd | \tilde{\Delta}_d                                         |
|       ka | \tilde{\kappa}_a                                         |
|       kd | \tilde{\kappa}_d                                         |
|      ka0 | {\kappa}_a                                               |
|      kd0 | {\kappa}_d                                               |
|       ga | \gamma_a                                                 |
|      lbd | \lambda                                                  |
| -------- | -------------------------------------------------------- |
|       Om | \Omega_\text{mec}                                        |
|      gam | \Gamma_\text{mec}                                        |
|      Nth | \bar{n}_\text{mec}                                       |
| -------- | -------------------------------------------------------- |
|     gwa0 | g^\omega_a                                               |
|     gwd0 | g^\omega_d                                               |
|     gka0 | g^\kappa_a                                               |
|     gkd0 | g^\kappa_d                                               |
|     gadp | g^{\kappa,sym}                                           |
|     gadm | g^{\kappa,asym}                                          |
|     tga  | \tilde{g}_a                                              |
|     tgd  | \tilde{g}_d                                              |
|     tgma | \tilde{g}_{\text{mec}, a}                                |
|     tgmd | \tilde{g}_{\text{mec}, d}                                |
| -------- | -------------------------------------------------------- |
|      ass | \bar{a}                                                  |
|      dss | \bar{d}                                                  |
|      qss | \bar{q}                                                  |
| -------- | -------------------------------------------------------- |

Note that the functions take in argument the effective (after linearization) values of the frequencies (wa, wd) and loss rates (ka, kd),
and, if needed, compute from there the bare values (denoted with a 0), so one has to check when choosing parameters that these values make sense,
first of all ka0 and kd0 have to be positive.
Furthermore, the steady-state values ass, dss and qss are not obtained by solving the full nonlinear set of equation (XX), but instead
ass and dss are computed directly from Da, Dd, ka, kd, ... and then qss is obtained by linearizing the last equation.
Therefore, one should make sure that such an approximation make sense for the chosen parameters.
For most experimentally relevant parameters, the corrections to wa0, wd0, ka0 and kd0 are negligible.
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

# --------------------------------- optics ------------------------------------

def t_CM(ga, ka, kd, lbd, wL, wa, wd):
    """Amplitude transmission of the optical setup <b_{out,R}>/<b_{in,L}>."""
    return 2*1j*sqrt(ga)*(sqrt(ka)*(wL - wd) + sqrt(kd)*lbd)/(ga*kd - 1j*ga*(wL - wd) - 2*1j*sqrt(ka)*sqrt(kd)*lbd - 1j*ka*(wL - wd) - 1j*kd*(wL - wa) + lbd**2 - (wL - wa)*(wL - wd))


def r_CM(ga, ka, kd, lbd, wL, wa, wd):
    """Amplitude reflexion coef of whole setup  <b_{out,L}>/<b_{in,L}>."""
    return -(ga*kd + 1j*ga*(wL - wd) - 2*1j*sqrt(ka)*sqrt(kd)*lbd - 1j*ka*(wL - wd) - 1j*kd*(wL - wa) - lbd**2 + (wL - wa)*(wL - wd))/(ga*kd - 1j*ga*(wL - wd) - 2*1j*sqrt(ka)*sqrt(kd)*lbd - 1j*ka*(wL - wd) - 1j*kd*(wL - wa) + lbd**2 - (wL - wa)*(wL - wd))


def dTs(ga, ka, kd, lbd, wL, wa, wd):
    """Derivatives of the transmission T with respect to Da, Dd, ka and kd."""
    Da = wa - wL
    Dd = wd - wL
    dDdDa = 2*Da*Dd**2 + 2*Da*kd**2 + 2*Dd*ka*kd - 2*Dd*lbd**2 - 4*sqrt(ka)*kd**(3/2)*lbd
    dDdDd = 2*Da**2*Dd + 2*Da*ka*kd - 2*Da*lbd**2 + 2*Dd*ga**2 + 4*Dd*ga*ka + 2*Dd*ka**2 - 4*ga*sqrt(ka)*sqrt(kd)*lbd - 4*ka**(3/2)*sqrt(kd)*lbd
    dDdka = 2*Da*Dd*kd - 2*Da*kd**(3/2)*lbd/sqrt(ka) + 2*Dd**2*ga + 2*Dd**2*ka - 2*Dd*ga*sqrt(kd)*lbd/sqrt(ka) - 6*Dd*sqrt(ka)*sqrt(kd)*lbd + 4*kd*lbd**2
    dDdkd = 2*Da**2*kd + 2*Da*Dd*ka - 6*Da*sqrt(ka)*sqrt(kd)*lbd - 2*Dd*ga*sqrt(ka)*lbd/sqrt(kd) - 2*Dd*ka**(3/2)*lbd/sqrt(kd) + 2*ga**2*kd + 2*ga*lbd**2 + 4*ka*lbd**2
    T = abs(t_CM(ga, ka, kd, lbd, wL, wa, wd))**2
    D = abs(-Da*Dd + 1j*Da*kd + 1j*Dd*ga + 1j*Dd*ka + ga*kd - 2*1j*sqrt(ka)*sqrt(kd)*lbd + lbd**2)**2
    dTdDa = - dDdDa*T/D
    dTdDd = - dDdDd*T/D + 8*ga*(Dd*ka - sqrt(ka)*sqrt(kd)*lbd)/D
    dTdka = - dDdka*T/D + 4*ga*(Dd**2 - Dd*sqrt(kd)*lbd/sqrt(ka))/D
    dTdkd = - dDdkd*T/D -4*ga*(Dd*sqrt(ka)*lbd/sqrt(kd) - lbd**2)/D
    return dTdDa, dTdDd, dTdka, dTdkd, T


def tw_p(ga, ka, kd, lbd, wa, wd):
    """Effective (complex) eigenfrequency \tilde{\Omega}_+"""
    return -1j*ga/2 - 1j*ka/2 - 1j*kd/2 + wa/2 + wd/2 + sqrt(-4*(sqrt(ka)*sqrt(kd) + 1j*lbd)**2 + (-1j*ga - 1j*ka + 1j*kd + wa - wd)**2)/2


def tw_m(ga, ka, kd, lbd, wa, wd):
    """Effective (complex) eigenfrequency \tilde{\Omega}_-"""
    return -1j*ga/2 - 1j*ka/2 - 1j*kd/2 + wa/2 + wd/2 - sqrt(-4*(sqrt(ka)*sqrt(kd) + 1j*lbd)**2 + (-1j*ga - 1j*ka + 1j*kd + wa - wd)**2)/2


def gp0(ga, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, wa, wd):
    """Effective single-photon coupling (complex) corresponding to eigenmode +"""
    return sqrt(2)*(2*sqrt(2)*1j*gka0 + 2*sqrt(2)*1j*gkd0 + 2*sqrt(2)*gwa0 + 2*sqrt(2)*gwd0 - (-2*(sqrt(ka)*sqrt(kd) + 1j*lbd)*(sqrt(2)*gka0*sqrt(kd)/sqrt(ka) + sqrt(2)*gkd0*sqrt(ka)/sqrt(kd)) + (sqrt(2)*1j*gka0 - sqrt(2)*1j*gkd0 + sqrt(2)*gwa0 - sqrt(2)*gwd0)*(1j*ga + 1j*ka - 1j*kd - wa + wd))/sqrt(-(sqrt(ka)*sqrt(kd) + 1j*lbd)**2 + (-1j*ga/2 - 1j*ka/2 + 1j*kd/2 + wa/2 - wd/2)**2))/8


def gm0(ga, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, wa, wd):
    """Effective single-photon coupling (complex) corresponding to eigenmode -"""
    return sqrt(2)*(2*sqrt(2)*1j*gka0 + 2*sqrt(2)*1j*gkd0 + 2*sqrt(2)*gwa0 + 2*sqrt(2)*gwd0 + (-2*(sqrt(ka)*sqrt(kd) + 1j*lbd)*(sqrt(2)*gka0*sqrt(kd)/sqrt(ka) + sqrt(2)*gkd0*sqrt(ka)/sqrt(kd)) + (sqrt(2)*1j*gka0 - sqrt(2)*1j*gkd0 + sqrt(2)*gwa0 - sqrt(2)*gwd0)*(1j*ga + 1j*ka - 1j*kd - wa + wd))/sqrt(-(sqrt(ka)*sqrt(kd) + 1j*lbd)**2 + (-1j*ga/2 - 1j*ka/2 + 1j*kd/2 + wa/2 - wd/2)**2))/8


# ------------------------------ steady state ---------------------------------

def Na_ss(Plas, ga, ka, kd, lbd, wL, wa, wd):
    """Steady-state cavity photon number as a function ssof effective parameters (that themselves depends on the ss)."""
    return 2*Plas*(2*sqrt(ka)*sqrt(kd)*lbd*wL - 2*sqrt(ka)*sqrt(kd)*lbd*wd + ka*wL**2 - 2*ka*wL*wd + ka*wd**2 + kd*lbd**2)/(h*wL*(ga**2*kd**2 + ga**2*wL**2 - 2*ga**2*wL*wd + ga**2*wd**2 + 4*ga*sqrt(ka)*sqrt(kd)*lbd*wL - 4*ga*sqrt(ka)*sqrt(kd)*lbd*wd + 2*ga*ka*wL**2 - 4*ga*ka*wL*wd + 2*ga*ka*wd**2 + 2*ga*kd*lbd**2 + 4*ka**(3/2)*sqrt(kd)*lbd*wL - 4*ka**(3/2)*sqrt(kd)*lbd*wd + 4*sqrt(ka)*kd**(3/2)*lbd*wL - 4*sqrt(ka)*kd**(3/2)*lbd*wa + ka**2*wL**2 - 2*ka**2*wL*wd + ka**2*wd**2 + 4*ka*kd*lbd**2 + 2*ka*kd*wL**2 - 2*ka*kd*wL*wa - 2*ka*kd*wL*wd + 2*ka*kd*wa*wd + kd**2*wL**2 - 2*kd**2*wL*wa + kd**2*wa**2 + lbd**4 - 2*lbd**2*wL**2 + 2*lbd**2*wL*wa + 2*lbd**2*wL*wd - 2*lbd**2*wa*wd + wL**4 - 2*wL**3*wa - 2*wL**3*wd + wL**2*wa**2 + 4*wL**2*wa*wd + wL**2*wd**2 - 2*wL*wa**2*wd - 2*wL*wa*wd**2 + wa**2*wd**2))


def Nd_ss(Plas, ga, ka, kd, lbd, wL, wa, wd):
    """Steady-state mirror photon number as a function of effective parameters (that themselves depends on the ss)."""
    return 2*Plas*(ga**2*kd + 2*sqrt(ka)*sqrt(kd)*lbd*wL - 2*sqrt(ka)*sqrt(kd)*lbd*wa + ka*lbd**2 + kd*wL**2 - 2*kd*wL*wa + kd*wa**2)/(h*wL*(ga**2*kd**2 + ga**2*wL**2 - 2*ga**2*wL*wd + ga**2*wd**2 + 4*ga*sqrt(ka)*sqrt(kd)*lbd*wL - 4*ga*sqrt(ka)*sqrt(kd)*lbd*wd + 2*ga*ka*wL**2 - 4*ga*ka*wL*wd + 2*ga*ka*wd**2 + 2*ga*kd*lbd**2 + 4*ka**(3/2)*sqrt(kd)*lbd*wL - 4*ka**(3/2)*sqrt(kd)*lbd*wd + 4*sqrt(ka)*kd**(3/2)*lbd*wL - 4*sqrt(ka)*kd**(3/2)*lbd*wa + ka**2*wL**2 - 2*ka**2*wL*wd + ka**2*wd**2 + 4*ka*kd*lbd**2 + 2*ka*kd*wL**2 - 2*ka*kd*wL*wa - 2*ka*kd*wL*wd + 2*ka*kd*wa*wd + kd**2*wL**2 - 2*kd**2*wL*wa + kd**2*wa**2 + lbd**4 - 2*lbd**2*wL**2 + 2*lbd**2*wL*wa + 2*lbd**2*wL*wd - 2*lbd**2*wa*wd + wL**4 - 2*wL**3*wa - 2*wL**3*wd + wL**2*wa**2 + 4*wL**2*wa*wd + wL**2*wd**2 - 2*wL*wa**2*wd - 2*wL*wa*wd**2 + wa**2*wd**2))


def a_ss(Plas, ga, ka, kd, lbd, thlas, wL, wa, wd):
    """Steady-state cavity field amplitude as a function of effective parameters (that themselves depends on the ss)."""
    return -sqrt(2)*1j*sqrt(Plas/(h*wL))*(sqrt(ka)*(wL - wd) + sqrt(kd)*lbd)*exp(1j*thlas)/(ga*kd - 1j*ga*(wL - wd) - 2*1j*sqrt(ka)*sqrt(kd)*lbd - 1j*ka*(wL - wd) - 1j*kd*(wL - wa) + lbd**2 - (wL - wa)*(wL - wd))


def d_ss(Plas, ga, ka, kd, lbd, thlas, wL, wa, wd):
    """Steady-state mirror field amplitude as a function of effective parameters (that themselves depends on the ss)."""
    return sqrt(2)*sqrt(Plas/(h*wL))*(ga*sqrt(kd) - 1j*sqrt(ka)*lbd - 1j*sqrt(kd)*(wL - wa))*exp(1j*thlas)/(ga*kd - 1j*ga*(wL - wd) - 2*1j*sqrt(ka)*sqrt(kd)*lbd - 1j*ka*(wL - wd) - 1j*kd*(wL - wa) + lbd**2 - (wL - wa)*(wL - wd))


def q_ss(Om, Plas, ga, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, wL, wa, wd):
    """Steady-state mechanical position as a function of effective parameters (that themselves depends on the ss)."""
    return 2*sqrt(2)*Plas*ka*kd*(-ga**2*gkd0*sqrt(ka)*sqrt(kd)*wL + ga**2*gkd0*sqrt(ka)*sqrt(kd)*wd + ga**2*gwd0*sqrt(ka)*kd**(3/2) + gka0*sqrt(ka)*sqrt(kd)*lbd**2*wL - gka0*sqrt(ka)*sqrt(kd)*lbd**2*wd - gka0*sqrt(ka)*sqrt(kd)*wL**3 + gka0*sqrt(ka)*sqrt(kd)*wL**2*wa + 2*gka0*sqrt(ka)*sqrt(kd)*wL**2*wd - 2*gka0*sqrt(ka)*sqrt(kd)*wL*wa*wd - gka0*sqrt(ka)*sqrt(kd)*wL*wd**2 + gka0*sqrt(ka)*sqrt(kd)*wa*wd**2 + gka0*kd*lbd**3 - gka0*kd*lbd*wL**2 + gka0*kd*lbd*wL*wa + gka0*kd*lbd*wL*wd - gka0*kd*lbd*wa*wd + gkd0*sqrt(ka)*sqrt(kd)*lbd**2*wL - gkd0*sqrt(ka)*sqrt(kd)*lbd**2*wa - gkd0*sqrt(ka)*sqrt(kd)*wL**3 + 2*gkd0*sqrt(ka)*sqrt(kd)*wL**2*wa + gkd0*sqrt(ka)*sqrt(kd)*wL**2*wd - gkd0*sqrt(ka)*sqrt(kd)*wL*wa**2 - 2*gkd0*sqrt(ka)*sqrt(kd)*wL*wa*wd + gkd0*sqrt(ka)*sqrt(kd)*wa**2*wd + gkd0*ka*lbd**3 - gkd0*ka*lbd*wL**2 + gkd0*ka*lbd*wL*wa + gkd0*ka*lbd*wL*wd - gkd0*ka*lbd*wa*wd + gwa0*ka**(3/2)*sqrt(kd)*wL**2 - 2*gwa0*ka**(3/2)*sqrt(kd)*wL*wd + gwa0*ka**(3/2)*sqrt(kd)*wd**2 + gwa0*sqrt(ka)*kd**(3/2)*lbd**2 + 2*gwa0*ka*kd*lbd*wL - 2*gwa0*ka*kd*lbd*wd + gwd0*ka**(3/2)*sqrt(kd)*lbd**2 + gwd0*sqrt(ka)*kd**(3/2)*wL**2 - 2*gwd0*sqrt(ka)*kd**(3/2)*wL*wa + gwd0*sqrt(ka)*kd**(3/2)*wa**2 + 2*gwd0*ka*kd*lbd*wL - 2*gwd0*ka*kd*lbd*wa)/(Om*ga**2*h*ka**(3/2)*kd**(7/2)*wL + Om*ga**2*h*ka**(3/2)*kd**(3/2)*wL**3 - 2*Om*ga**2*h*ka**(3/2)*kd**(3/2)*wL**2*wd + Om*ga**2*h*ka**(3/2)*kd**(3/2)*wL*wd**2 + 2*Om*ga*h*ka**(5/2)*kd**(3/2)*wL**3 - 4*Om*ga*h*ka**(5/2)*kd**(3/2)*wL**2*wd + 2*Om*ga*h*ka**(5/2)*kd**(3/2)*wL*wd**2 + 2*Om*ga*h*ka**(3/2)*kd**(5/2)*lbd**2*wL + 4*Om*ga*h*ka**2*kd**2*lbd*wL**2 - 4*Om*ga*h*ka**2*kd**2*lbd*wL*wd + Om*h*ka**(7/2)*kd**(3/2)*wL**3 - 2*Om*h*ka**(7/2)*kd**(3/2)*wL**2*wd + Om*h*ka**(7/2)*kd**(3/2)*wL*wd**2 + 4*Om*h*ka**(5/2)*kd**(5/2)*lbd**2*wL + 2*Om*h*ka**(5/2)*kd**(5/2)*wL**3 - 2*Om*h*ka**(5/2)*kd**(5/2)*wL**2*wa - 2*Om*h*ka**(5/2)*kd**(5/2)*wL**2*wd + 2*Om*h*ka**(5/2)*kd**(5/2)*wL*wa*wd + Om*h*ka**(3/2)*kd**(7/2)*wL**3 - 2*Om*h*ka**(3/2)*kd**(7/2)*wL**2*wa + Om*h*ka**(3/2)*kd**(7/2)*wL*wa**2 + Om*h*ka**(3/2)*kd**(3/2)*lbd**4*wL - 2*Om*h*ka**(3/2)*kd**(3/2)*lbd**2*wL**3 + 2*Om*h*ka**(3/2)*kd**(3/2)*lbd**2*wL**2*wa + 2*Om*h*ka**(3/2)*kd**(3/2)*lbd**2*wL**2*wd - 2*Om*h*ka**(3/2)*kd**(3/2)*lbd**2*wL*wa*wd + Om*h*ka**(3/2)*kd**(3/2)*wL**5 - 2*Om*h*ka**(3/2)*kd**(3/2)*wL**4*wa - 2*Om*h*ka**(3/2)*kd**(3/2)*wL**4*wd + Om*h*ka**(3/2)*kd**(3/2)*wL**3*wa**2 + 4*Om*h*ka**(3/2)*kd**(3/2)*wL**3*wa*wd + Om*h*ka**(3/2)*kd**(3/2)*wL**3*wd**2 - 2*Om*h*ka**(3/2)*kd**(3/2)*wL**2*wa**2*wd - 2*Om*h*ka**(3/2)*kd**(3/2)*wL**2*wa*wd**2 + Om*h*ka**(3/2)*kd**(3/2)*wL*wa**2*wd**2 + 4*Om*h*ka**3*kd**2*lbd*wL**2 - 4*Om*h*ka**3*kd**2*lbd*wL*wd + 4*Om*h*ka**2*kd**3*lbd*wL**2 - 4*Om*h*ka**2*kd**3*lbd*wL*wa + 2*Plas*ga**2*gkd0**2*ka**(3/2)*sqrt(kd)*wL - 2*Plas*ga**2*gkd0**2*ka**(3/2)*sqrt(kd)*wd - 2*Plas*gka0**2*sqrt(ka)*kd**(3/2)*lbd**2*wL + 2*Plas*gka0**2*sqrt(ka)*kd**(3/2)*lbd**2*wd + 2*Plas*gka0**2*sqrt(ka)*kd**(3/2)*wL**3 - 2*Plas*gka0**2*sqrt(ka)*kd**(3/2)*wL**2*wa - 4*Plas*gka0**2*sqrt(ka)*kd**(3/2)*wL**2*wd + 4*Plas*gka0**2*sqrt(ka)*kd**(3/2)*wL*wa*wd + 2*Plas*gka0**2*sqrt(ka)*kd**(3/2)*wL*wd**2 - 2*Plas*gka0**2*sqrt(ka)*kd**(3/2)*wa*wd**2 - 2*Plas*gka0**2*kd**2*lbd**3 + 2*Plas*gka0**2*kd**2*lbd*wL**2 - 2*Plas*gka0**2*kd**2*lbd*wL*wa - 2*Plas*gka0**2*kd**2*lbd*wL*wd + 2*Plas*gka0**2*kd**2*lbd*wa*wd - 2*Plas*gkd0**2*ka**(3/2)*sqrt(kd)*lbd**2*wL + 2*Plas*gkd0**2*ka**(3/2)*sqrt(kd)*lbd**2*wa + 2*Plas*gkd0**2*ka**(3/2)*sqrt(kd)*wL**3 - 4*Plas*gkd0**2*ka**(3/2)*sqrt(kd)*wL**2*wa - 2*Plas*gkd0**2*ka**(3/2)*sqrt(kd)*wL**2*wd + 2*Plas*gkd0**2*ka**(3/2)*sqrt(kd)*wL*wa**2 + 4*Plas*gkd0**2*ka**(3/2)*sqrt(kd)*wL*wa*wd - 2*Plas*gkd0**2*ka**(3/2)*sqrt(kd)*wa**2*wd - 2*Plas*gkd0**2*ka**2*lbd**3 + 2*Plas*gkd0**2*ka**2*lbd*wL**2 - 2*Plas*gkd0**2*ka**2*lbd*wL*wa - 2*Plas*gkd0**2*ka**2*lbd*wL*wd + 2*Plas*gkd0**2*ka**2*lbd*wa*wd)


# --------------------- effective mechanical properties -----------------------

def g_effs(Om, Plas, ga, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, wL, wa, wd):
    r"""Effective OM couplings \tilde{g}_{mec,a}, \tilde{g}_{mec,d}, \tilde{g}_{a} and \tilde{g}_{d}"""
    Da = -wL + wa
    Dd = -wL + wd
    a_las = sqrt(Plas/h/wL)*exp(1j*thlas)
    Denom = (ga*kd - 1j*ga*(-Dd) - 2*1j*sqrt(ka)*sqrt(kd)*lbd - 1j*ka*(-Dd) - 1j*kd*(-Da) + lbd**2 - (-Da)*(-Dd))
    ass = -sqrt(2)*1j*sqrt(Plas/(h*wL))*(sqrt(ka)*(-Dd) + sqrt(kd)*lbd)*exp(1j*thlas)/Denom
    dss = sqrt(2)*sqrt(Plas/(h*wL))*(ga*sqrt(kd) - 1j*sqrt(ka)*lbd - 1j*sqrt(kd)*(-Da))*exp(1j*thlas)/Denom
    qss = q_ss(Om, Plas, ga, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, wL, wa, wd)
    ka0 = -sqrt(2)*gka0*qss + ka
    kd0 = -sqrt(2)*gkd0*qss + kd
    gadm = sqrt(ka0)*sqrt(kd0)*(gka0/ka0 - gkd0/kd0)/2
    gadp = sqrt(ka0)*sqrt(kd0)*(gka0/ka0 + gkd0/kd0)/2
    tgma = -sqrt(2)*1j*a_las*gka0/(2*sqrt(ka0)) + ass*gwa0 + 1j*dss*gadm
    tgmd = -sqrt(2)*1j*a_las*gkd0/(2*sqrt(kd0)) - 1j*ass*gadm + dss*gwd0
    tga = ass*gwa0 + 1j*dss*gadp + 1j*gka0*(-sqrt(2)*a_las/sqrt(ka0) + 2*ass)/2
    tgd = 1j*ass*gadp + dss*gwd0 + 1j*gkd0*(-sqrt(2)*a_las/sqrt(kd0) + 2*dss)/2
    return tgma, tgmd, tga, tgd


def Xmeff(Nth, Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, w, wL, wa, wd):
    """Effective mechanical susceptibility."""
    Da = -wL + wa
    Dd = -wL + wd
    cG = sqrt(ka)*sqrt(kd) + 1j*lbd
    a_las = sqrt(Plas/h/wL)*exp(1j*thlas)
    Dw = -cG**2 + (1j*Dd + kd - 1j*w)*(1j*Da + ga + ka - 1j*w)
    Dmw = -cG**2 + (1j*Dd + kd + 1j*w)*(1j*Da + ga + ka + 1j*w)

    ass = -sqrt(2)*1j*sqrt(Plas/(h*wL))*(sqrt(ka)*(wL - wd) + sqrt(kd)*lbd)*exp(1j*thlas)/(ga*kd - 1j*ga*(wL - wd) - 2*1j*sqrt(ka)*sqrt(kd)*lbd - 1j*ka*(wL - wd) - 1j*kd*(wL - wa) + lbd**2 - (wL - wa)*(wL - wd))
    dss = sqrt(2)*sqrt(Plas/(h*wL))*(ga*sqrt(kd) - 1j*sqrt(ka)*lbd - 1j*sqrt(kd)*(wL - wa))*exp(1j*thlas)/(ga*kd - 1j*ga*(wL - wd) - 2*1j*sqrt(ka)*sqrt(kd)*lbd - 1j*ka*(wL - wd) - 1j*kd*(wL - wa) + lbd**2 - (wL - wa)*(wL - wd))
    qss = q_ss(Om, Plas, ga, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, wL, wa, wd)

    ka0 = -sqrt(2)*gka0*qss + ka
    kd0 = -sqrt(2)*gkd0*qss + kd
    gadm = sqrt(ka0)*sqrt(kd0)*(gka0/ka0 - gkd0/kd0)/2
    gadp = sqrt(ka0)*sqrt(kd0)*(gka0/ka0 + gkd0/kd0)/2

    tgma = -sqrt(2)*1j*a_las*gka0/(2*sqrt(ka0)) + ass*gwa0 + 1j*dss*gadm
    tgmd = -sqrt(2)*1j*a_las*gkd0/(2*sqrt(kd0)) - 1j*ass*gadm + dss*gwd0
    tga = ass*gwa0 + 1j*dss*gadp + 1j*gka0*(-sqrt(2)*a_las/sqrt(ka0) + 2*ass)/2
    tgd = 1j*ass*gadp + dss*gwd0 + 1j*gkd0*(-sqrt(2)*a_las/sqrt(kd0) + 2*dss)/2

    tgma_cc = tgma.conjugate()
    tgmd_cc = tgmd.conjugate()

    chi_a_inv = ka + ga + 1j*(Da-w)
    chi_d_inv = kd + 1j*(Dd-w)
    chi_a_inv_m = ka + ga + 1j*(Da+w)
    chi_d_inv_m = kd + 1j*(Dd+w)

    Ca_q = 1j*(chi_d_inv*tga - cG*tgd)/Dw
    Cd_q = 1j*(chi_a_inv*tgd - cG*tga)/Dw

    Ca_qm_cc = (1j*(chi_d_inv_m*tga - cG*tgd)/Dmw).conjugate()
    Cd_qm_cc = (1j*(chi_a_inv_m*tgd - cG*tga)/Dmw).conjugate()

    Xopti = -2*(tgma_cc*Ca_q + tgma*Ca_qm_cc) -2*(tgmd_cc*Cd_q + tgmd*Cd_qm_cc)

    return 1/(Xopti + (Om**2 - 1j*gam*w - w**2)/Om)


def Xopt_inv_components(Nth, Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, w, wL, wa, wd):
    """Components of \chi_\text{opt}^{-1}."""
    Da = -wL + wa
    Dd = -wL + wd
    cG = sqrt(ka)*sqrt(kd) + 1j*lbd
    a_las = sqrt(Plas/h/wL)*exp(1j*thlas)
    Dw = -cG**2 + (1j*Dd + kd - 1j*w)*(1j*Da + ga + ka - 1j*w)
    Dmw = -cG**2 + (1j*Dd + kd + 1j*w)*(1j*Da + ga + ka + 1j*w)

    ass = -sqrt(2)*1j*sqrt(Plas/(h*wL))*(sqrt(ka)*(wL - wd) + sqrt(kd)*lbd)*exp(1j*thlas)/(ga*kd - 1j*ga*(wL - wd) - 2*1j*sqrt(ka)*sqrt(kd)*lbd - 1j*ka*(wL - wd) - 1j*kd*(wL - wa) + lbd**2 - (wL - wa)*(wL - wd))
    dss = sqrt(2)*sqrt(Plas/(h*wL))*(ga*sqrt(kd) - 1j*sqrt(ka)*lbd - 1j*sqrt(kd)*(wL - wa))*exp(1j*thlas)/(ga*kd - 1j*ga*(wL - wd) - 2*1j*sqrt(ka)*sqrt(kd)*lbd - 1j*ka*(wL - wd) - 1j*kd*(wL - wa) + lbd**2 - (wL - wa)*(wL - wd))
    qss = q_ss(Om, Plas, ga, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, wL, wa, wd)

    ka0 = -sqrt(2)*gka0*qss + ka
    kd0 = -sqrt(2)*gkd0*qss + kd
    gadm = sqrt(ka0)*sqrt(kd0)*(gka0/ka0 - gkd0/kd0)/2
    gadp = sqrt(ka0)*sqrt(kd0)*(gka0/ka0 + gkd0/kd0)/2

    tgma = -sqrt(2)*1j*a_las*gka0/(2*sqrt(ka0)) + ass*gwa0 + 1j*dss*gadm
    tgmd = -sqrt(2)*1j*a_las*gkd0/(2*sqrt(kd0)) - 1j*ass*gadm + dss*gwd0
    tga = ass*gwa0 + 1j*dss*gadp + 1j*gka0*(-sqrt(2)*a_las/sqrt(ka0) + 2*ass)/2
    tgd = 1j*ass*gadp + dss*gwd0 + 1j*gkd0*(-sqrt(2)*a_las/sqrt(kd0) + 2*dss)/2

    tgma_cc = tgma.conjugate()
    tgmd_cc = tgmd.conjugate()

    chi_a_inv = ka + ga + 1j*(Da-w)
    chi_d_inv = kd + 1j*(Dd-w)
    chi_a_inv_m = ka + ga + 1j*(Da+w)
    chi_d_inv_m = kd + 1j*(Dd+w)

    Xopti_a = -2j*(tgma_cc*tga*chi_d_inv/Dw) + (-2j*(tgma_cc*tga*chi_d_inv_m/Dmw)).conjugate()
    Xopti_d = -2j*(tgmd_cc*tgd*chi_a_inv/Dw) + (-2j*(tgmd_cc*tgd*chi_a_inv_m/Dmw)).conjugate()
    Xopti_ad = 2j*(tgma_cc*tgd + tgmd_cc*tga)*cG/Dw + (2j*(tgma_cc*tgd + tgmd_cc*tga)*cG/Dmw).conjugate()
    return Xopti_a, Xopti_d, Xopti_ad


def Xopt_inv(Om, Plas, ga, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, w, wL, wa, wd):
    """Optical contribution to the inverse of the effective mechanical susceptibility."""
    Da = -wL + wa
    Dd = -wL + wd
    cG = sqrt(ka)*sqrt(kd) + 1j*lbd
    a_las = sqrt(Plas/h/wL)*exp(1j*thlas)
    Dw = -cG**2 + (1j*Dd + kd - 1j*w)*(1j*Da + ga + ka - 1j*w)
    ass = -sqrt(2)*1j*sqrt(Plas/(h*wL))*(sqrt(ka)*(wL - wd) + sqrt(kd)*lbd)*exp(1j*thlas)/(ga*kd - 1j*ga*(wL - wd) - 2*1j*sqrt(ka)*sqrt(kd)*lbd - 1j*ka*(wL - wd) - 1j*kd*(wL - wa) + lbd**2 - (wL - wa)*(wL - wd))
    dss = sqrt(2)*sqrt(Plas/(h*wL))*(ga*sqrt(kd) - 1j*sqrt(ka)*lbd - 1j*sqrt(kd)*(wL - wa))*exp(1j*thlas)/(ga*kd - 1j*ga*(wL - wd) - 2*1j*sqrt(ka)*sqrt(kd)*lbd - 1j*ka*(wL - wd) - 1j*kd*(wL - wa) + lbd**2 - (wL - wa)*(wL - wd))
    qss = q_ss(Om, Plas, ga, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, wL, wa, wd)
    ka0 = -sqrt(2)*gka0*qss + ka
    kd0 = -sqrt(2)*gkd0*qss + kd
    gadm = sqrt(ka0)*sqrt(kd0)*(gka0/ka0 - gkd0/kd0)/2
    gadp = sqrt(ka0)*sqrt(kd0)*(gka0/ka0 + gkd0/kd0)/2
    tgma = -sqrt(2)*1j*a_las*gka0/(2*sqrt(ka0)) + ass*gwa0 + 1j*dss*gadm
    tgmd = -sqrt(2)*1j*a_las*gkd0/(2*sqrt(kd0)) - 1j*ass*gadm + dss*gwd0
    tga = ass*gwa0 + 1j*dss*gadp + 1j*gka0*(-sqrt(2)*a_las/sqrt(ka0) + 2*ass)/2
    tgd = 1j*ass*gadp + dss*gwd0 + 1j*gkd0*(-sqrt(2)*a_las/sqrt(kd0) + 2*dss)/2
    Dmw = -cG**2 + (1j*Dd + kd + 1j*w)*(1j*Da + ga + ka + 1j*w)
    return 2*(Da*Dw*tgmd*conjugate(tgd) + Da*tgd*conjugate(Dmw)*conjugate(tgmd) + Dd*Dw*tgma*conjugate(tga) + Dd*tga*conjugate(Dmw)*conjugate(tgma) + 1j*Dw*ga*tgmd*conjugate(tgd) + 1j*Dw*ka*tgmd*conjugate(tgd) + 1j*Dw*kd*tgma*conjugate(tga) + Dw*tgma*w*conjugate(tga) - 1j*Dw*tgma*conjugate(cG)*conjugate(tgd) + Dw*tgmd*w*conjugate(tgd) - 1j*Dw*tgmd*conjugate(cG)*conjugate(tga) + 1j*cG*tga*conjugate(Dmw)*conjugate(tgmd) + 1j*cG*tgd*conjugate(Dmw)*conjugate(tgma) - 1j*ga*tgd*conjugate(Dmw)*conjugate(tgmd) - 1j*ka*tgd*conjugate(Dmw)*conjugate(tgmd) - 1j*kd*tga*conjugate(Dmw)*conjugate(tgma) - tga*w*conjugate(Dmw)*conjugate(tgma) - tgd*w*conjugate(Dmw)*conjugate(tgmd))/(Dw*conjugate(Dmw))


def dOm_Gopt(Om, Plas, ga, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, w, wL, wa, wd):
    """
    Mechanical frequency shift and Optical contribution to the mechanical damping.

    In the weak-coupling regime, the effective frequency shift and damping rate
    obtained by tracing out the optical part are also given by this formula when taking w=Om.
    """
    Xopti = Xopt_inv(Om, Plas, ga, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, w, wL, wa, wd)
    return re(Xopti)/2, -Om*im(Xopti)/w



# ------------------------------ Noise power spectra --------------------------

def spectum_coefs(Nth, Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, w, wL, wa, wd):
    """Return the coefficients needed to calculate the various output spectra (auxiliary function)."""
    Da = -wL + wa
    Dd = -wL + wd
    cG = sqrt(ka)*sqrt(kd) + 1j*lbd
    a_las = sqrt(Plas/h/wL)*exp(1j*thlas)
    Dw = -cG**2 + (1j*Dd + kd - 1j*w)*(1j*Da + ga + ka - 1j*w)
    Dmw = -cG**2 + (1j*Dd + kd + 1j*w)*(1j*Da + ga + ka + 1j*w)

    ass = -sqrt(2)*1j*sqrt(Plas/(h*wL))*(sqrt(ka)*(wL - wd) + sqrt(kd)*lbd)*exp(1j*thlas)/(ga*kd - 1j*ga*(wL - wd) - 2*1j*sqrt(ka)*sqrt(kd)*lbd - 1j*ka*(wL - wd) - 1j*kd*(wL - wa) + lbd**2 - (wL - wa)*(wL - wd))
    dss = sqrt(2)*sqrt(Plas/(h*wL))*(ga*sqrt(kd) - 1j*sqrt(ka)*lbd - 1j*sqrt(kd)*(wL - wa))*exp(1j*thlas)/(ga*kd - 1j*ga*(wL - wd) - 2*1j*sqrt(ka)*sqrt(kd)*lbd - 1j*ka*(wL - wd) - 1j*kd*(wL - wa) + lbd**2 - (wL - wa)*(wL - wd))
    qss = q_ss(Om, Plas, ga, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, wL, wa, wd)

    ka0 = -sqrt(2)*gka0*qss + ka
    kd0 = -sqrt(2)*gkd0*qss + kd
    gadm = sqrt(ka0)*sqrt(kd0)*(gka0/ka0 - gkd0/kd0)/2
    gadp = sqrt(ka0)*sqrt(kd0)*(gka0/ka0 + gkd0/kd0)/2

    tgma = -sqrt(2)*1j*a_las*gka0/(2*sqrt(ka0)) + ass*gwa0 + 1j*dss*gadm
    tgmd = -sqrt(2)*1j*a_las*gkd0/(2*sqrt(kd0)) - 1j*ass*gadm + dss*gwd0
    tga = ass*gwa0 + 1j*dss*gadp + 1j*gka0*(-sqrt(2)*a_las/sqrt(ka0) + 2*ass)/2
    tgd = 1j*ass*gadp + dss*gwd0 + 1j*gkd0*(-sqrt(2)*a_las/sqrt(kd0) + 2*dss)/2

    tgma_cc = tgma.conjugate()
    tgmd_cc = tgmd.conjugate()

    cX = -(gka0*sqrt(kd0)*im(ass) + gkd0*sqrt(ka0)*im(dss))/(sqrt(ka0)*sqrt(kd0))
    cP = (gka0*sqrt(kd0)*re(ass) + gkd0*sqrt(ka0)*re(dss))/(sqrt(ka0)*sqrt(kd0))

    chi_a_inv = ka + ga + 1j*(Da-w)
    chi_d_inv = kd + 1j*(Dd-w)
    chi_a_inv_m = ka + ga + 1j*(Da+w)
    chi_d_inv_m = kd + 1j*(Dd+w)

    Ca_q = 1j*(chi_d_inv*tga - cG*tgd)/Dw
    Ca_L = (sqrt(2*ka)*chi_d_inv - sqrt(2*kd)*cG)/Dw
    Ca_R = (sqrt(2*ga)*chi_d_inv)/Dw
    Cd_q = 1j*(chi_a_inv*tgd - cG*tga)/Dw
    Cd_L = (sqrt(2*kd)*chi_a_inv - sqrt(2*ka)*cG)/Dw
    Cd_R = (-sqrt(2*ga)*cG)/Dw

    Ca_qm_cc = (1j*(chi_d_inv_m*tga - cG*tgd)/Dmw).conjugate()
    Ca_Lm_cc = ((sqrt(2*ka)*chi_d_inv_m - sqrt(2*kd)*cG)/Dmw).conjugate()
    Ca_Rm_cc = ((sqrt(2*ga)*chi_d_inv_m)/Dmw).conjugate()
    Cd_qm_cc = (1j*(chi_a_inv_m*tgd - cG*tga)/Dmw).conjugate()
    Cd_Lm_cc = ((sqrt(2*kd)*chi_a_inv_m - sqrt(2*ka)*cG)/Dmw).conjugate()
    Cd_Rm_cc = ((-sqrt(2*ga)*cG)/Dmw).conjugate()

    Xopti = -2*(tgma_cc*Ca_q + tgma*Ca_qm_cc) -2*(tgmd_cc*Cd_q + tgmd*Cd_qm_cc)
    Xmeff = 1/(Xopti + (Om**2 - 1j*gam*w - w**2)/Om)

    # q
    c_q_XL = Xmeff*(tgma_cc*Ca_L + tgma*Ca_Lm_cc + tgmd_cc*Cd_L + tgmd*Cd_Lm_cc + cX*sqrt(2))
    c_q_XR = Xmeff*(tgma_cc*Ca_R + tgma*Ca_Rm_cc + tgmd_cc*Cd_R + tgmd*Cd_Rm_cc)
    c_q_PL = Xmeff*(1j*tgma_cc*Ca_L - 1j*tgma*Ca_Lm_cc + 1j*tgmd_cc*Cd_L - 1j*tgmd*Cd_Lm_cc+ cP*sqrt(2))
    c_q_PR = Xmeff*(1j*tgma_cc*Ca_R - 1j*tgma*Ca_Rm_cc + 1j*tgmd_cc*Cd_R - 1j*tgmd*Cd_Rm_cc)
    c_q_xi = Xmeff*sqrt(gam)

    # X_a
    c_Xa_XL = ((Ca_q + Ca_qm_cc)*c_q_XL + (Ca_L + Ca_Lm_cc)/2)
    c_Xa_PL = ((Ca_q + Ca_qm_cc)*c_q_PL + 1j*(Ca_L - Ca_Lm_cc)/2)
    c_Xa_XR = ((Ca_q + Ca_qm_cc)*c_q_XR + (Ca_R + Ca_Rm_cc)/2)
    c_Xa_PR = ((Ca_q + Ca_qm_cc)*c_q_PR + 1j*(Ca_R - Ca_Rm_cc)/2)
    c_Xa_xi = ((Ca_q + Ca_qm_cc)*c_q_xi)



    # X_d
    c_Xd_XL = ((Cd_q + Cd_qm_cc)*c_q_XL + (Cd_L + Cd_Lm_cc)/2)
    c_Xd_PL = ((Cd_q + Cd_qm_cc)*c_q_PL + 1j*(Cd_L - Cd_Lm_cc)/2)
    c_Xd_XR = ((Cd_q + Cd_qm_cc)*c_q_XR + (Cd_R + Cd_Rm_cc)/2)
    c_Xd_PR = ((Cd_q + Cd_qm_cc)*c_q_PR + 1j*(Cd_R - Cd_Rm_cc)/2)
    c_Xd_xi = ((Cd_q + Cd_qm_cc)*c_q_xi)

    # P_a
    c_Pa_XL = (-1j*(Ca_q - Ca_qm_cc)*c_q_XL - 1j*(Ca_L - Ca_Lm_cc)/2)
    c_Pa_PL = (-1j*(Ca_q - Ca_qm_cc)*c_q_PL + (Ca_L + Ca_Lm_cc)/2)
    c_Pa_XR = (-1j*(Ca_q - Ca_qm_cc)*c_q_XR - 1j*(Ca_R - Ca_Rm_cc)/2)
    c_Pa_PR = (-1j*(Ca_q - Ca_qm_cc)*c_q_PR + (Ca_R + Ca_Rm_cc)/2)
    c_Pa_xi = (-1j*(Ca_q - Ca_qm_cc)*c_q_xi)


    # P_d
    c_Pd_XL = (-1j*(Cd_q - Cd_qm_cc)*c_q_XL - 1j*(Cd_L - Cd_Lm_cc)/2)
    c_Pd_PL = (-1j*(Cd_q - Cd_qm_cc)*c_q_PL + (Cd_L + Cd_Lm_cc)/2)
    c_Pd_XR = (-1j*(Cd_q - Cd_qm_cc)*c_q_XR - 1j*(Cd_R - Cd_Rm_cc)/2)
    c_Pd_PR = (-1j*(Cd_q - Cd_qm_cc)*c_q_PR + (Cd_R + Cd_Rm_cc)/2)
    c_Pd_xi = (-1j*(Cd_q - Cd_qm_cc)*c_q_xi)

    coefs = (c_q_XL, c_q_XR, c_q_PL, c_q_PR, c_q_xi,
             c_Xa_XL, c_Xa_XR, c_Xa_PL, c_Xa_PR, c_Xa_xi,
             c_Pa_XL, c_Pa_XR, c_Pa_PL, c_Pa_PR, c_Pa_xi,
             c_Xd_XL, c_Xd_XR, c_Xd_PL, c_Xd_PR, c_Xd_xi,
             c_Pd_XL, c_Pd_XR, c_Pd_PL, c_Pd_PR, c_Pd_xi)

    return coefs, cX, cP

def S_outL(Nth, Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, w, wL, wa, wd):
    """Noise power spectrum of the quadratures of the mechanical position S_q and of the left output light field S_XoutL, S_PoutL."""
    coefs, cX, cP = spectum_coefs(Nth, Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, w, wL, wa, wd)
    coefs_cc = (coef.conjugate() for coef in coefs)

    (
        c_q_XL, c_q_XR, c_q_PL, c_q_PR, c_q_xi,
        c_Xa_XL, c_Xa_XR, c_Xa_PL, c_Xa_PR, c_Xa_xi,
        c_Pa_XL, c_Pa_XR, c_Pa_PL, c_Pa_PR, c_Pa_xi,
        c_Xd_XL, c_Xd_XR, c_Xd_PL, c_Xd_PR, c_Xd_xi,
        c_Pd_XL, c_Pd_XR, c_Pd_PL, c_Pd_PR, c_Pd_xi
    ) = coefs
    (
        c_q_XL_cc, c_q_XR_cc, c_q_PL_cc, c_q_PR_cc, c_q_xi_cc,
        c_Xa_XL_cc, c_Xa_XR_cc, c_Xa_PL_cc, c_Xa_PR_cc, c_Xa_xi_cc,
        c_Pa_XL_cc, c_Pa_XR_cc, c_Pa_PL_cc, c_Pa_PR_cc, c_Pa_xi_cc,
        c_Xd_XL_cc, c_Xd_XR_cc, c_Xd_PL_cc, c_Xd_PR_cc, c_Xd_xi_cc,
        c_Pd_XL_cc, c_Pd_XR_cc, c_Pd_PL_cc, c_Pd_PR_cc, c_Pd_xi_cc
    ) = coefs_cc

    S_Xa = 1/2*((c_Xa_XL*c_Xa_XL_cc) + (c_Xa_PL*c_Xa_PL_cc) + 1j*((c_Xa_XL*c_Xa_PL_cc) - (c_Xa_XL_cc*c_Xa_PL))
                + (c_Xa_XR*c_Xa_XR_cc) + (c_Xa_PR*c_Xa_PR_cc) + 1j*((c_Xa_XR*c_Xa_PR_cc) - (c_Xa_XR_cc*c_Xa_PR))).real + (2*Nth + 1)*(c_Xa_xi*c_Xa_xi_cc).real
    S_Pa = 1/2*((c_Pa_XL*c_Pa_XL_cc) + (c_Pa_PL*c_Pa_PL_cc) + 1j*((c_Pa_XL*c_Pa_PL_cc) - (c_Pa_XL_cc*c_Pa_PL))
                + (c_Pa_XR*c_Pa_XR_cc) + (c_Pa_PR*c_Pa_PR_cc) + 1j*((c_Pa_XR*c_Pa_PR_cc) - (c_Pa_XR_cc*c_Pa_PR))).real + (2*Nth + 1)*(c_Pa_xi*c_Pa_xi_cc).real
    S_Xd = 1/2*((c_Xd_XL*c_Xd_XL_cc) + (c_Xd_PL*c_Xd_PL_cc) + 1j*((c_Xd_XL*c_Xd_PL_cc) - (c_Xd_XL_cc*c_Xd_PL))
                + (c_Xd_XR*c_Xd_XR_cc) + (c_Xd_PR*c_Xd_PR_cc) + 1j*((c_Xd_XR*c_Xd_PR_cc) - (c_Xd_XR_cc*c_Xd_PR))).real + (2*Nth + 1)*(c_Xd_xi*c_Xd_xi_cc).real
    S_Pd = 1/2*((c_Pd_XL*c_Pd_XL_cc) + (c_Pd_PL*c_Pd_PL_cc) + 1j*((c_Pd_XL*c_Pd_PL_cc) - (c_Pd_XL_cc*c_Pd_PL))
                + (c_Pd_XR*c_Pd_XR_cc) + (c_Pd_PR*c_Pd_PR_cc) + 1j*((c_Pd_XR*c_Pd_PR_cc) - (c_Pd_XR_cc*c_Pd_PR))).real + (2*Nth + 1)*(c_Pd_xi*c_Pd_xi_cc).real
    S_q = 1/2*((c_q_XL*c_q_XL_cc) + (c_q_PL*c_q_PL_cc) + 1j*((c_q_XL*c_q_PL_cc) - (c_q_XL_cc*c_q_PL))
                + (c_q_XR*c_q_XR_cc) + (c_q_PR*c_q_PR_cc) + 1j*((c_q_XR*c_q_PR_cc) - (c_q_XR_cc*c_q_PR))).real + (2*Nth + 1)*(c_q_xi*c_q_xi_cc).real

    S_XaXd = 1/4*((c_Xa_XL*c_Xd_XL_cc) + (c_Xa_PL*c_Xd_PL_cc) + 1j*((c_Xa_XL*c_Xd_PL_cc) - (c_Xa_XL_cc*c_Xd_PL))
                + (c_Xa_XR*c_Xd_XR_cc) + (c_Xa_PR*c_Xd_PR_cc) + 1j*((c_Xa_XR*c_Xd_PR_cc) - (c_Xa_XR_cc*c_Xd_PR))
                + (c_Xd_XL*c_Xa_XL_cc) + (c_Xd_PL*c_Xa_PL_cc) + 1j*((c_Xd_XL*c_Xa_PL_cc) - (c_Xd_XL_cc*c_Xa_PL))
                + (c_Xd_XR*c_Xa_XR_cc) + (c_Xd_PR*c_Xa_PR_cc) + 1j*((c_Xd_XR*c_Xa_PR_cc) - (c_Xd_XR_cc*c_Xa_PR))).real \
        + (2*Nth + 1)*((c_Xa_xi*c_Xd_xi_cc) + (c_Xd_xi*c_Xa_xi_cc)).real/2
    S_Xaq = 1/4*((c_Xa_XL*c_q_XL_cc) + (c_Xa_PL*c_q_PL_cc) + 1j*((c_Xa_XL*c_q_PL_cc) - (c_Xa_XL_cc*c_q_PL))
                + (c_Xa_XR*c_q_XR_cc) + (c_Xa_PR*c_q_PR_cc) + 1j*((c_Xa_XR*c_q_PR_cc) - (c_Xa_XR_cc*c_q_PR))
                + (c_q_XL*c_Xa_XL_cc) + (c_q_PL*c_Xa_PL_cc) + 1j*((c_q_XL*c_Xa_PL_cc) - (c_q_XL_cc*c_Xa_PL))
                + (c_q_XR*c_Xa_XR_cc) + (c_q_PR*c_Xa_PR_cc) + 1j*((c_q_XR*c_Xa_PR_cc) - (c_q_XR_cc*c_Xa_PR))).real \
        + (2*Nth + 1)*((c_Xa_xi*c_q_xi_cc) + (c_q_xi*c_Xa_xi_cc)).real/2
    S_Xdq = 1/4*((c_Xd_XL*c_q_XL_cc) + (c_Xd_PL*c_q_PL_cc) + 1j*((c_Xd_XL*c_q_PL_cc) - (c_Xd_XL_cc*c_q_PL))
                + (c_Xd_XR*c_q_XR_cc) + (c_Xd_PR*c_q_PR_cc) + 1j*((c_Xd_XR*c_q_PR_cc) - (c_Xd_XR_cc*c_q_PR))
                + (c_q_XL*c_Xd_XL_cc) + (c_q_PL*c_Xd_PL_cc) + 1j*((c_q_XL*c_Xd_PL_cc) - (c_q_XL_cc*c_Xd_PL))
                + (c_q_XR*c_Xd_XR_cc) + (c_q_PR*c_Xd_PR_cc) + 1j*((c_q_XR*c_Xd_PR_cc) - (c_q_XR_cc*c_Xd_PR))).real \
        + (2*Nth + 1)*((c_Xd_xi*c_q_xi_cc) + (c_q_xi*c_Xd_xi_cc)).real/2

    S_PaPd = 1/4*((c_Pa_XL*c_Pd_XL_cc) + (c_Pa_PL*c_Pd_PL_cc) + 1j*((c_Pa_XL*c_Pd_PL_cc) - (c_Pa_XL_cc*c_Pd_PL))
                + (c_Pa_XR*c_Pd_XR_cc) + (c_Pa_PR*c_Pd_PR_cc) + 1j*((c_Pa_XR*c_Pd_PR_cc) - (c_Pa_XR_cc*c_Pd_PR))
                + (c_Pd_XL*c_Pa_XL_cc) + (c_Pd_PL*c_Pa_PL_cc) + 1j*((c_Pd_XL*c_Pa_PL_cc) - (c_Pd_XL_cc*c_Pa_PL))
                + (c_Pd_XR*c_Pa_XR_cc) + (c_Pd_PR*c_Pa_PR_cc) + 1j*((c_Pd_XR*c_Pa_PR_cc) - (c_Pd_XR_cc*c_Pa_PR))).real \
        + (2*Nth + 1)*((c_Pa_xi*c_Pd_xi_cc) + (c_Pd_xi*c_Pa_xi_cc)).real/2
    S_Paq = 1/4*((c_Pa_XL*c_q_XL_cc) + (c_Pa_PL*c_q_PL_cc) + 1j*((c_Pa_XL*c_q_PL_cc) - (c_Pa_XL_cc*c_q_PL))
                + (c_Pa_XR*c_q_XR_cc) + (c_Pa_PR*c_q_PR_cc) + 1j*((c_Pa_XR*c_q_PR_cc) - (c_Pa_XR_cc*c_q_PR))
                + (c_q_XL*c_Pa_XL_cc) + (c_q_PL*c_Pa_PL_cc) + 1j*((c_q_XL*c_Pa_PL_cc) - (c_q_XL_cc*c_Pa_PL))
                + (c_q_XR*c_Pa_XR_cc) + (c_q_PR*c_Pa_PR_cc) + 1j*((c_q_XR*c_Pa_PR_cc) - (c_q_XR_cc*c_Pa_PR))).real \
        + (2*Nth + 1)*((c_Pa_xi*c_q_xi_cc) + (c_q_xi*c_Pa_xi_cc)).real/2
    S_Pdq = 1/4*((c_Pd_XL*c_q_XL_cc) + (c_Pd_PL*c_q_PL_cc) + 1j*((c_Pd_XL*c_q_PL_cc) - (c_Pd_XL_cc*c_q_PL))
                + (c_Pd_XR*c_q_XR_cc) + (c_Pd_PR*c_q_PR_cc) + 1j*((c_Pd_XR*c_q_PR_cc) - c_Pd_XR_cc*c_q_PR)
                + (c_q_XL*c_Pd_XL_cc) + (c_q_PL*c_Pd_PL_cc) + 1j*((c_q_XL*c_Pd_PL_cc) - (c_q_XL_cc*c_Pd_PL))
                + (c_q_XR*c_Pd_XR_cc) + (c_q_PR*c_Pd_PR_cc) + 1j*((c_q_XR*c_Pd_PR_cc) - (c_q_XR_cc*c_Pd_PR))).real \
        + (2*Nth + 1)*((c_Pd_xi*c_q_xi_cc) + (c_q_xi*c_Pd_xi_cc)).real/2


    S_PoutL = 2*S_Pa*ka + 4*S_PaPd*sqrt(ka*kd) + 4*S_Paq*cX*sqrt(ka) + 2*S_Pd*kd + 4*S_Pdq*cX*sqrt(kd) + 2*S_q*cX**2 + 0.5
    S_XoutL = 2*S_Xa*ka + 4*S_XaXd*sqrt(ka*kd) + 4*S_Xaq*cP*sqrt(ka) + 2*S_Xd*kd + 4*S_Xdq*cP*sqrt(kd) + 2*S_q*cP**2 + 0.5
    return S_q, S_XoutL, S_PoutL


def S_outR(Nth, Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, w, wL, wa, wd):
    """Noise power spectrum of the quadratures of the mechanical position S_q and of the right output light field S_XoutR, S_PoutR."""
    coefs = spectum_coefs(Nth, Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, w, wL, wa, wd)
    coefs_cc = (coef.conjugate() for coef in coefs)

    (
        c_q_XL, c_q_XR, c_q_PL, c_q_PR, c_q_xi,
        c_Xa_XL, c_Xa_XR, c_Xa_PL, c_Xa_PR, c_Xa_xi,
        c_Pa_XL, c_Pa_XR, c_Pa_PL, c_Pa_PR, c_Pa_xi,
        c_Xd_XL, c_Xd_XR, c_Xd_PL, c_Xd_PR, c_Xd_xi,
        c_Pd_XL, c_Pd_XR, c_Pd_PL, c_Pd_PR, c_Pd_xi
    ) = coefs
    (
        c_q_XL_cc, c_q_XR_cc, c_q_PL_cc, c_q_PR_cc, c_q_xi_cc,
        c_Xa_XL_cc, c_Xa_XR_cc, c_Xa_PL_cc, c_Xa_PR_cc, c_Xa_xi_cc,
        c_Pa_XL_cc, c_Pa_XR_cc, c_Pa_PL_cc, c_Pa_PR_cc, c_Pa_xi_cc,
        c_Xd_XL_cc, c_Xd_XR_cc, c_Xd_PL_cc, c_Xd_PR_cc, c_Xd_xi_cc,
        c_Pd_XL_cc, c_Pd_XR_cc, c_Pd_PL_cc, c_Pd_PR_cc, c_Pd_xi_cc
    ) = coefs_cc

    S_Xa = 1/2*((c_Xa_XL*c_Xa_XL_cc) + (c_Xa_PL*c_Xa_PL_cc) + 1j*((c_Xa_XL*c_Xa_PL_cc) - (c_Xa_XL_cc*c_Xa_PL))
                + (c_Xa_XR*c_Xa_XR_cc) + (c_Xa_PR*c_Xa_PR_cc) + 1j*((c_Xa_XR*c_Xa_PR_cc) - (c_Xa_XR_cc*c_Xa_PR))).real + (2*Nth + 1)*(c_Xa_xi*c_Xa_xi_cc).real
    S_Pa = 1/2*((c_Pa_XL*c_Pa_XL_cc) + (c_Pa_PL*c_Pa_PL_cc) + 1j*((c_Pa_XL*c_Pa_PL_cc) - (c_Pa_XL_cc*c_Pa_PL))
                + (c_Pa_XR*c_Pa_XR_cc) + (c_Pa_PR*c_Pa_PR_cc) + 1j*((c_Pa_XR*c_Pa_PR_cc) - (c_Pa_XR_cc*c_Pa_PR))).real + (2*Nth + 1)*(c_Pa_xi*c_Pa_xi_cc).real
    S_q = 1/2*((c_q_XL*c_q_XL_cc) + (c_q_PL*c_q_PL_cc) + 1j*((c_q_XL*c_q_PL_cc) - (c_q_XL_cc*c_q_PL))
                + (c_q_XR*c_q_XR_cc) + (c_q_PR*c_q_PR_cc) + 1j*((c_q_XR*c_q_PR_cc) - (c_q_XR_cc*c_q_PR))).real + (2*Nth + 1)*(c_q_xi*c_q_xi_cc).real

    S_PoutR = 2*S_Pa*ga + 0.5
    S_XoutR = 2*S_Xa*ga + 0.5
    return S_q, S_XoutR, S_PoutR


def spectrum_coefs_approx(Nth, Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, w, wL, wa, wd):
    """
    Return the coefficients needed to calculate the various output spectra (auxiliary function).

    Approximation in the regime where the mechanical thermal noise dominates.
    """
    Da = -wL + wa
    Dd = -wL + wd
    cG = sqrt(ka)*sqrt(kd) + 1j*lbd
    a_las = sqrt(Plas/h/wL)*exp(1j*thlas)
    Dw = -cG**2 + (1j*Dd + kd - 1j*w)*(1j*Da + ga + ka - 1j*w)
    Dmw = -cG**2 + (1j*Dd + kd + 1j*w)*(1j*Da + ga + ka + 1j*w)

    ass = -sqrt(2)*1j*sqrt(Plas/(h*wL))*(sqrt(ka)*(wL - wd) + sqrt(kd)*lbd)*exp(1j*thlas)/(ga*kd - 1j*ga*(wL - wd) - 2*1j*sqrt(ka)*sqrt(kd)*lbd - 1j*ka*(wL - wd) - 1j*kd*(wL - wa) + lbd**2 - (wL - wa)*(wL - wd))
    dss = sqrt(2)*sqrt(Plas/(h*wL))*(ga*sqrt(kd) - 1j*sqrt(ka)*lbd - 1j*sqrt(kd)*(wL - wa))*exp(1j*thlas)/(ga*kd - 1j*ga*(wL - wd) - 2*1j*sqrt(ka)*sqrt(kd)*lbd - 1j*ka*(wL - wd) - 1j*kd*(wL - wa) + lbd**2 - (wL - wa)*(wL - wd))
    qss = 2*sqrt(2)*Plas*ka*kd*(-ga**2*gkd0*sqrt(ka)*sqrt(kd)*wL + ga**2*gkd0*sqrt(ka)*sqrt(kd)*wd + ga**2*gwd0*sqrt(ka)*kd**(3/2) + gka0*sqrt(ka)*sqrt(kd)*lbd**2*wL - gka0*sqrt(ka)*sqrt(kd)*lbd**2*wd - gka0*sqrt(ka)*sqrt(kd)*wL**3 + gka0*sqrt(ka)*sqrt(kd)*wL**2*wa + 2*gka0*sqrt(ka)*sqrt(kd)*wL**2*wd - 2*gka0*sqrt(ka)*sqrt(kd)*wL*wa*wd - gka0*sqrt(ka)*sqrt(kd)*wL*wd**2 + gka0*sqrt(ka)*sqrt(kd)*wa*wd**2 + gka0*kd*lbd**3 - gka0*kd*lbd*wL**2 + gka0*kd*lbd*wL*wa + gka0*kd*lbd*wL*wd - gka0*kd*lbd*wa*wd + gkd0*sqrt(ka)*sqrt(kd)*lbd**2*wL - gkd0*sqrt(ka)*sqrt(kd)*lbd**2*wa - gkd0*sqrt(ka)*sqrt(kd)*wL**3 + 2*gkd0*sqrt(ka)*sqrt(kd)*wL**2*wa + gkd0*sqrt(ka)*sqrt(kd)*wL**2*wd - gkd0*sqrt(ka)*sqrt(kd)*wL*wa**2 - 2*gkd0*sqrt(ka)*sqrt(kd)*wL*wa*wd + gkd0*sqrt(ka)*sqrt(kd)*wa**2*wd + gkd0*ka*lbd**3 - gkd0*ka*lbd*wL**2 + gkd0*ka*lbd*wL*wa + gkd0*ka*lbd*wL*wd - gkd0*ka*lbd*wa*wd + gwa0*ka**(3/2)*sqrt(kd)*wL**2 - 2*gwa0*ka**(3/2)*sqrt(kd)*wL*wd + gwa0*ka**(3/2)*sqrt(kd)*wd**2 + gwa0*sqrt(ka)*kd**(3/2)*lbd**2 + 2*gwa0*ka*kd*lbd*wL - 2*gwa0*ka*kd*lbd*wd + gwd0*ka**(3/2)*sqrt(kd)*lbd**2 + gwd0*sqrt(ka)*kd**(3/2)*wL**2 - 2*gwd0*sqrt(ka)*kd**(3/2)*wL*wa + gwd0*sqrt(ka)*kd**(3/2)*wa**2 + 2*gwd0*ka*kd*lbd*wL - 2*gwd0*ka*kd*lbd*wa)/(Om*ga**2*h*ka**(3/2)*kd**(7/2)*wL + Om*ga**2*h*ka**(3/2)*kd**(3/2)*wL**3 - 2*Om*ga**2*h*ka**(3/2)*kd**(3/2)*wL**2*wd + Om*ga**2*h*ka**(3/2)*kd**(3/2)*wL*wd**2 + 2*Om*ga*h*ka**(5/2)*kd**(3/2)*wL**3 - 4*Om*ga*h*ka**(5/2)*kd**(3/2)*wL**2*wd + 2*Om*ga*h*ka**(5/2)*kd**(3/2)*wL*wd**2 + 2*Om*ga*h*ka**(3/2)*kd**(5/2)*lbd**2*wL + 4*Om*ga*h*ka**2*kd**2*lbd*wL**2 - 4*Om*ga*h*ka**2*kd**2*lbd*wL*wd + Om*h*ka**(7/2)*kd**(3/2)*wL**3 - 2*Om*h*ka**(7/2)*kd**(3/2)*wL**2*wd + Om*h*ka**(7/2)*kd**(3/2)*wL*wd**2 + 4*Om*h*ka**(5/2)*kd**(5/2)*lbd**2*wL + 2*Om*h*ka**(5/2)*kd**(5/2)*wL**3 - 2*Om*h*ka**(5/2)*kd**(5/2)*wL**2*wa - 2*Om*h*ka**(5/2)*kd**(5/2)*wL**2*wd + 2*Om*h*ka**(5/2)*kd**(5/2)*wL*wa*wd + Om*h*ka**(3/2)*kd**(7/2)*wL**3 - 2*Om*h*ka**(3/2)*kd**(7/2)*wL**2*wa + Om*h*ka**(3/2)*kd**(7/2)*wL*wa**2 + Om*h*ka**(3/2)*kd**(3/2)*lbd**4*wL - 2*Om*h*ka**(3/2)*kd**(3/2)*lbd**2*wL**3 + 2*Om*h*ka**(3/2)*kd**(3/2)*lbd**2*wL**2*wa + 2*Om*h*ka**(3/2)*kd**(3/2)*lbd**2*wL**2*wd - 2*Om*h*ka**(3/2)*kd**(3/2)*lbd**2*wL*wa*wd + Om*h*ka**(3/2)*kd**(3/2)*wL**5 - 2*Om*h*ka**(3/2)*kd**(3/2)*wL**4*wa - 2*Om*h*ka**(3/2)*kd**(3/2)*wL**4*wd + Om*h*ka**(3/2)*kd**(3/2)*wL**3*wa**2 + 4*Om*h*ka**(3/2)*kd**(3/2)*wL**3*wa*wd + Om*h*ka**(3/2)*kd**(3/2)*wL**3*wd**2 - 2*Om*h*ka**(3/2)*kd**(3/2)*wL**2*wa**2*wd - 2*Om*h*ka**(3/2)*kd**(3/2)*wL**2*wa*wd**2 + Om*h*ka**(3/2)*kd**(3/2)*wL*wa**2*wd**2 + 4*Om*h*ka**3*kd**2*lbd*wL**2 - 4*Om*h*ka**3*kd**2*lbd*wL*wd + 4*Om*h*ka**2*kd**3*lbd*wL**2 - 4*Om*h*ka**2*kd**3*lbd*wL*wa + 2*Plas*ga**2*gkd0**2*ka**(3/2)*sqrt(kd)*wL - 2*Plas*ga**2*gkd0**2*ka**(3/2)*sqrt(kd)*wd - 2*Plas*gka0**2*sqrt(ka)*kd**(3/2)*lbd**2*wL + 2*Plas*gka0**2*sqrt(ka)*kd**(3/2)*lbd**2*wd + 2*Plas*gka0**2*sqrt(ka)*kd**(3/2)*wL**3 - 2*Plas*gka0**2*sqrt(ka)*kd**(3/2)*wL**2*wa - 4*Plas*gka0**2*sqrt(ka)*kd**(3/2)*wL**2*wd + 4*Plas*gka0**2*sqrt(ka)*kd**(3/2)*wL*wa*wd + 2*Plas*gka0**2*sqrt(ka)*kd**(3/2)*wL*wd**2 - 2*Plas*gka0**2*sqrt(ka)*kd**(3/2)*wa*wd**2 - 2*Plas*gka0**2*kd**2*lbd**3 + 2*Plas*gka0**2*kd**2*lbd*wL**2 - 2*Plas*gka0**2*kd**2*lbd*wL*wa - 2*Plas*gka0**2*kd**2*lbd*wL*wd + 2*Plas*gka0**2*kd**2*lbd*wa*wd - 2*Plas*gkd0**2*ka**(3/2)*sqrt(kd)*lbd**2*wL + 2*Plas*gkd0**2*ka**(3/2)*sqrt(kd)*lbd**2*wa + 2*Plas*gkd0**2*ka**(3/2)*sqrt(kd)*wL**3 - 4*Plas*gkd0**2*ka**(3/2)*sqrt(kd)*wL**2*wa - 2*Plas*gkd0**2*ka**(3/2)*sqrt(kd)*wL**2*wd + 2*Plas*gkd0**2*ka**(3/2)*sqrt(kd)*wL*wa**2 + 4*Plas*gkd0**2*ka**(3/2)*sqrt(kd)*wL*wa*wd - 2*Plas*gkd0**2*ka**(3/2)*sqrt(kd)*wa**2*wd - 2*Plas*gkd0**2*ka**2*lbd**3 + 2*Plas*gkd0**2*ka**2*lbd*wL**2 - 2*Plas*gkd0**2*ka**2*lbd*wL*wa - 2*Plas*gkd0**2*ka**2*lbd*wL*wd + 2*Plas*gkd0**2*ka**2*lbd*wa*wd)

    ka0 = -sqrt(2)*gka0*qss + ka
    kd0 = -sqrt(2)*gkd0*qss + kd
    gadm = sqrt(ka0)*sqrt(kd0)*(gka0/ka0 - gkd0/kd0)/2
    gadp = sqrt(ka0)*sqrt(kd0)*(gka0/ka0 + gkd0/kd0)/2

    tgma = -sqrt(2)*1j*a_las*gka0/(2*sqrt(ka0)) + ass*gwa0 + 1j*dss*gadm
    tgmd = -sqrt(2)*1j*a_las*gkd0/(2*sqrt(kd0)) - 1j*ass*gadm + dss*gwd0
    tga = ass*gwa0 + 1j*dss*gadp + 1j*gka0*(-sqrt(2)*a_las/sqrt(ka0) + 2*ass)/2
    tgd = 1j*ass*gadp + dss*gwd0 + 1j*gkd0*(-sqrt(2)*a_las/sqrt(kd0) + 2*dss)/2

    tgma_cc = tgma.conjugate()
    tgmd_cc = tgmd.conjugate()

    cX = -(gka0*sqrt(kd0)*im(ass) + gkd0*sqrt(ka0)*im(dss))/(sqrt(ka0)*sqrt(kd0))
    cP = (gka0*sqrt(kd0)*re(ass) + gkd0*sqrt(ka0)*re(dss))/(sqrt(ka0)*sqrt(kd0))

    chi_a_inv = ka + ga + 1j*(Da-w)
    chi_d_inv = kd + 1j*(Dd-w)
    chi_a_inv_m = ka + ga + 1j*(Da+w)
    chi_d_inv_m = kd + 1j*(Dd+w)

    Ca_q = 1j*(chi_d_inv*tga - cG*tgd)/Dw
    Cd_q = 1j*(chi_a_inv*tgd - cG*tga)/Dw

    Ca_qm_cc = (1j*(chi_d_inv_m*tga - cG*tgd)/Dmw).conjugate()
    Cd_qm_cc = (1j*(chi_a_inv_m*tgd - cG*tga)/Dmw).conjugate()

    Xopti = -2*(tgma_cc*Ca_q + tgma*Ca_qm_cc) -2*(tgmd_cc*Cd_q + tgmd*Cd_qm_cc)
    Xmeff = 1/(Xopti + (Om**2 - 1j*gam*w - w**2)/Om)

    # q
    c_q_xi = Xmeff*sqrt(gam)

    # X_a
    c_Xa_xi = ((Ca_q + Ca_qm_cc)*c_q_xi)

    # X_d
    c_Xd_xi = ((Cd_q + Cd_qm_cc)*c_q_xi)

    # P_a
    c_Pa_xi = (-1j*(Ca_q - Ca_qm_cc)*c_q_xi)

    # P_d
    c_Pd_xi = (-1j*(Cd_q - Cd_qm_cc)*c_q_xi)

    return (c_q_xi, c_Xa_xi, c_Pa_xi, c_Xd_xi, c_Pd_xi), cX, cP


def S_outL_approx(Nth, Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, w, wL, wa, wd):
    """
    Noise power spectrum of the quadratures of the mechanical position S_q and of the left output light field S_XoutL, S_PoutL.

    Approximation in the regime where the mechanical thermal noise dominates.
    """
    coefs, cX, cP = spectrum_coefs_approx(Nth, Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, w, wL, wa, wd)
    (c_q_xi, c_Xa_xi, c_Pa_xi, c_Xd_xi, c_Pd_xi) = coefs
    (c_q_xi_cc, c_Xa_xi_cc, c_Pa_xi_cc, c_Xd_xi_cc, c_Pd_xi_cc) = (coef.conjugate() for coef in coefs)

    S_Xa = (2*Nth + 1)*(c_Xa_xi*c_Xa_xi_cc).real
    S_Pa = (2*Nth + 1)*(c_Pa_xi*c_Pa_xi_cc).real
    S_Xd = (2*Nth + 1)*(c_Xd_xi*c_Xd_xi_cc).real
    S_Pd = (2*Nth + 1)*(c_Pd_xi*c_Pd_xi_cc).real
    S_q = (2*Nth + 1)*(c_q_xi*c_q_xi_cc).real

    S_XaXd = (2*Nth + 1)*((c_Xa_xi*c_Xd_xi_cc) + (c_Xd_xi*c_Xa_xi_cc)).real/2
    S_Xaq =  (2*Nth + 1)*((c_Xa_xi*c_q_xi_cc) + (c_q_xi*c_Xa_xi_cc)).real/2
    S_Xdq = (2*Nth + 1)*((c_Xd_xi*c_q_xi_cc) + (c_q_xi*c_Xd_xi_cc)).real/2

    S_PaPd = (2*Nth + 1)*((c_Pa_xi*c_Pd_xi_cc) + (c_Pd_xi*c_Pa_xi_cc)).real/2
    S_Paq = (2*Nth + 1)*((c_Pa_xi*c_q_xi_cc) + (c_q_xi*c_Pa_xi_cc)).real/2
    S_Pdq = (2*Nth + 1)*((c_Pd_xi*c_q_xi_cc) + (c_q_xi*c_Pd_xi_cc)).real/2

    S_PoutL = 2*S_Pa*ka + 4*S_PaPd*sqrt(ka*kd) + 4*S_Paq*cX*sqrt(ka) + 2*S_Pd*kd + 4*S_Pdq*cX*sqrt(kd) + 2*S_q*cX**2 + 0.5
    S_XoutL = 2*S_Xa*ka + 4*S_XaXd*sqrt(ka*kd) + 4*S_Xaq*cP*sqrt(ka) + 2*S_Xd*kd + 4*S_Xdq*cP*sqrt(kd) + 2*S_q*cP**2 + 0.5
    return S_q, S_XoutL, S_PoutL


def S_outR_approx(Nth, Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, w, wL, wa, wd):
    """
    Noise power spectrum of the quadratures of the mechanical position S_q and of the left output light field S_XoutR, S_PoutR.

    Approximation in the regime where the mechanical thermal noise dominates.
    """
    coefs, cX, cP = spectrum_coefs_approx(Nth, Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, w, wL, wa, wd)
    (c_q_xi, c_Xa_xi, c_Pa_xi, c_Xd_xi, c_Pd_xi) = coefs
    (c_q_xi_cc, c_Xa_xi_cc, c_Pa_xi_cc, c_Xd_xi_cc, c_Pd_xi_cc) = (coef.conjugate() for coef in coefs)

    S_Xa = (2*Nth + 1)*(c_Xa_xi*c_Xa_xi_cc).real
    S_Pa = (2*Nth + 1)*(c_Pa_xi*c_Pa_xi_cc).real
    S_q = (2*Nth + 1)*(c_q_xi*c_q_xi_cc).real

    S_PoutR = 2*S_Pa*ga + 0.5
    S_XoutR = 2*S_Xa*ga + 0.5
    return S_q, S_XoutR, S_PoutR


def D_out(Nth, Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, w, wL, wa, wd):
    """
    Return prefactor D in the noise power spectrum of the quadratures of the output light field S_XoutL, S_PoutL, S_XoutR, S_PoutR.

     Approximation in the regime where the mechanical thermal noise dominates.
    """
    coefs, cX, cP = spectrum_coefs_approx(Nth, Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, w, wL, wa, wd)
    (c_q_xi, c_Xa_xi, c_Pa_xi, c_Xd_xi, c_Pd_xi) = coefs
    (c_q_xi_cc, c_Xa_xi_cc, c_Pa_xi_cc, c_Xd_xi_cc, c_Pd_xi_cc) = (coef.conjugate() for coef in coefs)

    D_Xa = (c_Xa_xi*c_Xa_xi_cc).real
    D_Pa = (c_Pa_xi*c_Pa_xi_cc).real

    D_XoutL = 2*abs(cP + sqrt(ka)*c_Xa_xi + sqrt(kd)*c_Xd_xi)**2
    D_PoutL = 2*abs(cX + sqrt(ka)*c_Pa_xi + sqrt(kd)*c_Pd_xi)**2

    D_PoutR = 2*D_Pa*ga
    D_XoutR = 2*D_Xa*ga
    return D_XoutL, D_PoutL, D_XoutR, D_PoutR


# -------------------------- weak-coupling limit ------------------------------

def A_w(Om, Plas, ga, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, w, wL, wa, wd):
    """Auxiliary function to calculate the Stokes/anti-Stokes rates."""
    Da = -wL + wa
    Dd = -wL + wd
    cG = sqrt(ka)*sqrt(kd) + 1j*lbd
    a_las = sqrt(Plas/h/wL)*exp(1j*thlas)
    ass = -sqrt(2)*1j*sqrt(Plas/(h*wL))*(sqrt(ka)*(wL - wd) + sqrt(kd)*lbd)*exp(1j*thlas)/(ga*kd - 1j*ga*(wL - wd) - 2*1j*sqrt(ka)*sqrt(kd)*lbd - 1j*ka*(wL - wd) - 1j*kd*(wL - wa) + lbd**2 - (wL - wa)*(wL - wd))
    dss = sqrt(2)*sqrt(Plas/(h*wL))*(ga*sqrt(kd) - 1j*sqrt(ka)*lbd - 1j*sqrt(kd)*(wL - wa))*exp(1j*thlas)/(ga*kd - 1j*ga*(wL - wd) - 2*1j*sqrt(ka)*sqrt(kd)*lbd - 1j*ka*(wL - wd) - 1j*kd*(wL - wa) + lbd**2 - (wL - wa)*(wL - wd))
    qss = q_ss(Om, Plas, ga, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, wL, wa, wd)
    ka0 = -sqrt(2)*gka0*qss + ka
    kd0 = -sqrt(2)*gkd0*qss + kd
    gadm = sqrt(ka0)*sqrt(kd0)*(gka0/ka0 - gkd0/kd0)/2
    tgma = -sqrt(2)*1j*a_las*gka0/(2*sqrt(ka0)) + ass*gwa0 + 1j*dss*gadm
    tgmd = -sqrt(2)*1j*a_las*gkd0/(2*sqrt(kd0)) - 1j*ass*gadm + dss*gwd0
    tDp = -1j*ga/2 - 1j*ka/2 - 1j*kd/2 - wL + wa/2 + wd/2 + sqrt(-4*(sqrt(ka)*sqrt(kd) + 1j*lbd)**2 + (-1j*ga - 1j*ka + 1j*kd + wa - wd)**2)/2
    tDm = -1j*ga/2 - 1j*ka/2 - 1j*kd/2 - wL + wa/2 + wd/2 - sqrt(-4*(sqrt(ka)*sqrt(kd) + 1j*lbd)**2 + (-1j*ga - 1j*ka + 1j*kd + wa - wd)**2)/2
    Dm = re(tDm)
    Dp = re(tDp)
    km = -im(tDm)
    kp = -im(tDp)
    denom_CL = Dm**4*Dp**4*ka0*kd0 + 2*Dm**2*Dp**4*ka0*kd0*km**2 + Dp**4*ka0*kd0*km**4 + kp**4*(Dm**4*ka0*kd0 + 2*Dm**2*ka0*kd0*km**2 + ka0*kd0*km**4) + 2*kp**2*(Dm**4*Dp**2*ka0*kd0 + 2*Dm**2*Dp**2*ka0*kd0*km**2 + Dp**2*ka0*kd0*km**4) + w**4*(Dm**2*Dp**2*ka0*kd0 + Dp**2*ka0*kd0*km**2 + kp**2*(Dm**2*ka0*kd0 + ka0*kd0*km**2)) - 2*w**3*(ka0*kd0*km**2*(Dm*Dp**2 + Dp**3) + ka0*kd0*(Dm**3*Dp**2 + Dm**2*Dp**3) + kp**2*(ka0*kd0*km**2*(Dm + Dp) + ka0*kd0*(Dm**3 + Dm**2*Dp))) + w**2*(Dp**2*ka0*kd0*km**4 + ka0*kd0*km**2*(2*Dm**2*Dp**2 + 4*Dm*Dp**3 + Dp**4) + ka0*kd0*(Dm**4*Dp**2 + 4*Dm**3*Dp**3 + Dm**2*Dp**4) + kp**4*(Dm**2*ka0*kd0 + ka0*kd0*km**2) + kp**2*(ka0*kd0*km**4 + 2*ka0*kd0*km**2*(Dm**2 + 2*Dm*Dp + Dp**2) + ka0*kd0*(Dm**4 + 4*Dm**3*Dp + 2*Dm**2*Dp**2))) - 2*w*(Dp**3*ka0*kd0*km**4 + ka0*kd0*km**2*(2*Dm**2*Dp**3 + Dm*Dp**4) + ka0*kd0*(Dm**4*Dp**3 + Dm**3*Dp**4) + kp**4*(Dm**3*ka0*kd0 + Dm*ka0*kd0*km**2) + kp**2*(Dp*ka0*kd0*km**4 + 2*ka0*kd0*km**2*(Dm**2*Dp + Dm*Dp**2) + ka0*kd0*(Dm**4*Dp + 2*Dm**3*Dp**2)))
    denom_CR = Dm**2*Dp**2 - 2*Dm**2*Dp*w + Dm**2*kp**2 + Dm**2*w**2 - 2*Dm*Dp**2*w + 4*Dm*Dp*w**2 - 2*Dm*kp**2*w - 2*Dm*w**3 + Dp**2*km**2 + Dp**2*w**2 - 2*Dp*km**2*w - 2*Dp*w**3 + km**2*kp**2 + km**2*w**2 + kp**2*w**2 + w**4
    return (4*ga*(Dd**2*re(tgma)**2 + Dd**2*im(tgma)**2 + 2*Dd*sqrt(ka)*sqrt(kd)*re(tgma)*im(tgmd) - 2*Dd*sqrt(ka)*sqrt(kd)*re(tgmd)*im(tgma) - 2*Dd*lbd*re(tgma)*re(tgmd) - 2*Dd*lbd*im(tgma)*im(tgmd) - 2*Dd*w*re(tgma)**2 - 2*Dd*w*im(tgma)**2 - 2*sqrt(ka)*kd**(3/2)*re(tgma)*re(tgmd) - 2*sqrt(ka)*kd**(3/2)*im(tgma)*im(tgmd) - 2*sqrt(ka)*sqrt(kd)*w*re(tgma)*im(tgmd) + 2*sqrt(ka)*sqrt(kd)*w*re(tgmd)*im(tgma) + ka*kd*re(tgmd)**2 + ka*kd*im(tgmd)**2 + kd**2*re(tgma)**2 + kd**2*im(tgma)**2 - 2*kd*lbd*re(tgma)*im(tgmd) + 2*kd*lbd*re(tgmd)*im(tgma) + lbd**2*re(tgmd)**2 + lbd**2*im(tgmd)**2 + 2*lbd*w*re(tgma)*re(tgmd) + 2*lbd*w*im(tgma)*im(tgmd) + w**2*re(tgma)**2 + w**2*im(tgma)**2)/denom_CR - 2*(-1j*Da*a_las*gkd0*sqrt(ka0)*sqrt(kd)*w**2 + 1j*Da*a_las*gkd0*sqrt(ka0)*sqrt(kd)*w*(Dm + 1j*km) + 1j*Da*a_las*gkd0*sqrt(ka0)*sqrt(kd)*w*(Dp + 1j*kp) - 1j*Da*a_las*gkd0*sqrt(ka0)*sqrt(kd)*(Dm + 1j*km)*(Dp + 1j*kp) + sqrt(2)*1j*Da*sqrt(ka0)*sqrt(kd)*sqrt(kd0)*tgmd*(Dm + 1j*km)*(Dp + 1j*kp) - 1j*Dd*a_las*gka0*sqrt(ka)*sqrt(kd0)*w**2 + 1j*Dd*a_las*gka0*sqrt(ka)*sqrt(kd0)*w*(Dm + 1j*km) + 1j*Dd*a_las*gka0*sqrt(ka)*sqrt(kd0)*w*(Dp + 1j*kp) - 1j*Dd*a_las*gka0*sqrt(ka)*sqrt(kd0)*(Dm + 1j*km)*(Dp + 1j*kp) + sqrt(2)*1j*Dd*sqrt(ka)*sqrt(ka0)*sqrt(kd0)*tgma*(Dm + 1j*km)*(Dp + 1j*kp) + a_las*ga*gkd0*sqrt(ka0)*sqrt(kd)*w**2 - a_las*ga*gkd0*sqrt(ka0)*sqrt(kd)*w*(Dm + 1j*km) - a_las*ga*gkd0*sqrt(ka0)*sqrt(kd)*w*(Dp + 1j*kp) + a_las*ga*gkd0*sqrt(ka0)*sqrt(kd)*(Dm + 1j*km)*(Dp + 1j*kp) + a_las*gka0*sqrt(ka)*kd*sqrt(kd0)*w**2 - a_las*gka0*sqrt(ka)*kd*sqrt(kd0)*w*(Dm + 1j*km) - a_las*gka0*sqrt(ka)*kd*sqrt(kd0)*w*(Dp + 1j*kp) + a_las*gka0*sqrt(ka)*kd*sqrt(kd0)*(Dm + 1j*km)*(Dp + 1j*kp) - a_las*gka0*sqrt(kd)*sqrt(kd0)*w**2*conjugate(cG) + a_las*gka0*sqrt(kd)*sqrt(kd0)*w*(Dm + 1j*km)*conjugate(cG) + a_las*gka0*sqrt(kd)*sqrt(kd0)*w*(Dp + 1j*kp)*conjugate(cG) - a_las*gka0*sqrt(kd)*sqrt(kd0)*(Dm + 1j*km)*(Dp + 1j*kp)*conjugate(cG) - a_las*gkd0*sqrt(ka)*sqrt(ka0)*w**2*conjugate(cG) + a_las*gkd0*sqrt(ka)*sqrt(ka0)*w*(Dm + 1j*km)*conjugate(cG) + a_las*gkd0*sqrt(ka)*sqrt(ka0)*w*(Dp + 1j*kp)*conjugate(cG) - a_las*gkd0*sqrt(ka)*sqrt(ka0)*(Dm + 1j*km)*(Dp + 1j*kp)*conjugate(cG) + a_las*gkd0*ka*sqrt(ka0)*sqrt(kd)*w**2 - a_las*gkd0*ka*sqrt(ka0)*sqrt(kd)*w*(Dm + 1j*km) - a_las*gkd0*ka*sqrt(ka0)*sqrt(kd)*w*(Dp + 1j*kp) + a_las*gkd0*ka*sqrt(ka0)*sqrt(kd)*(Dm + 1j*km)*(Dp + 1j*kp) - sqrt(2)*ga*sqrt(ka0)*sqrt(kd)*sqrt(kd0)*tgmd*(Dm + 1j*km)*(Dp + 1j*kp) - sqrt(2)*sqrt(ka)*sqrt(ka0)*kd*sqrt(kd0)*tgma*(Dm + 1j*km)*(Dp + 1j*kp) - sqrt(2)*1j*sqrt(ka)*sqrt(ka0)*sqrt(kd0)*tgma*w*(Dm + 1j*km)*(Dp + 1j*kp) + sqrt(2)*sqrt(ka)*sqrt(ka0)*sqrt(kd0)*tgmd*(Dm + 1j*km)*(Dp + 1j*kp)*conjugate(cG) - sqrt(2)*ka*sqrt(ka0)*sqrt(kd)*sqrt(kd0)*tgmd*(Dm + 1j*km)*(Dp + 1j*kp) + sqrt(2)*sqrt(ka0)*sqrt(kd)*sqrt(kd0)*tgma*(Dm + 1j*km)*(Dp + 1j*kp)*conjugate(cG) - sqrt(2)*1j*sqrt(ka0)*sqrt(kd)*sqrt(kd0)*tgmd*w*(Dm + 1j*km)*(Dp + 1j*kp))*(-1j*Da*gkd0*sqrt(ka0)*sqrt(kd)*w**2*conjugate(a_las) + 1j*Da*gkd0*sqrt(ka0)*sqrt(kd)*w*(Dm - 1j*km)*conjugate(a_las) + 1j*Da*gkd0*sqrt(ka0)*sqrt(kd)*w*(Dp - 1j*kp)*conjugate(a_las) - 1j*Da*gkd0*sqrt(ka0)*sqrt(kd)*(Dm - 1j*km)*(Dp - 1j*kp)*conjugate(a_las) + sqrt(2)*1j*Da*sqrt(ka0)*sqrt(kd)*sqrt(kd0)*(Dm - 1j*km)*(Dp - 1j*kp)*conjugate(tgmd) - 1j*Dd*gka0*sqrt(ka)*sqrt(kd0)*w**2*conjugate(a_las) + 1j*Dd*gka0*sqrt(ka)*sqrt(kd0)*w*(Dm - 1j*km)*conjugate(a_las) + 1j*Dd*gka0*sqrt(ka)*sqrt(kd0)*w*(Dp - 1j*kp)*conjugate(a_las) - 1j*Dd*gka0*sqrt(ka)*sqrt(kd0)*(Dm - 1j*km)*(Dp - 1j*kp)*conjugate(a_las) + sqrt(2)*1j*Dd*sqrt(ka)*sqrt(ka0)*sqrt(kd0)*(Dm - 1j*km)*(Dp - 1j*kp)*conjugate(tgma) + cG*gka0*sqrt(kd)*sqrt(kd0)*w**2*conjugate(a_las) - cG*gka0*sqrt(kd)*sqrt(kd0)*w*(Dm - 1j*km)*conjugate(a_las) - cG*gka0*sqrt(kd)*sqrt(kd0)*w*(Dp - 1j*kp)*conjugate(a_las) + cG*gka0*sqrt(kd)*sqrt(kd0)*(Dm - 1j*km)*(Dp - 1j*kp)*conjugate(a_las) + cG*gkd0*sqrt(ka)*sqrt(ka0)*w**2*conjugate(a_las) - cG*gkd0*sqrt(ka)*sqrt(ka0)*w*(Dm - 1j*km)*conjugate(a_las) - cG*gkd0*sqrt(ka)*sqrt(ka0)*w*(Dp - 1j*kp)*conjugate(a_las) + cG*gkd0*sqrt(ka)*sqrt(ka0)*(Dm - 1j*km)*(Dp - 1j*kp)*conjugate(a_las) - sqrt(2)*cG*sqrt(ka)*sqrt(ka0)*sqrt(kd0)*(Dm - 1j*km)*(Dp - 1j*kp)*conjugate(tgmd) - sqrt(2)*cG*sqrt(ka0)*sqrt(kd)*sqrt(kd0)*(Dm - 1j*km)*(Dp - 1j*kp)*conjugate(tgma) - ga*gkd0*sqrt(ka0)*sqrt(kd)*w**2*conjugate(a_las) + ga*gkd0*sqrt(ka0)*sqrt(kd)*w*(Dm - 1j*km)*conjugate(a_las) + ga*gkd0*sqrt(ka0)*sqrt(kd)*w*(Dp - 1j*kp)*conjugate(a_las) - ga*gkd0*sqrt(ka0)*sqrt(kd)*(Dm - 1j*km)*(Dp - 1j*kp)*conjugate(a_las) + sqrt(2)*ga*sqrt(ka0)*sqrt(kd)*sqrt(kd0)*(Dm - 1j*km)*(Dp - 1j*kp)*conjugate(tgmd) - gka0*sqrt(ka)*kd*sqrt(kd0)*w**2*conjugate(a_las) + gka0*sqrt(ka)*kd*sqrt(kd0)*w*(Dm - 1j*km)*conjugate(a_las) + gka0*sqrt(ka)*kd*sqrt(kd0)*w*(Dp - 1j*kp)*conjugate(a_las) - gka0*sqrt(ka)*kd*sqrt(kd0)*(Dm - 1j*km)*(Dp - 1j*kp)*conjugate(a_las) - gkd0*ka*sqrt(ka0)*sqrt(kd)*w**2*conjugate(a_las) + gkd0*ka*sqrt(ka0)*sqrt(kd)*w*(Dm - 1j*km)*conjugate(a_las) + gkd0*ka*sqrt(ka0)*sqrt(kd)*w*(Dp - 1j*kp)*conjugate(a_las) - gkd0*ka*sqrt(ka0)*sqrt(kd)*(Dm - 1j*km)*(Dp - 1j*kp)*conjugate(a_las) + sqrt(2)*sqrt(ka)*sqrt(ka0)*kd*sqrt(kd0)*(Dm - 1j*km)*(Dp - 1j*kp)*conjugate(tgma) - sqrt(2)*1j*sqrt(ka)*sqrt(ka0)*sqrt(kd0)*w*(Dm - 1j*km)*(Dp - 1j*kp)*conjugate(tgma) + sqrt(2)*ka*sqrt(ka0)*sqrt(kd)*sqrt(kd0)*(Dm - 1j*km)*(Dp - 1j*kp)*conjugate(tgmd) - sqrt(2)*1j*sqrt(ka0)*sqrt(kd)*sqrt(kd0)*w*(Dm - 1j*km)*(Dp - 1j*kp)*conjugate(tgmd))/denom_CL).real/2


def A_p(Om, Plas, ga, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, wL, wa, wd):
    """Stokes rate A_+."""
    return A_w(Om, Plas, ga, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, -Om, wL, wa, wd)

def A_m(Om, Plas, ga, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, wL, wa, wd):
    """Anti-Stokes rate A_-."""
    return A_w(Om, Plas, ga, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, Om, wL, wa, wd)


# -------------------------- Lyapunov equations  ------------------------------

def A_Lyapunov(Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, wL, wa, wd):
    """A matrix in Lyapunov equation dV/dt = AV + VA^T + B."""
    Da = -wL + wa
    Dd = -wL + wd
    ass = -sqrt(2)*1j*sqrt(Plas/(h*wL))*(sqrt(ka)*(wL - wd) + sqrt(kd)*lbd)*exp(1j*thlas)/(ga*kd - 1j*ga*(wL - wd) - 2*1j*sqrt(ka)*sqrt(kd)*lbd - 1j*ka*(wL - wd) - 1j*kd*(wL - wa) + lbd**2 - (wL - wa)*(wL - wd))
    dss = sqrt(2)*sqrt(Plas/(h*wL))*(ga*sqrt(kd) - 1j*sqrt(ka)*lbd - 1j*sqrt(kd)*(wL - wa))*exp(1j*thlas)/(ga*kd - 1j*ga*(wL - wd) - 2*1j*sqrt(ka)*sqrt(kd)*lbd - 1j*ka*(wL - wd) - 1j*kd*(wL - wa) + lbd**2 - (wL - wa)*(wL - wd))
    qss = q_ss(Om, Plas, ga, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, wL, wa, wd)
    ka0 = -sqrt(2)*gka0*qss + ka
    kd0 = -sqrt(2)*gkd0*qss + kd
    a_las = sqrt(Plas/(h*wL))*exp(1j*thlas)
    gadm = sqrt(ka0)*sqrt(kd0)*(gka0/ka0 - gkd0/kd0)/2
    gadp = sqrt(ka0)*sqrt(kd0)*(gka0/ka0 + gkd0/kd0)/2
    return np.array([[-ga - ka, Da, -sqrt(ka)*sqrt(kd), lbd, -2*gadp*re(dss) - 2*gka0*re(ass) + sqrt(2)*gka0*re(a_las)/sqrt(ka0) - 2*gwa0*im(ass), 0], [-Da, -ga - ka, -lbd, -sqrt(ka)*sqrt(kd), -2*gadp*im(dss) - 2*gka0*im(ass) + sqrt(2)*gka0*im(a_las)/sqrt(ka0) + 2*gwa0*re(ass), 0], [-sqrt(ka)*sqrt(kd), lbd, -kd, Dd, -2*gadp*re(ass) - 2*gkd0*re(dss) + sqrt(2)*gkd0*re(a_las)/sqrt(kd0) - 2*gwd0*im(dss), 0], [-lbd, -sqrt(ka)*sqrt(kd), -Dd, -kd, -2*gadp*im(ass) - 2*gkd0*im(dss) + sqrt(2)*gkd0*im(a_las)/sqrt(kd0) + 2*gwd0*re(dss), 0], [0, 0, 0, 0, 0, Om], [-2*gadm*im(dss) + sqrt(2)*gka0*im(a_las)/sqrt(ka0) + 2*gwa0*re(ass), 2*gadm*re(dss) - sqrt(2)*gka0*re(a_las)/sqrt(ka0) + 2*gwa0*im(ass), 2*gadm*im(ass) + sqrt(2)*gkd0*im(a_las)/sqrt(kd0) + 2*gwd0*re(dss), -2*gadm*re(ass) - sqrt(2)*gkd0*re(a_las)/sqrt(kd0) + 2*gwd0*im(dss), -Om, -gam]])


def B_Lyapunov(Nth, Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, wL, wa, wd):
    """B matrix in Lyapunov equation dV/dt = AV + VA^T + B"""
    ass = -sqrt(2)*1j*sqrt(Plas/(h*wL))*(sqrt(ka)*(wL - wd) + sqrt(kd)*lbd)*exp(1j*thlas)/(ga*kd - 1j*ga*(wL - wd) - 2*1j*sqrt(ka)*sqrt(kd)*lbd - 1j*ka*(wL - wd) - 1j*kd*(wL - wa) + lbd**2 - (wL - wa)*(wL - wd))
    dss = sqrt(2)*sqrt(Plas/(h*wL))*(ga*sqrt(kd) - 1j*sqrt(ka)*lbd - 1j*sqrt(kd)*(wL - wa))*exp(1j*thlas)/(ga*kd - 1j*ga*(wL - wd) - 2*1j*sqrt(ka)*sqrt(kd)*lbd - 1j*ka*(wL - wd) - 1j*kd*(wL - wa) + lbd**2 - (wL - wa)*(wL - wd))
    qss = q_ss(Om, Plas, ga, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, wL, wa, wd)
    ka0 = -sqrt(2)*gka0*qss + ka
    kd0 = -sqrt(2)*gkd0*qss + kd
    return np.array([[ga + ka, 0, sqrt(ka)*sqrt(kd), 0, 0, -sqrt(ka)*(gka0*sqrt(kd0)*im(ass) + gkd0*sqrt(ka0)*im(dss))/(sqrt(ka0)*sqrt(kd0))], [0, ga + ka, 0, sqrt(ka)*sqrt(kd), 0, sqrt(ka)*(gka0*sqrt(kd0)*re(ass) + gkd0*sqrt(ka0)*re(dss))/(sqrt(ka0)*sqrt(kd0))], [sqrt(ka)*sqrt(kd), 0, kd, 0, 0, -sqrt(kd)*(gka0*sqrt(kd0)*im(ass) + gkd0*sqrt(ka0)*im(dss))/(sqrt(ka0)*sqrt(kd0))], [0, sqrt(ka)*sqrt(kd), 0, kd, 0, sqrt(kd)*(gka0*sqrt(kd0)*re(ass) + gkd0*sqrt(ka0)*re(dss))/(sqrt(ka0)*sqrt(kd0))], [0, 0, 0, 0, 0, 0], [-sqrt(ka)*(gka0*sqrt(kd0)*im(ass) + gkd0*sqrt(ka0)*im(dss))/(sqrt(ka0)*sqrt(kd0)), sqrt(ka)*(gka0*sqrt(kd0)*re(ass) + gkd0*sqrt(ka0)*re(dss))/(sqrt(ka0)*sqrt(kd0)), -sqrt(kd)*(gka0*sqrt(kd0)*im(ass) + gkd0*sqrt(ka0)*im(dss))/(sqrt(ka0)*sqrt(kd0)), sqrt(kd)*(gka0*sqrt(kd0)*re(ass) + gkd0*sqrt(ka0)*re(dss))/(sqrt(ka0)*sqrt(kd0)), 0, gam*(2*Nth + 1) + (gka0*sqrt(kd0)*re(ass) + gkd0*sqrt(ka0)*re(dss))**2/(ka0*kd0) + (gka0*sqrt(kd0)*im(ass) + gkd0*sqrt(ka0)*im(dss))**2/(ka0*kd0)]])


def solve_Lyapunov(Nth, Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, wL, wa, wd):
    '''
    Solve the steady-state Lyapunov equation using scipy.linalg.solve_continuous_lyapunov

    Return the elements of the covariance matrix V11, V12, V13, V14, V15, V16, V22, V23, V24, V25, V26, V33, V34, V35, V36, V44, V45, V46, V55, V56, V66
    '''
    Da = -wL + wa
    Dd = -wL + wd
    ass = -sqrt(2)*1j*sqrt(Plas/(h*wL))*(sqrt(ka)*(wL - wd) + sqrt(kd)*lbd)*exp(1j*thlas)/(ga*kd - 1j*ga*(wL - wd) - 2*1j*sqrt(ka)*sqrt(kd)*lbd - 1j*ka*(wL - wd) - 1j*kd*(wL - wa) + lbd**2 - (wL - wa)*(wL - wd))
    dss = sqrt(2)*sqrt(Plas/(h*wL))*(ga*sqrt(kd) - 1j*sqrt(ka)*lbd - 1j*sqrt(kd)*(wL - wa))*exp(1j*thlas)/(ga*kd - 1j*ga*(wL - wd) - 2*1j*sqrt(ka)*sqrt(kd)*lbd - 1j*ka*(wL - wd) - 1j*kd*(wL - wa) + lbd**2 - (wL - wa)*(wL - wd))
    qss = q_ss(Om, Plas, ga, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, wL, wa, wd)
    ka0 = -sqrt(2)*gka0*qss + ka
    kd0 = -sqrt(2)*gkd0*qss + kd
    a_las = sqrt(Plas/(h*wL))*exp(1j*thlas)
    gadm = sqrt(ka0)*sqrt(kd0)*(gka0/ka0 - gkd0/kd0)/2
    gadp = sqrt(ka0)*sqrt(kd0)*(gka0/ka0 + gkd0/kd0)/2
    A = np.array([[-ga - ka, Da, -sqrt(ka)*sqrt(kd), lbd, -2*gadp*re(dss) - 2*gka0*re(ass) + sqrt(2)*gka0*re(a_las)/sqrt(ka0) - 2*gwa0*im(ass), 0], [-Da, -ga - ka, -lbd, -sqrt(ka)*sqrt(kd), -2*gadp*im(dss) - 2*gka0*im(ass) + sqrt(2)*gka0*im(a_las)/sqrt(ka0) + 2*gwa0*re(ass), 0], [-sqrt(ka)*sqrt(kd), lbd, -kd, Dd, -2*gadp*re(ass) - 2*gkd0*re(dss) + sqrt(2)*gkd0*re(a_las)/sqrt(kd0) - 2*gwd0*im(dss), 0], [-lbd, -sqrt(ka)*sqrt(kd), -Dd, -kd, -2*gadp*im(ass) - 2*gkd0*im(dss) + sqrt(2)*gkd0*im(a_las)/sqrt(kd0) + 2*gwd0*re(dss), 0], [0, 0, 0, 0, 0, Om], [-2*gadm*im(dss) + sqrt(2)*gka0*im(a_las)/sqrt(ka0) + 2*gwa0*re(ass), 2*gadm*re(dss) - sqrt(2)*gka0*re(a_las)/sqrt(ka0) + 2*gwa0*im(ass), 2*gadm*im(ass) + sqrt(2)*gkd0*im(a_las)/sqrt(kd0) + 2*gwd0*re(dss), -2*gadm*re(ass) - sqrt(2)*gkd0*re(a_las)/sqrt(kd0) + 2*gwd0*im(dss), -Om, -gam]])
    B = np.array([[ga + ka, 0, sqrt(ka)*sqrt(kd), 0, 0, -sqrt(ka)*(gka0*sqrt(kd0)*im(ass) + gkd0*sqrt(ka0)*im(dss))/(sqrt(ka0)*sqrt(kd0))], [0, ga + ka, 0, sqrt(ka)*sqrt(kd), 0, sqrt(ka)*(gka0*sqrt(kd0)*re(ass) + gkd0*sqrt(ka0)*re(dss))/(sqrt(ka0)*sqrt(kd0))], [sqrt(ka)*sqrt(kd), 0, kd, 0, 0, -sqrt(kd)*(gka0*sqrt(kd0)*im(ass) + gkd0*sqrt(ka0)*im(dss))/(sqrt(ka0)*sqrt(kd0))], [0, sqrt(ka)*sqrt(kd), 0, kd, 0, sqrt(kd)*(gka0*sqrt(kd0)*re(ass) + gkd0*sqrt(ka0)*re(dss))/(sqrt(ka0)*sqrt(kd0))], [0, 0, 0, 0, 0, 0], [-sqrt(ka)*(gka0*sqrt(kd0)*im(ass) + gkd0*sqrt(ka0)*im(dss))/(sqrt(ka0)*sqrt(kd0)), sqrt(ka)*(gka0*sqrt(kd0)*re(ass) + gkd0*sqrt(ka0)*re(dss))/(sqrt(ka0)*sqrt(kd0)), -sqrt(kd)*(gka0*sqrt(kd0)*im(ass) + gkd0*sqrt(ka0)*im(dss))/(sqrt(ka0)*sqrt(kd0)), sqrt(kd)*(gka0*sqrt(kd0)*re(ass) + gkd0*sqrt(ka0)*re(dss))/(sqrt(ka0)*sqrt(kd0)), 0, gam*(2*Nth + 1) + (gka0*sqrt(kd0)*re(ass) + gkd0*sqrt(ka0)*re(dss))**2/(ka0*kd0) + (gka0*sqrt(kd0)*im(ass) + gkd0*sqrt(ka0)*im(dss))**2/(ka0*kd0)]])
    return scilin.solve_continuous_lyapunov(A, -B)


def solve_Lyapunov_linsys(Nth, Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, wL, wa, wd):
    '''
    Solve the steady-state Lyapunov equation recast as a linear system of equations using numpy.linalg.solve .

    Return the elements of the covariance matrix V11, V12, V13, V14, V15, V16, V22, V23, V24, V25, V26, V33, V34, V35, V36, V44, V45, V46, V55, V56, V66
    '''
    Da = -wL + wa
    Dd = -wL + wd
    ass = -sqrt(2)*1j*sqrt(Plas/(h*wL))*(sqrt(ka)*(wL - wd) + sqrt(kd)*lbd)*exp(1j*thlas)/(ga*kd - 1j*ga*(wL - wd) - 2*1j*sqrt(ka)*sqrt(kd)*lbd - 1j*ka*(wL - wd) - 1j*kd*(wL - wa) + lbd**2 - (wL - wa)*(wL - wd))
    dss = sqrt(2)*sqrt(Plas/(h*wL))*(ga*sqrt(kd) - 1j*sqrt(ka)*lbd - 1j*sqrt(kd)*(wL - wa))*exp(1j*thlas)/(ga*kd - 1j*ga*(wL - wd) - 2*1j*sqrt(ka)*sqrt(kd)*lbd - 1j*ka*(wL - wd) - 1j*kd*(wL - wa) + lbd**2 - (wL - wa)*(wL - wd))
    qss = q_ss(Om, Plas, ga, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, wL, wa, wd)
    ka0 = -sqrt(2)*gka0*qss + ka
    kd0 = -sqrt(2)*gkd0*qss + kd
    a_las = sqrt(Plas/(h*wL))*exp(1j*thlas)
    gadm = sqrt(ka0)*sqrt(kd0)*(gka0/ka0 - gkd0/kd0)/2
    gadp = sqrt(ka0)*sqrt(kd0)*(gka0/ka0 + gkd0/kd0)/2
    A = np.array([[-2*ga - 2*ka, 2*Da, -2*sqrt(ka)*sqrt(kd), 2*lbd, -4*gadp*re(dss) - 4*gka0*re(ass) + 2*sqrt(2)*gka0*re(a_las)/sqrt(ka0) - 4*gwa0*im(ass), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [-Da, -2*ga - 2*ka, -lbd, -sqrt(ka)*sqrt(kd), -2*gadp*im(dss) - 2*gka0*im(ass) + sqrt(2)*gka0*im(a_las)/sqrt(ka0) + 2*gwa0*re(ass), 0, Da, -sqrt(ka)*sqrt(kd), lbd, -2*gadp*re(dss) - 2*gka0*re(ass) + sqrt(2)*gka0*re(a_las)/sqrt(ka0) - 2*gwa0*im(ass), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [-sqrt(ka)*sqrt(kd), lbd, -ga - ka - kd, Dd, -2*gadp*re(ass) - 2*gkd0*re(dss) + sqrt(2)*gkd0*re(a_las)/sqrt(kd0) - 2*gwd0*im(dss), 0, 0, Da, 0, 0, 0, -sqrt(ka)*sqrt(kd), lbd, -2*gadp*re(dss) - 2*gka0*re(ass) + sqrt(2)*gka0*re(a_las)/sqrt(ka0) - 2*gwa0*im(ass), 0, 0, 0, 0, 0, 0, 0], [-lbd, -sqrt(ka)*sqrt(kd), -Dd, -ga - ka - kd, -2*gadp*im(ass) - 2*gkd0*im(dss) + sqrt(2)*gkd0*im(a_las)/sqrt(kd0) + 2*gwd0*re(dss), 0, 0, 0, Da, 0, 0, 0, -sqrt(ka)*sqrt(kd), 0, 0, lbd, -2*gadp*re(dss) - 2*gka0*re(ass) + sqrt(2)*gka0*re(a_las)/sqrt(ka0) - 2*gwa0*im(ass), 0, 0, 0, 0], [0, 0, 0, 0, -ga - ka, Om, 0, 0, 0, Da, 0, 0, 0, -sqrt(ka)*sqrt(kd), 0, 0, lbd, 0, -2*gadp*re(dss) - 2*gka0*re(ass) + sqrt(2)*gka0*re(a_las)/sqrt(ka0) - 2*gwa0*im(ass), 0, 0], [-2*gadm*im(dss) + sqrt(2)*gka0*im(a_las)/sqrt(ka0) + 2*gwa0*re(ass), 2*gadm*re(dss) - sqrt(2)*gka0*re(a_las)/sqrt(ka0) + 2*gwa0*im(ass), 2*gadm*im(ass) + sqrt(2)*gkd0*im(a_las)/sqrt(kd0) + 2*gwd0*re(dss), -2*gadm*re(ass) - sqrt(2)*gkd0*re(a_las)/sqrt(kd0) + 2*gwd0*im(dss), -Om, -ga - gam - ka, 0, 0, 0, 0, Da, 0, 0, 0, -sqrt(ka)*sqrt(kd), 0, 0, lbd, 0, -2*gadp*re(dss) - 2*gka0*re(ass) + sqrt(2)*gka0*re(a_las)/sqrt(ka0) - 2*gwa0*im(ass), 0], [0, -2*Da, 0, 0, 0, 0, -2*ga - 2*ka, -2*lbd, -2*sqrt(ka)*sqrt(kd), -4*gadp*im(dss) - 4*gka0*im(ass) + 2*sqrt(2)*gka0*im(a_las)/sqrt(ka0) + 4*gwa0*re(ass), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, -sqrt(ka)*sqrt(kd), -Da, 0, 0, 0, lbd, -ga - ka - kd, Dd, -2*gadp*re(ass) - 2*gkd0*re(dss) + sqrt(2)*gkd0*re(a_las)/sqrt(kd0) - 2*gwd0*im(dss), 0, -lbd, -sqrt(ka)*sqrt(kd), -2*gadp*im(dss) - 2*gka0*im(ass) + sqrt(2)*gka0*im(a_las)/sqrt(ka0) + 2*gwa0*re(ass), 0, 0, 0, 0, 0, 0, 0], [0, -lbd, 0, -Da, 0, 0, -sqrt(ka)*sqrt(kd), -Dd, -ga - ka - kd, -2*gadp*im(ass) - 2*gkd0*im(dss) + sqrt(2)*gkd0*im(a_las)/sqrt(kd0) + 2*gwd0*re(dss), 0, 0, -lbd, 0, 0, -sqrt(ka)*sqrt(kd), -2*gadp*im(dss) - 2*gka0*im(ass) + sqrt(2)*gka0*im(a_las)/sqrt(ka0) + 2*gwa0*re(ass), 0, 0, 0, 0], [0, 0, 0, 0, -Da, 0, 0, 0, 0, -ga - ka, Om, 0, 0, -lbd, 0, 0, -sqrt(ka)*sqrt(kd), 0, -2*gadp*im(dss) - 2*gka0*im(ass) + sqrt(2)*gka0*im(a_las)/sqrt(ka0) + 2*gwa0*re(ass), 0, 0], [0, -2*gadm*im(dss) + sqrt(2)*gka0*im(a_las)/sqrt(ka0) + 2*gwa0*re(ass), 0, 0, 0, -Da, 2*gadm*re(dss) - sqrt(2)*gka0*re(a_las)/sqrt(ka0) + 2*gwa0*im(ass), 2*gadm*im(ass) + sqrt(2)*gkd0*im(a_las)/sqrt(kd0) + 2*gwd0*re(dss), -2*gadm*re(ass) - sqrt(2)*gkd0*re(a_las)/sqrt(kd0) + 2*gwd0*im(dss), -Om, -ga - gam - ka, 0, 0, 0, -lbd, 0, 0, -sqrt(ka)*sqrt(kd), 0, -2*gadp*im(dss) - 2*gka0*im(ass) + sqrt(2)*gka0*im(a_las)/sqrt(ka0) + 2*gwa0*re(ass), 0], [0, 0, -2*sqrt(ka)*sqrt(kd), 0, 0, 0, 0, 2*lbd, 0, 0, 0, -2*kd, 2*Dd, -4*gadp*re(ass) - 4*gkd0*re(dss) + 2*sqrt(2)*gkd0*re(a_las)/sqrt(kd0) - 4*gwd0*im(dss), 0, 0, 0, 0, 0, 0, 0], [0, 0, -lbd, -sqrt(ka)*sqrt(kd), 0, 0, 0, -sqrt(ka)*sqrt(kd), lbd, 0, 0, -Dd, -2*kd, -2*gadp*im(ass) - 2*gkd0*im(dss) + sqrt(2)*gkd0*im(a_las)/sqrt(kd0) + 2*gwd0*re(dss), 0, Dd, -2*gadp*re(ass) - 2*gkd0*re(dss) + sqrt(2)*gkd0*re(a_las)/sqrt(kd0) - 2*gwd0*im(dss), 0, 0, 0, 0], [0, 0, 0, 0, -sqrt(ka)*sqrt(kd), 0, 0, 0, 0, lbd, 0, 0, 0, -kd, Om, 0, Dd, 0, -2*gadp*re(ass) - 2*gkd0*re(dss) + sqrt(2)*gkd0*re(a_las)/sqrt(kd0) - 2*gwd0*im(dss), 0, 0], [0, 0, -2*gadm*im(dss) + sqrt(2)*gka0*im(a_las)/sqrt(ka0) + 2*gwa0*re(ass), 0, 0, -sqrt(ka)*sqrt(kd), 0, 2*gadm*re(dss) - sqrt(2)*gka0*re(a_las)/sqrt(ka0) + 2*gwa0*im(ass), 0, 0, lbd, 2*gadm*im(ass) + sqrt(2)*gkd0*im(a_las)/sqrt(kd0) + 2*gwd0*re(dss), -2*gadm*re(ass) - sqrt(2)*gkd0*re(a_las)/sqrt(kd0) + 2*gwd0*im(dss), -Om, -gam - kd, 0, 0, Dd, 0, -2*gadp*re(ass) - 2*gkd0*re(dss) + sqrt(2)*gkd0*re(a_las)/sqrt(kd0) - 2*gwd0*im(dss), 0], [0, 0, 0, -2*lbd, 0, 0, 0, 0, -2*sqrt(ka)*sqrt(kd), 0, 0, 0, -2*Dd, 0, 0, -2*kd, -4*gadp*im(ass) - 4*gkd0*im(dss) + 2*sqrt(2)*gkd0*im(a_las)/sqrt(kd0) + 4*gwd0*re(dss), 0, 0, 0, 0], [0, 0, 0, 0, -lbd, 0, 0, 0, 0, -sqrt(ka)*sqrt(kd), 0, 0, 0, -Dd, 0, 0, -kd, Om, -2*gadp*im(ass) - 2*gkd0*im(dss) + sqrt(2)*gkd0*im(a_las)/sqrt(kd0) + 2*gwd0*re(dss), 0, 0], [0, 0, 0, -2*gadm*im(dss) + sqrt(2)*gka0*im(a_las)/sqrt(ka0) + 2*gwa0*re(ass), 0, -lbd, 0, 0, 2*gadm*re(dss) - sqrt(2)*gka0*re(a_las)/sqrt(ka0) + 2*gwa0*im(ass), 0, -sqrt(ka)*sqrt(kd), 0, 2*gadm*im(ass) + sqrt(2)*gkd0*im(a_las)/sqrt(kd0) + 2*gwd0*re(dss), 0, -Dd, -2*gadm*re(ass) - sqrt(2)*gkd0*re(a_las)/sqrt(kd0) + 2*gwd0*im(dss), -Om, -gam - kd, 0, -2*gadp*im(ass) - 2*gkd0*im(dss) + sqrt(2)*gkd0*im(a_las)/sqrt(kd0) + 2*gwd0*re(dss), 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2*Om, 0], [0, 0, 0, 0, -2*gadm*im(dss) + sqrt(2)*gka0*im(a_las)/sqrt(ka0) + 2*gwa0*re(ass), 0, 0, 0, 0, 2*gadm*re(dss) - sqrt(2)*gka0*re(a_las)/sqrt(ka0) + 2*gwa0*im(ass), 0, 0, 0, 2*gadm*im(ass) + sqrt(2)*gkd0*im(a_las)/sqrt(kd0) + 2*gwd0*re(dss), 0, 0, -2*gadm*re(ass) - sqrt(2)*gkd0*re(a_las)/sqrt(kd0) + 2*gwd0*im(dss), 0, -Om, -gam, Om], [0, 0, 0, 0, 0, -4*gadm*im(dss) + 2*sqrt(2)*gka0*im(a_las)/sqrt(ka0) + 4*gwa0*re(ass), 0, 0, 0, 0, 4*gadm*re(dss) - 2*sqrt(2)*gka0*re(a_las)/sqrt(ka0) + 4*gwa0*im(ass), 0, 0, 0, 4*gadm*im(ass) + 2*sqrt(2)*gkd0*im(a_las)/sqrt(kd0) + 4*gwd0*re(dss), 0, 0, -4*gadm*re(ass) - 2*sqrt(2)*gkd0*re(a_las)/sqrt(kd0) + 4*gwd0*im(dss), 0, -2*Om, -2*gam]])
    B = np.array([[ga + ka], [0], [sqrt(ka)*sqrt(kd)], [0], [0], [-sqrt(ka)*(gka0*sqrt(kd0)*im(ass) + gkd0*sqrt(ka0)*im(dss))/(sqrt(ka0)*sqrt(kd0))], [ga + ka], [0], [sqrt(ka)*sqrt(kd)], [0], [sqrt(ka)*(gka0*sqrt(kd0)*re(ass) + gkd0*sqrt(ka0)*re(dss))/(sqrt(ka0)*sqrt(kd0))], [kd], [0], [0], [-sqrt(kd)*(gka0*sqrt(kd0)*im(ass) + gkd0*sqrt(ka0)*im(dss))/(sqrt(ka0)*sqrt(kd0))], [kd], [0], [sqrt(kd)*(gka0*sqrt(kd0)*re(ass) + gkd0*sqrt(ka0)*re(dss))/(sqrt(ka0)*sqrt(kd0))], [0], [0], [gam*(2*Nth + 1) + (gka0*sqrt(kd0)*re(ass) + gkd0*sqrt(ka0)*re(dss))**2/(ka0*kd0) + (gka0*sqrt(kd0)*im(ass) + gkd0*sqrt(ka0)*im(dss))**2/(ka0*kd0)]])
    return np.linalg.solve(A, -B)


def neff_Lyapunov_linsys(Nth, Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, wL, wa, wd):
    '''Photons and phonons numbers in the fluctuations (<\delta a^\dagger \delta a>, <\delta d^\dagger \delta d>, \bar{n}_\text{fin})'''
    (V11, V12, V13, V14, V15, V16, V22, V23, V24, V25, V26, V33, V34, V35, V36, V44, V45, V46, V55, V56, V66) = solve_Lyapunov_linsys(Nth, Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, wL, wa, wd)
    return (V11 + V22 - 1)/2, (V44 + V33 - 1)/2, (V55 + V66 - 1)/2


def neff_2d(Nth, Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, wL, wa, wd, threads=8):
    """
    Phonons numbers in the fluctuations.

    Parallel calculations of neff_Lyapunov_linsys()[2] for 2d arrays wL and Plas.
    """
    neff = np.zeros_like(wL)
    ps = []

    with Pool(threads) as pool:
        for i in range(neff.shape[0]):
            ps.append([])
            for j in range(neff.shape[1]):
                ps[i].append(pool.apply_async(neff_Lyapunov_linsys,
                                              (Nth, Om, Plas[i, j], ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, wL[i, j], wa, wd)))
        for i in range(neff.shape[0]):
            for j in range(neff.shape[1]):
                neff[i, j] = ps[i][j].get()[2][0]

    return neff

# ---------------------------- stability --------------------------------------

def RH_criterion(Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, wL, wa, wd):
    """
    Routh-Hurwitz stabilioty criterion.

    Return the ration T_k/T_{k-1} for k = 2, ... ,6
    as defined in E. X. DeJesus, C. Kaufman, Phys. Rev. A 35, 5288 (1987).
    """
    A = A_Lyapunov(Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, wL, wa, wd)
    poly = np.polynomial.Polynomial.fromroots(np.linalg.eigvals(A))
    a1, a2, a3, a4, a5, a6 =  poly.coef[-2::-1].real  # coefficients of the characteristic polynomial of matrix A
    r2 = (a1*a2 - a3)/a1
    r3 = (-a1**2*a4 + a1*a2*a3 + a1*a5 - a3**2)/(a1*a2 - a3)
    r4 = (a1**2*a2*a6 - a1**2*a4**2 - a1*a2**2*a5 + a1*a2*a3*a4 - a1*a3*a6 + 2*a1*a4*a5 + a2*a3*a5 - a3**2*a4 - a5**2)/(-a1**2*a4 + a1*a2*a3 + a1*a5 - a3**2)
    r5 = (-a1**3*a6**2 + 2*a1**2*a2*a5*a6 + a1**2*a3*a4*a6 - a1**2*a4**2*a5 - a1*a2**2*a5**2 - a1*a2*a3**2*a6 + a1*a2*a3*a4*a5 - 3*a1*a3*a5*a6 + 2*a1*a4*a5**2 + a2*a3*a5**2 + a3**3*a6 - a3**2*a4*a5 - a5**3)/(a1**2*a2*a6 - a1**2*a4**2 - a1*a2**2*a5 + a1*a2*a3*a4 - a1*a3*a6 + 2*a1*a4*a5 + a2*a3*a5 - a3**2*a4 - a5**2)
    return r2, r3, r4, r5, a6


def RH_criterion_wL(Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, wL, wa, wd):
    """Routh-Hurwitz stabilioty criterion for 1d array wL."""
    return np.array([RH_criterion(Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, w, wa, wd) for w in wL]).T


def RH_criterion_Plas(Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, wL, wa, wd):
    """Routh-Hurwitz stabilioty criterion for 1d array Plas."""
    return np.array([RH_criterion(Om, pl, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, wL, wa, wd) for pl in Plas]).T


def detA(Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, wL, wa, wd):
    """ Determinant of A_Lyapunov (r6)."""
    return np.linalg.det(A_Lyapunov(Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, wL, wa, wd))


def detA_2d(Om, Plas, ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, wL, wa, wd, threads=8):
    """
    Determinant of A_Lyapunov (r6).

    Parallel calculations for 2d arrays wL and Plas.
    """
    r6 = np.zeros_like(wL)
    ps = []

    with Pool(threads) as pool:
        for i in range(r6.shape[0]):
            ps.append([])
            for j in range(r6.shape[1]):
                ps[i].append(pool.apply_async(detA, (Om, Plas[i, j], ga, gam, gka0, gkd0, gwa0, gwd0, ka, kd, lbd, thlas, wL[i, j], wa, wd)))

        for i in range(r6.shape[0]):
            for j in range(r6.shape[1]):
                r6[i, j] = ps[i][j].get()

    return r6
