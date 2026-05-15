# -*- coding: utf-8 -*-
"""
fano_unified_optomechanics.py

A thin, run-ready wrapper for reproducing the one-Fano-mode figures exactly
when d2 is switched off, while using the two-Fano-mode module when d2 is active.

Required files in the same directory:
    onemodeoptomechanics.py
    twofanomodeoptomechanics.py

Design:
    - The notebook has one switch: USE_TWO_FANO_MODES.
    - If USE_TWO_FANO_MODES=False, the device parameters are exactly the
      one-mode parameters from onemodefigures.ipynb. The displayed two-mode
      table still contains d2 entries, but d2 is set exactly to zero.
    - If USE_TWO_FANO_MODES=True, the device parameters are the two-mode
      parameters used in your current two-mode figures.
    - The plotting API is the same for both cases.

Important:
    The exact one-mode gtilde plots are produced by the algebraically simplified
    one-mode functions in onemodeoptomechanics.py. This avoids the removable 0/0
    singularities that appear if one naively substitutes d2=0 in unsimplified
    two-mode rational expressions for steady states and gtilde. The parameter
    switch is still exactly d2=0 in the displayed unified parameter table.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

pi = np.pi
TWOPI = 2*np.pi
sqrt = np.sqrt
exp = np.exp
c = 299792458.0
h = 6.62606957e-34/(2*pi)
kB = 1.3806488e-23


def _import_required_module(name: str):
    try:
        return importlib.import_module(name)
    except Exception as exc:
        raise ImportError(
            f"Could not import {name}.py. Put {name}.py in the same folder "
            f"as this file or add its directory to sys.path. Original error: {exc}"
        )


# These are loaded lazily by load_modules() so users can set sys.path first.
om1 = None
om2 = None


def load_modules(reload: bool = True):
    """Import/reload the old one-mode and current two-mode physics modules."""
    global om1, om2
    om1 = _import_required_module("onemodeoptomechanics")
    om2 = _import_required_module("twofanomodeoptomechanics")
    if reload:
        om1 = importlib.reload(om1)
        om2 = importlib.reload(om2)
    return om1, om2


# =============================================================================
# Unit helpers
# =============================================================================

def ang_to_Hz(x):
    return np.asarray(x)/(2*pi)


def ang_to_MHz(x):
    return np.asarray(x)/(2*pi*1e6)


def ang_to_GHz(x):
    return np.asarray(x)/(2*pi*1e9)


def ang_to_THz(x):
    return np.asarray(x)/(2*pi*1e12)


def Hz_to_ang(x):
    return 2*pi*np.asarray(x)


def THz_to_ang(x):
    return 2*pi*np.asarray(x)*1e12


# =============================================================================
# Parameter builders
# =============================================================================

def one_mode_parameters() -> Dict[str, Dict[str, Any]]:
    """Exact parameter dictionaries from onemodefigures.ipynb for exp, std, 1, 2."""
    pars_exp = dict(
        kd = 2*pi*3.8e12,
        wd = 2*pi*c/1473.05e-9,
        Om = 2*pi*514*1e3,
        Plas = 150e-6,
        thlas = 0.0,
        Qm = 3e4,
        Tmec = 300.0,
        wa = 1278564183307649,
        ka = 13335433312204,
        ga = 2297341624968,
        lbd = 25717299776008,
        gka0 = 2*pi*775e3,
        gkd0 = 2*pi*3213e3,
        gwa0 = 2*pi*845e3,
        gwd0 = -2*pi*1819e3,
    )

    Om = 2*pi*1.3e6
    Qm = 1.4e8
    Tmec = 300.0
    sideband_resolution = 0.05
    Plas1 = 5e-5
    g0_reduce = 1e4

    pars_std = dict(
        kd = 1e-12,
        wd = 1.0,
        Om = Om,
        Plas = 3.5e-6*Plas1,
        thlas = 0.0,
        Qm = Qm,
        Tmec = Tmec,
        wa = pars_exp["wa"],
        ka = sideband_resolution*Om/2,
        ga = sideband_resolution*Om/2,
        lbd = 1e-12,
        gka0 = 0.0,
        gkd0 = 0.0,
        gwa0 = 2*pi*845e3/g0_reduce,
        gwd0 = 0.0,
    )

    # sys1 - zero detuning
    wa = pars_exp["wa"]
    wd = wa
    ka = pars_exp["ka"]
    km = sideband_resolution*Om
    ga = 2*km
    kd = ka + ga
    lbd = pars_exp["lbd"]

    pars1 = dict(
        kd = kd,
        wd = wd,
        Om = Om,
        Plas = Plas1,
        thlas = 0.0,
        Qm = Qm,
        Tmec = Tmec,
        wa = wa,
        ka = ka,
        ga = ga,
        lbd = lbd,
        gka0 = 2*pi*775e3/g0_reduce,
        gkd0 = 2*pi*3213e3/g0_reduce,
        gwa0 = 2*pi*845e3/g0_reduce,
        gwd0 = -2*pi*1819e3/g0_reduce,
    )

    # sys2 - optimized one-mode device
    ga = 2*pi*6e8
    ka = 2*pi*10e12
    wa = pars_exp["wa"]
    km = sideband_resolution*Om
    kd = km*(ka + ga)/(ga - 2*km)
    dk = (ga + ka - kd)/2
    lbd = pars_exp["lbd"]/10
    dD = dk*lbd/sqrt(kd*ka)
    wd = wa - 2*dD

    pars2 = dict(
        kd = kd,
        wd = wd,
        Om = Om,
        Plas = Plas1*0.00513,
        thlas = 0.0,
        Qm = Qm,
        Tmec = Tmec,
        wa = wa,
        ka = ka,
        ga = ga,
        lbd = lbd,
        gka0 = 2*pi*775e3/g0_reduce,
        gkd0 = 2*pi*3213e3*(kd/pars_exp["kd"]/g0_reduce),
        gwa0 = 2*pi*845e3/g0_reduce,
        gwd0 = -2*pi*1819e3/g0_reduce,
    )

    return {"exp": pars_exp, "std": pars_std, "1": pars1, "2": pars2}


def one_to_two_display_parameters(pars1m: Dict[str, Any]) -> Dict[str, Any]:
    """
    Embed exact one-mode parameters into a two-mode-looking dictionary with d2=0.
    This is used for display and for the bare 3x3 H check.
    """
    return dict(
        kd1=pars1m["kd"], wd1=pars1m["wd"],
        kd2=0.0, wd2=0.0,
        Om=pars1m["Om"], Plas=pars1m["Plas"], thlas=pars1m.get("thlas", 0.0),
        Qm=pars1m["Qm"], Tmec=pars1m["Tmec"],
        wa=pars1m["wa"], ka=pars1m["ka"], ga=pars1m["ga"],
        lbd1=pars1m["lbd"], lbd2=0.0,
        gka0=pars1m["gka0"], gkd10=pars1m["gkd0"], gkd20=0.0,
        gwa0=pars1m["gwa0"], gwd10=pars1m["gwd0"], gwd20=0.0,
    )


def two_mode_parameters() -> Dict[str, Dict[str, Any]]:
    """Two-mode parameter dictionaries from the current two-mode figures."""
    pars_exp = dict(
        kd1=2*pi*3.80e12, wd1=2*pi*2.04e14,
        kd2=2*pi*6.53e12, wd2=2*pi*2.16e14,
        Om=2*pi*5.14e5, Plas=1.50e-4, thlas=0.0, Qm=3.07e4, Tmec=3.12e2,
        wa=2*pi*2.03e14, ka=2*pi*2.12e12, ga=2*pi*3.70e11,
        lbd1=2*pi*4.09e12, lbd2=2*pi*9.34e12,
        gka0=2*pi*7.75e5, gkd10=2*pi*3.21e6, gkd20=2*pi*3.81e6,
        gwa0=2*pi*8.45e5, gwd10=-2*pi*1.82e6, gwd20=-2*pi*3.82e6,
    )

    Om = 2*pi*1.3e6
    Qm = 1.4e8
    Tmec = 3.0e2
    sideband_resolution = 5.0e-2
    Plas1 = 5.0e-5
    g0_reduce = 1.0e4

    pars_std_1m = dict(
        kd=1e-12, wd=1.0,
        Om=Om, Plas=3.5e-6*Plas1, thlas=0.0, Qm=Qm, Tmec=Tmec,
        wa=pars_exp["wa"], ka=sideband_resolution*Om/2, ga=sideband_resolution*Om/2,
        lbd=1e-12,
        gka0=0.0, gkd0=0.0, gwa0=2*pi*845e3/g0_reduce, gwd0=0.0,
    )

    pars1 = dict(
        kd1=2*pi*2.12e12, kd2=2*pi*4.12e12,
        wd1=2*pi*2.03e14, wd2=2*pi*2.05e14,
        Om=Om, Plas=Plas1, thlas=0.0, Qm=Qm, Tmec=Tmec,
        wa=2*pi*2.09e14, ka=2*pi*2.12e12, ga=2*pi*1.00e11,
        lbd1=2*pi*4.09e12, lbd2=2*pi*9.34e12,
        gka0=2*pi*7.75e1, gkd10=2*pi*3.21e2, gkd20=2*pi*3.21e1,
        gwa0=2*pi*8.45e1, gwd10=-2*pi*1.82e2, gwd20=-2*pi*4.82e2,
    )

    pars2 = dict(
        kd1=2*pi*9.00e10, kd2=2*pi*6.00e10,
        wd1=2*pi*2.03e14, wd2=2*pi*2.04e14,
        Om=Om, Plas=Plas1*5.13e-3, thlas=0.0, Qm=Qm, Tmec=Tmec,
        wa=2*pi*2.04e14, ka=2*pi*9.40e10, ga=2*pi*8.00e8,
        lbd1=2*pi*2.65e11, lbd2=2*pi*2.35e11,
        gka0=2*pi*7.75e1, gkd10=2*pi*7.61e0, gkd20=2*pi*6.02e0,
        gwa0=2*pi*8.45e1, gwd10=-2*pi*1.82e2, gwd20=-2*pi*1.12e2,
    )

    return {"exp": pars_exp, "std": pars_std_1m, "1": pars1, "2": pars2}


# =============================================================================
# System wrappers
# =============================================================================

class UnifiedSystem:
    """Unified wrapper with the old notebook's convenient sys.func(...) behavior."""
    def __init__(self, name: str, mode: str, pars: Dict[str, Any], w_scale: float = 1e-9):
        if om1 is None or om2 is None:
            load_modules(reload=False)
        self.name = name
        self.mode = mode  # "one" or "two"
        self._init_kwargs = pars.copy()
        self.w_scale = w_scale
        self.__dict__.update(pars)
        self.thlas = pars.get("thlas", 0.0)
        self.Nth = 1/(exp(h*self.Om/kB/self.Tmec) - 1)

        if mode == "one":
            for freq in ["wa", "wd", "ka", "kd", "ga", "Om", "lbd", "gka0", "gkd0", "gwa0", "gwd0"]:
                self.__dict__[freq] *= w_scale
            self.Plas *= w_scale**2
            self.gam = self.Om/self.Qm if "gam" not in pars else pars["gam"]*w_scale
            self.twp = om1.tw_p(self.ga, self.ka, self.kd, self.lbd, self.wa, self.wd)
            self.wp = self.twp.real
            self.kp = -self.twp.imag
            self.twm = om1.tw_m(self.ga, self.ka, self.kd, self.lbd, self.wa, self.wd)
            self.wm = self.twm.real
            self.km = -self.twm.imag
            if name == "std":
                self.wm = self.wp
                self.km = self.kp
            self.tw = np.array([self.twp, self.twm, 0.0+0.0j])
            self.active_curve_count = 4
        elif mode == "two":
            for freq in ["wa", "wd1", "wd2", "ka", "kd1", "kd2", "ga", "Om", "lbd1", "lbd2", "gka0", "gkd10", "gkd20", "gwa0", "gwd10", "gwd20"]:
                self.__dict__[freq] *= w_scale
            self.Plas *= w_scale**2
            self.gam = self.Om/self.Qm if "gam" not in pars else pars["gam"]*w_scale
            self.tw1 = om2.tw_1(self.ga, self.ka, self.kd1, self.kd2, self.lbd1, self.lbd2, self.wa, self.wd1, self.wd2)
            self.tw2 = om2.tw_2(self.ga, self.ka, self.kd1, self.kd2, self.lbd1, self.lbd2, self.wa, self.wd1, self.wd2)
            self.tw3 = om2.tw_3(self.ga, self.ka, self.kd1, self.kd2, self.lbd1, self.lbd2, self.wa, self.wd1, self.wd2)
            self.tw = np.array([self.tw1, self.tw2, self.tw3], dtype=complex)
            ks = -self.tw.imag
            j = int(np.argmin(ks))
            self.wm = self.tw[j].real
            self.km = -self.tw[j].imag
            self.lowest_mode_index = j + 1
            self.active_curve_count = 6
        else:
            raise ValueError("mode must be 'one' or 'two'")

    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]

    def get_pars(self):
        return self._init_kwargs.copy()

    def __getattr__(self, attr):
        module = om1 if self.mode == "one" else om2
        func = getattr(module, attr)
        args = inspect.getfullargspec(func).args
        def func2(**kw):
            for arg in args:
                if arg in self.__dict__:
                    kw.setdefault(arg, self.__dict__[arg])
            return func(**kw)
        return func2

    def omega_for_scan(self, x, scale="Omega"):
        if scale == "Omega":
            return self.wm - x*self.Om
        if scale == "kappa":
            return self.wm - x*self.km
        raise ValueError("scale must be 'Omega' or 'kappa'")


def build_systems(use_two_fano_modes: bool):
    """Return devices dict and parameter tables for either exact one-mode or two-mode case."""
    load_modules(reload=True)
    if use_two_fano_modes:
        pars = two_mode_parameters()
        devices = {
            "exp": UnifiedSystem("exp", "two", pars["exp"]),
            "1": UnifiedSystem("1", "two", pars["1"]),
            "2": UnifiedSystem("2", "two", pars["2"]),
            # standard reference remains the exact one-mode standard device
            "std": UnifiedSystem("std", "one", pars["std"]),
        }
        display_pars = {k: v for k, v in pars.items() if k in ["exp", "1", "2"]}
    else:
        pars = one_mode_parameters()
        devices = {
            "exp": UnifiedSystem("exp", "one", pars["exp"]),
            "1": UnifiedSystem("1", "one", pars["1"]),
            "2": UnifiedSystem("2", "one", pars["2"]),
            "std": UnifiedSystem("std", "one", pars["std"]),
        }
        display_pars = {k: one_to_two_display_parameters(v) for k, v in pars.items() if k in ["exp", "1", "2"]}
    return devices, display_pars, pars


def make_parameter_table(display_pars: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    quantities = [
        (r"$\omega_a/2\pi$ (THz)", "wa"),
        (r"$\gamma_a/2\pi$ (THz)", "ga"),
        (r"$\kappa_a/2\pi$ (THz)", "ka"),
        (r"$\omega_{d_1}/2\pi$ (THz)", "wd1"),
        (r"$\kappa_{d_1}/2\pi$ (THz)", "kd1"),
        (r"$\lambda_1/2\pi$ (THz)", "lbd1"),
        (r"$\omega_{d_2}/2\pi$ (THz)", "wd2"),
        (r"$\kappa_{d_2}/2\pi$ (THz)", "kd2"),
        (r"$\lambda_2/2\pi$ (THz)", "lbd2"),
        (r"$\Omega_\mathrm{mec}/2\pi$ (Hz)", "Om_Hz"),
        (r"$P_\mathrm{las}$ ($\mu$W)", "Plas_uW"),
    ]
    table = pd.DataFrame({"Quantity": [q for q, _ in quantities]})
    for dev, p in display_pars.items():
        vals = []
        for _, key in quantities:
            if key == "Om_Hz": vals.append(ang_to_Hz(p["Om"]))
            elif key == "Plas_uW": vals.append(p["Plas"]*1e6)
            else: vals.append(ang_to_THz(p[key]))
        table[dev] = vals
    return table


def make_mode_table(devices: Dict[str, UnifiedSystem]) -> pd.DataFrame:
    rows = []
    for dev in ["exp", "1", "2"]:
        sys = devices[dev]
        if sys.mode == "one":
            labels = [r"$\Omega_+$", r"$\Omega_-$", r"$0$"]
        else:
            labels = [r"$\Omega_1$", r"$\Omega_2$", r"$\Omega_3$"]
        for i, z in enumerate(sys.tw):
            rows.append({"Device": dev, "Mode": labels[i], "ReOmega/2pi (THz)": ang_to_THz(z.real/sys.w_scale), "kappa/2pi (MHz)": ang_to_MHz((-z.imag)/sys.w_scale)})
    return pd.DataFrame(rows)


# =============================================================================
# Plotting functions through gtilde figures
# =============================================================================

ONE_CURVE_LABELS = [r"$\tilde{g}_{\mathrm{mec},a}$", r"$\tilde{g}_{\mathrm{mec},d}$", r"$\tilde{g}_{a}$", r"$\tilde{g}_{d}$"]
TWO_CURVE_LABELS = [r"$\tilde{g}_{\mathrm{mec},a}$", r"$\tilde{g}_{\mathrm{mec},d_1}$", r"$\tilde{g}_{\mathrm{mec},d_2}$", r"$\tilde{g}_{a}$", r"$\tilde{g}_{d_1}$", r"$\tilde{g}_{d_2}$"]


def geff_curve_style(sys: UnifiedSystem):
    if sys.mode == "one":
        return ["C0", "C2", "C1", "C3"], ["-", "-", "--", "--"], ONE_CURVE_LABELS
    return ["C0", "C2", "C1", "C3", "C4", "C5"], ["-", "-", "-", "--", "--", "--"], TWO_CURVE_LABELS


def compute_gtilde(sys: UnifiedSystem, x, scale="Omega"):
    wL = sys.omega_for_scan(x, scale=scale)
    return np.array(sys.g_effs(wL=wL), dtype=complex), wL


def compute_std_gtilde(sys_std: UnifiedSystem, x):
    wL = sys_std.wm - x*sys_std.Om
    return np.array(sys_std.g_effs(wL=wL), dtype=complex)[0], wL


def plot_geff_old_style(devices: Dict[str, UnifiedSystem], dev_name="1", x=None, scale=None, include_std=True):
    """Reproduce the old one-mode paper-style single-device gtilde figure."""
    if x is None:
        x = np.linspace(0, 2, 801)
    sys = devices[dev_name]
    if scale is None:
        scale = "kappa" if dev_name == "exp" else "Omega"
    g, wL = compute_gtilde(sys, x, scale=scale)
    colors, linestyles, labels = geff_curve_style(sys)
    gstd = None
    if include_std and "std" in devices:
        gstd, _ = compute_std_gtilde(devices["std"], x)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 3.75))
    for i in range(len(g)):
        ax1.plot(x, np.abs(g[i])/sys.Om, color=colors[i], ls=linestyles[i])
        ax2.plot(x, np.angle(g[i])/pi, color=colors[i], ls=linestyles[i])
    if gstd is not None:
        ax1.plot(x, np.abs(gstd)/devices["std"].Om, "k", lw=2, ls=(0, (1, 3)))
        ax2.plot(x, np.angle(gstd)/pi, "k", lw=2, ls=(0, (1, 3)))
        labels = labels + [r"$\tilde{g}^{\mathrm{std}}$"]
    ax1.legend(labels, ncol=1, handlelength=0.9, columnspacing=0.5, handletextpad=0.2, loc="center left", bbox_to_anchor=(1.02, 0.5))
    if dev_name == "1":
        ax2.set_yticks([-0.5, 0]); ax2.set_yticklabels([r"$-\frac{\pi}{2}$", "$0$"])
        ax1.set_yticks([0, 1e-3]); ax1.set_ylim(-1e-4, 1.25e-3)
    else:
        ax2.set_yticks([-1, 0, 1]); ax2.set_yticklabels([r"$-\pi$", "$0$", r"$\pi$"])
    ax2.set_xlabel(r"$\tilde\Delta_-/\Omega_\mathrm{mec}$" if scale == "Omega" else r"$\tilde\Delta_-/\kappa_-$", labelpad=-4)
    ax1.set_ylabel(r"$|\tilde{g}|/\Omega_\mathrm{mec}$", labelpad=10)
    ax2.set_ylabel(r"$\phi/\pi$", labelpad=-4)
    ax2.set_xticks(np.arange(0, 2, 0.5), minor=True)
    ax1.set_xlim(x[0], x[-1])
    fig.tight_layout(pad=0.1)
    fig.subplots_adjust(hspace=0)

    fig2, ax = plt.subplots(figsize=(3, 3))
    for i in range(len(g)):
        ax.plot(g[i].real/sys.Om, g[i].imag/sys.Om, color=colors[i], ls=linestyles[i])
        ax.plot(g[i][0].real/sys.Om, g[i][0].imag/sys.Om, "o", color=colors[i], mfc="w")
    ax.set_xlabel(r"$\Re(\tilde{g})/\Omega_\mathrm{mec}$", labelpad=12)
    ax.set_ylabel(r"$\Im(\tilde{g})/\Omega_\mathrm{mec}$")
    ax.axis("equal")
    ax.grid()
    fig2.tight_layout(pad=0.1)
    return fig, fig2


def plot_geff_three_panel(devices: Dict[str, UnifiedSystem], x_ranges=None, scales=None, include_std=True):
    if x_ranges is None:
        x_ranges = {"exp": (0, 2), "1": (0, 2), "2": (0, 2)}
    if scales is None:
        scales = {"exp": "kappa", "1": "Omega", "2": "Omega"}
    panel_labels = {"exp": "(a)", "1": "(b)", "2": "(c)"}
    titles = {"exp": "Device exp", "1": "Device 1", "2": "Device 2"}
    fig = plt.figure(figsize=(13.2, 10.2))
    gs = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.25)
    specs = {"exp": gs[0,0], "1": gs[0,1], "2": gs[1,0]}
    all_handles = None
    table_rows = []
    for dev in ["exp", "1", "2"]:
        sys = devices[dev]
        x = np.linspace(*x_ranges[dev], 801)
        scale = scales[dev]
        g, _ = compute_gtilde(sys, x, scale=scale)
        gstd = compute_std_gtilde(devices["std"], x)[0] if include_std and "std" in devices else None
        colors, lss, labels = geff_curve_style(sys)
        sub = specs[dev].subgridspec(2,1,hspace=0.0)
        ax1 = fig.add_subplot(sub[0,0]); ax2 = fig.add_subplot(sub[1,0], sharex=ax1)
        handles=[]
        for i in range(len(g)):
            h,=ax1.plot(x, np.abs(g[i])/sys.Om, color=colors[i], ls=lss[i], lw=2)
            ax2.plot(x, np.angle(g[i]), color=colors[i], ls=lss[i], lw=2)
            handles.append(Line2D([0],[0],color=colors[i],ls=lss[i],lw=2,label=labels[i]))
        if gstd is not None:
            ax1.plot(x, np.abs(gstd)/devices["std"].Om, color="k", lw=2, ls=(0,(1,3)))
            ax2.plot(x, np.angle(gstd), color="k", lw=2, ls=(0,(1,3)))
            handles.append(Line2D([0],[0],color="k",lw=2,ls=(0,(1,3)),label=r"$\tilde{g}^{\mathrm{std}}$"))
        if all_handles is None: all_handles=handles
        ax2.set_yticks([-np.pi,0,np.pi]); ax2.set_yticklabels([r"$-\pi$", "$0$", r"$\pi$"])
        ax1.tick_params(axis="x", labelbottom=False)
        ax1.set_ylabel(r"$|\tilde{g}|/\Omega_\mathrm{mec}$", fontsize=14)
        ax2.set_ylabel(r"$\phi$ (rad)", fontsize=14)
        ax2.set_xlabel(r"$\Delta_i/\kappa_i$" if scale=="kappa" else r"$\Delta_i/\Omega_\mathrm{mec}$", fontsize=14)
        ax1.set_xlim(x[0], x[-1])
        for ax in [ax1, ax2]:
            ax.grid(False)
            for sp in ax.spines.values(): sp.set_linewidth(1.2)
        title_extra = r"$\Omega_-$" if sys.mode == "one" else rf"$\Omega_{{{getattr(sys,'lowest_mode_index',1)}}}$"
        ax1.text(0.98,1.04,f"{panel_labels[dev]}  {titles[dev]}  ({title_extra})", transform=ax1.transAxes, ha="right", va="bottom", fontsize=10)
        table_rows.append({"Device": dev, "mode": title_extra, "ReOmega/2pi THz": ang_to_THz(sys.wm/sys.w_scale), "kappa/2pi MHz": ang_to_MHz(sys.km/sys.w_scale), "xscale": scale})
    ax_info = fig.add_subplot(gs[1,1]); ax_info.axis("off")
    ax_info.text(0.03, 0.97, "(d)", transform=ax_info.transAxes, va="top", ha="left")
    leg = ax_info.legend(handles=all_handles, loc="upper center", bbox_to_anchor=(0.5,0.98), ncol=2, frameon=True, fontsize=10)
    leg.get_frame().set_facecolor("#f0f0f0")
    df = pd.DataFrame(table_rows)
    cell_text = [[r["Device"], r["mode"], f'{r["ReOmega/2pi THz"]:.6f}', f'{r["kappa/2pi MHz"]:.4f}', r["xscale"]] for _,r in df.iterrows()]
    tbl = ax_info.table(cellText=cell_text, colLabels=["Device","mode",r"Re$\Omega/2\pi$ THz",r"$\kappa/2\pi$ MHz","x"], cellLoc="center", bbox=[0.02,0.03,0.96,0.55])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    return fig, df
