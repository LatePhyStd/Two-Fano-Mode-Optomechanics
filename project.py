import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from IPython.display import display
pi = np.pi
TWOPI = 2*np.pi

# =============================================================================
# Fixed laser frequencies for the bare optical Hamiltonian
# =============================================================================
# These define the rotating frame:
#     Delta_c = omega_c - omega_las0
#
# You can change these manually.
# =============================================================================

wL0_by_device = {
    "exp": 2*pi*2.03e14,
    "1":   2*pi*2.03e14,
    "2":   2*pi*2.03e14,
}

# =============================================================================
# Initial parameters
# =============================================================================

pars_exp = dict(
    # Fano mode d1
    kd1 = 2*pi*3.80e12,
    wd1 = 2*pi*2.04e14,

    # Fano mode d2
    kd2 = 2*pi*6.53e12,
    wd2 = 2*pi*2.16e14,

    # mechanics
    Om = 2*pi*5.14e5,
    Plas = 1.50e-4,
    thlas = 0.0,
    Qm = 3.07e4,
    Tmec = 3.12e2,

    # cavity
    wa = 2*pi*2.03e14,
    ka = 2*pi*2.12e12,
    ga = 2*pi*3.70e11,

    # coherent optical couplings
    lbd1 = 2*pi*4.09e12,
    lbd2 = 2*pi*9.34e12,

    # OM couplings
    gka0  = 2*pi*7.75e5,
    gkd10 = 2*pi*3.21e6,
    gkd20 = 2*pi*3.81e6,
    gwa0  = 2*pi*8.45e5,
    gwd10 = -2*pi*1.82e6,
    gwd20 = -2*pi*3.82e6,
)


# Shared ideal mechanical parameters
Om = 2*pi*1.3e6
Qm = 1.4e8
Tmec = 3.0e2
sideband_resolution = 5.0e-2
Plas1 = 5.0e-5
g0_reduce = 1.0e4


pars1 = dict(
    # Fano modes
    kd1 = 2*pi*2.22e12,
    kd2 = 2*pi*2.02e12,
    wd1 = 2*pi*2.03e14,
    wd2 = 2*pi*2.05e14,

    # mechanics
    Om = Om,
    Plas = Plas1,
    thlas = 0.0,
    Qm = Qm,
    Tmec = Tmec,

    # cavity
    wa = 2*pi*2.09e14,
    ka = 2*pi*2.12e12,
    ga = 2*pi*1.00e11,

    # coherent optical couplings
    lbd1 = 2*pi*4.09e12,
    lbd2 = 2*pi*9.34e12,

    # OM couplings
    gka0  = 2*pi*7.75e1,
    gkd10 = 2*pi*3.21e2,
    gkd20 = 2*pi*3.21e1,
    gwa0  = 2*pi*8.45e1,
    gwd10 = -2*pi*1.82e2,
    gwd20 = -2*pi*4.82e2,
)


pars2 = dict(
    # Fano modes
    kd1 = 2*pi*9.00e10,
    kd2 = 2*pi*6.00e10,
    wd1 = 2*pi*2.03e14,
    wd2 = 2*pi*2.04e14,

    # mechanics
    Om = Om,
    Plas = Plas1*5.13e-3,
    thlas = 0.0,
    Qm = Qm,
    Tmec = Tmec,

    # cavity
    wa = 2*pi*2.04e14,
    ka = 2*pi*9.40e10,
    ga = 2*pi*8.00e8,

    # coherent optical couplings
    lbd1 = 2*pi*2.65e11,
    lbd2 = 2*pi*2.35e11,

    # OM couplings
    gka0  = 2*pi*7.75e1,
    gkd10 = 2*pi*7.61e0,
    gkd20 = 2*pi*6.02e0,
    gwa0  = 2*pi*8.45e1,
    gwd10 = -2*pi*1.82e2,
    gwd20 = -2*pi*1.12e2,
)


# =============================================================================
# Apply switch at lab-parameter level
# =============================================================================

pars_by_device = {
    "exp": tfp.apply_fano_mode_switch_to_lab_parameters(
        pars_exp,
        wL0=wL0_by_device["exp"],
        switch=FANO_MODE_SWITCH,
    ),
    "1": tfp.apply_fano_mode_switch_to_lab_parameters(
        pars1,
        wL0=wL0_by_device["1"],
        switch=FANO_MODE_SWITCH,
    ),
    "2": tfp.apply_fano_mode_switch_to_lab_parameters(
        pars2,
        wL0=wL0_by_device["2"],
        switch=FANO_MODE_SWITCH,
    ),
}

initial_param_table = tfp.make_lbdbar_parameter_table(
    pars_by_device,
    wL0_by_device,
    switch=FANO_MODE_SWITCH,
)

print("Initial parameters")
display(initial_param_table.style.format({
    "exp": "{:.6e}",
    "1": "{:.6e}",
    "2": "{:.6e}",
}))



