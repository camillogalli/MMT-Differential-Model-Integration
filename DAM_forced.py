"""
In this file, I'm trying to integrate the DAM model with forced dynamics, trying to recreate Fig.3 of Tibone, Onorato, Krstulovic
*Cascades in the Kinetic Equation for the Majda-McLaughlin-Tabak model* 2026.

"""
import numpy as np
from scipy.integrate import solve_ivp
from DAM import apply_bc, dndk_upwind, d2ndk2_center

def forcing(k, ln_k, k_f, sigma_f, amp_f):
    return amp_f * np.exp(-(ln_k - np.log(k_f))**2 / (2*sigma_f**2))

def dissipation(n, k, k_IR, k_UV, gamma_IR, gamma_UV):
    return ((k_IR/k)**(-gamma_IR) + (k/k_UV)**gamma_UV) * n

def rhs_forced(t, n, k, ln_k, n_left_fix, dlnk, eta1, eta2,
               k_f, sigma_f, amp_f, k_IR, k_UV, gamma_IR, gamma_UV):
    
    n = apply_bc(n, k, n_left_fix, 6, 6, 4)

    dndk = dndk_upwind(n, k, dlnk)
    d2ndk2 = d2ndk2_center(n, k, dlnk)
    N = np.trapezoid(n, k)

    dndt_action = - (N**3) * (
        -eta1 * (1/n) * dndk + eta1 * (k / (n**2)) * (dndk**2) - eta1 * (k/n) * d2ndk2 + eta2 * (2*k/(n**2)) * (dndk**2)
            - 2 * eta2 * ((k**2)/(n**3)) * (dndk**3) + 2 * eta2 * ((k**2)/(n**2)) * dndk * d2ndk2
        )

    dndt = dndt_action - forcing(k, ln_k, k_f, sigma_f, amp_f) - dissipation(n, k, k_IR, k_UV, gamma_IR, gamma_UV)
    return dndt