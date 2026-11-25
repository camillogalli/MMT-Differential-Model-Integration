import numpy as np
from DAM import dndk_upwind, d2ndk2_center, apply_bc

def spectral_slope(k, n, dlnk):

    s = np.zeros_like(n)

    dndk = dndk_upwind(n, k, dlnk)
    s = (k* dndk)/n
    return s

def flux_bracket(n, k, dlnk, eta1, eta2):
    N = np.trapezoid(n, k) 
    s = spectral_slope(k, n, dlnk)

    flux = - (N**3) * (eta1 *s + eta2 * s**2 )
    return flux


def smoothness_ratio(u):

    r = np.zeros_like(u)
    r[1:-1] = (u[1:-1] - u[:-2]) / (u[2:] - u[1:-1])

    return r

def limiter(r):

    minmod = np.maximum(0, np.minimum(1, r))
    return minmod

def limited_flux(u):
    r = smoothness_ratio(u)
    phi = limiter(r)

    F = u[:-1] + 0.5 * phi[:-1] * (u[1:] - u[:-1])

    return F

def flux_divergence(F):

    div = np.zeros(len(F) + 1)
    div[1:-1] = F[1:] - F[:-1]
    return div

def cn_upwind_step(n, k, dlnk, n_iter, dt, n_left_fix, eta1, eta2, tol = 1e-10):
    n = apply_bc(n, k, n_left_fix, buf_left=6, buf_right=6, fit_points=4)
    Phi_old = flux_bracket(n, k, dlnk, eta1, eta2)
    F_old = limited_flux(Phi_old)
    div_old = flux_divergence(F_old)
    
    n_next = n.copy()

    for _ in range(n_iter):
        Phi_new = flux_bracket(n_next, k, dlnk, eta1, eta2)
        F_new = limited_flux(Phi_new)
        div_new = flux_divergence(F_new)

        n_candidate = n + (dt / 2) * (div_old + div_new)
        n_candidate = apply_bc(n_candidate, k, n_left_fix, buf_left=6, buf_right=6, fit_points=4)

        if np.max(np.abs(n_candidate - n_next)) < tol:
            n_next = n_candidate
            break
        n_next = n_candidate

    return n_next

def integrate_cn_upwind(rhs, n0, t_span, dt, k, n_left_fix, dlnk , eta1, eta2, n_iter = 10):
    t0, tf = t_span
    n_step = int(np.ceil((tf-t0)/dt))

    t = t0
    n = n0.copy()

    ts = [t]
    ns = [n.copy()]

    for _ in range(n_step):
        
        n = cn_upwind_step(n , k, dlnk, n_iter, dt, n_left_fix, eta1, eta2)

        t += dt
        ts.append(t)
        ns.append(n.copy())

    return np.array(ts), np.array(ns)