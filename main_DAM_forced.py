from DAM_forced import rhs_forced
from DAM import integrate_chunked, k_space, initial_condition, apply_bc, plot_results
import numpy as np

eta1 = 216
eta2 = 162
k_f, sigma_f, amp_f = 1.0, 0.3, 1e3      
k_IR, k_UV = 2e-3, 2.0                     
gamma_IR, gamma_UV = 2.0, 2.0             

k, dlnk = k_space(1e-3, 5, 500)
ln_k = np.log(k)

n_k = initial_condition(1.0, k, -2/3)
n_left_fix = n_k[:6].copy()

print("\n")
print("************************* Starting the simulation ************************* ")

ts, ns = integrate_chunked(
    rhs_forced, n_k, (0.0, 1.0), dt_chunk=1e-8,
    apply_bc_func=apply_bc,
    bc_args=(k, n_left_fix, 6, 6, 4),
    rhs_args=(k, ln_k, n_left_fix, dlnk, eta1, eta2,
            k_f, sigma_f, amp_f, k_IR, k_UV, gamma_IR, gamma_UV),
    rtol=1e-6, atol=1e-10
)

print("************************* Simulation completed ************************* ")
print("\n")
print("----------------------- Plotting Results ------------------------------- ")

plot_results(ts, ns, n_snap = 7, k = k, save_path = "outputs_DAM/DAM_results_forced.png", show = False)
