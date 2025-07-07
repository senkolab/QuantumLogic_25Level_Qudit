import numpy as np
import math
from scipy.linalg import expm, norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import numpy as np
import math  # For np.hypot
from scipy.linalg import expm
import matplotlib.pyplot as plt

def coupling_operator_with_phase(i, j, dim, phi):
    op = np.zeros((dim, dim), dtype=complex)
    op[i, j] = np.exp(+1j * phi)
    op[j, i] = np.exp(-1j * phi)
    return op

def pulse_duration_for_fraction(f, Omega):
    theta = np.pi * np.array(f)
    return theta / Omega if Omega != 0 else 0.0

def unitary(couplings, rabi_freqs, fractions, fixed_phase_flags, dim):
    U_seq = np.eye(dim, dtype=complex)
    for (levels, Omega, frac, fix_pflag) in zip(couplings, rabi_freqs, fractions, fixed_phase_flags):
        i, j = levels
        # In the original scheme the phase was given by fix flag * π.
        phi_fixed = fix_pflag * np.pi
        total_phase = phi_fixed 
        H_op = coupling_operator_with_phase(i, j, dim, total_phase)
        H_coupling = 0.5 * Omega * H_op
        t_pulse = pulse_duration_for_fraction(frac, Omega)
        U_pulse = expm(-1j * H_coupling * t_pulse)
        U_seq = U_pulse @ U_seq
    return U_seq

def fix_couplings_and_phases(couplings, fixed_phase_flags):
    new_couplings = []
    new_fixed_phase_flags = []
    for (cpl, phase_flag) in zip(couplings, fixed_phase_flags):
        i, j = cpl
        if i != 0 and j == 0:
            cpl_fixed = (0, i)
            phase_flag_fixed = phase_flag + 1.0
        else:
            cpl_fixed = cpl
            phase_flag_fixed = phase_flag
        new_couplings.append(cpl_fixed)
        new_fixed_phase_flags.append(phase_flag_fixed)
    return new_couplings, new_fixed_phase_flags

dim = 8
couplings =  [(0, 7), (0, 3), (0, 6), (0, 4), (0, 5), (0, 2), (0, 1), (0, 2), (0, 5), (0, 4), (0, 6), (0, 3), (0, 7), (0, 1), (0, 2), (0, 4), (0, 3), (0, 7), (0, 2), (0, 5), (0, 6), (0, 3), (0, 5), (0, 4), (0, 6), (0, 7), (0, 4), (0, 6), (0, 3), (0, 6), (0, 0), (0, 0)]
fractions =  [1.5, 0.3918265520306073, 0.33333333333333337, 0.2951672353008666, 0.26772047280122996, 0.24675171442884994, 0.7699465438373841, 0.45437105165701, 0.4195693767448337, 0.2951672353008666, 0.33333333333333326, 0.2163468959387855, 0.33333333333333326, 1.0, 1.0, 1.0, 0.33333333333333326, 0.3918265520306073, 1.0, 1.0, 1.6666666666666667, 0.49999999999999983, 1.0, 1.0, 1.6081734479693928, 0.5000000000000002, 1.0, 1.0, 1.0, 1.0, 0, 0]
phases =  [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.5, 1.5, 0.5, 1.5, 1.5, 0.5, 1.5, 0.5, 0.5, 1.5, 0.5, 1.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0]
fixed_phase_flags = phases
rabi_freqs = [1]*len(couplings)
dim = 4
couplings =  [(0, 3), (0, 2), (0, 1), (0, 2), (0, 3), (0, 1), (0, 2), (0, 3), (0, 2), (0, 0)]
fractions =  [1.5, 0.3918265520306073, 0.6666666666666667, 0.3918265520306073, 0.5, 1.0, 1.0, 1.0, 1.0, 0]
phases =  [1.5, 1.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 0.5, 0.0]
fixed_phase_flags = phases
rabi_freqs = [1]*len(couplings)
# couplings, fixed_phase_flags = fix_couplings_and_phases(couplings, fixed_phase_flags)
# print("Fixed Couplings and Phase Flags:")
# print(couplings, fixed_phase_flags)

# Build the target unitary using the original pulses.
A2 = unitary(couplings, rabi_freqs, fractions, fixed_phase_flags, dim)
print("\nTarget Unitary (rounded):")
print(np.round(A2, 3))
plt.imshow(np.real(A2))
plt.colorbar()
plt.show()

def compute_error(x, couplings, rabi_freqs, U_target, dim):
    """
    x = [f0, f1, ..., f_{M-1}, phi0, phi1, ..., phi_{M-1}]
    returns Frobenius norm ||U_seq(params) - U_target||.
    """
    M = len(couplings)
    fracs  = x[:M]
    phis   = x[M:]
    # print(phis)
    U_seq  = unitary(couplings, rabi_freqs, fracs, phis, dim)
    return norm(U_seq - U_target, 'fro')

def optimize_sequence(couplings, rabi_freqs, init_fracs, init_phis,
                      U_target, dim,
                      bounds_frac=[(0,2)], bounds_phi=[(0,2)],
                      maxiter=10000):
    M = len(couplings)
    x0 = np.hstack([init_fracs, init_phis])
    # build bounds arrays
    bnds = bounds_frac * M + bounds_phi * M
    res = minimize(
        compute_error, x0,
        args=(couplings, rabi_freqs, U_target, dim),
        method='L-BFGS-B',
        bounds=bnds,
        options={'maxiter': maxiter}
    )
    fracs_opt = res.x[:M]
    phis_opt  = res.x[M:]
    return fracs_opt, phis_opt, res.fun

def reduce_sequence(couplings, rabi_freqs, fracs, phis,
                    U_target, dim,
                    error_threshold=1e-3):
    """
    Greedily try 
    
    removing each pulse; if after re-optimizing the remaining pulses
    the error is below threshold, accept the removal and restart.
    """
    changed = True
    while changed and len(couplings) > 1:
        changed = False
        for i in range(len(couplings))[::-1]:
            # build trial sequence with pulse i removed
            c_trial = couplings[:i] + couplings[i+1:]
            f_trial = fracs[:i] + fracs[i+1:]
            p_trial = phis[:i]  + phis[i+1:]
            # re-optimize
            f_opt, p_opt, err = optimize_sequence(
                c_trial, rabi_freqs[:len(c_trial)],
                f_trial, p_trial,
                U_target, dim
            )
            print(f"  Trying removal of pulse {i}: err = {err:.2e}")
            if err < error_threshold:
                print(f"  → Pulse {i} removed (new length {len(c_trial)})")
                couplings = c_trial
                fracs     = list(f_opt)
                phis      = list(p_opt)
                rabi_freqs= rabi_freqs[:len(c_trial)]
                changed   = True
                break
        if not changed:
            print("No further removals possible within threshold.")
    return couplings, fracs, phis


U_target = A2
# show original
print("Original sequence length:", len(couplings))
print("Original error (should be 0):",
      compute_error(np.hstack([fractions, phases]),
                    couplings, rabi_freqs, U_target, dim))

# try to reduce
coupl_opt, fracs_opt, phis_opt = reduce_sequence(
    couplings, rabi_freqs, fractions, phases,
    U_target, dim,
    error_threshold=1e-1
)


print("\nOptimized sequence length:", len(coupl_opt))
print("Final error:",
      compute_error(np.hstack([fracs_opt, phis_opt]),
                    coupl_opt, rabi_freqs[:len(coupl_opt)],
                    U_target, dim))

# (optional) visualize
U_red = unitary(coupl_opt, rabi_freqs[:len(coupl_opt)],
                       fracs_opt, phis_opt, dim)
plt.figure()
plt.title("Real part of reduced-unitary vs target")
plt.imshow(np.real(U_red - U_target), cmap='bwr')
plt.colorbar()
plt.show()
