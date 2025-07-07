import numpy as np
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++DECOMPOSITION+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def detect_adjacent_same_couplings(rotation_mats, tol=1e-8):

    matches = []
    for U1, U2 in zip(rotation_mats[:-1], rotation_mats[1:]):
        # Find which level (other than 0) this pulse couples with
        def find_coupled_level(U):
            for j in range(1, U.shape[0]):
                if np.abs(U[0, j]) > tol or np.abs(U[j, 0]) > tol:
                    return j
            return None

        j1 = find_coupled_level(U1)
        j2 = find_coupled_level(U2)

        matches.append(int(j1 == j2 and j1 is not None))

    return matches

import numpy as np
from numpy.linalg import norm

def compress_rotations(rotation_mats, tol=1e-8):
    i = 0
    optimized = []
    while i < len(rotation_mats):
        if i + 1 < len(rotation_mats):
            U1 = rotation_mats[i]
            U2 = rotation_mats[i+1]

            # Check if both couple the same (0, j)
            def find_coupled(U):
                for j in range(1, U.shape[0]):
                    if np.abs(U[0, j]) > tol or np.abs(U[j, 0]) > tol:
                        return j
                return None

            j1 = find_coupled(U1)
            j2 = find_coupled(U2)

            if j1 == j2 and j1 is not None:
                combined = U2 @ U1  # Order matters!
                if norm(combined - np.eye(U1.shape[0])) < tol:
                    # Cancels out → skip both
                    i += 2
                    continue
                else:
                    # Replace with combined
                    optimized.append(combined)
                    i += 2
                    continue

        # Default: keep current
        optimized.append(rotation_mats[i])
        i += 1

    return optimized

import numpy as np
from numpy.linalg import norm
from math import acos, pi

def inverse_single_pulse(U, *, tol= 1e-8):

    U = np.asarray(U)
    dim = U.shape[0]

    cand = [(p, q) for p in range(dim) for q in range(p+1, dim)
            if abs(U[p, q]) > tol or abs(U[q, p]) > tol]

    if len(cand) != 1:
        return (0,0), 0, 0

    i, j = cand[0]

    # everything outside rows/cols i,j must look like the identity
    mask = np.ones_like(U, dtype=bool)
    mask[[i, j], :] = False
    mask[:, [i, j]] = False
    if norm(U[mask] - np.eye(dim)[mask]) > tol:
        raise ValueError("Extra couplings detected – not a single pulse.")

    c = U[i, i].real        
    if abs(U[j, j].real - c) > tol or abs(U[i, i].imag) > tol or abs(U[j, j].imag) > tol:
        raise ValueError("Diagonal elements are inconsistent with a single pulse.")

    # numerical safety
    c = max(min(c, 1.0), -1.0)
    theta = 2 * acos(c)       # θ  (0 ≤ θ ≤ 2π)
    fraction = theta / pi

    u_ij = U[i, j]
    if abs(u_ij) < tol:
        raise ValueError("Zero off‑diagonal element – ambiguous phase.")
    phi = (np.angle(u_ij) + pi/2) % (2*pi)   

    return (i, j), fraction, phi

def givens_rotation(m, j, theta):

    G = np.eye(m)
    c = np.cos(theta)
    s = np.sin(theta)
    G[0, 0] = c
    G[0, j] = s
    G[j, 0] = -s
    G[j, j] = c
    return G

def star_qr_decomposition(order, U, tol=1e-10):
    
    m = U.shape[0]
    rotations_info = []  
    rotation_mats = []     
    V = U.copy()          
    

    col = 0
    count = 0
    for i in order[::-1]:#range(m-1, 0,-1):
        a = V[0, col]
        b = V[i, col]
        if np.abs(b) < tol:
            continue
        theta = np.arctan2(b, a)
        G = givens_rotation(m, i, theta)
        V = G @ V
        rotations_info.append(("rotate", i, theta, col))
        rotation_mats.append(G)
    

    for col in order:#range(1, m):
        theta_swap = np.pi / 2
        G_swap = givens_rotation(m, col, theta_swap)
        V = G_swap @ V
        rotations_info.append(("swap", col, theta_swap, col))
        rotation_mats.append(G_swap)
        count += 1 
        if count % 2 == 0:
            for i in order[count:][::-1]:#range(1, m):
                a = V[0, col]
                b = V[i, col]
                if np.abs(b) < tol:
                    continue
                theta = np.arctan2(b, a)
                G = givens_rotation(m, i, theta)
                V = G @ V
                rotations_info.append(("rotate", i, theta, col))
                rotation_mats.append(G)
    
            V = G_swap @ V   
            rotations_info.append(("swap_back", col, theta_swap, col))
            rotation_mats.append(G_swap)
        else:
            for i in order[count:]:#range(1, m):
                a = V[0, col]
                b = V[i, col]
                if np.abs(b) < tol:
                    continue
                theta = np.arctan2(b, a)
                G = givens_rotation(m, i, theta)
                V = G @ V
                rotations_info.append(("rotate", i, theta, col))
                rotation_mats.append(G)
    
            V = G_swap @ V   
            rotations_info.append(("swap_back", col, theta_swap, col))
            rotation_mats.append(G_swap)
    return rotations_info, V, rotation_mats



# dim = 5
# init_state = np.array([0, 0, 1, 0, 0])

# couplings = [(0, 2), (0, 3), (0, 1), (0, 4), (0, 1), (0, 3), (0, 2)]
# fractions = [1,
#              1.5,
#              2.0 * np.arcsin(np.sqrt(1/3)) / np.pi,
#              2/3,
#              2.0 * np.arcsin(np.sqrt(1/3)) / np.pi,
#              0.5,
#              1]
# fixed_phase_flags = [0, 1, 0, 1, 0, 1, 1]
# rabi_freqs = [1, 1, 1, 1, 1, 1, 1]

dim = 8
# init_state = np.array([0,1,0,0,0,0,0,0,0])
# couplings = [(2,0),(0,1),(2,0)] + [(4,0),(3,0)]  + [(0,1),(3,0)] + [(0,2),(4,0)]\
#           + [(6,0),(0,5),(6,0)] + [(8,0),(7,0)]  + [(0,5),(7,0)] + [(6,0)]\
#           + [(0,2),(6,0)] + [(5,0),(0,1),(5,0)] + [(7,0),(0,3),(7,0)] + [(0,4),(8,0)]
# fractions = [1,1/2,1] + [1,1/2]  + [1/2,1] + [1/2,1]\
#           + [1,1/2,1] + [1,1/2]  + [1/2,1] + [1/2]\
#           + [1/2,1] + [1,1/2,1] + [1,1/2,1] + [1/2,1]
# rabi_freqs = [1,1,1]+[1,1]+[1,1]+[1,1]\
#            + [1,1,1]+[1,1]+[1,1]+[1,1]\
#            +[1,1]+[1,1,1]+[1,1,1]+[1,1,1]
# fixed_phase_flags = [0.5,0.5,0.5] + [0.5,0.5]  + [0.5,0.5] + [0.5,0.5]\
#                    +[0.5,0.5,0.5] + [0.5,0.5]  + [0.5,0.5] + [0.5]\
#                    + [0.5,0.5] + [0.5,0.5,0.5] + [0.5,0.5,0.5] + [0.5,0.5]
    
# couplings, fixed_phase_flags = fix_couplings_and_phases(couplings, fixed_phase_flags)
# print("Fixed Couplings and Phase Flags:")
# print(couplings, fixed_phase_flags)

# U = np.real(unitary(couplings, rabi_freqs, fractions, fixed_phase_flags, dim))

U = -1*np.real(np.array([[0.354-0.j,  0.354+0.j,  0.354+0.j,  0.354+0.j,  0.354+0.j,
   0.354+0.j,  0.354+0.j,  0.354+0.j],
 [0.354-0.j, -0.354+0.j,  0.354+0.j, -0.354-0.j,  0.354+0.j,
  -0.354-0.j,  0.354+0.j, -0.354-0.j],
 [0.354-0.j,  0.354-0.j, -0.354-0.j, -0.354+0.j,  0.354+0.j,
   0.354+0.j, -0.354-0.j, -0.354-0.j],
 [0.354-0.j, -0.354+0.j, -0.354+0.j,  0.354-0.j,  0.354+0.j,
  -0.354-0.j, -0.354-0.j,  0.354+0.j],
 [0.354-0.j,  0.354-0.j,  0.354-0.j,  0.354-0.j, -0.354-0.j,
  -0.354-0.j, -0.354-0.j, -0.354-0.j],
 [0.354-0.j, -0.354+0.j,  0.354-0.j, -0.354+0.j, -0.354+0.j,
   0.354+0.j, -0.354-0.j,  0.354+0.j],
 [0.354-0.j, 0.354-0.j, -0.354+0.j, -0.354+0.j, -0.354+0.j,
  -0.354+0.j,  0.354+0.j, 0.354+0.j],
 [0.354-0.j, -0.354+0.j, -0.354+0.j,  0.354-0.j, -0.354+0.j,
   0.354+0.j,  0.354+0.j, -0.354+0.j]],  dtype=complex))
# U = np.real(np.array([[ 1.        -0.j,  0.        +0.j,  0.        +0.j,
#          0.        +0.j,  0.        +0.j,  0.        +0.j,
#          0.        +0.j,  0.        +0.j,  0.        +0.j],
#        [ 0.        +0.j,  0.35355339-0.j,  0.35355339-0.j,
#          0.35355339-0.j,  0.35355339-0.j,  0.35355339-0.j,
#          0.35355339-0.j,  0.35355339-0.j,  0.35355339-0.j],
#        [ 0.        +0.j,  0.35355339+0.j, -0.35355339+0.j,
#          0.35355339-0.j, -0.35355339+0.j,  0.35355339-0.j,
#         -0.35355339+0.j,  0.35355339-0.j, -0.35355339+0.j],
#        [ 0.        +0.j,  0.35355339+0.j,  0.35355339-0.j,
#         -0.35355339+0.j, -0.35355339+0.j,  0.35355339-0.j,
#          0.35355339-0.j, -0.35355339+0.j, -0.35355339+0.j],
#        [ 0.        +0.j,  0.35355339+0.j, -0.35355339-0.j,
#         -0.35355339+0.j,  0.35355339-0.j,  0.35355339-0.j,
#         -0.35355339+0.j, -0.35355339+0.j,  0.35355339-0.j],
#        [ 0.        +0.j,  0.35355339+0.j,  0.35355339-0.j,
#          0.35355339-0.j,  0.35355339-0.j, -0.35355339+0.j,
#         -0.35355339+0.j, -0.35355339+0.j, -0.35355339+0.j],
#        [ 0.        +0.j,  0.35355339+0.j, -0.35355339-0.j,
#          0.35355339-0.j, -0.35355339+0.j, -0.35355339-0.j,
#          0.35355339-0.j, -0.35355339+0.j,  0.35355339-0.j],
#        [ 0.        +0.j,  0.35355339+0.j,  0.35355339+0.j,
#         -0.35355339-0.j, -0.35355339-0.j, -0.35355339-0.j,
#         -0.35355339+0.j,  0.35355339-0.j,  0.35355339-0.j],
#        [ 0.        +0.j,  0.35355339+0.j, -0.35355339-0.j,
#         -0.35355339-0.j,  0.35355339+0.j, -0.35355339-0.j,
#          0.35355339-0.j,  0.35355339-0.j, -0.35355339+0.j]]))
# U = np.real(np.array([
#     [0.5-0.j,  0.5+0.j,  0.5-0.j,  0.5-0.j],
#     [0.5+0.j, -0.5-0.j,  0.5-0.j, -0.5+0.j],
#     [0.5+0.j,  0.5+0.j, -0.5+0.j, -0.5+0.j],
#     [0.5+0.j, -0.5-0.j, -0.5-0.j,  0.5+0.j]
# ], dtype=complex))

import itertools

my_list = [1,2,3,4,5,6,7]
permutations = list(itertools.permutations(my_list))

list_comp = []
len_comp = 36
for order in permutations:
    # print(order)
    rotations_info, V_triangular, rotation_mats = star_qr_decomposition(order,U)
    compressed = compress_rotations(rotation_mats)
    # coup, f, ph = inverse_single_pulse(compressed[13])
    # print(f)
    # if len(compressed) < len_comp:
    #     print(len(compressed))
        # list_comp.append(compressed)
        # len_comp = len(compressed)
    # print(V_triangular)
    couplings = []
    phases = []
    fractions = []
    # print(len(rotation_mats))
    for r in compressed:
        coupling, f, phi = inverse_single_pulse(r)
        couplings.append(coupling)
        fractions.append(f)
        phases.append(phi/np.pi)
    # if couplings[0:6]==couplings[7:13][::-1]:
    # # if 3%np.round(fractions[0],2) == 0 and 3%np.round(fractions[12],2) == 0 and np.round(fractions[12],2) != 1:   
    # print('couplings = ',couplings)
    # print('fractions = ', fractions)
    # print('phases = ', phases)
# Display the sequence of rotations.
    # print("Sequence of rotations (each tuple: (operation, index, theta, column)):")
    # for op in rotations_info:
    #     print(op)
    
    # print("\nFinal matrix after applying rotations (ideally diagonal):")
    np.set_printoptions(precision=3, suppress=True)
    # print(V_triangular)
    
    U_reconstructed = np.eye(dim)
    for G in reversed(compressed[:]):
        U_reconstructed = G.T @ U_reconstructed
    print(U_reconstructed)
    # print("\nReconstructed U from the rotation matrices (should match the original U):")
    a = np.round(np.abs(U_reconstructed),3).tolist()
    b = np.round(np.abs(U),3).tolist()
    if a==b:
        print('____________________________This works_____________________________________',len(couplings))
        print('couplings = ',couplings)
        print('fractions = ', fractions)
        print('phases = ', phases)
print('done')
