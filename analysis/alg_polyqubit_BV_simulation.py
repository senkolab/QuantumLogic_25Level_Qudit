'''
VERY IMPORTANT. 2-POLYQUBIT HERE STARTS IN D5/2 AND IS BUSSED THROUGH S1/2.
3-POLYQUBIT STARTS IN S1/2 (I.E. |0> IS IN S1/2)
'''

import time
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from tqdm import tqdm

from plot_utils import nice_fonts, set_size
from unitary_simulator import Simulator

# mpl.rcParams.update(nice_fonts)
np.set_printoptions(suppress=True)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = colors * 4

# Define a couple dataclasses to store all relevant simulation parameters for
# different cases (eg. 2 or 3 polyqubits)

def get_oracle_sequence(secret=[0, 1], noisy_oracle=False):
    oracle_couplings = []
    oracle_phases = []
    oracle_angles = []

    oracle_phase_flips = [0] + [
        1 if np.dot([int(x) for x in format(i, f'0{len(secret)}b')], secret) %
        2 else 0 for i in range(2**len(secret))
    ]
    for idx, flipper in enumerate(oracle_phase_flips):
        if flipper == 1:
            oracle_couplings.append((0, idx))
            oracle_phases.append(0)
            oracle_angles.append(2)
        else:
            pass
    if noisy_oracle:
        oracle_noise_flags = [1] * len(oracle_couplings)
    else:
        oracle_noise_flags = [0] * len(oracle_couplings)

    result = [
        oracle_couplings, oracle_phases, oracle_angles, oracle_noise_flags
    ]
    return result


def virtual_oracle(secret, couplings, phases):
    '''This will ONLY change the subsequent phases of pulses in order
    to implement virtual-Z gates.'''

    if polyqubit == 2:
        oracle_phase_flips = [
            1 if np.dot([int(x)
                         for x in format(i, f'0{len(secret)}b')], secret) %
            2 else 0 for i in range(2**len(secret))
        ]
    elif polyqubit == 3:
        oracle_phase_flips = [
            1 if np.dot([int(x)
                         for x in format(i, f'0{len(secret)}b')], secret) %
            2 else 0 for i in range(2**len(secret))
        ]

        # breakpoint()
    phases_with_virtual_z = []
    for idx, t in enumerate(couplings):
        # flip phase of all subsequent transitions that
        # had a virtual z gate applied
        phases_with_virtual_z.append(
            (phases[idx] + oracle_phase_flips[t[1]]) % 2)

    return phases_with_virtual_z


def build_BVA_simulation(secret, couplings, phases, angles, noise_flags):

    # one way to implement a virtual-z is with no-noise 2pi pulses
    # oracle = get_oracle_sequence(secret=secret)

    # applying the oracle as gates
    # full_couplings = couplings + oracle[0] + couplings
    # full_phases = phases + oracle[1] + phases
    # full_angles = angles + oracle[2] + angles
    # full_noise_flags = noise_flags + oracle[3] + noise_flags

    # another way is with phase shifts on subsequent gates
    full_couplings = couplings + couplings
    full_phases = phases + virtual_oracle(secret, couplings, phases)
    full_angles = angles + angles
    full_noise_flags = noise_flags + noise_flags

    if polyqubit == 3:
        # # first attempt, with FULL Hadamard for initial state as well
        # full_couplings = couplings + couplings
        # full_phases = phases + virtual_oracle(secret, couplings, phases)
        # full_angles = angles + angles
        # full_noise_flags = noise_flags + noise_flags

        # using fast (~160us) initial state preparation
        full_couplings = poly3_sup_couplings + couplings
        full_phases = poly3_sup_phases + virtual_oracle(
            secret, couplings, phases)
        full_angles = poly3_sup_angles + angles
        full_noise_flags = poly3_sup_noise_flags + noise_flags

    return full_couplings, full_phases, full_angles, full_noise_flags


def run_BVA_simulation(class_Simulator,
                       shots: int = 100,
                       secrets=[[0, 0], [0, 1], [1, 0], [1, 1]]):

    sim = class_Simulator()

    sim.init_state = init_state
    sim.mappings = mappings

    results = np.zeros((len(sim.mappings), len(secrets), shots))
    for sec_idx, secret in enumerate(secrets):

        # generate the pulse sequence for each possible hidden string
        seq_full = build_BVA_simulation(secret, Had_couplings, Had_phases,
                                        Had_angles, Had_noise_flags)

        sim.couplings = seq_full[0]
        sim.phases = seq_full[1]
        sim.angles = seq_full[2]
        sim.noise_flags = seq_full[3]

        sim.initialise_system()
        for itera in tqdm(range(shots)):
            output = sim.evolve_state()
            results[:, sec_idx, itera] = output

    averaged_results = np.mean(results, axis=2)
    averaged_std = np.std(results, axis=2) / np.sqrt(shots)

    return averaged_results, averaged_std


def plot_result(class_Simulator,
                shots=100,
                plot_2d=False,
                plot_3d=False,
                plot_bar=True):

    if polyqubit == 2:
        hidden_strings = [r'\{00\}', r'\{10\}', r'\{01\}', r'\{11\}']
        secrets = [[0, 0], [0, 1], [1, 0], [1, 1]]

        # states_subsetter = 1  # ignore bussed state in 2-poly case
        states_subsetter = 0  # if not bussing this should be 0

    elif polyqubit == 3:
        hidden_strings = [
            r'\{000\}', r'\{001\}', r'\{010\}', r'\{011\}', r'\{100\}',
            r'\{101\}', r'\{110\}', r'\{111\}'
        ]
        secrets = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0],
                   [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        states_subsetter = 0  # no bussed state used in 3-poly

    sim_result = run_BVA_simulation(class_Simulator,
                                    shots=shots,
                                    secrets=secrets)

    data = sim_result[0][states_subsetter:]
    errors = sim_result[1][states_subsetter:]
    # data_rate = 1 - sim_result[0][0]

    if polyqubit == 2:
        results_dict = {
            '00': np.round(np.max(100 * data[0]), 2),
            '01': np.round(np.max(100 * data[1]), 2),
            '10': np.round(np.max(100 * data[2]), 2),
            '11': np.round(np.max(100 * data[3]), 2),
        }
    elif polyqubit == 3:
        results_dict = {
            '000': np.round(np.max(100 * data[0]), 2),
            '001': np.round(np.max(100 * data[1]), 2),
            '010': np.round(np.max(100 * data[2]), 2),
            '011': np.round(np.max(100 * data[3]), 2),
            '100': np.round(np.max(100 * data[4]), 2),
            '101': np.round(np.max(100 * data[5]), 2),
            '110': np.round(np.max(100 * data[6]), 2),
            '111': np.round(np.max(100 * data[7]), 2),
        }

    print(results_dict)

    if plot_2d:
        cmap = plt.get_cmap("viridis")
        fig, ax = plt.subplots(1, 1, figsize=set_size('half'))

        # main plot
        ax.imshow(data, cmap=cmap, origin="lower")

        # Create the list of x tick labels
        xtick_labels = ['{' + f'{i}' + '}' for i in hidden_strings]
        x_positions = np.arange(len(hidden_strings))  # Assuming 25 bars

        ax.set_xticks(x_positions)
        ax.set_xticklabels(xtick_labels, rotation=0, ha='center', fontsize=12)

        # # Create the list of y tick labels
        ytick_labels = ['{' + f'{i}' + '}' for i in hidden_strings]
        y_positions = np.arange(len(hidden_strings))  # Get number of runs
        ax.set_yticks(y_positions)
        ax.set_yticklabels(ytick_labels, rotation=0, ha='right', fontsize=12)

        # Set labels and title
        ax.set_ylabel('Found String')
        ax.set_xlabel('Hidden String')
        # date = date_time.split('_')
        # date_obj = datetime.strptime(date[0], '%Y%m%d')
        # plot_date = date_obj.strftime('%B %d, %Y')
        ax.set_title(
            f'{polyqubit}-polyqubit: Bernstein-Vazirani Result - Simulation')

        text_threshold = 0.0
        # Add text annotations for values above a certain threshold
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]
                if value > text_threshold:
                    ax.text(j,
                            i,
                            str(np.round(100 *
                                         value, 2)) if value < 1 else 100,
                            color="black" if value > 0.5 else "white",
                            fontsize=12,
                            ha="center",
                            va="center")

    if plot_bar:
        # Configure the subplots
        fig, axs = plt.subplots(2,
                                2**(polyqubit - 1),
                                figsize=(12, 5),
                                sharex=True,
                                sharey=True)

        plt.suptitle(
            f'{polyqubit}-polyqubit: Bernstein-Vazirani Result - Simulation')
        # Create bar graphs for each row with error bars
        for i in range(2**polyqubit):
            # Create bar plot and capture the bars for annotation
            bars = axs[i % 2, i // 2].bar(range(len(data[i])),
                                          data[i],
                                          yerr=errors[i],
                                          capsize=5)

            # add a reference bar graph
            ref = np.zeros(len(data[i]))
            ref[i] = 1
            axs[i % 2, i // 2].bar(
                range(len(data[i])),
                ref,
                # yerr=errors[i],
                alpha=0.5,
                edgecolor='black',
                linestyle='--',
                facecolor='none')

            axs[i % 2, i // 2].set_xticks(range(len(data[i])))
            axs[i % 2, i // 2].set_xticklabels(hidden_strings)
            axs[i % 2, i // 2].legend([f'Key = {hidden_strings[i]}'])
            axs[i % 2, i // 2].grid(alpha=0.5)

            # Annotate each bar with its percentage value
            for bar in bars:
                yval = bar.get_height()  # Get the height of each bar
                if yval > 0.25:
                    axs[i % 2, i // 2].text(bar.get_x() + bar.get_width() / 2,
                                            yval * 0.5,
                                            f'{100 * yval:.2f}%',
                                            ha='center',
                                            va='center',
                                            color='black')

        axs[0, 0].set_ylabel('Probabilities')
        axs[1, 0].set_ylabel('Probabilities')
        axs[1, 0].set_xlabel('Measured Keys')
        axs[1, 1].set_xlabel('Measured Keys')
        if polyqubit == 3:
            axs[1, 2].set_xlabel('Measured Keys')
            axs[1, 3].set_xlabel('Measured Keys')

    if plot_3d:
        norm = Normalize(vmin=data.min(), vmax=data.max())
        colors = plt.cm.viridis(data.flatten() / float(data.max()))

        # Create a 3D figure
        fig = plt.figure(figsize=set_size('half'))
        ax = fig.add_subplot(111, projection='3d')

        # Create meshgrid for x and y coordinates
        x_indices, y_indices = np.meshgrid(np.arange(data.shape[1]),
                                           np.arange(data.shape[0]))

        # Flatten the arrays for plotting
        x_flat = x_indices.flatten()
        y_flat = y_indices.flatten()
        # Heights are zero (since we're using colors)
        z_flat = np.zeros_like(x_flat)

        # Plot the bars with colors
        ax.bar3d(x_flat,
                 y_flat,
                 z_flat,
                 1,
                 1,
                 data.flatten(),
                 shade=True,
                 color=colors,
                 alpha=0.8)

        # Add a color bar
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.viridis),
                            ax=ax)
        cbar.set_label('Outcome Probability')

        # Create the list of x tick labels
        xtick_labels = ['{' + f'{i}' + '}' for i in hidden_strings]
        x_positions = np.arange(len(hidden_strings))  # Assuming 25 bars

        ax.set_xticks(x_positions)
        ax.set_xticklabels(xtick_labels, rotation=0, ha='center', fontsize=12)

        # # Create the list of y tick labels
        ytick_labels = ['{' + f'{i}' + '}' for i in hidden_strings]
        y_positions = np.arange(len(hidden_strings))  # Get number of runs
        ax.set_yticks(y_positions)
        ax.set_yticklabels(ytick_labels, rotation=0, ha='right', fontsize=12)

        # Set labels and title
        ax.set_ylabel('Found String')
        ax.set_xlabel('Hidden String')
        # date = date_time.split('_')
        # date_obj = datetime.strptime(date[0], '%Y%m%d')
        # plot_date = date_obj.strftime('%B %d, %Y')
        ax.set_title(
            f'{polyqubit}-polyqubit: Bernstein-Vazirani Result - Simulation')

    print("The mean success probability: ",
          np.round(100 * np.mean(np.diagonal(data)), 3), "%.")
    print("The mean success error: ",
          np.round(100 * np.std(np.diagonal(data)), 3), "%.")
    breakpoint()
    return data, results_dict


# def get_simulation_result(polyqubit=3):

#     if polyqubit == 2:
#         init_state = '-2'
#         mappings = {
#             0: '-2',
#             1: '[2, -1]',
#             3: '[3, -2]',
#             2: '[4, -4]',
#             # 4: '[3, -3]'
#         }

#         # DEFINE PULSE SEQUENCE FOR RELEVANT GATES
#         # IN THIS CASE JUST HADAMARD (ORACLE IS BUILT LATER)
#         # no bussing version
#         Had_couplings = [(0, 3), (0, 1), (0, 2), (0, 1), (0, 3)]
#         Had_angles = [
#             0.5, 2.0 * np.arcsin(np.sqrt(1 / 3)) / np.pi, 4 / 3,
#             2.0 * np.arcsin(np.sqrt(2 / 3)) / np.pi + 1, 0.5
#         ]
#         Had_phases = [1.5, 0.5, 1.5, 0.5, 0.5]
#         Had_noise_flags = [1] * len(Had_couplings)

#         # # version with bussing
#         # Had_couplings = [(0, 2), (0, 3), (0, 1), (0, 4), (0, 1), (0, 3),
#         #                  (0, 2)]
#         # # # "swapped" Hadamard - switched 01 and 10 states
#         # # Had_phases = [1, 1, 0, 1, 0, 1, 1]
#         # # Had_angles = [
#         # #     1, 0.5, 2.0 * np.arcsin(np.sqrt(1 / 3)) / np.pi, 2 / 3,
#         # #     2.0 * np.arcsin(np.sqrt(1 / 3)) / np.pi, 0.5, 1
#         # # ]
#         # # "true" Hadamard
#         # Had_phases = [1, 0, 1, 0, 1, 0, 0]
#         # Had_angles = [
#         #     1, 1.5, 2.0 * np.arcsin(np.sqrt(1 / 3)) / np.pi, 2 / 3,
#         #     2.0 * np.arcsin(np.sqrt(1 / 3)) / np.pi, 0.5, 1
#         # ]
#         # Had_noise_flags = [1] * len(Had_couplings)

#         # virtual_oracle([0,1],Had_couplings, Had_phases)

#         plot_result(Simulator, shots=1000, plot_2d=True, plot_3d=False)

#     elif polyqubit == 3:
#         # following from D9_Ramsey_PulseSequenceStarTopology txt file output
#         init_state = '1'
#         mappings = {
#             0: '1',
#             1: '[2, -1]',
#             2: '[3, 2]',
#             3: '[4, 2]',
#             4: '[4, 0]',
#             5: '[3, 3]',
#             6: '[4, 1]',
#             7: '[4, -1]'
#         }

#         # first the 7 pulses needed for creating the equal sup. in ~160us
#         poly3_sup_couplings = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
#                                (0, 7)]
#         poly3_sup_angles = [
#             2 * np.arcsin(np.sqrt(1 / 8)) / np.pi,
#             2 * np.arcsin(np.sqrt(1 / 7)) / np.pi,
#             2 * np.arcsin(np.sqrt(1 / 6)) / np.pi,
#             2 * np.arcsin(np.sqrt(1 / 5)) / np.pi,
#             2 * np.arcsin(np.sqrt(1 / 4)) / np.pi,
#             2 * np.arcsin(np.sqrt(1 / 3)) / np.pi,
#             2 * np.arcsin(np.sqrt(1 / 2)) / np.pi + 1
#         ]
#         poly3_sup_phases = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
#         poly3_sup_noise_flags = [1] * len(poly3_sup_couplings)

#         # then the full true Hadamard sequence of 21 pulses
#         Had_couplings = [
#             (0, 6), (0, 1), (0, 3), (0, 5), (0, 4), (0, 2), (0, 4), (0, 5),
#             (0, 7), (0, 3), (0, 1), (0, 2), (0, 4), (0, 7), (0, 1), (0, 5),
#             (0, 3), (0, 6), (0, 7), (0, 1), (0, 6)
#         ]
#         Had_angles = [
#             1.5, 1, 1., 1.5, 2.0 * np.arcsin(np.sqrt(1 / 3)) / np.pi, 2 / 3,
#             1 + 2.0 * np.arcsin(np.sqrt(1 / 3)) / np.pi, 1, 1, 0.5, 0.5, 0.5,
#             1.0, 1.5, 1.0, 1.5, 1, 1.5, 1.5, 1, 1.5
#         ]
#         Had_phases = [
#             0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 0.5,
#             1.5, 0.5, 0.5, 1.5, 1.5, 0.5, 1.5, 1.5
#         ]

#         Had_noise_flags = [1] * len(Had_couplings)

#         # virtual_oracle([0, 0, 1], Had_couplings, Had_phases)

#         sim_data, _ = plot_result(Simulator,
#                                   polyqubit=polyqubit,
#                                   shots=100,
#                                   plot_2d=False,
#                                   plot_3d=False,
#                                   plot_bar=False)

#     return sim_data


if __name__ == '__main__':
    start_time = time.time()

    polyqubit = 3

    if polyqubit == 2:
        init_state = '-2'
        mappings = {
            0: '-2',
            1: '[2, -1]',
            3: '[3, -2]',
            2: '[4, -4]',
            # 4: '[3, -3]'
        }

        # DEFINE PULSE SEQUENCE FOR RELEVANT GATES
        # IN THIS CASE JUST HADAMARD (ORACLE IS BUILT LATER)
        # no bussing version
        Had_couplings = [(0, 3), (0, 1), (0, 2), (0, 1), (0, 3)]
        Had_angles = [
            0.5, 2.0 * np.arcsin(np.sqrt(1 / 3)) / np.pi, 4 / 3,
            2.0 * np.arcsin(np.sqrt(2 / 3)) / np.pi + 1, 0.5
        ]
        Had_phases = [1.5, 0.5, 1.5, 0.5, 0.5]
        Had_noise_flags = [1] * len(Had_couplings)

        # # version with bussing
        # Had_couplings = [(0, 2), (0, 3), (0, 1), (0, 4), (0, 1), (0, 3),
        #                  (0, 2)]
        # # # "swapped" Hadamard - switched 01 and 10 states
        # # Had_phases = [1, 1, 0, 1, 0, 1, 1]
        # # Had_angles = [
        # #     1, 0.5, 2.0 * np.arcsin(np.sqrt(1 / 3)) / np.pi, 2 / 3,
        # #     2.0 * np.arcsin(np.sqrt(1 / 3)) / np.pi, 0.5, 1
        # # ]
        # # "true" Hadamard
        # Had_phases = [1, 0, 1, 0, 1, 0, 0]
        # Had_angles = [
        #     1, 1.5, 2.0 * np.arcsin(np.sqrt(1 / 3)) / np.pi, 2 / 3,
        #     2.0 * np.arcsin(np.sqrt(1 / 3)) / np.pi, 0.5, 1
        # ]
        # Had_noise_flags = [1] * len(Had_couplings)

        # virtual_oracle([0,1],Had_couplings, Had_phases)

        plot_result(Simulator, shots=1000, plot_2d=True, plot_3d=False)

    elif polyqubit == 3:
        # following from D9_Ramsey_PulseSequenceStarTopology txt file output
        init_state = '1'
        mappings = {
            0: '1',
            1: '[2, -1]',
            2: '[3, 2]',
            3: '[4, 2]',
            4: '[4, 0]',
            5: '[3, 3]',
            6: '[4, 1]',
            7: '[4, -1]'
        }

        # first the 7 pulses needed for creating the equal sup. in ~160us
        poly3_sup_couplings = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
                               (0, 7)]
        poly3_sup_angles = [
            2 * np.arcsin(np.sqrt(1 / 8)) / np.pi,
            2 * np.arcsin(np.sqrt(1 / 7)) / np.pi,
            2 * np.arcsin(np.sqrt(1 / 6)) / np.pi,
            2 * np.arcsin(np.sqrt(1 / 5)) / np.pi,
            2 * np.arcsin(np.sqrt(1 / 4)) / np.pi,
            2 * np.arcsin(np.sqrt(1 / 3)) / np.pi,
            2 * np.arcsin(np.sqrt(1 / 2)) / np.pi + 1
        ]
        poly3_sup_phases = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        poly3_sup_noise_flags = [1] * len(poly3_sup_couplings)

        # then the full true Hadamard sequence of 21 pulses
        Had_couplings = [
            (0, 6), (0, 1), (0, 3), (0, 5), (0, 4), (0, 2), (0, 4), (0, 5),
            (0, 7), (0, 3), (0, 1), (0, 2), (0, 4), (0, 7), (0, 1), (0, 5),
            (0, 3), (0, 6), (0, 7), (0, 1), (0, 6)
        ]
        Had_angles = [
            1.5, 1, 1., 1.5, 2.0 * np.arcsin(np.sqrt(1 / 3)) / np.pi, 2 / 3,
            1 + 2.0 * np.arcsin(np.sqrt(1 / 3)) / np.pi, 1, 1, 0.5, 0.5, 0.5,
            1.0, 1.5, 1.0, 1.5, 1, 1.5, 1.5, 1, 1.5
        ]
        Had_phases = [
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 0.5,
            1.5, 0.5, 0.5, 1.5, 1.5, 0.5, 1.5, 1.5
        ]

        Had_noise_flags = [1] * len(Had_couplings)

        # virtual_oracle([0, 0, 1], Had_couplings, Had_phases)

        plot_result(Simulator,
                    shots=1000,
                    plot_2d=True,
                    plot_3d=False,
                    plot_bar=True)

    plt.show()
    print("--- %s seconds ---" % (time.time() - start_time))
