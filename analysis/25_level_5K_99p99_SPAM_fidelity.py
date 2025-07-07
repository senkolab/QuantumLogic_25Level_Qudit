import csv
import os
import time
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# sys.path.insert(0, '/home/nicholas/Documents/Barium137_Qudit/analysis')
from calibration_analysis import Calibrations
from class_barium import Barium
from class_utils import Utils
from plot_utils import nice_fonts, set_size

find_errors = Utils.find_errors

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

mpl.rcParams.update(nice_fonts)
'''
Scaling studies and other associated plots and analysis for SPAM
outcomes with tests of the system parameters needed for 99.99%
SPAM fidelities.
'''


def translate_transitions_to_indices(transitions):
    trans_tuples = []
    for transition in transitions:
        trans_tuple1 = int(transition[0] + 2)

        helpers = np.array([1, 5, 11, 19])
        trans_tuple0 = helpers[int(transition[1]) - 1] - transition[2]
        trans_tuples.append([trans_tuple0, trans_tuple1])
    return trans_tuples


def spont_decay_fidelity(dimension: int = 25,
                         RO_time: float = 0.005,
                         D52_lifetime: float = 30.1):
    max_fids = np.empty(dimension)
    for it in range(dimension):
        if it == 0:
            max_fids[it] = 1
        else:
            max_f = np.exp(-it * RO_time / D52_lifetime)
            max_fids[it] = max_f

    return np.mean(max_fids)


def get_threshold(plot_threshold_histogram=True):

    # Specify the directory to traverse
    directory_path = (
        '/home/nicholas/Documents/Barium137_Qudit/data/25_level_5K_SPAM_data/')

    # List to hold all file paths
    file_list = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_list.append(os.path.join(root, file))

    # get pmt counts for thresholding
    pmt_counts = []
    for it in range(len(file_list)):
        with open(file_list[it], 'r') as filename:
            data = pd.read_csv(
                filename,
                delimiter="\[|\]|,|\"",  # noqa: W605
                engine='python',
                header=None).values
            data = data[:, 4:-19]
            pmt_counts += list(np.ravel(data))
            SPAM_dimension = data.shape[1] - 1
    # Count the occurrences of each entry
    entry_counts = Counter(pmt_counts)
    # Extract unique entry values and their corresponding counts
    array_counts = np.array(
        [entry_counts.get(key, 0) for key in range(max(entry_counts) + 1)])
    counts_dummy = np.arange(0, len(array_counts), 1)
    mask = counts_dummy < 20
    subset_array_counts = array_counts[mask]

    threshold = np.argmin(subset_array_counts)
    print(f'\n Found threshold is: {threshold}.')

    if plot_threshold_histogram:
        fig, ax = plt.subplots(figsize=set_size(width='half'))

        # Plot the histogram
        ax.bar(counts_dummy, array_counts, label='Original Histogram')
        ax.vlines(threshold,
                  min(array_counts),
                  max(array_counts),
                  colors='k',
                  linestyles='--',
                  label='Threshold')
        ax.set_yscale('log')
        ax.set_title('Fluorescence Readout Histogram')
        ax.set_ylabel('Occurences')
        ax.set_xlabel('PMT Counts')
        ax.legend()

    return threshold, SPAM_dimension


def get_spam_result(init_ket: int,
                    SPAM_dimension: int = 25,
                    threshold: int = 9,
                    num_exps: int = 1000,
                    post_processing: bool = True,
                    verbose: bool = False,
                    herald_measurement: bool = False,
                    get_best_run=False,
                    get_top_five=False,
                    reversed_readout: bool = False):

    file_path = (
        '/home/nicholas/Documents/' +
        f'Barium137_Qudit/data/25_level_5K_SPAM_data/InitKet_{init_ket}/')
    file_prefix = 'SPAM_10k_Heralded_'
    # date = date_time.split('_')
    # date_obj = datetime.strptime(date[0], '%Y%m%d')
    # plot_date = date_obj.strftime('%B %d, %Y')

    files = os.listdir(file_path)
    matching_files = []
    for fname in files:
        # Construct the full path to the file
        full_path = os.path.join(file_path, fname)

        # Check if the file exists and is a regular file
        if os.path.isfile(full_path):
            # Check if the search string exists in the file name
            if file_prefix in fname:
                matching_files.append(fname)

    percentage_data_used = []
    init_errors = []
    deshelve_errors = []
    spam_data = np.zeros((SPAM_dimension, len(matching_files)))
    for it in range(len(matching_files)):
        with open(file_path + matching_files[it], 'r') as filename:
            data = pd.read_csv(
                filename,
                delimiter="\[|\]|,|\"",  # noqa: W605
                engine='python',
                header=None).values
            data = data[:, 4:-19]

        # Implementing DYNAMIC thresholding
        pmt_counts = list(np.ravel(data))
        # Count the occurrences of each entry
        entry_counts = Counter(pmt_counts)
        # Extract unique entry values and their corresponding counts
        array_counts = np.array(
            [entry_counts.get(key, 0) for key in range(max(entry_counts) + 1)])
        counts_dummy = np.arange(0, len(array_counts), 1)
        mask = counts_dummy < 20
        subset_array_counts = array_counts[mask]
        threshold = np.argmin(subset_array_counts)
        # print(f'Found threshold is: {threshold}.')

        binary_data = data > threshold
        binary_data = binary_data[:, :SPAM_dimension + 1]

        # Initialize a mask to track the first occurrence of True in each row
        first_true_bin_array = np.zeros_like(binary_data, dtype=bool)

        # Iterate through each row
        num_shelve_error = 0
        num_deshelve_error = 0
        for row_idx in range(binary_data.shape[0]):
            row = binary_data[row_idx]
            first_true_idx = np.argmax(row)  # Find the index of the first True
            # print(row)
            # print(first_true_idx)
            # break
            if post_processing:
                if sum(row) > 0 and first_true_idx:
                    first_true_bin_array[row_idx, first_true_idx] = True
                elif sum(row) == 0:
                    num_deshelve_error += 1
                elif first_true_idx == 0:
                    num_shelve_error += 1
            else:
                first_true_bin_array[row_idx, first_true_idx] = True

        if herald_measurement:
            summed_bin_array = np.sum(first_true_bin_array, axis=0) / (
                first_true_bin_array.shape[0] - num_shelve_error -
                num_deshelve_error)
            data_rate = np.round(
                100 * (first_true_bin_array.shape[0] - num_shelve_error -
                       num_deshelve_error) / num_exps, 3)
        else:
            summed_bin_array = np.sum(first_true_bin_array, axis=0) / (
                first_true_bin_array.shape[0] - num_shelve_error)
            data_rate = np.round(
                100 * (first_true_bin_array.shape[0] - num_shelve_error) /
                num_exps, 3)

        spam_data[:, it] = summed_bin_array[1:]
        # print(f'Ket number {it}, fidelity: ' +
        #       f'{np.round(100*summed_bin_array[it+1],2)},
        #          fails: {num_fails}')
        state_id = np.argmax(summed_bin_array)
        state_fid = np.round(100 * summed_bin_array[state_id], 3)
        percentage_data_used.append(data_rate)
        init_errors.append(num_shelve_error)
        deshelve_errors.append(num_deshelve_error)
        if verbose:
            print(f'|{state_id-1}>, fidelity: {state_fid},' +
                  f' init. err: {num_shelve_error}, ' +
                  f' meas. err: {num_deshelve_error},' +
                  f' data rate: {data_rate},' +
                  f' file: {matching_files[it][-16:]}')

    if reversed_readout:
        percentage_data_used = np.concatenate(
            ([percentage_data_used[0]], percentage_data_used[1:][::-1]))
        # We need to flip the RO order and the order of kets to keep consistent
        # Flip the order of the columns (excluding the first column)
        flipped_columns = np.copy(
            spam_data)  # Create a copy of the original array
        # Reverse the columns from the second column onwards
        flipped_columns[:, 1:] = spam_data[:, 1:][:, ::-1]
        # Reverse the order of the rows (except first entry)
        result = flipped_columns[::-1, :]
        result = np.vstack((result[-1, :], result[:-1, :]))
        spam_data = result

    if get_top_five:
        # sort and take just the top 4 outcomes
        spam_data = spam_data[:, np.argsort(spam_data[init_ket])][:, -5:]

    # Write in the average SPAM fidelity
    if get_best_run:
        # Write in the average SPAM fidelity
        fidelities = spam_data[:, np.argmax(spam_data[init_ket])]
        fidelities_error = np.sqrt(
            (fidelities) * (1 - fidelities) / (len(matching_files) * num_exps))
        avg_data_rate = np.round(np.mean(percentage_data_used), 3)
    else:
        # Write in the average SPAM fidelity
        fidelities = np.mean(spam_data, axis=1)
        fidelities_error = np.sqrt(
            (fidelities) * (1 - fidelities) / (len(matching_files) * num_exps))
        avg_data_rate = np.round(np.mean(percentage_data_used), 3)

    if verbose:
        print('\n')
        print('Average fidelity', fidelities[init_ket])
        print('Average fidelity uncertainty', fidelities_error[init_ket])
        print('Average data rate', avg_data_rate)
        print('Init errors total', np.sum(init_errors))
        print(
            'Init error rate',
            np.round(
                100 * (np.sum(init_errors) / (len(matching_files) * num_exps)),
                2))
        print('Deshelve errors total', np.sum(deshelve_errors))
        print(
            'Deshelve error rate',
            np.round(
                100 * (np.sum(deshelve_errors) /
                       (len(matching_files) * num_exps)), 2), '\n')

    return (fidelities, fidelities_error, init_errors, deshelve_errors,
            avg_data_rate, len(matching_files))


def get_SPAM_data(num_exps: int = 1000,
                  threshold: int = 10,
                  SPAM_dimension: int = 25,
                  post_processing: bool = True,
                  plot_2D: bool = True,
                  plot_3D: bool = False,
                  plotting: bool = True,
                  plot_infidelity: bool = False,
                  reversed_readout: bool = False,
                  plot_threshold_histogram: bool = False,
                  flipped_RO_order: bool = False,
                  do_state_choice_plot: bool = False,
                  herald_measurement: bool = False,
                  get_best_run=False,
                  get_top_five=False,
                  verbose: bool = False):

    spam_data = np.empty((SPAM_dimension, SPAM_dimension))
    spam_data_err = np.empty((SPAM_dimension, SPAM_dimension))
    total_data_rate = np.empty(SPAM_dimension)
    init_errors = np.empty(SPAM_dimension)
    meas_errors = np.empty(SPAM_dimension)
    total_chunks = np.empty(SPAM_dimension)

    for itera in range(0, SPAM_dimension):
        fid, fid_err, init_err, meas_err, data_rate, chunks = get_spam_result(
            itera,
            SPAM_dimension=SPAM_dimension,
            verbose=verbose,
            herald_measurement=herald_measurement,
            get_best_run=get_best_run,
            get_top_five=get_top_five,
            threshold=threshold)

        spam_data[itera, :] = fid
        spam_data_err[itera, :] = fid_err
        total_data_rate[itera] = data_rate
        init_errors[itera] = np.sum(init_err)
        meas_errors[itera] = np.sum(meas_err)
        total_chunks[itera] = chunks

        print(f'|{itera}>, fidelity: {np.round(100*fid[itera],3)},' +
              f' init. err: {np.sum(init_err)}, ' +
              f' meas. err: {np.sum(meas_err)},' +
              f' data rate: {data_rate},' + f' chunks: {chunks}')

    # Write in the average SPAM fidelity
    fidelities = spam_data.diagonal()
    fidelities_error = np.empty(len(fidelities))

    for itera, fid in enumerate(fidelities):
        fidelities_error[itera] = np.sqrt(fid * (1 - fid) /
                                          (total_data_rate[itera] * num_exps))

    avg_fidelity = np.round(100 * np.mean(fidelities), 2)
    avg_fidelity_error = np.round(
        avg_fidelity * np.sqrt(np.sum((fidelities_error / fidelities)**2)), 3)

    avg_data_rate = np.round(np.mean(total_data_rate), 3)
    print('\n')
    print('Average fidelity', avg_fidelity)
    print('Average fidelity uncertainty', avg_fidelity_error)
    print('Average data rate', avg_data_rate)
    print('Init errors total', np.sum(init_errors))
    print(
        'Init error rate',
        np.round(100 * (np.sum(init_errors) / (len(total_chunks) * num_exps)),
                 2))
    print('Deshelve errors total', np.sum(meas_errors))
    print(
        'Deshelve error rate',
        np.round(100 * (np.sum(meas_errors) / (len(total_chunks) * num_exps)),
                 2), '\n')

    if do_state_choice_plot:

        sorting_indices_lowhigh = np.argsort(fidelities)
        sorting_indices_highlow = sorting_indices_lowhigh[::-1]

        optimal = fidelities[sorting_indices_highlow]
        worst = fidelities[sorting_indices_lowhigh]

        optimal_choice = []
        worst_choice = []
        optimal_choice_error_upper = []
        optimal_choice_error_lower = []
        worst_choice_error_upper = []
        worst_choice_error_lower = []
        qudit_dimensions = np.arange(2, 26, 1)

        for idx, dim in enumerate(qudit_dimensions):
            optimal_choice.append(np.mean(optimal[:idx + 2]))
            worst_choice.append(np.mean(worst[:idx + 2]))

            optimal_error_lower, optimal_error_upper = find_errors(
                1, np.mean(optimal[:idx + 2]), num_exps * dim)

            optimal_choice_error_upper.append(optimal_error_upper)
            optimal_choice_error_lower.append(optimal_error_lower)

            worst_error_lower, worst_error_upper = find_errors(
                1, np.mean(worst[:idx + 2]), num_exps * dim)

            worst_choice_error_upper.append(worst_error_upper)
            worst_choice_error_lower.append(worst_error_lower)

        optimal_choice = np.array(optimal_choice)
        optimal_choice_error_upper = np.array(optimal_choice_error_upper)
        optimal_choice_error_lower = np.array(optimal_choice_error_lower)

        worst_choice = np.array(worst_choice)
        worst_choice_error_upper = np.array(worst_choice_error_upper)
        worst_choice_error_lower = np.array(worst_choice_error_lower)

        fig, ax = plt.subplots(figsize=set_size(width='half'))

        ax.errorbar(qudit_dimensions,
                    optimal_choice,
                    fmt='o',
                    label='Optimal Choice',
                    yerr=[
                        np.abs(optimal_choice - optimal_choice_error_lower),
                        np.abs(optimal_choice_error_upper - optimal_choice)
                    ])
        ax.errorbar(qudit_dimensions,
                    worst_choice,
                    fmt='o',
                    label='Worst Choice',
                    yerr=[
                        np.abs(worst_choice - worst_choice_error_lower),
                        np.abs(worst_choice_error_upper - worst_choice)
                    ])
        ax.set_title('Qudit SPAM Fidelity vs. Dimension')
        ax.set_ylabel('Fidelity')
        ax.set_xlabel('Qudit Dimension')
        ax.legend()
        ax.grid()

    init_error_rate = np.array(init_errors) / num_exps
    meas_error_rate = np.array(meas_errors) / num_exps
    combined_data_rate = 1 - init_error_rate - meas_error_rate

    return spam_data, combined_data_rate, init_error_rate, meas_error_rate


def get_off_res_scat_fidelity(x_sec_freq_guess,
                              y_sec_freq_guess,
                              z_sec_freq_guess,
                              pitimes,
                              frequencies,
                              trans_tuples,
                              s12_init_fidelity,
                              LD_param,
                              avg_phonons,
                              create_plot=False):

    shelving_errors = np.zeros((24, 24))

    def Rabi_probability(Omega, delta, pulse_time):
        return (Omega**2 / (Omega**2 + delta**2)) * (np.sin(
            2 * np.pi * np.sqrt(Omega**2 + delta**2) * pulse_time / 2))**2

    for idx, tup in enumerate(trans_tuples):
        # print(tup)
        # print(frequencies[tup[0], tup[1]])
        # print(pitimes[tup[0], tup[1]])
        pulse_time = pitimes[tup[0], tup[1]]
        chosen_frequency = frequencies[tup[0], tup[1]]

        error = np.zeros(24)
        for idx2, tup2 in enumerate(trans_tuples):
            for i in [0, 1, 2, 3, 4]:
                if frequencies[tup2[0], i] != 0:
                    comparing_frequency = frequencies[tup2[0], i]

                    if tup[1] == i:
                        prefactor = s12_init_fidelity
                    else:
                        # assuming init infidelity is evenly split
                        # amongst s12 states
                        prefactor = (1 - s12_init_fidelity) / 4

                    Omega_trans = 1 / (2 * pitimes[tup2[0], i])
                    delta_carrier = chosen_frequency - comparing_frequency
                    delta_sbpx = (chosen_frequency - comparing_frequency +
                                  x_sec_freq_guess)
                    delta_sbnx = (chosen_frequency - comparing_frequency -
                                  x_sec_freq_guess)
                    delta_sbpy = (chosen_frequency - comparing_frequency +
                                  y_sec_freq_guess)
                    delta_sbny = (chosen_frequency - comparing_frequency -
                                  y_sec_freq_guess)
                    delta_sbpz = (chosen_frequency - comparing_frequency +
                                  z_sec_freq_guess)
                    delta_sbnz = (chosen_frequency - comparing_frequency -
                                  z_sec_freq_guess)

                    error[idx2] += prefactor * Rabi_probability(
                        Omega_trans, delta_carrier, pulse_time)
                    error[idx2] += prefactor * Rabi_probability(
                        LD_param * np.sqrt(avg_phonons + 1) * Omega_trans,
                        delta_sbpx, pulse_time)
                    error[idx2] += prefactor * Rabi_probability(
                        LD_param * np.sqrt(avg_phonons - 1) * Omega_trans,
                        delta_sbnx, pulse_time)
                    error[idx2] += prefactor * Rabi_probability(
                        LD_param * np.sqrt(avg_phonons + 1) * Omega_trans,
                        delta_sbpy, pulse_time)
                    error[idx2] += prefactor * Rabi_probability(
                        LD_param * np.sqrt(avg_phonons - 1) * Omega_trans,
                        delta_sbny, pulse_time)
                    error[idx2] += prefactor * Rabi_probability(
                        LD_param * np.sqrt(avg_phonons + 1) * Omega_trans,
                        delta_sbpz, pulse_time)
                    error[idx2] += prefactor * Rabi_probability(
                        LD_param * np.sqrt(avg_phonons - 1) * Omega_trans,
                        delta_sbnz, pulse_time)
                else:
                    pass

        shelving_errors[idx, :] = (error / np.sum(error))
        # shelving_errors[idx, idx] = 0
    # print(shelving_errors)

    # now the deshelving error
    deshelving_errors = np.zeros((24, 24))
    for idx, tup in enumerate(trans_tuples):
        # print(tup)
        # print(frequencies[tup[0], tup[1]])
        # print(pitimes[tup[0], tup[1]])

        error = np.zeros(24)
        # need to get frequencies for transitions from our given D52 state

        # for chosen_freq in d52_state_freqs:

        for idx2, tup2 in enumerate(trans_tuples):
            pulse_time = pitimes[tup2[0], tup2[1]]
            Omega_trans = 1 / (2 * pitimes[tup2[0], tup2[1]])
            deshelving_frequency = frequencies[tup2[0], tup2[1]]

            for i in [0, 1, 2, 3, 4]:
                if frequencies[tup[0], i] != 0:
                    comparing_frequency = frequencies[tup[0], i]

                    delta_carrier = (comparing_frequency -
                                     deshelving_frequency)
                    delta_sbpx = (comparing_frequency - deshelving_frequency +
                                  x_sec_freq_guess)
                    delta_sbnx = (comparing_frequency - deshelving_frequency -
                                  x_sec_freq_guess)
                    delta_sbpy = (comparing_frequency - deshelving_frequency +
                                  y_sec_freq_guess)
                    delta_sbny = (comparing_frequency - deshelving_frequency -
                                  y_sec_freq_guess)
                    delta_sbpz = (comparing_frequency - deshelving_frequency +
                                  z_sec_freq_guess)
                    delta_sbnz = (comparing_frequency - deshelving_frequency -
                                  z_sec_freq_guess)

                    error[idx2] += Rabi_probability(Omega_trans, delta_carrier,
                                                    pulse_time)
                    error[idx2] += Rabi_probability(
                        LD_param * np.sqrt(avg_phonons + 1) * Omega_trans,
                        delta_sbpx, pulse_time)
                    error[idx2] += Rabi_probability(
                        LD_param * np.sqrt(avg_phonons - 1) * Omega_trans,
                        delta_sbnx, pulse_time)
                    error[idx2] += Rabi_probability(
                        LD_param * np.sqrt(avg_phonons + 1) * Omega_trans,
                        delta_sbpy, pulse_time)
                    error[idx2] += Rabi_probability(
                        LD_param * np.sqrt(avg_phonons - 1) * Omega_trans,
                        delta_sbny, pulse_time)
                    error[idx2] += Rabi_probability(
                        LD_param * np.sqrt(avg_phonons + 1) * Omega_trans,
                        delta_sbpz, pulse_time)
                    error[idx2] += Rabi_probability(
                        LD_param * np.sqrt(avg_phonons - 1) * Omega_trans,
                        delta_sbnz, pulse_time)
                else:
                    pass

        deshelving_errors[idx, :] = (error / np.sum(error))
        # Initialize a 24x24 array with ones and 0.02 to simulate states
        # after we deshelve and readout the intended state in SPAM
        help_array = np.ones((24, 24))

        # Update the array based on the condition
        for i in range(24):
            for j in range(24):
                if j > i:
                    help_array[i, j] = 0.1
        deshelving_errors = deshelving_errors * help_array
    # print(deshelving_errors)

    off_res_scattering_matrix = (shelving_errors + deshelving_errors -
                                 shelving_errors * deshelving_errors)
    for idx, row in enumerate(off_res_scattering_matrix):
        if sum(row) == 0:
            off_res_scattering_matrix[idx, :] = row
        else:
            off_res_scattering_matrix[idx, :] = row / sum(row)
    off_res_scattering_matrix = off_res_scattering_matrix[:len(trans_tuples), :
                                                          len(trans_tuples)]
    avg_diagonal = np.trace(
        off_res_scattering_matrix) / off_res_scattering_matrix.shape[0]

    if create_plot:

        # plot total off_resonant scattering error from fit
        fig, ax = plt.subplots(figsize=set_size(width='full', square=True))

        # main plot
        # Plot the color map
        cmap = plt.get_cmap("viridis")
        im = ax.imshow(off_res_scattering_matrix, cmap=cmap,
                       origin="lower")  # Set origin to lower left corner
        cbar = plt.colorbar(im, ax=ax,
                            label="Probability")  # Add color bar with label
        cbar.set_label('Probability')

        # Create the list of x tick labels
        xtick_labels = [rf'$|{i}\rangle$' for i in range(1, 25)]
        x_positions = np.arange(24)  # Assuming 25 bars
        ax.set_xticks(x_positions[::2])
        ax.set_xticklabels(xtick_labels[::2],
                           rotation=0,
                           ha='center',
                           fontsize=10)

        # Create the list of y tick labels
        ytick_labels = [rf'$|{i}\rangle$' for i in range(1, 25)]
        y_positions = np.arange(24)  # Assuming 25 bars
        ax.set_yticks(y_positions[::2])
        ax.set_yticklabels(ytick_labels[::2],
                           rotation=0,
                           ha='right',
                           fontsize=10)

        # Set labels and title
        ax.set_ylabel('Prepared State')
        ax.set_xlabel('Measured State')
        ax.set_title('Off Resonant Scattering Error')

        text_threshold = 0.005
        # Add text annotations for values above a certain threshold
        for i in range(off_res_scattering_matrix.shape[0]):
            for j in range(off_res_scattering_matrix.shape[1]):
                value = off_res_scattering_matrix[i, j]
                if value > text_threshold:
                    ax.text(j,
                            i,
                            str(np.round(100 * value, 2)),
                            color="black"
                            if value > (np.max(off_res_scattering_matrix) /
                                        2) else "white",
                            fontsize=8,
                            ha="center",
                            va="center")

    return avg_diagonal


def find_optimal_secular_frequencies(
        # spam_data,
        measurement_order,
        B_field: float = 4.209,
        insensB: float = 545.381349,
        sensB: float = 623.181113,
        x_sec_freq_input: float = 1.163,
        y_sec_freq_input: float = 1.357,
        z_sec_freq_input: float = 0.142,
        LD_param: float = 0.014,
        avg_phonons: float = 140,
        s12_init_fidelity: float = 0.98,
        turn_on_D52_decay=False,
        resolution=0.002,
        size=301,
        save_file=False,
        plot_measured_sec_freqs=True,
        verbose=False):

    # the goal of this function is to find the optimal set of secular
    # frequencies for SPAM that minimises the off-resonant scattering rate

    # generate the arrays of secular frequencies
    x_sec_freqs_array = np.linspace(
        x_sec_freq_input - (resolution * (size // 2)),
        x_sec_freq_input + (resolution * (size // 2)), size)
    y_sec_freqs_array = np.linspace(
        y_sec_freq_input - (resolution * (size // 2)),
        y_sec_freq_input + (resolution * (size // 2)), size)

    if not save_file:
        # cal = Calibrations(F2M0f2m0=545.5756, use_old_data=False)
        # cal.run_calibrated_frequency_generation(insensB=insensB,
        #                                         sensB=sensB,
        #                                         print_result=False)

        # frequencies = cal.full_calibrated_transitions
        # pitimes = cal.theory_pitimes

        ba = Barium(B_field=B_field)
        ba.B_field_range = np.linspace(0.0001, B_field, 100)
        ba.generate_transition_frequencies(F2M0f2m0_input=545.2490)
        ba.generate_transition_strengths()
        frequencies = np.nan_to_num(ba.transition_frequencies, nan=0)
        pitimes = ba.transition_pitimes
        # pitimes = np.nan_to_num(ba.transition_pitimes,nan=0)

        ba.plot_mag_field(orbital='S12', estates_plot=False)
        ba.plot_mag_field(orbital='D52', estates_plot=False)

        # flatten transition frequencies array to find all
        # differences between all transition frequencies
        flattened_filtered_trans = ba.transition_frequencies[
            ~np.isnan(ba.transition_frequencies)]
        pairwise_diff = (flattened_filtered_trans[:, None] -
                         flattened_filtered_trans)

        mask = np.eye(len(flattened_filtered_trans), dtype=bool)
        pairwise_diff[mask] = np.inf
        # print(np.min(np.abs(pairwise_diff)))
        # plt.show()
        # breakpoint()
        # print(frequencies)
        # print(pitimes)

        trans_tuples = translate_transitions_to_indices(measurement_order)
        # print(trans_tuples)

        avg_fids_array = np.zeros(
            (len(x_sec_freqs_array), len(x_sec_freqs_array)))
        for x_idx, x_sec_freq in enumerate(x_sec_freqs_array):
            for y_idx, y_sec_freq in enumerate(y_sec_freqs_array):

                avg_fid = get_off_res_scat_fidelity(x_sec_freq, y_sec_freq,
                                                    z_sec_freq_input, pitimes,
                                                    frequencies, trans_tuples,
                                                    s12_init_fidelity,
                                                    LD_param, avg_phonons,
                                                    turn_on_D52_decay)
                # print(x_sec_freq)
                # print(y_sec_freq)
                # print(avg_fid, '\n')

                avg_fids_array[x_idx, y_idx] = avg_fid

        avg_fids_array = np.transpose(avg_fids_array)
        np.savetxt(f'25_level_scaling_for99p99/D{len(measurement_order)+1}_' +
                   'off_resonant_scattering_2D_plot.txt',
                   avg_fids_array,
                   delimiter=',')

    elif save_file:
        avg_fids_array = np.loadtxt('25_level_scaling_for99p99/' + save_file,
                                    delimiter=',')

    if verbose:
        # plot total off_resonant scattering error from fit
        fig, ax = plt.subplots(figsize=set_size(width='full', square=True))

        # main plot
        # Plot the color map
        # cmap = plt.get_cmap("viridis")
        im = plt.imshow(
            avg_fids_array,
            extent=[
                x_sec_freqs_array[0], x_sec_freqs_array[-1],
                y_sec_freqs_array[0], y_sec_freqs_array[-1]
            ],
            origin='lower',
            aspect='auto',
            cmap='viridis'
        )  # You can choose other colormaps like 'plasma', 'hot', etc.
        cbar = plt.colorbar(
            im, ax=ax, label="Average Fidelity")  # Add color bar with label
        cbar.set_label('Average Fidelity')

        if plot_measured_sec_freqs:

            directory_path = ("/home/nicholas/Documents/Barium137_Qudit/" +
                              "data/25_level_5K_SPAM_data/secular_frequencies")

            # Initialize lists to hold data from each column
            timestamps = []
            carrier, x_sideband, y_sideband, x_sec, y_sec = [], [], [], [], []

            # Open the files and read the data
            for filename in os.listdir(directory_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(directory_path, filename)
                    with open(file_path, 'r') as file:
                        reader = csv.reader(file)
                        for row in reader:
                            timestamps.append(row[0])
                            carrier.append(float(row[1]))
                            x_sideband.append(float(row[2]))
                            y_sideband.append(float(row[3]))
                            x_sec.append(float(row[4]))
                            y_sec.append(float(row[5]))

            print('y-secular freq. range:', np.max(y_sec) - np.min(y_sec))
            print('x-secular freq. range:', np.max(x_sec) - np.min(x_sec))

            plt.scatter(x_sec, y_sec, alpha=0.5, color='gray')
        # Draw lines to highlight apprx. position of our secular frequencies
        plt.axvline(x=x_sec_freq_input,
                    color='gray',
                    alpha=0.5,
                    linestyle='--',
                    linewidth=1)
        plt.axhline(y=y_sec_freq_input,
                    color='gray',
                    alpha=0.5,
                    linestyle='--',
                    linewidth=1)

        # Add line y=x to see where sec frequencies are with no x/y difference
        # min_val = max(
        #     x_sec_freqs_array[0],
        #     y_sec_freqs_array[0])  # Start from the lower bound of the data
        # max_val = min(
        #     x_sec_freqs_array[-1],
        #     y_sec_freqs_array[-1])  # End at the upper bound of the data
        # plt.plot([min_val, max_val], [min_val, max_val],
        #          color='gray',
        #          alpha=0.5,
        #          linestyle='-',
        #          linewidth=1)

        print(f'Max fidelity achieved: {np.max(avg_fids_array)}')
        print(f'Min infidelity achieved: {1-np.max(avg_fids_array)}')
        plt.xlabel('X Secular Frequencies/ MHz')
        plt.ylabel('Y Secular Frequencies/ MHz')
        plt.title('Average Fidelity vs. Secular Frequencies')

    return 1 - np.max(avg_fids_array)


def SPAM_error_scaling(datetime: str = '',
                       num_exps: int = 1001,
                       measurement_order: list = [],
                       darkbright_infidelity: float = 0.1e-3,
                       dimensions: np.ndarray = np.arange(2, 25, 1),
                       herald_measurement: bool = False,
                       get_best_run=False,
                       get_top_five=False,
                       use_saved_data: bool = True):

    # fidelities = []
    # for dim in dimensions:
    #     spam_data, combined_data_rate, init_err, deshelve_err =
    # get_SPAM_data(
    #         SPAM_dimension=dim,
    #         herald_measurement=herald_measurement,
    #         get_best_run=get_best_run,
    #         get_top_five=get_top_five)
    #     fid_dim = spam_data[:, :].diagonal()
    #     fidelities.append(np.mean(fid_dim))
    # fidelities = np.array(fidelities)

    # calculate pulse infidelities when not heralding out the de-shelving error
    if not herald_measurement:
        cal = Calibrations(F2M0f2m0=545.5756, use_old_data=False)
        cal.generate_frequencies_sensitivities_table()
        cal.generate_sensitivities_pitimes_table()
        sensitivities = cal.transition_sensitivities
        frequencies = cal.transition_frequencies
        pitimes = cal.transition_pitimes
        meas_order_indices = cal.convert_triplet_index([[-1, 4, -3]] +
                                                       measurement_order)

        freqs = []
        sens = []
        pitims = []
        pulse_infids = []
        pulse_infid_means = []
        for it, idx in enumerate(meas_order_indices):
            freqs.append(frequencies[idx[0], idx[1]])
            sens.append(sensitivities[idx[0], idx[1]])
            pitims.append(pitimes[idx[0], idx[1]])
            pulse_infids.append(0.995 -
                                2.5e-7 * np.square(sens[it] * pitims[it]))
            if it > 0:
                pulse_infid_means.append(np.mean(pulse_infids))

        print(measurement_order)
        print(freqs)
        print(sens)
        print(pitims)
    else:
        pass

    # best run only, while heralding out deshelving
    fidelities = np.array([
        0.99928537, 0.99918468, 0.997794, 0.99728295, 0.99753205, 0.99738608,
        0.99737511, 0.99736808, 0.99750797, 0.99737903, 0.99721985, 0.99692606,
        0.99629956, 0.99611408, 0.99600424, 0.99566871, 0.99564726, 0.99564815,
        0.99550691, 0.9954374, 0.9953814, 0.99532681, 0.99520317, 0.99512924
    ])

    sorting_indices_lowhigh = np.argsort(fidelities)
    sorting_indices_highlow = sorting_indices_lowhigh[::-1]
    measurement_order = np.array(measurement_order)

    optimal = fidelities[sorting_indices_highlow]
    worst = fidelities[sorting_indices_lowhigh]

    meas_highlow = measurement_order[sorting_indices_highlow]
    print('Sorted SPAM orders:', meas_highlow)

    optimal_choice = []
    optimal_choice_error_upper = []
    optimal_choice_error_lower = []

    worst_choice = []
    worst_choice_error_upper = []
    worst_choice_error_lower = []

    meas_infidelities = []
    infid_errors_lower = []
    infid_errors_upper = []

    off_res_err = []
    spont_decay_err = []

    for idx, dim in enumerate(dimensions):

        print(f'Starting dimension {dim} analysis.')

        # do the optimal/worst choice processing first
        optimal_choice.append(np.mean(optimal[:idx + 1]))
        worst_choice.append(np.mean(worst[:idx + 1]))

        optimal_error_lower, optimal_error_upper = find_errors(
            1, np.mean(optimal[:idx + 1]), num_exps * dim)

        optimal_choice_error_upper.append(optimal_error_upper)
        optimal_choice_error_lower.append(optimal_error_lower)

        worst_error_lower, worst_error_upper = find_errors(
            1, np.mean(worst[:idx + 1]), num_exps * dim)

        worst_choice_error_upper.append(worst_error_upper)
        worst_choice_error_lower.append(worst_error_lower)

        if use_saved_data:
            save_file = f'D{dim}_off_resonant_scattering_2D_plot.txt'
        else:
            save_file = None

        # print(fidelities)
        # print(1- np.mean(fidelities[:dim-1]))

        # now for the dimension dependent off-res scattering error
        infidelity = find_optimal_secular_frequencies(measurement_order[:dim -
                                                                        1],
                                                      insensB=545.833696,
                                                      sensB=623.635682,
                                                      x_sec_freq_input=1.271,
                                                      y_sec_freq_input=1.466,
                                                      z_sec_freq_input=0.215,
                                                      LD_param=0.024,
                                                      avg_phonons=200,
                                                      s12_init_fidelity=1,
                                                      B_field=4.216,
                                                      resolution=0.002,
                                                      size=20,
                                                      verbose=False,
                                                      save_file=save_file)

        off_res_err.append(infidelity)
        spont_decay_err.append(1 - spont_decay_fidelity(dimension=dim))
        darkbright_infidelities = [
            np.mean((np.arange(1, 25))[:idx + 1] * darkbright_infidelity)
            for idx in np.arange(1, 25)
        ]

        print('States subset:', measurement_order[:dim - 1])
        print('Off resonant scattering error,', infidelity)
        print('Spontaneous decay error',
              1 - spont_decay_fidelity(dimension=dim))
        print('Dark/bright discrimination error,', darkbright_infidelities[-1])

        meas_infidelities.append(np.mean(1 - fidelities[:idx + 1]))

        infid_err_lower, infid_err_upper = find_errors(
            1, np.mean(1 - fidelities[:idx + 1]), num_exps * dim)

        infid_errors_lower.append(infid_err_lower)
        infid_errors_upper.append(infid_err_upper)

    optimal_choice = np.array(optimal_choice)
    optimal_choice_error_upper = np.array(optimal_choice_error_upper)
    optimal_choice_error_lower = np.array(optimal_choice_error_lower)

    worst_choice = np.array(worst_choice)
    worst_choice_error_upper = np.array(worst_choice_error_upper)
    worst_choice_error_lower = np.array(worst_choice_error_lower)

    meas_infidelities = np.array(meas_infidelities)
    infid_errors_upper = np.array(infid_errors_upper)
    infid_errors_lower = np.array(infid_errors_lower)

    # plot the worst/best choice comparison figure
    fig, ax = plt.subplots(figsize=set_size(width='half'))
    ax.errorbar(dimensions,
                optimal_choice,
                fmt='o',
                label='Optimal Choice',
                yerr=[
                    np.abs(optimal_choice - optimal_choice_error_lower),
                    np.abs(optimal_choice_error_upper - optimal_choice)
                ])
    ax.errorbar(dimensions,
                worst_choice,
                fmt='o',
                label='Worst Choice',
                yerr=[
                    np.abs(worst_choice - worst_choice_error_lower),
                    np.abs(worst_choice_error_upper - worst_choice)
                ])
    ax.set_title('Qudit SPAM Fidelity vs. Dimension')
    ax.set_ylabel('Fidelity')
    ax.set_xlabel('Qudit Dimension')
    ax.legend()
    ax.grid()

    fig, ax = plt.subplots(figsize=set_size(width='half'))
    # data
    ax.errorbar(dimensions,
                1 - fidelities,
                fmt='o-',
                label='Measurement',
                color=colors[0],
                yerr=[
                    np.abs(meas_infidelities - infid_errors_lower),
                    np.abs(infid_errors_upper - meas_infidelities)
                ])

    # ax.errorbar(dimensions,
    #             1 - optimal_choice_infidelities[:dimensions],
    #             fmt='o',
    #             label='Avg. SPAM Infidelity',
    #             yerr=1 - optimal_choice_infid_errors)

    # off-res error
    ax.scatter(dimensions,
               off_res_err,
               label='Off-Resonant Driving',
               color=colors[2],
               alpha=0.25)
    ax.scatter(dimensions,
               spont_decay_err,
               label='Spontaneous Decay',
               color=colors[3],
               alpha=0.25)

    # print(darkbright_infidelities)
    ax.scatter(dimensions,
               darkbright_infidelities,
               alpha=0.25,
               color=colors[4],
               label='Readout Infidelity')
    if herald_measurement:
        total_est_err = darkbright_infidelities + np.array(
            off_res_err) + np.array(spont_decay_err)
    else:
        total_est_err = (dimensions - 1) * np.array(
            darkbright_infidelity) + np.array(off_res_err) + np.array(
                spont_decay_err) + (1 - np.array(pulse_infid_means))

        ax.scatter(dimensions,
                   1 - np.array(pulse_infid_means),
                   alpha=0.25,
                   color=colors[5],
                   label='De-shelving Infidelity')

    ax.plot(dimensions,
            total_est_err,
            label='Total Error',
            linestyle='--',
            color='black',
            alpha=0.5)

    ax.set_title('Qudit SPAM Infidelity and Errors vs. Dimension')
    ax.set_ylabel('Infidelity')
    ax.set_xlabel('Qudit Dimension')
    ax.legend()
    ax.grid()

    fig.savefig('25_level_scaling_for99p99/25_level_scaling_error.pdf')
    return None


# def SPAM_error_scaling(datetime: str = '',
#                        num_exps: int = 1001,
#                        measurement_order: list = [],
#                        measurement_infidelity: float = 0.056e-3,
#                        dimensions: np.ndarray = np.arange(2, 25, 1),
#                        get_best_run=False,
#                        get_top_five=False,
#                        use_saved_data: bool = True):

#     spam_data, combined_data_rate, init_err, deshelve_err = get_SPAM_data(
#         get_best_run=get_best_run, get_top_five=get_top_five)
#     fidelities = spam_data[1:, 1:].diagonal()

#     sorting_indices_lowhigh = np.argsort(fidelities)
#     sorting_indices_highlow = sorting_indices_lowhigh[::-1]
#     measurement_order = np.array(measurement_order)

#     optimal = fidelities[sorting_indices_highlow]
#     worst = fidelities[sorting_indices_lowhigh]

#     meas_highlow = measurement_order[sorting_indices_highlow]
#     print('Sorted SPAM orders:', meas_highlow)

#     optimal_choice = []
#     optimal_choice_error_upper = []
#     optimal_choice_error_lower = []

#     worst_choice = []
#     worst_choice_error_upper = []
#     worst_choice_error_lower = []

#     optimal_choice_infidelities = []
#     infid_errors_lower = []
#     infid_errors_upper = []

#     optimal_choice_off_res = []
#     optimal_choice_spont_decay = []

#     for idx, dim in enumerate(dimensions):

#         print(f'Starting dimension {dim} analysis.')

#         # do the optimal/worst choice processing first
#         optimal_choice.append(np.mean(optimal[:idx + 1]))
#         worst_choice.append(np.mean(worst[:idx + 1]))

#         optimal_error_lower, optimal_error_upper = find_errors(
#             1, np.mean(optimal[:idx + 1]), num_exps * dim)

#         optimal_choice_error_upper.append(optimal_error_upper)
#         optimal_choice_error_lower.append(optimal_error_lower)

#         worst_error_lower, worst_error_upper = find_errors(
#             1, np.mean(worst[:idx + 1]), num_exps * dim)

#         worst_choice_error_upper.append(worst_error_upper)
#         worst_choice_error_lower.append(worst_error_lower)

#         if use_saved_data:
#             save_file = f'D{dim}_off_resonant_scattering_2D_plot.txt'
#         else:
#             save_file = None

#         # now for the dimension dependent off-res scattering error
#         infidelity = find_optimal_secular_frequencies(spam_data,
#                                                       meas_highlow[:dim - 1],
#                                                       insensB=545.833696,
#                                                       sensB=623.635682,
#                                                       x_sec_freq_input=1.185,
#                                                       y_sec_freq_input=1.505,
#                                                       z_sec_freq_input=0.215,
#                                                       LD_param=0.024,
#                                                       avg_phonons=200,
#                                                       s12_init_fidelity=0.98,
#                                                       B_field=4.216,
#                                                       resolution=0.002,
#                                                       size=51,
#                                                       save_file=save_file)

#         optimal_choice_off_res.append(infidelity)
#         optimal_choice_spont_decay.append(1 -
#                                           spont_decay_fidelity(dimension=dim))

#         print('Optimal states chosen:', meas_highlow[:dim-1])
#         print('Off resonant scattering error,', infidelity)
#         print('Spontaneous decay error',
#               1 - spont_decay_fidelity(dimension=dim))

#         optimal_choice_infidelities.append(np.mean(1 - optimal[:idx + 1]))

#         infid_err_lower, infid_err_upper = find_errors(
#             1, np.mean(1 - optimal[:idx + 1]), num_exps * dim)

#         infid_errors_lower.append(infid_err_lower)
#         infid_errors_upper.append(infid_err_upper)

#     optimal_choice = np.array(optimal_choice)
#     optimal_choice_error_upper = np.array(optimal_choice_error_upper)
#     optimal_choice_error_lower = np.array(optimal_choice_error_lower)

#     worst_choice = np.array(worst_choice)
#     worst_choice_error_upper = np.array(worst_choice_error_upper)
#     worst_choice_error_lower = np.array(worst_choice_error_lower)

#     optimal_choice_infidelities = np.array(optimal_choice_infidelities)
#     infid_errors_upper = np.array(infid_errors_upper)
#     infid_errors_lower = np.array(infid_errors_lower)

#     # plot the worst/best choice comparison figure
#     fig, ax = plt.subplots(figsize=set_size(width='half'))
#     ax.errorbar(dimensions,
#                 optimal_choice,
#                 fmt='o',
#                 label='Optimal Choice',
#                 yerr=[
#                     np.abs(optimal_choice - optimal_choice_error_lower),
#                     np.abs(optimal_choice_error_upper - optimal_choice)
#                 ])
#     ax.errorbar(dimensions,
#                 worst_choice,
#                 fmt='o',
#                 label='Worst Choice',
#                 yerr=[
#                     np.abs(worst_choice - worst_choice_error_lower),
#                     np.abs(worst_choice_error_upper - worst_choice)
#                 ])
#     ax.set_title('Qudit SPAM Fidelity vs. Dimension')
#     ax.set_ylabel('Fidelity')
#     ax.set_xlabel('Qudit Dimension')
#     ax.legend()
#     ax.grid()

#     fig, ax = plt.subplots(figsize=set_size(width='half'))
#     # data
#     ax.errorbar(dimensions,
#                 optimal_choice_infidelities,
#                 fmt='o',
#                 label='Optimal Choice',
#                 color=colors[0],
#                 yerr=[
#                     np.abs(optimal_choice_infidelities - infid_errors_lower),
#                     np.abs(infid_errors_upper - optimal_choice_infidelities)
#                 ])
#     # ax.errorbar(dimensions,
#     #             1 - optimal_choice_infidelities[:dimensions],
#     #             fmt='o',
#     #             label='Avg. SPAM Infidelity',
#     #             yerr=1 - optimal_choice_infid_errors)

#     # off-res error
#     ax.scatter(dimensions,
#                optimal_choice_off_res,
#                label='Off-Resonant Driving',
#                color=colors[1],
#                alpha=0.25)
#     ax.scatter(dimensions,
#                optimal_choice_spont_decay,
#                label='Spontaneous Decay',
#                color=colors[2],
#                alpha=0.25)

#     plt.axhline(y=measurement_infidelity,
#                 color='gray',
#                 alpha=0.5,
#                 linestyle='--',
#                 linewidth=1,
#                 label='Readout Infidelity')

#     ax.scatter(dimensions,
#                np.array(measurement_infidelity) +
#                np.array(optimal_choice_off_res) +
#                np.array(optimal_choice_spont_decay),
#                label='Total Error',
#                color='black',
#                alpha=0.5)

#     ax.set_title('Qudit SPAM Infidelity and Errors vs. Dimension')
#     ax.set_ylabel('Infidelity')
#     ax.set_xlabel('Qudit Dimension')
#     ax.legend()
#     ax.grid()

#     fig.savefig('25_level_scaling_for99p99/25_level_scaling_error.pdf')
#     return None

if __name__ == '__main__':
    start_time = time.time()

    # order to identify each ket with a transition
    measurement_order = [[0, 4, 0], [-2, 4, -4], [-1, 4, -3], [0, 4, -2],
                         [-2, 3, 0], [1, 4, 3], [-2, 3, -3], [0, 3, 1],
                         [2, 4, 4], [2, 4, 2], [-2, 3, -2], [1, 3, 3],
                         [0, 4, -1], [0, 3, 2], [-2, 3, -1], [-2, 2, -1],
                         [-2, 2, -2], [2, 2, 1], [0, 2, 0], [0, 2, 2],
                         [0, 4, 1], [-1, 1, -1], [-2, 1, 0], [1, 1, 1]]

    ###########################################################################
    # # used to generate small maps of off-resonant driving for each
    # # dimension in the build up from D=2 to D=25 for scaling SPAM error
    B_value = 10.67
    for dim in np.arange(25, 26):
        # now for the dimension dependent off-res scattering error
        infidelity = find_optimal_secular_frequencies(
            measurement_order[:dim - 1],
            B_field=B_value,
            x_sec_freq_input=1.3,
            y_sec_freq_input=1.5,
            z_sec_freq_input=0.215,
            LD_param=0.014,
            avg_phonons=5,
            s12_init_fidelity=1,
            resolution=0.001,
            size=10,
            verbose=False,
            save_file=False)
    print('New B field of ', np.round(B_value, 2), 'infidelity',
            np.round(1e4 * infidelity, 3), '(in units of 10^-4).')

        # save_file='D25_off_resonant_scattering_2D_plot.txt')

    ###########################################################################
    # calculating the dark/bright discrimination fidelity
    darkbright_infidelity = 5e-4  # our current infidelity
    darkbright_infidelity = 2.5e-6  # target infidelity
    darkbright_infidelity = 7e-5  # what if we just want 1e-3?
    darkbright_infidelities = [
        np.mean((np.arange(1, 25))[:idx + 1] * darkbright_infidelity)
        for idx in np.arange(1, 25)
    ]
    print(
        f'\nThe dark/bright discrimination error of {darkbright_infidelity}' +
        f' yields {np.round(darkbright_infidelities[-1],6)} infid. for D=25.')

    ###########################################################################
    # D52 decay fidelity calculation for 99.99%
    RO_time = 0.005  # our current time
    RO_time = 0.00008  # target time
    RO_time = 0.0008  # what if we just need 1e-3?
    D52_limited_fid = spont_decay_fidelity(RO_time=RO_time, dimension=25)
    # print('The max fidelity limited by D52 decay:', D52_limited_fid)
    print(f'\nThe infidelity with RO time of {RO_time*1e6}us, D52 decay fid:',
          np.round(1 - D52_limited_fid, 6))

    # plt.show()

    print("--- %s seconds ---" % (time.time() - start_time))
