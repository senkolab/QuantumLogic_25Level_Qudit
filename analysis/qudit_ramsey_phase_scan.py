import ast
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from class_utils import Utils
from qudit_ramsey_unitary import QuditUnitary
from ramsey_pulse_finder import RamseyPulseFinder

find_errors = Utils.find_errors
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = colors * 4


def get_single_contrast(dimension: int = 3,
                        exp_num: int = 250,
                        U1_only_flag: bool = False):
    """Calculate the average population contrast for a specified dimension.

    The function reads data files corresponding to the specified dimension,
    processes the data to compute the average populations, and returns the
    averaged populations along with the total number of processed experiments.

    Parameters:
        dimension (int): The dimension to calculate the contrast for
    (default 3).
        U1_only_flag (bool): A flag indicating whether to filter for U1-only
    experiments (default is False).

    Returns:

        tuple: A tuple containing the averaged populations and the total number
        of processed experiments.

    """
    threshold = 8

    head_file_path = ('/home/nicholas/Documents/Barium137_Qudit' +
                      '/data/ramseys_phase_scans_dimensions_PM_data/')

    # List to store filtered file names
    filtered_file_names = []
    total_number_processed_experiments = 0

    # Iterate through files in the specified directory
    for file_name in os.listdir(head_file_path):
        # Check if both substrings are in the file name
        if (f"d={dimension}" in file_name) and (f"U1only_{U1_only_flag}"
                                                in file_name):
            modified_time = os.path.getmtime(head_file_path + file_name)
            # Add the matching file name and its modified time to the list
            filtered_file_names.append((file_name, modified_time))

    # Sort the list by modified time (the second element of the tuple)
    filtered_file_names.sort(key=lambda x: x[1])

    # Extract sorted file names from the tuples
    sorted_file_names = [file_name for file_name, _ in filtered_file_names]

    repeated_outcomes = []
    init_errors = []
    leakage_errors = []

    for file_name_PD in sorted_file_names:

        with open(head_file_path + file_name_PD, 'r') as hf_name:
            # iterate through lines to get the last one, which gives
            # a list of all raw data filenames associated with the run
            for line in hf_name:
                pass
            file_names = line.strip()

        # need to convert file_names to be a true list of strings
        file_names = ast.literal_eval(file_names)

        with open(head_file_path + file_name_PD, 'r') as hf_name:
            # get pulse times from the save txt data in head_file_name
            data = np.genfromtxt(hf_name, delimiter=',', skip_footer=9)
            wait_times = data[:, 0]

        file_path_raw = ('/home/nicholas/Documents/Barium137_Qudit' +
                         '/data/ramseys_phase_scans_dimensions_PM_data' +
                         '/raw_data_fluorescence/')

        first_file_last_part = file_names[0].rsplit('\\', 1)[-1]
        first_file_prefix_segment = file_names[0].rsplit('\\',
                                                         1)[0].rsplit('\\',
                                                                      1)[-1]
        first_file_file_name = (
            f'{first_file_prefix_segment}/{first_file_last_part}')
        # get the number of states by looking at the first data file
        with open(file_path_raw + first_file_file_name, 'r') as filename:
            data = pd.read_csv(filename,
                               delimiter="\[|\]|,|\"",
                               engine='python',
                               header=None).values
            data = data[:, 4:-19]
            data_dimension = np.shape(data)[1]

        full_data_array = np.zeros((data_dimension - 1, len(wait_times)))

        for idx, file_name in enumerate(file_names):

            last_part = file_name.rsplit('\\', 1)[-1]
            prefix_segment = file_name.rsplit('\\', 1)[0].rsplit('\\', 1)[-1]
            file_name = f'{prefix_segment}/{last_part}'

            with open(file_path_raw + file_name, 'r') as filename:
                data = pd.read_csv(
                    filename,
                    delimiter="\[|\]|,|\"",  # noqa: W605
                    engine='python',
                    header=None).values
                data = data[:, 4:-19]
            binary_data = data > threshold

            # Initialize a mask to track the first occurrence of True
            first_true_bin_array = np.zeros_like(binary_data, dtype=bool)

            # Iterate through each row
            num_shelve_error = 0
            num_deshelve_error = 0
            for row_idx in range(binary_data.shape[0]):
                # data_idx = data[row_idx]
                row = binary_data[row_idx]
                first_true_idx = np.argmax(
                    row)  # Find the index of the first True
                # print(row)
                # print(first_true_idx)
                # break
                if sum(row) > 0 and first_true_idx != 0:
                    first_true_bin_array[row_idx, first_true_idx] = True
                elif sum(row) == 0:
                    num_deshelve_error += 1
                elif first_true_idx == 0:
                    num_shelve_error += 1

            # print(file_name)
            summed_bin_array = np.sum(first_true_bin_array, axis=0) / (
                first_true_bin_array.shape[0] - num_shelve_error -
                num_deshelve_error)
            summed_bin_array = summed_bin_array[1:]
            full_data_array[:, idx] = summed_bin_array
            total_number_processed_experiments += first_true_bin_array.shape[
                0] - num_shelve_error - num_deshelve_error

            init_errors.append(num_shelve_error /
                               first_true_bin_array.shape[0])
            leakage_errors.append(num_deshelve_error /
                                  first_true_bin_array.shape[0])

            # print(file_name_PD)
            # print(summed_bin_array)

        repeated_outcomes.append(full_data_array)

    PD = repeated_outcomes[0]
    lower_error, upper_error = find_errors(1, PD, exp_num)
    asym_error_bar = np.abs([PD - lower_error, upper_error - PD])

    # averaged_populations = np.mean(np.array(repeated_outcomes), axis=0)
    # avg_init_errors = np.mean(np.array(init_errors))
    # avg_leakage_errors = np.mean(np.array(leakage_errors))
    # error_array = np.array([avg_init_errors, avg_leakage_errors])

    # quickly simulate the exact evolution as guide to the eye
    qd = QuditUnitary(dimension=dimension,
                      topology='Star',
                      line_amplitude_60=0,
                      line_amplitude_180=0,
                      line_phase_60=0,
                      line_phase_180=0,
                      line_offset=0,
                      mag_field_noise=0,
                      laser_noise_gaussian=0,
                      laser_noise_lorentzian=0,
                      pulse_time_miscalibration=0,
                      freq_miscalibration=0,
                      real_Ramsey_wait=0)
    phases, probs, U1_probs = qd.scan_phases()

    # print(probs)
    # need to fix colours so simulation and data match
    if dimension == 3:
        colour_fixer = [0, 3, 1]
    elif dimension == 4:
        colour_fixer = [0, 1, 3, 2]
    elif dimension == 5:
        colour_fixer = [0, 3, 4, 5, 6]
    elif dimension == 6:
        colour_fixer = [0, 3, 5, 4, 2, 7]
    elif dimension == 9:
        colour_fixer = [0, 7, 2, 9, 5, 3, 6, 10, 8]
    else:
        colour_fixer = np.arange(0, 20, 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    for idx in range(PD.shape[0]):
        # plotting just the |0> result for now (no colour fixer required)
        # for idx in range(1):
        # fix index number for runs with dimension>9
        ax.errorbar(2 * wait_times / 100,
                    PD[idx],
                    yerr=asym_error_bar[:, idx],
                    fmt='o',
                    label=rf'$|{idx}\rangle$',
                    color=colors[idx],
                    alpha=0.5)

        if dimension > 9:
            sim_idx = idx + 1
        else:
            sim_idx = idx
        ax.plot(phases / np.pi,
                probs[colour_fixer[sim_idx]],
                color=colors[idx],
                linestyle='--',
                alpha=1)
    ax.set_title(rf'Qudit ($d=$ {dimension}) Ramsey-style Phase Scan')
    ax.set_xlabel(r'Phase $\phi$ / $\pi$')
    ax.set_ylabel('Outcome Probability')
    ax.grid()
    ax.legend()

    plt.tight_layout()

    # return avg_populations, total_number_processed_experiments, error_array
    return repeated_outcomes


def plot_dimensional_contrast(dimensions: np.ndarray = np.arange(3, 8, 1),
                              topology: str = 'All',
                              plot_times_comparison: bool = False,
                              plot_U1_populations: bool = False):
    """Plot the contrast scaling with dimension.

    The function plots the average contrast of the |0⟩ state against the
    dimensions while optionally displaying populations after U_1, which should
    create an equal superposition. It can also plot the results as a function
    of the total Ramsey pulse sequence time.

    Parameters:

        dimensions (np.ndarray): The dimensions to plot (default is from 3 to
        7).

        plot_times_comparison (bool): A flag indicating whether to plot pulse
        times (default is False).

        plot_U1_populations (bool): A flag indicating whether to plot U1
        populations (default is False).

    Returns:
        tuple: A tuple containing the dimensions and contrast values.

    """

    if plot_times_comparison:
        rp = RamseyPulseFinder()
        total_pulse_times = np.zeros(len(dimensions))

    contrasts = np.zeros(len(dimensions))
    contrasts_errors = np.zeros((2, len(dimensions)))
    for idx, dimension in enumerate(dimensions):
        avg_pops, total_num_exps, errors = get_single_contrast(
            dimension=dimension, U1_only_flag=False)
        contrasts[idx] = avg_pops[0][0] - avg_pops[0][1]
        lower_error, upper_error = find_errors(1, contrasts[idx],
                                               total_num_exps)
        contrasts_errors[:, idx] = np.sqrt(2) * np.array(
            [contrasts[idx] - lower_error, upper_error - contrasts[idx]])

        if plot_times_comparison:
            total_pulse_time = rp.find_ordered_pulse_sequence(
                [dimension], verbose=False, save_to_file=False)
            total_pulse_times[idx] = total_pulse_time
        print(f'Dimension = {dimension},' +
              f' Contrast: {np.round(100*contrasts[idx],2)},' +
              f' Errors (init, leakage): {np.round(100*errors,3)}')

    qd = QuditUnitary(topology=topology)
    sim_dims, sim_contrasts, sim_errors = qd.run_qudit_contrast_measurement(
        dimensions=np.arange(3, 16, 1), iterations=1001)

    # Create figure and axes
    if plot_U1_populations:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6))
        # Plot contrasts on the first subplot
        ax1.errorbar(dimensions,
                     contrasts,
                     yerr=contrasts_errors,
                     fmt='o',
                     label='Data')
        ax1.errorbar(sim_dims,
                     sim_contrasts,
                     yerr=sim_errors,
                     fmt='o',
                     label='Simulation')
        ax1.set_title('Ramsey Contrast Scaling with Dimension')
        ax1.set_xlabel('Dimension $(d)$')
        ax1.set_ylabel(r'Contrast of $|0\rangle$ state')
        ax1.grid()
        ax1.legend()

        # Plot U1 populations on the second subplot
        for idx, dimension in enumerate(dimensions):
            ax2.axhline(y=1 / dimension, color='grey', linestyle='--')
            U1_pops = np.zeros(dimension)
            avg_pops, total_num_exps, errors = get_single_contrast(
                dimension=dimension, U1_only_flag=True)
            U1_pops = np.mean(avg_pops, axis=1)
            lower_error, upper_error = find_errors(1, U1_pops, total_num_exps)

            for it, pop in enumerate(U1_pops):
                errors = [[pop - lower_error[it]], [upper_error[it] - pop]]
                ax2.errorbar(dimension + it * 0.03, pop, yerr=errors, fmt='o')

        ax2.set_title('Superposition Populations')
        ax2.set_xlabel('Dimension $(d)$')
        ax2.set_ylabel('Average Populations in Superpositions')
        ax2.grid()

    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(dimensions,
                    contrasts,
                    yerr=contrasts_errors,
                    fmt='o',
                    label='Data')
        ax.errorbar(sim_dims,
                    sim_contrasts,
                    yerr=sim_errors,
                    fmt='o',
                    label='Simulation')
        ax.set_title('Ramsey Contrast Scaling with Dimension')
        ax.set_xlabel('Dimension $(d)$')
        ax.set_ylabel(r'Contrast of $|0\rangle$ state')
        ax.grid()
        ax.legend()

    plt.tight_layout()

    return dimensions, contrasts


if __name__ == '__main__':
    start_time = time.time()

    dimensions = np.arange(3, 14, 1)
    dimensions = [3, 5, 9]
    # dimensions = [9]
    for dim in dimensions:
        get_single_contrast(dimension=dim, U1_only_flag=False)
    # plot_dimensional_contrast(dimensions,
    #                           topology,
    #                           plot_times_comparison=True,
    #                           plot_U1_populations=True)

    plt.show()
    print("--- %s seconds ---" % (time.time() - start_time))
