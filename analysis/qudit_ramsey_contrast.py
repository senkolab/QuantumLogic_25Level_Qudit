import ast
import os
import time
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ramsey_pulse_finder import RamseyPulseFinder

from class_utils import Utils
from plot_utils import nice_fonts, set_size
from qudit_ramsey_unitary import QuditUnitary

# from testing_new_qudit_unitary import QuditUnitary

find_errors = Utils.find_errors

mpl.rcParams.update(nice_fonts)


def get_threshold(file_list, plot_threshold_histogram=False):

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
    # Count the occurrences of each entry
    entry_counts = Counter(pmt_counts)
    # Extract unique entry values and their corresponding counts
    array_counts = np.array(
        [entry_counts.get(key, 0) for key in range(max(entry_counts) + 1)])
    counts_dummy = np.arange(0, len(array_counts), 1)
    mask = counts_dummy < 20
    subset_array_counts = array_counts[mask]

    threshold = np.argmin(subset_array_counts)

    if plot_threshold_histogram:
        print(f'\n Found threshold is: {threshold}.')

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

    return threshold


def get_single_contrast(dimension: int = 3,
                        U1_only_flag: bool = False,
                        verbose: bool = False):
    """Calculate the average population contrast for a specified dimension.

    The function reads data files corresponding to the specified dimension,
    processes the data to compute the average populations, and returns the
    averaged populations along with the total number of processed experiments.

    Parameters:
        dimension (int): The dimension to calculate the contrast for.
        U1_only_flag (bool): A flag indicating whether to filter for U1-only
        experiments (default is False).

    Returns:

        tuple: A tuple containing the averaged populations and the total number
        of processed experiments.
    """

    head_file_path = ('/home/nicholas/Documents/Barium137_Qudit' +
                      '/data/ramseys_contrasts_dimensions_PM_data/')

    # List to store filtered file names
    filtered_file_names = []
    tot_num_processed_exps = 0

    # Iterate through files in the specified directory
    for file_name in os.listdir(head_file_path):
        # Check if both substrings are in the file name
        if (f"d={dimension}_" in file_name) and (f"U1only_{U1_only_flag}"
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
    readout_errors = []
    for file_name_PD in sorted_file_names:

        with open(head_file_path + file_name_PD, 'r') as hf_name:
            # iterate through lines to get the last one
            # raw data filenames associated with the run
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
                         '/data/ramseys_contrasts_dimensions_PM_data' +
                         '/raw_data_fluorescence/')

        first_file_last_part = file_names[0].rsplit('\\', 1)[-1]
        first_file_prefix_segment = file_names[0].rsplit('\\',
                                                         1)[0].rsplit('\\',
                                                                      1)[-1]
        first_file_file_name = (
            f'{first_file_prefix_segment}/{first_file_last_part}')

        # get the number of states by looking at the first data file
        with open(file_path_raw + first_file_file_name, 'r') as filename:
            data = pd.read_csv(
                filename,
                delimiter="\[|\]|,|\"",  # noqa
                engine='python',
                header=None).values
            data = data[:, 4:-19]
            dimension = np.shape(data)[1]

        full_data_array = np.zeros((dimension - 1, len(wait_times)))

        # get individual thresholds from each run
        threshold_file_names = []
        for idx, file_name in enumerate(file_names):

            last_part = file_name.rsplit('\\', 1)[-1]
            prefix_segment = file_name.rsplit('\\', 1)[0].rsplit('\\', 1)[-1]
            file_name = f'{prefix_segment}/{last_part}'

            threshold_file_names.append(file_path_raw + file_name)

        threshold = get_threshold(threshold_file_names)
        if verbose:
            print(f'\n Found threshold is: {threshold}.')

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
            # print(file_name_PD)
            # print(file_name)
            # print(np.shape(data))
            # breakpoint()

            # Initialize a mask to track the first occ of True in each row
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
                    # num_deshelve_error += 1
                    pass
                elif first_true_idx == 0:
                    num_shelve_error += 1

            summed_bin_array = np.sum(first_true_bin_array, axis=0) / (
                first_true_bin_array.shape[0] - num_shelve_error -
                num_deshelve_error)
            summed_bin_array = summed_bin_array[1:]
            full_data_array[:, idx] = summed_bin_array
            tot_num_processed_exps += first_true_bin_array.shape[
                0] - num_shelve_error - num_deshelve_error

            init_errors.append(num_shelve_error /
                               first_true_bin_array.shape[0])
            readout_errors.append(num_deshelve_error /
                                  first_true_bin_array.shape[0])

            # print(file_name_PD)
            # print(num_shelve_error / first_true_bin_array.shape[0],
            #       num_deshelve_error / first_true_bin_array.shape[0])
            # print(summed_bin_array)
            # print(full_data_array)

        repeated_outcomes.append(full_data_array)

    rep_outcomes = np.array(repeated_outcomes)

    if verbose:
        print(init_errors)
        print(readout_errors)
        print(rep_outcomes[:, 0])
    avg_pop = np.mean(rep_outcomes, axis=0)
    avg_init_errors = np.mean(np.array(init_errors))
    avg_readout_errors = np.mean(np.array(readout_errors))
    error_array = np.array([avg_init_errors, avg_readout_errors])

    return rep_outcomes, avg_pop, tot_num_processed_exps, error_array


def plot_dimensional_contrast(dimensions: np.ndarray = np.arange(3, 8, 1),
                              topology: str = 'All',
                              plot_simulation: bool = False,
                              plot_times_comparison: bool = False,
                              plot_U1_populations: bool = False,
                              iterations: int = 128,
                              verbose: bool = False):
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

    qd = QuditUnitary(topology=topology)
    # Get pulse times for plotting
    pulse_times_plotting = []
    for dim in dimensions:
        _, pulse_time = qd.get_full_experiment_runtime(dim)
        pulse_times_plotting.append(pulse_time)
    if plot_simulation:
        # sim_dims, sim_contrasts, sim_err = qd.run_qudit_contrast_measurement(
        #     dimensions=dimensions, iterations=iterations)
        # breakpoint()
        sim_contrasts = np.array([
            0.98214401, 0.98517927, 0.98434187, 0.98381514, 0.97888958,
            0.97532986, 0.97079359, 0.96376811, 0.95079959, 0.95225023,
            0.91632188, 0.9248739, 0.88940377, 0.87389729, 0.89129453,
            0.86445701, 0.72614223, 0.69453646, 0.66893262, 0.40685436,
            0.40126329, 0.31950439, 0.19064551
        ])

        sim_err = np.array([
            0.00055636, 0.00041762, 0.00044039, 0.00055806, 0.00069374,
            0.0009134, 0.00102073, 0.00144686, 0.00187615, 0.00209438,
            0.0030128, 0.00339336, 0.00370522, 0.00468382, 0.00485918,
            0.0053401, 0.00639005, 0.00735407, 0.00624621, 0.00295597,
            0.0033015, 0.00267982, 0.00226656
        ])
        sim_dims = np.array([
            2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24
        ])

        print('Simulated contrasts', sim_contrasts)
        print('Simulated contrast errors', sim_err)

    contrasts = np.zeros(len(dimensions))
    contrasts_errors = np.zeros((2, len(dimensions)))
    indiv_outcomes = []  # a list because we do know each length a priori

    contrasts = np.array([
        0.9787623, 0.97695082, 0.9814498, 0.98492677, 0.97171693, 0.9679671,
        0.9694631, 0.96052485, 0.95072729, 0.95481656, 0.93498656, 0.94033928,
        0.91946299, 0.91002923, 0.90137823, 0.86119699, 0.69789453, 0.6874485,
        0.61253402, 0.47782983, 0.42387411, 0.30319517, 0.14793561
    ])
    contrasts_errors = np.array([
        0.00984418, 0.01447916, 0.0098507, 0.00788959, 0.01064166, 0.01050164,
        0.01246716, 0.01408427, 0.02141266, 0.02127871, 0.02380241, 0.01981266,
        0.02404444, 0.0222354, 0.03704389, 0.05682174, 0.12299253, 0.0976164,
        0.06139097, 0.05264131, 0.04047939, 0.04024148, 0.05261314
    ])

    # for idx, dimension in enumerate(dimensions):
    #     rep_pops, avg_pops, total_num_exps, errors = get_single_contrast(
    #         dimension=dimension, U1_only_flag=False, verbose=verbose)
    #     contrasts[idx] = avg_pops[0][0] - avg_pops[0][1]
    #     individual_runs = rep_pops[:, 0][:, 0] - rep_pops[:, 0][:, 1]

    #     if verbose:
    #         print(individual_runs)
    #         # print(rep_pops)

    #     indiv_outcomes.append(individual_runs)

    #     # Wilson interval errors
    #     # lower_error, upper_error = find_errors(1, contrasts[idx],
    #     #                                        total_num_exps)
    #     # contrasts_errors[:, idx] = np.sqrt(2) * np.array(
    #     #     [contrasts[idx] - lower_error, upper_error - contrasts[idx]])

    #     # standard deviation errors
    #     contrasts_errors[:, idx] = np.sqrt(2) * np.std(individual_runs)
    #     # breakpoint()

    #     if plot_times_comparison:
    #         total_pulse_time = rp.find_ordered_pulse_sequence(
    #             [dimension], verbose=False, save_to_file=False)
    #         total_pulse_times[idx] = total_pulse_time
    #     print(f'Dimension = {dimension},' +
    #           f' Contrast: {np.round(100*contrasts[idx],2)},' +
    #           f' Errors (init, leakage): {np.round(100*errors,3)}' +
    #           f' State 0: {avg_pops[0]}, ' +
    #           f'Runs: {len(individual_runs)}, ' +
    #           f'Pulse Time: {np.round(pulse_times_plotting[idx])} us.')

    # Create figure and axes
    if plot_U1_populations:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6))
        # Plot contrasts on the first subplot
        ax1.errorbar(dimensions,
                     contrasts,
                     yerr=contrasts_errors,
                     fmt='o-',
                     label='Data')
        ax1.errorbar(sim_dims,
                     sim_contrasts,
                     yerr=sim_err,
                     fmt='o--',
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
            rep_pops, avg_pops, total_num_exps, errors = get_single_contrast(
                dimension=dimension, U1_only_flag=True)
            U1_pops = np.mean(avg_pops, axis=1)
            lower_error, upper_error = find_errors(1, U1_pops, total_num_exps)

            for it, pop in enumerate(U1_pops):
                errors = [[pop - lower_error[it]], [upper_error[it] - pop]]
                ax2.errorbar(dimension + it * 0.03, pop, yerr=errors, fmt='o-')

        ax2.set_title('Superposition Populations')
        ax2.set_xlabel('Dimension $(d)$')
        ax2.set_ylabel('Average Populations in Superpositions')
        ax2.grid()

    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.errorbar(dimensions,
                    1 - contrasts,
                    yerr=contrasts_errors,
                    fmt='o-',
                    label='Data')
        # ax.scatter(len(individual_runs) * [dimensions],
        #             individual_runs,
        #             color='black',
        #             alpha=0.1)

        # plot individual measurements
        for i, y_list in enumerate(indiv_outcomes):
            plt.scatter([dimensions[i]] * len(y_list),
                        1 - y_list,
                        color='black',
                        alpha=0.1)

        if plot_simulation:
            # simulated contrasts here accounts for SPAM infidelity
            ax.errorbar(
                sim_dims,
                1 - sim_contrasts,
                # 1 - sim_contrasts,
                yerr=sim_err,
                fmt='o--',
                label='Simulation')
        # ax.set_ylim(0,1)
        ax.set_title('Ramsey Contrast Scaling with Dimension')
        ax.set_xlabel('Dimension $(d)$')
        ax.set_ylabel(r'Contrast loss of $|0\rangle$ state')
        ax.grid()
        ax.legend(fontsize=14)
        plt.tight_layout()

        # Now plotting vs pulse times
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(pulse_times_plotting,
                    1 - contrasts,
                    yerr=contrasts_errors,
                    fmt='o-',
                    label='Data')
        # ax.scatter(len(individual_runs) * [dimensions],
        #             individual_runs,
        #             color='black',
        #             alpha=0.1)

        # plot individual measurements
        for i, y_list in enumerate(indiv_outcomes):
            plt.scatter([pulse_times_plotting[i]] * len(y_list),
                        1 - y_list,
                        color='black',
                        alpha=0.1)

        if plot_simulation:
            # simulated contrasts here accounts for SPAM infidelity
            # breakpoint()
            ax.errorbar(
                pulse_times_plotting,
                1 - sim_contrasts,
                # 1 - sim_contrasts,
                yerr=sim_err,
                fmt='o--',
                label='Simulation')
        # ax.set_ylim(0,1)
        ax.set_title('Ramsey Contrast Scaling with Dimension')
        ax.set_xlabel(r'Total Ramsey Pulse Time / $\mu s$')
        ax.set_ylabel(r'Contrast loss of $|0\rangle$ state')
        ax.grid()
        ax.legend()
        plt.tight_layout()

    return dimensions, contrasts


if __name__ == '__main__':
    start_time = time.time()

    # threshold = get_threshold(plot_threshold_histogram=True)

    dimensions = np.arange(2, 25)

    # get_single_contrast(dimension=6, U1_only_flag=False,verbose=True)

    plot_dimensional_contrast(dimensions=dimensions,
                              topology='Star',
                              plot_simulation=True,
                              plot_times_comparison=True,
                              plot_U1_populations=False,
                              iterations=1024,
                              verbose=False)

    plt.show()

    breakpoint()

    print("--- %s seconds ---" % (time.time() - start_time))
