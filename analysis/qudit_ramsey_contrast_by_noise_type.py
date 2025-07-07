import ast
import os
import time
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from class_utils import Utils
from plot_utils import nice_fonts, set_size
from qudit_ramsey_unitary import QuditUnitary
from ramsey_pulse_finder import RamseyPulseFinder

# from testing_new_qudit_unitary import QuditUnitary

find_errors = Utils.find_errors

mpl.rcParams.update(nice_fonts)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


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

        # print(file_name_PD)
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
    noise_types = [
        'A/C Line Signal', 'B-Field Noise', 'Laser Freq. Noise',
        'Freq. Cal. Error', 'Pulse Angle Error', 'Total Error',
        'Low Reported Noise Levels'
    ]
    # noise_types = ['A/C Line Signal']
    # noise_types = ['Total Error']

    # Get pulse times for plotting
    pulse_times_plotting = []
    for dim in dimensions:
        qd = QuditUnitary(topology=topology)
        _, pulse_time = qd.get_full_experiment_runtime(dim)
        pulse_times_plotting.append(pulse_time)

    # # uncomment the block below to re-run full simulation
    # sim_contrasts = np.zeros((len(noise_types), len(dimensions)))
    # sim_contrasts_errors = np.zeros((len(noise_types), len(dimensions)))
    # for idx, noise_type in enumerate(noise_types):

    #     if noise_type == 'A/C Line Signal':
    #         qd = QuditUnitary(
    #             topology=topology,
    #             line_amplitude_60=0.00015,  # Gauss
    #             line_amplitude_180=0.00005,  # Gauss
    #             line_phase_60=-0.636,  # radians
    #             line_phase_180=-1.551,  # radians
    #             line_offset=0.000247,  # Gauss
    #             mag_field_noise=0,
    #             laser_noise_gaussian=0,
    #             laser_noise_lorentzian=0,
    #             pulse_time_miscalibration=0,
    #             pulse_time_variation=0,
    #             freq_miscalibration=0)
    #     elif noise_type == 'B-Field Noise':
    #         qd = QuditUnitary(
    #             topology=topology,
    #             line_amplitude_60=0,
    #             line_amplitude_180=0,
    #             line_phase_60=0,
    #             line_phase_180=0,
    #             line_offset=0,
    #             mag_field_noise=0.000049 / (np.sqrt(2) * np.pi),  # Gauss
    #             laser_noise_gaussian=0,
    #             laser_noise_lorentzian=0,
    #             pulse_time_miscalibration=0,
    #             pulse_time_variation=0,
    #             freq_miscalibration=0)
    #     elif noise_type == 'Laser Noise':
    #         qd = QuditUnitary(
    #             topology=topology,
    #             line_amplitude_60=0,
    #             line_amplitude_180=0,
    #             line_phase_60=0,
    #             line_phase_180=0,
    #             line_offset=0,
    #             mag_field_noise=0,
    #             laser_noise_gaussian=1e6 / (2 * np.pi * 1950),  # Hz
    #             laser_noise_lorentzian=1e6 / (2 * np.pi * 2065),  # Hz
    #             pulse_time_miscalibration=0,
    #             pulse_time_variation=0,
    #             freq_miscalibration=0)
    #     elif noise_type == 'Frequency Error':
    #         qd = QuditUnitary(
    #             topology=topology,
    #             line_amplitude_60=0,
    #             line_amplitude_180=0,
    #             line_phase_60=0,
    #             line_phase_180=0,
    #             line_offset=0,
    #             mag_field_noise=0,
    #             laser_noise_gaussian=0,
    #             laser_noise_lorentzian=0,
    #             pulse_time_miscalibration=0,
    #             pulse_time_variation=0,
    #             freq_miscalibration=72,  # Hz
    #         )
    #     elif noise_type == 'Pulse Error':
    #         qd = QuditUnitary(
    #             topology=topology,
    #             line_amplitude_60=0,
    #             line_amplitude_180=0,
    #             line_phase_60=0,
    #             line_phase_180=0,
    #             line_offset=0,
    #             mag_field_noise=0,
    #             laser_noise_gaussian=0,
    #             laser_noise_lorentzian=0,
    #             pulse_time_miscalibration=0.0177,  # %
    #             pulse_time_variation=0.0262,
    #             freq_miscalibration=0,
    #         )
    #     elif noise_type == 'Total Error':
    #         qd = QuditUnitary(
    #             topology=topology,
    #             line_amplitude_60=0.00015,  # Gauss
    #             line_amplitude_180=0.00005,  # Gauss
    #             line_phase_60=-0.636,  # radians
    #             line_phase_180=-1.551,  # radians
    #             line_offset=0.0002476,  # Gauss
    #             mag_field_noise=0.000049 / (np.sqrt(2) * np.pi),  # Gauss
    #             laser_noise_gaussian=1e6 / (2 * np.pi * 1950),  # Hz
    #             laser_noise_lorentzian=1e6 / (2 * np.pi * 2065),  # Hz
    #             pulse_time_miscalibration=0.0177,  # %
    #             pulse_time_variation=0.0262,
    #             freq_miscalibration=72,  # Hz
    #         )

    #     if plot_simulation:
    #         if noise_type == 'A/C Line Signal':
    #             iterats = 1
    #         else:
    #             iterats = 512
    #         print('Now running...', noise_type)
    #         sim_dims, sim_res, sim_err = qd.run_qudit_contrast_measurement(
    #             dimensions=dimensions, iterations=iterats)

    #         sim_contrasts[idx, :] = sim_res
    #         sim_contrasts_errors[idx, :] = sim_err
    # breakpoint()

    # saved simulation results with 512 simulation results put together
    sim_dims = np.arange(2, 25)
    sim_contrasts = np.array(
        [[
            0.9999999966943462, 0.99999997, 0.99999993, 0.99999914, 0.9999832,
            0.9999678, 0.9999422, 0.99987083, 0.99982248, 0.99966068,
            0.99953811, 0.99884166, 0.99822605, 0.99672465, 0.99484235,
            0.98944225, 0.92843508, 0.93436523, 0.89307503, 0.60628414,
            0.59515845, 0.41696453, 0.29004663
        ],
         [
             0.99998865, 0.99999248, 0.99999146, 0.99996938, 0.99971748,
             0.99963468, 0.99949096, 0.99916683, 0.99926947, 0.99912897,
             0.99892814, 0.99840182, 0.99812166, 0.99718138, 0.99697344,
             0.99686466, 0.99147621, 0.99180008, 0.98905635, 0.97335341,
             0.97226234, 0.95308721, 0.92758108
         ],
         [
             0.99907712, 0.99852732, 0.99686075, 0.99400932, 0.99054936,
             0.98833045, 0.98520019, 0.98176386, 0.97241967, 0.96615603,
             0.95983844, 0.94359142, 0.94098456, 0.92344542, 0.9271539,
             0.90992453, 0.86736119, 0.81597208, 0.85354392, 0.83468735,
             0.82082309, 0.79055924, 0.79024709
         ],
         [
             0.9999127, 0.99978206, 0.99966768, 0.99941988, 0.9990807,
             0.99887795, 0.99864521, 0.99787251, 0.99704325, 0.99643073,
             0.99534831, 0.9941929, 0.99285745, 0.99033531, 0.98939414,
             0.98577395, 0.97763924, 0.96772758, 0.97421336, 0.97381118,
             0.97051819, 0.96587121, 0.95713257
         ],
         [
             0.98496145, 0.98874392, 0.98964213, 0.99022346, 0.99040922,
             0.99050459, 0.99128198, 0.9910123, 0.99121567, 0.99119412,
             0.99121464, 0.99237332, 0.99311577, 0.99280263, 0.99236535,
             0.9923255, 0.99166379, 0.9909527, 0.99077166, 0.99161794,
             0.99056321, 0.98990814, 0.99131455
         ],
         [
             0.98416385, 0.98476236, 0.98507688, 0.98446661, 0.979729,
             0.97539862, 0.97120493, 0.9632236, 0.94931654, 0.94567368,
             0.9158669, 0.92311135, 0.88022455, 0.880441, 0.88966436,
             0.85148455, 0.73451677, 0.6992517, 0.6590196, 0.40204474,
             0.41075282, 0.32919891, 0.19103534
         ],
         [
             0.9999891931226965, 0.9999900408840353, 0.9999895951664372,
             0.9999737184510252, 0.9999797836153058, 0.9999616378767504,
             0.9999712617420474, 0.999953663511086, 0.9999224504923763,
             0.9998724987994412, 0.9998518543311223, 0.9996897058887365,
             0.9995150205347356, 0.9993647597937743, 0.9990857647440551,
             0.9980406222565208, 0.9892850830107967, 0.9899885970598769,
             0.9846660982425335, 0.9312303894166, 0.9354925493993046,
             0.8331200462522508, 0.5874045980899223
         ]])

    sim_contrasts_errors = np.array([
        [
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0.
        ],
        [
            0.00000048, 0.00000033, 0.00000056, 0.00000179, 0.00001439,
            0.00001984, 0.00003031, 0.00004366, 0.00004365, 0.00004606,
            0.00006311, 0.00009945, 0.00010818, 0.00017994, 0.00018368,
            0.00017588, 0.00052963, 0.00047152, 0.00063558, 0.00151237,
            0.00166436, 0.00264978, 0.00387247
        ],
        [
            0.00007221, 0.00011372, 0.00032434, 0.00056345, 0.0009227,
            0.00113317, 0.00150667, 0.00188383, 0.0027657, 0.00319621,
            0.00372292, 0.00490023, 0.00524619, 0.00666562, 0.00709694,
            0.006871, 0.00877021, 0.0112719, 0.00954314, 0.01039297,
            0.01161029, 0.01183117, 0.01196614
        ],
        [
            0.00000412, 0.00000983, 0.00001717, 0.00003254, 0.00005707,
            0.00006814, 0.00008752, 0.00013424, 0.00017541, 0.00020485,
            0.0002753, 0.00038711, 0.00042491, 0.00053323, 0.00061743,
            0.00081156, 0.00132861, 0.00191669, 0.00148843, 0.00162185,
            0.00184705, 0.00204673, 0.00218914
        ],
        [
            0.00070735, 0.00054858, 0.00053621, 0.00047675, 0.0005999,
            0.00056755, 0.00052966, 0.00056141, 0.00050034, 0.0005428,
            0.00063654, 0.00045293, 0.00038706, 0.00045242, 0.00046727,
            0.00048118, 0.00046099, 0.00057478, 0.00054175, 0.00050369,
            0.00060723, 0.00053096, 0.00056365
        ],
        [
            0.00065866, 0.00063933, 0.00058339, 0.00075629, 0.00087275,
            0.00118049, 0.00144793, 0.00208, 0.00302961, 0.00374235,
            0.00400353, 0.00484177, 0.00593413, 0.0061017, 0.00649748,
            0.00758394, 0.00858504, 0.01013034, 0.00891379, 0.00403552,
            0.00418791, 0.00359111, 0.0031148
        ],
        [
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0.
        ],
        # [
        #     1.1561147122995005e-06, 1.220901394147507e-05,
        #     5.3613437249165104e-06, 9.883944376980064e-07,
        #     2.82155484727013e-05, 5.779346119927831e-06,
        #     0.00012568951148529174, 2.5587639507339493e-05,
        #     6.484455101848081e-06, 0.00016285828752095303,
        #     9.128216850686984e-06, 0.0007866768401186144,
        #     4.506290177146157e-06, 0.0003278996859272142,
        #     0.0003967744667005362, 9.818904131651888e-05,
        #     0.00032121060899725366, 0.000325842656008025,
        #     0.0009812880399360415, 5.932499255753552e-05,
        #     0.0003256111498961417, 0.00021264980239775659, 8.74609474651487e-05
        # ]
    ])

    contrasts = np.zeros(len(dimensions))
    contrasts_errors = np.zeros((2, len(dimensions)))
    indiv_outcomes = []  # a list because we do know each length a priori

    for idx, dimension in enumerate(dimensions):
        rep_pops, avg_pops, total_num_exps, errors = get_single_contrast(
            dimension=dimension, U1_only_flag=False, verbose=verbose)
        contrasts[idx] = avg_pops[0][0] - avg_pops[0][1]
        individual_runs = rep_pops[:, 0][:, 0] - rep_pops[:, 0][:, 1]

        if verbose:
            print(individual_runs)
            # print(rep_pops)

        indiv_outcomes.append(individual_runs)

        # Wilson interval errors
        # lower_error, upper_error = find_errors(1, contrasts[idx],
        #                                        total_num_exps)
        # contrasts_errors[:, idx] = np.sqrt(2) * np.array(
        #     [contrasts[idx] - lower_error, upper_error - contrasts[idx]])

        # standard deviation errors
        contrasts_errors[:, idx] = np.sqrt(2) * np.std(individual_runs)
        # breakpoint()

        if plot_times_comparison:
            total_pulse_time = rp.find_ordered_pulse_sequence(
                [dimension], verbose=False, save_to_file=False)
            total_pulse_times[idx] = total_pulse_time
        print(f'Dimension = {dimension},' +
              f' Contrast: {np.round(100*contrasts[idx],2)},' +
              f' Errors (init, leakage): {np.round(100*errors,3)}' +
              f' State 0: {avg_pops[0]}, ' +
              f'Runs: {len(individual_runs)}, ' +
              f'Pulse Time: {np.round(pulse_times_plotting[idx])} us.')

    fig, ax = plt.subplots(figsize=(8, 4))
    # ax.scatter(len(individual_runs) * [dimensions],
    #             individual_runs,
    #             color='black',
    #             alpha=0.1)

    # plot individual measurements
    # for i, y_list in enumerate(indiv_outcomes):
    #     plt.scatter([dimensions[i]] * len(y_list),
    #                 1 - y_list,
    #                 color='black',
    #                 alpha=0.1)

    alphas = [0.5, 0.5, 0.5, 0.5, 0.5, 1, 0.5]
    fmts = ['o--', 'o--', 'o--', 'o--', 'o--', 'o-', '*-']
    markers = [6,6,6,6,6,6,12]
    color_plots = [
        colors[1], colors[2], colors[3], colors[4], colors[5], 'black', 'grey'
    ]

    if plot_simulation:
        # simulated contrasts here accounts for SPAM infidelity
        for idx, noise_type in enumerate(noise_types):
            ax.errorbar(sim_dims,
                        1 - sim_contrasts[idx],
                        yerr=sim_contrasts_errors[idx],
                        fmt=fmts[idx],
                        markersize=markers[idx],
                        color=color_plots[idx],
                        alpha=alphas[idx],
                        label=noise_type)

    # finally the data on top
    ax.errorbar(dimensions,
                1 - contrasts,
                yerr=contrasts_errors,
                fmt='o-',
                label='Data')
    breakpoint()

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
    # for i, y_list in enumerate(indiv_outcomes):
    #     plt.scatter([pulse_times_plotting[i]] * len(y_list),
    #                 1 - y_list,
    #                 color='black',
    #                 alpha=0.1)

    if plot_simulation:
        # simulated contrasts here accounts for SPAM infidelity
        for idx, noise_type in enumerate(noise_types):
            ax.errorbar(pulse_times_plotting,
                        1 - sim_contrasts[idx],
                        yerr=sim_contrasts_errors[idx],
                        markersize=markers[idx],
                        color=color_plots[idx],
                        fmt=fmts[idx],
                        alpha=alphas[idx],
                        label=noise_type)

    ax.set_title('Ramsey Contrast Scaling with Dimension')
    ax.set_xlabel(r'Total Ramsey Pulse Time / $\mu s$')
    ax.set_ylabel(r'Contrast loss of $|0\rangle$ state')
    ax.grid()
    ax.legend(fontsize=14)
    plt.tight_layout()

    # how well does our simulation perform?
    print('The mean absolute deviation between simulation and data',
          np.mean(np.abs(contrasts - sim_contrasts[-1])))
    print('The maximum difference between simulation and data',
          np.max(np.abs(contrasts - sim_contrasts[-1])))

    return dimensions, contrasts


if __name__ == '__main__':
    start_time = time.time()

    dimensions = np.arange(2, 25)

    # threshold = get_threshold(plot_threshold_histogram=True)
    # get_single_contrast(dimension=6, U1_only_flag=False,verbose=True)

    plot_dimensional_contrast(dimensions=dimensions,
                              topology='Star',
                              plot_simulation=True,
                              plot_times_comparison=False,
                              plot_U1_populations=False,
                              verbose=False)

    plt.show()
    print("--- %s seconds ---" % (time.time() - start_time))
