import ast
import itertools
import json
import os
# import sys
import time
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc
from scipy.special import wofz

from class_barium import Barium
from class_utils import Utils
from plot_utils import colors, markers, nice_fonts, set_size

find_errors = Utils.find_errors

mpl.rcParams.update(nice_fonts)
np.set_printoptions(suppress=True)


def is_file_empty(file_path):
    """
    Check whether a file is completely empty.

    Parameters:
    - file_path (str): Path to the file.

    Returns:
    - bool: True if the file is empty, False otherwise.
    """
    return os.path.exists(file_path) and os.stat(file_path).st_size == 0


def get_threshold(transition_string, plot_threshold_histogram=True):

    head_file_path = ('../data/ramseys_segmented_PM_data/')

    all_files = os.listdir(head_file_path)
    matching_files = [
        # file for file in all_files if file.endswith(files_datetime + '.txt')
        file for file in all_files if transition_string in file
    ]
    matching_files.sort(
        key=lambda x: os.path.getmtime(os.path.join(head_file_path, x)))

    all_filenames = []
    for idx, filename in enumerate(matching_files):
        with open(head_file_path + filename, 'r') as hf_name:
            # iterate through lines to get the last one, which gives
            # a list of all raw data filenames associated with the run
            for line in hf_name:
                pass
            all_filenames.append(ast.literal_eval(line.strip()))

    all_filenames = list(itertools.chain(*all_filenames))

    file_path_raw = (
        '../data/ramseys_segmented_PM_data/raw_data_fluorescence/')

    # get pmt counts for thresholding
    pmt_counts = []
    for idx, file_name in enumerate(all_filenames):
        if len(file_name) > 40:
            # fix filenames for new naming convention that uses full paths
            last_part = file_name.rsplit('\\', 1)[-1]
            prefix_segment = file_name.rsplit('\\', 1)[0].rsplit('\\', 1)[-1]
            file_name = f'{prefix_segment}/{last_part}'

        if not is_file_empty(file_path_raw + file_name):
            with open(file_path_raw + file_name, 'r') as filename:
                data = pd.read_csv(
                    filename,
                    delimiter='\[|\]|,|\"',  # noqa: W605
                    engine='python',
                    header=None).values
                data = data[:, 4:-19]
                pmt_counts += list(np.ravel(data))
        else:
            pass
    # do some processing to find threshold dynamically
    # Count the occurrences of each entry
    entry_counts = Counter(pmt_counts)
    # Extract unique entry values and their corresponding counts
    array_counts = [
        entry_counts.get(key, 0) for key in range(max(entry_counts) + 1)
    ]
    counts_dummy = np.arange(0, len(array_counts), 1)

    # consider only array_counts for counts under 30 for finding min
    mask = counts_dummy < 25
    array_counts_subset = np.array(array_counts)[mask]

    threshold = np.argmin(array_counts_subset[:-10])
    print(f'Found threshold is: {threshold}.')
    if plot_threshold_histogram:
        fig, ax = plt.subplots(figsize=set_size(width='half'))

        # Plot the histogram
        ax.bar(counts_dummy, array_counts, label='Original Histogram')
        # ax.plot(counts_dummy,
        #         smoothed_values,
        #         label='Smoothed Curve',
        #         color='red')
        ax.vlines(threshold,
                  min(array_counts),
                  max(array_counts),
                  colors='k',
                  linestyles='--',
                  label='Threshold')
        # ax.set_yscale('log')
        ax.set_title('Fluorescence Readout Histogram')
        ax.set_ylabel('Occurences')
        ax.set_xlabel('PMT Counts')
        ax.grid()
        ax.legend()

    return threshold


def plot_segmented_ramsey(transition_string,
                          fit_type='sine',
                          exp_num=101,
                          bussed=False,
                          plotting=True):

    threshold = get_threshold(transition_string,
                              plot_threshold_histogram=False)

    head_file_path = ('../data/ramseys_segmented_PM_data/')

    all_files = os.listdir(head_file_path)
    matching_files = [
        # file for file in all_files if file.endswith(files_datetime + '.txt')
        file for file in all_files if transition_string in file
    ]
    matching_files.sort(
        key=lambda x: os.path.getmtime(os.path.join(head_file_path, x)))

    # plot the results all together
    if plotting:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel(r'Wait Time ($\mu s$)')
        ax.set_ylabel('Probabilities')
        ax.set_ylim(-0.05, 1.05)
        ax.grid()

    # define the lists of amplitudes and times used to fit the exp decay
    amplitudes = []
    amplitudes_errors = []
    phases = []
    phases_errors = []
    real_wait_times = []
    for idx, filename in enumerate(matching_files):

        with open(head_file_path + filename, 'r') as hf_name:
            # iterate through lines to get the last one, which gives
            # a list of all raw data filenames associated with the run
            # print(filename)
            split_name = filename.split('_')

            if bussed:
                real_wait_time = split_name[3]
            else:
                real_wait_time = split_name[2][8:-2]
            # print(real_wait_time)
            # s12_state = split_name[-7]
            # d52_state = [split_name[-6], split_name[-5]]
            # print('Wait time for this head file:', split_name[-4], 'us.')
            for line_num, line in enumerate(hf_name):
                if line_num == 17:
                    state_triplet = ast.literal_eval(line.strip())[0]
                else:
                    pass
            raw_file_names = line.strip()

            # need to convert raw_file_names to be a true list of strings
            raw_file_names = ast.literal_eval(raw_file_names)
            # print(raw_file_names)

            # print(filename)
            # get pulse times from the save txt data in head_file_name
            # datetime_format = '%Y%m%d_%H%M'
            # cutoff_datetime = '20241002_0000'
            # dt_fname = datetime.strptime(files_datetime, datetime_format)
            # dt_check = datetime.strptime(cutoff_datetime, datetime_format)
            # if dt_fname > dt_check:
            #     data = np.genfromtxt(hf_name, delimiter=',', skip_footer=5)
            # else:
            #     data = np.genfromtxt(hf_name, delimiter=',', skip_footer=4)
            # wait_times = float(real_wait_time) + data[:, 0]
            wait_times = float(real_wait_time) + np.arange(
                0, len(raw_file_names), 1)
            # print(wait_times)
            # print(float(real_wait_time) + np.arange(0,len(raw_file_names),1))
            # data_acq_times = np.arange(0, len(raw_file_names), 1)

        file_path_raw = (
            '../data/ramseys_segmented_PM_data/raw_data_fluorescence/')

        # Define the oscillations fit function
        def oscfunc(t, A, p, c, f):
            w = 2 * np.pi / 10
            return c - (A * np.cos(w * t + p)) / 2
            # return A * np.sin(w * t + f * t**2 + p) / 2 + c

        for non_empty_filename in raw_file_names:
            if len(non_empty_filename) > 40:
                # fix filenames for new naming convention that uses full paths
                last_part = non_empty_filename.rsplit('\\', 1)[-1]
                prefix_segment = non_empty_filename.rsplit('\\', 1)[0].rsplit(
                    '\\', 1)[-1]
                non_empty_filename = f'{prefix_segment}/{last_part}'
            # print(non_empty_filename)

            if not is_file_empty(file_path_raw + non_empty_filename):
                with open(file_path_raw + non_empty_filename, 'r') as filename:
                    data = pd.read_csv(
                        filename,
                        delimiter='\[|\]|,|\"',  # noqa: W605
                        engine='python',
                        header=None).values
                    data = data[:, 4:-19]
                # define the dummy data array to store info
                full_data_array = np.full((np.shape(data)[1], len(wait_times)),
                                          np.nan)
                break
            else:
                pass

        if np.shape(data)[1] > 3:
            bussed = True
        else:
            bussed = False

        if bussed:
            states_for_legend = split_name[2][1:-1]
            states_for_legend = states_for_legend.replace('[', '(').replace(
                ']', ')')
        else:
            states_for_legend = str('[' + str(split_name[2]) + ',' +
                                    split_name[3] + ',' + split_name[4] + ']')
            states_for_legend = np.array([
                int(state_triplet[0]),
                int(state_triplet[1]),
                int(state_triplet[2])
            ])

            states_for_legend = '(' + ', '.join(map(str,
                                                    states_for_legend)) + ')'

        if plotting:
            ax.set_title(r'Bussed Ramsey Plot : ($m_S,F_D,m_D$)=' +
                         states_for_legend)
        else:
            pass
        # now move on to data processing
        for idx, file_name in enumerate(raw_file_names):
            if len(file_name) > 40:
                # fix filenames for new naming convention that uses full paths
                last_part = file_name.rsplit('\\', 1)[-1]
                prefix_segment = file_name.rsplit('\\', 1)[0].rsplit('\\',
                                                                     1)[-1]
                file_name = f'{prefix_segment}/{last_part}'

            # print(non_empty_filename)
            if not is_file_empty(file_path_raw + file_name):
                with open(file_path_raw + file_name, 'r') as filename:
                    data = pd.read_csv(
                        filename,
                        delimiter='\[|\]|,|\"',  # noqa: W605
                        engine='python',
                        header=None).values
                    data = data[:, 4:-19]
                binary_data = data > threshold

                first_true_bin_array = np.zeros_like(binary_data, dtype=bool)

                # Iterate through each row
                num_shelve_error = 0
                num_deshelve_error = 0
                for row_idx in range(binary_data.shape[0]):
                    row = binary_data[row_idx]
                    first_true_idx = np.argmax(
                        row)  # Find the index of the first True
                    # print(row)
                    # print(first_true_idx)
                    # break
                    if sum(row) > 0 and first_true_idx:
                        first_true_bin_array[row_idx, first_true_idx] = True
                    elif sum(row) == 0:
                        num_deshelve_error += 1
                    elif first_true_idx == 0:
                        num_shelve_error += 1

                upload_error_check = np.shape(first_true_bin_array)[
                    0] - num_shelve_error - num_deshelve_error

                if upload_error_check == 0:
                    full_data_array[:,
                                    idx] = np.full(np.shape(data)[1], np.nan)
                else:
                    full_data_array[:, idx] = np.sum(
                        first_true_bin_array,
                        axis=0) / (np.shape(first_true_bin_array)[0] -
                                   num_shelve_error - num_deshelve_error)
            else:
                full_data_array[:, idx] = np.full(np.shape(data)[1], np.nan)

        PD = full_data_array[1, :]
        # Identify nan values and get their indices to pop them
        # This is needed for heralded runs where all
        # heralds fail due to AWG upload
        nan_mask = np.isnan(PD)
        nan_indices = np.argwhere(nan_mask)

        if nan_indices.size != 0:
            for index in reversed(nan_indices):
                PD = (np.delete(PD, index[0]))
                wait_times = (np.delete(wait_times, index[0]))

        # do a fit to get the decay time
        lower_error, upper_error = find_errors(1, PD, exp_num)
        asym_error_bar = np.abs([PD - lower_error, upper_error - PD])

        A_guess = 0.5
        # ff = np.fft.fftfreq(len(wait_times), (wait_times[1] - wait_times[0]))
        # F_PD = abs(np.fft.fft(PD))
        # guess_freq = abs(ff[np.argmax(F_PD[1:]) + 1])
        # w_guess = 2.0 * np.pi * guess_freq
        p_guess = 0
        c_guess = 0.5
        f_guess = 0
        p0 = np.array([A_guess, p_guess, c_guess, f_guess])
        x_new = np.linspace(wait_times.min(), wait_times.max(), 1000)

        bounds = (
            [0, -np.pi, 0, -np.inf],  # Lower bounds
            [1, np.pi, 1, np.inf])  # Upper bounds

        # print(file_name)
        # ax.plot(x_new, oscfunc(x_new, *p0), color=colors[1], alpha=0.5)
        # ax.errorbar(wait_times,
        #             PD,
        #             color='b',
        #             fmt='o',
        #             yerr=asym_error_bar)
        # plt.show()
        # print(wait_times)
        # print(PD)

        # print(wait_times-float(real_wait_time))
        # print(np.round(PD,2))
        try:
            popt, pcov = sc.optimize.curve_fit(oscfunc,
                                               wait_times -
                                               float(real_wait_time),
                                               PD,
                                               p0=p0,
                                               bounds=bounds,
                                               maxfev=10000)

            if plotting:
                ax.plot(x_new,
                        oscfunc(x_new - float(real_wait_time), *popt),
                        color=colors[1],
                        alpha=0.5)

                ax.errorbar(wait_times,
                            PD,
                            color=colors[0],
                            fmt='o',
                            yerr=asym_error_bar)
            else:
                pass

            amplitudes.append(popt[0])
            amplitudes_errors.append(np.sqrt(pcov[0][0]))
            phases.append(popt[1])
            phases_errors.append(np.sqrt(pcov[1][1]))

        except RuntimeError as e:
            amplitudes.append(max(PD) - min(PD))
            amplitudes_errors.append(0.5 / exp_num)
            phases.append(0)
            phases_errors.append(0)
        real_wait_times.append(float(real_wait_time))

    # handle cases where the data was taken out of order temporally
    sorter = np.argsort(real_wait_times)

    real_wait_times = np.array(real_wait_times)
    amplitudes = np.array(amplitudes)
    amplitudes_errors = np.array(amplitudes_errors)
    phases = np.array(phases)
    phases_errors = np.array(phases_errors)

    real_wait_times = real_wait_times[sorter]
    amplitudes = amplitudes[sorter]
    amplitudes_errors = amplitudes_errors[sorter]
    phases = phases[sorter]
    phases_errors = phases_errors[sorter]

    return (np.abs(amplitudes), amplitudes_errors, real_wait_times,
            states_for_legend, bussed, phases, phases_errors)


def fit_segmented_decay(transition_string, decay_type='exp', bussed=True):

    fig, ax = plt.subplots(figsize=set_size('half'))

    # Create an empty DataFrame with the desired columns
    data_df = pd.DataFrame(
        columns=['Decay Time', 'Decay Time Error', 'States'])

    # quadratic formula with c=-1
    def decay_calculator(a, b):
        return (-b + np.sqrt(b**2 + 4 * a)) / (2 * a)

    def partial_f_a(a, b):
        return 1 / (a * np.sqrt(b**2 + 4 * a)) + (
            b - np.sqrt(b**2 + 4 * a)) / (2 * a**2)

    def partial_f_b(a, b):
        return ((b / np.sqrt(b**2 + 4 * a)) - 1) / (2 * a)

    def propagate_error(a, b, delta_a, delta_b):
        df_da = partial_f_a(a, b)
        df_db = partial_f_b(a, b)
        delta_f = np.sqrt((df_da * delta_a)**2 + (df_db * delta_b)**2)
        return delta_f

    for idx, t_str in enumerate(transition_string):
        print('\n')
        print('Now fitting', t_str)
        (amplitudes, amplitudes_errors, real_wait_times, states_for_legend,
         bussed, _, _) = plot_segmented_ramsey(t_str,
                                               bussed=bussed,
                                               plotting=False)

        x_new = np.linspace(real_wait_times.min(), real_wait_times.max(), 1000)
        if decay_type == 'exp':

            def decay_func(t, d, a):
                return a * np.exp(-t / d)

            d_guess = np.mean(real_wait_times)
            p0 = np.array([d_guess, 1])
            bounds = (
                [0, 0],  # Lower bounds
                [np.inf, 1])  # Upper bounds

        elif decay_type == 'convolved':

            def decay_func(t, d1, d2, a):
                return a * np.exp(-t / d1 - (t / d2)**2)

            d1_guess = np.mean(real_wait_times)
            d2_guess = np.mean(real_wait_times)

            p0 = np.array([d1_guess, d2_guess, 1])

            bounds = (
                [0, 0, 0],  # Lower bounds
                [np.inf, np.inf, 1])  # Upper bounds

        popt, pcov = sc.optimize.curve_fit(decay_func,
                                           real_wait_times,
                                           amplitudes,
                                           p0=p0,
                                           bounds=bounds,
                                           maxfev=10000)

        ax.plot(x_new,
                decay_func(x_new, *popt),
                color=colors[idx % 10],
                linestyle='--')

        if decay_type == 'exp':
            if bussed:
                ax.errorbar(real_wait_times,
                            amplitudes,
                            color=colors[idx % 10],
                            marker=markers[idx],
                            linestyle='none',
                            yerr=amplitudes_errors,
                            label='($m_S,F_D,m_D$)=' + str(states_for_legend) +
                            ';  ' + r'$T_2^*=$' + f'{np.round(popt[0],2)}')
            else:
                ax.errorbar(real_wait_times,
                            amplitudes,
                            color=colors[idx % 10],
                            marker=markers[idx],
                            linestyle='none',
                            yerr=amplitudes_errors,
                            label='($m_S,F_D,m_D$)=' + str(states_for_legend) +
                            ';  ' + r'$T_2^*=$' + f'{np.round(popt[0],2)}')

            decay_time = popt[0]
            decay_time_error = np.sqrt(pcov[0][0])

        elif decay_type == 'convolved':
            print(f'Exp decay: {np.round(popt[0],2)},' +
                  f' Gaussian decay: {np.round(popt[1],2)}')
            print(f'Exp uncertainty: {np.round(np.sqrt(pcov[0][0]),2)},' +
                  f'Gaussian uncertainty: {np.round(np.sqrt(pcov[1][1]),2)}')
            a = (1 / popt[1])**2
            a_error = 2 * np.sqrt(pcov[1][1]) / popt[1]**3
            b = 1 / popt[0]
            b_error = np.sqrt(pcov[0][0]) / (popt[0]**2)
            decay_time = (-b + np.sqrt(b**2 + 4 * a)) / (2 * a)
            decay_time_error = propagate_error(a, b, a_error, b_error)
            print('Convolved decay time', decay_time)
            print('Convolved error', decay_time_error)
            if bussed:
                ax.errorbar(
                    real_wait_times,
                    amplitudes,
                    color=colors[idx % 10],
                    yerr=amplitudes_errors,
                    marker=markers[idx],
                    linestyle='none',
                    # label='Bussed ($m_S,F_D,m_D$)=' +
                    # str(states_for_legend) + ';  ' + r'$T_2^*=$' +
                    # f'{np.round(decay_time,2)}',
                    label='($m_S,F_D,m_D$)=' + str(states_for_legend) + ';  ' +
                    r'$T_2^*=$' + f'{np.round(decay_time,2)}')
            else:
                ax.errorbar(real_wait_times,
                            amplitudes,
                            color=colors[idx % 10],
                            yerr=amplitudes_errors,
                            marker=markers[idx],
                            linestyle='none',
                            label='($m_S,F_D,m_D$)=' + str(states_for_legend) +
                            ';  ' + r'$T_2^*=$' + f'{np.round(decay_time,2)}')

        # Convert the states_for_legend string to a numpy array
        states_array = np.array(ast.literal_eval(states_for_legend))

        # Append the decay time, error, and states to the DataFrame
        data_df.loc[idx] = [decay_time, decay_time_error, states_array]

    ax.set_xlabel(r'Wait Time / $\mu s$')
    ax.set_ylabel('Ramsey Contrast')
    ax.set_ylim(-0.05, 1.05)
    # ax.set_xlim(5,2e5)
    ax.set_title('Dephasing Time Fits')
    ax.legend()
    ax.grid()

    # Convert 'States' to a JSON string before saving
    data_df['States'] = data_df['States'].apply(
        lambda x: json.dumps(x.tolist()))

    if bussed:
        plt.savefig('ramseys_segmented_PM/bussed_ramseys_decays.pdf')
        ax.set_xscale('log')
        plt.savefig('ramseys_segmented_PM/bussed_ramseys_decays_logscale.pdf')
        data_df.to_csv('ramseys_segmented_PM/bussed_decay_times.csv',
                       index=False)
    else:
        plt.savefig('ramseys_segmented_PM/direct_ramseys_decays.pdf')
        # ax.set_xscale('log')
        plt.savefig('ramseys_segmented_PM/direct_ramseys_decays_logscale.pdf')
        data_df.to_csv('ramseys_segmented_PM/direct_decay_times.csv',
                       index=False)

    return data_df


def find_magnetic_noise(df_name, plot=True, compare_coils=True):
    '''This function takes data on the T_2^* times of various transitions,
    with varying magnetic field sensitivities kappa, and fits a value for
    sigma_B, the magnitude of magnetic field noise, based on the relation
    T_2^* = 1/kappa*sigma_B.
    '''

    # create figure object up top
    plt.figure(figsize=set_size('half'))

    # old data for potential comparison
    coils_df_name = 'ramseys_segmented/bussed_decay_times.csv'

    # will need sensitivities for the comparison with decay times
    ba = Barium()
    sensitivities = ba.generate_transition_sensitivities()
    helpers = np.array([1, 5, 11, 19])

    # Define the function to do the fitting to get B-field noise
    def inverse(x, b_field_noise):
        return 1 / (b_field_noise * x)

    def read_fit_plot(df_name, color_idx):
        # Import decay time data from saved dataframe
        df = pd.read_csv(df_name)

        # Convert the 'States' column from JSON back to a list of lists
        df['States'] = df['States'].apply(json.loads)

        # Extract the relevant data
        decay_time = df['Decay Time']
        decay_time_error = df['Decay Time Error']

        coherence_times = []
        coherence_times_errors = []
        sensitivities_trans = []

        # Loop through the DataFrame row by row using iterrows
        for idx, row in df.iterrows():
            decay_time = row['Decay Time']
            decay_time_error = row['Decay Time Error']
            states = row[
                'States']  # Convert string representation of list to list

            # Print or process the states as needed
            # print(f'Row {idx}:, Decay Time={decay_time}, States={states}')
            # Example: You can now process each state in states

            coherence_times.append(decay_time)
            coherence_times_errors.append(decay_time_error)
            sens1 = sensitivities[int(helpers[int(states[0][1] - 1)] -
                                      states[0][2])][int(2 + states[0][0])]
            sens2 = sensitivities[int(helpers[int(states[1][1] - 1)] -
                                      states[1][2])][int(2 + states[1][0])]
            sensitivities_trans.append(np.abs(sens1 - sens2))

        sensitivities_trans = np.array(sensitivities_trans)
        coherence_times = np.array(coherence_times)
        coherence_times_errors = np.array(coherence_times_errors)

        # Fit the data to the inverse function
        popt, pcov = sc.optimize.curve_fit(inverse,
                                           sensitivities_trans,
                                           coherence_times,
                                           sigma=coherence_times_errors,
                                           absolute_sigma=True)

        # Extract the fitted parameter (b_field_noise)
        b_field_noise = popt[0]
        b_field_noise_err = np.sqrt(pcov[0][0])

        print(
            f'\n Fitted b_field_noise: {b_field_noise} ± {b_field_noise_err}')

        # Plot the data with error bars
        if color_idx == 1:
            label = 'Magnetic Coils'
        elif color_idx == 0:
            label = 'Permanent Magnets'

        plt.errorbar(sensitivities_trans,
                     coherence_times,
                     yerr=coherence_times_errors,
                     marker=markers[color_idx],
                     linestyle='none',
                     capsize=5,
                     label=label)

        # Plot the fitted curve
        x_fit = np.linspace(min(sensitivities_trans), max(sensitivities_trans),
                            100)
        y_fit = inverse(x_fit, b_field_noise)
        plt.plot(x_fit,
                 y_fit,
                 label=r'Fit to: $T_2^*$ = 1 / ($\kappa \sigma_B$),' +
                 r' $\sqrt{2}\pi\sigma_B$ = ' + f'{1e3*b_field_noise:.3f} mG',
                 color=colors[color_idx])

        # adding a reference line to play around with, just a guess
        # y_guess = inverse(x_fit,0.00005)
        # plt.plot(x_fit,
        #          y_guess,
        #          label=r'Guess of: $T_2^*$ = 1 / ($\kappa \sigma_B$),' +
        #          r' $\sigma_B$ = ' + f'{1e3*0.00002:.3f} mG',
        #          color=colors[color_idx+1])

        return b_field_noise

    b_field_noise = read_fit_plot(df_name, 0)

    if compare_coils:
        _ = read_fit_plot(coils_df_name, 1)

    # Set labels and title
    plt.xlabel('Qubit Sensitivity / MHz/G')
    plt.ylabel(r'Decay Time / $\mu s$')
    plt.yscale('log')
    # plt.xscale('log')
    plt.title('Magnetic Field Noise Fit')
    plt.grid(True)
    plt.legend()

    return b_field_noise


def find_laser_noise(df_name, b_field_noise=0.000069):

    # will need sensitivities for the comparison with decay times
    ba = Barium()
    sensitivities = ba.generate_transition_sensitivities()
    helpers = np.array([1, 5, 11, 19])

    # Define the function to do the fitting to get B-field noise
    def inverse(x, laser_noise):
        return (-laser_noise +
                np.sqrt((laser_noise)**2 + 4 *
                        (b_field_noise * x)**2)) / (2 * (b_field_noise * x)**2)

    # Import decay time data from saved dataframe
    df = pd.read_csv(df_name)

    # Convert the 'States' column from JSON back to a list of lists
    df['States'] = df['States'].apply(json.loads)

    # Extract the relevant data
    decay_time = df["Decay Time"]
    decay_time_error = df["Decay Time Error"]

    coherence_times = []
    coherence_times_errors = []
    sensitivities_trans = []

    # Loop through the DataFrame row by row using iterrows
    for idx, row in df.iterrows():
        decay_time = row["Decay Time"]
        decay_time_error = row["Decay Time Error"]
        states = row[
            "States"]  # Convert string representation of list to actual list

        # Print or process the states as needed
        # print(f'Row {idx}:, Decay Time = {decay_time}, States = {states}')
        # Example: You can now process each state in states

        coherence_times.append(decay_time)
        coherence_times_errors.append(decay_time_error)
        sens1 = sensitivities[int(helpers[int(states[1] - 1)] -
                                  states[2])][int(2 + states[0])]
        sensitivities_trans.append(np.abs(sens1))

    sensitivities_trans = np.array(sensitivities_trans)
    coherence_times = np.array(coherence_times)
    coherence_times_errors = np.array(coherence_times_errors)

    initial_guess = [0.1]

    # Fit the data to the inverse function
    popt, pcov = sc.optimize.curve_fit(inverse,
                                       sensitivities_trans,
                                       coherence_times,
                                       sigma=coherence_times_errors,
                                       p0=initial_guess)

    # Extract the fitted parameter (b_field_noise)
    laser_noise = popt[0]
    laser_noise_err = np.sqrt(pcov[0][0])

    print(f"\n Fitted laser noise: {laser_noise}, error: {laser_noise_err}")

    # Plot the data with error bars
    plt.figure(figsize=(8, 6))
    plt.errorbar(sensitivities_trans,
                 coherence_times,
                 yerr=coherence_times_errors,
                 capsize=5,
                 label="Decay Time")

    # Plot the fitted curve
    x_fit = np.linspace(min(sensitivities_trans), max(sensitivities_trans),
                        10000)
    y_fit = inverse(x_fit, laser_noise)
    plt.plot(x_fit,
             y_fit,
             label=r'Fit to $\Delta_L$' + f' = {np.round(1e6*laser_noise)} Hz',
             color=colors[0])

    # Set labels and title
    plt.xlabel("Transition Sensitivity / MHzG$^{-1}$")
    plt.ylabel(r"Decay Time / $\mu s$")
    # plt.yscale('log')
    # plt.xscale('log')
    plt.title("Decay Time vs Sensitivity - Fit")
    plt.grid(True)
    plt.legend()

    return None


if __name__ == '__main__':
    start_time = time.time()

    # Permanent Magnets Installed 2025-03-11 ##################################
    ###########################################################################

    ###########################################################################
    # direct ramseys for laser noise

    trans_string = [
        'qubit[0, 2, 0]',  # 0,2,0
    ]

    # for idx, t_str in enumerate(trans_string):
    #     get_threshold(t_str)
    #     plot_segmented_ramsey(t_str, bussed=False, fit_type='sine')

    fit_segmented_decay(trans_string, decay_type='convolved', bussed=False)

    # # find_magnetic_noise('ramseys_segmented/bussed_decay_times.csv')
    # # find_laser_noise('ramseys_segmented/direct_decay_times.csv')

    # ###########################################################################
    # # bussed ramseys to get the magnetic field noise level
    # trans_string = [
    #     '[0, 3, 2], [0, 2, 2]',  # 20 kHz/G
    #     # '[0, 3, -2], [0, 2, -1]',  # 0.25 MHz/G (very weak transitions here!)
    #     '[0, 4, -1], [0, 2, -2]',  # 0.4 MHz/G
    #     '[0, 3, 1], [0, 3, 2]',  # 1.0 MHz/G
    #     '[0, 3, 1], [0, 2, 0]',  # 2.1 MHz/G
    #     '[0, 2, 0], [0, 2, 2]',  # 3 MHz/G
    #     '[0, 4, 0], [0, 3, 2]',  # 4.6 MHz/G
    #     '[0, 4, -1], [0, 3, 2]',  # 5.6 MHz/G
    #     '[0, 4, -2], [0, 2, 2]',  # 6.6 MHz/G
    # ]

    # for idx, t_str in enumerate(trans_string):
    #     # get_threshold(t_str)
    #     plot_segmented_ramsey(t_str, bussed=True, fit_type='sine')

    # fit_segmented_decay(trans_string, decay_type='convolved', bussed=True)

    # find_magnetic_noise('ramseys_segmented_PM/bussed_decay_times.csv',
    #                     compare_coils=True)
    # find_laser_noise('ramseys_segmented_PM/direct_decay_times.csv')

    ###########################################################################
    # make some sample probability distributions based on the measured results
    # # Define the Gaussian function
    # def gaussian(x, mean, stddev):
    #     return (1/(stddev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / stddev) ** 2)

    # # Create an array of x values
    # x = np.linspace(-200, 200, 1000)

    # # Define parameters for two Gaussian distributions
    # mean1, stddev1 = 0, 14.4  # First Gaussian: mean = 0, width = 1.0
    # mean2, stddev2 = 0, 61.4  # Second Gaussian: mean = 0, width = 2.0

    # # Compute the Gaussian function values
    # y1 = gaussian(x, mean1, stddev1)
    # y2 = gaussian(x, mean2, stddev2)

    # # Plot the Gaussian functions
    # plt.plot(x, y1, label=r'Permanent Magnets ($\sigma$=14.4)', color=colors[0])
    # plt.plot(x, y2, label=r'EM Coil ($\sigma$=61.4)', color=colors[1])

    # # Adding labels and grid
    # plt.xlabel(r'B-field fluctuations / $\mu$G')
    # plt.ylabel('Probability Density')
    # plt.grid(True)  # Enable grid
    # plt.legend()     # Show legend


    # # Define the Voigt profile function
    # def voigt(x, sigma, gamma):
    #     """Calculate the Voigt profile.

    #     Parameters:
    #     x (array-like): The variable over which to calculate the profile.
    #     sigma (float): The Gaussian width (standard deviation).
    #     gamma (float): The Lorentzian width (half-width).

    #     Returns:
    #     array-like: Values of the Voigt profile.
    #     """
    #     z = (x + 1j * gamma) / (sigma * np.sqrt(2))
    #     return np.real(wofz(z))

    # # Create an array of x values
    # x = np.linspace(-1000, 1000, 1000)

    # # Define parameters for the Voigt profile
    # sigma = 81.6  # Gaussian width
    # gamma = 77.1  # Lorentzian width

    # # Compute the Voigt profile values
    # y = voigt(x, sigma, gamma)

    # # Plot the Voigt profile
    # plt.plot(x, y, label=r'Voigt Profile ($\sigma$=81.6, $\gamma$=77.1)', color=colors[0])

    # # Adding labels and grid
    # plt.xlabel(r'$f$ fluctuations / Hz')
    # plt.ylabel('Intensity')
    # plt.title('Voigt Profile')
    # plt.grid(True)  # Enable grid
    # plt.legend()     # Show legend

    plt.show()

    print("--- %s seconds ---" % (time.time() - start_time))
