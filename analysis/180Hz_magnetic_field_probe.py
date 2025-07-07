import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from class_barium import Barium
from plot_utils import nice_fonts  # set_size

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

mpl.rcParams.update(nice_fonts)
np.set_printoptions(suppress=True)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']*2


def plot_line_signal(num_exp: int = 250,
                     Ramsey_wait_time=100,
                     active_compensation=True,
                     share_reference=False):

    # need sensitivities
    ba = Barium()
    ba.generate_transition_sensitivities()
    insensB_sens = ba.transition_sensitivities[22, 1]
    sensB_sens = ba.transition_sensitivities[5, 2]

    # Ramsey error based on wait time and experiment number
    Ramsey_fit_error = np.sqrt(
        (2 * 0.5 * 0.5) / num_exp) / (2 * np.pi * Ramsey_wait_time)  # in kHz

    # Define the directory containing the relevant files
    if active_compensation:
        folder_path_root = '../data/ramseys_calibrations_line_signal_data/'
    else:
        folder_path_root = ('../data/ramseys_calibrations_line_signal_data/' +
                            'no_active_compensation')
    file_string = 'RamseyWait' + str(Ramsey_wait_time) + 'us'

    # Collect all files with time modified
    # files_with_mtime = [
    #     (f, os.path.getmtime(os.path.join(folder_path_root, f)))
    #     for f in os.listdir(folder_path_root)
    #     if f.startswith(filename_root)
    # ]

    # Collect files that contain the desired string and retrieve their times
    # modified
    files_with_mtime = [
        (f, os.path.getmtime(os.path.join(folder_path_root, f)))
        for f in os.listdir(folder_path_root) if file_string in
        f  # Ensure we are filtering based on the correct file_string
    ]

    # Sort the collected files by modified time
    sorted_files = sorted(files_with_mtime, key=lambda x: x[1])

    # Extract just the filenames from the sorted list
    filenames = [f[0] for f in sorted_files]
    # print(filenames)

    # LT_waits = []
    # insensB_freq = []
    # sensB_freq = []
    # mag_fields = []
    # mag_fields_error = []

    LT_waits = np.full((10, 40), np.nan)
    insensB_freq = np.full((10, 40), np.nan)
    sensB_freq = np.full((10, 40), np.nan)
    mag_fields = np.full((10, 40), np.nan)
    mag_fields_error = np.full((10, 40), np.nan)

    print('Number of Ramsey calibration files included:', len(filenames))

    itera = 0
    marker = -1
    # Loop through each file and extract the required information
    for idx, filename in enumerate(filenames):
        file_path = os.path.join(folder_path_root, filename)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # Attempt to read and convert the first line
            if lines:
                LT_wait = float(lines[0].strip())
                if LT_wait == 0:
                    marker = int(marker + 1)
                    itera = 0
                # LT_waits.append(LT_wait)
                LT_waits[marker, itera] = LT_wait
                # print(LT_wait)

                if len(lines) > 1:
                    # Extract the second line
                    tuple_line = eval(lines[1].strip())
                    # insensB_freq.append(tuple_line[0])
                    # sensB_freq.append(tuple_line[1])

                    insensB_freq[marker, itera] = tuple_line[0]
                    sensB_freq[marker, itera] = tuple_line[1]

                    if share_reference:
                        if idx == 0:
                            first_delta_freq = tuple_line[1] - tuple_line[0]
                    else:
                        if LT_wait == 0:
                            first_delta_freq = tuple_line[1] - tuple_line[0]

                    # first_delta_freq = sensB_freq[0] - insensB_freq[0]

                    current_delta_freq = tuple_line[1] - tuple_line[0]

                    new_mag_value = (current_delta_freq - first_delta_freq) / (
                        sensB_sens - insensB_sens)

                    mag_fields_error_value = (2 * Ramsey_fit_error) / (
                        sensB_sens - insensB_sens)

                    # mag_fields.append(new_mag_value)
                    # mag_fields_error.append(mag_fields_error_value)
                    mag_fields[marker, itera] = new_mag_value
                    mag_fields_error[marker, itera] = mag_fields_error_value

                    # if idx ==1:
                    #     breakpoint()
            itera += 1

    # print('LT_waits:', LT_waits)
    # print('insensB_freq:', insensB_freq)
    # print('sensB_freq:', sensB_freq)

    # Create a figure and two vertically stacked subplots
    fig, ax1 = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    for plotter in range(marker + 1):
        if share_reference:
            offset_insensB = insensB_freq[0, 0]
            offset_sensB = sensB_freq[0, 0]
        else:
            offset_insensB = insensB_freq[plotter, 0]
            offset_sensB = sensB_freq[plotter, 0]

        # Plot insensB_freq vs LT_waits
        ax1[0].errorbar(1e-3 * LT_waits[plotter, :],
                        1e3 * (insensB_freq[plotter, :] - offset_insensB),
                        yerr=1e3 * Ramsey_fit_error,
                        fmt='o',
                        color=colors[plotter])
        # Plot sensB_freq vs LT_waits
        ax1[1].errorbar(1e-3 * LT_waits[plotter, :],
                        1e3 * (sensB_freq[plotter, :] - offset_sensB),
                        yerr=1e3 * Ramsey_fit_error,
                        fmt='o',
                        color=colors[plotter])

    ax1[0].set_title(r'$f_0$ Frequency Change', fontsize=14)
    ax1[0].set_ylabel(r'$\Delta f_0$ / kHz', fontsize=12)
    ax1[0].grid()
    ax1[1].set_title(r'$f_1$ Frequency Change', fontsize=14)
    ax1[1].set_xlabel('Line Trigger Wait / $ms$', fontsize=12)
    ax1[1].set_ylabel(r'$\Delta f_1$ / kHz', fontsize=12)
    ax1[1].grid()

    # Adjust layout for better spacing
    plt.tight_layout()

    # def sine_fit_function(x, a1, a2, phi1, phi2, b):
    #     return a1 * np.sin(2 * np.pi * 60 * x * 1e-6 + phi1) + a2 * np.sin(
    #         2 * np.pi * 180 * x * 1e-6 + phi2) + b

    def sine_fit_function(x, a1, a2, phi1, phi2, b):
        return a1 * np.sin(2 * np.pi * 60 * x * 1e-6 + phi1) + a2 * np.sin(
            2 * np.pi * 180 * x * 1e-6 + phi2) + b

    # Estimate amplitudes based on the range of mag_fields
    amp_estimate = (np.nanmax(mag_fields) -
                    np.nanmin(mag_fields)) / 4  # Rough guess

    # Initial guess for the parameters [a1, a2, phi1, phi2, b]
    initial_guess = [amp_estimate, amp_estimate, 0, 0, np.nanmean(mag_fields)]

    LT_waits_1d = LT_waits[~np.isnan(LT_waits)]
    mag_fields_1d = mag_fields[~np.isnan(mag_fields)]

    # Perform curve fitting
    popt, _ = curve_fit(sine_fit_function,
                        LT_waits_1d,
                        mag_fields_1d,
                        p0=initial_guess)

    a1, a2, phi1, phi2, b = popt

    # Generate a range of x values for the fitted curve
    LT_waits_fit = np.linspace(np.nanmin(LT_waits), np.nanmax(LT_waits), 100)
    mag_fields_fit = sine_fit_function(LT_waits_fit, *popt)

    mag_fields_fit_difference = mag_fields - sine_fit_function(LT_waits, *popt)
    mag_fields_diff_from_mean = mag_fields - np.nanmean(mag_fields)

    pure_60 = a1 * np.sin(2 * np.pi * 60 * LT_waits_fit * 1e6 + phi1) + b
    pure_180 = a2 * np.sin(2 * np.pi * 180 * LT_waits_fit * 1e6 + phi2) + b

    # Plot and fit magnetic field fluctuations
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    for plotter in range(marker + 1):
        ax[0].errorbar(1e-3 * LT_waits[plotter, :],
                       1e3 * mag_fields[plotter, :],
                       yerr=1e3 * np.abs(mag_fields_error[plotter, :]),
                       fmt='o',
                       color=colors[plotter])
        ax[1].errorbar(1e-3 * LT_waits[plotter, :],
                       np.abs(1e3 * mag_fields_fit_difference[plotter, :]),
                       yerr=1e3 * np.abs(mag_fields_error[plotter, :]),
                       fmt='o',
                       color=colors[plotter])
        ax[1].errorbar(1e-3 * LT_waits[plotter, :],
                       np.abs(1e3 * mag_fields_diff_from_mean[plotter, :]),
                       yerr=1e3 * np.abs(mag_fields_error[plotter, :]),
                       fmt='o',
                       alpha=0.25,
                       color=colors[plotter])

    # ax[0].axhline(y=1e3 * np.nanmean(mag_fields),
    #               label=r'Mean $\Delta B$',
    #               color='k',
    #               linestyle='--',
    #               alpha=0.25)
    ax[0].plot(1e-3 * LT_waits_fit,
               1e3 * mag_fields_fit,
               label='Fitted Mixed Wave',
               color='k')
    ax[1].axhline(y=np.nanmean(np.abs(1e3 * mag_fields_fit_difference)),
                  color='black',
                  label='Discrepancy, fit vs. measured values',
                  linestyle='--')
    ax[1].axhline(y=np.nanmean(np.abs(1e3 * mag_fields_diff_from_mean)),
                  color='grey',
                  label='Discrepancy, mean vs. measured values',
                  linestyle='--')

    # plot some pure waves as eye guides
    ax[0].plot(1e-3 * LT_waits_fit,
               1e3 * pure_60,
               label='Pure 60 Hz Wave',
               color='grey',
               linestyle='--')
    ax[0].plot(1e-3 * LT_waits_fit,
               1e3 * pure_180,
               label='Pure 180 Hz Wave',
               color='grey',
               linestyle='--')
    ax[0].set_title('Magnetic Field Changes vs. Line Signal')
    plt.xlabel(r'Line Trigger Wait / $ms$')
    ax[0].set_ylabel(r'$\Delta B$ / mG')
    ax[1].set_ylabel('Fit vs. Meas. / mG')
    ax[0].grid()
    ax[0].legend()
    ax[1].grid()
    ax[1].legend()
    plt.tight_layout()

    print(f'Amplitude of 60 Hz component: {1e3*a1:.3f} mG.')
    print(f'Phase of 60 Hz component (radians): {phi1:.3f}.\n')

    print(f'Amplitude of 180 Hz component: {1e3*a2:.3f} mG.')
    print(f'Phase of 180 Hz component (radians): {phi2:.3f}.\n')

    print(
        'Average difference between fit and measurement:' +
        f' {np.round(1e3*np.nanmean(np.abs(mag_fields_fit_difference)),3)} mG.'
    )

    print(
        'Average difference between mean and measured values:' +
        f' {np.round(1e3*np.nanmean(np.abs(mag_fields_diff_from_mean)),3)} mG.'
    )
    print(f'Fit offset: {b} Gauss.\n')
    return None


def self_referenced_line_signal(num_exp: int = 250,
                                Ramsey_wait_time=100,
                                field_source='magnet'):

    # need sensitivities
    ba = Barium()
    ba.generate_transition_sensitivities()
    insensB_sens = ba.transition_sensitivities[22, 1]
    sensB_sens = ba.transition_sensitivities[5, 2]

    # Ramsey error based on wait time and experiment number
    Ramsey_fit_error = np.sqrt(
        (2 * 0.5 * 0.5) / num_exp) / (2 * np.pi * Ramsey_wait_time)  # in kHz

    # Define the directory containing the relevant files
    if field_source == 'coil':
        folder_path_root = ('../data/ramseys_calibrations_line_signal_data' +
                            '/proper_referencing/coil_data')
    elif field_source == 'magnet':
        folder_path_root = ('../data/ramseys_calibrations_line_signal_data' +
                            '/proper_referencing/magnet_data_coils_grounded')
    file_string = 'RamseyWait' + str(Ramsey_wait_time) + 'us'

    # Collect files that contain the desired string and retrieve their times
    # modified
    files_with_mtime = [
        (f, os.path.getmtime(os.path.join(folder_path_root, f)))
        for f in os.listdir(folder_path_root) if file_string in
        f  # Ensure we are filtering based on the correct file_string
    ]

    # Sort the collected files by modified time
    sorted_files = sorted(files_with_mtime, key=lambda x: x[1])

    # Extract just the filenames from the sorted list
    filenames = [f[0] for f in sorted_files]
    # print(filenames)

    LT_waits = np.full((20, 40), np.nan)
    insensB_freq = np.full((20, 40), np.nan)
    sensB_freq = np.full((20, 40), np.nan)
    insensB_freq_ref = np.full((20, 40), np.nan)
    sensB_freq_ref = np.full((20, 40), np.nan)
    mag_fields = np.full((20, 40), np.nan)
    ref_mag_fields = np.full((20, 40), np.nan)
    mag_fields_error = np.full((20, 40), np.nan)

    print('Number of Ramsey calibration files included:', len(filenames))

    itera = 0
    marker = -1
    # Loop through each file and extract the required information
    for idx, filename in enumerate(filenames):
        file_path = os.path.join(folder_path_root, filename)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # Attempt to read and convert the first line
            if lines:
                LT_wait = float(lines[0].strip())
                if LT_wait == 0:
                    marker = int(marker + 1)
                    itera = 0

                LT_waits[marker, itera] = LT_wait
                # print(LT_wait)

                if len(lines) > 1:
                    # Extract the very first reference

                    # Extract the second line
                    tuple_line = eval(lines[1].strip())
                    ref_tuple_line = eval(lines[3].strip())

                    if LT_wait == 0:
                        first_ref_delta_freq = ref_tuple_line[
                            1] - ref_tuple_line[0]

                    insensB_freq[marker, itera] = tuple_line[0]
                    sensB_freq[marker, itera] = tuple_line[1]

                    insensB_freq_ref[marker, itera] = ref_tuple_line[0]
                    sensB_freq_ref[marker, itera] = ref_tuple_line[1]

                    ref_delta_freq = ref_tuple_line[1] - ref_tuple_line[0]
                    current_delta_freq = tuple_line[1] - tuple_line[0]

                    ref_new_mag_value = (ref_delta_freq - first_ref_delta_freq
                                         ) / (sensB_sens - insensB_sens)

                    new_mag_value = (current_delta_freq - ref_delta_freq) / (
                        sensB_sens - insensB_sens)

                    mag_fields_error_value = (2 * Ramsey_fit_error) / (
                        sensB_sens - insensB_sens)

                    # mag_fields.append(new_mag_value)
                    # mag_fields_error.append(mag_fields_error_value)
                    ref_mag_fields[marker, itera] = ref_new_mag_value
                    mag_fields[marker, itera] = new_mag_value
                    mag_fields_error[marker, itera] = mag_fields_error_value
            itera += 1

    # print('LT_waits:', LT_waits)
    # print('insensB_freq:', insensB_freq)
    # print('sensB_freq:', sensB_freq)

    # Create a figure and two vertically stacked subplots
    fig, ax1 = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    for plotter in range(marker + 1):
        # if share_reference:
        #     offset_insensB = insensB_freq[0, 0]
        #     offset_sensB = sensB_freq[0, 0]
        # else:
        # offset_insensB = insensB_freq[plotter, 0]
        # offset_sensB = sensB_freq[plotter, 0]

        # Plot insensB_freq vs LT_waits
        ax1[0].errorbar(
            1e-3 * LT_waits[plotter, :],
            1e3 * (insensB_freq[plotter, :] - insensB_freq_ref[plotter, :]),
            yerr=1e3 * Ramsey_fit_error,
            fmt='o',
            color=colors[plotter])
        # Also plot difference in references over time, faded out
        ax1[0].errorbar(
            1e-3 * LT_waits[plotter, :],
            1e3 *
            (insensB_freq_ref[plotter, :] - insensB_freq_ref[plotter, 0]),
            yerr=1e3 * Ramsey_fit_error,
            fmt='o',
            alpha=0.25,
            color=colors[plotter])
        # Plot sensB_freq vs LT_waits
        ax1[1].errorbar(1e-3 * LT_waits[plotter, :],
                        1e3 *
                        (sensB_freq[plotter, :] - sensB_freq_ref[plotter, :]),
                        yerr=1e3 * Ramsey_fit_error,
                        fmt='o',
                        color=colors[plotter])
        ax1[1].errorbar(
            1e-3 * LT_waits[plotter, :],
            1e3 * (sensB_freq_ref[plotter, :] - sensB_freq_ref[plotter, 0]),
            yerr=1e3 * Ramsey_fit_error,
            fmt='o',
            alpha=0.25,
            color=colors[plotter])

    ax1[0].set_title(r'$f_0$ Frequency Change', fontsize=14)
    ax1[0].set_ylabel(r'$\Delta f_0$ / kHz', fontsize=12)
    ax1[0].grid()
    ax1[1].set_title(r'$f_1$ Frequency Change', fontsize=14)
    ax1[1].set_xlabel('Line Trigger Wait / $ms$', fontsize=12)
    ax1[1].set_ylabel(r'$\Delta f_1$ / kHz', fontsize=12)
    ax1[1].grid()

    # Adjust layout for better spacing
    plt.tight_layout()

    # def sine_fit_function(x, a1, a2, phi1, phi2, b):
    #     return a1 * np.sin(2 * np.pi * 60 * x * 1e-6 + phi1) + a2 * np.sin(
    #         2 * np.pi * 180 * x * 1e-6 + phi2) + b

    def sine_fit_function(x, a1, a2, phi1, phi2, b):
        return a1 * np.sin(2 * np.pi * 60 * x * 1e-6 + phi1) + a2 * np.sin(
            2 * np.pi * 180 * x * 1e-6 + phi2) + b

    # Estimate amplitudes based on the range of mag_fields
    amp_estimate = (np.nanmax(mag_fields) -
                    np.nanmin(mag_fields)) / 4  # Rough guess

    # Initial guess for the parameters [a1, a2, phi1, phi2, b]
    initial_guess = [amp_estimate, amp_estimate, 0, 0, np.nanmean(mag_fields)]

    LT_waits_1d = LT_waits[~np.isnan(LT_waits)]
    mag_fields_1d = mag_fields[~np.isnan(mag_fields)]

    # Perform curve fitting
    popt, _ = curve_fit(sine_fit_function,
                        LT_waits_1d,
                        mag_fields_1d,
                        p0=initial_guess)

    a1, a2, phi1, phi2, b = popt

    # Generate a range of x values for the fitted curve
    LT_waits_fit = np.linspace(np.nanmin(LT_waits), np.nanmax(LT_waits), 100)
    mag_fields_fit = sine_fit_function(LT_waits_fit, *popt)

    mag_fields_fit_difference = mag_fields - sine_fit_function(LT_waits, *popt)
    mag_fields_diff_from_mean = mag_fields - np.nanmean(mag_fields)

    pure_60 = a1 * np.sin(2 * np.pi * 60 * LT_waits_fit * 1e6 + phi1) + b
    pure_180 = a2 * np.sin(2 * np.pi * 180 * LT_waits_fit * 1e6 + phi2) + b

    # Plot and fit magnetic field fluctuations
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    for plotter in range(marker + 1):
        ax[0].errorbar(1e-3 * LT_waits[plotter, :],
                       1e3 * mag_fields[plotter, :],
                       yerr=1e3 * np.abs(mag_fields_error[plotter, :]),
                       fmt='o',
                       color=colors[plotter])
        ax[0].errorbar(1e-3 * LT_waits[plotter, :],
                       1e3 * ref_mag_fields[plotter, :],
                       yerr=1e3 * np.abs(mag_fields_error[plotter, :]),
                       fmt='o',
                       alpha=0.25,
                       color=colors[plotter])
        ax[1].errorbar(1e-3 * LT_waits[plotter, :],
                       np.abs(1e3 * mag_fields_fit_difference[plotter, :]),
                       yerr=1e3 * np.abs(mag_fields_error[plotter, :]),
                       fmt='o',
                       color=colors[plotter])
        ax[1].errorbar(1e-3 * LT_waits[plotter, :],
                       np.abs(1e3 * mag_fields_diff_from_mean[plotter, :]),
                       yerr=1e3 * np.abs(mag_fields_error[plotter, :]),
                       fmt='o',
                       alpha=0.25,
                       color=colors[plotter])

    # ax[0].axhline(y=1e3 * np.nanmean(mag_fields),
    #               label=r'Mean $\Delta B$',
    #               color='k',
    #               linestyle='--',
    #               alpha=0.25)
    ax[0].plot(1e-3 * LT_waits_fit,
               1e3 * mag_fields_fit,
               label='Fitted Mixed Wave',
               color='k')
    ax[1].axhline(y=np.nanmean(np.abs(1e3 * mag_fields_fit_difference)),
                  color='black',
                  label='Discrepancy, fit vs. measured values',
                  linestyle='--')
    ax[1].axhline(y=np.nanmean(np.abs(1e3 * mag_fields_diff_from_mean)),
                  color='grey',
                  label='Discrepancy, mean vs. measured values',
                  linestyle='--')

    # plot some pure waves as eye guides
    ax[0].plot(1e-3 * LT_waits_fit,
               1e3 * pure_60,
               label='Pure 60 Hz Wave',
               color='grey',
               linestyle='--')
    ax[0].plot(1e-3 * LT_waits_fit,
               1e3 * pure_180,
               label='Pure 180 Hz Wave',
               color='grey',
               linestyle='--')
    ax[0].set_title('Magnetic Field Changes vs. Line Signal')
    plt.xlabel(r'Line Trigger Wait / $ms$')
    ax[0].set_ylabel(r'$\Delta B$ / mG')
    ax[1].set_ylabel('Fit vs. Meas. / mG')
    ax[0].grid()
    ax[0].legend()
    ax[1].grid()
    ax[1].legend()
    plt.tight_layout()

    print(f'Amplitude of 60 Hz component: {1e3*a1/2:.3f} mG.')
    print(f'Phase of 60 Hz component (radians): {phi1:.3f}.\n')

    print(f'Amplitude of 180 Hz component: {1e3*a2/2:.3f} mG.')
    print(f'Phase of 180 Hz component (radians): {phi2:.3f}.\n')

    print(
        'Average difference between fit and measurement:' +
        f' {np.round(1e3*np.nanmean(np.abs(mag_fields_fit_difference)),3)} mG.'
    )

    print(
        'Average difference between mean and measured values:' +
        f' {np.round(1e3*np.nanmean(np.abs(mag_fields_diff_from_mean)),3)} mG.'
    )
    print(f'Fit offset: {b}.\n')
    return None


if __name__ == '__main__':

    self_referenced_line_signal(Ramsey_wait_time=50, field_source='magnet')

    plt.show()
