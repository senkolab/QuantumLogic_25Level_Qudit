import os
import re
# import sys
import time
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc  # type: ignore
from matplotlib.cm import get_cmap
from tqdm import tqdm

from class_barium import Barium
from plot_utils import nice_fonts, set_size

mpl.rcParams.update(nice_fonts)
np.set_printoptions(suppress=True)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


class Calibrations(Barium):
    """Handles calibration analysis for Barium137 transitions, including
    retrieval and processing of frequency and pi-time data, fitting routines,
    and visualization of results.

    Attributes:

        F2M0f2m0 (float): Default frequency value for calibration.

        use_old_data (bool): Flag indicating whether to use older calibration
        data.

        exp_number (int): Experimental run number.

        data_cutoff_date (str): Date string used to filter out old data.

        data_end_date (str): End date for the calibration data.

        folder_path_root (str): Root folder path for data files.

        folder_path_raw (str): Path to the raw calibration data files.

        filename_root (str): Base name for calibration data files.

        file_names (list): List of files retrieved for calibration analysis.

        sim_freqs (np.ndarray): Array of simulated transition frequencies.

        sim_sensitivities (np.ndarray): Array of simulated transition
        sensitivities.

        good_freq_runs (np.ndarray): Array indicating successful
        frequency runs.

        chunked_freq_files (np.ndarray): Chunked frequency files associated
        with calibration.

        chunked_freq_fits (np.ndarray): Fits for the chunked frequency data.

        chunked_freq_fit_errors (np.ndarray): Errors corresponding to the
        fitted chunked frequency data.

        chunked_datetimes (np.ndarray): Datetime strings associated with the
        chunked frequency data.

       an1_an2_data (list): Fitting parameters for transitions.

        an1_an2_data_uncertainty (list): Uncertainties associated with the
        fitting parameters.

    """

    def __init__(self,
                 F2M0f2m0: float = 545.34783,
                 use_old_data: bool = False):
        super().__init__()

        self.F2M0f2m0 = F2M0f2m0
        # self.B_field = B_field
        self.use_old_data = use_old_data

        self.exp_number = 101
        # self.data_cutoff_date = '2024-04-25'
        # self.data_cutoff_date = '2024-05-05'
        # self.data_end_date = '2024-09-15'

        # cutoff dates for old frequency scan data
        self.data_cutoff_date = '2024-05-10'
        self.data_end_date = '2024-08-01'

        self.folder_path_raw = ('../data/frequency_scan_calibration_data')
        self.filename_root = 'New_D52_calibration_raw_data_'

        # First grab all the calibration files from the folder
        file_names_unsorted = [
            f for f in os.listdir(self.folder_path_raw)
            if f.startswith(self.filename_root)
        ]

        self.file_names = sorted(file_names_unsorted,
                                 key=lambda f: os.path.getmtime(
                                     os.path.join(self.folder_path_raw, f)))

        if self.use_old_data:
            pass
        else:
            # filter out data older than April 25th, 2024
            # this is when we changed the way we do frequency plotting
            cutoff_date = datetime.strptime(self.data_cutoff_date, "%Y-%m-%d")
            end_date = datetime.strptime(self.data_end_date, "%Y-%m-%d")

            filtered_file_names = [
                f for f in self.file_names if datetime.fromtimestamp(
                    os.path.getmtime(os.path.join(self.folder_path_raw, f))) >
                cutoff_date and datetime.fromtimestamp(
                    os.path.getmtime(os.path.join(self.folder_path_raw,
                                                  f))) < end_date
            ]
            self.file_names = filtered_file_names

        # self.file_names = self.file_names[-64:]
        # print('Number of old calibration files included:',
        #       len(self.file_names))
        # print('First and last file names:', self.file_names[0],
        #       self.file_names[-1])

    def get_sim_freqs(self):
        """Calculates and obtains simulated transition frequencies and
        sensitivities for the S12 to D52 transitions based on frequencies
        defined in the data.

        Returns:

            transition_frequencies (np.ndarray): An array of transition
            frequencies.

            transition_sensitivities (np.ndarray): An array of transition
            sensitivities.

        """

        self.B_field_range = np.append(
            np.linspace(0.001, self.B_field - 0.001, 10),
            np.array([self.B_field, self.B_field + 0.001]))

        self.B_field_range = self.B_field_range
        self.data_S12 = self.dataframe_evals_estates(orbital='S12')
        self.data_D52 = self.dataframe_evals_estates(orbital='D52')
        # generate the delta_m value for each transition
        if hasattr(self, 'delta_m'):
            pass
        else:
            self.generate_delta_m_table()
        delta_m = self.delta_m[:, -5:]

        transitions = np.empty((24, 5))
        delta_m_TF = np.zeros_like(transitions, dtype=bool)
        for it, D52_state in enumerate(self.F_values_D52):
            for it2, S12_state in enumerate(self.F_values_S12[-5:]):
                if np.absolute(delta_m[it, it2]) <= 2:
                    delta_m_TF[it, it2] = True
                    transitions[it, it2] = self.data_D52[
                        str(D52_state) +
                        ' evalue'].iloc[-2] - self.data_S12[str(S12_state) +
                                                            ' evalue'].iloc[-2]
                elif np.absolute(delta_m[it, it2]) > 2:
                    delta_m_TF[it, it2] = False
                    transitions[it, it2] = np.nan
        transition_frequencies = np.flip(-transitions, axis=0)

        transition_frequencies = transition_frequencies - (
            transition_frequencies[5, 2] - self.F2M0f2m0)
        self.sim_freqs = transition_frequencies

        sensitivities = np.zeros((24, 5))
        for it, D52_state in enumerate(self.F_values_D52):
            for it2, S12_state in enumerate(self.F_values_S12[-5:]):
                if np.absolute(delta_m[it, it2]) <= 2:
                    sensitivities[it, it2] = (
                        (self.data_D52[str(D52_state) + ' evalue'].iloc[-1] -
                         self.data_D52[str(D52_state) + ' evalue'].iloc[-3]) -
                        (self.data_S12[str(S12_state) + ' evalue'].iloc[-1] -
                         self.data_S12[str(S12_state) + ' evalue'].iloc[-3])
                    ) / (self.data_S12['B field value'].iloc[-1] -
                         self.data_S12['B field value'].iloc[-3])
                elif np.absolute(delta_m[it, it2]) > 2:
                    sensitivities[it, it2] = np.nan

        transition_sensitivities = np.flip(sensitivities, axis=0)
        self.sim_sensitivities = transition_sensitivities

        return transition_frequencies, transition_sensitivities

    # defining a helper function for finding Wilson intervals
    # def find_errors(self, num_SD, PD, exp_num):
    #     upper_error = ((PD + (num_SD**2 / (2 * exp_num))) /
    #                    (1 + (num_SD**2 / exp_num))) + (np.sqrt(
    #                        ((PD * (1 - PD) * num_SD**2) / exp_num) +
    #                        (num_SD**4 /
    #                         (4 * exp_num**2)))) / (1 + (num_SD**2 / exp_num))

    #     lower_error = ((PD + (num_SD**2 / (2 * exp_num))) /
    #                    (1 + (num_SD**2 / exp_num))) - (np.sqrt(
    #                        ((PD * (1 - PD) * num_SD**2) / exp_num) +
    #                        (num_SD**4 /
    #                         (4 * exp_num**2)))) / (1 + (num_SD**2 / exp_num))

    #     return lower_error, upper_error

    def fit_freq_Rabi_fringe(self,
                             create_plots=False,
                             plot_individual_scans=False):
        """Fits frequency data to Rabi fringe patterns and calculates the
        associated parameters.

        Parameters:

            create_plots (bool): If True, generates plots of the fits.

            plot_individual_scans (bool): If True, generates plots for
            individual scans.

        Returns:

            fitting_triplets (list): A list containing fitting parameters for
            transitions.

        """
        self.freq_files = list(
            filter(lambda string: 'freq' in string, self.file_names))

        self.get_sim_freqs()

        list_of_545 = []
        for freq_file in self.freq_files:
            chunks = freq_file.split('_')
            freq = float(chunks[6].replace('p', '.'))

            diff_from_545 = np.abs(self.sim_freqs - freq)
            diff_masked = np.ma.array(diff_from_545,
                                      mask=np.isnan(diff_from_545))
            idx = np.unravel_index(np.argmin(diff_masked, axis=None),
                                   diff_masked.shape)
            # idx = diff_from_545.argmin()
            if idx == (5, 2):
                list_of_545.append(freq_file)

        if self.use_old_data:
            run_length = 38
        else:
            run_length = 8
        self.good_freq_runs = np.zeros((len(list_of_545), run_length))
        # sort freq_files into chunks associated with 545 data
        self.chunked_freq_files = np.empty((len(list_of_545), run_length),
                                           dtype=object)
        self.chunked_freq_fits = np.zeros((len(list_of_545), run_length))
        self.chunked_freq_fit_errors = np.zeros((len(list_of_545), run_length))

        self.chunked_datetimes = np.zeros((len(list_of_545), run_length),
                                          dtype=int)
        self.chunked_datetimes = self.chunked_datetimes.astype(str)

        for idx, file_545 in enumerate(list_of_545):
            self.chunked_freq_files[idx, 0] = file_545

            file_545_date = file_545.split('_')[7]
            file_545_time = file_545.split('_')[8][:-4]
            file_545_datetime = file_545_date + '_' + file_545_time
            dummy_idx = 0
            for freq_file in self.freq_files:
                freq_file_date = freq_file.split('_')[7]
                freq_file_time = freq_file.split('_')[8][:-4]
                freq_file_datetime = freq_file_date + '_' + freq_file_time
                if (freq_file_datetime == file_545_datetime
                        and freq_file != file_545):
                    self.chunked_freq_files[idx, dummy_idx + 1] = freq_file
                    dummy_idx += 1

        # Define the sin-squared transition profile for fitting
        def sin_squared(f, f0, A, Omega, t):
            return A * (Omega**2 / (Omega**2 + (f - f0)**2)) * (np.sin(
                np.sqrt(Omega**2 + (f - f0)**2) * t / 2))**2

        if create_plots:
            n_cols = run_length
            n_rows = len(list_of_545)

            # Create a figure with subplots
            fig_comb, axs_comb = plt.subplots(n_rows,
                                              n_cols,
                                              figsize=(n_cols * 5, n_rows * 5))
            axs_comb = axs_comb.flatten(
            )  # Flatten in case of single row/column

        for idx, cal_chunk in enumerate(tqdm(self.chunked_freq_files)):
            file_545_date = cal_chunk[0].split('_')[7]
            file_545_time = cal_chunk[0].split('_')[8][:-4]
            file_545_datetime = file_545_date + '_' + file_545_time

            for idx_ind, cal_run in enumerate(cal_chunk):
                total_idx = run_length * idx + idx_ind
                if cal_run is not None:
                    # Load the data from the text file
                    data = np.genfromtxt(self.folder_path_raw + '/' + cal_run,
                                         delimiter=',',
                                         skip_footer=1)

                    # Extract the first and second entries of each row
                    freqs = data[:-1, 0]
                    PD = data[:-1, 1]
                    freqs_popped = data[:-1, 0]
                    PD_popped = data[:-1, 1]

                    # Define the initial guesses
                    # best way to find centre freq when pi-time is uncertain
                    f0_guess = np.sum(np.multiply(freqs, PD)) / np.sum(PD)
                    # need to check if this will be a good f0_guess
                    # it won't be if we are too frequency uncertain
                    f_range = freqs[-1] - freqs[0]
                    top_30_percent = freqs[-1] - f_range / 3
                    bot_30_percent = freqs[0] + f_range / 3
                    if f0_guess > top_30_percent or f0_guess < bot_30_percent:
                        f0_guess = freqs[np.argmax(PD)]

                    # Omega_guess = 0.1 * ((freqs[-1] - freqs[0]))  # in MHz
                    f_variance = (freqs - f0_guess)**2
                    Omega_guess = np.sqrt(
                        np.sum(np.multiply(f_variance, PD)) / np.sum(PD) / 4)

                    A_guess = max(PD)
                    t_guess = np.pi / Omega_guess
                    p0 = np.array([f0_guess, A_guess, Omega_guess, t_guess])
                    x_new = np.linspace(freqs.min(), freqs.max(), 1000)

                    # Find the errors for the plots using Wilson interval
                    num_SD = 1  # number of standard deviations, 1=>68.28%

                    lower_error, upper_error = self.find_errors(
                        num_SD, PD, self.exp_number)
                    asym_error_bar = [PD - lower_error, upper_error - PD]

                    # Fit the data to the Rabi fringe function
                    try:
                        popt, pcov = sc.optimize.curve_fit(sin_squared,
                                                           freqs,
                                                           PD,
                                                           p0=p0,
                                                           maxfev=10000)

                        for i in [1, 2, 3]:
                            pop_indices = [
                                idx for idx, freq in enumerate(freqs)
                                if np.absolute(
                                    sin_squared(freq, *popt) - PD[idx]) > 0.3
                            ]
                            freqs_popped = np.delete(freqs, pop_indices)
                            PD_popped = np.delete(PD, pop_indices)

                            if len(freqs_popped) >= 4:
                                popt, pcov = sc.optimize.curve_fit(
                                    sin_squared,
                                    freqs_popped,
                                    PD_popped,
                                    p0=p0,
                                    maxfev=10000)
                            else:
                                pass

                        # print(popt, pcov)
                        diff = np.abs(self.sim_freqs - popt[0])
                        diff_masked = np.ma.array(diff,
                                                  mask=np.isnan(diff_from_545))
                        idx_trans = np.unravel_index(
                            np.argmin(diff_masked, axis=None),
                            diff_masked.shape)

                        # calculating pi time and freqs with uncertainties
                        t_pi = 1 / (2 * np.absolute(popt[2]))
                        s_pi = t_pi * np.absolute(
                            np.sqrt(pcov[2][2]) / popt[2])

                        s_x0 = round(1e3 * np.sqrt(pcov[0][0]), 3)

                        # conditions for throwing out frequency sweeps
                        res_freq_too_off_centre = (
                            popt[0] < freqs[0] + f_range / 15
                            or popt[0] > freqs[-1] - f_range / 15)

                        fit_bad = (s_x0 > 1.0 or s_pi > 2500 or popt[1] < 0.4
                                   or popt[3] < 0 or len(pop_indices) > 4)

                        peak_points_too_low = np.count_nonzero(PD > 0.4) < 3

                        def too_many_zero_points(lst):
                            count_zeros = 0
                            for i in range(len(lst) - 1):
                                if lst[i] == 0 and lst[i + 1] == 0:
                                    count_zeros += 1
                                    if count_zeros > 2:  # for 3 0's in a row
                                        return True
                                else:
                                    count_zeros = 0
                            return False

                        upload_error = too_many_zero_points(PD)
                        # if np.abs(popt[0] - 545.31) < 0.1:
                        #     print(popt, s_x0, s_pi, pcov)

                        if (fit_bad or res_freq_too_off_centre
                                or peak_points_too_low or upload_error):
                            date_ind = cal_run.split('_')[7]
                            time_ind = cal_run.split('_')[8][:-4]
                            # print(popt[0], s_x0, s_pi, popt[1], popt[3],
                            #       date_ind + '_' + time_ind)

                            if create_plots:
                                if idx_ind == 0:
                                    axs_comb[total_idx].text(
                                        np.mean(freqs) -
                                        (freqs[-1] - freqs[0]) / 3,
                                        0.9,
                                        f'{file_545_datetime}',
                                        fontsize=14,
                                        ha='left',
                                        va='center')
                                else:
                                    pass

                                axs_comb[total_idx].errorbar(
                                    freqs,
                                    PD,
                                    yerr=asym_error_bar,
                                    fmt='o',
                                    color='r',
                                    label='data')
                                axs_comb[total_idx].text(np.mean(freqs),
                                                         0.5,
                                                         'Poor Data',
                                                         fontsize=16,
                                                         color='r',
                                                         ha='center',
                                                         va='center')
                                axs_comb[total_idx].set_ylim(0, 1)
                                s12_state = int(idx_trans[1] - 2)
                                d52_state = self.F_values_D52[23 -
                                                              idx_trans[0]]
                                axs_comb[total_idx].set_title(
                                    r'$S_{1/2}$ to $D_{5/2}$: ($m_S,F_D,m_D$)='
                                    + f'$({s12_state},' +
                                    f'{d52_state[0]},{d52_state[1]})$')
                                axs_comb[total_idx].legend(
                                    title=rf"$f$={popt[0]:.4f} MHz," +
                                    rf"$\sigma_f$={s_x0} kHz" + "\n" +
                                    rf"$\tau_\pi$={t_pi:.3f}$\mu s$," +
                                    rf" $\sigma_\pi$={s_pi:.3f}$\mu s$",
                                    loc='lower left',
                                    fontsize=6)
                                axs_comb[total_idx].plot(x_new,
                                                         sin_squared(
                                                             x_new, *popt),
                                                         'r-',
                                                         label='fit')
                            else:
                                pass

                        else:
                            self.good_freq_runs[idx, idx_ind] = 1
                            self.chunked_freq_fits[idx, idx_ind] = popt[0]
                            self.chunked_freq_fit_errors[idx,
                                                         idx_ind] = (np.sqrt(
                                                             pcov[0][0]))

                            date_ind = cal_run.split('_')[7]
                            time_ind = cal_run.split('_')[8][:-4]
                            self.chunked_datetimes[
                                idx, idx_ind] = date_ind + '_' + time_ind

                            if create_plots:
                                if idx_ind == 0:
                                    axs_comb[total_idx].text(
                                        np.mean(freqs) -
                                        (freqs[-1] - freqs[0]) / 3,
                                        0.9,
                                        f'{file_545_datetime}',
                                        fontsize=14,
                                        ha='center',
                                        va='center')
                                axs_comb[total_idx].set_ylim(0, 1)
                                lower_error_popped, upper_error_popped = (
                                    self.find_errors(num_SD, PD_popped,
                                                     self.exp_number))
                                asym_error_bar_popped = [
                                    PD_popped - lower_error_popped,
                                    upper_error_popped - PD_popped
                                ]
                                axs_comb[total_idx].errorbar(freqs,
                                                             PD,
                                                             color='r',
                                                             fmt='o',
                                                             label='outliers')
                                axs_comb[total_idx].errorbar(
                                    freqs_popped,
                                    PD_popped,
                                    yerr=asym_error_bar_popped,
                                    fmt='o',
                                    label='data')
                                axs_comb[total_idx].plot(x_new,
                                                         sin_squared(
                                                             x_new, *popt),
                                                         'r-',
                                                         label='fit')

                                s12_state = int(idx_trans[1] - 2)
                                d52_state = self.F_values_D52[23 -
                                                              idx_trans[0]]
                                axs_comb[total_idx].set_title(
                                    r'$S_{1/2}$ to $D_{5/2}$: ($m_S,F_D,m_D$)='
                                    + f'$({s12_state},' +
                                    f'{d52_state[0]},{d52_state[1]})$')
                                # Simplify x-axis labels
                                axs_comb[total_idx].xaxis.set_major_formatter(
                                    plt.FuncFormatter(lambda x, _: f'{x:.3f}'))

                                axs_comb[total_idx].legend(
                                    title=rf"$f$={popt[0]:.4f} MHz" +
                                    rf" $\sigma_f$={s_x0} kHz" + "\n" +
                                    rf"$\tau_\pi$={t_pi:.3f}$\mu s$," +
                                    rf" $\sigma_\pi$={s_pi:.3f}$\mu s$",
                                    loc='lower left',
                                    fontsize=6)
                                axs_comb[total_idx].grid()

                            if plot_individual_scans:
                                if idx_ind == 0 or idx_ind == 1:
                                    pass
                                else:
                                    fig, ax = plt.subplots(1, 1)
                                    ax.set_ylim(0, 1)
                                    ax.errorbar(freqs,
                                                PD,
                                                color='r',
                                                fmt='o',
                                                label='outliers')
                                    ax.errorbar(freqs_popped,
                                                PD_popped,
                                                yerr=asym_error_bar_popped,
                                                fmt='o',
                                                label='data')
                                    ax.plot(x_new,
                                            sin_squared(x_new, *popt),
                                            'r-',
                                            label='fit')

                                    s12_state = int(idx_trans[1] - 2)
                                    d52_state = self.F_values_D52[23 -
                                                                  idx_trans[0]]
                                    ax.set_title(
                                        r'$S_{1/2}$ to $D_{5/2}$: ' +
                                        r'($m_S,F_D,m_D$)=' +
                                        f'$({s12_state},' +
                                        f'{d52_state[0]},{d52_state[1]})$')
                                    # Simplify x-axis labels
                                    ax.xaxis.set_major_formatter(
                                        plt.FuncFormatter(
                                            lambda x, _: f'{x:.3f}'))

                                    ax.legend(
                                        title=rf"$f$={popt[0]:.4f} MHz" +
                                        rf" $\sigma_f$={s_x0} kHz" + "\n" +
                                        rf"$\tau_\pi$={t_pi:.3f}$\mu s$," +
                                        rf" $\sigma_\pi$={s_pi:.3f}$\mu s$",
                                        loc='lower left',
                                        fontsize=6)
                                    ax.grid()
                                    plt.show()

                    except RuntimeError:
                        pass
                else:
                    if create_plots:
                        axs_comb[total_idx].text(0.5,
                                                 0.5,
                                                 'No Data',
                                                 fontsize=16,
                                                 ha='center',
                                                 va='center')
        # print(self.good_freq_runs)
        if create_plots:
            fig_comb.tight_layout()
            fig_comb.savefig(
                'ramseys_calibrations/all_calibrations_freqs_plot.pdf')
            # plt.close(fig_comb)

        fitting_triplets = []
        fitting_triplets_uncertainty = []
        for idx, chunk in enumerate(self.chunked_freq_fits):
            for it in range(2, run_length, 1):
                if chunk[0] != 0 and chunk[1] != 0 and chunk[it] != 0:
                    # Convert timestamps to datetimes
                    chunk_time = datetime.strptime(
                        self.chunked_datetimes[idx, 0].split('_')[0], "%Y%m%d")
                    cutoff_date = datetime.strptime(self.data_cutoff_date,
                                                    "%Y-%m-%d")
                    # end_date = datetime.strptime(self.data_end_date,
                    #                              "%Y-%m-%d")

                    if chunk_time > cutoff_date:
                        fitting_triplets.append([
                            chunk[0], chunk[1], chunk[it],
                            float(self.chunked_datetimes[idx,
                                                         it].split('_')[0]) +
                            1e-4 * float(
                                self.chunked_datetimes[idx, it].split('_')[1])
                        ])
                        fitting_triplets_uncertainty.append([
                            self.chunked_freq_fit_errors[idx][0],
                            self.chunked_freq_fit_errors[idx][1],
                            self.chunked_freq_fit_errors[idx][it]
                        ])
                        # add in triplets for the reference 545 and 623 trans
                        if it == 0:
                            fitting_triplets.append([
                                chunk[0], chunk[1], chunk[0],
                                float(self.chunked_datetimes[idx,
                                                             it].split('_')[0])
                                + 1e-4 *
                                float(self.chunked_datetimes[idx,
                                                             it].split('_')[1])
                            ])
                            fitting_triplets_uncertainty.append([
                                self.chunked_freq_fit_errors[idx][0],
                                self.chunked_freq_fit_errors[idx][1],
                                self.chunked_freq_fit_errors[idx][0]
                            ])

                            fitting_triplets.append([
                                chunk[0], chunk[1], chunk[1],
                                self.chunked_datetimes[idx, it]
                            ])
                            fitting_triplets_uncertainty.append([
                                self.chunked_freq_fit_errors[idx][0],
                                self.chunked_freq_fit_errors[idx][1],
                                self.chunked_freq_fit_errors[idx][1]
                            ])

        self.an1_an2_data = fitting_triplets
        self.an1_an2_data_uncertainty = fitting_triplets_uncertainty
        np.savetxt('ramseys_calibrations/triplets_for_an1_an2.txt',
                   fitting_triplets,
                   delimiter=',')
        # print(fitting_triplets)

        print('\n Done fitting frequency scans. \n')

        return fitting_triplets

    def an1_an2_finder(self, create_freq_plots=False):
        """Finds the an1 and an2 values for transitions based on fitting
        results and generates plots if specified.

        Parameters:

            create_freq_plots (bool): If True, generates frequency plots for
            transitions.

        Returns:

            an1_table (np.ndarray): A table of calculated an1 values.

            an2_table (np.ndarray): A table of calculated an2 values.

            an1_theory_table (np.ndarray): A theoretical table of an1 values.

            an2_theory_table (np.ndarray): A theoretical table of an2 values.

        """

        an1_table = np.zeros((24, 5))
        an2_table = np.zeros((24, 5))
        an1_theory_table = np.zeros((24, 5))
        an1_theory_table = np.zeros((24, 5))
        sensitivities = self.generate_transition_sensitivities()

        # get the slopes an1 data from B field sensitivities
        for row_idx, row in enumerate(sensitivities):
            for col_idx, sensitivity in enumerate(row):
                if not np.isnan(sensitivity):
                    an1_theory_table[
                        row_idx,
                        col_idx] = (sensitivity - sensitivities[5, 2]) / (
                            sensitivities[22, 1] - sensitivities[5, 2])

        an2_theory_table = np.zeros((24, 5))
        data_points_table = np.zeros((24, 5))

        if hasattr(self, 'freq_files'):
            print('\n Done fitting frequency scans already. \n')
            pass
        else:
            print('\n Need to run frequency fitting. \n')
            self.fit_freq_Rabi_fringe(create_plots=False)

        def linear_function(x, slope, offset):
            return slope * x + offset

        def offset_function(a, offset):
            return a + offset

        # Determine the layout of the subplots
        n_cols = 7
        n_rows = 6
        # Create a figure with subplots
        fig_comb, axs_comb = plt.subplots(n_rows,
                                          n_cols,
                                          figsize=(n_cols * 5, n_rows * 5))
        axs_comb = axs_comb.flatten()
        # print(self.sim_freqs)

        plotting_counter = 0
        mean_delta_y = []
        delta_y_range = []
        for (row, col), transition in np.ndenumerate(self.sim_freqs):
            delta_y_list = []
            delta_x_list = []
            trans_list = []
            if transition != np.nan:
                for it, triplet in enumerate(self.an1_an2_data):
                    run_trans, run_pitimes = (
                        self.run_calibrated_frequency_generation(
                            insensB=triplet[0], sensB=triplet[1]))
                    trans = run_trans[row, col]
                    if np.absolute(triplet[2] - trans) < 0.020:
                        # print(triplet)
                        delta_x_list.append(triplet[1] - triplet[0])
                        delta_y_list.append(triplet[2] - triplet[0])
                        trans_list.append(triplet[2])
                    else:
                        pass
                delta_x_list = np.array(delta_x_list)
                delta_y_list = np.array(delta_y_list)
                data_points_table[row, col] = len(delta_x_list)

                if len(delta_x_list) >= 1 and len(delta_x_list) < 2:
                    slope_from_theory = (self.sim_sensitivities[row, col] /
                                         self.sim_sensitivities[22, 1])
                    delta_x_theory = slope_from_theory * delta_x_list
                    popt_theory, pcov_theory = sc.optimize.curve_fit(
                        offset_function, delta_x_theory, delta_y_list)
                    axs_comb[plotting_counter].plot(delta_x_list,
                                                    delta_y_list,
                                                    'o',
                                                    color='r',
                                                    label='data')
                    x_values = np.linspace(min(delta_x_list),
                                           max(delta_x_list), 100)
                    axs_comb[plotting_counter].plot(
                        x_values,
                        offset_function(slope_from_theory * x_values,
                                        popt_theory[0]),
                        label='theory')

                    axs_comb[plotting_counter].set_ylabel(
                        r'$f_n$ - $F=2,m=0$ (MHz)')
                    axs_comb[plotting_counter].set_xlabel(
                        r'$\Delta f_\pm$ (MHz)')
                    s12_state = int(col - 2)
                    d52_state = self.F_values_D52[23 - row]
                    axs_comb[plotting_counter].xaxis.set_major_formatter(
                        plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
                    axs_comb[plotting_counter].yaxis.set_major_formatter(
                        plt.FuncFormatter(lambda y, _: f'{y:.3f}'))
                    # get B field sensitivity theory predictions for slopes
                    an2_theory_table[row, col] = popt_theory[0]

                    axs_comb[plotting_counter].set_title(
                        r'$S_{1/2}$ to $D_{5/2}$: ($m_S,F_D,m_D$)=' +
                        f'$({s12_state},' + f'{d52_state[0]},{d52_state[1]})$')
                    axs_comb[plotting_counter].legend(
                        title=rf"$f$={trans_list[-1]:.4f} MHz",
                        loc='lower left',
                        fontsize=6)

                    axs_comb[plotting_counter].grid()
                    mean_delta_y.append(np.mean(delta_y_list))
                    delta_y_range.append(max(delta_y_list) - min(delta_y_list))

                elif len(delta_x_list) >= 2:
                    popt, pcov = sc.optimize.curve_fit(linear_function,
                                                       delta_x_list,
                                                       delta_y_list)

                    for i in [1, 2, 3]:
                        pop_indices = [
                            idx for idx, delta in enumerate(delta_x_list)
                            if np.absolute(
                                linear_function(delta, *popt) -
                                delta_y_list[idx]) > 0.004
                        ]
                        delta_x_list_popped = np.delete(
                            delta_x_list, pop_indices)
                        delta_y_list_popped = np.delete(
                            delta_y_list, pop_indices)

                        if len(delta_x_list_popped) >= 2:
                            popt, pcov = sc.optimize.curve_fit(
                                linear_function, delta_x_list_popped,
                                delta_y_list_popped)
                            slope_from_theory = (
                                self.sim_sensitivities[row, col] /
                                self.sim_sensitivities[22, 1])
                            delta_x_theory = slope_from_theory * delta_x_list
                            popt_theory, pcov_theory = sc.optimize.curve_fit(
                                offset_function, delta_x_theory, delta_y_list)
                        else:
                            pass

                    axs_comb[plotting_counter].plot(delta_x_list,
                                                    delta_y_list,
                                                    'o',
                                                    color='r',
                                                    label='outliers')
                    axs_comb[plotting_counter].plot(delta_x_list_popped,
                                                    delta_y_list_popped,
                                                    'o',
                                                    label='data')

                    x_values = np.linspace(min(delta_x_list),
                                           max(delta_x_list), 100)
                    axs_comb[plotting_counter].plot(x_values,
                                                    linear_function(
                                                        x_values, popt[0],
                                                        popt[1]),
                                                    linestyle='--',
                                                    label='fit')
                    axs_comb[plotting_counter].plot(
                        x_values,
                        offset_function(slope_from_theory * x_values,
                                        popt_theory[0]),
                        label='theory')
                    axs_comb[plotting_counter].set_ylabel(
                        r'$f_n$ - $F=2,m=0$ (MHz)')
                    axs_comb[plotting_counter].set_xlabel(
                        r'$\Delta f_\pm$ (MHz)')
                    s12_state = int(col - 2)
                    d52_state = self.F_values_D52[23 - row]
                    axs_comb[plotting_counter].xaxis.set_major_formatter(
                        plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
                    axs_comb[plotting_counter].yaxis.set_major_formatter(
                        plt.FuncFormatter(lambda y, _: f'{y:.3f}'))
                    # get B field sensitivity theory predictions for slopes
                    an2_theory_table[row, col] = popt_theory[0]

                    axs_comb[plotting_counter].set_title(
                        r'$S_{1/2}$ to $D_{5/2}$: ($m_S,F_D,m_D$)=' +
                        f'$({s12_state},' + f'{d52_state[0]},{d52_state[1]})$')
                    axs_comb[plotting_counter].legend(
                        title=rf"$f$={trans_list[-1]:.4f} MHz",
                        loc='lower left',
                        fontsize=6)

                    axs_comb[plotting_counter].grid()
                    mean_delta_y.append(np.mean(delta_y_list))
                    delta_y_range.append(max(delta_y_list) - min(delta_y_list))

                    an1_table[row, col] = popt[0]
                    an2_table[row, col] = popt[1]

                    plotting_counter += 1

            else:
                pass

        largest_plotting_range = max(delta_y_range)
        for it in range(plotting_counter):
            axs_comb[it].set_ylim(mean_delta_y[it] - largest_plotting_range,
                                  mean_delta_y[it] + largest_plotting_range)

        fig_comb.tight_layout()
        fig_comb.savefig('ramseys_calibrations/an1_an2_freq_linear_fits.pdf')

        print('\n an_1 Table \n', np.round(an1_table, 3))
        print('\n an_1 Theory Table \n', np.round(an1_theory_table, 3))
        print('\n an_2 Table \n', np.round(an2_table, 3))
        print('\n an_2 Theory Table \n', np.round(an2_theory_table, 3))

        # to save in tab separated format (as needed for an1,
        # an2 tables for IonControl)
        # np.savetxt(filename, matrix, delimiter='\t', fmt='%f')

        np.savetxt('ramseys_calibrations/an1_freq_table.txt',
                   an1_table,
                   delimiter=',')
        np.savetxt('ramseys_calibrations/an2_freq_table.txt',
                   an2_table,
                   delimiter=',')
        np.savetxt('ramseys_calibrations/an1_theory_freq_table.txt',
                   an1_theory_table,
                   delimiter=',')
        np.savetxt('ramseys_calibrations/an2_theory_freq_table.txt',
                   an2_theory_table,
                   delimiter=',')
        self.an1_exp_table = an1_table
        self.an2_exp_table = an2_table
        self.an1_theory_table = an1_theory_table
        self.an2_theory_table = an2_theory_table

        print('\n Done frequency an1, an2 finder. \n')

        return an1_table, an2_table, an1_theory_table, an2_theory_table

    def Ramsey_an2_finder(
            self,
            Ramsey_wait_time=100,  # us
            shots=250,
            data_cutoff_date=None,
            plot_individual_linears=False,
            compare_old_calibration=True):
        """Finds Ramsey an2 values based on calibration data and performs
        comparisons if specified. With Ramsey calibrations, no an1 values are
        needed, we use the theoretical magnetic field sensitivities directly to
        get the slopes.

        Parameters:

            Ramsey_wait_time (int): The wait time in microseconds for the
            Ramsey experiment.

            shots (int): The number of shots in the experiment.

            data_cutoff_date (str): Optional date string to filter data.

            plot_individual_linears (bool): If True, generates individual plots
            for linear fits.

            compare_old_calibration (bool): If True, compares with old
            calibration data.

        Returns:

            an1_theory_table (np.ndarray): A table of theoretical an1 values.

            an2_theory_table (np.ndarray): A table of theoretical an2 values.

        """

        # get the files from Ramsey calibrations
        folder_path_root = ('../data/ramseys_calibrations_data/')
        filename_root = 'ramsey_calibration_triplet_'
        filenames = [
            f for f in os.listdir(folder_path_root)
            if f.startswith(filename_root)
        ]
        filenames = sorted(
            filenames,
            key=lambda t: os.stat(os.path.join(folder_path_root, t)).st_mtime)
        if data_cutoff_date:
            cutoff_date = datetime.strptime(data_cutoff_date, "%Y%m%d_%H%M")

            filenames = [
                f for f in filenames if datetime.fromtimestamp(
                    os.path.getmtime(os.path.join(folder_path_root, f))) >
                cutoff_date
            ]
            filenames = sorted(filenames,
                               key=lambda t: os.stat(
                                   os.path.join(folder_path_root, t)).st_mtime)

        print('Number of Ramsey calibration files included:', len(filenames))

        # the Ramsey fit error only depends on wait time and number of shots
        Ramsey_fit_error = np.sqrt(
            (2 * 0.5 * 0.5) / shots) / (2 * np.pi * Ramsey_wait_time)

        self.get_sim_freqs()

        Ramsey_freq_triplets = np.empty((3, len(filenames)))
        Ramsey_state_triplets = np.empty((3, 3, len(filenames)))

        if hasattr(self, 'an1_an2_data'):
            print('\n Done old calibration fits already. \n')
            pass
        else:
            print('\n Need to run old calibration finder. \n')

            self.data_cutoff_date = '2024-05-05'
            self.data_end_date = '2024-09-15'
            self.fit_freq_Rabi_fringe(create_plots=False)

        for idx, filename in enumerate(filenames):
            with open('../data/ramseys_calibrations_data/' + filename,
                      'r') as file:
                # Read the first line (assuming it's a list of floats)
                first_line = file.readline()
                fl_vals = re.findall(r"[-+]?\d*\.\d+|\d+", first_line)
                Ramsey_freq_triplets[:, idx] = [
                    fl_vals[0], fl_vals[1], fl_vals[2]
                ]

                # Read the second line (assuming it's a nested list of ints)
                second_line = file.readline()
                sl_values = re.findall(r"[-+]?\d+", second_line)
                Ramsey_state_triplets[0, :, idx] = [
                    sl_values[0], sl_values[1], sl_values[2]
                ]
                Ramsey_state_triplets[1, :, idx] = [
                    sl_values[3], sl_values[4], sl_values[5]
                ]
                Ramsey_state_triplets[2, :, idx] = [
                    sl_values[6], sl_values[7], sl_values[8]
                ]

        Ramsey_freq_triplets = np.transpose(Ramsey_freq_triplets)

        an1_theory_table = np.zeros((24, 5))
        sensitivities = self.generate_transition_sensitivities()

        # get the slopes an1 data from B field sensitivities
        for row_idx, row in enumerate(sensitivities):
            for col_idx, sensitivity in enumerate(row):
                if not np.isnan(sensitivity):
                    an1_theory_table[
                        row_idx,
                        col_idx] = (sensitivity - sensitivities[5, 2]) / (
                            sensitivities[22, 1] - sensitivities[5, 2])

        an2_theory_table = np.zeros((24, 5))
        data_points_table = np.zeros((24, 5))
        data_points_table_old = np.zeros((24, 5))

        def linear_function(x, slope, offset):
            return slope * x + offset

        def offset_function(a, offset):
            return a + offset

        # Determine the layout of the subplots
        n_cols = 7
        n_rows = 9
        # Create a figure with subplots
        fig_comb, axs_comb = plt.subplots(n_rows,
                                          n_cols,
                                          figsize=(n_cols * 5, n_rows * 5))
        axs_comb = axs_comb.flatten()

        plotting_counter = 0
        mean_delta_y = []
        delta_y_range = []
        old_cal_discrepancies = []
        new_cal_discrepancies = []
        matched_transition_list = []  # list of transitions we calculate disc
        new_cal_disc_averaged = []  # discrepancies for each transition, avg
        for (row, col), transition in np.ndenumerate(self.sim_freqs):
            delta_y_list = []
            delta_x_list = []
            delta_y_list_uncertainty = []
            delta_x_list_uncertainty = []

            delta_y_list_old = []
            delta_x_list_old = []
            delta_y_list_old_uncertainty = []
            delta_x_list_old_uncertainty = []

            trans_list = []
            if transition != np.nan:
                for it, triplet in enumerate(self.an1_an2_data):
                    run_trans, run_pitimes = (
                        self.run_calibrated_frequency_generation(
                            insensB=triplet[0], sensB=triplet[1]))
                    trans = run_trans[row, col]
                    if np.absolute(triplet[2] - trans) < 0.020:
                        delta_x_list_old.append(triplet[1] - triplet[0])
                        delta_y_list_old.append(triplet[2] - triplet[0])
                        delta_x_list_old_uncertainty.append(
                            np.sqrt(self.an1_an2_data_uncertainty[it][1]**2 +
                                    self.an1_an2_data_uncertainty[it][0]**2))
                        delta_y_list_old_uncertainty.append(
                            np.sqrt(self.an1_an2_data_uncertainty[it][2]**2 +
                                    self.an1_an2_data_uncertainty[it][0]**2))
                        trans_list.append(triplet[2])
                    else:
                        pass

                delta_x_list_old = np.array(delta_x_list_old)
                delta_y_list_old = np.array(delta_y_list_old)
                delta_x_list_old_uncertainty = np.array(
                    delta_x_list_old_uncertainty)
                delta_y_list_old_uncertainty = np.array(
                    delta_y_list_old_uncertainty)
                data_points_table_old[row, col] = len(delta_x_list)

                if len(delta_x_list_old) >= 2:
                    popt, pcov = sc.optimize.curve_fit(linear_function,
                                                       delta_x_list_old,
                                                       delta_y_list_old)
                else:
                    pass

                for it, triplet in enumerate(Ramsey_freq_triplets):
                    # print(triplet)

                    run_trans, run_pitimes = (
                        self.run_calibrated_frequency_generation(
                            insensB=triplet[0], sensB=triplet[1]))
                    trans = run_trans[row, col]
                    if np.absolute(triplet[2] - trans) < 0.020:
                        # if row == 11 and col == 3:
                        #     print(triplet)
                        #     print(trans)
                        #     print(row, col)
                        #     print(triplet[2] - trans)

                        delta_x_list.append(triplet[1] - triplet[0])
                        delta_y_list.append(triplet[2] - triplet[0])
                        delta_x_list_uncertainty.append(
                            np.sqrt(2 * Ramsey_fit_error**2))
                        delta_y_list_uncertainty.append(
                            np.sqrt(2 * Ramsey_fit_error**2))
                        trans_list.append(triplet[2])
                    else:
                        pass
                delta_x_list = np.array(delta_x_list)
                delta_y_list = np.array(delta_y_list)
                data_points_table[row, col] = len(delta_x_list)

                if len(delta_x_list) >= 1:
                    slope_from_theory = an1_theory_table[row, col]

                    delta_x_theory = slope_from_theory * delta_x_list
                    popt_theory, pcov_theory = sc.optimize.curve_fit(
                        offset_function, delta_x_theory, delta_y_list)
                    an2_theory_table[row, col] = popt_theory[0]
                    axs_comb[plotting_counter].errorbar(
                        delta_x_list,
                        delta_y_list,
                        fmt='o',
                        xerr=delta_x_list_uncertainty,
                        yerr=delta_y_list_uncertainty,
                        label='Ramsey data')
                    if compare_old_calibration:
                        axs_comb[plotting_counter].errorbar(
                            delta_x_list_old,
                            delta_y_list_old,
                            fmt='o',
                            xerr=delta_x_list_old_uncertainty,
                            yerr=delta_y_list_old_uncertainty,
                            color='gray',
                            label='Freq. scan data',
                            alpha=0.5)
                        x_values = np.linspace(
                            min(
                                np.concatenate(
                                    (delta_x_list, delta_x_list_old), axis=0)),
                            max(
                                np.concatenate(
                                    (delta_x_list, delta_x_list_old), axis=0)),
                            100)
                    else:
                        x_values = np.linspace(min(delta_x_list),
                                               max(delta_x_list), 100)

                    axs_comb[plotting_counter].plot(
                        x_values,
                        offset_function(slope_from_theory * x_values,
                                        popt_theory[0]),
                        label=r'Ramsey fits',
                        color='black',
                        linestyle='--')
                    new_cal_disc_unaveraged = []
                    for idx, delta in enumerate(delta_y_list):
                        new_cal_disc_unaveraged.append(
                            delta - (slope_from_theory * delta_x_list[idx] +
                                     popt_theory[0]))

                        new_cal_discrepancies.append(
                            delta - (slope_from_theory * delta_x_list[idx] +
                                     popt_theory[0]))
                    new_cal_disc_averaged.append(
                        np.mean(np.absolute(new_cal_disc_unaveraged)))
                    matched_transition_list.append((row, col))
                    if len(delta_x_list_old) >= 2:
                        if compare_old_calibration:
                            axs_comb[plotting_counter].plot(
                                x_values,
                                linear_function(x_values, popt[0], popt[1]),
                                linestyle='--',
                                color='gray',
                                label='Freq. scan fits')
                        for idx, delta in enumerate(delta_y_list_old):
                            old_cal_discrepancies.append(
                                delta - linear_function(
                                    delta_x_list_old[idx], popt[0], popt[1]))
                    else:
                        pass

                    axs_comb[plotting_counter].set_ylabel(
                        r'$f_n$ - $F=2,m=0$ (MHz)')
                    axs_comb[plotting_counter].set_xlabel(
                        r'$\Delta f_\pm$ (MHz)')
                    s12_state = int(col - 2)
                    d52_state = self.F_values_D52[23 - row]
                    # if row == 11 and col == 3:
                    #     print(s12_state, d52_state)
                    axs_comb[plotting_counter].xaxis.set_major_formatter(
                        plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
                    axs_comb[plotting_counter].yaxis.set_major_formatter(
                        plt.FuncFormatter(lambda y, _: f'{y:.3f}'))
                    # get B field sensitivity theory predictions for slopes
                    an2_theory_table[row, col] = popt_theory[0]

                    axs_comb[plotting_counter].set_title(
                        r'$S_{1/2}$ to $D_{5/2}$: ($m_S,F_D,m_D$)=' +
                        f'$({s12_state},' + f'{d52_state[0]},{d52_state[1]})$')
                    axs_comb[plotting_counter].legend(
                        title=rf"$f$={trans_list[-1]:.8f} MHz",
                        loc='lower left',
                        fontsize=6)

                    axs_comb[plotting_counter].grid()

                    mean_delta_y.append(
                        np.mean(
                            np.concatenate((delta_y_list, delta_y_list_old),
                                           axis=0)))
                    delta_y_range.append(
                        max(
                            np.concatenate(
                                (delta_y_list, delta_y_list_old), axis=0)) -
                        min(
                            np.concatenate(
                                (delta_y_list, delta_y_list_old), axis=0)))

                    if plot_individual_linears:
                        fig, ax = plt.subplots(1, 1)

                        if compare_old_calibration:
                            ax.plot(x_values,
                                    linear_function(x_values, popt[0],
                                                    popt[1]),
                                    linestyle='--',
                                    color='gray',
                                    label='Freq. scan fits')
                        ax.plot(x_values,
                                offset_function(slope_from_theory * x_values,
                                                popt_theory[0]),
                                label=r'Ramsey fits',
                                color='black',
                                linestyle='--')
                        ax.errorbar(delta_x_list,
                                    delta_y_list,
                                    fmt='o',
                                    xerr=delta_x_list_uncertainty,
                                    yerr=delta_y_list_uncertainty,
                                    label='Ramsey data')
                        if compare_old_calibration:
                            ax.errorbar(delta_x_list_old,
                                        delta_y_list_old,
                                        fmt='o',
                                        xerr=delta_x_list_old_uncertainty,
                                        yerr=delta_y_list_old_uncertainty,
                                        color='gray',
                                        label='Freq. scan data',
                                        alpha=0.5)
                        ax.set_ylabel(r'$f_t - f_{int}$ / MHz')
                        ax.set_xlabel(r'$f_{sens} - f_{int}$ / MHz')
                        s12_state = int(col - 2)
                        d52_state = self.F_values_D52[23 - row]
                        ax.xaxis.set_major_formatter(
                            plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
                        ax.yaxis.set_major_formatter(
                            plt.FuncFormatter(lambda y, _: f'{y:.3f}'))
                        ax.set_title(
                            r'$S_{1/2}$ to $D_{5/2}$: ($m_S,F_D,m_D$)=' +
                            f'$({s12_state},' +
                            f'{d52_state[0]},{d52_state[1]})$')
                        ax.legend(fontsize=8)
                        ax.grid()
                        plt.show()
                    else:
                        pass

                    plotting_counter += 1
            else:
                pass

        largest_plotting_range = max(delta_y_range)
        for it in range(plotting_counter):
            axs_comb[it].set_ylim(mean_delta_y[it] - largest_plotting_range,
                                  mean_delta_y[it] + largest_plotting_range)

        plt.tight_layout()
        fig_comb.savefig(
            'ramseys_calibrations/an1_an2_Ramsey_freq_linear_fits.pdf')
        fig_comb.savefig(
            'ramseys_calibrations/an1_an2_Ramsey_freq_linear_fits.svg',
            format='svg')

        print('\n an_1 Ramsey Table \n', np.round(an1_theory_table, 9))
        print('\n an_2 Ramsey Table \n', np.round(an2_theory_table, 9))
        print('\n Total number of transitions included:', plotting_counter)
        np.savetxt('ramseys_calibrations/an1_ramsey_freq_table.txt',
                   an1_theory_table,
                   delimiter=',')
        np.savetxt('ramseys_calibrations/an2_ramsey_freq_table.txt',
                   an2_theory_table,
                   delimiter=',')

        print('\n Done Ramsey frequency an2 finder. \n\n')
        # print('\n Old calibration discrepancies:', old_cal_discrepancies)
        # print('\n New calibration discrepancies:', new_cal_discrepancies)
        print('Old calibration discrepancies length:',
              len(old_cal_discrepancies))
        print('New calibration discrepancies length:',
              len(new_cal_discrepancies), '\n')

        print('Old calibration mean absolute deviation:',
              np.round(1e6 * np.mean(np.absolute(old_cal_discrepancies)), 3),
              'Hz.')
        print('Old calibration discrepancies stdev:',
              np.round(1e6 * np.std(old_cal_discrepancies), 3), 'Hz.')
        print('Old calibration discrepancies mean:',
              np.round(1e9 * np.mean(old_cal_discrepancies), 3), 'mHz.\n')

        print('New calibration mean absolute deviation:',
              np.round(1e6 * np.mean(np.absolute(new_cal_discrepancies)), 3),
              'Hz.')
        print('New calibration discrepancies stdev:',
              np.round(1e6 * np.std(new_cal_discrepancies), 3), 'Hz.')
        print('New calibration discrepancies mean:',
              np.round(1e12 * np.mean(new_cal_discrepancies), 3), 'uHz.\n')
        print('Ramsey fitting error based on wait time and shots',
              1e6 * Ramsey_fit_error, 'Hz')

        fig, ax = plt.subplots(1, 1, figsize=set_size('half'))
        if compare_old_calibration:
            ax.bar(range(len(old_cal_discrepancies)),
                   1e3 * np.array(np.absolute(old_cal_discrepancies)),
                   label='Freq. scan fits',
                   color='gray',
                   alpha=0.5)
        ax.bar(range(len(new_cal_discrepancies)),
               1e3 * np.array(np.absolute(new_cal_discrepancies)),
               label='Ramsey fits')
        # hline_end = max(len(new_cal_discrepancies),
        #                 len(old_cal_discrepancies))

        ax.set_ylabel('Deviation / kHz')
        ax.set_xlabel('Run number')
        # ax.hlines(1e6 * np.std(np.absolute(old_cal_discrepancies)),
        #           0,
        #           hline_end,
        #           label=r'$\sigma$ - Freq. scans',
        #           linestyle='--',
        #           color='gray')
        # ax.hlines(1e6 * np.std(np.absolute(new_cal_discrepancies)),
        #           0,
        #           hline_end,
        #           label=r'$\sigma$ - Ramseys',
        #           linestyle='--',
        #           color='black')
        # ax.set_xlim(250,350)
        ax.set_title(r'$f_{meas}-f_{fit}$')
        ax.legend()
        ax.grid()
        # plt.yscale('log')

        fig.savefig('ramseys_calibrations/fit_deviation_comparisons.pdf')
        fig.savefig('ramseys_calibrations/fit_deviation_comparisons.svg',
                    format='svg')

        # Function to create a histogram and fit a Gaussian curve
        def plot_histogram_with_gaussian(data, label, ax, color):
            """Plot histogram and fit Gaussian curve.

            Parameters:
                data (array-like): Data points for the histogram.
                label (str): Label for the data set to display in
                the legend.

                ax (matplotlib.axes.Axes): Axes object to plot on.
            """
            # Number of bins calculated as 1 + log2(N) (Sturg's Rule)
            n_bins = int(1 + np.log2(len(data))) + 5

            # Plot histogram
            counts, bin_edges, _ = ax.hist(data,
                                           bins=n_bins,
                                           density=True,
                                           alpha=0.5,
                                           label=f'{label} Histogram',
                                           color=color)

            # Fit a Gaussian to the histogram
            mu, std = sc.stats.norm.fit(
                data)  # Fit a Gaussian (returns mean and std dev)

            # Create a range of values for the Gaussian curve
            x = np.linspace(bin_edges[0], bin_edges[-1], 100)

            # Calculate the Gaussian PDF at those points
            p = sc.stats.norm.pdf(x, mu, std)

            # Plot the Gaussian over the histogram
            ax.plot(x,
                    p,
                    color=color,
                    linewidth=1.5,
                    label=rf'{label} Fit: $\mu$={1e3*mu:.2f}' +
                    rf' Hz, $\sigma$={1e3*std:.2f} Hz')
            ax.legend()

        fig.savefig('ramseys_calibrations/fit_deviation_comparisons.pdf')
        fig.savefig('ramseys_calibrations/fit_deviation_comparisons.svg',
                    format='svg')

        # Prepare the figure and axes for plotting
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        if compare_old_calibration:
            plot_histogram_with_gaussian(1e3 * np.array(old_cal_discrepancies),
                                         'Old Calibration Discrepancies', ax,
                                         colors[0])
        plot_histogram_with_gaussian(1e3 * np.array(new_cal_discrepancies),
                                     'New Calibration Discrepancies', ax,
                                     colors[1])

        # Set titles and labels
        ax.set_ylabel('Density')
        ax.set_xlabel('Deviation / kHz')
        ax.set_title('Histogram and Gaussian Fit of Calibration Discrepancies')
        ax.grid()
        fig.savefig('ramseys_calibrations/fit_discrepancies_distributions.pdf')
        fig.savefig('ramseys_calibrations/fit_discrepancies_distributions.svg',
                    format='svg')

        fig, ax = plt.subplots(1, 1, figsize=set_size('full'))

        sensitivities_x_values = [
            np.abs(self.transition_sensitivities[x])
            for x in matched_transition_list
        ]

        y_values = 1e3 * np.array(new_cal_disc_averaged)
        ax.scatter(sensitivities_x_values, y_values)

        transes = []
        for (x_index, (x_i, x_j)), y in zip(enumerate(matched_transition_list),
                                            y_values):

            transes.append([s12_state, d52_state[0], d52_state[1]])
            s12_state = int(x_j - 2)
            d52_state = self.F_values_D52[23 - x_i]
            ax.text(
                sensitivities_x_values[x_index],
                y,
                # f'{(x_i, x_j)}',
                f'{(s12_state, d52_state[0],d52_state[1])}',
                fontsize=8,
                ha='right')

        ax.set_ylabel('Discrepancy / kHz')
        ax.set_xlabel('Transition Sensitivity / MHz/G')
        # ax.hlines(1e6 * np.std(old_cal_discrepancies),
        #           0,
        #           hline_end,
        #           label=r'$\sigma$ - Freq. scans',
        #           linestyle='--',
        #           color='gray')
        # ax.hlines(1e6 * np.std(new_cal_discrepancies),
        #           0,
        #           hline_end,
        #           label=r'$\sigma$ - Ramseys',
        #           linestyle='--',
        #           color='black')
        # ax.set_xlim(250,350)
        ax.set_title(r'Calibration Discrepancies (per Transition)')
        # ax.legend()
        ax.grid()
        # plt.yscale('log')
        fig.savefig('ramseys_calibrations/' +
                    'fit_deviation_comparisons_per_transition.pdf')
        fig.savefig('ramseys_calibrations/' +
                    'fit_deviation_comparisons_per_transition.svg',
                    format='svg')

        return an1_theory_table, an2_theory_table

    def mag_field_freqs_monitor(
            self,
            Ramsey_wait_time=100,  # us
            shots=250,
            data_cutoff_date=None):
        """Monitors the magnetic field and transition frequencies during Ramsey
        calibrations by analyzing data files.

        Parameters:

            Ramsey_wait_time (int): The wait time in microseconds for the
            Ramsey experiment.

            shots (int): The number of shots in the experiment.

            data_cutoff_date (str): Optional date string to filter data.

        """

        # get the files from Ramsey calibrations
        folder_path_root = ('../data/ramseys_calibrations_data/')
        filename_root = 'ramsey_calibration_triplet_'
        filenames = [
            f for f in os.listdir(folder_path_root)
            if f.startswith(filename_root)
        ]
        filenames = sorted(
            filenames,
            key=lambda t: os.stat(os.path.join(folder_path_root, t)).st_mtime)
        if data_cutoff_date:
            cutoff_date = datetime.strptime(data_cutoff_date, "%Y%m%d_%H%M")

            filenames = [
                f for f in filenames if datetime.fromtimestamp(
                    os.path.getmtime(os.path.join(folder_path_root, f))) >
                cutoff_date
            ]
            filenames = sorted(filenames,
                               key=lambda t: os.stat(
                                   os.path.join(folder_path_root, t)).st_mtime)

        Ramsey_fit_error = np.sqrt(
            (2 * 0.5 * 0.5) / shots) / (2 * np.pi * Ramsey_wait_time)
        Ramsey_freq_triplets = np.empty((3, len(filenames)))

        for idx, filename in enumerate(filenames):
            with open('../data/ramseys_calibrations_data/' + filename,
                      'r') as file:
                # Read the first line (assuming it's a list of floats)
                first_line = file.readline()
                fl_vals = re.findall(r"[-+]?\d*\.\d+|\d+", first_line)
                Ramsey_freq_triplets[:, idx] = [
                    fl_vals[0], fl_vals[1], fl_vals[2]
                ]

        Ramsey_freq_triplets = np.transpose(Ramsey_freq_triplets)

        # B-field taken from a fit a while ago
        # Just a starting point for the fluctuations
        self.B_field = 4.20945
        sensitivities = self.generate_transition_sensitivities()
        upper_sensitivity = sensitivities[22, 1]
        offset_sensitivity = sensitivities[5, 2]

        offset_freqs = []
        upper_freqs = []
        freq_error = []
        mag_fields = []
        mag_fields_error = []
        meas_times = []
        for it, triplet in enumerate(Ramsey_freq_triplets):
            # print(filenames[it])
            offset_freqs.append(triplet[0])
            upper_freqs.append(triplet[1])
            freq_error.append(Ramsey_fit_error)
            chunks = filenames[it].split('_')
            if it == 0:
                # time1 = (str(chunks[-2]) + '_' + str(chunks[-1][:4]))
                # first_time = datetime.strptime(time1, '%Y%m%d_%H%M')
                first_delta_freq = triplet[1] - triplet[0]

            time2 = (str(chunks[-2]) + '_' + str(chunks[-1][:4]))
            current_time = datetime.strptime(time2, '%Y%m%d_%H%M')
            current_delta_freq = triplet[1] - triplet[0]

            # make x-axis "time since last measurement"
            # meas_times.append(
            #     (current_time - first_time).total_seconds() / 3600)

            # or make it absolute date time (convenient for comparing with
            # other things like lab temperature etc)
            meas_times.append(current_time)

            # print(triplet[0], triplet[1],
            #       (current_time - first_time).total_seconds() / 3600)

            new_mag_value = self.B_field + (
                current_delta_freq - first_delta_freq) / (upper_sensitivity -
                                                          offset_sensitivity)
            mag_field_error_value = (2 * Ramsey_fit_error) / (
                upper_sensitivity - offset_sensitivity)
            mag_fields.append(new_mag_value)
            mag_fields_error.append(mag_field_error_value)

        fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        ax[0].set_title(r'$f_{offset}$ Monitor')
        ax[0].errorbar(meas_times,
                       1e3 * (offset_freqs - np.mean(offset_freqs)),
                       yerr=np.abs(freq_error),
                       color=colors[0],
                       fmt='o',
                       markersize=2)
        ax[0].set_ylabel(r'$f-f_0$ / kHz')
        ax[0].grid()

        ax[1].set_title(r'$f_{upper}$ Monitor')
        ax[1].errorbar(meas_times,
                       1e3 * (upper_freqs - np.mean(upper_freqs)),
                       yerr=np.abs(freq_error),
                       color=colors[1],
                       fmt='o',
                       markersize=2)
        ax[1].set_ylabel(r'$f-f_0$ / kHz')
        ax[1].grid()

        ax[2].set_title(r'$B$-field Monitor')
        ax[2].errorbar(meas_times,
                       mag_fields,
                       yerr=np.abs(mag_fields_error),
                       color=colors[2],
                       fmt='o',
                       markersize=2)
        ax[2].set_ylabel('$B$-field / G')
        ax[2].grid()

        plt.xlabel('Time since first measurement / hours')
        plt.xticks(rotation=45)
        fig.savefig('ramseys_calibrations/freqs_Bfield_monitor.pdf')

    def pitimes_monitor(self, data_cutoff_date=None) -> list:
        """Reads data files from the specified directory and extracts the first
        number from each line, appending them to a larger list if the file
        contains 7 lines.

        Parameters
        ----------
        directory : str
            The path to the directory containing the text files.

        Returns
        -------
        tuple
            A tuple containing:
            - List of extracted first numbers (frequencies_fit)
            - List of lists of extracted second numbers (pitimes_fit)
            - List of corresponding datetime strings

        """

        if hasattr(self, 'transition_sensitivities'):
            pass
        else:
            self.generate_transition_sensitivities()

        frequencies_fit = []  # List to hold first numbers from valid files
        pitimes_fit = []  # List of lists to hold second numbers from each line
        datetimes = []  # List to hold datetimes extracted from filenames
        mag_fields = []

        directory = '../data/ramseys_calibrations_pitimes_data'

        filenames = [f for f in os.listdir(directory)]
        filenames = sorted(
            filenames,
            key=lambda t: os.stat(os.path.join(directory, t)).st_mtime)
        if data_cutoff_date:
            cutoff_date = datetime.strptime(data_cutoff_date, "%Y%m%d_%H%M")

            filenames = [
                f for f in filenames if datetime.fromtimestamp(
                    os.path.getmtime(os.path.join(directory, f))) > cutoff_date
            ]
            filenames = sorted(
                filenames,
                key=lambda t: os.stat(os.path.join(directory, t)).st_mtime)
        idxer = 0
        for idx, filename in enumerate(filenames):
            if filename.endswith('.txt'):  # Ensure we only read .txt files
                # Extract date and time from the filename
                datetime_str = filename[-17:
                                        -4]  # Extracting YYYYMMDD_HHMM format
                filepath = os.path.join(directory, filename)

                with open(filepath, 'r') as file:
                    lines = file.readlines()
                    if len(lines
                           ) == 7:  # Check if the file has exactly 7 lines
                        idxer += 1
                        # Extract first and second numbers
                        freq_nums = [
                            float(line.split(',')[0]) for line in lines
                        ]  # First numbers
                        second_nums = [
                            float(line.split(',')[1]) for line in lines
                        ]  # Second numbers
                        if idxer == 1:
                            first_delta_freq = freq_nums[1] - freq_nums[0]
                        current_delta_freq = freq_nums[1] - freq_nums[0]

                        fieldB = 4.216  # just a starting point for B field
                        upper_sensitivity = self.transition_sensitivities[22,
                                                                          1]
                        offset_sensitivity = self.transition_sensitivities[5,
                                                                           2]
                        new_mag_value = fieldB + (
                            current_delta_freq - first_delta_freq) / (
                                upper_sensitivity - offset_sensitivity)

                        # Append data to respective lists
                        frequencies_fit.append(
                            freq_nums)  # Append first numbers
                        pitimes_fit.append(
                            second_nums)  # Append second numbers as a list
                        datetimes.append(
                            datetime.strptime(
                                datetime_str,
                                '%Y%m%d_%H%M'))  # Convert string to datetime
                        mag_fields.append(new_mag_value)

        frequencies_fit = np.array(frequencies_fit)
        pitimes_fit = np.array(pitimes_fit)

        # Plot frequencies over time, with each frequency
        # as a separate line, referenced to initial value
        # Add a bit of an offset to each subsequent plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=set_size('full'))
        for index in range(7):
            ax1.plot(datetimes,
                     0.01 * index + frequencies_fit[:, index] -
                     frequencies_fit[0, index],
                     marker='o',
                     label=f'Frequency {index}')

        # Adding labels and title for the first subplot
        ax1.set_xlabel('Date and Time')
        ax1.set_ylabel('Frequency Changes (kHz)')
        ax1.set_title('Frequencies Over Time')
        ax1.grid(True)
        ax1.legend()

        # Plot magnetic fields on the second subplot
        ax2.plot(datetimes,
                 mag_fields,
                 marker='o',
                 label='Magnetic Field',
                 color='r')

        # Adding labels and title for the second subplot
        ax2.set_xlabel('Date and Time')
        ax2.set_ylabel('Magnetic Field / Gauss')  # Adjust units accordingly
        ax2.set_title('Magnetic Field Values Over Time')
        ax2.grid(True)
        ax2.legend()

        # Adjust layout to prevent clipping of labels
        plt.tight_layout()

        fig, ax = plt.subplots(1, 1, figsize=set_size('full'))
        for index in range(5):
            ax.plot(
                datetimes,
                pitimes_fit[:, index + 2],  # - pitimes_fit[0, index + 2],
                marker='o',
                label=f'Strength {index+2}')

        # Adding labels and title
        ax.set_xlabel('Date and Time')
        ax.set_ylabel(r'Changes in $\tau_\pi$ / $\mu s$')
        ax.set_title('Transition Strengths Over Time')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()  # Adjust layout to prevent clipping of labels

        # Show the plot
        plt.show()

        # Return all lists as a tuple
        return frequencies_fit, pitimes_fit, datetimes

    def fit_Rabi_oscillation(self, create_plots=False):
        """Fits the data of Rabi oscillations and calculates parameters related
        to the pi-times. Generates a figure called
        all_calibrations_pitimes_plot.pdf for visualisation.

        Parameters:

            create_plots (bool): If True, generates plots of the fitted Rabi
            oscillations.

        Returns:
            None

        """

        self.pitime_files = list(
            filter(lambda string: 'pitime' in string, self.file_names))

        if hasattr(self, 'freq_files'):
            print('\n Done fitting frequency scans already. \n')
            pass
        else:
            print('\n Need to run frequency fitting. \n')
            self.fit_freq_Rabi_fringe(create_plots=False)

        list_of_545 = []
        for freq_file in self.freq_files:
            chunks = freq_file.split('_')
            freq = float(chunks[6].replace('p', '.'))

            diff_from_545 = np.abs(self.sim_freqs - freq)
            diff_masked = np.ma.array(diff_from_545,
                                      mask=np.isnan(diff_from_545))
            idx = np.unravel_index(np.argmin(diff_masked, axis=None),
                                   diff_masked.shape)
            # idx = diff_from_545.argmin()
            if idx == (5, 2):
                list_of_545.append(freq_file)

        if self.use_old_data:
            run_length = 38
        else:
            run_length = 8

        # sort pitime_files into chunks associated with 545 data
        self.chunked_pitime_files = np.empty((len(list_of_545), run_length),
                                             dtype=object)
        self.chunked_pitime_fits = np.zeros((len(list_of_545), run_length))
        self.chunked_pitime_fit_errors = np.zeros(
            (len(list_of_545), run_length))
        self.chunked_pitime_fit_percent_errors = np.zeros(
            (len(list_of_545), run_length))

        self.chunked_trans_ids = np.zeros((len(list_of_545), run_length, 3))
        self.chunked_datetimes = np.zeros((len(list_of_545), run_length),
                                          dtype=int)
        self.chunked_datetimes = self.chunked_datetimes.astype(str)

        for idx, file_545 in enumerate(list_of_545):
            # self.chunked_pitime_files[idx, 0] = file_545

            file_545_date = file_545.split('_')[7]
            file_545_time = file_545.split('_')[8][:-4]
            file_545_datetime = file_545_date + '_' + file_545_time
            dummy_idx = 0
            for pitime_file in self.pitime_files:
                pitime_file_date = pitime_file.split('_')[7]
                pitime_file_time = pitime_file.split('_')[8][:-4]
                pitime_file_datetime = (pitime_file_date + '_' +
                                        pitime_file_time)
                if (pitime_file_datetime == file_545_datetime
                        and pitime_file != file_545):
                    self.chunked_pitime_files[idx, dummy_idx] = pitime_file
                    dummy_idx += 1

        # Define the Rabi oscillations fit function
        def oscfunc(t, A, w, p, c, d):
            return A * np.cos(w * t + p) * np.exp(-t / d) + c

        if create_plots:
            n_cols = run_length
            n_rows = len(list_of_545)

            # Create a figure with subplots
            fig_comb, axs_comb = plt.subplots(n_rows,
                                              n_cols,
                                              figsize=(n_cols * 5, n_rows * 5))
            axs_comb = axs_comb.flatten(
            )  # Flatten in case of single row/column

        for idx, cal_chunk in enumerate(tqdm(self.chunked_pitime_files)):

            file_545_date = self.chunked_freq_files[idx][0].split('_')[7]
            file_545_time = self.chunked_freq_files[idx][0].split('_')[8][:-4]
            file_545_datetime = file_545_date + '_' + file_545_time

            for idx_ind, cal_run in enumerate(cal_chunk):
                total_idx = run_length * idx + idx_ind

                # find the associated transition freq/quantum numbers for
                # this pi time scan
                if self.chunked_freq_fits[idx, idx_ind] != 0:
                    diff = np.abs(self.sim_freqs -
                                  self.chunked_freq_fits[idx, idx_ind])
                    diff_masked = np.ma.array(diff,
                                              mask=np.isnan(diff_from_545))
                    idx_trans = np.unravel_index(
                        np.argmin(diff_masked, axis=None), diff_masked.shape)
                    s12_state = int(idx_trans[1] - 2)
                    d52_state = self.F_values_D52[23 - idx_trans[0]]
                elif self.chunked_freq_files[idx][idx_ind] is not None:
                    freq_bandaid = self.chunked_freq_files[idx][idx_ind].split(
                        '_')[6]
                    freq_bandaid = float(freq_bandaid.replace('p', '.'))
                    diff = np.abs(self.sim_freqs - freq_bandaid)
                    diff_masked = np.ma.array(diff,
                                              mask=np.isnan(diff_from_545))
                    idx_trans = np.unravel_index(
                        np.argmin(diff_masked, axis=None), diff_masked.shape)
                    s12_state = int(idx_trans[1] - 2)
                    d52_state = self.F_values_D52[23 - idx_trans[0]]
                else:
                    pass

                if cal_run is not None and self.good_freq_runs[idx,
                                                               idx_ind] == 0:
                    # Load the data from the text file
                    data = np.genfromtxt(self.folder_path_raw + '/' + cal_run,
                                         delimiter=',',
                                         skip_footer=1)

                    # Extract the first and second entries of each row
                    pulse_times = data[:-1, 0]
                    PD = data[:-1, 1]
                    num_SD = 1

                    lower_error, upper_error = self.find_errors(
                        num_SD, PD, self.exp_number)
                    asym_error_bar = [PD - lower_error, upper_error - PD]

                    if create_plots:
                        if idx_ind == 0:
                            axs_comb[total_idx].text(pulse_times[-1],
                                                     0.85,
                                                     f'{file_545_datetime}',
                                                     fontsize=14,
                                                     ha='right',
                                                     va='center')

                        axs_comb[total_idx].errorbar(pulse_times,
                                                     PD,
                                                     yerr=asym_error_bar,
                                                     fmt='o',
                                                     color='r',
                                                     label='data')
                        axs_comb[total_idx].text(np.mean(pulse_times),
                                                 0.5,
                                                 'Poor Frequency Data',
                                                 fontsize=16,
                                                 color='r',
                                                 ha='center',
                                                 va='center')
                        axs_comb[total_idx].set_ylim(0, 1)
                        axs_comb[total_idx].set_title(
                            r'$S_{1/2}$ to $D_{5/2}$: ($m_S,F_D,m_D$)=' +
                            f'$({s12_state},' +
                            f'{d52_state[0]},{d52_state[1]})$')
                    else:
                        pass

                elif cal_run is not None and self.good_freq_runs[idx,
                                                                 idx_ind] == 1:

                    # Load the data from the text file
                    data = np.genfromtxt(self.folder_path_raw + '/' + cal_run,
                                         delimiter=',',
                                         skip_footer=1)

                    # Extract the first and second entries of each row
                    pulse_times = data[:-1, 0]
                    PD = data[:-1, 1]

                    # Define the initial guesses
                    A_guess = np.max(PD) - np.mean(PD)
                    ff = np.fft.fftfreq(len(pulse_times),
                                        (pulse_times[1] - pulse_times[0]))
                    F_PD = abs(np.fft.fft(PD))
                    guess_freq = abs(ff[np.argmax(F_PD[1:]) + 1])
                    w_guess = 2.0 * np.pi * guess_freq
                    p_guess = np.pi
                    c_guess = np.mean(PD)
                    d_guess = 1e4  # us
                    p0 = np.array(
                        [A_guess, w_guess, p_guess, c_guess, d_guess])
                    x_new = np.linspace(pulse_times.min(), pulse_times.max(),
                                        1000)

                    # Find the errors for the plots using Wilson interval
                    num_SD = 1  # number of standard deviations, 1=>68.28%

                    lower_error, upper_error = self.find_errors(
                        num_SD, PD, self.exp_number)
                    asym_error_bar = [PD - lower_error, upper_error - PD]

                    # Fit the data to the Rabi oscillation
                    try:
                        popt, pcov = sc.optimize.curve_fit(oscfunc,
                                                           pulse_times,
                                                           PD,
                                                           p0=p0,
                                                           maxfev=10000)
                        # print(popt)
                        fit_bad = (pcov[2][2] > 10 or popt[2] < 0.3
                                   or pcov[0][0] > 0.005)

                        peak_points_too_low = np.count_nonzero(PD > 0.4) < 6

                        def too_many_zero_points(lst):
                            count_zeros = 0
                            for i in range(len(lst) - 1):
                                if lst[i] == 0 and lst[i + 1] == 0:
                                    count_zeros += 1
                                    if count_zeros > 2:
                                        return True
                                else:
                                    count_zeros = 0
                            return False

                        def too_many_unity_points(lst):
                            count_ones = 0
                            for i in range(len(lst) - 1):
                                if lst[i] == 1 and lst[i + 1] == 1:
                                    count_ones += 1
                                    if count_ones > 4:
                                        return True
                                else:
                                    count_ones = 0
                            return False

                        upload_error = too_many_zero_points(PD)
                        decrystal_error = too_many_unity_points(PD)
                        date_ind = cal_run.split('_')[7]
                        time_ind = cal_run.split('_')[8][:-4]

                        if (fit_bad or peak_points_too_low or upload_error
                                or decrystal_error):
                            if create_plots:
                                if idx_ind == 0:
                                    axs_comb[total_idx].text(
                                        pulse_times[-1],
                                        0.85,
                                        f'{file_545_datetime}',
                                        fontsize=14,
                                        ha='right',
                                        va='center')

                                axs_comb[total_idx].errorbar(
                                    pulse_times,
                                    PD,
                                    yerr=asym_error_bar,
                                    fmt='o',
                                    color='r',
                                    label='data')
                                axs_comb[total_idx].text(np.mean(pulse_times),
                                                         0.5,
                                                         'Poor Data',
                                                         fontsize=16,
                                                         color='r',
                                                         ha='center',
                                                         va='center')
                                axs_comb[total_idx].plot(x_new,
                                                         oscfunc(x_new, *popt),
                                                         'r-',
                                                         label='fit')
                                axs_comb[total_idx].set_ylim(0, 1)
                                s12_state = int(idx_trans[1] - 2)
                                d52_state = self.F_values_D52[23 -
                                                              idx_trans[0]]
                                axs_comb[total_idx].set_title(
                                    r'$S_{1/2}$ to $D_{5/2}$: ($m_S,F_D,m_D$)='
                                    + f'$({s12_state},' +
                                    f'{d52_state[0]},{d52_state[1]})$')
                            else:
                                pass
                        else:
                            t_pi = np.pi / np.absolute(popt[1])
                            s_pi = t_pi * np.absolute(
                                np.sqrt(pcov[1][1]) / (2 * np.pi * popt[1]))
                            s_x0 = round(1e3 * np.sqrt(pcov[1][1]), 3)

                            self.chunked_pitime_fits[idx, idx_ind] = t_pi
                            self.chunked_pitime_fit_errors[idx, idx_ind] = s_pi
                            self.chunked_pitime_fit_percent_errors[
                                idx, idx_ind] = s_pi / t_pi

                            self.chunked_trans_ids[idx, idx_ind,
                                                   0] = int(s12_state)
                            self.chunked_trans_ids[idx, idx_ind,
                                                   1] = int(d52_state[0])
                            self.chunked_trans_ids[idx, idx_ind,
                                                   2] = int(d52_state[1])

                            date_ind = cal_run.split('_')[7]
                            time_ind = cal_run.split('_')[8][:-4]
                            self.chunked_datetimes[
                                idx, idx_ind] = date_ind + '_' + time_ind

                            freq_leg = 1e3 * popt[1]

                            if create_plots:
                                if idx_ind == 0:
                                    axs_comb[total_idx].text(
                                        pulse_times[-1],
                                        0.85,
                                        f'{file_545_datetime}',
                                        fontsize=14,
                                        ha='right',
                                        va='center')
                                axs_comb[total_idx].errorbar(
                                    pulse_times,
                                    PD,
                                    yerr=asym_error_bar,
                                    fmt='o',
                                    label='data')
                                axs_comb[total_idx].set_xlabel(
                                    'Pulse Time (us)')
                                axs_comb[total_idx].set_ylabel(
                                    'Dark State Probability')
                                axs_comb[total_idx].set_title(
                                    r'$S_{1/2}$ to $D_{5/2}$: ($m_S,F_D,m_D$)='
                                    + f'$({s12_state},' +
                                    f'{d52_state[0]},{d52_state[1]})$')
                                axs_comb[total_idx].plot(x_new,
                                                         oscfunc(x_new, *popt),
                                                         'r-',
                                                         label='fit')
                                axs_comb[total_idx].set_ylim(0, 1)
                                axs_comb[total_idx].grid()
                                axs_comb[total_idx].legend(
                                    title=rf"$f$={freq_leg:.1f} kHz" +
                                    rf" $\sigma_f$={s_x0} kHz" + "\n" +
                                    rf"$\tau_\pi$={t_pi:.3f}$\mu s$," +
                                    rf"$\sigma_\pi$={s_pi:.3f}$\mu s$",
                                    loc='lower left',
                                    fontsize=6)
                            else:
                                pass

                    except RuntimeError:
                        pass

                else:
                    if create_plots:
                        axs_comb[total_idx].text(0.5,
                                                 0.5,
                                                 'No Data',
                                                 fontsize=16,
                                                 ha='center',
                                                 va='center')
        if create_plots:
            fig_comb.tight_layout()
            fig_comb.savefig(
                'ramseys_calibrations/all_calibrations_pitimes_plot.pdf')

        print(self.chunked_pitime_fits)
        print(self.chunked_pitime_fit_errors)
        print(self.chunked_pitime_fit_percent_errors)

        # find mean percent error from the fitting of Rabi oscillations
        mask_pitimes = self.chunked_pitime_fit_percent_errors != 0
        mean_pitime_percent_error = np.mean(
            self.chunked_pitime_fit_percent_errors[mask_pitimes])
        print('The mean percent error for pi-times from Rabi fits',
              mean_pitime_percent_error)
        print('\n Done fitting pi-time scans. \n')

    def an1_pitime_finder(
            self,
            time_diff: int = 86400,  # default 1 day time diff
            save_pitime_file=False):
        """Finds the an1 values for scaling of pi-times for all transitions
        using the existing fits and saves the results if specified.

        Parameters:

            time_diff (int): The maximum allowed time difference for matching
            transitions (in seconds).

            save_pitime_file (bool): If True, saves the pi-time results to a
            file.

        Returns:
            None

        """

        if hasattr(self, 'chunked_pitime_fits'):
            print('\n Done fitting pi-time scans already. \n')
        else:
            print('\n Need to run pi-time fitting. \n')
            self.fit_Rabi_oscillation(create_plots=False)

        reference_pi_time_ids = np.array([[-2, 4, -4], [-2, 3, -3], [2, 4, 2],
                                          [2, 4, 3], [2, 4, 4]])

        pi_time_an1s = np.zeros((24, 5))
        pi_time_list = []

        def linear_function(x, slope):
            return slope * x

        # Determine the layout of the subplots
        n_cols = 7
        n_rows = 6
        # Create a figure with subplots
        fig_comb, axs_comb = plt.subplots(n_rows,
                                          n_cols,
                                          figsize=(n_cols * 5, n_rows * 5))
        axs_comb = axs_comb.flatten()

        # going one transition at a time, find all its pi-times
        # and match it to the reference transition
        plotting_counter = 0
        for (row, col), trans in np.ndenumerate(self.sim_freqs):
            pi_times = []
            ref_pitimes = []

            if trans != np.nan:
                s12_state = int(col - 2)
                d52_state = self.F_values_D52[23 - row]
                id1 = [s12_state, d52_state[0], d52_state[1]]
                delta_m_value = id1[2] - id1[0]

                for chunk_id, chunk in enumerate(self.chunked_trans_ids):
                    for trans_id, trans_triplet in enumerate(chunk):
                        if np.array_equal(trans_triplet, id1):
                            check_ref = (np.array(trans_triplet) ==
                                         reference_pi_time_ids).all(axis=1)
                            if check_ref.any():
                                pi_times.append(
                                    self.chunked_pitime_fits[chunk_id,
                                                             trans_id])
                                ref_pitimes.append(
                                    self.chunked_pitime_fits[chunk_id,
                                                             trans_id])
                                break

                            for chunk_id2, chunk2 in enumerate(
                                    self.chunked_trans_ids):
                                for trans_id2, trans_triplet2 in enumerate(
                                        chunk2):
                                    if np.array_equal(
                                            trans_triplet2,
                                            reference_pi_time_ids[delta_m_value
                                                                  + 2]):
                                        # Define the datetime format
                                        FMT = '%Y%m%d_%H%M'
                                        # Convert timestamps to datetimes
                                        trans_time = datetime.strptime(
                                            self.chunked_datetimes[chunk_id,
                                                                   trans_id],
                                            FMT)
                                        ref_time = datetime.strptime(
                                            self.chunked_datetimes[chunk_id2,
                                                                   trans_id2],
                                            FMT)

                                        # Calculate the time difference
                                        time_difference = trans_time - ref_time
                                        if (time_difference.total_seconds()
                                                < time_diff
                                                and time_difference.
                                                total_seconds() >= 0):

                                            pi_times.append(
                                                self.chunked_pitime_fits[
                                                    chunk_id, trans_id])
                                            ref_pitimes.append(
                                                self.chunked_pitime_fits[
                                                    chunk_id2, trans_id2])
                                            pi_time_list.append([
                                                self.chunked_pitime_fits[
                                                    chunk_id, trans_id],
                                                self.chunked_pitime_fits[
                                                    chunk_id2, trans_id2],
                                                self.chunked_freq_fits[
                                                    chunk_id,
                                                    0], self.chunked_freq_fits[
                                                        chunk_id, 1],
                                                self.chunked_freq_fits[
                                                    chunk_id, trans_id],
                                                float(self.chunked_datetimes[
                                                    chunk_id,
                                                    trans_id].split('_')[0]) +
                                                1e-4 *
                                                float(self.chunked_datetimes[
                                                    chunk_id,
                                                    trans_id].split('_')[1])
                                            ])
                                        else:
                                            pass
                                    else:
                                        pass

                        else:
                            pass
                if save_pitime_file:
                    np.savetxt(
                        'ramseys_calibrations/triplets_for_an1_pitime.txt',
                        pi_time_list,
                        delimiter=',')
                else:
                    pass
                if len(pi_times) > 0:
                    popt, pcov = sc.optimize.curve_fit(linear_function,
                                                       ref_pitimes, pi_times)

                    axs_comb[plotting_counter].plot(ref_pitimes,
                                                    pi_times,
                                                    'o',
                                                    label='data')
                    axs_comb[plotting_counter].set_ylabel(
                        r'Transition $\pi$-time ($\mu$s)')
                    axs_comb[plotting_counter].set_xlabel(
                        r'Reference $\pi$-time ($\mu$s)')
                    axs_comb[plotting_counter].xaxis.set_major_formatter(
                        plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
                    axs_comb[plotting_counter].yaxis.set_major_formatter(
                        plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
                    axs_comb[plotting_counter].set_title(
                        r'$S_{1/2}$ to $D_{5/2}$: ($m_S,F_D,m_D$)=' +
                        f'$({id1[0]},' + f'{id1[1]},{id1[2]})$')

                    # x_values = np.linspace(0, max(ref_pitimes), 100)
                    x_values = np.linspace(
                        min(ref_pitimes) - 1,
                        max(ref_pitimes) + 1, 100)
                    axs_comb[plotting_counter].plot(x_values,
                                                    linear_function(
                                                        x_values, popt[0]),
                                                    label='fit')
                    axs_comb[plotting_counter].grid()
                    pi_time_an1s[row, col] = popt[0]
                    plotting_counter += 1
                else:
                    pass
        print('\n an_1 Pi-Time Table \n', pi_time_an1s)
        np.savetxt('ramseys_calibrations/an1_pitime_table.txt',
                   pi_time_an1s,
                   delimiter=',')
        fig_comb.tight_layout()
        fig_comb.savefig('ramseys_calibrations/an1_pitime_linear_fits.pdf')

    def fill_transition_table(self, trans_table, s_spacings):
        """Fills in the transition table by using measured transition
        frequencies to extract S1/2 level spacings, in order to get all 80
        transition frequencies without having to measure 80 transitions.

        Parameters:

            trans_table (np.ndarray): The transition table to fill in.

            s_spacings (list): List of spacing values for transitions.

        Returns:

            full_trans_table (np.ndarray): The filled transition table.

        """
        self.generate_delta_m_table()
        delta_m = np.flip(self.delta_m[:, -5:], axis=0)
        full_trans_table = trans_table

        for i in range(full_trans_table.shape[0]):
            for j in range(full_trans_table.shape[1]):
                if full_trans_table[i][j] != 0:
                    pass
                elif np.abs(delta_m[i][j]) > 2:
                    pass
                else:
                    if j > 0 and full_trans_table[i][j - 1] != 0:
                        full_trans_table[i][j] = full_trans_table[i][
                            j - 1] + s_spacings[j - 1]
                    else:
                        pass

        for i in range(full_trans_table.shape[0] - 1, -1, -1):
            for j in range(full_trans_table.shape[1] - 1, -1, -1):
                if full_trans_table[i][j] != 0:
                    pass
                elif np.abs(delta_m[i][j]) > 2:
                    pass
                else:
                    if j < 4 and full_trans_table[i][j + 1] != 0:
                        full_trans_table[i][j] = full_trans_table[i][
                            j + 1] - s_spacings[j]
                    else:
                        pass
        return full_trans_table

    def run_calibrated_frequency_generation(self,
                                            insensB: float = 545.50521,
                                            sensB: float = 623.26655,
                                            ref_pitimes=None,
                                            print_result=False):
        """Generates calibrated frequency and pi-time values based on the
        provided insensB and sensB values, and ref_pitimes values. This uses
        the an1, an2, and an1-pitime data analysed above.

        Parameters:

            insensB (float): The insensitive B-field frequency.

            sensB (float): The sensitive B-field frequency.

            ref_pitimes (list): A list of reference pi-times used for
            calculations. If not provided it will use self.ref_pitimes.

            print_result (bool): If True, prints the calculated results.

        Returns:

            full_calibrated_transitions (np.ndarray): The transition
            frequencies after calibration.

            calibrated_pitimes (np.ndarray): The calculated pi-times after
            calibration.

        """

        if ref_pitimes:
            pass
        elif not ref_pitimes:
            ref_pitimes = self.ref_pitimes

        self.an1_table = np.loadtxt(
            'ramseys_calibrations/an1_ramsey_freq_table.txt', delimiter=',')
        self.an2_table = np.loadtxt(
            'ramseys_calibrations/an2_ramsey_freq_table.txt', delimiter=',')
        self.an1_pitime_table = np.loadtxt(
            'ramseys_calibrations/an1_pitime_table.txt', delimiter=',')

        an1_pitime_theory_table = np.zeros_like(self.an1_pitime_table)

        calibrated_transitions = np.zeros((24, 5))
        calibrated_pitimes = np.zeros((24, 5))

        calibrated_transitions[22, 1] = sensB
        calibrated_transitions[5, 2] = insensB

        for (row, col), an2_val in np.ndenumerate(self.an2_table):
            if an2_val != 0:
                calibrated_transitions[row, col] = self.an1_table[row, col] * (
                    sensB - insensB) + insensB + an2_val

        s_spacings = []
        for idx in range(4):
            spacer = []
            for row in range(24):
                if calibrated_transitions[row][
                        idx] != 0 and calibrated_transitions[row][idx +
                                                                  1] != 0:
                    spacer.append(calibrated_transitions[row][idx + 1] -
                                  calibrated_transitions[row][idx])
            s_spacings.append(np.mean(spacer))

        full_calibrated_transitions = self.fill_transition_table(
            calibrated_transitions, s_spacings)

        for (row, col), an1_val in np.ndenumerate(self.an1_pitime_table):
            if an1_val != 0:
                s12_state = int(col - 2)
                d52_state = self.F_values_D52[23 - row]
                id1 = [s12_state, d52_state[0], d52_state[1]]
                delta_m_value = id1[2] - id1[0]
                calibrated_pitimes[row,
                                   col] = an1_val * ref_pitimes[delta_m_value +
                                                                2]

        # block below is for generating a comparison table for pi-times
        # which is generated entirely from theory
        theory_pitimes = np.round(self.generate_frequencies_pitimes_table(), 3)
        for (row, col), pi_time in np.ndenumerate(theory_pitimes):
            if not np.isnan(pi_time):
                s12_state = int(col - 2)
                d52_state = self.F_values_D52[23 - row]
                id1 = [s12_state, d52_state[0], d52_state[1]]
                delta_m_value = id1[2] - id1[0]
                an1_pitime_theory_table[
                    row, col] = pi_time / ref_pitimes[delta_m_value + 2]

        if print_result:
            # np.savetxt('ramseys_calibrations/an1_pitime_theory_table.txt',
            #            an1_pitime_theory_table,
            #            delimiter=',')
            print('\n calculated s-level spacings \n',
                  np.round(np.array(s_spacings), 6))
            print('\n full calculated transition table \n',
                  full_calibrated_transitions)
            print('\n full calculated pi-time table \n', calibrated_pitimes)
            print('\n full theory pi-time table \n', theory_pitimes)

        self.calibrated_pitimes = calibrated_pitimes
        self.theory_pitimes = theory_pitimes

        self.full_calibrated_transitions = full_calibrated_transitions

        return full_calibrated_transitions, calibrated_pitimes

    def compare_experiment_calibration(self):
        """Compares experimental calibration results with theoretical
        values for frequency and pi-time discrepancies.

        Returns:
            None

        """

        path = ('ramseys_calibrations/')
        self.an1_table = np.loadtxt(path + 'an1_freq_table.txt', delimiter=',')
        self.an2_table = np.loadtxt(path + 'an2_freq_table.txt', delimiter=',')
        self.an1_pitime_table = np.loadtxt(path + 'an1_pitime_table.txt',
                                           delimiter=',')
        self.an1_an2_data = np.loadtxt(path + 'triplets_for_an1_an2.txt',
                                       delimiter=',')
        self.an1_pitime_data = np.loadtxt(path + 'triplets_for_an1_pitime.txt',
                                          delimiter=',')

        if hasattr(self, 'an1_an2_data'):
            print('\n Done fitting frequency scans already. \n')
            pass
        else:
            print('\n Need to run frequency fitting. \n')
            self.fit_freq_Rabi_fringe(create_plots=False)

        freq_discrepancies = np.zeros((24, 5))
        meas_freqs = np.zeros((24, 5))
        pitime_discrepancies = np.zeros((24, 5))
        meas_pitimes = np.zeros((24, 5))
        predicted_pitimes = np.zeros((24, 5))

        for (row, col), an1_val in np.ndenumerate(self.an1_table):
            if an1_val != 0:
                for it, triplet in enumerate(np.flipud(self.an1_an2_data)):
                    transitions, pitimes = (
                        self.run_calibrated_frequency_generation(
                            insensB=triplet[0], sensB=triplet[1]))
                    if np.absolute(triplet[2] - transitions[row, col]) < 0.020:
                        # print(triplet[2])
                        # print(transitions[row, col])
                        # print(triplet[2] - triplet[0])
                        # print(triplet[1] - triplet[0])
                        freq_discrepancies[row, col] = 1e3 * np.absolute(
                            triplet[2] - transitions[row, col])
                        meas_freqs[row, col] = triplet[2]
                        # print(triplet)
                        break
        print(
            '\n Frequency discrepancies, ' +
            'experiment vs. calibration, in kHz. \n', freq_discrepancies)
        # print('\n Measured frequencies. \n', meas_freqs)

        for (row, col), an1_val in np.ndenumerate(self.an1_pitime_table):
            if an1_val != 0:
                for it, quintet in enumerate(np.flipud(self.an1_pitime_data)):
                    transitions, pitimes = (
                        self.run_calibrated_frequency_generation(
                            insensB=quintet[2],
                            sensB=quintet[3],
                            ref_pitimes=[
                                quintet[1], quintet[1], quintet[1], quintet[1],
                                quintet[1]
                            ]))
                    if np.absolute(quintet[4] - transitions[row, col]) < 0.020:
                        # print(quintet[2])
                        # print(transitions[row, col])
                        # print(quintet[2] - quintet[0])
                        # print(quintet[1] - quintet[0])
                        pitime_discrepancies[row, col] = (
                            np.absolute(pitimes[row, col] - quintet[0]) /
                            quintet[0]) * 100
                        # print(quintet[0], quintet[1], quintet[4], quintet[5])
                        meas_pitimes[row, col] = quintet[0]
                        predicted_pitimes[row, col] = pitimes[row, col]
                        break

        self.meas_freqs = meas_freqs
        self.predicted_freqs = meas_freqs - 1e-3 * freq_discrepancies

    def plot_predicted_measured_freqs_strengths(self):
        """Plots the predicted and measured frequencies along with their
        respective strengths.

        Returns:
            None

        """

        if hasattr(self, 'predicted_freqs'):
            pass
        else:
            self.compare_experiment_calibration()
        self.run_calibrated_frequency_generation(insensB=545.42015,
                                                 sensB=623.23880,
                                                 print_result=False)

        self.calibrated_pitimes[self.calibrated_pitimes == 0] = np.nan

        self.theory_pitimes = np.round(
            self.generate_frequencies_pitimes_table(), 3)
        print('printing')
        print(self.calibrated_pitimes, self.theory_pitimes, self.meas_freqs,
              self.predicted_freqs)

        # print(np.nanmin(self.theory_pitimes))

        self.calibrated_strengths = np.nanmin(
            self.calibrated_pitimes) / self.calibrated_pitimes
        self.theory_strengths = np.nanmin(
            self.calibrated_pitimes) / self.theory_pitimes

        plot_colors = [
            'C3', 'C3', 'C3', 'C2', 'C2', 'C2', 'C2', 'C2', 'C1', 'C1', 'C1',
            'C1', 'C1', 'C1', 'C1', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0',
            'C0', 'C0'
        ]

        # we make these frequencies relative to Ba138 - 3.014GHz (S1/2 shift)
        # add together D5/2 F=2 shift and 1762 laser lock
        offset = 33.144196 + 545.42015

        # fig, ax = plt.subplots(figsize=set_size(width='half'))
        fig, ax = plt.subplots(figsize=(430 * 2 / 72.27, 180 * 2 / 72.27))
        for row_idx, row in enumerate(self.calibrated_pitimes):
            for col_idx, val in enumerate(row):
                if not np.isnan(val) and self.meas_freqs[
                        row_idx,
                        col_idx] != 0 and self.predicted_freqs[row_idx,
                                                               col_idx] != 0:
                    if row_idx in [1, 6, 8, 15]:
                        i = np.argwhere(np.array([1, 6, 8, 15]) == row_idx) + 1
                        ax.vlines(offset -
                                  self.predicted_freqs[row_idx, col_idx],
                                  0,
                                  self.theory_strengths[row_idx, col_idx],
                                  color=plot_colors[row_idx],
                                  label=r'$\tilde{F}$' + f'={i[0][0]}',
                                  alpha=0.5)
                    else:
                        ax.vlines(offset -
                                  self.predicted_freqs[row_idx, col_idx],
                                  0,
                                  self.theory_strengths[row_idx, col_idx],
                                  color=plot_colors[row_idx],
                                  alpha=0.5)
                    ax.errorbar(offset - self.meas_freqs[row_idx, col_idx],
                                self.calibrated_strengths[row_idx, col_idx],
                                yerr=0.02 *
                                self.calibrated_strengths[row_idx, col_idx],
                                fmt='o',
                                markerfacecolor='none',
                                color=plot_colors[row_idx])
        ax.set_title('Transition Frequencies, Relative Strengths')
        ax.set_xlabel(r'$f - f_0$ / MHz')
        ax.set_ylabel('Transition Strength')
        ax.legend()
        ax.grid()

        strength_diffs = []
        strength_freqs = []
        strength_diffs_colours = []

        strength_diffs_table = (
            self.calibrated_strengths -
            self.theory_strengths) / self.calibrated_strengths
        print(strength_diffs_table)
        for row_idx, row in enumerate(strength_diffs_table):
            for col_idx, col in enumerate(row):
                if not np.isnan(col) and col != 0:
                    if (col_idx == 1 and
                            row_idx) == 22 or col_idx == 2 and row_idx == 5:
                        pass
                    else:
                        strength_diffs.append(np.abs(col))
                        strength_freqs.append(offset -
                                              self.meas_freqs[row_idx,
                                                              col_idx])
                        if row_idx < 3:
                            strength_diffs_colours.append('C3')
                        if row_idx < 8 and row_idx >= 3:
                            strength_diffs_colours.append('C2')
                        if row_idx < 15 and row_idx >= 8:
                            strength_diffs_colours.append('C1')
                        if row_idx >= 15:
                            strength_diffs_colours.append('C0')

        mean_abs_error = np.mean(strength_diffs)
        print('Mean absolute percent error, pi-times', mean_abs_error)
        # what is the transition infidelity one can expect given that
        # average pi-time error, assumes perfectly resonant transition
        avg_pi_time_error = np.square(np.sin((1 + mean_abs_error) * np.pi))
        print('Expected average pi-time error', avg_pi_time_error)

        f_levels = [1, 2, 3, 4]
        fig, ax = plt.subplots(figsize=(430 * 2 / 72.27, 70 * 2 / 72.27))
        j = 0
        for i in range(len(strength_diffs)):
            if i in [1, 6, 13, len(strength_diffs) - 1]:
                ax.vlines(strength_freqs[i],
                          0,
                          strength_diffs[i],
                          color=strength_diffs_colours[i],
                          label=r'$\tilde{F}$' + f'={f_levels[j]}')
                j += 1
            else:
                ax.vlines(strength_freqs[i],
                          0,
                          strength_diffs[i],
                          color=strength_diffs_colours[i])
        ax.set_title('Relative Strengths Error')
        ax.set_xlabel(r'$f - f_0$ / MHz')
        ax.set_ylabel('Percent Error')
        ax.legend(loc='upper left')
        ax.grid()

    def show_individual_transitions(self, transition=545.331):
        """Displays individual transition plots for a specified transition
        frequency, in order to easily see the spread in frequencies for various
        transitions and check for consistency with what is expected given their
        B field sensitivities.

        Parameters:

            transition (float): The transition frequency to analyze and display
            plots for.

        Returns:

            fitting_triplets (list): A list of fitting triplets associated with
            the transition.

        """

        fitting_triplets = self.fit_freq_Rabi_fringe(create_plots=False)
        self.fit_Rabi_oscillation(create_plots=False)
        chosen_frequencies = []
        chosen_freq_files = []
        for idx, cal_chunk in enumerate(self.chunked_freq_fits):
            if np.nanmin(np.absolute(cal_chunk - transition)) < 0.05:
                if self.good_freq_runs[idx, 0] == 1 and self.good_freq_runs[
                        idx, 1] == 1 and self.good_freq_runs[
                            idx,
                            np.argmin(np.absolute(cal_chunk -
                                                  transition))] == 1:
                    # print(cal_chunk)
                    # print(np.argmin(np.absolute(cal_chunk - transition)))
                    chosen_frequencies.append([
                        self.chunked_freq_fits[idx, 0],
                        self.chunked_freq_fits[idx, 1], self.chunked_freq_fits[
                            idx,
                            np.argmin(np.absolute(cal_chunk - transition))]
                    ])
                    chosen_freq_files.append([
                        self.chunked_freq_files[idx, 0],
                        self.chunked_freq_files[idx, 1],
                        self.chunked_freq_files[
                            idx,
                            np.argmin(np.absolute(cal_chunk - transition))]
                    ])
        # print(chosen_frequencies)
        # print(chosen_freq_files)

        # Define the sin-squared transition profile for fitting
        def sin_squared(f, f0, A, Omega, t):
            return A * (Omega**2 / (Omega**2 + (f - f0)**2)) * (np.sin(
                np.sqrt(Omega**2 + (f - f0)**2) * t / 2))**2

        n_cols = 3
        n_rows = 20
        colors = [plt.cm.tab20(i) for i in range(20)]

        # Create a figure with subplots
        fig_comb, axs_comb = plt.subplots(n_rows,
                                          n_cols,
                                          figsize=(n_cols * 5, n_rows * 5))
        axs_comb = axs_comb.flatten()  # Flatten in case of single row/column

        for idx, cal_chunk in enumerate(chosen_freq_files):
            file_545_date = cal_chunk[0].split('_')[7]
            file_545_time = cal_chunk[0].split('_')[8][:-4]
            file_545_datetime = file_545_date + '_' + file_545_time

            for idx_ind, cal_run in enumerate(cal_chunk):
                # Load the data from the text file
                data = np.genfromtxt(self.folder_path_raw + '/' + cal_run,
                                     delimiter=',',
                                     skip_footer=1)
                total_idx = 3 * idx + idx_ind

                # Extract the first and second entries of each row
                freqs = data[:-1, 0]
                PD = data[:-1, 1]
                freqs_popped = data[:-1, 0]
                PD_popped = data[:-1, 1]

                # Define the initial guesses
                # best way to find centre freq when pi-time is uncertain
                f0_guess = np.sum(np.multiply(freqs, PD)) / np.sum(PD)
                # need to check if this will be a good f0_guess
                # it won't be if we are too frequency uncertain
                f_range = freqs[-1] - freqs[0]
                top_30_percent = freqs[-1] - f_range / 3
                bot_30_percent = freqs[0] + f_range / 3
                if f0_guess > top_30_percent or f0_guess < bot_30_percent:
                    f0_guess = freqs[np.argmax(PD)]

                # Omega_guess = 0.1 * ((freqs[-1] - freqs[0]))  # in MHz
                f_variance = (freqs - f0_guess)**2
                Omega_guess = np.sqrt(
                    np.sum(np.multiply(f_variance, PD)) / np.sum(PD) / 4)

                A_guess = max(PD)
                t_guess = np.pi / Omega_guess
                p0 = np.array([f0_guess, A_guess, Omega_guess, t_guess])
                x_new = np.linspace(freqs.min(), freqs.max(), 1000)

                # Find the errors for the plots using Wilson interval
                num_SD = 1  # number of standard deviations, 1=>68.28%

                lower_error, upper_error = self.find_errors(
                    num_SD, PD, self.exp_number)
                # asym_error_bar = [PD - lower_error, upper_error - PD]

                # Fit the data to the Rabi fringe function
                try:
                    popt, pcov = sc.optimize.curve_fit(sin_squared,
                                                       freqs,
                                                       PD,
                                                       p0=p0,
                                                       maxfev=10000)

                    for i in [1, 2, 3]:
                        pop_indices = [
                            idx for idx, freq in enumerate(freqs)
                            if np.absolute(sin_squared(freq, *popt) -
                                           PD[idx]) > 0.3
                        ]
                        freqs_popped = np.delete(freqs, pop_indices)
                        PD_popped = np.delete(PD, pop_indices)

                        if len(freqs_popped) >= 4:
                            popt, pcov = sc.optimize.curve_fit(sin_squared,
                                                               freqs_popped,
                                                               PD_popped,
                                                               p0=p0,
                                                               maxfev=10000)
                        else:
                            pass

                    axs_comb[total_idx].text(np.mean(freqs) -
                                             (freqs[-1] - freqs[0]) / 3,
                                             0.9,
                                             f'{file_545_datetime}',
                                             fontsize=10,
                                             ha='center',
                                             va='center')
                    axs_comb[total_idx].set_ylim(0, 1)
                    lower_error_popped, upper_error_popped = (self.find_errors(
                        num_SD, PD_popped, self.exp_number))
                    asym_error_bar_popped = [
                        PD_popped - lower_error_popped,
                        upper_error_popped - PD_popped
                    ]
                    axs_comb[total_idx].errorbar(freqs, PD, color='r', fmt='o')
                    axs_comb[total_idx].errorbar(freqs_popped,
                                                 PD_popped,
                                                 yerr=asym_error_bar_popped,
                                                 fmt='o',
                                                 color=colors[idx],
                                                 label='data')
                    axs_comb[total_idx].plot(x_new,
                                             sin_squared(x_new, *popt),
                                             color=colors[idx])

                    diff = np.abs(self.sim_freqs - popt[0])
                    diff_masked = np.ma.array(diff, mask=np.isnan(diff))
                    idx_trans = np.unravel_index(
                        np.argmin(diff_masked, axis=None), diff_masked.shape)
                    s12_state = int(idx_trans[1] - 2)
                    d52_state = self.F_values_D52[23 - idx_trans[0]]
                    axs_comb[total_idx].set_title(
                        r'$S_{1/2}$ to $D_{5/2}$: ($m_S,F_D,m_D$)=' +
                        f'$({s12_state},' + f'{d52_state[0]},{d52_state[1]})$')
                    # Simplify x-axis labels
                    axs_comb[total_idx].xaxis.set_major_formatter(
                        plt.FuncFormatter(lambda x, _: f'{x:.3f}'))

                    axs_comb[total_idx].legend(title=rf"$f$={popt[0]:.6f} MHz",
                                               loc='lower left',
                                               fontsize=6)
                    axs_comb[total_idx].grid()
                except RuntimeError:
                    pass

        fig_comb.tight_layout()
        fig_comb.savefig(
            'ramseys_calibrations/selected_calibrations_freqs_plots.pdf')
        plt.close(fig_comb)

        # get id for transition for plotting legend
        diff = np.abs(self.sim_freqs - transition)
        diff_masked = np.ma.array(diff, mask=np.isnan(diff))
        idx_trans = np.unravel_index(np.argmin(diff_masked, axis=None),
                                     diff_masked.shape)
        s12_state = int(idx_trans[1] - 2)
        d52_state = self.F_values_D52[23 - idx_trans[0]]

        print('\n Getting fitting triplets of an1 an2 plot. \n')
        used_triplets = []
        indexer = 0
        fig, ax = plt.subplots()
        print(f'\n Triplets for desired frequency {transition}')
        for idx, triplet in enumerate(fitting_triplets):
            if np.absolute(triplet[2] - transition) < 0.05:
                print(triplet)
                used_triplets.append(triplet)
                plt.scatter(triplet[1] - triplet[0],
                            triplet[2] - triplet[0],
                            color=colors[indexer])
                plt.text(triplet[1] - triplet[0],
                         triplet[2] - triplet[0],
                         f'{indexer+1}',
                         fontsize=10,
                         ha='center',
                         va='top')
                indexer += 1
        ax.set_ylabel(r'$f_n$ - $F=2,m=0$ (MHz)')
        ax.set_xlabel(r'$\Delta f_\pm$ (MHz)')
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f'{y:.3f}'))
        plt.title(r'$S_{1/2}$ to $D_{5/2}$: ($m_S,F_D,m_D$)=' +
                  f'$({s12_state},' + f'{d52_state[0]},{d52_state[1]})$')
        plt.legend(title=rf"$f$={transition:.4f} MHz",
                   loc='lower left',
                   fontsize=6)

        # print(f'Triplets for desired frequency {transition}', used_triplets)
        plt.grid()

        return fitting_triplets

    def frequency_jungle_with_Rabi_fringes(self,
                                           B_field=None,
                                           x_sec_freq: float = 1.113,
                                           y_sec_freq: float = 1.447,
                                           micromotion_freq: float = 20.75,
                                           include_sidebands: bool = True,
                                           decohering_pulse=False):
        """Generates a comprehensive plot of various Rabi fringes and
        sidebands in frequency space.

        Parameters:

            x_sec_freq (float): The secular frequency in the x-direction.

            y_sec_freq (float): The secular frequency in the y-direction.

            micromotion_freq (float): The micromotion frequency.

            include_sidebands (bool): If True, plots sideband frequencies.

            decohering_pulse (bool): If True, applies the decohering pulse.

        Returns:
            None
        """

        # set some value for eta - the sideband suppression factor
        eta = 20

        if hasattr(self, 'theory_pitimes'):
            pass
        else:
            self.run_calibrated_frequency_generation()

        if B_field:
            self.B_field = B_field
        self.generate_transition_frequencies()

        frequencies = self.transition_frequencies
        pitimes = self.theory_pitimes

        # Get a colormap with as many colors as there are S-states
        cmap = get_cmap("tab10", frequencies.shape[1])

        # Initialise the plot
        fig, ax = plt.subplots(figsize=set_size(width='full'))

        # Define the sin-squared transition profile for plotting
        def sin_squared(f, f_res, pitime):
            Omega = 1 / (2 * pitime)
            if decohering_pulse:
                return (0.5 * Omega**2 / (Omega**2 + (f - f_res)**2))
            else:
                return (Omega**2 / (Omega**2 + (f - f_res)**2)) * (np.sin(
                    (np.pi / (2 * Omega)) * np.sqrt(Omega**2 +
                                                    (f - f_res)**2)))**2

        for col_idx in range(frequencies.shape[1]):
            column_values = frequencies[:, col_idx]
            label_name = r'$|$' + self.plotting_names_S12[col_idx +
                                                          3] + r'$\rangle$'
            for row_idx, value in enumerate(column_values):
                if value != 0:
                    # go 20kHz on either side to plot
                    pitime = pitimes[row_idx, col_idx]
                    omeg = 1 / (2 * pitime)
                    # sideband frequencies are much lower
                    # this assumes an eta of 1/20
                    omeg_sb = 1 / (2 * pitime * eta)

                    freqs_plot = np.linspace(value - 20 * omeg,
                                             value + 20 * omeg, 1000)

                    ax.plot(freqs_plot,
                            sin_squared(freqs_plot, value, pitime),
                            color=cmap(col_idx),
                            linestyle='-',
                            label=label_name)
                    if decohering_pulse:
                        annot_pos = 0.25
                    else:
                        annot_pos = 0.45
                    ax.annotate(r'$|$' +
                                self.plotting_names_D52[23 - row_idx] +
                                r'$\rangle$',
                                xy=(value, annot_pos),
                                xytext=(-5, 0),
                                textcoords="offset points",
                                rotation=-90,
                                color=cmap(col_idx),
                                bbox=dict(boxstyle='round,pad=0.3',
                                          facecolor='white',
                                          edgecolor='white'))
                    if include_sidebands:
                        # Plot positive x secular frequency sidebands
                        freqs_plot = np.linspace(
                            value + x_sec_freq - 20 * omeg_sb,
                            value + x_sec_freq + 20 * omeg_sb, 1000)
                        ax.plot(freqs_plot,
                                sin_squared(freqs_plot, value + x_sec_freq,
                                            pitime * eta),
                                color=cmap(col_idx),
                                linestyle='--',
                                label=label_name)
                        ax.annotate(r'$|$' +
                                    self.plotting_names_D52[23 - row_idx] +
                                    r'$\rangle$',
                                    xy=(value + x_sec_freq, 0.45),
                                    xytext=(-5, 0),
                                    textcoords="offset points",
                                    rotation=-90,
                                    color=cmap(col_idx),
                                    bbox=dict(boxstyle='round,pad=0.3',
                                              facecolor='white',
                                              edgecolor='white'))

                        # Plot negative x secular frequency sidebands
                        freqs_plot = np.linspace(
                            value - x_sec_freq - 20 * omeg_sb,
                            value - x_sec_freq + 20 * omeg_sb, 1000)
                        ax.plot(freqs_plot,
                                sin_squared(freqs_plot, value - x_sec_freq,
                                            pitime * eta),
                                color=cmap(col_idx),
                                linestyle='--',
                                label=label_name)
                        ax.annotate(r'$|$' +
                                    self.plotting_names_D52[23 - row_idx] +
                                    r'$\rangle$',
                                    xy=(value - x_sec_freq, 0.45),
                                    xytext=(-5, 0),
                                    textcoords="offset points",
                                    rotation=-90,
                                    color=cmap(col_idx),
                                    bbox=dict(boxstyle='round,pad=0.3',
                                              facecolor='white',
                                              edgecolor='white'))

                        # Plot positive y secular frequency sidebands
                        freqs_plot = np.linspace(
                            value + y_sec_freq - 20 * omeg_sb,
                            value + y_sec_freq + 20 * omeg_sb, 1000)
                        ax.plot(freqs_plot,
                                sin_squared(freqs_plot, value + y_sec_freq,
                                            pitime * eta),
                                color=cmap(col_idx),
                                linestyle='-.',
                                label=label_name)
                        ax.annotate(r'$|$' +
                                    self.plotting_names_D52[23 - row_idx] +
                                    r'$\rangle$',
                                    xy=(value + y_sec_freq, 0.45),
                                    xytext=(-5, 0),
                                    textcoords="offset points",
                                    rotation=-90,
                                    color=cmap(col_idx),
                                    bbox=dict(boxstyle='round,pad=0.3',
                                              facecolor='white',
                                              edgecolor='white'))

                        # Plot negative y secular frequency sidebands
                        freqs_plot = np.linspace(
                            value - y_sec_freq - 20 * omeg_sb,
                            value - y_sec_freq + 20 * omeg_sb, 1000)
                        ax.plot(freqs_plot,
                                sin_squared(freqs_plot, value - y_sec_freq,
                                            pitime * eta),
                                color=cmap(col_idx),
                                linestyle='-.',
                                label=label_name)
                        ax.annotate(r'$|$' +
                                    self.plotting_names_D52[23 - row_idx] +
                                    r'$\rangle$',
                                    xy=(value - y_sec_freq, 0.45),
                                    xytext=(-5, 0),
                                    textcoords="offset points",
                                    rotation=-90,
                                    color=cmap(col_idx),
                                    bbox=dict(boxstyle='round,pad=0.3',
                                              facecolor='white',
                                              edgecolor='white'))

                        # # Plot positive micromotion frequency sidebands
                        freqs_plot = np.linspace(
                            value + micromotion_freq - 20 * omeg_sb,
                            value + micromotion_freq + 20 * omeg_sb, 1000)
                        ax.plot(freqs_plot,
                                sin_squared(freqs_plot,
                                            value + micromotion_freq,
                                            pitime * eta),
                                color=cmap(col_idx),
                                linestyle=':',
                                label=label_name)
                        ax.annotate(r'$|$' +
                                    self.plotting_names_D52[23 - row_idx] +
                                    r'$\rangle$',
                                    xy=(value + micromotion_freq, 0.45),
                                    xytext=(-5, 0),
                                    textcoords="offset points",
                                    rotation=-90,
                                    color=cmap(col_idx),
                                    bbox=dict(boxstyle='round,pad=0.3',
                                              facecolor='white',
                                              edgecolor='white'))

                        # Plot negative micromotion frequency sidebands
                        freqs_plot = np.linspace(
                            value - micromotion_freq - 20 * omeg_sb,
                            value - micromotion_freq + 20 * omeg_sb, 1000)
                        ax.plot(freqs_plot,
                                sin_squared(freqs_plot,
                                            value - micromotion_freq,
                                            pitime * eta),
                                color=cmap(col_idx),
                                linestyle=':',
                                label=label_name)
                        ax.annotate(r'$|$' +
                                    self.plotting_names_D52[23 - row_idx] +
                                    r'$\rangle$',
                                    xy=(value - micromotion_freq, 0.45),
                                    xytext=(-5, 0),
                                    textcoords="offset points",
                                    rotation=-90,
                                    color=cmap(col_idx),
                                    bbox=dict(boxstyle='round,pad=0.3',
                                              facecolor='white',
                                              edgecolor='white'))
                    else:
                        pass

            # Add a concise legend
            handles, labels = ax.get_legend_handles_labels()
            # Get unique labels
            unique_labels = list(set(labels))
            unique_handles = [
                handles[labels.index(label)] for label in unique_labels
            ]  # Corresponding handles
            ax.legend(unique_handles, unique_labels)

            plt.xlabel("Transition Frequency (MHz)")
            if include_sidebands:
                plt.title("Plot of Transitions with Sidebands")
            else:
                plt.title("Plot of Transitions (Carrier Only)")
            plt.grid(True)

        return None

    def find_references_from_optical_pumping_transitions(
            self, freq_1: float = 609.993851, freq_2: float = 604.167711):
        '''When the magnetic field or the 1762 cavity drift has fallen far out
        of calibration, NBOP initialisation may fail and finding transitions
        will be difficult. To quickly the new magnetic field value or to just
        solve for [0,2,0] and [-1,4,-3] transitions, we measure two transitions
        that can be initialised with optical pumping to m=2 state in S1/2.

        This uses the calibration data we have and the relation

        f_t - f_int = m_i * (f_sens - f_int) + b_i

        where f_t is a target transition,
        f_int is [0,2,0] (intercept setter in calibration)
        f_sens is [-1,4,-3] (sensitive slope setter in calibration)
        m_i, b_i are slope and intercept (an1 and an2) for target transition

        '''

        # load an1 and an2 tables for calculating frequencies
        path = ('ramseys_calibrations/')

        self.an1_table = np.loadtxt(path + 'an1_ramsey_freq_table.txt',
                                    delimiter=',')
        self.an2_table = np.loadtxt(path + 'an2_ramsey_freq_table.txt',
                                    delimiter=',')
        # an1_freq1 = self.an1_table[17, 4]
        # an1_freq2 = self.an1_table[16, 4]
        # an2_freq1 = self.an2_table[17, 4]
        # an2_freq2 = self.an2_table[16, 4]
        # an1_freq1 = self.an1_table[7, 0]
        # an1_freq2 = self.an1_table[23, 0]
        # an2_freq1 = self.an2_table[7, 0]
        # an2_freq2 = self.an2_table[23, 0]

        an1_freq1 = self.an1_table[17, 4]
        an1_freq2 = self.an1_table[15, 4]
        an2_freq1 = self.an2_table[17, 4]
        an2_freq2 = self.an2_table[15, 4]

        a = 1 - an1_freq1
        b = an1_freq1
        c = 1 - an1_freq2
        d = an1_freq2

        e1 = freq_1 - an2_freq1
        e2 = freq_2 - an2_freq2

        matrix = np.array([[a, b], [c, d]])
        inverse = np.linalg.inv(matrix)

        result = inverse @ np.array([e1, e2])

        # return the result in the order [0,2,0] then [-1,4,-3]
        print('Reference frequencies found:', result)

        return result

    def predicted_frequencies_validity(self):

        # load an1 and an2 tables for calculating frequencies
        path = ('ramseys_calibrations/')

        # we'll need sensitivities later
        if hasattr(self, 'transition_sensitivities'):
            pass
        else:
            self.generate_transition_sensitivities()

        # use 'an1_freq_table.txt' for old freq fitting tables
        self.an1_table = np.loadtxt(path + 'an1_ramsey_freq_table.txt',
                                    delimiter=',')
        self.an2_table = np.loadtxt(path + 'an2_ramsey_freq_table.txt',
                                    delimiter=',')
        # get the new frequency files from Ramsey calibrations
        folder_path_root = (
            '../data/ramseys_calibrations_data/new_calibrations_data')
        filename_root = 'ramsey_calibration_triplet_'
        filenames = [
            f for f in os.listdir(folder_path_root)
            if f.startswith(filename_root)
        ]
        filenames = sorted(
            filenames,
            key=lambda t: os.stat(os.path.join(folder_path_root, t)).st_mtime)
        # if data_cutoff_date:
        #     cutoff_date = datetime.strptime(data_cutoff_date, "%Y%m%d_%H%M")

        #     filenames = [
        #         f for f in filenames if datetime.fromtimestamp(
        #             os.path.getmtime(os.path.join(folder_path_root, f))) >
        #         cutoff_date
        #     ]
        #     filenames = sorted(filenames,
        #                        key=lambda t: os.stat(
        #                            os.path.join(
        #                                   folder_path_root, t)).st_mtime)

        Ramsey_freq_triplets = np.empty((3, len(filenames)))
        Ramsey_state_triplets = np.empty((3, 3, len(filenames)))

        for idx, filename in enumerate(filenames):
            with open(folder_path_root + '/' + filename, 'r') as file:
                # Read the first line (assuming it's a list of floats)
                first_line = file.readline()
                fl_vals = re.findall(r"[-+]?\d*\.\d+|\d+", first_line)
                Ramsey_freq_triplets[:, idx] = [
                    fl_vals[0], fl_vals[1], fl_vals[2]
                ]

                # Read the second line (assuming it's a nested list of ints)
                second_line = file.readline()
                sl_values = re.findall(r"[-+]?\d+", second_line)
                Ramsey_state_triplets[0, :, idx] = [
                    sl_values[0], sl_values[1], sl_values[2]
                ]
                Ramsey_state_triplets[1, :, idx] = [
                    sl_values[3], sl_values[4], sl_values[5]
                ]
                Ramsey_state_triplets[2, :, idx] = [
                    sl_values[6], sl_values[7], sl_values[8]
                ]

        Ramsey_freq_triplets = np.transpose(Ramsey_freq_triplets)
        Ramsey_state_triplets = np.transpose(Ramsey_state_triplets)
        Ramsey_state_triplets = Ramsey_state_triplets[:, :, -1]
        Ramsey_state_triplets_unique = np.unique(Ramsey_state_triplets, axis=0)
        helpers = [1, 5, 11, 19]

        all_differences = []
        all_abs_differences = []
        all_sensitivities = []

        fig, ax = plt.subplots(1, 1, figsize=set_size('full'))
        for idx_outer, trans in enumerate(Ramsey_state_triplets_unique):
            trans_differences = []
            # print('\nstarting trans', trans)
            # print(idx_outer)

            for idx, trip in enumerate(Ramsey_freq_triplets):
                if np.all(trans == Ramsey_state_triplets[idx]):
                    # transition_triplet_indices = [
                    #     int(Ramsey_state_triplets[idx, 0]),
                    #     int(Ramsey_state_triplets[idx, 1]),
                    #     int(Ramsey_state_triplets[idx, 2])
                    # ]

                    transition_table_index = [
                        int(helpers[int(trans[1] - 1)] - trans[2]),
                        int(trans[0] + 2)
                    ]

                    run_trans, run_pitimes = (
                        self.run_calibrated_frequency_generation(
                            insensB=trip[0], sensB=trip[1]))

                    predicted_freq = run_trans[transition_table_index[0],
                                               transition_table_index[1]]

                    # predicted_freq = self.an1_table[
                    #     transition_table_index[0],
                    #     transition_table_index[1]] * (
                    #         trip[1] - trip[0]) + trip[0] + self.an2_table[
                    #             transition_table_index[0],
                    #             transition_table_index[1]]

                    # print(trip)
                    # print(predicted_freq)
                    # print(trans)
                    # print(1e3 * np.abs(predicted_freq - trip[2]))
                    trans_differences.append(1e6 * (predicted_freq - trip[2]))
                else:
                    pass
            all_differences.append(np.mean(trans_differences))
            all_abs_differences.append(np.mean(np.abs(trans_differences)))
            all_sensitivities.append(
                np.abs(
                    self.transition_sensitivities[transition_table_index[0],
                                                  transition_table_index[1]]))

            # plot the discrepancies for each individual transition
            # print('Differences', trans_differences)
            print('\nTransition:', trans)
            print('Avg', np.mean(trans_differences))
            print('Avg absolute value', np.mean(np.abs(trans_differences)))
            ax.scatter(
                idx_outer * 10 + np.arange(0, len(trans_differences), 1),
                trans_differences,
                label=f'({int(trans[0])}, {int(trans[1])}, {int(trans[2])})')
            ax.set_title(r'Calibration Discrepancies ' +
                         '(per Transition) for New Calibration Data')
            ax.set_ylabel('Discrepancy / Hz')
            ax.set_xlabel('Measurement Index')

            plt.grid()
            plt.legend()

        # print(all_differences)
        # print(all_sensitivities)
        # print(Ramsey_state_triplets_unique)

        print('\n\nAverage of all differences', np.mean(all_differences), '\n')
        print('Average of all absolute differences',
              np.mean(all_abs_differences), '\n')
        print('Number of unique transitions calibrated',
              len(Ramsey_state_triplets_unique), '\n')
        print('They are: \n', Ramsey_state_triplets_unique, '\n')

        fig, ax = plt.subplots(1, 1, figsize=set_size('full'))

        ax.scatter(all_sensitivities, all_abs_differences)

        for (x_index, (x_i, x_j,
                       x_k)), y in zip(enumerate(Ramsey_state_triplets_unique),
                                       all_abs_differences):

            ax.text(all_sensitivities[x_index],
                    y,
                    f'{(int(x_i),int(x_j),int(x_k))}',
                    fontsize=8,
                    ha='right')

        ax.set_ylabel('Discrepancy / Hz')
        ax.set_xlabel('Transition Sensitivity / MHz/G')
        ax.set_title(r'Average Calibration Discrepancies ' +
                     '(per Transition) for New Calibration Data')
        ax.grid()
        plt.show()
        breakpoint()

    def dummy_sine_wave_plot(self):
        """Generates a dummy sine wave plot used for visualization of Rabi
        calibration scheme (for thesis figure).

        Returns:
            None

        """

        x = np.linspace(0, 2 * np.pi, 1000)
        fig, ax = plt.subplots(1, 1, figsize=set_size('half'))
        ax.plot(x, (1 - np.cos(x)) / 2, label='Resonant')
        ax.plot(x, (1 - np.cos(x - np.pi / 3)) / 2, label='Detuned')
        ax.set_ylabel('Bright Probabilility')
        ax.set_xlabel(r'Phase ($\phi$)')
        ax.legend()
        ax.grid()
        ax.vlines(np.pi / 2, 0, 1, color='k', linestyle='--')
        ax.vlines(3 * np.pi / 2, 0, 1, color='k', linestyle='--')
        # Define the positions of the ticks
        tick_positions = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]

        # Define the labels for the ticks
        tick_labels = [
            '0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'
        ]

        # Set the x-ticks
        ax.set_xticks(tick_positions, tick_labels)


if __name__ == '__main__':
    start_time = time.time()
    cal = Calibrations(F2M0f2m0=545.685316)  # don't change this 545 frequency

    # cal.get_sim_freqs()
    # cal.fit_freq_Rabi_fringe(create_plots=True, plot_individual_scans=False)
    # cal.fit_Rabi_oscillation(create_plots=True)
    # # print(cal.chunked_datetimes)
    # # print(cal.chunked_pitime_fits)

    # cal.an1_an2_finder(create_freq_plots=False)
    # cal.an1_pitime_finder(save_pitime_file=True)
    # cal.run_calibrated_frequency_generation(
    #     insensB=545.684110,
    #     sensB=623.488399,
    #     ref_pitimes=[21.897, 41.031, 45.832, 35.6, 43.23],
    #     print_result=True)

    # cal.compare_experiment_calibration()
    cal.plot_predicted_measured_freqs_strengths()

    # cal.Ramsey_an2_finder()
    # cal.Ramsey_an2_finder(compare_old_calibration=False,
    #                       plot_individual_linears=False)
    # cal.mag_field_freqs_monitor(data_cutoff_date='20240802_2200',
    #                             Ramsey_wait_time=400,
    #                             shots=250)

    # cal.pitimes_monitor(data_cutoff_date='20241001_0000')
    # cal.predicted_frequencies_validity()

    # cal.show_individual_transitions(transition=621.46)
    # cal.show_individual_transitions(transition=616.925)
    # cal.show_individual_transitions(transition=599.06)

    # cal.dummy_sine_wave_plot() # for Ramsey calibration explainer plot

    # cal.frequency_jungle_with_Rabi_fringes(B_field=4.209,
    #                                        include_sidebands=True,
    #                                        decohering_pulse=False)

    # [freq_545, freq_623
    #  ] = cal.find_references_from_optical_pumping_transitions(freq_1=610.131,
    #                                                           freq_2=597.417)

    # for bf in np.linspace(4.21, 4.22, 10):
    # # for bf in np.linspace(3.4, 3.5, 10):
    #     cal.B_field = bf
    #     cal.generate_transition_sensitivities()
    #     cal.generate_transition_frequencies()
    #     print(bf)
    #     # print(cal.transition_frequencies[23,0])
    #     # print(cal.transition_sensitivities[5,2])
    #     print('623-545',
    #           (cal.transition_frequencies[22, 1] -
    #            cal.transition_frequencies[5, 2]) - (freq_623 - freq_545))
    #     print('244-242', (cal.transition_frequencies[17, 4] -
    #                       cal.transition_frequencies[15, 4]))

    plt.show()
    print("--- %s seconds ---" % (time.time() - start_time))
