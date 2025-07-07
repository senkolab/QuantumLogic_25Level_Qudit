import os
import sys
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import scipy as sc
from scipy.optimize import Bounds, curve_fit
from scipy.signal import argrelextrema
from tqdm import tqdm

from class_barium import Barium
from plot_ramsey import plot_ramsey
from plot_utils import nice_fonts, set_size

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


class Ramsey(Barium):

    def __init__(
            self,
            initial_state: list = [[0, 2, 0]],
            initial_manifold: str = 'D52',
            ref_pitimes: list = [28.6598, 56.9475, 60.9618, 46.0007, 53.8691],
            calculate_new_pi_times=True,
            measured_states: list = [2, 19, 23],
            periodicity: float = 0.1,
            wait_time: float = 0.001,
            sim_timesteps: float = 1e4,
            detunings=np.random.uniform(-2 * np.pi * 0.4,
                                        2 * np.pi * 0.4,
                                        size=(24, 5)),  # in kHz
            detuning_paths=[],
            detuning_givens=[],
            pi_time_error=np.random.uniform(-0.1, 0.1, size=(24, 5)),  # in %
            pulse_train: list = [],
            fractions: list = [],
            fixed_phase_mask: list = [],
            simulated_phase_mask: list = [],
            real_wait_times: bool = False,
            experimental_data_file=None,
            exp_number=101,
            noise_level: float = 0.0):
        super().__init__()

        self.dimension = 29  # system size = always 29 for S1/2, F=2 + D5/2
        self.initial_manifold = initial_manifold
        self.measured_states = measured_states
        self.ref_pitimes = ref_pitimes

        self.noise_level = noise_level
        self.wait_time = wait_time
        self.sim_timesteps = int(sim_timesteps)
        self.periodicity = periodicity

        self.pulse_train = np.array(pulse_train)
        self.fractions = np.array(fractions)
        self.fixed_phase_mask = np.array(fixed_phase_mask)
        self.simulated_phase_mask = np.array(simulated_phase_mask)
        self.real_wait_times = real_wait_times
        self.phase_times = np.linspace(0, 1.1 * self.periodicity, 101)

        self.experimental_data_file = experimental_data_file
        self.exp_number = exp_number

        basis_states = []
        for i in range(self.dimension):
            basis_states.append(qt.basis(self.dimension, i))
        self.basis_states = basis_states

        # Calculate pi-times, or load from txt
        # run below commented lines to re-gen the pi-times array
        if calculate_new_pi_times:
            _, pi_times = self.generate_transition_strengths()
            pi_times = np.flip(pi_times * 1e-3, axis=0)
            np.savetxt('ramseys_qudits/pi_times_saved.txt', pi_times, delimiter=',')
        elif not calculate_new_pi_times:
            pi_times = np.loadtxt('ramseys_qudits/pi_times_saved.txt', delimiter=',')

        self.Rabi_freqs = np.pi / pi_times
        # print(pi_times)
        # print(self.Rabi_freqs)
        self.detunings = detunings
        self.detuning_givens = detuning_givens
        self.pi_times = pi_times + pi_times * pi_time_error

        # print('pi_times\n', self.pi_times)
        # print('Rabi freqs\n', self.Rabi_freqs)
        # print('detunings\n', self.detunings)

        ##################################################################
        initial_state = initial_state[0]
        translated_init_state = self.helpers[initial_state[1] -
                                             1] - initial_state[2]

        if initial_manifold == 'S12':
            self.init_state = qt.basis(self.dimension, initial_state[0] + 2)
        elif initial_manifold == 'D52':
            self.init_state = qt.basis(self.dimension,
                                       23 - translated_init_state + 5)

        ket_tuples = self.translate_transitions_to_kets(pulse_train)

        self.trans_tuples = self.translate_kets_to_indices(ket_tuples)
        # print(ket_tuples)
        # print(self.trans_tuples)

        for idx, trans in enumerate(self.trans_tuples):
            if np.isnan(trans[0]):
                break
            else:
                print(
                    f'pi-time for {pulse_train[idx]} is ' +
                    f'{np.round(1e3*self.pi_times[trans[1],trans[0]],3)} us.')

        if detuning_givens:
            self.generate_detunings(detuning_givens)
            self.generate_detuning_paths(detuning_paths)

        # generate the state labels for plotting
        state_labels = []
        for state in self.measured_states:
            if state <= 4:
                state_labels.append(r'|$S_{1/2}$, ' +
                                    f'F=2, m={self.F_values_S12[3+state][1]}' +
                                    r'$\rangle$')
            elif state > 4:
                state_labels.append(r'|$D_{5/2}$, ' +
                                    f'F={self.F_values_D52[state-5][0]}, ' +
                                    f'm={self.F_values_D52[state-5][1]}' +
                                    r'$\rangle$')
        self.state_labels = state_labels

    def translate_transitions_to_kets(self, transitions):
        ket_tuples = []
        for transition in transitions:
            if np.isnan(transition[0]):
                ket_tuples.append([np.nan, np.nan])
            else:
                ket_tup1 = transition[0] + 26
                ket_tup2 = self.helpers[int(transition[1]) - 1] - transition[2]

                ket_tuple = [ket_tup1, ket_tup2]

                ket_tuples.append(ket_tuple)
        return ket_tuples

    def translate_kets_to_indices(self, ket_tuples):
        trans_tuples = []
        for ket_tuple in ket_tuples:
            if np.isnan(ket_tuple[0]):
                trans_tuples.append([np.nan, np.nan])
            elif ket_tuple[0] > ket_tuple[1]:
                trans_tuples.append([ket_tuple[0] - 24, 23 - ket_tuple[1]])
            elif ket_tuple[1] > ket_tuple[0]:
                trans_tuples.append([ket_tuple[1] - 24, 23 - ket_tuple[0]])
        return trans_tuples

    def generate_detunings(self, detuning_givens):
        for idx, trans in enumerate(detuning_givens[0]):
            # print('detuning triplet', trans)
            ket_detun = self.translate_transitions_to_kets([trans])
            index_detun = self.translate_kets_to_indices(ket_detun)
            # print('det index', index_detun)
            self.detunings[
                index_detun[0][1],
                index_detun[0][0]] = 2 * np.pi * detuning_givens[1][idx]

    def generate_detuning_paths(self, detuning_paths):
        detuning_kets = []
        detuning_indices = []
        for path in detuning_paths:
            kets_translated = self.translate_transitions_to_kets(path)
            indices_translated = self.translate_kets_to_indices(
                kets_translated)
            detuning_kets.append(kets_translated)
            detuning_indices.append(indices_translated)
        self.detuning_kets = detuning_kets
        self.detuning_indices = detuning_indices
        # print('det kets', self.detuning_kets)
        # print('det indices', self.detuning_indices)

    def ham(self, phase: float, transition: tuple):
        # make a dummy Qobj for the hamiltonian as a starting point
        ham = 0 * qt.Qobj(self.basis_states[0] * self.basis_states[0].dag())

        if hasattr(self, 'detuning_kets'):
            for idx, path in enumerate(self.detuning_kets):
                # print('indices in detuning table',
                #       self.detuning_indices[idx][0][1],
                #       self.detuning_indices[idx][0][0])
                # print(
                #     'detuning there',
                #     self.detunings[self.detuning_indices[idx][0][1],
                #                    self.detuning_indices[idx][0][0]])
                # print('basis state for detuning',
                #       self.basis_states[28 - path[0][1]])

                if self.initial_manifold == 'S12' and len(path) == 1:
                    ham += self.detunings[
                        self.detuning_indices[idx][0][1],
                        self.detuning_indices[idx][0][0]] * qt.Qobj(
                            self.basis_states[28 - path[0][1]] *
                            self.basis_states[28 - path[0][1]].dag())

                elif self.initial_manifold == 'D52' and len(path) == 1:
                    ham += self.detunings[
                        self.detuning_indices[idx][0][1],
                        self.detuning_indices[idx][0][0]] * qt.Qobj(
                            self.basis_states[28 - path[0][0]] *
                            self.basis_states[28 - path[0][0]].dag())

                else:
                    if path[-1][0] == path[-2][0]:
                        for idx2, trans in enumerate(path):
                            ham += self.detunings[
                                self.detuning_indices[idx][idx2][1],
                                self.detuning_indices[idx][idx2][0]] * qt.Qobj(
                                    self.basis_states[28 - path[-1][1]] *
                                    self.basis_states[28 - path[-1][1]].dag())
                    elif path[-1][1] == path[-2][1]:
                        for idx2, trans in enumerate(path):
                            ham += self.detunings[
                                self.detuning_indices[idx][idx2][1],
                                self.detuning_indices[idx][idx2][0]] * qt.Qobj(
                                    self.basis_states[28 - path[-1][0]] *
                                    self.basis_states[28 - path[-1][0]].dag())

        else:
            pass

        if np.isnan(transition[0]):
            return ham
        else:
            exp_phase = 1j * phase
            ham += self.Rabi_freqs[
                int(transition[1]), int(transition[0])] / 2 * qt.Qobj(
                    np.exp(exp_phase) * self.basis_states[int(transition[0])] *
                    self.basis_states[int(transition[1]) + 5].dag() +
                    np.exp(-exp_phase) *
                    self.basis_states[int(transition[1]) + 5] *
                    self.basis_states[int(transition[0])].dag())
            return ham

    def find_pulse_times(self):

        pulse_times = []
        for idx, fraction in enumerate(self.fractions):
            if np.isnan(fraction):
                pulse_times.append(self.wait_time)
            else:
                pulse_times.append(
                    ((2 * self.pi_times[int(self.trans_tuples[idx][1]),
                                        int(self.trans_tuples[idx][0])]) /
                     np.pi) * np.arcsin(np.sqrt(fraction)))
        pulse_times = np.array(pulse_times)
        return pulse_times

    def pulse_mask(self, times, pulse_times, pulse_number):
        if pulse_number == 1:
            output1 = times >= 0
        else:
            output1 = times >= sum(pulse_times[:pulse_number - 1])
        if pulse_number == len(pulse_times):
            output2 = times <= sum(pulse_times)
        else:
            output2 = times <= (sum(pulse_times) -
                                sum(pulse_times[pulse_number:]))
        output = output1 * output2
        return output.astype(int)

    def get_full_Hamiltonian(self, times, phases):
        H_tot = []
        pulse_times = self.find_pulse_times()
        for idx, trans_tup in enumerate(self.trans_tuples):
            H_tot.append([
                self.ham(phases[idx], trans_tup),
                self.pulse_mask(times, pulse_times, idx + 1)
            ])
        return H_tot

    def find_collapse_operators(self):
        c_ops = []
        for i in range(self.dimension):
            for j in range(self.dimension):
                if i < j:
                    c_ops.append(self.noise_level *
                                 (qt.Qobj(self.basis_states[i] *
                                          self.basis_states[j].dag()) +
                                  qt.Qobj(self.basis_states[j] *
                                          self.basis_states[i].dag())))
                    c_ops.append(
                        qt.Qobj(
                            self.noise_level *
                            (np.exp(1j * np.pi / 2) * self.basis_states[i] *
                             self.basis_states[j].dag() +
                             np.exp(-1j * np.pi / 2) * self.basis_states[j] *
                             self.basis_states[i].dag())))
        return c_ops

    def plot_pulse_evolution(self):

        pulse_times = self.find_pulse_times()
        print('The total pulse time for the Ramsey pulse is ' +
              f'{np.round(1e3*(sum(pulse_times)-self.wait_time),3)} us. \n')

        times = np.linspace(0, sum(pulse_times),
                            self.sim_timesteps)  # times in ms

        phases = self.fixed_phase_mask * np.pi

        H_tot = self.get_full_Hamiltonian(times, phases)

        # Solve the dynamics
        if self.noise_level != 0:
            c_ops = self.find_collapse_operators()

            # Solve the dynamics
            result = qt.mesolve(H_tot,
                                self.init_state,
                                times,
                                c_ops=c_ops,
                                args={})
        else:
            # Solve the dynamics
            result = qt.mesolve(H_tot, self.init_state, times, args={})
        probabilities = np.zeros((self.dimension, len(times)))
        for i in range(self.dimension):
            probabilities[i] = qt.expect(
                qt.ket2dm(qt.basis(self.dimension, i)), result.states)

        # Plot the probabilities
        fig, ax = plt.subplots(figsize=(8, 6))

        for idx, state in enumerate(self.measured_states):
            ax.plot(times,
                    probabilities[state],
                    label=fr'$|${idx}$\rangle \equiv$' +
                    self.state_labels[idx],
                    linestyle='-',
                    color=colors[idx % len(colors)])

        ax.set_xlabel(r'Time ($ms$)')
        ax.set_ylabel('Probability')
        ax.set_title('Ramsey Pulse - Measurement Probability vs. Time')
        if len(self.measured_states) > 5:
            ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
        else:
            ax.legend(loc='upper center')
        ax.grid()
        plt.savefig('ramseys_qudits/latest_pulse_evolution.png',
                    bbox_inches='tight')

    def plot_D2_data_and_fit(self):

        wait_times, full_data_array = plot_ramsey(self.experimental_data_file,
                                                  plotting=False)
        lower_error, upper_error = self.find_errors(1, full_data_array,
                                                    self.exp_number)
        asym_error_bar = [
            full_data_array - lower_error, upper_error - full_data_array
        ]
        asym_error_bar = np.absolute(asym_error_bar)

        def oscfunc(t, A, w, p, c, d):
            return A * np.cos(w * t + p) * np.exp(-t / d) + c

        fig, ax = plt.subplots(figsize=(8, 6))
        p_guesses = [0, np.pi]
        for i in range(np.shape(full_data_array)[0]):
            yerr = [asym_error_bar[0][i], asym_error_bar[1][i]]
            ax.errorbar(wait_times,
                        full_data_array[i, :],
                        fmt='o-',
                        yerr=yerr,
                        color=colors[i % len(colors)])

            A_guess = np.max(full_data_array[i, :]) - np.mean(
                full_data_array[i, :])
            ff = np.fft.fftfreq(len(wait_times),
                                (wait_times[1] - wait_times[0]))
            F_PD = abs(np.fft.fft(full_data_array[i, :]))
            guess_freq = abs(ff[np.argmax(F_PD[1:]) + 1])
            w_guess = 2.0 * np.pi * guess_freq
            p_guess = p_guesses[i]
            c_guess = np.mean(full_data_array[i, :])
            d_guess = 6e2  # us
            p0 = np.array([A_guess, w_guess, p_guess, c_guess, d_guess])
            popt, pcov = sc.optimize.curve_fit(oscfunc,
                                               wait_times,
                                               full_data_array[i, :],
                                               p0=p0,
                                               maxfev=10000)
            if i == 0:
                print('Fitted frequency:',
                      np.round(1e3 * popt[1] / 2 / np.pi, 3), ' kHz.')
                print('Fitted periodicity:',
                      np.round(1 / (1 * popt[1] / 2 / np.pi), 3), 'us.')
                print('Fitted decay time:', np.round(popt[4], 1), ' us.')
                print('Small number here means offset is fine:',
                      np.round(np.absolute(2 * (popt[3] - 0.5)), 3), '.\n')
            x_vals = np.linspace(0, wait_times[-1], 1000)
            # ax.plot(x_vals, oscfunc(x_vals, *p0), 'k-')
            ax.plot(x_vals,
                    oscfunc(x_vals, *popt),
                    color=colors[i % len(colors)],
                    label=fr'$|${i}$\rangle \equiv$' + self.state_labels[i],
                    linestyle='--',
                    alpha=0.6)
        ax.grid()
        ax.legend(title=rf'$\tau_d={np.round(popt[4],2)}$ $\mu s$.')
        ax.set_xlabel(r'Time ($\mu s$)')
        ax.set_ylabel('Probability')
        ax.set_title(r'Ramsey Coherence Probe ($T_2^*$)')

    def extract_coherence_time(self):

        # define empty array to fill in peak heights for exponential fit
        peak_maxima = []
        peak_wait_times = []

        wait_times, full_data_array = plot_ramsey(self.experimental_data_file,
                                                  plotting=False)

        # append the first point as the first maximum
        peak_maxima.append(full_data_array[0][0])
        peak_wait_times.append(wait_times[0])

        lower_error, upper_error = self.find_errors(1, full_data_array,
                                                    self.exp_number)
        asym_error_bar = [
            full_data_array - lower_error, upper_error - full_data_array
        ]
        asym_error_bar = np.absolute(asym_error_bar)

        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(np.shape(full_data_array)[0]):
            yerr = [asym_error_bar[0][i], asym_error_bar[1][i]]
            ax.errorbar(wait_times,
                        full_data_array[i, :],
                        fmt='o-',
                        yerr=yerr,
                        color=colors[i % len(colors)])
        ax.grid()
        # ax.legend(title=rf'$\tau_d={np.round(popt[4],2)}$ $\mu s$.')
        ax.set_xlabel(r'Time ($\mu s$)')
        ax.set_ylabel('Probability')
        ax.set_title(r'Ramsey Coherence Probe ($T_2^*$)')

        ket0_population = full_data_array[0, :]
        # find the local maxima
        maxima_indices = argrelextrema(ket0_population, np.greater)
        # local_maxima_values = ket0_population[maxima_indices[0]]

        # Filter out maxima with y value below 0.3
        filtered_maxima_indices = maxima_indices[0][ket0_population[
            maxima_indices[0]] > 0.57]
        filtered_maxima_values = ket0_population[filtered_maxima_indices]
        filtered_maxima_times = wait_times[filtered_maxima_indices]

        filtered_maxima_indices = list(filtered_maxima_indices)
        filtered_maxima_values = list(filtered_maxima_values)
        filtered_maxima_times = list(filtered_maxima_times)

        # Exclude peaks too close together
        popper_indices = []
        for idx, time in enumerate(filtered_maxima_times[:-1]):
            if np.absolute(filtered_maxima_times[idx + 1] -
                           filtered_maxima_times[idx]) < 30:
                if filtered_maxima_values[idx] < filtered_maxima_values[idx +
                                                                        1]:
                    popper_indices.append(idx)
                elif filtered_maxima_values[idx] > filtered_maxima_values[idx +
                                                                          1]:
                    popper_indices.append(idx + 1)

        for idx in sorted(popper_indices, reverse=True):
            print(idx)
            filtered_maxima_values.pop(idx)
            filtered_maxima_times.pop(idx)
            filtered_maxima_indices.pop(idx)

        # Fit a quadratic to each local maximum
        def quadratic(x, a, b, c):
            return a * x**2 + b * x + c

        for idx in filtered_maxima_indices:
            # Extract 5 data points centered around the local maximum
            window = np.arange(max(0, idx - 2),
                               min(len(ket0_population), idx + 3))

            window_populations = ket0_population[window]
            print(window_populations)
            window_times = wait_times[window]
            print(window_times)

            # Fit the quadratic
            params, _ = curve_fit(quadratic, window_times, window_populations)

            x_fit = np.linspace(window_times[0], window_times[-1], 100)
            y_fit = quadratic(x_fit, *params)

            max_value = np.max(y_fit)
            max_wait_time = x_fit[np.argwhere(y_fit == max_value)]
            peak_maxima.append(max_value)
            peak_wait_times.append(max_wait_time[0][0])

            # print(f"Local maximum at index {idx}: " +
            #       f"Fitted max value = {max_value:.4f}," +
            #       f" at wait time {max_wait_time[0][0]:.3f}.")

            ax.plot(x_fit,
                    y_fit,
                    label=f"Max {idx} ({max_value:.3f})",
                    linestyle='--',
                    color='red')

        def exp_decay(x, A, tau):
            return (1.0 - 1 / len(self.measured_states)) * np.exp(
                -x / tau) + 1 / len(self.measured_states)

        guess_A, guess_tau = 1.0 - 1 / len(self.measured_states), 500
        initial_guess = [guess_A, guess_tau]

        # Fit the data to the exponential decay function
        params, covariance = curve_fit(exp_decay,
                                       peak_wait_times,
                                       peak_maxima,
                                       p0=initial_guess)
        A_fit, tau_fit = params

        fitted_decay = exp_decay(wait_times, A_fit, tau_fit)
        ax.plot(wait_times,
                fitted_decay,
                label='Fitted Decay',
                linestyle='--',
                color='black')
        print(f'The fitted decay here is {np.round(params[1],2)} us.')

        return peak_wait_times, peak_maxima

    def phase_ramsey_helper(self, wait_time):
        probabilities = np.zeros((self.dimension))

        if self.real_wait_times:
            self.wait_time = wait_time
        else:
            pass
        pulse_times = self.find_pulse_times()
        times = np.linspace(0, sum(pulse_times),
                            self.sim_timesteps)  # times in ms
        revival_phases = self.fixed_phase_mask * np.pi + (
            2 * np.pi *
            (self.simulated_phase_mask) * wait_time) / self.periodicity

        H_tot = self.get_full_Hamiltonian(times, revival_phases)

        if self.noise_level != 0:
            c_ops = self.find_collapse_operators()

            # Solve the dynamics
            result = qt.mesolve(H_tot,
                                self.init_state,
                                times,
                                c_ops=c_ops,
                                args={})
        else:
            # Solve the dynamics
            result = qt.mesolve(H_tot, self.init_state, times, args={})

        for i in range(self.dimension):
            probabilities[i] = qt.expect(
                qt.ket2dm(qt.basis(self.dimension, i)), result.states[-1])
        return probabilities

    def plot_phase_ramsey(self, plot_from_saved_data=False, plot=True):

        if plot_from_saved_data:
            probabilities = np.loadtxt('29_lvl_phase_ramsey_simulation.txt',
                                       delimiter=',')
        else:
            probabilities = np.zeros((self.dimension, len(self.phase_times)))

            # Set the number of workers as half the available CPU cores
            num_workers = int(os.cpu_count() / 2)

            # multi-thread the computation
            with Pool(processes=num_workers) as pool:
                results = list(
                    tqdm(pool.imap(self.phase_ramsey_helper, self.phase_times),
                         total=len(self.phase_times),
                         desc='Ramsey Simulation'))
            probabilities = np.transpose(np.array(results))

        phases = 2 * np.pi * self.phase_times / self.periodicity
        np.savetxt('ramseys_qudits/29_lvl_phase_ramsey_simulation.txt',
                   probabilities,
                   delimiter=',')

        fit_probabilities = []
        if plot:
            fig, ax = plt.subplots(figsize=(8, 6))
        if self.experimental_data_file:
            wait_times, full_data_array = plot_ramsey(
                self.experimental_data_file, plotting=False)
            lower_error, upper_error = self.find_errors(
                1, full_data_array, self.exp_number)
            asym_error_bar = [
                full_data_array - lower_error, upper_error - full_data_array
            ]
            asym_error_bar = np.absolute(asym_error_bar)

            exp_phases = 2 * np.pi * wait_times / 100
            for i in range(np.shape(full_data_array)[0]):
                yerr = [asym_error_bar[0][i], asym_error_bar[1][i]]
                if plot:
                    if self.real_wait_times:
                        ax.errorbar(wait_times,
                                    full_data_array[i, :],
                                    fmt='o-',
                                    yerr=yerr,
                                    color=colors[i % len(colors)])
                    else:
                        ax.errorbar(2 * wait_times / 100,
                                    full_data_array[i, :],
                                    fmt='o-',
                                    yerr=yerr,
                                    color=colors[i % len(colors)])

        for idx, state in enumerate(self.measured_states):
            if plot:
                if self.real_wait_times:
                    # convert phase times to physical wait times
                    ax.plot(1e3 * self.phase_times,
                            probabilities[state],
                            label=fr'$|${idx}$\rangle \equiv$' +
                            self.state_labels[idx],
                            linestyle='--',
                            color=colors[idx % len(colors)],
                            alpha=0.6)
                else:
                    ax.plot(2 * self.phase_times / 0.1,
                            probabilities[state],
                            label=fr'$|${idx}$\rangle \equiv$' +
                            self.state_labels[idx],
                            linestyle='--',
                            color=colors[idx % len(colors)],
                            alpha=0.6)
            fit_probabilities.append(probabilities[state])

        if plot:
            if self.real_wait_times:
                ax.set_xlabel(r'Time ($\mu s$)')
            else:
                ax.set_xlabel(r'Phase ($n\phi/\pi$)')
            ax.set_ylabel('Probability')
            ax.set_title('Measurement Probability vs. Phase')
            ax.set_ylim(-0.1, 1.1)
            # ax.vlines(0.5, ymin=0, ymax=1, linestyle='--', color='black')
            # ax.vlines(1.5, ymin=0, ymax=1, linestyle='--', color='black')
            ax.minorticks_on()
            ax.grid()
            if len(self.measured_states) > 5:
                ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
            else:
                ax.legend(loc='upper center')
            plt.savefig('ramseys_qudits/latest_qudit_ramsey.png',
                        bbox_inches='tight')

        fit_probabilities = np.array(fit_probabilities)

        if self.experimental_data_file:
            return fit_probabilities, full_data_array, exp_phases
        else:
            return probabilities, phases

    def fit_detunings(self):

        wait_times, full_data_array = plot_ramsey(self.experimental_data_file,
                                                  plotting=False)
        phases_for_fitting = wait_times / 1000

        global_phase_times = self.phase_times
        self.phase_times = phases_for_fitting

        detuning_givens = self.detuning_givens

        def exp_theory_comparison(detunings):
            for idx, trans in enumerate(detuning_givens[0]):
                # print('detuning triplet', trans)
                ket_detun = self.translate_transitions_to_kets([trans])
                index_detun = self.translate_kets_to_indices(ket_detun)
                # print('det index', index_detun)
                self.detunings[index_detun[0][1],
                               index_detun[0][0]] = 2 * np.pi * detunings[idx]

            (fit_probabilities, exp_probabilities,
             exp_phases) = self.plot_phase_ramsey(plot=False)

            opt_param = np.sum(np.square(fit_probabilities -
                                         exp_probabilities))
            print('Current Optimisation Parameter Value:', opt_param)

            return opt_param

        def callback(xk):
            print(f'Current guess: {xk}')

        initial_guess = self.detuning_givens[1]
        print(f'Initial Guess: {initial_guess}')
        # Define the same bound for each entry
        lower_bound = -3
        upper_bound = 3

        # Create bounds for all parameters
        bounds = Bounds([lower_bound] * len(initial_guess),
                        [upper_bound] * len(initial_guess))

        print('Starting fit for detunings...')
        optimisation_params = sc.optimize.minimize(
            exp_theory_comparison,
            [initial_guess],
            method='Nelder-Mead',
            bounds=bounds,
            # options={'maxiter': 1},  # for testing
            callback=callback)

        found_detunings = optimisation_params.x
        print(f'\n The fitted detunings are: {found_detunings} kHz. \n')

        self.detuning_givens[1] = found_detunings
        self.phase_times = global_phase_times
        return found_detunings


if __name__ == '__main__':
    start_time = time.time()

    # keep these up here for easy toggling off of detunings
    detuning_paths = []
    detuning_givens = []
    ref_pitimes = [28.6598, 56.9475, 60.9618, 46.0007, 53.8691]
    ############################################
    # # D=2 pulse sequence
    ############################################
    # NON-BUSSED
    initial_state = [[-1, 4, -3]]
    initial_manifold = 'S12'
    ref_pitimes = [23.699, 46.327, 50.277, 37.142, 46.798]
    measured_states = [1, 6]
    exp_number = 100
    pulse_train = [[-1, 4, -3], [np.nan, np.nan, np.nan], [-1, 4, -3]]

    fractions = [1 / 2, np.nan, 1 / 2]
    fixed_phase_mask = [0, 0, 1]
    simulated_phase_mask = [0, 0, 1]

    # exp_data_file = 'Ramsey_experiment_Dimension_3_202407??_????.txt'
    detuning_paths = [[[-1, 4, -3]]]
    detuning_givens = [[[-1, 4, -3]], [-20.0]]
    ############################################
    initial_state = [[0, 2, 0]]
    initial_manifold = 'S12'
    ref_pitimes = [23.699, 46.327, 50.277, 37.142, 46.798]
    measured_states = [2, 23]
    exp_number = 100
    pulse_train = [[0, 2, 0], [np.nan, np.nan, np.nan], [0, 2, 0]]

    fractions = [1 / 2, np.nan, 1 / 2]
    fixed_phase_mask = [0, 0, 1]
    simulated_phase_mask = [0, 0, 1]

    # exp_data_file = 'Ramsey_experiment_Dimension_3_202407??_????.txt'
    detuning_paths = [[[0, 2, 0]]]
    # detuning_givens = [[[0, 2, 0]], [0.6]]
    ############################################
    # # D=3 pulse sequence
    ############################################
    # NON-BUSSED
    initial_state = [[0, 3, 0]]
    initial_manifold = 'S12'
    measured_states = [2, 23, 17]
    pulse_train = [[0, 2, 0], [0, 3, 0], [np.nan, np.nan, np.nan], [0, 3, 0],
                   [0, 2, 0]]

    fractions = [1 / 3, 1 / 2, np.nan, 1 / 2, 1 / 3]
    fixed_phase_mask = [0, 0, 0, 1, 1]
    simulated_phase_mask = [0, 0, 0, 2, 1]

    # # all these below used [0,2,0] and [0,3,2]
    # exp_data_file = 'Ramsey_experiment_Dimension_3_20240715_1038.txt'
    # exp_data_file = 'Ramsey_experiment_Dimension_3_20240715_1104.txt'
    # exp_data_file = 'Ramsey_experiment_Dimension_3_20240715_1543.txt'
    # # a 3kHz detuning set on [0,2,0] transition now:
    # exp_data_file = 'Ramsey_experiment_Dimension_3_20240715_1602.txt'
    # detuning_paths = [[[0, 3, 0]], [[0, 2, 0]]]
    # detuning_givens = [[[0, 2, 0], [0, 3, 0]], [3.0, 0.0]]

    # now using [0,3,0] and [0,2,0]
    exp_data_file = 'Ramsey_experiment_Dimension_3_20240715_1649.txt'
    # a 2kHz detuning set on [0,3,0] transition now:
    exp_data_file = 'Ramsey_experiment_Dimension_3_20240715_1704.txt'
    detuning_paths = [[[0, 2, 0]], [[0, 3, 0]]]
    detuning_givens = [[[0, 2, 0], [0, 3, 0]], [0, 0]]
    # fitted detunings
    # detuning_givens = [[[0, 2, 0], [0, 3, 0]], [-0.33011657, 0.5962425]]

    # NEW BEST RUN - UNBUSSED
    # initial_state = [[0, 2, 0]]
    # initial_manifold = 'S12'
    # pulse_train = [[0, 2, 0], [0, 3, 2], [np.nan, np.nan, np.nan], [0, 3, 2],
    #                [0, 2, 0]]

    # fractions = [1 / 3, 1 / 2, np.nan, 1 / 2, 1 / 3]
    # fixed_phase_mask = [0, 0, 0, 1, 1]
    # simulated_phase_mask = [0, 0, 0, 2, 1]

    # measured_states = [2, 23, 19]
    # detuning_paths = [[[0, 2, 0]], [[0, 3, 2]]]
    # # fitted detunings
    # detuning_givens = [[[0, 2, 0], [0, 3, 2]], [-0.21473067, 0.35959748]]
    # # Pei Jiang's
    # # detuning_givens = [[[0, 2, 0], [0, 3, 0]], [-0.43063257, 1.16854282]]
    # # this experiment has 501 experiments per point
    # exp_number = 501
    # exp_data_file = 'Ramsey_experiment_Dimension_3_20240717_1029.txt'

    # # BUSSED
    # initial_state = [[0, 3, 0]]
    # initial_manifold = 'D52'
    # pulse_train = [[-2, 3, 0], [-2, 3, -3], [0, 3,
    #                                          0], [np.nan, np.nan, np.nan],
    #                [0, 3, 0], [-2, 3, -3], [-2, 3, 0], [-2, 3, -3]]

    # fractions = [1 / 3, 1, 1 / 2, np.nan, 1 / 2, 1, 1 / 3, 1]
    # fixed_phase_mask = [0, 0, 0, 0, 1, 1, 1, 1]
    # simulated_phase_mask = [0, 0, 0, 0, 2, 0, 1, 0]

    # # BUSSED
    # initial_state = [[0, 3, 0]]
    # initial_manifold = 'S12'
    # pulse_train = [[0, 2, 0], [0, 3, 0], [-1, 3, 0], [-1, 4, -3], [-1, 3, 0],
    #                [np.nan, np.nan, np.nan], [-1, 3, 0], [-1, 4, -3],
    #                [-1, 3, 0], [0, 3, 0], [-1, 3, 0], [0, 2, 0]]

    # fractions = [1 / 3, 1, 1, 1 / 2, 1, np.nan, 1, 1 / 2, 1, 1, 1, 1 / 3]
    # fixed_phase_mask = [0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1]
    # simulated_phase_mask = [0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0]

    # detuning_paths = [[[0, 3, 0]], [[-1, 3, 0]]]
    # detuning_givens = [[[0, 3, 0]], [1.0]]

    # exp_data_file = 'Ramsey_experiment_Dimension_3_20240703_2224.txt'

    # ############################################
    # # # D=4 pulse sequence - starting in S1/2 m=-2
    # ############################################
    # initial_state = [[-2, 3, -3]]
    # initial_manifold = 'S12'
    # measured_states = [0, 14, 15, 16]

    # pulse_train = [[-2, 3, -3], [-2, 3, -2], [-1, 3, -2], [-1, 3, -2],
    #                [-1, 3, -1], [np.nan, np.nan, np.nan], [-1, 3, -1],
    #                [-1, 3, -2], [-1, 3, -2], [-2, 3, -2], [-1, 3, -2],
    #                [-2, 3, -3]]
    # fractions = [
    #     1 / 4, 1, 1, 1 / 3, 1 / 2, np.nan, 1 / 2, 1 / 3, 1, 1, 1, 1 / 4
    # ]
    # # phases fixed will be multiplied by pi during runs
    # fixed_phase_mask = [0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1]
    # simulated_phase_mask = [0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 1]

    # detuning_paths = [[[-2, 3, -3]], [[-2, 3, -2]], [[-2, 3, -2], [-1, 3, -2]],
    #                   [[-2, 3, -2], [-1, 3, -2], [-1, 3, -1]]]

    # # fitted detunings
    # detuning_givens = [[[-2, 3, -3], [-2, 3, -2], [-1, 3, -2], [-1, 3, -1]],
    #                    [-0.30376543, 0.70049292, -0.34325928, -0.29235278]]
    # # fitted with new pi-times
    # detuning_givens = [[[-2, 3, -3], [-2, 3, -2], [-1, 3, -2], [-1, 3, -1]],
    #                    [-0.19540545, 0.54039218, 0.08857375, -0.57369315]]

    # # # # # Pei Jiang's detunings found - why does this not match my detunings?
    # # # detuning_givens = [[[-2, 3, -3], [-2, 3, -2], [-1, 3, -2], [-1, 3, -1]],
    # # #                    [1.9166, 1.6769, -0.9092, -0.3861]]

    # exp_number = 101
    # exp_data_file = 'Ramsey_experiment_Dimension_4_20240711_1637.txt'

    ############################################
    # # D=5 pulse sequence
    ############################################
    # initial_state = [[0, 2, 0]]
    # initial_manifold = 'D52'
    # pulse_train = [[-2, 2, 0], [-2, 4, -4], [-2, 3, -3], [2, 2, 0], [2, 4, 4],
    #                [2, 4, 3], [np.nan, np.nan, np.nan], [2, 2, 0], [2, 4, 3],
    #                [2, 4, 4], [2, 2, 0], [-2, 2, 0], [-2, 3, -3], [-2, 4, -4],
    #                [-2, 2, 0]]
    # fractions = [
    #     2 / 5, 1 / 2, 1, 2 / 3, 1 / 2, 1, np.nan, 1, 1 / 2, 1 / 3, 1, 1, 1 / 4,
    #     1 / 5, 1
    # ]
    # # phases fixed will be multiplied by pi during runs
    # fixed_phase_mask = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1]
    # simulated_phase_mask = [0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 0, 0, 2, 1, 0]

    # ############################################
    # # # D=5 pulse sequence - new approach to symmetrise U1 and U2
    # ############################################
    # initial_state = [[0, 2, 0]]
    # initial_manifold = 'D52'
    # pulse_train = [[-2, 2, 0], [-2, 4, -4], [-2, 3, -3], [-2, 2, 0], [2, 2, 0],
    #                [2, 4, 4], [2, 4, 3], [2, 2, 0], [np.nan, np.nan, np.nan],
    #                [2, 2, 0], [2, 4, 3], [2, 4, 4], [2, 2, 0], [-2, 2, 0],
    #                [-2, 3, -3], [-2, 4, -4], [-2, 2, 0]]
    # fractions = [
    #     1, 1 / 5, 1 / 4, 1, 1, 1 / 3, 1 / 2, 1, np.nan, 1, 1 / 2, 1 / 3, 1, 1,
    #     1 / 4, 1 / 5, 1
    # ]
    # # phases fixed will be multiplied by pi during runs
    # fixed_phase_mask = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1]
    # simulated_phase_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 0, 0, 2, 1, 0]

    # exp_data_file = 'Ramsey_experiment_Dimension_5_20240705_1807.txt'

    # ############################################
    # # # D=5 pulse sequence - starting in S1/2 m=0
    # ############################################
    # initial_state = [[0, 2, 0]]
    # initial_manifold = 'D52'
    # pulse_train = [[-2, 2, 0], [-2, 4, -4], [-2, 3, -3], [-2, 2, 0], [2, 2, 0],
    #                [2, 4, 4], [2, 4, 3], [2, 2, 0], [np.nan, np.nan, np.nan],
    #                [2, 2, 0], [2, 4, 3], [2, 4, 4], [2, 2, 0], [-2, 2, 0],
    #                [-2, 3, -3], [-2, 4, -4], [-2, 2, 0]]
    # fractions = [
    #     1, 1 / 5, 1 / 4, 1, 1, 1 / 3, 1 / 2, 1, np.nan, 1, 1 / 2, 1 / 3, 1, 1,
    #     1 / 4, 1 / 5, 1
    # ]
    # # phases fixed will be multiplied by pi during runs
    # fixed_phase_mask = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1]
    # simulated_phase_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 0, 0, 2, 1, 0]

    # ############################################
    # # # D=6 pulse sequence - with symmetrised U1 and U2
    # ############################################
    # initial_state = [[0, 2, 0]]
    # initial_manifold = 'D52'
    # pulse_train = [[-2, 2, 0], [-2, 3, -3], [-2, 4, -4], [-2, 2, 0], [2, 2, 0],
    #                [2, 2, 1], [2, 4, 4], [2, 4, 3], [2, 2, 0],
    #                [np.nan, np.nan, np.nan], [2, 2, 0], [2, 4, 3], [2, 4, 4],
    #                [2, 2, 1], [2, 2, 0], [-2, 2, 0], [-2, 4, -4], [-2, 3, -3],
    #                [-2, 2, 0]]
    # fractions = [
    #     1, 1 / 6, 1 / 5, 1, 1, 1 / 4, 1 / 3, 1 / 2, 1, np.nan, 1, 1 / 2, 1 / 3,
    #     1 / 4, 1, 1, 1 / 5, 1 / 6, 1
    # ]
    # # phases fixed will be multiplied by pi during runs
    # fixed_phase_mask = [
    #     0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1
    # ]
    # simulated_phase_mask = [
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 4, 3, 0, 0, 2, 1, 0
    # ]

    # ############################################
    # # # D=7 pulse sequence - with symmetrised U1 and U2
    # ############################################
    # initial_state = [[0, 2, 0]]
    # initial_manifold = 'D52'
    # pulse_train = [[-2, 2, 0], [-2, 4, -4], [-2, 3, -3], [-2, 3, -2],
    #                [-2, 2, 0], [2, 2, 0], [2, 4, 4], [2, 4, 3], [2, 4, 2],
    #                [2, 2, 0], [np.nan, np.nan, np.nan], [2, 2, 0], [2, 4, 2],
    #                [2, 4, 3], [2, 4, 4], [2, 2, 0], [-2, 2, 0], [-2, 3, -2],
    #                [-2, 3, -3], [-2, 4, -4], [-2, 2, 0]]
    # fractions = [
    #     1, 1 / 7, 1 / 6, 1 / 5, 1, 1, 1 / 4, 1 / 3, 1 / 2, 1, np.nan, 1, 1 / 2,
    #     1 / 3, 1 / 4, 1, 1, 1 / 5, 1 / 6, 1 / 7, 1
    # ]
    # # phases fixed will be multiplied by pi during runs
    # fixed_phase_mask = [
    #     0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1
    # ]
    # simulated_phase_mask = [
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 5, 4, 0, 0, 3, 2, 1, 0
    # ]

    # ############################################
    # # # D=16 pulse sequence - init state from S1/2 m=-2
    # ############################################
    # initial_state = [[-2, 4, -4]]  # give transition of init state
    # initial_manifold = 'S12'

    # # fix this by adding the U2 pulses!
    # pulse_train = [[-2, 4, -4], [-2, 3, -3], [-2, 3, -2], [-1, 3, -2],
    #                [-1, 3, -2], [-1, 4, -3], [-1, 3, -1], [-1, 3, 0],
    #                [0, 3, 0], [0, 3, 0], [0, 4, -2], [0, 4, -1], [0, 3, 1],
    #                [0, 3, 2], [0, 4, 0], [1, 4, 0], [1, 4, 0], [1, 3, 3],
    #                [1, 4, 2], [2, 4, 2], [2, 4, 3], [2, 4, 4],
    #                [np.nan, np.nan, np.nan], []]

    # # everything below needs updating!
    # pulse_train = [[-2, 2, 0], [-2, 4, -4], [-2, 3, -3], [-2, 3, -2],
    #                [-2, 2, 0], [2, 2, 0], [2, 4, 4], [2, 4, 3], [2, 4, 2],
    #                [2, 2, 0], [np.nan, np.nan, np.nan], [2, 2, 0], [2, 4, 2],
    #                [2, 4, 3], [2, 4, 4], [2, 2, 0], [-2, 2, 0], [-2, 3, -2],
    #                [-2, 3, -3], [-2, 4, -4], [-2, 2, 0]]
    # fractions = [
    #     1, 1 / 7, 1 / 6, 1 / 5, 1, 1, 1 / 4, 1 / 3, 1 / 2, 1, np.nan, 1, 1 / 2,
    #     1 / 3, 1 / 4, 1, 1, 1 / 5, 1 / 6, 1 / 7, 1
    # ]
    # # phases fixed will be multiplied by pi during runs
    # fixed_phase_mask = [
    #     0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1
    # ]
    # simulated_phase_mask = [
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 5, 4, 0, 0, 3, 2, 1, 0
    # ]

    ra = Ramsey(
        initial_state=initial_state,
        initial_manifold=initial_manifold,
        ref_pitimes=ref_pitimes,
        measured_states=measured_states,  # by ket number not transition!
        periodicity=0.1,
        wait_time=0.0,
        detunings=np.zeros((24, 5)),
        detuning_paths=detuning_paths,
        detuning_givens=detuning_givens,
        pi_time_error=np.zeros((24, 5)),
        pulse_train=pulse_train,
        fractions=fractions,
        fixed_phase_mask=fixed_phase_mask,
        simulated_phase_mask=simulated_phase_mask,
        real_wait_times=True,
        experimental_data_file=None,
        exp_number=exp_number,
        noise_level=0.0)

    # ra.phase_times = np.linspace(0, 4.1 * ra.periodicity, 101)
    # ham = ra.ham(0, [2, 18])
    # print(ham)
    # ham = ra.ham(0, [2, 14])
    # print(ham)
    pulse_times = ra.find_pulse_times()
    print(
        'Length of U1:', 1e3 *
        sum(pulse_times[:np.argwhere(pulse_times == ra.wait_time)[0][0]]),
        'us.')
    print(
        'Length of U2:', 1e3 *
        sum(pulse_times[1 + np.argwhere(pulse_times == ra.wait_time)[0][0]:]),
        'us.')

    # plot pulse sequence ############################################
    ra.plot_pulse_evolution()

    # perform a detuning fit #########################################
    # detunings = ra.fit_detunings()

    # plotting a phase sweep #########################################
    ra.plot_phase_ramsey(plot_from_saved_data=False)

    plt.show()

    print('--- %s seconds ---' % (time.time() - start_time))
