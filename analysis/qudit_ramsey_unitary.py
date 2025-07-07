import os
import re
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy.special import voigt_profile
from scipy.stats import norm
from tqdm import tqdm

from calibration_analysis import Calibrations
from plot_utils import nice_fonts, set_size
from segmented_ramseys import plot_segmented_ramsey

mpl.rcParams.update(nice_fonts)
np.set_printoptions(suppress=True)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = colors * 4


class QuditUnitary(Calibrations):

    def __init__(
            self,
            dimension: int = 8,
            topology: str = 'All',
            detuning: float = 0,  # Hz, a set constant detuning option
            turn_on_real_detuning=True,
            real_Ramsey_wait: float = 0,  # us
            # line_amplitude_60=0,
            # line_amplitude_180=0,
            # line_phase_60=0,
            # line_phase_180=0,
            # line_offset=0,
            # mag_field_noise=0,
            # laser_noise_gaussian=0,
            # laser_noise_lorentzian=0,
            # pulse_time_miscalibration=0,
            # pulse_time_variation=0,
            # freq_miscalibration=0,
            # line_amplitude_60: float = 0.00015,  # Gauss
            # line_amplitude_180: float = 0.00005,  # Gauss
            # line_phase_60: float = -0.636,  # radians
            # line_phase_180: float = -1.551,  # radians
            # line_offset: float = 0.000247,  # Gauss
            # mag_field_noise: float = 0.000049 / (np.sqrt(2) * np.pi),  # Gauss
            # laser_noise_gaussian: float = 1e6 / (2 * np.pi * 1950),  # Hz
            # laser_noise_lorentzian: float = 1e6 / (2 * np.pi * 2065.9),  # Hz
            # pulse_time_miscalibration: float = 0.0177,
            # pulse_time_variation: float = 0.0261,
            # freq_miscalibration: float = 124.94,  # Hz (71.176 Hz measurement error)
            # best in class
            line_amplitude_60: float = 0.00007,  # Gauss, Ruster's 25Hz with 2.8MHz/G transition
            line_amplitude_180: float = 0.0000,  # Gauss, assumed 0
            line_phase_60: float = 0,  # radians
            line_phase_180: float = 0,  # radians
            line_offset: float = 0.0,  # Gauss
            mag_field_noise: float = 0.0000000382,  # Gauss, Ruster16's 2.7pT RMS field*sqrt(2)
            laser_noise_gaussian: float = 0.5,  # Hz, several sources, eg. Young99 and Jiang08
            laser_noise_lorentzian: float = 0.5,  # Hz
            freq_miscalibration: float = 5, # Hz
            pulse_time_miscalibration: float = 0.001,  # %
            pulse_time_variation: float = 0.001,  # %
    ):

        super().__init__()
        self.dimension = dimension
        self.topology = topology
        self.line_amplitude_60 = line_amplitude_60
        self.line_amplitude_180 = line_amplitude_180
        self.line_phase_60 = line_phase_60
        self.line_phase_180 = line_phase_180
        self.line_offset = line_offset
        self.mag_field_noise = mag_field_noise
        self.laser_noise_gaussian = laser_noise_gaussian
        self.laser_noise_lorentzian = laser_noise_lorentzian
        self.pulse_time_miscalibration = pulse_time_miscalibration
        self.pulse_time_variation = pulse_time_variation
        self.freq_miscalibration = freq_miscalibration
        self.input_detuning = detuning
        self.turn_on_real_detuning = turn_on_real_detuning
        self.real_wait = real_Ramsey_wait
        # use the below to regenerate the sensitivities and pitimes tables for
        # new B field or pi-time reference values
        # self.generate_transition_sensitivities()
        # self.generate_transition_strengths()

        # transition sensitivities in MHz/G
        self.transition_sensitivities = np.loadtxt(
            'quick_reference_arrays/transition_sensitivities.txt',
            delimiter=',')
        self.all_transition_sensitivities = np.loadtxt(
            'quick_reference_arrays/all_transition_sensitivities.txt',
            delimiter=',')
        self.transition_pitimes = np.loadtxt(
            'quick_reference_arrays/transition_pitimes.txt', delimiter=',')
        self.transition_strengths = np.loadtxt(
            'quick_reference_arrays/transition_strengths.txt', delimiter=',')
        # print(self.transition_sensitivities)
        # print(self.transition_pitimes)
        # print(self.transition_strengths)

    def get_pulse_sequence(self, verbose=False) -> dict:

        relative_dir = 'ramseys_pulse_sequences/'

        # Construct the file name based on the dimension
        if self.topology == 'All':
            filename = f'D{self.dimension}_Ramsey_PulseSequence.txt'
        elif self.topology == 'Star':
            filename = (f'D{self.dimension}_' +
                        'Ramsey_PulseSequenceStarTopology_' +
                        'EncodedHerald_False.txt')

        full_path = os.path.join(relative_dir, filename)

        # print(full_path)

        # Helper function to extract lists from matches
        def extract_list(match):
            return eval(match.group(1)) if match else []

        # Check if the file exists
        if os.path.isfile(full_path):
            with open(full_path, 'r') as file:
                initial_manifold = file.readline().rstrip()[-5:-1]

                content = file.read()

                # # Use regex to find all the required lists in the text
                initial_trans = extract_list(
                    re.search(r'initial_state\s*=\s*(\[\[.*?\]\])', content))

                pulse_train_U1 = extract_list(
                    re.search(r'pulse_train_U1\s*=\s*(\[\[.*?\]\])', content))
                fractions_U1 = extract_list(
                    re.search(r'fractions_U1\s*=\s*(\[.*?\])', content))
                simulated_phase_mask_U1 = extract_list(
                    re.search(r'simulated_phase_mask_U1\s*=\s*(\[.*?\])',
                              content))
                fixed_phase_mask_U1 = extract_list(
                    re.search(r'fixed_phase_mask_U1\s*=\s*(\[.*?\])', content))

                pulse_train_U2 = extract_list(
                    re.search(r'pulse_train_U2\s*=\s*(\[\[.*?\]\])', content))
                fractions_U2 = extract_list(
                    re.search(r'fractions_U2\s*=\s*(\[.*?\])', content))
                simulated_phase_mask_U2 = extract_list(
                    re.search(r'simulated_phase_mask_U2\s*=\s*(\[.*?\])',
                              content))
                fixed_phase_mask_U2 = extract_list(
                    re.search(r'fixed_phase_mask_U2\s*=\s*(\[.*?\])', content))

                probe_trans = extract_list(
                    re.search(r'probe_trans\s*=\s*(\[\[.*?\]\])', content))

                s12_state_shelvings = extract_list(
                    re.search(r's12_state_shelvings\s*=\s*(\[\[.*?\]\])',
                              content))
                s12_state_fractions = extract_list(
                    re.search(r's12_state_fractions\s*=\s*(\[.*?\])', content))
                s12_state_simulated_phases = extract_list(
                    re.search(r's12_state_simulated_phases\s*=\s*(\[.*?\])',
                              content))
                s12_state_fixed_phases = extract_list(
                    re.search(r's12_state_fixed_phases\s*=\s*(\[.*?\])',
                              content))

                # # From Pei Jiang's experiment
                # initial_manifold = 'D5/2'
                # initial_trans = [[0, 3, -1]]
                # pulse_train_U1 = [[0, 3, -1], [0, 4, 0], [0, 3, -1],
                #                     [0, 4, -1], [1, 3, -1], [-1, 3, -1],
                #                     [-1, 2, -2]]
                # fractions_U1 = [
                #     0.2, 1, 0.25, 1, 0.3333333333333333, 0.5, 1
                # ]
                # simulated_phase_mask_U1 = [0, 0, 0, 0, 0, 0, 0]
                # fixed_phase_mask_U1 = [1, 0, 1, 0, 1, 1, 0]
                # probe_trans = [[0, 4, -1], [-1, 2, -2], [0, 3, -1]]
                # pulse_train_U2 = [[-1, 2, -2], [-1, 3, -1], [-1, 2, -2],
                #                     [1, 3, -1], [0, 4, -1], [0, 3, -1],
                #                     [0, 4, -1], [0, 4, 0], [0, 3, -1],
                #                     [0, 4, 0], [1, 4, 1], [-1, 2, -2]]
                # fractions_U2 = [
                #     1, 0.5, 1, 0.3333333333333333, 1, 0.25, 1, 1, 0.2, 1,
                #     1, 1
                # ]
                # simulated_phase_mask_U2 = [
                #     0, 4, 0, 3, 0, 2, 0, 0, 1, 0, 0, 0
                # ]
                # fixed_phase_mask_U2 = [
                #     1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0
                # ]

                # # From Gaurav for comparison
                # initial_manifold = 'D5/2'
                # initial_state = [[0, 3, -1]]
                # pulse_train_U1 = [[0, 3, -1], [0, 4, 0], [0, 3, -1],
                #                     [0, 4, -1], [1, 3, -1], [-1, 3, -1],
                #                     [-1, 2, -2]]
                # fractions_U1 = [
                #     0.2, 1, 0.25, 1, 0.3333333333333333, 0.5, 1
                # ]
                # simulated_phase_mask_U1 = [0, 0, 0, 0, 0, 0, 0]
                # fixed_phase_mask_U1 = [1, 0, 1, 0, 1, 1, 0]
                # probe_trans = [[0, 4, -1], [-1, 2, -2], [0, 3, -1]]
                # pulse_train_U2 = [[-1, 2, -2], [-1, 3, -1], [-1, 2, -2],
                #                     [1, 3, -1], [0, 4, -1], [0, 3, -1],
                #                     [0, 4, -1], [0, 4, 0], [0, 3, -1],
                #                     [0, 4, 0], [1, 4, 1], [-1, 2, -2]]
                # fractions_U2 = [
                #     1, 0.5, 1, 0.3333333333333333, 1, 0.25, 1, 1, 0.2, 1,
                #     1, 1
                # ]
                # simulated_phase_mask_U2 = [
                #     0, 4, 0, 3, 0, 2, 0, 0, 1, 0, 0, 0
                # ]
                # fixed_phase_mask_U2 = [
                #     1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0
                # ]

                if initial_manifold == 'S1/2':
                    self.initial_state = np.array([0, initial_trans[0][0]])
                    self.herald_state = np.array(
                        [initial_trans[0][1], initial_trans[0][2]])
                else:
                    self.initial_state = np.array(
                        [initial_trans[0][1], initial_trans[0][2]])
                    self.herald_state = self.initial_state

                # include pre-readout shelving pulses into U2
                if len(s12_state_shelvings) != 0:
                    pulse_train_U2 = np.concatenate(
                        (pulse_train_U2, s12_state_shelvings))
                    fractions_U2 = np.concatenate(
                        (fractions_U2, s12_state_fractions))
                    simulated_phase_mask_U2 = np.concatenate(
                        (simulated_phase_mask_U2, s12_state_simulated_phases))
                    fixed_phase_mask_U2 = np.concatenate(
                        (fixed_phase_mask_U2, s12_state_fixed_phases))
                else:
                    pass

                if len(s12_state_shelvings) != 0:
                    all_transitions = np.concatenate(
                        (initial_trans, pulse_train_U1, pulse_train_U2,
                         s12_state_shelvings, probe_trans))

                    if initial_manifold == 'S1/2':
                        pulse_train_U1 = np.concatenate(
                            (initial_trans, pulse_train_U1))
                        fractions_U1 = np.concatenate(([1], fractions_U1))
                        simulated_phase_mask_U1 = np.concatenate(
                            ([0], simulated_phase_mask_U1))
                        fixed_phase_mask_U1 = np.concatenate(
                            ([1], fixed_phase_mask_U1))

                else:
                    all_transitions = np.concatenate(
                        (initial_trans, pulse_train_U1, pulse_train_U2,
                         probe_trans))
                    if initial_manifold == 'S1/2':
                        pulse_train_U1 = np.concatenate(
                            (initial_trans, pulse_train_U1))
                        fractions_U1 = np.concatenate(([1], fractions_U1))
                        simulated_phase_mask_U1 = np.concatenate(
                            ([0], simulated_phase_mask_U1))
                        fixed_phase_mask_U1 = np.concatenate(
                            ([1], fixed_phase_mask_U1))

                self.all_transitions = all_transitions

                # Get all states participating for the simulation
                # Extract unique states involved in the scheme using all_trans
                unique_s12_entries = np.array(
                    [[0, ind] for ind in np.unique(all_transitions[:, 0])])
                unique_d52_entries = np.array(
                    list({tuple(row[1:])
                          for row in all_transitions}))
                self.basis_order = np.concatenate(
                    (unique_s12_entries, unique_d52_entries))

                if verbose:
                    print('States involved in the Ramsey:\n', self.basis_order)

                pulse_dict = {
                    'init_trans': initial_trans,
                    'U1_transitions': pulse_train_U1,
                    'U1_fractions': fractions_U1,
                    'U1_sim_phase': simulated_phase_mask_U1,
                    'U1_fix_phase': fixed_phase_mask_U1,
                    'U2_transitions': pulse_train_U2,
                    'U2_fractions': fractions_U2,
                    'U2_sim_phase': simulated_phase_mask_U2,
                    'U2_fix_phase': fixed_phase_mask_U2,
                    's12_shelvings': s12_state_shelvings,
                    's12_fractions': s12_state_fractions,
                    's12_fix_phases': s12_state_fixed_phases,
                    's12_sim_phases': s12_state_simulated_phases,
                    'probes': probe_trans,
                    'all_trans': all_transitions
                }

                self.pulse_dict = pulse_dict

        return pulse_dict

    def get_full_experiment_runtime(self, dimension):

        self.dimension = dimension
        self.get_pulse_sequence()

        coherent_pulses = np.concatenate(
            (self.pulse_dict['U1_transitions'][1:],
             self.pulse_dict['U2_transitions']))

        coherent_fractions = np.concatenate(
            (self.pulse_dict['U1_fractions'][1:],
             self.pulse_dict['U2_fractions']))

        coherent_time = 0
        coherent_pulse_times = []
        for idx, trans in enumerate(coherent_pulses):
            table_index = self.convert_triplet_index(trans)
            trans_pitime = self.transition_pitimes[tuple(table_index)]  # us

            fraction = coherent_fractions[idx]

            coherent_time += (
                (2 * trans_pitime) / np.pi) * np.arcsin(np.sqrt(fraction))
            coherent_pulse_times.append(
                ((2 * trans_pitime) / np.pi) * np.arcsin(np.sqrt(fraction)))

        all_pulses = np.concatenate(
            (self.pulse_dict['U1_transitions'],
             self.pulse_dict['U2_transitions'], self.pulse_dict['probes']))

        all_fractions = np.concatenate(
            (self.pulse_dict['U1_fractions'], self.pulse_dict['U2_fractions'],
             np.ones(len(self.pulse_dict['probes']))))

        total_time = 0
        all_pulse_times = []
        for idx, trans in enumerate(all_pulses):
            table_index = self.convert_triplet_index(trans)
            trans_pitime = self.transition_pitimes[tuple(table_index)]  # us

            fraction = all_fractions[idx]

            total_time += (
                (2 * trans_pitime) / np.pi) * np.arcsin(np.sqrt(fraction))
            all_pulse_times.append(
                ((2 * trans_pitime) / np.pi) * np.arcsin(np.sqrt(fraction)))

        # breakpoint()

        # NOTE The true time will be ~200us longer on avg. due to init. pulses
        return total_time, coherent_time

    def initialise_system(self) -> np.ndarray:
        self.get_pulse_sequence()
        self.system_dim = len(self.basis_order)
        init_state_index = np.where(
            (self.basis_order == self.herald_state).all(axis=1))[0][0]

        init_vec = np.zeros(self.system_dim)
        init_vec[init_state_index] = 1  # assumed perfect initialisation
        self.init_vec = init_vec

        return init_vec

    def generate_magnetic_noise(self) -> float:
        '''Generates a single instance of magnetic field value (as a change
        from the initial value) according to a normal distribution.'''
        return np.random.normal(loc=0.0, scale=self.mag_field_noise)

    # def generate_laser_noise(self) -> float:
    #     '''Generates a single instance of laser noise (in Hz) assuming
    #     a normal distribution convolved with a Lorentzian function.'''
    #     # Generate Gaussian noise
    #     gaussian_noise = np.random.normal(loc=0.0,
    #                                       scale=self.laser_noise_gaussian)

    #     # Define the Lorentzian kernel
    #     lorentzian_width = self.laser_noise_lorentzian
    #     x = np.linspace(-3 * lorentzian_width, 3 * lorentzian_width, 100)
    #     lorentzian_kernel = 1 / (np.pi * lorentzian_width *
    #                              (1 + (x / lorentzian_width)**2))

    #     # Perform convolution
    #     convolved_noise = convolve([gaussian_noise],
    #                                lorentzian_kernel,
    #                                mode='same')

    #     # Return only the central value of the convolved result
    #     # return convolved_noise[len(convolved_noise) // 2]
    #     return gaussian_noise

    def generate_voigt(self,
                       x=np.linspace(-1000, 1000, int(1e6)),
                       verbose=False):
        '''Generates the Voigt profile.'''

        # Define parameters for Gaussian and Lorentzian
        sigma = self.laser_noise_gaussian  # Standard deviation of Gaussian
        gamma = self.laser_noise_lorentzian  # Width of Lorentzian

        def gaussian(x, sigma=1):
            '''Returns the Gaussian function evaluated at x.'''
            return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 *
                                                               (x / sigma)**2)

        def lorentzian(x, gamma=1):
            '''Returns the Lorentzian function evaluated at x.
            Here gamma is the half-width at half-maximum.'''
            return (1 / np.pi) * (gamma) / (x**2 + (gamma)**2)

        # individual plots for Gaussian and Lorentzian contributions
        G = gaussian(x, sigma)
        L = lorentzian(x, gamma)

        self.voigt_x_values = x
        self.voigt = voigt_profile(x, sigma, gamma)
        self.voigt_cdf = np.cumsum(self.voigt / np.sum(self.voigt))

        if verbose:

            # Plot the results
            plt.figure(figsize=(10, 6))
            plt.plot(x, G, label='Gaussian', color='blue', alpha=0.7)
            plt.plot(x, L, label='Lorentzian', color='red', alpha=0.7)
            plt.plot(x,
                     voigt_profile(x, sigma, gamma),
                     label='Voigt',
                     color='red',
                     linewidth=2)
            plt.legend(loc='center right')
            plt.grid()

            ax2 = plt.gca().twinx()  # gca() gets the current Axes instance
            ax2.plot(x,
                     self.voigt_cdf,
                     label='CDF',
                     color='orange',
                     linewidth=2)
            ax2.legend(loc='center left')

            plt.title('Convolution of Gaussian and Lorentzian')

    def generate_laser_noise(self) -> float:
        '''Generates a single random sample from the convolution CDF.'''
        if hasattr(self, 'voigt_cdf'):
            pass
        else:
            self.voigt_x_values = np.linspace(-1000, 1000, int(1e6))
            self.voigt = voigt_profile(self.voigt_x_values,
                                       self.laser_noise_gaussian,
                                       self.laser_noise_lorentzian)
            if np.sum(self.voigt) != 0:
                self.voigt_cdf = np.cumsum(self.voigt / np.sum(self.voigt))
            else:
                self.voigt_cdf = np.cumsum(self.voigt)

        if self.laser_noise_gaussian or self.laser_noise_lorentzian:
            # Generate a random value between 0 and 1
            random_value = np.random.rand()

            # Find the index corresponding to the random value in the CDF
            index = np.searchsorted(self.voigt_cdf, random_value)
            if index == 1000000:
                index = 999999

            # Return the corresponding x value
            # return self.voigt_x_values[index]
            return self.voigt_x_values[index]
        else:
            return 0

    def generate_frequency_miscalibration(self) -> float:
        '''Generates a single instance of frequency mis-calibration (as a
        change from the resonant value) in MHz according to a normal
        distribution.'''
        # use line below to just set a constant detuning to frequencies
        # return self.freq_miscalibration
        return np.random.normal(loc=0.0, scale=self.freq_miscalibration)

    def generate_pulse_time_miscalibration(self) -> float:
        '''Generates a single instance of pulse time miscalibration (as a
        percent change from the resonant value) in MHz according to a normal
        distribution.'''
        return np.random.normal(loc=0.0, scale=self.pulse_time_miscalibration)

    def generate_pulse_time_variation(self) -> float:
        '''Generates a single instance of pulse time variation (as a
        percent change from the resonant value) in MHz according to a normal
        distribution. This is the change over time we have measured on a time scale
        of several hours which is relevant for drifts in eg. 1762 laser power/
        alignment.'''
        return np.random.normal(loc=0.0, scale=self.pulse_time_variation)

    def generate_line_noise_detuning(self, t: float) -> float:
        '''This is the line signal at some time t (in us),
        since the integral will give us the phase accumulated.'''
        val = ((self.line_amplitude_60) *
               np.sin(2 * np.pi * 60 * t * 1e-6 + self.line_phase_60) +
               (self.line_amplitude_180) *
               np.sin(2 * np.pi * 180 * t * 1e-6 + self.line_phase_180) +
               self.line_offset)
        # ensure that the mixed signal starts at 0 amplitude
        # since the start of experiment sets the reference point
        val -= ((self.line_amplitude_60) * np.sin(self.line_phase_60) +
                (self.line_amplitude_180) * np.sin(self.line_phase_180) +
                self.line_offset)
        return val

    def generate_line_noise_integrand(self, t: float) -> float:
        '''This is the integral of the line signal at some time t,
        since the integral will give us the phase accumulated.'''

        val = ((-(self.line_amplitude_60 / (2 * np.pi * 60 * 1e-6)) *
                np.cos(2 * np.pi * 60 * t * 1e-6 + self.line_phase_60) -
                (self.line_amplitude_180 / (2 * np.pi * 180 * 1e-6)) *
                np.cos(2 * np.pi * 180 * t * 1e-6 + self.line_phase_180) +
                self.line_offset * t) -
               (-(self.line_amplitude_60 /
                  (2 * np.pi * 60 * 1e-6)) * np.cos(self.line_phase_60) -
                (self.line_amplitude_180 /
                 (2 * np.pi * 180 * 1e-6)) * np.cos(self.line_phase_180)))

        # need to again re-centre the integral as well as detuning value above
        # so it starts at 0
        val -= (((self.line_amplitude_60) * np.sin(self.line_phase_60) +
                 (self.line_amplitude_180) * np.sin(self.line_phase_180) +
                 self.line_offset) * t)

        return val

    def build_unitary(self,
                      transition: np.ndarray = [-2, 3, -1],
                      time: float = 1,
                      phase: float = 0,
                      real_detuning: float = 0,
                      pulse_miscal: float = 0,
                      pulse_var: float = 0) -> np.ndarray:

        hamiltonian = np.zeros((len(self.basis_order), len(self.basis_order)),
                               dtype=complex)
        table_index = self.convert_triplet_index(transition)

        s12_index = np.where(
            (self.basis_order == [0, transition[0]]).all(axis=1))[0][0]
        d52_index = np.where(
            (self.basis_order == [transition[1],
                                  transition[2]]).all(axis=1))[0][0]

        trans_pitime = self.transition_pitimes[tuple(table_index)]  # us

        # alter transition strength slightly to mimic laser intensity fluctuations
        trans_Omega = np.pi / (trans_pitime * (1 + pulse_miscal) *
                               (1 + pulse_var))  # rad/s

        # effective_Omega = np.sqrt(real_detuning**2 + trans_Omega**2)

        hamiltonian[s12_index,
                    d52_index] = (trans_Omega / 2) * np.exp(1j * phase)
        hamiltonian[d52_index,
                    s12_index] = np.conjugate(hamiltonian[s12_index,
                                                          d52_index])

        if self.turn_on_real_detuning:
            hamiltonian[s12_index, s12_index] = 0
            hamiltonian[d52_index, d52_index] = -real_detuning

        unitary = sc.linalg.expm(-1j * hamiltonian * time)

        return unitary

    def evolve_qudit_Ramsey(self,
                            sweep_phase: float = 0,
                            verbose: bool = False):

        # define the value of the magnetic field change for this run
        self.per_run_mag_noise = self.generate_magnetic_noise()
        self.per_run_laser_noise = self.generate_laser_noise()
        self.per_run_freq_miscal = self.generate_frequency_miscalibration()
        self.per_run_pulse_miscal = self.generate_pulse_time_miscalibration()
        self.per_run_pulse_var = self.generate_pulse_time_variation()

        total_experiment_time = 0

        # U1 evolution
        U1_unitaries = np.empty(
            (len(self.pulse_dict['U1_transitions']), len(
                self.basis_order), len(self.basis_order)),
            dtype=complex)

        for idx, trans in enumerate(self.pulse_dict['U1_transitions']):
            table_index = self.convert_triplet_index(trans)
            trans_pitime = self.transition_pitimes[tuple(table_index)]  # us

            # apply pi miscalibration only after the first pulse of the
            # sequence (ie. _after_ the pulse that brings down the population
            # to S1/2 after heralding if the initial state in S1/2)

            # change pulse time with noise for all but first pulse
            # if idx == 0:
            #     # add deshelving pulse infidelity, which can't be heralded out and
            #     # is a non-line triggered pulse - so larger error than usual
            #     trans_pitime *= (1 + self.pulse_time_miscalibration)
            # else:
            #     trans_pitime *= (1 + self.per_run_pulse_miscal)

            # Applying both pulse time errors here
            # trans_pitime *= (1 + self.per_run_pulse_miscal)
            # trans_pitime *= (1 + self.per_run_pulse_var)

            # print(self.per_run_pulse_miscal)
            # print(trans_pitime)
            trans_sens = self.transition_sensitivities[tuple(
                table_index)]  # MHz/G

            fraction = self.pulse_dict['U1_fractions'][idx]
            time = ((2 * trans_pitime) / np.pi) * np.arcsin(np.sqrt(fraction))

            if verbose:
                print(
                    2 * np.pi * trans_sens *
                    self.generate_line_noise_integrand(total_experiment_time))
                print('trans sens', trans_sens)
                print('exp time', total_experiment_time)

            # first put together whether pulses are going from S->D or D->S
            phase = np.pi * self.pulse_dict['U1_fix_phase'][idx]
            # add in simulated phases for phase scans to get contrast
            phase += sweep_phase * self.pulse_dict['U1_sim_phase'][idx]

            # add in noise sources, starting with line signal (60/180 Hz)
            # if idx == 0:
            #     real_detuning = (2 * np.pi * trans_sens *
            #                      self.generate_line_noise_detuning(
            #                          (1e6 / 60) * np.random.random()))
            # else:
            real_detuning = (
                2 * np.pi * trans_sens *
                self.generate_line_noise_detuning(total_experiment_time))

            phase += (
                2 * np.pi * trans_sens *
                self.generate_line_noise_integrand(total_experiment_time))
            if verbose:
                print(
                    '180 line phase', 2 * np.pi * trans_sens *
                    self.generate_line_noise_integrand(total_experiment_time))

            # now the magnetic field Gaussian noise
            real_detuning += 2 * np.pi * trans_sens * self.per_run_mag_noise
            phase += (2 * np.pi * trans_sens * self.per_run_mag_noise *
                      total_experiment_time)

            # then the laser noise (taken solely from the [0,2,0] decay fit)
            real_detuning += 2 * np.pi * 1e-6 * self.per_run_laser_noise
            phase += (2 * np.pi * 1e-6 * self.per_run_laser_noise *
                      total_experiment_time)

            # next the frequency miscalibration
            real_detuning += 2 * np.pi * 1e-6 * self.per_run_freq_miscal
            phase += (2 * np.pi * 1e-6 * self.per_run_freq_miscal *
                      total_experiment_time)

            # finally, if there is a constant detuning add it here
            real_detuning += 2 * np.pi * 1e-6 * self.input_detuning
            phase += (2 * np.pi * 1e-6 * self.input_detuning *
                      total_experiment_time)

            # print(phase)

            new_unit = self.build_unitary(
                transition=trans,
                time=time,
                phase=phase,
                real_detuning=real_detuning,
                pulse_miscal=self.per_run_pulse_miscal,
                pulse_var=self.per_run_pulse_var
                )
            U1_unitaries[idx, :, :] = new_unit

            if idx == 0:
                pass
            else:
                total_experiment_time += time

        U1_total_unitary = U1_unitaries[0, :, :]
        for i in range(1, len(self.pulse_dict['U1_transitions'])):
            U1_total_unitary = np.matmul(U1_unitaries[i, :, :],
                                         U1_total_unitary)

        # add in additional wait time for the experiment
        total_experiment_time += self.real_wait

        # U2 evolution
        U2_unitaries = np.empty(
            (len(self.pulse_dict['U2_transitions']), len(
                self.basis_order), len(self.basis_order)),
            dtype=complex)

        for idx, trans in enumerate(self.pulse_dict['U2_transitions']):
            table_index = self.convert_triplet_index(trans)
            trans_pitime = self.transition_pitimes[tuple(table_index)]  # us
            # apply pi miscalibration
            # trans_pitime *= (1 + self.per_run_pulse_miscal)
            trans_sens = self.transition_sensitivities[tuple(
                table_index)]  # MHz/G

            fraction = self.pulse_dict['U2_fractions'][idx]
            time = ((2 * trans_pitime) / np.pi) * np.arcsin(np.sqrt(fraction))
            if verbose:
                print(
                    2 * np.pi * trans_sens *
                    self.generate_line_noise_integrand(total_experiment_time))
                print('trans sens', trans_sens)
                print('exp time', total_experiment_time)

            # first put together whether pulses are going from S->D or D->S
            phase = np.pi * self.pulse_dict['U2_fix_phase'][idx]
            # add in simulated phases for phase scans to get contrast
            phase += sweep_phase * self.pulse_dict['U2_sim_phase'][idx]

            # add in noise sources, starting with line signal (60/180 Hz)
            real_detuning = (
                2 * np.pi * trans_sens *
                self.generate_line_noise_detuning(total_experiment_time))
            phase += (
                2 * np.pi * trans_sens *
                self.generate_line_noise_integrand(total_experiment_time))
            if verbose:
                print(
                    '180 line phase',
                    trans_sens *
                    self.generate_line_noise_integrand(total_experiment_time))

            # now the magnetic field Gaussian noise
            real_detuning += 2 * np.pi * trans_sens * self.per_run_mag_noise
            phase += (2 * np.pi * trans_sens * self.per_run_mag_noise *
                      total_experiment_time)

            # then the laser noise (taken solely from the [0,2,0] decay fit)
            real_detuning += 2 * np.pi * 1e-6 * self.per_run_laser_noise
            phase += (2 * np.pi * 1e-6 * self.per_run_laser_noise *
                      total_experiment_time)

            # next the frequency miscalibration
            real_detuning += 2 * np.pi * 1e-6 * self.per_run_freq_miscal
            phase += (2 * np.pi * 1e-6 * self.per_run_freq_miscal *
                      total_experiment_time)
            # finally, if there is a constant detuning add it here
            real_detuning += 2 * np.pi * 1e-6 * self.input_detuning
            phase += (2 * np.pi * 1e-6 * self.input_detuning *
                      total_experiment_time)

            # print(total_experiment_time)
            # print(phase)

            # print(phase)

            new_unit = self.build_unitary(
                transition=trans,
                time=time,
                phase=phase,
                real_detuning=real_detuning,
                pulse_miscal=self.per_run_pulse_miscal,
                pulse_var=self.per_run_pulse_var
                )
            U2_unitaries[idx, :, :] = new_unit

            total_experiment_time += time

        U2_total_unitary = U2_unitaries[0, :, :]
        for i in range(1, len(self.pulse_dict['U2_transitions'])):
            U2_total_unitary = np.matmul(U2_unitaries[i, :, :],
                                         U2_total_unitary)

        U1_output_vec = U1_total_unitary @ self.init_vec
        output_vec = U2_total_unitary @ U1_total_unitary @ self.init_vec

        # print(total_experiment_time)

        # all_pulses = np.concatenate(
        #     (self.pulse_dict['U1_transitions'],
        #      self.pulse_dict['U2_transitions']))
        # print(all_pulses)

        return np.square(
            (np.abs(output_vec))), np.square(np.abs(U1_output_vec))

    def scan_phases(
        self, phases: np.ndarray = np.linspace(0, 2 * np.pi,
                                               101)) -> np.ndarray:

        self.initialise_system()
        probs = np.empty((len(self.basis_order), len(phases)))
        U1_probs = np.empty((len(self.basis_order), len(phases)))
        for idx, phase in enumerate(phases):
            out_probs, U1_out_probs = self.evolve_qudit_Ramsey(
                sweep_phase=phase)
            probs[:, idx] = out_probs
            U1_probs[:, idx] = U1_out_probs
        return phases, probs, U1_probs

    def plot_result(self, phases, probs, just_ket_0=True):
        fig, ax = plt.subplots(figsize=set_size(width='full'))

        phases = phases / np.pi
        if just_ket_0:
            ket_0_index = np.where(
                (self.basis_order == self.initial_state).all(axis=1))[0][0]
            plt.plot(phases, probs[ket_0_index, :])
        else:
            for idx, state in enumerate(self.basis_order):
                if state[0] == 0:
                    plot_label = (rf'$|S_{{1/2}}, F=2, m={state[1]}\rangle$')
                else:
                    plot_label = (
                        rf'$|D_{{5/2}}, F={state[0]}, m={state[1]}\rangle$')
                plt.plot(phases, probs[idx, :], label=plot_label)

        plt.title(fr'$d={self.dimension}$ Ramsey Simulation')
        plt.xlabel(r'Phase $\phi$ (rad) / $\pi$')
        plt.legend()
        plt.grid()
        plt.tight_layout()

    def run_simulation(self,
                       phase_points: int = 51,
                       iterations: int = 10,
                       show_iterations=False,
                       plot_all_states=False):

        results = np.empty((len(self.basis_order), phase_points, iterations))

        ket_0_index = np.where(
            (self.basis_order == self.initial_state).all(axis=1))[0][0]
        state = self.initial_state
        if state[0] == 0:
            plot_label = (rf'$|S_{{1/2}}, F=2, m={state[1]}\rangle$')
        else:
            plot_label = (rf'$|D_{{5/2}}, F={state[0]}, m={state[1]}\rangle$')

        fig, ax = plt.subplots(figsize=set_size(width='full'))

        if show_iterations:
            for itera in tqdm(range(iterations)):
                phases = np.linspace(0, 2 * np.pi, phase_points)
                phases, probs, U1_probs = self.scan_phases(phases=phases)
                if plot_all_states:
                    for idx, base in enumerate(self.basis_order):
                        plt.plot(phases,
                                 probs[idx, :],
                                 color=colors[idx],
                                 alpha=3 / iterations)
                else:
                    plt.plot(phases,
                             probs[ket_0_index, :],
                             color='grey',
                             alpha=3 / iterations)
                results[:, :, itera] = probs

        else:
            for itera in tqdm(range(iterations)):
                phases = np.linspace(0, 2 * np.pi, phase_points)
                phases, probs, U1_probs = self.scan_phases(phases=phases)
                results[:, :, itera] = probs

        averaged_results = np.mean(results, axis=2)
        averaged_std = np.std(results, axis=2) / np.sqrt(iterations)

        if plot_all_states:
            for idx, base in enumerate(self.basis_order):
                if base[0] == 0:
                    plot_label = (rf'$|S_{{1/2}}, F=2, m={base[1]}\rangle$')
                else:
                    plot_label = (
                        rf'$|D_{{5/2}}, F={base[0]}, m={base[1]}\rangle$')

                plt.errorbar(phases,
                             averaged_results[idx, :],
                             yerr=averaged_std[idx, :],
                             fmt='-o',
                             label=plot_label,
                             color=colors[idx])
        else:
            if self.initial_state[0] == 0:
                plot_label = (
                    rf'$|S_{{1/2}}, F=2, m={self.initial_state[1]}\rangle$')
            else:
                plot_label = (rf'$|D_{{5/2}}, F={self.initial_state[0]}, $' +
                              rf'$m={self.initial_state[1]}\rangle$')

            plt.errorbar(phases,
                         averaged_results[ket_0_index, :],
                         yerr=averaged_std[ket_0_index, :],
                         fmt='-o',
                         label=plot_label,
                         color=colors[0])

        # plt.errorbar(phases,
        #              averaged_results[ket_0_index, :],
        #              yerr=averaged_std,
        #              fmt='-o',
        #              label=plot_label,
        #              color=colors[0])
        plt.title(fr'$d={self.dimension}$ Ramsey Simulation')
        plt.xlabel(r'Phase $\phi$ (rad) / $\pi$')
        plt.legend()
        plt.grid()
        plt.tight_layout()

    def run_qudit_contrast_measurement(self,
                                       dimensions: np.ndarray = np.arange(
                                           3, 7, 1),
                                       iterations: int = 101,
                                       verbose: bool = False):

        contrasts = np.empty(len(dimensions))
        contrasts_error = np.empty(len(dimensions))

        for idx, dim in enumerate(dimensions):
            print('Running dimension', dim)

            self.dimension = dim
            self.initialise_system()

            ket_0_index = np.where(
                (self.basis_order == self.initial_state).all(axis=1))[0][0]

            if dim % 2 == 0:
                contrast_phase = np.pi
            else:
                contrast_phase = np.pi * (dim - 1) / dim

            phases = np.array([0, contrast_phase])
            # print(phases)

            results = np.empty(
                (len(self.basis_order), len(phases), iterations))

            for itera in tqdm(range(iterations)):
                phases, probs, U1_probs = self.scan_phases(phases=phases)
                results[:, :, itera] = probs

            averaged_results = np.mean(results, axis=2)
            averaged_std = np.std(results, axis=2)

            # print('high', averaged_results[ket_0_index, 0])
            # print('low', averaged_results[ket_0_index, -1])

            contrasts[idx] = averaged_results[
                ket_0_index, 0] - averaged_results[ket_0_index, -1]
            contrasts_error[idx] = np.sqrt(
                (averaged_std[ket_0_index, 0])**2 +
                (averaged_std[ket_0_index, -1])**2) / np.sqrt(iterations)

            print('Contrast found is:', np.round(100 * contrasts[idx], 3))
            print('Contrast error is:', np.round(100 * contrasts_error[idx], 3))


        if verbose:
            print('\nContrast:', 100 * contrasts[0])

            print(contrasts)
            print(dimensions)

            fig, ax = plt.subplots(figsize=set_size(width='full'))
            plt.errorbar(dimensions, contrasts, yerr=contrasts_error, fmt='o')
            plt.title('Ramsey Simulation Contrast vs. Dimension')
            plt.xlabel(r'Dimension $(d)$')
            plt.ylabel(r'$|0\rangle$ Revival Contrast')
            plt.grid()
            plt.tight_layout()

        return dimensions, contrasts, contrasts_error

    def run_qubit_Ramsey_simulation(self,
                                    iterations: int = 101,
                                    transition: list = [-1, 4, -3],
                                    phases: np.ndarray = np.array(
                                        [np.pi / 2, np.pi, 3 * np.pi / 2]),
                                    pulse_type='direct',
                                    verbose=False):

        if pulse_type == 'direct':
            self.dimension = 2
            print(transition)

            all_transitions = np.array([transition])
            unique_s12_entries = np.array(
                [[0, ind] for ind in np.unique(all_transitions[:, 0])])
            unique_d52_entries = np.array(
                list({tuple(row[1:])
                      for row in all_transitions}))
            self.basis_order = np.concatenate(
                (unique_s12_entries, unique_d52_entries))

            # initial state is ALWAYS in S12 for direct qubit
            # then we shelve it up to D52 (perfectly) and start Ramsey
            # hence herald state is D52
            self.initial_state = np.array([0, transition[0]])
            self.herald_state = np.array([transition[1], transition[2]])

            # put the pulse_dict in the correct format for other methods
            # direct
            self.pulse_dict = {
                'init_trans': [transition],
                'U1_transitions': [transition, transition],
                'U1_fractions': [1, 0.5],
                'U1_sim_phase': [0, 0],
                'U1_fix_phase': [1, 0],
                'U2_transitions': [transition],
                'U2_fractions': [0.5],
                'U2_sim_phase': [1],
                'U2_fix_phase': [1],
                's12_shelvings': [],
                's12_fractions': [],
                's12_fix_phases': [],
                's12_sim_phases': [],
                'probes': [transition],
                'all_trans': [transition, transition, transition]
            }

            self.system_dim = len(self.basis_order)
            init_state_index = np.where(
                (self.basis_order == self.herald_state).all(axis=1))[0][0]

            init_vec = np.zeros(self.system_dim)
            init_vec[init_state_index] = 1  # assumed perfect initialisation
            self.init_vec = init_vec

        elif pulse_type == 'bussed':
            # bussed gets intialised in S12, then shelved to the first D52
            # state for heralding, then the Ramsey pulse sequence proceeds from
            # the D52 state
            # self.dimension = 3

            self.basis_order = np.array([[0, transition[0][0]],
                                         [transition[0][1], transition[0][2]],
                                         [transition[1][1], transition[1][2]]])

            # bussed
            self.pulse_dict = {
                'init_trans': [transition[0]],
                'U1_transitions':
                [transition[0], transition[0], transition[1]],
                'U1_fractions': [1, 0.5, 1],
                'U1_sim_phase': [0, 0, 0],
                'U1_fix_phase': [1, 0, 1],
                'U2_transitions': [transition[1], transition[0]],
                'U2_fractions': [1, 0.5],
                'U2_sim_phase': [0, 1],
                'U2_fix_phase': [0, 1],
                's12_shelvings': [],
                's12_fractions': [],
                's12_fix_phases': [],
                's12_sim_phases': [],
                'probes': [transition[0], transition[1]],
                'all_trans': [transition[0], transition[1]]
            }

            self.system_dim = len(self.basis_order)
            init_state_index = 0

            init_vec = np.zeros(self.system_dim)
            init_vec[init_state_index] = 1  # assumed perfect initialisation
            self.init_vec = init_vec

        results = np.empty((len(self.basis_order), len(phases), iterations))

        if verbose:
            # create the plot in order to show iterations
            fig, ax = plt.subplots(figsize=set_size(width='half'))

        for itera in tqdm(range(iterations)):
            phases, probs, U1_probs = self.scan_phases(phases=phases)
            results[:, :, itera] = probs
            alpha = min(5 / iterations, 0.05)
            if verbose:
                plt.plot(phases / np.pi,
                         probs[0],
                         linestyle='-',
                         color='grey',
                         alpha=alpha)

        averaged_results = np.mean(results, axis=2)

        averaged_std = np.std(results, axis=2)
        # lower, upper = self.find_errors(9,averaged_results,301)
        # asym_error_bar = np.abs(
        #     [averaged_results - lower, upper - averaged_results])

        if verbose:

            def oscfunc(t: float, A: float, p: float, c: float) -> float:
                w = 2 * np.pi / 2
                return c - (A * np.cos(w * t + p)) / 2

            # redo the sine fit to check it's working
            A_guess = 1
            p_guess = 0
            c_guess = 0.5
            p0 = np.array([A_guess, p_guess, c_guess])
            x_new = np.linspace(phases.min() / np.pi,
                                phases.max() / np.pi, 1000)

            bounds = (
                [0, -2 * np.pi, 0],  # Lower bounds
                [1, 2 * np.pi, 1])  # Upper bounds

            popt, pcov = sc.optimize.curve_fit(oscfunc,
                                               phases / np.pi,
                                               np.mean(results, axis=2)[0],
                                               p0=p0,
                                               bounds=bounds,
                                               maxfev=10000)

            plt.plot(x_new, oscfunc(x_new, *popt), 'r-')
            plt.errorbar(phases / np.pi,
                         averaged_results[0, :],
                         yerr=averaged_std[0, :],
                         fmt='o')
            # plt.errorbar(phases / np.pi,
            #              averaged_results[0, :],
            #              yerr=asym_error_bar[0],
            #              fmt='o')

            plt.title(f'Ramsey Fast Calibration - {transition}')
            plt.xlabel(r'Phase ($\phi$)')
            plt.ylabel(r'$|0\rangle$ Population')
            plt.ylim(-.1, 1.1)
            plt.grid()
            plt.tight_layout()

        return phases, results

    def run_qubit_segmented_Ramsey(
            self,
            iterations: int = 101,
            transition: list = [-1, 4, -3],
            phases: np.ndarray = np.array([np.pi / 2, np.pi, 3 * np.pi / 2]),
            wait_times: np.ndarray = np.arange(0, 1000, 100),
            data_dict=None,
            show_phase_scans=False,
            show_phase_drift=False,
            pulse_type='direct',
            verbose=False):

        contrasts = np.zeros(len(wait_times))  # Pre-allocate contrasts
        contrasts_error = np.zeros(len(wait_times))  # Pre-allocate error array

        phase_fits = np.zeros(len(wait_times))
        phase_fits_error = np.zeros(len(wait_times))

        # define function for fitting the results
        def oscfunc(t: float, A: float, p: float, c: float) -> float:
            w = 2 * np.pi / 2
            return c - (A * np.cos(w * t + p)) / 2

        for idx, wait in enumerate(wait_times):

            print(f'Calculating wait time of {wait} us..')
            # update the wait time
            self.real_wait = wait

            # get phase scan of Ramsey through many simulations
            _, idx_result = self.run_qubit_Ramsey_simulation(
                iterations=iterations,
                transition=transition,
                phases=phases,
                pulse_type=pulse_type,
                verbose=show_phase_scans)

            A_guess = 1
            p_guess = 0
            c_guess = 0.5
            p0 = np.array([A_guess, p_guess, c_guess])

            bounds = (
                [0, -np.pi, 0],  # Lower bounds
                [1, np.pi, 1])  # Upper bounds

            popt, pcov = sc.optimize.curve_fit(oscfunc,
                                               phases / np.pi,
                                               np.mean(idx_result, axis=2)[0],
                                               p0=p0,
                                               bounds=bounds,
                                               maxfev=10000)

            # need indices for the max and min values of
            # this particular Ramsey phase scan
            arg_of_max = np.argmax(np.mean(idx_result, axis=2)[0])
            arg_of_min = np.argmin(np.mean(idx_result, axis=2)[0])

            # build an array of contrast values
            # so we can have statistics/error calculation
            contrasts_array = idx_result[0, arg_of_max, :] - idx_result[
                0, arg_of_min, :]

            contrasts[idx] = np.mean(contrasts_array)
            contrasts_error[idx] = np.std(contrasts_array) / np.sqrt(
                iterations)

            contrasts[idx] = popt[0]
            contrasts_error[idx] = np.sqrt(pcov[0][0])

            phase_fits[idx] = popt[1]
            phase_fits_error[idx] = np.sqrt(pcov[1][1])

        if verbose:
            fig, ax = plt.subplots(figsize=set_size(width='half'))

            if data_dict:
                if isinstance(data_dict[str(transition)], list):
                    for dt in data_dict[str(transition)]:
                        (amplitudes, amplitudes_errors, real_wait_times, _, _,
                         phases, phases_errors) = plot_segmented_ramsey(
                             dt, plotting=show_phase_scans)

                else:
                    (amplitudes, amplitudes_errors, real_wait_times, _, _,
                     phases, phases_errors) = plot_segmented_ramsey(
                         data_dict[str(transition)], plotting=show_phase_scans)

                plt.errorbar(real_wait_times,
                             amplitudes,
                             yerr=amplitudes_errors,
                             fmt='o-',
                             label='Data')

            plt.errorbar(wait_times,
                         contrasts,
                         yerr=contrasts_error,
                         fmt='o-',
                         label='Simulation')

            plt.title(f'Qubit Ramsey Decay for {transition}')
            plt.xlabel(r'Ramsey Wait Time / $\mu s$')
            plt.ylabel(r'Contrast of Ramsey Phase Scan')
            # plt.ylim(-.1, 1.1)
            plt.grid()
            plt.legend()
            plt.tight_layout()

        if show_phase_drift:

            if verbose:
                fig, ax = plt.subplots(figsize=set_size(width='half'))
                plt.errorbar(wait_times,
                             phase_fits,
                             yerr=phase_fits_error,
                             fmt='o-',
                             label='Simulation')

                if data_dict:
                    plt.errorbar(real_wait_times,
                                 phases,
                                 yerr=phases_errors,
                                 fmt='o-',
                                 label='Data')

                plt.title(f'Qubit Ramsey Phase Drift for {transition}')
                plt.xlabel(r'Ramsey Wait Time / $\mu s$')
                plt.ylabel(r'Phase Drift ($\phi/\pi$)')
                # plt.ylim(-.1, 1.1)
                plt.grid()
                plt.legend()
                plt.tight_layout()

        return phases, contrasts, contrasts_error

    def plot_noise(self, source: str = 'laser'):

        if source == 'magnetic':
            noise_samples = [
                self.generate_magnetic_noise() for _ in range(int(1e5))
            ]

            fig, ax = plt.subplots()
            plt.hist(noise_samples,
                     density=True,
                     bins=1000,
                     alpha=0.5,
                     color=colors[0])
            plt.title('Histogram of Magnetic Field Noise')
            plt.xlabel(r'$\Delta_B$ / G')
            plt.ylabel('Occurrences')

        elif source == 'laser':
            self.generate_voigt(verbose=True)

            fig, ax = plt.subplots()
            noise_samples = [
                self.generate_laser_noise() for _ in range(int(1e5))
            ]

            plt.hist(noise_samples,
                     density=True,
                     bins=1000,
                     alpha=0.5,
                     color=colors[0])
            plt.title('Histogram of Laser Noise')

            plt.plot(self.voigt_x_values,
                     self.voigt,
                     label='Convolution',
                     color='green',
                     linewidth=2)
            plt.xlabel(r'$\Delta_f$ / Hz')
            plt.ylabel('Occurrences')

        elif source == 'line':
            fig, ax = plt.subplots()
            times = np.linspace(0, 1e6 / 60, 100)
            amps = [self.generate_line_noise_detuning(i) for i in times]
            # amps = [self.generate_line_noise_integrand(i) for i in times]
            plt.plot(times, amps)
            plt.xlabel(r'Time / $\mu s$')
            plt.ylabel(r'Magnetic field / $mG$')
        plt.grid()

    def get_simulation_consistency(self, iterations=101, datapoints=100):
        """
        Tests the consistency of results over a specified number of iterations
        and produces a histogram of the consistency measures.

        Parameters:
        iterations (int): Number of iterations for the contrast measurement.
        datapoints (int): Number of data points to collect.

        Returns:
        float: The standard deviation of the distribution of
        consistency measures.
        """
        # List to hold consistency measures
        conts = []

        # Generate consistency measures
        for i in range(datapoints):
            dims, contrasts, errors = qd.run_qudit_contrast_measurement(
                dimensions=[3], iterations=iterations, verbose=False)

            conts.append(1e2 * contrasts[0])

        # Print the consistency measures
        print(conts)

        # Plot the histogram using Sturges' rule for number of bins
        num_bins = int(np.ceil(np.log2(datapoints) +
                               1))  # Sturges' rule for bin count
        plt.hist(conts,
                 bins=num_bins,
                 density=True,
                 alpha=0.6,
                 color='g',
                 edgecolor='black')

        # Fit a Gaussian distribution to the data
        mu, std = norm.fit(conts)

        # Plot the Gaussian fit
        xmin, xmax = plt.xlim()  # Get the current x-axis limits
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)  # Probability density function
        plt.plot(x, p, 'k', linewidth=2)  # Black line for Gaussian fit
        title = r"Fit: $\mu$ = %.4f,  $\sigma$ = %.4f" % (mu, std)
        plt.title(title)

        # Set labels
        plt.xlabel('Consistency Measures')
        plt.ylabel('Density')
        plt.grid(True)

        # Print and return standard deviation of the distribution
        print(f"Standard Deviation of the distribution: {std:.4f} %.")
        return std


if __name__ == '__main__':
    start_time = time.time()

    qd = QuditUnitary(dimension=6, topology='Star', real_Ramsey_wait=0)

    # pulse_dict = qd.get_pulse_sequence()
    # print(pulse_dict)
    conts = []
    conts_err = []
    dimensions = np.arange(2, 25)
    dimensions = [16]
    for dim in dimensions:
        qd.dimension = dim
        # total_time, coherent_time = qd.get_full_experiment_runtime(dim)
        # print(f'Dimension {dim} total AWG time:', total_time, ' us.')
        # print(f'Dimension {dim} coherent pulse time:', coherent_time, ' us.')

        # qd.initialise_system()
        # evolved_vec, U1_evolved_vec = qd.evolve_qudit_Ramsey()

        dims, contrasts, errors = qd.run_qudit_contrast_measurement(
            dimensions=[dim], iterations=101)
        conts.append(contrasts[0])
        conts_err.append(errors[0])

    # breakpoint()
    plt.errorbar(dimensions, conts, yerr=errors)
    plt.grid()
    # phases, probs, U1_probs = qd.scan_phases()
    # qd.plot_result(phases, probs, just_ket_0=False)
    # # # qd.plot_result(phases, U1_probs, just_ket_0=False)

    # qd.run_simulation(phase_points=101, iterations=101, show_iterations=True)


    ###########################################################################
    # consistency of simulations

    # qd = QuditUnitary(
    #     dimension=3,
    #     line_amplitude_60=0,
    #     line_amplitude_180=0.0,
    #     line_phase_60=0,
    #     line_phase_180=0,
    #     line_offset=0,
    #     mag_field_noise=0.000273 / (np.sqrt(2) * np.pi),
    #     laser_noise_gaussian=0,
    #     laser_noise_lorentzian=0,
    #     # laser_noise_gaussian=91.82,
    #     # laser_noise_lorentzian=1e6 / (2 * np.pi * 2065.9),
    #     pulse_time_miscalibration=0,
    #     freq_miscalibration=0,
    #     real_Ramsey_wait=300)

    # qd.get_simulation_consistency(datapoints=100,iterations=101)

    # # looking at the 3-point calibration outputs
    # transition = [-1, 4, -3]
    # # transition = [0, 2, 0]
    # qd.run_qubit_Ramsey_simulation(
    #     iterations=250,  # same as in exp calibrations
    #     transition=transition,
    #     verbose=True)

    ###########################################################################
    # checking noise is generated properly

    # qd.plot_noise(source='laser')
    # qd.plot_noise(source='magnetic')
    # qd.plot_noise(source='line')

    plt.show()

    print("--- %s seconds ---" % (time.time() - start_time))
