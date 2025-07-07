import ast
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.special import voigt_profile
from tqdm import tqdm

from calibration_analysis import Calibrations
from plot_utils import nice_fonts, set_size

mpl.rcParams.update(nice_fonts)
np.set_printoptions(suppress=True)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = colors * 4


class Simulator(Calibrations):

    def __init__(
        self,
        mappings: dict = {0: '0'},
        init_state: list = ['0'],
        couplings: list = [(0, 1)],
        phases: list = [0],
        angles: list = [0],
        noise_flags=[0],
        # # line noise from magnets measurements
        # line_amplitude_60: float = 0.000255,  # Gauss
        # line_amplitude_180: float = 0.000090,  # Gauss
        # line_phase_60: float = -0.575,  # radians
        # line_phase_180: float = -1.455,  # radians
        # line_offset: float = 0.00022346854601197586,  # Gauss
        # mag_field_noise: float = 0.000064 / (np.sqrt(2) * np.pi),  # Gauss
        # laser_noise_gaussian: float = 1e6 / (2 * np.pi * 1950),  # Hz
        # laser_noise_lorentzian: float = 1e6 / (2 * np.pi * 2065.9),  # Hz
        # pulse_time_miscalibration: float = 0.0177,  # %
        # pulse_time_variation: float = 0.0261,  # %
        # freq_miscalibration: float = 124.94,  # Hz
        # best in class values
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
        # no noise settings below for troubleshooting
        # line_amplitude_60: float = 0,
        # line_amplitude_180: float = 0,
        # line_phase_60: float = 0,
        # line_phase_180: float = 0,
        # line_offset: float = 0,
        # mag_field_noise: float = 0,
        # laser_noise_gaussian: float = 0,
        # laser_noise_lorentzian: float = 0,
        # pulse_time_miscalibration: float = 0,
        # freq_miscalibration: float = 0,
    ):

        super().__init__()

        # system parameters
        self.mappings = mappings
        self.init_state = init_state

        # evolution parameters
        self.couplings = couplings
        self.phases = phases
        self.angles = angles
        self.noise_flags = noise_flags

        # noise parameters
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

        # transition sensitivities in MHz/G
        self.transition_sensitivities = np.loadtxt(
            'quick_reference_arrays/transition_sensitivities.txt',
            delimiter=',')
        self.transition_pitimes = np.loadtxt(
            'quick_reference_arrays/transition_pitimes.txt', delimiter=',')
        self.transition_strengths = np.loadtxt(
            'quick_reference_arrays/transition_strengths.txt', delimiter=',')

    def get_algorithm(self, verbose: bool = False) -> dict:

        # helper to translate couplings to transitions
        def couplings_to_transitions(mappings, couplings):
            transitions = []
            for tuple_idx, t in enumerate(couplings):
                if t[0] == 0:
                    trip = [
                        ast.literal_eval(mappings[t[0]]),
                        ast.literal_eval(mappings[t[1]])[0],
                        ast.literal_eval(mappings[t[1]])[1]
                    ]
                    transitions.append(trip)

                elif t[1] == 0:
                    trip = [
                        ast.literal_eval(mappings[t[1]]),
                        ast.literal_eval(mappings[t[0]])[0],
                        ast.literal_eval(mappings[t[0]])[1]
                    ]
                    transitions.append(trip)

            return np.array(transitions)

        self.transitions = couplings_to_transitions(self.mappings,
                                                    self.couplings)

        # Get all states participating for the simulation
        bases = []
        for i, ket_num in enumerate(self.mappings):
            if type(ast.literal_eval(self.mappings[i])) == int:
                bases.append([0, ast.literal_eval(self.mappings[i])])
            elif type(ast.literal_eval(self.mappings[i])) == list:
                bases.append(ast.literal_eval(self.mappings[i]))
        self.basis_order = np.array(bases)

        probe_trans = [[
            ast.literal_eval(self.mappings[0]),
            ast.literal_eval(self.mappings[t])[0],
            ast.literal_eval(self.mappings[t])[1]
        ] for t in range(len(self.mappings))[1:]]

        dict_evolution = {
            'init_state': self.init_state,
            'transitions': self.transitions,
            'angles': self.angles,
            'phases': self.phases,
            'noise_flags': self.noise_flags,
            'probes': probe_trans,
        }

        if verbose:
            print('Initial State:', self.init_state)
            print('State Mapping:', self.mappings)
            # print('States Involved:\n', self.basis_order)

        self.dict_evolution = dict_evolution

        return dict_evolution

    def initialise_system(self) -> np.ndarray:
        self.get_algorithm()
        self.dimension = len(self.mappings)
        init_vec = np.zeros(self.dimension)

        init_index = next((key for key, value in self.mappings.items()
                           if value == self.init_state), None)

        init_vec[init_index] = 1  # assumed perfect initialisation
        self.init_vec = init_vec

        return init_vec

    # noise simulation methods
    def generate_magnetic_noise(self) -> float:
        '''Generates a single instance of magnetic field value (as a change
        from the initial value) according to a normal distribution.'''
        return np.random.normal(loc=0.0, scale=self.mag_field_noise)

    def generate_voigt(self,
                       x=np.linspace(-1000, 1000, int(1e6)),
                       verbose=False):
        '''Generates the Voigt profile., for laser noise.'''

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
        '''This is the integral of the line signal at some time t,
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

        # def integrand(t: float) -> float:
        #     return self.generate_line_noise_detuning(t)

        # val, _ = quad(integrand, 0, t)

        val = (-(self.line_amplitude_60 / (2 * np.pi * 60 * 1e-6)) *
               np.cos(2 * np.pi * 60 * t * 1e-6 + self.line_phase_60) -
               (self.line_amplitude_180 / (2 * np.pi * 180 * 1e-6)) *
               np.cos(2 * np.pi * 180 * t * 1e-6 + self.line_phase_180) +
               self.line_offset * t) - (
                   -(self.line_amplitude_60 /
                     (2 * np.pi * 60 * 1e-6)) * np.cos(self.line_phase_60) -
                   (self.line_amplitude_180 /
                    (2 * np.pi * 180 * 1e-6)) * np.cos(self.line_phase_180))

        # need to again re-centre the integral as well as detuning value above
        # so it starts at 0
        val -= ((self.line_amplitude_60) * np.sin(self.line_phase_60) +
                (self.line_amplitude_180) * np.sin(self.line_phase_180) +
                self.line_offset) * t

        return val

    # unitary evolution builder
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

        hamiltonian[s12_index,
                    d52_index] = (trans_Omega / 2) * np.exp(1j * phase)
        hamiltonian[d52_index,
                    s12_index] = np.conjugate(hamiltonian[s12_index,
                                                          d52_index])

        hamiltonian[s12_index, s12_index] = 0
        hamiltonian[d52_index, d52_index] = -real_detuning

        unitary = sc.linalg.expm(-1j * hamiltonian * time)

        return unitary

    def evolve_state(self, verbose: bool = False):
        self.initialise_system()

        # define the value of the magnetic field change for this run
        self.per_run_mag_noise = self.generate_magnetic_noise()
        self.per_run_laser_noise = self.generate_laser_noise()
        self.per_run_freq_miscal = self.generate_frequency_miscalibration()
        self.per_run_pulse_miscal = self.generate_pulse_time_miscalibration()
        self.per_run_pulse_var = self.generate_pulse_time_variation()

        total_experiment_time = 0

        # U1 evolution
        unitaries = np.empty(
            (len(self.dict_evolution['transitions']), len(
                self.basis_order), len(self.basis_order)),
            dtype=complex)

        for idx, trans in enumerate(self.dict_evolution['transitions']):
            table_idx = self.convert_triplet_index(trans)
            trans_pitime = self.transition_pitimes[tuple(table_idx)]  # us

            trans_sens = self.transition_sensitivities[tuple(
                table_idx)]  # MHz/G

            # set pulse time for this transition
            angle = self.dict_evolution['angles'][idx]  # need angle first
            time = trans_pitime * angle

            if verbose:
                print(
                    2 * np.pi * trans_sens *
                    self.generate_line_noise_integrand(total_experiment_time))
                print('trans sens', trans_sens)
                print('exp time', total_experiment_time)

            # first put together whether pulses are going from S->D or D->S
            phase = np.pi * self.dict_evolution['phases'][idx]

            if self.dict_evolution['noise_flags'][idx] == 1:
                # add in noise sources, starting with line signal (60/180 Hz)
                phase += (
                    2 * np.pi * trans_sens *
                    self.generate_line_noise_integrand(total_experiment_time))

                if idx == 0:
                    real_detuning = (2 * np.pi * trans_sens *
                                     self.generate_line_noise_detuning(
                                         (1e6 / 60) * np.random.random()))
                else:
                    real_detuning = (2 * np.pi * trans_sens *
                                     self.generate_line_noise_detuning(
                                         total_experiment_time))

                # print(2 * np.pi * trans_sens *
                #     self.generate_line_noise_integrand(total_experiment_time))

                # now the magnetic field Gaussian noise
                real_detuning += 2 * np.pi * trans_sens * self.per_run_mag_noise
                phase += (2 * np.pi * trans_sens * self.per_run_mag_noise *
                          total_experiment_time)

                # then the laser noise (taken from the [0,2,0] decay fit)
                real_detuning += 2 * np.pi * 1e-6 * self.per_run_laser_noise
                phase += (2 * np.pi * 1e-6 * self.per_run_laser_noise *
                          total_experiment_time)

                # next the frequency miscalibration
                real_detuning += 2 * np.pi * 1e-6 * self.per_run_freq_miscal
                phase += (2 * np.pi * 1e-6 * self.per_run_freq_miscal *
                          total_experiment_time)

                # finally, if there is a constant detuning add it here
                # phase += (2 * np.pi * 1e-6 * self.input_detuning *
                #         total_experiment_time)

                # print(phase)

            new_unit = self.build_unitary(
                transition=trans,
                time=time,
                phase=phase,
                real_detuning=real_detuning,
                pulse_miscal=self.per_run_pulse_miscal,
                pulse_var=self.per_run_pulse_var)
            unitaries[idx, :, :] = new_unit

            # if idx == 0:
            #     pass
            # else:
            #     total_experiment_time += time
            total_experiment_time += time

        total_unitary = unitaries[0, :, :]
        for i in range(1, len(self.dict_evolution['transitions'])):
            total_unitary = np.matmul(unitaries[i, :, :], total_unitary)

        output_vec = total_unitary @ self.init_vec

        final_state = np.square((np.abs(output_vec)))

        return final_state


if __name__ == '__main__':
    start_time = time.time()

    def get_oracle(secret=[0, 1]):
        oracle_couplings = []
        oracle_phases = []
        oracle_angles = []

        oracle_phase_flips = [0] + [
            1 if np.dot([int(x)
                         for x in format(i, f'0{len(secret)}b')], secret) %
            2 else 0 for i in range(2**len(secret))
        ]
        for idx, flipper in enumerate(oracle_phase_flips):
            if flipper == 1:
                oracle_couplings.append((0, idx))
                oracle_phases.append(0)
                oracle_angles.append(2)
            else:
                pass
        oracle_noise_flags = [0] * len(oracle_couplings)

        result = [
            oracle_couplings, oracle_phases, oracle_angles, oracle_noise_flags
        ]
        return result

    init_state = '[2, -1]'
    mappings = {
        0: '-2',
        1: '[2, -1]',
        3: '[3, -2]',
        2: '[4, -4]',
        4: '[3, -3]'
    }

    # define evolution in terms of labeled states
    # Had_couplings = [(0, 2), (0, 3), (0, 1), (0, 4), (0, 1), (0, 3), (0, 2)]
    # Had_phases = [1, 1, 0, 1, 0, 1, 1]
    # Had_angles = [
    #     1, 0.5, 2.0 * np.arcsin(np.sqrt(1 / 3)) / np.pi, 2 / 3,
    #     2.0 * np.arcsin(np.sqrt(1 / 3)) / np.pi, 0.5, 1
    # ]
    # Had_noise_flags = [1] * len(Had_couplings)

    # secret = [0, 1]
    # oracle = get_oracle(secret=secret)

    # Had_couplings = Had_couplings + oracle[0] + Had_couplings
    # Had_phases = Had_phases + oracle[1] + Had_phases
    # Had_angles = Had_angles + oracle[2] + Had_angles
    # Had_noise_flags = Had_noise_flags + oracle[3] + Had_noise_flags

    # sim = Simulator(
    #     init_state=init_state,
    #     mappings=mappings,
    #     couplings=Had_couplings,
    #     phases=Had_phases,
    #     angles=Had_angles,
    #     noise_flags=Had_noise_flags,
    # )

    # sim.get_algorithm(verbose=True)
    # sim.initialise_system()
    # sim.evolve_state()

    Had_couplings = [(0, 2), (0, 3), (0, 1), (0, 4), (0, 1), (0, 3), (0, 2)]

    # # "swapped" Hadamard - switched 01 and 10 states
    # Had_phases = [1, 1, 0, 1, 0, 1, 1]
    # Had_angles = [
    #     1, 0.5, 2.0 * np.arcsin(np.sqrt(1 / 3)) / np.pi, 2 / 3,
    #     2.0 * np.arcsin(np.sqrt(1 / 3)) / np.pi, 0.5, 1
    # ]

    # "true" Hadamard
    Had_phases = [1, 0, 1, 0, 1, 0, 0]
    Had_angles = [
        1, 1.5, 2.0 * np.arcsin(np.sqrt(1 / 3)) / np.pi, 2 / 3,
        2.0 * np.arcsin(np.sqrt(1 / 3)) / np.pi, 0.5, 1
    ]

    Had_noise_flags = [1] * len(Had_couplings)

    # sim = Simulator(
    #     init_state=init_state,
    #     mappings=mappings,
    #     couplings=Had_couplings,
    #     phases=Had_phases,
    #     angles=Had_angles,
    #     noise_flags=Had_noise_flags,
    # )


    def run_simulation(class_Simulator,
                       shots: int = 100,
                       secrets=[[0, 0], [0, 1], [1, 0], [1, 1]]):

        sim = class_Simulator()

        sim.init_state = init_state
        sim.mappings = mappings

        results = np.zeros((len(sim.mappings), len(secrets), shots))
        for sec_idx, secret in enumerate(secrets):

            # generate the pulse sequence for each possible hidden string
            oracle = get_oracle(secret=secret)

            sec_Had_couplings = Had_couplings + oracle[0] + Had_couplings
            sec_Had_phases = Had_phases + oracle[1] + Had_phases
            sec_Had_angles = Had_angles + oracle[2] + Had_angles
            sec_Had_noise_flags = Had_noise_flags + oracle[3] + Had_noise_flags

            sim.couplings = sec_Had_couplings
            sim.phases = sec_Had_phases
            sim.angles = sec_Had_angles
            sim.noise_flags = sec_Had_noise_flags

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

        hidden_strings = [r'\{00\}', r'\{10\}', r'\{01\}', r'\{11\}']
        sim_result = run_simulation(class_Simulator, shots=shots)
        data = sim_result[0][1:]
        errors = sim_result[1][1:]
        # data_rate = 1 - sim_result[0][0]

        results_dict = {
            '00': np.round(np.max(100 * data[0]), 2),
            '01': np.round(np.max(100 * data[1]), 2),
            '10': np.round(np.max(100 * data[2]), 2),
            '11': np.round(np.max(100 * data[3]), 2),
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
            ax.set_xticklabels(xtick_labels,
                               rotation=0,
                               ha='center',
                               fontsize=12)

            # # Create the list of y tick labels
            ytick_labels = ['{' + f'{i}' + '}' for i in hidden_strings]
            y_positions = np.arange(len(hidden_strings))  # Get number of runs
            ax.set_yticks(y_positions)
            ax.set_yticklabels(ytick_labels,
                               rotation=0,
                               ha='right',
                               fontsize=12)

            # Set labels and title
            ax.set_ylabel('Found String')
            ax.set_xlabel('Hidden String')
            # date = date_time.split('_')
            # date_obj = datetime.strptime(date[0], '%Y%m%d')
            # plot_date = date_obj.strftime('%B %d, %Y')
            ax.set_title('2-polyqubit: Bernstein-Vazirani Result')

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
                                    2,
                                    figsize=set_size('half'),
                                    sharex=True,
                                    sharey=True)

            plt.suptitle('2-polyqubit: Bernstein-Vazirani Result')
            # Create bar graphs for each row with error bars
            for i in range(4):
                # Create bar plot and capture the bars for annotation
                bars = axs[i // 2, i % 2].bar(range(len(data[i])),
                                              data[i],
                                              yerr=errors[i],
                                              capsize=5)

                axs[i // 2, i % 2].set_xticks(range(len(data[i])))
                axs[i // 2, i % 2].set_xticklabels(hidden_strings)
                axs[i // 2, i % 2].legend([f'Input Key = {hidden_strings[i]}'])
                axs[i // 2, i % 2].grid()

                # Annotate each bar with its percentage value
                for bar in bars:
                    yval = bar.get_height()  # Get the height of each bar
                    if yval > 0.25:
                        axs[i // 2,
                            i % 2].text(bar.get_x() + bar.get_width() / 2,
                                        yval * 0.5,
                                        f'{100 * yval:.2f}%',
                                        ha='center',
                                        va='center',
                                        color='black')

            axs[0, 0].set_ylabel('Probabilities')
            axs[1, 0].set_ylabel('Probabilities')
            axs[1, 0].set_xlabel('Measured Keys')
            axs[1, 1].set_xlabel('Measured Keys')

        if plot_3d:
            norm = Normalize(vmin=data.min(), vmax=data.max())
            colors = plt.cm.viridis(data.flatten() / float(data.max()))

            # Create a 3D figure
            fig = plt.figure(figsize=set_size('full'))
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
            ax.set_xticklabels(xtick_labels,
                               rotation=0,
                               ha='center',
                               fontsize=12)

            # # Create the list of y tick labels
            ytick_labels = ['{' + f'{i}' + '}' for i in hidden_strings]
            y_positions = np.arange(len(hidden_strings))  # Get number of runs
            ax.set_yticks(y_positions)
            ax.set_yticklabels(ytick_labels,
                               rotation=0,
                               ha='right',
                               fontsize=12)

            # Set labels and title
            ax.set_ylabel('Found String')
            ax.set_xlabel('Hidden String')
            # date = date_time.split('_')
            # date_obj = datetime.strptime(date[0], '%Y%m%d')
            # plot_date = date_obj.strftime('%B %d, %Y')
            ax.set_title('2-polyqubit: Bernstein-Vazirani Result')

    plot_result(Simulator, shots=100, plot_2d=True, plot_3d=True)
    plt.show()

    print("--- %s seconds ---" % (time.time() - start_time))
