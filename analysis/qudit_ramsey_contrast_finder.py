import time
from logging import raiseExceptions

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

from plot_utils import nice_fonts, set_size
from qudit_ramsey_unitary import QuditUnitary

mpl.rcParams.update(nice_fonts)
np.set_printoptions(suppress=True)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = colors * 4


class ContrastFinder:

    def __init__(self, error_goal: float = 0.9999):
        self.error_goal = error_goal
        self.noise_savefiles_dict = {
            'Line Signal 60 Hz': 'line60',
            'Line Signal 180 Hz': 'line180',
            'Magnetic Field': 'mag_field',
            'Gaussian Laser Noise': 'laser_gaussian',
            'Lorentzian Laser Noise': 'laser_lorentzian',
            'Frequency Miscalibration': 'freq_miscal',
            'Pulse Time Miscalibration': 'pulsetime_miscal',
        }

        self.noise_units_dict = {
            'Line Signal 60 Hz': 'G',
            'Line Signal 180 Hz': 'G',
            'Magnetic Field': 'G',
            'Gaussian Laser Noise': 'Hz',
            'Lorentzian Laser Noise': 'Hz',
            'Frequency Miscalibration': 'Hz',
            'Pulse Time Miscalibration': '\%',
        }

    def contrast_finder(self,
                        dimensions: np.ndarray = np.arange(3, 10, 1),
                        error_type: str = 'Line Signal 60 Hz',
                        verbose: bool = False):
        '''Takes a single error source as input and finds the highest value
        for that error source that is still able to return a contrast over
        99.99% (ie. error smaller than 10^-4).
        '''

        # load a dummy instance of the QuditUnitary class to get
        # default noise values in order to set bounds on fitting
        qd = QuditUnitary()

        if error_type == 'Line Signal 60 Hz':
            noise_unit = 'Gauss'
            error_guess = qd.line_amplitude_60
        elif error_type == 'Line Signal 180 Hz':
            noise_unit = 'Gauss'
            error_guess = qd.line_amplitude_180
        elif error_type == 'Magnetic Field':
            noise_unit = 'Gauss'
            error_guess = qd.mag_field_noise
        elif error_type == 'Gaussian Laser Noise':
            noise_unit = 'Hz'
            error_guess = qd.laser_noise_gaussian
        elif error_type == 'Lorentzian Laser Noise':
            noise_unit = 'Hz'
            error_guess = qd.laser_noise_lorentzian
        elif error_type == 'Frequency Miscalibration':
            noise_unit = 'Hz'
            error_guess = qd.freq_miscalibration
        elif error_type == 'Pulse Time Miscalibration':
            noise_unit = 'us'
            error_guess = qd.pulse_time_miscalibration
        else:
            raiseExceptions('Invalid error type.')

        savefile_name = self.noise_savefiles_dict[error_type]

        error_rates = []
        error_rates_uncertainty = []
        noise_levels = []

        for idx, dim in enumerate(dimensions):

            def objective(error_level):

                error_level = np.absolute(error_level)
                # Initialise a noise-less instance of the
                qd = QuditUnitary(dimension=dim,
                                  line_amplitude_60=0,
                                  line_amplitude_180=0,
                                  line_phase_60=0,
                                  line_phase_180=0,
                                  line_offset=0.0,
                                  mag_field_noise=0,
                                  laser_noise_gaussian=0,
                                  laser_noise_lorentzian=0,
                                  pulse_time_miscalibration=0,
                                  freq_miscalibration=0,
                                  real_Ramsey_wait=0)

                if error_type == 'Line Signal 60 Hz':
                    qd.line_amplitude_60 = error_level
                    itera = 1
                elif error_type == 'Line Signal 180 Hz':
                    qd.line_amplitude_180 = error_level
                    itera = 1
                elif error_type == 'Magnetic Field':
                    qd.mag_field_noise = error_level
                    itera = 201
                elif error_type == 'Gaussian Laser Noise':
                    qd.laser_noise_gaussian = error_level
                    itera = 201
                elif error_type == 'Lorentzian Laser Noise':
                    qd.laser_noise_lorentzian = error_level
                    itera = 201
                elif error_type == 'Frequency Miscalibration':
                    qd.freq_miscalibration = error_level
                    itera = 201
                elif error_type == 'Pulse Time Miscalibration':
                    qd.pulse_time_miscalibration = error_level
                    itera = 201
                else:
                    raiseExceptions('Invalid error type.')

                dimension, contrast, error = qd.run_qudit_contrast_measurement(
                    [dim], iterations=itera)

                # if contrast[0] >= self.error_goal:
                #     return contrast[0] - self.error_goal
                # else:
                #     return 1

                return contrast[0] - self.error_goal

            initial_guess = 0

            print(f'Starting noise level finder for dimension = {dim}')
            # if (error_type == 'Line Signal 60 Hz'
            #         or error_type == 'Line Signal 180 Hz'):
            #     optimisation_params = sc.optimize.minimize(
            #         objective,
            #         initial_guess,
            #         # bounds=bounds,
            #         callback=callback,
            #         # options={'maxiter': 10},
            #         tol=1e-5,
            #         method='Nelder-Mead')
            # else:
            # use a different minimisation method for noisy outcomes
            # optimisation_params = sc.optimize.dual_annealing(
            #     objective,
            #     bounds,
            #     initial_temp=1e4,
            #     callback=callback_anneal,
            #     maxiter=300)

            # writing a new approach to optimising the noise level

            # this will be a while loop aiming for a desired contrast loss
            # in this case, if the contrast is within 0.0001 of 99.99%, it will end
            diff_from_goal = float('inf')
            obj_output = 1
            threshold = 0.00001
            count = 0
            while diff_from_goal > threshold:
                if count%10==0:
                    print(f'Running instance of optimiser: {count}.')

                if obj_output >= 0:
                    error_guess *= 1.5
                elif obj_output < 0:
                    error_guess = error_guess / 2

                obj_output = objective(error_guess)

                if obj_output >= 0:
                    diff_from_goal = np.absolute(obj_output)
                elif obj_output < 0:
                    diff_from_goal = threshold * 10
                count += 1

            found_noise_level = error_guess

            qd = QuditUnitary(dimension=dim,
                              line_amplitude_60=0,
                              line_amplitude_180=0,
                              line_phase_60=0,
                              line_phase_180=0,
                              line_offset=0.0,
                              mag_field_noise=0,
                              laser_noise_gaussian=0,
                              laser_noise_lorentzian=0,
                              pulse_time_miscalibration=0,
                              freq_miscalibration=0,
                              real_Ramsey_wait=0)

            if error_type == 'Line Signal 60 Hz':
                qd.line_amplitude_60 = found_noise_level
                itera = 1
            elif error_type == 'Line Signal 180 Hz':
                qd.line_amplitude_180 = found_noise_level
                itera = 1
            elif error_type == 'Magnetic Field':
                qd.mag_field_noise = found_noise_level
                itera = 101
            elif error_type == 'Gaussian Laser Noise':
                qd.laser_noise_gaussian = found_noise_level
                itera = 101
            elif error_type == 'Lorentzian Laser Noise':
                qd.laser_noise_lorentzian = found_noise_level
                itera = 101
            elif error_type == 'Frequency Miscalibration':
                qd.freq_miscalibration = found_noise_level
                itera = 101
            elif error_type == 'Pulse Time Miscalibration':
                qd.pulse_time_miscalibration = found_noise_level
                itera = 101
            else:
                raiseExceptions('Invalid error type.')

            dimension, contrast, error = qd.run_qudit_contrast_measurement(
                [dim], iterations=itera)

            error_rates.append(contrast[0])
            error_rates_uncertainty.append(error[0])
            noise_levels.append(np.absolute(found_noise_level))

            if verbose:

                print(f'\nDimension: {dim}')
                print(f'Found noise value: {found_noise_level}')
                print(f'Contrast at found value: {contrast[0]}\n\n')

        fig, ax = plt.subplots(figsize=set_size(width='full'))
        plt.scatter(dimensions, noise_levels, label=f'Noise in {noise_unit}')
        plt.title(f'{error_type} Noise Level Needed for' +
                  r' $<10^{-4}$ Error')
        plt.xlabel(r'Dimension $(d)$')
        plt.ylabel(f'Noise Level in {noise_unit}')
        plt.yscale('log')
        plt.legend()

        # ax2 = plt.gca().twinx()  # gca() gets the current Axes instance
        # ax2.errorbar(dimensions,
        #                 error_rates,
        #                 yerr=error_rates_uncertainty,
        #                 label='Error Rates',
        #                 color='orange',
        #                 linewidth=2)
        # ax2.set_ylabel('Error rate for noise level.')
        # ax2.legend(loc='lower right')

        plt.grid()
        plt.tight_layout()
        plt.savefig('ramseys_unitary_errors_scaling/' +
                    f'{savefile_name}_noise_level_for_4nines.pdf')

        np.savetxt(
            'ramseys_unitary_errors_scaling/' +
            f'{savefile_name}_noise_levels.txt',
            np.vstack((dimensions, noise_levels)))

        return dimensions, noise_levels, noise_unit

    def plot_noise_levels(self,
                          dimensions: np.ndarray = np.arange(3, 10, 1),
                          noise_sources: list = ['Line Signal 60 Hz'],
                          use_save_files=False):

        all_noise_levels = []
        all_dimensions = []

        for idx, error_type in enumerate(noise_sources):

            if use_save_files:

                data = np.loadtxt(
                    'ramseys_unitary_errors_scaling/' +
                    f'{self.noise_savefiles_dict[error_type]}_noise_levels.txt'
                )

                # Separate the data into dimensions and noise levels
                dimensions = data[
                    0]  # Assuming the first row contains dimensions
                noise_levels = data[
                    1]  # Assuming the second row contains noise levels

                all_dimensions.append(dimensions)
                all_noise_levels.append(noise_levels)

            else:
                print(f'Finding noise levels for {error_type}..')
                dimensions, noise_levels, noise_unit = self.contrast_finder(
                    dimensions=dimensions, error_type=error_type)
                all_noise_levels.append(noise_levels)

        fig, ax = plt.subplots(figsize=set_size(width='full'))

        for idx, error_type in enumerate(noise_sources):
            plt.plot(
                all_dimensions[idx],
                all_noise_levels[idx],
                label=f'{error_type} / {self.noise_units_dict[error_type]}',
                linestyle='--',
                marker='o')

        plt.title('Noise Levels Needed for' + r'$<10^{-4}$ Error')
        plt.xlabel(r'Dimension $(d)$')
        plt.ylabel('Noise Level (varying units)')
        plt.yscale('log')
        plt.legend()
        plt.grid()


if __name__ == '__main__':
    start_time = time.time()

    cf = ContrastFinder()
    dimensions = np.arange(3, 15, 1)

    # cf.contrast_finder(dimensions=dimensions,
    #                    error_type='Line Signal 60 Hz',
    #                    verbose=True)

    # cf.contrast_finder(dimensions=dimensions,
    #                    error_type='Line Signal 180 Hz',
    #                    verbose=True)

    # cf.contrast_finder(dimensions=dimensions,
    #                    error_type='Magnetic Field',
    #                    verbose=True)

    cf.contrast_finder(dimensions=dimensions,
                       error_type='Gaussian Laser Noise',
                       verbose=True)

    cf.contrast_finder(dimensions=dimensions,
                       error_type='Lorentzian Laser Noise',
                       verbose=True)

    cf.contrast_finder(dimensions=dimensions,
                       error_type='Frequency Miscalibration',
                       verbose=True)

    # cf.contrast_finder(dimensions=dimensions,
    #                    error_type='Pulse Time Miscalibration',
    #                    verbose=True)

    noise_sources = [
        'Line Signal 60 Hz',
        'Line Signal 180 Hz',
        'Magnetic Field',
        'Gaussian Laser Noise',
        'Lorentzian Laser Noise',
        'Frequency Miscalibration',
        # 'Pulse Time Miscalibration', # this only affects bussed Ramseys
    ]

    cf.plot_noise_levels(noise_sources=noise_sources, use_save_files=True)

    plt.show()

    print("--- %s seconds ---" % (time.time() - start_time))
