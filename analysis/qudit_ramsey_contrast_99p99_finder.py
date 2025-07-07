import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from class_utils import Utils
from plot_utils import nice_fonts
from qudit_ramsey_unitary import QuditUnitary

# from testing_new_qudit_unitary import QuditUnitary

find_errors = Utils.find_errors

mpl.rcParams.update(nice_fonts)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def get_ACline_noise_value(dimensions: np.ndarray = np.arange(2, 19),
                           topology: str = 'All',
                           verbose: bool = False):

    noises = []
    line_amplitude_60 = 0.0225  # Gauss
    line_amplitude_180 = 0.00009  # Gauss
    for idx, dim in enumerate(dimensions):
        while True:
            qd = QuditUnitary(
                topology=topology,
                line_amplitude_60=line_amplitude_60,  # Gauss
                line_amplitude_180=line_amplitude_180,  # Gauss
                line_phase_60=-0.575,  # radians
                line_phase_180=-1.455,  # radians
                line_offset=0.00022346854601197586,  # Gauss
                mag_field_noise=0,  # Gauss
                laser_noise_gaussian=0,
                laser_noise_lorentzian=0,
                pulse_time_miscalibration=0,
                freq_miscalibration=0)
            sim_dims, sim_res, sim_err = qd.run_qudit_contrast_measurement(
                dimensions=[dim], iterations=512)

            # Check if the output is below the desired threshold
            if sim_res < 0.9999:
                line_amplitude_60 = line_amplitude_60 / 2
                line_amplitude_180 = line_amplitude_180 / 2
            elif sim_res > 0.99991:
                line_amplitude_60 *= 1.25  # Increase input parameter
                line_amplitude_180 *= 1.25  # Increase input parameter
            else:
                break  # Desired output is within the acceptable range

        noises.append(np.sqrt(line_amplitude_60**2 + line_amplitude_180**2))

    noises = np.array(noises)
    return noises


def get_mag_noise_value(dimensions: np.ndarray = np.arange(2, 19),
                        topology: str = 'All',
                        verbose: bool = False):

    noises = []
    init_noise = 0.000049 / (np.sqrt(2) * np.pi)
    for idx, dim in enumerate(dimensions):
        while True:
            qd = QuditUnitary(
                topology=topology,
                line_amplitude_60=0,
                line_amplitude_180=0,
                line_phase_60=0,
                line_phase_180=0,
                line_offset=0,
                mag_field_noise=init_noise,  # Gauss
                laser_noise_gaussian=0,
                laser_noise_lorentzian=0,
                pulse_time_miscalibration=0,
                freq_miscalibration=0)
            sim_dims, sim_res, sim_err = qd.run_qudit_contrast_measurement(
                dimensions=[dim], iterations=512)

            # Check if the output is below the desired threshold
            if sim_res < 0.9999:
                init_noise = init_noise / 2
            elif sim_res > 0.99991:
                init_noise *= 1.25  # Increase input parameter
            else:
                break  # Desired output is within the acceptable range

        noises.append(init_noise)

    noises = np.array(noises)
    return noises


def get_pulsetime_error_value(dimensions: np.ndarray = np.arange(2, 19),
                              topology: str = 'All',
                              verbose: bool = False):

    noises = []
    pulse_time_miscalibration = 0.0177  # %
    for idx, dim in enumerate(dimensions):
        while True:
            qd = QuditUnitary(
                topology=topology,
                line_amplitude_60=0,
                line_amplitude_180=0,
                line_phase_60=0,
                line_phase_180=0,
                line_offset=0,
                mag_field_noise=0,
                laser_noise_gaussian=0,
                laser_noise_lorentzian=0,
                pulse_time_miscalibration=pulse_time_miscalibration,
                freq_miscalibration=0)
            sim_dims, sim_res, sim_err = qd.run_qudit_contrast_measurement(
                dimensions=[dim], iterations=512)

            # Check if the output is below the desired threshold
            if sim_res < 0.9999:
                pulse_time_miscalibration = pulse_time_miscalibration / 2
            elif sim_res > 0.99991:
                pulse_time_miscalibration *= 1.25  # Increase input parameter
            else:
                break  # Desired output is within the acceptable range

        noises.append(pulse_time_miscalibration)

    noises = np.array(noises)
    return noises


def get_freqcalibration_error_value(dimensions: np.ndarray = np.arange(2, 19),
                                    topology: str = 'All',
                                    verbose: bool = False):

    noises = []
    freq_miscalibration = 124.94  # Hz
    for idx, dim in enumerate(dimensions):
        while True:
            qd = QuditUnitary(topology=topology,
                              line_amplitude_60=0,
                              line_amplitude_180=0,
                              line_phase_60=0,
                              line_phase_180=0,
                              line_offset=0,
                              mag_field_noise=0,
                              laser_noise_gaussian=0,
                              laser_noise_lorentzian=0,
                              pulse_time_miscalibration=0,
                              freq_miscalibration=freq_miscalibration)
            sim_dims, sim_res, sim_err = qd.run_qudit_contrast_measurement(
                dimensions=[dim], iterations=512)

            # Check if the output is below the desired threshold
            if sim_res < 0.9999:
                freq_miscalibration = freq_miscalibration / 2
            elif sim_res > 0.99991:
                freq_miscalibration *= 1.25  # Increase input parameter
            else:
                break  # Desired output is within the acceptable range

        noises.append(freq_miscalibration)

    noises = np.array(noises)
    return noises


def get_calibration_error_value(dimensions: np.ndarray = np.arange(2, 19),
                                topology: str = 'All',
                                verbose: bool = False):

    noises = []
    pulse_time_miscalibration = 0.0177  # %
    freq_miscalibration = 124.94  # Hz
    for idx, dim in enumerate(dimensions):
        while True:
            qd = QuditUnitary(topology=topology,
                              line_amplitude_60=0,
                              line_amplitude_180=0,
                              line_phase_60=0,
                              line_phase_180=0,
                              line_offset=0,
                              mag_field_noise=0,
                              laser_noise_gaussian=0,
                              laser_noise_lorentzian=0,
                              pulse_time_miscalibration=0,
                              freq_miscalibration=freq_miscalibration)
            sim_dims, sim_res, sim_err = qd.run_qudit_contrast_measurement(
                dimensions=[dim], iterations=1028)

            print('Value is ', freq_miscalibration)
            # Check if the output is below the desired threshold
            if sim_res < 0.9999:
                freq_miscalibration = freq_miscalibration / 2
                # pulse_time_miscalibration = pulse_time_miscalibration / 2
            elif sim_res > 0.999905:
                freq_miscalibration *= 1.25  # Increase input parameter
                # pulse_time_miscalibration *= 1.25  # Increase input parameter
            else:
                break  # Desired output is within the acceptable range

        noises.append(np.sqrt(freq_miscalibration**2))
        # noises.append(
        #     np.sqrt(freq_miscalibration**2 + pulse_time_miscalibration**2))

    noises = np.array(noises)
    return noises


def get_laser_noise_value(dimensions: np.ndarray = np.arange(2, 19),
                          topology: str = 'All',
                          verbose: bool = False):

    noises = []
    init_gaussian = 1e6 / (2 * np.pi * 1950)  # Hz
    init_lorentzian = 1e6 / (2 * np.pi * 2065.9)  # Hz
    for idx, dim in enumerate(dimensions):
        while True:
            qd = QuditUnitary(topology=topology,
                              line_amplitude_60=0,
                              line_amplitude_180=0,
                              line_phase_60=0,
                              line_phase_180=0,
                              line_offset=0,
                              mag_field_noise=0,
                              laser_noise_gaussian=init_gaussian,
                              laser_noise_lorentzian=init_lorentzian,
                              pulse_time_miscalibration=0,
                              freq_miscalibration=0)
            sim_dims, sim_res, sim_err = qd.run_qudit_contrast_measurement(
                dimensions=[dim], iterations=512)

            # Check if the output is below the desired threshold
            if sim_res < 0.9999:
                init_gaussian = init_gaussian / 2
                init_lorentzian = init_lorentzian / 2
            elif sim_res > 0.99991:
                init_gaussian *= 1.25  # Increase input parameter
                init_lorentzian *= 1.25  # Increase input parameter
            else:
                break  # Desired output is within the acceptable range

        noises.append(np.sqrt(init_gaussian**2 + init_lorentzian**2))

    noises = np.array(noises)
    return noises


if __name__ == '__main__':
    start_time = time.time()

    # threshold = get_threshold(plot_threshold_histogram=True)

    redo_comp = False
    dimensions = np.arange(2, 19, 1)
    # dimensions = [9,14]
    # dimensions = [2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 16, 18]

    # get_single_contrast(dimension=6, U1_only_flag=False,verbose=True)

    # plot_dimensional_contrast(dimensions=dimensions,
    #                           topology='Star',
    #                           plot_simulation=True,
    #                           plot_times_comparison=True,
    #                           plot_U1_populations=False,
    #                           verbose=False)

    if redo_comp:
        # mag_noises = get_mag_noise_value(dimensions=dimensions)
        # breakpoint()
        # laser_noises = get_laser_noise_value(dimensions=dimensions)
        # breakpoint()
        # pulsetime_noises = get_pulsetime_error_value(dimensions=dimensions)
        # breakpoint()
        # freqcal_noises = get_freqcalibration_error_value(dimensions=dimensions)
        # breakpoint()
        calib_noises = get_calibration_error_value(dimensions=dimensions)
        breakpoint()
        # AC_line_noises = get_ACline_noise_value(dimensions=dimensions)
        # breakpoint()

    else:
        mag_noises = np.array([
            0.00005839, 0.00003649, 0.00003399, 0.00001439, 0.0000072,
            0.00000549, 0.00000419, 0.00000363, 0.00000068, 0.00000049,
            0.0000006, 0.00000055, 0.00000043, 0.00000043, 0.00000047,
            0.00000046, 0.00000036
        ])

        laser_noises = np.array([
            56.11706014, 4.98419618, 1.68871135, 1.96591875, 0.82100598,
            1.86675147, 0.93337574, 0.37698746, 0.43887116, 0.5357314,
            0.21336734, 0.15160661, 0.11295573, 0.41093055, 0.41093055,
            0.12662775, 0.12662775
        ])
        calib_noises = np.array([
            227.26453608, 71.02016752, 54.18408777, 41.33917829, 32.61086161,
            30.37123159, 27.62247422, 23.67363510, 22.84877515, 20.293789,
            18.45709358, 16.78662882, 14.1966572, 12.80718141, 12.21387998,
            12.21387998, 8.08246778
        ])
        AC_line_noises = np.array([
            0.00932522, 0.00670558, 0.00399683, 0.00113597, 0.00028399,
            0.0001775, 0.00011746, 0.00009176, 0.00000267, 0.00000163,
            0.00000194, 0.00000152, 0.00000116, 0.00000103, 0.00000096,
            0.00000075, 0.00000063
        ])

    plt.plot(dimensions,
             mag_noises / mag_noises[0],
             label='Magnetic Noise',
             marker='o',
             color=colors[0])

    # plt.fill_between(
    #     dimensions,
    #     (mag_noises / mag_noises[0]) - 0.2 * (mag_noises[0] / mag_noises[0]),
    #     (mag_noises / mag_noises[0]) + 0.2 * (mag_noises[0] / mag_noises[0]),
    #     color=colors[0],
    #     alpha=0.5,
    #     label=None)

    plt.plot(dimensions,
             laser_noises / laser_noises[0],
             label='Laser Noise',
             marker='s',
             color=colors[1])
    plt.plot(dimensions,
             calib_noises / calib_noises[0],
             label='Calibration Error',
             marker='^',
             color=colors[2])
    plt.plot(dimensions,
             AC_line_noises / AC_line_noises[0],
             label='A/C Line Signal Noise',
             marker='v',
             color=colors[3])
    # plt.plot(dimensions,
    #          pulsetime_noises / pulsetime_noises[0],
    #          label='Pulse Time Error')
    # plt.plot(dimensions,
    #          freqcal_noises / freqcal_noises[0],
    #          label='Frequency Error')

    plt.title(r'Noise Values for $10^{-4}$ Error per Dimension')
    plt.ylabel(r'Noise magnitude (normalised to $d=2$ level)')
    plt.xlabel('Qudit dimension ($d$)')
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.show()
    print("--- %s seconds ---" % (time.time() - start_time))
