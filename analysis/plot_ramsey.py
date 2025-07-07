import ast
# import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from mpl_toolkits.mplot3d import Axes3D


def plot_ramsey(head_file_name, plotting=True):

    threshold = 12

    head_file_path = ('../data/ramseys_qudits_data/')

    with open(head_file_path + head_file_name, 'r') as hf_name:
        # iterate through lines to get the last one, which gives a list of all
        # raw data filenames associated with the run
        for line in hf_name:
            pass
        file_names = line.strip()

    # need to convert file_names to be a true list of strings
    file_names = ast.literal_eval(file_names)

    with open(head_file_path + head_file_name, 'r') as hf_name:
        # get pulse times from the save txt data in head_file_name
        data = np.genfromtxt(hf_name, delimiter=',', skip_footer=5)
        wait_times = data[:, 0]

    file_path_raw = ('../data/ramseys_qudits_data/raw_data_fluorescence/')

    # get the number of states by looking at the first data file
    with open(file_path_raw + file_names[0], 'r') as filename:
        data = pd.read_csv(
            filename,
            delimiter="\[|\]|,|\"",  # noqa: W605
            engine='python',
            header=None).values
        data = data[:, 4:-19]
        dimension = np.shape(data)[1]

    full_data_array = np.zeros((dimension, len(wait_times)))

    for idx, file_name in enumerate(file_names):
        with open(file_path_raw + file_name, 'r') as filename:
            data = pd.read_csv(
                filename,
                delimiter="\[|\]|,|\"",  # noqa: W605
                engine='python',
                header=None).values
            data = data[:, 4:-19]
        binary_data = data > threshold

        # Initialize a mask to track the first occurrence of True in each row
        first_true_bin_array = np.zeros_like(binary_data, dtype=bool)

        # Iterate through each row
        num_fails = 0
        for row_idx in range(binary_data.shape[0]):
            row = binary_data[row_idx]
            first_true_idx = np.argmax(row)  # Find the index of the first True
            if sum(row) > 0:
                first_true_bin_array[row_idx, first_true_idx] = True
            else:
                num_fails += 1

        full_data_array[:, idx] = np.sum(first_true_bin_array, axis=0) / (
            np.shape(first_true_bin_array)[0] - num_fails)

    if plotting:
        # plot the results!
        fig, ax = plt.subplots(figsize=(8, 6))

        states = [
            r'$|S_{1/2},m=0\rangle$', r'$|D_{5/2},F=4,m=1\rangle$',
            r'$|D_{5/2},F=2,m=0\rangle$'
        ]
        # Plot the histogram
        for i in range(np.shape(full_data_array)[0]):
            ax.plot(wait_times,
                    full_data_array[i, :],
                    label=fr'|{i}$\rangle$ $\equiv$ ' + states[i])
        ax.set_title(fr'$d =$ {dimension} Qudit Ramsey Plot')
        ax.set_xlabel(r'Wait Time ($\mu s$)')
        ax.set_ylabel('Probabilities')
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid()
    return wait_times, full_data_array


if __name__ == '__main__':
    start_time = time.time()

    file_name = 'Ramsey_experiment_Dimension_3_20240619_1659.txt'
    file_name = 'Ramsey_experiment_Dimension_3_20240619_1833.txt'

    filenames = [
        'Ramsey_experiment_Dimension_3_20240619_1659.txt',
        # 'Ramsey_experiment_Dimension_3_20240619_1833.txt',
        # 'Ramsey_experiment_Dimension_3_20240620_1014.txt',
        # 'Ramsey_experiment_Dimension_3_20240625_1554.txt',
        # 'Ramsey_experiment_Dimension_3_20240625_1644.txt',
        # 'Ramsey_experiment_Dimension_2_20240916_1536.txt',
    ]

    for file_name in filenames:
        plot_ramsey(file_name)
    plt.show()

    print("--- %s seconds ---" % (time.time() - start_time))
