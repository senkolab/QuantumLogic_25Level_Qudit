import os
import re
# import sys
import time
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize

from class_utils import Utils
from plot_utils import nice_fonts, set_size

find_errors = Utils.find_errors

mpl.rcParams.update(nice_fonts)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
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


def get_threshold(files_datetime, plot_threshold_histogram=True):

    head_file_path = ('../data/algorithm_Toffoli_data/')

    all_files = os.listdir(head_file_path)
    matching_files = [
        file for file in all_files if file.endswith(files_datetime + '.txt')
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
            all_filenames.append(line)

    file_path_raw = ('../data/algorithm_Toffoli_data/raw_data_fluorescence/')

    # get pmt counts for thresholding
    pmt_counts = []
    for idx, file_name in enumerate(all_filenames):
        if len(file_name) > 40:
            # fix filenames for new naming convention that uses full paths
            last_part = file_name.rsplit('\\', 1)[-1]
            prefix_segment = file_name.rsplit('\\', 1)[0].rsplit('\\', 1)[-1]
            file_name = f'{prefix_segment}/{last_part}'

        if not is_file_empty(file_path_raw + file_name):
            # print(file_path_raw + file_name)
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

    threshold = np.argmin(array_counts_subset[:-15])
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


def plot_Toffoli(files_datetime, exp_num=1000, plot_3D=True):

    # initialise an empty 6x4 array to store the output data
    data_rate = []
    Toffoli_result = np.zeros((16, 16))

    threshold = get_threshold(files_datetime, plot_threshold_histogram=True)

    head_file_path = ('../data/algorithm_Toffoli_data/')

    all_files = os.listdir(head_file_path)
    matching_files = [
        file for file in all_files if file.endswith(files_datetime + '.txt')
    ]
    matching_files.sort(
        key=lambda x: os.path.getmtime(os.path.join(head_file_path, x)))

    def ket_number_key(filename):
        """Extracts the numeric value following the 'Ket_' prefix in a filename
        for sorting purposes.

        Parameters:
        filename (str): The filename from which to extract the
            relevant number.

        Returns:
        tuple: A tuple containing the portion before 'Ket_' and the
            extracted number as an integer.

        """
        # Split the filename at 'Ket_' and extract the numeric part
        match = re.search(r'Ket_(\d+)', filename)
        if match:
            number = int(match.group(1))
            prefix = filename.split('Ket_')[0]  # Get the part before 'Ket_'
            return (prefix, number)
        return (filename, float('inf')
                )  # Return a high value if 'Ket_' is not present

    matching_files = sorted(matching_files, key=ket_number_key)

    for file_idx, filename in enumerate(matching_files):
        # print(filename)

        with open(head_file_path + filename, 'r') as hf_name:
            # iterate through lines to get the last one, which gives
            # a list of all raw data filenames associated with the run
            for line_num, line in enumerate(hf_name):
                pass
            raw_file_names = [line.strip()]

        file_path_raw = (head_file_path + 'raw_data_fluorescence/')

        for non_empty_filename in raw_file_names:
            # fix filenames for new naming convention that uses full paths
            last_part = non_empty_filename.rsplit('\\', 1)[-1]
            prefix_segment = non_empty_filename.rsplit('\\',
                                                       1)[0].rsplit('\\',
                                                                    1)[-1]
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
                full_data_array = np.full((np.shape(data)[1], 1), np.nan)
                break
            else:
                pass

        for idx, file_name in enumerate(raw_file_names):
            # fix filenames for new naming convention that uses full paths
            last_part = file_name.rsplit('\\', 1)[-1]
            prefix_segment = file_name.rsplit('\\', 1)[0].rsplit('\\', 1)[-1]
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

                data_rate.append(
                    (np.shape(data)[0] - num_deshelve_error - num_shelve_error)
                    / np.shape(data)[0])

                print('This run data rate:',
                      (np.shape(data)[0] - num_deshelve_error -
                       num_shelve_error) / np.shape(data)[0])
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

        # need to only save the last 16 readout results - first one is herald
        Toffoli_result[file_idx, :] = np.array(
            [it[0] for it in full_data_array[1:]])

    Toffoli_result = np.flip(Toffoli_result)

    lower_error, upper_error = find_errors(1, Toffoli_result, exp_num)
    asym_error_bar = np.abs(
        [Toffoli_result - lower_error, upper_error - Toffoli_result])

    # Plotting
    # Choose a colormap
    cmap = plt.get_cmap("viridis")
    fig, (ax, ax_side) = plt.subplots(1,
                                      2,
                                      gridspec_kw={
                                          'width_ratios': [25, 2],
                                          'wspace': 0.1,
                                      },
                                      figsize=(12, 10))

    # side-bar
    side_data = np.array(data_rate).reshape(-1, 1)
    ax_side.imshow(
        side_data,
        cmap='viridis',
        aspect='auto',
        # origin='lower',
        vmin=0,
        # vmax=100
    )

    for it, rate in enumerate(data_rate):
        ax_side.text(0,
                     it,
                     str(np.round(100 * rate, 2)),
                     color="black" if 100 * rate > 50 else "white",
                     fontsize=8,
                     ha="center",
                     va="center")

    # main plot
    # Plot the color map
    ax.imshow(Toffoli_result, cmap=cmap,
              origin="lower")  # Set origin to lower left corner
    # cbar = plt.colorbar(im, ax=ax,
    #                     label="Probability")  # Add color bar with label
    # cbar.set_label('Probability')

    # Create the list of x tick labels
    # xtick_labels = [f'{i}' for i in sorted_hidden_strings]
    # x_positions = np.arange(len(matching_files))

    # ytick_labels = [f'{i}' for i in sorted_hidden_strings]
    # y_positions = np.arange(len(matching_files))
    # ax.set_xticks(x_positions)
    # ax.set_xticklabels(xtick_labels, rotation=0, ha='center', fontsize=10)
    # ax_side.set_yticks(y_positions)
    # ax_side.set_yticklabels(ytick_labels, rotation=0, ha='right', fontsize=10)
    # ax_side.set_xticks([])
    # ax_side.set_xticklabels([])
    # ax_side.set_title('Data Rate')

    # # Create the list of y tick labels
    # ax.set_yticks(y_positions)
    # ax.set_yticklabels(ytick_labels, rotation=0, ha='right', fontsize=10)

    # Set labels and title
    ax.set_ylabel('Found String')
    ax.set_xlabel('Hidden String')
    # date = date_time.split('_')
    # date_obj = datetime.strptime(date[0], '%Y%m%d')
    # plot_date = date_obj.strftime('%B %d, %Y')
    ax.set_title('4-polyqubit C-Toffoli Result - Measurement')

    text_threshold = 0.0
    # Add text annotations for values above a certain threshold
    for i in range(Toffoli_result.shape[0]):
        for j in range(Toffoli_result.shape[1]):
            value = Toffoli_result[i, j]
            if value > text_threshold:
                ax.text(j,
                        i,
                        str(np.round(100 * value, 1)) if value < 1 else 100,
                        color="black" if value > 0.5 else "white",
                        fontsize=12,
                        ha="center",
                        va="center")

    # fig.savefig(
    #     '/home/nicholas/Documents/Barium137_Qudit/analysis/' +
    #     f'25_level_heralded_SPAM/SPAM_result_2D_graph_{date_time}.pdf')

    Toffoli_expected_states = []
    for idx in range(0, 16):
        if idx == 14:
            Toffoli_expected_states.append(Toffoli_result[idx, idx + 1])
        elif idx == 15:
            Toffoli_expected_states.append(Toffoli_result[idx, idx - 1])
        else:
            Toffoli_expected_states.append(Toffoli_result[idx, idx])

    print('The mean success probability: ',
          np.round(100 * np.mean(Toffoli_expected_states), 3), '%.')
    print(
        'Error on mean success: ', 100 * (1 / 4) * np.sqrt(
            np.sum([i**2
                    for i in np.mean(asym_error_bar, axis=0).diagonal()])),
        '%.')
    print('The overall data rate is', np.round(np.mean(data_rate), 3))

    if plot_3D:
        # Normalize the data for color mapping
        norm = Normalize(vmin=Toffoli_result.min(), vmax=Toffoli_result.max())
        norm = Normalize(vmin=0, vmax=1)
        colors = plt.cm.viridis(Toffoli_result.flatten() /
                                float(Toffoli_result.max()))
        # Toffoli_result[0,0]=np.nan
        # Toffoli_mask = np.zeros_like(Toffoli_result)
        # Toffoli_mask[0,0]=2

        # Create a 3D figure
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111, projection='3d')

        # Create meshgrid for x and y coordinates
        x_indices, y_indices = np.meshgrid(np.arange(Toffoli_result.shape[1]),
                                           np.arange(Toffoli_result.shape[0]))

        # Flatten the arrays for plotting
        x_flat = x_indices.flatten()
        y_flat = y_indices.flatten()
        # Heights are zero (since we're using colors)
        z_flat = np.zeros_like(x_flat)

        # Plot the bars with colors
        ax.bar3d(x_flat,
                 y_flat,
                 z_flat,
                 0.8,
                 0.8,
                 Toffoli_result.flatten(),
                 shade=True,
                 color=colors,
                 edgecolor='black',
                 alpha=1)

        # ax.bar3d(x_flat,
        #          y_flat,
        #          z_flat,
        #          1,
        #          1,
        #          Toffoli_result.flatten(),
        #          shade=True,
        #          color=colors,
        #          alpha=0.8)
        # Add a color bar
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.viridis),
                            ax=ax)
        cbar.set_label('Probability')

        # Create the list of x tick labels
        basis_strings = [format(i, '04b') for i in range(16)]
        xtick_labels = [rf'$|{i}\rangle$' for i in basis_strings]
        # Set x tick positions and labels
        x_positions = np.arange(0.5, 16.5)
        ax.set_xticks(x_positions[::4])
        # Rotate labels for readability
        ax.set_xticklabels(xtick_labels[::4],
                           rotation=45,
                           ha='center',
                           va='bottom',
                           fontsize=14)

        # Create the list of y tick labels
        ytick_labels = [rf'$|{i}\rangle$' for i in basis_strings]
        # Set y tick positions and labels
        y_positions = np.arange(0.5, 16.5)
        ax.set_yticks(y_positions[::4])
        # Rotate labels for readability
        ax.set_yticklabels(ytick_labels[::4],
                           rotation=-30,
                           ha='center',
                           va='bottom',
                           fontsize=14)

        ax.tick_params(axis='x', pad=15)  # Increase distance of x tick labels
        ax.tick_params(axis='y', pad=15)  # Increase distance of y tick labels
        ax.tick_params(axis='z', grid_markersize=14)

        # Set labels and title
        ax.xaxis.set_label_position('top')
        ax.yaxis.set_label_position('top')
        ax.set_zlim(0, 1)
        ax.set_xlabel('Prepared State', labelpad=10)
        ax.set_ylabel('Measured State', labelpad=10)
        ax.set_title('CCCNOT Truth Table Measurement')
    else:
        pass

    return Toffoli_expected_states


if __name__ == '__main__':
    start_time = time.time()

    datetimes = ['20250604_1728']

    for idx, dt in enumerate(datetimes):
        Toffoli_result = plot_Toffoli(dt)

    plt.tight_layout()
    plt.show()

    print("--- %s seconds ---" % (time.time() - start_time))
