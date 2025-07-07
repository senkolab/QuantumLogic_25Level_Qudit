import os
# import sys
import time
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

    head_file_path = ('../data/algorithm_BVA_data/')

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

    file_path_raw = ('../data/algorithm_BVA_data/raw_data_fluorescence/')

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


def plot_BVA(files_datetime,
             polyqubit=2,
             exp_num=1000,
             plot_bar=True,
             plotting=True):

    # initialise an empty 6x4 array to store the output data
    hidden_strings = []
    data_rate = []
    BVA_result = np.zeros((2**polyqubit, 2**polyqubit))

    threshold = get_threshold(files_datetime, plot_threshold_histogram=False)

    head_file_path = ('../data/algorithm_BVA_data/')

    all_files = os.listdir(head_file_path)
    matching_files = [
        file for file in all_files if file.endswith(files_datetime + '.txt')
    ]
    matching_files.sort(
        key=lambda x: os.path.getmtime(os.path.join(head_file_path, x)))

    # define the lists of amplitudes and times used to fit the exp decay
    for file_idx, filename in enumerate(matching_files):
        print(filename)

        with open(head_file_path + filename, 'r') as hf_name:
            # iterate through lines to get the last one, which gives
            # a list of all raw data filenames associated with the run
            # print(filename)
            split_name = filename.split('_')
            hidden_string = split_name[4]

            for line_num, line in enumerate(hf_name):
                pass
            raw_file_names = [line.strip()]

        file_path_raw = ('../data/algorithm_BVA_data/raw_data_fluorescence/')

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

        if polyqubit == 2:
            data_size_fixer = 2
        elif polyqubit == 3:
            data_size_fixer = 1

        BVA_result[file_idx, :] = np.array(
            [it[0] for it in full_data_array[data_size_fixer:]])
        hidden_strings.append(hidden_string)

    # sort the results by hidden string
    sorted_indices = sorted(range(len(hidden_strings)),
                            key=lambda i: hidden_strings[i])
    sorted_hidden_strings = [hidden_strings[i] for i in sorted_indices]
    sorted_BVA_result = np.array(
        [BVA_result[i] / (np.sum(BVA_result[i])) for i in sorted_indices])

    lower_error, upper_error = find_errors(1, sorted_BVA_result, exp_num)
    asym_error_bar = np.abs(
        [sorted_BVA_result - lower_error, upper_error - sorted_BVA_result])

    # Get simulation results to match
    if polyqubit == 2:
        sim_data = np.array([[0.98641679, 0.00839968, 0.00766451, 0.00736001],
                             [0.00537788, 0.98930061, 0.0033113, 0.00640406],
                             [0.00081548, 0.00084773, 0.9865522, 0.00327459],
                             [0.00738985, 0.00145197, 0.002472, 0.98296134]])

    elif polyqubit == 3:
        sim_data = np.array([[
            0.92065691, 0.02437877, 0.00142459, 0.00716432, 0.00352182,
            0.00248586, 0.04019688, 0.003066
        ],
                             [
                                 0.02042114, 0.88372025, 0.00783682,
                                 0.00193585, 0.00185339, 0.00507649,
                                 0.00540424, 0.06131823
                             ],
                             [
                                 0.00182458, 0.00745854, 0.91233754, 0.0034838,
                                 0.06860646, 0.00154865, 0.00505335, 0.00176493
                             ],
                             [
                                 0.00776664, 0.00417564, 0.00216137,
                                 0.74977031, 0.00242225, 0.22403132,
                                 0.00275618, 0.00797191
                             ],
                             [
                                 0.00397429, 0.00273506, 0.06895442,
                                 0.00259414, 0.90650198, 0.00404522,
                                 0.00180986, 0.01166192
                             ],
                             [
                                 0.00457722, 0.0066317, 0.00147753, 0.2269938,
                                 0.00258205, 0.75191121, 0.0080962, 0.00270508
                             ],
                             [
                                 0.03813301, 0.00477505, 0.00421594,
                                 0.00290795, 0.00166881, 0.00952094,
                                 0.92499912, 0.01406105
                             ],
                             [
                                 0.0026462, 0.06612499, 0.00159179, 0.00514983,
                                 0.01284323, 0.00138031, 0.01168417, 0.89745087
                             ]])

    # Plotting
    # Choose a colormap
    cmap = plt.get_cmap("viridis")
    fig, (ax, ax_side) = plt.subplots(1,
                                      2,
                                      gridspec_kw={
                                          'width_ratios': [25, 2],
                                          'wspace': 0.1,
                                      },
                                      figsize=(8, 6))

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
    ax.imshow(sorted_BVA_result, cmap=cmap,
              origin="lower")  # Set origin to lower left corner
    # cbar = plt.colorbar(im, ax=ax,
    #                     label="Probability")  # Add color bar with label
    # cbar.set_label('Probability')

    # Create the list of x tick labels
    xtick_labels = [f'{i}' for i in sorted_hidden_strings]
    x_positions = np.arange(len(matching_files))

    ytick_labels = [f'{i}' for i in sorted_hidden_strings]
    y_positions = np.arange(len(matching_files))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(xtick_labels, rotation=0, ha='center', fontsize=10)
    ax_side.set_yticks(y_positions)
    ax_side.set_yticklabels(ytick_labels, rotation=0, ha='right', fontsize=10)
    ax_side.set_xticks([])
    ax_side.set_xticklabels([])
    ax_side.set_title('Data Rate')

    # # Create the list of y tick labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ytick_labels, rotation=0, ha='right', fontsize=10)

    # Set labels and title
    ax.set_ylabel('Found String')
    ax.set_xlabel('Hidden String')
    # date = date_time.split('_')
    # date_obj = datetime.strptime(date[0], '%Y%m%d')
    # plot_date = date_obj.strftime('%B %d, %Y')
    ax.set_title(
        f'{polyqubit}-polyqubit: Bernstein-Vazirani Result - Measurement')

    text_threshold = 0.0
    # Add text annotations for values above a certain threshold
    for i in range(sorted_BVA_result.shape[0]):
        for j in range(sorted_BVA_result.shape[1]):
            value = sorted_BVA_result[i, j]
            if value > text_threshold:
                ax.text(j,
                        i,
                        str(np.round(100 * value, 2)) if value < 1 else 100,
                        color="black" if value > 0.5 else "white",
                        fontsize=12,
                        ha="center",
                        va="center")

    # fig.savefig(
    #     '/home/nicholas/Documents/Barium137_Qudit/analysis/' +
    #     f'25_level_heralded_SPAM/SPAM_result_2D_graph_{date_time}.pdf')
    if polyqubit == 2:
        results_dict = {
            '00': np.round(np.max(100 * sorted_BVA_result[0]), 2),
            '01': np.round(np.max(100 * sorted_BVA_result[1]), 2),
            '10': np.round(np.max(100 * sorted_BVA_result[2]), 2),
            '11': np.round(np.max(100 * sorted_BVA_result[3]), 2),
        }
    elif polyqubit == 3:
        results_dict = {
            '000': np.round(np.max(100 * sorted_BVA_result[0]), 2),
            '001': np.round(np.max(100 * sorted_BVA_result[1]), 2),
            '010': np.round(np.max(100 * sorted_BVA_result[2]), 2),
            '011': np.round(np.max(100 * sorted_BVA_result[3]), 2),
            '100': np.round(np.max(100 * sorted_BVA_result[4]), 2),
            '101': np.round(np.max(100 * sorted_BVA_result[5]), 2),
            '110': np.round(np.max(100 * sorted_BVA_result[6]), 2),
            '111': np.round(np.max(100 * sorted_BVA_result[7]), 2),
        }

    print(results_dict)

    if plot_bar:
        # Configure the subplots
        if polyqubit == 2:
            figure_size = (5, 5)
            plt_colour = 0
        elif polyqubit == 3:
            figure_size = (12, 5)
            plt_colour = 1
        fig, axs = plt.subplots(2,
                                2**(polyqubit - 1),
                                figsize=figure_size,
                                sharey=True)

        plt.suptitle(
            f'{polyqubit}-polyqubit: Bernstein-Vazirani Result - Measurement',
            fontsize=18)
        # Create bar graphs for each row with error bars
        for i in range(2**polyqubit):
            # Create bar plot and capture the bars for annotation
            bars = axs[i % 2, i // 2].bar(
                np.arange(len(sorted_BVA_result[i])) - 0.2,
                sorted_BVA_result[i],
                yerr=[asym_error_bar[0][i], asym_error_bar[1][i]],
                alpha=1,
                color=colors[plt_colour],
                width=0.4,
                capsize=5)

            axs[i % 2, i // 2].bar(
                np.arange(len(sorted_BVA_result[i])) + 0.2,
                sim_data[i],
                # yerr  = [asym_error_bar[0][i], asym_error_bar[1][i]],
                alpha=0.5,
                color=colors[plt_colour],
                width=0.4,
                capsize=5)

            # add a reference bar graph
            ref = np.zeros(len(sorted_BVA_result[i]))
            ref[i] = 1
            axs[i % 2, i // 2].bar(range(len(sorted_BVA_result[i])),
                                   ref,
                                   alpha=0.5,
                                   edgecolor='black',
                                   linestyle='--',
                                   facecolor='none')

            # breakpoint()
            if polyqubit == 2:
                axs[i % 2, i // 2].set_xticks(range(len(sorted_BVA_result[i])))
                axs[i % 2, i // 2].set_xticklabels(hidden_strings)

            elif polyqubit == 3:
                if i % 2 == 0 and i // 2 < 4:
                    axs[i % 2, i // 2].set_xticks(
                        range(len(sorted_BVA_result[i]))[::2])
                    axs[i % 2, i // 2].set_xticklabels(hidden_strings[::2])
                else:
                    axs[i % 2, i // 2].set_xticks(
                        range(len(sorted_BVA_result[i]))[1::2])
                    axs[i % 2, i // 2].set_xticklabels(hidden_strings[1::2])

            axs[i % 2, i // 2].legend([f'Key = {hidden_strings[i]}'],
                                      fontsize=14)
            axs[i % 2, i // 2].grid()

            # Annotate each bar with its percentage value
            for bar in bars:
                yval = bar.get_height()  # Get the height of each bar
                if yval > 0.25:
                    axs[i % 2, i // 2].text(bar.get_x() + bar.get_width() / 2,
                                            yval * 0.5,
                                            f'{100 * yval:.1f}',
                                            ha='center',
                                            va='center',
                                            color='black',
                                            fontsize=12,
                                            rotation=90)

        axs[0, 0].set_ylabel('Probabilities', fontsize=16)
        axs[1, 0].set_ylabel('Probabilities', fontsize=16)
        axs[1, 0].set_xlabel('Measured Keys', fontsize=16)
        axs[1, 1].set_xlabel('Measured Keys', fontsize=16)

        axs[0, 0].tick_params(axis='both', labelsize=16)  # Set tick size to 10
        axs[0, 1].tick_params(axis='both', labelsize=16)  # Set tick size to 10
        axs[1, 0].tick_params(axis='both', labelsize=16)  # Set tick size to 10
        axs[1, 1].tick_params(axis='both', labelsize=16)  # Set tick size to 10

        if polyqubit == 3:
            axs[1, 2].set_xlabel('Measured Keys', fontsize=16)
            axs[1, 3].set_xlabel('Measured Keys', fontsize=16)
            axs[0, 2].tick_params(axis='both',
                                  labelsize=16)  # Set tick size to 10
            axs[0, 3].tick_params(axis='both',
                                  labelsize=16)  # Set tick size to 10
            axs[1, 2].tick_params(axis='both',
                                  labelsize=16)  # Set tick size to 10
            axs[1, 3].tick_params(axis='both',
                                  labelsize=16)  # Set tick size to 10

    print('The mean success probability: ',
          np.round(100 * np.mean(np.diagonal(sorted_BVA_result)), 3), '%.')
    print(
        'Error on mean success: ', 100 * (1 / 4) * np.sqrt(
            np.sum([i**2
                    for i in np.mean(asym_error_bar, axis=0).diagonal()])),
        '%.')
    return sorted_BVA_result, results_dict


if __name__ == '__main__':
    start_time = time.time()

    polyqubit = 3

    if polyqubit == 2:
        datetimes = ['20250325_1320']  # 1000 shots, regular H algorithm

    elif polyqubit == 3:
        datetimes = ['20250430_1653']

    for idx, dt in enumerate(datetimes):
        # get_threshold(dt)
        plot_BVA(dt, polyqubit=polyqubit, plot_bar=True)

    plt.tight_layout()
    plt.show()

    print("--- %s seconds ---" % (time.time() - start_time))
