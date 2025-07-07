import os
import time
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize

from calibration_analysis import Calibrations
from class_utils import Utils
from plot_utils import nice_fonts, set_size

find_errors = Utils.find_errors

mpl.rcParams.update(nice_fonts)
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#7f7f7f']


def get_threshold(plot_threshold_histogram=True):

    # Specify the directory to traverse
    directory_path = (
        '/home/nicholas/Documents/Barium137_Qudit/data/25_level_5K_SPAM_data/')

    # List to hold all file paths
    file_list = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_list.append(os.path.join(root, file))

    # get pmt counts for thresholding
    pmt_counts = []
    for it in range(len(file_list)):
        with open(file_list[it], 'r') as filename:
            data = pd.read_csv(
                filename,
                delimiter="\[|\]|,|\"",  # noqa: W605
                engine='python',
                header=None).values
            data = data[:, 4:-19]
            pmt_counts += list(np.ravel(data))
            SPAM_dimension = data.shape[1] - 1
    # Count the occurrences of each entry
    entry_counts = Counter(pmt_counts)
    # Extract unique entry values and their corresponding counts
    array_counts = np.array(
        [entry_counts.get(key, 0) for key in range(max(entry_counts) + 1)])
    counts_dummy = np.arange(0, len(array_counts), 1)
    mask = counts_dummy < 20
    subset_array_counts = array_counts[mask]

    threshold = np.argmin(subset_array_counts)
    print(f'\n Found threshold is: {threshold}.')

    if plot_threshold_histogram:
        fig, ax = plt.subplots(figsize=set_size(width='half'))

        # Plot the histogram
        ax.bar(counts_dummy, array_counts, label='Original Histogram')
        ax.vlines(threshold,
                  min(array_counts),
                  max(array_counts),
                  colors='k',
                  linestyles='--',
                  label='Threshold')
        ax.set_yscale('log')
        ax.set_title('Fluorescence Readout Histogram')
        ax.set_ylabel('Occurences')
        ax.set_xlabel('PMT Counts')
        ax.legend()

    return threshold, SPAM_dimension


def get_spam_result(init_ket: int,
                    SPAM_dimension: int = 25,
                    num_exps: int = 1000,
                    post_processing: bool = True,
                    plotting=False,
                    verbose: bool = False,
                    get_top_five=False,
                    get_best_run=False,
                    herald_measurement=True,
                    reversed_readout: bool = False):

    file_path = (
        '/home/nicholas/Documents/' +
        f'Barium137_Qudit/data/25_level_5K_SPAM_data/InitKet_{init_ket}/')
    file_prefix = 'SPAM_10k_Heralded_'
    # date = date_time.split('_')
    # date_obj = datetime.strptime(date[0], '%Y%m%d')
    # plot_date = date_obj.strftime('%B %d, %Y')

    files = os.listdir(file_path)
    matching_files = []
    for fname in files:
        # Construct the full path to the file
        full_path = os.path.join(file_path, fname)

        # Check if the file exists and is a regular file
        if os.path.isfile(full_path):
            # Check if the search string exists in the file name
            if file_prefix in fname:
                matching_files.append(fname)

    matching_files = sorted(
        matching_files,
        key=lambda f: os.path.getmtime(os.path.join(file_path, f)))

    percentage_data_used = []
    init_errors = []
    deshelve_errors = []
    spam_data = np.zeros((SPAM_dimension, len(matching_files)))
    for it in range(len(matching_files)):
        with open(file_path + matching_files[it], 'r') as filename:
            data = pd.read_csv(
                filename,
                delimiter="\[|\]|,|\"",  # noqa: W605
                engine='python',
                header=None).values
            data = data[:, 4:-19]

        # Plot raw PMT counts per dimension/state
        # fig = plt.subplots()
        # for i in range(np.shape(data)[0]):
        #     plt.plot(range(25),data[i,1:])
        # plt.show()

        # Implementing DYNAMIC thresholding
        pmt_counts = list(np.ravel(data))
        # Count the occurrences of each entry
        entry_counts = Counter(pmt_counts)
        # Extract unique entry values and their corresponding counts
        array_counts = np.array(
            [entry_counts.get(key, 0) for key in range(max(entry_counts) + 1)])
        counts_dummy = np.arange(0, len(array_counts), 1)
        mask = counts_dummy < 20
        subset_array_counts = array_counts[mask]
        threshold = np.argmin(subset_array_counts)
        # print(f'Found threshold is: {threshold}.')

        # fig, ax = plt.subplots(figsize=set_size(width='half'))
        # # Plot the histogram
        # ax.bar(counts_dummy, array_counts, label='Original Histogram')
        # ax.vlines(threshold,
        #           min(array_counts),
        #           max(array_counts),
        #           colors='k',
        #           linestyles='--',
        #           label='Threshold')
        # ax.set_yscale('log')
        # ax.set_title('Fluorescence Readout Histogram')
        # ax.set_ylabel('Occurences')
        # ax.set_xlabel('PMT Counts')
        # ax.legend()
        # plt.show()

        binary_data = data > threshold

        # Initialize a mask to track the first occurrence of True in each row
        first_true_bin_array = np.zeros_like(binary_data, dtype=bool)

        # Iterate through each row
        num_shelve_error = 0
        num_deshelve_error = 0
        for row_idx in range(binary_data.shape[0]):
            row = binary_data[row_idx]
            first_true_idx = np.argmax(row)  # Find the index of the first True
            # print(row)
            # print(first_true_idx)
            # break
            if post_processing:
                if sum(row) > 0 and first_true_idx:
                    first_true_bin_array[row_idx, first_true_idx] = True
                elif sum(row) == 0:
                    num_deshelve_error += 1
                elif first_true_idx == 0:
                    num_shelve_error += 1
            else:
                first_true_bin_array[row_idx, first_true_idx] = True

        if herald_measurement:
            summed_bin_array = np.sum(first_true_bin_array, axis=0) / (
                first_true_bin_array.shape[0] - num_shelve_error -
                num_deshelve_error)
            data_rate = np.round(
                100 * (first_true_bin_array.shape[0] - num_shelve_error -
                       num_deshelve_error) / num_exps, 3)
        else:
            summed_bin_array = np.sum(first_true_bin_array, axis=0) / (
                first_true_bin_array.shape[0] - num_shelve_error)
            data_rate = np.round(
                100 * (first_true_bin_array.shape[0] - num_shelve_error) /
                num_exps, 3)

        spam_data[:, it] = summed_bin_array[1:]
        # print(f'Ket number {it}, fidelity: ' +
        #       f'{np.round(100*summed_bin_array[it+1],2)},
        #          fails: {num_fails}')
        state_id = np.argmax(summed_bin_array)
        state_fid = np.round(100 * summed_bin_array[state_id], 3)
        percentage_data_used.append(data_rate)
        init_errors.append(num_shelve_error)
        deshelve_errors.append(num_deshelve_error)
        if verbose:
            print(f'|{state_id-1}>, fidelity: {state_fid},' +
                  f' init. err: {num_shelve_error}, ' +
                  f' meas. err: {num_deshelve_error},' +
                  f' data rate: {data_rate},' +
                  f' file: {matching_files[it][-16:]}')

    if reversed_readout:
        percentage_data_used = np.concatenate(
            ([percentage_data_used[0]], percentage_data_used[1:][::-1]))
        # We need to flip the RO order and the order of kets to keep consistent
        # Flip the order of the columns (excluding the first column)
        flipped_columns = np.copy(
            spam_data)  # Create a copy of the original array
        # Reverse the columns from the second column onwards
        flipped_columns[:, 1:] = spam_data[:, 1:][:, ::-1]
        # Reverse the order of the rows (except first entry)
        result = flipped_columns[::-1, :]
        result = np.vstack((result[-1, :], result[:-1, :]))
        spam_data = result

    if get_top_five:
        # sort and take just the top 4 outcomes
        spam_data = spam_data[:, np.argsort(spam_data[init_ket])][:, -5:]
    # print(spam_data)

    if get_best_run:
        # Write in the average SPAM fidelity
        fidelities = spam_data[:, np.argmax(spam_data[init_ket])]
        fidelities_error = np.sqrt(
            (fidelities) * (1 - fidelities) / (len(matching_files) * num_exps))
        avg_data_rate = np.round(np.mean(percentage_data_used), 3)
    else:
        # Write in the average SPAM fidelity
        fidelities = np.mean(spam_data, axis=1)
        fidelities_error = np.sqrt(
            (fidelities) * (1 - fidelities) / (len(matching_files) * num_exps))
        avg_data_rate = np.round(np.mean(percentage_data_used), 3)

    if verbose:
        print('\n')
        print('Average fidelity', fidelities[init_ket])
        print('Average fidelity uncertainty', fidelities_error[init_ket])
        print('Average data rate', avg_data_rate)
        print('Init errors total', np.sum(init_errors))
        print(
            'Init error rate',
            np.round(
                100 * (np.sum(init_errors) / (len(matching_files) * num_exps)),
                2))
        print('Deshelve errors total', np.sum(deshelve_errors))
        print(
            'Deshelve error rate',
            np.round(
                100 * (np.sum(deshelve_errors) /
                       (len(matching_files) * num_exps)), 2), '\n')

    if plotting:

        spam_data = np.transpose(spam_data)
        # Choose a colormap
        cmap = plt.get_cmap("viridis")

        # Create a figure and axis
        # fig, ax, ax_side = plt.subplots(figsize=(8, 6))

        fig, (ax, ax_side) = plt.subplots(
            1,
            2,
            gridspec_kw={
                'width_ratios': [SPAM_dimension, 2],
                'wspace': 0.1,
            },
            figsize=(10, 1 + 8 * (len(matching_files) / SPAM_dimension)))

        # side-bar
        side_data = np.array(percentage_data_used).reshape(-1, 1)
        ax_side.imshow(side_data,
                       cmap='viridis',
                       aspect='auto',
                       origin='lower',
                       vmin=0,
                       vmax=100)

        ax_side.set_xticks([])
        ax_side.set_xticklabels([])
        ax_side.set_yticks([])
        ax_side.set_yticklabels([])
        ax_side.set_title('Data Rate')

        for it, rate in enumerate(percentage_data_used):
            ax_side.text(0,
                         it,
                         str(np.round(rate, 2)),
                         color="black" if rate > 50 else "white",
                         fontsize=8,
                         ha="center",
                         va="center")

        # main plot
        # Plot the color map
        ax.imshow(spam_data, cmap=cmap,
                  origin="lower")  # Set origin to lower left corner

        # Create the list of x tick labels
        xtick_labels = [rf'$|{i}\rangle$' for i in range(SPAM_dimension)]
        x_positions = np.arange(SPAM_dimension)

        ax.set_xticks(x_positions[::2])
        ax.set_xticklabels(xtick_labels[::2],
                           rotation=0,
                           ha='center',
                           fontsize=10)

        # Create the list of y tick labels
        ytick_labels = [rf'$|{i}\rangle$' for i in range(len(matching_files))]
        y_positions = np.arange(len(matching_files))  # Get number of runs
        ax.set_yticks(y_positions[::2])
        ax.set_yticklabels(ytick_labels[::2],
                           rotation=0,
                           ha='right',
                           fontsize=10)

        # Set labels and title
        ax.set_ylabel('Prepared State')
        ax.set_xlabel('Measured State')
        ax.set_title('State Preparation and Measurement Fidelities')

        text_threshold = 0.0
        # Add text annotations for values above a certain threshold
        for i in range(spam_data.shape[0]):
            for j in range(spam_data.shape[1]):
                value = spam_data[i, j]
                if value > text_threshold:
                    ax.text(j,
                            i,
                            str(np.round(100 *
                                         value, 1)) if value < 1 else 100,
                            color="black" if value > 0.5 else "white",
                            fontsize=8,
                            ha="center",
                            va="center")

        # fig.savefig('/home/nicholas/Documents/Barium137_Qudit/analysis/' +
        #             f'{SPAM_dimension}_level_5K_SPAM/SPAM_5K_result_2D_graph.pdf')

    else:
        pass

    return (fidelities, fidelities_error, init_errors, deshelve_errors,
            avg_data_rate, len(matching_files))


def plot_all_spam_results(kets: np.ndarray = np.arange(0, 25, 1),
                          num_exps: int = 1000,
                          SPAM_dimension: int = 25,
                          post_processing: bool = True,
                          plot_2D: bool = False,
                          plot_3D: bool = False,
                          plotting: bool = True,
                          plot_infidelity: bool = True,
                          plot_infidelity_two_schemes: bool = True,
                          reversed_readout: bool = False,
                          plot_threshold_histogram: bool = False,
                          flipped_RO_order: bool = False,
                          do_state_choice_plot: bool = False,
                          long_output: bool = False,
                          herald_measurement: bool = True,
                          get_best_run: bool = False,
                          get_top_five: bool = False,
                          verbose: bool = False):

    spam_data = np.empty((SPAM_dimension, SPAM_dimension))
    spam_data_err = np.empty((SPAM_dimension, SPAM_dimension))
    total_data_rate = np.empty(SPAM_dimension)
    init_err_rate = np.empty(SPAM_dimension)
    meas_err_rate = np.empty(SPAM_dimension)
    init_errors = np.empty(SPAM_dimension)
    meas_errors = np.empty(SPAM_dimension)
    total_chunks = np.empty(SPAM_dimension)

    for itera in range(0, max(kets) + 1):
        fid, fid_err, init_err, meas_err, data_rate, chunks = get_spam_result(
            itera,
            verbose=verbose,
            herald_measurement=herald_measurement,
            get_best_run=get_best_run,
            get_top_five=get_top_five)

        spam_data[itera, :] = fid
        spam_data_err[itera, :] = fid_err
        total_data_rate[itera] = data_rate
        init_errors[itera] = np.sum(init_err)
        meas_errors[itera] = np.sum(meas_err)
        init_err_rate[itera] = 100 * np.sum(init_err) / (chunks * 1000)
        meas_err_rate[itera] = 100 * np.sum(meas_err) / (chunks * 1000 -
                                                         np.sum(init_err))
        total_chunks[itera] = chunks

        print(f'|{itera}>, fidelity: {np.round(100*fid[itera],3)},' +
              f' init. err: {np.sum(init_err)}, ' +
              f' meas. err: {np.sum(meas_err)},' +
              f' data rate: {data_rate},' + f' chunks: {chunks}')

    # Write in the average SPAM fidelity
    fidelities = spam_data.diagonal()
    fidelities_error = np.empty(len(fidelities))

    for itera, fid in enumerate(fidelities):
        fidelities_error[itera] = np.sqrt(
            fid * (1 - fid) / (total_data_rate[itera] * num_exps * 5))

    avg_fidelity = np.round(100 * np.mean(fidelities), 3)
    avg_fidelity_error = np.round(
        avg_fidelity * np.sqrt(np.sum((fidelities_error / fidelities)**2)), 3)

    avg_data_rate = np.round(np.mean(total_data_rate), 3)
    print('\n')
    print('Average fidelity', avg_fidelity)
    print('Average fidelity uncertainty', avg_fidelity_error)
    print('Average data rate', avg_data_rate)
    print('Init errors total', np.sum(init_errors))
    print(
        'Init error rate',
        np.round(100 * (np.sum(init_errors) / (len(total_chunks) * num_exps)),
                 2))
    print('Deshelve errors total', np.sum(meas_errors))
    print(
        'Deshelve error rate',
        np.round(100 * (np.sum(meas_errors) / (len(total_chunks) * num_exps)),
                 2), '\n')

    if plotting:
        ##########################################
        # Creating figures #######################
        ##########################################

        if plot_3D:
            # Normalize the data for color mapping
            norm = Normalize(vmin=spam_data.min(), vmax=spam_data.max())
            colors = plt.cm.viridis(spam_data.flatten() /
                                    float(spam_data.max()))

            # Create a 3D figure
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(111, projection='3d')

            # Create meshgrid for x and y coordinates
            x_indices, y_indices = np.meshgrid(np.arange(spam_data.shape[1]),
                                               np.arange(spam_data.shape[0]))

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
                     spam_data.flatten(),
                     shade=True,
                     color=colors,
                     alpha=0.8)

            # Add a color bar
            cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.viridis),
                                ax=ax)
            cbar.set_label('Probability')

            # Create the list of x tick labels
            xtick_labels = [rf'$|{i+1}\rangle$' for i in range(SPAM_dimension)]
            # Set x tick positions and labels
            x_positions = np.arange(SPAM_dimension)
            ax.set_xticks(x_positions[::3])
            # Rotate labels for readability
            ax.set_xticklabels(xtick_labels[::3],
                               rotation=0,
                               ha='center',
                               va='bottom',
                               fontsize=8)

            # Create the list of x tick labels
            ytick_labels = [rf'$|{i+1}\rangle$' for i in range(SPAM_dimension)]
            # Set y tick positions and labels
            y_positions = np.arange(SPAM_dimension)
            ax.set_yticks(y_positions[::3])
            # Rotate labels for readability
            ax.set_yticklabels(ytick_labels[::3],
                               rotation=0,
                               ha='center',
                               va='bottom',
                               fontsize=8)

            # Set labels and title
            ax.xaxis.set_label_position('top')
            ax.yaxis.set_label_position('top')
            ax.set_xlabel('Prepared State')
            ax.set_ylabel('Measured State', labelpad=10)
            ax.set_title('State Preparation and Measurement Fidelities')
        else:
            pass

        ##########################################
        # Now for a 2D version of the figure #####
        ##########################################

        if plot_2D:
            # Choose a colormap
            cmap = plt.get_cmap("viridis")

            # Create a figure and axis
            # fig, ax, ax_side = plt.subplots(figsize=(8, 6))

            fig, (ax, ax_side_INIT,
                  ax_side_RO) = plt.subplots(1,
                                             3,
                                             gridspec_kw={
                                                 'width_ratios':
                                                 [SPAM_dimension, 2, 2],
                                                 'wspace':
                                                 0.2,
                                             },
                                             figsize=(12, 1 + 8))

            # side-bar for initialisation error
            side_data_init = np.array(init_err_rate).reshape(-1, 1)
            ax_side_INIT.imshow(100 - side_data_init,
                                cmap='viridis',
                                aspect='auto',
                                origin='lower',
                                vmin=0,
                                vmax=100)

            ax_side_INIT.set_xticks([])
            ax_side_INIT.set_xticklabels([])
            ytick_labels = [rf'$|{i}\rangle$' for i in range(SPAM_dimension)]
            y_positions = np.arange(SPAM_dimension)
            ax_side_INIT.set_yticks(y_positions[::2])
            ax_side_INIT.set_yticklabels(ytick_labels[::2],
                                         rotation=0,
                                         ha='right',
                                         fontsize=10)
            ax_side_INIT.set_title('Init. Rate')

            for it, rate in enumerate(init_err_rate):
                ax_side_INIT.text(0,
                                  it,
                                  str(np.round(100 - rate, 2)),
                                  color="black" if 100 -
                                  rate > 50 else "white",
                                  fontsize=8,
                                  ha="center",
                                  va="center")

            # side-bar for initialisation error
            side_data_readout = np.array(meas_err_rate).reshape(-1, 1)
            ax_side_RO.imshow(100 - side_data_readout,
                              cmap='viridis',
                              aspect='auto',
                              origin='lower',
                              vmin=0,
                              vmax=100)

            ax_side_RO.set_xticks([])
            ax_side_RO.set_xticklabels([])
            ytick_labels = [rf'$|{i}\rangle$' for i in range(SPAM_dimension)]
            y_positions = np.arange(SPAM_dimension)
            ax_side_RO.set_yticks(y_positions[::2])
            ax_side_RO.set_yticklabels(ytick_labels[::2],
                                       rotation=0,
                                       ha='right',
                                       fontsize=10)
            ax_side_RO.set_title('Readout Rate')

            for it, rate in enumerate(meas_err_rate):
                ax_side_RO.text(0,
                                it,
                                str(np.round(100 - rate, 2)),
                                color="black" if 100 - rate > 50 else "white",
                                fontsize=8,
                                ha="center",
                                va="center")

            # main plot
            # Plot the color map
            ax.imshow(spam_data, cmap=cmap, origin="lower")
            # cbar = plt.colorbar(im, ax=ax,
            #                     label="Probability")
            # cbar.set_label('Probability')

            # Create the list of x tick labels
            xtick_labels = [rf'$|{i}\rangle$' for i in range(SPAM_dimension)]
            x_positions = np.arange(SPAM_dimension)
            ax.set_xticks(x_positions[::2])
            ax.set_xticklabels(xtick_labels[::2],
                               rotation=0,
                               ha='center',
                               fontsize=10)

            # Create the list of y tick labels
            ytick_labels = [rf'$|{i}\rangle$' for i in range(SPAM_dimension)]
            y_positions = np.arange(SPAM_dimension)
            ax.set_yticks(y_positions[::2])
            ax.set_yticklabels(ytick_labels[::2],
                               rotation=0,
                               ha='right',
                               fontsize=10)

            # Set labels and title
            ax.set_ylabel('Prepared State')
            ax.set_xlabel('Measured State')
            ax.set_title('State Preparation and Measurement Fidelities')

            ax.text(0.6,
                    0.12,
                    f"Avg. Fidelity: {avg_fidelity} " + r"$\pm$" +
                    f" {avg_fidelity_error}%, \n" +
                    f"Avg. Data Rate: {avg_data_rate}%.",
                    color="white",
                    transform=ax.transAxes,
                    fontsize=10,
                    ha="left",
                    va="top")

            text_threshold = 0.001
            # Add text annotations for values above a certain threshold
            for i in range(spam_data.shape[0]):
                for j in range(spam_data.shape[1]):
                    value = spam_data[i, j]
                    if value > text_threshold:
                        ax.text(j,
                                i,
                                str(np.round(100 *
                                             value, 1)) if value < 1 else 100,
                                color="black" if value > 0.5 else "white",
                                fontsize=8,
                                ha="center",
                                va="center")

            fig.savefig('/home/nicholas/Documents/Barium137_Qudit/analysis/' +
                        '25_level_5K_SPAM/SPAM_5K_result_2D_graph.pdf')
        else:
            pass

        if plot_infidelity_two_schemes:
            ##########################################
            # Now for a 2D version of the figure #####
            ##########################################

            # Choose a colormap
            mapper = mpl.cm.viridis(np.linspace(0.0, 0.3, 256))
            cmap = cm.colors.ListedColormap(mapper)

            mapper2 = mpl.cm.viridis(np.linspace(0.7, 1, 256))
            cmap2 = cm.colors.ListedColormap(mapper2)

            # Create a figure and axis
            # fig, ax, ax_side = plt.subplots(figsize=(8, 6))

            fig, (ax, ax_side_INIT,
                  ax_side_RO) = plt.subplots(1,
                                             3,
                                             gridspec_kw={
                                                 'width_ratios':
                                                 [SPAM_dimension, 2, 2],
                                                 'wspace':
                                                 0.2,
                                             },
                                             figsize=(12, 1 + 8))

            # side-bar for initialisation error
            side_data_init = np.array(init_err_rate).reshape(-1, 1)
            ax_side_INIT.imshow(side_data_init,
                                cmap='viridis',
                                aspect='auto',
                                origin='lower',
                                vmin=0,
                                vmax=np.max(side_data_init))

            ax_side_INIT.set_xticks([])
            ax_side_INIT.set_xticklabels([])
            ytick_labels = [rf'$|{i}\rangle$' for i in range(SPAM_dimension)]
            y_positions = np.arange(SPAM_dimension)
            ax_side_INIT.set_yticks(y_positions[::2])
            ax_side_INIT.set_yticklabels(ytick_labels[::2],
                                         rotation=0,
                                         ha='right',
                                         fontsize=10)
            ax_side_INIT.set_title('Init.')

            for it, rate in enumerate(init_err_rate):
                ax_side_INIT.text(
                    0,
                    it,
                    str(np.round(rate, 2)),
                    color="black" if rate > np.max(side_data_init) /
                    2 else "white",
                    fontsize=12,
                    ha="center",
                    va="center")

            # side-bar for initialisation error
            side_data_readout = np.array(meas_err_rate).reshape(-1, 1)
            ax_side_RO.imshow(side_data_readout,
                              cmap='viridis',
                              aspect='auto',
                              origin='lower',
                              vmin=0,
                              vmax=np.max(side_data_readout))

            ax_side_RO.set_xticks([])
            ax_side_RO.set_xticklabels([])
            ytick_labels = [rf'$|{i}\rangle$' for i in range(SPAM_dimension)]
            y_positions = np.arange(SPAM_dimension)
            ax_side_RO.set_yticks(y_positions[::2])
            ax_side_RO.set_yticklabels(ytick_labels[::2],
                                       rotation=0,
                                       ha='right',
                                       fontsize=10)
            ax_side_RO.set_title('Readout')

            for it, rate in enumerate(meas_err_rate):
                ax_side_RO.text(
                    0,
                    it,
                    str(np.round(rate, 2)),
                    color="black" if rate > np.max(side_data_readout) /
                    2 else "white",
                    fontsize=12,
                    ha="center",
                    va="center")

            # main plot
            spam_data_infid = spam_data.copy()
            spam_data_infid[np.diag_indices(25)] = np.nan

            spam_data_diag = np.full(spam_data.shape, np.nan)
            np.fill_diagonal(spam_data_diag, np.diag(spam_data))

            # Plot the color map
            ax.imshow(spam_data_infid, cmap=cmap, origin="lower")
            ax.imshow(spam_data_diag, cmap=cmap2, origin="lower")

            norm = Normalize(vmin=np.nanmin(100 * spam_data_infid),
                             vmax=np.nanmax(100 * spam_data_infid))
            cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

            norm2 = Normalize(vmin=np.nanmin(100 * spam_data_diag),
                              vmax=np.nanmax(100 * spam_data_diag))
            cbar2 = plt.colorbar(cm.ScalarMappable(norm=norm2, cmap=cmap2),
                                 ax=ax)

            # Create the list of x tick labels
            xtick_labels = [rf'$|{i}\rangle$' for i in range(25)]
            x_positions = np.arange(25)  # Assuming 25 bars
            ax.set_xticks(x_positions[::2])
            ax.set_xticklabels(xtick_labels[::2],
                               rotation=0,
                               ha='center',
                               fontsize=14)

            # Create the list of y tick labels
            ytick_labels = [rf'$|{i}\rangle$' for i in range(25)]
            y_positions = np.arange(25)  # Get number of runs
            ax.set_yticks(y_positions[::2])
            ax.set_yticklabels(ytick_labels[::2],
                               rotation=0,
                               ha='right',
                               fontsize=14)

            # Set labels and title
            ax.set_ylabel('Prepared State')
            ax.set_xlabel('Measured State')
            ax.set_title(r'SPAM Infidelities ($10^{-2}$)')

            ax.text(0.6,
                    0.25,
                    f"Avg. Fidelity: {np.round(avg_fidelity,2)} " + r"$\pm$" +
                    f" {avg_fidelity_error}\%, \n" +
                    f"Avg. Data Rate: {np.round(avg_data_rate,2)}\%.",
                    color="white",
                    transform=ax.transAxes,
                    fontsize=12,
                    ha="left",
                    va="top")

            text_threshold = 0.001
            # Add text annotations for values above a certain threshold
            for i in range(spam_data_infid.shape[0]):
                for j in range(spam_data_infid.shape[1]):
                    value = spam_data_infid[i, j]
                    if value > text_threshold and i != j:
                        ax.text(j,
                                i,
                                str(np.round(100 *
                                             value, 1)) if value < 1 else 100,
                                color="black" if value
                                > np.nanmax(spam_data_infid) / 2 else "white",
                                fontsize=10,
                                ha="center",
                                va="center")

            text_threshold = 0.0
            # Add text annotations for values above a certain threshold
            for i in range(spam_data.shape[0]):
                value = spam_data[i, i]
                if value > text_threshold:
                    ax.text(i,
                            i,
                            str(np.round(100 *
                                         value, 1)) if value < 1 else 100,
                            color="black",
                            fontsize=10,
                            ha="center",
                            va="center")
        else:
            pass

        if plot_infidelity:
            ##########################################
            # Now for a 2D version of the figure #####
            ##########################################

            # Choose a colormap
            cmap = plt.get_cmap("viridis")

            # Create a figure and axis
            # fig, ax, ax_side = plt.subplots(figsize=(8, 6))

            fig, (ax, ax_side_INIT,
                  ax_side_RO) = plt.subplots(1,
                                             3,
                                             gridspec_kw={
                                                 'width_ratios':
                                                 [SPAM_dimension, 2, 2],
                                                 'wspace':
                                                 0.2,
                                             },
                                             figsize=(12, 1 + 8))

            # side-bar for initialisation error
            side_data_init = np.array(init_err_rate).reshape(-1, 1)
            ax_side_INIT.imshow(side_data_init,
                                cmap='viridis',
                                aspect='auto',
                                origin='lower',
                                vmin=0,
                                vmax=np.max(side_data_init))

            ax_side_INIT.set_xticks([])
            ax_side_INIT.set_xticklabels([])
            ytick_labels = [rf'$|{i}\rangle$' for i in range(SPAM_dimension)]
            y_positions = np.arange(SPAM_dimension)
            ax_side_INIT.set_yticks(y_positions[::2])
            ax_side_INIT.set_yticklabels(ytick_labels[::2],
                                         rotation=0,
                                         ha='right',
                                         fontsize=10)
            ax_side_INIT.set_title('Init.')

            for it, rate in enumerate(init_err_rate):
                ax_side_INIT.text(
                    0,
                    it,
                    str(np.round(rate, 2)),
                    color="black" if rate > np.max(side_data_init) /
                    2 else "white",
                    fontsize=12,
                    ha="center",
                    va="center")

            # side-bar for initialisation error
            side_data_readout = np.array(meas_err_rate).reshape(-1, 1)
            ax_side_RO.imshow(side_data_readout,
                              cmap='viridis',
                              aspect='auto',
                              origin='lower',
                              vmin=0,
                              vmax=np.max(side_data_readout))

            ax_side_RO.set_xticks([])
            ax_side_RO.set_xticklabels([])
            ytick_labels = [rf'$|{i}\rangle$' for i in range(SPAM_dimension)]
            y_positions = np.arange(SPAM_dimension)
            ax_side_RO.set_yticks(y_positions[::2])
            ax_side_RO.set_yticklabels(ytick_labels[::2],
                                       rotation=0,
                                       ha='right',
                                       fontsize=10)
            ax_side_RO.set_title('Readout')

            for it, rate in enumerate(meas_err_rate):
                ax_side_RO.text(
                    0,
                    it,
                    str(np.round(rate, 2)),
                    color="black" if rate > np.max(side_data_readout) /
                    2 else "white",
                    fontsize=12,
                    ha="center",
                    va="center")

            # main plot
            subtractor = 1 * np.eye(25)
            subtractor = subtractor[:25]
            spam_data_infid = np.abs(subtractor - spam_data)
            # Plot the color map
            ax.imshow(spam_data_infid, cmap=cmap, origin="lower")

            norm = Normalize(vmin=spam_data_infid.min(),
                             vmax=spam_data_infid.max())
            cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.viridis),
                                ax=ax)

            # Create the list of x tick labels
            xtick_labels = [rf'$|{i}\rangle$' for i in range(25)]
            x_positions = np.arange(25)  # Assuming 25 bars
            ax.set_xticks(x_positions[::2])
            ax.set_xticklabels(xtick_labels[::2],
                               rotation=0,
                               ha='center',
                               fontsize=14)

            # Create the list of y tick labels
            ytick_labels = [rf'$|{i}\rangle$' for i in range(25)]
            y_positions = np.arange(25)  # Get number of runs
            ax.set_yticks(y_positions[::2])
            ax.set_yticklabels(ytick_labels[::2],
                               rotation=0,
                               ha='right',
                               fontsize=14)

            # Set labels and title
            ax.set_ylabel('Prepared State')
            ax.set_xlabel('Measured State')
            ax.set_title(r'SPAM Infidelities ($10^{-2}$)')

            ax.text(0.6,
                    0.25,
                    f"Avg. Infidelity: {np.round(100-avg_fidelity,3)} " +
                    r"$\pm$" + f" {avg_fidelity_error}%, \n" +
                    f"Avg. Data Loss: {np.round(100-avg_data_rate,2)}%.",
                    color="white",
                    transform=ax.transAxes,
                    fontsize=12,
                    ha="left",
                    va="top")

            text_threshold = 0.001
            # Add text annotations for values above a certain threshold
            for i in range(spam_data_infid.shape[0]):
                for j in range(spam_data_infid.shape[1]):
                    value = spam_data_infid[i, j]
                    if value > text_threshold:
                        ax.text(
                            j,
                            i,
                            str(np.round(100 *
                                         value, 1)) if value < 1 else 100,
                            color="black" if value > np.max(spam_data_infid) /
                            2 else "white",
                            fontsize=12,
                            ha="center",
                            va="center")
        else:
            pass
    else:
        pass

    if do_state_choice_plot:

        sorting_indices_lowhigh = np.argsort(fidelities)
        sorting_indices_highlow = sorting_indices_lowhigh[::-1]

        optimal = fidelities[sorting_indices_highlow]
        worst = fidelities[sorting_indices_lowhigh]

        optimal_choice = []
        worst_choice = []
        optimal_choice_error_upper = []
        optimal_choice_error_lower = []
        worst_choice_error_upper = []
        worst_choice_error_lower = []
        qudit_dimensions = np.arange(2, 26, 1)

        for idx, dim in enumerate(qudit_dimensions):
            optimal_choice.append(np.mean(optimal[:idx + 2]))
            worst_choice.append(np.mean(worst[:idx + 2]))

            optimal_error_lower, optimal_error_upper = find_errors(
                1, np.mean(optimal[:idx + 2]), num_exps * dim)

            optimal_choice_error_upper.append(optimal_error_upper)
            optimal_choice_error_lower.append(optimal_error_lower)

            worst_error_lower, worst_error_upper = find_errors(
                1, np.mean(worst[:idx + 2]), num_exps * dim)

            worst_choice_error_upper.append(worst_error_upper)
            worst_choice_error_lower.append(worst_error_lower)

        optimal_choice = np.array(optimal_choice)
        optimal_choice_error_upper = np.array(optimal_choice_error_upper)
        optimal_choice_error_lower = np.array(optimal_choice_error_lower)

        worst_choice = np.array(worst_choice)
        worst_choice_error_upper = np.array(worst_choice_error_upper)
        worst_choice_error_lower = np.array(worst_choice_error_lower)

        fig, ax = plt.subplots(figsize=set_size(width='half'))

        ax.errorbar(qudit_dimensions,
                    optimal_choice,
                    fmt='o',
                    label='Optimal Choice',
                    yerr=[
                        np.abs(optimal_choice - optimal_choice_error_lower),
                        np.abs(optimal_choice_error_upper - optimal_choice)
                    ])
        ax.errorbar(qudit_dimensions,
                    worst_choice,
                    fmt='o',
                    label='Worst Choice',
                    yerr=[
                        np.abs(worst_choice - worst_choice_error_lower),
                        np.abs(worst_choice_error_upper - worst_choice)
                    ])
        ax.set_title('Qudit SPAM Fidelity vs. Dimension')
        ax.set_ylabel('Fidelity')
        ax.set_xlabel('Qudit Dimension')
        ax.legend()
        ax.grid()

    if long_output:
        return spam_data, fidelities, num_exps, init_errors, meas_errors
    else:
        return spam_data, init_err_rate, meas_err_rate


def compute_error_rates(measurement_order, init_err_rate, meas_err_rate):
    '''
    Takes measurements order and measured error rates and computes the expected
    error from transition pulse times and B-field sensitivities, as well as
    S_12 initialisation fidelities (which are taken from measurement).
    '''

    def generate_line_noise_detuning(t: float) -> float:
        '''This is the line signal at some time t,
        since the integral will give us the phase accumulated.

        Returns the line noise B-field value in uG.'''

        # following line signal noise values from 180Hz magnetic probe script
        line_amplitude_60 = 0.000225  # Gauss
        line_amplitude_180 = 0.00009  # Gauss
        line_phase_60 = -0.575  # radians
        line_phase_180 = -1.455  # radians
        line_offset = 0.00022346854601197586  # Gauss

        val = ((line_amplitude_60) *
               np.sin(2 * np.pi * 60 * t * 1e-6 + line_phase_60) +
               (line_amplitude_180) *
               np.sin(2 * np.pi * 180 * t * 1e-6 + line_phase_180) +
               line_offset)
        # ensure that the mixed signal starts at 0 amplitude
        # since the start of experiment sets the reference point
        val -= ((line_amplitude_60) * np.sin(line_phase_60) +
                (line_amplitude_180) * np.sin(line_phase_180) + line_offset)
        return val * 1e6

    cal = Calibrations(F2M0f2m0=545.5756, use_old_data=False)
    cal.generate_transition_frequencies()
    cal.generate_transition_sensitivities()
    cal.generate_transition_strengths()

    frequencies = cal.transition_frequencies
    sensitivities = cal.transition_sensitivities
    pitimes = cal.transition_pitimes

    all_shelving_transitions = [[-1, 4, -3]] + measurement_order

    shelving_indices = cal.convert_triplet_index(all_shelving_transitions)
    s12_levels = [int(t[0]) for t in all_shelving_transitions]
    s12_labels = [r'$m=-2$', r'$m=-1$', r'$m=0$', r'$m=1$', r'$m=2$']
    colour_map = [colors[s12_lev + 2] for s12_lev in s12_levels]

    freqs = []
    sens = []
    pitims = []
    omegas = []
    pulse_infids = []
    deshelve_infids = []
    for it, idx in enumerate(shelving_indices):
        freqs.append(frequencies[idx[0], idx[1]])
        sens.append(sensitivities[idx[0], idx[1]])
        pitims.append(pitimes[idx[0], idx[1]])

        omega = 1e6 * np.pi / pitimes[idx[0], idx[1]]  # Rabi freq in Hz
        omegas.append(omega)
        b_noise = 34  # micro-Gauss, from deviation to 180Hz signal
        detuning = 2 * np.pi * 124.94  # frequency miscalibration
        delta = np.abs(
            sensitivities[idx[0], idx[1]] * b_noise) + 2 * np.pi * detuning

        pulse_infid = 1 - ((omega**2 / (omega**2 + delta**2)) *
                           (np.sin(1e-6 * np.sqrt(omega**2 + delta**2) *
                                   pitimes[idx[0], idx[1]] / 2))**2)

        deshelve_pulse_timing = np.sum(pitims) + it * 5000
        deshelve_delta = np.abs(
            2 * np.pi * sensitivities[idx[0], idx[1]] *
            generate_line_noise_detuning(deshelve_pulse_timing))

        deshelve_infid = 1 - (
            (omega**2 / (omega**2 + deshelve_delta**2)) *
            (np.sin(1e-6 * np.sqrt(omega**2 + deshelve_delta**2) *
                    pitimes[idx[0], idx[1]] / 2))**2) + 0.875e-2

        pulse_infids.append(pulse_infid)
        deshelve_infids.append(deshelve_infid)

    # this is from thesis, data from April 15th, 2024
    s12_initialisations = [0.9973, 0.9891, 0.9864, 0.9955, 0.9755]
    # this data is from May 27-June 2, 2025
    s12_initialisations = [0.9955, 0.9883, 0.9865, 0.9827, 0.9802]

    # fitted manually (may have drifted since last April..)
    # s12_initialisations = [0.9873, 0.9791, 0.9814, 0.9955, 0.9855]

    init_err_theory = []
    for idx_trip, trip in enumerate(all_shelving_transitions):
        init_err_theory.append(
            np.round((1 - s12_initialisations[int(trip[0] + 2)]) +
                     pulse_infids[idx_trip], 3))

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 2))

    init_err_rate = np.array(init_err_rate) / 100
    lower_error, upper_error = find_errors(1, init_err_rate, 1000)
    asym_error_bar = np.abs(
        [init_err_rate - lower_error, upper_error - init_err_rate])

    for x, y, color, up_error, low_error in zip(np.arange(0,
                                                          25), init_err_rate,
                                                colour_map, asym_error_bar[0],
                                                asym_error_bar[1]):
        ax1.errorbar(x,
                     y,
                     yerr=low_error,
                     fmt='o',
                     markerfacecolor='none',
                     color=color)

    # Plot vertical lines and prepare distinct handles for the legend
    handles = []  # To hold the unique handles for legend
    labels = []  # To hold the corresponding labels
    for idx, err_val in enumerate(init_err_theory):
        vline = ax1.vlines(idx,
                           0,
                           init_err_theory[idx],
                           color=colour_map[idx],
                           label=s12_labels[int(s12_levels[idx] + 2)],
                           alpha=0.5)
        # Only add unique entries to the handles and labels
        if s12_labels[int(s12_levels[idx] + 2)] not in labels:
            handles.append(vline)
            labels.append(s12_labels[int(s12_levels[idx] + 2)])
    meas_err_rate = np.array(meas_err_rate) / 100
    lower_error, upper_error = find_errors(1, meas_err_rate, 1000)
    asym_error_bar = np.abs(
        [meas_err_rate - lower_error, upper_error - meas_err_rate])

    for x, y, color, up_error, low_error in zip(np.arange(0,
                                                          25), meas_err_rate,
                                                colour_map, asym_error_bar[0],
                                                asym_error_bar[1]):
        ax2.errorbar(x,
                     y,
                     yerr=low_error,
                     fmt='o',
                     markerfacecolor='none',
                     color=color)

    # Plot vertical lines and prepare distinct handles for the legend
    handles = []  # To hold the unique handles for legend
    labels = []  # To hold the corresponding labels
    meas_err_theory = []
    for idx, err_val in enumerate(init_err_theory):
        meas_err = deshelve_infids[idx]
        meas_err_theory.append(meas_err)
        vline = ax2.vlines(idx,
                           0,
                           meas_err,
                           color=colour_map[idx],
                           label=s12_labels[int(s12_levels[idx] + 2)],
                           alpha=0.5)

        # Only add unique entries to the handles and labels
        if s12_labels[int(s12_levels[idx] + 2)] not in labels:
            handles.append(vline)
            labels.append(s12_labels[int(s12_levels[idx] + 2)])

    # ax1.legend(handles, labels, title=r"$S_{1/2}$ Levels:", loc="upper left")
    # ax2.legend(handles, labels, title=r"$S_{1/2}$ Levels:", loc="upper left")

    ax1.set_title('Initialisation Errors')
    ax1.set_ylabel('Percent Error')
    ax1.set_ylim(0, 1.4 * np.max(init_err_rate))
    ax1.grid()

    ax2.set_title('Measurement Error')
    ax1.set_xlabel(r'Qudit State $|i\rangle$')
    ax2.set_xlabel(r'Qudit State $|i\rangle$')
    ax2.grid()

    ax1.tick_params(axis='both', labelsize=16)  # Set tick size to 10
    ax2.tick_params(axis='both', labelsize=16)  # Set tick size to 10

    print('Mean initialisation error (data):',
          np.round(100 * np.mean(init_err_rate), 4))
    print('Mean initialisation error (theory):',
          np.round(100 * np.mean(init_err_theory), 4))
    print('Mean measurement error (data):',
          np.round(100 * np.mean(meas_err_rate), 4))
    print('Mean measurement error (theory):',
          np.round(100 * np.mean(meas_err_theory), 4))

    return init_err_theory, meas_err_theory


def plot_spam_results(init_ket,
                      post_processing=True,
                      plot_2D=True,
                      plot_3D=False,
                      plotting=True,
                      plot_infidelity=False,
                      num_exps=1000,
                      reversed_readout=False,
                      threshold=10,
                      plot_threshold_histogram=False,
                      flipped_RO_order=False,
                      do_state_choice_plot=False,
                      long_output=False):

    file_path = (
        '/home/nicholas/Documents/' +
        f'Barium137_Qudit/data/25_level_5K_SPAM_data/InitKet_{init_ket}/')
    file_prefix = 'SPAM_10k_Heralded_'
    # date = date_time.split('_')
    # date_obj = datetime.strptime(date[0], '%Y%m%d')
    # plot_date = date_obj.strftime('%B %d, %Y')

    files = os.listdir(file_path)
    matching_files = []
    for fname in files:
        # Construct the full path to the file
        full_path = os.path.join(file_path, fname)

        # Check if the file exists and is a regular file
        if os.path.isfile(full_path):
            # Check if the search string exists in the file name
            if file_prefix in fname:
                matching_files.append(fname)
    # matching_files.sort()
    matching_files = np.sort(matching_files)
    if flipped_RO_order:
        matching_files = np.flip(matching_files)
    # print(matching_files)

    percentage_data_used = []
    init_errors = []
    deshelve_errors = []
    spam_data = np.zeros((25, len(matching_files)))
    for it in range(len(matching_files)):
        with open(file_path + matching_files[it], 'r') as filename:
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
        num_shelve_error = 0
        num_deshelve_error = 0
        for row_idx in range(binary_data.shape[0]):
            row = binary_data[row_idx]
            first_true_idx = np.argmax(row)  # Find the index of the first True
            # print(row)
            # print(first_true_idx)
            # break
            if post_processing:
                if sum(row) > 0 and first_true_idx:
                    first_true_bin_array[row_idx, first_true_idx] = True
                elif sum(row) == 0:
                    num_deshelve_error += 1
                elif first_true_idx == 0:
                    num_shelve_error += 1
            else:
                first_true_bin_array[row_idx, first_true_idx] = True

        summed_bin_array = np.sum(first_true_bin_array, axis=0) / (
            first_true_bin_array.shape[0] - num_shelve_error -
            num_deshelve_error)
        spam_data[:, it] = summed_bin_array[1:]
        # print(f'Ket number {it}, fidelity: ' +
        #       f'{np.round(100*summed_bin_array[it+1],2)},
        #          fails: {num_fails}')
        state_id = np.argmax(summed_bin_array)
        state_fid = np.round(100 * summed_bin_array[state_id], 3)
        data_rate = np.round(
            100 * (first_true_bin_array.shape[0] - num_shelve_error -
                   num_deshelve_error) / num_exps, 3)
        percentage_data_used.append(data_rate)
        init_errors.append(num_shelve_error)
        deshelve_errors.append(num_deshelve_error)
        print(f'|{state_id-1}>, fidelity: {state_fid},' +
              f' init. err: {num_shelve_error}, ' +
              f' meas. err: {num_deshelve_error},' +
              f' data rate: {data_rate},' +
              f' file: {matching_files[it][-16:]}')

    if reversed_readout:
        percentage_data_used = np.concatenate(
            ([percentage_data_used[0]], percentage_data_used[1:][::-1]))
        # We need to flip the RO order and the order of kets to be consistent
        # Flip the order of the columns (excluding the first column)
        flipped_columns = np.copy(
            spam_data)  # Create a copy of the original array
        # Reverse the columns from the second column onwards
        flipped_columns[:, 1:] = spam_data[:, 1:][:, ::-1]
        # Reverse the order of the rows (except first entry)
        result = flipped_columns[::-1, :]
        result = np.vstack((result[-1, :], result[:-1, :]))
        spam_data = result

    # Write in the average SPAM fidelity
    fidelities = spam_data.diagonal()
    fid_errors_lower, fid_errors_upper = find_errors(1, fidelities, num_exps)

    fid_errors = [fidelities - fid_errors_lower, fid_errors_upper - fidelities]
    avg_fidelity = np.mean(fidelities)
    avg_fidelity_error = np.round(
        avg_fidelity * np.sqrt(np.sum((fid_errors / fidelities)**2)), 3)
    avg_fidelity = np.round(100 * avg_fidelity, 2)
    avg_data_rate = np.round(np.mean(percentage_data_used), 3)
    print('\n')
    print('Average fidelity', avg_fidelity)
    print('Average fidelity uncertainty', avg_fidelity_error)
    print('Average data rate', avg_data_rate)
    print('Init errors total', np.sum(init_errors))
    print(
        'Init error rate',
        np.round(
            100 * (np.sum(init_errors) / (len(matching_files) * num_exps)), 2))
    print('Deshelve errors total', np.sum(deshelve_errors))
    print(
        'Deshelve error rate',
        np.round(
            100 * (np.sum(deshelve_errors) / (len(matching_files) * num_exps)),
            2), '\n')

    if plotting:
        ##########################################
        # Creating figures #######################
        ##########################################

        if plot_3D:
            # Normalize the data for color mapping
            norm = Normalize(vmin=spam_data.min(), vmax=spam_data.max())
            colors = plt.cm.viridis(spam_data.flatten() /
                                    float(spam_data.max()))

            # Create a 3D figure
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(111, projection='3d')

            # Create meshgrid for x and y coordinates
            x_indices, y_indices = np.meshgrid(np.arange(spam_data.shape[1]),
                                               np.arange(spam_data.shape[0]))

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
                     spam_data.flatten(),
                     shade=True,
                     color=colors,
                     alpha=0.8)

            # Add a color bar
            cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.viridis),
                                ax=ax)
            cbar.set_label('Probability')

            # Create the list of x tick labels
            xtick_labels = [
                rf'$|{i+1}\rangle$' for i in range(len(matching_files))
            ]
            # Set x tick positions and labels
            x_positions = np.arange(len(matching_files))  # Assuming 25 bars
            ax.set_xticks(x_positions[::3])
            # Rotate labels for readability
            ax.set_xticklabels(xtick_labels[::3],
                               rotation=0,
                               ha='center',
                               va='bottom',
                               fontsize=8)

            # Create the list of x tick labels
            ytick_labels = [
                rf'$|{i+1}\rangle$' for i in range(len(matching_files))
            ]
            # Set y tick positions and labels
            y_positions = np.arange(len(matching_files))  # Assuming 25 bars
            ax.set_yticks(y_positions[::3])
            # Rotate labels for readability
            ax.set_yticklabels(ytick_labels[::3],
                               rotation=0,
                               ha='center',
                               va='bottom',
                               fontsize=8)

            # Set labels and title
            ax.xaxis.set_label_position('top')
            ax.yaxis.set_label_position('top')
            ax.set_xlabel('Prepared State')
            ax.set_ylabel('Measured State', labelpad=10)
            ax.set_title('State Preparation and Measurement Fidelities')
        else:
            pass

        ##########################################
        # Now for a 2D version of the figure #####
        ##########################################

        spam_data = np.transpose(spam_data)
        if plot_2D:
            # Choose a colormap
            cmap = plt.get_cmap("viridis")

            # Create a figure and axis
            # fig, ax, ax_side = plt.subplots(figsize=(8, 6))

            fig, (ax, ax_side) = plt.subplots(
                1,
                2,
                gridspec_kw={
                    'width_ratios': [25, 2],
                    'wspace': 0.1,
                },
                figsize=(10, 1 + 8 * (len(matching_files) / 25)))

            # side-bar
            side_data = np.array(percentage_data_used).reshape(-1, 1)
            ax_side.imshow(side_data,
                           cmap='viridis',
                           aspect='auto',
                           origin='lower',
                           vmin=0,
                           vmax=100)

            for it, rate in enumerate(percentage_data_used):
                ax_side.text(0,
                             it,
                             str(np.round(rate, 2)),
                             color="black" if rate > 50 else "white",
                             fontsize=8,
                             ha="center",
                             va="center")

            # main plot
            # Plot the color map
            ax.imshow(spam_data, cmap=cmap, origin="lower")

            # Create the list of x tick labels
            xtick_labels = [rf'$|{i}\rangle$' for i in range(25)]
            x_positions = np.arange(25)  # Assuming 25 bars

            ytick_labels = [
                rf'$|{i}\rangle$' for i in range(len(matching_files))
            ]
            y_positions = np.arange(len(matching_files))  # Get number of runs
            ax.set_xticks(x_positions[::2])
            ax.set_xticklabels(xtick_labels[::2],
                               rotation=0,
                               ha='center',
                               fontsize=10)
            ax_side.set_yticks(y_positions[::2])
            ax_side.set_yticklabels(ytick_labels[::2],
                                    rotation=0,
                                    ha='right',
                                    fontsize=10)
            ax_side.set_xticks([])
            ax_side.set_xticklabels([])
            ax_side.set_title('Data Rate')

            # # Create the list of y tick labels
            # ytick_labels = [rf'$|{i}\rangle$' for i in range(25)]
            # y_positions = np.arange(25)  # Assuming 25 bars
            ax.set_yticks(y_positions[::2])
            ax.set_yticklabels(ytick_labels[::2],
                               rotation=0,
                               ha='right',
                               fontsize=10)

            # Set labels and title
            ax.set_ylabel('Prepared State')
            ax.set_xlabel('Measured State')
            # date = date_time.split('_')
            # date_obj = datetime.strptime(date[0], '%Y%m%d')
            # plot_date = date_obj.strftime('%B %d, %Y')
            ax.set_title('State Preparation and Measurement Fidelities')

            if len(matching_files) > 10:
                ax.text(0.6,
                        0.12,
                        f"Avg. Fidelity: {avg_fidelity} " + r"$\pm$" +
                        f" {avg_fidelity_error}%, \n" +
                        f"Avg. Data Rate: {avg_data_rate}%.",
                        color="white",
                        transform=ax.transAxes,
                        fontsize=10,
                        ha="left",
                        va="top")

            text_threshold = 0.0
            # Add text annotations for values above a certain threshold
            for i in range(spam_data.shape[0]):
                for j in range(spam_data.shape[1]):
                    value = spam_data[i, j]
                    if value > text_threshold:
                        ax.text(j,
                                i,
                                str(np.round(100 *
                                             value, 1)) if value < 1 else 100,
                                color="black" if value > 0.5 else "white",
                                fontsize=8,
                                ha="center",
                                va="center")

            fig.savefig('/home/nicholas/Documents/Barium137_Qudit/analysis/' +
                        '25_level_5K_SPAM/SPAM_5K_result_2D_graph.pdf')
        else:
            pass

        if plot_infidelity:
            ##########################################
            # Now for a 2D version of the figure #####
            ##########################################

            # Choose a colormap
            cmap = plt.get_cmap("viridis")

            # Create a figure and axis
            # fig, ax, ax_side = plt.subplots(figsize=(8, 6))

            fig, (ax, ax_side) = plt.subplots(
                1,
                2,
                gridspec_kw={
                    'width_ratios': [25, 2],
                    'wspace': 0.1,
                },
                figsize=(10, 1 + 8 * (len(matching_files) / 25)))

            # side-bar
            side_data = 100 - np.array(percentage_data_used).reshape(-1, 1)
            ax_side.imshow(
                side_data,
                cmap='viridis',
                aspect='auto',
                origin='lower',
            )

            for it, rate in enumerate(side_data):
                ax_side.text(0,
                             it,
                             str(np.round(rate[0], 2)),
                             color="black" if rate > np.max(side_data) /
                             2 else "white",
                             fontsize=8,
                             ha="center",
                             va="center")

            # main plot
            subtractor = 1 * np.eye(25)
            subtractor = subtractor[:len(matching_files)]
            spam_data_infid = np.abs(subtractor - spam_data)
            # Plot the color map
            ax.imshow(spam_data_infid, cmap=cmap, origin="lower")

            # Create the list of x tick labels
            xtick_labels = [rf'$|{i}\rangle$' for i in range(25)]
            x_positions = np.arange(25)  # Assuming 25 bars

            ytick_labels = [
                rf'$|{i}\rangle$' for i in range(len(matching_files))
            ]
            y_positions = np.arange(len(matching_files))  # Get number of runs
            ax.set_xticks(x_positions[::2])
            ax.set_xticklabels(xtick_labels[::2],
                               rotation=0,
                               ha='center',
                               fontsize=10)
            ax_side.set_yticks(y_positions[::2])
            ax_side.set_yticklabels(ytick_labels[::2],
                                    rotation=0,
                                    ha='right',
                                    fontsize=10)
            ax_side.set_xticks([])
            ax_side.set_xticklabels([])
            ax_side.set_title('Data Loss')

            # # Create the list of y tick labels
            # ytick_labels = [rf'$|{i}\rangle$' for i in range(25)]
            # y_positions = np.arange(25)  # Assuming 25 bars
            ax.set_yticks(y_positions[::2])
            ax.set_yticklabels(ytick_labels[::2],
                               rotation=0,
                               ha='right',
                               fontsize=10)

            # Set labels and title
            ax.set_ylabel('Prepared State')
            ax.set_xlabel('Measured State')
            # date = date_time.split('_')
            # date_obj = datetime.strptime(date[0], '%Y%m%d')
            # plot_date = date_obj.strftime('%B %d, %Y')
            ax.set_title(r'SPAM Infidelities ($10^{-2}$)')

            if len(matching_files) > 10:
                ax.text(0.6,
                        0.12,
                        f"Avg. Infidelity: {np.round(100-avg_fidelity,3)} " +
                        r"$\pm$" + f" {avg_fidelity_error}, \n" +
                        f"Avg. Data Loss: {np.round(100-avg_data_rate,2)}.",
                        color="white",
                        transform=ax.transAxes,
                        fontsize=10,
                        ha="left",
                        va="top")
            text_threshold = 0.0
            # Add text annotations for values above a certain threshold
            for i in range(spam_data_infid.shape[0]):
                for j in range(spam_data_infid.shape[1]):
                    value = spam_data_infid[i, j]
                    if value > text_threshold:
                        ax.text(
                            j,
                            i,
                            str(np.round(100 *
                                         value, 1)) if value < 1 else 100,
                            color="black" if value > np.max(spam_data_infid) /
                            2 else "white",
                            fontsize=8,
                            ha="center",
                            va="center")
        else:
            pass
    else:
        pass

    if do_state_choice_plot:

        sorting_indices_lowhigh = np.argsort(fidelities)
        sorting_indices_highlow = sorting_indices_lowhigh[::-1]

        optimal = fidelities[sorting_indices_highlow]
        worst = fidelities[sorting_indices_lowhigh]

        optimal_choice = []
        worst_choice = []
        optimal_choice_error_upper = []
        optimal_choice_error_lower = []
        worst_choice_error_upper = []
        worst_choice_error_lower = []
        qudit_dimensions = np.arange(2, 26, 1)

        for idx, dim in enumerate(qudit_dimensions):
            optimal_choice.append(np.mean(optimal[:idx + 2]))
            worst_choice.append(np.mean(worst[:idx + 2]))

            optimal_error_lower, optimal_error_upper = find_errors(
                1, np.mean(optimal[:idx + 2]), num_exps * dim)

            optimal_choice_error_upper.append(optimal_error_upper)
            optimal_choice_error_lower.append(optimal_error_lower)

            worst_error_lower, worst_error_upper = find_errors(
                1, np.mean(worst[:idx + 2]), num_exps * dim)

            worst_choice_error_upper.append(worst_error_upper)
            worst_choice_error_lower.append(worst_error_lower)

        optimal_choice = np.array(optimal_choice)
        optimal_choice_error_upper = np.array(optimal_choice_error_upper)
        optimal_choice_error_lower = np.array(optimal_choice_error_lower)

        worst_choice = np.array(worst_choice)
        worst_choice_error_upper = np.array(worst_choice_error_upper)
        worst_choice_error_lower = np.array(worst_choice_error_lower)

        fig, ax = plt.subplots(figsize=set_size(width='half'))

        ax.errorbar(qudit_dimensions,
                    optimal_choice,
                    fmt='o',
                    label='Optimal Choice',
                    yerr=[
                        np.abs(optimal_choice - optimal_choice_error_lower),
                        np.abs(optimal_choice_error_upper - optimal_choice)
                    ])
        ax.errorbar(qudit_dimensions,
                    worst_choice,
                    fmt='o',
                    label='Worst Choice',
                    yerr=[
                        np.abs(worst_choice - worst_choice_error_lower),
                        np.abs(worst_choice_error_upper - worst_choice)
                    ])
        ax.set_title('Qudit SPAM Fidelity vs. Dimension')
        ax.set_ylabel('Fidelity')
        ax.set_xlabel('Qudit Dimension')
        ax.legend()
        ax.grid()

    if long_output:
        return spam_data, fidelities, num_exps, init_errors, deshelve_errors
    else:
        return spam_data


if __name__ == '__main__':
    start_time = time.time()

    threshold, SPAM_dimension = get_threshold()

    measurement_order = [[0, 4, 0], [-2, 4, -4], [-1, 4, -3], [0, 4, -2],
                         [2, 4, 3], [-2, 3, -3], [0, 3, 1],
                         [2, 4, 4], [2, 4, 2], [-2, 3, -2], [1, 3, 3],
                         [0, 4, -1], [-1, 3, 0], [0, 3, 2], [-2, 3, -1],
                         [-2, 2, -1], [-2, 2, -2], [2, 2, 1], [0, 2, 0],
                         [0, 2, 2], [0, 4, 1], [-1, 1, -1], [-2, 1, 0],
                         [1, 1, 1]]

    # # putting together all SPAM plots
    initial_states = np.arange(0, 25)
    spam_data, init_err_rate, meas_err_rate = plot_all_spam_results(
        kets=initial_states,
        plot_3D=False,
        plot_2D=False,
        plot_infidelity=False,
        SPAM_dimension=25)
    mean_init_err = np.mean(init_err_rate*1e-2)
    mean_meas_err = np.mean(meas_err_rate*1e-2)
    # Get a simple standard deviation for error rates
    init_err_std = np.sqrt(mean_init_err*(1-mean_init_err)/(25*5000))
    meas_err_std = np.sqrt(mean_meas_err*(1-mean_meas_err)/(25*5000))
    print('Init err:', np.mean(init_err_rate), '+/-', init_err_std*1e2)
    print('Meas err:', np.mean(meas_err_rate), '+/-', meas_err_std*1e2)

    # computing the initialisation and measurement errors from theory
    compute_error_rates(measurement_order, init_err_rate, meas_err_rate)

    plt.tight_layout()
    plt.show()

    print("--- %s seconds ---" % (time.time() - start_time))
