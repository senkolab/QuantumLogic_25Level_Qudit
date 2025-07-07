"""
This data plotted in this file is taken with 6 probes for the initialised
state and single probes for the S12 levels that are expected to be empty.
This was done to disentangle off-resonant pumping from initialisation
fidelity."""

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from plot_utils import nice_fonts, set_size  # noqa: E402

# from scipy.optimize import curve_fit
mpl.rcParams.update(nice_fonts)


def make_single_s12_plot(file_name_search_string,
                         s12_states,
                         includes_dummy=False,
                         plot_title='Default Title'):

    file_names = [
        filename for filename in os.listdir(folder_path)
        if file_name_search_string in filename
    ]

    fig, ax = plt.subplots()
    avgs_array = []
    all_values_x = []
    all_values_y = []
    for idx, filename in enumerate(file_names):
        data_points = []

        with open(os.path.join(folder_path, filename), 'r') as f:
            for line in f:
                x, y = map(float, line.strip().split(','))
                data_points.append((x, y))
        x_values, y_values = zip(*data_points)
        all_values_x.append(np.mean(x_values))
        all_values_y.append(np.mean(y_values))

        if includes_dummy:
            plt.scatter(x_values,
                        y_values,
                        label=f'Probed State: {s12_states[idx]}')
            avgs_array.append(np.mean(y_values))
        else:
            plt.scatter(x_values,
                        y_values,
                        label=f'Probed State: {s12_states[idx+1]}')
            avgs_array.append(np.mean(y_values[:10]))
            avgs_array.append(np.mean(y_values[10:20]))
            avgs_array.append(np.mean(y_values[20:30]))
            avgs_array.append(np.mean(y_values[30:40]))
            avgs_array.append(np.mean(y_values[40:]))


    # Convert x_values and y_values to numpy arrays for easy manipulation
    x_array = np.array(all_values_x)
    y_array = np.array(all_values_y)

    # Sort indices based on x_array
    sorted_indices = np.argsort(x_array)

    # Sort x_array and y_array according to the sorted_indices
    sorted_x_values = x_array[sorted_indices]
    sorted_y_values = y_array[sorted_indices]

    plt.xlabel('Arbitrary')
    plt.ylabel('Dark State Probability')
    plt.title(f'{plot_title}')
    plt.grid(True)
    plt.legend()

    # print('Average population outcomes:', avgs_array)
    # print('Sum of all populations:', sum(avgs_array))
    return sorted_y_values


def s12_2D_init_plot(s12_states, includes_dummy=True):
    array_2D_init = np.zeros((5, 5))
    for idx, string in enumerate(s12_states[1:]):
        avg_array = make_single_s12_plot(string,
                                         s12_states,
                                         includes_dummy=True)
        array_2D_init[idx, :] = avg_array[1:]

    # print(array_2D_init)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=set_size(width='half'))

    # Choose a colormap
    cmap = plt.get_cmap("viridis")

    # Plot the color map
    im = ax.imshow(array_2D_init, cmap=cmap,
                   origin="lower")  # Set origin to lower left corner
    cbar = plt.colorbar(im, ax=ax)  # Add color bar with label
    cbar.set_label('Probability')

    # Set custom tick positions
    labels = ['-2', '-1', '0', '1', '2']
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    # Create the list of x tick labels
    tick_labels = [f'${i}$' for i in labels]
    ax.set_xticklabels(tick_labels, rotation=0, ha='center',fontsize=14)
    ax.set_yticklabels(tick_labels, rotation=90, va='center', ha='center',fontsize=14)
    plt.tick_params(axis='y', pad=10)  # Change the spacing for the ticks

    # # Create the list of y tick labels
    # ytick_labels = [rf'{dates[i]}' for i in range(len(date_times))]
    # ytick_labels.append('Average')
    # y_positions = np.arange(len(date_times) + 1)
    # ax.set_yticks(y_positions)
    # ax.set_yticklabels(ytick_labels, rotation=0, ha='right', fontsize=10)

    # Set labels and title
    ax.set_xlabel('Measured State')
    ax.set_ylabel('Prepared State')
    # ax.set_title(r'$S_{1/2}$ *SPAM* (6 probes for Init State, 1 Probe Else)')
    ax.set_title(r'$^2S_{1/2}$ State Preparation ($n=80$)')
    plt.tight_layout()

    # # Write in the average SPAM fidelity
    # avg_fidelity = np.mean(SPAM_fidelities.diagonal())
    # avg_fidelity = np.round(100 * avg_fidelity, 2)

    text_threshold = 0.5
    # Add text annotations for values above a certain threshold
    for i in range(array_2D_init.shape[0]):
        for j in range(array_2D_init.shape[1]):
            value = array_2D_init[i, j]
            if value >= text_threshold:
                ax.text(j,
                        i,
                        str(np.round(100 * value, 2)) + '%',
                        color="black" if value > 0.5 else "white",
                        fontsize=12,
                        ha="center",
                        va="center")
    fidelities = array_2D_init.diagonal()
    fid_errors = np.sqrt((1 - fidelities) * fidelities / 1000)

    avg_fidelity = np.mean(fidelities)
    avg_fidelity_error = np.round(
        avg_fidelity * np.sqrt(np.sum((fid_errors / fidelities)**2)), 4)
    print('Average initialisation fidelity', avg_fidelity)
    print('Average initialisation fidelity error', avg_fidelity_error)


if __name__ == '__main__':
    s12_states = ['dummy', 'mn2', 'mn1', 'mp0', 'mp1', 'mp2']

    # Define the folder path containing the txt files
    folder_path = ("/home/nicholas/Documents/Barium137_Qudit/" +
                   "data/S12_initialisation_data/" +
                   "S12_population_probing/S12_SPAM_80Inits_6Probes_PaperData")
    s12_2D_init_plot(s12_states, includes_dummy=True)

    plt.show()
