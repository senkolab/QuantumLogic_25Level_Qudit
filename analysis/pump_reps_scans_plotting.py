import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from class_utils import Utils
from plot_utils import nice_fonts, set_size

find_errors = Utils.find_errors
mpl.rcParams.update(nice_fonts)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#7f7f7f']

# Define the folder path containing the txt files
folder_path = (
    '../data/S12_initialisation_data/S12_pump_reps_scan_initialisation')

# Initialize an empty list to store data points (2D array)
data_points = []

filenames = []
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        filenames.append(filename)

file_names = sorted(
    filenames, key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))
print(file_names)

for filename in file_names:
    # Iterate over each txt file in the folder
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r") as file:
            file_data = []  # Initialize a sub-array for each file
            for line in file:
                # Split the line by comma and convert to float
                x, y = map(float, line.strip().split(","))
                file_data.append([x, y])  # Append [x, y] to the sub-array
            data_points.append(
                file_data)  # Append the sub-array to the main 2D array

labels = []
for idx, filename in enumerate(file_names):
    print(filename)
    chunks = file_names[idx].split('_')
    labels.append(f"Init State: {chunks[9]}")
    print(labels[idx])

print(labels)


# Do sigmoid fit
def custom_sigmoid(x, a, b, c):
    return a + (b / np.exp(x / c))


initial_guesses = [1, -0.7, 1]
saturation_values = []
labels = [r'm=-2', r'm=-1', r'm=0', r'm=1', r'm=2']

fig, ax = plt.subplots(figsize=set_size(width='half'))
# Create a scatter plot
for it, file_data in enumerate(data_points):
    x_values, y_values = zip(*file_data)  # Unzip the data points for each file
    y_values = 1-np.array(y_values)
    lower_error, upper_error = find_errors(1, np.array(y_values), 501)
    asym_error_bar = [
        np.array(y_values) - lower_error, upper_error - np.array(y_values)
    ]
    asym_error_bar = np.absolute(asym_error_bar)

    try:
        popt, _ = curve_fit(custom_sigmoid,
                            x_values,
                            y_values,
                            p0=initial_guesses,
                            maxfev=10000)
        x_fit = np.linspace(0, 100, 300)
        saturation_values.append(max(custom_sigmoid(x_fit, *popt)))

        print(popt)
    except RuntimeError:
        print(f"Error fitting curve for {it}")

    plt.tick_params(axis='both', labelsize=14)  # Set tick size to 10
    ax.errorbar(
        x_values,
        y_values,
        yerr=asym_error_bar,
        fmt='--o',
        color=colors[it],
        # label=f"{labels[it]}, Sat: {np.round(100*popt[0],2)}")
        label=f"{labels[it]}")
    # plt.plot(x_fit, custom_sigmoid(x_fit, *popt), linestyle='--')

print(saturation_values)

# Create a scatter plot
plt.xlabel(r"Initialisation Repetitions ($n$)")
plt.ylabel("Initial State Probability")
# plt.yscale("log")
chunks = filenames[0].split('_')
plt.title(r"Probing Initialised State Probabilities, $S_{1/2}, F=2$")
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()
