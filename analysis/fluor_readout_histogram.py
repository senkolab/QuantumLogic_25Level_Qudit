import os
import sys
import time
from collections import Counter
from datetime import datetime
from itertools import zip_longest

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.signal import argrelextrema, savgol_filter

from plot_utils import nice_fonts, set_size

mpl.rcParams.update(nice_fonts)


def find_RO_error_rates(file_path):

    joint_histograms = []
    for idx, string in enumerate(['bright', 'dark']):

        file_prefix = 'fluorescent_readout_histogram_' + string
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

        # get pmt counts for thresholding
        pmt_counts = []
        for it in range(len(matching_files)):
            with open(file_path + matching_files[it], 'r') as filename:
                data = pd.read_csv(filename,
                                   delimiter="\[|\]|,|\"",
                                   engine='python',
                                   header=None).values
                data = data[:, 4:-19]
                pmt_counts += list(np.ravel(data))
        # Count the occurrences of each entry
        entry_counts = Counter(pmt_counts)
        # Extract unique entry values and their corresponding counts
        array_counts = [
            entry_counts.get(key, 0) for key in range(max(entry_counts) + 1)
        ]
        joint_histograms.append(array_counts)
        counts_dummy = np.arange(0, len(array_counts), 1)

        # Plot the histogram
    res = []
    for i, j in enumerate(joint_histograms[0]):
        if i < len(joint_histograms[1]):
            res.append((j, joint_histograms[1][i]))
        else:
            res.append((j, 0))

    joint_histograms = np.transpose(np.array(res))

    # find ideal threshold
    thresholds = np.arange(0.5, len(joint_histograms[0]), 1)
    false_dark_rates = np.zeros(len(thresholds))
    false_bright_rates = np.zeros(len(thresholds))
    false_dark_rates_uncertainty = np.zeros(len(thresholds))
    false_bright_rates_uncertainty = np.zeros(len(thresholds))
    for idx, threshold in enumerate(thresholds):
        false_dark_rates[idx] = np.sum(
            joint_histograms[0][:int(threshold + 0.5)]) / np.sum(
                joint_histograms[0])
        false_bright_rates[idx] = np.sum(
            joint_histograms[1][int(threshold + 0.5):]) / np.sum(
                joint_histograms[1])
        false_dark_rates_uncertainty[idx] = np.sqrt(
            false_dark_rates[idx] * (1 - false_dark_rates[idx]) /
            np.sum(joint_histograms[0]))
        false_bright_rates_uncertainty[idx] = np.sqrt(
            false_bright_rates[idx] * (1 - false_bright_rates[idx]) /
            np.sum(joint_histograms[1]))
    print('Data points for dark:', np.sum(joint_histograms[1]))
    print('Data points for bright:', np.sum(joint_histograms[0]))

    fig, ax = plt.subplots(figsize=set_size(width='full'))
    ax.plot(thresholds, false_dark_rates, label='False Negatives')
    ax.plot(thresholds, false_bright_rates, label='False Positives')

    # ax.set_yscale('log')
    ax.set_title(f'Fluorescent Readout Error Rates')
    ax.set_ylabel('Error Rates')
    ax.set_xlabel('Threshold')
    ax.legend()
    ax.grid()
    joint_error_rate = false_dark_rates + false_bright_rates
    ideal_threshold = np.argmin(joint_error_rate)
    ideal_threshold_error_rate = joint_error_rate[np.argmin(joint_error_rate)]
    error_rate_uncertainty = np.sqrt(
        np.square(false_dark_rates_uncertainty[np.argmin(joint_error_rate)]) +
        np.square(false_bright_rates_uncertainty[np.argmin(joint_error_rate)]))
    outputs = [
        joint_histograms, ideal_threshold, ideal_threshold_error_rate,
        error_rate_uncertainty
    ]
    return outputs


def plot_RO_histogram(joint_histograms, threshold=9):

    fig, ax = plt.subplots(figsize=set_size(width='half'))

    plot_labels = ['Bright', 'Dark']
    for idx, string in enumerate(['bright', 'dark']):

        counts_dummy = np.arange(0, len(joint_histograms[idx]), 1)

        # Plot the histogram
        ax.bar(counts_dummy,
               joint_histograms[idx],
               label=plot_labels[idx],
               alpha=0.7)
    ax.vlines(threshold,
              0,
              max(joint_histograms[1][:]),
              colors='k',
              linestyles='--',
              label='Threshold')
    ax.set_yscale('log')
    ax.set_title(f'Fluorescence Readout Bright Counts Histogram')
    ax.set_ylabel('Occurences')
    ax.set_xlabel('PMT Counts')
    ax.legend()
    ax.grid()
    res = []
    for i, j in enumerate(joint_histograms[0]):
        if i < len(joint_histograms[1]):
            res.append((j, joint_histograms[1][i]))
        else:
            res.append((j, 0))

    joint_histograms = np.transpose(np.array(res))
    return joint_histograms


file_path = ('../data/' +
             'fluorescent_readout_histogram_data/5ms_PMT_integration_time/')

outputs = find_RO_error_rates(file_path)
[
    joint_histograms, ideal_threshold, threshold_error_rate,
    error_rate_uncertainty
] = outputs
print('Ideal threshold', ideal_threshold)
print('Error rate at ideal threshold', threshold_error_rate)
print('Error rate uncertainty', error_rate_uncertainty)
joint_histograms = plot_RO_histogram(joint_histograms,
                                     threshold=ideal_threshold)

plt.tight_layout()
plt.show()
