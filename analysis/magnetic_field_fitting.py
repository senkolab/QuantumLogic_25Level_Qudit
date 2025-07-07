import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from plot_utils import nice_fonts

mpl.rcParams.update(nice_fonts)
np.set_printoptions(suppress=True)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# Define a Gaussian function
def generalized_gaussian(x, amp, mean, stddev, beta):
    return amp * np.exp(-0.5 * np.abs((x - mean) / stddev)**beta)


# Define a Lorentzian function
def lorentzian(x, amp, mean, gamma):
    return (amp * gamma**2) / ((x - mean)**2 + gamma**2)


def read_data(file_path):
    x_data, y_data, z_data = [], [], []

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(', ')
            x_val = float(parts[0].split(': ')[1])
            y_val = float(parts[1].split(': ')[1])
            z_val = float(parts[2].split(': ')[1])
            x_data.append(x_val)
            y_data.append(y_val)
            z_data.append(z_val)

    return np.array(x_data), np.array(y_data), np.array(z_data)


# Sturge's rule - some criticism of this and not optimally derived
def optimal_bins(data):
    N = len(data)
    return int(np.ceil(np.log2(N) + 1))


# # Freedman-Diaconis rule - returns "ugly" histograms..
# def optimal_bins(data):
#     """
#     Calculate the optimal number of bins for a histogram using the Freedman-Diaconis rule.

#     Parameters:
#     data (array-like): An array of numerical data points.

#     Returns:
#     int: The optimal number of bins.
#     """
#     # Calculate number of data points
#     N = len(data)

#     # Check if there are enough data points
#     if N < 2:
#         return 1  # If there's less than 2 data points, we can only have 1 bin

#     # Calculate the first and third quartiles
#     Q1 = np.percentile(data, 25)
#     Q3 = np.percentile(data, 75)

#     # Calculate the interquartile range (IQR)
#     IQR = Q3 - Q1

#     # Calculate the bin width using the Freedman-Diaconis rule
#     bin_width = 2 * (IQR / (N**(1 / 3)))

#     # Determine the range of the data
#     data_range = np.max(data) - np.min(data)

#     # Calculate the optimal number of bins
#     if bin_width == 0:
#         return 1  # Prevent division by zero if all data points are identical

#     optimal_bins = int(np.ceil(data_range / bin_width))

#     return optimal_bins


def chi_squared(y_data, y_fit, uncertainties):
    # Calculate the chi-squared statistic
    return np.sum(((y_data - y_fit) / uncertainties)**2)


def plot_histogram(data, axis_label, ax):
    num_bins = optimal_bins(data)  # Determine optimal bins
    count, bins, ignored = ax.hist(data,
                                   bins=num_bins,
                                   density=True,
                                   alpha=0.6,
                                   color=colors[0])

    mean = np.mean(data)
    stddev = np.std(data)
    gamma = stddev  # Set gamma as the standard deviation for initial guess

    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    initial_guess_gaussian = [300, mean, stddev, 2]
    initial_guess_lorentzian = [300, mean, gamma]

    # Fit Gaussian
    popt_gaussian, pcov_gaussian = curve_fit(generalized_gaussian,
                                             bin_centers,
                                             count,
                                             p0=initial_guess_gaussian)

    # Fit Lorentzian
    popt_lorentzian, pcov_lorentzian = curve_fit(lorentzian,
                                                 bin_centers,
                                                 count,
                                                 p0=initial_guess_lorentzian)

    # Calculate model fits
    y_fit_gaussian = generalized_gaussian(bin_centers, *popt_gaussian)
    y_fit_lorentzian = lorentzian(bin_centers, *popt_lorentzian)

    # Calculate chi-squared values assuming uniform uncertainties (for simplicity)
    uncertainties = np.sqrt(count / num_bins)  # Estimate uncertainties
    chi2_gaussian = chi_squared(count, y_fit_gaussian, uncertainties)
    chi2_lorentzian = chi_squared(count, y_fit_lorentzian, uncertainties)

    # Reduced chi-squared (number of data points - number of parameters)
    dof_gaussian = len(count) - len(popt_gaussian)
    dof_lorentzian = len(count) - len(popt_lorentzian)
    reduced_chi2_gaussian = chi2_gaussian / dof_gaussian
    reduced_chi2_lorentzian = chi2_lorentzian / dof_lorentzian

    print('Generalised Gaussian exponent', popt_gaussian[3], '+/-',
          np.sqrt(pcov_gaussian[3][3]))
    print('Reduced chi-squared Gaussian', reduced_chi2_gaussian)
    print('Reduced chi-squared Lorentzian', reduced_chi2_lorentzian)

    x_smooth = np.linspace(min(data), max(data), 1000)

    # Plotting Lorentzian fit
    y_smooth_lorentzian = lorentzian(x_smooth, *popt_lorentzian)
    # y_smooth_lorentzian = lorentzian(x_smooth, *initial_guess_lorentzian)
    ax.plot(x_smooth,
            y_smooth_lorentzian,
            color=colors[1],
            linewidth=2,
            label='Lorentzian Fit')

    # Plotting Gaussian fit
    y_smooth_gaussian = generalized_gaussian(x_smooth, *popt_gaussian)
    ax.plot(x_smooth,
            y_smooth_gaussian,
            color=colors[3],
            linewidth=2,
            label='Gaussian Fit')

    # Textbox with parameters
    textstr = (r'$\sigma$ = ' + f'{1e3*np.abs(popt_gaussian[2]):.3f} mG \n'
               r'$\gamma$ = ' + f'{1e3*np.abs(popt_lorentzian[2]):.3f} mG')
    ax.text(0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    ax.set_ylim(0, 700)
    ax.set_xlabel(axis_label)
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid()


def main(file_path):
    x_data, y_data, z_data = read_data(file_path)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    plot_histogram(x_data, r'$x$-axis / G', axs[0])
    plot_histogram(y_data, r'$y$-axis / G', axs[1])
    plot_histogram(z_data, r'$z$-axis / G', axs[2])

    fig.suptitle('Magnetic Field Noise Profile Across Axes', fontsize=16)
    plt.tight_layout()
    plt.show()


# Call the main function with your file path
file_path = ('../data/magnetic_field_data/2024_10_01_2049.txt')
main(file_path)
