import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson


def find_optimal_threshold(lam1, lam2):

    # Define the range of x values
    x = np.arange(0, 60)

    # Calculate the PMFs for both distributions
    pmf1 = poisson.pmf(x, lam1)
    pmf2 = poisson.pmf(x, lam2)

    # Initialize variables to find the optimal threshold
    min_error = float('inf')
    optimal_threshold = None
    false_high_rate = None
    false_low_rate = None

    # Evaluate each value in x as a potential threshold
    for threshold in x:
        # Calculate false-high and false-low rates
        current_false_high = np.sum(pmf1[threshold:])
        current_false_low = np.sum(pmf2[:threshold])
        total_error = current_false_high + current_false_low  # Total error

        # Update the optimal values if a lower error rate is found
        if total_error < min_error:
            min_error = total_error
            optimal_threshold = threshold
            false_high_rate = current_false_high
            false_low_rate = current_false_low

    # Plotting the PMFs with the identified threshold
    plt.figure(figsize=(10, 6))
    plt.bar(x - 0.2,
            pmf1,
            width=1,
            label=f'Poisson(λ={lam1})',
            color='blue',
            alpha=0.5)
    plt.bar(x + 0.2,
            pmf2,
            width=1,
            label=f'Poisson(λ={lam2})',
            color='orange',
            alpha=0.5)

    # Highlight the optimal threshold
    plt.axvline(optimal_threshold,
                color='red',
                linestyle='--',
                label=f'Optimal Threshold = {optimal_threshold}')

    plt.xlabel('Number of Events')
    plt.ylabel('Probability')
    plt.title('Optimal Threshold for Classifying Outcomes')
    plt.legend()
    plt.grid()

    return optimal_threshold, false_high_rate, false_low_rate


# this set has an error rate of 5.2e-5
optimal_thresh, false_high, false_low = find_optimal_threshold(0.5, 19.7)

# the following has an error rate of 2.28e-6
# optimal_thresh, false_high, false_low = find_optimal_threshold(0.5, 25.7)
print(f"Optimal threshold: {optimal_thresh}," +
      f" False-high rate: {false_high:.8f}, False-low rate: {false_low:.8f}")
print(f"Average error rate: {np.mean((false_high,false_low)):.8f}")

# With 2MHz of 493 photons in readout, and 80us PMT integration time, we can
# get 40 photons with a 25% collection + detection efficiency
# This implies a 0.86 NA lens to collect the fluorescence..

plt.show()
