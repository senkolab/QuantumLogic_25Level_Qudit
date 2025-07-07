import ast
import itertools
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging import raiseExceptions
from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from path_finder import PathFinder

# from plot_utils import nice_fonts
# mpl.rcParams.update(nice_fonts)
# np.set_printoptions(suppress=True)


class StateFinder(PathFinder):

    def __init__(
            self,
            dimension: Union[float, int] = 3,
            B_field: float = 4.2165,
            ref_pitimes: list = [21.897, 41.031, 45.832, 35.6, 43.23],
            B_field_noise: float = 0.0000634,  # Gauss
            laser_noise_lorentzian: float = 0.000515,  # MHz
            laser_noise_gaussian: float = 0.0004775,  # MHz
            topology: str = 'All',  # can be 'All' or 'Star'
            type_cost:
        str = 'WithMediators',  # only relevant if topology is 'All'
            all_D52=False):

        super().__init__(B_field, ref_pitimes)

        # these noise values are taken from segmented_ramseys.py script results
        self.B_field_noise = B_field_noise
        self.laser_noise_gaussian = laser_noise_gaussian
        self.laser_noise_lorentzian = laser_noise_lorentzian

        self.transition_pitimes = np.loadtxt(
            'quick_reference_arrays/transition_pitimes.txt', delimiter=',')
        self.pitimes = self.transition_pitimes

        self.transition_sensitivities = np.loadtxt(
            'quick_reference_arrays/transition_sensitivities.txt',
            delimiter=',')

        if hasattr(self, 'transition_pitimes'):
            self.pitimes = self.transition_pitimes
        else:
            self.generate_transition_strengths()
            self.pitimes = self.transition_pitimes

        self.generate_delta_m_table()
        self.generate_cost_matrix()

        self.dimension = int(dimension)
        self.all_D52 = all_D52
        self.topology = topology
        self.type_cost = type_cost

        self.delta_m = np.flip(self.delta_m[:, -5:], axis=0)

        self.connection_graph = self.generate_coupling_graph()
        # load in paths dataframe
        self.df_paths = pd.read_csv('path_finder/all_paths_dataframe.csv')

        if self.dimension > 19 and self.topology == 'Star':
            raise ValueError("Error: Cannot find star " +
                             "topologies with dimension over 19.")

        self.generate_inverted_coherences()

    # some helper functions for conversion from array indices to state names
    def convert_index_to_state(self, num):
        if num >= 24:
            state = [0, int(26 - num)]
        else:
            state = self.F_values_D52[23 - num]
        return list(state)

    # and back
    def convert_state_to_index(self, state):
        helper = [1, 5, 11, 19]
        if state[0] == 0:
            index = int(26 - state[1])
        else:
            index = int(helper[int(state[0] - 1)] - state[1])
        return index

    # def generate_all_sensitivities(self, verbose=False):
    #     """Generate a matrix of sensitivities for various states based on the
    #     changes in energy values with respect to the magnetic field.

    #     This method computes the sensitivities of different quantum states to
    #     variations in the magnetic field, populating a matrix where each
    #     element represents the sensitivity between two states. The results can
    #     be visualized as a heatmap if specified by the `verbose` flag. The
    #     diagonal of the sensitivity matrix is filled with NaN values to
    #     indicate that a state does not have sensitivity to itself.

    #     Parameters:
    #     -----------
    #     verbose : bool, optional
    #         If set to True, a visual representation of the sensitivities will
    #         be displayed as a heatmap. Default is False.

    #     Returns:
    #     --------
    #     None

    #     Side Effects:
    #     --------------

    #     - Updates the instance attribute `self.all_sensitivities` with the
    #       computed sensitivity matrix.
    #     - Optionally displays a heatmap visualization of the sensitivities.

    #     Notes:
    #     -------

    #     - The method calculates slopes based on the energy evaluations for each
    #     state derived from magnetic field values in the `self.data_S12` and
    #     `self.data_D52` DataFrames.

    #     - The method utilizes the last three energy evaluations for
    #     calculating the slopes to determine the sensitivity of energy levels
    #     with respect to magnetic field variations.

    #     - The final sensitivity matrix is constructed by taking the difference
    #     between the slopes of states, and the elements below the anti-diagonal
    #     are negated for symmetry.

    #     """

    #     slopes_1d = []
    #     self.B_field_range = np.append(
    #         np.linspace(0.001, self.B_field - 0.0001, 10),
    #         np.array([self.B_field, self.B_field + 0.0001]))
    #     self.data_S12 = self.dataframe_evals_estates(orbital='S12')
    #     self.data_D52 = self.dataframe_evals_estates(orbital='D52')

    #     for it, D52_state in enumerate(reversed(self.F_values_D52)):
    #         slopes_1d.append(
    #             (self.data_D52[str(D52_state) + ' evalue'].iloc[-1] -
    #              self.data_D52[str(D52_state) + ' evalue'].iloc[-3]) /
    #             (self.data_S12['B field value'].iloc[-1] -
    #              self.data_S12['B field value'].iloc[-3]))
    #     for it2, S12_state in enumerate(reversed(self.F_values_S12[-5:])):
    #         slopes_1d.append(
    #             (self.data_S12[str(S12_state) + ' evalue'].iloc[-1] -
    #              self.data_S12[str(S12_state) + ' evalue'].iloc[-3]) /
    #             (self.data_S12['B field value'].iloc[-1] -
    #              self.data_S12['B field value'].iloc[-3]))

    #     slopes_1d = np.array(slopes_1d)
    #     column_slopes_1d = slopes_1d[:, np.newaxis]
    #     flipped_row = slopes_1d[::-1]
    #     all_sensitivities = np.array(column_slopes_1d - flipped_row)

    #     for i in range(all_sensitivities.shape[0]):
    #         for j in range(all_sensitivities.shape[0] - i - 1,
    #                        all_sensitivities.shape[0]):
    #             all_sensitivities[
    #                 i,
    #                 j] *= -1  # Multiply elements below the anti-diagonal by -1

    #     # Make entries with same state to same state filled with np.nan
    #     all_sensitivities = np.flipud(all_sensitivities)
    #     np.fill_diagonal(all_sensitivities, np.nan)
    #     all_sensitivities = np.flip(all_sensitivities)

    #     self.all_sensitivities = all_sensitivities

    #     if verbose:
    #         # create a new figure
    #         fig, ax = plt.subplots(figsize=(12, 8))

    #         im = plt.imshow(
    #             np.abs(all_sensitivities),
    #             aspect=0.7,
    #             # norm=norm,
    #             cmap='viridis_r')

    #         # Add a colorbar with a fraction of the axes height and padding
    #         cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    #         cbar.set_label(r'$\Delta_B$')

    #         S12_list = self.plotting_names_S12[-5:]
    #         D52_list = self.plotting_names_D52
    #         full_plotting_names = list(
    #             reversed([item
    #                       for item in S12_list] + [item for item in D52_list]))
    #         plt.xticks(np.arange(all_sensitivities.shape[1]),
    #                    full_plotting_names,
    #                    rotation=90)
    #         plt.yticks(np.arange(all_sensitivities.shape[0]),
    #                    full_plotting_names)

    #         for i in range(all_sensitivities.shape[0]):
    #             for j in range(all_sensitivities.shape[1]):
    #                 plt.text(j,
    #                          i,
    #                          f"{np.round(all_sensitivities[i,j],3)}",
    #                          fontsize=5,
    #                          ha="center",
    #                          va="center",
    #                          color="black" if np.abs(all_sensitivities[i, j])
    #                          < 4 else "white")

    #         # set the title
    #         plt.title(r"All Sensitivities / MHzG$^{-1}$")
    #         plt.tight_layout()

    #         # Save the plot as both PNG and PDF
    #         plt.savefig('state_finder/all_sensitivities_table.png')
    #         plt.savefig('state_finder/all_sensitivities_table.pdf')

    def generate_all_coherences(self, verbose=False):
        """OUTDATED APPROACH TO PAIRWISE COHERENCES

        Generate a 29x29 coherence matrix based on sensitivity data.

        This method calculates coherence times for a range of states based on
        the previously computed sensitivities. The coherence matrix quantifies
        the temporal stability of quantum states in the presence of noise,
        particularly in the context of magnetic field influences. If the
        `verbose` flag is set, a visual representation of the coherence matrix
        is displayed.

        Parameters:
        -----------
        verbose : bool, optional

            If set to True, a heatmap visualization of the coherence matrix
            will be displayed. Default is False.

        Returns:
        --------
        None

        Side Effects:
        --------------
        - Updates the instance attribute `self.all_coherences` with the
          constructed coherence matrix.

        - Optionally displays a heatmap visualization of the coherence matrix.

        Notes:
        -------
        - The method first checks if the `all_sensitivities` attribute exists;
        if not, it computes the sensitivities by calling
        `generate_all_sensitivities`.

        - The coherence times are computed using reference values for specific
        states, along with a noise parameter for the magnetic field.

        - The matrix is filled based on different conditions depending on the
        indices of the states being compared.

        - Coherence times are converted from seconds to milliseconds.

        - The visualization includes a color scale based on the logarithm of
        the absolute values of the matrix elements, with appropriate labels
        for the axes.

        """
        if hasattr(self, 'all_sensitivities'):
            pass
        else:
            self.generate_all_sensitivities()

        all_coherences = np.zeros((29, 29))

        # laser_noise = 0.000804  # MHz
        laser_noise = np.sum(self.laser_noise_gaussian +
                             self.laser_noise_lorentzian)

        # Calculate coherence times for each entry in the sensitivities matrix
        for (row, col), value in np.ndenumerate(self.all_sensitivities):
            # D-to-D transitions
            if row < 24 and col < 24:
                all_coherences[row, col] = 1 / np.abs(
                    self.all_sensitivities[row, col] * self.B_field_noise)
            # S-to-S transitions
            elif row >= 24 and col >= 24:
                all_coherences[row, col] = 1 / np.abs(
                    self.all_sensitivities[row, col] * self.B_field_noise)
            # S-to-D transitions
            else:
                all_coherences[row,
                               col] = (-laser_noise +
                                       np.sqrt(laser_noise**2 + 4 *
                                               (value * self.B_field_noise)**2)
                                       ) / (2 *
                                            (value * self.B_field_noise)**2)

        # convert coherence times to ms
        all_coherences = all_coherences * 1e-3
        self.all_coherences = all_coherences

        if verbose:
            # create a new figure
            fig, ax = plt.subplots(figsize=(12, 8))

            norm = mpl.colors.LogNorm(np.nanmin(np.abs(all_coherences)),
                                      np.nanmax(np.abs(all_coherences)),
                                      clip=True)

            im = plt.imshow(np.abs(all_coherences),
                            aspect=0.7,
                            norm=norm,
                            cmap='viridis')

            # Add a colorbar with a fraction of the axes height and padding
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(r'$T_2^*$')

            S12_list = self.plotting_names_S12[-5:]
            D52_list = self.plotting_names_D52
            full_plotting_names = list(
                reversed([item
                          for item in S12_list] + [item for item in D52_list]))
            plt.xticks(np.arange(all_coherences.shape[1]),
                       full_plotting_names,
                       rotation=90)
            plt.yticks(np.arange(all_coherences.shape[0]), full_plotting_names)

            for i in range(all_coherences.shape[0]):
                for j in range(all_coherences.shape[1]):
                    plt.text(
                        j,
                        i,
                        f"{np.round(all_coherences[i,j],3)}",
                        fontsize=5,
                        ha="center",
                        va="center",
                        color="black" if all_coherences[i,
                                                        j] > 1000 else "white")

            # set the title
            plt.title(r"All $T_2^*$ Times Table / $ms$")
            plt.tight_layout()

            # Save the plot as both PNG and PDF
            plt.savefig('state_finder/all_coherences_table.png')
            plt.savefig('state_finder/all_coherences_table.pdf')

    def generate_inverted_coherences(self, verbose=False):
        """Generate a 29x29 coherence matrix based on sensitivity data.

        This method calculates an entry of the form:

        1/T_laser,Lorentzian + (1/T_laser,Gaussian)^2 +
        (1/T_magnetic,Gaussian)^2

        for every possible pairwise combo from 6S1/2,F=2 to 5D5/2. This is meant
        to be used directly in the computing of a newer cost function. If
        the `verbose` flag is set, a visual representation is displayed.

        Parameters:
        -----------
        verbose : bool, optional

            If set to True, a heatmap visualization of the coherence matrix
            will be displayed. Default is False.

        Returns:
        --------
        None

        Side Effects:
        --------------
        - Creates the instance attribute `self.inverted_coherences` with the
          constructed coherence matrix.

        - Optionally displays a heatmap visualization of the coherence matrix.

        Notes:
        -------
        - The method first checks if the `all_sensitivities` attribute exists;
        if not, it computes the sensitivities by calling
        `generate_all_sensitivities`.

        - The coherence times are computed using reference values for specific
        states, along with a noise parameter for the magnetic field.

        - The matrix is filled based on different conditions depending on the
        indices of the states being compared.

        - Coherence times are converted from seconds to milliseconds.

        - The visualization includes a color scale based on the logarithm of
        the absolute values of the matrix elements, with appropriate labels
        for the axes.

        """
        if hasattr(self, 'all_sensitivities'):
            pass
        else:
            self.generate_all_sensitivities()

        inverted_coherences_laser_lorentz = np.zeros((29, 29))
        inverted_coherences_laser_gaussian = np.zeros((29, 29))
        inverted_coherences_magnet_gaussian = np.zeros((29, 29))

        # this new form of the coherences matrix is discussed in the November
        # 22nd, 2024 entry of the Ions Wiki for SenkoLab

        # take (1/T2)^2 for B-field noise squared (because Gaussian), take same
        # for Gaussian part of laser noise, then just 1/T2 for Lorentzian part
        # of laser noise
        for (row, col), value in np.ndenumerate(self.all_sensitivities):
            # D-to-D transitions
            if row < 24 and col < 24:
                inverted_coherences_magnet_gaussian[row, col] = (np.abs(
                    self.all_sensitivities[row, col] * self.B_field_noise))**2
            # S-to-S transitions
            elif row >= 24 and col >= 24:
                inverted_coherences_magnet_gaussian[row, col] = (np.abs(
                    self.all_sensitivities[row, col] * self.B_field_noise))**2
            # S-to-D transitions
            else:
                inverted_coherences_laser_lorentz[
                    row, col] = self.laser_noise_lorentzian

                inverted_coherences_laser_gaussian[
                    row, col] = self.laser_noise_gaussian**2

                inverted_coherences_magnet_gaussian[row, col] = (np.abs(
                    self.all_sensitivities[row, col] * self.B_field_noise))**2

        self.inverted_coherences_laser_lorentz = inverted_coherences_laser_lorentz
        self.inverted_coherences_laser_gaussian = inverted_coherences_laser_gaussian
        self.inverted_coherences_magnet_gaussian = inverted_coherences_magnet_gaussian

        # if verbose:
        #     # create a new figure
        #     fig, ax = plt.subplots(figsize=(12, 8))

        #     norm = mpl.colors.LogNorm(np.nanmin(np.abs(inverted_coherences)),
        #                               np.nanmax(np.abs(inverted_coherences)),
        #                               clip=True)

        #     im = plt.imshow(np.abs(inverted_coherences),
        #                     aspect=0.7,
        #                     norm=norm,
        #                     cmap='viridis')

        #     # Add a colorbar with a fraction of the axes height and padding
        #     cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        #     cbar.set_label(r'$T_2^*$')

        #     S12_list = self.plotting_names_S12[-5:]
        #     D52_list = self.plotting_names_D52
        #     full_plotting_names = list(
        #         reversed([item
        #                   for item in S12_list] + [item for item in D52_list]))
        #     plt.xticks(np.arange(inverted_coherences.shape[1]),
        #                full_plotting_names,
        #                rotation=90)
        #     plt.yticks(np.arange(inverted_coherences.shape[0]),
        #                full_plotting_names)

        #     for i in range(inverted_coherences.shape[0]):
        #         for j in range(inverted_coherences.shape[1]):
        #             plt.text(j,
        #                      i,
        #                      f"{np.round(inverted_coherences[i,j],3)}",
        #                      fontsize=5,
        #                      ha="center",
        #                      va="center",
        #                      color="black"
        #                      if inverted_coherences[i, j] > 1000 else "white")

        #     # set the title
        #     plt.title(r"All $T_2^*$ Times Table / $ms$")
        #     plt.tight_layout()

        #     # Save the plot as both PNG and PDF
        #     plt.savefig('state_finder/inverted_coherences_table.png')
        #     plt.savefig('state_finder/inverted_coherences_table.pdf')

    def generate_all_paths_tensor(self, recompute_paths=False):
        """Generate "matrix" of 29x29 transitions where each entry is actually a
        list which is the path-information for that pair.
        """

        all_paths = np.full((29, 29, 10), np.nan)

        all_path_states = [
            f'[{i[0]}, {i[1]}]' for i in np.flip(self.F_values_D52, axis=0)
        ] + [f'{i[1]}' for i in np.flip(self.F_values_S12[3:], axis=0)]

        if not recompute_paths:
            filename = f'path_finder/all_paths_dataframe.csv'
            if os.path.exists(filename):
                df_paths = pd.read_csv(filename)
            else:
                df_paths = self.find_all_paths()
        else:
            df_paths = self.find_all_paths()

        # Loop over all_path_states pairs
        for row, init_state in enumerate(all_path_states):
            for col, final_state in enumerate(all_path_states):
                if row != col:
                    # Filter the DataFrame to find the df_entry where "Initial
                    # State" and "Final State" match
                    df_entry = df_paths[
                        (df_paths['Initial State'] == str(init_state))
                        & (df_paths['Final State'] == str(final_state))]

                    # If the df_entry exists (i.e., there is a match), extract
                    # the "Effective Pi-Time"
                    pathway = df_entry['Path'].values[0]
                    # print(pathway)
                    # print(ast.literal_eval(pathway))
                    # print(type(ast.literal_eval(pathway)))

                    padded_array = np.full(10, '', dtype=object)
                    # Fill the array with the input list elements
                    padded_array[:len(ast.literal_eval(pathway)
                                      )] = ast.literal_eval(pathway)
                    # print(padded_array)
                    for idx, entry in enumerate(padded_array):

                        entry = entry.strip(
                        )  # Remove any surrounding whitespace

                        # Try converting to float first
                        try:
                            num_entry = int(26 - float(entry))
                        except ValueError:
                            # If it fails, check if it's a list
                            if entry.startswith("[") and entry.endswith("]"):
                                # Convert to a numpy array from the list
                                arr = np.array(ast.literal_eval(
                                    entry))  # Safely evaluate the list
                                # Convert array to an int corresponding to entry in 29x29 matrices
                                helpers = np.array([1, 5, 11, 19])
                                arr_index = int(helpers[int(arr[0] - 1)] -
                                                arr[1])
                                num_entry = int(
                                    arr_index)  # Add array to the list
                            else:
                                # If it's neither, make it a nan
                                num_entry = np.nan

                        all_paths[row, col, idx] = num_entry

        # # need to clean up the ordering of these paths
        # for row, init_state in enumerate(all_path_states):
        #     for col, final_state in enumerate(all_path_states):
        #         if row < col:
        #             if int(all_paths[row,col,0]) != int(row):
        #                 np.flip(all_paths[row,col,:])
        #             else:
        #                 pass
        #         if row > col:
        #             if int(all_paths[row,col,0]) != int(col):
        #                 np.flip(all_paths[row,col,:])
        #             else:
        #                 pass

        self.all_paths = all_paths

    def generate_all_pitimes(self, recompute_paths=False, verbose=False):
        """Generate a matrix of effective pi-times for all pairs of initial and
        final states.

        This method computes effective pi-times between various initial and
        final states, populating a 29x29 matrix. The matrix is then saved to a
        text file to avoid recalculating it in future calls. If the matrix
        needs to be recalculated, the method will read from an existing CSV
        file containing path data if available. The results can be visualized
        in a color-coded heatmap if specified by the `verbose` flag.

        Parameters:
        -----------

        recompute_paths : bool, optional
            If set to True, will recalculate the optimal paths. Default is
            False.

        verbose : bool, optional
            If set to True, a visual representation of the effective pi-times
            will be displayed as a heatmap. Default is False.

        Returns:
        --------
        None

        Side Effects:
        --------------
        - Generates and saves a text file containing the effective pi-times
          matrix.
        - Updates the instance attribute `self.all_pitimes` with the computed
          matrix.
        - Optionally displays a heatmap visualization of the matrix.

        Notes:
        -------

        - The effective pi-times are obtained from a previously computed paths
        DataFrame, either by reading an existing file or by invoking the
        `find_all_paths` method to compute them anew.

        - The diagonal of the effective pi-times matrix is filled with NaN
        values, as these represent the time of a state transitioning to itself,
        which is not applicable.

        """

        all_pitimes = np.zeros((29, 29))

        all_path_states = [
            f'[{i[0]}, {i[1]}]' for i in np.flip(self.F_values_D52, axis=0)
        ] + [f'{i[1]}' for i in np.flip(self.F_values_S12[3:], axis=0)]

        if not recompute_paths:
            filename = f'path_finder/all_paths_dataframe.csv'
            if os.path.exists(filename):
                df_paths = pd.read_csv(filename)
            else:
                df_paths = self.find_all_paths()
        else:
            df_paths = self.find_all_paths()

        # Loop over all_path_states pairs
        for row, init_state in enumerate(all_path_states):
            for col, final_state in enumerate(all_path_states):
                if row < col:
                    # Filter the DataFrame to find the df_entry where "Initial
                    # State" and "Final State" match
                    df_entry = df_paths[
                        (df_paths['Initial State'] == str(init_state))
                        & (df_paths['Final State'] == str(final_state))]

                    # If the df_entry exists (i.e., there is a match), extract
                    # the "Effective Pi-Time"
                    effective_pitime = df_entry['Effective Pi-Time'].values[0]
                    all_pitimes[row, col] = effective_pitime
                    all_pitimes[col, row] = effective_pitime

        # flip output array to get right ordering of columns
        all_pitimes = np.flip(all_pitimes, axis=1)

        # Make entries with same state to same state filled with np.nan
        all_pitimes = np.flipud(all_pitimes)
        np.fill_diagonal(all_pitimes, np.nan)
        all_pitimes = np.flip(all_pitimes)

        self.all_pitimes = all_pitimes

        if verbose:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Normalize the color scale
            norm = mpl.colors.LogNorm(np.nanmin(all_pitimes),
                                      np.nanmax(all_pitimes),
                                      clip=True)

            # Display the matrix with the specified color map and normalization
            im = ax.imshow(np.abs(all_pitimes),
                           aspect=0.7,
                           norm=norm,
                           cmap='viridis_r')

            # Add a colorbar with a fraction of the axes height and padding
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(r'$\pi$-time')

            # Set tick labels
            S12_list = self.plotting_names_S12[-5:]
            D52_list = self.plotting_names_D52
            full_plotting_names = list(
                reversed([item
                          for item in S12_list] + [item for item in D52_list]))

            # Set the tick labels and rotate the x-ticks
            plt.xticks(np.arange(all_pitimes.shape[1]),
                       full_plotting_names,
                       rotation=90)
            plt.yticks(np.arange(all_pitimes.shape[0]), full_plotting_names)

            # Add text annotations for each cell in the matrix
            for i in range(all_pitimes.shape[0]):
                for j in range(all_pitimes.shape[1]):
                    plt.text(
                        j,
                        i,
                        f"{np.round(all_pitimes[i,j],3)}",
                        fontsize=5,
                        ha="center",
                        va="center",
                        color="black" if all_pitimes[i, j] < 100 else "white")

            # Set the title of the plot
            plt.title(r"All $\tau_\pi$ Table / $\mu s$")
            plt.tight_layout()

            # Save the plot as both PNG and PDF
            plt.savefig('state_finder/all_pitimes_table.png')
            plt.savefig('state_finder/all_pitimes_table.pdf')

    def generate_cost_matrix(self, recompute_paths=False, verbose=False):
        """Generate a 29x29 cost matrix based on effective pi-times and
        coherences.

        This method constructs an cost matrix by dividing the effective
        pi-times by the corresponding coherence values. The resulting matrix
        represents the sensitivities of indirect transitions between states.
        The method also visualizes the cost matrix as a heatmap if the
        `verbose` flag is set to True.

        Parameters:
        -----------
        recompute_paths : bool, optional
            If set to True, the method will recompute paths even if they are
            already calculated. Default is False.

        verbose : bool, optional
            If set to True, a visual representation of the cost matrix
            will be displayed as a heatmap. Default is False.

        Returns:
        --------
        np.ndarray
            A 29x29 cost matrix constructed from effective pi-times
            divided by coherences.

        Side Effects:
        --------------
        - Updates the instance attribute `self.cost_matrix` with the
          constructed matrix.

        - Optionally displays a heatmap visualization of the cost matrix.

        Notes:
        -------
        - The method relies on the `generate_all_pitimes` and
        `generate_all_coherences` methods to compute the necessary data for
        constructing the cost matrix.

        - The visualization includes a color scale based on the logarithm of
        the absolute values of the elements in the matrix, with appropriate
        labels for the axes.

        """

        # generate the necessary helper pitimes and coherences matrices
        self.generate_all_pitimes(recompute_paths=recompute_paths,
                                  verbose=verbose)

        self.generate_all_coherences(verbose=verbose)

        cost_matrix = self.all_pitimes / self.all_coherences

        if verbose:
            # create a new figure
            fig, ax = plt.subplots(figsize=(12, 8))

            norm = mpl.colors.LogNorm(np.nanmin(np.abs(cost_matrix)),
                                      np.nanmax(np.abs(cost_matrix)),
                                      clip=True)

            im = plt.imshow(np.abs(cost_matrix),
                            aspect=0.7,
                            norm=norm,
                            cmap='viridis_r')

            # Add a colorbar with a fraction of the axes height and padding
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(r'$\tau_\pi/T_2^*$')

            S12_list = self.plotting_names_S12[-5:]
            D52_list = self.plotting_names_D52
            full_plotting_names = list(
                reversed([item
                          for item in S12_list] + [item for item in D52_list]))
            plt.xticks(np.arange(cost_matrix.shape[1]),
                       full_plotting_names,
                       rotation=90)
            plt.yticks(np.arange(cost_matrix.shape[0]), full_plotting_names)

            for i in range(cost_matrix.shape[0]):
                for j in range(cost_matrix.shape[1]):
                    plt.text(j,
                             i,
                             f"{np.round(cost_matrix[i,j],3)}",
                             fontsize=5,
                             ha="center",
                             va="center",
                             color="black" if cost_matrix[i,
                                                          j] < 10 else "white")

            # set the title
            plt.title(r"all $\tau_\pi/T_2^*$")
            plt.tight_layout()

            # Save the plot as both PNG and PDF
            plt.savefig('state_finder/cost_matrix_table.png')
            plt.savefig('state_finder/cost_matrix_table.pdf')

        self.cost_matrix = cost_matrix

    def generate_two_way_cost_matrix(self,
                                     recompute_paths=False,
                                     verbose=False):

        # generate the necessary helper pitimes and coherences matrices
        self.generate_all_pitimes(recompute_paths=recompute_paths,
                                  verbose=False)

        self.generate_all_coherences(verbose=False)
        self.generate_all_paths_tensor()

        two_way_cost_matrix = np.full((29, 29), np.nan)

        # populate the two way cost matrix
        for idx1 in range(29):
            for idx2 in range(29):
                array = self.all_paths[idx1, idx2]
                if np.all(np.isnan(array)):
                    pass
                else:
                    group = array[~np.isnan(array)].astype(int).tolist()
                    cost = self.cost(group)
                    two_way_cost_matrix[idx1, idx2] = cost
                    two_way_cost_matrix[idx2, idx1] = cost

        if verbose:
            # create a new figure
            fig, ax = plt.subplots(figsize=(12, 8))

            norm = mpl.colors.LogNorm(np.nanmin(np.abs(two_way_cost_matrix)),
                                      np.nanmax(np.abs(two_way_cost_matrix)),
                                      clip=True)

            im = plt.imshow(np.abs(two_way_cost_matrix),
                            aspect=0.7,
                            norm=norm,
                            cmap='viridis_r')

            # Add a colorbar with a fraction of the axes height and padding
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(r'$\tau_\pi/T_2^*$')

            S12_list = self.plotting_names_S12[-5:]
            D52_list = self.plotting_names_D52
            full_plotting_names = list(
                reversed([item
                          for item in S12_list] + [item for item in D52_list]))
            plt.xticks(np.arange(two_way_cost_matrix.shape[1]),
                       full_plotting_names,
                       rotation=90)
            plt.yticks(np.arange(two_way_cost_matrix.shape[0]),
                       full_plotting_names)

            for i in range(two_way_cost_matrix.shape[0]):
                for j in range(two_way_cost_matrix.shape[1]):
                    plt.text(j,
                             i,
                             f"{np.round(two_way_cost_matrix[i,j],3)}",
                             fontsize=5,
                             ha="center",
                             va="center",
                             color="black"
                             if two_way_cost_matrix[i, j] < 10 else "white")

            # set the title
            plt.title(r"All Pair-Wise Cost Values")
            plt.tight_layout()

            # Save the plot as both PNG and PDF
            plt.savefig('state_finder/two_way_cost_matrix_table.png')
            plt.savefig('state_finder/two_way_cost_matrix_table.pdf')

        self.two_way_cost_matrix = two_way_cost_matrix

    def cost(self, group: list) -> float:
        """cost function to evaluate the score of a group based on the
        all-to-all connectivity and effective coherence.

        Parameters:

        pitimes (ndarray): Array of pitimes for all transitions.
        sensitivities (ndarray): Array of sensitivities for all transitions.
        group (list): List of tuples containing (row, col) indices.

        Returns:

        float: The calculated score for the group. This score is the mean of
        (effective pitime)/(effective coherence time) for {i,j} where i,j are
        the elements of the set of states. There are therefore (dimension
        choose 2) such pairings to take into account.

        """
        if self.topology == 'All':
            unique_pairs = list(itertools.combinations(group, 2))
        elif self.topology == 'Star':
            unique_pairs = [(group[0], group[i]) for i in range(1, len(group))]

        score = 0.0

        if self.topology == 'All':
            if self.type_cost == 'Euclidean':
                # old version of cost - taking tau_pi/T_2 per transition
                # this version used the cost matrix we made
                for r, c in unique_pairs:
                    score += self.cost_matrix[r, c]
                cost = score / len(unique_pairs)

            elif self.type_cost == 'Manhattan':
                # new version - taking sum of tau_pi's over sum of T_2's
                pis = 0
                t2s = 0
                for r, c in unique_pairs:
                    pis += self.all_pitimes[r, c]
                    t2s += 1 / self.all_coherences[r, c]
                cost = pis * t2s

            elif self.type_cost == 'WithMediators':
                pis = 0
                inverted_t2s_laser_lorentz = 0
                inverted_t2s_laser_gaussian = 0
                inverted_t2s_magnet_gaussian = 0
                for r, c in unique_pairs:
                    pis += self.all_pitimes[r, c]
                for r, c in unique_pairs:
                    inverted_t2s_laser_gaussian += pis**2 * self.inverted_coherences_laser_gaussian[
                        r, c]
                    inverted_t2s_laser_lorentz += pis * self.inverted_coherences_laser_lorentz[
                        r, c]
                    inverted_t2s_magnet_gaussian += pis**2 * self.inverted_coherences_magnet_gaussian[
                        r, c]
                cost = (inverted_t2s_laser_gaussian +
                        inverted_t2s_laser_lorentz +
                        inverted_t2s_magnet_gaussian) / (len(group))
            else:
                pass
        elif self.topology == 'Star':
            pis = 0
            inverted_t2s_laser_lorentz = 0
            inverted_t2s_laser_gaussian = 0
            inverted_t2s_magnet_gaussian = 0
            for c, r in unique_pairs:
                pis += self.pitimes[r, int(28 - c)]
            for c, r in unique_pairs:
                inverted_t2s_laser_gaussian += pis**2 * self.inverted_coherences_laser_gaussian[
                    r, c]
                inverted_t2s_laser_lorentz += pis * self.inverted_coherences_laser_lorentz[
                    r, c]
                inverted_t2s_magnet_gaussian += pis**2 * self.inverted_coherences_magnet_gaussian[
                    r, c]
            cost = (inverted_t2s_laser_gaussian + inverted_t2s_laser_lorentz +
                    inverted_t2s_magnet_gaussian) / (len(group))

        return cost

    def random_group(self, print_initial_group=False):
        """Generate a random connected group starting from a random point,
        ensuring the number of unique rows + unique columns equals a specified
        number.

        Parameters:
        - target_unique_count (int): The target number of unique
        rows + unique columns.
        - print_initial_group (bool): Whether to print the initial group.

        Returns:
        - list: A randomly generated connected group.

        """

        if self.topology == 'All':
            # create a list of numbers from 0-23 (or 0-28 if including S12 states)
            if self.all_D52:
                group = random.sample(range(0, 24), self.dimension)
            else:
                group = random.sample(range(0, 29), self.dimension)

        elif self.topology == 'Star':
            if self.dimension < 15:
                init_s12_state = random.choice(range(24, 29))
            elif self.dimension < 18 and self.dimension >= 15:
                init_s12_state = random.choice(range(25, 28))
            elif self.dimension == 18:
                init_s12_state = 26

            group = [init_s12_state]
            non_nan_indices = np.where(
                ~np.isnan(self.pitimes[:, 28 - init_s12_state]))[0]
            self.star_indices = non_nan_indices

            while len(group) < self.dimension:
                new_node = random.choice(non_nan_indices)
                if new_node not in group:
                    group.append(new_node)

        # Debug print of generated group
        if print_initial_group:
            random_group_members = []
            for num in np.array(group):
                if num >= 24:
                    # convert index in 29x29 array to S1/2 m value
                    random_group_members.append(str(int(26 - num)))
                else:
                    # convert index to D5/2 state
                    random_group_members.append(
                        str(self.F_values_D52[23 - num]))

            print('\n Random initial group: \n ' +
                  f'{np.array(random_group_members)}')
            print('\n Random initial group length: \n ' +
                  f'{len(np.array(random_group_members))}')

        return group

    def modify_group(self, group):
        """Change one member of the group while maintaining connectivity and
        ensuring the total number of unique rows + unique columns remains
        constant.

        Parameters:
        - group (list): List of tuples containing (row, col) indices.

        Returns:
        - list: The modified group if connectivity is maintained,
        otherwise the original group.

        """

        if self.topology == 'All':
            new_group = group.copy()
            # Choose a random index to modify
            index_to_replace = random.randint(0, len(group) - 1)

            # Generate a new integer that is not already in the list
            if self.all_D52:
                new_integer = random.randint(0, 23)
                while new_integer in new_group:
                    new_integer = random.randint(0, 23)
            else:
                new_integer = random.randint(0, 28)
                while new_integer in new_group:
                    new_integer = random.randint(0, 28)

            # Replace the selected index with the new integer
            new_group[index_to_replace] = new_integer

        elif self.topology == 'Star':
            new_group = group.copy()
            # Choose a random index to modify
            index_to_replace = random.randint(1, len(group) - 1)

            new_state = random.choice(self.star_indices)
            while new_state in new_group:
                new_state = random.choice(self.star_indices)

            # Replace the selected index with the new integer
            new_group[index_to_replace] = new_state

        return new_group

    def expand_group(self, group):

        if hasattr(self, 'all_paths'):
            all_paths = self.all_paths
        else:
            self.generate_all_paths_tensor()
            all_paths = self.all_paths

        large_group = []
        for init_state in group:
            for final_state in group:
                large_group.append(all_paths[int(init_state),
                                             int(final_state)])

        stacked_array = np.vstack(large_group)
        flattened_array = stacked_array.flatten()
        expanded_group = np.unique(
            flattened_array[~np.isnan(flattened_array)]).astype(int)

        # print(group)
        # print(expanded_group)
        return expanded_group

    def optimise_single_group(self,
                              iterations: int = int(1e6),
                              patience: int = 100,
                              verbose: bool = False) -> tuple:
        """Perform the optimization of a single group with an early stopping
        condition if the group doesn't change for a certain number of
        iterations.

        Parameters:
        - iterations (int): The number of optimization iterations.
        - patience (int): Number of iterations to wait before quitting if no
          changes occur.

        Returns:
        - tuple: A tuple containing the final cost value and the
        optimized group.

        """
        # # Generate a random connected group
        current_group = self.random_group()

        if self.topology == 'All':
            if self.type_cost == 'WithMediators':
                current_cost = self.cost(self.expand_group(current_group))
            else:
                current_cost = self.cost(current_group)
        elif self.topology == 'Star':
            current_cost = self.cost(current_group)

        no_change_counter = 0  # Counter to track unchanged iterations

        for iteration in range(iterations):
            # Modify the current group to create a new candidate group
            new_group = self.modify_group(current_group)
            if self.topology == 'All':
                if self.type_cost == 'WithMediators':
                    new_cost = self.cost(self.expand_group(new_group))
                else:
                    new_cost = self.cost(new_group)
            elif self.topology == 'Star':
                new_cost = self.cost(new_group)

            # Accept new group if it has lower cost
            # This seems to work better than the probabilistic approach
            if new_cost < current_cost:
                current_group = new_group
                current_cost = new_cost
                no_change_counter = 0  # Reset counter if group changes
            elif new_cost >= current_cost:
                no_change_counter += 1  # Increment counter if no change

            # Check if the group hasn't changed for 'patience' iterations
            if no_change_counter >= patience:
                if verbose:
                    print(
                        f'Stopping early after {iteration + 1} iterations. ' +
                        f'No change in the last {patience} iterations.')
                else:
                    pass
                break

        group_names = []
        expanded_group_names = []

        # now we look up the path information..
        for num in np.array(current_group):
            if num >= 24:
                group_names.append([0, int(26 - num)])
            else:
                group_names.append(self.F_values_D52[23 - num])
        group_names = np.array(group_names)

        for num in np.array(self.expand_group(current_group)):
            if num >= 24:
                expanded_group_names.append([0, int(26 - num)])
            else:
                expanded_group_names.append(self.F_values_D52[23 - num])
        expanded_group_names = np.array(expanded_group_names)
        # print(group_names)
        # filter out doubled states so that only one of
        # each D52 state appears in the final list
        # group_names = np.unique(group_names, axis=0)

        # I want to include the path information for each state pair in the
        # output of the script, as a small dataframe
        df_paths = pd.read_csv('path_finder/all_paths_dataframe.csv')

        filtered_rows = []

        def find_and_append_row(df, init_state, final_state, filtered_rows):
            """Find and append the first matching row from a DataFrame
            based on initial and final states.

            This function constructs a query string to filter rows in the
            DataFrame based on the specified initial and final states. If
            a matching row is found, it appends the first result to the
            provided list.

            Parameters:
                df (pandas.DataFrame): The DataFrame to search.
                init_state (list): A list representing the initial state.
                final_state (list): A list representing the final state.
                filtered_rows (list): A list to which the found row will
                be appended.

            Returns:
                None: This function does not return a value.

            """
            # make conversion for S1/2 states that are written in the format
            # [0, #] so that everything is a tuple and arrays play nice
            if init_state[0] == 0:
                init_str = f'`Initial State` == "{init_state[1]}" '
            else:
                init_str = f'`Initial State` == "[{init_state[0]}, {init_state[1]}]" '

            if final_state[0] == 0:
                final_str = f'and `Final State` == "{final_state[1]}" '
            else:
                final_str = (f'and `Final State` == ' +
                             f'"[{final_state[0]}, {final_state[1]}]"')

            # Proper formatting of the query string
            query_str = (init_str + final_str)
            result = df.query(query_str)

            if not result.empty:
                # Append the row to the filtered rows list
                filtered_rows.append(result.iloc[0])
                return

        if self.topology == 'All':
            for init_state in group_names:
                for final_state in group_names:

                    find_and_append_row(df_paths, init_state, final_state,
                                        filtered_rows)

            # Convert the filtered rows list to a smaller DataFrame
            filtered_df = pd.DataFrame(filtered_rows)

        elif self.topology == 'Star':

            init_states = [26 - current_group[0]] * (len(current_group) - 1)
            final_states = []
            pi_times = []
            for idx, state_num in enumerate(current_group[1:]):
                # breakpoint()
                final_states.append(group_names[idx + 1])
                pi_time = self.pitimes[state_num, int(28 - current_group[0])]
                pi_times.append(pi_time)

            df_data = {
                'Initial State': init_states,
                'Final State': final_states,
                'Pi-Time': pi_times
            }

            filtered_df = pd.DataFrame(df_data)

        if verbose:
            print(f'Optimal group found: {group_names}')
            print(f'Optimal cost function value: {current_cost}')
            print(f'Paths found for states: {filtered_df}')
        else:
            pass

        return current_cost, group_names, filtered_df, expanded_group_names

    def optimise_multiple_groups(self,
                                 dimension: int = 0,
                                 init_groups: Union[float, int] = 5e2,
                                 iterations: Union[float, int] = 1e4,
                                 save_to_file: bool = False,
                                 verbose: bool = False) -> list:
        """Generate y random groups in parallel and optimize each group
        iterations times. Return the best three groups.

        Parameters:
        - init_groups (int): The number of random groups to generate.
        - iterations (int): The number of optimization iterations per group.

        Returns:
        - list: The top 3 groups based on their cost values.

        """

        if dimension != 0:
            self.dimension = dimension
        else:
            pass

        all_groups = []
        init_groups = int(init_groups)
        iterations = int(iterations)
        # Set the number of workers as half the available CPU cores
        num_workers = int(os.cpu_count() / 2)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all group optimizations to the executor
            futures = [
                executor.submit(self.optimise_single_group, iterations)
                for _ in range(init_groups)
            ]

            # Collect results as they complete
            for future in tqdm(as_completed(futures), total=init_groups):
                result = future.result()
                all_groups.append(result)

        # Sort the groups by their cost values in descending order
        all_groups.sort(key=lambda x: x[0], reverse=False)

        # Return the top 3 groups
        top_groups = all_groups[:3]
        top_group = all_groups[0]

        save_path = ('state_finder/')

        if save_to_file:
            # Save the results to a text file
            with open(
                    save_path + f'D{self.dimension}_{init_groups}_' +
                    'InitGroups_states_found_' +
                    f'AllD52_{self.all_D52}_{self.type_cost}' +
                    f'Norm_{self.topology}Topology.txt', 'w') as file:
                file.write('Total number of initial group guesses: ' +
                           f'{init_groups}. \n\n')
                for i, (cost_value, group, paths,
                        expanded_group) in enumerate(top_groups, 1):
                    file.write(f'{i} Cost = {cost_value}\n')
                    file.write(f'{i} Group Members: \n{group} \n\n')
                    file.write(f'{i} Group Paths: \n' +
                               f'{paths.to_string(index=False)} \n\n')
                    file.write('####################################' +
                               '#################################### \n\n')
        else:
            pass

        if verbose:
            # Print results if not saving to file..
            for i, (cost, group, paths,
                    expanded_group) in enumerate(top_groups, 1):
                print(f'{i} Cost = {cost}')
                print(f'{i} Group Members: \n{group} \n')
                # mediators = np.array(
                #     [tup for tup in expanded_group if tup not in group])
                # print(f'{i} Mediator Members: \n{mediators} \n')
                print(f'{i} Group Paths: \n{paths} \n\n')
                print('####################################' +
                      '#################################### \n\n')
        else:
            pass

        return top_groups

    # def find_ordered_states(self,
    #                         dimensions,
    #                         init_groups=500,
    #                         save_to_file=False,
    #                         verbose=False):
    #     """Finds and orders states based on the provided dimensions and initial
    #     groups.

    #     This method generates a two-way cost matrix and computes a list of
    #     states and mediators for each specified dimension. It evaluates the
    #     costs associated with converting states to their index representation
    #     and determines potential mediator states based on a cost-effective path.
    #     The results can be saved to a file or printed to the console for
    #     verification purposes.

    #     Parameters:

    #     - dimensions (list): A list of dimensions for which states need to be
    #       found.
    #     - init_groups (int, optional): The number of initial groups to consider
    #     for state optimization. Default is 500.
    #     - save_to_file (bool, optional): If set to True, saves the results to a
    #     specified text file. Default is False.
    #     - verbose (bool, optional): If set to True, prints detailed results to
    #     the console. Default is False.

    #     Returns:
    #     - cost (float): The total cost associated with the final state
    #       configuration.
    #     - states (list): A list of states that have been validated and ordered.
    #     - mediators (list): A list of mediator states used in the conversion
    #       process.
    #     - filtered_df (DataFrame): A DataFrame containing filtered path data
    #     based on the selected initial state.

    #     Note:
    #     The function relies on internal methods such as
    #     `generate_two_way_cost_matrix`, `convert_state_to_index`, and
    #     `optimise_multiple_groups`, and examines the paths stored in
    #     `self.all_paths` for determining valid pathways and costs.

    #     """
    #     self.generate_two_way_cost_matrix()

    #     # Define some helper functions
    #     def find_states_mediators_order(group):
    #         two_way_costs = []
    #         mediators = []
    #         num_group = []
    #         for entry in group:
    #             num_entry = self.convert_state_to_index(entry)
    #             num_group.append(num_entry)

    #         # need to iterate through possible pairs, eval cost of each
    #         # and sum them for each potential initial state
    #         for num1 in num_group:
    #             indiv_cost = 0
    #             for num2 in num_group:
    #                 if num1 != num2:
    #                     indiv_cost += self.two_way_cost_matrix[num1, num2]

    #             two_way_costs.append(indiv_cost)

    #         paired_lists = list(zip(group, num_group, two_way_costs))

    #         # Sort the paired lists by the first element of each tuple (the first list)
    #         sorted_pairs = sorted(paired_lists, key=lambda x: x[2])

    #         # Unzip the sorted pairs back into two lists
    #         sorted_group, sorted_num_group, sorted_two_way_costs = zip(
    #             *sorted_pairs)
    #         # print(sorted_group)
    #         # print(sorted_num_group)
    #         # print(sorted_two_way_costs)

    #         state_indices = []
    #         # mediators_indices = list(set(mediators))
    #         # now let's look at the path information for this sorted set of states
    #         for second_ket in sorted_num_group:
    #             zeroth_ket = sorted_num_group[0]
    #             path = self.all_paths[zeroth_ket, second_ket]
    #             clean_path = path[~np.isnan(path)]
    #             if len(clean_path) > 2:
    #                 # Append entries between the first and last elements to mediators
    #                 mediators.extend(clean_path[1:-1])

    #         # Get entries in "mediators" that are used more than once
    #         # Count occurrences of each number in the original list
    #         counts = Counter(mediators)

    #         # Create a new list of numbers that appear more than once
    #         mediators_indices = [
    #             num for num, count in counts.items() if count > 1
    #         ]
    #         mediators_used_once = [
    #             num for num, count in counts.items() if count == 1
    #         ]

    #         # if first ket is in "mediators" list, remove it
    #         # (all paths start at ket 0)
    #         if sorted_num_group[0] in mediators_indices:
    #             mediators_indices.remove(sorted_num_group[0])

    #         for idx, entry in enumerate(sorted_num_group):
    #             # need to deal with states that are only accessed via a bus state
    #             # which is only used once (because we don't want to use those states
    #             # in that case, as we'll just use the bus instead)

    #             # print(entry)
    #             # print(self.convert_index_to_state(entry))
    #             # print(self.all_paths[sorted_num_group[0]][entry])
    #             path = self.all_paths[sorted_num_group[0]][entry]
    #             clean_path = path[~np.isnan(path)]
    #             if len(clean_path) > 2:
    #                 # Append entries between the first and last elements to mediators
    #                 mediator_check = clean_path[1:-1]
    #             else:
    #                 mediator_check = []

    #             if idx == 0:
    #                 state_indices.append(entry)
    #             else:
    #                 # don't add state if it is a mediator used multiple times
    #                 if entry in mediators_indices:
    #                     pass
    #                 # don't add state if a mediator used just once is in its path
    #                 elif not set(mediator_check).isdisjoint(
    #                         set(mediators_used_once)):
    #                     pass
    #                 else:
    #                     state_indices.append(entry)

    #         states = []
    #         mediators = []
    #         for ind in state_indices:
    #             states.append(self.convert_index_to_state(ind))
    #         for ind in mediators_indices:
    #             mediators.append(self.convert_index_to_state(int(ind)))

    #         return states, mediators

    #     # now run state finder for all dimensions input and save results
    #     for dim in dimensions:
    #         # Get top group info for given dim
    #         top_groups = self.optimise_multiple_groups(dimension=dim,
    #                                                    init_groups=init_groups,
    #                                                    save_to_file=False)

    #         cost_group_path = top_groups[0]
    #         cost = cost_group_path[0]

    #         group_array = cost_group_path[1]
    #         group = []
    #         for entry in group_array:
    #             group.append(list(entry))

    #         df_path = cost_group_path[2]

    #         states, mediators = find_states_mediators_order(group)

    #         # pear down the path dataframe to only include paths starting or
    #         # ending with the zeroth ket
    #         init_ket = states[0]

    #         # Proper formatting of the query string
    #         if init_ket[0] == 0:
    #             query_str = f'`Initial State` == "{init_ket[1]}" '
    #         else:
    #             query_str = f'`Initial State` == "[{init_ket[0]}, {init_ket[1]}]" '

    #         filtered_df = df_path.query(query_str)

    #         # get full list of states in order to track which are unused
    #         all_states = list(list(tup) for tup in self.F_values_D52) + [[
    #             0, 2
    #         ], [0, 1], [0, 0], [0, -1], [0, -2]]
    #         free_states1 = [tup for tup in all_states if tup not in states]
    #         free_states = [tup for tup in free_states1 if tup not in mediators]

    #         # Save the results to a text file
    #         if save_to_file:
    #             save_path = ('state_finder/')

    #             with open(
    #                     save_path + f'D{len(states)}_Seeds{init_groups}_' +
    #                     'StateSet.txt', 'w') as file:
    #                 file.write(f'Dimension set = {self.dimension}\n')
    #                 file.write(f'Dimension found = {len(states)}\n')
    #                 file.write(f'Cost = {cost}\n')
    #                 file.write('Total number of initial group guesses = ' +
    #                            f'{int(init_groups)} \n\n')
    #                 file.write(f'Group: \n{group} \n\n')
    #                 file.write(f'States: \n{states} \n\n')
    #                 file.write(f'Mediators: \n{mediators} \n\n')
    #                 file.write(f'Free: \n{free_states} \n\n')
    #                 file.write(f'Paths: \n' +
    #                            f'{filtered_df.to_string(index=False)} \n\n')
    #                 file.write('####################################' +
    #                            '####################################')
    #         else:
    #             pass

    #         if verbose:
    #             # Print results if not saving to file..
    #             print(f'\nDimension set = {self.dimension}')
    #             print(f'Dimension found = {len(states)}')
    #             print(f'Cost = {cost}')
    #             print('Total number of initial group guesses = ' +
    #                   f'{int(init_groups)} \n')
    #             print(f'Group: \n{group} \n')
    #             print(f'States: \n{states} \n')
    #             print(f'Mediators: \n{mediators} \n')
    #             print(f'Free: \n{free_states} \n')
    #             print(f'Paths: \n{filtered_df} \n')
    #             print('####################################' +
    #                   '#################################### \n\n')
    #         else:
    #             pass

    #     return cost, states, mediators, filtered_df

    # def find_ordered_pulse_sequence(self,
    #                                 dimensions,
    #                                 init_groups=500,
    #                                 verbose=False):
    #     """
    #     Find the ordered pulse sequence based on the provided dimensions and initial groups.

    #     This method generates all possible paths in the provided state space and retrieves
    #     the corresponding states from a pre-defined file. It processes these states to
    #     calculate transition times and other relevant pulse sequence parameters, which are
    #     stored in various lists including population fractions, transitions, transition times,
    #     and transition fractions.

    #     Parameters:
    #         dimensions (list): A list of dimensions for which to find the pulse sequences.
    #         init_groups (int, optional): The number of initial groups to consider when loading
    #                                     state sets from a file. Defaults to 500.

    #     Returns:
    #         None: This method does not return any value, but it performs internal calculations
    #             and updates the related class attributes.

    #     Notes:
    #         The method uses a helper function `pulse_time` to compute the pulse duration
    #         based on the given parameters. The expected format of the state data is crucial
    #         for the regular expression search. The internal attributes such as
    #         `self.transition_pitimes` and `self.F_values_D52` must be properly initialized
    #         beforehand.
    #     """
    #     # Generate all possible paths in the state space
    #     self.generate_all_paths_tensor()
    #     # Also the two way cost matrix which will be used later for heralding pulse finding
    #     self.generate_two_way_cost_matrix()

    #     for dimension in dimensions:
    #         # Open the save file for the current dimension to read saved state data
    #         with open(
    #                 f'state_finder/D{dimension}_Seeds{init_groups}_StateSet.txt',
    #                 'r') as file:
    #             content = file.read()

    #         # Extract the states from the content of the file using a regex search
    #         states_match = re.search(r'States:\s*\[\[(.*?)\]\]', content,
    #                                  re.DOTALL)
    #         # Evaluate the matched states string to create a list of states
    #         states = eval(
    #             f'[[{states_match.group(1)}]]') if states_match else None

    #         states_num = []
    #         for state in states:
    #             # Convert the state to its corresponding index
    #             states_num.append(self.convert_state_to_index(state))

    #         def pulse_time(pitime, fraction):
    #             """
    #             Calculate the pulse duration based on the pitime and fraction.

    #             Parameters:
    #                 pitime (float): The initial pulse time.
    #                 fraction (float): The fraction value to be used in calculation.

    #             Returns:
    #                 float: The calculated pulse duration.
    #             """
    #             return (2 * pitime / np.pi) * np.arcsin(np.sqrt(fraction))

    #         # Initialize lists to store pulse sequence parameters for U1
    #         U1_pop_fractions = []
    #         U1_transitions = []
    #         U1_trans_pitimes = []
    #         U1_trans_fractions = []
    #         U1_fixed_phases = []
    #         U1_sim_phase_mask = []

    #         # Process each state pair to derive transitions and corresponding times
    #         for idx, second_state in enumerate(states_num[1:]):
    #             first_state = states_num[0]
    #             print(first_state)
    #             print(second_state)

    #             # Retrieve the path between the first and second states
    #             path = self.all_paths[first_state][second_state]
    #             clean_path = path[~np.isnan(path)]  # Remove NaN values
    #             print(clean_path)

    #             if len(clean_path) > 2:
    #                 # Loop through intermediate states only if there are
    #                 # multiple states in the path
    #                 for itera in range(len(clean_path) - 2):

    #                     first_num = clean_path[itera]
    #                     second_num = clean_path[itera + 1]

    #                     # Determine pulse time based on the state indices
    #                     if first_num > 23:
    #                         pitime = self.transition_pitimes[int(second_num)][
    #                             int(28 - first_num)]
    #                         U1_transitions.append([
    #                             int(26 - first_num),
    #                             self.F_values_D52[int(23 - second_num)][0],
    #                             self.F_values_D52[int(23 - second_num)][1]
    #                         ])
    #                         U1_trans_pitimes.append(pitime)
    #                         U1_fixed_phases.append(0)
    #                         U1_sim_phase_mask.append(0)
    #                     else:
    #                         pitime = self.transition_pitimes[int(first_num)][
    #                             int(28 - second_num)]
    #                         U1_transitions.append([
    #                             int(26 - second_num),
    #                             self.F_values_D52[int(23 - first_num)][0],
    #                             self.F_values_D52[int(23 - first_num)][1]
    #                         ])
    #                         U1_trans_pitimes.append(pitime)
    #                         U1_fixed_phases.append(1)
    #                         U1_sim_phase_mask.append(0)

    #                     # fractional pulses from initial state |0> always
    #                     if itera == 0:
    #                         U1_pop_fractions.append(1 / (dimension - idx))
    #                         U1_trans_fractions.append(
    #                             pulse_time(pitime, 1 / (dimension - idx)))

    #                     # for all subsequent mediating transitions, do pi-pulses
    #                     elif itera != 0:
    #                         U1_pop_fractions.append(1)
    #                         U1_trans_fractions.append(pulse_time(pitime, 1))

    #                 # Handle the last two states in the clean path
    #                 first_num = clean_path[-2]
    #                 second_num = clean_path[-1]
    #                 if first_num > 23:
    #                     pitime = self.transition_pitimes[int(second_num)][int(
    #                         28 - first_num)]
    #                     U1_trans_pitimes.append(pitime)
    #                     U1_transitions.append([
    #                         int(26 - first_num),
    #                         self.F_values_D52[int(23 - second_num)][0],
    #                         self.F_values_D52[int(23 - second_num)][1]
    #                     ])
    #                 else:
    #                     pitime = self.transition_pitimes[int(first_num)][int(
    #                         28 - second_num)]
    #                     U1_trans_pitimes.append(pitime)
    #                     U1_transitions.append([
    #                         int(26 - second_num),
    #                         self.F_values_D52[int(23 - first_num)][0],
    #                         self.F_values_D52[int(23 - first_num)][1]
    #                     ])

    #                 # Calculate population fraction and transition fraction for
    #                 # the current dimension
    #                 U1_pop_fractions.append(1)
    #                 U1_trans_fractions.append(pulse_time(pitime, 1))

    #             else:
    #                 # Handle case of direct transitions for U1
    #                 first_num = clean_path[-2]
    #                 second_num = clean_path[-1]
    #                 if first_num > 23:
    #                     pitime = self.transition_pitimes[int(second_num)][int(
    #                         28 - first_num)]
    #                     U1_trans_pitimes.append(pitime)
    #                     U1_transitions.append([
    #                         int(26 - first_num),
    #                         self.F_values_D52[int(23 - second_num)][0],
    #                         self.F_values_D52[int(23 - second_num)][1]
    #                     ])
    #                 else:
    #                     pitime = self.transition_pitimes[int(first_num)][int(
    #                         28 - second_num)]
    #                     U1_trans_pitimes.append(pitime)
    #                     U1_transitions.append([
    #                         int(26 - second_num),
    #                         self.F_values_D52[int(23 - first_num)][0],
    #                         self.F_values_D52[int(23 - first_num)][1]
    #                     ])

    #                 # Compute the last population fraction and transition
    #                 # fraction for the current dimension
    #                 U1_pop_fractions.append(1 / (dimension - idx))
    #                 U1_trans_fractions.append(
    #                     pulse_time(pitime, 1 / (dimension - idx)))

    #         # Initialize lists to store pulse sequence parameters for U2
    #         U2_pop_fractions = []
    #         U2_transitions = []
    #         U2_trans_pitimes = []
    #         U2_trans_fractions = []

    #         # Process each state pair to derive transitions and corresponding times
    #         for idx, second_state in enumerate(np.flip(states_num[1:])):
    #             first_state = states_num[0]
    #             print(first_state)
    #             print(second_state)

    #             # Retrieve the path between the first and second states
    #             # but now backwards
    #             path = self.all_paths[second_state][first_state]
    #             clean_path = path[~np.isnan(path)]  # Remove NaN values
    #             print(clean_path)

    #             if len(clean_path) > 2:
    #                 # Loop through intermediate states only if there are
    #                 # multiple states in the path
    #                 for itera in range(len(clean_path) - 2):
    #                     # Do full pi-pulses for mediating transitions
    #                     U2_pop_fractions.append(1)

    #                     first_num = clean_path[itera]
    #                     second_num = clean_path[itera + 1]

    #                     # Determine pulse time based on the state indices
    #                     if first_num > 23:
    #                         pitime = self.transition_pitimes[int(second_num)][
    #                             int(28 - first_num)]
    #                         U2_transitions.append([
    #                             int(26 - first_num),
    #                             self.F_values_D52[int(23 - second_num)][0],
    #                             self.F_values_D52[int(23 - second_num)][1]
    #                         ])
    #                         U2_trans_pitimes.append(pitime)
    #                         U2_trans_fractions.append(pulse_time(pitime, 1))
    #                     else:
    #                         pitime = self.transition_pitimes[int(first_num)][
    #                             int(28 - second_num)]
    #                         U2_transitions.append([
    #                             int(26 - second_num),
    #                             self.F_values_D52[int(23 - first_num)][0],
    #                             self.F_values_D52[int(23 - first_num)][1]
    #                         ])
    #                         U2_trans_pitimes.append(pitime)
    #                         U2_trans_fractions.append(pulse_time(pitime, 1))

    #                 # Handle the last transition in the path, which is fractional
    #                 first_num = clean_path[-2]
    #                 second_num = clean_path[-1]
    #                 if first_num > 23:
    #                     pitime = self.transition_pitimes[int(second_num)][int(
    #                         28 - first_num)]
    #                     U2_trans_pitimes.append(pitime)
    #                     U2_transitions.append([
    #                         int(26 - first_num),
    #                         self.F_values_D52[int(23 - second_num)][0],
    #                         self.F_values_D52[int(23 - second_num)][1]
    #                     ])
    #                 else:
    #                     pitime = self.transition_pitimes[int(first_num)][int(
    #                         28 - second_num)]
    #                     U2_trans_pitimes.append(pitime)
    #                     U2_transitions.append([
    #                         int(26 - second_num),
    #                         self.F_values_D52[int(23 - first_num)][0],
    #                         self.F_values_D52[int(23 - first_num)][1]
    #                     ])

    #                 # Calculate population fraction and transition fraction for
    #                 # the current dimension
    #                 U2_pop_fractions.append(1 / (2 + idx))
    #                 U2_trans_fractions.append(pulse_time(
    #                     pitime, 1 / (2 + idx)))

    #                 # Now need to handle shuffling population back to encoded
    #                 # state get the path from first mediator back to encoded
    #                 # state, in right order
    #                 clean_path_back = np.flip(clean_path[:-1])

    #                 for itera in range(len(clean_path_back) - 1):
    #                     # Do full pi-pulses for mediating transitions
    #                     U2_pop_fractions.append(1)

    #                     first_num = clean_path_back[itera]
    #                     second_num = clean_path_back[itera + 1]

    #                     # Determine pulse time based on the state indices
    #                     if first_num > 23:
    #                         pitime = self.transition_pitimes[int(second_num)][
    #                             int(28 - first_num)]
    #                         U2_transitions.append([
    #                             int(26 - first_num),
    #                             self.F_values_D52[int(23 - second_num)][0],
    #                             self.F_values_D52[int(23 - second_num)][1]
    #                         ])
    #                         U2_trans_pitimes.append(pitime)
    #                         U2_trans_fractions.append(pulse_time(pitime, 1))
    #                     else:
    #                         pitime = self.transition_pitimes[int(first_num)][
    #                             int(28 - second_num)]
    #                         U2_transitions.append([
    #                             int(26 - second_num),
    #                             self.F_values_D52[int(23 - first_num)][0],
    #                             self.F_values_D52[int(23 - first_num)][1]
    #                         ])
    #                         U2_trans_pitimes.append(pitime)
    #                         U2_trans_fractions.append(pulse_time(pitime, 1))

    #             else:
    #                 # Handle case of direct transitions for U2
    #                 first_num = clean_path[-2]
    #                 second_num = clean_path[-1]
    #                 if first_num > 23:
    #                     pitime = self.transition_pitimes[int(second_num)][int(
    #                         28 - first_num)]
    #                     U2_trans_pitimes.append(pitime)
    #                     U2_transitions.append([
    #                         int(26 - first_num),
    #                         self.F_values_D52[int(23 - second_num)][0],
    #                         self.F_values_D52[int(23 - second_num)][1]
    #                     ])
    #                 else:
    #                     pitime = self.transition_pitimes[int(first_num)][int(
    #                         28 - second_num)]
    #                     U2_trans_pitimes.append(pitime)
    #                     U2_transitions.append([
    #                         int(26 - second_num),
    #                         self.F_values_D52[int(23 - first_num)][0],
    #                         self.F_values_D52[int(23 - first_num)][1]
    #                     ])

    #                 # Compute the last population fraction and transition
    #                 # fraction for the current dimension
    #                 U2_pop_fractions.append(1 / (2 + idx))
    #                 U2_trans_fractions.append(pulse_time(
    #                     pitime, 1 / (2 + idx)))

    #         # rename lists to fit names in IonControl for ease of copy/paste

    #         # transition to use for heralding initialisation
    #         # this will depend on whether the |0> state is in S12 or D52

    #         if states[0][0] == 0:
    #             # need to find the next best state to shelve to that isn't in
    #             # the list of states encoded
    #             next_best = np.inf
    #             first_state = states_num[0]
    #             state_for_herald_num = states_num[0]
    #             for second_state in range(29):
    #                 if second_state not in states_num:
    #                     two_way_cost = self.two_way_cost_matrix[first_state][
    #                         second_state]
    #                     if two_way_cost < next_best:
    #                         next_best = two_way_cost
    #                         state_for_herald_num = second_state
    #                     else:
    #                         pass
    #                 else:
    #                     pass
    #             state_for_herald = self.convert_index_to_state(
    #                 state_for_herald_num)
    #             initial_state = [
    #                 states[0][0], state_for_herald[0], state_for_herald[1]
    #             ]

    #         else:
    #             # this uses the S1/2 state from this transition
    #             initial_state = U1_transitions[0]

    #         # # U1 part of Ramsey pulse train, fractions, and phases
    #         pulse_train_U1 = U1_transitions
    #         fractions_U1 = U1_pop_fractions
    #         simulated_phase_mask_U1 = U1_sim_phase_mask
    #         fixed_phase_mask_U1 = U1_fixed_phases

    #         # # U2 part of Ramsey pulse train, fractions, and phases
    #         # pulse_train_U2 = U2_transitions
    #         # fractions_U2 = U2_pop_fractions
    #         # simulated_phase_mask_U2 = U2_sim_phase_mask
    #         # fixed_phase_mask_U2 = U2_fixed_phases

    #         # transitions used for readout of encoded states
    #         probe_trans = []
    #         for state in states:
    #             if state[0] == 0:
    #                 pass
    #             else:
    #                 for triplet in U1_transitions:
    #                     if triplet[1:] == state:
    #                         probe_trans.append(triplet)
    #                     else:
    #                         pass

    #         # TODO add probe_trans generation for readout pulses, and initial_state
    #         # CONFIRMED - initial state IS the transition used to herald
    #         # initialisation! Do something about this!

    #         if verbose:
    #             # Print results if not saving to file..
    #             print(f'\ninitial_state = ', initial_state, '\n')

    #             print(f'pulse_train_U1 = ', pulse_train_U1)
    #             print(f'fractions_U1 = ', fractions_U1)
    #             print(f'simulated_phase_mask_U1 = ', simulated_phase_mask_U1)
    #             print(f'fixed_phase_mask_U1 = ', fixed_phase_mask_U1)

    #             # print(f'pulse_train_U2 = ', pulse_train_U2)
    #             # print(f'fractions_U2 = ', fractions_U2)
    #             # print(f'simulated_phase_mask_U2 = ', simulated_phase_mask_U2)
    #             # print(f'fixed_phase_mask_U2 = ', fixed_phase_mask_U2)

    #             print(f'\nprobe_trans = ', probe_trans, '\n')

    #             print('####################################' +
    #                   '#################################### \n\n')
    #         else:
    #             pass

    #         breakpoint()

    #     return None

    def scaling_plots(self,
                      df_path_name: str = 'path_finder/all_paths_dataframe.csv'
                      ):
        """Generate and plot various metrics related to coherence and
        effective Pi-times for a optimal state sets found using StateFinder.

        This method performs the following tasks:
        1. Reads a path information dataframe from a CSV file.
        2. Initializes lists to store D values, cost values, effective
        Pi-times, and coherence values.
        3. For each dimension (D) from 2 to 24:
            - Reads a corresponding text file containing group state data.
            - Extracts the cost value and appends it to a list.
            - Identifies and collects group member states from the file.
            - Computes effective Pi-times and coherence values for each pair
            of states in the group.
            - Appends the computed average, minimum, and maximum coherence
            values to respective lists.
        4. Creates three plots:
            - A plot of cost values against D values.
            - A plot of average effective Pi-times against D values.
            - A plot of best, average, and worst case coherence values against
            D values, all displayed on a logarithmic y-axis.

        Returns:
            None: This method does not return a value but produces visual
            plots for analysis.

        """
        # read in path information dataframe
        df_paths = pd.read_csv(df_path_name)
        # print(df_paths)

        # generate all_coherences matrix
        self.generate_all_coherences()
        # Initialize lists for D values and cost values
        D_values = []
        cost_values = []
        eff_pitime_values = []
        avg_coherence_values = []
        worst_coherence_values = []
        best_coherence_values = []

        # define a function to get the list of effective pitimes for
        # every pair of tuples in the optimal set of states
        def find_and_append_pitimes(df, init_state, final_state,
                                    filtered_eff_pitimes):
            """
            Find and append effective Pi-time from a DataFrame based on
            initial and final states.

            Constructs a query string to filter rows in the DataFrame.
            If a matching row is found, appends the effective Pi-time
            to the provided list.

            Parameters:
                df (pandas.DataFrame): The DataFrame to search.
                init_state (list): A list representing the initial state.
                final_state (list): A list representing the final state.
                filtered_eff_pitimes (list): A list to which the effective
                Pi-time will be appended.

            Returns:
                None: This function does not return a value.
            """

            # Proper formatting of the query string
            query_str = (
                f'`Initial State` == "[{init_state[0]}, {init_state[1]}]" ' +
                'and `Final State` == ' +
                f'"[{final_state[0]}, {final_state[1]}]"')
            result = df.query(query_str)

            if not result.empty:
                # Append the row to the filtered rows list
                filtered_eff_pitimes.append(result['Effective Pi-Time'])
                return

        # Loop through files for D2 to D24
        for D in range(2, 25):  # Adjust the range to cover D2 to D24
            file_name = (f'state_finder/D{D}_10000InitGroups_' +
                         'states_found_AllD52_True.txt')

            # Check if the file exists before processing
            if os.path.exists(file_name):
                # print(f"Processing {file_name}")

                # Step 2: Read the text file
                with open(file_name, 'r') as file:
                    lines = file.readlines()

                # Step 3: Extract the cost value from line 3
                cost_line = lines[2].strip()
                cost_value = float(cost_line.split('=')[1].strip())

                # Append D value and cost value to respective lists
                D_values.append(D)
                cost_values.append(cost_value)

                # Step 4: Find the line with "Group Members"
                # and extract relevant lines
                group_members_lines = []
                for i, line in enumerate(lines):
                    if 'Group Members' in line:
                        group_members_lines = lines[i + 1:i + 1 + D]
                        break

                # Step 5: Convert the extracted lines into a numpy array
                group_members = []
                for line in group_members_lines:
                    group_members.append([
                        int(x) for x in line.strip().replace('[', '').replace(
                            ']', '').split()
                    ])

                group_members_np = np.array(group_members)

            else:
                print(f"File {file_name} not found")

            filtered_eff_pitimes = []
            filtered_eff_coherences = []

            for init_state in group_members_np:
                for final_state in group_members_np:

                    find_and_append_pitimes(df_paths, init_state, final_state,
                                            filtered_eff_pitimes)

                    helper = [1, 5, 11, 19]
                    idx1 = int(helper[int(init_state[0] - 1)] - init_state[1])
                    idx2 = int(28 - (helper[int(final_state[0] - 1)] -
                                     final_state[1]))

                    # print((self.all_coherences[0]))
                    coherence_val = self.all_coherences[idx1, idx2]

                    filtered_eff_coherences.append(coherence_val)

            eff_pitime_values.append(np.mean(np.array(filtered_eff_pitimes)))
            avg_coherence_values.append(
                np.nanmean(np.array(filtered_eff_coherences)))
            worst_coherence_values.append(
                np.nanmin(np.array(filtered_eff_coherences)))
            best_coherence_values.append(
                np.nanmax(np.array(filtered_eff_coherences)))

        # Create plots
        plt.figure(figsize=(8, 6))
        plt.plot(D_values, cost_values, marker='o', linestyle='-')
        plt.xlabel('Dimension')
        plt.ylabel('Cost Value')
        plt.title('Optimal Cost Function Value vs Dimension')
        plt.grid(True)

        plt.figure(figsize=(8, 6))
        plt.plot(D_values, eff_pitime_values, marker='o', linestyle='-')
        plt.xlabel('Dimension')
        plt.ylabel(r'Effective $\tau_\pi$ Value / $\mu s$')
        plt.title(r'Average Pairwise Effective $\tau_\pi$ Value vs Dimension')
        plt.grid(True)

        # Create a figure sharing an x-axis for coherence times plots
        plt.figure(figsize=(8, 6))

        plt.plot(D_values,
                 best_coherence_values,
                 marker='o',
                 linestyle='-',
                 label=r'Best Case $T_2^*$')
        plt.plot(D_values,
                 avg_coherence_values,
                 marker='o',
                 linestyle='-',
                 label=r'Average $T_2^*$')
        plt.plot(D_values,
                 worst_coherence_values,
                 marker='o',
                 linestyle='-',
                 label=r'Worst Case $T_2^*$')

        # Set labels and title
        plt.xlabel('Dimension')
        plt.ylabel(r'$T_2^*$ Value / $ms$')
        plt.title(r'$T_2^*$ Value vs Dimension')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        return None


if __name__ == '__main__':

    start_time = time.time()
    sf = StateFinder(dimension=2,
                     all_D52=False,
                     type_cost='WithMediators',
                     topology='Star')

    # TROUBLESHOOTING
    # sf.generate_coupling_graph(plot_graph=True)
    # sf.generate_all_pitimes(verbose=True)
    # sf.generate_all_sensitivities(verbose=True)
    # sf.generate_all_coherences(verbose=True)
    # sf.generate_all_paths_tensor()
    # # sf.generate_two_way_cost_matrix(verbose=True)
    # sf.generate_cost_matrix(recompute_paths=False, verbose=True)

    # group = sf.random_group(print_initial_group=True)
    # print(group)
    # sf.cost(group)
    # sf.optimise_single_group(patience=10, verbose=True)

    # for i in range(10):
    #     sf.random_group(print_initial_group=True)
    # sf.optimise_single_group(verbose=True)

    # sf.optimise_multiple_groups(init_groups=100,
    #                             iterations=1e6,
    #                             save_to_file=False,
    #                             verbose=True)
    ###############################

    # # MAIN WORKHORSE OF THIS SCRIPT
    dimensions = np.arange(2, 19, 1)
    # dimensions = [3]
    for dim in dimensions:
        sf = StateFinder(dimension=dim,
                         all_D52=False,
                         type_cost='WithMediators',
                         topology='All')

        sf.optimise_multiple_groups(dimension=dim,
                                    init_groups=100,
                                    iterations=1e6,
                                    save_to_file=True,
                                    verbose=True)

        sf.all_D52 = True
        sf.optimise_multiple_groups(dimension=dim,
                                    init_groups=100,
                                    iterations=1e6,
                                    save_to_file=True,
                                    verbose=True)
        # breakpoint()

    # for dim in dimensions:
    #     sf = StateFinder(dimension=dim,
    #                      all_D52=False,
    #                      type_cost='Euclidean')
    #     sf.optimise_multiple_groups(init_groups=100,
    #                                 iterations=1e6,
    #                                 save_to_file=False,
    #                                 verbose=True)

    # for dim in dimensions:
    #     sf = StateFinder(dimension=dim,
    #                      all_D52=False,
    #                      type_cost='Manhattan')
    #     sf.optimise_multiple_groups(init_groups=100,
    #                                 iterations=1e6,
    #                                 save_to_file=False,
    #                                 verbose=True)
    ###############################

    # sf.scaling_plots()

    ### Code to find groups according to (outdated) Ramsey cost function
    # sf.optimise_ramsey_groups(save_to_file=True, verbose=True)

    # # Finding Ramsey groups- optimising for best initial state
    # dimensions = np.arange(2, 25, 1)
    # for dim in dimensions:
    #     sf = StateFinder(dimension=dim, all_D52=False)
    #     sf.optimise_ramsey_groups(save_to_file=True)
    ###############################

    print('--- Total time: %s seconds ---' % (time.time() - start_time))
    plt.show()
