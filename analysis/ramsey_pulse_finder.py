import ast
import itertools
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ramsey_star_pulse_finder import RamseyStarPulseFinder
from state_finder import StateFinder

# from plot_utils import nice_fonts
# mpl.rcParams.update(nice_fonts)
# np.set_printoptions(suppress=True)


class RamseyPulseFinder(StateFinder):

    def __init__(self):
        super().__init__()

    def find_ordered_states(self,
                            dimensions,
                            init_groups=500,
                            save_to_file=False,
                            verbose=False):
        """Finds and orders states based on the provided dimensions and initial
        groups.

        This method generates a two-way cost matrix and computes a list of
        states and mediators for each specified dimension. It evaluates the
        costs associated with converting states to their index representation
        and determines potential mediator states based on a cost-effective path.
        The results can be saved to a file or printed to the console for
        verification purposes.

        Parameters:

        - dimensions (list): A list of dimensions for which states need to be
          found.
        - init_groups (int, optional): The number of initial groups to consider
        for state optimization. Default is 500.
        - save_to_file (bool, optional): If set to True, saves the results to a
        specified text file. Default is False.
        - verbose (bool, optional): If set to True, prints detailed results to
        the console. Default is False.

        Returns:
        - cost (float): The total cost associated with the final state
          configuration.
        - states (list): A list of states that have been validated and ordered.
        - mediators (list): A list of mediator states used in the transition
          process.
        - filtered_df (DataFrame): A DataFrame containing filtered path data
        based on the selected initial state.

        Note:
        The function relies on internal methods such as
        `generate_two_way_cost_matrix`, `convert_state_to_index`, and
        `optimise_multiple_groups`, and examines the paths stored in
        `self.all_paths` for determining valid pathways and costs.

        """
        self.generate_two_way_cost_matrix()
        self.generate_all_pitimes()
        self.generate_all_coherences()

        # Define some helper functions
        def find_states_mediators_order(group):
            two_way_costs = []
            mediators = []
            num_group = []
            for entry in group:
                num_entry = self.convert_state_to_index(entry)
                num_group.append(num_entry)

            # first approach uses cost function that we used to find states
            # need to iterate through possible pairs, eval cost of each
            # and sum them for each potential initial state
            for num1 in num_group:
                indiv_cost = 0
                for num2 in num_group:
                    if num1 != num2:
                        indiv_cost += self.two_way_cost_matrix[num1, num2]

                two_way_costs.append(indiv_cost)

            paired_lists = list(zip(group, num_group, two_way_costs))

            # Sort the paired lists by the first element of each tuple (the first list)
            sorted_pairs = sorted(paired_lists, key=lambda x: x[2])

            # Unzip the sorted pairs back into two lists
            sorted_group, sorted_num_group, sorted_two_way_costs = zip(
                *sorted_pairs)

            #####################################
            # may be best to find "two way cost" another way by finding the
            # estimated coherence between |0> and each |n>, then finding
            # pi-times between the same, and putting shortest pi-times last in
            # order, while putting longest coherence first

            two_way_pitimes = []
            two_way_coherences = []
            for num1 in sorted_num_group[1:]:
                two_way_pitimes.append(self.all_pitimes[sorted_num_group[0],
                                                        num1])
                two_way_coherences.append(
                    self.all_coherences[sorted_num_group[0], num1])

            two_way_pitimes = np.array(two_way_pitimes)
            two_way_coherences = np.array(two_way_coherences)

            # pick which type of ordering you want
            # for now we will be sorting pitimes from high to low
            # so fastest transitions occur when most superposition exists
            sorter = np.flip(np.argsort(
                two_way_pitimes))  # based on pitimes high to low only

            # sorter = np.argsort(
            #     two_way_coherences)  # based on coherences low to high only

            # based on both
            # sorter = np.argsort(
            #     np.argsort(two_way_pitimes[::-1]) +
            #     np.argsort(two_way_coherences))

            sorted_num_group = np.insert(
                np.array(sorted_num_group[1:])[sorter], 0, sorted_num_group[0])
            #####################################

            state_indices = []
            # mediators_indices = list(set(mediators))
            # now let's look at the path information for this sorted set of states
            for second_ket in sorted_num_group:
                zeroth_ket = sorted_num_group[0]
                path = self.all_paths[zeroth_ket, second_ket]
                clean_path = path[~np.isnan(path)]
                if len(clean_path) > 2:
                    # Append entries between the first and last elements to mediators
                    mediators.extend(clean_path[1:-1])

            # Get entries in "mediators" that are used more than once
            # Count occurrences of each number in the original list
            counts = Counter(mediators)

            # Create a new list of numbers that appear more than once
            mediators_indices = [
                int(num) for num, count in counts.items() if count > 1
            ]
            mediators_used_once = [
                int(num) for num, count in counts.items() if count == 1
            ]

            # if first ket is in "mediators" list, remove it
            # (all paths start at ket 0)
            if int(sorted_num_group[0]) in mediators_indices:
                mediators_indices.remove(int(sorted_num_group[0]))

            for idx, entry in enumerate(sorted_num_group):
                # need to deal with states that are only accessed via a bus state
                # which is only used once (because we don't want to use those states
                # in that case, as we'll just use the bus instead)

                # print(entry)
                # print(self.convert_index_to_state(entry))
                # print(self.all_paths[sorted_num_group[0]][entry])
                path = self.all_paths[sorted_num_group[0]][entry]
                clean_path = path[~np.isnan(path)]
                if len(clean_path) > 2:
                    # Append entries between the first and last elements to mediators
                    mediator_check = clean_path[1:-1]
                else:
                    mediator_check = []

                if idx == 0:
                    state_indices.append(entry)
                else:
                    # don't add state if it is a mediator used multiple times
                    if entry in mediators_indices:
                        pass
                    # don't add state if a mediator used just once is in its path
                    elif not set(mediator_check).isdisjoint(
                            set(mediators_used_once)):
                        pass
                    else:
                        state_indices.append(entry)

            new_cost = self.cost(state_indices + mediators_indices)

            states = []
            mediators = []
            for ind in state_indices:
                states.append(self.convert_index_to_state(ind))
            for ind in mediators_indices:
                mediators.append(self.convert_index_to_state(ind))

            return states, mediators, new_cost

        # now run state finder for all dimensions input and save results
        for dim in dimensions:
            # Get top group info for given dim
            top_groups = self.optimise_multiple_groups(dimension=dim,
                                                       init_groups=init_groups,
                                                       save_to_file=False)

            cost_group_path = top_groups[0]
            cost = cost_group_path[0]

            group_array = cost_group_path[1]
            group = []
            for entry in group_array:
                group.append(list(entry))

            df_path = cost_group_path[2]

            states, mediators, new_cost = find_states_mediators_order(group)

            # pear down the path dataframe to only include paths starting or
            # ending with the zeroth ket
            init_ket = states[0]

            # Proper formatting of the query string
            if init_ket[0] == 0:
                query_str = f'`Initial State` == "{init_ket[1]}" '
            else:
                query_str = f'`Initial State` == "[{init_ket[0]}, {init_ket[1]}]" '

            filtered_df = df_path.query(query_str)

            # get full list of states in order to track which are unused
            all_states = list(list(tup) for tup in self.F_values_D52) + [[
                0, 2
            ], [0, 1], [0, 0], [0, -1], [0, -2]]
            free_states1 = [tup for tup in all_states if tup not in states]
            free_states = [tup for tup in free_states1 if tup not in mediators]

            # Save the results to a text file
            if save_to_file:
                save_path = (
                    '/home/nicholas/Documents/' +
                    'Barium137_Qudit/analysis/ramseys_pulse_sequences/')

                if os.path.exists(save_path +
                                  f'D{len(states)}_Seeds{init_groups}_' +
                                  'StateSet.txt'):

                    with open(
                            save_path + f'D{len(states)}_Seeds{init_groups}_' +
                            'StateSet.txt', 'r') as file:

                        for line in file:
                            if "Updated Cost" in line:
                                updated_cost = float(
                                    line.split('=')[1].strip())
                                break

                    # only update the file for states if we found a better cost
                    if updated_cost > new_cost:
                        with open(
                                save_path +
                                f'D{len(states)}_Seeds{init_groups}_' +
                                'StateSet.txt', 'w') as file:
                            file.write(f'Dimension set = {self.dimension}\n')
                            file.write(f'Dimension found = {len(states)}\n')
                            file.write(f'Original Cost = {cost}\n')
                            file.write(f'Updated Cost = {new_cost}\n')
                            file.write(
                                'Total number of initial group guesses = ' +
                                f'{int(init_groups)} \n\n')
                            file.write(f'Group: \n{group} \n\n')
                            file.write(f'States: \n{states} \n\n')
                            file.write(f'Mediators: \n{mediators} \n\n')
                            file.write(f'Free: \n{free_states} \n\n')
                            file.write(
                                f'Paths: \n' +
                                f'{filtered_df.to_string(index=False)} \n\n')
                            file.write('####################################' +
                                       '####################################')
                    else:
                        pass
                else:
                    with open(
                            save_path + f'D{len(states)}_Seeds{init_groups}_' +
                            'StateSet.txt', 'w') as file:
                        file.write(f'Dimension set = {self.dimension}\n')
                        file.write(f'Dimension found = {len(states)}\n')
                        file.write(f'Original Cost = {cost}\n')
                        file.write(f'Updated Cost = {new_cost}\n')
                        file.write('Total number of initial group guesses = ' +
                                   f'{int(init_groups)} \n\n')
                        file.write(f'Group: \n{group} \n\n')
                        file.write(f'States: \n{states} \n\n')
                        file.write(f'Mediators: \n{mediators} \n\n')
                        file.write(f'Free: \n{free_states} \n\n')
                        file.write(
                            f'Paths: \n' +
                            f'{filtered_df.to_string(index=False)} \n\n')
                        file.write('####################################' +
                                   '####################################')
            else:
                pass

            if verbose:
                # Print results if not saving to file..
                print(f'\nDimension set = {self.dimension}')
                print(f'Dimension found = {len(states)}')
                print(f'Original Cost = {cost}')
                print(f'Updated Cost = {new_cost}')
                print('Total number of initial group guesses = ' +
                      f'{int(init_groups)} \n')
                print(f'Group: \n{group} \n')
                print(f'States: \n{states} \n')
                print(f'Mediators: \n{mediators} \n')
                print(f'Free: \n{free_states} \n')
                print(f'Paths: \n{filtered_df} \n')
                print('####################################' +
                      '#################################### \n\n')
            else:
                pass

        return cost, states, mediators, filtered_df

    def find_ordered_pulse_sequence(self,
                                    dimensions,
                                    init_groups=500,
                                    verbose=False,
                                    save_to_file=False):
        """
        Find the ordered pulse sequence based on the provided dimensions and initial groups.

        This method generates all possible paths in the provided state space and retrieves
        the corresponding states from a pre-defined file. It processes these states to
        calculate transition times and other relevant pulse sequence parameters, which are
        stored in various lists including population fractions, transitions, transition times,
        and transition fractions.

        Parameters:
            dimensions (list): A list of dimensions for which to find the pulse sequences.
            init_groups (int, optional): The number of initial groups to consider when loading
                                        state sets from a file. Defaults to 500.

        Returns:
            None: This method does not return any value, but it performs internal calculations
                and updates the related class attributes.

        Notes:
            The method uses a helper function `pulse_time` to compute the pulse duration
            based on the given parameters. The expected format of the state data is crucial
            for the regular expression search. The internal attributes such as
            `self.transition_pitimes` and `self.F_values_D52` must be properly initialized
            beforehand.
        """
        # Generate all possible paths in the state space
        self.generate_all_paths_tensor()
        # Also the two way cost matrix which will be used later for heralding pulse finding
        self.generate_two_way_cost_matrix()

        for dimension in dimensions:
            # Open the save file for the current dimension to read saved state data
            with open(
                    f'ramseys_pulse_sequences/D{dimension}_Seeds{init_groups}_StateSet.txt',
                    'r') as file:
                content = file.read()

            # Extract the states from the content of the file using a regex search
            states_match = re.search(r'States:\s*\[\[(.*?)\]\]', content,
                                     re.DOTALL)
            # Evaluate the matched states string to create a list of states
            states = eval(
                f'[[{states_match.group(1)}]]') if states_match else None

            # Extract the states from the content of the file using a regex search
            mediators_match = re.search(r'Mediators:\s*\[\[(.*?)\]\]', content,
                                        re.DOTALL)
            # Evaluate the matched states string to create a list of states
            mediators = eval(
                f'[[{mediators_match.group(1)}]]') if mediators_match else None

            states_num = []
            for state in states:
                # Convert the state to its corresponding index
                states_num.append(self.convert_state_to_index(state))

            def pulse_time(pitime, fraction):
                """
                Calculate the pulse duration based on the pitime and fraction.

                Parameters:
                    pitime (float): The initial pulse time.
                    fraction (float): The fraction value to be used in calculation.

                Returns:
                    float: The calculated pulse duration.
                """
                return (2 * pitime / np.pi) * np.arcsin(np.sqrt(fraction))

            # Initialize lists to store pulse sequence parameters for U1
            U1_pop_fractions = []
            U1_transitions = []
            U1_trans_pitimes = []
            U1_trans_fractions = []
            U1_fixed_phases = []
            U1_sim_phase_mask = []

            # Process each state pair to derive transitions and corresponding times
            for idx, second_state in enumerate(states_num[1:]):
                first_state = states_num[0]
                # print(first_state)
                # print(second_state)

                # Retrieve the path between the first and second states
                path = self.all_paths[first_state][second_state]
                clean_path = path[~np.isnan(path)]  # Remove NaN values
                # print(clean_path)

                if len(clean_path) > 2:
                    # Loop through intermediate states only if there are
                    # multiple states in the path
                    for itera in range(len(clean_path) - 2):

                        first_num = clean_path[itera]
                        second_num = clean_path[itera + 1]

                        # Determine pulse time based on the state indices
                        if first_num > 23:
                            pitime = self.transition_pitimes[int(second_num)][
                                int(28 - first_num)]
                            U1_transitions.append([
                                int(26 - first_num),
                                self.F_values_D52[int(23 - second_num)][0],
                                self.F_values_D52[int(23 - second_num)][1]
                            ])
                            U1_trans_pitimes.append(pitime)
                            U1_fixed_phases.append(0)
                            U1_sim_phase_mask.append(0)
                        else:
                            pitime = self.transition_pitimes[int(first_num)][
                                int(28 - second_num)]
                            U1_transitions.append([
                                int(26 - second_num),
                                self.F_values_D52[int(23 - first_num)][0],
                                self.F_values_D52[int(23 - first_num)][1]
                            ])
                            U1_trans_pitimes.append(pitime)
                            U1_fixed_phases.append(1)
                            U1_sim_phase_mask.append(0)

                        # fractional pulses from initial state |0> always
                        if itera == 0:
                            U1_pop_fractions.append(1 / (dimension - idx))
                            U1_trans_fractions.append(
                                pulse_time(pitime, 1 / (dimension - idx)))

                        # for all subsequent mediating transitions, do pi-pulses
                        elif itera != 0:
                            U1_pop_fractions.append(1)
                            U1_trans_fractions.append(pulse_time(pitime, 1))

                    # Handle the last transition in the clean path
                    first_num = clean_path[-2]
                    second_num = clean_path[-1]
                    if first_num > 23:
                        pitime = self.transition_pitimes[int(second_num)][int(
                            28 - first_num)]
                        U1_trans_pitimes.append(pitime)
                        U1_transitions.append([
                            int(26 - first_num),
                            self.F_values_D52[int(23 - second_num)][0],
                            self.F_values_D52[int(23 - second_num)][1]
                        ])
                        U1_fixed_phases.append(0)
                        U1_sim_phase_mask.append(0)
                    else:
                        pitime = self.transition_pitimes[int(first_num)][int(
                            28 - second_num)]
                        U1_trans_pitimes.append(pitime)
                        U1_transitions.append([
                            int(26 - second_num),
                            self.F_values_D52[int(23 - first_num)][0],
                            self.F_values_D52[int(23 - first_num)][1]
                        ])
                        U1_fixed_phases.append(1)
                        U1_sim_phase_mask.append(0)

                    # Calculate population fraction and transition fraction for
                    # the current dimension
                    U1_pop_fractions.append(1)
                    U1_trans_fractions.append(pulse_time(pitime, 1))

                else:
                    # Handle case of direct transitions for U1
                    first_num = clean_path[-2]
                    second_num = clean_path[-1]
                    if first_num > 23:
                        pitime = self.transition_pitimes[int(second_num)][int(
                            28 - first_num)]
                        U1_trans_pitimes.append(pitime)
                        U1_transitions.append([
                            int(26 - first_num),
                            self.F_values_D52[int(23 - second_num)][0],
                            self.F_values_D52[int(23 - second_num)][1]
                        ])
                        U1_fixed_phases.append(0)
                        U1_sim_phase_mask.append(0)

                    else:
                        pitime = self.transition_pitimes[int(first_num)][int(
                            28 - second_num)]
                        U1_trans_pitimes.append(pitime)
                        U1_transitions.append([
                            int(26 - second_num),
                            self.F_values_D52[int(23 - first_num)][0],
                            self.F_values_D52[int(23 - first_num)][1]
                        ])
                        U1_fixed_phases.append(1)
                        U1_sim_phase_mask.append(0)

                    # Compute the last population fraction and transition
                    # fraction for the current dimension
                    U1_pop_fractions.append(1 / (dimension - idx))
                    U1_trans_fractions.append(
                        pulse_time(pitime, 1 / (dimension - idx)))

            # Initialize lists to store pulse sequence parameters for U2
            U2_pop_fractions = []
            U2_transitions = []
            U2_trans_pitimes = []
            U2_trans_fractions = []
            U2_fixed_phases = []
            U2_sim_phase_mask = []

            # Process each state pair to derive transitions and corresponding times
            for idx, second_state in enumerate(np.flip(states_num[1:])):
                first_state = states_num[0]
                # print(first_state)
                # print(second_state)

                # Retrieve the path between the first and second states
                # but now backwards
                path = self.all_paths[second_state][first_state]
                clean_path = path[~np.isnan(path)]  # Remove NaN values
                # print(clean_path)

                if len(clean_path) > 2:
                    # Loop through intermediate states only if there are
                    # multiple states in the path
                    for itera in range(len(clean_path) - 2):
                        # Do full pi-pulses for mediating transitions
                        U2_pop_fractions.append(1)

                        first_num = clean_path[itera]
                        second_num = clean_path[itera + 1]

                        # Determine pulse time based on the state indices
                        if first_num > 23:
                            pitime = self.transition_pitimes[int(second_num)][
                                int(28 - first_num)]
                            U2_transitions.append([
                                int(26 - first_num),
                                self.F_values_D52[int(23 - second_num)][0],
                                self.F_values_D52[int(23 - second_num)][1]
                            ])
                            U2_trans_pitimes.append(pitime)
                            U2_trans_fractions.append(pulse_time(pitime, 1))
                            U2_fixed_phases.append(0)
                            U2_sim_phase_mask.append(0)
                        else:
                            pitime = self.transition_pitimes[int(first_num)][
                                int(28 - second_num)]
                            U2_transitions.append([
                                int(26 - second_num),
                                self.F_values_D52[int(23 - first_num)][0],
                                self.F_values_D52[int(23 - first_num)][1]
                            ])
                            U2_trans_pitimes.append(pitime)
                            U2_trans_fractions.append(pulse_time(pitime, 1))
                            U2_fixed_phases.append(1)
                            U2_sim_phase_mask.append(0)

                    # Handle the last transition in the path, which is fractional
                    # It also has the variable phase for simulated_phase_mask
                    first_num = clean_path[-2]
                    second_num = clean_path[-1]
                    if first_num > 23:
                        pitime = self.transition_pitimes[int(second_num)][int(
                            28 - first_num)]
                        U2_trans_pitimes.append(pitime)
                        U2_transitions.append([
                            int(26 - first_num),
                            self.F_values_D52[int(23 - second_num)][0],
                            self.F_values_D52[int(23 - second_num)][1]
                        ])
                        U2_fixed_phases.append(0)
                        U2_sim_phase_mask.append(int(dimension - idx - 1))
                    else:
                        pitime = self.transition_pitimes[int(first_num)][int(
                            28 - second_num)]
                        U2_trans_pitimes.append(pitime)
                        U2_transitions.append([
                            int(26 - second_num),
                            self.F_values_D52[int(23 - first_num)][0],
                            self.F_values_D52[int(23 - first_num)][1]
                        ])
                        U2_fixed_phases.append(1)
                        U2_sim_phase_mask.append(int(dimension - idx - 1))

                    # Calculate population fraction and transition fraction for
                    # the current dimension
                    U2_pop_fractions.append(1 / (2 + idx))
                    U2_trans_fractions.append(pulse_time(
                        pitime, 1 / (2 + idx)))

                    # Now need to handle shuffling population back to encoded
                    # state get the path from first mediator back to encoded
                    # state, in right order
                    clean_path_back = np.flip(clean_path[:-1])

                    for itera in range(len(clean_path_back) - 1):
                        # Do full pi-pulses for mediating transitions
                        U2_pop_fractions.append(1)

                        first_num = clean_path_back[itera]
                        second_num = clean_path_back[itera + 1]

                        # Determine pulse time based on the state indices
                        if first_num > 23:
                            pitime = self.transition_pitimes[int(second_num)][
                                int(28 - first_num)]
                            U2_transitions.append([
                                int(26 - first_num),
                                self.F_values_D52[int(23 - second_num)][0],
                                self.F_values_D52[int(23 - second_num)][1]
                            ])
                            U2_trans_pitimes.append(pitime)
                            U2_trans_fractions.append(pulse_time(pitime, 1))
                            U2_fixed_phases.append(0)
                            U2_sim_phase_mask.append(0)

                        else:
                            pitime = self.transition_pitimes[int(first_num)][
                                int(28 - second_num)]
                            U2_transitions.append([
                                int(26 - second_num),
                                self.F_values_D52[int(23 - first_num)][0],
                                self.F_values_D52[int(23 - first_num)][1]
                            ])
                            U2_trans_pitimes.append(pitime)
                            U2_trans_fractions.append(pulse_time(pitime, 1))
                            U2_fixed_phases.append(1)
                            U2_sim_phase_mask.append(0)

                else:
                    # Handle case of direct transitions for U2
                    first_num = clean_path[-2]
                    second_num = clean_path[-1]
                    if first_num > 23:
                        pitime = self.transition_pitimes[int(second_num)][int(
                            28 - first_num)]
                        U2_trans_pitimes.append(pitime)
                        U2_transitions.append([
                            int(26 - first_num),
                            self.F_values_D52[int(23 - second_num)][0],
                            self.F_values_D52[int(23 - second_num)][1]
                        ])
                        U2_fixed_phases.append(0)
                        U2_sim_phase_mask.append(dimension - idx - 1)
                    else:
                        pitime = self.transition_pitimes[int(first_num)][int(
                            28 - second_num)]
                        U2_trans_pitimes.append(pitime)
                        U2_transitions.append([
                            int(26 - second_num),
                            self.F_values_D52[int(23 - first_num)][0],
                            self.F_values_D52[int(23 - first_num)][1]
                        ])
                        U2_fixed_phases.append(1)
                        U2_sim_phase_mask.append(dimension - idx - 1)

                    # Compute the last population fraction and transition
                    # fraction for the current dimension
                    U2_pop_fractions.append(1 / (2 + idx))
                    U2_trans_fractions.append(pulse_time(
                        pitime, 1 / (2 + idx)))

            # transition to use for heralding initialisation
            # this will depend on whether the |0> state is in S12 or D52
            if states[0][0] == 0:
                # need to find the next best state to shelve to that isn't in
                # the list of states encoded
                next_best = np.inf
                first_state = states_num[0]
                for second_state in range(29):
                    if second_state not in states_num:
                        two_way_cost = self.two_way_cost_matrix[first_state][
                            second_state]
                        if two_way_cost < next_best:
                            next_best = two_way_cost
                            state_for_herald_num = second_state
                        else:
                            pass
                    else:
                        pass
                state_for_herald = self.convert_index_to_state(
                    state_for_herald_num)
                initial_state = [
                    states[0][1], state_for_herald[0], state_for_herald[1]
                ]

            else:
                # this uses the S1/2 state from this transition
                initial_state = U1_transitions[0]

            # rename lists to fit names in IonControl for ease of copy/paste
            # U1 part of Ramsey pulse train, fractions, and phases
            pulse_train_U1 = U1_transitions
            fractions_U1 = U1_pop_fractions
            simulated_phase_mask_U1 = U1_sim_phase_mask
            fixed_phase_mask_U1 = U1_fixed_phases

            # U2 part of Ramsey pulse train, fractions, and phases
            pulse_train_U2 = U2_transitions
            fractions_U2 = U2_pop_fractions
            simulated_phase_mask_U2 = U2_sim_phase_mask
            fixed_phase_mask_U2 = U2_fixed_phases

            # transitions used for readout of encoded states
            # we choose the fastest transitions associated with each state
            s12_trips = []
            s12_fracs = []
            s12_phases = []
            s12_sims = []
            if mediators:  # only do this S12 shelving step if we are bussing
                for idx, state in enumerate(states):
                    if state[0] == 0:
                        # need to find the next best state to shelve to that isn't in
                        # the list of states encoded
                        next_best = np.inf
                        first_state = states_num[idx]
                        for second_state in range(29):
                            if states[0][0] == 0:
                                check_state_list = [state_for_herald_num
                                                    ] + states_num
                            else:
                                check_state_list = states_num
                            if second_state not in check_state_list:
                                two_way_pi = self.all_pitimes[first_state][
                                    second_state]
                                if two_way_pi < next_best:
                                    next_best = two_way_pi
                                    state_for_RO_herald_num = second_state
                                else:
                                    pass
                            else:
                                pass
                        state_for_RO_herald = self.convert_index_to_state(
                            state_for_RO_herald_num)
                        s12_trips.append([
                            state[1], state_for_RO_herald[0],
                            state_for_RO_herald[1]
                        ])
                        s12_fracs.append(1)
                        s12_phases.append(0)
                        s12_sims.append(0)
                    else:
                        pass
            else:
                pass

            probe_trans = []
            for idx, state in enumerate(states):
                if state[0] == 0:
                    if mediators:
                        # need to find the next best state to shelve to that isn't in
                        # the list of states encoded
                        next_best = np.inf
                        first_state = states_num[idx]
                        for second_state in range(29):
                            if states[0][0] == 0:
                                check_state_list = [state_for_herald_num
                                                    ] + states_num
                            else:
                                check_state_list = states_num
                            if second_state not in check_state_list:
                                two_way_pi = self.all_pitimes[first_state][
                                    second_state]
                                if two_way_pi < next_best:
                                    next_best = two_way_pi
                                    state_for_RO_herald_num = second_state
                                else:
                                    pass
                            else:
                                pass
                        state_for_RO_herald = self.convert_index_to_state(
                            state_for_RO_herald_num)
                        probe_trans.append([
                            state[1], state_for_RO_herald[0],
                            state_for_RO_herald[1]
                        ])
                    else:
                        pass
                else:
                    # get index of state
                    state_num = self.convert_state_to_index(state)
                    # find which pitime is smallest to get the transition
                    trans_trip = [
                        self.convert_index_to_state(
                            np.nanargmin(self.all_pitimes[state_num]))[1],
                        state[0], state[1]
                    ]
                    probe_trans.append(trans_trip)

            total_Ramsey_pulse_time = np.sum(
                np.array(U1_trans_fractions + U2_trans_fractions))

            if save_to_file:
                save_path = (
                    '/home/nicholas/Documents/' +
                    'Barium137_Qudit/analysis/ramseys_pulse_sequences/')

                with open(
                        save_path + f'D{len(states)}_Ramsey_PulseSequence.txt',
                        'w') as file:
                    if states[0][0] == 0:
                        file.write('Initial State Manifold is S1/2.\n')
                    else:
                        file.write('Initial State Manifold is D5/2.\n')
                    file.write(
                        f'Total pulse time: {np.round(total_Ramsey_pulse_time,2)} us.\n'
                    )
                    file.write('####################################' +
                               '#################################### \n\n')

                    file.write(f'initial_state = [{initial_state}]\n\n')

                    file.write(f'pulse_train_U1 = {pulse_train_U1}\n')
                    file.write(f'fractions_U1 = {fractions_U1}\n')
                    file.write(
                        f'simulated_phase_mask_U1 = {simulated_phase_mask_U1}\n'
                    )
                    file.write(
                        f'fixed_phase_mask_U1 = {fixed_phase_mask_U1}\n')

                    file.write(f'\npulse_train_U2 = {pulse_train_U2}\n')
                    file.write(f'fractions_U2 = {fractions_U2}\n')
                    file.write(
                        f'simulated_phase_mask_U2 = {simulated_phase_mask_U2}\n'
                    )
                    file.write(
                        f'fixed_phase_mask_U2 = {fixed_phase_mask_U2}\n')

                    file.write(f'\ns12_state_shelvings = {s12_trips}\n')
                    file.write(f's12_state_fractions = {s12_fracs}\n')
                    file.write(f's12_state_fixed_phases = {s12_phases}\n')
                    file.write(f's12_state_simulated_phases = {s12_sims}\n')

                    file.write(f'\nprobe_trans = {probe_trans}\n')

                    file.write(
                        '\npulse_train_U2 = pulse_train_U2 + s12_state_shelvings'
                    )
                    file.write(
                        '\nfractions_U2 = fractions_U2 + s12_state_fractions')
                    file.write('\nfixed_phase_mask_U2 = ' +
                               'fixed_phase_mask_U2 + s12_state_fixed_phases')
                    file.write(
                        '\nsimulated_phase_mask_U2 = ' +
                        'simulated_phase_mask_U2 + s12_state_simulated_phases\n'
                    )
                    file.write('####################################' +
                               '#################################### \n\n')
            else:
                pass

            if verbose:
                # Print results if not saving to file..
                if states[0][0] == 0:
                    print('\nInitial State Manifold is S1/2.')
                else:
                    print('Initial State Manifold is D5/2.')
                print(
                    f'Total pulse time: {np.round(total_Ramsey_pulse_time,2)} us.'
                )
                print('####################################' +
                      '#################################### \n')
                print(f'initial_state = [{initial_state}]\n')

                print(f'pulse_train_U1 = {pulse_train_U1}')
                print(f'fractions_U1 = {fractions_U1}')
                print(f'simulated_phase_mask_U1 = {simulated_phase_mask_U1}')
                print(f'fixed_phase_mask_U1 = {fixed_phase_mask_U1}')

                print(f'\npulse_train_U2 = {pulse_train_U2}')
                print(f'fractions_U2 = {fractions_U2}')
                print(f'simulated_phase_mask_U2 = {simulated_phase_mask_U2}')
                print(f'fixed_phase_mask_U2 = {fixed_phase_mask_U2}')

                print(
                    f'\nU1 and U2 pitimes = {U1_trans_pitimes+U2_trans_pitimes}'
                )

                print(f'\ns12_state_shelvings = {s12_trips}\n')
                print(f'probe_trans = {probe_trans}\n')

                print('####################################' +
                      '#################################### \n\n')
            else:
                pass

        # return the full pulse time for the Ramsey experiment
        return total_Ramsey_pulse_time

    def all_used_transitions(self, max_dimension: int = 16) -> list:
        """Load triplets from pulse sequence files and return a combined list.

        Parameters:
        max_dimension (int): The maximum dimension number to load files from D2 to D<max_dimension>.

        Returns:
        list: A list containing all triplets from pulse_train_U1, pulse_train_U2,
            s12_state_shelvings, and probe_trans from the specified files.
        """
        combined_triplets = []
        relative_dir = 'ramseys_pulse_sequences/'

        for dimension in range(2, max_dimension + 1):
            print('dimension', dimension)
            # Construct the file name based on the dimension
            filename = f'D{dimension}_Ramsey_PulseSequence.txt'
            # Construct the full path using the relative directory
            full_path = os.path.join(relative_dir, filename)

            # Check if the file exists
            if os.path.isfile(full_path):
                with open(full_path, 'r') as file:
                    content = file.read()

                    # Use regex to find all the required lists in the text
                    initial_state = re.search(
                        r'initial_state\s*=\s*(\[\[.*?\]\])', content)
                    pulse_train_U1 = re.search(
                        r'pulse_train_U1\s*=\s*(\[\[.*?\]\])', content)
                    pulse_train_U2 = re.search(
                        r'pulse_train_U2\s*=\s*(\[\[.*?\]\])', content)
                    s12_state_shelvings = re.search(
                        r's12_state_shelvings\s*=\s*(\[\[.*?\]\])', content)
                    probe_trans = re.search(r'probe_trans\s*=\s*(\[\[.*?\]\])',
                                            content)

                    # Helper function to extract lists from matches
                    def extract_list(match):
                        return eval(match.group(1)) if match else []

                    # Extract and add the lists to the combined list
                    combined_triplets.extend(extract_list(pulse_train_U1))
                    combined_triplets.extend(extract_list(pulse_train_U2))
                    combined_triplets.extend(extract_list(s12_state_shelvings))
                    combined_triplets.extend(extract_list(probe_trans))

                    # Print what was found
                    print(f'dimension {dimension}')
                    print(f'initial_state: {initial_state}')
                    print(f'pulse_train_U1: {pulse_train_U1}')
                    print(f'pulse_train_U2: {pulse_train_U2}')
                    print(f's12_state_shelvings: {s12_state_shelvings}')
                    print(f'probe_trans: {probe_trans}')

        print('The combined list of transitions is:\n', combined_triplets)
        all_used_transitions = np.unique(combined_triplets, axis=0)
        print('The combined, unique list of transitions is:\n',
              all_used_transitions)
        print('Total number of transitions used: ', len(all_used_transitions),
              '\n')

        # taken from ramsey_calibration data
        calibrated_transitions = np.array([[0, 2, 0], [-1, 4, -3], [-2, 1, -1],
                                           [-2, 1, 0], [-2, 2,
                                                        -2], [-2, 2, -1],
                                           [-2, 2, 0], [-2, 3,
                                                        -3], [-2, 3, -2],
                                           [-2, 3, -1], [-2, 4,
                                                         -4], [-1, 1, -1],
                                           [-1, 1, 1], [-1, 2, -2], [-1, 2, 0],
                                           [-1, 3, -2], [-1, 3, -1],
                                           [-1, 3, 0], [-1, 4, -2], [0, 1, 1],
                                           [0, 2, 2], [0, 3, -1], [0, 3, 0],
                                           [0, 3, 1], [0, 3, 2], [0, 4, -2],
                                           [0, 4, -1], [0, 4, 0], [0, 4, 1],
                                           [1, 1, 1], [1, 2, 2], [1, 3, -1],
                                           [1, 3, 2], [1, 3, 3], [1, 4, -1],
                                           [1, 4, 0], [1, 4, 1], [1, 4, 2],
                                           [2, 2, 0], [2, 2, 1], [2, 2, 2],
                                           [2, 4, 0], [2, 4, 1], [2, 4, 2],
                                           [2, 4, 3], [2, 4, 4]])

        triplets_in_calibrated_not_in_all_used = np.array([
            x for x in calibrated_transitions
            if x.tolist() not in all_used_transitions.tolist()
        ])

        # Find triplets in all_used_transitions but not in calibrated_transitions
        triplets_in_all_used_not_in_calibrated = np.array([
            x for x in all_used_transitions
            if x.tolist() not in calibrated_transitions.tolist()
        ])

        print(
            "Triplets in calibrated_transitions not in all_used_transitions:")
        print(triplets_in_calibrated_not_in_all_used)

        print(
            "Triplets in all_used_transitions not in calibrated_transitions:")
        print(triplets_in_all_used_not_in_calibrated)

        return all_used_transitions


if __name__ == '__main__':
    start_time = time.time()

    rp = RamseyPulseFinder()
    # dimensions = [8, 9, 16]
    # dimensions = [3]
    dimensions = np.arange(2, 29, 1)
    rp.find_ordered_states(dimensions, verbose=True, save_to_file=True)

    # dimensions = np.arange(2, 18, 1)
    Ramsey_pulse_lengths = []
    for dimension in dimensions:
        Ramsey_pulse_length = rp.find_ordered_pulse_sequence([dimension],
                                                             verbose=True,
                                                             save_to_file=True)
        Ramsey_pulse_lengths.append(Ramsey_pulse_length)

        print(f'Dimension: {dimension}. Pulse time:' +
              f' {np.round(Ramsey_pulse_length, 2)} us.')

    rsp = RamseyStarPulseFinder()
    rsp_dimensions = np.arange(2, 18, 1)
    # rsp_dimensions = [16]
    # rsp.find_ordered_states(rsp_dimensions, verbose=True, save_to_file=True)

    Ramsey_star_pulse_lengths = []
    for dimension in rsp_dimensions:
        Ramsey_star_pulse_length = rsp.find_ordered_pulse_sequence(
            [dimension], encoded_herald=False, verbose=True, save_to_file=True)
        Ramsey_star_pulse_lengths.append(Ramsey_star_pulse_length)

        print(f'Dimension: {dimension}. Pulse time:' +
              f' {np.round(Ramsey_star_pulse_length, 2)} us.')

    plt.plot(dimensions, Ramsey_pulse_lengths, label='Bussed')
    plt.plot(rsp_dimensions, Ramsey_star_pulse_lengths, label='Star')
    plt.title('Ramsey Pulse Times')
    plt.xlabel(r'Dimension ($d$)')
    plt.ylabel(r'Pulse Time ($\mu s$)')
    plt.legend()
    plt.grid()

    # rp.all_used_transitions(max_dimension=22)

    plt.show()
    print('--- Total time: %s seconds ---' % (time.time() - start_time))
