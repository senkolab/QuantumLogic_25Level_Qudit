import concurrent.futures
import os
import time
from collections import deque

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from class_barium import Barium

# from plot_utils import nice_fonts

# mpl.rcParams.update(nice_fonts)
np.set_printoptions(suppress=True)


class PathFinder(Barium):
    """PathFinder class for finding optimal paths in a coupling graph based on
    delta_m values.

    Inherits from the Barium class, initializing with a magnetic field and
    reference pi times.

    """

    def __init__(self,
                 B_field: float = 4.216,
                 ref_pi_times: list = [20.9982, 39.0571, 45.5531, 33.9067, 45.1339]
                 ):
                 # ref_pi_times: list = [21.897, 41.031, 45.832, 35.6, 43.23]):
        """Initialize the PathFinder with default values for the magnetic field
        and reference pi times.

        Parameters:
            B_field (float): Magnetic field strength. Default is 4.216.
            ref_pi_times (list): Reference pi times for transitions. Default is
            a predefined list.

        """
        super().__init__(B_field, ref_pi_times)

        self.transition_pitimes = np.loadtxt(
            'quick_reference_arrays/transition_pitimes.txt', delimiter=',')
        self.pitimes = self.transition_pitimes

        if hasattr(self, 'transition_pitimes'):
            self.pitimes = self.transition_pitimes
        else:
            self.generate_transition_strengths()
            self.pitimes = self.transition_pitimes

        # load in pitimes, delta_m values, and connection graph
        # self.pitimes = np.loadtxt('state_finder/pitimes_table_dummy.txt',
        #                           delimiter=',')
        self.generate_delta_m_table()
        self.delta_m = np.flip(self.delta_m[:, -5:], axis=0)

        self.connection_graph = self.generate_coupling_graph()

    def generate_coupling_graph(self, pos=None, plot_graph=False):
        """Generate a coupling graph based on the delta_m matrix.

        Parameters:
        pos : dict, optional

            A dictionary specifying the positions for the nodes. If None,
            default positions will be used.

        plot_graph : bool, optional
            If True, plots the generated coupling graph. Defaults to False.

        Returns:
        G : networkx.Graph
            The generated bipartite graph representing the coupling structure.

        """

        matrix = np.abs(self.delta_m) <= 2
        rows, cols = matrix.shape

        # Create a bipartite graph
        G = nx.Graph()
        Fs = [1, 2, 3, 4]
        row_labels = []
        for i in Fs:
            for j in range(2 * i + 1):
                mF = i - j
                row_labels.append([i, mF])
        col_labels = [-2, -1, 0, 1, 2]

        # Add edges based on non-zero elements in the matrix and assign weights
        for i in range(rows):
            for j in range(cols):
                if matrix[i, j] != 0:
                    G.add_edge(f"{row_labels[i]}",
                               f"{col_labels[j]}",
                               weight=self.pitimes[i, j])

        # Generate positions for the nodes
        if pos is None:
            pos = {
                '-2': (-2, -2),
                '-1': (-1, -2),
                '0': (0, -2),
                '1': (1, -2),
                '2': (2, -2),
            }
        for i in row_labels:
            pos[f"{i}"] = (i[1], abs(i[0] - 4))
        for i in row_labels:
            G.add_node(f"{i}", pos=pos[f'{i}'])
        for j in col_labels:
            G.add_node(f"{j}", pos=pos[f'{i}'])

        if plot_graph:
            colors = list(mpl.colors.TABLEAU_COLORS.values())
            plt.figure(figsize=(10, 7))
            # Draw edges with different colors for each column
            for j in range(cols):
                edges = [(f"{row_labels[i]}", f"{col_labels[j]}")
                         for i in range(rows) if matrix[i, j] != 0]
                # weights = [
                #     matrix[i, j] for i in range(rows) if matrix[i, j] != 0
                # ]
                nx.draw_networkx_edges(G,
                                       pos,
                                       edgelist=edges,
                                       edge_color=colors[j % len(colors)],
                                       width=2,
                                       alpha=0.6)
            # Draw nodes with rectangular patches
            xh = 0.5
            yh = 0.25
            ax = plt.gca()
            for node, (x, y) in pos.items():
                ax.add_patch(
                    mpl.patches.Rectangle((x - xh / 2, y - yh / 2),
                                          xh,
                                          yh,
                                          fill=True,
                                          color='white',
                                          ec='black',
                                          lw=1))
            nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')
            plt.xlim(-4.5, 4.5)
            plt.ylim(-2.5, 3.5)
            # plt.title('Coupling Graph from Matrix')

        return G

    def single_path_finder(self,
                           init_state: str = '[4, -4]',
                           final_state: str = '[2, -2]'):
        """Find the best path from the initial state to the final state within a
        connection graph.

        This method employs a breadth-first search approach to explore all
        possible paths from the initial state to the final state. It keeps track
        of the paths and calculates the total "pi-time" based on the weights of
        the connections. The path with the smallest total pi-time is returned.
        Crucially, it tracks the "best pi-time" path per iteration, and kills
        branches that already exceed that best time. It is therefore exhaustive,
        and optimal.

        Args:
            init_state (str): The starting state for the pathfinding. Default is
            '[4, -4]'.
            final_state (str): The target state for the pathfinding. Default is
            '[2, -2]'.

        Returns:
            tuple: A tuple containing:
                - list: The best path from the initial state to the final state.
                - float: The total pi-time of the best path, or np.nan if no
                  valid path is found.

        """
        # First check whether init_state == final_state
        if init_state == final_state:
            return [], np.nan

        queue = deque([[init_state]])
        paths_with_products = []
        # set a wildly large initial best time
        best_pi_time = np.inf

        while queue:
            path = queue.popleft()
            node = path[-1]

            if node == final_state:
                path_length = len(path)
                # Change sum to product of weights
                pi_time = np.sum([
                    self.connection_graph[path[i]][path[i + 1]]['weight']
                    for i in range(len(path) - 1)
                ])

                if pi_time < best_pi_time:
                    best_pi_time = pi_time
                    best_path = (path, path_length, best_pi_time)
                else:
                    pass

            else:
                for neighbor in self.connection_graph.neighbors(node):
                    if neighbor not in path:  # Avoid cycles
                        edge_weight = self.connection_graph[node][neighbor][
                            'weight']
                        new_path = list(path)
                        new_path.append(neighbor)
                        new_path_pi_time = np.sum([
                            self.connection_graph[path[i]][path[i +
                                                                1]]['weight']
                            for i in range(len(path) - 1)
                        ])

                        if new_path_pi_time < best_pi_time:
                            queue.append(new_path)
                        else:
                            pass

        # Find the path with the smallest product of weights
        return best_path[0], best_path[-1]

    def find_all_paths(self, show_mediators=True):
        """Find all paths between states and save the results to data files.

        This method computes paths from an initial state to a final state using
        concurrent processing. It collects results in a list, converts it to a
        pandas DataFrame, and saves the DataFrame to both a text file and a CSV
        file.

        Parameters:
            show_mediators (bool): If True, counts and displays mediators in
            paths. Defaults to True.

        Returns:
            pd.DataFrame: A DataFrame containing the initial state, final state,
                        path taken, and the effective pitime for each path.

        """
        # Initialise object to turn into dataframe later
        data = []

        all_path_states = [
            f'[{i[0]}, {i[1]}]' for i in np.flip(self.F_values_D52, axis=0)
        ] + [f'{i[1]}' for i in np.flip(self.F_values_S12[3:], axis=0)]

        # Modify compute_path to return the path and total pitime
        def compute_path(row, col, init_state, final_state):
            path, total_pitime = self.single_path_finder(
                init_state=init_state, final_state=final_state)
            return init_state, final_state, path, total_pitime

        # Set number of concurrent workers
        workers = int(os.cpu_count() / 2)
        print(f'Finding all paths using {workers} workers.')

        futures = []
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers) as executor:
            for row, init_state in enumerate(all_path_states):
                for col, final_state in enumerate(all_path_states):
                    # Don't calculate if row == col
                    if row!=col:
                        futures.append(
                            executor.submit(compute_path, row, col, init_state,
                                            final_state))
                    else:
                        pass

            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(futures)):
                init_state, final_state, path, total_pitime = future.result()
                # Append each result as a row to the data list
                data.append(
                    [str(init_state),
                     str(final_state), path,
                     total_pitime])  # Join path elements for readability

        # Convert the data into a pandas DataFrame
        df_paths = pd.DataFrame(data,
                                columns=[
                                    "Initial State", "Final State", "Path",
                                    "Effective Pi-Time"
                                ])

        # Save the DataFrame to a text file using to_string
        file_path = ('path_finder/all_paths_dataframe.txt')
        with open(file_path, 'w') as file:
            file.write(df_paths.to_string(index=False))

        # Save the DataFrame to a CSV file as well
        df_paths.to_csv(f'path_finder/all_paths_dataframe.csv', index=False)

        # print("Paths DataFrame saved to " + f"{file_path}")
        print(df_paths)

        if show_mediators:
            # Extract the middle entries of the paths to find mediators
            mediator_count = {}

            # Iterate through the paths
            for path in df_paths['Path']:
                # If there are more than 2 states in the path, we can have mediators
                if len(path) > 2:
                    # Get mediators (all but the first and last)
                    mediators = path[1:-1]  # Directly use the list slicing

                    # Count occurrences of each mediator
                    for mediator in mediators:
                        if mediator in mediator_count:
                            mediator_count[mediator] += 1
                        else:
                            mediator_count[mediator] = 1

            # Create a DataFrame from the mediator counts
            mediator_df = pd.DataFrame(list(mediator_count.items()),
                                       columns=['Mediator State', 'Count'])

            # Save the mediators count to a file if needed
            # mediator_df.to_csv('path_finder/mediators_count.csv', index=False)
            print("\nMediator states and their usage count:")
            print(mediator_df)

        return df_paths


if __name__ == '__main__':
    start_time = time.time()
    pf = PathFinder()
    # pf.generate_coupling_graph(plot_graph=True)

    # Uncomment the following lines to find a specific path:
    # result = pf.single_path_finder(init_state='[4, -4]', final_state='[2, -2]')
    # print(result)

    df_paths = pf.find_all_paths()

    plt.show()

    print('--- Total time: %s seconds ---' % (time.time() - start_time))
