# import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qutip as qt
import scipy as sc
from matplotlib import colors
from matplotlib.cm import get_cmap

from class_utils import Utils
from plot_utils import nice_fonts, set_size

mpl.rcParams.update(nice_fonts)
np.set_printoptions(suppress=True)


class Barium(Utils):

    def __init__(
        self,
        B_field: float = 4.20958,  # old coil value 4.216
        # ref_pitimes: list = [19.470, 35.554, 41.166, 30.108, 39.326]):
        # ref_pitimes: list = [21.46,36.92,45.203,32.072,44.55]):
        ref_pitimes: list = [18.6219, 31.7456, 37.8649, 27.5957, 36.4292]):
        super().__init__()

        self.B_field = B_field
        self.ref_pitimes = ref_pitimes
        self.B_field_range = np.linspace(0.00001, self.B_field, 10)

        # string attributes for reading in measurement data
        self.path_to_lab_data = ('../data/')
        self.folder_name = ('complete_calibrations_data/')
        self.file_name_stem = 'New_initialized_calibration_freq_files'
        self.input_file_datetime = '_20240406_1145'

    # Begin analysis methods
    def H_hyp(self,
              magnetic_dipole: float = 4000,
              electric_quadrupole: float = 0,
              magnetic_octupole: float = 0,
              J: float = 0.5):
        """
        Hyperfine Hamiltonian from Steck's Quantum Optics, page 318.
        """

        I_dot_J = qt.tensor(self.sp_mat(spin=self.nuc_I, axis='x'),
                            self.sp_mat(spin=J, axis='x')) + qt.tensor(
                                self.sp_mat(spin=self.nuc_I, axis='y'),
                                self.sp_mat(spin=J, axis='y')) + qt.tensor(
                                    self.sp_mat(spin=self.nuc_I, axis='z'),
                                    self.sp_mat(spin=J, axis='z'))

        if J == 0.5:
            quad_term_prefactor = 0
            quad_term = 0
            octu_term_prefactor = 0
            octu_term = 0
        else:
            quad_term_prefactor = (1 /
                                   (2 * self.nuc_I * (2 * self.nuc_I - 1) * J *
                                    (2 * J - 1)))

            quad_term = 3 * (I_dot_J)**2 + (1.5) * I_dot_J - self.nuc_I * (
                self.nuc_I + 1) * J * (J + 1) * qt.tensor(
                    self.sp_mat(spin=self.nuc_I, axis='eye'),
                    self.sp_mat(spin=J, axis='eye'))

            octu_term_prefactor = (1 / (self.nuc_I * (self.nuc_I - 1) *
                                        (2 * self.nuc_I - 1) * J * (J - 1) *
                                        (2 * J - 1)))

            octu_term = 10 * (I_dot_J)**3 + 20 * (I_dot_J)**2 + 2 * (
                I_dot_J) * (self.nuc_I * (self.nuc_I + 1) + J *
                            (J + 1) + 3 - 3 * self.nuc_I *
                            (self.nuc_I + 1) * J *
                            (J + 1)) - 5 * self.nuc_I * (
                                self.nuc_I + 1) * J * (J + 1) * qt.tensor(
                                    self.sp_mat(spin=self.nuc_I, axis='eye'),
                                    self.sp_mat(spin=J, axis='eye'))

        H_hyp = magnetic_dipole * I_dot_J + (
            electric_quadrupole * quad_term_prefactor *
            quad_term) + (magnetic_octupole * octu_term_prefactor * octu_term)

        return H_hyp

    def H_zee(self,
              B_field: float = 5,
              J: float = 0.5,
              S: float = 0.5,
              L: float = 0):
        """
        Calculate the Zeeman Hamiltonian for a given magnetic field
        strength and atomic state.

        Parameters:
        -----------
        B_field : float, optional
            Magnetic field strength in Gauss. Default is 5 Gauss.
        J : float, optional
            Total angular momentum quantum number. Default is 0.5.
        S : float, optional
            Spin angular momentum quantum number. Default is 0.5.
        L : float, optional
            Orbital angular momentum quantum number. Default is 0.

        Returns:
        --------
        H_zee : qutip.Qobj
            Zeeman Hamiltonian as a QuTiP quantum object.

        Note:
        -----
        This method uses the Landé g-factor calculated from J, S, and L,
        and includes both electronic and nuclear Zeeman effects.
        """

        H_zee = (self.bohr_magneton / self.planck_constant) * B_field * (
            self.LandeGj(J=J, S=S, L=L) *
            qt.tensor(self.sp_mat(spin=self.nuc_I, axis='eye'),
                      self.sp_mat(spin=J, axis='z')) +
            self.LandeGi * qt.tensor(self.sp_mat(spin=self.nuc_I, axis='z'),
                                     self.sp_mat(spin=J, axis='eye')))
        return H_zee

    def eigensolver(self,
                    orbital: str = 'S12',
                    B_field: float = 0.001,
                    basis: str = 'F'):

        orbital_params = {
            'S12': {
                'J': self.J12,
                'L': self.L12,
                'mag_dip': self.magnetic_dipole_S12,
                'elec_quad': self.electric_quadrupole_S12,
                'mag_octu': self.magnetic_octupole_S12,
                'transform': self.basis_transform_IJ_to_F(orbital='S12')
            },
            'D52': {
                'J': self.J52,
                'L': self.L52,
                'mag_dip': self.magnetic_dipole_D52,
                'elec_quad': self.electric_quadrupole_D52,
                'mag_octu': self.magnetic_octupole_D52,
                'transform': self.basis_transform_IJ_to_F(orbital='D52')
            }
        }

        if orbital not in orbital_params:
            raise ValueError(f"Unknown orbital: {orbital}")

        params = orbital_params[orbital]

        J = params['J']
        L = params['L']
        mag_dip = params['mag_dip']
        elec_quad = params['elec_quad']
        mag_octu = params['mag_octu']
        transform = params['transform']

        full_H = self.H_hyp(magnetic_dipole=mag_dip,
                            electric_quadrupole=elec_quad,
                            magnetic_octupole=mag_octu,
                            J=J) + self.H_zee(
                                B_field=B_field, J=J, S=self.S, L=L)

        full_H = qt.Qobj(full_H)

        # sort eigenstates and eigenenergies differently for S1/2 and D5/2
        # D5/2 has negative nuclear moment, and so flipped eigenenergies
        if orbital == 'S12':
            eigenenergies, eigenstates = full_H.eigenstates(sort='low')
        elif orbital == 'D52':
            eigenenergies, eigenstates = full_H.eigenstates(sort='low')
        else:
            raise (Exception)

        if basis == 'F':
            # transform to the F, m_F basis for eigenstates
            eigenstates = [transform * state for state in eigenstates]
        else:
            pass

        # convert eigenstates to numpy arrays and squeeze out extra axis
        eigenstates = np.squeeze(
            [eigenstate.full() for eigenstate in eigenstates])
        return eigenenergies, eigenstates, full_H

    def dataframe_evals_estates(self, orbital: str = 'S12', basis: str = 'F'):

        dataframe_names = {
            'S12': self.F_values_S12,
            'D52': self.F_values_D52
        }.get(orbital)

        if dataframe_names is None:
            raise ValueError(f"Unknown orbital: {orbital}")

        dataframe_keys = ['B field value', 'Hamiltonian'] + [
            str(name) + ' evalue' for name in dataframe_names
        ] + [str(name) + ' estate' for name in dataframe_names]
        df = pd.DataFrame(columns=dataframe_keys)

        for it, B_field in enumerate(self.B_field_range):

            fixed_order_estate = np.zeros(
                (len(dataframe_names), len(dataframe_names)))
            fixed_order_evalue = np.zeros(len(dataframe_names))
            if it == 0:
                evalue, estate, Ham = self.eigensolver(orbital=orbital,
                                                       B_field=B_field,
                                                       basis=basis)

                # the eigensolver throws in arbitrary signs for evectors
                # to make each eigenstate start out with +1 population in their
                # corres. F,mF basis state (at 0 B field) we implement this:
                if basis == 'F':
                    sign_fix = np.array(
                        [np.round(np.sum(state)) for state in estate])
                    estate = np.multiply(estate, sign_fix[:, np.newaxis])

                estate_last = estate

            elif it > 0:
                evalue, estate, Ham = self.eigensolver(orbital=orbital,
                                                       B_field=B_field,
                                                       basis=basis)

                for it2, state in enumerate(estate_last):
                    check_overlap = np.dot(estate, state)
                    idx = np.argmax(check_overlap**2)
                    if np.sum(check_overlap) < 0:
                        fixed_order_estate[it2, :] = -1 * np.real(
                            estate[idx, :])
                    elif np.sum(check_overlap) > 0:
                        fixed_order_estate[it2, :] = np.real(estate[idx, :])
                    fixed_order_evalue[it2] = evalue[idx]

                estate = fixed_order_estate
                evalue = fixed_order_evalue

            estate_last = estate

            df.loc[it] = [B_field, Ham] + [
                evalue[i] for i in range(len(dataframe_names))
            ] + [estate[i] for i in range(len(dataframe_names))]

        return df

    def check_lewty(self):
        """
        Checking the simulated hyperfine splitting of the F levels with
        those splittings listed in Lewty 2013 paper. Useful for checking that
        our hyperfine parameters are being used correctly.

        """

        # checking D52 manifold spacing at 0 field with Lewty13!
        eval, estate, Ham = self.eigensolver(orbital='D52', B_field=0)
        print('our splittings in Hz')
        print(eval[0] - eval[9])
        print(eval[9] - eval[16])
        print(eval[16] - eval[23])
        print('differences between Lewty and us in kHz')
        print(1e3 * (eval[0] - eval[9] + 0.5035105))
        print(1e3 * (eval[9] - eval[16] + 62.872301))
        print(1e3 * (eval[16] - eval[23] + 71.6759024))
        return None

    def plot_mag_field(self, orbital: str = 'S12', estates_plot: bool = True):

        if orbital == 'S12':
            dataframe_names = self.F_values_S12
            plotting_names = self.plotting_names_S12
            if hasattr(self, 'data_S12'):
                df = self.data_S12
            else:
                df = self.dataframe_evals_estates(orbital=orbital)

        elif orbital == 'D52':
            dataframe_names = self.F_values_D52
            plotting_names = self.plotting_names_D52
            if hasattr(self, 'data_D52'):
                df = self.data_D52
            else:
                df = self.dataframe_evals_estates(orbital=orbital)

        # Create the plot
        if orbital == 'S12':
            fig, (ax1, ax2) = plt.subplots(nrows=2,
                                           ncols=1,
                                           figsize=set_size(width='half'),
                                           sharex=True)

            for it, state in enumerate(dataframe_names[3:8]):
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                if it == 0:
                    ax1.plot(df['B field value'][:],
                             df[str(state) + ' evalue'][:],
                             linewidth=2.0,
                             color=colors[1],
                             label=r'$F=2$')
                else:
                    ax1.plot(df['B field value'][:],
                             df[str(state) + ' evalue'][:],
                             linewidth=2.0,
                             color=colors[1])

            for it, state in enumerate(dataframe_names[:3]):
                if it == 0:
                    ax2.plot(df['B field value'][:],
                             df[str(state) + ' evalue'][:],
                             linewidth=2.0,
                             color=colors[0],
                             label=r'$F=1$')
                else:
                    ax2.plot(df['B field value'][:],
                             df[str(state) + ' evalue'][:],
                             linewidth=2.0,
                             color=colors[0])

            ax1.grid(True)
            ax2.grid(True)
            ax1.legend()
            ax2.legend()
            ax1.set_title(r'$S_{1/2}$ Energies - B Field Dependence')
            ax1.set_ylabel('Energy Level / MHz')
            ax2.set_ylabel('Energy Level / MHz')

        elif orbital == 'D52':
            plt.figure(figsize=set_size(width='half'))
            plot_colors = [
                'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C0', 'C1',
                'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C2', 'C2', 'C2', 'C2',
                'C2', 'C3', 'C3', 'C3'
            ]
            vals = []
            for idx, state in enumerate(dataframe_names):
                # add in F level labels
                if idx in [21, 17, 10, 0]:
                    i = np.argwhere(np.array([21, 17, 10, 0]) == idx) + 1
                    plt.plot(df['B field value'][:],
                             df[str(state) + ' evalue'][:],
                             color=plot_colors[idx],
                             label=r'$\tilde{F}$' + f'={i[0][0]}')
                else:
                    plt.plot(df['B field value'][:],
                             df[str(state) + ' evalue'][:],
                             color=plot_colors[idx])
                vals.append(df[str(state) +
                               ' evalue'][len(self.B_field_range) - 1])

            plt.title(r'$D_{5/2}$ Energies - B Field Dependence')
            plt.ylabel('Energy Level / MHz')
            plt.legend()
            plt.grid()

        plt.xlabel('Magnetic Field / G')

        if estates_plot:
            for it, state in enumerate(dataframe_names):
                plt.figure(figsize=set_size(width='half'))
                for it2, state2 in enumerate(dataframe_names):
                    if np.sum(
                            np.absolute(df[str(state) + ' estate'].apply(
                                lambda x: x[it2]))) > 0.001:
                        plt.plot(df['B field value'][:],
                                 np.real(
                                     df[str(state) +
                                        ' estate'].apply(lambda x: x[it2])),
                                 label=plotting_names[it2])
                plt.title(plotting_names[it])
                # plt.xlabel('Magnetic Field Strength / G')
                # plt.ylabel('State Amplitude')
                plt.grid()
                plt.legend()
        else:
            pass

    def generate_transition_frequencies(self, F2M0f2m0_input: float = 545.689):

        # data = np.loadtxt(self.path_to_lab_data + self.folder_name +
        #                   self.file_name_stem + self.input_file_datetime +
        #                   '.txt',
        #                   delimiter=',')
        # measured_freqs = data[:, 0]
        # diff_from_545 = np.abs(measured_freqs - F2M0f2m0_guess)
        # idx = diff_from_545.argmin()
        # F2M0f2m0 = measured_freqs[idx]
        self.F2M0f2m0 = F2M0f2m0_input

        self.B_field_range = np.linspace(0.001, self.B_field, 10)

        self.data_S12 = self.dataframe_evals_estates(orbital='S12')
        self.data_D52 = self.dataframe_evals_estates(orbital='D52')

        # generate the delta_m value for each transition
        self.generate_delta_m_table()
        delta_m = self.delta_m[:, -5:]

        transitions = np.empty((24, 5))
        delta_m_TF = np.zeros_like(transitions, dtype=bool)
        for it, D52_state in enumerate(self.F_values_D52):
            for it2, S12_state in enumerate(self.F_values_S12[-5:]):
                if np.absolute(delta_m[it, it2]) <= 2:
                    delta_m_TF[it, it2] = True
                    transitions[it, it2] = self.data_D52[
                        str(D52_state) +
                        ' evalue'].iloc[-1] - self.data_S12[str(S12_state) +
                                                            ' evalue'].iloc[-1]
                elif np.absolute(delta_m[it, it2]) > 2:
                    delta_m_TF[it, it2] = False
                    transitions[it, it2] = np.nan
        transition_frequencies = np.flip(-transitions, axis=0)

        transition_frequencies = transition_frequencies - (
            transition_frequencies[5, 2] - self.F2M0f2m0)
        self.transition_frequencies = transition_frequencies

        return transition_frequencies

    def fit_mag_field(self, initial_guess: float = 4.2, delta=None):
        """delta is the splitting between the most
        positively and negatively B field sensitive transitions.

        initial_guess is the first guess for B field.

        """
        if delta:
            delta = delta[1] - delta[0]
        else:
            measured_matrix = self.generate_measured_transition_order()
            delta = measured_matrix[22, 1] - measured_matrix[5, 2]

        # define the helper function that will find the splitting each time
        def splitting_difference(B_field_array: list):

            B_field = B_field_array[
                0]  # Extract the B_field value from the array
            self.B_field = B_field

            transition_freq_matrix = self.generate_transition_frequencies()

            calculated_splitting = transition_freq_matrix[
                22, 1] - transition_freq_matrix[5, 2]

            return np.absolute(calculated_splitting - delta)

        print('Starting preliminary B field fit...')
        optimisation_params = sc.optimize.minimize(
            splitting_difference,
            [initial_guess],
            method='Nelder-Mead',
            # options={'maxiter': 10},
            bounds=[(3, 8)])
        found_B_field = optimisation_params.x
        self.B_field = found_B_field[0]
        print(f'\n The B field found is: {self.B_field} Gauss. \n')

        return found_B_field[0]

    def generate_sim_meas_comparison_table(self, do_B_field_fit: bool = False):
        """A wrapper for the compare_measured_transitions method. Compares
        simulated transition frequencies to those measured for
        calibration and generates a table."""

        if hasattr(self, 'measured_transition_order'):
            pass
        else:
            self.generate_measured_transition_order()
        measured_transition_order = self.measured_transition_order

        data = np.loadtxt(self.path_to_lab_data + self.folder_name +
                          self.file_name_stem + self.input_file_datetime +
                          '.txt',
                          delimiter=',')
        measured_frequencies = data[:, 0]

        if do_B_field_fit:
            fitted_B_field_value = self.fit_mag_field()
            self.B_field = fitted_B_field_value
        else:
            pass

        transitions = self.generate_transition_frequencies()

        calculated_frequencies = [
            transitions[row][col]
            for row, col in self.measured_transition_order
        ]

        measured_frequencies = np.pad(measured_frequencies,
                                      (0, 38 - len(measured_frequencies)),
                                      mode='constant')

        print('Measured transition frequencies \n',
              np.round(measured_frequencies, 6))

        print('Simulated transition frequencies \n',
              np.round(np.array(calculated_frequencies), 6))

        differences = 1e3 * \
            (calculated_frequencies - measured_frequencies)

        print('Differences \n', np.round(differences, 6))
        print('\n The mean discrepancy for all transitions is:',
              np.mean(np.abs(differences)))

        diff_array = np.round(differences, 3)

        diff_table = np.zeros((24, 5))

        for idx, pair in enumerate(measured_transition_order):
            diff_table[pair[0]][pair[1]] = diff_array[idx]
        print("\n Discrepancy, simulated vs. measured, table\n", diff_table)

        print('Simulation frequency table\n',
              np.array(np.round(self.transition_frequencies, 6)))
        return None

    def generate_transition_sensitivities(self):

        self.generate_delta_m_table()
        delta_m = self.delta_m[:, -5:]

        # if hasattr(self, 'data_D52') and hasattr(self, 'data_S12'):
        #     pass
        # else:

        self.B_field_range = np.append(
            np.linspace(0.001, self.B_field - 0.001, 10),
            np.array([self.B_field, self.B_field + 0.001]))
        self.data_S12 = self.dataframe_evals_estates(orbital='S12')
        self.data_D52 = self.dataframe_evals_estates(orbital='D52')

        sensitivities = np.zeros((24, 5))
        for it, D52_state in enumerate(self.F_values_D52):
            for it2, S12_state in enumerate(self.F_values_S12[-5:]):
                if np.absolute(delta_m[it, it2]) <= 2:
                    sensitivities[it, it2] = (
                        (self.data_D52[str(D52_state) + ' evalue'].iloc[-1] -
                         self.data_D52[str(D52_state) + ' evalue'].iloc[-3]) -
                        (self.data_S12[str(S12_state) + ' evalue'].iloc[-1] -
                         self.data_S12[str(S12_state) + ' evalue'].iloc[-3])
                    ) / (self.data_S12['B field value'].iloc[-1] -
                         self.data_S12['B field value'].iloc[-3])
                elif np.absolute(delta_m[it, it2]) > 2:
                    sensitivities[it, it2] = np.nan

        transition_sensitivities = np.flip(sensitivities, axis=0)
        self.transition_sensitivities = transition_sensitivities

        np.savetxt('quick_reference_arrays/transition_sensitivities.txt',
                   transition_sensitivities,
                   delimiter=',')
        return transition_sensitivities

    def generate_all_sensitivities(self, verbose=False):
        """Generate a matrix of sensitivities for various states based on the
        changes in energy values with respect to the magnetic field.

        This method computes the sensitivities of different quantum states to
        variations in the magnetic field, populating a matrix where each
        element represents the sensitivity between two states. The results can
        be visualized as a heatmap if specified by the `verbose` flag. The
        diagonal of the sensitivity matrix is filled with NaN values to
        indicate that a state does not have sensitivity to itself.

        Parameters:
        -----------
        verbose : bool, optional
            If set to True, a visual representation of the sensitivities will
            be displayed as a heatmap. Default is False.

        Returns:
        --------
        None

        Side Effects:
        --------------

        - Updates the instance attribute `self.all_sensitivities` with the
          computed sensitivity matrix.
        - Optionally displays a heatmap visualization of the sensitivities.

        Notes:
        -------

        - The method calculates slopes based on the energy evaluations for each
        state derived from magnetic field values in the `self.data_S12` and
        `self.data_D52` DataFrames.

        - The method utilizes the last three energy evaluations for
        calculating the slopes to determine the sensitivity of energy levels
        with respect to magnetic field variations.

        - The final sensitivity matrix is constructed by taking the difference
        between the slopes of states, and the elements below the anti-diagonal
        are negated for symmetry.

        """

        slopes_1d = []
        self.B_field_range = np.append(
            np.linspace(0.001, self.B_field - 0.0001, 10),
            np.array([self.B_field, self.B_field + 0.0001]))
        self.data_S12 = self.dataframe_evals_estates(orbital='S12')
        self.data_D52 = self.dataframe_evals_estates(orbital='D52')

        for it, D52_state in enumerate(reversed(self.F_values_D52)):
            slopes_1d.append(
                (self.data_D52[str(D52_state) + ' evalue'].iloc[-1] -
                 self.data_D52[str(D52_state) + ' evalue'].iloc[-3]) /
                (self.data_S12['B field value'].iloc[-1] -
                 self.data_S12['B field value'].iloc[-3]))
        for it2, S12_state in enumerate(reversed(self.F_values_S12[-5:])):
            slopes_1d.append(
                (self.data_S12[str(S12_state) + ' evalue'].iloc[-1] -
                 self.data_S12[str(S12_state) + ' evalue'].iloc[-3]) /
                (self.data_S12['B field value'].iloc[-1] -
                 self.data_S12['B field value'].iloc[-3]))

        slopes_1d = np.array(slopes_1d)
        column_slopes_1d = slopes_1d[:, np.newaxis]
        flipped_row = slopes_1d[::-1]
        all_sensitivities = np.array(column_slopes_1d - flipped_row)

        for i in range(all_sensitivities.shape[0]):
            for j in range(all_sensitivities.shape[0] - i - 1,
                           all_sensitivities.shape[0]):
                all_sensitivities[
                    i,
                    j] *= -1  # Multiply elements below the anti-diagonal by -1

        # Make entries with same state to same state filled with np.nan
        all_sensitivities = np.flipud(all_sensitivities)
        np.fill_diagonal(all_sensitivities, np.nan)
        all_sensitivities = np.flip(all_sensitivities)

        self.all_sensitivities = all_sensitivities

        if verbose:
            # create a new figure
            fig, ax = plt.subplots(figsize=(12, 8))

            im = plt.imshow(
                np.abs(all_sensitivities),
                aspect=0.7,
                # norm=norm,
                cmap='viridis_r')

            # Add a colorbar with a fraction of the axes height and padding
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(r'$\Delta_B$')

            S12_list = self.plotting_names_S12[-5:]
            D52_list = self.plotting_names_D52
            full_plotting_names = list(
                reversed([item
                          for item in S12_list] + [item for item in D52_list]))
            plt.xticks(np.arange(all_sensitivities.shape[1]),
                       full_plotting_names,
                       rotation=90)
            plt.yticks(np.arange(all_sensitivities.shape[0]),
                       full_plotting_names)

            for i in range(all_sensitivities.shape[0]):
                for j in range(all_sensitivities.shape[1]):
                    plt.text(j,
                             i,
                             f"{np.round(all_sensitivities[i,j],3)}",
                             fontsize=5,
                             ha="center",
                             va="center",
                             color="black" if np.abs(all_sensitivities[i, j])
                             < 4 else "white")

            # set the title
            plt.title(r"All Sensitivities / MHzG$^{-1}$")
            plt.tight_layout()

            # Save the plot as both PNG and PDF
            # plt.savefig('quick_reference_arrays/all_sensitivities_table.png')
            # plt.savefig('quick_reference_arrays/all_sensitivities_table.pdf')
            np.savetxt(
                'quick_reference_arrays/all_transition_sensitivities.txt',
                all_sensitivities,
                delimiter=',')

    def generate_frequencies_sensitivities_table(self, Ba138_relative=False):

        if hasattr(self, 'transition_sensitivities'):
            sensitivities = self.transition_sensitivities
        else:
            sensitivities = self.generate_transition_sensitivities()
        if hasattr(self, 'transition_frequencies'):
            frequencies = self.transition_frequencies[:, -5:]
        else:
            frequencies = self.generate_transition_frequencies()

        # create a new figure
        fig, ax = plt.subplots(figsize=(8, 6))

        im = plt.imshow(np.absolute(sensitivities),
                        aspect=0.25,
                        cmap='viridis_r')
        plt.colorbar(im)

        plt.xticks(np.arange(sensitivities.shape[1]),
                   self.plotting_names_S12[-5:])
        plt.yticks(np.arange(sensitivities.shape[0]),
                   np.flip(self.plotting_names_D52))

        # generate the delta_m value for each transition
        self.generate_delta_m_table()
        delta_m = np.flip(self.delta_m[:, -5:], axis=0)

        if Ba138_relative:
            # two options here.

            # a) make it relative to Ba138's bare S-D5/2 transition at 0 field
            # b) make it relative to the S1/2, F=2 to D5/2, F=2 transition of
            # Ba137 at 0 field (this makes the transitions a bit easier to read)

            # a) uses all three fixes. b) doesn't use the 3GHz shift
            frequencies = frequencies - 33.144196  # D5/2 F=2 shift
            frequencies = frequencies - self.F2M0f2m0  # 545.689  # 1762 locking offset
            frequencies = frequencies + 3014.153139  # S1/2 F=2 shift

            frequencies = (-1) * frequencies

        # loop over data dimensions and create text annotations
        sensitivities = np.round(sensitivities, 3)
        frequencies = np.round(frequencies, 3)
        # print(np.round(sensitivities, 4))

        for i in range(sensitivities.shape[0]):
            for j in range(sensitivities.shape[1]):
                if np.absolute(delta_m[i, j]) <= 2:
                    plt.text(j,
                             i,
                             rf"$f_r$: {frequencies[i,j]}" + "\n"
                             rf"$\Delta_B$: {np.round(sensitivities[i,j], 3)}",
                             fontsize=8,
                             ha="center",
                             va="center",
                             color="black"
                             if np.abs(sensitivities[i, j]) < 2 else "white")

        # set the title
        plt.title("Frequencies (MHz) and Sensitivities (MHz/G)")

    def generate_init_SPAM_frequencies_plot(self):

        data = np.loadtxt(self.path_to_lab_data + self.folder_name +
                          self.file_name_stem + self.input_file_datetime +
                          '.txt',
                          delimiter=',')
        measured_freqs = data[:, 0]

        # Create a figure and axis
        fig, ax = plt.subplots()

        init_indices = [0, 1, 2, 3, 4, 5, 6, 20, 25, 27]
        for freq in measured_freqs[init_indices]:
            # Plot vertical lines for initialisation transitions
            ax.plot([freq, freq], [0, 1],
                    color='k',
                    linestyle='-',
                    label='Initialisation')

        for freq in measured_freqs[13:37]:
            # Plot vertical lines for initialisation transitions
            ax.plot([freq, freq], [0, 1],
                    color='b',
                    linestyle='-',
                    label='Shelving')

        # Add a concise legend
        handles, labels = ax.get_legend_handles_labels()
        # Get unique labels
        unique_labels = list(set(labels))
        unique_handles = [
            handles[labels.index(label)] for label in unique_labels
        ]  # Corresponding handles
        ax.legend(unique_handles, unique_labels)

        plt.xlabel("Transition Frequency (MHz)")
        plt.title("Plot of Initialisation and Shelving Frequencies")
        plt.grid(True)

        # Create sample data with a discontinuity
        x1 = np.linspace(0, 5, 100)
        y1 = np.sin(x1)

        x2 = np.linspace(10, 15, 100)
        y2 = np.cos(x2)

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(8, 4))

        # Plot the first data segment
        ax.plot(x1, y1, label='Segment 1', color='blue')

        # Plot the second data segment
        ax.plot(x2, y2, label='Segment 2', color='green')

        # Set x-axis limits to show only the relevant segments
        ax.set_xlim(0, 15)

        # Add a vertical line to indicate the discontinuity
        ax.axvline(x=5, color='red', linestyle='--', label='Discontinuity')

        # Add labels and legend
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title('Plot with Discontinuous X-axis')
        ax.legend()

    def generate_transition_frequency_sidebands(
            self,
            x_sec_freq: float = 1.163,
            y_sec_freq: float = 1.357,
            micromotion_freq: float = 20.772,
            include_sidebands: bool = True,
            SPAM_freqs_only: bool = False,
            gen_plot: bool = True):
        '''
        A fully theoretical calculation for frequencies with sidebands.
        '''

        if hasattr(self, 'transition_frequencies'):
            frequencies = self.transition_frequencies
        else:
            frequencies = self.generate_transition_frequencies()

        freq_1D = np.ravel(frequencies[~np.isnan(frequencies)])
        freq_px_1D = freq_1D + x_sec_freq
        freq_py_1D = freq_1D + y_sec_freq
        freq_pmu_1D = freq_1D + micromotion_freq
        freq_nx_1D = freq_1D - x_sec_freq
        freq_ny_1D = freq_1D - y_sec_freq
        freq_nmu_1D = freq_1D - micromotion_freq
        all_sidebands = np.array([
            freq_1D, freq_px_1D, freq_py_1D, freq_pmu_1D, freq_nx_1D,
            freq_ny_1D, freq_nmu_1D
        ])

        # Get a colormap with as many colors as there are columns
        cmap = get_cmap("tab10", frequencies.shape[1])
        self.generate_measured_transition_order()
        SPAM_order = self.measured_transition_order[13:37]
        SPAM_order[3, :] = [20, 3]

        if gen_plot:
            # Create a figure and axis
            fig, ax = plt.subplots(figsize=set_size(width='full'))

            if SPAM_freqs_only:
                for pair in SPAM_order:
                    value = frequencies[pair[0], pair[1]]
                    col_idx = pair[1]
                    row_idx = pair[0]
                    label_name = r'$|$' + self.plotting_names_S12[
                        col_idx + 3] + r'$\rangle$'

                    # Plot vertical lines for main transitions
                    ax.plot([value, value], [0, 1],
                            color=cmap(col_idx),
                            linestyle='-',
                            label=label_name)
                    # ax.annotate(
                    #     r'$|$' + self.plotting_names_D52[23 - row_idx] +
                    #     r'$\rangle$',
                    #     xy=(value, 0.45),
                    #     xytext=(-5, 0),
                    #     textcoords="offset points",
                    #     rotation=-90,
                    #     color=cmap(col_idx),
                    #     bbox=dict(boxstyle='round,pad=0.3',
                    #               facecolor='white',
                    #               edgecolor='white'))
                    if include_sidebands:
                        # Plot positive x secular frequency sidebands
                        ax.plot([value + x_sec_freq, value + x_sec_freq],
                                [0, 1],
                                color=cmap(col_idx),
                                linestyle='--')
                        ax.annotate(r'$|$' +
                                    self.plotting_names_D52[23 - row_idx] +
                                    r'$\rangle$',
                                    xy=(value + x_sec_freq, 0.45),
                                    xytext=(-5, 0),
                                    textcoords="offset points",
                                    rotation=-90,
                                    color=cmap(col_idx),
                                    bbox=dict(boxstyle='round,pad=0.3',
                                              facecolor='white',
                                              edgecolor='white'))

                        # Plot negative x secular frequency sidebands
                        ax.plot([value - x_sec_freq, value - x_sec_freq],
                                [0, 1],
                                color=cmap(col_idx),
                                linestyle='--')
                        ax.annotate(r'$|$' +
                                    self.plotting_names_D52[23 - row_idx] +
                                    r'$\rangle$',
                                    xy=(value - x_sec_freq, 0.45),
                                    xytext=(-5, 0),
                                    textcoords="offset points",
                                    rotation=-90,
                                    color=cmap(col_idx),
                                    bbox=dict(boxstyle='round,pad=0.3',
                                              facecolor='white',
                                              edgecolor='white'))

                        # Plot positive y secular frequency sidebands
                        ax.plot([value + y_sec_freq, value + y_sec_freq],
                                [0, 1],
                                color=cmap(col_idx),
                                linestyle='-.')
                        ax.annotate(r'$|$' +
                                    self.plotting_names_D52[23 - row_idx] +
                                    r'$\rangle$',
                                    xy=(value + y_sec_freq, 0.45),
                                    xytext=(-5, 0),
                                    textcoords="offset points",
                                    rotation=-90,
                                    color=cmap(col_idx),
                                    bbox=dict(boxstyle='round,pad=0.3',
                                              facecolor='white',
                                              edgecolor='white'))

                        # Plot negative y secular frequency sidebands
                        ax.plot([value - y_sec_freq, value - y_sec_freq],
                                [0, 1],
                                color=cmap(col_idx),
                                linestyle='-.')
                        ax.annotate(r'$|$' +
                                    self.plotting_names_D52[23 - row_idx] +
                                    r'$\rangle$',
                                    xy=(value - y_sec_freq, 0.45),
                                    xytext=(-5, 0),
                                    textcoords="offset points",
                                    rotation=-90,
                                    color=cmap(col_idx),
                                    bbox=dict(boxstyle='round,pad=0.3',
                                              facecolor='white',
                                              edgecolor='white'))

                        # Plot positive micromotion frequency sidebands
                        ax.plot([
                            value + micromotion_freq, value + micromotion_freq
                        ], [0, 1],
                                color=cmap(col_idx),
                                linestyle=':')
                        ax.annotate(r'$|$' +
                                    self.plotting_names_D52[23 - row_idx] +
                                    r'$\rangle$',
                                    xy=(value + micromotion_freq, 0.45),
                                    xytext=(-5, 0),
                                    textcoords="offset points",
                                    rotation=-90,
                                    color=cmap(col_idx),
                                    bbox=dict(boxstyle='round,pad=0.3',
                                              facecolor='white',
                                              edgecolor='white'))

                        # Plot negative micromotion frequency sidebands
                        ax.plot([
                            value - micromotion_freq, value - micromotion_freq
                        ], [0, 1],
                                color=cmap(col_idx),
                                linestyle=':')
                        ax.annotate(r'$|$' +
                                    self.plotting_names_D52[23 - row_idx] +
                                    r'$\rangle$',
                                    xy=(value - micromotion_freq, 0.45),
                                    xytext=(-5, 0),
                                    textcoords="offset points",
                                    rotation=-90,
                                    color=cmap(col_idx),
                                    bbox=dict(boxstyle='round,pad=0.3',
                                              facecolor='white',
                                              edgecolor='white'))
                    else:
                        pass

            else:
                # Loop over columns
                for col_idx in range(frequencies.shape[1]):
                    column_values = frequencies[:, col_idx]
                    label_name = r'$|$' + self.plotting_names_S12[
                        col_idx + 3] + r'$\rangle$'
                    for row_idx, value in enumerate(column_values):
                        if not np.isnan(value):
                            # Plot vertical lines for main transitions
                            ax.plot([value, value], [0, 1],
                                    color=cmap(col_idx),
                                    linestyle='-',
                                    label=label_name)
                            ax.annotate(r'$|$' +
                                        self.plotting_names_D52[23 - row_idx] +
                                        r'$\rangle$',
                                        xy=(value, 0.45),
                                        xytext=(-5, 0),
                                        textcoords="offset points",
                                        rotation=-90,
                                        color=cmap(col_idx),
                                        bbox=dict(boxstyle='round,pad=0.3',
                                                  facecolor='white',
                                                  edgecolor='white'))
                            if include_sidebands:
                                # Plot positive x secular frequency sidebands
                                ax.plot(
                                    [value + x_sec_freq, value + x_sec_freq],
                                    [0, 1],
                                    color=cmap(col_idx),
                                    linestyle='--')
                                ax.annotate(
                                    r'$|$' +
                                    self.plotting_names_D52[23 - row_idx] +
                                    r'$\rangle$',
                                    xy=(value + x_sec_freq, 0.45),
                                    xytext=(-5, 0),
                                    textcoords="offset points",
                                    rotation=-90,
                                    color=cmap(col_idx),
                                    bbox=dict(boxstyle='round,pad=0.3',
                                              facecolor='white',
                                              edgecolor='white'))

                                # Plot negative x secular frequency sidebands
                                ax.plot(
                                    [value - x_sec_freq, value - x_sec_freq],
                                    [0, 1],
                                    color=cmap(col_idx),
                                    linestyle='--')
                                ax.annotate(
                                    r'$|$' +
                                    self.plotting_names_D52[23 - row_idx] +
                                    r'$\rangle$',
                                    xy=(value - x_sec_freq, 0.45),
                                    xytext=(-5, 0),
                                    textcoords="offset points",
                                    rotation=-90,
                                    color=cmap(col_idx),
                                    bbox=dict(boxstyle='round,pad=0.3',
                                              facecolor='white',
                                              edgecolor='white'))

                                # Plot positive y secular frequency sidebands
                                ax.plot(
                                    [value + y_sec_freq, value + y_sec_freq],
                                    [0, 1],
                                    color=cmap(col_idx),
                                    linestyle='-.')
                                ax.annotate(
                                    r'$|$' +
                                    self.plotting_names_D52[23 - row_idx] +
                                    r'$\rangle$',
                                    xy=(value + y_sec_freq, 0.45),
                                    xytext=(-5, 0),
                                    textcoords="offset points",
                                    rotation=-90,
                                    color=cmap(col_idx),
                                    bbox=dict(boxstyle='round,pad=0.3',
                                              facecolor='white',
                                              edgecolor='white'))

                                # Plot negative y secular frequency sidebands
                                ax.plot(
                                    [value - y_sec_freq, value - y_sec_freq],
                                    [0, 1],
                                    color=cmap(col_idx),
                                    linestyle='-.')
                                ax.annotate(
                                    r'$|$' +
                                    self.plotting_names_D52[23 - row_idx] +
                                    r'$\rangle$',
                                    xy=(value - y_sec_freq, 0.45),
                                    xytext=(-5, 0),
                                    textcoords="offset points",
                                    rotation=-90,
                                    color=cmap(col_idx),
                                    bbox=dict(boxstyle='round,pad=0.3',
                                              facecolor='white',
                                              edgecolor='white'))

                                # Plot positive micromotion frequency sidebands
                                ax.plot([
                                    value + micromotion_freq,
                                    value + micromotion_freq
                                ], [0, 1],
                                        color=cmap(col_idx),
                                        linestyle=':')
                                ax.annotate(
                                    r'$|$' +
                                    self.plotting_names_D52[23 - row_idx] +
                                    r'$\rangle$',
                                    xy=(value + micromotion_freq, 0.45),
                                    xytext=(-5, 0),
                                    textcoords="offset points",
                                    rotation=-90,
                                    color=cmap(col_idx),
                                    bbox=dict(boxstyle='round,pad=0.3',
                                              facecolor='white',
                                              edgecolor='white'))

                                # Plot negative micromotion frequency sidebands
                                ax.plot([
                                    value - micromotion_freq,
                                    value - micromotion_freq
                                ], [0, 1],
                                        color=cmap(col_idx),
                                        linestyle=':')
                                ax.annotate(
                                    r'$|$' +
                                    self.plotting_names_D52[23 - row_idx] +
                                    r'$\rangle$',
                                    xy=(value - micromotion_freq, 0.45),
                                    xytext=(-5, 0),
                                    textcoords="offset points",
                                    rotation=-90,
                                    color=cmap(col_idx),
                                    bbox=dict(boxstyle='round,pad=0.3',
                                              facecolor='white',
                                              edgecolor='white'))
                            else:
                                pass

            # Add a concise legend
            handles, labels = ax.get_legend_handles_labels()
            # Get unique labels
            unique_labels = list(set(labels))
            unique_handles = [
                handles[labels.index(label)] for label in unique_labels
            ]  # Corresponding handles
            ax.legend(unique_handles, unique_labels)

            plt.xlabel("Transition Frequency (MHz)")
            if include_sidebands:
                plt.title("Plot of Transitions with Sidebands")
            else:
                plt.title("Plot of Transitions (Carrier Only)")
            plt.grid(True)
        else:
            pass

        return all_sidebands

    def generate_geometric_factors(self,
                                   xy_plane_angle: float = 58.0,
                                   wavevector_angle: float = 45.0):

        def geom_factor(q: int = 0,
                        xy_plane_angle: float = 58.0,
                        wavevector_angle: float = 45.0):

            # convert input angles to radians
            xy_plane_angle = (np.pi * xy_plane_angle) / 180
            wavevector_angle = (np.pi * wavevector_angle) / 180

            # From Pei Jiang PhD thesis, page 51
            if q == 0:
                return (np.sqrt(6) / 4) * np.absolute(
                    np.cos(xy_plane_angle) * np.sin(2 * wavevector_angle))
            elif np.absolute(q) == 1:
                return (1 / 2) * np.absolute(-q * np.cos(xy_plane_angle) *
                                             np.cos(2 * wavevector_angle) +
                                             1j * np.sin(xy_plane_angle) *
                                             np.cos(wavevector_angle))
            elif np.absolute(q) == 2:
                return (1 / 4) * np.absolute(
                    np.cos(xy_plane_angle) * np.sin(2 * wavevector_angle) -
                    q * 1j * np.sin(xy_plane_angle) * np.sin(wavevector_angle))

            else:
                return 0

        factors = np.zeros((24, 8))
        for it, D52_state in enumerate(self.F_values_D52):
            for it2, S12_state in enumerate(self.F_values_S12):
                qF_dummy = D52_state[1] - S12_state[1]
                factors[it,
                        it2] = geom_factor(q=qF_dummy,
                                           xy_plane_angle=xy_plane_angle,
                                           wavevector_angle=wavevector_angle)

        return factors

    def generate_raw_transition_strengths(self, table_size: str = 'F2'):

        self.B_field_range = np.linspace(0.001, self.B_field, 10)

        # generate the dataframes for S12 and D52 in IJ basis
        df_S12 = self.dataframe_evals_estates(orbital='S12', basis='IJ')
        df_D52 = self.dataframe_evals_estates(orbital='D52', basis='IJ')

        # # generate the delta_m value for each transition
        # delta_m = self.generate_delta_m_table()
        # delta_m = self.delta_m[:, -5:]
        self.generate_delta_m_table()
        delta_m = self.delta_m

        raw_transition_strengths = np.zeros((24, 8))
        for it, D52_state in enumerate(self.F_values_D52):
            for it2, S12_state in enumerate(self.F_values_S12):
                if np.absolute(delta_m[it, it2]) <= 2:
                    strength = 0
                    for idx_D52, evec_mag_D52 in enumerate(
                            df_D52[str(D52_state) + ' estate'].iloc[-1]):
                        for idx_S12, evec_mag_S12 in enumerate(
                                df_S12[str(S12_state) + ' estate'].iloc[-1]):
                            if self.IJ_values_D52[idx_D52][
                                    0] - self.IJ_values_S12[idx_S12][0] == 0:
                                qJ_dummy = self.IJ_values_D52[idx_D52][
                                    1] - self.IJ_values_S12[idx_S12][1]
                                cg_coeff_dummy = self.clebsch(
                                    0.5, self.IJ_values_S12[idx_S12][1], 2.0,
                                    qJ_dummy, 2.5,
                                    self.IJ_values_D52[idx_D52][1])
                                strength += (evec_mag_S12 * evec_mag_D52 *
                                             cg_coeff_dummy)
                            else:
                                strength += 0
                    raw_transition_strengths[it, it2] = strength

                elif np.absolute(delta_m[it, it2]) > 2:
                    raw_transition_strengths[it, it2] = np.nan

        return raw_transition_strengths

    def generate_transition_strengths(self,
                                      xy_plane_angle: float = 58.0,
                                      wavevector_angle: float = 45.0,
                                      table_size: str = 'F2'):

        raw_transition_strengths = self.generate_raw_transition_strengths(
            table_size='full')
        geom_factor_array = self.generate_geometric_factors(
            xy_plane_angle=xy_plane_angle, wavevector_angle=wavevector_angle)

        transition_strengths = geom_factor_array * raw_transition_strengths

        transition_strengths = np.flip(np.absolute(transition_strengths),
                                       axis=0)

        # transition_strengths = np.round(transition_strengths, 4)
        transition_strengths[:, [0, 2]] = transition_strengths[:, [2, 0]]

        self.generate_delta_m_table()
        delta_m = np.flip(self.delta_m[:, -5:], axis=0)

        pitimes = np.zeros((24, 5))
        ref_pitimes_indices = [[23, 0], [14, 0], [17, 4], [16, 4], [15, 4]]

        strengths = transition_strengths[:, -5:]
        for row_idx, row in enumerate(strengths):
            # print(row)
            for col_idx, strength in enumerate(row):
                # print(strength)
                if np.absolute(delta_m[row_idx, col_idx]) <= 2:
                    pitimes[row_idx, col_idx] = self.ref_pitimes[
                        2 + int(delta_m[row_idx, col_idx])] * strengths[
                            ref_pitimes_indices[2 + int(delta_m[row_idx,
                                                                col_idx])][0],
                            ref_pitimes_indices[2 + int(delta_m[
                                row_idx, col_idx])][1]] / strengths[row_idx,
                                                                    col_idx]
                else:
                    pitimes[row_idx, col_idx] = np.nan

        # need delta_m values if we are to get accurate times
        if table_size == 'F2':
            transition_strengths = transition_strengths[:, -5:]
            pitimes = pitimes[:, -5:]
            self.transition_strengths = transition_strengths
            self.transition_pitimes = pitimes
        else:
            self.transition_strengths = transition_strengths
            self.transition_pitimes = pitimes

        np.savetxt('quick_reference_arrays/transition_strengths.txt',
                   transition_strengths,
                   delimiter=',')
        np.savetxt('quick_reference_arrays/transition_pitimes.txt',
                   pitimes,
                   delimiter=',')

        return transition_strengths, pitimes

    def generate_frequencies_strengths_table(self):

        if hasattr(self, 'transition_strengths'):
            strengths = self.transition_strengths[:, -5:]
        else:
            strengths, pitimes = self.generate_transition_strengths(
                xy_plane_angle=58.0,
                wavevector_angle=45.0,
                table_size='F2',
            )

        if hasattr(self, 'transition_frequencies'):
            frequencies = self.transition_frequencies
        else:
            frequencies = self.generate_transition_frequencies()

        # create a new figure
        fig, ax = plt.subplots(figsize=(8, 6))

        im = plt.imshow(strengths, aspect=0.25)
        plt.colorbar(im)

        plt.xticks(np.arange(strengths.shape[1]), self.plotting_names_S12[-5:])
        plt.yticks(np.arange(strengths.shape[0]),
                   np.flip(self.plotting_names_D52))

        # loop over data dimensions and create text annotations
        strengths = np.round(strengths, 4)
        frequencies = np.round(frequencies, 3)
        print(strengths)

        # generate the delta_m value for each transition
        self.generate_delta_m_table()
        delta_m = np.flip(self.delta_m[:, -5:], axis=0)

        for i in range(strengths.shape[0]):
            for j in range(strengths.shape[1]):
                if np.absolute(delta_m[i, j]) <= 2:
                    plt.text(j,
                             i,
                             f"Strength: {strengths[i,j]}" +
                             f"\n $f_r$: {frequencies[i,j]}",
                             fontsize=8,
                             ha="center",
                             va="center",
                             color="black"
                             if np.mean(strengths[i, j]) > 0.15 else "white")

        # set the title
        plt.title("Relative Transition Strengths (arb. units)")

    def generate_frequencies_pitimes_table(self, gen_plot_table=False):

        if hasattr(self, 'transition_pitimes'):
            pitimes = self.transition_pitimes[:, -5:]
        else:
            _, pitimes = self.generate_transition_strengths(
                xy_plane_angle=58.0,
                wavevector_angle=45.0,
                table_size='F2',
            )

        self.generate_delta_m_table()
        delta_m = np.flip(self.delta_m[:, -5:], axis=0)

        # pitimes = np.zeros((24, 5))
        # ref_pitimes_indices = [[23, 0], [14, 0], [17, 4], [16, 4], [15, 4]]
        # for row_idx, row in enumerate(strengths):
        #     # print(row)
        #     for col_idx, strength in enumerate(row):
        #         # print(strength)
        #         if np.absolute(delta_m[row_idx, col_idx]) <= 2:
        #             pitimes[row_idx, col_idx] = self.ref_pitimes[
        #                 2 + int(delta_m[row_idx, col_idx])] * strengths[
        #                     ref_pitimes_indices[2 + int(delta_m[row_idx,
        #                                                          col_idx])][0],
        #                     ref_pitimes_indices[2 + int(delta_m[
        #                         row_idx, col_idx])][1]] / strengths[row_idx,
        #                                                             col_idx]
        #             # USE THIS IF YOU WANT TO CALIBRATE OFF 1 PI-TIME ONLY
        #             # pitimes[row_idx, col_idx] = self.ref_pitimes[
        #             #     2] * strengths[
        #             #         ref_pitimes_indices[2][0],
        #             #         ref_pitimes_indices[2][
        #             #                    1]] / strengths[row_idx,
        #             #                                 col_idx]
        #         else:
        #             pitimes[row_idx, col_idx] = np.nan

        if hasattr(self, 'transition_frequencies'):
            frequencies = self.transition_frequencies
        else:
            frequencies = self.generate_transition_frequencies()

        if gen_plot_table:
            fig, ax = plt.subplots(figsize=(8, 6))

            norm = colors.LogNorm(np.nanmin(pitimes),
                                  np.nanmax(pitimes),
                                  clip=True)
            im = plt.imshow(pitimes, aspect=0.25, norm=norm, cmap='viridis_r')
            # create a new figure
            plt.colorbar(im)

            plt.xticks(np.arange(pitimes.shape[1]),
                       self.plotting_names_S12[-5:])
            plt.yticks(np.arange(pitimes.shape[0]),
                       np.flip(self.plotting_names_D52))

            # loop over data dimensions and create text annotations
            pitimes = np.round(pitimes, 4)
            frequencies = np.round(frequencies, 3)

            # generate the delta_m value for each transition
            self.generate_delta_m_table()
            delta_m = np.flip(self.delta_m[:, -5:], axis=0)

            for i in range(pitimes.shape[0]):
                for j in range(pitimes.shape[1]):
                    if np.absolute(delta_m[i, j]) <= 2:
                        plt.text(j,
                                 i,
                                 rf"$f_r$: {frequencies[i,j]}" + "\n"
                                 rf"$t_\pi$: {np.round(pitimes[i,j], 2)}",
                                 fontsize=8,
                                 ha="center",
                                 va="center",
                                 color="black"
                                 if np.mean(pitimes[i, j]) < 900 else "white")

            # set the title
            plt.title(r"$t_\pi$-times / $\mu$s")
        return pitimes

    def generate_sensitivities_pitimes_table(self, gen_plot_table=False):

        if hasattr(self, 'transition_sensitivities'):
            sensitivities = self.transition_sensitivities
        else:
            sensitivities = self.generate_transition_sensitivities()

        if hasattr(self, 'transition_pitimes'):
            pitimes = self.transition_pitimes
        else:
            pitimes = self.generate_frequencies_pitimes_table()

        combined_kappa_tau = np.multiply(sensitivities, pitimes)

        kappa_tau_squared = np.square(combined_kappa_tau)
        kappa_tau_squared = -1.0 * np.log10(kappa_tau_squared)
        self.kappa_tau_squared = kappa_tau_squared
        kappa_tau_squared[5, 2] = -2

        if gen_plot_table:
            # create a new figure
            fig, ax = plt.subplots(figsize=(8, 6))

            im = plt.imshow(kappa_tau_squared, aspect=0.25)
            plt.colorbar(im)

            plt.xticks(np.arange(kappa_tau_squared.shape[1]),
                       self.plotting_names_S12[-5:])
            plt.yticks(np.arange(kappa_tau_squared.shape[0]),
                       np.flip(self.plotting_names_D52))

            # loop over data dimensions and create text annotations
            sensitivities = np.round(sensitivities, 4)
            pitimes = np.round(pitimes, 4)

            # generate the delta_m value for each transition
            self.generate_delta_m_table()
            delta_m = np.flip(self.delta_m[:, -5:], axis=0)

            for i in range(pitimes.shape[0]):
                for j in range(pitimes.shape[1]):
                    if np.absolute(delta_m[i, j]) <= 2:
                        plt.text(j,
                                 i,
                                 rf"$\Delta_B$: {sensitivities[i,j]}" + "\n" +
                                 rf"$\pi$-time: {pitimes[i,j]}",
                                 fontsize=8,
                                 ha="center",
                                 va="center",
                                 color="black" if np.mean(
                                     kappa_tau_squared[i,
                                                       j]) > -4.5 else "white")

            # set the title
            plt.title(r"Combined Sensitivities and $\pi$-time ($-log(1/G^2)$)")
        else:
            pass
        return self.kappa_tau_squared

    def fit_geometric_angles(self,
                             guess_xy_plane: float = 58.0,
                             guess_wavevector: float = 45.0):

        raw_strengths = self.generate_raw_transition_strengths(
            table_size='full')

        data = np.loadtxt(self.path_to_lab_data + self.folder_name +
                          self.file_name_stem + self.input_file_datetime +
                          '.txt',
                          delimiter=',')
        measured_pitimes = data[:, 1]
        measured_pitimes_table = np.nan((24, 5))
        for it, pair in enumerate(self.measured_transition_order):
            measured_pitimes_table[pair[0]][pair[1]] = measured_pitimes[it]
        print('tabulated measured pi times \n', measured_pitimes_table)

        # define the helper function that will output the angles needed
        def pitime_comparer(params):

            xy_plane_angle, wavevector_angle = params

            geom_factor_array = self.generate_geometric_factors(
                xy_plane_angle=xy_plane_angle,
                wavevector_angle=wavevector_angle)
            strengths = geom_factor_array * raw_strengths

            transition_strengths = np.flip(np.absolute(strengths), axis=0)
            transition_strengths = np.round(
                transition_strengths * (0.2676 / 0.32779), 4)
            transition_strengths[:, [0, 2]] = transition_strengths[:, [2, 0]]
            conversion = 29.7 * transition_strengths[16, 7]
            pitimes_array = np.round(conversion / transition_strengths, 2)
            pitimes_array = pitimes_array[:, -5:]
            # print('params \n', params)
            # print('new pi times array \n', pitimes_array)

            differences = np.zeros(len(measured_pitimes))
            for it, pair in enumerate(self.measured_transition_order):
                if measured_pitimes[it] < 100:
                    differences[it] = np.absolute(
                        measured_pitimes[it] - pitimes_array[pair[0]][pair[1]])
                else:
                    pass

            return np.sum(differences**2)

        initial_guess = np.array([guess_xy_plane, guess_wavevector])
        optimisation_params = sc.optimize.minimize(pitime_comparer,
                                                   initial_guess,
                                                   method='Nelder-Mead')

        found_xy_plane_angle, found_wavevector_angle = optimisation_params.x

        return found_xy_plane_angle, found_wavevector_angle

    def generate_measured_transition_order(self, print_matrix=False):

        def find_closest(given_number, number_list):
            """Find the closest number in the list to the given number."""
            closest_number = min(number_list,
                                 key=lambda x: abs(x - given_number))
            return closest_number

        def replace_with_closest(matrix, number_list):
            """Replace each element in the matrix
            with the closest number from the list or 0."""
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if np.isnan(matrix[i, j]):
                        matrix[i, j] = 0
                        continue  # Skip np.nan values

                    closest_number = find_closest(matrix[i, j], number_list)
                    if abs(matrix[i, j] - closest_number) > 0.05:
                        # Set to 0 if difference is more than 0.5
                        matrix[i, j] = 0
                    else:
                        matrix[i, j] = closest_number
            return matrix

        data = np.loadtxt(self.path_to_lab_data + self.folder_name +
                          self.file_name_stem + self.input_file_datetime +
                          '.txt',
                          delimiter=',')
        measured_freqs = data[:, 0]

        self.generate_transition_frequencies()

        new_matrix = replace_with_closest(
            np.array(self.transition_frequencies), measured_freqs)

        if print_matrix:
            print(new_matrix)

        measured_freqs_positions = []
        for freq in measured_freqs:
            measured_freqs_positions.append([
                np.where(new_matrix == freq)[0][0],
                np.where(new_matrix == freq)[1][0]
            ])
        self.measured_transition_order = np.array(measured_freqs_positions)
        return new_matrix

    def generate_init_lookup_table(self,
                                   init_indices: list = [
                                       0, 1, 2, 3, 4, 5, 6, 20, 25, 27
                                   ]):

        data = np.loadtxt(self.path_to_lab_data + self.folder_name +
                          self.file_name_stem + self.input_file_datetime +
                          '.txt',
                          delimiter=',')
        measured_pitimes = data[:, 1]

        # Need to load in all the attributes needed to populate the table
        self.generate_measured_transition_order()
        self.generate_transition_frequencies()
        self.generate_transition_strengths()
        self.generate_transition_sensitivities()
        self.generate_sensitivities_pitimes_table(gen_plot_table=False)
        all_sidebands = self.generate_transition_frequency_sidebands(
            gen_plot=False)

        # Ready to generate the lookup table now
        init_keys = [
            'Frequency (MHz)', r'Meas. $\pi$-time ($\mu s$)',
            r'Init. $S_{1/2}, F=2$', r'$\Delta_B$ (MHz/G)',
            r'$\kappa^2\tau^2$ (arb.)', 'Near SB Freq. (MHz)',
            'SB Spacing(kHz)', r'SB Init. $S_{1/2}, F=2$', 'SB Strength (arb.)'
        ]

        s_names = [
            r'$m_F=-2$', r'$m_F=-1$', r'$m_F=0$', r'$m_F=1$', r'$m_F=2$'
        ]
        init_transitions = self.measured_transition_order[init_indices]

        df_init = pd.DataFrame(columns=init_keys)
        # filling the table with just the 24 init transitions
        for idx, pair in enumerate(init_transitions):
            df_init.at[idx, init_keys[0]] = np.round(
                self.transition_frequencies[pair[0], pair[1]], 4)
            df_init.at[idx, init_keys[1]] = np.round(
                measured_pitimes[init_indices[idx]], 2)
            df_init.at[idx, init_keys[2]] = s_names[pair[1]]
            df_init.at[idx, init_keys[3]] = np.round(
                self.transition_sensitivities[pair[0], pair[1]], 4)
            df_init.at[idx, init_keys[4]] = np.round(
                self.kappa_tau_squared[pair[0], pair[1]], 4)

            # Need some logic for the sideband spacings and strengths
            freq = self.transition_frequencies[pair[0], pair[1]]
            extended_freq = np.array([[freq] * all_sidebands.shape[1]] *
                                     all_sidebands.shape[0])
            comp_array = np.absolute(extended_freq - all_sidebands)
            non_zero_indices = np.argwhere(comp_array != 0)
            min_index = np.argmin(comp_array[non_zero_indices[:, 0],
                                             non_zero_indices[:, 1]])
            row_index, col_index = non_zero_indices[min_index]
            sb_tuple = np.argwhere(
                self.transition_frequencies == all_sidebands[0, col_index])[0]

            nearest_sb_freq = np.round(
                self.transition_frequencies[sb_tuple[0], sb_tuple[1]], 4)
            nearest_sb_spacing = np.round(
                1e3 * comp_array[row_index, col_index], 2)
            nearest_sb_strength = np.round(
                self.transition_strengths[sb_tuple[0], sb_tuple[1]], 4)

            df_init.at[idx, init_keys[5]] = nearest_sb_freq
            df_init.at[idx, init_keys[6]] = nearest_sb_spacing
            df_init.at[idx, init_keys[7]] = s_names[sb_tuple[1]]
            df_init.at[idx, init_keys[8]] = nearest_sb_strength

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')
        table = ax.table(cellText=df_init.values,
                         colLabels=df_init.columns,
                         cellLoc='center',
                         colLoc='center',
                         loc='center',
                         cellColours=[['lightgray'] * len(df_init.columns)] *
                         len(df_init),
                         colColours=['gray'] * len(df_init.columns))
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        return df_init

    def generate_SPAM_lookup_table(self):

        data = np.loadtxt(self.path_to_lab_data + self.folder_name +
                          self.file_name_stem + self.input_file_datetime +
                          '.txt',
                          delimiter=',')
        measured_pitimes = data[:, 1]

        # Need to load in all the attributes needed to populate the table
        self.generate_measured_transition_order()
        self.generate_transition_frequencies()
        self.generate_transition_strengths()
        self.generate_transition_sensitivities()
        self.generate_sensitivities_pitimes_table(gen_plot_table=False)
        all_sidebands = self.generate_transition_frequency_sidebands(
            gen_plot=False)

        # Ready to generate the lookup table now
        SPAM_keys = [
            'Ket #', 'Frequency (MHz)', r'Meas. $\pi$-time ($\mu s$)',
            r'Init. $S_{1/2}, F=2$', r'$\Delta_B$ (MHz/G)',
            r'$\kappa^2\tau^2$ (arb.)', 'Near SB Freq. (MHz)',
            'SB Spacing(kHz)', r'SB Init. $S_{1/2}, F=2$', 'SB Strength (arb.)'
        ]
        s_names = [
            r'$m_F=-2$', r'$m_F=-1$', r'$m_F=0$', r'$m_F=1$', r'$m_F=2$'
        ]

        df_SPAM = pd.DataFrame(columns=SPAM_keys)
        # filling the table with just the 24 SPAM transitions
        for idx, pair in enumerate(self.measured_transition_order[13:37]):
            df_SPAM.at[idx, SPAM_keys[0]] = rf'$|{idx+1}\rangle$'
            df_SPAM.at[idx, SPAM_keys[1]] = np.round(
                self.transition_frequencies[pair[0], pair[1]], 4)
            df_SPAM.at[idx,
                       SPAM_keys[2]] = np.round(measured_pitimes[idx + 13], 2)
            df_SPAM.at[idx, SPAM_keys[3]] = s_names[pair[1]]
            df_SPAM.at[idx, SPAM_keys[4]] = np.round(
                self.transition_sensitivities[pair[0], pair[1]], 4)
            df_SPAM.at[idx, SPAM_keys[5]] = np.round(
                self.kappa_tau_squared[pair[0], pair[1]], 4)

            # Need some logic for the sideband spacings and strengths
            freq = self.transition_frequencies[pair[0], pair[1]]
            extended_freq = np.array([[freq] * all_sidebands.shape[1]] *
                                     all_sidebands.shape[0])
            comp_array = np.absolute(extended_freq - all_sidebands)
            non_zero_indices = np.argwhere(comp_array != 0)
            min_index = np.argmin(comp_array[non_zero_indices[:, 0],
                                             non_zero_indices[:, 1]])
            row_index, col_index = non_zero_indices[min_index]
            sb_tuple = np.argwhere(
                self.transition_frequencies == all_sidebands[0, col_index])[0]

            nearest_sb_freq = np.round(
                self.transition_frequencies[sb_tuple[0], sb_tuple[1]], 4)
            nearest_sb_spacing = np.round(
                1e3 * comp_array[row_index, col_index], 2)
            nearest_sb_strength = np.round(
                self.transition_strengths[sb_tuple[0], sb_tuple[1]], 4)

            df_SPAM.at[idx, SPAM_keys[6]] = nearest_sb_freq
            df_SPAM.at[idx, SPAM_keys[7]] = nearest_sb_spacing
            df_SPAM.at[idx, SPAM_keys[8]] = s_names[sb_tuple[1]]
            df_SPAM.at[idx, SPAM_keys[9]] = nearest_sb_strength

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')
        table = ax.table(cellText=df_SPAM.values,
                         colLabels=df_SPAM.columns,
                         cellLoc='center',
                         colLoc='center',
                         loc='center',
                         cellColours=[['lightgray'] * len(df_SPAM.columns)] *
                         len(df_SPAM),
                         colColours=['gray'] * len(df_SPAM.columns))
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        return df_SPAM

    def fit_mag_field_using_S12(self, B_field_guess=4.216):

        # get the measured data for comparison for fitting
        data = np.loadtxt(self.path_to_lab_data + self.folder_name +
                          self.file_name_stem + self.input_file_datetime +
                          '.txt',
                          delimiter=',')
        measured_frequencies = data[:, 0]

        self.generate_measured_transition_order()
        fitted_freqs_table = np.zeros((24, 5))

        for idx, freq in enumerate(measured_frequencies):
            fitted_freqs_table[self.measured_transition_order[idx][0],
                               self.measured_transition_order[idx]
                               [1]] = measured_frequencies[idx]

        measured_spacings = []
        for idx in range(4):
            spacer = []
            for row in range(24):
                if fitted_freqs_table[row][idx] != 0 and fitted_freqs_table[
                        row][idx + 1] != 0:
                    spacer.append(fitted_freqs_table[row][idx + 1] -
                                  fitted_freqs_table[row][idx])
            measured_spacings.append(np.mean(spacer))
        print('Measured S1/2 level spacings:', measured_spacings)

        def calculate_average_spacings(B_field=4, print_spacing=False):
            """
            Calculate the average spacing between non-0 elements for each row.
            """

            self.B_field = B_field
            self.generate_transition_frequencies()

            average_spacings = []

            for it, row in enumerate(self.transition_frequencies):

                # Indices of non-zero (valid) entries
                valid_indices = np.where(row != np.nan)[0]
                if len(valid_indices) < 2:
                    continue

                # Calculate spacings between adjacent valid entries
                spacings = np.diff(row)
                spacings[abs(spacings) > 100] = 0
                average_spacings.append(spacings)
                if it == 1:
                    spacer = spacings

            if print_spacing:
                print(self.B_field, spacer)
            return average_spacings[1]

        def fit_helper(B_field_array: list):

            B_field = B_field_array[
                0]  # Extract the B_field value from the array

            calculated_spacings = calculate_average_spacings(
                B_field=B_field, print_spacing=True)
            return np.sum(np.absolute(measured_spacings - calculated_spacings))

        fit_helper([B_field_guess])

        print('Starting preliminary B field fit using S1/2 spacings...')
        optimisation_params = sc.optimize.minimize(
            fit_helper,
            [B_field_guess],
            method='Nelder-Mead',
            # options={'maxiter': 10},
            bounds=[(4.1, 4.3)])
        found_B_field = optimisation_params.x
        self.B_field = found_B_field[0]
        print(f'\n The B field found is: {self.B_field} Gauss. \n')
        print('Measured S1/2 level spacings:', measured_spacings)

        fitted_spacings = calculate_average_spacings(found_B_field[0])
        print('Fitted S1/2 spacings:', fitted_spacings)

        return found_B_field[0]

    def fit_Hamiltonian(self,
                        B_guess: float = 4.2,
                        mdS12_guess: float = 4018.8708338,
                        mdD52_guess: float = -12.0297241,
                        eqD52_guess: float = 59.519566,
                        eoD52_guess: float = -0.00004173,
                        compare_to_measurement=False):
        """delta_F2MN1f4mn3_F2MN1f1m1 is the splitting between the most
        positively and negatively B field sensitive transitions.

        initial_guess is the first guess for B field.

        """
        guess = [B_guess, mdS12_guess, mdD52_guess, eqD52_guess, eoD52_guess]

        data = np.loadtxt(self.path_to_lab_data + self.folder_name +
                          self.file_name_stem + self.input_file_datetime +
                          '.txt',
                          delimiter=',')
        measured_freqs = data[:, 0]

        # get the measured transition order
        if hasattr(self, 'measured_transition_order'):
            pass
        else:
            self.generate_measured_transition_order()

        # define the helper function that will find the splitting each time
        def fit_helper(params):
            print('params', params)
            (B_field, mag_dip_S12, mag_dip_D52, elec_quad_D52,
             mag_octu_D52) = params
            self.B_field = B_field
            self.magnetic_dipole_S12 = mag_dip_S12
            self.magnetic_dipole_D52 = mag_dip_D52
            self.electric_quadrupole_D52 = elec_quad_D52
            self.magnetic_octupole_D52 = mag_octu_D52

            transition_freq_matrix = self.generate_transition_frequencies()

            diff_helper = sum([(
                transition_freq_matrix[self.measured_transition_order[it][0],
                                       self.measured_transition_order[it][1]] -
                measured_freqs[it])**2 for it in range(len(measured_freqs))])

            return diff_helper

        print('Running Hamiltonian parameter fit...')
        optimisation_params = sc.optimize.minimize(
            fit_helper,
            guess,
            method='BFGS',
            options={'maxiter': 1000},
            # bounds=[(3, 8)]
        )

        self.B_field = optimisation_params.x[0]
        self.magnetic_dipole_S12 = optimisation_params.x[1]
        self.magnetic_dipole_D52 = optimisation_params.x[2]
        self.electric_quadrupole_D52 = optimisation_params.x[3]
        self.magnetic_octupole_D52 = optimisation_params.x[4]

        if compare_to_measurement:
            self.generate_sim_meas_comparison_table()

        return optimisation_params

    def find_perfect_transition_pair(self,
                                     initial_guess=4.216,
                                     state_0=[0, 2, 2],
                                     state_1=[0, 3, 2]):

        D52_F_vals = self.F_values_D52[:, 0]
        D52_m_vals = self.F_values_D52[:, 1]

        state_0_row = np.intersect1d(np.where(D52_F_vals == int(state_0[1])),
                                     np.where(D52_m_vals == int(state_0[2])))
        state_1_row = np.intersect1d(np.where(D52_F_vals == int(state_1[1])),
                                     np.where(D52_m_vals == int(state_1[2])))

        state_0_col = state_0[0] + 2
        state_1_col = state_1[0] + 2

        # define the helper function that will find the splitting each time
        def sensitivity_difference(B_field_array: list):

            B_field = B_field_array[
                0]  # Extract the B_field value from the array
            self.B_field = B_field

            transition_sens_matrix = self.generate_transition_sensitivities()

            calculated_splitting = transition_sens_matrix[
                23 - state_0_row,
                state_0_col] - transition_sens_matrix[23 - state_1_row,
                                                      state_1_col]

            return np.absolute(calculated_splitting)

        # Define the callback function
        def callback(xk):
            print(f"Current guess: {xk}")
            print("Sensitivity difference:" +
                  f" {sensitivity_difference([self.B_field])}")

        print('Starting fitter for field-insensitive transition...')

        # add this to limit number of iterations: options={'maxiter': 10},
        optimisation_params = sc.optimize.minimize(sensitivity_difference,
                                                   [initial_guess],
                                                   method='Nelder-Mead',
                                                   callback=callback,
                                                   bounds=[(2, 4.5)])
        found_B_field = optimisation_params.x
        self.B_field = found_B_field[0]
        print(f'\n The B field found is: {self.B_field} Gauss. \n')

        self.generate_transition_frequencies()
        # print(self.transition_frequencies)

        return self.B_field

    def generate_repump_map(self, verbose=False, horizontal_graph=False):

        self.B_field_range = np.linspace(0.001, self.B_field, 10)

        # generate the dataframes for S12 and D52 in IJ basis
        df_S12 = self.dataframe_evals_estates(orbital='S12', basis='IJ')
        df_D52 = self.dataframe_evals_estates(orbital='D52', basis='IJ')

        # let's get the P3/2 states in the IJ basis, just for the 7 F=3 states
        # there are 16 total P3/2 states, so let's start from there
        # F_mat_P32 = np.eye(16)

        IJ_mat_P32 = np.zeros((16, 16))
        for i, row in enumerate(IJ_mat_P32):
            for j, val in enumerate(row):
                entry = self.clebsch(self.nuc_I, self.IJ_values_P32[j][0],
                                     self.J32, self.IJ_values_P32[j][1],
                                     self.F_values_P32[i][0],
                                     self.F_values_P32[i][1])

                IJ_mat_P32[i, j] = entry

        trans = np.zeros((24, 16))
        for it, D52_state in enumerate(self.F_values_D52):
            for it2, P32_state in enumerate(self.F_values_P32):
                if np.abs(P32_state[1] - D52_state[1]) == 1:
                    strength = 0
                    for idx_D52, evec_mag_D52 in enumerate(
                            df_D52[str(D52_state) + ' estate'].iloc[-1]):
                        for idx_P32, evec_mag_P32 in enumerate(
                                IJ_mat_P32[it2]):
                            if (self.IJ_values_D52[idx_D52][0] -
                                    self.IJ_values_P32[idx_P32][0]) == 0:
                                qJ_dummy = self.IJ_values_D52[idx_D52][
                                    1] - self.IJ_values_P32[idx_P32][1]
                                cg_coeff_dummy = self.clebsch(
                                    1.5, self.IJ_values_P32[idx_P32][1], 1.0,
                                    qJ_dummy, 2.5,
                                    self.IJ_values_D52[idx_D52][1])
                                strength += (evec_mag_P32 * evec_mag_D52 *
                                             cg_coeff_dummy)
                            else:
                                strength += 0
                    # square the matrix element to get branching ratio
                    trans[it, it2] = strength**2

                elif np.abs(P32_state[1] - D52_state[1]) > 1:
                    trans[it, it2] = 0

        # take just the F=3 level of P3/2, ie. first 7 columns
        trans = trans[:, :7]
        # print(np.round(trans,3))

        DtoP_transition_strengths = np.flip(
            trans / np.nansum(trans, axis=1, keepdims=True), axis=0)
        # print(np.round(DtoP_transition_strengths,3))

        # now the P to S matrix
        trans = np.zeros((8, 16))
        for it, S12_state in enumerate(self.F_values_S12):
            for it2, P32_state in enumerate(self.F_values_P32):
                if np.abs(P32_state[1] - S12_state[1]) <= 1:
                    strength = 0
                    for idx_S12, evec_mag_S12 in enumerate(
                            df_S12[str(S12_state) + ' estate'].iloc[-1]):
                        for idx_P32, evec_mag_P32 in enumerate(
                                IJ_mat_P32[it2]):
                            if (self.IJ_values_S12[idx_S12][0] -
                                    self.IJ_values_P32[idx_P32][0]) == 0:
                                qJ_dummy = self.IJ_values_S12[idx_S12][
                                    1] - self.IJ_values_P32[idx_P32][1]
                                cg_coeff_dummy = self.clebsch(
                                    1.5, self.IJ_values_P32[idx_P32][1], 1.0,
                                    qJ_dummy, 0.5,
                                    self.IJ_values_S12[idx_S12][1])
                                strength += (evec_mag_P32 * evec_mag_S12 *
                                             cg_coeff_dummy)
                            else:
                                strength += 0
                    # square the matrix element to get branching ratio
                    trans[it, it2] = strength**2

                elif np.abs(P32_state[1] - S12_state[1]) > 1:
                    trans[it, it2] = 0

        # take just the F=3 level of P3/2, ie. first 7 columns
        # as well as just the F=2 levels in S1/2 so last 5 rows
        trans = trans[3:, :7]

        # normalise each row to get from strengths to probabilities
        PtoS_transition_strengths = np.transpose(
            np.flip(trans / np.nansum(trans, axis=0, keepdims=True)))

        # finally we make the combined plot for D to S repumping
        repump_map = np.dot(DtoP_transition_strengths,
                            PtoS_transition_strengths)

        # make entries nan's for plotting with imshow
        DtoP_transition_strengths[DtoP_transition_strengths == 0] = np.nan
        PtoS_transition_strengths[PtoS_transition_strengths == 0] = np.nan
        repump_map[repump_map == 0] = np.nan

        if verbose:
            plt.figure(figsize=(8, 6))
            cax = plt.imshow(DtoP_transition_strengths,
                             aspect='auto',
                             cmap='viridis',
                             interpolation='nearest')

            # Configure the axes and labels
            plt.colorbar(cax, label='Probability')
            # plt.xticks(ticks=np.arange(len(F_values_P32_labels)),
            #            labels=F_values_P32_labels)
            # plt.yticks(ticks=np.arange(len(F_values_D52)),
            #            labels=F_values_D52)

            plt.xticks(np.arange(DtoP_transition_strengths.shape[1]),
                       self.plotting_names_P32[:7])
            plt.yticks(np.arange(DtoP_transition_strengths.shape[0]),
                       np.flip(self.plotting_names_D52))

            plt.xlabel(r'$P_{3/2}, F=3$')
            plt.ylabel(r'$D_{5/2}$')
            plt.title('Transition Probability for $5D_{5/2}$ to $6P_{3/2}$')

            for i in range(DtoP_transition_strengths.shape[0]):
                for j in range(DtoP_transition_strengths.shape[1]):
                    if DtoP_transition_strengths[i, j] != np.nan:
                        plt.text(
                            j,
                            i,
                            f"{np.round(DtoP_transition_strengths[i,j], 4)}",
                            fontsize=8,
                            ha="center",
                            va="center",
                            color="black"
                            if DtoP_transition_strengths[i,
                                                         j] > 0.5 else "white")

            # now the P to S figure
            plt.figure(figsize=set_size(width='half'))
            cax = plt.imshow(PtoS_transition_strengths,
                             aspect='auto',
                             cmap='viridis',
                             interpolation='nearest')

            # Configure the axes and labels
            plt.colorbar(cax, label='Probability')
            plt.xticks(np.arange(PtoS_transition_strengths.shape[1]),
                       self.plotting_names_S12_m_only[3:])
            plt.yticks(np.arange(PtoS_transition_strengths.shape[0]),
                       self.plotting_names_P32[:7])

            plt.xlabel(r'$S_{1/2}, F=2$')
            plt.ylabel(r'$P_{3/2}, F=3$')
            plt.title('Transition Probability for $6P_{3/2}$ to $6S_{1/2}$')

            for i in range(PtoS_transition_strengths.shape[0]):
                for j in range(PtoS_transition_strengths.shape[1]):
                    if PtoS_transition_strengths[i, j] != np.nan:
                        plt.text(
                            j,
                            i,
                            f"{np.round(PtoS_transition_strengths[i,j], 4)}",
                            fontsize=8,
                            ha="center",
                            va="center",
                            color="black"
                            if PtoS_transition_strengths[i,
                                                         j] > 0.5 else "white")

            # now the full repump map
            plt.figure(figsize=(8, 10))
            cax = plt.imshow(repump_map,
                             aspect='auto',
                             cmap='viridis',
                             interpolation='nearest')

            # Configure the axes and labels
            plt.colorbar(cax, label='Probability')
            plt.xticks(np.arange(PtoS_transition_strengths.shape[1]),
                       self.plotting_names_S12[3:])
            plt.yticks(np.arange(DtoP_transition_strengths.shape[0]),
                       np.flip(self.plotting_names_D52))

            plt.title(
                'Repumping Map for $5D_{5/2}$ via $6P_{3/2}$ to $6S_{1/2}$')

            for i in range(repump_map.shape[0]):
                for j in range(repump_map.shape[1]):
                    if repump_map[i, j] != np.nan:
                        plt.text(
                            j,
                            i,
                            f"{np.round(repump_map[i,j], 4)}",
                            fontsize=8,
                            ha="center",
                            va="center",
                            color="black" if repump_map[i,
                                                        j] > 0.5 else "white")

        if horizontal_graph:

            repump_horiz = np.transpose(repump_map)

            # now the full repump map
            plt.figure(figsize=(12, 3))
            cax = plt.imshow(repump_horiz,
                             aspect='auto',
                             cmap='viridis',
                             interpolation='nearest')

            # Configure the axes and labels
            plt.colorbar(cax, label='Probability')
            plt.yticks(np.arange(PtoS_transition_strengths.shape[1]),
                       self.plotting_names_S12[3:])
            plt.xticks(np.arange(DtoP_transition_strengths.shape[0]),
                       np.flip(self.plotting_names_D52),
                       rotation=70)

            plt.title(
                'Repumping Map for $5D_{5/2}$ via $6P_{3/2}, F=3$ to $6S_{1/2}$'
            )

            # for i in range(repump_horiz.shape[0]):
            #     for j in range(repump_horiz.shape[1]):
            #         if repump_horiz[i, j] != np.nan:
            #             plt.text(
            #                 j,
            #                 i,
            #                 f"{np.round(repump_horiz[i,j], 4)}",
            #                 fontsize=8,
            #                 ha="center",
            #                 va="center",
            #                 color="black" if repump_horiz[i,
            #                                             j] > 0.5 else "white")
            plt.tight_layout()
        return repump_map

    def generate_sorted_kappa_tau_squared_list(self):

        if hasattr(self, 'transition_sensitivities'):
            sensitivities = self.transition_sensitivities
        else:
            sensitivities = self.generate_transition_sensitivities()
        if hasattr(self, 'transition_pitimes'):
            pitimes = self.transition_pitimes()
        else:
            strengths, pitimes = self.generate_transition_strengths()

        transitions = []
        kappa_taus = []
        taus_squared = []
        kappas = []
        taus = []
        for (row, col), pitime in np.ndenumerate(pitimes):
            if not np.isnan(pitime):
                # print(row,col)
                # print(pitime)
                # print(sensitivities[row,col])
                # print((pitime*sensitivities[row,col])**2)
                transitions.append([
                    int(col - 2), self.F_values_D52[int(23 - row)][0],
                    self.F_values_D52[int(23 - row)][1]
                ])
                kappa_taus.append((pitime * sensitivities[row, col])**2)
                taus_squared.append(pitime**2)
                taus.append(pitime)
                kappas.append(sensitivities[row, col])

        combined = list(
            zip(transitions, kappa_taus, taus_squared, taus, kappas))
        sorted_combined = sorted(combined, key=lambda x: x[1])

        # for itera in sorted_combined:
        #     print(itera)
        kappa_tau_list = sorted_combined

        return kappa_tau_list

    def find_B_field_020_insensitivity(self):

        B_fields = []
        sensies = []
        for bf in np.linspace(4.1, 4.3, 30):
            ba.B_field = bf
            ba.generate_transition_sensitivities()
            ba.generate_transition_frequencies()
            print(bf)
            print(ba.transition_sensitivities[5, 2])
            B_fields.append(bf)
            sensies.append(ba.transition_sensitivities[5, 2])

        plt.plot(B_fields, sensies)
        plt.xlabel("Magnetic Field / G")
        plt.ylabel("Sensitivity / MHz/G")
        plt.grid()
        plt.title("Sensitivity of [0,2,0] transition.")


if __name__ == '__main__':
    start_time = time.time()
    ba = Barium()
    ba.B_field_range = np.linspace(0.00001, 5, 10)
    B_field_found = 4.216246093750001  # coils at 5A
    B_field_found = 4.20958740234375  # permanent magnets
    ba.B_field = B_field_found
    ba.B_field = 12.3
    # ba.check_lewty()

    # ba.path_to_lab_data = '../data/'
    # ba.folder_name = ('complete_calibrations_data/')
    # ba.file_name_stem = 'New_initialized_calibration_freq_files'

    # ba.input_file_datetime = '_20240314_2352'
    # ba.input_file_datetime = '_20240316_0946'
    # ba.input_file_datetime = '_20240321_1836'
    # ba.input_file_datetime = '_20240324_1440'
    # ba.input_file_datetime = '_20240325_1931' # bad run for calibration!
    # ba.input_file_datetime = '_20240326_1801' # bad run for calibration!
    # ba.input_file_datetime = '_20240328_2031'
    ba.input_file_datetime = '_20240330_1649'

    # B_field_found = ba.fit_mag_field()

    # ba.plot_mag_field(orbital='S12', estates_plot=False)
    # ba.plot_mag_field(orbital='D52', estates_plot=False)
    # ba.generate_measured_transition_order(print_matrix=False)
    # ba.generate_transition_frequency_sidebands(include_sidebands=False,
    #                                            SPAM_freqs_only=False)

    # ba.find_B_field_020_insensitivity()
    # for bf in np.linspace(4, 4.3, 10):

    # for bf in np.linspace(4.2, 4.3, 10):
    #     ba.B_field = bf
    #     ba.generate_transition_sensitivities()
    #     ba.generate_transition_frequencies()
    #     print(bf)
    #     # print(ba.transition_frequencies[23,0])
    #     # print(ba.transition_sensitivities[5,2])
    #     print('623-545', (ba.transition_frequencies[22, 1] -
    #                       ba.transition_frequencies[5, 2]) -
    #           (623.710 - 545.687))
    #     print('244-242', (ba.transition_frequencies[17, 4] -
    #                       ba.transition_frequencies[15, 4]))

    # ba.B_field = 300
    # ba.B_field_range = np.linspace(0.0001, ba.B_field, 100)
    # ba.generate_transition_frequencies(F2M0f2m0_input=545.2490)

    ba.plot_mag_field(orbital='S12', estates_plot=False)
    # ba.plot_mag_field(orbital='D52', estates_plot=False)
    # flatten transition frequencies array to find all
    # differences between all transition frequencies
    # flattened_filtered_trans = ba.transition_frequencies[
    #     ~np.isnan(ba.transition_frequencies)]
    # pairwise_diff = flattened_filtered_trans[:,
    #                                          None] - flattened_filtered_trans

    # mask = np.eye(len(flattened_filtered_trans), dtype=bool)
    # pairwise_diff[mask] = np.inf
    # print(np.min(np.abs(pairwise_diff)))
    # print(np.round(ba.transition_frequencies, 4))

    # transition_strengths, pitimes = ba.generate_transition_strengths()
    # print(np.round(pitimes, 2))
    # ba.generate_frequencies_strengths_table()
    # ba.generate_frequencies_pitimes_table(gen_plot_table=True)
    ba.generate_frequencies_sensitivities_table(Ba138_relative=True)
    # ba.generate_sensitivities_pitimes_table(gen_plot_table=True)
    # ba.generate_sim_meas_comparison_table(do_B_field_fit=True)
    # ba.generate_sorted_kappa_tau_squared_list()

    # ba.generate_SPAM_lookup_table()
    # ba.generate_init_lookup_table()
    # ba.generate_init_SPAM_frequencies_plot()

    # ba.fit_mag_field_using_S12()
    # print('NOW DOING HAMILTONIAN FIT')
    # ba.fit_Hamiltonian(compare_to_measurement=True)
    # ba.generate_sensitivities_pitimes_table(gen_plot_table=True)

    # ba.find_perfect_transition_pair()
    # ba.fit_mag_field(delta=[545.684831,623.48558])

    # ba.generate_all_sensitivities(verbose=True)
    # ba.generate_repump_map(verbose=False, horizontal_graph=True)
    plt.show()
    print("--- %s seconds ---" % (time.time() - start_time))
