import numpy as np
import qutip as qt
import sympy.physics.quantum.cg as sp
from sympy import S

from class_constants import Constants


class Utils(Constants):

    def __init__(self):
        super().__init__()
        self.helpers = np.array([1, 5, 11, 19]) # array to help in conversions

    def clebsch(self, j1, m1, j2, m2, j3, m3):
        """
        Computes the Clebsch-Gordan coefficient <j1 m1 j2 m2 | j3 m3>.

        Parameters:
            j1 (float): First angular momentum quantum number.
            m1 (float): Magnetic quantum number for the first angular momentum.
            j2 (float): Second angular momentum quantum number.
            m2 (float): Magnetic quantum number for the second angular momentum.
            j3 (float): Total angular momentum quantum number.
            m3 (float): Magnetic quantum number for the total angular momentum.

        Returns:
            float: The evaluated Clebsch-Gordan coefficient.
        """
        return sp.CG(S(j1), S(m1), S(j2), S(m2), S(j3), S(m3)).doit().evalf()

    def basis_transform_IJ_to_F(self, orbital='S12'):
        """
        Transforms the basis from IJ representation to F representation.

        Parameters:
            orbital (str): Specifies the orbital type ('S12' or 'D52'). Default is 'S12'.

        Returns:
            Qobj: A Qobj representing the transformation matrix.
        """
        if orbital == 'S12':
            trans = np.empty((8, 8))
            IJ_values = self.IJ_values_S12
            F_values = self.F_values_S12
            for i in range(8):
                for j in range(8):
                    entry = self.clebsch(self.nuc_I, IJ_values[j][0], self.J12,
                                         IJ_values[j][1], F_values[i][0],
                                         F_values[i][1])
                    trans[i, j] = float(entry)

            return qt.Qobj(trans, dims=[[4, 2], [4, 2]])

        elif orbital == 'D52':
            trans = np.empty((24, 24))
            IJ_values = self.IJ_values_D52
            F_values = self.F_values_D52

            for i in range(24):
                for j in range(24):
                    entry = self.clebsch(self.nuc_I, IJ_values[j][0], self.J52,
                                         IJ_values[j][1], F_values[i][0],
                                         F_values[i][1])
                    trans[i, j] = float(entry)

            return qt.Qobj(trans, dims=[[4, 6], [4, 6]])

    def sp_mat(self, spin: float = 0.5, axis: str = 'x'):
        """
        Defines the spin matrices used to build the Hamiltonian.

        Parameters:
            spin (float): The spin of the system (must be half-integer or integer).
            axis (str): The axis of projection ('x', 'y', 'z', or 'eye').

        Returns:
            Qobj: A Qobj representing the spin matrix for the specified axis.

        Raises:
            ValueError: If an invalid axis is provided or if spin is not valid.
        """
        dim = int(2 * spin + 1)
        spin_proj = np.linspace(-spin, spin, int(2 * spin + 1))
        matrix = np.zeros((dim, dim), dtype=complex)

        if axis not in ['x', 'y', 'z', 'eye']:
            raise ValueError(
                "Invalid value for 'axis'. Expected 'x', 'y', 'z', or 'eye'.")

        if spin % 0.5 != 0 or spin < 0.5:
            raise ValueError(
                "The input parameter 'spin' must be integer or half-integer.")

        for i in range(dim):
            for j in range(dim):
                if axis == 'x':
                    if i == j + 1:
                        matrix[i,
                               j] = (1.0 /
                                     2) * np.sqrt(spin * (spin + 1) -
                                                  spin_proj[i] * spin_proj[j])
                    elif i + 1 == j:
                        matrix[i,
                               j] = (1.0 /
                                     2) * np.sqrt(spin * (spin + 1) -
                                                  spin_proj[i] * spin_proj[j])
                    else:
                        pass
                elif axis == 'y':
                    if i == j + 1:
                        matrix[i,
                               j] = (1.0j /
                                     2) * np.sqrt(spin * (spin + 1) -
                                                  spin_proj[i] * spin_proj[j])
                    elif i + 1 == j:
                        matrix[i,
                               j] = (-1.0j /
                                     2) * np.sqrt(spin * (spin + 1) -
                                                  spin_proj[i] * spin_proj[j])
                    else:
                        pass
                elif axis == 'z':
                    if i == j:
                        matrix[i, j] = -spin_proj[i]
                    else:
                        pass
                elif axis == 'eye':
                    matrix[i, j] = 1.0 if i == j else 0.0

        return qt.Qobj(matrix)

    def LandeGj(self, J: float = 0.5, S: float = 0.5, L: float = 0):
        """
        Calculates the Landé g-factor for a given total angular momentum (J),
        spin (S), and orbital angular momentum (L).

        Parameters:
            J (float): Total angular momentum quantum number.
            S (float): Spin quantum number.
            L (float): Orbital angular momentum quantum number.

        Returns:
            float: The calculated Landé g-factor (g_J).
        """
        g_L = 1

        # g_L = 1 - self.mass_electron / (56 *
        # self.mass_proton + 81 * self.mass_neutron)
        g_J = g_L * ((J * (J + 1) - S * (S + 1) + L * (L + 1)) /
                     (2 * J * (J + 1))) + self.g_S * ((J * (J + 1) + S *
                                                       (S + 1) - L * (L + 1)) /
                                                      (2 * J * (J + 1)))
        return g_J

    def LandeGf(self,
                F: float = 2,
                J: float = 0.5,
                S: float = 0.5,
                L: float = 0):
        """
        Calculates the Landé g-factor for a given F state using the total
        angular momentum (J), spin (S), and orbital angular momentum (L).

        Parameters:
            F (float): Total angular momentum state.
            J (float): Total angular momentum quantum number (default is 0.5).
            S (float): Spin quantum number (default is 0.5).
            L (float): Orbital angular momentum quantum number (default is 0).

        Returns:
            float: The calculated Landé g-factor (g_f).
        """
        return self.LandeGj(J=J, S=S, L=L) * ((F * (F + 1) - self.nuc_I *
                                               (self.nuc_I + 1) + J *
                                               (J + 1)) / (2 * F * (F + 1)))

    def K_term(self, F: float = 2, J: float = 0.5):
        """
        Calculates the K term in the context of angular momentum coupling.

        Parameters:
            F (float): Total angular momentum state (default is 2).
            J (float): Total angular momentum quantum number (default is 0.5).

        Returns:
            float: The calculated K term value.
        """
        return F * (F + 1) - J * (J + 1) - self.nuc_I * (self.nuc_I + 1)

    def generate_delta_m_table(self):
        """
        Generates a delta m table based on the defined F values for S12 and D52 states.

        Returns:
            np.ndarray: A 2D array containing the delta m values.
        """
        delta_m = np.zeros((24, 8))
        for it, D52_state in enumerate(self.F_values_D52[:, 1]):
            for it2, S12_state in enumerate(self.F_values_S12[:, 1]):
                delta_m[it, it2] = D52_state - S12_state
        self.delta_m = delta_m
        return delta_m

    @staticmethod
    def find_errors(num_SD, PD, exp_num):
        """Calculate the lower and upper error bounds based on standard deviation.

        Parameters:
            num_SD (float): The number of standard deviations.
            PD (float): The probability of dark state outcomes.
            exp_num (int): The number of experiments.

        Returns:
            tuple: A tuple containing the lower error bound and the upper error bound.

        """
        upper_error = ((PD + (num_SD**2 / (2 * exp_num))) /
                    (1 + (num_SD**2 / exp_num))) + (np.sqrt(
                        ((PD * (1 - PD) * num_SD**2) / exp_num) +
                        (num_SD**4 /
                            (4 * exp_num**2)))) / (1 + (num_SD**2 / exp_num))

        lower_error = ((PD + (num_SD**2 / (2 * exp_num))) /
                    (1 + (num_SD**2 / exp_num))) - (np.sqrt(
                        ((PD * (1 - PD) * num_SD**2) / exp_num) +
                        (num_SD**4 /
                            (4 * exp_num**2)))) / (1 + (num_SD**2 / exp_num))

        return lower_error, upper_error

    def convert_triplet_index(self, item):
        """Convert [m_S, F_D, m_D] type formatted transitions to their table
        indices, handling cases where a list is given or it is given alone. Also
        converts back, again handling lists properly. Converts all outputs to
        NumPy arrays.

        Parameters:
        item: The input item to be converted. This can be:
            - A 1D NumPy array of shape (3,) representing a single triplet.
            - A 2D NumPy array of shape (N, 3) representing a list of triplets.
            - A 1D NumPy array of shape (2,) representing a single table index.
            - A 2D NumPy array of shape (M, 2) representing a list of table indices.

        Returns:
        numpy.ndarray: A NumPy array containing the converted data based on the input
        format. The output structure varies depending on whether a single triplet,
        a list of triplets, a single table index, or a list of table indices was provided.

        """
        # Convert input item to a NumPy array
        try:
            array_item = np.array(item)
        except Exception as e:
            return f"Invalid input format: {e}"

        # Check if item is a single triplet of [m_S,F_D,m_D]
        if array_item.ndim == 1 and array_item.shape[0] == 3 and np.issubdtype(
                array_item.dtype, np.integer):
            return np.array([
                self.helpers[int(array_item[1] - 1)] - int(array_item[2]),
                int(array_item[0] + 2)
            ])

        # Check if item is a list of triplets of [m_S,F_D,m_D]
        elif array_item.ndim == 2 and array_item.shape[
                1] == 3 and np.issubdtype(array_item[0].dtype, np.integer):

            converted_list = []
            for sub_item in array_item:
                converted_list.append([
                    self.helpers[int(sub_item[1] - 1)] - int(sub_item[2]),
                    int(sub_item[0] + 2)
                ])
            return np.array(converted_list)

        # Check if item is a single table index like [row, col]
        elif array_item.ndim == 1 and array_item.shape[
                0] == 2 and np.issubdtype(array_item.dtype, np.integer):

            return np.array([
                int(array_item[1] - 2),
                self.F_values_D52[23 - array_item[0]][0],
                self.F_values_D52[23 - array_item[0]][1],
            ])

        # Check if item is a list of table indices like [row, col]
        elif array_item.ndim == 2 and array_item.shape[
                1] == 2 and np.issubdtype(array_item[0].dtype, np.integer):

            converted_list = []
            for sub_item in array_item:
                converted_list.append([
                    int(sub_item[1] - 2),
                    self.F_values_D52[23 - sub_item[0]][0],
                    self.F_values_D52[23 - sub_item[0]][1],
                ])
            return np.array(converted_list)

if __name__ == '__main__':

    utils = Utils()
    print(utils.LandeGj(J=5 / 2, S=1 / 2, L=2))
    print(utils.LandeGi)
