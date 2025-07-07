import numpy as np


class Constants:

    def __init__(self):

        # Defining fundamental constants
        self.bohr_magneton = 9.274010078e-28  # J/Gauss
        self.planck_constant = 6.62607015e-28  # J/MHz

        # from Hucul 2017, citing Blatt 1982
        self.magnetic_dipole_S12 = 4018.8708338  # MHz
        self.electric_quadrupole_S12 = 0  # MHz
        self.magnetic_octupole_S12 = 0  # MHz

        self.magnetic_dipole_D52 = -12.0297241  # MHz, from Lewty 2013
        self.electric_quadrupole_D52 = 59.519566  # MHz, from Lewty 2013
        self.magnetic_octupole_D52 = -0.00004173  # MHz, from Lewty 2013

        # 'corrected' values from Lewty 2013
        # self.magnetic_dipole_D52 = -12.029234  # MHz, from Lewty 2013
        # self.electric_quadrupole_D52 = 59.52552  # MHz, from Lewty 2013
        # self.magnetic_octupole_D52 = -0.00001241  # MHz, from Lewty 2013

        self.mass_electron = 0.510998950  # MeV/c^2
        self.mass_proton = 938.272088  # MeV/c^2
        self.mass_neutron = 939.565420  # MeV/c^2

        self.g_S = 2.002319  # bare electron g-factor

        # electron g-factors measured for Ba137
        # self.g_S = 2.00285  # from Lewty 2013
        # self.g_S = 2.00186  # from Hoffman 2013

        # self.LandeGi = 0

        # naive estimate for nuclear g-factor
        # self.LandeGi = 5 / 8 * (self.constant.mass_electron /
        # self.constant.mass_proton)

        # measured g-factor values
        self.LandeGi = 0.624867 * (self.mass_electron /
                                   (self.mass_proton))  # from Beloy 2008
        # you calculate that 0.62 number from the mu of the nucleus, then
        # divide by 3/2 for the nuclear spin to get the g factor
        # it is weighted by m_e/m_p since the Hamiltonian must have
        # Bohr magneton out front

        # from Hay 1941, we have 0.6236
        # self.LandeGi = 0.6236 * (self.mass_electron /
        # self.mass_proton)

        # angular momentum quantum numbers for different orbitals
        self.S = 0.5
        self.J12 = 0.5
        self.J32 = 1.5
        self.J52 = 2.5
        self.L12 = 0
        self.L32 = 1
        self.L52 = 2
        self.nuc_I = 1.5

        # an ordered list of m_I and m_J values for the manifold
        self.IJ_values_S12 = np.array([[1.5, 0.5], [1.5, -0.5], [0.5, 0.5],
                                       [0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5],
                                       [-1.5, 0.5], [-1.5, -0.5]])

        # an ordered list of F and m_F values for the manifold
        self.F_values_S12 = np.array([[1, 1], [1, 0], [1, -1], [2, -2],
                                      [2, -1], [2, 0], [2, 1], [2, 2]])

        # an ordered list of m_I and m_J values for the manifold
        self.IJ_values_D52 = np.array([[1.5, 2.5], [1.5, 1.5], [1.5, 0.5],
                                       [1.5, -0.5], [1.5, -1.5], [1.5, -2.5],
                                       [0.5, 2.5], [0.5, 1.5], [0.5, 0.5],
                                       [0.5, -0.5], [0.5, -1.5], [0.5, -2.5],
                                       [-0.5, 2.5], [-0.5, 1.5], [-0.5, 0.5],
                                       [-0.5, -0.5], [-0.5, -1.5],
                                       [-0.5, -2.5], [-1.5, 2.5], [-1.5, 1.5],
                                       [-1.5, 0.5], [-1.5, -0.5], [-1.5, -1.5],
                                       [-1.5, -2.5]])

        # an ordered list of F and m_F values for the manifold
        self.F_values_D52 = np.array([[4, -4], [4, -3], [4, -2], [4,
                                                                  -1], [4, 0],
                                      [4, 1], [4, 2], [4, 3], [4, 4], [3, -3],
                                      [3, -2], [3, -1], [3, 0], [3, 1], [3, 2],
                                      [3, 3], [2, -2], [2, -1], [2, 0], [2, 1],
                                      [2, 2], [1, -1], [1, 0], [1, 1]])

        # need F and IJ values for P3/2 for F=3 at least for repumping map
        self.IJ_values_P32 = np.array([[1.5, 1.5], [1.5, 0.5], [1.5, -0.5],
                                       [1.5, -1.5], [0.5, 1.5], [0.5, 0.5],
                                       [0.5, -0.5], [0.5, -1.5], [-0.5, 1.5],
                                       [-0.5, 0.5], [-0.5, -0.5], [-0.5, -1.5],
                                       [-1.5, 1.5], [-1.5, 0.5], [-1.5, -0.5],
                                       [-1.5, -1.5]])
        self.F_values_P32 = np.array([[3, -3], [3, -2], [3, -1], [3, 0],
                                      [3, 1], [3, 2], [3, 3], [2, -2], [2, -1],
                                      [2, 0], [2, 1], [2, 2], [1, -1], [1, 0],
                                      [1, 1], [0, 0]])

        # generate some lists of Latex'd names for plotting purposes
        self.plotting_names_D52 = [
            r"$\tilde{F}$" + rf"$={F}$, " + r"$m_{\tilde{F}}$" + rf"$={m}$"
            # f"$\\tilde{{F}} = {F}, m_\\tilde{{F}} = {m}$"
            for F, m in self.F_values_D52
        ]

        self.plotting_names_P32 = [
            #rf"$F={F}$, " +
            rf"$m_F={m}$"
            # f"$\\tilde{{F}} = {F}, m_\\tilde{{F}} = {m}$"
            for F, m in self.F_values_P32
        ]

        self.plotting_names_S12 = [
            # r"$\tilde{F}$" + f"={F}, " + r"$\tilde{m}$" + f"={m}"
            rf"$F={F}$, " + rf"$m_F={m}$"
            # f"$\\tilde{{F}} = {F}, m_\\tilde{{F}} = {m}$"
            for F, m in self.F_values_S12
        ]
        self.plotting_names_S12_m_only = [
            rf"$m_F={m}$"
            for F, m in self.F_values_S12
        ]


if __name__ == '__main__':
    con = Constants()
    print(con.LandeGi)
    pass
