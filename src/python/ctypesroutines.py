import numpy as np

import ctypes
from ctypes import c_double, c_int, c_void_p, POINTER
# ============================================
# STEP 1: Load the Fortran library (ONCE)
# ============================================
#lib = ctypes.CDLL('./libhamiltonian.so')
lib_3rd = ctypes.CDLL('./libhamiltonian_3rd.so')
hamiltonian_func_3rd = lib_3rd.hamiltonian_cubic3d_2nd

hamiltonian_2d_n1_func_3rd = lib_3rd.hamiltonian_cubic2d_n1_3rd
hamiltonian_2d_n2_func_3rd = lib_3rd.hamiltonian_cubic2d_n2_3rd
hamiltonian_2d_n3_func_3rd = lib_3rd.hamiltonian_cubic2d_n3_3rd

hamiltonian_cubic2d_general = lib_3rd.hamiltonian_cubic2d_general
hamiltonian_cubic2d_general.argtypes = [
    c_int,                                       # n_layers
    c_double, c_double, c_double,                # kx, ky, kz
    c_double, c_double, c_double, c_double,      # Es0, Ep0, Es1, Ep1
    c_double, c_double, c_double, c_double, c_double,  # Vss, Vsp, Vps, Vppsig, Vpppi
    c_double, c_double,                          # deltaSO0, deltaSO1
    c_double, c_double, c_double, c_double,      # Vss_II, Vsp_II, Vppsig_II, Vpppi_II
    c_double, c_double, c_double, c_double,      # Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz
    c_double, c_double, c_double, c_double,      # Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb
    c_void_p,                                    # ham array pointer
    c_int,                                       # n (matrix dimension = 32*n_layers + 8)
]
hamiltonian_cubic2d_general.restype = None

# Define the function signature: all scalars passed by value, then array pointer, then n
hamiltonian_func_3rd.argtypes = [
    c_double, c_double, c_double,           # kx, ky, kz
    c_double, c_double, c_double, c_double, # Es0, Ep0, Es1, Ep1
    c_double, c_double, c_double, c_double, c_double,  # Vss, Vsp, Vps, Vppsig, Vpppi
    c_double, c_double,                     # deltaSO0, deltaSO1
    c_double, c_double, c_double, c_double, # Vss_II, Vsp_II, Vppsig_II, Vpppi_II
    c_double, c_double, c_double, c_double, # Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb
    c_void_p,                               # ham array pointer
    c_int                                   # n
]
hamiltonian_func_3rd.restype = None


#3rd neighbor functions
hamiltonian_2d_n1_func_3rd.argtypes = [
    c_double, c_double, c_double,
    c_double, c_double, c_double, c_double,
    c_double, c_double, c_double, c_double, c_double,
    c_double, c_double,
    c_double, c_double, c_double, c_double,   # Vss_II, Vsp_II, Vppsig_II, Vpppi_II
    c_double, c_double, c_double, c_double,   # Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz
    c_double, c_double, c_double, c_double,            # Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb
    c_void_p, c_int
]
hamiltonian_2d_n1_func_3rd.restype = None

hamiltonian_2d_n2_func_3rd.argtypes = [
    c_double, c_double, c_double,
    c_double, c_double, c_double, c_double,
    c_double, c_double, c_double, c_double, c_double,
    c_double, c_double,
    c_double, c_double, c_double, c_double,   # Vss_II, Vsp_II, Vppsig_II, Vpppi_II
    c_double, c_double, c_double, c_double,   # Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz
    c_double, c_double, c_double, c_double,            # Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb
    c_void_p, c_int
]
hamiltonian_2d_n2_func_3rd.restype = None

hamiltonian_2d_n3_func_3rd.argtypes = [
    c_double, c_double, c_double,
    c_double, c_double, c_double, c_double,
    c_double, c_double, c_double, c_double, c_double,
    c_double, c_double,
    c_double, c_double, c_double, c_double,   # Vss_II, Vsp_II, Vppsig_II, Vpppi_II
    c_double, c_double, c_double, c_double,   # Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz
    c_double, c_double, c_double, c_double,            # Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb
    c_void_p, c_int
]
hamiltonian_2d_n3_func_3rd.restype = None

# ============================================
# STEP 2: Python wrapper (no byref)
# ============================================
def hamiltonian_cubic3D_fortran(kx, ky, kz, Es0, Ep0, Es1, Ep1,
                                 Vss, Vsp, Vps, Vppsig, Vpppi,
                                 deltaSO0, deltaSO1, Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
                                 Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb):
    n = 32
    ham = np.zeros((n, n), dtype=np.complex128, order='F')
    hamiltonian_func_3rd(kx, ky, kz,
                     Es0, Ep0, Es1, Ep1,
                     Vss, Vsp, Vps, Vppsig, Vpppi,
                     deltaSO0, deltaSO1,
                     Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
                     Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb,
                     ham.ctypes.data, n)
    return ham
    

def hamiltonian_cubic2D_n1_fortran_3rd(kx, ky, kz, Es0, Ep0, Es1, Ep1,
                                       Vss, Vsp, Vps, Vppsig, Vpppi,
                                       deltaSO0, deltaSO1, Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
                                       Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
                                       Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb):
    n = 40  # output matrix size
    ham = np.zeros((n, n), dtype=np.complex128, order='F')
    hamiltonian_2d_n1_func_3rd(kx, ky, kz,
                            Es0, Ep0, Es1, Ep1,
                            Vss, Vsp, Vps, Vppsig, Vpppi,
                            deltaSO0, deltaSO1,
                            Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
                            Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
                            Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb,
                            ham.ctypes.data, n)
    return ham

def hamiltonian_cubic2D_n2_fortran_3rd(kx, ky, kz, Es0, Ep0, Es1, Ep1,
                                       Vss, Vsp, Vps, Vppsig, Vpppi,
                                       deltaSO0, deltaSO1, Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
                                       Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
                                       Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb):
    n = 72  # output matrix size
    ham = np.zeros((n, n), dtype=np.complex128, order='F')
    hamiltonian_2d_n2_func_3rd(kx, ky, kz,
                            Es0, Ep0, Es1, Ep1,
                            Vss, Vsp, Vps, Vppsig, Vpppi,
                            deltaSO0, deltaSO1,
                            Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
                            Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
                            Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb,
                            ham.ctypes.data, n)
    return ham

def hamiltonian_cubic2D_n3_fortran_3rd(kx, ky, kz, Es0, Ep0, Es1, Ep1,
                                        Vss, Vsp, Vps, Vppsig, Vpppi,
                                        deltaSO0, deltaSO1, Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
                                        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
                                        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb):
     n = 104
     ham = np.zeros((n, n), dtype=np.complex128, order='F')
     hamiltonian_2d_n3_func_3rd(
          kx, ky, kz,
          Es0, Ep0, Es1, Ep1,
          Vss, Vsp, Vps, Vppsig, Vpppi,
          deltaSO0, deltaSO1,
          Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
          Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
          Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb,
          ham.ctypes.data_as(c_void_p), c_int(n)
     )
     return ham

def hamiltonian_cubic2D_general_fortran(n_layers, kx, ky, kz,
                                        Es0, Ep0, Es1, Ep1,
                                        Vss, Vsp, Vps, Vppsig, Vpppi,
                                        deltaSO0, deltaSO1,
                                        Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
                                        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
                                        Vss_PbPb, Vsp_PbPb,
                                        Vppsig_PbPb, Vpppi_PbPb):
    n = 32 * n_layers + 8
    ham = np.zeros((n, n), dtype=np.complex128, order='F')
    hamiltonian_cubic2d_general(
        c_int(n_layers),
        kx, ky, kz,
        Es0, Ep0, Es1, Ep1,
        Vss, Vsp, Vps, Vppsig, Vpppi,
        deltaSO0, deltaSO1,
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb,
        ham.ctypes.data_as(c_void_p), c_int(n),
    )
    return ham


def effective_mass_2d_general_fortran(n_layers, kx, ky, kz,
                                      band_idx, ax, ay, az, eps,
                                      Es0, Ep0, Es1, Ep1,
                                      Vss, Vsp, Vps, Vppsig, Vpppi,
                                      deltaSO0, deltaSO1,
                                      Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
                                      Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
                                      Vss_PbPb, Vsp_PbPb,
                                      Vppsig_PbPb, Vpppi_PbPb):
    n = 32 * n_layers + 8
    inv_mass = np.zeros(3, dtype=np.float64)
    eff_mass_func_2d_general(
        c_int(n_layers),
        kx, ky, kz,
        c_int(band_idx),
        ax, ay, az, eps,
        Es0, Ep0, Es1, Ep1,
        Vss, Vsp, Vps, Vppsig, Vpppi,
        deltaSO0, deltaSO1,
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb,
        inv_mass.ctypes.data_as(c_void_p), c_int(n),
    )
    return inv_mass  # [1/mx, 1/my, 1/mz] in units of 1/m0


# ============================================
# Diagonalization wrappers
# ============================================

# Get the diagonalization functions from the library
diag_func = lib_3rd.diagonalize_hamiltonian
eigen_func = lib_3rd.eigenvalues_only

# Define function signatures
diag_func.argtypes = [c_void_p, c_void_p, c_int]  # ham, eigenvalues, n
diag_func.restype = None

eigen_func.argtypes = [c_void_p, c_void_p, c_int]  # ham, eigenvalues, n
eigen_func.restype = None

                        
def diagonalize_hamiltonian_fortran(ham):
    """
    Diagonalize a complex Hermitian matrix using LAPACK's ZHEEV.
    Returns (eigenvalues, eigenvectors)
    Note: The input matrix is overwritten with eigenvectors.
    """
    n = ham.shape[0]
    # Make a copy to avoid modifying the original
    ham_copy = ham.copy(order='F')
    eigenvalues = np.zeros(n, dtype=np.float64)
    
    diag_func(ham_copy.ctypes.data, eigenvalues.ctypes.data, n)
    
    return eigenvalues, ham_copy

def eigenvalues_only_fortran(ham):
    """
    Compute only eigenvalues (faster than full diagonalization).
    Note: The input matrix is destroyed in the process.
    """
    n = ham.shape[0]
    ham_copy = ham.copy(order='F')
    eigenvalues = np.zeros(n, dtype=np.float64)
    
    eigen_func(ham_copy.ctypes.data, eigenvalues.ctypes.data, n)
    
    return eigenvalues
    

eff_mass_func_2d_general = lib_3rd.effective_mass_2d_general
eff_mass_func_2d_general.argtypes = [
    c_int,                                       # n_layers
    c_double, c_double, c_double,                # kx, ky, kz
    c_int,                                       # band_idx (1-based)
    c_double, c_double, c_double, c_double,      # ax, ay, az, eps
    c_double, c_double, c_double, c_double,      # Es0, Ep0, Es1, Ep1
    c_double, c_double, c_double, c_double, c_double,  # Vss, Vsp, Vps, Vppsig, Vpppi
    c_double, c_double,                          # deltaSO0, deltaSO1
    c_double, c_double, c_double, c_double,      # Vss_II, Vsp_II, Vppsig_II, Vpppi_II
    c_double, c_double, c_double, c_double,      # Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz
    c_double, c_double, c_double, c_double,      # Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb
    c_void_p,                                    # inv_mass(3) output
    c_int,                                       # n (matrix dimension = 32*n_layers + 8)
]
eff_mass_func_2d_general.restype = None

# ---- apical-corrected 2D routines (25 parameters: 23 base + Es1_ap + Ep1_ap) ----
hamiltonian_cubic2d_apical = lib_3rd.hamiltonian_cubic2d_apical
hamiltonian_cubic2d_apical.argtypes = [
    c_int,                                       # n_layers
    c_double, c_double, c_double,                # kx, ky, kz
    c_double, c_double, c_double, c_double,      # Es0, Ep0, Es1, Ep1
    c_double, c_double, c_double, c_double, c_double,  # Vss, Vsp, Vps, Vppsig, Vpppi
    c_double, c_double,                          # deltaSO0, deltaSO1
    c_double, c_double, c_double, c_double,      # Vss_II, Vsp_II, Vppsig_II, Vpppi_II
    c_double, c_double, c_double, c_double,      # Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz
    c_double, c_double, c_double, c_double,      # Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb
    c_double, c_double, c_double,                # Es1_ap, Ep1_ap, Ep0_surf
    c_void_p,                                    # ham array pointer
    c_int,                                       # n (matrix dimension = 32*n_layers + 8)
]
hamiltonian_cubic2d_apical.restype = None

eff_mass_func_2d_apical = lib_3rd.effective_mass_2d_apical
eff_mass_func_2d_apical.argtypes = [
    c_int,                                       # n_layers
    c_double, c_double, c_double,                # kx, ky, kz
    c_int,                                       # band_idx
    c_double, c_double, c_double, c_double,      # ax, ay, az, eps
    c_double, c_double, c_double, c_double,      # Es0, Ep0, Es1, Ep1
    c_double, c_double, c_double, c_double, c_double,  # Vss, Vsp, Vps, Vppsig, Vpppi
    c_double, c_double,                          # deltaSO0, deltaSO1
    c_double, c_double, c_double, c_double,      # Vss_II, Vsp_II, Vppsig_II, Vpppi_II
    c_double, c_double, c_double, c_double,      # Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz
    c_double, c_double, c_double, c_double,      # Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb
    c_double, c_double, c_double,                # Es1_ap, Ep1_ap, Ep0_surf
    c_void_p,                                    # inv_mass(3) output
    c_int,                                       # n (matrix dimension = 32*n_layers + 8)
]
eff_mass_func_2d_apical.restype = None


def hamiltonian_cubic2D_apical_fortran(n_layers, kx, ky, kz,
                                       Es0, Ep0, Es1, Ep1,
                                       Vss, Vsp, Vps, Vppsig, Vpppi,
                                       deltaSO0, deltaSO1,
                                       Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
                                       Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
                                       Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb,
                                       Es1_ap, Ep1_ap, Ep0_surf):
    n = 32 * n_layers + 8
    ham = np.zeros((n, n), dtype=np.complex128, order='F')
    hamiltonian_cubic2d_apical(
        c_int(n_layers),
        kx, ky, kz,
        Es0, Ep0, Es1, Ep1,
        Vss, Vsp, Vps, Vppsig, Vpppi,
        deltaSO0, deltaSO1,
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb,
        Es1_ap, Ep1_ap, Ep0_surf,
        ham.ctypes.data_as(c_void_p), c_int(n),
    )
    return ham


def effective_mass_2d_apical_fortran(n_layers, kx, ky, kz,
                                     band_idx, ax, ay, az, eps,
                                     Es0, Ep0, Es1, Ep1,
                                     Vss, Vsp, Vps, Vppsig, Vpppi,
                                     deltaSO0, deltaSO1,
                                     Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
                                     Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
                                     Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb,
                                     Es1_ap, Ep1_ap, Ep0_surf):
    n = 32 * n_layers + 8
    inv_mass = np.zeros(3, dtype=np.float64)
    eff_mass_func_2d_apical(
        c_int(n_layers),
        kx, ky, kz,
        c_int(band_idx),
        ax, ay, az, eps,
        Es0, Ep0, Es1, Ep1,
        Vss, Vsp, Vps, Vppsig, Vpppi,
        deltaSO0, deltaSO1,
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb,
        Es1_ap, Ep1_ap, Ep0_surf,
        inv_mass.ctypes.data_as(c_void_p), c_int(n),
    )
    return inv_mass


eff_mass_func_3D_2nd = lib_3rd.effective_mass_3d_2nd

#3rd neighbor effective mass functions
eff_mass_func_2Dn1_3rd = lib_3rd.effective_mass_2dn1_3rd
eff_mass_func_2Dn2_3rd = lib_3rd.effective_mass_2dn2_3rd
eff_mass_func_2Dn3_3rd = lib_3rd.effective_mass_2dn3_3rd


eff_mass_func_3D_2nd.argtypes = [c_double, c_double, c_double,  # kx, ky, kz
                     c_int,                         # band_idx (1‑based)
                     c_double, c_double, c_double,  # ax, ay, az
                     c_double,                      # eps
                     c_double, c_double, c_double, c_double,            # Es0, Ep0, Es1, Ep1
                     c_double, c_double, c_double, c_double, c_double,  # Vss, Vsp, Vps, Vppsig, Vpppi
                     c_double, c_double,                                # deltaSO0, deltaSO1
                     c_double, c_double, c_double, c_double,            # Vss_II, Vsp_II, Vppsig_II, Vpppi_II
                     c_double, c_double, c_double, c_double,            # Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb
                     c_void_p,                      # inv_mass (output)
                     c_int]                         # n
eff_mass_func_3D_2nd.restype = None


#3rd neighbor effective mass functions
eff_mass_func_2Dn1_3rd.argtypes = [c_double, c_double, c_double,  # kx, ky, kz
                     c_int,                         # band_idx (1‑based)    
                        c_double, c_double, c_double,  # ax, ay, az
                        c_double,                      # eps
                        c_double, c_double, c_double, c_double,            # Es0, Ep0, Es1, Ep1
                        c_double, c_double, c_double, c_double, c_double,  # Vss, Vsp, Vps, Vppsig, Vpppi
                        c_double, c_double,                                # deltaSO0, deltaSO1
                        c_double, c_double, c_double, c_double,            # Vss_II, Vsp_II, Vppsig_II, Vpppi_II
                        c_double, c_double, c_double, c_double,            # Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz
                        c_double, c_double, c_double, c_double,            # Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb
                        c_void_p,                      # inv_mass (output)
                        c_int]                         # n
eff_mass_func_2Dn1_3rd.restype = None

eff_mass_func_2Dn2_3rd.argtypes = [c_double, c_double, c_double,  # kx, ky, kz
                     c_int,                         # band_idx (1‑based) 
                        c_double, c_double, c_double,  # ax, ay, az
                        c_double,                      # eps
                        c_double, c_double, c_double, c_double,            # Es0, Ep0, Es1, Ep1
                        c_double, c_double, c_double, c_double, c_double,  # Vss, Vsp, Vps, Vppsig, Vpppi
                        c_double, c_double,                                # deltaSO0, deltaSO1
                        c_double, c_double, c_double, c_double,            # Vss_II, Vsp_II, Vppsig_II, Vpppi_II
                        c_double, c_double, c_double, c_double,            # Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz
                        c_double, c_double, c_double, c_double,            # Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb
                        c_void_p,                      # inv_mass (output)
                        c_int]                         # n
eff_mass_func_2Dn2_3rd.restype = None

eff_mass_func_2Dn3_3rd.argtypes = [c_double, c_double, c_double,  # kx, ky, kz
                     c_int,                         # band_idx (1‑based)
                        c_double, c_double, c_double,  # ax, ay, az
                        c_double,                      # eps
                        c_double, c_double, c_double, c_double,            # Es0, Ep0, Es1, Ep1
                        c_double, c_double, c_double, c_double, c_double,  # Vss, Vsp, Vps, Vppsig, Vpppi
                        c_double, c_double,                                # deltaSO0, deltaSO1
                        c_double, c_double, c_double, c_double,            # Vss_II, Vsp_II, Vppsig_II, Vpppi_II
                        c_double, c_double, c_double, c_double,            # Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz
                        c_double, c_double, c_double, c_double,            # Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb
                        c_void_p,                      # inv_mass (output)
                        c_int]                         # n
eff_mass_func_2Dn3_3rd.restype = None

def effective_mass_3d_fortran(kx, ky, kz,
                                    band_idx, n, ax, ay, az, eps,
                                    Es0, Ep0, Es1, Ep1,
                                    Vss, Vsp, Vps, Vppsig, Vpppi,
                                    deltaSO0, deltaSO1, Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
                                    Vss_PbPb=0.0, Vsp_PbPb=0.0, Vppsig_PbPb=0.0, Vpppi_PbPb=0.0):
    inv_mass = np.zeros((3), dtype=np.float64)
    eff_mass_func_3D_2nd(
        kx, ky, kz, band_idx, ax, ay, az, eps,
        Es0, Ep0, Es1, Ep1,
        Vss, Vsp, Vps, Vppsig, Vpppi,
        deltaSO0, deltaSO1, Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb,
        inv_mass.ctypes.data_as(c_void_p), c_int(n)
    )
    return inv_mass



#3rd neighbor effective mass functions
def effective_mass_2dn1_fortran_3rd(kx, ky, kz,
                                    band_idx,n, ax, ay, az, eps,
                                    Es0, Ep0, Es1, Ep1,
                                    Vss, Vsp, Vps, Vppsig, Vpppi,
                                    deltaSO0, deltaSO1, Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
                                    Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
                                    Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb):
    #n = 32
    inv_mass = np.zeros((3), dtype=np.float64)
    eff_mass_func_2Dn1_3rd(
        kx, ky, kz, band_idx, ax, ay, az, eps,
        Es0, Ep0, Es1, Ep1,
        Vss, Vsp, Vps, Vppsig, Vpppi,
        deltaSO0, deltaSO1,
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb,
        inv_mass.ctypes.data_as(c_void_p), c_int(n)
    )
    return inv_mass

def effective_mass_2dn2_fortran_3rd(kx, ky, kz,
                                    band_idx,n, ax, ay, az, eps,
                                    Es0, Ep0, Es1, Ep1,
                                    Vss, Vsp, Vps, Vppsig, Vpppi,
                                    deltaSO0, deltaSO1, Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
                                    Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
                                    Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb):
    #n = 32
    inv_mass = np.zeros((3), dtype=np.float64)
    eff_mass_func_2Dn2_3rd(
        kx, ky, kz, band_idx, ax, ay, az, eps,
        Es0, Ep0, Es1, Ep1,
        Vss, Vsp, Vps, Vppsig, Vpppi,
        deltaSO0, deltaSO1,
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb,
        inv_mass.ctypes.data_as(c_void_p), c_int(n)
    )
    return inv_mass

def effective_mass_2dn3_fortran_3rd(kx, ky, kz,
                                    band_idx,n, ax, ay, az, eps,
                                    Es0, Ep0, Es1, Ep1,
                                    Vss, Vsp, Vps, Vppsig, Vpppi,
                                    deltaSO0, deltaSO1, Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
                                    Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
                                    Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb):
    #n = 32
    inv_mass = np.zeros((3), dtype=np.float64)
    eff_mass_func_2Dn3_3rd(
        kx, ky, kz, band_idx, ax, ay, az, eps,
        Es0, Ep0, Es1, Ep1,
        Vss, Vsp, Vps, Vppsig, Vpppi,
        deltaSO0, deltaSO1,
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb,
        inv_mass.ctypes.data_as(c_void_p), c_int(n)
    )
    return inv_mass

