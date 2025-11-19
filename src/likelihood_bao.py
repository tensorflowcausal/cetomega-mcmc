"""
likelihood_bao.py

BAO likelihood using the CETΩ model and (mock or real) BAO data.
"""

import numpy as np
from model_cetomega import DM_of_z, DH_of_z, H0_DEFAULT

# ------------------------------------------------------------------
# Lectura de datos BAO (mock o DESI-like)
# ------------------------------------------------------------------
def load_bao_data(filename):
    """
    Lee un archivo de texto con columnas:
    z, DM_over_rd, DH_over_rd, sigma_DM, sigma_DH, corr

    Por ejemplo: data/desi_dr2_bao.txt
    """
    data = np.loadtxt(filename)
    z          = data[:, 0]
    DM_over_rd = data[:, 1]
    DH_over_rd = data[:, 2]
    sig_DM     = data[:, 3]
    sig_DH     = data[:, 4]
    corr       = data[:, 5]   # por ahora NO usamos corr en el chi2 simple
    return z, DM_over_rd, DH_over_rd, sig_DM, sig_DH, corr

# ------------------------------------------------------------------
# Chi^2 BAO (suponiendo errores independientes DM/DH)
# ------------------------------------------------------------------
def chi2_bao(params, z, DM_over_rd_obs, DH_over_rd_obs, sig_DM, sig_DH, rd):
    """
    params = (H0, Omega_m, Omega_r, Omega_Om, alpha0, alpha1, kappa)
    rd     = sound horizon (Mpc)
    """
    H0, Omega_m, Omega_r, Omega_Om, alpha0, alpha1, kappa = params

    # Vectorizar el cálculo teórico
    DM_th = np.array([DM_of_z(zz, params) for zz in z]) / rd
    DH_th = np.array([DH_of_z(zz, params) for zz in z]) / rd

    chi2_DM = np.sum(((DM_th - DM_over_rd_obs) / sig_DM) ** 2)
    chi2_DH = np.sum(((DH_th - DH_over_rd_obs) / sig_DH) ** 2)

    return chi2_DM + chi2_DH


if __name__ == "__main__":
    # Ejemplo de uso con datos tipo DESI (archivo en data/desi_dr2_bao.txt)
    filename = "data/desi_dr2_bao.txt"
    z, DMrd_obs, DHrd_obs, sDM, sDH, corr = load_bao_data(filename)

    # Parámetros de prueba
    Omega_m  = 0.3
    Omega_r  = 9e-5
    Omega_Om = 1.0 - Omega_m - Omega_r
    alpha0   = 3.0
    alpha1   = 0.5
    kappa    = 1.0
    rd       = 147.0  # Mpc, valor típico

    params = (H0_DEFAULT, Omega_m, Omega_r, Omega_Om, alpha0, alpha1, kappa)

    chi2 = chi2_bao(params, z, DMrd_obs, DHrd_obs, sDM, sDH, rd)
    print("chi^2 BAO (DESI-like) =", chi2)