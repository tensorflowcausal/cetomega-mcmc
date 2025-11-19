"""
run_chi2_mock.py

Calcula chi^2 para datos BAO mock usando el modelo CETΩ.
"""

from likelihood_bao import load_mock_bao, chi2_bao
from model_cetomega import H0_DEFAULT, Omega_r_DEFAULT

def main():
    # 1. Cargo los datos mock
    filename = "data/mock_bao.txt"
    z, DMrd_obs, DHrd_obs, sDM, sDH = load_mock_bao(filename)

    # 2. Defino parámetros de prueba
    Omega_m  = 0.3
    Omega_r  = Omega_r_DEFAULT
    Omega_Om = 1.0 - Omega_m - Omega_r
    alpha0   = 3.0
    alpha1   = 0.5
    kappa    = 1.0
    rd       = 147.0  # Mpc, valor típico para el sound horizon

    params = (H0_DEFAULT, Omega_m, Omega_r, Omega_Om, alpha0, alpha1, kappa)

    # 3. Calculo chi^2
    chi2 = chi2_bao(params, z, DMrd_obs, DHrd_obs, sDM, sDH, rd)
    print("chi^2 (mock BAO, CETΩ) =", chi2)

if __name__ == "__main__":
    main()