"""
run_mcmc.py

Simple Metropolis–Hastings MCMC for the CETΩ cosmological model
using the BAO likelihood defined in likelihood_bao.py.

Guarda las cadenas en data/chains.npy
"""

import numpy as np

from likelihood_bao import load_bao_data, chi2_bao
from model_cetomega import H0_DEFAULT, Omega_r_DEFAULT

# ---------------------------------------------------------
# Configuración básica del MCMC
# ---------------------------------------------------------

N_STEPS      = 20000   # número de pasos de la cadena
BURN_IN      = 5000    # pasos de burn-in
RANDOM_SEED  = 42

# Priors simples (uniformes) para los parámetros
PRIORS = {
    "Omega_m": (0.1, 0.5),
    "alpha0":  (1.5, 6.0),
    "alpha1":  (-2.0, 2.0),
}

# Dispersión de las propuestas (gaussianas) para cada parámetro
STEP_SIGMA = {
    "Omega_m": 0.02,
    "alpha0":  0.2,
    "alpha1":  0.1,
}


# ---------------------------------------------------------
# Funciones de ayuda
# ---------------------------------------------------------

def in_prior(Omega_m, alpha0, alpha1):
    """True si los parámetros están dentro de los priors."""
    omin, omax = PRIORS["Omega_m"]
    a0min, a0max = PRIORS["alpha0"]
    a1min, a1max = PRIORS["alpha1"]

    if not (omin <= Omega_m <= omax):
        return False
    if not (a0min <= alpha0 <= a0max):
        return False
    if not (a1min <= alpha1 <= a1max):
        return False
    return True


def log_prior(Omega_m, alpha0, alpha1):
    """Log-prior uniforme simple."""
    return 0.0 if in_prior(Omega_m, alpha0, alpha1) else -np.inf


def log_likelihood(Omega_m, alpha0, alpha1, z, DMrd_obs, DHrd_obs, sDM, sDH, rd):
    """Log-likelihood = -0.5 * chi^2."""
    Omega_r = Omega_r_DEFAULT
    Omega_Om = 1.0 - Omega_m - Omega_r
    kappa = 1.0
    params = (H0_DEFAULT, Omega_m, Omega_r, Omega_Om, alpha0, alpha1, kappa)

    chi2 = chi2_bao(params, z, DMrd_obs, DHrd_obs, sDM, sDH, rd)
    return -0.5 * chi2


def log_posterior(Omega_m, alpha0, alpha1, z, DMrd_obs, DHrd_obs, sDM, sDH, rd):
    """Log-posterior = log_prior + log_likelihood."""
    lp = log_prior(Omega_m, alpha0, alpha1)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(Omega_m, alpha0, alpha1,
                               z, DMrd_obs, DHrd_obs, sDM, sDH, rd)


# ---------------------------------------------------------
# Rutina principal de MCMC
# ---------------------------------------------------------

def run_mcmc():
    np.random.seed(RANDOM_SEED)

    # 1. Cargar datos REAL DESI DR2
    filename = "data/desi_dr2_bao.txt"
    z, DMrd_obs, DHrd_obs, sDM, sDH, corr = load_bao_data(filename)
    rd = 147.0  # Mpc (valor típico, suficiente para BAO publicados)

    # 2. Valores iniciales razonables
    Omega_m = 0.3
    alpha0  = 3.0
    alpha1  = 0.5

    current_logpost = log_posterior(Omega_m, alpha0, alpha1,
                                    z, DMrd_obs, DHrd_obs, sDM, sDH, rd)

    # 3. Arrays para almacenar la cadena
    chain = np.zeros((N_STEPS, 3))
    logpost_chain = np.zeros(N_STEPS)

    acceptance = 0

    for i in range(N_STEPS):
        # Propuestas gaussianas
        prop_Omega_m = Omega_m + STEP_SIGMA["Omega_m"] * np.random.randn()
        prop_alpha0  = alpha0  + STEP_SIGMA["alpha0"]  * np.random.randn()
        prop_alpha1  = alpha1  + STEP_SIGMA["alpha1"]  * np.random.randn()

        prop_logpost = log_posterior(prop_Omega_m, prop_alpha0, prop_alpha1,
                                     z, DMrd_obs, DHrd_obs, sDM, sDH, rd)

        # Ratio de aceptación
        delta = prop_logpost - current_logpost
        if np.log(np.random.rand()) < delta:
            # aceptar
            Omega_m = prop_Omega_m
            alpha0  = prop_alpha0
            alpha1  = prop_alpha1
            current_logpost = prop_logpost
            acceptance += 1

        chain[i, :] = [Omega_m, alpha0, alpha1]
        logpost_chain[i] = current_logpost

        if (i+1) % 1000 == 0:
            print(f"Paso {i+1}/{N_STEPS}")

    acc_rate = acceptance / N_STEPS
    print(f"Tasa de aceptación: {acc_rate:.3f}")

    # 4. Guardar las cadenas después del burn-in
    chain_post = chain[BURN_IN:, :]
    logpost_post = logpost_chain[BURN_IN:]

    np.save("data/chains.npy", chain_post)
    np.save("data/logpost.npy", logpost_post)

    # 5. Imprimir el mejor punto aproximado
    best_idx = np.argmax(logpost_post)
    best_params = chain_post[best_idx]
    print("Mejores parámetros aproximados (Omega_m, alpha0, alpha1):")
    print(best_params)


if __name__ == "__main__":
    run_mcmc()