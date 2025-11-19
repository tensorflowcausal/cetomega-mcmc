import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------
# 1) Cargar cadena MCMC
# ---------------------------
chains_path = "data/chains.csv"
chains = pd.read_csv(chains_path)

# nombres de columnas (los que ves en la captura)
M_star_samples   = chains["M_*"].values
alpha_samples    = chains["alpha"].values
epsilon_samples  = chains["epsilon"].values

# elige un resumen: media
M_star   = np.mean(M_star_samples)
alpha    = np.mean(alpha_samples)
epsilon  = np.mean(epsilon_samples)

print("Parámetros efectivos que voy a usar para las figuras:")
print("M_*   =", M_star)
print("alpha =", alpha)
print("eps   =", epsilon)

# ---------------------------
# 2) Cargar datos DESI DR2
# ---------------------------
desi_path = "data/desi_dr2_bao.txt"
desi = np.loadtxt(desi_path, comments="#")

z_eff   = desi[:,0]
DH_over_rd_data = desi[:,2]
sigma_DH        = desi[:,4]

r_d = 1.0  # placeholder


# ---------------------------
# 3) Modelo CETΩ
# ---------------------------
def D_H_cetomega(z, M_star, alpha, epsilon):
    return 3000.0 / (1.0 + z)**alpha * (1.0 + epsilon*z)


DH_over_rd_model = D_H_cetomega(z_eff, M_star, alpha, epsilon) / r_d

# ---------------------------
# 4) Figura D_H vs DESI
# ---------------------------
os.makedirs("figures", exist_ok=True)

plt.figure()
plt.errorbar(z_eff, DH_over_rd_data, yerr=sigma_DH,
             fmt="o", label="DESI DR2 (radial BAO)")
z_fine = np.linspace(0.2, 2.2, 300)
DH_model_fine = D_H_cetomega(z_fine, M_star, alpha, epsilon) / r_d
plt.plot(z_fine, DH_model_fine, label="CET Ω (best fit)")

plt.xlabel(r"$z$")
plt.ylabel(r"$D_H(z)/r_d$")
plt.legend()
plt.tight_layout()
plt.savefig("figures/DH_cetomega_vs_DESI.png", dpi=300)

# ---------------------------
# 5) Figura residuales
# ---------------------------
residuals = (DH_over_rd_data - DH_over_rd_model) / sigma_DH

plt.figure()
plt.axhline(0.0, linestyle="--")
plt.errorbar(z_eff, residuals, yerr=np.ones_like(sigma_DH),
             fmt="o", label="DESI DR2 – CET Ω")
plt.xlabel(r"$z$")
plt.ylabel(r"$[D_H^{data}-D_H^{model}]/σ$")
plt.legend()
plt.tight_layout()
plt.savefig("figures/BAO_residuals_cetomega.png", dpi=300)