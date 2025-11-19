"""
run_test.py

Quick test script to verify that model_cetomega.py works correctly.
Calculates H(z), DM(z), DH(z) for a range of redshifts and plots them.
"""

import numpy as np
import matplotlib.pyplot as plt

from model_cetomega import (
    H0_DEFAULT,
    Omega_m_DEFAULT,
    Omega_r_DEFAULT,
    Omega_Om_DEFAULT,
    alpha0_DEFAULT,
    alpha1_DEFAULT,
    kappa_DEFAULT,
    H_of_z,
    DM_of_z,
    DH_of_z,
)

# ----------------------------------------------------------
# Armamos el vector de parámetros
# ----------------------------------------------------------
params = (
    H0_DEFAULT,
    Omega_m_DEFAULT,
    Omega_r_DEFAULT,
    Omega_Om_DEFAULT,
    alpha0_DEFAULT,
    alpha1_DEFAULT,
    kappa_DEFAULT,
)

# ----------------------------------------------------------
# Probamos en un rango de z
# ----------------------------------------------------------
zs = np.linspace(0, 2, 50)

Hz_list  = [H_of_z(z, params) for z in zs]
DM_list  = [DM_of_z(z, params) for z in zs]
DH_list  = [DH_of_z(z, params) for z in zs]

# ----------------------------------------------------------
# Plots
# ----------------------------------------------------------

plt.figure(figsize=(7,5))
plt.plot(zs, Hz_list, label="H(z) CETΩ")
plt.xlabel("z")
plt.ylabel("H(z)  [km/s/Mpc]")
plt.title("CETΩ Expansion Rate")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("./figures/H_of_z_test.png")
plt.close()

plt.figure(figsize=(7,5))
plt.plot(zs, DM_list, label="D_M(z) CETΩ")
plt.xlabel("z")
plt.ylabel("D_M(z) [Mpc]")
plt.title("CETΩ Transverse Comoving Distance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("./figures/H_of_z_test.png")
plt.close()

plt.figure(figsize=(7,5))
plt.plot(zs, DH_list, label="D_H(z) CETΩ")
plt.xlabel("z")
plt.ylabel("D_H(z) [Mpc]")
plt.title("CETΩ Radial Distance D_H")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("./figures/H_of_z_test.png")
plt.close()

print("Listo. Los gráficos se guardaron en /figures/")