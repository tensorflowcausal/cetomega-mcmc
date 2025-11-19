"""
model_cetomega.py

Core cosmological model for the causal–informational (CETΩ) framework.

Provides:
    - alpha_of_a(a, alpha0, alpha1)
    - w_Omega_of_a(a, alpha0, alpha1, kappa)
    - H_of_z(z, params)
    - chi_of_z(z, params)
    - DM_of_z(z, params)
    - DH_of_z(z, params)

`params` is a tuple:
    (H0, Omega_m, Omega_r, Omega_Om, alpha0, alpha1, kappa)
"""

import numpy as np

# ---------------------------------------------------------------------
# Parámetros por defecto (podés cambiarlos o pasarlos por parámetro)
# ---------------------------------------------------------------------
H0_DEFAULT       = 70.0          # km/s/Mpc
Omega_m_DEFAULT  = 0.3
Omega_r_DEFAULT  = 9e-5          # ~valor estándar para radiación
Omega_Om_DEFAULT = 1.0 - Omega_m_DEFAULT - Omega_r_DEFAULT
alpha0_DEFAULT   = 3.0
alpha1_DEFAULT   = 0.5
kappa_DEFAULT    = 1.0
c_LIGHT          = 299792.458    # km/s


# ---------------------------------------------------------------------
# Funciones del modelo CETΩ
# ---------------------------------------------------------------------
def alpha_of_a(a, alpha0, alpha1):
    """
    Structural exponent alpha(a) = alpha0 + alpha1 (1 - a).
    """
    return alpha0 + alpha1 * (1.0 - a)


def w_Omega_of_a(a, alpha0, alpha1, kappa=1.0):
    """
    Effective equation of state w_Ω(a) for the CETΩ component.

    w_Ω(a) ≈ -1 + 1/alpha(a) - kappa * ln(a) * alpha'(a) / alpha(a)^2
    with alpha'(a) = -alpha1.
    """
    if a <= 0.0:
        raise ValueError("a debe ser > 0")

    alpha = alpha_of_a(a, alpha0, alpha1)
    dalpha_da = -alpha1

    return -1.0 + 1.0 / alpha - kappa * np.log(a) * (dalpha_da / alpha**2)


def H_of_z(z, params):
    """
    H(z) para el modelo CETΩ dinámico.

    params = (H0, Omega_m, Omega_r, Omega_Om, alpha0, alpha1, kappa)
    """
    H0, Omega_m, Omega_r, Omega_Om, alpha0, alpha1, kappa = params

    # integrand for the exponential term in ρ_Ω(a)
    def integrand(zp):
        a = 1.0 / (1.0 + zp)
        return (1.0 + w_Omega_of_a(a, alpha0, alpha1, kappa)) / (1.0 + zp)

    # numerical integral 0->z
    if z == 0.0:
        integral = 0.0
    else:
        zs = np.linspace(0.0, z, 300)  # podés subir el número para más precisión
        integrand_vals = np.array([integrand(zp) for zp in zs])
        integral = np.trapz(integrand_vals, zs)

    expo = np.exp(3.0 * integral)

    return H0 * np.sqrt(
        Omega_m * (1.0 + z) ** 3
        + Omega_r * (1.0 + z) ** 4
        + Omega_Om * expo
    )


def chi_of_z(z, params):
    """
    Distancia comóvil χ(z) = ∫ c / H(z') dz'
    """
    if z == 0.0:
        return 0.0

    zs = np.linspace(0.0, z, 600)
    Hz = np.array([H_of_z(zp, params) for zp in zs])
    return np.trapz(c_LIGHT / Hz, zs)  # Mpc


def DM_of_z(z, params):
    """
    Distancia comóvil transversal D_M(z) (en universo plano = χ(z)).
    """
    return chi_of_z(z, params)


def DH_of_z(z, params):
    """
    Distancia radial D_H(z) = c / H(z).
    """
    return c_LIGHT / H_of_z(z, params)


# ---------------------------------------------------------------------
# Ejemplo rápido si ejecutás este archivo directamente
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Parámetros por defecto en un solo tuple
    params_default = (
        H0_DEFAULT,
        Omega_m_DEFAULT,
        Omega_r_DEFAULT,
        Omega_Om_DEFAULT,
        alpha0_DEFAULT,
        alpha1_DEFAULT,
        kappa_DEFAULT,
    )

    z_test = 0.5
    Hz = H_of_z(z_test, params_default)
    DM = DM_of_z(z_test, params_default)
    DH = DH_of_z(z_test, params_default)

    print("Parámetros por defecto CETΩ:")
    print("  H0       =", H0_DEFAULT)
    print("  Omega_m  =", Omega_m_DEFAULT)
    print("  Omega_r  =", Omega_r_DEFAULT)
    print("  Omega_Ω  =", Omega_Om_DEFAULT)
    print("  alpha0   =", alpha0_DEFAULT)
    print("  alpha1   =", alpha1_DEFAULT)
    print()
    print(f"H(z={z_test})  = {Hz:.3f} km/s/Mpc")
    print(f"DM(z={z_test}) = {DM:.3f} Mpc")
    print(f"DH(z={z_test}) = {DH:.3f} Mpc")