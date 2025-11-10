import numpy as np
import pandas as pd

# Crear datos MCMC de ejemplo
samples = np.random.normal(size=(5000, 3))
df = pd.DataFrame(samples, columns=["M_*", "alpha", "epsilon"])

# Crear carpeta y guardar CSV
import os
os.makedirs("data", exist_ok=True)
df.to_csv("data/chains.csv", index=False)

print("✅ Archivo data/chains.csv creado correctamente.")
