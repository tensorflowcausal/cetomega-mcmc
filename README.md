# CETΩ – MCMC Corner Plot & Correlation Matrix

Este repositorio genera:
- `figures/corner_plot.pdf`
- `figures/corr_matrix.pdf`

## Instrucciones

1. Crear `data/chains.csv` con columnas: `M_star, alpha, epsilon`
2. Si no tenés MCMC real, podés generar un mock con:

```python
import numpy as np, pandas as pd
N=10000
M_star=np.random.normal(3.4e-5,0.7e-5,N)
alpha=np.random.normal(-1.1,0.3,N)
epsilon=np.random.normal(0.021,0.005,N)
pd.DataFrame({"M_star":M_star,"alpha":alpha,"epsilon":epsilon}).to_csv("data/chains.csv",index=False)
```

3. Ejecutar:
```
pip install -r requirements.txt
python make_corner.py
```
