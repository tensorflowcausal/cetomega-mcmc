import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

csv_path = "data/chains.csv"
params = [
    ("M_*", r"$M_\ast$ [$10^{-5}\,\mathrm{m}^{-1}$]"),
    ("alpha",  r"$\alpha$"),
    ("epsilon", r"$\epsilon$")
]
burn_in = 0.10
thin = 5

df = pd.read_csv(csv_path)
n0 = int(len(df)*burn_in)
df = df.iloc[n0::thin].reset_index(drop=True)

X = np.vstack([df[col].to_numpy() for col, _ in params]).T

if "M_star" in df.columns:
    X[:, 0] *= 1e5

K = X.shape[1]
fig, axes = plt.subplots(K, K, figsize=(2.4*K, 2.4*K))

for i in range(K):
    for j in range(K):
        ax = axes[i, j]
        if i == j:
            ax.hist(X[:, i], bins=40, density=True, color="steelblue")
        elif j < i:
            H, xedges, yedges = np.histogram2d(X[:, j], X[:, i], bins=60, density=True)
            Xc = 0.5*(xedges[1:]+xedges[:-1])
            Yc = 0.5*(yedges[1:]+yedges[:-1])
            ax.contour(Xc, Yc, H.T, levels=[0.1*np.max(H), 0.5*np.max(H)], colors="k")
        else:
            ax.axis("off")
        if i == K-1:
            ax.set_xlabel(params[j][1])
        if j == 0:
            ax.set_ylabel(params[i][1])

plt.tight_layout()
plt.savefig("figures/corner_plot.pdf", bbox_inches="tight")
plt.close(fig)

C = np.corrcoef(X, rowvar=False)
fig2, ax2 = plt.subplots(figsize=(1.8*K, 1.5*K))
im = ax2.imshow(C, vmin=-1, vmax=1, cmap="coolwarm")
ax2.set_xticks(range(K))
ax2.set_xticklabels([lab for _, lab in params], rotation=45, ha='right')
ax2.set_yticks(range(K))
ax2.set_yticklabels([lab for _, lab in params])
for i in range(K):
    for j in range(K):
        ax2.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", fontsize=9)
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="corr")
plt.tight_layout()
plt.savefig("figures/corr_matrix.pdf", bbox_inches="tight")
plt.close(fig2)

print("✅ Guardado: figures/corner_plot.pdf y figures/corr_matrix.pdf")
