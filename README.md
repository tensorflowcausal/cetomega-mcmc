# CETΩ – Cosmological MCMC Validation with BAO (DESI DR2)

This repository performs an empirical validation of the CETΩ expansion model using radial BAO data (e.g., DESI DR2) through a complete, clean MCMC pipeline.

The code:
- Fits CETΩ parameters using real BAO data
- Saves MCMC chains
- Generates all validation figures:
  - figures/corner_plot.pdf
  - figures/corr_matrix.pdf
  - figures/DH_cetomega_vs_DESI.png
  - figures/BAO_residuals_cetomega.png
  - figures/H_of_z_test.png

This repository is a stand-alone BAO validation and complements the full multisurvey CETΩ cosmology project.

## Repository Structure
cetomega-mcmc/
├── data/
│   ├── desi_dr2_bao.txt
│   └── …
├── figures/
│   ├── BAO_residuals_cetomega.png
│   ├── DH_cetomega_vs_DESI.png
│   ├── H_of_z_test.png
│   ├── corner_plot.pdf
│   └── corr_matrix.pdf
├── src/
│   ├── model_cetomega.py
│   ├── likelihood_bao.py
│   ├── run_mcmc.py
│   ├── run_chi2_mock.py
│   └── run_test.py
├── make_corner.py
├── make_figures.py
├── create_fake_chains.py
└── README.md

## Requirements

Python ≥ 3.10

Install dependencies:

```bash
pip install numpy scipy matplotlib emcee corner pandas

git clone https://github.com/tensorflowcausal/cetomega-mcmc.git
cd cetomega-mcmc

pip install numpy scipy matplotlib emcee corner pandas
data/desi_dr2_bao.txt
python -m src.run_mcmc
python make_figures.py
python make_corner.py
python create_fake_chains.py
src/model_cetomega.py

BALFAGON C.


