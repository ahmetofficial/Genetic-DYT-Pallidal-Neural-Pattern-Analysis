"""
Utilisation function for plotting
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def oscillatory_characteristics(dataset, gene_factor):
    osc_delta  = np.sum(dataset[dataset.gene == gene_factor][['delta_band_oscillatory']].sum(axis=1) == 1)
    osc_theta  = np.sum(dataset[dataset.gene == gene_factor][['theta_band_oscillatory']].sum(axis=1) == 1)
    osc_alpha  = np.sum(dataset[dataset.gene == gene_factor][['alpha_band_oscillatory']].sum(axis=1) == 1)
    osc_beta   = np.sum(dataset[dataset.gene == gene_factor][['beta_band_oscillatory']].sum(axis=1) == 1)
    osc_gamma  = np.sum(dataset[dataset.gene == gene_factor][['gamma_band_oscillatory']].sum(axis=1) == 1)
    osc_non    = np.sum(dataset[dataset.gene == gene_factor][['delta_band_oscillatory', 'theta_band_oscillatory','alpha_band_oscillatory', 'beta_band_oscillatory','gamma_band_oscillatory']].sum(axis=1) == 0)
    
    oscillations = [osc_delta, osc_theta, osc_alpha, osc_beta, osc_gamma, osc_non]
    oscillations = [ x / len(dataset[dataset.gene == gene_factor]) *100 for x in oscillations]
    return oscillations

