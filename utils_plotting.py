"""
Utilisation function for plotting
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from scipy.signal import spectrogram
from mpl_toolkits.axes_grid1 import make_axes_locatable
from decimal import Decimal
from matplotlib.colors import BoundaryNorm

import utils_statistics

labelsize             = 5

gene_colors           = {}
gene_colors["AOPEP"]  = "#240E5E"
gene_colors["GNAL"]   = "#044D3E"
gene_colors["KMT2B"]  = "#155A8D"
gene_colors["PANK2"]  = "#1BBB9D"
gene_colors["PLA2G6"] = "#FFD939" 
gene_colors["SGCE"]   = "#FE9284"
gene_colors["THAP1"]  = "#FC8502"
gene_colors["TOR1A"]  = "#E11500"
gene_colors["VPS16"]  = "#95006F"

def get_figure_template():
    
    plt.rc('font', serif="Neue Haas Grotesk Text Pro")
    fig = plt.figure(edgecolor='none')
    fig.tight_layout()
    fig.patch.set_visible(False)
    cm = 1/2.54  # centimeters in inches
    plt.subplots(figsize=(18.5*cm, 21*cm))
    plt.axis('off') 
    return plt

def set_axis(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=5)
    ax.tick_params(axis='both', which='minor', labelsize=5)
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    ax.set_xlabel(ax.get_xlabel(), fontsize=5)
    ax.set_ylabel(ax.get_ylabel(), fontsize=5)
    ax.yaxis.offsetText.set_fontsize(5)

def radar_chart(dataframe, axis):
    # Define the categories for the radar chart
    categories = dataframe.index.tolist()
    values1    = dataframe['PC1'].values.tolist()
    values2    = dataframe['PC2'].values.tolist()
    values1   += [values1[0]]
    values2   += [values2[0]]

    angles     = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
    angles    += angles[:1]
    
    axis.plot(angles, values1, 'o-', linewidth=0.2, markersize=0.25, label="PC1", color = "#9B59B6")
    axis.fill(angles, values1, alpha=0.25, color = "#9B59B6")
    axis.plot(angles, values2, 'o-', linewidth=0.2, markersize=0.25, label="PC2", color = "#C0392B")
    axis.fill(angles, values2, alpha=0.25, color = "#C0392B")

    # Set the labels for each category
    axis.set_thetagrids(np.degrees(angles[:-1]), [])

    # Set the category labels parallel to the lines
    for i, label in enumerate(categories):
        angle = angles[i]
        x = np.cos(angle)
        y = np.sin(angle)
        if (angle <= np.pi/2):
            axis.text(angle, 1, label, ha='left', va='center', rotation=angle*180/np.pi, rotation_mode='anchor', 
                      bbox=dict(facecolor='white', edgecolor='none'), fontsize=labelsize, weight='bold')
        elif(angle >= np.pi/2*3):
            axis.text(angle, 1, label, ha='left', va='center', rotation=angle*180/np.pi, rotation_mode='anchor', 
                      bbox=dict(facecolor='white', edgecolor='none'), fontsize=labelsize, weight='bold')
        else:
            axis.text(angle, 1, label, ha='right', va='center', rotation=(angle-np.pi)*180/np.pi, rotation_mode='anchor', 
                      bbox=dict(facecolor='white', edgecolor='none'), fontsize=labelsize,  weight='bold')

    axis.set_rgrids([0.2,0.4, 0.6, 0.8], angle=90, fontsize=labelsize)
    
def plot_PCA(data, gene1, gene2, axis, colors, score):
    axis = sns.scatterplot(data=data[(data.gene==gene2)], x='PC1', y='PC2', alpha=1, color=gene_colors[gene2], ax=axis, s=15)
    axis = sns.scatterplot(data=data[(data.gene==gene1)], x='PC1', y='PC2', alpha=1, color=gene_colors[gene1], ax=axis, s=15)
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['bottom'].set_visible(False)
    axis.set_xlim([np.floor(data.PC1.min()),np.ceil(data.PC1.max())])
    axis.set_ylim([np.floor(data.PC2.min()),np.ceil(data.PC2.max())])
    axis.set_ylabel(gene1+" vs "+gene2, fontsize=labelsize, weight='bold')
    axis.set_xlabel("Score: " + "%.2f" % score, fontsize=labelsize)
    axis.tick_params(axis='both', which='major', labelsize=0)
    axis.tick_params(axis='both', which='minor', labelsize=0)

def plot_biomarker_value_distribution(axis, dataset, genes, biomarker, biomarker_text):
    from matplotlib.ticker import FormatStrFormatter
    
    axis = sns.pointplot(data=dataset, x=biomarker, y="gene", join=False, dodge=.8 - .8 / 8, palette="viridis", scale=0.5,
                         estimator=np.median, errorbar=("pi",50), capsize=0.5, ax=axis, order=genes)
    
    axis.set_facecolor("white")
    axis.set_title(biomarker_text, fontsize=labelsize, fontweight="bold", pad=2)
    axis.title.set_size(labelsize)
    axis.set_ylabel("",fontsize=labelsize)
    axis.set_xlabel("",fontsize=labelsize)
    axis.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axis.set_yticks(ticks=list(range(len(genes))), labels=genes, size=labelsize, fontweight="bold")
    axis.set_xticklabels([str(round(float(label), 2)) for label in axis.get_xticks()], weight='bold')
    axis.tick_params(labelsize=labelsize)
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_visible(False)
    
    lines = axis.lines
    for line in lines:
        line.set_linewidth(0.5)
    return axis

def plot_mwu_results(axis, mwu, genes, biomarker):
    mwu_bm  = mwu[(mwu.biomarker == biomarker)]
    mwu_bm  = mwu_bm.pivot(index="gene1", columns="gene2", values="pvalue")
    mwu_bm  = mwu_bm[genes]
    mwu_bm  = mwu_bm.reindex(genes)

    bounds  = [0, 0.001, 0.01, 0.05, 1]
    heat_c  = ['seagreen', 'mediumseagreen', 'lightgreen', 'white']
    heat_n  = BoundaryNorm(bounds, ncolors=len(heat_c))
    val_min = 0
    val_max = 0.05

    annot = pd.DataFrame(columns=genes, index=genes)
    for g1 in genes:
        for g2 in genes:
            pval = float(mwu_bm.loc[g1,g2])
            if((pval > 0.01) & (pval <= 0.05)):
                pval = 2
            elif((pval > 0.001) & (pval <= 0.01)):
                pval = 3
            elif(pval <= 0.001):
                pval = 4
            pval = format(pval, '.2f')
            annot.at[g1, g2] = pval

    for g1 in genes:
        #annot.loc[annot[g1] <  "2.000", g1] = ""
        annot.loc[annot[g1] == "2.00", g1] = "*"
        annot.loc[annot[g1] == "3.00", g1] = "**"
        annot.loc[annot[g1] == "4.00", g1] = "***"
    
    annot = annot.reindex(genes)
        
    axis = sns.heatmap(mwu_bm, vmin=val_min, vmax=val_max, cmap=heat_c, norm = heat_n , annot=annot, annot_kws={"fontsize":labelsize}, cbar=False, fmt="", ax=axis)
    
    
    axis.tick_params(labelsize=labelsize)
    axis.set_title("Mann-Whitney U Test Corrected P-Values", fontweight="bold", pad=2)
    axis.title.set_size(labelsize)
    axis.set_yticks([])
    axis.set_xlabel('', fontsize=labelsize)
    axis.set_ylabel('', fontsize=labelsize)
    axis.set_facecolor("white")
    axis.set_xticklabels(genes, rotation = 90, weight='bold')

    for _, spine in axis.spines.items():
        spine.set_visible(True)
        
    return axis


def plot_oscillations(dataset, gene_factor, axis):

    osc_non    = np.sum(dataset[dataset.gene == gene_factor][['delta_band_oscillatory', 'theta_band_oscillatory','alpha_band_oscillatory', 'beta_band_oscillatory','gamma_band_oscillatory']].sum(axis=1) == 0)
    osc_delta  = np.sum(dataset[dataset.gene == gene_factor][['delta_band_oscillatory']].sum(axis=1) == 1)
    osc_theta  = np.sum(dataset[dataset.gene == gene_factor][['theta_band_oscillatory']].sum(axis=1) == 1)
    osc_alpha  = np.sum(dataset[dataset.gene == gene_factor][['alpha_band_oscillatory']].sum(axis=1) == 1)
    osc_beta   = np.sum(dataset[dataset.gene == gene_factor][['beta_band_oscillatory']].sum(axis=1) == 1)
    osc_gamma  = np.sum(dataset[dataset.gene == gene_factor][['gamma_band_oscillatory']].sum(axis=1) == 1)
    neuron_cnt = len(dataset[dataset.gene == gene_factor])

    size = 0.10
    axis.pie([neuron_cnt-osc_non, osc_non  ]  , radius=1-size*5, colors=["indigo","whitesmoke"], wedgeprops=dict(width=size, edgecolor='w'), startangle=90)
    axis.pie([osc_delta, neuron_cnt-osc_delta], radius=1-size*4, colors=["midnightblue","whitesmoke"], wedgeprops=dict(width=size, edgecolor='w'), startangle=90)
    axis.pie([osc_theta, neuron_cnt-osc_theta], radius=1-size*3, colors=["teal","whitesmoke"], wedgeprops=dict(width=size, edgecolor='w'), startangle=90)
    axis.pie([osc_alpha, neuron_cnt-osc_alpha], radius=1-size*2, colors=["seagreen","whitesmoke"], wedgeprops=dict(width=size, edgecolor='w'), startangle=90)
    axis.pie([osc_beta , neuron_cnt-osc_beta] , radius=1-size*1, colors=["yellowgreen","whitesmoke"], wedgeprops=dict(width=size, edgecolor='w'), startangle=90)
    axis.pie([osc_gamma, neuron_cnt-osc_gamma], radius=1-size*0, colors=["goldenrod","whitesmoke"], wedgeprops=dict(width=size, edgecolor='w'), startangle=90)
    axis.text(0., 0., gene_factor, horizontalalignment='center', verticalalignment='center', size=labelsize, fontweight="bold")


def heatmap(x, y, size, color, pvalues,  ax, size_scale = 10, lw = 0.5, palette = 'coolwarm'):
    
    n_colors = 256 # Use 256 colors for the diverging color palette
    palette  = sns.color_palette(palette, as_cmap=False, n_colors=n_colors) # Create the palette
    color_min, color_max = [-1, 1] # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
        ind          = int(val_position * (n_colors - 1)) # target index in the color palette
        return palette[ind]
    
    # Mapping from column names to integer coordinates
    x_labels   = [v for v in x.unique()]
    y_labels   = [v for v in y.unique()]
    x_to_num   = {p[1]:p[0] for p in enumerate(x_labels)} # Create a dict where the key is the name of the columns and index is the value
    y_to_num   = {p[1]:p[0] for p in enumerate(y_labels)} # Same as above

    mask_sign  = pvalues<0.05
    colors_vec = [value_to_color(v) for v in color.values]

    ax.scatter(x=x.map(x_to_num), # Use mapping for x
               y=y.map(y_to_num), # Use mapping for y
               s=size * size_scale, # Vector of square sizes, proportional to size parameter
               marker='s', # Use square as scatterplot marker
               edgecolors='k',
               linewidths= lw*mask_sign,
               c=colors_vec)

    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=90, fontsize=5)
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels, fontsize=5)
    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5]) 
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    return ax


def plot_corr_matrix(dataset, features, ax,  palette='coolwarm', p_adjust=''):
    corr = utils_statistics.measure_spearman_corr(dataset, features, p_adjust)
    ax   = heatmap(x=corr['x'], y=corr['y'], size=corr['value'].abs(), color = corr['value'],
                   ax=ax, size_scale=25, pvalues=corr['p'].values, palette = palette)
    return ax, corr
    
def plot_corr_line(dataset, feat_x, feat_y, size, ax):
    ax = sns.regplot(data=dataset, x=feat_x, y=feat_y, scatter=False, ax=ax)
    ax = sns.scatterplot(data=dataset[dataset.gene=="AOPEP"], x=feat_x, y=feat_y, color=gene_colors["AOPEP"], s=size, ax=ax)
    ax = sns.scatterplot(data=dataset[dataset.gene=="GNAL"], x=feat_x, y=feat_y, color=gene_colors["GNAL"], s=size, ax=ax)
    ax = sns.scatterplot(data=dataset[dataset.gene=="KMT2B"], x=feat_x, y=feat_y, color=gene_colors["KMT2B"], s=size, ax=ax)
    ax = sns.scatterplot(data=dataset[dataset.gene=="PANK2"], x=feat_x, y=feat_y, color=gene_colors["PANK2"], s=size, ax=ax)
    ax = sns.scatterplot(data=dataset[dataset.gene=="PLA2G6"], x=feat_x, y=feat_y, color=gene_colors["PLA2G6"], s=size, ax=ax)
    ax = sns.scatterplot(data=dataset[dataset.gene=="SGCE"], x=feat_x, y=feat_y, color=gene_colors["SGCE"], s=size, ax=ax)
    ax = sns.scatterplot(data=dataset[dataset.gene=="THAP1"], x=feat_x, y=feat_y, color=gene_colors["THAP1"], s=size, ax=ax)
    ax = sns.scatterplot(data=dataset[dataset.gene=="TOR1A"], x=feat_x, y=feat_y, color=gene_colors["TOR1A"], s=size, ax=ax)
    ax = sns.scatterplot(data=dataset[dataset.gene=="VPS16"], x=feat_x, y=feat_y, color=gene_colors["VPS16"], s=size, ax=ax)
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('none') 
    ax.yaxis.set_ticks_position('none') 
    return ax