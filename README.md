# Analysis Scripts for "Spiking patterns in the globus pallidus highlight convergent neural dynamics across diverse genetic dystonia syndromes"

---

## **Overview**

This repository contains Python scripts and associated data processing tools used in our study on the neurophysiological characterization of genetic dystonia patients undergoing bilateral GPi-DBS surgery. These tools were developed to analyze and visualize neural dynamics, perform statistical tests, stratify genes, and build machine learning models.

The study's methodology and analysis cover a range of topics:
- Preprocessing of spiking neural data
- Statistical and machine learning analyses
- Gene stratification via thresholding and distance metrics

By sharing this repository, we aim to support reproducibility, foster collaboration, and advance research in the field of dystonia and DBS.

---

## **Contents**

1. **Data Processing**:
   - For the spiking feature extraction toolbox please visit:  [Spike Feature Generator Matlab Toolbox](https://github.com/ahmetofficial/Spike-Feature-Generator)

2. **Statistical Analyses**:
   - Tools for comparing neural features among genetic subgroups.
   - Implements Kruskal‒Wallis, Mann‒Whitney U, chi-square, and Fisher’s Exact tests with Holm–Bonferroni correction.

3. **Principal Component Analysis**:
   - PCA scripts to analyze and visualize the variance structure of neural data.

4. **Gene Stratification**:
   - Frameworks for stratifying genes based on neural activity dynamics using thresholding and distance-based clustering techniques.

5. **Machine Learning Pipelines**:
   - Code for one-vs-one (OVO) and one-vs-rest (OVR) classification pipelines for neural decoding.

6. **Visualization Tools**:
   - Scripts for generating dendrograms, clustering visualizations, and Supplementary Figures.

---

## **Usage**

### **Prerequisites**
- Python (>=3.7)

