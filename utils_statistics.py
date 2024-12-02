import pandas as pd
import numpy as np
import math
from scipy import stats
import logging
import scikit_posthocs as sp
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import calinski_harabasz_score


# -------------------------------------------------------------------------------------------------------------------------------------------------
# - NORMALITY TEST --------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------
    

# input parameters
#      population : a Dataframe contains the biomarker values of a specific population 
#      features   : a string lists contains the names of the non-parametric biomarkers
# output parameters
#      gaussian_features : the list of biomarkers that follows normal distribution for the given population
# description
#      The method takes a Dataframe that the biomarker values of a specific neural population. To conduct any population-wise
#      difference analyses, the normality assumption should be validated for the biomarker value distribution for the population. 
#      Hence, we used the Shapiro-Wilk normality test which is a popular normality test that gives good results for even low population sizes.
#      We set the significance value as 0.05.
def shapiro_wilk_normality_test(population, features):
# Shapiro, S. S. & Wilk, M.B (1965). An analysis of variance test for normality (complete samples), Biometrika, Vol. 52, pp. 591-611.
    
    gaussian_features = []
    for feature_index in range(len(features)):
        feature    = features[feature_index]
        value_dist = population[~population[feature].isnull()][feature] # getting rid of nan values since they are affecting the test results
            
        if(len(value_dist) > 3): # the condition for the shapiro wilk normality test
            # we accept the value distribution as non-gaussian when N_pop < 3
            response   = stats.shapiro(value_dist)
            if(response.pvalue > 0.5):
                gaussian_features.append(feature)
            
    return gaussian_features
    
# -------------------------------------------------------------------------------------------------------------------------------------------------
    

# input parameters
#      populations : a list contains multiple Dataframe structures that corresponds to the each population (with multiple neural biomarkers)
#                    that we involved to our study
#      features    : the list of biomarkers that we want to conduct normality test based on Shapiro-Wilk test
# output parameters
#      normal_features     : the list of normally distributed biomarkers across all populations.
#      non_normal_features : the list of non-normally distributed biomarkers across all populations.
def normality_analysis(populations, features):
    normal_features = features
    for population in populations:
        normal_features_of_population = shapiro_wilk_normality_test(population, features)
        normal_features               = list(set(normal_features) & set(normal_features_of_population))

    non_normal_features   = list(set(features) - set(normal_features))  

    return normal_features, non_normal_features

# -------------------------------------------------------------------------------------------------------------------------------------------------
# - KRUSKALL-WALLIS TEST --------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------
    
def multiple_comparison_correction(group, method):
    group['pvalue'] = multipletests(group['pvalue'], method=method)[1]
    return group
    
# input parameters
#      H : the results of the Kruskall-Wallis test (H value)
#      k : number of populations that was considered for the Kruskall-Wallis Test
#      n : the size of the all population.
# output parameters
#      eta : estimation of eta square effect size for the conducted Kruskall-Wallis Test
def ranked_eta_square(H, k, n):
    # Tomczak, M.; Tomczak, E. The need to report effect size estimates revisited. 
    # An overview of some recommended measures of effect size. Trends Sport. Sci. 2014, 1, 19–25.

    # citation for effect size interpretation
    # Lakens D. Calculating and reporting effect sizes to facilitate cumulative science: a practical primer for t-tests and ANOVAs. Front Psychol. 
    # 2013 Nov 26;4:863. doi: 10.3389/fpsyg.2013.00863. PMID: 24324449; PMCID: PMC3840331.
        
    # small  : 0.01 - 0.06
    # medium : 0.06 – 0.14
    # large  : >0.14
        
    eta = (H - k + 1) / (n - k)
    if(eta < 0.06):
        eta_interpretation = "small"
    elif((eta>= 0.06) & (eta < 0.14)):
        eta_interpretation = "medium"
    else:
        eta_interpretation = "large"
    return eta, eta_interpretation
    
# -------------------------------------------------------------------------------------------------------------------------------------------------
    
# input parameters
#      H : the results of the Kruskall-Wallis test (H value)
#      n : the size of the all population.
# output parameters
#      epsilon : estimation of epsilon square effect size for the conducted Kruskall-Wallis Test
def ranked_epsilon_square(H, n):
    # Tomczak, M.; Tomczak, E. The need to report effect size estimates revisited. 
    # An overview of some recommended measures of effect size. Trends Sport. Sci. 2014, 1, 19–25.

    epsilon = H / ((n**2 - 1) * (n + 1))
    return epsilon
    
# -------------------------------------------------------------------------------------------------------------------------------------------------
    
# input parameters
#      eta : estimated value of eta square effect size for the conducted Kruskall-Wallis Test
# output parameters
#      cohens_d : the estimation of Cohen's d effect size for the conducted Kruskall-Wallis Test based on eta square transformation

def ranked_eta_square_to_cohens_d(eta):
    # Fritz CO, Morris PE, Richler JJ. Effect size estimates: current use, calculations, and interpretation. J Exp Psychol Gen. 
    # 2012 Feb;141(1):2-18. doi: 10.1037/a0024338. Epub 2011 Aug 8. Erratum in: J Exp Psychol Gen. 2012 Feb;141(1):30. PMID: 21823805.
        
    # citation for effect size interpretation
    # Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences. New York, NY: Routledge Academic.
        
    # small  : 0.2 - 0.5
    # medium : 0.5 – 0.8
    # large  : >0.8

    try:
        cohens_d = 2 * math.sqrt(abs(eta)) / math.sqrt(1 - abs(eta))
    except:
        print(eta)
    if(cohens_d < 0.5):
        cohens_d_interpretation = "small"
    elif((cohens_d>= 0.5) & (cohens_d < 0.8)):
        cohens_d_interpretation = "medium"
    elif(cohens_d>= 0.8):
        cohens_d_interpretation = "large"
    return cohens_d, cohens_d_interpretation
    
    
# -------------------------------------------------------------------------------------------------------------------------------------------------
    
# input parameters
#      populations : a list contains multiple Dataframe structures that corresponds to the each population (with multiple neural biomarkers)
#                    that we involved to our study
#      features    : the list of biomarkers that we want to conduct median based population comparison via nonparametric test: Kruskall-Wallis Test
#      N           : the total number of elements across all populations
# output parameters
#      results : a Dataframe structure that contains the results of non-parametric population tests with their corresponding effect size.
def nonparametric_population_test(dataset, features):

    results = pd.DataFrame(columns = ["biomarker", "H", "pvalue", "k", "N", "eta_square", "eta_square_interpretation", 
                                      "epsilon_square", "cohens_d", "cohens_d_interpretation"])
    for biomarker in features:

        genes            = dataset[dataset[biomarker].notna()].gene.unique()
        biomarker_values = []
        for gene in genes:
            biomarker_values.append(dataset[dataset[biomarker].notna()][dataset["gene"] == gene][biomarker].to_list())

        N          = sum([len(listElem) for listElem in biomarker_values])
        k          = len(biomarker_values)
        try:
            H, p_value = stats.kruskal(*biomarker_values, nan_policy="omit")   
        except:
            row     = {"biomarker": biomarker, "H":H, "pvalue":p_value, "k":k, "N":N, "eta_square": np.nan, "eta_square_interpretation": np.nan, 
                        "epsilon_square": np.nan,"cohens_d": np.nan, "cohens_d_interpretation": np.nan}

        if(np.isnan(H)):
            row     = {"biomarker": biomarker, "H":H, "pvalue":p_value, "k":k, "N":N, "eta_square": np.nan, "eta_square_interpretation": np.nan, 
                        "epsilon_square": np.nan,"cohens_d": np.nan, "cohens_d_interpretation": np.nan}
        elif(p_value > 0.05):
            row     = {"biomarker": biomarker, "H":H, "pvalue":p_value, "k":k, "N":N, "eta_square": 0, "eta_square_interpretation": np.nan, 
                        "epsilon_square": 0,"cohens_d": 0.025, "cohens_d_interpretation": np.nan}
        else:
            eta, eta_interpretation           = ranked_eta_square(H, k, N)
            epsilon                           = ranked_epsilon_square(H, N)
            cohens_d, cohens_d_interpretation = ranked_eta_square_to_cohens_d(eta)
            row     =  {"biomarker": biomarker, "H":H, "pvalue":p_value, "k":k, "N":N, "eta_square": eta, "eta_square_interpretation": eta_interpretation,
                        "epsilon_square": epsilon, "cohens_d": cohens_d, "cohens_d_interpretation": cohens_d_interpretation}
        results.loc[len(results)] = row


    # apply p-value correction
    _, corrected_p_values, _, _ = multipletests(results.pvalue.values, method='holm')
    results["pvalue"]           = corrected_p_values  
    # after corrected p-values reset effect sizes if corrected p-value>0.05
    results.loc[results['pvalue'] > 0.05, ['cohens_d', 'cohens_d_interpretation']] = [0.025, np.nan]
    return results
    
# -------------------------------------------------------------------------------------------------------------------------------------------------
# - MANN-WHITNEY U TEST ---------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------
    

# input parameters
#      N1 : the size of the first population 
#      N2 : the size of the second population 
#      U1 : U value of the first population
# output parameters
#      z  : the z score of the U value of the Mann- Whitney U Test
#      r  : the r effect size
#      U  : the statistic of Mann-Whitney Test

def r_effect_size(N1, N2, U1):
    # Fritz CO, Morris PE, Richler JJ. Effect size estimates: current use, calculations, and interpretation. J Exp Psychol Gen. 
    # 2012 Feb;141(1):2-18. doi: 10.1037/a0024338. Epub 2011 Aug 8. Erratum in: J Exp Psychol Gen. 2012 Feb;141(1):30. PMID: 21823805.

    # interpretation is universal
    # small  : 0.10 - 0.30
    # medium : 0.30 – 0.50
    # large  : >0.50
        
    U2 = N1*N2 - U1
    U  = min(U1, U2)
    z  = (U - N1*N2/2 + 0.5) / np.sqrt(N1*N2*(N1+N2+1)/ 12)
    r  = np.abs(z) / np.sqrt(N1+N2)
        
    if(r < 0.30):
        r_interpretation = "small"
    elif((r>= 0.30) & (r < 0.50)):
        r_interpretation = "medium"
    elif(r>= 0.50):
        r_interpretation = "large"
            
    return z, r, r_interpretation, U, U2
    
# -------------------------------------------------------------------------------------------------------------------------------------------------

# input parameters
#      r : the r effect size
# output parameters
#      eta_square : the transformed eta square value from r effect size
def r_to_eta_square(z, N):
    # Tomczak, M.; Tomczak, E. The need to report effect size estimates revisited. 
    # An overview of some recommended measures of effect size. Trends Sport. Sci. 2014, 1, 19–25.
        
    # citation for effect size interpretation
    # Lakens D. Calculating and reporting effect sizes to facilitate cumulative science: a practical primer for t-tests and ANOVAs. Front Psychol. 
    # 2013 Nov 26;4:863. doi: 10.3389/fpsyg.2013.00863. PMID: 24324449; PMCID: PMC3840331.
        
    # small  : 0.01 - 0.06
    # medium : 0.06 – 0.14
    # large  : >0.14
        
    eta = z*z / (N)
    if(eta < 0.06):
        eta_interpretation = "small"
    elif((eta>= 0.06) & (eta < 0.14)):
        eta_interpretation = "medium"
    elif(eta>= 0.14):
        eta_interpretation = "large"
    return eta, eta_interpretation
    
# -------------------------------------------------------------------------------------------------------------------------------------------------

# input parameters
#      r : the r effect size
# output parameters
#      cohens_d : the transformed cohen's d value from r effect size
def r_to_cohens_d(r):
    # Borenstein, M., Hedges, L. V., Higgins, J. P. T., & Rothstein, H. R. (2009). Introduction to Meta-Analysis, 
    # Chapter 7: Converting Among Effect Sizes . Chichester, West Sussex, UK: Wiley. 
    # https://doi.org/10.1002/9780470743386.ch7

    # citation for effect size interpretation
    # Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences. New York, NY: Routledge Academic.
        
    # small  : 0.2 - 0.5
    # medium : 0.5 – 0.8
    # large  : >0.8

    cohens_d = 2*r / np.sqrt(1-r**2)
    if(cohens_d < 0.5):
        cohens_d_interpretation = "small"
    elif((cohens_d>= 0.5) & (cohens_d < 0.8)):
        cohens_d_interpretation = "medium"
    elif(cohens_d>= 0.8):
        cohens_d_interpretation = "large"
    return cohens_d, cohens_d_interpretation
    
# -------------------------------------------------------------------------------------------------------------------------------------------------

# input parameters
#      U1 : the U score of the first population
#      N1 : the size of the first population 
#      U2 : the U score of the second population
#      N2 : the size of the second population
# output parameters
#      rg : estimated glass biserial correlation effect size for the Mann-Whitney U Test
def glass_biserial_correlation(U1, N1, U2, N2):
    # Glass, G. V. (1965) A ranking variable analogue of biserial correlation: Implications for short-cut item analysis. 
    # Journal of Educational Measurement, 2(1), 91–95. DOI: 10.1111/j.1745-3984.1965.tb00396.x

    # citation for the interpretation of Cliff’s delta or rg (Glass's biserial correlation)
    # Vargha, A. and H.D. Delaney. A Critique and Improvement of the CL Common Language Effect Size Statistics of McGraw and Wong. 2000. 
    # Journal of Educational and Behavioral Statistics 25(2):101–132.
        
    # small  : 0.11 - 0.28
    # medium : 0.28 – 0.43
    # large  : 0.43

    R1 = U1 / N1 # mean rank for group 1
    R2 = U2 / N2 # mean rank for group 2
    rg = 2*(R1 - R2) / (N1 + N2)
        
    if(np.abs(rg) < 0.28):
        rg_interpretation = "small"
    elif((np.abs(rg)>= 0.28) & (np.abs(rg) < 0.43)):
        rg_interpretation = "medium"
    elif(np.abs(rg)>= 0.43):
        rg_interpretation = "large"
    return rg, rg_interpretation
    
# -------------------------------------------------------------------------------------------------------------------------------------------------

# input parameters
#      populations      : a list that contains multiple Dataframe structures that each of them corresponds to a specific population
#      features         : the list of biomarkers that we want to conduct nonparametric post adhoc test: Mann-Whitney U Test
#      population_names : a string list contains the name of the populations
# output parameters
#      result      : a Dataframe that contains results of Mann-Whitney U Test with multiple effect size definitions
def nonparametric_postadhoc_test(populations, features, population_names):
    logger = logging.getLogger('ftpuploader')
    result = pd.DataFrame(columns = ["biomarker", "gene1" , "gene2", "U", "pvalue", "zscore", "r", "r_interpretation", 
                                     "eta_square", "eta_interpretation", "cohens_d", "cohens_d_interpretation",
                                     "glass_biserial_correlation", "glass_biserial_correlation_interpretation"])

    for feature in features:
        for pop1_i in range(len(populations)):
            for pop2_i in range(len(populations)):
                pop1 = populations[pop1_i];
                pop2 = populations[pop2_i];
                N1   = len(pop1)
                N2   = len(pop2)
                N    = N1 + N2
                try:
                    U1, pvalue                        = stats.mannwhitneyu(pop1[feature], pop2[feature], nan_policy="omit")

                    z, r, r_interpretation, U, U2     = r_effect_size(N1, N2, U1)
                    eta, eta_interpretation           = r_to_eta_square(z, N)
                    cohens_d, cohens_d_interpretation = r_to_cohens_d(r)
                    rg, rg_interpretation             = glass_biserial_correlation(U1, N1, U2, N2)
                    row         = {"biomarker": feature, "gene1": population_names[pop1_i], "gene2":population_names[pop2_i], "U":U, "pvalue": pvalue, 
                                   "zscore": z, "r":r, "r_interpretation": r_interpretation, "eta_square": eta, "eta_interpretation":eta_interpretation,
                                   "cohens_d": cohens_d, "cohens_d_interpretation": cohens_d_interpretation, 
                                   "glass_biserial_correlation": rg, "glass_biserial_correlation_interpretation": rg_interpretation}              
                        
                    result.loc[len(result)] = row
                except Exception as e:
                    logger.error(feature + ' : ' + population_names[pop1_i] + ' : ' +  population_names[pop2_i] + ' ||| '+ str(e))
                    row         = {"biomarker": feature, "gene1": population_names[pop1_i], "gene2":population_names[pop2_i], "U":np.nan, "pvalue": np.nan, 
                                   "zscore": np.nan, "r":np.nan, "r_interpretation": np.nan, "eta_square": np.nan, "eta_interpretation":np.nan,
                                   "cohens_d": np.nan, "cohens_d_interpretation": np.nan,
                                   "glass_biserial_correlation": np.nan, "glass_biserial_correlation_interpretation": np.nan}

    # remove comparison between the same gene
    result = result[result.gene1 != result.gene2]
    # apply p-value correction based on gene-pairs
    result = result.groupby(['gene1','gene2']).apply(multiple_comparison_correction, "holm")
    result.reset_index(drop=True, inplace=True)
    # after corrected p-values reset effect sizes if corrected p-value>0.05
    result.loc[result['pvalue'] > 0.05, ['cohens_d', 'cohens_d_interpretation']] = [0.025, "small"]
    return result
    
    
def fisher_exact_test(dataframe, genes, features):
    
    fexact_results = pd.DataFrame(columns=("biomarker","gene1","gene2","test_type","pvalue","odds_ratio","odds_ratio_interpretation"))

    for feat in features:
        for gen1 in genes:
            for gen2 in genes:
                x              = pd.crosstab(index=dataframe['gene'], columns=dataframe[feat])
                con_table      = x.loc[[gen1, gen2]]

                res_ts         = stats.fisher_exact(con_table)
                odd_ts         = res_ts[0]
                pvalue_ts      = res_ts[1]
                res_os         = stats.fisher_exact(con_table, alternative="greater")
                odd_os         = res_os[0]
                pvalue_os      = res_os[1]
                    
                odd_interpret  = "small"
                    
                if(pvalue_ts<=0.05):
                    odd_ratio      = odd_os
                    if((odd_ratio < 1)):
                        odd_ratio = 1 / odd_ratio;
                    if((odd_ratio >=  2.48) & (odd_ratio <  4.27)):
                        odd_interpret = "medium"
                    elif((odd_ratio >=  4.27)):
                        odd_interpret = "large"
                        

                fexact_results.loc[len(fexact_results)] = {"biomarker":feat,"gene1":gen1,"gene2":gen2,"test_type":"two-sided",
                                                           "pvalue":pvalue_ts, "odds_ratio":odd_ts, "odds_ratio_interpretation": odd_interpret}
                fexact_results.loc[len(fexact_results)] = {"biomarker":feat,"gene1":gen1,"gene2":gen2, "test_type":"one-sided",
                                                           "pvalue":pvalue_os, "odds_ratio":odd_os, "odds_ratio_interpretation": odd_interpret}

    # remove comparison between the same gene
    fexact_results = fexact_results[fexact_results.gene1 != fexact_results.gene2]
    # correct p-values for gene-pairs seperately for test type (one-side or two sided)
    fexact_results = fexact_results.groupby(['gene1', 'gene2', 'test_type']).apply(multiple_comparison_correction, "holm") 
    fexact_results.reset_index(drop=True, inplace=True)
    # after corrected p-values reset effect sizes if corrected p-value>0.05
    fexact_results.loc[fexact_results['pvalue'] > 0.05, ['odds_ratio', 'odds_ratio_interpretation']] = [0.025, "small"]
    
    return fexact_results
    
def barnard_exact_test(dataframe, genes, features):
    
    fexact_results = pd.DataFrame(columns=("biomarker","gene1","gene2","test_type","pvalue"))

    for feat in features:
        for gen1 in genes:
            for gen2 in genes:
                x              = pd.crosstab(index=dataframe['gene'], columns=dataframe[feat])
                con_table      = x.loc[[gen1, gen2]]

                res_ts         = stats.barnard_exact(con_table)
                pvalue_ts      = res_ts.pvalue
                res_os         = stats.barnard_exact(con_table, alternative="greater")
                pvalue_os      = res_ts.pvalue

                fexact_results.loc[len(fexact_results)] = {"biomarker":feat,"gene1":gen1,"gene2":gen2,"test_type":"two-sided", "pvalue":pvalue_ts}
                fexact_results.loc[len(fexact_results)] = {"biomarker":feat,"gene1":gen1,"gene2":gen2, "test_type":"one-sided", "pvalue":pvalue_os}
    return fexact_results
    

def cramers_v_effect_size(chi2, N, R, C):
    
    #Cohen, J. (1988). Statistical power analysis for the behavioral sciences (2nd ed.). Hillsdale, NJ: Erlbaum. (p. 284)
    v                = np.sqrt(chi2/N) / np.sqrt(min(R-1, C-1))
    v_interpretation = "small"

    # for dof=9
    if((v<0.25) & (v>=0.15)):
        v_interpretation = "medium"
    elif((v>=0.25)):
        v_interpretation = "large"
    return v, v_interpretation

def chi_square_test(dataframe, features):

    chi2_results = pd.DataFrame(columns=("biomarker","pvalue","cramers_v", "cramers_v_interpretation"))

    for feat in features:
        x                      = pd.crosstab(index=dataframe['gene'], columns=dataframe[feat])
        chi2, pvalue, dof, xyz = stats.chi2_contingency(x)
        pvalue                 = multipletests(pvalue, alpha=0.05, method='holm')[1][0]
        R                      = len(x)
        C                      = len(x.columns)
        N                      = x.sum().sum()
        cramer_v, cramer_v_int = cramers_v_effect_size(chi2, N, R, C)
        chi2_results.loc[len(chi2_results)] = {"biomarker":feat, "pvalue":pvalue, "cramers_v":cramer_v, "cramers_v_interpretation": cramer_v_int}

    # apply p-value correction
    _, corrected_p_values, _, _ = multipletests(chi2_results.pvalue.values, method='holm')
    chi2_results["pvalue"] = corrected_p_values
    # after corrected p-values reset effect sizes if corrected p-value>0.05
    chi2_results.loc[chi2_results['pvalue'] > 0.05, ['cramers_v', 'cramers_v_interpretation']] = [0.025, np.nan]
    
    return chi2_results


def ch_score(X, Y, genes, group_mapping):
    calinski_harabasz = np.zeros([len(genes), len(genes)])
    for i in range(len(genes)):
        for j in range(len(genes)):
            gene1 = genes[i]
            gene2 = genes[j]
            if(i!=j):
                X_temp = X[(Y==gene1) | (Y==gene2)].values
                labels = Y[(Y==gene1) | (Y==gene2)].map(group_mapping)
                calinski_harabasz[i][j] = calinski_harabasz_score(X_temp, labels)
            else:
                calinski_harabasz[i][j] = 0
    calinski_harabasz = pd.DataFrame(data=calinski_harabasz, columns=genes, index=genes)
    return calinski_harabasz

def create_cross_tab(gene_data, feature):

    # get the cross tap [neurons do not present phenomenon, neurons do present phenomenon]
    ct           = {}
    for gene in gene_data.keys():
        ct[gene]  = gene_data[gene][feature].value_counts().reindex([0,1]).to_list()

    # get the fraction of neurons showing the phenomenon per gene
    ct_percentage = ct.copy()
    for gene in ct_percentage.keys():
        ct_percentage[gene] = ct_percentage[gene][1] / np.sum(ct_percentage[gene])*100
    ct_percentage = dict(sorted(ct_percentage.items(), key=lambda x: x[1], reverse=True))
    
    return ct, ct_percentage
    
def define_boundary(data, g1, g2, alternative):
    from scipy.stats import fisher_exact
    from scipy.stats import barnard_exact
    gup_t     = data[g1]
    gdown_t   = data[g2]
    iteration = gup_t[1]
    
    if(alternative == "less"):
        for i in range(iteration):
            gup_t_updated     = [gup_t[0]+i, gup_t[1]-i]
            table             = [gup_t_updated, gdown_t]
            statistic, pvalue = fisher_exact(table, alternative='less')
            if(pvalue >= 0.05):
                threshold = (gup_t_updated[1]-1) / np.sum(gup_t_updated) * 100
                break
    else:
        for i in range(iteration):
            gdown_t_updated   = [gdown_t[0]-i, gdown_t[1]+i]
            table             = [gup_t, gdown_t_updated]
            statistic, pvalue = fisher_exact(table, alternative='greater')
            if(pvalue >= 0.05):
                threshold = (gdown_t_updated[1]+1) / np.sum(gdown_t_updated) * 100
                break
        
    return threshold

def measure_spearman_corr(dataset, features, p_adjust = ''):
    from statsmodels.stats.multitest import fdrcorrection
    v = [] # Compute pvalues
    for y in features:
        for x in features:
            var1 = dataset[x].values
            var2 = dataset[y].values
            r,p = stats.spearmanr(var1,var2)
            v.append([x,y,r,p])
    corr = pd.DataFrame(data = v, columns = ['x','y','value','p'])

   # multiple test correction
    if p_adjust:
        features_included = features.copy()
        vec_names = []
        vec_p = []
        corr_corrected = corr.copy()

        for y in features:
            features_included.remove(y)
            for x in features_included:
                p = float(corr['p'][(corr['x']==x) & (corr['y']==y)])
                vec_p.append(p)
                vec_names.append((x,y))

        if p_adjust == 'FDR':
            _,pvalues = fdrcorrection(pvals = vec_p, alpha = 0.05, method = 'indep', is_sorted=False) # Benjamini/Hochberg correction
        elif p_adjust == 'bonferroni':
            pvalues = np.array(vec_p)*int(len(vec_p))
        else: 
            print('no correction applied: specify a valid method (FDR,bonferroni)')
            pvalues = np.array(vec_p)
            
        for ind, (x,y) in enumerate(vec_names):
            corr_corrected['p'][(corr_corrected['x']==x) & (corr_corrected['y']==y)] = pvalues[ind]
            corr_corrected['p'][(corr_corrected['x']==y) & (corr_corrected['y']==x)] = pvalues[ind]       
        return corr_corrected
    else:
        return corr

