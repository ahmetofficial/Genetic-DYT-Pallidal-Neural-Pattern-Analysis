import itertools
import pandas as pd;
import numpy as np;
import random
from os.path import exists
import pickle 

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from sklearn.utils import shuffle
import scipy.stats as st
import scipy.stats

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks 

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC

import utils_hyperparameter_optimizer

def model_performance(X, Y, model, resampling_type, labels):
        
    k_fold     = 5
    skf        = StratifiedKFold(n_splits=k_fold, random_state=0,  shuffle=True)
    acc_scores = []
    auc_scores = []
    f1_scores  = []
    cm         = []
    cr         = []
        
    for train_index, test_index in skf.split(X, Y):
            
        X_train, X_test  = X.iloc[train_index], X.iloc[test_index] 
        y_train, y_test  = Y.iloc[train_index], Y.iloc[test_index] 
           
        X_train, y_train = balance_dataset(X_train, y_train, resampling_type)

        clf              = model.fit(X_train, y_train)
        y_pred           = clf.predict(X_test)          
        y_pred_prob      = clf.predict_proba(X_test)
            
        acc              = balanced_accuracy_score(y_test, y_pred)
        f1               = f1_score(y_test, y_pred, average="weighted")
            
        try:
            auc              = roc_auc_score(y_test, y_pred_prob, multi_class="ovr", average="weighted")                
        except:
            auc              = roc_auc_score(y_test, y_pred_prob[:, 1], average="weighted")
                           
        acc_scores.append(acc)
        auc_scores.append(auc)
        f1_scores.append(f1)
        cm.append(confusion_matrix(y_test, y_pred))
        cr.append(classification_report(y_test, y_pred, labels = labels, output_dict= True))

    cr                     = {}
    cr["auc"]              = {}
    cr["auc"]["mean"]      = np.mean(auc_scores)
    cr["auc"]["std"]       = np.std(auc_scores)
    cr["accuracy"]         = {}
    cr["accuracy"]["mean"] = np.mean(acc_scores)
    cr["accuracy"]["std"]  = np.std(acc_scores)
    cr["f1"]               = {}
    cr["f1"]["mean"]       = np.mean(f1_scores)
    cr["f1"]["std"]        = np.std(f1_scores)
        
        
    condensed_cm = cm[0]
    for i in range(len(cm)-1):
        condensed_cm = condensed_cm + cm[i+1]
        
    return condensed_cm, cr
    
        

def balance_dataset(X, Y, resampling_type):
        
    np.random.seed(seed=0)
        
    if(resampling_type == "up"):
        try:
            X_resampled, Y_resampled = SMOTE().fit_resample(X, Y)
        except:
            print("SMOTE algorithm cannot work with current fold...")
            X_resampled, Y_resampled = RandomOverSampler(random_state=0).fit_resample(X, Y)
    else:
        try:
            X_resampled, Y_resampled = TomekLinks().fit_resample(X, Y)
        except:
            print("TomekLinks algorithm cannot work with current fold...")
            X_resampled, Y_resampled = RandomUnderSampler(random_state=0).fit_resample(X, Y)

    X_resampled = X_resampled.reset_index(drop=True)
    Y_resampled = Y_resampled.reset_index(drop=True)
        
    return X_resampled, Y_resampled
    

def classification_report_summary(cr, number_of_labels, k_fold):
        
    metrics          = {} 
    precision_values = {}
    recalls_values   = {}
    f1_values        = {}
    support_values   = {}
    accuracy         = []
    f1               = []
        
    for i in range(number_of_labels):
        metrics[str(i)]              = {}
        metrics[str(i)]["precision"] = {}
        metrics[str(i)]["recall"]    = {}
        metrics[str(i)]["f1"]        = {}
        metrics[str(i)]["support"]   = {}
            
        precision_values[str(i)]     = []
        recalls_values[str(i)]       = []
        f1_values[str(i)]            = []
        support_values[str(i)]       = []
        
    for k in range(k_fold):
        for l in range(number_of_labels):
            precision_values[str(l)].append(cr[k][str(l)]["precision"])
            recalls_values[str(l)].append(cr[k][str(l)]["recall"])   
            f1_values[str(l)].append(cr[k][str(l)]["f1-score"])
            support_values[str(l)].append(cr[k][str(l)]["support"])
        f1.append(cr[k]["weighted avg"]["f1-score"])
        
    for l in range(number_of_labels):
        metrics[str(l)]["precision"]["mean"] = np.mean(precision_values[str(l)])
        metrics[str(l)]["precision"]["std"]  = np.std(precision_values[str(l)])
        metrics[str(l)]["recall"]["mean"]    = np.mean(recalls_values[str(l)])
        metrics[str(l)]["recall"]["std"]     = np.std(recalls_values[str(l)])
        metrics[str(l)]["f1"]["mean"]        = np.mean(f1_values[str(l)])
        metrics[str(l)]["f1"]["std"]         = np.std(f1_values[str(l)])
        metrics[str(l)]["support"]           = np.mean(support_values[str(l)])
            
    return metrics
    
###################################################################################################################################################
###################################################################################################################################################
# CONFIDENCE INTERVAL #############################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
    


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h
    

def get_confidence_interval(X, Y, models, selected_models, feature, num_epoch = 100):
    
    acc_scores = []
    auc_scores = []
    f1_scores  = []

    for i in range(num_epoch):
            
        estimator_list = [models[x] for x in selected_models]
        estimator_list = tuple(zip(selected_models, estimator_list)) 
        model          = VotingClassifier(estimators=estimator_list,
                                          voting='soft',
                                          verbose=False)

        Y_shuffled     = shuffle(Y.copy())
        Y_shuffled     = Y_shuffled.reset_index(drop=True)
        X_shuffled     = shuffle(X.copy())
        X_shuffled     = X_shuffled.reset_index(drop=True)
        X_shuffled     = pd.DataFrame(X_shuffled[feature])
            
        if(type(feature) == str): #single feature case
            acc, auc, f1 = model_performance_single_feature(X_shuffled, Y_shuffled, model, resampling_type = "up")
            acc_scores.append(acc)
            auc_scores.append(auc)
            f1_scores.append(f1)
        else: # the whole dataset case
            acc, auc, f1 = model_performance_single_feature(X_shuffled, Y_shuffled, model, resampling_type = "up")
            acc_scores.append(acc)
            auc_scores.append(auc)
            f1_scores.append(f1)
        
    mean_acc, lower_acc, upper_acc = mean_confidence_interval(acc_scores, confidence=0.95)
    acc_CI                         = (lower_acc, upper_acc)
    mean_auc, lower_auc, upper_auc = mean_confidence_interval(auc_scores, confidence=0.95)
    auc_CI                         = (lower_auc, upper_auc)
    mean_f1 , lower_f1 , upper_f1  = mean_confidence_interval(f1_scores, confidence=0.95)
    f1_CI                          = (lower_f1, upper_f1)

    return acc_CI, auc_CI, f1_CI
    
###################################################################################################################################################
###################################################################################################################################################
# GET CLASSIFIERS RESULTS #########################################################################################################################
###################################################################################################################################################
###################################################################################################################################################
    

def get_rf_results(parameters, X, Y, resampling_type, labels):
    model      = RandomForestClassifier(n_estimators      = 500,
                                        criterion         = parameters["criterion"],
                                        max_depth         = parameters["max_depth"],
                                        max_leaf_nodes    = parameters["max_leaf_nodes"],
                                        min_samples_leaf  = parameters["min_samples_leaf"],
                                        min_samples_split = parameters["min_samples_split"],
                                        random_state      = parameters["random_state"])

    cm, cr     = model_performance(X, Y, model, resampling_type, labels)
            
    #print('Mean Accuracy  : %.3f (%.3f)' % (cr["accuracy"]["mean"], cr["accuracy"]["std"]))
    #print('Mean AUC       : %.3f (%.3f)' % (cr["auc"]["mean"],      cr["auc"]["std"]))
    #print('Mean F1        : %.3f (%.3f)' % (cr["f1"]["mean"],       cr["f1"]["std"]))
    return model, cm, cr
    
def get_gbc_results(parameters, X, Y, resampling_type, labels):
    model      = GradientBoostingClassifier(random_state      = 0,
                                            verbose           = 0,
                                            n_estimators      = parameters.get('n_estimators'),
                                            learning_rate     = parameters.get('learning_rate'),
                                            min_samples_split = parameters.get('min_samples_split'),
                                            min_samples_leaf  = parameters.get('min_samples_leaf'),
                                            max_depth         = parameters.get('max_depth'),
                                            max_leaf_nodes    = parameters.get('max_leaf_nodes'))

    cm, cr = model_performance(X, Y, model, resampling_type, labels)
            
    #print('Mean Accuracy  : %.3f (%.3f)' % (cr["accuracy"]["mean"], cr["accuracy"]["std"]))
    #print('Mean AUC       : %.3f (%.3f)' % (cr["auc"]["mean"],      cr["auc"]["std"]))
    #print('Mean F1        : %.3f (%.3f)' % (cr["f1"]["mean"],       cr["f1"]["std"]))
    return model, cm, cr 

def get_vc_results(models, selected_models, X, Y, resampling_type, labels):
        
    estimator_list = [models[x] for x in selected_models]
    estimator_list = list(zip(selected_models, estimator_list)) 
    model_vc       = VotingClassifier(estimators=estimator_list,
                                      voting='soft',
                                      verbose=False)

    cm, cr         = model_performance(X, Y, model_vc, resampling_type, labels) 
        
    #print('Mean Accuracy  : %.3f (%.3f)' % (cr["accuracy"]["mean"], cr["accuracy"]["std"]))
    #print('Mean AUC       : %.3f (%.3f)' % (cr["auc"]["mean"],      cr["auc"]["std"]))
    #print('Mean F1        : %.3f (%.3f)' % (cr["f1"]["mean"],       cr["f1"]["std"]))
    return model_vc, cm, cr

def get_optimized_parameters(X_data, Y_data, classification_mode, model, gene_pair, params_extra=None):
    
    parameters_path = r'hyperparameters/' + classification_mode + '/' + model + '_' + gene_pair
    
    if(exists(parameters_path) == False):
        if(model == 'rf'):
            parameters = utils_hyperparameter_optimizer.rf_optimizator(X_data, Y_data)
        elif(model == 'gbc'):
            parameters = utils_hyperparameter_optimizer.gbc_optimizator(X_data, Y_data, params_extra)
        pickle.dump(parameters, open(parameters_path, 'wb'))

    parameters = pickle.load(open(parameters_path, "rb"))
    return parameters
    
def get_model_output(X_data, Y_data, parameters, model, data_type, classes, performance, model_all=np.NaN, model_pair=np.NaN):
    
    if(model == 'rf'):
        model_trained, cm, cr = get_rf_results(parameters, X_data, Y_data, resampling_type="up", labels=[1,2])
    elif(model == 'gbc'):
        model_trained, cm, cr = get_gbc_results(parameters, X_data, Y_data, resampling_type="up", labels=[1,2])
    elif(model == 'vc'):
        model_trained, cm, cr = get_vc_results(model_all, model_pair, X_data, Y_data, resampling_type="up", labels=[1,2])

    row = {"model":model, "analysis": data_type, "balanced_accuracy":cr["accuracy"]["mean"], "weighted_auc":cr["auc"]["mean"], "weighted_f1":cr["f1"]["mean"]}
    performance.loc[len(performance)] = row
    
    return model_trained, performance

def get_model_confusion_matrix(X_data, Y_data, parameters, model, data_type, classes, performance, model_all=np.NaN, model_pair=np.NaN):
    model_trained, cm, cr = get_vc_results(model_all, model_pair, X_data, Y_data, resampling_type="up", labels=[1,2])
    return cm
    
def get_model_CI(X_data, Y_data, parameters, model, iteration, model_all=np.NaN, model_pair=np.NaN):
    
    auc_list   = []
    acc_list   = []
    f1_list    = []
    Y_shuffled = Y_data.copy()
    
    for i in range(iteration):
        
        if(np.mod(i+1,10)==0):
            print("------> iteration : " + str((i+1)))
        
        Y_shuffled = shuffle(Y_shuffled)
        
        if(model == 'rf'):
            model_trained, cm, cr = get_rf_results(parameters, X_data, Y_shuffled, resampling_type="up", labels=[1,2])
        elif(model == 'gbc'):
            model_trained, cm, cr = get_gbc_results(parameters, X_data, Y_shuffled, resampling_type="up", labels=[1,2])
        elif(model == 'vc'):
            model_trained, cm, cr = get_vc_results(model_all, model_pair, X_data, Y_shuffled, resampling_type="up", labels=[1,2])
            
        acc_list.append(cr["accuracy"]["mean"])
        auc_list.append(cr["auc"]["mean"])
        f1_list.append(cr["f1"]["mean"])

    acc_threshold = np.percentile(acc_list, 95)
    auc_threshold = np.percentile(auc_list, 95)
    f1_threshold  = np.percentile(f1_list, 95)

    return acc_threshold, auc_threshold, f1_threshold
