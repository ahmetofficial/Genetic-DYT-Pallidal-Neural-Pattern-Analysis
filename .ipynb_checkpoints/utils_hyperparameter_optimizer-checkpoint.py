import itertools
import pandas as pd;
import numpy as np;

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.svm import SVC


def rf_optimizator(X, Y):
    parameters = {"random_state"      : [0], 
                  "criterion"         : ["entropy"],
                  "class_weight"      : ["balanced"],
                  "n_estimators"      : [50],
                  "max_depth"         : np.arange(0, 50 , 5)+1,
                  "min_samples_leaf"  : np.arange(0, 50 , 10)+1,
                  "min_samples_split" : np.arange(1, 50 , 10)+1,
                  "max_leaf_nodes"    : np.arange(1, 40 , 10)+1}
    clf = GridSearchCV(RandomForestClassifier(), parameters, scoring='balanced_accuracy', cv=5, n_jobs=20)
    clf.fit(X, Y)
    parameters = clf.best_params_
    return parameters
    

def gbc_optimizator(X, Y, params):
    parameters = {"random_state"      : [0],
                  "verbose"           : [0],
                  "n_estimators"      : [2000],
                  "learning_rate"     : [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25],
                  "max_depth"         : [params.get("max_depth")],
                  "min_samples_leaf"  : [params.get("min_samples_leaf")],
                  "min_samples_split" : [params.get("min_samples_split")],
                  "max_leaf_nodes"    : [32]}
    clf = GridSearchCV(GradientBoostingClassifier(), parameters, scoring='balanced_accuracy', cv=5, n_jobs=20)
    clf.fit(X, Y)
    parameters = clf.best_params_
    return parameters
