import numpy as np
import pandas as pd
import os
from itertools import permutations
from sklearn import tree, metrics, ensemble, neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from DATA.feature_labels import features
from classifier_models import *
from ast import literal_eval
from sklearn import tree
from sklearn import neighbors

def simple(x_train, x_test, y_train, y_test, max_depth = 17, min_samples_leaf = 1):
    model = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_leaf = min_samples_leaf)           #Generates a model
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    probs = model.predict_proba(x_test)
    results = pd.DataFrame(columns = ["y_true","y_pred","probs_0", "probs_1"])
    results["y_true"] = y_test; results["y_pred"] = pred; results["probs_0"] = probs[:,0]; results["probs_1"] = probs[:,1]
    results.to_csv("classification_results/simple_tree_results.csv")
    scores = score(results)
    return results, scores

def boosting(x_train, x_test, y_train, y_test, max_depth = 17, min_samples_leaf = 1, n_estimators = 17):
    enc = LabelEncoder()
    y_train = enc.fit_transform(y_train)
    y_test = enc.fit_transform(y_test)
    base_model = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_leaf = min_samples_leaf)           #Generates a model
    model = ensemble.AdaBoostClassifier(base_estimator = base_model, n_estimators = n_estimators, random_state=240624)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    probs = model.predict_proba(x_test)
    results = pd.DataFrame(columns = ["y_true","y_pred","probs_0", "probs_1"])
    results["y_true"] = y_test; results["y_pred"] = pred; results["probs_0"] = probs[:,0]; results["probs_1"] = probs[:,1]
    results.to_csv("classification_results/boosting_tree_results.csv")
    scores = score(results)
    return results, scores

def bagging(x_train, x_test, y_train, y_test, max_depth = 17, min_samples_leaf = 1, n_estimators = 15):
    base_model = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_leaf = min_samples_leaf)           #Generates a model
    model = ensemble.BaggingClassifier(base_estimator = base_model, n_estimators = n_estimators)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    probs = model.predict_proba(x_test)
    results = pd.DataFrame(columns = ["y_true","y_pred","probs_0", "probs_1"])
    results["y_true"] = y_test; results["y_pred"] = pred; results["probs_0"] = probs[:,0]; results["probs_1"] = probs[:,1]
    results.to_csv("classification_results/bagging_tree_results.csv")
    scores = score(results)
    return results, scores

def random_forest(x_train, x_test, y_train, y_test, n_estimators = 32, max_features = 15):
    model = ensemble.RandomForestClassifier(n_estimators = n_estimators, max_features = max_features)           #Generates a model
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    probs = model.predict_proba(x_test)
    results = pd.DataFrame(columns = ["y_true","y_pred","probs_0", "probs_1"])
    results["y_true"] = y_test; results["y_pred"] = pred; results["probs_0"] = probs[:,0]; results["probs_1"] = probs[:,1]
    results.to_csv("classification_results/random_forest_results.csv")
    scores = score(results)
    return results, scores

def gradient_boosting(x_train, x_test, y_train, y_test, n_estimators = 49):
    model = ensemble.GradientBoostingClassifier(n_estimators = 49)           #Generates a model
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    probs = model.predict_proba(x_test)
    results = pd.DataFrame(columns = ["y_true","y_pred","probs_0", "probs_1"])
    results["y_true"] = y_test; results["y_pred"] = pred; results["probs_0"] = probs[:,0]; results["probs_1"] = probs[:,1]
    results.to_csv("classification_results/gradient_boosting_tree_results.csv")
    scores = score(results)
    return results, scores

def k_neighbours(x_train, x_test, y_train, y_test, n_neighbors = 3, weights = "distance"):
    model = neighbors.KNeighborsClassifier(n_neighbors = n_neighbors, weights = weights)           #Generates a model
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    probs = model.predict_proba(x_test)
    results = pd.DataFrame(columns = ["y_true","y_pred","probs_0", "probs_1"])
    results["y_true"] = y_test; results["y_pred"] = pred; results["probs_0"] = probs[:,0]; results["probs_1"] = probs[:,1]
    results.to_csv("classification_results/k_neighbours_results.csv")
    scores = score(results)
    return results, scores

def svm_sgd(x_train, x_test, y_train, y_test):
    model = make_pipeline(StandardScaler(), SGDClassifier(random_state=26))
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    results = pd.DataFrame(columns = ["y_true","y_pred"])
    results["y_true"] = y_test; results["y_pred"] = pred
    results.to_csv("classification_results/svm_sgd_results.csv")
    scores = score(results)
    return results, scores

def svm_lin(x_train, x_test, y_train, y_test):
    model = make_pipeline(StandardScaler(), LinearSVC(random_state=26))
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    results = pd.DataFrame(columns = ["y_true","y_pred"])
    results["y_true"] = y_test; results["y_pred"] = pred
    results.to_csv("classification_results/svm_linear_results.csv")
    scores = score(results)
    return results, scores

def score(results):
    A = metrics.accuracy_score(results["y_true"], results["y_pred"])
    P = metrics.precision_score(results["y_true"], results["y_pred"])
    R = metrics.recall_score(results["y_true"], results["y_pred"])
    F1 = metrics.f1_score(results["y_true"], results["y_pred"])
    MCC = metrics.matthews_corrcoef(results["y_true"], results["y_pred"])
    if "probs_1" in results.columns:
        AUC = metrics.roc_auc_score(results["y_true"], results["probs_1"])
        scores = [A, P, R, F1, AUC, MCC]
    else:
        scores = [A, P, R, F1, np.nan, MCC]
    return scores

