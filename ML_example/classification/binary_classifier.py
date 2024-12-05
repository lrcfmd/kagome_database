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

def check_cls(features = features, random_seed=None, test_max=None, fname=None, size=None):
    fs = ["x_train.csv", "x_test.csv", "y_train.csv", "y_test.csv"]                         ### Check whether the dataset has been formatted already and read in if so
    if all(f in os.listdir("DATA") for f in fs):
        print("Found existing training and testing data...")
        x_train = pd.read_csv("DATA/x_train.csv"); x_test = pd.read_csv("DATA/x_test.csv")
        #x_train["Features"] = x_train["Features"].apply(literal_eval); x_test["Features"] = x_test["Features"].apply(literal_eval)
        y_train = pd.read_csv("DATA/y_train.csv"); y_test = pd.read_csv("DATA/y_test.csv")
    else:
        data = pd.read_csv(f"../datasets/{fname}")                                          ### If not, limit the phase field size, featurise and split the data 
        if test_max:
            data = data.iloc[range(test_max)]
        print(f"Collecting phase fields of size {size}")
        data = limit_size(data, size)
        print("Splitting data")
        y_train, y_test = split(data, random_seed)
        print("Permuting data")
        y_train = permute(y_train); y_test = permute(y_test)
        x_train = featurise(y_train, features); x_test = featurise(y_test, features)
        y_train.to_csv("DATA/y_train.csv",index=False); y_test.to_csv("DATA/y_test.csv",index=False)
        x_train.to_csv("DATA/x_train.csv",index=False); x_test.to_csv("DATA/x_test.csv",index=False)
        print("Data saved to x_train, x_test, y_train, y_test in DATA directory")
    return x_train, x_test, y_train, y_test
    
def limit_size(data,size):                                                                  ### Remove any phase fields that aren't quaternary
    inds = [i for i, PF in enumerate(data["Phase Field"]) if len(PF.split(" ")) == size]
    data = data.iloc[inds]
    data = data.reset_index(drop=True)
    return data

def permute(data):                                                                          ### Permute the ordering of elements in each phase field
    dfs = [data]
    if "Target" in list(data.columns.values):
        for PF,target in zip(data["Phase Field"],data["Target"]):
            perms = list(permutations(PF.split(" ")))
            perms = perms[1:]
            new_PFs = list()
            for perm in perms:
                new_PF = str()
                for el in perm:
                    new_PF = new_PF + el + " "
                new_PFs.append(new_PF.strip(" "))
            df = pd.DataFrame(); df["Phase Field"] = new_PFs; df["Target"] = [target]*len(new_PFs)
            dfs.append(df)
        data = pd.concat(dfs,ignore_index=True)
    else:
        for PF in data["Phase Field"]:
            perms = list(permutations(PF.split(" ")))
            perms = perms[1:]
            new_PFs = list()
            for perm in perms:
                new_PF = str()
                for el in perm:
                    new_PF = new_PF + el + " "
                new_PFs.append(new_PF.strip(" "))
            df = pd.DataFrame(); df["Phase Field"] = new_PFs
            dfs.append(df)
        data = pd.concat(dfs,ignore_index=True)
    return data

def featurise(data, features):                                                              ### Featurise the data using MagPie libraries
    features = [f.strip() for f in features]
    dics = make_dics(features)
    feats = make_vecs(data, dics)
    df = pd.DataFrame(feats, columns = [f"feature_{i}" for i in range(len(feats[0]))])
    return df

def read_features(f):
    lines = open(f,'r').readlines()
    return [float(l.strip()) if l.strip().isdigit else 0 for l in lines] 

def make_dics(features):                                                                    ### Make dictionaries, with each dictionary containing the features of each element
    dics = [ {} for f in features]
    symbols = [s.strip() for s in open('DATA/magpie/magpie_tables/Abbreviation.table', 'r').readlines()]
    for i,f in enumerate(features):
        try:
            table = read_features(f'DATA/magpie/magpie_tables/{f}.table')
            dics[i]  = {sym: float(num) for sym, num in zip(symbols, table)} 
        except:
            pass
    return dics

def make_vecs(data, dics):                                                                  ### Build phase vectors
    feats = []
    for PF in data["Phase Field"]:
        PF = PF.split()
        PV = []
        for el in PF:
            for dic in dics:
                PV.append(float(dic[el]))
        feats.append(PV)
    return feats

def format_values(data, feats):                                                                     
    data["Target"] = data["Target"].apply(lambda x: int(x))
    feats = feats.apply(make_float)
    return data, feats

def make_float(array):
    return [float(a) for a in array]

def split(data, random_seed):
    y = data
    y_train, y_test = train_test_split(y, train_size = 0.9, random_state = random_seed) 
    return y_train, y_test

def run_classifier(x_train, x_test, y_train, y_test, how):                                  ### Run all or a selected classifier
    if how == "simple":
        print("Running simple decision tree...")
        results, scores = simple(x_train, x_test, y_train, y_test)
    elif how == "boosting":
        print("Running boosting decision tree...")
        results, scores = boosting(x_train, x_test, y_train, y_test)
    elif how == "bagging":
        print("Running bagging decision tree...")
        results, scores = bagging(x_train, x_test, y_train, y_test)
    elif how == "random_forest":
        print("Running random forest...")
        results, scores = random_forest(x_train, x_test, y_train, y_test)
    elif how == "gradient_boosting":
        print("Running gradient boosting decision tree...")
        results, scores = gradient_boosting(x_train, x_test, y_train, y_test)
    elif how == "k_neighbours":
        print("Running k nearest neighbours...")
        results, scores = k_neighbours(x_train, x_test, y_train, y_test)
    elif how == "svm_sgd":
        print("Running support vector machine with stochastic gradient descent...")
        results, scores = svm_sgd(x_train, x_test, y_train, y_test)
    elif how == "svm_lin":
        print("Running basic support vector machine...")
        results, scores = svm_lin(x_train, x_test, y_train, y_test)
    elif how == "all":
        scores = []
        r, s = simple(x_train, x_test, y_train, y_test); scores.append(s)
        r, s = boosting(x_train, x_test, y_train, y_test); scores.append(s)
        r, s = bagging(x_train, x_test, y_train, y_test); scores.append(s)
        r, s = random_forest(x_train, x_test, y_train, y_test); scores.append(s)
        r, s = gradient_boosting(x_train, x_test, y_train, y_test); scores.append(s)
        r, s = k_neighbours(x_train, x_test, y_train, y_test); scores.append(s)
        r, s = svm_sgd(x_train, x_test, y_train, y_test); scores.append(s)
        r, s = svm_lin(x_train, x_test, y_train, y_test); scores.append(s)
        results = "See DATA for ALL saved results"
    return results, scores

def train_BST(train_x, train_y):
    enc = LabelEncoder()
    train_y = enc.fit_transform(train_y)

    base_model = tree.DecisionTreeClassifier(max_depth = 17, min_samples_leaf = 1)           #Generates a model
    model = ensemble.AdaBoostClassifier(base_estimator=base_model, n_estimators=17, random_state = 240624)
    model.fit(train_x, train_y)

    return model

def predict_BST(model, test_x):
    probs = model.predict_proba(test_x)
    kag_probs = probs[:,1]
    return kag_probs

if __name__ == "__main__":
    #####Initial testing of binary classifiers#####

    fname = "phase_field_dataset.csv"
    x_train, x_test, y_train, y_test = check_cls(features = features, random_seed = 26, fname = fname, size = 4)
    print(y_train,x_train)
    y_train, x_train = format_values(y_train, x_train); y_test, x_test = format_values(y_test, x_test)
    print(y_train, x_train)
    print("Length is...", (len(x_train)+len(x_test)))
    train_PFs = y_train["Phase Field"]; y_train = y_train.drop(labels="Phase Field",axis=1); train_PFs.to_csv("DATA/train_PFs.csv",index=False)
    test_PFs = y_test["Phase Field"]; y_test = y_test.drop(labels="Phase Field",axis=1); test_PFs.to_csv("DATA/test_PFs.csv",index=False)
    how = "all"
    results, scores = run_classifier(x_train, x_test, y_train, y_test, how)
    ### Uncomment if all ###
    models = ["Simple", "Boosting", "Bagging", "Random Forest", "Gradient Boosting", "K Neighbours", "SVM with SGD", "SVM linear"]
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC", "MCC"]
    scores_df = pd.DataFrame(scores,index=models,columns=metrics)
    if how == "all":
        best = scores_df["MCC"].idxmax()
        print(f"The best model was {best}")
    scores_df.to_csv("classification_results/scores.csv")
    ### Uncomment if using just one model ###
    #print(scores)
    
    #####Classification of feasible phase fields, run after ranking#####
    
    #features = features                                                                                                
    #train = pd.read_csv("../datasets/phase_field_dataset.csv")
    #train = limit_size(train,4)
    #train = permute(train)
    #x_train = featurise(train, features)
    #y_train = train["Target"]

    #candidates = pd.read_csv("../similarity_ranking/ranking_results/chemically_feasible_candidates.csv")                                            
    #x_test = featurise(candidates, features)                                                                   ### Make test data using chemically feasible candidates 
    #print(len(x_test)); print(len(x_train))
    #model = train_BST(x_train, y_train)
    #probs = predict_BST(model, x_test)

    #candidates["Kagome Probability"] = probs                                                      ### Only probability is significant here
    #candidates = candidates.loc[candidates["Kagome Probability"]>0.5]
    #candidates.to_csv("../results/final_candidates.csv",index=False)
