import os
import sys
home = os.getcwd()
sys.path.append(os.path.join(home,"classification"))
sys.path.append(os.path.join(home,"similarity_ranking"))
sys.path.append(os.path.join(home,"results"))
from classification.binary_classifier import *
from classification.classifier_models import *
from similarity_ranking.rank import *
from similarity_ranking.autoencoders import *
from similarity_ranking.get_candidates import *
from results.plot_pareto import *

####Phase field dataset must be built first by running build_phasefields.py in the datasets folder####

####Initial testing of binary classifiers#####

os.chdir("classification")
if "classification_results" not in os.listdir():
    os.mkdir("classification_results")
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
## Uncomment if using just one model ###
print(scores)
os.chdir(home)

####Chemical similarity ranking####

os.chdir("similarity_ranking")
if "ranking_results" not in os.listdir():
    os.mkdir("ranking_results")
atoms = [s.strip() for s in open('../classification/DATA/magpie/magpie_tables/Abbreviation.table', 'r').readlines()]
features = features
ground_truth, query = check_rnk(atoms, size = 4, fname = "phase_field_dataset.csv")
GT_atom_inds = P2I(atoms, ground_truth)
Q_atom_inds = P2I(atoms, query)
el_feats = get_elemental_features(atoms, features)
plot_his = ""#Setting to anything plots history
main(2, 9, el_feats, ground_truth, query, GT_atom_inds=GT_atom_inds, Q_atom_inds=Q_atom_inds, plot_his=plot_his)

####Calculaton of MFD for each ranking####

os.chdir("similarity_ranking")
vals = plot_MFD(2, 9, 2, 4, del_ax=3)
k = vals[0]; best_nvecs = k
print("The number of latent features is", k)
GT_means, GT_uni = mean_per_PF(k,4,"GT");Q_means, Q_uni = mean_per_PF(k,4,"Q")
CFC = pd.DataFrame()
print(Q_uni)
query = pd.read_csv("DATA/query.csv")
CFC["Phase Field"] = query["Phase Field"].iloc[:Q_uni]; CFC["RE"] = Q_means
CFC = CFC.loc[CFC["RE"]<vals[2]]                                                                          
CFC.to_csv("ranking_results/chemically_feasible_candidates.csv", index=False)                             
os.chdir(home)

####Classification of query candidates####

os.chdir("classification")
features = features
train = pd.read_csv("../datasets/phase_field_dataset.csv")
train = limit_size(train,4)
train = permute(train)
x_train = featurise(train, features)
y_train = train["Target"]

candidates = pd.read_csv("../similarity_ranking/ranking_results/chemically_feasible_candidates.csv")
x_test = featurise(candidates, features)                                                                   
print(len(x_test)); print(len(x_train))
model = train_BST(x_train, y_train)
probs = predict_BST(model, x_test)
candidates["Kagome Probability"] = probs                                                      
candidates = candidates.loc[candidates["Kagome Probability"]>0.5]
candidates.to_csv("../results/final_candidates.csv",index=False)
os.chdir(home)

####Optimisation of both metrics with pareto fronts####

os.chdir("results")
data = check_prt()
plot_html(data)
plot_basic(data)
os.chdir(home)

