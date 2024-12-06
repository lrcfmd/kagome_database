import numpy as np
import pandas as pd
import os
import sys
rnk_dir = os.path.dirname("os.path.abspath(__file__))")
home_dir = os.path.abspath(os.path.join(rnk_dir,".."))
cls_dir = os.path.abspath(os.path.join(home_dir,"classification"))
if cls_dir not in sys.path:
    sys.path.append(cls_dir)
from binary_classifier import limit_size, permute, read_features, make_dics
from DATA.ELEMENTS import ELEMENTS
from DATA.feature_labels import features
from itertools import combinations
from autoencoders import compress, run_AE

def check_rnk(atoms,size=None,fname=None):                                                                      ### Checks whether ground truth and query datasets are already
    ground_truth = check_for_GT(atoms,size,fname)
    query = check_for_Q(atoms,ground_truth,size,fname)
    return ground_truth, query 

def check_for_GT(atoms=None,size=None,fname=None):
    if "ground_truth.csv" in os.listdir("DATA"):
        data = pd.read_csv(f"DATA/ground_truth.csv")
        print("Found existing ground truth dataset...")
    else:
        print("Building a new ground truth dataset...")                                                         ### Constructs the ground truth from scratch using the csv of 
        data = pd.read_csv(f"../datasets/{fname}")                                                                     ### phase fields if they hadn't been formatted prior.
        print(f"Limiting phase fields to size {size}")
        data = limit_size(data, size)
        print("Permuting the phase fields")
        data = permute(data)
        print("Removing phase fields containing atoms with atomic number > 87")
        data = rm86(data,atoms)
        data.to_csv("DATA/ground_truth.csv",index=False)
    return data

def rm86(data,atoms):                                                                                           ### Remove elements with an atomic number greater than 87
    rminds = []
    for i,PF in enumerate(data["Phase Field"]):
        if any(atoms.index(el)>86 for el in PF.split(" ")):
            rminds.append(i)
    data.drop(rminds,inplace=True)
    data.reset_index(drop=True,inplace=True)
    return data

def check_for_Q(atoms,GT,size,fname):
    if "query.csv" in os.listdir("DATA"):                                                                             ### Loads existing query set
        print("Found existing query dataset...")
        query_df = pd.read_csv(f"DATA/query.csv")
    else:
        print("Building new query dataset...")                                                                  ### Builds new query set using all the elements present in the 
        print("Getting unique elements in the ground truth dataset")                                            ### ground truth dataset. Every unique quaternary combination 
        uni_els = unique_elements(pd.read_csv(f"DATA/{fname}"))                                                 ### of elements found in the ground truth are in the query. 
        print("Constructing query phase fields")
        query = unique_combinations(uni_els, GT, size)
        query_df = pd.DataFrame(columns = ["Phase Field"]); query_df["Phase Field"] = query
        query_df = rm86(query_df, atoms)
        query_df = rm_not_present(query_df, GT)
        query_df.to_csv("temp_query.csv",index=False)
        print("Permuting query phase fields")
        query_df = query_df[~query_df["Phase Field"].isin(GT["Phase Field"])]
        query_df = permute(query_df)
        query_df.to_csv("DATA/query.csv",index=False)
    return query_df

def unique_elements(data):                                                                                        ### Unique set of elements present in the ground truth
    PFs = data["Phase Field"]
    el_list = list()
    for PF in PFs:
        PF = PF.split(" ")
        for el in PF:
            el_list.append(el)
    uni_els = []
    for el in el_list:
        if el not in uni_els:
            uni_els.append(el)
    return uni_els

def unique_combinations(uni_els, GT, size):                                                                     ### First creates unique quaternary combinations of indices
    print(uni_els)
    inds = range(len(uni_els)-1)
    combs = combinations(inds, size)
    query = make_fields(uni_els,combs)
    query = [PF for PF in query if PF not in GT["Phase Field"]]
    return query

def make_fields(uni_els,combs):                                                                                 ### Then makes the query phase fields using the indices
    query = []
    for c in combs:
        QPF = str()
        for i in c:
            QPF = QPF + uni_els[i] + " "
        query.append(QPF.strip(" "))
    return query

def rm_not_present(query, GT):
    PFs = GT["Phase Field"]
    el_list = list()
    print(len(query))
    for PF in PFs:
        PF = PF.split(" ")
        for el in PF:
            el_list.append(el)
    el_list = set(el_list)
    print(el_list)
    inds= []
    for i,PF in enumerate(query["Phase Field"]):
            if not all(el in el_list for el in PF.split(" ")):
                inds.append(i)
    query.drop(inds,inplace=True)
    query.reset_index(drop=True,inplace=True)
    return query

def P2I(atoms, PFs):                                                                                            ### Converts phase fields to elemental indexes as the compressed 
    all_inds = []                                                                                               ### features are element-wise
    for PF in PFs["Phase Field"]:
        inds = [atoms.index(el) for el in PF.split(" ")]
        all_inds.append(inds)
    return all_inds

def get_elemental_features(atoms, features):                                                                    ### Builds elemental feature vectors
    features = [f.strip() for f in features]
    feats = []
    dics = make_dics(features)
    for el in atoms:
        el_feats = []
        for dic in dics:
            el_feats.append(float(dic[el]))
        feats.append(el_feats)
    return feats

def main(kmin, kmax, el_feats, ground_truth, query, GT_atom_inds=None, Q_atom_inds=None, use_vecs=True, run_MP=True, plot_his=None):
    if run_MP:
        GT_vecs = P2V(el_feats, GT_atom_inds); Q_vecs = P2V(el_feats, Q_atom_inds)
        print("Running chemical similarity ranking")
        GT_results, Q_results, history = run_AE(GT_vecs, Q_vecs)                                                ### Ranks chemical similarity
        GT = pd.DataFrame({"Phase Field": ground_truth["Phase Field"],"RE": GT_results["RE"]})
        Q = pd.DataFrame({"Phase Field": query["Phase Field"], "RE":  Q_results["RE"]})
        print("Done make df")
        GT.to_csv(f"ranking_results/ground_truth_ranking_MP_features.csv",index=False); Q.to_csv(f"ranking_results/query_ranking_MP_features.csv",index=False)
        print("Saved")
        if plot_his:
            plot_history(history,"MP")
    for k in range(kmin,kmax):                                                                                  ### Loops through the given size range of compressed feature 
        if use_vecs:                                                                                            ### vectors
            print(f"Reading in {k}-dimensional vectors...")                                                     ### Read in pre-built vectors    
            GT_vecs = pd.read_csv(f"DATA/vecs/GT_{k}.csv")
            Q_vecs = pd.read_csv(f"DATA/vecs/Q_{k}.csv")
        else:
            print(f"Compressing features to {k} dimensions...")                                                
            el_vecs, RE, encoder, autoencoder = compress(el_feats,k)                                            ### Compresses features
            print("Constructing phase vectors")
            GT_vecs = P2V(el_vecs, GT_atom_inds); Q_vecs = P2V(el_vecs, Q_atom_inds)
        print("Running chemical similarity ranking")
        GT_results, Q_results, history = run_AE(GT_vecs, Q_vecs)                                                ### Ranks chemical similarity
        GT = pd.DataFrame({"Phase Field": ground_truth["Phase Field"],"RE": GT_results["RE"]})
        Q = pd.DataFrame({"Phase Field": query["Phase Field"], "RE":  Q_results["RE"]})
        print("Done make df")
        GT.to_csv(f"ranking_results/ground_truth_ranking_{k}_features.csv",index=False); Q.to_csv(f"ranking_results/query_ranking_{k}_features.csv",index=False)
        print("Saved")
        if plot_his:
            plot_history(history,k)

def P2V(el_vecs, atom_inds):
    num_rows = len(atom_inds)
    num_cols = sum(len(el_vecs[i]) for i in atom_inds[0])

    PVs = np.empty((num_rows, num_cols), dtype=np.float32)

    for row, inds in enumerate(atom_inds):
        PVs[row, :] = np.concatenate([el_vecs[i] for i in inds])

    return PVs

def plot_history(history,k):                                                                                    ### Plot losses over epochs
    loss = np.array(history.history["loss"])
    val_loss = np.array(history.history["val_loss"])
    epochs = [i for i in range(len(loss))]

    fig, ax = plt.subplots()
    line1, = ax.plot(epochs, loss)
    line2, = ax.plot(epochs, val_loss)
    ax.set_title(f"History of ranking using {k} features")
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    ax.legend([line1, line2],["Loss", "Val Loss"])
    ax.set(xlim=(0,N))
    plt.show()

if __name__ == "__main__":
    atoms = [s.strip() for s in open('../classification/DATA/Abbreviation.table', 'r').readlines()]
    features = features
    ground_truth, query = check_rnk(atoms, size = 4, fname = "phase_field_dataset.csv")
    GT_atom_inds = P2I(atoms, ground_truth)
    Q_atom_inds = P2I(atoms, query)
    el_feats = get_elemental_features(atoms, features)
    plot_his = ""#Setting to anything plots history
    main(2, 9, el_feats, ground_truth, query, GT_atom_inds=GT_atom_inds, Q_atom_inds=Q_atom_inds, plot_his=plot_his)
