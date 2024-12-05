import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import cm
import itertools
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn import datasets, metrics
from numpy import nan as NaN
from matplotlib.pyplot import figure
from math import ceil,factorial

def plot_MFD(kmin, kmax,L0,L1,del_ax=None):                                                                     ### Plots MFDs for all rankings over a range
    vals = None
    plt.figure(dpi=200)
    fig, axes = plt.subplots(L0,L1, figsize = (20,10), sharey=True)
    axes = axes.flatten()
    if del_ax:
        fig.delaxes(axes[del_ax]); axes = np.delete(axes, del_ax)
    for k in range(kmin, kmax):
        GT =  pd.read_csv(f"ranking_results/ground_truth_ranking_{k}_features.csv")
        Q = pd.read_csv(f"ranking_results/query_ranking_{k}_features.csv")
        #GT = pd.read_csv("GT_test.csv")
        #Q = pd.read_csv("Q_test.csv")
        GT_RE = GT["RE"]
        Q_RE = Q["RE"]
        ax = axes[k-kmin]
        fontsize = 20
        GT_fraction, Q_fraction, single_eval, iter_num = calculate(GT_RE, Q_RE)
        MFD = max(single_eval);MFD_ind = single_eval.index(MFD); MFD_threshold = iter_num[MFD_ind]              ### MFD is the maximum fractoinal difference, explained below
        vals = save_or_not(k, MFD, MFD_threshold, vals)
        plot(ax, k, GT_fraction, Q_fraction, single_eval, iter_num)
    fig.text(0.04, 0.5, "Proportion Below Threshold", va = "center", rotation = "vertical", fontsize = 20)
    fig.text(0.5, 0.04, "Reconstruction Error Threshold", ha = "center", fontsize = 20)
    plt.subplots_adjust(wspace=0.5, hspace = 0.3)
    plt.show()
    return vals

def calculate(GT_RE, Q_RE):
    iter_num = np.linspace(0.0,20,num=101)                                                                      ### Get num points over a range of RE, obtained from the rankings
    GT_fraction = []; Q_fraction = []
    lenGT = len(GT_RE); lenQ = len(Q_RE)
    single_eval = []
    
    for i in iter_num:
            threshold = i
            GT_count_inlier = np.sum(GT_RE[:] <= threshold)                                                     ### At each threhshold, count the number of ground truth and 
            Q_count_inlier = np.sum(Q_RE[:] <= threshold)                                                       ### query phase fields with an RE below.
            GT_percent = GT_count_inlier/lenGT                                                                  ### Find the proportion at each point for both sets.
            Q_percent = Q_count_inlier/lenQ
            single = GT_percent - Q_percent                                                                     ### Find the difference between the proportions of each set below
            GT_fraction.append(GT_percent)                                                                      ### a given error
            Q_fraction.append(Q_percent)
            single_eval.append(single)

    return GT_fraction, Q_fraction, single_eval, iter_num

def save_or_not(k,MFD,MFD_threshold,vals=None):                                                                 ### Most conservative ranking has 'generally' the largest MFD
    if vals==None:
        vals = list([k,MFD,MFD_threshold])
    elif vals[1]<MFD:                                                                                           ### If new MFD is larger, update the values
        vals = list([k,MFD,MFD_threshold])
    else:
        vals = vals
    return vals

def plot(ax, k, GT_fraction, Q_fraction, single_eval, iter_num):
    letters=["a","b","c","d","e","f","g"]
    ax.plot(iter_num,GT_fraction,color='blue',label='Labelled')
    ax.plot(iter_num,Q_fraction,color='orange',label='Unlabelled')
    ax.plot(iter_num,single_eval,color='gray',label='Fraction Difference')
    single_max = max(single_eval)
    temp = single_eval.index(single_max)
    threshold_max = iter_num[temp]
    ax.plot([0,threshold_max],[single_max,single_max], '--', color="gray")
    ax.plot([threshold_max,threshold_max],[0,single_max], '--', color='gray')
    single_max = round(single_max,2)
    threshold_max = round(threshold_max,3)
    cord =  '('+format(threshold_max, '.2f')+', '+format(single_max,'.2f')+')'
    ax.text(threshold_max, single_max, cord, fontsize=20, fontweight='semibold', c='gray')
    Q_max = Q_fraction[temp]
    Q_max = round(Q_max,3)
    ax.scatter(threshold_max, Q_max, c="orange")
    ax.text(threshold_max, Q_max-0.05, Q_max, fontsize=20,c='orange')
    GT_max = GT_fraction[temp]
    GT_max = round(GT_max,3)
    ax.scatter(threshold_max, GT_max, c="blue")
    ax.text(threshold_max+0.1, GT_max+0.0, GT_max, fontsize=20,c='blue')
    refer_min, refer_max = min(GT_max, Q_max), max(GT_max, Q_max)
    ax.plot([threshold_max,threshold_max],[refer_min,refer_max], '--', color='gray')
    ax.set_xlim(0, 20)
    ax.set_title(f"{letters[k-2]})", fontsize = 20)
    ax.tick_params(axis="y",labelsize=15)
    ax.tick_params(axis="x",labelsize=15)

def mean_per_PF(k,size,ranking):                                                                                ### Calculate the mean RE for each phase field by averaging the 
    if ranking=="GT":                                                                                           ### RE acheived by each permutation
        data = pd.read_csv(f"ranking_results/ground_truth_ranking_{k}_features.csv")
        data = data["RE"]
    elif ranking=="Q":
        data = pd.read_csv(f"ranking_results/query_ranking_{k}_features.csv")
        data = data["RE"]
    Nperms = factorial(size)
    Nuni = int(len(data)/Nperms)
    print(Nuni)
    means = []
    for i in range(Nuni):
        start = Nuni + i*(Nperms-1); fin = Nuni + (i+1)*(Nperms-1)
        inds = [i for i in range(start, fin)]; inds.append(i)
        df = data.iloc[inds]
        sm = df.sum()
        mean = sm/Nperms; means.append(mean)
    return means, Nuni

if __name__ == "__main__":
    vals = plot_MFD(2, 9, 2, 4, del_ax=3)
    k = vals[0]
    print("The number of latent features is", k)
    GT_means, GT_uni = mean_per_PF(k,4,"GT");Q_means, Q_uni = mean_per_PF(k,4,"Q")
    CFC = pd.DataFrame()
    print(Q_uni)
    query = pd.read_csv("DATA/query.csv")
    CFC["Phase Field"] = query["Phase Field"].iloc[:Q_uni]; CFC["RE"] = Q_means
    CFC = CFC.loc[CFC["RE"]<vals[2]]                                                                            ### Compares mean RE's to the threshold acheived with the largest
    CFC.to_csv("ranking_results/chemically_feasible_candidates.csv", index=False)                                          ### MFD.
