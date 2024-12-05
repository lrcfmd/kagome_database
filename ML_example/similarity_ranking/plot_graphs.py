import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import factorial, ceil

graphs = ["hist", "single_MFD", "mean_MFD"]                                                                             ### Enter any/all of "hist", "single_MFD" or "mean_MFD" 
nvecs = "6"                                                                                                             ### to plot these graphs. For the single/mean MFD, nvecs 
                                                                                                                        ### must also be entered to identify the relevant files.
### PLOT HISTOGRAMS SHOWING DISTRIBUTION OF RECONSTRUCTION ERRORS ###

if "hist" in graphs:
    fig1, axes1 = plt.subplots(nrows=2, ncols=4, sharey = True)
    fig2, axes2 = plt.subplots(nrows=2, ncols=4, sharey = True)
    axes1 = axes1.flatten(); fig1.delaxes(axes1[3]); axes1 = np.delete(axes1, 3)
    axes2 = axes2.flatten(); fig2.delaxes(axes2[3]); axes2 = np.delete(axes2, 3)
    letters = ["a", "b", "c", "d", "e", "f", "g"]
    
    for k in range(2,9):
        xind = k-2
        GT = np.array(pd.read_csv(f"DATA/ground_truth_ranking_{k}_features.csv")["RE"])
        Q = np.array(pd.read_csv(f"DATA/query_ranking_{k}_features.csv")["RE"])
        GT_mean = np.mean(GT); Q_mean = np.mean(Q)
        GT_std = np.std(GT); Q_std = np.std(Q)
        GT_n, GT_bins, GT_patches = axes1[xind].hist(GT, 20, density=True)
        GT_y = ((1 / (np.sqrt(2 * np.pi) * GT_std)) * np.exp(-0.5 * (1 / GT_std * (GT_bins - GT_mean))**2))
        axes1[xind].plot(GT_bins, GT_y, "--")
        axes1[xind].set_title(f"{letters[xind]})",fontsize=15)
        axes1[xind].set(xlim = (0, 20))
        axes1[xind].tick_params(axis="x",labelsize=15)
        axes1[xind].tick_params(axis="y",labelsize=15)
        Q_n, Q_bins, Q_patches = axes2[xind].hist(Q, 20, density=True)
        Q_y = ((1 / (np.sqrt(2 * np.pi) * Q_std)) * np.exp(-0.5 * (1 / Q_std * (Q_bins - Q_mean))**2))
        axes2[xind].plot(Q_bins, Q_y, "--")
        axes2[xind].set_title(f"{letters[xind]})",fontsize=15)
        axes2[xind].set(xlim = (0, 20))
        axes2[xind].tick_params(axis="x",labelsize=15)
        axes2[xind].tick_params(axis="y",labelsize=15)
        fig1.text(0.01, 0.5, "Frequency", va = "center", rotation = "vertical", fontsize = 20)
        fig1.text(0.5, 0.01, "Reconstruction Error", ha = "center", fontsize = 20)
        fig1.text(0.5, 0.95, "Ground Truth", ha = "center", fontsize = 20)
        fig2.text(0.01, 0.5, "Frequency", va = "center", rotation = "vertical", fontsize = 20)
        fig2.text(0.5, 0.01, "Reconstruction Error", ha = "center", fontsize = 20)
        fig2.text(0.5, 0.95, "Query", ha = "center", fontsize = 20)
        
        fig1.tight_layout(); fig2.tight_layout()
    
    
    fig3, axes3 = plt.subplots(nrows=1,ncols=2, sharey=True)
    GT = np.array(pd.read_csv(f"DATA/ground_truth_ranking_MP_features.csv")["RE"])
    Q = np.array(pd.read_csv(f"DATA/query_ranking_MP_features.csv")["RE"])
    GT_mean = np.mean(GT); Q_mean = np.mean(Q)
    GT_std = np.std(GT); Q_std = np.std(Q)
    GT_n, GT_bins, GT_patches = axes3[0].hist(GT, 20, density=True)
    GT_y = ((1 / (np.sqrt(2 * np.pi) * GT_std)) * np.exp(-0.5 * (1 / GT_std * (GT_bins - GT_mean))**2))
    axes3[0].plot(GT_bins, GT_y, "--")
    axes3[0].set_title(f"Ground Truth",fontsize=15)
    axes3[0].set(xlim = (0, 20))
    axes3[0].tick_params(axis="x",labelsize=15)
    axes3[0].tick_params(axis="y",labelsize=15)
    
    Q_n, Q_bins, Q_patches = axes3[1].hist(Q, 20, density=True)
    Q_y = ((1 / (np.sqrt(2 * np.pi) * Q_std)) * np.exp(-0.5 * (1 / Q_std * (Q_bins - Q_mean))**2))
    axes3[1].plot(Q_bins, Q_y, "--")
    axes3[1].set_title(f"Query",fontsize=15)
    axes3[1].set(xlim = (0, 20))
    axes3[1].tick_params(axis="x",labelsize=15)
    axes3[1].tick_params(axis="y",labelsize=15)
    
    fig3.text(0.01, 0.5, "Frequency", va = "center", rotation = "vertical", fontsize = 20)
    fig3.text(0.5, 0.01, "Reconstruction Error", ha = "center", fontsize = 20)
    fig3.text(0.5, 0.95, "Magpie Vectors Without Compression", ha = "center", fontsize = 20)
    fig3.tight_layout()
    plt.show()

### PLOT A SINGLE MFD PLOT ###

def calculate(GT_RE, Q_RE):
    iter_num = np.linspace(0.0,20,num=101)                                                                     
    GT_fraction = []; Q_fraction = []
    lenGT = len(GT_RE); lenQ = len(Q_RE)
    single_eval = []

    for i in iter_num:
            threshold = i
            GT_count_inlier = np.sum(GT_RE[:] <= threshold)                                                    
            Q_count_inlier = np.sum(Q_RE[:] <= threshold)                                                      
            GT_percent = GT_count_inlier/lenGT                                                                 
            Q_percent = Q_count_inlier/lenQ
            single = GT_percent - Q_percent                                                                    
            GT_fraction.append(GT_percent)             
            Q_fraction.append(Q_percent)
            single_eval.append(single)

    return GT_fraction, Q_fraction, single_eval, iter_num

def plot(ax, k, GT_fraction, Q_fraction, single_eval, iter_num):
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
    ax.set_xlim(0, 10)
    ax.set_title(f"{k}", fontsize = 20)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)

def mean_per_PF(k,size,ranking):                                                                                
    if ranking=="GT":                                                                                           
        data = pd.read_csv(f"DATA/ground_truth_ranking_{k}_features.csv")
        data = data["RE"]
    elif ranking=="Q":
        data = pd.read_csv(f"DATA/query_ranking_{k}_features.csv")
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

if "single_MFD" in graphs:
    plt.figure(dpi=200)
    fig, axes = plt.subplots(figsize = (20,10))
    GT =  pd.read_csv(f"DATA/ground_truth_ranking_{nvecs}_features.csv")
    Q = pd.read_csv(f"DATA/query_ranking_{nvecs}_features.csv")
    GT_RE = GT["RE"]
    Q_RE = Q["RE"]
    ax = axes
    fontsize = 20
    title = f"Magpie vectors compressed to {nvecs} dimensions"
    GT_fraction, Q_fraction, single_eval, iter_num = calculate(GT_RE, Q_RE)
    MFD = max(single_eval);MFD_ind = single_eval.index(MFD); MFD_threshold = iter_num[MFD_ind]              
    plot(ax, title, GT_fraction, Q_fraction, single_eval, iter_num)
    fig.text(0.04, 0.5, "Proportion Below Threshold", va = "center", rotation = "vertical", fontsize = 20)
    fig.text(0.5, 0.04, "Reconstruction Error Threshold", ha = "center", fontsize = 20)
    plt.subplots_adjust(wspace=0.5, hspace = 0.3)
    plt.show()

if "mean_MFD" in graphs:
    plt.figure(dpi=200)
    fig, axes = plt.subplots(figsize = (20,10))
    GT_RE,GTuni = mean_per_PF(nvecs,4,"GT")
    Q_RE,Quni = mean_per_PF(nvecs,4,"Q")
    ax = axes
    fontsize = 20
    title = f"Mean reconstruction errors of Magpie vectors compressed to {nvecs} dimensions"
    GT_fraction, Q_fraction, single_eval, iter_num = calculate(GT_RE, Q_RE)
    MFD = max(single_eval);MFD_ind = single_eval.index(MFD); MFD_threshold = iter_num[MFD_ind]
    plot(ax, title, GT_fraction, Q_fraction, single_eval, iter_num)
    fig.text(0.04, 0.5, "Proportion Below Threshold", va = "center", rotation = "vertical", fontsize = 20)
    fig.text(0.5, 0.04, "Reconstruction Error Threshold", ha = "center", fontsize = 20)
    plt.subplots_adjust(wspace=0.5, hspace = 0.3)
    plt.show()

