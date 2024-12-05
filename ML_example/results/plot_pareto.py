import pandas as pd
import numpy as np
import plotly.express as exp
import matplotlib.pyplot as plt
import os

def check_prt():                                                                                ### Checks whether fronts have previously been calculated 
    if "candidates_with_fronts.csv" in os.listdir():
        data = pd.read_csv("candidates_with_fronts.csv")
    else:
        data = pd.read_csv("final_candidates.csv")                                          ### Use final candidates, with both probability and chemical feasibility calculated
        X = data["Kagome Probability"].apply(lambda x: x*-1)                                ### Negative probability for minimisation of both metrics
        Y = data["RE"]
        pareto_fronts = calculate_pareto_fronts(X,Y)                                        
        data = order(data,pareto_fronts)
        data["Negative Kagome Probability"] = X
        data.to_csv("candidates_with_fronts.csv")
    return data

def calculate_pareto_fronts(x, y):
    combined_data = np.column_stack((x, y))
    num_points = len(x)
    pareto_fronts = []

    remaining_points = list(range(num_points))

    while remaining_points:                                                                 ### Loop through points in the dataframe
        current_front = []
        for i in remaining_points:
            is_pareto = True
            x1, y1 = combined_data[i]

            for j in remaining_points:
                if i == j:
                    continue

                x2, y2 = combined_data[j]

                if x2 <= x1 and y2 <= y1 and (x2 < x1 or y2 < y1):                          ### For a given point, if any other is smaller with respect to both metrics then the 
                    is_pareto = False                                                       ### the first point is not on the current front
                    break

            if is_pareto:
                current_front.append(i)

        pareto_fronts.append(current_front)
        print('Pareto front:' , len(pareto_fronts))
        remaining_points = [p for p in remaining_points if p not in current_front]          ### Remove points remaining if they are on current front

    return pareto_fronts

def order(data,pareto_fronts):                                                              ### Output of "calculate pareto fronts" is an array of arrays where each array is a 
    N = len(data)                                                                           ### a front containing the indices of points on that front. 
    fronts = np.zeros(N)
    for i, f in enumerate(pareto_fronts):                                                   ### Go through the fronts and identify the indexes of each point on that front
        for ind in f:
            fronts[ind] = i+1                                                               ### Assign the front number to the correct indexed position in a list
    data["Front"] = fronts                                                                  ### Add this column into the dataframe
    mx = int(data["Front"].max())
    dfs = []
    for j in range(1,mx+1):                                                                 ### Reorganise the df so that it is in order of increasing front number
        df = data[data["Front"]==j]
        dfs.append(df)
    data = pd.concat(dfs)
    return data

def plot_html(data):
    fig = exp.scatter(data, x = "Negative Kagome Probability", y = "RE", 
            color = "Front", hover_name = "Phase Field")
    fig.update_traces(marker_size=12)
    fig.update_layout(coloraxis_colorbar_x=-0.15)
    fig.update_layout(showlegend=True)
    fig.update_layout(font=dict(size=30))
    fig.show()
    fig.write_html(f'pareto_front_plotly.html')
    return

def plot_basic(data):
    first_front = data[data["Front"] == 1]
    second_tenth = data[data["Front"].isin(range(2,11))]
    rest = data[~data["Front"].isin(range(1,11))]
    x = first_front["Negative Kagome Probability"]; y = first_front["RE"]
    x2 = second_tenth["Negative Kagome Probability"]; y2 = second_tenth["RE"]
    x3 = rest["Negative Kagome Probability"]; y3 = rest["RE"]
    fig, ax = plt.subplots()
    ax.scatter(x,y,color="r",edgecolor="none",s=100,zorder=3)
    ax.scatter(x2,y2,color="k",edgecolor="none",s=50,zorder=2)
    ax.scatter(x3,y3,color="slategrey",edgecolor="none",s=20,zorder=1,alpha=0.7)
    ax.set_xlabel("Negative kagome probability", fontsize = 25)
    ax.set_ylabel("Reconstruction error", fontsize = 25)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.show()

if __name__ == "__main__":
    data = check_prt()
    plot_html(data)
    plot_basic(data)
