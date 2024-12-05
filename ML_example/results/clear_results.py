import os

for f in os.listdir():
    if f not in ["clear_results.py", "plot_pareto.py"]:
        os.remove(f)

os.chdir("../classification/classification_results/")
for f in os.listdir():
    os.remove(f)

os.chdir("../../similarity_ranking/ranking_results/")
for f in os.listdir():
    os.remove(f)

