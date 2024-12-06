import os

for f in os.listdir():
    if f not in ["clear_results.py", "plot_pareto.py"]:
        if os.path.isfile(f):
            os.remove(f)
        elif os.path.isdir(f):
            continue

os.chdir("../classification/classification_results/")
for f in os.listdir():
    if os.path.isfile(f):
        os.remove(f)
    elif os.path.isdir(f):
        continue

os.chdir("../../similarity_ranking/ranking_results/")
for f in os.listdir():
    if os.path.isfile(f):
        os.remove(f)
    elif os.path.isdir(f):
        continue
