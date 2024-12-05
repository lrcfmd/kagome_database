import pandas as pd

dataset = pd.read_csv("kagome_dataset.csv")                                                             ### Dataset introduced in the paper
dataset = dataset[["Formula", "Target"]]                                                                ### Only employing elemental features to classify phase fields

def make_fields(x):
    x = "".join(s for s in x if s not in "(.)" and not s.isdigit())                                     ### Remove brackets and stoichiometries, leaving only elements
    elements = str()
    for i,s in enumerate(x):                                                                            ### Convert into a phase field format, here "A B C D"
        if s.isupper() and i != 0:                                                                      ### Here, each composition is given a phase field, so duplicates present
            elements += " "+s                                       
        else:
            elements += s
    string = [el for el in elements.split(" ")]
    sorted_str = sorted(string)                                                                         ### Order elements alphabetically so that duplicates can be removed 
    PF = str()                                                                                          ### easily
    for el in sorted_str:
        PF += el + " "
    PF = PF.strip(" ")
    return PF

def get_targets(x):                                                                                     ### Convert "kagome" and "non-kagome" labels into ML interpretable values
    if x == "kagome":                                                                                   ### eg. 1s and 0s
        t = 1
    else:
        t = 0
    return t

all_PFs = pd.DataFrame(columns=["Phase Field", "Target"])
all_PFs["Phase Field"] = dataset["Formula"].apply(make_fields)
all_PFs["Target"] = dataset["Target"].apply(get_targets)

def rm_duplicate_elements(x):                                                                           ### Remove duplicate elements where present in phase fields
    x = sorted(set(x.split(" ")))
    PF = str()
    for el in x:
        PF += el + " "
    return PF.strip(" ")

all_PFs["Phase Field"] = all_PFs["Phase Field"].apply(rm_duplicate_elements)

uni_PFs = []; uni_Ts = []
for i, PF in enumerate(all_PFs["Phase Field"]):
    if PF not in uni_PFs:                                                                               ### Aggregate unique phase fields and their associated target, 
        uni_PFs.append(PF)                                                                              ### automatically removing duplicates
        uni_Ts.append(all_PFs.at[i, "Target"])
    elif PF in uni_PFs and uni_Ts[uni_PFs.index(PF)] == 0 and all_PFs.at[i, "Target"] == 1:             ### If the phase field is already present in the list of unique phase 
        uni_Ts[uni_PFs.index(PF)] = 1                                                                   ### fields, but has been assigned the positive kagome-containing class,
    else:                                                                                               ### then update the target value.
        continue

uni_PFs = pd.DataFrame({"Phase Field": uni_PFs, "Target": uni_Ts})
uni_PFs.to_csv("phase_field_dataset.csv",index=False)
