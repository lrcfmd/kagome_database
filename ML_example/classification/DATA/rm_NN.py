import os 
import shutil
import glob
from feature_labels import features

features = [f.strip() for f in features]
tables = [x.strip(".table") for x in os.listdir("magpie_tables")]
for f in features:
    if f not in tables:
        print(f)
