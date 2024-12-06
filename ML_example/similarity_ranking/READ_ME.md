## rank.py

   - Size of phase field can be changed in check_rnk but must be set to match the phase fields being studied as in binary_classifier.py
   - Query dataset is built of every unique combination of elements found in the ground truth dataset, that dont also constitute a phase field in the ground truth dataset
   - In **main()** ranking function, to use pre-built vectors **use_vecs=True**
   - If **use_vecs=False**, the dimensionality for the elemental features to be compressed to can be specified in **main()**
   - Whether to perform a similarity ranking on elemental features without compression can be specified with **run_MP=True/False**
   - Produces rankings for each of set of vectors for the ground truth and query datasets

## get_candidates.py

   - Calculate the MFD between each ground truth-query ranking pair
   - The pair with the largest MFD is then used to assign query phase fields to the positive class
   - This outputs chemically_feasible_candidates.csv

## plot_graphs.py

   - Plots graphs that aren't required in the main code for analysis
   - Graphs variable can be set to include **"hist"**, **"single_MFD"** and **"mean_MFD"**
   - If running single/mean MFD, the nvecs variable must also be set to identify the relevant ranking pair
