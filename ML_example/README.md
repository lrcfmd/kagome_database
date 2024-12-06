# ML Example Workflow #

## Summary ##

The dataset was used to train a binary classifier model to classify quaternary phase fields that contain at least one structure with kagome layers. A query dataset was generated from every unique quaternary combination of elements present in the training dataset. Autoencoders were used to compress the elemental features and rank the chemical feasibility of query phase fields, by proxy of chemical similarity to the phase fields reported in the ICSD. Feasible query phase fields were then inputted into the best performing binary classifier model for probability evaluation. Pareto fronts were plotted to identify phase fields with an optimal combination of both metrics.

## Downloading vectors ##

The large vector file required for this workflow can be found on the releases page:

[link to vectors](https://github.com/lrcfmd/kagome_database/releases/tag/V1.0)

Once downloaded, the file should be extracted in similarity_ranking.DATA

## Running workflow ##

1. results.build_phasefields.py
   - Must be run first
   - Extracts compositions and simple structure type (kagome/non-kagome) from the dataset
   - Aggregates compositions into phase fields, removing duplicates in the process
   - If any structure within a phase field has kagome layers, the phase field is assigned to the positive class
   - Outputs a condensed dataset of phase fields and binary targets indicating the presence of structures with kagome layers

2. main.py
   - Run to complete all parts of workflow: testing classifiers, similarity ranking, classification of query phase fields and calculation of pareto fronts
   - Vectors should be downloaded first from the link above [Downloading vectors(#Downloading-vectors)

3. similarity_ranking.plot_graphs.py
   - Creates graphs that weren't needed for analysis including histograms of reconstruction error distribution, and single/average MFD plots

## To run scripts independently ##

1. build_phasefields.py
   - As above

2. binary_classifier.py
   - Tests binary classification models 
   - Use the part indicated in the file to split and permute the data, then train and test models
   - Outputs scores acheived by each, or one, model. MCC was used to determine the best performing model 

3. rank.py
   - Vectors should be downloaded first from the link above [Downloading vectors](#Downloading-vectors)
   - Generates a query dataset
   - Runs a chemical similarity ranking between the reported "ground truth" and query phase fields as proxy for chemical feasibility of the query dataset
   - Use pre-calculated vectors to obtain same results. Vectors can also be compressed independently by setting "use_vecs" variable in "main" function to False.
   - Outputs rankings

4. get_candidates.py
   - Performs MFD calculations to determine the most conservative ranking of query phase fields
   - Uses that ranking to assign query phase fields to the feasible class
   - Outputs chemically feasible candidates
   - To obtain the MFD plot for a single/mean ranking pair eg. Magpie vectors without comrpession, see plot_graphs.py

5. binary_classifier.py
   - Use the second indicated part to perform binary classification on the query phase fields
   - Outputs candidates that are likely to contain a kagome structure and are chemically feasible. "Final_candidates" in results

6. plot_pareto.py
   - Calculates pareto fronts to optimise both metrics
   - Outputs interactive pareto plot, plot emphasising the first front and file containing each phase field and corresponding front

7. plot_graphs.py
   - Plots histograms of reconstruction error distribution as well as single/mean MFD plots. 
   - Desired plots can be inputted into the graphs variable. If plotting single or mean MFD, specific ranking required by nvecs, which can be the number of vecs or "MP" for pure magpie
   - All MFDs and pareto plots are built in get_candidates.py and plot_pareto.py respectively.
