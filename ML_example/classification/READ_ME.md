## binary_classifier.py

   - Splits, permutes and featurises dataset
   - Also limits the size of studied phase fields, 4 elements were used here but this can be changed provided it is continued throughout
   - Tests 8 different classifier models (5 variations of decision trees, k-nearest neighbours and 2 support vector machines) and ranks based on MCC
   - After the best model has been selected, it can be used to classify query data
   - Outputs scores acheived by each model in testing, and final canidates which are both likely to contain a kagome structure and are chemically feasible

## classifier_models.py

   - Different classifier models with hyperparameters selected from qualitative grid searches
