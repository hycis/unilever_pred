# TODO: how to interpolate test data?
# TODO: check out the feature weights
# TODO: recursive feature elimination

# unilever_pred
This repo for the unilever product score prediction competition

Model | Cross-Validation Score | Test Score | Remarks | Features
------------- | ------------- | ----------- | --------| --------
GradientBoostingRegressor  | | 0.222446 | Learning-rate = 0.1, n_estimators = 100 | [158:]
GradientBoostingRegressor  | | 0.221658 | Learning-rate = 0.1, n_estimators = 100 | [1:]
GradientBoostingRegressor  | | 0.218353 | Learning-rate = 0.1, n_estimators = 100-140, max-depth=5 or 10 | top 101 features
GradientBoostingRegressor  | | 0.217173 | Learning-rate = 0.08, n_estimators = 100-140, max-depth=5,7,9 | na_Zero,no_ingre_prob
GradientBoostingRegressor  | | 0.21678 | Learning-rate = 0.08, n_estimators = 140, max-depth=7 | na_Zero,no_ingre_prob


135,6
2212,6

test size=2525
