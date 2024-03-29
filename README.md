# unilever_pred
This repo for the unilever product score prediction competition

__Things To Do__
1. Find the best number of top features using cross-validation
2. Find the best number of estimators and learning rate for GradientBoostingRegressor using cross-validation

# TODO: how to interpolate test data?
# TODO: check out the feature weights
# TODO: recursive feature elimination
# TODO: do the avg plots and auto-detect gradient
# TODO: interpolatio by label
# TODO: which ingredient correlate with which attribute
# TODO: correlation with OO score

Test Labels
===========
135,6
2212,6
test size=2525

=======
Model ID | Model | Cross-Validation Score | Test Score | Remarks | Features
---------|------ | ---------------------- | ---------- | --------| --------
        | GradientBoostingRegressor  | | 0.222446 | Learning-rate = 0.1, n_estimators = 100 | [158:]
        | GradientBoostingRegressor  | | 0.221658 | Learning-rate = 0.1, n_estimators = 100 | [1:]
        | GradientBoostingRegressor  | | 0.218353 | Learning-rate = 0.1, n_estimators = 100-140, max-depth=5 or 10 | top 101  features
        | GradientBoostingRegressor  | | 0.217173 | Learning-rate = 0.08, n_estimators = 100-140, max-depth=5,7,9 | na_Zero,no_ingre_prob
        | GradientBoostingRegressor  | | 0.21678 | Learning-rate = 0.08, n_estimators = 140, max-depth=7 | na_Zero,no_ingre_prob
        | GradientBoostingRegressor  | | 0.21366 | learning_rate=0.07, n_estimators=200, max-depth=6 | na_zero,no_ingre_prob
        | GradientBoostingRegressor  | | 0.212674 | learning_rate=0.07, n_estimators=280, max-depth=6 | na_zero,no_ingre_prob
        | GradientBoostingRegressor  | | 0.212533 | learning_rate=0.07, n_estimators=380, max-depth=6 | na_zero,no_ingre_prob
        | GradientBoostingRegressor |          | 0.222446 | Learning-rate = 0.1, n_estimators = 100 | [158:]
        | GradientBoostingRegressor |          | 0.221658 | Learning-rate = 0.1, n_estimators = 100 | [1:]
        | AverageModel              | 0.195236 | 0.222848 | Average of rfr, etr, gbr, br, br                     | [158:]
        | AverageModel              | 0.196189 |          | Average of rfr, etr, gbr, br, br(br)                 | Top 50
        | AverageModel              | 0.192408 | 0.221837 | Average of rfr, etr, gbr, br, br, svr                | Top 50
        | AverageModel              | 0.192768 |  | Average of rfr, gbr, br, br, svr                             | Top 50
        | AverageModel              | 0.191267 | 0.218865 | Average of  rfr, etr, svr, gbr, br, br(gbr), br(gbr) | Top 50
ave_model1 | AverageMode            | 0.1896 | 0.21511  |                                                        | Top 100


Phase 2: MSE

Rank

Model| CV score  | Pub Score | Params
-----|-----------|-----------|--------
RF   |           | .483824   | max_depth=4_max_features=15_n_estimators=350
RF   |           | .4103     | max_depth=5_max_features=15_n_estimators=250
RF   |           | .398529   | max_depth=5_max_features=12_n_estimators=210
RF   |           | .397794   | max_depth=5_max_features=12_n_estimators=210 and max_depth=5_max_features=15_n_estimators=250

Model| CV score  | Pub Score | Params
-----|-----------|-----------|--------
GBR  |           | 0.5156    | learning_rate=0.05_max_depth=2_n_estimators=200
GBR  |           | 0.5153    | learning_rate=0.05_max_depth=2_n_estimators=150
GBR  |           | 0.5152    | learning_rate=0.04_max_depth=2_n_estimators=140
