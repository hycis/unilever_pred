# unilever_pred
This repo for the unilever product score prediction competition

__Things To Do__
1. Find the best number of top features using cross-validation
2. Find the best number of estimators and learning rate for GradientBoostingRegressor using cross-validation

Model | Cross-Validation Score | Test Score | Remarks | Features
----- | ---------------------- | ---------- | --------| --------
GradientBoostingRegressor |          | 0.222446 | Learning-rate = 0.1, n_estimators = 100 | [158:]
GradientBoostingRegressor |          | 0.221658 | Learning-rate = 0.1, n_estimators = 100 | [1:]
AverageModel              | 0.195236 | 0.222848 | Average of rfr, etr, gbr, br, br                     | [158:]
AverageModel              | 0.196189 |          | Average of rfr, etr, gbr, br, br(br)                 | Top 50
AverageModel              | 0.192408 | 0.221837 | Average of rfr, etr, gbr, br, br, svr                | Top 50
AverageModel              | 0.192768 |  | Average of rfr, gbr, br, br, svr                             | Top 50
AverageModel              | 0.191267 | 0.218865 | Average of  rfr, etr, svr, gbr, br, br(gbr), br(gbr) | Top 50