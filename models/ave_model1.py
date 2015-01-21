


from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
import numpy as np
import operator
from utils import write_pred

from sklearn.metrics import mean_squared_error


def log(file, line):
    file.write(line + "\n")
    file.flush()
    print(line)

if __name__ == '__main__':
    from data import DataSet
    from sklearn.ensemble import *
    from sklearn.linear_model import *
    from sklearn.svm import SVR
    from sklearn import cross_validation
    from sklearn.cross_validation import KFold

    clf2 = RandomForestRegressor()
    clf3 = ExtraTreesRegressor()
    clf4 = SVR()
    clf5 = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, loss='ls')
    clf6 = BaggingRegressor(n_estimators=10)
    clf7 = BaggingRegressor(base_estimator=clf5, n_estimators=10, max_samples=0.8)
    clf8 = BaggingRegressor(base_estimator=clf5, n_estimators=10, max_samples=0.8)
    clf9 = BaggingRegressor(base_estimator=clf2, n_estimators=10, max_samples=0.8)
    clf10 = AdaBoostRegressor(base_estimator=clf5)
    clf11 = BaggingRegressor(base_estimator=clf5, n_estimators=10, max_samples=0.8)
    clf12 = BaggingRegressor(base_estimator=clf5, n_estimators=10, max_samples=0.8)
    clf13 = BaggingRegressor(base_estimator=clf5, n_estimators=10, max_samples=0.8)


    clfs = [clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9, clf10, clf11, clf12, clf13]

    eclf = EnsembleClassifier(clfs=[clf2, clf6, clf5])
    data = DataSet('../dataset/train.npy')
    idx = data.idx_by_gradient_boost(topk=100)
    X = data.features_no_ingre[:,idx]
    # X = data.features_no_ingre
    y = data.labels
    logfout = open('log.txt','wb')

    kf = KFold(n=X.shape[0], n_folds=5)

    errors = {}

    for clf in clfs:
        errors[clf.__class__.__name__] = []

    errors['mean_model'] = []

    for train, test in kf:
        y_train_preds = []
        y_test_preds = []
        for clf in clfs:
            log(logfout, 'fitting: ' + clf.__class__.__name__)
            clf.fit(X[train], y[train])
            y_train_pred = clf.predict(X[train])
            log(logfout, 'train score: %.4f'%mean_squared_error(y_train_pred, y[train]))
            y_test_pred = clf.predict(X[test])
            mse = mean_squared_error(y_test_pred, y[test])
            log(logfout, 'test score: %.4f'%mse)
            y_train_preds.append(y_train_pred)
            y_test_preds.append(y_test_pred)

            errors[clf.__class__.__name__].append(mse)

        y_trains = np.asarray(y_train_preds)
        y_tests = np.asarray(y_test_preds)
        y_test_pred = y_tests.swapaxes(0,1).mean(axis=1)
        mse = mean_squared_error(y_test_pred, y[test])
        log(logfout, 'mean test score: %.4f'%mse)
        errors['mean_model'].append(mse)
        print

    for k in errors:
        log(logfout, "%s: %.4f"%(k, np.mean(errors[k])))

    test = DataSet('../dataset/test.npy')
    test_X = test.features_no_ingre[:,idx]
    # test_X = test.features_no_ingre

    y_preds = []
    for clf in clfs:
        print clf.__class__.__name__
        clf.fit(X, y)
        y_preds.append(clf.predict(test_X))

    y_npy = np.asarray(y_preds).mean(axis=0)

    write_pred(filename='mean_model.csv', ids=test.ids, preds=y_npy)
