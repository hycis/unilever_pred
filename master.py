#!/usr/bin/env python

# TODO: train svm per product and test using each svm, then avg the score

import datetime
import numpy as np
from sklearn import svm, preprocessing, cross_validation, linear_model, ensemble, tree

from data import DataSet, TRAIN_FILENAME, TEST_FILENAME
from cluster import compute_clusters
from utils import write_pred, filter_index, mse


SEP = "=" * 80 + "\n"


def get_filename(fn_name, params, suffix):
    keys = list(params.keys())
    keys.sort()
    pairs = []
    for key in keys:
        value = params[key]
        pairs.append(key + "=" + str(value))
    if suffix:
        suffix = suffix + "_"
    return fn_name + "_" + suffix + "_".join(pairs) + ".csv"


def str_params(p):
    return {str(k): v for (k, v) in p.iteritems()}


def do_cluster(modelcls, **kwargs):
    params = kwargs['params']
    clusters = kwargs['clusters']
    train_prod_ids = kwargs['train_prod_ids']
    train_features = kwargs['train_features']
    train_labels = kwargs['train_labels']
    test_ids = kwargs['test_ids']
    test_prod_ids = kwargs['test_prod_ids']
    test_features = kwargs['test_features']

    clfs = {}
    prod_id_to_cluster_idx = {}
    i = 0
    for cluster in clusters:
        for pid in cluster:
            prod_id_to_cluster_idx[pid] = i
        i += 1

    i = 0
    for cluster in clusters:
        print("Training for cluster={} ...".format(i))
        indexes = filter_index(train_prod_ids, lambda x: x in cluster)
        np.random.seed(123)
        clf = modelcls(**params)
        clf.fit(train_features[indexes, :], train_labels[indexes, :])
        clfs[i] = clf
        i += 1

    out_test_ids = []
    out_preds = []
    for prod_id in set(test_prod_ids):
        print("Testing prod_id={} ...".format(prod_id))
        indexes = filter_index(test_prod_ids, lambda x: x == prod_id)
        clf = clfs[prod_id_to_cluster_idx[prod_id]]
        preds = clf.predict(test_features[indexes, :])
        for p in preds:
            out_preds.append(p)
        for p in test_ids[indexes, :]:
            out_test_ids.append(p)

    return out_test_ids, out_preds, None


def main():
    train = DataSet(TRAIN_FILENAME)
    test = DataSet(TEST_FILENAME)
    clusters, _ = compute_clusters(train, test)

    # feature sets
    fsets = ['', '_no_ingre', '_no_ingre_prob']

    models = [
        {
            "model": svm.SVR,
            "params": [
                #{"kernel": "linear"},
                {"kernel": "rbf"},
            ],
        },
        {
            "model": linear_model.LogisticRegression,
            "params": [
                {"penalty": "l1", "dual": False},
                {"penalty": "l2", "dual": False},
                {"penalty": "l2", "dual": True},
            ],
        },
        {
            "model": linear_model.LinearRegression,
            "params": [
                {},
            ],
        },
        {
            "model": ensemble.RandomForestRegressor,
            "params": [
                {"n_estimators": 10},
                {"n_estimators": 20},
                {"n_estimators": 30},
                {"n_estimators": 100},
            ],
        },
        {
            "model": ensemble.ExtraTreesRegressor,
            "params": [
                {"n_estimators": 10},
                {"n_estimators": 20},
                {"n_estimators": 30},
                {"n_estimators": 100},
            ],
        },
        {
            "model": tree.DecisionTreeRegressor,
            "params": [
                {"n_estimators": 10},
                {"n_estimators": 20},
                {"n_estimators": 30},
                {"n_estimators": 100},
            ],
        },
    ]

    train_log = open("train.log", "w+")
    train_log.write(SEP)
    train_log.write(str(datetime.datetime.now()) + "\n")
    train_log.write(SEP)

    overall = []

    for fset in fsets:
        print("Feature set={}".format(fset))
        train_features = getattr(train, 'features' + fset)
        test_features = getattr(test, 'features' + fset)
        out_suffix = fset

        for to_scale in [0, 1]:
            out_suffix += "_scaled" if to_scale else ""
            for model in models:
                modelcls = model['model']
                modelparams = model['params']
                for params in modelparams:
                    np.random.seed(123)
                    params = str_params(params)
                    fn_name = modelcls.__name__
                    print("{}: training with params={} ...".format(fn_name, params))
                    clf = modelcls(**params)
                    clf.fit(train_features, train.labels)
                    out_preds = clf.predict(test_features)
                    out_test_ids = test.ids
                    out_filename = get_filename(fn_name, params, out_suffix)
                    write_pred(out_filename, out_test_ids, out_preds)
                    train_preds = clf.predict(train_features)
                    this_mse = mse(train.labels, train_preds)
                    scoreline = "{}: mse={:.4f}\n".format(fn_name, this_mse)
                    train_log.write(scoreline)
                    train_log.flush()
                    print(scoreline)
                    overall.append({"mse": this_mse, "filename": out_filename})

                    print("{}: CV with params={} ...".format(fn_name, params))
                    np.random.seed(123)
                    clf = modelcls(**params)
                    scores = cross_validation.cross_val_score(clf, train_features, train.labels,
                                                              cv=5, scoring='mean_squared_error')
                    out_preds = clf.predict(test_features)
                    out_test_ids = test.ids
                    out_filename = "cv_" + get_filename(fn_name, params, out_suffix)
                    write_pred(out_filename, out_test_ids, out_preds)
                    train_preds = clf.predict(train_features)
                    this_mse = mse(train.labels, train_preds)
                    scoreline = "{}: mean={:.4f} std={:.4f}\n".format(fn_name, scores.mean(), scores.std())
                    train_log.write(scoreline)
                    train_log.flush()
                    print(scoreline)
                    overall.append({"mse": this_mse, "filename": out_filename})

                    args = {
                        "params": params,
                        "clusters": clusters,
                        "train_prod_ids": train.prod_ids,
                        "train_features": train_features,
                        "train_labels": train.labels,
                        "test_ids": test.ids,
                        "test_prod_ids": test.prod_ids,
                        "test_features": test_features,
                    }
                    args = str_params(args)
                    out_test_ids, out_preds, _ = do_cluster(modelcls, **args)
                    out_filename = "cluster_" + get_filename(fn_name, params, out_suffix)
                    write_pred(out_filename, out_test_ids, out_preds)
                    train_preds = clf.predict(train_features)
                    this_mse = mse(train.labels, train_preds)
                    scoreline = "{}: mse={:.4f}\n".format(fn_name, this_mse)
                    train_log.write(scoreline)
                    train_log.flush()
                    print(scoreline)
                    overall.append({"mse": this_mse, "filename": out_filename})

                    for ensemble_model in [ensemble.BaggingRegressor, ensemble.AdaBoostRegressor]:
                        for nestimators in [10, 20, 50, 100]:
                            ensemble_params = {'n_estimators': nestimators}
                            print("{}: {} ensemble with params={} ...".format(fn_name, ensemble_model.__name__, ensemble_params))
                            np.random.seed(123)
                            clf = modelcls(**params)
                            clf2 = ensemble_model(base_estimator=clf, **ensemble_params)
                            clf2.fit(train_features, train.labels)
                            out_preds = clf2.predict(test_features)
                            out_test_ids = test.ids

                            ens_out_filename = get_filename(ensemble_model.__name__, ensemble_params, "").replace(".csv", "_")
                            out_filename = ens_out_filename + get_filename(fn_name, params, out_suffix)
                            write_pred(out_filename, out_test_ids, out_preds)
                            train_preds = clf.predict(train_features)
                            this_mse = mse(train.labels, train_preds)
                            scoreline = "{}: mse={:.4f}\n".format(fn_name, this_mse)
                            train_log.write(scoreline)
                            train_log.flush()
                            print(scoreline)
                            overall.append({"mse": this_mse, "filename": out_filename})

            train_features = preprocessing.scale(train_features)
            test_features = preprocessing.scale(test_features)


    train_log.write(SEP)
    overall = sorted(overall, key=lambda a: a['mse'])
    for a in overall:
        train_log.write("{:.4f}: {}\n".format(a['mse'], a['filename']))
    train_log.close()


if __name__ == "__main__":
    main()
