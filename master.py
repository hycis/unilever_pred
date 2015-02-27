#!/usr/bin/env python

import argparse
import itertools

import datetime
import numpy as np
import sklearn
from sklearn import svm, preprocessing, cross_validation, linear_model, ensemble, tree
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from data import DataSet, TRAIN_FILENAME, TEST_FILENAME, TRAIN_RAW_FILENAME, TEST_RAW_FILENAME, load_names, gen_fake
from cluster import compute_clusters, get_non_na_only
from utils import write_pred, write_rank_path, filter_index, mse, drange


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


def log(file, line):
    file.write(line + "\n")
    file.flush()
    print(line)


def perform(actual, dataset, fsets, non_na_indexes, models, clusters, train_log, overall,
            ensemble_models, ens_params):
    out_suffixes = [""] * 3
    out_suffixes[0] = ("neutral", "norm")[dataset]

    if dataset == 0:
        train = DataSet("train_neutral.npy")
        test = DataSet("test_neutral.npy")
    else:
        train = DataSet(TRAIN_FILENAME)
        test = DataSet(TEST_FILENAME)

    for fset in fsets:
        out_suffixes[1] = fset
        if fset == 'non_na':
            if dataset == 0:
                continue
            train_features = train.data[:, non_na_indexes]
            test_features = test.data[:, non_na_indexes]
        else:
            train_features = getattr(train, 'features' + fset)
            test_features = getattr(test, 'features' + fset)
        train_features = preprocessing.scale(train_features)
        test_features = preprocessing.scale(test_features)

        for model in models:
            modelcls = model['model']
            modelparams = model['params']
            for params in modelparams:
                params = str_params(params)
                fn_name = modelcls.__name__
                out_suffix = ",".join(out_suffixes)

                np.random.seed(123)
                clf = modelcls(**params)

                if actual:
                    log(train_log, "{}, model={}: training with params={} ...".format(out_suffix, fn_name, params))
                    clf.fit(train_features, train.labels)
                    out_preds = clf.predict(test_features)
                    out_test_ids = test.ids
                    out_filename = get_filename(fn_name, params, out_suffix)
                    write_pred(out_filename, out_test_ids, out_preds)

                    if modelcls == tree.DecisionTreeRegressor:
                        tree.export_graphviz(clf, out_file=out_filename.replace(".csv", ".dot"))
                else:
                    scores = cross_validation.cross_val_score(clf, train_features, train.labels,
                                                              cv=5, scoring='mean_squared_error')
                    scoreline = "{}, model={}, mean={:.4f} std={:.4f}".format(out_suffix, fn_name, scores.mean(), scores.std())
                    log(train_log, scoreline)
                    overall.append({
                        "mse": abs(scores.mean()),
                        "model": modelcls,
                        "params": params,
                        "dataset": dataset,
                        "fset": fset,
                        "out_suffix": out_suffix,
                    })

                if actual:
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

                # if model is already an ensemble, then skip below.
                if fn_name in dir(ensemble):
                    continue

                for ensemble_model in ensemble_models:
                    if actual:
                        np.random.seed(123)
                        clf = modelcls(**params)
                        clf2 = ensemble_model(base_estimator=clf, **ens_params)

                        ens_out_filename = get_filename(ensemble_model.__name__, ens_params, "").replace(".csv", "_")
                        out_filename = ens_out_filename + get_filename(fn_name, params, out_suffix)
                        out_preds = clf2.predict(test_features)
                        out_test_ids = test.ids
                        write_pred(out_filename, out_test_ids, out_preds)
                        continue

                    for nestimators in [10, 20, 50, 100]:
                        ensemble_params = {'n_estimators': nestimators}
                        log(train_log, "{}, model={}, ens={} with params={} ...".format(out_suffix, fn_name, ensemble_model.__name__, ensemble_params))

                        np.random.seed(123)
                        clf = modelcls(**params)
                        clf2 = ensemble_model(base_estimator=clf, **ensemble_params)

                        scores = cross_validation.cross_val_score(clf2, train_features, train.labels,
                                                                  cv=5, scoring='mean_squared_error')
                        scoreline = "{}, model={}, mean={:.4f} std={:.4f}".format(out_suffix, fn_name, scores.mean(), scores.std())
                        log(train_log, scoreline)
                        overall.append({
                            "mse": abs(scores.mean()),
                            "model": modelcls,
                            "params": params,
                            "dataset": dataset,
                            "fset": fset,
                            "ensemble_params": ensemble_params,
                            "ensemble_model": ensemble_model,
                            "out_suffix": out_suffix,
                        })


def param_set(param_specs):
    param_names = []
    param_ranges = []
    for p in param_specs:
        name, specs = p.split('=')
        typ = float if '.' in specs else int
        param_names.append(name.strip())
        specs = map(str.strip, specs.split(','))
        specs = map(typ, specs)
        if len(specs) == 1:
            # constant value param.
            param_ranges.append(specs)
        else:
            if typ == float:
                # sometimes, due to floating point imprecision,
                # the "end" value is not included in the range.
                specs[1] += specs[2] - specs[2] / 10.
            param_ranges.append(drange(*specs))

    for params in itertools.product(*param_ranges):
        param_set = dict(zip(param_names, params))
        yield param_set


def get_class(class_path):
    if '.' not in class_path:
        for p2 in ('ensemble', 'tree', 'svm'):
            pp = getattr(getattr(sklearn, p2), class_path, None)
            if pp:
                return pp
        return None
    parts = class_path.split('.')
    cur = sklearn
    for p in parts:
        cur = getattr(cur, p)
    return cur


def main(args):
    suffixes = [
        args.transform,
        args.feature,
        'bal' if args.balance else '',
    ]
    data_names = load_names()
    file_suffix = ",".join(suffixes)
    model_cls = get_class(args.model)
    model_name = model_cls.__name__

    train = DataSet(TRAIN_RAW_FILENAME)
    test = DataSet(TEST_RAW_FILENAME)

    getattr(train, 'transform_' + args.transform)()
    getattr(test, 'transform_' + args.transform)()

    train_features = getattr(train, 'features_' + args.feature)
    test_features = getattr(test, 'features_' + args.feature)
    train_names = getattr(train, 'get_features_' + args.feature + '_names')(data_names)

    """
    sel = SelectKBest(chi2, k=2)
    feat_name = args.feature or 'features_no_ingre_prob'
    sel.fit(getattr(train, feat_name), train.labels)

    bestidxes = list(np.argsort(sel.scores_))
    bestidxes.reverse()
    """


    train_log = open("{}-{}.log".format(model_name, file_suffix), "a")
    train_log.write(SEP)
    train_log.write(str(datetime.datetime.now()) + "\n")
    train_log.write(SEP)
    train_log.flush()

    final_train_features = train_features
    final_train_labels = train.labels

    if args.balance:
        label_sizes = train.get_label_sizes()
        max_size = max(label_sizes.values())
        for (lbl, size) in label_sizes.iteritems():
            if size == max_size:
                continue
            idxes = filter_index(train.labels, lambda x: x == lbl)
            feats = train_features[idxes, :]
            n_samples = max_size - size
            fake = gen_fake(feats, max_size - size)
            final_train_features = np.concatenate((final_train_features, fake))
            final_train_labels = np.concatenate((final_train_labels, [lbl] * n_samples))
            log(train_log, "lbl={} n_samples={}".format(lbl, n_samples))

    overall = []

    #for nfeat in xrange(min(100, len(bestidxes)) + 1, 10, -10):

    for params in param_set(args.param):
        np.random.seed(123)
        clf = model_cls(**params)
        # feat_idxes = bestidxes[:nfeat]
        feat_idxes = xrange(0, len(final_train_features[0]))

        clf.fit(final_train_features[:, feat_idxes], final_train_labels)
        out_preds = clf.predict(test_features[:, feat_idxes])
        out_test_ids = test.ids
        out_filename = get_filename(model_name, params, file_suffix)
        out_filename_rank = get_filename(model_name, params, file_suffix + "_rank")
        write_pred(out_filename, out_test_ids, out_preds, dir=args.dest)
        write_rank_path(out_filename_rank, test, out_test_ids, out_preds, dir=args.dest)

        if getattr(clf, 'feature_importances_', None) is not None:
            fidxes = np.argsort(-clf.feature_importances_)
            for i in fidxes:
                line = "{:.6f}: {}".format(clf.feature_importances_[i], train_names[i])
                log(train_log, line)

        np.random.seed(123)
        if args.cv:
            clf = model_cls(**params)
            scores = cross_validation.cross_val_score(clf, final_train_features[:, feat_idxes], final_train_labels,
                                                      cv=5, scoring='mean_squared_error')
            mse_mean = scores.mean()
            mse_std = scores.std()
        else:
            train_preds = clf.predict(final_train_features[:, feat_idxes])
            diff = (train_preds - final_train_labels) ** 2
            mse_mean = (sum(diff) / len(diff))
            mse_std = np.std(diff)

        line = "mean={:.4f} std={:.4f} params={}".format(mse_mean, mse_std, params)
        log(train_log, line)
        overall.append({
            "mse": abs(mse_mean),
            "std": mse_std,
            "params": params,
        })

    overall = sorted(overall, key=lambda a: (a['mse'], a['std']))
    train_log.write(SEP)
    for a in overall:
        train_log.write("{:.4f}: std={:.4f}, params={}\n".format(
            a['mse'], a['std'], a['params'],))
    train_log.flush()


def main2():
    import sys
    train = DataSet(TRAIN_FILENAME)
    test = DataSet(TEST_FILENAME)
    clusters, prod_id_to_mask = compute_clusters(train, test)
    non_na_indexes = get_non_na_only(prod_id_to_mask)

    # feature sets
    fsets = ['_oo_only', 'non_na', '_no_ingre_prob', '_no_ingre', '',]

    models = [
        {
            "model": svm.SVR,
            "params": [
                {"kernel": "linear"},
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
            "params": [{"n_estimators": p[0], "min_samples_leaf": p[1], "max_depth": p[2]}
                       for p in itertools.product(xrange(5, 25, 5), xrange(5, 25, 5), xrange(5, 20, 5))],
        },
        {
            "model": ensemble.ExtraTreesRegressor,
            "params": [{"n_estimators": p[0], "min_samples_leaf": p[1], "max_depth": p[2]}
                       for p in itertools.product(xrange(5, 25, 5), xrange(5, 25, 5), xrange(5, 20, 5))],
        },
        {
            "model": tree.DecisionTreeRegressor,
            "params": [{"min_samples_leaf": p[0], "max_depth": p[1]}
                       for p in itertools.product(xrange(5, 25, 5), xrange(5, 20, 5))],
        },
        {
            "model": ensemble.GradientBoostingRegressor,
            "params": [{"n_estimators": p[0], "min_samples_leaf": p[1], "max_depth": p[2]}
                       for p in itertools.product(xrange(100, 200, 30), xrange(5, 25, 5), xrange(5, 20, 5))],
        },
    ]

    trainfilenamesuffix = ""
    if len(sys.argv) >= 2:
        model_idx = int(sys.argv[1])
        models = [models[model_idx]]
        trainfilenamesuffix = str(model_idx)

    train_log = open("train" + trainfilenamesuffix + ".log", "w+")
    train_log.write(SEP)
    train_log.write(str(datetime.datetime.now()) + "\n")
    train_log.write(SEP)
    train_log.flush()

    ensemble_models = [ensemble.BaggingRegressor, ensemble.AdaBoostRegressor]
    overall = []
    perform(False, 0, fsets, non_na_indexes, models, clusters, train_log, overall, ensemble_models, None)
    # perform(False, 1, fsets, non_na_indexes, models, clusters, train_log, overall, ensemble_models)

    overall = sorted(overall, key=lambda a: a['mse'])

    train_log.write(SEP)
    for a in overall:
        ens = a.get('ensemble_model').__name__ if a.get('ensemble_model') else ""
        ens_params = a.get('ensemble_params')
        train_log.write("{:.4f}: model={}, params={}, suffix={}, ens={}, ensParams={}\n".format(
            a['mse'], a['model'].__name__, a['params'], a['out_suffix'], ens, ens_params))
    train_log.flush()

    for a in overall[:50]:
        ens = a.get('ensemble_model')
        ens_params = a.get('ensemble_params')
        dataset = a['dataset']
        fset = a['fset']
        modelcls = a['model']
        params = a['params']
        perform(True, dataset, [fset], non_na_indexes, [{"model": modelcls, "params": [params]}],
                clusters, train_log, overall, [ens] if ens else [], ens_params)

    train_log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='learning module.')
    parser.add_argument("--dest", help="output dir name", default="results")
    parser.add_argument("-t", "--transform", help="feature transformation", required=True)
    parser.add_argument("-f", "--feature", help="feature set to use", required=True)
    parser.add_argument("-m", "--model", help="model to use", required=True)
    parser.add_argument('-p', '--param', help="model param: param_name=start,end,step", nargs='+')
    parser.add_argument("--cv", help="do cross validation",
                        action="store_true", default=False)
    parser.add_argument("--balance", help="balance",
                        action="store_true", default=False)
    args = parser.parse_args()
    main(args)

