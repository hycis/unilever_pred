#!/usr/bin/env python

import sys

import numpy as np
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from data import load_names, DataSet, TRAIN_FILENAME, TEST_FILENAME
from utils import filter_index


def avgfeats(filename, names, features, labels):
    with open(filename, "wb") as f:
        f.write("label")
        for name in names:
            f.write(","+name)
        f.write("\n")

        for s in xrange(1, 8):
            idxes = filter_index(labels, lambda x : x == s)
            feats = features[idxes, :]
            avg = np.average(feats, axis=0)
            f.write("{}".format(s))
            for a in avg:
                f.write(",{:.4f}".format(a))
            f.write("\n")


def main():
    names = np.array(load_names())
    train = DataSet("train_neutral.npy")
    test = DataSet("test_neutral.npy")
    feat_indexes = train.features_no_ingre_prob_indexes
    names_no_ingre_prob = names[feat_indexes, :]

    avgfeats("avgfeats.csv", names_no_ingre_prob, train.features_no_ingre_prob, train.labels)

    sel = SelectKBest(chi2, k=2)
    sel.fit(train.features_no_ingre_prob, train.labels)

    with open("feats-scores.csv", "wb") as f:
        for name in names_no_ingre_prob:
            f.write(name + ",")
        f.write("\n")
        for s in sel.scores_:
            f.write("{:.4f},".format(s))
        f.write("\n")

    bestidxes = list(np.argsort(sel.scores_))
    bestidxes.reverse()
    avgfeats("bestfeats.csv", names_no_ingre_prob[bestidxes, :], train.features_no_ingre_prob[:, bestidxes], train.labels)

    clf = svm.LinearSVC(C=0.01, penalty="l1", dual=False)
    clf.fit(train.features_no_ingre_prob, train.labels)

    i = 1
    for classweights in clf.coef_:
        print("Label={} :".format(i))
        absweights = map(abs, classweights)
        sorted_indexes = np.argsort(absweights)
        for idx in sorted_indexes[-20:]:
            print("\t{:.5f}: {}".format(classweights[idx], names_no_ingre_prob[idx]))
        print("")
        i += 1


if __name__ == "__main__":
    main()
