#!/usr/bin/env python

import numpy as np
from sklearn import svm

from data import load_names, DataSet, TRAIN_FILENAME, TEST_FILENAME


def main():
    names = np.array(load_names())
    train = DataSet("train_neutral.npy")
    test = DataSet("test_neutral.npy")
    feat_indexes = train.features_no_ingre_prob_indexes
    names_no_ingre_prob = names[feat_indexes, :]

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
