#!/usr/bin/env python

from __future__ import unicode_literals

import numpy as np
from sklearn import svm

from utils import write_pred


def main():
    train = np.load('train.npy')
    test = np.load('test.npy')

    train_ids = train[:, 0] # first column
    train_labels = train[:, len(train[0]) - 1] # last column
    train = train[:, 1:len(train[0]) - 1] # features

    test_ids = test[:, 0] # first column
    test = test[:, 1:len(test[0]) - 1] # features

    clf = svm.SVC()
    clf.fit(train, train_labels)
    preds = clf.predict(test)
    write_pred("svm_preds.csv", test_ids, preds)


if __name__ == "__main__":
    main()

