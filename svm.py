#!/usr/bin/env python

from __future__ import unicode_literals

import numpy as np
from sklearn import svm

from utils import write_pred


def filter_index(arr, filter_fn):
    """
    :param arr:
    :param filter_fn:
    :return: Returns array of indexes of elements in arr that satisfies the filter.
    """
    indexes = []
    i = 0
    for elem in arr:
        if filter_fn(elem):
            indexes.append(i)
        i += 1
    return indexes


def find_max(arr):
    themax = arr[0]
    theindex = 0
    for i in xrange(1, len(arr)):
        if arr[i] > themax:
            themax = arr[i]
            theindex = i
    return theindex


def main():
    train = np.load('train.npy')
    test = np.load('test.npy')

    train_ids = train[:, 0] # first column
    train_prod_ids = train[:, 1]
    train_unique_prod_ids = set(train_prod_ids)
    train_labels = train[:, len(train[0]) - 1] # last column
    train = train[:, 2:len(train[0]) - 1] # features
    train_no_ingre = train[:, 153:] # features

    test_ids = test[:, 0] # first column
    test_prod_ids = test[:, 1]
    test = test[:, 2:len(test[0]) - 1] # features
    test_no_ingre = test[:, 153:] # features

    # Train products and test products are disjoint!!

    clf = svm.SVC()
    clf.fit(train, train_labels)
    out_preds = clf.predict(test)
    out_test_ids = test_ids
    write_pred("svm_preds.csv", out_test_ids, out_preds)
    
    clf = svm.SVC()
    clf.fit(train_no_ingre, train_labels)
    out_preds = clf.predict(test_no_ingre)
    out_test_ids = test_ids
    write_pred("svm_preds_no_ingre.csv", out_test_ids, out_preds)


    """
    By product:
    svms = {}

    for prod_id in train_unique_prod_ids:
        prod_id = int(prod_id)
        indexes = filter_index(train_prod_ids, lambda x: int(x) == prod_id)
        print("training prod={} size={} ...".format(prod_id, len(indexes)))
        clf = svm.SVC(probability=True)
        clf.fit(train[indexes, :], train_labels[indexes])
        svms[prod_id] = clf

    out_test_ids = test_ids
    out_preds = []
    for test_row in test:
        best_pred = 0
        best_prob = -1e99
        for clf in svms.values():
            # get log probability for each class
            probs = clf.predict_log_proba(test_row)
            max_index = find_max(probs[0])
            max_prob = probs[0][max_index]
            if max_prob > best_prob:
                best_pred = clf.classes_[max_index]
                best_prob = max_prob
        out_preds.append(best_pred)

    write_pred("svm_preds_by_prod.csv", out_test_ids, out_preds)
    """


if __name__ == "__main__":
    main()

