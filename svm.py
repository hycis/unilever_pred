#!/usr/bin/env python

from __future__ import unicode_literals

import numpy as np
from sklearn import svm

from utils import write_pred, filter_index, find_max
from data import DataSet, TEST_FILENAME, TRAIN_FILENAME


def main():
    train = DataSet(TRAIN_FILENAME)
    test = DataSet(TEST_FILENAME)

    # Train products and test products are disjoint!!

    clf = svm.SVC()
    clf.fit(train.features, train.labels)
    out_preds = clf.predict(test.features)
    out_test_ids = test.ids
    write_pred("svm_preds.csv", out_test_ids, out_preds)

    clf = svm.SVC()
    clf.fit(train.features_no_ingre, train.labels)
    out_preds = clf.predict(test.features_no_ingre)
    out_test_ids = test.ids
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

