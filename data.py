from __future__ import unicode_literals

import numpy as np

import random

from utils import interval, filter_index, csv_reader_utf8


TRAIN_FILENAME = 'train.npy'
TEST_FILENAME = 'test.npy'


def load_names():
    csv = csv_reader_utf8("desc.csv")
    for row in csv:
        break
    return [row[1] for row in csv]


class DataSet(object):
    def __init__(self, filename):
        self.data = data = np.load(filename)
        self.ids = np.array(map(lambda x: int(round(x)), data[:, 0])) # first column
        self.prod_ids = np.array(map(lambda x: int(round(x)), data[:, 1]))
        self.unique_prod_ids = set(self.prod_ids)
        self.labels = data[:, -1] # last column
        self.features = data[:, 2:-1] # exclude first 2 and last columns

    def get_prod_indexes(self, id):
        return filter_index(self.prod_ids, lambda x: x == id)

    def get_label_sizes(self):
        sizes = {}
        for lbl in set(self.labels):
            sizes[lbl] = 0
        for lbl in self.labels:
            sizes[lbl] += 1
        return sizes

    @property
    def features_no_ingre(self):
        """
        :return: Features without ingredients.
        """
        return self.data[:, 155:-1]

    @property
    def features_no_ingre_prob_indexes(self):
        return interval(155, 230) + interval(254, len(self.data[0]) - 2)

    @property
    def features_no_ingre_prob(self):
        """
        :return: Features without ingredients and without the optional problems.
        """

        return self.data[:, self.features_no_ingre_prob_indexes]

    @property
    def features_oo_only_indexes(self):
        return interval(155, 161) + [216, 277, 278, 287, 288, 293]

    @property
    def features_oo_only(self):
        """
        :return: overall opinion only
        """
        indexes = self.features_oo_only_indexes
        return self.data[:, indexes]


def count_unique(feats):
    counts = []
    for i in xrange(0, len(feats[0])):
        counts.append({})
    for row in feats:
        for i in xrange(0, len(feats[0])):
            counts[i][row[i]] = counts[i].get(row[i], 0) + 1
    return counts


def _create(lst, total):
    def a(i):
        r = random.randint(0, total - 1)
        for p in lst:
            if r < p[1]:
                return p[0]
        raise Exception("invalid")
    return a


def gen_fake(feats, n_samples):
    """
    :param feats: Rows of features for a specific label.
    :return:
    """
    std = np.std(feats, axis=0)
    for i in xrange(0, len(std)):
        if std[i] < 0.00001:
            std[i] = 0.000001
    avg = np.average(feats, axis=0)

    gen_random = count_unique(feats)
    for i in xrange(0, len(gen_random)):
        if len(gen_random[i]) > 10:
            gen_random[i] = lambda i: (np.random.normal(loc=avg[i], scale=std[i]))
            continue
        lst = []
        accu = 0
        for (k, v) in gen_random[i].iteritems():
            accu += v
            lst.append((k, accu))
        gen_random[i] = _create(lst, accu)


    rows = []
    for i in xrange(0, n_samples):
        row = [0] * len(feats[0])
        rows.append(row)
        for j in xrange(0, len(row)):
            row[j] = gen_random[j](j)

    return np.array(rows)
