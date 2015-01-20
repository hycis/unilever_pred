from __future__ import unicode_literals

import numpy as np
import itertools

import random

from parse import AGREE_INDEXES
from utils import interval, filter_index, csv_reader_utf8, xinterval


TRAIN_FILENAME = 'train.npy'
TEST_FILENAME = 'test.npy'
TRAIN_RAW_FILENAME = 'train_raw.npy'
TEST_RAW_FILENAME = 'test_raw.npy'


def load_names():
    csv = csv_reader_utf8("desc.csv")
    for row in csv:
        break
    return [row[1] for row in csv]


def _to_int(a):
    """
    Transform to int in-place.
    """
    for i in xrange(0, len(a)):
        a[i] = int(round(a[i]))


class DataSet(object):
    COMBINED_INDEX_LIST = [
        xinterval(195, 204), # smell
        xinterval(205, 214), # strength of smell
        xinterval(230, 241), # problems
        xinterval(242, 253), # problems additional
    ]
    COMBINED_SUM_LIST = [
        False,
        False,
        True,
        True,
    ]

    def __init__(self, filename=None, data=None):
        if data is not None:
            self.data = data
        else:
            self.data = data = np.load(filename)
        self.ids = data[:, 0] # first column
        self.prod_ids = data[:, 1]
        self.labels = data[:, -1] # last column
        self.features = data[:, 2:-1] # exclude first 2 and last columns
        _to_int(self.ids)
        _to_int(self.prod_ids)
        _to_int(self.labels)
        self.unique_prod_ids = set(self.prod_ids)
        self.unique_labels = set(self.labels)

    def get_prod_indexes(self, id):
        return filter_index(self.prod_ids, lambda x: x == id)

    def transform_na_interpolated(self):
        """ Interpolate all NA. Transform in-place. """
        data = self.data
        agree_count = len(AGREE_INDEXES)
        random.seed(123)

        for lbl in self.unique_labels:
            lbl_idxes = filter_index(self.labels, lambda x: x == lbl)
            lbl_data = data[lbl_idxes, :]
            non_na = [
                (filter(lambda x: x >= 0, lbl_data[:, AGREE_INDEXES[i]]) or [0]) for i in xrange(0, agree_count)
            ]
            for rowidx in lbl_idxes:
                row = data[rowidx, :]
                for i in xrange(0, agree_count):
                    idx = AGREE_INDEXES[i]
                    if row[idx] < 0:
                        row[idx] = non_na[i][random.randint(0, len(non_na[i]) - 1)]
        # change remaining NA to 0
        self.transform_na_zero()

    def transform_na_zero(self):
        """ Sets all NA (-1) to 0. Transform in-place.  """
        for row in self.features:
            for i in xrange(0, len(row)):
                if row[i] < 0:
                    row[i] = 0

    def get_label_sizes(self):
        sizes = {lbl: 0 for lbl in self.unique_labels}
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
        return interval(155, 229) + interval(254, len(self.data[0]) - 2)

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

    @property
    def features_combined(self):
        leftover_idxes = list(set(interval(155, len(self.data[0])-2)) - set(itertools.chain(*self.COMBINED_INDEX_LIST)))
        leftover_count = len(leftover_idxes)
        new = np.empty((len(self.data), len(self.COMBINED_INDEX_LIST) + len(leftover_idxes)))
        new[:, 0:leftover_count] = self.data[:, leftover_idxes]

        for rowidx in xrange(0, len(self.data)):
            rangeidx = leftover_count
            i = 0
            for therange in self.COMBINED_INDEX_LIST:
                non_na = filter(lambda x: x > 0, self.data[rowidx, therange])
                if non_na:
                    if self.COMBINED_SUM_LIST[i]:
                        avg = sum(non_na)
                    else:
                        avg = sum(non_na) / float(len(non_na))
                else:
                    avg = 0 # TODO: interpolate this?
                new[rowidx, rangeidx] = avg
                rangeidx += 1
                i += 1
        return new


def count_unique(feats):
    feat_count = len(feats[0])
    counts = [{} for _ in xrange(0, feat_count)]

    for row in feats:
        for i in xrange(0, feat_count):
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
    feat_count = len(feats[0])
    random.seed(123)

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

    rows = np.empty((n_samples, feat_count))
    for row in rows:
        for j in xrange(0, len(row)):
            row[j] = gen_random[j](j)

    return rows


