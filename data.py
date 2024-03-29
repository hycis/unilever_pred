from __future__ import unicode_literals

import numpy as np
import itertools

import random

from parse import AGREE_INDEXES
from utils import interval, filter_index, csv_reader_utf8, xinterval

from sklearn.ensemble import GradientBoostingRegressor


TRAIN_FILENAME = 'train.npy'
TEST_FILENAME = 'test.npy'
TRAIN_RAW_FILENAME = 'train_raw.npy'
TEST_RAW_FILENAME = 'test_raw.npy'


def load_names():
    csv = csv_reader_utf8("desc.csv")
    for row in csv:
        break
    return np.array([row[1] for row in csv])


def _to_int(a):
    """
    Returns a new list.
    """
    return map(lambda x: int(round(x)), a)


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
    COMBINED_PROB_INDEX_LIST = [
        xinterval(230, 241), # problems
        xinterval(242, 253), # problems additional
    ]
    COMBINED_PROB_SUM_LIST = [
        True,
        True,
    ]
    INTERPOLATION_ROW_INDEX_LIST = [
        xinterval(162, 194),
        xinterval(195, 204), # smell
        xinterval(205, 214), # strength of smell
        xinterval(282, 286), # amount of foam
    ]

    def __init__(self, filename=None, data=None):
        if data is not None:
            self.data = data
        else:
            self.data = data = np.load(filename)
        self.ids = _to_int(data[:, 0]) # first column
        self.prod_ids = _to_int(data[:, 1])
        self.labels = _to_int(data[:, -1]) # last column
        self.features = data[:, 2:-1] # exclude first 2 and last columns
        self.unique_prod_ids = set(self.prod_ids)
        self.unique_labels = set(self.labels)

    def get_prod_indexes(self, id):
        return filter_index(self.prod_ids, lambda x: x == id)

    def transform_rank_internal(self, r):
        labels = np.array(self.labels)
        scores = []
        for pid in set(self.prod_ids):
            pid_idxes = filter_index(self.prod_ids, lambda x : x == pid)
            pid_labels = labels[pid_idxes]
            count = 0
            for lbl in pid_labels:
                if lbl == r:
                    count += 1
            labels[pid_idxes] = count
        self.labels = labels

    def transform_rank1(self):
        self.transform_rank_internal(1)

    def transform_rank2(self):
        self.transform_rank_internal(2)

    def transform_rank3(self):
        self.transform_rank_internal(3)

    def transform_rank4(self):
        self.transform_rank_internal(4)

    def transform_rank5(self):
        self.transform_rank_internal(5)

    def transform_rank6(self):
        self.transform_rank_internal(6)

    def transform_rank7(self):
        self.transform_rank_internal(7)

    def transform_rank(self):
        labels = np.array(self.labels)
        scores = []
        for pid in set(self.prod_ids):
            pid_idxes = filter_index(self.prod_ids, lambda x : x == pid)
            pid_labels = labels[pid_idxes]
            avg = sum(pid_labels) / len(pid_labels)
            scores.append({'pid': pid, 'avg': avg})
        scores = sorted(scores, key=lambda x: -x['avg'])
        num_prods = len(self.unique_prod_ids)
        for rank in xrange(1, num_prods + 1):
            score = scores[rank - 1]
            pid = score['pid']
            pid_idxes = filter_index(self.prod_ids, lambda x : x == pid)
            labels[pid_idxes] = (rank - 1.) / (num_prods - 1.) * 27 + 1
        self.labels = labels

    def transform_na_interpolated_row(self):
        """ Row-wise interpolation. Transform in-place. """
        data = self.data
        range_count = len(self.INTERPOLATION_ROW_INDEX_LIST)
        random.seed(123)

        for row in data:
            for rng in self.INTERPOLATION_ROW_INDEX_LIST:
                non_na = (filter(lambda x: x >= 0, row[:, rng]) or [0])

                for i in rng:
                    if row[i] < 0:
                        row[i] = non_na[random.randint(0, len(non_na) - 1)]

        # change remaining NA to 0
        self.transform_na_zero()

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
    def features_ingre_indexes(self):
        return interval(2, 154)

    def get_features_ingre_names(self, names):
        return names[(self.features_ingre_indexes)]

    @property
    def features_ingre(self):
        """
        :return: Features only ingredients.
        """
        return self.data[:, self.features_ingre_indexes]

    @property
    def features_no_ingre_indexes(self):
        return interval(158, len(self.data[0]) - 2)

    def get_features_no_ingre_names(self, names):
        return names[self.features_no_ingre_indexes, :]

    @property
    def features_no_ingre(self):
        """
        :return: Features without ingredients.
        """
        return self.data[:, self.features_no_ingre_indexes]

    @property
    def features_no_ingre_prob_combined_indexes(self):
        return self.features_no_ingre_prob_indexes

    def get_features_no_ingre_prob_combined_names(self, names):
        r = self.get_features_no_ingre_prob_names(names)
        index_list = self.COMBINED_INDEX_LIST
        extra_len = len(index_list) + 1
        return np.append(r, ["extra-{}".format(i) for i in xrange(0, extra_len)])

    @property
    def features_no_ingre_prob_combined(self):
        """
        :return: Features without ingredients and without the optional problems.
        """
        index_list = self.COMBINED_INDEX_LIST
        sum_list = self.COMBINED_SUM_LIST
        len_index = len(index_list)
        new = np.empty((len(self.data), 1 + len_index))

        for rowidx in xrange(0, len(self.data)):
            rangeidx = 0
            i = 0
            for therange in index_list:
                non_na = filter(lambda x: x > 0, self.data[rowidx, therange])
                if non_na:
                    if sum_list[i]:
                        avg = sum(non_na)
                    else:
                        avg = sum(non_na) / float(len(non_na))
                else:
                    avg = 0 # TODO: interpolate this?
                new[rowidx, rangeidx] = avg
                rangeidx += 1
                i += 1
            what = filter(lambda x: x > 0, self.data[rowidx, [158,160,161,278]])
            if not what: what = [self.labels[rowidx]]
            new[rowidx, rangeidx] = sum(what) / len(what)

        no_ingre_prob = self.data[:, self.features_no_ingre_prob_combined_indexes]
        return np.concatenate((no_ingre_prob, new), axis=1)

    @property
    def features_no_ingre_prob_indexes(self):
        return interval(155, 229) + interval(254, len(self.data[0]) - 2)

    def get_features_no_ingre_prob_names(self, names):
        return names[self.features_no_ingre_prob_indexes, :]

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

    def idx_by_gradient_boost(self, topk=0, features=None):
        '''
        Select the topk most relevant features using Gradient Boosting Tree
        '''
        if features is None:
            # input features default to features without ingredient
            features=self.features_no_ingre

        clf = GradientBoostingRegressor()
        clf.fit(features, self.labels)
        topk_idx = clf.feature_importances_.argsort()[-topk:]
        topk_idx.sort()
        return topk_idx

    @property
    def features_combined(self):
        return self._get_features_combined(self.COMBINED_INDEX_LIST, self.COMBINED_SUM_LIST)

    def get_features_combined_names(self, names):
        index_list = self.COMBINED_INDEX_LIST
        names = ["combined-{}".format(i) for i in xrange(0, len(index_list))]
        leftover_idxes = self._get_combined_leftover_indexes(index_list)
        return np.append(names[leftover_idxes, :], names)

    @property
    def features_combined_prob(self):
        return self._get_features_combined(self.COMBINED_PROB_INDEX_LIST, self.COMBINED_PROB_SUM_LIST)

    def get_features_combined_prob_names(self, names):
        index_list = self.COMBINED_PROB_INDEX_LIST
        names = ["combined-{}".format(i) for i in xrange(0, len(index_list))]
        leftover_idxes = self._get_combined_leftover_indexes(index_list)
        return np.append(names[leftover_idxes, :], names)

    def _get_combined_leftover_indexes(self, index_list):
        return list(set(interval(155, len(self.data[0])-2)) - set(itertools.chain(*index_list)))

    def _get_features_combined(self, index_list, sum_list):
        leftover_idxes = self._get_combined_leftover_indexes(index_list)
        leftover_count = len(leftover_idxes)
        new = np.empty((len(self.data), len(index_list) + len(leftover_idxes)))
        new[:, 0:leftover_count] = self.data[:, leftover_idxes]

        for rowidx in xrange(0, len(self.data)):
            rangeidx = leftover_count
            i = 0
            for therange in index_list:
                non_na = filter(lambda x: x > 0, self.data[rowidx, therange])
                if non_na:
                    if sum_list[i]:
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
