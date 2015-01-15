from __future__ import unicode_literals

import numpy as np

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
    def features_oo_only(self):
        """
        :return: overall opinion only
        """
        indexes = interval(155, 161) + [216, 277, 278, 287, 288, 293]
        return self.data[:, indexes]

