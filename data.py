from __future__ import unicode_literals

import numpy as np

from utils import interval, filter_index


TRAIN_FILENAME = 'train.npy'
TEST_FILENAME = 'test.npy'


class DataSet(object):
    def __init__(self, filename):
        self.data = data = np.load(filename)
        self.ids = map(lambda x: int(round(x)), data[:, 0]) # first column
        self.prod_ids = map(lambda x: int(round(x)), data[:, 1])
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
    def features_no_ingre_prob(self):
        """
        :return: Features without ingredients and without the optional problems.
        """
        indexes = interval(155, 230) + interval(254, len(self.data[0]) - 2)
        return self.data[:, indexes]

