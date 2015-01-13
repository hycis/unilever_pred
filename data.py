from __future__ import unicode_literals

import numpy as np


TRAIN_FILENAME = 'train.npy'
TEST_FILENAME = 'test.npy'


class DataSet(object):
    def __init__(self, filename):
        self.data = data = np.load(filename)
        self.ids = map(lambda x: int(round(x)), data[:, 0])
        self.prod_ids = map(lambda x: int(round(x)), data[:, 1])
        self.unique_prod_ids = set(self.prod_ids)
        self.column_count = len(data[0])
        self.labels = data[:, self.column_count - 1]
        self.features = data[:, 2:self.column_count - 1]

    @property
    def features_no_ingre(self):
        """
        :return: Features without ingredients.
        """
        return self.features[:, 153:]

