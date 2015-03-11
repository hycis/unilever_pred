#!/usr/bin/env python

import glob
import os
from os import listdir
from os.path import isfile, join

from utils import write_rank_path, write_pred
from data import DataSet


def read(path):
    pairs = {}
    with open(path, 'r') as f:
        for line in f:
            break
        for line in f:
            id, s = line.strip().split(',')
            pairs[int(id)] = float(s)
    return pairs


def main():
    test_data = DataSet('test.npy')
    for filepath in glob.glob('rank1/*ingre,_*'):
        if '_rank_' in filepath or '_rr_' in filepath:
            continue
        filename = os.path.basename(filepath)
        master = read(filepath)
        sums = {k: v for (k, v) in master.iteritems()}
        for r in xrange(2, 8):
            rs = 'rank' + str(r)
            filepath2 = filepath.replace('rank1', rs)
            pairs = read(filepath2)
            for (k, v) in pairs.iteritems():
                master[k] += v * r
                sums[k] += v
        ids = []
        preds = []
        for k in test_data.ids:
            s = master[k] / sums[k]
            preds.append(s)
            ids.append(k)
        rankfilename = filename.replace('ingre,', 'ingre,_rank')
        write_pred(filename, ids, preds, dir='rank-combined')
        write_rank_path(rankfilename, test_data, ids, preds, dir='rank-combined')



if __name__ == '__main__':
    main()
