#!/usr/bin/env python

from os import listdir
from os.path import isfile, join

from utils import write_rank_path
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
    for filename in listdir('rank1'):
        filepath = join('rank1', filename)
        if not isfile(filepath):
            continue
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
        for (k, v) in master.iteritems():
            master[k] = v / sums[k]
            preds.append(master[k])
            ids.append(k)
        write_rank_path(filename, test_data, ids, preds, dir='rank-combined')



if __name__ == '__main__':
    main()
