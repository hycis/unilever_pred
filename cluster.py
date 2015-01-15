#!/usr/bin/env python

from __future__ import unicode_literals

import numpy as np

import csv as csv_mod

from data import DataSet, TEST_FILENAME, TRAIN_FILENAME
from utils import filter_index, csv_reader_utf8, interval


def distance(mask1, mask2):
    sum = 0
    for i in xrange(0, len(mask1)):
        if mask1[i] != mask2[i]:
            sum += 1
    return sum


def bits(mask):
    return "".join(map(str, mask))


def calc(dataset, prod_masks):
    feature_indexes = interval(158, 228) + interval(277, 287) + interval(289, 293) + interval(295, 296) + interval(299, 300)
    for prod_id in dataset.unique_prod_ids:
        prod_indexes = dataset.get_prod_indexes(prod_id)
        prod_rows = dataset.data[prod_indexes, :]
        prod_rows = prod_rows[:, feature_indexes]
        prod_mask = [0] * len(prod_rows[0])
        for row in prod_rows:
            i = 0
            for cell in row:
                if int(cell) != 0: # is it not NA?
                    prod_mask[i] += 1
                i += 1
        prod_sample_size = len(prod_indexes)
        for i in xrange(0, len(prod_mask)):
            if prod_mask[i] > prod_sample_size * .9:
                prod_mask[i] = 1
            else:
                prod_mask[i] = 0
        prod_masks.append({
            "id": prod_id,
            "mask": prod_mask,
        })


def compute_clusters(train, test):
    prod_id_to_mask = {}
    prod_id_to_cluster = {}
    prod_masks = []
    clusters = []
    taken = set()

    calc(train, prod_masks)
    calc(test, prod_masks)

    for i in xrange(0, len(prod_masks)):
        pm1 = prod_masks[i]
        m1 = pm1['mask']
        pid1 = pm1['id']
        prod_id_to_mask[pid1] = m1
        if i in taken:
            continue
        cluster = [pid1]
        taken.add(i)
        for j in xrange(i + 1, len(prod_masks)):
            if j in taken:
                continue

            pm2 = prod_masks[j]
            m2 = pm2['mask']
            pid2 = pm2['id']

            if distance(m1, m2) <= 3:
                taken.add(j)
                cluster.append(pid2)
        clusters.append(cluster)

    # merge small clusters into other clusters
    for i in xrange(0, len(clusters)):
        cluster = clusters[i]
        if len(cluster) >= 3:
            continue
        for pid in cluster:
            best_cluster_idx = 0
            best_distance = 1e99
            for j in xrange(0, len(clusters)):
                if i == j:
                    continue
                for pid2 in clusters[j]:
                    d = distance(prod_id_to_mask[pid], prod_id_to_mask[pid2])
                    if d < best_distance:
                        best_distance = d
                        best_cluster_idx = j
            clusters[best_cluster_idx].append(pid)
        clusters[i] = []

    # remove empty cluster
    clusters = [c for c in clusters if c]

    return clusters, prod_id_to_mask


def main():
    import pprint
    train = DataSet(TRAIN_FILENAME)
    clusters, prod_id_to_mask = compute_clusters(train, DataSet(TEST_FILENAME))
    i = 0
    for cluster in clusters:
        print("Cluster index={}".format(i))
        for pid in cluster:
            mark = '*' if pid in train.unique_prod_ids else ' '
            print("{:04d}{}: {}".format(pid, mark, bits(prod_id_to_mask[pid])))
        print("")
        i += 1
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(clusters)

    #for cluster in clusters:
    #    print("[" + ", ".join(map(str, cluster)) + "],")
    #print("")

    #for (pid, cluster_idx) in prod_id_to_cluster.iteritems():
    #    print("{: 4d}: {},".format(pid, cluster_idx))
    #print("")


if __name__ == '__main__':
    main()
