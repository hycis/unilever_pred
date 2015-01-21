#!/usr/bin/env python

from __future__ import unicode_literals

import numpy as np
import argparse
from os import listdir
from os.path import isfile, join

from data import DataSet, TRAIN_FILENAME
from utils import write_pred_path, mkdir_p


def main(args):
    train = DataSet(TRAIN_FILENAME)
    lbl_sizes = train.get_label_sizes()
    total_train = sum(lbl_sizes.values())
    sorted_lbl_sizes = sorted(lbl_sizes.iteritems(), key=lambda x: x[0])

    mkdir_p(args.dest)

    for in_filename in listdir(args.src):
        in_path = join(args.src, in_filename)
        if not isfile(in_path) or not in_path.endswith(".csv"):
            continue
        with open(in_path, "r") as in_file:
            # skip header
            for line in in_file:
                break
            id_pred_pairs = map(lambda x: x.strip().split(','), in_file)
            ids = map(lambda x: int(x[0]), id_pred_pairs)
            preds = np.array(map(lambda x: float(x[1]), id_pred_pairs))
            sorted_idxes = np.argsort(preds)
            prev_idx = 0
            accum = 0

            for (lbl, s) in sorted_lbl_sizes:
                accum += s
                next_idx = int(round(accum/float(total_train) * len(sorted_idxes)))
                preds[sorted_idxes[prev_idx:next_idx], :] = lbl
                prev_idx = next_idx

        out_path = join(args.dest, in_filename)
        write_pred_path(out_path, ids, preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='results transform.')
    parser.add_argument("-t", "--transform", help="type of transform", required=True,
                        choices=["dist"])
    parser.add_argument("-s", "--src", help="src path", required=True)
    parser.add_argument("-d", "--dest", help="dest path", required=True)
    args = parser.parse_args()
    main(args)
