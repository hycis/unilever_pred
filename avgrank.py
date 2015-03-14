#!/usr/bin/env python

import argparse
import glob

import numpy as np


def read(filepath):
    m = {}
    with open(filepath, 'r') as f:
        for line in f:
            break
        for line in f:
            id, pid, rank = line.strip().split(',')
            m[(int(id), pid.strip().upper())] = float(rank)
    return m


def main(args):
    master = {}
    ind = {}
    sums = {}
    for filename in glob.glob(args.src):
        p = read(filename)
        for (k, v) in p.iteritems():
            if k not in master:
                ind[k] = [v]
                master[k] = v
                sums[k] = 1
            else:
                sums[k] += 1
                master[k] += v
                ind[k].append(v)

    srted = sorted(list(master.iteritems()), key=lambda x: x[1] / sums[x[0]])

    with open(args.dest, 'w') as f:
        f.write('ID,ProductD,Rank\n')
        rank = 1
        for a in srted:
            k = a[0]
            v = a[1]
            avg = v / sums[k]
            if 3 <= rank < 14.5:
                avg = (3+14.5)/2
            elif 14.5 < rank <= 26:
                avg = (14.5+26)/2
            rank += 1
            print("{}, std={}".format(k, np.std(ind[k])))
            f.write('{},{},{}\n'.format(k[0], k[1], avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='learning module.')
    parser.add_argument("--dest", help="output filename", required=True)
    parser.add_argument("--src", help="input glob filename", required=True)
    args = parser.parse_args()
    main(args)


