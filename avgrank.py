#!/usr/bin/env python

import argparse
import glob

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
    sums = {}
    for filename in glob.glob(args.src):
        p = read(filename)
        for (k, v) in p.iteritems():
            if k not in master:
                master[k] = v
                sums[k] = 1
            else:
                sums[k] += 1
                master[k] += v
        with open(args.dest, 'w') as f:
            f.write('ID,ProductD,Rank\n')
            for (k, v) in master.iteritems():
                avg = v / sums[k]
                f.write('{},{},{}\n'.format(k[0], k[1], avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='learning module.')
    parser.add_argument("--dest", help="output filename", required=True)
    parser.add_argument("--src", help="input glob filename", required=True)
    args = parser.parse_args()
    main(args)


