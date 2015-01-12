#!/usr/bin/env python

""" Compute distinct values for each column. """

from __future__ import unicode_literals

import csv as csv_mod

from utils import csv_reader_utf8, is_float


def compute_distinct(file_path, sets=None):
    csv = csv_reader_utf8(file_path, dialect=csv_mod.excel)

    # skip first row which is header
    for row in csv:
        columns = len(row)
        break

    if not sets:
        sets = []
        for i in xrange(0, columns):
            sets.append(set())

    for row in csv:
        i = -1
        for cell in row:
            i += 1
            if is_float(cell):
                sets[i].add('<num>')
                continue
            cell = cell.lower()
            sets[i].add(cell)

    return sets


if __name__ == '__main__':
    sets = compute_distinct('train.csv')
    sets = compute_distinct('sub.csv', sets)

    for i in xrange(0, len(sets)):
        print(str(i) + ":")
        for s in sets[i]:
            print(s)
        print("========================")
