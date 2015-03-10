from __future__ import unicode_literals

import numpy as np
import csv as csv_mod
import codecs
import os
import errno
import re

from sklearn import svm, preprocessing


COLLAPSE_WHITESPACE_RE = re.compile(r'\s+')
AGREE_RE = re.compile(r"\s*\((?P<number>\d+)\)$")
AGREE_STD_RE = re.compile(r"(?<!\w)(little|much|a|it)(?!\w)", re.I)

_RANK_DATA = [
1,"DP00103",
2,"DP00108",
3,"DP00113",
4,"DP00120",
5,"DP00127",
6,"DP0013",
7,"DP00131",
8,"DP00133",
9,"DP00137",
10,"DP00152",
11,"DP00153",
12,"DP0018",
13,"DP0019",
14,"DP002",
15,"DP0031",
16,"DP0032",
17,"DP0033",
18,"DP0040",
19,"DP0047",
20,"DP0052",
21,"DP0057",
22,"DP0059",
23,"DP0061",
24,"DP0073",
25,"DP0076",
26,"DP0079",
27,"DP0088",
28,"DP0095",
]
PROD_ID_TO_ID = {
    int(_RANK_DATA[i + 1][2:]): (_RANK_DATA[i], _RANK_DATA[i + 1]) for i in xrange(0, len(_RANK_DATA), 2)
}


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def drange(start, stop, step):
    r = start
    while r <= stop:
        yield r
        r += step


def xinterval(start, end):
    return xrange(start, end + 1)


def interval(start, end):
    """
    :param start:
    :param end:
    :return: a list of integers between start and end inclusively.
    """
    return range(start, end + 1)


def norm(features):
    return preprocessing.scale(features)


def mse(arr1, arr2):
    sum = 0
    for a, b in zip(arr1, arr2):
        sum += (a - b) ** 2
    return sum / len(arr1)


def collapse_whitespace(s):
    return COLLAPSE_WHITESPACE_RE.sub(' ', s)


def standardize(s):
    cell = s.lower()
    m = AGREE_RE.search(cell)
    if m:
        cell = AGREE_RE.sub(r" (\g<number>)", cell)
        cell = AGREE_STD_RE.sub("", cell)
    return collapse_whitespace(cell.strip("* "))


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def clamp(v, minv, maxv):
    return max(minv, min(maxv, v))


def write_raw_rank(filename, test_data, ids, preds, dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    path = dir + "/" + filename
    with open(path, "w") as f:
        f.write("ID,ProductD,Rank\n")
        for pid in test_data.unique_prod_ids:
            pidxes = filter_index(test_data.prod_ids, lambda x: x == pid)
            test_id = test_data.ids[pidxes[0]]
            test_idxes = filter_index(ids, lambda x: x == test_id)

            rank = clamp(preds[test_idxes[0]], 1, len(test_data.unique_prod_ids))
            id, pid_str = PROD_ID_TO_ID[pid]
            f.write("{},{},{}\n".format(id, pid_str, rank))


def write_pred_path(path, ids, preds, clamp=True):
    themin = 1
    themax = 7
    if not clamp:
        themin = 0
        themax = 1e10
    with open(path, "w") as f:
        f.write("ID,Overall.Opinion\n")
        for (id, p) in zip(ids, preds):
            f.write("{},{:.6f}\n".format(id, clamp(p, themin, themax)))


def write_pred(filename, ids, preds, dir="results", clamp=True):
    if not os.path.exists(dir):
        os.mkdir(dir)
    write_pred_path(dir + "/" + filename, ids, preds, clamp=clamp)


def write_rank_path(filename, test_data, ids, preds, dir):

    if not os.path.exists(dir):
        os.mkdir(dir)
    path = dir + "/" + filename
    preds = np.array(list(preds))
    ranks = []

    for pid in set(test_data.prod_ids):
        pidxes = filter_index(test_data.prod_ids, lambda x: x == pid)
        test_ids = np.array(test_data.ids)
        tids = set(test_ids[pidxes])
        pidxes = filter_index(ids, lambda x: x in tids)
        prod_preds = preds[pidxes]
        avg = np.average(prod_preds)
        ranks.append({'avg': avg, 'pid': pid})

    ranks = sorted(ranks, key=lambda x: -x['avg'])

    with open(path, "w") as f:
        f.write("ID,ProductD,Rank\n")
        for rank in xrange(0, len(ranks)):
            pid = ranks[rank]['pid']
            id, pid_str = PROD_ID_TO_ID[pid]
            f.write("{},{},{}\n".format(id, pid_str, rank + 1))


def filter_index(arr, filter_fn):
    """
    :param arr:
    :param filter_fn:
    :return: Returns array of indexes of elements in arr that satisfies the filter.
    """
    indexes = []
    i = 0
    for elem in arr:
        if filter_fn(elem):
            indexes.append(i)
        i += 1
    return indexes


def find_max(arr):
    """
    :param arr: list of numbers.
    :return: zero-based index of the max element.
    """
    themax = arr[0]
    theindex = 0
    for i in xrange(1, len(arr)):
        if arr[i] > themax:
            themax = arr[i]
            theindex = i
    return theindex


def _utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8')


def csv_reader_utf8(file_path, **kwargs):
    """
    Adapted from http://docs.python.org/2/library/csv.html
    """

    f = codecs.open(file_path, encoding='utf8', mode='rb', errors='ignore')

    # csv.py doesn't do Unicode; encode temporarily as UTF-8:
    csv_reader = csv_mod.reader(_utf_8_encoder(f), **kwargs)

    for row in csv_reader:
        # decode UTF-8 back to Unicode, cell by cell:
        yield [unicode(cell, 'utf-8') for cell in row]

