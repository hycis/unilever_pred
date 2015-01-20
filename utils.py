from __future__ import unicode_literals

import csv as csv_mod
import codecs
import os
import errno
import re

from sklearn import svm, preprocessing


COLLAPSE_WHITESPACE_RE = re.compile(r'\s+')
AGREE_RE = re.compile(r"\s*\((?P<number>\d+)\)$")
AGREE_STD_RE = re.compile(r"(?<!\w)(little|much|a|it)(?!\w)", re.I)


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


def write_pred_path(path, ids, preds):
    with open(path, "w") as f:
        f.write("ID,Overall.Opinion\n")
        for (id, p) in zip(ids, preds):
            f.write("{},{:.6f}\n".format(id, clamp(p, 1, 7)))


def write_pred(filename, ids, preds, dir="results"):
    if not os.path.exists(dir):
        os.mkdir(dir)
    write_pred(dir + "/" + filename, ids, preds)


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
