from __future__ import unicode_literals

import csv as csv_mod
import codecs
import os


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def write_pred(filename, ids, preds):
    os.mkdir('results')
    with open("results/svm_preds.csv", "w") as f:
        f.write("ID,Overall.Opinion\n")
        for (id, p) in zip(ids, preds):
            f.write("{},{}\n".format(int(id), int(p)))


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
