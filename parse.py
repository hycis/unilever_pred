#!/usr/bin/env python

from __future__ import unicode_literals

import csv as csv_mod
import codecs
import re

import numpy as np

AGREE_RE = re.compile(r'^.*?\((?P<number>\d+)\)$')
TIMES_USED_RE = re.compile(r'(?P<number>\d+)')

invalid_agree = set()


def _utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8')


def csv_reader_utf8(file_path, **kwargs):
    """
    Adapted from http://docs.python.org/2/library/csv.html
    """

    f = codecs.open(file_path, encoding='ascii', mode='rb', errors='replace')

    # csv.py doesn't do Unicode; encode temporarily as UTF-8:
    csv_reader = csv_mod.reader(_utf_8_encoder(f), **kwargs)

    for row in csv_reader:
        # decode UTF-8 back to Unicode, cell by cell:
        yield [unicode(cell, 'utf-8') for cell in row]


def parse_bool(s):
    cell = s.lower()
    if cell == 'yes':
        return 1
    if cell in ('na', 'no', 'none'):
        return 0
    raise Exception("invalid bool: " + cell)


def parse_na_or_number(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0


def parse_na_bool_or_number(s):
    s = s.lower()
    if s in ('na', 'no', 'none'):
        return 0
    if s in ('yes', ):
        return 1
    return float(s)


def parse_identity(s):
    return s


def parse_times_used(s):
    cell = s.lower()
    if 'oct' in cell: # a date?
        v = 1
    elif cell == 'na':
        v = 0
    elif cell == 'everyday':
        v = 20 # arbitrary
    else:
        m = AGREE_RE.search(cell)
        if m:
            v = float(m.group('number'))
        else:
            # get the first number found
            m = TIMES_USED_RE.search(cell)
            v = int(m.group('number'))
    return v


def parse_how_much(s):
    cell = s.lower()
    if 'a fourth' in cell or 'a quarter' in cell:
        v = .25
    elif 'half' in cell:
        v = .5
    elif 'three' in cell:
        v = .75
    elif 'whole' in cell or 'fully' in cell:
        v = 1
    elif cell == 'na':
        v = 0
    else:
        raise Exception('invalid how much: ' + cell)
    return v


def parse_agree(s):
    cell = s.lower()
    m = AGREE_RE.match(cell)
    if not m:
        invalid_agree.add(cell)
        return 0
    return int(m.group('number'))


def parse_soak(s):
    cell = s.lower()

    if 'less than 2' in cell:
        return 1.5
    if 'less than 15 mins' in cell:
        return .25
    if 'less than 1' in cell or '15 min' in cell:
        return .5
    if cell == 'overnight':
        return 8 # arbitrary
    if 'more than 2' in cell or '2 hrs' in cell:
        return 3 # arbitrary
    return 0


def parse_problem(s):
    cell = s.lower()
    if cell.startswith('no ') or cell == 'na' or cell == 'no':
        return 0
    return 1


def parse_dissolve(s):
    cell = s.lower()
    if 'quickly' in cell or 'completely' in cell or 'both' in cell:
        return 5
    return parse_agree(s)


def read_csv(file_path):
    csv = csv_reader_utf8(file_path, dialect=csv_mod.excel)
    # skip first row which is header
    for row in csv:
        break

    formats = (
        # response ID
        ([0], int),
        # product ID
        ([1], lambda s: int(s[2:])),
        # parse ingredients
        (xrange(2, 155), parse_na_or_number),
        ([155, 229], parse_bool),
        ([156], parse_how_much),
        ([157], parse_times_used),
        (xrange(158, 229), parse_agree),
        (xrange(229, 254), parse_problem),
        (xrange(254, 259), parse_na_bool_or_number),
        ([259], parse_soak),
        (xrange(260, 264), parse_bool),
        # Income, marrital status, employment, ... set to 0
        (xrange(264, 272), None),

        # Employment hours
        ([268], parse_na_or_number),

        # occupants
        (xrange(272, 277), parse_na_or_number),

        (xrange(277, 288), parse_agree),
        ([288], parse_dissolve),
        (xrange(289, 294), parse_agree),
        ([294], lambda s: 0 if s.lower() in ('na', 'no', 'none') else 1),
        (xrange(295, 297), parse_agree),
        ([297], parse_bool),
        ([298], parse_soak),
        (xrange(299, 301), parse_agree),

        # overall opinion
        ([301], parse_na_or_number),
    )

    sets = {}
    parsed_rows = []
    rownum = 1

    for row in csv:
        # allocate array
        parsed_row = range(0, len(row))
        for i in xrange(0, len(row)):
            parsed_row[i] = 0
        for format in formats:
            # Find distinct values in column
            if format[1] is None:
                for i in format[0]:
                    if not sets.get(i):
                        sets[i] = set()
                    sets[i].add(row[i].strip().lower())
            else:
                for i in format[0]:
                    try:
                        parsed_row[i] = format[1](row[i])
                    except Exception as e:
                        print("row#={} i={} cell={}".format(rownum, i, row[i]))
                        raise
        parsed_rows.append(parsed_row)
        rownum += 1

    # Print distinct values for each col
    for (k, v) in sets.iteritems():
        print("key: " + str(k))
        for s in v:
            print(s)
        print("")

    for a in invalid_agree:
        print(a)

    return parsed_rows


if __name__ == '__main__':
    train = np.array(read_csv('train.csv'))
    # Last column is the opinion score 0-7 (label)
    np.save('train.npy', train)

    sub = np.array(read_csv('sub.csv'))
    # Last column is the test label which is all 0
    np.save('test.npy', sub)
