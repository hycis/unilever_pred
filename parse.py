#!/usr/bin/env python

from __future__ import unicode_literals

import csv as csv_mod
import re

import numpy as np

from utils import csv_reader_utf8, is_float

AGREE_RE = re.compile(r'^.*?\((?P<number>\d+)\)$')
FIRST_NUMBER_RE = re.compile(r'(?P<number>\d+)')
RANGE_RE = re.compile(r"(?P<a>\d+)\s*(?:to|-)\s*(?P<b>\d+)", re.I)

invalid_agree = set()


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
    if cell == 'na':
        v = 0
    elif cell == 'everyday':
        v = 20 # arbitrary
    else:
        m = RANGE_RE.search(cell)
        if m:
            a = float(m.group('a'))
            b = float(m.group('b'))
            v = (a + b) / 2.0
        else:
            # get the first number found
            m = FIRST_NUMBER_RE.search(cell)
            v = float(m.group('number'))
    return v


def parse_income(s):
    cell = s.replace(',', '')
    return parse_range(cell.lower().replace('rs.',''))


def parse_range(cell):
    m = RANGE_RE.search(cell)
    if m:
        a = float(m.group('a'))
        b = float(m.group('b'))
        v = (a + b) / 2.0
    else:
        # get the first number found
        m = FIRST_NUMBER_RE.search(cell)
        if m:
            v = float(m.group('number'))
        else:
            v = 0
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
    cell = s.lower().strip()

    if cell.startswith('hour to less'):
        return .75
    if cell.startswith('15 mins to less'):
        return (.25 + 1) / 2.
    if 'less than 2' in cell:
        return 1.5
    if 'less than 15 mins' in cell:
        return .2
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
    if 'quickly' in cell:
        return 1
    if 'completely' in cell:
        return 2
    if 'both' in cell:
        return 3
    return parse_agree(s)


def read_csv(file_path):
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
        (xrange(230, 254), parse_problem),
        (xrange(254, 259), parse_na_bool_or_number),
        ([259], parse_soak),
        (xrange(260, 264), parse_bool),

        # Rinse method, dry method
        # (xrange(264, 266), None),

        # income
        ([266], parse_income),

        # education, marrital status, employment, ... set to 0
        # (xrange(267, 272), None),

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

    csv = csv_reader_utf8(file_path, dialect=csv_mod.excel)
    # skip first row which is header
    for row in csv:
        break

    parsed_rows = []
    rownum = 1

    try:
        for row in csv:
            # allocate array
            parsed_row = [0] * len(row)
            for format in formats:
                for i in format[0]:
                    parsed_row[i] = format[1](row[i])
            parsed_rows.append(parsed_row)
            rownum += 1
    except Exception as e:
        print("row#={} i={} cell={}".format(rownum, i, row[i]))
        raise

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
