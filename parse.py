#!/usr/bin/env python

<<<<<<< HEAD
=======
# TODO: interpolate the agree columns only
# TODO: average the columns, group by category


>>>>>>> 4187624595137e110474f0ad60305b4998edbebe
from __future__ import unicode_literals

import csv as csv_mod
import itertools
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
    if cell in ('no', 'none'):
        return 0
    if cell == 'na':
        return -1
    raise Exception("invalid bool: " + cell)


def parse_number_or_zero(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0


def parse_na_or_number(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return -1


def parse_na_bool_or_number(s):
    s = s.lower()
    if s in ('no', 'none'):
        return 0
    if s in ('yes', ):
        return 1
    if s == 'na':
        return -1
    return float(s)


def parse_identity(s):
    return s


def parse_times_used(s):
    cell = s.lower()
    if cell == 'na':
        v = -1
    elif cell == 'everyday':
        v = 30 # arbitrary
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
        v = -1
    else:
        raise Exception('invalid how much: ' + cell)
    return v


def parse_agree(s):
    cell = s.lower()
    if cell == 'na':
        return -1
    m = AGREE_RE.match(cell)
    if not m:
        invalid_agree.add(cell)
        return -1
    return int(m.group('number'))


def parse_soak(s):
    cell = s.lower().strip().replace('  ', ' ')

    if cell == '+ hour to less than 1 hour' or cell == 'hour to less than 1 hour':
        return .75
    if cell == '1 + hour to less than 2 hours' or cell == '1 hour to less than 2 hours':
        return 1.75
    if cell == '1 hour to less than 1 hours' or cell == '1 hour to less than 1+ hours':
        return 1.25
    if cell.startswith('15 mins to less'):
        return (.25 + .5) / 2.
    if 'less than 15 mins' in cell:
        return .25 / 2.
    if cell == 'overnight':
        return 8 # arbitrary
    if 'more than 2' in cell or '2 hrs' in cell:
        return 3 # arbitrary
    if 'not' in cell:
        return 0
    if cell == 'na':
        return -1
    raise Exception("invalid soak: " + cell)


def parse_rinse(s):
    cell = s.lower().lstrip('*')
    if cell.startswith('pre treat'):
        return 1
    if cell.startswith('pre rinse'):
        return 2
    if cell.startswith('soak'):
        return 3
    if cell.startswith('wash'):
        return 4
    if cell == 'na':
        return -1
    raise Exception("invalid rinse: " + cell)


def parse_dry(s):
    cell = s.lower()
    if 'dryer' in cell or 'with heat' in cell:
        return 1
    if 'inside' in cell or 'without heat' in cell or 'tumble' in cell:
        return 2
    if 'sun and shade' in cell:
        return 3.5
    if ' shade' in cell:
        return 3
    if ' sun' in cell:
        return 4
    if cell == 'na':
        return -1
    raise Exception("invalid dry: " + cell)


def parse_problem(s):
    cell = s.lower()
    if cell.startswith('no ') or cell == 'no':
        return 0
    if cell == 'na':
        return -1
    return 1


def parse_dissolve(s):
    if '(' in s:
        return parse_agree(s)
    cell = s.lower()
    if 'quickly' in cell:
<<<<<<< HEAD
        return 1
    if 'completely' in cell:
        return 2
    if 'both' in cell:
        return 3
    return parse_agree(s)
=======
        return 4
    if 'completely' in cell or 'both' in cell:
        return 5
    if cell == 'na':
        return -1
    raise Exception("invalid diss: " + cell)


def parse_marriage(s):
    cell = s.lower().strip()
    if cell.startswith('separated'):
        return 0
    if cell.startswith('cohabiting'):
        return .5
    if cell.startswith('single - living'):
        return 0
    if cell.startswith('married'):
        return 1
    if cell == 'na':
        return -1
    raise Exception("invalid marriage: " + cell)
>>>>>>> 4187624595137e110474f0ad60305b4998edbebe


def parse_exp(s):
    cell = s.lower().replace('  ', ' ').strip()
    if cell == 'a little better than i expected (2)':
        return 4
    if cell == 'much better than i expected (1)':
        return 5
    return parse_agree(cell)


def parse_flow(s):
    cell = s.lower().replace('  ', ' ')
    if 'ease' in cell:
        if 'very poor' in cell:
            return 1
        if 'somewhat poor' in cell:
            return 1 + 1/4. * 6.
        if 'right' in cell:
            return 1 + 2/4. * 6.
        if 'somewhat good' in cell:
            return 1 + 3/4. * 6.
        if 'very good' in cell:
            return 1 + 4/4. * 6
        raise Exception("invalid flow: " + cell)
    return parse_agree(cell)


def parse_desire(s):
    cell = s.lower()
    v = parse_agree(cell)
    if 'desire' in cell or 'too' in cell or 'right' in cell:
        return 1 + (v - 1) / 2. * 6.
    return v


def parse_speckles(s):
    cell = s.lower()
    v = parse_agree(cell)
    if 'liked' in cell:
        return 1 + (v - 1) / 4. * 6.
    if 'desire' in cell or 'right' in cell or 'speckle' in cell:
        return 1 + (v - 1) / 2. * 6.
    return v


def parse_fond_sparkles(s):
    cell = s.lower()
    v = parse_agree(cell)
    if 'liked' in cell:
        return 1 + (v - 1) / 4. * 6.
    if v > 0:
        return 1 + (v - 1) / 2. * 6.
    return v


SCHEMA = (
    # response ID
    ([0], int),
    # product ID
    ([1], lambda s: int(s[2:])),
    # parse ingredients
    (xrange(2, 155), parse_number_or_zero),
    ([155, 229], parse_bool),
    ([156], parse_how_much),
    ([157], parse_times_used),
    (xrange(158, 227), parse_agree),
    ([221, 225], parse_desire),
    ([227], parse_speckles),
    ([228], parse_fond_sparkles),
    (xrange(230, 254), parse_problem),
    (xrange(254, 259), parse_na_bool_or_number),
    ([259], parse_soak),
    (xrange(260, 264), parse_bool),

    # Rinse method, dry method
    ([264], parse_rinse),
    ([265], parse_dry),

    # income
    ([266], parse_income),

    # education, marrital status, employment, ... set to 0
    # (xrange(267, 272), None),

    # Employment hours
    ([268], parse_na_or_number),

    # occupants
    (xrange(272, 277), parse_na_or_number),
    ([277], parse_exp),
    (xrange(278, 288), parse_agree),
    ([288], parse_dissolve),
    (xrange(289, 294), parse_agree),
    ([294], lambda s: {'none': 0, 'no': 0, 'na': -1}.get(s.lower(), 1)),
    ([295], parse_flow),
    (xrange(296, 297), parse_agree),
    ([297], parse_bool),
    ([298], parse_soak),
    (xrange(299, 301), parse_agree),

    # overall opinion
    ([301], parse_number_or_zero),
)


AGREE_INDEXES = list(itertools.chain(*map(lambda x: x[0] if x[1] == parse_agree else [], SCHEMA))) + [277, 295]


def read_csv(file_path):
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
            for format in SCHEMA:
                for i in format[0]:
                    parsed_row[i] = format[1](row[i])
            parsed_rows.append(parsed_row)
            rownum += 1
    except Exception:
        print("row#={} i={} cell={}".format(rownum, i, row[i]))
        raise

    for a in invalid_agree:
        print(a)

    return parsed_rows


if __name__ == '__main__':
<<<<<<< HEAD
    train = np.array(read_csv('/Volumes/Storage/Unilever_Challenge/dataset/20141211154303-Data_in_csv/Training_Data.csv'))
    # Last column is the opinion score 0-7 (label)
    np.save('train.npy', train)

    sub = np.array(read_csv('/Volumes/Storage/Unilever_Challenge/dataset/20141211154303-Data_in_csv/Submission_Data.csv'))
    # Last column is the test label which is all 0
    np.save('test.npy', sub)
=======
    import data
    from data import DataSet
    train_raw = np.array(read_csv('train.csv'))
    # this row has a score of NA (invalid), change to 5.
    train_raw[6269, -1] = 5
    np.save(data.TRAIN_RAW_FILENAME, train_raw)

    train = DataSet(data=train_raw)
    train.transform_na_zero()
    np.save(data.TRAIN_FILENAME, train.data)


    test_raw = np.array(read_csv('sub.csv'))
    np.save(data.TEST_RAW_FILENAME, test_raw)

    test = DataSet(data=test_raw)
    test.transform_na_zero()
    np.save(data.TEST_FILENAME, test.data)


>>>>>>> 4187624595137e110474f0ad60305b4998edbebe
