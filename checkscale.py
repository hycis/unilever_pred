#!/usr/bin/env python


import numpy as np

from data import load_names
from utils import filter_index


names = load_names()
a = np.load('train.npy')
m = []

for pid in set(a[:, 1]):
    idxes = filter_index(a[:, 1], lambda x: x==pid)
    pa = a[idxes, :]
    m.append (np.max(pa, axis=0))

m = np.array(m)

# output max of each column grouped by prod -- each row is a product
with open('checkscale-max.csv','w') as f:
    f.write(",".join(names)+"\n")
    for mm in m:
        for mmm in mm:
            f.write(str(mmm)+",")
        f.write("\n")

# print names of fields with diff scales
for i in xrange(0, len(m[0])):
    row2 = filter(lambda x:x>0, m[:,i]) or [0,0]
    std = np.std(row2)
    if std != 0:
        print("{:03d}: {}".format(i, names[i]))
