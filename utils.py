from __future__ import unicode_literals

import os


def write_pred(filename, ids, preds):
    os.mkdir('results')
    with open("results/svm_preds.csv", "w") as f:
        f.write("ID,Overall.Opinion\n")
        for (id, p) in zip(ids, preds):
            f.write("{},{}\n".format(int(id), int(p)))
