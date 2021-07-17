'''
Code for paper "Confidence Scores Make Instance-dependent Label-noise Learning Possible"
Antonin Berthon, 2021
-----------
Script description:
Utils function for model calibration.
'''

import numpy as np

from scipy.optimize import fmin


def find_T(model, x_test, y_test, nb_bins = 10, xtol=1e-1,ftol=1e-2):
    def f(T):
        return ECE(model.forward_TS_custom(x_test, T[0]), y_test, nb_bins=10, verb = False).item()
    print("Searching for optimal T...")
    res = fmin(f,5,xtol=xtol,ftol=ftol, full_output = 1, maxiter=10)
    print("Optimal T found : T *= %.2f, f(T*) = %.3f" % (res[0][0], res[1]))
    return res[0][0]


def ECE(output, y_true, nb_bins, verb = False):
    n = output.shape[0]
    probas = output.max(dim = 1)[0]
    bools = (y_true == output.argmax(dim = 1))

    bins = get_bins(output, nb_bins)
    ece = 0

    for i, b_index in enumerate(bins):
        if len(b_index) == 0:
            continue
        conf = probas[b_index].mean()
        acc = bools[b_index].float().mean()
        ece += len(b_index) * abs(acc - conf)
        if verb:
            print("Bin %i [n: %d]: acc %.2f ; conf %.2f. Dif %.3f" % (
            i, len(b_index), acc, conf, len(b_index) * abs(acc - conf)))
    return ece / n


def get_reliability_diag(output, y_true, nb_bins, verb = False, min_points = 5):
    n = output.shape[0]
    probas = output.max(dim = 1)[0]
    bools = (y_true == output.argmax(dim = 1))

    intervals = np.linspace(0, 1, nb_bins + 1)[1:]
    bins = get_bins(output, nb_bins)
    ece = 0

    tab = []
    for i, b_index in enumerate(bins):
        if len(b_index) < min_points:
            tab.append(np.array([0]))
            continue
        acc = bools[b_index].float().mean()
        tab.append(acc.data.item())
    return intervals, np.array(tab)

def get_bins(output, nb_bins = 10):
    intervals = np.linspace(0, 1, nb_bins+1)[1:]
    probas = output.max(dim=1)[0]
    bins = [[] for i in range(nb_bins)]
    assigned = 0
    for i, o in enumerate(probas):
        for ind,inter in enumerate(intervals):
            if o <= inter:
                bins[ind].append(i)
                assigned += 1
                break
    return bins