'''
Code for paper "Confidence Scores Make Instance-dependent Label-noise Learning Possible"
Antonin Berthon, 2021
-----------
Script description:
Miscellaneous utility functions.
'''

import matplotlib.pyplot as plt
import torch
import numpy as np


def to_int(y):
    return np.argmax(y, axis=1)

def get_dummies(y):
    yp = np.zeros((len(y), 10))
    for i,a in enumerate(y):
        yp[i, a] = 1
    assert (yp.argmax(axis=1) == y).all()
    return yp

# Prediction
def plot_contour(model, title = "", h=0.06):
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    points = np.c_[xx.ravel(), yy.ravel()]
    points = torch.tensor(points)
    Z = model(points.float())
    Z = np.argmax(Z.detach().numpy(), axis=1)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.3)
    plt.title(title, fontsize=28)
    #sns.scatterplot(data=dt_noisy_train, x="x1", y="x2", hue = "y", legend=False)

def pertubate_noise(noises, pert_size, form = "normal", plot=False):
    if form == "normal":
        ## Normal dist
        pert = np.random.normal(scale=pert_size, size=noises.shape)
    elif form == "skewed":
        ## Skewed normal dist using beta distribu shifted and rescaled
        pert = np.random.beta(2, 5, size=noises.shape)
        # Normalize and scale
        pert = (pert-(pert.min()+pert.max())/2)/pert.std()
        # Get wanted std
        pert *= pert_size
    elif form == "over_conf":
        ## Add pert_size% to each confidence score
        pert = noises * pert_size
    if plot:
        plt.hist(pert, bins=50)
        plt.show()
    return np.clip(noises + pert, a_min=0, a_max=1)

def get_corrupted_means(y, y_p, n_class=4):
    output = []
    for i in range(n_class):
        ind = to_int(y) == i
        output.append(1-(y_p[ind] == y[ind]).all(axis=1).mean())
    return np.array(output)

def get_noise_quantiles(r,y,args, dummy=True):
    if dummy:
        return np.array([np.quantile(r[y.argmax(axis=1) == i], np.arange(0, 1, 0.1)) for i in range (args["n_class"])])
    else:
        return np.array([np.quantile(r[y == i], np.arange(0, 1, 0.1)) for i in range(args["n_class"])])


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
            tab.append([0])
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