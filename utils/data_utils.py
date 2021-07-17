'''
Code for paper "Confidence Scores Make Instance-dependent Label-noise Learning Possible"
Antonin Berthon, 2021
-----------
Script description:
Utils for data generating and loading data.
'''

import numpy as np
from math import *
import pandas as pd
import torch


class Data(object):
    def __init__(self, x_train, y_train, y_noisy, r_train, x_test, y_test, y_noisy_test, r_test):
        self.x_train = x_train
        self.y_train = y_train
        self.y_noisy = y_noisy
        self.r_train = r_train
        self.x_test = x_test
        self.y_test = y_test
        self.y_noisy_test = y_noisy_test
        self.r_test = r_test


def gaussian_dist(n):
    centers = np.array([[-2, 2],
                        [2, 2],
                        [2, -2],
                        [-2, -2]
                        ])

    std = sqrt(1.2)
    x_train, y_train = [], []

    for _ in range(n):
        k = np.random.choice(4)
        x = np.random.normal(loc = centers[k], scale = std, size = (2))
        x_train.append(x)
        y_train.append(k)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train = pd.get_dummies(y_train).values
    return x_train, y_train

def gaussian_data(n, val_split):
    x_train, y_train = gaussian_dist(int(n*val_split))
    x_test, y_test = gaussian_dist(int(n * (1-val_split)))
    output = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}
    return output


def circle_dist(n, n_class = 4):
    rayons = np.array([i+1 for i in range(n_class)])
    width = 0.4

    center = np.array([0, 0])

    x_train, y_train = [], []

    for _ in range(n):
        k = np.random.choice(n_class)
        x = 2 * np.random.rand(2) - 1  # -1 < x1, x2 < 1
        jiggle = np.random.normal() * width  # (2*np.random.rand()-1)*width
        x = (jiggle + rayons[k]) * x / np.linalg.norm(x)
        x_train.append(x)
        y_train.append(k)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train = pd.get_dummies(y_train).values
    return x_train, y_train


def circle_data(n, val_split, n_class):
    x_train, y_train = circle_dist(int(n*(1-val_split)), n_class)
    x_test, y_test = circle_dist(int(n * val_split), n_class)
    output = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}
    return output

def get_loader(x_train, y_train, y_noisy, r_train, x_test, y_test, bs = 32):
    X = torch.from_numpy(x_train).float()
    Y = torch.from_numpy(y_train).long()
    Y_noisy = torch.from_numpy(y_noisy).long()
    R = torch.from_numpy(r_train).float()

    X_test = torch.from_numpy(x_test).float()
    Y_test = torch.from_numpy(y_test).long()

    train_dataset = torch.utils.data.TensorDataset(X, Y, Y_noisy, R)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, drop_last=True)

    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, drop_last=True)

    return train_loader, test_loader