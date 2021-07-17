'''
Code for paper "Confidence Scores Make Instance-dependent Label-noise Learning Possible"
Antonin Berthon, 2021
-----------
Script description:
Utils for noise creation.
'''

import numpy as np
import torch


def S_to_T(S, mus = 0.35, n_class = 4):
    '''Converts switch matrix S to transition T based on the mean noise eta_x'''
    mus = 1 - mus
    T = S * mus + (1 - mus) * np.eye(n_class)
    return T


def S_to_T_torch(S, diag, n_class = 4, plot = False):
    '''Converts switch matrix S to transition T based on the corruption vector mus'''
    diag = 1 - diag
    T = torch.eye(n_class, n_class) - torch.matmul((torch.eye(n_class, n_class) - S),
                                                   (torch.eye(n_class, n_class) * diag))
    T = torch.transpose(T, 0, 1)
    return T


def apply_class_noise(y, T, n_class):
    '''Apply class transformation T to labels y'''
    y_noisy = np.zeros_like(y)
    probas = y.dot(T)
    for i, p in enumerate(probas):
        p /= p.sum()
        k = np.random.choice(n_class, p = p)
        y_noisy[i][k] = 1
    print("Switched {}% of occurences".format(100 * (1 - (y_noisy == y).all(axis = 1).mean())))
    return y_noisy


def apply_noise(y, noises, S, n_class):
    '''Apply class transformation -S- with magnitude -noises- to labels -y-'''
    y_noisy = np.zeros_like(y)
    for i, noise in enumerate(noises):
        T = S_to_T(S, mus = noise, n_class = n_class)
        p = y[i].dot(T)
        p /= p.sum()
        if (p < 0).any():
            print(noise, T, p)
        k = np.random.choice(n_class, p = p)
        y_noisy[i][k] = 1
    print("Switched {}% of occurences".format(100 * (1 - (y_noisy == y).all(axis = 1).mean())))
    return y_noisy


def directional_instance_noise(x_tab, y_tab, max_noise = 0.7):
    directions = np.array([[0, 1], [0, 1], [0, 1]])
    dists = []
    for x, y in zip(x_tab, y_tab):
        d = (directions[np.argmax(y)].dot(x / np.linalg.norm(x)) + 1) / 2
        dists.append(d)
    dists = np.array(dists)
    noises = max_noise * dists
    return noises
