# -*- coding: utf-8 -*-
"""
Correlation helper functions.

Functions
---------
nanxcorr(x, y, maxlag=25)
    Cross correlation ignoring NaNs.
corr2_coeff(A, B)
    Calculate the correlation coefficient between two 2D arrays.

Written 2025, DMM
"""

import numpy as np
import pandas as pd


def nanxcorr(x, y, maxlag=25):

    lags = range(-maxlag, maxlag)
    cc = []

    for i in range(0,len(lags)):
        
        yshift = np.roll(y, lags[i])
        
        use = ~pd.isnull(x + yshift)
        
        x_arr = np.asarray(x, dtype=object)
        yshift_arr = np.asarray(yshift, dtype=object)

        x_use = x_arr[use]
        yshift_use = yshift_arr[use]
        x_use = (x_use - np.mean(x_use)) / (np.std(x_use) * len(x_use))

        yshift_use = (yshift_use - np.mean(yshift_use)) / np.std(yshift_use)
        
        cc.append(np.correlate(x_use, yshift_use))

    cc_out = np.hstack(np.stack(cc))
    
    return cc_out, lags


def corr2_coeff(A, B):
    # this is faster than scipy implementations

    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    corr_coeff = np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))

    return corr_coeff[0][0]


def corrcoef(x, y):
    # equivilent to the matlab function corrcoef
    # two 1D arrays
    # this will be faster than scipy

    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")
    if x.size != y.size:
        raise ValueError("Arrays must have the same length.")

    x_centered = x - x.mean()
    y_centered = y - y.mean()

    d = np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
    corr = np.dot(x_centered, y_centered) / d

    return corr


def calc_cohen_d(a, b):
    # A and B must be 1D vectors

    n1 = np.size(a)
    n2 = np.size(b)
    std1 = np.std(a)
    std2 = np.std(b)

    pooled_std = np.sqrt(((n1 - 1)*std1**2 + (n2 - 1)*std2**2) / (n1 + n2 - 2))

    d = (np.mean(a) - np.mean(b)) / pooled_std

    return d
