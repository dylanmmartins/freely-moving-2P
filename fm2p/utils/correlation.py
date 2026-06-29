# -*- coding: utf-8 -*-
"""
fm2p/utils/correlation.py

Correlation and effect-size helper functions.

Functions
---------
nanxcorr
    Cross-correlation across lags, ignoring NaN samples.
corr2_coeff
    Pearson r between two 2D arrays (faster than scipy for small matrices).
corrcoef
    Pearson r between two 1D arrays (equivalent to MATLAB corrcoef).
calc_cohen_d
    Cohen's d effect size between two 1D distributions.


DMM, March 2025
"""

import numpy as np
import pandas as pd


def nanxcorr(x, y, maxlag=25):
    """ Cross-correlation between x and y at integer lags, ignoring NaN samples.

    Parameters
    ----------
    x : array-like
        Reference signal.
    y : array-like
        Signal to shift; NaN samples are excluded pairwise at each lag.
    maxlag : int
        Maximum lag magnitude; lags run from -maxlag to maxlag-1.

    Returns
    -------
    cc_out : np.ndarray
        Correlation coefficient at each lag.
    lags : range
        Corresponding lag values.
    """

    lags = range(-maxlag, maxlag)
    cc = []

    for i in range(0, len(lags)):

        yshift = np.roll(y, lags[i])

        # Exclude frames where either signal is NaN.
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
    """ Pearson r between rows of A and rows of B.

    Faster than scipy for the common (1xN, 1xN) case used in split-half tests.
    Returns a scalar when both A and B have a single row.

    Parameters
    ----------
    A : np.ndarray, shape (1, N)
    B : np.ndarray, shape (1, N)

    Returns
    -------
    float
        Pearson correlation coefficient.
    """

    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    corr_coeff = np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))

    return corr_coeff[0][0]


def corrcoef(x, y):
    """ Pearson r between two 1D arrays (equivalent to MATLAB corrcoef(x, y)). """

    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError('Inputs must be 1D arrays.')
    if x.size != y.size:
        raise ValueError('Arrays must have the same length.')

    x_centered = x - x.mean()
    y_centered = y - y.mean()

    d = np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
    corr = np.dot(x_centered, y_centered) / d

    return corr


def calc_cohen_d(a, b):
    """ Cohen's d effect size between two 1D distributions.

    Parameters
    ----------
    a : np.ndarray
        First group.
    b : np.ndarray
        Second group.

    Returns
    -------
    d : float
        Positive d means a > b.
    """

    n1 = np.size(a)
    n2 = np.size(b)
    std1 = np.std(a)
    std2 = np.std(b)

    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    d = (np.mean(a) - np.mean(b)) / pooled_std

    return d
