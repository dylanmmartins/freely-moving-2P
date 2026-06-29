# -*- coding: utf-8 -*-
"""
fm2p/utils/plot.py

Miscellaneous matplotlib helpers.

Functions
---------
plot_confidence_ellipse
    Draw a confidence ellipse for 2D data on an existing axes.


DMM, September 2025
"""

import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import chi2


def plot_confidence_ellipse(x, y, ax, confidence=0.90, **kwargs):
    """ Draw a confidence ellipse for 2D data on ax.

    Parameters
    ----------
    x : np.ndarray
        X coordinates.
    y : np.ndarray
        Y coordinates (same length as x).
    ax : matplotlib.axes.Axes
        Target axes.
    confidence : float
        Confidence level, e.g. 0.90 for 90%.
    **kwargs
        Passed to matplotlib.patches.Ellipse.
    """

    if x.size != y.size:
        raise ValueError('x and y must be the same size')

    cov = np.cov(x, y)
    mean_x, mean_y = np.mean(x), np.mean(y)

    # Eigenvectors give the ellipse orientation; eigenvalues give axis lengths.
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Chi-squared CDF inverse with 2 DOF maps confidence to a radial scale factor.
    chi2_val = chi2.ppf(confidence, df=2)
    width, height = 2 * np.sqrt(vals * chi2_val)

    ellipse = Ellipse((mean_x, mean_y), width, height, angle=angle, **kwargs)
    ax.add_patch(ellipse)
