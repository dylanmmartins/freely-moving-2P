# -*- coding: utf-8 -*-
"""
Tuning curve functions.

Functions
---------
tuning_curve(sps, x, x_range)
    Calculate tuning curve  of neurons to a 1D variable.
plot_tuning(ax, var_cent, tuning, tuning_err, color, rad=True)
    Plot tuning curve of neurons to a 1D variable.
calc_modind(bins, tuning, fr, thresh=0.33)
    Calculate modulation index and peak of tuning curve.
calc_tuning_reliability1(spikes, behavior, bins, splits_inds)
    Calculate tuning reliability of a neuron across peak/trough comparisons of 10 splits.
calc_tuning_reliability(spikes, behavior, bins, ncnk=10)
    Calculate tuning reliability between two halves of the data.

Author: DMM, last modified May 2025
"""


import numpy as np
import scipy.stats
# from scipy.stats import pearson

import fm2p


def tuning_curve(sps, x, x_range):
    """ Calculate tuning curve of neurons to a 1D variable.

    Parameters
    ----------
    sps : np.array
        Spike data. Shape should be (n_cells, n_timepoints).
    x : np.array
        Variable data. Shape should be (n_cells, n_timepoints). The
        timepoints should match those for `sps`, either by interpolation
        or by binning.
    x_range : np.array
        Array of values to bin x into.
    
    Returns
    -------
    var_cent : np.array
        Array of values at the center of each bin. Shape is (n_bins,)
    tuning : np.array
        Array of mean spike counts for each bin. Shape is (n_cells, n_bins).
    tuning_err : np.array
        Array of standard error of the mean spike counts for each bin. Shape
        is (n_cells, n_bins).
    """

    n_cells = np.size(sps,0)

    scatter = np.zeros((n_cells, np.size(x,0)))

    tuning = np.zeros((n_cells, len(x_range)-1))
    tuning_err = tuning.copy()
    var_cent = np.zeros(len(x_range)-1)
    
    # Calculate the bin centers
    for j in range(len(x_range)-1):

        var_cent[j] = 0.5*(x_range[j] + x_range[j+1])
    
    # Calculate the mean and standard error within each bin
    for n in range(n_cells):
        
        scatter[n,:] = sps[n,:]
        
        for j in range(len(x_range)-1):
            
            usePts = (x>=x_range[j]) & (x<x_range[j+1])
            
            tuning[n,j] = np.nanmean(scatter[n, usePts])
            
            # Normalize by count
            tuning_err[n,j] = np.nanstd(scatter[n, usePts]) / np.sqrt(np.count_nonzero(usePts))

    return var_cent, tuning, tuning_err


def plot_tuning(ax, var_cent, tuning, tuning_err, color, rad=True):
    """ Plot tuning curve of neurons to a 1D variable.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    var_cent : np.array
        Array of values at the center of each bin. Shape is (n_bins,).
    tuning : np.array
        Array of mean spike counts for each bin. Shape is (n_cells, n_bins).
    tuning_err : np.array
        Array of standard error of the mean spike counts for each bin. Shape
        is (n_cells, n_bins).
    color : str
        Color to plot the tuning curve.
    rad : bool
        If True, convert the variable centers to degrees. Default is True.
    """

    if rad:
        usebins = np.rad2deg(var_cent)
    else:
        usebins = var_cent.copy()

    ax.plot(usebins, tuning[0], color=color)
    ax.fill_between(
        usebins,
        tuning[0]+tuning_err[0],
        tuning[0]-tuning_err[0],
        alpha=0.3, color=color
    )
    ax.set_xlim([var_cent[0], var_cent[-1]])


def calc_modind(bins, tuning, fr, thresh=0.33):
    """ Calculate modulation index and peak of tuning curve.

    Modulation index of 0.33 is a double of firing rate relative to the baseline.

    Parameters
    ----------
    bins : np.array
        Array of values at the center of each bin. Shape is (n_bins,).
    tuning : np.array
        Array of mean spike counts for each bin. Shape is (n_cells, n_bins).
    fr : np.array
        Firing rate of the neuron over the entire recording. Shape is (n_cells,).
        This will be used to calculate the baseline firing rate.
    thresh : float
        Threshold for modulation index. Default is 0.33.
    
    Returns
    -------
    modind : float
        Modulation index of the tuning curve. This is a measure of how much the
        firing rate changes relative to the baseline firing rate.
    peak : float
        Peak of the tuning curve. This is the value of the variable at which
        the firing rate is highest.
    """

    # Mean firing rate across the recording
    b = np.nanmean(fr)
    peak_val = np.nanmax(tuning)

    # print(b, peak_val)

    # diff over sum
    modind = (peak_val - b) / (peak_val + b)

    peak = np.nan
    if modind > thresh:
        peak = bins[np.nanargmax(tuning)]

    return modind, peak


def calc_tuning_reliability1(spikes, behavior, bins, splits_inds):
    """ Calculate tuning reliability of a neuron across peak/trough comparisons of 10 splits.

    Parameters
    ----------
    spikes : np.array
        Spike data. Shape should be (n_cells, n_timepoints).
    behavior : np.array
        Variable data. Shape should be (n_cells, n_timepoints). The
        timepoints should match those for `sps`, either by interpolation
        or by binning.
    
    """
  
    cnk_mins = []
    cnk_maxs = []

    for cnk in range(len(splits_inds)):
        hist_cents, cnk_behavior_tuning, _ = tuning_curve(
            spikes[np.newaxis, splits_inds[cnk]],
            behavior[splits_inds[cnk]],
            bins
        )
        cnk_mins = hist_cents[np.nanargmin(cnk_behavior_tuning)]
        cnk_maxs = hist_cents[np.nanargmax(cnk_behavior_tuning)]

    try:
        pval_across_cnks = scipy.stats.wilcoxon(
            cnk_mins,
            cnk_maxs,
            alternative='less'
        ).pvalue
    except ValueError:
        print('x-y==0 for all elements of this cell, which cannot be computed for wilcox. Skipping this cell.')
        pval_across_cnks = np.nan

    # If the p value is small, the two distributions are significantly different from
    # one another, i.e., the peaks are all different from the troughs. This means that
    # the cell has a reliable peak.

    return pval_across_cnks

def calc_tuning_reliability(spikes, behavior, bins, ncnk=10):
    """ Calculate tuning reliability between two halves of the data.

    Parameters
    ----------
    spikes : np.array
        Spike data. Shape should be (n_cells, n_timepoints).
    behavior : np.array
        Variable data. Shape should be (n_cells, n_timepoints). The
        timepoints should match those for `sps`, either by interpolation
        or by binning.
    bins : np.array
        Array of values to bin x into.
    ncnk : int
        Number of chunks to split the data into. Default is 10.

    Returns
    -------
    p_value : float
        P-value of the correlation between the two halves of the data.
    cc : float
        Correlation coefficient between the two halves of the data.
    """

    _len = np.size(behavior)
    cnk_sz = _len // ncnk

    _all_inds = np.arange(0,_len)

    chunk_order = np.arange(ncnk)
    np.random.shuffle(chunk_order)

    split1_inds = []
    split2_inds = []

    for cnk_i, cnk in enumerate(chunk_order[:(ncnk//2)]):
        _inds = _all_inds[(cnk_sz*cnk) : ((cnk_sz*(cnk+1)))]
        split1_inds.extend(_inds)

    for cnk_i, cnk in enumerate(chunk_order[(ncnk//2):]):
        _inds = _all_inds[(cnk_sz*cnk) : ((cnk_sz*(cnk+1)))]
        split2_inds.extend(_inds)

    # list of every index that goes into the two halves of the data
    split1_inds = np.array(np.sort(split1_inds)).astype(int)
    split2_inds = np.array(np.sort(split2_inds)).astype(int)

    if len(split1_inds)<1 or len(split2_inds)<1:
        print('no indices used for tuning reliability measure... len of usable recording was:')
        print(_len)

    _, tuning1, _ = tuning_curve(
        spikes[:, split1_inds],
        behavior[split1_inds],
        bins
    )
    _, tuning2, _ = tuning_curve(
        spikes[:, split2_inds],
        behavior[split2_inds],
        bins
    )
    
    # Calculate the correlation coefficient (this custom func is
    # more efficient than scipy but does not calculate the p value)
    [tuning1, tuning2] = fm2p.nan_filt([tuning1, tuning2])
    pearson_result = fm2p.corr2_coeff(tuning1, tuning2)

    return pearson_result


def norm_tuning(tuning):

    tuning = tuning - np.nanmean(tuning)
    tuning = tuning / np.std(tuning)
    
    return tuning


def plot_running_median(ax, x, y, n_bins=7):
    """ Plot median of a dataset along a set of horizontal bins.
    
    """

    bins = np.linspace(np.min(x), np.max(x), n_bins)

    bin_means, bin_edges, bin_number = scipy.stats.binned_statistic(
        x[~np.isnan(x) & ~np.isnan(y)],
        y[~np.isnan(x) & ~np.isnan(y)],
        statistic=np.median,
        bins=bins)
    
    bin_std, _, _ = scipy.stats.binned_statistic(
        x[~np.isnan(x) & ~np.isnan(y)],
        y[~np.isnan(x) & ~np.isnan(y)],
        statistic=np.nanstd,
        bins=bins)
    
    hist, _ = np.histogram(
        x[~np.isnan(x) & ~np.isnan(y)],
        bins=bins)
    
    tuning_err = bin_std / np.sqrt(hist)

    ax.plot(bin_edges[:-1] + (np.median(np.diff(bins))/2),
               bin_means,
               '-', color='k')
    
    ax.fill_between(bin_edges[:-1] + (np.median(np.diff(bins))/2),
                       bin_means-tuning_err,
                       bin_means+tuning_err,
                       color='k', alpha=0.2)