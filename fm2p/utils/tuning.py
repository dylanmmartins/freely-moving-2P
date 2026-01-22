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
from tqdm import tqdm
from scipy.fft import dct

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


def calc_modind(bins, tuning, fr=None, thresh=0.33):
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

    if fr is not None:
        # Mean firing rate across the recording
        b = np.nanmean(fr)
    else:
        b = np.nanmin(tuning)
    peak_val = np.nanmax(tuning)

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

def calc_tuning_reliability(spikes, behavior, bins, ncnk=10, ret_terr=False):
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

    if ret_terr:
        total_error = np.sum(np.abs(tuning1[0] - tuning2[0]))
        return pearson_result, total_error

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


def calc_reliability_d(spikes, behavior, bins, n_cnk=10, n_shfl=100, thresh=1.):
    # for all cells at once (spikes must be 2D, axis=0 is all cells in a recording)

    n_cells = np.size(spikes, 0)
    n_frames = np.size(spikes, 1)

    cnk_sz = n_frames // n_cnk
    all_inds = np.arange(0, n_frames)

    tunings = np.zeros([
        2,  # state (true or null)
        n_shfl,
        2,  # split (first or second half)
        n_cells,
        np.size(bins) - 1
    ]) * np.nan

    correlations = np.zeros([
        2,
        n_shfl,
        n_cells
    ])

    for state_i in range(2):

        # state 0 is the true data
        # state 1 is the null data / rolled spikes

        for shfl_i in tqdm(range(n_shfl)):
        
            np.random.seed(shfl_i)

            use_spikes = spikes.copy()

            if state_i == 1:
                # roll spikes a random distance relative to behavior
                roll_distance = np.random.randint(int(n_frames*0.10), int(n_frames*0.90))
                use_spikes = np.roll(use_spikes, roll_distance, axis=1)

            chunk_order = np.arange(n_cnk)
            np.random.shuffle(chunk_order)

            split1_inds = []
            split2_inds = []

            for cnk_i, cnk in enumerate(chunk_order[:(n_cnk//2)]):
                _inds = all_inds[(cnk_sz*cnk) : ((cnk_sz*(cnk+1)))]
                split1_inds.extend(_inds)

            for cnk_i, cnk in enumerate(chunk_order[(n_cnk//2):]):
                _inds = all_inds[(cnk_sz*cnk) : ((cnk_sz*(cnk+1)))]
                split2_inds.extend(_inds)

            # list of every index that goes into the two halves of the data
            split1_inds = np.array(np.sort(split1_inds)).astype(int)
            split2_inds = np.array(np.sort(split2_inds)).astype(int)

            if len(split1_inds)<1 or len(split2_inds)<1:
                print('no indices used for tuning reliability measure... len of usable recording was:')
                print(n_frames)

            _, tuning1, _ = fm2p.tuning_curve(
                use_spikes[:, split1_inds],
                behavior[split1_inds],
                bins
            )
            _, tuning2, _ = fm2p.tuning_curve(
                use_spikes[:, split2_inds],
                behavior[split2_inds],
                bins
            )

            tunings[state_i,shfl_i,0,:,:] = tuning1
            tunings[state_i,shfl_i,1,:,:] = tuning2

    correlations = np.zeros([
        n_shfl,
        2,    # state [true, null]
        n_cells
    ]) * np.nan

    tunings_masked = tunings.copy()
    tunings_masked[np.isnan(tunings_masked)] = 0

    for shfl_i in range(n_shfl):
        bin_mask = ~np.isnan(tunings[0,0,0,0,:])
        correlations[shfl_i,0,:] = [fm2p.corrcoef(tunings_masked[0,shfl_i,0,c,:], tunings_masked[0,shfl_i,1,c,:]) for c in range(n_cells)]
        correlations[shfl_i,1,:] = [fm2p.corrcoef(tunings_masked[1,shfl_i,0,c,:], tunings_masked[1,shfl_i,1,c,:]) for c in range(n_cells)]

    # If correlation is np.nan for any shuffles, drop it so that cohen d values can still be calculated across cells
    # since it will be a nan spanning all cells if one shuffle failed.
    mask = ~np.isnan(correlations[:,0,:])[:,0] * ~np.isnan(correlations[:,1,:])[:,0]
    cohen_d_vals = np.array([fm2p.calc_cohen_d(correlations[mask,0,c], correlations[mask,1,c]) for c in range(n_cells)])

    is_reliable = cohen_d_vals > thresh
    reliable_inds = np.where(is_reliable)[0]

    reliability_dict = {
        'tunings': tunings,
        'correlations': correlations,
        'cohen_d_vals': cohen_d_vals,
        'reliable_by_shuffle': is_reliable,
    }

    return reliability_dict


def spectral_slope(tuning_curve):
    coeffs = dct(tuning_curve, norm='ortho')
    power = coeffs**2
    freqs = np.arange(1, len(power))  # Skip DC
    log_power = np.log(power[1:])     # Ignore DC (coeff[0])
    slope, _ = np.polyfit(np.log(freqs), log_power, 1)
    return slope  # more negative = smoother


def calc_spectral_noise(tunings, thresh=-1.25):
    nCells = np.size(tunings, 0)
    vals = np.zeros(nCells) * np.nan
    rel = np.zeros(nCells)
    for c in range(nCells):
        try:
            vals[c] = spectral_slope(tunings[c,:])
        except np.linalg.LinAlgError:
            vals[c] = np.nan
            rel[c] = np.nan
            continue
        if vals[c] <= thresh:
            rel[c] = 1
    return vals, rel


def calc_multicell_modulation(tunings, thresh=0.33):
    # if calculating for a light/dark recording, spikes should
    # be spikes for the specific condition, not the full recording
    # since baseline firing rates will be different

    peaks = np.max(tunings,1)
    baselines = np.zeros_like(peaks)
    for c in range(np.size(peaks)):
        baselines[c] = np.nanpercentile(tunings[c,:,], 10) # changed from simple nanmean across spike rate... jan 21 2026

    mod = np.zeros(np.size(peaks)) * np.nan
    # diff over sum
    for c in range(np.size(peaks)):
        mod[c] = (peaks[c] - baselines[c]) / (peaks[c] + baselines[c])

    is_modulated = mod > thresh

    return mod, is_modulated



def calc_radhist(orientation, depth, spikes, xbins, ybins):
    
    occ_hist, _, _ = np.histogram2d(orientation, depth,
                                    bins=(xbins, ybins))
    sp_hist, _, _ = np.histogram2d(orientation, depth,
                                   bins=(xbins, ybins), weights=spikes)

    hist = sp_hist.copy() / occ_hist.copy()
    hist[np.isnan(hist)] = 0.
    hist[~np.isfinite(hist)] = 0.

    return hist