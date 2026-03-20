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

from .helper import nan_filt
from .correlation import corr2_coeff, corrcoef, calc_cohen_d


def tuning_curve(sps, x, x_range):

    n_cells = np.size(sps,0)
    scatter = np.zeros((n_cells, np.size(x,0)))

    tuning = np.zeros((n_cells, len(x_range)-1))
    tuning_err = tuning.copy()
    var_cent = np.zeros(len(x_range)-1)
    
    for j in range(len(x_range)-1):

        var_cent[j] = 0.5*(x_range[j] + x_range[j+1])
    
    for n in range(n_cells):
        
        scatter[n,:] = sps[n,:]
        
        for j in range(len(x_range)-1):
            
            usePts = (x>=x_range[j]) & (x<x_range[j+1])
            tuning[n,j] = np.nanmean(scatter[n, usePts])
            tuning_err[n,j] = np.nanstd(scatter[n, usePts]) / np.sqrt(np.count_nonzero(usePts))

    return var_cent, tuning, tuning_err


def plot_tuning(ax, var_cent, tuning, tuning_err, color, rad=True):

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

    if fr is not None:
        b = np.nanmean(fr)
    else:
        b = np.nanmin(tuning)
    peak_val = np.nanmax(tuning)

    modind = (peak_val - b) / (peak_val + b)

    peak = np.nan
    if modind > thresh:
        peak = bins[np.nanargmax(tuning)]

    return modind, peak


def calc_tuning_reliability1(spikes, behavior, bins, splits_inds):

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

    return pval_across_cnks

def calc_tuning_reliability(spikes, behavior, bins, ncnk=10, ret_terr=False):

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
    
    [tuning1, tuning2] = nan_filt([tuning1, tuning2])
    pearson_result = corr2_coeff(tuning1, tuning2)

    if ret_terr:
        total_error = np.sum(np.abs(tuning1[0] - tuning2[0]))
        return pearson_result, total_error

    return pearson_result


def norm_tuning(tuning):

    tuning = tuning - np.nanmean(tuning)
    tuning = tuning / np.std(tuning)
    
    return tuning


def plot_running_median(ax, x, y, n_bins=7):

    mask = ~np.isnan(x)
    x = x[mask]
    y = y[mask]

    bins = np.linspace(np.min(x), np.max(x), n_bins)

    bin_means, bin_edges, bin_number = scipy.stats.binned_statistic(
        x,
        y,
        statistic=np.nanmedian,
        bins=bins)
    
    bin_std, _, _ = scipy.stats.binned_statistic(
        x,
        y,
        statistic=np.nanstd,
        bins=bins)
    
    hist, _, _ = scipy.stats.binned_statistic(
        x,
        y,
        statistic=lambda y: np.sum(~np.isnan(y)),
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

    n_cells = np.size(spikes, 0)
    n_frames = np.size(spikes, 1)

    cnk_sz = n_frames // n_cnk
    all_inds = np.arange(0, n_frames)

    tunings = np.zeros([
        2,
        n_shfl,
        2,
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

            split1_inds = np.array(np.sort(split1_inds)).astype(int)
            split2_inds = np.array(np.sort(split2_inds)).astype(int)

            if len(split1_inds)<1 or len(split2_inds)<1:
                print('no indices used for tuning reliability measure... len of usable recording was:')
                print(n_frames)

            _, tuning1, _ = tuning_curve(
                use_spikes[:, split1_inds],
                behavior[split1_inds],
                bins
            )
            _, tuning2, _ = tuning_curve(
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
        correlations[shfl_i,0,:] = [corrcoef(tunings_masked[0,shfl_i,0,c,:], tunings_masked[0,shfl_i,1,c,:]) for c in range(n_cells)]
        correlations[shfl_i,1,:] = [corrcoef(tunings_masked[1,shfl_i,0,c,:], tunings_masked[1,shfl_i,1,c,:]) for c in range(n_cells)]

    mask = ~np.isnan(correlations[:,0,:])[:,0] * ~np.isnan(correlations[:,1,:])[:,0]
    cohen_d_vals = np.array([calc_cohen_d(correlations[mask,0,c], correlations[mask,1,c]) for c in range(n_cells)])

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

    peaks = np.nanmax(tunings, 1)
    baselines = np.array([np.nanmean(tunings[c]) for c in range(len(peaks))])

    mod = np.zeros(len(peaks)) * np.nan
    for c in range(len(peaks)):
        denom = peaks[c] + baselines[c]
        mod[c] = (peaks[c] - baselines[c]) / denom if denom > 0 else 0.0

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