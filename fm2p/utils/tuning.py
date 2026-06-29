# -*- coding: utf-8 -*-
"""
fm2p/utils/tuning.py

1D tuning curve computation, reliability scoring, and modulation indices.

Functions
---------
tuning_curve
    Occupancy-normalized spike rate as a function of a 1D behavioral variable.
plot_tuning
    Plot a tuning curve with shaded standard-error band.
calc_modind
    Modulation index and peak location of a tuning curve.
calc_tuning_reliability1
    Tuning reliability via Wilcoxon test across N split-chunk min/max pairs.
calc_tuning_reliability
    Split-half Pearson r of tuning curves across randomly shuffled chunks.
norm_tuning
    Z-score a tuning curve.
plot_running_median
    Plot binned running median and SEM on an axes.
calc_reliability_d
    Split-half reliability scored as Cohen's d vs. a shuffled null distribution.
spectral_slope
    DCT-based spectral slope of a tuning curve (roughness measure).
calc_spectral_noise
    Classify cells as smooth vs noisy based on spectral slope.
calc_multicell_modulation
    Modulation index for a population of tuning curves.
calc_radhist
    2D occupancy-normalized histogram of spike rate vs orientation and depth.


DMM, March 2025
"""

import numpy as np
import scipy.stats
from tqdm import tqdm
from scipy.fft import dct

from .helper import nan_filt
from .correlation import corr2_coeff, corrcoef, calc_cohen_d


def tuning_curve(sps, x, x_range):
    """ Compute occupancy-normalized spike rate as a function of a 1D variable.

    Parameters
    ----------
    sps : np.ndarray, shape (N_cells, N_frames)
        Spike rate array.
    x : np.ndarray, shape (N_frames,)
        Behavioral variable value at each frame.
    x_range : array-like
        Bin edges.

    Returns
    -------
    var_cent : np.ndarray
        Bin centres.
    tuning : np.ndarray, shape (N_cells, N_bins)
        Mean spike rate per bin.
    tuning_err : np.ndarray, shape (N_cells, N_bins)
        Standard error per bin.
    """

    n_cells = np.size(sps, 0)
    scatter = np.zeros((n_cells, np.size(x, 0)))

    tuning = np.zeros((n_cells, len(x_range) - 1))
    tuning_err = tuning.copy()
    var_cent = np.zeros(len(x_range) - 1)

    for j in range(len(x_range) - 1):
        var_cent[j] = 0.5 * (x_range[j] + x_range[j + 1])

    for n in range(n_cells):
        scatter[n, :] = sps[n, :]
        for j in range(len(x_range) - 1):
            usePts = (x >= x_range[j]) & (x < x_range[j + 1])
            tuning[n, j] = np.nanmean(scatter[n, usePts])
            tuning_err[n, j] = np.nanstd(scatter[n, usePts]) / np.sqrt(np.count_nonzero(usePts))

    return var_cent, tuning, tuning_err


def plot_tuning(ax, var_cent, tuning, tuning_err, color, rad=True):
    """ Plot a tuning curve with shaded SEM band.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    var_cent : np.ndarray
        Bin centres (radians if rad=True, else degrees).
    tuning : np.ndarray, shape (1, N_bins)
    tuning_err : np.ndarray, shape (1, N_bins)
    color : str or tuple
    rad : bool
        If True, convert var_cent from radians to degrees for display.
    """

    if rad:
        usebins = np.rad2deg(var_cent)
    else:
        usebins = var_cent.copy()

    ax.plot(usebins, tuning[0], color=color)
    ax.fill_between(
        usebins,
        tuning[0] + tuning_err[0],
        tuning[0] - tuning_err[0],
        alpha=0.3, color=color
    )
    ax.set_xlim([var_cent[0], var_cent[-1]])


def calc_modind(bins, tuning, fr=None, thresh=0.33):
    """ Compute modulation index and peak location of a tuning curve.

    Parameters
    ----------
    bins : np.ndarray
        Bin centres.
    tuning : np.ndarray, shape (N_bins,)
    fr : np.ndarray or None
        If provided, use the mean of fr as the baseline instead of the tuning minimum.
    thresh : float
        Minimum modulation index for a peak to be reported.

    Returns
    -------
    modind : float
    peak : float or np.nan
        Bin centre of the peak bin, or NaN if modind <= thresh.
    """

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
    """ Tuning reliability via Wilcoxon test on min/max locations across split chunks.

    Parameters
    ----------
    spikes : np.ndarray, shape (N_frames,)
        Single-cell spike trace.
    behavior : np.ndarray, shape (N_frames,)
        Behavioral variable.
    bins : array-like
        Bin edges.
    splits_inds : list of array-like
        Each element is an index array for one chunk.

    Returns
    -------
    pval_across_cnks : float
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
        print('x-y==0 for all elements of this cell -- cannot compute Wilcoxon. Skipping.')
        pval_across_cnks = np.nan

    return pval_across_cnks


def calc_tuning_reliability(spikes, behavior, bins, ncnk=10, ret_terr=False):
    """ Split-half Pearson r of tuning curves across randomly shuffled chunks.

    Parameters
    ----------
    spikes : np.ndarray, shape (N_cells, N_frames)
    behavior : np.ndarray, shape (N_frames,)
    bins : array-like
        Bin edges.
    ncnk : int
        Number of chunks to split the data into before creating two halves.
    ret_terr : bool
        If True, also return the total absolute error between the two half-tunings.

    Returns
    -------
    pearson_result : float
        Split-half correlation.
    total_error : float (only if ret_terr=True)
    """

    _len = np.size(behavior)
    cnk_sz = _len // ncnk

    _all_inds = np.arange(0, _len)

    chunk_order = np.arange(ncnk)
    np.random.shuffle(chunk_order)

    split1_inds = []
    split2_inds = []

    for cnk_i, cnk in enumerate(chunk_order[:(ncnk // 2)]):
        _inds = _all_inds[(cnk_sz * cnk): ((cnk_sz * (cnk + 1)))]
        split1_inds.extend(_inds)

    for cnk_i, cnk in enumerate(chunk_order[(ncnk // 2):]):
        _inds = _all_inds[(cnk_sz * cnk): ((cnk_sz * (cnk + 1)))]
        split2_inds.extend(_inds)

    split1_inds = np.array(np.sort(split1_inds)).astype(int)
    split2_inds = np.array(np.sort(split2_inds)).astype(int)

    if len(split1_inds) < 1 or len(split2_inds) < 1:
        print('No indices used for tuning reliability -- len of usable recording: {}'.format(_len))

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
    """ Z-score a tuning curve. """

    tuning = tuning - np.nanmean(tuning)
    tuning = tuning / np.std(tuning)

    return tuning


def plot_running_median(ax, x, y, n_bins=7):
    """ Plot binned running median and SEM on an axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x : np.ndarray
    y : np.ndarray
    n_bins : int
    """

    mask = ~np.isnan(x)
    x = x[mask]
    y = y[mask]

    bins = np.linspace(np.min(x), np.max(x), n_bins)

    bin_means, bin_edges, _ = scipy.stats.binned_statistic(
        x, y, statistic=np.nanmedian, bins=bins)

    bin_std, _, _ = scipy.stats.binned_statistic(
        x, y, statistic=np.nanstd, bins=bins)

    hist, _, _ = scipy.stats.binned_statistic(
        x, y, statistic=lambda y: np.sum(~np.isnan(y)), bins=bins)

    tuning_err = bin_std / np.sqrt(hist)

    ax.plot(bin_edges[:-1] + (np.median(np.diff(bins)) / 2), bin_means, '-', color='k')

    ax.fill_between(
        bin_edges[:-1] + (np.median(np.diff(bins)) / 2),
        bin_means - tuning_err,
        bin_means + tuning_err,
        color='k', alpha=0.2
    )


def calc_reliability_d(spikes, behavior, bins, n_cnk=10, n_shfl=100, thresh=1.):
    """ Split-half reliability scored as Cohen's d vs. a shuffled null distribution.

    Runs n_shfl split-half Pearson r computations on both real data and
    roll-shuffled data, then computes Cohen's d between the two correlation
    distributions. Higher d means the real split-half correlations are more
    consistently above the null.

    Parameters
    ----------
    spikes : np.ndarray, shape (N_cells, N_frames)
    behavior : np.ndarray, shape (N_frames,)
    bins : array-like
        Bin edges.
    n_cnk : int
        Number of chunks for the split-half procedure.
    n_shfl : int
        Number of shuffle repetitions.
    thresh : float
        Cohen's d threshold for classifying a cell as reliable.

    Returns
    -------
    reliability_dict : dict
        'tunings', 'correlations', 'cohen_d_vals', 'reliable_by_shuffle'.
    """

    n_cells = np.size(spikes, 0)
    n_frames = np.size(spikes, 1)

    cnk_sz = n_frames // n_cnk
    all_inds = np.arange(0, n_frames)

    tunings = np.zeros([2, n_shfl, 2, n_cells, np.size(bins) - 1]) * np.nan

    for state_i in range(2):

        # state 0 = real data; state 1 = roll-shuffled null
        for shfl_i in tqdm(range(n_shfl)):

            np.random.seed(shfl_i)

            use_spikes = spikes.copy()

            if state_i == 1:
                roll_distance = np.random.randint(int(n_frames * 0.10), int(n_frames * 0.90))
                use_spikes = np.roll(use_spikes, roll_distance, axis=1)

            chunk_order = np.arange(n_cnk)
            np.random.shuffle(chunk_order)

            split1_inds = []
            split2_inds = []

            for cnk_i, cnk in enumerate(chunk_order[:(n_cnk // 2)]):
                _inds = all_inds[(cnk_sz * cnk): ((cnk_sz * (cnk + 1)))]
                split1_inds.extend(_inds)

            for cnk_i, cnk in enumerate(chunk_order[(n_cnk // 2):]):
                _inds = all_inds[(cnk_sz * cnk): ((cnk_sz * (cnk + 1)))]
                split2_inds.extend(_inds)

            split1_inds = np.array(np.sort(split1_inds)).astype(int)
            split2_inds = np.array(np.sort(split2_inds)).astype(int)

            if len(split1_inds) < 1 or len(split2_inds) < 1:
                print('No indices used for tuning reliability -- N usable frames: {}'.format(n_frames))

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

            tunings[state_i, shfl_i, 0, :, :] = tuning1
            tunings[state_i, shfl_i, 1, :, :] = tuning2

    correlations = np.zeros([n_shfl, 2, n_cells]) * np.nan

    tunings_masked = tunings.copy()
    tunings_masked[np.isnan(tunings_masked)] = 0

    for shfl_i in range(n_shfl):
        correlations[shfl_i, 0, :] = [corrcoef(tunings_masked[0, shfl_i, 0, c, :], tunings_masked[0, shfl_i, 1, c, :]) for c in range(n_cells)]
        correlations[shfl_i, 1, :] = [corrcoef(tunings_masked[1, shfl_i, 0, c, :], tunings_masked[1, shfl_i, 1, c, :]) for c in range(n_cells)]

    mask = ~np.isnan(correlations[:, 0, :])[:, 0] * ~np.isnan(correlations[:, 1, :])[:, 0]
    cohen_d_vals = np.array([calc_cohen_d(correlations[mask, 0, c], correlations[mask, 1, c]) for c in range(n_cells)])

    is_reliable = cohen_d_vals > thresh

    reliability_dict = {
        'tunings': tunings,
        'correlations': correlations,
        'cohen_d_vals': cohen_d_vals,
        'reliable_by_shuffle': is_reliable,
    }

    return reliability_dict


def spectral_slope(tuning_curve):
    """ DCT-based spectral slope of a tuning curve; more negative = smoother.

    Parameters
    ----------
    tuning_curve : np.ndarray, shape (N_bins,)

    Returns
    -------
    slope : float
    """

    coeffs = dct(tuning_curve, norm='ortho')
    power = coeffs ** 2
    freqs = np.arange(1, len(power))
    log_power = np.log(power[1:])
    slope, _ = np.polyfit(np.log(freqs), log_power, 1)
    return slope


def calc_spectral_noise(tunings, thresh=-1.25):
    """ Classify cells as smooth vs. noisy based on DCT spectral slope.

    Parameters
    ----------
    tunings : np.ndarray, shape (N_cells, N_bins)
    thresh : float
        Spectral slope threshold; values <= thresh are classified as smooth (reliable).

    Returns
    -------
    vals : np.ndarray, shape (N_cells,)
        Spectral slope per cell.
    rel : np.ndarray, shape (N_cells,)
        1 if smooth (reliable), 0 otherwise.
    """

    nCells = np.size(tunings, 0)
    vals = np.zeros(nCells) * np.nan
    rel = np.zeros(nCells)
    for c in range(nCells):
        try:
            vals[c] = spectral_slope(tunings[c, :])
        except np.linalg.LinAlgError:
            vals[c] = np.nan
            rel[c] = np.nan
            continue
        if vals[c] <= thresh:
            rel[c] = 1
    return vals, rel


def calc_multicell_modulation(tunings, thresh=0.33):
    """ Compute modulation index for a population of tuning curves.

    Parameters
    ----------
    tunings : np.ndarray, shape (N_cells, N_bins)
    thresh : float
        Modulation index threshold for classification.

    Returns
    -------
    mod : np.ndarray, shape (N_cells,)
    is_modulated : np.ndarray of bool
    """

    peaks = np.nanmax(tunings, 1)
    baselines = np.array([np.nanmean(tunings[c]) for c in range(len(peaks))])

    mod = np.zeros(len(peaks)) * np.nan
    for c in range(len(peaks)):
        denom = peaks[c] + baselines[c]
        mod[c] = (peaks[c] - baselines[c]) / denom if denom > 0 else 0.0

    is_modulated = mod > thresh

    return mod, is_modulated


def calc_radhist(orientation, depth, spikes, xbins, ybins):
    """ Occupancy-normalized 2D histogram of spike rate vs orientation and depth.

    Parameters
    ----------
    orientation : np.ndarray
    depth : np.ndarray
    spikes : np.ndarray
    xbins : array-like
        Bin edges for orientation axis.
    ybins : array-like
        Bin edges for depth axis.

    Returns
    -------
    hist : np.ndarray, shape (N_xbins-1, N_ybins-1)
    """

    occ_hist, _, _ = np.histogram2d(orientation, depth, bins=(xbins, ybins))
    sp_hist, _, _ = np.histogram2d(orientation, depth, bins=(xbins, ybins), weights=spikes)

    hist = sp_hist.copy() / occ_hist.copy()
    hist[np.isnan(hist)] = 0.
    hist[~np.isfinite(hist)] = 0.

    return hist
