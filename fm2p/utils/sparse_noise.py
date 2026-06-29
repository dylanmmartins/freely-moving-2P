# -*- coding: utf-8 -*-
"""
fm2p/utils/sparse_noise.py

Spike-triggered average (STA) computation from sparse-noise visual stimuli.

Functions
---------
find_delay_frames
    Estimate the stimulus-response lag via cross-correlation.
jaccard_topk
    Jaccard overlap of the top-k pixels between two STAs.
compute_calcium_sta_spatial
    Compute per-cell STAs from a sparse-noise stimulus and calcium spike traces.
keep_best_STA_lag
    Retain the single best temporal lag for each cell's STA.
compute_split_STAs
    Compute full, first-half, and second-half STAs and their split correlation.


DMM, September 2025
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.signal import correlate
import gc

from .time import find_closest_timestamp


def find_delay_frames(stim_s, pop_s, max_lag=80):
    """ Estimate the stimulus-to-response lag (in frames) via cross-correlation.

    Parameters
    ----------
    stim_s : array-like
        Stimulus-derived trace (e.g., mean luminance per frame).
    pop_s : array-like
        Population-mean spike trace at the stimulus frame rate.
    max_lag : int
        Maximum lag magnitude to consider.

    Returns
    -------
    lag : int
        Best lag in frames (positive = stimulus leads response).
    """

    stim_s = np.asarray(stim_s).ravel()
    pop_s = np.asarray(pop_s).ravel()

    stim_s = (stim_s - np.mean(stim_s)) / np.std(stim_s)
    pop_s = (pop_s - np.mean(pop_s)) / np.std(pop_s)

    corr = correlate(stim_s, pop_s, mode='full')
    lags = np.arange(-len(stim_s) + 1, len(pop_s))
    mask = (lags >= -max_lag) & (lags <= max_lag)
    lag = lags[mask][np.argmax(corr[mask])]

    return lag


def jaccard_topk(sta1, sta2, pct=1.0):
    """ Jaccard overlap of the top-pct% highest-magnitude pixels between two STAs.

    Tests whether the same pixels are strongest in both halves of the data,
    which is a robust split-half reliability check that does not require the
    magnitude scales to match.

    Parameters
    ----------
    sta1, sta2 : np.ndarray
        Flattened or 2D STA arrays.
    pct : float
        Percentage of pixels to threshold (default 1%).

    Returns
    -------
    float
        Jaccard index in [0, 1].
    """

    n = sta1.size
    k = max(1, int(np.round(n * (pct / 100.0))))

    idx1 = np.argpartition(np.abs(sta1.ravel()), -k)[-k:]
    idx2 = np.argpartition(np.abs(sta2.ravel()), -k)[-k:]

    s1 = set(idx1.tolist())
    s2 = set(idx2.tolist())

    inter = len(s1 & s2)
    union = len(s1 | s2)

    return inter / union if union > 0 else 0.0


def compute_calcium_sta_spatial(
        stimulus,
        spikes,
        stim_times,
        spike_times,
        window=20,
        delay=None,
        max_lag_frames=80,
        skip_trim=False
):
    """ Compute spike-triggered averages from a sparse-noise stimulus.

    Interpolates each cell's spike trace onto the stimulus timebase, then
    computes a weighted average of stimulus frames over a temporal window.

    Parameters
    ----------
    stimulus : np.ndarray, shape (N_frames, H, W)
        Sparse noise stimulus frames.
    spikes : np.ndarray, shape (N_cells, N_spike_samples)
        Spike rate traces at spike_times.
    stim_times : np.ndarray
        Timestamp for each stimulus frame (seconds).
    spike_times : np.ndarray
        Timestamp for each spike sample (seconds).
    window : int
        Number of frames before the spike to include in the average.
    delay : np.ndarray or None
        Per-cell delay in frames. If None, estimated from cross-correlation.
    max_lag_frames : int
        Maximum lag for auto-detected delay.
    skip_trim : bool
        If False, trim spike_times to the stimulus duration.

    Returns
    -------
    sta_all : np.ndarray, shape (N_cells, window+1, H*W)
    lag_axis : np.ndarray
    est_delay_frames : int
    """

    stimulus = np.asarray(stimulus)
    spikes = np.asarray(spikes)
    stim_times = np.asarray(stim_times)
    spike_times = np.asarray(spike_times)

    if not skip_trim:
        # Trim extra 2P frames that extend beyond the stimulus.
        stimend = np.size(stimulus, 0) / 2
        spikeend, _ = find_closest_timestamp(spike_times, stimend)
        spikes = spikes[:, :spikeend]
        spike_times = spike_times[:spikeend]

    nFrames, stimY, stimX = np.shape(stimulus)

    stim_mean_trace = np.mean(stimulus, axis=(1, 2))

    bg_est = np.median(stimulus)
    white_mask = (stimulus > bg_est)
    black_mask = (stimulus < bg_est)
    signed_stim = (white_mask.astype(np.int16) - black_mask.astype(np.int16))

    flat_signed = np.reshape(signed_stim, [nFrames, stimY * stimX])
    flat_signed = flat_signed - np.mean(flat_signed, axis=0, keepdims=True)

    n_stim, n_features = flat_signed.shape
    n_cells, n_spike_samples = spikes.shape

    if n_spike_samples != len(spike_times):
        raise ValueError('spikes.shape[1] ({}) != len(spike_times) ({})'.format(
            n_spike_samples, len(spike_times)))

    pop_trace = np.mean(spikes, axis=0)

    bin_edges = np.concatenate([
        stim_times,
        [stim_times[-1] + np.median(np.diff(stim_times))],
    ])
    pop_sum, _ = np.histogram(spike_times, bins=bin_edges, weights=pop_trace)
    counts, _ = np.histogram(spike_times, bins=bin_edges)
    counts[counts == 0] = 1
    pop_rate_per_frame = pop_sum / counts

    est_delay_frames = 0
    shift_time_cellwise = False
    dt = np.median(np.diff(stim_times))
    if delay is None:
        est_delay_frames = find_delay_frames(
            stim_mean_trace,
            pop_rate_per_frame,
            max_lag=max_lag_frames
        )
        delay_ = est_delay_frames * dt

        stim_times_shifted = stim_times + delay_

    else:
        shift_time_cellwise = True

    sta_all = np.zeros((n_cells, window + 1, n_features))
    eps = 1e-9

    for cell_idx in tqdm(range(n_cells)):
        cell_spikes = spikes[cell_idx, :]

        interp_fn = interp1d(
            spike_times,
            cell_spikes,
            kind='linear',
            fill_value='extrapolate',
            assume_sorted=True
        )

        if shift_time_cellwise:
            stim_times_shifted = stim_times.copy() + (delay[cell_idx] * dt)

        spike_rate_per_frame = interp_fn(stim_times_shifted)

        sta = np.zeros((window + 1, n_features))
        total_rate = 0.

        for i, rate in enumerate(spike_rate_per_frame):
            if rate <= 0 or i < window or i + window + 1 >= n_stim:
                continue

            stim_segment = flat_signed[i - window: i + 1]

            sta += rate * stim_segment
            total_rate += rate

        sta /= (total_rate + eps)
        sta_all[cell_idx] = sta

    lag_axis = np.arange(-window, window + 1)

    del signed_stim, flat_signed
    gc.collect()

    return sta_all, lag_axis, est_delay_frames


def keep_best_STA_lag(STAs):
    """ Retain the single best temporal lag for each cell's STA.

    Parameters
    ----------
    STAs : np.ndarray, shape (N_cells, N_lags, N_features)

    Returns
    -------
    kept_STAs : np.ndarray, shape (N_cells, N_features)
    best_lags : np.ndarray, shape (N_cells,)
    """

    n_cells = np.size(STAs, 0)
    best_lags = np.zeros(n_cells)
    kept_STAs = np.zeros([n_cells, np.size(STAs, 2)])
    for c in range(n_cells):
        lagmax = np.zeros(np.size(STAs, 1)) * np.nan
        for l in range(np.size(STAs, 1)):
            lagmax[l] = np.nanmax(np.abs(STAs[c, l, :]))
        best_lags[c] = np.nanargmax(lagmax)
        kept_STAs[c] = STAs[c, int(best_lags[c]), :]
    return kept_STAs, best_lags


def compute_split_STAs(
        stimulus,
        spikes,
        stim_times,
        spike_times,
        window=13,
        delay=None
):
    """ Compute full, first-half, and second-half STAs and their Jaccard split correlation.

    Parameters
    ----------
    stimulus : np.ndarray, shape (N_frames, H, W)
    spikes : np.ndarray, shape (N_cells, N_samples)
    stim_times : np.ndarray
    spike_times : np.ndarray
    window : int
    delay : np.ndarray or None

    Returns
    -------
    STA : np.ndarray, shape (N_cells, N_features)
    STA1 : np.ndarray -- first-half STA
    STA2 : np.ndarray -- second-half STA
    split_corr : np.ndarray, shape (N_cells,) -- Jaccard index per cell
    best_lags : np.ndarray, shape (N_cells,)
    """

    print('Setting up spike splits.')

    stimulus = np.asarray(stimulus)
    spikes = np.asarray(spikes)
    stim_times = np.asarray(stim_times)
    spike_times = np.asarray(spike_times)

    n_cells = spikes.shape[0]

    spike_split_ind = np.size(spike_times) // 2
    spikes1 = spikes.copy()
    spikes2 = spikes.copy()
    spikes1[:, spike_split_ind:] = 0.
    spikes2[:, :spike_split_ind] = 0.

    print('Computing full sparse noise STAs...')
    STA_, lag_axis, delay = compute_calcium_sta_spatial(
        stimulus, spikes, stim_times, spike_times,
        window=window, delay=np.zeros(n_cells)
    )
    STA, best_lags = keep_best_STA_lag(STA_)

    print('Computing sparse noise STAs for first half of recording...')
    STA1_, lag_axis1, delay1 = compute_calcium_sta_spatial(
        stimulus, spikes1, stim_times, spike_times,
        window=window, delay=np.zeros(n_cells)
    )
    STA1, best_lags1 = keep_best_STA_lag(STA1_)

    print('Computing sparse noise STAs for second half of recording...')
    STA2_, lag_axis2, delay2 = compute_calcium_sta_spatial(
        stimulus, spikes2, stim_times, spike_times,
        window=window, delay=np.zeros(n_cells)
    )
    STA2, best_lags2 = keep_best_STA_lag(STA2_)

    print('Checking 2D overlap between two halves...')
    split_corr = np.zeros(n_cells)

    for c in range(n_cells):
        split_corr[c] = jaccard_topk(STA1[c, :], STA2[c, :])

    return STA, STA1, STA2, split_corr, best_lags
