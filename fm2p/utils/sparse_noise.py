
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.signal import correlate
import gc

import fm2p
import imgtools


def find_delay_frames(stim_s, pop_s, max_lag=80):

    stim_s = np.asarray(stim_s).ravel()
    pop_s  = np.asarray(pop_s).ravel()

    stim_s = (stim_s - np.mean(stim_s)) / np.std(stim_s)
    pop_s = (pop_s - np.mean(pop_s)) / np.std(pop_s)

    corr = correlate(stim_s, pop_s, mode='full')
    lags = np.arange(-len(stim_s)+1, len(pop_s))
    mask = (lags >= -max_lag) & (lags <= max_lag)
    lag = lags[mask][np.argmax(corr[mask])]

    return lag


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
    
    stimulus = np.asarray(stimulus)
    spikes = np.asarray(spikes)
    stim_times = np.asarray(stim_times)
    spike_times = np.asarray(spike_times)
    
    if not skip_trim:
        # trim off extra frames at end of 2P data
        stimend = np.size(stimulus,0)/2
        spikeend, _ = fm2p.find_closest_timestamp(spike_times, stimend)
        spikes = spikes[:,:spikeend]
        spike_times = spike_times[:spikeend]

    nFrames, stimY, stimX = np.shape(stimulus)

    stim_mean_trace = np.mean(stimulus, axis=(1,2))

    bg_est = np.median(stimulus)
    white_mask = (stimulus > bg_est)
    black_mask = (stimulus < bg_est)
    signed_stim = (white_mask.astype(np.int16) - black_mask.astype(np.int16))

    flat_signed = np.reshape(signed_stim, [nFrames, stimY*stimX])
    flat_signed = flat_signed - np.mean(flat_signed, axis=0, keepdims=True)

    n_stim, n_features = flat_signed.shape
    n_cells, n_spike_samples = spikes.shape

    if n_spike_samples != len(spike_times):
        raise ValueError(f"spikes.shape[1] ({n_spike_samples}) != len(spike_times) ({len(spike_times)})")

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
        cell_spikes = spikes[cell_idx,:]

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

            stim_segment = flat_signed[i - window : i+1]
 
            sta += rate * stim_segment
            total_rate += rate
                
        sta /= (total_rate + eps)
        sta_all[cell_idx] = sta

    lag_axis = np.arange(-window, window + 1)

    return sta_all, lag_axis, est_delay_frames


def keep_best_STA_lag(STAs):
    n_cells = np.size(STAs, 0)
    best_lags = np.zeros(n_cells)
    kept_STAs = np.zeros([
        n_cells,
        np.size(STAs,2)
    ])
    for c in range(n_cells):
        lagmax = np.zeros(np.size(STAs, 1)) * np.nan
        for l in range(np.size(STAs, 1)):
            lagmax[l] = np.nanmax(np.abs(STAs[c,l,:]))
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

    stimulus = np.asarray(stimulus)
    spikes = np.asarray(spikes)
    stim_times = np.asarray(stim_times)
    spike_times = np.asarray(spike_times)

    nCells  = spikes.shape[0]

    # split point
    spike_split_ind = np.size(spike_times) // 2
    # first_split_end = (np.size(spike_times) // 4) * 3
    # second_split_start = (np.size(spike_times) // 4)
    # instead of dropping both stimulus and spike data, keep it simpler and
    # zero-out the spikes during each half of the stimulus so that only half
    # of the stimulus contributes to the STA
    spikes1 = spikes.copy()
    spikes2 = spikes.copy()
    spikes1[:, :spike_split_ind] = 0.
    spikes2[:, spike_split_ind:] = 0.

    print('  -> Computing full sparse noise STAs.')
    STA_, lag_axis, delay = fm2p.compute_calcium_sta_spatial(
        stimulus,
        spikes,
        stim_times,
        spike_times,
        window=window,
        delay=[0,0]
    )
    STA, best_lags = keep_best_STA_lag(STA_)

    print('  -> Computing sparse noise STAs for first half of recording.')
    STA1_, lag_axis1, delay1 = fm2p.compute_calcium_sta_spatial(
        stimulus,
        spikes1,
        stim_times,
        spike_times,
        window=window,
        delay=[0,0]
    )
    STA1, best_lags1 = keep_best_STA_lag(STA1_)

    print('  -> Computing sparse  noise STAs for second half of recording.')
    STA2_, lag_axis2, delay2 = fm2p.compute_calcium_sta_spatial(
        stimulus,
        spikes2,
        stim_times,
        spike_times,
        window=window,
        delay=[0,0]
    )
    STA2, best_lags2 = keep_best_STA_lag(STA2_)

    # correlation across halves
    print('  -> Checking 2D correlation between two halves.')
    split_corr = np.zeros(nCells)

    for c in range(nCells):
        A = fm2p.convolve2d(STA1[c].reshape(768,1360), np.ones([50,50]))
        B = fm2p.convolve2d(STA2[c].reshape(768,1360), np.ones([50,50]))
        A[(A < np.nanpercentile(np.abs(A),98)) & (A > -np.nanpercentile(np.abs(A),2))] = 1e-9
        B[(B < np.nanpercentile(np.abs(B),98)) & (B > -np.nanpercentile(np.abs(B),2))] = 1e-9

        split_corr[c] = fm2p.corr2_coeff(A, B)

    return STA, STA1, STA2, split_corr, best_lags

