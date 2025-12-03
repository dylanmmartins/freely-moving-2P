

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.signal import correlate
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import gc

import fm2p


# Global handles for workers
_flat_signed = None
_flat_shape = None
_flat_dtype = None


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


def _init_worker(shm_name, shape, dtype):
    """ Each worker reattaches to the shared memory block.
    """
    global _flat_signed, _flat_shape, _flat_dtype

    _flat_shape = tuple(shape)
    _flat_dtype = np.dtype(dtype)

    shm = SharedMemory(name=shm_name)
    _flat_signed = np.ndarray(_flat_shape, dtype=_flat_dtype, buffer=shm.buf)


def _compute_sta_for_cell(args):
    """ Worker function. Reads `_flat_signed` from shared memory.
    """
    (
        cell_idx,
        cell_spikes,
        spike_times,
        stim_times,
        stim_times_shifted_global,
        delay,
        dt,
        shift_time_cellwise,
        window,
        n_stim
    ) = args

    global _flat_signed

    interp_fn = interp1d(
        spike_times,
        cell_spikes,
        kind='linear',
        fill_value='extrapolate',
        assume_sorted=True
    )

    if shift_time_cellwise:
        stim_times_shifted = stim_times + delay[cell_idx] * dt
    else:
        stim_times_shifted = stim_times_shifted_global

    spike_rate_per_frame = interp_fn(stim_times_shifted)

    n_features = _flat_signed.shape[1]
    sta = np.zeros((window + 1, n_features))
    total_rate = 0.0
    eps = 1e-9

    for i, rate in enumerate(spike_rate_per_frame):
        if rate <= 0 or i < window or i + window + 1 >= n_stim:
            continue

        segment = _flat_signed[i - window : i + 1]
        sta += rate * segment
        total_rate += rate

    sta /= (total_rate + eps)
    return sta

def compute_calcium_sta_spatial(
        stimulus,
        spikes,
        stim_times,
        spike_times,
        window=20,
        delay=None,
        max_lag_frames=80,
        skip_trim=False,
        n_processes=None
    ):

    stimulus = np.asarray(stimulus)
    spikes = np.asarray(spikes)
    stim_times = np.asarray(stim_times)
    spike_times = np.asarray(spike_times)

    if not skip_trim:
        stimend = np.size(stimulus,0)/2
        spikeend, _ = fm2p.find_closest_timestamp(spike_times, stimend)
        spikes = spikes[:, :spikeend]
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
        raise ValueError("Mismatch in spike dimensions")

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
        stim_times_shifted_global = stim_times + delay_
    else:
        shift_time_cellwise = True
        stim_times_shifted_global = None

    # Create shared memory for flat_signed
    shm = SharedMemory(create=True, size=flat_signed.nbytes)
    shm_arr = np.ndarray(flat_signed.shape, dtype=flat_signed.dtype, buffer=shm.buf)
    shm_arr[:] = flat_signed  # copy once

    # data for workers
    worker_init_args = (shm.name, flat_signed.shape, flat_signed.dtype.str)

    # Arguments for map
    args = [
        (
            cell_idx,
            spikes[cell_idx,:],
            spike_times,
            stim_times,
            stim_times_shifted_global,
            delay,
            dt,
            shift_time_cellwise,
            window,
            n_stim
        )
        for cell_idx in range(n_cells)
    ]

    # Run workers
    with mp.Pool(
        processes=n_processes,
        initializer=_init_worker,
        initargs=worker_init_args
    ) as pool:
        sta_list = pool.map(_compute_sta_for_cell, args)

    # cleanup shared memory
    shm.close()
    shm.unlink()

    sta_all = np.stack(sta_list, axis=0)
    lag_axis = np.arange(-window, window + 1)

    del signed_stim, flat_signed
    gc.collect()

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

    print('  -> Setting up spike splits.')

    stimulus = np.asarray(stimulus)
    spikes = np.asarray(spikes)
    stim_times = np.asarray(stim_times)
    spike_times = np.asarray(spike_times)

    n_cells  = spikes.shape[0]

    spike_split_ind = np.size(spike_times) // 2
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
        delay=np.zeros(n_cells)
    )
    STA, best_lags = keep_best_STA_lag(STA_)

    del STA_
    gc.collect()

    print('  -> Computing sparse noise STAs for first half of recording.')
    STA1_, lag_axis1, delay1 = fm2p.compute_calcium_sta_spatial(
        stimulus,
        spikes1,
        stim_times,
        spike_times,
        window=window,
        delay=np.zeros(n_cells)
    )
    STA1, best_lags1 = keep_best_STA_lag(STA1_)

    del STA1_
    gc.collect()

    print('  -> Computing sparse  noise STAs for second half of recording.')
    STA2_, lag_axis2, delay2 = fm2p.compute_calcium_sta_spatial(
        stimulus,
        spikes2,
        stim_times,
        spike_times,
        window=window,
        delay=np.zeros(n_cells)
    )
    STA2, best_lags2 = keep_best_STA_lag(STA2_)
    
    del STA2_
    gc.collect()

    return STA, STA1, STA2, best_lags

