# -*- coding: utf-8 -*-
"""
fm2p/utils/sparse_noise_mp.py

Multiprocessing version of sparse-noise STA computation using shared memory.

Loads the full stimulus into POSIX shared memory once, then dispatches per-cell
STA computation to a worker pool -- avoiding repeated stimulus pickling.

Functions / helpers
-------------------
_worker_init
    Pool initializer: attaches each worker to the shared memory segment.
_worker_sta_cell
    Worker function: computes full, first-half, and second-half STAs for one cell.
_build_flat_signed_shm
    Stream a stimulus NPY file into shared memory as a mean-centered int8 array.
keep_best_STA_lag
    Retain the single highest-magnitude temporal lag per cell.
compute_split_STAs_mp
    Top-level entry: run split-STA pipeline across all cells in parallel.


DMM, November 2025
"""

import os
import gc
import math
import numpy as np
from scipy.interpolate import interp1d
from multiprocessing.shared_memory import SharedMemory
import multiprocessing as mp
from tqdm import tqdm

from .time import find_closest_timestamp
from .sparse_noise import compute_calcium_sta_spatial

# Index of the stimulus channel to use when the stimulus array has a channel dim.
_STIM_CHANNEL = 0

# Module-level globals populated by _worker_init; shared across all pool workers.
_g_shm = None
_g_flat = None
_g_col_means = None


def _worker_init(shm_name, shm_shape, col_means_bytes):
    """ Attach this worker process to the shared stimulus memory segment.

    Parameters
    ----------
    shm_name : str
        POSIX shared memory block name.
    shm_shape : tuple
        (n_stim, n_features) shape of the flat stimulus array.
    col_means_bytes : bytes
        Per-feature column means (float32) serialised as raw bytes.
    """

    global _g_shm, _g_flat, _g_col_means
    _g_shm = SharedMemory(name=shm_name)
    _g_flat = np.ndarray(shm_shape, dtype=np.int8, buffer=_g_shm.buf)
    _g_col_means = np.frombuffer(col_means_bytes, dtype=np.float32).copy()


def _worker_sta_cell(args):
    """ Compute full, first-half, and second-half STAs for a single cell.

    Parameters
    ----------
    args : tuple
        (cell_idx, cell_spikes_f32, spike_times, stim_times, spike_split_ind, window)

    Returns
    -------
    cell_idx : int
    sta : np.ndarray, shape (n_features,)
    sta1 : np.ndarray -- first-half STA
    sta2 : np.ndarray -- second-half STA
    """

    (cell_idx, cell_spikes_f32, spike_times, stim_times,
     spike_split_ind, window) = args

    flat = _g_flat
    col_means = _g_col_means

    n_stim, n_features = flat.shape
    eps = 1e-9

    interp_fn = interp1d(spike_times, cell_spikes_f32.astype(np.float32),
                         kind='linear', fill_value='extrapolate',
                         assume_sorted=True)
    rates_full = np.maximum(interp_fn(stim_times), 0).astype(np.float32)

    # Zero out the second/first half to create split-half spike traces.
    rates1 = rates_full.copy()
    rates1[spike_split_ind:] = 0.0
    rates2 = rates_full.copy()
    rates2[:spike_split_ind] = 0.0

    sta = np.zeros((window + 1, n_features), dtype=np.float32)
    sta1 = np.zeros((window + 1, n_features), dtype=np.float32)
    sta2 = np.zeros((window + 1, n_features), dtype=np.float32)
    tot = 0.0
    tot1 = 0.0
    tot2 = 0.0

    # Pre-allocate scratch arrays to avoid per-iteration allocations.
    seg = np.empty((window + 1, n_features), dtype=np.float32)
    tmp = np.empty((window + 1, n_features), dtype=np.float32)

    for i in range(window, n_stim - 1):
        r = rates_full[i]
        r1 = rates1[i]
        r2 = rates2[i]

        if r <= 0.0 and r1 <= 0.0 and r2 <= 0.0:
            continue

        np.subtract(flat[i - window: i + 1], col_means, out=seg)

        if r > 0.0:
            np.multiply(seg, r, out=tmp)
            sta += tmp
            tot += r
        if r1 > 0.0:
            np.multiply(seg, r1, out=tmp)
            sta1 += tmp
            tot1 += r1
        if r2 > 0.0:
            np.multiply(seg, r2, out=tmp)
            sta2 += tmp
            tot2 += r2

    sta /= (tot + eps)
    sta1 /= (tot1 + eps)
    sta2 /= (tot2 + eps)

    def _best_lag(s):
        lag = int(np.argmax(np.max(np.abs(s), axis=1)))
        return s[lag].copy()

    return cell_idx, _best_lag(sta), _best_lag(sta1), _best_lag(sta2)


def _build_flat_signed_shm(stimpath, batch_frames=64):
    """ Stream a stimulus NPY file into POSIX shared memory as a mean-centered int8 array.

    Reads the stimulus in batches to avoid loading it all at once, computes a
    background estimate from a random sample, and stores the signed (+1/-1/0)
    deviation from background.

    Parameters
    ----------
    stimpath : str
        Path to a .npy stimulus array (N, H, W) or (N, H, W, C).
    batch_frames : int
        Number of frames per streaming batch.

    Returns
    -------
    shm : SharedMemory
    flat_arr : np.ndarray -- view into shared memory, shape (n_stim, H*W), dtype int8
    col_means : np.ndarray -- per-feature column means, dtype float32
    n_stim : int
    n_features : int
    """

    stim_mmap = np.load(stimpath, mmap_mode='r')
    has_channel = stim_mmap.ndim == 4
    n_stim = stim_mmap.shape[0]
    H = stim_mmap.shape[1]
    W = stim_mmap.shape[2]
    n_features = H * W

    print('Stimulus: {} frames x {}x{} = {} features'.format(n_stim, H, W, n_features))
    print('Shared memory for flat_signed (int8): {:.2f} GB'.format(
        n_stim * n_features / 1e9))

    rng = np.random.default_rng(0)
    sample_idx = np.sort(rng.choice(n_stim, min(512, n_stim), replace=False))
    if has_channel:
        sample = stim_mmap[sample_idx, :, :, _STIM_CHANNEL]
    else:
        sample = stim_mmap[sample_idx, :, :]
    bg_est = float(np.median(sample))
    del sample
    print('bg_est = {:.2f}'.format(bg_est))

    shm = SharedMemory(create=True, size=n_stim * n_features)
    flat_arr = np.ndarray((n_stim, n_features), dtype=np.int8, buffer=shm.buf)

    col_sums = np.zeros(n_features, dtype=np.float64)

    print('Streaming stimulus to shared memory (int8)...')
    n_batches = math.ceil(n_stim / batch_frames)
    for b in range(n_batches):
        s = b * batch_frames
        e = min(s + batch_frames, n_stim)
        if has_channel:
            chunk = stim_mmap[s:e, :, :, _STIM_CHANNEL]
        else:
            chunk = stim_mmap[s:e, :, :]
        chunk = np.asarray(chunk)
        signed = (chunk > bg_est).astype(np.int8) - (chunk < bg_est).astype(np.int8)
        flat_arr[s:e, :] = signed.reshape(e - s, n_features)
        col_sums += signed.reshape(e - s, n_features).sum(axis=0)

        if (b + 1) % max(1, n_batches // 20) == 0 or b == n_batches - 1:
            print('{}/{} frames written'.format(e, n_stim), end='\r')

    print()
    col_means = (col_sums / n_stim).astype(np.float32)
    del col_sums, stim_mmap

    return shm, flat_arr, col_means, n_stim, n_features


def keep_best_STA_lag(STAs):
    """ Retain the single highest-magnitude temporal lag per cell.

    Parameters
    ----------
    STAs : np.ndarray, shape (N_cells, N_lags, N_features)

    Returns
    -------
    kept_STAs : np.ndarray, shape (N_cells, N_features)
    best_lags : np.ndarray, shape (N_cells,) int
    """

    n_cells = STAs.shape[0]
    best_lags = np.zeros(n_cells, dtype=int)
    kept_STAs = np.zeros((n_cells, STAs.shape[2]), dtype=STAs.dtype)
    for c in range(n_cells):
        lag_peaks = np.max(np.abs(STAs[c]), axis=1)
        best_lags[c] = int(np.nanargmax(lag_peaks))
        kept_STAs[c] = STAs[c, best_lags[c]]
    return kept_STAs, best_lags


def compute_split_STAs_mp(
        stimpath,
        spikes,
        stim_times,
        spike_times,
        window=13,
        n_processes=None
):
    """ Compute full and split-half STAs for all cells using shared-memory multiprocessing.

    Parameters
    ----------
    stimpath : str
        Path to the .npy stimulus file.
    spikes : np.ndarray, shape (N_cells, N_frames)
    stim_times : np.ndarray
    spike_times : np.ndarray
    window : int
        Temporal window (frames) before the spike included in the STA.
    n_processes : int or None
        Number of worker processes; defaults to min(cpu_count, n_cells).

    Returns
    -------
    STA_out : np.ndarray, shape (N_cells, H*W)
    STA1_out : np.ndarray -- first-half STAs
    STA2_out : np.ndarray -- second-half STAs
    split_corr : np.ndarray, shape (N_cells,) -- Jaccard index per cell
    best_lags : np.ndarray, shape (N_cells,) int
    """

    spikes = np.asarray(spikes, dtype=np.float32)
    stim_times = np.asarray(stim_times, dtype=np.float64)
    spike_times = np.asarray(spike_times, dtype=np.float64)
    stim_times = stim_times - stim_times[0]

    n_cells = spikes.shape[0]

    if n_processes is None:
        n_processes = min(os.cpu_count(), n_cells)

    shm, flat_arr, col_means, n_stim, n_features = _build_flat_signed_shm(stimpath)
    col_means_bytes = col_means.tobytes()

    STA_out = np.zeros((n_cells, n_features), dtype=np.float32)
    STA1_out = np.zeros((n_cells, n_features), dtype=np.float32)
    STA2_out = np.zeros((n_cells, n_features), dtype=np.float32)

    stimend_sec = stim_times[-1]
    spikeend = int(np.searchsorted(spike_times, stimend_sec, side='right'))
    spike_times_trim = spike_times[:spikeend]
    spikes_trim = spikes[:, :spikeend]

    midpoint_time = float(spike_times_trim[spike_times_trim.size // 2])
    stim_split_ind = int(np.searchsorted(stim_times, midpoint_time, side='right'))
    stim_split_ind = int(np.clip(stim_split_ind, 1, n_stim - 1))
    print('Split at stim frame {}/{} (t={:.1f}s, first={:.1f}% / second={:.1f}%)'.format(
        stim_split_ind, n_stim, midpoint_time,
        stim_split_ind / n_stim * 100,
        100 - stim_split_ind / n_stim * 100))

    print('{} cells, {} stim frames, {} workers, window={}'.format(
        n_cells, n_stim, n_processes, window))

    tasks = [
        (c,
         spikes_trim[c],
         spike_times_trim,
         stim_times,
         stim_split_ind,
         window)
        for c in range(n_cells)
    ]

    print('Computing STAs...')
    with mp.Pool(
        processes=n_processes,
        initializer=_worker_init,
        initargs=(shm.name, (n_stim, n_features), col_means_bytes)
    ) as pool:
        with tqdm(total=n_cells) as pbar:
            for cell_idx, sta, sta1, sta2 in pool.imap_unordered(
                    _worker_sta_cell, tasks, chunksize=1):
                STA_out[cell_idx] = sta
                STA1_out[cell_idx] = sta1
                STA2_out[cell_idx] = sta2
                pbar.update()

    shm.close()
    shm.unlink()
    del flat_arr
    gc.collect()

    from .sparse_noise import jaccard_topk
    split_corr = np.array([
        jaccard_topk(STA1_out[c], STA2_out[c])
        for c in range(n_cells)
    ])

    best_lags = np.zeros(n_cells, dtype=int)

    return STA_out, STA1_out, STA2_out, split_corr, best_lags
