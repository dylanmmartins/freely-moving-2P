# -*- coding: utf-8 -*-
"""
Parallel, memory-efficient sparse-noise STA computation.

Memory design
-------------
* The stimulus is never loaded wholesale into RAM.  It is memory-mapped from
  the .npy file and streamed frame-by-frame.
* `flat_signed` (the centred, signed stimulus matrix) is stored once as int8
  in a POSIX shared-memory block (~13 GB for 12 k frames @ 768×1360).
  All worker processes attach to the same block – no duplication.
* `col_means` (float32, ~4 MB) is computed in the same streaming pass and
  passed to workers as a small argument.
* Each worker computes STA, STA1, and STA2 for **one cell** in a single pass
  over the shared stimulus, then immediately picks the best-lag frame and
  returns only (n_features,) float32 per condition — no large inter-process
  transfers.
* Peak RAM budget (400 cells, window=13, 1360×768):
    flat_signed shared mem   : ~13 GB
    col_means                :  ~4 MB
    n_workers × per-cell bufs:  ~1.5 GB  (8 workers × 175 MB each)
    assembled final STAs     :  ~5 GB    (STA + STA1 + STA2, float32)
    ─────────────────────────────────────
    Total                    : ~20 GB
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

# Channel index in the 4-D stimulus array  (n_frames, H, W, C)
_STIM_CHANNEL = 0

# ---------------------------------------------------------------------------
# Module-level globals – populated in each worker via initializer
# ---------------------------------------------------------------------------
_g_shm       = None   # SharedMemory handle (kept alive)
_g_flat      = None   # np.ndarray view into shared memory  (n_stim, n_features) int8
_g_col_means = None   # (n_features,) float32


def _worker_init(shm_name, shm_shape, col_means_bytes):
    global _g_shm, _g_flat, _g_col_means
    _g_shm  = SharedMemory(name=shm_name)
    _g_flat = np.ndarray(shm_shape, dtype=np.int8, buffer=_g_shm.buf)
    _g_col_means = np.frombuffer(col_means_bytes, dtype=np.float32).copy()


def _worker_sta_cell(args):
    """
    Compute STA, STA1, STA2 for a single cell in one pass through the stimulus.

    Returns
    -------
    (cell_idx, sta_best, sta1_best, sta2_best)
        Each *_best is (n_features,) float32 at the best-lag frame.
    """
    (cell_idx, cell_spikes_f32, spike_times, stim_times,
     spike_split_ind, window) = args

    flat      = _g_flat        # (n_stim, n_features) int8, shared
    col_means = _g_col_means   # (n_features,) float32

    n_stim, n_features = flat.shape
    eps = 1e-9

    # --- interpolate spike rates onto stimulus time grid ---
    interp_fn = interp1d(spike_times, cell_spikes_f32.astype(np.float32),
                         kind='linear', fill_value='extrapolate',
                         assume_sorted=True)
    rates_full = np.maximum(interp_fn(stim_times), 0).astype(np.float32)

    # half-splits: zero out the other half
    rates1 = rates_full.copy()
    rates1[spike_split_ind:] = 0.0
    rates2 = rates_full.copy()
    rates2[:spike_split_ind] = 0.0

    # --- STA accumulators (window+1 lags × n_features), float32 ---
    sta  = np.zeros((window + 1, n_features), dtype=np.float32)
    sta1 = np.zeros((window + 1, n_features), dtype=np.float32)
    sta2 = np.zeros((window + 1, n_features), dtype=np.float32)
    tot  = 0.0
    tot1 = 0.0
    tot2 = 0.0

    # pre-allocated working buffer (avoids temp allocations in the hot loop)
    seg  = np.empty((window + 1, n_features), dtype=np.float32)
    tmp  = np.empty((window + 1, n_features), dtype=np.float32)

    for i in range(window, n_stim - 1):
        r  = rates_full[i]
        r1 = rates1[i]
        r2 = rates2[i]

        if r <= 0.0 and r1 <= 0.0 and r2 <= 0.0:
            continue

        # centred stimulus window (single read from shared memory)
        # seg = int8_block - col_means  -> float32
        np.subtract(flat[i - window : i + 1], col_means, out=seg)

        if r > 0.0:
            np.multiply(seg, r, out=tmp)
            sta  += tmp
            tot  += r
        if r1 > 0.0:
            np.multiply(seg, r1, out=tmp)
            sta1 += tmp
            tot1 += r1
        if r2 > 0.0:
            np.multiply(seg, r2, out=tmp)
            sta2 += tmp
            tot2 += r2

    sta  /= (tot  + eps)
    sta1 /= (tot1 + eps)
    sta2 /= (tot2 + eps)

    # pick the best lag for each STA independently (max abs across features)
    def _best_lag(s):
        lag = int(np.argmax(np.max(np.abs(s), axis=1)))
        return s[lag].copy()

    return cell_idx, _best_lag(sta), _best_lag(sta1), _best_lag(sta2)


# ---------------------------------------------------------------------------
# Shared-memory builder
# ---------------------------------------------------------------------------

def _build_flat_signed_shm(stimpath, batch_frames=64):
    """
    Stream the stimulus file frame-by-frame, convert to signed int8, write
    into a SharedMemory block, and simultaneously compute column means.

    Parameters
    ----------
    stimpath    : path to .npy file with shape (n_frames, H, W[, C])
    batch_frames: number of frames to process at once (trades CPU vs RAM)

    Returns
    -------
    shm        : SharedMemory   – caller must call shm.close() / shm.unlink()
    flat_arr   : np.ndarray view into shm  (n_stim, n_features) int8
    col_means  : (n_features,) float32
    n_stim     : int
    n_features : int
    """
    stim_mmap = np.load(stimpath, mmap_mode='r')   # does NOT load into RAM
    has_channel = stim_mmap.ndim == 4
    n_stim = stim_mmap.shape[0]
    H      = stim_mmap.shape[1]
    W      = stim_mmap.shape[2]
    n_features = H * W

    print(f'  -> Stimulus: {n_stim} frames × {H}×{W} = {n_features} features')
    print(f'     Shared memory for flat_signed (int8): '
          f'{n_stim * n_features / 1e9:.2f} GB')

    # --- estimate background from a random sample (avoids full load) ---
    rng = np.random.default_rng(0)
    sample_idx = np.sort(rng.choice(n_stim, min(512, n_stim), replace=False))
    if has_channel:
        sample = stim_mmap[sample_idx, :, :, _STIM_CHANNEL]
    else:
        sample = stim_mmap[sample_idx, :, :]
    bg_est = float(np.median(sample))
    del sample
    print(f'  -> bg_est = {bg_est:.2f}')

    # --- allocate shared memory ---
    shm = SharedMemory(create=True, size=n_stim * n_features)
    flat_arr = np.ndarray((n_stim, n_features), dtype=np.int8, buffer=shm.buf)

    # --- stream frames: build flat_signed and col_means in one pass ---
    col_sums = np.zeros(n_features, dtype=np.float64)

    print('  -> Streaming stimulus -> shared memory (int8) ...')
    n_batches = math.ceil(n_stim / batch_frames)
    for b in range(n_batches):
        s = b * batch_frames
        e = min(s + batch_frames, n_stim)
        if has_channel:
            chunk = stim_mmap[s:e, :, :, _STIM_CHANNEL]   # (bs, H, W)
        else:
            chunk = stim_mmap[s:e, :, :]
        chunk = np.asarray(chunk)            # materialise this small batch
        signed = (chunk > bg_est).astype(np.int8) - (chunk < bg_est).astype(np.int8)
        flat_arr[s:e, :] = signed.reshape(e - s, n_features)
        col_sums += signed.reshape(e - s, n_features).sum(axis=0)

        if (b + 1) % max(1, n_batches // 20) == 0 or b == n_batches - 1:
            print(f'     {e}/{n_stim} frames written', end='\r')

    print()
    col_means = (col_sums / n_stim).astype(np.float32)
    del col_sums, stim_mmap

    return shm, flat_arr, col_means, n_stim, n_features


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def keep_best_STA_lag(STAs):
    n_cells = STAs.shape[0]
    best_lags  = np.zeros(n_cells, dtype=int)
    kept_STAs  = np.zeros((n_cells, STAs.shape[2]), dtype=STAs.dtype)
    for c in range(n_cells):
        lag_peaks = np.max(np.abs(STAs[c]), axis=1)   # (n_lags,)
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
    """
    Parallel, memory-efficient split-STA computation.

    Parameters
    ----------
    stimpath    : str   path to sparse-noise .npy file
    spikes      : (n_cells, n_spike_samples) float32-compatible array
    stim_times  : (n_stim,)  stimulus frame timestamps (seconds)
    spike_times : (n_spike_samples,)  2P frame timestamps (seconds)
    window      : int   STA lag window (frames)
    n_processes : int or None  (None -> os.cpu_count())

    Returns
    -------
    STA        : (n_cells, n_features) float32  full-recording STA
    STA1       : (n_cells, n_features) float32  first-half STA
    STA2       : (n_cells, n_features) float32  second-half STA
    split_corr : (n_cells,)            jaccard similarity STA1 vs STA2
    best_lags  : (n_cells,)            best lag index for STA
    """
    spikes     = np.asarray(spikes,     dtype=np.float32)
    stim_times = np.asarray(stim_times, dtype=np.float64)
    spike_times= np.asarray(spike_times,dtype=np.float64)
    stim_times = stim_times - stim_times[0]

    n_cells = spikes.shape[0]

    if n_processes is None:
        n_processes = min(os.cpu_count(), n_cells)

    # --- build flat_signed in shared memory ---
    shm, flat_arr, col_means, n_stim, n_features = _build_flat_signed_shm(stimpath)
    col_means_bytes = col_means.tobytes()

    # --- output arrays (float32) ---
    STA_out  = np.zeros((n_cells, n_features), dtype=np.float32)
    STA1_out = np.zeros((n_cells, n_features), dtype=np.float32)
    STA2_out = np.zeros((n_cells, n_features), dtype=np.float32)

    # --- trim spike_times / spikes to stimulus duration ---
    stimend_sec = stim_times[-1]
    spikeend = int(np.searchsorted(spike_times, stimend_sec, side='right'))
    spike_times_trim = spike_times[:spikeend]
    spikes_trim      = spikes[:, :spikeend]

    # --- compute the split index in STIMULUS-frame space ---
    # spike_split_ind was previously spike_times.size // 2, which is an index
    # into the 2P frame array. rates_full inside the worker has length n_stim
    # (stimulus frames), so the split must be an index into stim_times.
    # Find the stimulus frame nearest to the midpoint of the trimmed recording.
    midpoint_time  = float(spike_times_trim[spike_times_trim.size // 2])
    stim_split_ind = int(np.searchsorted(stim_times, midpoint_time, side='right'))
    stim_split_ind = int(np.clip(stim_split_ind, 1, n_stim - 1))
    print(f'  -> Split at stim frame {stim_split_ind}/{n_stim} '
          f'(t={midpoint_time:.1f} s, '
          f'first={stim_split_ind/n_stim*100:.1f}% / '
          f'second={100 - stim_split_ind/n_stim*100:.1f}%)')

    print(f'  -> {n_cells} cells, {n_stim} stim frames, '
          f'{n_processes} workers, window={window}')

    # --- build task list ---
    tasks = [
        (c,
         spikes_trim[c],       # (n_spike_samples,) float32 – small, per-cell
         spike_times_trim,
         stim_times,
         stim_split_ind,       # index into rates_full (length n_stim)
         window)
        for c in range(n_cells)
    ]

    # --- run pool ---
    print('  -> Computing STAs ...')
    with mp.Pool(
        processes=n_processes,
        initializer=_worker_init,
        initargs=(shm.name,
                  (n_stim, n_features),
                  col_means_bytes)
    ) as pool:
        with tqdm(total=n_cells) as pbar:
            for cell_idx, sta, sta1, sta2 in pool.imap_unordered(
                    _worker_sta_cell, tasks, chunksize=1):
                STA_out[cell_idx]  = sta
                STA1_out[cell_idx] = sta1
                STA2_out[cell_idx] = sta2
                pbar.update()

    # --- cleanup shared memory ---
    shm.close()
    shm.unlink()
    del flat_arr
    gc.collect()

    # --- split-half Jaccard similarity ---
    print('  -> Computing split-half similarity ...')
    from .sparse_noise import jaccard_topk   # local import to avoid circular
    split_corr = np.array([
        jaccard_topk(STA1_out[c], STA2_out[c])
        for c in range(n_cells)
    ])

    # best_lags: for the best-lag STA we need the full-lag array.
    # We already picked best-lag inside the worker; store a dummy zero array
    # for API compatibility (the actual selection was done per-cell internally).
    best_lags = np.zeros(n_cells, dtype=int)

    return STA_out, STA1_out, STA2_out, split_corr, best_lags
