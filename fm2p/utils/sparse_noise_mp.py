

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


_STIM_CHANNEL = 0

_g_shm = None
_g_flat = None
_g_col_means = None


def _worker_init(shm_name, shm_shape, col_means_bytes):
    global _g_shm, _g_flat, _g_col_means
    _g_shm  = SharedMemory(name=shm_name)
    _g_flat = np.ndarray(shm_shape, dtype=np.int8, buffer=_g_shm.buf)
    _g_col_means = np.frombuffer(col_means_bytes, dtype=np.float32).copy()


def _worker_sta_cell(args):

    (cell_idx, cell_spikes_f32, spike_times, stim_times,
     spike_split_ind, window) = args

    flat      = _g_flat
    col_means = _g_col_means

    n_stim, n_features = flat.shape
    eps = 1e-9

    interp_fn = interp1d(spike_times, cell_spikes_f32.astype(np.float32),
                         kind='linear', fill_value='extrapolate',
                         assume_sorted=True)
    rates_full = np.maximum(interp_fn(stim_times), 0).astype(np.float32)

    rates1 = rates_full.copy()
    rates1[spike_split_ind:] = 0.0
    rates2 = rates_full.copy()
    rates2[:spike_split_ind] = 0.0

    sta  = np.zeros((window + 1, n_features), dtype=np.float32)
    sta1 = np.zeros((window + 1, n_features), dtype=np.float32)
    sta2 = np.zeros((window + 1, n_features), dtype=np.float32)
    tot  = 0.0
    tot1 = 0.0
    tot2 = 0.0

    seg  = np.empty((window + 1, n_features), dtype=np.float32)
    tmp  = np.empty((window + 1, n_features), dtype=np.float32)

    for i in range(window, n_stim - 1):
        r  = rates_full[i]
        r1 = rates1[i]
        r2 = rates2[i]

        if r <= 0.0 and r1 <= 0.0 and r2 <= 0.0:
            continue

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

    def _best_lag(s):
        lag = int(np.argmax(np.max(np.abs(s), axis=1)))
        return s[lag].copy()

    return cell_idx, _best_lag(sta), _best_lag(sta1), _best_lag(sta2)


def _build_flat_signed_shm(stimpath, batch_frames=64):

    stim_mmap = np.load(stimpath, mmap_mode='r')
    has_channel = stim_mmap.ndim == 4
    n_stim = stim_mmap.shape[0]
    H      = stim_mmap.shape[1]
    W      = stim_mmap.shape[2]
    n_features = H * W

    print(f'  -> Stimulus: {n_stim} frames × {H}×{W} = {n_features} features')
    print(f'     Shared memory for flat_signed (int8): '
          f'{n_stim * n_features / 1e9:.2f} GB')

    rng = np.random.default_rng(0)
    sample_idx = np.sort(rng.choice(n_stim, min(512, n_stim), replace=False))
    if has_channel:
        sample = stim_mmap[sample_idx, :, :, _STIM_CHANNEL]
    else:
        sample = stim_mmap[sample_idx, :, :]
    bg_est = float(np.median(sample))
    del sample
    print(f'  -> bg_est = {bg_est:.2f}')

    shm = SharedMemory(create=True, size=n_stim * n_features)
    flat_arr = np.ndarray((n_stim, n_features), dtype=np.int8, buffer=shm.buf)

    col_sums = np.zeros(n_features, dtype=np.float64)

    print('  -> Streaming stimulus -> shared memory (int8) ...')
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
            print(f'     {e}/{n_stim} frames written', end='\r')

    print()
    col_means = (col_sums / n_stim).astype(np.float32)
    del col_sums, stim_mmap

    return shm, flat_arr, col_means, n_stim, n_features


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

    spikes     = np.asarray(spikes,     dtype=np.float32)
    stim_times = np.asarray(stim_times, dtype=np.float64)
    spike_times= np.asarray(spike_times,dtype=np.float64)
    stim_times = stim_times - stim_times[0]

    n_cells = spikes.shape[0]

    if n_processes is None:
        n_processes = min(os.cpu_count(), n_cells)

    shm, flat_arr, col_means, n_stim, n_features = _build_flat_signed_shm(stimpath)
    col_means_bytes = col_means.tobytes()

    STA_out  = np.zeros((n_cells, n_features), dtype=np.float32)
    STA1_out = np.zeros((n_cells, n_features), dtype=np.float32)
    STA2_out = np.zeros((n_cells, n_features), dtype=np.float32)

    stimend_sec = stim_times[-1]
    spikeend = int(np.searchsorted(spike_times, stimend_sec, side='right'))
    spike_times_trim = spike_times[:spikeend]
    spikes_trim      = spikes[:, :spikeend]

    midpoint_time  = float(spike_times_trim[spike_times_trim.size // 2])
    stim_split_ind = int(np.searchsorted(stim_times, midpoint_time, side='right'))
    stim_split_ind = int(np.clip(stim_split_ind, 1, n_stim - 1))
    print(f'  -> Split at stim frame {stim_split_ind}/{n_stim} '
          f'(t={midpoint_time:.1f} s, '
          f'first={stim_split_ind/n_stim*100:.1f}% / '
          f'second={100 - stim_split_ind/n_stim*100:.1f}%)')

    print(f'  -> {n_cells} cells, {n_stim} stim frames, '
          f'{n_processes} workers, window={window}')

    tasks = [
        (c,
         spikes_trim[c],
         spike_times_trim,
         stim_times,
         stim_split_ind,
         window)
        for c in range(n_cells)
    ]

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
