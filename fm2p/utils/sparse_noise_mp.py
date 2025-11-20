
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.signal import correlate
import gc
from multiprocessing import Pool, cpu_count, shared_memory

import fm2p


def init_worker(shared_specs, params):
    """ Initialize workers, attach to shared memory buffers & stores parameter dict.
    """
    global _shared_arrays, _params
    _shared_arrays = {}
    _params = params

    # attach each shared array
    for name, (shm_name, shape, dtype) in shared_specs.items():
        shm = shared_memory.SharedMemory(name=shm_name)
        _shared_arrays[name] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)


def worker_compute_sta(cell_idx):
    """ Compute STA for one cell using shared-memory arrays.
    """

    # unpack
    flat_signed = _shared_arrays["flat_signed"]
    spikes = _shared_arrays["spikes"]
    spike_times = _shared_arrays["spike_times"]
    stim_times = _shared_arrays["stim_times"]

    window = _params["window"]
    n_stim = _params["n_stim"]
    n_features = _params["n_features"]
    shift_time_cellwise = _params["shift_time_cellwise"]
    dt = _params["dt"]
    delay = _params["delay"]
    eps = _params["eps"]

    cell_spikes = spikes[cell_idx]

    interp_fn = interp1d(
        spike_times,
        cell_spikes,
        kind="linear",
        fill_value="extrapolate",
        assume_sorted=True
    )

    if shift_time_cellwise:
        stim_times_shifted = stim_times + delay[cell_idx] * dt
    else:
        stim_times_shifted = stim_times

    spike_rate = interp_fn(stim_times_shifted)

    sta = np.zeros((window + 1, n_features), dtype=float)
    total_rate = 0.0

    for i, rate in enumerate(spike_rate):
        if rate <= 0 or i < window or i + window + 1 >= n_stim:
            continue
        stim_segment = flat_signed[i - window : i + 1]
        sta += rate * stim_segment
        total_rate += rate

    sta /= (total_rate + eps)

    return (cell_idx, sta)


def parallel_compute_sta_shared(
    flat_signed,
    spikes,
    spike_times,
    stim_times,
    dt,
    delay,
    shift_time_cellwise,
    window,
    eps=1e-12,
    n_workers=None
):

    n_cells = spikes.shape[0]
    n_stim, n_features = flat_signed.shape
    if n_workers is None:
        n_workers = max(1, cpu_count()-1)

    # Allocate arr into shared memory
    shared_specs = {}
    shm_blocks = []

    def register_shared(name, arr):
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        shm_arr[:] = arr[:]  # copy data
        shared_specs[name] = (shm.name, arr.shape, arr.dtype)
        shm_blocks.append(shm)

    register_shared("flat_signed", flat_signed)
    register_shared("spikes", spikes)
    register_shared("spike_times", spike_times)
    register_shared("stim_times", stim_times)

    # parameters passed to workers
    params = dict(
        window=window,
        n_stim=n_stim,
        n_features=n_features,
        shift_time_cellwise=shift_time_cellwise,
        dt=dt,
        delay=delay,
        eps=eps,
    )

    sta_all = np.zeros((n_cells, window + 1, n_features), dtype=float)

    try:
        chunksize = max(1, n_cells // (4 * n_workers))

        with Pool(
            processes=n_workers,
            initializer=init_worker,
            initargs=(shared_specs, params)
        ) as pool:

            results = pool.imap_unordered(
                worker_compute_sta, range(n_cells), chunksize=chunksize
            )

            for idx, sta in tqdm(results, total=n_cells, desc="STA (parallel)"):
                sta_all[idx] = sta

        del results

    finally:
        # cleanup shared memory
        for shm in shm_blocks:
            shm.close()
            shm.unlink()
        gc.collect()

    return sta_all


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


def compute_STA_parallel(preproc_path=None, stimpath=None):

    window = 13
    # max_lag_frames = 80
    skip_trim = False
    window = 13

    _shared_arrays = {}
    _params = {}

    if preproc_path is None:
        preproc_path = fm2p.select_file(
            'Select preprocessed HDF file.',
            filetypes=[('HDF','.h5'),]
        )

    if stimpath is None:
        stimpath = r'T:\goard_lab\sparse_noise_stimuli\sparse_noise_sequence_v7.npy'

    print('  -> Loading preprocessed data.')
    data = fm2p.read_h5(preproc_path)

    print('  -> Loading stimulus.')
    stimulus = np.load(stimpath)[:,:,:,0]
    spikes = data['s2p_spks']
    stimT = data['stimT']
    stimT = stimT - stimT[0]
    twopT = data['twopT']

    if stimulus.max() <= 1.0:
        stimulus = stimulus * 255.

    n_cells = np.size(spikes, 0)

    spike_split_ind = np.size(twopT) // 2
    spikes1 = spikes.copy()
    spikes2 = spikes.copy()
    spikes1[:, :spike_split_ind] = 0.
    spikes2[:, spike_split_ind:] = 0.
    
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

    # stim_mean_trace = np.mean(stimulus, axis=(1,2))

    bg_est = np.median(stimulus)
    white_mask = (stimulus > bg_est)
    black_mask = (stimulus < bg_est)
    signed_stim = (white_mask.astype(np.int16) - black_mask.astype(np.int16))

    flat_signed = np.reshape(signed_stim, [nFrames, stimY*stimX])
    flat_signed = flat_signed - np.mean(flat_signed, axis=0, keepdims=True)

    # n_stim, n_features = flat_signed.shape
    n_cells, n_spike_samples = spikes.shape

    if n_spike_samples != len(spike_times):
        raise ValueError(f"spikes.shape[1] ({n_spike_samples}) != len(spike_times) ({len(spike_times)})")

    # pop_trace = np.mean(spikes, axis=0)

    dt = np.median(np.diff(stim_times))

    # sta_all = np.zeros((n_cells, window + 1, n_features))
    # eps = 1e-9

    delay = np.zeros(n_cells)

    print('  -> Computing full sparse noise STAs.')
    STA_ = parallel_compute_sta_shared(
        flat_signed,
        spikes,
        spike_times,
        stim_times,
        dt,
        delay,
        shift_time_cellwise=True,
        window=window
    )
    STA, best_lags = keep_best_STA_lag(STA_)

    print('  -> Computing sparse noise STAs for first half of recording.')
    STA1_ = parallel_compute_sta_shared(
        flat_signed,
        spikes1,
        spike_times,
        stim_times,
        dt,
        delay,
        shift_time_cellwise=True,
        window=window
    )
    STA1, best_lags1 = keep_best_STA_lag(STA1_)

    print('  -> Computing sparse  noise STAs for second half of recording.')
    STA2_ = parallel_compute_sta_shared(
        flat_signed,
        spikes2,
        spike_times,
        stim_times,
        dt,
        delay,
        shift_time_cellwise=True,
        window=window
    )
    STA2, best_lags2 = keep_best_STA_lag(STA2_)

    print('  -> Checking 2D correlation between two halves.')
    split_corr = np.zeros(n_cells)

    for c in range(n_cells):
        A = fm2p.convolve2d(STA1[c].reshape(768,1360), np.ones([50,50]))
        B = fm2p.convolve2d(STA2[c].reshape(768,1360), np.ones([50,50]))
        A[(A < np.nanpercentile(np.abs(A),98)) & (A > -np.nanpercentile(np.abs(A),2))] = 1e-9
        B[(B < np.nanpercentile(np.abs(B),98)) & (B > -np.nanpercentile(np.abs(B),2))] = 1e-9

        split_corr[c] = fm2p.corr2_coeff(A, B)

    dict_out = {
        'STA': STA,
        'STA1': STA1,
        'STA2': STA2,
        'lags': best_lags,
        'corr': split_corr
    }

    savepath = os.path.join(os.path.split(preproc_path)[0], 'sparse_noise.h5')
    print('  -> Writing {}'.format(savepath))
    fm2p.write_h5(savepath, dict_out)


if __name__ == "__main__":

    compute_STA_parallel(
        r'Y:\Mini2P_V1PPC_cohort02_processed\251031_DMM_DMM061_pos19\sn1\sn1_preproc.h5'
    )

