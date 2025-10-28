
import os
import numpy as np
from tqdm import tqdm
from matplotlib.colors import Normalize
import numpy as np
from scipy.signal import correlate

import fm2p


def calc_combined_on_off_map(rf_on, rf_off, clim=None):
    if clim is None:
        clim = max(np.max(np.abs(rf_on)), np.max(np.abs(rf_off)))
    norm = Normalize(vmin=0, vmax=clim, clip=True)
    # scale to [0,1]
    on_scaled = norm(np.maximum(rf_on, 0))
    off_scaled = norm(np.maximum(rf_off, 0))
    rgb = np.zeros(rf_on.shape + (3,), dtype=float)
    rgb[...,0] = on_scaled
    rgb[...,2] = off_scaled

    return rgb


def find_delay_frames(stim_s, pop_s, max_lag=80):
    stim_s = (stim_s - np.mean(stim_s)) / np.std(stim_s)
    pop_s = (pop_s - np.mean(pop_s)) / np.std(pop_s)
    corr = correlate(pop_s, stim_s, mode='full')
    lags = np.arange(-len(stim_s)+1, len(pop_s))
    # restrict search window
    mask = (lags >= -max_lag) & (lags <= max_lag)
    lag = lags[mask][np.argmax(corr[mask])]
    return lag


def compute_calcium_sta_spatial(stimulus, spikes, stim_times, spike_times,
                    window=10, delay=None, separate_light_dark=True,
                    auto_delay=True, max_lag_frames=80):
    """
    Compute spike-triggered averages (STAs) for multiple cells from calcium imaging data.
    """

    stimulus = np.asarray(stimulus)
    spikes = np.asarray(spikes)
    stim_times = np.asarray(stim_times)
    spike_times = np.asarray(spike_times)

    n_stim, n_features = stimulus.shape
    n_cells, n_spike_samples = spikes.shape

    # chek time alignment
    if n_spike_samples != len(spike_times):
        raise ValueError(f"spikes.shape[1] ({n_spike_samples}) != len(spike_times) ({len(spike_times)})")

    # estimate delay between twop start and stimulus start; the stimulus starts
    # 20-40 frames after the twop has already been going. can't seem to change
    # that at the acquisition point, so this is the posthoc solution
    est_delay_frames = 0
    if auto_delay:
        stim_drive = np.std(stimulus, axis=1)
        pop_resp = np.sum(spikes, axis=0)
        est_delay_frames = find_delay_frames(stim_drive, pop_resp, max_lag=max_lag_frames)
        # delay = est_delay_frames * np.nanmedian(np.diff(stim_times))

    # instead of rolling stimulus data, just crop off beginning of twop recording
    spikes = spikes[:, est_delay_frames:]
    spike_times = spike_times[est_delay_frames:]

    bin_edges = np.concatenate([
        stim_times,
        [stim_times[-1] + np.median(np.diff(stim_times))],
    ])

    sta_light_all = np.zeros((n_cells, 2 * window + 1, n_features))
    sta_dark_all = np.zeros((n_cells, 2 * window + 1, n_features))
    mean_stim_intensity = np.mean(stimulus, axis=1)
    eps = 1e-9

    print('  -> Computing STA for cells (slow!).')
    for cell_idx in tqdm(range(n_cells)):
        cell_spikes = spikes[cell_idx]

        # bin cell's spikes into stimulus frame bins
        spike_sum, _ = np.histogram(spike_times, bins=bin_edges, weights=cell_spikes)
        counts, _ = np.histogram(spike_times, bins=bin_edges)
        counts[counts == 0] = 1
        spike_rate_per_frame = spike_sum / counts

        sta_light = np.zeros((2 * window + 1, n_features))
        sta_dark = np.zeros((2 * window + 1, n_features))
        total_light_rate = 0.0
        total_dark_rate = 0.0

        for i, rate in enumerate(spike_rate_per_frame):
            if rate <= 0 or i < window or i + window >= n_stim:
                continue

            stim_segment = stimulus[i - window : i + window + 1]

            if separate_light_dark:
                if mean_stim_intensity[i] > 0:
                    sta_light += rate * stim_segment
                    total_light_rate += rate
                else:
                    sta_dark += rate * stim_segment
                    total_dark_rate += rate
            else:
                sta_light += rate * stim_segment
                total_light_rate += rate

        if separate_light_dark:
            sta_light /= (total_light_rate + eps)
            sta_dark /= (total_dark_rate + eps)
        else:
            sta_light /= (total_light_rate + eps)
            sta_dark = None # is this a bad idea?

        sta_light_all[cell_idx] = sta_light
        if separate_light_dark:
            sta_dark_all[cell_idx] = sta_dark

    lag_axis = np.arange(-window, window + 1)

    return sta_light_all, sta_dark_all, lag_axis, est_delay_frames


if __name__ == '__main__':

    cfg_path = fm2p.select_file(
        'Select config.yaml file.',
        filetypes=[('YAML','.yaml'),]
    )
    cfg = fm2p.read_yaml(cfg_path)
    data_path = fm2p.select_file(
        'Select preprocessed HDF file.',
        filetypes=[('HDF','.h5'),]
    )
    data = fm2p.read_h5(data_path)

    sta_light_all, sta_dark_all, lag_axis, est_delay_frames = fm2p.measure_sparse_noise_receptive_fields(
        cfg,
        data,
        use_lags=True
    )

    dict_out = {
        'lightSTA': sta_light_all,
        'darkSTA': sta_dark_all,
        'lags': lag_axis,
        'delay': est_delay_frames
    }

    savepath = os.path.join(os.path.split(data_path)[0], 'sparse_noise_receptive_fields.h5')
    fm2p.write_h5(savepath, dict_out)

