
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from collections import deque
from tqdm import tqdm
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import numpy as np
from scipy.signal import correlate
from sklearn.linear_model import LinearRegression
from numpy.polynomial import Polynomial
from sklearn.preprocessing import PolynomialFeatures

import fm2p


def calc_combined_on_off_map(rf_on, rf_off, clim=None):
    """
    Overlay ON and OFF receptive fields in Ringach-style color coding.
    
    rf_on : 2D numpy array
        Response map for light stimuli (ON subfields).
    rf_off : 2D numpy array
        Response map for dark stimuli (OFF subfields).
    clim : float or None
        Color scale limit. If None, it uses max(|rf_on|, |rf_off|).
    """

    # normalize responses
    if clim is None:
        clim = max(np.max(np.abs(rf_on)), np.max(np.abs(rf_off)))
    
    norm = Normalize(vmin=0, vmax=clim, clip=True)

    # scale into [0,1]
    on_scaled = norm(np.maximum(rf_on, 0))   # only positive ON responses
    off_scaled = norm(np.maximum(rf_off, 0)) # only positive OFF responses

    # make RGB image: ON -> red channel, OFF -> blue channel
    rgb = np.zeros(rf_on.shape + (3,), dtype=float)
    rgb[...,0] = on_scaled     # red = ON
    rgb[...,2] = off_scaled    # blue = OFF

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


def shift_stimulus(stim, delay_frames, fill_value=0):
    # stim: (n_frames, n_pixels) or (n_frames, ...)
    if delay_frames == 0:
        return stim.copy()
    out = np.roll(stim, shift=delay_frames, axis=0)
    if delay_frames > 0:
        # shifted forward: first `delay_frames` rows are invalid (came from end)
        out[:delay_frames, ...] = fill_value
    else:
        # shifted backward: last `-delay_frames` rows are invalid (came from start)
        out[delay_frames:, ...] = fill_value  # delay_frames negative -> slices correct
    return out


def measure_sparse_noise_receptive_fields(cfg, data, ISI=False, use_lags=False):

    print('  -> Loading data.')

    if 'sparse_noise_stim_path' not in cfg.keys():
        stim_path = 'T:/dylan/sparse_noise_sequence_v7.npy'
    else:
        stim_path = cfg['sparse_noise_stim_path']
    stimarr = np.load(stim_path)[:,:,:,0] # drop color channel
    n_stim_frames = np.size(stimarr, 0)

    stim_f = stimarr.astype(float)
    # make sure it's scaled to 0:255
    if stim_f.max() <= 1.0:
        stim_f = stim_f * 255.0

    twopT = data['twopT']

    bg_est = np.median(stim_f)
    white_mask = (stim_f > bg_est)
    black_mask = (stim_f < bg_est)
    signed_stim = (white_mask.astype(np.int16) - black_mask.astype(np.int16))

    if ISI:
        stimT = np.arange(0, n_stim_frames, 1)
        isiT = np.arange(0.5, n_stim_frames, 1)
    else:
        stimT = data['stimT'] - data['stimT'][0]

    if use_lags:
        lags = np.arange(-5,5,1)

    norm_spikes = data['s2p_spks'].copy()[:10,:] # do just a subset of cells

    summed_stim_spikes = np.zeros([
        np.size(norm_spikes, 0),
        np.size(stimT)
    ]) * np.nan

    if ISI:
        summed_isi_spikes = np.zeros([
            np.size(norm_spikes, 0),
            np.size(stimT)
        ]) * np.nan

    if ISI:

        print('  -> Summing spikes during stimulus and ISI periods.')
        for c in tqdm(range(np.size(norm_spikes,0))):
            for i,t in enumerate(stimT[:-1]): # in sec
                start_win, _ = fm2p.find_closest_timestamp(twopT, t)
                end_win, _ = fm2p.find_closest_timestamp(twopT, isiT[i])
                next_win, _ = fm2p.find_closest_timestamp(twopT, stimT[i+1])
                summed_stim_spikes[c,i] = np.nanmean(norm_spikes[c, start_win:end_win])
                summed_isi_spikes[c,i] = np.nanmean(norm_spikes[c, end_win:next_win])

    else:

        print('  -> Summing spikes during stimulus (no ISI)')
        for c in tqdm(range(np.size(norm_spikes,0))):
            for i,t in enumerate(stimT[:-1]): # in sec
                start_win, _ = fm2p.find_closest_timestamp(twopT, t)
                next_win, _ = fm2p.find_closest_timestamp(twopT, stimT[i+1])
                summed_stim_spikes[c,i] = np.nanmean(norm_spikes[c, start_win:next_win])

    nFrames, stimY, stimX = np.shape(stimarr)

    # Flatten: shape (nFrames, nPixels)
    flat_signed = np.reshape(signed_stim, [nFrames, stimY*stimX])

    # Subtract pixel-wise time mean (center each pixel across frames)
    flat_signed = flat_signed - np.mean(flat_signed, axis=0, keepdims=True)


    print('  -> Estimating 2P vs stim misalignment.')

    stim_drive = np.std(flat_signed, axis=1)
    pop_resp = np.nansum(data.get('s2p_spks', np.zeros((1, twopT.shape[0]))), axis=0)

    delay_frames = find_delay_frames(stim_drive, pop_resp)
    print('Using {} as frame delay.'.format(delay_frames))

    stim_shifted = shift_stimulus(flat_signed, delay_frames)

    if use_lags:
        sta = np.zeros([
            np.size(norm_spikes, 0),
            len(lags),
            2,
            stimY,
            stimX
        ])

        rgb_maps = np.zeros([
            np.size(norm_spikes, 0),
            len(lags),
            stimY,
            stimX,
            3      # color channels
        ])

    else:
        sta = np.zeros([
            np.size(norm_spikes, 0),
            2,
            stimY,
            stimX
        ])

        rgb_maps = np.zeros([
            np.size(norm_spikes, 0),
            stimY,
            stimX,
            3      # color channels
        ])

    print('  -> Calculating spike-triggered averages (slow).')

    if not use_lags:
        for c in tqdm(range(np.size(norm_spikes, 0))):

            sp = summed_stim_spikes[c,:].copy()[:, np.newaxis]
            sp[np.isnan(sp)] = 0
            total_sp = np.sum(sp)
            if total_sp == 0:
                signed_sta = np.zeros((stimY*stimX, 1), dtype=float)
            else:
                # sta from the 0-centered signed stimulus
                signed_sta = (stim_shifted @ sp) / (total_sp + 1e-12)

            signed_sta_2d = np.reshape(signed_sta, [stimY, stimX])

            # split into on/off
            light_sta = np.maximum(signed_sta_2d, 0.)
            dark_sta = np.maximum(-signed_sta_2d, 0.)

            sta[c,0,:,:] = light_sta
            sta[c,1,:,:] = dark_sta

            rgb_maps[c,:,:,:] = calc_combined_on_off_map(light_sta, dark_sta)

    elif use_lags:
        for c in tqdm(range(np.size(norm_spikes, 0))):
            for l_i, lag in enumerate(lags):

                sp = summed_stim_spikes[c,:].copy()[:, np.newaxis]
                sp[np.isnan(sp)] = 0

                total_sp = np.sum(sp)
                if total_sp == 0:
                    signed_sta = np.zeros((stimY*stimX, 1), dtype=float)
                else:
                    rolled = shift_stimulus(stim_shifted, lag)
                    signed_sta = (rolled.T @ sp) / (total_sp + 1e-12)

                signed_sta_2d = np.reshape(signed_sta, [stimY, stimX])

                light_sta = np.maximum(signed_sta_2d, 0.0)
                dark_sta = np.maximum(-signed_sta_2d, 0.0)

                sta[c,l_i,0,:,:] = light_sta
                sta[c,l_i,1,:,:] = dark_sta

                rgb_maps[c,l_i,:,:,:] = calc_combined_on_off_map(light_sta, dark_sta)

    dict_out = {
        'STAs': sta,
        'rgb_maps': rgb_maps
    }

    return dict_out


if __name__ == '__main__':

    cfg_path = r'T:\dylan\251015_DMM_DMM056_sparsenoise\config.yaml'
    data_path = r'T:\dylan\251015_DMM_DMM056_sparsenoise\sn1\sn1_preproc.h5'

    # cfg_path = fm2p.select_file(
    #     'Select config.yaml file.',
    #     filetypes=[('YAML','.yaml'),]
    # )
    cfg = fm2p.read_yaml(cfg_path)
    # data_path = fm2p.select_file(
    #     'Select preprocessed HDF file.',
    #     filetypes=[('HDF','.h5'),]
    # )
    data = fm2p.read_h5(data_path)

    dict_out = fm2p.measure_sparse_noise_receptive_fields(
        cfg,
        data,
        use_lags=True
    )

    savepath = os.path.join(os.path.split(data_path)[0], 'sparse_noise_lags_n5_to_p10_arangeStimTime.h5')
    fm2p.write_h5(savepath, dict_out)

