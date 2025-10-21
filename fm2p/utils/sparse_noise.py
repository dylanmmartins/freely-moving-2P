
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

def correct_stim_timing(stimarr, data, savepath):
    # correct for stimulus timing with a drift and offset

    # need actual timestamps from scanimage, not the synthetic times
    twopT = fm2p.read_scanimage_time(r'T:\dylan\251008_DMM_DMM061_sparsenoise\sn1\file_00001.tif')

    dt = 0.500
    n_stim_frames= np.size(stimarr, 0)
    stimT = np.arange(0, n_stim_frames*dt, dt)

    cropind, _ = fm2p.find_closest_timestamp(twopT, stimT[-1])

    # compute candidate drives
    flat_stimarr = np.reshape(
        stimarr,
        [np.size(stimarr,0), np.size(stimarr, 1)*np.size(stimarr,2)]
    ).T

    # stim_frames: (n_pixels, T_stim)
    # mean_play = np.mean(flat_stimarr, axis=0)
    std_play = np.std(flat_stimarr, axis=0)

    # # temporal absolute diff
    # diff = np.zeros(flat_stimarr.shape[1])
    # diff[1:] = np.mean(np.abs(flat_stimarr[:,1:] - flat_stimarr[:,:-1]), axis=0)
    # # PC1 projection
    # # center
    # X = flat_stimarr.T - np.mean(flat_stimarr, axis=1)
    # # compute first left singular vector (cheap PCA for PC1)
    # try:
    #     u, s, vt = np.linalg.svd(X, full_matrices=False)
    #     pc1 = vt[0]  # principal component weights per pixel
    #     proj_pc1 = (pc1 @ flat_stimarr)  # shape (T_stim,)
    # except Exception:
    #     proj_pc1 = mean_play  # fallback

    # drives = dict(mean=mean_play, std=std_play, diff=diff, pc1=proj_pc1)

    sps = data['norm_spikes'].copy()

    # # pick a drive (or test all): e.g. 'std' or 'diff' (good when mean is constant)
    # drive_name = 'std'
    # stim_drive = drives[drive_name]
    stim_drive = std_play

    f = interp1d(stimT, stim_drive, bounds_error=False, fill_value='extrapolate')
    stim_on_2p = f(twopT[:cropind])
    stim_s = (stim_on_2p - np.nanmean(stim_on_2p)) / (np.nanstd(stim_on_2p) + 1e-12)
    stim_s = gaussian_filter1d(np.nan_to_num(stim_s), sigma=1)

    # cell population response
    pop = np.nansum(sps[:,:cropind], axis=0)
    pop_s = (pop - np.mean(pop)) / (np.std(pop) + 1e-12)
    pop_s = gaussian_filter1d(pop_s, sigma=1)

    # estimate best lag per segment
    seg_len_s = 60.*2   # length of each segment (in sec)
    step_s = seg_len_s  # non-overlapping; set smaller for overlap
    t0 = twopT[0]
    seg_centers = []
    lags_seconds = []
    maxlag_s = 120.0  # search window (in sec)
    maxlag_frames = int(np.ceil(maxlag_s / dt))

    i = 0
    while True:
        
        start = t0 + i*step_s
        stop = start + seg_len_s
        mask = (twopT[:cropind] >= start) & (twopT[:cropind] < stop)
        
        if mask.sum() < 10:
            break
        
        sd = stim_s[mask] - np.nanmean(stim_s[mask])
        rd = pop_s[mask] - np.nanmean(pop_s[mask])
        cc = correlate(sd, rd, mode='full')
        lags = np.arange(-len(sd)+1, len(sd))

        center = len(cc)//2
        low = max(0, center - maxlag_frames)
        high = min(len(cc), center + maxlag_frames + 1)
        sub = cc[low:high]
        sublags = lags[low:high]
        
        best_idx = np.argmax(sub)
        best_lag_frames = sublags[best_idx]
        best_lag_s = best_lag_frames * dt
        
        seg_centers.append((start + stop)/2.0)
        lags_seconds.append(best_lag_s)

        i += 1

    seg_centers = np.array(seg_centers)
    lags_seconds = np.array(lags_seconds)

    # lag(t) = m * t + b
    lr = LinearRegression()
    lr.fit(seg_centers.reshape(-1,1), lags_seconds)
    m = lr.coef_[0]
    b = lr.intercept_
    stim_times_corrected = stimT - (b + m * stimT)

    plt.figure(figsize=(3,2), dpi=300)
    plt.plot(seg_centers, lags_seconds, 'o', label='segment lag estimates')
    tt = np.linspace(seg_centers.min(), seg_centers.max(), 200)
    plt.plot(tt, lr.predict(tt.reshape(-1,1)), '-', label=f'fit: lag={b:.3f}+{m:.3e}*t')
    plt.xlabel('time (s)'); plt.ylabel('lag (s)')
    plt.legend(); plt.title('Per-segment lag and linear drift fit')
    plt.show()
    plt.savefig(os.path.join(savepath, 'sparse_noise_linear_timing_fit.png'))

    # cross-corr on whole recording after correction
    f2 = interp1d(stim_times_corrected, stim_drive, bounds_error=False, fill_value='extrapolate')
    stim_on_2p_corr = f2(twopT)
    stim_s_corr = (stim_on_2p_corr - np.nanmean(stim_on_2p_corr)) / (np.nanstd(stim_on_2p_corr) + 1e-12)
    stim_s_corr = gaussian_filter1d(np.nan_to_num(stim_s_corr), sigma=1)
    cc_corr = correlate(stim_s_corr - stim_s_corr.mean(), pop_s - pop_s.mean(), mode='full')
    lags_full = np.arange(-len(stim_s_corr)+1, len(stim_s_corr))

    degree = 5
    poly = Polynomial.fit(seg_centers, lags_seconds, deg=degree)
    lag_fit = poly(seg_centers)
    residuals = lags_seconds - lag_fit

    plt.figure(figsize=(3,2), dpi=300)
    plt.scatter(seg_centers, lags_seconds, label="segment estimates", color='C0')
    tt = np.linspace(seg_centers.min(), seg_centers.max(), 400)
    plt.plot(tt, poly(tt), 'r-', label=f'poly deg={degree}')
    plt.xlabel("time (s)")
    plt.ylabel("lag (s)")
    plt.legend()
    plt.title("Polynomial fit of lag vs time")
    plt.show()
    plt.savefig(os.path.join(savepath, 'sparse_noise_polynomial_timing_fit.png'))

    # should look like noise
    plt.figure(figsize=(3,2), dpi=300)
    plt.plot(seg_centers, residuals, 'o-')
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel("time (s)")
    plt.ylabel("residual lag (s)")
    plt.show()
    plt.savefig(os.path.join(savepath, 'sparse_noise_residuals.png'))

    lag_predicted = poly(stimT)
    stim_times_corrected = stimT.copy() - lag_predicted

    plt.figure(figsize=(3,2), dpi=300)
    plt.hist(np.diff(stim_times_corrected), bins=25)
    plt.show()
    plt.savefig(os.path.join(savepath, 'sparse_noise_corrected_time_diff.png'))

    return stim_times_corrected


def measure_sparse_noise_receptive_fields(cfg, data, ISI=False, use_lags=False):

    if 'sparse_noise_stim_path' not in cfg.keys():
        stim_path = 'T:/dylan/sparse_noise_sequence_v7.npy'
    else:
        stim_path = cfg['sparse_noise_stim_path']
    stimarr = np.load(stim_path)[:,:,:,0] # drop color channel
    n_stim_frames = np.size(stimarr, 0)

    # Build a signed stimulus: +1 for white, -1 for black, 0 for background/gray.
    stim_f = stimarr.astype(float)
    # If floats in 0..1, scale to 0..255
    if stim_f.max() <= 1.0:
        stim_f = stim_f * 255.0

    twopT = data['twopT']

    bg_est = np.median(stim_f)
    white_mask = (stim_f > bg_est)
    black_mask = (stim_f < bg_est)
    signed_stim = (white_mask.astype(np.int16) - black_mask.astype(np.int16))

    # stim will end after twop has already ended
    if ISI:
        stimT = np.arange(0, n_stim_frames, 1)
        isiT = np.arange(0.5, n_stim_frames, 1)
    else:
        # stimT = correct_stim_timing(stimarr, data)
        # TODO: make sure this reads in as the expected format
        stimT = data['stimT']

    if use_lags:
        lags = np.arange(0,-10,-1)

    norm_spikes = data['s2p_spks'].copy() # [:15,:] do just a subset of cells

    if not use_lags:
        norm_spikes = np.roll(norm_spikes, shift=2, axis=1)


    # find timing correction
    stimT = correct_stim_timing(stimarr, data, r'T:\dylan\251008_DMM_DMM061_sparsenoise\sn1')

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

        # TODO: test different lags and see what works best, prob. ~500 msec
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

    # calculate spike-triggered average
    if use_lags:
        sta = np.zeros([
            np.size(norm_spikes, 0),
            len(lags),
            2,
            stimY,
            stimX
        ])

        # rgb_maps = np.zeros([
        #     np.size(norm_spikes, 0),
        #     len(lags),
        #     stimY,
        #     stimX,
        #     3      # color channels
        # ])

    else:
        sta = np.zeros([
            np.size(norm_spikes, 0),
            2,
            stimY,
            stimX
        ])

        # rgb_maps = np.zeros([
        #     np.size(norm_spikes, 0),
        #     stimY,
        #     stimX,
        #     3      # color channels
        # ])

    print('  -> Calculating spike-triggered averages (slow).')

    if not use_lags:
        for c in tqdm(range(np.size(norm_spikes, 0))):

            sp = summed_stim_spikes[c,:].copy()[:, np.newaxis]
            sp[np.isnan(sp)] = 0
            total_sp = np.sum(sp)
            if total_sp == 0:
                signed_sta = np.zeros((stimY*stimX, 1), dtype=float)
            else:
                # compute signed STA from the 0-centered signed stimulus
                signed_sta = (flat_signed.T @ sp) / (total_sp + 1e-12)

            signed_sta_2d = np.reshape(signed_sta, [stimY, stimX])

            # split into ON (+) and OFF (-)
            light_sta = np.maximum(signed_sta_2d, 0.)
            dark_sta = np.maximum(-signed_sta_2d, 0.)

            sta[c,0,:,:] = light_sta
            sta[c,1,:,:] = dark_sta

            # rgb_maps[c,:,:,:] = calc_combined_on_off_map(light_sta, dark_sta)

    elif use_lags:
        for c in tqdm(range(np.size(norm_spikes, 0))):
            for l_i, lag in enumerate(lags):

                sp = summed_stim_spikes[c,:].copy()[:, np.newaxis]
                sp[np.isnan(sp)] = 0

                total_sp = np.sum(sp)
                if total_sp == 0:
                    signed_sta = np.zeros((stimY*stimX, 1), dtype=float)
                else:
                    signed_sta = (np.roll(flat_signed, shift=lag, axis=0).T @ sp) / (total_sp + 1e-12)

                signed_sta_2d = np.reshape(signed_sta, [stimY, stimX])

                light_sta = np.maximum(signed_sta_2d, 0.0)
                dark_sta = np.maximum(-signed_sta_2d, 0.0)

                sta[c,l_i,0,:,:] = light_sta
                sta[c,l_i,1,:,:] = dark_sta

                # rgb_maps[c,l_i,:,:,:] = calc_combined_on_off_map(light_sta, dark_sta)

    dict_out = {
        'STAs': sta #,
        # 'stimT': stimT # ,
        # 'rgb_maps': rgb_maps
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

    savepath = os.path.join(os.path.split(data_path)[0], 'sparse_noise_receptive_fields_FULL.h5')
    fm2p.write_h5(savepath, dict_out)

    # fm2p.write_h5(r'T:\dylan\251008_DMM_DMM061_sparsenoise\sn1\sparse_noise_outputs_timecorrection_v6.h5')