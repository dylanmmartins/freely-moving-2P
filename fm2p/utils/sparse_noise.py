
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
        lags_ = np.arange(-len(sd)+1, len(sd))

        center = len(cc)//2
        low = max(0, center - maxlag_frames)
        high = min(len(cc), center + maxlag_frames + 1)
        sub = cc[low:high]
        sublags = lags_[low:high]
        
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


from scipy.signal import correlate

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
    stim_shifted = np.full_like(stim, fill_value)
    if delay_frames > 0:
        stim_shifted[delay_frames:, :] = stim[:-delay_frames, :]
    elif delay_frames < 0:
        stim_shifted[:delay_frames, :] = stim[-delay_frames:, :]
    else:
        stim_shifted[:] = stim
    return  stim_shifted


def measure_sparse_noise_receptive_fields(cfg, data, ISI=False, use_lags=False):

    print('  -> Loading data.')

    if 'sparse_noise_stim_path' not in cfg.keys():
        stim_path = 'T:/dylan/sparse_noise_sequence_v7.npy'
    else:
        stim_path = cfg['sparse_noise_stim_path']
    stimarr = np.load(stim_path)[:,:,:,0] # drop color channel
    n_stim_frames = np.size(stimarr, 0)

    # signed stim... +1 for white, -1 for black, 0 for background
    stim_f = stimarr.astype(float)
    # make sure it's scaled to 0:255
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
        # because the starting value is unreliable and does not correpond
        # to the 2P timestamps, i'm turning them into relative timestamps.
        # Later they will be aligned post-hot by cross correlation of stimulus
        # drive.
        # stimT = data['stimT'] - data['stimT'][0]

        # 251021 try switching back to sytnthetic timestamps,
        # since real ones may be misreported from psychtoolbox
        stimT = np.arange(0, n_stim_frames, 1)

    if use_lags:
        # Lags are in frames. Positive lag means we look backward in time:
        # the STA at lag L uses stimulus frames that occurred L frames before each spike
        # (useful for measuring causal stimulus -> spike relationships).
        # Negative lag means we look forward: STA at negative lag uses stimulus frames
        # that occur after the spike (useful for diagnostics but not causal).
        lags = np.arange(-5,5,1)

    norm_spikes = data['s2p_spks'].copy()[:10,:] # do just a subset of cells

    # if not use_lags:
        # shift spikes forward by 2 frames without wrap-around (pad with zeros)
        # shift = 0
        # if shift != 0:
        #     shifted = np.zeros_like(norm_spikes)
        #     if shift < norm_spikes.shape[1]:
        #         shifted[:, shift:] = norm_spikes[:, :-shift]
        #     # else leave as zeros
        #     norm_spikes = shifted

    # find timing correction
    # stimT = correct_stim_timing(stimarr, data, r'T:\dylan\251008_DMM_DMM061_sparsenoise\sn1')

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


    print('  -> Estimating stimulus delay via FFT-based cross-correlation.')

    # Allow correction for a measured stimulus delay (in frames).
    # If the stimulus timestamps are delayed relative to the two-photon frames
    # (e.g., stim is late by ~35 frames), set cfg['sparse_noise_stim_delay_frames']=35
    # Positive delay means the stim is late and we shift the stimulus earlier
    # so that flat_signed_shifted[t] = original_flat_signed[t + delay].
    # First check cfg-provided explicit delay
    delay_frames = 0
    if isinstance(cfg, dict):
        delay_frames = int(cfg.get('sparse_noise_stim_delay_frames', 0) or 0)

    stim_drive = np.std(flat_signed, axis=1)
    pop_resp = np.nansum(data.get('s2p_spks', np.zeros((1, twopT.shape[0]))), axis=0)

    # If no explicit delay provided, estimate it automatically using a fast FFT-based xcorr
    # if delay_frames == 0 and isinstance(cfg, dict) and cfg.get('sparse_noise_auto_estimate', True):
    #     try:
    #         #stimulus drive per stim-frame
    #         stim_drive = np.std(flat_signed, axis=1)

    #         # interp stim drive onto two-photon timebase using real timestamps
    #         stim_on_twop = None
    #         try:
    #             # stimT should be available and match stim_drive length
    #             if ('stimT' in locals() or 'stimT' in globals()) and len(stim_drive) == len(stimT):
    #                 f_stim = interp1d(stimT, stim_drive, bounds_error=False, fill_value='extrapolate')
    #                 stim_on_twop = f_stim(twopT)
    #         except Exception:
    #             stim_on_twop = None

    #         if stim_on_twop is None:
    #             # fallback: proportional index-based resample to twopT length
    #             idx_old = np.linspace(0, 1, num=stim_drive.shape[0])
    #             idx_new = np.linspace(0, 1, num=twopT.shape[0])
    #             f_idx = interp1d(idx_old, stim_drive, bounds_error=False, fill_value=0.0)
    #             stim_on_twop = f_idx(idx_new)

    #         # population response on twop timebase
    #         pop_resp = np.nansum(data.get('s2p_spks', np.zeros((1, twopT.shape[0]))), axis=0)
    #         # iff lengths mismatch with twopT, resample pop_resp proportionally to twopT length
    #         if pop_resp.shape[0] != twopT.shape[0]:
    #             idx_old = np.linspace(0, 1, num=pop_resp.shape[0])
    #             idx_new = np.linspace(0, 1, num=twopT.shape[0])
    #             f_idx2 = interp1d(idx_old, pop_resp, bounds_error=False, fill_value=0.0)
    #             pop_on_twop = f_idx2(idx_new)
    #         else:
    #             pop_on_twop = pop_resp

    #         # smooth and z-score both signals to emphasize slower, causal structure
    #         sigma = int(cfg.get('sparse_noise_xcorr_smooth_sigma', 2))
    #         stim_s = gaussian_filter1d(np.nan_to_num(stim_on_twop - np.nanmean(stim_on_twop)), sigma=sigma)
    #         pop_s = gaussian_filter1d(np.nan_to_num(pop_on_twop - np.nanmean(pop_on_twop)), sigma=sigma)
    #         if np.nanstd(stim_s) > 0:
    #             stim_s = stim_s / (np.nanstd(stim_s) + 1e-12)
    #         if np.nanstd(pop_s) > 0:
    #             pop_s = pop_s / (np.nanstd(pop_s) + 1e-12)

    #         # FFT cross-corr to find lag in frames
    #         L = len(stim_s)
    #         nfft = int(2 ** np.ceil(np.log2(L * 2)))
    #         S = np.fft.rfft(stim_s, n=nfft)
    #         R = np.fft.rfft(pop_s, n=nfft)

    #         cross_power = S * np.conj(R)
    #         cross_power /= np.abs(cross_power) + 1e-12  # normalize phase

    #         cc_full = np.fft.irfft(cross_power, n=nfft)
    #         cc = np.concatenate((cc_full[-(L-1):], cc_full[:L]))
    #         delay_test_lags = np.arange(-L+1, L)

    #         # restrict search to a plausible window
    #         search_min = int(cfg.get('sparse_noise_search_min_frames', -80))
    #         search_max = int(cfg.get('sparse_noise_search_max_frames', 80))
    #         search_min = max(search_min, delay_test_lags.min())
    #         search_max = min(search_max, delay_test_lags.max())
    #         mask = (delay_test_lags >= search_min) & (delay_test_lags <= search_max)
    #         if mask.sum() == 0:
    #             delay_frames = 0
    #         else:
    #             sub = cc[mask]
    #             sublags = delay_test_lags[mask]
    #             best_idx = np.nanargmax(sub)
    #             best_lag = int(sublags[best_idx])
    #             # clamp extreme results
    #             max_cap = min(500, L//2)
    #             if abs(best_lag) > max_cap:
    #                 delay_frames = 0
    #             else:
    #                 delay_frames = best_lag
    #     except Exception:
    #         delay_frames = 0

    # # Normalize the cross-correlation
    # cc_full = np.fft.irfft(S * np.conjugate(R), n=nfft)
    # cc_full = cc_full / (L * np.std(stim_s) * np.std(pop_s))
    # cc = np.concatenate((cc_full[-(L-1):], cc_full[:L]))

    delay_frames = find_delay_frames(stim_drive, pop_resp)
    print('Using {} as frame delay.'.format(delay_frames))

    # apply zero-padded shift for the estimated or provided delay
    # if delay_frames != 0:
    #     d = int(delay_frames)
    #     if d > 0:
    #         # shift stimulus earlier in time: drop first d rows and pad zeros at end
    #         pad = np.zeros((d, flat_signed.shape[1]), dtype=flat_signed.dtype)
    #         flat_signed = np.vstack((flat_signed[d:, :], pad))
    #     else:
    #         d = -d
    #         # negative delay: shift stimulus later (pad at start)
    #         pad = np.zeros((d, flat_signed.shape[1]), dtype=flat_signed.dtype)
    #         flat_signed = np.vstack((pad, flat_signed[:-d, :]))

    # stim_shifted = np.roll(flat_signed, shift=delay_frames, axis=0)
    stim_shifted = shift_stimulus(flat_signed, delay_frames)

    # calculate spike-triggered average
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
                    # avoid circular wrap-around from np.roll by 0-padding
                    # if lag == 0:
                    #     rolled = flat_signed
                    # elif lag > 0:
                    #     # shift stimulus forward in time ... pad start with zeros
                    #     rolled = np.vstack((np.zeros((lag, flat_signed.shape[1])), flat_signed[:-lag, :]))
                    # else:
                    #     s = -int(lag)
                    #     rolled = np.vstack((flat_signed[s:, :], np.zeros((s, flat_signed.shape[1]))))

                    rolled = np.roll(stim_shifted, shift=lag, axis=0)

                    signed_sta = (rolled.T @ sp) / (total_sp + 1e-12)

                signed_sta_2d = np.reshape(signed_sta, [stimY, stimX])

                light_sta = np.maximum(signed_sta_2d, 0.0)
                dark_sta = np.maximum(-signed_sta_2d, 0.0)

                sta[c,l_i,0,:,:] = light_sta
                sta[c,l_i,1,:,:] = dark_sta

                rgb_maps[c,l_i,:,:,:] = calc_combined_on_off_map(light_sta, dark_sta)

    lags = np.arange(-20, 21)
    snr = []
    for lag in lags:
        sp = summed_stim_spikes[c,:].copy()[:, np.newaxis]
        sp[np.isnan(sp)] = 0
        shifted = np.roll(stim_shifted, -lag, axis=0)
        shifted[:lag, :] = 0
        sta = (shifted.T @ sp) / (np.sum(sp) + 1e-12)
        snr.append(np.std(sta) / (np.mean(np.abs(sta)) + 1e-12))
    plt.figure()
    plt.plot(lags, snr)
    plt.xlabel('Lag (frames, stim before spike)')
    plt.ylabel('Relative STA amplitude')
    plt.title(f'Cell 0: STA “energy” vs lag')
    plt.tight_layout()
    plt.show()

    def compute_sta_lags(stim, spikes, lags):
        # stim: (T, P), spikes: (T,) counts or bool
        T, P = stim.shape
        spikes = spikes.ravel()
        sta_by_lag = np.zeros((len(lags), P), dtype=float)
        counts = np.zeros(len(lags), dtype=int)
        for i, lag in enumerate(lags):
            # stim frame used for a spike at time t is stim[t - lag]
            # implement by rolling stimulus backward by lag (no wrap)
            if lag == 0:
                stim_lag = stim.copy()
            elif lag > 0:
                stim_lag = np.zeros_like(stim)
                stim_lag[lag:, :] = stim[:-lag, :]
            else:  # negative lags (stim after spike)
                d = -lag
                stim_lag = np.zeros_like(stim)
                stim_lag[:-d, :] = stim[d:, :]

            valid = spikes > 0
            # remove times that used wrapped/zeroed frames (first/last lag frames)
            if lag > 0:
                valid[:lag] = False
            elif lag < 0:
                valid[lag:] = False

            idx = np.where(valid)[0]
            counts[i] = idx.size
            if idx.size > 0:
                # if spikes are counts, weight by counts
                w = spikes[idx].astype(float)
                sta_by_lag[i, :] = (stim_lag[idx, :].T @ w) / (w.sum() + 1e-12)
            else:
                sta_by_lag[i, :] = 0.0
        return sta_by_lag, counts

    def snr_metrics_per_lag(sta_by_lag, baseline_lags=None):
        # sta_by_lag: (n_lags, P)
        peak_abs = np.max(np.abs(sta_by_lag), axis=1)  # per-lag peak
        # compute noise std using baseline_lags or all other lags
        if baseline_lags is None:
            # use lags far from peak (e.g. all lags)
            noise_std = np.std(sta_by_lag, axis=1)  # per-lag std across pixels
        else:
            noise_std = np.std(sta_by_lag[baseline_lags, :], axis=1).mean()  # single noise estimate
        # Avoid div by zero
        snr = peak_abs / (noise_std + 1e-12)
        return peak_abs, noise_std, snr

    # === Parameters ===
    lags = np.arange(-20, 41)   # e.g. -20..+40 frames around spike
    cell = 0
    sp = summed_stim_spikes[cell, :]  # (T,)
    sta_by_lag, counts = compute_sta_lags(stim_shifted, sp, lags)

    # SNR: compute using pixel std within each lag as numerator/denom
    peak_abs, noise_std, snr_simple = snr_metrics_per_lag(sta_by_lag)

    # Null: shuffle spike times many times to get null SNR distribution
    n_shuf = 200
    snr_null = np.zeros((len(lags), n_shuf))
    T = stim_shifted.shape[0]
    for k in range(n_shuf):
        shuf = np.random.permutation(sp)
        sta_shuf, _ = compute_sta_lags(stim_shifted, shuf, lags)
        peak_shuf = np.max(np.abs(sta_shuf), axis=1)
        std_shuf = np.std(sta_shuf, axis=1)
        snr_null[:, k] = peak_shuf / (std_shuf + 1e-12)

    # Compute null mean/std
    null_mean = snr_null.mean(axis=1)
    null_std = snr_null.std(axis=1)

    # Plot results
    fig, axs = plt.subplots(3, 1, figsize=(7, 9))
    axs[0].plot(lags, counts, marker='o')
    axs[0].axvline(0, color='k', ls='--'); axs[0].set_ylabel('Spike counts contributing')
    axs[0].set_title(f'Cell {cell}: counts per lag')

    axs[1].plot(lags, peak_abs, label='peak_abs')
    axs[1].plot(lags, noise_std, label='pixel std (noise)')
    axs[1].legend(); axs[1].axvline(0, color='k', ls='--'); axs[1].set_ylabel('Amplitude')

    axs[2].plot(lags, snr_simple, label='SNR simple')
    axs[2].plot(lags, null_mean, label='null mean', color='gray')
    axs[2].fill_between(lags, null_mean - null_std, null_mean + null_std, color='gray', alpha=0.25)
    axs[2].axvline(0, color='k', ls='--'); axs[2].set_ylabel('SNR'); axs[2].set_xlabel('Lag (frames)')
    axs[2].legend()
    plt.tight_layout()
    plt.show()


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

    # fm2p.write_h5(r'T:\dylan\251008_DMM_DMM061_sparsenoise\sn1\sparse_noise_outputs_timecorrection_v6.h5')