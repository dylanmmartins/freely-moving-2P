
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from collections import deque
from tqdm import tqdm
from matplotlib.colors import Normalize

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


def measure_sparse_noise_receptive_fields(cfg, data, ISI=False, use_lags=False):

    if 'sparse_noise_stim_path' not in cfg.keys():
        stim_path = 'T:/sparse_noise_sequence_v5.npy'
    else:
        stim_path = cfg['sparse_noise_stim_path']
    stimarr = np.load(stim_path)
    n_stim_frames = np.size(stimarr, 0)

    light_stim = stimarr.copy()[:,:,:]
    dark_stim = stimarr.copy()[:,:,:]

    # split light vs dark dots
    light_stim[light_stim < 129] = 0
    light_stim[light_stim > 129] = 1

    dark_stim[dark_stim == 0] = 1
    dark_stim[dark_stim > 1] = 0
    
    twopT = data['twopT']

    light_stim[light_stim > 1] = 1
    dark_stim[dark_stim > 1] = 1

    # stim will end after twop has already ended
    if ISI:
        stimT = np.arange(0, n_stim_frames, 1)
        isiT = np.arange(0.5, n_stim_frames, 1)
    else:
        dt = 0.500
        stimT = np.arange(0, n_stim_frames*dt, dt)

    if use_lags:
        lags = [-4,-3,-2,-1,0,1,2,3,4]

    norm_spikes = data['s2p_spks'].copy()

    if not use_lags:
        norm_spikes = np.roll(norm_spikes, shift=-2, axis=1)

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

    flat_light_stim = np.reshape(
        light_stim,
        [nFrames, stimX*stimY]
    ) - 0.5
    flat_light_stim = flat_light_stim - np.mean(flat_light_stim, axis=1, keepdims=True)

    flat_dark_stim = np.reshape(
        dark_stim,
        [nFrames, stimX*stimY]
    )
    flat_dark_stim = flat_dark_stim - np.mean(flat_dark_stim, axis=1, keepdims=True)

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

    # light_stim_sum = np.sum(flat_light_stim, axis=0)
    # dark_stim_sum = np.sum(flat_dark_stim, axis=0)

    print('  -> Calculating spike-triggered averages (slow).')

    if not use_lags:
        for c in tqdm(range(np.size(norm_spikes, 0))):

            sp = summed_stim_spikes[c,:].copy()[:, np.newaxis]
            sp[np.isnan(sp)] = 0

            light_sta_flat = flat_light_stim.T @ sp
            light_sta_flat = light_sta_flat.T / (np.sum(sp) + 1e-6)

            light_sta = np.reshape(
                light_sta_flat,
                [stimY, stimX]
            )

            # light_sta = light_sta - np.nanmean(light_sta)
            sta[c,0,:,:] = light_sta

            dark_sta_flat = flat_dark_stim.T @ sp
            # dark_sta_flat = dark_sta_flat.T / dark_stim_sum

            dark_sta = np.reshape(
                dark_sta_flat,
                [stimY, stimX]
            )

            dark_sta = dark_sta - np.nanmean(dark_sta)
            sta[c,1,:,:] = dark_sta

            rgb_maps[c,:,:,:] = calc_combined_on_off_map(light_sta, dark_sta)

    elif use_lags:
        for c in tqdm(range(np.size(norm_spikes, 0))):
            for l_i, lag in tqdm(enumerate(lags)):

                sp = summed_stim_spikes[c,:].copy()[:, np.newaxis]
                sp[np.isnan(sp)] = 0

                sp = np.roll(sp, lag)

                light_sta_flat = flat_light_stim.T @ sp
                # light_sta_flat = light_sta_flat.T / light_stim_sum

                light_sta = np.reshape(
                    light_sta_flat,
                    [stimY, stimX]
                )

                light_sta = light_sta - np.nanmean(light_sta)
                sta[c,l_i,0,:,:] = light_sta

                dark_sta_flat = flat_dark_stim.T @ sp
                # dark_sta_flat = dark_sta_flat.T / dark_stim_sum

                dark_sta = np.reshape(
                    dark_sta_flat,
                    [stimY, stimX]
                )

                dark_sta = dark_sta - np.nanmean(dark_sta)
                sta[c,l_i,1,:,:] = dark_sta

                rgb_maps[c,l_i,:,:,:] = calc_combined_on_off_map(light_sta, dark_sta)

    dict_out = {
        'STAs': sta,
        'stimT': stimT,
        # 'light_stim_sum': light_stim_sum,
        # 'dark_stim_sum': dark_stim_sum,
        'rgb_maps': rgb_maps
    }

    return dict_out


### Some plots
# fig, ax = plt.subplots(1, 1, figsize=(5,4), dpi=300)
# ax.imshow(pop_sta, cmap='coolwarm', vmin=-600, vmax=600)
# # plt.colorbar()
# ax.axis('off')
# ax.set_title('population receptive field')
# fig.tight_layout()


# fig, axs = plt.subplots(15, 10, dpi=300, figsize=(8.5,11))
# axs = axs.flatten()

# for c, ax in enumerate(axs):
#     ax.imshow(sta[c,:,:], cmap='coolwarm', vmin=-10, vmax=10)
#     ax.axis('off')
#     ax.set_title(c)


if __name__ == '__main__':

    data = fm2p.read_h5(r'T:\Mini2P_V1PPC\251008_DMM_DMM061_sparsenoise\sn1\sn1_preproc.h5')

    data['norm_spikes'] = data['norm_spikes'][:5,:]
    dict_out = fm2p.measure_sparse_noise_receptive_fields(
        {},
        data
    )

    fm2p.write_h5(r'T:\Mini2P_V1PPC\251008_DMM_DMM061_sparsenoise\sn1\sparse_noise_outputs_5cell_v2.h5')