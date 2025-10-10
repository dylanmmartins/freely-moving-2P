
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from collections import deque
from tqdm import tqdm

import fm2p


def measure_sparse_noise_receptive_fields(cfg, data, ISI=False, lag=0, testcell=False):

    if testcell:
        usecell = 1

    if 'sparse_noise_stim_path' not in cfg.keys():
        stim_path = 'T:/sparse_noise_sequence_v4.npy'
    else:
        stim_path = cfg['sparse_noise_stim_path']
    stimarr = np.load(stim_path)
    n_stim_frames = np.size(stimarr, 0)

    light_stim = stimarr.copy()[:,:,:]
    dark_stim = stimarr.copy()[:,:,:]

    light_stim[light_stim < 129] = 0
    light_stim[light_stim > 129] = 1

    dark_stim[dark_stim == 0] = 1
    dark_stim[dark_stim > 1] = 0
    
    twopT = data['twopT']
    # stim will end after twop has already ended
    if ISI:
        stimT = np.arange(0, n_stim_frames, 1)
        isiT = np.arange(0.5, n_stim_frames, 1)
    else:
        dt = 0.500
        stimT = np.arange(0, n_stim_frames*dt, dt)

    norm_spikes = data['norm_spikes'].copy()

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
                summed_stim_spikes[c,i] = np.sum(norm_spikes[c, start_win:end_win])
                summed_isi_spikes[c,i] = np.sum(norm_spikes[c, end_win:next_win])

    else:

        # TODO: test different lags and see what works best, prob. ~500 msec
        print('  -> Summing spikes during stimulus (no ISI)')
        for c in tqdm(range(np.size(norm_spikes,0))):
            for i,t in enumerate(stimT[:-1]): # in sec
                start_win, _ = fm2p.find_closest_timestamp(twopT, t)
                next_win, _ = fm2p.find_closest_timestamp(twopT, stimT[i+1])
                summed_stim_spikes[c,i] = np.sum(norm_spikes[c, start_win:next_win])

    nFrames, stimY, stimX = np.shape(stimarr)

    flat_light_stim = np.reshape(
        light_stim,
        [nFrames, stimX*stimY]
    )

    flat_dark_stim = np.reshape(
        dark_stim,
        [nFrames, stimX*stimY]
    )

    # calculate spike-triggered average
    sta = np.zeros([
        np.size(norm_spikes, 0),
        2,
        stimY,
        stimX
    ])

    print('  -> Calculating spike-triggered averages.')
    # for c in tqdm(range(np.size(norm_spikes, 0))):
    for c in tqdm(range(10)):

        sp = summed_stim_spikes[c,:].copy()[:, np.newaxis]
        sp[np.isnan(sp)] = 0

        sp = np.roll(sp, shift=lag)

        light_sta_flat = flat_light_stim.T @ sp

        light_sta = np.reshape(
            light_sta_flat,
            [stimY, stimX]
        )
        nsp = np.nansum(sp)

        light_sta = light_sta / nsp
        light_sta = light_sta - np.nanmean(light_sta)
        sta[c,0,:,:] = light_sta

        dark_sta_flat = flat_dark_stim.T @ sp

        dark_sta = np.reshape(
            dark_sta_flat,
            [stimY, stimX]
        )
        nsp = np.nansum(sp)

        dark_sta = dark_sta / nsp
        dark_sta = dark_sta - np.nanmean(dark_sta)
        sta[c,1,:,:] = dark_sta

    dict_out = {
        'STAs': sta,
        'stimT': stimT
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