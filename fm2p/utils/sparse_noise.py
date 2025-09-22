
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from collections import deque
from tqdm import tqdm

import fm2p


def measure_sparse_noise_receptive_fields(cfg, data):

    stim_path = cfg['sparse_noise_stim_path']
    stimarr = np.load(stim_path)
    n_stim_frames = np.size(stimarr, 0)

    light_stim = stimarr.copy()[:,:,:,0]
    dark_stim = stimarr.copy()[:,:,:,0]

    light_stim[light_stim < 129] = 0
    light_stim[light_stim > 129] = 1

    dark_stim[dark_stim == 0] = 1
    dark_stim[dark_stim > 1] = 0
    
    twopT = data['twopT']
    # stim will end after twop has already ended
    stimT = np.arange(0, n_stim_frames, 1) # 1000 frames presented for 0.5on, 0.5 off (1.0 sec total)
    isiT = np.arange(0.5, n_stim_frames, 1)

    norm_spikes = data['norm_spikes'].copy()

    summed_stim_spikes = np.zeros([
        np.size(norm_spikes, 0),
        np.size(stimT)
    ]) * np.nan
    summed_isi_spikes = np.zeros([
        np.size(norm_spikes, 0),
        np.size(stimT)
    ]) * np.nan

    # sum spikes that occur during a single frame
    # also sum spikes in each ISI period
    for c in tqdm(range(np.size(norm_spikes,0))):
        for i,t in enumerate(stimT[:-1]): # in sec
            start_win, _ = fm2p.find_closest_timestamp(twopT, t)
            end_win, _ = fm2p.find_closest_timestamp(twopT, isiT[i])
            next_win, _ = fm2p.find_closest_timestamp(twopT, stimT[i+1])
            summed_stim_spikes[c,i] = np.sum(norm_spikes[c, start_win:end_win])
            summed_isi_spikes[c,i] = np.sum(norm_spikes[c, end_win:next_win])

    nFrames, stimY, stimX, _ = np.shape(stimarr)

    flat_light_stim = np.reshape(
        light_stim, # drop color channel, all are identical since it's a b/w stimulus
        [nFrames, stimX*stimY]
    )

    flat_dark_stim = np.reshape(
        dark_stim, # drop color channel, all are identical since it's a b/w stimulus
        [nFrames, stimX*stimY]
    )

    # calculate spike-triggered average
    sta = np.zeros([
        np.size(norm_spikes, 0),
        2,
        stimY,
        stimX
    ])

    for c in tqdm(range(np.size(norm_spikes, 0))):
        sp = summed_stim_spikes[c,:].copy()
        sp[np.isnan(sp)] = 0

        light_sta_flat = flat_light_stim.T @ sp
        light_sta = np.reshape(
            light_sta_flat,
            [stimY, stimX]
        )
        nsp = np.nansum(sp)

        light_sta_flat = light_sta_flat / nsp
        light_sta_flat = light_sta_flat - np.nanmean(light_sta_flat)
        sta[c,0,:,:] = light_sta_flat

        dark_sta_flat = flat_dark_stim.T @ sp
        dark_sta = np.reshape(
            dark_sta_flat,
            [stimY, stimX]
        )
        nsp = np.nansum(sp)

        dark_sta_flat = dark_sta_flat / nsp
        dark_sta_flat = dark_sta_flat - np.nanmean(dark_sta_flat)
        sta[c,1,:,:] = dark_sta_flat

    # calculate the population receptive field
    # pop_sta = np.sum(sta, axis=0)

    # Would it be meaningful to calculate a spontaneous rate to the ISI?
    # Could try this later...

    return sta


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