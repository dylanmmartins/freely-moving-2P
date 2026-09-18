# -*- coding: utf-8 -*-
"""
tests/test_synthetic_rf.py

Recover a known receptive field from synthetic sparse noise to test STA code.


DMM, January 2026
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from fm2p.utils import sparse_noise

stimY, stimX = 24, 24
nFrames = 4000

dt = 0.5

# Sparse noise stimarr as uint8 0..255 with gray baseline 128.
stimarr = np.zeros((nFrames, stimY, stimX), dtype=np.uint8) + 128

# Randomly place small spots (white or black) per frame.
np.random.seed(0)
for i in range(nFrames):
    n_dots = np.random.randint(1, 5)
    for _ in range(n_dots):
        y = np.random.randint(0, stimY)
        x = np.random.randint(0, stimX)
        val = 255 if np.random.rand() > 0.5 else 0
        stimarr[i, y, x] = val

n_neurons = 10
# Small ON Gaussian
# create Gaussian receptive fields for each neuron (no pure-noise cells).
rf_maps = np.zeros((n_neurons, stimY, stimX), dtype=float)
np.random.seed(1)
for n in range(n_neurons):
    # Pick a center away from edges.
    cy = np.random.randint(4, stimY-4)
    cx = np.random.randint(4, stimX-4)
    sigma = 2.0 + np.random.rand() * 1.5
    for y in range(stimY):
        for x in range(stimX):
            d2 = (y - cy)**2 + (x - cx)**2
            rf_maps[n, y, x] = np.exp(-d2 / (2*(sigma**2)))
    # Normalize to be firing probability multiplier (max extra spike prob 0.5).
    rf_maps[n] = rf_maps[n] / rf_maps[n].max() * 0.5

# Synth spikes for 10 neurons: each neuron has its own Gaussian RF
# per-frame spike count.
sp_per_frame = np.zeros((n_neurons, nFrames), dtype=float)

base_rate = 0.01
for i in range(nFrames):
    frame = stimarr[i]

    white_mask = (frame > 128).astype(float)
    # Compute drive for each neuron from its RF map and generate Poisson spikes.
    for n in range(n_neurons):
        drive_n = (white_mask * rf_maps[n]).sum()
        p_n = base_rate + drive_n
        sp_per_frame[n, i] = np.random.poisson(p_n)

twopT = np.arange(0, nFrames*dt, dt)

data = {}
data['twopT'] = twopT
data['s2p_spks'] = sp_per_frame.copy()
data['stimT'] = twopT

# Apply lag to make sure analysis can correclty identify it.
stimarr = np.roll(stimarr, axis=0, shift=4)

tmp_stim_path = os.path.join(repo_root, 'tests', 'tmp_synthetic_stim.npy')
# Add color channel.
np.save(tmp_stim_path, stimarr[:,:,:,np.newaxis])

cfg = {'sparse_noise_stim_path': tmp_stim_path}

flat_stim = np.reshape(
    stimarr,
    [np.size(stimarr,0), np.size(stimarr,1)*np.size(stimarr,2)]
)

out = sparse_noise.compute_calcium_sta_spatial(flat_stim, sp_per_frame[0,:], twopT, twopT)

# print('STA shape:', out['STAs'].shape)
# print('rgb_maps shape:', out['rgb_maps'].shape)

plt.figure(figsize=(4,4))
plt.imshow(rf_maps[0], cmap='gray')
plt.title('Neuron 0 ground truth RF')
plt.axis('off')
plt.colorbar()
plt.savefig(os.path.join(repo_root, 'tests', 'synthetic_rf_ground_truth.png'), dpi=200)
print('Saved synthetic_rf_ground_truth.png')

# for l_i, lag in enumerate(np.arange(-5,5,1)):
#     rgb0 = out['rgb_maps'][0,l_i,:,:,:]
#     plt.figure(figsize=(4,4))
#     plt.imshow(rgb0)
#     plt.title('cell 0 lag={}'.format(lag))
#     plt.axis('off')
#     # plt.colorbar()
#     plt.savefig(os.path.join(repo_root, 'tests', 'synthetic_rf_result_neuron0_lag{}_v2.png'.format(lag)), dpi=200)
#     print('Saved synthetic_rf_result_neuron0_lag{}.png'.format(lag))

sta_light = out['lightSTA']
sta_dark = out['darkSTA']
lags = out['lags']

fig, axs = plt.subplots(2,10, dpi=300, figsize=(12,4))
for i in range(10):
    lightmax = np.nanmax(sta_light)
    axs[0,i].imshow(
        sta_light[i,:].reshape([stimX, stimY]),
        vmin=-lightmax, vmax=lightmax, cmap='coolwarm')
    axs[0,i].axis('off')
    axs[0,i].set_title('light, lag={:.03}'.format(lags[i]*0.5))
    darkmax = np.nanmax(sta_dark)
    axs[1,i].imshow(
        sta_dark[i,:].reshape([stimX, stimY]),
        vmin=-darkmax, vmax=darkmax, cmap='coolwarm')
    axs[1,i].axis('off')
    axs[1,i].set_title('dark, lag={:.03}'.format(lags[i]*0.5))
fig.tight_layout()
plt.savefig(os.path.join(repo_root, 'tests', 'STAs.png'), dpi=200)
