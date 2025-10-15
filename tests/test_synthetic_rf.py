import numpy as np
import matplotlib.pyplot as plt
import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from fm2p.utils import sparse_noise

stimY, stimX = 32, 32
nFrames = 2000

dt = 0.5

# sparse noise stimarr as uint8 0..255 with gray baseline 128
stimarr = np.zeros((nFrames, stimY, stimX), dtype=np.uint8) + 128

# randomly place small spots (white or black) per frame
np.random.seed(0)
for i in range(nFrames):
    n_dots = np.random.randint(1, 5)
    for _ in range(n_dots):
        y = np.random.randint(0, stimY)
        x = np.random.randint(0, stimX)
        val = 255 if np.random.rand() > 0.5 else 0
        stimarr[i, y, x] = val

# small ON Gaussian
rf = np.zeros((stimY, stimX), dtype=float)
cy, cx = 16, 18
for y in range(stimY):
    for x in range(stimX):
        d2 = (y - cy)**2 + (x - cx)**2
        rf[y,x] = np.exp(-d2 / (2*(2.0**2)))

# normalize to be firing probability multiplier
rf = rf / rf.max() * 0.5  # max extra spike prob 0.5

# synth spikes for 10 neurons: 1 neuron with RF, rest random/noise
n_neurons = 10
spikes = np.zeros((n_neurons, int(nFrames*dt*2)), dtype=float)  # high-res timeline not required
# per-frame spike count
sp_per_frame = np.zeros((n_neurons, nFrames), dtype=float)

base_rate = 0.1
for i in range(nFrames):
    frame = stimarr[i]

    white_mask = (frame > 128)
    # compute drive for neuron 0 as dot product
    drive0 = (white_mask.astype(float) * rf).sum()
    # prob of spiking ~ base + drive0
    p0 = base_rate + drive0
    sp_per_frame[0, i] = np.random.poisson(p0)

    for n in range(1, n_neurons):
        sp_per_frame[n, i] = np.random.poisson(base_rate)

twopT = np.arange(0, nFrames*dt, dt)

data = {}
data['twopT'] = twopT
data['s2p_spks'] = sp_per_frame.copy()

tmp_stim_path = os.path.join(repo_root, 'tests', 'tmp_synthetic_stim.npy')
np.save(tmp_stim_path, stimarr)

cfg = {'sparse_noise_stim_path': tmp_stim_path}

out = sparse_noise.measure_sparse_noise_receptive_fields(cfg, data, ISI=False, use_lags=False)

print('STA shape:', out['STAs'].shape)
print('rgb_maps shape:', out['rgb_maps'].shape)

rgb0 = out['rgb_maps'][0]
plt.figure(figsize=(4,4))
plt.imshow(rgb0)
plt.title('Recovered RGB map neuron 0')
plt.axis('off')
plt.savefig(os.path.join(repo_root, 'tests', 'synthetic_rf_result_neuron0.png'), dpi=200)
print('Saved synthetic_rf_result_neuron0.png')
