import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

import fm2p


def load_stim(stim_path):
    stim = np.load(stim_path)
    print(f'stim shape={stim.shape}, dtype={stim.dtype}, min={stim.min()}, max={stim.max()}')
    return stim


def build_signed_centered(stimarr):
    s = stimarr.astype(float)
    if s.max() <= 1.0:
        s = s * 255.0
    bg = np.median(s)
    print('estimated background (median) =', bg)
    white = (s > bg).astype(float)
    black = (s < bg).astype(float)
    signed = white - black  # +1 white, -1 black, 0 bg

    nFrames, h, w = signed.shape
    flat = signed.reshape(nFrames, h*w)
    # center per pixel (time mean)
    flat_centered = flat - flat.mean(axis=0, keepdims=True)
    return flat_centered, (h, w)


def load_spikes_from_h5(h5path, key_candidates=('s2p_spks','norm_spikes','s2p_spikes')):
    data = fm2p.read_h5(h5path)
    for k in key_candidates:
        if k in data:
            spikes = data[k]
            print(f'Loaded spikes from key "{k}" shape {spikes.shape}')
            return spikes, data
    raise KeyError('No spikes key found in h5; available keys: ' + ','.join(data.keys()))


def compute_sta(flat_signed, spikes, lags=(0,), min_spikes=5):
    # flat_signed: nFrames x nPixels
    nFrames, nPixels = flat_signed.shape
    n_cells = spikes.shape[0]
    sta_out = np.zeros((n_cells, len(lags), int(np.sqrt(nPixels)), int(np.sqrt(nPixels))))

    for ci in range(n_cells):
        spike_vec = spikes[ci]
        if spike_vec.shape[0] != nFrames:
            raise ValueError(f'spike length {spike_vec.shape[0]} != nFrames {nFrames}')
        for li, lag in enumerate(lags):
            flat_lag = np.roll(flat_signed, shift=lag, axis=0)
            nsp = spike_vec.sum()
            if nsp < min_spikes:
                sta = np.zeros(nPixels, dtype=float)
            else:
                sta = (flat_lag.T @ spike_vec) / (nsp + 1e-12)
            h = int(np.sqrt(nPixels))
            sta_out[ci, li] = sta.reshape(h, h)
    return sta_out


def shuffle_null(flat_signed, spikes, lags, n_shuffles=100):
    # circular shift spikes to build null distribution per pixel
    nFrames = flat_signed.shape[0]
    nPixels = flat_signed.shape[1]
    n_cells = spikes.shape[0]
    h = int(np.sqrt(nPixels))
    null_mean = np.zeros((n_cells, len(lags), h, h))
    null_std = np.zeros_like(null_mean)

    for s in range(n_shuffles):
        if s % 10 == 0:
            print('shuffle', s)
        for ci in range(n_cells):
            shift = np.random.randint(0, nFrames)
            sp_sh = np.roll(spikes[ci], shift)
            for li, lag in enumerate(lags):
                flat_lag = np.roll(flat_signed, shift=lag, axis=0)
                nsp = sp_sh.sum()
                if nsp == 0:
                    sta = np.zeros(nPixels, dtype=float)
                else:
                    sta = (flat_lag.T @ sp_sh) / (nsp + 1e-12)
                sta2 = sta.reshape(h, h)
                null_mean[ci, li] += sta2
                null_std[ci, li] += sta2**2
    null_mean /= n_shuffles
    null_std = np.sqrt(np.maximum(0, null_std / n_shuffles - null_mean**2))
    return null_mean, null_std


def save_sta_images(sta, outdir, prefix='sta'):
    os.makedirs(outdir, exist_ok=True)
    n_cells, n_lags, h, w = sta.shape
    for ci in range(n_cells):
        for li in range(n_lags):
            im = sta[ci, li]
            vmax = np.max(np.abs(im))
            plt.figure(figsize=(3,3))
            plt.imshow(im, cmap='bwr', vmin=-vmax, vmax=vmax)
            plt.colorbar()
            plt.title(f'cell{ci}_lag{li}')
            fname = os.path.join(outdir, f'{prefix}_cell{ci}_lag{li}.png')
            plt.savefig(fname, dpi=150)
            plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stim', required=True)
    p.add_argument('--h5', required=False)
    p.add_argument('--spikes', required=False, help='optional npy spikes n_cells x n_frames')
    p.add_argument('--outdir', default='./tests/diag_out')
    p.add_argument('--lags', default='-4,-3,-2,-1,0,1,2,3,4')
    p.add_argument('--nshuffles', type=int, default=100)
    args = p.parse_args()

    stim = load_stim(args.stim)
    flat_signed, (h, w) = build_signed_centered(stim)

    if args.h5:
        spikes, data = load_spikes_from_h5(args.h5)
    elif args.spikes:
        spikes = np.load(args.spikes)
        print('Loaded spikes from', args.spikes, 'shape', spikes.shape)
    else:
        # try to find spikes.npy next to stim
        spath = os.path.join(os.path.dirname(args.stim), 'spikes.npy')
        if os.path.exists(spath):
            spikes = np.load(spath)
            print('Loaded spikes from', spath, 'shape', spikes.shape)
        else:
            raise RuntimeError('No spikes provided. Provide --h5 or --spikes or place spikes.npy next to stim')

    lags = [int(x) for x in args.lags.split(',')]
    sta = compute_sta(flat_signed, spikes, lags=lags, min_spikes=5)
    save_sta_images(sta, args.outdir, prefix='signed_sta')

    print('Computing null (this may take a while)...')
    null_mean, null_std = shuffle_null(flat_signed, spikes, lags, n_shuffles=args.nshuffles)

    # z-score and save
    z = (sta - null_mean) / (null_std + 1e-12)
    save_sta_images(z, args.outdir, prefix='z_sta')

    print('Saved outputs to', args.outdir)


if __name__ == '__main__':
    main()
