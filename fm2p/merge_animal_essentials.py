# -*- coding: utf-8 -*-


import os
import argparse
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom
from scipy.io import loadmat
import scipy.stats
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import fm2p


def merge_animal_essentials():

    parser = argparse.ArgumentParser()
    parser.add_argument('-animal', '--animal', type=str)
    parser.add_argument('-codir', '--codir', type=str)
    parser.add_argument('-mapdir', '--mapdir', type=str)
    args = parser.parse_args()

    if args.codir is None:
        cohort_dir = fm2p.select_directory(
            'Select cohort directory.'
        )
    else:
        cohort_dir = args.codir
    
    if args.mapdir is None:
        map_dir = fm2p.select_directory(
            'Select sign map and composite directory.'
        )
    else:
        map_dir = args.mapdir

    animalID = args.animal

    animal_dict = {}

    preproc_paths = fm2p.find(
        '*{}*preproc.h5'.format(animalID),
        cohort_dir
    )
    for p in preproc_paths:
        main_key = os.path.split(os.path.split(os.path.split(p)[0])[0])[1]
        pos_key = main_key.split('_')[-1]
        # v2 is the batch that were run jan 16-17 to calculate a seperate reliability score
        # for light vs dark conditions
        r = fm2p.find('eyehead_revcorrs_v4cent.h5', os.path.split(p)[0], MR=True)
        sn = os.path.join(os.path.split(os.path.split(p)[0])[0], 'sn1/sparse_noise_labels_gaussfit.npz')

        animal_dict[pos_key] = {
            'preproc': p,
            'revcorr': r,
            'sparsenoise': sn,
            'name': main_key
        }


    full_dict = {}

    all_pdata = []
    all_rdata = []
    all_pos = []
    all_cell_positions = []
    full_map = np.zeros([512*5, 512*5]) * np.nan

    row = 0
    col = 0
    for pos in range(1,26):
        pos_str = 'pos{:02d}'.format(pos)

        if pos_str not in list(animal_dict.keys()):
            if (pos%5)==0: # if you're at the end of a row
                col = 0
                row += 1
            else:
                col += 1
            continue

        pdata = fm2p.read_h5(animal_dict[pos_str]['preproc'])
        rdata = fm2p.read_h5(animal_dict[pos_str]['revcorr'])
        if os.path.isfile(animal_dict[pos_str]['sparsenoise']):
            sndata = np.load(animal_dict[pos_str]['sparsenoise'])
            snarr = np.concatenate([sndata['true_indices'][:,np.newaxis], sndata['pos_centroids']], axis=1)
        else:
            snarr = np.nan

        all_pdata.append(pdata)
        all_rdata.append(rdata)
        all_pos.append((row, col))

        singlemap = pdata['twop_ref_img']

        full_map[row*512 : (row*512)+512, col*512 : (col*512)+512] = singlemap

        cell_positions = np.zeros([len(pdata['cell_x_pix'].keys()), 2]) * np.nan

        for ki, k in enumerate(pdata['cell_x_pix'].keys()):
            # cellx = np.median(512 - pdata['cell_x_pix'][k]) + col*512
            cellx = np.median(pdata['cell_x_pix'][k]) + col*512
            celly = np.median(pdata['cell_y_pix'][k]) + row*512

            cell_positions[ki,:] = np.array([cellx, celly])

        full_dict[pos_str] = {
            'rdata': rdata,
            'tile_pos': np.array([row,col]),
            'cell_pos': cell_positions,
            'sn_cents': snarr
        }

        all_cell_positions.append(cell_positions)

        col += 1

        if (pos%5)==0: # if you're at the end of a row
            col = 0
            row += 1

    full_dict['rigid_tiled_map'] = full_map

    vfs_path = os.path.join(map_dir, 'VFS_maps.mat')
    vfs = loadmat(vfs_path)
    overlay = gaussian_filter(zoom(vfs['VFS_raw'].copy(), 2.555), 2)

    refpath = fm2p.find('*.tif', map_dir, MR=True)

    fullimg = np.array(Image.open(refpath))
    newshape = (fullimg.shape[0] // 2, fullimg.shape[1] // 2)
    zoom_factors = (
        (newshape[0]/ fullimg.shape[0]),
        (newshape[1]/ fullimg.shape[1]),
    )
    resized_fullimg = zoom(fullimg, zoom=zoom_factors, order=1)

    full_dict['sign_map'] = overlay
    full_dict['ref_img'] = resized_fullimg

    # save as v5 (jan 17)
    savepath = os.path.join(map_dir, '{}_merged_essentials_v7cent.h5'.format(animalID))
    fm2p.write_h5(savepath, full_dict)


def plot_running_median(ax, x, y, n_bins=7, vertical=False):

    bins = np.linspace(np.min(x), np.max(x), n_bins)

    bin_means, bin_edges, _ = scipy.stats.binned_statistic(
        x[~np.isnan(x) & ~np.isnan(y)],
        y[~np.isnan(x) & ~np.isnan(y)],
        statistic=np.median,
        bins=bins)
    
    bin_std, _, _ = scipy.stats.binned_statistic(
        x[~np.isnan(x) & ~np.isnan(y)],
        y[~np.isnan(x) & ~np.isnan(y)],
        statistic=np.nanstd,
        bins=bins)
    
    hist, _ = np.histogram(
        x[~np.isnan(x) & ~np.isnan(y)],
        bins=bins)
    
    tuning_err = bin_std / np.sqrt(hist)

    if not vertical:
        ax.plot(bin_edges[:-1] + (np.median(np.diff(bins))/2),
                bin_means,
                '-', color='k')
        
        ax.fill_between(bin_edges[:-1] + (np.median(np.diff(bins))/2),
                        bin_means-tuning_err,
                        bin_means+tuning_err,
                        color='k', alpha=0.2)
    
    elif vertical:
        ax.plot(bin_means,
                bin_edges[:-1] + (np.median(np.diff(bins))/2),
                '-', color='k')
        
        ax.fill_betweenx(bin_edges[:-1] + (np.median(np.diff(bins))/2),
                        bin_means-tuning_err,
                        bin_means+tuning_err,
                        color='k', alpha=0.2)


def visualize_topographic_map(messentials, composite, key, cond):

    cmap = cm.seismic
    norm = colors.Normalize(vmin=-1, vmax=1)

    h_hist_data = []
    v_hist_data = []

    fig = plt.figure(figsize=(6,6), dpi=300)

    gs = GridSpec(5,5)

    ax = fig.add_subplot(gs[1:5, 0:4])

    ax.imshow(messentials['rigid_tiled_map'], cmap='gray', alpha=0.5)
    ax.imshow(messentials['sign_map'], cmap='jet', alpha=0.15)

    for poskey in composite.keys():
        for c in range(np.size(messentials[poskey]['rdata']['{}_isrel'.format(key)], 0)):
            cellx = composite[poskey][c,2]
            celly = composite[poskey][c,3]
            cellrel = messentials[poskey]['rdata']['{}_isrel'.format(key)][c]

            if cellrel:
                cellmod = messentials[poskey]['rdata']['{}_{}_mod'.format(key, cond)][c]
                ax.plot(cellx, celly, '.', ms=3, color=cmap(norm(cellmod)))
                h_hist_data.append([cellx, cellmod])
                v_hist_data.append([celly, cellmod])

            elif not cellrel:
                ax.plot(cellx, celly, '.', ms=3, color='gray', alpha=0.2)

    ax_histx  = fig.add_subplot(gs[0, 0:4], sharex=ax)
    ax_histy  = fig.add_subplot(gs[1:5, 4], sharey=ax)

    h_hist_data = np.array(h_hist_data)
    v_hist_data = np.array(v_hist_data)

    plot_running_median(ax_histx, h_hist_data[:,0], h_hist_data[:,1], 9)
    plot_running_median(ax_histy, v_hist_data[:,0], v_hist_data[:,1], 9, vertical=True)

    # ax.invert_yaxis()
    fig.suptitle('{} ({})'.format(key, cond))

    fig.tight_layout()

    return fig


if __name__ == '__main__':

    merge_animal_essentials()

