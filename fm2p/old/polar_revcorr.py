# -*- coding: utf-8 -*-

import os
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

import warnings
warnings.filterwarnings('ignore')

import fm2p


def plot_polar_retino(map_, angle_edges, dist_edges, ax=None, fov_edge=110):

    parula_map = fm2p.make_parula()

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(4,4),dpi=300,subplot_kw={'projection': 'polar'})

    rate_mesh_X, rate_mesh_Y = np.meshgrid(np.deg2rad(angle_edges), dist_edges)
    mask = np.logical_and(rate_mesh_X < np.deg2rad(360-fov_edge), rate_mesh_X > np.deg2rad(fov_edge))[:-1,:-1]

    # Z = np.roll(data['smoothed_rate_maps'][6,:,:].copy().T, shift=int(90/3), axis=1)
    Z1 = map_.T
    Z1[mask] = np.nan
    Z2 = map_.T
    Z2[~mask] = np.nan

    ax.pcolormesh(
        # (np.pi*2) - rate_mesh_X, # for egocentric
        rate_mesh_X,
        rate_mesh_Y,
        Z1,
        edgecolors='face', cmap=parula_map, shading='flat'
    )
    ax.pcolormesh(
        # (np.pi*2) - rate_mesh_X, # for egocentric
        rate_mesh_X,
        rate_mesh_Y,
        Z2,
        edgecolors='face', cmap=parula_map, shading='flat',
        alpha=0.2
    )
    ax.axis('off')
    ax.set_theta_zero_location('N') # Sets 0 degrees to the top
    ax.set_theta_direction(-1) # make it clockwise
    fig.tight_layout()
    

def polar_histogram2d(theta, r, spikes, r_bins=13, r_max=26, theta_width=3):

    theta_width = np.deg2rad(theta_width)

    theta = np.asarray(theta)
    r = np.asarray(r)
    spikes = np.asarray(spikes) # FOR SINGLE CELL!

    # wrap angles as -pi to pi
    theta = (theta + np.pi) % (2 * np.pi) - np.pi

    # Compute polar coordinates (removed 10/6/25)
    # r = np.sqrt(x**2 + y**2)
    # theta = np.arctan2(y, x) # -pi to pi

    r_edges = np.linspace(0, r_max, r_bins + 1)

    n_theta_bins = int(np.ceil((2 * np.pi) / theta_width))
    theta_edges = np.linspace(-np.pi, np.pi, n_theta_bins + 1)

    # H_occ, _, _ = np.histogram2d(
    #     r,
    #     theta,
    #     bins=[r_edges, theta_edges]
    # )
    H, _, _ = np.histogram2d(
        r,
        theta,
        bins=[r_edges, theta_edges],
        weights=spikes
    )
    # H_norm = H / H_occ
    # H_norm[np.isnan(H_norm)] = 0.

    return H, r_edges, theta_edges


def smooth_2d_rate_maps(rate_maps):

    smoothed_rate_maps = rate_maps.copy()

    for c in range(rate_maps.shape[0]):
        # pad the rate map by concatenating three copies along the angular axis
        temp_padded = np.vstack((smoothed_rate_maps[c, :, :], smoothed_rate_maps[c, :, :], smoothed_rate_maps[c, :, :]))
        # smooth the padded rate map
        # TODO: try sigma shapes that are not symetric (i.e., smooth more along angles than i do along distance axis)
        temp_smoothed = gaussian_filter(temp_padded, sigma=1)

        # slice the middle third to get the final smoothed rate map
        smoothed_rate_maps[c, :, :] = temp_smoothed[smoothed_rate_maps.shape[1]:2*smoothed_rate_maps.shape[1], :]

    return smoothed_rate_maps


def valid_mask(items):

    mask = np.ones_like(items[0]).astype(bool)

    for x in items:
        mask &= ~np.isnan(x)

    return mask.astype(bool)


def polar_revcorr(preproc_path=None):

    if preproc_path is None:
        preproc_path = fm2p.select_file(
            'Select preprocessing HDF file.',
            filetypes=[('HDF','.h5'),]
        )
        data = fm2p.read_h5(preproc_path)
    elif type(preproc_path) == str:
        data = fm2p.read_h5(preproc_path)
    elif type(preproc_path) == dict:
        data = preproc_path

    spikes = data['norm_spikes'].copy()

    theta = data['theta_interp'].copy()
    phi = data['phi_interp'].copy()

    n_cells = np.size(spikes, 0)
    n_x = 20
    n_y = 20

    all_heatmaps = np.zeros([
        n_cells,
        n_x,
        n_y
    ]) * np.nan

    retinocentric = data['retinocentric'].copy()
    egocentric = data['egocentric'].copy()
    distance = data['dist_to_pillar'].copy()

    pupil_mask = valid_mask([theta, phi])
    ret_mask = valid_mask([retinocentric, distance])
    ego_mask = valid_mask([egocentric, distance])

    x_bins = np.linspace(np.nanpercentile(theta, 10), np.nanpercentile(theta, 90), num=n_x+1)
    y_bins = np.linspace(np.nanpercentile(phi, 10), np.nanpercentile(phi, 90), num=n_y+1)

    # first, make theta/phi tuning heatmap
    print('  -> Measuring theta/phi heatmaps.')
    for c in tqdm(range(n_cells)):
        x_vals = theta.copy()[pupil_mask]
        y_vals = phi.copy()[pupil_mask]
        rates = spikes[c,pupil_mask].copy()

        heatmap = np.zeros((n_y, n_x))

        # avg firing rate in each bin
        for i in range(n_x):
            for j in range(n_y):
                in_bin = (x_vals >= x_bins[i]) & (x_vals < x_bins[i+1]) & \
                        (y_vals >= y_bins[j]) & (y_vals < y_bins[j+1])
                heatmap[j, i] = np.nanmean(rates[in_bin])

        all_heatmaps[c,:,:] = heatmap.copy()

        # ax.imshow(heatmap, origin='lower', 
        #         extent=[x_bins[0], x_bins[-1], y_bins[-1], y_bins[0]],
        #         aspect='auto', cmap='coolwarm', vmin=0, vmax=np.nanpercentile(heatmap.flatten(), 95))


    # Next, retino / ego 2D polar heatmaps
    H_ret = np.zeros([
        n_cells,
        int(26/2),
        int(360/3)
    ])
    H_ego = np.zeros_like(H_ret)

    print('  -> Measuring polar histograms for retinocentric and egocentric orientations.')
    for c in tqdm(range(n_cells)):
        H_ret[c,:,:], _, _ = polar_histogram2d(
            retinocentric[ret_mask],
            distance[ret_mask],
            spikes[c, ret_mask]
        )
        H_ego[c,:,:], r_edges, theta_edges = polar_histogram2d(
            egocentric[ego_mask],
            distance[ego_mask],
            spikes[c, ego_mask]
        )
    
    h_ret_smoothed = smooth_2d_rate_maps(H_ret)
    h_ego_smoothed = smooth_2d_rate_maps(H_ego)

    ret_occ, _, _ = polar_histogram2d(
        retinocentric[ret_mask],
        distance[ret_mask],
        np.ones_like(retinocentric[ret_mask])
    )
    ego_occ, _, _ = polar_histogram2d(
        egocentric[ego_mask],
        distance[ego_mask],
        np.ones_like(egocentric[ego_mask])
    )

    # Plot
    # fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    # H_r, H_theta = np.meshgrid(r_edges, theta_edges)
    # pcm = ax.pcolormesh(H_theta, H_r, H.T, cmap='plasma')
    # fig.colorbar(pcm, ax=ax, label="Normalized Density")
    # plt.show()

    dict_out = {
        'retino_maps_smoothed': h_ret_smoothed,
        'ego_maps_smoothed': h_ego_smoothed,
        'retino_maps': H_ret,
        'ego_maps': H_ego,
        'theta_phi_heatmaps': all_heatmaps,
        'theta_bins': x_bins,
        'phi_bins': y_bins,
        'r_edges': r_edges,
        'theta_edges': theta_edges,
        'ret_occ': ret_occ,
        'ego_occ': ego_occ
    }

    basedir, _ = os.path.split(preproc_path)
    savename = os.path.join(basedir, 'polar_revcorr_maps.h5')
    fm2p.write_h5(savename, dict_out)


if __name__ == '__main__':

    polar_revcorr()