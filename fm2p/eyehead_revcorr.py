# -*- coding: utf-8 -*-
# polar revcorr updated for cohort 2

import os
import argparse
import numpy as np
from tqdm import tqdm
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter

import warnings
warnings.filterwarnings('ignore')

import fm2p


def valid_mask(items):
    mask = np.ones_like(items[0]).astype(bool)
    for x in items:
        mask &= ~np.isnan(x)
    return mask.astype(bool)


def calc_1d_tuning(spikes, var, ltdk, bound=10, n_bins=13):
    # spikes should be 2D and have shape (cells, time)
    # var should be 1d
    # ltdk is the light/dark state vector, bool, 1==lights on

    bins = np.linspace(
        np.nanpercentile(var, bound),
        np.nanpercentile(var, 100-bound),
        n_bins-1
    )
    n_cells = np.size(spikes, 0)

    tuning_out = np.zeros([
        n_cells,
        len(bins-1),
        2
    ]) * np.nan
    err_out = np.zeros_like(tuning_out) * np.nan

    for state in range(2):
        # state 0 is dark, state 1 is light
        if state == 0:
            usesamp = ~ltdk
        elif state==1:
            usesamp = ltdk

        usespikes = spikes[:,usesamp]
        usevar = var[usesamp]

        bin_edges, tuning_out[:,:,state], err_out[:,:,state] = fm2p.tuning_curve(usespikes, usevar, bins)

    # for output's last dimension, 0 is dark, 1 is light
    return bin_edges, tuning_out, err_out


def tuning2d(x_vals, y_vals, rates, n_x=20, n_y=20):

    x_bins = np.linspace(np.nanpercentile(x_vals, 10), np.nanpercentile(x_vals, 90), num=n_x+1)
    y_bins = np.linspace(np.nanpercentile(y_vals, 10), np.nanpercentile(y_vals, 90), num=n_y+1)

    n_cells = np.size(rates, 0)

    heatmap = np.zeros((n_cells, n_y, n_x)) * np.nan

    for c in range(n_cells):
        for i in range(n_x):
            for j in range(n_y):
                in_bin = (x_vals >= x_bins[i]) & (x_vals < x_bins[i+1]) & \
                        (y_vals >= y_bins[j]) & (y_vals < y_bins[j+1])
                heatmap[c, j, i] = np.nanmean(rates[c, in_bin])

    return x_bins, y_bins, heatmap


def revcorr2(preproc_path=None):

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

    print('  -> Loading data.')

    eyeT = data['eyeT'][data['eyeT_startInd']:data['eyeT_endInd']]
    eyeT = eyeT - eyeT[0]

    if 'dPhi' not in data.keys():
        phi_full = np.rad2deg(data['phi'][data['eyeT_startInd']:data['eyeT_endInd']])
        dPhi  = np.diff(fm2p.interp_short_gaps(phi_full, 5)) / np.diff(eyeT)
        dPhi = np.roll(dPhi, -2)
        data['dPhi'] = dPhi

    if 'dTheta' not in data.keys():
        data['dTheta'] = data['dEye'].copy()

    # interpolate dEye values to twop data
    dTheta = fm2p.interp_short_gaps(data['dTheta'])
    dTheta = fm2p.interpT(dTheta, data['eyeT1'], data['twopT'])
    dPhi = fm2p.interp_short_gaps(data['dPhi'])
    dPhi = fm2p.interpT(dPhi, data['eyeT1'], data['twopT'])

    spikes = data['norm_spikes'].copy()
    ltdk = data['ltdk_state_vec'].copy()

    # at some point, add in accelerations
    behavior_vars = {
        # eye positions
        'theta': data['theta_interp'].copy(),
        'phi': data['phi_interp'].copy(),
        # eye speeds
        'dTheta': dTheta,
        'dPhi': dPhi,
        # head positions
        'pitch': data['pitch_trim'].copy(),
        'roll': data['roll_trim'].copy(),
        'yaw': data['head_yaw_deg'].copy(),
        # head angular rotation speeds
        'gyro_x': data['gyro_x_trim'].copy(),
        'gyro_y': data['gyro_y_trim'].copy(),
        'gyro_z': data['gyro_z_trim'].copy(),
        # head accelerations
        'acc_x': data['acc_x_trim'].copy(),
        'acc_y': data['acc_y_trim'].copy(),
        'acc_z': data['acc_z_trim'].copy()
    }

    dict_out = {}

    print('  -> Measuring 1D tuning to all eye/head variables.')
    for behavior_k, behavior_v in tqdm(behavior_vars.items()):

        b, t, e = calc_1d_tuning(spikes, behavior_v, ltdk)

        # shape of tuning and error will be (cells, bins, ltdk_state)
        dict_out['{}_1dbins'.format(behavior_k)] = b
        dict_out['{}_1dtuning'.format(behavior_k)] = t
        dict_out['{}_1derr'.format(behavior_k)] = e

    # get every pairwise combination and do the comparisons as heatmap
    print('  -> Measuring 2D tuning to all combinations of eye/head variables.')
    pairwise_combinations = tqdm(list(itertools.combinations(behavior_vars.keys(), 2)))
    for var1_key, var2_key in pairwise_combinations:
        x, y, H = tuning2d(
            behavior_vars[var1_key],
            behavior_vars[var2_key],
            spikes
        )
        dict_out['{}_vs{}_xbins'.format(var1_key, var2_key)] = x
        dict_out['{}_vs{}_ybins'.format(var1_key, var2_key)] = y
        dict_out['{}_vs{}_heatmap'.format(var1_key, var2_key)] = H

    basedir, _ = os.path.split(preproc_path)
    savename = os.path.join(basedir, 'eyehead_revcorrs.h5')
    print('  -> Writing {}'.format(savename))
    fm2p.write_h5(savename, dict_out)


if __name__ == '__main__':

    revcorr2(r'D:/freely_moving_data/V1PPC_cohort02/251016_DMM_DMM061_pos18/fm1/251016_DMM_DMM061_fm_01_preproc.h5')

