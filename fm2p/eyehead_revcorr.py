# -*- coding: utf-8 -*-

import os
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

import fm2p


def valid_mask(items):
    mask = np.ones_like(items[0]).astype(bool)
    for x in items:
        mask &= ~np.isnan(x)
    return mask.astype(bool)


def _half_tuning_curve(spikes_1d, beh, bins):
    """Mean firing rate per bin for one cell/half of the recording."""
    n_bins = len(bins) - 1
    tc = np.full(n_bins, np.nan)
    for b in range(n_bins):
        if b < n_bins - 1:
            in_bin = (beh >= bins[b]) & (beh < bins[b + 1])
        else:
            in_bin = (beh >= bins[b]) & (beh <= bins[b + 1])
        valid = in_bin & np.isfinite(beh)
        if np.sum(valid) > 0:
            tc[b] = np.nanmean(spikes_1d[valid])
    return tc


def _split_half_corrcoef(sp_h1, sp_h2, beh_h1, beh_h2, bins):
    """Split-half tuning curve correlation for one cell.

    Returns NaN when either half is constant (caller treats NaN as 1.0 —
    a consistently flat response is a reliable response).
    """
    tc1 = _half_tuning_curve(sp_h1, beh_h1, bins)
    tc2 = _half_tuning_curve(sp_h2, beh_h2, bins)

    valid = np.isfinite(tc1) & np.isfinite(tc2)
    if np.sum(valid) < 2:
        return np.nan

    t1, t2 = tc1[valid], tc2[valid]
    if np.std(t1) < 1e-10 or np.std(t2) < 1e-10:
        return np.nan  # flat → caller treats as 1.0 (reliable)

    return np.corrcoef(t1, t2)[0, 1]


def calc_reliability_over(spikes, behavior, n_cnk=20, n_shfl=100, relthresh=10,
                          n_bins=13, bound=10):
    """Calculate reliability of neural response to a specific behavioral variable.

    A cell is reliable if it consistently shows the same tuning to the behavioral
    variable across split halves of the recording.

    Approach:
      1. Divide the recording into n_cnk equal chunks. Assign alternating chunks
         to two halves (interleaved: even → h1, odd → h2).
      2. Compute the 1-D tuning curve for each half.
      3. True reliability = Pearson correlation between the two tuning curves.
      4. Null distribution: circularly roll the behavioral variable by a large
         random amount, then recompute the split-half correlation. Rolling the
         behavior breaks spike–behavior alignment while preserving each signal's
         marginal distribution.

    Flat-tuning-curve handling:
      If BOTH halves have zero variance (truly flat response), corrcoef is
      undefined → NaN → treated as 1.0 (perfectly consistent flat response).

    Classification: a cell is reliable if fewer than `relthresh` null shuffles
    have a split-half correlation exceeding the true value (out of `n_shfl`).

    Parameters
    ----------
    spikes : (n_cells, n_frames) array
    behavior : (n_frames,) array of the behavioral variable being tested
    n_cnk : int
        Number of equal-duration time chunks (must be even).
    n_shfl : int
        Number of null-distribution shuffles.
    relthresh : int
        Maximum number of null shuffles that may exceed the true correlation
        for a cell to be classified as reliable.

    Returns
    -------
    reliabilities : (n_cells,) — count of null shuffles exceeding true corr.
    reliable_inds : (n_cells,) bool — True if reliabilities <= relthresh.
    """
    n_cells = np.size(spikes, 0)
    n_frames = np.size(spikes, 1)

    if behavior is None or n_frames < 2 * n_cnk:
        return np.zeros(n_cells), np.ones(n_cells, dtype=bool)

    beh = np.asarray(behavior, dtype=float)
    if np.sum(np.isfinite(beh)) < n_bins + 1:
        return np.zeros(n_cells), np.ones(n_cells, dtype=bool)

    bins = np.linspace(
        np.nanpercentile(beh, bound),
        np.nanpercentile(beh, 100 - bound),
        n_bins + 1
    )

    # Split into two interleaved halves (even chunks → h1, odd → h2)
    chunk_size = n_frames // n_cnk
    h1_frames = np.concatenate([
        np.arange(i * chunk_size, (i + 1) * chunk_size)
        for i in range(0, n_cnk, 2)
    ])
    h2_frames = np.concatenate([
        np.arange(i * chunk_size, (i + 1) * chunk_size)
        for i in range(1, n_cnk, 2)
    ])

    beh_h1 = beh[h1_frames]
    beh_h2 = beh[h2_frames]

    # True split-half correlation
    true_corr = np.zeros(n_cells)
    for c in range(n_cells):
        val = _split_half_corrcoef(
            spikes[c, h1_frames], spikes[c, h2_frames], beh_h1, beh_h2, bins
        )
        true_corr[c] = 1.0 if np.isnan(val) else val

    # Null: circularly roll the behavior vector to break spike–behavior
    # alignment while preserving each marginal distribution.
    min_roll = max(1, n_frames // 10)
    max_roll = n_frames - min_roll

    null_corr = np.zeros((n_shfl, n_cells))
    for s in range(n_shfl):
        np.random.seed(s)
        roll_amt = np.random.randint(min_roll, max_roll)
        beh_rolled = np.roll(beh, roll_amt)
        beh_r_h1 = beh_rolled[h1_frames]
        beh_r_h2 = beh_rolled[h2_frames]
        for c in range(n_cells):
            val = _split_half_corrcoef(
                spikes[c, h1_frames], spikes[c, h2_frames], beh_r_h1, beh_r_h2, bins
            )
            null_corr[s, c] = 1.0 if np.isnan(val) else val

    # Low count → true corr exceeds null → reliable
    reliabilities = np.sum(null_corr > true_corr[np.newaxis, :], axis=0)
    reliable_inds = reliabilities <= relthresh

    return reliabilities, reliable_inds
    

def calc_1d_tuning(spikes, var, ltdk, bound=10, n_bins=13):
    # spikes should be 2D and have shape (cells, time)
    # var should be 1d
    # ltdk is the light/dark state vector, bool, 1==lights on

    bins = np.linspace(
        np.nanpercentile(var, bound),
        np.nanpercentile(var, 100-bound),
        n_bins
    )
    n_cells = np.size(spikes, 0)

    tuning_out = np.zeros([
        n_cells,
        len(bins)-1,
        2
    ]) * np.nan
    err_out = np.zeros_like(tuning_out) * np.nan

    for state in range(2):
        # state 0 is dark, state 1 is light
        if state == 0:
            usesamp = ~ltdk
        elif state==1:
            usesamp = ltdk

        # usespikes = fm2p.zscore_spikes(spikes[:,usesamp]) # stopped z-scoring spikes... jan 21 2026
        usespikes = spikes[:, usesamp]
        usevar = var[usesamp]

        bin_edges, tuning_out[:,:,state], err_out[:,:,state] = fm2p.tuning_curve(usespikes, usevar, bins)

    # for output's last dimension, 0 is dark, 1 is light
    return bin_edges, tuning_out, err_out


def tuning2d(x_vals, y_vals, rates, n_x=13, n_y=13):

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


def eyehead_revcorr(preproc_path=None):

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

        if 'dEye' not in data.keys():
            t = eyeT.copy()[:-1]
            t1 = t + (np.diff(eyeT) / 2)
            theta_full = np.rad2deg(data['theta'][data['eyeT_startInd']:data['eyeT_endInd']])
            dEye  = np.diff(fm2p.interp_short_gaps(theta_full, 5)) / np.diff(eyeT)
            data['dTheta'] = np.roll(dEye, -2) # static offset correction
            data['eyeT1'] = t1

        else:
            data['dTheta'] = data['dEye'].copy()

    # interpolate dEye values to twop data
    dTheta = fm2p.interp_short_gaps(data['dTheta'])
    dTheta = fm2p.interpT(dTheta, data['eyeT1'], data['twopT'])
    dPhi = fm2p.interp_short_gaps(data['dPhi'])
    dPhi = fm2p.interpT(dPhi, data['eyeT1'], data['twopT'])

    spikes = data['norm_spikes'].copy()

    ltdk = data['ltdk_state_vec'].copy()

    if 'dGaze' in data.keys():
        gaze = np.cumsum(data['dGaze'].copy())
        dGaze = data['dGaze'].copy()
        gazeT = data['eyeT_trim'] + (np.nanmedian(data['eyeT_trim']) / 2)
        gazeT = gazeT[:-1]
        # dGazeT = data['eyeT_trim']
        gaze = fm2p.interpT(gaze, gazeT, data['twopT'])
        dGaze = fm2p.interpT(dGaze, gazeT, data['twopT'])

    # at some point, add in accelerations
    if 'gyro_x_twop_interp' in data.keys():
        behavior_vars = {not
            # head positions
            'yaw': data['head_yaw_deg'].copy(),
            'pitch': data['pitch_twop_interp'].copy(),
            'roll': data['roll_twop_interp'].copy(),
            # gaze
            'gaze': gaze,
            'dGaze': dGaze,
            # eye positions
            'theta': data['theta_interp'].copy() - np.nanmean(data['theta_interp']),
            'phi': data['phi_interp'].copy() - np.nanmean(data['phi_interp']),
            # eye speeds
            'dTheta': dTheta,
            'dPhi': dPhi,
            # head angular rotation speeds
            'gyro_x': data['gyro_x_twop_interp'].copy(),
            'gyro_y': data['gyro_y_twop_interp'].copy(),
            'gyro_z': data['gyro_z_twop_interp'].copy(),
            # head accelerations
            'acc_x': data['acc_x_twop_interp'].copy(),
            'acc_y': data['acc_y_twop_interp'].copy(),
            'acc_z': data['acc_z_twop_interp'].copy()
        }
    else:
        behavior_vars = {
            # gaze
            # 'gaze': gaze,
            # 'dGaze': dGaze,
            # eye positions
            'theta': data['theta_interp'].copy() - np.nanmean(data['theta_interp']),
            'phi': data['phi_interp'].copy() - np.nanmean(data['phi_interp']),
            # eye speeds
            'dTheta': dTheta,
            'dPhi': dPhi,
        }

    dict_out = {}

    print('  -> Measuring 1D tuning to all eye/head variables.')
    for behavior_k, behavior_v in behavior_vars.items():

        print(behavior_k)

        try:
            b, t, e = calc_1d_tuning(spikes, behavior_v, ltdk)
        except IndexError:

            if len(behavior_v) != len(np.arange(len(behavior_v)*(1/7.5), 1/7.5)):

                print('Interpolating {} as {} to {}.'.format(len(behavior_v), len(np.arange(0, len(behavior_v)*(1/7.5), 1/7.5)[:-1]), len(data['twopT'])))

                try:
                    behavior_v = fm2p.interpT(
                        fm2p.nan_interp(behavior_v),
                        np.arange(0, len(behavior_v)*(1/7.5), 1/7.5)[:-1],
                        data['twopT']
                    )

                except:

                    behavior_v = fm2p.interpT(
                        fm2p.nan_interp(behavior_v),
                        np.arange(0, (len(behavior_v))*(1/7.5), 1/7.5),
                        data['twopT']
                    )


                b, t, e = calc_1d_tuning(spikes, behavior_v, ltdk)
            else:

                print('Interpolating {} as {} to {}.'.format(len(behavior_v), len(np.arange(0, len(behavior_v)*(1/7.5), 1/7.5)), len(data['twopT'])))
                behavior_v = fm2p.interpT(
                    fm2p.nan_interp(behavior_v),
                    np.arange(0, len(behavior_v)*(1/7.5), 1/7.5),
                    data['twopT']
                )
                b, t, e = calc_1d_tuning(spikes, behavior_v, ltdk)


        mod_l, ismod_l = fm2p.calc_multicell_modulation(t[:,:,1])
        mod_d, ismod_d = fm2p.calc_multicell_modulation(t[:,:,0])

        relL, isrelL = calc_reliability_over(spikes[:,ltdk], behavior_v[ltdk])
        relD, isrelD = calc_reliability_over(spikes[:,~ltdk], behavior_v[~ltdk])

        # shape of tuning and error will be (cells, bins, ltdk_state)
        dict_out['{}_1dbins'.format(behavior_k)] = b
        dict_out['{}_1dtuning'.format(behavior_k)] = t
        dict_out['{}_1derr'.format(behavior_k)] = e
        dict_out['{}_l_mod'.format(behavior_k)] = mod_l
        dict_out['{}_l_ismod'.format(behavior_k)] = ismod_l
        dict_out['{}_d_mod'.format(behavior_k)] = mod_d
        dict_out['{}_d_ismod'.format(behavior_k)] = ismod_d
        dict_out['{}_l_rel'.format(behavior_k)] = relL
        dict_out['{}_l_isrel'.format(behavior_k)] = isrelL
        dict_out['{}_d_rel'.format(behavior_k)] = relD
        dict_out['{}_d_isrel'.format(behavior_k)] = isrelD

    basedir, _ = os.path.split(preproc_path)
    savename = os.path.join(basedir, 'eyehead_revcorrs_v5.h5')
    print('  -> Writing {}'.format(savename))
    fm2p.write_h5(savename, dict_out)


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-path', '--path', type=str, default=None)
    # args = parser.parse_args()

    # preproc_path = '/home/dylan/Storage/freely_moving_data/_V1PPC/cohort01_recordings/250616_DMM_DMM042_pos13/fm2/250616_DMM_DMM042_fm_02_preproc.h5'

    # eyehead_revcorr(preproc_path)

    # batch processing
    all_fm_preproc_files = fm2p.find('*DMM*fm*preproc.h5', '/home/dylan/Storage/freely_moving_data/_V1PPC')
    
    for f in tqdm(all_fm_preproc_files):
        # print('Analyzing {}'.format(f))
        fm2p.eyehead_revcorr(f)
