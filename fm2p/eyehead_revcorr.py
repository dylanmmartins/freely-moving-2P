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


def calc_reliability_over(spikes, behavior, n_cnk=10, n_shfl=100, relthresh=10, n_bins=13, bound=10):
    # calculate reliability based on overlap between null distribution and
    # real data, instead of old method of calculating `d`

    bins = np.linspace(
        np.nanpercentile(behavior, bound),
        np.nanpercentile(behavior, 100-bound),
        n_bins
    )

    n_cells = np.size(spikes, 0)
    n_frames = np.size(spikes, 1)

    cnk_sz = n_frames // n_cnk
    all_inds = np.arange(0, n_frames)

    tunings = np.zeros([
        2,  # state (true or null)
        n_shfl,
        2,  # split (first or second half)
        n_cells,
        np.size(bins) - 1
    ]) * np.nan

    correlations = np.zeros([
        2,
        n_shfl,
        n_cells
    ])

    # print('  -> Checking reliability across shuffles.')

    for state_i in range(2):

        # state 0 is the true data
        # state 1 is the null data / rolled spikes
        
        for shfl_i in range(n_shfl):
        
            np.random.seed(shfl_i)

            use_spikes = spikes.copy()

            if state_i == 1:
                # roll spikes a random distance relative to behavior
                roll_distance = np.random.randint(int(n_frames*0.10), int(n_frames*0.90))
                use_spikes = np.roll(use_spikes, roll_distance, axis=1)

            chunk_order = np.arange(n_cnk)
            np.random.shuffle(chunk_order)

            split1_inds = []
            split2_inds = []

            for cnk_i, cnk in enumerate(chunk_order[:(n_cnk//2)]):
                _inds = all_inds[(cnk_sz*cnk) : ((cnk_sz*(cnk+1)))]
                split1_inds.extend(_inds)

            for cnk_i, cnk in enumerate(chunk_order[(n_cnk//2):]):
                _inds = all_inds[(cnk_sz*cnk) : ((cnk_sz*(cnk+1)))]
                split2_inds.extend(_inds)

            # list of every index that goes into the two halves of the data
            split1_inds = np.array(np.sort(split1_inds)).astype(int)
            split2_inds = np.array(np.sort(split2_inds)).astype(int)

            if len(split1_inds)<1 or len(split2_inds)<1:
                print('No indices used for tuning reliability measure... len of usable recording was:')
                print(n_frames)

            _, tuning1, _ = fm2p.tuning_curve(
                use_spikes[:, split1_inds],
                behavior[split1_inds],
                bins
            )
            _, tuning2, _ = fm2p.tuning_curve(
                use_spikes[:, split2_inds],
                behavior[split2_inds],
                bins
            )

            tunings[state_i,shfl_i,0,:,:] = tuning1
            tunings[state_i,shfl_i,1,:,:] = tuning2

    correlations = np.zeros([
        n_shfl,
        2,    # state [true, null]
        n_cells
    ]) * np.nan

    tunings_masked = tunings.copy()
    tunings_masked[np.isnan(tunings_masked)] = 0

    for shfl_i in range(n_shfl):
        bin_mask = ~np.isnan(tunings[0,0,0,0,:])
        correlations[shfl_i,0,:] = [fm2p.corrcoef(tunings_masked[0,shfl_i,0,c,:], tunings_masked[0,shfl_i,1,c,:]) for c in range(n_cells)]
        correlations[shfl_i,1,:] = [fm2p.corrcoef(tunings_masked[1,shfl_i,0,c,:], tunings_masked[1,shfl_i,1,c,:]) for c in range(n_cells)]
    
    reliabilities = np.zeros(n_cells)
    for c in range(n_cells):
        # what percent of null shuffles fall above the median of true data?
        reliabilities[c] = np.sum(correlations[:,1,c] > np.median(correlations[:,0,c]))

    reliable_inds = reliabilities <= relthresh

    # dict_out = {
    #     'past_thresh_count': reliabilities,
    #     'reliable_inds': reliable_inds,
    #     'correlations': correlations,
    #     'tunings': tunings_masked
    # }

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
    savename = os.path.join(basedir, 'eyehead_revcorrs_v4cent.h5')
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
