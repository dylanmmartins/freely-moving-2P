# -*- coding: utf-8 -*-
"""
Reverse correlation analysis utilities for freely-moving 2P experiments.

This script provides functions to compute tuning reliability and modulation for neural data
using reverse correlation, for both standard and light/dark (ltdk) experiments.

Functions:
    calc_revcorr(preproc_path, axons=False):
        Compute tuning reliability and modulation for a single preprocessed file (no light/dark periods).
    calc_revcorr_ltdk(preproc_path, restrict_by_deviation=False):
        Compute tuning reliability and modulation for a single preprocessed file with light/dark periods.
    revcorr():
        Batch process all recordings specified in a config YAML file.

Example usage:
    $ python revcorr.py -cfg config.yaml

Author: DMM, last modified July 2025
"""


import os
import argparse
import numpy as np

import fm2p


def calc_revcorr(preproc_path, IMU=False, save=True):
    """
    Compute tuning reliability and modulation for a single preprocessed file with light/dark periods.

    Parameters
    ----------
    preproc_path : str
        Path to the preprocessed HDF5 file.
    """

    # Load preprocessed data
    if type(preproc_path) == str:
        data = fm2p.read_h5(preproc_path)
    elif type(preproc_path) == dict:
        data = preproc_path.copy()

    spikes = data['norm_spikes'].copy()

    theta = data['theta_interp'].copy()
    phi = data['phi_interp'].copy()

    retinocentric = data['retinocentric'].copy()
    egocentric = data['egocentric'].copy()

    distance = data['dist_to_pillar'].copy()
    cdistance = data['dist_to_center'].copy()
    pillar_size = data['pillar_size'].copy()
    yaw = data['head_yaw_deg'].copy()[:-1]
    # yaw = data['head_yaw_deg'].copy()

    speed = data['speed'].copy()
    # speed = np.append(speed, speed[-1])
    speeduse = speed > 1.5
    ltdk = data['ltdk_state_vec'].copy()

    twopT = data['twopT'].copy()

    if IMU:
        roll = data['roll_twop_interp'].copy()
        pitch = data['pitch_twop_interp'].copy()
        acc_x = data['acc_x_twop_interp'].copy()
        acc_y = data['acc_y_twop_interp'].copy()
        acc_z = data['acc_z_twop_interp'].copy()
        gyro_x = data['gyro_x_twop_interp'].copy()
        gyro_y = data['gyro_y_twop_interp'].copy()
        gyro_z = data['gyro_z_twop_interp'].copy()

    len_check = [
        np.size(spikes,1),
        np.size(theta),
        np.size(phi),
        np.size(retinocentric),
        np.size(egocentric),
        np.size(distance),
        np.size(cdistance),
        np.size(yaw),
        np.size(speed),
        np.size(ltdk),
        np.size(twopT)
    ]
    if len(set(len_check)) != 1:
        print('Fixing missing timepoint in topdown data.')
        speed = data['speed'].copy()
        speed = np.append(speed, speed[-1])
        speeduse = speed > 1.5

        yaw = data['head_yaw_deg'].copy()

        len_check = [
            np.size(spikes,1),
            np.size(theta),
            np.size(phi),
            np.size(retinocentric),
            np.size(egocentric),
            np.size(distance),
            np.size(cdistance),
            np.size(yaw),
            np.size(speed),
            np.size(ltdk),
            np.size(twopT)
        ]

        if len(set(len_check)) != 1:
            print('Extending by one addtl timepoint. If it still errors after this, check your data alignment.')
            # in case we need to do it one more time.
            # if this doesn't work, let it error because its a bigger problem
            try:
                yaw = np.append(yaw, yaw[-1])
                speed = np.append(speed, speed[-1])
            except:
                yaw = yaw[:-1]
                speed = speed[:-1]


    assert len(set(len_check)) == 1, 'Unequal lengths along time axis. Lenghts are {}'.format(len_check)

    lbound = 10
    ubound = 90

    retino_bins = np.linspace(-180, 180, 27)
    ego_bins = np.linspace(-180, 180, 27)
    yaw_bins = np.linspace(-180, 180, 27)
    theta_bins = np.linspace(
        np.nanpercentile(theta, lbound),
        np.nanpercentile(theta, ubound),
        13
    )
    phi_bins = np.linspace(
        np.nanpercentile(phi, lbound),
        np.nanpercentile(phi, ubound),
        13
    )
    dist_bins = np.linspace(
        np.nanpercentile(distance, lbound),
        np.nanpercentile(distance, ubound),
        13
    )
    cdist_bins = np.linspace(
        np.nanpercentile(cdistance, lbound),
        np.nanpercentile(cdistance, ubound),
        13
    )
    psize_bins = np.linspace(
        np.nanpercentile(pillar_size, lbound),
        np.nanpercentile(pillar_size, ubound),
        13
    )
    if IMU:
        roll_bins = np.linspace(
            np.nanpercentile(roll, lbound),
            np.nanpercentile(roll, ubound),
            13
        )
        pitch_bins = np.linspace(
            np.nanpercentile(pitch, lbound),
            np.nanpercentile(pitch, ubound),
            13
        )
        acc_x_bins = np.linspace(
            np.nanpercentile(acc_x, lbound),
            np.nanpercentile(acc_x, ubound),
            13
        )
        acc_y_bins = np.linspace(
            np.nanpercentile(acc_y, lbound),
            np.nanpercentile(acc_y, ubound),
            13
        )
        acc_z_bins = np.linspace(
            np.nanpercentile(acc_z, lbound),
            np.nanpercentile(acc_z, ubound),
            13
        )
        gyro_x_bins = np.linspace(
            np.nanpercentile(gyro_x, lbound),
            np.nanpercentile(gyro_x, ubound),
            13
        )
        gyro_y_bins = np.linspace(
            np.nanpercentile(gyro_y, lbound),
            np.nanpercentile(gyro_y, ubound),
            13
        )
        gyro_z_bins = np.linspace(
            np.nanpercentile(gyro_z, lbound),
            np.nanpercentile(gyro_z, ubound),
            13
        )

    vardict = {
        'yaw': {
            'vec': yaw,
            'bins': yaw_bins
        },
        'theta': {
            'vec': theta,
            'bins': theta_bins
        },
        'phi': {
            'vec': phi,
            'bins': phi_bins
        },
        'retinocentric': {
            'vec': retinocentric,
            'bins': retino_bins
        },
        'egocentric': {
            'vec': egocentric,
            'bins': ego_bins
        },
        'distance_to_pillar': {
            'vec': distance,
            'bins': dist_bins
        }
    }
    if IMU:
        imu_dict = {
            'roll': {
                'vec': roll,
                'bins': roll_bins
            },
            'pitch': {
                'vec': pitch,
                'bins': pitch_bins
            },
            'acc_x': {
                'vec': acc_x,
                'bins': acc_x_bins
            },
            'acc_y': {
                'vec': acc_y,
                'bins': acc_y_bins
            },
            'acc_z': {
                'vec': acc_z,
                'bins': acc_z_bins
            },
            'gyro_x': {
                'vec': gyro_x,
                'bins': gyro_x_bins
            },
            'gyro_y': {
                'vec': gyro_y,
                'bins': gyro_y_bins
            },
            'gyro_z': {
                'vec': gyro_z,
                'bins': gyro_z_bins
            }
        }
        vardict = {**vardict, **imu_dict}

    full_reliability_dict = {}


    for state in range(0,2):
        # 0 is dark condition, 1 is light condition
        state = bool(state)

        if state == 0:
            statename = 'dark'

            # if restrict_by_deviation:
            #     use1 = (~ltdk.copy()) * speeduse.copy() * use_thdev
            use0 = (~ltdk.copy()) * speeduse.copy()

        elif state == 1:
            statename = 'light'
        
            # if restrict_by_deviation:
            #     use1 = ltdk.copy() * speeduse.copy() * use_thdev
            use0 = ltdk.copy() * speeduse.copy()

        reliability_dict = {}

        print('  -> Analyzing {} periods.'.format(statename))

        for k,v in vardict.items():

            # if restrict_by_deviation and (v!='theta') and (v!='phi'):
            #     use = use1
            # else:
            #     use = use0
            use = use0

            print('  -> Calculating reliability for tuning to: {}'.format(k))

            behavior = v['vec']
            bins = v['bins']

            add_dict = fm2p.calc_reliability_d(
                spikes[:,use],
                behavior[use],
                bins,
                10,
                100,
                thresh=1.
            )

            tbins, tunings, errors = fm2p.tuning_curve(
                spikes[:,use],
                behavior[use],
                bins
            )
            add_dict['tuning_bins'] = tbins
            add_dict['tuning_curve'] = tunings
            add_dict['tuning_stderr'] = errors

            # In addition to the shufffle reliability metric, check the spectral slope of power across
            # frequencies; smooth curves will decay steeply while noisy curves will have a flat or shallow
            # decay. A clean curve will have a value of -2 to -5. Noisy will be -1 and above. Don't want to
            # be too strict, so I'm applying the threshold of -1.25, which seems to exclude the appropriate
            # curves while including clean responses.
            spec_val, spec_rel = fm2p.calc_spectral_noise(
                tunings,
                thresh=-1.25
            )
            add_dict['reliable_by_noise'] = spec_rel
            add_dict['spectral_noise'] = spec_val

            is_reliable = spec_rel.copy() * add_dict['reliable_by_shuffle'].copy()
            add_dict['is_reliable'] = is_reliable

            mod, is_modulated = fm2p.calc_multicell_modulation(
                tunings,
                spikes[:,speeduse],   # don't apply use, just speed
                0.33
            )

            add_dict['modulation'] = mod
            add_dict['is_modulated'] = is_modulated

            reliability_dict[k] = add_dict

        full_reliability_dict[statename] = reliability_dict

    # full_reliability_dict['behavior_inputs'] = vardict

    if save:
        savedir = os.path.split(preproc_path)[0]
        basename = os.path.split(preproc_path)[1][:-11]
        savepath = os.path.join(savedir, '{}_revcorr_results_v4.h5'.format(basename))
        fm2p.write_h5(savepath, full_reliability_dict)

        print('Saved {}'.format(savepath))

    return full_reliability_dict


def revcorr():
    """
    Batch process all recordings specified in a config YAML file.
    Loads config, finds all recordings, and runs revcorr analysis for each.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--cfg', type=str, default=None)
    args = parser.parse_args()
    cfg_path = args.cfg

    if cfg_path is None:
        cfg_path = fm2p.select_file(
            title='Choose a config yaml file.',
            filetypes=[('YAML', '*.yaml'),('YML', '*.yml'),]
        )

    cfg = fm2p.read_yaml(cfg_path)

    recording_names = fm2p.list_subdirs(cfg['spath'], givepath=False)
    num_recordings = len(recording_names)
    if (type(cfg['include_recordings']) == list) and (len(cfg['include_recordings']) > 0):
        num_specified_recordings = len(cfg['include_recordings'])
    elif (type(cfg['include_recordings']) == list) and (len(cfg['include_recordings']) == 0):
        num_specified_recordings = num_recordings

    if num_recordings != num_specified_recordings:
        recording_names = [x for x in recording_names if x in cfg['include_recordings']]

    n_rec = len(recording_names)

    for rec_i, rec in enumerate(recording_names):

        print('  -> Analyzing {} recording ({}/{}).'.format(rec, rec_i+1, n_rec))

        preproc_path = fm2p.find('*_preproc.h5', os.path.join(cfg['spath'], rec), MR=True)

        if cfg['ltdk']:

            _ = calc_revcorr(preproc_path, IMU=cfg['imu'])
        
        elif not cfg['ltdk']:

            calc_revcorr(preproc_path, axons=cfg['axons'])


if __name__ == '__main__':

    revcorr()

