# -*- coding: utf-8 -*-
"""
Fit the linear-nonlinear-Poisson model to neural/behavior data.

Functions
---------
fit_model(cfg_path=None)
    Fit the linear-nonlinear-Poisson model to neural/behavior data.
    
Example usage
-------------
    $ python -m fm2p.fit_model.py -cfg config.yaml
or alternatively, leave out the -cfg flag and select the config file from a file dialog box.
    $ python -m fm2p.fit_model.py

Authors: DMM, 2024
"""


import os
import argparse
import numpy as np
from tqdm import tqdm

import fm2p


def fit_simple_GLM(cfg, opts, inds=None):

    print('  -> Analyzing data for config file: {}'.format(cfg['spath']))
    print('  -> Using Model 2 (simple GLM)')

    reclist = cfg['include_recordings']

    all_glm_fit_results = []

    for rname in reclist:

        h5_path = cfg['{}_preproc_file'.format(rname)]
        print('  -> Reading {}'.format(h5_path))
        data = fm2p.read_h5(h5_path)

        rec_dir = os.path.split(h5_path)[0]

        pupil = data['pupil_from_head'].copy()
        retinocentric = data['retinocentric'].copy()
        egocentric = data['egocentric'].copy()
        speed = data['speed'].copy()
        spikes = data['norm_spikes'].copy().T

        if inds is not None:

            if len(inds) != np.size(spikes,1):
                inds_ = np.zeros(np.size(spikes,1))
                inds_[inds] = 1
                inds = inds_.copy().astype(bool)

            spikes = spikes[:,inds]
            

        glm_fit_results = fm2p.fit_pred_GLM(spikes, pupil, retinocentric, egocentric, speed, opts=opts)

        savepath = os.path.join(rec_dir, 'GLM2_fit_results.h5')
        print('Saving GLM results to {}'.format(savepath))
        fm2p.write_h5(savepath, glm_fit_results)

        all_glm_fit_results.append(glm_fit_results)

    return all_glm_fit_results


def fit_LNLP(cfg):
    """ Fit the linear-nonlinear-Poisson model to neural/behavior data.

    'Model 1'

    Parameters
    ----------
    cfgh : dict
        Config dictionary.
    """

    if cfg is None:
        cfg = fm2p.make_default_cfg()

    print('  -> Analyzing data for config file: {}'.format(cfg['spath']))
    print('  -> Using Model 1 (Hardcastle LNLP)')

    recording_names = fm2p.list_subdirs(cfg['spath'], givepath=False)
    num_recordings = len(recording_names)

    if (type(cfg['include_recordings']) == list) and (len(cfg['include_recordings']) > 0):
        num_specified_recordings = len(cfg['include_recordings'])
    elif (type(cfg['include_recordings']) == list) and (len(cfg['include_recordings']) == 0):
        num_specified_recordings = num_recordings
    else:
        print('Issue determining how many recordings were specified.')
        num_specified_recordings = -1

    if num_recordings != num_specified_recordings:
        recording_names = [x for x in recording_names if x in cfg['include_recordings'] and 'sn' not in x]

    A_bins = np.linspace(-15, 15, 10) # theta
    B_bins = np.linspace(-15, 15, 10) # phi
    C_bins = np.linspace(-60, 60, 12) # dTheta
    D_bins = np.linspace(-60, 60, 12) # dPhi

    var_bins = [A_bins, B_bins, C_bins, D_bins]

    # Iterate through each recording (only if it is specified in the cfg file, ignore the rest).
    for rnum, rname in enumerate(recording_names):

        print('  -> Fitting model for {}.'.format(rname))
        
        print('  -> Reading preprocessed data.')

        h5_path = cfg['{}_preproc_file'.format(rname)]
        data = fm2p.read_h5(h5_path)

        rec_dir = os.path.split(h5_path)[0]

        print('  -> Interpolating time and setting up arrays.')

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

        theta = data['theta_interp'].copy()
        phi = data['phi'].copy()

        # interpolate dEye values to twop data
        dTheta = fm2p.interp_short_gaps(data['dTheta'])
        dTheta = fm2p.interpT(dTheta, data['eyeT1'], data['twopT'])
        dPhi = fm2p.interp_short_gaps(data['dPhi'])
        dPhi = fm2p.interpT(dPhi, data['eyeT1'], data['twopT'])

        spikes = data['norm_spikes'].copy()
        ltdk = data['ltdk_state_vec'].copy()
        speed = data['speed'].copy()

        # Apply lag before dropping stationary frames
        # *** FOR NOW, JUST USE LIGHT PERIOD ***
        speed = np.append(speed, speed[-1])
        use = (speed > cfg['speed_thresh']) * ltdk

        # Bin behavior data into variable maps. At the same time, be sure to drop the
        # stationary periods.
        mapA = fm2p.make_varmap(theta[use], A_bins)
        mapB = fm2p.make_varmap(phi[use], B_bins)
        mapC = fm2p.make_varmap(dTheta[use], C_bins)
        mapD = fm2p.make_varmap(dPhi[use], D_bins)
        var_maps = [mapA, mapB, mapC, mapD]

        if cfg['compute_model_performance']:
            for lag_val in cfg['lags']:

                spiketrains = np.zeros([
                    np.size(spikes,0),
                    np.sum(use)
                ]) * np.nan

                # Apply time lag
                # Want to shift the spike train forwards so that behavior precedes neural
                # activity, so sign of lag_frames should be positive. Then, once lag is applied,
                # drop frames that are stationary in the behavior data.
                for cell_i in range(np.size(spikes,0)):
                    spiketrains[cell_i,:] = np.roll(spikes[cell_i,:], shift=lag_val)[use]

                lagstr = str(lag_val)
                if '-' in lagstr:
                    lagstr = lagstr.replace('-','neg')
                else:
                    lagstr = 'pos{}'.format(lagstr)
                model_name = '{}_lag_{}'.format(cfg['model_save_key'], lagstr)
                model_save_path = os.path.join(rec_dir, model_name)
                if not os.path.isdir(model_save_path):
                    os.mkdir(model_save_path)

                print('  -> Fiting model {}.'.format(model_name))

                model_results = fm2p.fit_all_LNLP_models(
                    var_maps,
                    var_bins,
                    spiketrains,
                    savedir=model_save_path
                )

        # Calculate the model fits for the null spikes (rolled random temporal distances from
        # ground truth). Do I need to calculate a different null distribution across rolls? Probably
        # not. 
        if cfg['compute_null_performance']:

            spiketrains = np.zeros([np.size(spikes,0), np.sum(use)]) * np.nan

            # Add a random temporal roll to the spike data, with a size somewhere
            # from 15% to 85% of the length of the recording.
            num_timebins = np.size(spiketrains, axis=1)
            set_low_dist = int(np.round(num_timebins * 0.15))
            set_high_dist = int(np.round(num_timebins * 0.85))

            for cell_i in range(np.size(spiketrains,0)):

                # Determine a different roll distance for each cell
                shift_dist = np.random.randint(
                    low=set_low_dist,
                    high=set_high_dist,
                    size=1
                )

                # Apply the roll
                rolled_spikes = spikes[cell_i,use].copy()
                spiketrains[cell_i,:] = np.roll(rolled_spikes, shift=shift_dist)

            # Make the directories
            null_name = '{}_null'.format(cfg['model_save_key'])
            null_save_path = os.path.join(rec_dir, null_name)
            if not os.path.isdir(null_save_path):
                os.mkdir(null_save_path)

            # Fit the models
            print('  -> Fitting model {} (null data has spikes rolled a random distance).'.format(null_name))
            model_results = fm2p.fit_all_LNLP_models(
                var_maps,
                var_bins,
                spiketrains,
                savedir=null_save_path
            )


def fit_model():

    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--cfg', type=str, default=None)
    parser.add_argument('-v', '--model_version', type=str, default='LNLP')
    args = parser.parse_args()

    # if args.model_version is None:
    #     modver = fm2p.get_string_input('Which model version should be fit? (enter an str)')
    # elif args.model_version is not None:
    #     modver = args.model_version

    modver = args.model_version
    
    if (cfg is None) and (modver == 'LNLP' or modver == 'simpleGLM'):

        if args.cfg is None:
            cfg_path = fm2p.select_file(
                title='Select config yaml file.',
                filetypes=[('YAML','.yaml'),('YML','.yml'),]
            )
        elif args.cfg is not None:
            cfg_path = args.cfg
            
        cfg = fm2p.read_yaml(cfg_path)


    if modver == 'LNLP':
    
        fit_LNLP(cfg)


    elif modver == 'simpleGLM':

        opts = {
            'learning_rate': 0.1,
            'epochs': 3000,
            'l1_penalty': 0.01,
            'l2_penalty': 0.01,  # 0.01 was used in tweedie regresssor
            'num_lags': 30,  # 30 lags is access to 4.0 seconds
            'multiprocess': False
        }

        fit_simple_GLM(cfg, opts, inds=np.arange(15))


    elif modver == 'mcGLM':

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

            print('  -> Fitting model for {} recording ({}/{}).'.format(rec, rec_i+1, n_rec))

            preproc_path = fm2p.find('*_preproc.h5', os.path.join(cfg['spath'], rec), MR=True)
        
            fm2p.fit_multicell_GLM(preproc_path)


    elif modver == 'mcGLM_all':

        preproc_path = fm2p.select_file(
            'Select preprocessed file.',
            filetypes=[('HDF','.h5'),]
        )

        dict_out = fm2p.run_all_GLMs(preproc_path)

        return dict_out


if __name__ == '__main__':

    fit_model()

    # for batch processing....
    # preproc_paths = fm2p.find('*fm*_preproc.h5', '/home/dylan/Storage/freely_moving_data/_V1PPC/cohort01_recordings')
    # print('Found {} recordings.'.format(len(preproc_paths)))
    # for preproc_path in tqdm(preproc_paths):
    #     _ = fm2p.run_all_GLMs(preproc_path)

