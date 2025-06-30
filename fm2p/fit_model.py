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

        savepath = os.path.join(rec_dir, 'GLM_fit_results.h5')
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

    reclist = cfg['include_recordings']

    ego_bins = np.linspace(-180, 180, 19)
    retino_bins = np.linspace(-180, 180, 19) # 20 deg bins
    pupil_bins = np.linspace(45, 95, 11) # 5 deg bins

    var_bins = [pupil_bins, retino_bins, ego_bins]

    # Iterate through each recording (only if it is specified in the cfg file, ignore the rest).
    for rname in reclist:

        print('  -> Fitting model for {}.'.format(rname))
        
        print('  -> Reading preprocessed data.')

        h5_path = cfg['{}_preproc_file'.format(rname)]
        data = fm2p.read_h5(h5_path)

        rec_dir = os.path.split(h5_path)[0]

        print('  -> Interpolating time and setting up arrays.')

        # Pupil position relative to the animal's head. 0deg is looking forward parallel to nose
        pupil = data['pupil_from_head'].copy()
        speed = data['speed'].copy()
        egocentric = data['egocentric'].copy()
        retinocentric = data['retinocentric'].copy()
        spikes = data['oasis_spks'].copy()

        # Apply lag BEFORE dropping stationary frames
        speed = np.append(speed, speed[-1])
        use = speed > cfg['speed_thresh']

        # Bin behavior data into variable maps. At the same time, be sure to drop the
        # stationary periods.
        mapP = fm2p.make_varmap(pupil[use], pupil_bins)
        mapR = fm2p.make_varmap(retinocentric[use], retino_bins, circ=True)
        mapE = fm2p.make_varmap(egocentric[use], ego_bins, circ=True)
        var_maps = [mapP, mapR, mapE]

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
        if cfg['compute_null_model_performance']:

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
    parser.add_argument('-v', '--model_version', type=int, default=None)
    args = parser.parse_args()

    if args.cfg is None:
        print('Select config yaml file.')
        cfg_path = fm2p.select_file(
            title='Select config yaml file.',
            filetypes=[('YAML','.yaml'),('YML','.yml'),]
        )
    elif args.cfg is not None:
        cfg_path = args.cfg

    if args.model_version is None:
        print('Which model version should be fit? (enter an integer)')
        modver = fm2p.get_string_input('Which model version should be fit? (enter an integer)')
    elif args.model_version is not None:
        modver = args.model_version

    cfg = fm2p.read_yaml(cfg_path)


    if modver == 1:
        fit_LNLP(cfg)


    elif modver == 2:

        opts = {
            'learning_rate': 0.1,
            'epochs': 3000,
            'l1_penalty': 0.01,
            'l2_penalty': 0.01,  # 0.01 was used in tweedie regresssor
            'num_lags': 4,
            'multiprocess': True
        }

        fit_simple_GLM(cfg, opts, inds=np.arange(15))


if __name__ == '__main__':

    fit_model()

    # opts = {
    #     'learning_rate': 0.1,
    #     'epochs': 3000,
    #     'l1_penalty': 0.01,
    #     'l2_penalty': 0.01,  # 0.01 was used in tweedie regresssor
    #     'num_lags': 4,
    #     'multiprocess': True
    # }

    # cfg_path = r'K:\Mini2P\250306_DMM_DMM038_pillar\preprocessed_config.yaml'
    # cfg = fm2p.read_yaml(cfg_path)
    # all_glm_fit_results = fm2p.fit_simple_GLM(cfg, opts, inds=np.arange(15))
    # glmdata = all_glm_fit_results[0]
