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


def fit_model(cfg_path=None):
    """ Fit the linear-nonlinear-Poisson model to neural/behavior data.

    Parameters
    ----------
    cfg_path : str, optional
        Path to the config file. If None, a file dialog will be opened to select the config file.
    """
    
    if cfg_path is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('-cfg', '--cfg', type=str, default=None)
        args = parser.parse_args()

    if args.cfg is None:
        print('Select config yaml file.')
        cfg_path = fm2p.select_file(
            title='Select config yaml file.',
            filetypes=[('YAML','.yaml'),('YML','.yml'),]
        )
    elif args.cfg is not None:
        cfg_path = args.cfg

    cfg = fm2p.read_yaml(cfg_path)

    print('  -> Analyzing data for config file: {}'.format(cfg_path))

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


if __name__ == '__main__':

    fit_model()

