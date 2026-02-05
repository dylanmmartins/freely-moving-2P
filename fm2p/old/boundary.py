# -*- coding: utf-8 -*-


import argparse
import os

import warnings
warnings.filterwarnings('ignore')

import fm2p


def boundary():

    skip_classification = True

    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--cfg', type=str, default=None)
    args = parser.parse_args()
    cfg_path = args.cfg

    # cfg_path = r'K:\Mini2P\250630_DMM_DMM037_ltdk\config.yaml'

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

        preproc_data = fm2p.read_h5(preproc_path)
        savedir = os.path.split(preproc_path)[0]
        basename = os.path.split(preproc_path)[1][:-11]

        saveflag = 'p180'
        v = 8
        light_head_savepath = os.path.join(savedir, '{}_boundary_tuning_ltego_v{}_{}.h5'.format(basename, v, saveflag))
        dark_head_savepath = os.path.join(savedir, '{}_boundary_tuning_dkego_v{}_{}.h5'.format(basename, v, saveflag))
        light_eye_savepath = os.path.join(savedir, '{}_boundary_tuning_ltret_v{}_{}.h5'.format(basename, v, saveflag))
        dark_eye_savepath = os.path.join(savedir, '{}_boundary_tuning_dkret_v{}_{}.h5'.format(basename, v, saveflag))
        light_ego_savepath = os.path.join(savedir, '{}_boundary_tuning_ltegopillar_v{}_{}.h5'.format(basename, v, saveflag))
        dark_ego_savepath = os.path.join(savedir, '{}_boundary_tuning_dkegopillar_v{}_{}.h5'.format(basename, v, saveflag))
        light_ret_savepath = os.path.join(savedir, '{}_boundary_tuning_ltretpillar_v{}_{}.h5'.format(basename, v, saveflag))
        dark_ret_savepath = os.path.join(savedir, '{}_boundary_tuning_dkretpillar_v{}_{}.h5'.format(basename, v, saveflag))

        print('  -> Starting to analyze egocentric tuning in light condition.')
        lthead_bt = fm2p.BoundaryTuning(preproc_data)
        lthead_bt.identify_responses(use_angle='egow', use_light=True, skip_classification=skip_classification)
        print('Writing {}'.format(os.path.split(light_head_savepath)[1]))
        lthead_bt.save_results(light_head_savepath)
        del lthead_bt

        print('  -> Starting to analyze egocentric tuning in dark condition.')
        dkhead_bt = fm2p.BoundaryTuning(preproc_data)
        dkhead_bt.identify_responses(use_angle='egow', use_dark=True, skip_classification=skip_classification)
        print('Writing {}'.format(os.path.split(dark_head_savepath)[1]))
        dkhead_bt.save_results(dark_head_savepath)
        del dkhead_bt

        print('  -> Starting to analyze retinocentric tuning in light condition.')
        ltret_bt = fm2p.BoundaryTuning(preproc_data)
        ltret_bt.identify_responses(use_angle='pupil', use_light=True, skip_classification=skip_classification)
        print('Writing {}'.format(os.path.split(light_eye_savepath)[1]))
        ltret_bt.save_results(light_eye_savepath)
        del ltret_bt

        print('  -> Starting to analyze retinocentric tuning in dark condition.')
        dkret_bt = fm2p.BoundaryTuning(preproc_data)
        dkret_bt.identify_responses(use_angle='pupil', use_dark=True, skip_classification=skip_classification)
        print('Writing {}'.format(os.path.split(dark_eye_savepath)[1]))
        dkret_bt.save_results(dark_eye_savepath)
        del dkret_bt

        print('  -> Starting to analyze egocentric pillar tuning in light condition.')
        ltego_bt = fm2p.BoundaryTuning(preproc_data)
        ltego_bt.identify_responses(use_angle='egop', use_light=True, skip_classification=skip_classification)
        print('Writing {}'.format(os.path.split(light_ego_savepath)[1]))
        ltego_bt.save_results(light_ego_savepath)
        del ltego_bt

        print('  -> Starting to analyze egocentric pillar tuning in dark condition.')
        dkego_bt = fm2p.BoundaryTuning(preproc_data)
        dkego_bt.identify_responses(use_angle='egop', use_dark=True, skip_classification=skip_classification)
        print('Writing {}'.format(os.path.split(dark_ego_savepath)[1]))
        dkego_bt.save_results(dark_ego_savepath)
        del dkego_bt

        print('  -> Starting to analyze retinocentric pillar tuning in light condition.')
        ltret_bt = fm2p.BoundaryTuning(preproc_data)
        ltret_bt.identify_responses(use_angle='retino', use_light=True, skip_classification=skip_classification)
        print('Writing {}'.format(os.path.split(light_ret_savepath)[1]))
        ltret_bt.save_results(light_ret_savepath)
        del ltret_bt

        print('  -> Starting to analyze retinocentric pillar tuning in dark condition.')
        dkret_bt = fm2p.BoundaryTuning(preproc_data)
        dkret_bt.identify_responses(use_angle='retino', use_light=True, skip_classification=skip_classification)
        print('Writing {}'.format(os.path.split(dark_ret_savepath)[1]))
        dkret_bt.save_results(dark_ret_savepath)
        del dkret_bt


if __name__ == '__main__':

    boundary()

    