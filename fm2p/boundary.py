import argparse
import os


import fm2p


def boundary():

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

        preproc_data = fm2p.read_h5(preproc_path)
        savedir = os.path.split(preproc_path)[0]
        basename = os.path.split(preproc_path)[1][:-11]

        light_head_savepath = os.path.join(savedir, '{}_boundary_tuning_ltego'.format(basename))
        dark_head_savepath = os.path.join(savedir, '{}_boundary_tuning_dkego'.format(basename))
        light_eye_savepath = os.path.join(savedir, '{}_boundary_tuning_ltret'.format(basename))
        dark_eye_savepath = os.path.join(savedir, '{}_boundary_tuning_dkret'.format(basename))

        print('  -> Starting to analyze egocentric tuning in light condition.')
        ltego_bt = fm2p.BoundaryTuning(preproc_data)
        ltego_bt.identify_responses(use_angle='head', use_light=True)
        ltego_bt.save_results(light_head_savepath)
        del ltego_bt

        print('  -> Starting to analyze egocentric tuning in dark condition.')
        dkego_bt = fm2p.BoundaryTuning(preproc_data)
        dkego_bt.identify_responses(use_angle='head', use_dark=True)
        dkego_bt.save_results(dark_head_savepath)
        del dkego_bt

        print('  -> Starting to analyze retinocentric tuning in light condition.')
        ltret_bt = fm2p.BoundaryTuning(preproc_data)
        ltret_bt.identify_responses(use_angle='eye', use_light=True)
        ltret_bt.save_results(light_eye_savepath)
        del ltret_bt

        print('  -> Starting to analyze retinocentric tuning in dark condition.')
        dkret_bt = fm2p.BoundaryTuning(preproc_data)
        dkret_bt.identify_responses(use_angle='eye', use_dark=True)
        dkret_bt.save_results(dark_eye_savepath)
        del dkret_bt

if __name__ == '__main__':

    boundary()