"""
fm2p/preprocess.py
Preprocess recording, convering raw data to a set of several .h5 files.

DMM, 2025
"""


import os
import argparse
import numpy as np
import yaml
import PySimpleGUI as sg

sg.theme('Default1')

import fm2p


def preprocess(cfg_path=None, spath=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--cfg', type=str, default=None)
    args = parser.parse_args()

    if args.cfg is not None:
        cfg_path = args.cfg

    # NOTE: each recording directory should have no subdirectories / be completely flat. the session
    # directory should have one directory per recording and no other subdirectories that are not
    # independent recordings

    # Read in the config file
    # If no config path, use defaults but ignore the spath from default
    if (cfg_path is None) and (spath is not None):
        internals_config_path = os.path.join(fm2p.up_dir(__file__, 1), 'utils/internals.yaml')
        with open(internals_config_path, 'r') as infile:
            cfg = yaml.load(infile, Loader=yaml.FullLoader)
        cfg['spath'] = spath
    elif (cfg_path is None) and (spath is None):
        print('Choose config yaml file.')
        cfg_path = sg.popup_get_file(
            'Choose config yaml file.', 'Choose config yaml file.',
            initial_folder='K:/', no_window=True, keep_on_top=True,
            file_types=(('YAML', '*.yaml'),('YML', '*.yml'),)
        )
        with open(cfg_path, 'r') as infile:
            cfg = yaml.load(infile, Loader=yaml.FullLoader)
    elif (cfg_path is not None):
        with open(cfg_path, 'r') as infile:
            cfg = yaml.load(infile, Loader=yaml.FullLoader)

    # Find the number of recordings in the session
    # Every folder in the session directory is assumed to be a recording
    recording_names = fm2p.list_subdirs(cfg['spath'], givepath=False)
    num_recordings = len(recording_names)

    # Check recordings against list of included recordings. If the list is empty, analyze all.
    # Otherwise, only do the ones listed.
    if (type(cfg['include_recordings']) == list) and (len(cfg['include_recordings']) > 0):
        num_specified_recordings = len(cfg['include_recordings'])
    elif (type(cfg['include_recordings']) == list) and (len(cfg['include_recordings']) == 0):
        num_specified_recordings = num_recordings
    else:
        print('Issue determining how many recordings were specified.')
        num_specified_recordings = -1

    # Apply directory exclusion
    if num_recordings != num_specified_recordings:
        recording_names = [x for x in recording_names if x in cfg['include_recordings']]

    for rnum, rname in enumerate(recording_names):

        # Recording path
        rpath = os.path.join(cfg['spath'], rname)

        print('  -> Analyzing {}.'.format(rpath))
        print('  -> Finding files.')

        # Eye camera files
        eyecam_raw_video = fm2p.find('*_eyecam.avi', rpath, MR=True)
        eyecam_TTL_voltage = fm2p.find('*_logTTL.csv', rpath, MR=True)
        eyecam_TTL_timestamps = fm2p.find('*_ttlTS.csv', rpath, MR=True)
        eyecam_video_timestamps = fm2p.find('*_eyecam.csv', rpath, MR=True)

        # Topdown camera files
        topdown_video = fm2p.find('*00*.mp4', rpath, MR=True)

        # Suite2p files
        F_path = fm2p.find('F.npy', rpath, MR=True)
        Fneu_path = fm2p.find('Fneu.npy', rpath, MR=True)
        suite2p_spikes = fm2p.find('spks.npy', rpath, MR=True)
        suite2p_stat_path = fm2p.find('stat.npy', rpath, MR=True)
        suite2p_ops_path = fm2p.find('ops.npy', rpath, MR=True)
        iscell_path = fm2p.find('iscell.npy', rpath, MR=True)

        if cfg['run_deinterlace']:

            print('  -> Rotating and deinterlacing eye camera video.')

            # Deinterlace and rotate eyecam video
            eyecam_deinter_video = fm2p.deinterlace(eyecam_raw_video, do_rotation=True)

        if ('eyecam_deinter_video' not in vars()) and ('eyecam_deinter_video' not in globals()):
            eyecam_deinter_video = eyecam_raw_video = fm2p.find('*_eyecam_deinter.avi', rpath, MR=True)

        if cfg['run_pose_estimation']:

            print('  -> Running pose estimation for eye camera video.')

            # Run dlc on eyecam video
            fm2p.run_pose_estimation(
                eyecam_deinter_video,
                project_cfg=cfg['eye_DLC_project'],
                filter=False
            )

            # print('  -> Running pose estimation for topdown camera video.')

            # fm2p.run_pose_estimation(
            #     topdown_video,
            #     project_cfg=cfg['top_DLC_project'],
            #     filter=False
            # )

        # Find dlc files
        eyecam_pts_path = fm2p.find('*_eyecam_deinterDLC_resnet50_*freely_moving_eyecams*.h5', rpath, MR=True)
        topdown_pts_path = fm2p.find('*DLC_resnet50_*freely_moving_topdown*.h5', rpath, MR=True)

        print('  -> Reading fluorescence data.')

        # Read suite2p data
        F = np.load(F_path)
        Fneu = np.load(Fneu_path)
        spks = np.load(suite2p_spikes)
        # stat = np.load(suite2p_stat_path, allow_pickle=True)
        # ops =  np.load(suite2p_ops_path, allow_pickle=True)
        iscell = np.load(iscell_path)

        # Create recording name
        # 250218_DMM_DMM038_rec_01_eyecam.avi
        full_rname = '_'.join(eyecam_raw_video.split('_')[:-1])

        print('  -> Measuring pupil orientation via ellipse fit.')

        # Pupil tracking
        reye_cam = fm2p.Eyecam(rpath, full_rname)
        reye_cam.add_files(
            eye_dlc_h5=eyecam_pts_path,
            eye_avi=eyecam_deinter_video,
            eyeT=eyecam_video_timestamps
        )
        eye_xyl, ellipse_dict = reye_cam.track_pupil()
        eyevid_arr = fm2p.pack_video_frames(reye_cam.eye_avi)
        eye_preproc_path = reye_cam.save_tracking(ellipse_dict, eye_xyl, eyevid_arr)

        print('  -> Measuring locomotor behavior.')

        # Topdown behavior and obstacle/arena tracking
        top_cam = fm2p.Topcam(rpath, full_rname)
        top_cam.add_files(
            top_dlc_h5=topdown_pts_path,
            top_avi=topdown_video
        )
        top_xyl, top_tracking_dict = top_cam.track_body()
        arena_coords, pillar_coords, pillar_fit = top_cam.track_arena()
        topvid_arr = fm2p.pack_video_frames(top_cam.top_avi)
        top_preproc_path = top_cam.save_tracking(
            top_tracking_dict, top_xyl, topvid_arr,
            arena_coords, pillar_coords, pillar_fit)

        print('  -> Running spike inference.')

        # Load processed two photon data from suite2p
        twop = fm2p.TwoP
        twop.add_files(F, Fneu, spks, iscell)
        twop_dict = twop.calc_dFF(neu_correction=0.7)
        twop_preproc_path = twop.save_fluor(twop_dict)

        print('  ->Aligning eye camera data streams to 2P and behavior data using TTL voltage.')

        eyeStart, eyeEnd = fm2p.align_eyecam_using_TTL(
            eye_dlc_h5=eyecam_pts_path,
            eye_TS_csv=eyecam_video_timestamps,
            eye_TTLV_csv=eyecam_TTL_voltage,
            eye_TTLTS_csv=eyecam_TTL_timestamps,
        )

        # If a real config path was given, write some new data into the dictionary and then overwrite it
        if cfg_path is not None:

            temp_dict = {}

            temp_dict['rpath'] = os.path.join()
            temp_dict['top_preproc_path'] = top_preproc_path
            temp_dict['twop_preproc_path'] = twop_preproc_path
            temp_dict['eye_preproc_path'] = eye_preproc_path
            temp_dict['eyecam_raw_video'] = eyecam_raw_video
            temp_dict['eyecam_TTL_voltage'] = eyecam_TTL_voltage
            temp_dict['eyecam_TTL_timestamps'] = eyecam_TTL_timestamps
            temp_dict['eyecam_video_timestamps'] = eyecam_video_timestamps
            temp_dict['topdown_video'] = topdown_video
            temp_dict['eyecam_deinter_video'] = eyecam_deinter_video
            temp_dict['eyecam_pts_path'] = eyecam_pts_path
            temp_dict['topdown_pts_path'] = topdown_pts_path
            temp_dict['eyeT_startInd'] = eyeStart
            temp_dict['eyeT_endInd'] = eyeEnd
            temp_dict['F_path'] = F_path
            temp_dict['Fneu_path'] = Fneu_path
            temp_dict['suite2p_spikes'] = suite2p_spikes
            temp_dict['suite2p_stat_path'] = suite2p_stat_path
            temp_dict['suite2p_ops_path'] = suite2p_ops_path
            temp_dict['iscell_path'] = iscell_path

            cfg[rname] = temp_dict


    # If a real config file path was given, write the updated config file to a new path
    if cfg_path is not None:

        print('  -> Updating config yaml file.')

        # Write a new version of the config file. Maybe change this to overwrite previous?
        _newsavepath = os.path.join(os.path.split(cfg_path)[0], 'preprocessed_config.yaml')
        with open(_newsavepath, 'w') as outfile:
            yaml.dump(cfg, outfile, default_flow_style=False)


# # Deinterlace worldcam
# if has_worldcam:
#     # raw_worldvid_path = fm2p.find('{}*world.avi'.format(rec_name), rec_path, MR=True)
#     world_deinter_vid = fm2p.deinterlace(raw_worldvid_path)
#     _ = fm2p.undistort_video(world_deinter_vid, worldcam_distortion_mtx_path)


if __name__ == '__main__':

    preprocess()

