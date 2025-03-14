"""
fm2p/preprocess.py
Preprocess recording, convering raw data to a set of several .h5 files.

DMM, 2025
"""


import os
import argparse
import numpy as np
import yaml

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
        cfg = fm2p.read_yaml(internals_config_path)
        cfg['spath'] = spath

    elif (cfg_path is None) and (spath is None):
        print('Choose config yaml file.')
        cfg_path = fm2p.select_file(
            title='Choose config yaml file.',
            filetypes=[('YAML', '*.yaml'),('YML', '*.yml'),]
        )
        cfg = fm2p.read_yaml(cfg_path)

    elif (cfg_path is not None):
        cfg = fm2p.read_yaml(cfg_path)

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
        possible_topdown_videos = fm2p.find('*.mp4', rpath, MR=False)
        topdown_video = fm2p.filter_file_search(possible_topdown_videos, toss=['labeled','resnet50'], MR=True)

        # Suite2p files
        F_path = fm2p.find('F.npy', rpath, MR=True)
        Fneu_path = fm2p.find('Fneu.npy', rpath, MR=True)
        suite2p_spikes = fm2p.find('spks.npy', rpath, MR=True)
        iscell_path = fm2p.find('iscell.npy', rpath, MR=True)

        if cfg['run_deinterlace']:

            print('  -> Rotating and deinterlacing eye camera video.')

            # Deinterlace and rotate eyecam video
            eyecam_deinter_video = fm2p.deinterlace(eyecam_raw_video, do_rotation=True)

        if ('eyecam_deinter_video' not in vars()) and ('eyecam_deinter_video' not in globals()):
            eyecam_deinter_video = fm2p.find('*_eyecam_deinter.avi', rpath, MR=True)

        if cfg['run_pose_estimation']:

            print('  -> Running pose estimation for eye camera video.')

            # Run dlc on eyecam video
            fm2p.run_pose_estimation(
                eyecam_deinter_video,
                project_cfg=cfg['eye_DLC_project'],
                filter=False
            )

            print('  -> Running pose estimation for topdown camera video.')

            fm2p.run_pose_estimation(
                topdown_video,
                project_cfg=cfg['top_DLC_project'],
                filter=False
            )

        # Find dlc files
        eyecam_pts_path = fm2p.find('*_eyecam_deinterDLC_resnet50_*freely_moving_eyecams*.h5', rpath, MR=True)
        topdown_pts_path = fm2p.find('*DLC_resnet50_*trackbar-v5*.h5', rpath, MR=True)

        print('  -> Reading fluorescence data.')

        # Read suite2p data
        F = np.load(F_path, allow_pickle=True)
        Fneu = np.load(Fneu_path, allow_pickle=True)
        spks = np.load(suite2p_spikes, allow_pickle=True)
        # stat = np.load(suite2p_stat_path, allow_pickle=True)
        # ops =  np.load(suite2p_ops_path, allow_pickle=True)
        iscell = np.load(iscell_path, allow_pickle=True)

        # Create recording name
        # 250218_DMM_DMM038_rec_01_eyecam.avi
        full_rname = '_'.join(os.path.split(eyecam_raw_video)[1].split('_')[:-1])

        print('  -> Measuring locomotor behavior.')

        # Topdown behavior and obstacle/arena tracking
        top_cam = fm2p.Topcam(rpath, full_rname, cfg=cfg)
        top_cam.add_files(
            top_dlc_h5=topdown_pts_path,
            top_avi=topdown_video
        )
        top_xyl, top_tracking_dict = top_cam.track_body()
        arena_dict = top_cam.track_arena()

        print('  -> Aligning eye camera data streams to 2P and behavior data using TTL voltage.')

        eyeStart, eyeEnd = fm2p.align_eyecam_using_TTL(
            eye_dlc_h5=eyecam_pts_path,
            eye_TS_csv=eyecam_video_timestamps,
            eye_TTLV_csv=eyecam_TTL_voltage,
            eye_TTLTS_csv=eyecam_TTL_timestamps
        )
        eyeStart = int(eyeStart)
        eyeEnd = int(eyeEnd)

        print('  -> Measuring pupil orientation via ellipse fit.')

        # Pupil tracking
        reye_cam = fm2p.Eyecam(rpath, full_rname, cfg=cfg)
        reye_cam.add_files(
            eye_dlc_h5=eyecam_pts_path,
            eye_avi=eyecam_deinter_video,
            eyeT=eyecam_video_timestamps
        )
        eye_xyl, ellipse_dict = reye_cam.track_pupil()

        if cfg['run_cyclotorsion']:
            cyclotorsion_dict = reye_cam.measure_cyclotorsion(
                ellipse_dict,
                eyecam_deinter_video,
                startInd=eyeStart,
                endInd=eyeEnd,
                usemp=True,
                doVideo=False
            )
        else:
            cyclotorsion_dict = {}


        print('  -> Running spike inference.')

        # Load processed two photon data from suite2p
        twop_recording = fm2p.TwoP(rpath, full_rname, cfg=cfg)
        twop_recording.add_data(
            F=F,
            Fneu=Fneu,
            spikes=spks,
            iscell=iscell
        )
        twop_dict = twop_recording.calc_dFF(neu_correction=0.7)

        twop_dt = 1./cfg['twop_rate']
        twopT = np.arange(0, np.size(twop_dict['s2p_spks'], 1)*twop_dt, twop_dt)
        twop_dict['twopT'] = twopT


        print('  -> Calculating retinocentric and egocentric orientations.')

        # All values in units of pixels or degrees (not cm or rads)
        learx = top_tracking_dict['lear_x']
        leary = top_tracking_dict['lear_y']
        rearx = top_tracking_dict['rear_x']
        reary = top_tracking_dict['rear_y']
        yaw = top_tracking_dict['head_yaw_deg']
        pillarx = arena_dict['pillar_centroid']['x']
        pillary = arena_dict['pillar_centroid']['y']
        theta = np.rad2deg(ellipse_dict['theta'])

        headx = np.array([np.mean([rearx[f], learx[f]]) for f in range(len(rearx))])
        heady = np.array([np.mean([reary[f], leary[f]]) for f in range(len(reary))])

        eyeT = fm2p.read_timestamp_file(eyecam_video_timestamps, position_data_length=len(theta))
        theta_interp = fm2p.interpT(
            theta[eyeStart:eyeEnd],
            eyeT[eyeStart:eyeEnd] - eyeT[eyeStart],
            twopT
        )

        refframe_dict = fm2p.calc_reference_frames(
            cfg,
            headx,
            heady,
            yaw,
            theta_interp,
            arena_dict
        )

        print('  -> Saving preprocessed dataset to file.')

        preprocessed_dict = {
            **top_tracking_dict,
            **top_xyl.to_dict(),
            **arena_dict,
            **ellipse_dict,
            **eye_xyl.to_dict(),
            **twop_dict,
            **refframe_dict
        }

        preprocessed_dict['eyeT_startInd'] = eyeStart
        preprocessed_dict['eyeT_endInd'] = eyeEnd

        if len(cyclotorsion_dict.keys()) > 0:
            preprocessed_dict = {**preprocessed_dict, **cyclotorsion_dict}

        _savepath = os.path.join(rpath, '{}_preproc.h5'.format(full_rname))
        print('Writing preprocessed data to {}'.format(_savepath))
        fm2p.write_h5(_savepath, preprocessed_dict)

        # If a real config path was given, write some new data into the dictionary and then save a new preprocessed_config
        cfg['{}_preproc_file'.format(rname)] = _savepath
        cfg['{}_topdown_video'.format(rname)] = topdown_video
        cfg['{}_eye_video'.format(rname)] = eyecam_deinter_video


    if cfg_path is not None:

        print('  -> Updating config yaml file.')

        # Write a new version of the config file. Maybe change this to overwrite previous?
        _newsavepath = os.path.join(os.path.split(cfg_path)[0], 'preprocessed_config.yaml')
        fm2p.write_yaml(_newsavepath, cfg)



if __name__ == '__main__':

    preprocess()

