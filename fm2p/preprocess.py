# -*- coding: utf-8 -*-
"""
Preprocessing pipeline for freely moving two-photon calcium imaging data.

Functions
---------
preprocess(cfg_path=None, spath=None)
    Preprocess recording, converting raw data to an .h5 file.

Example usage
-------------
    $ python -m fm2p.preprocess.py -cfg /path/to/config.yaml
or alternatively, you can run
    $ python -m fm2p.preprocess.py
which will prompt you to select a config file in a dialog box.


Each session should be structured so that each recording is in its own subdirectory
within the session directory. The config file should be in the session directory.
For each recording directory the required files for a somatic recording are:
    *_eyecam.avi
        Eye camera video, recoded as 30 Hz interlaced video.
    *_eyecam.csv
        Eye camera video times, where each is recorded as a decimal float value (i.e.,
        not a datetime).
    *_logTTL.csv
        Eye camera TTL voltage. This is the TTL voltage sent from Scanimage to
        the topdown camera and recorded on an arduino throuhg Bonsai to align the
        eye camera times with the rest of the data.
    *_ttlTS.csv
        Eye camera TTL timestamps. This is the time of each sampled TTL voltage sent from
        Scanimage to the topdown camera.
    *_topdown.mp4
        Top-down camera video, recorded at the same timebase as the scope data from Scanimage.
    F.npy
        Suite2p fluorescence matrix, where each row is a cell and each column is a frame.
        This is not in the main recording directory, but in the suite2p directory
        (i.e., /suite2p/plane0/F.npy).
    Fneu.npy
        Suite2p neuropul fluorescence matrix.
    spks.npy
        Suite2p spike matrix.
    iscell.npy
        Suite2p iscell matrix.
If the recording is an axonal recording, none of the suite2p files will be present and are not
required. Instead, the following file is required:
    *_denoised_SRA_data.mat
        This is the output of the denoising pipeline for axonal data. It is a .mat file
        containing the dF/F traces, spike times, and other information.
Once this code is run, it will create the following new files in the recording directory:
    *_eyecam_deinter.avi
        This is the deinterlaced and rotated eyecam video.
    *_eyecam_deinterDLC_resnet50_*freely_moving_eyecams_02*.h5
        This is the output of the DeepLabCut pipeline for the eyecam video.
    *_DLC_resnet50_*freely_moving_topdown_06*.h5
        This is the output of the DLC pipeline for the topdown video.
    *_preproc.h5
        This is the preprocessed data file containing all of the data from the recording.
    *_preproc_config.yaml
        This is the config file used to run the preprocessing pipeline. It contains
        all of the parameters used in the pipeline, including the paths to the input
        files and the output files.

Author: DMM, last updated May 2025
"""


import os
import argparse
import numpy as np

import fm2p


def preprocess(cfg_path=None, spath=None):
    """
    Preprocess recording, convering raw data to a set of several .h5 files.

    This function processes the raw data from two-photon calcium imaging recordings, including
    eyecam and topdown camera videos, and generates a set of preprocessed .h5 files. The preprocessing
    includes deinterlacing and rotating the eyecam video, running pose estimation on both the eyecam
    and topdown camera videos, aligning the data streams using TTL voltage, measuring pupil orientation,
    and calculating retinocentric and egocentric orientations. The script also runs spike inference on
    the two-photon data and saves the processed data to .h5 files.
    
    Parameters
    ----------
    cfg_path : str, optional
        Path to the config file. The default is None.
    spath : str, optional
        Path to the session directory. The default is None.
    
    Returns
    -------
    None
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--cfg', type=str, default=None)
    args = parser.parse_args()

    if args.cfg is not None:
        cfg_path = args.cfg

    # NOTE: each recording directory should have no subdirectories / be completely flat. the session
    # directory should have one directory per recording and no other subdirectories that are not
    # independent recordings. only exception is the suite2p directory

    # Read in the config file
    # If no config path, use defaults but ignore the spath from default
    if (cfg_path is None) and (spath is not None):
        internals_config_path = os.path.join(fm2p.up_dir(__file__, 1), 'utils/internals.yaml')
        cfg = fm2p.read_yaml(internals_config_path)
        cfg['spath'] = spath

    elif (cfg_path is None) and (spath is None):
        cfg_path = fm2p.select_file(
            title='Choose config yaml file.',
            filetypes=[('YAML', '*.yaml'),('YML', '*.yml'),]
        )
        cfg = fm2p.read_yaml(cfg_path)

    elif (cfg_path is not None):
        cfg = fm2p.read_yaml(cfg_path)

    
    # Switch to alternative preprocessing if config indicates that
    # this is a hippocampal plug recording.
    if cfg['hp_plug']:
        fm2p.hippocampal_preprocess(cfg_path)
        return
    

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


    if cfg['axons'] is True:
        axons = True
    else:
        axons = False
    if cfg['ltdk'] is True:
        ltdk = True
    else:
        ltdk = False
    

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

        if ltdk:
            ltdk_TTL_voltage = fm2p.find('*_ltdklogTTL.csv', rpath, MR=True)
            ltdk_TTL_timestamps = fm2p.find('*_ltdkttlTS.csv', rpath, MR=True)

        # Topdown camera files
        possible_topdown_videos = fm2p.find('*.mp4', rpath, MR=False)
        topdown_video = fm2p.filter_file_search(possible_topdown_videos, toss=['labeled','resnet50'], MR=True)

        if not axons:
            # Suite2p files
            F_path = fm2p.find('F.npy', rpath, MR=True)
            Fneu_path = fm2p.find('Fneu.npy', rpath, MR=True)
            suite2p_spikes = fm2p.find('spks.npy', rpath, MR=True)
            iscell_path = fm2p.find('iscell.npy', rpath, MR=True)
            stat_path = fm2p.find('stat.npy', rpath, MR=True)
            ops_path = fm2p.find('ops.npy', rpath, MR=True)

        elif axons:
            F_axons_path = fm2p.find('*_denoised_SRA_data.mat', rpath, MR=True)


        if cfg['run_deinterlace']:

            print('  -> Rotating and deinterlacing eye camera video.')

            # Deinterlace and rotate eyecam video
            eyecam_deinter_video = fm2p.deinterlace(eyecam_raw_video, do_rotation=True)

        if ('eyecam_deinter_video' not in vars()) and ('eyecam_deinter_video' not in globals()):
            eyecam_deinter_video = fm2p.find('*_eyecam_deinter.avi', rpath, MR=True)

        if cfg['run_pose_estimation']:

            if cfg['eye_DLC_project'] != 'none':

                print('  -> Running pose estimation for eye camera video.')

                # Run dlc on eyecam video
                fm2p.run_pose_estimation(
                    eyecam_deinter_video,
                    project_cfg=cfg['eye_DLC_project'],
                    filter=False
                )

            if cfg['top_DLC_project'] != 'none':

                print('  -> Running pose estimation for topdown camera video.')
                
                fm2p.run_pose_estimation(
                    topdown_video,
                    project_cfg=cfg['top_DLC_project'],
                    filter=False
                )

        # Find DLC files
        eyecam_pts_path = fm2p.find(cfg['eyecam_DLC_search_key'], rpath, MR=True)
        topdown_pts_path = fm2p.find(cfg['topdown_DLC_search_key'], rpath, MR=True)

        print('  -> Reading fluorescence data.')

        if not axons:
            # Read suite2p data
            F = np.load(F_path, allow_pickle=True)
            Fneu = np.load(Fneu_path, allow_pickle=True)
            spks = np.load(suite2p_spikes, allow_pickle=True)
            stat = np.load(stat_path, allow_pickle=True)
            ops =  np.load(ops_path, allow_pickle=True)
            iscell = np.load(iscell_path, allow_pickle=True)

        elif axons:
            dFF_out, denoised_dFF, sps, usecells = fm2p.get_independent_axons(F_axons_path)

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

        arena_yaml_path = os.path.join(rpath, 'arena_props.yaml')
        # if os.path.isfile(arena_yaml_path):
        #     arena_dict = fm2p.read_yaml(arena_yaml_path)
        # else:
        arena_dict = top_cam.track_arena()
        arena_dict = fm2p.fix_dict_dtype(arena_dict, float)
        fm2p.write_yaml(arena_yaml_path, arena_dict)

        pxls2cm = arena_dict['pxls2cm']
        top_xyl, top_tracking_dict = top_cam.track_body(pxls2cm)


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


        print('  -> Aligning eye camera data streams to 2P and behavior data using TTL voltage.')

        eyeStart, eyeEnd = fm2p.align_eyecam_using_TTL(
            eye_dlc_h5=eyecam_pts_path,
            eye_TS_csv=eyecam_video_timestamps,
            eye_TTLV_csv=eyecam_TTL_voltage,
            eye_TTLTS_csv=eyecam_TTL_timestamps,
            theta=ellipse_dict['theta'].copy()
        )
        eyeStart = int(eyeStart)
        eyeEnd = int(eyeEnd)


        print('  -> Running spike inference.')

        # Load processed two photon data from suite2p
        if not axons:
            twop_recording = fm2p.TwoP(rpath, full_rname, cfg=cfg)
            twop_recording.add_data(
                F=F,
                Fneu=Fneu,
                spikes=spks,
                iscell=iscell
            )
            twop_dict = twop_recording.calc_dFF(neu_correction=0.7, oasis=False)
            dFF_transients = twop_recording.calc_dFF_transients()
            # Set a maximum spike rate for each cell, then normalize spikes
            normspikes = twop_recording.normalize_spikes()
            recording_props = twop_recording.get_recording_props(
                stat=stat,
                ops=ops
            )

            twop_dt = 1./cfg['twop_rate']
            twopT = np.arange(0, np.size(twop_dict['s2p_spks'], 1)*twop_dt, twop_dt)
            twop_dict['twopT'] = twopT
            twop_dict['matlab_cellinds'] = np.arange(np.size(twop_dict['raw_F'],0))
            twop_dict['norm_spikes'] = normspikes
            twop_dict['dFF_transients'] = dFF_transients

            twop_dict = {**twop_dict, **recording_props}

        elif axons:
            twop_dict = {}
            
            twop_dt = 1./cfg['twop_rate']
            twopT = np.arange(0, np.size(sps, 1)*twop_dt, twop_dt)
            twop_dict['twopT'] = twopT

            twop_dict['raw_F0'] = np.zeros(np.size(dFF_out,0))
            twop_dict['raw_F'] =  np.zeros([np.size(dFF_out,0), np.size(dFF_out,1)])
            twop_dict['norm_F'] =  np.zeros([np.size(dFF_out,0), np.size(dFF_out,1)])
            twop_dict['raw_Fneu'] =  np.zeros([np.size(dFF_out,0), np.size(dFF_out,1)])
            twop_dict['raw_dFF'] = dFF_out
            twop_dict['norm_dFF'] = np.zeros([np.size(dFF_out,0), np.size(dFF_out,1)])
            twop_dict['denoised_dFF'] = denoised_dFF
            twop_dict['s2p_spks'] = sps
            twop_dict['matlab_cellinds'] = np.array(usecells)

        print('  -> Calculating retinocentric and egocentric orientations.')

        # All values in units of pixels or degrees (not cm or rads)
        learx = top_tracking_dict['lear_x']
        leary = top_tracking_dict['lear_y']
        rearx = top_tracking_dict['rear_x']
        reary = top_tracking_dict['rear_y']
        yaw = top_tracking_dict['head_yaw_deg']
        theta = np.rad2deg(ellipse_dict['theta'])
        phi = np.rad2deg(ellipse_dict['phi'])


        _len_diff = np.size(learx) - np.size(twop_dict['s2p_spks'], 1)
        while _len_diff != 0:
            if _len_diff > 0:
                # top tracking is too long for spike data
                learx = learx[:-1]
                leary = leary[:-1]
                rearx = rearx[:-1]
                reary = reary[:-1]
                yaw = yaw[:-1]
            elif _len_diff < 0:
                # spike data is too long for top tracking
                twop_dict['twopT'] = twop_dict['twopT'][:-1]
                twop_dict['raw_F0'] = twop_dict['raw_F0'][:-1]
                twop_dict['raw_F'] = twop_dict['raw_F'][:-1]
                twop_dict['norm_F'] = twop_dict['norm_F'][:-1]
                twop_dict['raw_Fneu'] = twop_dict['raw_Fneu'][:-1]
                twop_dict['raw_dFF'] = twop_dict['raw_dFF'][:-1]
                twop_dict['norm_dFF'] = twop_dict['norm_dFF'][:-1]
                twop_dict['denoised_dFF'] = twop_dict['denoised_dFF'][:-1]
                twop_dict['s2p_spks'] = twop_dict['s2p_spks'][:-1]
            _len_diff = np.size(learx) - np.size(twop_dict['s2p_spks'], 1)

        headx = np.array([np.mean([rearx[f], learx[f]]) for f in range(len(rearx))])
        heady = np.array([np.mean([reary[f], leary[f]]) for f in range(len(reary))])

        eyeT = fm2p.read_timestamp_file(eyecam_video_timestamps, position_data_length=len(theta))
        theta_interp = fm2p.interpT(
            theta[eyeStart:eyeEnd],
            eyeT[eyeStart:eyeEnd] - eyeT[eyeStart],
            twopT
        )
        phi_interp = fm2p.interpT(
            phi[eyeStart:eyeEnd],
            eyeT[eyeStart:eyeEnd] - eyeT[eyeStart],
            twopT
        )

        # Calculate retinocentric and egocentric orientations
        refframe_dict = fm2p.calc_reference_frames(
            cfg,
            headx,
            heady,
            yaw,
            theta_interp,
            arena_dict
        )

        if ltdk:
            print('  -> Using TTL to calculate light/dark state vector')
            ltdk_state_vec, light_onsets, dark_onsets = fm2p.align_lightdark_using_TTL(
                ltdk_TTL_voltage,
                ltdk_TTL_timestamps,
                eyeT,
                twopT,
                eyeStart,
                eyeEnd
            )

        print('  -> Saving preprocessed dataset to file.')

        top_xyl_ = fm2p.to_dict_of_arrays(top_xyl)
        eye_xyl_ = fm2p.to_dict_of_arrays(eye_xyl)

        preprocessed_dict = {
            **top_tracking_dict,
            **top_xyl_,
            **arena_dict,
            **ellipse_dict,
            **eye_xyl_,
            **twop_dict,
            **refframe_dict
        }

        preprocessed_dict['eyeT_startInd'] = eyeStart
        preprocessed_dict['eyeT_endInd'] = eyeEnd
        preprocessed_dict['theta_interp'] = theta_interp
        preprocessed_dict['phi_interp'] = phi_interp
        preprocessed_dict['head_x'] = headx
        preprocessed_dict['head_y'] = heady

        preprocessed_dict['ltdk'] = ltdk # was 'tldk', will need to build in a flag to make sure
        # the one that was used in existing preprocessing files is found.

        if ltdk:
            preprocessed_dict['ltdk_state_vec'] = ltdk_state_vec
            preprocessed_dict['light_onsets'] = light_onsets
            preprocessed_dict['dark_onsets'] = dark_onsets


        if len(cyclotorsion_dict.keys()) > 0:
            preprocessed_dict = {**preprocessed_dict, **cyclotorsion_dict}

        # fm2p.run_preprocessing_diagnostics(preprocessed_dict, ltdk=ltdk)

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

