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


if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

import os
import sys
import matplotlib
import os
if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
    matplotlib.use('TkAgg')
import argparse
import copy
import subprocess
import numpy as np
# NumPy version shim: files saved with NumPy 2.x reference numpy._core,
# but NumPy 1.x only has numpy.core.  Register aliases so pickle can find them.
import sys as _sys_np_shim
_sys_np_shim.modules.setdefault('numpy._core', np.core)
for _a in dir(np.core):
    _sys_np_shim.modules.setdefault(f'numpy._core.{_a}', getattr(np.core, _a))
del _sys_np_shim, _a
import pandas as pd
from pathlib import Path

from .utils.paths import up_dir, find, filter_file_search, list_subdirs
from .utils.files import read_yaml, write_yaml, write_h5
from .utils.gui_funcs import select_file
from .utils.helper import fix_dict_dtype, to_dict_of_arrays
from .utils.cameras import deinterlace, run_pose_estimation
from .utils.eyecam import Eyecam
from .utils.topcam import Topcam
from .utils.twop import TwoP, normalize_axonal_spikes
from .utils.time import read_timestamp_file, interpT, read_scanimage_time
from .utils.alignment import align_eyecam_using_TTL, align_lightdark_using_TTL, align_crop_IMU
from .utils.ref_frame import calc_reference_frames, calc_vor_eye_offset
from .utils.axons import get_independent_axons
from .utils.imu import read_IMU, detrend_gyroz_weighted_gaussian, check_and_trim_imu_disconnect
from .utils.PETH import get_discrete_spike_times, calc_eye_head_movement_times
from .utils.hippocampus_preprocessing import hippocampal_preprocess


DEFAULT_BATCH_PATH = '/home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings'

# Config used for every recording in a --batch run. This is the repo's own
# config/config.yaml, not a per-session config.yaml (session configs carry
# stale spaths and incomplete include_recordings lists) and not
# utils/internals.yaml (which lacks several keys preprocess() requires).
DEFAULT_BATCH_CONFIG = os.path.join(up_dir(__file__, 2), 'config', 'config.yaml')


def preprocess(cfg_path=None, spath=None, batch_mode=False, cfg=None):
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
    batch_mode : bool, optional
        If True, never open the interactive arena/pillar annotation GUI.
        Recordings whose `arena_props.yaml` hasn't already been saved from
        a prior manual run are skipped (with a warning) instead of
        blocking on `top_cam.track_arena()`. Used by `preprocess_batch()`
        so an unattended batch run can't hang waiting for GUI input.
        The default is False.
    cfg : dict, optional
        A fully-built config dict to use as-is, bypassing cfg_path/spath/
        select_file entirely. Used by `preprocess_batch()`, which builds
        cfg itself from fm2p/utils/internals.yaml rather than trusting a
        session's own (possibly stale) config.yaml. The default is None.

    Returns
    -------
    None
    """

    # parse_known_args (not parse_args): preprocess() may be called from
    # preprocess_batch(), which is itself dispatched from a CLI parser that
    # also defines --batch/--batch_path. Those flags remain in sys.argv for
    # every nested call, but only -cfg/--cfg matters here, so unknown flags
    # must be tolerated rather than raising.
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--cfg', type=str, default=None)
    args, _ = parser.parse_known_args()

    if args.cfg is not None:
        cfg_path = args.cfg

    # NOTE: each recording directory should have no subdirectories / be completely flat. the session
    # directory should have one directory per recording and no other subdirectories that are not
    # independent recordings. only exception is the suite2p directory

    # Read in the config file
    # If no config path, use defaults but ignore the spath from default
    if cfg is not None:
        pass

    elif (cfg_path is None) and (spath is not None):
        internals_config_path = os.path.join(up_dir(__file__, 1), 'utils/internals.yaml')
        cfg = read_yaml(internals_config_path)
        cfg['spath'] = spath

    elif (cfg_path is None) and (spath is None):
        cfg_path = select_file(
            title='Choose config yaml file.',
            filetypes=[('YAML', '*.yaml'),('YML', '*.yml'),]
        )
        cfg = read_yaml(cfg_path)

    elif (cfg_path is not None):
        cfg = read_yaml(cfg_path)


    # Switch to alternative preprocessing if config indicates that
    # this is a hippocampal plug recording.
    if cfg['hp_plug']:
        hippocampal_preprocess(cfg_path)
        return
    

    # Find the number of recordings in the session
    # Every folder in the session directory is assumed to be a recording
    recording_names = list_subdirs(cfg['spath'], givepath=False)
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

        # twop tif
        if axons is True:
            twop_tiff_path = find('file_0*_denoised_registered.tif', rpath, MR=True)
        else:
            twop_tiff_path = find('file_0*.tif', rpath, MR=True)

        # Topdown camera files
        possible_topdown_videos = find('*.mp4', rpath, MR=False, retempty=True)

        sn = False
        if possible_topdown_videos is None:
            print('  -> Treating recording {} as hea-fixed sparse noise (no topdown found).'.format(rname))
            sn = True

        if sn:
            # stimulus timestamps
            sn_stimT_path = find('*sparsenoise.csv', rpath, MR=True)

        if not sn:
            topdown_video = filter_file_search(possible_topdown_videos, toss=['labeled','resnet50'], MR=True)

        if not sn:
            # Eye camera files
            eyecam_raw_video = find('*_eyecam.avi', rpath, MR=True)
            eyecam_TTL_voltage = find('*_logTTL.csv', rpath, MR=True)
            eyecam_TTL_timestamps = find('*_ttlTS.csv', rpath, MR=True)
            eyecam_video_timestamps = find('*_eyecam.csv', rpath, MR=True)

            if ltdk:
                ltdk_TTL_voltage = find('*_ltdklogTTL.csv', rpath, MR=True)
                ltdk_TTL_timestamps = find('*_ltdkttlTS.csv', rpath, MR=True)

            if cfg['imu']:
                imu_vals = find('*_IMUvals.csv', rpath, MR=True)
                imu_timestamps = find('*_IMUtime.csv', rpath, MR=True)

        if not axons:
        # Suite2p files
            F_path = find('F.npy', rpath, MR=True)
            Fneu_path = find('Fneu.npy', rpath, MR=True)
            suite2p_spikes = find('spks.npy', rpath, MR=True)
            iscell_path = find('iscell.npy', rpath, MR=True)
            stat_path = find('stat.npy', rpath, MR=True)
            ops_path = find('ops.npy', rpath, MR=True)
        # if axons:
        #     bin_path = find('data.bin', rpath, MR=True)

        elif axons:
            F_axons_path = find('*_denoised_registered_data.mat', rpath, MR=True)


        if cfg['run_deinterlace'] and not sn:

            print('  -> Rotating and deinterlacing eye camera video.')

            # Deinterlace and rotate eyecam video
            eyecam_deinter_video = deinterlace(eyecam_raw_video, do_rotation=True)

        if not sn:
            if ('eyecam_deinter_video' not in vars()) and ('eyecam_deinter_video' not in globals()):
                eyecam_deinter_video = find('*_eyecam_deinter.avi', rpath, MR=True)

        if cfg['run_pose_estimation'] and not sn:

            if cfg['eye_DLC_project'] != 'none':

                print('  -> Running pose estimation for eye camera video.')
                
                if 'freely_moving_eyecams_03-dylan-' not in cfg['eye_DLC_project']:

                    print('using dlc2')
                    run_pose_estimation(
                        eyecam_deinter_video,
                        project_cfg=cfg['eye_DLC_project'],
                        filter=False
                    )

                else:
                    print('Switching to dlc3 for {}, {}'.format(eyecam_deinter_video, cfg['eye_DLC_project']))
                    print('You will get no updates until it is completed (running as subprocess)')
                    # prep command to run in different conda environment
                    # this project was made in deeplabcut 3.0.0rc6, which cannot run in this
                    # python 3.9 environment, so we need to switch over to a different
                    # conda environment.

                    proj_string = str(Path(cfg['eye_DLC_project'])).replace("\\", "\\\\")
                    vid_str = str(Path(eyecam_deinter_video)).replace("\\", "\\\\")
                    python_code = (
                        f"import deeplabcut; "
                        f"deeplabcut.analyze_videos('{proj_string}', ['{vid_str}'])"
                    )
                    import os as _os
                    _env = _os.environ.copy()
                    _env.pop('LD_LIBRARY_PATH', None)
                    result = subprocess.run(
                        ['conda', 'run', '-n', 'dlc3', 'python', '-c', python_code],
                        capture_output=True,
                        text=True,
                        env=_env,
                    )

                    print("Output from other env:", result.stdout)
                    print("Errors:", result.stderr)


            if cfg['top_DLC_project'] != 'none' and (sn is False):

                print('  -> Running pose estimation for topdown camera video.')
                
                # run_pose_estimation(
                #     topdown_video,
                #     project_cfg=cfg['top_DLC_project'],
                #     filter=False
                # )

                proj_string = str(Path(cfg['top_DLC_project'])).replace("\\", "\\\\")
                vid_str = str(Path(topdown_video)).replace("\\", "\\\\")
                python_code = (
                    f"import deeplabcut; "
                    f"deeplabcut.analyze_videos('{proj_string}', ['{vid_str}'])"
                )
                import os as _os
                _env = _os.environ.copy()
                _env.pop('LD_LIBRARY_PATH', None)
                result = subprocess.run(
                    ['conda', 'run', '-n', 'dlc3', 'python', '-c', python_code],
                    capture_output=True,
                    text=True,
                    env=_env,
                )

                print("Output from other env:", result.stdout)
                print("Errors:", result.stderr)


        print('  -> Reading fluorescence data.')

        if not axons:
            # Read suite2p data
            F = np.load(F_path, allow_pickle=True)
            Fneu = np.load(Fneu_path, allow_pickle=True)
            spks = np.load(suite2p_spikes, allow_pickle=True)
            stat = np.load(stat_path, allow_pickle=True)
            ops =  np.load(ops_path, allow_pickle=True)
            iscell = np.load(iscell_path, allow_pickle=True)

        if axons:

            dFF_out, denoised_dFF, sps, kept_groups = get_independent_axons(cfg, matpath=F_axons_path, merge_duplicates=True)

        print('  -> Running spike inference.')

        if sn:
            full_rname = os.path.split(rpath)[1]

        if not sn:
            # 250218_DMM_DMM038_rec_01_eyecam.avi
            full_rname = '_'.join(os.path.split(eyecam_raw_video)[1].split('_')[:-1])


        # Load processed two photon data from suite2p
        if not axons:
            twop_recording = TwoP(rpath, full_rname, cfg=cfg)
            twop_recording.add_data(
                F=F,
                Fneu=Fneu,
                spikes=spks,
                iscell=iscell
            )
            twop_dict = twop_recording.calc_dFF(neu_correction=0.7, use_oasis=True)
            dFF_transients = twop_recording.calc_dFF_transients()
            # Set a maximum spike rate for each cell, then normalize spikes
            normspikes = twop_recording.normalize_spikes()
            recording_props = twop_recording.get_recording_props(
                stat=stat,
                ops=ops
            )

            # twop_dt = 1./cfg['twop_rate']
            # twopT = np.arange(0, np.size(twop_dict['s2p_spks'], 1)*twop_dt, twop_dt)
            twopT = read_scanimage_time(twop_tiff_path)

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
            twop_dict['matlab_cellinds'] = kept_groups
            twop_dict['norm_spikes'] = normalize_axonal_spikes(sps, cfg)

        if not sn:
            try:
                eyecam_pts_path = find(cfg['eyecam_DLC_search_key'], rpath, MR=True)
            except FileNotFoundError:
                if 'resnet' in cfg['eyecam_DLC_search_key']:
                    eyecam_pts_path = find(cfg['eyecam_DLC_search_key'].replace('resnet', 'Resnet'), rpath, MR=True)
                elif 'Resnet' in cfg['eyecam_DLC_search_key']:
                    eyecam_pts_path = find(cfg['eyecam_DLC_search_key'].replace('Resnet', 'resnet'), rpath, MR=True)
                else:
                    raise

            topdown_pts_path = find(cfg['topdown_DLC_search_key'], rpath, MR=True)



        if sn:

            sn_stimT = pd.read_csv(sn_stimT_path)['psychopy_time'].to_numpy()

        if not sn:

            print('  -> Measuring locomotor behavior.')

            # Topdown behavior and obstacle/arena tracking
            top_cam = Topcam(rpath, full_rname, cfg=cfg)
            top_cam.add_files(
                top_dlc_h5=topdown_pts_path,
                top_avi=topdown_video
            )

            arena_yaml_path = os.path.join(rpath, 'arena_props.yaml')
            try:
                arena_dict = read_yaml(arena_yaml_path)
            except Exception:
                if batch_mode:
                    print(f'  -> No saved arena_props.yaml for {rpath}. '
                          f'Skipping this recording in batch mode (the '
                          f'interactive arena/pillar annotation GUI cannot '
                          f'be opened unattended). Annotate it manually '
                          f'with a single-recording preprocess() run, then '
                          f're-run the batch.')
                    continue
                arena_dict = top_cam.track_arena()
                arena_dict = fix_dict_dtype(arena_dict, float)
                write_yaml(arena_yaml_path, arena_dict)

            pxls2cm = arena_dict['pxls2cm']
            top_xyl, top_tracking_dict = top_cam.track_body(pxls2cm)


            print('  -> Measuring pupil orientation via ellipse fit.')

            # Pupil tracking
            reye_cam = Eyecam(rpath, full_rname, cfg=cfg)
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

            eyeStart, eyeEnd, apply_t0, apply_tEnd = align_eyecam_using_TTL(
                eye_dlc_h5=eyecam_pts_path,
                eye_TS_csv=eyecam_video_timestamps,
                eye_TTLV_csv=eyecam_TTL_voltage,
                eye_TTLTS_csv=eyecam_TTL_timestamps,
                theta=ellipse_dict['theta'].copy()
            )
            eyeStart = int(eyeStart)
            eyeEnd = int(eyeEnd)


        if not sn:
            theta = np.rad2deg(ellipse_dict['theta'])
            phi = - np.rad2deg(ellipse_dict['phi']) # sign flip added 9/15/25

            print('  -> Calculating retinocentric and egocentric orientations.')

            # All values in units of pixels or degrees (not cm or rads)
            learx = top_tracking_dict['lear_x']
            leary = top_tracking_dict['lear_y']
            rearx = top_tracking_dict['rear_x']
            reary = top_tracking_dict['rear_y']
            yaw = top_tracking_dict['head_yaw_deg']


            # Match the number of top-camera tracking samples to the number of
            # 2P frames.  The calcium arrays are (n_cells, n_frames), so frames
            # are trimmed along axis 1; the per-cell arrays (raw_F0, norm_F0,
            # matlab_cellinds) index cells and are left alone.
            _n_track = int(np.size(learx))
            _n_twop = int(np.size(twop_dict['s2p_spks'], 1))
            _n_keep = min(_n_track, _n_twop)

            if _n_track != _n_twop:
                print('     trimming to match: top tracking {} frames, '
                      '2P {} frames -> {}'.format(_n_track, _n_twop, _n_keep))

            if _n_track > _n_keep:
                # top tracking is too long for spike data
                learx = learx[:_n_keep]
                leary = leary[:_n_keep]
                rearx = rearx[:_n_keep]
                reary = reary[:_n_keep]
                yaw = yaw[:_n_keep]

            if _n_twop > _n_keep:
                # spike data is too long for top tracking
                _n_drop = _n_twop - _n_keep
                twop_dict['twopT'] = twop_dict['twopT'][:len(twop_dict['twopT']) - _n_drop]
                twopT = twop_dict['twopT']
                for _key in ('raw_F', 'norm_F', 'raw_Fneu', 'raw_dFF', 'norm_dFF',
                             'denoised_dFF', 's2p_spks', 'oasis_spks',
                             'norm_spikes', 'dFF_transients'):
                    if _key in twop_dict:
                        twop_dict[_key] = twop_dict[_key][:, :_n_keep]

            headx = np.array([np.mean([rearx[f], learx[f]]) for f in range(len(rearx))])
            heady = np.array([np.mean([reary[f], leary[f]]) for f in range(len(reary))])

        if len(twop_dict['twopT']) == np.size(twop_dict['s2p_spks'], 1)+1:
            twop_dict['twopT'] = twop_dict['twopT'][:-1]
            twopT = twopT[:-1]

        if not sn:
            eyeT = read_timestamp_file(eyecam_video_timestamps, position_data_length=len(theta))
            theta_interp = interpT(
                theta[eyeStart:eyeEnd],
                eyeT[eyeStart:eyeEnd] - eyeT[eyeStart],
                twopT
            )
            phi_interp = interpT(
                phi[eyeStart:eyeEnd],
                eyeT[eyeStart:eyeEnd] - eyeT[eyeStart],
                twopT
            )
            theta_trim = theta[eyeStart:eyeEnd]
            eyeT_trim = eyeT[eyeStart:eyeEnd] - eyeT[eyeStart]
            phi_trim = phi[eyeStart:eyeEnd]


            if cfg['imu']:

                print('  -> Reading IMU data in and performing sensor fusion.')

                imu_df, imuT = read_IMU(imu_vals, imu_timestamps)

                print('  -> Aligning IMU to 2P and other behavior data.')

                imu_dict = align_crop_IMU(imu_df, imuT, apply_t0, apply_tEnd, eyeT[eyeStart:eyeEnd], twopT)

            print('  -> Calibrating eye-camera angular offset via VOR.')

            if cfg['imu']:
                # Prefer the highest timebase: gyro_z already interpolated to
                # eye camera timestamps by align_crop_IMU, paired with raw theta
                # at the same eye camera rate.
                _fps_eye = float(1.0 / np.nanmedian(np.diff(eyeT_trim)))
                vor_offset_dict = calc_vor_eye_offset(
                    theta_trim,
                    None,
                    fps=_fps_eye,
                    head_vel_deg_s=imu_dict['gyro_z_eye_interp']
                )
            else:
                # Fallback when no IMU: use head_yaw_deg differentiated at the
                # 2P frame rate (lower resolution).
                vor_offset_dict = calc_vor_eye_offset(
                    theta_interp,
                    yaw,
                    fps=cfg.get('twop_rate', float(1.0 / np.nanmedian(np.diff(twopT))))
                )
            print('     ang_offset_vor_null       = {:.2f} deg'.format(
                vor_offset_dict['ang_offset_vor_null']))
            print('     ang_offset_vor_regression = {:.2f} deg'.format(
                vor_offset_dict['ang_offset_vor_regression']))
            print('     VOR gain                  = {:.3f}'.format(
                vor_offset_dict['vor_gain']))

            # Use the empirically-derived VOR offset, not the static config value.
            # The regression-based estimate is preferred as it's more robust.
            cfg_updated = cfg.copy()
            cfg_updated['eyecam_angular_offset'] = vor_offset_dict['ang_offset_vor_regression']

            # Calculate retinocentric and egocentric orientations
            refframe_dict = calc_reference_frames(
                cfg_updated,
                headx,
                heady,
                yaw,
                theta_interp,
                arena_dict
            )

            if ltdk:
                print('  -> Using TTL to calculate light/dark state vector')
                ltdk_state_vec, light_onsets, dark_onsets = align_lightdark_using_TTL(
                    ltdk_TTL_voltage,
                    ltdk_TTL_timestamps,
                    eyeT,
                    twopT,
                    eyeStart,
                    eyeEnd
                )

        if not sn:
            top_xyl_ = to_dict_of_arrays(top_xyl)
            eye_xyl_ = to_dict_of_arrays(eye_xyl)


            preprocessed_dict = {
                **top_tracking_dict,
                **top_xyl_,
                **arena_dict,
                **ellipse_dict,
                **eye_xyl_,
                **twop_dict,
                **refframe_dict,
                **vor_offset_dict,
            }

            preprocessed_dict['eyeT_startInd'] = eyeStart
            preprocessed_dict['eyeT_endInd'] = eyeEnd
        
            preprocessed_dict['theta_interp'] = theta_interp
            preprocessed_dict['phi_interp'] = phi_interp
            preprocessed_dict['theta_trim'] = theta_trim
            preprocessed_dict['phi_trim'] = phi_trim
            preprocessed_dict['eyeT_trim'] = eyeT_trim
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

            if cfg['imu']:
                preprocessed_dict = {**preprocessed_dict, **imu_dict}

            # Apply IMU disconnection trimming regardless of cfg['imu'] —
            # check_and_trim_imu_disconnect is a no-op if gyro_z_trim is absent.
            print('  -> Checking for IMU disconnection.')
            preprocessed_dict = check_and_trim_imu_disconnect(preprocessed_dict)

        # fm2p.run_preprocessing_diagnostics(preprocessed_dict)

            # revcorr_dict = fm2p.calc_revcorr(preprocessed_dict, save=False, IMU=cfg['imu'])
            # preprocessed_dict = {**preprocessed_dict, **revcorr_dict}

        if not sn and cfg['imu']:

            print('  -> Detrending integrated gyro along z-axis.')
            # use integral of gyro_z and use topdown measure of head yaw as a template
            # to correct for the accumulated error.  The smoothing window has come
            # down twice for the same reason: 120s let the corrected yaw wander
            # ~90-100deg within a recording, 5s fixed that on a 2P session but
            # still left 135deg of drift on the worldcam session (where topdown
            # yaw is 35% NaN, so anchors are sparse).  0.5s -- the function
            # default now -- leaves 2.4deg.  See fm2p/utils/imu.py for the
            # ground-truth measurement behind those numbers.
            upsampled_yaw = detrend_gyroz_weighted_gaussian(preprocessed_dict,
                                                            gaussian_weight=1.)
            preprocessed_dict['upsampled_yaw'] = upsampled_yaw

            # print('  -> Calculating PETHS.')
            # peth_imu_dict = fm2p.calc_PETHs(preprocessed_dict)
            # preprocessed_dict = {**preprocessed_dict, **peth_imu_dict}

            print('  -> Computing eye/head movement kinematics.')
            saccade_dict = calc_eye_head_movement_times(preprocessed_dict)
            preprocessed_dict.update(saccade_dict)

        if sn:

            preprocessed_dict = twop_dict
            preprocessed_dict['stimT'] = sn_stimT


        print('  -> Inferring spike times via MCMC deconvolution (fMCSI)...')
        _twopT = preprocessed_dict['twopT']
        _fs    = float(1.0 / np.nanmedian(np.diff(_twopT)))
        _spike_times_list = get_discrete_spike_times(
            preprocessed_dict['raw_F'], preprocessed_dict['raw_Fneu'], fs=_fs
        )
        _n_cells = len(_spike_times_list)
        _max_sp  = max((len(st) for st in _spike_times_list), default=1)
        _sp_array = np.full((_n_cells, _max_sp), np.nan, dtype=np.float64)
        for _i, _st in enumerate(_spike_times_list):
            _sp_array[_i, :len(_st)] = _st
        preprocessed_dict['spike_times'] = _sp_array

        _savepath = os.path.join(rpath, '{}_preproc.h5'.format(full_rname))
        print('  -> Writing preprocessed data to {}'.format(_savepath))
        write_h5(_savepath, preprocessed_dict)

        # If a real config path was given, write some new data into the dictionary and then save a new preprocessed_config
        cfg['{}_preproc_file'.format(rname)] = _savepath
        if not sn:
            cfg['{}_topdown_video'.format(rname)] = topdown_video
            cfg['{}_eye_video'.format(rname)] = eyecam_deinter_video


    # if cfg_path is not None:

    #     print('  -> Updating config yaml file.')

    #     # Write a new version of the config file. Maybe change this to overwrite previous?
    #     _newsavepath = os.path.join(os.path.split(cfg_path)[0], 'preprocessed_config.yaml')
    #     write_yaml(_newsavepath, cfg)


def preprocess_batch(batch_path=DEFAULT_BATCH_PATH, start_at=1,
                     config_path=DEFAULT_BATCH_CONFIG):
    """
    Re-preprocess every already-preprocessed fm recording under `batch_path`,
    using one shared config file for all of them.

    Every recording is run with the config at `config_path` (by default the
    repo's own config/config.yaml). Session config.yaml files are not used
    here: some have a stale 'spath' (e.g. a Windows path from the acquisition
    machine that doesn't exist on this host) and some have an incomplete or
    wrong 'include_recordings' list. Recordings are instead discovered
    directly from disk -- every existing '<rec>_preproc.h5' inside an 'fm*'
    directory (each one is ground truth that the recording was already
    established as a valid fm recording worth processing, as opposed to a
    sparse-noise or "_Unusable" sibling directory in the same session).

    Only two keys are overridden per recording, and both are structural rather
    than scientific:

      - 'spath' is set to the recording's session directory.
      - 'include_recordings' is set to just that one recording directory, so
        its non-fm siblings are never touched.

    Everything else -- run_deinterlace, run_pose_estimation, imu, ltdk,
    hp_plug, the DLC project paths and search keys, likelihood_thresh,
    eye_trackable_N, cell_sd_thresh, and the rest -- comes from `config_path`
    and is identical for every recording. To change how the batch runs, edit
    that file rather than this function.

    The presence of IMU (*_IMUvals.csv / *_IMUtime.csv) and light/dark TTL
    (*_ltdklogTTL.csv / *_ltdkttlTS.csv) files is still checked per recording,
    but only to warn when what's on disk disagrees with the config's 'imu' /
    'ltdk' flags. The config wins; the warning is there so a mismatch shows up
    at the top of the log instead of as an obscure failure later in the run.

    Runs with batch_mode=True, so a recording missing a saved
    arena_props.yaml is skipped (with a warning) instead of opening the
    interactive annotation GUI.

    Parameters
    ----------
    batch_path : str
        Root directory to search for existing fm recording preproc.h5
        files. Defaults to DEFAULT_BATCH_PATH.
    start_at : int
        1-indexed position in the (sorted, so deterministic across runs)
        recording list to resume from. Recordings before this position are
        skipped entirely -- not even discovered/logged -- so a prior run's
        already-clean recordings are never touched again. The default (1)
        starts from the first recording, i.e. no skipping.
    config_path : str
        Config yaml applied to every recording in the batch. Defaults to
        DEFAULT_BATCH_CONFIG (the repo's config/config.yaml).

    Returns
    -------
    results : dict
        Maps each recording directory to 'ok', or to the exception raised
        while reprocessing it. Skipped (pre-start_at) recordings are not
        included.
    """

    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f'Batch config not found: {config_path}. Pass --batch_config with '
            f'the config yaml to apply to every recording.')

    base_cfg = read_yaml(config_path)
    print(f'Batch config: {config_path}')
    print(f'  run_deinterlace={base_cfg["run_deinterlace"]}, '
          f'run_pose_estimation={base_cfg["run_pose_estimation"]}, '
          f'imu={base_cfg["imu"]}, ltdk={base_cfg["ltdk"]}, '
          f'hp_plug={base_cfg["hp_plug"]}')
    print(f'  eye_DLC_project={base_cfg["eye_DLC_project"]}')
    print(f'  top_DLC_project={base_cfg["top_DLC_project"]}')

    preproc_files = find('*_preproc.h5', batch_path, MR=False, retempty=True) or []
    rec_dirs = sorted({
        os.path.dirname(p) for p in preproc_files
        if 'boundary' not in p
        and os.path.basename(os.path.dirname(p)).startswith('fm')
    })
    n_total = len(rec_dirs)

    print(f'Found {n_total} fm recording(s) with an existing preproc.h5 '
          f'under {batch_path}.')
    if start_at > 1:
        print(f'Resuming at recording {start_at}/{n_total} -- skipping the '
              f'first {start_at - 1} (already processed cleanly).')

    results = {}
    for i in range(start_at - 1, n_total):
        rec_dir     = rec_dirs[i]
        rec_name    = os.path.basename(rec_dir)
        session_dir = os.path.dirname(rec_dir)

        print(f'\n=== [{i + 1}/{n_total}] Batch recording: {rec_dir} ===')

        try:
            has_imu = bool(
                find('*_IMUvals.csv', rec_dir, MR=False, retempty=True)
                and find('*_IMUtime.csv', rec_dir, MR=False, retempty=True)
            )
            has_ltdk = bool(
                find('*_ltdklogTTL.csv', rec_dir, MR=False, retempty=True)
                and find('*_ltdkttlTS.csv', rec_dir, MR=False, retempty=True)
            )
            if bool(base_cfg['imu']) != has_imu:
                print(f'  -> WARNING: config says imu={base_cfg["imu"]} but IMU '
                      f'csv files are {"present" if has_imu else "absent"} in '
                      f'{rec_dir}. Using the config value.')
            if bool(base_cfg['ltdk']) != has_ltdk:
                print(f'  -> WARNING: config says ltdk={base_cfg["ltdk"]} but '
                      f'light/dark TTL csv files are '
                      f'{"present" if has_ltdk else "absent"} in {rec_dir}. '
                      f'Using the config value.')

            # Everything except the two structural keys below comes from
            # config_path, so every recording in the batch is processed
            # identically.
            cfg = copy.deepcopy(base_cfg)
            cfg['spath']              = session_dir
            cfg['include_recordings'] = [rec_name]

            preprocess(cfg=cfg, batch_mode=True)
            results[rec_dir] = 'ok'
        except Exception as e:
            print(f'  -> ERROR reprocessing {rec_dir}: {e}')
            results[rec_dir] = e

    n_ok  = sum(1 for v in results.values() if v == 'ok')
    n_err = len(results) - n_ok
    print(f'\nBatch preprocessing complete: {n_ok} succeeded, {n_err} failed '
          f'out of {len(results)} recordings.')
    if n_err:
        print('Failed recordings:')
        for k, v in results.items():
            if v != 'ok':
                print(f'  {k}: {v}')

    return results


if __name__ == '__main__':

    _cli_parser = argparse.ArgumentParser()
    _cli_parser.add_argument('-cfg', '--cfg', type=str, default=None,
                              help='Path to a single session config.yaml.')
    _cli_parser.add_argument('--batch', action='store_true', default=False,
                              help='Run preprocess_batch() instead of a single '
                                   'session, applying --batch_config to every fm '
                                   'recording found under --batch_path.')
    _cli_parser.add_argument('--batch_config', type=str, default=DEFAULT_BATCH_CONFIG,
                              help='Config yaml applied to every recording in a '
                                   '--batch run (default: %(default)s).')
    _cli_parser.add_argument('--batch_path', type=str, default=DEFAULT_BATCH_PATH,
                              help='Root directory to search for fm recordings '
                                   'when --batch is set (default: %(default)s).')
    _cli_parser.add_argument('--batch_start', type=int, default=1,
                              help='1-indexed recording to resume --batch from '
                                   '(skips earlier recordings in the sorted list '
                                   'entirely). Default: 1 (start from the first).')
    _cli_args, _ = _cli_parser.parse_known_args()

    if _cli_args.batch:
        preprocess_batch(batch_path=_cli_args.batch_path,
                         start_at=_cli_args.batch_start,
                         config_path=_cli_args.batch_config)
    else:
        preprocess(cfg_path=_cli_args.cfg)

