# -*- coding: utf-8 -*-
"""
fm2p/preprocess.py

Preprocessing pipeline for freely moving two-photon calcium imaging data.

Example usage
-------------
    $ python -m fm2p.preprocess -d /path/to/session_dir

No per-recording config file is needed. All parameters come from the repo's
default config/config.yaml (DEFAULT_CONFIG).

Each session should be structured so that each recording (freely moving and
sparse noise conditions) is in its own subdirectory within the session
directory, which is what `-d`/`--dir` points to. A subdirectory with no
fm*.mp4 topdown video is treated as head-fixed sparse noise.
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

Functions
---------
_dlc_h5_nrows
    Row count of a DeepLabCut .h5 without materializing the table.
_video_nframes
    Frame count of a video, from the container header.
_resolve_topdown_dlc
    The DLC .h5 that belongs to `topdown_video`, not merely the newest match.
preprocess
    Preprocess recording, converting raw data to an .h5 file.
preprocess_batch
    Re-preprocess every already-preprocessed fm recording under `batch_path`, using the default config for all of them.


DMM, May 2025
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
import time
import numpy as np
# NumPy version shim: files saved with NumPy 2.x reference numpy._core,
# but NumPy 1.x only has numpy.core.  Register aliases so pickle can find them.
import sys as _sys_np_shim
_sys_np_shim.modules.setdefault('numpy._core', np.core)
for _a in dir(np.core):
    _sys_np_shim.modules.setdefault(f'numpy._core.{_a}', getattr(np.core, _a))
del _sys_np_shim, _a
import pandas as pd
import h5py
from pathlib import Path

from .utils.paths import up_dir, find, filter_file_search, list_subdirs
from .utils.files import read_yaml, write_yaml, write_h5
from .utils.helper import fix_dict_dtype, to_dict_of_arrays
from .utils.cameras import deinterlace, run_pose_estimation
from .utils.eyecam import Eyecam
from .utils.topcam import Topcam
from .utils.twop import (TwoP, normalize_axonal_spikes, run_omsi,
                        bin_spike_times_to_frames)
from .utils.time import read_timestamp_file, interpT, read_scanimage_time
from .utils.alignment import align_eyecam_using_TTL, align_lightdark_using_TTL, align_crop_IMU
from .utils.ref_frame import calc_reference_frames, calc_vor_eye_offset
from .utils.axons import get_independent_axons
from .utils.imu import (read_IMU, detrend_gyroz_weighted_gaussian,
                        check_and_trim_imu_disconnect, rotate_tilt_to_head,
                        MOUNT_AZIMUTH_DEG)
from .utils.PETH import get_discrete_spike_times, calc_eye_head_movement_times


DEFAULT_BATCH_PATH = '/home/dylan/Storage/freely_moving_data/V1PPC/cohort02_recordings'

# Defaults used for every recording, in both single-session and --batch runs.
# This is the repo's own config/config.yaml, not a per-session config.yaml
# (session configs carry stale spaths and incomplete include_recordings lists)
# and not utils/internals.yaml (which lacks several keys preprocess() requires).
DEFAULT_CONFIG = os.path.join(up_dir(__file__, 2), 'config', 'config.yaml')
DEFAULT_BATCH_CONFIG = DEFAULT_CONFIG


def _dlc_h5_nrows(path):
    """Row count of a DeepLabCut .h5 without materializing the table."""

    try:
        with h5py.File(path, 'r') as f:
            for key in ('df_with_missing/table', 'df_with_missing/block0_values'):
                if key in f:
                    return int(f[key].shape[0])
    except Exception:
        pass
    try:
        return int(len(pd.read_hdf(path)))
    except Exception:
        return None


def _video_nframes(path):
    """Frame count of a video, from the container header."""

    try:
        # Local: preprocessing runs without cv2 in some envs.
        import cv2 as _cv2
        cap = _cv2.VideoCapture(path)
        n = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n if n > 0 else None
    except Exception:
        return None


def _resolve_topdown_dlc(topdown_video, rpath, search_key):
    """The DLC .h5 that belongs to `topdown_video`, not merely the newest match.

    The search key is a network-name glob, so it matches the DLC output for
    EVERY video in the recording folder that has been run through that network.
    Diagnostic videos written by get_retinal_image live in the same folder and
    get analyzed alongside the real recording, and because the old code took the
    most recent match it silently picked those: on three cohort02 sessions the
    entire top-down trace in the _preproc.h5 came from a diagnostic MP4 rather
    than from the arena camera.  The head "position" was then whatever DLC found
    in a 1680x960 figure panel, and every downstream array -- head_x, head_y,
    head_yaw_deg, and the 2P data truncated to match it -- was wrong.

    Resolution order:
      1. the match whose filename begins with the top-down video's stem
      2. among the rest, matches whose row count equals the video's frame count
      3. failing both, raise rather than guess

    Name-matching alone is not enough: one of the three bad sessions was tracked
    from `yaw_diagnostic_0_3600.mp4`, which shares no substring with
    "retinal_diagnostic", so a blacklist would have missed it.  The frame-count
    test is the one that generalizes.
    """

    candidates = find(search_key, rpath, retempty=True)
    if isinstance(candidates, str):
        candidates = [candidates]
    candidates = [c for c in (candidates or []) if os.path.isfile(c)]
    if not candidates:
        raise FileNotFoundError(
            'Found no file(s) matching key {} in directory {}'.format(search_key, rpath))

    stem = os.path.splitext(os.path.basename(topdown_video))[0]
    by_name = [c for c in candidates if os.path.basename(c).startswith(stem)]
    if by_name:
        pick = max(by_name, key=os.path.getmtime)
        if len(candidates) > 1:
            print('  Topdown DLC: {} (matched to {} by name; ignored {} other '
                  'file(s) matching the same network key)'.format(
                      os.path.basename(pick), os.path.basename(topdown_video),
                      len(candidates) - len(by_name)))
        return pick

    n_vid = _video_nframes(topdown_video)
    if n_vid:
        by_len = [c for c in candidates if (_dlc_h5_nrows(c) or -1) == n_vid]
        if by_len:
            pick = max(by_len, key=os.path.getmtime)
            print('  Topdown DLC: {} (matched to {} by frame count {})'.format(
                os.path.basename(pick), os.path.basename(topdown_video), n_vid))
            return pick

    raise FileNotFoundError(
        'None of the {} file(s) matching {} in {} belong to the top-down video '
        '{} ({} frames). Candidates and their row counts: {}. Re-run DLC on the '
        'top-down video, or delete the DLC output for the diagnostic videos in '
        'this folder.'.format(
            len(candidates), search_key, rpath, os.path.basename(topdown_video),
            n_vid, {os.path.basename(c): _dlc_h5_nrows(c) for c in candidates}))


def preprocess(spath=None, batch_mode=False, cfg=None):
    """ Preprocess recording, convering raw data to a set of several .h5 files.

    This function processes the raw data from two-photon calcium imaging recordings, including
    eyecam and topdown camera videos, and generates a set of preprocessed .h5 files. The preprocessing
    includes deinterlacing and rotating the eyecam video, running pose estimation on both the eyecam
    and topdown camera videos, aligning the data streams using TTL voltage, measuring pupil orientation,
    and calculating retinocentric and egocentric orientations. The script also runs spike inference on
    the two-photon data and saves the processed data to .h5 files.

    All parameters are read from the default config (DEFAULT_CONFIG); no
    per-recording config file is used.

    Parameters
    ----------
    spath : str, optional
        Path to the session directory, i.e. the directory containing one
        subdirectory per recording (freely moving and sparse noise). Required
        unless `cfg` already has 'spath' set.
    batch_mode : bool, optional
        If True, never open the interactive arena/pillar annotation GUI.
        Recordings whose `arena_props.yaml` hasn't already been saved from
        a prior manual run are skipped (with a warning) instead of
        blocking on `top_cam.track_arena()`. Used by `preprocess_batch()`
        so an unattended batch run can't hang waiting for GUI input.
        The default is False.
    cfg : dict, optional
        A fully-built config dict to use as-is instead of the defaults. Used
        by `preprocess_batch()`, which overrides 'spath' and
        'include_recordings' per recording. The default is None.

    Returns
    -------
    None
    """

    # NOTE: each recording directory should have no subdirectories / be completely flat.
    # the session
    # directory should have one directory per recording and no other subdirectories that
    # are not
    # independent recordings. only exception is the suite2p directory

    if cfg is None:
        cfg = read_yaml(DEFAULT_CONFIG)
        cfg['include_recordings'] = []
    if spath is not None:
        cfg['spath'] = spath
    if not cfg.get('spath') or not os.path.isdir(cfg['spath']):
        raise FileNotFoundError(
            'Session directory not found: {!r}. Pass it with -d/--dir.'.format(cfg.get('spath')))

    # Find the number of recordings in the session
    # Every folder in the session directory is a recording except the shared
    # suite2p output and folders set aside with a leading underscore (e.g.
    # _Unusable); hidden folders are already dropped by list_subdirs.
    recording_names = list_subdirs(cfg['spath'], givepath=False)
    skipped = [x for x in recording_names if x == 'suite2p' or x.startswith('_')]
    recording_names = sorted(x for x in recording_names if x not in skipped)
    if skipped:
        print('  Not recordings, skipped: {}'.format(', '.join(sorted(skipped))))
    num_recordings = len(recording_names)

    # Check recordings against list of included recordings. If the list is empty,
    # analyze all.
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

        print('  Analyzing {}.'.format(rpath))
        print('  Finding files.')

        # twop tif
        if axons is True:
            twop_tiff_path = find('file_0*_denoised_registered.tif', rpath, MR=True)
        else:
            twop_tiff_path = find('file_0*.tif', rpath, MR=True)

        # Topdown camera files
        possible_topdown_videos = find('fm*.mp4', rpath, MR=False, retempty=True)

        sn = False
        if possible_topdown_videos is None:
            print('  Treating recording {} as head-fixed sparse noise (no topdown found).'.format(rname))
            sn = True

        if sn:
            # Stimulus timestamps
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

            print('  Rotating and deinterlacing eye camera video.')

            # Deinterlace and rotate eyecam video
            eyecam_deinter_video = deinterlace(eyecam_raw_video, do_rotation=True)

        if not sn:
            if ('eyecam_deinter_video' not in vars()) and ('eyecam_deinter_video' not in globals()):
                eyecam_deinter_video = find('*_eyecam_deinter.avi', rpath, MR=True)

        if cfg['run_pose_estimation'] and not sn:

            if cfg['eye_DLC_project'] != 'none':

                print('  Running pose estimation for eye camera video.')

                if 'freely_moving_eyecams_03-dylan-' not in cfg['eye_DLC_project']:

                    print('Using dlc2...')
                    run_pose_estimation(
                        eyecam_deinter_video,
                        project_cfg=cfg['eye_DLC_project'],
                        filter=False
                    )

                else:
                    print('Switching to dlc3 for {}, {}...'.format(eyecam_deinter_video, cfg['eye_DLC_project']))
                    print('You will get no updates until it is completed (running as subprocess)')
                    # Prep command to run in a different conda environment. This
                    # project was made in deeplabcut 3.0.0rc6, which cannot run in this
                    # python 3.9 environment, so switch over to a different conda env.

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

                print('  Running pose estimation for topdown camera video.')

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

        print('  Reading fluorescence data.')

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

        print('  Running spike inference.')

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

            # OMSI MCMC spike inference
            # OASIS (in calc_dFF) and the suite2p deconvolution still run and
            # are still written out; OMSI runs on top of them and its binned
            # rate becomes the `norm_spikes` every downstream analysis reads.
            # OMSI returns discrete spike TIMES in seconds, so they are binned
            # onto the 2P frame times rather than assumed to line up index-wise
            # -- twopT comes from the ScanImage header and is not perfectly
            # uniform.
            if cfg.get('use_omsi_spikes', True):
                _F  = np.asarray(twop_dict['raw_F'], dtype=float)
                _Fn = np.asarray(twop_dict['raw_Fneu'], dtype=float)
                _nfr = min(_F.shape[1], len(twopT))
                _fs = float(1.0 / np.nanmedian(np.diff(twopT[:_nfr]))) \
                    if _nfr > 1 else float(cfg.get('twop_rate', 7.49))
                print('  OMSI spike inference on {} cells x '
                      '{} frames at {:.3f} Hz (conda env '
                      '{!r})...'.format(_F.shape[0], _nfr, _fs, cfg.get("omsi_env", "spikeinf")))
                _t0 = time.time()
                _omsi = run_omsi(_F[:, :_nfr], _Fn[:, :_nfr], _fs,
                                 env=cfg.get('omsi_env', 'spikeinf'),
                                 f_corr=0.7)
                if _omsi['spike_times'] is not None:
                    _rate = bin_spike_times_to_frames(
                        _omsi['spike_times'], twopT[:_nfr])
                    twop_dict['omsi_spike_rate'] = _rate
                    if _omsi['prob_trace'] is not None:
                        twop_dict['omsi_prob_trace'] = np.asarray(
                            _omsi['prob_trace'], dtype=np.float32)
                    # Keep what norm_spikes used to be, so nothing that was
                    # already computed against it becomes unreproducible.
                    twop_dict['norm_spikes_s2p'] = normspikes
                    twop_dict['norm_spikes'] = _rate
                    _nsp = int(sum(len(x) for x in _omsi['spike_times']))
                    print('     Done in {:.0f}s: {} spikes, '
                          'mean rate {:.3f} Hz, '
                          '{:.1f}% of bins occupied'.format(time.time() - _t0, _nsp, np.nanmean(_rate), float(np.mean(_rate > 0)) * 100))
                    print('     norm_spikes is now the OMSI rate in Hz '
                          '(range 0-{:.1f}); the previous '
                          '[0,1]-scaled suite2p trace is kept as '
                          'norm_spikes_s2p. OASIS output is unchanged in '
                          'oasis_spks.'.format(np.nanmax(_rate)))
                else:
                    print('     OMSI unavailable -- norm_spikes left as the '
                          'suite2p trace. Nothing else changed.')

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

            topdown_pts_path = _resolve_topdown_dlc(
                topdown_video, rpath, cfg['topdown_DLC_search_key'])

        if sn:

            sn_stimT = pd.read_csv(sn_stimT_path)['psychopy_time'].to_numpy()

        if not sn:

            print('  Measuring locomotor behavior.')

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
                    print('  No saved arena_props.yaml for {}. '
                          'Skipping this recording in batch mode (the '
                          'interactive arena/pillar annotation GUI cannot '
                          'be opened unattended). Annotate it manually '
                          'with a single-recording preprocess() run, then '
                          're-run the batch.'.format(rpath))
                    continue
                arena_dict = top_cam.track_arena()
                arena_dict = fix_dict_dtype(arena_dict, float)
                write_yaml(arena_yaml_path, arena_dict)

            pxls2cm = arena_dict['pxls2cm']
            top_xyl, top_tracking_dict = top_cam.track_body(pxls2cm)

            print('  Measuring pupil orientation via ellipse fit.')

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

            print('  Aligning eye camera data streams to 2P and behavior data using TTL voltage.')

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

            # eye/IMU timing offset
            # The pupil traces lag the IMU by about two eye-camera samples
            # (~32 ms).  Correcting it takes the slow-phase horizontal VOR gain
            # from 0.11 to 0.31 and its correlation from 0.25 to 0.59, with a
            # clean peak at -2 samples in 10 of 14 sessions (-1 in the other 4)
            # and a symmetric falloff either side.
            #
            # This used to be applied downstream as `np.roll(dEye, -2)` in six
            # places across PETH, multicell_GLM and eyehead_revcorr.  It is now
            # applied here, once, so every consumer inherits it and none can
            # drift.  Shift with NaN fill rather than np.roll, so samples do not
            # wrap from one end of the recording to the other.
            _lag = cfg.get('eye_imu_lag_samples', -2)
            if _lag:
                def _shift_nan(a, k):
                    """ Shift array by k samples, filling vacated end with NaN. """

                    a = np.asarray(a, dtype=float)
                    out = np.full_like(a, np.nan)
                    if k < 0:
                        out[:k] = a[-k:]
                    elif k > 0:
                        out[k:] = a[:-k]
                    else:
                        out[:] = a
                    return out
                theta = _shift_nan(theta, _lag)
                phi = _shift_nan(phi, _lag)
                print('  Eye/IMU timing: theta and phi shifted {} '
                      'samples to align with the IMU.'.format(_lag))

            print('  Calculating retinocentric and egocentric orientations.')

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
                print('     Trimming to match: top tracking {} frames, '
                      '2P {} frames to {}'.format(_n_track, _n_twop, _n_keep))

            if _n_track > _n_keep:
                # Top tracking is too long for spike data.
                learx = learx[:_n_keep]
                leary = leary[:_n_keep]
                rearx = rearx[:_n_keep]
                reary = reary[:_n_keep]
                yaw = yaw[:_n_keep]

            if _n_twop > _n_keep:
                # Spike data is too long for top tracking.
                _n_drop = _n_twop - _n_keep
                twop_dict['twopT'] = twop_dict['twopT'][:len(twop_dict['twopT']) - _n_drop]
                twopT = twop_dict['twopT']
                for _key in ('raw_F', 'norm_F', 'raw_Fneu', 'raw_dFF', 'norm_dFF',
                             'denoised_dFF', 's2p_spks', 'oasis_spks',
                             'norm_spikes', 'norm_spikes_s2p',
                             'omsi_spike_rate', 'omsi_prob_trace',
                             'dFF_transients'):
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

                print('  Reading IMU data in and performing sensor fusion.')

                imu_df, imuT = read_IMU(imu_vals, imu_timestamps)

                # IMU to head mount rotation
                # The IMU is bolted to the eye camera, so read_IMU's fused roll
                # and pitch describe the CAMERA ASSEMBLY's tilt, not the head's.
                # At the -65 deg mount the two mix hard: the "pitch" is
                # 0.42 x head pitch + 0.91 x head roll.  Correct it here, at the
                # source, so every consumer -- the reconstruction, ffNLE,
                # boundary tuning, decoding -- sees head axes.  Doing it in each
                # consumer instead is how stored and recomputed values drift
                # apart.
                #
                # Validated against the hand-annotated worldcam masks: aggregate
                # IoU 0.3553 to 0.3684, mean per-frame 0.2963 to 0.3275,
                # detection MCC 0.750 to 0.784, with the optimum landing on the
                # physical mount angle when swept blind.
                _mount_az = cfg.get('imu_mount_azimuth_deg', MOUNT_AZIMUTH_DEG)
                if _mount_az is not None and 'roll' in imu_df and 'pitch' in imu_df:
                    _r, _p = rotate_tilt_to_head(
                        imu_df['roll'].to_numpy(dtype=float),
                        imu_df['pitch'].to_numpy(dtype=float),
                        _mount_az)
                    _dp = float(np.nanmedian(np.abs(_p - imu_df['pitch'].to_numpy(dtype=float))))
                    _dr = float(np.nanmedian(np.abs(_r - imu_df['roll'].to_numpy(dtype=float))))
                    imu_df['roll'] = _r
                    imu_df['pitch'] = _p
                    print('  IMU-to-head mount rotation (azimuth {:+.1f} '
                          'deg): pitch changed by a median {:.1f} deg, roll '
                          'by {:.1f} deg.'.format(_mount_az, _dp, _dr))
                else:
                    _mount_az = None
                    print('  IMU-to-head mount rotation DISABLED; roll/pitch '
                          'stay in camera-assembly axes.')

                print('  Aligning IMU to 2P and other behavior data.')

                imu_dict = align_crop_IMU(imu_df, imuT, apply_t0, apply_tEnd, eyeT[eyeStart:eyeEnd], twopT)

            print('  Calibrating eye-camera angular offset via VOR.')

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
                print('  Using TTL to calculate light/dark state vector...')
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
            # Marker so downstream code knows roll/pitch are already in head
            # coordinates and does not rotate them a second time.  NaN means the
            # rotation was not applied.
            preprocessed_dict['eye_imu_lag_samples'] = float(_lag)
            preprocessed_dict['imu_mount_azimuth_deg'] = (
                float(_mount_az) if (cfg['imu'] and _mount_az is not None)
                else np.nan)

            preprocessed_dict['theta_interp'] = theta_interp
            preprocessed_dict['phi_interp'] = phi_interp
            preprocessed_dict['theta_trim'] = theta_trim
            preprocessed_dict['phi_trim'] = phi_trim
            preprocessed_dict['eyeT_trim'] = eyeT_trim
            preprocessed_dict['head_x'] = headx
            preprocessed_dict['head_y'] = heady

            # Was 'tldk', will need to build in a flag to make sure.
            preprocessed_dict['ltdk'] = ltdk
            # The one that was used in existing preprocessing files is found.

            if ltdk:
                preprocessed_dict['ltdk_state_vec'] = ltdk_state_vec
                preprocessed_dict['light_onsets'] = light_onsets
                preprocessed_dict['dark_onsets'] = dark_onsets

            if len(cyclotorsion_dict.keys()) > 0:
                preprocessed_dict = {**preprocessed_dict, **cyclotorsion_dict}

            if cfg['imu']:
                preprocessed_dict = {**preprocessed_dict, **imu_dict}

            # Apply IMU disconnection trimming regardless of cfg['imu'] --
            # check_and_trim_imu_disconnect is a no-op if gyro_z_trim is absent.
            print('  Checking for IMU disconnection.')
            preprocessed_dict = check_and_trim_imu_disconnect(preprocessed_dict)

        # fm2p.run_preprocessing_diagnostics(preprocessed_dict)

            # revcorr_dict = fm2p.calc_revcorr(preprocessed_dict, save=False, IMU=cfg['imu'])
            # preprocessed_dict = {**preprocessed_dict, **revcorr_dict}

        if not sn and cfg['imu']:

            print('  Detrending integrated gyro along z-axis.')
            # use integral of gyro_z and use topdown measure of head yaw as a template
            # to correct for the accumulated error.  The smoothing window has come
            # down twice for the same reason: 120s let the corrected yaw wander
            # ~90-100deg within a recording, 5s fixed that on a 2P session but
            # still left 135deg of drift on the worldcam session (where topdown
            # yaw is 35% NaN, so anchors are sparse).  0.5s -- the function
            # default now -- leaves 2.4deg.  See fm2p/utils/imu.py for the
            # ground-truth measurement behind those numbers.  The fusion itself
            # matches get_retinal_image._fuse_head_yaw, so the reconstruction's
            # --refit_yaw reproduces this rather than replacing it.
            upsampled_yaw = detrend_gyroz_weighted_gaussian(preprocessed_dict,
                                                            gaussian_weight=1.)
            preprocessed_dict['upsampled_yaw'] = upsampled_yaw

            # print(' to Calculating PETHS.')
            # peth_imu_dict = fm2p.calc_PETHs(preprocessed_dict)
            # preprocessed_dict = {**preprocessed_dict, **peth_imu_dict}

            print('  Computing eye/head movement kinematics.')
            saccade_dict = calc_eye_head_movement_times(preprocessed_dict)
            preprocessed_dict.update(saccade_dict)

        if sn:

            preprocessed_dict = twop_dict
            preprocessed_dict['stimT'] = sn_stimT

        print('  Inferring spike times via MCMC deconvolution...')
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
        print('  Writing preprocessed data to {}...'.format(_savepath))
        write_h5(_savepath, preprocessed_dict)

        # If a real config path was given, write some new data into the dictionary and
        # then save a new preprocessed_config
        cfg['{}_preproc_file'.format(rname)] = _savepath
        if not sn:
            cfg['{}_topdown_video'.format(rname)] = topdown_video
            cfg['{}_eye_video'.format(rname)] = eyecam_deinter_video


    # if cfg_path is not None:

    #     print(' to Updating config yaml file.')

    #     # Write a new version of the config file. Maybe change this to overwrite previous?
    #     _newsavepath = os.path.join(os.path.split(cfg_path)[0], 'preprocessed_config.yaml')
    #     write_yaml(_newsavepath, cfg)


def preprocess_batch(batch_path=DEFAULT_BATCH_PATH, start_at=1,
                     config_path=DEFAULT_BATCH_CONFIG):
    """ Re-preprocess every already-preprocessed fm recording under `batch_path`,
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
    print('Batch config: {}'.format(config_path))
    print('  run_deinterlace={}, '
          'run_pose_estimation={}, '
          'imu={}, ltdk={}, '
          'hp_plug={}'.format(base_cfg["run_deinterlace"], base_cfg["run_pose_estimation"], base_cfg["imu"], base_cfg["ltdk"], base_cfg["hp_plug"]))
    print('  eye_DLC_project={}'.format(base_cfg["eye_DLC_project"]))
    print('  top_DLC_project={}'.format(base_cfg["top_DLC_project"]))

    preproc_files = find('*_preproc.h5', batch_path, MR=False, retempty=True) or []
    rec_dirs = sorted({
        os.path.dirname(p) for p in preproc_files
        if 'boundary' not in p
        and os.path.basename(os.path.dirname(p)).startswith('fm')
    })
    n_total = len(rec_dirs)

    print('Found {} fm recording(s) with an existing preproc.h5 '
          'under {}.'.format(n_total, batch_path))
    if start_at > 1:
        print('Resuming at recording {}/{} -- skipping the '
              'first {} (already processed cleanly).'.format(start_at, n_total, start_at - 1))

    results = {}
    for i in range(start_at - 1, n_total):
        rec_dir     = rec_dirs[i]
        rec_name    = os.path.basename(rec_dir)
        session_dir = os.path.dirname(rec_dir)

        print('\n=== [{}/{}] Batch recording: {} ==='.format(i + 1, n_total, rec_dir))

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
                print('  WARNING: config says imu={} but IMU '
                      'csv files are {} in '
                      '{}. Using the config value.'.format(base_cfg["imu"], ("present" if has_imu else "absent"), rec_dir))
            if bool(base_cfg['ltdk']) != has_ltdk:
                print('  WARNING: config says ltdk={} but '
                      'light/dark TTL csv files are '
                      '{} in {}. '
                      'Using the config value.'.format(base_cfg["ltdk"], ("present" if has_ltdk else "absent"), rec_dir))

            # Everything except the two structural keys below comes from
            # config_path, so every recording in the batch is processed
            # identically.
            cfg = copy.deepcopy(base_cfg)
            cfg['spath']              = session_dir
            cfg['include_recordings'] = [rec_name]

            preprocess(cfg=cfg, batch_mode=True)
            results[rec_dir] = 'ok'
        except Exception as e:
            print('  ERROR reprocessing {}: {}'.format(rec_dir, e))
            results[rec_dir] = e

    n_ok  = sum(1 for v in results.values() if v == 'ok')
    n_err = len(results) - n_ok
    print('\nBatch preprocessing complete: {} succeeded, {} failed '
          'out of {} recordings.'.format(n_ok, n_err, len(results)))
    if n_err:
        print('Failed recordings:')
        for k, v in results.items():
            if v != 'ok':
                print('  {}: {}'.format(k, v))

    return results


if __name__ == '__main__':

    _cli_parser = argparse.ArgumentParser()
    _cli_parser.add_argument('-d', '--dir', type=str, default=None,
                              help='Session directory containing one subdirectory '
                                   'per recording (freely moving and sparse noise).')
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
        if _cli_args.dir is None:
            _cli_parser.error('-d/--dir is required (session directory to preprocess).')
        preprocess(spath=_cli_args.dir)


# python -m fm2p.preprocess --batch --batch_path /home/dylan/Storage/freely_moving_data/V1PPC/cohort02_recordings