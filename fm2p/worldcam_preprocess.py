# -*- coding: utf-8 -*-
"""
fm2p/worldcam_preprocess.py

Preprocess worldcam recordings: align topdown, IMU and worldcam video without 2P data.

Functions
---------
_pillar_from_existing_h5
    Pillar annotation already stored in out_h5, or None.
_annotate_pillar
    Click the pillar outline on a mid-video frame; returns fitted circle and points.
_find_one
    Exactly one file matching pattern, else raise.
_read_video_pts
    Per-frame presentation timestamps (s) from the container, or None.
_get_video_fps_and_frames
    Frame rate and frame count from ffprobe.
_track_topdown
    Filtered topdown markers plus head position and yaw.
_align_ltdk
    Light/dark state and onsets from ltdk TTL, aligned to eye and 2P time.
worldcam_preprocess
    Align topdown, IMU and worldcam for a recording with no 2P data.
main
    Command-line entry point.


DMM, August 2026
"""


import os
import sys
import argparse
import subprocess
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob

_root = str(Path(__file__).resolve().parents[1])
if _root not in sys.path:
    sys.path.insert(0, _root)

import cv2

from fm2p.utils.files import open_dlc_h5, write_h5, read_h5
from fm2p.utils.time import (read_timestamp_file, read_timestamp_series,
                               interpT, find_closest_timestamp)
from fm2p.utils.helper import apply_liklihood_thresh, split_xyl
from fm2p.utils.filter import convfilt, nanmedfilt
from fm2p.utils.imu import read_IMU, detrend_gyroz_weighted_gaussian
from fm2p.utils.alignment import align_crop_IMU


# Fallback pillar annotation, so a recording can be preprocessed without opening
# the click-through GUI (which needs a display -- not available over SSH).
# Values read back from 251028_DMM_DMM000_fm_04_preproc.h5, i.e. the annotation
# that was actually used for that session, in native top-down pixels.
#
# Resolution order when annotate_pillar is False:
#   1. explicit pillar_centroid= / pillar_radius= arguments
#   2. the pillar already stored in an existing *_preproc.h5 in rec_dir
#   3. these constants
# Step 2 means re-running preprocessing on a session that has been annotated
# once keeps its own annotation rather than inheriting this one.
# Median-filter width on the top-down marker traces, which feed head_x, head_y
# AND head_yaw_deg.  At the 7.5 Hz top-down rate the old width of 7 was a 0.93 s
# window: it displaced the head point a median 21 px (90th pct 70 px, ~2 cm)
# from where DLC put it and removed 43% of the frame-to-frame motion.  Because
# head_yaw_deg is also built from these traces, that smoothing propagated into
# the gyro drift correction as well.  1 = disabled; the Kalman fusion in
# get_retinal_image is the only filter position needs.
TOPDOWN_MEDFILT_N = 1

DEFAULT_PILLAR_CX     = 1589.475556
DEFAULT_PILLAR_CY     = 818.346667
DEFAULT_PILLAR_RADIUS = 93.166543
DEFAULT_PILLAR_PTS_X  = [1583.7867, 1645.2267, 1679.3600, 1667.9822,
                         1601.9911, 1520.0711, 1490.4889, 1526.8978]
DEFAULT_PILLAR_PTS_Y  = [730.4533, 748.6578, 814.6489, 866.9867,
                         912.4978, 894.2933, 823.7511, 755.4844]


def _pillar_from_existing_h5(out_h5):
    """Pillar annotation already stored in out_h5, or None."""

    if not os.path.isfile(out_h5):
        return None
    try:
        with h5py.File(out_h5, 'r') as f:
            if 'pillar_centroid' not in f or 'pillar_radius' not in f:
                return None
            cx = float(f['pillar_centroid']['x'][()])
            cy = float(f['pillar_centroid']['y'][()])
            r = float(f['pillar_radius'][()])
            if not (np.isfinite(cx) and np.isfinite(cy) and r > 0):
                # A --no_annotate run wrote zeros; ignore it.
                return None
            px = np.asarray(f['pillar_x'][:], float).tolist() if 'pillar_x' in f else []
            py = np.asarray(f['pillar_y'][:], float).tolist() if 'pillar_y' in f else []
        return cx, cy, r, px, py
    except Exception as exc:
        print('  Could not read pillar from {}: {}'.format(os.path.basename(out_h5), exc))
        return None


def _annotate_pillar(video_path):
    """ Click the pillar outline on a mid-video frame; returns fitted circle and points. """

    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n_frames // 2)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f'Could not read frame from {video_path}')

    h, w = frame.shape[:2]
    scale = min(1.0, 1200 / w, 900 / h)
    disp_w, disp_h = int(w * scale), int(h * scale)
    base = cv2.resize(frame, (disp_w, disp_h))

    NPTS = 8
    pts_disp = []
    def _draw_phase1(img, pts):
        """ Draw clicked points and outline with instructions. """

        img = img.copy()
        for i, pt in enumerate(pts):
            cv2.circle(img, pt, 5, (0, 0, 255), -1)
            if i > 0:
                cv2.line(img, pts[i - 1], pt, (0, 255, 255), 1)
        if len(pts) == NPTS:
            cv2.line(img, pts[-1], pts[0], (0, 255, 255), 1)
        msg = (f'Click around pillar: {len(pts)}/{NPTS}  '
               f'(Backspace=undo  Enter=done)')
        cv2.putText(img, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2, cv2.LINE_AA)
        return img

    def _cb_phase1(event, x, y, flags, param):
        """ Mouse callback: add a point. """

        if event == cv2.EVENT_LBUTTONDOWN and len(pts_disp) < NPTS:
            pts_disp.append((x, y))

    win = 'Pillar annotation'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp_w, disp_h)
    cv2.setMouseCallback(win, _cb_phase1)
    print('Phase 1: click 8 points around the pillar. Backspace=undo, Enter=confirm.')

    while True:
        cv2.imshow(win, _draw_phase1(base, pts_disp))
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32) and len(pts_disp) == NPTS:
            break
        elif key == 8 and pts_disp:
            pts_disp.pop()
        elif key == ord('q'):
            cv2.destroyWindow(win)
            raise RuntimeError('Pillar annotation canceled.')

    poly = list(pts_disp)
    drag_state = {'active': False, 'ox': 0, 'oy': 0, 'start_poly': None}

    def _point_in_poly(x, y, polygon):
        """ Ray-casting point-in-polygon test. """

        n = len(polygon)
        inside = False
        px0, py0 = polygon[-1]
        for px1, py1 in polygon:
            if ((py1 > y) != (py0 > y)) and (x < (px0 - px1) * (y - py1) / (py0 - py1) + px1):
                inside = not inside
            px0, py0 = px1, py1
        return inside

    def _draw_phase2(img, p):
        """ Draw draggable polygon with instructions. """

        img = img.copy()
        pts_arr = np.array(p, dtype=np.int32)
        cv2.polylines(img, [pts_arr], True, (0, 255, 0), 2)
        for pt in p:
            cv2.circle(img, pt, 4, (0, 200, 0), -1)
        cv2.putText(img, 'Drag polygon to refine. Enter=confirm  Q=cancel',
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2, cv2.LINE_AA)
        return img

    def _cb_phase2(event, x, y, flags, param):
        """ Mouse callback: drag whole polygon. """

        if event == cv2.EVENT_LBUTTONDOWN:
            if _point_in_poly(x, y, poly):
                drag_state['active'] = True
                drag_state['ox'] = x
                drag_state['oy'] = y
                drag_state['start_poly'] = list(poly)
        elif event == cv2.EVENT_MOUSEMOVE and drag_state['active']:
            dx, dy = x - drag_state['ox'], y - drag_state['oy']
            for i, (px, py) in enumerate(drag_state['start_poly']):
                poly[i] = (px + dx, py + dy)
        elif event == cv2.EVENT_LBUTTONUP:
            drag_state['active'] = False

    cv2.setMouseCallback(win, _cb_phase2)
    print('Phase 2: drag polygon to refine. Enter=confirm.')

    while True:
        cv2.imshow(win, _draw_phase2(base, poly))
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32):
            break
        elif key == ord('q'):
            cv2.destroyWindow(win)
            raise RuntimeError('Pillar annotation canceled.')

    cv2.destroyWindow(win)

    px_orig = np.array([p[0] / scale for p in poly])
    py_orig = np.array([p[1] / scale for p in poly])
    cx = float(np.mean(px_orig))
    cy = float(np.mean(py_orig))
    radius = float(np.mean(np.sqrt((px_orig - cx) ** 2 + (py_orig - cy) ** 2)))

    print('  Pillar centroid: ({:.1f}, {:.1f})  radius: {:.1f} px'.format(cx, cy, radius))
    return cx, cy, radius, px_orig.tolist(), py_orig.tolist()


def _find_one(pattern, directory, desc='file'):
    """ Exactly one file matching pattern, else raise. """

    matches = glob(os.path.join(directory, pattern))
    if not matches:
        raise FileNotFoundError(
            f'Could not find {desc} matching "{pattern}" in {directory}')
    if len(matches) > 1:
        raise RuntimeError(
            f'Ambiguous {desc}: multiple files match "{pattern}" in {directory}:\n'
            + '\n'.join(matches))
    return matches[0]


def _read_video_pts(video_path):
    """Per-frame presentation timestamps (s) from the container, or None."""

    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'frame=pts_time', '-of', 'csv=p=0', video_path],
            capture_output=True, text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    pts = []
    for tok in result.stdout.split():
        tok = tok.strip().rstrip(',')
        if not tok:
            continue
        try:
            pts.append(float(tok))
        except ValueError:
            continue
    if len(pts) < 2:
        return None
    return np.asarray(pts, dtype=float)


def _get_video_fps_and_frames(video_path):
    """ Frame rate and frame count from ffprobe. """

    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=r_frame_rate,nb_frames',
         '-of', 'default=noprint_wrappers=1', video_path],
        capture_output=True, text=True
    )
    fps, n_frames = None, None
    for line in result.stdout.strip().splitlines():
        if line.startswith('r_frame_rate='):
            frac = line.split('=')[1]
            num, den = frac.split('/')
            fps = float(num) / float(den)
        elif line.startswith('nb_frames='):
            n_frames = int(line.split('=')[1])
    if fps is None:
        fps = 30.0
    return fps, n_frames


def _track_topdown(dlc_h5_path, pxls2cm, likelihood_thresh=0.9,
                   medfilt_n=TOPDOWN_MEDFILT_N):
    """ Likelihood-thresholded, optionally median-filtered topdown markers plus head position and yaw. """

    xyl, _ = open_dlc_h5(dlc_h5_path)
    x_vals, y_vals, likelihood = split_xyl(xyl)

    x_vals = apply_liklihood_thresh(x_vals, likelihood, threshold=likelihood_thresh)
    y_vals = apply_liklihood_thresh(y_vals, likelihood, threshold=likelihood_thresh)

    def _mf(v):
        """Median filter the marker trace, or pass it through when disabled."""

        v = np.asarray(v, dtype=float)
        if medfilt_n is None or int(medfilt_n) <= 1:
            return v
        return nanmedfilt(v, int(medfilt_n))[0]

    # Speed still gets its own heavier smoothing -- it is a derivative and only
    # used for the motion filter, never for the reconstruction.
    smooth_x = convfilt(nanmedfilt(x_vals['nose_x'], 7)[0], box_pts=20)
    smooth_y = convfilt(nanmedfilt(y_vals['nose_y'], 7)[0], box_pts=20)

    rear_x = _mf(x_vals['rightbar_x'])
    rear_y = _mf(y_vals['rightbar_y'])
    lear_x = _mf(x_vals['leftbar_x'])
    lear_y = _mf(y_vals['leftbar_y'])

    head_yaw = np.arctan2((lear_y - rear_y), (lear_x - rear_x)) + np.deg2rad(90)
    head_yaw_deg = np.rad2deg(head_yaw % (2 * np.pi))

    x_disp = np.diff((smooth_x * 60.0) / pxls2cm)
    y_disp = np.diff((smooth_y * 60.0) / pxls2cm)
    speed = np.sqrt(x_disp**2 + y_disp**2)

    headx = 0.5 * (rear_x + lear_x)
    heady = 0.5 * (rear_y + lear_y)

    return {
        'head_x':          headx,
        'head_y':          heady,
        'head_yaw_deg':    head_yaw_deg,
        'speed':           speed,
        'lear_x':          lear_x,
        'lear_y':          lear_y,
        'rear_x':          rear_x,
        'rear_y':          rear_y,
        'x':               smooth_x,
        'y':               smooth_y,
    }


def _align_ltdk(ltdk_v_path, ltdk_t_path, eyeT, twopT, eyeStart, eyeEnd):
    """ Light/dark state and onsets from ltdk TTL, aligned to eye and 2P time. """

    ltdkV = pd.read_csv(ltdk_v_path, header=None).squeeze().to_numpy()
    ltdkT = read_timestamp_series(
        pd.read_csv(ltdk_t_path, header=None).squeeze()
    )

    light_onsets_mask = np.diff(ltdkV) > np.nanmean(ltdkV)
    dark_onsets_mask  = np.diff(ltdkV) < -np.nanmean(ltdkV)

    eye_slice = eyeT[eyeStart:eyeEnd]

    def _times(mask):
        """ Eye-camera times closest to each TTL edge. """

        ts = ltdkT[:-1][mask]
        out = []
        for t in ts:
            try:
                out.append(find_closest_timestamp(eye_slice, t)[1])
            except Exception:
                pass
        return out

    eyet_light = _times(light_onsets_mask)
    eyet_dark  = _times(dark_onsets_mask)

    t0 = eyeT[eyeStart]
    twop_light = np.array([find_closest_timestamp(twopT, t - t0)[0]
                            for t in eyet_light])
    twop_dark  = np.array([find_closest_timestamp(twopT, t - t0)[0]
                            for t in eyet_dark])

    light_state = np.zeros(len(twopT), dtype=bool)
    for ind in range(len(twopT)):
        lo_past = twop_light[twop_light < ind]
        dk_past = twop_dark[twop_dark < ind]
        if len(lo_past) and len(dk_past):
            light_state[ind] = lo_past[-1] > dk_past[-1]
        elif len(lo_past):
            light_state[ind] = True
        elif len(dk_past):
            light_state[ind] = False
        else:
            if len(twop_light) and len(twop_dark):
                light_state[ind] = twop_light[0] < twop_dark[0]

    return light_state, twop_light, twop_dark


def worldcam_preprocess(
    rec_dir,
    pxls2cm=10.0,
    annotate_pillar=True,
    pillar_centroid=None,
    pillar_radius=None,
    likelihood_thresh=0.9,
    topdown_medfilt_n=TOPDOWN_MEDFILT_N,
    sigma_s=0.5,
):
    """ Align topdown, IMU and worldcam for a recording with no 2P data.

    Parameters
    ----------
    rec_dir : str
        Recording directory.
    pxls2cm : float
        Topdown pixels per cm.
    annotate_pillar : bool
        Open the pillar annotation GUI.
    pillar_centroid : tuple of float or None
        Pillar (x, y) in topdown pixels when not annotating.
    pillar_radius : float or None
        Pillar radius in topdown pixels.
    likelihood_thresh : float
        DLC likelihood cutoff.
    topdown_medfilt_n : int or None
        Median filter width on topdown markers; None or 1 disables.
    sigma_s : float
        Gaussian width in seconds for gyro-z drift correction.

    Returns
    -------
    out_h5 : str
        Written preproc .h5.
    """

    rec_dir = str(rec_dir)
    print('\n=== worldcam_preprocess: {} ===\n'.format(rec_dir))

    print('Locating files...')

    topdown_video   = _find_one('fm*.mp4', rec_dir, 'topdown video')
    topdown_dlc_h5  = _find_one('*DLC*.h5', rec_dir, 'topdown DLC h5')
    worldcam_ts     = _find_one('*_eyecam.csv', rec_dir, 'worldcam timestamps')
    ttl_v_path      = _find_one('*_logTTL.csv', rec_dir, 'TTL voltage')
    ttl_t_path      = _find_one('*_ttlTS.csv', rec_dir, 'TTL timestamps')
    imu_vals_path   = _find_one('*_IMUvals.csv', rec_dir, 'IMU values')
    imu_time_path   = _find_one('*_IMUtime.csv', rec_dir, 'IMU timestamps')

    ltdk_v_paths = glob(os.path.join(rec_dir, '*_ltdklogTTL.csv'))
    ltdk_t_paths = glob(os.path.join(rec_dir, '*_ltdkttlTS.csv'))
    has_ltdk = bool(ltdk_v_paths and ltdk_t_paths)
    if has_ltdk:
        ltdk_v_path = ltdk_v_paths[0]
        ltdk_t_path = ltdk_t_paths[0]
        print('  Found light/dark TTL files.')
    else:
        print('  No light/dark TTL files found -- using dummy onsets.')

    base = os.path.basename(worldcam_ts).replace('_eyecam.csv', '')
    out_h5 = os.path.join(rec_dir, f'{base}_preproc.h5')

    if annotate_pillar:
        print('\nAnnotating pillar...')
        pillar_cx, pillar_cy, pillar_radius, pillar_pts_x, pillar_pts_y = \
            _annotate_pillar(topdown_video)
    elif pillar_centroid is not None:
        pillar_cx, pillar_cy = float(pillar_centroid[0]), float(pillar_centroid[1])
        pillar_radius = float(pillar_radius if pillar_radius is not None
                              else DEFAULT_PILLAR_RADIUS)
        pillar_pts_x, pillar_pts_y = [], []
        print('\nPillar from arguments: ({:.1f}, {:.1f}) px, '
              'r={:.1f} px'.format(pillar_cx, pillar_cy, pillar_radius))
    else:
        _prev = _pillar_from_existing_h5(out_h5)
        if _prev is not None:
            pillar_cx, pillar_cy, pillar_radius, pillar_pts_x, pillar_pts_y = _prev
            print('\nPillar reused from existing {}: '
                  '({:.1f}, {:.1f}) px, r={:.1f} px'.format(os.path.basename(out_h5), pillar_cx, pillar_cy, pillar_radius))
        else:
            pillar_cx, pillar_cy = DEFAULT_PILLAR_CX, DEFAULT_PILLAR_CY
            pillar_radius = DEFAULT_PILLAR_RADIUS
            pillar_pts_x = list(DEFAULT_PILLAR_PTS_X)
            pillar_pts_y = list(DEFAULT_PILLAR_PTS_Y)
            print('\nPillar from module defaults: ({:.1f}, '
                  '{:.1f}) px, r={:.1f} px'.format(pillar_cx, pillar_cy, pillar_radius))
            print('  NOTE: these came from 251028_DMM_DMM000_fm_04. Pass '
                  '--pillar_cx/--pillar_cy for a different rig or arena.')

    print('Reading topdown video metadata...')
    top_fps, n_top_frames_vid = _get_video_fps_and_frames(topdown_video)
    print('  Topdown: {} frames @ {:.4f} fps'.format(n_top_frames_vid, top_fps))

    print('Reading topdown DLC tracking...')
    top_dict = _track_topdown(topdown_dlc_h5, pxls2cm, likelihood_thresh,
                              medfilt_n=topdown_medfilt_n)
    print('  Top-down marker median filter: n={} '
          '({})'.format(topdown_medfilt_n, ("disabled" if topdown_medfilt_n <= 1 else '{:.2f} s window'.format(topdown_medfilt_n / 7.5))))
    n_top_frames = len(top_dict['head_yaw_deg'])
    print('  DLC frames: {}'.format(n_top_frames))

    if n_top_frames_vid is not None and n_top_frames != n_top_frames_vid:
        print('  WARNING: DLC frame count ({}) != video frame count '
              '({}). Using DLC count.'.format(n_top_frames, n_top_frames_vid))

    # Topdown timebase.  Prefer the container's real presentation timestamps:
    # arange(n)/nominal_fps drifts against the actual frame times (0.18 s by the
    # end of a 41 s clip on the reference recording, i.e. >1 topdown frame),
    # which mistimes head position and yaw against the 60 Hz eye/IMU data.
    top_pts = _read_video_pts(topdown_video)
    if top_pts is not None and len(top_pts) >= n_top_frames:
        twopT = top_pts[:n_top_frames]
        _drift = twopT - np.arange(n_top_frames) / top_fps
        print('  Using real video timestamps for twopT '
              '(drift vs arange/{:.4f}: max {:.0f} ms)'.format(top_fps, np.max(np.abs(_drift)) * 1000))
    else:
        twopT = np.arange(n_top_frames) / top_fps
        print('  WARNING: could not read per-frame timestamps from '
              '{} -- falling back to '
              'arange(n)/{:.4f}, which drifts against the true frame times.'.format(os.path.basename(topdown_video), top_fps))

    print('Aligning worldcam to TTL...')

    eyeT = read_timestamp_file(worldcam_ts, force_timestamp_shift=True)

    ttlV = pd.read_csv(ttl_v_path, header=None).squeeze().to_numpy()
    ttlT = read_timestamp_series(pd.read_csv(ttl_t_path, header=None).squeeze())

    if len(ttlV) != len(ttlT):
        print('  WARNING: TTL voltage length ({}) != TTL timestamp length '
              '({}). Using minimum.'.format(len(ttlV), len(ttlT)))
        n_ttl = min(len(ttlV), len(ttlT))
        ttlV, ttlT = ttlV[:n_ttl], ttlT[:n_ttl]

    ttl_high = np.argwhere(ttlV > 0).ravel()
    if len(ttl_high) == 0:
        raise RuntimeError('No TTL-high samples found in logTTL.csv.')
    apply_t0   = ttlT[ttl_high[0]]
    apply_tEnd = ttlT[ttl_high[-1]]

    eyeStart, _ = find_closest_timestamp(eyeT, apply_t0)
    eyeEnd,   _ = find_closest_timestamp(eyeT, apply_tEnd)
    eyeStart, eyeEnd = int(eyeStart), int(eyeEnd)

    eyeT_trim = eyeT[eyeStart:eyeEnd] - eyeT[eyeStart]
    n_eye = len(eyeT_trim)
    print('  eyeT_trim: {} frames  ({:.2f} s)'.format(n_eye, eyeT_trim[-1]))

    print('Filling theta/phi with zeros (worldcam = no pupil tracking)...')
    theta_full = np.zeros(len(eyeT), dtype=float)
    phi_full   = np.zeros(len(eyeT), dtype=float)

    theta_trim = theta_full[eyeStart:eyeEnd]
    phi_trim   = phi_full[eyeStart:eyeEnd]

    theta_interp = np.zeros(n_top_frames, dtype=float)
    phi_interp   = np.zeros(n_top_frames, dtype=float)

    print('Reading IMU data and running sensor fusion...')

    _imu_lines = Path(imu_vals_path).read_text().splitlines()
    if _imu_lines and _imu_lines[0].count(',') < 5:
        print('  Fixing incomplete first IMU row (prepending 0).')
        _imu_lines[0] = '0,' + _imu_lines[0]
        import io
        _fixed = '\n'.join(_imu_lines)
        imu_df, imuT = read_IMU(io.StringIO(_fixed), imu_time_path)
    else:
        imu_df, imuT = read_IMU(imu_vals_path, imu_time_path)

    print('Aligning IMU to recording window...')
    imu_dict = align_crop_IMU(imu_df, imuT, apply_t0, apply_tEnd,
                               eyeT[eyeStart:eyeEnd], twopT)

    print('Detrending integrated gyro Z (yaw upsampling)...')
    combined = {
        **top_dict,
        **imu_dict,
        'twopT':    twopT,
        'eyeT_trim': eyeT_trim,
    }
    upsampled_yaw = detrend_gyroz_weighted_gaussian(combined, sigma_s=sigma_s,
                                                     gaussian_weight=1.0)

    if has_ltdk:
        print('Aligning light/dark TTL...')
        try:
            light_state, light_onsets, dark_onsets = _align_ltdk(
                ltdk_v_path, ltdk_t_path, eyeT, twopT, eyeStart, eyeEnd)
            if len(light_onsets) == 0:
                print('  No light/dark transitions found -- using dummy onsets.')
                has_ltdk = False
        except Exception as e:
            print('  WARNING: ltdk alignment failed ({}). Using dummy onsets.'.format(e))
            has_ltdk = False

    if not has_ltdk:

        light_state  = np.ones(n_top_frames, dtype=bool)
        light_onsets = np.array([0], dtype=int)
        dark_onsets  = np.array([n_top_frames - 1], dtype=int)

    print('Assembling preproc dict...')
    preproc = {}

    preproc['twopT']         = twopT
    preproc['head_x']        = top_dict['head_x']
    preproc['head_y']        = top_dict['head_y']
    preproc['head_yaw_deg']  = top_dict['head_yaw_deg']
    preproc['speed']         = top_dict['speed']
    preproc['lear_x']        = top_dict['lear_x']
    preproc['lear_y']        = top_dict['lear_y']
    preproc['rear_x']        = top_dict['rear_x']
    preproc['rear_y']        = top_dict['rear_y']
    preproc['x']             = top_dict['x']
    preproc['y']             = top_dict['y']

    preproc['eyeT_trim']     = eyeT_trim
    preproc['eyeT_startInd'] = eyeStart
    preproc['eyeT_endInd']   = eyeEnd
    preproc['theta']         = theta_full
    preproc['phi']           = phi_full
    preproc['theta_trim']    = theta_trim
    preproc['phi_trim']      = phi_trim
    preproc['theta_interp']  = theta_interp
    preproc['phi_interp']    = phi_interp

    preproc.update(imu_dict)

    preproc['upsampled_yaw'] = upsampled_yaw

    preproc['ltdk']           = has_ltdk
    preproc['ltdk_state_vec'] = light_state
    preproc['light_onsets']   = light_onsets
    preproc['dark_onsets']    = dark_onsets

    preproc['pxls2cm']           = float(pxls2cm)
    preproc['pillar_centroid']   = {'x': pillar_cx, 'y': pillar_cy}
    preproc['pillar_radius']     = pillar_radius
    preproc['pillar_x']          = np.array(pillar_pts_x)
    preproc['pillar_y']          = np.array(pillar_pts_y)

    print('Writing to {}...'.format(out_h5))
    write_h5(out_h5, preproc)
    print('Done.')
    return out_h5


def main():
    """ Command-line entry point. """

    parser = argparse.ArgumentParser(
        description='Worldcam preprocessing -- aligns topdown+IMU+worldcam without 2P data.'
    )
    parser.add_argument('--rec_dir', type=str,
                        help='Path to recording directory.')
    # Was default of 10.0.
    parser.add_argument('--pxls2cm', type=float, default=34.34470046082949,
                        help='Topdown pixels-to-cm conversion factor.')
    parser.add_argument('--no_annotate', action='store_true',
                        help='Skip the pillar annotation GUI (which needs a '
                             'display). The pillar is then taken from '
                             '--pillar_cx/--pillar_cy if given, else from an '
                             'existing _preproc.h5 in rec_dir, else from the '
                             'module defaults.')
    parser.add_argument('--pillar_cx', type=float, default=None,
                        help='Pillar centroid x in top-down pixels. Implies '
                             '--no_annotate.')
    parser.add_argument('--pillar_cy', type=float, default=None,
                        help='Pillar centroid y in top-down pixels.')
    parser.add_argument('--pillar_radius', type=float, default=None,
                        help=f'Pillar radius in top-down pixels '
                             f'(default: {DEFAULT_PILLAR_RADIUS:.1f}).')
    parser.add_argument('--likelihood_thresh', type=float, default=0.70,
                        help='DLC likelihood threshold for topdown tracking (default: 0.6).')
    parser.add_argument('--topdown_medfilt_n', type=int,
                        default=TOPDOWN_MEDFILT_N,
                        help='Median filter width on the top-down markers '
                             '(default: %(default)s = disabled). These feed '
                             'head_x/head_y and head_yaw_deg.')
    parser.add_argument('--sigma_s', type=float, default=0.5,
                        help='Gaussian smoothing half-width (s) for IMU yaw '
                             'detrending (default: %(default)s). Larger values '
                             'let integrated-gyro drift through: 5 s left 135 '
                             'deg of drift over a 41 s recording.')
    args = parser.parse_args()

    _cent = None
    if args.pillar_cx is not None and args.pillar_cy is not None:
        _cent = (args.pillar_cx, args.pillar_cy)
    worldcam_preprocess(
        rec_dir=args.rec_dir,
        pxls2cm=args.pxls2cm,
        annotate_pillar=(not args.no_annotate) and _cent is None,
        pillar_centroid=_cent,
        pillar_radius=args.pillar_radius,
        likelihood_thresh=args.likelihood_thresh,
        topdown_medfilt_n=args.topdown_medfilt_n,
        sigma_s=args.sigma_s,
    )


if __name__ == '__main__':

    main()
