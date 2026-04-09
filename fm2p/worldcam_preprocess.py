
import os
import sys
import argparse
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob

_root = str(Path(__file__).resolve().parents[1])
if _root not in sys.path:
    sys.path.insert(0, _root)

from fm2p.utils.files import open_dlc_h5, write_h5, read_h5
from fm2p.utils.time import (read_timestamp_file, read_timestamp_series,
                               interpT, find_closest_timestamp)
from fm2p.utils.helper import apply_liklihood_thresh, split_xyl
from fm2p.utils.filter import convfilt, nanmedfilt
from fm2p.utils.imu import read_IMU, detrend_gyroz_weighted_gaussian
from fm2p.utils.alignment import align_crop_IMU


def _find_one(pattern, directory, desc='file'):

    matches = glob(os.path.join(directory, pattern))
    if not matches:
        raise FileNotFoundError(
            f'Could not find {desc} matching "{pattern}" in {directory}')
    if len(matches) > 1:
        raise RuntimeError(
            f'Ambiguous {desc}: multiple files match "{pattern}" in {directory}:\n'
            + '\n'.join(matches))
    return matches[0]


def _get_video_fps_and_frames(video_path):

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


def _track_topdown(dlc_h5_path, pxls2cm, likelihood_thresh=0.9):
    
    xyl, _ = open_dlc_h5(dlc_h5_path)
    x_vals, y_vals, likelihood = split_xyl(xyl)

    x_vals = apply_liklihood_thresh(x_vals, likelihood, threshold=likelihood_thresh)
    y_vals = apply_liklihood_thresh(y_vals, likelihood, threshold=likelihood_thresh)

    smooth_x = convfilt(nanmedfilt(x_vals['nose_x'], 7)[0], box_pts=20)
    smooth_y = convfilt(nanmedfilt(y_vals['nose_y'], 7)[0], box_pts=20)

    rear_x = nanmedfilt(x_vals['rightbar_x'], 7)[0]
    rear_y = nanmedfilt(y_vals['rightbar_y'], 7)[0]
    lear_x = nanmedfilt(x_vals['leftbar_x'], 7)[0]
    lear_y = nanmedfilt(y_vals['leftbar_y'], 7)[0]

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

    ltdkV = pd.read_csv(ltdk_v_path, header=None).squeeze().to_numpy()
    ltdkT = read_timestamp_series(
        pd.read_csv(ltdk_t_path, header=None).squeeze()
    )

    light_onsets_mask = np.diff(ltdkV) > np.nanmean(ltdkV)
    dark_onsets_mask  = np.diff(ltdkV) < -np.nanmean(ltdkV)

    eye_slice = eyeT[eyeStart:eyeEnd]

    def _times(mask):
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
    pillar_x=0.0,
    pillar_y=0.0,
    likelihood_thresh=0.9,
    sigma_s=120.0,
):

    rec_dir = str(rec_dir)
    print(f'\n=== worldcam_preprocess: {rec_dir} ===\n')

    print('Locating files...')

    topdown_video   = _find_one('*.mp4', rec_dir, 'topdown video')
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
        print('  No light/dark TTL files found — using dummy onsets.')

    base = os.path.basename(worldcam_ts).replace('_eyecam.csv', '')
    out_h5 = os.path.join(rec_dir, f'{base}_preproc.h5')

    print('Reading topdown video metadata...')
    top_fps, n_top_frames_vid = _get_video_fps_and_frames(topdown_video)
    print(f'  Topdown: {n_top_frames_vid} frames @ {top_fps:.4f} fps')

    print('Reading topdown DLC tracking...')
    top_dict = _track_topdown(topdown_dlc_h5, pxls2cm, likelihood_thresh)
    n_top_frames = len(top_dict['head_yaw_deg'])
    print(f'  DLC frames: {n_top_frames}')

    if n_top_frames_vid is not None and n_top_frames != n_top_frames_vid:
        print(f'  WARNING: DLC frame count ({n_top_frames}) != video frame count '
              f'({n_top_frames_vid}). Using DLC count.')

    twopT = np.arange(n_top_frames) / top_fps   # relative, starts at 0

    print('Aligning worldcam to TTL...')

    eyeT = read_timestamp_file(worldcam_ts, force_timestamp_shift=True)

    ttlV = pd.read_csv(ttl_v_path, header=None).squeeze().to_numpy()
    ttlT = read_timestamp_series(pd.read_csv(ttl_t_path, header=None).squeeze())

    if len(ttlV) != len(ttlT):
        print(f'  WARNING: TTL voltage length ({len(ttlV)}) != TTL timestamp length '
              f'({len(ttlT)}). Using minimum.')
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

    eyeT_trim = eyeT[eyeStart:eyeEnd] - eyeT[eyeStart]   # relative, starts at 0
    n_eye = len(eyeT_trim)
    print(f'  eyeT_trim: {n_eye} frames  ({eyeT_trim[-1]:.2f} s)')

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
                print('  No light/dark transitions found — using dummy onsets.')
                has_ltdk = False
        except Exception as e:
            print(f'  WARNING: ltdk alignment failed ({e}). Using dummy onsets.')
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

    preproc['pxls2cm']          = float(pxls2cm)
    preproc['pillar_centroid']  = {'x': float(pillar_x), 'y': float(pillar_y)}

    print(f'Writing -> {out_h5}')
    write_h5(out_h5, preproc)
    print('Done.')
    return out_h5


def main():
    parser = argparse.ArgumentParser(
        description='Worldcam preprocessing — aligns topdown+IMU+worldcam without 2P data.'
    )
    parser.add_argument('rec_dir', type=str,
                        help='Path to recording directory.')
    parser.add_argument('--pxls2cm', type=float, default=10.0,
                        help='Topdown pixels-to-cm conversion factor (default: 10.0).')
    parser.add_argument('--pillar_x', type=float, default=0.0,
                        help='Pillar centroid X in topdown pixels (default: 0).')
    parser.add_argument('--pillar_y', type=float, default=0.0,
                        help='Pillar centroid Y in topdown pixels (default: 0).')
    parser.add_argument('--likelihood_thresh', type=float, default=0.9,
                        help='DLC likelihood threshold for topdown tracking (default: 0.9).')
    parser.add_argument('--sigma_s', type=float, default=120.0,
                        help='Gaussian smoothing half-width (s) for IMU yaw detrending (default: 120).')
    args = parser.parse_args()

    worldcam_preprocess(
        rec_dir=args.rec_dir,
        pxls2cm=args.pxls2cm,
        pillar_x=args.pillar_x,
        pillar_y=args.pillar_y,
        likelihood_thresh=args.likelihood_thresh,
        sigma_s=args.sigma_s,
    )


if __name__ == '__main__':

    main()
