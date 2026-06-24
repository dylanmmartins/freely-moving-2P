# -*- coding: utf-8 -*-
"""
Processing for Intertial Measurement Unit.

Functions
---------
_process_frame
    Helper for parallel processing of sensor fusion.
read_IMU
    Read IMU from csv files and perform sensor fusion to get out
    head pitch and roll.


Author: DMM, 2025
"""


from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from scipy.ndimage import gaussian_filter1d

from .sensor_fusion import ImuOrientation
from .time import read_timestamp_file, interpT
from .helper import nan_interp, angular_diff_deg
from .files import read_h5


def _process_frame(args):
    """
    Helper for parallel processing of sensor fusion.
    """
    acc, gyro = args
    imu = ImuOrientation()  # local instance avoids state corruption across processes
    return imu.process((acc, gyro))


def read_IMU(vals_path, time_path):
    """ Read IMU from csv files and perform sensor fusion
    to get out head pitch and roll.

    Parameters
    ----------
    vals_path : str
        Path to csv with IMU measurements with no header (i.e., first row is
        already measured values), and six columns in the order:  gyro_x, gyro_y,
        gyro_z, acc_x, acc_y, acc_z.
    time_path : str
        Path to timestamp file. They should be absolute (not relative) timestamps.
        Again, no header. They should be in the format:
            13:54:27.9693056
            13:54:27.9741312
            13:54:27.9812736
            13:54:27.9885184

    Notes
    -----
    * Sensor fusion scales poorly with length of the recording, and can be fairly
        slow (10+ min).
    * If you get a ParseError, check the values csv file and look for missing values
        in the first row. Pandas will be expecting just three columns and fail to
        read the rest of the file if it only finds three values (and no commas between
        the missing values), meaning it gets
            60.6545, 12.3543, 19.3543
        instead of
            , , , 60.6545, 12.3543, 19.3543
        Open the file and add zeros to the missing positions.
    """

    imuT = read_timestamp_file(time_path)

    df = pd.read_csv(vals_path, header=None)

    df.columns = [
        'acc_y',
        'acc_z',
        'acc_x',
        'gyro_y',
        'gyro_z',
        'gyro_x'
    ]

    # Invert sign of channels
    df['gyro_y'] = -df['gyro_y']
    df['gyro_z'] = -df['gyro_z']

    if np.abs(np.round(len(imuT)*2) - len(df)) < 10:
        print('  -> Dropping interleaved IMU NaNs')
        if np.isfinite(df['gyro_x'][0]):
            df = df.iloc[::2]
        elif np.isnan(df['gyro_x'][0]):
            df = df.iloc[1::2]
        df.reset_index(inplace=True)

    n_samps = np.size(df,0)
    roll_pitch = np.zeros([n_samps, 2])

    n_jobs = multiprocessing.cpu_count()
    chunk_size= 100
    
    args = [
        (
            df.loc[i, ['acc_x', 'acc_y', 'acc_z']].to_numpy(),
            df.loc[i, ['gyro_x', 'gyro_y', 'gyro_z']].to_numpy(),
        )
        for i in range(n_samps)
    ]

    print('  -> Starting sensor fusion with {} cores.'.format(n_jobs))

    with multiprocessing.Pool(processes=n_jobs) as pool:
        roll_pitch = list(
            tqdm(
                pool.imap(_process_frame, args, chunksize=chunk_size),
                total=n_samps,
                desc="Processing frames"
            )
        )

    roll_pitch = np.array(roll_pitch)
    df['roll'] = roll_pitch[:,0]
    df['pitch'] = roll_pitch[:,1]

    return df, imuT


def unwrap_degrees(angles, period=360, threshold=180):
    

    unwrapped = [angles[0]]
    offset = 0
    for prev, curr in zip(angles[:-1], angles[1:]):
        delta = curr - prev
        if delta > threshold:
            offset -= period
        elif delta < -threshold:
            offset += period
        unwrapped.append(curr + offset)

    return np.array(unwrapped)


def detrend_gyroz_simple_linear(data, do_plot=False):

    dt = 1 / np.nanmedian(np.diff(data['imuT_trim']))
    gyro_z = np.cumsum(data['gyro_z_trim'])/dt
    gyro_z = gyro_z % 360

    yaw = data['head_yaw_deg'][:-1]
    gyro_yaw_diff = gyro_z - interpT(yaw, data['twopT'], data['imuT_trim'])

    drift_unwrapped = unwrap_degrees(gyro_yaw_diff%360)

    imuT = data['imuT_trim']
    p = np.polyfit(nan_interp(imuT), nan_interp(np.deg2rad(drift_unwrapped)), 1)
    y_fit = np.polyval(p, imuT)

    gyro_z_corrected = gyro_z - (p[0] * imuT + p[1])

    if do_plot:
        
        fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2, 2, dpi=300, figsize=(8.5,5))
        ax1.plot(imuT, gyro_yaw_diff%360, 'k.', ms=1)
        ax1.set_xlim([0,1000])
        ax1.set_xlabel('time (sec)')
        ax1.set_ylabel('dGyro-yaw offset (deg)')
        ax1.set_ylim([0,360])
        ax1.set_title('difference drifts over recording')

        ax2.plot(imuT, np.deg2rad(drift_unwrapped), 'k.', ms=1)
        ax2.plot(imuT, y_fit, '.', color='tab:red', ms=1)
        ax2.set_xlabel('time (sec)')
        ax2.set_ylabel('dGyro-yaw offset (deg)')
        ax2.set_title('linear fit on unwrapped difference')

        ax3.plot(imuT, gyro_z, '.', ms=1, color='tab:cyan', label='raw')
        ax3.plot(imuT, gyro_z_corrected%360, '.', color='tab:pink', ms=1, label='corrected')
        ax3.set_xlim([0,60])
        ax3.set_ylim([0,360])
        ax3.set_xlabel('time (sec)')
        ax3.set_ylabel('dGyro (deg)')
        ax3.set_title('corrected signal now varies in distance from template')
        fig.legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', label='raw', markerfacecolor='tab:cyan', markersize=5),
            plt.Line2D([0], [0], marker='o', color='w', label='corrected', markerfacecolor='tab:pink', markersize=5),
        ], loc='lower left', frameon=False)

        ax4.plot(imuT, gyro_z, '.', ms=1, color='tab:cyan', label='raw')
        ax4.plot(imuT, gyro_z_corrected%360, '.', color='tab:pink', ms=1, label='corrected')
        ax4.set_xlim([0,700])
        ax4.set_ylim([0,360])
        ax4.set_xlabel('time (sec)')
        ax4.set_ylabel('dGyro (deg)')
        ax4.set_title('shown on a longer time scale')

        fig.tight_layout()
        fig.show()

    return gyro_z_corrected


def _interp_angle_deg(angle_deg, t_from, t_to):
    """Interpolate a wrapped (0-360 deg) angle by interpolating its
    sin/cos components instead of the raw degree value. Interpolating a
    wrapped angle directly produces a spurious ~180 deg swing every time
    it crosses the wrap boundary (e.g. 359 -> 0) -- on a real ~3600-frame
    test recording, head_yaw_deg wraps 101 times, so the old direct
    interpolation corrupted the camera-yaw reference used by
    detrend_gyroz_weighted_gaussian's drift correction at every single
    head turn, exactly when an accurate reference matters most.

    This is still plain linear interpolation (just wrap-safe), used only
    for the rare twopT/head_yaw_deg length-mismatch reconciliation below
    (near-equal lengths, not a real low-rate-to-high-rate upsampling). For
    the actual low-rate (twopT, ~7.5 Hz) -> high-rate (imuT) upsampling,
    see _kalman_rts_upsample_angle instead."""
    rad = np.deg2rad(np.asarray(angle_deg, dtype=float))
    cos_i = interpT(np.cos(rad), t_from, t_to)
    sin_i = interpT(np.sin(rad), t_from, t_to)
    return np.rad2deg(np.arctan2(sin_i, cos_i)) % 360.0


def _kalman_rts_smooth_1d(sig, fps, Q_frac=1e-3, R_frac=1e-2):
    """Constant-velocity Kalman filter + RTS backward smoothing pass over
    a signal with NaN gaps. Same method as
    fm2p/get_retinal_image.py:_kalman_rts_smooth (duplicated here rather
    than imported -- that module is a pipeline script, not a shared
    utility library), used there to upsample eye/head angular kinematics
    (yaw_eye, pitch_eye, roll_eye) onto the 60 Hz eye-camera grid. Unlike
    linear interpolation, the model carries a velocity state, so it
    extrapolates through gaps using the locally-estimated rate of change
    rather than just drawing a straight line between the surrounding
    measurements."""
    sig = np.asarray(sig, dtype=float)
    n = len(sig)
    finite = sig[np.isfinite(sig)]
    if len(finite) < 2:
        return sig.copy()
    sv = float(np.var(finite))
    if sv == 0.0:
        return np.where(np.isfinite(sig), sig, finite[0])

    dt = 1.0 / fps
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[Q_frac * sv * dt ** 2, 0.0],
                  [0.0,                   Q_frac * sv]])
    R = R_frac * sv

    i0 = int(np.where(np.isfinite(sig))[0][0])
    x = np.array([[sig[i0]], [0.0]])
    P = np.eye(2) * sv

    xs, Ps     = np.zeros((n, 2)), np.zeros((n, 2, 2))
    xprs, Pprs = np.zeros((n, 2)), np.zeros((n, 2, 2))

    for i in range(n):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        xprs[i] = x_pred[:, 0]
        Pprs[i] = P_pred
        x, P = x_pred, P_pred
        if np.isfinite(sig[i]):
            innov = sig[i] - float((H @ x)[0, 0])
            S = float((H @ P @ H.T)[0, 0]) + R
            K = (P @ H.T) / S
            x = x + K * innov
            P = (np.eye(2) - K @ H) @ P
        xs[i] = x[:, 0]
        Ps[i] = P

    FT = F.T
    for i in range(n - 2, -1, -1):
        G     = Ps[i] @ FT @ np.linalg.inv(Pprs[i + 1])
        xs[i] = xs[i] + G @ (xs[i + 1] - xprs[i + 1])

    return xs[:, 0]


def _kalman_rts_upsample_angle(angle_deg, src_t, dst_t, fps_dst=None, Q_frac=1e-3, R_frac=1e-2):
    """Upsample a wrapped (0-360 deg) angle from its low-rate timebase
    (src_t) onto a higher-rate target grid (dst_t) with a constant-
    velocity Kalman filter + RTS smoothing, instead of linear
    interpolation -- "really upsampling" rather than just connecting the
    dots. Unwraps first (unwrap_degrees) so the filter sees a continuous
    signal, places each low-rate sample at its nearest dst_t index
    (everything else NaN), then runs _kalman_rts_smooth_1d through the
    gaps. Re-wraps to 0-360 on return."""
    unwrapped = unwrap_degrees(np.asarray(angle_deg, dtype=float))

    gapped = np.full(len(dst_t), np.nan)
    idx = np.clip(np.searchsorted(dst_t, src_t), 0, len(dst_t) - 1)
    gapped[idx] = unwrapped

    if fps_dst is None:
        fps_dst = 1.0 / float(np.nanmedian(np.diff(dst_t)))
    smoothed = _kalman_rts_smooth_1d(gapped, fps_dst, Q_frac=Q_frac, R_frac=R_frac)
    return smoothed % 360.0


def detrend_gyroz_weighted_gaussian(data, sigma_s=5.0, gaussian_weight=1.0):

    imuT   = data['imuT_trim']
    dt_s   = float(np.nanmedian(np.diff(imuT)))   # seconds per sample
    fps    = 1.0 / dt_s
    sigma_samples = sigma_s * fps

    # np.cumsum propagates a single NaN through every sample after it --
    # gyro_z_trim has NaN gaps ranging from isolated dropped samples up to
    # an exact 50% alternating-sample pattern in some recordings, either of
    # which previously made the corrected yaw ~100% NaN from the first gap
    # onward. Linearly interpolate over NaNs before integrating; if the
    # whole trace is NaN this is a no-op and the existing all-NaN fallback
    # below still applies.
    gyro_z = np.asarray(data['gyro_z_trim'], dtype=float)
    nan_mask = np.isnan(gyro_z)
    if nan_mask.any() and not nan_mask.all():
        gyro_z = nan_interp(gyro_z)

    yaw_gyro = np.cumsum(gyro_z) * dt_s # unwrapped


    if len(data['twopT']) == len(data['head_yaw_deg']):
        yaw_cam_2p = data['head_yaw_deg'].copy().astype(float)
    else:
        ts   = np.linspace(0, 1, len(data['twopT']))
        tref = np.linspace(0, 1, len(data['head_yaw_deg']))
        yaw_cam_2p = _interp_angle_deg(data['head_yaw_deg'], tref, ts)

    yaw_cam = _kalman_rts_upsample_angle(yaw_cam_2p, data['twopT'], imuT, fps_dst=fps)
    yaw_cam_initial = yaw_cam.copy()

    jump = np.concatenate([[False], np.abs(angular_diff_deg(yaw_cam)) > 15.0])
    yaw_cam[jump] = np.nan

    offset_rad = np.angle(
        np.exp(1j * np.deg2rad(yaw_cam - yaw_gyro))
    )
    offset_deg = np.rad2deg(offset_rad)

    valid = np.isfinite(offset_deg)

    if valid.sum() == 0:

        raw_deg = yaw_gyro % 360
        return {
            'igyro_corrected_deg':       raw_deg,
            'igyro_corrected_rad':       np.deg2rad(yaw_gyro),
            'igyro_yaw_raw_diff':        yaw_gyro % 360 - yaw_cam_initial,
            'igyro_yaw_detrended_diff':  np.full_like(yaw_gyro, np.nan),
        }

    valid_idx   = np.where(valid)[0]
    offset_unwrapped_valid = unwrap_degrees(offset_deg[valid])

    offset_filled = np.interp(
        np.arange(len(imuT)),
        valid_idx,
        offset_unwrapped_valid,
    )

    drift = gaussian_filter1d(offset_filled, sigma_samples)

    yaw_corrected_unwrap = yaw_gyro + gaussian_weight * drift
    corrected_deg        = yaw_corrected_unwrap % 360

    gyro_yaw_raw_diff      = yaw_gyro % 360 - yaw_cam_initial
    igyro_yaw_detrended_diff = corrected_deg - yaw_cam_initial

    return {
        'igyro_corrected_deg':       corrected_deg,
        'igyro_corrected_rad':       np.deg2rad(yaw_corrected_unwrap),
        'igyro_yaw_raw_diff':        gyro_yaw_raw_diff,
        'igyro_yaw_detrended_diff':  igyro_yaw_detrended_diff,
    }


def check_and_trim_imu_disconnect(data_input):

    if isinstance(data_input, (str, Path)):
        data = read_h5(data_input)
    elif isinstance(data_input, dict):
        data = data_input.copy()
    else:
        raise ValueError("Input must be a file path or a dictionary.")

    if 'gyro_z_trim' not in data:
        return data

    gyro_z = data['gyro_z_trim']

    # Detect trailing flatline (constant value = disconnected sensor)
    diff = np.diff(gyro_z)
    is_flat = np.abs(diff) < 1e-6

    last_valid_idx = len(gyro_z) - 1
    for i in range(len(diff) - 1, -1, -1):
        if not is_flat[i]:
            last_valid_idx = i + 1
            break
    else:
        last_valid_idx = 0

    imuT = data['imuT_trim']
    flat_samples = len(gyro_z) - 1 - last_valid_idx

    if flat_samples <= 0:
        return data

    if len(imuT) > last_valid_idx:
        flat_duration = imuT[-1] - imuT[last_valid_idx]
    else:
        flat_duration = 0

    if flat_duration < 1.0:
        return data

    print(f"IMU disconnection detected. Trimming {flat_duration:.2f}s from end.")

    # disconnect_time is in seconds relative to the shared recording start
    # (same origin as eyeT_trim and imuT_trim).
    disconnect_time = imuT[last_valid_idx]

    twopT = data['twopT']
    new_n_frames = int(np.searchsorted(twopT, disconnect_time, side='right'))
    old_n_frames = len(twopT)
    data['twopT'] = twopT[:new_n_frames]

    for k, v in list(data.items()):
        if not isinstance(v, np.ndarray):
            continue
        if v.shape[0] == old_n_frames:
            data[k] = v[:new_n_frames]
        elif v.ndim > 1 and v.shape[1] == old_n_frames:
            data[k] = v[:, :new_n_frames]
        elif k.endswith('_twop_interp') and v.shape[0] >= new_n_frames:
            data[k] = v[:new_n_frames]

    # eyeT_trim and imuT_trim share the same relative time origin (TTL trigger),
    # so searchsorted gives the correct cut-point directly.
    if 'eyeT_trim' in data:
        new_len_eye = int(np.searchsorted(data['eyeT_trim'], disconnect_time,
                                          side='right'))

        # Update the end-index into the full (absolute) eyeT array.
        if 'eyeT_startInd' in data and 'eyeT_endInd' in data:
            new_eyeEnd = int(data['eyeT_startInd']) + new_len_eye
            if new_eyeEnd > data['eyeT_endInd']:
                new_eyeEnd = int(data['eyeT_endInd'])
            data['eyeT_endInd'] = new_eyeEnd

        eye_explicit = {'eyeT_trim', 'theta_trim', 'phi_trim'}
        for k in list(data.keys()):
            if not isinstance(data[k], np.ndarray):
                continue
            if k in eye_explicit or k.endswith('_eye_interp'):
                if len(data[k]) >= new_len_eye:
                    data[k] = data[k][:new_len_eye]

    new_n_imu = last_valid_idx + 1
    old_n_imu = len(imuT)
    data['imuT_trim'] = imuT[:new_n_imu]

    eye_skip = {'eyeT_trim', 'theta_trim', 'phi_trim'}
    for k in list(data.keys()):
        if k.endswith('_trim') and k not in eye_skip:
            v = data[k]
            if isinstance(v, np.ndarray) and len(v) == old_n_imu:
                data[k] = v[:new_n_imu]

    return data
