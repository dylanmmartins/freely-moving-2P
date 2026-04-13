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
from scipy.interpolate import interp1d

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


def detrend_gyroz_weighted_gaussian(data, sigma_s=5.0, gaussian_weight=1.0):

    imuT   = data['imuT_trim']
    dt_s   = float(np.nanmedian(np.diff(imuT)))   # seconds per sample
    fps    = 1.0 / dt_s
    sigma_samples = sigma_s * fps

    yaw_gyro = np.cumsum(data['gyro_z_trim']) * dt_s # unwrapped


    if len(data['twopT']) == len(data['head_yaw_deg']):
        yaw_cam_2p = data['head_yaw_deg'].copy().astype(float)
    else:
        ts   = np.linspace(0, 1, len(data['twopT']))
        tref = np.linspace(0, 1, len(data['head_yaw_deg']))
        yaw_cam_2p = interp1d(tref, data['head_yaw_deg'], kind='linear')(ts)

    yaw_cam = interpT(yaw_cam_2p, data['twopT'], imuT).astype(float)
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
