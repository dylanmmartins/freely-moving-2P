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

import fm2p


def _process_frame(args):
    """
    Helper for parallel processing of sensor fusion.
    """
    acc, gyro = args
    imu = fm2p.ImuOrientation()  # local instance avoids state corruption across processes
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

    imuT = fm2p.read_timestamp_file(time_path)

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
    gyro_yaw_diff = gyro_z - fm2p.interpT(yaw, data['twopT'], data['imuT_trim'])

    drift_unwrapped = unwrap_degrees(gyro_yaw_diff%360)

    imuT = data['imuT_trim']
    p = np.polyfit(fm2p.nan_interp(imuT), fm2p.nan_interp(np.deg2rad(drift_unwrapped)), 1)
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


def detrend_gyroz_weighted_gaussian(data, sigma=5, gaussian_weight=1.0):

    # signal_a is the integral of gyro along z-axis
    # signal_b is the yaw from topdown camera
    # probably should name these more intuitively

    dt = 1 / np.nanmedian(np.diff(data['imuT_trim']))
    signal_a = np.cumsum(data['gyro_z_trim'])/dt
    signal_a = signal_a % 360

    if len(data['twopT']) == len(data['head_yaw_deg']):
        yaw = data['head_yaw_deg']
    else:
        timestamps = np.linspace(0, 1, num=len(data['twopT']))
        signal_time = np.linspace(0, 1, num=len(data['head_yaw_deg']))
        f = interp1d(signal_time, data['head_yaw_deg'], kind='linear')
        yaw = f(timestamps) # resampled to match real timestamp length

    signal_b = fm2p.interpT(yaw, data['twopT'], data['imuT_trim'])
    signal_b_initial = signal_b.copy()

    gyro_yaw_diff = signal_a - fm2p.interpT(yaw, data['twopT'], data['imuT_trim'])

    # nan out any huge jumps in the head yae from topdown. could
    # be poor tracking thorwing a crazy value in that gets past likelihoood threshold
    signal_b[np.concatenate([[0],(np.abs(fm2p.angular_diff_deg(signal_b))>15.)])] = np.nan

    a_rad = np.deg2rad(signal_a)
    b_rad = np.deg2rad(signal_b)
    n = len(a_rad)

    # handle nans in signal_b
    valid_b = ~np.isnan(b_rad)
    b_rad_filled = np.nan_to_num(b_rad, nan=0.0)

    # circular components
    sin_a, cos_a = np.sin(a_rad), np.cos(a_rad)
    sin_b, cos_b = np.sin(b_rad_filled), np.cos(b_rad_filled)

    # gaussian-smoothed circular means
    sin_mean_a = gaussian_filter1d(sin_a, sigma)
    cos_mean_a = gaussian_filter1d(cos_a, sigma)
    sin_mean_b = gaussian_filter1d(sin_b, sigma)
    cos_mean_b = gaussian_filter1d(cos_b, sigma)

    # Normalize for missing data
    norm = gaussian_filter1d(valid_b.astype(float), sigma)
    norm = np.maximum(norm, 1e-8)
    sin_mean_b /= norm
    cos_mean_b /= norm

    # convert means back to angles
    mean_a = np.arctan2(sin_mean_a, cos_mean_a)
    mean_b = np.arctan2(sin_mean_b, cos_mean_b)

    # wrapped difference between signals
    diff = np.angle(np.exp(1j * (b_rad - a_rad))) # rad

    # smooth difference w/out nans
    diff_filled = np.nan_to_num(diff, nan=0.0)
    smoothed_diff = gaussian_filter1d(diff_filled, sigma)
    smoothed_diff /= norm  # renormalize for missing data

    # apply weight
    corrected_rad = a_rad + gaussian_weight * smoothed_diff

    corrected_deg = np.rad2deg(np.mod(corrected_rad, 2 * np.pi))

    new_g_v_y_diff = corrected_deg - signal_b_initial

    detrended_out = {
        'igyro_corrected_deg': corrected_deg,
        'igyro_corrected_rad': corrected_rad,
        'igyro_yaw_raw_diff': gyro_yaw_diff,
        'igyro_yaw_detrended_diff': new_g_v_y_diff
    }

    return detrended_out


def check_and_trim_imu_disconnect(data_input):

    if isinstance(data_input, (str, Path)):
        data = fm2p.read_h5(data_input)
    elif isinstance(data_input, dict):
        data = data_input.copy()
    else:
        raise ValueError("Input must be a file path or a dictionary.")

    if 'gyro_z_trim' not in data:
        return data

    gyro_z = data['gyro_z_trim']
    
    # chk for flatline at disconnection
    diff = np.diff(gyro_z)
    is_flat = np.abs(diff) < 1e-6
    
    last_valid_idx = len(gyro_z) - 1
    for i in range(len(diff)-1, -1, -1):
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
    
    disconnect_time = imuT[last_valid_idx]
    
    twopT = data['twopT']
    valid_twop = twopT <= disconnect_time
    new_n_frames = np.sum(valid_twop)
    
    old_n_frames = len(twopT)
    data['twopT'] = twopT[:new_n_frames]
    
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            if v.shape[0] == old_n_frames:
                data[k] = v[:new_n_frames]
            elif v.ndim > 1 and v.shape[1] == old_n_frames:
                data[k] = v[:, :new_n_frames]
            elif k.endswith('_twop_interp') and v.shape[0] >= new_n_frames:
                data[k] = v[:new_n_frames]
                
    eyeT = data['eyeT']
    eyeStart = int(data['eyeT_startInd'])
    t0 = eyeT[eyeStart]
    target_abs = t0 + disconnect_time
    
    new_eyeEnd, _ = fm2p.find_closest_timestamp(eyeT, target_abs)
    if new_eyeEnd > data['eyeT_endInd']:
        new_eyeEnd = data['eyeT_endInd']
    data['eyeT_endInd'] = new_eyeEnd
    
    if 'eyeT_trim' in data:
        new_len_eye = new_eyeEnd - eyeStart
        data['eyeT_trim'] = data['eyeT_trim'][:new_len_eye]
        
        for k in ['theta_trim', 'phi_trim']:
            if k in data:
                data[k] = data[k][:new_len_eye]
                
    new_n_imu = last_valid_idx + 1
    old_n_imu = len(imuT)
    
    data['imuT_trim'] = imuT[:new_n_imu]
    
    for k in list(data.keys()):
        if k.endswith('_trim') and k not in ['eyeT_trim', 'theta_trim', 'phi_trim']:
             if isinstance(data[k], np.ndarray) and len(data[k]) == old_n_imu:
                 data[k] = data[k][:new_n_imu]

    return data
