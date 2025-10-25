# -*- coding: utf-8 -*-
"""
Processing for Intertial Measurement Unit.

Functions
---------
_process_frame
    Helper for parallel processing of sensor fusion.
read_IMU
    Read IMU fo=rom csv files and perform sensor fusion to get out
    head pitch and roll.


Author: DMM, 2025
"""


from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing

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
    # IMU = fm2p.ImuOrientation()

    # for i in tqdm(range(n_samps)):
    #     roll_pitch[i] = IMU.process((
    #         df[['acc_x','acc_y','acc_z']].iloc[i].to_numpy(),
    #         df[['gyro_x','gyro_y','gyro_z']].iloc[i].to_numpy(),
    #     ))

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


def upsample_yaw(data, do_plot=False):

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

    return gyro_z_corrected