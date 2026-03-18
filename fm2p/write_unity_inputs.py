# -*- coding: utf-8 -*-



import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation as R

from .utils.paths import find, filter_file_search
from .utils.files import read_h5, write_h5
from .utils.time import read_timestamp_file, interpT
from .utils.helper import nan_interp, fix_dict_dtype
from .utils.imu import read_IMU
from .utils.cameras import deinterlace, run_pose_estimation
from .utils.topcam import Topcam
from .utils.gui_funcs import select_file, select_directory


def rotation_matrices(yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]])
    Rx = np.array([[1,   0,    0],
                   [0,  cr, -sr],
                   [0,  sr,  cr]])
    return Rz @ Ry @ Rx


def gaze_angles(eye_h_deg, eye_v_deg, head_yaw_deg, head_pitch_deg, head_roll_deg):
    # convert to radians
    eh = np.deg2rad(eye_h_deg)
    ev = np.deg2rad(eye_v_deg)
    yaw = np.deg2rad(head_yaw_deg)
    pitch = np.deg2rad(head_pitch_deg)
    roll = np.deg2rad(head_roll_deg)

    # eye vector in head coords (unit)
    v_eye = np.array([np.cos(ev)*np.cos(eh),
                      np.cos(ev)*np.sin(eh),
                      np.sin(ev)])

    R_head = rotation_matrices(yaw, pitch, roll)
    v_gaze = R_head @ v_eye

    # gaze orientations
    azimuth = np.arctan2(v_gaze[1], v_gaze[0])
    elevation = np.arctan2(v_gaze[2], np.hypot(v_gaze[0], v_gaze[1]))

    return np.rad2deg(azimuth), np.rad2deg(elevation)


def combine_camera_gaze(camera_euler, gaze_az_el):
    """ Combine camera posture with eye gaze orientation.
    Parameters:
        camera_euler: tuple of (pitch, yaw, roll) in degrees
        gaze_az_el: tuple of (azimuth, elevation) in degrees
    Returns:
        combined_euler: tuple of (pitch, yaw, roll) in degrees
    """
    pitch, yaw, roll = camera_euler
    azimuth, elevation = gaze_az_el
    
    # 1. Camera rotation in world space
    camera_rot = R.from_euler('xyz', [pitch, yaw, roll], degrees=True)
    
    # 2. Gaze rotation relative to camera (local X=right, Y=up)
    # Elevation rotates around X, Azimuth around Y
    gaze_rot = R.from_euler('yx', [azimuth, -elevation], degrees=True)  # notice order YX
    
    # 3. Combine: gaze relative to camera
    combined_rot = camera_rot * gaze_rot
    
    # 4. Convert back to Euler angles for Unity
    qx, qy, qz, qw = combined_rot.as_quat()
    r = R.from_quat([qx, qy, qz, qw])
    euler = combined_rot.as_euler('zyx', degrees=True)
    pitch, yaw, roll = euler[2], euler[1], euler[0]
    
    # # Ensure values are in -180..180 range
    # combined_euler = (euler + 180) % 360 - 180
    
    return (pitch, yaw, roll)


def write_ego_inputs(data_path, savepath):

    data_path = find('*preproc.h5', data_path, MR=True)
    data = read_h5(data_path)

    if 'twopT' not in data.keys():
        twop_dt = 1/7.5
        data['twopT'] = np.arange(0, len(data['x'])*twop_dt, twop_dt)
        if len(data['twopT']) > len(data['x']):
            data['twopT'] = data['twopT'][:-1]

    pitch = - data['pitch']
    roll = data['roll']
    imuT = data['imuT'] - data['imuT'][0]
    yaw = interpT(data['head_yaw_deg'], data['twopT'], imuT)
    x = interpT(data['x'], data['twopT'], imuT) / 34.21935483870968 # to convert from pixels to cm
    y = interpT(data['y'], data['twopT'], imuT) / 34.21935483870968

    df = pd.DataFrame({
        'theta': np.zeros(len(x))*np.nan,
        'phi': np.zeros(len(x))*np.nan,
        'pitch': pitch,
        'roll': roll,
        'yaw': yaw,
        'x': x,
        'y': y,
        'eyeT': imuT
    })

    df_interp = df.copy()
    for col in df.columns:
        if np.sum(np.isnan(df[col])) != len(df):
            df_interp[col] = nan_interp(df[col].to_numpy())

    df_interp['gaze_az'] = np.zeros(len(x))*np.nan
    df_interp['gaze_el'] = np.zeros(len(x))*np.nan

    df_interp.to_csv(savepath, index=False)

    return df_interp


def write_gaze_inputs(data_path, savepath):

    data_path = find('*preproc.h5', data_path, MR=True)
    data = read_h5(data_path)

    if 'twopT' not in data.keys():
        twop_dt = 1/7.5
        data['twopT'] = np.arange(0, len(data['x'])*twop_dt, twop_dt)
        if len(data['twopT']) > len(data['x']):
            data['twopT'] = data['twopT'][:-1]

    theta = np.rad2deg(data['theta'][data['eyeT_startInd']:data['eyeT_endInd']])
    phi = np.rad2deg(data['phi'][data['eyeT_startInd']:data['eyeT_endInd']])
    pitch = - data['pitch']
    roll = data['roll']
    eyeT = data['eyeT'][data['eyeT_startInd']:data['eyeT_endInd']]
    eyeT = eyeT - eyeT[0]
    imuT = data['imuT'] - data['imuT'][0]
    twopT = np.hstack([data['twopT'], data['twopT'][-1] + np.median(np.diff(data['twopT']))])
    yaw = interpT(data['head_yaw_deg'], data['twopT'], imuT)
    x = interpT(data['x'], data['twopT'], imuT) / 34.21935483870968 # to convert from pixels to cm
    y = interpT(data['y'], data['twopT'], imuT) / 34.21935483870968

    df = pd.DataFrame({
        'theta': theta,
        'phi': phi,
        'pitch': pitch,
        'roll': roll,
        'yaw': yaw,
        'x': x,
        'y': y,
        'eyeT': imuT
    })

    df_interp = df.copy()
    for col in df.columns:
        if np.sum(np.isnan(df[col])) != len(df):
            df_interp[col] = nan_interp(df[col].to_numpy())

    df_interp['gaze_az'] = pd.Series(np.ones(len(df_interp))*np.nan)
    df_interp['gaze_el'] = pd.Series(np.ones(len(df_interp))*np.nan)
    for ind in df_interp.index.values:
        az, el = gaze_angles(
            df_interp.loc[ind, 'theta'],
            df_interp.loc[ind, 'phi'],
            df_interp.loc[ind, 'yaw'],
            df_interp.loc[ind, 'pitch'],
            df_interp.loc[ind, 'roll']
        )
        df_interp.loc[ind, 'gaze_az'] = az
        df_interp.loc[ind, 'gaze_el'] = el

    df_interp.to_csv(savepath, index=False)

    return df_interp


def handheld_worldcam_preprocessing(datadir):
    # for recordings done without 2P data or eyecam; just hand-held camera

    possible_topdown_videos = find('*.mp4', datadir, MR=False, retempty=True)
    topdown_video = filter_file_search(possible_topdown_videos, toss=['labeled','resnet50'], MR=True)

    eyecam_raw_video = find('*_eyecam.avi', datadir, MR=True)
    
    full_rname = '_'.join(os.path.split(eyecam_raw_video)[1].split('_')[:-1])
    _savepath = os.path.join(datadir, '{}_preproc.h5'.format(full_rname))

    # if os.path.isfile(_savepath):
    #     # exit preprocessing if it has already been run
    #     return _savepath
    
    # eyecam_TTL_voltage = find('*_logTTL.csv', datadir, MR=True)
    # eyecam_TTL_timestamps = find('*_ttlTS.csv', datadir, MR=True)
    eyecam_video_timestamps = find('*_eyecam.csv', datadir, MR=True)
    imu_vals = find('*_IMUvals.csv', datadir, MR=True)
    imu_timestamps = find('*_IMUtime.csv', datadir, MR=True)

    eyecam_deinter_video = deinterlace(eyecam_raw_video, do_rotation=True)

    run_pose_estimation(
        topdown_video,
        project_cfg=r'T:/dlc_projects/freely_moving_topdown_06B/config.yaml',
        filter=False
    )
    topdown_pts_path = find('*DLC_resnet50_*handheld_worldcamOct17*.h5', datadir, MR=True)

    top_cam = Topcam(datadir, full_rname, cfg=None)
    top_cam.add_files(
        top_dlc_h5=topdown_pts_path,
        top_avi=topdown_video
    )
    arena_dict = top_cam.track_arena()
    arena_dict = fix_dict_dtype(arena_dict, float)
    pxls2cm = arena_dict['pxls2cm']
    top_xyl, top_tracking_dict = top_cam.track_body(pxls2cm)

    learx = top_tracking_dict['lear_x']
    leary = top_tracking_dict['lear_y']
    rearx = top_tracking_dict['rear_x']
    reary = top_tracking_dict['rear_y']

    headx = np.array([np.mean([rearx[f], learx[f]]) for f in range(len(rearx))])
    heady = np.array([np.mean([reary[f], leary[f]]) for f in range(len(reary))])

    eyeT = read_timestamp_file(eyecam_video_timestamps, position_data_length=None)

    imu_df, imuT = read_IMU(imu_vals, imu_timestamps)

    preprocessed_dict = {
        **top_xyl.to_dict(),
        **top_tracking_dict,
        **imu_df.to_dict()
    }
    preprocessed_dict['headx'] = headx
    preprocessed_dict['heady'] = heady
    preprocessed_dict['eyeT'] = eyeT
    preprocessed_dict['imuT'] = imuT

    for col in imu_df.columns.values:
        preprocessed_dict[col] = imu_df[col].to_numpy()

    _savepath = os.path.join(datadir, '{}_preproc.h5'.format(full_rname))
    print('Writing preprocessed data to {}'.format(_savepath))
    write_h5(_savepath, preprocessed_dict)

    return _savepath


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-method', '--method', type=str, default='ego_inputs')
    args = parser.parse_args()

    if args.method == 'ego_inputs':

        datapath = select_file(
            'Choose preprocessing file.',
            filetypes=[('HDF','.h5'),]
        )
        savedir, dataname = os.path.split(datapath)
        savepath = os.path.join(savedir, '{}_unity_inputs.csv'.format(dataname[:-10]))

        write_ego_inputs(datapath, savepath)


    elif args.method == 'gaze_inputs':

        datapath = select_file(
            'Choose preprocessing file.',
            filetypes=[('HDF','.h5'),]
        )
        savedir, dataname = os.path.split(datapath)
        savepath = os.path.join(savedir, '{}_unity_inputs.csv'.format(dataname[:-10]))

        write_gaze_inputs(datapath, savepath)


    elif args.method == 'worldcam_preprocess':

        datadir = select_directory(
            'Select data directory for handheld worldcam video.'
        )

        handheld_worldcam_preprocessing(datadir)

