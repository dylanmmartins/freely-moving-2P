# -*- coding: utf-8 -*-
"""
Functions for processing videos and performing camera calibration.

Functions
---------
deinterlace(video, exp_fps=30, quiet=False, allow_overwrite=False, do_rotation=False)
    Deinterlace and rotate videos and shift timestamps to match new video frames.
flip_headcams(video, h, v, quiet=True, allow_overwrite=None)
    Flip headcam videos horizontally and/or vertically without deinterlacing.
run_pose_estimation(video, project_cfg, filter=False)
    Run DLC pose estimation on videos.
pack_video_frames(video_path, ds=1.)
    Read in video and pack the frames into a numpy array.
load_video_frame(video_path, fr, ds=1., fps=7.5)
    Read in a single video frame and downsample it.
compute_camera_distortion(video_path, savepath, boardw=9, boardh=6)
    Compute the camera calibration matrix from a video of a moving checkerboard.
undistort_video(video_path, npz_path)
    Correct distortion by applying calibration matrix to a novel video.

Author: DMM, last modified 2024
"""


import os
import cv2
import subprocess
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from .helper import blockPrint, enablePrint

try:
    blockPrint()
    os.environ["DLClight"] = "True"
    import deeplabcut
    enablePrint()
    _dlc_available = True
except ModuleNotFoundError:
    enablePrint()
    _dlc_available = False


def subtract_band(frame_gray):

    left    = frame_gray[:, :10].mean(axis=1).astype(float)
    right   = frame_gray[:, -10:].mean(axis=1).astype(float)
    profile = gaussian_filter1d(0.5 * (left + right), sigma=15)
    corrected = frame_gray.astype(float) - profile[:, np.newaxis]
    corrected -= corrected.min()
    mx = corrected.max()
    if mx > 0:
        corrected = corrected / mx * 255.0
    return np.clip(corrected, 0, 255).astype(np.uint8)


def deinterlace(video, exp_fps=30, quiet=False,
                allow_overwrite=False, do_rotation=False):

    current_path = os.path.split(video)[0]
    vid_name     = os.path.split(video)[1]
    base_name    = vid_name.split('.avi')[0]

    print('Deinterlacing {}'.format(vid_name))

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps != exp_fps:
        return

    temp_path  = os.path.join(current_path, base_name + '_deintertemp.avi')
    final_path = os.path.join(current_path, base_name + '_deinter.avi')

    if do_rotation:
        vf_val = 'yadif=1:-1:0, vflip, hflip, scale=640:480'
    else:
        vf_val = 'yadif=1:-1:0, scale=640:480'

    cmd = ['ffmpeg', '-i', video, '-vf', vf_val,
           '-c:v', 'libopenh264', '-b:v', '2M', '-an']
    if allow_overwrite:
        cmd.extend(['-y'])
    else:
        cmd.extend(['-n'])
    cmd.extend([temp_path])
    if quiet:
        cmd.extend(['-loglevel', 'quiet'])

    subprocess.call(cmd)

    # Read deinterlaced temp, apply band subtraction, write final output
    cap      = cv2.VideoCapture(temp_path)
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_out  = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(final_path, fourcc, fps_out, (w, h), isColor=False)

    print('Applying band subtraction ({} frames)...'.format(n_frames))
    for _ in tqdm(range(n_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        writer.write(subtract_band(gray))

    cap.release()
    writer.release()
    os.remove(temp_path)

    return final_path


def flip_headcams(video, h, v, quiet=True, allow_overwrite=None):

    if h is True and v is True:
        vf_val = 'vflip, hflip'
    elif h is True and v is False:
        vf_val = 'hflip'
    elif h is False and v is True:
        vf_val = 'vflip'

    vid_name = os.path.split(video)[1]
    key_pieces = vid_name.split('.')[:-1]
    key = '.'.join(key_pieces)
    savepath = os.path.join(os.path.split(video)[0], (key + 'deinter.avi'))
    cmd = ['ffmpeg', '-i', video, '-vf', vf_val,
        '-c:v', 'libopenh264', '-b:v', '2M', '-an']
    if allow_overwrite:
        cmd.extend(['-y'])
    else:
        cmd.extend(['-n'])
    cmd.extend([savepath])

    if quiet is True:
        cmd.extend(['-loglevel', 'quiet'])
    if h is True or v is True:
        subprocess.call(cmd)


def run_pose_estimation(video, project_cfg, filter=False):
    if not _dlc_available:
        raise ImportError("deeplabcut is not installed. Install it to use run_pose_estimation.")
    deeplabcut.analyze_videos(project_cfg, [video])
    if filter:
        deeplabcut.filterpredictions(project_cfg, video)


def pack_video_frames(video_path, ds=1.):

    print('Reading {}'.format(os.path.split(video_path)[1]))
    vidread = cv2.VideoCapture(video_path)
    
    all_frames = np.empty([int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)),
                        int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT)*ds),
                        int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH)*ds)], dtype=np.uint8)
    
    for frame_num in tqdm(range(0,int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = vidread.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sframe = cv2.resize(frame, (0,0),
                            fx=ds, fy=ds,
                            interpolation=cv2.INTER_NEAREST)
        all_frames[frame_num,:,:] = sframe.astype(np.int8)
    
    return all_frames


def load_video_frame(video_path, fr, ds=1., fps=7.5):

    vidread = cv2.VideoCapture(video_path)

    nF = int(vidread.get(cv2.CAP_PROP_FRAME_COUNT))

    if np.isnan(fr):
        fr = int(nF / fps) // 2

    print('Reading frame {} from {}'.format(fr, os.path.split(video_path)[1]))

    frame_out = np.empty(
        [int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT)*ds), int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH)*ds)],
        dtype=np.uint8)
    
    vidread.set(cv2.CAP_PROP_POS_FRAMES, int(fr))

    ret, frame = vidread.read()

    if not ret:
        return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sframe = cv2.resize(
        frame,
        (0,0),
        fx=ds, fy=ds,
        interpolation=cv2.INTER_NEAREST
    )

    frame_out[:,:] = sframe.astype(np.int8)

    return frame_out


def compute_camera_distortion(video_path, savepath, boardw=9, boardh=6):

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    objp = np.zeros((boardh*boardw,3), np.float32)
    objp[:,:2] = np.mgrid[0:boardw,0:boardh].T.reshape(-1,2)
    
    calib_vid = cv2.VideoCapture(video_path)
    
    print('Finding chessboard corners for each frame')
    nF = int(calib_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    for step in tqdm(range(0, nF)):

        ret, img = calib_vid.read()
        if not ret:
            break
        if img.shape[2] > 1:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        ret, corners = cv2.findChessboardCorners(gray, (boardw,boardh), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    print('Calculating calibration correction') # this is slow!
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                    gray.shape[::-1], None, None)
    np.savez(savepath, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)


def undistort_video(video_path, npz_path):

    current_path = os.path.split(video_path)[0]
    vid_name = os.path.split(video_path)[1]
    base_name = vid_name.split('.avi')[0]
    savepath = os.path.join(current_path, (base_name + '_undistorted.avi'))

    print('Removing worldcam lens distortion for {}'.format(vid_name))

    checker_in = np.load(npz_path)

    mtx = checker_in['mtx']
    dist = checker_in['dist']
        
    cap = cv2.VideoCapture(video_path)
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(savepath, fourcc, real_fps,
                    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    for step in tqdm(range(0,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        
        ret, frame = cap.read()
        if not ret:
            break
        
        undist_frame = cv2.undistort(frame, mtx, dist, None, mtx)
        
        out_vid.write(undist_frame)

    out_vid.release()

    return savepath

