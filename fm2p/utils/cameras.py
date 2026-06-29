# -*- coding: utf-8 -*-
"""
fm2p/utils/cameras.py

Video processing and camera calibration utilities.

Functions
---------
subtract_band
    Remove vertical illumination bands from a grayscale eye-camera frame.
deinterlace
    Deinterlace a video with ffmpeg and apply band subtraction.
flip_headcams
    Flip a head-camera video horizontally and/or vertically.
run_pose_estimation
    Run DLC pose estimation on a video file.
pack_video_frames
    Read all frames of a video into a numpy array.
load_video_frame
    Load a single frame from a video (optionally downsampled).
compute_camera_distortion
    Fit a distortion matrix from a checkerboard calibration video.
undistort_video
    Apply a stored distortion matrix to correct a novel video.


DMM, December 2024
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
    """ Remove vertical illumination bands from a grayscale frame.

    Estimates the band profile from the left and right 10-pixel margins,
    smooths with a Gaussian, and subtracts it column-wise.

    Parameters
    ----------
    frame_gray : np.ndarray, shape (H, W), dtype uint8

    Returns
    -------
    np.ndarray, shape (H, W), dtype uint8
    """

    left = frame_gray[:, :10].mean(axis=1).astype(float)
    right = frame_gray[:, -10:].mean(axis=1).astype(float)
    profile = gaussian_filter1d(0.5 * (left + right), sigma=15)
    corrected = frame_gray.astype(float) - profile[:, np.newaxis]
    corrected -= corrected.min()
    mx = corrected.max()
    if mx > 0:
        corrected = corrected / mx * 255.0
    return np.clip(corrected, 0, 255).astype(np.uint8)


def deinterlace(video, exp_fps=30, quiet=False, allow_overwrite=False, do_rotation=False):
    """ Deinterlace a video with ffmpeg (yadif) and apply column-band subtraction.

    Parameters
    ----------
    video : str
        Path to the input AVI.
    exp_fps : int
        Only deinterlace if the video matches this frame rate (guards against
        processing already-deinterlaced files).
    quiet : bool
        Suppress ffmpeg output.
    allow_overwrite : bool
        Pass -y to ffmpeg to allow overwriting existing output.
    do_rotation : bool
        If True, apply vflip+hflip (180-degree rotation) during deinterlacing.

    Returns
    -------
    final_path : str or None
        Path to the written output file, or None if fps check failed.
    """

    current_path = os.path.split(video)[0]
    vid_name = os.path.split(video)[1]
    base_name = vid_name.split('.avi')[0]

    print('Deinterlacing {}'.format(vid_name))

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps != exp_fps:
        return

    temp_path = os.path.join(current_path, base_name + '_deintertemp.avi')
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

    # Apply band subtraction to the deinterlaced temp before writing the final output.
    cap = cv2.VideoCapture(temp_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_out = cap.get(cv2.CAP_PROP_FPS)
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
    """ Flip a head-camera video horizontally and/or vertically via ffmpeg.

    Parameters
    ----------
    video : str
        Input video path.
    h : bool
        Flip horizontally.
    v : bool
        Flip vertically.
    quiet : bool
        Suppress ffmpeg output.
    allow_overwrite : bool or None
        Pass -y to ffmpeg if True.
    """

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
    """ Run DLC pose estimation on a video file.

    Parameters
    ----------
    video : str
        Path to the video.
    project_cfg : str
        Path to the DLC project config.yaml.
    filter : bool
        If True, run filterpredictions after analysis.
    """

    if not _dlc_available:
        raise ImportError('deeplabcut is not installed. Install it to use run_pose_estimation.')
    deeplabcut.analyze_videos(project_cfg, [video])
    if filter:
        deeplabcut.filterpredictions(project_cfg, video)


def pack_video_frames(video_path, ds=1.):
    """ Read all frames of a video into a uint8 numpy array.

    Parameters
    ----------
    video_path : str
    ds : float
        Spatial downsample factor (1.0 = no downsampling).

    Returns
    -------
    all_frames : np.ndarray, shape (N_frames, H*ds, W*ds)
    """

    print('Reading {}'.format(os.path.split(video_path)[1]))
    vidread = cv2.VideoCapture(video_path)

    all_frames = np.empty([
        int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT) * ds),
        int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH) * ds)
    ], dtype=np.uint8)

    for frame_num in tqdm(range(0, int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = vidread.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sframe = cv2.resize(frame, (0, 0), fx=ds, fy=ds, interpolation=cv2.INTER_NEAREST)
        all_frames[frame_num, :, :] = sframe.astype(np.int8)

    return all_frames


def load_video_frame(video_path, fr, ds=1., fps=7.5):
    """ Load a single frame from a video, optionally downsampled.

    Parameters
    ----------
    video_path : str
    fr : int or float
        Frame number to read. If NaN, defaults to the middle frame.
    ds : float
        Spatial downsample factor.
    fps : float
        Used only when fr is NaN to estimate the middle frame.

    Returns
    -------
    frame_out : np.ndarray, shape (H*ds, W*ds), dtype uint8
    """

    vidread = cv2.VideoCapture(video_path)

    nF = int(vidread.get(cv2.CAP_PROP_FRAME_COUNT))

    if np.isnan(fr):
        fr = int(nF / fps) // 2

    print('Reading frame {} from {}'.format(fr, os.path.split(video_path)[1]))

    frame_out = np.empty(
        [int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT) * ds), int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH) * ds)],
        dtype=np.uint8
    )

    vidread.set(cv2.CAP_PROP_POS_FRAMES, int(fr))

    ret, frame = vidread.read()

    if not ret:
        return

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sframe = cv2.resize(
        frame,
        (0, 0),
        fx=ds, fy=ds,
        interpolation=cv2.INTER_NEAREST
    )

    frame_out[:, :] = sframe.astype(np.int8)

    return frame_out


def compute_camera_distortion(video_path, savepath, boardw=9, boardh=6):
    """ Fit a lens-distortion matrix from a checkerboard calibration video.

    Parameters
    ----------
    video_path : str
        Path to the calibration video.
    savepath : str
        Path to save the NPZ file containing mtx, dist, rvecs, tvecs.
    boardw : int
        Number of inner corners along the board's width.
    boardh : int
        Number of inner corners along the board's height.
    """

    objpoints = []  # 3D object points in world space
    imgpoints = []  # 2D image-plane points

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((boardh * boardw, 3), np.float32)
    objp[:, :2] = np.mgrid[0:boardw, 0:boardh].T.reshape(-1, 2)

    calib_vid = cv2.VideoCapture(video_path)

    print('Finding chessboard corners for each frame...')
    nF = int(calib_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    for step in tqdm(range(0, nF)):

        ret, img = calib_vid.read()
        if not ret:
            break
        if img.shape[2] > 1:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        ret, corners = cv2.findChessboardCorners(gray, (boardw, boardh), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    # Calibration is slow; may take several minutes for a long video.
    print('Calculating calibration correction...')
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    np.savez(savepath, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)


def undistort_video(video_path, npz_path):
    """ Apply a stored distortion matrix to correct a novel video.

    Parameters
    ----------
    video_path : str
        Path to the input video.
    npz_path : str
        Path to the NPZ calibration file (output of compute_camera_distortion).

    Returns
    -------
    savepath : str
        Path to the undistorted output video.
    """

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

    for step in tqdm(range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):

        ret, frame = cap.read()
        if not ret:
            break

        undist_frame = cv2.undistort(frame, mtx, dist, None, mtx)

        out_vid.write(undist_frame)

    out_vid.release()

    return savepath
