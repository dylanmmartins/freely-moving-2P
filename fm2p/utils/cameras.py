"""
fm2p/utils/cameras.py
Camera utilities

DMM, 2024
"""


import os
import cv2
import subprocess
os.environ["DLClight"] = "True"
import deeplabcut
import numpy as np
from tqdm import tqdm


def deinterlace(video, exp_fps=30, quiet=False,
                allow_overwrite=False, do_rotation=False):
    """ Deinterlace videos and shift timestamps to match new video frames.

    If videos and timestamps are provided (as lists), only the provided
    filepaths will be processed. If lists are not provided, subdirectories
    will be searched within animal_directory in the options dictionary, config.
    Both videos and timestamps must be provided, for either to be used.

    Videos will also be rotated 180 deg (so that they are flipped in the horizontal
    and vertical directions) if the option is set in the config file.

    Parameters
    ----------
    videos : list
        List of eyecam and/or worldcam videos at 30fps (default is None). If
        the list is None, the subdirectories will be searched for videos.
    exp_fps : int
        Expected framerate of the videos (default is 30 Hz). If a video does not
        match this framerate, it will be skipped (e.g. if it has a frame rate of
        60 fps, it is assumed to have already been deinterlaced).
    quiet : bool
        When True, the function will not print status updates (default is False).
    allow_overwrite : bool
        When True, it will allow video files to be overwritten.
    do_rotation : bool
        When True, the input video will be rotated 180 deg (one horizontal flip and
        one vertical flip). Otherwise, the orientation is preserved.

    Returns
    -------
    savepath : str
        Filepath at which the new video was written.
    """

    current_path = os.path.split(video)[0]

    # Make a save path that keeps the subdirectories. Get out a
    # key from the name of the video that will be shared with
    # all other data of this trial.
    
    vid_name = os.path.split(video)[1]
    base_name = vid_name.split('.avi')[0]

    print('Deinterlacing {}'.format(vid_name))
    
    # open the video
    cap = cv2.VideoCapture(video)
    
    # get info about the video
    fps = cap.get(cv2.CAP_PROP_FPS) # frame rate

    if fps != exp_fps:
        return

    savepath = os.path.join(current_path, (base_name + '_deinter.avi'))

    if do_rotation:
        vf_val = 'yadif=1:-1:0, vflip, hflip, scale=640:480'
    elif not do_rotation:
        vf_val = 'yadif=1:-1:0, scale=640:480'

    # could add a '-y' after 'ffmpeg' and before ''-i' so that it overwrites
    # an existing file by default
    cmd = ['ffmpeg', '-i', video, '-vf', vf_val, '-c:v', 'libx264',
        '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a',
        '256k']

    if allow_overwrite:
        cmd.extend(['-y'])
    else:
        cmd.extend(['-n'])

    cmd.extend([savepath])
    
    if quiet is True:
        cmd.extend(['-loglevel', 'quiet'])

    subprocess.call(cmd)

    return savepath


def flip_headcams(video, h, v, quiet=True, allow_overwrite=None):
    """ Flip headcam videos horizontally and/or vertically.

    This function will flip headcam videos horizontally and/or vertically
    based on the options in the config file. This is only needed for videos
    that need to have their orientation changed but do not need to be
    deinterlaced.

    Parameters
    ----------
    video : str
        File path to the video, which should be an .avi
    h : bool
        Whether to flip the video horizontally.
    v : bool
        Whether to flip the video vertically.
    quiet : bool
        When True, the function will not print status updates (default
        is True).
    allow_overwrite : bool
        When True, it will allow video files to be overwritten.
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

    cmd = ['ffmpeg', '-i', video, '-vf', vf_val, '-c:v',
        'libx264', '-preset', 'slow', '-crf', '19',
        '-c:a', 'aac', '-b:a', '256k']

    if allow_overwrite:
        cmd.extend(['-y'])
    else:
        cmd.extend(['-n'])

    cmd.extend([savepath])

    if quiet is True:
        cmd.extend(['-loglevel', 'quiet'])

    # Only do the rotation is at least one axis is being flipped
    if h is True or v is True:
        subprocess.call(cmd)


def run_pose_estimation(video, project_cfg, filter=False):
    """ Run DLC pose estimation on videos.

    Parameters
    ----------
    videos : str or list
        The path to the video file(s) to be analyzed.
    project_cfg : str
        The path to the project config file.
    filter : bool
        Whether to create additional files of the median
        filtered pose estimate. Default is False.
    """
    deeplabcut.analyze_videos(project_cfg, [video])
    
    if filter:
        deeplabcut.filterpredictions(project_cfg, video)


def pack_video_frames(video_path, dwnsmpl=1.):
    """ Read in video and pack the frames into a numpy array.

    Parameters
    ----------
    video_path : str
        File path to the video, which should be an .avi. This may work
        with other file types, but is untested.
    dwnsmpl : float
        Value by which to downsample the image, e.g., 0.5 scales the
        image to half of its origional x/y resolution). This does not
        downsample in the time dimension.

    Returns
    -------
    all_frames : np.array
        3D array of shape (time, height, width).
    """

    print('Reading {}'.format(os.path.split(video_path)[1]))
    
    # open the .avi file
    vidread = cv2.VideoCapture(video_path)
    
    # empty array that is the target shape
    # should be number of frames x downsampled height x downsampled width
    all_frames = np.empty([int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)),
                        int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT)*dwnsmpl),
                        int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH)*dwnsmpl)], dtype=np.uint8)
    
    # iterate through each frame
    for frame_num in tqdm(range(0,int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)))):
        
        # read the frame in and make sure it is read in correctly
        ret, frame = vidread.read()
        if not ret:
            break
        
        # convert to grayyscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # downsample the frame by an amount specified in the config file
        sframe = cv2.resize(frame, (0,0),
                            fx=dwnsmpl, fy=dwnsmpl,
                            interpolation=cv2.INTER_NEAREST)
        
        # add the downsampled frame to all_frames as int8
        all_frames[frame_num,:,:] = sframe.astype(np.int8)
    
    return all_frames


def compute_camera_distortion(video_path, savepath, boardw=9, boardh=6):
    """ Compute the camera calibration matrix from a video
    of a moving checkerboard.

    Parameters
    ----------
    video_path : str
        Path to the video. It should be several minutes long and capture
        diverse angles and distances of a printed checkerboard being moved
        in front of the camera. The checkerboard pattern should be fixed
        to a rigid surface so that it cannot bend.
    savepath : str
        Path to save the calibration matrix, ending in the extension .npz
    boardw : int
        Checkerboard width in number of squares. Default, 9, works with the
        standard opencv checkerboard file.
    boardh : int
        Same as boardw for height.
    """

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points
    objp = np.zeros((boardh*boardw,3), np.float32)
    objp[:,:2] = np.mgrid[0:boardw,0:boardh].T.reshape(-1,2)
    
    # Read in file path of video
    calib_vid = cv2.VideoCapture(video_path)
    
    # Iterate through frames
    print('Finding chessboard corners for each frame')
    
    nF = int(calib_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for step in tqdm(range(0, nF)):
        
        # Open frame
        ret, img = calib_vid.read()
        
        # Make sure the frame is read in correctly
        if not ret:
            break
        
        # Convert to grayscale
        if img.shape[2] > 1:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (boardw,boardh), None)
        
        # If found, add object points, image points (after refining them)
        if ret == True:

            objpoints.append(objp)

            # corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)


    # Calibrate the camera (this is a little slow)
    print('Calculating calibration correction')
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                    gray.shape[::-1], None, None)
    
    # Format as .npz and save the file
    np.savez(savepath, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)


def undistort_video(video_path, npz_path):
    """ Correct distortion by applying calibration matrix to a novel video.

    Parameters
    ----------
    video_path : str
        Path to the video
    npz_path : str
        Filepath of the worldcam calibration matrix written by the function
        compute_camera_distortion().

    Returns
    -------
    savepath : str
        The filepath of the new video written to disk.
    """

    current_path = os.path.split(video_path)[0]
    vid_name = os.path.split(video_path)[1]
    base_name = vid_name.split('.avi')[0]
    savepath = os.path.join(current_path, (base_name + '_undistorted.avi'))

    print('Removing worldcam lens distortion for {}'.format(vid_name))

    # load the parameters
    checker_in = np.load(npz_path)

    # unpack camera properties
    mtx = checker_in['mtx']
    dist = checker_in['dist']
    # rvecs = checker_in['rvecs']
    # tvecs = checker_in['tvecs']
        
    cap = cv2.VideoCapture(video_path)
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # setup the file writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(savepath, fourcc, real_fps,
                    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    # iterate through all frames
    for step in tqdm(range(0,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        
        # open frame and check that it opens correctly
        ret, frame = cap.read()
        if not ret:
            break
        
        # run opencv undistortion function
        undist_frame = cv2.undistort(frame, mtx, dist, None, mtx)
        
        # write the frame to the video
        out_vid.write(undist_frame)

    out_vid.release()

    return savepath

