import os
import cv2
import subprocess
import pandas as pd
os.environ["DLClight"] = "True"
import deeplabcut
import numpy as np
from tqdm import tqdm

import fmr2e

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
        timestamps : list
            List of timestamp csv files for each video in videos (default is None).
            If the list is None, the subdirectories will be searched for timestamps.
        exp_fps : int
            Expected framerate of the videos (default is 30 Hz). If a video does not
            match this framerate, it will be skipped (e.g. if it has a frame rate of
            60 fps, it is assumed to have already been deinterlaced).
        quiet : bool
            When True, the function will not print status updates (default is False).
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


def flip_headcams(video, h, v, quiet=True, allow_overwrite=None):
    """ Flip headcam videos horizontally and/or vertically.

    This function will flip headcam videos horizontally and/or vertically
    based on the options in the config file. This is only needed for videos
    that need to have their orientation changed but do not need to be
    deinterlaced.

    Parameters
    ----------
    quiet : bool
        When True, the function will not print status updates (default
        is True).
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

    """
    deeplabcut.analyze_videos(project_cfg, [video])
    
    if filter:
        deeplabcut.filterpredictions(project_cfg, video)


def pose_estimation(self):
    """ Run DLC pose estimation on the eye camera videos.
    """
    
    if self.camname in self.cfg['dlc_projects'].keys():
        cam_project = self.cfg['dlc_projects'][self.camname]
    else:
        cam_project = None

    if cam_project != '' and cam_project != 'None' and cam_project != None:
        
        # if it's one of the cameras that needs to needs to be deinterlaced first, make sure and read in the deinterlaced 
        if self.camname=='REYE' or self.camname=='LEYE':
            
            # find all the videos in the data directory that are from the current camera and are deinterlaced
            if self.cfg['strict_name'] is True:
                vids_this_cam = fmr2e.find('*'+self.camname+'*deinter.avi',  self.recording_path)
            
            elif self.cfg['strict_name'] is False:
                vids_this_cam = fmr2e.find('*'+self.camname+'*.avi', self.recording_path)
            
            # remove unflipped videos generated during jumping analysis
            bad_vids = fmr2e.find('*'+self.camname+'*unflipped*.avi', self.recording_path)
            
            for x in bad_vids:
                if x in vids_this_cam:
                    vids_this_cam.remove(x)
            ir_vids = fmr2e.find('*IR*.avi', self.recording_path)
            
            for x in ir_vids:
                if x in vids_this_cam:
                    vids_this_cam.remove(x)
            
            # warning for user if no videos found
            if len(vids_this_cam) == 0:
                print('no ' + self.camname + ' videos found -- maybe the videos are not deinterlaced yet?')
        
        else:
            
            # find all the videos for camera types that don't neeed to be deinterlaced
            if self.cfg['strict_name'] is True:
                vids_this_cam = fmr2e.find('*'+self.camname+'*.avi', self.recording_path)
            
            elif self.cfg['strict_name'] is False:
                vids_this_cam = fmr2e.find('*'+self.camname+'*.avi', self.recording_path)
        
        # analyze the videos with DeepLabCut
        # this gives the function a list of files that it will iterate over with the same DLC config file
        vids2run = [vid for vid in vids_this_cam if 'plot' not in vid]
        self.batch_dlc_analysis(vids2run, cam_project)


def pack_video_frames(video_path, dwnsmpl=1):
    
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


# def pack_position_data():