"""
Top-down camera tracking and analysis.

Classes
-------
Topcam
    Top-down camera tracking and analysis.

Author: DMM, 2024
"""


import os
import gc
import yaml
import json
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import fm2p


class Topcam():
    """ Top-down camera tracking and analysis.
    """

    def __init__(self, recording_path, recording_name, cfg=None):
        """
        Parameters
        ----------
        recording_path : str
            Path to the recording folder.
        recording_name : str
            Name of the recording (without file extension).
        cfg : str or dict, optional
            Path to the configuration file or a dictionary of configuration parameters.
            If None, the default configuration file will be used.
        """

        self.recording_path = recording_path
        self.recording_name = recording_name

        if cfg is None:
            internals_config_path = os.path.join(fm2p.up_dir(__file__, 1), 'internals.yaml')
            with open(internals_config_path, 'r') as infile:
                cfg = yaml.load(infile, Loader=yaml.FullLoader)
        elif type(cfg)==str:
            with open(cfg, 'r') as infile:
                cfg = yaml.load(infile, Loader=yaml.FullLoader)

        self.cfg = cfg

        # Overwrite threshold until a better topdown DLC network is trained
        # self.cfg['likelihood_thresh'] = 0.9


    def find_files(self):
        """ Find the top-down camera files in the recording folder.
        """
        
        self.top_dlc_h5 = fm2p.find('{}*topDLC_resnet50*.h5'.format(self.recording_name), self.recording_path, MR=True)
        self.top_avi = fm2p.find('{}*top.mp4'.format(self.recording_name), self.recording_path, MR=True)


    def add_files(self, top_dlc_h5, top_avi):
        """ Add the top-down camera files to the object manually from file paths.
        
        Parameters
        ----------
        top_dlc_h5 : str
            Path to the top-down camera DLC h5 file.
        top_avi : str
            Path to the top-down camera.
        """

        self.top_dlc_h5 = top_dlc_h5
        self.top_avi = top_avi


    def track_body(self, pxls2cm=None):
        """ Track the body position and orientation from the top-down camera data.

        Parameters
        ----------
        pxls2cm : float
            Conversion factor from pixels to centimeters.
        
        Returns
        -------
        xyl : pd.DataFrame
            X, y, and likelihood from DLC tracking.
        topcam_dict : dict
            Dictionary of top-down camera tracking results, including speed, head yaw, and movement yaw.
        """

        if pxls2cm is None:
            pxls2cm = self.cfg['pxls2cm']

        likelihood_thresh = self.cfg['likelihood_thresh']

        # Read DLC data and filter by likelihood
        xyl, _ = fm2p.open_dlc_h5(self.top_dlc_h5)
        x_vals, y_vals, likelihood = fm2p.split_xyl(xyl)

        # Threshold by likelihoods
        x_vals = fm2p.apply_liklihood_thresh(x_vals, likelihood, threshold=likelihood_thresh)
        y_vals = fm2p.apply_liklihood_thresh(y_vals, likelihood, threshold=likelihood_thresh)

        # Topdown speed using neck point
        smooth_x = fm2p.convfilt(fm2p.nanmedfilt(x_vals['nose_x'], 7)[0], box_pts=20)
        smooth_y = fm2p.convfilt(fm2p.nanmedfilt(y_vals['nose_y'], 7)[0], box_pts=20)
        top_speed = np.sqrt(np.diff((smooth_x*60) / pxls2cm)**2 + np.diff((smooth_y*60) / pxls2cm)**2)

        # Get head angle from ear points
        rear_x = fm2p.nanmedfilt(x_vals['rightbar_x'], 7)[0]
        rear_y = fm2p.nanmedfilt(y_vals['rightbar_y'], 7)[0]
        lear_x = fm2p.nanmedfilt(x_vals['leftbar_x'], 7)[0]
        lear_y = fm2p.nanmedfilt(y_vals['leftbar_y'], 7)[0]

        # Rotate 90deg because ears are perpendicular to head yaw
        head_yaw = np.arctan2((lear_y - rear_y), (lear_x - rear_x)) + np.deg2rad(90)
        head_yaw_deg = np.rad2deg(head_yaw % (2*np.pi))

        # Angle of body movement ("movement yaw")
        x_disp = np.diff((smooth_x*60) / pxls2cm)
        y_disp = np.diff((smooth_y*60) / pxls2cm)

        # Get the angle of the vector of motion
        movement_yaw = np.arctan2(y_disp, x_disp)
        movement_yaw_deg = np.rad2deg(movement_yaw % (2*np.pi))

        topcam_dict = {
            'speed': top_speed,
            'lear_x': lear_x,
            'lear_y': lear_y,
            'rear_x': rear_x,
            'rear_y': rear_y,
            'head_yaw': head_yaw,
            'movement_yaw': movement_yaw,
            'x': smooth_x,
            'y': smooth_y,
            'head_yaw_deg': head_yaw_deg,
            'movement_yaw_deg': movement_yaw_deg,
            'x_displacement': x_disp,
            'y_displacement': y_disp,
        }

        return xyl, topcam_dict
    
    def _track_arena_no_pillar(self):

        frame = fm2p.load_video_frame(self.top_avi, fr=np.nan, ds=1.)

        print('Place points at each corner of the arena. Order MUST be [top-left, top-right, bottom-left, bottom-right].')

        user_arena_x, user_arena_y = fm2p.place_points_on_image(
            frame,
            num_pts=4,
            color='tab:blue',
            tight_scale=True
        )

        # Conversion from pixels to cm
        arena_width_cm = self.cfg['arena_width_cm']
        # right - left
        pxls2cm_1 = (user_arena_x[1] - user_arena_x[0]) / arena_width_cm
        pxls2cm_2 = (user_arena_x[3] - user_arena_x[2]) / arena_width_cm
        pxls2cm = np.nanmean([pxls2cm_1, pxls2cm_2])

        arena_dict = {
            'arenaTL': {
                'x': user_arena_x[0],
                'y': user_arena_y[0]
            },
            'arenaTR': {
                'x': user_arena_x[1],
                'y': user_arena_y[1]
            },
            'arenaBR': {
                'x': user_arena_x[3],
                'y': user_arena_y[3]
            },
            'arenaBL': {
                'x': user_arena_x[2],
                'y': user_arena_y[2]
            },
            'pxls2cm': pxls2cm
        }

        return arena_dict
    

    def track_arena(self, no_pillar=False):
        """ Track the arena and pillar from the top-down camera data.

        Returns
        -------
        arena_dict : dict
            Dictionary of arena tracking results, including arena corners, pillar points, and conversion factor.
        """

        if no_pillar:
            return self._track_arena_no_pillar()

        frame = fm2p.load_video_frame(self.top_avi, fr=np.nan, ds=1.)

        print('Place points at each corner of the arena. Order MUST be [top-left, top-right, bottom-left, bottom-right].')

        user_arena_x, user_arena_y = fm2p.place_points_on_image(
            frame,
            num_pts=4,
            color='tab:blue',
            tight_scale=False
        )

        # Conversion from pixels to cm
        arena_width_cm = self.cfg['arena_width_cm']
        # right - left
        pxls2cm_1 = (user_arena_x[1] - user_arena_x[0]) / arena_width_cm
        pxls2cm_2 = (user_arena_x[3] - user_arena_x[2]) / arena_width_cm
        pxls2cm = np.nanmean([pxls2cm_1, pxls2cm_2])

        print('Place points around the perimeter of the pillar (align to the top of the pillar, even if that is different from the base).')

        user_pillar_x, user_pillar_y = fm2p.place_points_on_image(
            frame,
            num_pts=8,
            color='tab:red',
            tight_scale=False
        )

        # Convert from two lists of points to a single list of (x,y) pairs.
        user_pts = []
        for i in range(len(user_pillar_x)):
            user_pts.append((user_pillar_x[i], user_pillar_y[i]))

        print('Drag the polygon to a better position over the pillar. Close the figure when done.')

        shifted_user_pts = fm2p.user_polygon_translation(pts=user_pts, image=frame)

        # Convert from single list of (x,y) pairs to two lists of points.
        pillar_x = []
        pillar_y = []
        for i in range(len(shifted_user_pts)):
            pillar_x.append(shifted_user_pts[i][0])
            pillar_y.append(shifted_user_pts[i][1])

        pillar_dict = fm2p.Eyecam.fit_ellipse('', x=pillar_x, y=pillar_y)
        pillar_centroid = [pillar_dict['Y0'], pillar_dict['X0']]
        pillar_axes = (pillar_dict['long_axis'], pillar_dict['short_axis'])
        pillar_radius = np.mean(pillar_axes)

        # xyl, _ = fm2p.open_dlc_h5(self.top_dlc_h5)
        # x_vals, y_vals, likelihood = fm2p.split_xyl(xyl)
        # x_vals = fm2p.apply_liklihood_thresh(x_vals, likelihood)
        # y_vals = fm2p.apply_liklihood_thresh(y_vals, likelihood)

        arena_dict = {
            'arenaTL': {
                'x': user_arena_x[0],
                'y': user_arena_y[0]
            },
            'arenaTR': {
                'x': user_arena_x[1],
                'y': user_arena_y[1]
            },
            'arenaBR': {
                'x': user_arena_x[3],
                'y': user_arena_y[3]
            },
            'arenaBL': {
                'x': user_arena_x[2],
                'y': user_arena_y[2]
            },
            'pillar_x': pillar_x,
            'pillar_y': pillar_y,
            'pillar_radius': pillar_radius,
            'pillar_centroid': {
                'x': pillar_centroid[0],
                'y': pillar_centroid[1]
            },
            'pxls2cm': pxls2cm
        }

        return arena_dict
    
    
    def write_diagnostic_video(self, savepath, vidarr, xyl, body_tracking_results, startF=1000, lenF=3600):
        """
        Parameters
        ----------
        savepath : str
            Filepath to save video. Must end in .avi
        vidarr : np.array
            Array of topdown video, with shape (time, height, width).
        xyl : pd.DataFrame
            X, y, and likelihood from DLC tracking.
        body_tracking_results : dict
            Tracked body positions, orientations, and running state from track_body().
        startF : int
            Frame to start the diagnostic video from.
        lenF : int
            How many frames to include in the diagnostic video. Default is 3600 (1 min @ 60 Hz).
        """

        x_vals, y_vals, likelihood = fm2p.split_xyl(xyl)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_vid = cv2.VideoWriter(savepath, fourcc, 60.0, (640, 480))
        maxprev = 25

        lear_x = x_vals['left_ear_x']
        rear_x = x_vals['right_ear_x']
        lear_y = y_vals['left_ear_y']
        rear_y = y_vals['right_ear_y']
        neck_x = x_vals['top_skull_x']
        neck_y = y_vals['top_skull_y']
        back_x = x_vals['base_tail_x']
        back_y = y_vals['base_tail_y']
        head_yaw = np.deg2rad(body_tracking_results['head_yaw_deg'])
        body_yaw = np.deg2rad(body_tracking_results['body_yaw_deg'])
        x_disp = body_tracking_results['x_displacement']
        y_disp = body_tracking_results['y_displacement']

        for f in tqdm(range(startF,startF+lenF)):

            fig = plt.figure()

            plt.imshow(vidarr[f,:,:].astype(np.uint8), cmap='gray')
            plt.axis('off')

            plt.plot(lear_x[f], lear_y[f], 'b*')
            plt.plot(rear_x[f], rear_y[f], 'b*')

            plt.plot([neck_x[f], (neck_x[f])+15*np.cos(head_yaw[f])],
                        [neck_y[f],(neck_y[f])+15*np.sin(head_yaw[f])],
                        '-', linewidth=2, color='cyan') # head yaw
            
            plt.plot([back_x[f], (back_x[f])-15*np.cos(body_yaw[f])],
                        [back_y[f], (back_y[f])-15*np.sin(body_yaw[f])],
                        '-', linewidth=2, color='pink') # body yaw
            
            for p in range(maxprev):

                prevf = f - p

                plt.plot(neck_x[prevf],
                            neck_y[prevf], 'o', color='tab:purple',
                            alpha=(maxprev-p)/maxprev) # neck position history
                
            # arrow for vector of motion
            if body_tracking_results['forward_run'][f]:
                movvec_color = 'tab:green'
            elif body_tracking_results['backward_run'][f]:
                movvec_color = 'tab:orange'
            elif body_tracking_results['fine_motion'][f]:
                movvec_color = 'tab:olive'
            elif body_tracking_results['stationary'][f]:
                movvec_color = 'tab:red'
            
            plt.arrow(neck_x[f], neck_y[f],
                        x_disp[f]*3, y_disp[f]*3,
                        color=movvec_color, width=1)
            
            # Save the frame out
            fig.canvas.draw()
            frame_as_array = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            frame_as_array = frame_as_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()

            img = cv2.cvtColor(frame_as_array, cv2.COLOR_RGB2BGR)
            out_vid.write(img.astype('uint8'))

        out_vid.release()


    def save_tracking(self, topcam_dict, dlc_xyl, vid_array, arena_dict=None):
        """ Save the tracking results to an h5 file.
        
        Parameters
        ----------
        topcam_dict : dict
            Dictionary of top-down camera tracking results, including speed, head yaw, and movement yaw.
        dlc_xyl : pd.DataFrame
            X, y, and likelihood from DLC tracking.
        vid_array : np.array
            Array of top-down video, with shape (time, height, width).
        arena_dict : dict, optional
            Dictionary of arena tracking results, including arena corners, pillar points, and conversion factor.

        Returns
        -------
        _savepath : str
            Path to the saved h5 file.
        """

        xyl_dict = dlc_xyl.to_dict()
        vid_dict = {'video': vid_array}

        if arena_dict is None:
            save_dict = {**xyl_dict, **topcam_dict, **vid_dict}
        elif arena_dict is not None:
            save_dict = {**xyl_dict, **topcam_dict, **vid_dict, **arena_dict}

        savedir = os.path.join(self.recording_path, self.recording_name)
        _savepath = os.path.join(savedir, '{}_top_tracking.h5'.format(self.recording_name))
        fm2p.write_h5(_savepath, save_dict)

        return _savepath
        

if __name__ == '__main__':
    
    basepath = r'K:\FreelyMovingEyecams\241204_DMM_DMM031_freelymoving'
    rec_name = '241204_DMM_DMM031_freelymoving_01'
    top = Topcam(basepath, rec_name)
    top.find_files()
    top.get_head_body_yaw()
