"""
fm2p/utils/topcam.py
Topdown tracking class

DMM, 2024
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

    def __init__(self, recording_path, recording_name, cfg=None):

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
        self.cfg['likelihood_thresh'] = 0.5


    def find_files(self):
        
        self.top_dlc_h5 = fm2p.find('{}*topDLC_resnet50*.h5'.format(self.recording_name), self.recording_path, MR=True)
        self.top_avi = fm2p.find('{}*top.mp4'.format(self.recording_name), self.recording_path, MR=True)


    def add_files(self, top_dlc_h5, top_avi):

        self.top_dlc_h5 = top_dlc_h5
        self.top_avi = top_avi


    def track_body(self):

        likelihood_thresh = self.cfg['likelihood_thresh']
        arena_width_cm = self.cfg['arena_width_cm']

        # Read DLC data and filter by likelihood
        xyl, _ = fm2p.open_dlc_h5(self.top_dlc_h5)
        x_vals, y_vals, likelihood = fm2p.split_xyl(xyl)

        # Threshold by likelihoods
        x_vals = fm2p.apply_liklihood_thresh(x_vals, likelihood, threshold=likelihood_thresh)
        y_vals = fm2p.apply_liklihood_thresh(y_vals, likelihood, threshold=likelihood_thresh)

        # Conversion from pixels to cm
        left = 'arena_TL_x' # top left X
        right = 'arena_TR_x' # top right X
        
        dist_pxls = np.nanmedian(x_vals[right]) - np.nanmedian(x_vals[left])
        pxls2cm = dist_pxls / arena_width_cm

        # Topdown speed using neck point
        smooth_x = fm2p.convfilt(fm2p.nanmedfilt(x_vals['head_backleft_x'], 7)[0], box_pts=20)
        smooth_y = fm2p.convfilt(fm2p.nanmedfilt(y_vals['head_backleft_y'], 7)[0], box_pts=20)
        top_speed = np.sqrt(np.diff((smooth_x*60) / pxls2cm)**2 + np.diff((smooth_y*60) / pxls2cm)**2)

        # Get head angle from ear points
        lear_x = fm2p.nanmedfilt(x_vals['head_backleft_x'], 7)[0]
        lear_y = fm2p.nanmedfilt(y_vals['head_backleft_y'], 7)[0]
        rear_x = fm2p.nanmedfilt(x_vals['head_backright_x'], 7)[0]
        rear_y = fm2p.nanmedfilt(y_vals['head_backright_y'], 7)[0]

        # Rotate 90deg because ears are perpendicular to head yaw
        head_yaw = np.arctan2((lear_y - rear_y), (lear_x - rear_x)) + np.deg2rad(90)
        head_yaw_deg = np.rad2deg(head_yaw % (2*np.pi))

        # Angle of body movement ("movement yaw")
        x_disp = np.diff((smooth_x*60) / pxls2cm)
        y_disp = np.diff((smooth_y*60) / pxls2cm)

        movement_yaw = np.arctan2(y_disp, x_disp)
        movement_yaw_deg = np.rad2deg(movement_yaw % (2*np.pi))

        topcam_dict = {
            'speed': top_speed,
            'head_yaw': head_yaw,
            'movement_yaw': movement_yaw,
            'x': smooth_x,
            'y': smooth_y,
            'head_yaw_deg': head_yaw_deg,
            'movement_yaw_deg': movement_yaw_deg,
            'x_displacement': x_disp,
            'y_displacement': y_disp,
            'pxls2cm': pxls2cm
        }

        return xyl, topcam_dict
    

    def track_arena(self, vidpath_for_annotation=None):

        likelihood_thresh = self.cfg['likelihood_thresh']

        xyl, _ = fm2p.open_dlc_h5(self.top_dlc_h5)
        x_vals, y_vals, likelihood = fm2p.split_xyl(xyl)

        x_vals = fm2p.apply_liklihood_thresh(x_vals, likelihood, threshold=likelihood_thresh)
        y_vals = fm2p.apply_liklihood_thresh(y_vals, likelihood, threshold=likelihood_thresh)

        # assume single continuous pillar position over entire recording
        topP = (np.nanmedian(x_vals['pillar_T_x']), np.nanmedian(y_vals['pillar_T_y']))
        bottomP = (np.nanmedian(x_vals['pillar_B_x']), np.nanmedian(y_vals['pillar_B_y']))
        leftP = (np.nanmedian(x_vals['pillar_L_x']), np.nanmedian(y_vals['pillar_L_y']))
        rightP = (np.nanmedian(x_vals['pillar_R_x']), np.nanmedian(y_vals['pillar_R_y']))

        if vidpath_for_annotation is not None:

            print('Preparing for annotation of pillar position.')

            print('Reading in topdown video.')
            topdown_video = fm2p.pack_video_frames(vidpath_for_annotation, dwnsmpl=0.5)

            nF = np.size(topdown_video, 0)
            nF = int(nF/2)
            topdown_stillframe = topdown_video[nF,:,:].copy()

            # Clear the full video from memory
            del topdown_video
            gc.collect()

            print('Drag the polygon to a better position over the pillar. Close the figure when done.')
            topP = (topP[0]/2, topP[1]/2)
            rightP = (rightP[0]/2, rightP[1]/2)
            bottomP = (bottomP[0]/2, bottomP[1]/2)
            leftP = (leftP[0]/2, leftP[1]/2)

            [topP, rightP, bottomP, leftP] = fm2p.user_polygon_translation(
                pts=[topP, rightP, bottomP, leftP],
                image=topdown_stillframe)

            topP = (topP[0]*2, topP[1]*2)
            rightP = (rightP[0]*2, rightP[1]*2)
            bottomP = (bottomP[0]*2, bottomP[1]*2)
            leftP = (leftP[0]*2, leftP[1]*2)

        pillarX = [topP[0], bottomP[0], leftP[0], rightP[0]]
        pillarY = [topP[1], bottomP[1], leftP[1], rightP[1]]

        pillar_dict = fm2p.Eyecam.fit_ellipse(_, x=pillarX, y=pillarY)
        pillar_centroid = [pillar_dict['Y0'], pillar_dict['X0']]
        pillar_axes = (pillar_dict['long_axis'], pillar_dict['short_axis'])
        pillar_radius = np.mean(pillar_axes)

        tlA = (np.nanmedian(x_vals['arena_TL_x']), np.nanmedian(y_vals['arena_TL_y']))
        trA = (np.nanmedian(x_vals['arena_TR_x']), np.nanmedian(y_vals['arena_TR_y']))
        brA = (np.nanmedian(x_vals['arena_BR_x']), np.nanmedian(y_vals['arena_BR_y']))
        blA = (np.nanmedian(x_vals['arena_BL_x']), np.nanmedian(y_vals['arena_BL_y']))

        arena_dict = {
            'arenaTL': {
                'x': tlA[0],
                'y': tlA[1]
            },
            'arenaTR': {
                'x': trA[0],
                'y': trA[1]
            },
            'arenaBR': {
                'x': brA[0],
                'y': brA[1]
            },
            'arenaBL': {
                'x': blA[0],
                'y': blA[1]
            },
            'pillarT': {
                'x': topP[0],
                'y': topP[1]
            },
            'pillarB': {
                'x': bottomP[0],
                'y': bottomP[1]
            },
            'pillarL': {
                'x': leftP[0],
                'y': leftP[1]
            },
            'pillarR': {
                'x': rightP[0],
                'y': rightP[1]
            },
            'pillar_radius': pillar_radius,
            'pillar_centroid': {
                'x': pillar_centroid[0],
                'y': pillar_centroid[1]
            }
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
