"""
fm2p/utils/topcam.py
Topdown tracking class

DMM, 2024
"""


import os
import json
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import fm2p


class Topcam():

    def __init__(self, recording_path, recording_name, cfg=None, props=None, rnum=np.nan):

        self.recording_path = recording_path
        self.recording_name = recording_name

        if cfg is None:
            self.likelihood_thresh = 0.99
            self.arena_width_cm = 22. # cm
            self.running_thresh = 2. # cm/s
            self.forward_thresh = 40 # deg
        elif cfg is not None:
            self.likelihood_thresh = cfg['likelihood_thresh']
            self.arena_width_cm = cfg['arena_width_cm']
            self.running_thresh = cfg['running_thresh']
            self.forward_thresh = cfg['forward_thresh']
        
        if (props is not None) and (~np.isnan(rnum)):
            with open(props, 'r') as f:
                session_props = json.load(f)
            self.rstr = 'R{:02}'.format(rnum)
            rdir = session_props[self.rstr]['rec_dir']
            self.top_dlc_h5 = os.path.join(rdir, session_props[self.rstr]['top_dlc'])
            self.top_avi = os.path.join(rdir, session_props[self.rstr]['top_vid'])


    def find_files(self):
        
        self.top_dlc_h5 = fm2p.find('{}*topDLC_resnet50*.h5'.format(self.recording_name), self.recording_path, MR=True)
        self.top_avi = fm2p.find('{}*top.mp4'.format(self.recording_name), self.recording_path, MR=True)
        # self.topT_csv = fm2p.find('{}*top.csv'.format(self.recording_name), self.recording_path, MR=True)

    def track_body(self):

        # Get timestamps
        # topT = self.xrpts.timestamps.copy()
        # topT = fm2p.read_timestamp_file(self.topT_csv)

        # Read DLC data and filter by likelihood
        xyl, _ = fm2p.open_dlc_h5(self.top_dlc_h5)
        x_vals, y_vals, likelihood = fm2p.split_xyl(xyl)

        # Threshold by likelihoods
        x_vals = fm2p.apply_liklihood_thresh(x_vals, likelihood, threshold=0.8)
        y_vals = fm2p.apply_liklihood_thresh(y_vals, likelihood, threshold=0.8)

        # Conversion from pixels to cm
        left = 'tl_corner_x'
        right = 'tr_corner_x'
        
        dist_pxls = np.nanmedian(x_vals[right]) - np.nanmedian(x_vals[left])
        
        pxls2cm = dist_pxls / self.arena_width_cm
        print(pxls2cm)

        # Topdown speed using neck point
        smooth_x = fm2p.convfilt(fm2p.nanmedfilt(x_vals['base_tail_x'], 7)[0], box_pts=20)
        smooth_y = fm2p.convfilt(fm2p.nanmedfilt(y_vals['base_tail_y'], 7)[0], box_pts=20)
        top_speed = np.sqrt(np.diff((smooth_x*60) / pxls2cm)**2 + np.diff((smooth_y*60) / pxls2cm)**2)
        # top_speed[top_speed>25] = np.nan

        # Get head angle from ear points
        lear_x = fm2p.nanmedfilt(x_vals['left_ear_x'], 7)[0]
        lear_y = fm2p.nanmedfilt(y_vals['left_ear_y'], 7)[0]
        rear_x = fm2p.nanmedfilt(x_vals['right_ear_x'], 7)[0]
        rear_y = fm2p.nanmedfilt(y_vals['right_ear_y'], 7)[0]

        # Rotate 90deg because ears are perpendicular to head yaw
        head_yaw = np.arctan2((lear_y - rear_y), (lear_x - rear_x)) + np.deg2rad(90)
        head_yaw_deg = np.rad2deg(head_yaw % (2*np.pi))

        # Body angle from neck and back points
        neck_x = np.mean([fm2p.nanmedfilt(lear_x, 7)[0].squeeze(), fm2p.nanmedfilt(rear_x, 7)[0].squeeze()])
        neck_y = np.mean([fm2p.nanmedfilt(lear_y, 7)[0].squeeze(), fm2p.nanmedfilt(rear_y, 7)[0].squeeze()])
        back_x = fm2p.nanmedfilt(x_vals['base_tail_x'], 7)[0].squeeze()
        back_y = fm2p.nanmedfilt(y_vals['base_tail_y'], 7)[0].squeeze()
        
        body_yaw = np.arctan2((neck_y - back_y), (neck_x - back_x))
        body_yaw_deg = np.rad2deg(body_yaw % (2*np.pi))

        body_head_diff = head_yaw - body_yaw

        body_head_diff[body_head_diff<-np.deg2rad(120)] = np.nan
        body_head_diff[body_head_diff>np.deg2rad(120)] = np.nan

        # Angle of body movement ("movement yaw")
        x_disp = np.diff((smooth_x*60) / pxls2cm)
        y_disp = np.diff((smooth_y*60) / pxls2cm)

        movement_yaw = np.arctan2(y_disp, x_disp)
        movement_yaw_deg = np.rad2deg(movement_yaw % (2*np.pi))

        # Definitions of state
        movement_minus_body = movement_yaw - body_yaw[:-1]

        running = (top_speed>self.running_thresh)
        forward = (np.abs(movement_minus_body) < np.deg2rad(self.forward_thresh))
        backward = (np.abs(movement_minus_body + np.deg2rad(180) % (2*np.pi)) < np.deg2rad(self.forward_thresh))
        
        forward_run = running * forward
        backward_run = running * backward
        fine_motion = running * ~forward * ~backward
        stationary = ~running

        topcam_dict = {
            'speed': top_speed,
            'head_yaw': head_yaw,
            'body_yaw': body_yaw,
            'body_head_diff': body_head_diff,
            'movement_yaw': movement_yaw,
            'movement_minus_body': movement_minus_body,
            'forward_run': forward_run,
            'backward_run': backward_run,
            'fine_motion': fine_motion,
            'stationary': stationary,
            # 'topT': topT,
            'x': smooth_x,
            'y': smooth_y,
            'head_yaw_deg': head_yaw_deg,
            'body_yaw_deg': body_yaw_deg,
            'movement_yaw_deg': movement_yaw_deg,
            'x_displacement': x_disp,
            'y_displacement': y_disp
        }

        return xyl, topcam_dict
    
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


    def save_tracking(self, topcam_dict, dlc_xyl, vid_array):

        xyl_dict = dlc_xyl.to_dict()
        vid_dict = {'video': vid_array}

        save_dict = {**xyl_dict, **topcam_dict, **vid_dict}

        savedir = os.path.join(self.recording_path, self.recording_name)
        _savepath = os.path.join(savedir, '{}_top_tracking.h5'.format(self.recording_name))
        fm2p.write_h5(_savepath, save_dict)
        

if __name__ == '__main__':
    
    basepath = r'K:\FreelyMovingEyecams\241204_DMM_DMM031_freelymoving'
    rec_name = '241204_DMM_DMM031_freelymoving_01'
    top = Topcam(basepath, rec_name)
    top.find_files()
    top.get_head_body_yaw()
