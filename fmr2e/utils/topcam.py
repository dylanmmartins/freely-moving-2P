import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import fmr2e

class Topcam():

    def __init__(self, recording_path, recording_name, cfg=None):

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


    def find_files(self):
        
        self.top_dlc_h5 = fmr2e.find('{}*topDLC_resnet50*.h5'.format(self.recording_name), self.recording_path, MR=True)
        self.top_avi = fmr2e.find('{}*top.avi'.format(self.recording_name), self.recording_path, MR=True)
        self.topT_csv = fmr2e.find('{}*top.csv'.format(self.recording_name), self.recording_path, MR=True)

    def track_body(self):

        # pdf = PdfPages(os.path.join(self.recording_path,
        #                (self.recording_name + '_' + self.camname + '_tracking_figs.pdf')))

        # Get timestamps
        # topT = self.xrpts.timestamps.copy()
        topT = fmr2e.read_timestamp_file(self.topT_csv)

        # Read DLC data and filter by likelihood
        xyl, _ = fmr2e.open_dlc_h5(self.top_dlc_h5)
        x_vals, y_vals, likelihood = fmr2e.split_xyl(xyl)

        # Threshold by likelihoods
        x_vals = fmr2e.apply_liklihood_thresh(x_vals, likelihood)
        y_vals = fmr2e.apply_liklihood_thresh(y_vals, likelihood)

        # Conversion from pixels to cm
        left = 'tl_corner_x'
        right = 'tr_corner_x'
        
        dist_pxls = np.nanmedian(x_vals[right]) - np.nanmedian(x_vals[left])
        
        pxls2cm = dist_pxls / self.arena_width_cm

        # Topdown speed using neck point
        smooth_x = fmr2e.convfilt(fmr2e.nanmedfilt(x_vals['top_skull_x'], 7)[0], box_pts=20)
        smooth_y = fmr2e.convfilt(fmr2e.nanmedfilt(y_vals['top_skull_y'], 7)[0], box_pts=20)
        top_speed = np.sqrt(np.diff((smooth_x*60) / pxls2cm)**2 + np.diff((smooth_y*60) / pxls2cm)**2)
        # top_speed[top_speed>25] = np.nan

        # Get head angle from ear points
        lear_x = fmr2e.nanmedfilt(x_vals['left_ear_x'], 7)[0]
        lear_y = fmr2e.nanmedfilt(y_vals['left_ear_y'], 7)[0]
        rear_x = fmr2e.nanmedfilt(x_vals['right_ear_x'], 7)[0]
        rear_y = fmr2e.nanmedfilt(y_vals['right_ear_y'], 7)[0]

        # Rotate 90deg because ears are perpendicular to head yaw
        head_yaw = np.arctan2((lear_y - rear_y), (lear_x - rear_x)) + np.deg2rad(90)
        head_yaw_deg = np.rad2deg(head_yaw % (2*np.pi))

        # Body angle from neck and back points
        neck_x = fmr2e.nanmedfilt(x_vals['top_skull_x'], 7)[0].squeeze()
        neck_y = fmr2e.nanmedfilt(y_vals['top_skull_y'], 7)[0].squeeze()
        back_x = fmr2e.nanmedfilt(x_vals['base_tail_x'], 7)[0].squeeze()
        back_y = fmr2e.nanmedfilt(y_vals['base_tail_y'], 7)[0].squeeze()
        
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
            'topT': topT,
            'x': smooth_x,
            'y': smooth_y,
            'head_yaw_deg': head_yaw_deg,
            'body_yaw_deg': body_yaw_deg,
            'movement_yaw_deg': movement_yaw_deg
        }

        return topcam_dict

        # Plot traces of each labeled point and show frequency of good tracking
        # pt_names = list(self.xrpts['point_loc'].values)
        # x_cols = [i for i in pt_names if '_x' in i]
        # y_cols = [i for i in pt_names if '_y' in i]

        # plt.subplots(int(np.ceil(len(pt_names)/9)), 3,
        #              figsize=(20,15))

        # for i in range(len(x_cols)):

        #     x = self.xrpts.sel(point_loc=x_cols[i])
        #     y = self.xrpts.sel(point_loc=y_cols[i])

        #     plt.subplot(int(np.ceil(len(pt_names)/9)), 3, i+1)
        #     plt.plot(x)
        #     plt.plot(y)

        #     frac_good = np.sum(~np.isnan(x) * ~np.isnan(y)) / len(x)

        #     plt.title(pt_names[::3][i][:-2] + ' good='+str(np.round(frac_good.values,4)))

        # plt.tight_layout()
        # pdf.savefig()
        # plt.close()

        # plt.figure()
        # plt.plot(topT[:-1], top_speed, linewidth=1)
        # plt.xlabel('sec')
        # plt.ylabel('cm/sec')
        # plt.tight_layout()
        # pdf.savefig()
        # plt.close()

        # plt.figure()
        # plt.hist(top_speed, bins=40, density=True)
        # plt.xlabel('cm/sec')
        # plt.ylabel('fraction of time')
        # plt.tight_layout()
        # pdf.savefig()
        # plt.close()

        # plt.figure(figsize=(15,5))
        # plt.plot(topT, head_yaw_deg, '.', markersize=1)
        # plt.ylabel('head yaw (deg)')
        # plt.xlabel('sec')
        # plt.tight_layout()
        # pdf.savefig()
        # plt.close()

        # plt.figure(figsize=(15,5))
        # plt.plot(topT, body_yaw_deg, '.', markersize=1)
        # plt.ylabel('body yaw (deg)')
        # plt.xlabel('sec')
        # plt.tight_layout()
        # pdf.savefig()
        # plt.close()

        # plt.figure(figsize=(15,5))
        # plt.plot(topT, np.rad2deg(body_head_diff), '.', markersize=1)
        # plt.ylabel('head yaw - body yaw (deg)')
        # plt.xlabel('sec')
        # plt.tight_layout()
        # pdf.savefig()
        # plt.close()



        # plt.figure(figsize=(15,5))
        # plt.plot(topT[:-1], movement_yaw_deg, '.', markersize=1)
        # plt.ylabel('animal yaw (deg)')
        # plt.xlabel('sec')
        # plt.tight_layout()
        # pdf.savefig()
        # plt.close()

        # plt.figure(figsize=(15,5))
        # plt.plot(topT[:-1], movement_yaw_deg - body_yaw_deg[:-1], '.', markersize=1)
        # plt.ylabel('movement yaw - body yaw (deg)')
        # plt.xlabel('sec')
        # plt.tight_layout()
        # pdf.savefig()
        # plt.close()


        # # Plot of movement, heading, and speed
        # if self.make_all_plots:

        #     xbounds = np.array([np.nanmedian(self.xrpts.sel(point_loc='tl_p_corner_x').values),
        #                         np.nanmedian(self.xrpts.sel(point_loc='tr_p_corner_x').values),
        #                         np.nanmedian(self.xrpts.sel(point_loc='br_p_corner_x').values),
        #                         np.nanmedian(self.xrpts.sel(point_loc='bl_p_corner_x').values),
        #                         np.nanmedian(self.xrpts.sel(point_loc='tl_p_corner_x').values)])
            
        #     ybounds = np.array([np.nanmedian(self.xrpts.sel(point_loc='tl_p_corner_y').values),
        #                         np.nanmedian(self.xrpts.sel(point_loc='tr_p_corner_y').values),
        #                         np.nanmedian(self.xrpts.sel(point_loc='br_p_corner_y').values),
        #                         np.nanmedian(self.xrpts.sel(point_loc='bl_p_corner_y').values),
        #                         np.nanmedian(self.xrpts.sel(point_loc='tl_p_corner_y').values)])
            
        #     start = 1000
        #     maxf = 3600

        #     plt.subplots(1,2,figsize=(15,6))
        #     plt.subplot(121)
        #     plt.plot(xbounds, ybounds, 'k-')

        #     cmap = plt.cm.jet(np.linspace(0,1,maxf))

        #     for f in range(start, start+maxf):
        #         plt.plot(neck_x[f], neck_y[f], '.', color=cmap[f-start])

        #     plt.subplot(122)
        #     plt.plot(xbounds, ybounds, 'k-')

        #     speed_bins = np.linspace(0,int(np.ceil(np.nanmax(top_speed)/3)))
        #     spdcolors = plt.cm.magma(speed_bins)

        #     for f in np.arange(start,start+maxf+10,10):

        #         if ~np.isnan(top_speed[f]):
        #             usecolor = spdcolors[np.argmin(np.abs(top_speed[f] - speed_bins))]
                
        #         else:
        #             continue

        #         x0 = neck_x[f]
        #         y0 = neck_y[f]

        #         dX = 15*np.cos(head_yaw[f])
        #         dY = 15*np.sin(head_yaw[f])

        #         plt.arrow(x0, y0, dX, dY, facecolor=usecolor, width=7, edgecolor='k')

        #     plt.tight_layout()
        #     pdf.savefig()
        #     plt.close()

        # pdf.close()

        # if self.cfg['write_diagnostic_videos'] and self.make_speed_yaw_video:

        #     vid_save_path = os.path.join(self.recording_path,
        #                         (self.recording_name+'_'+self.camname+'_speed_yaw.avi'))
            
        #     start = 1000
        #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #     out_vid = cv2.VideoWriter(vid_save_path, fourcc, 60.0, (432, 288))
        #     maxprev = 25

        #     for f in tqdm(range(start,start+3600)):

        #         fig = plt.figure()

        #         plt.imshow(self.xrframes[f,:,:].astype(np.uint8), cmap='gray')
        #         plt.ylim([135,0])
        #         plt.xlim([0,180])
        #         plt.axis('off')

        #         plt.plot(lear_x[f]*0.25, lear_y[f]*0.25, 'b*')
        #         plt.plot(rear_x[f]*0.25, rear_y[f]*0.25, 'b*')

        #         plt.plot([neck_x[f]*0.25, (neck_x[f]*0.25)+15*np.cos(head_yaw[f])],
        #                  [neck_y[f]*0.25,(neck_y[f]*0.25)+15*np.sin(head_yaw[f])],
        #                  '-', linewidth=2, color='cyan') # head yaw
                
        #         plt.plot([back_x[f]*0.25, (back_x[f]*0.25)-15*np.cos(body_yaw[f])],
        #                  [back_y[f]*0.25, (back_y[f]*0.25)-15*np.sin(body_yaw[f])],
        #                  '-', linewidth=2, color='pink') # body yaw
                
        #         for p in range(maxprev):

        #             prevf = f - p

        #             plt.plot(neck_x[prevf]*0.25,
        #                      neck_y[prevf]*0.25, 'o', color='tab:purple',
        #                      alpha=(maxprev-p)/maxprev) # neck position history
                    
        #         # arrow for vector of motion
        #         if forward_run[f]:
        #             movvec_color = 'tab:green'
        #         elif backward_run[f]:
        #             movvec_color = 'tab:orange'
        #         elif fine_motion[f]:
        #             movvec_color = 'tab:olive'
        #         elif immobility[f]:
        #             movvec_color = 'tab:red'
                
        #         plt.arrow(neck_x[f]*0.25, neck_y[f]*0.25,
        #                   x_disp[f]*3, y_disp[f]*3,
        #                   color=movvec_color, width=1)
                
        #         # Save the frame out
        #         fig.canvas.draw()
        #         frame_as_array = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        #         frame_as_array = frame_as_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #         plt.close()

        #         img = cv2.cvtColor(frame_as_array, cv2.COLOR_RGB2BGR)
        #         out_vid.write(img.astype('uint8'))

        #     out_vid.release()
        

if __name__ == '__main__':
    
    basepath = r'K:\FreelyMovingEyecams\241204_DMM_DMM031_freelymoving'
    rec_name = '241204_DMM_DMM031_freelymoving_01'
    top = Topcam(basepath, rec_name)
    top.find_files()
    top.get_head_body_yaw()
