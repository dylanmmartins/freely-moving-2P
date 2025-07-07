# -*- coding: utf-8 -*-
"""
Eye/pupil tracking.

Example usage
-------------
    eyecam = fm2p.Eyecam(recording_path, recording_name)
    # Search for files automatically
    eyecam.find_files()
    # OR... add the files manually with file paths
    eyecam.add_files(eye_dlc_h5, eye_avi, eyeT)
    xyl, ellipse_dict = eyecam.track_pupil()
    savepath = eyecam.save_tracking(ellipse_dict, xyl, cyclotorsion_dict)

Classes
-------
Eyecam
    Class for tracking pupil in eye camera recordings.

Written: DMM, 2022-2024
"""


import os
import yaml
import numpy as np
import pandas as pd
import multiprocessing
import scipy.signal
import scipy.optimize
import scipy.stats
import cv2
from tqdm import tqdm
import astropy.convolution
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

import fm2p


def sigmoid_curve(xval, a, b, c):
    """ Sigmoid curve function.

    Parameters
    ----------
    xval : np.ndarray
        Array of values.
    a : float
        Minimum value of the curve.
    b : float
        Maximum value of the curve.
    c : float
        Midpoint of the curve.

    Returns
    -------
    curve : np.ndarray
        Sigmoid curve.
    """

    curve = a + (b-a) / (1 + 10**((c - xval)*2))

    return curve


def sigmoid_fit(d):
    """ Fit a sigmoid curve to data.

    Parameters
    ----------
    d : np.ndarray
        Array of values.

    Returns
    -------
    popt : np.ndarray
        Fitted parameters of the sigmoid curve.
    ci : np.ndarray
        Confidence intervals of the fitted parameters.
    """

    try:
        # Fit the sigmoid curve to the data.
        popt, pcov = scipy.optimize.curve_fit(
            sigmoid_curve,
            xdata=range(1,len(d)+1),
            ydata=d,
            p0=[100.0,200.0,len(d)/2],
            method='lm',
            xtol=10**-3,
            ftol=10**-3
        )
        # Calculate the confidence intervals.
        ci = np.sqrt(np.diagonal(pcov))

    except RuntimeError:
        # If the fit fails, return NaN values.
        popt = np.nan * np.zeros(4)
        ci = np.nan * np.zeros(4)

    return (popt, ci)


class Eyecam():

    def __init__(self, recording_path, recording_name, cfg=None):
        """
        Parameters
        ----------
        recording_path : str
            Directory of the recording.
        recording_name : str
            Name of the recording (e.g., '241219_DMM_DMM037_mini2p')
        cfg : dict
            Optional. Dictionary of config options. If not provided, default values will be used.
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

    def find_files(self):
        """ Gather files.

        This function will search the recording directory for the following files:
            *eye_deinterDLC_resnet50*.h5
            *_eye.csv
            *_eye_deinter.avi

        """

        self.eye_dlc_h5 = fm2p.find('{}*eye_deinterDLC_resnet50*.h5'.format(self.recording_name), self.recording_path, MR=True)
        self.eye_avi = fm2p.find('{}*eye_deinter.avi'.format(self.recording_name), self.recording_path, MR=True)
        self.eyeT_csv = fm2p.find('{}*_eye.csv'.format(self.recording_name), self.recording_path, MR=True)

    def add_files(self, eye_dlc_h5, eye_avi, eyeT):
        """ Add files without searching.

        Parameters
        ----------
        eye_dlc_h5 : str
            Path to the eye DLC h5 file.
        eye_avi : str
            Path to the eye video file.
        eyeT : str
            Path to the eye timestamps file.
        """
        
        self.eye_dlc_h5 = eye_dlc_h5
        self.eye_avi = eye_avi
        self.eyeT_csv = eyeT

    def fit_ellipse(self, x, y):
        """ Fit an ellipse to points labeled around the perimeter of pupil.

        Parameters
        ----------
        x : np.array
            Positions of points along the x-axis for a single video frame.
        y : np.array
            Positions of labeled points along the y-axis for a single video frame.

        Returns
        -------
        ellipse_dict : dict
            Parameters of the ellipse...
            X0 : center at the x-axis of the non-tilt ellipse
            Y0 : center at the y-axis of the non-tilt ellipse
            a : radius of the x-axis of the non-tilt ellipse
            b : radius of the y-axis of the non-tilt ellipse
            long_axis : radius of the long axis of the ellipse
            short_axis : radius of the short axis of the ellipse
            angle_to_x : angle from long axis to horizontal plane
            angle_from_x : angle from horizontal plane to long axis
            X0_in : center at the x-axis of the tilted ellipse
            Y0_in : center at the y-axis of the tilted ellipse
            phi : tilt orientation of the ellipse in radians
        """

        # Remove bias of the ellipse
        meanX = np.mean(x)
        meanY = np.mean(y)
        x = x - meanX
        y = y - meanY

        # Estimation of the conic equation
        X = np.array([x**2, x*y, y**2, x, y])
        X = np.stack(X).T
        a = np.dot(np.sum(X, axis=0), np.linalg.pinv(np.matmul(X.T,X)))

        # Extract parameters from the conic equation
        a, b, c, d, e = a[0], a[1], a[2], a[3], a[4]

        # Eigen decomp
        Q = np.array([[a, b/2],[b/2, c]])
        eig_val, eig_vec = np.linalg.eig(Q)

        # Get angle to long axis
        if eig_val[0] < eig_val[1]:
            angle_to_x = np.arctan2(eig_vec[1,0], eig_vec[0,0])
        else:
            angle_to_x = np.arctan2(eig_vec[1,1], eig_vec[0,1])

        # Get ellipse angles.
        angle_from_x = angle_to_x
        orientation_rad = 0.5 * np.arctan2(b, (c-a))
        cos_phi = np.cos(orientation_rad)
        sin_phi = np.sin(orientation_rad)

        # Rotate ellipse to find the center point of the tilted ellipse.
        a, b, c, d, e = [a*cos_phi**2 - b*cos_phi*sin_phi + c*sin_phi**2,
                        0,
                        a*sin_phi**2 + b*cos_phi*sin_phi + c*cos_phi**2,
                        d*cos_phi - e*sin_phi,
                        d*sin_phi + e*cos_phi]

        meanX, meanY = [cos_phi*meanX - sin_phi*meanY,
                        sin_phi*meanX + cos_phi*meanY]

        # Check if conc expression represents an ellipse
        test = a*c

        if test > 0:

            # Make sure coefficients are positive
            if a<0:
                a, c, d, e = [-a, -c, -d, -e]

            # Final ellipse parameters
            X0 = meanX - d/2/a
            Y0 = meanY - e/2/c
            F = 1 + (d**2)/(4*a) + (e**2)/(4*c)
            a = np.sqrt(F/a)
            b = np.sqrt(F/c)
            long_axis = 2*np.maximum(a,b)
            short_axis = 2*np.minimum(a,b)

            # Rotate axes backwards to find center point of
            # original tilted ellipse
            R = np.array([[cos_phi, sin_phi], [-sin_phi, cos_phi]])
            P_in = R @ np.array([[X0],[Y0]])
            X0_in = P_in[0][0]
            Y0_in = P_in[1][0]

            # Organize parameters in dictionary to return
            ellipse_dict = {
                'X0':X0,
                'Y0':Y0,
                'F':F,
                'a':a,
                'b':b,
                'long_axis':long_axis/2,
                'short_axis':short_axis/2,
                'angle_to_x':angle_to_x,
                'angle_from_x':angle_from_x,
                'cos_phi':cos_phi,
                'sin_phi':sin_phi,
                'X0_in':X0_in,
                'Y0_in':Y0_in,
                'phi':orientation_rad
            }

        else:

            # If the conic equation didn't return an ellipse, do not
            # return any real values and fill the dictionary with NaNs.
            dict_keys = ['X0','Y0','F','a','b','long_axis',
                         'short_axis','angle_to_x','angle_from_x',
                         'cos_phi','sin_phi','X0_in','Y0_in','phi']
            dict_vals = list(np.ones([len(dict_keys)]) * np.nan)

            ellipse_dict = dict(zip(dict_keys, dict_vals))
        
        return ellipse_dict


    def track_pupil(self):
        """ Track the pupil in the current recording.
        Need to run self.find_files() before running this function.
        
        Returns
        -------
        xyl : pd.DataFrame
            Positions in x and y for all points tracked by DLC, along
            with the likelihood values for each of those points. Rows are
            camera frames, columns are differnet tracked positions.
        ellipse_dict : dict
            Ellipse fit results. Each key is a measured value, and value is
            either a list or a float.
            Parameters are:
                theta : horizontal pupil orientation
                phi : vertical pupil orientation
                longaxis : radius in pixels of the longer axis of the pupil
                shortaxis : radius in pixels of the shorter axis of the pupil,
                    i.e., off axis from the tilted angle.
                X0 : center of the pupil in the x dimension. Values are in pixels.
                Y0 : center of the pupil in the y dimension. Values are in pixels.
                ellipse_phi : Tilt of the ellipse relative to horizontal/vertical
                    axes. This is different than theta or phi values, and is only
                    needed if you want to visualize the ellipse itself.
                eyeT : Eye camera timestamps.
                cam_center_x : Center of the camera that the ellipse tilts around
                    along the x axis.
                cam_center_y : Center of the camera that the elipse tilts around along
                    the y axis.
        """

        eye_dist_thresh = self.cfg['eye_dist_thresh']
        eye_pxl2cm = self.cfg['eye_pxl2cm']
        likelihood_thresh = self.cfg['likelihood_thresh']
        eye_trackable_N = self.cfg['eye_trackable_N']
        eye_calibration_N = self.cfg['eye_calibration_N']
        eye_ellipse_thresh = self.cfg['eye_ellipse_thresh']

        # Set up the pdf to be saved out with diagnostic figures
        pdf_name = '{}_eye_tracking_figs.pdf'.format(self.recording_name)
        pdf = PdfPages(os.path.join(self.recording_path, pdf_name))

        # read deeplabcut file
        xyl, _ = fm2p.open_dlc_h5(self.eye_dlc_h5)
        x_vals, y_vals, likelihood = fm2p.split_xyl(xyl)

        # Threshold by likelihoods
        x_vals = fm2p.apply_liklihood_thresh(
            x_vals, likelihood, threshold=likelihood_thresh
        )
        y_vals = fm2p.apply_liklihood_thresh(
            y_vals, likelihood, threshold=likelihood_thresh
        )

        # Threshold by number of sucessfully tracked pupil points
        pupil_count = np.sum(likelihood >= likelihood_thresh, 1)
        usegood_eye = pupil_count >= eye_trackable_N
        usegood_eyecalib = pupil_count >= eye_calibration_N

        print(' !!  N={}/{} frames dropped for not meeting required number of tracked points ({}).'.format(
            np.sum(~usegood_eye), len(pupil_count), eye_trackable_N
        ))

        # Threshold out pts more than a given distance away from nanmean of that point
        std_thresh_x = np.empty(np.shape(x_vals))
        std_thresh_y = np.empty(np.shape(y_vals))

        # Get rid of points that are too far away from the mean
        for point_loc in range(0,np.size(x_vals, 1)):
            _val = x_vals.iloc[:,point_loc]
            std_thresh_x[:,point_loc] = (np.abs(np.nanmean(_val) - _val) / eye_pxl2cm) > eye_dist_thresh

        # Same for y values.
        for point_loc in range(0,np.size(x_vals, 1)):
            _val = y_vals.iloc[:,point_loc]
            std_thresh_y[:,point_loc] = (np.abs(np.nanmean(_val) - _val) / eye_pxl2cm) > eye_dist_thresh

        # Threshold using the standard deviation
        std_thresh_x = np.nanmean(std_thresh_x, 1)
        std_thresh_y = np.nanmean(std_thresh_y, 1)
        x_vals[std_thresh_x > 0] = np.nan
        y_vals[std_thresh_y > 0] = np.nan

        num_removed_for_dist_std = np.sum((std_thresh_x > 0) * (std_thresh_y > 0))
        print(' !!  N={}/{} frames dropped for distance std threshold.'.format(num_removed_for_dist_std, np.size(x_vals, 0)))

        ellipse = np.empty([len(usegood_eye), 14])
        
        # Step through each frame, fit an ellipse to points, and add ellipse
        # parameters to array with data for all frames together.
        cols = [
            'X0',           # 0
            'Y0',           # 1
            'F',            # 2
            'a',            # 3
            'b',            # 4
            'long_axis',    # 5
            'short_axis',   # 6
            'angle_to_x',   # 7
            'angle_from_x', # 8
            'cos_phi',      # 9
            'sin_phi',      # 10
            'X0_in',        # 11
            'Y0_in',        # 12
            'phi'           # 13
        ]

        linalgerror = 0
        for step in tqdm(range(0,len(usegood_eye))):
            
            if usegood_eye[step] == True:
                
                try:

                    e_t = self.fit_ellipse(x_vals.iloc[step].values,
                                           y_vals.iloc[step].values)
                    
                    ellipse[step] = [
                        e_t['X0'],              # 0
                        e_t['Y0'],              # 1
                        e_t['F'],               # 2
                        e_t['a'],               # 3
                        e_t['b'],               # 4
                        e_t['long_axis'],       # 5
                        e_t['short_axis'],      # 6
                        e_t['angle_to_x'],      # 7
                        e_t['angle_from_x'],    # 8
                        e_t['cos_phi'],         # 9
                        e_t['sin_phi'],         # 10
                        e_t['X0_in'],           # 11
                        e_t['Y0_in'],           # 12
                        e_t['phi']              # 13
                    ]
                
                except np.linalg.LinAlgError as e:

                    linalgerror = linalgerror + 1
                    ellipse[step] = list(np.ones([len(cols)]) * np.nan)
            
            elif usegood_eye[step] == False:

                ellipse[step] = list(np.ones([len(cols)]) * np.nan)

        print('LinAlg error count = ' + str(linalgerror))
    
        # Get indices where the ellipse fit does not exceed an ellipticity threshold
        ellipticity_test = ((ellipse[:,6] / ellipse[:,5]) < eye_ellipse_thresh)
        usegood_ellipcalb = np.where((usegood_eyecalib == True) & ellipticity_test)

        num_removed_for_ellipticity = np.sum(~ellipticity_test)
        print(' !!  N={}/{} frames dropped because of ellipticity threshold.'.format(num_removed_for_ellipticity, np.sum(usegood_eyecalib)))
        
        # Limit number of frames used for calibration
        f_lim = 50000
        if np.size(usegood_ellipcalb,1) > f_lim:
            shortlist = sorted(np.random.choice(usegood_ellipcalb[0],
                                size=f_lim, replace=False))
        else:
            shortlist = usegood_ellipcalb
        
        # Find camera center
        A = np.vstack([np.cos(ellipse[shortlist,7]),
                       np.sin(ellipse[shortlist,7])])
        b = np.expand_dims(np.diag(A.T @ np.squeeze(ellipse[shortlist, 11:13].T)), axis=1)
        cam_cent = np.linalg.inv(A @ A.T) @ A @ b
        
        # Ellipticity and scale
        ellipticity = (ellipse[shortlist,6] / ellipse[shortlist,5]).T
        
        try:
            scale = np.nansum(np.sqrt(1 - (ellipticity)**2) *                       \
            (np.linalg.norm(ellipse[shortlist, 11:13] - cam_cent.T, axis=0)))       \
            / np.sum(1 - (ellipticity)**2)
        
        except ValueError:
            scale = np.nansum(np.sqrt(1 - (ellipticity)**2) *                       \
            (np.linalg.norm(ellipse[shortlist, 11:13] - cam_cent.T, axis=1)))       \
            / np.sum(1 - (ellipticity)**2)
        
        # Pupil angles

        # Horizontal orientation (THETA)
        theta = np.arcsin((ellipse[:,11] - cam_cent[0]) / scale)

        # Vertical orientation (PHI)
        phi = np.arcsin((ellipse[:,12] - cam_cent[1]) / np.cos(theta) / scale)

        # Timestamps
        eyeT = fm2p.read_timestamp_file(self.eyeT_csv,
                                  position_data_length=len(theta))

        # Organize data to return as an xarray of most essential parameters
        ellipse_dict = {
            'theta':list(theta),
            'phi':list(phi),
            'longaxis':list(ellipse[:,5]),
            'shortaxis':list(ellipse[:,6]),
            'X0':list(ellipse[:,11]),
            'Y0':list(ellipse[:,12]),
            'ellipse_phi':list(ellipse[:,7]),
            'eyeT': eyeT,
            'cam_center_x': cam_cent[0,0],
            'cam_center_y': cam_cent[1,0]
        }
        
        fig1, [[ax1,ax2,ax3,ax4],[ax5,ax6,ax7,ax8]] = plt.subplots(2,4, figsize=(15,5.5), dpi=300)

        # How well did eye track?
        ax1.plot(pupil_count[0:-1:10])
        ax1.set_title('{:.3}% good'.format(np.mean(usegood_eye)*100))
        ax1.set_ylabel('num good pupil points')
        ax1.set_xlabel('every 10th frame')

        # Hist of eye tracking quality
        ax2.hist(pupil_count, bins=9, range=(0,9), density=True)
        ax2.set_xlabel('num good eye points')
        ax2.set_ylabel('fraction of frames')

        # Trace of horizontal orientation
        ax3.plot(np.rad2deg(theta)[0:-1:10])
        ax3.set_title('theta')
        ax3.set_ylabel('deg')
        ax3.set_xlabel('every 10th frame')

        # Trace of vertical orientation
        ax4.plot(np.rad2deg(phi)[0:-1:10])
        ax4.set_title('phi')
        ax4.set_ylabel('deg')
        ax4.set_xlabel('every 10th frame')

        # Ellipticity histogram
        fig_dwnsmpl = 100

        try:
            # Hist of ellipticity
            ax5.hist(ellipticity, density=True)
            ax5.set_title('ellipticity; thresh='+str(eye_ellipse_thresh))
            ax5.set_ylabel('ellipticity')
            ax5.set_xlabel('fraction of frames')
            
            # Eye axes relative to center
            w = ellipse[:,7]
            for i in range(0,len(usegood_ellipcalb)):

                _show = usegood_ellipcalb[i::fig_dwnsmpl]

                ax6.plot((ellipse[_show,11] + [-5 * np.cos(w[_show]),       \
                         5 * np.cos(w[_show])]),                            \
                         (ellipse[_show,12] + [-5*np.sin(w[_show]),         \
                         5*np.sin(w[_show])]))

            ax6.plot(cam_cent[0], cam_cent[1], 'r*')
            ax6.set_title('eye axes relative to center')

        except Exception as e:
            print('Figure error in plots of ellipticity and axes relative to center')
            print(e)
            
        # Check calibration
        try:
            
            xvals = np.linalg.norm(ellipse[usegood_eyecalib, 11:13].T - cam_cent, axis=0)

            yvals = scale * np.sqrt( 1 - (ellipse[usegood_eyecalib, 6]              \
                                        / ellipse[usegood_eyecalib, 5]) **2)

            calib_mask = ~np.isnan(xvals) & ~np.isnan(yvals)

            slope, _, r_value, _, _ = scipy.stats.linregress(xvals[calib_mask],
                                                             yvals[calib_mask].T)
        
        except ValueError:
            print('No good frames that meet criteria... check DLC tracking!')

        # Save out camera center and scale as np array
        ellipse_dict['scale'] = float(scale)
        ellipse_dict['regression_r'] = float(r_value)
        ellipse_dict['regression_m'] = float(slope)

        # Figures of scale and center
        try:
            ax7.plot(xvals[::fig_dwnsmpl],
                     yvals[::fig_dwnsmpl], '.', markersize=1)
            ax7.plot(np.linspace(0,50), np.linspace(0,50), 'r')
            ax7.set_title('scale={:.3} r={:.3} m={:.3}'.format(scale, r_value, slope))
            ax7.set_xlabel('pupil camera dist')
            ax7.set_ylabel('scale * ellipticity')

            # Calibration of camera center
            delta = (cam_cent - ellipse[:, 11:13].T)

            _useec = usegood_eyecalib[::fig_dwnsmpl]
            _use3 = np.squeeze(usegood_ellipcalb)[::fig_dwnsmpl]

            ax8.plot(np.linalg.norm(delta[:,_useec], 2, axis=0),                \
                    ((delta[0, _useec].T * np.cos(ellipse[_useec, 7]))          \
                    + (delta[1, _useec].T * np.sin(ellipse[_useec, 7])))        \
                    / np.linalg.norm(delta[:, _useec], 2, axis=0).T,            \
                    'y.', markersize=1)

            ax8.plot(np.linalg.norm(delta[:,_use3], 2, axis=0),                 \
                    ((delta[0, _use3].T * np.cos(ellipse[_use3,7]))             \
                    + (delta[1, _use3].T * np.sin(ellipse[_use3, 7])))          \
                    / np.linalg.norm(delta[:, _use3], 2, axis=0).T,             \
                    'r.', markersize=1)

            ax8.set_title('camera center calibration')
            ax8.set_ylabel('abs([PC-EC]).[cos(w);sin(w)]')
            ax8.set_xlabel('abs(PC-EC)')

            patch0 = mpatches.Patch(color='y', label='all pts')
            patch1 = mpatches.Patch(color='y', label='calibration pts')
            plt.legend(handles=[patch0, patch1])

        except Exception as e:
            print(e)
            print('Error in scale, center, and calibration figures. Skipping these for now')
        
        fig1.tight_layout()
        pdf.savefig()
        plt.close()

        pdf.close()

        for k,v in ellipse_dict.items():
            ellipse_dict[k] = np.array(v)

        return xyl, ellipse_dict
    
    
    def save_tracking(self, ellipse_dict, dlc_xyl, vid_array, cyclotorsion):
        """ Save eye tracking data out.

        Parameters
        ----------
        ellipse_dict : dict
            Dictionary of ellipse fit values returned from track_pupil().
        dlc_xyl : pd.DataFrame
            X, y, and likelihoods as a dataframe. Each row is a camera frame.
        vid_array : np.array
            Numpy array of the video data, where shape is (time, height, width).
        cyclotorsion : np.array
            Cyclotorsion values for each frame of the video. This is a 1D array
            with the same length as the number of frames in the video.
        
        Returns
        -------
        _savepath : str
            Path to the saved h5 file containing the eye tracking data.
        """

        xyl_dict = dlc_xyl.to_dict()
        vid_dict = {'video': vid_array}

        save_dict = {**xyl_dict, **ellipse_dict, **vid_dict}

        save_dict['omega'] = cyclotorsion

        _savepath = os.path.join(self.recording_path, '{}_eye_tracking.h5'.format(self.recording_name))
        fm2p.write_h5(_savepath, save_dict)

        return _savepath


    def measure_cyclotorsion(self, ellipse_dict, vidpath, startInd=0, endInd=-1,
                             usemp=True, doVideo=False):
        """ Measure cyclotorsion of the eye.

        Parameters
        ----------
        ellipse_dict : dict
            Dictionary of ellipse fit values returned from track_pupil().
        vidpath : str
            Path to the eye video file.
        startInd : int
            Start index for the video frames to analyze.
        endInd : int
            End index for the video frames to analyze.
        usemp : bool
            Whether to use multiprocessing for the sigmoid fit function.
        doVideo : bool
            Whether to create a video of the cyclotorsion measurement.
        
        Returns
        -------
        cyclotorsion_dict : dict
            Dictionary of cyclotorsion values for each frame of the video.
            Parameters are:
                cyclotorsion_shift: Cyclotorsion shift values for each frame.
                cyclotorsion_raw_total_shift: Raw total shift values for each frame.
                cyclotorsion_final_template: Final template used for cyclotorsion measurement.
                cyclotorsion_raw_rfit: Raw radius fit values for each frame.
                cyclotorsion_conv_rfit: Convolved radius fit values for each frame.
        """

        # Get the arrays from the start and end indices for the video frames
        eyeT = ellipse_dict['eyeT'][startInd:endInd]
        longaxis = ellipse_dict['longaxis'][startInd:endInd]
        shortaxis = ellipse_dict['shortaxis'][startInd:endInd]
        centX = ellipse_dict['X0'][startInd:endInd]
        centY = ellipse_dict['Y0'][startInd:endInd]

        # Align timestamps relative to recording onset
        eyeT = eyeT.copy() - eyeT[0]

        # Load video and crop in time around TTL start/end
        eyevid = fm2p.pack_video_frames(vidpath, ds=1.)[startInd:endInd,:,:]

        # Set up range of degrees in radians
        rad_range = np.deg2rad(np.arange(360))

        # Video dimensions
        totalF = np.size(eyevid, 0)
        frame_inds = np.arange(0, totalF)
        set_size = (np.size(eyevid,2), np.size(eyevid,1)) # width, height

        # Set up for the multiprocessing that'll be used during sigmoid fit function
        if usemp:
            print('Multiprocessing CPU count = {}'.format(multiprocessing.cpu_count()))
            n_proc = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(processes=n_proc)

        print('Getting cross-section of pupil at each angle and fitting to sigmoid (SLOW!)')
        all_raw_rfit = np.zeros([totalF, len(rad_range)]) * np.nan
        all_conv_rfit = np.zeros([totalF, len(rad_range)]) * np.nan

        key_error_count = 0

        for f in tqdm(frame_inds):
            try:

                img = eyevid[f,:,:].copy()

                # Range of values over mean radius
                ranger = 10
                meanr = 0.5 * (longaxis[f] + shortaxis[f])
                r = range(int(meanr - ranger), int(meanr + ranger))

                # Get cross-section of pupil at each angle 1-360 and fit to sigmoid
                pupil_edge = np.zeros([len(rad_range), len(r)])
                for i in range(0, len(r)):
                    pupil_edge[:,i] = img[
                        ((centY[f]+r[i]*np.sin(rad_range)).astype(int),
                        (centY[f]+r[i]*np.cos(rad_range)).astype(int))
                    ]

                if usemp:
                    # Apply sigmoid fit with multiprocessing. The result.get() is the slow process
                    param_mp = [pool.apply_async(sigmoid_fit, args=(pupil_edge[n,:],)) for n in range(len(rad_range))]
                    params_output = [result.get() for result in param_mp]
                elif not usemp:
                    params_output = [sigmoid_fit(pupil_edge[n,:]) for n in range(len(rad_range))]

                # Unpack outputs of sigmoid fit
                params = []
                ci = []
                for vals in params_output:
                    params.append(vals[0])
                    ci.append(vals[1])
                params = np.stack(params)
                ci = np.stack(ci)

                # Extract radius variable from parameters
                rfit_raw = params[:,2] - 1

                # Drop frames based on confidence interval
                ci_temp = (ci[:,0]>5) | (ci[:,1]>5)  | (ci[:,2]>0.75)
                rfit_raw[ci_temp] = np.nan

                # Remove if luminance goes the wrong way
                # rfit_raw[(params[:,1] - params[:,0]) < 10] = np.nan
                # rfit_raw[params[:,1] > 250] = np.nan

                try:
                    # Median filtered
                    rfit_filt = fm2p.nanmedfilt(rfit_raw, 3).flatten() # was 5

                    # Apply convolution
                    filtsize = 25 # was 31
                    rfit_conv_ = astropy.convolution.convolve(
                        rfit_filt,
                        np.ones(filtsize)/filtsize,
                        boundary='wrap'
                    )

                    # Subtract baseline because our points aren't perfectly centered on ellipse
                    rfit_conv = rfit_filt - rfit_conv_

                except ValueError as e: 
                    # In case every value in rfit is NaN
                    rfit_raw = np.zeros(len(rad_range)) * np.nan
                    rfit_conv = np.zeros(len(rad_range)) * np.nan

            except (KeyError, ValueError) as e:
                key_error_count = key_error_count + 1
                rfit_raw = np.zeros(len(rad_range)) * np.nan
                rfit_conv = np.zeros(len(rad_range)) * np.nan

            # Get rid of outlier points
            rfit_conv[np.abs(rfit_conv)>3] = np.nan
            
            all_raw_rfit[f,:] = rfit_raw.copy()
            all_conv_rfit[f,:] = rfit_conv.copy()

        # Correlation across first minute of recording
        timepoint_corr_rfit = pd.DataFrame(all_conv_rfit[frame_inds[:3600]]).T.corr().to_numpy()

        pupil_update = all_conv_rfit[frame_inds].copy()
        total_shift = np.zeros(len(frame_inds))
        peak = np.zeros(len(frame_inds)) * np.nan
        c = total_shift.copy()
        # Use mean as template
        template = np.nanmean(all_conv_rfit[frame_inds].copy(), 0)

        # xcorr of two random timepoints
        tworandframes = False
        while tworandframes is False:
            try:
                rind0 = np.random.random_integers(frame_inds[0], frame_inds[-2])
                rind1 = np.random.random_integers(frame_inds[0], frame_inds[-2])
                rfit2times_cc, rfit2times_lags = fm2p.nanxcorr(all_conv_rfit[rind0], all_conv_rfit[rind1], 11)
                tworandframes = True
            except ZeroDivisionError:
                pass

        # Cross correlation between rand frame and the template
        template_rfitconv_cc, template_rfit_cc_lags = fm2p.nanxcorr(all_conv_rfit[rind0], template, 30)

        fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2, dpi=300, figsize=(7,6))

        im_ = ax1.imshow(timepoint_corr_rfit)
        ax1.set_title('Pairwise correlation of radius fits (first 60 sec)')
        ax1.set_xlabel('frames')
        ax1.set_ylabel('frames')

        ax2.plot(template)
        ax2.set_title('Radial fit template (conv)')
        ax2.set_xlabel('radial distance (deg)')
        ax2.set_ylabel('distance to edge (a.u.)')

        ax3.plot(template_rfit_cc_lags, template_rfitconv_cc)
        ax3.set_xlabel('radial distance (deg)')
        ax3.set_ylabel('xcorr')
        ax3.set_title('Template cross correlation with frame {}'.format(rind0))

        rindp0, rindp1 = sorted([rind0,rind1])
        ax4.plot(rfit2times_lags, rfit2times_cc)
        ax4.set_title('xcorr of frames {},{}'.format(rindp0, rindp1))
        ax4.set_xlabel('radial distance (deg)')
        ax4.set_ylabel('xcorr')

        fig.tight_layout()

        num_demo_cells = 25
        ind2plot_rfit = sorted(np.random.randint(0, np.size(pupil_update,0), num_demo_cells))

        # Iterative fit to alignment. Start with mean as template. On each of 12 iterations,
        # shift individual frames to maximize cross-correlation with the template. Then,
        # recalculate mean template.

        fig1, axs1 = plt.subplots(6,2, dpi=300, figsize=(6,11))
        fig2, axs2 = plt.subplots(6,2, dpi=300, figsize=(6,11))

        print('Shifting each frame to maximize xcorr with template.\nTemplate is recalculated between each of 12 iterations.')
        num_iter = 12
        for rep in tqdm(range(num_iter)):

            # For each frame, get correlation, and shift
            for f in range(np.size(pupil_update,0)):

                try:
                    # Calc xcorr between frame's convolved rfit and current template
                    xc, lags = fm2p.nanxcorr(template, pupil_update[f,:], 20)

                    c[f] = np.amax(xc)
                    peaklag = np.argmax(xc)
                    # Find the shift distance that maximizes correlation between frame and template
                    peak[f] = lags[peaklag]
                    # Update cumulative total of shift distance across iterations (for this single frame)
                    total_shift[f] = total_shift[f] + peak[f]
                    # Apply the shift
                    pupil_update[f,:] = np.roll(pupil_update[f,:], int(peak[f]))

                except ZeroDivisionError:
                    total_shift[f] = np.nan
                    pupil_update[f,:] = np.zeros(np.size(pupil_update,1)) * np.nan

            # Update template
            template = np.nanmean(pupil_update, axis=0)

            if rep<=5:
                ax1 = axs1[rep,0]
                ax2 = axs1[rep,1]
            elif rep>5:
                ax1 = axs2[rep-6,0]
                ax2 = axs2[rep-6,1]
            
            # Plot template with pupil_update for each iteration of fit
            ax1.set_title('iter={}/{}'.format(rep+1,num_iter))
            ax1.plot(pupil_update[ind2plot_rfit,:].T, alpha=0.2)
            ax1.plot(template, 'k-', alpha=0.8)

            # Histogram of correlations
            ax2.hist(c[c>0], bins=30, color='k') # gets rid of NaNs in plot
            ax2.set_xlabel('xcorr')

        fig1.tight_layout()
        fig2.tight_layout()
        plt.show()

        # Invert total shift so that it is a measure of the pupil's shift rather than a
        # measure of the shift applied to reach the template
        shift_nan = -total_shift.copy() # shift in degrees

        # Only shift when correlation was high (prev. was c<0.35)
        shift_nan[c < 0.25] = np.nan

        shift_nan = shift_nan - np.nanmedian(shift_nan)

        # Get rid of very large shifts
        shift_nan[np.abs(shift_nan) >= 30] = np.nan

        # Median filter to get rid of outliers
        shift_smooth = scipy.signal.medfilt(shift_nan, 3)

        # Convolve to smooth and fill in nans. Need radial convolution, since it's 0 to
        # 360 degrees around the pupil
        win = 5
        shift_smooth = astropy.convolution.convolve(shift_nan, np.ones(win)/win)
        shift_smooth = shift_smooth - np.nanmedian(shift_smooth)


        fig, [ax0,ax1,ax3] = plt.subplots(3,1, figsize=(8.5,6), dpi=300)

        ax0.plot(eyeT[:900], -total_shift[:900], 'k', alpha=0.3, label='raw')
        ax0.plot(eyeT[:900], shift_smooth[:900], color='tab:blue', label='smoothed')
        ax0.set_xlim([0,15])
        ax0.set_ylabel('shift (deg)')
        ax0.set_xlabel('time (sec)')
        ax0.legend()
        ax0.set_ylim([-50,50])

        ax1.plot(eyeT[:3600], shift_smooth[:3600], 'k')
        ax1.set_ylabel('shift (deg)')
        ax1.set_xlabel('time (sec)')
        ax1.set_title('smoothed cyclotorsion')
        ax1.set_xlim([0,60])
        ax1.set_ylim([-50,50])

        for f in ind2plot_rfit:
            ax3.plot(np.arange(360), all_conv_rfit[frame_inds][f,:].T, alpha=0.2)
        ax3.plot(np.arange(360), template, 'k--', alpha=0.8)
        ax3.set_ylabel('rfit distance (a.u.)')
        ax3.set_xlabel('radial distance (deg)')
        ax3.set_title('convolved radius fit (n={} cells)'.format(num_demo_cells))
        ax3.set_xlim([0,360])
        ax3.set_ylim([-4,4])

        fig.tight_layout()


        fig, axs = plt.subplots(5,2, dpi=300, figsize=(7,9))
        axs = axs.ravel()

        # Get random frames to plot
        rand_frames = sorted(np.random.randint(frame_inds[0], frame_inds[-1]-1, 10))

        for i, f in enumerate(rand_frames):
            axs[i].plot(np.arange(len(rad_range)), template, 'k--', alpha=0.5)
            axs[i].plot(np.arange(len(rad_range)), all_conv_rfit[f])
            axs[i].set_title('f={} c={:.2} omega={:.4}'.format(f,c[f-frame_inds[0]], shift_smooth[f-frame_inds[0]]))
            axs[i].set_xlim([0,360])
            axs[i].set_ylim([-5,5])
            axs[i].set_xlabel('radial distance (deg)')
            axs[i].set_xlabel('rfit distance (a.u.)')
        fig.suptitle('convolved radius fit')
        fig.tight_layout()


        if doVideo:
            # Write diagnostic video
            vidsavepath = 'pupil_rotation_test_2.avi'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')

            vidout = cv2.VideoWriter(
                vidsavepath,
                fourcc,
                60.0,
                (set_size[0]*2, set_size[1])
            )

            for i, f in tqdm(enumerate(frame_inds)):

                frame = cv2.cvtColor(eyevid[f,:,:].copy(), cv2.COLOR_GRAY2BGR)
                annotated_frame = cv2.cvtColor(eyevid[f,:,:].copy(), cv2.COLOR_GRAY2BGR)

                current_longaxis = longaxis[f]
                current_shortaxis = shortaxis[f]
                current_centX = centX[f]
                current_centY = centY[f]

                # Plot the ellipse edge
                rmin = 0.5 * (current_longaxis + current_shortaxis) - ranger
                
                for deg_th, rad_th in enumerate(rad_range):

                    edge_x = np.round(current_centX + (rmin + all_raw_rfit[f,deg_th]) * np.cos(rad_th))
                    edge_y = np.round(current_centY + (rmin + all_raw_rfit[f,deg_th]) * np.sin(rad_th))

                    if pd.isnull(edge_x) is False and pd.isnull(edge_y) is False:
                        annotated_frame = cv2.circle(annotated_frame, (int(edge_x),int(edge_y)), 1, (50,168,58), thickness=-1)

                # Plot the rotation of the eye as a horizontal line that spans across the pupil made up
                # of 100 circles.
                for d in np.linspace(-0.5, 0.5, 100):

                    rot_x = np.round(current_centX + d * (np.rad2deg(np.cos(np.deg2rad(shift_smooth[i])))))
                    rot_y = np.round(current_centY + d * (np.rad2deg(np.sin(np.deg2rad(shift_smooth[i])))))

                    if pd.isnull(rot_x) is False and pd.isnull(rot_y) is False:
                        annotated_frame = cv2.circle(annotated_frame, (int(rot_x), int(rot_y)), 1, (227,32,59), thickness=-1)

                # Plot the center of the eye on the frame as a larger dot than the others
                if pd.isnull(current_centX) is False and pd.isnull(current_centY) is False:

                    annotated_frame = cv2.circle(annotated_frame, (int(current_centX), int(current_centY)), 3, (43,52,227), thickness=-1)

                # What was the point of this? Does it do anything important?
                frame_out = np.concatenate([frame, annotated_frame], axis=1)

                vidout.write(frame_out)

            vidout.release()

        cyclotorsion_dict = {
            'cyclotorsion_shift': shift_smooth,
            'cyclotorsion_raw_total_shift': -total_shift.copy(),
            'cyclotorsion_final_template': template,
            'cyclotorsion_raw_rfit': all_raw_rfit,
            'cyclotorsion_conv_rfit': all_conv_rfit
        }

        return cyclotorsion_dict


def plot_pupil_ellipse_video(video_path, ellipse_dict, xyl_df, savepath,
                             startframe = 3600, maxframes=10000, thresh=0.85):
        """ Plot video of eye tracking.
        """

        vidread = cv2.VideoCapture(video_path)
        width = int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_vid = cv2.VideoWriter(savepath, fourcc, 60.0, (width, height))

        # Only do the first number of frames (limit of frames to use should
        # be set in cfg dict)
        nFrames = int(vidread.get(cv2.CAP_PROP_FRAME_COUNT))
        if maxframes > nFrames:
            num_save_frames = nFrames
        else:
            num_save_frames = maxframes

        # set starting frame
        vidread.set(cv2.CAP_PROP_POS_FRAMES, startframe)

        for f in tqdm(range(startframe, startframe+num_save_frames)):
            
            # Read frame and make sure it's read in correctly
            ret, frame = vidread.read()
            if not ret:
                break

            
            # first, visualize the ellipse fit
            try:
                # Get out ellipse long/short axes and put into tuple
                ellipse_axes = (
                    int(ellipse_dict['longaxis'][f]),
                    int(ellipse_dict['shortaxis'][f])
                )

                # Get out ellipse phi and round to int
                # Note: this is ellipse_phi not phi
                ellipse_phi = int(ellipse_dict['ellipse_phi'][f])
                
                ellipse_cent = (
                    int(ellipse_dict['X0'][f]),
                    int(ellipse_dict['Y0'][f])
                )
                
                # Update this frame with an ellipse
                # ellipse plotted in blue
                frame = cv2.ellipse(
                    frame,
                    ellipse_cent,
                    ellipse_axes,
                    ellipse_phi,
                    0,
                    360,
                    (255,0,0),
                    2
                )
            
            # Skip if the ell data from this frame are bad
            except (ValueError, KeyError):
                pass

            # then add the DLC points
            try:
                # iterate through each point in the list
                for k in range(0, int(xyl_df.shape[1]), 3):

                    # get the point center of each point num, k
                    pt_cent = (
                        int(xyl_df.iloc[f,k]),
                        int(xyl_df.iloc[f,k+1])
                    )

                    if xyl_df.iloc[f,k+2] < thresh:
                        # bad points in red
                        frame = cv2.circle(frame, pt_cent, 3, (0,0,255), -1)
                    
                    elif xyl_df.iloc[f,k+2] >= thresh:
                        # good points in green
                        frame = cv2.circle(frame, pt_cent, 3, (0,255,0), -1)
                
            except (ValueError, KeyError):
                pass

            out_vid.write(frame)
        out_vid.release()



if __name__ == '__main__':
    
    basepath = r'K:\FreelyMovingEyecams\241204_DMM_DMM031_freelymoving'
    rec_name = '241204_DMM_DMM031_freelymoving_01'

    reye = fm2p.Eyecam(basepath, rec_name)
    reye.find_files()
    ellipse_fit_results = reye.track_pupil()

