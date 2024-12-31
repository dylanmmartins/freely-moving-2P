"""
fm2p/utils/eyecam.py
Eye tracking class

DMM, 2024
"""


import os
import numpy as np
import scipy.stats
import scipy.signal
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

import fm2p


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
            Optional. Dictionary of config options. If not provided,
            default values will be used.
        """

        self.recording_path = recording_path
        self.recording_name = recording_name

        if cfg is None:
            self.ridge_cyclotorsion = False
            self.likelihood_thresh = 0.99
            self.eye_ellipse_thresh = 0.85
            self.eye_fig_pts_dwnspl = 100
            self.eye_dist_thresh = 4.1
            self.eye_pxl2cm = 24
            self.eye_radius_thresh = 50
            self.eye_calibration_N = 8
            self.eye_trackable_N = 7
        elif cfg is not None:
            self.ridge_cyclotorsion = cfg['ridge_cyclotorsion']
            self.likelihood_thresh = cfg['likelihood_thresh']
            self.eye_ellipse_thresh = cfg['eye_ellipse_thresh']
            self.eye_fig_pts_dwnspl = cfg['eye_fig_pts_dwnspl']
            self.eye_dist_thresh = cfg['eye_dist_thresh']
            self.eye_pxl2cm = cfg['eye_pxl2cm']
            self.eye_radius_thresh = cfg['eye_radius_thresh']
            self.eye_calibration_N = cfg['eye_calibration_N']
            self.eye_trackable_N = cfg['eye_trackable_N']

    def find_files(self):
        """ Gather files.
        """

        self.eye_dlc_h5 = fm2p.find('{}*eye_deinterDLC_resnet50*.h5'.format(self.recording_name), self.recording_path, MR=True)
        self.eye_avi = fm2p.find('{}*eye_deinter.avi'.format(self.recording_name), self.recording_path, MR=True)
        self.eyeT_csv = fm2p.find('{}*_eye.csv'.format(self.recording_name), self.recording_path, MR=True)


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

        angle_from_x = angle_to_x
        orientation_rad = 0.5 * np.arctan2(b, (c-a))
        cos_phi = np.cos(orientation_rad)
        sin_phi = np.sin(orientation_rad)

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

        # Set up the pdf to be saved out with diagnostic figures
        pdf_name = '{}_eye_tracking_figs.pdf'.format(self.recording_name)
        pdf = PdfPages(os.path.join(self.recording_path, pdf_name))

        # read deeplabcut file
        xyl, _ = fm2p.open_dlc_h5(self.eye_dlc_h5)
        x_vals, y_vals, likelihood = fm2p.split_xyl(xyl)

        # Threshold by likelihoods
        x_vals = fm2p.apply_liklihood_thresh(x_vals, likelihood)
        y_vals = fm2p.apply_liklihood_thresh(y_vals, likelihood)

        # Threshold by number of sucessfully tracked pupil points
        pupil_count = np.sum(likelihood >= self.likelihood_thresh, 1)
        usegood_eye = pupil_count >= self.eye_trackable_N
        usegood_eyecalib = pupil_count >= self.eye_calibration_N

        # Threshold out pts more than a given distance away from nanmean of that point
        std_thresh_x = np.empty(np.shape(x_vals))
        std_thresh_y = np.empty(np.shape(y_vals))

        for point_loc in range(0,np.size(x_vals, 1)):
            _val = x_vals.iloc[:,point_loc]
            std_thresh_x[:,point_loc] = (np.abs(np.nanmean(_val) - _val) / self.eye_pxl2cm) > self.eye_dist_thresh

        for point_loc in range(0,np.size(x_vals, 1)):
            _val = y_vals.iloc[:,point_loc]
            std_thresh_y[:,point_loc] = (np.abs(np.nanmean(_val) - _val) / self.eye_pxl2cm) > self.eye_dist_thresh

        std_thresh_x = np.nanmean(std_thresh_x, 1)
        std_thresh_y = np.nanmean(std_thresh_y, 1)

        x_vals[std_thresh_x > 0] = np.nan
        y_vals[std_thresh_y > 0] = np.nan

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
        
        # List of all places where the ellipse meets threshold
        # R = np.linspace(0, 2*np.pi, 100)

        # (short axis / long axis) < thresh
        usegood_ellipcalb = np.where((usegood_eyecalib == True)                     \
                & ((ellipse[:,6] / ellipse[:,5]) < self.eye_ellipse_thresh))
        
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
            ax5.set_title('ellipticity; thresh='+str(self.eye_ellipse_thresh))
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

        # Save out camera center and scale as np array (but only if this is
        # a freely moving recording).
            
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
    
    
    def save_tracking(self, ellipse_dict, dlc_xyl, vid_array):
        """ Save eye tracking data out.

        Parameters
        ----------
        ellipse_dict : dict
            Dictionary of ellipse fit values returned from track_pupil().
        dlc_xyl : pd.DataFrame
            X, y, and likelihoods as a dataframe. Each row is a camera frame.
        vid_array : np.array
            Numpy array of the video data, where shape is (time, height, width).
        """

        xyl_dict = dlc_xyl.to_dict()
        vid_dict = {'video': vid_array}

        save_dict = {**xyl_dict, **ellipse_dict, **vid_dict}

        _savepath = os.path.join(self.recording_path, '{}_eye_tracking.h5'.format(self.recording_name))
        fm2p.write_h5(_savepath, save_dict)



if __name__ == '__main__':
    
    basepath = r'K:\FreelyMovingEyecams\241204_DMM_DMM031_freelymoving'
    rec_name = '241204_DMM_DMM031_freelymoving_01'
    reye = fm2p.Eyecam(basepath, rec_name)
    reye.find_files()
    ellipse_fit_results = reye.track_pupil()

    