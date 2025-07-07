# -*- coding: utf-8 -*-
"""
Utility functions for aligning eyecam data using TTL pulses.

Functions
---------
align_eyecam_using_TTL(eye_dlc_h5, eye_TS_csv, eye_TTLV_csv, eye_TTLTS_csv, quiet=True)
    Align eyecam data using TTL pulses.

Author: DMM, last modified May 2025
"""


import numpy as np
import pandas as pd

import fm2p


def align_eyecam_using_TTL(eye_dlc_h5, eye_TS_csv, eye_TTLV_csv, eye_TTLTS_csv, theta, quiet=True):
    """ Align eyecam data using TTL pulses.

    Parameters
    ----------
    eye_dlc_h5 : str
        Path to the DLC h5 file containing eyecam data.
    eye_TS_csv : str
        Path to the CSV file containing eyecam timestamps.
    eye_TTLV_csv : str
        Path to the CSV file containing TTL voltages.
    eye_TTLTS_csv : str
        Path to the CSV file containing TTL timestamps.
    quiet : bool, optional
        If True, suppress print statements. Default is True.

    Returns
    -------
    eyeStart : int
        Start index of the aligned eyecam data.
    eyeEnd : int
        End index of the aligned eyecam data.
    """

    # Read in the DLC data
    pts, _ = fm2p.open_dlc_h5(eye_dlc_h5)
    num_frames = pts['t_x'].size

    # Read in the timestamps for each video frame
    eyeT = fm2p.read_timestamp_file(eye_TS_csv, position_data_length=num_frames)

    # Read in the TTL voltages
    ttlV = pd.read_csv(eye_TTLV_csv, encoding='utf-8', engine='c', header=None).squeeze().to_numpy()

    # Read in the timestamps for each TTL voltage reading
    ttlT_series = pd.read_csv(eye_TTLTS_csv, encoding='utf-8', engine='c', header=None).squeeze()
    ttlT = fm2p.read_timestamp_series(ttlT_series)

    if len(ttlV) != len(ttlT):
        print('Warning! Length of TTL voltages ({}) does not match the length of TTL timestamps ({}).'.format(len(ttlV), len(ttlT)))

    # Get start and stop index from TTL data
    startInd = int(np.argwhere(ttlV>0)[0])
    endInd = int(np.argwhere(ttlV>0)[-1])

    # Get the first and last video frame for which enough points (probably 7, depending on the config
    # file options) were tracked to fit an ellipse to the pupil.
    # reye_cam = fm2p.Eyecam('', '', cfg)
    # reye_cam.add_files(
    #     eye_dlc_h5=eye_dlc_h5,
    #     eye_avi='',
    #     eyeT=eye_TS_csv
    # )
    # eye_xyl, ellipse_dict = reye_cam.track_pupil()
    # # Use theta as the measure of this, but using other params (e.g., phi, centroid) would be equivilent
    # theta = ellipse_dict['theta']
    
    firstTheta = int(np.argwhere(~np.isnan(theta))[0])
    lastTheta = int(np.argwhere(~np.isnan(theta))[-1])

    if not quiet:
        print('Theta: ', eyeT[firstTheta], ' to ', eyeT[lastTheta])
        print('TTL: ', ttlT[startInd], ' to ', ttlT[endInd])

    # Use the TTL timestamps to get the onset
    apply_t0 = ttlT[startInd]
    apply_tEnd = ttlT[endInd]

    # Find the closest timestamps in the eyecam data to the TTL timestamps
    eyeStart, _ = fm2p.find_closest_timestamp(eyeT, apply_t0)
    eyeEnd, _ = fm2p.find_closest_timestamp(eyeT, apply_tEnd)

    return eyeStart, eyeEnd



def align_lightdark_using_TTL(ltdk_TTL_path, ltdk_TS_path, eyeT, twopT, eyeStart, eyeEnd):
    # this needs to be the version of eyeT that is NOT already cropped to the
    # start/end of the recording and NOT with T0 subtracted (still in
    # absoluite time).

    # lt/dk TTL
    ltdkV = pd.read_csv(
        ltdk_TTL_path,
        encoding='utf-8',
        engine='c',
        header=None
    ).squeeze().to_numpy()

    ltdkT_series = pd.read_csv(
        ltdk_TS_path,
        encoding='utf-8',
        engine='c',
        header=None
    ).squeeze()
    ltdkT = fm2p.read_timestamp_series(ltdkT_series)

    light_onsets = np.diff(ltdkV) > np.nanmean(ltdkV)
    dark_onsets = np.diff(ltdkV) < -np.nanmean(ltdkV)

    # find closest eyecam timestamp to TTL edge
    eyet_light_onset_times = [fm2p.find_closest_timestamp(eyeT[eyeStart:eyeEnd], t)[1] for t in ltdkT[:-1][light_onsets]]
    eyet_dark_onset_times = [fm2p.find_closest_timestamp(eyeT[eyeStart:eyeEnd], t)[1] for t in ltdkT[:-1][dark_onsets]]

    # find the corresponding topdown
    t0 = eyeT[eyeStart]
    twopInds_light_onsets = np.array([fm2p.find_closest_timestamp(twopT, t-t0)[0] for t in eyet_light_onset_times])
    twopInds_dark_onsets = np.array([fm2p.find_closest_timestamp(twopT, t-t0)[0] for t in eyet_dark_onset_times])

    # true when lights are on, otherwise false
    light_state_vec = np.zeros(len(twopT), dtype=bool)
    for ind in range(len(twopT)):
        
        # last light onset that already happened
        last_onset = twopInds_light_onsets[twopInds_light_onsets<ind]
        last_offset = twopInds_dark_onsets[twopInds_dark_onsets<ind]

        # if there has been both a rising and falling edge already
        if (len(last_offset)>0) and (len(last_onset)>0):
            last_onset = last_onset[-1]
            last_offset = last_offset[-1]
            # most recent change was lights turning on
            if last_onset > last_offset:
                light_state_vec[ind] = True
            # or, most recent change was lights turning off
            elif last_onset < last_offset:
                light_state_vec[ind] = False

        # if there has been a falling edge but no rising edge yet
        elif (len(last_onset)==0) and (len(last_offset)>0):
            light_state_vec[ind] = False

        # there has been a rising edge but no falling edge yet
        elif (len(last_onset)>0) and (len(last_offset)==0):
            light_state_vec[ind] = True

        # There has been on rising or falling edge yet. In this case, check
        # to see which will come next.
        elif (len(last_onset)==0) and (len(last_offset)==0):
            if twopInds_light_onsets[0] < twopInds_dark_onsets[0]:
                light_state_vec[ind] = True
            elif twopInds_light_onsets[0] > twopInds_dark_onsets[0]:
                light_state_vec[ind] = False

    return light_state_vec, twopInds_light_onsets, twopInds_dark_onsets




