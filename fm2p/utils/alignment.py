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

from .time import find_closest_timestamp, read_timestamp_file, read_timestamp_series, interpT
from .files import open_dlc_h5


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

    if eye_dlc_h5 is not None:
        pts, _ = open_dlc_h5(eye_dlc_h5)
        num_frames = pts['t_x'].size
    else:
        num_frames = None

    eyeT = read_timestamp_file(eye_TS_csv, position_data_length=num_frames)

    ttlV = pd.read_csv(eye_TTLV_csv, encoding='utf-8', engine='c', header=None).squeeze().to_numpy()

    # read in the timestamps for each TTL voltage reading
    ttlT_series = pd.read_csv(eye_TTLTS_csv, encoding='utf-8', engine='c', header=None).squeeze()
    ttlT = read_timestamp_series(ttlT_series)

    if len(ttlV) != len(ttlT):
        print('Warning! Length of TTL voltages ({}) does not match the length of TTL timestamps ({}).'.format(len(ttlV), len(ttlT)))

    # get start and stop index from TTL data
    startInd = int(np.argwhere(ttlV>0)[0])
    endInd = int(np.argwhere(ttlV>0)[-1])

    if theta is not None:
        firstTheta = int(np.argwhere(~np.isnan(theta))[0])
        lastTheta = int(np.argwhere(~np.isnan(theta))[-1])

        if not quiet:
            print('Theta: ', eyeT[firstTheta], ' to ', eyeT[lastTheta])
            print('TTL: ', ttlT[startInd], ' to ', ttlT[endInd])

    # use the TTL timestamps to get the onset
    apply_t0 = ttlT[startInd]
    apply_tEnd = ttlT[endInd]

    # find the closest timestamps in the eyecam data to the TTL timestamps
    eyeStart, _ = find_closest_timestamp(eyeT, apply_t0)
    eyeEnd, _ = find_closest_timestamp(eyeT, apply_tEnd)

    return eyeStart, eyeEnd, apply_t0, apply_tEnd



def align_lightdark_using_TTL(ltdk_TTL_path, ltdk_TS_path, eyeT, twopT, eyeStart, eyeEnd):
    """
    Docstring for align_lightdark_using_TTL
    
    :param ltdk_TTL_path: Description
    :param ltdk_TS_path: Description
    :param eyeT: Description
    :param twopT: Description
    :param eyeStart: Description
    :param eyeEnd: Description
    """

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
    ltdkT = read_timestamp_series(ltdkT_series)

    light_onsets = np.diff(ltdkV) > np.nanmean(ltdkV)
    dark_onsets = np.diff(ltdkV) < -np.nanmean(ltdkV)

    # find closest eyecam timestamp to TTL edge
    try:
        eyet_light_onset_times = [find_closest_timestamp(eyeT[eyeStart:eyeEnd], t)[1] for t in ltdkT[:-1][light_onsets]]
        eyet_dark_onset_times = [find_closest_timestamp(eyeT[eyeStart:eyeEnd], t)[1] for t in ltdkT[:-1][dark_onsets]]
    except IndexError:
        try:
            eyet_light_onset_times = [find_closest_timestamp(eyeT[eyeStart:eyeEnd], t)[1] for t in ltdkT[light_onsets[:-1]]]
            eyet_dark_onset_times = [find_closest_timestamp(eyeT[eyeStart:eyeEnd], t)[1] for t in ltdkT[dark_onsets[:-1]]]
        except IndexError: # this is a bad solution. could rewrite as while loop but that seems dangerous too
            eyet_light_onset_times = [find_closest_timestamp(eyeT[eyeStart:eyeEnd], t)[1] for t in ltdkT[light_onsets[:-2]]]
            eyet_dark_onset_times = [find_closest_timestamp(eyeT[eyeStart:eyeEnd], t)[1] for t in ltdkT[dark_onsets[:-2]]]

    # find the corresponding topdown
    t0 = eyeT[eyeStart]
    twopInds_light_onsets = np.array([find_closest_timestamp(twopT, t-t0)[0] for t in eyet_light_onset_times])
    twopInds_dark_onsets = np.array([find_closest_timestamp(twopT, t-t0)[0] for t in eyet_dark_onset_times])

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

        # There has been no rising or falling edge yet. In this case, check
        # to see which will come next.
        elif (len(last_onset)==0) and (len(last_offset)==0):
            if twopInds_light_onsets[0] < twopInds_dark_onsets[0]:
                light_state_vec[ind] = True
            elif twopInds_light_onsets[0] > twopInds_dark_onsets[0]:
                light_state_vec[ind] = False

    return light_state_vec, twopInds_light_onsets, twopInds_dark_onsets


def align_crop_IMU(df, imuT, apply_t0, apply_tEnd, eyeT, twopT):
    """
    """

    outputs = {}

    imuStart, _ = find_closest_timestamp(imuT, apply_t0)
    imuEnd, _ = find_closest_timestamp(imuT, apply_tEnd)

    keys = df.keys()

    for k in keys:

        v_eye_interp = interpT(
            df[k].iloc[imuStart:imuEnd].to_numpy(),
            imuT[imuStart:imuEnd] - imuT[imuStart],
            eyeT - eyeT[0]
        )

        v_twop_interp = interpT(
            df[k].iloc[imuStart:imuEnd].to_numpy(),
            imuT[imuStart:imuEnd] - imuT[imuStart],
            twopT
        )

        outputs['{}_raw'.format(k)] = df[k].to_numpy()
        outputs['{}_trim'.format(k)] = df[k].iloc[imuStart:imuEnd].to_numpy()
        outputs['{}_eye_interp'.format(k)] = v_eye_interp
        outputs['{}_twop_interp'.format(k)] = v_twop_interp

    outputs['imuT_raw'] = imuT
    trim_IMU_time = imuT[imuStart:imuEnd]
    outputs['imuT_trim'] = trim_IMU_time - trim_IMU_time[0]

    return outputs