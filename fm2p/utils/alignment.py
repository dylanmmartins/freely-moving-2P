# -*- coding: utf-8 -*-
"""
fm2p/utils/alignment.py

Functions for aligning eyecam, IMU, and light-dark stimulus data to a common
2-photon timeline using TTL pulse signals.

Functions
---------
align_eyecam_using_TTL
    Find eyecam frame window that overlaps the TTL-on period.
align_lightdark_using_TTL
    Build a per-frame boolean vector marking light vs. dark epochs.
align_crop_IMU
    Crop IMU channels to the TTL window and interpolate onto eye/2p timelines.


DMM, February 2025
"""

import numpy as np
import pandas as pd

from .time import find_closest_timestamp, read_timestamp_file, read_timestamp_series, interpT
from .files import open_dlc_h5


def align_eyecam_using_TTL(eye_dlc_h5, eye_TS_csv, eye_TTLV_csv, eye_TTLTS_csv, theta, quiet=True):
    """ Find eyecam frame indices that bracket the TTL-on recording window.

    Parameters
    ----------
    eye_dlc_h5 : str or None
        Path to DLC output HDF5; used only to get frame count. Pass None to
        skip length validation.
    eye_TS_csv : str
        Camera timestamp file.
    eye_TTLV_csv : str
        Voltage trace CSV for the TTL signal.
    eye_TTLTS_csv : str
        Timestamp CSV paired with the TTL voltage trace.
    theta : np.ndarray or None
        Pupil angle array; printed for debugging when quiet=False.
    quiet : bool
        Suppress diagnostic prints when True.

    Returns
    -------
    eyeStart : int
        Index into eyeT of the first TTL-high frame.
    eyeEnd : int
        Index into eyeT of the last TTL-high frame.
    apply_t0 : float
        Absolute timestamp of the TTL rising edge.
    apply_tEnd : float
        Absolute timestamp of the TTL falling edge.
    """

    if eye_dlc_h5 is not None:
        pts, _ = open_dlc_h5(eye_dlc_h5)
        num_frames = pts['t_x'].size
    else:
        num_frames = None

    eyeT = read_timestamp_file(eye_TS_csv, position_data_length=num_frames)

    ttlV = pd.read_csv(eye_TTLV_csv, encoding='utf-8', engine='c', header=None).squeeze().to_numpy()

    ttlT_series = pd.read_csv(eye_TTLTS_csv, encoding='utf-8', engine='c', header=None).squeeze()
    ttlT = read_timestamp_series(ttlT_series)

    if len(ttlV) != len(ttlT):
        print('Warning -- TTL voltage length ({}) does not match TTL timestamp length ({}).'.format(
            len(ttlV), len(ttlT)))

    # First and last sample where the TTL is high define the recording window.
    startInd = int(np.argwhere(ttlV > 0)[0])
    endInd = int(np.argwhere(ttlV > 0)[-1])

    if theta is not None:
        firstTheta = int(np.argwhere(~np.isnan(theta))[0])
        lastTheta = int(np.argwhere(~np.isnan(theta))[-1])

        if not quiet:
            print('Theta: {} to {}'.format(eyeT[firstTheta], eyeT[lastTheta]))
            print('TTL:   {} to {}'.format(ttlT[startInd], ttlT[endInd]))

    apply_t0 = ttlT[startInd]
    apply_tEnd = ttlT[endInd]

    eyeStart, _ = find_closest_timestamp(eyeT, apply_t0)
    eyeEnd, _ = find_closest_timestamp(eyeT, apply_tEnd)

    return eyeStart, eyeEnd, apply_t0, apply_tEnd


def align_lightdark_using_TTL(ltdk_TTL_path, ltdk_TS_path, eyeT, twopT, eyeStart, eyeEnd):
    """ Build a boolean vector over 2p frames marking whether lights were on.

    Detects light and dark onsets from threshold crossings in the light/dark
    TTL voltage trace, maps them onto the 2p timeline, then walks forward in
    time to assign a light state to each frame.

    Parameters
    ----------
    ltdk_TTL_path : str
        CSV with light/dark TTL voltage samples.
    ltdk_TS_path : str
        CSV with timestamps paired to the TTL voltage samples.
    eyeT : np.ndarray
        Eyecam timestamps (absolute).
    twopT : np.ndarray
        2-photon frame timestamps (relative to recording start).
    eyeStart : int
        First valid eyecam frame index (from TTL alignment).
    eyeEnd : int
        Last valid eyecam frame index (from TTL alignment).

    Returns
    -------
    light_state_vec : np.ndarray of bool
        True for each 2p frame that falls inside a lit epoch.
    twopInds_light_onsets : np.ndarray
        2p frame indices of light-on events.
    twopInds_dark_onsets : np.ndarray
        2p frame indices of light-off events.
    """

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

    # Threshold crossings relative to the mean signal level.
    light_onsets = np.diff(ltdkV) > np.nanmean(ltdkV)
    dark_onsets = np.diff(ltdkV) < -np.nanmean(ltdkV)

    # np.diff shortens the array by 1; we try three indexing variants to
    # handle off-by-one edge cases between the voltage and timestamp arrays.
    try:
        eyet_light_onset_times = [find_closest_timestamp(eyeT[eyeStart:eyeEnd], t)[1] for t in ltdkT[:-1][light_onsets]]
        eyet_dark_onset_times = [find_closest_timestamp(eyeT[eyeStart:eyeEnd], t)[1] for t in ltdkT[:-1][dark_onsets]]
    except IndexError:
        try:
            eyet_light_onset_times = [find_closest_timestamp(eyeT[eyeStart:eyeEnd], t)[1] for t in ltdkT[light_onsets[:-1]]]
            eyet_dark_onset_times = [find_closest_timestamp(eyeT[eyeStart:eyeEnd], t)[1] for t in ltdkT[dark_onsets[:-1]]]
        except IndexError:
            eyet_light_onset_times = [find_closest_timestamp(eyeT[eyeStart:eyeEnd], t)[1] for t in ltdkT[light_onsets[:-2]]]
            eyet_dark_onset_times = [find_closest_timestamp(eyeT[eyeStart:eyeEnd], t)[1] for t in ltdkT[dark_onsets[:-2]]]

    # Eye timestamps are absolute; subtract t0 so they match twopT (relative).
    t0 = eyeT[eyeStart]
    twopInds_light_onsets = np.array([find_closest_timestamp(twopT, t - t0)[0] for t in eyet_light_onset_times])
    twopInds_dark_onsets = np.array([find_closest_timestamp(twopT, t - t0)[0] for t in eyet_dark_onset_times])

    light_state_vec = np.zeros(len(twopT), dtype=bool)
    for ind in range(len(twopT)):

        last_onset = twopInds_light_onsets[twopInds_light_onsets < ind]
        last_offset = twopInds_dark_onsets[twopInds_dark_onsets < ind]

        # Both a rising and falling edge have occurred -- most recent wins.
        if (len(last_offset) > 0) and (len(last_onset) > 0):
            last_onset = last_onset[-1]
            last_offset = last_offset[-1]
            if last_onset > last_offset:
                light_state_vec[ind] = True
            elif last_onset < last_offset:
                light_state_vec[ind] = False

        # Falling edge seen but no rising edge yet -- session started dark.
        elif (len(last_onset) == 0) and (len(last_offset) > 0):
            light_state_vec[ind] = False

        # Rising edge seen but no falling edge yet -- still in first lit period.
        elif (len(last_onset) > 0) and (len(last_offset) == 0):
            light_state_vec[ind] = True

        # No edges yet -- infer initial state from which event comes first overall.
        elif (len(last_onset) == 0) and (len(last_offset) == 0):
            if twopInds_light_onsets[0] < twopInds_dark_onsets[0]:
                light_state_vec[ind] = True
            elif twopInds_light_onsets[0] > twopInds_dark_onsets[0]:
                light_state_vec[ind] = False

    return light_state_vec, twopInds_light_onsets, twopInds_dark_onsets


def align_crop_IMU(df, imuT, apply_t0, apply_tEnd, eyeT, twopT):
    """ Crop IMU channels to the TTL window and interpolate onto two timelines.

    For each channel in df, produces four variants: raw, trimmed to the TTL
    window, interpolated to eyecam timestamps, and interpolated to 2p frame
    timestamps.

    Parameters
    ----------
    df : pd.DataFrame
        IMU data; each column is a channel (e.g. pitch, roll, yaw).
    imuT : np.ndarray
        IMU sample timestamps (absolute).
    apply_t0 : float
        Start of the TTL window (absolute timestamp).
    apply_tEnd : float
        End of the TTL window (absolute timestamp).
    eyeT : np.ndarray
        Eyecam timestamps (absolute).
    twopT : np.ndarray
        2-photon frame timestamps (relative to recording start).

    Returns
    -------
    outputs : dict
        Keys per channel: '<k>_raw', '<k>_trim', '<k>_eye_interp',
        '<k>_twop_interp'. Also contains 'imuT_raw' and 'imuT_trim'.
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
    # Trim time reset to 0 so it matches the relative twopT convention.
    outputs['imuT_trim'] = trim_IMU_time - trim_IMU_time[0]

    return outputs
