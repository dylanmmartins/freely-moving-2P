# -*- coding: utf-8 -*-
"""
fm2p/utils/ref_frame.py

Reference frame calculations for freely moving behavior.

Converts allocentric coordinates (head yaw, target positions) into
egocentric and retinocentric angles relative to the animal's head or gaze.

See MEMORY.md for the gaze direction sign convention used throughout this module:
  gaze = head_yaw_deg + (ang_offset - theta)

Functions
---------
visual_angle_degrees
    Convert object size and distance to visual angle.
angle_to_target
    Egocentric angle from a given position and heading to a target.
calc_reference_frames
    Compute egocentric, retinocentric, and distance frames for all frames.
calc_vor_eye_offset
    Estimate the camera angular offset (ang_offset) using VOR-based regressions.
get_ang_offset
    Retrieve or compute ang_offset from a preprocessed data dict.


DMM, March 2025
"""

import math
import numpy as np
from scipy import stats


def visual_angle_degrees(distance_cm, size_cm=4.):
    """ Visual angle subtended by an object of size_cm at distance_cm.

    Parameters
    ----------
    distance_cm : float
        Distance from observer to object centre (cm).
    size_cm : float
        Diameter/height of the object (cm). Default 4 cm (pillar diameter).

    Returns
    -------
    float
        Visual angle in degrees.
    """

    angle_radians = 2 * math.atan(size_cm / (2 * distance_cm))
    return math.degrees(angle_radians)


def angle_to_target(x_c, y_c, heading, x_t, y_t):
    """ Absolute and egocentric angle from a position/heading to a target.

    Parameters
    ----------
    x_c : float
        Observer x position.
    y_c : float
        Observer y position.
    heading : float
        Current heading in degrees (allocentric, 0 = east).
    x_t : float
        Target x position.
    y_t : float
        Target y position.

    Returns
    -------
    absolute_angle : float
        Allocentric angle to target in [0, 360).
    angle_difference : float
        Signed egocentric angle to target in (-180, 180].
    """

    absolute_angle = math.degrees(math.atan2(y_t - y_c, x_t - x_c))
    absolute_angle = (absolute_angle + 360) % 360

    heading = (heading + 360) % 360

    angle_difference = (absolute_angle - heading + 180) % 360 - 180

    return absolute_angle, angle_difference


def calc_reference_frames(cfg, headx, heady, yaw, theta, arena_dict):
    """ Compute egocentric, retinocentric, and distance frames for all frames.

    Parameters
    ----------
    cfg : dict
        Must contain 'eyecam_angular_offset' (degrees).
    headx : np.ndarray
        Head x positions (pixels).
    heady : np.ndarray
        Head y positions (pixels).
    yaw : np.ndarray
        Allocentric head yaw (degrees).
    theta : np.ndarray
        Pupil angle from camera (degrees); same length as yaw.
    arena_dict : dict
        Must contain 'pillar_centroid' (dict with 'x','y'), corner dicts
        'arenaTL/TR/BL/BR' (each with 'x','y'), and 'pxls2cm'.

    Returns
    -------
    reframe_dict : dict
        'egocentric', 'retinocentric', 'pupil_from_head', 'dist_to_center',
        'dist_to_pillar', 'pillar_size' -- all arrays of length N_frames.
    """

    pillarx = arena_dict['pillar_centroid']['x']
    pillary = arena_dict['pillar_centroid']['y']

    pillar_ego = np.zeros_like(headx) * np.nan
    pillar_abs = np.zeros_like(headx) * np.nan
    pupil_from_head = np.zeros_like(headx) * np.nan
    pillar_retino = np.zeros_like(headx) * np.nan

    for f in range(len(headx)):
        pillar_abs[f], pillar_ego[f] = angle_to_target(headx[f], heady[f], yaw[f], pillarx, pillary)

    if np.size(theta) != np.size(pillar_ego):
        print('Check length of theta vs egocentric angle -- sizes do not match. '
              'Is theta already aligned by TTL and interpolated to 2P timestamps?')
        print('Sizes are theta={}, ego={}'.format(np.size(theta), np.size(pillar_ego)))

    # gaze = head + (ang_offset - theta); see MEMORY.md sign convention.
    ang_offset = cfg['eyecam_angular_offset']
    for f in range(len(headx)):
        pfh = ang_offset - theta[f]
        pupil_from_head[f] = pfh
        pillar_retino[f] = ((((pillar_ego[f] - pfh) + 180) % 360) - 180)

    tlx = arena_dict['arenaTL']['x']
    tly = arena_dict['arenaTL']['y']
    trx = arena_dict['arenaTR']['x']
    try_ = arena_dict['arenaTR']['y']
    blx = arena_dict['arenaBL']['x']
    bly = arena_dict['arenaBL']['y']
    brx = arena_dict['arenaBR']['x']
    bry = arena_dict['arenaBR']['y']

    centx = np.nanmean([(trx - tlx), (brx - blx)])
    centy = np.nanmean([(bry - try_), (bly - tly)])

    dist_to_center = np.array([
        np.sqrt((headx[f] - centx) ** 2 + (heady[f] - centy) ** 2)
        for f in range(len(headx))
    ])
    pxls2cm = arena_dict.get('pxls2cm', 1.0)
    dist_to_pillar_px = np.array([
        np.sqrt((headx[f] - pillarx) ** 2 + (heady[f] - pillary) ** 2)
        for f in range(len(headx))
    ])
    dist_to_pillar_cm = dist_to_pillar_px / pxls2cm
    pillar_size = np.array([
        visual_angle_degrees(dist_to_pillar_cm[f], 4.)
        for f in range(len(headx))
    ])

    reframe_dict = {
        'egocentric': pillar_ego,
        'retinocentric': pillar_retino,
        'pupil_from_head': pupil_from_head,
        'dist_to_center': dist_to_center,
        'dist_to_pillar': dist_to_pillar_cm,
        'pillar_size': pillar_size
    }

    return reframe_dict


def calc_vor_eye_offset(theta_interp, head_yaw_deg, fps, head_vel_deg_s=None):
    """ Estimate the camera angular offset (ang_offset) from VOR statistics.

    Two estimates are returned:
    - vor_null: median theta during near-stationary head frames.
    - vor_regression: median of theta after removing head-velocity-coupled
      component (VOR gain correction via linear regression in velocity space,
      applied as a position-level shift to avoid integration drift).

    Parameters
    ----------
    theta_interp : array-like
        Pupil angle trace at 2P timestamps (degrees).
    head_yaw_deg : array-like or None
        Allocentric head direction (degrees). Used if head_vel_deg_s is None.
    fps : float
        Sampling rate (frames per second).
    head_vel_deg_s : array-like or None
        Pre-computed head angular velocity (deg/s). Gyro preferred over
        differentiated yaw because it avoids 0/360 wrap-around artifacts.

    Returns
    -------
    dict
        'ang_offset_vor_null', 'ang_offset_vor_regression', 'vor_gain'.
    """

    if head_vel_deg_s is not None:
        n = min(len(theta_interp), len(head_vel_deg_s))
    elif head_yaw_deg is not None:
        n = min(len(theta_interp), len(head_yaw_deg))
    else:
        n = len(theta_interp)

    theta = np.array(theta_interp[:n], dtype=float)

    if head_vel_deg_s is not None:
        head_vel = np.array(head_vel_deg_s[:n], dtype=float)
        # Treat NaN frames as zero velocity so cumsum stays continuous.
        hv_filled = np.where(np.isnan(head_vel), 0.0, head_vel)
        head_pos = np.cumsum(hv_filled) / fps  # degrees
        head_delta = head_pos - np.nanmedian(head_pos)
    else:
        head = np.array(head_yaw_deg[:n], dtype=float)
        # Unwrap before differentiating to avoid spikes at the 0/360 boundary.
        head_unwrap = np.rad2deg(np.unwrap(np.deg2rad(head)))
        head_vel = np.gradient(head_unwrap) * fps
        head_delta = head_unwrap - np.nanmedian(head_unwrap)

    eye_vel = np.gradient(theta) * fps
    valid = (~np.isnan(head_vel)) & (~np.isnan(eye_vel)) & (~np.isnan(theta))

    still_thresh = 10.  # deg/s
    still = valid & (np.abs(head_vel) < still_thresh)
    if still.sum() > 10:
        ang_offset_vor_null = float(np.nanmedian(theta[still]))
    else:
        ang_offset_vor_null = float(np.nanmedian(theta[valid]))

    active_thresh = 20.  # deg/s
    active = valid & (np.abs(head_vel) > active_thresh)

    vor_gain = 1.0
    slope = -1.0
    intercept = 0.0

    if active.sum() > 200:
        from scipy import stats as _stats
        slope, intercept, _, _, _ = _stats.linregress(
            head_vel[active], eye_vel[active])
        vor_gain = float(-slope)

    # Remove VOR-coupled component using position-level slope only.
    # Intercept (deg/s) is not applied at position level -- integrating it
    # over a long session would accumulate thousands of degrees of drift.
    theta_corr = theta - slope * head_delta

    if valid.sum() > 10:
        ang_offset_vor_regression = float(np.nanmedian(theta_corr[valid]))
    else:
        ang_offset_vor_regression = ang_offset_vor_null

    return {
        'ang_offset_vor_null': ang_offset_vor_null,
        'ang_offset_vor_regression': ang_offset_vor_regression,
        'vor_gain': vor_gain,
    }


def get_ang_offset(data, fps=None):
    """ Retrieve ang_offset from stored keys or compute it from raw traces.

    Parameters
    ----------
    data : dict
        Preprocessed data dict. Checked for 'ang_offset_vor_regression',
        then 'ang_offset_vor_null', then computed from 'theta_interp' and
        'head_yaw_deg'.
    fps : float or None
        Sampling rate; inferred from 'twopT' if not provided.

    Returns
    -------
    float or None
    """

    if 'ang_offset_vor_regression' in data:
        return float(data['ang_offset_vor_regression'])

    if 'ang_offset_vor_null' in data:
        return float(data['ang_offset_vor_null'])

    theta = data.get('theta_interp', None)
    head = data.get('head_yaw_deg', None)
    if theta is None or head is None:
        return None

    if fps is None:
        twopT = data.get('twopT', None)
        if twopT is not None and len(twopT) > 1:
            fps = float(1.0 / np.nanmedian(np.diff(twopT)))
        else:
            fps = 30.0

    result = calc_vor_eye_offset(theta, head, fps)
    return result['ang_offset_vor_regression']
