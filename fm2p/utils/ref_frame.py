# -*- coding: utf-8 -*-
"""
Reference frame calculations for freely moving behavior.

Functions
---------
angle_to_target(x_c, y_c, heading, x_t, y_t)
    Calculate the angle to a target from a given position and heading.
calc_reference_frames(cfg, headx, heady, yaw, theta, arena_dict)
    Calculate reference frames for freely moving behavior.
calc_vor_eye_offset(theta_interp, head_yaw_deg, fps)
    Estimate the eye-camera angular offset via two VOR-based methods.
get_ang_offset(data, fps=None)
    Retrieve or compute ang_offset from a preprocessed-data dict.

Author: DMM, 2024
"""


import math
import numpy as np
from scipy import stats


def visual_angle_degrees(distance_cm, size_cm=4.):
    """
    Calculate the visual angle in degrees.

    Parameters:
    - size_cm: float, the size of the object in centimeters
    - distance_cm: float, the distance from the observer in centimeters

    Returns:
    - float, visual angle in degrees
    """
    angle_radians = 2 * math.atan(size_cm / (2 * distance_cm))
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees


def angle_to_target(x_c, y_c, heading, x_t, y_t):
    """ Calculate the angle to a target from a given position and heading.

    Parameters
    ----------
    x_c : float
        X-coordinate of the current position.
    y_c : float
        Y-coordinate of the current position.
    heading : float
        Current heading angle in degrees.
    x_t : float
        X-coordinate of the target position.
    y_t : float
        Y-coordinate of the target position.

    Returns
    -------
    absolute_angle : float
        Absolute angle to the target in degrees.
    angle_difference : float
        Angle difference between the target and the current heading in degrees.
    """
    
    # Calculate the absolute angle to the target relative to the eastern horizontal
    absolute_angle = math.degrees(math.atan2(y_t - y_c, x_t - x_c))
    
    # Normalize absolute angle to [0, 360)
    absolute_angle = (absolute_angle + 360) % 360
    
    # Normalize heading to [0, 360)
    heading = (heading + 360) % 360  

    # Calculate the smallest angle difference (-180 to 180)
    angle_difference = (absolute_angle - heading + 180) % 360 - 180

    return absolute_angle, angle_difference


def calc_reference_frames(cfg, headx, heady, yaw, theta, arena_dict):
    """ Calculate reference frames for freely moving behavior.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing parameters for calculations.
    headx : np.ndarray
        X-coordinates of the head position.
    heady : np.ndarray
        Y-coordinates of the head position.
    yaw : np.ndarray
        Head yaw angles of the head position.
    theta : np.ndarray
        Pupil angle in degrees.
    arena_dict : dict
        Dictionary containing arena information, including pillar centroid and corners.

    Returns
    -------
    reframe_dict : dict
        Dictionary containing calculated reference frames:
            egocentric: np.ndarray, Egocentric angle to the pillar.
            retinocentric: np.ndarray, Retinocentric angle to the pillar.
            pupil_from_head: np.ndarray, Pupil angle from head position.
            dist_to_center: np.ndarray, Distance to the center of the arena.
    """

    pillarx = arena_dict['pillar_centroid']['x']
    pillary = arena_dict['pillar_centroid']['y']

    # headx and heady are vectors with len == num frames
    # pillarx and pillary are each a single int value
    pillar_ego = np.zeros_like(headx) * np.nan
    pillar_abs = np.zeros_like(headx) * np.nan
    pupil_from_head = np.zeros_like(headx) * np.nan
    pillar_retino = np.zeros_like(headx) * np.nan

    # Calculate egocentric angle to the pillar
    for f in range(len(headx)):
        pillar_abs[f], pillar_ego[f] = angle_to_target(headx[f], heady[f], yaw[f], pillarx, pillary)

    if np.size(theta) != np.size(pillar_ego):
        print('Check length of theta versus egocentric angle, which do not match! Is theta '
              + 'already aligned by TTL values and interpolated to 2P timestamps?')
        print('Sizes are theta={}, ego={}'.format(np.size(theta), np.size(pillar_ego)))

    # Calculate retinocentric angle to the pillar.
    # For now, only calculated in the horizontal plane.
    ang_offset = cfg['eyecam_angular_offset']
    for f in range(len(headx)):
        pfh = ang_offset - theta[f]
        pupil_from_head[f] = pfh
        pillar_retino[f] = ((((pfh - pillar_ego[f])+180) % 360) - 180)

    # Calculate the distance from the animal's current position to the center of the arena
    tlx = arena_dict['arenaTL']['x']
    tly = arena_dict['arenaTL']['y']
    trx = arena_dict['arenaTR']['x']
    try_ = arena_dict['arenaTR']['y']
    blx = arena_dict['arenaBL']['x']
    bly = arena_dict['arenaBL']['y']
    brx = arena_dict['arenaBR']['x']
    bry = arena_dict['arenaBR']['y']

    centx = np.nanmean([
        (trx - tlx),
        (brx - blx)
    ])
    centy = np.nanmean([
        (bry - try_),
        (bly - tly)
    ])

    dist_to_center = np.array([np.sqrt((headx[f]-centx)**2 + (heady[f]-centy)**2) for f in range(len(headx))])
    dist_to_pillar = np.array([np.sqrt((headx[f]-pillarx)**2 + (heady[f]-centy)**2) for f in range(len(headx))])
    pillar_size = np.array([visual_angle_degrees(dist_to_pillar[f], 4.) for f in range(len(headx))])

    reframe_dict = {
        'egocentric': pillar_ego,
        'retinocentric': pillar_retino,
        'pupil_from_head': pupil_from_head,
        'dist_to_center': dist_to_center,
        'dist_to_pillar': dist_to_pillar,
        'pillar_size': pillar_size
    }

    return reframe_dict


def calc_vor_eye_offset(theta_interp, head_yaw_deg, fps, head_vel_deg_s=None):
    """
    Estimate the eye-camera angular offset (ang_offset) using two VOR-based
    methods.  Both methods assume that when the animal's head is stationary the
    pupil returns to a characteristic resting position that equals ang_offset.

    The stored ``pupil_from_head = ang_offset_config - theta`` uses the config
    file value of ang_offset, which may not account for camera rotation relative
    to the head midline.  These empirical estimates replace that value.

    Parameters
    ----------
    theta_interp : np.ndarray, shape (N,)
        Horizontal eye angle (degrees) at the timebase given by ``fps``.
        Positive = rightward / temporal.
    head_yaw_deg : np.ndarray, shape (N,) or None
        Allocentric head direction (degrees, image-CW convention).  Only used
        when ``head_vel_deg_s`` is None.  Pass None when providing
        ``head_vel_deg_s`` directly.
    fps : float
        Frame rate of ``theta_interp`` (and ``head_vel_deg_s`` if given) in Hz.
        Use the highest available timebase (e.g. eye camera rate, not 2P rate).
    head_vel_deg_s : np.ndarray, shape (N,), optional
        Pre-computed head angular velocity in deg/s, at the same timebase as
        ``theta_interp``.  When provided (e.g. gyro_z interpolated to eye
        camera timestamps), ``head_yaw_deg`` is ignored.  This avoids
        differentiating ``head_yaw_deg`` and retains the full temporal
        resolution of the gyroscope.

    Returns
    -------
    dict with keys
        ang_offset_vor_null : float
            Median theta at frames where |head_vel| < 10 deg/s.  Simple and
            robust; equivalent to the eye resting position between head turns.
        ang_offset_vor_regression : float
            Velocity-regression estimate.  Eye velocity is regressed on head
            velocity during active head rotation; the VOR gain (slope) is used
            to remove the head-motion-coupled component from theta across ALL
            frames, giving a cleaner estimate of the eye's intrinsic resting
            position.
        vor_gain : float
            Absolute VOR gain (should be ≈ 1 for a well-calibrated eye).
    """
    if head_vel_deg_s is not None:
        n = min(len(theta_interp), len(head_vel_deg_s))
    elif head_yaw_deg is not None:
        n = min(len(theta_interp), len(head_yaw_deg))
    else:
        n = len(theta_interp)

    theta = np.array(theta_interp[:n], dtype=float)

    # Build head angular velocity and the cumulative displacement used for the
    # position-level correction (avoids integration drift from cumsum of vel).
    if head_vel_deg_s is not None:
        # Gyro_z is already in deg/s — no unwrapping or differentiation needed.
        head_vel = np.array(head_vel_deg_s[:n], dtype=float)
        # Integrate velocity to get cumulative angular displacement.
        # NaN frames are treated as zero velocity so cumsum stays continuous.
        hv_filled = np.where(np.isnan(head_vel), 0.0, head_vel)
        head_pos  = np.cumsum(hv_filled) / fps  # degrees
        head_delta = head_pos - np.nanmedian(head_pos)
    else:
        head = np.array(head_yaw_deg[:n], dtype=float)
        # Unwrap head direction to avoid velocity spikes at 0/360 boundary
        head_unwrap = np.rad2deg(np.unwrap(np.deg2rad(head)))
        head_vel    = np.gradient(head_unwrap) * fps
        head_delta  = head_unwrap - np.nanmedian(head_unwrap)

    eye_vel = np.gradient(theta) * fps
    valid   = (~np.isnan(head_vel)) & (~np.isnan(eye_vel)) & (~np.isnan(theta))

    # ── Method 1: VOR null ────────────────────────────────────────────────────
    # Median theta during frames where the head is (near-)stationary.
    still_thresh = 10.  # deg/s
    still = valid & (np.abs(head_vel) < still_thresh)
    if still.sum() > 10:
        ang_offset_vor_null = float(np.nanmedian(theta[still]))
    else:
        ang_offset_vor_null = float(np.nanmedian(theta[valid]))

    # ── Method 2: VOR regression ─────────────────────────────────────────────
    # During active head rotation the VOR drives:
    #   eye_vel ≈ slope * head_vel + intercept
    # slope ≈ -VOR_gain  (compensatory, so negative)
    # intercept ≈ slow eye-position drift rate
    #
    # We remove the head-velocity-correlated component from theta across the
    # whole recording using position-level slope correction.
    active_thresh = 20.  # deg/s
    active = valid & (np.abs(head_vel) > active_thresh)

    vor_gain  = 1.0
    slope     = -1.0
    intercept = 0.0

    if active.sum() > 200:
        from scipy import stats as _stats
        slope, intercept, _, _, _ = _stats.linregress(
            head_vel[active], eye_vel[active])
        vor_gain = float(-slope)   # make positive; expect ≈ 1

    # Remove the VOR-coupled component from theta using the position-level
    # slope.  The intercept (deg/s) was fit in velocity space; applying it at
    # position level as `intercept * t_vec` integrates slow drift into a
    # cumulative offset that can reach thousands of degrees over a long session,
    # making ang_offset_vor_regression wildly wrong.  Only the slope term
    # (dimensionless gain applied to head angular displacement) is valid here.
    theta_corr = theta - slope * head_delta

    if valid.sum() > 10:
        ang_offset_vor_regression = float(np.nanmedian(theta_corr[valid]))
    else:
        ang_offset_vor_regression = ang_offset_vor_null

    return {
        'ang_offset_vor_null':       ang_offset_vor_null,
        'ang_offset_vor_regression': ang_offset_vor_regression,
        'vor_gain':                  vor_gain,
    }


def get_ang_offset(data, fps=None):
    """
    Retrieve or compute the eye-camera angular offset from a preprocessed-data
    dict.

    Priority
    --------
    1. ``ang_offset_vor_regression`` — preferred (regression-based).
    2. ``ang_offset_vor_null``       — VOR-null fallback.
    3. Compute on-the-fly from ``theta_interp`` + ``head_yaw_deg``.
    4. Return ``None`` if insufficient data.

    Parameters
    ----------
    data : dict
        Preprocessed data dictionary.
    fps : float, optional
        Frame rate (Hz).  Inferred from ``twopT`` if not supplied.

    Returns
    -------
    ang_offset : float or None
    """
    if 'ang_offset_vor_regression' in data:
        return float(data['ang_offset_vor_regression'])

    if 'ang_offset_vor_null' in data:
        return float(data['ang_offset_vor_null'])

    theta = data.get('theta_interp', None)
    head  = data.get('head_yaw_deg', None)
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

