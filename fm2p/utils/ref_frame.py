# -*- coding: utf-8 -*-
"""
Reference frame calculations for freely moving behavior.

Functions
---------
angle_to_target(x_c, y_c, heading, x_t, y_t)
    Calculate the angle to a target from a given position and heading.
calc_reference_frames(cfg, headx, heady, yaw, theta, arena_dict)
    Calculate reference frames for freely moving behavior.

Author: DMM, 2024
"""


import math
import numpy as np


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

    dist_to_center = np.zeros_like(headx) * np.nan

    for f in range(len(headx)):
        dist_to_center[f] = np.sqrt((headx[f]-centx)**2 + (heady[f]-centy)**2)

    reframe_dict = {
        'egocentric': pillar_ego,
        'retinocentric': pillar_retino,
        'pupil_from_head': pupil_from_head,
        'dist_to_center': dist_to_center
    }

    return reframe_dict

