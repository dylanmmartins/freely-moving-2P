
import math
import numpy as np


def angle_to_target(x_c, y_c, heading, x_t, y_t):
    
    # Calculate the absolute angle to the target relative to the eastern horizontal
    absolute_angle = math.degrees(math.atan2(y_t - y_c, x_t - x_c))
    
    # Normalize absolute angle to [0, 360)
    absolute_angle = (absolute_angle + 360) % 360
    
    # Normalize heading to [0, 360)
    heading = (heading + 360) % 360  

    # Calculate the smallest angle difference (-180 to 180)
    angle_difference = (absolute_angle - heading + 180) % 360 - 180

    return absolute_angle, angle_difference


def calc_reference_frames(cfg, headx, heady, yaw, pillarx, pillary, theta):
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
        print('Check length of theta versus egocentric angle, which do not match! Is theta already aligned by TTL values and interpolated to 2P timestamps?')

    # Calculate retinocentric angle to the pillar. For now, only calculated in the horizontal plane
    ang_offset = cfg['eyecam_angular_offset']
    for f in range(len(headx)):
        pfh = ang_offset - theta[f]
        pupil_from_head[f] = pfh
        pillar_retino[f] = ((yaw[f] + pfh) + 180) % 360 - 180

    return pillar_ego, pillar_retino

