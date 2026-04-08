
"""
Transform 3D pillar from world coordinates to mouse 2D retinal projection.
linear units in mm, ang in deg.

DMM April 2026
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

def simulate_retinal_projection(
    pillar_x, pillar_y, mouse_x, mouse_y, mouse_yaw, mouse_pitch, mouse_roll,
    pupil_tilt_h, pupil_tilt_v,
    pillar_h=210.0, pillar_d=40.0,
    eye_offset_x=3.5, eye_offset_y=-5.0, eye_offset_z=3.5
):

    
    fov_deg = 120.0
    res_w, res_h = 120, 120 # 1 pixel is 1 visual deg
    
    # focal length
    f = (res_w / 2) / np.tan(np.radians(fov_deg / 2))
    cx, cy = res_w / 2, res_h / 2
    
    # pinhole mdoel
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1]
    ])

    head_z = 25.0 # head height

    r = pillar_d / 2.0
    # 8 corners of the bounding box of the cylinder
    corners_world = np.array([
        [pillar_x + r, pillar_y + r, 0, 1],
        [pillar_x + r, pillar_y - r, 0, 1],
        [pillar_x - r, pillar_y + r, 0, 1],
        [pillar_x - r, pillar_y - r, 0, 1],
        [pillar_x + r, pillar_y + r, pillar_h, 1],
        [pillar_x + r, pillar_y - r, pillar_h, 1],
        [pillar_x - r, pillar_y + r, pillar_h, 1],
        [pillar_x - r, pillar_y - r, pillar_h, 1]
    ]).T

    # yaw, Pitch, Roll)
    R_head = R.from_euler('zyx', [mouse_yaw, mouse_pitch, mouse_roll], degrees=True).as_matrix()
    t_head = np.array([[mouse_x], [mouse_y], [head_z]])
    
    M_head = np.eye(4)
    M_head[:3, :3] = R_head
    M_head[:3, 3:] = t_head
    
    # invert so world -> head
    M_head_inv = np.linalg.inv(M_head)
    corners_head = M_head_inv @ corners_world

    # head to eye transform... theta/yaw and phi/pitch
    R_eye = R.from_euler('zyx', [pupil_tilt_h, pupil_tilt_v, 0], degrees=True).as_matrix()
    t_eye = np.array([[eye_offset_x], [eye_offset_y], [eye_offset_z]])
    
    M_eye = np.eye(4)
    M_eye[:3, :3] = R_eye
    M_eye[:3, 3:] = t_eye
    
    # invert so head -> eye
    M_eye_inv = np.linalg.inv(M_eye)
    corners_eye = M_eye_inv @ corners_head

    R_align = np.array([
        [ 0, -1,  0],
        [ 0,  0, -1],
        [ 1,  0,  0]
    ])
    
    corners_optical = R_align @ corners_eye[:3, :]

    # now the retinal projection (thjis is going to 2D now)
    retina_image = np.zeros((res_h, res_w), dtype=np.uint8)
    points_2d = []

    for i in range(8):
        pt_3d = corners_optical[:, i]
        
        # is pt behind eye?
        if pt_3d[2] <= 0:
            continue 
            
        pt_proj = K @ pt_3d
        
        u = int(pt_proj[0] / pt_proj[2])
        v = int(pt_proj[1] / pt_proj[2])
        
        points_2d.append([u, v])

    if len(points_2d) > 0:
        points_2d = np.array(points_2d, dtype=np.int32)
        
        hull = cv2.convexHull(points_2d)
        
        cv2.fillPoly(retina_image, [hull], color=255)

    return retina_image