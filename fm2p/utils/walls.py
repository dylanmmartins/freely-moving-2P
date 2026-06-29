# -*- coding: utf-8 -*-
"""
fm2p/utils/walls.py

Wall representation and ray-casting to arena walls.

Classes
-------
Wall
    A single arena wall segment with start/end points.

Functions
---------
closest_wall_per_ray
    Distance to nearest wall for each ray cast from a single position.
calc_rays
    Distance to nearest wall for all rays across all frames (legacy helper).


DMM, January 2025
"""

import numpy as np


class Wall:

    def __init__(self, x1=0, y1=0, x2=0, y2=0):
        """ Represent a wall segment from (x1, y1) to (x2, y2). """

        self.start = np.array([x1, y1])
        self.end = np.array([x2, y2])
        self.vector = self.end - self.start

    def get_walls(arena):
        """ Build a list of Wall objects from an arena definition. """

        walls_list = []
        for i in range(len(arena.walls)):
            wall = Wall()
            wall.start = arena.walls[i][0]
            wall.end = arena.walls[i][1]
            wall.vector = wall.end - wall.start
            walls_list.append(wall)

        return walls_list


def closest_wall_per_ray(x, y, hd_radians, walls_list, ego_rays_deg=3):
    """ Distance to the nearest wall for each ray cast from position (x, y).

    Rays are evenly spaced at ego_rays_deg intervals around 360 degrees,
    anchored to hd_radians as the reference direction.

    Parameters
    ----------
    x : float
        Animal x position in cm.
    y : float
        Animal y position in cm.
    hd_radians : float
        Reference heading in radians.
    walls_list : list of Wall
        Arena walls.
    ego_rays_deg : float
        Angular spacing between rays in degrees.

    Returns
    -------
    ray_distances : list of float
        One distance per ray (length = 360 / ego_rays_deg).
    """

    rays_rad = hd_radians + np.radians(np.arange(0, 360, ego_rays_deg))
    rays_vect = np.column_stack((
        np.cos(rays_rad),
        np.sin(rays_rad)
    ))

    ray_distances = []
    for ray_vector in rays_vect:

        intersections = []
        closest_walls = []

        for wall in walls_list:
            det = np.cross(wall.vector, ray_vector)
            if det == 0:
                continue
            relative_pos = np.array([x, y]) - wall.start
            t = np.cross(relative_pos, ray_vector) / det
            if t < 0 or t > 1:
                continue
            intersection = wall.start + t * wall.vector
            # Ignore intersections behind the ray origin.
            if np.dot(intersection - np.array([x, y]), ray_vector) < 0:
                continue
            intersections.append(intersection)
            distance = np.linalg.norm(intersection - np.array([x, y]))
            closest_walls.append(distance)

        min_dist = min(closest_walls)
        ray_distances.append(min_dist)
    return ray_distances


def calc_rays(topdlc, body_tracking_results):
    """ Compute per-frame ray distances for all valid head-direction frames.

    Legacy helper; uses a hardcoded pxls2cm conversion. The BoundaryTuning
    class is the preferred path for new analyses.

    Parameters
    ----------
    topdlc : dict
        Top-camera DLC output with corner keypoints.
    body_tracking_results : dict
        Dict with 'head_yaw_deg', 'x', 'y' arrays.

    Returns
    -------
    raydists_above_sps_thresh : list of list
        One list of ray distances per valid frame.
    """

    # Hardcoded conversion for this arena setup.
    pxls2cm = 86.33960307728161

    x1 = np.nanmedian(topdlc['tl_corner_x']) / pxls2cm
    x2 = np.nanmedian(topdlc['tr_corner_x']) / pxls2cm
    y1 = np.nanmedian(topdlc['tl_corner_y']) / pxls2cm
    y2 = np.nanmedian(topdlc['br_corner_y']) / pxls2cm

    wall_list = [
        Wall(x1, y1, x2, y1),
        Wall(x1, y1, x1, y2),
        Wall(x2, y1, x2, y2),
        Wall(x1, y2, x2, y2)
    ]

    raydists_above_sps_thresh = []

    for i in range(len(body_tracking_results['head_yaw_deg'])):
        if (~np.isnan(body_tracking_results['head_yaw_deg'][i])):
            valerr_count = 0
            try:
                ray_distances = closest_wall_per_ray(
                    body_tracking_results['x'][i] / pxls2cm,
                    body_tracking_results['y'][i] / pxls2cm,
                    np.deg2rad(body_tracking_results['head_yaw_deg'][i]),
                    wall_list,
                    ego_rays_deg=1
                )
                raydists_above_sps_thresh.append(ray_distances)

            except ValueError as e:
                valerr_count += 1

    return raydists_above_sps_thresh
