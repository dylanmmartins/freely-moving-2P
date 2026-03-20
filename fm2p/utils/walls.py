# -*- coding: utf-8 -*-
"""
Determining wall positions and distances.

Classes
-------
Wall
    Wall class to represent a wall in the arena.

Functions
---------
closest_wall_per_ray(x, y, hd_radians, walls_list, ego_rays_deg=3)
    Calculate the distance to the closest wall for each ray.
calc_rays(topdlc, body_tracking_results)
    Calculate the distances to the closest walls for all rays.

Written 2024 DMM
"""


import numpy as np


class Wall:

    def __init__(self, x1=0, y1=0, x2=0, y2=0):

        self.start = np.array([x1, y1])
        self.end = np.array([x2, y2])
        self.vector = self.end - self.start

    def get_walls(arena):

        walls_list = []
        for i in range(len(arena.walls)):
            wall = Wall()
            wall.start = arena.walls[i][0]
            wall.end = arena.walls[i][1]
            wall.vector = wall.end - wall.start
            walls_list.append(wall)

        return walls_list


def closest_wall_per_ray(x, y, hd_radians, walls_list, ego_rays_deg=3):

    rays_rad = hd_radians + np.radians(np.arange(0,360,ego_rays_deg))
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
            if np.dot(intersection - np.array([x, y]), ray_vector) < 0:
                continue  # skip
            intersections.append(intersection)
            distance = np.linalg.norm(intersection - np.array([x, y]))
            closest_walls.append(distance)

        min_dist = min(closest_walls)
        ray_distances.append(min_dist)
    return ray_distances


def calc_rays(topdlc, body_tracking_results):

    pxls2cm = 86.33960307728161

    x1 = np.nanmedian(topdlc['tl_corner_x']) / pxls2cm
    x2 = np.nanmedian(topdlc['tr_corner_x']) / pxls2cm
    y1 = np.nanmedian(topdlc['tl_corner_y']) / pxls2cm
    y2 = np.nanmedian(topdlc['br_corner_y']) / pxls2cm

    wall_list = [
        Wall(x1,y1,x2,y1),
        Wall(x1,y1,x1,y2),
        Wall(x2,y1,x2,y2),
        Wall(x1,y2,x2,y2)
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

