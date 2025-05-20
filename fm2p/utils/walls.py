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

"""


import numpy as np


class Wall:

    def __init__(self, x1=0, y1=0, x2=0, y2=0):
        """ Initialize wall with start and end coordinates.

        Parameters
        ----------
        x1 : float
            x coordinate of start point
        y1 : float
            y coordinate of start point
        x2 : float
            x coordinate of end point
        y2 : float
            y coordinate of end point
        """

        self.start = np.array([x1, y1])
        self.end = np.array([x2, y2])
        self.vector = self.end - self.start

    def get_walls(arena):
        """ Get walls from arena object.

        Parameters
        ----------
        arena : object
            Arena object containing wall information
        
        Returns
        -------
        walls_list : list
            List of Wall objects representing the walls in the arena.
        """

        walls_list = []
        for i in range(len(arena.walls)):
            wall = Wall()
            wall.start = arena.walls[i][0]
            wall.end = arena.walls[i][1]
            wall.vector = wall.end - wall.start
            walls_list.append(wall)

        return walls_list


def closest_wall_per_ray(x, y, hd_radians, walls_list, ego_rays_deg=3):
    """ Calculate the distance to the closest wall for each ray.

    Parameters
    ----------
    x : float
        x coordinate of the point
    y : float
        y coordinate of the point
    hd_radians : float
        head direction in radians
    walls_list : list
        List of Wall objects representing the walls in the arena
    ego_rays_deg : int, optional
        Distance between rays to cast, default is 3 degrees.

    Returns
    -------
    ray_distances : list
        List of distances to the closest wall for each ray.
    """

    # Calculate ray directions in radians around current hd_radians
    rays_rad = hd_radians + np.radians(np.arange(0,360,ego_rays_deg))
    
    # Translate ray radians into vectors (need vectors to calculate intersections)
    rays_vect = np.column_stack((
        np.cos(rays_rad), # x component
        np.sin(rays_rad)  # y component
    ))

    ray_distances = []
    for ray_vector in rays_vect:

        intersections = []
        closest_walls = [] # distances of all intersecting walls

        for wall in walls_list:
            
            # calculate the determinant (if 0, lines are parallel, no intersection)
            det = np.cross(wall.vector, ray_vector)
            if det == 0:
                continue  # skip this wall and move to the next one
            
            # calculate the relative position of ray origin to wall start point
            relative_pos = np.array([x, y]) - wall.start
            
            # calculate how far along the wall line the intersection occurs
            # (t = 0 -> wall.start; t = 1 -> wall.end)
            t = np.cross(relative_pos, ray_vector) / det
            
            # if t is not between 0 and 1, the intersection is outside the finite wall line
            if t < 0 or t > 1:
                continue  # skip
            
            # after these checks are passed, calculate the intersection coordinates
            intersection = wall.start + t * wall.vector
            
            # check if the intersection is really in the direction of the ray
            if np.dot(intersection - np.array([x, y]), ray_vector) < 0:
                continue  # skip
            intersections.append(intersection)
            
            # calculate Euclidean distance from (x, y) to the intersection
            distance = np.linalg.norm(intersection - np.array([x, y]))
            closest_walls.append(distance)

        min_dist = min(closest_walls) # distance of closest wall for that ray
        ray_distances.append(min_dist) # append that distance to bin distance list

    # distances of wall for every clockwise 3 deg ray from input head direction
    return ray_distances


def calc_rays(topdlc, body_tracking_results):
    """ Calculate the distances to the closest walls for all rays.

    Parameters
    ----------
    topdlc : DataFrame
        DataFrame containing the top-down camera data.
    body_tracking_results : DataFrame
        DataFrame containing the body tracking results.

    Returns
    -------
    raydists_above_sps_thresh : list
        List of distances to the closest wall for each ray.
    """

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
        if (~np.isnan(body_tracking_results['head_yaw_deg'][i])):# and (sps[cellind,i]>Fram):
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

