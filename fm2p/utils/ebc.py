# -*- coding: utf-8 -*-
"""
Egocentric boundary cell rate map calculations.

Functions
---------
calculate_egocentric_rate_map(trajectory_data, spike_rate, boundaries, distance_bins, angle_bins)
    Calculate the egocentric boundary cell rate map.
calc_EBC(data, sps_ind)
    Calculate the egocentric boundary cell rate map for a specific cell.
calc_show_rate_maps(data, show_inds)
    Calculate and show egocentric boundary cell rate maps for specific cells.
plot_single_polar_ratemap(rate_map, ax=None)
    Plot a single polar rate map.
plot_allocentric_spikes(fig, ax, data, cellind, spikethresh='auto')
    Plot allocentric spikes.
plot_egocentic_wall_positions(fig, ax, topdlc, body_tracking_results, sps, cellind, spikethresh=20, pxls2cm=None)
    Plot egocentric wall positions.

Author: DMM, last modified May 2025
"""


import numpy as np
import matplotlib.pyplot as plt

import fm2p


def calculate_egocentric_rate_map(trajectory_data, spike_rate, boundaries, distance_bins, angle_bins):
    """
    Calculates the egocentric boundary cell rate map.

    Arguments
    ---------
    trajectory_data : np.array
        Array of shape (n_time_bins, 3) containing position (x, y) and head direction (theta)
        at each time bin.
    spike_times : np.array
        Array of spike times.
    boundaries : np.array
        Array of shape (n_boundary_points, 2) defining the environment boundaries.
    distance_bins : np.array
        Array defining the distance bin edges.
    angle_bins : np.array
        Array defining the angle bin edges (in radians).

    Returns
    -------
    rate_map : np.array
        2D array representing the firing rate map in egocentric coordinates.
        Rows correspond to distance bins, columns correspond to angle bins.
    """

    n_distance_bins = len(distance_bins) - 1
    n_angle_bins = len(angle_bins) - 1
    rate_map = np.zeros((n_distance_bins, n_angle_bins))
    occupancy_map = np.zeros((n_distance_bins, n_angle_bins))

    # dt = trajectory_data[1,0] - trajectory_data[0,0]

    # spike_indices = np.floor(spike_times / dt).astype(int)

    for t_idx, (time, x, y, theta) in enumerate(trajectory_data):
      
        # Calculate egocentric distance and angle to the nearest boundary
        min_distance = float('inf')
        min_angle = 0

        for bx, by in boundaries:
            dx = bx - x
            dy = by - y
            distance = np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx) - theta
            angle = np.arctan2(np.sin(angle), np.cos(angle))  # Normalize angle to [-pi, pi]

            if distance < min_distance:
                min_distance = distance
                min_angle = angle

        # Find the corresponding bins
        distance_bin_idx = np.digitize(min_distance, distance_bins) - 1
        angle_bin_idx = np.digitize(min_angle, angle_bins) - 1

        # Handle edge cases
        if 0 <= distance_bin_idx < n_distance_bins and 0 <= angle_bin_idx < n_angle_bins:
            occupancy_map[distance_bin_idx, angle_bin_idx] += 1
            # if t_idx in spike_indices:
            rate_map[distance_bin_idx, angle_bin_idx] += spike_rate[int(time)]

    # Calculate firing rate
    rate_map = np.divide(rate_map, occupancy_map, out=np.zeros_like(rate_map), where=occupancy_map!=0)
    
    return rate_map


def calc_EBC(data, sps_ind):
    """ Calculate egocentric boundary cell rate map.

    Parameters
    ----------
    data : dict
        Dictionary containing the following keys:
            - 'pxls2cm': Conversion factor from pixels to cm.
            - 'arenaTL': Top-left corner of the arena.
            - 'arenaTR': Top-right corner of the arena.
            - 'arenaBR': Bottom-right corner of the arena.
            - 'x': X-coordinates of the animal's position.
            - 'y': Y-coordinates of the animal's position.
            - 'head_yaw_deg': Head yaw angles in degrees.
            - 's2p_spks': Spike times for each cell.
    sps_ind : int
        Index of the cell for which to calculate the rate map.
    
    Returns
    -------
    rate_map : np.array
        2D array representing the firing rate map in egocentric coordinates.
        Rows correspond to distance bins, columns correspond to angle bins.
    """
    
    pxls2cm = data['pxls2cm']

    x1 = np.nanmedian(data['arenaTL']['x']) / pxls2cm
    x2 = np.nanmedian(data['arenaTR']['x']) / pxls2cm
    y1 = np.nanmedian(data['arenaTL']['y']) / pxls2cm
    y2 = np.nanmedian(data['arenaBR']['y']) / pxls2cm

    traj_arr = np.stack([
        data['x'] / pxls2cm,
        data['y'] / pxls2cm,
        np.deg2rad(data['head_yaw_deg'])
    ], axis=1)

    throw_inds = np.sum(np.isnan(traj_arr),axis=1) > 0
    times = np.arange(np.sum(~throw_inds))
    trajectory_data = np.delete(traj_arr, throw_inds, axis=0)
    trajectory_data = np.concatenate([times[:,np.newaxis], trajectory_data],axis=1)

    boundaries = np.array([
        [x1,y1], [x1,y2], [x2,y1], [x2,y2]
    ])

    distance_bins = np.linspace(0,15,8)
    angle_bins = np.deg2rad(np.arange(-180,184,8))

    rate_map = calculate_egocentric_rate_map(
        trajectory_data=trajectory_data,
        spike_rate=data['s2p_spks'][sps_ind, ~throw_inds],
        boundaries=boundaries,
        distance_bins=distance_bins,
        angle_bins=angle_bins
    )

    return rate_map


def calc_show_rate_maps(data, show_inds):
    """ Calculate and show egocentric boundary cell rate maps.

    Parameters
    ----------
    data : dict
        Dictionary containing the following keys:
            - 'pxls2cm': Conversion factor from pixels to cm.
            - 'arenaTL': Top-left corner of the arena.
            - 'arenaTR': Top-right corner of the arena.
            - 'arenaBR': Bottom-right corner of the arena.
            - 'x': X-coordinates of the animal's position.
            - 'y': Y-coordinates of the animal's position.
            - 'head_yaw_deg': Head yaw angles in degrees.
            - 's2p_spks': Spike times for each cell.
    show_inds : list
        List of indices of the cells to show.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object with the rate maps.
    """

    parula_map = fm2p.make_parula()

    pxls2cm = data['pxls2cm']

    x1 = np.nanmedian(data['arenaTL']['x']) / pxls2cm
    x2 = np.nanmedian(data['arenaTR']['x']) / pxls2cm
    y1 = np.nanmedian(data['arenaTL']['y']) / pxls2cm
    y2 = np.nanmedian(data['arenaBR']['y']) / pxls2cm

    traj_arr = np.stack([
        data['x'] / pxls2cm,
        data['y'] / pxls2cm,
        np.deg2rad(data['head_yaw_deg'])
    ], axis=1)

    throw_inds = np.sum(np.isnan(traj_arr),axis=1) > 0
    times = np.arange(np.sum(~throw_inds))
    trajectory_data = np.delete(traj_arr, throw_inds, axis=0)
    trajectory_data = np.concatenate([times[:,np.newaxis], trajectory_data],axis=1)

    boundaries = np.array([
        [x1,y1], [x1,y2], [x2,y1], [x2,y2]
    ])
    distance_bins = np.linspace(0,17,8)
    angle_bins = np.deg2rad(np.arange(-180,184,8))

    fig, axs = plt.subplots(
        10, int(np.ceil(cellind/10)),
        figsize=(12,12), dpi=300, subplot_kw={'projection': 'polar'})
    axs = axs.flatten()

    for cellind in show_inds:

        ax = axs[cellind]

        rate_map = calculate_egocentric_rate_map(
            trajectory_data=trajectory_data,
            spike_rate=data['s2p_spks'][cellind, ~throw_inds],
            boundaries=boundaries,
            distance_bins=distance_bins,
            angle_bins=angle_bins
        )

        rate_mesh_X, rate_mesh_Y = np.meshgrid(angle_bins+(np.pi/2), distance_bins)
        ax.pcolormesh(rate_mesh_X, rate_mesh_Y, rate_map, edgecolors='face', cmap=parula_map)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(cellind)

    fig.tight_layout()

    return fig


def plot_single_polar_ratemap(rate_map, ax=None):
    """ Plot a single polar rate map.
    
    Parameters
    ----------
    rate_map : np.array
        2D array representing the firing rate map in egocentric coordinates.
        Rows correspond to distance bins, columns correspond to angle bins.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new axes will be created.
    """

    parula_map = fm2p.make_parula()

    distance_bins = np.linspace(0,17,8)
    angle_bins = np.deg2rad(np.arange(-180,184,8))

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    rate_mesh_X, rate_mesh_Y = np.meshgrid(angle_bins+(np.pi/2), distance_bins) # +(np.pi/2)
    ax.pcolormesh(rate_mesh_X, rate_mesh_Y, rate_map, edgecolors='face', cmap=parula_map)
    ax.set_yticks([])
    ax.set_xticks([])
    if ax is None:
        plt.colorbar(label='sp/s')


def plot_allocentric_spikes(fig, ax, data, cellind, spikethresh='auto'):
    """ Plot allocentric spikes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to plot on.
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    data : dict
        Dictionary containing the following:
            - 'pxls2cm': Conversion factor from pixels to cm.
            - 'x': X-coordinates of the animal's position.
            - 'y': Y-coordinates of the animal's position.
            - 'head_yaw_deg': Head yaw angles in degrees.
            - 's2p_spks': Spike times for each cell.
    cellind : int
        Index of the cell for which to plot spikes.
    spikethresh : float or str, optional
        Spike threshold. If 'auto', the 90th percentile of spikes will be used.
        The default is 'auto'.
    """

    pxls2cm = data['pxls2cm']

    cmap = plt.cm.hsv(np.linspace(0,1,360))

    if (type(spikethresh) == str) and (spikethresh=='auto'):
        spikethresh = np.percentile(data['s2p_spks'][cellind,:], 90)

    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(1,1, dpi=300)

    ax.axis('equal')
    ax.plot(data['x'] / pxls2cm, data['y'] / pxls2cm, color='k', lw=1)
    for i in range(len(data['head_yaw_deg'])):
        if (~np.isnan(data['head_yaw_deg'][i])) and (data['s2p_spks'][cellind,i]>spikethresh):
            ax.plot(data['x'][i] / pxls2cm, data['y'][i] / pxls2cm,
                'o', ms=1, color=cmap[int(data['head_yaw_deg'][i])])
    ax.invert_yaxis()

    return fig


def plot_egocentic_wall_positions(fig, ax, topdlc, body_tracking_results, sps, cellind, spikethresh=20, pxls2cm=None):
    """ Plot egocentric wall positions.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to plot on.
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    topdlc : pd.DataFrame
        DataFrame containing top-down camera data with columns 'tl_corner_x', 'tr_corner_x',
        'tl_corner_y', and 'br_corner_y'.
    body_tracking_results : pd.DataFrame
        DataFrame containing body tracking results with columns 'x', 'y', and 'head_yaw_deg'.
    sps : np.array
        Array of shape (n_cells, n_time_bins) containing spike times for each cell.
    cellind : int
        Index of the cell for which to plot spikes.
    spikethresh : float, optional
        Spike threshold. The default is 20.
    pxls2cm : float, optional
        Conversion factor from pixels to cm. If None, a default value will be used.
        The default is None.
    """

    if pxls2cm is None:
        pxls2cm = 86.33960307728161

    x1 = np.nanmedian(topdlc['tl_corner_x']) / pxls2cm
    x2 = np.nanmedian(topdlc['tr_corner_x']) / pxls2cm
    y1 = np.nanmedian(topdlc['tl_corner_y']) / pxls2cm
    y2 = np.nanmedian(topdlc['br_corner_y']) / pxls2cm

    wall_list = [
        fm2p.Wall(x1,y1,x2,y1),
        fm2p.Wall(x1,y1,x1,y2),
        fm2p.Wall(x2,y1,x2,y2),
        fm2p.Wall(x1,y2,x2,y2)
    ]

    raydists_above_sps_thresh = []

    for i in range(len(body_tracking_results['head_yaw_deg'])):
        if (~np.isnan(body_tracking_results['head_yaw_deg'][i])) and (sps[cellind,i]>spikethresh):
            valerr_count = 0
            try:
                ray_distances = fm2p.closest_wall_per_ray(
                    body_tracking_results['x'][i] / pxls2cm,
                    body_tracking_results['y'][i] / pxls2cm,
                    np.deg2rad(body_tracking_results['head_yaw_deg'][i]),
                    wall_list,
                    ego_rays_deg=1
                )
                raydists_above_sps_thresh.append(ray_distances)
            except ValueError as e:
                valerr_count += 1

    fig, ax = plt.subplots(1,1, subplot_kw=dict(projection='polar'), dpi=300)
    for rd in raydists_above_sps_thresh:
        ax.plot(np.deg2rad(np.arange(0,360))+(np.pi/2), rd, color='k', alpha=0.2)
    ax.set_yticks([])
    ax.set_xticks([])
    fig.tight_layout()

    return fig


