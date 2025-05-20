# -*- coding: utf-8 -*-
"""
Utility functions for calculating and visualizing behavior data.

Functions
---------
plot_yaw_distribution(fig, ax, body_tracking_results)
    Plot the distribution of head yaw angles.
plot_speed_distribution(fig, ax, body_tracking_results)
    Plot the distribution of speed.
plot_movement_yaw_distribution(fig, ax, body_tracking_results)
    Plot the distribution of movement yaw angles.

Author: DMM, last modified May 2025
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_yaw_distribution(fig, ax, body_tracking_results):
    """ Plot the distribution of head yaw angles.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to plot on. If None, a new figure will be created.
    ax : matplotlib.axes.Axes
        Axes object to plot on. If None, a new axes will be created.
    body_tracking_results : pd.DataFrame
        DataFrame containing body tracking results with a column 'head_yaw_deg'.
            
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object with the plot.
    """

    if (fig is None) and (ax is None):
        fig, ax = plt.figure(figsize=(3,2),dpi=300)

    ax.hist(np.deg2rad(body_tracking_results['head_yaw_deg']-180), bins=20, color='k')
    ax.set_xlim([-np.pi,np.pi])
    ax.set_xlabel('head azimuth (rad)')
    ax.set_ylabel('# frames')
    fig.tight_layout()

    return fig


def plot_speed_distribution(fig, ax, body_tracking_results):
    """ Plot the distribution of speed.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to plot on. If None, a new figure will be created.
    ax : matplotlib.axes.Axes
        Axes object to plot on. If None, a new axes will be created.
    body_tracking_results : pd.DataFrame
        DataFrame containing body tracking results with a column 'speed'.
            
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object with the plot.
    """

    if (fig is None) and (ax is None):
        fig, ax = plt.figure(figsize=(3,2),dpi=300)

    ax.hist(body_tracking_results['speed'], bins=20, color='k')
    ax.set_xlim([0,40])
    ax.set_xlabel('speed (cm/s)')
    ax.set_ylabel('# frames')
    fig.tight_layout()

    return fig


def plot_movement_yaw_distribution(fig, ax, body_tracking_results):
    """ Plot the distribution of movement yaw angles.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to plot on. If None, a new figure will be created.
    ax : matplotlib.axes.Axes
        Axes object to plot on. If None, a new axes will be created.
    body_tracking_results : pd.DataFrame
        DataFrame containing body tracking results with a column 'movement_yaw_deg'.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object with the plot.
    """


    if (fig is None) and (ax is None):
        fig, ax = plt.figure(figsize=(3,2),dpi=300)

    ax.hist(np.deg2rad(body_tracking_results['movement_yaw_deg']-180), bins=20, color='k')
    ax.set_xlim([-np.pi,np.pi])
    ax.set_xlabel('movement azimuth (rad)')
    ax.set_ylabel('# frames')
    fig.tight_layout()

    return fig

