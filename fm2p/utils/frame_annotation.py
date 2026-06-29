# -*- coding: utf-8 -*-
"""
fm2p/utils/frame_annotation.py

Interactive matplotlib tools for annotating arena frames.

Classes
-------
DraggablePolygon
    A polygon overlay the user can click and drag to new positions.

Functions
---------
user_polygon_translation
    Display a polygon over an image and let the user drag it to a new position.
place_points_on_image
    Display an image and collect N click positions from the user.


DMM, March 2025
"""

import numpy as np
import matplotlib

_non_interactive = {'agg', 'pdf', 'ps', 'svg', 'pgf', 'cairo'}
if matplotlib.get_backend().lower() in _non_interactive:
    for _backend in ('TkAgg', 'Qt5Agg', 'Qt4Agg', 'GTK3Agg', 'WXAgg'):
        try:
            matplotlib.use(_backend)
            break
        except Exception:
            continue
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def _ensure_interactive_backend():
    """No-op -- backend is set at module import time above."""
    pass


class DraggablePolygon:
    """ A matplotlib polygon patch the user can reposition by clicking and dragging.

    Adapted from stackoverflow.com/questions/57770331.
    """

    lock = None

    def __init__(self, pts, image=None):
        """ Draw the polygon on a new figure, optionally overlaid on image. """

        self.press = None
        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111)
        if image is not None:
            ax.imshow(image, alpha=0.5, cmap='gray')
        self.geometry = pts
        self.newGeometry = []
        poly = plt.Polygon(self.geometry, closed=True, fill=False, linewidth=3, color='#F97306')
        ax.add_patch(poly)
        self.poly = poly

    def connect(self):
        """ Wire up mouse press, release, and motion callbacks. """

        self.cidpress = self.poly.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.poly.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.poly.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        """ Record the click anchor if the click landed inside the polygon. """

        if event.inaxes != self.poly.axes:
            return
        if DraggablePolygon.lock is not None:
            return
        contains, attrd = self.poly.contains(event)
        if not contains:
            return

        if not self.newGeometry:
            x0, y0 = self.geometry[0]
        else:
            x0, y0 = self.newGeometry[0]

        self.press = x0, y0, event.xdata, event.ydata
        DraggablePolygon.lock = self

    def on_motion(self, event):
        """ Translate all vertices by (dx, dy) as the mouse moves. """

        if DraggablePolygon.lock is not self:
            return
        if event.inaxes != self.poly.axes:
            return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress

        xdx = [i + dx for i, _ in self.geometry]
        ydy = [i + dy for _, i in self.geometry]
        self.newGeometry = [[a, b] for a, b in zip(xdx, ydy)]
        self.poly.set_xy(self.newGeometry)
        self.poly.figure.canvas.draw()

    def on_release(self, event):
        """ Commit the drag and release the lock. """

        if DraggablePolygon.lock is not self:
            return

        self.press = None
        DraggablePolygon.lock = None
        self.geometry = self.newGeometry

    def disconnect(self):
        """ Disconnect all event callbacks. """

        self.poly.figure.canvas.mpl_disconnect(self.cidpress)
        self.poly.figure.canvas.mpl_disconnect(self.cidrelease)
        self.poly.figure.canvas.mpl_disconnect(self.cidmotion)


def user_polygon_translation(pts, image=None):
    """ Let the user drag a polygon to a new position over an optional image.

    Parameters
    ----------
    pts : list of [x, y]
        Initial polygon vertices.
    image : np.ndarray or None
        Background image to display behind the polygon.

    Returns
    -------
    geometry : list of [x, y]
        Updated vertex positions after the user closes the window.
    """

    _ensure_interactive_backend()
    dp = DraggablePolygon(pts=pts, image=image)
    dp.connect()

    plt.show()

    return dp.geometry


def place_points_on_image(image, num_pts=8, color='red', tight_scale=False):
    """ Display an image and collect num_pts click positions from the user.

    Parameters
    ----------
    image : np.ndarray
        Image to display.
    num_pts : int
        Number of points to collect before closing automatically.
    color : str
        Marker color for placed points.
    tight_scale : bool
        If True, clip display range to 10th-90th percentile of image values.

    Returns
    -------
    x_positions : np.ndarray
    y_positions : np.ndarray
    """

    _ensure_interactive_backend()
    fig, ax = plt.subplots(figsize=(9, 8))

    if tight_scale is True:
        vmin = np.percentile(image.flatten(), 10)
        vmax = np.percentile(image.flatten(), 90)
        ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    else:
        ax.imshow(image, cmap='gray')

    x_positions = []
    y_positions = []

    def on_click(event):
        if len(x_positions) < num_pts:
            print('Placed point {}/{}.'.format(len(x_positions) + 1, num_pts))
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                x_positions.append(x)
                y_positions.append(y)
                ax.plot(x, y, '.', color=color)
                plt.draw()
            if len(x_positions) == num_pts:
                plt.close(fig)

    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

    if len(x_positions) < num_pts:
        raise ValueError(
            'Expected {} points but only {} were placed. '
            'Close the figure only after placing all points.'.format(num_pts, len(x_positions))
        )

    return np.array(x_positions), np.array(y_positions)


if __name__ == '__main__':

    pts_out = user_polygon_translation()

    print(pts_out)
