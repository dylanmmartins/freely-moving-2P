# -*- coding: utf-8 -*-
"""
Annotate frames of a video.

Classes
-------
DraggablePolygon
    Class to create a draggable polygon on a matplotlib figure.

Functions
---------
user_polygon_translation(pts, image=None)
    Translate a polygon of plotted points across an image by clicking and dragging.
place_points_on_image(image, num_pts=8, color='red', tight_scale=False)
    Display an image and allow the user to click to place points.

Author: DMM, 2024
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
    """No-op — backend is set at module import time above."""
    pass


class DraggablePolygon:
    """ Modified from: https://stackoverflow.com/questions/57770331/how-to-plot-a-draggable-polygon

    """
    
    lock = None

    def __init__(self, pts, image=None):

        self.press = None

        fig = plt.figure(figsize=(9,8))
        ax = fig.add_subplot(111)
        if image is not None:
            ax.imshow(image, alpha=0.5, cmap='gray')

        self.geometry = pts
        self.newGeometry = []
        poly = plt.Polygon(self.geometry, closed=True, fill=False, linewidth=3, color='#F97306')
        ax.add_patch(poly)
        self.poly = poly


    def connect(self):

        self.cidpress = self.poly.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.poly.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.poly.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)


    def on_press(self, event):

        if event.inaxes != self.poly.axes: return
        if DraggablePolygon.lock is not None: return
        contains, attrd = self.poly.contains(event)
        if not contains: return

        if not self.newGeometry:
            x0, y0 = self.geometry[0]
        else:
            x0, y0 = self.newGeometry[0]

        self.press = x0, y0, event.xdata, event.ydata
        DraggablePolygon.lock = self


    def on_motion(self, event):

        if DraggablePolygon.lock is not self:
            return
        if event.inaxes != self.poly.axes: return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress

        xdx = [i+dx for i,_ in self.geometry]
        ydy = [i+dy for _,i in self.geometry]
        self.newGeometry = [[a, b] for a, b in zip(xdx, ydy)]
        self.poly.set_xy(self.newGeometry)
        self.poly.figure.canvas.draw()


    def on_release(self, event):

        if DraggablePolygon.lock is not self:
            return

        self.press = None
        DraggablePolygon.lock = None
        self.geometry = self.newGeometry


    def disconnect(self):

        self.poly.figure.canvas.mpl_disconnect(self.cidpress)
        self.poly.figure.canvas.mpl_disconnect(self.cidrelease)
        self.poly.figure.canvas.mpl_disconnect(self.cidmotion)


def user_polygon_translation(pts, image=None):

    _ensure_interactive_backend()
    dp = DraggablePolygon(pts=pts, image=image)
    dp.connect()

    plt.show()

    return dp.geometry


def place_points_on_image(image, num_pts=8, color='red', tight_scale=False):

    _ensure_interactive_backend()
    fig, ax = plt.subplots(figsize=(9,8))

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

            print('Placed point {}/{}.'.format(len(x_positions)+1, num_pts))

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
            f'Expected {num_pts} points but only {len(x_positions)} were placed. '
            'Close the figure only after placing all points.'
        )

    return np.array(x_positions), np.array(y_positions)


if __name__ == '__main__':

    pts_out = user_polygon_translation()

    print(pts_out)

