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
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


class DraggablePolygon:
    """ Class to create a draggable polygon on a matplotlib figure.

    Modified from: https://stackoverflow.com/questions/57770331/how-to-plot-a-draggable-polygon

    """
    
    lock = None

    def __init__(self, pts, image=None):
        """ Initialize the DraggablePolygon class.

        Parameters
        ----------
        pts : list
            List of points to create the polygon.
        image : np.array, optional
            Image to display in the background. The default is None.
        """

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
        """ Connect the polygon to the figure.
        """

        self.cidpress = self.poly.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.poly.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.poly.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)


    def on_press(self, event):
        """ Handle the press event on the polygon.
        
        Parameters
        ----------
        event : matplotlib.backend_bases.Event
            The event object containing information about the event.
        """

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
        """ Handle the motion event on the polygon.

        Parameters
        ----------
        event : matplotlib.backend_bases.Event
            The event object containing information about the event.
        """

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
        """ Handle the release event on the polygon.

        Parameters
        ----------
        event : matplotlib.backend_bases.Event
            The event object containing information about the event.
        """

        if DraggablePolygon.lock is not self:
            return

        self.press = None
        DraggablePolygon.lock = None
        self.geometry = self.newGeometry


    def disconnect(self):
        """ Disconnect the polygon from the figure.
        """

        self.poly.figure.canvas.mpl_disconnect(self.cidpress)
        self.poly.figure.canvas.mpl_disconnect(self.cidrelease)
        self.poly.figure.canvas.mpl_disconnect(self.cidmotion)


def user_polygon_translation(pts, image=None):
    """ Translate a polygon of plotted points across an image by clicking and dragging.

    After points are placed, returns the x and y coordinates of those points.

    Parameters
    ----------
    pts : list
        List of points to create the polygon.
    image : np.array, optional
        Image to display in the background. The default is None.
    
    Returns
    -------
    pts : list
        List of points in the polygon.
    """

    dp = DraggablePolygon(pts=pts, image=image)
    dp.connect()

    plt.show()

    return dp.geometry


def place_points_on_image(image, num_pts=8, color='red', tight_scale=False):
    """ Display an image and allow the user to click to place points.

    Parameters
    ----------
    image : np.array
        Image to display.
    num_pts : int, optional
        Number of points to place. The default is 8.
    color : str, optional
        Color of the points. The default is 'red'.
    tight_scale : bool, optional
        If True, use tight scale for the image. The default is False.

    Returns
    -------
    x_positions : np.array
        X coordinates of the points placed.
    y_positions : np.array
        Y coordinates of the points placed.
    """
    
    fig, ax = plt.subplots(figsize=(9,8))

    if tight_scale is True:
        vmin = np.percentile(image.flatten(), 10)
        vmax = np.percentile(image.flatten(), 90)
        ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    
    else:
        ax.imshow(image, cmap='gray')
    
    
    x_positions = []
    y_positions = []
    
    # Callback function to capture mouse click positions
    def on_click(event):
        if len(x_positions) < num_pts:

            print('Placed point {}/{}.'.format(len(x_positions)+1, num_pts))

            # Get the x and y coordinates of the click
            x, y = event.xdata, event.ydata
            
            if x is not None and y is not None:
                # Store the coordinates
                x_positions.append(x)
                y_positions.append(y)
                
                # Add a red point at the clicked location
                ax.plot(x, y, '.', color=color)
                plt.draw()

            # If all points have been clicked, close the figure
            if len(x_positions) == num_pts:
                plt.close(fig)

    # Connect the callback to the figure
    cid = fig.canvas.mpl_connect('button_press_event', on_click)

    # Show the plot and wait for the user to click all points
    plt.show()

    # Return the x and y positions as numpy arrays
    return np.array(x_positions), np.array(y_positions)


if __name__ == '__main__':

    pts_out = user_polygon_translation()

    print(pts_out)

