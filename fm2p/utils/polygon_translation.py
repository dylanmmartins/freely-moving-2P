import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

class DraggablePolygon:
    # Modified from: https://stackoverflow.com/questions/57770331/how-to-plot-a-draggable-polygon
    lock = None
    def __init__(self, pts, img=None):
        self.press = None

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img, alpha=0.5, cmap='gray')

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

def user_polygon_translation(pts, img=None):

    dp = DraggablePolygon(pts=pts, image=img)
    dp.connect()

    plt.show()

    return dp.geometry


if __name__ == '__main__':

    pts_out = user_polygon_translation()

    print(pts_out)

