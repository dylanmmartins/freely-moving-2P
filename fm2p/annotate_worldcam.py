

"""Annotate world-camera video.

Two objects, chosen with --obj:

  pillar  (default)  a closed polygon around the pillar, saved as a per-frame
                     binary mask stack -- the ground truth the reconstruction
                     is scored against in world-camera space.

  corner             the VERTICAL edge where two arena walls meet, clicked as
                     two points: where it meets the floor, then where the wall
                     tops meet.  This is the mode to reach for.  The horizon is
                     ambiguous exactly where a corner is in view -- the
                     floor/wall boundary is a V there, not a line -- whereas the
                     corner edge is unmistakable in those same frames, and its
                     two endpoints are individually identifiable in a way no
                     point along a horizon is.  Between them the two modes cover
                     every frame: 57% of this session's frames face a flat wall
                     (use floor), 43% have a corner in view (use corner).

                     Both endpoints carry the calibration.  The base sits at
                     elevation pitch - atan(h/d) and the top at
                     pitch + atan((wall_h - h)/d), with d known from the head
                     position and the arena outline, so the pair separates pitch
                     from camera height instead of trading one against the
                     other.  The segment's tilt gives roll.  Only annotate when
                     BOTH ends are clear of the frame edge -- a clipped endpoint
                     is a lower bound, not a measurement, and the GUI warns when
                     a click lands near the border.

  floor              the floor/wall horizon, as a two-point line per frame.
                     The pillar barely constrains head pitch (a vertical
                     cylinder near the image centre hardly moves when the head
                     tilts), whereas the horizon is almost pure pitch: its
                     height in the image sets the pitch offset and its tilt
                     sets roll.  A few dozen annotated horizons are enough to
                     fit the two unknowns the pillar cannot -- pitch offset and
                     camera height.

Neither mode will overwrite an existing output file; move or rename the old one,
or pass --out, so a re-run cannot silently destroy annotation work.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox


VERTEX_RADIUS   = 7          # hit-test radius in canvas pixels
POLY_FILL_COLOR = '#00ff88'  # colour for filled polygon (+ outline)
POLY_ALPHA      = 0.35       # alpha for the fill overlay (simulated via blend)
LINE_COLOR      = '#00e5ff'  # colour for the horizon line in --obj floor
CORNER_COLOR    = '#ffb000'  # colour for the wall-corner edge in --obj corner

OBJ_PILLAR = 'pillar'
OBJ_FLOOR  = 'floor'
OBJ_CORNER = 'corner'

# Points that make a complete annotation, per object.  A horizon and a corner
# edge are both two-point objects -- further clicks would be ambiguous rather
# than additive.
OBJ_NPTS = {OBJ_PILLAR: None, OBJ_FLOOR: 2, OBJ_CORNER: 2}
OBJ_SUFFIX = {OBJ_PILLAR: '_polygon_masks.npy',
              OBJ_FLOOR:  '_horizon_lines.npy',
              OBJ_CORNER: '_corner_lines.npy'}

# How close to the frame border a corner endpoint may land before it is flagged.
# A clipped endpoint is a lower bound on where the corner really ends, and
# fitting to it would bias pitch and camera height together.
EDGE_MARGIN_PX = 6


def _blend_overlay(frame_rgb, points, h, w, canvas_h, canvas_w):
    """Return a PIL-compatible bytes image with polygon overlay drawn on frame."""
    from PIL import Image, ImageDraw

    # Scale factors: canvas coords -> original frame coords
    sx = w / canvas_w
    sy = h / canvas_h

    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img, 'RGBA')

    if len(points) >= 3:
        # Convert canvas-space points back to image-space for drawing
        img_pts = [(int(px * sx), int(py * sy)) for (px, py) in points]
        r, g, b = 0x00, 0xff, 0x88
        alpha = int(POLY_ALPHA * 255)
        draw.polygon(img_pts, fill=(r, g, b, alpha), outline=(r, g, b, 255))

    return img

class AnnotationGUI:

    def __init__(self, root, video_path, start_frame=0, out_path=None,
                 obj=OBJ_PILLAR):
        self.root       = root
        self.video_path = video_path
        self.start_frame = start_frame
        self.obj        = obj
        self.max_pts    = OBJ_NPTS[obj]

        # Refuse before loading anything, so a mistake costs no annotation time.
        if out_path is None:
            out_path = os.path.splitext(video_path)[0] + OBJ_SUFFIX[obj]
        if os.path.exists(out_path):
            raise FileExistsError(
                f'{out_path} already exists. Move or rename it, or pass --out '
                f'with a different path -- this tool never overwrites an '
                f'annotation file.')

        # ---- load all frames ----
        print(f'Loading frames from {os.path.basename(video_path)} ...')
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        n_load = total_frames - start_frame
        self.frames_rgb = []  # list of (orig_h, orig_w, 3) uint8 RGB
        for _ in range(n_load):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb  = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            self.frames_rgb.append(rgb)
        cap.release()

        self.n_frames = len(self.frames_rgb)
        self.orig_h   = orig_h
        self.orig_w   = orig_w
        print(f'  Loaded {self.n_frames} frames  ({orig_w}x{orig_h})')

        self.out_path = out_path

        # per-frame polygon storage: list of lists of (canvas_x, canvas_y) tuples
        self.polygons = [[] for _ in range(self.n_frames)]

        # state
        self.current_frame = 0
        self._drag_idx     = None
        self._photo        = None

        self._build_ui()
        self._show_frame()

    def _build_ui(self):
        self.root.title(f'Worldcam annotator - {self.obj}')
        self.root.configure(bg='#1a1a1a')


        max_dim    = 900
        scale      = min(max_dim / self.orig_w, max_dim / self.orig_h, 1.0)
        self.cw    = max(1, int(self.orig_w * scale))
        self.ch    = max(1, int(self.orig_h * scale))


        top = tk.Frame(self.root, bg='#1a1a1a')
        top.pack(fill='x', padx=8, pady=(8, 2))

        self.frame_label = tk.Label(
            top, text='', fg='#cccccc', bg='#1a1a1a',
            font=('Helvetica', 12, 'bold'))
        self.frame_label.pack(side='left')

        self.annotated_label = tk.Label(
            top, text='', fg='#88ff88', bg='#1a1a1a', font=('Helvetica', 11))
        self.annotated_label.pack(side='right')


        self.canvas = tk.Canvas(
            self.root, width=self.cw, height=self.ch,
            bg='#000000', cursor='crosshair',
            highlightthickness=1, highlightbackground='#444444')
        self.canvas.pack(padx=8, pady=4)


        btn_bar = tk.Frame(self.root, bg='#1a1a1a')
        btn_bar.pack(fill='x', padx=8, pady=(2, 8))

        btn_style = dict(
            bg='#2a2a2a', fg='#dddddd', relief='flat',
            padx=14, pady=6, font=('Helvetica', 11),
            activebackground='#444444', activeforeground='white',
            cursor='hand2')

        self.btn_back = tk.Button(btn_bar, text='◀  Back',
                                  command=self._go_back, **btn_style)
        self.btn_back.pack(side='left', padx=(0, 4))

        self.btn_next = tk.Button(btn_bar, text='Next  ▶',
                                  command=self._go_next, **btn_style)
        self.btn_next.pack(side='left', padx=(0, 12))

        tk.Button(btn_bar, text='Clear frame',
                  command=self._clear_frame, **btn_style).pack(side='left', padx=(0, 4))

        tk.Button(btn_bar, text='Save & Quit',
                  command=self._save_and_quit,
                  bg='#1a4a2a', fg='#88ff88',
                  relief='flat', padx=14, pady=6,
                  font=('Helvetica', 11, 'bold'),
                  activebackground='#2a6a3a', activeforeground='white',
                  cursor='hand2').pack(side='right')

        hint_txt = ('Left-click: add point  |  Drag point: move  |  '
                    'Right-click point: delete  |  '
                    'Frames without annotation are left empty')
        if self.obj == OBJ_FLOOR:
            hint_txt = ('Click TWO points on the floor/wall horizon  |  '
                        'Drag to adjust  |  Right-click a point to delete  |  '
                        'Skip frames where no horizon is visible')
        elif self.obj == OBJ_CORNER:
            hint_txt = ('Click the wall corner BASE (floor), then its TOP  |  '
                        'Drag to adjust  |  Right-click a point to delete  |  '
                        'Skip the frame if either end is cut off by the border')
        hint = tk.Label(
            self.root, text=hint_txt,
            fg='#666666', bg='#1a1a1a', font=('Helvetica', 9))
        hint.pack(pady=(0, 4))


        self.canvas.bind('<ButtonPress-1>',   self._on_lpress)
        self.canvas.bind('<B1-Motion>',       self._on_lmove)
        self.canvas.bind('<ButtonRelease-1>', self._on_lrelease)
        self.canvas.bind('<ButtonPress-3>',   self._on_rclick)


        self.root.bind('<Right>', lambda _: self._go_next())
        self.root.bind('<Left>',  lambda _: self._go_back())
        self.root.bind('<Escape>', lambda _: self.root.destroy())

    def _show_frame(self):
        from PIL import Image, ImageTk

        pts   = self.polygons[self.current_frame]
        frame = self.frames_rgb[self.current_frame]

        img = _blend_overlay(frame, pts, self.orig_h, self.orig_w, self.ch, self.cw)
        img_resized = img.resize((self.cw, self.ch), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img_resized)

        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self._photo, tags='frame')

        self._draw_polygon_overlay()
        self._update_labels()

    def _draw_polygon_overlay(self):
        self.canvas.delete('poly_line')
        self.canvas.delete('vertex')
        pts = self.polygons[self.current_frame]
        if not pts:
            return

        if self.obj == OBJ_CORNER:
            # The segment only, never extended: unlike the horizon, WHERE this
            # line ends is the measurement.
            if len(pts) >= 2:
                (x0, y0), (x1, y1) = pts[0], pts[1]
                self.canvas.create_line(x0, y0, x1, y1, fill=CORNER_COLOR,
                                        width=2, tags='poly_line')
            for j, (x, y) in enumerate(pts):
                near = (x < EDGE_MARGIN_PX or y < EDGE_MARGIN_PX
                        or x > self.cw - EDGE_MARGIN_PX
                        or y > self.ch - EDGE_MARGIN_PX)
                self.canvas.create_text(
                    x + 12, y - 12,
                    text=('base' if j == 0 else 'top')
                         + ('  CLIPPED?' if near else ''),
                    fill='#ff5555' if near else CORNER_COLOR,
                    font=('Helvetica', 9, 'bold'), anchor='w', tags='poly_line')
        elif self.obj == OBJ_FLOOR:
            # Draw the horizon extended to both canvas edges rather than just
            # between the two clicks: the eye judges a line against a long edge
            # far better than against a short segment, and the fit uses the
            # line, not the endpoints.
            if len(pts) >= 2:
                (x0, y0), (x1, y1) = pts[0], pts[1]
                if x1 != x0:
                    m = (y1 - y0) / float(x1 - x0)
                    self.canvas.create_line(
                        0, y0 - m * x0, self.cw, y0 + m * (self.cw - x0),
                        fill=LINE_COLOR, width=2, tags='poly_line')
                else:
                    self.canvas.create_line(x0, 0, x0, self.ch,
                                            fill=LINE_COLOR, width=2,
                                            tags='poly_line')
        else:
            # lines
            for i in range(len(pts)):
                x0, y0 = pts[i]
                x1, y1 = pts[(i + 1) % len(pts)]
                self.canvas.create_line(
                    x0, y0, x1, y1,
                    fill=POLY_FILL_COLOR, width=2, tags='poly_line')

        # vertices
        r = VERTEX_RADIUS
        outline = {OBJ_FLOOR: LINE_COLOR, OBJ_CORNER: CORNER_COLOR}.get(
            self.obj, POLY_FILL_COLOR)
        for i, (x, y) in enumerate(pts):
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill='#ffffff', outline=outline,
                width=2, tags=('vertex', f'v{i}'))

    def _update_labels(self):
        self.frame_label.config(
            text=f'Frame {self.current_frame + 1} / {self.n_frames}')
        need = 3 if self.max_pts is None else self.max_pts
        n_ann = sum(1 for p in self.polygons if len(p) >= need)
        self.annotated_label.config(
            text=f'{n_ann} annotated  |  {self.n_frames - n_ann} empty')

    def _go_next(self):
        if self.current_frame < self.n_frames - 1:
            prev_pts = list(self.polygons[self.current_frame])
            self.current_frame += 1
            # Copy previous polygon to next frame if it has none
            if not self.polygons[self.current_frame] and prev_pts:
                self.polygons[self.current_frame] = list(prev_pts)
            self._show_frame()

    def _go_back(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            self._show_frame()

    def _clear_frame(self):
        self.polygons[self.current_frame] = []
        self._show_frame()

    def _hit_vertex(self, x, y):
        """Return index of vertex under (x, y), or None."""
        pts = self.polygons[self.current_frame]
        for i, (vx, vy) in enumerate(pts):
            if (x - vx) ** 2 + (y - vy) ** 2 <= VERTEX_RADIUS ** 2:
                return i
        return None

    def _on_lpress(self, event):
        x, y = event.x, event.y
        hit = self._hit_vertex(x, y)
        if hit is not None:
            self._drag_idx = hit
        else:
            self._drag_idx = None
            pts = self.polygons[self.current_frame]
            if self.max_pts is not None and len(pts) >= self.max_pts:
                return          # a horizon is two points; ignore further clicks
            pts.append((x, y))
            self._show_frame()

    def _on_lmove(self, event):
        if self._drag_idx is None:
            return
        x, y = event.x, event.y
        # clamp to canvas
        x = max(0, min(x, self.cw - 1))
        y = max(0, min(y, self.ch - 1))
        self.polygons[self.current_frame][self._drag_idx] = (x, y)
        self._show_frame()

    def _on_lrelease(self, _event):
        self._drag_idx = None

    def _on_rclick(self, event):
        x, y = event.x, event.y
        hit = self._hit_vertex(x, y)
        if hit is not None:
            del self.polygons[self.current_frame][hit]
            self._show_frame()

    def _save_and_quit(self):
        sx = self.orig_w / self.cw
        sy = self.orig_h / self.ch

        # Re-check: the file may have appeared while the GUI was open, and an
        # hour of annotation is not worth risking to a race.
        if os.path.exists(self.out_path):
            messagebox.showerror(
                'Not saved',
                f'{self.out_path} now exists -- refusing to overwrite.\n\n'
                f'Move it aside and save again.')
            return

        if self.obj in (OBJ_FLOOR, OBJ_CORNER):
            # For corner mode row 0 is the BASE and row 1 the TOP; the fit needs
            # to know which is which, so click order is preserved.
            # Two endpoints per frame in ORIGINAL image pixels, NaN where the
            # frame was skipped.  Endpoints rather than slope/intercept so a
            # vertical line stays representable and the annotation is exactly
            # what was clicked; float, because a sub-pixel horizon matters at
            # 0.036 deg/px.
            lines = np.full((self.n_frames, 2, 2), np.nan, dtype=np.float64)
            for i, pts_canvas in enumerate(self.polygons):
                if len(pts_canvas) < 2:
                    continue
                for j in range(2):
                    lines[i, j, 0] = pts_canvas[j][0] * sx
                    lines[i, j, 1] = pts_canvas[j][1] * sy
            np.save(self.out_path, lines)
            n = int(np.isfinite(lines[:, 0, 0]).sum())
            print(f'Saved {self.obj} lines -> {self.out_path}  '
                  f'shape={lines.shape}  ({n} frames annotated)')
            messagebox.showinfo('Saved', f'{n} {self.obj} lines saved to:'
                                         f'\n{self.out_path}')
            self.root.destroy()
            return

        masks = np.zeros((self.n_frames, self.orig_h, self.orig_w), dtype=np.uint8)
        for i, pts_canvas in enumerate(self.polygons):
            if len(pts_canvas) < 3:
                continue
            # convert canvas coords -> original image coords
            pts_img = [(int(px * sx), int(py * sy)) for (px, py) in pts_canvas]
            arr = np.array(pts_img, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(masks[i], [arr], 1)

        np.save(self.out_path, masks)
        print(f'Saved masks -> {self.out_path}  shape={masks.shape}')
        messagebox.showinfo('Saved', f'Saved to:\n{self.out_path}')
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(
        description='Annotate worldcam video: pillar polygons or the '
                    'floor/wall horizon.')
    parser.add_argument('--video_path', help='Path to the worldcam .avi video file', default=None)
    parser.add_argument('--obj', choices=[OBJ_PILLAR, OBJ_FLOOR, OBJ_CORNER],
                        default=OBJ_PILLAR,
                        help='What to annotate. pillar (default): a closed '
                             'polygon per frame, saved as a mask stack. floor: '
                             'the floor/wall horizon as a two-point line per '
                             'frame. corner: the vertical wall-corner edge, '
                             'clicked base then top -- unambiguous exactly '
                             'where the horizon is not, and its two endpoints '
                             'separate pitch from camera height. Neither is '
                             'constrained by the pillar.')
    parser.add_argument('--start', type=int, default=0,
                        help='First frame index to load (default: 0)')
    parser.add_argument('--out', default=None,
                        help='Output .npy path (default: <video> plus '
                             '_polygon_masks.npy or _horizon_lines.npy '
                             'according to --obj). Never overwritten.')
    args = parser.parse_args()

    if args.video_path is None:
        # print(f'ERROR: video not found: {args.video_path}', file=sys.stderr)
        print('Using default video.')
        video_path = '/home/dylan/Fast1/ret2ego_reconstruction/251028_DMM_worldcam/fm4_251028_121027_776/251028_DMM_DMM000_fm_04_eyecam_deinter.avi'
    else:
        video_path = args.video_path

    try:
        import PIL  # noqa: check available
        del PIL
    except ImportError:
        print('ERROR: Pillow is required.  Install with: pip install Pillow', file=sys.stderr)
        sys.exit(1)

    root = tk.Tk()
    try:
        AnnotationGUI(root, video_path, start_frame=args.start,
                      out_path=args.out, obj=args.obj)
    except FileExistsError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
    root.mainloop()


if __name__ == '__main__':
    main()
