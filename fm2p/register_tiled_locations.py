# -*- coding: utf-8 -*-
"""
Calculate and plot receptive fields of cells in a 2P calcium imaging recording recorded
during head-fixation. The presented stimulus is a series of vertical and horizontal bars
of sweeping gratings.

Functions
---------


Example usage
-------------


Author: DMM, Dec. 2025
"""


import math
import os
from tqdm import tqdm
import tkinter as tk
from tkinter import Button, HORIZONTAL, Scale
import numpy as np
from PIL import Image, ImageTk
import random
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, zoom
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import fm2p


def array_to_pil(arr):
    """ Convert array to uint8 PIL image
    """
    if isinstance(arr, Image.Image):
        return arr

    a = np.asarray(arr)
    if a.dtype != np.uint8:
        # norm to 0-255
        try:
            amin = float(np.nanmin(a))
            amax = float(np.nanmax(a))
        except Exception:
            amin, amax = 0.0, 1.0
        if amax == amin:
            a = np.zeros_like(a, dtype=np.uint8)
        else:
            a = ((a - amin) / (amax - amin) * 255.0).astype(np.uint8)

    if a.ndim == 2:
        return Image.fromarray(a, mode='L') # will actually use
    if a.ndim == 3 and a.shape[2] == 3:
        return Image.fromarray(a, mode='RGB')
    if a.ndim == 3 and a.shape[2] == 4:
        return Image.fromarray(a, mode='RGBA')
    return Image.fromarray(a)


class ManualImageAligner:
    def __init__(self, fullimg, small_images, position_keys, scale_factor=1.0):
        """
        fullimg: numpy array
        small_images: list of numpy arrays
        scale_factor: initial scale factor for small images
        """
        self.fullimg_arr = fullimg
        # keep a consistent RGBA base image so we can paste accepted tiles onto it
        self.fullimg_pil = array_to_pil(fullimg).convert('RGBA')
        self.base_image = self.fullimg_pil.copy()
        self.position_keys = position_keys

        self.small_imgs_arr = small_images
        self.small_imgs_pil = [array_to_pil(img).convert('RGBA') for img in small_images]

        self.scale_factor = scale_factor

        # will store as [x_center, y_center, angle_degrees]
        self.transforms = []

        self.index = 0
        self.current_angle = 0
        self.current_offset = np.array([50, 50], float)


    def choose_scale_factor(self, fullimg_arr, small_images, start_idx=0):
        """ Resize a single tile overalid on full WF img, then save out scale factor
        """
        full_pil = array_to_pil(fullimg_arr)
        full_rgba = full_pil.convert('RGBA')

        self.preview_idx = int(start_idx)
        print('Displaying stitching position: {}'.format(self.position_keys[self.preview_idx]))
        small_pil_holder = {
            'pil': array_to_pil(small_images[self.preview_idx]),
            'angle': 0.0,
            'offset': np.array([0.0, 0.0]),
        }

        win = tk.Toplevel()
        win.title("Choose Scale Factor")

        canvas = tk.Canvas(win, width=full_pil.width, height=full_pil.height)
        canvas.pack()

        scale_factor_local = [self.scale_factor]

        def draw_small():
            small_pil = small_pil_holder['pil']
            angle = float(small_pil_holder.get('angle', 0.0))
            offset = small_pil_holder.get('offset', np.array([0.0, 0.0]))

            w, h = small_pil.size
            w2 = int(max(1, int(w * scale_factor_local[0])))
            h2 = int(max(1, int(h * scale_factor_local[0])))

            resized = small_pil.resize((w2, h2), Image.BILINEAR)

            resized_rgba = resized.convert('RGBA')
            alpha = int(255 * 0.99)
            alpha_mask = Image.new('L', resized_rgba.size, color=alpha)
            resized_rgba.putalpha(alpha_mask)

            rotated = resized_rgba.rotate(angle, expand=True)

            s = max(rotated.width, rotated.height)
            square_rgba = Image.new('RGBA', (s, s), (0, 0, 0, 0))
            paste_off = ((s - rotated.width) // 2, (s - rotated.height) // 2)
            square_rgba.paste(rotated, paste_off, rotated)

            overlay = Image.new('RGBA', full_rgba.size, (0, 0, 0, 0))
            paste_x = int(full_rgba.width // 2 - square_rgba.width // 2 + offset[0])
            paste_y = int(full_rgba.height // 2 - square_rgba.height // 2 + offset[1])
            overlay.paste(square_rgba, (paste_x, paste_y), square_rgba)

            composite = Image.alpha_composite(full_rgba, overlay)
            composite_tk = ImageTk.PhotoImage(composite)

            # del old composite if present
            if hasattr(draw_small, 'comp_id'):
                canvas.delete(draw_small.comp_id)

            draw_small.comp_id = canvas.create_image(0, 0, anchor='nw', image=composite_tk)
            # keep a ref on win
            win.composite_tk = composite_tk

        # some handlers...
        def _start_drag(event):
            # starting mouse position
            canvas._drag_start = np.array([event.x, event.y], dtype=float)

        def _drag_motion(event):
            if not hasattr(canvas, '_drag_start'):
                canvas._drag_start = np.array([event.x, event.y], dtype=float)
            cur = np.array([event.x, event.y], dtype=float)
            delta = cur - canvas._drag_start
            canvas._drag_start = cur
            small_pil_holder['offset'] = small_pil_holder.get('offset', np.array([0.0, 0.0])) + delta
            draw_small()

        def _rotate_event(event):
            # for windosw
            delta = getattr(event, 'delta', 0)
            if delta:
                step = (delta / 120.0) * 5.0
            else:
                step = 0.0
            small_pil_holder['angle'] = (small_pil_holder.get('angle', 0.0) + step) % 360.0
            draw_small()

        def _rotate_linux(event):
            if event.num == 4:
                step = 5.0
            elif event.num == 5:
                step = -5.0
            else:
                step = 0.0
            small_pil_holder['angle'] = (small_pil_holder.get('angle', 0.0) + step) % 360.0
            draw_small()

        # scale slider at bottom
        # TODO: never going to be greater than 1, so could set max to 1...?
        scale_slider = Scale(
            win, from_=0.05, to=1.5, resolution=0.01,
            orient=HORIZONTAL, label="Scale Factor",
            length=1000,
            command=lambda v: update_scale(float(v)))
        scale_slider.set(self.scale_factor)
        scale_slider.pack()

        def update_scale(val):
            scale_factor_local[0] = val
            draw_small()

        def mousewheel(event):
            # windows
            delta = getattr(event, 'delta', 0)
            if delta:
                scale_factor_local[0] *= (1 + delta/120 * 0.05)
            draw_small()

        def mousewheel_linux(event):
            if event.num == 4:
                scale_factor_local[0] *= 1.05
            elif event.num == 5:
                scale_factor_local[0] /= 1.05
            scale_factor_local[0] = max(0.01, min(10.0, scale_factor_local[0]))
            scale_slider.set(scale_factor_local[0])
            draw_small()

        # drag to move, scroll wheel to rotate
        canvas.bind("<ButtonPress-1>", _start_drag)
        canvas.bind("<B1-Motion>", _drag_motion)
        canvas.bind("<MouseWheel>", _rotate_event)
        canvas.bind("<Button-4>", _rotate_linux)
        canvas.bind("<Button-5>", _rotate_linux)

        def accept():
            self.scale_factor = scale_factor_local[0]
            self.index = int(self.preview_idx)
            print("New scale factor:", self.scale_factor)
            print("Chosen start index:", self.index)
            win.destroy()

        Button(win, text="Accept Scale", command=accept).pack()

        def pick_random():
            if len(small_images) <= 1:
                return
            new_idx = random.randrange(len(small_images))
            # if that one isn't a good tile, choose a new random one that will be easier to tell correct scale.
            # usually seems to be 0.27 as best scale factor. maybe start with that as default?
            tries = 0
            while new_idx == self.preview_idx and tries < 10:
                new_idx = random.randrange(len(small_images))
                tries += 1
            self.preview_idx = new_idx
            small_pil_holder['pil'] = array_to_pil(small_images[self.preview_idx])
            draw_small()
            print('Displaying stitching position: {}'.format(self.position_keys[self.preview_idx]))

        Button(win, text="Random Preview", command=pick_random).pack()

        draw_small()

        win.mainloop()


    def _setup_alignment_window(self):

        print('Opening alignment window.')

        self.root = tk.Tk()
        
        self.root.title("Manual Image Registration")

        self.canvas = tk.Canvas(self.root,
                                width=self.fullimg_pil.width,
                                height=self.fullimg_pil.height)
        self.canvas.pack()

        # current base img
        self.base_tk = ImageTk.PhotoImage(self.base_image)
        self.base_canvas_id = self.canvas.create_image(0, 0, anchor="nw", image=self.base_tk)

        self.btn_accept = Button(self.root, text="Accept Alignment", command=self.accept_alignment)
        self.btn_accept.pack()

        self.btn_quit = Button(self.root, text="Quit", command=self.root.destroy)
        self.btn_quit.pack()

        # bindings
        self.canvas.bind("<ButtonPress-1>", self.start_move)
        self.canvas.bind("<B1-Motion>", self.move_image)
        # rotation bindinsg
        self.canvas.bind("<MouseWheel>", self.rotate_image)
        self.canvas.bind("<Button-4>", self.rotate_image)
        self.canvas.bind("<Button-5>", self.rotate_image)


    def load_small_image(self):
        
        if self.index >= len(self.small_imgs_pil):
            return

        img = self.small_imgs_pil[self.index]
        w, h = img.size

        scaled = img.resize((int(w*self.scale_factor), int(h*self.scale_factor)),
                            Image.BILINEAR)
        rotated = scaled.rotate(self.current_angle, expand=True)

        self.current_pil = rotated
        self.current_tk = ImageTk.PhotoImage(rotated)


        print('Drawing image from position {}'.format(self.position_keys[self.index]))
        self.draw_small_image()

    def draw_small_image(self):
        if hasattr(self, "small_img_canvas_id"):
            self.canvas.delete(self.small_img_canvas_id)

        x, y = self.current_offset
        self.small_img_canvas_id = self.canvas.create_image(
            x, y, anchor="center", image=self.current_tk
        )

    def start_move(self, event):
        self.drag_start = np.array([event.x, event.y])

    def move_image(self, event):
        delta = np.array([event.x, event.y]) - self.drag_start
        self.drag_start = np.array([event.x, event.y])
        self.current_offset += delta
        self.draw_small_image()

    def rotate_image(self, event):
        delta = getattr(event, 'delta', None)
        if delta is not None:
            step = (delta / 120.0) * 2.0
        else:
            if getattr(event, 'num', None) == 4:
                step = 2.0
            elif getattr(event, 'num', None) == 5:
                step = -2.0
            else:
                step = 0.0

        self.current_angle = (self.current_angle + step) % 360.0
        self.load_small_image()

    def accept_alignment(self):
        # paste the sml img into WF img
        self.transforms.append((
            float(self.current_offset[0]),
            float(self.current_offset[1]),
            float(self.current_angle)
        ))

        if hasattr(self, 'current_pil') and self.current_pil is not None:
            # ensure RGBA and paste using its alpha
            tile = self.current_pil.convert('RGBA')
            paste_x = int(self.current_offset[0] - tile.width // 2)
            paste_y = int(self.current_offset[1] - tile.height // 2)
            base = self.base_image.copy()
            base.paste(tile, (paste_x, paste_y), tile)
            self.base_image = base

            # update canvas background image
            if self.base_image.mode not in ('RGB', 'RGBA'):
                display_image = self.base_image.convert('RGB')
            else:
                display_image = self.base_image

            self.base_tk = ImageTk.PhotoImage(display_image)
            self.canvas.itemconfig(self.base_canvas_id, image=self.base_tk)

        # adv to next image
        self.index += 1
        if self.index >= len(self.small_imgs_pil):
            print("All images aligned.")
            self.root.quit()
            return

        self.current_angle = 0
        self.current_offset = np.array([50, 50], float)
        self.load_small_image()

    def run(self):
        # create the main root first so that choose_scale_factor() can safely
        # create a top lvl window without raising TclError
        self._setup_alignment_window()

        if len(self.small_imgs_arr) == 0:
            raise RuntimeError("No small images available for scale selection.")

        # save preview idx
        self.choose_scale_factor(self.fullimg_arr, self.small_imgs_arr)

        # load first small image for manual alignment
        self.load_small_image()
        self.root.mainloop()

        return self.transforms

    # do coord transform from within small img to full widefield map
    # needs to know which small tile / stitching positoin the cell is in,
    # and it's local x/y coordinates within that image. then computes from
    # the global coordinates from alignment.

    # TODO: load in an existing transform composite so you can do local to
    # global transform any time, not just after alignment. prob need to convert
    # to a dict, save as h5, then convert back to list when you load it in.
    def local_to_global(self, img_index, x_local, y_local):
        if img_index >= len(self.transforms):
            raise ValueError("Image transform not available yet.")

        x_center, y_center, angle_deg = self.transforms[img_index]
        theta = math.radians(angle_deg)

        img_pil = self.small_imgs_pil[img_index]
        w, h = img_pil.size

        # scale
        x_s = x_local * self.scale_factor
        y_s = y_local * self.scale_factor
        # center
        cx = (w * self.scale_factor) / 2
        cy = (h * self.scale_factor) / 2
        dx = x_s - cx
        dy = y_s - cy
        # rotate
        xr = dx * math.cos(theta) - dy * math.sin(theta)
        yr = dx * math.sin(theta) + dy * math.cos(theta)
        # translate
        Xg = x_center + xr
        Yg = y_center + yr

        return float(Xg), float(Yg)


# TODO: could have user draw edge-to-edge of window and measure distance, to calc
# pixel to mm conversion so it can all be in real coordinates? probably not a good idea

def overlay_registered_images(fullimg, small_images, transforms, scale_factor=1.0):
    
    base = Image.fromarray(fullimg).copy()
    
    for img_arr, (x, y, angle) in zip(small_images, transforms):
        pil = Image.fromarray(img_arr)
        w, h = pil.size

        scaled = pil.resize((int(w * scale_factor), int(h * scale_factor)), Image.BILINEAR)
        rotated = scaled.rotate(angle, expand=True)

        # w/ alpha mask
        base.paste(rotated, (int(x - rotated.width // 2),
                             int(y - rotated.height // 2)),
                   rotated if rotated.mode == 'RGBA' else None)

    return np.array(base)



def register_tiled_locations():

    # fullimg_path = fm2p.select_file(
    #     'Choose widefield template TIF.',
    #     filetypes=[('TIF','.tif'), ('TIFF', '.tiff'), ]
    # )
    fullimg_path = '/home/dylan/Fast0/Dropbox/_temp/250929_DMM056_signmap/250929_DMM056_signmap_refimg.tif'
    fullimg = np.array(Image.open(fullimg_path))
    
    newshape = (fullimg.shape[0] // 2, fullimg.shape[1] // 2)
    zoom_factors = (
        (newshape[0]/ fullimg.shape[0]),
        (newshape[1]/ fullimg.shape[1]),
    )
    resized_fullimg = zoom(fullimg, zoom=zoom_factors, order=1)
    
    # Properly downsample using PIL (np.resize repeats/truncates data)
    # pil_full = Image.fromarray(fullimg, mode='L')


    # pil_full_small = pil_full.resize((pil_full.width // 2, pil_full.height// 2), Image.LANCZOS)
    # fullimg = np.array(pil_full_small)

    # make list of numpy arrays for each position in order. will need to load
    # in each preproc HDF file, then append
    # animalID = fm2p.get_string_input(
    #     'Animal ID to use as search key.'
    # )

    smallimgs = []
    pos_keys = []
    preproc_paths = fm2p.find('*DMM056*preproc.h5', '/home/dylan/Storage4/V1PPC_cohort02')
    preproc_paths = [p for p in preproc_paths if '251016_DMM_DMM056_pos13' not in p]
    for p in tqdm(preproc_paths):
        main_key = os.path.split(os.path.split(os.path.split(p)[0])[0])[1]
        pos_key = main_key.split('_')[-1]
        pdata = fm2p.read_h5(p)
        singlemap = pdata['twop_ref_img']
        smallimgs.append(singlemap)
        pos_keys.append(pos_key)


    aligner = ManualImageAligner(resized_fullimg, smallimgs, pos_keys, scale_factor=1.0)
    transforms = aligner.run()

    composite = overlay_registered_images(
        resized_fullimg,
        smallimgs,
        transforms,
        scale_factor=0.27)
    
    Image.fromarray(composite).save("composite_aligned_frames_v2.png")

    all_global_positions = {}

    for pi, p in tqdm(enumerate(preproc_paths)):
        main_key = os.path.split(os.path.split(os.path.split(p)[0])[0])[1]
        pos_key = main_key.split('_')[-1]
        pos = int(pos_key[-2:])
        pdata = fm2p.read_h5(p)

        cell_positions = np.zeros([len(pdata['cell_x_pix'].keys()), 4])
        for ki, k in enumerate(pdata['cell_x_pix'].keys()):
            cellx = np.median(pdata['cell_x_pix'][k])
            celly = np.median(pdata['cell_y_pix'][k])
            global_x, global_y = aligner.local_to_global(pi, cellx, celly)
            cell_positions[ki,:] = np.array([cellx, celly, global_x, global_y])

        all_global_positions[pos_key] = cell_positions

    fm2p.write_h5(r'D:\freely_moving_data\V1PPC_cohort02\DMM056_aligned_composite_local_to_global_transform_v2.h5',  all_global_positions)



if __name__ == '__main__':

    register_tiled_locations()

    