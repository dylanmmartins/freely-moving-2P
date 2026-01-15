# -*- coding: utf-8 -*-


import os
import numpy as np
import argparse
from tqdm import tqdm
import math
import tkinter as tk
from tkinter import Button, messagebox
from PIL import Image, ImageTk
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, zoom
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

import fm2p


def array_to_pil(arr):
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
        return Image.fromarray(a, mode='L')
    if a.ndim == 3 and a.shape[2] == 3:
        return Image.fromarray(a, mode='RGB')
    if a.ndim == 3 and a.shape[2] == 4:
        return Image.fromarray(a, mode='RGBA')
    return Image.fromarray(a)


def _wheel_step_from_event(event, scale=2.0):
    delta = getattr(event, 'delta', None)
    if delta is not None and delta != 0:
        try:
            d = float(delta)
            if abs(d) >= 1.0:
                return math.copysign(scale, d)
            return math.copysign(scale, d)
        except Exception:
            return 0.0
    num = getattr(event, 'num', None)
    if num == 4:
        return float(scale)
    if num == 5:
        return float(-scale)

    return 0.0


class AlignmentWindow:
    """Aligning a novel recording overlay to a reference image."""
    
    def __init__(self, ref_img_arr, novel_img_arr, ref_overlay_arr=None):
        """
        ref_img_arr: numpy array of reference image
        novel_img_arr: numpy array of novel/overlay image
        ref_overlay_arr: optional reference overlay to show on top of ref image
        """

        self.ref_img_arr = ref_img_arr
        self.novel_img_arr = novel_img_arr
        self.ref_overlay_arr = ref_overlay_arr
        
        self.ref_img_pil = array_to_pil(ref_img_arr).convert('RGB')

        if ref_overlay_arr is not None:
            self.ref_overlay_pil = array_to_pil(ref_overlay_arr).convert('RGBA')
        else:
            self.ref_overlay_pil = None

        self.current_angle = 0.0
        self.current_offset = np.array([self.ref_img_pil.width // 2,
                                        self.ref_img_pil.height // 2], dtype=float)
        self.flipped = False
        self.alpha_value = 0.7 # for novel overlay

        novel_arr = np.asarray(novel_img_arr)
        self.novel_overlay_pil = None
        try:
            if novel_arr.ndim == 2:
                nmin = float(np.nanmin(novel_arr))
                nmax = float(np.nanmax(novel_arr))
                norm = (novel_arr - nmin) / (nmax - nmin + 1e-12)

                jet = cm.get_cmap('jet')(norm)

                alpha_scale = 0.9
                min_alpha = 0.12
                alpha_mask = np.clip(norm * alpha_scale + min_alpha, 0.0, 1.0)
                alpha_mask = (alpha_mask * self.alpha_value)
                jet[..., 3] = np.clip(alpha_mask, 0.0, 1.0)

                rgba = (jet * 255).astype(np.uint8)
                self.novel_overlay_pil = array_to_pil(rgba).convert('RGBA')
            elif novel_arr.ndim == 3 and novel_arr.shape[2] == 3:

                rgb = (novel_arr * 255).astype(np.uint8) if novel_arr.dtype != np.uint8 else novel_arr
                alpha = (np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.uint8) * int(255 * self.alpha_value))
                rgba = np.dstack([rgb, alpha])
                self.novel_overlay_pil = array_to_pil(rgba).convert('RGBA')
            elif novel_arr.ndim == 3 and novel_arr.shape[2] == 4:

                rgba = novel_arr.copy()
                if rgba.dtype != np.uint8:
                    rgba = (rgba * 255).astype(np.uint8)
                rgba[..., 3] = (rgba[..., 3].astype(float) * self.alpha_value).astype(np.uint8)
                self.novel_overlay_pil = array_to_pil(rgba).convert('RGBA')
            else:
                # fallback: convert and ensure RGBA
                self.novel_overlay_pil = array_to_pil(novel_img_arr).convert('RGBA')
        except Exception:
            # on any failure, create a transparent placeholder
            self.novel_overlay_pil = Image.new('RGBA', (self.ref_img_pil.width, self.ref_img_pil.height), (0, 0, 0, 0))

        # the transform that will be returned
        self.transform = None
        
    def _create_composite_image(self):
        """Create composite of reference + novel with current transform applied."""

        width, height = self.ref_img_pil.width, self.ref_img_pil.height
        composite = Image.new('RGBA', (width, height), (0, 0, 0, 0))

        # only show ref overlay
        ref_x = int(round(self.ref_offset[0]))
        ref_y = int(round(self.ref_offset[1]))
        if self.ref_overlay_pil is not None:
            try:
                composite.paste(self.ref_overlay_pil, (ref_x, ref_y), self.ref_overlay_pil)
            except Exception:
                composite.paste(self.ref_overlay_pil, (ref_x, ref_y))
        
        novel = self.novel_overlay_pil.copy()
        
        if self.flipped:
            novel = novel.transpose(Image.FLIP_LEFT_RIGHT)
        
        novel = novel.rotate(self.current_angle, expand=True)
        
        paste_x = int(self.current_offset[0] - novel.width // 2)
        paste_y = int(self.current_offset[1] - novel.height // 2)
        try:
            composite.paste(novel, (paste_x, paste_y), novel)
        except Exception:
            composite.paste(novel, (paste_x, paste_y))
        
        return composite
    
    def run(self):
        """Show the alignment window and return the transform (x, y, angle, flipped)."""
        
        root = tk.Tk()
        root.title("Align Novel Recording to Reference")
        
        canvas = tk.Canvas(root, width=self.ref_img_pil.width, 
                          height=self.ref_img_pil.height)
        canvas.pack()
        
        # disp init composite
        def update_display():
            # if root destroyed, skip update
            try:
                if not root.winfo_exists():
                    return
            except Exception:
                pass

            composite = self._create_composite_image()
            # store persistent reference on the canvas to stop if from beign gc'ed.
            try:
                self.current_tk = ImageTk.PhotoImage(composite, master=root)
            except TypeError:
                # fall bacjk
                self.current_tk = ImageTk.PhotoImage(composite)

            if hasattr(update_display, 'img_id'):
                try:
                    canvas.delete(update_display.img_id)
                except Exception:
                    pass
            update_display.img_id = canvas.create_image(0, 0, anchor='nw', image=self.current_tk)
            # keep ref on the canvas so it's not gc'ed
            canvas._img_ref = self.current_tk

        # ref image offset (panning reference + overlay)
        self.ref_offset = np.array([0.0, 0.0], dtype=float)
        
        drag_start = {'pos': None}
        
        def on_button_press(event):

            drag_start['pos'] = np.array([event.x, event.y], dtype=float)
            drag_start['mode'] = 'novel' if getattr(event, 'num', 1) == 1 else 'ref'
        
        def on_drag_motion(event):
            if drag_start.get('pos') is None:
                drag_start['pos'] = np.array([event.x, event.y], dtype=float)
                drag_start['mode'] = drag_start.get('mode', 'novel')

            current = np.array([event.x, event.y], dtype=float)
            delta = current - drag_start['pos']

            if drag_start.get('mode') == 'ref':
                # pan the reference image and its overlay together
                self.ref_offset += delta
                # also move the novel overlay (visual) with the reference so they stay aligned
                self.current_offset += delta
            else:
                # move only the novel overlay
                self.current_offset += delta

            drag_start['pos'] = current
            update_display()
        
        def on_button_release(event):

            drag_start.update({'pos': None})

            try:
                print(
                    f"Transform tested at -> x: {self.current_offset[0]-200:.1f} px, "
                    f"y: {self.current_offset[1]-200:.1f} px, "
                    f"angle: {self.current_angle:.1f} deg, "
                    f"flipped: {self.flipped}"
                )
            except Exception:
                print("Test transform -> (unable to read current transform)")

        def on_mouse_wheel(event):
            step = _wheel_step_from_event(event, scale=2.0)
            if step != 0.0:
                self.current_angle = (self.current_angle + step) % 360.0
                update_display()
        
        def toggle_flip():
            self.flipped = not self.flipped
            btn_flip.config(relief='sunken' if self.flipped else 'raised')
            update_display()
        
        def accept_alignment():
            self.transform = (
                float(self.current_offset[0]),
                float(self.current_offset[1]),
                float(self.current_angle),
                bool(self.flipped),
            )

            # float(self.current_offset[0] + self.ref_offset[0]), # x
            # float(self.current_offset[1] + self.ref_offset[1]), # y

            try:
                # stop the mainloop cleanly
                root.quit()
            except Exception:
                try:
                    root.destroy()
                except Exception:
                    pass
    
        canvas.bind("<ButtonPress-1>", on_button_press)
        canvas.bind("<B1-Motion>", on_drag_motion)
        canvas.bind("<ButtonPress-3>", on_button_press)
        canvas.bind("<B3-Motion>", on_drag_motion)
        canvas.bind("<ButtonRelease-1>", on_button_release)
        canvas.bind("<ButtonRelease-3>", on_button_release)
        canvas.bind("<MouseWheel>", on_mouse_wheel)
        canvas.bind("<Button-4>", on_mouse_wheel)
        canvas.bind("<Button-5>", on_mouse_wheel)
        
        try:
            root.bind_all("<MouseWheel>", on_mouse_wheel)
            root.bind_all("<Button-4>", on_mouse_wheel)
            root.bind_all("<Button-5>", on_mouse_wheel)
        except Exception:
            pass
        
        btn_flip = Button(root, text="Flip", command=toggle_flip)
        btn_flip.pack(side='left', padx=5, pady=5)
        
        btn_accept = Button(root, text="Accept Alignment", command=accept_alignment)
        btn_accept.pack(side='left', padx=5, pady=5)
        
        btn_quit = Button(root, text="Quit", command=lambda: root.quit())
        btn_quit.pack(side='left', padx=5, pady=5)
        
        # disp init composite
        update_display()
        
        # run the Tk event loop; `accept_alignment` will call `root.quit()` to exit
        try:
            root.mainloop()
        finally:
            # ensure the window is destroyed after mainloop exits
            try:
                root.destroy()
            except Exception:
                pass

        return self.transform


def create_shared_ref():

    wf_dir = fm2p.select_directory(
        'Select widefield output directory (should include mat files and reference tiff).'
    )
    ref_tif_path = fm2p.find('*.tif', wf_dir, MR=True)
    # am_mat_path = os.path.join(wf_dir, 'additional_maps.mat')
    vfs_mat_path = os.path.join(wf_dir, 'VFS_maps.mat')

    im = Image.open(ref_tif_path)
    img = np.array(im)

    matfile = loadmat(vfs_mat_path)

    smlimg = zoom(img, 400 / 2048)
    smlimg = smlimg.astype(float)
    smlimg = 1-(smlimg-np.min(smlimg)) / 65535

    overlay = matfile['VFS_raw'].copy()

    # h_over = matfile2['maps']['HorizontalRetinotopy'].copy()[0][0]
    # v_over = matfile2['maps']['VerticalRetinotopy'].copy()[0][0]

    overlay_img_raw = gaussian_filter(overlay, 1.5).astype(float)
    omin = np.nanmin(overlay_img_raw)
    omax = np.nanmax(overlay_img_raw)
    norm_overlay = (overlay_img_raw - omin) / (omax - omin + 1e-12)

    t2b = np.zeros([256, 4])
    t2b[:, 3] = np.linspace(0, 1, 256)
    t2b = ListedColormap(t2b)

    jet_rgba = cm.get_cmap('jet')(norm_overlay)

    alpha_mask = t2b(norm_overlay)[..., 3]
    alpha_scale = 0.9
    min_alpha = 0.12
    alpha_mask = np.clip(alpha_mask * alpha_scale + min_alpha, 0.0, 1.0)
    jet_rgba[..., 3] = alpha_mask

    overlay_img = (jet_rgba * 255).astype(np.uint8)

    out = {
        'overlay': overlay_img,
        'refimg': smlimg
    }

    savepath = os.path.join(wf_dir, 'animal_reference_{}.h5'.format(fm2p.fmt_now(c=True)))
    fm2p.write_h5(savepath, out)


def align_novel_rec_to_ref():
    """Align a novel recording to a shared reference and transform cell coordinates.
    
    This function:
    1. Loads a reference recording with its reference image and overlay
    2. Loads a novel recording's reference image
    3. Allows user to manually align the novel overlay to the reference image
    4. Applies the global-to-global transformation to all cell coordinates
    5. Saves the transformed data to a new HDF file with timestamp
    """
    
    ref_composite_file = fm2p.select_file(
        'Select animal reference file (with global coordinates).',
        filetypes=[('HDF', '.h5'),]
    )
    ref_data = fm2p.read_h5(ref_composite_file)
    
    if 'refimg' not in ref_data:
        raise ValueError("Reference file must contain 'refimg' key")
    ref_img = ref_data['refimg']
    ref_overlay = ref_data.get('overlay', None)
    
    novel_composite_file = fm2p.select_file(
        'Select novel recording composite file (to be aligned).',
        filetypes=[('HDF', '.h5'),]
    )
    novel_data = fm2p.read_h5(novel_composite_file)

    novel_rec_dir = os.path.split(novel_composite_file)[0]
    ref_tif_path = fm2p.find('*.tif', novel_rec_dir, MR=True)
    vfs_mat_path = os.path.join(novel_rec_dir, 'VFS_maps.mat')

    print('  -> Loading novel images.')
    
    im = Image.open(ref_tif_path)
    img = np.array(im)

    smlimg = zoom(img, 400 / 2048)
    smlimg = smlimg.astype(float)
    novel_img = 1-(smlimg-np.min(smlimg)) / 65535

    matfile = loadmat(vfs_mat_path)
    overlay = matfile['VFS_raw'].copy()

    # smooth the overlay and normalize to 0:1
    overlay_img_raw = gaussian_filter(overlay, 1.5).astype(float)
    omin = np.nanmin(overlay_img_raw)
    omax = np.nanmax(overlay_img_raw)
    norm_overlay = (overlay_img_raw - omin) / (omax - omin + 1e-12)

    t2b = np.zeros([256, 4])
    t2b[:, 3] = np.linspace(0, 1, 256)
    t2b = ListedColormap(t2b)

    jet_rgba = cm.get_cmap('jet')(norm_overlay)
    alpha_mask = t2b(norm_overlay)[..., 3]
    alpha_scale = 0.9
    min_alpha = 0.12
    alpha_mask = np.clip(alpha_mask * alpha_scale + min_alpha, 0.0, 1.0)
    jet_rgba[..., 3] = alpha_mask
    overlay_img = (jet_rgba * 255).astype(np.uint8)
    
    print('  -> Launching alignment GUI.')
    aligner = AlignmentWindow(ref_img, overlay_img, ref_overlay_arr=ref_overlay)

    print('  -> Waiting for user to complete alignment...')
    transform = aligner.run()
    
    if transform is None:
        print("Alignment cancelled.")
        return
    
    print('  -> Transform computed, applying to cell coordinates')
    novel_x, novel_y, novel_angle, novel_flipped = transform

    theta = math.radians(novel_angle)
    
    # The transform gives us where the novel overlay was positioned in GUI space (400 pixels)
    # We need to convert this to full resolution and invert it for cell application
    # Center of 400-pixel GUI image is at (200, 200)
    gui_center = 200.0
    
    # Convert from GUI space to full resolution space and compute offset from expected center
    scale_to_full = 2048.0 / 400.0
    novel_x_full = novel_x * scale_to_full
    novel_y_full = novel_y * scale_to_full
    
    # Expected center in full resolution
    full_center = 1024.0
    
    # The offset needed to move cells to align with the positioned overlay
    offset_x = novel_x_full - full_center
    offset_y = novel_y_full - full_center
    
    all_transformed_positions = {}
    
    for pos_key, cell_array in tqdm(novel_data.items()):
        if pos_key in ('refimg', 'overlay'):
            continue
        
        if not isinstance(cell_array, np.ndarray) or cell_array.size == 0:
            continue
        
        transformed_cells = cell_array.copy()
        
        if cell_array.shape[1] >= 2:
            h_novel, w_novel = novel_img.shape[:2]
            h_full = int(h_novel * 2048.0 / 400.0)
            w_full = int(w_novel * 2048.0 / 400.0)
            
            for cell_idx in range(cell_array.shape[0]):
                x_local = float(cell_array[cell_idx, 2])
                y_local = float(cell_array[cell_idx, 3])
                
                # Center offset in full resolution
                cx = w_full / 2.0
                cy = h_full / 2.0
                dx = x_local - cx
                dy = y_local - cy

                if novel_flipped:
                    dx = -dx
                
                # Apply rotation
                xr = dx * math.cos(theta) - dy * math.sin(theta)
                yr = dx * math.sin(theta) + dy * math.cos(theta)
                
                # Apply transform offset
                x_global = xr + cx + offset_x
                y_global = yr + cy + offset_y
                
                transformed_cells[cell_idx, 0] = x_local
                transformed_cells[cell_idx, 1] = y_local

                transformed_cells[cell_idx, 2] = x_global
                transformed_cells[cell_idx, 3] = y_global
        
        all_transformed_positions[pos_key] = transformed_cells
    
    print('  -> Adding reference overlay and image data to output')

    for pos_key, cell_array in ref_data.items():
        if pos_key not in all_transformed_positions and pos_key not in ('refimg', 'overlay'):
            all_transformed_positions[pos_key] = cell_array
    
    output_dir = os.path.dirname(novel_composite_file)
    timestamp = fm2p.fmt_now(c=True)
    output_filename = f"aligned_composite_{timestamp}.h5"
    output_path = os.path.join(output_dir, output_filename)
    
    fm2p.write_h5(output_path, all_transformed_positions)
    print(f"Alignment saved to: {output_path}")
    
    print(f"Applied transform - x_offset: {offset_x:.1f} px, y_offset: {offset_y:.1f} px, "
          f"angle: {novel_angle:.1f}deg, flipped: {novel_flipped}")


def register_animals():

    parser = argparse.ArgumentParser()
    parser.add_argument('-cr', '--create_ref', type=fm2p.str_to_bool, default=False)
    args = parser.parse_args()


    if args.create_ref is True:
        print('  -> Creating shared reference that other recordings will align to.')
        create_shared_ref()

    elif args.create_ref is False:
        print('  -> Aligning novel recording to shared reference.')
        align_novel_rec_to_ref()


if __name__ == '__main__':

    register_animals()