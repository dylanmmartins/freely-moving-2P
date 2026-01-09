


import os
import numpy as np
import argparse
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
    """Convert array to PIL image, handling various dtypes and dimensions."""

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
    """Normalize a mouse wheel event to a rotation step in degrees.
    
    Handles Windows (<MouseWheel> with event.delta), X11 (<Button-4/5> with event.num),
    and systems that send small fractional deltas (e.g. smooth scrolling).
    Returns a signed float (positive = rotate, negative = rotate opposite).
    """
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
    """Simple GUI for aligning a novel recording overlay to a reference image."""
    
    def __init__(self, ref_img_arr, novel_img_arr, ref_overlay_arr=None):
        """
        ref_img_arr: numpy array of reference image
        novel_img_arr: numpy array of novel/overlay image
        ref_overlay_arr: optional reference overlay to show on top of ref image
        """
        self.ref_img_arr = ref_img_arr
        self.novel_img_arr = novel_img_arr
        self.ref_overlay_arr = ref_overlay_arr
        
        # PIL conversions
        # keep reference image for canvas sizing only; we will not display it
        self.ref_img_pil = array_to_pil(ref_img_arr).convert('RGB')

        # Convert reference overlay to RGBA if provided (expected as RGBA array)
        if ref_overlay_arr is not None:
            self.ref_overlay_pil = array_to_pil(ref_overlay_arr).convert('RGBA')
        else:
            self.ref_overlay_pil = None

        # Alignment state (set early so overlay construction can use alpha)
        self.current_angle = 0.0
        self.current_offset = np.array([self.ref_img_pil.width // 2,
                                        self.ref_img_pil.height // 2], dtype=float)
        self.flipped = False
        self.alpha_value = 0.7  # Alpha for novel overlay

        # Build a colored RGBA novel overlay from novel_img_arr (could be grayscale)
        novel_arr = np.asarray(novel_img_arr)
        self.novel_overlay_pil = None
        try:
            if novel_arr.ndim == 2:
                # normalize to 0..1
                nmin = float(np.nanmin(novel_arr))
                nmax = float(np.nanmax(novel_arr))
                norm = (novel_arr - nmin) / (nmax - nmin + 1e-12)

                # apply jet colormap
                jet = cm.get_cmap('jet')(norm)  # floats 0..1

                # alpha based on norm and overall alpha_value
                alpha_scale = 0.9
                min_alpha = 0.12
                alpha_mask = np.clip(norm * alpha_scale + min_alpha, 0.0, 1.0)
                alpha_mask = (alpha_mask * self.alpha_value)
                jet[..., 3] = np.clip(alpha_mask, 0.0, 1.0)

                rgba = (jet * 255).astype(np.uint8)
                self.novel_overlay_pil = array_to_pil(rgba).convert('RGBA')
            elif novel_arr.ndim == 3 and novel_arr.shape[2] == 3:
                # RGB image: add alpha channel
                rgb = (novel_arr * 255).astype(np.uint8) if novel_arr.dtype != np.uint8 else novel_arr
                alpha = (np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.uint8) * int(255 * self.alpha_value))
                rgba = np.dstack([rgb, alpha])
                self.novel_overlay_pil = array_to_pil(rgba).convert('RGBA')
            elif novel_arr.ndim == 3 and novel_arr.shape[2] == 4:
                # Already RGBA: scale alpha
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
        # Transform to return
        self.transform = None
        
    def _create_composite_image(self):
        """Create composite of reference + novel with current transform applied."""
        # Start with a transparent canvas the same size as the reference image
        width, height = self.ref_img_pil.width, self.ref_img_pil.height
        composite = Image.new('RGBA', (width, height), (0, 0, 0, 0))

        # Paste only the reference overlay (ignore showing the reference image itself)
        ref_x = int(round(self.ref_offset[0]))
        ref_y = int(round(self.ref_offset[1]))
        if self.ref_overlay_pil is not None:
            try:
                composite.paste(self.ref_overlay_pil, (ref_x, ref_y), self.ref_overlay_pil)
            except Exception:
                composite.paste(self.ref_overlay_pil, (ref_x, ref_y))
        
        # Apply transforms to novel overlay image
        novel = self.novel_overlay_pil.copy()
        
        # Apply flip if needed
        if self.flipped:
            novel = novel.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Apply rotation
        novel = novel.rotate(self.current_angle, expand=True)
        
        # Note: novel overlay already contains per-pixel alpha; don't overwrite it here
        
        # Paste novel image centered at current_offset
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
        
        # Create canvas
        canvas = tk.Canvas(root, width=self.ref_img_pil.width, 
                          height=self.ref_img_pil.height)
        canvas.pack()
        
        # Display initial composite
        def update_display():
            # If root has been destroyed, skip updating
            try:
                if not root.winfo_exists():
                    return
            except Exception:
                pass

            composite = self._create_composite_image()
            # Tie the PhotoImage to the Tk root to avoid 'pyimageX doesn't exist'
            # and store a persistent reference on the canvas to prevent GC.
            try:
                self.current_tk = ImageTk.PhotoImage(composite, master=root)
            except TypeError:
                # Older PIL builds may not accept master; fall back.
                self.current_tk = ImageTk.PhotoImage(composite)

            if hasattr(update_display, 'img_id'):
                try:
                    canvas.delete(update_display.img_id)
                except Exception:
                    pass
            update_display.img_id = canvas.create_image(0, 0, anchor='nw', image=self.current_tk)
            # Keep a strong reference on the canvas so it's not garbage-collected
            canvas._img_ref = self.current_tk

        # reference image offset (for panning the reference + its overlay together)
        self.ref_offset = np.array([0.0, 0.0], dtype=float)
        
        # Mouse drag state
        drag_start = {'pos': None}
        
        def on_button_press(event):
            # record which button started the drag: left (1) moves novel overlay,
            # right (3) pans the reference and its overlay together
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
            root.destroy()
        
        # Bindings
        canvas.bind("<ButtonPress-1>", on_button_press)
        canvas.bind("<B1-Motion>", on_drag_motion)
        canvas.bind("<ButtonPress-3>", on_button_press)
        canvas.bind("<B3-Motion>", on_drag_motion)
        canvas.bind("<ButtonRelease-1>", lambda e: drag_start.update({'pos': None}))
        canvas.bind("<ButtonRelease-3>", lambda e: drag_start.update({'pos': None}))
        canvas.bind("<MouseWheel>", on_mouse_wheel)
        canvas.bind("<Button-4>", on_mouse_wheel)
        canvas.bind("<Button-5>", on_mouse_wheel)
        
        try:
            root.bind_all("<MouseWheel>", on_mouse_wheel)
            root.bind_all("<Button-4>", on_mouse_wheel)
            root.bind_all("<Button-5>", on_mouse_wheel)
        except Exception:
            pass
        
        # Control buttons
        btn_flip = Button(root, text="Flip", command=toggle_flip)
        btn_flip.pack(side='left', padx=5, pady=5)
        
        btn_accept = Button(root, text="Accept Alignment", command=accept_alignment)
        btn_accept.pack(side='left', padx=5, pady=5)
        
        btn_quit = Button(root, text="Quit", command=root.destroy)
        btn_quit.pack(side='left', padx=5, pady=5)
        
        # Display initial composite
        update_display()
        
        root.mainloop()
        
        return self.transform


def create_shared_ref():

    # choose a .mat and png directory of best widefield map
    wf_dir = fm2p.select_directory(
        'Select widefield output directory (should include mat files and reference tiff).'
    )
    ref_tif_path = fm2p.find('*.tif', wf_dir, MR=True)
    # am_mat_path = os.path.join(wf_dir, 'additional_maps.mat')
    vfs_mat_path = os.path.join(wf_dir, 'VFS_maps.mat')

    im = Image.open(ref_tif_path)
    img = np.array(im)

    matfile = loadmat(vfs_mat_path)

    # matfile2 = loadmat(am_mat_path)

    smlimg = zoom(img, 400 / 2048)
    smlimg = smlimg.astype(float)
    smlimg = 1-(smlimg-np.min(smlimg)) / 65535

    overlay = matfile['VFS_raw'].copy()

    # h_over = matfile2['maps']['HorizontalRetinotopy'].copy()[0][0]
    # v_over = matfile2['maps']['VerticalRetinotopy'].copy()[0][0]

    # smooth the overlay and normalize to 0..1
    overlay_img_raw = gaussian_filter(overlay, 1.5).astype(float)
    omin = np.nanmin(overlay_img_raw)
    omax = np.nanmax(overlay_img_raw)
    norm_overlay = (overlay_img_raw - omin) / (omax - omin + 1e-12)

    # build alpha ramp colormap (transparent -> opaque)
    t2b = np.zeros([256, 4])
    t2b[:, 3] = np.linspace(0, 1, 256)
    t2b = ListedColormap(t2b)

    # apply Jet colormap for RGB and use t2b for alpha
    jet_rgba = cm.get_cmap('jet')(norm_overlay)  # returns floats in 0..1
    # Make the overlay more visible by scaling alpha and enforcing a minimum
    alpha_mask = t2b(norm_overlay)[..., 3]
    alpha_scale = 0.9
    min_alpha = 0.12
    alpha_mask = np.clip(alpha_mask * alpha_scale + min_alpha, 0.0, 1.0)
    jet_rgba[..., 3] = alpha_mask

    # convert to uint8 RGBA image array
    overlay_img = (jet_rgba * 255).astype(np.uint8)

    # save out the ref img, overlay
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
    
    # Load the reference composite file with global coordinates
    ref_composite_file = fm2p.select_file(
        'Select animal reference file (with global coordinates).',
        filetypes=[('HDF', '.h5'),]
    )
    ref_data = fm2p.read_h5(ref_composite_file)
    
    # Extract reference image and overlay
    if 'refimg' not in ref_data:
        raise ValueError("Reference file must contain 'refimg' key")
    ref_img = ref_data['refimg']
    ref_overlay = ref_data.get('overlay', None)
    
    # Load the novel recording's reference image
    novel_composite_file = fm2p.select_file(
        'Select novel recording composite file (to be aligned).',
        filetypes=[('HDF', '.h5'),]
    )
    novel_data = fm2p.read_h5(novel_composite_file)

    novel_rec_dir = os.path.split(novel_composite_file)[0]
    ref_tif_path = fm2p.find('*.tif', novel_rec_dir, MR=True)
    vfs_mat_path = os.path.join(novel_rec_dir, 'VFS_maps.mat')
    
    im = Image.open(ref_tif_path)
    img = np.array(im)

    smlimg = zoom(img, 400 / 2048)
    smlimg = smlimg.astype(float)
    novel_img = 1-(smlimg-np.min(smlimg)) / 65535

    matfile = loadmat(vfs_mat_path)
    overlay = matfile['VFS_raw'].copy()

    # smooth the overlay and normalize to 0..1
    overlay_img_raw = gaussian_filter(overlay, 1.5).astype(float)
    omin = np.nanmin(overlay_img_raw)
    omax = np.nanmax(overlay_img_raw)
    norm_overlay = (overlay_img_raw - omin) / (omax - omin + 1e-12)

    # build alpha ramp colormap (transparent -> opaque)
    t2b = np.zeros([256, 4])
    t2b[:, 3] = np.linspace(0, 1, 256)
    t2b = ListedColormap(t2b)

    # apply Jet colormap for RGB and use t2b for alpha
    jet_rgba = cm.get_cmap('jet')(norm_overlay)  # returns floats in 0..1
    # Make the overlay more visible by scaling alpha and enforcing a minimum
    alpha_mask = t2b(norm_overlay)[..., 3]
    alpha_scale = 0.9
    min_alpha = 0.12
    alpha_mask = np.clip(alpha_mask * alpha_scale + min_alpha, 0.0, 1.0)
    jet_rgba[..., 3] = alpha_mask

    # convert to uint8 RGBA image array
    overlay_img = (jet_rgba * 255).astype(np.uint8)
    
    # Create alignment window - show reference image with its overlay,
    # and allow user to align the novel image on top
    # Pass the reference overlay (from ref_data) and the novel overlay (from this recording)
    aligner = AlignmentWindow(ref_img, overlay_img, ref_overlay_arr=ref_overlay)
    transform = aligner.run()
    
    if transform is None:
        print("Alignment cancelled.")
        return
    
    novel_x, novel_y, novel_angle, novel_flipped = transform
    theta = math.radians(novel_angle)
    
    # Extract cell coordinates from novel data
    # Expected structure: keys are position IDs, values are arrays with cell positions
    all_transformed_positions = {}
    
    for pos_key, cell_array in novel_data.items():
        if pos_key in ('refimg', 'overlay'):
            # Skip image data, only process coordinate arrays
            continue
        
        if not isinstance(cell_array, np.ndarray) or cell_array.size == 0:
            continue
        
        # cell_array shape: [n_cells, n_coords] where coords might be [x_local, y_local, ...]
        # We'll transform each cell's x, y coordinates
        transformed_cells = cell_array.copy()
        
        if cell_array.shape[1] >= 2:
            # Apply local-to-global transformation following logic from register_tiled_locations.py
            # Assumes first two columns are x_local, y_local
            h_novel, w_novel = novel_img.shape[:2]
            
            for cell_idx in range(cell_array.shape[0]):
                x_local = float(cell_array[cell_idx, 0])
                y_local = float(cell_array[cell_idx, 1])
                
                # Scale by reference scale if applicable
                # (in this case we assume 1:1 scale since both are composite images)
                scale_factor = 1.0
                
                x_s = x_local * scale_factor
                y_s = y_local * scale_factor
                
                # Center the image
                cx = (w_novel * scale_factor) / 2.0
                cy = (h_novel * scale_factor) / 2.0
                dx = x_s - cx
                dy = y_s - cy
                
                # Apply flip if needed
                if novel_flipped:
                    dx = -dx
                
                # Apply rotation
                xr = dx * math.cos(theta) - dy * math.sin(theta)
                yr = dx * math.sin(theta) + dy * math.cos(theta)
                
                # Translate to global position (in reference frame)
                x_global = novel_x + xr
                y_global = novel_y + yr
                
                transformed_cells[cell_idx, 0] = x_global
                transformed_cells[cell_idx, 1] = y_global
        
        all_transformed_positions[pos_key] = transformed_cells
    
    # Also include reference data so we have all recordings in one file
    for pos_key, cell_array in ref_data.items():
        if pos_key not in all_transformed_positions and pos_key not in ('refimg', 'overlay'):
            all_transformed_positions[pos_key] = cell_array
    
    # Save output with timestamp to avoid overwriting
    output_dir = os.path.dirname(novel_composite_file)
    timestamp = fm2p.fmt_now(c=True)
    output_filename = f"aligned_composite_{timestamp}.h5"
    output_path = os.path.join(output_dir, output_filename)
    
    fm2p.write_h5(output_path, all_transformed_positions)
    print(f"Alignment saved to: {output_path}")
    
    # Print transform for reference
    print(f"Applied transform - x: {novel_x:.1f}, y: {novel_y:.1f}, "
          f"angle: {novel_angle:.1f}°, flipped: {novel_flipped}")


def register_animals():

    parser = argparse.ArgumentParser()
    parser.add_argument('-cr', '--create_ref', type=fm2p.str_to_bool, default=False)
    args = parser.parse_args()


    if args.create_ref is True:
        create_shared_ref()

    elif args.create_ref is False:
        align_novel_rec_to_ref()


if __name__ == '__main__':

    register_animals()