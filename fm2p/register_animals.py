# -*- coding: utf-8 -*-
"""
Register multiple animals to one template animal.

Author: DMM, last modified Jan 2026
"""


import os
import numpy as np
import argparse
from tqdm import tqdm
import math
import tkinter as tk
from tkinter import Button
from PIL import Image, ImageTk
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, zoom
from matplotlib import cm
from matplotlib.colors import ListedColormap

import cv2
import tifffile
from fm2p.utils.vfs_composite import align_single_vfs_to_reference
from fm2p.utils.vfs_contours import (
    getMapsFromMATFile, contours_to_aligned_signmap, resize_contours
)

import fm2p





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
    """ Align novel recording overlay to a reference image.
    """
    
    def __init__(self, ref_img_arr, novel_img_arr, ref_overlay_arr=None):
        """
        ref_img_arr: numpy array of reference image
        novel_img_arr: numpy array of novel/overlay image
        ref_overlay_arr: optional reference overlay to show on top of ref image
        """

        self.ref_img_arr = ref_img_arr
        self.novel_img_arr = novel_img_arr
        self.ref_overlay_arr = ref_overlay_arr
        
        self.ref_img_pil = fm2p.array_to_pil(ref_img_arr).convert('RGB')

        if ref_overlay_arr is not None:
            self.ref_overlay_pil = fm2p.array_to_pil(ref_overlay_arr).convert('RGBA')
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
                self.novel_overlay_pil = fm2p.array_to_pil(rgba).convert('RGBA')
            elif novel_arr.ndim == 3 and novel_arr.shape[2] == 3:

                rgb = (novel_arr * 255).astype(np.uint8) if novel_arr.dtype != np.uint8 else novel_arr
                alpha = (np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.uint8) * int(255 * self.alpha_value))
                rgba = np.dstack([rgb, alpha])
                self.novel_overlay_pil = fm2p.array_to_pil(rgba).convert('RGBA')
            elif novel_arr.ndim == 3 and novel_arr.shape[2] == 4:

                rgba = novel_arr.copy()
                if rgba.dtype != np.uint8:
                    rgba = (rgba * 255).astype(np.uint8)
                rgba[..., 3] = (rgba[..., 3].astype(float) * self.alpha_value).astype(np.uint8)
                self.novel_overlay_pil = fm2p.array_to_pil(rgba).convert('RGBA')
            else:
                # ensure RGBA
                self.novel_overlay_pil = fm2p.array_to_pil(novel_img_arr).convert('RGBA')
        except Exception:
            # or make transparent placeholder
            self.novel_overlay_pil = Image.new('RGBA', (self.ref_img_pil.width, self.ref_img_pil.height), (0, 0, 0, 0))

        self.transform = None
        

    def _create_composite_image(self):
        """ Create composite of reference + novel with transform applied.
        """

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
        """ Show alignment win & ret transform.
        
        Returns as (x, y, angle, flipped)
        """
        
        root = tk.Tk()
        root.title("Align Novel Recording to Reference")
        
        canvas = tk.Canvas(root, width=self.ref_img_pil.width, 
                          height=self.ref_img_pil.height)
        canvas.pack()
        

        def update_display():
            # disp init composite
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
                # pan ref img and overlay together
                self.ref_offset += delta
                # also move novel overlay so they stay aligned.
                self.current_offset += delta
            else:
                # move only novel overlay
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
                # stop mainloop cleanly
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
        
        # run Tk event loop
        try:
            root.mainloop()
        finally:
            # destroy win after exit
            try:
                root.destroy()
            except Exception:
                pass

        return self.transform


def create_shared_ref(wf_dir=None):
    # Use my best widefield map as a template for all other animals. This makes the
    # template file from that best recording. Need tif stack and the VFS maps from
    # kevin's sign mapping code. Saves this out, then this needs to be loaded back in
    # when aligning novel animals to the template.

    if wf_dir is None:
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
    # Align novel recording to the shared ref map. Then go throguh each cell and
    # transform cell coordinates from animal to shared coordinates.
    
    ref_composite_file = fm2p.select_file(
        'Select animal reference file (with global coordinates).',
        filetypes=[('HDF', '.h5'),]
    )
    ref_data = fm2p.read_h5(ref_composite_file)
    
    if 'refimg' not in ref_data:
        ref_tif_path = fm2p.select_file(
            'Reference image not found in h5. Select a reference tif for this animal.',
            filetypes=[('TIFF files', '.tif')]
        )
        im = Image.open(ref_tif_path)
        img = np.array(im)
        smlimg = zoom(img, 400 / 2048)
        smlimg = smlimg.astype(float)
        ref_img = 1 - (smlimg - np.min(smlimg)) / 65535
    else:
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

    # smooth the overlay & norm to 0:1
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
    
    # cent of 400x400 img
    gui_center = 200.0
    
    # GUI space to full resolution. then compute offset from cent
    scale_to_full = 2048.0 / 400.0
    novel_x_full = novel_x * scale_to_full
    novel_y_full = novel_y * scale_to_full
    
    # center of img in full resolution
    full_center = 1024.0
    
    # move cells to align with the positioned overlay
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


                # index 2,3 are after the local-to-animal transform has been applied
                # index 0,1 were just the local coords
                x_local = float(cell_array[cell_idx, 2])
                y_local = float(cell_array[cell_idx, 3])
                
                # center offset w/ full resolution
                cx = w_full / 2.0
                cy = h_full / 2.0
                dx = x_local - cx
                dy = y_local - cy

                if novel_flipped:
                    dx = -dx
                
                # rotation
                xr = dx * math.cos(theta) - dy * math.sin(theta)
                yr = dx * math.sin(theta) + dy * math.cos(theta)
                
                # apply transform
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
    parser.add_argument('-wf', '--wf_dir', type=str, default=None)
    args = parser.parse_args()


    if args.create_ref is True:
        print('  -> Creating shared reference that other recordings will align to.')
        create_shared_ref(args.wf_dir)

    elif args.create_ref is False:
        print('  -> Aligning novel recording to shared reference.')
        align_novel_rec_to_ref()


def register_animals_using_shared_template():
    """Align a novel animal's recordings to the shared VFS template using
    automatic VFS-based alignment.

    Unlike register_animals() / align_novel_rec_to_ref() which requires a
    manual GUI step, this function uses sign-map cross-correlation to
    automatically find the best affine transform from the animal's widefield
    space to the reference VFS coordinate system.

    Reads
    -----
    mat_path        : additional_maps.mat for this animal
    ref_vfs_path    : mean-composite reference VFS tif (shared across animals)
    contours_path   : reference area-contours pickle (in reference VFS space)
    composite_path  : per-position per-cell array h5 (cols 2,3 = animal
                      widefield coords in 2048x2048 space)

    Saves (to output_dir, defaults to same folder as mat_path)
    -----
    vfs_aligned_composite_<timestamp>.h5
        Per-position cell arrays where cols 0,1 = animal widefield coords and
        cols 2,3 = reference VFS coords.  Same key structure as the legacy
        aligned_composite_*.h5 produced by align_novel_rec_to_ref().

    vfs_area_contours_<timestamp>.h5
        Reference area contours reverse-transformed into the animal's widefield
        (2048x2048) space.  Keys: 'contour_<area_name>' → ndarray (N,2).
        Read by merge_animal_essentials.py.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-mat', '--mat_path', type=str, default=None,
                        help='Path to additional_maps.mat file')
    parser.add_argument('-comp', '--composite_path', type=str, default=None,
                        help='Path to animal composite h5 (per-position cell arrays)')
    args = parser.parse_args()

    mat_path = args.mat_path or fm2p.select_file(
        'Select additional_maps.mat file', filetypes=[('MAT files', '.mat')])
    composite_path = args.composite_path or fm2p.select_file(
        'Select animal local-to-global composite h5 (per-position cell arrays)',
        filetypes=[('HDF5 files', '.h5')])
    output_dir = os.path.dirname(mat_path)

    _here = os.path.dirname(os.path.abspath(__file__))
    ref_vfs_path = os.path.join(_here, 'utils/vfs_mean_composite.tif')
    contours_path = os.path.join(_here, 'utils/vfs_contours.json')

    print('  -> Running VFS alignment...')
    transform_params, aligned_vfs = align_single_vfs_to_reference(
        mat_path, ref_vfs_path)
    
    print(f'  -> Alignment done.'
          f'  pearson_r={transform_params["pearson_r"]:.3f}'
          f'  dx={transform_params["dx"]:.1f}'
          f'  dy={transform_params["dy"]:.1f}'
          f'  rot={transform_params["rotation_deg"]:.1f}'
          f'  scale={transform_params["scale_factor"]:.3f}')

    # Build the 2×3 affine matrix M that maps signmap coords → reference VFS
    # coords.  Both spaces share the same pixel dimensions (ref_shape).
    ref_vfs = tifffile.imread(ref_vfs_path)
    ref_shape = ref_vfs.shape[:2]          # (H, W), typically (400, 400)
    h, w = ref_shape
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(
        center, transform_params['rotation_deg'], transform_params['scale_factor'])
    M[0, 2] += transform_params['dx']
    M[1, 2] += transform_params['dy']

    # Transform cell coordinates
    # Cell positions in the composite are in widefield space (2048×2048).
    # Scale to signmap resolution, then apply the VFS affine transform to get
    # reference VFS coordinates.
    WF_SIZE = 2048.0
    scale_to_sm = ref_shape[0] / WF_SIZE

    print('  -> Loading composite and transforming cell positions...')
    composite = fm2p.read_h5(composite_path)

    all_transformed = {}
    for pos_key, cell_array in tqdm(composite.items()):
        if not isinstance(cell_array, np.ndarray):
            continue
        if cell_array.ndim < 2 or cell_array.shape[1] < 4:
            continue

        transformed = cell_array.copy().astype(float)
        for ci in range(cell_array.shape[0]):
            # Cols 2,3 of the input composite hold animal widefield coords.
            x_wf = float(cell_array[ci, 2])
            y_wf = float(cell_array[ci, 3])

            # Scale from widefield (2048) to signmap/ref-VFS resolution.
            x_sm = x_wf * scale_to_sm
            y_sm = y_wf * scale_to_sm

            # Apply forward VFS transform: signmap space → reference VFS space.
            x_ref = M[0, 0] * x_sm + M[0, 1] * y_sm + M[0, 2]
            y_ref = M[1, 0] * x_sm + M[1, 1] * y_sm + M[1, 2]

            # Output layout (same as legacy aligned_composite format):
            #   cols 0,1 = animal widefield coords (input cols 2,3)
            #   cols 2,3 = reference VFS coords
            transformed[ci, 0] = x_wf
            transformed[ci, 1] = y_wf
            transformed[ci, 2] = x_ref
            transformed[ci, 3] = y_ref

        all_transformed[pos_key] = transformed

    # Store transform metadata as scalar arrays for later reference.
    all_transformed['_ref_vfs_shape'] = np.array(ref_shape)
    all_transformed['_transform_dx'] = np.array([transform_params['dx']])
    all_transformed['_transform_dy'] = np.array([transform_params['dy']])
    all_transformed['_transform_rotation_deg'] = np.array([transform_params['rotation_deg']])
    all_transformed['_transform_scale_factor'] = np.array([transform_params['scale_factor']])
    all_transformed['_transform_pearson_r'] = np.array([transform_params['pearson_r']])

    timestamp = fm2p.fmt_now(c=True)
    composite_savepath = os.path.join(
        output_dir, f'vfs_aligned_composite_{timestamp}.h5')
    fm2p.write_h5(composite_savepath, all_transformed)
    print(f'  -> Saved aligned composite: {composite_savepath}')

    # Area contours in animal widefield space
    # Reverse-transform the reference contours into the animal's widefield space
    # so that merge_animal_essentials can assign area labels using the local
    # (animal widefield) cell coordinates (cols 0,1 of the composite).
    print('  -> Computing area contours in animal widefield space...')
    _, _, signMapf = getMapsFromMATFile(mat_path, filter=False)
    reverse_contours = contours_to_aligned_signmap(
        contours_path, transform_params, ref_vfs_path)
    reverse_contours_resized = resize_contours(
        reverse_contours,
        source_shape=signMapf.shape,
        target_shape=(int(WF_SIZE), int(WF_SIZE)))

    contours_out = {'_ref_vfs_shape': np.array(ref_shape)}
    for area_name, coords in reverse_contours_resized.items():
        if coords is not None and len(coords) >= 3:
            contours_out[f'contour_{area_name}'] = np.array(coords)

    contours_savepath = os.path.join(
        output_dir, f'vfs_area_contours_{timestamp}.h5')
    fm2p.write_h5(contours_savepath, contours_out)
    print(f'  -> Saved area contours: {contours_savepath}')


if __name__ == '__main__':

    register_animals_using_shared_template()

