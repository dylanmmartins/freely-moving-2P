# -*- coding: utf-8 -*-
"""
Register tiled locations to a full animal vasculature from widefield image.

Author: DMM, last modified Jan 2026
"""


import math
import os
from tqdm import tqdm
import tkinter as tk
from tkinter import Button, HORIZONTAL, Scale
import numpy as np
from PIL import Image, ImageTk
import random
from scipy.ndimage import zoom

import fm2p


class ManualImageAligner:
    def __init__(self, fullimg, small_images, position_keys, scale_factor=1.0):
        """
        fullimg: numpy array
        small_images: list of numpy arrays
        scale_factor: initial scale factor for small images
        """
        self.fullimg_arr = fullimg

        self.fullimg_pil = fm2p.array_to_pil(fullimg).convert('RGBA')
        self.base_image = self.fullimg_pil.copy()
        self.position_keys = position_keys

        self.small_imgs_arr = small_images
        self.small_imgs_pil = [fm2p.array_to_pil(img).convert('RGBA') for img in small_images]

        self.scale_factor = scale_factor

        # per-image horizontal flip flags
        self.flipped_flags = [False] * len(self.small_imgs_pil)

        # stored as [x_center, y_center, angle_degrees]
        self.transforms = []

        self.index = 0
        self.current_angle = 0
        self.current_offset = np.array([50, 50], float)
        self.debug_wheel = False


    def _wheel_step_from_event(self, event, scale=2.0):
        """ Norm mouse wheel event to rotation step in deg.
        """
        delta = getattr(event, 'delta', None)
        if delta is not None and delta != 0:
            try:
                d = float(delta)
                if abs(d) >= 1.0:
                    return (d / 120.0) * scale # 120 bc windows
                # small fractionals
                return math.copysign(scale, d)
            except Exception:
                return 0.0

        num = getattr(event, 'num', None)
        if num == 4: # up
            return float(scale)
        if num == 5: # DOwn
            return float(-scale)

        return 0.0


    def choose_scale_factor(self, fullimg_arr, small_images, start_idx=0):
        """ Resize single tile overalid on full WF img, then save out scale factor
        """
        full_pil = fm2p.array_to_pil(fullimg_arr)
        full_rgba = full_pil.convert('RGBA')

        self.preview_idx = int(start_idx)
        print('Displaying stitching position: {}'.format(
            self.position_keys[self.preview_idx])
        )
        small_pil_holder = {
            'pil': fm2p.array_to_pil(small_images[self.preview_idx]),
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
            composite_tk = ImageTk.PhotoImage(composite, master=win)

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
            small_pil_holder['offset'] = small_pil_holder.get(
                'offset', np.array([0.0, 0.0])
            ) + delta
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
            win, from_=0.05, to=1.5, resolution=0.01, # started as ~3 or something like that.
            # it's usually 0.27, so i should change it to have max of 0.5. Need to test first.
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
            while new_idx == self.preview_idx and tries < 10: # why'd I create a limit on tries?
                new_idx = random.randrange(len(small_images))
                tries += 1
            self.preview_idx = new_idx
            small_pil_holder['pil'] = fm2p.array_to_pil(small_images[self.preview_idx])
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
        self.base_tk = ImageTk.PhotoImage(self.base_image, master=self.root)
        self.base_canvas_id = self.canvas.create_image(0, 0, anchor="nw", image=self.base_tk)

        self.btn_accept = Button(self.root, text="", command=self.accept_alignment)
        self.btn_accept.pack()

        def update_accept_button_text():
            try:
                if self.index <= 0:
                    self.btn_accept.config(text="Place First Tile")
                else:
                    self.btn_accept.config(text="Accept Alignment and Place Next")
            except Exception:
                pass

        # expose as attribute for other methods to call
        self.update_accept_button_text = update_accept_button_text

        self.btn_flip = Button(self.root, text="Flip", command=self.toggle_flip)
        self.btn_flip.pack()

        self.btn_quit = Button(self.root, text="Quit", command=self.root.destroy)
        self.btn_quit.pack()

        # bindings
        self.canvas.bind("<ButtonPress-1>", self.start_move)
        self.canvas.bind("<B1-Motion>", self.move_image)
        # rotation bindinsg
        self.canvas.bind("<MouseWheel>", self.rotate_image)
        self.canvas.bind("<Button-4>", self.rotate_image)
        self.canvas.bind("<Button-5>", self.rotate_image)

        # also bind application-wide so wheel events are caught even
        # when the canvas doesn't have focus (fix for linux)
        try:
            self.root.bind_all("<MouseWheel>", self.rotate_image)
            self.root.bind_all("<Button-4>", self.rotate_image)
            self.root.bind_all("<Button-5>", self.rotate_image)
        except Exception:
            # fallback to widget-level binds if bind_all not supported
            try:
                self.root.bind("<MouseWheel>", self.rotate_image)
                self.root.bind("<Button-4>", self.rotate_image)
                self.root.bind("<Button-5>", self.rotate_image)
            except Exception:
                pass


    def load_small_image(self):
        
        if self.index >= len(self.small_imgs_pil):
            return

        img = self.small_imgs_pil[self.index]
        w, h = img.size

        scaled = img.resize((int(w*self.scale_factor), int(h*self.scale_factor)),
                    Image.BILINEAR)

        # apply horizontal flip if user toggled it for this tile
        if self.flipped_flags[self.index]:
            scaled = scaled.transpose(Image.FLIP_LEFT_RIGHT)

        rotated = scaled.rotate(self.current_angle, expand=True)

        self.current_pil = rotated
        self.current_tk = ImageTk.PhotoImage(rotated, master=self.root)


        print('Drawing image from position {}'.format(self.position_keys[self.index]))
        self.draw_small_image()
        # update accept button text to reflect first/next tile state
        # TODO: the first button is blank / has no label, then it appears once you
        # place the first one. how to fix this?
        try:
            if hasattr(self, 'update_accept_button_text'):
                self.update_accept_button_text()
        except Exception:
            pass


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
        # normalize wheel event to rotation step
        step = self._wheel_step_from_event(event, scale=2.0)
        if self.debug_wheel:
            print(f"rotate_image event: delta={getattr(event,'delta',None)} num={getattr(event,'num',None)} -> step={step}")

        if step != 0.0:
            self.current_angle = (self.current_angle + step) % 360.0
            self.load_small_image()


    def toggle_flip(self):
        # toggle the flip flag for the current tile and redraw
        if self.index < len(self.flipped_flags):
            self.flipped_flags[self.index] = not self.flipped_flags[self.index]
            print(f"Tile {self.index} flipped: {self.flipped_flags[self.index]}")
            self.load_small_image()


    def accept_alignment(self):
        # paste the sml img into WF img
        # record transform: x_center, y_center, angle_degrees, flipped_flag
        flipped_flag = bool(self.flipped_flags[self.index]) if self.index < len(self.flipped_flags) else False
        self.transforms.append((
            float(self.current_offset[0]),
            float(self.current_offset[1]),
            float(self.current_angle),
            flipped_flag,
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

            self.base_tk = ImageTk.PhotoImage(display_image, master=self.root)
            self.canvas.itemconfig(self.base_canvas_id, image=self.base_tk)

        # adv to next image
        self.index += 1
        if self.index >= len(self.small_imgs_pil):
            print("All images aligned.")
            self.root.quit()
            return

        self.current_angle = 0
        self.current_offset = np.array([50, 50], float)
        # update accept button text for subsequent tiles
        try:
            if hasattr(self, 'update_accept_button_text'):
                self.update_accept_button_text()
        except Exception:
            pass

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

        # Close the alignment window before opening fine-tune window
        self.root.destroy()

        # After per-tile alignment, offer a composite fine-tune stage
        try:
            self.fine_tune_transforms()
        except Exception as e:
            print('Fine-tune stage failed or was cancelled:', e)

        return self.transforms


    def fine_tune_transforms(self):
        """Open a window showing all tiles together for fine adjustments.

        - Click a tile to select it
        - Drag to move selected tile (or all tiles if "Move All" is active)
        - Mouse wheel to rotate selected tile
        - Flip button to toggle horizontal flip on selected tile
        - Accept button to save adjustments into self.transforms
        """
        if len(self.transforms) == 0:
            print('No transforms to fine-tune.')
            return

        win = tk.Toplevel()
        win.title('Composite Fine-tune')

        canvas = tk.Canvas(win, width=self.fullimg_pil.width, height=self.fullimg_pil.height)
        canvas.pack()

        # base composite image (without tiles) - start from the original full image
        # use original fullimg so we don't double-draw tiles that were pasted
        base_img = self.fullimg_pil.copy()
        base_tk = ImageTk.PhotoImage(base_img, master=win)
        base_id = canvas.create_image(0, 0, anchor='nw', image=base_tk)

        # state for each tile: keep current angle, flipped, center coords, tk image
        tile_state = []

        for idx, img_pil in enumerate(self.small_imgs_pil):
            # get transform tuple
            t = self.transforms[idx] if idx < len(self.transforms) else (50.0, 50.0, 0.0, False)
            if len(t) >= 4:
                x, y, angle, flipped = t[0], t[1], t[2], bool(t[3])
            else:
                x, y, angle = t
                flipped = False

            w, h = img_pil.size
            scaled = img_pil.resize((int(w * self.scale_factor), int(h * self.scale_factor)), Image.BILINEAR)
            if flipped:
                scaled = scaled.transpose(Image.FLIP_LEFT_RIGHT)
            rotated = scaled.rotate(angle, expand=True)

            tkimg = ImageTk.PhotoImage(rotated, master=win)
            cid = canvas.create_image(int(x), int(y), anchor='center', image=tkimg)

            tile_state.append({
                'cid': cid,
                'angle': float(angle),
                'flipped': bool(flipped),
                'x': float(x),
                'y': float(y),
                'tkimg': tkimg,
                'base_pil': img_pil,
                'w_rot': rotated.width,
                'h_rot': rotated.height,
            })
            

        # keep references so they are not GC'd
        win._tile_state = tile_state
        win._base_tk = base_tk

        selected = {'idx': None}
        move_all = {'active': False}

        # highlight rectangle + label for selected tile
        sel_visual = {'rect': None, 'label': None}

        drag = {'start': None, 'offset': (0, 0)}

        def update_selection_visual():
            # remove old visuals
            if sel_visual['rect'] is not None:
                try:
                    canvas.delete(sel_visual['rect'])
                except Exception:
                    pass
                sel_visual['rect'] = None
            if sel_visual['label'] is not None:
                try:
                    canvas.delete(sel_visual['label'])
                except Exception:
                    pass
                sel_visual['label'] = None

            idx = selected['idx']
            if idx is None:
                return
            s =   tile_state[idx]
            w =  s.get('w_rot', 0)
            h =  s.get('h_rot', 0)
            x =  s['x']
            y =  s['y']
            x0 = int(x - w/2)
            y0 = int(y - h/2)
            x1 = int(x + w/2)
            y1 = int(y + h/2)

            # draw rectangle and a small label with index/key
            sel_visual['rect'] = canvas.create_rectangle(x0, y0, x1, y1, outline='red', width=2)
            label_text = f"{idx}: {self.position_keys[idx] if idx < len(self.position_keys) else ''}"
            sel_visual['label'] = canvas.create_text(x0+5, y0+10, anchor='nw', text=label_text, fill='yellow', font=('TkDefaultFont', 10, 'bold'))

        def find_tile_at(x, y):
            hits = canvas.find_overlapping(x, y, x, y)
            # find topmost tile id that is not base
            for h in reversed(hits):
                for i, s in enumerate(tile_state):
                    if s['cid'] == h:
                        return i
            return None

        def on_click(event):
            idx = find_tile_at(event.x, event.y)
            selected['idx'] = idx
            # bring selected on top
            if idx is not None:
                cid = tile_state[idx]['cid']
                canvas.tag_raise(cid)
            update_selection_visual()

            # prepare drag offset so tile doesn't jump when user starts dragging
            if idx is not None:
                drag['start'] = (event.x, event.y)
                s = tile_state[idx]
                # offset mouse - tile_center so during drag tile remains under cursor
                drag['offset'] = (event.x - s['x'], event.y - s['y'])

        def on_start_drag(event):
            # generic start drag for middle/right buttons
            drag['start'] = (event.x, event.y)
            # if a tile is selected, compute offset so movement is relative to cursor
            idx = selected['idx']
            if idx is not None:
                s = tile_state[idx]
                drag['offset'] = (event.x - s['x'], event.y - s['y'])

        def on_drag(event):
            # If moving all, use delta motion (so group moves smoothly)
            if drag['start'] is None:
                drag['start'] = (event.x, event.y)
            dx = event.x - drag['start'][0]
            dy = event.y - drag['start'][1]
            drag['start'] = (event.x, event.y)

            if move_all['active']:
                # move every tile by delta
                for s in tile_state:
                    s['x'] += dx
                    s['y'] += dy
                    canvas.coords(s['cid'], int(s['x']), int(s['y']))
                update_selection_visual()
            else:
                idx = selected['idx']
                if idx is None:
                    return
                s = tile_state[idx]
                # use stored offset so the tile remains under the cursor position
                offx, offy = drag.get('offset', (0, 0))
                s['x'] = event.x - offx
                s['y'] = event.y - offy
                canvas.coords(s['cid'], int(s['x']), int(s['y']))
                # update selection rectangle if the moved tile is selected
                update_selection_visual()

        def on_wheel(event):
            # rotate selected tile
            idx = selected['idx']
            if idx is None:
                return

            step = self._wheel_step_from_event(event, scale=2.0)
            if self.debug_wheel:
                print(f"fine-tune on_wheel event: delta={getattr(event,'delta',None)} num={getattr(event,'num',None)} -> step={step}")

            if step == 0.0:
                return

            s = tile_state[idx]
            s['angle'] = (s['angle'] + step) % 360.0
            # recreate image
            pil = s['base_pil']
            w, h = pil.size
            scaled = pil.resize((int(w * self.scale_factor), int(h * self.scale_factor)), Image.BILINEAR)
            if s['flipped']:
                scaled = scaled.transpose(Image.FLIP_LEFT_RIGHT)
            rotated = scaled.rotate(s['angle'], expand=True)
            s['tkimg'] = ImageTk.PhotoImage(rotated, master=win)
            canvas.itemconfig(s['cid'], image=s['tkimg'])
            # keep reference
            win._tile_state[idx] = s

        def toggle_move_all():
            move_all['active'] = not move_all['active']
            btn_move_all.config(relief='sunken' if move_all['active'] else 'raised')
            # clear selection visual when switching to move-all
            update_selection_visual()


        def flip_selected():
            idx = selected['idx']
            if idx is None:
                return
            s = tile_state[idx]
            s['flipped'] = not s['flipped']
            pil = s['base_pil']
            w, h = pil.size
            scaled = pil.resize((int(w * self.scale_factor), int(h * self.scale_factor)), Image.BILINEAR)
            if s['flipped']:
                scaled = scaled.transpose(Image.FLIP_LEFT_RIGHT)
            rotated = scaled.rotate(s['angle'], expand=True)
            s['tkimg'] = ImageTk.PhotoImage(rotated, master=win)
            canvas.itemconfig(s['cid'], image=s['tkimg'])
            win._tile_state[idx] = s
            # update visual sizes stored
            tile_state[idx] = s
            # update stored rotated sizes for selection rectangle
            tile_state[idx]['w_rot'] = rotated.width
            tile_state[idx]['h_rot'] = rotated.height
            update_selection_visual()

        def accept():
            # write back to self.transforms
            for i, s in enumerate(tile_state):
                # update existing transform entries
                if i < len(self.transforms):
                    t = self.transforms[i]
                    if len(t) >= 4:
                        self.transforms[i] = (float(s['x']), float(s['y']), float(s['angle']), bool(s['flipped']))
                    else:
                        self.transforms[i] = (float(s['x']), float(s['y']), float(s['angle']), bool(s['flipped']))
                else:
                    self.transforms.append((float(s['x']), float(s['y']), float(s['angle']), bool(s['flipped'])))

            win.quit()

        canvas.bind('<ButtonPress-1>', on_click)
        canvas.bind('<ButtonPress-3>', on_start_drag)
        canvas.bind('<B3-Motion>', on_drag)
        canvas.bind('<ButtonPress-2>', on_start_drag)
        canvas.bind('<B2-Motion>', on_drag)
        canvas.bind('<B1-Motion>', on_drag)
        canvas.bind('<MouseWheel>', on_wheel)
        canvas.bind('<Button-4>', on_wheel)
        canvas.bind('<Button-5>', on_wheel)

        try:
            win.bind_all('<MouseWheel>', on_wheel)
            win.bind_all('<Button-4>', on_wheel)
            win.bind_all('<Button-5>', on_wheel)
        except Exception:
            pass

        ctrl_frame = tk.Frame(win)
        ctrl_frame.pack(fill='x')

        btn_move_all = Button(ctrl_frame, text='Move All', command=toggle_move_all)
        btn_move_all.pack(side='left')

        btn_flip = Button(ctrl_frame, text='Flip Selected', command=flip_selected)
        btn_flip.pack(side='left')

        btn_accept = Button(ctrl_frame, text='Accept', command=accept)
        btn_accept.pack(side='right')

        win.mainloop()
        win.destroy()

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
        t = self.transforms[img_index]
        # support older transforms without flip flag
        if len(t) >= 4:
            x_center, y_center, angle_deg, flipped_flag = t[0], t[1], t[2], bool(t[3])
        else:
            x_center, y_center, angle_deg = t
            flipped_flag = False
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
        # if tile was flipped horizontally, mirror x before rotation
        # needs to be before
        if flipped_flag:
            dx = -dx
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
    
    for img_arr, t in zip(small_images, transforms):
        # support both (x,y,angle) and (x,y,angle,flipped)
        if len(t) >= 4:
            x, y, angle, flipped = t[0], t[1], t[2], bool(t[3])
        else:
            x, y, angle = t
            flipped = False

        pil = Image.fromarray(img_arr)
        w, h = pil.size

        scaled = pil.resize((int(w * scale_factor), int(h * scale_factor)), Image.BILINEAR)
        if flipped:
            scaled = scaled.transpose(Image.FLIP_LEFT_RIGHT)
        rotated = scaled.rotate(angle, expand=True)

        # w/ alpha mask
        base.paste(rotated, (int(x - rotated.width // 2),
                             int(y - rotated.height // 2)),
                   rotated if rotated.mode == 'RGBA' else None)

    return np.array(base)



def register_tiled_locations():

    fullimg_path = fm2p.select_file(
        'Choose widefield template TIF.',
        filetypes=[('TIF','.tif'), ('TIFF', '.tiff'), ]
    )

    animalID = os.path.split(os.path.split(fullimg_path)[0])[1]

    
    fullimg = np.array(Image.open(fullimg_path))
    
    newshape = (fullimg.shape[0] // 2, fullimg.shape[1] // 2)
    zoom_factors = (
        (newshape[0]/ fullimg.shape[0]),
        (newshape[1]/ fullimg.shape[1]),
    )
    resized_fullimg = zoom(fullimg, zoom=zoom_factors, order=1)

    cohort_dir = fm2p.select_directory('Select cohort directory (does not need to be just this animal).')

    smallimgs = []
    pos_keys = []
    preproc_paths = fm2p.find('*{}*preproc.h5'.format(animalID), cohort_dir)
    entries = []
    for p in preproc_paths:
        main_key = os.path.split(os.path.split(os.path.split(p)[0])[0])[1]
        pos_key = main_key.split('_')[-1]
        # numeric suffix
        try:
            pos_num = int(''.join([c for c in pos_key if c.isdigit()]))
        except Exception:
            pos_num = pos_key
        pdata = fm2p.read_h5(p)
        singlemap = pdata['twop_ref_img']
        entries.append((pos_num, pos_key, singlemap, p))

    # sort by numeric position (if parse succeeded), otherwise by pos_key
    entries.sort(key=lambda x: x[0])
    for pos_num, pos_key, singlemap, p in entries:
        smallimgs.append(singlemap)
        pos_keys.append(pos_key)
        
    # replace preproc_paths with the sorted order for later loops
    # TODO: does this work? not tested...
    preproc_paths = [e[3] for e in entries]


    aligner = ManualImageAligner(resized_fullimg, smallimgs, pos_keys, scale_factor=1.0)
    transforms = aligner.run()

    # composite = overlay_registered_images(
    #     resized_fullimg,
    #     smallimgs,
    #     transforms,
    #     scale_factor=0.27)
    
    # Image.fromarray(composite).save("composite_aligned_frames_v1.png")

    all_global_positions = {}

    for pi, p in tqdm(enumerate(preproc_paths)):
        main_key = os.path.split(os.path.split(os.path.split(p)[0])[0])[1]
        pos_key = main_key.split('_')[-1]
        # pos = int(pos_key[-2:])
        pdata = fm2p.read_h5(p)

        cell_positions = np.zeros([len(pdata['cell_x_pix'].keys()), 4])
        for ki, k in enumerate(pdata['cell_x_pix'].keys()):
            cellx = np.median(pdata['cell_x_pix'][k])
            celly = np.median(pdata['cell_y_pix'][k])
            global_x, global_y = aligner.local_to_global(pi, cellx, celly)
            cell_positions[ki,:] = np.array([cellx, celly, global_x, global_y])

        all_global_positions[pos_key] = cell_positions

    fm2p.write_h5(
        os.path.join(
            os.path.split(fullimg_path)[0],
            '{}_aligned_composite_local_to_global_transform_v2.h5'.format(animalID)
        ),
        all_global_positions
    )



if __name__ == '__main__':

    register_tiled_locations()

    