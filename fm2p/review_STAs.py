
from PIL import Image, ImageTk
import tkinter as tk
from matplotlib import cm
import argparse

import os
import numpy as np
from tqdm import tqdm
import multiprocessing
from scipy.optimize import curve_fit

import fm2p


def label_and_fit_gui(STA, STA1, STA2, out_path=None):
    """Interactive GUI to label STA pages and run gaus_eval on positives.

    STA, STA1, STA2: 3D numpy arrays shaped (n_pages, H, W)
    float_value: additional float to display in title
    out_path: path to save results (.npz). If None, saved to './label_results.npz'
    """
    n = STA.shape[0]
    # queue of indices to visit
    queue = list(range(n))
    labels = -1 * np.ones(n, dtype=int)
    history = []  # tuples for undo: ('mark', idx) or ('skip', idx, old_pos)

    # If an output path exists with saved labels, load them and skip the GUI review
    skip_review = False
    if out_path is not None and os.path.exists(out_path):
        try:
            npz = np.load(out_path, allow_pickle=True)
            if 'labels' in npz.files:
                loaded_labels = npz['labels']
                if loaded_labels.shape[0] == n:
                    labels = loaded_labels.astype(int)
                    skip_review = True
                    print(f'Loaded existing labels from {out_path}; skipping review and running gaus_eval on labeled cells.')
                else:
                    print(f'Found labels in {out_path} but length mismatch ({loaded_labels.shape[0]} != {n}); launching review GUI.')
            else:
                print(f'No labels array in {out_path}; launching review GUI.')
        except Exception as e:
            print(f'Could not load {out_path}: {e}; launching review GUI.')

    root = None
    if not skip_review:
        root = tk.Tk()
        root.title('Label STAs')

    if not skip_review:
        title_var = tk.StringVar()
        title_label = tk.Label(root, textvariable=title_var, font=('TkDefaultFont', 12))
        title_label.pack(side='top', fill='x')

        frame = tk.Frame(root)
        frame.pack(side='top')

        canvases = []
        img_refs = [None, None, None]
        for i in range(3):
            c = tk.Label(frame)
            c.pack(side='left', padx=5, pady=5)
            canvases.append(c)

        btn_frame = tk.Frame(root)
        btn_frame.pack(side='top', fill='x')

        def update_title(cur_pos, float_value):
            # Print progress/title to terminal, overwriting previous line
            labeled = np.sum(labels != -1)
            title_str = f'Progress: {labeled}/{n}    corr2d: {float_value:.4f}    index: {cur_pos+1}/{len(queue)}'
            print('\r' + title_str, end='', flush=True)

        def make_image_from_array(arr):
            # normalize and map through jet
            a = np.asarray(arr).astype(float)
            # center colormap at zero: normalize symmetrically so 0 -> 0.5 (middle of colormap)
            amin = np.nanmin(a)
            amax = np.nanmax(a)
            max_abs = max(abs(amin), abs(amax))
            if max_abs == 0 or np.isnan(max_abs):
                norm = np.zeros_like(a)
            else:
                norm = (a + max_abs) / (2.0 * max_abs)

            # use coolwarm colormap
            rgba = (cm.get_cmap('coolwarm')(norm) * 255).astype(np.uint8)
            im = Image.fromarray(rgba)

            # scale down to 0.25 of original size
            w, h = im.size
            nw = max(1, int(round(w * 0.25)))
            nh = max(1, int(round(h * 0.25)))
            im = im.resize((nw, nh), resample=Image.BILINEAR)
            return im

        cur_idx_ptr = 0

        def show_current():
            nonlocal cur_idx_ptr
            if cur_idx_ptr >= len(queue):
                # finished
                root.quit()
                return
            
            idx = queue[cur_idx_ptr]
            
            corrval_to_show = fm2p.corr2_coeff(STA1[idx], STA2[idx])

            update_title(cur_idx_ptr, corrval_to_show)
            im1 = make_image_from_array(STA[idx])
            im2 = make_image_from_array(STA1[idx])
            im3 = make_image_from_array(STA2[idx])
            for i, im in enumerate((im1, im2, im3)):
                tkimg = ImageTk.PhotoImage(im, master=root)
                canvases[i].config(image=tkimg)
                canvases[i].image = tkimg
                img_refs[i] = tkimg

        def do_mark(value):
            nonlocal cur_idx_ptr
            if cur_idx_ptr >= len(queue):
                return
            idx = queue[cur_idx_ptr]
            prev = labels[idx]
            labels[idx] = value
            history.append(('mark', idx, prev))
            cur_idx_ptr += 1
            if cur_idx_ptr >= len(queue):
                root.quit()
                return
            show_current()

        def do_skip():
            nonlocal cur_idx_ptr
            if cur_idx_ptr >= len(queue):
                return
            idx = queue.pop(cur_idx_ptr)
            queue.append(idx)
            history.append(('skip', idx, cur_idx_ptr))
            # show same cur_idx_ptr (now new item at this position)
            show_current()

        def do_back():
            nonlocal cur_idx_ptr
            if not history:
                return
            action = history.pop()
            if action[0] == 'mark':
                _, idx, prev = action
                labels[idx] = prev
                # set current pointer to this item's position
                try:
                    cur_idx_ptr = queue.index(idx)
                except ValueError:
                    cur_idx_ptr = 0
            elif action[0] == 'skip':
                _, idx, old_pos = action
                # remove last element (should be idx) and insert back
                if queue and queue[-1] == idx:
                    queue.pop()
                    queue.insert(old_pos, idx)
                    cur_idx_ptr = old_pos
            show_current()

        def on_left(event=None):
            do_mark(0)

        def on_right(event=None):
            do_mark(1)

        btn_back = tk.Button(btn_frame, text='Back', command=do_back)
        btn_back.pack(side='left')
        btn_skip = tk.Button(btn_frame, text='Skip', command=do_skip)
        btn_skip.pack(side='left')
        btn_left = tk.Button(btn_frame, text='Mark 0 (Left)', command=on_left)
        btn_left.pack(side='left')
        btn_right = tk.Button(btn_frame, text='Mark 1 (Right)', command=on_right)
        btn_right.pack(side='left')

        root.bind('<Left>', on_left)
        root.bind('<Right>', on_right)
        root.bind('b', lambda e: do_back())

    if not skip_review:
        show_current()
        root.mainloop()

    # After GUI done (or if we skipped review), compute fraction true
    # If any items remain unlabeled, leave them as -1; consider them false
    labeled_mask = labels != -1
    frac_true = np.sum(labels == 1) / float(n)
    print()  # finish the overwritten progress line
    print(f'Fraction labeled true: {frac_true:.4f} ({np.sum(labels==1)}/{n})')

    # Run gaus_eval on positive-labeled cells and collect numeric arrays
    true_indices = np.where(labels == 1)[0]
    m = len(true_indices)

    corr2d = np.full((m,), np.nan)

    pos_centroids = np.full((m, 2), np.nan)
    pos_amplitudes = np.full((m,), np.nan)
    pos_baselines = np.full((m,), np.nan)
    pos_sigmas = np.full((m, 2), np.nan)
    pos_tilts = np.full((m, 2), np.nan)

    neg_centroids = np.full((m, 2), np.nan)
    neg_amplitudes = np.full((m,), np.nan)
    neg_baselines = np.full((m,), np.nan)
    neg_sigmas = np.full((m, 2), np.nan)
    neg_tilts = np.full((m, 2), np.nan)

    print('Fitting gaussian to good cells.')

    for i, idx in enumerate(true_indices):
        # compute a display value (corr) and print progress, overwriting previous line
        corrval = fm2p.corr2_coeff(STA1[idx], STA2[idx])
        print(f'\rEvaluating {i+1}/{m} (cell index {idx})', end='', flush=True)
        res = fm2p.gaus_eval(STA[idx], STA1[idx], STA2[idx])
        corr2d[i] = res.get('corr2d', np.nan)
        pos = res.get('positive', {})
        neg = res.get('negative', {})
        try:
            pc = pos.get('centroid', (np.nan, np.nan))
            pos_centroids[i, 0] = pc[0]
            pos_centroids[i, 1] = pc[1]
            pos_amplitudes[i] = pos.get('amplitude', np.nan)
            pos_baselines[i] = pos.get('baseline', np.nan)
            pos_sigmas[i, 0] = pos.get('sigma_x', np.nan)
            pos_sigmas[i, 1] = pos.get('sigma_y', np.nan)
            pt = pos.get('tilt', (np.nan, np.nan))
            pos_tilts[i, 0] = pt[0]
            pos_tilts[i, 1] = pt[1]
        except Exception:
            pass
        try:
            nc = neg.get('centroid', (np.nan, np.nan))
            neg_centroids[i, 0] = nc[0]
            neg_centroids[i, 1] = nc[1]
            neg_amplitudes[i] = neg.get('amplitude', np.nan)
            neg_baselines[i] = neg.get('baseline', np.nan)
            neg_sigmas[i, 0] = neg.get('sigma_x', np.nan)
            neg_sigmas[i, 1] = neg.get('sigma_y', np.nan)
            nt = neg.get('tilt', (np.nan, np.nan))
            neg_tilts[i, 0] = nt[0]
            neg_tilts[i, 1] = nt[1]
        except Exception:
            pass

    # finish progress line
    print()

    if out_path is None:
        out_path = os.path.join(os.getcwd(), 'label_results.npz')

    # Save numeric arrays and labels
    np.savez(out_path,
             labels=labels,
             true_indices=true_indices,
             frac_true=frac_true,
             corr2d=corr2d,
             pos_centroids=pos_centroids,
             pos_amplitudes=pos_amplitudes,
             pos_baselines=pos_baselines,
             pos_sigmas=pos_sigmas,
             pos_tilts=pos_tilts,
             neg_centroids=neg_centroids,
             neg_amplitudes=neg_amplitudes,
             neg_baselines=neg_baselines,
             neg_sigmas=neg_sigmas,
             neg_tilts=neg_tilts)

    print(f'Saved labeling and numeric results to: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--label', action='store_true', help='Launch interactive labeling GUI')
    # parser.add_argument('--value', type=float, default=0.0, help='Float value to display in title')
    # parser.add_argument('--out', type=str, default=None, help='Output .npz path for labels/results')
    args = parser.parse_args()

    sn_path = fm2p.select_file(
        'Select sparse noise receptive field HDF file.',
        filetypes=[('HDF','.h5'),]
    )

    pathout = os.path.join(os.path.split(sn_path)[0], 'sparse_noise_labels_gaussfit.npz')

    data = fm2p.read_h5(sn_path)
    STA = data['STA'].reshape(-1, 768, 1360)
    STA1 = data['STA1'].reshape(-1, 768, 1360)
    STA2 = data['STA2'].reshape(-1, 768, 1360)
    label_and_fit_gui(STA, STA1, STA2, out_path=pathout)