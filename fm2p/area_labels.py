# -*- coding: utf-8 -*-
"""
fm2p/area_labels.py

Visual area of every cell in a recording, from the animal's widefield
registration in <V1PPC>/mouse_composites/<animal>/.

DMM, October 2026
"""


if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

import os
import re
import glob
import json
import argparse
import numpy as np
import h5py
from matplotlib.path import Path

from .utils.files import write_h5


ATLAS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils', 'vfs_contours.json')
# Same ids as merge_animal_essentials._AREA_IDS, 0 = outside every contour.
AREA_IDS = {'none': 0, 'RL': 2, 'AM': 3, 'PM': 4, 'V1': 5, 'AL': 7, 'LM': 8, 'P': 9, 'A': 10}
# Sign of each area in the animals' own maps (sign_map convention), A left out.
AREA_SIGN = {'V1': -1, 'LM': 1, 'AL': -1, 'RL': 1, 'AM': -1, 'PM': 1}
BORDER_PX = 5.0
REF_SIZE = 400.0
NON_AREAS = ('border', 'none', 'unassigned')
AREA_COLORS = {'V1': '#2a78d6', 'AM': '#eb6834', 'A': '#1baf7a', 'RL': '#6250d6', 'PM': '#e87ba4',
               'AL': '#eda100', 'LM': '#52514e', 'border': '#b0b0b0', 'none': '#d8d8d8'}


def load_atlas(path=ATLAS_PATH):
    """ Reference area contours, name -> (N, 2) in reference VFS px. """

    with open(path, 'r') as f:
        raw = json.load(f)
    return {k: np.asarray(v, float) for k, v in raw.items() if v is not None and len(v) >= 3}


def animal_registration(animal_dir):
    """ Affine from widefield-composite px to reference VFS px for one animal.

    Rebuilds what merge_animal_essentials applies: composite px scaled by
    REF_SIZE / composite_size, then the rotation / scale / shift stored in the
    newest vfs_aligned_composite_*.h5 (cv2.getRotationMatrix2D convention).

    Returns
    -------
    A : (2, 3) array
    info : dict
        'pearson_r', 'composite_size', 'source'.
    """

    path = sorted(glob.glob(os.path.join(animal_dir, 'vfs_aligned_composite_*.h5')))[-1]
    with h5py.File(path, 'r') as f:
        dx, dy, rot, sc, r = [float(np.asarray(f[k]).flat[0]) for k in
                              ('_transform_dx', '_transform_dy', '_transform_rotation_deg',
                               '_transform_scale_factor', '_transform_pearson_r')]
        ref_h = float(np.asarray(f['_ref_vfs_shape'])[0]) if '_ref_vfs_shape' in f else REF_SIZE
    a = np.deg2rad(rot)
    al, be = sc * np.cos(a), sc * np.sin(a)
    cx = cy = ref_h / 2.0
    A = np.array([[al, be, (1 - al) * cx - be * cy + dx],
                  [-be, al, be * cx + (1 - al) * cy + dy]])
    # Composite = widefield template halved, as in merge_animal_essentials.
    tifs = glob.glob(os.path.join(animal_dir, '*.tif'))
    size = 1024.0
    if tifs:
        from PIL import Image
        with Image.open(max(tifs, key=os.path.getmtime)) as im:
            size = float(im.size[1] // 2)
    A = A @ np.diag([ref_h / size, ref_h / size, 1.0])
    return A, {'pearson_r': r, 'composite_size': size, 'source': path}


def fit_tile_transform(local_xy, global_xy):
    """ Similarity transform (with optional mirror) from tile px to widefield px.

    register_tiled_locations' ManualImageAligner applies scale, optional
    horizontal flip, rotation and shift, so a least-squares similarity fit to
    the stored pairs recovers it exactly (residual ~1e-9 px).

    Returns
    -------
    t : tuple
        (flip, (a, b, tx, ty), max residual px).
    """

    best = None
    for flip in (False, True):
        s = local_xy * np.array([-1.0 if flip else 1.0, 1.0])
        n = len(s)
        M = np.zeros((2 * n, 4))
        M[0::2] = np.c_[s[:, 0], -s[:, 1], np.ones(n), np.zeros(n)]
        M[1::2] = np.c_[s[:, 1], s[:, 0], np.zeros(n), np.ones(n)]
        p = np.linalg.lstsq(M, global_xy.ravel(), rcond=None)[0]
        err = float(np.abs(M @ p - global_xy.ravel()).max())
        if best is None or err < best[2]:
            best = (flip, p, err)
    return best


def apply_tile_transform(t, xy):
    """ Tile px to widefield px. """

    flip, (a, b, tx, ty), _ = t
    x = xy[:, 0] * (-1.0 if flip else 1.0)
    y = xy[:, 1]
    return np.c_[a * x - b * y + tx, b * x + a * y + ty]


def _sort_cell_keys(keys):
    """ Cell keys in numeric order ('0', '1', '2', ... not '0', '1', '10'). """

    def k(x):
        m = re.search(r'\d+', x)
        return (int(m.group()) if m else np.inf, x)
    return sorted(keys, key=k)


def cell_centroids(h5_path, numeric=True):
    """ Median pixel of every cell, in true (numeric) cell order.

    numeric=False gives the string order the stored composite files use.
    """

    with h5py.File(h5_path, 'r') as f:
        keys = list(f['cell_x_pix'].keys())
        if numeric:
            keys = _sort_cell_keys(keys)
        return np.array([[np.median(f['cell_x_pix'][k][()]), np.median(f['cell_y_pix'][k][()])]
                         for k in keys])


def own_sign(animal_dir, wf_xy, composite_size):
    """ Animal's own visual field sign at widefield-composite points.

    Same map merge_animal_essentials stores as `sign_map`:
    gaussian_filter(zoom(VFS_raw, 2.555), 2), on a grid of ~composite size.
    """

    from scipy.io import loadmat
    from scipy.ndimage import gaussian_filter, zoom, map_coordinates
    path = os.path.join(animal_dir, 'VFS_maps.mat')
    if not os.path.isfile(path):
        return np.full(len(wf_xy), np.nan), None
    sm = gaussian_filter(zoom(loadmat(path)['VFS_raw'].astype(float), 2.555), 2)
    k = sm.shape[0] / composite_size
    v = map_coordinates(sm, [wf_xy[:, 1] * k, wf_xy[:, 0] * k], order=1, cval=np.nan)
    return v, sm


def label_points(ref_xy, atlas):
    """ Atlas area and distance to the nearest other area for reference VFS points.

    Returns
    -------
    names : (N,) object
        Area name, 'none' outside every contour.
    dist : (N,) float
        Ref px to the nearest contour of a different area (to any contour
        for 'none').
    """

    paths = {k: Path(v) for k, v in atlas.items()}
    names = np.array(['none'] * len(ref_xy), dtype=object)
    for k, p in paths.items():
        names[(names == 'none') & p.contains_points(ref_xy)] = k
    dist = np.full(len(ref_xy), np.nan)
    for k, c in atlas.items():
        d = np.min(np.hypot(ref_xy[:, None, 0] - c[None, :, 0], ref_xy[:, None, 1] - c[None, :, 1]), 1)
        other = names != k
        dist[other] = np.fmin(dist[other], d[other])
    return names, dist


def assign_areas(h5_path, composites_dir=None, border_px=BORDER_PX, atlas=None):
    """ Area of every cell in one recording.

    Parameters
    ----------
    h5_path : str
        `_preproc.h5` inside <cohort>/<date>_DMM_<animal>_<pos>/<fm*>/.
    composites_dir : str or None
        Defaults to <V1PPC>/mouse_composites, two levels above the cohort.
    border_px : float
        Cells nearer than this (ref VFS px) to another area are 'border'.

    Returns
    -------
    out : dict
        'area' (C,) analysis label (atlas area, 'border' or 'none');
        'area_atlas' (C,) atlas area; 'area_id' (C,) atlas id;
        'border_dist' (C,) ref px; 'wf_pos', 'vfs_pos' (C, 2);
        'own_sign' (C,) animal's sign map at the cell;
        'info' dict (registration r, tile fit residual, ...).  Raises if the
        animal or position has no registration.
    """

    sess = os.path.basename(os.path.dirname(os.path.dirname(h5_path)))
    m = re.search(r'(DMM\d+)_(pos\d+)', sess)
    if m is None:
        raise ValueError('No DMM<animal>_pos<NN> in session name {}'.format(sess))
    animal, pos = m.groups()
    if composites_dir is None:
        cohort = os.path.dirname(os.path.dirname(os.path.dirname(h5_path)))
        composites_dir = os.path.join(os.path.dirname(cohort), 'mouse_composites')
    animal_dir = os.path.join(composites_dir, animal)
    l2g = glob.glob(os.path.join(animal_dir, '{}_aligned_composite_local_to_global_transform*.h5'.format(animal)))
    if not l2g:
        raise FileNotFoundError('No local_to_global transform in {}'.format(animal_dir))
    l2g = max(l2g, key=os.path.getmtime)
    with h5py.File(l2g, 'r') as f:
        if pos not in f:
            raise KeyError('{} not registered in {}'.format(pos, os.path.basename(l2g)))
        L = np.asarray(f[pos][()], float)
    t = fit_tile_transform(L[:, :2], L[:, 2:4])
    A, reg = animal_registration(animal_dir)
    atlas = atlas or load_atlas()

    # Own cells, own order; the position's tile transform as registered,
    # whichever day it was registered on.
    loc = cell_centroids(h5_path)
    wf = apply_tile_transform(t, loc)
    ref = (A @ np.c_[wf, np.ones(len(wf))].T).T
    names, dist = label_points(ref, atlas)
    area = names.copy()
    area[(names != 'none') & (dist < border_px)] = 'border'
    sg, _ = own_sign(animal_dir, wf, reg['composite_size'])
    info = {'animal': animal, 'pos': pos, 'registration_r': reg['pearson_r'],
            'registration_file': os.path.basename(reg['source']), 'tile_file': os.path.basename(l2g),
            'tile_fit_err_px': t[2], 'tile_flip': bool(t[0]), 'border_px': float(border_px),
            'atlas': os.path.basename(ATLAS_PATH)}
    return {'area': area, 'area_atlas': names, 'area_id': np.array([AREA_IDS.get(n, -1) for n in names]),
            'border_dist': dist, 'wf_pos': wf, 'vfs_pos': ref, 'own_sign': sg, 'info': info}


def save_areas(h5_path, out):
    """ Write assign_areas output beside the recording as <prefix>_areas.h5. """

    path = h5_path.replace('_preproc.h5', '_areas.h5')
    d = {k: (','.join(v) if k in ('area', 'area_atlas') else v) for k, v in out.items() if k != 'info'}
    d['info'] = {k: v for k, v in out['info'].items()}
    write_h5(path, d)
    return path


def plot_animal(animal_dir, results, path, atlas=None):
    """ Atlas over the animal's own sign map, cells coloured by area. """

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import map_coordinates

    atlas = atlas or load_atlas()
    A, reg = animal_registration(animal_dir)
    _, sm = own_sign(animal_dir, np.zeros((1, 2)), reg['composite_size'])
    fig, ax = plt.subplots(figsize=(9, 9))
    if sm is not None:
        Y, X = np.mgrid[0:int(REF_SIZE), 0:int(REF_SIZE)].astype(float)
        s = np.linalg.inv(np.r_[A, [[0, 0, 1]]]) @ np.c_[X.ravel(), Y.ravel(), np.ones(X.size)].T
        k = sm.shape[0] / reg['composite_size']
        W = map_coordinates(sm, [s[1] * k, s[0] * k], order=1, cval=np.nan).reshape(X.shape)
        v = np.nanpercentile(np.abs(W), 98)
        ax.imshow(W, cmap='RdBu_r', vmin=-v, vmax=v, alpha=0.75)
    for k, c in atlas.items():
        ax.plot(*np.r_[c, c[:1]].T, color='k', lw=1)
        ax.text(*c.mean(0), k, fontsize=12, weight='bold', ha='center')
    for name, r in results:
        ax.scatter(r['vfs_pos'][:, 0], r['vfs_pos'][:, 1], s=4, edgecolors='none',
                   c=[AREA_COLORS.get(a, '#888888') for a in r['area']])
        ax.text(*np.nanmean(r['vfs_pos'], 0), name, fontsize=6)
    ax.set_xlim(0, REF_SIZE)
    ax.set_ylim(REF_SIZE, 0)
    ax.set_title('{}: own sign map in atlas space (registration r = {:.2f}); cells by area, grey = '
                 'within {:.0f} px of another area'.format(os.path.basename(animal_dir), reg['pearson_r'],
                                                         BORDER_PX), fontsize=9)
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def area_batch(batch_path, composites_dir=None, border_px=BORDER_PX, save=True):
    """ assign_areas for every fm recording under a root, plus per-animal figures.

    Figures go to <composites_dir>/<animal>/<animal>_cell_areas.png.

    Returns
    -------
    results : dict
        preproc.h5 path -> assign_areas output, or the exception raised.
    """

    from .analyze import find_recordings

    atlas = load_atlas()
    results = {}
    by_animal = {}
    for h5 in find_recordings(batch_path):
        sess = os.path.basename(os.path.dirname(os.path.dirname(h5)))
        try:
            out = assign_areas(h5, composites_dir=composites_dir, border_px=border_px, atlas=atlas)
        except Exception as e:
            print('{:28s} ERROR: {}'.format(sess, e))
            results[h5] = e
            continue
        results[h5] = out
        if save:
            save_areas(h5, out)
        u, c = np.unique(out['area'], return_counts=True)
        inf = out['info']
        print('{:28s} {:4s} n={:4d} {} | sign agrees {}'.format(
            sess, os.path.basename(os.path.dirname(h5)), len(out['area']),
            ', '.join('{} {}'.format(a, n) for a, n in zip(u, c)),
            '{:.0%}'.format(np.mean([np.sign(s) == AREA_SIGN[a] for a, s in zip(out['area'], out['own_sign'])
                                     if a in AREA_SIGN and np.isfinite(s)]))
            if any(a in AREA_SIGN for a in out['area']) else '  - '))
        by_animal.setdefault(inf['animal'], []).append((inf['pos'], out))
    if composites_dir is None and results:
        h5 = next(iter(results))
        cohort = os.path.dirname(os.path.dirname(os.path.dirname(h5)))
        composites_dir = os.path.join(os.path.dirname(cohort), 'mouse_composites')
    for an, rs in by_animal.items():
        fig_path = os.path.join(composites_dir, an, '{}_cell_areas.png'.format(an))
        plot_animal(os.path.join(composites_dir, an), rs, fig_path, atlas)
        print('Saved {}'.format(fig_path))
    return results


def main():
    """ Command-line entry point. """

    ap = argparse.ArgumentParser(description='Visual area of every cell from the widefield registration.')
    ap.add_argument('--h5_path', default=None, help='One recording\'s _preproc.h5.')
    ap.add_argument('--batch', action='store_true', help='Every fm recording under --batch_path.')
    ap.add_argument('--batch_path', default='/home/dylan/Storage/freely_moving_data/V1PPC/cohort02_recordings')
    ap.add_argument('--composites_dir', default=None,
                    help='Defaults to <V1PPC>/mouse_composites beside the cohort directory.')
    ap.add_argument('--border_px', type=float, default=BORDER_PX)
    args = ap.parse_args()
    if args.batch:
        area_batch(args.batch_path, args.composites_dir, args.border_px)
    elif args.h5_path:
        out = assign_areas(args.h5_path, args.composites_dir, args.border_px)
        print('Saved {}'.format(save_areas(args.h5_path, out)))
    else:
        ap.error('--h5_path or --batch is required.')


if __name__ == '__main__':
    main()
