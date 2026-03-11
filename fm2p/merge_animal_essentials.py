# -*- coding: utf-8 -*-


import os
import json
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom
from scipy.io import loadmat
import scipy.stats
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.path import Path
from tqdm import tqdm

import fm2p


# Canonical mapping from area name to integer label ID.  Must stay consistent
# with the LABEL_MAP in topography.py and topography_plots.py.
_AREA_IDS = {'RL': 2, 'AM': 3, 'PM': 4, 'V1': 5, 'AL': 7, 'LM': 8, 'P': 9, 'A': 10}
_LABEL_MAP = {
    0: 'unassigned', 2: 'RL', 3: 'AM', 4: 'PM', 5: 'V1',
    7: 'AL', 8: 'LM', 9: 'P', 10: 'A'
}


def build_labeled_array_from_contours(contours_data, shape=(2048, 2048)):
    """Create an integer labeled array from named area contours.

    Parameters
    ----------
    contours_data : dict
        Keys of the form 'contour_<area_name>' mapping to (N, 2) ndarrays of
        (x, y) polygon vertices (as saved by register_animals_using_shared_template).
    shape : tuple
        (height, width) of the output array.

    Returns
    -------
    labeled_array : ndarray, shape
        Each pixel set to the area's integer ID (from _AREA_IDS) or 0.
    label_map : dict
        Integer ID → area name string.
    """
    labeled_array = np.zeros(shape, dtype=int)
    h, w = shape

    # Pre-compute grid of pixel centres for fast point-in-polygon queries.
    grid_y, grid_x = np.mgrid[:h, :w]
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    for key, coords in contours_data.items():
        if not key.startswith('contour_'):
            continue
        area_name = key[len('contour_'):]
        if area_name not in _AREA_IDS:
            continue
        if not isinstance(coords, np.ndarray) or coords.shape[0] < 3:
            continue

        pts = coords[:, :2].astype(float)
        try:
            path = Path(pts)
            mask = path.contains_points(grid_pts).reshape(shape)
            labeled_array[mask] = _AREA_IDS[area_name]
        except Exception:
            continue

    return labeled_array, dict(_LABEL_MAP)





def merge_animal_essentials(animalID):

    # cohort_dir = '/home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings/cohort02_recordings/'
    cohort_dir = '/home/dylan/Storage/freely_moving_data/_V1PPC/'
    map_dir = '/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites/{}/'.format(animalID)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-animal', '--animal', type=str)
    # parser.add_argument('-codir', '--codir', type=str)
    # parser.add_argument('-mapdir', '--mapdir', type=str)
    # args = parser.parse_args()

    # if args.codir is None:
    #     cohort_dir = fm2p.select_directory(
    #         'Select cohort directory.'
    #     )
    # else:
    #     cohort_dir = args.codir
    
    # if args.mapdir is None:
    #     map_dir = fm2p.select_directory(
    #         'Select sign map and composite directory.'
    #     )
    # else:
    #     map_dir = args.mapdir

    # animalID = args.animal

    animal_dict = {}

    preproc_paths = fm2p.find(
        '*{}*preproc.h5'.format(animalID),
        cohort_dir
    )
    for p in preproc_paths:
        main_key = os.path.split(os.path.split(os.path.split(p)[0])[0])[1]
        pos_key = main_key.split('_')[-1]
        # v2 is the batch that were run jan 16-17 to calculate a seperate reliability score
        # for light vs dark conditions
        r = fm2p.find('eyehead_revcorrs_v06.h5', os.path.split(p)[0], MR=True)
        sn = os.path.join(os.path.split(os.path.split(p)[0])[0], 'sn1/sparse_noise_labels_gaussfit.npz')
        try:
            modeldata = fm2p.find('pytorchGLM_predictions_v09b.h5', os.path.split(p)[0], MR=True)
        except:
            modeldata = 'none'

        try:
            boundarydata = fm2p.find('*boundary_results*.h5', os.path.split(p)[0], MR=True)
        except:
            boundarydata = 'none'

        animal_dict[pos_key] = {
            'preproc': p,
            'revcorr': r,
            'sparsenoise': sn,
            'name': main_key,
            'model': modeldata,
            'boundary': boundarydata
        }


    full_dict = {}

    all_pdata = []
    all_rdata = []
    all_pos = []
    all_cell_positions = []
    all_model_data = []
    full_map = np.zeros([512*5, 512*5]) * np.nan

    row = 0
    col = 0
    for pos in tqdm(range(1,26)):
        pos_str = 'pos{:02d}'.format(pos)

        if pos_str not in list(animal_dict.keys()):
            if (pos%5)==0: # if you're at the end of a row
                col = 0
                row += 1
            else:
                col += 1
            continue

        pdata = fm2p.read_h5(animal_dict[pos_str]['preproc'])
        rdata = fm2p.read_h5(animal_dict[pos_str]['revcorr'])
        if animal_dict[pos_str]['model'] != 'none':
            modeldata = fm2p.read_h5(animal_dict[pos_str]['model'])
        else:
            print(f'  -> No model data for {pos_str}: {animal_dict[pos_str]["model"]}')
            modeldata = {}

        if animal_dict[pos_str]['boundary'] != 'none':
            boundarydata = fm2p.read_h5(animal_dict[pos_str]['boundary'])
        else:
            boundarydata = {}

        if os.path.isfile(animal_dict[pos_str]['sparsenoise']):
            sndata = np.load(animal_dict[pos_str]['sparsenoise'])
            snarr = np.concatenate([sndata['true_indices'][:,np.newaxis], sndata['pos_centroids']], axis=1)
        else:
            snarr = np.nan

        all_pdata.append(pdata)
        all_rdata.append(rdata)
        all_pos.append((row, col))
        all_model_data.append(modeldata)

        singlemap = pdata['twop_ref_img']

        full_map[row*512 : (row*512)+512, col*512 : (col*512)+512] = singlemap

        cell_positions = np.zeros([len(pdata['cell_x_pix'].keys()), 2]) * np.nan

        for ki, k in enumerate(pdata['cell_x_pix'].keys()):
            # cellx = np.median(512 - pdata['cell_x_pix'][k]) + col*512
            cellx = np.median(pdata['cell_x_pix'][k]) + col*512
            celly = np.median(pdata['cell_y_pix'][k]) + row*512

            cell_positions[ki,:] = np.array([cellx, celly])

        full_dict[pos_str] = {
            'rdata': rdata,
            'tile_pos': np.array([row,col]),
            'cell_pos': cell_positions,
            'sn_cents': snarr,
            'model': modeldata,
            'boundary': boundarydata
        }

        all_cell_positions.append(cell_positions)

        col += 1

        if (pos%5)==0: # if you're at the end of a row
            col = 0
            row += 1

    full_dict['rigid_tiled_map'] = full_map

    vfs_path = os.path.join(map_dir, 'VFS_maps.mat')
    vfs = loadmat(vfs_path)
    overlay = gaussian_filter(zoom(vfs['VFS_raw'].copy(), 2.555), 2)

    refpath = fm2p.find('*.tif', map_dir, MR=True)

    fullimg = np.array(Image.open(refpath))
    newshape = (fullimg.shape[0] // 2, fullimg.shape[1] // 2)
    zoom_factors = (
        (newshape[0]/ fullimg.shape[0]),
        (newshape[1]/ fullimg.shape[1]),
    )
    resized_fullimg = zoom(fullimg, zoom=zoom_factors, order=1)

    full_dict['sign_map'] = overlay
    full_dict['ref_img'] = resized_fullimg

    # VFS area labels
    # Load the outputs of register_animals_using_shared_template() if present.
    # vfs_aligned_composite: per-position cell arrays where
    #   cols 0,1 = composite-space cell coords (legacy widefield-halved, ~0-1024)
    #   cols 2,3 = reference VFS coords (stored but computed with wrong WF_SIZE=2048)
    # We recompute VFS coords here using the correct scale and reference contours
    # from vfs_contours.json (in reference VFS space, ~0-400).
    try:
        aligned_path = fm2p.find('vfs_aligned_composite_*.h5', map_dir, MR=True)
        aligned_composite = fm2p.read_h5(aligned_path)

        # Reference VFS shape and stored transform parameters.
        ref_vfs_shape = tuple(
            aligned_composite.get('_ref_vfs_shape', np.array([400, 400])).astype(int))
        ref_h, ref_w = ref_vfs_shape

        dx       = float(np.asarray(aligned_composite['_transform_dx']).flat[0])
        dy       = float(np.asarray(aligned_composite['_transform_dy']).flat[0])
        rot_deg  = float(np.asarray(aligned_composite['_transform_rotation_deg']).flat[0])
        scale_f  = float(np.asarray(aligned_composite['_transform_scale_factor']).flat[0])

        # Reconstruct the 2×3 affine matrix equivalent to cv2.getRotationMatrix2D.
        # Positive angle = counter-clockwise in standard image coords.
        angle_rad = np.deg2rad(rot_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        cx, cy = ref_w / 2.0, ref_h / 2.0
        alpha = scale_f * cos_a
        beta  = scale_f * sin_a
        M = np.array([
            [alpha,  beta,  (1 - alpha) * cx - beta  * cy + dx],
            [-beta,  alpha,  beta       * cx + (1 - alpha) * cy + dy],
        ])

        # Correct scale: composite coords are in a halved widefield space
        # (TIF downsampled 2×), so we scale by ref_h / (TIF_size / 2).
        # resized_fullimg is the TIF halved once, so its height equals TIF/2.
        composite_size = float(resized_fullimg.shape[0])  # e.g. 1024
        scale_to_sm = ref_h / composite_size

        print(f'  -> VFS transform: scale_to_sm={scale_to_sm:.4f}, '
              f'rot={rot_deg:.1f}°, scale_f={scale_f:.3f}, '
              f'dx={dx:.1f}, dy={dy:.1f}')

        # Load reference VFS contours (in 0-400 reference VFS space).
        _here = os.path.dirname(os.path.abspath(__file__))
        vfs_contours_path = os.path.join(_here, 'utils', 'vfs_contours.json')
        with open(vfs_contours_path, 'r') as _f:
            vfs_contours_raw = json.load(_f)

        area_paths = {}
        for area_name, coords in vfs_contours_raw.items():
            if area_name not in _AREA_IDS:
                continue
            if coords is None or len(coords) < 3:
                continue
            area_paths[area_name] = Path(np.array(coords, dtype=float))
        print(f'  -> Built VFS-space paths for {len(area_paths)} areas: {list(area_paths.keys())}')

        # Build a canonical key map so pos01 ↔ pos1 both resolve.
        composite_key_map = {}
        for ck in aligned_composite.keys():
            composite_key_map[ck] = ck
            if ck.startswith('pos'):
                try:
                    num = int(ck[3:])
                    composite_key_map['pos{:02d}'.format(num)] = ck
                    composite_key_map['pos{}'.format(num)]     = ck
                except ValueError:
                    pass

        for pos_str in list(full_dict.keys()):
            composite_key = composite_key_map.get(pos_str)
            if composite_key is None:
                continue
            cell_transforms = aligned_composite[composite_key]
            if not isinstance(cell_transforms, np.ndarray) or cell_transforms.ndim < 2:
                continue
            if cell_transforms.shape[1] < 2:
                continue

            n_cells = cell_transforms.shape[0]
            area_ids = np.full(n_cells, 0, dtype=int)
            vfs_pos  = np.full((n_cells, 2), np.nan)

            for ci in range(n_cells):
                # Composite coords (cols 0,1) — legacy widefield-halved space.
                x_comp = float(cell_transforms[ci, 0])
                y_comp = float(cell_transforms[ci, 1])

                # Scale to sign-map resolution then apply VFS affine transform.
                x_sm = x_comp * scale_to_sm
                y_sm = y_comp * scale_to_sm
                x_ref = M[0, 0] * x_sm + M[0, 1] * y_sm + M[0, 2]
                y_ref = M[1, 0] * x_sm + M[1, 1] * y_sm + M[1, 2]

                vfs_pos[ci, 0] = x_ref
                vfs_pos[ci, 1] = y_ref

                for area_name, path_obj in area_paths.items():
                    if path_obj.contains_point((x_ref, y_ref)):
                        area_ids[ci] = _AREA_IDS[area_name]
                        break

            full_dict[pos_str]['visual_area_id'] = area_ids
            full_dict[pos_str]['vfs_cell_pos'] = vfs_pos

            n_labeled = np.count_nonzero(area_ids)
            print(f'  -> {pos_str}: {n_labeled}/{n_cells} cells assigned to an area')

        full_dict['_ref_vfs_shape'] = np.array(ref_vfs_shape)
        print('  -> Area labels assigned from VFS contours.')

    except Exception as e:
        import traceback
        print(f'  Warning: could not assign VFS area labels: {e}')
        traceback.print_exc()

    # save as v5 (jan 17)
    savepath = os.path.join(map_dir, '{}_merged_essentials_v9.h5'.format(animalID))
    fm2p.write_h5(savepath, full_dict)

    print('Wrote {}'.format(savepath))


def plot_running_median(ax, x, y, n_bins=7, vertical=False):

    bins = np.linspace(np.min(x), np.max(x), n_bins)

    bin_means, bin_edges, _ = scipy.stats.binned_statistic(
        x[~np.isnan(x) & ~np.isnan(y)],
        y[~np.isnan(x) & ~np.isnan(y)],
        statistic=np.median,
        bins=bins)
    
    bin_std, _, _ = scipy.stats.binned_statistic(
        x[~np.isnan(x) & ~np.isnan(y)],
        y[~np.isnan(x) & ~np.isnan(y)],
        statistic=np.nanstd,
        bins=bins)
    
    hist, _ = np.histogram(
        x[~np.isnan(x) & ~np.isnan(y)],
        bins=bins)
    
    tuning_err = bin_std / np.sqrt(hist)

    if not vertical:
        ax.plot(bin_edges[:-1] + (np.median(np.diff(bins))/2),
                bin_means,
                '-', color='k')
        
        ax.fill_between(bin_edges[:-1] + (np.median(np.diff(bins))/2),
                        bin_means-tuning_err,
                        bin_means+tuning_err,
                        color='k', alpha=0.2)
    
    elif vertical:
        ax.plot(bin_means,
                bin_edges[:-1] + (np.median(np.diff(bins))/2),
                '-', color='k')
        
        ax.fill_betweenx(bin_edges[:-1] + (np.median(np.diff(bins))/2),
                        bin_means-tuning_err,
                        bin_means+tuning_err,
                        color='k', alpha=0.2)


def visualize_topographic_map(messentials, composite, key, cond):

    cmap = cm.seismic
    norm = colors.Normalize(vmin=-1, vmax=1)

    h_hist_data = []
    v_hist_data = []

    fig = plt.figure(figsize=(6,6), dpi=300)

    gs = GridSpec(5,5)

    ax = fig.add_subplot(gs[1:5, 0:4])

    ax.imshow(messentials['rigid_tiled_map'], cmap='gray', alpha=0.5)
    ax.imshow(messentials['sign_map'], cmap='jet', alpha=0.15)

    for poskey in composite.keys():
        for c in range(np.size(messentials[poskey]['rdata']['{}_isrel'.format(key)], 0)):
            cellx = composite[poskey][c,2]
            celly = composite[poskey][c,3]
            cellrel = messentials[poskey]['rdata']['{}_isrel'.format(key)][c]

            if cellrel:
                cellmod = messentials[poskey]['rdata']['{}_{}_mod'.format(key, cond)][c]
                ax.plot(cellx, celly, '.', ms=3, color=cmap(norm(cellmod)))
                h_hist_data.append([cellx, cellmod])
                v_hist_data.append([celly, cellmod])

            elif not cellrel:
                ax.plot(cellx, celly, '.', ms=3, color='gray', alpha=0.2)

    ax_histx  = fig.add_subplot(gs[0, 0:4], sharex=ax)
    ax_histy  = fig.add_subplot(gs[1:5, 4], sharey=ax)

    h_hist_data = np.array(h_hist_data)
    v_hist_data = np.array(v_hist_data)

    plot_running_median(ax_histx, h_hist_data[:,0], h_hist_data[:,1], 9)
    plot_running_median(ax_histy, v_hist_data[:,0], v_hist_data[:,1], 9, vertical=True)

    fig.suptitle('{} ({})'.format(key, cond))

    fig.tight_layout()

    return fig


if __name__ == '__main__':

    animal_ids = ['DMM037','DMM041','DMM042','DMM056','DMM061']
    for animal_id in animal_ids:
        print('Merging data for {}'.format(animal_id))
        merge_animal_essentials(animal_id)
