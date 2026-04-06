# -*- coding: utf-8 -*-

if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom
from skimage.measure import label
import scipy.stats
from scipy.stats import kruskal, mannwhitneyu
from scipy.interpolate import griddata
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import skimage.transform
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
try:
    import umap
    _umap_available = True
except ModuleNotFoundError:
    _umap_available = False

from .utils.cmap import make_parula
from .utils.files import read_h5, write_h5
from .utils.paths import find, choose_most_recent
from .utils.time import interpT
from .utils.ref_frame import get_ang_offset
from .utils.imu import check_and_trim_imu_disconnect
from .utils.PETH import analyze_gaze_state_changes


def plot_running_median(ax, x, y, n_bins=7, vertical=False, fb=True, color='k'):

    mask = ~np.isnan(x)
    
    if np.sum(mask) == 0:
        return np.nan

    x_use = x[mask]
    y_use = y[mask]

    bins = np.linspace(np.min(x_use), np.max(x_use), n_bins)

    bin_means, bin_edges, _ = scipy.stats.binned_statistic(
        x_use,
        y_use,
        statistic=np.nanmedian,
        bins=bins)
    
    bin_std, _, _ = scipy.stats.binned_statistic(
        x_use,
        y_use,
        statistic=np.nanstd,
        bins=bins)
    
    hist, _, _ = scipy.stats.binned_statistic(
        x_use,
        y_use,
        statistic=lambda y: np.sum(~np.isnan(y)),
        bins=bins)
    
    tuning_err = bin_std / np.sqrt(hist)

    if not vertical:
        ax.plot(bin_edges[:-1] + (np.median(np.diff(bins))/2),
                bin_means,
                '-', color=color)
        if fb:
            ax.fill_between(bin_edges[:-1] + (np.median(np.diff(bins))/2),
                        bin_means-tuning_err,
                        bin_means+tuning_err,
                        color=color, alpha=0.2)
        return np.nanmax(bin_means + tuning_err)
    
    elif vertical:
        ax.plot(bin_means,
                bin_edges[:-1] + (np.median(np.diff(bins))/2),
                '-', color=color)
        
        if fb:
            ax.fill_betweenx(bin_edges[:-1] + (np.median(np.diff(bins))/2),
                        bin_means-tuning_err,
                        bin_means+tuning_err,
                        color=color, alpha=0.2)
        return np.nanmax(bin_means + tuning_err)


def shift_image(image, dx, dy):
    translation_vector = (dx, dy) 
    transform = skimage.transform.AffineTransform(
        translation=translation_vector
    )
    shifted_image = skimage.transform.warp(
        image,
        transform,
        mode='constant',
        preserve_range=True
    )
    return shifted_image




def make_earth_tones():
    """ Create a custom categorical earth-tone colormap with 10 colors in pairs.

    The pairs are:
        1. Moss & Sage (Green)
        2. Clay & Sand (Brown)
        3. Slate & Sky (Blue-Grey)
        4. Rust & Peach (Red-Orange)
        5. Ochre & Straw (Yellow)
    """

    colors = [
        '#2ECC71', '#82E0AA', # Green
        '#FF9800', '#FFCC80', # Orange
        '#03A9F4', '#81D4FA', # Blue
        '#9C27B0', '#E1BEE7', # Purple
        '#FFEB3B', '#FFF59D'  # Yellow
    ]
    rgb_colors = [tuple(int(h.lstrip('#')[i:i+2], 16) / 255.0 for i in (0, 2, 4)) for h in colors]

    earth_map = LinearSegmentedColormap.from_list('earth_tones', rgb_colors, N=10)

    return earth_map


def make_area_colors():
    """ Create 4 distinct colors for V1, RL, AM, PM.
    Distinct from earth tones (Green, Brown, Blue-Grey, Red-Orange, Yellow).
    """
    return get_equally_spaced_colormap_values('Dark2', 4)


def get_equally_spaced_colormap_values(colormap_name, num_values):
    if not isinstance(num_values, int) or num_values <= 0:
        raise ValueError("num_values must be a positive integer.")
    if colormap_name == 'parula':
        cmap = make_parula()
    elif colormap_name == 'earth_tones':
        cmap = make_earth_tones()
    else:
        cmap = cm.get_cmap(colormap_name)
    normalized_positions = np.linspace(0, 1, num_values)
    colors = [cmap(pos) for pos in normalized_positions]
    return colors

goodred = '#D96459'


def add_scatter_col(ax, pos, vals, color='k'):

    vals = pd.to_numeric(vals, errors='coerce')
    vals = np.array(vals)
    vals = vals[~np.isnan(vals)]
    
    if len(vals) == 0:
        return

    ax.scatter(
        np.ones_like(vals)*pos + (np.random.rand(len(vals))-0.5)/2,
        vals,
        s=2, c=color
    )
    ax.hlines(np.nanmean(vals), pos-.1, pos+.1, color='k')

    stderr = np.nanstd(vals) / np.sqrt(len(vals))
    ax.vlines(pos, np.nanmean(vals)-stderr, np.nanmean(vals)+stderr, color='k')


def make_aligned_sign_maps(map_items, animal_dirs, pdf=None):

    uniref = map_items['uniref']
    main_basepath = map_items['composite_basepath']
    img_array = map_items['img_array']

    fig, ax = plt.subplots(1, 1, figsize=(2,2), dpi=300)

    ax.imshow(gaussian_filter(zoom(uniref['overlay'][:,:,0], 2.555),12), cmap='gray', alpha=0.3)

    for animal_dir in animal_dirs:

        basepath = os.path.join(main_basepath, animal_dir)
        if animal_dir != 'DMM056':
            transform_g2u = read_h5(find('aligned_composite_*.h5', basepath, MR=True))
            messentials = read_h5(find('*_merged_essentials_v6.h5', basepath, MR=True))
        else:
            continue

        k = list(transform_g2u.keys())[0]
        x_shift = transform_g2u[k][0][2] - transform_g2u[k][0][0]
        y_shift = transform_g2u[k][0][3] - transform_g2u[k][0][1]

        shifted_sign_map = shift_image(messentials['sign_map'], -x_shift, -y_shift)

        ax.imshow(gaussian_filter(shifted_sign_map, 10), alpha=0.3, cmap='jet')

    ax.set_title('Aligned sign maps')

    ax.imshow(img_array)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    ax.set_ylim([1022, 0])
    ax.set_xlim([0,1022])

    fig.tight_layout()

    if pdf is not None:
        pdf.savefig(fig)
        plt.close(fig)
    else:
        plt.show()


def get_labeled_array_from_contours(pooled_data):
    """Build a labeled array from VFS reference contours stored in pooled_data.

    Replaces the old get_labeled_array() which derived region boundaries from
    a hand-drawn PNG.  The contours come from register_animals_using_shared_template()
    (via make_pooled_dataset()) and are in reference VFS coordinate space.

    Parameters
    ----------
    pooled_data : dict
        The pooled dataset as loaded by topography.py main().  Must contain
        keys of the form 'ref_contour_<area_name>' (ndarray of (x,y) coords
        in reference VFS space) and optionally 'ref_vfs_shape'.

    Returns
    -------
    labeled_array : ndarray
        Integer array in reference VFS pixel coordinates.
    label_map : dict
        Integer label ID -> area name string, consistent with topography_plots.py.
    """
    from matplotlib.path import Path

    AREA_IDS = {'RL': 2, 'AM': 3, 'PM': 4, 'V1': 5, 'AL': 7, 'LM': 8, 'P': 9, 'A': 10}
    label_map = {
        0: 'boundary', 1: 'outside',
        2: 'RL', 3: 'AM', 4: 'PM', 5: 'V1',
        7: 'AL', 8: 'LM', 9: 'P', 10: 'A'
    }

    ref_vfs_shape_arr = pooled_data.get('ref_vfs_shape', np.array([400, 400]))
    shape = tuple(np.asarray(ref_vfs_shape_arr).astype(int))

    labeled_array = np.zeros(shape, dtype=int)
    h, w = shape

    grid_y, grid_x = np.mgrid[:h, :w]
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    for key, coords in pooled_data.items():
        if not isinstance(key, str) or not key.startswith('ref_contour_'):
            continue
        area_name = key[len('ref_contour_'):]
        if area_name not in AREA_IDS:
            continue
        if not isinstance(coords, np.ndarray) or coords.shape[0] < 3:
            continue
        pts = coords[:, :2].astype(float)
        try:
            path = Path(pts)
            mask = path.contains_points(grid_pts).reshape(shape)
            labeled_array[mask] = AREA_IDS[area_name]
        except Exception:
            continue

    # If no contours were embedded in pooled_data, fall back to building the
    # labeled array from per-cell visual_area_id + VFS positions stored in
    # the per-animal messentials data.
    if np.all(labeled_array == 0):
        print('get_labeled_array_from_contours: no ref_contour_* keys found; '
              'building labeled_array from visual_area_id in messentials.')
        try:
            from scipy.interpolate import NearestNDInterpolator
            xs, ys, ids = [], [], []
            for ak, av in pooled_data.items():
                if not isinstance(av, dict):
                    continue
                transform_d = av.get('transform', {})
                messentials_d = av.get('messentials', {})
                if not isinstance(transform_d, dict) or not isinstance(messentials_d, dict):
                    continue
                for pk, cell_arr in transform_d.items():
                    if not (isinstance(pk, str) and pk.startswith('pos')):
                        continue
                    if not isinstance(cell_arr, np.ndarray) or cell_arr.ndim < 2 or cell_arr.shape[1] < 4:
                        continue
                    pos_data = messentials_d.get(pk)
                    if not isinstance(pos_data, dict):
                        continue
                    va = pos_data.get('visual_area_id')
                    if va is None:
                        continue
                    va = np.asarray(va, dtype=int)
                    n = min(len(va), cell_arr.shape[0])
                    xv = cell_arr[:n, 2].astype(float)
                    yv = cell_arr[:n, 3].astype(float)
                    valid = np.isfinite(xv) & np.isfinite(yv) & (va[:n] > 1)
                    xs.extend(xv[valid])
                    ys.extend(yv[valid])
                    ids.extend(va[:n][valid])
            if len(xs) >= 3:
                h2, w2 = shape
                gy, gx = np.mgrid[:h2, :w2]
                interp = NearestNDInterpolator(
                    np.column_stack([xs, ys]), np.array(ids))
                labeled_array = interp(
                    gx.ravel(), gy.ravel()).reshape(shape).astype(int)
        except Exception as e:
            print(f'  Warning: fallback labeled_array construction failed: {e}')

    return labeled_array, label_map


def get_region_for_points(labeled_array, points_to_check, label_map):
    rows, cols = labeled_array.shape
    results = []
    for i, (x, y) in enumerate(points_to_check):
        if 0 <= y < rows and 0 <= x < cols:
            region_id = labeled_array[int(y), int(x)]
            results.append([i, y, x, region_id])
        else:
            results.append([i, y, x, -1])

    results = np.array(results)

    return results


def _get_cell_regions(messentials_pos, n_cells, labeled_array, x_vfs, y_vfs):
    """Return integer region IDs for n_cells, using visual_area_id if available.

    Tries ``messentials_pos['visual_area_id']`` first (pre-computed by
    merge_animal_essentials.py from WF-space contours).  Falls back to a
    per-pixel spatial lookup in *labeled_array* when the key is absent.
    """
    if isinstance(messentials_pos, dict):
        va = messentials_pos.get('visual_area_id')
        if va is not None:
            va = np.asarray(va, dtype=int)
            if len(va) >= n_cells:
                return va[:n_cells]
    # Spatial fallback
    h, w = labeled_array.shape
    regions = np.zeros(n_cells, dtype=int)
    for i in range(min(n_cells, len(x_vfs))):
        x, y = float(x_vfs[i]), float(y_vfs[i])
        if 0 <= y < h and 0 <= x < w:
            regions[i] = int(labeled_array[int(y), int(x)])
    return regions


def get_cell_data(rdata, key, cond, cv_thresh=0.1):
    """Return (isrel, mod, peak, cv_mi) for *key* and condition *cond*.

    Both *isrel* (bool flag) and *mod* (continuous value) are derived from the
    cross-validated MI stored under ``{key}_{cond}_rel``.  Old ``_ismod`` /
    ``_mod`` keys are no longer read.
    """
    use_key = key
    reverse_map = {'dRoll': 'gyro_x', 'dPitch': 'gyro_y', 'dYaw': 'gyro_z'}
    if key in reverse_map:
        use_key = reverse_map[key]

    cv_mi = None
    peak  = None

    rel_key = f'{use_key}_{cond}_rel'
    if rel_key in rdata:
        cv_mi = np.asarray(rdata[rel_key], dtype=float)

    # isrel and mod are derived from CV-MI for caller compatibility
    isrel = (cv_mi > cv_thresh).astype(int) if cv_mi is not None else None
    mod   = cv_mi  # callers that plot 'mod' will now show CV-MI

    pref_key = f'{use_key}_{cond}_pref'
    if pref_key in rdata:
        peak = rdata[pref_key]
    else:
        tuning_key = f'{use_key}_1dtuning'
        bins_key   = f'{use_key}_1dbins'

        if tuning_key in rdata and bins_key in rdata:
            tuning = rdata[tuning_key]
            bins   = rdata[bins_key]

            cond_idx = 1 if cond == 'l' else 0

            if tuning.ndim == 3 and tuning.shape[2] > cond_idx:
                peak_indices = np.argmax(tuning[:, :, cond_idx], axis=1)
                peak = bins[peak_indices]

    return isrel, mod, peak, cv_mi


def get_glm_keys(key, cond=None):

    map_imp = {
        'theta': 'theta',
        'phi': 'phi',
        'dTheta': 'dTheta',
        'dPhi': 'dPhi',
        'pitch': 'pitch',
        'yaw': 'yaw',
        'roll': 'roll',
        'dPitch': 'gyro_y',
        'dYaw': 'gyro_z',
        'dRoll': 'gyro_x'
    }
    
    if key in map_imp:
        if cond == 'l':
            return f'full_trainLight_testLight_importance_{map_imp[key]}'
        elif cond == 'd':
            return f'full_trainDark_testDark_importance_{map_imp[key]}'
        else:
            return f'full_importance_{map_imp[key]}'
    return None


def create_smoothed_map(x, y, values, shape=(1024, 1024), sigma=25): # started as sigma=10
    
    if len(values) < 4:
        return np.full(shape, np.nan)

    grid_y, grid_x = np.mgrid[0:shape[0], 0:shape[1]]
    points = np.column_stack((y, x))
    
    smoothed = griddata(points, values, (grid_y, grid_x), method='linear')
    
    # nan-aware gaussian smoothing
    V = smoothed.copy()
    V[np.isnan(V)] = 0
    VV = gaussian_filter(V, sigma=sigma)

    W = 0*smoothed.copy() + 1
    W[np.isnan(smoothed)] = 0
    WW = gaussian_filter(W, sigma=sigma)

    smoothed = np.divide(VV, WW, out=np.full_like(VV, np.nan), where=WW!=0)
    
    return smoothed


def plot_region_outlines(pdf, labeled_array, label_map):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    
    for region_id, region_name in label_map.items():
        if region_id <= 1: continue
        
        mask = (labeled_array == region_id).astype(float)
        contours = ax.contour(mask, levels=[0.5], colors='k', linewidths=1)
        
        y, x = np.where(mask)
        if len(x) > 0:
            ax.text(np.mean(x), np.mean(y), region_name, ha='center', va='center', fontsize=8, fontweight='bold')

    center_x, center_y = labeled_array.shape[1] // 2, labeled_array.shape[0] // 2
    arrow_len = 300
    
    theta = np.pi / 4
    dx = arrow_len * np.cos(theta)
    dy = arrow_len * np.sin(theta)
    
    ax.arrow(center_x, center_y, dx, -dy, head_width=20, head_length=20, fc='tab:blue', ec='tab:blue', label='X-axis')
    ax.arrow(center_x, center_y, dy, dx, head_width=20, head_length=20, fc='tab:red', ec='tab:red', label='Y-axis')
    
    ax.invert_yaxis()
    ax.axis('off')
    ax.legend(loc='upper right')
    pdf.savefig(fig)
    plt.close(fig)

def plot_variable_summary(pdf, data, key, cond, uniref, img_array, animal_dirs, labeled_array, label_map):
    
    imp_key = get_glm_keys(key, cond=cond)
    
    cells = []

    for animal_dir in animal_dirs:
        if animal_dir not in data.keys():
            continue
            
        if 'transform' not in data[animal_dir].keys():
             continue

        for poskey in data[animal_dir]['transform'].keys():
            if not poskey.startswith('pos'):
                continue
            if (animal_dir=='DMM056') and (cond=='d') and ((poskey=='pos15') or (poskey=='pos03')):
                continue

            rdata = data[animal_dir]['messentials'][poskey]['rdata']
            isrel, mod, peak, _ = get_cell_data(rdata, key, cond)

            if isrel is None:
                continue

            transform = data[animal_dir]['transform'][poskey]

            model_data = data[animal_dir]['messentials'][poskey].get('model', {})

            # Compute mean firing rate from tuning curve as a proxy for responsiveness
            _use_key_rate = key
            _rmap = {'dRoll': 'gyro_x', 'dPitch': 'gyro_y', 'dYaw': 'gyro_z'}
            if key in _rmap:
                _use_key_rate = _rmap[key]
            _tc_key = f'{_use_key_rate}_1dtuning'
            mean_rates = None
            if _tc_key in rdata:
                _tc = rdata[_tc_key]
                _ci = 1 if cond == 'l' else 0
                if _tc.ndim == 3 and _tc.shape[2] > _ci:
                    mean_rates = np.nanmean(_tc[:, :, _ci], axis=1)
                elif _tc.ndim == 2:
                    mean_rates = np.nanmean(_tc, axis=1)

            n_cells_pos = np.size(isrel, 0)
            pos_regions = _get_cell_regions(
                data[animal_dir]['messentials'][poskey],
                n_cells_pos, labeled_array,
                transform[:n_cells_pos, 2], transform[:n_cells_pos, 3])

            for c in range(n_cells_pos):
                c_imp = np.nan
                c_r2 = np.nan

                if isinstance(model_data, dict):
                    if imp_key and imp_key in model_data and c < len(model_data[imp_key]):
                        c_imp = model_data[imp_key][c]
                    if 'full_r2' in model_data and c < len(model_data['full_r2']):
                        c_r2 = model_data['full_r2'][c]

                cells.append({
                    'x': transform[c, 2],
                    'y': transform[c, 3],
                    'region': int(pos_regions[c]),
                    'rel': isrel[c],
                    'mod': mod[c],
                    'peak': peak[c] if peak is not None else np.nan,
                    'imp': c_imp, # feature importance
                    'full_r2': c_r2,
                    'mean_rate': mean_rates[c] if (mean_rates is not None and c < len(mean_rates)) else np.nan,
                })

    if not cells:
        return {}

    area_colors = make_area_colors()

    df = pd.DataFrame(cells)

    cond_name = 'Light' if cond == 'l' else 'Dark'

    metrics_to_plot = ['mod']
    if key not in ['dTheta', 'dPhi']:
        metrics_to_plot.append('peak')
    metrics_to_plot.append('imp')

    if key in ['dYaw', 'dPitch', 'dRoll']:
        metrics_to_plot = [m for m in metrics_to_plot if m != 'peak']

    for metric in metrics_to_plot:
        
        if metric == 'mod':
            cmap = cm.plasma
            norm = colors.Normalize(vmin=0, vmax=0.5)
            label_str = 'CV-MI'
        elif metric == 'peak':
            if key in ['theta', 'phi', 'yaw', 'roll', 'pitch']:
                cmap = cm.coolwarm
                norm = colors.Normalize(vmin=-15, vmax=15)
                label_str = f'{key} Peak (deg)'
            else:
                limit = np.nanpercentile(np.abs(df['peak']), 95)
                cmap = cm.plasma
                norm = colors.Normalize(vmin=-limit, vmax=limit)
                label_str = f'{key} Peak'
        elif metric == 'imp':
            cmap = cm.plasma
            norm = colors.Normalize(vmin=0, vmax=0.1)
            label_str = 'Variable Importance (Shuffle)'

        # Minimum mean firing rate to include a cell (normalized spike rate units)
        MIN_RATE = 0.05

        if metric == 'peak':
            rel = df[(df['rel'] == 1) & (df['mod'] > 0.33)]
        elif metric == 'imp':
            rel = df[(df['rel'] == 1) & (df['full_r2'] > 0.1)]
        elif metric == 'mod':
            # All cells above firing rate threshold, not just statistically reliable ones
            rel = df[df['mean_rate'] > MIN_RATE]
        else:
            rel = df[df['rel'] == 1]

        if len(rel) == 0:
            continue

        # rel already has 'region' from df

        if metric == 'mod':
            # Show as histogram per region (works better with all cells)
            region_names_ordered = list(label_map.values())[2:6]
            for show_stats in [True, False]:
                fig, ax = plt.subplots(1, 1, figsize=(5, 3.5), dpi=300)
                hist_bins = np.linspace(0, 0.75, 25)
                groups = []
                for i in range(4):
                    region_vals = rel[rel['region'] == i+2]['mod'].dropna().values
                    groups.append(region_vals)
                    if len(region_vals) > 0:
                        ax.hist(region_vals, bins=hist_bins, histtype='step', density=True,
                                color=area_colors[i], label=f'{region_names_ordered[i]} (n={len(region_vals)})',
                                linewidth=1.5)
                ax.axvline(0.33, color='tab:grey', ls='--', alpha=0.56, linewidth=0.8)
                ax.axvline(0.5, color='tab:grey', ls='--', alpha=0.56, linewidth=0.8)

                if show_stats:
                    valid_groups = [g for g in groups if len(g) > 0]
                    if len(valid_groups) > 1:
                        try:
                            stat, p_kw = kruskal(*valid_groups)
                            ax.text(0.05, 0.95, f'KW p={p_kw:.1e}', transform=ax.transAxes, fontsize=8)
                        except ValueError:
                            pass

                ax.set_xlim([0, 0.75])
                ax.set_xlabel('CV-MI')
                ax.set_ylabel('Density')
                ax.set_title(f'{key} CV-MI by Region ({cond_name}) — all cells, rate>{MIN_RATE:.2f}')
                ax.legend(fontsize=7, frameon=False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        else:
            # Scatter column plots for peak and imp metrics (reliable cells only)
            for show_stats in [True, False]:
                fig, ax = plt.subplots(1, 1, figsize=(5, 3.5), dpi=300)

                groups = []
                for i in range(4):
                    region_vals = rel[rel['region'] == i+2][metric]
                    groups.append(region_vals)
                    if len(region_vals) > 0:
                        add_scatter_col(ax, i, region_vals, color=area_colors[i])

                if show_stats:
                    valid_groups = [g[~np.isnan(g)] for g in groups if len(g[~np.isnan(g)]) > 0]
                    if len(valid_groups) > 1:
                        try:
                            stat, p_kw = kruskal(*valid_groups)
                            ax.text(0.05, 0.95, f'KW p={p_kw:.1e}', transform=ax.transAxes, fontsize=8)
                        except ValueError:
                            pass
                    else:
                        ax.text(0.05, 0.95, 'Insufficient data for stats', transform=ax.transAxes, fontsize=6)

                ax.set_xticks(np.arange(4), labels=list(label_map.values())[2:6])

                if key in ['pitch', 'roll'] and metric in ['peak']:
                    ax.set_ylim([-35, 35])
                elif metric == 'imp':
                    ax.set_ylim([0, 0.25])
                elif metric == 'peak':
                    if key in ['theta', 'phi', 'yaw', 'roll', 'pitch']:
                        ax.set_ylim([-15, 15])

                ax.set_xlim([-.5, 3.5])
                ax.set_ylabel(label_str)
                plt.title(f'{key} {metric} by Region ({cond_name})')
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        fig, axs = plt.subplots(2, 2, dpi=300, figsize=(8,6))
        axs = axs.flatten()

        for i in range(4):
            region_id = [2,3,4,5][i]
            region_name = label_map.get(region_id, f'Region {region_id}')
            
            region_mask = (labeled_array == region_id)
            axs[i].imshow(region_mask, cmap='gray_r', alpha=0.4)
            axs[i].imshow(labeled_array == 0, cmap='binary', alpha=0.1)
            axs[i].contour(region_mask, levels=[0.5], colors='k', linestyles='--', linewidths=0.5)
            
            region_cells = rel[rel['region'] == region_id]
            if len(region_cells) > 0:
                axs[i].scatter(region_cells['x'], region_cells['y'], s=2, c=region_cells[metric], cmap=cmap, norm=norm)
            
            axs[i].set_title(f'{region_name}')
            axs[i].axis('off')
            axs[i].set_ylim([labeled_array.shape[0], 0])
            axs[i].set_xlim([0, labeled_array.shape[1]])

        fig.suptitle(f'{key} {metric} Map by Region ({cond_name})')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axs = plt.subplots(2, 2, dpi=300, figsize=(8,6))
        axs = axs.flatten()

        for i in range(4):
            region_id = [2,3,4,5][i]
            region_name = label_map.get(region_id, f'Region {region_id}')
            
            region_mask = (labeled_array == region_id)
            axs[i].imshow(labeled_array == 0, cmap='binary', alpha=0.1)
            axs[i].contour(region_mask, levels=[0.5], colors='k', linestyles='--', linewidths=0.5)
            
            region_cells = rel[rel['region'] == region_id]
            if len(region_cells) > 0:
                smoothed = create_smoothed_map(region_cells['x'].values, region_cells['y'].values, region_cells[metric].values, shape=labeled_array.shape, sigma=50)
                smoothed[~region_mask] = np.nan
                axs[i].imshow(smoothed, cmap=cmap, norm=norm)
            
            axs[i].set_title(f'{region_name}')
            axs[i].axis('off')
            axs[i].set_ylim([labeled_array.shape[0], 0])
            axs[i].set_xlim([0, labeled_array.shape[1]])

        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, label=label_str, fraction=0.05, shrink=0.6)
        fig.suptitle(f'{key} {metric} Smoothed Map by Region ({cond_name})')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if metric == 'mod' or metric == 'peak':
            fig_rot, (ax_rot_x, ax_rot_y) = plt.subplots(1, 2, figsize=(5, 2.5), dpi=300)
            
            theta = np.pi / 4  # 45 deg CCW
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            
            rel_rot = rel.copy()
            rel_rot['x_rot'] = rel['x'] * cos_t - rel['y'] * sin_t
            rel_rot['y_rot'] = rel['x'] * sin_t + rel['y'] * cos_t

            max_y_val = 0

            for i in range(4):
                region_id = [2,3,4,5][i]
                region_name = label_map.get(region_id, f'Region {region_id}')
                
                region_cells = rel_rot[rel_rot['region'] == region_id]
                
                if len(region_cells) > 0:
                    x_rot_norm = (region_cells['x_rot'] - region_cells['x_rot'].min()) / (region_cells['x_rot'].max() - region_cells['x_rot'].min())
                    y_rot_norm = (region_cells['y_rot'] - region_cells['y_rot'].min()) / (region_cells['y_rot'].max() - region_cells['y_rot'].min())

                    my1 = plot_running_median(ax_rot_x, x_rot_norm, region_cells[metric], n_bins=5, fb=True, color=area_colors[i])
                    if i == 0:
                        ax_rot_x.plot([], [], color=area_colors[i], label=region_name)

                    my2 = plot_running_median(ax_rot_y, y_rot_norm, region_cells[metric], n_bins=5, fb=True, color=area_colors[i])
                    ax_rot_y.plot([], [], color=area_colors[i], label=region_name)
                    
                    max_y_val = np.nanmax([max_y_val, my1, my2])

            if metric == 'mod':
                ax_rot_x.set_ylim(bottom=0.0, top=max_y_val*1.1)
                ax_rot_y.set_ylim(bottom=0.0, top=max_y_val*1.1)
            elif metric == 'peak':
                lim = 15
                if key in ['pitch', 'roll']:
                    lim = 35
                    ax_rot_x.set_ylim([-lim, lim])
                    ax_rot_y.set_ylim([-lim, lim])
                else:
                    ax_rot_x.set_ylim([-15, 15])
                    ax_rot_y.set_ylim([-15, 15])

            ax_rot_x.set_title('Along Rotated X-axis (45 deg CCW)')
            ax_rot_x.set_xlabel('Position along rotated axis')
            if metric == 'mod':
                ax_rot_x.set_ylabel('Modulation Index')
            elif metric == 'peak':
                ax_rot_x.set_ylabel('Peak Position')
            ax_rot_x.legend()
            if metric == 'mod':
                ax_rot_y.set_ylabel('Modulation Index')
            elif metric == 'peak':
                ax_rot_y.set_ylabel('Peak Position')
            ax_rot_x.legend()

            ax_rot_y.set_title('Along Rotated Y-axis (45 deg CCW)')
            ax_rot_y.set_xlabel('Position along rotated axis')
            ax_rot_y.legend()

            fig_rot.suptitle(f'{label_str} along Rotated Axes for {key} ({cond_name})')
            fig_rot.tight_layout()
            pdf.savefig(fig_rot)
            plt.close(fig_rot)
            
    return {f'variable_summary_{key}_{cond}': df.to_dict(orient='list')}


def plot_signal_noise_correlations(pdf, data, key, cond, animal_dirs, labeled_array, label_map):
    """Compute and plot signal and noise correlations between cell pairs.

    Signal correlation is computed from per-variable single-input model predictions
    (y_hat from a model trained on only `key`'s variable), which captures the
    similarity in how each behavioral variable drives pairs of cells.  Falls back
    to 1D tuning-curve correlation when model predictions are unavailable.

    Noise correlation is computed from full-model residuals (y_true − y_hat_full),
    which represent shared variability NOT explained by any behavioral predictor.

    A positive relationship between signal and noise correlation is consistent
    with shared presynaptic inputs or common neuromodulation.
    """
    print(f"Calculating signal/noise correlations for {key} ({cond})")

    pooled_sig = []
    pooled_noise = []
    pooled_regions = []
    pooled_fov = []
    fov_idx = 0
    area_colors = make_area_colors()

    cond_name = 'Light' if cond == 'l' else 'Dark'
    cond_idx = 1 if cond == 'l' else 0

    for animal_dir in animal_dirs:
        if animal_dir not in data: continue
        if 'transform' not in data[animal_dir]: continue

        for poskey in data[animal_dir]['transform']:
            if not poskey.startswith('pos'):
                continue

            if (animal_dir == 'DMM056') and (cond == 'd') and (poskey in ('pos15', 'pos03')):
                continue

            messentials = data[animal_dir]['messentials'][poskey]
            rdata = messentials.get('rdata', {})
            model_data = messentials.get('model', {})
            transform = data[animal_dir]['transform'][poskey]

            # Resolve key alias (e.g. dRoll -> gyro_x, dPitch -> gyro_y, dYaw -> gyro_z)
            use_key = key
            _rmap = {'dRoll': 'gyro_x', 'dPitch': 'gyro_y', 'dYaw': 'gyro_z'}
            if key in _rmap:
                mapped = _rmap[key]
                use_key = mapped if (f'{mapped}_1dtuning' in rdata or
                                     f'{mapped}_train{cond_name}_test{cond_name}_y_hat' in model_data) else key

            # ── Signal correlation ────────────────────────────────────────────
            # Prefer single-variable model predictions over tuning curves.
            sig_corr_mat = None
            sig_yhat_key = f'{use_key}_train{cond_name}_test{cond_name}_y_hat'

            if isinstance(model_data, dict) and sig_yhat_key in model_data:
                y_hat_var = model_data[sig_yhat_key]   # (n_eval_frames, n_cells)
                if y_hat_var.ndim == 2 and y_hat_var.shape[1] >= 2:
                    n_cells = y_hat_var.shape[1]
                    sig_corr_mat = np.corrcoef(y_hat_var.T)   # (n_cells, n_cells)

            # Fall back to 1D tuning curve if model predictions not available
            if sig_corr_mat is None:
                tuning_key = f'{use_key}_1dtuning'
                if tuning_key not in rdata:
                    continue
                tuning_curves = rdata[tuning_key]   # (n_cells, n_bins[, n_conds])
                if tuning_curves.ndim == 3 and tuning_curves.shape[2] > cond_idx:
                    tc = tuning_curves[:, :, cond_idx]
                elif tuning_curves.ndim == 2:
                    tc = tuning_curves
                else:
                    continue
                n_cells = tc.shape[0]
                if n_cells < 2:
                    continue
                tc_filled = np.where(np.isnan(tc), np.nanmean(tc, axis=1, keepdims=True), tc)
                tc_filled[np.all(np.isnan(tc), axis=1)] = 0.0
                sig_corr_mat = np.corrcoef(tc_filled)
            else:
                if n_cells < 2:
                    continue

            # ── Noise correlation (full-model residuals) ──────────────────────
            noise_corr_mat = None
            full_prefix = f'full_train{cond_name}_test{cond_name}'
            y_true_cond_key = f'{full_prefix}_y_true'
            y_hat_cond_key  = f'{full_prefix}_y_hat'

            if (isinstance(model_data, dict) and
                    y_true_cond_key in model_data and y_hat_cond_key in model_data):
                y_true = model_data[y_true_cond_key]
                y_hat_full = model_data[y_hat_cond_key]
                if y_true.shape[1] == n_cells:
                    residuals = y_true - y_hat_full
                    noise_corr_mat = np.corrcoef(residuals.T)
            else:
                # Fall back to the global full_y_hat / full_y_true
                y_true_key_fb = ('full_y_test' if 'full_y_test' in model_data
                                 else ('full_y_true' if 'full_y_true' in model_data else None))
                if y_true_key_fb and 'full_y_hat' in model_data:
                    y_true = model_data[y_true_key_fb]
                    y_hat_full = model_data['full_y_hat']
                    if y_true.shape[1] == n_cells:
                        residuals = y_true - y_hat_full
                        noise_corr_mat = np.corrcoef(residuals.T)

            if noise_corr_mat is None:
                print(f"  Skipping {animal_dir} {poskey} — no model residuals")
                continue   # skip this recording — no noise data

            regions = _get_cell_regions(
                messentials, n_cells, labeled_array,
                transform[:, 2], transform[:, 3])

            iu = np.triu_indices(n_cells, k=1)
            n_pairs = len(iu[0])
            pooled_sig.extend(sig_corr_mat[iu])
            pooled_noise.extend(noise_corr_mat[iu])
            pooled_fov.extend([fov_idx] * n_pairs)
            fov_idx += 1

            for idx_i, idx_j in zip(iu[0], iu[1]):
                pooled_regions.append((regions[idx_i], regions[idx_j]))

    if not pooled_sig:
        print(f"No pooled data for {key} ({cond})")
        return {}

    pooled_sig     = np.array(pooled_sig,     dtype=float)
    pooled_noise   = np.array(pooled_noise,   dtype=float)
    pooled_regions = np.array(pooled_regions, dtype=int)
    pooled_fov     = np.array(pooled_fov,     dtype=int)

    pm_pairs = int(np.sum((pooled_regions[:, 0] == 4) & (pooled_regions[:, 1] == 4)))
    print(f"  PM-PM pairs: {pm_pairs}, total pairs: {len(pooled_sig)}")

    valid = np.isfinite(pooled_sig) & np.isfinite(pooled_noise)

    region_ids   = [5, 2, 3, 4]
    region_names = ['V1', 'RL', 'AM', 'PM']

    # ── Scatter: signal vs noise correlation ──────────────────────────────────
    fig, axs = plt.subplots(2, 3, figsize=(6, 4), dpi=300)
    axs = axs.flatten()

    ds = max(1, len(pooled_sig) // 3000)    # subsample for scatter speed
    v = valid
    axs[0].scatter(pooled_sig[v][::ds], pooled_noise[v][::ds], s=2, c='k', alpha=0.3)
    axs[0].set_title('All Pairs')
    axs[0].set_xlabel('Signal Corr (model pred)')
    axs[0].set_ylabel('Noise Corr (residuals)')
    axs[0].set_xlim([-1, 1]); axs[0].set_ylim([-1, 1])

    for i, (rid, rname) in enumerate(zip(region_ids, region_names)):
        mask = (pooled_regions[:, 0] == rid) & (pooled_regions[:, 1] == rid)
        ax = axs[i + 1]
        mask_valid = mask & valid
        if np.sum(mask_valid) > 1:
            ds_r = max(1, np.sum(mask_valid) // 2000)
            ax.scatter(pooled_sig[mask_valid][::ds_r], pooled_noise[mask_valid][::ds_r],
                       s=2, c=area_colors[i], alpha=0.3)
            r_val = np.corrcoef(pooled_sig[mask_valid], pooled_noise[mask_valid])[0, 1]
            ax.text(0.05, 0.9, f'r={r_val:.2f}', transform=ax.transAxes, fontsize=8)
        ax.set_title(f'{rname} Pairs')
        ax.set_xlabel('Signal Corr (model pred)')
        ax.set_ylabel('Noise Corr (residuals)')
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1])

    axs[5].axis('off')
    fig.suptitle(f'Signal (model pred) vs Noise (residual) Corr: {key} ({cond_name})')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    sig_bins = np.linspace(-1, 1, 11)
    sig_bin_centers = 0.5 * (sig_bins[:-1] + sig_bins[1:])

    fig2, axs2 = plt.subplots(1, 5, figsize=(10, 2.5), dpi=300, sharey=True)

    def _plot_binned(ax, sig_arr, noise_arr, color, label=None,
                     lw=1.5, alpha=1.0, show_fill=True, min_n=5):
        mean_noise = []
        sem_noise  = []
        for lo, hi in zip(sig_bins[:-1], sig_bins[1:]):
            in_bin = (sig_arr >= lo) & (sig_arr < hi)
            n_in = np.sum(in_bin)
            if n_in < min_n:
                mean_noise.append(np.nan); sem_noise.append(np.nan)
            else:
                mean_noise.append(np.nanmean(noise_arr[in_bin]))
                sem_noise.append(np.nanstd(noise_arr[in_bin]) / np.sqrt(n_in))
        mean_noise = np.array(mean_noise)
        sem_noise  = np.array(sem_noise)
        ax.plot(sig_bin_centers, mean_noise, '-o', markersize=3, color=color,
                label=label, lw=lw, alpha=alpha)
        if show_fill:
            ax.fill_between(sig_bin_centers, mean_noise - sem_noise, mean_noise + sem_noise,
                            color=color, alpha=0.2)
        ax.axhline(0, color='k', lw=0.5, ls='--')
        ax.axvline(0, color='k', lw=0.5, ls='--')

    v_sig   = pooled_sig[valid]
    v_noise = pooled_noise[valid]
    _plot_binned(axs2[0], v_sig, v_noise, 'k', 'All')
    axs2[0].set_title('All Pairs')
    axs2[0].set_xlabel('Signal Corr')
    axs2[0].set_ylabel('Mean Noise Corr')

    for i, (rid, rname) in enumerate(zip(region_ids, region_names)):
        mask = (pooled_regions[:, 0] == rid) & (pooled_regions[:, 1] == rid) & valid
        ax = axs2[i + 1]

        for fid in np.unique(pooled_fov[mask]):
            fov_mask = mask & (pooled_fov == fid)
            if np.sum(fov_mask) > 10:
                _plot_binned(ax, pooled_sig[fov_mask], pooled_noise[fov_mask],
                             color=area_colors[i], lw=1.0, alpha=0.6,
                             show_fill=False, min_n=3)

        if np.sum(mask) > 10:
            _plot_binned(ax, pooled_sig[mask], pooled_noise[mask],
                         color=area_colors[i], label=rname, lw=2.0, alpha=1.0,
                         show_fill=True)
        ax.set_title(f'{rname} Pairs')
        ax.set_xlabel('Signal Corr')
        ax.set_xlim([-1, 1])

    fig2.suptitle(f'Noise Corr vs Signal Corr/model pred (binned): {key} ({cond_name})')
    fig2.tight_layout()
    pdf.savefig(fig2)
    plt.close(fig2)

    return {f'signal_noise_corr_{key}_{cond}': {
        'pooled_sig':     pooled_sig,
        'pooled_noise':   pooled_noise,
        'pooled_regions': pooled_regions,
        'pooled_fov':     pooled_fov,
    }}


def get_aligned_behavior(pdata):
    
    if 'twopT' not in pdata:
        print("Warning: 'twopT' missing in pdata")
        return pd.DataFrame()

    twopT = pdata['twopT']
    n_frames = len(twopT)
    
    beh = {}
    
    direct_keys = {
        'theta': 'theta_interp',
        'phi': 'phi_interp',
        'pitch': 'pitch_twop_interp',
        'roll': 'roll_twop_interp',
        'gyro_x': 'gyro_x_twop_interp',
        'gyro_y': 'gyro_y_twop_interp',
        'gyro_z': 'gyro_z_twop_interp',
        'speed': 'speed'
    }
    
    for name, key in direct_keys.items():
        if key in pdata:
            arr = pdata[key]
            if len(arr) == n_frames:
                beh[name] = arr
            elif len(arr) > n_frames:
                beh[name] = arr[:n_frames]
            else:
                pass

    if 'eyeT' in pdata and 'eyeT_startInd' in pdata:
        eyeT = pdata['eyeT'][int(pdata['eyeT_startInd']):int(pdata['eyeT_endInd'])]
        eyeT = eyeT - eyeT[0]
        
        if 'dTheta' in pdata:
            dTh = pdata['dTheta']
            if len(dTh) == len(eyeT) - 1:
                t_src = eyeT[:-1] + np.diff(eyeT)/2
                beh['dTheta'] = interpT(dTh, t_src, twopT)
            elif len(dTh) == len(eyeT):
                beh['dTheta'] = interpT(dTh, eyeT, twopT)
                
        if 'dPhi' in pdata:
            dPh = pdata['dPhi']
            if len(dPh) == len(eyeT) - 1:
                t_src = eyeT[:-1] + np.diff(eyeT)/2
                beh['dPhi'] = interpT(dPh, t_src, twopT)
            elif len(dPh) == len(eyeT):
                beh['dPhi'] = interpT(dPh, eyeT, twopT)
                
        if 'longaxis' in pdata:
            la = pdata['longaxis'][int(pdata['eyeT_startInd']):int(pdata['eyeT_endInd'])]
            if len(la) == len(eyeT):
                beh['pupil'] = interpT(la, eyeT, twopT)

    if 'head_yaw_deg' in pdata and 'imuT_trim' in pdata:
        yaw = pdata['head_yaw_deg']
        imuT = pdata['imuT_trim']
        if len(yaw) == len(imuT):
             beh['yaw'] = interpT(yaw, imuT, twopT)
        elif len(yaw) == len(imuT) + 1:
             beh['yaw'] = interpT(yaw[:-1], imuT, twopT)

    # ── VOR-based eye-offset calibration ─────────────────────────────────────
    # Check for pre-computed calibration keys; compute on-the-fly if absent.
    # Stores ang_offset and corrected gaze into the behavior DataFrame so that
    # downstream analyses can use a sign-correct, calibrated gaze direction
    # without re-running preprocessing.
    fps = float(1.0 / np.nanmedian(np.diff(twopT))) if len(twopT) > 1 else 30.0
    ang_offset = get_ang_offset(pdata, fps=fps)

    if ang_offset is not None:
        beh['ang_offset'] = ang_offset  # scalar broadcast to constant column
        theta_col = beh.get('theta', None)
        head_col  = pdata.get('head_yaw_deg', None)
        if theta_col is not None and head_col is not None:
            theta_arr = np.asarray(theta_col)
            head_arr  = np.asarray(head_col)
            n = min(len(theta_arr), len(head_arr), n_frames)
            # Corrected gaze: head + (theta - ang_offset)  [sign-corrected formula]
            beh['gaze_deg'] = (head_arr[:n] + theta_arr[:n] - ang_offset) % 360

    return pd.DataFrame(beh)


def plot_all_variable_importance(pdf, data, animal_dirs, labeled_array, label_map):
    
    variables = ['theta', 'phi', 'dTheta', 'dPhi', 'pitch', 'yaw', 'roll', 'dPitch', 'dYaw', 'dRoll']
    region_names = ['V1', 'RL', 'AM', 'PM']
    region_ids = [5, 2, 3, 4]
    earth_cmap = make_earth_tones()
    
    cells_data = []
    
    for animal_dir in animal_dirs:
        if animal_dir not in data: continue
        for poskey in data[animal_dir]['transform']:
            if not poskey.startswith('pos'):
                continue
            transform = data[animal_dir]['transform'][poskey]
            messentials_pos = data[animal_dir]['messentials'][poskey]
            model_data = messentials_pos.get('model', {})

            n_pos = transform.shape[0]
            regions = _get_cell_regions(
                messentials_pos, n_pos, labeled_array,
                transform[:, 2], transform[:, 3])

            for c in range(len(regions)):
                cell_entry = {'region': int(regions[c])}
                for var in variables:
                    imp_key = get_glm_keys(var)
                    if imp_key in model_data and c < len(model_data[imp_key]):
                        cell_entry[var] = model_data[imp_key][c]
                    else:
                        cell_entry[var] = np.nan
                cells_data.append(cell_entry)
                
    df = pd.DataFrame(cells_data)
    
    for i, (rid, rname) in enumerate(zip(region_ids, region_names)):
        region_df = df[df['region'] == rid]
        
        fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
        
        for v_idx, var in enumerate(variables):
            vals = region_df[var].dropna()
            add_scatter_col(ax, v_idx, vals, color=earth_cmap(v_idx/len(variables)))
            
        ax.set_xticks(range(len(variables)))
        ax.set_xticklabels(variables, rotation=45, ha='right')
        ax.set_ylabel('Importance (Shuffle)')
        ax.set_title(f'Variable Importance in {rname}')
        ax.set_ylim([-0.05, 0.2])
        
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
    return {'all_variable_importance': df.to_dict(orient='list')}


def plot_all_model_performance(pdf, data, animal_dirs, labeled_array, label_map):
    
    models = ['full', 'position_only', 'velocity_only', 'head_only', 'eyes_only']
    region_names = ['V1', 'RL', 'AM', 'PM']
    region_ids = [5, 2, 3, 4]
    
    cells_data = []
    
    for animal_dir in animal_dirs:
        if animal_dir not in data: continue
        for poskey in data[animal_dir]['transform']:
            if not poskey.startswith('pos'):
                continue
            transform = data[animal_dir]['transform'][poskey]
            messentials_pos = data[animal_dir]['messentials'][poskey]
            model_data = messentials_pos.get('model', {})

            n_pos = transform.shape[0]
            regions = _get_cell_regions(
                messentials_pos, n_pos, labeled_array,
                transform[:, 2], transform[:, 3])

            for c in range(len(regions)):
                cell_entry = {'region': int(regions[c])}
                for model in models:

                    r2_key = f'{model}_r2'
                    if r2_key in model_data and c < len(model_data[r2_key]):
                        cell_entry[f'{model}_r2'] = model_data[r2_key][c]
                    else:
                        cell_entry[f'{model}_r2'] = np.nan

                    corr_key = f'{model}_corrs'
                    if corr_key in model_data and c < len(model_data[corr_key]):
                        cell_entry[f'{model}_corr'] = model_data[corr_key][c]
                    else:
                        cell_entry[f'{model}_corr'] = np.nan
                        
                cells_data.append(cell_entry)
                
    df = pd.DataFrame(cells_data)
    
    for i, (rid, rname) in enumerate(zip(region_ids, region_names)):
        region_df = df[df['region'] == rid]
        
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        for m_idx, model in enumerate(models):
            vals = region_df[f'{model}_r2'].dropna()[::10]
            add_scatter_col(ax, m_idx, vals, color='lightgrey')
            
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('R^2')
        ax.set_title(f'Model Performance (R^2) in {rname}')
        ax.set_ylim([-0.1, 0.4])
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    for i, (rid, rname) in enumerate(zip(region_ids, region_names)):
        region_df = df[df['region'] == rid]
        
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        for m_idx, model in enumerate(models):
            vals = region_df[f'{model}_corr'].dropna()[::10]
            add_scatter_col(ax, m_idx, vals, color='lightgrey')
            
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Correlation')
        ax.set_title(f'Model Performance (Correlation) in {rname}')
        ax.set_ylim([-0.1, 0.6])
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
    return {'all_model_performance': df.to_dict(orient='list')}

def plot_manifold_analysis(pdf, data, animal_dirs, labeled_array, label_map, root_dir, img_array=None):

    regions_to_plot = [5, 2, 3, 4] # V1, RL, AM, PM
    region_names = ['V1', 'RL', 'AM', 'PM']
    
    print(f"Starting manifold analysis for {len(animal_dirs)} animals.")

    earth_tone_colors = [
        '#2ECC71', '#82E0AA', # Green
        '#FF9800', '#FFCC80', # Orange
        '#03A9F4', '#81D4FA', # Blue
        '#9C27B0', '#E1BEE7', # Purple
        '#FFEB3B', '#FFF59D'  # Yellow
    ]
    var_order = ['theta', 'dTheta', 'phi', 'dPhi', 'pitch', 'dPitch', 'roll', 'dRoll', 'yaw', 'dYaw']
    var_color_map = {v: c for v, c in zip(var_order, earth_tone_colors)}
    
    earth_cmap = make_earth_tones()
    area_colors = make_area_colors()

    collected_results = []
    collected_data = {}
    
    total_files = sum(len(data[animal]['transform']) for animal in animal_dirs if animal in data and 'transform' in data[animal])
    file_counter = 0

    for animal in animal_dirs:
        if animal not in data: 
            # print(f"Skipping {animal}: not in data dictionary.")
            continue
        
        # print(f"Processing animal: {animal}")
        for poskey in data[animal]['transform']:
            file_counter += 1
            
            try:
                pos_num = int(poskey.replace('pos', ''))
                pos_str = f'pos{pos_num:02d}'
            except:
                # print(f"Skipping {poskey}: could not parse position number.")
                continue
            
            filename_pattern = f'*{animal}*preproc.h5'
            try:
                candidates = find(filename_pattern, root_dir, MR=False)
            except:
                # print(f"Skipping {animal} {poskey}: no preproc files found for animal pattern {filename_pattern}")
                continue
            
            valid_candidates = [c for c in candidates if pos_str in c]
            
            if not valid_candidates:
                # print(f"Skipping {animal} {poskey}: preproc files found but none matched {pos_str} in path.")
                continue
            
            ppath = choose_most_recent(valid_candidates)

            pdata = read_h5(ppath)
            
            pdata = check_and_trim_imu_disconnect(pdata)
            
            keys_to_keep = [
                'norm_dFF', 'norm_spikes', 'twopT', 'ltdk_state_vec',
                'theta_interp', 'phi_interp', 'head_yaw_deg', 
                'roll_twop_interp', 'pitch_twop_interp',
                'gyro_x_twop_interp', 'gyro_y_twop_interp', 'gyro_z_twop_interp',
                'dTheta', 'dPhi', 'eyeT', 'eyeT_startInd', 'eyeT_endInd', 'eyeT1',
                'speed'
            ]
            
            rec_data = {k: pdata[k] for k in keys_to_keep if k in pdata}
            
            rec_id = f"{animal}_{poskey}"

            if 'norm_spikes' not in pdata: 
                # print(f"Skipping {animal} {poskey}: 'norm_spikes' not in pdata.")
                continue
            
            transform = data[animal]['transform'][poskey]

            cell_indices = transform[:, 0].astype(int)

            cell_regions = _get_cell_regions(
                data[animal]['messentials'][poskey], len(transform),
                labeled_array, transform[:, 2], transform[:, 3])
            
            rec_data['cell_regions'] = cell_regions
            collected_data[rec_id] = rec_data
            
            spikes = pdata['norm_spikes'].T # (n_frames, n_cells)

            if len(cell_indices) > 0:
                if spikes.shape[1] == len(cell_indices):
                    pass
                elif np.max(cell_indices) < spikes.shape[1]:
                    spikes = spikes[:, cell_indices]
                else:
                    continue

            beh_df = get_aligned_behavior(pdata)
            
            valid_frames = beh_df.notna().all(axis=1)
            if valid_frames.sum() < 100: continue
            n_valid = valid_frames.sum()
            if n_valid < 100: 
                continue
            
            spikes_valid = spikes[valid_frames]
            beh_valid = beh_df[valid_frames]
            
            rename_map = {
                'gyro_x': 'dRoll',
                'gyro_y': 'dPitch',
                'gyro_z': 'dYaw'
            }
            beh_valid = beh_valid.rename(columns=rename_map)
            
            
            for i, (rid, rname) in enumerate(zip(regions_to_plot, region_names)):
                
                region_mask = (cell_regions == rid)
                n_cells_region = np.sum(region_mask)
                if n_cells_region < 10:
                    continue
                
                print(f"\rLoading preprocessed file {file_counter} of {total_files}; Analyzing region {rname} for {animal} {poskey} with {n_cells_region} cells.", end='', flush=True)
                print(f"Analyzing region {rname} for {animal} {poskey} with {n_cells_region} cells.")
                
                X = spikes_valid[:, region_mask]
                
                pca = PCA(n_components=10)
                X_pca = pca.fit_transform(X)
                explained_variance = pca.explained_variance_ratio_[:3]
                
                ev = pca.explained_variance_
                participation_ratio = (np.sum(ev)**2) / np.sum(ev**2)
                
                reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
                if X.shape[0] > 5000:
                    idx = np.linspace(0, X.shape[0]-1, 5000, dtype=int)
                    X_umap = reducer.fit_transform(X[idx])

                else:
                    X_umap = reducer.fit_transform(X)
                    idx = np.arange(X.shape[0])
                
                pc_corrs = []
                
                exclude_vars = ['speed', 'pupil', 'longaxis']
                beh_cols = [c for c in beh_valid.columns if c not in exclude_vars]

                for pc_idx in range(3):
                    corrs = {}
                    for col in beh_cols:
                        if col in var_order:
                            r = np.corrcoef(X_pca[:, pc_idx], beh_valid[col])[0, 1]
                            corrs[col] = np.abs(r)
                    pc_corrs.append(corrs)

                c_vals = np.mean(X, axis=1)
                c_label = 'Pop. Rate'
                
                if len(c_vals) > 5000:
                    c_vals = c_vals[idx]

                collected_results.append({
                    'animal': animal,
                    'poskey': poskey,
                    'region': rname,
                    'n_cells': n_cells_region,
                    'umap': X_umap,
                    'pca': X_pca[idx],
                    'color_vals': c_vals,
                    'color_label': c_label,
                    'explained_variance': explained_variance,
                    'participation_ratio': participation_ratio,
                    'raw_ev': pca.explained_variance_[:3],
                    'ev_curve': pca.explained_variance_ratio_,
                    'pc_corrs': pc_corrs
                })

    rows_per_page = 5
    n_results = len(collected_results)
    
    if n_results == 0:
        print("No manifold analysis results to plot.")
        return {}
        
    all_pca = np.vstack([res['pca'] for res in collected_results])
    pca_min = np.min(all_pca, axis=0)
    pca_max = np.max(all_pca, axis=0)
    pca_limits = [(pca_min[i], pca_max[i]) for i in range(3)]

    for start_idx in range(0, n_results, rows_per_page):
        end_idx = min(start_idx + rows_per_page, n_results)
        batch = collected_results[start_idx:end_idx]
        
        fig = plt.figure(figsize=(15, 3 * len(batch)), dpi=300)
        gs = fig.add_gridspec(len(batch), 5)
        
        for i, res in enumerate(batch):

            ax_umap = fig.add_subplot(gs[i, 0])
            ax_umap.scatter(res['umap'][:, 0], res['umap'][:, 1], s=1, c=res['color_vals'], cmap='plasma', alpha=0.5)
            ax_umap.set_title(f"{res['animal']} {res['poskey']} {res['region']}", fontsize=8)
            ax_umap.set_title(f"{res['animal']} {res['poskey']} {res['region']} (n={res['n_cells']})", fontsize=8)
            ax_umap.axis('off')
            
            ax_pca = fig.add_subplot(gs[i, 1], projection='3d')
            ax_pca.scatter(res['pca'][:, 0], res['pca'][:, 1], res['pca'][:, 2], s=1, c=res['color_vals'], cmap='plasma', alpha=0.5)
            ax_pca.set_title(f"PCA (EV: {res['explained_variance'][0]:.2f}, {res['explained_variance'][1]:.2f})", fontsize=8)
            ax_pca.set_xlabel('PC1', fontsize=7)
            ax_pca.set_ylabel('PC2', fontsize=7)
            ax_pca.set_zlabel('PC3', fontsize=7)
            ax_pca.tick_params(labelsize=6)
            ax_pca.set_xlim(pca_limits[0])
            ax_pca.set_ylim(pca_limits[1])
            ax_pca.set_zlim(pca_limits[2])
            ax_pca.view_init(elev=30, azim=45)
            
            for pc_idx in range(3):
                ax_corr = fig.add_subplot(gs[i, 2 + pc_idx])
                corrs_dict = res['pc_corrs'][pc_idx]
                
                keys = var_order
                vals = [corrs_dict.get(k, 0) for k in keys]
                colors_bar = [var_color_map.get(k, 'gray') for k in keys]
                
                y_pos = np.arange(len(keys))
                ax_corr.barh(y_pos, vals, align='center', color=colors_bar)
                ax_corr.set_yticks(y_pos)
                ax_corr.set_yticklabels(keys, fontsize=6)
                ax_corr.invert_yaxis()
                ax_corr.set_title(f'PC{pc_idx+1} Corrs')
                ax_corr.set_xlim([0, 1])
                ax_corr.grid(False)
        
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


    fig_corr, axs_corr = plt.subplots(2, 2, figsize=(6, 5), dpi=300)
    axs_corr = axs_corr.flatten()
    
    all_counts = {}
    global_max = 0

    for i, region in enumerate(region_names):
        counts_matrix = np.zeros((len(var_order), 3))
        availability = np.zeros(len(var_order))
        
        for res in collected_results:
            if res['region'] == region:
                if res['pc_corrs']:
                    present_vars = res['pc_corrs'][0].keys()
                    for v_idx, var in enumerate(var_order):
                        if var in present_vars:
                            availability[v_idx] += 1

                for pc_idx in range(3):
                    corrs = res['pc_corrs'][pc_idx]
                    if not corrs: continue
                    top_var = max(corrs, key=corrs.get)
                    if top_var in var_order:
                        row_idx = var_order.index(top_var)
                        counts_matrix[row_idx, pc_idx] += 1
        
        with np.errstate(divide='ignore', invalid='ignore'):
            counts_matrix = counts_matrix / availability[:, None]
            counts_matrix[~np.isfinite(counts_matrix)] = 0

        all_counts[region] = counts_matrix
        global_max = max(global_max, np.max(counts_matrix))

    for i, region in enumerate(region_names):
        counts_matrix = all_counts[region]

        if np.sum(counts_matrix) > 0:
            im = axs_corr[i].imshow(counts_matrix, cmap='plasma', aspect='auto', vmin=0)
            axs_corr[i].set_yticks(range(len(var_order)))
            axs_corr[i].set_yticklabels(var_order, fontsize=7)
            axs_corr[i].set_xticks([1, 2, 3])
            axs_corr[i].set_xticklabels(['PC1', 'PC2', 'PC3'], fontsize=8)
            axs_corr[i].set_title(f'{region}')
            axs_corr[i].set_xticks(np.arange(3))
            axs_corr[i].set_xticklabels(['PC1', 'PC2', 'PC3'])
            fig_corr.colorbar(im, ax=axs_corr[i], label='Count')
        else:
            axs_corr[i].text(0.5, 0.5, 'No Data', ha='center')
            axs_corr[i].set_title(f'{region}')
            
    fig_corr.suptitle('Top Correlated Variable per PC')
    fig_corr.tight_layout()
    pdf.savefig(fig_corr)
    plt.close(fig_corr)

    fig_dim, ax_dim = plt.subplots(1, 1, figsize=(4,3), dpi=300)
    
    pr_means = []
    pr_stds = []

    fig_scree, ax_scree = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
    for i, region in enumerate(region_names): # region_names is unique ['V1', 'RL', 'AM', 'PM']
        curves = [np.cumsum(res['ev_curve']) for res in collected_results if res['region'] == region]
        if curves:
            mean_curve = np.mean(curves, axis=0)
            x_vals = np.arange(1, len(mean_curve) + 1)
            ax_scree.plot(x_vals, mean_curve, label=region, marker='o', markersize=3, color=area_colors[i])
    ax_scree.set_xlabel('PCs')
    ax_scree.set_ylabel('Cumulative Explained Variance')
    ax_scree.set_title('compactness')
    ax_scree.legend()
    ax_scree.grid(True, alpha=0.3)
    pdf.savefig(fig_scree)
    plt.close(fig_scree)
    
    return {'manifold_analysis': collected_results, 'raw_data_for_modeling': collected_data}


def plot_sorted_tuning_curves(pdf, data, animal_dirs, cond='l'):
    
    variables = ['theta', 'phi', 'dTheta', 'dPhi', 'pitch', 'yaw', 'roll', 'dPitch', 'dYaw', 'dRoll']
    cond_idx = 1 if cond == 'l' else 0
    cond_name = 'Light' if cond == 'l' else 'Dark'
    
    all_sorted_curves = {}
    
    reverse_map = {'dRoll': 'gyro_x', 'dPitch': 'gyro_y', 'dYaw': 'gyro_z'}

    for key in variables:
        cells = []
        
        use_key = key
        if key in reverse_map:
            use_key = reverse_map[key]

        for animal in animal_dirs:
            if animal not in data: continue
            if 'transform' not in data[animal]: continue
            
            for poskey in data[animal]['transform']:
                if not poskey.startswith('pos'):
                    continue
                if (animal=='DMM056') and (cond=='d') and ((poskey=='pos15') or (poskey=='pos03')):
                    continue

                messentials = data[animal]['messentials'][poskey]
                rdata = messentials.get('rdata', {})
                
                curr_use_key = use_key
                if key in reverse_map:
                    mapped = reverse_map[key]
                    if f'{mapped}_1dtuning' in rdata:
                        curr_use_key = mapped
                    elif f'{key}_1dtuning' in rdata:
                        curr_use_key = key
                    else:
                        curr_use_key = mapped

                tuning_key = f'{curr_use_key}_1dtuning'
                bins_key   = f'{curr_use_key}_1dbins'
                cvmi_key   = f'{curr_use_key}_{cond}_rel'

                err_key = f'{curr_use_key}_1derr'
                if err_key not in rdata:
                    err_key = f'{curr_use_key}_1dstderr'

                if tuning_key not in rdata or bins_key not in rdata or cvmi_key not in rdata:
                    continue

                tuning   = rdata[tuning_key]
                bins     = rdata[bins_key]
                cv_mi    = rdata[cvmi_key]

                errs = None
                if err_key in rdata:
                    errs = rdata[err_key]

                n_cells = tuning.shape[0]

                for c in range(n_cells):
                    if np.isnan(cv_mi[c]): continue
                    
                    if tuning.ndim == 3:
                        if tuning.shape[2] > cond_idx:
                            t_curve = tuning[c, :, cond_idx]
                            t_err = errs[c, :, cond_idx] if errs is not None else np.zeros_like(t_curve)
                        else:
                            continue
                    else:
                        t_curve = tuning[c, :]
                        t_err = errs[c, :] if errs is not None else np.zeros_like(t_curve)

                    cells.append({
                        'cv_mi': cv_mi[c],
                        'tuning': t_curve,
                        'err': t_err,
                        'bins': bins,
                        'id': f'{animal} {poskey} {c}',
                    })
        
        if not cells:
            continue

        cells.sort(key=lambda x: x['cv_mi'], reverse=True)
        top_cells = cells[:64]

        if top_cells:
            all_sorted_curves[key] = {
                'mods': np.array([c['cv_mi'] for c in top_cells]),
                'tuning': np.vstack([c['tuning'] for c in top_cells]),
                'errs': np.vstack([c['err'] for c in top_cells]),
                'bins': top_cells[0]['bins'],
                'rel_vals': np.array([c['cv_mi'] for c in top_cells]),
            }
        
        fig, axs = plt.subplots(8, 8, figsize=(16, 16), dpi=300)
        axs = axs.flatten()
        
        for i, ax in enumerate(axs):
            if i < len(top_cells):
                cell = top_cells[i]
                bin_edges = cell['bins']
                if len(bin_edges) == len(cell['tuning']) + 1:
                    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                else:
                    bin_centers = bin_edges
                
                ax.plot(bin_centers, cell['tuning'], 'k-')
                ax.fill_between(bin_centers, cell['tuning'] - cell['err'], cell['tuning'] + cell['err'], color='k', alpha=0.3)
                
                title_str = f"{cell['id']}\ncvMI={cell['cv_mi']:.2f}"
                ax.set_title(title_str, fontsize=6)
                ax.tick_params(labelsize=6)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            else:
                ax.axis('off')
                
        fig.suptitle(f'Top 64 Modulated Cells for {key} ({cond_name})', fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)
        
    return {f'sorted_tuning_curves_{cond}': all_sorted_curves}


def plot_modulation_summary(pdf, data, animal_dirs, labeled_array, label_map, cond='l', savedir=None):

    variables = ['theta', 'phi', 'dTheta', 'dPhi', 'pitch', 'yaw', 'roll', 'dPitch', 'dYaw', 'dRoll']
    regions = [5, 2, 3, 4, 10] # V1, RL, AM, PM, A
    region_names = ['V1', 'RL', 'AM', 'PM', 'A']
    region_id_to_name = {5: 'V1', 2: 'RL', 3: 'AM', 4: 'PM', 10: 'A'}
    area_colors = get_equally_spaced_colormap_values('Dark2', 5)

    # Data structure: results[metric][var][region_name] = [val_mouse1, val_mouse2, ...]
    results = {
        'tuning': {v: {rn: [] for rn in region_names} for v in variables},
        'importance': {v: {rn: [] for rn in region_names} for v in variables},
        'mod_any': {rn: [] for rn in region_names},
    }

    cv_thresh = 0.1  # cells with CV-MI > cv_thresh are counted as reliable/modulated

    total_cells_tracked = 0
    modulated_cells_tracked = 0

    for animal in animal_dirs:
        if animal not in data: continue

        # Per animal aggregation
        animal_cells = {r: {'total': 0, 'mod_any': 0, 'tuning': {v: 0 for v in variables}, 'importance': {v: 0 for v in variables}} for r in regions}
        
        has_data = False
        
        if 'transform' not in data[animal]: continue

        for poskey in data[animal]['transform']:
            if not poskey.startswith('pos'):
                continue
            if (animal=='DMM056') and (cond=='d') and ((poskey=='pos15') or (poskey=='pos03')):
                continue
            
            transform = data[animal]['transform'][poskey]
            messentials = data[animal]['messentials'][poskey]
            rdata = messentials.get('rdata', {})
            model_data = messentials.get('model', {})

            n_cells = transform.shape[0]
            cell_regions = _get_cell_regions(
                messentials, n_cells, labeled_array,
                transform[:, 2], transform[:, 3])

            if n_cells == 0: continue
            has_data = True
            
            # Store variable data for this poskey
            pos_var_data = {}
            for var in variables:
                isrel, mod, _, rel_val = get_cell_data(rdata, var, cond)
                pos_var_data[var] = {'isrel': isrel, 'rel': rel_val, 'mod': mod}

            for c_idx, region in enumerate(cell_regions):
                if region not in regions: continue

                animal_cells[region]['total'] += 1
                
                total_cells_tracked += 1
                is_modulated_any = False

                for var in variables:
                    vdata = pos_var_data[var]
                    
                    if vdata['mod'] is not None and c_idx < len(vdata['mod']):
                        if not np.isnan(vdata['mod'][c_idx]) and vdata['mod'][c_idx] > 0.33:
                            is_modulated_any = True

                    # Tuning: cell counted if CV-MI exceeds threshold
                    if vdata['isrel'] is not None and c_idx < len(vdata['isrel']):
                        if vdata['isrel'][c_idx] == 1:
                            animal_cells[region]['tuning'][var] += 1

                    # Importance: same CV-MI threshold
                    if vdata['isrel'] is not None and c_idx < len(vdata['isrel']):
                        if vdata['isrel'][c_idx] == 1:
                            animal_cells[region]['importance'][var] += 1
                
                if is_modulated_any:
                    modulated_cells_tracked += 1
                    animal_cells[region]['mod_any'] += 1

        if not has_data: continue
        
        # Calculate percentages for this animal
        for r in regions:
            total = animal_cells[r]['total']
            
            for var in variables:
                if total < 10:
                    pct_tune = np.nan
                    pct_imp = np.nan
                else:
                    pct_tune = (animal_cells[r]['tuning'][var] / total) * 100
                    pct_imp = (animal_cells[r]['importance'][var] / total) * 100
                    
                    if pct_tune == 0:
                        pct_tune = np.nan
                    if pct_imp == 0:
                        pct_imp = np.nan
                
                results['tuning'][var][region_id_to_name[r]].append(pct_tune)
                results['importance'][var][region_id_to_name[r]].append(pct_imp)

            total = animal_cells[r]['total']
            if total < 10:
                pct_any = np.nan
            else:
                pct_any = (animal_cells[r]['mod_any'] / total) * 100
                if pct_any == 0:
                    pct_any = np.nan
            results['mod_any'][region_id_to_name[r]].append(pct_any)

    # Plotting
    cond_name = 'Light' if cond == 'l' else 'Dark'

    if total_cells_tracked > 0:
        pct_mod = (modulated_cells_tracked / total_cells_tracked) * 100
        print(f"Percentage of cells with MI > 0.33 to at least one variable ({cond_name}): {pct_mod:.2f}%")

    for metric in ['tuning', 'importance']:
        fig, axs = plt.subplots(2, 5, figsize=(6.5, 2.5), dpi=300)
        axs = axs.flatten()

        for i, var in enumerate(variables):
            ax = axs[i]

            ax.axhline(50, color='lightgrey', linestyle='--', alpha=0.5)

            for j, rn in enumerate(region_names):
                vals = results[metric][var][rn]
                if vals:
                    add_scatter_col(ax, j, vals, color=area_colors[j])
            
            ax.set_xticks(range(len(region_names)))
            ax.set_xticklabels(region_names)
            ax.set_title(var)
            if metric == 'tuning':
                ax.set_ylim([0, 50])
            elif metric == 'importance':
                ax.set_ylim([0, 100.])
            if i % 5 == 0:
                ax.set_ylabel('% Modulated Cells')
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
        fig.suptitle(f'Percentage of Modulated Cells by {metric.capitalize()} ({cond_name})')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
    # Per-region percent modulated any variable figure
    fig_any, ax_any = plt.subplots(figsize=(3.5, 3), dpi=300)
    for j, rn in enumerate(region_names):
        vals = [v for v in results['mod_any'][rn] if not np.isnan(v)]
        if vals:
            add_scatter_col(ax_any, j, vals, color=area_colors[j])
    ax_any.set_xticks(range(len(region_names)))
    ax_any.set_xticklabels(region_names)
    ax_any.set_ylim([0, 100])
    ax_any.set_ylabel('% Modulated (Any Var)')
    ax_any.set_title(f'Modulated by Any Variable ({cond_name})')
    ax_any.spines['top'].set_visible(False)
    ax_any.spines['right'].set_visible(False)
    fig_any.tight_layout()
    pdf.savefig(fig_any)
    if savedir is not None:
        fig_any.savefig(os.path.join(savedir, f'percent_modulated_any_var_{cond}.svg'),
                        dpi=300, bbox_inches='tight')
    plt.close(fig_any)

    return {f'modulation_summary_{cond}': results}


def plot_modulation_histograms(pdf, data, animal_dirs, labeled_array, label_map, cond='l'):

    variables = ['theta', 'phi', 'dTheta', 'dPhi', 'pitch', 'yaw', 'roll', 'dPitch', 'dYaw', 'dRoll']
    regions = [5, 2, 3, 4] # V1, RL, AM, PM
    region_names = ['V1', 'RL', 'AM', 'PM']
    region_id_to_name = {5: 'V1', 2: 'RL', 3: 'AM', 4: 'PM'}
    area_colors = make_area_colors()

    cond_name = 'Light' if cond == 'l' else 'Dark'

    results = {
        'mod': {v: {rn: [] for rn in region_names} for v in variables},
        'imp': {v: {rn: [] for rn in region_names} for v in variables}
    }
    
    for animal in animal_dirs:
        if animal not in data: continue
        if 'transform' not in data[animal]: continue

        for poskey in data[animal]['transform']:
            if not poskey.startswith('pos'):
                continue
            if (animal=='DMM056') and (cond=='d') and ((poskey=='pos15') or (poskey=='pos03')):
                continue
            
            transform = data[animal]['transform'][poskey]
            messentials = data[animal]['messentials'][poskey]
            rdata = messentials.get('rdata', {})
            model_data = messentials.get('model', {})

            n_cells = transform.shape[0]
            cell_regions = _get_cell_regions(
                messentials, n_cells, labeled_array,
                transform[:, 2], transform[:, 3])

            if n_cells == 0: continue

            for var in variables:
                _, cv_mi, _, _ = get_cell_data(rdata, var, cond)

                for c_idx, region in enumerate(cell_regions):
                    if region not in regions: continue
                    rn = region_id_to_name[region]

                    if cv_mi is not None and c_idx < len(cv_mi):
                        val = cv_mi[c_idx]
                        if not np.isnan(val):
                            results['mod'][var][rn].append(val)

    for metric in ['mod', 'imp']:
        fig, axs = plt.subplots(2, 5, figsize=(7, 3.5), dpi=300)
        axs = axs.flatten()

        metric_label = 'Modulation Index' if metric == 'mod' else 'Reliability Score (1 - null count/100)'

        all_values = []
        for v in variables:
            for rn in region_names:
                all_values.extend(results[metric][v][rn])

        if not all_values:
            plt.close(fig)
            continue

        xlim = (0, 1.0)
        bins = np.linspace(0, 1.0, 21)
            
        for i, var in enumerate(variables):
            ax = axs[i]

            for j, rn in enumerate(region_names):
                vals = results[metric][var][rn]
                if vals:
                    ax.hist(vals, bins=bins, density=True, histtype='step',
                            color=area_colors[j], label=rn, linewidth=1.5)
            
            # imp: threshold = (100-1)/100 = 0.99 marks the relthresh=1 boundary
            ref_lines = [0, 0.33, 0.5] if metric == 'mod' else [0.99]
            for line_val in ref_lines:
                ax.axvline(line_val, color='k', linestyle='--', alpha=0.5, linewidth=0.8)
            
            ax.set_title(var)
            ax.set_xlim(xlim)
            
            if i % 5 == 0:
                ax.set_ylabel('Density')
            if i >= 5:
                ax.set_xlabel(metric_label)
                
            if i == 0:
                ax.legend(fontsize=8, frameon=False)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
        fig.suptitle(f'{metric_label} Distributions by Area ({cond_name})')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
    return {f'modulation_histograms_{cond}': results}


def plot_lightdark_modulation_histograms(pdf, data, animal_dirs, labeled_array, label_map):

    variables = ['theta', 'phi', 'dTheta', 'dPhi', 'pitch', 'yaw', 'roll',
                 'dPitch', 'dYaw', 'dRoll']
    hist_bins = np.linspace(0, 1.0, 26)

    saved_data = {}

    for var in variables:

        dark_rel_mod_d, dark_rel_mod_l = [], []
        light_rel_mod_d, light_rel_mod_l = [], []

        for animal in animal_dirs:
            if animal not in data: continue
            if 'transform' not in data[animal]: continue

            for poskey in data[animal]['transform']:
                if not poskey.startswith('pos'):
                    continue
                if (animal == 'DMM056') and ((poskey == 'pos15') or (poskey == 'pos03')):
                    continue

                rdata = data[animal]['messentials'][poskey].get('rdata', {})
                isrel_d, mod_d, _, _ = get_cell_data(rdata, var, 'd')
                isrel_l, mod_l, _, _ = get_cell_data(rdata, var, 'l')

                if isrel_d is None or mod_d is None: continue
                if isrel_l is None or mod_l is None: continue

                n_cells = min(len(isrel_d), len(isrel_l), len(mod_d), len(mod_l))

                for c in range(n_cells):
                    md = mod_d[c] if not np.isnan(mod_d[c]) else np.nan
                    ml = mod_l[c] if not np.isnan(mod_l[c]) else np.nan

                    if isrel_d[c] == 1:
                        dark_rel_mod_d.append(md)
                        dark_rel_mod_l.append(ml)

                    if isrel_l[c] == 1:
                        light_rel_mod_d.append(md)
                        light_rel_mod_l.append(ml)

        def _clean(lst):
            a = np.array(lst, dtype=float)
            return a[~np.isnan(a)]

        fig, axs = plt.subplots(2, 2, figsize=(7, 5), dpi=300)

        # Row 0: cells reliable in dark
        axs[0, 0].hist(_clean(dark_rel_mod_d), bins=hist_bins, density=True,
                       color='navy', alpha=0.7)
        axs[0, 0].set_title(f'Dark-reliable -> CV-MI in Dark  (n={len(_clean(dark_rel_mod_d))})')
        axs[0, 0].set_ylabel('Density')

        axs[0, 1].hist(_clean(dark_rel_mod_l), bins=hist_bins, density=True,
                       color='goldenrod', alpha=0.7)
        axs[0, 1].set_title(f'Dark-reliable -> CV-MI in Light  (n={len(_clean(dark_rel_mod_l))})')

        # Row 1: cells reliable in light
        axs[1, 0].hist(_clean(light_rel_mod_d), bins=hist_bins, density=True,
                       color='navy', alpha=0.7)
        axs[1, 0].set_title(f'Light-reliable -> CV-MI in Dark  (n={len(_clean(light_rel_mod_d))})')
        axs[1, 0].set_ylabel('Density')

        axs[1, 1].hist(_clean(light_rel_mod_l), bins=hist_bins, density=True,
                       color='goldenrod', alpha=0.7)
        axs[1, 1].set_title(f'Light-reliable -> CV-MI in Light  (n={len(_clean(light_rel_mod_l))})')

        for ax in axs.flatten():
            ax.axvline(0.1, color='k', ls='--', alpha=0.5, lw=0.8)
            ax.set_xlim([0, 1.0])
            ax.set_xlabel('CV-MI')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        fig.suptitle(f'{var}: Cross-condition CV-MI for Condition-Reliable Cells')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        saved_data[var] = {
            'dark_rel_mod_d': _clean(dark_rel_mod_d),
            'dark_rel_mod_l': _clean(dark_rel_mod_l),
            'light_rel_mod_d': _clean(light_rel_mod_d),
            'light_rel_mod_l': _clean(light_rel_mod_l),
        }

    return {'lightdark_modulation': saved_data}


def plot_position_occupancy(pdf, data, animal_dirs, root_dir):
    """Downsampled scatter of head pitch vs roll and eye theta vs phi per recording.

    One panel per recording arranged in a multi-panel grid.  Data are
    downsampled by a factor of 100 to reduce file size while preserving the
    occupancy distribution shape.  Two multi-panel figures are saved:
      1. Pitch (y) vs Roll (x) — head posture occupancy.
      2. Phi (y) vs Theta (x) — eye position occupancy.
    """
    DS = 100  # downsample factor

    recording_data = []

    for animal in animal_dirs:
        if animal not in data: continue
        if 'transform' not in data[animal]: continue

        for poskey in data[animal]['transform']:
            try:
                pos_num = int(poskey.replace('pos', ''))
                pos_str = f'pos{pos_num:02d}'
            except Exception:
                continue

            try:
                candidates = find(f'*{animal}*preproc.h5', root_dir, MR=False)
            except Exception:
                continue

            valid_candidates = [c for c in candidates if pos_str in c]
            if not valid_candidates:
                continue

            ppath = choose_most_recent(valid_candidates)
            try:
                pdata = read_h5(ppath)
            except Exception:
                continue

            pitch = pdata.get('pitch_twop_interp', None)
            roll  = pdata.get('roll_twop_interp', None)
            theta = pdata.get('theta_interp', None)
            phi   = pdata.get('phi_interp', None)

            if any(v is None for v in [pitch, roll, theta, phi]):
                continue

            # theta/phi are typically in radians in the preproc file
            theta_deg = np.rad2deg(np.array(theta, dtype=float))
            phi_deg   = np.rad2deg(np.array(phi,   dtype=float))

            recording_data.append({
                'label': f'{animal} {poskey}',
                'pitch': np.array(pitch, dtype=float)[::DS],
                'roll':  np.array(roll,  dtype=float)[::DS],
                'theta': theta_deg[::DS],
                'phi':   phi_deg[::DS],
            })

    if not recording_data:
        print("plot_position_occupancy: no recordings found.")
        return {}

    n_recs = len(recording_data)
    ncols  = min(4, n_recs)
    nrows  = int(np.ceil(n_recs / ncols))

    for var_pair, xlabel, ylabel, fig_title in [
        ('pitch_roll', 'Roll (deg)', 'Pitch (deg)', 'Head Pitch vs Roll Occupancy (1/100 sampled)'),
        ('theta_phi',  'Theta (deg)', 'Phi (deg)',  'Eye Theta vs Phi Occupancy (1/100 sampled)'),
    ]:
        fig, axs = plt.subplots(nrows, ncols,
                                figsize=(3 * ncols, 2.5 * nrows), dpi=150)
        # Normalise axes shape
        if nrows == 1 and ncols == 1:
            axs = np.array([[axs]])
        elif nrows == 1:
            axs = axs[np.newaxis, :]
        elif ncols == 1:
            axs = axs[:, np.newaxis]

        for idx, rec in enumerate(recording_data):
            row, col = idx // ncols, idx % ncols
            ax = axs[row, col]

            if var_pair == 'pitch_roll':
                x, y = rec['roll'], rec['pitch']
            else:
                x, y = rec['theta'], rec['phi']

            # Remove NaNs
            valid = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[valid], y[valid], s=1, c='k', alpha=0.25, rasterized=True)
            ax.set_title(rec['label'], fontsize=7)
            ax.set_xlabel(xlabel, fontsize=7)
            ax.set_ylabel(ylabel, fontsize=7)
            ax.tick_params(labelsize=6)

        # Hide unused panels
        for idx in range(n_recs, nrows * ncols):
            axs[idx // ncols, idx % ncols].axis('off')

        fig.suptitle(fig_title)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    return {
        'position_occupancy_recordings': [r['label'] for r in recording_data],
        'position_occupancy_data': {
            str(i): {
                'pitch': rec['pitch'],
                'roll': rec['roll'],
                'theta': rec['theta'],
                'phi': rec['phi'],
            }
            for i, rec in enumerate(recording_data)
        },
    }


def plot_model_performance(pdf, data, animal_dirs, labeled_array, label_map):
    
    models = ['full', 'velocity_only', 'position_only', 'eyes_only', 'head_only']
    metrics = ['r2', 'corrs']
    
    all_results = {}
    
    for model in models:
        for metric in metrics:
            
            cells = []
            for animal_dir in animal_dirs:
                if animal_dir not in data: continue
                if 'transform' not in data[animal_dir]: continue
                
                for poskey in data[animal_dir]['transform']:
                    if not poskey.startswith('pos'):
                        continue
                    transform = data[animal_dir]['transform'][poskey]
                    messentials_pos = data[animal_dir]['messentials'][poskey]
                    model_data = messentials_pos.get('model', {})

                    m_key = f'{model}_{metric}'
                    if m_key not in model_data:
                        continue

                    vals = model_data[m_key]
                    n_vals = len(vals)
                    pos_regions = _get_cell_regions(
                        messentials_pos, n_vals, labeled_array,
                        transform[:, 2], transform[:, 3])

                    for c in range(n_vals):
                        cells.append({
                            'x': transform[c, 2],
                            'y': transform[c, 3],
                            'region': int(pos_regions[c]),
                            'val': vals[c]
                        })
            
            if not cells:
                print(f"No data found for {model} {metric}")
                continue

            df = pd.DataFrame(cells)
            all_results[f'{model}_{metric}'] = df.to_dict(orient='list')
            
            cmap = cm.plasma
            if metric == 'r2':
                norm = colors.Normalize(vmin=-0.2, vmax=0.3)
                label_str = f'{model} R^2'
            else:
                norm = colors.Normalize(vmin=-0.2, vmax=0.5)
                label_str = f'{model} Correlation'

            fig, axs = plt.subplots(2, 2, dpi=300, figsize=(8,8))
            axs = axs.flatten()

            for i in range(4):
                region_id = [2,3,4,5][i]
                region_name = label_map.get(region_id, f'Region {region_id}')

                region_mask = (labeled_array == region_id)
                axs[i].imshow(region_mask, cmap='gray_r', alpha=0.4)
                axs[i].imshow(labeled_array == 0, cmap='binary', alpha=0.1)
                axs[i].contour(region_mask, levels=[0.5], colors='k', linestyles='--', linewidths=0.5)
                
                region_cells = df[df['region'] == region_id]
                if len(region_cells) > 0:
                    axs[i].scatter(region_cells['x'], region_cells['y'], s=2, c=region_cells['val'], cmap=cmap, norm=norm)
                
                axs[i].set_title(f'{region_name}')
                axs[i].axis('off')
                axs[i].set_ylim([labeled_array.shape[0], 0])
                axs[i].set_xlim([0, labeled_array.shape[1]])
            
            fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, label=label_str, fraction=0.05, shrink=0.6)

            fig.suptitle(f'{label_str} Map by Region')
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            fig, axs = plt.subplots(2, 2, dpi=300, figsize=(8,6))
            axs = axs.flatten()

            for i in range(4):
                region_id = [2,3,4,5][i]
                region_name = label_map.get(region_id, f'Region {region_id}')
                
                region_mask = (labeled_array == region_id)
                axs[i].imshow(labeled_array == 0, cmap='binary', alpha=0.1)
                axs[i].contour(region_mask, levels=[0.5], colors='k', linestyles='--', linewidths=0.5)
                
                region_cells = df[df['region'] == region_id]
                if len(region_cells) > 0:
                    smoothed = create_smoothed_map(region_cells['x'].values, region_cells['y'].values, region_cells['val'].values, shape=labeled_array.shape)
                    smoothed[~region_mask] = np.nan
                    axs[i].imshow(smoothed, cmap=cmap, norm=norm)
                
                axs[i].set_title(f'{region_name}')
                axs[i].axis('off')
                axs[i].set_ylim([labeled_array.shape[0], 0])
                axs[i].set_xlim([0, labeled_array.shape[1]])

            fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, label=label_str, fraction=0.05, shrink=0.6)
            fig.suptitle(f'{label_str} Smoothed Map by Region')
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            
    return {'model_performance_maps': all_results}


def run_gaze_analysis(data, animal_dirs, root_dir):
    
    print("Starting gaze state change analysis.")

    for animal in animal_dirs:
        if animal not in data: 
            continue
        
        for poskey in data[animal]['transform']:
            
            try:
                pos_num = int(poskey.replace('pos', ''))
                pos_str = f'pos{pos_num:02d}'
            except:
                continue
            
            filename_pattern = f'*{animal}*preproc.h5'
            try:
                candidates = find(filename_pattern, root_dir, MR=False)
            except:
                continue
            
            valid_candidates = [c for c in candidates if pos_str in c]
            
            if not valid_candidates:
                continue
            
            ppath = choose_most_recent(valid_candidates)

            try:
                pdata = read_h5(ppath)
                
                savepath = os.path.join(os.path.split(ppath)[0], f'{animal}_{poskey}_gaze_state_changes.png')
                
                print(f"Analyzing {animal} {poskey}")
                analyze_gaze_state_changes(pdata, savepath=savepath)
            except Exception as e:
                print(f"Error analyzing {animal} {poskey}: {e}")


def make_behavior_corr_matrix(pdf, data, root_dir):

    print("Generating behavior correlation matrix...")

    variables = ['theta', 'phi', 'dTheta', 'dPhi', 'pitch', 'roll', 'dPitch', 'dYaw', 'dRoll']
    
    all_corrs = []
    
    all_data = {v: [] for v in variables}
    
    animal_dirs = list(data.keys())
    
    for animal in animal_dirs:
        if 'transform' not in data[animal]: continue
        
        for poskey in data[animal]['transform']:
            
            try:
                pos_num = int(poskey.replace('pos', ''))
                pos_str = f'pos{pos_num:02d}'
            except:
                continue
                
            filename_pattern = f'*{animal}*preproc.h5'
            try:
                candidates = find(filename_pattern, root_dir, MR=False)
            except:
                continue
                
            valid_candidates = [c for c in candidates if pos_str in c]
            if not valid_candidates: continue
            
            ppath = choose_most_recent(valid_candidates)
            
            try:
                pdata = read_h5(ppath)
            except:
                continue
                
            pdata = check_and_trim_imu_disconnect(pdata)
            
            beh_df = get_aligned_behavior(pdata)
            
            rename_map = {'gyro_x': 'dRoll', 'gyro_y': 'dPitch', 'gyro_z': 'dYaw'}
            beh_df = beh_df.rename(columns=rename_map)
            
            current_vars = [v for v in variables if v in beh_df.columns]
            if len(current_vars) < 2: continue
            
            df_subset = beh_df[current_vars].dropna()
            if len(df_subset) < 100: continue
            
            if len(df_subset) > 1000:
                sub = df_subset.sample(n=1000, random_state=42)
            else:
                sub = df_subset
                
            for v in current_vars:
                all_data[v].extend(sub[v].values)

            corr_mat = np.full((len(variables), len(variables)), np.nan)
            c = df_subset.corr().loc[current_vars, current_vars]
            
            for i, v1 in enumerate(variables):
                for j, v2 in enumerate(variables):
                    if v1 in c.index and v2 in c.columns:
                        corr_mat[i, j] = c.at[v1, v2]
            
            all_corrs.append(corr_mat)

    if not all_corrs:
        print("No behavior data found for correlation matrix.")
        return {}

    all_corrs = np.array(all_corrs)
    
    fig_hist, axs_hist = plt.subplots(2, 5, figsize=(8,3), dpi=300)
    axs_hist = axs_hist.flatten()
    for i, var in enumerate(variables):
        vals = np.array(all_data[var])
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            axs_hist[i].hist(vals, bins=30, color='tab:blue', alpha=0.7, density=True)
            axs_hist[i].set_title(var)
    fig_hist.suptitle('Behavior Variable Distributions (Aggregated)')
    fig_hist.tight_layout()
    pdf.savefig(fig_hist)
    plt.close(fig_hist)
    
    mean_corr = np.nanmean(all_corrs, axis=0)
    std_corr = np.nanstd(all_corrs, axis=0)
    mask = np.triu(np.ones_like(mean_corr, dtype=bool), k=0)
    masked_mean = np.ma.array(mean_corr, mask=mask)
    
    fig_mat, ax_mat = plt.subplots(figsize=(5,5), dpi=300)
    im = ax_mat.imshow(masked_mean, cmap=plt.cm.coolwarm, vmin=-1, vmax=1)
    ax_mat.set_xticks(np.arange(len(variables))); ax_mat.set_yticks(np.arange(len(variables)))
    ax_mat.set_xticklabels(variables, rotation=45, ha='right'); ax_mat.set_yticklabels(variables)
    
    for i in range(len(variables)):
        for j in range(len(variables)):
            if i > j:
                val = mean_corr[i, j]
                err = std_corr[i, j]
                if not np.isnan(val):
                    text_color = "white" if abs(val) > 0.5 else "black"
                    ax_mat.text(j, i, f"{val:.2f}\n±{err:.2f}", ha="center", va="center", color=text_color, fontsize=7)
    
    ax_mat.set_title("Behavior Correlation Matrix (Mean ± Std across recordings)")
    ax_mat.spines['top'].set_visible(False); ax_mat.spines['right'].set_visible(False)
    fig_mat.tight_layout()
    pdf.savefig(fig_mat)
    plt.close(fig_mat)
    
    return {
        'behavior_correlations': all_corrs,
        'behavior_distributions': {v: np.array(vals) for v, vals in all_data.items() if vals},
    }


def aggregate_boundary_data(data, animal_dirs, labeled_array, label_map):

    from collections import defaultdict

    id_to_name = {v: k for k, v in label_map.items() if k > 1}

    cells    = []          # list of per-cell scalar dicts
    params   = None        # axis params (grabbed from first valid position)
    area_ebc = defaultdict(list)   # area_name -> [smoothed_map, ...]
    area_rbc = defaultdict(list)

    for animal_dir in animal_dirs:
        if animal_dir not in data:
            continue
        adat = data[animal_dir]
        if 'transform' not in adat or 'messentials' not in adat:
            continue

        for poskey in adat['transform']:
            if not poskey.startswith('pos'):
                continue

            transform       = adat['transform'][poskey]
            messentials_pos = adat['messentials'][poskey]
            bdata           = messentials_pos.get('boundary', {})

            if not bdata or 'is_EBC' not in bdata:
                continue

            is_EBC   = np.asarray(bdata['is_EBC'],  dtype=int)
            is_RBC   = np.asarray(bdata['is_RBC'],  dtype=int)
            is_fr_e  = np.asarray(bdata.get('is_fully_reliable_EBC', is_EBC), dtype=int)
            is_fr_r  = np.asarray(bdata.get('is_fully_reliable_RBC', is_RBC), dtype=int)
            ebc_mrl  = np.asarray(bdata.get('ebc_mrl',  np.full(len(is_EBC), np.nan)))
            rbc_mrl  = np.asarray(bdata.get('rbc_mrl',  np.full(len(is_RBC), np.nan)))
            ebc_corr = np.asarray(bdata.get('ebc_corr_coeff', np.full(len(is_EBC), np.nan)))
            rbc_corr = np.asarray(bdata.get('rbc_corr_coeff', np.full(len(is_RBC), np.nan)))

            n_cells = min(len(is_EBC), transform.shape[0])
            vfs_x = transform[:n_cells, 2] if transform.shape[1] >= 4 else np.full(n_cells, np.nan)
            vfs_y = transform[:n_cells, 3] if transform.shape[1] >= 4 else np.full(n_cells, np.nan)

            regions = _get_cell_regions(
                messentials_pos, n_cells, labeled_array, vfs_x, vfs_y)

            for ci in range(n_cells):
                cells.append({
                    'area_id':               int(regions[ci]),
                    'vfs_x':                 float(vfs_x[ci]),
                    'vfs_y':                 float(vfs_y[ci]),
                    'is_EBC':                int(is_EBC[ci]),
                    'is_RBC':                int(is_RBC[ci]),
                    'is_fully_reliable_EBC': int(is_fr_e[ci]) if ci < len(is_fr_e) else 0,
                    'is_fully_reliable_RBC': int(is_fr_r[ci]) if ci < len(is_fr_r) else 0,
                    'ebc_mrl':               float(ebc_mrl[ci]) if ci < len(ebc_mrl) else np.nan,
                    'rbc_mrl':               float(rbc_mrl[ci]) if ci < len(rbc_mrl) else np.nan,
                    'ebc_corr_coeff':        float(ebc_corr[ci]) if ci < len(ebc_corr) else np.nan,
                    'rbc_corr_coeff':        float(rbc_corr[ci]) if ci < len(rbc_corr) else np.nan,
                })

            ebc_maps_arr = bdata.get('ebc_smoothed_rate_maps')
            rbc_maps_arr = bdata.get('rbc_smoothed_rate_maps')
            if ebc_maps_arr is not None:
                ebc_maps_arr = np.asarray(ebc_maps_arr)
                for ci in range(min(n_cells, len(is_EBC))):
                    if is_EBC[ci]:
                        aname = id_to_name.get(int(regions[ci]))
                        if aname:
                            area_ebc[aname].append(ebc_maps_arr[ci])
            if rbc_maps_arr is not None:
                rbc_maps_arr = np.asarray(rbc_maps_arr)
                for ci in range(min(n_cells, len(is_RBC))):
                    if is_RBC[ci]:
                        aname = id_to_name.get(int(regions[ci]))
                        if aname:
                            area_rbc[aname].append(rbc_maps_arr[ci])

            if params is None and 'dist_bin_edges' in bdata:
                params = {k: np.asarray(bdata[k])
                          for k in ('dist_bin_edges', 'dist_bin_cents', 'angle_rad')
                          if k in bdata}
                params['ray_width'] = float(bdata.get('ray_width', 5.0))

    if not cells:
        print('  aggregate_boundary_data: no boundary data found in messentials.')
        return {}

    # Flatten per-cell dicts into arrays
    cell_data = {k: np.array([c[k] for c in cells]) for k in cells[0]}

    ebc_mean = {a: np.nanmean(np.stack(maps), axis=0) for a, maps in area_ebc.items()}
    rbc_mean = {a: np.nanmean(np.stack(maps), axis=0) for a, maps in area_rbc.items()}

    n_ebc = int(np.sum(cell_data['is_EBC']))
    n_rbc = int(np.sum(cell_data['is_RBC']))
    print(f'  aggregate_boundary_data: {len(cells)} cells, '
          f'{n_ebc} EBC ({len(ebc_mean)} areas), {n_rbc} RBC ({len(rbc_mean)} areas)')

    out = {
        'boundary_cell_data':      cell_data,
        'boundary_ebc_mean_maps':  ebc_mean,
        'boundary_rbc_mean_maps':  rbc_mean,
    }
    if params is not None:
        out['boundary_params'] = params
    return out


def main():

    uniref = read_h5('/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites/DMM056/animal_reference_260115_10h-06m-52s.h5')
    data = read_h5('/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites/pooled_260331a.h5')
    root_dir = '/home/dylan/Storage/freely_moving_data/_V1PPC'

    variables = ['theta', 'phi', 'dTheta', 'dPhi', 'pitch', 'yaw', 'roll', 'dPitch', 'dYaw', 'dRoll']
    conditions = ['l', 'd']
    animal_dirs = ['DMM037','DMM041', 'DMM042','DMM056', 'DMM061']

    # Build the labeled array from VFS reference contours embedded in the
    # pooled data by make_pooled_dataset().  Cell coordinates in
    # data[animal]['transform'][poskey][:,2:4] are in the same reference VFS
    # coordinate space, so no additional transform is needed for area lookup.
    labeled_array, label_map = get_labeled_array_from_contours(data)
    img_array = None  # no longer used for area labeling
    
    master_dict = {'labeled_array': labeled_array}

    with PdfPages('/home/dylan/Fast2/topography_summary_260318a.pdf') as pdf:

        res = make_behavior_corr_matrix(pdf, data, root_dir)
        if res: master_dict.update(res)
        
        plot_region_outlines(pdf, labeled_array, label_map)

        for key in tqdm(variables, desc="Processing variables"):
            for cond in conditions:
                res = plot_variable_summary(
                    pdf, data, key, cond, uniref, img_array,
                    animal_dirs, labeled_array, label_map
                )
                if res: master_dict.update(res)
                
                # res = plot_signal_noise_correlations(
                #     pdf, data, key, cond, animal_dirs, labeled_array, label_map
                # )
                # if res: master_dict.update(res)
        
        res = plot_all_variable_importance(pdf, data, animal_dirs, labeled_array, label_map)
        if res: master_dict.update(res)
        
        res = plot_all_model_performance(pdf, data, animal_dirs, labeled_array, label_map)
        if res: master_dict.update(res)

        res = plot_model_performance(
            pdf, data, animal_dirs, labeled_array, label_map
        )
        if res: master_dict.update(res)
        
        # res = plot_manifold_analysis(pdf, data, animal_dirs, labeled_array, label_map, root_dir, img_array)
        # if res: master_dict.update(res)
        
        res = plot_sorted_tuning_curves(pdf, data, animal_dirs, cond='l')
        if res: master_dict.update(res)
        res = plot_sorted_tuning_curves(pdf, data, animal_dirs, cond='d')
        if res: master_dict.update(res)
        
        res = plot_modulation_summary(pdf, data, animal_dirs, labeled_array, label_map, cond='l', savedir='/home/dylan/Fast2')
        if res: master_dict.update(res)
        res = plot_modulation_summary(pdf, data, animal_dirs, labeled_array, label_map, cond='d', savedir='/home/dylan/Fast2')
        if res: master_dict.update(res)
        
        res = plot_modulation_histograms(pdf, data, animal_dirs, labeled_array, label_map, cond='l')
        if res: master_dict.update(res)
        res = plot_modulation_histograms(pdf, data, animal_dirs, labeled_array, label_map, cond='d')
        if res: master_dict.update(res)

        # Cross-condition modulation histograms
        res = plot_lightdark_modulation_histograms(pdf, data, animal_dirs, labeled_array, label_map)
        if res: master_dict.update(res)

        # Position occupancy scatter plots (one per recording)
        res = plot_position_occupancy(pdf, data, animal_dirs, root_dir)
        if res: master_dict.update(res)

    # Aggregate boundary-cell data (no PDF pages; pure data aggregation)
    res = aggregate_boundary_data(data, animal_dirs, labeled_array, label_map)
    if res: master_dict.update(res)

    write_h5('/home/dylan/Fast2/topography_analysis_results_260331a.h5', master_dict)


if __name__ == '__main__':

    main()
