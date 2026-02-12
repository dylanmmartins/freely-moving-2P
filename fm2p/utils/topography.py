# -*- coding: utf-8 -*-

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

import umap

import fm2p


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
    return ['#E6194B', '#F032E6', '#46F0F0', '#bcbd22'] # Red, Magenta, Cyan, olive


def get_equally_spaced_colormap_values(colormap_name, num_values):
    if not isinstance(num_values, int) or num_values <= 0:
        raise ValueError("num_values must be a positive integer.")
    if colormap_name == 'parula':
        cmap = fm2p.make_parula()
    elif colormap_name == 'earth_tones':
        cmap = make_earth_tones()
    else:
        cmap = cm.get_cmap(colormap_name)
    normalized_positions = np.linspace(0, 1, num_values)
    colors = [cmap(pos) for pos in normalized_positions]
    return colors

goodred = '#D96459'


def add_scatter_col(ax, pos, vals, color='k'):

    ax.scatter(
        np.ones_like(vals)*pos + (np.random.rand(len(vals))-0.5)/2,
        vals,
        s=2, c=color
    )
    ax.hlines(np.nanmean(vals), pos-.1, pos+.1, color='k')

    stderr = np.nanstd(vals) / np.sqrt(len(vals))
    ax.vlines(pos, np.nanmean(vals)-stderr, np.nanmean(vals)+stderr, color='r')


def make_aligned_sign_maps(map_items, animal_dirs, pdf=None):

    uniref = map_items['uniref']
    main_basepath = map_items['composite_basepath']
    img_array = map_items['img_array']

    fig, ax = plt.subplots(1, 1, figsize=(2,2), dpi=300)

    ax.imshow(gaussian_filter(zoom(uniref['overlay'][:,:,0], 2.555),12), cmap='gray', alpha=0.3)

    for animal_dir in animal_dirs:

        basepath = os.path.join(main_basepath, animal_dir)
        if animal_dir != 'DMM056':
            transform_g2u = fm2p.read_h5(fm2p.find('aligned_composite_*.h5', basepath, MR=True))
            messentials = fm2p.read_h5(fm2p.find('*_merged_essentials_v6.h5', basepath, MR=True))
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


def get_labeled_array(image):
    label_map = {
        0: 'boundary',
        1: 'outside',
        2: 'RL',
        3: 'AM',
        4: 'PM',
        5: 'V1',
        7: 'AL',
        8: 'LM',
        9: 'P'
    }

    labeled_array = label(image, connectivity=1)
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


def get_cell_data(rdata, key, cond):

    use_key = key
    reverse_map = {'dRoll': 'gyro_x', 'dPitch': 'gyro_y', 'dYaw': 'gyro_z'}

    if key in reverse_map:
        mapped = reverse_map[key]
        if f'{mapped}_{cond}_isrel' in rdata:
            use_key = mapped
        elif f'{key}_{cond}_isrel' in rdata:
            use_key = key
        else:
            use_key = mapped

    isrel = None
    mod = None
    peak = None

    isrel_key = f'{use_key}_{cond}_isrel'
    mod_key = f'{use_key}_{cond}_mod'
    
    if isrel_key in rdata:
        isrel = rdata[isrel_key]
        mod = rdata[mod_key]
    
    pref_key = f'{use_key}_{cond}_pref'
    if pref_key in rdata:
        peak = rdata[pref_key]
    else:
        tuning_key = f'{use_key}_1dtuning'
        bins_key = f'{use_key}_1dbins'
        
        if tuning_key in rdata and bins_key in rdata:
            tuning = rdata[tuning_key]
            bins = rdata[bins_key]

            cond_idx = 1 if cond == 'l' else 0
            
            if tuning.ndim == 3 and tuning.shape[2] > cond_idx:
                 peak_indices = np.argmax(tuning[:, :, cond_idx], axis=1)
                 peak = bins[peak_indices]

    return isrel, mod, peak


def get_glm_keys(key):

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


def plot_variable_summary(pdf, data, key, cond, uniref, img_array, animal_dirs, labeled_array, label_map):
    
    imp_key = get_glm_keys(key)
    
    cells = []
    
    for animal_dir in animal_dirs:
        if animal_dir not in data.keys():
            continue
            
        if 'transform' not in data[animal_dir].keys():
             continue

        for poskey in data[animal_dir]['transform'].keys():
            if (animal_dir=='DMM056') and (cond=='d') and ((poskey=='pos15') or (poskey=='pos03')):
                continue

            isrel, mod, peak = get_cell_data(data[animal_dir]['messentials'][poskey]['rdata'], key, cond)
            
            if isrel is None:
                continue
            
            transform = data[animal_dir]['transform'][poskey]

            model_data = data[animal_dir]['messentials'][poskey].get('model', {})

            for c in range(np.size(isrel, 0)):
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
                    'rel': isrel[c],
                    'mod': mod[c],
                    'peak': peak[c] if peak is not None else np.nan,
                    'imp': c_imp, # feature importance
                    'full_r2': c_r2
                })

    if not cells:
        return

    area_colors = make_area_colors()

    df = pd.DataFrame(cells)
    cond_name = 'Light' if cond == 'l' else 'Dark'

    metrics_to_plot = ['mod']
    if key not in ['dTheta', 'dPhi']:
        metrics_to_plot.append('peak')
    metrics_to_plot.append('imp')

    for metric in metrics_to_plot:
        
        if metric == 'mod':
            cmap = cm.plasma
            norm = colors.Normalize(vmin=0, vmax=0.5)
            label_str = 'Modulation Index'
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

        if metric == 'peak':
            rel = df[(df['rel'] == 1) & (df['mod'] > 0.33)]
        elif metric == 'imp':
            rel = df[(df['rel'] == 1) & (df['full_r2'] > 0.1)]
        else:
            rel = df[df['rel'] == 1]

        if len(rel) == 0:
            continue
            
        points_lt = list(zip(rel['x'], rel['y']))
        results = get_region_for_points(labeled_array, points_lt, label_map)
        
        rel = rel.copy()
        rel['region'] = results[:, 3]

        fig, ax = plt.subplots(1, 1, figsize=(5,3.5), dpi=300)
        if metric == 'mod':
            ax.hlines(0.33, -1, 7, color='tab:grey', ls='--', alpha=0.56)
            ax.hlines(0.5, -1, 7, color='tab:grey', ls='--', alpha=0.56)
        
        groups = []
        for i in range(4):
            region_vals = rel[rel['region'] == i+2][metric]
            groups.append(region_vals)
            if len(region_vals) > 0:
                add_scatter_col(ax, i, region_vals, color=area_colors[i])

        valid_groups = [g for g in groups if len(g) > 0]
        if len(valid_groups) > 1:
            try:
                stat, p_kw = kruskal(*valid_groups)
                ax.text(0.05, 0.95, f'KW p={p_kw:.1e}', transform=ax.transAxes, fontsize=8)
                
                # Just indicating global significance for now to avoid clutter
                if p_kw < 0.05:
                    ax.text(0.05, 0.90, '*', transform=ax.transAxes, fontsize=12, color='r')
            except ValueError:
                pass
        else:
            ax.text(0.05, 0.95, 'Insufficient data for stats', transform=ax.transAxes, fontsize=6)

        ax.set_xticks(np.arange(4), labels=list(label_map.values())[2:6])
        if metric == 'mod':
            ax.set_ylim([0,0.75])
        elif metric == 'imp':
            ax.set_ylim([0, 0.25])
        ax.set_xlim([-.5,3.5])
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

                    my1 = plot_running_median(ax_rot_x, x_rot_norm, region_cells[metric], n_bins=7, fb=True, color=area_colors[i])
                    ax_rot_x.plot([], [], color=area_colors[i], label=region_name)

                    my2 = plot_running_median(ax_rot_y, y_rot_norm, region_cells[metric], n_bins=7, fb=True, color=area_colors[i])
                    ax_rot_y.plot([], [], color=area_colors[i], label=region_name)
                    
                    max_y_val = np.nanmax([max_y_val, my1, my2])

            if metric == 'mod':
                ax_rot_x.set_ylim(bottom=0.0, top=max_y_val*1.1)
                ax_rot_y.set_ylim(bottom=0.0, top=max_y_val*1.1)
            elif metric == 'peak':
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


def plot_signal_noise_correlations(pdf, data, key, cond, animal_dirs, labeled_array, label_map):
    
    print(f"Calculating signal/noise correlations for {key} ({cond})")
    
    pooled_sig = []
    pooled_noise = []
    pooled_regions = []
    area_colors = make_area_colors()

    cond_idx = 1 if cond == 'l' else 0
    cond_name = 'Light' if cond == 'l' else 'Dark'

    for animal_dir in animal_dirs:
        if animal_dir not in data: continue
        if 'transform' not in data[animal_dir]: continue

        for poskey in data[animal_dir]['transform']:
             # skip bad sessions
            if (animal_dir=='DMM056') and (cond=='d') and ((poskey=='pos15') or (poskey=='pos03')):
                continue

            messentials = data[animal_dir]['messentials'][poskey]
            rdata = messentials.get('rdata', {})
            model_data = messentials.get('model', {})
            transform = data[animal_dir]['transform'][poskey]

            use_key = key
            reverse_map = {'dRoll': 'gyro_x', 'dPitch': 'gyro_y', 'dYaw': 'gyro_z'}
            
            if key in reverse_map:
                mapped = reverse_map[key]
                if f'{mapped}_1dtuning' in rdata:
                    use_key = mapped
                elif f'{key}_1dtuning' in rdata:
                    use_key = key
                else:
                    use_key = mapped

            tuning_key = f'{use_key}_1dtuning'
            if tuning_key not in rdata: 
                print(f"Skipping {animal_dir} {poskey}: {tuning_key} not in rdata")
                continue
            
            y_true_key = 'full_y_test'
            if y_true_key not in model_data:
                if 'full_y_true' in model_data:
                    y_true_key = 'full_y_true'
                else:
                    print(f"Skipping {animal_dir} {poskey}: y_true/y_test not in model_data")
                    continue

            if 'full_y_hat' not in model_data:
                print(f"Skipping {animal_dir} {poskey}: full_y_hat not in model_data")
                continue

            tuning_curves = rdata[tuning_key] # (n_cells, n_bins, n_conds)
            y_true = model_data[y_true_key]
            y_hat = model_data['full_y_hat']
            residuals = y_true - y_hat

            points = list(zip(transform[:, 2], transform[:, 3]))
            results = get_region_for_points(labeled_array, points, label_map)
            regions = results[:, 3]

            n_cells = tuning_curves.shape[0]
            if n_cells != residuals.shape[1]:
                print(f"Skipping {animal_dir} {poskey}: n_cells mismatch ({n_cells} vs {residuals.shape[1]})")
                continue

            if tuning_curves.ndim == 3:
                sig_corr_mat = np.corrcoef(tuning_curves[:, :, cond_idx])
            elif tuning_curves.ndim == 2:
                sig_corr_mat = np.corrcoef(tuning_curves)
            else:
                print(f"Skipping {animal_dir} {poskey}: tuning_curves shape {tuning_curves.shape} unexpected")
                continue

            noise_corr_mat = np.corrcoef(residuals.T) # transpose to (cells, time)

            iu = np.triu_indices(n_cells, k=1)
            
            pooled_sig.extend(sig_corr_mat[iu])
            pooled_noise.extend(noise_corr_mat[iu])
            
            # region pairs (region_i, region_j)
            for i, j in zip(iu[0], iu[1]):
                pooled_regions.append((regions[i], regions[j]))

    if not pooled_sig:
        print(f"No pooled data for {key} ({cond})")
        return

    pooled_sig = np.array(pooled_sig)
    pooled_noise = np.array(pooled_noise)
    pooled_regions = np.array(pooled_regions)

    print(f"Plotting signal/noise for {key} ({cond}) with {len(pooled_sig)} pairs")

    fig, axs = plt.subplots(2, 3, figsize=(12, 8), dpi=300)
    axs = axs.flatten()

    axs[0].scatter(pooled_sig[::2000], pooled_noise[::2000], s=3, c='k')
    axs[0].set_title('All Pairs')
    axs[0].set_xlabel('Signal Correlation')
    axs[0].set_ylabel('Noise Correlation')
    axs[0].set_xlim([-1, 1])
    axs[0].set_ylim([-1, 1])

    region_ids = [5, 2, 3, 4] # V1, RL, AM, PM
    region_names = ['V1', 'RL', 'AM', 'PM']

    for i, (rid, rname) in enumerate(zip(region_ids, region_names)):
        # filt for just when they're from the same visual area
        mask = (pooled_regions[:, 0] == rid) & (pooled_regions[:, 1] == rid)
        
        ax = axs[i+1]
        if np.sum(mask) > 0:
            ax.scatter(pooled_sig[mask][::2000], pooled_noise[mask][::2000], s=3, c=area_colors[i])
            
            sig_m = pooled_sig[mask]
            noise_m = pooled_noise[mask]
            valid = np.isfinite(sig_m) & np.isfinite(noise_m)
            if np.sum(valid) > 1:
                r_val = np.corrcoef(sig_m[valid], noise_m[valid])[0,1]
                ax.text(0.05, 0.9, f'r={r_val:.2f}', transform=ax.transAxes)
            else:
                ax.text(0.05, 0.9, f'r=NaN', transform=ax.transAxes)

        ax.set_title(f'{rname} Pairs')
        ax.set_xlabel('Signal Correlation')
        ax.set_ylabel('Noise Correlation')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])

    axs[5].axis('off')
    
    fig.suptitle(f'Signal vs Noise Correlations: {key} ({cond_name})')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def get_aligned_behavior(pdata):
    """ Extract and align behavioral variables from preproc data. """
    
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
                beh['dTheta'] = fm2p.interpT(dTh, t_src, twopT)
            elif len(dTh) == len(eyeT):
                beh['dTheta'] = fm2p.interpT(dTh, eyeT, twopT)
                
        if 'dPhi' in pdata:
            dPh = pdata['dPhi']
            if len(dPh) == len(eyeT) - 1:
                t_src = eyeT[:-1] + np.diff(eyeT)/2
                beh['dPhi'] = fm2p.interpT(dPh, t_src, twopT)
            elif len(dPh) == len(eyeT):
                beh['dPhi'] = fm2p.interpT(dPh, eyeT, twopT)
                
        if 'longaxis' in pdata:
            la = pdata['longaxis'][int(pdata['eyeT_startInd']):int(pdata['eyeT_endInd'])]
            if len(la) == len(eyeT):
                beh['pupil'] = fm2p.interpT(la, eyeT, twopT)

    if 'head_yaw_deg' in pdata and 'imuT_trim' in pdata:
        yaw = pdata['head_yaw_deg']
        imuT = pdata['imuT_trim']
        if len(yaw) == len(imuT):
             beh['yaw'] = fm2p.interpT(yaw, imuT, twopT)
        elif len(yaw) == len(imuT) + 1:
             beh['yaw'] = fm2p.interpT(yaw[:-1], imuT, twopT)

    return pd.DataFrame(beh)


def plot_manifold_analysis(pdf, data, animal_dirs, labeled_array, label_map, root_dir, img_array=None):
    
    if umap is None:
        print("UMAP not installed, skipping manifold analysis.")
        return

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

    collected_results = []

    for animal in animal_dirs:
        if animal not in data: 
            print(f"Skipping {animal}: not in data dictionary.")
            continue
        
        print(f"Processing animal: {animal}")
        for poskey in data[animal]['transform']:
            
            try:
                pos_num = int(poskey.replace('pos', ''))
                pos_str = f'pos{pos_num:02d}'
            except:
                print(f"Skipping {poskey}: could not parse position number.")
                continue
            
            filename_pattern = f'*{animal}*preproc.h5'
            try:
                candidates = fm2p.find(filename_pattern, root_dir, MR=False)
            except:
                print(f"Skipping {animal} {poskey}: no preproc files found for animal pattern {filename_pattern}")
                continue
            
            valid_candidates = [c for c in candidates if pos_str in c]
            
            if not valid_candidates:
                print(f"Skipping {animal} {poskey}: preproc files found but none matched {pos_str} in path.")
                continue
            
            ppath = fm2p.choose_most_recent(valid_candidates)
            print(f"Loading {ppath}")
            pdata = fm2p.read_h5(ppath)
            if 'norm_spikes' not in pdata: 
                print(f"Skipping {animal} {poskey}: 'norm_spikes' not in pdata.")
                continue
            
            transform = data[animal]['transform'][poskey]

            cell_indices = transform[:, 0].astype(int)

            points = list(zip(transform[:, 2], transform[:, 3]))
            results = get_region_for_points(labeled_array, points, label_map)
            cell_regions = results[:, 3]
            
            spikes = pdata['norm_spikes'].T # (n_frames, n_cells)

            if len(cell_indices) > 0:
                if spikes.shape[1] == len(cell_indices):
                    pass
                elif np.max(cell_indices) < spikes.shape[1]:
                    spikes = spikes[:, cell_indices]
                else:
                    print(f"Skipping {animal} {poskey}: transform indices max ({np.max(cell_indices)}) >= spikes shape ({spikes.shape[1]}).")
                    continue

            beh_df = get_aligned_behavior(pdata)
            
            valid_frames = beh_df.notna().all(axis=1)
            if valid_frames.sum() < 100: continue
            n_valid = valid_frames.sum()
            if n_valid < 100: 
                print(f"Skipping {animal} {poskey}: insufficient valid frames ({n_valid} < 100).")
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

                if 'theta' in beh_valid.columns:
                    c_vals = beh_valid['theta'].values
                    c_label = 'Theta'
                elif 'speed' in beh_valid.columns:
                    c_vals = beh_valid['speed'].values
                    c_label = 'Speed'
                else:
                    c_vals = np.arange(len(X))
                    c_label = 'Time'
                
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
                    'ev_curve': pca.explained_variance_ratio_,
                    'pc_corrs': pc_corrs
                })

    rows_per_page = 5
    n_results = len(collected_results)
    
    if n_results == 0:
        print("No manifold analysis results to plot.")
        return

    for start_idx in range(0, n_results, rows_per_page):
        end_idx = min(start_idx + rows_per_page, n_results)
        batch = collected_results[start_idx:end_idx]
        
        fig = plt.figure(figsize=(15, 3 * len(batch)), dpi=300)
        gs = fig.add_gridspec(len(batch), 5)
        
        for i, res in enumerate(batch):

            ax_umap = fig.add_subplot(gs[i, 0])
            ax_umap.scatter(res['umap'][:, 0], res['umap'][:, 1], s=1, c=res['color_vals'], cmap=earth_cmap, alpha=0.5)
            ax_umap.set_title(f"{res['animal']} {res['poskey']} {res['region']}", fontsize=8)
            ax_umap.axis('off')
            
            ax_pca = fig.add_subplot(gs[i, 1], projection='3d')
            ax_pca.scatter(res['pca'][:, 0], res['pca'][:, 1], res['pca'][:, 2], s=1, c=res['color_vals'], cmap=earth_cmap, alpha=0.5)
            ax_pca.set_title(f"PCA (EV: {res['explained_variance'][0]:.2f}, {res['explained_variance'][1]:.2f})", fontsize=8)
            ax_pca.set_xlabel('PC1', fontsize=7)
            ax_pca.set_ylabel('PC2', fontsize=7)
            ax_pca.set_zlabel('PC3', fontsize=7)
            ax_pca.tick_params(labelsize=6)
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

    fig_sum, ax_sum = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
    
    area_ev = {r: [] for r in region_names}
    for res in collected_results:
        area_ev[res['region']].append(res['explained_variance'])
    
    # mean variance explained for PC1, PC2, PC3 per area
    x = np.arange(len(region_names))
    width = 0.25
    
    for i, pc_i in enumerate(range(3)):
        means = [np.mean([ev[pc_i] for ev in area_ev[r]]) if area_ev[r] else 0 for r in region_names]
        stds = [np.std([ev[pc_i] for ev in area_ev[r]]) if area_ev[r] else 0 for r in region_names]
        ax_sum.bar(x + i*width, means, width, yerr=stds, label=f'PC{i+1}')
        
    ax_sum.set_xticks(x + width)
    ax_sum.set_xticklabels(region_names)
    ax_sum.set_ylabel('Explained Variance Ratio')
    ax_sum.set_title('PCA Variance Explained by Area')
    ax_sum.legend()
    
    pdf.savefig(fig_sum)
    plt.close(fig_sum)

    # Heatmap of top correlates
    fig_corr, axs_corr = plt.subplots(2, 2, figsize=(6, 5), dpi=300)
    axs_corr = axs_corr.flatten()
    
    for i, region in enumerate(region_names):
        # counts of top correlate
        counts_matrix = np.zeros((len(var_order), 3))
        
        for res in collected_results:
            if res['region'] == region:
                for pc_idx in range(3):
                    corrs = res['pc_corrs'][pc_idx]
                    if not corrs: continue
                    # Find max correlation variable
                    top_var = max(corrs, key=corrs.get)
                    if top_var in var_order:
                        row_idx = var_order.index(top_var)
                        counts_matrix[row_idx, pc_idx] += 1
        
        if np.sum(counts_matrix) > 0:
            im = axs_corr[i].imshow(counts_matrix, cmap='plasma', aspect='auto')
            axs_corr[i].set_yticks(range(len(var_order)))
            axs_corr[i].set_yticklabels(var_order, fontsize=7)
            axs_corr[i].set_xticks([1, 2, 3])
            axs_corr[i].set_xticklabels(['PC1', 'PC2', 'PC3'], fontsize=8)
            axs_corr[i].set_title(f'{region}')
            axs_corr[i].set_xticks(np.arange(3))
            axs_corr[i].set_xticklabels(['PC1', 'PC2', 'PC3'])
        else:
            axs_corr[i].text(0.5, 0.5, 'No Data', ha='center')
            axs_corr[i].set_title(f'{region}')
            
    fig_corr.tight_layout()
    pdf.savefig(fig_corr)
    plt.close(fig_corr)

    fig_dim, ax_dim = plt.subplots(1, 1, figsize=(4,3), dpi=300)
    
    pr_means = []
    pr_stds = []
    
    # participation ratio
    for region in region_names:
        prs = [res['participation_ratio'] for res in collected_results if res['region'] == region]
        if prs:
            pr_means.append(np.mean(prs))
            pr_stds.append(np.std(prs))
        else:
            pr_means.append(0)
            pr_stds.append(0)
            
    ax_dim.bar(region_names, pr_means, yerr=pr_stds, capsize=5, color='tab:purple')
    ax_dim.set_ylabel('Participation Ratio (Dimensionality)')
    ax_dim.set_title('Effective Dimensionality by Region')
    pdf.savefig(fig_dim)
    plt.close(fig_dim)

    # cumulative variance
    fig_scree, ax_scree = plt.subplots(1, 1, figsize=(4, 3), dpi=300)
    for region in region_names:
        curves = [np.cumsum(res['ev_curve']) for res in collected_results if res['region'] == region]
        if curves:
            mean_curve = np.mean(curves, axis=0)
            ax_scree.plot(mean_curve, label=region, marker='o', markersize=3)
    ax_scree.set_xlabel('PC Component')
    ax_scree.set_ylabel('Cumulative Explained Variance')
    ax_scree.set_title('Manifold Compactness (Scree Plot)')
    ax_scree.legend()
    ax_scree.grid(True, alpha=0.3)
    pdf.savefig(fig_scree)
    plt.close(fig_scree)

def plot_model_performance(pdf, data, uniref, img_array, animal_dirs, labeled_array, label_map):
    
    models = ['full', 'velocity_only', 'position_only', 'eyes_only', 'head_only']
    metrics = ['r2', 'corrs']
    
    for model in models:
        for metric in metrics:
            
            cells = []
            for animal_dir in animal_dirs:
                if animal_dir not in data: continue
                if 'transform' not in data[animal_dir]: continue
                
                for poskey in data[animal_dir]['transform']:
                    transform = data[animal_dir]['transform'][poskey]
                    model_data = data[animal_dir]['messentials'][poskey].get('model', {})
                    
                    m_key = f'{model}_{metric}'
                    if m_key not in model_data:
                        continue
                        
                    vals = model_data[m_key]
                    
                    for c in range(len(vals)):
                        cells.append({
                            'x': transform[c, 2],
                            'y': transform[c, 3],
                            'val': vals[c]
                        })
            
            if not cells:
                print(f"No data found for {model} {metric}")
                continue
                
            df = pd.DataFrame(cells)
            
            points_lt = list(zip(df['x'], df['y']))
            results = get_region_for_points(labeled_array, points_lt, label_map)
            df['region'] = results[:, 3]
            
            cmap = cm.plasma
            if metric == 'r2':
                norm = colors.Normalize(vmin=-0.2, vmax=0.3)
                label_str = f'{model} R²'
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


def main():

    uniref = fm2p.read_h5('/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites/DMM056/animal_reference_260115_10h-06m-52s.h5')
    data = fm2p.read_h5('/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites/pooled_260210.h5')
    img = Image.open('/home/dylan/Desktop/V1_HVAs_trace.png').convert("RGBA")
    img_array = np.array(img)
    # composite_basepath = '/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites'
    root_dir = '/home/dylan/Storage/freely_moving_data/_V1PPC'

    variables = ['theta', 'phi', 'dTheta', 'dPhi', 'pitch', 'yaw', 'roll', 'dPitch', 'dYaw', 'dRoll']
    conditions = ['l']
    animal_dirs = ['DMM037','DMM041', 'DMM042','DMM056', 'DMM061']
    labeled_array, label_map = get_labeled_array(img_array[:,:,0].clip(max=1))

    with PdfPages('topography_summary_v07j.pdf') as pdf:
    
        for key in tqdm(variables, desc="Processing variables"):
            for cond in conditions:
                plot_variable_summary(
                    pdf, data, key, cond, uniref, img_array,
                    animal_dirs, labeled_array, label_map
                )
                
                plot_signal_noise_correlations(
                    pdf, data, key, cond, animal_dirs, labeled_array, label_map
                )

        plot_model_performance(
            pdf, data, uniref, img_array, animal_dirs, labeled_array, label_map
        )
        
        plot_manifold_analysis(pdf, data, animal_dirs, labeled_array, label_map, root_dir, img_array)

if __name__ == '__main__':

    main()
