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
from scipy.interpolate import griddata
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import skimage.transform

from sklearn.decomposition import PCA

import umap

import fm2p


def plot_running_median(ax, x, y, n_bins=7, vertical=False, fb=True, color='k'):

    mask = ~np.isnan(x) & ~np.isnan(y)
    
    if np.sum(mask) == 0:
        return np.nan

    x_use = x[mask]
    y_use = y[mask]

    bins = np.linspace(np.min(x_use), np.max(x_use), n_bins)

    bin_means, bin_edges, _ = scipy.stats.binned_statistic(
        x_use,
        y_use,
        statistic=np.median,
        bins=bins)
    
    bin_std, _, _ = scipy.stats.binned_statistic(
        x_use,
        y_use,
        statistic=np.nanstd,
        bins=bins)
    
    hist, _ = np.histogram(
        x_use,
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


def add_scatter_col(ax, pos, vals):

    ax.scatter(
        np.ones_like(vals)*pos + (np.random.rand(len(vals))-0.5)/2,
        vals,
        s=2, c='k'
    )
    ax.hlines(np.nanmean(vals), pos-.1, pos+.1, color='r')

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
        use_key = reverse_map[key]

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


def create_smoothed_map(x, y, values, shape=(1024, 1024), sigma=25): # started as 10
    
    if len(values) < 4:
        return np.full(shape, np.nan)

    grid_y, grid_x = np.mgrid[0:shape[0], 0:shape[1]]
    points = np.column_stack((y, x))
    
    smoothed = griddata(points, values, (grid_y, grid_x), method='linear')
    
    # Nan-aware gaussian smoothing
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
                
                if isinstance(model_data, dict):
                    if imp_key and imp_key in model_data and c < len(model_data[imp_key]):
                        c_imp = model_data[imp_key][c]

                cells.append({
                    'x': transform[c, 2],
                    'y': transform[c, 3],
                    'rel': isrel[c],
                    'mod': mod[c],
                    'peak': peak[c] if peak is not None else np.nan,
                    'imp': c_imp # feature importance
                })

    if not cells:
        return

    df = pd.DataFrame(cells)
    cond_name = 'Light' if cond == 'l' else 'Dark'

    metrics_to_plot = ['mod']
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
            norm = colors.Normalize(vmin=-0.05, vmax=0.1)
            label_str = 'Variable Importance (Shuffle)'

        if metric == 'peak':
            rel = df[(df['rel'] == 1) & (df['mod'] > 0.33)]
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
        
        for i in range(4):
            region_vals = rel[rel['region'] == i+2][metric]
            if len(region_vals) > 0:
                add_scatter_col(ax, i, region_vals)

        ax.set_xticks(np.arange(4), labels=list(label_map.values())[2:6])
        if metric == 'mod':
            ax.set_ylim([0,0.75])
        elif metric == 'imp':
            ax.set_ylim([-0.1, 0.25])
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
                smoothed = create_smoothed_map(region_cells['x'].values, region_cells['y'].values, region_cells[metric].values, shape=labeled_array.shape)
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
            fig_rot, (ax_rot_x, ax_rot_y) = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
            
            theta = np.pi / 4  # 45 deg CCW
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            
            rel_rot = rel.copy()
            rel_rot['x_rot'] = rel['x'] * cos_t - rel['y'] * sin_t
            rel_rot['y_rot'] = rel['x'] * sin_t + rel['y'] * cos_t

            colors_region = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

            max_y_val = 0

            for i in range(4):
                region_id = [2,3,4,5][i]
                region_name = label_map.get(region_id, f'Region {region_id}')
                
                region_cells = rel_rot[rel_rot['region'] == region_id]
                
                if len(region_cells) > 0:
                    x_rot_norm = (region_cells['x_rot'] - region_cells['x_rot'].min()) / (region_cells['x_rot'].max() - region_cells['x_rot'].min())
                    y_rot_norm = (region_cells['y_rot'] - region_cells['y_rot'].min()) / (region_cells['y_rot'].max() - region_cells['y_rot'].min())

                    my1 = plot_running_median(ax_rot_x, x_rot_norm, region_cells[metric], n_bins=10, fb=True, color=colors_region[i])
                    ax_rot_x.plot([], [], color=colors_region[i], label=region_name)

                    my2 = plot_running_median(ax_rot_y, y_rot_norm, region_cells[metric], n_bins=10, fb=True, color=colors_region[i])
                    ax_rot_y.plot([], [], color=colors_region[i], label=region_name)
                    
                    max_y_val = np.nanmax([max_y_val, my1, my2])

            if max_y_val > 0:
                ax_rot_x.set_ylim(bottom=0.0, top=max_y_val*1.1)
                ax_rot_y.set_ylim(bottom=0.0, top=max_y_val*1.1)

            ax_rot_x.set_title('Along Rotated X-axis (45 deg CCW)')
            ax_rot_x.set_xlabel('Position along rotated axis')
            ax_rot_x.set_ylabel('Modulation Index')
            ax_rot_x.legend()

            ax_rot_y.set_title('Along Rotated Y-axis (45 deg CCW)')
            ax_rot_y.set_xlabel('Position along rotated axis')
            ax_rot_y.legend()

            fig_rot.suptitle(f'{label_str} along Rotated Axes for {key} ({cond_name})')
            fig_rot.tight_layout()
            pdf.savefig(fig_rot)
            plt.close(fig_rot)


def plot_signal_noise_correlations(pdf, data, key, cond, animal_dirs, labeled_array, label_map):
    
    pooled_sig = []
    pooled_noise = []
    pooled_regions = []

    cond_idx = 1 if cond == 'l' else 0
    cond_name = 'Light' if cond == 'l' else 'Dark'

    use_key = key
    reverse_map = {'dRoll': 'gyro_x', 'dPitch': 'gyro_y', 'dYaw': 'gyro_z'}
    if key in reverse_map:
        use_key = reverse_map[key]

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

            tuning_key = f'{use_key}_1dtuning'
            if tuning_key not in rdata: continue
            
            if 'full_y_test' not in model_data or 'full_y_hat' not in model_data:
                continue

            tuning_curves = rdata[tuning_key] # (n_cells, n_bins, n_conds)
            y_true = model_data['full_y_test']
            y_hat = model_data['full_y_hat']
            residuals = y_true - y_hat

            points = list(zip(transform[:, 2], transform[:, 3]))
            results = get_region_for_points(labeled_array, points, label_map)
            regions = results[:, 3]

            n_cells = tuning_curves.shape[0]
            if n_cells != residuals.shape[1]:
                continue

            # pairwise corr within a single FOV
            sig_corr_mat = np.corrcoef(tuning_curves[:, :, cond_idx])
            noise_corr_mat = np.corrcoef(residuals.T) # transpose to (cells, time)

            iu = np.triu_indices(n_cells, k=1)
            
            pooled_sig.extend(sig_corr_mat[iu])
            pooled_noise.extend(noise_corr_mat[iu])
            
            # region pairs (region_i, region_j)
            for i, j in zip(iu[0], iu[1]):
                pooled_regions.append((regions[i], regions[j]))

    if not pooled_sig:
        return

    pooled_sig = np.array(pooled_sig)
    pooled_noise = np.array(pooled_noise)
    pooled_regions = np.array(pooled_regions)

    fig, axs = plt.subplots(2, 3, figsize=(12, 8), dpi=300)
    axs = axs.flatten()

    axs[0].scatter(pooled_sig, pooled_noise, s=1, c='k', alpha=0.1)
    axs[0].set_title('All Pairs')
    axs[0].set_xlabel('Signal Correlation')
    axs[0].set_ylabel('Noise Correlation')

    region_ids = [5, 2, 3, 4] # V1, RL, AM, PM
    region_names = ['V1', 'RL', 'AM', 'PM']

    for i, (rid, rname) in enumerate(zip(region_ids, region_names)):
        # filt for just when they're from the same visual area
        mask = (pooled_regions[:, 0] == rid) & (pooled_regions[:, 1] == rid)
        
        ax = axs[i+1]
        if np.sum(mask) > 0:
            ax.scatter(pooled_sig[mask], pooled_noise[mask], s=1, c='k', alpha=0.1)
            
            r_val = np.corrcoef(pooled_sig[mask], pooled_noise[mask])[0,1]
            ax.text(0.05, 0.9, f'r={r_val:.2f}', transform=ax.transAxes)

        ax.set_title(f'{rname} Pairs')
        ax.set_xlabel('Signal Correlation')
        ax.set_ylabel('Noise Correlation')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-0.5, 0.5])

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


def plot_manifold_analysis(pdf, data, animal_dirs, labeled_array, label_map, root_dir):
    
    if umap is None:
        print("UMAP not installed, skipping manifold analysis.")
        return

    regions_to_plot = [5, 2, 3, 4] # V1, RL, AM, PM
    region_names = ['V1', 'RL', 'AM', 'PM']
    
    print(f"Starting manifold analysis for {len(animal_dirs)} animals.")

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
            points = list(zip(transform[:, 2], transform[:, 3]))
            results = get_region_for_points(labeled_array, points, label_map)
            cell_regions = results[:, 3]
            
            spikes = pdata['norm_spikes'].T # (n_frames, n_cells)
            beh_df = get_aligned_behavior(pdata)
            
            valid_frames = beh_df.notna().all(axis=1)
            if valid_frames.sum() < 100: continue
            n_valid = valid_frames.sum()
            if n_valid < 100: 
                print(f"Skipping {animal} {poskey}: insufficient valid frames ({n_valid} < 100).")
                continue
            
            spikes_valid = spikes[valid_frames]
            beh_valid = beh_df[valid_frames]
            
            fig = plt.figure(figsize=(15, 12), dpi=300)
            gs = fig.add_gridspec(len(regions_to_plot), 5)
            
            has_plot = False
            
            for i, (rid, rname) in enumerate(zip(regions_to_plot, region_names)):
                
                region_mask = (cell_regions == rid)
                n_cells_region = np.sum(region_mask)
                if n_cells_region < 10:
                    continue
                
                print(f"Plotting region {rname} for {animal} {poskey} with {n_cells_region} cells.")
                has_plot = True
                
                X = spikes_valid[:, region_mask]
                
                pca = PCA(n_components=3)
                X_pca = pca.fit_transform(X)
                
                reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
                if X.shape[0] > 5000:
                    idx = np.linspace(0, X.shape[0]-1, 5000, dtype=int)
                    X_umap = reducer.fit_transform(X[idx])

                else:
                    X_umap = reducer.fit_transform(X)
                    idx = np.arange(X.shape[0])
                
                ax_umap = fig.add_subplot(gs[i, 0])
                ax_umap.scatter(X_umap[:, 0], X_umap[:, 1], s=1, c='k', alpha=0.3)
                ax_umap.set_title(f'{rname} UMAP (n={np.sum(region_mask)})')
                ax_umap.axis('off')
                
                ax_pca = fig.add_subplot(gs[i, 1])
                ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], s=1, c='k', alpha=0.3)
                ax_pca.set_title(f'{rname} PCA')
                ax_pca.set_xlabel('PC1')
                ax_pca.set_ylabel('PC2')
                
                for pc_idx in range(3):
                    ax_corr = fig.add_subplot(gs[i, 2 + pc_idx])
                    
                    corrs = {}
                    for col in beh_valid.columns:
                        r = np.corrcoef(X_pca[:, pc_idx], beh_valid[col])[0, 1]
                        corrs[col] = r
                    
                    sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
                    
                    keys = [x[0] for x in sorted_corrs]
                    vals = [x[1] for x in sorted_corrs]
                    
                    y_pos = np.arange(len(keys))
                    ax_corr.barh(y_pos, vals, align='center')
                    ax_corr.set_yticks(y_pos)
                    ax_corr.set_yticklabels(keys, fontsize=6)
                    ax_corr.invert_yaxis()
                    ax_corr.set_title(f'PC{pc_idx+1} Corrs')
                    ax_corr.set_xlim([-1, 1])
                    ax_corr.grid(axis='x', linestyle='--', alpha=0.5)

            if has_plot:
                print(f"Saving figure for {animal} {poskey}")
                fig.suptitle(f'Manifold Analysis: {animal} {poskey}')
                fig.tight_layout()
                pdf.savefig(fig)
            else:
                print(f"No regions plotted for {animal} {poskey}")
            
            plt.close(fig)


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

    with PdfPages('topography_summary_v07e.pdf') as pdf:
    
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
        
        plot_manifold_analysis(pdf, data, animal_dirs, labeled_array, label_map, root_dir)

if __name__ == '__main__':

    main()
