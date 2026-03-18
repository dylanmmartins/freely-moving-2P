# -*- coding: utf-8 -*-
"""topography_plots.py

Reads the HDF5 output from topography.py and regenerates all plots as
individual PNG files.  No new calculations are performed here.

Usage
-----
Edit the paths in main() and run:

    python topography_plots.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from scipy.stats import kruskal
from .files import read_h5

import matplotlib as mpl
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 7


COLORS = {
    # Behavioral variables
    'theta':  '#4B9CD3',   # steel blue
    'phi':    '#E07B54',   # terracotta
    'pitch':  '#5BAD6F',   # medium green
    'roll':   '#C55A5A',   # muted red
    'yaw':    '#8B68AB',   # muted purple
    'dTheta': '#85C1E9',   # light blue
    'dPhi':   '#F0A882',   # light orange
    'dPitch': '#85C99A',   # light green
    'dYaw':   '#B8A0CC',   # light purple
    'dRoll':  '#E09090',   # light red
    # Lighting conditions
    'light':  '#E8B84B',   # goldenrod
    'dark':   '#2C3E70',   # dark navy
    # Visual areas
    'V1':     '#1B9E77',   # dark teal
    'RL':     '#D95F02',   # dark orange
    'AM':     '#7570B3',   # medium purple
    'PM':     '#E7298A',   # deep pink
    'A':      '#1A5A34',   # dark green
    'AL':     '#E6AB02',   # mustard
    'LM':     '#A6761D',   # brown
    'P':      '#666666',   # dark grey
}

REGION_ORDER = ['V1', 'RL', 'AM', 'PM', 'A', 'AL', 'LM', 'P']
REGION_IDS   = {'V1': 5, 'RL': 2, 'AM': 3, 'PM': 4, 'A': 10, 'AL': 7, 'LM': 8, 'P': 9}
ID_TO_NAME   = {5: 'V1', 2: 'RL', 3: 'AM', 4: 'PM', 10: 'A', 7: 'AL', 8: 'LM', 9: 'P'}

VARIABLES = ['theta', 'phi', 'dTheta', 'dPhi', 'pitch', 'yaw', 'roll',
             'dPitch', 'dYaw', 'dRoll']

LABEL_MAP = {
    0: 'boundary', 1: 'outside',
    2: 'RL', 3: 'AM', 4: 'PM', 5: 'V1',
    7: 'AL', 8: 'LM', 9: 'P', 10: 'A'
}


def area_colors():
    return [COLORS[r] for r in REGION_ORDER]



def create_smoothed_map(x, y, values, shape=(1024, 1024), sigma=25):
    if len(values) < 4:
        return np.full(shape, np.nan)
    grid_y, grid_x = np.mgrid[0:shape[0], 0:shape[1]]
    points = np.column_stack((y, x))
    smoothed = griddata(points, values, (grid_y, grid_x), method='linear')
    V = smoothed.copy(); V[np.isnan(V)] = 0
    VV = gaussian_filter(V, sigma=sigma)
    W = 0 * smoothed.copy() + 1; W[np.isnan(smoothed)] = 0
    WW = gaussian_filter(W, sigma=sigma)
    return np.divide(VV, WW, out=np.full_like(VV, np.nan), where=WW != 0)


def add_scatter_col(ax, pos, vals, color='k', error_mode='sem'):
    vals = np.array(vals).flatten()
    vals = pd.to_numeric(vals, errors='coerce')
    vals = np.array(vals)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return
    ax.scatter(
        np.ones_like(vals) * pos + (np.random.rand(len(vals)) - 0.5) / 2,
        vals, s=2, c=color
    )
    ax.hlines(np.nanmean(vals), pos - .1, pos + .1, color='k')
    if error_mode == 'std':
        stderr = np.nanstd(vals)
    else:
        stderr = np.nanstd(vals) / np.sqrt(len(vals))
    ax.vlines(pos, np.nanmean(vals) - stderr, np.nanmean(vals) + stderr, color='k')


def plot_running_median(ax, x, y, n_bins=7, vertical=False, fb=True, color='k'):
    import scipy.stats
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) == 0:
        return np.nan
    x_use, y_use = x[mask], y[mask]
    bins = np.linspace(np.min(x_use), np.max(x_use), n_bins)
    bin_means, bin_edges, _ = scipy.stats.binned_statistic(x_use, y_use, np.nanmedian, bins=bins)
    bin_std, _, _  = scipy.stats.binned_statistic(x_use, y_use, np.nanstd,    bins=bins)
    hist, _, _     = scipy.stats.binned_statistic(x_use, y_use,
                                                  lambda v: np.sum(~np.isnan(v)), bins=bins)
    tuning_err = bin_std / np.sqrt(hist)
    centers = bin_edges[:-1] + np.median(np.diff(bins)) / 2
    if not vertical:
        ax.plot(centers, bin_means, '-', color=color)
        if fb:
            ax.fill_between(centers, bin_means - tuning_err, bin_means + tuning_err,
                            color=color, alpha=0.2)
    else:
        ax.plot(bin_means, centers, '-', color=color)
        if fb:
            ax.fill_betweenx(centers, bin_means - tuning_err, bin_means + tuning_err,
                             color=color, alpha=0.2)
    return np.nanmax(bin_means + tuning_err)


def savefig(fig, savedir, name):
    fig.savefig(os.path.join(savedir, f'{name}.svg'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_region_outlines(labeled_array, savedir):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)

    for region_id, region_name in LABEL_MAP.items():
        if region_id <= 1:
            continue
        mask = (labeled_array == region_id).astype(float)
        ax.contour(mask, levels=[0.5], colors='k', linewidths=1)
        y, x = np.where(mask)
        if len(x) > 0:
            color = COLORS.get(region_name, 'k')
            ax.text(np.mean(x), np.mean(y), region_name,
                    ha='center', va='center', fontsize=8, fontweight='bold', color=color)

    center_x = labeled_array.shape[1] // 2
    center_y = labeled_array.shape[0] // 2
    arrow_len = 300
    theta = np.pi / 4
    dx, dy = arrow_len * np.cos(theta), arrow_len * np.sin(theta)
    ax.arrow(center_x, center_y, dx, -dy, head_width=20, head_length=20,
             fc='tab:blue', ec='tab:blue', label='X-axis')
    ax.arrow(center_x, center_y, dy, dx, head_width=20, head_length=20,
             fc='tab:red', ec='tab:red', label='Y-axis')
    ax.invert_yaxis()
    ax.axis('off')
    ax.legend(loc='upper right')
    fig.tight_layout()
    savefig(fig, savedir, 'region_outlines')


def plot_behavior_corr_matrix(data, savedir):
    variables = ['theta', 'phi', 'dTheta', 'dPhi', 'pitch', 'roll', 'dPitch', 'dYaw', 'dRoll']

    if 'behavior_distributions' in data:
        dists = data['behavior_distributions']
        fig, axs = plt.subplots(2, 5, figsize=(8, 3), dpi=300)
        axs = axs.flatten()
        for i, var in enumerate(variables):
            if i >= len(axs):
                break
            vals = dists.get(var, np.array([]))
            if isinstance(vals, np.ndarray) and len(vals) > 0:
                axs[i].hist(vals[~np.isnan(vals)], bins=30,
                            color=COLORS.get(var, 'tab:blue'), alpha=0.7, density=True)
            axs[i].set_title(var)
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
        for j in range(len(variables), len(axs)):
            axs[j].axis('off')
        fig.suptitle('Behavior Variable Distributions (Aggregated)')
        fig.tight_layout()
        savefig(fig, savedir, 'behavior_var_distributions')


    if 'behavior_correlations' not in data:
        return
    all_corrs = np.array(data['behavior_correlations'])
    if all_corrs.ndim == 3:
        mean_corr = np.nanmean(all_corrs, axis=0)
        std_corr  = np.nanstd(all_corrs, axis=0)
    else:
        mean_corr = all_corrs
        std_corr  = np.zeros_like(all_corrs)

    n = len(variables)
    if mean_corr.shape[0] != n:

        return

    mask = np.triu(np.ones((n, n), dtype=bool), k=0)
    masked_mean = np.ma.array(mean_corr, mask=mask)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    ax.imshow(masked_mean, cmap=plt.cm.coolwarm, vmin=-1, vmax=1)
    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
    ax.set_xticklabels(variables, rotation=45, ha='right')
    ax.set_yticklabels(variables)
    for i in range(n):
        for j in range(n):
            if i > j:
                val = mean_corr[i, j]
                err = std_corr[i, j]
                if not np.isnan(val):
                    tc = 'white' if abs(val) > 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}\n±{err:.2f}',
                            ha='center', va='center', color=tc, fontsize=7)
    ax.set_title('Behavior Correlation Matrix (Mean ± Std)')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    savefig(fig, savedir, 'behavior_corr_matrix')


def plot_variable_summary(data, key, cond, labeled_array, savedir):

    MIN_RATE = 0.05

    dk = f'variable_summary_{key}_{cond}'
    if dk not in data:
        return

    df = pd.DataFrame(data[dk])
    # ensure numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    cond_name = 'Light' if cond == 'l' else 'Dark'

    metrics_to_plot = ['mod']
    if key not in ['dTheta', 'dPhi']:
        metrics_to_plot.append('peak')
    metrics_to_plot.append('imp')
    if key in ['dYaw', 'dPitch', 'dRoll']:
        metrics_to_plot = [m for m in metrics_to_plot if m != 'peak']

    # Bar plot of % modulated
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=300)
    pct_mod = []
    regions_present = []
    for rn in REGION_ORDER:
        rid = REGION_IDS[rn]
        # Use cells with sufficient firing rate
        region_cells = df[(df['region'] == rid) & (df['mean_rate'] > MIN_RATE)]
        if len(region_cells) > 0:
            # Calculate % modulated (CV-MI > 0.33)
            n_mod = np.sum(region_cells['mod'] > 0.33)
            pct = (n_mod / len(region_cells)) * 100
            pct_mod.append(pct)
            regions_present.append(rn)
    
    if len(regions_present) > 0:
        ax.bar(regions_present, pct_mod, color=[COLORS[r] for r in regions_present])
        ax.set_ylabel('% Modulated (CV-MI > 0.33)')
        ax.set_title(f'{key} % Modulated by Region ({cond_name})')
        savefig(fig, savedir, f'varsummary_{key}_{cond}_pct_modulated')

    

    for metric in metrics_to_plot:

        if metric == 'mod':
            cmap = cm.plasma
            norm = mcolors.Normalize(vmin=0, vmax=0.5)
            label_str = 'Modulation Index'
        elif metric == 'peak':
            if key in ['theta', 'phi', 'yaw', 'roll', 'pitch']:
                cmap = cm.coolwarm
                norm = mcolors.Normalize(vmin=-15, vmax=15)
                label_str = f'{key} Peak (deg)'
            else:
                limit = np.nanpercentile(np.abs(df['peak']), 95) if df['peak'].notna().any() else 1
                cmap = cm.plasma
                norm = mcolors.Normalize(vmin=-limit, vmax=limit)
                label_str = f'{key} Peak'
        elif metric == 'imp':
            cmap = cm.plasma
            norm = mcolors.Normalize(vmin=0, vmax=0.1)
            label_str = 'Variable Importance (Shuffle)'

        if metric == 'peak':
            rel = df[(df['rel'] == 1) & (df['mod'] > 0.33)]
        elif metric == 'imp':
            rel = df[(df['rel'] == 1) & (df['full_r2'] > 0.1)]
        elif metric == 'mod':
            rel = df[df['mean_rate'] > MIN_RATE]
        else:
            rel = df[df['rel'] == 1]

        if len(rel) == 0:
            continue

        if metric == 'mod':
            fig, ax = plt.subplots(1, 1, figsize=(5, 3.5), dpi=300)
            hist_bins = np.linspace(0, 0.75, 25)
            groups = []
            for i, rn in enumerate(REGION_ORDER):
                rid = REGION_IDS.get(rn, -1)
                region_vals = rel[rel['region'] == rid]['mod'].dropna().values
                groups.append(region_vals)
                if len(region_vals) > 0:
                    ax.hist(region_vals, bins=hist_bins, histtype='step', density=True,
                            color=COLORS[rn], label=f'{rn} (n={len(region_vals)})',
                            linewidth=1.5)
            ax.axvline(0.33, color='tab:grey', ls='--', alpha=0.56, linewidth=0.8)
            ax.axvline(0.5,  color='tab:grey', ls='--', alpha=0.56, linewidth=0.8)
            valid_groups = [g for g in groups if len(g) > 0]
            if len(valid_groups) > 1:
                try:
                    _, p_kw = kruskal(*valid_groups)
                    ax.text(0.05, 0.95, f'KW p={p_kw:.1e}', transform=ax.transAxes, fontsize=8)
                except ValueError:
                    pass
            ax.set_xlim([0, 0.75])
            ax.set_xlabel('CV-MI')
            ax.set_ylabel('Density')
            ax.set_title(f'{key} CV-MI by Region ({cond_name}) — all cells, rate>{MIN_RATE:.2f}')
            ax.legend(fontsize=7, frameon=False)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            fig.tight_layout()
            savefig(fig, savedir, f'varsummary_{key}_{cond}_cvmi_hist')

        if metric != 'mod' or True: # Force scatter for mod as well if requested
            
            fig, ax = plt.subplots(1, 1, figsize=(6, 3.5), dpi=300)
            groups = []
            for i, rn in enumerate(REGION_ORDER):
                rid = REGION_IDS.get(rn, -1)
                region_vals = rel[rel['region'] == rid][metric].dropna()
                groups.append(region_vals.values)
                if len(region_vals) > 0:
                    add_scatter_col(ax, i, region_vals.values, color=COLORS[rn], error_mode='std')
            valid_groups = [g[~np.isnan(g)] for g in groups if len(g[~np.isnan(g)]) > 0]
            if len(valid_groups) > 1:
                try:
                    _, p_kw = kruskal(*valid_groups)
                    ax.text(0.05, 0.95, f'KW p={p_kw:.1e}', transform=ax.transAxes, fontsize=8)
                except ValueError:
                    pass
            ax.set_xticks(np.arange(len(REGION_ORDER)), labels=REGION_ORDER)
            if key in ['pitch', 'roll'] and metric == 'peak':
                ax.set_ylim([-35, 35])
            elif metric == 'imp':
                ax.set_ylim([0, 0.25])
            elif metric == 'peak' and key in ['theta', 'phi', 'yaw', 'roll', 'pitch']:
                ax.set_ylim([-15, 15])
            ax.set_xlim([-.5, len(REGION_ORDER)-0.5])
            ax.set_ylabel(label_str)
            ax.set_title(f'{key} {metric} by Region ({cond_name})')
            fig.tight_layout()
            savefig(fig, savedir, f'varsummary_{key}_{cond}_{metric}_scatter')

        fig, axs = plt.subplots(2, 2, dpi=300, figsize=(8, 6))
        axs = axs.flatten()
        for i, rn in enumerate(REGION_ORDER):
            if i >= len(axs): break
            rid = REGION_IDS.get(rn, -1)
            region_mask = (labeled_array == rid)
            axs[i].imshow(region_mask, cmap='gray_r', alpha=0.4)
            axs[i].imshow(labeled_array == 0, cmap='binary', alpha=0.1)
            axs[i].contour(region_mask, levels=[0.5], colors='k', linestyles='--', linewidths=0.5)
            region_cells = rel[rel['region'] == rid]
            if len(region_cells) > 0:
                axs[i].scatter(region_cells['x'], region_cells['y'], s=2,
                               c=region_cells[metric], cmap=cmap, norm=norm)
            axs[i].set_title(rn)
            axs[i].axis('off')
        fig.suptitle(f'{key} {metric} Map by Region ({cond_name})')
        fig.tight_layout()
        savefig(fig, savedir, f'varsummary_{key}_{cond}_{metric}_map')

        fig, axs = plt.subplots(2, 2, dpi=300, figsize=(8, 6))
        axs = axs.flatten()
        for i, rn in enumerate(REGION_ORDER):
            if i >= len(axs): break
            rid = REGION_IDS.get(rn, -1)
            region_mask = (labeled_array == rid)
            axs[i].imshow(labeled_array == 0, cmap='binary', alpha=0.1)
            axs[i].contour(region_mask, levels=[0.5], colors='k', linestyles='--', linewidths=0.5)
            region_cells = rel[rel['region'] == rid]
            if len(region_cells) > 0:
                smoothed = create_smoothed_map(
                    region_cells['x'].values, region_cells['y'].values,
                    region_cells[metric].values, shape=labeled_array.shape, sigma=50)
                smoothed[~region_mask] = np.nan
                axs[i].imshow(smoothed, cmap=cmap, norm=norm)
            axs[i].set_title(rn)
            axs[i].axis('off')
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs,
                     label=label_str, fraction=0.05, shrink=0.6)
        fig.suptitle(f'{key} {metric} Smoothed Map by Region ({cond_name})')
        fig.tight_layout()
        savefig(fig, savedir, f'varsummary_{key}_{cond}_{metric}_smoothed')


        if metric in ['mod', 'peak']:
            fig_rot, (ax_rot_x, ax_rot_y) = plt.subplots(1, 2, figsize=(5, 2.5), dpi=300)
            theta_rot = np.pi / 4
            cos_t, sin_t = np.cos(theta_rot), np.sin(theta_rot)
            rel_rot = rel.copy()
            rel_rot['x_rot'] = rel['x'] * cos_t - rel['y'] * sin_t
            rel_rot['y_rot'] = rel['x'] * sin_t + rel['y'] * cos_t
            max_y_val = 0
            for i, rn in enumerate(REGION_ORDER):
                rid = REGION_IDS.get(rn, -1)
                region_cells = rel_rot[rel_rot['region'] == rid]
                if len(region_cells) < 5:
                    continue
                x_norm = (region_cells['x_rot'] - region_cells['x_rot'].min()) / \
                         (region_cells['x_rot'].max() - region_cells['x_rot'].min() + 1e-9)
                y_norm = (region_cells['y_rot'] - region_cells['y_rot'].min()) / \
                         (region_cells['y_rot'].max() - region_cells['y_rot'].min() + 1e-9)
                my1 = plot_running_median(ax_rot_x, x_norm.values,
                                         region_cells[metric].values, n_bins=5, color=COLORS[rn])
                ax_rot_x.plot([], [], color=COLORS[rn], label=rn)
                my2 = plot_running_median(ax_rot_y, y_norm.values,
                                         region_cells[metric].values, n_bins=5, color=COLORS[rn])
                ax_rot_y.plot([], [], color=COLORS[rn], label=rn)
                max_y_val = np.nanmax([max_y_val, my1 if my1 is not None else 0,
                                       my2 if my2 is not None else 0])
            if metric == 'mod':
                ax_rot_x.set_ylim(bottom=0.0, top=max(max_y_val * 1.1, 0.01))
                ax_rot_y.set_ylim(bottom=0.0, top=max(max_y_val * 1.1, 0.01))
                ylabel = 'Modulation Index'
            else:
                lim = 35 if key in ['pitch', 'roll'] else 15
                ax_rot_x.set_ylim([-lim, lim]); ax_rot_y.set_ylim([-lim, lim])
                ylabel = 'Peak Position'
            for ax, title in [(ax_rot_x, 'Along Rotated X-axis (45° CCW)'),
                              (ax_rot_y, 'Along Rotated Y-axis (45° CCW)')]:
                ax.set_title(title); ax.set_xlabel('Position along rotated axis')
                ax.set_ylabel(ylabel); ax.legend(fontsize=7, frameon=False)
            fig_rot.suptitle(f'{label_str} along Rotated Axes — {key} ({cond_name})')
            fig_rot.tight_layout()
            savefig(fig_rot, savedir, f'varsummary_{key}_{cond}_{metric}_rotaxes')


def plot_signal_noise_correlations(data, key, cond, savedir):
    dk = f'signal_noise_corr_{key}_{cond}'
    if dk not in data:
        return
    d = data[dk]
    pooled_sig     = np.array(d['pooled_sig'],    dtype=float)
    pooled_noise   = np.array(d['pooled_noise'],  dtype=float)
    pooled_regions = np.array(d['pooled_regions'], dtype=int)
    raw_fov        = d.get('pooled_fov', None)
    pooled_fov     = np.array(raw_fov, dtype=int) if raw_fov is not None else None

    cond_name = 'Light' if cond == 'l' else 'Dark'
    valid = np.isfinite(pooled_sig) & np.isfinite(pooled_noise)

    region_ids   = [5, 2, 3, 4]
    region_names = REGION_ORDER


    fig, axs = plt.subplots(2, 3, figsize=(6, 4), dpi=300)
    axs = axs.flatten()
    ds = max(1, int(np.sum(valid)) // 3000)
    v = valid
    axs[0].scatter(pooled_sig[v][::ds], pooled_noise[v][::ds], s=2, c='k', alpha=0.3)
    axs[0].set_title('All Pairs')
    axs[0].set_xlabel('Signal Corr'); axs[0].set_ylabel('Noise Corr (residuals)')
    axs[0].set_xlim([-1, 1]); axs[0].set_ylim([-1, 1])
    for i, (rid, rname) in enumerate(zip(region_ids, region_names)):
        if i >= 4: break # Only plot first 4 regions in grid
        mask = (pooled_regions[:, 0] == rid) & (pooled_regions[:, 1] == rid)
        ax = axs[i + 1]
        mask_valid = mask & valid
        if np.sum(mask_valid) > 1:
            ds_r = max(1, int(np.sum(mask_valid)) // 2000)
            ax.scatter(pooled_sig[mask_valid][::ds_r], pooled_noise[mask_valid][::ds_r],
                       s=2, c=COLORS[rname], alpha=0.3)
            r_val = np.corrcoef(pooled_sig[mask_valid], pooled_noise[mask_valid])[0, 1]
            ax.text(0.05, 0.9, f'r={r_val:.2f}', transform=ax.transAxes, fontsize=8)
        ax.set_title(f'{rname} Pairs')
        ax.set_xlabel('Signal Corr'); ax.set_ylabel('Noise Corr')
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1])
    axs[5].axis('off')
    fig.suptitle(f'Signal vs Noise Corr: {key} ({cond_name})')
    fig.tight_layout()
    savefig(fig, savedir, f'signoise_{key}_{cond}_scatter')


    sig_bins = np.linspace(-1, 1, 11)
    sig_bin_centers = 0.5 * (sig_bins[:-1] + sig_bins[1:])

    def _plot_binned(ax, sig_arr, noise_arr, color, label=None,
                     lw=1.5, alpha=1.0, show_fill=True, min_n=5):
        mean_noise, sem_noise = [], []
        for lo, hi in zip(sig_bins[:-1], sig_bins[1:]):
            in_bin = (sig_arr >= lo) & (sig_arr < hi)
            n_in = np.sum(in_bin)
            if n_in < min_n:
                mean_noise.append(np.nan); sem_noise.append(np.nan)
            else:
                mean_noise.append(np.nanmean(noise_arr[in_bin]))
                sem_noise.append(np.nanstd(noise_arr[in_bin]) / np.sqrt(n_in))
        mean_noise = np.array(mean_noise); sem_noise = np.array(sem_noise)
        ax.plot(sig_bin_centers, mean_noise, '-o', markersize=3, color=color,
                label=label, lw=lw, alpha=alpha)
        if show_fill:
            ax.fill_between(sig_bin_centers, mean_noise - sem_noise, mean_noise + sem_noise,
                            color=color, alpha=0.2)
        ax.axhline(0, color='k', lw=0.5, ls='--')
        ax.axvline(0, color='k', lw=0.5, ls='--')

    fig2, axs2 = plt.subplots(1, 5, figsize=(10, 2.5), dpi=300, sharey=True)
    v_sig   = pooled_sig[valid]
    v_noise = pooled_noise[valid]
    _plot_binned(axs2[0], v_sig, v_noise, 'k', 'All')
    axs2[0].set_title('All Pairs'); axs2[0].set_xlabel('Signal Corr')
    axs2[0].set_ylabel('Mean Noise Corr')
    for i, (rid, rname) in enumerate(zip(region_ids, region_names[:4])):
        mask = (pooled_regions[:, 0] == rid) & (pooled_regions[:, 1] == rid) & valid
        ax = axs2[i + 1]
        # Per-FOV lines
        if pooled_fov is not None:
            for fid in np.unique(pooled_fov[mask]):
                fov_mask = mask & (pooled_fov == fid)
                if np.sum(fov_mask) > 10:
                    _plot_binned(ax, pooled_sig[fov_mask], pooled_noise[fov_mask],
                                 color=COLORS[rname], lw=1.0, alpha=0.6,
                                 show_fill=False, min_n=3)
        # Mean line on top
        if np.sum(mask) > 10:
            _plot_binned(ax, pooled_sig[mask], pooled_noise[mask],
                         color=COLORS[rname], label=rname, lw=2.0, alpha=1.0,
                         show_fill=True)
        ax.set_title(f'{rname} Pairs')
        ax.set_xlabel('Signal Corr'); ax.set_xlim([-1, 1])
    fig2.suptitle(f'Noise Corr vs Signal Corr (binned): {key} ({cond_name})')
    fig2.tight_layout()
    savefig(fig2, savedir, f'signoise_{key}_{cond}_binned')


def plot_all_variable_importance(data, savedir):
    if 'all_variable_importance' not in data:
        return
    df = pd.DataFrame(data['all_variable_importance'])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    for rn in REGION_ORDER:
        rid = REGION_IDS.get(rn, -1)
        region_df = df[df['region'] == rid]
        fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
        for v_idx, var in enumerate(VARIABLES):
            if var in region_df.columns:
                vals = region_df[var].dropna()
                add_scatter_col(ax, v_idx, vals, color=COLORS.get(var, 'grey'))
        ax.set_xticks(range(len(VARIABLES)))
        ax.set_xticklabels(VARIABLES, rotation=45, ha='right')
        ax.set_ylabel('Importance (Shuffle)')
        ax.set_title(f'Variable Importance in {rn}')
        ax.set_ylim([-0.05, 0.2])
        fig.tight_layout()
        savefig(fig, savedir, f'all_var_importance_{rn}')


def plot_all_model_performance(data, savedir):
    if 'all_model_performance' not in data:
        return
    df = pd.DataFrame(data['all_model_performance'])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    models = ['full', 'position_only', 'velocity_only', 'head_only', 'eyes_only']

    for rn in REGION_ORDER:
        rid = REGION_IDS.get(rn, -1)
        region_df = df[df['region'] == rid]

        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        for m_idx, model in enumerate(models):
            col = f'{model}_r2'
            if col in region_df.columns:
                vals = region_df[col].dropna().values[::10]
                add_scatter_col(ax, m_idx, vals, color='lightgrey')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('R²'); ax.set_title(f'Model Performance (R²) in {rn}')
        ax.set_ylim([-0.1, 0.4])
        fig.tight_layout()
        savefig(fig, savedir, f'all_model_r2_{rn}')


        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        for m_idx, model in enumerate(models):
            col = f'{model}_corr'
            if col in region_df.columns:
                vals = region_df[col].dropna().values[::10]
                add_scatter_col(ax, m_idx, vals, color='lightgrey')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Correlation'); ax.set_title(f'Model Performance (Corr) in {rn}')
        ax.set_ylim([-0.1, 0.6])
        fig.tight_layout()
        savefig(fig, savedir, f'all_model_corr_{rn}')


def plot_model_performance_maps(data, labeled_array, savedir):
    if 'model_performance_maps' not in data:
        return
    maps_data = data['model_performance_maps']

    for key_name, d in maps_data.items():
        df = pd.DataFrame(d)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'r2' in key_name:
            norm = mcolors.Normalize(vmin=-0.2, vmax=0.3)
            label_str = f'{key_name.replace("_", " ").title()}'
        else:
            norm = mcolors.Normalize(vmin=-0.2, vmax=0.5)
            label_str = f'{key_name.replace("_", " ").title()}'
        cmap = cm.plasma


        # Use pre-saved region column if available and non-trivial,
        # otherwise fall back to spatial lookup in labeled_array.
        if 'region' not in df.columns or df['region'].eq(0).all():
            region_vals = []
            for xi, yi in zip(df['x'], df['y']):
                xi_i = int(np.clip(xi, 0, labeled_array.shape[1] - 1))
                yi_i = int(np.clip(yi, 0, labeled_array.shape[0] - 1))
                region_vals.append(labeled_array[yi_i, xi_i])
            df['region'] = region_vals


        fig, axs = plt.subplots(2, 2, dpi=300, figsize=(8, 8))
        axs = axs.flatten()
        for i, rn in enumerate(REGION_ORDER):
            if i >= len(axs): break
            rid = REGION_IDS.get(rn, -1)
            region_mask = (labeled_array == rid)
            axs[i].imshow(region_mask, cmap='gray_r', alpha=0.4)
            axs[i].imshow(labeled_array == 0, cmap='binary', alpha=0.1)
            axs[i].contour(region_mask, levels=[0.5], colors='k', linestyles='--', linewidths=0.5)
            rc = df[df['region'] == rid]
            if len(rc) > 0:
                axs[i].scatter(rc['x'], rc['y'], s=2, c=rc['val'], cmap=cmap, norm=norm)
            axs[i].set_title(rn); axs[i].axis('off')
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs,
                     label=label_str, fraction=0.05, shrink=0.6)
        fig.suptitle(f'{label_str} Map by Region')
        fig.tight_layout()
        savefig(fig, savedir, f'model_map_{key_name}')


        fig, axs = plt.subplots(2, 2, dpi=300, figsize=(8, 6))
        axs = axs.flatten()
        for i, rn in enumerate(REGION_ORDER):
            if i >= len(axs): break
            rid = REGION_IDS.get(rn, -1)
            region_mask = (labeled_array == rid)
            axs[i].imshow(labeled_array == 0, cmap='binary', alpha=0.1)
            axs[i].contour(region_mask, levels=[0.5], colors='k', linestyles='--', linewidths=0.5)
            rc = df[df['region'] == rid]
            if len(rc) > 0:
                smoothed = create_smoothed_map(rc['x'].values, rc['y'].values,
                                               rc['val'].values, shape=labeled_array.shape)
                smoothed[~region_mask] = np.nan
                axs[i].imshow(smoothed, cmap=cmap, norm=norm)
            axs[i].set_title(rn); axs[i].axis('off')
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs,
                     label=label_str, fraction=0.05, shrink=0.6)
        fig.suptitle(f'{label_str} Smoothed Map by Region')
        fig.tight_layout()
        savefig(fig, savedir, f'model_map_{key_name}_smoothed')


def plot_sorted_tuning_curves(data, key, cond, savedir):
    dk = f'sorted_tuning_curves_{cond}'
    if dk not in data or key not in data[dk]:
        return
    d = data[dk][key]
    mods   = np.array(d['mods'],   dtype=float)
    tuning = np.array(d['tuning'], dtype=float)
    errs   = np.array(d['errs'],   dtype=float)
    bins   = np.array(d['bins'],   dtype=float)

    n_cells = min(64, len(mods))
    cond_name = 'Light' if cond == 'l' else 'Dark'

    fig, axs = plt.subplots(8, 8, figsize=(16, 16), dpi=300)
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        if i < n_cells:
            if len(bins) == tuning.shape[1] + 1:
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
            else:
                bin_centers = bins
            ax.plot(bin_centers, tuning[i], 'k-')
            ax.fill_between(bin_centers,
                            tuning[i] - errs[i], tuning[i] + errs[i],
                            color=COLORS.get(key, 'k'), alpha=0.3)
            ax.set_title(f'MI={mods[i]:.2f}', fontsize=6)
            ax.tick_params(labelsize=6)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        else:
            ax.axis('off')
    fig.suptitle(f'Top {n_cells} Modulated Cells — {key} ({cond_name})', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    savefig(fig, savedir, f'sorted_tuning_{key}_{cond}')


def plot_modulation_summary(data, cond, savedir):
    dk = f'modulation_summary_{cond}'
    if dk not in data:
        return
    results = data[dk]
    cond_name = 'Light' if cond == 'l' else 'Dark'

    for metric in ['tuning', 'importance']:
        if metric not in results:
            continue
        fig, axs = plt.subplots(2, 5, figsize=(6.5, 2.5), dpi=300)
        axs = axs.flatten()
        for i, var in enumerate(VARIABLES):
            ax = axs[i]
            ax.axhline(50, color='lightgrey', linestyle='--', alpha=0.5)
            for col_idx, rn in enumerate(REGION_ORDER):
                if var in results[metric] and rn in results[metric][var] and rn in ID_TO_NAME.values():
                    vals = results[metric][var][rn]
                    if isinstance(vals, dict):
                        vals = list(vals.values())
                    elif isinstance(vals, np.ndarray):
                        vals = vals.tolist()
                    if vals:
                        add_scatter_col(ax, col_idx, vals, color=COLORS[rn])
            ax.set_xticks(range(len(REGION_ORDER)))
            ax.set_xticklabels(REGION_ORDER, rotation=90)
            ax.set_title(var)
            if metric == 'tuning':
                ax.set_ylim([0, 50])
            else:
                ax.set_ylim([0, 100.])
            if i % 5 == 0:
                ax.set_ylabel('% Modulated Cells')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.suptitle(f'Percentage of Modulated Cells by {metric.capitalize()} ({cond_name})')
        fig.tight_layout()
        savefig(fig, savedir, f'modulation_summary_{cond}_{metric}')


def plot_modulation_histograms(data, cond, savedir):
    dk = f'modulation_histograms_{cond}'
    if dk not in data:
        return
    results = data[dk]
    cond_name = 'Light' if cond == 'l' else 'Dark'

    for metric in ['mod', 'imp']:
        if metric not in results:
            continue
        metric_label = 'CV-MI' if metric == 'mod' else 'Reliability Score'

        all_values = []
        for v in VARIABLES:
            if v not in results[metric]:
                continue
            for rn in REGION_ORDER:
                if rn in results[metric][v]:
                    vals = results[metric][v][rn]
                    if isinstance(vals, dict):
                        all_values.extend(list(vals.values()))
                    elif isinstance(vals, np.ndarray):
                        all_values.extend(vals.tolist())
                    else:
                        all_values.extend(vals)

        if not all_values:
            continue

        all_values = np.array(all_values, dtype=float)
        all_values = all_values[~np.isnan(all_values)]
        if len(all_values) == 0:
            continue

        xlim = (0, 0.5); bins = np.linspace(0, 0.5, 21)

        fig, axs = plt.subplots(2, 5, figsize=(7, 3.5), dpi=300)
        axs = axs.flatten()
        for i, var in enumerate(VARIABLES):
            ax = axs[i]
            if var not in results[metric]:
                ax.axis('off'); continue
            for rn in REGION_ORDER:
                if rn in results[metric][var] and rn in ID_TO_NAME.values():
                    vals = results[metric][var][rn]
                    if isinstance(vals, dict):
                        vals = list(vals.values())
                    vals = np.array(vals, dtype=float)
                    vals = vals[~np.isnan(vals)]
                    if len(vals) > 0:
                        ax.hist(vals, bins=bins, density=True, histtype='step',
                                color=COLORS[rn], label=rn, linewidth=1.5)
            ref_lines = [0.1, 0.33] if metric == 'mod' else [0.1]
            for line_val in ref_lines:
                ax.axvline(line_val, color='k', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.set_title(var); ax.set_xlim(xlim)
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
        savefig(fig, savedir, f'modulation_hists_{cond}_{metric}')


def plot_lightdark_modulation_histograms(data, savedir):
    if 'lightdark_modulation' not in data:
        return
    ld = data['lightdark_modulation']
    hist_bins = np.linspace(0, 1.0, 26)

    for var in VARIABLES:
        if var not in ld:
            continue
        d = ld[var]
        dark_rel_mod_d  = np.array(d['dark_rel_mod_d'],  dtype=float)
        dark_rel_mod_l  = np.array(d['dark_rel_mod_l'],  dtype=float)
        light_rel_mod_d = np.array(d['light_rel_mod_d'], dtype=float)
        light_rel_mod_l = np.array(d['light_rel_mod_l'], dtype=float)

        fig, axs = plt.subplots(2, 2, figsize=(7, 5), dpi=300)
        axs[0, 0].hist(dark_rel_mod_d[np.isfinite(dark_rel_mod_d)], bins=hist_bins, density=True,
                       color=COLORS['dark'], alpha=0.7)
        axs[0, 0].set_title(f'Dark-reliable → CV-MI in Dark  (n={len(dark_rel_mod_d)})')
        axs[0, 0].set_ylabel('Density')
        axs[0, 1].hist(dark_rel_mod_l[np.isfinite(dark_rel_mod_l)], bins=hist_bins, density=True,
                       color=COLORS['light'], alpha=0.7)
        axs[0, 1].set_title(f'Dark-reliable → CV-MI in Light  (n={len(dark_rel_mod_l)})')
        axs[1, 0].hist(light_rel_mod_d[np.isfinite(light_rel_mod_d)], bins=hist_bins, density=True,
                       color=COLORS['dark'], alpha=0.7)
        axs[1, 0].set_title(f'Light-reliable → CV-MI in Dark  (n={len(light_rel_mod_d)})')
        axs[1, 0].set_ylabel('Density')
        axs[1, 1].hist(light_rel_mod_l[np.isfinite(light_rel_mod_l)], bins=hist_bins, density=True,
                       color=COLORS['light'], alpha=0.7)
        axs[1, 1].set_title(f'Light-reliable → CV-MI in Light  (n={len(light_rel_mod_l)})')
        for ax in axs.flatten():
            ax.axvline(0.1, color='k', ls='--', alpha=0.5, lw=0.8)
            ax.set_xlim([0, 1.0]); ax.set_xlabel('CV-MI')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.suptitle(f'{var}: Cross-condition CV-MI for Condition-Reliable Cells')
        fig.tight_layout()
        savefig(fig, savedir, f'lightdark_modulation_{var}')


def plot_position_occupancy(data, savedir):
    if 'position_occupancy_data' not in data:
        return
    occ = data['position_occupancy_data']

    indices = sorted(occ.keys(), key=lambda k: int(k))
    recording_data = [occ[i] for i in indices]

    labels = data.get('position_occupancy_recordings', [])
    if isinstance(labels, dict):
        labels = [str(labels[k]) for k in sorted(labels.keys(), key=lambda k: int(k))]
    elif isinstance(labels, np.ndarray):
        labels = [str(l) for l in labels]
    else:
        labels = list(labels)

    n_recs = len(recording_data)
    if n_recs == 0:
        return
    ncols = min(4, n_recs)
    nrows = int(np.ceil(n_recs / ncols))

    for var_pair, xlabel, ylabel, fig_title, name in [
        ('pitch_roll', 'Roll (deg)', 'Pitch (deg)',
         'Head Pitch vs Roll Occupancy (1/100 sampled)', 'position_occ_pitch_roll'),
        ('theta_phi',  'Theta (deg)', 'Phi (deg)',
         'Eye Theta vs Phi Occupancy (1/100 sampled)', 'position_occ_theta_phi'),
    ]:
        fig, axs_grid = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows), dpi=150)
        if nrows == 1 and ncols == 1:
            axs_grid = np.array([[axs_grid]])
        elif nrows == 1:
            axs_grid = axs_grid[np.newaxis, :]
        elif ncols == 1:
            axs_grid = axs_grid[:, np.newaxis]

        for idx, rec in enumerate(recording_data):
            row, col = idx // ncols, idx % ncols
            ax = axs_grid[row, col]
            if var_pair == 'pitch_roll':
                x = np.array(rec['roll'],  dtype=float)
                y = np.array(rec['pitch'], dtype=float)
            else:
                x = np.array(rec['theta'], dtype=float)
                y = np.array(rec['phi'],   dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[valid], y[valid], s=1, c='k', alpha=0.25, rasterized=True)
            label = labels[idx] if idx < len(labels) else str(idx)
            ax.set_title(label, fontsize=7)
            ax.set_xlabel(xlabel, fontsize=7)
            ax.set_ylabel(ylabel, fontsize=7)
            ax.tick_params(labelsize=6)

        for idx in range(n_recs, nrows * ncols):
            axs_grid[idx // ncols, idx % ncols].axis('off')

        fig.suptitle(fig_title)
        fig.tight_layout()
        savefig(fig, savedir, name)


# ---------------------------------------------------------------------------
# Boundary-tuning (EBC / RBC) summary figures
# ---------------------------------------------------------------------------

def _extract_bt_criteria_array(criteria, key, N_cells):
    """Pull a scalar metric out of a criteria dict into a length-N_cells array."""
    out = np.full(N_cells, np.nan)
    for c in range(N_cells):
        ckey = f'cell_{c:03d}'
        if ckey in criteria and key in criteria[ckey]:
            v = criteria[ckey][key]
            try:
                out[c] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def _bt_peak_dist_from_maps(smoothed_maps, dist_bin_cents):
    """Return the distance-bin centre of the peak for each cell's rate map."""
    N_cells = smoothed_maps.shape[0]
    out = np.full(N_cells, np.nan)
    for c in range(N_cells):
        rm = smoothed_maps[c]
        col_max = np.nanmax(rm, axis=0)   # (N_dist,)
        if np.all(np.isnan(col_max)):
            continue
        out[c] = float(dist_bin_cents[np.nanargmax(col_max)])
    return out


def _bt_peak_angle_from_maps(smoothed_maps, angle_rad):
    """Return the preferred angle (radians) from the peak of each cell's rate map."""
    N_cells = smoothed_maps.shape[0]
    out = np.full(N_cells, np.nan)
    for c in range(N_cells):
        rm = smoothed_maps[c]
        row_max = np.nanmax(rm, axis=1)   # (N_ang,)
        if np.all(np.isnan(row_max)):
            continue
        out[c] = float(angle_rad[np.nanargmax(row_max)])
    return out


def build_bt_cell_df(pooled_data, bt_results_by_pos, labeled_array,
                     animal_dirs=None):
    """Build a per-cell DataFrame combining BT metrics with brain positions.

    Parameters
    ----------
    pooled_data : dict
        Output of make_pooled_dataset() (contains per-animal transform dicts).
    bt_results_by_pos : dict
        Nested dict ``{animal_dir: {poskey: bt_dict}}`` where each ``bt_dict``
        is the dict loaded from a ``save_results_combined()`` HDF5 file.
    labeled_array : np.ndarray
        Integer region label array in reference VFS space.
    animal_dirs : list of str, optional
        Animals to include.  Defaults to all keys in bt_results_by_pos.

    Returns
    -------
    df : pd.DataFrame
        One row per cell, columns:
        animal, pos, cell_idx, x, y, region,
        is_EBC, is_RBC, is_IEBC, is_IRBC, is_either, is_both,
        ebc_mrl, ebc_mra, ebc_corr, ebc_mrl_thresh, ebc_peak_ang, ebc_peak_dist,
        rbc_mrl, rbc_mra, rbc_corr, rbc_mrl_thresh, rbc_peak_ang, rbc_peak_dist.
    """
    if animal_dirs is None:
        animal_dirs = list(bt_results_by_pos.keys())

    rows = []
    for animal in animal_dirs:
        if animal not in bt_results_by_pos:
            continue
        animal_bt = bt_results_by_pos[animal]

        animal_pooled = pooled_data.get(animal, {})
        transform_data = animal_pooled.get('transform', {})

        for poskey, bt in animal_bt.items():
            transform = transform_data.get(poskey)
            if transform is None:
                continue

            params   = bt.get('params', {})
            angle_rad     = np.array(params.get('angle_rad', []))
            dist_bin_cents = np.array(params.get('dist_bin_cents', []))

            ebc_block = bt.get('ebc', {})
            rbc_block = bt.get('rbc', {})
            cls_block = bt.get('classification', {})

            is_EBC   = np.array(cls_block.get('is_EBC',   []), dtype=bool)
            is_RBC   = np.array(cls_block.get('is_RBC',   []), dtype=bool)
            is_either = np.array(cls_block.get('is_either', []), dtype=bool)
            is_both   = np.array(cls_block.get('is_both',   []), dtype=bool)
            is_IEBC  = np.array(ebc_block.get('is_IEBC', np.zeros_like(is_EBC)), dtype=bool)
            is_IRBC  = np.array(rbc_block.get('is_IRBC', np.zeros_like(is_RBC)), dtype=bool)

            N_cells = len(is_EBC)
            if N_cells == 0:
                continue

            ebc_crit = ebc_block.get('criteria', {})
            rbc_crit = rbc_block.get('criteria', {})

            ebc_mrl    = _extract_bt_criteria_array(ebc_crit, 'mean_resultant_length', N_cells)
            ebc_mra    = _extract_bt_criteria_array(ebc_crit, 'mean_resultant_angle',  N_cells)
            ebc_corr   = _extract_bt_criteria_array(ebc_crit, 'corr_coeff',            N_cells)
            ebc_thresh = _extract_bt_criteria_array(ebc_crit, 'mrl_99_pctl',           N_cells)
            rbc_mrl    = _extract_bt_criteria_array(rbc_crit, 'mean_resultant_length', N_cells)
            rbc_mra    = _extract_bt_criteria_array(rbc_crit, 'mean_resultant_angle',  N_cells)
            rbc_corr   = _extract_bt_criteria_array(rbc_crit, 'corr_coeff',            N_cells)
            rbc_thresh = _extract_bt_criteria_array(rbc_crit, 'mrl_99_pctl',           N_cells)

            ebc_smaps = np.array(ebc_block.get('smoothed_rate_maps', np.zeros((N_cells, 1, 1))))
            rbc_smaps = np.array(rbc_block.get('smoothed_rate_maps', np.zeros((N_cells, 1, 1))))

            if len(angle_rad) > 0 and len(dist_bin_cents) > 0:
                ebc_peak_ang  = _bt_peak_angle_from_maps(ebc_smaps, angle_rad)
                ebc_peak_dist = _bt_peak_dist_from_maps(ebc_smaps, dist_bin_cents)
                rbc_peak_ang  = _bt_peak_angle_from_maps(rbc_smaps, angle_rad)
                rbc_peak_dist = _bt_peak_dist_from_maps(rbc_smaps, dist_bin_cents)
            else:
                ebc_peak_ang = ebc_peak_dist = rbc_peak_ang = rbc_peak_dist = np.full(N_cells, np.nan)

            # Brain position: transform has shape (N_cells, ≥4); cols 2,3 = ref VFS x,y
            n_tr = transform.shape[0]
            xs = transform[:, 2] if n_tr >= N_cells else np.full(N_cells, np.nan)
            ys = transform[:, 3] if n_tr >= N_cells else np.full(N_cells, np.nan)

            # Region lookup
            h, w = labeled_array.shape
            regions = np.zeros(N_cells, dtype=int)
            for c in range(N_cells):
                xi, yi = int(np.clip(xs[c], 0, w - 1)), int(np.clip(ys[c], 0, h - 1))
                regions[c] = labeled_array[yi, xi]

            for c in range(N_cells):
                rows.append({
                    'animal': animal,
                    'pos': poskey,
                    'cell_idx': c,
                    'x': xs[c],
                    'y': ys[c],
                    'region': regions[c],
                    'is_EBC':   int(is_EBC[c])   if c < len(is_EBC)   else 0,
                    'is_RBC':   int(is_RBC[c])   if c < len(is_RBC)   else 0,
                    'is_IEBC':  int(is_IEBC[c])  if c < len(is_IEBC)  else 0,
                    'is_IRBC':  int(is_IRBC[c])  if c < len(is_IRBC)  else 0,
                    'is_either':int(is_either[c]) if c < len(is_either) else 0,
                    'is_both':  int(is_both[c])  if c < len(is_both)  else 0,
                    'ebc_mrl':       ebc_mrl[c],
                    'ebc_mra':       ebc_mra[c],
                    'ebc_corr':      ebc_corr[c],
                    'ebc_mrl_thresh':ebc_thresh[c],
                    'ebc_peak_ang':  ebc_peak_ang[c],
                    'ebc_peak_dist': ebc_peak_dist[c],
                    'rbc_mrl':       rbc_mrl[c],
                    'rbc_mra':       rbc_mra[c],
                    'rbc_corr':      rbc_corr[c],
                    'rbc_mrl_thresh':rbc_thresh[c],
                    'rbc_peak_ang':  rbc_peak_ang[c],
                    'rbc_peak_dist': rbc_peak_dist[c],
                })

    return pd.DataFrame(rows)


def plot_ebc_rbc_proportions(df, labeled_array, savedir):
    """Stacked bar + individual-animal dot plot of EBC/RBC fractions by area.

    Saves
    -----
    bt_proportions_stacked.png  — stacked bar (EBC-only / RBC-only / both / neither)
    bt_proportions_dots.png     — per-animal dot plot of EBC% and RBC%
    """
    regions_in_data = set(df['region'].unique())
    region_order = [r for r in REGION_ORDER if REGION_IDS[r] in regions_in_data]
    if not region_order:
        return

    # ---- Figure 1: stacked bar chart ----------------------------------------
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=300)
    x = np.arange(len(region_order))
    bar_w = 0.55
    cat_colors = {'EBC only': '#1a7f37', 'RBC only': '#1a5fa8',
                  'Both': '#9B59B6', 'Neither': '#cccccc'}

    fracs = {cat: [] for cat in cat_colors}
    ns = []
    for rn in region_order:
        rid = REGION_IDS[rn]
        sub = df[df['region'] == rid]
        n = len(sub)
        ns.append(n)
        if n == 0:
            for cat in cat_colors:
                fracs[cat].append(0.0)
            continue
        fracs['EBC only'].append(np.mean((sub['is_EBC'] == 1) & (sub['is_RBC'] == 0)) * 100)
        fracs['RBC only'].append(np.mean((sub['is_EBC'] == 0) & (sub['is_RBC'] == 1)) * 100)
        fracs['Both'].append(np.mean(sub['is_both'] == 1) * 100)
        fracs['Neither'].append(np.mean((sub['is_EBC'] == 0) & (sub['is_RBC'] == 0)) * 100)

    bottoms = np.zeros(len(region_order))
    for cat, color in cat_colors.items():
        vals = np.array(fracs[cat])
        ax.bar(x, vals, bar_w, bottom=bottoms, color=color, label=cat)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f'{r}\n(n={ns[i]})' for i, r in enumerate(region_order)])
    ax.set_ylabel('% of cells')
    ax.set_ylim([0, 100])
    ax.legend(fontsize=7, frameon=False, loc='upper right')
    ax.set_title('EBC / RBC proportion by visual area')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    savefig(fig, savedir, 'bt_proportions_stacked')

    # ---- Figure 2: per-animal dot plot ---------------------------------------
    fig2, axs2 = plt.subplots(1, 2, figsize=(6, 3), dpi=300, sharey=False)
    for ax_i, (cell_type, col) in enumerate([('EBC', '#1a7f37'), ('RBC', '#1a5fa8')]):
        ax = axs2[ax_i]
        is_col = 'is_EBC' if cell_type == 'EBC' else 'is_RBC'
        for xi, rn in enumerate(region_order):
            rid = REGION_IDS[rn]
            per_animal = []
            for animal in df['animal'].unique():
                sub = df[(df['region'] == rid) & (df['animal'] == animal)]
                if len(sub) >= 5:
                    per_animal.append(np.mean(sub[is_col] == 1) * 100)
            if per_animal:
                jitter = (np.random.rand(len(per_animal)) - 0.5) * 0.3
                ax.scatter(np.ones(len(per_animal)) * xi + jitter, per_animal,
                           s=20, color=col, alpha=0.7, zorder=3)
                ax.hlines(np.mean(per_animal), xi - 0.2, xi + 0.2, color='k', lw=1.5)
        ax.set_xticks(range(len(region_order)))
        ax.set_xticklabels(region_order)
        ax.set_ylabel(f'% {cell_type}')
        ax.set_title(f'{cell_type} fraction by area')
        ax.set_ylim(bottom=0)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig2.tight_layout()
    savefig(fig2, savedir, 'bt_proportions_dots')


def plot_population_rf(bt_results_by_pos, df, labeled_array, savedir,
                       animal_dirs=None):
    """Mean population rate map for EBC and RBC cells in each visual area.

    Saves
    -----
    bt_population_rf_EBC.png
    bt_population_rf_RBC.png
    """
    if animal_dirs is None:
        animal_dirs = list(bt_results_by_pos.keys())

    for cell_type in ('EBC', 'RBC'):
        is_col = 'is_EBC' if cell_type == 'EBC' else 'is_RBC'
        block_key = 'ebc' if cell_type == 'EBC' else 'rbc'
        labels = (['fwd', 'right', 'bkwd', 'left'] if cell_type == 'EBC'
                  else ['center', 'temporal', 'surround', 'nasal'])

        # ---- collect mean maps per area ----
        area_maps = {rn: [] for rn in REGION_ORDER}
        angle_rad = None
        dist_bin_cents = None

        for animal in animal_dirs:
            if animal not in bt_results_by_pos:
                continue
            for poskey, bt in bt_results_by_pos[animal].items():
                if angle_rad is None:
                    params = bt.get('params', {})
                    angle_rad     = np.array(params.get('angle_rad', []))
                    dist_bin_cents = np.array(params.get('dist_bin_cents', []))

                block    = bt.get(block_key, {})
                smaps    = np.array(block.get('smoothed_rate_maps', []))
                is_arr   = np.array(block.get(f'is_{cell_type.upper()}', []), dtype=bool)
                if smaps.ndim != 3 or len(is_arr) == 0:
                    continue

                # look up regions from df
                sub = df[(df['animal'] == animal) & (df['pos'] == poskey)]
                for c in range(min(len(is_arr), len(sub))):
                    if not is_arr[c]:
                        continue
                    row = sub[sub['cell_idx'] == c]
                    if len(row) == 0:
                        continue
                    rn = ID_TO_NAME.get(int(row.iloc[0]['region']))
                    if rn not in area_maps:
                        continue
                    rm = smaps[c].astype(float)
                    peak = np.nanmax(rm)
                    if peak > 1e-9:
                        area_maps[rn].append(rm / peak)

        if angle_rad is None or len(angle_rad) == 0:
            continue

        regions_in_data = [r for r in REGION_ORDER if area_maps[r]]
        if not regions_in_data:
            continue

        n_areas = len(regions_in_data)
        fig, axs = plt.subplots(1, n_areas, figsize=(3 * n_areas, 3.2),
                                dpi=300,
                                subplot_kw={'projection': 'polar'})
        if n_areas == 1:
            axs = [axs]

        r_edges = np.concatenate([dist_bin_cents - (dist_bin_cents[1] - dist_bin_cents[0]) / 2,
                                   [dist_bin_cents[-1] + (dist_bin_cents[1] - dist_bin_cents[0]) / 2]])
        theta_edges = np.concatenate([angle_rad - np.deg2rad(360 / len(angle_rad)) / 2,
                                      [angle_rad[-1] + np.deg2rad(360 / len(angle_rad)) / 2]])

        for ai, rn in enumerate(regions_in_data):
            ax = axs[ai]
            stack = np.array(area_maps[rn])
            mean_map = np.nanmean(stack, axis=0)   # (N_ang, N_dist)
            # pcolormesh on polar axes: theta along rows, r along cols
            mesh = ax.pcolormesh(theta_edges, r_edges, mean_map.T,
                                  cmap='hot_r', vmin=0, vmax=1, shading='flat')
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
            ax.set_xticklabels(labels, fontsize=7)
            r_max = float(r_edges[-1])
            ax.set_yticks([r_max * 0.5, r_max])
            ax.set_yticklabels([f'{r_max*0.5:.0f}', f'{r_max:.0f} cm'], fontsize=6)
            ax.set_title(f'{rn}\n(n={len(area_maps[rn])})', fontsize=8)
            color = COLORS.get(rn, 'k')
            ax.spines['polar'].set_edgecolor(color)
            ax.spines['polar'].set_linewidth(2)

        fig.colorbar(mesh, ax=axs, label='Norm. firing rate', shrink=0.7, pad=0.08)
        fig.suptitle(f'Population RF — {cell_type} cells by visual area', fontsize=10)
        fig.tight_layout()
        savefig(fig, savedir, f'bt_population_rf_{cell_type}')


def plot_bc_metric_maps(df, labeled_array, savedir):
    """Brain topography scatter/smoothed maps of BT metrics.

    Saves one scatter + one smoothed map per (cell_type × metric).
    Metrics: MRL, preferred angle, preferred distance, split-half corr.
    """
    metric_specs = [
        # (column,     cell_type, cmap,       vmin, vmax,  label)
        ('ebc_mrl',        'EBC', 'plasma',   0.0,  0.5,  'EBC MRL'),
        ('rbc_mrl',        'RBC', 'plasma',   0.0,  0.5,  'RBC MRL'),
        ('ebc_peak_dist',  'EBC', 'viridis',  0.0,  None, 'EBC preferred distance (cm)'),
        ('rbc_peak_dist',  'RBC', 'viridis',  0.0,  None, 'RBC preferred distance (cm)'),
        ('ebc_corr',       'EBC', 'coolwarm', -1.0, 1.0,  'EBC split-half corr'),
        ('rbc_corr',       'RBC', 'coolwarm', -1.0, 1.0,  'RBC split-half corr'),
    ]

    for col, cell_type, cmap_name, vmin, vmax, label in metric_specs:
        is_col = 'is_EBC' if cell_type == 'EBC' else 'is_RBC'
        sub = df[df[is_col] == 1].copy()
        sub = sub.dropna(subset=[col, 'x', 'y'])
        if len(sub) < 4:
            continue

        if vmax is None:
            vmax = float(np.nanpercentile(sub[col], 95))
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = getattr(cm, cmap_name)

        # ---- scatter per area ----
        fig, axs = plt.subplots(2, 2, figsize=(8, 6), dpi=300)
        axs = axs.flatten()
        for i, rn in enumerate(REGION_ORDER):
            rid = REGION_IDS[rn]
            region_mask = (labeled_array == rid)
            axs[i].imshow(region_mask, cmap='gray_r', alpha=0.3)
            axs[i].contour(region_mask, levels=[0.5], colors='k', linestyles='--', linewidths=0.5)
            rc = sub[sub['region'] == rid]
            if len(rc) > 0:
                axs[i].scatter(rc['x'], rc['y'], s=4, c=rc[col], cmap=cmap, norm=norm, zorder=3)
            axs[i].set_title(f'{rn} (n={len(rc)})'); axs[i].axis('off')
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs,
                     label=label, fraction=0.05, shrink=0.6)
        fig.suptitle(f'{label} — {cell_type} cells')
        fig.tight_layout()
        savefig(fig, savedir, f'bt_map_{col}')

        # ---- smoothed map ----
        fig2, axs2 = plt.subplots(2, 2, figsize=(8, 6), dpi=300)
        axs2 = axs2.flatten()
        for i, rn in enumerate(REGION_ORDER):
            rid = REGION_IDS[rn]
            region_mask = (labeled_array == rid)
            axs2[i].contour(region_mask, levels=[0.5], colors='k', linestyles='--', linewidths=0.5)
            rc = sub[sub['region'] == rid]
            if len(rc) >= 4:
                smoothed = create_smoothed_map(rc['x'].values, rc['y'].values,
                                               rc[col].values, shape=labeled_array.shape,
                                               sigma=50)
                smoothed[~region_mask] = np.nan
                axs2[i].imshow(smoothed, cmap=cmap, norm=norm)
            axs2[i].set_title(f'{rn} (n={len(rc)})'); axs2[i].axis('off')
        fig2.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs2,
                      label=label, fraction=0.05, shrink=0.6)
        fig2.suptitle(f'{label} smoothed — {cell_type} cells')
        fig2.tight_layout()
        savefig(fig2, savedir, f'bt_map_{col}_smoothed')


def plot_preferred_angle_by_area(df, savedir):
    """Polar histograms of preferred egocentric (EBC) and retinocentric (RBC)
    angle for each visual area.

    Saves
    -----
    bt_pref_angle_EBC.png
    bt_pref_angle_RBC.png
    """
    for cell_type, col, labels in [
        ('EBC', 'ebc_peak_ang', ['fwd', 'right', 'bkwd', 'left']),
        ('RBC', 'rbc_peak_ang', ['center', 'temporal', 'surround', 'nasal']),
    ]:
        is_col = 'is_EBC' if cell_type == 'EBC' else 'is_RBC'
        sub = df[(df[is_col] == 1) & df[col].notna()]

        regions_in_data = [r for r in REGION_ORDER if REGION_IDS[r] in sub['region'].values]
        if not regions_in_data:
            continue

        n_areas = len(regions_in_data)
        fig, axs = plt.subplots(1, n_areas, figsize=(2.8 * n_areas, 2.8),
                                dpi=300, subplot_kw={'projection': 'polar'})
        if n_areas == 1:
            axs = [axs]

        n_bins = 24
        angle_bins = np.linspace(0, 2 * np.pi, n_bins + 1)

        for ai, rn in enumerate(regions_in_data):
            rid = REGION_IDS[rn]
            angles = sub[sub['region'] == rid][col].values
            counts, _ = np.histogram(angles, bins=angle_bins)
            width = 2 * np.pi / n_bins
            ax = axs[ai]
            ax.bar(angle_bins[:-1], counts, width=width, align='edge',
                   color=COLORS.get(rn, 'grey'), alpha=0.8)
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_title(f'{rn}\n(n={len(angles)})', fontsize=8)

        fig.suptitle(f'Preferred angle distribution — {cell_type}', fontsize=10)
        fig.tight_layout()
        savefig(fig, savedir, f'bt_pref_angle_{cell_type}')


def plot_mrl_by_area(df, savedir):
    """Scatter + violin of EBC and RBC MRL by visual area.

    Saves
    -----
    bt_mrl_by_area.png   — side-by-side scatter for EBC MRL and RBC MRL
    bt_mrl_hist.png      — density histograms of MRL (all cells) by area
    """
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.5), dpi=300, sharey=True)
    for ax_i, (cell_type, col, color_key) in enumerate([
            ('EBC', 'ebc_mrl', '#1a7f37'),
            ('RBC', 'rbc_mrl', '#1a5fa8')]):

        ax = axs[ax_i]
        groups = []
        for xi, rn in enumerate(REGION_ORDER):
            rid = REGION_IDS.get(rn, -1)
            vals = df[df['region'] == rid][col].dropna().values
            groups.append(vals)
            add_scatter_col(ax, xi, vals, color=COLORS.get(rn, color_key))

        valid_groups = [g for g in groups if len(g) > 1]
        if len(valid_groups) > 1:
            try:
                _, p_kw = kruskal(*valid_groups)
                ax.text(0.05, 0.95, f'KW p={p_kw:.2e}', transform=ax.transAxes, fontsize=7)
            except ValueError:
                pass

        ax.set_xticks(range(len(REGION_ORDER)))
        ax.set_xticklabels(REGION_ORDER, rotation=90)
        ax.set_ylabel('MRL')
        ax.set_ylim([0, 1])
        ax.set_title(f'{cell_type} MRL by area (all cells)')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    fig.tight_layout()
    savefig(fig, savedir, 'bt_mrl_by_area')

    # ---- density histograms (only reliable cells) ----
    fig2, axs2 = plt.subplots(1, 2, figsize=(8, 3.5), dpi=300)
    for ax_i, (cell_type, mrl_col, thresh_col, color_key) in enumerate([
            ('EBC', 'ebc_mrl', 'ebc_mrl_thresh', '#1a7f37'),
            ('RBC', 'rbc_mrl', 'rbc_mrl_thresh', '#1a5fa8')]):
        ax = axs2[ax_i]
        bins = np.linspace(0, 1, 30)
        for rn in REGION_ORDER:
            rid = REGION_IDS.get(rn, -1)
            vals = df[df['region'] == rid][mrl_col].dropna().values
            if len(vals) > 0:
                ax.hist(vals, bins=bins, histtype='step', density=True,
                        color=COLORS.get(rn, 'grey'), label=f'{rn} (n={len(vals)})',
                        linewidth=1.5)
        ax.set_xlabel('MRL')
        ax.set_ylabel('Density')
        ax.set_title(f'{cell_type} MRL distribution by area')
        ax.legend(fontsize=7, frameon=False)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig2.tight_layout()
    savefig(fig2, savedir, 'bt_mrl_hist')


def plot_ebc_rbc_mrl_scatter(df, savedir):
    """EBC MRL vs RBC MRL scatter, coloured by visual area.

    Reveals whether EBC and RBC responsiveness are correlated within cells,
    and whether that relationship differs across areas.

    Saves
    -----
    bt_ebc_vs_rbc_mrl.png
    """
    fig, axs = plt.subplots(1, len(REGION_ORDER) + 1, figsize=(3.5 * (len(REGION_ORDER) + 1), 3.5), dpi=300)

    def _scatter_ax(ax, sub, color, title):
        x = sub['ebc_mrl'].values
        y = sub['rbc_mrl'].values
        valid = np.isfinite(x) & np.isfinite(y)
        if np.sum(valid) < 2:
            ax.set_title(title); return
        ax.scatter(x[valid], y[valid], s=3, color=color, alpha=0.4)
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
        r = np.corrcoef(x[valid], y[valid])[0, 1]
        ax.text(0.05, 0.93, f'r={r:.2f}\nn={np.sum(valid)}',
                transform=ax.transAxes, fontsize=7)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.set_xlabel('EBC MRL'); ax.set_ylabel('RBC MRL')
        ax.set_title(title)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    _scatter_ax(axs[0], df, 'grey', 'All areas')
    for ai, rn in enumerate(REGION_ORDER):
        rid = REGION_IDS.get(rn, -1)
        _scatter_ax(axs[ai + 1], df[df['region'] == rid], COLORS.get(rn, 'grey'), rn)

    fig.suptitle('EBC MRL vs RBC MRL per cell', fontsize=10)
    fig.tight_layout()
    savefig(fig, savedir, 'bt_ebc_vs_rbc_mrl')


def plot_preferred_distance_by_area(df, savedir):
    """Preferred wall distance distribution for EBC and RBC cells, by area.

    Also shows a brain-position scatter coloured by preferred distance.

    Saves
    -----
    bt_pref_dist_hist.png
    """
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.5), dpi=300)
    for ax_i, (cell_type, col) in enumerate([('EBC', 'ebc_peak_dist'), ('RBC', 'rbc_peak_dist')]):
        is_col = 'is_EBC' if cell_type == 'EBC' else 'is_RBC'
        ax = axs[ax_i]
        for rn in REGION_ORDER:
            rid = REGION_IDS.get(rn, -1)
            vals = df[(df[is_col] == 1) & (df['region'] == rid)][col].dropna().values
            if len(vals) > 0:
                ax.hist(vals, bins=15, histtype='step', density=True,
                        color=COLORS.get(rn, 'grey'), label=f'{rn} (n={len(vals)})',
                        linewidth=1.5)
        ax.set_xlabel('Preferred distance (cm)')
        ax.set_ylabel('Density')
        ax.set_title(f'{cell_type} preferred distance by area')
        ax.legend(fontsize=7, frameon=False)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    savefig(fig, savedir, 'bt_pref_dist_hist')


def plot_ebc_rbc_angle_correlation(df, savedir):
    """For cells that are both EBC and RBC, plot the relationship between
    their preferred egocentric and retinocentric angles.

    Also shows, for each area, how preferred EBC angle distributes for
    RBC-only vs EBC-only cells, to test if tuning interacts.

    Saves
    -----
    bt_angle_ebc_vs_rbc.png   — scatter of preferred angles for 'both' cells
    """
    both = df[df['is_both'] == 1].dropna(subset=['ebc_peak_ang', 'rbc_peak_ang'])
    if len(both) < 4:
        return

    fig, axs = plt.subplots(1, len(REGION_ORDER) + 1, figsize=(3.3 * (len(REGION_ORDER) + 1), 3.3), dpi=300)

    def _ax(ax, sub, color, title):
        if len(sub) < 2:
            ax.set_title(title); return
        x = np.rad2deg(sub['ebc_peak_ang'].values)
        y = np.rad2deg(sub['rbc_peak_ang'].values)
        ax.scatter(x, y, s=6, color=color, alpha=0.6)
        ax.set_xlim([0, 360]); ax.set_ylim([0, 360])
        ax.set_xlabel('EBC pref. angle (°)')
        ax.set_ylabel('RBC pref. angle (°)')
        ax.set_title(f'{title}\n(n={len(sub)})', fontsize=8)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    _ax(axs[0], both, 'grey', 'All areas')
    for ai, rn in enumerate(REGION_ORDER):
        rid = REGION_IDS.get(rn, -1)
        _ax(axs[ai + 1], both[both['region'] == rid], COLORS.get(rn, 'grey'), rn)

    fig.suptitle('EBC vs RBC preferred angle (cells classified as both)', fontsize=9)
    fig.tight_layout()
    savefig(fig, savedir, 'bt_angle_ebc_vs_rbc')


def plot_bt_mrl_above_threshold(df, labeled_array, savedir):
    """Brain map showing, per cell, how far the MRL exceeds the shuffle threshold.

    MRL excess = (MRL - 99th-pct shuffle threshold) / threshold.
    Positive = reliable; negative = not.  Mapped onto brain position.

    Saves
    -----
    bt_mrl_excess_map_EBC.png
    bt_mrl_excess_map_RBC.png
    """
    for cell_type, mrl_col, thresh_col in [
            ('EBC', 'ebc_mrl', 'ebc_mrl_thresh'),
            ('RBC', 'rbc_mrl', 'rbc_mrl_thresh')]:

        sub = df.dropna(subset=[mrl_col, thresh_col, 'x', 'y']).copy()
        sub['mrl_excess'] = (sub[mrl_col] - sub[thresh_col]) / (sub[thresh_col] + 1e-9)

        limit = float(np.nanpercentile(np.abs(sub['mrl_excess']), 95))
        norm  = mcolors.Normalize(vmin=-limit, vmax=limit)
        cmap  = cm.coolwarm

        fig, axs = plt.subplots(2, 2, figsize=(8, 6), dpi=300)
        axs = axs.flatten()
        for i, rn in enumerate(REGION_ORDER):
            if i >= len(axs): break
            rid = REGION_IDS.get(rn, -1)
            region_mask = (labeled_array == rid)
            axs[i].imshow(region_mask, cmap='gray_r', alpha=0.3)
            axs[i].contour(region_mask, levels=[0.5], colors='k', linestyles='--', linewidths=0.5)
            rc = sub[sub['region'] == rid]
            if len(rc) > 0:
                axs[i].scatter(rc['x'], rc['y'], s=3, c=rc['mrl_excess'],
                               cmap=cmap, norm=norm, zorder=3)
            axs[i].set_title(f'{rn} (n={len(rc)})'); axs[i].axis('off')
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs,
                     label='(MRL - threshold) / threshold', fraction=0.05, shrink=0.6)
        fig.suptitle(f'{cell_type} MRL excess above shuffle threshold')
        fig.tight_layout()
        savefig(fig, savedir, f'bt_mrl_excess_map_{cell_type}')


def plot_boundary_tuning_summary(pooled_data, bt_results_by_pos, labeled_array,
                                 savedir, animal_dirs=None):
    """Run all boundary-tuning summary figures.

    Parameters
    ----------
    pooled_data : dict
        Loaded pooled dataset (from make_pooled_dataset / read_h5).
    bt_results_by_pos : dict
        ``{animal_dir: {poskey: bt_dict}}``  where each bt_dict is loaded
        from a ``save_results_combined()`` HDF5 file via ``fm2p.read_h5()``.
    labeled_array : np.ndarray
        Integer region label array in reference VFS space.
    savedir : str
        Directory to save PNG files.
    animal_dirs : list of str, optional
        Subset of animals to include.
    """
    os.makedirs(savedir, exist_ok=True)

    print('Building BT cell DataFrame...')
    df = build_bt_cell_df(pooled_data, bt_results_by_pos, labeled_array, animal_dirs)
    if len(df) == 0:
        print('  No BT data found.')
        return df

    print(f'  {len(df)} cells total, '
          f'{df["is_EBC"].sum()} EBC, {df["is_RBC"].sum()} RBC.')

    print('  EBC/RBC proportions...')
    plot_ebc_rbc_proportions(df, labeled_array, savedir)

    print('  Population RF maps...')
    plot_population_rf(bt_results_by_pos, df, labeled_array, savedir, animal_dirs)

    print('  BT metric brain maps...')
    plot_bc_metric_maps(df, labeled_array, savedir)

    print('  Preferred angle distributions...')
    plot_preferred_angle_by_area(df, savedir)

    print('  MRL by area...')
    plot_mrl_by_area(df, savedir)

    print('  EBC vs RBC MRL scatter...')
    plot_ebc_rbc_mrl_scatter(df, savedir)

    print('  Preferred distance distributions...')
    plot_preferred_distance_by_area(df, savedir)

    print('  EBC vs RBC angle correlation...')
    plot_ebc_rbc_angle_correlation(df, savedir)

    print('  MRL excess brain maps...')
    plot_bt_mrl_above_threshold(df, labeled_array, savedir)

    print(f'Done. BT summary figures saved to {savedir}')
    return df


def plot_fov_stitched_dmm056(animal_dir, savedir):
    """Composite all 2P fields of view for DMM056 over its widefield reference image.

    Reads the widefield reference TIF, the local-to-global cell transform file,
    and one preproc.h5 per position.  For each position the 2P reference image is
    warped (via a least-squares affine estimated from cell correspondences) into the
    1024×1024 widefield coordinate space and blended onto the base image.  FOV
    outlines (transformed 512×512 corners) and visual-area contours are overlaid.

    Parameters
    ----------
    animal_dir : str
        Directory containing DMM056 data files
        (``mouse_composites/DMM056/``).
    savedir : str
        Output directory for the saved SVG.
    """
    import h5py
    import glob
    from PIL import Image as PILImage
    from matplotlib.patches import Polygon as MplPolygon

    ref_img_path  = os.path.join(animal_dir, '250929_DMM056_signmap_refimg.tif')
    lg_path       = os.path.join(animal_dir,
                                 'DMM056_aligned_composite_local_to_global_transform.h5')
    contour_path  = next(
        (os.path.join(animal_dir, f) for f in os.listdir(animal_dir)
         if f.startswith('vfs_area_contours') and f.endswith('.h5')),
        None
    )

    if not os.path.exists(ref_img_path) or not os.path.exists(lg_path):
        print(f'plot_fov_stitched_dmm056: missing data files in {animal_dir}')
        return

    # ---- reference image (2048×2048 uint16) → 1024×1024 base ----
    ref_arr = np.array(PILImage.open(ref_img_path)).astype(float)
    ref_lo  = float(np.percentile(ref_arr, 1))
    ref_hi  = float(np.percentile(ref_arr, 99))
    ref_norm  = np.clip((ref_arr - ref_lo) / (ref_hi - ref_lo + 1e-6), 0, 1)
    ref_uint8 = (ref_norm * 255).astype(np.uint8)
    base_size = 1024
    ref_resized = np.array(
        PILImage.fromarray(ref_uint8).resize((base_size, base_size), PILImage.BILINEAR)
    )
    base_pil = PILImage.fromarray(ref_resized).convert('RGBA')

    # ---- local→global cell transforms ----
    with h5py.File(lg_path, 'r') as g:
        positions      = sorted(g.keys())
        transform_data = {pos: g[pos][()] for pos in positions}

    # ---- find one preproc.h5 per position ----
    pos_to_preproc = {}
    search_root = os.path.dirname(os.path.dirname(animal_dir))
    for preproc_file in glob.glob(
            os.path.join(search_root, '**', '*DMM056*preproc.h5'), recursive=True):
        dirname = os.path.basename(os.path.dirname(os.path.dirname(preproc_file)))
        for part in dirname.split('_'):
            if part.startswith('pos'):
                if part not in pos_to_preproc:
                    pos_to_preproc[part] = preproc_file
                break

    # ---- composite each FOV onto the reference ----
    cmap_tab = plt.cm.tab20
    n_pos    = max(len(positions), 1)

    for i, pos in enumerate(positions):
        data_arr = transform_data[pos]
        local_xy = data_arr[:, :2]   # (N, 2) in 512×512 FOV pixel space
        global_xy = data_arr[:, 2:4]  # (N, 2) in 1024-space
        N = len(local_xy)
        if N < 4 or pos not in pos_to_preproc:
            continue

        color = np.array(cmap_tab(i / n_pos))

        try:
            with h5py.File(pos_to_preproc[pos], 'r') as pf:
                if 'twop_ref_img' not in pf:
                    continue
                fov_img = pf['twop_ref_img'][()].astype(float)
        except Exception:
            continue

        flo = float(np.percentile(fov_img, 1))
        fhi = float(np.percentile(fov_img, 99))
        fov_n = np.clip((fov_img - flo) / (fhi - flo + 1e-6), 0, 1)

        # tint FOV with position colour
        fh, fw = fov_img.shape[:2]
        fov_rgba = np.zeros((fh, fw, 4), dtype=np.uint8)
        fov_rgba[:, :, 0] = (fov_n * color[0] * 255).astype(np.uint8)
        fov_rgba[:, :, 1] = (fov_n * color[1] * 255).astype(np.uint8)
        fov_rgba[:, :, 2] = (fov_n * color[2] * 255).astype(np.uint8)
        fov_rgba[:, :, 3] = (fov_n * 200).astype(np.uint8)
        fov_pil = PILImage.fromarray(fov_rgba, mode='RGBA')

        # Estimate global → local affine (PIL AFFINE = inverse mapping)
        aug_g = np.column_stack([global_xy, np.ones(N)])
        B, _, _, _ = np.linalg.lstsq(aug_g, local_xy, rcond=None)  # (3, 2)
        # PIL data: (a, b, c, d, e, f)  where  x_in = a·x_out + b·y_out + c
        #                                       y_in = d·x_out + e·y_out + f
        pil_data = (float(B[0, 0]), float(B[1, 0]), float(B[2, 0]),
                    float(B[0, 1]), float(B[1, 1]), float(B[2, 1]))
        warped = fov_pil.transform(
            (base_size, base_size), PILImage.AFFINE, pil_data, resample=PILImage.BILINEAR
        )
        base_pil.alpha_composite(warped)

    # ---- draw figure ----
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    ax.imshow(np.array(base_pil))  # imshow inverts y automatically

    # FOV outlines + labels
    for i, pos in enumerate(positions):
        data_arr = transform_data[pos]
        local_xy = data_arr[:, :2]
        global_xy = data_arr[:, 2:4]
        N = len(local_xy)
        if N < 4:
            continue
        color = np.array(cmap_tab(i / n_pos))
        # Estimate local → global affine for corners
        aug_l = np.column_stack([local_xy, np.ones(N)])
        A, _, _, _ = np.linalg.lstsq(aug_l, global_xy, rcond=None)
        fh_l, fw_l = 512, 512
        corners_l = np.array([[0, 0], [fw_l, 0], [fw_l, fh_l], [0, fh_l]], dtype=float)
        corners_g  = np.column_stack([corners_l, np.ones(4)]) @ A
        poly = MplPolygon(corners_g, closed=True, fill=False,
                          edgecolor=color[:3], linewidth=1.5)
        ax.add_patch(poly)
        center_g = np.array([[fw_l / 2, fh_l / 2, 1]]) @ A
        ax.text(center_g[0, 0], center_g[0, 1], pos.replace('pos', 'p'),
                ha='center', va='center', fontsize=6,
                color=color[:3], fontweight='bold')

    # Visual area contours (vfs_area_contours are in 2048-space → scale to 1024)
    if contour_path and os.path.exists(contour_path):
        with h5py.File(contour_path, 'r') as f:
            for k in sorted(f.keys()):
                if not k.startswith('contour_'):
                    continue
                area_name = k[len('contour_'):]
                pts = f[k][()].astype(float) / 2.0
                ax.plot(pts[:, 0], pts[:, 1], 'w-', lw=1.5, alpha=0.85)
                color_a = COLORS.get(area_name, 'white')
                ax.text(float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])),
                        area_name, color=color_a, fontsize=9,
                        ha='center', va='center', fontweight='bold')

    ax.axis('off')
    ax.set_title(f'DMM056 — {len(positions)} 2P recording positions', fontsize=10)
    fig.tight_layout()
    savefig(fig, savedir, 'dmm056_fov_stitched')


def plot_cells_randomized_jet_dmm056(animal_dir, savedir):
    """Scatter all cells for DMM056 in randomized jet colors over visual area contours.

    Each cell receives an independently drawn uniform-random value mapped through
    the jet colormap, making individual cells visually distinct while the overall
    spatial density and clustering across areas remain apparent.

    Parameters
    ----------
    animal_dir : str
        Directory containing DMM056 data files.
    savedir : str
        Output directory for the saved SVG.
    """
    import h5py

    lg_path = os.path.join(
        animal_dir, 'DMM056_aligned_composite_local_to_global_transform.h5'
    )
    contour_path = next(
        (os.path.join(animal_dir, f) for f in os.listdir(animal_dir)
         if f.startswith('vfs_contours') and f.endswith('.json')),
        None
    )

    if not os.path.exists(lg_path):
        print(f'plot_cells_randomized_jet_dmm056: missing {lg_path}')
        return

    # collect all global cell positions (1024-space)
    with h5py.File(lg_path, 'r') as g:
        positions = sorted(g.keys())
        all_global_xy = np.vstack([g[pos][:, 2:4] for pos in positions])

    n_cells = len(all_global_xy)
    rng = np.random.default_rng(0)
    cell_colors = rng.random(n_cells)  # uniform [0, 1] → jet cmap

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)

    # Visual area contours (2048-space → 1024)
    if contour_path and os.path.exists(contour_path):
        with h5py.File(contour_path, 'r') as f:
            for k in sorted(f.keys()):
                if not k.startswith('contour_'):
                    continue
                area_name = k[len('contour_'):]
                pts = f[k][()].astype(float) / 2.0
                ax.plot(pts[:, 0], pts[:, 1], 'k-', lw=1.5)
                cx = float(np.mean(pts[:, 0]))
                cy = float(np.mean(pts[:, 1]))
                ax.text(cx, cy, area_name,
                        color=COLORS.get(area_name, 'k'), fontsize=9,
                        ha='center', va='center', fontweight='bold')

    ax.scatter(all_global_xy[:, 0], all_global_xy[:, 1],
               c=cell_colors, cmap='jet', s=4, alpha=0.8, linewidths=0)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'DMM056 — all cells (n={n_cells}, randomised jet)', fontsize=10)
    fig.tight_layout()
    savefig(fig, savedir, 'dmm056_cells_jet')


def plot_cells_randomized_jet_all_animals(pooled_data, savedir):
    """Scatter all cells from all animals in randomized jet colors over visual area contours.

    Parameters
    ----------
    pooled_data : dict
        Pooled dataset containing 'transform' for each animal and 'ref_contour_*'.
    savedir : str
        Output directory.
    """
    
    all_global_xy = []
    
    for animal_key in pooled_data:
        if not isinstance(pooled_data[animal_key], dict) or 'transform' not in pooled_data[animal_key]:
            continue
            
        transform_data = pooled_data[animal_key]['transform']
        if not isinstance(transform_data, dict):
            continue

        for pos_key, arr in transform_data.items():
            if not isinstance(arr, np.ndarray) or arr.ndim < 2 or arr.shape[1] < 4:
                continue
            xy = arr[:, 2:4].astype(float)
            if len(xy) == 0:
                continue

            # DMM056 cell coordinates are in 1024-space from register_tiled_locations.
            # Other animals' coordinates from register_animals were incorrectly scaled
            # using a 2048px reference instead of 1024px, making them half the size
            # they should be in the 400px reference space.
            if animal_key == 'DMM056':
                xy *= (400.0 / 1024.0)
            else:
                xy *= 2.0
            
            all_global_xy.append(xy)
            
    if not all_global_xy:
        print("No cell positions found for all_animals plot.")
        return

    all_global_xy = np.vstack(all_global_xy)
    n_cells = len(all_global_xy)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    plot_cells_randomized_jet_dmm056_generic(all_global_xy, pooled_data, savedir, title=f'All Animals — {n_cells} cells', fig=fig, ax=ax)


def plot_cells_randomized_jet_dmm056_generic(all_global_xy, data, savedir, title='', fig=None, ax=None):
    """Helper to plot provided XY coordinates over contours from pooled data."""
    n_cells = len(all_global_xy)
    rng = np.random.default_rng(0)
    cell_colors = rng.random(n_cells)

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=200)

    # Plot contours from pooled data
    contour_keys = [k for k in data.keys() if k.startswith('ref_contour_')]
    # If contours missing from data, fall back to bundled vfs_contours.json
    if not contour_keys:
        import json
        _json_path = os.path.join(os.path.dirname(__file__), 'vfs_contours.json')
        if os.path.isfile(_json_path):
            with open(_json_path) as _f:
                _json_contours = json.load(_f)
            for _area, _coords in _json_contours.items():
                if _coords and len(_coords) >= 3:
                    data = dict(data)  # don't mutate caller's dict
                    data[f'ref_contour_{_area}'] = np.array(_coords)
                    contour_keys.append(f'ref_contour_{_area}')

    for k in sorted(contour_keys):
        area_name = k.replace('ref_contour_', '')
        pts = data[k]
        if pts is None or len(pts) < 3:
            continue

        ax.plot(pts[:, 0], pts[:, 1], 'k-', lw=1.5)
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        ax.text(cx, cy, area_name,
                color=COLORS.get(area_name, 'k'), fontsize=9,
                ha='center', va='center', fontweight='bold')

    ax.scatter(all_global_xy[:, 0], all_global_xy[:, 1],
               c=cell_colors, cmap='jet', s=2, alpha=0.6, linewidths=0)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    savefig(fig, savedir, 'all_animals_cells_jet')


def calculate_and_print_modulation_stats(pooled_data, savedir):
    """
    Calculate and print the percentage of cells with modulation index > 0.33
    to at least one variable, for Light and Dark conditions.
    Restricted to visual areas V1, RL, AM, PM (IDs 5, 2, 3, 4).
    """
    if pooled_data is None:
        return

    variables = ['theta', 'phi', 'dTheta', 'dPhi', 'pitch', 'yaw', 'roll', 'dPitch', 'dYaw', 'dRoll']
    alias_map = {'dRoll': 'gyro_x', 'dPitch': 'gyro_y', 'dYaw': 'gyro_z'}
    target_regions = {2, 3, 4, 5, 10} # RL, AM, PM, V1, A
    region_names = {5: 'V1', 2: 'RL', 3: 'AM', 4: 'PM', 10: 'A'}

    fig, axs = plt.subplots(2, 1, figsize=(3.5, 6), dpi=300)

    print("-" * 60)
    for i, cond in enumerate(['l', 'd']):
        cond_name = 'Light' if cond == 'l' else 'Dark'
        ax = axs[i]
        stats = {rid: {'total': 0, 'mod': 0} for rid in target_regions}
        total_cells_tracked = 0
        modulated_cells_tracked = 0
        
        for animal in pooled_data:
            if animal == 'uniref' or not isinstance(pooled_data[animal], dict):
                continue
            
            if 'messentials' not in pooled_data[animal]:
                continue
                
            messentials = pooled_data[animal]['messentials']
            
            for pos in messentials:
                if not pos.startswith('pos'):
                    continue
                
                # Exclusions matching topography.py
                if (animal == 'DMM056') and (cond == 'd') and ((pos == 'pos15') or (pos == 'pos03')):
                    continue
                
                pos_data = messentials[pos]
                if 'rdata' not in pos_data:
                    continue
                rdata = pos_data['rdata']
                
                visual_area_ids = pos_data.get('visual_area_id', None)
                
                # Determine n_cells from a tuning array
                n_cells = 0
                for k in rdata:
                    if k.endswith('_1dtuning'):
                        n_cells = len(rdata[k])
                        break
                
                if n_cells == 0 or visual_area_ids is None:
                    continue
                
                if not isinstance(visual_area_ids, np.ndarray):
                    visual_area_ids = np.array(visual_area_ids)

                # Iterate cells
                limit = min(len(visual_area_ids), n_cells)
                for c_idx in range(limit):
                    rid = visual_area_ids[c_idx]
                    if rid not in target_regions:
                        continue
                    
                    stats[rid]['total'] += 1
                    total_cells_tracked += 1
                    is_mod = False
                    
                    for var in variables:
                        use_key = alias_map.get(var, var)
                        rel_key = f'{use_key}_{cond}_rel'
                        
                        if rel_key in rdata:
                            val = rdata[rel_key][c_idx]
                            if not np.isnan(val) and val > 0.33:
                                is_mod = True
                                break
                    
                    if is_mod:
                        stats[rid]['mod'] += 1
                        modulated_cells_tracked += 1

        plot_labels = ['All']
        plot_vals = []
        plot_colors = ['grey']

        print(f"Condition: {cond_name}")
        if total_cells_tracked > 0:
            pct = (modulated_cells_tracked / total_cells_tracked) * 100
            print(f"  ALL: {pct:.2f}% ({modulated_cells_tracked}/{total_cells_tracked})")
            plot_vals.append(pct)
            
            for rid in [5, 2, 3, 4, 10]: # V1, RL, AM, PM, A
                s = stats[rid]
                rname = region_names[rid]
                plot_labels.append(rname)
                plot_colors.append(COLORS.get(rname, 'k'))
                if s['total'] > 0:
                    pct_r = (s['mod'] / s['total']) * 100
                    print(f"  {rname}: {pct_r:.2f}% ({s['mod']}/{s['total']})")
                    plot_vals.append(pct_r)
                else:
                    plot_vals.append(0)
        else:
            print(f"  No cells found in target regions.")
            
        ax.bar(plot_labels, plot_vals, color=plot_colors)
        ax.set_ylim([0, 20])
        ax.set_ylabel('% Modulated (Any Var)')
        ax.set_title(cond_name)
        
    print("-" * 60)
    fig.tight_layout()
    savefig(fig, savedir, 'percent_modulated_any_var_summary')

def main():

    h5_path  = '/home/dylan/Fast2/topography_analysis_results_260310_v03.h5'
    savedir  = '/home/dylan/Fast2/topography_plots_260310_v02'

    os.makedirs(savedir, exist_ok=True)

    print(f'Loading {h5_path} ...')
    data = read_h5(h5_path)
    print('Loaded.')

    labeled_array = np.array(data['labeled_array'], dtype=int)

    print('Plotting modulation summaries ...')
    for cond in ['l', 'd']:
        plot_modulation_summary(data, cond, savedir)

    print('Plotting modulation histograms ...')
    for cond in ['l', 'd']:
        plot_modulation_histograms(data, cond, savedir)

    print('Plotting region outlines ...')
    plot_region_outlines(labeled_array, savedir)

    print('Plotting behavior correlation matrix ...')
    plot_behavior_corr_matrix(data, savedir)

    print('Plotting variable summaries ...')
    for key in VARIABLES:
        for cond in ['l', 'd']:
            print(f'  {key} {cond}')
            plot_variable_summary(data, key, cond, labeled_array, savedir)

    print('Plotting signal/noise correlations ...')
    for key in VARIABLES:
        for cond in ['l', 'd']:
            plot_signal_noise_correlations(data, key, cond, savedir)

    print('Plotting all variable importance ...')
    plot_all_variable_importance(data, savedir)

    print('Plotting all model performance ...')
    plot_all_model_performance(data, savedir)

    print('Plotting model performance maps ...')
    plot_model_performance_maps(data, labeled_array, savedir)

    print('Plotting sorted tuning curves ...')
    for key in VARIABLES:
        for cond in ['l', 'd']:
            plot_sorted_tuning_curves(data, key, cond, savedir)

    print('Plotting light/dark modulation ...')
    plot_lightdark_modulation_histograms(data, savedir)

    print('Plotting position occupancy ...')
    plot_position_occupancy(data, savedir)

    print(f'Done. PNGs saved to {savedir}')

    dmm056_dir = '/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites/DMM056'
    if os.path.isdir(dmm056_dir):
        print('Plotting DMM056 FOV stitched figure ...')
        plot_fov_stitched_dmm056(dmm056_dir, savedir)

        print('Plotting DMM056 cells (randomised jet) ...')
        plot_cells_randomized_jet_dmm056(dmm056_dir, savedir)

    pooled_path = '/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites/pooled_260310a.h5'
    print('Plotting all animals cells (randomised jet) ...')
    if os.path.exists(pooled_path):
        pooled_data = fm2p.read_h5(pooled_path)
        calculate_and_print_modulation_stats(pooled_data, savedir)
        plot_cells_randomized_jet_all_animals(pooled_data, savedir)
    else:
        print(f"Warning: Pooled dataset not found at {pooled_path}")



if __name__ == '__main__':
    main()
