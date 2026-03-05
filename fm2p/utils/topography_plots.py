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
import fm2p


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
}

REGION_ORDER = ['V1', 'RL', 'AM', 'PM']
REGION_IDS   = {'V1': 5, 'RL': 2, 'AM': 3, 'PM': 4}
ID_TO_NAME   = {5: 'V1', 2: 'RL', 3: 'AM', 4: 'PM'}

VARIABLES = ['theta', 'phi', 'dTheta', 'dPhi', 'pitch', 'yaw', 'roll',
             'dPitch', 'dYaw', 'dRoll']

LABEL_MAP = {
    0: 'boundary', 1: 'outside',
    2: 'RL', 3: 'AM', 4: 'PM', 5: 'V1',
    7: 'AL', 8: 'LM', 9: 'P'
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


def add_scatter_col(ax, pos, vals, color='k'):
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
    fig.savefig(os.path.join(savedir, f'{name}.png'), dpi=300, bbox_inches='tight')
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

    MIN_RATE = 0.05

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
                rid = REGION_IDS[rn]
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
            ax.set_xlabel('Modulation Index')
            ax.set_ylabel('Density')
            ax.set_title(f'{key} MI by Region ({cond_name}) — all cells, rate>{MIN_RATE:.2f}')
            ax.legend(fontsize=7, frameon=False)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            fig.tight_layout()
            savefig(fig, savedir, f'varsummary_{key}_{cond}_mod_hist')

        else:

            fig, ax = plt.subplots(1, 1, figsize=(5, 3.5), dpi=300)
            groups = []
            for i, rn in enumerate(REGION_ORDER):
                rid = REGION_IDS[rn]
                region_vals = rel[rel['region'] == rid][metric].dropna()
                groups.append(region_vals.values)
                if len(region_vals) > 0:
                    add_scatter_col(ax, i, region_vals.values, color=COLORS[rn])
            valid_groups = [g[~np.isnan(g)] for g in groups if len(g[~np.isnan(g)]) > 0]
            if len(valid_groups) > 1:
                try:
                    _, p_kw = kruskal(*valid_groups)
                    ax.text(0.05, 0.95, f'KW p={p_kw:.1e}', transform=ax.transAxes, fontsize=8)
                except ValueError:
                    pass
            ax.set_xticks(np.arange(4), labels=REGION_ORDER)
            if key in ['pitch', 'roll'] and metric == 'peak':
                ax.set_ylim([-35, 35])
            elif metric == 'imp':
                ax.set_ylim([0, 0.25])
            elif metric == 'peak' and key in ['theta', 'phi', 'yaw', 'roll', 'pitch']:
                ax.set_ylim([-15, 15])
            ax.set_xlim([-.5, 3.5])
            ax.set_ylabel(label_str)
            ax.set_title(f'{key} {metric} by Region ({cond_name})')
            fig.tight_layout()
            savefig(fig, savedir, f'varsummary_{key}_{cond}_{metric}_scatter')

        fig, axs = plt.subplots(2, 2, dpi=300, figsize=(8, 6))
        axs = axs.flatten()
        for i, rn in enumerate(REGION_ORDER):
            rid = REGION_IDS[rn]
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
            axs[i].set_ylim([labeled_array.shape[0], 0])
            axs[i].set_xlim([0, labeled_array.shape[1]])
        fig.suptitle(f'{key} {metric} Map by Region ({cond_name})')
        fig.tight_layout()
        savefig(fig, savedir, f'varsummary_{key}_{cond}_{metric}_map')

        fig, axs = plt.subplots(2, 2, dpi=300, figsize=(8, 6))
        axs = axs.flatten()
        for i, rn in enumerate(REGION_ORDER):
            rid = REGION_IDS[rn]
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
            axs[i].set_ylim([labeled_array.shape[0], 0])
            axs[i].set_xlim([0, labeled_array.shape[1]])
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
                rid = REGION_IDS[rn]
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
    pooled_sig    = np.array(d['pooled_sig'],    dtype=float)
    pooled_noise  = np.array(d['pooled_noise'],  dtype=float)
    pooled_regions = np.array(d['pooled_regions'])

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

    def _plot_binned(ax, sig_arr, noise_arr, color, label):
        mean_noise, sem_noise = [], []
        for lo, hi in zip(sig_bins[:-1], sig_bins[1:]):
            in_bin = (sig_arr >= lo) & (sig_arr < hi)
            n_in = np.sum(in_bin)
            if n_in < 5:
                mean_noise.append(np.nan); sem_noise.append(np.nan)
            else:
                mean_noise.append(np.nanmean(noise_arr[in_bin]))
                sem_noise.append(np.nanstd(noise_arr[in_bin]) / np.sqrt(n_in))
        mean_noise = np.array(mean_noise); sem_noise = np.array(sem_noise)
        ax.plot(sig_bin_centers, mean_noise, '-o', markersize=3, color=color, label=label)
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
    for i, (rid, rname) in enumerate(zip(region_ids, region_names)):
        mask = (pooled_regions[:, 0] == rid) & (pooled_regions[:, 1] == rid) & valid
        if np.sum(mask) > 10:
            _plot_binned(axs2[i + 1], pooled_sig[mask], pooled_noise[mask], COLORS[rname], rname)
        axs2[i + 1].set_title(f'{rname} Pairs')
        axs2[i + 1].set_xlabel('Signal Corr'); axs2[i + 1].set_xlim([-1, 1])
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
        rid = REGION_IDS[rn]
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
        rid = REGION_IDS[rn]
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


        region_vals = []
        for xi, yi in zip(df['x'], df['y']):
            xi_i, yi_i = int(np.clip(xi, 0, labeled_array.shape[1] - 1)), \
                         int(np.clip(yi, 0, labeled_array.shape[0] - 1))
            region_vals.append(labeled_array[yi_i, xi_i])
        df['region'] = region_vals


        fig, axs = plt.subplots(2, 2, dpi=300, figsize=(8, 8))
        axs = axs.flatten()
        for i, rn in enumerate(REGION_ORDER):
            rid = REGION_IDS[rn]
            region_mask = (labeled_array == rid)
            axs[i].imshow(region_mask, cmap='gray_r', alpha=0.4)
            axs[i].imshow(labeled_array == 0, cmap='binary', alpha=0.1)
            axs[i].contour(region_mask, levels=[0.5], colors='k', linestyles='--', linewidths=0.5)
            rc = df[df['region'] == rid]
            if len(rc) > 0:
                axs[i].scatter(rc['x'], rc['y'], s=2, c=rc['val'], cmap=cmap, norm=norm)
            axs[i].set_title(rn); axs[i].axis('off')
            axs[i].set_ylim([labeled_array.shape[0], 0])
            axs[i].set_xlim([0, labeled_array.shape[1]])
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs,
                     label=label_str, fraction=0.05, shrink=0.6)
        fig.suptitle(f'{label_str} Map by Region')
        fig.tight_layout()
        savefig(fig, savedir, f'model_map_{key_name}')


        fig, axs = plt.subplots(2, 2, dpi=300, figsize=(8, 6))
        axs = axs.flatten()
        for i, rn in enumerate(REGION_ORDER):
            rid = REGION_IDS[rn]
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
            axs[i].set_ylim([labeled_array.shape[0], 0])
            axs[i].set_xlim([0, labeled_array.shape[1]])
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
                if var in results[metric] and rn in results[metric][var]:
                    vals = results[metric][var][rn]
                    if isinstance(vals, dict):
                        vals = list(vals.values())
                    elif isinstance(vals, np.ndarray):
                        vals = vals.tolist()
                    if vals:
                        add_scatter_col(ax, col_idx, vals, color=COLORS[rn])
            ax.set_xticks(range(4))
            ax.set_xticklabels(REGION_ORDER)
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
        metric_label = 'Modulation Index' if metric == 'mod' else 'Reliability Score (1 - null count/100)'

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

        xlim = (0, 1.0); bins = np.linspace(0, 1.0, 21)

        fig, axs = plt.subplots(2, 5, figsize=(7, 3.5), dpi=300)
        axs = axs.flatten()
        for i, var in enumerate(VARIABLES):
            ax = axs[i]
            if var not in results[metric]:
                ax.axis('off'); continue
            for rn in REGION_ORDER:
                if rn in results[metric][var]:
                    vals = results[metric][var][rn]
                    if isinstance(vals, dict):
                        vals = list(vals.values())
                    vals = np.array(vals, dtype=float)
                    vals = vals[~np.isnan(vals)]
                    if len(vals) > 0:
                        ax.hist(vals, bins=bins, density=True, histtype='step',
                                color=COLORS[rn], label=rn, linewidth=1.5)
            # imp: threshold = (100-10)/100 = 0.9 marks the relthresh=10 boundary
            ref_lines = [0, 0.33, 0.5] if metric == 'mod' else [0.9]
            for line_val in ref_lines:
                ax.axvline(line_val, color='k', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.set_title(var); ax.set_xlim(xlim)
            if i % 5 == 0:
                ax.set_ylabel('Density')
            if i >= 5:
                ax.set_xlabel(metric_label)
            if i == 0:
                ax.legend(fontsize=8, frameon=False)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
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
        axs[0, 0].set_title(f'Dark-reliable → MI in Dark  (n={len(dark_rel_mod_d)})')
        axs[0, 0].set_ylabel('Density')
        axs[0, 1].hist(dark_rel_mod_l[np.isfinite(dark_rel_mod_l)], bins=hist_bins, density=True,
                       color=COLORS['light'], alpha=0.7)
        axs[0, 1].set_title(f'Dark-reliable → MI in Light  (n={len(dark_rel_mod_l)})')
        axs[1, 0].hist(light_rel_mod_d[np.isfinite(light_rel_mod_d)], bins=hist_bins, density=True,
                       color=COLORS['dark'], alpha=0.7)
        axs[1, 0].set_title(f'Light-reliable → MI in Dark  (n={len(light_rel_mod_d)})')
        axs[1, 0].set_ylabel('Density')
        axs[1, 1].hist(light_rel_mod_l[np.isfinite(light_rel_mod_l)], bins=hist_bins, density=True,
                       color=COLORS['light'], alpha=0.7)
        axs[1, 1].set_title(f'Light-reliable → MI in Light  (n={len(light_rel_mod_l)})')
        for ax in axs.flatten():
            ax.axvline(0.33, color='k', ls='--', alpha=0.5, lw=0.8)
            ax.set_xlim([0, 1.0]); ax.set_xlabel('Modulation Index')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.suptitle(f'{var}: Cross-condition MI for Condition-Reliable Cells')
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
            ax.set_xlabel(xlabel, fontsize=7); ax.set_ylabel(ylabel, fontsize=7)
            ax.tick_params(labelsize=6)

        for idx in range(n_recs, nrows * ncols):
            axs_grid[idx // ncols, idx % ncols].axis('off')

        fig.suptitle(fig_title)
        fig.tight_layout()
        savefig(fig, savedir, name)


def main():

    h5_path  = '/home/dylan/Fast2/topography_analysis_results_v09e.h5'
    savedir  = '/home/dylan/Fast2/topography_plots'

    os.makedirs(savedir, exist_ok=True)

    print(f'Loading {h5_path} ...')
    data = fm2p.read_h5(h5_path)
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


if __name__ == '__main__':
    main()
