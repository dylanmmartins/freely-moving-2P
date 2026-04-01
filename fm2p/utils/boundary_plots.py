# -*- coding: utf-8 -*-
"""boundary_plots.py

Reads the topography analysis results HDF5 (output of topography.py) and
generates summary figures describing the distribution of EBC and RBC cells
across visual areas.

Requires that the topography results h5 contains the keys written by
``aggregate_boundary_data()`` (added to topography.py):
  boundary_cell_data, boundary_ebc_mean_maps, boundary_rbc_mean_maps,
  boundary_params, labeled_array.

Usage
-----
Edit the paths in main() and run:

    python -m fm2p.utils.boundary_plots
"""

if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
    __package__ = 'fm2p.utils'

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats
from scipy.stats import kruskal
from collections import defaultdict

from .files import read_h5

import matplotlib as mpl
mpl.rcParams['axes.spines.top']   = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype']      = 42
mpl.rcParams['ps.fonttype']       = 42
mpl.rcParams['font.size']         = 7


REGION_ORDER = ['V1', 'RL', 'AM', 'PM', 'A', 'AL', 'LM', 'P']
REGION_IDS   = {'V1': 5, 'RL': 2, 'AM': 3, 'PM': 4, 'A': 10, 'AL': 7, 'LM': 8, 'P': 9}
ID_TO_NAME   = {v: k for k, v in REGION_IDS.items()}

COLORS = {
    'V1': '#1B9E77', 'RL': '#D95F02', 'AM': '#7570B3', 'PM': '#E7298A',
    'A':  '#1A5A34', 'AL': '#E6AB02', 'LM': '#A6761D', 'P':  '#666666',
}

EBC_COLOR  = '#1a7f37'
RBC_COLOR  = '#1a5fa8'
BOTH_COLOR = '#9b2dca'


def load_topo_results(topo_h5_path):

    topo = read_h5(topo_h5_path)

    cd = topo.get('boundary_cell_data', {})
    if not cd or 'is_EBC' not in cd:
        raise ValueError(
            'boundary_cell_data not found in topography results. '
            'Re-run topography.py after updating with aggregate_boundary_data().')

    area_id  = np.asarray(cd['area_id'],  dtype=int)
    vfs_x    = np.asarray(cd['vfs_x'],   dtype=float)
    vfs_y    = np.asarray(cd['vfs_y'],   dtype=float)
    is_EBC   = np.asarray(cd['is_EBC'],  dtype=bool)
    is_RBC   = np.asarray(cd['is_RBC'],  dtype=bool)
    is_fr_e  = np.asarray(cd.get('is_fully_reliable_EBC', cd['is_EBC']), dtype=bool)
    is_fr_r  = np.asarray(cd.get('is_fully_reliable_RBC', cd['is_RBC']), dtype=bool)
    ebc_mrl  = np.asarray(cd.get('ebc_mrl',  np.full(len(is_EBC), np.nan)))
    rbc_mrl  = np.asarray(cd.get('rbc_mrl',  np.full(len(is_RBC), np.nan)))
    ebc_rf_corr = np.asarray(cd.get('ebc_rf_corr', np.full(len(is_EBC), np.nan)))
    rbc_rf_corr = np.asarray(cd.get('rbc_rf_corr', np.full(len(is_RBC), np.nan)))

    n = len(area_id)
    records = [{
        'area_id':               int(area_id[i]),
        'area_name':             ID_TO_NAME.get(int(area_id[i])),
        'vfs_x':                 float(vfs_x[i]),
        'vfs_y':                 float(vfs_y[i]),
        'is_EBC':                bool(is_EBC[i]),
        'is_RBC':                bool(is_RBC[i]),
        'is_fully_reliable_EBC': bool(is_fr_e[i]),
        'is_fully_reliable_RBC': bool(is_fr_r[i]),
        'ebc_mrl':               float(ebc_mrl[i]),
        'rbc_mrl':               float(rbc_mrl[i]),
        'ebc_rf_corr':           float(ebc_rf_corr[i]),
        'rbc_rf_corr':           float(rbc_rf_corr[i]),
    } for i in range(n)]

    params = topo.get('boundary_params', None)
    if params is not None:
        params = {k: np.asarray(v) for k, v in params.items()}

    return topo, records, params


def _add_scatter_col(ax, pos, vals, color='k'):
    vals = np.asarray([v for v in vals if not np.isnan(v)])
    if not len(vals):
        return
    rng = np.random.default_rng(pos)
    ax.scatter(np.ones(len(vals)) * pos + (rng.random(len(vals)) - 0.5) * 0.4,
               vals, s=3, c=color, alpha=0.6, linewidths=0)
    mn  = np.nanmean(vals)
    sem = np.nanstd(vals) / np.sqrt(len(vals))
    ax.hlines(mn, pos - 0.15, pos + 0.15, color='k', linewidth=1.5)
    ax.vlines(pos, mn - sem, mn + sem, color='k', linewidth=1.5)


def _plot_polar_map(ax, rate_map, theta_edges, r_edges, vmax=None):
    rm = np.asarray(rate_map, dtype=float)
    vmax = vmax or (np.nanpercentile(rm, 99) if not np.all(np.isnan(rm)) else 1.)
    ax.pcolormesh(theta_edges, r_edges, rm.T,
                  cmap='inferno', vmin=0, vmax=max(vmax, 1e-9), shading='flat')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels(['', '', '', ''], fontsize=5)
    ax.tick_params(axis='both', labelsize=5)


def _areas_present(records):
    present = {r['area_name'] for r in records if r['area_name'] is not None}
    return [a for a in REGION_ORDER if a in present]


def _recording_proportions(records, flag_key):

    from collections import defaultdict
    by_area = defaultdict(list)
    for r in records:
        if r['area_name']:
            by_area[r['area_name']].append(r[flag_key])
    return {a: [np.mean(v)] for a, v in by_area.items()}


def plot_cell_counts(records, pdf):
    areas = _areas_present(records)
    if not areas:
        return
    rows = []
    for area in areas:
        ar = [r for r in records if r['area_name'] == area]
        n  = len(ar)
        ne = sum(r['is_EBC'] for r in ar)
        nr = sum(r['is_RBC'] for r in ar)
        nb = sum(r['is_EBC'] and r['is_RBC'] for r in ar)
        nfe = sum(r['is_fully_reliable_EBC'] for r in ar)
        nfr = sum(r['is_fully_reliable_RBC'] for r in ar)
        rows.append([area, n,
                     ne, f'{100*ne/n:.1f}%' if n else '—',
                     nr, f'{100*nr/n:.1f}%' if n else '—',
                     nb, nfe, nfr])

    cols = ['Area', 'N', 'N EBC', '% EBC', 'N RBC', '% RBC',
            'N both', 'N FR-EBC', 'N FR-RBC']
    fig, ax = plt.subplots(figsize=(10, 0.45 * (len(rows) + 2)), dpi=200)
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.4)
    for j in range(len(cols)):
        tbl[(0, j)].set_facecolor('#dddddd')
    fig.suptitle('EBC / RBC cell counts per visual area', fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_proportion_per_area(records, pdf):
    areas = _areas_present(records)
    if not areas:
        return

    fig, ax = plt.subplots(figsize=(6, 3), dpi=200)
    x = np.arange(len(areas))
    w = 0.35

    for i, (label, flag, color) in enumerate([
        ('EBC', 'is_EBC', EBC_COLOR),
        ('RBC', 'is_RBC', RBC_COLOR),
    ]):
        fracs = []
        for area in areas:
            ar = [r for r in records if r['area_name'] == area]
            fracs.append(np.mean([r[flag] for r in ar]) if ar else 0.)
        ax.bar(x + (i - 0.5) * w, fracs, width=w, color=color,
               alpha=0.85, label=label, edgecolor='none')

    ax.set_xticks(x)
    ax.set_xticklabels(areas)
    ax.set_ylabel('Fraction of cells')
    ax.set_ylim(0, None)
    ax.legend(frameon=False)
    ax.set_title('EBC / RBC proportion per visual area')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_classification_breakdown(records, pdf):
    areas = _areas_present(records)
    if not areas:
        return

    cats   = ['EBC only', 'RBC only', 'Both', 'Neither']
    ccolors = [EBC_COLOR, RBC_COLOR, BOTH_COLOR, '#cccccc']

    counts = {a: {c: 0 for c in cats} for a in areas}
    for r in records:
        a = r['area_name']
        if a not in counts:
            continue
        if   r['is_EBC'] and not r['is_RBC']:  counts[a]['EBC only'] += 1
        elif r['is_RBC'] and not r['is_EBC']:  counts[a]['RBC only'] += 1
        elif r['is_EBC'] and r['is_RBC']:      counts[a]['Both']     += 1
        else:                                   counts[a]['Neither']  += 1

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=200)

    for ax, use_frac in zip(axes, [False, True]):
        bottoms = np.zeros(len(areas))
        totals  = np.array([sum(counts[a].values()) for a in areas], dtype=float)
        for cat, cc in zip(cats, ccolors):
            vals = np.array([counts[a][cat] for a in areas], dtype=float)
            plot_vals = (np.where(totals > 0, vals / totals, 0.) if use_frac else vals)
            ax.bar(range(len(areas)), plot_vals, bottom=bottoms,
                   color=cc, label=cat, edgecolor='none')
            bottoms += plot_vals
        ax.set_xticks(range(len(areas)))
        ax.set_xticklabels(areas)
        ax.set_ylabel('Fraction' if use_frac else 'Cell count')
        ax.set_title('Fractions' if use_frac else 'Counts')
        if use_frac:
            ax.set_ylim(0, 1)

    axes[0].legend(frameon=False, fontsize=6)
    fig.suptitle('EBC / RBC classification breakdown per area', fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_mean_rate_maps(topo, params, pdf, cell_type='EBC'):
    if params is None:
        return
    maps_key = f'boundary_{cell_type.lower()}_mean_maps'
    area_maps = topo.get(maps_key, {})
    if not area_maps:
        print(f'  No {maps_key} in topography results — skipping.')
        return

    ray_width   = float(params.get('ray_width', 5.))
    theta_edges = np.deg2rad(np.arange(0, 360 + ray_width, ray_width))
    r_edges     = np.asarray(params['dist_bin_edges'])

    areas  = [a for a in REGION_ORDER if a in area_maps]
    if not areas:
        return
    ncols = min(len(areas), 4)
    nrows = int(np.ceil(len(areas) / ncols))
    color = EBC_COLOR if cell_type == 'EBC' else RBC_COLOR

    fig = plt.figure(figsize=(ncols * 2.8, nrows * 2.8), dpi=200)
    for i, area in enumerate(areas):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='polar')
        rm = np.asarray(area_maps[area])
        vmax = np.nanpercentile(rm, 99) if not np.all(np.isnan(rm)) else 1.
        _plot_polar_map(ax, rm, theta_edges, r_edges, vmax=max(vmax, 1e-9))
        ax.set_title(area, fontsize=8, color=color)

    fig.suptitle(f'Mean smoothed rate map per area — {cell_type}', fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_mrl_per_area(records, pdf):
    areas = _areas_present(records)
    if not areas:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=200)
    for ax, (cell_type, mrl_key, color) in zip(axes, [
        ('EBC', 'ebc_mrl', EBC_COLOR),
        ('RBC', 'rbc_mrl', RBC_COLOR),
    ]):
        flag = f'is_{cell_type}'
        for xi, area in enumerate(areas):
            vals = [r[mrl_key] for r in records
                    if r['area_name'] == area and r[flag] and not np.isnan(r[mrl_key])]
            _add_scatter_col(ax, xi, vals, color=color)

        ax.set_xticks(range(len(areas)))
        ax.set_xticklabels(areas)
        ax.set_ylabel('MRL')
        ax.set_title(f'{cell_type} MRL per area (classified cells only)')

        grouped = [[r[mrl_key] for r in records
                    if r['area_name'] == a and r[flag] and not np.isnan(r[mrl_key])]
                   for a in areas]
        grouped = [g for g in grouped if len(g) > 1]
        if len(grouped) >= 2:
            try:
                _, pval = kruskal(*grouped)
                ax.set_xlabel(f'Kruskal–Wallis p = {pval:.3f}', fontsize=6)
            except Exception:
                pass

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_rf_corr_per_area(records, pdf):

    areas = _areas_present(records)
    if not areas:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=200)
    for ax, (cell_type, corr_key, color) in zip(axes, [
        ('EBC', 'ebc_rf_corr', EBC_COLOR),
        ('RBC', 'rbc_rf_corr', RBC_COLOR),
    ]):
        flag = f'is_{cell_type}'
        for xi, area in enumerate(areas):
            vals = [r[corr_key] for r in records
                    if r['area_name'] == area and r[flag] and not np.isnan(r[corr_key])]
            _add_scatter_col(ax, xi, vals, color=color)

        ax.axhline(0.6, color='k', ls='--', lw=0.8, alpha=0.6)
        ax.set_xticks(range(len(areas)))
        ax.set_xticklabels(areas)
        ax.set_ylabel('RF split-half correlation')
        ax.set_title(f'{cell_type} RF corr per area (classified cells only)')

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_ebc_vs_rbc_mrl(records, pdf):
    x = np.array([r['ebc_mrl'] for r in records])
    y = np.array([r['rbc_mrl'] for r in records])
    valid = ~(np.isnan(x) | np.isnan(y))
    if not valid.any():
        return

    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
    for area in REGION_ORDER:
        mask = valid & np.array([r['area_name'] == area for r in records])
        if not mask.any():
            continue
        ax.scatter(x[mask], y[mask], s=4, c=COLORS.get(area, 'k'),
                   alpha=0.6, linewidths=0, label=area)
    lim = max(np.nanmax(x[valid]), np.nanmax(y[valid])) * 1.05
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.plot([0, lim], [0, lim], 'k--', linewidth=0.5, alpha=0.4)
    ax.set_xlabel('EBC MRL'); ax.set_ylabel('RBC MRL')
    ax.set_title('EBC vs RBC MRL (coloured by area)')
    ax.legend(markerscale=2, frameon=False, fontsize=6, ncol=2)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_vfs_scatter(records, labeled_array, pdf):

    x = np.array([r['vfs_x'] for r in records])
    y = np.array([r['vfs_y'] for r in records])
    valid = ~(np.isnan(x) | np.isnan(y))

    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    if labeled_array is not None:
        ax.imshow(labeled_array, cmap='tab10', alpha=0.15,
                  origin='upper', vmin=0, vmax=10)
        for aid in np.unique(labeled_array):
            if aid <= 1:
                continue
            mask = (labeled_array == aid).astype(float)
            ax.contour(mask, levels=[0.5], colors='k', linewidths=0.5, alpha=0.5)
            ys, xs = np.where(labeled_array == aid)
            ax.text(np.mean(xs), np.mean(ys), ID_TO_NAME.get(int(aid), '?'),
                    ha='center', va='center', fontsize=7, fontweight='bold')

    neither = valid & ~np.array([r['is_EBC'] or r['is_RBC'] for r in records])
    ax.scatter(x[neither], y[neither], s=2, c='#cccccc', alpha=0.3,
               linewidths=0, label='Neither')
    for label, fe, fr, color in [
        ('EBC only',  True,  False, EBC_COLOR),
        ('RBC only',  False, True,  RBC_COLOR),
        ('Both',      True,  True,  BOTH_COLOR),
    ]:
        mask = valid & np.array(
            [r['is_EBC'] == fe and r['is_RBC'] == fr for r in records])
        ax.scatter(x[mask], y[mask], s=8, c=color, alpha=0.8,
                   linewidths=0, label=label)

    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.legend(markerscale=2, frameon=False, fontsize=6)
    ax.set_title('EBC / RBC cells in VFS reference space')
    ax.set_xlabel('VFS x (px)'); ax.set_ylabel('VFS y (px)')
    ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_spatial_proportion_map(records, labeled_array, pdf, cell_type='EBC'):

    from scipy.ndimage import gaussian_filter
    flag_key = f'is_{cell_type}'
    color    = EBC_COLOR if cell_type == 'EBC' else RBC_COLOR

    recs = [r for r in records if not np.isnan(r['vfs_x']) and not np.isnan(r['vfs_y'])]
    if not recs or labeled_array is None:
        return

    h, w = labeled_array.shape
    count_map = np.zeros((h, w))
    bc_map    = np.zeros((h, w))
    for r in recs:
        xi = int(np.clip(round(r['vfs_x']), 0, w - 1))
        yi = int(np.clip(round(r['vfs_y']), 0, h - 1))
        count_map[yi, xi] += 1
        if r[flag_key]:
            bc_map[yi, xi] += 1

    sigma   = 10
    count_s = gaussian_filter(count_map.astype(float), sigma)
    bc_s    = gaussian_filter(bc_map.astype(float),    sigma)
    prop    = np.where(count_s > 0.5, bc_s / count_s, np.nan)
    area_px = labeled_array > 1
    prop_m  = np.where(area_px, prop, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=200)

    vmax = np.nanpercentile(prop_m[area_px], 99) if area_px.any() else 1.
    im = axes[0].imshow(prop_m, cmap='hot', vmin=0, vmax=vmax, origin='upper')
    for aid in np.unique(labeled_array):
        if aid <= 1: continue
        mask = (labeled_array == aid).astype(float)
        axes[0].contour(mask, levels=[0.5], colors='white', linewidths=0.5, alpha=0.7)
    plt.colorbar(im, ax=axes[0], shrink=0.7, label='Proportion')
    axes[0].set_title(f'{cell_type} proportion map (smoothed)')
    axes[0].axis('off')

    areas = _areas_present(records)
    fracs = []
    for area in areas:
        ar = [r for r in records if r['area_name'] == area]
        fracs.append(np.mean([r[flag_key] for r in ar]) if ar else 0.)
    axes[1].bar(range(len(areas)), fracs, color=color, alpha=0.85, edgecolor='none')
    axes[1].set_xticks(range(len(areas)))
    axes[1].set_xticklabels(areas)
    axes[1].set_ylabel('Fraction of cells')
    axes[1].set_title(f'{cell_type} proportion per area')

    fig.suptitle(f'{cell_type} spatial distribution', fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def make_boundary_plots(topo_h5_path, out_pdf_path):

    print(f'Loading topography results from {topo_h5_path}...')
    topo, records, params = load_topo_results(topo_h5_path)
    labeled_array = topo.get('labeled_array', None)
    if labeled_array is not None:
        labeled_array = np.asarray(labeled_array)

    n_ebc = sum(r['is_EBC'] for r in records)
    n_rbc = sum(r['is_RBC'] for r in records)
    print(f'  {len(records)} cells: {n_ebc} EBC, {n_rbc} RBC')

    with PdfPages(out_pdf_path) as pdf:
        print('  Cell counts table...')
        plot_cell_counts(records, pdf)

        print('  Proportions per area...')
        plot_proportion_per_area(records, pdf)

        print('  Classification breakdown...')
        plot_classification_breakdown(records, pdf)

        print('  VFS scatter...')
        plot_vfs_scatter(records, labeled_array, pdf)

        print('  Spatial proportion maps...')
        plot_spatial_proportion_map(records, labeled_array, pdf, cell_type='EBC')
        plot_spatial_proportion_map(records, labeled_array, pdf, cell_type='RBC')

        print('  MRL per area...')
        plot_mrl_per_area(records, pdf)

        print('  RF corr per area...')
        plot_rf_corr_per_area(records, pdf)

        print('  EBC vs RBC MRL scatter...')
        plot_ebc_vs_rbc_mrl(records, pdf)

        print('  Mean EBC rate maps per area...')
        plot_mean_rate_maps(topo, params, pdf, cell_type='EBC')

        print('  Mean RBC rate maps per area...')
        plot_mean_rate_maps(topo, params, pdf, cell_type='RBC')

    print(f'Saved -> {out_pdf_path}')


def main():
    topo_h5 = '/home/dylan/Fast2/topography_analysis_results_260331a.h5'
    out_pdf = '/home/dylan/Fast2/boundary_plots_260331a.pdf'
    make_boundary_plots(topo_h5, out_pdf)


if __name__ == '__main__':
    main()
