# -*- coding: utf-8 -*-
"""
fm2p/utils/boundary_plots.py

Summary figures describing the distribution of EBC and RBC cells across visual
areas, generated from the topography analysis HDF5.

Requires the topography results h5 to contain keys written by
aggregate_boundary_data(): boundary_cell_data, boundary_ebc_mean_maps,
boundary_rbc_mean_maps, boundary_params, labeled_array.

Usage: edit paths in main() and run:
    python -m fm2p.utils.boundary_plots

Functions
---------
load_topo_results
    Parse topography HDF5 into per-cell records and parameter dict.
plot_cell_counts
    Table of N / %EBC / %RBC per area.
plot_proportion_per_area
    Grouped bar chart of EBC/RBC fractions per area.
plot_classification_breakdown
    Stacked bar chart (EBC only / RBC only / Both / Neither).
plot_mean_rate_maps
    Polar rate map averaged across cells per area.
plot_mrl_per_area
    Scatter of MRL values per area with Kruskal-Wallis p-value.
plot_rf_corr_per_area
    Scatter of split-half RF correlations per area.
plot_ebc_vs_rbc_mrl
    EBC MRL vs RBC MRL coloured by area.
plot_vfs_scatter
    All cells in VFS space, coloured by area.
plot_spatial_proportion_map
    Smoothed proportion map in VFS space plus bar chart.
plot_light_dependence
    Fraction of classified cells that are light-dependent per area.
make_boundary_plots
    Run all plots and save to a single PDF.


DMM, April 2026
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
    """ Parse topography HDF5 into per-cell records and parameter dict.

    Parameters
    ----------
    topo_h5_path : str
        Path to HDF5 file produced by topography.py aggregate_boundary_data().

    Returns
    -------
    topo : dict
        Raw contents of the HDF5.
    records : list of dict
        One dict per cell with area assignment and classification flags.
    params : dict or None
        Boundary analysis parameters (ray_width, dist_bin_edges, etc.).
    """

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
    ebc_rf_corr = np.asarray(cd.get('ebc_rf_corr_shfl_99pct', np.full(len(is_EBC), np.nan)))
    rbc_rf_corr = np.asarray(cd.get('rbc_rf_corr_shfl_99pct', np.full(len(is_RBC), np.nan)))

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
    """ Jittered scatter at x=pos with mean +/- SEM crosshairs. """

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
    """ Render a polar rate map onto a polar axes. """

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
    """ Ordered list of area names that appear in records. """

    present = {r['area_name'] for r in records if r['area_name'] is not None}
    return [a for a in REGION_ORDER if a in present]


def _recording_proportions(records, flag_key):
    """ Per-area mean of a boolean flag, as a single-element list (for plotting). """

    by_area = defaultdict(list)
    for r in records:
        if r['area_name']:
            by_area[r['area_name']].append(r[flag_key])
    return {a: [np.mean(v)] for a, v in by_area.items()}


def plot_cell_counts(records, pdf):
    """ Table of N, N EBC, % EBC, N RBC, % RBC, N both per area.

    Parameters
    ----------
    records : list of dict
        Per-cell records from load_topo_results.
    pdf : PdfPages
        Open PDF to append to.
    """

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
                     ne, '{:.1f}%'.format(100 * ne / n) if n else '-',
                     nr, '{:.1f}%'.format(100 * nr / n) if n else '-',
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
    """ Grouped bar chart of EBC and RBC fractions per visual area.

    Parameters
    ----------
    records : list of dict
        Per-cell records from load_topo_results.
    pdf : PdfPages
        Open PDF to append to.
    """

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
    """ Stacked bar chart of EBC-only / RBC-only / Both / Neither per area.

    Parameters
    ----------
    records : list of dict
        Per-cell records from load_topo_results.
    pdf : PdfPages
        Open PDF to append to.
    """

    areas = _areas_present(records)
    if not areas:
        return

    cats    = ['EBC only', 'RBC only', 'Both', 'Neither']
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
    """ Polar rate map averaged across classified cells, one panel per area.

    Parameters
    ----------
    topo : dict
        Raw topography HDF5 contents.
    params : dict or None
        Analysis parameters with 'ray_width' and 'dist_bin_edges'.
    pdf : PdfPages
        Open PDF to append to.
    cell_type : str
        'EBC' or 'RBC'.
    """

    if params is None:
        return
    maps_key  = 'boundary_{}_mean_maps'.format(cell_type.lower())
    area_maps = topo.get(maps_key, {})
    if not area_maps:
        print('  No {} in topography results -- skipping.'.format(maps_key))
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

    fig.suptitle('Mean smoothed rate map per area -- {}'.format(cell_type), fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_mrl_per_area(records, pdf):
    """ Jittered scatter of MRL values per area with Kruskal-Wallis p-value.

    Only classified cells (is_EBC or is_RBC) are included in the scatter.

    Parameters
    ----------
    records : list of dict
        Per-cell records from load_topo_results.
    pdf : PdfPages
        Open PDF to append to.
    """

    areas = _areas_present(records)
    if not areas:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=200)
    for ax, (cell_type, mrl_key, color) in zip(axes, [
        ('EBC', 'ebc_mrl', EBC_COLOR),
        ('RBC', 'rbc_mrl', RBC_COLOR),
    ]):
        flag = 'is_{}'.format(cell_type)
        for xi, area in enumerate(areas):
            vals = [r[mrl_key] for r in records
                    if r['area_name'] == area and r[flag] and not np.isnan(r[mrl_key])]
            _add_scatter_col(ax, xi, vals, color=color)

        ax.set_xticks(range(len(areas)))
        ax.set_xticklabels(areas)
        ax.set_ylabel('MRL')
        ax.set_title('{} MRL per area (classified cells only)'.format(cell_type))

        grouped = [[r[mrl_key] for r in records
                    if r['area_name'] == a and r[flag] and not np.isnan(r[mrl_key])]
                   for a in areas]
        grouped = [g for g in grouped if len(g) > 1]
        if len(grouped) >= 2:
            try:
                _, pval = kruskal(*grouped)
                ax.set_xlabel('Kruskal-Wallis p = {:.3f}'.format(pval), fontsize=6)
            except Exception:
                pass

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_rf_corr_per_area(records, pdf):
    """ Jittered scatter of split-half RF correlations per area.

    Parameters
    ----------
    records : list of dict
        Per-cell records from load_topo_results.
    pdf : PdfPages
        Open PDF to append to.
    """

    areas = _areas_present(records)
    if not areas:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=200)
    for ax, (cell_type, corr_key, color) in zip(axes, [
        ('EBC', 'ebc_rf_corr', EBC_COLOR),
        ('RBC', 'rbc_rf_corr', RBC_COLOR),
    ]):
        flag = 'is_{}'.format(cell_type)
        for xi, area in enumerate(areas):
            vals = [r[corr_key] for r in records
                    if r['area_name'] == area and r[flag] and not np.isnan(r[corr_key])]
            _add_scatter_col(ax, xi, vals, color=color)

        ax.axhline(0.5, color='k', ls='--', lw=0.8, alpha=0.6)
        ax.set_xticks(range(len(areas)))
        ax.set_xticklabels(areas)
        ax.set_ylabel('RF split-half correlation')
        ax.set_title('{} RF corr per area (classified cells only)'.format(cell_type))

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_ebc_vs_rbc_mrl(records, pdf):
    """ Scatter of EBC MRL vs RBC MRL, coloured by visual area.

    Parameters
    ----------
    records : list of dict
        Per-cell records from load_topo_results.
    pdf : PdfPages
        Open PDF to append to.
    """

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
    """ All cells in VFS space, coloured by area; reliable cells at full opacity.

    Parameters
    ----------
    records : list of dict
        Per-cell records from load_topo_results.
    labeled_array : np.ndarray or None
        Area label map in VFS space.
    pdf : PdfPages
        Open PDF to append to.
    """

    x = np.array([r['vfs_x'] for r in records])
    y = np.array([r['vfs_y'] for r in records])
    valid = ~(np.isnan(x) | np.isnan(y))
    is_reliable = np.array([r['is_EBC'] or r['is_RBC'] for r in records])

    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)

    # Unreliable cells underneath so reliable ones are visible on top.
    for area in REGION_ORDER:
        color = COLORS.get(area, '#999999')
        mask = valid & ~is_reliable & np.array([r['area_name'] == area for r in records])
        if mask.any():
            ax.scatter(x[mask], y[mask], s=3, c=color, alpha=0.08,
                       linewidths=0)

    no_area_unrel = valid & ~is_reliable & np.array([r['area_name'] is None for r in records])
    if no_area_unrel.any():
        ax.scatter(x[no_area_unrel], y[no_area_unrel], s=3, c='#bbbbbb',
                   alpha=0.08, linewidths=0)

    for area in REGION_ORDER:
        color = COLORS.get(area, '#999999')
        mask = valid & is_reliable & np.array([r['area_name'] == area for r in records])
        if mask.any():
            ax.scatter(x[mask], y[mask], s=8, c=color, alpha=0.85,
                       linewidths=0, label=area)

    no_area_rel = valid & is_reliable & np.array([r['area_name'] is None for r in records])
    if no_area_rel.any():
        ax.scatter(x[no_area_rel], y[no_area_rel], s=8, c='#bbbbbb',
                   alpha=0.85, linewidths=0)

    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.legend(markerscale=2, frameon=False, fontsize=6, title='Area')
    ax.set_title('EBC / RBC cells in VFS space (colour = area, faint = unreliable)')
    ax.set_xlabel('VFS x (px)'); ax.set_ylabel('VFS y (px)')
    ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_spatial_proportion_map(records, labeled_array, pdf, cell_type='EBC'):
    """ Smoothed proportion map in VFS space plus bar chart per area.

    Projects cell positions onto the VFS image, counts classified vs. total
    per pixel, Gaussian-smooths, and masks outside labeled areas.

    Parameters
    ----------
    records : list of dict
        Per-cell records from load_topo_results.
    labeled_array : np.ndarray or None
        Area label map in VFS space.
    pdf : PdfPages
        Open PDF to append to.
    cell_type : str
        'EBC' or 'RBC'.
    """

    from scipy.ndimage import gaussian_filter
    flag_key = 'is_{}'.format(cell_type)
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
    # Only compute proportion where smoothed count is meaningful.
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
    axes[0].set_title('{} proportion map (smoothed)'.format(cell_type))
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
    axes[1].set_title('{} proportion per area'.format(cell_type))

    fig.suptitle('{} spatial distribution'.format(cell_type), fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_light_dependence(records, pdf):
    """ Fraction of classified cells that are light-dependent per area.

    'Light-dependent' means is_EBC/is_RBC=True but is_fully_reliable=False,
    i.e. reliable under light but not in the dark condition.

    Parameters
    ----------
    records : list of dict
        Per-cell records from load_topo_results.
    pdf : PdfPages
        Open PDF to append to.
    """

    areas = _areas_present(records)
    if not areas:
        return

    # If every cell has is_fully_reliable == is_EBC/RBC, the dark condition
    # was never run and there is nothing to plot.
    any_dark_ebc = any(r['is_EBC'] and not r['is_fully_reliable_EBC'] for r in records)
    any_dark_rbc = any(r['is_RBC'] and not r['is_fully_reliable_RBC'] for r in records)
    if not any_dark_ebc and not any_dark_rbc:
        print('  No dark-condition data found -- skipping light-dependence plot.')
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=200)

    for ax, (cell_type, flag, fr_flag, color) in zip(axes, [
        ('EBC', 'is_EBC', 'is_fully_reliable_EBC', EBC_COLOR),
        ('RBC', 'is_RBC', 'is_fully_reliable_RBC', RBC_COLOR),
    ]):
        fracs, ns = [], []
        for area in areas:
            classified = [r for r in records if r['area_name'] == area and r[flag]]
            n = len(classified)
            ns.append(n)
            if n:
                n_light_dep = sum(not r[fr_flag] for r in classified)
                fracs.append(n_light_dep / n)
            else:
                fracs.append(0.)

        x = np.arange(len(areas))
        ax.bar(x, fracs, color=color, alpha=0.85, edgecolor='none')
        ax.set_xticks(x)
        ax.set_xticklabels(
            ['{}\n(n={})'.format(a, ns[i]) for i, a in enumerate(areas)], fontsize=6)
        ax.set_ylabel('Fraction light-dependent')
        ax.set_ylim(0, 1)
        ax.set_title('{}: reliable in light, not in dark'.format(cell_type))

    fig.suptitle('Light-dependent boundary cells per area', fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def make_boundary_plots(topo_h5_path, out_pdf_path):
    """ Run all boundary summary plots and save to a single PDF.

    Parameters
    ----------
    topo_h5_path : str
        Path to topography results HDF5.
    out_pdf_path : str
        Destination PDF path.
    """

    print('Loading topography results from {}...'.format(topo_h5_path))
    topo, records, params = load_topo_results(topo_h5_path)
    labeled_array = topo.get('labeled_array', None)
    if labeled_array is not None:
        labeled_array = np.asarray(labeled_array)

    n_ebc = sum(r['is_EBC'] for r in records)
    n_rbc = sum(r['is_RBC'] for r in records)
    print('  {} cells: {} EBC, {} RBC'.format(len(records), n_ebc, n_rbc))

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

        print('  Light dependence...')
        plot_light_dependence(records, pdf)

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

    print('Saved to {}.'.format(out_pdf_path))


def main():

    topo_h5 = '/home/dylan/Fast2/topography_analysis_results_260331a.h5'
    out_pdf = '/home/dylan/Fast2/boundary_plots_260331a.pdf'
    make_boundary_plots(topo_h5, out_pdf)


if __name__ == '__main__':
    main()
