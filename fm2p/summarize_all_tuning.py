

if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

import argparse
import os

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch, Polygon as MPoly

import matplotlib as mpl
mpl.rcParams['axes.spines.top']   = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['font.size']    = 7

from .utils.paths import find
from .summarize_head_tuning import (
    _build_pooled_lookup, _match_to_pooled, _norm01,
)


DEFAULT_POOLED  = '/home/dylan/Fast2/pooled_260407a.h5'
DEFAULT_BASE    = '/home/dylan/Storage/freely_moving_data/_V1PPC'
DEFAULT_OUT_DIR = '.'
MIN_CELLS_AREA  = 5
TOP_N_HEATMAP   = 100
TOP_N_PER_AREA  = 24
MOD_THRESHOLD   = 0.33

ID_TO_NAME   = {5: 'V1', 2: 'RL', 3: 'AM', 4: 'PM', 10: 'A', 7: 'AL', 8: 'LM', 9: 'P'}
REGION_ORDER = ['V1', 'RL', 'AM', 'PM', 'A', 'AL', 'LM', 'P']
COLORS = {
    'V1': '#1B9E77', 'RL': '#D95F02', 'AM': '#7570B3', 'PM': '#E7298A',
    'A':  '#1A5A34', 'AL': '#E6AB02', 'LM': '#A6761D', 'P':  '#666666',
}

VARIABLES = [
    dict(name='theta', label=r'θ (eye horiz.)', is_imu=False),
    dict(name='phi',   label=r'φ (eye vert.)',  is_imu=False),
    dict(name='pitch', label='Pitch',            is_imu=True),
    dict(name='roll',  label='Roll',             is_imu=True),
    dict(name='yaw',   label='Yaw',              is_imu=True),
]
VAR_NAMES = [v['name'] for v in VARIABLES]

_HATCH = '////'   # 45-degree hatch marks for dark condition


def collect_data(pooled_path: str, base_dir: str) -> list:
    """Load both light and dark tuning data in one pass."""
    pooled_lookup = _build_pooled_lookup(pooled_path)
    revcorr_files = find('eyehead_revcorrs_v06.h5', base_dir)
    print(f'Found {len(revcorr_files)} eyehead_revcorrs_v06.h5 files.')

    all_cells = []

    for rcf in sorted(revcorr_files):
        try:
            with h5py.File(rcf, 'r') as f:
                if not any(f'{v}_l_rel' in f for v in VAR_NAMES):
                    continue

                n_cells_f = None
                var_data  = {}

                for vname in VAR_NAMES:
                    l_rel_key  = f'{vname}_l_rel'
                    d_rel_key  = f'{vname}_d_rel'
                    tuning_key = f'{vname}_1dtuning'
                    err_key    = f'{vname}_1derr'
                    bins_key   = f'{vname}_1dbins'

                    if l_rel_key not in f:
                        var_data[vname] = None
                        continue

                    l_rel  = f[l_rel_key][()].astype(float)
                    d_rel  = (f[d_rel_key][()].astype(float)
                              if d_rel_key in f
                              else np.full_like(l_rel, np.nan))
                    isrel  = (f[f'{vname}_l_isrel'][()].astype(bool)
                              if f'{vname}_l_isrel' in f
                              else np.zeros(len(l_rel), dtype=bool))
                    tuning = f[tuning_key][()].astype(float) if tuning_key in f else None
                    err    = f[err_key][()].astype(float)    if err_key    in f else None
                    bins   = f[bins_key][()].astype(float)   if bins_key   in f else None

                    var_data[vname] = dict(
                        l_rel=l_rel, d_rel=d_rel, isrel=isrel,
                        tuning=tuning, err=err, bins=bins,
                    )
                    if n_cells_f is None:
                        n_cells_f = len(l_rel)

        except Exception as e:
            print(f'  Read error {rcf}: {e}')
            continue

        if n_cells_f is None:
            continue

        match = _match_to_pooled(rcf, pooled_lookup)
        if match is None:
            print(f'  No pooled match: {rcf}')
            continue

        animal, pos, va_ids = match
        n_cells = len(va_ids)

        for vname in list(var_data):
            vd = var_data[vname]
            if vd is not None and len(vd['l_rel']) != n_cells:
                print(f'  {vname} cell count mismatch '
                      f'({len(vd["l_rel"])} vs {n_cells}), dropping for {rcf}')
                var_data[vname] = None

        named = {ID_TO_NAME[i] for i in np.unique(va_ids) if i in ID_TO_NAME}
        print(f'  {animal}/{pos}: {n_cells} cells  areas={sorted(named)}')

        for ci in range(n_cells):
            area_id = int(va_ids[ci])
            if area_id not in ID_TO_NAME:
                continue
            area = ID_TO_NAME[area_id]

            cell = dict(animal=animal, pos=pos, area=area, area_id=area_id)

            for vname in VAR_NAMES:
                vd = var_data.get(vname)
                if vd is None:
                    cell[f'{vname}_rel']          = np.nan
                    cell[f'{vname}_rel_dark']      = np.nan
                    cell[f'{vname}_isrel']         = False
                    cell[f'{vname}_tuning']        = None
                    cell[f'{vname}_tuning_dark']   = None
                    cell[f'{vname}_err']           = None
                    cell[f'{vname}_err_dark']      = None
                    cell[f'{vname}_bins']          = None
                else:
                    cell[f'{vname}_rel']      = float(vd['l_rel'][ci])
                    cell[f'{vname}_rel_dark'] = float(vd['d_rel'][ci])
                    cell[f'{vname}_isrel']    = bool(vd['isrel'][ci])
                    if vd['tuning'] is not None:
                        cell[f'{vname}_tuning']      = vd['tuning'][ci, :, 1].copy()
                        cell[f'{vname}_tuning_dark'] = vd['tuning'][ci, :, 0].copy()
                        cell[f'{vname}_err']         = vd['err'][ci, :, 1].copy()
                        cell[f'{vname}_err_dark']    = vd['err'][ci, :, 0].copy()
                    else:
                        cell[f'{vname}_tuning']      = None
                        cell[f'{vname}_tuning_dark'] = None
                        cell[f'{vname}_err']         = None
                        cell[f'{vname}_err_dark']    = None
                    cell[f'{vname}_bins'] = vd['bins'].copy() if vd['bins'] is not None else None

            all_cells.append(cell)

    print(f'Total cells with named area: {len(all_cells)}')
    return all_cells


def _split_by_imu(all_cells):
    imu_vars = ['pitch', 'roll', 'yaw']
    recordings_with_imu = {
        (c['animal'], c['pos'])
        for c in all_cells
        if any(np.isfinite(c[f'{v}_rel']) for v in imu_vars)
    }
    imu_cells    = [c for c in all_cells
                    if (c['animal'], c['pos'])     in recordings_with_imu]
    no_imu_cells = [c for c in all_cells
                    if (c['animal'], c['pos']) not in recordings_with_imu]
    return imu_cells, no_imu_cells


def _ldi(light_mi, dark_mi):
    """Light-Dependence Index: 1=light-only, 0.5=equal, 0=dark-only."""
    if not (np.isfinite(light_mi) and np.isfinite(dark_mi)):
        return np.nan
    denom = light_mi + dark_mi
    if denom == 0:
        return np.nan
    return light_mi / denom


def _hatch_polygon(ax, bins, lo, hi, color, alpha=0.20):
    """Hatched fill-between polygon used for the dark condition."""
    vx = np.concatenate([bins, bins[::-1]])
    vy = np.concatenate([lo,   hi[::-1]])
    poly = MPoly(
        np.column_stack([vx, vy]),
        facecolor=color, edgecolor=color, linewidth=0.5,
        hatch=_HATCH, alpha=alpha, zorder=2,
    )
    ax.add_patch(poly)


def _add_legend(fig):
    handles = [
        Patch(facecolor='0.6', edgecolor='k', linewidth=0.7, label='Light'),
        Patch(facecolor='0.6', edgecolor='k', linewidth=0.7,
              hatch=_HATCH, label='Dark'),
    ]
    fig.legend(handles=handles, loc='upper right', fontsize=6,
               framealpha=0.8, handlelength=1.8, handleheight=1.0,
               borderpad=0.5)


def _violin_ax_combined(ax, all_cells, vspec):
    """Light (solid) and dark (hatched) violins side-by-side for each area."""
    vname = vspec['name']
    area_vals_l = {a: [] for a in REGION_ORDER}
    area_vals_d = {a: [] for a in REGION_ORDER}
    area_n      = {a: 0  for a in REGION_ORDER}

    for c in all_cells:
        if c['area'] not in area_n:
            continue
        area_n[c['area']] += 1
        rl = c[f'{vname}_rel']
        rd = c[f'{vname}_rel_dark']
        if np.isfinite(rl):
            area_vals_l[c['area']].append(rl)
        if np.isfinite(rd):
            area_vals_d[c['area']].append(rd)

    areas_present = [a for a in REGION_ORDER if area_n[a] >= MIN_CELLS_AREA]
    if not areas_present:
        ax.set_visible(False)
        return []

    off = 0.22
    vw  = 0.38

    for xi, a in enumerate(areas_present):
        color = COLORS.get(a, '#888888')
        for vals_list, xpos, do_hatch in [
            (area_vals_l[a], xi - off, False),
            (area_vals_d[a], xi + off, True),
        ]:
            vals = np.array(vals_list)
            if len(vals) >= 2:
                parts = ax.violinplot([vals], positions=[xpos],
                                      widths=vw, showmedians=False, showextrema=False)
                body = parts['bodies'][0]
                body.set_facecolor(color)
                body.set_edgecolor('k')
                body.set_linewidth(0.5)
                body.set_alpha(0.75 if not do_hatch else 0.50)
                if do_hatch:
                    body.set_hatch(_HATCH)
            if len(vals) >= 1:
                med = np.nanmedian(vals)
                q25, q75 = np.nanpercentile(vals, [25, 75])
                ax.vlines(xpos, q25, q75, colors='k', linewidths=2.0, zorder=4)
                ax.scatter([xpos], [med], s=14, color='w', edgecolors='k',
                           linewidths=0.7, zorder=5)
        ax.text(xi, -0.01, f'n={area_n[a]}', ha='center', va='top',
                fontsize=4, color='0.4', transform=ax.get_xaxis_transform())

    ax.set_xticks(range(len(areas_present)))
    ax.set_xticklabels(areas_present, fontsize=6)
    ax.set_title(vspec['label'], fontsize=8)
    ax.set_ylabel('CV MI', fontsize=7)
    ax.set_xlim(-0.6, len(areas_present) - 0.4)
    ax.axhline(0, color='0.7', lw=0.8, ls='--')
    return areas_present


def make_violin_page(pdf, all_cells):
    nv  = len(VARIABLES)
    fig, axes = plt.subplots(1, nv, figsize=(nv * 2.2 + 0.5, 3.8), dpi=300)
    for ax, vspec in zip(axes, VARIABLES):
        _violin_ax_combined(ax, all_cells, vspec)
    _add_legend(fig)
    fig.suptitle(
        'Tuning reliability (CV modulation index) by visual area'
        '  (solid = light · hatched = dark)',
        fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def _fraction_ax_combined(ax, all_cells, vspec, threshold):
    vname  = vspec['name']
    is_imu = vspec['is_imu']

    area_total   = {a: 0 for a in REGION_ORDER}
    area_valid_l = {a: 0 for a in REGION_ORDER}
    area_valid_d = {a: 0 for a in REGION_ORDER}
    area_above_l = {a: 0 for a in REGION_ORDER}
    area_above_d = {a: 0 for a in REGION_ORDER}

    for c in all_cells:
        if c['area'] not in area_total:
            continue
        area_total[c['area']] += 1
        rl = c[f'{vname}_rel']
        rd = c[f'{vname}_rel_dark']
        if np.isfinite(rl):
            area_valid_l[c['area']] += 1
            if rl > threshold:
                area_above_l[c['area']] += 1
        if np.isfinite(rd):
            area_valid_d[c['area']] += 1
            if rd > threshold:
                area_above_d[c['area']] += 1

    areas_present = [a for a in REGION_ORDER if area_total[a] >= MIN_CELLS_AREA]
    if not areas_present:
        ax.set_visible(False)
        return

    w        = 0.35
    xs       = np.arange(len(areas_present))
    max_frac = 0.0

    for xi, a in enumerate(areas_present):
        color  = COLORS.get(a, '#888888')
        denom_l = area_valid_l[a] if is_imu else area_total[a]
        denom_d = area_valid_d[a] if is_imu else area_total[a]
        fl = area_above_l[a] / denom_l * 100 if denom_l > 0 else 0.0
        fd = area_above_d[a] / denom_d * 100 if denom_d > 0 else 0.0
        max_frac = max(max_frac, fl, fd)

        ax.bar(xi - w / 2, fl, width=w, color=color, edgecolor='k', linewidth=0.5)
        ax.bar(xi + w / 2, fd, width=w, color=color, edgecolor='k', linewidth=0.5,
               hatch=_HATCH)
        ax.text(xi - w / 2, fl + 0.3, f'{fl:.0f}%',
                ha='center', va='bottom', fontsize=4)
        ax.text(xi + w / 2, fd + 0.3, f'{fd:.0f}%',
                ha='center', va='bottom', fontsize=4)
        ax.text(xi, -2.5, f'n={area_total[a]}', ha='center', va='top',
                fontsize=4, color='0.4', transform=ax.get_xaxis_transform())

    ax.set_xticks(xs)
    ax.set_xticklabels(areas_present, fontsize=6)
    ax.set_title(vspec['label'] + (' *' if is_imu else ''), fontsize=8)
    ax.set_ylabel(f'% CV MI > {threshold}', fontsize=7)
    ax.set_xlim(-0.6, len(areas_present) - 0.4)
    ax.set_ylim(0, max(max_frac * 1.3, 10))
    ax.axhline(0, color='k', lw=0.5)


def make_fraction_page(pdf, all_cells, threshold=MOD_THRESHOLD, label=''):
    nv  = len(VARIABLES)
    fig, axes = plt.subplots(1, nv, figsize=(nv * 3.2 + 0.5, 3.2), dpi=300)
    for ax, vspec in zip(axes, VARIABLES):
        _fraction_ax_combined(ax, all_cells, vspec, threshold)
    _add_legend(fig)
    suffix = f'  [{label}]' if label else ''
    fig.suptitle(
        f'% cells with CV MI > {threshold}'
        f'  (* IMU variables: n = cells with IMU data){suffix}'
        '  (solid = light · hatched = dark)',
        fontsize=9)
    fig.tight_layout(w_pad=2.5)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


# ── any-modulated combined ────────────────────────────────────────────────────

def make_any_modulated_page(pdf, all_cells, threshold=MOD_THRESHOLD, label=''):
    area_total = {a: 0 for a in REGION_ORDER}
    area_any_l = {a: 0 for a in REGION_ORDER}
    area_any_d = {a: 0 for a in REGION_ORDER}

    for c in all_cells:
        if c['area'] not in area_total:
            continue
        area_total[c['area']] += 1
        rl_vals = [c[f'{v}_rel']      for v in VAR_NAMES]
        rd_vals = [c[f'{v}_rel_dark'] for v in VAR_NAMES]
        if any(np.isfinite(r) and r > threshold for r in rl_vals):
            area_any_l[c['area']] += 1
        if any(np.isfinite(r) and r > threshold for r in rd_vals):
            area_any_d[c['area']] += 1

    areas_present = [a for a in REGION_ORDER if area_total[a] >= MIN_CELLS_AREA]
    if not areas_present:
        return

    xs     = np.arange(len(areas_present))
    colors = [COLORS.get(a, '#888888') for a in areas_present]
    ns     = [area_total[a] for a in areas_present]
    w      = 0.35

    fig, ax = plt.subplots(figsize=(len(areas_present) * 0.9 + 0.8, 3.2), dpi=300)

    max_frac = 0.0
    for xi, (a, color, n) in enumerate(zip(areas_present, colors, ns)):
        fl = area_any_l[a] / n * 100
        fd = area_any_d[a] / n * 100
        max_frac = max(max_frac, fl, fd)
        ax.bar(xi - w / 2, fl, width=w, color=color, edgecolor='k', linewidth=0.5)
        ax.bar(xi + w / 2, fd, width=w, color=color, edgecolor='k', linewidth=0.5,
               hatch=_HATCH)
        ax.text(xi - w / 2, fl + 0.5, f'{fl:.0f}%', ha='center', va='bottom', fontsize=5)
        ax.text(xi + w / 2, fd + 0.5, f'{fd:.0f}%', ha='center', va='bottom', fontsize=5)
        ax.text(xi, -2.5, f'n={n}', ha='center', va='top',
                fontsize=5, color='0.4', transform=ax.get_xaxis_transform())

    ax.set_xticks(xs)
    ax.set_xticklabels(areas_present, fontsize=8)
    ax.set_ylabel(f'% cells (any variable CV MI > {threshold})', fontsize=8)
    ax.set_xlim(-0.6, len(areas_present) - 0.4)
    ax.set_ylim(0, max(max_frac * 1.3, 10))
    ax.axhline(0, color='k', lw=0.5)

    _add_legend(fig)
    suffix = f'  [{label}]' if label else ''
    fig.suptitle(
        f'% cells tuned to at least one variable  (CV MI > {threshold}){suffix}'
        '  (solid = light · hatched = dark)',
        fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

def make_ldi_page(pdf, all_cells):
    """Violin of LDI distributions by area for each variable."""
    nv = len(VARIABLES)
    fig, axes = plt.subplots(1, nv, figsize=(nv * 2.2 + 0.5, 3.8), dpi=300)

    for ax, vspec in zip(axes, VARIABLES):
        vname     = vspec['name']
        area_vals = {a: [] for a in REGION_ORDER}
        area_n    = {a: 0  for a in REGION_ORDER}

        for c in all_cells:
            if c['area'] not in area_vals:
                continue
            area_n[c['area']] += 1
            ldi_val = _ldi(c[f'{vname}_rel'], c[f'{vname}_rel_dark'])
            if np.isfinite(ldi_val):
                area_vals[c['area']].append(ldi_val)

        areas_present = [a for a in REGION_ORDER if area_n[a] >= MIN_CELLS_AREA]
        if not areas_present:
            ax.set_visible(False)
            continue

        for xi, a in enumerate(areas_present):
            color = COLORS.get(a, '#888888')
            vals  = np.array(area_vals[a])
            if len(vals) >= 2:
                parts = ax.violinplot([vals], positions=[xi],
                                      widths=0.7, showmedians=False, showextrema=False)
                body = parts['bodies'][0]
                body.set_facecolor(color)
                body.set_edgecolor('k')
                body.set_linewidth(0.5)
                body.set_alpha(0.75)
            if len(vals) >= 1:
                med = np.nanmedian(vals)
                q25, q75 = np.nanpercentile(vals, [25, 75])
                ax.vlines(xi, q25, q75, colors='k', linewidths=2.0, zorder=4)
                ax.scatter([xi], [med], s=14, color='w', edgecolors='k',
                           linewidths=0.7, zorder=5)
            ax.text(xi, -0.01, f'n={area_n[a]}', ha='center', va='top',
                    fontsize=4, color='0.4', transform=ax.get_xaxis_transform())

        ax.set_xticks(range(len(areas_present)))
        ax.set_xticklabels(areas_present, fontsize=6)
        ax.set_title(vspec['label'], fontsize=8)
        ax.set_ylabel('LDI', fontsize=7)
        ax.set_xlim(-0.6, len(areas_present) - 0.4)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color='0.7', lw=0.8, ls='--')
        ax.axhline(0,   color='0.5', lw=0.5)

    fig.suptitle(
        r'Light-Dependence Index (LDI) by visual area'
        '\n'
        r'LDI = lightMI / (lightMI + darkMI)'
        '\n'
        'LDI = 1: light-only  ·  LDI = 0.5: equal in both  ·  LDI = 0: dark-only',
        fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def make_ldi_histogram_pages(pdf, all_cells):
    """One page per visual area: horizontal LDI histograms per variable, dashed line at 0.5."""
    n_bins_hist = 20
    bin_edges = np.linspace(0, 1, n_bins_hist + 1)

    for area in REGION_ORDER:
        cells_area = [c for c in all_cells if c['area'] == area]
        if len(cells_area) < MIN_CELLS_AREA:
            continue

        nv = len(VARIABLES)
        fig, axes = plt.subplots(1, nv, figsize=(nv * 2.0, 3.5), dpi=300,
                                 sharey=True)
        color = COLORS.get(area, '#888888')

        for ax, vspec in zip(axes, VARIABLES):
            vname = vspec['name']
            ldis = np.array([_ldi(c[f'{vname}_rel'], c[f'{vname}_rel_dark'])
                             for c in cells_area])
            ldis = ldis[np.isfinite(ldis)]

            ax.axhspan(0.5, 1.0, color='gold',   alpha=0.08, zorder=0)
            ax.axhspan(0.0, 0.5, color='steelblue', alpha=0.08, zorder=0)

            if len(ldis) > 0:
                ax.hist(ldis, bins=bin_edges, orientation='horizontal',
                        color=color, alpha=0.80, edgecolor='k', linewidth=0.4)
                med = np.median(ldis)
                ax.axhline(med, color=color, lw=1.5, ls='-', alpha=0.95, zorder=4,
                           label=f'median={med:.2f}')
                ax.text(0.97, 0.97, f'n={len(ldis)}\nmed={med:.2f}',
                        ha='right', va='top', transform=ax.transAxes,
                        fontsize=5, color='0.3')

            ax.axhline(0.5, color='k', lw=1.0, ls='--', zorder=5)
            ax.set_ylim(0, 1)
            ax.set_title(vspec['label'], fontsize=8)
            ax.set_xlabel('Count', fontsize=7)

        axes[0].set_ylabel('LDI', fontsize=7)
        axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])

        fig.suptitle(
            f'{area}  —  LDI distributions per variable\n'
            'Gold = light-dominant (LDI > 0.5)  ·  Blue = dark-dominant (LDI < 0.5)\n'
            'Dashed = 0.5 (equal)  ·  Solid = median',
            fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)


def make_ldi_summary_heatmap(pdf, all_cells):
    """Heatmap of median LDI: visual areas (rows) × variables (cols)."""
    areas = [a for a in REGION_ORDER
             if sum(c['area'] == a for c in all_cells) >= MIN_CELLS_AREA]
    n_areas = len(areas)
    nv = len(VARIABLES)

    mat   = np.full((n_areas, nv), np.nan)
    n_mat = np.zeros((n_areas, nv), dtype=int)

    for ai, area in enumerate(areas):
        for vi, vspec in enumerate(VARIABLES):
            vname = vspec['name']
            ldis = [_ldi(c[f'{vname}_rel'], c[f'{vname}_rel_dark'])
                    for c in all_cells if c['area'] == area]
            ldis = [v for v in ldis if np.isfinite(v)]
            if ldis:
                mat[ai, vi]   = np.median(ldis)
                n_mat[ai, vi] = len(ldis)

    fig, ax = plt.subplots(figsize=(nv * 1.4 + 1.0, n_areas * 0.65 + 1.2), dpi=300)
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                   interpolation='nearest')

    ax.set_xticks(range(nv))
    ax.set_xticklabels([v['label'] for v in VARIABLES], fontsize=7)
    ax.set_yticks(range(n_areas))
    ax.set_yticklabels(areas, fontsize=7)

    for ai in range(n_areas):
        for vi in range(nv):
            if np.isfinite(mat[ai, vi]):
                txt_color = 'k' if 0.2 < mat[ai, vi] < 0.8 else 'w'
                ax.text(vi, ai, f'{mat[ai, vi]:.2f}\nn={n_mat[ai, vi]}',
                        ha='center', va='center', fontsize=5, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('Median LDI', fontsize=7)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    fig.suptitle(
        'Median LDI per visual area × variable\n'
        'Green = light-dominant · Red = dark-dominant · 0.5 = equal',
        fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def make_ldi_cdf_page(pdf, all_cells):
    """Cumulative LDI distribution per variable, one line per area."""
    nv = len(VARIABLES)
    fig, axes = plt.subplots(1, nv, figsize=(nv * 2.2 + 0.5, 3.0), dpi=300)

    for ax, vspec in zip(axes, VARIABLES):
        vname = vspec['name']
        for area in REGION_ORDER:
            cells_a = [c for c in all_cells if c['area'] == area]
            if len(cells_a) < MIN_CELLS_AREA:
                continue
            ldis = np.array([_ldi(c[f'{vname}_rel'], c[f'{vname}_rel_dark'])
                             for c in cells_a])
            ldis = np.sort(ldis[np.isfinite(ldis)])
            if len(ldis) == 0:
                continue
            cdf = np.arange(1, len(ldis) + 1) / len(ldis)
            ax.plot(ldis, cdf, color=COLORS.get(area, '#888888'),
                    lw=1.2, label=f'{area} (n={len(ldis)})')

        ax.axvline(0.5, color='k', lw=0.8, ls='--', zorder=5)
        ax.axhline(0.5, color='0.7', lw=0.5, ls=':', zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('LDI', fontsize=7)
        ax.set_ylabel('Cumulative fraction', fontsize=7)
        ax.set_title(vspec['label'], fontsize=8)

    axes[0].legend(fontsize=4, loc='upper left', framealpha=0.6)
    fig.suptitle(
        'LDI cumulative distributions by area\n'
        'Curves shifted right of 0.5 = light-dominant  ·  Dashed = equal (0.5)',
        fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def make_ldi_fraction_page(pdf, all_cells):
    """Stacked bar: % cells light-dominant (LDI > 0.5) vs dark-dominant per area/variable."""
    nv = len(VARIABLES)
    fig, axes = plt.subplots(1, nv, figsize=(nv * 2.2 + 0.5, 3.2), dpi=300)

    for ax, vspec in zip(axes, VARIABLES):
        vname = vspec['name']
        area_n_light = {a: 0 for a in REGION_ORDER}
        area_n_dark  = {a: 0 for a in REGION_ORDER}
        area_n_total = {a: 0 for a in REGION_ORDER}

        for c in all_cells:
            if c['area'] not in area_n_total:
                continue
            ldi = _ldi(c[f'{vname}_rel'], c[f'{vname}_rel_dark'])
            if not np.isfinite(ldi):
                continue
            area_n_total[c['area']] += 1
            if ldi > 0.5:
                area_n_light[c['area']] += 1
            else:
                area_n_dark[c['area']] += 1

        areas_present = [a for a in REGION_ORDER if area_n_total[a] >= MIN_CELLS_AREA]
        if not areas_present:
            ax.set_visible(False)
            continue

        xs = np.arange(len(areas_present))
        for xi, a in enumerate(areas_present):
            n  = area_n_total[a]
            fl = area_n_light[a] / n * 100
            fd = area_n_dark[a]  / n * 100
            ec = COLORS.get(a, '#888888')
            ax.bar(xi, fl, color='gold',      edgecolor=ec, linewidth=1.0)
            ax.bar(xi, fd, bottom=fl, color='steelblue', edgecolor=ec, linewidth=1.0)
            if fl > 8:
                ax.text(xi, fl / 2,     f'{fl:.0f}%', ha='center', va='center',
                        fontsize=4, color='k')
            if fd > 8:
                ax.text(xi, fl + fd / 2, f'{fd:.0f}%', ha='center', va='center',
                        fontsize=4, color='w')
            ax.text(xi, -3, f'n={n}', ha='center', va='top', fontsize=4, color='0.4',
                    transform=ax.get_xaxis_transform())

        ax.axhline(50, color='k', lw=0.8, ls='--')
        ax.set_xticks(xs)
        ax.set_xticklabels(areas_present, fontsize=6)
        ax.set_ylim(0, 100)
        ax.set_ylabel('% cells', fontsize=7)
        ax.set_title(vspec['label'], fontsize=8)

    from matplotlib.patches import Patch as _Patch
    legend_handles = [
        _Patch(facecolor='gold',      edgecolor='k', linewidth=0.5, label='Light-dominant (LDI > 0.5)'),
        _Patch(facecolor='steelblue', edgecolor='k', linewidth=0.5, label='Dark-dominant (LDI ≤ 0.5)'),
    ]
    axes[0].legend(handles=legend_handles, fontsize=5, loc='upper right', framealpha=0.8)
    fig.suptitle(
        '% cells light-dominant vs dark-dominant per area\n'
        'Dashed line = 50%',
        fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def make_ldi_scatter_page(pdf, all_cells):
    """Scatter of lightMI vs darkMI per variable, colored by area."""
    nv = len(VARIABLES)
    fig, axes = plt.subplots(1, nv, figsize=(nv * 2.2 + 0.5, 2.8), dpi=300)

    for ax, vspec in zip(axes, VARIABLES):
        vname = vspec['name']
        all_l, all_d = [], []
        for a in REGION_ORDER:
            cells_a = [c for c in all_cells if c['area'] == a]
            xs = np.array([c[f'{vname}_rel']      for c in cells_a], dtype=float)
            ys = np.array([c[f'{vname}_rel_dark'] for c in cells_a], dtype=float)
            ok = np.isfinite(xs) & np.isfinite(ys)
            if ok.sum() > 0:
                ax.scatter(xs[ok], ys[ok], s=4, alpha=0.5,
                           color=COLORS.get(a, '#888888'), label=a)
                all_l.extend(xs[ok].tolist())
                all_d.extend(ys[ok].tolist())

        if all_l:
            lim_max = max(np.nanmax(all_l), np.nanmax(all_d), 0.1) * 1.05
            ax.plot([0, lim_max], [0, lim_max], 'k--', lw=0.8, alpha=0.5, zorder=0)
            ax.set_xlim(0, lim_max)
            ax.set_ylim(0, lim_max)
        ax.set_xlabel('Light CV MI', fontsize=7)
        ax.set_ylabel('Dark CV MI', fontsize=7)
        ax.set_title(vspec['label'], fontsize=8)

    axes[0].legend(fontsize=4, markerscale=2, loc='upper left', framealpha=0.6)
    fig.suptitle(
        'Light vs Dark CV MI  (points above diagonal = stronger in dark)',
        fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def make_heatmap_pages(pdf, all_cells, top_n=TOP_N_HEATMAP, condition='light'):
    tc_suffix = '' if condition == 'light' else '_dark'
    rk_fn = (lambda vname: f'{vname}_rel') if condition == 'light' \
        else (lambda vname: f'{vname}_rel_dark')

    for vspec in VARIABLES:
        vname  = vspec['name']
        tc_key = f'{vname}_tuning{tc_suffix}'
        rk     = rk_fn(vname)

        cells_v = [c for c in all_cells
                   if c[tc_key] is not None
                   and np.isfinite(c[f'{vname}_rel'])]
        if not cells_v:
            continue

        show   = sorted(cells_v, key=lambda c: c[f'{vname}_rel'], reverse=True)[:top_n]
        n_show = len(show)
        bins   = show[0][f'{vname}_bins']
        n_bins = len(bins)

        mat      = np.array([_norm01(c[tc_key]) for c in show])
        area_rgb = np.array([mpl.colors.to_rgb(COLORS.get(c['area'], '#888888'))
                             for c in show])
        mi_vals  = np.array([c[rk] for c in show])

        fig = plt.figure(figsize=(6.5, max(4, n_show * 0.11 + 1.5)), dpi=300)
        gs  = fig.add_gridspec(1, 3, width_ratios=[0.18, 4.5, 0.8], wspace=0.04,
                               left=0.01, right=0.97, top=0.93, bottom=0.07)
        ax_area = fig.add_subplot(gs[0])
        ax_heat = fig.add_subplot(gs[1])
        ax_mi   = fig.add_subplot(gs[2])

        ax_area.imshow(area_rgb[:, np.newaxis, :], aspect='auto', interpolation='none')
        ax_area.set_xticks([])
        ax_area.set_yticks(range(n_show))
        ax_area.set_yticklabels([c['area'] for c in show], fontsize=4)
        ax_area.tick_params(length=0)

        im = ax_heat.imshow(mat, aspect='auto', cmap='magma',
                            vmin=0, vmax=1, interpolation='nearest')
        ax_heat.set_yticks([])
        ax_heat.set_xlabel(f'{vspec["label"]} bins', fontsize=6)
        if n_bins <= 14:
            ax_heat.set_xticks(range(n_bins))
            ax_heat.set_xticklabels([f'{b:.0f}°' for b in bins],
                                    fontsize=5, rotation=45)
        else:
            ax_heat.set_xticks([])

        cax = ax_heat.inset_axes([1.01, 0.0, 0.03, 1.0], transform=ax_heat.transAxes)
        fig.colorbar(im, cax=cax, label='norm. rate')
        cax.tick_params(labelsize=5)

        mi_display = np.where(np.isfinite(mi_vals), mi_vals, 0.0)
        colors_bar = [COLORS.get(c['area'], '#888888') for c in show]
        ax_mi.barh(range(n_show), mi_display, color=colors_bar, height=0.85)
        ax_mi.set_xlim(0, max(mi_display.max() * 1.1, 0.3))
        ax_mi.set_ylim(-0.5, n_show - 0.5)
        ax_mi.invert_yaxis()
        ax_mi.set_yticks([])
        ax_mi.set_xlabel('CV MI', fontsize=6)
        ax_mi.axvline(0.1, color='0.5', lw=0.7, ls='--')

        fig.suptitle(
            f'Top {n_show} {vspec["label"]}-tuned cells'
            f'  (sorted by light CV MI) — {condition}',
            fontsize=8)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)



def make_per_area_pages(pdf, all_cells, top_n=TOP_N_PER_AREA):
    """Per-area grid of tuning curves with light (solid) and dark (hatched) overlaid."""
    ncols = 4

    for vspec in VARIABLES:
        vname = vspec['name']

        for area in REGION_ORDER:
            cells = [c for c in all_cells
                     if c['area'] == area
                     and c[f'{vname}_tuning'] is not None
                     and np.isfinite(c[f'{vname}_rel'])]
            if len(cells) < MIN_CELLS_AREA:
                continue

            cells.sort(key=lambda c: c[f'{vname}_rel'], reverse=True)
            cells = cells[:top_n]

            nrows = int(np.ceil(len(cells) / ncols))
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(ncols * 1.8, nrows * 1.6),
                                     dpi=200, squeeze=False)
            color = COLORS.get(area, '#888888')

            for i, c in enumerate(cells):
                ax    = axes[i // ncols][i % ncols]
                bins  = c[f'{vname}_bins']
                tc_l  = c[f'{vname}_tuning']
                tc_d  = c[f'{vname}_tuning_dark']
                err_l = c[f'{vname}_err']
                err_d = c[f'{vname}_err_dark']
                mi_l  = c[f'{vname}_rel']
                mi_d  = c[f'{vname}_rel_dark']

                # Light: solid line + semi-transparent fill
                ax.plot(bins, tc_l, color=color, lw=1.2)
                if err_l is not None:
                    ax.fill_between(bins, tc_l - err_l, tc_l + err_l,
                                    alpha=0.25, color=color)

                # Dark: dashed line + hatched polygon fill
                if tc_d is not None:
                    ax.plot(bins, tc_d, color=color, lw=1.0, ls='--')
                    if err_d is not None:
                        _hatch_polygon(ax, bins, tc_d - err_d, tc_d + err_d,
                                       color, alpha=0.20)

                mi_l_str = f'{mi_l:.3f}' if np.isfinite(mi_l) else 'NaN'
                mi_d_str = f'{mi_d:.3f}' if np.isfinite(mi_d) else 'NaN'
                ldi_val  = _ldi(mi_l, mi_d)
                ldi_str  = f'{ldi_val:.2f}' if np.isfinite(ldi_val) else 'NaN'
                ax.set_title(f'L={mi_l_str}  D={mi_d_str}  LDI={ldi_str}',
                             fontsize=5, pad=2)
                mid = len(bins) // 2
                ax.set_xticks([bins[0], bins[mid], bins[-1]])
                ax.set_xticklabels([f'{bins[0]:.0f}°', f'{bins[mid]:.0f}°',
                                    f'{bins[-1]:.0f}°'], fontsize=5)
                ax.tick_params(labelsize=5)
                ax.set_xlabel(f'{vspec["label"]} (°)', fontsize=5)
                ax.text(0.97, 0.95, f'#{i + 1}  {c["animal"]}/{c["pos"]}',
                        ha='right', va='top', transform=ax.transAxes,
                        fontsize=4, color='0.5')

                # Y limit covering both conditions
                top_pieces = [tc_l]
                if err_l is not None:
                    top_pieces.append(tc_l + err_l)
                if tc_d is not None:
                    top_pieces.append(tc_d)
                    if err_d is not None:
                        top_pieces.append(tc_d + err_d)
                top_val = np.nanmax(np.concatenate(top_pieces)) * 1.1
                if np.isfinite(top_val) and top_val > 0:
                    ax.set_ylim(0, top_val)

            for j in range(len(cells), nrows * ncols):
                axes[j // ncols][j % ncols].set_visible(False)

            fig.suptitle(
                f'{area} — {vspec["label"]} — top {len(cells)} cells'
                '  (solid = light · dashed + hatch = dark)',
                fontsize=9)
            fig.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)



def print_tuning_stats(all_cells):
    """Print responsiveness fractions and FOV counts to terminal."""
    from collections import defaultdict

    areas_present = [a for a in REGION_ORDER
                     if sum(c['area'] == a for c in all_cells) >= MIN_CELLS_AREA]

    def _pct(n, total):
        return f'{100.0 * n / total:.1f}%' if total > 0 else 'N/A'

    print('\n' + '=' * 60)
    print('TUNING RESPONSIVENESS SUMMARY')
    print('=' * 60)

    # ---- theta and/or phi ----
    print('\n% cells responsive to theta and/or phi (at least one):')
    print(f'  {"Area":<8} {"n_cells":>8} {"theta|phi":>12}')
    gaze_total = 0
    gaze_resp  = 0
    for area in areas_present:
        cells_a = [c for c in all_cells if c['area'] == area]
        n = len(cells_a)
        resp = sum(c['theta_isrel'] or c['phi_isrel'] for c in cells_a)
        print(f'  {area:<8} {n:>8} {_pct(resp, n):>12}  (n={resp})')
        gaze_total += n
        gaze_resp  += resp
    print(f'  {"ALL":<8} {gaze_total:>8} {_pct(gaze_resp, gaze_total):>12}  (n={gaze_resp})')

    # ---- pitch, roll, or yaw ----
    print('\n% cells responsive to pitch, roll, or yaw (at least one):')
    print(f'  {"Area":<8} {"n_cells":>8} {"p|r|y":>12}')
    imu_total = 0
    imu_resp  = 0
    for area in areas_present:
        cells_a = [c for c in all_cells if c['area'] == area]
        n = len(cells_a)
        resp = sum(c['pitch_isrel'] or c['roll_isrel'] or c['yaw_isrel']
                   for c in cells_a)
        print(f'  {area:<8} {n:>8} {_pct(resp, n):>12}  (n={resp})')
        imu_total += n
        imu_resp  += resp
    print(f'  {"ALL":<8} {imu_total:>8} {_pct(imu_resp, imu_total):>12}  (n={imu_resp})')

    # ---- FOV counts and cell counts per area ----
    print('\nFields of view (unique recordings split by visual area):')
    print(f'  {"Area":<8} {"n_FOVs":>8} {"mean cells/FOV":>16} {"std":>8}')
    fov_map = defaultdict(lambda: defaultdict(list))
    for c in all_cells:
        fov_map[c['area']][(c['animal'], c['pos'])].append(c)

    all_fov_counts = defaultdict(list)
    for c in all_cells:
        all_fov_counts[(c['animal'], c['pos'])].append(c)

    for area in areas_present:
        fovs = fov_map[area]
        counts = [len(v) for v in fovs.values()]
        print(f'  {area:<8} {len(fovs):>8} {np.mean(counts):>16.1f} {np.std(counts):>8.1f}')

    all_counts = [len(v) for v in all_fov_counts.values()]
    print(f'  {"ALL":<8} {len(all_fov_counts):>8} {np.mean(all_counts):>16.1f} {np.std(all_counts):>8.1f}')

    print('=' * 60 + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Summarize 1-D tuning across all variables and visual areas.')
    parser.add_argument('--pooled',    default=DEFAULT_POOLED)
    parser.add_argument('--base_dir',  default=DEFAULT_BASE)
    parser.add_argument('--out_dir',   default=DEFAULT_OUT_DIR)
    parser.add_argument('--threshold', type=float, default=MOD_THRESHOLD,
                        help='CV MI threshold for % modulated page')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'Pooled dataset : {args.pooled}')
    print(f'Search root    : {args.base_dir}')

    all_cells = collect_data(args.pooled, args.base_dir)
    if not all_cells:
        print('No cells collected. Exiting.')
        return

    imu_cells, no_imu_cells = _split_by_imu(all_cells)
    print(f'  IMU recordings    : {len(imu_cells)} cells')
    print(f'  Non-IMU recordings: {len(no_imu_cells)} cells')

    print_tuning_stats(all_cells)

    pdf_path = os.path.join(args.out_dir, 'all_tuning_summary.pdf')
    print(f'\nWriting PDF: {pdf_path}')

    with PdfPages(pdf_path) as pdf:

        make_violin_page(pdf, all_cells)
        make_fraction_page(pdf, all_cells,    threshold=args.threshold)
        make_fraction_page(pdf, imu_cells,    threshold=args.threshold,
                           label='IMU animals')
        make_fraction_page(pdf, no_imu_cells, threshold=args.threshold,
                           label='non-IMU animals')
        make_any_modulated_page(pdf, imu_cells,    threshold=args.threshold,
                                label='IMU animals')
        make_any_modulated_page(pdf, no_imu_cells, threshold=args.threshold,
                                label='non-IMU animals')

        # ---- LDI pages ----
        make_ldi_page(pdf, all_cells)            # violins per area
        make_ldi_histogram_pages(pdf, all_cells) # one page per area, horizontal histograms
        make_ldi_summary_heatmap(pdf, all_cells) # area × variable median heatmap
        make_ldi_cdf_page(pdf, all_cells)        # cumulative distributions
        make_ldi_fraction_page(pdf, all_cells)   # % light vs dark dominant
        make_ldi_scatter_page(pdf, all_cells)    # light vs dark MI scatter

        # ---- per-cell tuning curves (both conditions overlaid) ----
        # make_heatmap_pages(pdf, all_cells)
        make_per_area_pages(pdf, all_cells)

    print(f'Done. PDF: {pdf_path}')


if __name__ == '__main__':
    main()
