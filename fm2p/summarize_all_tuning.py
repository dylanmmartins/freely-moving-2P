

if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

import argparse
import os
import concurrent.futures as cf

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch, Polygon as MPoly
from matplotlib.lines import Line2D

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


DEFAULT_POOLED     = '/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites/pooled_260619a.h5'
DEFAULT_POOLED_GLM = '/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites/pooled_260619a.h5'
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
VAR_NAMES    = [v['name'] for v in VARIABLES]
VARS_NO_YAW  = [v for v in VARIABLES if v['name'] != 'yaw']

_HATCH = '////'   # 45-degree hatch marks for dark condition

# Every figure-generating function calls _save_svg_png(fig, svg_path) instead
# of fig.savefig()+plt.close() directly. Each figure's SVG+PNG save (and the
# matching plt.close) is dispatched as one task to a shared thread pool, so
# different figures' file I/O overlaps; call _finish_pending_saves() once at
# the end of main() to wait for everything and print the 'Saved:' lines.
_SAVE_EXECUTOR = cf.ThreadPoolExecutor(max_workers=4)
_PENDING_SAVES = []


def _save_both_formats(fig, svg_path, dpi=300):
    png_path = os.path.splitext(svg_path)[0] + '.png'
    fig.savefig(svg_path, format='svg', bbox_inches='tight', dpi=dpi)
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    return svg_path, png_path


def _save_svg_png(fig, svg_path, dpi=300):
    _PENDING_SAVES.append(_SAVE_EXECUTOR.submit(_save_both_formats, fig, svg_path, dpi))


def _finish_pending_saves():
    for fut in cf.as_completed(_PENDING_SAVES):
        svg_path, png_path = fut.result()
        print(f'Saved: {svg_path}')
        print(f'Saved: {png_path}')
    _PENDING_SAVES.clear()


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

            cell = dict(animal=animal, pos=pos, ci=ci, area=area, area_id=area_id)

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
    if light_mi <= 0 or dark_mi <= 0:
        return np.nan
    return light_mi / (light_mi + dark_mi)


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


def make_combined_overview_svg(all_cells, out_dir):
    """2x4 SVG: top row = CV MI violins, bottom row = LDI violins (yaw excluded).
    N values are printed to terminal rather than annotated on the figure."""
    n_vars = len(VARS_NO_YAW)
    panel_w, panel_h = 2.0, 2.5
    fig, axes = plt.subplots(2, n_vars,
                              figsize=(n_vars * panel_w, 2 * panel_h),
                              constrained_layout=True)

    print('\n' + '=' * 60)
    print('COMBINED OVERVIEW FIGURE — N values')
    print('=' * 60)

    for vi, vspec in enumerate(VARS_NO_YAW):
        vname = vspec['name']

        # ── top row: CV MI ───────────────────────────────────────────────
        ax_mi = axes[0, vi]
        mi_by_area = {a: [] for a in REGION_ORDER}
        area_n     = {a: 0  for a in REGION_ORDER}

        for c in all_cells:
            if c['area'] not in area_n:
                continue
            area_n[c['area']] += 1
            rl = c[f'{vname}_rel']
            if np.isfinite(rl):
                mi_by_area[c['area']].append(rl)

        areas_mi = [a for a in REGION_ORDER if area_n[a] >= MIN_CELLS_AREA]

        print(f'\nCV MI — {vspec["label"]}:')
        for a in areas_mi:
            print(f'  {a}: n_total={area_n[a]}, n_finite={len(mi_by_area[a])}')

        for xi, a in enumerate(areas_mi):
            color = COLORS.get(a, '#888888')
            vals  = np.array(mi_by_area[a])
            if len(vals) >= 2:
                parts = ax_mi.violinplot([vals], positions=[xi],
                                          widths=0.7, showmedians=False, showextrema=False)
                body = parts['bodies'][0]
                body.set_facecolor(color)
                body.set_edgecolor('k')
                body.set_linewidth(0.5)
                body.set_alpha(0.75)
            if len(vals) >= 1:
                med = np.nanmedian(vals)
                q25, q75 = np.nanpercentile(vals, [25, 75])
                ax_mi.vlines(xi, q25, q75, colors='k', linewidths=2.0, zorder=4)
                ax_mi.scatter([xi], [med], s=14, color='w', edgecolors='k',
                               linewidths=0.7, zorder=5)

        ax_mi.set_xticks(range(len(areas_mi)))
        ax_mi.set_xticklabels(areas_mi, fontsize=6)
        ax_mi.set_title(vspec['label'], fontsize=8)
        ax_mi.set_xlim(-0.6, len(areas_mi) - 0.4)
        ax_mi.axhline(0, color='0.7', lw=0.8, ls='--')
        if vi == 0:
            ax_mi.set_ylabel('CV MI', fontsize=7)

        # ── bottom row: LDI ──────────────────────────────────────────────
        ax_ldi = axes[1, vi]
        ldi_by_area = {a: [] for a in REGION_ORDER}
        ldi_area_n  = {a: 0  for a in REGION_ORDER}

        for c in all_cells:
            if c['area'] not in ldi_area_n:
                continue
            ldi_area_n[c['area']] += 1
            ldi_val = _ldi(c[f'{vname}_rel'], c[f'{vname}_rel_dark'])
            if np.isfinite(ldi_val):
                ldi_by_area[c['area']].append(ldi_val)

        areas_ldi = [a for a in REGION_ORDER if ldi_area_n[a] >= MIN_CELLS_AREA]

        print(f'\nLDI — {vspec["label"]}:')
        for a in areas_ldi:
            print(f'  {a}: n_total={ldi_area_n[a]}, n_finite_ldi={len(ldi_by_area[a])}')

        for xi, a in enumerate(areas_ldi):
            color = COLORS.get(a, '#888888')
            vals  = np.array(ldi_by_area[a])
            if len(vals) >= 2:
                parts = ax_ldi.violinplot([vals], positions=[xi],
                                           widths=0.7, showmedians=False, showextrema=False)
                body = parts['bodies'][0]
                body.set_facecolor(color)
                body.set_edgecolor('k')
                body.set_linewidth(0.5)
                body.set_alpha(0.75)
            if len(vals) >= 1:
                med = np.nanmedian(vals)
                q25, q75 = np.nanpercentile(vals, [25, 75])
                ax_ldi.vlines(xi, q25, q75, colors='k', linewidths=2.0, zorder=4)
                ax_ldi.scatter([xi], [med], s=14, color='w', edgecolors='k',
                                linewidths=0.7, zorder=5)

        ax_ldi.set_xticks(range(len(areas_ldi)))
        ax_ldi.set_xticklabels(areas_ldi, fontsize=6)
        ax_ldi.set_xlim(-0.6, len(areas_ldi) - 0.4)
        ax_ldi.set_ylim(-0.05, 1.05)
        ax_ldi.axhline(0.5, color='0.7', lw=0.8, ls='--')
        ax_ldi.axhline(0,   color='0.5', lw=0.5)
        if vi == 0:
            ax_ldi.set_ylabel('LDI', fontsize=7)

    print('=' * 60)

    svg_path = os.path.join(out_dir, 'overview_mi_ldi.svg')
    _save_svg_png(fig, svg_path)


def _box_strip_panel(ax, by_area, areas, rng, cap=250):
    """Box plot (median/IQR/5-95th pctile whiskers, no fill) drawn on top of
    jittered raw per-cell points (subsampled to `cap` per area). Used as a
    replacement for violins on heavily skewed, bounded data, where the
    violin's kernel-smoothed shape compresses into an uninformative thin
    neck near the boundary."""
    for xi, a in enumerate(areas):
        vals = np.asarray(by_area[a])
        if vals.size == 0:
            continue
        color = COLORS.get(a, '#888888')
        show_vals = vals if vals.size <= cap else rng.choice(vals, size=cap, replace=False)
        jitter = rng.uniform(-0.18, 0.18, size=show_vals.size)
        ax.scatter(xi + jitter, show_vals, color=color, s=4, alpha=0.25,
                  linewidths=0, zorder=1)
        bp = ax.boxplot([vals], positions=[xi], widths=0.5, whis=[5, 95],
                        showfliers=False, patch_artist=True, zorder=3)
        for el in ('boxes', 'whiskers', 'caps', 'medians'):
            for artist in bp[el]:
                artist.set_color(color)
                artist.set_linewidth(1.2)
        bp['boxes'][0].set_facecolor('none')


def make_overview_mi_ldi_boxstrip_svg(all_cells, out_dir):
    """Improved companion to make_combined_overview_svg() -- saved as a
    separate file, the original is left unchanged. Violins on CV MI (heavily
    right-skewed, most mass near 0) compress almost all shape information
    into a thin neck near the bottom, and LDI's narrow real-world range made
    its violins look like nearly uniform-width bars top to bottom. This
    version uses a box (median/IQR/5-95th pctile whiskers) with jittered raw
    points underneath: the point cloud shows density directly (no kernel
    bandwidth artifact) and the whisker length communicates skew."""
    rng = np.random.default_rng(0)
    n_vars = len(VARS_NO_YAW)
    panel_w, panel_h = 2.0, 2.5
    fig, axes = plt.subplots(2, n_vars, figsize=_scaled(n_vars * panel_w, 2 * panel_h),
                              constrained_layout=True)

    for vi, vspec in enumerate(VARS_NO_YAW):
        vname = vspec['name']

        ax_mi = axes[0, vi]
        mi_by_area = {a: [] for a in REGION_ORDER}
        area_n = {a: 0 for a in REGION_ORDER}
        for c in all_cells:
            if c['area'] not in area_n:
                continue
            area_n[c['area']] += 1
            rl = c[f'{vname}_rel']
            if np.isfinite(rl):
                mi_by_area[c['area']].append(rl)
        areas_mi = [a for a in REGION_ORDER if area_n[a] >= MIN_CELLS_AREA]
        _box_strip_panel(ax_mi, mi_by_area, areas_mi, rng)
        ax_mi.set_xticks(range(len(areas_mi)))
        ax_mi.set_xticklabels(areas_mi, fontsize=6)
        ax_mi.set_title(vspec['label'], fontsize=8)
        ax_mi.set_xlim(-0.6, len(areas_mi) - 0.4)
        ax_mi.axhline(0, color='0.7', lw=0.8, ls='--')
        if vi == 0:
            ax_mi.set_ylabel('CV MI', fontsize=7)

        ax_ldi = axes[1, vi]
        ldi_by_area = {a: [] for a in REGION_ORDER}
        ldi_area_n = {a: 0 for a in REGION_ORDER}
        for c in all_cells:
            if c['area'] not in ldi_area_n:
                continue
            ldi_area_n[c['area']] += 1
            ldi_val = _ldi(c[f'{vname}_rel'], c[f'{vname}_rel_dark'])
            if np.isfinite(ldi_val):
                ldi_by_area[c['area']].append(ldi_val)
        areas_ldi = [a for a in REGION_ORDER if ldi_area_n[a] >= MIN_CELLS_AREA]
        _box_strip_panel(ax_ldi, ldi_by_area, areas_ldi, rng)
        ax_ldi.set_xticks(range(len(areas_ldi)))
        ax_ldi.set_xticklabels(areas_ldi, fontsize=6)
        ax_ldi.set_xlim(-0.6, len(areas_ldi) - 0.4)
        ax_ldi.set_ylim(-0.05, 1.05)
        ax_ldi.axhline(0.5, color='0.7', lw=0.8, ls='--')
        ax_ldi.axhline(0, color='0.5', lw=0.5)
        if vi == 0:
            ax_ldi.set_ylabel('LDI', fontsize=7)

    fig.suptitle('CV MI and LDI by area\n(box = median/IQR/5-95th pctile; points = individual cells, jittered)',
                 fontsize=7)

    path = os.path.join(out_dir, 'overview_mi_ldi_boxstrip.svg')
    _save_svg_png(fig, path)


def make_example_tuning_svgs(all_cells, out_dir):
    """Per-area SVG: most-modulated cell (with non-NaN LDI) per variable (yaw excluded).
    Light = solid line, dark = dashed."""
    for area in REGION_ORDER:
        cells_area = [c for c in all_cells if c['area'] == area]
        if len(cells_area) < MIN_CELLS_AREA:
            continue

        n_vars = len(VARS_NO_YAW)
        fig, axes = plt.subplots(1, n_vars,
                                  figsize=(n_vars * 2.0, 2.5),
                                  constrained_layout=True)
        color = COLORS.get(area, '#888888')
        any_plotted = False

        for vi, vspec in enumerate(VARS_NO_YAW):
            ax    = axes[vi]
            vname = vspec['name']

            candidates = [
                c for c in cells_area
                if c[f'{vname}_tuning'] is not None
                and np.isfinite(c[f'{vname}_rel'])
                and np.isfinite(_ldi(c[f'{vname}_rel'], c[f'{vname}_rel_dark']))
            ]

            if not candidates:
                ax.set_visible(False)
                continue

            best  = max(candidates, key=lambda c: c[f'{vname}_rel'])
            bins  = best[f'{vname}_bins']
            tc_l  = best[f'{vname}_tuning']
            tc_d  = best[f'{vname}_tuning_dark']
            err_l = best[f'{vname}_err']
            err_d = best[f'{vname}_err_dark']
            mi_l  = best[f'{vname}_rel']
            mi_d  = best[f'{vname}_rel_dark']
            ldi   = _ldi(mi_l, mi_d)

            ax.plot(bins, tc_l, color=color, lw=1.5)
            if err_l is not None:
                ax.fill_between(bins, tc_l - err_l, tc_l + err_l,
                                alpha=0.25, color=color)
            if tc_d is not None:
                ax.plot(bins, tc_d, color=color, lw=1.2, ls='--')
                if err_d is not None:
                    _hatch_polygon(ax, bins, tc_d - err_d, tc_d + err_d, color, alpha=0.20)

            ax.set_title(f'{vspec["label"]}\nMI={mi_l:.2f}  LDI={ldi:.2f}', fontsize=7)
            ax.set_xlabel(f'{vspec["label"]} (°)', fontsize=6)
            if vi == 0:
                ax.set_ylabel('Firing rate', fontsize=6)
            ax.tick_params(labelsize=5)

            mid = len(bins) // 2
            ax.set_xticks([bins[0], bins[mid], bins[-1]])
            ax.set_xticklabels([f'{bins[0]:.0f}°', f'{bins[mid]:.0f}°',
                                 f'{bins[-1]:.0f}°'], fontsize=5)

            pieces = [tc_l]
            if err_l is not None:
                pieces.append(tc_l + err_l)
            if tc_d is not None:
                pieces.append(tc_d)
                if err_d is not None:
                    pieces.append(tc_d + err_d)
            top_val = np.nanmax(np.concatenate(pieces)) * 1.1
            if np.isfinite(top_val) and top_val > 0:
                ax.set_ylim(0, top_val)

            any_plotted = True

        if not any_plotted:
            plt.close(fig)
            continue

        fig.suptitle(
            f'{area} — most modulated cell per variable  (solid = light · dashed = dark)',
            fontsize=8)
        svg_path = os.path.join(out_dir, f'example_tuning_{area}.svg')
        _save_svg_png(fig, svg_path)


def print_mi_ldi_stats(all_cells):
    """Print mean ± std of CV MI and LDI per area per variable (yaw excluded)."""
    print('\n' + '=' * 82)
    print('CV MI AND LDI STATISTICS (mean ± std, yaw excluded)')
    print('=' * 82)

    for vspec in VARS_NO_YAW:
        vname = vspec['name']
        print(f'\n  {vspec["label"]} ({vname}):')
        print(f'    {"Area":<6}  {"n_MI":>5}  {"mean MI":>8}  {"std MI":>8}'
              f'  {"n_LDI":>6}  {"mean LDI":>9}  {"std LDI":>9}')

        for area in REGION_ORDER:
            cells_a = [c for c in all_cells if c['area'] == area]
            if len(cells_a) < MIN_CELLS_AREA:
                continue

            mi_vals  = [c[f'{vname}_rel'] for c in cells_a
                        if np.isfinite(c[f'{vname}_rel'])]
            ldi_vals = [_ldi(c[f'{vname}_rel'], c[f'{vname}_rel_dark'])
                        for c in cells_a]
            ldi_vals = [v for v in ldi_vals if np.isfinite(v)]

            mi_mean  = np.mean(mi_vals)  if mi_vals  else float('nan')
            mi_std   = np.std(mi_vals)   if mi_vals  else float('nan')
            ldi_mean = np.mean(ldi_vals) if ldi_vals else float('nan')
            ldi_std  = np.std(ldi_vals)  if ldi_vals else float('nan')

            print(f'    {area:<6}  {len(mi_vals):>5}  {mi_mean:>8.3f}  {mi_std:>8.3f}'
                  f'  {len(ldi_vals):>6}  {ldi_mean:>9.3f}  {ldi_std:>9.3f}')

    print('=' * 82 + '\n')



# Variable pairs shown in the importance figure (position row, velocity row).
# Each tuple: (pos_key, vel_key, pos_label, vel_label)
_IMP_PAIRS = [
    ('theta',  'dTheta', r'θ',      r'dθ'),
    ('phi',    'dPhi',   r'φ',      r'dφ'),
    ('pitch',  'gyro_y', 'Pitch',   'dPitch'),
    ('roll',   'gyro_x', 'Roll',    'dRoll'),
]

_IMP_VAR_ORDER  = ['theta', 'dTheta', 'phi', 'dPhi',
                   'pitch', 'gyro_y', 'roll', 'gyro_x', 'yaw', 'gyro_z']
_IMP_ID_TO_NAME = {5: 'V1', 2: 'RL', 3: 'AM', 4: 'PM', 10: 'A'}
_IMP_REGION_ORDER = ['V1', 'RL', 'AM', 'PM', 'A']


def _load_importance_cells(pooled_glm_path):
    """Load per-cell permutation importance from the GLM pooled h5."""
    from .utils.files import read_h5
    pooled = read_h5(pooled_glm_path)

    records = []
    for animal in sorted(pooled.keys()):
        adat = pooled[animal]
        if 'messentials' not in adat:
            continue
        me = adat['messentials']
        for pos in sorted(me.keys()):
            pdat = me[pos]
            if not isinstance(pdat, dict):
                continue
            model = pdat.get('model', {})
            if 'full_r2' not in model:
                continue

            n = len(np.atleast_1d(np.asarray(model['full_r2'], dtype=float)))

            def _imp(prefix):
                mat = np.full((n, len(_IMP_VAR_ORDER)), np.nan)
                for vi, var in enumerate(_IMP_VAR_ORDER):
                    k = f'{prefix}ablation_index_{var}'
                    if k in model:
                        v = np.atleast_1d(np.asarray(model[k], dtype=float))
                        v = np.clip(v, 0.0, 1.0)
                        m = min(len(v), n)
                        mat[:m, vi] = v[:m]
                return mat

            light_imp = _imp('full_trainLight_testLight_')
            dark_imp  = _imp('full_trainDark_testDark_')

            area_id = np.zeros(n, dtype=int)
            raw_aid = pdat.get('visual_area_id', None)
            if raw_aid is not None:
                raw_aid = np.atleast_1d(np.asarray(raw_aid, dtype=int))
                m = min(len(raw_aid), n)
                area_id[:m] = raw_aid[:m]

            for ci in range(n):
                name = _IMP_ID_TO_NAME.get(int(area_id[ci]))
                if name is None:
                    continue
                records.append(dict(
                    area=name,
                    animal=animal, pos=pos, ci=ci,
                    light_imp=light_imp[ci].copy(),
                    dark_imp=dark_imp[ci].copy(),
                ))

    print(f'Importance cells loaded: {len(records)}')
    return records


def make_importance_svg(out_dir, pooled_glm_path=DEFAULT_POOLED_GLM, records=None):
    """2×4 SVG: permutation importance by visual area.
    Row 0 = position variables, row 1 = velocity variables (yaw excluded).
    Each panel: light (solid) and dark (hatched) violins side-by-side per area.
    N values printed to terminal."""
    if records is None:
        if not os.path.exists(pooled_glm_path):
            print(f'GLM pooled file not found: {pooled_glm_path} — skipping importance SVG.')
            return
        records = _load_importance_cells(pooled_glm_path)

    if not records:
        print('No importance data — skipping importance SVG.')
        return

    n_cols   = len(_IMP_PAIRS)
    panel_w, panel_h = 2.0, 2.5
    fig, axes = plt.subplots(2, n_cols,
                              figsize=(n_cols * panel_w, 2 * panel_h),
                              constrained_layout=True)

    print('\n' + '=' * 60)
    print('IMPORTANCE FIGURE — N values')
    print('=' * 60)

    off = 0.22
    vw  = 0.38

    for ci_col, (pos_key, vel_key, pos_lbl, vel_lbl) in enumerate(_IMP_PAIRS):
        pos_vi = _IMP_VAR_ORDER.index(pos_key)
        vel_vi = _IMP_VAR_ORDER.index(vel_key)

        for row, (vi, label) in enumerate([(pos_vi, pos_lbl), (vel_vi, vel_lbl)]):
            ax = axes[row, ci_col]

            area_vals_l = {a: [] for a in _IMP_REGION_ORDER}
            area_vals_d = {a: [] for a in _IMP_REGION_ORDER}
            area_n      = {a: 0  for a in _IMP_REGION_ORDER}

            for r in records:
                a = r['area']
                if a not in area_n:
                    continue
                area_n[a] += 1
                vl = float(r['light_imp'][vi])
                vd = float(r['dark_imp'][vi])
                if np.isfinite(vl):
                    area_vals_l[a].append(vl)
                if np.isfinite(vd):
                    area_vals_d[a].append(vd)

            areas_present = [a for a in _IMP_REGION_ORDER if area_n[a] >= MIN_CELLS_AREA]

            print(f'\n{label} ({"pos" if row == 0 else "vel"}):')
            for a in areas_present:
                vl_arr = np.array(area_vals_l[a])
                vd_arr = np.array(area_vals_d[a])
                l_mean = np.mean(vl_arr) if vl_arr.size else float('nan')
                l_sem  = np.std(vl_arr, ddof=1) / np.sqrt(vl_arr.size) if vl_arr.size > 1 else float('nan')
                d_mean = np.mean(vd_arr) if vd_arr.size else float('nan')
                d_sem  = np.std(vd_arr, ddof=1) / np.sqrt(vd_arr.size) if vd_arr.size > 1 else float('nan')
                print(f'  {a}: n_total={area_n[a]}, '
                      f'n_light={len(area_vals_l[a])}, n_dark={len(area_vals_d[a])}, '
                      f'light={l_mean:.3f}±{l_sem:.3f} (SEM), dark={d_mean:.3f}±{d_sem:.3f} (SEM)')

            for xi, a in enumerate(areas_present):
                color = COLORS.get(a, '#888888')
                for vals, xpos, do_hatch in [
                    (area_vals_l[a], xi - off, False),
                    (area_vals_d[a], xi + off, True),
                ]:
                    vals_arr = np.array(vals)
                    if len(vals_arr) >= 2:
                        parts = ax.violinplot([vals_arr], positions=[xpos],
                                              widths=vw, showmedians=False, showextrema=False)
                        body = parts['bodies'][0]
                        body.set_facecolor(color)
                        body.set_edgecolor('k')
                        body.set_linewidth(0.5)
                        body.set_alpha(0.75 if not do_hatch else 0.50)
                        if do_hatch:
                            body.set_hatch(_HATCH)
                    if len(vals_arr) >= 1:
                        med = np.nanmedian(vals_arr)
                        q25, q75 = np.nanpercentile(vals_arr, [25, 75])
                        ax.vlines(xpos, q25, q75, colors='k', linewidths=2.0, zorder=4)
                        ax.scatter([xpos], [med], s=14, color='w', edgecolors='k',
                                   linewidths=0.7, zorder=5)

            ax.set_xticks(range(len(areas_present)))
            ax.set_xticklabels(areas_present, fontsize=6)
            ax.set_title(label, fontsize=8)
            ax.set_xlim(-0.6, len(areas_present) - 0.4)
            ax.set_ylim(0, 1)
            ax.axhline(0, color='0.7', lw=0.8, ls='--')
            if ci_col == 0:
                ax.set_ylabel('Permutation importance', fontsize=7)

    print('=' * 60)

    _add_legend(fig)
    fig.suptitle(
        'Permutation importance by visual area\n'
        'Row 1: position variables  ·  Row 2: velocity variables\n'
        '(solid = light · hatched = dark)',
        fontsize=8)

    svg_path = os.path.join(out_dir, 'importance_by_area.svg')
    _save_svg_png(fig, svg_path)



_ALL_VAR_KEYS   = ['theta', 'dTheta', 'phi', 'dPhi', 'pitch', 'gyro_y', 'roll', 'gyro_x']
_ALL_VAR_LABELS = {'theta': 'theta', 'dTheta': 'dTheta', 'phi': 'phi', 'dPhi': 'dPhi',
                    'pitch': 'pitch', 'gyro_y': 'dPitch', 'roll': 'roll', 'gyro_x': 'dRoll'}
_POS_VAR_KEYS = ['theta', 'phi', 'pitch', 'roll']
_VEL_VAR_KEYS = ['dTheta', 'dPhi', 'gyro_y', 'gyro_x']
_HEATMAP_VAR_ORDER = _POS_VAR_KEYS + _VEL_VAR_KEYS  # position block, then velocity block
# 'D' (diamond) reads too much like 's' (square) at a glance -> use '*' for roll.
_POS_MARKERS  = {'theta': 'o', 'phi': 's', 'pitch': '^', 'roll': '*'}

_var_cmap   = plt.cm.get_cmap('tab10')
_VAR_COLORS = {k: _var_cmap(i % 10) for i, k in enumerate(_ALL_VAR_KEYS)}
# Velocity variables get distinct, saturated colors -- not grey/neutral (that's
# reserved for "position" as a null/pooled group elsewhere) and not close to
# any anatomical-area color in COLORS.
_VAR_COLORS.update({
    'dTheta': '#E6194B',  # crimson
    'dPhi':   '#3CB44B',  # vivid green
    'gyro_y': '#4363D8',  # royal blue    (dPitch)
    'gyro_x': '#F032E6',  # vivid magenta (dRoll)
})

_FIG_SCALE = 0.75   # shrink factor applied to figsize across these figures
_FIG_DPI   = 300


def _scaled(w, h):
    return (w * _FIG_SCALE, h * _FIG_SCALE)


def _collect_imp_stats(records):
    """area -> var_key -> dict(light_vals, dark_vals, n_light, n_dark,
    light_mean, light_sem, dark_mean, dark_sem)."""
    raw = {a: {v: {'light': [], 'dark': []} for v in _ALL_VAR_KEYS} for a in _IMP_REGION_ORDER}
    for r in records:
        a = r['area']
        if a not in raw:
            continue
        for vi, var in enumerate(_IMP_VAR_ORDER):
            if var not in _ALL_VAR_KEYS:
                continue
            vl = float(r['light_imp'][vi])
            vd = float(r['dark_imp'][vi])
            if np.isfinite(vl):
                raw[a][var]['light'].append(vl)
            if np.isfinite(vd):
                raw[a][var]['dark'].append(vd)

    stats = {}
    for a in _IMP_REGION_ORDER:
        stats[a] = {}
        for v in _ALL_VAR_KEYS:
            lv = np.array(raw[a][v]['light'])
            dv = np.array(raw[a][v]['dark'])
            stats[a][v] = dict(
                light_vals=lv, dark_vals=dv,
                n_light=lv.size, n_dark=dv.size,
                light_mean=float(np.mean(lv)) if lv.size else np.nan,
                light_sem=float(np.std(lv, ddof=1) / np.sqrt(lv.size)) if lv.size > 1 else np.nan,
                dark_mean=float(np.mean(dv)) if dv.size else np.nan,
                dark_sem=float(np.std(dv, ddof=1) / np.sqrt(dv.size)) if dv.size > 1 else np.nan,
            )
    return stats


def _collect_ldi_mi_stats(all_cells):
    """area -> var_name (theta/phi/pitch/roll) -> dict(mi_vals, ldi_vals,
    n_mi, n_ldi, mi_mean, mi_std, ldi_mean, ldi_std)."""
    stats = {a: {} for a in REGION_ORDER}
    for v in _POS_VAR_KEYS:
        for area in REGION_ORDER:
            cells_a = [c for c in all_cells if c['area'] == area]
            mi_vals = np.array([c[f'{v}_rel'] for c in cells_a
                                 if np.isfinite(c[f'{v}_rel'])])
            ldi_raw = [_ldi(c[f'{v}_rel'], c[f'{v}_rel_dark']) for c in cells_a]
            ldi_vals = np.array([x for x in ldi_raw if np.isfinite(x)])
            stats[area][v] = dict(
                mi_vals=mi_vals, ldi_vals=ldi_vals,
                n_mi=mi_vals.size, n_ldi=ldi_vals.size,
                mi_mean=float(np.mean(mi_vals)) if mi_vals.size else np.nan,
                mi_std=float(np.std(mi_vals)) if mi_vals.size else np.nan,
                ldi_mean=float(np.mean(ldi_vals)) if ldi_vals.size else np.nan,
                ldi_std=float(np.std(ldi_vals)) if ldi_vals.size else np.nan,
            )
    return stats


def make_ablation_heatmap_svg(records, out_dir):
    """(1) 2-panel (light/dark) heatmap: rows=area, cols=variable, color=mean
    ablation index (shared 0-1 scale), cell text = mean +/- SEM."""
    stats = _collect_imp_stats(records)
    areas = _IMP_REGION_ORDER
    var_keys   = _HEATMAP_VAR_ORDER
    var_labels = [_ALL_VAR_LABELS[v] for v in var_keys]

    fig, axes = plt.subplots(1, 2, figsize=_scaled(10, 4.2), constrained_layout=True)
    im = None
    for cond, ax in zip(['light', 'dark'], axes):
        mat     = np.full((len(areas), len(var_keys)), np.nan)
        sem_mat = np.full((len(areas), len(var_keys)), np.nan)
        for ai, a in enumerate(areas):
            for vi, v in enumerate(var_keys):
                d = stats[a][v]
                mat[ai, vi]     = d[f'{cond}_mean']
                sem_mat[ai, vi] = d[f'{cond}_sem']
        im = ax.imshow(mat, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(var_keys)))
        ax.set_xticklabels(var_labels, fontsize=7, rotation=45, ha='right')
        ax.set_yticks(range(len(areas)))
        ax.set_yticklabels(areas, fontsize=8)
        ax.set_title(cond.capitalize(), fontsize=9)
        ax.axvline(len(_POS_VAR_KEYS) - 0.5, color='white', lw=1.2)
        for ai in range(len(areas)):
            for vi in range(len(var_keys)):
                val, sem = mat[ai, vi], sem_mat[ai, vi]
                if np.isfinite(val):
                    color = 'white' if val < 0.5 else 'black'
                    sem_str = f'{sem:.2f}' if np.isfinite(sem) else 'n/a'
                    ax.text(vi, ai, f'{val:.2f}\n±{sem_str}', ha='center', va='center',
                            fontsize=5, color=color)
    fig.colorbar(im, ax=axes, shrink=0.8, label='Mean ablation index')
    fig.suptitle('Ablation index by area and variable (light vs. dark)', fontsize=9)

    path = os.path.join(out_dir, 'ablation_heatmap.svg')
    _save_svg_png(fig, path)


def make_slope_graph_svg(records, out_dir):
    """(2) Slope graph (paired line plot): x=light/dark, y=mean ablation
    index, one line per area, faceted by velocity variable."""
    stats = _collect_imp_stats(records)
    fig, axes = plt.subplots(1, 4, figsize=_scaled(11, 3.2), constrained_layout=True, sharey=True)

    for vi, v in enumerate(_VEL_VAR_KEYS):
        ax = axes[vi]
        for area in _IMP_REGION_ORDER:
            d = stats[area][v]
            if d['n_light'] < MIN_CELLS_AREA or d['n_dark'] < MIN_CELLS_AREA:
                continue
            color = COLORS.get(area, '#888888')
            ys = [d['light_mean'], d['dark_mean']]
            es = [d['light_sem'], d['dark_sem']]
            ax.errorbar([0, 1], ys, yerr=es, color=color, marker='o', ms=4,
                        lw=1.5, capsize=2, label=area)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Light', 'Dark'])
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(_ALL_VAR_LABELS[v], fontsize=9)
        if vi == 0:
            ax.set_ylabel('Mean ablation index')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=6, ncol=1)
    fig.suptitle('Light -> dark ablation index by area (velocity variables)', fontsize=9)

    path = os.path.join(out_dir, 'slope_graph_velocity.svg')
    _save_svg_png(fig, path)


def make_fold_change_svg(records, out_dir, layout='grouped', scope='velocity'):
    """(3) Fold change (dark AI / light AI) bar chart, with both a
    grouped-by-area and faceted-by-variable layout, and both a velocity-only
    and all-8-variable scope (called 4x from main)."""
    stats = _collect_imp_stats(records)
    var_keys = _VEL_VAR_KEYS if scope == 'velocity' else _ALL_VAR_KEYS
    areas = _IMP_REGION_ORDER

    fold = {a: {} for a in areas}
    for a in areas:
        for v in var_keys:
            d = stats[a][v]
            ok = (d['n_light'] >= MIN_CELLS_AREA and d['n_dark'] >= MIN_CELLS_AREA
                  and np.isfinite(d['light_mean']) and d['light_mean'] > 0)
            fold[a][v] = (d['dark_mean'] / d['light_mean']) if ok else np.nan

    print(f'\nFold change (dark AI / light AI), scope={scope}:')
    for a in areas:
        for v in var_keys:
            val = fold[a][v]
            print(f'  {a} {_ALL_VAR_LABELS[v]}: '
                  f'{val:.3f}' if np.isfinite(val) else f'  {a} {_ALL_VAR_LABELS[v]}: nan')

    if layout == 'grouped':
        fig, ax = plt.subplots(figsize=_scaled(max(6, len(areas) * 1.8), 4), constrained_layout=True)
        n_v = len(var_keys)
        width = 0.8 / n_v
        for vi, v in enumerate(var_keys):
            xs = np.arange(len(areas)) + (vi - (n_v - 1) / 2) * width
            ys = [fold[a][v] for a in areas]
            ax.bar(xs, ys, width=width, color=_VAR_COLORS[v], edgecolor='k',
                   linewidth=0.4, label=_ALL_VAR_LABELS[v])
        ax.set_xticks(range(len(areas)))
        ax.set_xticklabels(areas)
        ax.legend(fontsize=6, ncol=min(4, n_v))
        ax.set_title(f'Fold change by area (grouped, {scope})', fontsize=9)
        if scope == 'velocity':
            ax.axhline(2.0, color='0.4', lw=1, ls=':')
        fname = f'fold_change_grouped_{scope}.svg'
    else:
        fig, axes = plt.subplots(1, len(var_keys), figsize=_scaled(2.3 * len(var_keys), 3.4),
                                  constrained_layout=True, sharey=True)
        axes = np.atleast_1d(axes)
        for vi, v in enumerate(var_keys):
            ax = axes[vi]
            ys = [fold[a][v] for a in areas]
            colors = [COLORS.get(a, '#888888') for a in areas]
            ax.bar(range(len(areas)), ys, color=colors, edgecolor='k', linewidth=0.4)
            ax.set_xticks(range(len(areas)))
            ax.set_xticklabels(areas, fontsize=7)
            ax.set_title(_ALL_VAR_LABELS[v], fontsize=8)
            if vi == 0:
                ax.set_ylabel('Fold change (dark AI / light AI)')
        fig.suptitle(f'Fold change by variable (faceted, {scope})', fontsize=9)
        fname = f'fold_change_faceted_{scope}.svg'

    for ax in (np.atleast_1d(fig.axes)):
        ax.axhline(1.0, color='k', lw=1, ls='--')

    path = os.path.join(out_dir, fname)
    _save_svg_png(fig, path)


def make_ldi_vs_ai_scatter_svg(records, all_cells, out_dir):
    """(4) Scatter: x = LDI-0.5 (marginal tuning bias), y = AI_light-AI_dark
    (unique multivariate contribution bias). 20 points (5 areas x 4 position
    variables). Color=area, marker=variable. RL-pitch and AM-phi annotated."""
    imp_stats = _collect_imp_stats(records)
    ldi_stats = _collect_ldi_mi_stats(all_cells)

    fig, ax = plt.subplots(figsize=_scaled(5.5, 5.5), constrained_layout=True)
    ax.set_box_aspect(1)  # square panel; data units need not be 1:1 (ranges differ a lot)
    pts = []
    for area in _IMP_REGION_ORDER:
        for v in _POS_VAR_KEYS:
            ld = ldi_stats[area][v]
            ip = imp_stats[area][v]
            if ld['n_ldi'] < MIN_CELLS_AREA or ip['n_light'] < MIN_CELLS_AREA or ip['n_dark'] < MIN_CELLS_AREA:
                continue
            x = ld['ldi_mean'] - 0.5
            y = ip['light_mean'] - ip['dark_mean']
            pts.append((area, v, x, y))
            ax.scatter(x, y, color=COLORS.get(area, '#888888'), marker=_POS_MARKERS[v],
                       s=70, edgecolors='k', linewidths=0.6, zorder=3)

    ax.axhline(0, color='0.5', lw=0.8)
    ax.axvline(0, color='0.5', lw=0.8)

    for area, v, x, y in pts:
        if (area, v) in [('RL', 'pitch'), ('AM', 'phi')]:
            ax.annotate(f'{area}-{v}', (x, y), textcoords='offset points',
                        xytext=(7, 7), fontsize=7)

    ax.set_xlabel('LDI - 0.5  (marginal tuning bias; + = light-leaning)')
    ax.set_ylabel('AI$_{light}$ - AI$_{dark}$  (+ = unique contribution greater in light)')
    ax.set_title('Marginal tuning bias vs. unique multivariate contribution', fontsize=8)

    area_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS.get(a, '#888888'),
                            markeredgecolor='k', markersize=7, label=a) for a in _IMP_REGION_ORDER]
    var_handles = [Line2D([0], [0], marker=_POS_MARKERS[v], color='w', markerfacecolor='0.6',
                           markeredgecolor='k', markersize=7, label=v) for v in _POS_VAR_KEYS]
    leg1 = ax.legend(handles=area_handles, title='Area', loc='upper left',
                     fontsize=6, title_fontsize=6)
    ax.add_artist(leg1)
    ax.legend(handles=var_handles, title='Variable', loc='center left',
              bbox_to_anchor=(1.02, 0.5), fontsize=6, title_fontsize=6)

    path = os.path.join(out_dir, 'ldi_vs_ai_scatter.svg')
    _save_svg_png(fig, path)

    n_concordant = sum(1 for _, _, x, y in pts if np.sign(x) == np.sign(y))
    print(f'LDI-vs-AI sign concordance: {n_concordant}/{len(pts)} = {n_concordant / len(pts):.1%}')
    return pts


def make_ldi_beeswarm_svg(all_cells, out_dir):
    """(5) Strip plot: x=variable, y=LDI, one point per area (mean), colored
    by area. Vertical band behind each point = within-area SEM of per-cell
    LDI. Horizontal reference line at 0.5 (no light/dark bias)."""
    ldi_stats = _collect_ldi_mi_stats(all_cells)
    n_areas = len(_IMP_REGION_ORDER)

    fig, ax = plt.subplots(figsize=_scaled(6, 4.2), constrained_layout=True)
    for vi, v in enumerate(_POS_VAR_KEYS):
        for ai, area in enumerate(_IMP_REGION_ORDER):
            d = ldi_stats[area][v]
            if d['n_ldi'] < MIN_CELLS_AREA:
                continue
            x = vi + (ai - (n_areas - 1) / 2) * 0.12
            mean = d['ldi_mean']
            sem = d['ldi_std'] / np.sqrt(d['n_ldi']) if d['n_ldi'] > 1 else np.nan
            color = COLORS.get(area, '#888888')
            ax.vlines(x, mean - sem, mean + sem, color=color, alpha=0.35, lw=5, zorder=1)
            ax.scatter(x, mean, color=color, s=45, edgecolors='k', linewidths=0.6,
                      zorder=3, label=area)

    ax.axhline(0.5, color='k', lw=1, ls='--')
    ax.set_xticks(range(len(_POS_VAR_KEYS)))
    ax.set_xticklabels(_POS_VAR_KEYS)
    ax.set_ylabel('LDI')
    ax.set_title('LDI by area and variable\n(band = within-area SEM of per-cell LDI)', fontsize=8)

    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys(), fontsize=6, ncol=n_areas,
             loc='upper center', bbox_to_anchor=(0.5, -0.12))

    path = os.path.join(out_dir, 'ldi_beeswarm.svg')
    _save_svg_png(fig, path)


def make_ldi_swarm_points_svg(all_cells, out_dir):
    """(5b) Companion to make_ldi_beeswarm_svg: instead of area-mean +/- SD
    bands, plot every individual cell's LDI (jittered by area within each
    variable column), colored by area. Per-area N is in the hundreds to low
    thousands, so this uses alpha-blended jittered points (a density-revealing
    strip plot) rather than a literal non-overlapping swarm, which would be
    illegible/slow at this N."""
    ldi_stats = _collect_ldi_mi_stats(all_cells)
    n_areas = len(_IMP_REGION_ORDER)
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=_scaled(6, 4.2), constrained_layout=True)
    for vi, v in enumerate(_POS_VAR_KEYS):
        for ai, area in enumerate(_IMP_REGION_ORDER):
            d = ldi_stats[area][v]
            vals = d['ldi_vals']
            if vals.size < MIN_CELLS_AREA:
                continue
            base_x = vi + (ai - (n_areas - 1) / 2) * 0.16
            jitter = rng.uniform(-0.06, 0.06, size=vals.size)
            color = COLORS.get(area, '#888888')
            ax.scatter(base_x + jitter, vals, color=color, s=3, alpha=0.2,
                      linewidths=0, zorder=2)

    ax.axhline(0.5, color='k', lw=1, ls='--', zorder=3)
    ax.set_xticks(range(len(_POS_VAR_KEYS)))
    ax.set_xticklabels(_POS_VAR_KEYS)
    ax.set_ylabel('LDI (per cell)')
    ax.set_title('LDI by area and variable\n(every individual cell plotted, jittered)', fontsize=8)

    area_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS.get(a, '#888888'),
                           markeredgecolor='none', markersize=7, label=a) for a in _IMP_REGION_ORDER]
    ax.legend(handles=area_handles, title='Area', fontsize=6, title_fontsize=6,
             ncol=n_areas, loc='upper center', bbox_to_anchor=(0.5, -0.12))

    path = os.path.join(out_dir, 'ldi_swarm_points.svg')
    _save_svg_png(fig, path)


def make_mi_vs_ldi_scatter_svg(all_cells, out_dir):
    """(6) Scatter: x = mean CV MI (raw modulation depth), y = mean LDI.
    Makes the point that AM has the highest raw MI for eye variables but
    LDI near 0.5 (no strong light-dependence)."""
    ldi_stats = _collect_ldi_mi_stats(all_cells)

    fig, ax = plt.subplots(figsize=_scaled(5.5, 5.5), constrained_layout=True)
    plotted_ldi = []
    for area in _IMP_REGION_ORDER:
        for v in _POS_VAR_KEYS:
            d = ldi_stats[area][v]
            if d['n_mi'] < MIN_CELLS_AREA or d['n_ldi'] < MIN_CELLS_AREA:
                continue
            x, y = d['mi_mean'], d['ldi_mean']
            plotted_ldi.append(y)
            ax.scatter(x, y, color=COLORS.get(area, '#888888'), marker=_POS_MARKERS[v],
                      s=70, edgecolors='k', linewidths=0.6, zorder=3)
            if area == 'AM' and v in ('theta', 'phi'):
                # theta/phi points sit almost on top of each other here -- offset
                # the two labels in opposite directions so they don't overlap.
                offset = (-45, 8) if v == 'theta' else (8, -14)
                ax.annotate(f'AM-{v}', (x, y), textcoords='offset points',
                           xytext=offset, fontsize=7)

    ax.axhline(0.5, color='0.5', lw=0.8, ls='--')
    ax.set_xlim(0, 0.25)
    if plotted_ldi:
        half_range = 1.1 * max(abs(y - 0.5) for y in plotted_ldi)
        ax.set_ylim(0.5 - half_range, 0.5 + half_range)
    ax.set_xlabel('Mean CV MI (raw modulation depth)')
    ax.set_ylabel('Mean LDI')
    ax.set_title('Raw modulation depth vs. light-dependence\n'
                 '(AM: highest MI for eye vars, but LDI near 0.5)', fontsize=8)

    area_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS.get(a, '#888888'),
                            markeredgecolor='k', markersize=7, label=a) for a in _IMP_REGION_ORDER]
    var_handles = [Line2D([0], [0], marker=_POS_MARKERS[v], color='w', markerfacecolor='0.6',
                           markeredgecolor='k', markersize=7, label=v) for v in _POS_VAR_KEYS]
    leg1 = ax.legend(handles=area_handles, title='Area', loc='upper left',
                     fontsize=6, title_fontsize=6)
    ax.add_artist(leg1)
    ax.legend(handles=var_handles, title='Variable', loc='center left',
              bbox_to_anchor=(1.02, 0.5), fontsize=6, title_fontsize=6)

    path = os.path.join(out_dir, 'mi_vs_ldi_scatter.svg')
    _save_svg_png(fig, path)


def make_diverging_dotplot_svg(records, out_dir):
    """(7) Diverging dot plot: y=40 rows (area x variable), grouped into
    position (top) and velocity (bottom) sections, x=AI_dark-AI_light.
    Rows are y-tick-labeled by area only; a bracket + variable name to the
    left of the axis groups each block of 5 areas (instead of repeating
    'V1 theta', 'RL theta', ... on every row)."""
    from matplotlib.transforms import blended_transform_factory

    stats = _collect_imp_stats(records)
    var_keys = _POS_VAR_KEYS + _VEL_VAR_KEYS
    rows = []  # (area, value, group, var_key)
    for v in var_keys:
        grp = 'pos' if v in _POS_VAR_KEYS else 'vel'
        for area in _IMP_REGION_ORDER:
            d = stats[area][v]
            if d['n_light'] < MIN_CELLS_AREA or d['n_dark'] < MIN_CELLS_AREA:
                continue
            rows.append((area, d['dark_mean'] - d['light_mean'], grp, v))

    n_pos = sum(1 for r in rows if r[2] == 'pos')
    # Deliberately more compact than the other figures here, per request.
    fig, ax = plt.subplots(figsize=(3.4, 0.13 * len(rows) + 0.6), constrained_layout=True)
    ys = np.arange(len(rows))[::-1]
    for y, (_area, val, grp, _v) in zip(ys, rows):
        color = '0.35' if grp == 'pos' else '#1f77b4'
        ax.hlines(y, 0, val, color=color, lw=1.0, zorder=2)
        ax.scatter(val, y, color=color, s=14, zorder=3)

    ax.axvline(0, color='k', lw=0.8)
    if 0 < n_pos < len(rows):
        ax.axhline(ys[n_pos - 1] - 0.5, color='0.7', lw=0.8, ls='--')
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=4.5)
    ax.set_xlabel('AI$_{dark}$ - AI$_{light}$  (+ = dark dominant)', fontsize=6)
    ax.set_title('Position (top, mixed) vs.\nvelocity (bottom, uniformly dark-dominant)', fontsize=6.5)
    ax.tick_params(axis='x', labelsize=5.5)

    # Bracket + variable label for each contiguous run of rows sharing a var_key.
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    bracket_x, tick_len = -0.20, 0.03
    i = 0
    while i < len(rows):
        v = rows[i][3]
        j = i
        while j < len(rows) and rows[j][3] == v:
            j += 1
        y_top, y_bot = ys[i] + 0.4, ys[j - 1] - 0.4
        ax.plot([bracket_x, bracket_x], [y_top, y_bot], transform=trans,
                color='k', lw=0.7, clip_on=False)
        ax.plot([bracket_x, bracket_x + tick_len], [y_top, y_top], transform=trans,
                color='k', lw=0.7, clip_on=False)
        ax.plot([bracket_x, bracket_x + tick_len], [y_bot, y_bot], transform=trans,
                color='k', lw=0.7, clip_on=False)
        ax.text(bracket_x - 0.03, (y_top + y_bot) / 2, _ALL_VAR_LABELS[v], transform=trans,
                ha='right', va='center', fontsize=5, clip_on=False)
        i = j

    path = os.path.join(out_dir, 'diverging_dotplot.svg')
    _save_svg_png(fig, path)


def make_diff_histogram_svg(records, out_dir):
    """(8) Histogram of AI_dark-AI_light: position (grey, all 4 vars pooled)
    overlaid with velocity (colored by variable, 4 separate overlays)."""
    stats = _collect_imp_stats(records)
    bins = np.linspace(-0.3, 0.3, 31)  # 2x the original bin count

    pos_vals = []
    for v in _POS_VAR_KEYS:
        for area in _IMP_REGION_ORDER:
            d = stats[area][v]
            if d['n_light'] >= MIN_CELLS_AREA and d['n_dark'] >= MIN_CELLS_AREA:
                pos_vals.append(d['dark_mean'] - d['light_mean'])

    fig, ax = plt.subplots(figsize=_scaled(6, 4), constrained_layout=True)
    ax.hist(pos_vals, bins=bins, color='0.6', alpha=0.6, label='Position (all 4 vars)',
            edgecolor='k', linewidth=0.3)

    for v in _VEL_VAR_KEYS:
        vals = []
        for area in _IMP_REGION_ORDER:
            d = stats[area][v]
            if d['n_light'] >= MIN_CELLS_AREA and d['n_dark'] >= MIN_CELLS_AREA:
                vals.append(d['dark_mean'] - d['light_mean'])
        ax.hist(vals, bins=bins, color=_VAR_COLORS[v], alpha=0.55,
                label=_ALL_VAR_LABELS[v], edgecolor='k', linewidth=0.3)

    ax.axvline(0, color='k', lw=1, ls='--')
    ax.set_xlabel('AI$_{dark}$ - AI$_{light}$')
    ax.set_ylabel('Count (area x variable combinations)')
    ax.set_title('Position (grey) centered near zero; velocity (colored) shifted positive', fontsize=8)
    ax.legend(fontsize=6)

    path = os.path.join(out_dir, 'diff_histogram.svg')
    _save_svg_png(fig, path)


def make_concordance_bar_svg(pts, out_dir):
    """(9) Bar chart of observed LDI-vs-AI sign concordance rate against
    reference expectations: pure multiplicative (100%, most concordant),
    chance (50%), pure additive (0%, most discordant)."""
    n = len(pts)
    n_concordant = sum(1 for _, _, x, y in pts if np.sign(x) == np.sign(y))
    observed = n_concordant / n if n else np.nan

    fig, ax = plt.subplots(figsize=_scaled(4.2, 4.2), constrained_layout=True)
    labels = ['Pure\nmultiplicative', 'Equal\nmixture', 'Pure\nadditive', 'Observed']
    values = [1.0, 0.5, 0.0, observed]
    colors = ['0.75', '0.75', '0.75', '#D95F02']
    ax.bar(labels, values, color=colors, edgecolor='k', linewidth=0.6)
    for i, v in enumerate(values):
        ax.text(i, v + 0.03, f'{v:.0%}', ha='center', fontsize=8)

    ax.axhline(0.5, color='k', lw=0.8, ls=':')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Concordance rate\n(sign(LDI-0.5) == sign(AI$_{light}$-AI$_{dark}$))')
    ax.set_title(f'Observed: {n_concordant}/{n} position area x variable pairs', fontsize=8)

    path = os.path.join(out_dir, 'concordance_bar.svg')
    _save_svg_png(fig, path)
    print(f'Concordance rate: {n_concordant}/{n} = {observed:.1%} '
          f'(multiplicative=100%, equal mixture=50%, additive=0%)')


def make_ai_ci_bar_svg(records, out_dir):
    """(10) Bar chart: mean ablation index (pooled across all 8 variables)
    per area, light vs. dark, with 95% CI error bars (mean +/- 1.96*SEM).
    Precision tracks N: tight CI for V1 (large N), wider CI for the other
    areas (much smaller N)."""
    areas = _IMP_REGION_ORDER
    light_pool = {a: [] for a in areas}
    dark_pool  = {a: [] for a in areas}
    for r in records:
        a = r['area']
        if a not in light_pool:
            continue
        for vi, var in enumerate(_IMP_VAR_ORDER):
            if var not in _ALL_VAR_KEYS:
                continue
            vl, vd = float(r['light_imp'][vi]), float(r['dark_imp'][vi])
            if np.isfinite(vl):
                light_pool[a].append(vl)
            if np.isfinite(vd):
                dark_pool[a].append(vd)

    fig, ax = plt.subplots(figsize=_scaled(6, 4), constrained_layout=True)
    width = 0.35
    print('\nMean AI (pooled across all 8 variables) +/- 95% CI, by area:')
    for xi, a in enumerate(areas):
        color = COLORS.get(a, '#888888')
        for pool, xoff, alpha, hatch, cond in [
            (light_pool[a], -width / 2, 0.85, None,   'light'),
            (dark_pool[a],   width / 2, 0.5,  _HATCH, 'dark'),
        ]:
            vals = np.array(pool)
            if vals.size < 2:
                continue
            mean = vals.mean()
            ci95 = 1.96 * vals.std(ddof=1) / np.sqrt(vals.size)
            ax.bar(xi + xoff, mean, width=width, yerr=ci95, capsize=3, color=color,
                  alpha=alpha, hatch=hatch, edgecolor='k', linewidth=0.6,
                  error_kw=dict(lw=1.0))
            print(f'  {a} {cond}: n={vals.size}, mean={mean:.3f}, '
                  f'95% CI=[{mean - ci95:.3f}, {mean + ci95:.3f}]')

    ax.set_xticks(range(len(areas)))
    ax.set_xticklabels(areas)
    ax.set_ylabel('Mean ablation index (pooled across variables)')
    ax.set_title('Mean AI by area, light vs. dark\n(error bars = 95% CI; solid=light, hatched=dark)', fontsize=8)

    path = os.path.join(out_dir, 'ai_ci_bar.svg')
    _save_svg_png(fig, path)


def make_ai_ridgeline_svg(records, out_dir):
    """(11) Ridgeline plot: one stacked density per variable (8 total,
    position block then velocity block), x = per-cell ablation index [0,1],
    pooled across areas and light/dark conditions. Demonstrates that, among
    cells whose ablation index didn't hit the [0,1] clip ceiling, no
    variable's per-cell distribution approaches that ceiling -- distributed,
    multiplexed coding holds at the single-cell level, not just in the
    area-level means.

    NOTE: the per-feature ablation index formula in compute_permutation_
    importance() (ffNLE.py) only floors at 0 -- it has no upper clip like
    calc_ablation_index()/compute_group_importance() do. We clip to [0,1] at
    load time (_load_importance_cells), which means cells whose raw value
    was >1 (a real, non-trivial 5-14% of cells per variable; some raw values
    run into the hundreds) get piled up exactly at 1.0. Plotting those
    cells would put a spurious mode right at the ceiling -- the opposite of
    the point being made -- so they are excluded here and the excluded
    fraction is reported instead."""
    from scipy.stats import gaussian_kde

    var_keys = _HEATMAP_VAR_ORDER  # position block, then velocity block
    xs = np.linspace(0, 1, 400)

    fig, ax = plt.subplots(figsize=_scaled(5, 7), constrained_layout=True)
    offset_step = 1.15
    print('\nPer-cell ablation index (pooled across areas and conditions, '
          'excluding clip-ceiling cells):')
    for i, var in enumerate(var_keys):
        vi = _IMP_VAR_ORDER.index(var)
        vals = []
        for r in records:
            vl, vd = float(r['light_imp'][vi]), float(r['dark_imp'][vi])
            if np.isfinite(vl):
                vals.append(vl)
            if np.isfinite(vd):
                vals.append(vd)
        vals = np.array(vals)
        n_total = vals.size
        at_ceiling = vals >= 0.999
        n_ceiling = int(at_ceiling.sum())
        vals = vals[~at_ceiling]
        if vals.size < 10:
            continue
        kde = gaussian_kde(vals, bw_method=0.12)
        density = kde(xs)
        density = density / density.max()
        y_base = (len(var_keys) - 1 - i) * offset_step  # position block on top
        color = _VAR_COLORS.get(var, '#888888')
        ax.fill_between(xs, y_base, y_base + density, color=color, alpha=0.7,
                        edgecolor='k', linewidth=0.6, zorder=3 + i)
        ax.axhline(y_base, color='0.85', lw=0.5, zorder=1)
        ax.text(-0.02, y_base + 0.08, _ALL_VAR_LABELS[var], ha='right', va='bottom',
                fontsize=7, clip_on=False)
        ax.text(1.02, y_base + 0.08, f'{100 * n_ceiling / n_total:.1f}% clipped',
                ha='left', va='bottom', fontsize=5.5, color='0.4', clip_on=False)
        print(f'  {_ALL_VAR_LABELS[var]}: n={vals.size} (excluded {n_ceiling}/{n_total} '
              f'= {100 * n_ceiling / n_total:.1f}% at clip ceiling), mean={vals.mean():.3f}, '
              f'95th pctile={np.percentile(vals, 95):.3f}, max={vals.max():.3f}')

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    for spine in ('left', 'top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.set_xlabel('Ablation index (per cell, excluding clip-ceiling cells)')
    ax.set_title('Per-cell ablation index distributions\n'
                 '(pooled across areas + light/dark, clip-ceiling cells excluded;\n'
                 'no mode approaches 1.0 among the rest)', fontsize=8)

    path = os.path.join(out_dir, 'ai_ridgeline.svg')
    _save_svg_png(fig, path)


def make_velocity_fold_trend_svg(records, out_dir):
    """(12) Dot-and-line plot: x = area, ordered V1 -> RL -> AM -> PM -> A
    (rough hierarchical distance from V1), y = mean fold change (dark AI /
    light AI) averaged across the 4 velocity variables, error bars = SEM
    across those 4 variables (n=4 per area). A monotonic trend would
    suggest the magnitude of velocity reweighting in the dark scales with
    anatomical distance from V1."""
    stats = _collect_imp_stats(records)
    areas = _IMP_REGION_ORDER

    means, sems = [], []
    print('\nVelocity fold change (dark AI / light AI), averaged across 4 velocity vars:')
    for a in areas:
        folds = []
        for v in _VEL_VAR_KEYS:
            d = stats[a][v]
            if (d['n_light'] >= MIN_CELLS_AREA and d['n_dark'] >= MIN_CELLS_AREA
                    and d['light_mean'] > 0):
                folds.append(d['dark_mean'] / d['light_mean'])
        folds = np.array(folds)
        mean = float(folds.mean()) if folds.size else np.nan
        sem = float(folds.std(ddof=1) / np.sqrt(folds.size)) if folds.size > 1 else np.nan
        means.append(mean)
        sems.append(sem)
        print(f'  {a}: n_vars={folds.size}, mean={mean:.3f}, SEM={sem:.3f}, '
              f'folds={np.round(folds, 3).tolist()}')

    fig, ax = plt.subplots(figsize=_scaled(4.5, 4), constrained_layout=True)
    xs = np.arange(len(areas))
    ax.plot(xs, means, '-', color='0.3', lw=1.2, zorder=1)
    for xi, a in enumerate(areas):
        ax.errorbar(xi, means[xi], yerr=sems[xi], fmt='o', color=COLORS.get(a, '#888888'),
                    ecolor='k', elinewidth=1.0, capsize=3, markersize=8,
                    markeredgecolor='k', markeredgewidth=0.6, zorder=3)

    ax.axhline(1.0, color='k', lw=0.8, ls='--')
    ax.set_xticks(xs)
    ax.set_xticklabels(areas)
    ax.set_xlabel('Area (V1 -> ... -> A, rough hierarchical order)')
    ax.set_ylabel('Mean velocity fold change\n(dark AI / light AI, averaged over 4 vars)')
    ax.set_title('Velocity reweighting in the dark vs. anatomical distance from V1', fontsize=8)

    path = os.path.join(out_dir, 'velocity_fold_trend.svg')
    _save_svg_png(fig, path)


def _join_mi_ai_cells(all_cells, records):
    """Join the tuning-curve dataset (all_cells) and the GLM-pooled dataset
    (records) on (animal, pos, ci). Verified 1:1 with zero area mismatches
    on the current pooled files (5246/5246 matched)."""
    ai_lookup = {(r['animal'], r['pos'], r['ci']): r for r in records}

    joined = []
    n_area_mismatch = 0
    for c in all_cells:
        r = ai_lookup.get((c['animal'], c['pos'], c['ci']))
        if r is None:
            continue
        if r['area'] != c['area']:
            n_area_mismatch += 1
            continue
        joined.append((c, r))

    print(f'MI/AI per-cell join: {len(joined)} matched cells '
          f'({n_area_mismatch} area mismatches dropped)')
    return joined


def make_mi_ai_percell_scatter_svg(all_cells, records, out_dir):
    """(13) Per-cell scatter: x = CV MI, y = ablation index, one panel per
    position variable (theta/phi/pitch/roll), colored by area. Light and
    dark conditions are both included as separate points per cell."""
    joined = _join_mi_ai_cells(all_cells, records)

    # Pre-sort matched cells by area once (not per variable/condition), then
    # build whole-array (x, y) vectors per area and issue one scatter() call
    # per area/panel instead of one call per point -- a few thousand
    # individual scatter() calls in a Python loop is extremely slow.
    by_area = {a: [] for a in _IMP_REGION_ORDER}
    for c, r in joined:
        if c['area'] in by_area:
            by_area[c['area']].append((c, r))

    fig, axes = plt.subplots(1, 4, figsize=_scaled(7.2, 2.2), constrained_layout=True,
                              sharex=True, sharey=True)
    for vi, v in enumerate(_POS_VAR_KEYS):
        ax = axes[vi]
        ai_vi = _IMP_VAR_ORDER.index(v)
        for area in _IMP_REGION_ORDER:
            pairs = by_area[area]
            if not pairs:
                continue
            mi_l = np.array([c[f'{v}_rel'] for c, _ in pairs])
            mi_d = np.array([c[f'{v}_rel_dark'] for c, _ in pairs])
            ai_l = np.array([r['light_imp'][ai_vi] for _, r in pairs])
            ai_d = np.array([r['dark_imp'][ai_vi] for _, r in pairs])
            x = np.concatenate([mi_l, mi_d])
            y = np.concatenate([ai_l, ai_d])
            finite = np.isfinite(x) & np.isfinite(y)
            color = COLORS.get(area, '#888888')
            ax.scatter(x[finite], y[finite], color=color, s=3, alpha=0.2,
                      linewidths=0, zorder=2)
        ax.set_title(v, fontsize=7)
        ax.set_xlabel('CV MI', fontsize=6)
        if vi == 0:
            ax.set_ylabel('Ablation index', fontsize=6)
        ax.tick_params(labelsize=5)

    area_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS.get(a, '#888888'),
                            markeredgecolor='none', markersize=6, label=a) for a in _IMP_REGION_ORDER]
    fig.legend(handles=area_handles, title='Area', fontsize=5, title_fontsize=5,
              loc='center left', bbox_to_anchor=(1.0, 0.5))

    path = os.path.join(out_dir, 'mi_ai_percell_scatter.svg')
    _save_svg_png(fig, path)


def make_case_study_svg(all_cells, records, out_dir):
    """(14) Two case studies stacked in a 2x3 grid: RL-pitch (row 0, the
    strongest light-leaning effect by both metrics) and AM-phi (row 1, the
    starkest MI-vs-AI mismatch). Col 1 = example single-cell tuning curve
    (light vs. dark; the most-modulated cell with finite LDI in that
    area/variable). Col 2 = area-level LDI decomposition (light_frac=LDI,
    dark_frac=1-LDI, ref line at 0.5). Col 3 = area-level mean ablation
    index (light, dark bars)."""
    ldi_stats = _collect_ldi_mi_stats(all_cells)
    imp_stats = _collect_imp_stats(records)
    cases = [('RL', 'pitch', 'Pitch'), ('AM', 'phi', r'$\phi$')]

    fig, axes = plt.subplots(2, 3, figsize=_scaled(7.5, 5), constrained_layout=True)

    for row, (area, vname, vlabel) in enumerate(cases):
        color = COLORS.get(area, '#888888')
        cells_area = [c for c in all_cells if c['area'] == area]
        candidates = [
            c for c in cells_area
            if c[f'{vname}_tuning'] is not None
            and np.isfinite(c[f'{vname}_rel'])
            and np.isfinite(_ldi(c[f'{vname}_rel'], c[f'{vname}_rel_dark']))
        ]

        ax_tc = axes[row, 0]
        if candidates:
            best = max(candidates, key=lambda c: c[f'{vname}_rel'])
            bins = best[f'{vname}_bins']
            tc_l, tc_d = best[f'{vname}_tuning'], best[f'{vname}_tuning_dark']
            err_l, err_d = best[f'{vname}_err'], best[f'{vname}_err_dark']
            ax_tc.plot(bins, tc_l, color=color, lw=1.5, label='Light')
            if err_l is not None:
                ax_tc.fill_between(bins, tc_l - err_l, tc_l + err_l, alpha=0.25, color=color)
            if tc_d is not None:
                ax_tc.plot(bins, tc_d, color=color, lw=1.2, ls='--', label='Dark')
                if err_d is not None:
                    _hatch_polygon(ax_tc, bins, tc_d - err_d, tc_d + err_d, color, alpha=0.20)
            ax_tc.legend(fontsize=5, loc='best')
        ax_tc.set_title(f'{area} {vlabel}: example cell tuning', fontsize=7)
        ax_tc.set_xlabel(f'{vlabel} (deg)', fontsize=6)
        ax_tc.set_ylabel('Firing rate', fontsize=6)
        ax_tc.tick_params(labelsize=5)

        ax_ldi = axes[row, 1]
        ldi_mean = ldi_stats[area][vname]['ldi_mean']
        ax_ldi.bar(0, ldi_mean, color=color, alpha=0.85, edgecolor='k', linewidth=0.6)
        ax_ldi.bar(1, 1 - ldi_mean, color=color, alpha=0.5, hatch=_HATCH, edgecolor='k', linewidth=0.6)
        ax_ldi.axhline(0.5, color='k', lw=1, ls='--')
        ax_ldi.set_xticks([0, 1])
        ax_ldi.set_xticklabels(['Light frac\n(=LDI)', 'Dark frac'], fontsize=6)
        ax_ldi.set_ylim(0, 1)
        ax_ldi.set_title(f'{area} {vlabel}: LDI = {ldi_mean:.2f}', fontsize=7)
        ax_ldi.tick_params(labelsize=5)

        ax_ai = axes[row, 2]
        d_ai = imp_stats[area][vname]
        ax_ai.bar(0, d_ai['light_mean'], yerr=d_ai['light_sem'], capsize=3, color=color,
                 alpha=0.85, edgecolor='k', linewidth=0.6)
        ax_ai.bar(1, d_ai['dark_mean'], yerr=d_ai['dark_sem'], capsize=3, color=color,
                 alpha=0.5, hatch=_HATCH, edgecolor='k', linewidth=0.6)
        ax_ai.set_xticks([0, 1])
        ax_ai.set_xticklabels(['Light', 'Dark'], fontsize=6)
        ax_ai.set_ylim(0, 1)
        ax_ai.set_ylabel('Ablation index', fontsize=6)
        ax_ai.set_title(f'{area} {vlabel}: AI light={d_ai["light_mean"]:.2f}, '
                        f'dark={d_ai["dark_mean"]:.2f}', fontsize=6.5)
        ax_ai.tick_params(labelsize=5)

    path = os.path.join(out_dir, 'case_study_RL_pitch_AM_phi.svg')
    _save_svg_png(fig, path)


def make_permutation_null_svg(records, out_dir, n_perm=10000, seed=0):
    """(15) Permutation test: for the 20 position-variable area x variable
    pairs and the 20 velocity-variable pairs, count how many are
    dark-dominant (AI_dark > AI_light). Build a null distribution by
    randomly flipping which member of each pair is called 'light' vs.
    'dark' and recomputing the count, n_perm times. Mark the observed
    counts with vertical lines."""
    stats = _collect_imp_stats(records)
    rng = np.random.default_rng(seed)

    def gather(var_keys):
        light, dark = [], []
        for v in var_keys:
            for area in _IMP_REGION_ORDER:
                d = stats[area][v]
                if d['n_light'] < MIN_CELLS_AREA or d['n_dark'] < MIN_CELLS_AREA:
                    continue
                light.append(d['light_mean'])
                dark.append(d['dark_mean'])
        return np.array(light), np.array(dark)

    def null_counts(light, dark, n_perm):
        n = len(light)
        flips = rng.random((n_perm, n)) < 0.5
        dark_wins = np.where(flips, light > dark, dark > light)
        return dark_wins.sum(axis=1)

    pos_light, pos_dark = gather(_POS_VAR_KEYS)
    vel_light, vel_dark = gather(_VEL_VAR_KEYS)
    pos_obs = int(np.sum(pos_dark > pos_light))
    vel_obs = int(np.sum(vel_dark > vel_light))
    n_pos, n_vel = len(pos_light), len(vel_light)

    pos_null = null_counts(pos_light, pos_dark, n_perm)
    vel_null = null_counts(vel_light, vel_dark, n_perm)

    fig, axes = plt.subplots(1, 2, figsize=_scaled(8, 3.4), constrained_layout=True)
    bins = np.arange(-0.5, max(n_pos, n_vel) + 1.5, 1)

    axes[0].hist(pos_null, bins=bins, color='0.6', edgecolor='k', linewidth=0.3)
    axes[0].axvline(pos_obs, color='#D95F02', lw=2)
    axes[0].set_title(f'Position vars: observed {pos_obs}/{n_pos}', fontsize=8)
    axes[0].set_xlabel('# dark-dominant pairs (permuted)')
    axes[0].set_ylabel('Permutation count')

    axes[1].hist(vel_null, bins=bins, color='0.6', edgecolor='k', linewidth=0.3)
    axes[1].axvline(vel_obs, color='#1f77b4', lw=2)
    axes[1].set_title(f'Velocity vars: observed {vel_obs}/{n_vel}', fontsize=8)
    axes[1].set_xlabel('# dark-dominant pairs (permuted)')

    fig.suptitle(f'Permutation null (n={n_perm}): random light/dark relabeling', fontsize=8)

    path = os.path.join(out_dir, 'permutation_null.svg')
    _save_svg_png(fig, path)

    p_pos = np.mean(pos_null >= pos_obs) if pos_obs >= n_pos / 2 else np.mean(pos_null <= pos_obs)
    p_vel = np.mean(vel_null >= vel_obs) if vel_obs >= n_vel / 2 else np.mean(vel_null <= vel_obs)
    print(f'Position: observed {pos_obs}/{n_pos}, null mean={pos_null.mean():.2f}, '
          f'one-sided p~{p_pos:.4f}')
    print(f'Velocity: observed {vel_obs}/{n_vel}, null mean={vel_null.mean():.2f}, '
          f'one-sided p~{p_vel:.4f}')


def main():
    parser = argparse.ArgumentParser(
        description='Summarize 1-D tuning across all variables and visual areas.')
    parser.add_argument('--pooled',    default=DEFAULT_POOLED)
    parser.add_argument('--base_dir',  default=DEFAULT_BASE)
    parser.add_argument('--out_dir',   default=DEFAULT_OUT_DIR)
    parser.add_argument('--threshold', type=float, default=MOD_THRESHOLD,
                        help='CV MI threshold for % modulated page')
    parser.add_argument('--pooled_glm', default=DEFAULT_POOLED_GLM,
                        help='GLM pooled h5 for importance SVG')
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
    print_mi_ldi_stats(all_cells)

    # ---- SVG exports ----
    make_combined_overview_svg(all_cells, args.out_dir)
    make_overview_mi_ldi_boxstrip_svg(all_cells, args.out_dir)
    make_example_tuning_svgs(all_cells, args.out_dir)

    records = []
    if os.path.exists(args.pooled_glm):
        records = _load_importance_cells(args.pooled_glm)
    make_importance_svg(args.out_dir, records=records)

    if records:
        make_ablation_heatmap_svg(records, args.out_dir)
        make_slope_graph_svg(records, args.out_dir)
        for layout in ('grouped', 'faceted'):
            for scope in ('velocity', 'all8'):
                make_fold_change_svg(records, args.out_dir, layout=layout, scope=scope)
        pts = make_ldi_vs_ai_scatter_svg(records, all_cells, args.out_dir)
        make_ldi_beeswarm_svg(all_cells, args.out_dir)
        make_ldi_swarm_points_svg(all_cells, args.out_dir)
        make_mi_vs_ldi_scatter_svg(all_cells, args.out_dir)
        make_diverging_dotplot_svg(records, args.out_dir)
        make_diff_histogram_svg(records, args.out_dir)
        make_concordance_bar_svg(pts, args.out_dir)

        make_ai_ci_bar_svg(records, args.out_dir)
        make_ai_ridgeline_svg(records, args.out_dir)
        make_velocity_fold_trend_svg(records, args.out_dir)
        make_mi_ai_percell_scatter_svg(all_cells, records, args.out_dir)
        make_case_study_svg(all_cells, records, args.out_dir)
        make_permutation_null_svg(records, args.out_dir)
    else:
        print('No importance data — skipping ablation-index comparison figures.')

    _finish_pending_saves()

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

        make_ldi_page(pdf, all_cells)            # violins per area
        make_ldi_histogram_pages(pdf, all_cells) # one page per area, horizontal histograms
        make_ldi_summary_heatmap(pdf, all_cells) # area × variable median heatmap
        make_ldi_cdf_page(pdf, all_cells)        # cumulative distributions
        make_ldi_fraction_page(pdf, all_cells)   # % light vs dark dominant
        make_ldi_scatter_page(pdf, all_cells)    # light vs dark MI scatter

        # make_heatmap_pages(pdf, all_cells)
        make_per_area_pages(pdf, all_cells)

    print(f'Done. PDF: {pdf_path}')


if __name__ == '__main__':
    main()
