# -*- coding: utf-8 -*-

if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import kurtosis as scipy_kurtosis

import pandas as pd

from .utils.files import read_h5

import matplotlib as mpl
mpl.rcParams['axes.spines.top']   = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype']      = 42
mpl.rcParams['ps.fonttype']       = 42
mpl.rcParams['font.size']         = 8


class FigureSaver:

    def __init__(self, out_dir: str, stem: str = 'fig'):
        os.makedirs(out_dir, exist_ok=True)
        self._dir  = out_dir
        self._stem = stem
        self._n    = 0

    def savefig(self, fig):
        base = os.path.join(self._dir, f'{self._stem}_{self._n:03d}')
        fig.savefig(base + '.png', bbox_inches='tight')
        fig.savefig(base + '.svg', bbox_inches='tight')
        self._n += 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


REGION_ORDER = ['V1', 'RL', 'AM', 'PM', 'A', 'AL', 'LM', 'P']
REGION_IDS   = {'V1': 5, 'RL': 2, 'AM': 3, 'PM': 4, 'A': 10, 'AL': 7, 'LM': 8, 'P': 9}
ID_TO_NAME   = {v: k for k, v in REGION_IDS.items()}

SUMMARY_AREAS = ['V1', 'RL', 'AM', 'PM', 'A']
SUMMARY_IDS   = {k: REGION_IDS[k] for k in SUMMARY_AREAS}

COLORS = {
    'theta':  '#4B9CD3', 'phi':    '#E07B54',
    'pitch':  '#5BAD6F', 'roll':   '#C55A5A',
    'yaw':    '#8B68AB', 'dTheta': '#85C1E9',
    'dPhi':   '#F0A882', 'dPitch': '#85C99A',
    'dYaw':   '#B8A0CC', 'dRoll':  '#E09090',
    'V1':     '#1B9E77', 'RL':     '#D95F02',
    'AM':     '#7570B3', 'PM':     '#E7298A',
    'A':      '#1A5A34', 'AL':     '#E6AB02',
    'LM':     '#A6761D', 'P':      '#666666',
}

_VAR_RDATA = {
    'theta': 'theta', 'phi': 'phi',
    'dTheta': 'dTheta', 'dPhi': 'dPhi',
    'pitch': 'pitch', 'yaw': 'yaw', 'roll': 'roll',
    'dPitch': 'gyro_y', 'dYaw': 'gyro_z', 'dRoll': 'gyro_x',
}

def _imp_key(var, cond):
    base = _VAR_RDATA.get(var, var)
    cond_str = 'Light' if cond == 'l' else 'Dark'
    return f'{base}_train{cond_str}_test{cond_str}_importance_{base}'


def _cell_regions(transform, labeled_array, n_cells):

    h, w = labeled_array.shape
    xi = np.clip(transform[:n_cells, 2].astype(int), 0, w - 1)
    yi = np.clip(transform[:n_cells, 3].astype(int), 0, h - 1)
    return labeled_array[yi, xi]


def _get_cv_mi(rdata, var, cond):

    use_key = _VAR_RDATA.get(var, var)
    raw = rdata.get(f'{use_key}_{cond}_rel', None)
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=float)
    return arr if arr.size > 0 else None


def _get_importance(model_data, var, cond):
    """Return feature-importance array, trying several plausible key formats."""
    if not isinstance(model_data, dict):
        return None

    base = _VAR_RDATA.get(var, var)
    cond_str = 'Light' if cond == 'l' else 'Dark'

    # Try keys in order of specificity (most specific first)
    # Primary pattern: {base}_train{Cond}_test{Cond}_importance_{base}
    candidates = [
        _imp_key(var, cond),
        f'{var}_train{cond_str}_test{cond_str}_importance_{var}',
        f'full_train{cond_str}_test{cond_str}_importance_{base}',
        f'full_train{cond_str}_test{cond_str}_importance_{var}',
        f'full_importance_{base}',
        f'importance_{base}',
        f'importance_{var}',
    ]
    for key in candidates:
        raw = model_data.get(key, None)
        if raw is not None:
            arr = np.asarray(raw, dtype=float)
            if arr.size > 0:
                return arr
    return None


def _draw_violin(ax, data, pos, color, hatch=None, width=0.38, median_color='k'):

    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 3:
        return
    vp = ax.violinplot([data], positions=[pos], widths=width,
                       showmeans=False, showmedians=True, showextrema=False)
    for body in vp['bodies']:
        if hatch:
            body.set_facecolor(color)
            body.set_alpha(0.3)
            body.set_hatch(hatch)
            body.set_edgecolor(color)
        else:
            body.set_facecolor(color)
            body.set_alpha(0.75)
            body.set_edgecolor(color)
    for key in ('cmedians', 'cmeans', 'cbars', 'cmins', 'cmaxes'):
        if key in vp:
            vp[key].set_color(median_color)
            vp[key].set_linewidth(1.2)


def _animal_dirs_from_pooled(pooled_data):
    return [k for k in pooled_data if isinstance(pooled_data[k], dict)
            and 'transform' in pooled_data[k]]

_VAR_XLABEL = {
    'theta':  'theta (deg)',
    'dTheta': 'dTheta (deg/s)',
    'phi':    'phi (deg)',
    'dPhi':   'dPhi (deg/s)',
    'pitch':  'pitch (deg)',
    'dPitch': 'dPitch (deg/s)',
    'roll':   'roll (deg)',
    'dRoll':  'dRoll (deg/s)',
    'yaw':    'yaw (deg)',
    'dYaw':   'dYaw (deg/s)',
}

def plot_top8_tuning_curves(topo_data, var, cond, pdf, rng_seed=42):

    dk = f'sorted_tuning_curves_{cond}'
    if dk not in topo_data or var not in topo_data[dk]:
        print(f'  [top8] no data for {var} ({cond})')
        return

    d      = topo_data[dk][var]
    mods   = np.asarray(d['mods'],   dtype=float)
    tuning = np.asarray(d['tuning'], dtype=float)
    errs   = np.asarray(d['errs'],   dtype=float)
    bins   = np.asarray(d['bins'],   dtype=float)

    if bins.size == tuning.shape[1] + 1:
        centers = 0.5 * (bins[:-1] + bins[1:])
    else:
        centers = bins

    n_total = len(mods)
    if var == 'dTheta':

        top10_n = max(8, int(np.ceil(n_total * 0.10)))
        top10_n = min(top10_n, n_total)
        pool    = np.arange(top10_n)
        rng     = np.random.default_rng(rng_seed)
        chosen  = np.sort(rng.choice(pool, size=min(8, len(pool)), replace=False))
    else:
        chosen = np.arange(min(8, n_total))   # top 8 in order

    color  = COLORS.get(var, 'k')
    xlabel = _VAR_XLABEL.get(var, var)

    fig, axs = plt.subplots(2, 4, figsize=(9, 4.2), dpi=200)
    axs = axs.flatten()
    for slot, ax in enumerate(axs):
        if slot < len(chosen):
            idx = chosen[slot]
            ax.plot(centers, tuning[idx], color=color, lw=1.4)
            ax.fill_between(centers, tuning[idx] - errs[idx], tuning[idx] + errs[idx],
                            color=color, alpha=0.25)
            ax.set_title(f'modulation = {mods[idx]:.2f}')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('inferred spike rate')
        else:
            ax.axis('off')

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

def plot_dtheta_tuning_kurtosis_pages(topo_data, cond, out_dir, stem,
                                      n_pages=5, cells_per_page=8):

    dk = f'sorted_tuning_curves_{cond}'
    if dk not in topo_data or 'dTheta' not in topo_data[dk]:
        print(f'  [dTheta kurtosis pages] no data for dTheta ({cond}) — skipping')
        return

    d      = topo_data[dk]['dTheta']
    mods   = np.asarray(d['mods'],   dtype=float)
    tuning = np.asarray(d['tuning'], dtype=float)   # (N_cells, N_bins)
    errs   = np.asarray(d['errs'],   dtype=float)
    bins   = np.asarray(d['bins'],   dtype=float)

    if bins.size == tuning.shape[1] + 1:
        centers = 0.5 * (bins[:-1] + bins[1:])
    else:
        centers = bins

    n_cells = tuning.shape[0]

    kurt = np.array([
        scipy_kurtosis(tuning[i], fisher=True, nan_policy='omit')
        if np.any(np.isfinite(tuning[i])) else -np.inf
        for i in range(n_cells)
    ])
    ranked = np.argsort(kurt)[::-1]   # descending: highest kurtosis first

    n_needed = n_pages * cells_per_page
    ranked   = ranked[:min(n_needed, len(ranked))]

    color  = COLORS.get('dTheta', '#85C1E9')
    xlabel = _VAR_XLABEL.get('dTheta', 'dTheta (deg/s)')
    cond_label = 'light' if cond == 'l' else 'dark'

    os.makedirs(out_dir, exist_ok=True)

    for page in range(n_pages):
        page_start = page * cells_per_page
        page_idx   = ranked[page_start: page_start + cells_per_page]

        if len(page_idx) == 0:
            break

        fig, axs = plt.subplots(2, 4, figsize=(9, 4.2), dpi=200)
        axs = axs.flatten()

        for slot, ax in enumerate(axs):
            if slot < len(page_idx):
                idx = page_idx[slot]
                ax.plot(centers, tuning[idx], color=color, lw=1.4)
                ax.fill_between(centers,
                                tuning[idx] - errs[idx],
                                tuning[idx] + errs[idx],
                                color=color, alpha=0.25)
                ax.set_title(
                    f'mod={mods[idx]:.2f}  kurt={kurt[idx]:.1f}',
                    fontsize=7,
                )
                ax.set_xlabel(xlabel)
                ax.set_ylabel('inferred spike rate')
            else:
                ax.axis('off')

        fig.suptitle(
            f'dTheta tuning — {cond_label} — kurtosis rank '
            f'{page_start + 1}–{page_start + len(page_idx)}',
            fontsize=9,
        )
        fig.tight_layout()

        base = os.path.join(out_dir, f'{stem}_dTheta_kurtosis_{cond}_page{page + 1:02d}')
        fig.savefig(base + '.png', bbox_inches='tight')
        fig.savefig(base + '.svg', bbox_inches='tight')
        plt.close(fig)
        print(f'    saved page {page + 1}: {base}.png / .svg')


def _ecdf(values):

    v = np.sort(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.array([]), np.array([])
    return v, np.arange(1, len(v) + 1) / len(v)


def plot_modulation_cdfs(topo_data, pdf):
 
    MIN_RATE = 0.05

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), dpi=200)

    for ax, var in zip(axes, ['theta', 'phi']):
        for area in ['All'] + SUMMARY_AREAS:
            color = COLORS.get(var, 'k') if area == 'All' else COLORS.get(area, 'k')

            for cond, ls, lw, alpha in [('l', '-', 1.4, 0.9), ('d', '--', 1.1, 0.7)]:
                dk = f'variable_summary_{var}_{cond}'
                if dk not in topo_data:
                    continue
                df = pd.DataFrame(topo_data[dk])
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                if area == 'All':
                    mask = (df['mean_rate'] > MIN_RATE) if 'mean_rate' in df.columns \
                           else pd.Series(True, index=df.index)
                else:
                    rid  = REGION_IDS[area]
                    mask = df['region'] == rid
                    if 'mean_rate' in df.columns:
                        mask = mask & (df['mean_rate'] > MIN_RATE)

                vals = df.loc[mask, 'mod'].dropna().values
                x, y = _ecdf(vals)
                if len(x) == 0:
                    continue
                label = f'{area} L' if cond == 'l' else f'{area} D'
                ax.plot(x, y, ls=ls, lw=lw, color=color, alpha=alpha, label=label)

        ax.axvline(0.33, color='grey', ls='--', lw=0.8, alpha=0.4)
        ax.axvline(0.50, color='grey', ls='--', lw=0.8, alpha=0.4)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 1)
        ax.set_xlabel('modulation')
        ax.set_ylabel('cumulative fraction')
        ax.set_title(var, color=COLORS.get(var, 'k'))
        ax.legend(frameon=False, ncol=2)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _compute_fov_pcts(pooled_data, labeled_array, var, cond,
                      metric='mod', mod_thresh=0.33, imp_thresh=0.02,
                      min_cells=5):

    animal_dirs = _animal_dirs_from_pooled(pooled_data)
    result      = {a: [] for a in SUMMARY_AREAS}

    n_fovs_seen = 0
    n_fovs_no_key = 0

    for animal in animal_dirs:
        adat = pooled_data[animal]
        for poskey in adat.get('messentials', {}).keys():
            if not poskey.startswith('pos'):
                continue

            messentials = adat.get('messentials', {}).get(poskey, {})
            rdata       = messentials.get('rdata', {})
            model_data  = messentials.get('model', {})

            n_fovs_seen += 1

            if metric == 'mod':
                vals = _get_cv_mi(rdata, var, cond)
            else:
                vals = _get_importance(model_data, var, cond)
                if vals is None:
                    n_fovs_no_key += 1

            if vals is None or vals.size == 0:
                continue

            vis_id_raw = messentials.get('visual_area_id', None)
            if vis_id_raw is not None:
                regions = np.asarray(vis_id_raw, dtype=int)
            else:
                transform = np.asarray(transform)
                if transform.ndim < 2 or transform.shape[1] < 4:
                    continue
                regions = _cell_regions(transform, labeled_array, transform.shape[0])

            n_cells  = min(regions.size, vals.size)
            regions  = regions[:n_cells]
            vals_use = vals[:n_cells]

            for area in SUMMARY_AREAS:
                rid  = REGION_IDS[area]
                mask = regions == rid
                n    = int(np.sum(mask))
                if n < min_cells:
                    continue
                v = vals_use[mask]
                v = v[np.isfinite(v)]
                if len(v) == 0:
                    continue
                if metric == 'mod':
                    pct = float(np.mean(v > mod_thresh) * 100)
                else:
                    pct = float(np.mean(v > imp_thresh) * 100)
                result[area].append(pct)

    if metric == 'imp':
        n_found = sum(len(v) for v in result.values())
        print(f'    [{var} {cond} imp] FOVs seen={n_fovs_seen}, '
              f'no key={n_fovs_no_key}, data points={n_found}')
        if n_fovs_no_key == n_fovs_seen:
            print(f'    WARNING: importance key not found in any FOV. '
                  f'Check that GLM was run and model_data contains an '
                  f'importance key matching pattern: {_imp_key(var, cond)!r}')

    return result


def _fov_violin_figure(pcts_light, pcts_dark, pdf, ylabel):

    PAIR_GAP = 0.5
    AREA_SEP = 1.5

    areas = [a for a in SUMMARY_AREAS if pcts_light.get(a) or pcts_dark.get(a)]
    if not areas:
        print(f'  [violin] no data for any area — skipping.')
        return

    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=200)

    tick_pos, tick_lab = [], []
    for ai, area in enumerate(areas):
        color = COLORS[area]
        x_l   = ai * AREA_SEP
        x_d   = x_l + PAIR_GAP

        _draw_violin(ax, pcts_light.get(area, []), x_l, color, hatch=None,   width=0.42)
        _draw_violin(ax, pcts_dark.get(area,  []), x_d, color, hatch='////', width=0.42)

        tick_pos.append((x_l + x_d) / 2)
        tick_lab.append(area)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)

    light_patch = mpatches.Patch(facecolor='grey', alpha=0.75, label='Light')
    dark_patch  = mpatches.Patch(facecolor='grey', alpha=0.3,
                                 hatch='////', edgecolor='grey', label='Dark')
    ax.legend(handles=[light_patch, dark_patch], frameon=False)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_pct_modulated_violin(pooled_data, labeled_array, pdf,
                              var='theta', mod_thresh=0.33):
    pcts_l = _compute_fov_pcts(pooled_data, labeled_array, var, 'l',
                               metric='mod', mod_thresh=mod_thresh)
    pcts_d = _compute_fov_pcts(pooled_data, labeled_array, var, 'd',
                               metric='mod', mod_thresh=mod_thresh)
    _fov_violin_figure(
        pcts_l, pcts_d, pdf,
        ylabel=f'{var}: % cells modulated (modulation > {mod_thresh}) per FOV',
    )


def plot_pct_important_violin(pooled_data, labeled_array, pdf,
                              var='theta', imp_thresh=0.4):
    pcts_l = _compute_fov_pcts(pooled_data, labeled_array, var, 'l',
                               metric='imp', imp_thresh=imp_thresh)
    pcts_d = _compute_fov_pcts(pooled_data, labeled_array, var, 'd',
                               metric='imp', imp_thresh=imp_thresh)
    _fov_violin_figure(
        pcts_l, pcts_d, pdf,
        ylabel=f'{var}: % cells with significant importance (imp > {imp_thresh}) per FOV',
    )


def plot_cells_randomized_jet_all_animals(pooled_data, savedir):

    all_global_xy  = []
    all_region_ids = []

    for animal_key in pooled_data:
        if not isinstance(pooled_data[animal_key], dict):
            continue
        adat = pooled_data[animal_key]
        messentials = adat.get('messentials', {})
        if not isinstance(messentials, dict) or not messentials:
            continue

        transform_data = adat.get('transform', {})

        for pos_key in list(messentials.keys()):
            if not pos_key.startswith('pos'):
                continue
            pos_dat = messentials[pos_key]
            if not isinstance(pos_dat, dict):
                continue

            vfs_pos = pos_dat.get('vfs_cell_pos', None)
            if vfs_pos is not None:
                xy = np.asarray(vfs_pos, dtype=float)
            else:
                arr = transform_data.get(pos_key, None) if isinstance(transform_data, dict) else None
                if arr is None or not isinstance(arr, np.ndarray) or arr.ndim < 2 or arr.shape[1] < 4:
                    continue
                xy = arr[:, 2:4].astype(float)

            if len(xy) == 0:
                continue

            vis_id = pos_dat.get('visual_area_id', None)
            n = len(xy)
            if vis_id is not None:
                vis_id = np.asarray(vis_id, dtype=int)
                n = min(n, len(vis_id))
                all_global_xy.append(xy[:n])
                all_region_ids.append(vis_id[:n])
            else:
                all_global_xy.append(xy)
                all_region_ids.append(np.zeros(n, dtype=int))

    if not all_global_xy:
        print("No cell positions found for all_animals plot.")
        return

    all_global_xy  = np.vstack(all_global_xy)
    all_region_ids = np.concatenate(all_region_ids)
    n_cells = len(all_global_xy)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    plot_cells_randomized_jet_dmm056_generic(all_global_xy, pooled_data, savedir,
                                             title=f'All Animals — {n_cells} cells',
                                             fig=fig, ax=ax, region_ids=all_region_ids)


def plot_cells_randomized_jet_dmm056_generic(all_global_xy, data, savedir, title='',
                                              fig=None, ax=None, region_ids=None):

    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=200)

    contour_keys = [k for k in data.keys() if k.startswith('ref_contour_')]
    if not contour_keys:
        import json
        _json_path = os.path.join(os.path.dirname(__file__), 'vfs_contours.json')
        if os.path.isfile(_json_path):
            with open(_json_path) as _f:
                _json_contours = json.load(_f)
            for _area, _coords in _json_contours.items():
                if _coords and len(_coords) >= 3:
                    data = dict(data)
                    data[f'ref_contour_{_area}'] = np.array(_coords)
                    contour_keys.append(f'ref_contour_{_area}')

    for k in sorted(contour_keys):
        area_name = k.replace('ref_contour_', '')
        pts = data[k]
        if pts is None or len(pts) < 3:
            continue
        ax.plot(pts[:, 0], pts[:, 1], 'k-', lw=1.5)
        cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        ax.text(cx, cy, area_name,
                color=COLORS.get(area_name, 'k'), fontsize=9,
                ha='center', va='center', fontweight='bold')

    legend_handles = []
    if region_ids is not None:
        region_ids = np.asarray(region_ids, dtype=int)
        known_ids  = set(REGION_IDS.values())
        for area in REGION_ORDER:
            rid  = REGION_IDS[area]
            mask = region_ids == rid
            if not np.any(mask):
                continue
            color = COLORS[area]
            ax.scatter(all_global_xy[mask, 0], all_global_xy[mask, 1],
                       c=color, s=2, alpha=0.7, linewidths=0)
            legend_handles.append(mpatches.Patch(color=color, label=area))
        unassigned = ~np.isin(region_ids, list(known_ids))
        if np.any(unassigned):
            ax.scatter(all_global_xy[unassigned, 0], all_global_xy[unassigned, 1],
                       c='#AAAAAA', s=1, alpha=0.3, linewidths=0)
    else:
        ax.scatter(all_global_xy[:, 0], all_global_xy[:, 1],
                   c='#AAAAAA', s=2, alpha=0.5, linewidths=0)

    if legend_handles:
        ax.legend(handles=legend_handles, frameon=False, fontsize=7,
                  loc='lower right', markerscale=3, handlelength=1)

    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=10)
    fig.tight_layout()

    fig.savefig(os.path.join(savedir, 'cells_by_area.png'), dpi=300)
    fig.savefig(os.path.join(savedir, 'cells_by_area.svg'), dpi=300)
    plt.close(fig)


def make_topography_plots_2(topo_h5_path, pooled_h5_path, out_dir):

    print(f'Loading topography results: {topo_h5_path}')
    topo_data = read_h5(topo_h5_path)

    print(f'Loading pooled data: {pooled_h5_path}')
    pooled_data = read_h5(pooled_h5_path)

    labeled_array = topo_data.get('labeled_array', None)
    if labeled_array is not None:
        labeled_array = np.asarray(labeled_array, dtype=int)
    else:
        raise RuntimeError('labeled_array not found in topography results HDF5.')

    stem = os.path.splitext(os.path.basename(topo_h5_path))[0]
    with FigureSaver(out_dir, stem=stem) as pdf:

        for var in ['theta', 'dTheta', 'pitch', 'dPitch']:
            print(f'  Top-8 tuning curves: {var} ...')
            plot_top8_tuning_curves(topo_data, var, 'l', pdf)

        print('  Modulation CDFs (theta / phi) ...')
        plot_modulation_cdfs(topo_data, pdf)

        for var in ['theta', 'phi']:
            print(f'  % modulated per FOV: {var} ...')
            plot_pct_modulated_violin(pooled_data, labeled_array, pdf, var=var)

        for var in ['theta', 'phi']:
            print(f'  % important per FOV: {var} ...')
            plot_pct_important_violin(pooled_data, labeled_array, pdf, var=var)

    print('  dTheta kurtosis pages (light) ...')
    plot_dtheta_tuning_kurtosis_pages(topo_data, 'l', out_dir, stem)
    print('  dTheta kurtosis pages (dark) ...')
    plot_dtheta_tuning_kurtosis_pages(topo_data, 'd', out_dir, stem)

    plot_cells_randomized_jet_all_animals(pooled_data, out_dir)

    print(f'Saved {pdf._n} figures (PNG + SVG) -> {out_dir}')

    

def main():

    topo_h5   = '/home/dylan/Fast2/topography_analysis_results_260331a.h5'
    pooled_h5 = '/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites/pooled_260331a.h5'
    out_dir   = '/home/dylan/Fast2/topography_plots_2_260408a'
    make_topography_plots_2(topo_h5, pooled_h5, out_dir)


if __name__ == '__main__':

    main()
