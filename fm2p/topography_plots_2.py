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
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from .utils.files import read_h5

import matplotlib as mpl
mpl.rcParams['axes.spines.top']   = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype']      = 42
mpl.rcParams['ps.fonttype']       = 42
mpl.rcParams['font.size']         = 7


REGION_ORDER = ['V1', 'RL', 'AM', 'PM', 'A', 'AL', 'LM', 'P']
REGION_IDS   = {'V1': 5, 'RL': 2, 'AM': 3, 'PM': 4, 'A': 10, 'AL': 7, 'LM': 8, 'P': 9}

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
        _imp_key(var, cond),                                                     # primary
        f'{var}_train{cond_str}_test{cond_str}_importance_{var}',                # var not remapped
        f'full_train{cond_str}_test{cond_str}_importance_{base}',                # old 'full_' prefix
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
    """2×4 grid of tuning curves for 8 cells chosen by modulation index.

    For all variables except dTheta: the 8 highest-MI cells are shown in order.
    For dTheta: 8 cells are drawn randomly (seeded) from the top-10% by MI,
    so the panel shows representative cells rather than the extreme outliers.
    """
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

    # Choose which 8 rows to display
    n_total = len(mods)
    if var == 'dTheta':
        # random sample from top 10%
        top10_n = max(8, int(np.ceil(n_total * 0.10)))
        top10_n = min(top10_n, n_total)
        pool    = np.arange(top10_n)          # sorted_tuning_curves is already sorted desc
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
            ax.set_title(f'modulation = {mods[idx]:.2f}', fontsize=7.5)
            ax.tick_params(labelsize=7)
            ax.set_xlabel(xlabel, fontsize=7.5)
            ax.set_ylabel('inferred spike rate', fontsize=7.5)
        else:
            ax.axis('off')

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

def _ecdf(values):
    """Return (x, y) for an empirical CDF of *values*, NaNs excluded."""
    v = np.sort(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.array([]), np.array([])
    return v, np.arange(1, len(v) + 1) / len(v)


def plot_modulation_cdfs(topo_data, pdf):
    """ECDF of per-cell modulation by visual area — one subplot per variable.

    ECDFs are far more revealing than violins for right-skewed modulation
    distributions where most cells cluster near zero.  Light = solid line;
    dark = dashed line of the same colour.  Reference verticals at 0.33 and 0.5.
    """
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
        ax.set_xlabel('modulation', fontsize=8)
        ax.set_ylabel('cumulative fraction', fontsize=8)
        ax.set_title(var, color=COLORS.get(var, 'k'), fontsize=9)
        ax.legend(fontsize=5, frameon=False, ncol=2)

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

            # Use pre-computed visual_area_id if available (more reliable than
            # recomputing from transform coords), fall back to _cell_regions
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
    ax.set_xticklabels(tick_lab, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_ylim(bottom=0)

    light_patch = mpatches.Patch(facecolor='grey', alpha=0.75, label='Light')
    dark_patch  = mpatches.Patch(facecolor='grey', alpha=0.3,
                                 hatch='////', edgecolor='grey', label='Dark')
    ax.legend(handles=[light_patch, dark_patch], fontsize=6, frameon=False)

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
                              var='theta', imp_thresh=0.25):
    pcts_l = _compute_fov_pcts(pooled_data, labeled_array, var, 'l',
                               metric='imp', imp_thresh=imp_thresh)
    pcts_d = _compute_fov_pcts(pooled_data, labeled_array, var, 'd',
                               metric='imp', imp_thresh=imp_thresh)
    _fov_violin_figure(
        pcts_l, pcts_d, pdf,
        ylabel=f'{var}: % cells with significant importance (imp > {imp_thresh}) per FOV',
    )


def make_topography_plots_2(topo_h5_path, pooled_h5_path, out_pdf_path):

    print(f'Loading topography results: {topo_h5_path}')
    topo_data = read_h5(topo_h5_path)

    print(f'Loading pooled data: {pooled_h5_path}')
    pooled_data = read_h5(pooled_h5_path)

    labeled_array = topo_data.get('labeled_array', None)
    if labeled_array is not None:
        labeled_array = np.asarray(labeled_array, dtype=int)
    else:
        raise RuntimeError('labeled_array not found in topography results HDF5.')

    with PdfPages(out_pdf_path) as pdf:

 
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

    print(f'Saved -> {out_pdf_path}')


def main():

    topo_h5 = '/home/dylan/Fast2/topography_analysis_results_260331a.h5'
    pooled_h5 = '/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites/pooled_260331a.h5'
    out_pdf = '/home/dylan/Fast2/topography_plots_2_260331a.pdf'
    make_topography_plots_2(topo_h5, pooled_h5, out_pdf)


if __name__ == '__main__':

    main()
