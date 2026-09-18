# -*- coding: utf-8 -*-
"""
visualizations/encoding_relative_to_v1.py

Distribution of the encoding-model ablation index for each behavioral input, per visual area.

Per cell, the ablation index for an input is ffNLE's shuffle-corrected
importance, computed on held-out frames:
(signal_full - signal_shuffled) / |signal_full|, where each signal is R^2
minus the R^2 against time-permuted activity.  That signal equals
2 cov(y, y_hat) / var(y), so the index is the fraction of the prediction's
covariance with the cell that is lost when the input is shuffled.  ffNLE clips
it at 0 only; values above 1 (the shuffled prediction anti-correlates with the
cell, or signal_full is near zero) are clipped to 1 here, matching
calc_ablation_index.  Unlike an R^2
drop, it is not inflated by the extra prediction variance a shuffled input
causes, and it is relative to each cell's own fit.  Indices are not additive
across inputs: correlated inputs each claim shared signal.

Each panel shows one violin per area (cut at the data range), every cell as a
jittered point, the area median as a black bar, and one diamond per recording
(median of that recording's cells, recordings with >= MIN_CELLS_PER_REC cells)
so that areas drawn from few recordings are visible.  The fraction of cells
with index > 0 is printed above each violin, since clipping piles cells at 0.

One figure per condition (light: trained and tested on light frames; dark:
trained and tested on dark frames) and cell set (all cells, or cells with
condition full-model R^2 above R2_THRESHOLD), each a grid with one panel per
behavioral input.

Usage
-----
    python visualizations/encoding_relative_to_v1.py --out_dir /path/to/figs

Functions
---------
load_ablation
    Per-cell ablation index for every input, area and recording, for one condition.
area_stats
    Median, fraction above zero and per-recording medians per area and input.
make_ablation_figure
    Grid of panels, one per input: violin, cells and recording medians per area.
main
    Command-line entry: all four figures.


DMM, September 2026
"""

if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

import argparse
import glob
import os

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
mpl.rcParams['axes.spines.top']   = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['font.size']    = 8

from summarize_all_tuning import (
    DEFAULT_POOLED_GLM, DEFAULT_BASE, R2_THRESHOLD, COLORS,
    _IMP_VAR_ORDER, _IMP_ID_TO_NAME, _IMP_REGION_ORDER, _ALL_VAR_LABELS,
)
from summarize_all_tuning_with_maps import resolve_data_path

# Condition name to the model-key prefix ffNLE writes for it.
CONDITIONS = {
    'light': 'full_trainLight_testLight',
    'dark':  'full_trainDark_testDark',
}
# Areas shown; PM is excluded from these figures.
AREAS = [a for a in _IMP_REGION_ORDER if a != 'PM']
FFNLE_FILENAME = 'ffNLE_outputs_v01.h5'
MIN_CELLS_PER_REC = 5
# Figure layout: positions on the top row, their velocities directly below.
PANEL_ROWS = [
    ['theta', 'phi', 'pitch', 'roll', 'yaw'],
    ['dTheta', 'dPhi', 'gyro_y', 'gyro_x', 'gyro_z'],
]
# Y-axis top for each row of PANEL_ROWS; cells above it fall outside the panel.
ROW_YMAX = [1.0, 0.4]
DEFAULT_OUT_DIR = '.'


def _find_model_arrays(pdat, animal, pos, base_dir, r2_key, var_keys, n_cells):
    """ R^2 and per-input model arrays for one position, from the recording whose cell count matches.

    A position can have several recordings, and the pooled 'model' group holds
    whichever was merged last, which need not be the one visual_area_id came
    from.  The ffNLE_outputs file on disk with this condition's keys and
    exactly n_cells cells is used; the pooled model is the fallback only if
    its length also matches.  Never truncates.

    Returns
    -------
    r2 : np.ndarray or None
    arrays : list of np.ndarray or None
    reason : str
        Source used, or why nothing usable was found.
    """

    lengths = []
    if base_dir is not None:
        pattern = os.path.join(base_dir, '*_recordings', '*_{}_{}'.format(animal, pos),
                               'fm*', FFNLE_FILENAME)
        matches = []
        for path in sorted(glob.glob(pattern)):
            with h5py.File(path, 'r') as f:
                if r2_key not in f or not all(k in f for k in var_keys):
                    continue
                n = f[r2_key].shape[0]
                lengths.append(n)
                if n == n_cells:
                    matches.append((path, np.asarray(f[r2_key], dtype=float),
                                    [np.asarray(f[k], dtype=float) for k in var_keys]))
        if len(matches) == 1:
            path, r2, imps = matches[0]
            return r2, imps, os.path.relpath(path, base_dir)
        if len(matches) > 1:
            return None, None, 'ambiguous: {} recordings with {} cells'.format(len(matches), n_cells)

    if 'model' in pdat and r2_key in pdat['model'] and all(k in pdat['model'] for k in var_keys):
        model = pdat['model']
        n = model[r2_key].shape[0]
        if n == n_cells:
            return (np.asarray(model[r2_key], dtype=float),
                    [np.asarray(model[k], dtype=float) for k in var_keys], 'pooled model')
        lengths.append(n)

    if lengths:
        return None, None, 'cell count mismatch: model {} vs area ids {}'.format(
            sorted(set(lengths)), n_cells)
    return None, None, 'no {} model'.format(r2_key[:-3])


def load_ablation(pooled_glm_path, cond_prefix, base_dir=None):
    """ Per-cell ablation index for every input, area and recording, for one condition.

    Positions without this condition's model (e.g. no dark frames, or ffNLE
    never run) are skipped and listed, with the number of cells lost per
    area.

    Parameters
    ----------
    pooled_glm_path : str
        Pooled h5 with per-position 'visual_area_id' (and 'model' fallback).
    cond_prefix : str
        e.g. 'full_trainLight_testLight'.
    base_dir : str or None
        Data root holding *_recordings/<date>_DMM_<animal>_<pos>/fm*/ffNLE_outputs_v01.h5.

    Returns
    -------
    cells : dict
        'ablation' (n_cells, n_vars), 'full_r2' (n_cells,), 'area' (n_cells,)
        area names, 'rec' (n_cells,) recording index; only cells in AREAS.
    """

    r2_key = '{}_r2'.format(cond_prefix)
    abl_keys = ['{}_ablation_index_{}'.format(cond_prefix, v) for v in _IMP_VAR_ORDER]
    ablation, full_r2, area, rec = [], [], [], []
    dropped = []
    n_rec = 0
    with h5py.File(pooled_glm_path, 'r') as f:
        for animal in sorted(f.keys()):
            adat = f[animal]
            if not isinstance(adat, h5py.Group) or 'messentials' not in adat:
                continue
            for pos in sorted(adat['messentials'].keys()):
                pdat = adat['messentials'][pos]
                if not isinstance(pdat, h5py.Group) or 'visual_area_id' not in pdat:
                    continue
                aid = np.atleast_1d(np.asarray(pdat['visual_area_id'], dtype=int))
                names = np.array([_IMP_ID_TO_NAME.get(int(a), '') for a in aid])
                keep = np.isin(names, AREAS)
                if not keep.any():
                    continue

                r2, abls, source = _find_model_arrays(pdat, animal, pos, base_dir,
                                                      r2_key, abl_keys, len(aid))
                counts = {a: int(np.sum(names == a)) for a in AREAS if np.any(names == a)}
                if r2 is None:
                    dropped.append((animal, pos, source, counts))
                    continue
                print('    {} {}: {} cells from {}'.format(animal, pos, int(keep.sum()), source))

                ablation.append(np.clip(np.column_stack(abls)[keep], 0.0, 1.0))
                full_r2.append(r2[keep])
                area.append(names[keep])
                rec.append(np.full(int(keep.sum()), n_rec))
                n_rec += 1

    if dropped:
        print('  Positions dropped ({}):'.format(cond_prefix))
        lost = dict.fromkeys(AREAS, 0)
        for animal, pos, reason, counts in dropped:
            print('    {} {}: {}  {}'.format(animal, pos, reason, counts))
            for a, c in counts.items():
                lost[a] += c
        print('  Cells lost per area: {}'.format(lost))

    if not ablation:
        return None
    return {
        'ablation': np.vstack(ablation),
        'full_r2': np.concatenate(full_r2),
        'area': np.concatenate(area),
        'rec': np.concatenate(rec),
    }


def area_stats(cells):
    """ Median, fraction above zero and per-recording medians per area and input.

    Parameters
    ----------
    cells : dict
        Output of load_ablation, already filtered to the cell set wanted.

    Returns
    -------
    out : dict
        Per area: 'values' (n_cells, n_vars), 'median', 'frac_pos' (n_vars,),
        'rec_medians' (n_recs_kept, n_vars) for recordings with at least
        MIN_CELLS_PER_REC cells, 'n_cells', 'n_recs'.
    """

    vals, area, rec = cells['ablation'], cells['area'], cells['rec']
    out = {}
    for name in AREAS:
        sel = area == name
        if not sel.any():
            continue
        recs = np.unique(rec[sel])
        rec_meds = [np.nanmedian(vals[sel & (rec == r)], axis=0) for r in recs
                    if np.sum(sel & (rec == r)) >= MIN_CELLS_PER_REC]
        out[name] = {
            'values': vals[sel],
            'median': np.nanmedian(vals[sel], axis=0),
            'frac_pos': np.mean(vals[sel] > 0, axis=0),
            'rec_medians': (np.vstack(rec_meds) if rec_meds
                            else np.empty((0, vals.shape[1]))),
            'n_cells': int(sel.sum()),
            'n_recs': int(recs.size),
        }
    return out


def make_ablation_figure(stats, out_path, title, seed=0):
    """ Grid of panels, one per input: violin, cells and recording medians per area.

    Parameters
    ----------
    stats : dict
        Output of area_stats.
    out_path : str
        SVG path; a PNG is written beside it.
    title : str
        Figure title.
    seed : int
        Seed for the point jitter.
    """

    rng = np.random.default_rng(seed)
    areas = [a for a in AREAS if a in stats]
    n_rows, n_cols = len(PANEL_ROWS), max(len(r) for r in PANEL_ROWS)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 3.0 * n_rows),
                            sharey='row', squeeze=False)
    # xticklabels = ['{}\nn={}, {} rec'.format(a, stats[a]['n_cells'], stats[a]['n_recs'])
    #                for a in areas]
    xticklabels = ['{}'.format(a) for a in areas]
    panels = [(ri, ci, var) for ri, row in enumerate(PANEL_ROWS) for ci, var in enumerate(row)]
    for ri, ci, var in panels:
        vi = _IMP_VAR_ORDER.index(var)
        ax = axs[ri, ci]
        for xi, a in enumerate(areas):
            s = stats[a]
            col = COLORS.get(a, 'k')
            v = s['values'][:, vi]
            v = v[np.isfinite(v)]
            # gaussian_kde needs spread; a flat or tiny sample gets points only.
            if v.size >= 3 and np.ptp(v) > 0:
                parts = ax.violinplot([v], positions=[xi], widths=0.8,
                                      showextrema=False)
                for body in parts['bodies']:
                    body.set_facecolor(col)
                    body.set_edgecolor('none')
                    body.set_alpha(0.35)
            jit = rng.uniform(-0.18, 0.18, size=v.size)
            ax.scatter(xi + jit, v, s=2, color=col, alpha=0.5, linewidths=0, zorder=2)
            ax.hlines(s['median'][vi], xi - 0.28, xi + 0.28, color='k', lw=1.8, zorder=4)
            # if s['rec_medians'].shape[0]:
            #     ax.scatter(np.full(s['rec_medians'].shape[0], xi + 0.34),
            #                s['rec_medians'][:, vi], marker='D', s=10, color='w',
            #                edgecolors='k', linewidths=0.6, zorder=5)
            # ax.text(xi, 1.06, '{:.0f}%'.format(100 * s['frac_pos'][vi]),
            #         ha='center', va='bottom', fontsize=6)
        ax.set_title(_ALL_VAR_LABELS.get(var, var), fontsize=10, pad=10)
        ax.set_xticks(range(len(areas)))
        ax.set_xticklabels(xticklabels, fontsize=6)
        ax.set_xlim(-0.6, len(areas) - 0.4)
        ymax = ROW_YMAX[ri]
        ax.set_ylim(-0.03 * ymax, ymax)
        ax.set_yticks(np.linspace(0.0, ymax, 5))
        if ci == 0:
            ax.set_ylabel('ablation index', fontsize=9)
    for ri, row in enumerate(PANEL_ROWS):
        for ci in range(len(row), n_cols):
            axs[ri, ci].set_visible(False)

    # fig.suptitle(title + '\n(bar = median; diamonds = recording medians; '
    #              '% = cells with index > 0)', fontsize=10)
    fig.tight_layout()
    png_path = os.path.splitext(out_path)[0] + '.png'
    fig.savefig(out_path, format='svg', bbox_inches='tight', dpi=300)
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('Saved: {}'.format(out_path))
    print('Saved: {}'.format(png_path))


def _print_table(stats, label):
    """ Print median ablation index and fraction above zero per area and input. """

    print('\n' + '=' * 78)
    print('ABLATION INDEX -- {}'.format(label))
    print('=' * 78)
    for a in [a for a in AREAS if a in stats]:
        s = stats[a]
        print('  {} ({} cells, {} recordings)'.format(a, s['n_cells'], s['n_recs']))
        for vi, var in enumerate(_IMP_VAR_ORDER):
            print('    {:<7} median {:.3f}   >0 {:5.1f}%'.format(
                _ALL_VAR_LABELS.get(var, var), s['median'][vi], 100 * s['frac_pos'][vi]))


def main():
    """ Command-line entry: all four figures. """

    p = argparse.ArgumentParser(
        description='Ablation index per behavioral input, per visual area.')
    p.add_argument('--pooled_glm', default=DEFAULT_POOLED_GLM,
                   help='Pooled h5 with ffNLE model results per position.')
    p.add_argument('--base_dir', default=DEFAULT_BASE,
                   help='Data root searched for per-recording {} files.'.format(FFNLE_FILENAME))
    p.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    p.add_argument('--r2_thresh', type=float, default=R2_THRESHOLD,
                   help='Condition full-model R^2 cutoff for the reliable-cell '
                        'figures (default: %(default)s).')
    args = p.parse_args()

    pooled = resolve_data_path(args.pooled_glm)
    base_dir = resolve_data_path(args.base_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    for cond, prefix in CONDITIONS.items():
        print('\nLoading {} models from {}...'.format(cond, pooled))
        cells = load_ablation(pooled, prefix, base_dir=base_dir)
        if cells is None:
            print('  No {} models found -- skipping.'.format(cond))
            continue
        print('  {} cells in {} recordings.'.format(len(cells['area']),
                                                   np.unique(cells['rec']).size))

        reliable = cells['full_r2'] > args.r2_thresh
        cell_sets = (
            ('all', 'all cells', np.ones(len(reliable), dtype=bool)),
            ('reliable', 'cells with full R$^2$ > {:g}'.format(args.r2_thresh), reliable),
        )
        for tag, desc, sel in cell_sets:
            sub = {k: v[sel] for k, v in cells.items()}
            if not sub['area'].size:
                print('  No {} cells for {} -- skipping.'.format(tag, cond))
                continue
            stats = area_stats(sub)
            _print_table(stats, '{}, {}'.format(cond, desc.replace('$^2$', '^2')))
            title = 'Ablation index per input -- {} frames, {}'.format(cond, desc)
            out = os.path.join(args.out_dir,
                               'encoding_relative_to_v1_{}_{}.svg'.format(cond, tag))
            make_ablation_figure(stats, out, title)


if __name__ == '__main__':
    main()
