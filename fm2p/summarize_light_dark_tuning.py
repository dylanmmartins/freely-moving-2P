## summarize light/dark tuning curves and importance metrics from model

if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon as MplPolygon

from .utils.files import read_h5
from .utils.ffNLE_figs import (
    load_contours, AREA_IDS, AREA_COLORS,
    _EARTH_HEX, _VAR_ORDER, _VAR_NICE, _GOODRED,
)

# ── Variable groupings ────────────────────────────────────────────────────────
_POS_VARS   = ['theta',  'phi',    'pitch',  'roll']
_POS_NICE   = ['θ',      'φ',      'pitch',  'roll']
_POS_COLORS = [_EARTH_HEX[0], _EARTH_HEX[2], _EARTH_HEX[4], _EARTH_HEX[6]]

_VEL_VARS   = ['dTheta', 'dPhi',   'gyro_y', 'gyro_x', 'gyro_z']
_VEL_NICE   = ['dθ',     'dφ',     'dPitch', 'dRoll',  'dYaw']
_VEL_COLORS = [_EARTH_HEX[1], _EARTH_HEX[3], _EARTH_HEX[5], _EARTH_HEX[7], _EARTH_HEX[9]]

_TRAIN_TEST_LIGHT = 'trainLight_testLight'
_TRAIN_TEST_DARK  = 'trainDark_testDark'

_ALLOWED_ANIMALS  = {'DMM056', 'DMM061'}
_MIN_FRAMES       = 1000

# Importance prefix lists: (prefix_template, label)
# For the full model the prefix is either 'full_trainLight_testLight_' (newer)
# or 'full_' (older runs without condition split); both are tried.
_SUBSET_LABELS = ['full', 'eyes only', 'head only', 'pos. only', 'vel. only']
_SUBSET_LIGHT_PREFIXES = [
    'full_trainLight_testLight_',
    'eyes_only_trainLight_testLight_',
    'head_only_trainLight_testLight_',
    'position_only_trainLight_testLight_',
    'velocity_only_trainLight_testLight_',
]
_SUBSET_DARK_PREFIXES = [
    'full_trainDark_testDark_',
    'eyes_only_trainDark_testDark_',
    'head_only_trainDark_testDark_',
    'position_only_trainDark_testDark_',
    'velocity_only_trainDark_testDark_',
]



def _revcorr_cell(rdata, var, cond_idx, ci):
    """Return (bins, tuning, err) for one cell from revcorr 1D tuning data.

    cond_idx : 0 = light, 1 = dark
    Returns (None, None, None) if data unavailable.
    """
    bk = f'{var}_1dbins'
    tk = f'{var}_1dtuning'
    ek = f'{var}_1derr'
    if bk not in rdata or tk not in rdata:
        return None, None, None
    bins   = np.asarray(rdata[bk], dtype=float)
    tuning = np.asarray(rdata[tk], dtype=float)
    err    = np.asarray(rdata[ek], dtype=float) if ek in rdata else None
    # tuning shape: (n_cells, n_bins, 2)  or  (n_bins, 2)  or  (n_bins,)
    if tuning.ndim == 3:
        if ci >= tuning.shape[0]:
            return None, None, None
        t = tuning[ci, :, cond_idx]
        e = err[ci, :, cond_idx] if err is not None and err.ndim == 3 else None
    elif tuning.ndim == 2:
        t = tuning[:, cond_idx]
        e = err[:, cond_idx] if err is not None and err.ndim == 2 else None
    else:
        t = tuning
        e = None
    return bins, t, e


def _n_frames(model):
    """Estimate total frame count from model dict."""
    # Full-model positions store train/test indices separately
    if 'full_train_indices' in model and 'full_test_indices' in model:
        return (len(np.atleast_1d(model['full_train_indices'])) +
                len(np.atleast_1d(model['full_test_indices'])))
    # Per-variable (PDP) positions store eval indices per variable
    for var in _VAR_ORDER:
        k = f'{var}_{_TRAIN_TEST_LIGHT}_eval_indices'
        if k in model:
            return len(np.atleast_1d(model[k]))
    return 0


def _imp_cell(model, prefix, ci):
    """Return (n_vars,) importance for one cell.
    For the full-model prefix, falls back to the plain 'full_importance_{var}'
    key used by older runs that did not split by condition."""
    _is_full = prefix.startswith('full_')
    out = np.full(len(_VAR_ORDER), np.nan)
    for vi, var in enumerate(_VAR_ORDER):
        k = f'{prefix}importance_{var}'
        if k not in model and _is_full:
            k = f'full_importance_{var}'   # older key without cond split
        if k in model:
            v = np.atleast_1d(np.asarray(model[k], dtype=float))
            if ci < len(v):
                out[vi] = float(v[ci])
    return out


# ── Per-cell figure ───────────────────────────────────────────────────────────

def _make_cell_fig(cell_rec, all_vfs_pos, contours):
    """
    Build a 5-row × 5-col figure for a single cell.

    Parameters
    ----------
    cell_rec : dict with keys:
        animal, pos, ci, r2, corr, area_id, vfs_pos, model, rdata
    all_vfs_pos : (N_all, 2) float array – all cell positions in this FOV (gray dots)
    contours : dict from load_contours()
    """
    animal  = cell_rec['animal']
    pos     = cell_rec['pos']
    ci      = cell_rec['ci']
    r2      = cell_rec['r2']
    corr    = cell_rec['corr']
    area_id = cell_rec['area_id']
    vfs_pos = cell_rec['vfs_pos']   # (2,)
    model   = cell_rec['model']
    rdata   = cell_rec.get('rdata', {})

    cell_aname = next((n for n, i in AREA_IDS.items() if i == area_id), None)

    fig = plt.figure(figsize=(15, 13), dpi=150)
    gs  = GridSpec(5, 5, figure=fig, hspace=0.58, wspace=0.40)

    # ── Pre-compute tuning data for per-row shared y-axes ────────────────────
    # cond_idx: 1 = light, 0 = dark  (per calc_1d_tuning convention)
    tuning_data = {}  # var -> (bins, tuning, err)
    pos_lo, pos_hi = np.inf, -np.inf
    vel_lo, vel_hi = np.inf, -np.inf
    for var in _POS_VARS + _VEL_VARS:
        bins, tuning, err = _revcorr_cell(rdata, var, 1, ci)  # 1 = light
        tuning_data[var] = (bins, tuning, err)
        if bins is not None:
            lo = float(np.nanmin(tuning - err if err is not None else tuning))
            hi = float(np.nanmax(tuning + err if err is not None else tuning))
            if var in _POS_VARS:
                pos_lo = min(pos_lo, lo); pos_hi = max(pos_hi, hi)
            else:
                vel_lo = min(vel_lo, lo); vel_hi = max(vel_hi, hi)
    def _ylim(lo, hi, fallback=(-0.01, 0.1)):
        if not np.isfinite(lo):
            return fallback
        pad = max((hi - lo) * 0.12, 0.005)
        return (lo - pad, hi + pad)
    pos_ylim = _ylim(pos_lo, pos_hi)
    vel_ylim = _ylim(vel_lo, vel_hi)

    # ── Pre-compute importance data for shared y-axis ─────────────────────────
    all_imp_data = {}  # (row_cond, col_label) -> imp array
    imp_lo, imp_hi = np.inf, -np.inf
    for cond_label, subset_prefixes in [('Light', _SUBSET_LIGHT_PREFIXES),
                                        ('Dark',  _SUBSET_DARK_PREFIXES)]:
        for prefix, label in zip(subset_prefixes, _SUBSET_LABELS):
            imp = _imp_cell(model, prefix, ci)
            all_imp_data[(cond_label, label)] = imp
            valid = imp[~np.isnan(imp)]
            if len(valid):
                imp_lo = min(imp_lo, float(valid.min()))
                imp_hi = max(imp_hi, float(valid.max()))
    if not np.isfinite(imp_hi):
        imp_hi = 0.1
    imp_pad = max(imp_hi * 0.10, 0.005)
    imp_ylim = (0.0, imp_hi + imp_pad)

    # ── Row 0: position revcorr tuning (4 cols, light) + reliability bar ──────
    for col, (var, nice, color) in enumerate(zip(_POS_VARS, _POS_NICE, _POS_COLORS)):
        ax = fig.add_subplot(gs[0, col])
        bins, tuning, err = tuning_data[var]
        if bins is not None:
            ax.plot(bins, tuning, color=color, lw=1.5)
            if err is not None:
                ax.fill_between(bins, tuning - err, tuning + err,
                                color=color, alpha=0.20)
        ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
        ax.set_ylim(pos_ylim)
        ax.set_title(nice, fontsize=8)
        ax.set_xlabel(nice, fontsize=7)
        ax.set_ylabel('rate (sp/s)', fontsize=7)
        ax.tick_params(labelsize=6)

    # Reliability bar – cross-validated split-half metric (l_rel)
    ax_mod = fig.add_subplot(gs[0, 4])
    rel_vals = []
    for var in _VAR_ORDER:
        rk = f'{var}_l_rel'
        if rk in rdata:
            arr = np.atleast_1d(np.asarray(rdata[rk], dtype=float))
            rel_vals.append(float(arr[ci]) if ci < len(arr) else 0.0)
        else:
            rel_vals.append(0.0)
    x_bar = np.arange(len(_VAR_ORDER))
    ax_mod.bar(x_bar, rel_vals, color=_EARTH_HEX, alpha=0.85)
    ax_mod.set_xticks(x_bar)
    ax_mod.set_xticklabels(_VAR_NICE, rotation=45, ha='right', fontsize=5)
    ax_mod.set_ylabel('reliability (CV)', fontsize=7)
    ax_mod.set_title('CV reliability (light)', fontsize=8)
    ax_mod.tick_params(labelsize=6)

    # ── Row 1: velocity revcorr tuning (5 cols, light) ───────────────────────
    for col, (var, nice, color) in enumerate(zip(_VEL_VARS, _VEL_NICE, _VEL_COLORS)):
        ax = fig.add_subplot(gs[1, col])
        bins, tuning, err = tuning_data[var]
        if bins is not None:
            ax.plot(bins, tuning, color=color, lw=1.5)
            if err is not None:
                ax.fill_between(bins, tuning - err, tuning + err,
                                color=color, alpha=0.20)
        ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
        ax.set_ylim(vel_ylim)
        ax.set_title(nice, fontsize=8)
        ax.set_xlabel(nice, fontsize=7)
        ax.set_ylabel('rate (sp/s)', fontsize=7)
        ax.tick_params(labelsize=6)

    # ── Rows 2 & 3: importance bars ───────────────────────────────────────────
    for row, (cond_label, subset_prefixes) in enumerate([
        ('Light', _SUBSET_LIGHT_PREFIXES),
        ('Dark',  _SUBSET_DARK_PREFIXES),
    ], start=2):
        for col, (prefix, label) in enumerate(zip(subset_prefixes, _SUBSET_LABELS)):
            ax  = fig.add_subplot(gs[row, col])
            imp = all_imp_data[(cond_label, label)]
            x   = np.arange(len(_VAR_ORDER))
            colors_bar = [
                _EARTH_HEX[vi] if not np.isnan(imp[vi]) else '#cccccc'
                for vi in range(len(_VAR_ORDER))
            ]
            ax.bar(x, np.where(np.isnan(imp), 0.0, imp),
                   color=colors_bar, alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(_VAR_NICE, rotation=45, ha='right', fontsize=5)
            ax.set_ylim(imp_ylim)
            ax.set_ylabel('importance', fontsize=6)
            ax.set_title(f'{cond_label}: {label}', fontsize=7)
            ax.tick_params(labelsize=5)

    # ── Row 4: light trace (cols 0-1) | dark trace (cols 2-3) | VFS map (col 4)
    ax_light = fig.add_subplot(gs[4, 0:2])
    ax_dark  = fig.add_subplot(gs[4, 2:4])
    ax_map   = fig.add_subplot(gs[4, 4])

    def _plot_trace(ax, yh_arr, yt_arr, title):
        if yh_arr is not None and yt_arr is not None:
            yh_arr = np.asarray(yh_arr, dtype=float)
            yt_arr = np.asarray(yt_arr, dtype=float)
            yh = yh_arr[:, ci] if yh_arr.ndim == 2 else yh_arr
            yt = yt_arr[:, ci] if yt_arr.ndim == 2 else yt_arr
            lim = min(len(yt), 2000)
            t   = np.arange(lim)
            ax.plot(t, yt[:lim], color='k',       lw=0.5, alpha=0.85, label='y')
            ax.plot(t, yh[:lim], color='tab:red',  lw=0.7, alpha=0.85,
                    label=r'$\hat{y}$')
            ax.legend(fontsize=6, frameon=False)
        ax.set_title(title, fontsize=8, pad=3)
        ax.set_xlabel('frame', fontsize=7)
        ax.set_ylabel('spikes/s', fontsize=7)
        ax.tick_params(labelsize=6)

    # Light trace
    yh_l  = model.get(f'full_{_TRAIN_TEST_LIGHT}_y_hat',
             model.get(f'eyes_only_{_TRAIN_TEST_LIGHT}_y_hat', None))
    yt_l  = model.get(f'full_{_TRAIN_TEST_LIGHT}_y_true',
             model.get(f'eyes_only_{_TRAIN_TEST_LIGHT}_y_true', None))
    _plot_trace(ax_light, yh_l, yt_l, f'Light  R²={r2:.3f}')

    # Dark trace
    yh_d  = model.get(f'full_{_TRAIN_TEST_DARK}_y_hat',
             model.get(f'eyes_only_{_TRAIN_TEST_DARK}_y_hat', None))
    yt_d  = model.get(f'full_{_TRAIN_TEST_DARK}_y_true',
             model.get(f'eyes_only_{_TRAIN_TEST_DARK}_y_true', None))
    r2_d_arr = model.get(f'full_{_TRAIN_TEST_DARK}_r2',
               model.get(f'eyes_only_{_TRAIN_TEST_DARK}_r2', None))
    r2_d = float(np.atleast_1d(r2_d_arr)[ci]) if r2_d_arr is not None else float('nan')
    _plot_trace(ax_dark, yh_d, yt_d, f'Dark  R²={r2_d:.3f}')

    # area map: all FOV cells as gray dots, this cell highlighted
    valid = ~np.any(np.isnan(all_vfs_pos), axis=1)
    if valid.any():
        ax_map.scatter(all_vfs_pos[valid, 0], all_vfs_pos[valid, 1],
                       s=1, color='gray', alpha=0.3)

    for aname, pts in contours.items():
        pts_arr = np.asarray(pts)
        if pts_arr.ndim == 2 and pts_arr.shape[1] == 2:
            closed = np.vstack([pts_arr, pts_arr[:1]])
            ax_map.plot(closed[:, 0], closed[:, 1], color='k', lw=0.8)
            cx, cy = pts_arr.mean(axis=0)
            ax_map.text(cx, cy, aname, ha='center', va='center',
                        fontsize=5, color=AREA_COLORS.get(aname, '#555'))

    if cell_aname and cell_aname in contours:
        pts_arr = np.asarray(contours[cell_aname])
        if pts_arr.ndim == 2 and pts_arr.shape[1] == 2:
            poly = MplPolygon(pts_arr, closed=True,
                              facecolor=AREA_COLORS.get(cell_aname, '#999'),
                              alpha=0.30, edgecolor='none')
            ax_map.add_patch(poly)

    if not np.any(np.isnan(vfs_pos)):
        ax_map.scatter(*vfs_pos, s=50,
                       color=AREA_COLORS.get(cell_aname, _GOODRED),
                       edgecolors='k', linewidths=0.6, zorder=5)

    ax_map.invert_yaxis()
    ax_map.set_title(f'{cell_aname or "?"}', fontsize=8)
    ax_map.set_aspect('equal', adjustable='datalim')
    ax_map.set_xlabel('VFS x', fontsize=6)
    ax_map.set_ylabel('VFS y', fontsize=6)
    ax_map.tick_params(labelsize=5)

    fig.suptitle(
        f'{animal} / {pos} / cell {ci}   ({cell_aname or "unknown"})   '
        f'R²={r2:.3f}',
        fontsize=10
    )
    return fig


# ── Public entry point ────────────────────────────────────────────────────────

def summarize_light_dark(pooled_path, out_pdf=None,
                         area_name='V1', top_n=200):
    """
    One PDF page per cell for the top-N cells (by full-model R²) in one
    visual area.

    Parameters
    ----------
    pooled_path : str   path to pooled data .h5 file
    out_pdf     : str   output PDF (default: <pooled_path>_<area>_top<N>_summary.pdf)
    area_name   : str   visual area to filter to (default 'V1')
    top_n       : int   how many cells to include (default 200)

    Layout (per cell)
    -----------------
    Row 0  : PDP for θ, φ, pitch, roll  |  modulation bar (all vars)
    Row 1  : PDP for dθ, dφ, dPitch, dRoll, dYaw
    Row 2  : Light importance bars (full / eyes / head / pos / vel)
    Row 3  : Dark  importance bars
    Row 4  : Firing-rate trace (first 3000 frames)  |  VFS area map  |  blank
    """
    data     = read_h5(pooled_path)
    contours = load_contours()

    target_aid = AREA_IDS.get(area_name)
    if target_aid is None:
        raise ValueError(f"Unknown area '{area_name}'. Choose from: {list(AREA_IDS)}")

    if out_pdf is None:
        stem    = pooled_path.replace('.h5', '')
        out_pdf = f'{stem}_{area_name}_top{top_n}_summary.pdf'

    # ── Pass 1: collect all cells matching the target area
    # Two position types coexist in the pooled data:
    #   A) full-model positions: have full_r2 / full_corrs but no PDPs
    #   B) per-variable positions: have per-variable PDPs but no full_r2
    # We include both; r2 for ranking comes from the best available source.
    records = []

    def _best_r2(model):
        """Return the best per-cell R² array available in this model dict."""
        if 'full_r2' in model:
            return np.atleast_1d(np.asarray(model['full_r2'], dtype=float))
        # fall back to eyes_only light R²
        k = f'eyes_only_{_TRAIN_TEST_LIGHT}_r2'
        if k in model:
            return np.atleast_1d(np.asarray(model[k], dtype=float))
        # last resort: mean of all per-variable light R²s
        var_r2s = [
            np.atleast_1d(np.asarray(model[f'{v}_{_TRAIN_TEST_LIGHT}_r2'], dtype=float))
            for v in _VAR_ORDER if f'{v}_{_TRAIN_TEST_LIGHT}_r2' in model
        ]
        if var_r2s:
            n = min(len(a) for a in var_r2s)
            return np.nanmean(np.stack([a[:n] for a in var_r2s], axis=0), axis=0)
        return None

    def _best_corr(model, n):
        for k in ('full_corrs',
                  f'eyes_only_{_TRAIN_TEST_LIGHT}_corrs',
                  f'theta_{_TRAIN_TEST_LIGHT}_corrs'):
            if k in model:
                v = np.atleast_1d(np.asarray(model[k], dtype=float))
                out = np.full(n, np.nan)
                out[:min(len(v), n)] = v[:min(len(v), n)]
                return out
        return np.zeros(n)

    for animal in sorted(data.keys()):
        if animal not in _ALLOWED_ANIMALS:
            continue
        adat = data[animal]
        if 'messentials' not in adat:
            continue
        me = adat['messentials']
        for pos in sorted(me.keys()):
            pdat = me[pos]
            if not isinstance(pdat, dict):
                continue
            model = pdat.get('model', {})
            if not model:
                continue

            # Skip recordings with too few frames
            if _n_frames(model) < _MIN_FRAMES:
                continue

            # Skip positions that don't have all subset models
            # (only positions with full/head/position/velocity models are useful)
            if f'full_trainLight_testLight_importance_theta' not in model:
                continue

            r2_arr = _best_r2(model)
            if r2_arr is None:
                continue
            corr_arr = _best_corr(model, len(r2_arr))
            n_cells  = len(r2_arr)
            rdata    = pdat.get('rdata', {})

            area_id_arr = np.zeros(n_cells, dtype=int)
            raw_aid = pdat.get('visual_area_id', None)
            if raw_aid is not None:
                raw_aid = np.atleast_1d(np.asarray(raw_aid, dtype=int))
                m = min(len(raw_aid), n_cells)
                area_id_arr[:m] = raw_aid[:m]

            vfs_pos_arr = np.full((n_cells, 2), np.nan)
            raw_vp = pdat.get('vfs_cell_pos', None)
            if raw_vp is not None:
                raw_vp = np.asarray(raw_vp, dtype=float)
                if raw_vp.ndim == 1:
                    raw_vp = raw_vp.reshape(-1, 2)
                m = min(len(raw_vp), n_cells)
                vfs_pos_arr[:m] = raw_vp[:m]

            for ci in range(n_cells):
                if area_id_arr[ci] != target_aid:
                    continue
                records.append({
                    'animal':      animal,
                    'pos':         pos,
                    'ci':          ci,
                    'r2':          float(r2_arr[ci]),
                    'corr':        float(corr_arr[ci]),
                    'area_id':     int(area_id_arr[ci]),
                    'vfs_pos':     vfs_pos_arr[ci].copy(),
                    'model':       model,
                    'rdata':       rdata,
                    'all_vfs_pos': vfs_pos_arr,
                })

    if not records:
        print(f'No cells found for area {area_name} (id={target_aid}).')
        return

    records.sort(key=lambda r: r['r2'], reverse=True)
    records = records[:top_n]
    print(f'Found {len(records)} {area_name} cells; saving top {len(records)}.')

    with PdfPages(out_pdf) as pdf:
        for rank, rec in enumerate(records, 1):
            fig = _make_cell_fig(rec, rec['all_vfs_pos'], contours)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            print(f'\r  -> {rank}/{len(records)}  '
                  f'{rec["animal"]}/{rec["pos"]} cell {rec["ci"]}  '
                  f'R²={rec["r2"]:.3f}',
                  end='', flush=True)

    print(f'\nSaved: {out_pdf}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Summarize light/dark ffNLE tuning — one page per cell')
    parser.add_argument('--pooled_path', help='Path to pooled data .h5 file',
                        default='/home/dylan/Fast2/topography/pooled_260318a.h5')
    parser.add_argument('--out_pdf', default=None,
                        help='Output PDF path')
    parser.add_argument('--area', default='V1',
                        help='Visual area to summarise (default: V1)')
    parser.add_argument('--top_n', type=int, default=200,
                        help='Number of top-R² cells to include (default: 200)')
    args = parser.parse_args()
    summarize_light_dark(args.pooled_path, out_pdf=args.out_pdf,
                         area_name=args.area, top_n=args.top_n)
