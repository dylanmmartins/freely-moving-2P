"""
Cortical maps of GLM variable importance.

Reads the same pooled GLM dataset as summarize_all_tuning.py and renders the
per-cell ablation index onto the reference VFS map, in the style of the maps in
topography_plots.py / topography_plots_2.py.

One map per PDF page:

    1. per-variable importance (theta, phi, dTheta, ...) in light, then in dark
    2. group importance (eye-only, head-only, position-only, velocity-only),
       light then dark
    3. dark - light ablation index for each variable  (positive = more
       important in the dark)
    4. dark - light ablation index for each group
    5. observed / expected overlap of light-tuned and dark-tuned cells

Every ablation index on these pages is LEAVE-ONE-OUT (see compute_group_importance
and compute_permutation_importance in fm2p/utils/ffNLE.py): the named variable or
group is permuted in the already-trained full model, and the index is the fraction
of that cell's decodable signal the knockout destroys.  It is a measure of
NECESSITY, not of sufficiency -- 'position 0.92' means 'destroying position costs
92% of the signal', NOT 'position alone reconstructs 92%'.  Because the model is
never refit, a redundant group can score near 0 even when it carries the variable
perfectly on its own, and indices across groups need not sum to 1.

DMM August 2026
"""

if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _root = _pl.Path(__file__).resolve().parents[1]
    _sys.path.insert(0, str(_root))
    _sys.path.insert(0, str(_root / 'visualizations'))

import argparse
import glob
import json
import os
import pickle

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from matplotlib.path import Path as MplPath
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.stats import chi2_contingency, fisher_exact

import matplotlib as mpl
mpl.rcParams['axes.spines.top']   = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['font.size']    = 7

from summarize_all_tuning import (
    DEFAULT_POOLED, DEFAULT_POOLED_GLM, DEFAULT_BASE,
    APPLY_R2_THRESHOLD, R2_THRESHOLD, MIN_CELLS_AREA,
    COLORS, _ldi, _IMP_VAR_ORDER, _IMP_ID_TO_NAME, _IMP_REGION_ORDER,
    _GROUP_VAR_ORDER, _GROUP_LABELS, _GROUP_COLORS, _ALL_VAR_LABELS,
)
from summarize_head_tuning import _build_pooled_lookup, _match_to_pooled

from fm2p.utils.helper import interp_short_gaps
from fm2p.utils.time import interpT

import pathlib as _pathlib
_REPO_ROOT = _pathlib.Path(__file__).resolve().parents[1]

DEFAULT_CONTOURS = str(_REPO_ROOT / 'fm2p' / 'utils' / 'vfs_contours.json')
DEFAULT_OUT_DIR  = '.'
DEFAULT_PDF_NAME = 'all_tuning_importance_maps.pdf'

REF_SHAPE_FALLBACK = (400, 400)

AREA_IDS = {'RL': 2, 'AM': 3, 'PM': 4, 'V1': 5, 'AL': 7, 'LM': 8, 'P': 9, 'A': 10}

# Variables that get a map page, in the order used by the importance heatmaps.
MAP_VAR_ORDER = ['theta', 'phi', 'pitch', 'roll',
                 'dTheta', 'dPhi', 'gyro_y', 'gyro_x', 'yaw', 'gyro_z']
MAP_VAR_LABELS = dict(_ALL_VAR_LABELS, yaw='yaw', gyro_z='dYaw')

MIN_CELLS_SMOOTH = 6      # cells needed inside an area before it is smoothed
SMOOTH_SIGMA     = 10     # px, in reference VFS space (400 x 400)
COVERAGE_RADIUS  = 25     # px; smoothed pixels further than this from any cell
                          # are blanked instead of extrapolated
SCATTER_SIZE     = 6

# Ablation indices are LEAVE-ONE-OUT knockouts, so they read as "fraction of
# decodable signal lost when this is permuted out of the trained full model".
# Spelled out on every panel because leave-one-out and leave-one-in support
# opposite sentences: 1.0 = indispensable, 0.0 = fully redundant with what is
# left in (which is NOT the same as carrying no information).
ABL_SENSE_SHORT = 'leave-one-out'
ABL_SENSE_NOTE  = ('leave-one-out: variable permuted in the trained full model; '
                   '1 = all decodable signal lost, 0 = fully redundant with the '
                   'variables left in')

# Ablation indices are bounded in [0, 1], so every importance page shares one
# colour scale and the dark - light pages share one symmetric scale.
IMP_NORM  = mcolors.Normalize(vmin=0.0, vmax=1.0)
DIFF_NORM = mcolors.TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------

def _read(model, key, n):
    """Read a length-n float vector from an h5 model group, else NaN."""
    if key not in model:
        return np.full(n, np.nan)
    v = np.atleast_1d(np.asarray(model[key], dtype=float))
    out = np.full(n, np.nan)
    m = min(len(v), n)
    out[:m] = v[:m]
    return out


def load_importance_maps(pooled_glm_path):
    """Per-cell ablation indices plus reference-VFS positions.

    Mirrors _load_importance_cells() in summarize_all_tuning.py, but reads the
    HDF5 directly (so the multi-GB per-cell traces are never touched) and also
    keeps 'vfs_cell_pos' so each cell can be placed on the map.
    """
    n_var = len(_IMP_VAR_ORDER)
    n_grp = len(_GROUP_VAR_ORDER)

    x, y, area_id, full_r2 = [], [], [], []
    light_imp, dark_imp, light_grp, dark_grp = [], [], [], []
    animals, poss, cis = [], [], []
    ref_shape = None

    with h5py.File(pooled_glm_path, 'r') as f:
        for animal in sorted(f.keys()):
            adat = f[animal]
            if not isinstance(adat, h5py.Group) or 'messentials' not in adat:
                continue
            me = adat['messentials']

            if ref_shape is None and '_ref_vfs_shape' in me:
                ref_shape = tuple(np.asarray(me['_ref_vfs_shape']).astype(int))

            for pos in sorted(me.keys()):
                pdat = me[pos]
                if not isinstance(pdat, h5py.Group) or 'model' not in pdat:
                    continue
                model = pdat['model']
                if 'full_r2' not in model or 'vfs_cell_pos' not in pdat:
                    continue

                r2 = np.atleast_1d(np.asarray(model['full_r2'], dtype=float))
                n = len(r2)

                xy = np.asarray(pdat['vfs_cell_pos'], dtype=float)
                if xy.ndim != 2 or xy.shape[1] < 2:
                    continue
                n = min(n, xy.shape[0])

                aid = np.zeros(n, dtype=int)
                if 'visual_area_id' in pdat:
                    raw = np.atleast_1d(np.asarray(pdat['visual_area_id'], dtype=int))
                    m = min(len(raw), n)
                    aid[:m] = raw[:m]

                li = np.column_stack([
                    _read(model, f'full_trainLight_testLight_ablation_index_{v}', n)
                    for v in _IMP_VAR_ORDER]) if n_var else np.zeros((n, 0))
                di = np.column_stack([
                    _read(model, f'full_trainDark_testDark_ablation_index_{v}', n)
                    for v in _IMP_VAR_ORDER]) if n_var else np.zeros((n, 0))
                lg = np.column_stack([
                    _read(model, f'full_trainLight_testLight_group_ablation_index_{g}', n)
                    for g in _GROUP_VAR_ORDER]) if n_grp else np.zeros((n, 0))
                dg = np.column_stack([
                    _read(model, f'full_trainDark_testDark_group_ablation_index_{g}', n)
                    for g in _GROUP_VAR_ORDER]) if n_grp else np.zeros((n, 0))

                animals.append(np.array([animal] * n))
                poss.append(np.array([pos] * n))
                cis.append(np.arange(n))
                x.append(xy[:n, 0])
                y.append(xy[:n, 1])
                area_id.append(aid)
                full_r2.append(r2[:n])
                light_imp.append(np.clip(li, 0.0, 1.0))
                dark_imp.append(np.clip(di, 0.0, 1.0))
                light_grp.append(np.clip(lg, 0.0, 1.0))
                dark_grp.append(np.clip(dg, 0.0, 1.0))

    if not x:
        return None

    cells = dict(
        x=np.concatenate(x), y=np.concatenate(y),
        area_id=np.concatenate(area_id), full_r2=np.concatenate(full_r2),
        light_imp=np.vstack(light_imp), dark_imp=np.vstack(dark_imp),
        light_grp=np.vstack(light_grp), dark_grp=np.vstack(dark_grp),
        animal=np.concatenate(animals), pos=np.concatenate(poss),
        ci=np.concatenate(cis),
        ref_shape=ref_shape if ref_shape is not None else REF_SHAPE_FALLBACK,
    )
    cells['area'] = np.array([_IMP_ID_TO_NAME.get(int(a), '') for a in cells['area_id']])

    print(f'Importance cells loaded: {len(cells["x"])}')
    return cells


def apply_r2_filter(cells):
    if not APPLY_R2_THRESHOLD:
        print('R^2 threshold filter OFF: using all cells.')
        return cells
    keep = np.isfinite(cells['full_r2']) & (cells['full_r2'] > R2_THRESHOLD)
    n_before = len(keep)
    out = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == n_before else v)
           for k, v in cells.items()}
    print(f'R^2 threshold filter ON: kept {int(keep.sum())}/{n_before} cells '
          f'(full_r2 > {R2_THRESHOLD}).')
    return out


def load_ref_contours(contours_path):
    """Area outlines in reference VFS space, {area_name: (K, 2) xy array}."""
    with open(contours_path) as fh:
        raw = json.load(fh)
    out = {}
    for area, coords in raw.items():
        pts = np.asarray(coords, dtype=float)
        if pts.ndim == 2 and pts.shape[0] >= 3 and area in AREA_IDS:
            out[area] = pts[:, :2]
    return out


def build_labeled_array(contours, shape):
    labeled = np.zeros(shape, dtype=int)
    gy, gx = np.mgrid[:shape[0], :shape[1]]
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    for area, pts in contours.items():
        mask = MplPath(pts).contains_points(grid).reshape(shape)
        labeled[mask] = AREA_IDS[area]
    return labeled


# ----------------------------------------------------------------------------
# map drawing
# ----------------------------------------------------------------------------

def create_smoothed_map(x, y, values, shape, sigma=SMOOTH_SIGMA):
    """NaN-aware interpolate-then-smooth, as in topography_plots.py."""
    if len(values) < 4:
        return np.full(shape, np.nan)
    grid_y, grid_x = np.mgrid[0:shape[0], 0:shape[1]]
    points = np.column_stack((y, x))
    smoothed = griddata(points, values, (grid_y, grid_x), method='linear')
    V = smoothed.copy(); V[np.isnan(V)] = 0
    VV = gaussian_filter(V, sigma=sigma)
    W = np.ones_like(smoothed); W[np.isnan(smoothed)] = 0
    WW = gaussian_filter(W, sigma=sigma)
    return np.divide(VV, WW, out=np.full_like(VV, np.nan), where=WW != 0)


def _covered(x, y, shape, radius=COVERAGE_RADIUS):
    """Boolean image: pixels within `radius` px of at least one cell.

    Without this the interpolate-then-smooth step extrapolates a value across
    every pixel of an area, even where no cells were recorded.
    """
    grid_y, grid_x = np.mgrid[0:shape[0], 0:shape[1]]
    tree = cKDTree(np.column_stack([x, y]))
    dist, _ = tree.query(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
    return (dist <= radius).reshape(shape)


def _draw_outlines(ax, contours, label=True):
    for area, pts in contours.items():
        closed = np.vstack([pts, pts[:1]])
        ax.plot(closed[:, 0], closed[:, 1], '-', color='k', lw=0.8, alpha=0.7,
                zorder=6)
        if label:
            ax.text(np.mean(pts[:, 0]), np.mean(pts[:, 1]), area,
                    ha='center', va='center', fontsize=7, fontweight='bold',
                    color=COLORS.get(area, 'k'), zorder=7,
                    path_effects=[pe.withStroke(linewidth=2, foreground='w')])


def _finish_map_ax(ax, shape, title):
    ax.set_xlim(0, shape[1])
    ax.set_ylim(0, shape[0])
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=8)


def _area_smoothed(cells, vals, labeled_array, contours):
    """Per-area smoothed map, each area masked to its own outline."""
    shape = labeled_array.shape
    out = np.full(shape, np.nan)
    n_used = 0
    for area in contours:
        rid = AREA_IDS[area]
        region_mask = labeled_array == rid
        sel = (cells['area_id'] == rid) & np.isfinite(vals)
        if int(sel.sum()) < MIN_CELLS_SMOOTH:
            continue
        sm = create_smoothed_map(cells['x'][sel], cells['y'][sel], vals[sel], shape)
        sm[~_covered(cells['x'][sel], cells['y'][sel], shape)] = np.nan
        out[region_mask] = sm[region_mask]
        n_used += 1
    return out, n_used


def make_map_page(pdf, cells, vals, labeled_array, contours,
                  title, cbar_label, cmap, norm, cbar_extend='neither'):
    """One PDF page: cell scatter + per-area smoothed map of `vals`."""
    shape = labeled_array.shape
    finite = np.isfinite(vals)
    if int(finite.sum()) < MIN_CELLS_AREA:
        print(f'  skip "{title}" — only {int(finite.sum())} cells with data')
        return False

    fig, axs = plt.subplots(1, 2, figsize=(8.5, 4.6), dpi=200)

    ax = axs[0]
    ax.imshow(labeled_array > 1, cmap='binary', alpha=0.08)
    _draw_outlines(ax, contours)
    ax.scatter(cells['x'][finite], cells['y'][finite], s=SCATTER_SIZE,
               c=vals[finite], cmap=cmap, norm=norm, linewidths=0)
    _finish_map_ax(ax, shape, f'cells (n = {int(finite.sum())})')

    ax = axs[1]
    smoothed, n_areas = _area_smoothed(cells, vals, labeled_array, contours)
    ax.imshow(labeled_array > 1, cmap='binary', alpha=0.08)
    if n_areas:
        ax.imshow(smoothed, cmap=cmap, norm=norm)
    _draw_outlines(ax, contours)
    _finish_map_ax(ax, shape, f'smoothed within area ({n_areas} areas)')

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs,
                 label=cbar_label, fraction=0.045, shrink=0.75,
                 extend=cbar_extend)
    fig.suptitle(title, fontsize=9)

    # per-area medians, so the page is readable without eyeballing the colours
    lines = []
    for area in _IMP_REGION_ORDER:
        rid = AREA_IDS.get(area)
        sel = (cells['area_id'] == rid) & finite
        if int(sel.sum()) >= MIN_CELLS_AREA:
            lines.append(f'{area}: {np.median(vals[sel]):.3f} (n={int(sel.sum())})')
    if lines:
        fig.text(0.5, 0.015, 'median  ' + '   ·   '.join(lines),
                 ha='center', va='bottom', fontsize=6.5)

    pdf.savefig(fig)
    plt.close(fig)
    return True


def _robust_vmax(vals, pct=95, floor=0.05):
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return floor
    return float(max(np.percentile(v, pct), floor))


# ---------------------------------------------------------------------------
# Block-split tuning reliability: light-only / dark-only / both / neither
#
# The reliability already stored in the eyehead_revcorrs files
# ('<var>_l_rel' / '<var>_l_isrel', from calc_reliability_over() in
# fm2p/eyehead_revcorr.py) is NOT what is computed here. That one:
#   * scores a cross-validated modulation index (hi - lo) / (hi + lo) rather
#     than a split-half correlation,
#   * splits on 20 randomly shuffled micro-chunks rather than on the
#     alternating light/dark blocks,
#   * calls a cell reliable by thresholding that index at a fixed 0.1, with
#     no null distribution at all.
#
# What is computed below instead, per cell and per variable, within each
# condition separately:
#   * tuning curve from the odd-numbered blocks of that condition,
#   * tuning curve from the even-numbered blocks,
#   * Pearson correlation between them = within-condition reliability,
#   * a null built by circularly shifting the neural trace against the
#     behavioral variable (equivalent to shifting behavior against the trace,
#     and it preserves the autocorrelation of both), re-running the same
#     odd/even split-half correlation at every shift.
# A cell is light-tuned if its within-light reliability beats its own null,
# dark-tuned if within-dark does, giving a clean 2x2.
#
# Nothing here changes how reliability is computed anywhere else in the repo.
# ---------------------------------------------------------------------------

# yaw and gyro_z are read straight from the preproc traces by _load_traces
# ('head_yaw_deg' and 'gyro_z_twop_interp'), so both can be tested here even
# though the MI pipeline in summarize_all_tuning.py never wrote MI fields for
# gyro_z.  Note yaw is binned linearly over its 0-360 range: the tuning curve
# is circular but the split-half and shift-null tests do not care, and the two
# end bins straddle the wrap point.
REL_VAR_KEYS = ['theta', 'phi', 'pitch', 'roll', 'dTheta', 'dPhi', 'gyro_y',
                'gyro_x', 'yaw', 'gyro_z']

REL_N_BINS      = 12    # tuning bins (13 edges), matching calc_1d_tuning()
REL_BIN_BOUND   = 2     # percentile bound for the bin edges, as in calc_1d_tuning()
REL_MIN_PER_BIN = 5     # frames a bin needs, in each half, to be usable
REL_MIN_BINS    = 5     # usable bins needed to correlate two half-curves
REL_ALPHA       = 0.05
REL_MIN_SHIFT   = 0.05  # null shifts closer than this fraction of the condition
                        # length to 0 (or to a full wrap) are discarded
REL_CELL_CHUNK  = 16    # cells per FFT batch, to bound peak memory
REL_CORR_THRESH = 0.2   # |r| below this is "unrelated"; matches
                        # _TUNING_CORR_INVARIANT_THRESH in summarize_all_tuning

REL_MIN_CELLS_AREA = 20   # cells an area needs before its bar is drawn

# Light/dark overlap panel.  The CI is a CLUSTER bootstrap over recordings,
# not over cells: cells within a recording share an animal, a FOV, a set of
# light/dark blocks and a behavioural trajectory, so a cell-level bootstrap
# would report an interval several times too narrow.
OVERLAP_N_BOOT = 5000
OVERLAP_EQUIV  = (0.8, 1.25)   # pre-specified "practically independent" band

REL_CATEGORIES = ['light only', 'dark only', 'both', 'neither']
REL_CAT_COLORS = {'light only': '#E8B84B', 'dark only': '#2C3E70',
                  'both': '#4A8C6F', 'neither': '#BBBBBB'}
REL_SPLIT_COLORS = {'preserved': '#2E7D5B', 'inverted': '#B03A48',
                    'unrelated': '#BBBBBB'}


def resolve_data_path(path):
    """Tolerate the _V1PPC -> V1PPC rename of the data root.

    The paths inherited from summarize_all_tuning.py still point at
    .../freely_moving_data/_V1PPC/..., which no longer exists.
    """
    if path is None or os.path.exists(path):
        return path
    alt = path.replace('_V1PPC', 'V1PPC')
    if alt != path and os.path.exists(alt):
        print(f'{path} not found; using {alt} instead.')
        return alt
    return path


def _preproc_files(base_dir):
    """Every fm recording preproc.h5 under base_dir."""
    hits = glob.glob(os.path.join(base_dir, '**', '*preproc.h5'), recursive=True)
    return sorted(
        p for p in hits
        if 'boundary' not in p
        and not os.path.basename(p).startswith('sn')
        and os.path.basename(os.path.dirname(p)).startswith('fm')
    )


def _load_traces(preproc_path, var_keys=REL_VAR_KEYS):
    """Spikes, light/dark state, and the behavioral variables, at 2P frame rate.

    dTheta / dPhi are stored at eye-camera rate, so they get interpolated onto
    twopT exactly the way eyehead_revcorr.py does it.
    """
    direct = {'theta': 'theta_interp', 'phi': 'phi_interp',
              'pitch': 'pitch_twop_interp', 'roll': 'roll_twop_interp',
              'gyro_x': 'gyro_x_twop_interp', 'gyro_y': 'gyro_y_twop_interp',
              'gyro_z': 'gyro_z_twop_interp', 'yaw': 'head_yaw_deg'}

    with h5py.File(preproc_path, 'r') as f:
        if 'norm_spikes' not in f or 'ltdk_state_vec' not in f:
            return None
        spikes = f['norm_spikes'][:].astype(np.float64)
        ltdk   = np.asarray(f['ltdk_state_vec'][:]).astype(bool)
        twopT  = f['twopT'][:].astype(float) if 'twopT' in f else None

        n_frames = spikes.shape[1]
        ltdk = ltdk[:n_frames]
        if len(ltdk) < n_frames:
            return None

        beh = {}
        for v in var_keys:
            arr = None
            if v in direct and direct[v] in f:
                arr = f[direct[v]][:].astype(float)
            elif v in ('dTheta', 'dPhi') and v in f and 'eyeT1' in f and twopT is not None:
                raw   = f[v][:].astype(float)
                eyeT1 = f['eyeT1'][:].astype(float)
                n     = min(len(raw), len(eyeT1))
                try:
                    arr = interpT(interp_short_gaps(raw[:n]), eyeT1[:n], twopT)
                except Exception:
                    arr = None
            if arr is None:
                continue
            if len(arr) >= n_frames:
                beh[v] = arr[:n_frames]

    return dict(spikes=spikes, ltdk=ltdk, beh=beh)


def _block_ids(ltdk, want_light):
    """Block index per frame for one condition, -1 for frames in the other.

    The light/dark TTL alternates in ~3-minute blocks, so contiguous runs of
    the state vector are exactly those blocks.
    """
    want = ltdk if want_light else ~ltdk
    ids  = np.full(len(want), -1, dtype=int)
    if not np.any(want):
        return ids
    edges = np.flatnonzero(np.diff(want.astype(int))) + 1
    starts = np.concatenate([[0], edges])
    stops  = np.concatenate([edges, [len(want)]])
    b = 0
    for s, e in zip(starts, stops):
        if want[s]:
            ids[s:e] = b
            b += 1
    return ids


def _grouped_mean(spikes, bidx, n_bins):
    """Mean rate per bin, (n_cells, n_bins); NaN for empty bins."""
    counts = np.bincount(bidx, minlength=n_bins).astype(float)
    onehot = np.zeros((len(bidx), n_bins))
    onehot[np.arange(len(bidx)), bidx] = 1.0
    sums = spikes @ onehot
    out = np.full_like(sums, np.nan)
    ok = counts > 0
    out[:, ok] = sums[:, ok] / counts[ok]
    return out


def _rowwise_corr(a, b, axis):
    """Pearson r between a and b along `axis`, broadcasting the rest."""
    a = a - a.mean(axis=axis, keepdims=True)
    b = b - b.mean(axis=axis, keepdims=True)
    num = (a * b).sum(axis=axis)
    den = np.sqrt((a * a).sum(axis=axis) * (b * b).sum(axis=axis))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)


def _condition_reliability(spikes, beh, block_ids, edges):
    """Odd-vs-even-block split-half reliability and its circular-shift null.

    Returns (r_real, p, tuning) with one entry per cell: the split-half
    correlation, the one-sided p-value against the shift null, and the
    full-condition tuning curve.

    The null shifts the neural trace circularly against the behavioral
    variable. Doing it that way round (rather than shifting behavior) leaves
    the bin assignment fixed, which lets every shift be evaluated at once as a
    circular cross-correlation via the FFT.
    """
    n_cells = spikes.shape[0]
    nan = np.full(n_cells, np.nan)

    use = (block_ids >= 0) & np.isfinite(beh)
    if int(use.sum()) < 2 * REL_N_BINS:
        return nan, nan, np.full((n_cells, REL_N_BINS), np.nan)

    S    = spikes[:, use]
    bidx = np.clip(np.digitize(beh[use], edges) - 1, 0, REL_N_BINS - 1)
    blk  = block_ids[use]
    T    = S.shape[1]

    tuning = _grouped_mean(S, bidx, REL_N_BINS)

    if len(np.unique(blk)) < 2:      # need both an odd and an even block
        return nan, nan, tuning

    masks, counts, Mf = [], [], []
    for parity in (0, 1):
        m = (blk % 2) == parity
        oh = np.zeros((T, REL_N_BINS))
        idx = np.flatnonzero(m)
        oh[idx, bidx[idx]] = 1.0
        masks.append(oh)
        counts.append(oh.sum(axis=0))
        Mf.append(np.fft.rfft(oh.T, n=T, axis=1))

    ok = (counts[0] >= REL_MIN_PER_BIN) & (counts[1] >= REL_MIN_PER_BIN)
    if int(ok.sum()) < REL_MIN_BINS:
        return nan, nan, tuning

    min_shift = max(int(REL_MIN_SHIFT * T), 1)
    shifts = np.arange(T)
    valid_shift = (shifts >= min_shift) & (shifts <= T - min_shift)
    if int(valid_shift.sum()) < 20:
        return nan, nan, tuning

    Sf = np.fft.rfft(S, axis=1)

    r_real = np.full(n_cells, np.nan)
    p_val  = np.full(n_cells, np.nan)

    for c0 in range(0, n_cells, REL_CELL_CHUNK):
        c1 = min(c0 + REL_CELL_CHUNK, n_cells)
        curves = []
        for parity in (0, 1):
            prod = Sf[c0:c1][:, None, :] * np.conj(Mf[parity])[None, :, :]
            sums = np.fft.irfft(prod, n=T, axis=2)          # (chunk, bins, shifts)
            curves.append(sums[:, ok, :] / counts[parity][ok][None, :, None])
        r = _rowwise_corr(curves[0], curves[1], axis=1)      # (chunk, shifts)

        r0 = r[:, 0]
        null = r[:, valid_shift]
        n_null = null.shape[1]
        ge = np.sum(null >= r0[:, None], axis=1)
        r_real[c0:c1] = r0
        p_val[c0:c1]  = (1.0 + ge) / (1.0 + n_null)

    return r_real, p_val, tuning


def compute_block_reliability(base_dir, pooled_path, var_keys=REL_VAR_KEYS):
    """Per-cell odd/even-block reliability in light and in dark, for every recording."""
    pooled_lookup = _build_pooled_lookup(pooled_path)
    files = _preproc_files(base_dir)
    print(f'Found {len(files)} fm preproc.h5 file(s) under {base_dir}.')

    records = []
    n_no_match = n_one_block = 0

    for pp in files:
        match = _match_to_pooled(pp, pooled_lookup)
        if match is None:
            n_no_match += 1
            continue
        animal, pos, va_ids = match

        tr = _load_traces(pp, var_keys)
        if tr is None:
            continue
        spikes, ltdk, beh_all = tr['spikes'], tr['ltdk'], tr['beh']

        n_cells = min(spikes.shape[0], len(va_ids))
        blk_l = _block_ids(ltdk, True)
        blk_d = _block_ids(ltdk, False)
        n_bl  = len(np.unique(blk_l[blk_l >= 0]))
        n_bd  = len(np.unique(blk_d[blk_d >= 0]))
        if n_bl < 2 or n_bd < 2:
            print(f'  {animal}/{pos}: only {n_bl} light / {n_bd} dark block(s) '
                  f'-- cannot split odd vs even, skipping')
            n_one_block += 1
            continue

        print(f'  {animal}/{pos}: {n_cells} cells, {n_bl} light / {n_bd} dark blocks')

        per_var = {}
        for v in var_keys:
            beh = beh_all.get(v)
            if beh is None:
                continue
            fin = np.isfinite(beh)
            if int(fin.sum()) < 4 * REL_N_BINS:
                continue
            # One set of bin edges per variable, shared by light and dark, so
            # the two conditions' curves are directly comparable.
            edges = np.linspace(np.nanpercentile(beh[fin], REL_BIN_BOUND),
                                np.nanpercentile(beh[fin], 100 - REL_BIN_BOUND),
                                REL_N_BINS + 1)
            rl, pl, tl = _condition_reliability(spikes[:n_cells], beh, blk_l, edges)
            rd, pd, td = _condition_reliability(spikes[:n_cells], beh, blk_d, edges)
            per_var[v] = dict(r_light=rl, p_light=pl, r_dark=rd, p_dark=pd,
                              corr_ld=_rowwise_corr(tl, td, axis=1))

        for ci in range(n_cells):
            area = _IMP_ID_TO_NAME.get(int(va_ids[ci]))
            if area is None:
                continue
            rec = dict(animal=animal, pos=pos, ci=ci, area=area)
            for v in var_keys:
                d = per_var.get(v)
                if d is None:
                    rec[f'{v}_p_light'] = np.nan
                    rec[f'{v}_p_dark']  = np.nan
                    rec[f'{v}_r_light'] = np.nan
                    rec[f'{v}_r_dark']  = np.nan
                    rec[f'{v}_corr_ld'] = np.nan
                else:
                    rec[f'{v}_p_light'] = float(d['p_light'][ci])
                    rec[f'{v}_p_dark']  = float(d['p_dark'][ci])
                    rec[f'{v}_r_light'] = float(d['r_light'][ci])
                    rec[f'{v}_r_dark']  = float(d['r_dark'][ci])
                    rec[f'{v}_corr_ld'] = float(d['corr_ld'][ci])
            records.append(rec)

    print(f'Block-reliability cells: {len(records)} '
          f'({n_no_match} recording(s) unmatched to pooled, '
          f'{n_one_block} with too few blocks)')
    return records


def load_or_compute_block_reliability(base_dir, pooled_path, cache_path,
                                      recompute=False, var_keys=REL_VAR_KEYS):
    """Block-split reliability records, cached so reruns skip the FFT nulls."""
    if (not recompute) and cache_path and os.path.isfile(cache_path):
        with open(cache_path, 'rb') as fh:
            cached = pickle.load(fh)
        # Older caches are a bare list with no record of which variables were
        # tested.  Accept one only if it covers every variable asked for --
        # otherwise a stale pickle silently reinstates a narrower variable set.
        if isinstance(cached, dict) and 'records' in cached:
            records, have = cached['records'], set(cached.get('var_keys', []))
        else:
            records, have = cached, set()
            for rec in records[:50]:
                have |= {k[:-8] for k in rec if k.endswith('_p_light')}
        missing = [v for v in var_keys if v not in have]
        if missing:
            print(f'Cached reliability is missing {missing} — recomputing.')
        else:
            print(f'Loaded {len(records)} block-reliability cells from {cache_path}')
            return records

    records = compute_block_reliability(base_dir, pooled_path, var_keys)
    if cache_path:
        with open(cache_path, 'wb') as fh:
            pickle.dump({'records': records, 'var_keys': list(var_keys)}, fh)
        print(f'Cached block-reliability records -> {cache_path}')
    return records


def annotate_glm_membership(records, cells):
    """Flag which reliability cells are also encoding-model cells.

    `cells` is the R^2-filtered GLM population the maps are built from, so this
    is what lets the same 2x2 be reported for the encoding-model cells alone.
    """
    if cells is None:
        for rec in records:
            rec['in_glm'] = False
        return records
    glm_keys = {(a, p, int(c)) for a, p, c in
                zip(cells['animal'], cells['pos'], cells['ci'])}
    n = 0
    for rec in records:
        rec['in_glm'] = (rec['animal'], rec['pos'], int(rec['ci'])) in glm_keys
        n += int(rec['in_glm'])
    print(f'{n}/{len(records)} block-reliability cells are also '
          f'encoding-model (GLM) cells.')
    return records


def _classify(rec, var, alpha=REL_ALPHA):
    """light only / dark only / both / neither, or None if untestable."""
    pl, pd = rec.get(f'{var}_p_light'), rec.get(f'{var}_p_dark')
    if pl is None or pd is None or not (np.isfinite(pl) and np.isfinite(pd)):
        return None
    l, d = pl < alpha, pd < alpha
    if l and d:
        return 'both'
    if l:
        return 'light only'
    if d:
        return 'dark only'
    return 'neither'


def _contingency(records, var, areas, categories, alpha=REL_ALPHA):
    """Counts of each category per area, (n_areas, n_categories)."""
    tab = np.zeros((len(areas), len(categories)), dtype=int)
    for rec in records:
        if rec['area'] not in areas:
            continue
        cat = _classify(rec, var, alpha)
        if cat is None or cat not in categories:
            continue
        tab[areas.index(rec['area']), categories.index(cat)] += 1
    return tab


def _chi2(tab):
    """chi2_contingency on a table with all-zero rows/columns removed."""
    keep_r = tab.sum(axis=1) > 0
    keep_c = tab.sum(axis=0) > 0
    sub = tab[np.ix_(keep_r, keep_c)]
    if sub.shape[0] < 2 or sub.shape[1] < 2:
        return None
    try:
        chi2, p, dof, _ = chi2_contingency(sub)
        return chi2, p, dof
    except ValueError:
        return None


def _stacked_bars(ax, tab, areas, categories, colors, min_n=REL_MIN_CELLS_AREA):
    """Stacked proportion bars, one per area, with n annotated above."""
    drawn = []
    for ai, area in enumerate(areas):
        total = tab[ai].sum()
        if total < min_n:
            continue
        bottom = 0.0
        for ci, cat in enumerate(categories):
            frac = tab[ai, ci] / total
            ax.bar(ai, frac, bottom=bottom, width=0.72,
                   color=colors[cat], edgecolor='k', linewidth=0.4)
            bottom += frac
        ax.text(ai, 1.01, f'{total}', ha='center', va='bottom', fontsize=5.5,
                color='0.3')
        drawn.append(ai)
    ax.set_xticks(range(len(areas)))
    ax.set_xticklabels(areas, fontsize=6.5)
    ax.set_xlim(-0.6, len(areas) - 0.4)
    ax.set_ylim(0, 1)
    return drawn


def make_tuning_class_by_area_svg(records, out_dir, var_keys=REL_VAR_KEYS,
                                  fname='tuning_light_dark_class_by_area.svg'):
    """Does the light-only / dark-only / both split map onto visual area?"""
    areas = [a for a in _IMP_REGION_ORDER
             if sum(r['area'] == a for r in records) >= REL_MIN_CELLS_AREA]
    if not areas:
        print('  no areas with enough cells — skipping tuning-class figure')
        return False

    cats3 = ['light only', 'dark only', 'both']

    # Grid sized from the variable list, not hardcoded: the list grew from 8
    # to 10 when yaw and dYaw were added back.
    n_col = 4
    n_row = int(np.ceil(len(var_keys) / n_col))
    fig, axs = plt.subplots(n_row, n_col,
                            figsize=(11, 2.6 * n_row), dpi=300,
                            constrained_layout=True, squeeze=False)
    axs = axs.flatten()
    for _ax in axs[len(var_keys):]:
        _ax.axis('off')

    print('\n' + '=' * 74)
    print('WITHIN-CONDITION BLOCK-SPLIT RELIABILITY (odd vs even blocks, '
          f'circular-shift null, alpha={REL_ALPHA})')
    print('=' * 74)

    for vi, var in enumerate(var_keys):
        ax = axs[vi]
        tab4 = _contingency(records, var, areas, REL_CATEGORIES)
        tab3 = tab4[:, [REL_CATEGORIES.index(c) for c in cats3]]

        _stacked_bars(ax, tab4, areas, REL_CATEGORIES, REL_CAT_COLORS)

        res3 = _chi2(tab3)
        res4 = _chi2(tab4)
        label = _ALL_VAR_LABELS.get(var, var)
        ptxt = 'n/a' if res3 is None else f'p={res3[1]:.2g}'
        ax.set_title(f'{label}   $\\chi^2$ {ptxt}', fontsize=7.5)
        if vi % 4 == 0:
            ax.set_ylabel('fraction of cells', fontsize=7)

        n_tested = tab4.sum()
        n_tuned  = tab3.sum()
        print(f'\n{label}: {n_tested} cells tested, {n_tuned} tuned in at '
              f'least one condition')
        for ai, area in enumerate(areas):
            tot = tab4[ai].sum()
            if tot < REL_MIN_CELLS_AREA:
                continue
            parts = '  '.join(f'{c}={tab4[ai, REL_CATEGORIES.index(c)]:4d} '
                              f'({tab4[ai, REL_CATEGORIES.index(c)] / tot:5.1%})'
                              for c in REL_CATEGORIES)
            print(f'   {area:<3s} n={tot:5d}  {parts}')
        if res3 is not None:
            print(f'   area x (light-only/dark-only/both): '
                  f'chi2={res3[0]:.1f}, dof={res3[2]}, p={res3[1]:.3g}')
        if res4 is not None:
            print(f'   area x all four categories:         '
                  f'chi2={res4[0]:.1f}, dof={res4[2]}, p={res4[1]:.3g}')

    for ax in axs[len(var_keys):]:
        ax.axis('off')

    fig.legend(handles=[Patch(facecolor=REL_CAT_COLORS[c], edgecolor='k',
                              linewidth=0.4, label=c) for c in REL_CATEGORIES],
               frameon=False, fontsize=7, ncol=4, loc='lower center')
    fig.suptitle('Tuned in light only / dark only / both, by visual area\n'
                 'within-condition odd-vs-even-block split-half reliability '
                 f'vs. circular-shift null (alpha={REL_ALPHA}); n above each bar',
                 fontsize=9)
    print('=' * 74)

    _save_svg_png(fig, os.path.join(out_dir, fname))
    return True


def make_preserved_inverted_by_area_svg(records, out_dir, var_keys=REL_VAR_KEYS,
                                        fname='both_tuned_preserved_inverted_by_area.svg'):
    """For cells tuned in BOTH conditions, split by light-vs-dark tuning correlation."""
    areas = [a for a in _IMP_REGION_ORDER
             if sum(r['area'] == a for r in records) >= REL_MIN_CELLS_AREA]
    if not areas:
        return False

    cats = ['preserved', 'inverted', 'unrelated']

    # Grid sized from the variable list, not hardcoded: the list grew from 8
    # to 10 when yaw and dYaw were added back.
    n_col = 4
    n_row = int(np.ceil(len(var_keys) / n_col))
    fig, axs = plt.subplots(n_row, n_col,
                            figsize=(11, 2.6 * n_row), dpi=300,
                            constrained_layout=True, squeeze=False)
    axs = axs.flatten()
    for _ax in axs[len(var_keys):]:
        _ax.axis('off')

    print('\n' + '=' * 74)
    print('CELLS TUNED IN BOTH CONDITIONS: preserved vs inverted '
          f'(|r| > {REL_CORR_THRESH} on the light-vs-dark tuning correlation)')
    print('=' * 74)

    for vi, var in enumerate(var_keys):
        ax = axs[vi]
        tab = np.zeros((len(areas), len(cats)), dtype=int)
        for rec in records:
            if rec['area'] not in areas:
                continue
            if _classify(rec, var) != 'both':
                continue
            r = rec.get(f'{var}_corr_ld', np.nan)
            if not np.isfinite(r):
                continue
            cat = ('preserved' if r > REL_CORR_THRESH
                   else 'inverted' if r < -REL_CORR_THRESH
                   else 'unrelated')
            tab[areas.index(rec['area']), cats.index(cat)] += 1

        _stacked_bars(ax, tab, areas, cats, REL_SPLIT_COLORS)

        res = _chi2(tab)
        label = _ALL_VAR_LABELS.get(var, var)
        ptxt = 'n/a' if res is None else f'p={res[1]:.2g}'
        ax.set_title(f'{label}   $\\chi^2$ {ptxt}', fontsize=7.5)
        if vi % 4 == 0:
            ax.set_ylabel('fraction of "both" cells', fontsize=7)

        print(f'\n{label}: {tab.sum()} cells tuned in both conditions')
        for ai, area in enumerate(areas):
            tot = tab[ai].sum()
            if tot < REL_MIN_CELLS_AREA:
                continue
            parts = '  '.join(f'{c}={tab[ai, ci]:4d} ({tab[ai, ci] / tot:5.1%})'
                              for ci, c in enumerate(cats))
            print(f'   {area:<3s} n={tot:5d}  {parts}')
        if res is not None:
            print(f'   area x (preserved/inverted/unrelated): '
                  f'chi2={res[0]:.1f}, dof={res[2]}, p={res[1]:.3g}')

    for ax in axs[len(var_keys):]:
        ax.axis('off')

    fig.legend(handles=[Patch(facecolor=REL_SPLIT_COLORS[c], edgecolor='k',
                              linewidth=0.4, label=c) for c in cats],
               frameon=False, fontsize=7, ncol=3, loc='lower center')
    fig.suptitle('Cells tuned in BOTH conditions, split by the light-vs-dark '
                 'tuning correlation\n'
                 f'preserved r > {REL_CORR_THRESH}, inverted r < -{REL_CORR_THRESH}; '
                 'n above each bar', fontsize=9)
    print('=' * 74)

    _save_svg_png(fig, os.path.join(out_dir, fname))
    return True


# ----------------------------------------------------------------------------
# light-tuned vs dark-tuned overlap
# ----------------------------------------------------------------------------

def _overlap_counts_per_cluster(records, var, alpha=REL_ALPHA):
    """Per-recording (N, n_light, n_dark, n_both) counts for one variable.

    A cell contributes only if it was testable in BOTH conditions, so the
    denominator of every proportion below is the same set of cells.
    """
    per = {}
    for rec in records:
        cat = _classify(rec, var, alpha)
        if cat is None:
            continue
        key = (rec['animal'], rec['pos'])
        c = per.setdefault(key, np.zeros(4))
        c[0] += 1                                        # N, testable in both
        c[1] += cat in ('light only', 'both')            # tuned in light
        c[2] += cat in ('dark only', 'both')             # tuned in dark
        c[3] += cat == 'both'                            # tuned in both
    if not per:
        return np.zeros((0, 4))
    return np.vstack([per[k] for k in sorted(per)])


OVERLAP_VERDICT_COLORS = {
    'above chance':  '#B03A48',
    'below chance':  '#4C6EB1',
    'independent':   '#2E7D5B',
    'inconclusive':  '#8A8A8A',
}


def _overlap_verdict(lo, hi):
    """Three-way read of a CI: excess overlap, independence, or too wide to say.

    A binary "inside the equivalence band or not" would file a wide,
    underpowered CI (say [0.60, 1.24]) under the same heading as a CI that
    sits entirely above 1, even though the first says nothing at all and the
    second is real excess overlap.  They get different colours and different
    words.
    """
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return 'inconclusive'
    if lo > 1.0:
        return 'above chance'
    if hi < 1.0:
        return 'below chance'
    if lo >= OVERLAP_EQUIV[0] and hi <= OVERLAP_EQUIV[1]:
        return 'independent'
    return 'inconclusive'


def _oe_ratio(counts):
    """Observed / expected overlap from summed (N, n_L, n_D, n_both) counts.

    Expected number of double-tuned cells under independence is
    N * (n_L / N) * (n_D / N) = n_L * n_D / N, so the ratio is
    n_both * N / (n_L * n_D).  1.0 is exact independence.
    """
    N, nL, nD, nB = counts
    if N <= 0 or nL <= 0 or nD <= 0:
        return np.nan
    return float(nB * N / (nL * nD))


def overlap_stats(records, var, alpha=REL_ALPHA, n_boot=OVERLAP_N_BOOT, seed=0):
    """Observed/expected overlap of light- and dark-tuned cells, with a CI.

    Returns None if the variable has no testable cells.  The CI is a
    percentile cluster bootstrap: recordings are resampled with replacement
    and the ratio recomputed, which is legitimate here because all four counts
    the ratio is built from are plain sums over cells.

    Note on direction: under any model where a shared latent population of
    tuned cells is detected independently in each condition with probability
    q, the expected ratio is 1 / (fraction truly tuned) >= 1, and any
    cell-quality confound (bright, high-SNR cells being detectable in both
    conditions) pushes it further above 1.  So imperfect detection power
    cannot manufacture a ratio of 1 -- it can only inflate it.  A ratio at 1
    is therefore genuine independence, and this test is conservative against
    the independence claim.
    """
    per_cluster = _overlap_counts_per_cluster(records, var, alpha)
    if per_cluster.shape[0] == 0:
        return None
    tot = per_cluster.sum(axis=0)
    N, nL, nD, nB = tot
    if N <= 0:
        return None

    ratio = _oe_ratio(tot)
    expected = (nL * nD / N) if N > 0 else np.nan

    rng = np.random.default_rng(seed)
    n_clu = per_cluster.shape[0]
    idx = rng.integers(0, n_clu, size=(n_boot, n_clu))
    boot_tot = per_cluster[idx].sum(axis=1)              # (n_boot, 4)
    with np.errstate(invalid='ignore', divide='ignore'):
        bN, bL, bD, bB = boot_tot.T
        boot = np.where((bN > 0) & (bL > 0) & (bD > 0), bB * bN / (bL * bD), np.nan)
    boot = boot[np.isfinite(boot)]
    if boot.size >= 20:
        lo, hi = np.percentile(boot, [2.5, 97.5])
    else:
        lo = hi = np.nan

    # Fisher's exact on the pooled 2x2 ignores clustering, so it is reported
    # only as a cross-check on the point estimate, never as the headline test.
    a = int(round(nB))                        # light+ dark+
    b = int(round(nL - nB))                   # light+ dark-
    c = int(round(nD - nB))                   # light- dark+
    d = int(round(N - nL - nD + nB))          # light- dark-
    try:
        odds, p_fisher = fisher_exact([[a, b], [c, d]])
    except ValueError:
        odds, p_fisher = np.nan, np.nan

    union = nL + nD - nB
    return dict(
        var=var, n=int(N), n_light=int(nL), n_dark=int(nD), n_both=int(nB),
        n_clusters=n_clu, expected=float(expected), ratio=ratio,
        ci=(float(lo), float(hi)), boot=boot,
        p_light=float(nL / N), p_dark=float(nD / N),
        jaccard=float(nB / union) if union > 0 else np.nan,
        odds_ratio=float(odds), p_fisher=float(p_fisher),
        equivalent=bool(np.isfinite(lo) and np.isfinite(hi)
                        and lo >= OVERLAP_EQUIV[0] and hi <= OVERLAP_EQUIV[1]),
        verdict=_overlap_verdict(lo, hi),
    )


def make_light_dark_overlap_svg(records, out_dir, var_keys=REL_VAR_KEYS,
                                fname='light_dark_overlap_ratio.svg',
                                show_verdicts=True, ymax=None,
                                bottom_title=None):
    """Panel: is being light-tuned informative about being dark-tuned?

    Top axis is the observed/expected overlap ratio per variable on a log
    scale, with a cluster-bootstrap 95% CI and the independence line at 1.
    Bottom axis shows the arithmetic behind each ratio -- the observed count
    of double-tuned cells against the count independence predicts.
    """
    stats = [st for st in (overlap_stats(records, v) for v in var_keys)
             if st is not None]
    if not stats:
        print('  no testable cells — skipping light/dark overlap figure')
        return False

    labels = [_ALL_VAR_LABELS.get(st['var'], st['var']) for st in stats]
    xs = np.arange(len(stats))

    fig, axs = plt.subplots(2, 1, figsize=(6, 4), dpi=300, sharex=True,
                            gridspec_kw={'height_ratios': [1.35, 1.0]},
                            constrained_layout=True)

    print('\n' + '=' * 74)
    print('LIGHT-TUNED vs DARK-TUNED OVERLAP  (observed / expected under '
          'independence)')
    print(f'expected n_both = n_light * n_dark / N;  ratio 1.0 = independence;'
          f'\n95% CI = percentile cluster bootstrap over recordings '
          f'({OVERLAP_N_BOOT} reps);\npractical-independence band = '
          f'[{OVERLAP_EQUIV[0]}, {OVERLAP_EQUIV[1]}]')
    print('=' * 74)

    # --- top: the ratio -----------------------------------------------------
    ax = axs[0]
    if show_verdicts:
        ax.axhspan(OVERLAP_EQUIV[0], OVERLAP_EQUIV[1], color='0.85', alpha=0.6,
                   zorder=0, lw=0)
    ax.axhline(1.0, color='k', lw=0.9, ls='--', alpha=0.8, zorder=1)

    for i, st in enumerate(stats):
        lo, hi = st['ci']
        col = OVERLAP_VERDICT_COLORS[st['verdict']] if show_verdicts else '#4C6EB1'
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([i, i], [lo, hi], color=col, lw=1.6, solid_capstyle='round',
                    zorder=2)
        ax.plot(i, st['ratio'], 'o', ms=5.5, color=col, mec='k', mew=0.5,
                zorder=3)
        ax.text(i, 0.02, f'n={st["n"]}', transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=5, rotation=90, color='0.3')

    ax.set_yscale('log')
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator([0.5, 0.8, 1.0, 1.25, 2.0, 4.0]))
    ax.yaxis.set_major_formatter(mpl.ticker.FixedFormatter(
        ['0.5', '0.8', '1', '1.25', '2', '4']))
    if ymax is not None:
        ax.set_ylim(0.45, ymax)
    else:
        ax.set_ylim(0.45, max(4.0, max(st['ci'][1] for st in stats
                                       if np.isfinite(st['ci'][1])) * 1.15))
    ax.set_ylabel('observed / expected\ndouble-tuned cells', fontsize=7)
    if show_verdicts:
        ax.text(1.012, 0.99, 'overlap above chance', transform=ax.transAxes,
                rotation=90, va='top', ha='left', fontsize=5.5, color='0.35')
        ax.text(1.012, 0.01, 'overlap below chance', transform=ax.transAxes,
                rotation=90, va='bottom', ha='left', fontsize=5.5, color='0.35')
        ax.set_title('Do light-tuned and dark-tuned cells overlap more than chance?\n'
                     'ratio = n$_{both}$ / (n$_{light}$ n$_{dark}$ / N); '
                     '1 = independent; shaded band = practical independence '
                     f'[{OVERLAP_EQUIV[0]}, {OVERLAP_EQUIV[1]}]\n'
                     'error bars = 95% cluster bootstrap CI over recordings',
                     fontsize=8)
        ax.legend(handles=[Patch(facecolor=c, edgecolor='k', linewidth=0.4, label=v)
                           for v, c in OVERLAP_VERDICT_COLORS.items()],
                  frameon=False, fontsize=6, ncol=4, loc='upper center')
    else:
        ax.set_title('Light-tuned and dark-tuned cells are close to independent\n'
                     'ratio = n$_{both}$ / (n$_{light}$ n$_{dark}$ / N); '
                     '1 = exactly what independence predicts\n'
                     'error bars = 95% cluster bootstrap CI over recordings',
                     fontsize=8)

    # --- bottom: observed vs expected counts --------------------------------
    ax = axs[1]
    w = 0.36
    ax.bar(xs - w / 2, [st['n_both'] for st in stats], width=w,
           color='#4C6EB1', edgecolor='k', linewidth=0.4, label='observed both')
    ax.bar(xs + w / 2, [st['expected'] for st in stats], width=w,
           color='0.75', edgecolor='k', linewidth=0.4,
           label='expected if independent')
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlim(-0.6, len(stats) - 0.4)
    ax.set_ylabel('cells tuned in both conditions', fontsize=7)
    if bottom_title:
        ax.set_title(bottom_title, fontsize=8)
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc='upper right')
    ax.set_ylim(0, max(max(st['n_both'], st['expected']) for st in stats) * 1.42)
    # Jaccard is the absolute overlap: a ratio can sit well above 1 while the
    # two sets still share almost no cells, and the slide needs both numbers.
    for i, st in enumerate(stats):
        ax.text(i, max(st['n_both'], st['expected']) * 1.04,
                f'J={st["jaccard"]:.2f}', ha='center', va='bottom',
                fontsize=5.5, color='0.3')

    for st in stats:
        lo, hi = st['ci']
        flag = st['verdict']
        print(f'\n{_ALL_VAR_LABELS.get(st["var"], st["var"])}: '
              f'N={st["n"]} cells testable in both, '
              f'{st["n_clusters"]} recordings')
        print(f'   tuned light={st["n_light"]:4d} ({st["p_light"]:5.1%})   '
              f'tuned dark={st["n_dark"]:4d} ({st["p_dark"]:5.1%})   '
              f'both={st["n_both"]:4d}   expected={st["expected"]:6.1f}')
        print(f'   O/E ratio = {st["ratio"]:.3f}  '
              f'95% CI [{lo:.3f}, {hi:.3f}]  -> {flag}')
        print(f'   Jaccard={st["jaccard"]:.3f}   '
              f'odds ratio={st["odds_ratio"]:.3f}   '
              f'Fisher p={st["p_fisher"]:.3g} (unclustered, cross-check only)')

    from collections import Counter
    tally = Counter(st['verdict'] for st in stats)
    print(f'\nOut of {len(stats)} variables: '
          + ', '.join(f'{tally.get(v, 0)} {v}' for v in OVERLAP_VERDICT_COLORS))
    print(f'("independent" = 95% CI inside [{OVERLAP_EQUIV[0]}, '
          f'{OVERLAP_EQUIV[1]}]; "inconclusive" = CI too wide to place, '
          'NOT evidence of independence.)')
    print('Detection noise cannot push this ratio down to 1: under a shared '
          'tuned population\ndetected independently per condition the expected '
          'ratio is 1/(fraction tuned) >= 1,\nand SNR-driven detectability '
          'inflates it further. The test is conservative.')
    print('=' * 74)

    _save_svg_png(fig, os.path.join(out_dir, fname))
    return True



# ----------------------------------------------------------------------------
# pages
# ----------------------------------------------------------------------------
def _save_svg_png(fig, svg_path, dpi=300):
    png_path = os.path.splitext(svg_path)[0] + '.png'
    fig.savefig(svg_path, format='svg', bbox_inches='tight', dpi=dpi)
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    print(f'Saved: {svg_path}')
    print(f'Saved: {png_path}')


def group_ldi(cells):
    """Per-cell light/dark index of each group ablation index.

    Uses _ldi() from summarize_all_tuning: light / (light + dark), so 0.5 means
    the group matters equally in both conditions, > 0.5 more important in the
    light and < 0.5 more important in the dark.  Cells where either condition
    is missing or non-positive drop out, as they do everywhere else.
    """
    n = len(cells['x'])
    out = np.full((n, len(_GROUP_VAR_ORDER)), np.nan)
    for gi in range(len(_GROUP_VAR_ORDER)):
        light = cells['light_grp'][:, gi]
        dark  = cells['dark_grp'][:, gi]
        out[:, gi] = [_ldi(l, d) for l, d in zip(light, dark)]
    return out


def _bar_with_ci(ax, xpos, vals, width, color, alpha=0.85, hatch=None):
    """Bar of the mean with a 95% CI error bar, drawn relative to 0.5."""
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return None
    mean = float(vals.mean())
    ci95 = float(1.96 * vals.std(ddof=1) / np.sqrt(vals.size))
    ax.bar(xpos, mean - 0.5, width=width, bottom=0.5, yerr=ci95, capsize=2.5,
           color=color, alpha=alpha, hatch=hatch, edgecolor='k', linewidth=0.6,
           error_kw=dict(lw=1.0))
    return mean, ci95, vals.size


def make_group_ldi_bar_svg(cells, out_dir, fname='group_ablation_light_dark_index.svg'):
    """Bar plot of the group-level ablation index expressed as a light/dark index.

    Grouped by anatomical region: one cluster of bars per visual area, one bar
    per variable group within it.  Saved as its own SVG (+ PNG), not a PDF page.
    """
    ldi = group_ldi(cells)
    if not np.any(np.isfinite(ldi)):
        print('  no cells with both light and dark group AI — skipping LDI bar figure')
        return False

    labels = [_GROUP_LABELS.get(g, g) for g in _GROUP_VAR_ORDER]
    areas  = [a for a in _IMP_REGION_ORDER
              if int(np.sum(cells['area_id'] == AREA_IDS.get(a))) >= MIN_CELLS_AREA]
    if not areas:
        print('  no areas with enough cells — skipping LDI bar figure')
        return False

    fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=300, constrained_layout=True)

    print('\n' + '=' * 62)
    print('LEAVE-ONE-GROUP-OUT INDEX AS LIGHT/DARK INDEX  [light / (light + dark)]')
    print(ABL_SENSE_NOTE)
    print('=' * 62)

    n_g   = len(_GROUP_VAR_ORDER)
    width = 0.8 / n_g
    ylo, yhi = 0.15, 0.90

    for ai, area in enumerate(areas):
        sel = cells['area_id'] == AREA_IDS.get(area)
        print(f'  {area}:')
        for gi, grp in enumerate(_GROUP_VAR_ORDER):
            xpos = ai + (gi - (n_g - 1) / 2) * width
            res = _bar_with_ci(ax, xpos, ldi[sel, gi], width * 0.9,
                               _GROUP_COLORS.get(grp, '#888888'))
            if res is None:
                print(f'      {labels[gi]:<15s}: too few cells')
                continue
            mean, ci95, n = res
            ax.text(xpos, ylo + 0.012, f'n={n}', rotation=90, ha='center',
                    va='bottom', fontsize=4.5, color='0.25')
            print(f'      {labels[gi]:<15s}: n={n:5d}, LDI={mean:.3f}, '
                  f'95% CI=[{mean - ci95:.3f}, {mean + ci95:.3f}]')
    print('=' * 62)

    ax.axhline(0.5, color='k', lw=0.8, ls='--', alpha=0.7)
    ax.set_xticks(range(len(areas)))
    ax.set_xticklabels(areas, fontsize=8)
    ax.set_xlim(-0.6, len(areas) - 0.4)
    ax.set_ylim(ylo, yhi)
    ax.set_ylabel('light / (light + dark)\nleave-one-group-out index', fontsize=7)
    # outside the right spine, so they clear the legend and the n= labels
    ax.text(1.012, 0.99, 'more important in the light', transform=ax.transAxes,
            rotation=90, va='top', ha='left', fontsize=5.5, color='0.35')
    ax.text(1.012, 0.01, 'more important in the dark', transform=ax.transAxes,
            rotation=90, va='bottom', ha='left', fontsize=5.5, color='0.35')
    ax.legend(handles=[Patch(facecolor=_GROUP_COLORS.get(g, '#888888'),
                             edgecolor='k', linewidth=0.6, label=lab)
                       for g, lab in zip(_GROUP_VAR_ORDER, labels)],
              frameon=False, fontsize=6, ncol=n_g, loc='upper center')
    ax.set_title('LEAVE-ONE-GROUP-OUT index as a light/dark index by visual area\n'
                 '(0.5 = equally indispensable in both conditions; '
                 'error bars = 95% CI)\n'
                 f'({ABL_SENSE_NOTE})', fontsize=8)

    _save_svg_png(fig, os.path.join(out_dir, fname))
    return True


# Eye vs head split of the reliability variables, used by the collapsed
# figures below.  Same eight variables as REL_VAR_KEYS, partitioned by sensor.
# Matches ffNLE._FEATURE_GROUPS, so the collapsed figures below count the same
# variables the group ablation indices were computed from.
_EYE_VAR_KEYS  = ['theta', 'phi', 'dTheta', 'dPhi']
_HEAD_VAR_KEYS = ['pitch', 'roll', 'gyro_y', 'gyro_x', 'yaw', 'gyro_z']

# Areas used by the collapsed bar figures.  PM is dropped: it carries the
# fewest cells of the five and its bars are the noisiest.
_LDI_BAR_AREAS = [a for a in _IMP_REGION_ORDER if a != 'PM']


def make_pos_vel_eye_head_ldi_svg(cells, out_dir,
                                  fname='pos_vel_eye_head_ldi.svg'):
    """Two panels of the group light/dark index, by area.

    Left: position-only against velocity-only.  Right: eye-only against
    head-only.  The index is light / (light + dark) on the leave-one-group-out
    ablation, so 0.5 means the group is equally indispensable in both
    conditions.  Both panels carry the same claim: darkness moves the code off
    head position and onto velocity.
    """
    ldi = group_ldi(cells)
    if not np.any(np.isfinite(ldi)):
        print('  no cells with both light and dark group AI — skipping')
        return False

    gidx = {g: i for i, g in enumerate(_GROUP_VAR_ORDER)}
    areas = [a for a in _LDI_BAR_AREAS
             if int(np.sum(cells['area_id'] == AREA_IDS.get(a))) >= MIN_CELLS_AREA]
    if not areas:
        print('  no areas with enough cells — skipping')
        return False

    panels = [('position', 'velocity'), ('eyes', 'head')]
    fig, axs = plt.subplots(1, 2, figsize=(5, 3), dpi=300,
                            constrained_layout=True, sharey=True)

    print('\n' + '=' * 70)
    print('GROUP LIGHT/DARK INDEX  light / (light + dark),  PM excluded')
    print(ABL_SENSE_NOTE)
    print('=' * 70)

    ylo, yhi = 0.15, 0.90
    width = 0.8 / 2

    for ax, groups in zip(axs, panels):
        print(f'  {" vs ".join(_GROUP_LABELS[g] for g in groups)}:')
        for ai, area in enumerate(areas):
            sel = cells['area_id'] == AREA_IDS.get(area)
            for gi, grp in enumerate(groups):
                xpos = ai + (gi - 0.5) * width
                res = _bar_with_ci(ax, xpos, ldi[sel, gidx[grp]], width * 0.9,
                                   _GROUP_COLORS.get(grp, '#888888'))
                if res is None:
                    print(f'      {area} {_GROUP_LABELS[grp]:<15s}: too few cells')
                    continue
                mean, ci95, n = res
                ax.text(xpos, ylo + 0.012, f'n={n}', rotation=90, ha='center',
                        va='bottom', fontsize=4.5, color='0.25')
                print(f'      {area:<3s} {_GROUP_LABELS[grp]:<15s}: n={n:5d}  '
                      f'LDI={mean:.3f}  95% CI=[{mean - ci95:.3f}, '
                      f'{mean + ci95:.3f}]')

        ax.axhline(0.5, color='k', lw=0.8, ls='--', alpha=0.7)
        ax.set_xticks(range(len(areas)))
        ax.set_xticklabels(areas, fontsize=8)
        ax.set_xlim(-0.6, len(areas) - 0.4)
        ax.set_ylim(ylo, yhi)
        ax.legend(handles=[Patch(facecolor=_GROUP_COLORS.get(g, '#888888'),
                                 edgecolor='k', linewidth=0.6,
                                 label=_GROUP_LABELS[g]) for g in groups],
                  frameon=False, fontsize=6.5, ncol=2, loc='upper center')

    axs[0].set_ylabel('light / (light + dark)\nleave-one-group-out index',
                      fontsize=7)
    axs[0].set_title('Position vs velocity', fontsize=8)
    axs[1].set_title('Eye vs head', fontsize=8)
    axs[1].text(1.012, 0.99, 'matters more in the light', transform=axs[1].transAxes,
                rotation=90, va='top', ha='left', fontsize=5.5, color='0.35')
    axs[1].text(1.012, 0.01, 'matters more in the dark', transform=axs[1].transAxes,
                rotation=90, va='bottom', ha='left', fontsize=5.5, color='0.35')
    fig.suptitle('Darkness shifts the code from head position toward velocity\n'
                 '0.5 = equally indispensable in light and dark; '
                 'error bars = 95% CI; PM excluded', fontsize=8.5)
    print('=' * 70)

    _save_svg_png(fig, os.path.join(out_dir, fname))
    return True


def make_dark_tuning_eye_vs_head_svg(records, cells, out_dir,
                                     fname='dark_tuning_eye_vs_head.svg'):
    """Two panels collapsing every variable onto its sensor: eye or head.

    Left: the percentage of testable cells with any dark tuning, i.e. reliably
    tuned in the dark to at least one variable of that sensor.  Right: the
    dark - light leave-one-group-out ablation index, which is positive where a
    group matters more in the dark.
    """
    if not records:
        print('  no reliability records — skipping')
        return False

    # ---- left: fraction of cells with any dark tuning ----------------------
    def _any_dark(rec, keys):
        """True/False if any variable is testable, else None."""
        cats = [_classify(rec, v) for v in keys]
        cats = [c for c in cats if c is not None]
        if not cats:
            return None
        return any(c in ('dark only', 'both') for c in cats)

    per = {}
    for rec in records:
        key = (rec['animal'], rec['pos'])
        c = per.setdefault(key, np.zeros(4))     # nEye, kEye, nHead, kHead
        for j, keys in enumerate((_EYE_VAR_KEYS, _HEAD_VAR_KEYS)):
            v = _any_dark(rec, keys)
            if v is None:
                continue
            c[2 * j] += 1
            c[2 * j + 1] += bool(v)
    if not per:
        print('  no testable cells — skipping')
        return False
    counts = np.vstack([per[k] for k in sorted(per)])
    tot = counts.sum(axis=0)

    rng = np.random.default_rng(0)
    idx = rng.integers(0, counts.shape[0], size=(2000, counts.shape[0]))
    boot = counts[idx].sum(axis=1)
    fracs, cis = [], []
    for j in range(2):
        n, k = tot[2 * j], tot[2 * j + 1]
        fracs.append(k / n if n else np.nan)
        bn, bk = boot[:, 2 * j], boot[:, 2 * j + 1]
        b = np.where(bn > 0, bk / np.maximum(bn, 1), np.nan)
        b = b[np.isfinite(b)]
        cis.append(np.percentile(b, [2.5, 97.5]) if b.size >= 20
                   else (np.nan, np.nan))

    # ---- right: dark - light group ablation index --------------------------
    gidx = {g: i for i, g in enumerate(_GROUP_VAR_ORDER)}
    dmls, dcis, dns = [], [], []
    for grp in ('eyes', 'head'):
        d = cells['dark_grp'][:, gidx[grp]] - cells['light_grp'][:, gidx[grp]]
        d = d[np.isfinite(d)]
        dns.append(d.size)
        if d.size < 2:
            dmls.append(np.nan); dcis.append(np.nan); continue
        dmls.append(float(d.mean()))
        dcis.append(float(1.96 * d.std(ddof=1) / np.sqrt(d.size)))

    fig, axs = plt.subplots(1, 2, figsize=(5,3), dpi=300,
                            constrained_layout=True)
    cols = [_GROUP_COLORS['eyes'], _GROUP_COLORS['head']]
    names = ['Eye', 'Head']

    print('\n' + '=' * 70)
    print('DARK TUNING COLLAPSED TO SENSOR (eye vs head)')
    print('=' * 70)

    ax = axs[0]
    for i in range(2):
        lo, hi = cis[i]
        ax.bar(i, fracs[i] * 100, width=0.6, color=cols[i], edgecolor='k',
               linewidth=0.6, alpha=0.85)
        if np.isfinite(lo):
            ax.plot([i, i], [lo * 100, hi * 100], color='k', lw=1.2)
        ax.text(i, 1.0, f'n={int(tot[2 * i])}', ha='center', va='bottom',
                fontsize=5.5, color='0.25')
        print(f'  {names[i]:<5s}: {fracs[i]:.1%} of {int(tot[2 * i])} testable '
              f'cells dark-tuned to >=1 variable  '
              f'95% CI [{lo:.1%}, {hi:.1%}]')
    ax.set_xticks([0, 1]); ax.set_xticklabels(names, fontsize=8)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel('% of testable cells', fontsize=7)
    ax.set_title('Cells with any dark tuning\n(reliably tuned in the dark to '
                 '≥1 variable)', fontsize=8)

    ax = axs[1]
    for i, grp in enumerate(('eyes', 'head')):
        if not np.isfinite(dmls[i]):
            continue
        ax.bar(i, dmls[i], width=0.6, yerr=dcis[i], capsize=3, color=cols[i],
               edgecolor='k', linewidth=0.6, alpha=0.85,
               error_kw=dict(lw=1.0))
        print(f'  {names[i]:<5s}: dark - light group ablation index = '
              f'{dmls[i]:+.4f} +/- {dcis[i]:.4f} (95% CI, n={dns[i]})')
    ax.axhline(0.0, color='k', lw=0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(names, fontsize=8)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel('dark − light\nleave-one-group-out index', fontsize=7)
    ax.set_title('Which sensor the model leans on in the dark\n'
                 '(> 0 = more indispensable in the dark)', fontsize=8)
    _lim = max(abs(np.nanmin(dmls)), abs(np.nanmax(dmls))) * 2.2
    if np.isfinite(_lim) and _lim > 0:
        ax.set_ylim(-_lim, _lim)
    print('=' * 70)

    _save_svg_png(fig, os.path.join(out_dir, fname))
    return True


def make_overview_page(pdf, cells, labeled_array, contours):
    shape = labeled_array.shape
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 4.6), dpi=200)

    ax = axs[0]
    ax.imshow(labeled_array > 1, cmap='binary', alpha=0.08)
    _draw_outlines(ax, contours)
    for area in _IMP_REGION_ORDER:
        sel = cells['area_id'] == AREA_IDS.get(area)
        if not np.any(sel):
            continue
        ax.scatter(cells['x'][sel], cells['y'][sel], s=SCATTER_SIZE,
                   c=COLORS.get(area, '#888888'), linewidths=0, alpha=0.8,
                   label=f'{area} (n={int(sel.sum())})')
    unassigned = ~np.isin(cells['area_id'], [AREA_IDS[a] for a in _IMP_REGION_ORDER])
    if np.any(unassigned):
        ax.scatter(cells['x'][unassigned], cells['y'][unassigned], s=3,
                   c='#BBBBBB', linewidths=0, alpha=0.5)
    ax.legend(frameon=False, fontsize=6, loc='lower right', markerscale=1.5)
    _finish_map_ax(ax, shape, f'cells by area (n = {len(cells["x"])})')

    ax = axs[1]
    r2 = cells['full_r2']
    norm = mcolors.Normalize(vmin=0, vmax=_robust_vmax(r2, 95, 0.1))
    ax.imshow(labeled_array > 1, cmap='binary', alpha=0.08)
    _draw_outlines(ax, contours)
    sc = ax.scatter(cells['x'], cells['y'], s=SCATTER_SIZE, c=r2,
                    cmap=cm.viridis, norm=norm, linewidths=0)
    fig.colorbar(sc, ax=ax, label='full model $R^2$', fraction=0.045, shrink=0.75)
    _finish_map_ax(ax, shape, 'full model $R^2$')

    fig.suptitle('GLM cells on the reference VFS map', fontsize=9)
    pdf.savefig(fig)
    plt.close(fig)


def make_variable_importance_pages(pdf, cells, labeled_array, contours):
    n = 0
    for var in MAP_VAR_ORDER:
        if var not in _IMP_VAR_ORDER:
            continue
        vi = _IMP_VAR_ORDER.index(var)
        label = MAP_VAR_LABELS.get(var, var)
        for cond, key in [('light', 'light_imp'), ('dark', 'dark_imp')]:
            ok = make_map_page(
                pdf, cells, cells[key][:, vi], labeled_array, contours,
                title=f'{label} — leave-one-out importance, {cond}\n'
                      f'({ABL_SENSE_NOTE})',
                cbar_label=f'{label} leave-one-out ablation index ({cond}): '
                           f'fraction of signal lost',
                cmap=cm.plasma, norm=IMP_NORM)
            n += int(ok)
    return n


def make_group_importance_pages(pdf, cells, labeled_array, contours):
    n = 0
    for gi, grp in enumerate(_GROUP_VAR_ORDER):
        label = _GROUP_LABELS.get(grp, grp)
        for cond, key in [('light', 'light_grp'), ('dark', 'dark_grp')]:
            ok = make_map_page(
                pdf, cells, cells[key][:, gi], labeled_array, contours,
                title=f'{label} group — LEAVE-ONE-GROUP-OUT importance, {cond}\n'
                      f'({ABL_SENSE_NOTE})',
                cbar_label=f'{label} leave-one-group-out index ({cond}): '
                           f'fraction of signal lost',
                cmap=cm.plasma, norm=IMP_NORM)
            n += int(ok)
    return n


def make_variable_diff_pages(pdf, cells, labeled_array, contours):
    n = 0
    for var in MAP_VAR_ORDER:
        if var not in _IMP_VAR_ORDER:
            continue
        vi = _IMP_VAR_ORDER.index(var)
        label = MAP_VAR_LABELS.get(var, var)
        diff = cells['dark_imp'][:, vi] - cells['light_imp'][:, vi]
        ok = make_map_page(
            pdf, cells, diff, labeled_array, contours,
            title=f'{label} — dark $-$ light leave-one-out ablation index '
                  f'(red = less dispensable in the dark)\n'
                  f'({ABL_SENSE_NOTE})',
            cbar_label=f'{label} leave-one-out index: dark $-$ light',
            cmap=cm.RdBu_r, norm=DIFF_NORM, cbar_extend='both')
        n += int(ok)
    return n


def make_group_diff_pages(pdf, cells, labeled_array, contours):
    n = 0
    for gi, grp in enumerate(_GROUP_VAR_ORDER):
        label = _GROUP_LABELS.get(grp, grp)
        diff = cells['dark_grp'][:, gi] - cells['light_grp'][:, gi]
        ok = make_map_page(
            pdf, cells, diff, labeled_array, contours,
            title=f'{label} group — dark $-$ light LEAVE-ONE-GROUP-OUT index '
                  f'(red = less dispensable in the dark)\n'
                  f'({ABL_SENSE_NOTE})',
            cbar_label=f'{label} leave-one-group-out index: dark $-$ light',
            cmap=cm.RdBu_r, norm=DIFF_NORM, cbar_extend='both')
        n += int(ok)
    return n


# ----------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description='Cortical maps of GLM variable importance and its '
                    'dark - light difference.')
    parser.add_argument('--pooled_glm', default=DEFAULT_POOLED_GLM,
                        help='GLM pooled h5 (same file used by summarize_all_tuning.py)')
    parser.add_argument('--contours', default=DEFAULT_CONTOURS,
                        help='reference-VFS area contours (json)')
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--pdf_name', default=DEFAULT_PDF_NAME)
    parser.add_argument('--pooled', default=DEFAULT_POOLED,
                        help='pooled h5 used to map recordings to visual areas')
    parser.add_argument('--base_dir', default=DEFAULT_BASE,
                        help='root to search for recording preproc.h5 files')
    parser.add_argument('--skip_reliability', action='store_true',
                        help='skip the block-split reliability analysis')
    parser.add_argument('--recompute_reliability', action='store_true',
                        help='ignore the cached reliability records and redo them')
    args = parser.parse_args()

    args.pooled_glm = resolve_data_path(args.pooled_glm)
    args.pooled     = resolve_data_path(args.pooled)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'Pooled GLM dataset : {args.pooled_glm}')
    print(f'Reference contours : {args.contours}')

    if not os.path.exists(args.pooled_glm):
        print(f'GLM pooled file not found: {args.pooled_glm} — exiting.')
        return

    cells = load_importance_maps(args.pooled_glm)
    if cells is None:
        print('No importance data found. Exiting.')
        return
    cells = apply_r2_filter(cells)
    if len(cells['x']) == 0:
        print('No cells survived the R^2 filter. Exiting.')
        return

    contours = load_ref_contours(args.contours)
    shape = tuple(int(s) for s in cells['ref_shape'])
    labeled_array = build_labeled_array(contours, shape)
    print(f'Reference VFS shape: {shape}, areas: {sorted(contours)}')

    print('\n  group light/dark index bar plot ...')
    make_group_ldi_bar_svg(cells, args.out_dir)

    print('\n  position/velocity and eye/head light-dark index ...')
    make_pos_vel_eye_head_ldi_svg(cells, args.out_dir)

    if not args.skip_reliability:
        base_dir = resolve_data_path(args.base_dir)
        cache = os.path.join(args.out_dir, 'block_reliability.pkl')
        print('\n  block-split tuning reliability (light / dark) ...')
        rel_records = load_or_compute_block_reliability(
            base_dir, resolve_data_path(args.pooled), cache,
            recompute=args.recompute_reliability)
        if rel_records:
            annotate_glm_membership(rel_records, cells)
            make_tuning_class_by_area_svg(rel_records, args.out_dir)
            make_preserved_inverted_by_area_svg(rel_records, args.out_dir)
            print('\n  light/dark tuning overlap ratio ...')
            make_light_dark_overlap_svg(rel_records, args.out_dir)
            # Same analysis, values only: no chance-band shading, no verdict
            # colouring, y capped at 2.0 so the points are readable.
            make_light_dark_overlap_svg(
                rel_records, args.out_dir,
                fname='light_dark_overlap_ratio_values.svg',
                show_verdicts=False, ymax=2.0,
                bottom_title='Almost no cell is tuned in both conditions: '
                             'observed double-tuned cells vs the number\n'
                             'independence predicts (J = Jaccard, the fraction '
                             'of all tuned cells that are tuned in both)')

            print('\n  dark tuning collapsed to eye vs head ...')
            make_dark_tuning_eye_vs_head_svg(rel_records, cells, args.out_dir)

            glm_only = [r for r in rel_records if r.get('in_glm')]
            if len(glm_only) >= REL_MIN_CELLS_AREA:
                print('\n  same analysis restricted to encoding-model cells ...')
                make_tuning_class_by_area_svg(
                    glm_only, args.out_dir,
                    fname='tuning_light_dark_class_by_area_glm_cells.svg')
                make_preserved_inverted_by_area_svg(
                    glm_only, args.out_dir,
                    fname='both_tuned_preserved_inverted_by_area_glm_cells.svg')
                make_light_dark_overlap_svg(
                    glm_only, args.out_dir,
                    fname='light_dark_overlap_ratio_glm_cells.svg')
        else:
            print('  no block-reliability records — skipping those figures.')

    pdf_path = os.path.join(args.out_dir, args.pdf_name)
    print(f'\nWriting PDF: {pdf_path}')

    n_pages = 0
    with PdfPages(pdf_path) as pdf:
        make_overview_page(pdf, cells, labeled_array, contours)
        n_pages += 1

        print('  per-variable importance maps ...')
        n_pages += make_variable_importance_pages(pdf, cells, labeled_array, contours)

        print('  group importance maps ...')
        n_pages += make_group_importance_pages(pdf, cells, labeled_array, contours)

        print('  per-variable dark - light maps ...')
        n_pages += make_variable_diff_pages(pdf, cells, labeled_array, contours)

        print('  group dark - light maps ...')
        n_pages += make_group_diff_pages(pdf, cells, labeled_array, contours)

    print(f'Done. {n_pages} pages -> {pdf_path}')


if __name__ == '__main__':
    main()
