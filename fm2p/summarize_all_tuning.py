

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

# If True, cells whose full ffNLE model R^2 (held-out, full-feature model)
# falls at or below R2_THRESHOLD are dropped from `records` before any of the
# ablation-index figures are built -- a poorly-predicted cell's ablation
# index isn't meaningful. Set to False to restore the original (unfiltered)
# behavior. R2_THRESHOLD = 0 is the principled floor: R^2 <= 0 means the
# model does no better than predicting the mean firing rate, i.e. no real
# signal was captured for that cell.
APPLY_R2_THRESHOLD = True
R2_THRESHOLD       = 0.025

# If True, run the light/dark occupancy analysis (collect_occupancy_pvalues,
# collect_occupancy_sample_traces, collect_occupancy_speed_sample_traces and
# their figures). The per-recording permutation tests in
# collect_occupancy_pvalues are slow (~minutes, one ProcessPoolExecutor
# worker per recording) -- set to False to skip all of it once you already
# have those figures and are iterating on something else.
RUN_OCCUPANCY_ANALYSIS = False

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

# Angular-velocity counterparts of the four non-yaw position variables above,
# in the same order (theta<->dTheta, phi<->dPhi, pitch<->gyro_y, roll<->gyro_x).
SPEED_VARIABLES = [
    dict(name='dTheta', label=r'$\dot{\theta}$ (eye horiz. speed)'),
    dict(name='dPhi',   label=r'$\dot{\phi}$ (eye vert. speed)'),
    dict(name='gyro_y', label='Pitch speed'),
    dict(name='gyro_x', label='Roll speed'),
]
SPEED_VAR_NAMES = [v['name'] for v in SPEED_VARIABLES]

# Raw per-frame trace key (in *preproc.h5, sibling to eyehead_revcorrs_v06.h5)
# for each tracked behavior variable -- used only for the light/dark
# occupancy comparison, which needs the actual frame-by-frame values rather
# than per-cell tuning curves.
_OCC_PREPROC_GLOB = '*DMM*fm*preproc.h5'
_OCC_VAR_KEYS = {
    'theta': 'theta_interp',
    'phi':   'phi_interp',
    'pitch': 'pitch_twop_interp',
    'roll':  'roll_twop_interp',
    'yaw':   'head_yaw_deg',   # one sample longer than ltdk_state_vec -- truncated to match
}

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


MIN_PCT_RUNNING = 50.0  # % of frames with speed > _OCC_SPEED_THRESH required in BOTH
# light and dark for a recording to be kept. Defined here but evaluated against
# _OCC_SPEED_THRESH/_OCC_PREPROC_GLOB/_pct_moving_by_condition, which live further
# down in the occupancy section -- fine, since collect_data() (and this helper)
# only ever run from main(), after the whole module has loaded.
# Set empirically: across all 57 recordings, %-time-running ranges 28-99% in
# light and 67-98% in dark -- far higher than a naive "5%" guess would suggest,
# so a low threshold would filter nothing. 30% sits just above the single clear
# outlier (28.1% light) and below the rest of the pack (next-lowest is 38.7%).


def _recording_pct_running(rcf, thresh=2.0):
    """% of frames with speed > thresh, separately for light and dark, from
    the *preproc.h5 sibling to revcorr file `rcf`. Returns (pct_light,
    pct_dark); either may be NaN if speed/ltdk data is missing."""
    hits = find(_OCC_PREPROC_GLOB, os.path.dirname(rcf))
    if not hits:
        return np.nan, np.nan
    try:
        with h5py.File(sorted(hits)[0], 'r') as f:
            if 'ltdk_state_vec' not in f or 'speed' not in f:
                return np.nan, np.nan
            ltdk = f['ltdk_state_vec'][()].astype(bool)
            n = len(ltdk)
            pm = _pct_moving_by_condition(f, n, ltdk, thresh=thresh)
            if pm is None:
                return np.nan, np.nan
            return pm['light'], pm['dark']
    except Exception as e:
        print(f'  pct-running check failed for {rcf}: {e}')
        return np.nan, np.nan


def collect_data(pooled_path: str, base_dir: str) -> list:
    """Load both light and dark tuning data in one pass. Recordings where
    the animal spent too little time running (speed > _OCC_SPEED_THRESH)
    in EITHER light or dark are dropped entirely (see MIN_PCT_RUNNING) --
    behavior-locked tuning estimated from a condition with almost no
    locomotion isn't trustworthy, whichever condition it is."""
    pooled_lookup = _build_pooled_lookup(pooled_path)
    revcorr_files = find('eyehead_revcorrs_v06.h5', base_dir)
    print(f'Found {len(revcorr_files)} eyehead_revcorrs_v06.h5 files.')

    all_cells = []
    n_dropped_recordings = 0
    dropped_area_counts = {a: 0 for a in REGION_ORDER}

    for rcf in sorted(revcorr_files):
        try:
            with h5py.File(rcf, 'r') as f:
                if not any(f'{v}_l_rel' in f for v in VAR_NAMES + SPEED_VAR_NAMES):
                    continue

                n_cells_f = None
                var_data  = {}

                for vname in VAR_NAMES + SPEED_VAR_NAMES:
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

        pct_light, pct_dark = _recording_pct_running(rcf, thresh=_OCC_SPEED_THRESH)
        if (not np.isfinite(pct_light) or pct_light < MIN_PCT_RUNNING or
                not np.isfinite(pct_dark) or pct_dark < MIN_PCT_RUNNING):
            print(f'    DROPPED (low % running): light={pct_light:.1f}%, dark={pct_dark:.1f}% '
                  f'(need >= {MIN_PCT_RUNNING}% in both)')
            n_dropped_recordings += 1
            for a in named:
                dropped_area_counts[a] += 1
            continue

        for ci in range(n_cells):
            area_id = int(va_ids[ci])
            if area_id not in ID_TO_NAME:
                continue
            area = ID_TO_NAME[area_id]

            cell = dict(animal=animal, pos=pos, ci=ci, area=area, area_id=area_id)

            for vname in VAR_NAMES + SPEED_VAR_NAMES:
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
    print(f'\nRecordings dropped for low % time running (< {MIN_PCT_RUNNING}% in light or dark): '
          f'{n_dropped_recordings}')
    for a in REGION_ORDER:
        if dropped_area_counts[a] > 0:
            print(f'  {a}: {dropped_area_counts[a]} recordings dropped')
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


_OCC_PERM_N     = 2000   # permutations per recording x variable test
_OCC_PERM_MAX_N = 5000   # subsample cap per condition (KS is O(n log n) per permutation)
_OCC_MIN_N      = 20     # minimum frames required in each condition to run the test


def _permutation_test_ks(x, y, n_perm=_OCC_PERM_N, max_n=_OCC_PERM_MAX_N, seed=0):
    """Permutation test for a distributional difference between x and y.
    Statistic = KS D (max abs. difference between empirical CDFs); its null
    distribution is built by reshuffling the pooled (x, y) values into two
    groups of the original sizes `n_perm` times, rather than reading off
    scipy's asymptotic KS p-value -- that formula assumes iid samples, a
    poor fit for autocorrelated per-frame behavior traces, and is trivially
    significant at pooled sample sizes (1e5-1e6 frames) regardless of effect
    size. Both samples are subsampled to `max_n` first since recomputing KS D
    per permutation is the bottleneck.
    Returns (observed D, two-sided permutation p-value)."""
    from scipy import stats
    rng = np.random.default_rng(seed)
    if len(x) > max_n:
        x = rng.choice(x, max_n, replace=False)
    if len(y) > max_n:
        y = rng.choice(y, max_n, replace=False)

    obs = float(stats.ks_2samp(x, y).statistic)
    pooled = np.concatenate([x, y])
    n_x = len(x)
    null = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(pooled)
        null[i] = stats.ks_2samp(pooled[:n_x], pooled[n_x:]).statistic
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return obs, p


def _process_occupancy_file(pf, n_perm, max_n, min_n):
    """Worker for collect_occupancy_pvalues: runs every behavior variable's
    permutation test for one recording. Submitted one-per-process (not
    batched) so each recording's tests -- a CPU-bound, pure-Python-loop
    bottleneck -- run on their own core instead of serializing behind a
    single core/GIL. Must stay at module level (not nested) so it can be
    pickled and shipped to worker processes.
    Returns {var_name: dict(rec=path, n_light=, n_dark=, D=, p=)} (only for
    variables with enough light/dark frames in this recording)."""
    out = {}
    try:
        with h5py.File(pf, 'r') as f:
            if 'ltdk_state_vec' not in f:
                return out
            ltdk = f['ltdk_state_vec'][()].astype(bool)
            n = len(ltdk)
            for vname, key in _OCC_VAR_KEYS.items():
                if key not in f:
                    continue
                vals = f[key][()].astype(float)[:n]
                if len(vals) != n:
                    continue
                ok = np.isfinite(vals)
                lv = vals[ok & ltdk]
                dv = vals[ok & ~ltdk]
                if len(lv) < min_n or len(dv) < min_n:
                    continue
                D, p = _permutation_test_ks(lv, dv, n_perm=n_perm, max_n=max_n)
                out[vname] = dict(rec=pf, n_light=len(lv), n_dark=len(dv), D=D, p=p)
    except Exception as e:
        print(f'  Occupancy read error {pf}: {e}')
    return out


def collect_occupancy_pvalues(base_dir: str, n_perm=_OCC_PERM_N,
                               max_n=_OCC_PERM_MAX_N, min_n=_OCC_MIN_N,
                               n_workers=None) -> dict:
    """Run the light-vs-dark permutation KS test (_permutation_test_ks)
    separately for each recording's *preproc.h5 (sibling to
    eyehead_revcorrs_v06.h5), for each tracked behavior variable. Testing
    per-recording rather than pooling everything into one giant light/dark
    comparison avoids a single pooled test -- trivially significant at
    N~1e5-1e6 frames -- masking real animal-to-animal and
    recording-to-recording heterogeneity in whether light/dark actually
    changes the range of behavior sampled.

    Recordings are distributed one-per-process across a ProcessPoolExecutor
    (default: one worker per CPU core) -- each recording's full set of
    per-variable permutation tests runs to completion on a single core,
    rather than threading (which wouldn't parallelize this CPU-bound,
    GIL-bound Python loop at all).

    Returns {var_name: [dict(rec=path, n_light=, n_dark=, D=, p=), ...]}."""
    preproc_files = sorted(find(_OCC_PREPROC_GLOB, base_dir))
    print(f'Found {len(preproc_files)} preproc.h5 files for occupancy.')

    results = {vname: [] for vname in _OCC_VAR_KEYS}
    n_workers = n_workers or os.cpu_count()

    with cf.ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_process_occupancy_file, pf, n_perm, max_n, min_n)
                for pf in preproc_files]
        for fut in cf.as_completed(futs):
            for vname, rec in fut.result().items():
                results[vname].append(rec)

    for vname, recs in results.items():
        print(f'  Occupancy per-recording tests for {vname}: {len(recs)} recordings tested.')
    return results


def make_occupancy_pvalue_histogram_svg(occ_pvals, out_dir, alpha=0.05):
    """2-row figure, one column per behavior variable: top row = histogram
    of per-recording light-vs-dark occupancy permutation-test p-values,
    bottom row = histogram of the matching effect-size (KS D) per recording.
    Each p-value/D comes from a single recording's own light/dark frames, so
    this shows recording-to-recording heterogeneity that a single pooled
    test can't -- the top row directly answers, per variable, how many
    individual recordings show no evidence of a light/dark occupancy
    difference (p > alpha); the bottom row shows whether the detected
    differences are actually large (D close to 1) or just barely detectable
    (D close to 0) thanks to large per-recording N."""
    vars_present = [v for v in VARIABLES if len(occ_pvals.get(v['name'], [])) > 0]
    if not vars_present:
        print('No per-recording occupancy p-values — skipping.')
        return

    fig, axes = plt.subplots(2, len(vars_present),
                              figsize=_scaled(len(vars_present) * 2.4 + 0.5, 5.6),
                              constrained_layout=True, squeeze=False)

    print('\n' + '=' * 60)
    print(f'PER-RECORDING LIGHT VS DARK OCCUPANCY P-VALUES (alpha={alpha})')
    print('=' * 60)

    for vi, vspec in enumerate(vars_present):
        vname = vspec['name']
        recs  = occ_pvals[vname]
        pvals = np.array([r['p'] for r in recs])
        dvals = np.array([r['D'] for r in recs])
        n_total   = len(pvals)
        n_no_diff = int(np.sum(pvals > alpha))
        pct_no_diff = 100.0 * n_no_diff / n_total if n_total else float('nan')

        print(f'  {vspec["label"]}: {n_no_diff}/{n_total} recordings '
              f'({pct_no_diff:.0f}%) show no evidence of a difference (p > {alpha}), '
              f'median D={np.median(dvals) if n_total else float("nan"):.3f}')

        ax_p = axes[0, vi]
        ax_p.hist(pvals, bins=np.linspace(0, 1, 21), color='#888888', alpha=0.85,
                  edgecolor='k', linewidth=0.4)
        ax_p.axvline(alpha, color='r', lw=1.0, ls='--', zorder=5)
        ax_p.text(0.97, 0.97, f'n={n_total}\n{n_no_diff} (p>{alpha})',
                  ha='right', va='top', transform=ax_p.transAxes, fontsize=6, color='0.2')
        ax_p.set_xlim(0, 1)
        ax_p.set_title(vspec['label'], fontsize=8)
        ax_p.set_xlabel('Permutation p-value', fontsize=7)

        ax_d = axes[1, vi]
        ax_d.hist(dvals, bins=np.linspace(0, 1, 21), color='#4C72B0', alpha=0.85,
                  edgecolor='k', linewidth=0.4)
        ax_d.text(0.97, 0.97, f'median D={np.median(dvals):.2f}' if n_total else 'n/a',
                  ha='right', va='top', transform=ax_d.transAxes, fontsize=6, color='0.2')
        ax_d.set_xlim(0, 1)
        ax_d.set_xlabel('KS D (effect size)', fontsize=7)

    axes[0, 0].set_ylabel('Recording count\n(p-value)', fontsize=7)
    axes[1, 0].set_ylabel('Recording count\n(D statistic)', fontsize=7)

    print('=' * 60)

    fig.suptitle(
        'Per-recording light-vs-dark occupancy: permutation p-value (top) and KS D effect size (bottom)\n'
        f'Top dashed line = alpha ({alpha}); bars right of it = no evidence of a difference for that recording',
        fontsize=9)

    path = os.path.join(out_dir, 'occupancy_pvalue_distribution.svg')
    _save_svg_png(fig, path)


_OCC_SAMPLE_N     = 10
_OCC_SPEED_THRESH = 2.0   # cm/s; "running" cutoff, matches compare_light_dark_behavior.py


def _occ_label_from_path(pf):
    fm_dir  = os.path.basename(os.path.dirname(pf))
    pos_dir = os.path.basename(os.path.dirname(os.path.dirname(pf)))
    return f'{pos_dir}/{fm_dir}'


def _pct_moving_by_condition(f, n, ltdk, thresh=_OCC_SPEED_THRESH):
    """% of frames with speed > thresh (cm/s), separately for light and
    dark frames of one recording. Returns {'light': pct, 'dark': pct} or
    None if no 'speed' trace is present. `speed` is occasionally 1 sample
    shorter than ltdk_state_vec/twopT (same off-by-one quirk as several
    other fields in this pipeline, e.g. head_yaw_deg) -- truncated to the
    common length rather than treated as missing."""
    if 'speed' not in f:
        return None
    speed = f['speed'][()].astype(float)
    m = min(len(speed), n, len(ltdk))
    speed = speed[:m]
    ltdk  = ltdk[:m]
    ok = np.isfinite(speed)
    out = {}
    for cond, mask in (('light', ltdk), ('dark', ~ltdk)):
        mm = ok & mask
        out[cond] = 100.0 * np.sum(speed[mm] > thresh) / mm.sum() if mm.sum() else float('nan')
    return out


def _sample_eligible_recordings(base_dir, n_sample, seed):
    """Eligible recordings for the occupancy sample-trace figures
    (collect_occupancy_sample_traces / collect_occupancy_speed_sample_traces):
    must have ltdk_state_vec, all five tracked behavior variables present
    (_OCC_VAR_KEYS), and at least MIN_PCT_RUNNING% of frames with
    speed > _OCC_SPEED_THRESH in BOTH light and dark -- recordings with too
    little active time in either condition are excluded as a sampling option."""
    preproc_files = sorted(find(_OCC_PREPROC_GLOB, base_dir))
    eligible = []
    for pf in preproc_files:
        try:
            with h5py.File(pf, 'r') as f:
                if 'ltdk_state_vec' not in f:
                    continue
                if not all(key in f for key in _OCC_VAR_KEYS.values()):
                    continue
                ltdk = f['ltdk_state_vec'][()].astype(bool)
                n = len(ltdk)
                pct_moving = _pct_moving_by_condition(f, n, ltdk)
                if pct_moving is None:
                    continue
                if (not np.isfinite(pct_moving['light']) or pct_moving['light'] < MIN_PCT_RUNNING or
                        not np.isfinite(pct_moving['dark']) or pct_moving['dark'] < MIN_PCT_RUNNING):
                    continue
                eligible.append(pf)
        except Exception:
            continue
    rng = np.random.default_rng(seed)
    n_sample = min(n_sample, len(eligible))
    return rng.choice(np.array(eligible, dtype=object), size=n_sample, replace=False), len(eligible)


def collect_occupancy_sample_traces(base_dir: str, n_sample=_OCC_SAMPLE_N,
                                     seed=0, min_n=_OCC_MIN_N) -> list:
    """Randomly sample `n_sample` recordings and load their raw per-frame
    light/dark behavior-variable traces, plus % time spent moving
    (speed > _OCC_SPEED_THRESH) per condition -- no permutation test here,
    this is purely for visual inspection of what an individual recording's
    light-vs-dark occupancy shift actually looks like (a companion to the
    summary statistics in make_occupancy_pvalue_histogram_svg).
    Returns [dict(pf=path, label=str, vars={var_name: {'light': arr, 'dark': arr}},
    pct_moving={'light': pct, 'dark': pct})]."""
    sampled, n_eligible = _sample_eligible_recordings(base_dir, n_sample, seed)
    print(f'Sampled {len(sampled)}/{n_eligible} recordings for occupancy trace figure.')

    out = []
    for pf in sampled:
        rec = dict(pf=pf, label=_occ_label_from_path(pf), vars={}, pct_moving=None)
        try:
            with h5py.File(pf, 'r') as f:
                ltdk = f['ltdk_state_vec'][()].astype(bool)
                n = len(ltdk)
                for vname, key in _OCC_VAR_KEYS.items():
                    if key not in f:
                        continue
                    vals = f[key][()].astype(float)[:n]
                    if len(vals) != n:
                        continue
                    ok = np.isfinite(vals)
                    lv = vals[ok & ltdk]
                    dv = vals[ok & ~ltdk]
                    if len(lv) < min_n or len(dv) < min_n:
                        continue
                    rec['vars'][vname] = {'light': lv, 'dark': dv}
                rec['pct_moving'] = _pct_moving_by_condition(f, n, ltdk)
        except Exception as e:
            print(f'  Occupancy sample read error {pf}: {e}')
        out.append(rec)
    return out


def _plot_pct_moving_bar(ax, pct_moving):
    """Shared rightmost-column panel for the occupancy sample-trace grids:
    2 bars (light, dark) showing % of frames with speed > _OCC_SPEED_THRESH."""
    if pct_moving is None or not np.isfinite(pct_moving.get('light', np.nan)) \
            or not np.isfinite(pct_moving.get('dark', np.nan)):
        ax.set_visible(False)
        return
    vals = [pct_moving['light'], pct_moving['dark']]
    ax.bar([0, 1], vals, color=['#E8A838', '#5B7FA6'], edgecolor='k', linewidth=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['L', 'D'], fontsize=5)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=5)
    for xi, v in enumerate(vals):
        ax.text(xi, v + 2, f'{v:.0f}', ha='center', va='bottom', fontsize=5)


def make_occupancy_sample_traces_svg(samples, out_dir, n_bins=20):
    """Grid of overlaid light/dark occupancy histograms: one row per
    randomly sampled recording (see collect_occupancy_sample_traces), one
    column per behavior variable, plus a rightmost column showing % time
    spent moving (speed > _OCC_SPEED_THRESH) as a light-vs-dark bar pair. A
    visual companion to make_occupancy_pvalue_histogram_svg -- that figure
    summarizes the permutation-test p-value/D per recording, this one lets
    you actually see what those numbers correspond to in a given session."""
    vars_present = [v for v in VARIABLES
                    if any(v['name'] in s['vars'] for s in samples)]
    if not samples or not vars_present:
        print('No sampled recordings with occupancy data — skipping.')
        return

    n_rows, n_vars = len(samples), len(vars_present)
    n_cols = n_vars + 1
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=_scaled(n_cols * 2.0 + 0.5, n_rows * 1.5 + 0.5),
                              constrained_layout=True, squeeze=False)

    for ri, s in enumerate(samples):
        for ci, vspec in enumerate(vars_present):
            ax = axes[ri, ci]
            vd = s['vars'].get(vspec['name'])
            if vd is None:
                ax.set_visible(False)
                continue
            lv, dv = vd['light'], vd['dark']
            pooled_vals = np.concatenate([lv, dv])
            lo, hi = np.nanpercentile(pooled_vals, [0.5, 99.5])
            bins = np.linspace(lo, hi, n_bins + 1)

            ax.hist(lv, bins=bins, density=True, color='#E8A838', alpha=0.55,
                    edgecolor='k', linewidth=0.3, label='Light')
            ax.hist(dv, bins=bins, density=True, color='#5B7FA6', alpha=0.55,
                    edgecolor='k', linewidth=0.3, label='Dark')
            ax.tick_params(labelsize=5)
            ax.set_yticks([])
            if ri == 0:
                ax.set_title(vspec['label'], fontsize=8)
            if ci == 0:
                ax.set_ylabel(s['label'], fontsize=5)
            if ri == n_rows - 1:
                ax.set_xlabel(vspec['label'], fontsize=6)

        ax_pm = axes[ri, n_vars]
        _plot_pct_moving_bar(ax_pm, s['pct_moving'])
        if ri == 0:
            ax_pm.set_title(f'% time\n>{_OCC_SPEED_THRESH:.0f} cm/s', fontsize=7)

    axes[0, 0].legend(fontsize=6, loc='upper right')

    fig.suptitle(
        f'Light vs. dark occupancy for {n_rows} randomly sampled recordings\n'
        '(raw per-frame distributions, density-normalized; one row per recording; '
        'rightmost column = % time moving)',
        fontsize=9)

    path = os.path.join(out_dir, 'occupancy_sample_traces.svg')
    _save_svg_png(fig, path)


def _load_speed_trace(f, vname, n):
    """Load one speed-variable trace from an open preproc.h5, aligned to the
    twop/ltdk frame count `n`.

    gyro_x/gyro_y (head angular velocity) are already stored at twop
    resolution ('{vname}_twop_interp') and only exist for IMU recordings.

    dTheta/dPhi (eye angular velocity) are recomputed locally with
    np.gradient on theta_interp/phi_interp (already at twop resolution,
    present in every recording) rather than read from the legacy
    precomputed 'dTheta'/'dPhi'/'eyeT1' fields -- those only exist in a
    subset (28/59) of these recordings' preproc.h5 files. This mirrors
    fm2p/utils/ffNLE.py's load_position_data(), which derives the model's
    own eye-speed feature the same way (np.gradient on the interpolated eye
    trace) rather than trusting the legacy fields, and is why ffNLE could
    run on every recording regardless of whether they were ever computed.
    eyehead_revcorr.py does the analogous thing for the tuning-curve
    pipeline: its own dTheta/dPhi fallback (np.diff-based, not
    np.gradient-based) only fires when the legacy fields are missing from
    a given recording -- so the eyehead_revcorrs_v06.h5 tuning curves are
    likewise already complete for all recordings, with no gap to fix here.

    Returns None if unavailable."""
    if vname in ('gyro_y', 'gyro_x'):
        key = f'{vname}_twop_interp'
        if key not in f:
            return None
        vals = f[key][()].astype(float)
        return vals[:n] if len(vals) >= n else None
    if vname in ('dTheta', 'dPhi'):
        pos_key = 'theta_interp' if vname == 'dTheta' else 'phi_interp'
        if pos_key not in f or 'twopT' not in f:
            return None
        from .utils.helper import interp_short_gaps
        pos   = f[pos_key][()].astype(float)[:n]
        twopT = f['twopT'][()].astype(float)[:n]
        if len(pos) != n or len(twopT) != n:
            return None
        pos = interp_short_gaps(pos)
        return np.gradient(pos, twopT)
    return None


def collect_occupancy_speed_sample_traces(base_dir: str, n_sample=_OCC_SAMPLE_N,
                                           seed=0, min_n=_OCC_MIN_N) -> list:
    """Speed-variable counterpart of collect_occupancy_sample_traces: same
    idea (randomly sample recordings, load raw per-frame light/dark traces
    plus % time moving, no permutation test) but for the four angular-
    velocity variables (dTheta, dPhi, gyro_y/Pitch speed, gyro_x/Roll speed)
    instead of position variables.
    Returns [dict(pf=path, label=str, vars={var_name: {'light': arr, 'dark': arr}},
    pct_moving={'light': pct, 'dark': pct})]."""
    sampled, n_eligible = _sample_eligible_recordings(base_dir, n_sample, seed)
    print(f'Sampled {len(sampled)}/{n_eligible} recordings for speed occupancy trace figure.')

    out = []
    for pf in sampled:
        rec = dict(pf=pf, label=_occ_label_from_path(pf), vars={}, pct_moving=None)
        try:
            with h5py.File(pf, 'r') as f:
                ltdk = f['ltdk_state_vec'][()].astype(bool)
                n = len(ltdk)
                for vspec in SPEED_VARIABLES:
                    vname = vspec['name']
                    vals = _load_speed_trace(f, vname, n)
                    if vals is None or len(vals) != n:
                        continue
                    ok = np.isfinite(vals)
                    lv = vals[ok & ltdk]
                    dv = vals[ok & ~ltdk]
                    if len(lv) < min_n or len(dv) < min_n:
                        continue
                    rec['vars'][vname] = {'light': lv, 'dark': dv}
                rec['pct_moving'] = _pct_moving_by_condition(f, n, ltdk)
        except Exception as e:
            print(f'  Speed occupancy sample read error {pf}: {e}')
        out.append(rec)
    return out


def make_occupancy_speed_sample_traces_svg(samples, out_dir, n_bins=20, pctl=(5, 95)):
    """Speed-variable counterpart of make_occupancy_sample_traces_svg
    (duplicate grid layout, plus the same rightmost % time moving column).
    Uses a much narrower x-range (5th-95th percentile by default, vs. the
    position-variable figure's 0.5th-99.5th) because angular-velocity
    distributions are heavily concentrated near zero with long tails from
    occasional fast saccades/head turns -- the wide range used for position
    variables would compress the whole bulk of the distribution into a
    single bin and hide exactly the small light/dark shifts this figure is
    meant to reveal."""
    vars_present = [v for v in SPEED_VARIABLES
                    if any(v['name'] in s['vars'] for s in samples)]
    if not samples or not vars_present:
        print('No sampled recordings with speed occupancy data — skipping.')
        return

    n_rows, n_vars = len(samples), len(vars_present)
    n_cols = n_vars + 1
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=_scaled(n_cols * 2.0 + 0.5, n_rows * 1.5 + 0.5),
                              constrained_layout=True, squeeze=False)

    for ri, s in enumerate(samples):
        for ci, vspec in enumerate(vars_present):
            ax = axes[ri, ci]
            vd = s['vars'].get(vspec['name'])
            if vd is None:
                ax.set_visible(False)
                continue
            lv, dv = vd['light'], vd['dark']
            pooled_vals = np.concatenate([lv, dv])
            lo, hi = np.nanpercentile(pooled_vals, list(pctl))
            bins = np.linspace(lo, hi, n_bins + 1)

            ax.hist(lv, bins=bins, density=True, color='#E8A838', alpha=0.55,
                    edgecolor='k', linewidth=0.3, label='Light')
            ax.hist(dv, bins=bins, density=True, color='#5B7FA6', alpha=0.55,
                    edgecolor='k', linewidth=0.3, label='Dark')
            ax.set_xlim(lo, hi)
            ax.tick_params(labelsize=5)
            ax.set_yticks([])
            if ri == 0:
                ax.set_title(vspec['label'], fontsize=8)
            if ci == 0:
                ax.set_ylabel(s['label'], fontsize=5)
            if ri == n_rows - 1:
                ax.set_xlabel(vspec['label'], fontsize=6)

        ax_pm = axes[ri, n_vars]
        _plot_pct_moving_bar(ax_pm, s['pct_moving'])
        if ri == 0:
            ax_pm.set_title(f'% time\n>{_OCC_SPEED_THRESH:.0f} cm/s', fontsize=7)

    axes[0, 0].legend(fontsize=6, loc='upper right')

    fig.suptitle(
        f'Light vs. dark occupancy for {n_rows} randomly sampled recordings -- speed variables\n'
        f'(raw per-frame distributions, density-normalized; x-range = {pctl[0]}th-{pctl[1]}th pctile '
        'to reveal small shifts; rightmost column = % time moving)',
        fontsize=9)

    path = os.path.join(out_dir, 'occupancy_sample_traces_speed.svg')
    _save_svg_png(fig, path)


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
    """Heatmap of median LDI: visual areas (rows) x variables (cols)."""
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


def make_overview_mi_ldi_boxstrip_svg(all_cells, out_dir, variables=None, file_suffix=''):
    """Improved companion to make_combined_overview_svg() -- saved as a
    separate file, the original is left unchanged. Violins on CV MI (heavily
    right-skewed, most mass near 0) compress almost all shape information
    into a thin neck near the bottom, and LDI's narrow real-world range made
    its violins look like nearly uniform-width bars top to bottom. This
    version uses a box (median/IQR/5-95th pctile whiskers) with jittered raw
    points underneath: the point cloud shows density directly (no kernel
    bandwidth artifact) and the whisker length communicates skew.

    `variables` defaults to VARS_NO_YAW (position variables); pass
    SPEED_VARIABLES for the angular-velocity counterpart (with a matching
    `file_suffix` so the two don't overwrite each other's output)."""
    if variables is None:
        variables = VARS_NO_YAW
    rng = np.random.default_rng(0)
    n_vars = len(variables)
    panel_w, panel_h = 2.0, 2.5
    fig, axes = plt.subplots(2, n_vars, figsize=_scaled(n_vars * panel_w, 2 * panel_h),
                              constrained_layout=True)

    for vi, vspec in enumerate(variables):
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

    path = os.path.join(out_dir, f'overview_mi_ldi_boxstrip{file_suffix}.svg')
    _save_svg_png(fig, path)


_EXAMPLE_TUNING_MODES = {
    'light': dict(key=lambda mi_l, mi_d: mi_l,
                  suffix='_light', label='highest light MI'),
    'dark':  dict(key=lambda mi_l, mi_d: mi_d,
                  suffix='_dark',  label='highest dark MI'),
    'avg':   dict(key=lambda mi_l, mi_d: (mi_l + mi_d) / 2.0,
                  suffix='_avg',   label='highest mean(light, dark) MI'),
}


def _make_example_tuning_pages(all_cells, out_dir, variables, file_prefix, tick_fmt, xlabel_fmt):
    """Shared implementation behind make_example_tuning_svgs and
    make_example_tuning_speed_svgs. Per-area SVGs, one figure per
    variable-selection mode, each picking the cell (with non-NaN LDI) that
    ranks highest under that mode's criterion — highest light MI, highest
    dark MI, or highest mean of light & dark MI. Titles report both light
    and dark MI. Light = solid line, dark = dashed."""
    for mode in _EXAMPLE_TUNING_MODES.values():
        for area in REGION_ORDER:
            cells_area = [c for c in all_cells if c['area'] == area]
            if len(cells_area) < MIN_CELLS_AREA:
                continue

            n_vars = len(variables)
            fig, axes = plt.subplots(1, n_vars,
                                      figsize=(n_vars * 2.0, 2.5),
                                      constrained_layout=True)
            color = COLORS.get(area, '#888888')
            any_plotted = False

            for vi, vspec in enumerate(variables):
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

                best  = max(candidates,
                            key=lambda c: mode['key'](c[f'{vname}_rel'], c[f'{vname}_rel_dark']))
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

                ax.set_title(f'{vspec["label"]}\nL={mi_l:.2f}  D={mi_d:.2f}  LDI={ldi:.2f}',
                             fontsize=7)
                ax.set_xlabel(xlabel_fmt(vspec['label']), fontsize=6)
                if vi == 0:
                    ax.set_ylabel('Firing rate', fontsize=6)
                ax.tick_params(labelsize=5)

                mid = len(bins) // 2
                ax.set_xticks([bins[0], bins[mid], bins[-1]])
                ax.set_xticklabels([tick_fmt(bins[0]), tick_fmt(bins[mid]), tick_fmt(bins[-1])],
                                    fontsize=5)

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
                f'{area} — cell with {mode["label"]} per variable'
                '  (solid = light · dashed = dark)',
                fontsize=8)
            svg_path = os.path.join(out_dir, f'{file_prefix}_{area}{mode["suffix"]}.svg')
            _save_svg_png(fig, svg_path)


def make_example_tuning_svgs(all_cells, out_dir):
    """Position-variable (theta, phi, pitch, roll; yaw excluded) version of
    _make_example_tuning_pages — see that function for selection criteria."""
    _make_example_tuning_pages(
        all_cells, out_dir, VARS_NO_YAW, 'example_tuning',
        tick_fmt=lambda b: f'{b:.0f}°',
        xlabel_fmt=lambda label: f'{label} (°)',
    )


def make_example_tuning_speed_svgs(all_cells, out_dir):
    """Angular-velocity ('speed') counterpart of make_example_tuning_svgs,
    for the four speed variables: dTheta, dPhi, gyro_y (pitch speed),
    gyro_x (roll speed). See _make_example_tuning_pages for selection
    criteria."""
    _make_example_tuning_pages(
        all_cells, out_dir, SPEED_VARIABLES, 'example_tuning_speed',
        tick_fmt=lambda b: f'{b:.0f}',
        xlabel_fmt=lambda label: f'{label} (°/s)',
    )


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

            full_r2_arr = np.atleast_1d(np.asarray(model['full_r2'], dtype=float))
            n = len(full_r2_arr)

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
                    full_r2=float(full_r2_arr[ci]),
                ))

    print(f'Importance cells loaded: {len(records)}')
    return records


def _load_cross_condition_generalization(pooled_glm_path, example_min_cells=50):
    """Does a full (all-8-variable) model trained only on LIGHT frames
    predict held-out DARK frames, and vice versa? The pooled GLM h5 already
    has both directions computed per cell: full_trainLight_testDark_* and
    full_trainDark_testLight_* (r2, corrs, y_hat, y_true), alongside the
    same-condition full_trainLight_testLight_*/full_trainDark_testDark_*
    used elsewhere in this script for ablation index.

    Collects per-cell scalars (correlation, R^2) for every recording that
    has both cross-condition directions (for the all-recordings summary
    panel), plus the full y_hat/y_true arrays for one example recording --
    whichever eligible recording (n_cells >= example_min_cells) has the
    highest mean R^2 averaged across both directions -- for the
    example-session scatter panels.

    Returns (rows, example):
      rows: [dict(animal, pos, ci, area, r2_l2d, corr_l2d, r2_d2l, corr_d2l)]
      example: dict(animal, pos, light_to_dark=dict(y_true, y_hat, r2, ci),
                     dark_to_light=dict(...)) or None if nothing qualified."""
    from .utils.files import read_h5
    pooled = read_h5(pooled_glm_path)

    rows = []
    best_example = None  # (mean_r2_both, animal, pos, model)

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
            need = ['full_trainLight_testDark_r2', 'full_trainLight_testDark_corrs',
                    'full_trainDark_testLight_r2', 'full_trainDark_testLight_corrs']
            if not all(k in model for k in need):
                continue

            r2_l2d   = np.atleast_1d(np.asarray(model['full_trainLight_testDark_r2'], dtype=float))
            corr_l2d = np.atleast_1d(np.asarray(model['full_trainLight_testDark_corrs'], dtype=float))
            r2_d2l   = np.atleast_1d(np.asarray(model['full_trainDark_testLight_r2'], dtype=float))
            corr_d2l = np.atleast_1d(np.asarray(model['full_trainDark_testLight_corrs'], dtype=float))
            n = len(r2_l2d)

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
                rows.append(dict(animal=animal, pos=pos, ci=ci, area=name,
                                  r2_l2d=float(r2_l2d[ci]), corr_l2d=float(corr_l2d[ci]),
                                  r2_d2l=float(r2_d2l[ci]), corr_d2l=float(corr_d2l[ci])))

            if n >= example_min_cells:
                mean_r2_both = float(np.nanmean(r2_l2d) + np.nanmean(r2_d2l)) / 2.0
                if best_example is None or mean_r2_both > best_example[0]:
                    best_example = (mean_r2_both, animal, pos, model)

    example = None
    if best_example is not None:
        _, animal, pos, model = best_example
        r2_l2d = np.atleast_1d(np.asarray(model['full_trainLight_testDark_r2'], dtype=float))
        r2_d2l = np.atleast_1d(np.asarray(model['full_trainDark_testLight_r2'], dtype=float))
        ci_l2d = int(np.nanargmax(np.where(np.isfinite(r2_l2d), r2_l2d, -np.inf)))
        ci_d2l = int(np.nanargmax(np.where(np.isfinite(r2_d2l), r2_d2l, -np.inf)))
        example = dict(
            animal=animal, pos=pos,
            light_to_dark=dict(
                y_true=np.asarray(model['full_trainLight_testDark_y_true'])[:, ci_l2d],
                y_hat=np.asarray(model['full_trainLight_testDark_y_hat'])[:, ci_l2d],
                r2=float(r2_l2d[ci_l2d]), ci=ci_l2d,
            ),
            dark_to_light=dict(
                y_true=np.asarray(model['full_trainDark_testLight_y_true'])[:, ci_d2l],
                y_hat=np.asarray(model['full_trainDark_testLight_y_hat'])[:, ci_d2l],
                r2=float(r2_d2l[ci_d2l]), ci=ci_d2l,
            ),
        )

    n_recs = len({(r['animal'], r['pos']) for r in rows})
    print(f'Cross-condition generalization: {len(rows)} cells from {n_recs} recordings.')
    return rows, example


def make_cross_condition_generalization_svg(rows, example, out_dir):
    """3-panel figure: does a model trained on LIGHT data generalize to
    DARK frames, and vice versa? Panels 1-2: one example recording's
    best-R^2 cell for each direction, true-vs-predicted scatter (light-fit
    model evaluated on held-out dark frames; dark-fit model evaluated on
    held-out light frames). Panel 3: per-recording mean cross-condition
    correlation, light->dark (x) vs. dark->light (y), colored by area --
    summarizes whether generalization is symmetric and how it varies
    across the dataset, rather than just showing one example."""
    if not rows:
        print('No cross-condition generalization data -- skipping.')
        return

    fig, axes = plt.subplots(1, 3, figsize=_scaled(11.5, 3.8), constrained_layout=True)

    for ax, key, title in [
        (axes[0], 'light_to_dark', 'Light-fit model -> predict DARK'),
        (axes[1], 'dark_to_light', 'Dark-fit model -> predict LIGHT'),
    ]:
        if example is None:
            ax.set_visible(False)
            continue
        d = example[key]
        ok = np.isfinite(d['y_true']) & np.isfinite(d['y_hat'])
        ax.scatter(d['y_true'][ok], d['y_hat'][ok], s=4, alpha=0.3, color='0.3', linewidths=0)
        lims = [min(np.min(d['y_true'][ok]), np.min(d['y_hat'][ok])),
                max(np.max(d['y_true'][ok]), np.max(d['y_hat'][ok]))]
        ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.6, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('True response', fontsize=7)
        ax.set_ylabel('Predicted response', fontsize=7)
        ax.set_title(f'{title}\n{example["animal"]}/{example["pos"]} cell {d["ci"]}, '
                     f'R$^2$={d["r2"]:.2f}', fontsize=7.5)

    ax3 = axes[2]
    by_rec = {}
    for r in rows:
        key = (r['animal'], r['pos'])
        by_rec.setdefault(key, {'corr_l2d': [], 'corr_d2l': [], 'area': r['area']})
        by_rec[key]['corr_l2d'].append(r['corr_l2d'])
        by_rec[key]['corr_d2l'].append(r['corr_d2l'])

    print('\nCross-condition generalization, per-recording mean correlation:')
    for (animal, pos), d in by_rec.items():
        x = float(np.nanmean(d['corr_l2d']))
        y = float(np.nanmean(d['corr_d2l']))
        color = COLORS.get(d['area'], '#888888')
        ax3.scatter(x, y, s=22, color=color, edgecolors='k', linewidths=0.5, alpha=0.85, zorder=3)
        print(f'  {animal}/{pos} ({d["area"]}): light->dark corr={x:.2f}, dark->light corr={y:.2f}')

    lims = [-0.2, 1.0]
    ax3.plot(lims, lims, 'k--', lw=0.8, alpha=0.6, zorder=0)
    ax3.axhline(0, color='0.7', lw=0.6)
    ax3.axvline(0, color='0.7', lw=0.6)
    ax3.set_xlim(lims)
    ax3.set_ylim(lims)
    ax3.set_xlabel('Light-fit -> dark, mean corr (per recording)', fontsize=7)
    ax3.set_ylabel('Dark-fit -> light, mean corr (per recording)', fontsize=7)
    ax3.set_title(f'All recordings (n={len(by_rec)})', fontsize=7.5)

    area_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS.get(a, '#888888'),
                            markeredgecolor='k', markersize=6, label=a) for a in _IMP_REGION_ORDER]
    ax3.legend(handles=area_handles, fontsize=5.5, loc='upper left', framealpha=0.7)


    axes[0].set_xlim([-2, 5])
    axes[0].set_ylim([-2, 5])
    axes[0].set_box_aspect(1)
    axes[1].set_xlim([-2, 5])
    axes[1].set_ylim([-2, 5])
    axes[1].set_box_aspect(1)
    ax3.set_xlim([-0.1, 0.2])
    ax3.set_ylim([-0.1, 0.2])
    ax3.set_box_aspect(1)

    fig.suptitle('Cross-condition generalization: does a model fit in one\n'
                 'lighting condition predict the other?', fontsize=9)

    path = os.path.join(out_dir, 'cross_condition_generalization.svg')
    _save_svg_png(fig, path)


def make_importance_svg(out_dir, pooled_glm_path=DEFAULT_POOLED_GLM, records=None):
    """2x4 SVG: permutation importance by visual area.
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


def make_r2_histogram_svg(records, out_dir, threshold=R2_THRESHOLD):
    """Histogram of per-cell full ffNLE model R^2 (full model only, no
    ablations), one subpanel per visual area. Dashed line marks `threshold`;
    annotation reports % of cells passing it."""
    areas_present = [a for a in _IMP_REGION_ORDER
                     if sum(r['area'] == a for r in records) >= MIN_CELLS_AREA]
    if not areas_present:
        print('No areas with enough cells for R^2 histogram — skipping.')
        return

    fig, axes = plt.subplots(1, len(areas_present),
                              figsize=_scaled(len(areas_present) * 2.2 + 0.5, 3.0),
                              constrained_layout=True, sharey=False)
    if len(areas_present) == 1:
        axes = [axes]

    print('\n' + '=' * 60)
    print(f'FULL MODEL R^2 HISTOGRAM (threshold = {threshold})')
    print('=' * 60)

    for ax, a in zip(axes, areas_present):
        color = COLORS.get(a, '#888888')
        r2_vals = np.array([r['full_r2'] for r in records
                            if r['area'] == a and np.isfinite(r['full_r2'])])
        n_pass = int(np.sum(r2_vals > threshold))
        pct_pass = 100.0 * n_pass / len(r2_vals) if len(r2_vals) else float('nan')
        print(f'  {a}: n={len(r2_vals)}, pass={n_pass} ({pct_pass:.1f}%), '
              f'median={np.median(r2_vals) if len(r2_vals) else float("nan"):.3f}')

        ax.hist(r2_vals, bins=30, range=(-0.5, 0.5), color=color, alpha=0.80,
                edgecolor='k', linewidth=0.4)
        ax.axvline(threshold, color='k', lw=1.0, ls='--', zorder=5)
        ax.text(0.97, 0.97, f'n={len(r2_vals)}\n{pct_pass:.0f}% pass',
                ha='right', va='top', transform=ax.transAxes, fontsize=6, color='0.2')
        ax.set_xlim(-0.5, 0.5)
        ax.set_title(a, fontsize=8)
        ax.set_xlabel(r'Full model $R^2$', fontsize=7)

    axes[0].set_ylabel('Cell count', fontsize=7)

    fig.suptitle(
        f'Full ffNLE model $R^2$ distribution by visual area\n'
        f'Dashed line = threshold ({threshold})',
        fontsize=9)

    path = os.path.join(out_dir, 'full_model_r2_histogram.svg')
    _save_svg_png(fig, path)


def make_r2_pass_fraction_svg(records, out_dir, threshold=R2_THRESHOLD):
    """Bar chart: % of cells per visual area whose full ffNLE model R^2
    exceeds `threshold` (full model only, no ablations)."""
    areas_present = [a for a in _IMP_REGION_ORDER
                     if sum(r['area'] == a for r in records) >= MIN_CELLS_AREA]
    if not areas_present:
        print('No areas with enough cells for R^2 pass-fraction bar — skipping.')
        return

    fracs, ns = [], []
    for a in areas_present:
        r2_vals = np.array([r['full_r2'] for r in records
                            if r['area'] == a and np.isfinite(r['full_r2'])])
        ns.append(len(r2_vals))
        fracs.append(100.0 * np.sum(r2_vals > threshold) / len(r2_vals) if len(r2_vals) else 0.0)

    xs = np.arange(len(areas_present))
    colors = [COLORS.get(a, '#888888') for a in areas_present]

    fig, ax = plt.subplots(figsize=_scaled(len(areas_present) * 0.9 + 0.8, 3.2),
                            constrained_layout=True)
    ax.bar(xs, fracs, color=colors, edgecolor='k', linewidth=0.5)
    for xi, (f, n) in enumerate(zip(fracs, ns)):
        ax.text(xi, f + 1.0, f'{f:.0f}%', ha='center', va='bottom', fontsize=6)
        ax.text(xi, -2.5, f'n={n}', ha='center', va='top', fontsize=5, color='0.4',
                transform=ax.get_xaxis_transform())

    ax.set_xticks(xs)
    ax.set_xticklabels(areas_present, fontsize=8)
    ax.set_ylabel(f'% cells with full $R^2$ > {threshold}', fontsize=8)
    ax.set_xlim(-0.6, len(areas_present) - 0.4)
    ax.set_ylim(0, max(max(fracs) * 1.3 if fracs else 10, 10))
    ax.axhline(0, color='k', lw=0.5)

    fig.suptitle(
        f'% cells passing full model $R^2$ threshold ({threshold}) by visual area',
        fontsize=9)

    path = os.path.join(out_dir, 'full_model_r2_pass_fraction.svg')
    _save_svg_png(fig, path)


_ALL_VAR_KEYS   = ['theta', 'dTheta', 'phi', 'dPhi', 'pitch', 'gyro_y', 'roll', 'gyro_x']
_ALL_VAR_LABELS = {'theta': 'theta', 'dTheta': 'dTheta', 'phi': 'phi', 'dPhi': 'dPhi',
                    'pitch': 'pitch', 'gyro_y': 'dPitch', 'roll': 'roll', 'gyro_x': 'dRoll'}
_POS_VAR_KEYS = ['theta', 'phi', 'pitch', 'roll']
_VEL_VAR_KEYS = ['dTheta', 'dPhi', 'gyro_y', 'gyro_x']
_HEATMAP_VAR_ORDER = _POS_VAR_KEYS + _VEL_VAR_KEYS  # position block, then velocity block
# 'D' (diamond) reads too much like 's' (square) at a glance -> use '*' for roll.
_POS_MARKERS  = {'theta': 'o', 'phi': 's', 'pitch': '^', 'roll': '*'}

# Each position variable and its angular-velocity counterpart share a hue
# family (position = darker/more saturated, velocity = a lighter tint of
# the same hue), so a pair is recognizable as "the same thing" at a glance
# in the legend, while the 4 families themselves are spread around the hue
# wheel for mutual separation. Colors are hand-picked, not drawn from a
# continuous colormap (a colormap slice reads as a gradient, not discrete
# categories) and chosen to sit away from the anatomical-area palette in
# COLORS (muted teal-green/orange/purple/pink/gold/brown).
_VAR_COLOR_PAIRS = {
    'theta': ('#3B5B92', '#A8C2E0'),  # blue family  (dTheta)
    'phi':   ('#A23B3B', '#E3A3A3'),  # red family   (dPhi)
    'pitch': ('#2E8B8B', '#A8D8D8'),  # teal family  (gyro_y / dPitch)
    'roll':  ('#8C8C3D', '#D9D9A3'),  # olive family (gyro_x / dRoll)
}
_VAR_COLORS = {
    'theta':  _VAR_COLOR_PAIRS['theta'][0], 'dTheta': _VAR_COLOR_PAIRS['theta'][1],
    'phi':    _VAR_COLOR_PAIRS['phi'][0],   'dPhi':   _VAR_COLOR_PAIRS['phi'][1],
    'pitch':  _VAR_COLOR_PAIRS['pitch'][0], 'gyro_y': _VAR_COLOR_PAIRS['pitch'][1],
    'roll':   _VAR_COLOR_PAIRS['roll'][0],  'gyro_x': _VAR_COLOR_PAIRS['roll'][1],
}

_FIG_SCALE = 0.75   # shrink factor applied to figsize across these figures
_FIG_DPI   = 300


def _scaled(w, h):
    return (w * _FIG_SCALE, h * _FIG_SCALE)


# Shared figsize for the LDI beeswarm and eye/head tuning-curve-correlation
# beeswarm plots, fixed to the n_areas=5 case so position/velocity figures
# of either kind are pixel-identical in size.
_BEESWARM_FIGSIZE = (4,3)


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


def _collect_ldi_mi_stats(all_cells, var_keys=_POS_VAR_KEYS):
    """area -> var_name (default theta/phi/pitch/roll, or `var_keys`) -> dict(
    mi_vals, ldi_vals, n_mi, n_ldi, mi_mean, mi_std, ldi_mean, ldi_std)."""
    stats = {a: {} for a in REGION_ORDER}
    for v in var_keys:
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
    """(1) 2-panel (light/dark) heatmap: rows=variable, cols=area, color=mean
    ablation index (0-0.5 scale -- values rarely exceed ~0.3, so a 0-1 scale
    washed out most of the dynamic range), cell text = mean +/- SEM."""
    stats = _collect_imp_stats(records)
    areas = _IMP_REGION_ORDER
    var_keys   = _HEATMAP_VAR_ORDER
    var_labels = [_ALL_VAR_LABELS[v] for v in var_keys]
    vmax = 0.5

    fig, axes = plt.subplots(1, 2, figsize=_scaled(10, 7.0), constrained_layout=True)
    im = None
    for cond, ax in zip(['light', 'dark'], axes):
        mat     = np.full((len(var_keys), len(areas)), np.nan)
        sem_mat = np.full((len(var_keys), len(areas)), np.nan)
        for ai, a in enumerate(areas):
            for vi, v in enumerate(var_keys):
                d = stats[a][v]
                mat[vi, ai]     = d[f'{cond}_mean']
                sem_mat[vi, ai] = d[f'{cond}_sem']
        im = ax.imshow(mat, vmin=0, vmax=vmax, cmap='plasma', aspect='auto')
        ax.set_xticks(range(len(areas)))
        ax.set_xticklabels(areas, fontsize=9)
        ax.set_yticks(range(len(var_keys)))
        ax.set_yticklabels(var_labels, fontsize=9)
        ax.set_title(cond.capitalize(), fontsize=9)
        ax.axhline(len(_POS_VAR_KEYS) - 0.5, color='white', lw=1.2)
        for ai in range(len(areas)):
            for vi in range(len(var_keys)):
                val, sem = mat[vi, ai], sem_mat[vi, ai]
                if np.isfinite(val):
                    color = 'white' if val < vmax / 2 else 'black'
                    sem_str = f'{sem:.2f}' if np.isfinite(sem) else 'n/a'
                    ax.text(ai, vi, f'{val:.2f}\n±{sem_str}', ha='center', va='center',
                            fontsize=8, color=color)
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
    # _HEATMAP_VAR_ORDER = _POS_VAR_KEYS + _VEL_VAR_KEYS -- all 4 position
    # variables, then all 4 velocity variables, not interleaved pairwise.
    var_keys = _VEL_VAR_KEYS if scope == 'velocity' else _HEATMAP_VAR_ORDER
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


def make_mi_vs_ai_scatter_svg(records, all_cells, out_dir):
    """Companion to make_ldi_vs_ai_scatter_svg: same area x variable grid
    (20 points: 5 areas x 4 position variables, color=area, marker=
    variable), but using the RAW light-condition values directly -- mean
    CV MI (x) and mean ablation index (y) -- instead of their
    light-dependence-difference forms (LDI-0.5, AI_light-AI_dark)."""
    imp_stats = _collect_imp_stats(records)
    ldi_stats = _collect_ldi_mi_stats(all_cells)

    fig, ax = plt.subplots(figsize=_scaled(5.5, 5.5), constrained_layout=True)
    pts = []
    for area in _IMP_REGION_ORDER:
        for v in _POS_VAR_KEYS:
            ld = ldi_stats[area][v]
            ip = imp_stats[area][v]
            if ld['n_mi'] < MIN_CELLS_AREA or ip['n_light'] < MIN_CELLS_AREA:
                continue
            x = ld['mi_mean']
            y = ip['light_mean']
            pts.append((area, v, x, y))
            ax.scatter(x, y, color=COLORS.get(area, '#888888'), marker=_POS_MARKERS[v],
                       s=70, edgecolors='k', linewidths=0.6, zorder=3)

    ax.set_xlabel('Mean CV MI (raw modulation depth, light)')
    ax.set_ylabel('Mean ablation index (light)')
    ax.set_title('Raw modulation depth vs. raw ablation index\n'
                 '(light condition; cf. LDI-vs-AI-difference scatter)', fontsize=8)

    area_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS.get(a, '#888888'),
                            markeredgecolor='k', markersize=7, label=a) for a in _IMP_REGION_ORDER]
    var_handles = [Line2D([0], [0], marker=_POS_MARKERS[v], color='w', markerfacecolor='0.6',
                           markeredgecolor='k', markersize=7, label=v) for v in _POS_VAR_KEYS]
    leg1 = ax.legend(handles=area_handles, title='Area', loc='upper left',
                     fontsize=6, title_fontsize=6)
    ax.add_artist(leg1)
    ax.legend(handles=var_handles, title='Variable', loc='center left',
              bbox_to_anchor=(1.02, 0.5), fontsize=6, title_fontsize=6)

    path = os.path.join(out_dir, 'mi_vs_ai_scatter.svg')
    _save_svg_png(fig, path)
    return pts


def make_ldi_vs_ai_scatter_fit_svg(records, all_cells, out_dir):
    """Duplicate of make_ldi_vs_ai_scatter_svg, with a per-area line of best
    fit drawn through that area's (LDI-0.5, AI_light-AI_dark) points."""
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

    for area in _IMP_REGION_ORDER:
        area_pts = [(x, y) for a, v, x, y in pts if a == area]
        if len(area_pts) < 2:
            continue
        xs = np.array([p[0] for p in area_pts])
        ys = np.array([p[1] for p in area_pts])
        slope, intercept = np.polyfit(xs, ys, 1)
        x_fit = np.array([xs.min(), xs.max()])
        ax.plot(x_fit, slope * x_fit + intercept,
                color=COLORS.get(area, '#888888'), lw=1.2, alpha=0.7, zorder=2)

    for area, v, x, y in pts:
        if (area, v) in [('RL', 'pitch'), ('AM', 'phi')]:
            ax.annotate(f'{area}-{v}', (x, y), textcoords='offset points',
                        xytext=(7, 7), fontsize=7)

    ax.set_xlabel('LDI - 0.5  (marginal tuning bias; + = light-leaning)')
    ax.set_ylabel('AI$_{light}$ - AI$_{dark}$  (+ = unique contribution greater in light)')
    ax.set_title('Marginal tuning bias vs. unique multivariate contribution\n'
                  '(with per-area line of best fit)', fontsize=8)

    area_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS.get(a, '#888888'),
                            markeredgecolor='k', markersize=7, label=a) for a in _IMP_REGION_ORDER]
    var_handles = [Line2D([0], [0], marker=_POS_MARKERS[v], color='w', markerfacecolor='0.6',
                           markeredgecolor='k', markersize=7, label=v) for v in _POS_VAR_KEYS]
    leg1 = ax.legend(handles=area_handles, title='Area', loc='upper left',
                     fontsize=6, title_fontsize=6)
    ax.add_artist(leg1)
    ax.legend(handles=var_handles, title='Variable', loc='center left',
              bbox_to_anchor=(1.02, 0.5), fontsize=6, title_fontsize=6)

    path = os.path.join(out_dir, 'ldi_vs_ai_scatter_fit.svg')
    _save_svg_png(fig, path)

    return pts


_EYE_VAR_KEYS  = ['theta', 'phi']    # eye position
_HEAD_VAR_KEYS = ['pitch', 'roll']   # head position


def _bh_fdr(pvals_by_key):
    """Benjamini-Hochberg FDR correction. Input/output: {key: p-value}."""
    keys = sorted(pvals_by_key, key=lambda k: pvals_by_key[k])
    m = len(keys)
    sorted_p = [pvals_by_key[k] for k in keys]
    adj = [0.0] * m
    adj[m - 1] = sorted_p[m - 1]
    for i in range(m - 2, -1, -1):
        adj[i] = min(adj[i + 1], sorted_p[i] * m / (i + 1))
    return {k: adj[i] for i, k in enumerate(keys)}


_MIN_TUNING_CORR_BINS = 5


def _tuning_light_dark_corr(c, v):
    """Pearson correlation between a cell's light and dark tuning curves
    for variable v -- the raw-data-grounded alternative to LDI. LDI is a
    ratio of two cross-validated modulation indices (each already a
    multi-step derived quantity: split-half resampling, peak/trough
    detection, then a (hi-lo)/(hi+lo) ratio, averaged over 50 repeats) --
    a ratio of two noisy ratios compounds noise badly, which is why eye LDI
    came out empirically indistinguishable from uniform-random noise. This
    metric instead directly correlates the two raw binned tuning curves
    ({v}_tuning vs {v}_tuning_dark, the actual mean firing rate per bin --
    no modulation-index machinery at all):
      1  = identical tuning shape regardless of light condition (fully light-invariant)
      0  = no relationship between the two conditions' tuning
      -1 = inverted tuning between conditions
    Returns NaN if either curve is missing, the variable isn't flagged
    reliable in light ({v}_isrel, a basic quality gate -- correlating two
    noisy/flat curves is meaningless), or there are too few jointly-finite
    bins to correlate."""
    tc_l = c.get(f'{v}_tuning')
    tc_d = c.get(f'{v}_tuning_dark')
    if tc_l is None or tc_d is None or not c.get(f'{v}_isrel', False):
        return np.nan
    tc_l = np.asarray(tc_l, dtype=float)
    tc_d = np.asarray(tc_d, dtype=float)
    mask = np.isfinite(tc_l) & np.isfinite(tc_d)
    if mask.sum() < _MIN_TUNING_CORR_BINS:
        return np.nan
    if np.std(tc_l[mask]) == 0 or np.std(tc_d[mask]) == 0:
        return np.nan
    return float(np.corrcoef(tc_l[mask], tc_d[mask])[0, 1])


def _motion_preference_score(tuning, bins):
    """Correlation between firing rate and |velocity| (distance from zero)
    across the bins of ONE tuning curve:
      +  = motion-preferring (fires more away from zero velocity)
      -  = rest-preferring   (fires more near zero velocity)
    Used to determine, for a single light or dark velocity tuning curve,
    which side of zero velocity the cell prefers -- independent of
    _tuning_light_dark_corr, which only measures whether the light and dark
    curves have the same SHAPE, not what that shape means behaviorally.
    Returns NaN if too few jointly-finite bins, or no variance in either the
    rate or |bins| (e.g. all bins on the same side of zero)."""
    tc = np.asarray(tuning, dtype=float)
    b  = np.asarray(bins, dtype=float)
    mask = np.isfinite(tc) & np.isfinite(b)
    if mask.sum() < _MIN_TUNING_CORR_BINS:
        return np.nan
    absb = np.abs(b[mask])
    if np.std(tc[mask]) == 0 or np.std(absb) == 0:
        return np.nan
    return float(np.corrcoef(tc[mask], absb)[0, 1])


def test_eye_vs_head_tuning_corr(all_cells, eye_keys=_EYE_VAR_KEYS,
                                  head_keys=_HEAD_VAR_KEYS, label='position'):
    """Per-area paired Wilcoxon signed-rank test, same design as the
    (superseded) LDI-based version, but on light-dark tuning-curve
    correlation (_tuning_light_dark_corr) instead of |LDI-0.5|. Pass
    eye_keys=['dTheta','dPhi'], head_keys=['gyro_y','gyro_x'],
    label='velocity' for the angular-velocity counterpart.

    Paired at the CELL level: each cell's available eye correlations are
    averaged into one eye_score, head correlations into one head_score,
    and only cells with both defined are kept (avoids pseudo-replication
    from a cell supplying up to 2 eye rows and 2 head rows).

    The 5 per-area p-values are Benjamini-Hochberg FDR-corrected. A second,
    separate Mann-Whitney U test asks -- for each area -- whether THAT
    area's per-cell (eye-head) gap is itself different from the pooled gap
    in every other area; that's the test that supports "area X is the
    outlier," not just "area X individually has a significant gap."

    Sign convention is FLIPPED relative to the LDI version: higher
    correlation = MORE light-invariant, so a POSITIVE median(eye-head)
    here means eye is MORE invariant than head.

    Returns dict(per_area_eye, per_area_head, per_area_diffs, raw_p, fdr_p,
    medians, ns, vs_rest_p, areas_present)."""
    from scipy.stats import wilcoxon, mannwhitneyu

    per_area_eye   = {a: [] for a in REGION_ORDER}
    per_area_head  = {a: [] for a in REGION_ORDER}
    per_area_diffs = {a: [] for a in REGION_ORDER}

    for c in all_cells:
        if c['area'] not in per_area_diffs:
            continue
        eye_vals  = [_tuning_light_dark_corr(c, v) for v in eye_keys]
        head_vals = [_tuning_light_dark_corr(c, v) for v in head_keys]
        eye_vals  = [v for v in eye_vals  if np.isfinite(v)]
        head_vals = [v for v in head_vals if np.isfinite(v)]
        if not eye_vals or not head_vals:
            continue
        eye_score, head_score = float(np.mean(eye_vals)), float(np.mean(head_vals))
        per_area_eye[c['area']].append(eye_score)
        per_area_head[c['area']].append(head_score)
        per_area_diffs[c['area']].append(eye_score - head_score)

    areas_present = [a for a in REGION_ORDER if len(per_area_diffs[a]) >= MIN_CELLS_AREA]

    raw_p, medians, ns = {}, {}, {}
    for a in areas_present:
        diffs = np.array(per_area_diffs[a])
        ns[a] = len(diffs)
        medians[a] = float(np.median(diffs))
        try:
            _, p = wilcoxon(diffs)
        except ValueError:
            p = np.nan
        raw_p[a] = p
    fdr_p = _bh_fdr(raw_p)

    vs_rest_p = {}
    for a in areas_present:
        this = np.array(per_area_diffs[a])
        rest = np.concatenate([np.array(per_area_diffs[o]) for o in areas_present if o != a])
        try:
            _, p = mannwhitneyu(this, rest, alternative='two-sided')
        except ValueError:
            p = np.nan
        vs_rest_p[a] = p

    print('\n' + '=' * 70)
    print(f'EYE vs HEAD TUNING-CURVE LIGHT/DARK CORRELATION ({label}): per-area paired Wilcoxon')
    print('  (positive median = eye MORE light-invariant than head)')
    print('=' * 70)
    for a in areas_present:
        sig = '*' if fdr_p[a] < 0.05 else ' '
        print(f'  {a}: n={ns[a]:5d}  median(eye-head)={medians[a]:+.3f}  '
              f'p={raw_p[a]:.2e}  FDR-p={fdr_p[a]:.2e} {sig}')
    print("\n  Area-vs-rest (is this area's eye-head gap different from the other areas'?):")
    for a in areas_present:
        print(f'  {a} vs rest: p={vs_rest_p[a]:.2e}')
    print('=' * 70)

    return dict(per_area_eye=per_area_eye, per_area_head=per_area_head,
                per_area_diffs=per_area_diffs, raw_p=raw_p, fdr_p=fdr_p,
                medians=medians, ns=ns, vs_rest_p=vs_rest_p, areas_present=areas_present)


def _eye_head_tuning_corr_autolim(res):
    """Auto-fit (y_lo, y_hi) for the eye/head tuning-corr beeswarm -- the
    per-area mean+/-SEM extents plus headroom for the significance stars,
    matching the y-limit logic in make_eye_head_tuning_corr_svg. Used to
    compute a shared ylim across the position/velocity figures."""
    area_tops = []
    y_lo, y_hi = np.inf, -np.inf
    for a in res['areas_present']:
        local_top = -np.inf
        for vals in (res['per_area_eye'][a], res['per_area_head'][a]):
            vals = np.asarray(vals)
            mean = vals.mean()
            sem  = vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0
            local_top = max(local_top, mean + sem)
            y_lo = min(y_lo, mean - sem)
            y_hi = max(y_hi, mean + sem)
        area_tops.append(local_top)
    pad = (y_hi - y_lo) * 0.25 if y_hi > y_lo else 0.02
    ymax = max(area_tops) + pad if area_tops else y_hi + pad
    return y_lo - pad, ymax


def make_eye_head_tuning_corr_svg(all_cells, out_dir, eye_keys=_EYE_VAR_KEYS,
                                   head_keys=_HEAD_VAR_KEYS, label='position',
                                   fname='eye_head_tuning_corr_position.svg',
                                   res=None, ylim=None):
    """Beeswarm-style dot plot (same format as make_ldi_beeswarm_svg)
    answering 'is eye encoding less light-dependent than head encoding?'
    using the raw-data-grounded light-dark tuning-curve correlation instead
    of LDI (see test_eye_vs_head_tuning_corr / _tuning_light_dark_corr).
    Per area: mean correlation for eye-side vs head-side variables, dot at
    the mean with a vertical SEM band, '*' from the paired per-area
    Wilcoxon test (FDR-corrected across the 5 areas). y-limits are fit to
    the data range unless an explicit `ylim` is passed (e.g. to share the
    same scale between the position and velocity calls).
    Pass a precomputed `res` (from test_eye_vs_head_tuning_corr) to avoid
    re-running the stats/printing them twice.
    Call with eye_keys=['dTheta','dPhi'], head_keys=['gyro_y','gyro_x'],
    label='velocity' for the angular-velocity counterpart."""
    if res is None:
        res = test_eye_vs_head_tuning_corr(all_cells, eye_keys=eye_keys,
                                            head_keys=head_keys, label=label)
    areas_present = res['areas_present']
    if not areas_present:
        print(f'No areas with enough cells for eye/head tuning-corr ({label}) -- skipping.')
        return res

    EYE_COLOR, HEAD_COLOR = '#C9824A', '#4A7AA8'  # warm/cool, distinct from area + variable palettes
    off = 0.12
    fig, ax = plt.subplots(figsize=_BEESWARM_FIGSIZE, constrained_layout=True)

    area_tops = []
    y_lo, y_hi = np.inf, -np.inf
    for xi, a in enumerate(areas_present):
        local_top = -np.inf
        for vals, xoff, color, cond in [
            (res['per_area_eye'][a],  -off, EYE_COLOR,  'eye'),
            (res['per_area_head'][a],  off, HEAD_COLOR, 'head'),
        ]:
            vals = np.asarray(vals)
            mean = vals.mean()
            sem  = vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0
            ax.vlines(xi + xoff, mean - sem, mean + sem, color=color, alpha=0.35, lw=5, zorder=1)
            ax.scatter(xi + xoff, mean, color=color, s=45, edgecolors='k', linewidths=0.6,
                       zorder=3, label=cond if xi == 0 else None)
            local_top = max(local_top, mean + sem)
            y_lo = min(y_lo, mean - sem)
            y_hi = max(y_hi, mean + sem)
        area_tops.append(local_top)

    pad = (y_hi - y_lo) * 0.25 if y_hi > y_lo else 0.02
    ymax = max(area_tops) + pad if area_tops else y_hi + pad
    for xi, (a, top) in enumerate(zip(areas_present, area_tops)):
        if res['fdr_p'][a] < 0.05:
            ax.text(xi, top + 0.15 * pad, '*', ha='center', va='bottom', fontsize=14)

    ax.axhline(0, color='0.7', lw=0.8, ls='--', zorder=0)
    ax.set_xticks(range(len(areas_present)))
    ax.set_xticklabels(areas_present)
    ax.set_xlim(-0.6, len(areas_present) - 0.4)
    ax.set_ylim(*(ylim if ylim is not None else (y_lo - pad, ymax)))
    ax.set_ylabel('Light-dark tuning-curve correlation\n(1 = identical tuning regardless of light, 0 = no relationship)',
                  fontsize=8)
    ax.legend(fontsize=7, loc='upper left')
    ax.set_title(f'Eye vs. head tuning-curve light/dark correlation ({label})\n'
                 '* = FDR-significant paired Wilcoxon (per area); higher = more light-invariant',
                 fontsize=8)

    path = os.path.join(out_dir, fname)
    _save_svg_png(fig, path)
    return res


_TUNING_CORR_INVARIANT_THRESH = 0.2  # |r| below this = "no relationship"


def _categorize_tuning_corr(vals, thresh=_TUNING_CORR_INVARIANT_THRESH):
    """% of per-cell light-dark tuning-curve correlations falling into each
    of 3 categories: invariant (r >= thresh), no relationship (|r| <
    thresh), inverted (r <= -thresh). See the negative-correlation
    investigation in _tuning_light_dark_corr's docstring/usage -- negative
    values here reflect a real, measured shape inversion concentrated in
    the best-sampled bins (not low-occupancy noise), so they're kept as
    their own category rather than clipped to 0."""
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    if n == 0:
        return dict(invariant=np.nan, no_rel=np.nan, inverted=np.nan, n=0)
    return dict(
        invariant=100.0 * np.sum(vals >= thresh) / n,
        no_rel=100.0 * np.sum(np.abs(vals) < thresh) / n,
        inverted=100.0 * np.sum(vals <= -thresh) / n,
        n=n,
    )


def quantify_velocity_mi_by_corr_category(all_cells, thresh=_TUNING_CORR_INVARIANT_THRESH):
    """For velocity variables (dTheta, dPhi, gyro_y, gyro_x), classify each
    (cell, variable) pair by its light-dark tuning-curve correlation into
    invariant / no-relationship / inverted (same categories and threshold
    as _categorize_tuning_corr), then compare each cell's own LIGHT vs.
    DARK raw CV MI (modulation depth) within that category. Paired (same
    cell, same variable), so a Wilcoxon signed-rank test on (light-dark)
    applies directly. Answers: when a cell's velocity tuning inverts
    between light and dark, is that because the cell is MORE tuned to
    movement in light, or LESS tuned in light, relative to dark -- and is
    that pattern actually distinctive to inverted cells, or just true of
    velocity-tuned cells in general?
    Returns {category: dict(light, dark, n, pct_light_gt_dark, median_light,
    median_dark, wilcoxon_p)}."""
    from scipy.stats import wilcoxon

    vel_vars = ['dTheta', 'dPhi', 'gyro_y', 'gyro_x']
    by_cat = {'invariant': {'light': [], 'dark': []},
              'no_rel':    {'light': [], 'dark': []},
              'inverted':  {'light': [], 'dark': []}}

    for c in all_cells:
        for v in vel_vars:
            r = _tuning_light_dark_corr(c, v)
            if not np.isfinite(r):
                continue
            light_mi = c.get(f'{v}_rel')
            dark_mi  = c.get(f'{v}_rel_dark')
            if light_mi is None or dark_mi is None or not (np.isfinite(light_mi) and np.isfinite(dark_mi)):
                continue
            cat = 'invariant' if r >= thresh else ('inverted' if r <= -thresh else 'no_rel')
            by_cat[cat]['light'].append(light_mi)
            by_cat[cat]['dark'].append(dark_mi)

    results = {}
    print('\nVelocity light vs. dark CV MI by light-dark tuning-curve-correlation category:')
    for cat in ('invariant', 'no_rel', 'inverted'):
        light = np.array(by_cat[cat]['light'])
        dark  = np.array(by_cat[cat]['dark'])
        n = len(light)
        if n < 2:
            results[cat] = dict(light=light, dark=dark, n=n, pct_light_gt_dark=np.nan,
                                 median_light=np.nan, median_dark=np.nan, wilcoxon_p=np.nan)
            continue
        pct_gt = 100.0 * np.sum(light > dark) / n
        try:
            _, p = wilcoxon(light - dark)
        except ValueError:
            p = np.nan
        results[cat] = dict(light=light, dark=dark, n=n, pct_light_gt_dark=pct_gt,
                             median_light=float(np.median(light)), median_dark=float(np.median(dark)),
                             wilcoxon_p=p)
        print(f'  {cat}: n={n}  median_light_MI={results[cat]["median_light"]:.4f}  '
              f'median_dark_MI={results[cat]["median_dark"]:.4f}  '
              f'%light>dark={pct_gt:.1f}%  wilcoxon p={p:.2e}')

    return results


def make_velocity_mi_by_corr_category_svg(results, out_dir):
    """Visualizes quantify_velocity_mi_by_corr_category: for each
    light-dark-correlation category (invariant / no relationship /
    inverted), paired light vs. dark CV MI as box + jittered strip points.
    The invariant/no_rel categories are shown alongside inverted as
    context -- they answer whether "more tuned in light" is something
    distinctive about inverted cells specifically, or just true of
    velocity-tuned cells generally."""
    cats = ['invariant', 'no_rel', 'inverted']
    cat_labels = {'invariant': 'Invariant\n(r≥0.2)',
                  'no_rel':    'No relationship\n(|r|<0.2)',
                  'inverted':  'Inverted\n(r≤-0.2)'}
    LIGHT_COLOR, DARK_COLOR = '#E8A838', '#5B7FA6'
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=_scaled(6, 4.4), constrained_layout=True)
    width = 0.32

    for xi, cat in enumerate(cats):
        d = results[cat]
        cat_top = 0.0
        for vals, xoff, color in [(d['light'], -width / 2 - 0.02, LIGHT_COLOR),
                                   (d['dark'],  width / 2 + 0.02, DARK_COLOR)]:
            vals = np.asarray(vals)
            if vals.size == 0:
                continue
            show_vals = vals if vals.size <= 300 else rng.choice(vals, 300, replace=False)
            jitter = rng.uniform(-0.06, 0.06, size=show_vals.size)
            ax.scatter(xi + xoff + jitter, show_vals, color=color, s=4, alpha=0.25,
                       linewidths=0, zorder=1)
            bp = ax.boxplot([vals], positions=[xi + xoff], widths=width * 0.85, whis=[5, 95],
                            showfliers=False, patch_artist=True, zorder=3)
            for el in ('boxes', 'whiskers', 'caps', 'medians'):
                for artist in bp[el]:
                    artist.set_color('k')
                    artist.set_linewidth(1.0)
            bp['boxes'][0].set_facecolor(color)
            bp['boxes'][0].set_alpha(0.85)
            cat_top = max(cat_top, float(np.percentile(vals, 95)))

        if np.isfinite(d['wilcoxon_p']):
            sig = '*' if d['wilcoxon_p'] < 0.05 else 'ns'
            ax.text(xi, cat_top * 1.08, f'{sig}  {d["pct_light_gt_dark"]:.0f}% light>dark',
                    ha='center', va='bottom', fontsize=6)

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([cat_labels[c] for c in cats], fontsize=7)
    ax.set_ylabel('CV MI (raw modulation depth)', fontsize=8)
    ax.set_title('Velocity: light vs. dark modulation depth,\nby light-dark tuning-curve relationship', fontsize=8.5)

    handles = [Patch(facecolor=LIGHT_COLOR, edgecolor='k', linewidth=0.6, label='light'),
               Patch(facecolor=DARK_COLOR, edgecolor='k', linewidth=0.6, label='dark')]
    ax.legend(handles=handles, fontsize=7, loc='upper right')

    path = os.path.join(out_dir, 'velocity_mi_by_corr_category.svg')
    _save_svg_png(fig, path)


def quantify_inverted_velocity_motion_direction(all_cells, thresh=_TUNING_CORR_INVARIANT_THRESH):
    """For velocity-tuned (cell, variable) pairs whose light/dark tuning
    curves are 'inverted' (light-dark correlation <= -thresh, same
    category as quantify_velocity_mi_by_corr_category), determine WHICH
    condition is the motion-preferring one. 'Motion-preferring' means
    firing rate increases with |velocity| -- the cell fires more during
    fast eye/head movements than near zero velocity; 'rest-preferring' is
    the opposite (firing peaks near zero velocity). Scored per curve with
    _motion_preference_score (correlation between rate and |bin center|),
    separately for each cell's light and dark curves -- since the pair is
    inverted, the two scores are expected to differ in sign, and this
    classifies which one is positive (motion-preferring): light or dark.
    A two-sided binomial test against the 50/50 null then asks whether the
    inversion has a consistent preferred direction across the population,
    rather than being a coin-flip relabeling from cell to cell.
    Returns dict(rows=[dict(area, var, score_light, score_dark, corr), ...],
    n_light_motion, n_dark_motion, n_ambiguous, n, pct_light_motion, binom_p)."""
    from scipy.stats import binomtest

    vel_vars = ['dTheta', 'dPhi', 'gyro_y', 'gyro_x']
    rows = []
    n_light_motion = 0
    n_dark_motion  = 0
    n_ambiguous    = 0

    for c in all_cells:
        for v in vel_vars:
            r = _tuning_light_dark_corr(c, v)
            if not np.isfinite(r) or r > -thresh:
                continue
            bins = c.get(f'{v}_bins')
            tc_l = c.get(f'{v}_tuning')
            tc_d = c.get(f'{v}_tuning_dark')
            if bins is None or tc_l is None or tc_d is None:
                continue
            s_l = _motion_preference_score(tc_l, bins)
            s_d = _motion_preference_score(tc_d, bins)
            if not (np.isfinite(s_l) and np.isfinite(s_d)):
                continue
            rows.append(dict(area=c['area'], var=v, score_light=s_l, score_dark=s_d, corr=r))
            if s_l > s_d:
                n_light_motion += 1
            elif s_d > s_l:
                n_dark_motion += 1
            else:
                n_ambiguous += 1

    n = n_light_motion + n_dark_motion
    pct_light_motion = 100.0 * n_light_motion / n if n else float('nan')
    p = binomtest(n_light_motion, n, 0.5).pvalue if n > 0 else float('nan')

    print('\n' + '=' * 70)
    print(f'INVERTED VELOCITY TUNING: which condition is motion-preferring? (|r|>={thresh})')
    print('=' * 70)
    print(f'  n inverted (cell, variable) pairs = {len(rows)}  (ambiguous/tied: {n_ambiguous})')
    if n:
        print(f'  light motion-preferring (dark rest-preferring): {n_light_motion}/{n} ({pct_light_motion:.1f}%)')
        print(f'  dark motion-preferring (light rest-preferring): {n_dark_motion}/{n} ({100 - pct_light_motion:.1f}%)')
        print(f'  binomial test vs. 50/50: p={p:.2e}')
    print('=' * 70)

    return dict(rows=rows, n_light_motion=n_light_motion, n_dark_motion=n_dark_motion,
                n_ambiguous=n_ambiguous, n=n, pct_light_motion=pct_light_motion, binom_p=p)


def make_inverted_velocity_motion_direction_svg(results, out_dir):
    """Visualizes quantify_inverted_velocity_motion_direction. Left panel:
    scatter of each inverted (cell, variable) pair's light motion-
    preference score (x) vs. dark motion-preference score (y) --
    _motion_preference_score, + = fires more away from zero velocity
    ('motion-preferring'), - = fires more near zero ('rest-preferring').
    Points are colored by which side of the y=x diagonal they fall on
    (light score > dark score => light is the motion-preferring condition
    for that pair, and vice versa). Right panel: bar chart of the overall
    % split with the binomial p-value from
    quantify_inverted_velocity_motion_direction."""
    rows = results['rows']
    if not rows:
        print('No inverted velocity (cell, variable) pairs -- skipping motion-direction figure.')
        return

    LIGHT_COLOR, DARK_COLOR = '#E8A838', '#5B7FA6'
    sx = np.array([r['score_light'] for r in rows])
    sy = np.array([r['score_dark']  for r in rows])
    light_wins = sx > sy

    fig, (ax_sc, ax_bar) = plt.subplots(
        1, 2, figsize=_scaled(6.4, 3.6), constrained_layout=True,
        gridspec_kw=dict(width_ratios=[1.0, 1.0]))

    ax_sc.scatter(sx[light_wins], sy[light_wins], s=3, marker='.', color=LIGHT_COLOR,
                  linewidths=0, alpha=0.6, zorder=3,
                  label=f'light motion-pref. (n={results["n_light_motion"]})')
    ax_sc.scatter(sx[~light_wins], sy[~light_wins], s=3, marker='.', color=DARK_COLOR,
                  linewidths=0, alpha=0.6, zorder=3,
                  label=f'dark motion-pref. (n={results["n_dark_motion"]})')
    lo = min(sx.min(), sy.min()) - 0.05
    hi = max(sx.max(), sy.max()) + 0.05
    ax_sc.plot([lo, hi], [lo, hi], color='k', lw=0.8, ls='--', zorder=1)
    ax_sc.axhline(0, color='0.8', lw=0.6, zorder=0)
    ax_sc.axvline(0, color='0.8', lw=0.6, zorder=0)
    ax_sc.set_xlim(lo, hi)
    ax_sc.set_aspect('equal', adjustable='box')
    ax_sc.set_ylim(lo, hi)
    ax_sc.set_xlabel('Light motion-preference score\n(+ = fires more away from zero velocity)', fontsize=7)
    ax_sc.set_ylabel('Dark motion-preference score', fontsize=7)
    ax_sc.set_title('Inverted velocity tuning: per-(cell, variable)\nlight vs. dark motion preference', fontsize=8)
    ax_sc.legend(fontsize=6, loc='upper left')

    pct_l = results['pct_light_motion']
    pct_d = 100.0 - pct_l
    ax_bar.bar([0, 1], [pct_l, pct_d], color=[LIGHT_COLOR, DARK_COLOR],
               edgecolor='k', linewidth=0.6)
    ax_bar.set_xticks([0, 1])
    ax_bar.set_xticklabels(['Light\nmotion-pref.', 'Dark\nmotion-pref.'], fontsize=7)
    ax_bar.set_ylabel('% of inverted pairs', fontsize=7)
    ax_bar.set_ylim(0, 100)
    ax_bar.axhline(50, color='k', lw=0.8, ls='--')
    sig = '*' if results['binom_p'] < 0.05 else 'ns'
    ax_bar.text(0.5, 0.95, f'{sig}  p={results["binom_p"]:.2e}', ha='center', va='top',
                fontsize=6.5, transform=ax_bar.transAxes)
    for xi, v in enumerate([pct_l, pct_d]):
        ax_bar.text(xi, v + 2, f'{v:.0f}%', ha='center', va='bottom', fontsize=7)
    ax_bar.set_title(f'n={results["n"]} inverted pairs', fontsize=8)

    fig.suptitle(
        'Direction of light/dark inversion in velocity tuning\n'
        '(dTheta, dPhi, gyro_y, gyro_x; inverted = light-dark tuning-curve r '
        f'<= -{_TUNING_CORR_INVARIANT_THRESH:.1f})',
        fontsize=8.5)

    path = os.path.join(out_dir, 'inverted_velocity_motion_direction.svg')
    _save_svg_png(fig, path)


def make_eye_head_tuning_corr_category_bar_svg(all_cells, out_dir, eye_keys=_EYE_VAR_KEYS,
                                                head_keys=_HEAD_VAR_KEYS, label='position',
                                                thresh=_TUNING_CORR_INVARIANT_THRESH,
                                                fname='eye_head_tuning_corr_category_position.svg'):
    """100%-stacked bar chart: % of cells categorized as invariant,
    no-relationship, or inverted (see _categorize_tuning_corr), one stacked
    bar per area for eye and one for head. Same eye/head color convention
    as make_eye_head_tuning_corr_svg; category is shown via hatch pattern
    rather than color -- diagonal '////' hatch is reserved elsewhere in
    this script for the dark condition, so this uses dotted/crosshatch
    patterns instead to avoid colliding with that meaning."""
    res = test_eye_vs_head_tuning_corr(all_cells, eye_keys=eye_keys,
                                        head_keys=head_keys, label=label)
    areas_present = res['areas_present']
    if not areas_present:
        print(f'No areas with enough cells for eye/head tuning-corr categories ({label}) -- skipping.')
        return res

    EYE_COLOR, HEAD_COLOR = '#C9824A', '#4A7AA8'
    CAT_ORDER = ['invariant', 'no_rel', 'inverted']
    CAT_HATCH = {'invariant': None, 'no_rel': '..', 'inverted': 'xx'}
    CAT_LABEL = {'invariant': f'invariant (r≥{thresh:.1f})',
                 'no_rel':    f'no relationship (|r|<{thresh:.1f})',
                 'inverted':  f'inverted (r≤-{thresh:.1f})'}

    width = 0.32
    fig, ax = plt.subplots(figsize=_BEESWARM_FIGSIZE, constrained_layout=True)

    print(f'\n% invariant / no-relationship / inverted ({label}, |r|>={thresh} cutoff):')
    for xi, a in enumerate(areas_present):
        for vals, xoff, color, cond in [
            (res['per_area_eye'][a],  -width / 2 - 0.02, EYE_COLOR,  'eye'),
            (res['per_area_head'][a],  width / 2 + 0.02, HEAD_COLOR, 'head'),
        ]:
            cats = _categorize_tuning_corr(vals, thresh)
            print(f'  {a} {cond}: n={cats["n"]}  invariant={cats["invariant"]:.1f}%  '
                  f'no_rel={cats["no_rel"]:.1f}%  inverted={cats["inverted"]:.1f}%')
            bottom = 0.0
            for cat in CAT_ORDER:
                val = cats[cat]
                ax.bar(xi + xoff, val, bottom=bottom, width=width, color=color,
                       hatch=CAT_HATCH[cat], edgecolor='k', linewidth=0.6)
                bottom += val

    ax.set_xticks(range(len(areas_present)))
    ax.set_xticklabels(areas_present)
    ax.set_xlim(-0.6, len(areas_present) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_ylabel('% of cells', fontsize=8)
    ax.set_title(f'Eye vs. head: invariant / no-relationship / inverted tuning ({label})\n'
                 f'(light-dark tuning-curve correlation; |r|<{thresh:.1f} = no relationship)',
                 fontsize=8)

    color_handles = [Patch(facecolor=EYE_COLOR, edgecolor='k', linewidth=0.6, label='eye'),
                      Patch(facecolor=HEAD_COLOR, edgecolor='k', linewidth=0.6, label='head')]
    hatch_handles = [Patch(facecolor='0.75', edgecolor='k', linewidth=0.6,
                            hatch=CAT_HATCH[cat], label=CAT_LABEL[cat]) for cat in CAT_ORDER]
    leg1 = ax.legend(handles=color_handles, title='Group', fontsize=6.5, title_fontsize=6.5,
                      loc='center left', bbox_to_anchor=(1.02, 0.75))
    ax.add_artist(leg1)
    ax.legend(handles=hatch_handles, title='Category', fontsize=6.5, title_fontsize=6.5,
              loc='center left', bbox_to_anchor=(1.02, 0.30))

    path = os.path.join(out_dir, fname)
    _save_svg_png(fig, path)
    return res


def _ldi_beeswarm_autolim(ldi_stats, var_keys):
    """Auto-fit (y_lo, y_hi) for the LDI beeswarm -- the per-area mean+/-SEM
    extents plus a little headroom, matching the y-limit logic
    make_ldi_beeswarm_svg would otherwise use via plain autoscaling. Used to
    compute a shared ylim across the position/velocity calls."""
    y_lo, y_hi = np.inf, -np.inf
    for v in var_keys:
        for area in _IMP_REGION_ORDER:
            d = ldi_stats[area][v]
            if d['n_ldi'] < MIN_CELLS_AREA:
                continue
            mean = d['ldi_mean']
            sem = d['ldi_std'] / np.sqrt(d['n_ldi']) if d['n_ldi'] > 1 else np.nan
            y_lo = min(y_lo, mean - sem)
            y_hi = max(y_hi, mean + sem)
    pad = (y_hi - y_lo) * 0.1 if y_hi > y_lo else 0.02
    return y_lo - pad, y_hi + pad


def make_ldi_beeswarm_svg(all_cells, out_dir, var_keys=_POS_VAR_KEYS, label='position',
                           fname='ldi_beeswarm.svg', ylim=None, ldi_stats=None):
    """(5) Strip plot: x=variable, y=LDI, one point per area (mean), colored
    by area. Vertical band behind each point = within-area SEM of per-cell
    LDI. Horizontal reference line at 0.5 (no light/dark bias). y-limits
    are fit to the data unless an explicit `ylim` is passed (e.g. to share
    the same scale between the position and velocity calls).
    Pass a precomputed `ldi_stats` (from _collect_ldi_mi_stats) to avoid
    recomputing it.
    Pass var_keys=_VEL_VAR_KEYS, label='velocity' for the angular-velocity
    counterpart."""
    if ldi_stats is None:
        ldi_stats = _collect_ldi_mi_stats(all_cells, var_keys=var_keys)
    n_areas = len(_IMP_REGION_ORDER)

    fig, ax = plt.subplots(figsize=_BEESWARM_FIGSIZE, constrained_layout=True)
    for vi, v in enumerate(var_keys):
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
    ax.set_xticks(range(len(var_keys)))
    ax.set_xticklabels(var_keys)
    ax.set_ylim(ylim if ylim is not None else _ldi_beeswarm_autolim(ldi_stats, var_keys))
    ax.set_ylabel('LDI')
    ax.set_title(f'LDI by area and variable ({label})\n(band = within-area SEM of per-cell LDI)', fontsize=8)

    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys(), fontsize=6, ncol=n_areas,
             loc='upper center', bbox_to_anchor=(0.5, -0.12))

    path = os.path.join(out_dir, fname)
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
    """(7) Horizontal bar plot: y=40 rows, top-level split into a position
    half (top) and velocity half (bottom); within each half, a bracketed
    sub-block per visual area (5 areas x 4 variables = 20 rows per half),
    so e.g. 'V1' appears as two separate brackets -- 'V1 position' in the
    top half and 'V1 velocity' in the bottom half. Rows are y-tick-labeled
    by variable only; x=AI_dark-AI_light."""
    from matplotlib.transforms import blended_transform_factory

    stats = _collect_imp_stats(records)
    rows = []  # (area, value, group, var_key)
    for grp, var_keys in (('pos', _POS_VAR_KEYS), ('vel', _VEL_VAR_KEYS)):
        for area in _IMP_REGION_ORDER:
            for v in var_keys:
                d = stats[area][v]
                if d['n_light'] < MIN_CELLS_AREA or d['n_dark'] < MIN_CELLS_AREA:
                    continue
                rows.append((area, d['dark_mean'] - d['light_mean'], grp, v))

    n_pos = sum(1 for r in rows if r[2] == 'pos')

    # Deliberately more compact than the other figures here, per request.
    fig, ax = plt.subplots(figsize=(3.4, 0.13 * len(rows) + 0.6), constrained_layout=True)
    ys   = np.arange(len(rows))[::-1]
    vals = [val for _, val, _, _ in rows]
    colors = ['0.35' if grp == 'pos' else '#1f77b4' for _, _, grp, _ in rows]
    ax.barh(ys, vals, color=colors, height=0.8, zorder=2)

    ax.axvline(0, color='k', lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([_ALL_VAR_LABELS[r[3]] for r in rows], fontsize=4.5)
    ax.set_ylim(ys[-1] - 0.6, ys[0] + 0.6)
    ax.set_xlabel('AI$_{dark}$ - AI$_{light}$  (+ = dark dominant)', fontsize=6)
    ax.set_title('Position (top, grey) vs.\nvelocity (bottom, blue) variables, by area', fontsize=6.5)
    ax.tick_params(axis='x', labelsize=5.5)

    # Dashed line between each area's block; solid line at the single
    # position/velocity boundary (drawn heavier so it reads as the primary
    # split, with the area boundaries as a secondary one).
    i = 0
    while i < len(rows):
        area, grp = rows[i][0], rows[i][2]
        j = i
        while j < len(rows) and rows[j][0] == area and rows[j][2] == grp:
            j += 1
        if j < len(rows):
            if j == n_pos:
                ax.axhline(ys[j - 1] - 0.5, color='0.4', lw=1.2, ls='-', zorder=1)
            else:
                ax.axhline(ys[j - 1] - 0.5, color='0.7', lw=0.6, ls='--', zorder=1)
        i = j

    # Bracket + "{area} {position|velocity}" label for each contiguous run
    # of rows sharing the same area within the same half, placed to the
    # left of the (variable-name) y-tick labels.
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    bracket_x, tick_len = -0.20, 0.03
    i = 0
    while i < len(rows):
        area, grp = rows[i][0], rows[i][2]
        j = i
        while j < len(rows) and rows[j][0] == area and rows[j][2] == grp:
            j += 1
        y_top, y_bot = ys[i] + 0.4, ys[j - 1] - 0.4
        ax.plot([bracket_x, bracket_x], [y_top, y_bot], transform=trans,
                color='k', lw=0.7, clip_on=False)
        ax.plot([bracket_x, bracket_x + tick_len], [y_top, y_top], transform=trans,
                color='k', lw=0.7, clip_on=False)
        ax.plot([bracket_x, bracket_x + tick_len], [y_bot, y_bot], transform=trans,
                color='k', lw=0.7, clip_on=False)
        grp_label = 'position' if grp == 'pos' else 'velocity'
        ax.text(bracket_x - 0.03, (y_top + y_bot) / 2, f'{area} {grp_label}', transform=trans,
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


def _percell_concordance(all_cells, records):
    """Per-cell sign concordance: sign(LDI_cell - 0.5) vs.
    sign(AI_light_cell - AI_dark_cell), pooled across the 4 position
    variables, tallied per area. Unlike make_concordance_bar_svg (which
    collapses each area x variable group to its mean LDI and mean AI before
    comparing signs -- 20 points total), this compares signs cell-by-cell,
    so N per area is in the hundreds-to-thousands instead of 4. Cells with a
    clip-floor/ceiling AI (artifact, see _mi_ai_xy) or a zero-valued
    difference (no sign to compare) are excluded.
    Returns {area: (n_concordant, n_total)}."""
    joined = _join_mi_ai_cells(all_cells, records)
    by_area = {a: [] for a in _IMP_REGION_ORDER}
    for c, r in joined:
        if c['area'] in by_area:
            by_area[c['area']].append((c, r))

    results = {}
    for area in _IMP_REGION_ORDER:
        n_concordant = 0
        n_total = 0
        for v in _POS_VAR_KEYS:
            ai_vi = _IMP_VAR_ORDER.index(v)
            for c, r in by_area[area]:
                ldi = _ldi(c[f'{v}_rel'], c[f'{v}_rel_dark'])
                if not np.isfinite(ldi):
                    continue
                ai_l = float(r['light_imp'][ai_vi])
                ai_d = float(r['dark_imp'][ai_vi])
                if not (np.isfinite(ai_l) and np.isfinite(ai_d)):
                    continue
                if ai_l <= 0.001 or ai_l >= 0.999 or ai_d <= 0.001 or ai_d >= 0.999:
                    continue
                diff_ldi = ldi - 0.5
                diff_ai  = ai_l - ai_d
                if diff_ldi == 0 or diff_ai == 0:
                    continue
                n_total += 1
                if np.sign(diff_ldi) == np.sign(diff_ai):
                    n_concordant += 1
        results[area] = (n_concordant, n_total)
    return results


def make_concordance_bar_percell_svg(all_cells, records, out_dir):
    """Per-cell companion to make_concordance_bar_svg. See
    _percell_concordance for why this is more informative: N per bar is in
    the hundreds-to-thousands rather than 4, and a per-area 95% CI (normal
    approximation to the binomial) shows how much each area's rate could
    plausibly vary, which the original (mean-of-4-points) version can't."""
    results = _percell_concordance(all_cells, records)

    areas = [a for a in _IMP_REGION_ORDER if results[a][1] >= MIN_CELLS_AREA]
    if not areas:
        print('No areas with enough cells for per-cell concordance — skipping.')
        return

    n_con_all = sum(results[a][0] for a in areas)
    n_tot_all = sum(results[a][1] for a in areas)

    labels = areas + ['ALL']
    counts = [results[a] for a in areas] + [(n_con_all, n_tot_all)]
    rates  = [n / t if t else np.nan for n, t in counts]
    errs   = [1.96 * np.sqrt(r * (1 - r) / t) if t else np.nan
              for r, (_, t) in zip(rates, counts)]

    fig, ax = plt.subplots(figsize=_scaled(len(labels) * 1.1 + 1.0, 4.2),
                            constrained_layout=True)
    colors = [COLORS.get(a, '#888888') for a in areas] + ['0.3']
    ax.bar(range(len(labels)), rates, yerr=errs, color=colors, edgecolor='k',
           linewidth=0.6, capsize=3)
    for i, ((n, t), r) in enumerate(zip(counts, rates)):
        ax.text(i, r + 0.05, f'{r:.0%}\nn={t}', ha='center', va='bottom', fontsize=6)

    for ref in (1.0, 0.5, 0.0):
        ax.axhline(ref, color='k', lw=0.8, ls=':', alpha=0.6)
    ax.text(-0.45, 1.02, 'pure multiplicative', fontsize=5, color='0.4', ha='left')
    ax.text(-0.45, 0.52, 'chance',              fontsize=5, color='0.4', ha='left')
    ax.text(-0.45, 0.02, 'pure additive',       fontsize=5, color='0.4', ha='left')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel('Per-cell concordance rate\n'
                  r'(sign(LDI-0.5) == sign(AI$_{light}$-AI$_{dark}$))')
    ax.set_title('Per-cell LDI-vs-AI sign concordance\n'
                 'pooled across position variables (95% CI, normal approx.)', fontsize=8)

    path = os.path.join(out_dir, 'concordance_bar_percell.svg')
    _save_svg_png(fig, path)

    print(f'\nPer-cell concordance (pooled across {len(_POS_VAR_KEYS)} position variables):')
    for label, (n, t), r in zip(labels, counts, rates):
        print(f'  {label}: {n}/{t} = {r:.1%}' if t else f'  {label}: n/a')


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


def make_pm_velocity_light_dark_bar_svg(records, out_dir, area='PM'):
    """New figure: mean ablation index for the 4 velocity variables in a
    single area (PM by default), light vs. dark as side-by-side bars.
    Same light/dark + error-bar convention as make_ai_ci_bar_svg (95% CI =
    mean +/- 1.96*SEM; solid=light, hatched=dark), but broken out per
    velocity variable for one area instead of pooled-across-variables per
    area. Colors use _VAR_COLORS, the same hue-paired palette as the
    fold-change figures, so dTheta/dPhi/dPitch/dRoll read consistently
    across figures."""
    var_keys = _VEL_VAR_KEYS

    light_pool = {v: [] for v in var_keys}
    dark_pool  = {v: [] for v in var_keys}
    for r in records:
        if r['area'] != area:
            continue
        for vi, var in enumerate(_IMP_VAR_ORDER):
            if var not in var_keys:
                continue
            vl, vd = float(r['light_imp'][vi]), float(r['dark_imp'][vi])
            if np.isfinite(vl):
                light_pool[var].append(vl)
            if np.isfinite(vd):
                dark_pool[var].append(vd)

    fig, ax = plt.subplots(figsize=_scaled(5, 4), constrained_layout=True)
    width = 0.35
    print(f'\nMean AI for {area}, light vs. dark, by velocity variable (+/- 95% CI):')
    for xi, v in enumerate(var_keys):
        color = _VAR_COLORS[v]
        for pool, xoff, alpha, hatch, cond in [
            (light_pool[v], -width / 2, 0.85, None,   'light'),
            (dark_pool[v],   width / 2, 0.5,  _HATCH, 'dark'),
        ]:
            vals = np.array(pool)
            if vals.size < 2:
                continue
            mean = vals.mean()
            ci95 = 1.96 * vals.std(ddof=1) / np.sqrt(vals.size)
            ax.bar(xi + xoff, mean, width=width, yerr=ci95, capsize=3, color=color,
                   alpha=alpha, hatch=hatch, edgecolor='k', linewidth=0.6,
                   error_kw=dict(lw=1.0))
            print(f'  {_ALL_VAR_LABELS[v]} {cond}: n={vals.size}, mean={mean:.3f}, '
                  f'95% CI=[{mean - ci95:.3f}, {mean + ci95:.3f}]')

    ax.set_xticks(range(len(var_keys)))
    ax.set_xticklabels([_ALL_VAR_LABELS[v] for v in var_keys])
    ax.set_ylabel('Mean ablation index')
    ax.set_title(f'{area}: mean AI by velocity variable, light vs. dark\n'
                 '(error bars = 95% CI; solid=light, hatched=dark)', fontsize=8)

    path = os.path.join(out_dir, f'{area.lower()}_velocity_ai_light_dark.svg')
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


def _mi_ai_xy(pairs, v):
    """Per-cell (CV MI, ablation index) arrays for variable v, pooling light
    + dark as separate points, with non-finite and AI clip-floor/ceiling
    (==0 or ==1) points dropped."""
    ai_vi = _IMP_VAR_ORDER.index(v)
    mi_l = np.array([c[f'{v}_rel'] for c, _ in pairs])
    mi_d = np.array([c[f'{v}_rel_dark'] for c, _ in pairs])
    ai_l = np.array([r['light_imp'][ai_vi] for _, r in pairs])
    ai_d = np.array([r['dark_imp'][ai_vi] for _, r in pairs])
    x = np.concatenate([mi_l, mi_d])
    y = np.concatenate([ai_l, ai_d])
    finite  = np.isfinite(x) & np.isfinite(y)
    clipped = (y <= 0.001) | (y >= 0.999)
    keep = finite & ~clipped
    return x[keep], y[keep], int(finite.sum()), int((finite & clipped).sum())


def make_mi_ai_percell_scatter_svg(all_cells, records, out_dir):
    """(13) Per-cell scatter, 5 rows (area) x 4 columns (position variable):
    x = CV MI, y = ablation index. Light and dark conditions are both
    included as separate points per cell. Cells whose ablation index hit the
    [0,1] clip floor/ceiling (see _load_importance_cells) are excluded --
    those are clipping artifacts, not real AI values, and previously
    inflated the marginal piles at y=0/y=1 that made the relationship hard
    to read."""
    joined = _join_mi_ai_cells(all_cells, records)

    # Pre-sort matched cells by area once (not per variable/condition), then
    # build whole-array (x, y) vectors per area and issue one scatter() call
    # per area/panel instead of one call per point -- a few thousand
    # individual scatter() calls in a Python loop is extremely slow.
    by_area = {a: [] for a in _IMP_REGION_ORDER}
    for c, r in joined:
        if c['area'] in by_area:
            by_area[c['area']].append((c, r))

    n_rows = len(_IMP_REGION_ORDER)
    n_cols = len(_POS_VAR_KEYS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=_scaled(7.2, 9.0),
                              constrained_layout=True, sharex=True, sharey=True)

    n_clipped_total = 0
    n_total = 0
    for ri, area in enumerate(_IMP_REGION_ORDER):
        pairs = by_area[area]
        color = COLORS.get(area, '#888888')
        for vi, v in enumerate(_POS_VAR_KEYS):
            ax = axes[ri, vi]
            if pairs:
                x, y, n_finite, n_clip = _mi_ai_xy(pairs, v)
                n_total += n_finite
                n_clipped_total += n_clip
                ax.scatter(x, y, color=color, s=3, alpha=0.25,
                          linewidths=0, zorder=2)
            if ri == 0:
                ax.set_title(v, fontsize=7)
            if ri == n_rows - 1:
                ax.set_xlabel('CV MI', fontsize=6)
            if vi == 0:
                ax.set_ylabel(f'{area}\nAblation index', fontsize=6)
            ax.tick_params(labelsize=5)

    print(f'\nMI-vs-AI scatter: excluded {n_clipped_total}/{n_total} '
          f'({100 * n_clipped_total / n_total:.1f}%) points clipped to AI=0 or 1.')

    fig.suptitle('Per-cell CV MI vs. ablation index, by area and variable\n'
                 '(AI clip-floor/ceiling points excluded)', fontsize=8)

    path = os.path.join(out_dir, 'mi_ai_percell_scatter.svg')
    _save_svg_png(fig, path)


def _binned_median_iqr(x, y, edges, min_per_bin=8):
    """Apply pre-computed (shared, equal-width) bin edges to (x, y); return
    per-bin (median x, median y, 25th, 75th pctile y) for bins with at least
    min_per_bin points. Edges must be computed once per variable (pooled
    across areas) and passed in, so every area's line uses the exact same
    bin boundaries -- otherwise per-area quantile edges differ and the same
    nominal bin covers a different MI range in each area/panel, making the
    lines impossible to compare."""
    if len(edges) < 3:
        return None
    bin_idx = np.clip(np.digitize(x, edges[1:-1], right=False), 0, len(edges) - 2)
    xs, meds, lo, hi = [], [], [], []
    for b in range(len(edges) - 1):
        m = bin_idx == b
        if m.sum() < min_per_bin:
            continue
        xs.append(np.median(x[m]))
        meds.append(np.median(y[m]))
        lo.append(np.percentile(y[m], 25))
        hi.append(np.percentile(y[m], 75))
    if len(xs) < 2:
        return None
    return np.array(xs), np.array(meds), np.array(lo), np.array(hi)


def _compute_mi_ai_binned(all_cells, records, n_bins=5):
    """Per-(area, variable) binned (median, IQR) AI-vs-CV-MI stats, using a
    single set of equal-width bin edges pooled across ALL areas AND ALL
    variables. One shared edge set (rather than one per area or one per
    variable) is what makes every line in every panel -- regardless of
    which figure layout draws it -- directly comparable: the same bin index
    always covers the same CV-MI range everywhere."""
    joined  = _join_mi_ai_cells(all_cells, records)
    by_area = {a: [] for a in _IMP_REGION_ORDER}
    for c, r in joined:
        if c['area'] in by_area:
            by_area[c['area']].append((c, r))

    raw = {}
    pooled_x = []
    for area in _IMP_REGION_ORDER:
        pairs = by_area[area]
        if not pairs:
            continue
        for v in _POS_VAR_KEYS:
            x, y, _, _ = _mi_ai_xy(pairs, v)
            raw[(area, v)] = (x, y)
            pooled_x.append(x)

    if not pooled_x:
        return {}, None
    pooled_x = np.concatenate(pooled_x)
    edges = np.linspace(pooled_x.min(), pooled_x.max(), n_bins + 1)

    binned = {}
    for key, (x, y) in raw.items():
        b = _binned_median_iqr(x, y, edges)
        if b is not None:
            binned[key] = b
    return binned, edges


def make_mi_ai_running_median_svg(all_cells, records, out_dir, n_bins=5):
    """(13b) Running-median view of the same CV-MI-vs-ablation-index
    relationship as make_mi_ai_percell_scatter_svg, collapsed across the
    per-cell scatter's overplotting: one panel per position variable, one
    line per area, line = per-bin median AI, band = per-bin 25th-75th
    pctile. Same AI clip-floor/ceiling exclusion. Bins are equal-width and
    shared across every area and variable (see _compute_mi_ai_binned)."""
    binned, edges = _compute_mi_ai_binned(all_cells, records, n_bins=n_bins)
    if edges is None:
        print('No MI/AI data -- skipping running-median figure.')
        return

    fig, axes = plt.subplots(1, len(_POS_VAR_KEYS), figsize=_scaled(7.2, 2.4),
                              constrained_layout=True, sharex=True, sharey=True)

    for vi, v in enumerate(_POS_VAR_KEYS):
        ax = axes[vi]
        for area in _IMP_REGION_ORDER:
            b = binned.get((area, v))
            if b is None:
                continue
            xs, meds, lo, hi = b
            color = COLORS.get(area, '#888888')
            ax.plot(xs, meds, color=color, lw=1.5, marker='o', markersize=2.5, zorder=3)
            ax.fill_between(xs, lo, hi, color=color, alpha=0.15, zorder=1)
        ax.set_title(v, fontsize=7)
        ax.set_xlabel('CV MI', fontsize=6)
        if vi == 0:
            ax.set_ylabel('Ablation index\n(median, IQR band)', fontsize=6)
        ax.tick_params(labelsize=5)

    area_handles = [Line2D([0], [0], color=COLORS.get(a, '#888888'), lw=1.5,
                            marker='o', markersize=3, label=a) for a in _IMP_REGION_ORDER]
    fig.legend(handles=area_handles, title='Area', fontsize=5, title_fontsize=5,
              loc='center left', bbox_to_anchor=(1.0, 0.5))

    fig.suptitle('CV MI vs. ablation index: per-area running median\n'
                 '(equal-width bins shared across areas+variables; band = IQR; '
                 'AI clip-floor/ceiling cells excluded)',
                 fontsize=8)

    path = os.path.join(out_dir, 'mi_ai_running_median.svg')
    _save_svg_png(fig, path)


def make_mi_ai_running_median_by_area_svg(all_cells, records, out_dir, n_bins=5):
    """(13c) Same data and bins as make_mi_ai_running_median_svg, transposed:
    one panel per area, one line per position variable. Useful for asking
    "within this area, which variables show a clean MI-AI relationship?"
    rather than "for this variable, which areas do?"."""
    binned, edges = _compute_mi_ai_binned(all_cells, records, n_bins=n_bins)
    if edges is None:
        print('No MI/AI data -- skipping by-area running-median figure.')
        return

    fig, axes = plt.subplots(1, len(_IMP_REGION_ORDER), figsize=_scaled(9.0, 2.4),
                              constrained_layout=True, sharex=True, sharey=True)

    for ai_, area in enumerate(_IMP_REGION_ORDER):
        ax = axes[ai_]
        for v in _POS_VAR_KEYS:
            b = binned.get((area, v))
            if b is None:
                continue
            xs, meds, lo, hi = b
            color = _VAR_COLORS.get(v, '#888888')
            ax.plot(xs, meds, color=color, lw=1.5, marker=_POS_MARKERS[v],
                    markersize=3, zorder=3)
            ax.fill_between(xs, lo, hi, color=color, alpha=0.15, zorder=1)
        ax.set_title(area, fontsize=7)
        ax.set_xlabel('CV MI', fontsize=6)
        if ai_ == 0:
            ax.set_ylabel('Ablation index\n(median, IQR band)', fontsize=6)
        ax.tick_params(labelsize=5)

    var_handles = [Line2D([0], [0], color=_VAR_COLORS.get(v, '#888888'), lw=1.5,
                           marker=_POS_MARKERS[v], markersize=4, label=v) for v in _POS_VAR_KEYS]
    fig.legend(handles=var_handles, title='Variable', fontsize=5, title_fontsize=5,
              loc='center left', bbox_to_anchor=(1.0, 0.5))

    fig.suptitle('CV MI vs. ablation index: per-variable running median\n'
                 '(equal-width bins shared across areas+variables; band = IQR; '
                 'AI clip-floor/ceiling cells excluded)',
                 fontsize=8)

    path = os.path.join(out_dir, 'mi_ai_running_median_by_area.svg')
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
    _res_pos = test_eye_vs_head_tuning_corr(all_cells, label='position')
    _res_vel = test_eye_vs_head_tuning_corr(
        all_cells, eye_keys=['dTheta', 'dPhi'], head_keys=['gyro_y', 'gyro_x'], label='velocity')
    _lo_pos, _hi_pos = _eye_head_tuning_corr_autolim(_res_pos)
    _lo_vel, _hi_vel = _eye_head_tuning_corr_autolim(_res_vel)
    _shared_corr_ylim = (min(_lo_pos, _lo_vel), max(_hi_pos, _hi_vel))
    make_eye_head_tuning_corr_svg(all_cells, args.out_dir, res=_res_pos, ylim=_shared_corr_ylim)
    make_eye_head_tuning_corr_svg(
        all_cells, args.out_dir, eye_keys=['dTheta', 'dPhi'], head_keys=['gyro_y', 'gyro_x'],
        label='velocity', fname='eye_head_tuning_corr_velocity.svg',
        res=_res_vel, ylim=_shared_corr_ylim)
    make_eye_head_tuning_corr_category_bar_svg(all_cells, args.out_dir)
    make_eye_head_tuning_corr_category_bar_svg(
        all_cells, args.out_dir, eye_keys=['dTheta', 'dPhi'], head_keys=['gyro_y', 'gyro_x'],
        label='velocity', fname='eye_head_tuning_corr_category_velocity.svg')

    _inverted_vel_dir = quantify_inverted_velocity_motion_direction(all_cells)
    make_inverted_velocity_motion_direction_svg(_inverted_vel_dir, args.out_dir)

    # ---- SVG exports ----
    make_combined_overview_svg(all_cells, args.out_dir)
    make_overview_mi_ldi_boxstrip_svg(all_cells, args.out_dir)
    make_overview_mi_ldi_boxstrip_svg(all_cells, args.out_dir,
                                       variables=SPEED_VARIABLES, file_suffix='_speed')
    make_example_tuning_svgs(all_cells, args.out_dir)
    make_example_tuning_speed_svgs(all_cells, args.out_dir)

    if RUN_OCCUPANCY_ANALYSIS:
        occ_pvals = collect_occupancy_pvalues(args.base_dir)
        make_occupancy_pvalue_histogram_svg(occ_pvals, args.out_dir)

        occ_samples = collect_occupancy_sample_traces(args.base_dir)
        make_occupancy_sample_traces_svg(occ_samples, args.out_dir)

        occ_speed_samples = collect_occupancy_speed_sample_traces(args.base_dir)
        make_occupancy_speed_sample_traces_svg(occ_speed_samples, args.out_dir)
    else:
        print('RUN_OCCUPANCY_ANALYSIS=False -- skipping occupancy analysis.')

    records = []
    if os.path.exists(args.pooled_glm):
        records = _load_importance_cells(args.pooled_glm)

        cross_rows, cross_example = _load_cross_condition_generalization(args.pooled_glm)
        make_cross_condition_generalization_svg(cross_rows, cross_example, args.out_dir)

    if records:
        make_r2_histogram_svg(records, args.out_dir)
        make_r2_pass_fraction_svg(records, args.out_dir)

        if APPLY_R2_THRESHOLD:
            n_before = len(records)
            records = [r for r in records
                       if np.isfinite(r['full_r2']) and r['full_r2'] > R2_THRESHOLD]
            print(f'R^2 threshold filter ON: kept {len(records)}/{n_before} cells '
                  f'(full_r2 > {R2_THRESHOLD}).')
        else:
            print('R^2 threshold filter OFF: using all cells (original behavior).')

    make_importance_svg(args.out_dir, records=records)

    if records:
        make_ablation_heatmap_svg(records, args.out_dir)
        make_slope_graph_svg(records, args.out_dir)
        for layout in ('grouped', 'faceted'):
            for scope in ('velocity', 'all8'):
                make_fold_change_svg(records, args.out_dir, layout=layout, scope=scope)
        pts = make_ldi_vs_ai_scatter_svg(records, all_cells, args.out_dir)
        make_ldi_vs_ai_scatter_fit_svg(records, all_cells, args.out_dir)
        _ldi_stats_pos = _collect_ldi_mi_stats(all_cells, var_keys=_POS_VAR_KEYS)
        _ldi_stats_vel = _collect_ldi_mi_stats(all_cells, var_keys=_VEL_VAR_KEYS)
        _lo_pos, _hi_pos = _ldi_beeswarm_autolim(_ldi_stats_pos, _POS_VAR_KEYS)
        _lo_vel, _hi_vel = _ldi_beeswarm_autolim(_ldi_stats_vel, _VEL_VAR_KEYS)
        _shared_ldi_ylim = (min(_lo_pos, _lo_vel), max(_hi_pos, _hi_vel))
        make_ldi_beeswarm_svg(all_cells, args.out_dir, ylim=_shared_ldi_ylim, ldi_stats=_ldi_stats_pos)
        make_ldi_beeswarm_svg(all_cells, args.out_dir, var_keys=_VEL_VAR_KEYS,
                               label='velocity', fname='ldi_beeswarm_velocity.svg',
                               ylim=_shared_ldi_ylim, ldi_stats=_ldi_stats_vel)
        make_ldi_swarm_points_svg(all_cells, args.out_dir)
        make_mi_vs_ldi_scatter_svg(all_cells, args.out_dir)
        make_mi_vs_ai_scatter_svg(records, all_cells, args.out_dir)
        make_diverging_dotplot_svg(records, args.out_dir)
        make_diff_histogram_svg(records, args.out_dir)
        make_concordance_bar_svg(pts, args.out_dir)
        make_concordance_bar_percell_svg(all_cells, records, args.out_dir)

        make_ai_ci_bar_svg(records, args.out_dir)
        make_pm_velocity_light_dark_bar_svg(records, args.out_dir)
        make_ai_ridgeline_svg(records, args.out_dir)
        make_velocity_fold_trend_svg(records, args.out_dir)
        make_mi_ai_percell_scatter_svg(all_cells, records, args.out_dir)
        make_mi_ai_running_median_svg(all_cells, records, args.out_dir)
        make_mi_ai_running_median_by_area_svg(all_cells, records, args.out_dir)
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
