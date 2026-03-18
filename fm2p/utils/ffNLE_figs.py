"""
ffNLE_figs.py — Summary figures for ffNLE (freely-moving nonlinear encoding) models.

Reads from pooled data structure produced by make_pooled_dataset() in
merge_animal_essentials.py.  All figures are driven by the updated model key
naming convention:

    full_r2, full_corrs
    full_trainDark_testDark_r2/corrs/importance_{var}
    full_trainLight_testLight_r2/corrs/importance_{var}

Visual-area boundaries come from vfs_contours.json (VFS space, 0-400 px).

Usage (CLI):
    python -m fm2p.utils.ffNLE_figs /path/to/pooled.h5 /path/to/save_dir

Importable API:
    from fm2p.utils.ffNLE_figs import main, load_pooled_cells
    cells = load_pooled_cells(fm2p.read_h5(pooled_path))
    main(pooled_path, save_dir)
"""

import os
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap

from .cmap import make_parula
from .files import read_h5


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GOODRED = '#D96459'

_EARTH_HEX = [
    '#2ECC71', '#82E0AA',   # Green  — theta, dTheta
    '#FF9800', '#FFCC80',   # Orange — phi,   dPhi
    '#03A9F4', '#81D4FA',   # Blue   — pitch, dPitch
    '#9C27B0', '#E1BEE7',   # Purple — roll,  dRoll
    '#FFEB3B', '#FFF59D',   # Yellow — yaw,   dYaw
]

_VAR_ORDER = ['theta', 'dTheta', 'phi', 'dPhi',
              'pitch', 'gyro_y', 'roll', 'gyro_x', 'yaw', 'gyro_z']
_VAR_NICE  = ['θ', 'dθ', 'φ', 'dφ',
              'pitch', 'dPitch', 'roll', 'dRoll', 'yaw', 'dYaw']

AREA_IDS = {'V1': 5, 'RL': 2, 'AM': 3, 'PM': 4, 'AL': 7, 'LM': 8, 'P': 9}
AREA_COLORS = {
    'V1': '#2ECC71',
    'RL': '#FF9800',
    'AM': '#03A9F4',
    'PM': '#9C27B0',
    'AL': '#CDDC39',
    'LM': '#E74C3C',
    'P':  '#7F8C8D',
}

VFS_CONTOURS_PATH = os.path.join(os.path.dirname(__file__), 'vfs_contours.json')


# ---------------------------------------------------------------------------
# Colormap helpers (public — kept for external callers)
# ---------------------------------------------------------------------------

def make_earth_tones():
    """Custom categorical earth-tone colormap with 10 colours in pairs."""
    rgb = [tuple(int(h.lstrip('#')[i:i+2], 16) / 255.0
                 for i in (0, 2, 4)) for h in _EARTH_HEX]
    return LinearSegmentedColormap.from_list('earth_tones', rgb, N=10)


def get_equally_spaced_colormap_values(colormap_name, num_values):
    if not isinstance(num_values, int) or num_values <= 0:
        raise ValueError("num_values must be a positive integer.")
    if colormap_name == 'parula':
        cmap = make_parula()
    elif colormap_name == 'earth_tones':
        cmap = make_earth_tones()
    else:
        cmap = cm.get_cmap(colormap_name)
    return [cmap(p) for p in np.linspace(0, 1, num_values)]


def _earth_colors():
    return get_equally_spaced_colormap_values('earth_tones', 10)


# kept for backward compat
goodred = _GOODRED


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def calculate_r2_numpy(true, pred):
    true, pred = np.array(true), np.array(pred)
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return 0.0 if ss_tot == 0 else 1 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pooled_cells(pooled_data, r2_threshold=0.0):
    """Collect all cells from pooled data into flat arrays.

    Parameters
    ----------
    pooled_data : dict
        Nested dict loaded via fm2p.read_h5(pooled_path).
        Expected layout: pooled_data[animal]['messentials'][pos] containing
        'model', 'rdata', 'visual_area_id', 'vfs_cell_pos'.
    r2_threshold : float
        Minimum full_r2 to retain a cell (default 0.0 = all cells).

    Returns
    -------
    cells : dict
        Arrays of length N_cells:
          'animal'         — list of str
          'pos'            — list of str
          'cell_idx'       — int array (within-position index)
          'full_r2'        — float array
          'full_corrs'     — float array
          'dark_r2'        — float array (NaN when condition absent)
          'dark_corrs'     — float array
          'light_r2'       — float array
          'light_corrs'    — float array
          'full_imp'       — (N, 10) float array
          'dark_imp'       — (N, 10) float array
          'light_imp'      — (N, 10) float array
          'visual_area_id' — int array
          'vfs_cell_pos'   — (N, 2) float array
          '_rdata_refs'    — list of (rdata_dict, local_cell_idx)
    """
    lists = {k: [] for k in [
        'animal', 'pos', 'cell_idx',
        'full_r2', 'full_corrs',
        'dark_r2', 'dark_corrs',
        'light_r2', 'light_corrs',
        'full_imp', 'dark_imp', 'light_imp',
        'visual_area_id', 'vfs_cell_pos',
        '_rdata_refs',
    ]}

    for animal in sorted(pooled_data.keys()):
        adat = pooled_data[animal]
        if 'messentials' not in adat:
            continue
        me = adat['messentials']
        for pos in sorted(me.keys()):
            pdat = me[pos]
            model = pdat.get('model', {})
            rdata = pdat.get('rdata', {})

            if 'full_r2' not in model:
                continue
            r2_arr = np.atleast_1d(np.asarray(model['full_r2'], dtype=float))
            n = len(r2_arr)

            def _arr(key, fallback=None):
                if key in model:
                    v = np.atleast_1d(np.asarray(model[key], dtype=float))
                    # pad/trim to n
                    out = np.full(n, np.nan)
                    m = min(len(v), n)
                    out[:m] = v[:m]
                    return out
                if fallback is not None:
                    return np.full(n, fallback)
                return np.full(n, np.nan)

            def _imp(prefix):
                mat = np.full((n, len(_VAR_ORDER)), np.nan)
                for vi, var in enumerate(_VAR_ORDER):
                    k = f'{prefix}importance_{var}'
                    if k in model:
                        v = np.atleast_1d(np.asarray(model[k], dtype=float))
                        m = min(len(v), n)
                        mat[:m, vi] = v[:m]
                return mat

            corrs   = _arr('full_corrs')
            dark_r2 = _arr('full_trainDark_testDark_r2')
            dark_cc = _arr('full_trainDark_testDark_corrs')
            lite_r2 = _arr('full_trainLight_testLight_r2')
            lite_cc = _arr('full_trainLight_testLight_corrs')

            full_imp  = _imp('full_')
            dark_imp  = _imp('full_trainDark_testDark_')
            light_imp = _imp('full_trainLight_testLight_')

            area_id = np.zeros(n, dtype=int)
            raw_aid = pdat.get('visual_area_id', None)
            if raw_aid is not None:
                raw_aid = np.atleast_1d(np.asarray(raw_aid, dtype=int))
                m = min(len(raw_aid), n)
                area_id[:m] = raw_aid[:m]

            vfs_pos = np.full((n, 2), np.nan)
            raw_vp = pdat.get('vfs_cell_pos', None)
            if raw_vp is not None:
                raw_vp = np.asarray(raw_vp, dtype=float)
                if raw_vp.ndim == 1:
                    raw_vp = raw_vp.reshape(-1, 2)
                m = min(len(raw_vp), n)
                vfs_pos[:m] = raw_vp[:m]

            for ci in range(n):
                if r2_arr[ci] < r2_threshold:
                    continue
                lists['animal'].append(animal)
                lists['pos'].append(pos)
                lists['cell_idx'].append(ci)
                lists['full_r2'].append(r2_arr[ci])
                lists['full_corrs'].append(corrs[ci])
                lists['dark_r2'].append(dark_r2[ci])
                lists['dark_corrs'].append(dark_cc[ci])
                lists['light_r2'].append(lite_r2[ci])
                lists['light_corrs'].append(lite_cc[ci])
                lists['full_imp'].append(full_imp[ci])
                lists['dark_imp'].append(dark_imp[ci])
                lists['light_imp'].append(light_imp[ci])
                lists['visual_area_id'].append(area_id[ci])
                lists['vfs_cell_pos'].append(vfs_pos[ci])
                lists['_rdata_refs'].append((rdata, ci))

    cells = {}
    for k, v in lists.items():
        if k in ('animal', 'pos', '_rdata_refs'):
            cells[k] = v
        else:
            cells[k] = np.array(v)

    print(f"Loaded {len(cells['full_r2'])} cells "
          f"(threshold full_r2 >= {r2_threshold}).")
    return cells


def load_contours():
    """Load VFS area contours from vfs_contours.json (VFS space 0-400)."""
    if not os.path.exists(VFS_CONTOURS_PATH):
        print(f"Warning: {VFS_CONTOURS_PATH} not found — no contours available.")
        return {}
    with open(VFS_CONTOURS_PATH, 'r') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Summary plots
# ---------------------------------------------------------------------------

def plot_r2_comparison(cells, pdf):
    """Light vs dark full-model R² scatter."""
    fig, ax = plt.subplots(1, 1, dpi=300, figsize=(2.5, 2.5))
    lim = [-0.3, 0.7]
    ax.plot(lim, lim, ls='--', color='tab:red', alpha=0.4, lw=1)
    mask = ~np.isnan(cells['light_r2']) & ~np.isnan(cells['dark_r2'])
    ax.scatter(cells['light_r2'][mask], cells['dark_r2'][mask],
               s=1, c='k', alpha=0.5)
    ax.set_xlabel('light R²')
    ax.set_ylabel('dark R²')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.axis('equal')
    ax.set_title(f'full model (N={mask.sum()})')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_importance_scatter(cells, pdf):
    """Per-feature scatter: dark importance vs light importance."""
    colors = _earth_colors()
    fig, axs = plt.subplots(2, 5, dpi=300, figsize=(9, 4), constrained_layout=True)
    axs = axs.flatten()

    for vi, nice in enumerate(_VAR_NICE):
        ax = axs[vi]
        x = cells['light_imp'][:, vi]
        y = cells['dark_imp'][:, vi]
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() > 1:
            lo = min(np.nanpercentile(x[mask], 1), np.nanpercentile(y[mask], 1))
            hi = max(np.nanpercentile(x[mask], 99), np.nanpercentile(y[mask], 99))
            ax.plot([lo, hi], [lo, hi], ls='--', color='tab:red', alpha=0.4, lw=1)
            ax.scatter(x[mask], y[mask], s=1, color=colors[vi], alpha=0.5)
            ax.set_xlim([lo, hi]); ax.set_ylim([lo, hi])
        ax.set_xlabel('light imp.', fontsize=6)
        ax.set_ylabel('dark imp.', fontsize=6)
        ax.set_title(f'{nice}  (N={mask.sum()})', fontsize=7)

    fig.suptitle('Feature importance: light vs dark (same-condition)')
    pdf.savefig(fig)
    plt.close(fig)


def plot_importance_histograms(cells, pdf):
    """Overlapping histograms of light and dark importances per feature."""
    fig, axs = plt.subplots(2, 5, figsize=(9, 4), dpi=300, constrained_layout=True)
    axs = axs.flatten()

    for vi, nice in enumerate(_VAR_NICE):
        ax = axs[vi]
        lv = cells['light_imp'][:, vi]
        dv = cells['dark_imp'][:, vi]
        mask = ~np.isnan(lv) & ~np.isnan(dv)
        if mask.sum() > 1:
            all_v = np.concatenate([lv[mask], dv[mask]])
            bins = np.linspace(np.nanpercentile(all_v, 1),
                               np.nanpercentile(all_v, 99), 30)
            ax.hist(lv[mask], bins=bins, alpha=0.5, label='Light',
                    color='tab:orange', density=True)
            ax.hist(dv[mask], bins=bins, alpha=0.5, label='Dark',
                    color='tab:blue', density=True)
            ax.axvline(np.nanmean(lv[mask]), color='tab:orange', ls='--', lw=1)
            ax.axvline(np.nanmean(dv[mask]), color='tab:blue',   ls='--', lw=1)
        ax.set_title(nice, fontsize=7)
        ax.set_xlabel('importance', fontsize=6)
        if vi == 0:
            ax.legend(fontsize=5)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_per_area_importance(cells, pdf):
    """Violin plots of feature importances per visual area for each condition."""
    present_areas = [a for a in AREA_IDS
                     if np.any(cells['visual_area_id'] == AREA_IDS[a])]

    for imp_key, title in [
        ('light_imp', 'Light condition — importance per area'),
        ('dark_imp',  'Dark condition — importance per area'),
    ]:
        fig, axs = plt.subplots(2, 5, figsize=(11, 5), dpi=150,
                                constrained_layout=True)
        axs = axs.flatten()

        for vi, nice in enumerate(_VAR_NICE):
            ax = axs[vi]
            data_per, labels = [], []
            for aname in present_areas:
                aid = AREA_IDS[aname]
                mask = cells['visual_area_id'] == aid
                imp = cells[imp_key][mask, vi]
                imp = imp[~np.isnan(imp)]
                if len(imp) >= 3:
                    data_per.append(imp)
                    labels.append(f'{aname}\n(N={len(imp)})')

            if data_per:
                vp = ax.violinplot(data_per, showmedians=True,
                                   showextrema=False)
                for i, body in enumerate(vp['bodies']):
                    body.set_facecolor(
                        AREA_COLORS.get(present_areas[i], '#999'))
                    body.set_alpha(0.7)
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels, fontsize=5, rotation=45)
                ax.axhline(0, color='k', ls='--', lw=0.5, alpha=0.5)

            ax.set_title(nice, fontsize=7)
            ax.set_ylabel('imp.', fontsize=6)

        fig.suptitle(title)
        pdf.savefig(fig)
        plt.close(fig)


def plot_area_r2_comparison(cells, pdf):
    """Bar chart of mean R² per visual area for light and dark conditions."""
    area_names = list(AREA_IDS.keys())
    fig, axs = plt.subplots(1, 2, figsize=(6, 3), dpi=200,
                            constrained_layout=True)

    for ax, cond, key in [(axs[0], 'Light', 'light_r2'),
                          (axs[1], 'Dark',  'dark_r2')]:
        means, sems, ns, names = [], [], [], []
        for aname in area_names:
            aid = AREA_IDS[aname]
            vals = cells[key][cells['visual_area_id'] == aid]
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                means.append(np.mean(vals))
                sems.append(np.std(vals) / np.sqrt(len(vals)))
                ns.append(len(vals))
                names.append(aname)

        x = np.arange(len(names))
        colors_bar = [AREA_COLORS.get(n, '#999') for n in names]
        ax.bar(x, means, color=colors_bar, alpha=0.85)
        ax.errorbar(x, means, yerr=sems, fmt='none', color='k',
                    capsize=3, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f'{n}\n(N={ns[i]})' for i, n in enumerate(names)],
            fontsize=6, rotation=45)
        ax.set_ylabel('mean R²')
        ax.set_title(f'{cond} condition R²')
        ax.axhline(0, color='k', ls='--', lw=0.5)

    pdf.savefig(fig)
    plt.close(fig)


def plot_mean_importance_heatmap(cells, pdf):
    """Heatmap (areas × features) of mean importance, dark and light."""
    area_names = list(AREA_IDS.keys())
    nv = len(_VAR_ORDER)

    for imp_key, title in [
        ('light_imp', 'Mean light importance by area'),
        ('dark_imp',  'Mean dark importance by area'),
    ]:
        mat = np.full((len(area_names), nv), np.nan)
        ns  = np.zeros_like(mat, dtype=int)

        for ai, aname in enumerate(area_names):
            aid = AREA_IDS[aname]
            mask = cells['visual_area_id'] == aid
            for vi in range(nv):
                imp = cells[imp_key][mask, vi]
                imp = imp[~np.isnan(imp)]
                if len(imp) > 0:
                    mat[ai, vi] = np.nanmean(imp)
                    ns[ai, vi]  = len(imp)

        vmax = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 1.0
        fig, ax = plt.subplots(figsize=(9, 3), dpi=200)
        im = ax.imshow(mat, aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label='mean importance (drop in R²)')
        ax.set_xticks(range(nv))
        ax.set_xticklabels(_VAR_NICE, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(len(area_names)))
        ax.set_yticklabels(area_names, fontsize=7)
        ax.set_title(title)

        for ai in range(len(area_names)):
            for vi in range(nv):
                if not np.isnan(mat[ai, vi]):
                    ax.text(vi, ai,
                            f'{mat[ai,vi]:.3f}\n(N={ns[ai,vi]})',
                            ha='center', va='center', fontsize=4)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def plot_area_cell_scatter(cells, contours, save_dir):
    """Scatter all cells in VFS space, coloured by area, with area contours."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=200)

    # Draw area boundaries
    for area_name, pts in contours.items():
        pts_arr = np.array(pts)
        if pts_arr.ndim == 2 and pts_arr.shape[1] == 2:
            closed = np.vstack([pts_arr, pts_arr[:1]])
            ax.plot(closed[:, 0], closed[:, 1], color='k', lw=0.8)
            cx, cy = pts_arr.mean(axis=0)
            ax.text(cx, cy, area_name, ha='center', va='center',
                    fontsize=6, color=AREA_COLORS.get(area_name, '#555'))

    pos      = cells['vfs_cell_pos']
    area_ids = cells['visual_area_id']
    assigned = np.zeros(len(area_ids), dtype=bool)

    for aname, aid in AREA_IDS.items():
        mask = area_ids == aid
        assigned |= mask
        if mask.sum() > 0:
            ax.scatter(pos[mask, 0], pos[mask, 1], s=1,
                       color=AREA_COLORS.get(aname, '#999'),
                       alpha=0.5, label=f'{aname} (N={mask.sum()})')

    if (~assigned).sum() > 0:
        ax.scatter(pos[~assigned, 0], pos[~assigned, 1],
                   s=0.5, color='gray', alpha=0.3, label='unassigned')

    ax.set_xlabel('VFS x (px)')
    ax.set_ylabel('VFS y (px)')
    ax.legend(fontsize=5, markerscale=4, loc='upper left')
    ax.set_title(f'All cells in VFS space (N={len(pos)})')
    fig.tight_layout()

    out_path = os.path.join(save_dir, 'area_cell_scatter.svg')
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_population_r2_histogram(cells, pdf):
    """Distribution of full R² across all cells."""
    fig, ax = plt.subplots(figsize=(3, 2.5), dpi=300)
    r2 = cells['full_r2']
    r2 = r2[~np.isnan(r2)]
    ax.hist(r2, bins=50, color='k', alpha=0.7)
    ax.axvline(np.median(r2), color=_GOODRED, ls='--', lw=1,
               label=f'median={np.median(r2):.3f}')
    ax.set_xlabel('full model R²')
    ax.set_ylabel('cells')
    ax.set_title(f'Population R² (N={len(r2)})')
    ax.legend(fontsize=6)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)



def main(pooled_path, save_dir, r2_threshold=0.0):
    """Generate summary figures from pooled ffNLE data.

    Parameters
    ----------
    pooled_path : str
        Path to pooled data h5 (output of make_pooled_dataset).
    save_dir : str
        Directory to write output PDF and SVGs.
    r2_threshold : float
        Minimum full_r2 to include a cell (default 0.0 = all cells).
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading {pooled_path} ...")
    pooled_data = read_h5(pooled_path)

    cells    = load_pooled_cells(pooled_data, r2_threshold=r2_threshold)
    contours = load_contours()

    pdf_path = os.path.join(save_dir, 'ffNLE_summary.pdf')
    print(f"Writing {pdf_path} ...")

    with PdfPages(pdf_path) as pdf:
        plot_population_r2_histogram(cells, pdf)
        plot_r2_comparison(cells, pdf)
        plot_importance_scatter(cells, pdf)
        plot_importance_histograms(cells, pdf)
        plot_area_r2_comparison(cells, pdf)
        plot_per_area_importance(cells, pdf)
        plot_mean_importance_heatmap(cells, pdf)

    plot_area_cell_scatter(cells, contours, save_dir)

    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ffNLE summary figures')
    parser.add_argument('pooled_path', help='Path to pooled data h5 file')
    parser.add_argument('save_dir',    help='Output directory')
    parser.add_argument('--r2_threshold', type=float, default=0.0,
                        help='Minimum full_r2 to include (default 0.0)')
    args = parser.parse_args()
    main(args.pooled_path, args.save_dir, r2_threshold=args.r2_threshold)
