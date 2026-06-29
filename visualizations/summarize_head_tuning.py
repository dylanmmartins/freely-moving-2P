

if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

import argparse
import glob
import os
import re


import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib as mpl
mpl.rcParams['axes.spines.top']   = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['font.size']    = 7

from ..fm2p.utils.paths import find


DEFAULT_POOLED  = '/home/dylan/Fast2/pooled_260407a.h5'
DEFAULT_BASE    = '/home/dylan/Storage/freely_moving_data/_V1PPC'
DEFAULT_OUT_DIR = '.'
MIN_CELLS_AREA  = 5
TOP_N_HEATMAP   = 100
TOP_N_PER_AREA  = 24

ID_TO_NAME   = {5: 'V1', 2: 'RL', 3: 'AM', 4: 'PM', 10: 'A', 7: 'AL', 8: 'LM', 9: 'P'}
REGION_ORDER = ['V1', 'RL', 'AM', 'PM', 'A', 'AL', 'LM', 'P']
COLORS = {
    'V1': '#1B9E77', 'RL': '#D95F02', 'AM': '#7570B3', 'PM': '#E7298A',
    'A':  '#1A5A34', 'AL': '#E6AB02', 'LM': '#A6761D', 'P':  '#666666',
}


def _build_pooled_lookup(pooled_path: str) -> dict:

    lookup = {}
    with h5py.File(pooled_path, 'r') as pool:
        for animal in pool.keys():
            if animal == 'uniref':
                continue
            if 'messentials' not in pool[animal]:
                continue
            for pos in sorted(pool[animal]['messentials'].keys()):
                if not pos.startswith('pos'):
                    continue
                grp = pool[animal]['messentials'][pos]
                if 'visual_area_id' not in grp:
                    continue
                va_ids = grp['visual_area_id'][:]
                lookup[(animal, pos, int(len(va_ids)))] = va_ids.copy()
    return lookup


def _preproc_for_revcorr(revcorr_path: str):

    rcdir = os.path.dirname(revcorr_path)
    hits = [
        p for p in glob.glob(os.path.join(rcdir, '*preproc.h5'))
        if 'boundary' not in p and not os.path.basename(p).startswith('sn')
    ]
    if not hits:
        return None
    return sorted(hits)[-1]


def _match_to_pooled(revcorr_path: str, pooled_lookup: dict):

    preproc = _preproc_for_revcorr(revcorr_path)
    if preproc is None:
        return None

    n_cells = None
    try:
        with h5py.File(preproc, 'r') as f:
            for key in ('norm_spikes', 'norm_dFF'):
                if key in f:
                    n_cells = int(f[key].shape[0])
                    break
    except Exception:
        return None
    if n_cells is None:
        return None

    m_animal = re.search(r'(DMM\d+)', revcorr_path)
    m_pos    = re.search(r'(pos\d+)', revcorr_path)
    if not m_animal or not m_pos:
        return None
    animal = m_animal.group(1)
    pos    = m_pos.group(1)

    key = (animal, pos, n_cells)
    if key in pooled_lookup:
        return animal, pos, pooled_lookup[key]
    return None


def _load_yaw_occupancy(preproc_path: str) -> dict:

    OCC_BINS = np.linspace(0, 360, 37)
    try:
        with h5py.File(preproc_path, 'r') as f:
            yaw = f['head_yaw_deg'][()].astype(float)
            ltdk = f['ltdk_state_vec'][()].astype(bool) if 'ltdk_state_vec' in f else None
    except Exception:
        return None

    n_total  = len(yaw)
    n_nan    = int(np.sum(np.isnan(yaw)))
    yaw_good = yaw[~np.isnan(yaw)]

    counts_all, _ = np.histogram(yaw_good, bins=OCC_BINS)

    counts_light = counts_dark = None
    if ltdk is not None:

        n = min(len(yaw), len(ltdk))
        yaw_t, ltdk_t = yaw[:n], ltdk[:n]
        yaw_l = yaw_t[ltdk_t & ~np.isnan(yaw_t)]
        yaw_d = yaw_t[~ltdk_t & ~np.isnan(yaw_t)]
        counts_light, _ = np.histogram(yaw_l, bins=OCC_BINS)
        counts_dark,  _ = np.histogram(yaw_d, bins=OCC_BINS)

    return {
        'occ_bins':     OCC_BINS,
        'counts_all':   counts_all,
        'counts_light': counts_light,
        'counts_dark':  counts_dark,
        'n_total':      n_total,
        'n_nan':        n_nan,
        'yaw_p2':       float(np.nanpercentile(yaw, 2)),
        'yaw_p98':      float(np.nanpercentile(yaw, 98)),
        'saved_bin_lo': None,
        'saved_bin_hi': None,
    }


def collect_data(pooled_path: str, base_dir: str):

    pooled_lookup = _build_pooled_lookup(pooled_path)
    revcorr_files = find('eyehead_revcorrs_v06.h5', base_dir)
    print(f'Found {len(revcorr_files)} eyehead_revcorrs_v06.h5 files.')

    all_cells  = []
    recordings = []

    for rcf in sorted(revcorr_files):
        try:
            with h5py.File(rcf, 'r') as f:
                if 'yaw_l_rel' not in f:
                    continue
                yaw_l_rel    = f['yaw_l_rel'   ][()].astype(float)
                yaw_l_isrel  = f['yaw_l_isrel' ][()].astype(bool)
                yaw_1dtuning = f['yaw_1dtuning'][()].astype(float)
                yaw_1derr    = f['yaw_1derr'   ][()].astype(float)
                yaw_1dbins   = f['yaw_1dbins'  ][()].astype(float)
                yaw_d_rel    = f['yaw_d_rel'][()].astype(float) if 'yaw_d_rel' in f \
                               else np.full_like(yaw_l_rel, np.nan)
        except Exception as e:
            print(f'  Read error {rcf}: {e}')
            continue

        match = _match_to_pooled(rcf, pooled_lookup)
        if match is None:
            print(f'  No pooled match: {rcf}')
            continue

        animal, pos, va_ids = match
        n_cells = len(va_ids)

        if len(yaw_l_rel) != n_cells:
            print(f'  Cell count mismatch ({len(yaw_l_rel)} vs {n_cells}): {rcf}')
            continue

        named = {ID_TO_NAME[i] for i in np.unique(va_ids) if i in ID_TO_NAME}
        print(f'  {animal}/{pos}: {n_cells} cells  areas={sorted(named)}')

        preproc_path = _preproc_for_revcorr(rcf)
        occ = _load_yaw_occupancy(preproc_path) if preproc_path else None
        if occ is not None:
            occ['animal']       = animal
            occ['pos']          = pos
            occ['saved_bin_lo'] = float(yaw_1dbins[0])
            occ['saved_bin_hi'] = float(yaw_1dbins[-1])
            recordings.append(occ)

        for ci in range(n_cells):
            area_id = int(va_ids[ci])
            if area_id not in ID_TO_NAME:
                continue
            area = ID_TO_NAME[area_id]
            all_cells.append({
                'animal':      animal,
                'pos':         pos,
                'area':        area,
                'area_id':     area_id,
                'yaw_l_rel':   float(yaw_l_rel[ci]),
                'yaw_l_isrel': bool(yaw_l_isrel[ci]),
                'yaw_d_rel':   float(yaw_d_rel[ci]),
                'tuning':      yaw_1dtuning[ci, :, 1].copy(),  # light = index 1
                'err':         yaw_1derr   [ci, :, 1].copy(),
                'tuning_dark': yaw_1dtuning[ci, :, 0].copy(),  # dark  = index 0
                'err_dark':    yaw_1derr   [ci, :, 0].copy(),
                'bins':        yaw_1dbins.copy(),
            })

    print(f'Total cells with named area: {len(all_cells)}')
    print(f'Recordings with occupancy:   {len(recordings)}')
    return all_cells, recordings


def _scatter_col(ax, x_pos, vals, color, label=None):
    vals = np.array(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return
    jitter = (np.random.rand(len(vals)) - 0.5) * 0.4
    ax.scatter(np.ones(len(vals)) * x_pos + jitter, vals,
               s=8, c=color, alpha=0.7, zorder=3, label=label)
    mn  = np.nanmedian(vals)
    sem = np.nanstd(vals) / np.sqrt(len(vals))
    ax.hlines(mn, x_pos - 0.15, x_pos + 0.15, colors='k', linewidths=1.5, zorder=4)
    ax.vlines(x_pos, mn - sem, mn + sem, colors='k', linewidths=1.5, zorder=4)


def _norm01(tc):
    lo, hi = np.nanmin(tc), np.nanmax(tc)
    if hi - lo > 0:
        return (tc - lo) / (hi - lo)
    return np.full_like(tc, 0.5)


def make_occupancy_page(pdf, recordings):

    if not recordings:
        return

    ncols  = 6
    nrows  = int(np.ceil(len(recordings) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 1.7, nrows * 1.5),
                             dpi=200, squeeze=False)

    for i, rec in enumerate(recordings):
        ax   = axes[i // ncols][i % ncols]
        bins = rec['occ_bins']
        ctrs = (bins[:-1] + bins[1:]) / 2

        if rec['counts_light'] is not None and rec['counts_dark'] is not None:
            ax.bar(ctrs, rec['counts_dark'],  width=9.5, color='0.4',  alpha=0.7, label='dark')
            ax.bar(ctrs, rec['counts_light'], width=9.5, color='goldenrod', alpha=0.8,
                   bottom=rec['counts_dark'], label='light')
        else:
            ax.bar(ctrs, rec['counts_all'], width=9.5, color='steelblue', alpha=0.8)

        pct_nan = rec['n_nan'] / rec['n_total'] * 100
        ax.set_title(f"{rec['animal']}/{rec['pos']}", fontsize=5, pad=2)
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 180, 360])
        ax.set_xticklabels(['0', '180', '360'], fontsize=4)
        ax.tick_params(labelsize=4)
        ax.set_xlabel('yaw (°)', fontsize=4)
        ax.text(0.97, 0.97, f'NaN {pct_nan:.1f}%',
                ha='right', va='top', transform=ax.transAxes, fontsize=4, color='0.4')

    for j in range(len(recordings), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    if recordings[0]['counts_light'] is not None:
        axes[0][0].legend(fontsize=4, frameon=False, loc='upper left')

    fig.suptitle(
        'Head yaw occupancy per recording  (red dashed = saved tuning-curve range)',
        fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def make_summary_page(pdf, all_cells):
    """Per-area violin of yaw_l_rel (cross-validated MI), light condition."""
    area_mi = {a: [] for a in REGION_ORDER}
    area_n  = {a: 0  for a in REGION_ORDER}
    for c in all_cells:
        if c['area'] in area_mi:
            area_mi[c['area']].append(c['yaw_l_rel'])
            area_n [c['area']] += 1

    areas_present = [a for a in REGION_ORDER if area_n[a] >= MIN_CELLS_AREA]
    if not areas_present:
        return

    data   = [np.array(area_mi[a], dtype=float) for a in areas_present]
    colors = [COLORS.get(a, '#888888') for a in areas_present]

    fig, ax = plt.subplots(figsize=(len(areas_present) * 0.85 + 0.8, 3.5), dpi=300)

    parts = ax.violinplot(data, positions=range(len(areas_present)),
                          showmedians=False, showextrema=False)
    for body, color in zip(parts['bodies'], colors):
        body.set_facecolor(color)
        body.set_edgecolor('k')
        body.set_linewidth(0.5)
        body.set_alpha(0.75)

    for xi, vals in enumerate(data):
        med = np.nanmedian(vals)
        q25, q75 = np.nanpercentile(vals, [25, 75])
        ax.vlines(xi, q25, q75, colors='k', linewidths=2.5, zorder=4)
        ax.scatter([xi], [med], s=18, color='w', edgecolors='k',
                   linewidths=0.8, zorder=5)
        ax.text(xi, ax.get_ylim()[0] - 0.01, f'n={area_n[areas_present[xi]]}',
                ha='center', va='top', fontsize=5, color='0.4',
                transform=ax.get_xaxis_transform())

    ax.set_xticks(range(len(areas_present)))
    ax.set_xticklabels(areas_present, fontsize=7)
    ax.set_ylabel('yaw CV modulation index (light)', fontsize=8)
    ax.set_xlim(-0.6, len(areas_present) - 0.4)
    ax.axhline(0, color='0.7', lw=0.8, ls='--')

    fig.suptitle('Yaw head-direction tuning by visual area', fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def make_fraction_modulated_page(pdf, all_cells, threshold=0.33):

    area_n     = {a: 0 for a in REGION_ORDER}
    area_above = {a: 0 for a in REGION_ORDER}

    for c in all_cells:
        if c['area'] in area_n:
            area_n    [c['area']] += 1
            if c['yaw_l_rel'] > threshold:
                area_above[c['area']] += 1

    areas_present = [a for a in REGION_ORDER if area_n[a] >= MIN_CELLS_AREA]
    if not areas_present:
        return

    fracs = np.array([area_above[a] / area_n[a] * 100 for a in areas_present])
    ns    = [area_n[a] for a in areas_present]
    colors = [COLORS.get(a, '#888888') for a in areas_present]

    fig, ax = plt.subplots(figsize=(len(areas_present) * 0.9 + 0.8, 3.2), dpi=300)
    xs = np.arange(len(areas_present))
    bars = ax.bar(xs, fracs, color=colors, width=0.6, edgecolor='k', linewidth=0.5)

    for xi, (frac, n) in enumerate(zip(fracs, ns)):
        ax.text(xi, frac + 0.8, f'{frac:.1f}%', ha='center', va='bottom', fontsize=6)
        ax.text(xi, -2.5, f'n={n}', ha='center', va='top',
                fontsize=5, color='0.4', transform=ax.get_xaxis_transform())

    ax.set_xticks(xs)
    ax.set_xticklabels(areas_present, fontsize=8)
    ax.set_ylabel(f'% cells with CV MI > {threshold}', fontsize=8)
    ax.set_ylim(0, 4)
    ax.set_xlim(-0.6, len(areas_present) - 0.4)
    ax.axhline(0, color='k', lw=0.5)

    fig.suptitle(f'Fraction head-yaw modulated (CV MI > {threshold}) by visual area', fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def make_heatmap_page(pdf, all_cells, top_n=TOP_N_HEATMAP, condition='light'):

    sorted_cells = sorted(all_cells, key=lambda c: c['yaw_l_rel'], reverse=True)
    show = sorted_cells[:top_n]
    if not show:
        return

    tc_key = 'tuning' if condition == 'light' else 'tuning_dark'

    n_show = len(show)
    n_bins = len(show[0]['bins'])

    mat = np.zeros((n_show, n_bins))
    for i, c in enumerate(show):
        mat[i] = _norm01(c[tc_key])

    area_rgb = np.array([mpl.colors.to_rgb(COLORS.get(c['area'], '#888888'))
                         for c in show])

    mi_vals = np.array([c['yaw_l_rel'] for c in show])

    fig = plt.figure(figsize=(6.5, max(4, n_show * 0.11 + 1.5)), dpi=300)
    gs  = fig.add_gridspec(1, 3, width_ratios=[0.18, 4.5, 0.8], wspace=0.04,
                           left=0.01, right=0.97, top=0.93, bottom=0.07)
    ax_area = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])
    ax_mi   = fig.add_subplot(gs[2])

    area_img = area_rgb[:, np.newaxis, :]
    ax_area.imshow(area_img, aspect='auto', interpolation='none')
    ax_area.set_xticks([])
    ax_area.set_yticks(range(n_show))
    ax_area.set_yticklabels([c['area'] for c in show], fontsize=4)
    ax_area.tick_params(length=0)

    im = ax_heat.imshow(mat, aspect='auto', cmap='magma',
                        vmin=0, vmax=1, interpolation='nearest')
    ax_heat.set_yticks([])
    ax_heat.set_xlabel('yaw bins (0° -> 360°)', fontsize=6)
    ax_heat.set_xticks(range(n_bins))
    ax_heat.set_xticklabels(
        [f'{b:.0f}°' for b in show[0]['bins']] if n_bins <= 14 else [],
        fontsize=5, rotation=45)

    cax = ax_heat.inset_axes([1.01, 0.0, 0.03, 1.0], transform=ax_heat.transAxes)
    fig.colorbar(im, cax=cax, label='norm. rate')
    cax.tick_params(labelsize=5)

    colors_bar = [COLORS.get(c['area'], '#888888') for c in show]
    ax_mi.barh(range(n_show), mi_vals, color=colors_bar, height=0.85)
    ax_mi.set_xlim(0, max(mi_vals.max() * 1.1, 0.3))
    ax_mi.set_ylim(-0.5, n_show - 0.5)
    ax_mi.invert_yaxis()
    ax_mi.set_yticks([])
    ax_mi.set_xlabel('CV MI', fontsize=6)
    ax_mi.axvline(0.1, color='0.5', lw=0.7, ls='--')

    fig.suptitle(f'Top {n_show} yaw-tuned cells (sorted by CV modulation index, light) — {condition}',
                 fontsize=8)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def make_per_area_pages(pdf, all_cells, top_n_per_area=TOP_N_PER_AREA, condition='light'):

    ncols   = 4
    tc_key  = 'tuning'     if condition == 'light' else 'tuning_dark'
    err_key = 'err'        if condition == 'light' else 'err_dark'
    mi_key  = 'yaw_l_rel'  if condition == 'light' else 'yaw_d_rel'

    for area in REGION_ORDER:
        cells = [c for c in all_cells if c['area'] == area]
        if len(cells) < MIN_CELLS_AREA:
            continue
        cells.sort(key=lambda c: c['yaw_l_rel'], reverse=True)
        cells = cells[:top_n_per_area]

        nrows = int(np.ceil(len(cells) / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 1.8, nrows * 1.6),
                                 dpi=200, squeeze=False)

        color = COLORS.get(area, '#888888')

        for i, c in enumerate(cells):
            ax   = axes[i // ncols][i % ncols]
            bins = c['bins']
            tc   = c[tc_key]
            err  = c[err_key]

            ax.plot(bins, tc, color=color, lw=1.2)
            ax.fill_between(bins, tc - err, tc + err,
                            alpha=0.25, color=color)
            mi_val = c[mi_key]
            mi_str = f'{mi_val:.3f}' if np.isfinite(mi_val) else 'NaN'
            ax.set_title(f'MI={mi_str}', fontsize=6, pad=2)
            ax.set_xlim(0, 360)
            ax.set_xticks([0, 180, 360])
            ax.set_xticklabels(['0°', '180°', '360°'], fontsize=5)
            ax.tick_params(labelsize=5)
            ax.set_xlabel('yaw (°)', fontsize=5)
            ax.text(0.97, 0.95, f'#{i + 1}  {c["animal"]}/{c["pos"]}',
                    ha='right', va='top', transform=ax.transAxes,
                    fontsize=4, color='0.5')

            _setmax = np.max(tc + err) * 1.1
            ax.set_ylim(0, _setmax)

        for j in range(len(cells), nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)

        fig.suptitle(f'{area} — top {len(cells)} yaw-tuned cells ({condition})', fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Summarize yaw head-direction tuning across visual areas.')
    parser.add_argument('--pooled',   default=DEFAULT_POOLED,
                        help='Path to pooled_*.h5')
    parser.add_argument('--base_dir', default=DEFAULT_BASE,
                        help='Root directory to search for eyehead_revcorrs_v07.h5')
    parser.add_argument('--out_dir',  default=DEFAULT_OUT_DIR,
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Pooled dataset : {args.pooled}')
    print(f'Search root    : {args.base_dir}')

    all_cells, recordings = collect_data(args.pooled, args.base_dir)

    if not all_cells:
        print('No cells collected. Exiting.')
        return

    pdf_path = os.path.join(args.out_dir, 'head_yaw_tuning_summary.pdf')
    print(f'\nWriting PDF: {pdf_path}')

    with PdfPages(pdf_path) as pdf:
        make_summary_page(pdf, all_cells)
        make_fraction_modulated_page(pdf, all_cells)
        make_occupancy_page(pdf, recordings)
        # make_heatmap_page(pdf, all_cells, condition='light')
        make_per_area_pages(pdf, all_cells, condition='light')

    print(f'Done. PDF: {pdf_path}')

    pdf_dark_path = os.path.join(args.out_dir, 'head_yaw_tuning_summary_dark.pdf')
    print(f'\nWriting dark PDF: {pdf_dark_path}')

    with PdfPages(pdf_dark_path) as pdf:
        # make_heatmap_page(pdf, all_cells, condition='dark')
        make_per_area_pages(pdf, all_cells, condition='dark')

    print(f'Done. Dark PDF: {pdf_dark_path}')


if __name__ == '__main__':
    
    main()
