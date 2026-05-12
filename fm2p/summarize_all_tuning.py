

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


def collect_data(pooled_path: str, base_dir: str, condition: str = 'light') -> list:

    cond_key = 'l' if condition == 'light' else 'd'
    cond_idx = 1   if condition == 'light' else 0

    pooled_lookup = _build_pooled_lookup(pooled_path)
    revcorr_files = find('eyehead_revcorrs_v06.h5', base_dir)
    print(f'Found {len(revcorr_files)} eyehead_revcorrs_v06.h5 files  [{condition}].')

    all_cells = []

    for rcf in sorted(revcorr_files):
        try:
            with h5py.File(rcf, 'r') as f:
                if not any(f'{v}_l_rel' in f for v in VAR_NAMES):
                    continue

                n_cells_f = None
                var_data  = {}

                for vname in VAR_NAMES:
                    rel_key    = f'{vname}_{cond_key}_rel'
                    isrel_key  = f'{vname}_{cond_key}_isrel'
                    tuning_key = f'{vname}_1dtuning'
                    err_key    = f'{vname}_1derr'
                    bins_key   = f'{vname}_1dbins'

                    if rel_key not in f:
                        var_data[vname] = None
                        continue

                    rel    = f[rel_key][()].astype(float)
                    isrel  = (f[isrel_key][()].astype(bool)
                              if isrel_key in f
                              else np.zeros(len(rel), dtype=bool))
                    tuning = f[tuning_key][()].astype(float) if tuning_key in f else None
                    err    = f[err_key][()].astype(float)    if err_key    in f else None
                    bins   = f[bins_key][()].astype(float)   if bins_key   in f else None

                    var_data[vname] = dict(rel=rel, isrel=isrel,
                                          tuning=tuning, err=err, bins=bins)
                    if n_cells_f is None:
                        n_cells_f = len(rel)

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
            if vd is not None and len(vd['rel']) != n_cells:
                print(f'  {vname} cell count mismatch '
                      f'({len(vd["rel"])} vs {n_cells}), dropping for {rcf}')
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
                    cell[f'{vname}_rel']    = np.nan
                    cell[f'{vname}_isrel']  = False
                    cell[f'{vname}_tuning'] = None
                    cell[f'{vname}_err']    = None
                    cell[f'{vname}_bins']   = None
                else:
                    cell[f'{vname}_rel']   = float(vd['rel'][ci])
                    cell[f'{vname}_isrel'] = bool(vd['isrel'][ci])
                    if vd['tuning'] is not None:
                        cell[f'{vname}_tuning'] = vd['tuning'][ci, :, cond_idx].copy()
                        cell[f'{vname}_err']    = vd['err'][ci, :, cond_idx].copy()
                    else:
                        cell[f'{vname}_tuning'] = None
                        cell[f'{vname}_err']    = None
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


def _violin_ax(ax, all_cells, vspec):

    vname = vspec['name']

    area_vals = {a: [] for a in REGION_ORDER}
    area_n    = {a: 0  for a in REGION_ORDER}
    for c in all_cells:
        if c['area'] in area_vals:
            area_n[c['area']] += 1
            rel = c[f'{vname}_rel']
            if np.isfinite(rel):
                area_vals[c['area']].append(rel)

    areas_present = [a for a in REGION_ORDER if area_n[a] >= MIN_CELLS_AREA]
    if not areas_present:
        ax.set_visible(False)
        return []

    data   = [np.array(area_vals[a]) for a in areas_present]
    colors = [COLORS.get(a, '#888888') for a in areas_present]

    nonempty = [(xi, d) for xi, d in enumerate(data) if len(d) >= 2]
    if nonempty:
        positions, datasets = zip(*nonempty)
        parts = ax.violinplot(list(datasets), positions=list(positions),
                              showmedians=False, showextrema=False)
        for body, pos in zip(parts['bodies'], positions):
            body.set_facecolor(colors[pos])
            body.set_edgecolor('k')
            body.set_linewidth(0.5)
            body.set_alpha(0.75)

    for xi, (a, vals) in enumerate(zip(areas_present, data)):
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
    ax.set_ylabel('CV MI', fontsize=7)
    ax.set_xlim(-0.6, len(areas_present) - 0.4)
    ax.axhline(0, color='0.7', lw=0.8, ls='--')
    return areas_present


def _fraction_ax(ax, all_cells, vspec, threshold):
    
    vname  = vspec['name']
    is_imu = vspec['is_imu']

    area_total = {a: 0 for a in REGION_ORDER}
    area_valid = {a: 0 for a in REGION_ORDER}
    area_above = {a: 0 for a in REGION_ORDER}

    for c in all_cells:
        if c['area'] not in area_total:
            continue
        area_total[c['area']] += 1
        rel = c[f'{vname}_rel']
        if np.isfinite(rel):
            area_valid[c['area']] += 1
            if rel > threshold:
                area_above[c['area']] += 1

    areas_present = [a for a in REGION_ORDER if area_total[a] >= MIN_CELLS_AREA]
    if not areas_present:
        ax.set_visible(False)
        return

    fracs, ns = [], []
    for a in areas_present:

        denom = area_valid[a] if is_imu else area_total[a]
        fracs.append(area_above[a] / denom * 100 if denom > 0 else 0.0)
        ns.append(denom)

    fracs  = np.array(fracs)
    colors = [COLORS.get(a, '#888888') for a in areas_present]
    xs     = np.arange(len(areas_present))

    ax.bar(xs, fracs, color=colors, width=0.6, edgecolor='k', linewidth=0.5)
    for xi, (frac, n) in enumerate(zip(fracs, ns)):
        ax.text(xi, frac + 0.5, f'{frac:.0f}%',
                ha='center', va='bottom', fontsize=5)
        ax.text(xi, -2.5, f'n={n}', ha='center', va='top',
                fontsize=4, color='0.4', transform=ax.get_xaxis_transform())

    ax.set_xticks(xs)
    ax.set_xticklabels(areas_present, fontsize=6)
    ax.set_title(vspec['label'] + (' *' if is_imu else ''), fontsize=8)
    ax.set_ylabel(f'% CV MI > {threshold}', fontsize=7)
    ax.set_xlim(-0.6, len(areas_present) - 0.4)
    ax.set_ylim(0, max(fracs.max() * 1.25, 10))
    ax.axhline(0, color='k', lw=0.5)


def make_violin_page(pdf, all_cells):

    nv  = len(VARIABLES)
    fig, axes = plt.subplots(1, nv, figsize=(nv * 2.2 + 0.5, 3.8), dpi=300)

    for ax, vspec in zip(axes, VARIABLES):
        _violin_ax(ax, all_cells, vspec)

    fig.suptitle('Tuning reliability (CV modulation index) by visual area  [light]',
                 fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def make_fraction_page(pdf, all_cells, threshold=MOD_THRESHOLD, label=''):

    nv  = len(VARIABLES)
    fig, axes = plt.subplots(1, nv, figsize=(nv * 3.2 + 0.5, 3.2), dpi=300)

    for ax, vspec in zip(axes, VARIABLES):
        _fraction_ax(ax, all_cells, vspec, threshold)
        ax.set_ylim([0,20])

    suffix = f'  [{label}]' if label else ''
    fig.suptitle(
        f'% cells with CV MI > {threshold}'
        f'  (* IMU variables: n = cells with IMU data){suffix}',
        fontsize=9)
    fig.tight_layout(w_pad=2.5)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def make_any_modulated_page(pdf, all_cells, threshold=MOD_THRESHOLD, label=''):

    area_total = {a: 0 for a in REGION_ORDER}
    area_any   = {a: 0 for a in REGION_ORDER}

    for c in all_cells:
        if c['area'] not in area_total:
            continue
        area_total[c['area']] += 1
        if any(np.isfinite(c[f'{v}_rel']) and c[f'{v}_rel'] > threshold
               for v in VAR_NAMES):
            area_any[c['area']] += 1

    areas_present = [a for a in REGION_ORDER if area_total[a] >= MIN_CELLS_AREA]
    if not areas_present:
        return

    fracs  = np.array([area_any[a] / area_total[a] * 100 for a in areas_present])
    ns     = [area_total[a] for a in areas_present]
    colors = [COLORS.get(a, '#888888') for a in areas_present]
    xs     = np.arange(len(areas_present))

    fig, ax = plt.subplots(figsize=(len(areas_present) * 0.9 + 0.8, 3.2), dpi=300)
    ax.bar(xs, fracs, color=colors, width=0.6, edgecolor='k', linewidth=0.5)
    for xi, (frac, n) in enumerate(zip(fracs, ns)):
        ax.text(xi, frac + 0.5, f'{frac:.0f}%', ha='center', va='bottom', fontsize=6)
        ax.text(xi, -2.5, f'n={n}', ha='center', va='top',
                fontsize=5, color='0.4', transform=ax.get_xaxis_transform())

    ax.set_xticks(xs)
    ax.set_xticklabels(areas_present, fontsize=8)
    ax.set_ylabel(f'% cells (any variable CV MI > {threshold})', fontsize=8)
    ax.set_xlim(-0.6, len(areas_present) - 0.4)
    # ax.set_ylim(0, max(fracs.max() * 1.25, 10))
    ax.set_ylim([0,42])
    ax.axhline(0, color='k', lw=0.5)

    suffix = f'  [{label}]' if label else ''
    fig.suptitle(
        f'% cells tuned to at least one variable  (CV MI > {threshold}){suffix}',
        fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def make_heatmap_pages(pdf, all_cells, top_n=TOP_N_HEATMAP):

    for vspec in VARIABLES:
        vname = vspec['name']

        cells_v = [c for c in all_cells
                   if c[f'{vname}_tuning'] is not None
                   and np.isfinite(c[f'{vname}_rel'])]
        if not cells_v:
            continue

        show   = sorted(cells_v, key=lambda c: c[f'{vname}_rel'], reverse=True)[:top_n]
        n_show = len(show)
        bins   = show[0][f'{vname}_bins']
        n_bins = len(bins)

        mat      = np.array([_norm01(c[f'{vname}_tuning']) for c in show])
        area_rgb = np.array([mpl.colors.to_rgb(COLORS.get(c['area'], '#888888'))
                             for c in show])
        mi_vals  = np.array([c[f'{vname}_rel'] for c in show])

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

        colors_bar = [COLORS.get(c['area'], '#888888') for c in show]
        ax_mi.barh(range(n_show), mi_vals, color=colors_bar, height=0.85)
        ax_mi.set_xlim(0, max(mi_vals.max() * 1.1, 0.3))
        ax_mi.set_ylim(-0.5, n_show - 0.5)
        ax_mi.invert_yaxis()
        ax_mi.set_yticks([])
        ax_mi.set_xlabel('CV MI', fontsize=6)
        ax_mi.axvline(0.1, color='0.5', lw=0.7, ls='--')

        fig.suptitle(
            f'Top {n_show} {vspec["label"]}-tuned cells  (sorted by CV MI, light)',
            fontsize=8)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)


def make_per_area_pages(pdf, all_cells, top_n=TOP_N_PER_AREA):

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
                ax   = axes[i // ncols][i % ncols]
                bins = c[f'{vname}_bins']
                tc   = c[f'{vname}_tuning']
                err  = c[f'{vname}_err']

                ax.plot(bins, tc, color=color, lw=1.2)
                if err is not None:
                    ax.fill_between(bins, tc - err, tc + err,
                                    alpha=0.25, color=color)
                ax.set_title(f'MI={c[f"{vname}_rel"]:.3f}', fontsize=6, pad=2)
                mid = len(bins) // 2
                ax.set_xticks([bins[0], bins[mid], bins[-1]])
                ax.set_xticklabels([f'{bins[0]:.0f}°', f'{bins[mid]:.0f}°',
                                    f'{bins[-1]:.0f}°'], fontsize=5)
                ax.tick_params(labelsize=5)
                ax.set_xlabel(f'{vspec["label"]} (°)', fontsize=5)
                ax.text(0.97, 0.95, f'{c["animal"]}/{c["pos"]}',
                        ha='right', va='top', transform=ax.transAxes,
                        fontsize=4, color='0.5')
                top_val = np.nanmax(tc + (err if err is not None else 0)) * 1.1
                if np.isfinite(top_val) and top_val > 0:
                    ax.set_ylim(0, top_val)

            for j in range(len(cells), nrows * ncols):
                axes[j // ncols][j % ncols].set_visible(False)

            fig.suptitle(
                f'{area} — {vspec["label"]} — top {len(cells)} cells (light)',
                fontsize=9)
            fig.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Summarize 1-D tuning across all variables and visual areas.')
    parser.add_argument('--pooled',   default=DEFAULT_POOLED)
    parser.add_argument('--base_dir', default=DEFAULT_BASE)
    parser.add_argument('--out_dir',  default=DEFAULT_OUT_DIR)
    parser.add_argument('--threshold', type=float, default=MOD_THRESHOLD,
                        help='CV MI threshold for % modulated page')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'Pooled dataset : {args.pooled}')
    print(f'Search root    : {args.base_dir}')

    for condition in ('light', 'dark'):
        all_cells = collect_data(args.pooled, args.base_dir, condition=condition)
        if not all_cells:
            print(f'No cells collected for {condition}. Skipping.')
            continue

        imu_cells, no_imu_cells = _split_by_imu(all_cells)
        print(f'  IMU recordings    : {len(imu_cells)} cells')
        print(f'  Non-IMU recordings: {len(no_imu_cells)} cells')

        pdf_path = os.path.join(args.out_dir, f'all_tuning_summary_{condition}.pdf')
        print(f'\nWriting PDF: {pdf_path}')

        with PdfPages(pdf_path) as pdf:
            make_violin_page(pdf, all_cells)
            make_fraction_page(pdf, all_cells, threshold=args.threshold)
            make_fraction_page(pdf, imu_cells,    threshold=args.threshold, label='IMU animals')
            make_fraction_page(pdf, no_imu_cells, threshold=args.threshold, label='non-IMU animals')
            make_any_modulated_page(pdf, imu_cells,    threshold=args.threshold, label='IMU animals')
            make_any_modulated_page(pdf, no_imu_cells, threshold=args.threshold, label='non-IMU animals')
            make_heatmap_pages(pdf, all_cells)
            make_per_area_pages(pdf, all_cells)

        print(f'Done. PDF: {pdf_path}')


if __name__ == '__main__':
    main()
