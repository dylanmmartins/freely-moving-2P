# -*- coding: utf-8 -*-

if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
    __package__ = 'fm2p.utils'

import glob
import os
import argparse

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from .files import read_h5, write_h5
from .filter import convfilt

_LABEL_MAP   = {2: 'RL', 3: 'AM', 4: 'PM', 5: 'V1', 10: 'A', 7: 'AL', 8: 'LM', 9: 'P'}
_REGION_ORDER = ['V1', 'RL', 'AM', 'PM', 'A', 'AL', 'LM', 'P']
_REGION_COLORS = {
    'V1': '#1B9E77', 'RL': '#D95F02', 'AM': '#7570B3', 'PM': '#E7298A',
    'A':  '#1A5A34', 'AL': '#E6AB02', 'LM': '#A6761D', 'P':  '#666666',
}

_BEHAVIOR_VARS = [
    'theta', 'phi', 'head_yaw', 'pitch', 'roll',
    'dTheta', 'dPhi', 'gyro_x', 'gyro_y', 'gyro_z',
]

_DEFAULT_CONFIG = {
    'min_cells':   5,
    'min_frames':  200,
    'alpha':       1.0,
    'smooth_win':  5,
    'speed_thr':   2.0,
    'n_chunks':    10,
    'test_frac':   0.25,
    'signal':      'spikes',
}

def _scalar_dict_to_array(d):
    """Convert a string-integer-keyed scalar dict (from read_h5) to a sorted array."""
    keys = sorted(d.keys(), key=lambda x: int(x))
    return np.array([d[k] for k in keys])


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-10)


def _pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    denom = (np.std(a) * np.std(b) + 1e-10) * len(a)
    return float(np.sum(a * b) / denom)


def _find_preproc_file(animal, poskey, search_dirs, n_cells_expected=None):

    _EXCLUDE = ('boundary', 'revcorr', 'GLM', 'sn1', 'sn_')
    _tag     = f'{animal}_{poskey}'

    candidates = []
    for d in search_dirs:
        if not os.path.isdir(d):
            print(f"  [find_preproc] WARNING: search dir does not exist: {d}")
            continue
        pat = os.path.join(d, '**', '*preproc.h5')
        for p in glob.glob(pat, recursive=True):
            if _tag in p and not any(s in p for s in _EXCLUDE):
                candidates.append(p)

    if not candidates:
        print(f"  [find_preproc] {animal} {poskey}: no file found under "
              + ', '.join(search_dirs))
        return None
    if n_cells_expected is None or len(candidates) == 1:
        return candidates[0]


    for c in candidates:
        try:
            with h5py.File(c, 'r') as f:
                if 'norm_spikes' in f:
                    if f['norm_spikes'].shape[0] == n_cells_expected:
                        return c
        except Exception:
            continue
    return candidates[0]

def _load_fov(preproc_path, vis_id, config):

    with h5py.File(preproc_path, 'r') as f:
        sig_key = 'norm_dFF' if config.get('signal', 'spikes') == 'dff' else 'norm_spikes'
        if sig_key not in f:
            alt = 'norm_spikes' if sig_key == 'norm_dFF' else 'norm_dFF'
            if alt in f:
                print(f"  [load_fov] {sig_key} not found, falling back to {alt}")
                sig_key = alt
            else:
                raise KeyError(f"Neither norm_spikes nor norm_dFF found in {preproc_path}")
        spk = f[sig_key][()].astype(float)
        twopT = f['twopT'][()].astype(float)
        ltdk  = f['ltdk_state_vec'][()].astype(bool) if 'ltdk_state_vec' in f \
                else np.ones(spk.shape[1], dtype=bool)
        spd   = f['speed'][()].astype(float) if 'speed' in f \
                else np.zeros(spk.shape[1])

        _keys = {
            'theta':    'theta_interp',
            'phi':      'phi_interp',
            'head_yaw': 'head_yaw_deg',
            'pitch':    'pitch_twop_interp',
            'roll':     'roll_twop_interp',
            'gyro_x':   'gyro_x_twop_interp',
            'gyro_y':   'gyro_y_twop_interp',
            'gyro_z':   'gyro_z_twop_interp',
        }
        Y_raw = {}
        for name, key in _keys.items():
            if key in f:
                arr = f[key][()].astype(float)
                Y_raw[name] = arr

    N_cells_spk = spk.shape[0]
    N_vis        = len(vis_id)
    N_cells      = min(N_cells_spk, N_vis)
    if N_cells_spk != N_vis:
        print(f'    [load_fov] cell count mismatch: spk={N_cells_spk}, vis_id={N_vis} '
              f'-> using first {N_cells}')
    spk = spk[:N_cells, :]

    T_raw = spk.shape[1]

    for name in list(Y_raw.keys()):
        Y_raw[name] = Y_raw[name][:T_raw]
    ltdk = ltdk[:T_raw]
    spd  = spd[:T_raw]

    if 'theta' in Y_raw:
        d = np.diff(Y_raw['theta'])
        dt = np.diff(twopT[:T_raw])
        dTheta = np.empty(T_raw)
        dTheta[:-1] = d / (dt + 1e-9)
        dTheta[-1]  = dTheta[-2]
        Y_raw['dTheta'] = dTheta
    if 'phi' in Y_raw:
        d = np.diff(Y_raw['phi'])
        dt = np.diff(twopT[:T_raw])
        dPhi = np.empty(T_raw)
        dPhi[:-1] = d / (dt + 1e-9)
        dPhi[-1]  = dPhi[-2]
        Y_raw['dPhi'] = dPhi


    win = config['smooth_win']
    for ci in range(N_cells):
        spk[ci] = convfilt(spk[ci], win)
    X = spk.T

    return X, Y_raw, ltdk, spd, N_cells

def _ridge_decode(X_train, y_train, X_test, alpha):

    n, d = X_train.shape

    Xb       = np.column_stack([X_train, np.ones(n)])
    Xb_test  = np.column_stack([X_test,  np.ones(len(X_test))])

    A  = Xb.T @ Xb
    A[:d, :d] += alpha * np.eye(d)
    b  = Xb.T @ y_train
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    return Xb_test @ w

def _decode_fov(animal, poskey, pooled_path, preproc_path, config):

    with h5py.File(pooled_path, 'r') as f:
        mess = f[animal]['messentials']
        if poskey not in mess or not isinstance(mess[poskey], h5py.Group):
            return []
        vis_id = mess[poskey]['visual_area_id'][()].astype(int)

    X, Y_raw, ltdk, spd, N_cells = _load_fov(preproc_path, vis_id, config)

    vis_id = vis_id[:N_cells]
    T      = X.shape[0]

    results = []

    for cond_label, cond_flag in [('l', True), ('d', False)]:

        cond_mask  = (ltdk == cond_flag)
        speed_mask = spd > config['speed_thr']
        use_mask   = cond_mask & speed_mask
        use_idx    = np.where(use_mask)[0]

        if len(use_idx) < config['min_frames']:
            continue

        n_chunks   = config['n_chunks']
        chunks     = np.array_split(use_idx, n_chunks)
        rng        = np.random.default_rng(seed=42)
        chunk_ord  = rng.permutation(n_chunks)
        n_test     = max(1, int(round(config['test_frac'] * n_chunks)))
        test_chunks  = [chunks[i] for i in chunk_ord[:n_test]]
        train_chunks = [chunks[i] for i in chunk_ord[n_test:]]
        train_idx = np.sort(np.concatenate(train_chunks))
        test_idx  = np.sort(np.concatenate(test_chunks))

        X_mu  = X[train_idx].mean(axis=0)
        X_sd  = X[train_idx].std(axis=0) + 1e-8
        X_tr  = (X[train_idx] - X_mu) / X_sd
        X_te  = (X[test_idx]  - X_mu) / X_sd

        for region_id, region_name in _LABEL_MAP.items():
            cell_mask = (vis_id == region_id)
            n_area    = int(cell_mask.sum())
            if n_area < config['min_cells']:
                continue

            X_tr_area = X_tr[:, cell_mask]
            X_te_area = X_te[:, cell_mask]

            for bname in _BEHAVIOR_VARS:
                if bname not in Y_raw:
                    continue
                y_full = Y_raw[bname]

                y_tr_raw = y_full[train_idx]
                y_te_raw = y_full[test_idx]

                ok_tr = np.isfinite(y_tr_raw)
                ok_te = np.isfinite(y_te_raw)
                if ok_tr.sum() < config['min_frames'] // 2:
                    continue
                if ok_te.sum() < config['min_frames']:
                    continue

                mu_y = np.nanmean(y_tr_raw[ok_tr])
                sd_y = np.nanstd(y_tr_raw[ok_tr]) + 1e-8
                y_tr_n = (y_tr_raw[ok_tr] - mu_y) / sd_y
                y_te_n = (y_te_raw[ok_te] - mu_y) / sd_y

                y_pred_n = _ridge_decode(
                    X_tr_area[ok_tr], y_tr_n,
                    X_te_area[ok_te], config['alpha'],
                )

                y_pred = y_pred_n * sd_y + mu_y
                y_true = y_te_raw[ok_te]

                twop_idx = test_idx[ok_te].astype(np.int32)

                results.append({
                    'animal':       animal,
                    'pos':          poskey,
                    'region':       region_name,
                    'behavior':     bname,
                    'cond':         cond_label,
                    'n_cells':      n_area,
                    'r2':           float(_r2(y_true, y_pred)),
                    'corr':         float(_pearson(y_true, y_pred)),
                    'y_true':       y_true,
                    'y_pred':       y_pred,
                    'test_twop_idx': twop_idx,
                })

    return results


def decode_all_fovs(pooled_path, search_dirs, config=None):

    if config is None:
        config = _DEFAULT_CONFIG.copy()

    all_results = []

    with h5py.File(pooled_path, 'r') as f:
        animals = [a for a in f.keys() if a != 'uniref']
        fov_list = []
        for animal in animals:
            if 'messentials' not in f[animal] or 'transform' not in f[animal]:
                continue
            for poskey in f[animal]['messentials'].keys():
                if not poskey.startswith('pos'):
                    continue
                n_cells = None
                t_obj = f[animal]['transform'].get(poskey)
                if t_obj is not None and hasattr(t_obj, 'shape'):
                    n_cells = t_obj.shape[0]
                fov_list.append((animal, poskey, n_cells))

    print(f"Found {len(fov_list)} FOVs in pooled HDF5.")

    for animal, poskey, n_cells in tqdm(fov_list, desc='Decoding FOVs'):
        preproc = _find_preproc_file(animal, poskey, search_dirs, n_cells)
        if preproc is None:
            print(f"  [{animal} {poskey}] preproc file not found — skipping.")
            continue

        try:
            fov_results = _decode_fov(animal, poskey, pooled_path, preproc, config)
            all_results.extend(fov_results)
        except Exception as e:
            print(f"  [{animal} {poskey}] error: {e}")
            continue

    print(f"Decoded {len(all_results)} (area x behavior x condition) results "
          f"from {len(fov_list)} FOVs.")
    return all_results


def plot_r2_summary(results, save_path):

    from collections import defaultdict

    data = defaultdict(lambda: defaultdict(list))
    for r in results:
        if np.isfinite(r['r2']):
            data[r['behavior']][r['region']].append(r['r2'])

    behaviors_present = [b for b in _BEHAVIOR_VARS if b in data]
    if not behaviors_present:
        print("  [r2_summary] no results to plot.")
        return

    with PdfPages(save_path) as pdf:
        for bname in behaviors_present:
            rdict = data[bname]
            regions = [r for r in _REGION_ORDER if r in rdict and len(rdict[r]) >= 2]
            if not regions:
                continue

            fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(regions)), 4), dpi=200)
            plot_data = [np.array(rdict[r]) for r in regions]
            colors    = [_REGION_COLORS.get(r, '#888') for r in regions]

            vp = ax.violinplot(plot_data, positions=range(len(regions)),
                               showmedians=True, showextrema=False)
            for body, col in zip(vp['bodies'], colors):
                body.set_facecolor(col)
                body.set_alpha(0.75)
                body.set_edgecolor(col)
            vp['cmedians'].set_color('k')
            vp['cmedians'].set_linewidth(1.4)

            rng = np.random.default_rng(0)
            for xi, (r, vals) in enumerate(zip(regions, plot_data)):
                jitter = rng.uniform(-0.15, 0.15, len(vals))
                ax.scatter(xi + jitter, vals, s=10, alpha=0.5,
                           color=_REGION_COLORS.get(r, '#888'), zorder=3,
                           linewidths=0)

            ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
            ax.set_xticks(range(len(regions)))
            ax.set_xticklabels(regions)
            ax.set_ylabel('Decoding R^2')
            ax.set_title(f'{bname}  —  population decoding R^2 by visual area', fontsize=9)

            for xi, r in enumerate(regions):
                n = len(rdict[r])
                ax.text(xi, ax.get_ylim()[0] - 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                        f'n={n}', ha='center', va='top', fontsize=6)

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"  R^2 summary saved to {save_path}")


def plot_corr_summary(results, save_path):

    from collections import defaultdict

    data = defaultdict(lambda: defaultdict(list))
    for r in results:
        if np.isfinite(r['corr']):
            data[r['behavior']][r['region']].append(r['corr'])

    behaviors_present = [b for b in _BEHAVIOR_VARS if b in data]
    if not behaviors_present:
        return

    with PdfPages(save_path) as pdf:
        for bname in behaviors_present:
            rdict = data[bname]
            regions = [r for r in _REGION_ORDER if r in rdict and len(rdict[r]) >= 2]
            if not regions:
                continue

            fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(regions)), 4), dpi=200)
            plot_data = [np.array(rdict[r]) for r in regions]
            colors    = [_REGION_COLORS.get(r, '#888') for r in regions]

            vp = ax.violinplot(plot_data, positions=range(len(regions)),
                               showmedians=True, showextrema=False)
            for body, col in zip(vp['bodies'], colors):
                body.set_facecolor(col)
                body.set_alpha(0.75)
                body.set_edgecolor(col)
            vp['cmedians'].set_color('k')
            vp['cmedians'].set_linewidth(1.4)

            rng = np.random.default_rng(0)
            for xi, (r, vals) in enumerate(zip(regions, plot_data)):
                jitter = rng.uniform(-0.15, 0.15, len(vals))
                ax.scatter(xi + jitter, vals, s=10, alpha=0.5,
                           color=_REGION_COLORS.get(r, '#888'), zorder=3,
                           linewidths=0)

            ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
            ax.set_xticks(range(len(regions)))
            ax.set_xticklabels(regions)
            ax.set_ylabel('Pearson r')
            ax.set_title(f'{bname}   population decoding correlation by visual area',
                         fontsize=9)

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"  Correlation summary saved to {save_path}")


def plot_r2_heatmap(results, save_path):

    from collections import defaultdict
    import matplotlib.colors as mcolors

    data = defaultdict(lambda: defaultdict(list))
    for r in results:
        if np.isfinite(r['r2']):
            data[r['region']][r['behavior']].append(r['r2'])

    regions  = [r for r in _REGION_ORDER  if r in data]
    behaviors = [b for b in _BEHAVIOR_VARS if any(b in data[r] for r in regions)]
    if not regions or not behaviors:
        return

    mat = np.full((len(regions), len(behaviors)), np.nan)
    for ri, reg in enumerate(regions):
        for bi, beh in enumerate(behaviors):
            vals = data[reg].get(beh, [])
            if vals:
                mat[ri, bi] = np.median(vals)

    fig, ax = plt.subplots(figsize=(max(6, len(behaviors) * 0.9),
                                    max(3, len(regions) * 0.7)), dpi=200)
    vmax = np.nanpercentile(mat, 95)
    vmax = max(vmax, 0.05)
    im = ax.imshow(mat, aspect='auto', cmap='viridis', vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Median R^2')
    ax.set_xticks(range(len(behaviors)))
    ax.set_xticklabels(behaviors, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(regions, fontsize=8)
    ax.set_title('Population decoding R^2 — area × behavior', fontsize=9)

    # Annotate cells with the value
    for ri in range(len(regions)):
        for bi in range(len(behaviors)):
            v = mat[ri, bi]
            if np.isfinite(v):
                ax.text(bi, ri, f'{v:.2f}', ha='center', va='center',
                        fontsize=6, color='white' if v > vmax * 0.5 else 'black')

    fig.tight_layout()
    with PdfPages(save_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print(f"  R^2 heatmap saved to {save_path}")


def plot_example_traces(results, save_path, behaviors=('theta', 'phi'),
                        n_traces=3, n_frames=600):

    from collections import defaultdict

    idx = defaultdict(list)
    for r in results:
        if r['behavior'] in behaviors and np.isfinite(r['r2']):
            idx[(r['behavior'], r['region'])].append(r)

    with PdfPages(save_path) as pdf:
        for bname in behaviors:
            for region in _REGION_ORDER:
                entries = idx.get((bname, region), [])
                if not entries:
                    continue
                entries = sorted(entries, key=lambda x: -x['r2'])[:n_traces]

                color = _REGION_COLORS.get(region, 'steelblue')
                n = len(entries)
                fig, axes = plt.subplots(n, 2, figsize=(12, 3.2 * n), dpi=150,
                                         squeeze=False)
                fig.suptitle(
                    f'{bname} decoding — {region} cells\n'
                    f'(linear ridge decoder, population -> behavior)',
                    fontsize=10,
                )

                for row, res in enumerate(entries):
                    T = min(n_frames, len(res['y_true']))
                    t = np.arange(T)

                    ax_tr = axes[row, 0]
                    ax_sc = axes[row, 1]

                    ax_tr.plot(t, res['y_true'][:T], 'k',   lw=0.9, alpha=0.65, label='True')
                    ax_tr.plot(t, res['y_pred'][:T], color=color,
                               lw=0.9, alpha=0.85, label='Decoded')
                    ax_tr.set_xlabel('Frame (test set)')
                    ax_tr.set_ylabel(bname)
                    ax_tr.set_title(
                        f"{res['animal']} {res['pos']}  |  "
                        f"n_cells={res['n_cells']}  |  "
                        f"R^2={res['r2']:.3f}  r={res['corr']:.3f}",
                        fontsize=8,
                    )
                    if row == 0:
                        ax_tr.legend(frameon=False, fontsize=7)

                    lo = min(res['y_true'].min(), res['y_pred'].min())
                    hi = max(res['y_true'].max(), res['y_pred'].max())
                    ax_sc.scatter(res['y_true'], res['y_pred'],
                                  s=3, alpha=0.2, color=color)
                    ax_sc.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.5)
                    ax_sc.set_xlabel(f'True {bname}')
                    ax_sc.set_ylabel(f'Decoded {bname}')
                    ax_sc.set_title(f'R^2={res["r2"]:.3f}', fontsize=8)

                fig.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

    print(f"  Example traces saved to {save_path}")


def plot_n_cells_summary(results, save_path):

    from collections import defaultdict

    n_per_area = defaultdict(list)
    seen = set()
    for r in results:
        key = (r['animal'], r['pos'], r['region'])
        if key not in seen:
            n_per_area[r['region']].append(r['n_cells'])
            seen.add(key)

    regions = [r for r in _REGION_ORDER if r in n_per_area]
    if not regions:
        return

    fig, ax = plt.subplots(figsize=(max(4, len(regions) * 0.9), 3.5), dpi=200)
    medians = [np.median(n_per_area[r]) for r in regions]
    sems    = [np.std(n_per_area[r]) / np.sqrt(len(n_per_area[r])) for r in regions]
    colors  = [_REGION_COLORS.get(r, '#888') for r in regions]
    ax.bar(regions, medians, yerr=sems, color=colors, capsize=3, alpha=0.85)
    ax.set_ylabel('Cells (median per FOV)')
    ax.set_title('Cells per visual area per FOV', fontsize=9)
    fig.tight_layout()

    with PdfPages(save_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print(f"  Cell count summary saved to {save_path}")


def run_decoding_analysis(pooled_path, search_dirs, save_dir, config=None):

    os.makedirs(save_dir, exist_ok=True)
    if config is None:
        config = _DEFAULT_CONFIG.copy()
    all_results = decode_all_fovs(pooled_path, search_dirs, config)

    if not all_results:
        print("No results — check that preproc files are discoverable via --datadir.")
        return

    summary = {
        'animal':   [r['animal']   for r in all_results],
        'pos':      [r['pos']      for r in all_results],
        'region':   [r['region']   for r in all_results],
        'behavior': [r['behavior'] for r in all_results],
        'cond':     [r['cond']     for r in all_results],
        'n_cells':  np.array([r['n_cells'] for r in all_results]),
        'r2':       np.array([r['r2']      for r in all_results]),
        'corr':     np.array([r['corr']    for r in all_results]),
    }
    write_h5(os.path.join(save_dir, 'decoding_results.h5'), summary)

    traces = {}
    for r in all_results:
        ap   = f"{r['animal']}_{r['pos']}"
        reg  = r['region']
        bkey = f"{r['behavior']}_{r['cond']}"
        traces.setdefault(ap, {}).setdefault(reg, {})[bkey] = {
            'y_true':        r['y_true'].astype(np.float32),
            'y_pred':        r['y_pred'].astype(np.float32),
            'test_twop_idx': r['test_twop_idx'],
            'r2':            np.float32(r['r2']),
            'corr':          np.float32(r['corr']),
        }
    write_h5(os.path.join(save_dir, 'decoding_traces.h5'), traces)
    print(f"  Numeric results and per-FOV traces saved.")

    plot_r2_summary(
        all_results, os.path.join(save_dir, 'decoding_r2_by_area.pdf'))
    plot_corr_summary(
        all_results, os.path.join(save_dir, 'decoding_corr_by_area.pdf'))
    plot_r2_heatmap(
        all_results, os.path.join(save_dir, 'decoding_r2_heatmap.pdf'))
    plot_example_traces(
        all_results, os.path.join(save_dir, 'decoding_traces.pdf'),
        behaviors=['theta', 'phi', 'head_yaw'])
    plot_n_cells_summary(
        all_results, os.path.join(save_dir, 'decoding_cell_counts.pdf'))

    print(f"\nAll outputs written to {save_dir}")
    return all_results


def ffNLD():
    parser = argparse.ArgumentParser(
        description='Population linear decoding: neural activity -> behavior, '
                    'grouped by visual area.')
    parser.add_argument('--pooled',  type=str, required=True,
                        help='Path to pooled data HDF5 (contains visual_area_id)')
    parser.add_argument('--datadir', type=str, nargs='+', required=True,
                        help='Root directory/directories to search for preproc .h5 files')
    parser.add_argument('--outdir',  type=str, default=None,
                        help='Output directory (default: same folder as --pooled)')
    parser.add_argument('--alpha',   type=float, default=1.0,
                        help='Ridge regularisation alpha (default: 1.0)')
    parser.add_argument('--mincells', type=int, default=20,
                        help='Minimum cells per area to decode (default: 20)')
    parser.add_argument('--speed',   type=float, default=2.0,
                        help='Speed threshold cm/s (default: 2.0)')
    parser.add_argument('--signal',  type=str, default='spikes',
                        choices=['spikes', 'dff'],
                        help='Neural signal to decode from: spikes (norm_spikes) '
                             'or dff (norm_dFF) (default: spikes)')
    args = parser.parse_args()

    out = args.outdir or os.path.dirname(args.pooled)
    config = _DEFAULT_CONFIG.copy()
    config['alpha']     = args.alpha
    config['min_cells'] = args.mincells
    config['speed_thr'] = args.speed
    config['signal']    = args.signal

    run_decoding_analysis(args.pooled, args.datadir, out, config)


if __name__ == '__main__':
    ffNLD()
