
if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter

from .utils.files import read_h5
from .utils.paths import find

matplotlib.rcParams['axes.spines.top']   = False
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['pdf.fonttype']      = 42
matplotlib.rcParams['ps.fonttype']       = 42
matplotlib.rcParams['font.size']         = 9

SPEED_THRESH  = 2.0
OCC_BINS      = 40 
OCC_SIGMA     = 1.2
POLAR_BINS    = 36
TORTUOSITY_WIN = 30

LIGHT_COLOR = '#E8A838'
DARK_COLOR  = '#5B7FA6'
LINE_COLOR  = '#AAAAAA'

DEFAULT_DIR = (
    '/home/dylan/Fast2/freely_moving_data/V1PPC/cohort03_recordings'
)


def find_preproc_files(root_dir):

    import glob
    hits = sorted(glob.glob(os.path.join(root_dir, '**', '*preproc.h5'), recursive=True))
    if not hits:
        raise FileNotFoundError(f'No *preproc.h5 files found under {root_dir}')
    print(f'  Found {len(hits)} preproc.h5 file(s) in {root_dir}')
    return hits



def _ltdk_mask(pdata, n):

    if 'ltdk_state_vec' in pdata:
        ltdk = np.asarray(pdata['ltdk_state_vec'], dtype=float)[:n]
        light = ltdk > 0.5
    else:
        light = np.zeros(n, dtype=bool)
        lo = np.asarray(pdata.get('light_onsets', []), dtype=int)
        dk = np.asarray(pdata.get('dark_onsets',  []), dtype=int)
        for l in lo:
            nxt = dk[dk > l]
            end = int(nxt[0]) if len(nxt) else n
            light[l:min(end, n)] = True
    return light, ~light


def _tortuosity(x_cm, y_cm, win):

    n = len(x_cm)
    vals = []
    for i in range(0, n - win, win // 2):
        sx, sy = x_cm[i:i + win], y_cm[i:i + win]
        ok = np.isfinite(sx) & np.isfinite(sy)
        if ok.sum() < 3:
            continue
        sx, sy = sx[ok], sy[ok]
        path = float(np.sum(np.sqrt(np.diff(sx)**2 + np.diff(sy)**2)))
        disp = float(np.sqrt((sx[-1] - sx[0])**2 + (sy[-1] - sy[0])**2))
        if disp > 0.5:
            vals.append(path / disp)
    return float(np.nanmean(vals)) if vals else np.nan


def _scatter_panel(ax, l_vals, d_vals, ylabel):

    rng = np.random.default_rng(0)
    for xi, vals, color in [(0, l_vals, LIGHT_COLOR), (1, d_vals, DARK_COLOR)]:
        ok = np.isfinite(vals)
        if not ok.any():
            continue
        jitter = rng.uniform(-0.13, 0.13, ok.sum())
        ax.scatter(xi + jitter, vals[ok], color=color, s=22, alpha=0.75,
                   edgecolors='none', zorder=3)
        m = float(np.nanmean(vals[ok]))
        sem = float(np.nanstd(vals[ok]) / np.sqrt(ok.sum()))
        ax.hlines(m, xi - 0.22, xi + 0.22, color=color, lw=2.5, zorder=4)
        ax.vlines(xi, m - sem, m + sem, color=color, lw=2.5, zorder=4)

    n = min(len(l_vals), len(d_vals))
    for lv, dv in zip(l_vals[:n], d_vals[:n]):
        if np.isfinite(lv) and np.isfinite(dv):
            ax.plot([0, 1], [lv, dv], color=LINE_COLOR, lw=0.7, alpha=0.45, zorder=2)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['light', 'dark'], fontsize=8)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel(ylabel, fontsize=8)


def load_recording(path):

    pdata = read_h5(path)

    twopT = np.asarray(pdata['twopT'], dtype=float)
    n     = len(twopT)
    dt    = np.empty(n)
    dt[1:] = np.diff(twopT)
    dt[0]  = dt[1] if n > 1 else 1.0 / 30.0

    p2c   = float(pdata.get('pxls2cm', 1.0))  # divide pixels by p2c to get cm
    speed = np.asarray(pdata['speed'],  dtype=float)[:n] if 'speed'  in pdata else np.full(n, np.nan)
    hx    = np.asarray(pdata['head_x'], dtype=float)[:n] if 'head_x' in pdata else np.full(n, np.nan)
    hy    = np.asarray(pdata['head_y'], dtype=float)[:n] if 'head_y' in pdata else np.full(n, np.nan)
    yaw   = np.asarray(pdata.get('head_yaw_deg', np.full(n, np.nan)), dtype=float)[:n]

    x_cm = hx / p2c
    y_cm = hy / p2c

    # Arena center in cm
    if all(k in pdata for k in ('arenaBL', 'arenaBR', 'areaTL', 'arenaTR')):
        cxp = np.mean([pdata['arenaBL']['x'], pdata['arenaBR']['x'],
                       pdata['areaTL']['x'], pdata['arenaTR']['x']])
        cyp = np.mean([pdata['arenaBL']['y'], pdata['arenaBR']['y'],
                       pdata['areaTL']['y'], pdata['arenaTR']['y']])
        cx, cy = cxp / p2c, cyp / p2c
    else:
        cx = float(np.nanmedian(x_cm[np.isfinite(x_cm)]))
        cy = float(np.nanmedian(y_cm[np.isfinite(y_cm)]))

    light_mask, dark_mask = _ltdk_mask(pdata, n)

    # Eye tracking success rate per condition
    pct_tracked = {'light': np.nan, 'dark': np.nan}
    if 'eyeT_trim' in pdata and 'theta_trim' in pdata:
        eyeT  = np.asarray(pdata['eyeT_trim'],  dtype=float)
        theta = np.asarray(pdata['theta_trim'], dtype=float)
        n_eye = len(eyeT)
        ltdk_f = np.asarray(pdata.get('ltdk_state_vec', np.zeros(n)), dtype=float)[:n]
        ltdk_eye = np.interp(eyeT, twopT, ltdk_f) > 0.5
        for cond, emask in [('light', ltdk_eye), ('dark', ~ltdk_eye)]:
            idx = emask[:n_eye]
            if idx.sum() > 0:
                pct_tracked[cond] = float(np.sum(np.isfinite(theta[:n_eye][idx])) /
                                          idx.sum() * 100)

    out = {'arena_center': (cx, cy)}

    for cond, mask in [('light', light_mask), ('dark', dark_mask)]:
        sp  = speed[mask]
        xi  = x_cm[mask]
        yi  = y_cm[mask]
        dti = dt[mask]
        yi_v = yaw[mask]
        nf   = int(mask.sum())

        if nf == 0:
            out[cond] = {k: np.nan for k in (
                'pct_moving', 'pct_tracked', 'total_dist_cm',
                'speed_integral_cm', 'mean_speed', 'tortuosity')}
            out[f'{cond}_xy']  = (np.array([]), np.array([]))
            out[f'{cond}_yaw'] = np.array([])
            continue

        dx = np.diff(xi, prepend=xi[0])
        dy = np.diff(yi, prepend=yi[0])

        out[cond] = {
            'pct_moving':        float(np.sum(sp > SPEED_THRESH) / nf * 100),
            'pct_tracked':       pct_tracked[cond],
            'total_dist_cm':     float(np.nansum(np.sqrt(dx**2 + dy**2))),
            'speed_integral_cm': float(np.nansum(sp * dti)),
            'mean_speed':        float(np.nanmean(sp[np.isfinite(sp)])),
            'tortuosity':        _tortuosity(xi, yi, TORTUOSITY_WIN),
        }
        out[f'{cond}_xy']  = (xi, yi)
        out[f'{cond}_yaw'] = yi_v

    return out



def collect_metrics(paths):

    _keys = ('pct_moving', 'pct_tracked', 'total_dist_cm',
             'speed_integral_cm', 'mean_speed', 'tortuosity')
    metrics = {c: {k: [] for k in _keys} for c in ('light', 'dark')}

    occ_raw    = {'light': [], 'dark': []}
    polar_raw  = {'light': [], 'dark': []}
    all_x, all_y = [], []

    recs = []
    for p in paths:
        try:
            r = load_recording(p)
            recs.append(r)
            for cond in ('light', 'dark'):
                xi, yi = r[f'{cond}_xy']
                all_x.extend(xi[np.isfinite(xi)].tolist())
                all_y.extend(yi[np.isfinite(yi)].tolist())
        except Exception as e:
            print(f'  Warning: skipping {p}: {e}')

    if not recs:
        raise RuntimeError('No recordings could be loaded.')

    xp = np.percentile(all_x, [2, 98]) if all_x else [0, 1]
    yp = np.percentile(all_y, [2, 98]) if all_y else [0, 1]
    xe = np.linspace(xp[0], xp[1], OCC_BINS + 1)
    ye = np.linspace(yp[0], yp[1], OCC_BINS + 1)

    ang_edges = np.linspace(-180, 180, POLAR_BINS + 1)

    for r in recs:
        cx, cy = r['arena_center']

        for cond in ('light', 'dark'):
            m = r[cond]
            for k in _keys:
                metrics[cond][k].append(m.get(k, np.nan))

            xi, yi = r[f'{cond}_xy']
            if len(xi) > 0:
                h, _, _ = np.histogram2d(xi, yi, bins=[xe, ye])
                occ_raw[cond].append(h)

            yaw_c = r[f'{cond}_yaw']
            if len(xi) > 0 and len(yaw_c) > 0:
                n = min(len(xi), len(yaw_c))
                world_ang = np.degrees(np.arctan2(cy - yi[:n], cx - xi[:n]))
                ego_ang   = (world_ang - yaw_c[:n] + 180) % 360 - 180
                ok = np.isfinite(ego_ang)
                counts, _ = np.histogram(ego_ang[ok], bins=ang_edges)
                polar_raw[cond].append(counts.astype(float))

    for cond in ('light', 'dark'):
        for k in _keys:
            arr = np.array(metrics[cond][k], dtype=float)
            arr[arr == 0.0] = np.nan
            metrics[cond][k] = arr

    avg_occ = {}
    for cond in ('light', 'dark'):
        if occ_raw[cond]:
            avg_occ[cond] = gaussian_filter(
                np.nanmean(np.stack(occ_raw[cond], 0), 0), sigma=OCC_SIGMA)
        else:
            avg_occ[cond] = np.zeros((OCC_BINS, OCC_BINS))

    avg_polar = {}
    for cond in ('light', 'dark'):
        if polar_raw[cond]:
            m = np.nanmean(np.stack(polar_raw[cond], 0), 0)
            avg_polar[cond] = m / (m.sum() + 1e-12)
        else:
            avg_polar[cond] = np.zeros(POLAR_BINS)

    ang_centers = (ang_edges[:-1] + ang_edges[1:]) / 2.0

    return metrics, avg_occ, avg_polar, ang_centers, xe, ye



def make_figure(metrics, avg_occ, avg_polar, ang_centers, xe, ye, save_path):

    fig = plt.figure(figsize=(20, 9))
    gs  = gridspec.GridSpec(2, 12, figure=fig,
                            hspace=0.60, wspace=0.50,
                            left=0.05, right=0.97, top=0.92, bottom=0.09)

    scatter_specs = [
        (gs[0, 0:2],  'pct_moving',        '% time moving'),
        (gs[0, 2:4],  'pct_tracked',       '% frames tracked'),
        (gs[0, 4:6],  'total_dist_cm',     'total distance (cm)'),
        (gs[0, 6:8],  'speed_integral_cm', 'cumsum(speed) * dt (cm)'),
        (gs[0, 8:10], 'mean_speed',        'mean speed (cm/s)'),
        (gs[0, 10:12],'tortuosity',        'mean tortuosity'),
    ]

    lm, dm = metrics['light'], metrics['dark']

    for spec, key, ylabel in scatter_specs:
        ax = fig.add_subplot(spec)
        _scatter_panel(ax, lm[key], dm[key], ylabel)

    vmax = max(avg_occ['light'].max(), avg_occ['dark'].max(), 1e-6)

    for spec, cond, title in [
        (gs[1, 0:3],  'light', 'Occupancy  (light)'),
        (gs[1, 3:6],  'dark',  'Occupancy  (dark)'),
    ]:
        ax = fig.add_subplot(spec)
        im = ax.imshow(avg_occ[cond].T,
                       origin='lower',
                       extent=[xe[0], xe[-1], ye[0], ye[-1]],
                       cmap='hot', vmin=0, vmax=vmax,
                       aspect='equal', interpolation='bilinear')
        ax.set_xlabel('x (cm)', fontsize=8)
        ax.set_ylabel('y (cm)', fontsize=8)
        ax.set_title(title, fontsize=10)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('mean frames', fontsize=7)
        cb.ax.tick_params(labelsize=7)

    ang_rad = np.radians(ang_centers)
    dang    = ang_rad[1] - ang_rad[0]

    for spec, cond, title, color in [
        (gs[1, 6:9],  'light', 'Egocentric bearing to center\n(light)', LIGHT_COLOR),
        (gs[1, 9:12], 'dark',  'Egocentric bearing to center\n(dark)',  DARK_COLOR),
    ]:
        ax = fig.add_subplot(spec, projection='polar')
        vals = avg_polar[cond]
        ax.bar(ang_rad, vals, width=dang * 0.92, color=color,
               alpha=0.85, edgecolor='none', linewidth=0)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_xticklabels(['0°\n(ahead)', '45°', '90°\n(right)',
                            '135°', '180°\n(behind)', '-135°',
                            '-90°\n(left)', '-45°'], fontsize=6)
        ax.set_yticklabels([])
        ax.grid(True, alpha=0.25)
        ax.set_title(title, fontsize=9, pad=14)

    n_recs = int(np.sum(np.isfinite(lm['pct_moving'])))
    fig.suptitle(f'Light vs Dark Behavior  (n = {n_recs} recordings)',
                 fontsize=12, fontweight='bold')

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {save_path}')


def main(paths=None, root_dir=None, save_path=None):
    if paths is None:
        root_dir = root_dir or DEFAULT_DIR
        paths = find_preproc_files(root_dir)
    if save_path is None:
        out_dir = root_dir or os.path.dirname(os.path.abspath(paths[0]))
        save_path = os.path.join(out_dir, 'light_dark_behavior.pdf')

    print(f'Loading {len(paths)} recording(s) ...')
    metrics, avg_occ, avg_polar, ang_centers, xe, ye = collect_metrics(paths)

    n = {c: int(np.sum(np.isfinite(metrics[c]['pct_moving'])))
         for c in ('light', 'dark')}
    print(f'  Loaded: {n["light"]} light, {n["dark"]} dark condition segments')

    print('Building figure ...')
    make_figure(metrics, avg_occ, avg_polar, ang_centers, xe, ye, save_path)


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(
        description='Compare behavioral metrics between light and dark conditions.')
    parser.add_argument('--dir', default=None,
                        help=f'Root directory to search for preproc.h5 files '
                             f'(default: {DEFAULT_DIR})')
    parser.add_argument('paths', nargs='*', default=None,
                        help='Explicit preproc.h5 paths (overrides --dir)')
    parser.add_argument('--out', default=None, help='Output PDF path')
    args = parser.parse_args()
    main(paths=args.paths or None, root_dir=args.dir, save_path=args.out)
