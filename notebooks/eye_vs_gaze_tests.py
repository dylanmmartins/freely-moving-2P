
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, 'cross_area_summary.ipynb')
OUT_DIR = os.path.join(HERE, 'manuscript_figures_py_v02')

W_GRID = np.round(np.arange(-2.0, 3.0 + 1e-9, 0.25), 2)
N_SIM = 2
TAG, VERSION = 'gazeweight', 1


def _setup():
    """ Run the notebook cells this needs (session table, refit helpers, stored J/AE results) into globals. """
    cells = json.load(open(NB))['cells']
    by_id = {c.get('id'): c for c in cells}
    g = globals()
    for cid in ('c001', 'c003', 'c005', 'n_refit_helpers', 'refit-store', 'n_refit_loop', 'ae0cd'):
        src = ''.join(by_id[cid]['source'])
        src = '\n'.join(l for l in src.split('\n') if not l.strip().startswith('%'))
        if cid == 'c003':
            src = src.replace("SAVE_DIR = '/home/dylan/Documents/Github/freely-moving-2P/notebooks/'", 'SAVE_DIR = None')
        if cid == 'ae0cd':
            src = src.split('\nAE = {}')[0]         # definitions only; the stored AE results are read below
        exec(compile(src, cid, 'exec'), g)


def gaze_weight_session(R, tuned, seed=0):
    D = R['D']
    A = ae_designs(D)
    use, mz, prod, B = A['use'], A['mz'], A['prod'], A['B']
    idx = np.flatnonzero(use)
    cells = np.flatnonzero(tuned)
    if len(idx) < 3 * NULL_MIN_SHIFT or len(cells) == 0:
        return None
    Y = D['Y'][:, cells]
    folds = na._folds(D, use)
    rng = np.random.default_rng(seed)
    hd, e = A['hd'], D['e']
    EYE = prod(mz[:, None])
    GW = lambda ang: prod(np.nan_to_num(_cs(ang)))
    X_he, X_g = np.c_[B, GW(hd), EYE], np.c_[B, GW(A['gw'])]
    X_w = [np.c_[B, GW(hd + w * e), EYE] for w in W_GRID]

    # eye-position thirds (light, usable frames); heading gain fit separately in each
    lo_e, hi_e = np.nanpercentile(e[use], [100 / 3, 200 / 3])
    T = np.stack([e < lo_e, (e >= lo_e) & (e < hi_e), e >= hi_e], 1).astype(float) * use[:, None]
    cs_h = np.nan_to_num(_cs(hd))
    split = np.concatenate([B, EYE, prod(T)] + [prod(cs_h * T[:, [j]]) for j in range(3)], 1)
    n0 = B.shape[1] + EYE.shape[1] + prod(T).shape[1]
    k = na.N_VM
    vr_grid = na._vm_basis(AE_GRID)
    e_third = np.array([np.nanmean(e[use & (T[:, j] > 0)]) for j in range(3)])

    def heading_vectors(X, Yc, start, n_sets):
        """ Complex preferred-heading vector of the heading gain, summed over the visible retinal grid. """
        f = full_fit(X, Yc, use, na._cv(X, Yc, folds)['lam'])
        out = []
        for j in range(n_sets):
            s0 = start + j * 2 * k
            a = vr_grid @ (f['coef'][s0:s0 + k] / f['x_sd'][s0:s0 + k, None])
            b = vr_grid @ (f['coef'][s0 + k:s0 + 2 * k] / f['x_sd'][s0 + k:s0 + 2 * k, None])
            out.append(np.sum(a + 1j * b, 0))     # gain(h) = Re(conj(z) e^{ih}) at each rho; pref heading = arg z
        return np.stack(out)

    def run(Yc):
        r2 = np.stack([na._cv(X, Yc, folds)['r2'] for X in X_w])          # (w, cells)
        z_all = heading_vectors(X_he, Yc, B.shape[1], 1)[0]
        z_thirds = heading_vectors(split, Yc, n0, 3)
        return r2, z_all, z_thirds

    out = {'cells': cells, 'w_grid': W_GRID, 'e_third': e_third}
    out['r2'], out['z_all'], out['z_thirds'] = run(Y)
    for kind, X in (('sim_eye', X_he), ('sim_gw', X_g)):
        fit = full_fit(X, Y, use, na._cv(X, Y, folds)['lam'])
        G = np.where(use[:, None], predict(fit, X), 0.0)
        res = np.where(use[:, None], Y - G, 0.0)
        r2s, zas, zts = [], [], []
        for _ in range(N_SIM):
            Ys = G.copy()
            sh = rng.integers(NULL_MIN_SHIFT, len(idx) - NULL_MIN_SHIFT, len(cells))
            Ys[idx] += np.stack([np.roll(res[idx, j], sh[j]) for j in range(len(cells))], 1)
            a, b, c = run(Ys)
            r2s.append(a), zas.append(b), zts.append(c)
        out[kind + '_r2'], out[kind + '_z_all'], out[kind + '_z_thirds'] = np.stack(r2s), np.stack(zas), np.stack(zts)
    return out


def compute():
    _setup()
    restore('J_L')
    restore('AE')
    res = {}
    t_all = time.time()
    for i, s in SES.iterrows():
        if s['session'] not in GN:
            continue

        def fn():
            try:
                R, why = rebuild_session(s)
            except (OSError, KeyError):
                R = None
            return None if R is None else gaze_weight_session(R, GN[s['session']]['tuned'].astype(bool))

        t0 = time.time()
        r = cached(s, TAG, {'w': list(W_GRID), 'sims': N_SIM, 'v': VERSION}, fn)
        if r is not None:
            res[s['session']] = r
        print('[{}/{}] {}: {:.0f} s'.format(i + 1, len(SES), s['session'], time.time() - t0), flush=True)
    print('{} sessions in {:.1f} min'.format(len(res), (time.time() - t_all) / 60))
    area = AEC.set_index(['session', 'cell'])['area']
    return res, area




# --------------------------------------------------------------------------------------------------------------
# Summary and figures
# --------------------------------------------------------------------------------------------------------------
SRCS = ('real', 'sim_eye', 'sim_gw')
SPLIT_W = np.round(np.arange(-3.0, 4.0 + 1e-9, 0.05), 2)
BOOT = 1000


def cell_table(res, area):
    """ Per-cell arrays pooled over sessions: dR2(w) vs w = 0, heading vectors, eye-third means. """
    i0 = int(np.flatnonzero(W_GRID == 0)[0])
    rows = {'area': [], 'session': [], 'de': []}
    T = {s: {'dr2': [], 'z_all': [], 'z3': []} for s in SRCS}
    for sess, r in res.items():
        c = np.asarray(r['cells'], int)
        a = np.array([area.get((sess, int(x)), 'none') for x in c])
        rows['area'] += list(a)
        rows['session'] += [sess] * len(c)
        rows['de'] += [float(r['e_third'][2] - r['e_third'][0])] * len(c)
        for s in SRCS:
            p = '' if s == 'real' else s + '_'
            r2 = np.asarray(r[p + 'r2'])
            z_all, z3 = np.asarray(r[p + 'z_all']), np.asarray(r[p + 'z_thirds'])
            if s == 'real':
                r2, z_all, z3 = r2[None], z_all[None], z3[None]
            T[s]['dr2'].append(r2 - r2[:, [i0]])          # (draws, w, cells)
            T[s]['z_all'].append(z_all)                   # (draws, cells)
            T[s]['z3'].append(z3)                         # (draws, 3, cells)
    for s in SRCS:
        T[s] = {k: np.concatenate(v, -1) for k, v in T[s].items()}
    rows = {k: np.asarray(v) for k, v in rows.items()}
    return rows, T


def weight_curve(T, s, m):
    return np.nanmean(T[s]['dr2'][:, :, m], (0, 2))


def weight_mix(T, m, w=1.0):
    """ Fraction gaze-in-world: mix of the two simulations whose mean dR2 at w matches the real cells'. """
    j = int(np.flatnonzero(W_GRID == w)[0])
    S = {s: np.nanmean(T[s]['dr2'][:, j][:, m]) for s in SRCS}
    return (S['sim_eye'] - S['real']) / (S['sim_eye'] - S['sim_gw'])


def split_profile(T, s, m, de):
    """ Alignment of the high- and low-eye heading gains after rotating the high one back by w x delta e;
    the peak w is the fraction of an eye movement that moves the heading gain (0 eye-in-head, 1 gaze-in-world). """
    z3 = T[s]['z3'][:, :, m]                                   # (draws, 3, cells)
    u = z3[:, 2] * np.conj(z3[:, 0])                           # angle = pref heading high - low
    rot = np.exp(1j * np.radians(SPLIT_W[:, None, None] * de[m][None, None, :]))
    return np.real(np.nansum(u[None] * rot, (1, 2))) / np.nansum(np.abs(u))


def peak(x, y):
    j = int(np.nanargmax(y))
    if 0 < j < len(x) - 1:
        a, b, c = y[j - 1], y[j], y[j + 1]
        d = a - 2 * b + c
        if d < 0:
            return x[j] + 0.5 * (a - c) / d * (x[1] - x[0])
    return x[j]


def aligned_curves(T, s, m, h=np.arange(-180, 181, 5.0)):
    """ Mean heading-gain curve of the low and high eye thirds, each cell rotated to its all-frames preferred
    heading and scaled by its all-frames gain amplitude. """
    za = T[s]['z_all'][:, m]
    z3 = T[s]['z3'][:, :, m]
    v = z3 * np.conj(za)[:, None] / np.maximum(np.abs(za), 1e-12)[:, None] / np.nanmean(np.abs(za))
    V = np.nanmean(v, (0, 2))                                   # (3,) complex
    return h, np.real(np.conj(V)[:, None] * np.exp(1j * np.radians(h))[None])


def boot(fn, n, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([fn(rng.integers(0, n, n)) for _ in range(BOOT)])


def draw(res, area, AEC):
    import make_summary_figures_v02 as V
    from scipy.stats import trim_mean, fisher_exact
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt = V.plt
    rows, T = cell_table(res, area)
    areas = V.AREAS
    names = areas + ['all']
    four = np.isin(rows['area'], areas)
    groups = {a: np.flatnonzero(four if a == 'all' else rows['area'] == a) for a in names}
    col = {'real': V.REAL, 'sim_eye': V.EYE_GAIN, 'sim_gw': '#8c564b'}
    lab = {'real': 'real', 'sim_eye': 'simulated eye-in-head', 'sim_gw': 'simulated gaze-in-world'}

    def scan(s, idx):
        x = np.moveaxis(T[s]['dr2'][:, :, idx], 1, 0).reshape(len(W_GRID), -1)
        return trim_mean(x, 0.05, axis=1)

    def split(s, idx):
        return split_profile(T, s, idx, rows['de'])

    stats = {}
    for a, idx in groups.items():
        stats[a] = {'scan': peak(W_GRID, scan('real', idx)), 'split': peak(SPLIT_W, split('real', idx)),
                    'scan_b': boot(lambda i: peak(W_GRID, scan('real', idx[i])), len(idx)),
                    'split_b': boot(lambda i: peak(SPLIT_W, split('real', idx[i])), len(idx))}
        for s in ('sim_eye', 'sim_gw'):
            stats[a]['scan_' + s] = peak(W_GRID, scan(s, idx))
            stats[a]['split_' + s] = peak(SPLIT_W, split(s, idx))

    fig = plt.figure(figsize=(V.WIDTH, 8.4), layout='constrained')
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1])
    leg = [Line2D([], [], color=col[s], lw=1.6) for s in SRCS]

    def curve_panel(ax, x, ys, ylabel, letter_):
        for s in SRCS:
            ax.plot(x, ys[s], color=col[s], lw=1.6, zorder=3 if s == 'real' else 2)
        ax.axvline(0, color=V.EYE_GAIN, lw=0.85, ls='--')
        ax.axvline(1, color=col['sim_gw'], lw=0.85, ls='--')
        ax.axhline(0, color=V.INK2, lw=0.45) if min(np.nanmin(v) for v in ys.values()) < 0 else None
        ax.set_xlabel('eye weight w  (heading + w x eye)')
        ax.set_ylabel(ylabel)
        V.legend_above(ax, leg, [lab[s] for s in SRCS], 2)
        V.letter(ax, letter_)

    def peak_panel(ax, key, lim, letter_):
        ax.axhline(0, color=V.EYE_GAIN, lw=0.85, ls='--')
        ax.axhline(1, color=col['sim_gw'], lw=0.85, ls='--')
        for i, a in enumerate(names):
            c = V.AREA_COLOR.get(a, V.INK)
            lo, hi = np.clip(np.nanpercentile(stats[a][key + '_b'], [2.5, 97.5]), *lim)
            ax.plot([i, i], [lo, hi], color=c, lw=V.ERR_LW, zorder=3)
            ax.plot(i, stats[a][key], 'o', color=c, ms=4.5, zorder=4)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_ylim(*lim)
        ax.set_ylabel('best eye weight w')
        ax.text(1.02, 0, 'eye-in-\nhead', transform=ax.get_yaxis_transform(), color=V.EYE_GAIN, va='center',
                fontsize=V.FONT)
        ax.text(1.02, 1, 'gaze-in-\nworld', transform=ax.get_yaxis_transform(), color=col['sim_gw'], va='center',
                fontsize=V.FONT)
        V.letter(ax, letter_)

    # A/B: eye-weight scan
    ax = fig.add_subplot(gs[0, 0])
    curve_panel(ax, W_GRID, {s: 1e4 * scan(s, groups['all']) for s in SRCS}, 'CV ' + V.R2 + ' gain vs w = 0\n(x1e-4)', 'A')
    ax = fig.add_subplot(gs[0, 1])
    peak_panel(ax, 'scan', (W_GRID[0], W_GRID[-1]), 'B')

    # C/D: heading gain split by eye position
    ax = fig.add_subplot(gs[1, 0])
    m = (SPLIT_W >= -2) & (SPLIT_W <= 3)
    curve_panel(ax, SPLIT_W[m], {s: split(s, groups['all'])[m] for s in SRCS},
                'alignment, high vs low\neye heading gain', 'C')
    ax = fig.add_subplot(gs[1, 1])
    peak_panel(ax, 'split', (-2, 3), 'D')

    # E: eye-in-head gain beyond the gaze-in-world gain (stored notebook AE calls)
    ax = fig.add_subplot(gs[2, 0])
    D = AEC[AEC['area'].isin(areas)]
    srcs = [('real', V.REAL, 'real'), ('sim_eye', V.EYE_GAIN, 'simulated eye-in-head'),
            ('sim_gw', col['sim_gw'], 'simulated gaze-in-world'), ('null', V.SHUF, 'eye shuffled')]
    w4 = 0.2
    hi = 0
    E = {}
    for i, a in enumerate(areas):
        d = D[D['area'] == a]
        for j, (s, c, _) in enumerate(srcs):
            if s == 'real':
                v = d['eye_beyond_gw'].mean()
            else:
                n = 10 if s == 'null' else 2
                v = np.mean([d['{}_eye_beyond_gw_{}'.format(s, k)].mean() for k in range(n)])
            sim = s.startswith('sim')
            ax.bar(i + (j - 1.5) * w4, v, width=w4 * 0.9, color=c, hatch=V.HATCH if sim else None,
                   edgecolor=V.INK2 if sim else 'none', lw=0.28)
            if s == 'real':
                k = int(d['eye_beyond_gw'].sum())
                _, l, h = V.wilson(k, len(d))
                V.errbar(ax, i + (j - 1.5) * w4, l, h)
                v = h
            hi = max(hi, v)
        k = int(d['eye_beyond_gw'].sum())
        sg = np.concatenate([d['sim_gw_eye_beyond_gw_{}'.format(q)].to_numpy(float) for q in range(2)])
        E[a] = fisher_exact([[k, len(d) - k], [int(sg.sum()), len(sg) - int(sg.sum())]])[1]
    V.area_ticks(ax)
    ax.set_ylim(0, 1.08 * hi)
    ax.set_ylabel('frac. tuned cells, eye gain\nadds beyond gaze-in-world')
    V.legend_above(ax, [Patch(facecolor=c, hatch=V.HATCH if s.startswith('sim') else None,
                              edgecolor=V.INK2 if s.startswith('sim') else 'none') for s, c, _ in srcs],
                   [l for _, _, l in srcs], 2)
    V.letter(ax, 'E')
    V.save(fig, 'Gaze_tests')

    # numbers for the text
    for a in names:
        st = stats[a]
        print('{:4s} n={:5d}  scan peak {:5.2f} [{:5.2f}, {:5.2f}] (sim eye {:5.2f}, gw {:5.2f})   split peak {:5.2f} '
              '[{:5.2f}, {:5.2f}] (sim eye {:5.2f}, gw {:5.2f})'.format(
                  a, len(groups[a]), st['scan'], *np.nanpercentile(st['scan_b'], [2.5, 97.5]), st['scan_sim_eye'],
                  st['scan_sim_gw'], st['split'], *np.nanpercentile(st['split_b'], [2.5, 97.5]),
                  st['split_sim_eye'], st['split_sim_gw']))
    print('E: real vs simulated gaze-in-world, Fisher p:', {a: '{:.2g}'.format(p) for a, p in E.items()})


if __name__ == '__main__':
    import pickle
    pk = os.path.join('/home/dylan/Storage/freely_moving_data/V1PPC/cross_area_cache', 'gazeweight_all.pkl')
    if 'recompute' in sys.argv[1:] or not os.path.isfile(pk):
        res, area = compute()
        with open(pk, 'wb') as f:
            pickle.dump({'res': res, 'area': area}, f)
    _setup()
    restore('AE')
    with open(pk, 'rb') as f:
        P = pickle.load(f)
    draw(P['res'], P['area'], AEC)
