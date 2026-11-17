
import glob
import os
import re
import sys
import time

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from fm2p import new_analysis as na  # noqa: E402
from fm2p.ffNLE_scene import _svd_prep, _ridge_coefs  # noqa: E402

ROOTS = [os.path.join('/home/dylan/Storage/freely_moving_data/V1PPC', c)
         for c in ('cohort01_recordings', 'cohort02_recordings')]
CACHE = os.path.join(HERE, 'manuscript_figures', 'gaze_controls')
VERSION = 7

FS_B = 60.0
DT_B = 1.0 / FS_B
SG_WIN_B, SG_ORD_B = 7, 3
BRIDGE_N = 6
BRIDGE_MAX_JUMP = 5.0          # ... if the eye moves by no more than 5 deg beyond slow-phase counter-rotation across it
GAP_DILATE = 1                 # velocity is masked this many samples either side of a remaining gap
GAZE_ON_B = 60.0               # shift onset: |gaze| last rose through 60 deg/s (as nonvisual_probe)
MAX_WALK_B = 12                # onset search back from the first 240 deg/s sample, samples (200 ms)
SHIFT_TAIL = 0.1               # shift end = 0.1 s after its last 240 deg/s sample (as before)
COVER_PRE, COVER_POST = 0.5, 0.3   # fixation-type frames need gaze velocity throughout this margin around their
                                   # spike window, so no gaze shift can hide in a tracking gap
BEH_CACHE = os.path.join(CACHE, 'behaviour60')
BEH_VERSION = 4

EARLY_S = 0.5                  # early fixation: frame can hold spikes from the first 0.5 s after the shift
LATE_S = 0.5                   # late fixation: no spike within 0.5 s after the shift ...
PRE_S = 0.3                    # ... nor within 0.3 s before the next one
CLASS_POST_S = 0.3             # behaviour classes (Figures 13-14): 'post' = spikes possible < 0.3 s after a shift;
CLASS_PRE_S = 0.1              # compensatory / still frames >= 0.3 s after one and >= 0.1 s before the next. The
                               # population post-shift activity is gone 0.1-0.2 s into the spike window and the
                               # pre-shift rise starts < 0.1 s before onset; the fixation epochs keep 0.5 / 0.3 s
ACTIVE_FIX_S = 3.0             # fixations shorter than this = active behaviour (state-matched early vs late)
COMP_HEAD, COMP_GAZE = 60.0, 120.0   # compensatory frame: |head| > 60 deg/s and |head + eye| < 120 deg/s (frame means)
STILL_HEAD, STILL_SPEED = 10.0, 1.0  # still frame: |head| < 10 deg/s and speed < 1 cm/s
N_SHUF = 10                    # shuffled-eye gain fits per condition (null call rate)
NULL_MIN_SHIFT = 400
KERNEL_EDGES = np.round(np.arange(-0.4, 1.6 + 1e-9, 0.2), 3)   # s from shift onset
PAIR_DRHO = 5.0                # within-fixation pairs: retinal positions within 5 deg
PAIR_DE = 3.0                  # ... and eye positions at least 3 deg apart
PAIR_MAX = 20000               # pairs kept per session (random subset)
MIN_ACTIVE_PAIRS = 100         # a cell needs this many pairs whose spike count changed
MIN_ACTIVE_TRAIN = 10          # raw / population summaries: a cell needs this many active training frames (a near-
                               # silent cell's residual SD or mean rate is ~0, and dividing by it explodes)
RHO_GRID = np.arange(-175.0, 180.0, 10.0)


def _s(x):
    x = x[()] if hasattr(x, 'shape') else x
    return x.decode() if isinstance(x, (bytes, np.bytes_)) else x

def sessions():
    best = {}
    for root in ROOTS:
        for p in sorted(glob.glob(os.path.join(root, '**', '*_neural.h5'), recursive=True)):
            if re.search(r'_neural_[^/]+\.h5$', p):
                continue
            nv = glob.glob(os.path.join(os.path.dirname(p), '*_nonvisual.h5'))
            if not nv:
                continue
            sess = os.path.basename(os.path.dirname(os.path.dirname(p)))
            m = re.search(r'(DMM\d+)_(pos\d+)', sess)
            with h5py.File(p, 'r') as f:
                if int(_s(f['meta']['no_imu'])):
                    continue
                usable = float(_s(f['meta']['n_frames'])) * float(_s(f['meta']['use_frac']))
            key = (m.group(1), m.group(2))
            if key not in best or usable > best[key][0]:
                best[key] = (usable, p, nv[0], sess, key[0])
    return [best[k][1:] for k in sorted(best)]


def _bridge(x, n):
    x = np.array(x, float)
    for a, b in na._runs(~np.isfinite(x)):
        if b - a <= n and a > 0 and b < len(x):
            x[a:b] = np.interp(np.arange(a, b), [a - 1, b], [x[a - 1], x[b]])
    return x


def _sg_masked(x, deriv):
    from scipy.signal import savgol_filter
    ok = np.isfinite(x)
    if ok.sum() < SG_WIN_B:
        return np.full(len(x), np.nan), np.ones(len(x), bool)
    xf = np.interp(np.arange(len(x)), np.flatnonzero(ok), x[ok])
    y = savgol_filter(xf, SG_WIN_B, SG_ORD_B, deriv=deriv, delta=DT_B)
    gap = np.convolve((~ok).astype(float), np.ones(2 * GAP_DILATE + 1), mode='same') > 0
    y[gap] = np.nan
    return y, gap


def _window_clean(t, gap, lo, hi):
    cg = np.r_[0, np.cumsum(gap)]
    return (cg[np.searchsorted(t, hi)] - cg[np.searchsorted(t, lo)]) == 0


def behaviour60(neural_path, sess, twopT, dt):
    path = os.path.join(BEH_CACHE, sess + '.npz')
    if os.path.isfile(path):
        z = dict(np.load(path))
        if int(z['version']) == BEH_VERSION and len(z['covered']) == len(twopT):
            return z
    from fm2p.nonvisual_probe import _crossing
    import contextlib
    import io
    fs0, dt0 = na.FS, na.DT
    na.FS, na.DT = FS_B, DT_B
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            s = na.load_session(neural_path.replace('_neural.h5', '_preproc.h5'),
                                neural_path.replace('_neural.h5', '_retinal_pose.npz'))
    finally:
        na.FS, na.DT = fs0, dt0
    t = s['t']
    head, hgap = _sg_masked(_bridge(-s['gyro_z'], BRIDGE_N), 0)
    th_raw = s['theta']
    eye1, _ = _sg_masked(_bridge(th_raw, BRIDGE_N), 1)
    sl = np.isfinite(eye1) & np.isfinite(head) & (np.abs(head) > 30) & (np.abs(head) < 300) & (np.abs(eye1) < 150)
    slope = -0.6
    if sl.sum() > 100:
        _, V = np.linalg.eigh(np.cov(np.stack([head[sl], eye1[sl]])))
        slope = float(V[1, -1] / V[0, -1])
    theta = np.array(th_raw, float)
    n_bridged = n_refused = 0
    for a, b in na._runs(~np.isfinite(th_raw)):
        if b - a <= BRIDGE_N and a > 0 and b < len(theta):
            resid = (th_raw[b] - th_raw[a - 1]) - slope * np.nansum(head[a - 1:b]) * DT_B
            if abs(resid) <= BRIDGE_MAX_JUMP:
                theta[a:b] = np.interp(np.arange(a, b), [a - 1, b], [th_raw[a - 1], th_raw[b]])
                n_bridged += 1
            else:
                n_refused += 1
    eye, gap = _sg_masked(theta, 1)
    gap |= hgap
    gaze = head + eye
    v = {'head': head, 'eye': eye, 'gaze': gaze}
    with contextlib.redirect_stdout(io.StringIO()):
        lab = na.classify_gated(v)
    rows = []
    for a, b, d in na._episodes(lab, gaze):
        t_on, _ = _crossing(gaze, t, a, GAZE_ON_B, d, max_walk=MAX_WALK_B)
        rows.append((t_on, t[b - 1] + SHIFT_TAIL, a, b, d, float(np.nansum(eye[a:b]) * DT_B),
                     float(np.nansum(head[a:b]) * DT_B), bool(s['light'][a])))
    rows.sort()
    cols = ('t_on', 't_end', 'ep_start', 'ep_end', 'direction', 'eye_amp', 'head_amp', 'light')
    z = {c: np.array([r[i] for r in rows]) for i, c in enumerate(cols)}
    z['cover_pre'] = _window_clean(t, gap, z['t_on'] - ISO_PRE, z['t_on'])
    gi = np.flatnonzero(gap)
    nxt = np.searchsorted(t[gi], z['t_on'], side='left')
    z['clean_until'] = np.where(nxt < len(gi), t[gi][np.clip(nxt, None, len(gi) - 1)] - z['t_on'], np.inf)
    # per 2P frame
    z['h60'] = na._bin_to_2p(t, head, twopT)[0]
    z['g60'] = na._bin_to_2p(t, gaze, twopT)[0]
    z['covered'] = _window_clean(t, gap, twopT - dt - COVER_PRE, twopT + dt + COVER_POST)
    z['covered_b'] = _window_clean(t, gap, twopT - dt - CLASS_POST_S, twopT + dt + CLASS_PRE_S)
    z['frame_clean'] = _window_clean(t, gap, twopT - dt, twopT + dt)
    L = s['light'].astype(bool)
    z['frac_theta_light'] = np.array(np.isfinite(s['theta'][L]).mean())
    z['frac_gaze_light'] = np.array(np.isfinite(gaze[L]).mean())
    z['vor_slope'], z['n_bridged'], z['n_refused'] = np.array(slope), np.array(n_bridged), np.array(n_refused)
    z['version'] = np.array(BEH_VERSION)
    os.makedirs(BEH_CACHE, exist_ok=True)
    np.savez(path, **z)
    return z


def load(neural_path, nonvis_path):
    with h5py.File(neural_path, 'r') as f:
        cv = f['covariates']
        D = {k: cv[k][()] for k in cv.keys() if not k.startswith('group_')}
        D['names'] = _s(cv['names']).split(',')
        h5p, sk = _s(f['meta']['h5_path']), _s(f['meta']['spike_key'])
        mean_rate = f['cells']['mean_rate'][()]
        cells = {'area': np.array(_s(f['cells']['area']).split(',')),
                 'tuned': f['refframe']['eye_models']['tuned'][()].astype(bool),
                 'gain': f['refframe']['eye_models']['gain'][()].astype(bool)}
    with h5py.File(nonvis_path, 'r') as f:
        lag = int(f['lag_check']['peak_lag'][()]) if 'lag_check' in f else 0
    with h5py.File(h5p, 'r') as f:
        Yr = np.asarray(f[sk][()], float).T
        twopT = np.asarray(f['twopT'][()], float)
        n_rows = f['twop_mean_img'].shape[0]
        yc = np.array([np.mean(f['cell_y_pix/{}'.format(c)][()]) for c in range(Yr.shape[1])])
    n = len(D['t'])
    if Yr.shape[0] < n or Yr.shape[1] != len(mean_rate):
        raise ValueError('preproc activity shape {} does not match {} frames x {} cells'.format(Yr.shape, n, len(mean_rate)))
    Yr = Yr[:n]
    if not np.allclose(np.nanmean(Yr, 0), mean_rate, rtol=1e-5, atol=1e-9, equal_nan=True):
        raise ValueError('preproc activity changed since the _neural.h5 was written')
    Y = np.zeros_like(Yr)
    if lag > 0:
        Y[lag:] = Yr[:-lag]
    else:
        Y[:] = Yr
    ok_y = np.all(np.isfinite(Y), 1)
    ok_y[:lag] = False
    D['Y'] = np.nan_to_num(Y)
    D['X'] = D['X'].astype(float)
    D['use'] = D['use'].astype(bool) & ok_y
    D['light'] = D['light'].astype(bool)
    D['dt'] = float(np.median(np.diff(D['t'])))
    D['twopT'] = twopT[:n]
    cells['offset'] = yc / n_rows * D['dt']
    sess = os.path.basename(os.path.dirname(os.path.dirname(neural_path)))
    B = behaviour60(neural_path, sess, D['twopT'], D['dt'])
    ev = {k: B[k] for k in ('t_on', 't_end', 'ep_start', 'ep_end', 'eye_amp', 'head_amp', 'cover_pre', 'clean_until')}
    ev['light'] = B['light'].astype(bool)
    ev['direction'] = B['direction'].astype(int)
    ev['cover_pre'] = ev['cover_pre'].astype(bool)
    D['h60'], D['g60'] = B['h60'], B['g60']
    D['covered'] = B['covered'].astype(bool)
    D['covered_b'] = B['covered_b'].astype(bool)
    D['frame_clean'] = B['frame_clean'].astype(bool)
    return D, ev, cells, lag


def frame_epochs(D, ev):
    T = D['twopT']
    dt = D['dt']
    lo, hi = T - dt, T + dt
    on, end = ev['t_on'], ev['t_end']
    o = np.argsort(on)
    on, end = on[o], end[o]
    k_prev = np.searchsorted(end, lo, side='right') - 1
    k_next = np.searchsorted(on, hi, side='left')
    j = np.searchsorted(on, hi, side='left') - 1
    inside = np.zeros(len(T), bool)
    jj = np.clip(j, 0, len(on) - 1)
    inside |= (j >= 0) & (end[jj] > lo)
    since = np.where(k_prev >= 0, lo - end[np.clip(k_prev, 0, None)], np.inf)
    until = np.where(k_next < len(on), on[np.clip(k_next, None, len(on) - 1)] - hi, np.inf)
    fix_id = np.where(inside, -1, k_prev)
    dur = np.where(k_prev >= 0, np.r_[on[1:], np.inf][np.clip(k_prev, 0, None)] - end[np.clip(k_prev, 0, None)], np.inf)
    return {'inside': inside, 'since': since, 'until': until, 'fix': fix_id, 'fix_dur': np.where(inside, np.nan, dur),
            'order': o}


def eye_nuis_cols(names):
    nm = np.array(names)
    return np.r_[np.flatnonzero(np.char.startswith(nm, 'hd_vm')),
                 [names.index(k) for k in ('pitch', 'roll', 'speed', 'log_dist')]].astype(int)


class Design:

    def __init__(self, D, norm_rows):
        e = D['e']
        self.mu, self.sd = np.nanmean(e[norm_rows]), np.nanstd(e[norm_rows])
        self.centers = np.nanpercentile(e[norm_rows], [10, 37, 63, 90])
        self.width = max(self.sd / 1.5, 1.0)
        self.mz = self.z(e)
        self.nuis = D['X'][:, eye_nuis_cols(D['names'])]
        self.vr, self.vh = na._vm_basis(D['rho']), na._vm_basis(D['eta'])
        self.eb = na._gauss_basis(e, self.centers, self.width)
        self.block = {'NUIS': np.c_[self.nuis, self.mz], 'RET': np.c_[self.nuis, self.mz, self.vr],
                      'ADD': np.c_[self.nuis, self.mz, self.vr, self.eb],
                      'HEAD': np.c_[self.nuis, self.mz, self.vh, self.eb]}
        self.n_add = self.block['ADD'].shape[1]

    def z(self, e):
        return (np.asarray(e, float) - self.mu) / self.sd

    def gain(self, *ez, extra=None):
        cols = [self.block['ADD']] + [self.vr * np.nan_to_num(z)[:, None] for z in ez]
        if extra is not None:
            cols.append(extra)
        return np.concatenate(cols, 1)


def cv(D, X, rows):
    return na._cv(np.nan_to_num(X), D['Y'], na._folds(D, rows))


def beats(fa, fb):
    return (fa['r2'] > fb['r2']) & (np.sum(fa['r2_folds'] - fb['r2_folds'] > 0, 0) >= 4)


def full_fit(D, X, rows, lam):
    X = np.nan_to_num(X)
    xm, xs = X[rows].mean(0), X[rows].std(0)
    xs[~(xs > 0)] = 1.0
    Yu = D['Y'][rows]
    ym = Yu.mean(0)
    U, s_, Vt = _svd_prep((X[rows] - xm) / xs)
    lam = np.nanmedian(lam, 0)
    coef = _ridge_coefs(U, s_, Vt, Yu - ym, np.where(np.isfinite(lam), lam, 1e3))
    pred = ((X - xm) / xs) @ coef + ym
    return {'coef': coef, 'x_mean': xm, 'x_sd': xs, 'y_mean': ym, 'pred': pred,
            'res_sd': np.std(Yu - pred[rows], 0)}


def profiles(fit, col0):
    beta = fit['coef'] / fit['x_sd'][:, None]
    B = na._vm_basis(RHO_GRID)
    f = B @ beta[col0:col0 + na.N_VM]
    g = B @ beta[-na.N_VM:]
    return f.T, g.T


def gain_call_set(D, des, rows, n_shuf, rng, ez=None, eval_rows=None):
    ez = des.mz if ez is None else ez
    fits = {k: cv(D, X, rows) for k, X in des.block.items()}
    Xg = des.gain(ez)
    fits['GAIN'] = cv(D, Xg, rows)
    tuned = beats(fits['RET'], fits['NUIS']) | beats(fits['HEAD'], fits['NUIS'])
    idx = np.flatnonzero(rows)
    ff = full_fit(D, Xg, rows, fits['GAIN']['lam'])
    fa = full_fit(D, des.block['ADD'], rows, fits['ADD']['lam'])
    f, g = profiles(ff, des.nuis.shape[1] + 1)
    cross = {}
    base = {k: r2_on(D['Y'], fa['pred'], m) for k, m in (eval_rows or {}).items()}
    for k, m in (eval_rows or {}).items():
        cross[k] = {'gain': r2_on(D['Y'], ff['pred'], m) - base[k], 'null': []}
    nulls = []
    for _ in range(n_shuf):
        zs = ez.copy()
        k = int(rng.integers(len(idx) // 10, len(idx) - len(idx) // 10))
        zs[idx] = np.roll(ez[idx], k)
        nulls.append(cv(D, des.gain(zs), rows))
        if eval_rows:
            P = full_fit(D, des.gain(zs), rows, nulls[-1]['lam'])['pred']
            for kk, m in eval_rows.items():
                cross[kk]['null'].append(r2_on(D['Y'], P, m) - base[kk])
    for v in cross.values():
        v['null'] = np.stack(v['null']) if v['null'] else np.zeros((0, D['Y'].shape[1]))
    return cross, {'call': beats(fits['GAIN'], fits['ADD']), 'tuned': tuned,
            'd': fits['GAIN']['r2'] - fits['ADD']['r2'],
            'r2_add': fits['ADD']['r2'], 'r2_ret': fits['RET']['r2'],
            'null_call': np.stack([beats(n, fits['ADD']) for n in nulls]) if nulls else np.zeros((0, D['Y'].shape[1]), bool),
            'd_null': np.stack([n['r2'] - fits['ADD']['r2'] for n in nulls]) if nulls else np.zeros((0, D['Y'].shape[1])),
            'field': f, 'gain_prof': g / np.maximum(ff['res_sd'], 1e-12)[:, None],
            'n_frames': np.array(int(rows.sum())), 'e_sd': np.array(float(np.nanstd(D['e'][rows])))}


def _subsample(mask, n, rng):
    idx = np.flatnonzero(mask)
    out = np.zeros_like(mask)
    out[rng.choice(idx, min(n, len(idx)), replace=False)] = True
    return out

# the four tests
def light_rows(D):
    return D['light'] & D['use'] & np.isfinite(D['rho']) & np.isfinite(D['eta']) & np.isfinite(D['e'])


def epoch_masks(D, E):
    L = light_rows(D)

    fixf = L & ~E['inside'] & D['covered']
    base = {'all': L, 'fixation': fixf,
            'early': fixf & (E['since'] < EARLY_S),
            'late': fixf & (E['since'] >= LATE_S) & (E['until'] >= PRE_S),
            'short': fixf & (E['fix_dur'] < 1.5),
            'long': fixf & (E['fix_dur'] >= 2.0),
            # active periods: fixations shorter than ACTIVE_FIX_S, so early and late frames come from the same
            # behaviour (late frames otherwise come mostly from long still periods)
            'early_act': fixf & (E['since'] < EARLY_S) & (E['fix_dur'] < ACTIVE_FIX_S),
            'late_act': fixf & (E['since'] >= LATE_S) & (E['until'] >= PRE_S) & (E['fix_dur'] < ACTIVE_FIX_S)}
    # behaviour classes (Figures 13-14): gaze-shift frames, the first CLASS_POST_S after a shift, compensatory
    # head turns >= CLASS_POST_S after and >= CLASS_PRE_S before any shift (gaze steady, eye counter-rotating, no
    # saccade), and stillness; tracking must cover the same margins
    h, g = D['h60'], D['g60']                      # 60 Hz head and gaze velocity, frame means
    fixb = L & ~E['inside'] & D['covered_b']
    clear = (E['since'] >= CLASS_POST_S) & (E['until'] >= CLASS_PRE_S)
    M_ = {'shift': L & E['inside'], 'post': fixb & (E['since'] < CLASS_POST_S),
          'comp': fixb & clear & (np.abs(h) > COMP_HEAD) & (np.abs(g) < COMP_GAZE),
          'still': fixb & clear & (np.abs(h) < STILL_HEAD) & (D['speed'] < STILL_SPEED)}
    base.update(M_)
    return base


def r2_on(Y, P, rows):
    y, p = Y[rows], P[rows]
    ss = np.sum((y - y.mean(0)) ** 2, 0)
    return 1.0 - np.sum((y - p) ** 2, 0) / np.where(ss > 0, ss, np.nan)


def test_epochs(D, E, rng, n_shuf=N_SHUF):
    """ Gain fits on each epoch's own frames (early vs late in fixations, short vs long fixations, all light
    frames), each with its shuffled-eye null; and cross-epoch prediction: the gain field fitted on one epoch
    predicting the other (R2 gained over the same epoch's ADD fit, real minus shuffled-eye fits). """
    M = epoch_masks(D, E)
    des = Design(D, M['all'])
    out = {}
    partner = {'early': 'late', 'late': 'early', 'short': 'long', 'long': 'short'}
    for k in ('early', 'late', 'short', 'long', 'all'):
        ev_rows = {partner[k]: M[partner[k]]} if k in partner else None
        cross, out[k] = gain_call_set(D, des, M[k], n_shuf, rng, eval_rows=ev_rows)
        for b, v in cross.items():
            out['cross_{}_to_{}'.format(k, b)] = v
    return out


def shift_eye(D, E, ev):
    """ Eye position just before each shift (previous fixation) and just after it lands (current fixation),
    assigned to every frame of the fixation that the shift starts. """
    T, dt = D['twopT'], D['dt']
    o = E['order']
    on, end = ev['t_on'][o], ev['t_end'][o]
    e = D['e']
    ok = D['use'] & np.isfinite(e)
    k_pre = np.searchsorted(T + dt / 2, on, side='right') - 1          # last frame ending before the onset
    k_post = np.searchsorted(T - dt / 2, end, side='left')             # first frame starting after landing
    pre = np.where((k_pre >= 0) & ok[np.clip(k_pre, 0, len(T) - 1)], e[np.clip(k_pre, 0, len(T) - 1)], np.nan)
    post = np.where((k_post < len(T)) & ok[np.clip(k_post, 0, len(T) - 1)], e[np.clip(k_post, 0, len(T) - 1)], np.nan)
    fid = E['fix']
    good = fid >= 0
    e_prev, e_cur = np.full(len(T), np.nan), np.full(len(T), np.nan)
    e_prev[good], e_cur[good] = pre[fid[good]], post[fid[good]]
    return e_prev, e_cur


def test_prev(D, E, ev, rng, n_shuf=5):
    """ Gain term's eye position from the frame itself, the current fixation's landing position, or the
    previous fixation's position (just before the shift), additive terms unchanged. """
    M = epoch_masks(D, E)
    des = Design(D, M['all'])
    e_prev, e_cur = shift_eye(D, E, ev)
    zp, zc = des.z(e_prev), des.z(e_cur)
    ok = np.isfinite(zp) & np.isfinite(zc)
    out = {}
    for name in ('fixation', 'early', 'late'):
        rows = M[name] & ok
        add = cv(D, des.block['ADD'], rows)
        r = {'n_frames': np.array(int(rows.sum())),
             'corr_frame_prev': np.array(np.corrcoef(D['e'][rows], e_prev[rows])[0, 1]),
             'corr_frame_cur': np.array(np.corrcoef(D['e'][rows], e_cur[rows])[0, 1]),
             'corr_cur_prev': np.array(np.corrcoef(e_cur[rows], e_prev[rows])[0, 1])}
        fits = {'frame': cv(D, des.gain(des.mz), rows), 'cur': cv(D, des.gain(zc), rows),
                'prev': cv(D, des.gain(zp), rows), 'both': cv(D, des.gain(des.mz, zp), rows)}
        for k, f in fits.items():
            r['d_' + k] = f['r2'] - add['r2']
            r['call_' + k] = beats(f, add)
        r['call_prev_beyond_frame'] = beats(fits['both'], fits['frame'])
        r['call_frame_beyond_prev'] = beats(fits['both'], fits['prev'])
        idx = np.flatnonzero(rows)
        nb, nf = [], []
        for _ in range(n_shuf):
            k = int(rng.integers(len(idx) // 10, len(idx) - len(idx) // 10))
            zps, zfs = zp.copy(), des.mz.copy()
            zps[idx], zfs[idx] = np.roll(zp[idx], k), np.roll(des.mz[idx], k)
            nb.append(beats(cv(D, des.gain(des.mz, zps), rows), fits['frame']))
            nf.append(beats(cv(D, des.gain(zfs, zp), rows), fits['prev']))
        r['null_prev_beyond_frame'] = np.stack(nb)
        r['null_frame_beyond_prev'] = np.stack(nf)
        out[name] = r
    return out


def prev_on_rows(D, des, e_prev, rows, rng, n_shuf=5):
    """ Unique gain from the current frame's eye position vs the eye position before the last gaze shift. """
    zp = des.z(e_prev)
    rows = rows & np.isfinite(zp)
    fr = cv(D, des.gain(des.mz), rows)
    pv = cv(D, des.gain(zp), rows)
    both = cv(D, des.gain(des.mz, zp), rows)
    r = {'n_frames': np.array(int(rows.sum())), 'corr': np.array(np.corrcoef(D['e'][rows], e_prev[rows])[0, 1]),
         'call_prev_beyond_frame': beats(both, fr), 'call_frame_beyond_prev': beats(both, pv)}
    idx = np.flatnonzero(rows)
    nb, nf = [], []
    for _ in range(n_shuf):
        k = int(rng.integers(len(idx) // 10, len(idx) - len(idx) // 10))
        zps, zfs = zp.copy(), des.mz.copy()
        zps[idx], zfs[idx] = np.roll(zp[idx], k), np.roll(des.mz[idx], k)
        nb.append(beats(cv(D, des.gain(des.mz, zps), rows), fr))
        nf.append(beats(cv(D, des.gain(zfs, zp), rows), pv))
    r['null_prev_beyond_frame'], r['null_frame_beyond_prev'] = np.stack(nb), np.stack(nf)
    return r


def kernel_features(D, ev):
    """ Gaze-shift transient regressors: per direction, onsets counted and |eye amplitude|-weighted in
    0.2 s bins of time since onset (-0.4 to 1.6 s), at each frame's sample time. """
    T = D['twopT']
    cols = []
    for d in (1, -1):
        m = ev['direction'] == d
        on = ev['t_on'][m]
        o = np.argsort(on)
        on = on[o]
        amp = np.nan_to_num(np.abs(ev['eye_amp'][m][o]))
        ca = np.r_[0.0, np.cumsum(amp)]
        for a, b in zip(KERNEL_EDGES[:-1], KERNEL_EDGES[1:]):
            hi = np.searchsorted(on, T - a, side='right')
            lo = np.searchsorted(on, T - b, side='right')
            cols += [(hi - lo).astype(float), ca[hi] - ca[lo]]
    return np.stack(cols, 1)


def test_kernels(D, ev, rng, n_shuf=N_SHUF):
    """ ADD and GAIN with and without gaze-shift transient regressors, on all light frames. """
    rows = light_rows(D)
    des = Design(D, rows)
    K = kernel_features(D, ev)
    f = {'ADD': cv(D, des.block['ADD'], rows), 'GAIN': cv(D, des.gain(des.mz), rows),
         'ADDK': cv(D, np.c_[des.block['ADD'], K], rows), 'GAINK': cv(D, des.gain(des.mz, extra=K), rows)}
    idx = np.flatnonzero(rows)
    nulls = []
    for _ in range(n_shuf):
        zs = des.mz.copy()
        zs[idx] = np.roll(des.mz[idx], int(rng.integers(NULL_MIN_SHIFT, len(idx) - NULL_MIN_SHIFT)))
        nulls.append(beats(cv(D, des.gain(zs, extra=K), rows), f['ADDK']))
    return {'call': beats(f['GAIN'], f['ADD']), 'call_k': beats(f['GAINK'], f['ADDK']),
            'd': f['GAIN']['r2'] - f['ADD']['r2'], 'd_k': f['GAINK']['r2'] - f['ADDK']['r2'],
            'kernel_r2': f['ADDK']['r2'] - f['ADD']['r2'], 'kernel_call': beats(f['ADDK'], f['ADD']),
            'null_call_k': np.stack(nulls) if nulls else np.zeros((0, D['Y'].shape[1]), bool)}


def kernel_null(D, ev, rng, n_shuf=5, min_offset=60.0):
    """ How often gaze-shift kernels 'improve' a cell when the shift times are wrong: every onset moved by one
    random offset (>= 60 s, wrapped around the recording), so their timing statistics are kept. """
    rows = light_rows(D)
    des = Design(D, rows)
    add = cv(D, des.block['ADD'], rows)
    t0, t1 = D['twopT'][0], D['twopT'][-1]
    out = []
    for _ in range(n_shuf):
        off = rng.uniform(min_offset, (t1 - t0) - min_offset)
        evs = dict(ev)
        evs['t_on'] = t0 + (ev['t_on'] - t0 + off) % (t1 - t0)
        K = kernel_features(D, evs)
        out.append(beats(cv(D, np.c_[des.block['ADD'], K], rows), add))
    return np.stack(out)


def _dvm(ang):
    """ d vm(rho) / d rho, per degree. """
    c = np.radians(np.arange(na.N_VM) * 360.0 / na.N_VM)
    a = np.radians(np.asarray(ang, float))[:, None] - c[None, :]
    return -na.VM_KAPPA * np.sin(a) * np.exp(na.VM_KAPPA * (np.cos(a) - 1.0)) * np.pi / 180.0


def _wrapd(x):
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


def between_pairs(D, E, rows, rng, n_draw=4000000):
    """ Pairs of frames from DIFFERENT fixations (a gaze shift between them), same matching as within_pairs. """
    fid, e, rho = E['fix'], D['e'], D['rho']
    idx = np.flatnonzero(rows)
    if len(idx) < 2:
        return np.zeros(0, int), np.zeros(0, int)
    a, b = rng.choice(idx, n_draw), rng.choice(idx, n_draw)
    ok = (fid[a] != fid[b]) & (np.abs(_wrapd(rho[a] - rho[b])) < PAIR_DRHO) & (np.abs(e[a] - e[b]) >= PAIR_DE)
    a, b = a[ok], b[ok]
    if len(a) > PAIR_MAX:
        s_ = rng.choice(len(a), PAIR_MAX, replace=False)
        a, b = a[s_], b[s_]
    return a, b


def within_pairs(D, E, rows, rng, max_per_fix=3000):
    """ Pairs of frames in the same fixation, retinal position within PAIR_DRHO, eye at least PAIR_DE apart. """
    fid, e, rho = E['fix'], D['e'], D['rho']
    I, J = [], []
    idx = np.flatnonzero(rows)
    for f in np.unique(fid[idx]):
        k = idx[fid[idx] == f]
        if len(k) < 2:
            continue
        a, b = np.triu_indices(len(k), 1)
        a, b = k[a], k[b]
        ok = (np.abs(_wrapd(rho[a] - rho[b])) < PAIR_DRHO) & (np.abs(e[a] - e[b]) >= PAIR_DE)
        a, b = a[ok], b[ok]
        if len(a) > max_per_fix:
            s = rng.choice(len(a), max_per_fix, replace=False)
            a, b = a[s], b[s]
        I.append(a)
        J.append(b)
    if not I:
        return np.zeros(0, int), np.zeros(0, int)
    I, J = np.concatenate(I), np.concatenate(J)
    if len(I) > PAIR_MAX:
        s = rng.choice(len(I), PAIR_MAX, replace=False)
        I, J = I[s], J[s]
    return I, J


PROFILE_EDGES = np.arange(-180.0, 181.0, 30.0)


def pair_regression(dY, f_bar, fp_bar, de, drho, dnuis):
    """ Per cell: dr = a f(rho) de + b f'(rho) de + c de + d f'(rho) drho + dnuis w. Returns a, b, c. """
    C = dY.shape[1]
    out = np.full((C, 3), np.nan)
    for c in range(C):
        X = np.c_[f_bar[:, c] * de, fp_bar[:, c] * de, de, fp_bar[:, c] * drho, dnuis]
        if not np.all(np.isfinite(X)):
            continue
        coef = np.linalg.lstsq(X, dY[:, c], rcond=None)[0]
        out[c] = coef[:3]
    return out


def _pair_tests(D, des, G, H, Rm, Xg, n0, beta_r, beta_g, I, J):
    out = {'n_pairs': np.array(len(I))}
    if len(I) < 200:
        return out
    rho, e = D['rho'], D['e']
    rbar = np.degrees(np.angle(np.exp(1j * np.radians(rho[I])) + np.exp(1j * np.radians(rho[J]))))
    de, drho = e[J] - e[I], _wrapd(rho[J] - rho[I])
    f_bar = na._vm_basis(rbar) @ beta_r
    fp_bar = _dvm(rbar) @ beta_r
    dnuis = des.nuis[J] - des.nuis[I]
    res = {}
    for name, Yv in (('real', D['Y']), ('ret_model', Rm['pred']), ('gain_model', G['pred']), ('head_model', H['pred'])):
        res[name] = pair_regression(Yv[J] - Yv[I], f_bar, fp_bar, de, drho, dnuis)
    # population profile: slope of dr on de by retinal position relative to each cell's preferred position
    grid = na._vm_basis(RHO_GRID) @ beta_r
    pref = RHO_GRID[np.argmax(grid, 0)]
    amp = grid.max(0) - grid.min(0)
    gsign = np.sign(np.sum((na._vm_basis(RHO_GRID) @ beta_g) * (grid - grid.mean(0)), 0))
    nb = len(PROFILE_EDGES) - 1
    prof = {k: np.full((D['Y'].shape[1], nb), np.nan) for k in res}
    rel = _wrapd(rbar[:, None] - pref[None, :])
    bins = np.clip(np.digitize(rel, PROFILE_EDGES) - 1, 0, nb - 1)
    for name, Yv in (('real', D['Y']), ('ret_model', Rm['pred']), ('gain_model', G['pred']), ('head_model', H['pred'])):
        dY = Yv[J] - Yv[I]
        for k in range(nb):
            m = bins == k
            num = np.sum(np.where(m, dY * de[:, None], 0.0), 0)
            den = np.sum(np.where(m, de[:, None] ** 2, 0.0), 0)
            prof[name][:, k] = num / np.where(den > 0, den, np.nan) / np.maximum(amp, 1e-12)
    # gain expression: regress each cell's real change on the changes its own GAIN fit predicts, term by term
    # (gain term, retinal term, additive eye terms, nuisance). 1 on the gain term = the fitted gain holds
    # within fixations at matched retinal position; 0 = it is absent there.
    Z = (np.nan_to_num(Xg) - G['x_mean']) / G['x_sd']
    nn = des.nuis.shape[1]
    blocks = [np.arange(Xg.shape[1] - na.N_VM, Xg.shape[1]), np.arange(n0, n0 + na.N_VM),
              np.r_[nn, np.arange(n0 + na.N_VM, n0 + na.N_VM + 4)], np.arange(nn)]
    dT = [(Z[J][:, b] - Z[I][:, b]) @ G['coef'][b] for b in blocks]          # 4 x (pairs, C)
    for name, Yv in (('real', D['Y']), ('ret_model', Rm['pred']), ('gain_model', G['pred']), ('head_model', H['pred'])):
        dY = Yv[J] - Yv[I]
        coef, se = np.full((dY.shape[1], 4), np.nan), np.full(dY.shape[1], np.nan)
        for c in range(dY.shape[1]):
            X = np.stack([t[:, c] for t in dT], 1)
            if not np.all(np.isfinite(X)) or np.std(X[:, 0]) == 0:
                continue
            b_, *_ = np.linalg.lstsq(X, dY[:, c], rcond=None)
            r = dY[:, c] - X @ b_
            cov = np.linalg.pinv(X.T @ X) * np.var(r)
            coef[c], se[c] = b_, np.sqrt(max(cov[0, 0], 0))
        out['expr_' + name] = coef
        out['expr_se_' + name] = se
    out['gain_term_sd'] = np.std(dT[0], 0)
    active = np.sum((D['Y'][J] - D['Y'][I]) != 0, 0)
    quiet = active < MIN_ACTIVE_PAIRS
    for k in list(out):
        if k.startswith(('expr_', 'expr_se_')):
            out[k][quiet] = np.nan
    for k in res:
        res[k][quiet] = np.nan
        prof[k][quiet] = np.nan
    out['n_active'] = active
    out.update({'abc_' + k: v for k, v in res.items()})
    out.update({'prof_' + k: v for k, v in prof.items()})

    out.update({'gain_sign': gsign, 'pref': pref, 'field_amp': amp,
                'sample_de': de[:5000], 'sample_drho': drho[:5000],
                'all_de_sd': np.array(float(np.std(de))), 'all_drho_sd': np.array(float(np.std(drho)))})
    return out


def test_within(D, E, rng, late_key='late'):
    """ Late frames of the same fixation at matched retinal position: does eye position scale the field
    (gain), shift it (head frame), or do nothing (pure retinal)? Fields come from frames outside the late set.

    b is in natural units: a head-centred cell (response = h(rho + e)) has b = 1, an eye-centred cell b = 0.
    a is the fractional change of the retinal field per degree of eye position.
    The same regression on the GAIN and HEAD models' own predictions gives each cell's two references.
    """
    M = epoch_masks(D, E)
    des = Design(D, M['all'])
    late = M[late_key]
    train = M['all'] & ~late
    Xg = des.gain(des.mz)
    gfit = cv(D, Xg, train)
    G = full_fit(D, Xg, train, gfit['lam'])
    hfit = cv(D, des.block['HEAD'], train)
    H = full_fit(D, des.block['HEAD'], train, hfit['lam'])
    rfit = cv(D, des.block['RET'], train)
    Rm = full_fit(D, des.block['RET'], train, rfit['lam'])
    n0 = des.nuis.shape[1] + 1
    beta_r = (G['coef'] / G['x_sd'][:, None])[n0:n0 + na.N_VM]          # retinal field, rate per unit vm
    beta_g = (G['coef'] / G['x_sd'][:, None])[-na.N_VM:]
    out = {}
    for kind, (I, J) in (('same', within_pairs(D, E, late, rng)), ('across', between_pairs(D, E, late, rng))):
        out[kind] = _pair_tests(D, des, G, H, Rm, Xg, n0, beta_r, beta_g, I, J)
    return out


# ---------------------------------------------------------------------------
# gaze-shift transients (Figure 12)
# ---------------------------------------------------------------------------

PETH_HALF, PETH_BIN = 1.5, 0.025
ISO_PRE, ISO_POST = 0.8, 1.2           # PETH events: no other shift 0.8 s before or 1.2 s after


def transients(D, ev, cells, rng):
    """ Per-cell gaze-shift PETH (25 ms bins from scan-row sample times) on isolated light shifts, and
    responsive cells picked on one half of the events, measured on the other. """
    from fm2p.nonvisual_probe import peth_samples, peth_from_samples
    o = np.argsort(ev['t_on'])
    on = ev['t_on'][o]
    gap_pre = on - np.r_[-np.inf, on[:-1]]
    gap_post = np.r_[on[1:], np.inf] - on
    # isolated = no detected shift 0.8 s before / 1.2 s after, AND tracking throughout both windows (a shift
    # hidden in a gap would otherwise count as isolation)
    cov_pre, clean = ev['cover_pre'][o], ev['clean_until'][o]
    iso = (gap_pre > ISO_PRE) & (gap_post > ISO_POST) & ev['light'][o] & cov_pre & (clean > ISO_POST)
    T = on[iso]
    rate = D['Y'].T.copy()
    rate[:, ~D['use']] = np.nan
    samp = D['twopT'][None, :] + cells['offset'][:, None]
    S, tau = peth_samples(rate, samp, T, half=PETH_HALF, binw=PETH_BIN)
    nb = len(tau)
    par = np.arange(len(T)) % 2
    P = [peth_from_samples(S, nb, (par == k).astype(float)) for k in (0, 1)]
    base = (tau >= -0.5) & (tau < -0.1)
    resp = (tau > 0.0) & (tau <= 0.4)
    sel = []
    for k in (0, 1):
        b, r = np.nanmean(P[k][:, base], 1), np.nanmean(P[k][:, resp], 1)
        sd = np.nanstd(P[k][:, base], 1)
        sel.append((r - b) > 3 * np.maximum(sd, 1e-12) / np.sqrt(resp.sum()))
    # per-cell test on every light shift isolated by 0.8 s before (response 0-0.4 s minus -0.5 to -0.1 s),
    # t-test over events (spike counts are mostly zero, which a rank test with ties handles badly),
    # Benjamini-Hochberg over the session's cells
    from fm2p.nonvisual_probe import window_mean
    from scipy.stats import ttest_1samp
    Tt = on[(gap_pre > ISO_PRE) & ev['light'][o] & cov_pre & (clean > 0.4)]
    r1, _ = window_mean(rate, samp, D['dt'], Tt, 0.0, 0.4)
    r0, _ = window_mean(rate, samp, D['dt'], Tt, -0.5, -0.1)
    d = r1 - r0
    p = np.ones(len(d))
    for c in range(len(d)):
        x = d[c][np.isfinite(d[c])]
        if len(x) >= 20 and np.any(x != 0):
            p[c] = ttest_1samp(x, 0.0).pvalue
    q = np.minimum.accumulate((p * len(p) / (np.argsort(np.argsort(p)) + 1))[np.argsort(p)][::-1])[::-1]
    qq = np.empty_like(q)
    qq[np.argsort(p)] = np.minimum(q, 1)
    return {'tau': tau, 'peth_a': P[0], 'peth_b': P[1], 'enh_a': sel[0], 'enh_b': sel[1], 'n_events': np.array(len(T)),
            'intervals': np.diff(on[ev['light'][o]]), 'resp_p': p, 'resp_q': qq, 'resp_d': np.nanmean(d, 1),
            'n_test_events': np.array(len(Tt))}


def test_active(D, E, rng, n_shuf=N_SHUF):
    """ Early vs late fixation frames within active periods only (fixations < ACTIVE_FIX_S), with cross-epoch
    prediction and the matched-retina pair tests on the active late frames. """
    M = epoch_masks(D, E)
    des = Design(D, M['all'])
    out = {}
    for k, b in (('early_act', 'late_act'), ('late_act', 'early_act')):
        cross, out[k] = gain_call_set(D, des, M[k], n_shuf, rng, eval_rows={b: M[b]})
        out['cross_{}_to_{}'.format(k, b)] = cross[b]
    out['within'] = test_within(D, E, rng, late_key='late_act')
    for k in ('early_act', 'late_act', 'early', 'late'):
        out['state_' + k] = {'speed': np.array(float(np.nanmedian(D['speed'][M[k]]))),
                             'head_speed': np.array(float(np.nanmedian(np.abs(D['h60'][M[k]])))),
                             'n': np.array(int(M[k].sum()))}
    return out


BEH_CLASSES = ('shift', 'post', 'comp', 'still', 'all')
BEH_CROSS = (('comp', 'post'), ('post', 'comp'), ('comp', 'shift'), ('shift', 'comp'))


def test_behaviour(D, E, ev, rng, n_shuf=N_SHUF):
    """ Gain fits per behaviour class with shuffled-eye nulls; cross-class prediction; previous vs current eye
    position in post-shift and compensatory frames; matched-retina pairs among compensatory frames. """
    M = epoch_masks(D, E)
    des = Design(D, M['all'])
    out = {'n_frames': {k: np.array(int(M[k].sum())) for k in BEH_CLASSES}}
    for k in BEH_CLASSES:
        targets = {b: M[b] for a, b in BEH_CROSS if a == k}
        cross, out[k] = gain_call_set(D, des, M[k], n_shuf, rng, eval_rows=targets or None)
        for b, v in cross.items():
            out['cross_{}_to_{}'.format(k, b)] = v
    e_prev, _ = shift_eye(D, E, ev)
    out['prev'] = {k: prev_on_rows(D, des, e_prev, M[k], rng) for k in ('post', 'comp')}
    out['pairs'] = test_within(D, E, rng, late_key='comp')
    out['state'] = {k: {'speed': np.array(float(np.nanmedian(D['speed'][M[k]]))),
                        'head_speed': np.array(float(np.nanmedian(np.abs(D['h60'][M[k]])))),
                        'eye_sd': np.array(float(np.nanstd(D['e'][M[k]])))} for k in BEH_CLASSES}
    return out


HELDOUT_CLASSES = ('shift', 'post', 'comp', 'still')


def test_heldout(D, E, rng, n_shuf=N_SHUF):
    """ For each behaviour class: fit ADD and GAIN on every OTHER light frame, predict the class's frames.
    gain = R2(GAIN) - R2(ADD) on the held-out class; null = the same with eye position shuffled inside the gain
    term of the training fit. A gain field that is a gaze-shift artefact should not predict compensatory frames. """
    M = epoch_masks(D, E)
    des = Design(D, M['all'])
    out = {}
    for k in HELDOUT_CLASSES:
        train, test = M['all'] & ~M[k], M[k]
        if test.sum() < 50:
            continue
        Xg = des.gain(des.mz)
        fa, fg = cv(D, des.block['ADD'], train), cv(D, Xg, train)
        A = full_fit(D, des.block['ADD'], train, fa['lam'])
        G = full_fit(D, Xg, train, fg['lam'])
        base = r2_on(D['Y'], A['pred'], test)
        idx = np.flatnonzero(train)
        nulls = []
        for _ in range(n_shuf):
            zs = des.mz.copy()
            zs[idx] = np.roll(des.mz[idx], int(rng.integers(NULL_MIN_SHIFT, len(idx) - NULL_MIN_SHIFT)))
            fn = cv(D, des.gain(zs), train)
            nulls.append(r2_on(D['Y'], full_fit(D, des.gain(zs), train, fn['lam'])['pred'], test) - base)
        out[k] = {'gain': r2_on(D['Y'], G['pred'], test) - base, 'null': np.stack(nulls),
                  'n_test': np.array(int(test.sum()))}
    return out


RAW_RHO_EDGES = np.arange(-180.0, 181.0, 45.0)
RAW_Q = 5


def test_raw(D, E, cells, rng):
    """ Data close to the raw spikes for Figure 13: models fitted on every frame outside a class, evaluated on it.

    calib/<class>: per cell, the class's frames in RAW_Q quantile bins of the predicted gain effect
        (GAIN prediction - ADD prediction); per bin the mean predicted effect and the mean measured effect
        (activity - ADD prediction), both in units of the cell's residual SD on the training frames.
    tuning: per cell, compensatory frames in 45 deg retinal bins x eye half (split at the median eye position of
        the compensatory frames): measured mean, SEM and count, and the GAIN and ADD predictions on the same frames.
    """
    M = epoch_masks(D, E)
    des = Design(D, M['all'])
    C = D['Y'].shape[1]
    out = {'calib': {}}
    for k in ('comp', 'shift', 'still'):
        train, test = M['all'] & ~M[k], M[k]
        if test.sum() < 50:
            continue
        Xg = des.gain(des.mz)
        A = full_fit(D, des.block['ADD'], train, cv(D, des.block['ADD'], train)['lam'])
        G = full_fit(D, Xg, train, cv(D, Xg, train)['lam'])
        sd = np.maximum(np.std(D['Y'][train] - G['pred'][train], 0), 1e-12)
        g = (G['pred'][test] - A['pred'][test]) / sd
        y = (D['Y'][test] - A['pred'][test]) / sd
        px, py, n = np.full((C, RAW_Q), np.nan), np.full((C, RAW_Q), np.nan), np.zeros((C, RAW_Q))
        for c in range(C):
            q = np.quantile(g[:, c], np.linspace(0, 1, RAW_Q + 1))
            b = np.clip(np.searchsorted(q, g[:, c], side='right') - 1, 0, RAW_Q - 1)
            for j in range(RAW_Q):
                m = b == j
                if m.sum():
                    px[c, j], py[c, j], n[c, j] = g[m, c].mean(), y[m, c].mean(), m.sum()
        quiet = np.sum(D['Y'][train] > 0, 0) < MIN_ACTIVE_TRAIN
        px[quiet], py[quiet], n[quiet] = np.nan, np.nan, 0
        out['calib'][k] = {'pred': px, 'meas': py, 'n': n}
        if k == 'comp':
            e = D['e'][test]
            half = (e > np.median(e)).astype(int)
            rb = np.clip(np.digitize(D['rho'][test], RAW_RHO_EDGES) - 1, 0, len(RAW_RHO_EDGES) - 2)
            nb = len(RAW_RHO_EDGES) - 1
            T = {key: np.full((C, 2, nb), np.nan) for key in ('meas', 'sem', 'gain', 'add')}
            T['n'] = np.zeros((2, nb))
            for h in (0, 1):
                for j in range(nb):
                    m = (half == h) & (rb == j)
                    T['n'][h, j] = m.sum()
                    if m.sum() < 3:
                        continue
                    Yv = D['Y'][test][m]
                    T['meas'][:, h, j] = Yv.mean(0)
                    T['sem'][:, h, j] = Yv.std(0) / np.sqrt(m.sum())
                    T['gain'][:, h, j] = G['pred'][test][m].mean(0)
                    T['add'][:, h, j] = A['pred'][test][m].mean(0)
            T['eye_split'] = np.array(float(np.median(e)))
            T['eye_means'] = np.array([float(e[half == h].mean()) for h in (0, 1)])
            T['rate'] = D['Y'][test].mean(0)
            out['tuning'] = T
    return out


def update_raw():
    for neural, nonvis, sess, animal in sessions():
        pth = cache_path(sess)
        if not os.path.isfile(pth):
            continue
        D, ev, cells, lag = load(neural, nonvis)
        E = frame_epochs(D, ev)
        r = test_raw(D, E, cells, np.random.default_rng(5))
        with h5py.File(pth, 'a') as f:
            if 'raw' in f:
                del f['raw']
            _write(f.create_group('raw'), r)
        print('  raw-data summaries added:', sess, flush=True)


def example_curves(neural, nonvis, cells_idx, n_bins=4, half_width=45.0):
    """ Example cells, compensatory frames only: activity vs eye position with the pillar inside the cell's
    field (within half_width of its preferred retinal position) and outside it (> 90 deg away), with the
    predictions of GAIN and ADD fitted on every other light frame, on the same frames. """
    D, ev, cells, lag = load(neural, nonvis)
    E = frame_epochs(D, ev)
    M = epoch_masks(D, E)
    des = Design(D, M['all'])
    train, test = M['all'] & ~M['comp'], M['comp']
    Xg = des.gain(des.mz)
    A = full_fit(D, des.block['ADD'], train, cv(D, des.block['ADD'], train)['lam'])
    G = full_fit(D, Xg, train, cv(D, Xg, train)['lam'])
    f, g = profiles(G, des.nuis.shape[1] + 1)
    e, rho = D['e'][test], D['rho'][test]
    edges = np.quantile(e, np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.searchsorted(edges, e, side='right') - 1, 0, n_bins - 1)
    out = {'eye_edges': edges, 'eye_centres': np.array([e[b == k].mean() for k in range(n_bins)])}
    for c in cells_idx:
        pref = RHO_GRID[np.argmax(f[c])]
        d = np.abs(_wrapd(rho - pref))
        r = {'pref': np.array(pref)}
        for name, m0 in (('in', d < half_width), ('out', d > 90)):
            stats = {k: np.full(n_bins, np.nan) for k in ('meas', 'sem', 'gain', 'add', 'n', 'nz')}
            for k in range(n_bins):
                m = m0 & (b == k)
                if m.sum() < 3:
                    continue
                y = D['Y'][test][m, c]
                stats['meas'][k], stats['sem'][k] = y.mean(), y.std() / np.sqrt(m.sum())
                stats['gain'][k], stats['add'][k] = G['pred'][test][m, c].mean(), A['pred'][test][m, c].mean()
                stats['n'][k], stats['nz'][k] = m.sum(), np.sum(y > 0)
            r[name] = stats
        out[str(c)] = r
    return out


POP_EYE_EDGES = np.array([-30.0, -12.0, -4.0, 4.0, 12.0, 30.0])   # deg from the session's median eye position


def test_popgain(D, E, rng):
    """ Population eye-position tuning at the preferred retinal position, per behaviour class.

    For each class, GAIN and ADD are fitted on every OTHER light frame. Each cell's preferred retinal position
    and the sign of its gain there come from that fit (so never from the frames being tested). On the class's
    frames: activity relative to the cell's mean rate on the training frames, binned by eye position (deg from
    the session median; mirrored for cells whose gain falls with eye position), with the pillar inside the
    field (< 45 deg from preferred) or outside it (> 90 deg). Sums and counts per cell and bin.
    """
    M = epoch_masks(D, E)
    des = Design(D, M['all'])
    e0 = np.nanmedian(D['e'][M['all']])
    nb = len(POP_EYE_EDGES) - 1
    out = {}
    for k in ('comp', 'shift', 'still'):
        train, test = M['all'] & ~M[k], M[k]
        if test.sum() < 50:
            continue
        Xg = des.gain(des.mz)
        A = full_fit(D, des.block['ADD'], train, cv(D, des.block['ADD'], train)['lam'])
        G = full_fit(D, Xg, train, cv(D, Xg, train)['lam'])
        f, g = profiles(G, des.nuis.shape[1] + 1)
        ip = np.argmax(f, 1)
        pref = RHO_GRID[ip]
        sign = np.sign(g[np.arange(len(ip)), ip])
        rate = np.maximum(D['Y'][train].mean(0), 1e-12)
        e = D['e'][test] - e0
        rho = D['rho'][test]
        C = D['Y'].shape[1]
        r = {kk: np.zeros((C, 2, nb)) for kk in ('meas', 'gain', 'add', 'n')}
        for c in range(C):
            ec = e * (sign[c] if sign[c] != 0 else 1.0)
            b = np.digitize(ec, POP_EYE_EDGES) - 1
            d = np.abs(_wrapd(rho - pref[c]))
            for j, m0 in enumerate((d < 45, d > 90)):
                for q in range(nb):
                    m = m0 & (b == q)
                    if not m.any():
                        continue
                    r['meas'][c, j, q] = D['Y'][test][m, c].sum() / rate[c]
                    r['gain'][c, j, q] = G['pred'][test][m, c].sum() / rate[c]
                    r['add'][c, j, q] = A['pred'][test][m, c].sum() / rate[c]
                    r['n'][c, j, q] = m.sum()
        quiet = np.sum(D['Y'][train] > 0, 0) < MIN_ACTIVE_TRAIN
        for kk in ('meas', 'gain', 'add', 'n'):
            r[kk][quiet] = 0
        r['sign'] = sign
        out[k] = r
    return out


def example_all_frames(neural, nonvis, c, n_rho=12):
    """ One cell over all light frames: retinal tuning at the lower and upper half of eye positions, measured,
    and as predicted by its GAIN and ADD fits (cross-validated predictions, 5 block folds). """
    D, ev, cells, lag = load(neural, nonvis)
    E = frame_epochs(D, ev)
    M = epoch_masks(D, E)
    des = Design(D, M['all'])
    rows = M['all']
    from fm2p.ffNLE_scene import cv_ridge
    preds = {}
    for name, X in (('add', des.block['ADD']), ('gain', des.gain(des.mz))):
        trm, tem, block = na._folds(D, rows)
        r = cv_ridge(na._zbuilder(np.nan_to_num(X)), D['Y'][:, [c]], trm, tem, block, return_oof=True)
        preds[name] = r['oof'][:, 0]
    e, rho = D['e'][rows], D['rho'][rows]
    half = (e > np.median(e)).astype(int)
    edges = np.linspace(-180, 180, n_rho + 1)
    b = np.clip(np.digitize(rho, edges) - 1, 0, n_rho - 1)
    out = {'edges': edges, 'eye_means': np.array([e[half == h].mean() for h in (0, 1)])}
    for kk in ('meas', 'sem', 'gain', 'add', 'n'):
        out[kk] = np.full((2, n_rho), np.nan)
    y = D['Y'][rows, c]
    for h in (0, 1):
        for j in range(n_rho):
            m = (half == h) & (b == j)
            if m.sum() < 10:
                continue
            out['meas'][h, j], out['sem'][h, j] = y[m].mean(), y[m].std() / np.sqrt(m.sum())
            out['gain'][h, j], out['add'][h, j] = preds['gain'][rows][m].mean(), preds['add'][rows][m].mean()
            out['n'][h, j] = m.sum()
    return out


ARR_BLOCK_S = 60.0             # arrival test: cells defined on alternate 60 s blocks, measured on the others
ARR_RHO_EDGES = np.arange(-180.0, 181.0, 30.0)
ARR_MIN_FRAMES = 5             # frames needed in each eye side x arrival group
ARR_CLASSES = ('settled', 'still', 'comp')
ARR_SINCE_EDGES = (0.0, 0.3, 1.0, 3.0, 10.0, np.inf)   # time since the last shift ended (start of spike window), s
ARR_ECC_EDGES = (0.0, 2.0, 4.0, 6.0, 9.0, 13.0, 20.0, np.inf)   # distance from the median eye position, deg; the
                               # matched eye-side difference compares the two sides within each of these bins
ARR_ECC_NEAR, ARR_ECC_FAR = 4.0, 9.0   # off-centre effect: rate >= FAR minus rate < NEAR deg from the median


def arrival_since_sets():
    """ Frame sets for the arrival test by time since the last gaze shift: since_<i> (all), since_<i>_move and
    since_<i>_still (still = |head| < STILL_HEAD and speed < STILL_SPEED), each >= CLASS_PRE_S before the next
    shift, outside shifts, tracking covered. """
    def mk(a, b, state):
        def f(D, E, M):
            m = (M['all'] & ~E['inside'] & D['covered_b'] & (E['until'] >= CLASS_PRE_S)
                 & (E['since'] >= a) & (E['since'] < b))
            still = (np.abs(D['h60']) < STILL_HEAD) & (D['speed'] < STILL_SPEED)
            return m & still if state == 'still' else m & ~still if state == 'move' else m
        return f
    out = {}
    for i, (a, b) in enumerate(zip(ARR_SINCE_EDGES[:-1], ARR_SINCE_EDGES[1:])):
        for state, suf in ((None, ''), ('move', '_move'), ('still', '_still')):
            out['since_{}{}'.format(i, suf)] = mk(a, b, state)
    return out


def shift_direction_in_e(ev):
    """ Sign of every gaze shift in the units of D['e'] (+1 = the eye moved toward larger e). eye_amp is the change
    in the 60 Hz theta during the shift, and e is the same theta binned to 2P frames (r > 0.99 in every session
    checked, 2026-11-05), so the signs agree. Across-shift changes in e itself are no use for this: at 7.5 Hz the
    saccade shares frames with the slow phase before it, which it often undoes. """
    return np.sign(ev['eye_amp'])


def test_arrival(D, E, ev, cells, rng=None, extra=None):
    """ Where the eye is versus how it got there (model-free; Figure 13G).

    Per cell and per fold (alternate ARR_BLOCK_S blocks): on the training blocks' light frames, the retinal field
    = the 90 deg (3 x 30 deg) window with the highest rate, outside = the 180 deg opposite it, and the high-gain
    eye side = the side of the session's median eye position with the higher in-field rate. On the other blocks,
    frames of each class (settled = >= CLASS_POST_S after and >= CLASS_PRE_S before any shift, tracking covered;
    still; compensatory) are split by eye side (toward / away from the high-gain side) x direction of the last gaze
    shift (toward / away). Rates are divided by the cell's mean rate on the test blocks' light frames and the two
    folds averaged.
    rate_in / rate_out: (cells, 2, 2) = [eye away, eye toward] x [shift away, shift toward]; n_*: frames.
    diff_*_u: eye toward - away (both shift directions pooled); diff_*_m: the same, matched for distance from the
    median eye position (difference within each ARR_ECC_EDGES bin, weighted by the smaller side's frame count),
    so an effect of the eye being off-centre on either side cannot pass as a one-sided gain; ecc_in: in-field rate
    >= ARR_ECC_FAR minus < ARR_ECC_NEAR deg from the median, both sides pooled. All x the cell's mean rate. """
    M = epoch_masks(D, E)
    L = M['all']
    clear = (E['since'] >= CLASS_POST_S) & (E['until'] >= CLASS_PRE_S)
    cls = {'settled': L & ~E['inside'] & D['covered_b'] & clear, 'still': M['still'], 'comp': M['comp']}
    if extra is not None:                          # {name: function(D, E, M) -> frame mask}, extra frame sets
        cls.update({k: f(D, E, M) for k, f in extra.items()})
    d_ev = shift_direction_in_e(ev)
    d_sorted = d_ev[E['order']]
    kp = E['fix']
    last = np.where(kp >= 0, d_sorted[np.clip(kp, 0, None)], 0)
    side = np.sign(D['e'] - np.nanmedian(D['e'][L]))
    ecc = np.abs(D['e'] - np.nanmedian(D['e'][L]))
    eb = np.clip(np.digitize(ecc, ARR_ECC_EDGES) - 1, 0, len(ARR_ECC_EDGES) - 2)
    ne = len(ARR_ECC_EDGES) - 1
    rb = np.clip(np.digitize(D['rho'], ARR_RHO_EDGES) - 1, 0, len(ARR_RHO_EDGES) - 2)
    nb = len(ARR_RHO_EDGES) - 1
    blk = (np.floor((D['t'] - D['t'][0]) / ARR_BLOCK_S).astype(int) % 2).astype(bool)
    Y = D['Y']
    C = Y.shape[1]
    acc = {k: {'rate_in': np.zeros((2, C, 2, 2)) * np.nan, 'rate_out': np.zeros((2, C, 2, 2)) * np.nan,
               'n_in': np.zeros((2, C, 2, 2)), 'n_out': np.zeros((2, C, 2, 2)),
               **{d: np.full((2, C), np.nan) for d in ('diff_in_u', 'diff_in_m', 'diff_out_u', 'diff_out_m',
                                                       'ecc_in')}} for k in cls}
    hi_side = np.zeros((2, C))
    for f, tr_blk in enumerate((False, True)):
        tr, te = L & (blk == tr_blk), L & (blk != tr_blk)
        occ = np.bincount(rb[tr], minlength=nb).astype(float)
        rate = np.stack([Y[tr & (rb == j)].mean(0) if occ[j] > 0 else np.zeros(C) for j in range(nb)], 1)  # (C, nb)
        win = rate + np.roll(rate, -1, 1) + np.roll(rate, 1, 1)
        best = np.argmax(win, 1)                                   # centre bin of the 90 deg field
        dist = np.abs(((np.arange(nb)[None, :] - best[:, None]) + nb // 2) % nb - nb // 2)
        infield, outfield = dist <= 1, dist >= 3                   # (C, nb)
        IN = infield[:, rb].T                                      # (frames, C)
        OUT = outfield[:, rb].T
        hi_rate = np.array([Y[tr & (side > 0) & IN[:, c], c].mean() if np.any(tr & (side > 0) & IN[:, c]) else np.nan
                            for c in range(C)])
        lo_rate = np.array([Y[tr & (side < 0) & IN[:, c], c].mean() if np.any(tr & (side < 0) & IN[:, c]) else np.nan
                            for c in range(C)])
        hs = np.where(hi_rate >= lo_rate, 1.0, -1.0)
        hs[~(np.isfinite(hi_rate) & np.isfinite(lo_rate))] = np.nan
        hi_side[f] = hs
        norm = Y[te].mean(0)
        quiet = np.sum(Y[tr] > 0, 0) < MIN_ACTIVE_TRAIN
        for k, m in cls.items():
            base = te & m & (last != 0)
            for c in range(C):
                if quiet[c] or not np.isfinite(hs[c]) or norm[c] <= 0:
                    continue
                eye_t = side * hs[c] > 0                           # eye toward the high-gain side
                sh_t = last * hs[c] > 0                            # last shift moved toward it
                for key, FLD in (('in', IN[:, c]), ('out', OUT[:, c])):
                    for a, ev_ in enumerate((~eye_t, eye_t)):
                        for b, sv in enumerate((~sh_t, sh_t)):
                            mm = base & FLD & ev_ & sv
                            nm = int(mm.sum())
                            acc[k]['n_' + key][f, c, a, b] = nm
                            if nm >= ARR_MIN_FRAMES:
                                acc[k]['rate_' + key][f, c, a, b] = Y[mm, c].mean() / norm[c]
                    # eye-side difference, unmatched and matched for distance from the median eye position
                    mt, ma = base & FLD & eye_t, base & FLD & ~eye_t
                    if mt.sum() >= ARR_MIN_FRAMES and ma.sum() >= ARR_MIN_FRAMES:
                        acc[k]['diff_{}_u'.format(key)][f, c] = (Y[mt, c].mean() - Y[ma, c].mean()) / norm[c]
                    nt, na_ = np.bincount(eb[mt], minlength=ne), np.bincount(eb[ma], minlength=ne)
                    st = np.bincount(eb[mt], weights=Y[mt, c], minlength=ne)
                    sa = np.bincount(eb[ma], weights=Y[ma, c], minlength=ne)
                    w = np.where((nt >= 2) & (na_ >= 2), np.minimum(nt, na_), 0)
                    if w.sum() >= 2 * ARR_MIN_FRAMES:
                        with np.errstate(invalid='ignore', divide='ignore'):
                            dd = np.where(w > 0, st / np.maximum(nt, 1) - sa / np.maximum(na_, 1), 0.0)
                        acc[k]['diff_{}_m'.format(key)][f, c] = np.sum(w * dd) / w.sum() / norm[c]
                    if key == 'in':
                        far, near = base & FLD & (ecc >= ARR_ECC_FAR), base & FLD & (ecc < ARR_ECC_NEAR)
                        if far.sum() >= ARR_MIN_FRAMES and near.sum() >= ARR_MIN_FRAMES:
                            acc[k]['ecc_in'][f, c] = (Y[far, c].mean() - Y[near, c].mean()) / norm[c]
    out = {'meta': {'side_split': np.array(float(np.nanmedian(D['e'][L]))),
                    'side_agree_folds': np.array(float(np.nanmean(hi_side[0] == hi_side[1])))}}
    for k in cls:
        with np.errstate(invalid='ignore'):
            out[k] = {'rate_in': np.nanmean(acc[k]['rate_in'], 0), 'rate_out': np.nanmean(acc[k]['rate_out'], 0),
                      'n_in': acc[k]['n_in'].sum(0), 'n_out': acc[k]['n_out'].sum(0),
                      'n_frames': np.array(int((cls[k] & (last != 0)).sum())),
                      **{d: np.nanmean(acc[k][d], 0) for d in ('diff_in_u', 'diff_in_m', 'diff_out_u',
                                                                'diff_out_m', 'ecc_in')}}
    return out


def update_arrival():
    for neural, nonvis, sess, animal in sessions():
        pth = cache_path(sess)
        if not os.path.isfile(pth):
            continue
        D, ev, cells, lag = load(neural, nonvis)
        E = frame_epochs(D, ev)
        a = test_arrival(D, E, ev, cells, extra=arrival_since_sets())
        a['meta']['since_edges'] = np.array(ARR_SINCE_EDGES)
        with h5py.File(pth, 'a') as f:
            if 'arrival' in f:
                del f['arrival']
            _write(f.create_group('arrival'), a)
        print('  arrival test added: {} (high-gain side fold agreement {:.2f}, settled frames {})'.format(
            sess, float(a['meta']['side_agree_folds']), int(a['settled']['n_frames'])), flush=True)


def update_popgain():
    for neural, nonvis, sess, animal in sessions():
        pth = cache_path(sess)
        if not os.path.isfile(pth):
            continue
        D, ev, cells, lag = load(neural, nonvis)
        E = frame_epochs(D, ev)
        r = test_popgain(D, E, np.random.default_rng(6))
        with h5py.File(pth, 'a') as f:
            if 'popgain' in f:
                del f['popgain']
            _write(f.create_group('popgain'), r)
        print('  population eye tuning added:', sess, flush=True)


def update_heldout():
    for neural, nonvis, sess, animal in sessions():
        pth = cache_path(sess)
        if not os.path.isfile(pth):
            continue
        D, ev, cells, lag = load(neural, nonvis)
        E = frame_epochs(D, ev)
        h = test_heldout(D, E, np.random.default_rng(4))
        with h5py.File(pth, 'a') as f:
            if 'heldout' in f:
                del f['heldout']
            _write(f.create_group('heldout'), h)
        print('  held-out class prediction added:', sess, flush=True)


def update_behaviour():
    """ Add the behaviour-class tests (Figures 13-14) to every cached session. """
    for neural, nonvis, sess, animal in sessions():
        pth = cache_path(sess)
        if not os.path.isfile(pth):
            continue
        D, ev, cells, lag = load(neural, nonvis)
        E = frame_epochs(D, ev)
        b = test_behaviour(D, E, ev, np.random.default_rng(3))
        with h5py.File(pth, 'a') as f:
            if 'behaviour' in f:
                del f['behaviour']
            _write(f.create_group('behaviour'), b)
        print('  behaviour tests added:', sess, {k: int(v) for k, v in b['n_frames'].items()}, flush=True)


def update_active():
    """ Add the state-matched (active-period) tests to every cached session. """
    for neural, nonvis, sess, animal in sessions():
        pth = cache_path(sess)
        if not os.path.isfile(pth):
            continue
        D, ev, cells, lag = load(neural, nonvis)
        E = frame_epochs(D, ev)
        a = test_active(D, E, np.random.default_rng(2))
        with h5py.File(pth, 'a') as f:
            if 'active' in f:
                del f['active']
            _write(f.create_group('active'), a)
        print('  active-period tests added:', sess, flush=True)


def update_within_and_kernel_null():
    """ Recompute the within-fixation pair tests and add the kernel null in every cached session. """
    for neural, nonvis, sess, animal in sessions():
        pth = cache_path(sess)
        if not os.path.isfile(pth):
            continue
        D, ev, cells, lag = load(neural, nonvis)
        E = frame_epochs(D, ev)
        rng = np.random.default_rng(1)
        w = test_within(D, E, rng)
        kn = kernel_null(D, ev, rng)
        with h5py.File(pth, 'a') as f:
            del f['within']
            _write(f.create_group('within'), w)
            if 'kernel_null_call' in f['kernels']:
                del f['kernels']['kernel_null_call']
            f['kernels']['kernel_null_call'] = kn
        print('  within + kernel null updated:', sess, flush=True)


def update_transients():
    """ Recompute only the transients group of every cached session (after changing transients()). """
    for neural, nonvis, sess, animal in sessions():
        pth = cache_path(sess)
        if not os.path.isfile(pth):
            continue
        D, ev, cells, lag = load(neural, nonvis)
        tr = transients(D, ev, cells, np.random.default_rng(0))
        with h5py.File(pth, 'a') as f:
            del f['transients']
            _write(f.create_group('transients'), tr)
        print('  transients updated:', sess, flush=True)


# ---------------------------------------------------------------------------
# run + cache
# ---------------------------------------------------------------------------

def _write(g, d):
    for k, v in d.items():
        if isinstance(v, dict):
            _write(g.create_group(k), v)
        else:
            a = np.asarray(v)
            g[k] = a.astype('S') if a.dtype.kind == 'U' else a


def _read(g):
    return {k: (_read(v) if isinstance(v, h5py.Group) else v[()]) for k, v in g.items()}


def run_session(neural_path, nonvis_path, sess, animal, seed=0):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    D, ev, cells, lag = load(neural_path, nonvis_path)
    E = frame_epochs(D, ev)
    M = epoch_masks(D, E)
    with h5py.File(nonvis_path, 'r') as f:
        q_shift = f['comp_norm_spikes_light']['q_shift'][()] if 'comp_norm_spikes_light' in f else np.full(len(cells['area']), np.nan)
    R = {'meta': {'session': sess, 'animal': animal, 'lag': lag, 'version': VERSION, 'dt': D['dt']},
         'cells': {'area': cells['area'].astype('S'), 'tuned': cells['tuned'], 'gain': cells['gain'], 'q_shift': q_shift},
         'frames': {k: np.array(int(v.sum())) for k, v in M.items()},
         'e_hist': {k: np.histogram(D['e'][M[k]] - np.nanmedian(D['e'][M['all']]), np.arange(-40, 41, 2))[0]
                    for k in ('early', 'late')}}
    R['frames']['inside_light'] = np.array(int((light_rows(D) & E['inside']).sum()))
    print('  {}: {} cells, {} light frames (early {}, late {}, short {}, long {})'.format(
        sess, len(cells['area']), *(int(M[k].sum()) for k in ('all', 'early', 'late', 'short', 'long'))), flush=True)
    R['transients'] = transients(D, ev, cells, rng)
    R['epochs'] = test_epochs(D, E, rng)
    R['prev'] = test_prev(D, E, ev, rng)
    R['kernels'] = test_kernels(D, ev, rng)
    R['within'] = test_within(D, E, rng)
    R['meta']['runtime_s'] = time.time() - t0
    return R


def cache_path(sess):
    return os.path.join(CACHE, sess + '.h5')


def compute(which=None, force=False):
    os.makedirs(CACHE, exist_ok=True)
    for neural, nonvis, sess, animal in sessions():
        if which and not any(w in sess for w in which):
            continue
        p = cache_path(sess)
        if os.path.isfile(p) and not force:
            with h5py.File(p, 'r') as f:
                if int(f['meta']['version'][()]) == VERSION:
                    continue
        try:
            R = run_session(neural, nonvis, sess, animal)
        except ValueError as err:
            print('  {}: skipped ({})'.format(sess, err), flush=True)
            continue
        try:
            with h5py.File(p, 'w') as f:
                _write(f, R)
        except Exception:
            os.remove(p)
            raise
        print('  saved {} ({:.0f} s)'.format(p, R['meta']['runtime_s']), flush=True)


def results():
    """ Every cached session, as {session: nested dict}. """
    out = {}
    for p in sorted(glob.glob(os.path.join(CACHE, '*.h5'))):
        with h5py.File(p, 'r') as f:
            out[os.path.basename(p)[:-3]] = _read(f)
    return out


def update_classes():
    """ Recompute every group that uses the behaviour classes (Figures 13-14) and stamp the current VERSION;
    the fixation-epoch groups (Figure 12) do not depend on the class margins. """
    for f in (update_behaviour, update_heldout, update_raw, update_popgain, update_arrival):
        print(f.__name__, flush=True)
        f()
    for p in glob.glob(os.path.join(CACHE, '*.h5')):
        with h5py.File(p, 'a') as f:
            f['meta']['version'][()] = VERSION


def compute_all(which=None, force=False):
    """ Base tests, then every group the figures read, in the order they were first added. """
    compute(which, force)
    for f in (update_within_and_kernel_null, update_active, update_behaviour, update_heldout, update_raw,
              update_popgain, update_arrival):
        print(f.__name__, flush=True)
        f()


if __name__ == '__main__':
    # python gaze_shift_controls.py [--force] [--all] [session substrings];  --classes: class groups only
    args = [a for a in sys.argv[1:] if not a.startswith('--')] or None
    if '--classes' in sys.argv:
        update_classes()
    else:
        (compute_all if '--all' in sys.argv else compute)(args, force='--force' in sys.argv)


# ---------------------------------------------------------------------------
# pooling for the figures
# ---------------------------------------------------------------------------

AREAS = ['V1', 'RL', 'A', 'AM']


def pooled():
    """ Concatenate per-cell results over sessions. Returns (cells table dict, results dict). """
    R = results()
    cat = {}

    def add(key, v):
        cat.setdefault(key, []).append(np.asarray(v))

    for sess, r in R.items():
        C = len(r['cells']['area'])
        add('session', np.array([sess] * C))
        add('area', np.char.decode(r['cells']['area'].astype('S'), 'utf-8'))
        for k in ('tuned', 'gain', 'q_shift'):
            add(k, r['cells'][k])
        tr = r['transients']
        add('peth_a', tr['peth_a'])
        add('peth_b', tr['peth_b'])
        add('enh_a', tr['enh_a'])
        add('enh_b', tr['enh_b'])
        for k in ('resp_p', 'resp_q', 'resp_d'):
            add(k, tr[k] if k in tr else np.full(C, np.nan))
        for k, v in r['epochs'].items():
            if k.startswith('cross'):
                add(k + '/gain', v['gain'])
                add(k + '/null', v['null'].T)
            else:
                for kk in ('call', 'd', 'tuned'):
                    add(k + '/' + kk, v[kk])
                add(k + '/null_call', v['null_call'].T)
                add(k + '/d_null', v['d_null'].T)
                add(k + '/gain_prof', v['gain_prof'])
        for k, v in r['prev'].items():
            for kk, vv in v.items():
                if np.ndim(vv) == 0:
                    continue
                add('prev_' + k + '/' + kk, vv.T if vv.ndim == 2 else vv)
        for kk, vv in r['kernels'].items():
            add('kernels/' + kk, vv.T if vv.ndim == 2 else vv)
        nb = len(PROFILE_EDGES) - 1
        shapes = {'gain_sign': (C,), 'pref': (C,), 'field_amp': (C,), 'gain_term_sd': (C,), 'n_active': (C,)}
        for m in ('real', 'ret_model', 'gain_model', 'head_model'):
            shapes.update({'abc_' + m: (C, 3), 'prof_' + m: (C, nb), 'expr_' + m: (C, 4), 'expr_se_' + m: (C,)})
        H = r.get('heldout', {})
        for k in HELDOUT_CLASSES:
            v = H.get(k, {})
            add('held_' + k + '/gain', v['gain'] if 'gain' in v else np.full(C, np.nan))
            add('held_' + k + '/null', v['null'].T if 'null' in v else np.full((C, N_SHUF), np.nan))
        B = r.get('behaviour', {})
        for k in BEH_CLASSES:
            v = B.get(k, {})
            for kk in ('call', 'tuned', 'd'):
                add('beh_' + k + '/' + kk, v[kk] if kk in v else np.full(C, np.nan))
            add('beh_' + k + '/null_call', v['null_call'].T if 'null_call' in v else np.full((C, N_SHUF), np.nan))
        for a, b in BEH_CROSS:
            v = B.get('cross_{}_to_{}'.format(a, b), {})
            add('beh_cross_{}_to_{}/gain'.format(a, b), v['gain'] if 'gain' in v else np.full(C, np.nan))
            add('beh_cross_{}_to_{}/null'.format(a, b), v['null'].T if 'null' in v else np.full((C, N_SHUF), np.nan))
        for k in ('post', 'comp'):
            v = B.get('prev', {}).get(k, {})
            for kk in ('call_prev_beyond_frame', 'call_frame_beyond_prev'):
                add('beh_prev_{}/{}'.format(k, kk), v[kk] if kk in v else np.full(C, np.nan))
            for kk in ('null_prev_beyond_frame', 'null_frame_beyond_prev'):
                add('beh_prev_{}/{}'.format(k, kk), v[kk].T if kk in v else np.full((C, 5), np.nan))
        for kind in ('same', 'across'):
            w = B.get('pairs', {}).get(kind, {})
            for kk, shp in (('expr_real', (C, 4)), ('abc_real', (C, 3)), ('abc_head_model', (C, 3))):
                add('beh_' + kind + '/' + kk, w[kk] if kk in w else np.full(shp, np.nan))
            add('beh_' + kind + '/n_pairs', np.full(C, int(w.get('n_pairs', 0))))
        A = r.get('active', {})
        for k in ('early_act', 'late_act'):
            v = A.get(k, {})
            for kk in ('call', 'tuned', 'd'):
                add('act_' + k + '/' + kk, v[kk] if kk in v else np.full(C, np.nan))
            add('act_' + k + '/null_call', v['null_call'].T if 'null_call' in v else np.full((C, N_SHUF), np.nan))
        for k in ('cross_early_act_to_late_act', 'cross_late_act_to_early_act'):
            v = A.get(k, {})
            add('act_' + k + '/gain', v['gain'] if 'gain' in v else np.full(C, np.nan))
            add('act_' + k + '/null', v['null'].T if 'null' in v else np.full((C, N_SHUF), np.nan))
        for kind in ('same', 'across'):
            w = A.get('within', {}).get(kind, {})
            for kk in ('expr_real', 'abc_real', 'abc_head_model', 'gain_sign'):
                shp = {'expr_real': (C, 4), 'abc_real': (C, 3), 'abc_head_model': (C, 3), 'gain_sign': (C,)}[kk]
                add('act_' + kind + '/' + kk, w[kk] if kk in w else np.full(shp, np.nan))
            add('act_' + kind + '/n_pairs', np.full(C, int(w.get('n_pairs', 0))))
        for kind in ('same', 'across'):
            w = r['within'].get(kind, {})
            add(kind + '/n_pairs', np.full(C, int(w.get('n_pairs', 0))))
            for kk, shp in shapes.items():
                add(kind + '/' + kk, w[kk] if kk in w else np.full(shp, np.nan))
    T = {}
    for k, v in cat.items():
        try:
            T[k] = np.concatenate(v)
        except ValueError:
            pass
    return T, R
