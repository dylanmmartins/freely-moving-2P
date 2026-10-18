# -*- coding: utf-8 -*-
"""
fm2p/nonvisual_probe.py

Is a gaze-shift response in V1 visual or non-visual?  Four tests on one freely
moving session, run on the measured eye scale and the renderer's own pose.

1. Compensatory movements as the non-visual probe.  A compensatory head turn
   moves the head like a gaze shift but leaves gaze nearly still, so the
   retinal image barely changes.  Head movements with and without a gaze shift
   are matched one to one on peak head velocity and head amplitude (exactly on
   direction and light), and each cell's compensatory response is divided by
   its gaze-shift response.  The residual image change is measured, not
   assumed: the eye-on-head gain is below 1, so a compensatory turn still
   slips the image a little.
2. Regression within gaze shifts.  Each shift's response is modelled against
   the image change inside the cell's receptive field (luminance change, the
   pillar entering or leaving) and against head and eye kinematics.  Visual
   responses should follow image change, movement responses kinematics.  Dark
   shifts are the control: the same geometric image-change numbers exist
   there, but nothing is visible, so any variance they explain is leakage
   through the kinematics.
3. Onset timing.  Activity before shift onset is a movement signal.  2P frames
   are 133 ms long, but each cell is sampled when the scan passes its row, and
   shift onsets fall at random phases of the frame clock, so pooling events
   gives a peri-event histogram at 25 ms resolution.  A sample at time s holds
   the spikes inferred for (s - 133 ms, s], so a bin at tau <= 0 is purely
   pre-onset.  OMSI, OASIS and dF/F are compared; deconvolution is what makes
   the onset resolvable.
4. Direction preferences (Miura & Scanziani 2022, Nature 610:135).  Each
   cell's preference for nasal vs temporal gaze shifts is compared with its
   preference for temporal vs nasal image motion inside its receptive field
   outside gaze shifts, head velocity held fixed by stratification.  Both are
   signed so that a purely visual cell lands on the diagonal.  Split-half
   reliabilities give the ceiling, so "uncorrelated" is distinguishable from
   "too noisy to tell".

Receptive fields
----------------
From the sparse-noise session one directory up (`sn1/`): Gaussian fits in
monitor pixels (`sparse_noise_labels_gaussfit.npz`).  The ROIs are the same
418 suite2p cells as the freely moving session (checked pixel for pixel).
Monitor pixels go to retinal angles through a flat-screen model whose centre
sits on the optic axis; the focal length in pixels comes from the stimulus
script ("15-350 px is 1-40 deg", i.e. ~8.7 px/deg at the centre).  That
placement is an assumption, so every RF is widened by a 10 deg mapping
uncertainty, and the regression is repeated with the RF mirrored about the
vertical meridian and with a whole-field image change.  Cells without a fit
use the population RF.

Usage
-----
    python -m fm2p.nonvisual_probe \\
        --h5_path <prefix>_preproc.h5 \\
        --npz_path <prefix>_retinal_pose.npz

Functions
---------
scan_offsets
    Per-cell sample time inside the frame from the cell's scan row.
load_rates
    Spike-rate arrays for each requested key.
window_mean
    Each cell's mean rate in a window after each event, sample-time exact.
peth_samples, peth_from_samples
    Sub-frame peri-event histogram per cell; bootstrap-ready.
ramp_onset_fit
    Onset of a ramp seen through the 133 ms sample integration window.
_crossing
    Interpolated threshold crossing walking back from a frame.
gaze_shift_events
    Gaze-shift episodes with onsets, kinematics and isolation flags.
head_events
    Head movements, split into those carrying a gaze shift and compensatory ones.
match_events
    One-to-one nearest-neighbour matching inside strata, with a caliper.
load_rfs
    Receptive fields from sparse noise, in retinal degrees.
rf_weights
    Gaussian RF weights over a grid of directions.
render_labels
    Surface label per grid ray at chosen frames (renderer geometry).
image_change
    RF-weighted luminance change and pillar entry/exit between two frames.
compensatory_probe
    Matched compensatory vs gaze-shift responses and their ratio.
_ridge_cv, shift_regression
    Visual vs kinematic unique variance within gaze shifts.
onset_timing
    Population onset relative to gaze and head onset, with bootstrap CIs.
omsi_lag_check
    Lag between OMSI spikes and the first rise of dF/F.
direction_preference
    Gaze-shift vs visual-motion direction preference and their correlation.
run_probe
    Every test for one session.
save_probe
    One-session HDF5 with the per-cell results.
_page_summary, _page_comp, _page_regression, _page_timing, _page_direction,
_page_methods
    PDF pages.
make_pdf
    Multi-page summary PDF.
main
    Command-line entry point.


DMM, October 2026
"""

if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

import argparse
import json
import os
import textwrap
import time

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr, ttest_1samp

from . import new_analysis as na
from .utils.files import write_h5

# Event detection on the 40 Hz grid.
HEAD_RUN = 60.0
HEAD_ON = 30.0
GAZE_ON = 60.0
MAX_WALK = 8
PAD = 4
ISO_S = 0.4
ISO_TIMING = 0.8
# Response windows, s from onset.
BASE_WIN = (-0.5, -0.1)
MATCH_WIN = (0.0, 0.5)
SHIFT_WIN = (0.0, 0.4)
CALIPER = 0.5
# Peri-event histograms.
PETH_HALF = 1.0
PETH_BIN = 0.025
FIT_WIN = (-0.8, 0.6)
# Visual motion preference outside gaze shifts.
U0 = 5.0
HEAD_STRATA = np.array([-np.inf, -150, -90, -45, -15, 15, 45, 90, 150, np.inf])
EXCL_PRE, EXCL_POST = 0.3, 0.5
MIN_STRATUM = 15
# Receptive fields.
MON_SHAPE = (768, 1360)
MON_F_PX = 500.0
MAP_SIGMA = 10.0
GRID_HALF, GRID_STEP = 60.0, 2.0
LUM = np.array([255.0, 170.0, 100.0, 237.0]) / 255.0
# Statistics.
N_SHUF = 200
RIDGE_LAM = 20.0
N_BOOT = 300
N_PERM = 1000
SPIKE_KEYS = ('norm_spikes', 'oasis_spks', 'norm_dFF')
KEY_LABEL = {'norm_spikes': 'OMSI', 'oasis_spks': 'OASIS', 'norm_dFF': 'dF/F', 'omsi_asis': 'OMSI as stored'}
KEY_COLOR = {'norm_spikes': '#2a78d6', 'oasis_spks': '#eb6834', 'norm_dFF': '#8a8986', 'omsi_asis': '#9ec1ef'}
COND_COLOR = {'light': '#eda100', 'dark': '#2b2b6b', 'comp': '#b158c3', 'shift': '#2a78d6'}
INK, INK2, GRID = na.INK, na.INK2, na.GRID


def scan_offsets(h5_path):
    """ Per-cell sample time inside the frame from the cell's scan row.

    A resonant scanner sweeps the frame top to bottom once per frame, so a cell
    at mean row y is sampled y / n_rows of a frame after the frame starts.
    `twopT` is taken as the frame start (ScanImage frame timestamps).

    Returns
    -------
    twopT : (F,)
    dt : float
        Frame period, s.
    offset : (C,)
        Seconds after the frame start at which each cell is sampled.
    """

    with h5py.File(h5_path, 'r') as f:
        twopT = np.asarray(f['twopT'][:], float)
        n_rows = f['twop_mean_img'].shape[0]
        n_cells = f['norm_spikes'].shape[0]
        yc = np.array([np.mean(f['cell_y_pix/{}'.format(c)][()]) for c in range(n_cells)])
    dt = float(np.median(np.diff(twopT)))
    return twopT, dt, yc / n_rows * dt


def load_rates(h5_path, keys):
    """ Spike-rate arrays (C, F) for each requested key, NaN-free where possible. """

    out = {}
    with h5py.File(h5_path, 'r') as f:
        for k in keys:
            if k in f:
                out[k] = np.asarray(f[k][:], float)
            else:
                print('  WARNING: {} not in the preproc file -- skipped.'.format(k))
    return out


def window_mean(rate, samp, dt, T, a, b):
    """ Each cell's mean rate in [T + a, T + b), sample-time exact.

    A sample at time s integrates (s - dt, s]; it is counted when the centre of
    that interval falls in the window.  Windows running off either end of the
    recording are NaN.

    Returns
    -------
    m : (C, E) mean rate.
    n : (C, E) samples used.
    """

    C, F = rate.shape
    T = np.asarray(T, float)
    cen = samp - dt / 2.0
    fin = np.isfinite(rate)
    cs = np.concatenate([np.zeros((C, 1)), np.cumsum(np.where(fin, rate, 0.0), 1)], 1)
    cn = np.concatenate([np.zeros((C, 1)), np.cumsum(fin, 1)], 1)
    m = np.full((C, len(T)), np.nan)
    n = np.zeros((C, len(T)))
    for c in range(C):
        lo = np.searchsorted(cen[c], T + a)
        hi = np.searchsorted(cen[c], T + b)
        cnt = cn[c, hi] - cn[c, lo]
        inside = (T + a >= cen[c, 0]) & (T + b <= cen[c, -1]) & (cnt > 0)
        m[c, inside] = (cs[c, hi] - cs[c, lo])[inside] / cnt[inside]
        n[c] = cnt
    return m, n


def peth_samples(rate, samp, T, half=PETH_HALF, binw=PETH_BIN):
    """ Per-cell (bin, event, value) triplets for a sub-frame PETH.

    tau = sample time - onset, so the bin at tau holds spikes from
    (tau - dt, tau].  Kept unaggregated so a bootstrap over events only needs
    event weights.

    Returns
    -------
    S : list of (bin, event, value) array triplets, one per cell.
    centres : (nb,) bin centres, s.
    """

    T = np.asarray(T, float)
    nb = int(round(2 * half / binw))
    S = []
    for c in range(rate.shape[0]):
        lo = np.searchsorted(samp[c], T - half)
        hi = np.searchsorted(samp[c], T + half)
        cnt = hi - lo
        tot = int(cnt.sum())
        ev = np.repeat(np.arange(len(T)), cnt)
        fr = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(lo, cnt)
        tau = samp[c, fr] - T[ev]
        b = np.floor((tau + half) / binw).astype(int)
        val = rate[c, fr]
        ok = (b >= 0) & (b < nb) & np.isfinite(val)
        S.append((b[ok], ev[ok], val[ok]))
    centres = -half + binw * (np.arange(nb) + 0.5)
    return S, centres


def peth_from_samples(S, nb, w=None, cells=None):
    """ (C, nb) PETH from peth_samples, optionally with event weights. """

    cells = range(len(S)) if cells is None else cells
    out = np.full((len(cells), nb), np.nan)
    for i, c in enumerate(cells):
        b, ev, val = S[c]
        ww = np.ones(len(b)) if w is None else w[ev]
        num = np.bincount(b, weights=val * ww, minlength=nb)
        den = np.bincount(b, weights=ww, minlength=nb)
        out[i] = num / np.where(den > 0, den, np.nan)
    return out


def _ramp_integral(t, t0, rise):
    """ Integral of a 0-to-1 ramp starting at t0 and lasting `rise` s. """

    x = t - t0
    return np.where(x < 0, 0.0, np.where(x < rise, x ** 2 / (2 * rise), rise / 2 + (x - rise)))


def ramp_onset_fit(tau, y, dt, win=FIT_WIN):
    """ Onset of a ramp seen through the (s - dt, s] sample integration.

    The population response is modelled as baseline + A * ramp(t; t0, rise);
    a sample at tau sees its mean over (tau - dt, tau].  Grid search over t0
    and rise, least squares for baseline and A.

    Returns
    -------
    dict with t0 (ramp start), t50 (ramp midpoint), rise, amp, base, sse.
    """

    m = (tau >= win[0]) & (tau <= win[1]) & np.isfinite(y)
    tau, y = tau[m], y[m]
    best = {'sse': np.inf, 't0': np.nan, 'rise': np.nan, 'amp': np.nan, 'base': np.nan, 't50': np.nan}
    if len(y) < 10:
        return best
    for rise in (0.025, 0.05, 0.1, 0.15, 0.2, 0.3):
        for t0 in np.arange(-0.6, 0.4, 0.005):
            r = (_ramp_integral(tau, t0, rise) - _ramp_integral(tau - dt, t0, rise)) / dt
            X = np.stack([np.ones_like(r), r], 1)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            if coef[1] <= 0:
                continue
            sse = float(np.sum((y - X @ coef) ** 2))
            if sse < best['sse']:
                best = {'sse': sse, 't0': float(t0), 'rise': rise, 'amp': float(coef[1]),
                        'base': float(coef[0]), 't50': float(t0 + rise / 2)}
    return best


def _crossing(x, t, i, thr, sgn, max_walk=MAX_WALK):
    """ Interpolated time x crosses thr walking back from frame i (sign sgn). """

    j = i
    k = 0
    while j > 0 and k < max_walk and np.isfinite(x[j - 1]) and sgn * x[j - 1] > thr:
        j -= 1
        k += 1
    if j == 0 or not np.isfinite(x[j - 1]) or not np.isfinite(x[j]):
        return float(t[j]), j
    a, b = sgn * x[j - 1], sgn * x[j]
    frac = np.clip((thr - a) / (b - a), 0.0, 1.0) if b != a else 0.0
    return float(t[j - 1] + frac * (t[j] - t[j - 1])), j


def gaze_shift_events(s, v, lab):
    """ Gaze-shift episodes with onsets, kinematics and isolation flags.

    Onset = the interpolated time |gaze| last rose through 60 deg/s before the
    240 deg/s classifier frame; head onset the same on |head| at 30 deg/s
    when the head is moving with the shift.

    Returns
    -------
    g : dict of (E,) arrays.
    """

    t = s['t']
    ev = na.episode_events(lab, v)
    rows = []
    for a, b, d in zip(ev['ep_start'], ev['ep_end'], ev['direction']):
        t_on, j = _crossing(v['gaze'], t, a, GAZE_ON, d)
        lo, hi = max(a - PAD, 0), min(b + PAD, len(t))
        h = v['head'][lo:hi]
        hd = np.sign(np.nanmean(h)) if np.isfinite(h).any() else 0.0
        if hd != 0 and np.nanmax(hd * h) > HEAD_ON:
            th_on, _ = _crossing(v['head'], t, int(lo + np.nanargmax(hd * h)), HEAD_ON, hd, max_walk=16)
        else:
            th_on = np.nan
        seg = slice(a, b)
        rows.append((t_on, th_on, a, b, j, d, hd,
                     float(np.nanmax(np.abs(v['head'][lo:hi]))) if np.isfinite(v['head'][lo:hi]).any() else np.nan,
                     float(np.nansum(v['head'][seg]) * na.DT),
                     float(np.nansum(v['eye'][seg]) * na.DT),
                     float(np.nansum(v['gaze'][seg]) * na.DT),
                     float(np.nanmax(np.abs(v['eye'][seg]))) if np.isfinite(v['eye'][seg]).any() else np.nan,
                     float(np.nanmax(np.abs(v['gaze'][seg]))),
                     bool(s['light'][a])))
    cols = ('t_on', 't_head_on', 'ep_start', 'ep_end', 'on_idx', 'direction', 'head_dir', 'head_peak',
            'head_amp', 'eye_amp', 'gaze_amp', 'eye_peak', 'gaze_peak', 'light')
    g = {c: np.array([r[i] for r in rows]) for i, c in enumerate(cols)}
    for c in ('ep_start', 'ep_end', 'on_idx', 'direction'):
        g[c] = g[c].astype(int)
    g['light'] = g['light'].astype(bool)
    # Isolated: no other gaze-shift onset in the preceding ISO_S.
    on = g['t_on']
    g['gap'] = on - np.r_[-np.inf, on[:-1]]
    g['isolated'] = g['gap'] > ISO_S
    return g


def head_events(s, v, lab, gaze_onsets=None):
    """ Head movements, split into gaze-shift-carrying and compensatory ones.

    A head movement is a same-sign run of |head| > 60 deg/s (gaps of up to 3
    frames merged, at least 3 frames), its onset and offset where |head|
    crosses 30 deg/s.  It carries a gaze shift if a gaze-shift frame in the
    head's direction falls inside it (+-100 ms); it is compensatory if instead
    |gaze| stays below 120 deg/s throughout.  Movements with missing eye data
    are dropped.

    Returns
    -------
    h : dict of (E,) arrays; 'kind' 1 = shift, 0 = compensatory, -1 neither.
    """

    t, hv, gz = s['t'], v['head'], v['gaze']
    n = len(t)
    rows = []
    for sgn in (1.0, -1.0):
        runs = na._runs(np.nan_to_num(sgn * hv) > HEAD_RUN)
        merged = []
        for a, b in runs:
            if merged and a - merged[-1][1] <= 3:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        for a, b in merged:
            if b - a < 3:
                continue
            t_on, on_i = _crossing(hv, t, a, HEAD_ON, sgn, max_walk=16)
            off = b
            while off < n - 1 and np.isfinite(hv[off]) and sgn * hv[off] > HEAD_ON and off - b < 16:
                off += 1
            lo, hi = max(on_i - PAD, 0), min(off + PAD, n)
            seg = slice(on_i, off)
            g_ok = np.isfinite(gz[lo:hi]).all()
            shift_frames = np.isin(lab[lo:hi], (1, 2, -2))
            same_dir = shift_frames & (np.sign(gz[lo:hi]) == sgn)
            gmax = float(np.nanmax(np.abs(gz[lo:hi]))) if g_ok else np.nan
            if not g_ok:
                kind = -1
            elif same_dir.any():
                kind = 1
            elif not shift_frames.any() and gmax < na.STILL_GAZE:
                kind = 0
            else:
                kind = -1
            rows.append((t_on, on_i, off, sgn, kind,
                         float(np.nanmax(np.abs(hv[on_i:off]))),
                         float(abs(np.nansum(hv[seg]) * na.DT)),
                         float(np.nansum(gz[seg]) * na.DT) if g_ok else np.nan,
                         gmax, (off - on_i) * na.DT, bool(s['light'][on_i])))
    rows.sort()
    cols = ('t_on', 'on_idx', 'off_idx', 'direction', 'kind', 'head_peak', 'head_amp',
            'gaze_disp', 'gaze_max', 'dur', 'light')
    h = {c: np.array([r[i] for r in rows]) for i, c in enumerate(cols)}
    for c in ('on_idx', 'off_idx', 'kind'):
        h[c] = h[c].astype(int)
    h['light'] = h['light'].astype(bool)
    # Quiet before: no other head-movement onset and no gaze-shift onset.
    on = h['t_on']
    g_on = np.sort(np.asarray(gaze_onsets, float)) if gaze_onsets is not None else np.array([])
    prev_h = np.r_[-np.inf, on[:-1]]
    j = np.searchsorted(g_on, on - 1e-6) - 1
    prev_g = np.where(j >= 0, g_on[np.clip(j, 0, None)] if len(g_on) else -np.inf, -np.inf)
    # A shift inside the movement itself does not count against it.
    own = (h['kind'] == 1) & (on - prev_g < 0.2)
    prev_g = np.where(own, -np.inf, prev_g)
    h['gap'] = on - np.maximum(prev_h, prev_g)
    h['isolated'] = h['gap'] > ISO_S
    return h


def match_events(fa, fb, sa, sb, caliper=CALIPER, seed=0):
    """ One-to-one nearest-neighbour matching inside strata, with a caliper.

    fa, fb : (Ea, k), (Eb, k) features, standardized together.
    sa, sb : stratum labels (exact match).

    Returns
    -------
    pairs : (P, 2) int indices into a and b.
    """

    rng = np.random.default_rng(seed)
    allf = np.vstack([fa, fb])
    mu, sd = np.nanmean(allf, 0), np.nanstd(allf, 0)
    za, zb = (fa - mu) / sd, (fb - mu) / sd
    pairs = []
    for st in np.unique(sa):
        ia = np.flatnonzero(sa == st)
        ib = list(np.flatnonzero(sb == st))
        for i in rng.permutation(ia):
            if not ib:
                break
            d = np.linalg.norm(zb[ib] - za[i], axis=1)
            k = int(np.argmin(d))
            if d[k] < caliper:
                pairs.append((i, ib.pop(k)))
    return np.array(pairs, int).reshape(-1, 2)


def load_rfs(sn_dir, n_cells, f_px=MON_F_PX, mirror=False):
    """ Receptive fields from sparse noise, in retinal degrees.

    Monitor pixel (x, y) to the optical frame of the right eye: x grows to the
    animal's right (temporal), y down; the screen centre is on the optic axis.

    Returns
    -------
    rf : dict with az, el, sig (C,) deg, has_fit (C,) bool and the population RF.
    """

    z = np.load(os.path.join(sn_dir, 'sparse_noise_labels_gaussfit.npz'))
    idx = np.asarray(z['true_indices'], int)
    cx, cy = MON_SHAPE[1] / 2.0, MON_SHAPE[0] / 2.0
    x = np.asarray(z['pos_centroids'][:, 0], float) - cx
    y = np.asarray(z['pos_centroids'][:, 1], float) - cy
    az = np.degrees(np.arctan2(x, f_px)) * (-1 if mirror else 1)
    el = -np.degrees(np.arctan2(y, np.hypot(f_px, x)))
    sig = np.degrees(np.mean(np.abs(z['pos_sigmas']), 1) / f_px)
    ok = np.isfinite(az) & np.isfinite(el) & np.isfinite(sig) & (np.abs(az) < 60) & (np.abs(el) < 60)
    out = {'az': np.full(n_cells, np.nan), 'el': np.full(n_cells, np.nan),
           'sig': np.full(n_cells, np.nan), 'has_fit': np.zeros(n_cells, bool)}
    out['az'][idx[ok]] = az[ok]
    out['el'][idx[ok]] = el[ok]
    out['sig'][idx[ok]] = sig[ok]
    out['has_fit'][idx[ok]] = True
    pop_az, pop_el = float(np.median(az[ok])), float(np.median(el[ok]))
    spread = float(np.sqrt(0.5 * (np.var(az[ok]) + np.var(el[ok]))))
    pop_sig = float(np.sqrt(np.median(sig[ok]) ** 2 + spread ** 2))
    for k, val in (('az', pop_az), ('el', pop_el), ('sig', pop_sig)):
        out[k][~out['has_fit']] = val
    out.update({'pop_az': pop_az, 'pop_el': pop_el, 'pop_sig': pop_sig, 'mirror': mirror,
                'n_fit': int(ok.sum())})
    return out


def rf_weights(az_g, el_g, az, el, sig, map_sigma=MAP_SIGMA):
    """ Gaussian RF weights over a grid of directions, rows summing to 1.

    The width adds the mapping uncertainty in quadrature; cos(el) weights each
    grid direction by its solid angle.
    """

    s2 = sig[:, None] ** 2 + map_sigma ** 2
    d2 = (az_g[None] - az[:, None]) ** 2 + (el_g[None] - el[:, None]) ** 2
    w = np.exp(-d2 / (2 * s2)) * np.cos(np.radians(el_g))[None]
    return w / w.sum(1, keepdims=True)


def _grid_dirs(half=GRID_HALF, step=GRID_STEP):
    """ Equiangular grid of unit directions in the optical frame. """

    a = np.arange(-half, half + step / 2, step)
    AZ, EL = np.meshgrid(a, a)
    az, el = np.radians(AZ.ravel()), np.radians(EL.ravel())
    d = np.stack([np.cos(el) * np.sin(az), -np.sin(el), np.cos(el) * np.cos(az)], 1)
    return d, AZ.ravel(), EL.ravel()


def render_labels(s, origin, rot, idx, dirs, chunk=400):
    """ Surface label (0 sky, 1 wall, 2 floor, 3 pillar) per ray at frames idx.

    Returns
    -------
    lab : (len(idx), P) uint8; ok : (len(idx),) bool where the pose was valid.
    """

    idx = np.asarray(idx, int)
    lab = np.zeros((len(idx), len(dirs)), np.uint8)
    ok = np.isfinite(rot[idx, 0, 0]) & np.isfinite(origin[idx, 0])
    good = np.flatnonzero(ok)
    for a in range(0, len(good), chunk):
        pos = good[a:a + chunk]
        jj = idx[pos]
        dw = np.einsum('nij,pj->npi', rot[jj], dirs)
        _, lb = na._raycast_depth(origin[jj], dw, s['arena'], s['pillar_xy'],
                                  s['pillar_r'], s['pillar_h'], s['wall_h'])
        lab[pos] = lb
    return lab, ok


def image_change(lab_pre, lab_post, W):
    """ RF-weighted luminance change and pillar entry/exit between two frames.

    Returns
    -------
    dict of (R, E): dI (luminance change, 0-1), enter, leave (pillar fraction).
    """

    Lp, Lq = LUM[lab_pre], LUM[lab_post]
    pp, pq = (lab_pre == 3).astype(float), (lab_post == 3).astype(float)
    dI = W @ np.abs(Lq - Lp).T
    d_pil = W @ (pq - pp).T
    return {'dI': dI, 'enter': np.clip(d_pil, 0, None), 'leave': np.clip(-d_pil, 0, None),
            'pil_pre': W @ pp.T}


def _bh(p):
    """ Benjamini-Hochberg q-values. """

    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    ps = p[ok]
    o = np.argsort(ps)
    r = ps[o] * len(ps) / (np.arange(len(ps)) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    qq = np.empty_like(ps)
    qq[o] = np.clip(r, 0, 1)
    q[ok] = qq
    return q


def compensatory_probe(rate, samp, dt, hev, imc_cells, seed=0):
    """ Matched compensatory vs gaze-shift responses and their ratio.

    Responses are baseline-subtracted mean rates in MATCH_WIN after head
    onset.  Responsive cells are picked on one half of the pairs and the ratio
    read off the other half (then swapped), so selection does not inflate the
    gaze-shift response the ratio divides by.

    Returns
    -------
    out : dict per condition ('light', 'dark') of pairs, per-cell responses,
        ratios, population ratio with bootstrap CI, and residual image change.
    """

    rng = np.random.default_rng(seed)
    out = {}
    # Head movements come every ~0.4 s, so isolation would leave a handful of
    # pairs.  Instead both groups are matched on the time since the previous
    # movement or shift (capped at 2 s): equal history, equal contamination.
    feats = np.stack([np.log(hev['head_peak']), np.log(hev['head_amp']),
                      np.minimum(hev['gap'], 2.0)], 1)
    for cond, lt in (('light', True), ('dark', False)):
        base = (hev['light'] == lt) & np.isfinite(feats).all(1)
        ia = np.flatnonzero(base & (hev['kind'] == 1))
        ib = np.flatnonzero(base & (hev['kind'] == 0))
        if len(ia) < 10 or len(ib) < 10:
            continue
        pr = match_events(feats[ia], feats[ib], hev['direction'][ia], hev['direction'][ib], seed=seed)
        pa, pb = ia[pr[:, 0]], ib[pr[:, 1]]
        T_all = hev['t_on'][np.r_[pa, pb]]
        r1, _ = window_mean(rate, samp, dt, T_all, *MATCH_WIN)
        r0, _ = window_mean(rate, samp, dt, T_all, *BASE_WIN)
        d = r1 - r0
        ds, dc = d[:, :len(pa)], d[:, len(pa):]
        good = np.isfinite(ds).all(0) & np.isfinite(dc).all(0)
        ds, dc, pa, pb = ds[:, good], dc[:, good], pa[good], pb[good]
        P = ds.shape[1]
        # Cross-validated selection: odd pairs pick, even pairs measure; swap.
        half = np.arange(P) % 2
        sel = np.zeros((2, rate.shape[0]), bool)
        ratio_h = np.full((2, rate.shape[0]), np.nan)
        for k in (0, 1):
            pick, meas = half == k, half != k
            tt = ttest_1samp(ds[:, pick], 0, axis=1)
            sel[k] = (_bh(tt.pvalue) < 0.1) & np.isfinite(tt.pvalue)
            ms, mc = ds[:, meas].mean(1), dc[:, meas].mean(1)
            sgn = np.sign(ds[:, pick].mean(1))
            with np.errstate(invalid='ignore', divide='ignore'):
                ratio_h[k] = np.where(sel[k] & (sgn * ms > 0), mc / ms, np.nan)
        tt_all = ttest_1samp(ds, 0, axis=1)
        sgn_all = np.sign(ds.mean(1))

        def pop_ratio(cols, cells):
            num = np.sum(sgn_all[cells] * dc[cells][:, cols].mean(1))
            den = np.sum(sgn_all[cells] * ds[cells][:, cols].mean(1))
            return num / den if den > 0 else np.nan

        resp = np.flatnonzero(sel.any(0))
        pr_obs = pop_ratio(np.arange(P), resp) if len(resp) else np.nan
        boots = np.array([pop_ratio(rng.integers(0, P, P), resp) for _ in range(N_BOOT)]) if len(resp) else np.array([])
        # No selection at all: the pooled rate of every cell, as a fractional
        # change from the pooled baseline, for each movement type.
        b_s = r0[:, :len(pa)][:, good]
        b_c = r0[:, len(pa):][:, good]

        def pooled(cols):
            fs = ds[:, cols].sum() / max(b_s[:, cols].sum(), 1e-12)
            fc = dc[:, cols].sum() / max(b_c[:, cols].sum(), 1e-12)
            return fs, fc
        fs0, fc0 = pooled(np.arange(P))
        bt = np.array([pooled(rng.integers(0, P, P)) for _ in range(N_BOOT)])
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio_bt = bt[:, 1] / bt[:, 0]
        # Residual image change in each half of the pairs, population RF.
        out[cond] = {
            'pairs': np.stack([pa, pb], 1), 'n_pairs': P,
            'resp_shift': ds.mean(1), 'resp_comp': dc.mean(1),
            'p_shift': tt_all.pvalue, 'q_shift': _bh(tt_all.pvalue), 'sign_shift': sgn_all,
            'selected': sel.any(0), 'ratio_cell': np.nanmean(ratio_h, 0),
            'pooled_shift': fs0, 'pooled_shift_ci': np.nanpercentile(bt[:, 0], [2.5, 97.5]),
            'pooled_comp': fc0, 'pooled_comp_ci': np.nanpercentile(bt[:, 1], [2.5, 97.5]),
            'pooled_ratio': fc0 / fs0 if fs0 > 0 else np.nan,
            'pooled_ratio_ci': np.nanpercentile(ratio_bt[bt[:, 0] > 0], [2.5, 97.5]) if (bt[:, 0] > 0).any() else np.array([np.nan, np.nan]),
            'pop_ratio': pr_obs, 'pop_ratio_ci': (np.nanpercentile(boots, [2.5, 97.5]) if len(boots) else np.array([np.nan, np.nan])),
            'balance': {k: (hev[k][pa], hev[k][pb]) for k in ('head_peak', 'head_amp', 'gaze_disp', 'gaze_max', 'dur', 'gap')},
            'dI_shift': imc_cells['dI'][pa] if imc_cells is not None else None,
            'dI_comp': imc_cells['dI'][pb] if imc_cells is not None else None,
        }
    return out


def _ridge_cv(X, Y, folds, lam=RIDGE_LAM):
    """ Cross-validated ridge R^2 per column of Y, X z-scored on each training set. """

    sse = np.zeros(Y.shape[1])
    sst = np.zeros(Y.shape[1])
    for te in folds:
        tr = np.ones(len(X), bool)
        tr[te] = False
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        Xt = np.c_[np.ones(tr.sum()), (X[tr] - mu) / sd]
        Xe = np.c_[np.ones(len(te)), (X[te] - mu) / sd]
        A = Xt.T @ Xt + lam * np.diag(np.r_[0.0, np.ones(X.shape[1])])
        B = np.linalg.solve(A, Xt.T @ Y[tr])
        sse += np.sum((Y[te] - Xe @ B) ** 2, 0)
        sst += np.sum((Y[te] - Y[tr].mean(0)) ** 2, 0)
    return 1.0 - sse / np.where(sst > 0, sst, np.nan)


def shift_regression(rate, samp, dt, gev, imc, kin_names=('head_peak', 'head_amp', 'eye_amp', 'eye_peak', 'direction'),
                     min_spikes=10.0, seed=0, counts=True):
    """ Visual vs kinematic unique variance within gaze shifts.

    Response = expected spike count in SHIFT_WIN after gaze onset
    (Anscombe-transformed when `counts`), with the baseline count as a nuisance
    term in every model.  Visual group: RF luminance change, pillar entering,
    pillar leaving.  Kinematic group: head peak velocity and amplitude, eye
    amplitude and peak velocity, direction.  Unique R^2 of a group = CV R^2 of
    the full model minus the model without it, 5 contiguous folds.  Null:
    visual (or kinematic) rows permuted across events within direction.

    imc : dict of (C, E) RF image-change measures for these events.

    Returns
    -------
    out : dict per condition with per-cell R^2, unique R^2, p-values and the
        population mean against its null.
    """

    rng = np.random.default_rng(seed)
    out = {}
    for cond, lt in (('light', True), ('dark', False)):
        e = np.flatnonzero(gev['isolated'] & (gev['light'] == lt))
        T = gev['t_on'][e]
        r1, n1 = window_mean(rate, samp, dt, T, *SHIFT_WIN)
        r0, n0 = window_mean(rate, samp, dt, T, *BASE_WIN)
        y = r1 * n1 * dt
        y0 = r0 * n0 * dt
        K = np.stack([np.abs(gev[k][e]) if k != 'direction' else gev[k][e].astype(float) for k in kin_names], 1)
        good = np.isfinite(y).all(0) & np.isfinite(y0).all(0) & np.isfinite(K).all(1)
        good &= np.isfinite(imc['dI'][:, e]).all(0)
        e, y, y0, K = e[good], y[:, good], y0[:, good], K[good]
        E = len(e)
        if E < 40:
            continue
        cells = np.flatnonzero(np.nansum(y, 1) >= min_spikes) if counts else np.arange(rate.shape[0])
        Y = (2 * np.sqrt(np.clip(y[cells], 0, None) + 3.0 / 8) if counts else y[cells]).T
        B0 = (2 * np.sqrt(np.clip(y0[cells], 0, None) + 3.0 / 8) if counts else y0[cells]).T
        Vis = np.stack([imc['dI'][cells][:, e], imc['enter'][cells][:, e], imc['leave'][cells][:, e]], 2)
        folds = np.array_split(np.arange(E), 5)
        dirs = gev['direction'][e]

        def r2(i, *blocks):
            """ CV R^2 of cell i from its baseline plus the given blocks. """
            return _ridge_cv(np.column_stack((B0[:, i],) + blocks), Y[:, [i]], folds)[0]

        def strat_perm():
            """ Event permutation within direction. """
            perm = np.arange(E)
            for d in (-1, 1):
                ii = np.flatnonzero(dirs == d)
                perm[ii] = rng.permutation(ii)
            return perm

        nC = len(cells)
        r_full = np.array([r2(i, Vis[i], K) for i in range(nC)])
        r_nov = np.array([r2(i, K) for i in range(nC)])
        r_nok = np.array([r2(i, Vis[i]) for i in range(nC)])
        u_vis, u_kin = r_full - r_nov, r_full - r_nok
        n_null = N_SHUF // 4
        null_vis = np.zeros((n_null, nC))
        null_kin = np.zeros((n_null, nC))
        for k in range(n_null):
            pv = strat_perm()
            null_vis[k] = np.array([r2(i, Vis[i][pv], K) for i in range(nC)]) - r_nov
            # Direction stays put: it is the stratum, not a free kinematic.
            Kp = K[strat_perm()].copy()
            Kp[:, -1] = K[:, -1]
            null_kin[k] = np.array([r2(i, Vis[i], Kp) for i in range(nC)]) - r_nok
        p_vis = (1 + np.sum(null_vis >= u_vis, 0)) / (1 + len(null_vis))
        p_kin = (1 + np.sum(null_kin >= u_kin, 0)) / (1 + len(null_kin))
        # Covariate correlations for a cell on the population RF if there is one.
        same = [i for i in range(nC) if np.array_equal(Vis[i], Vis[0])]
        corr = np.corrcoef(np.c_[Vis[same[-1]], K].T)
        out[cond] = {'cells': cells, 'n_events': E, 'r2_full': r_full, 'unique_vis': u_vis,
                     'unique_kin': u_kin, 'p_vis': p_vis, 'p_kin': p_kin,
                     'pop_vis': float(np.nanmedian(u_vis)), 'pop_vis_null': np.nanmedian(null_vis, 1),
                     'pop_kin': float(np.nanmedian(u_kin)), 'pop_kin_null': np.nanmedian(null_kin, 1),
                     'cov_names': ['dI', 'enter', 'leave'] + list(kin_names), 'cov_corr': corr}
    return out


def omsi_lag_check(spk, dff, max_lag=3):
    """ Lag between OMSI spikes and the rise of dF/F, population mean.

    Correlates each cell's spike rate with the first difference of its dF/F
    at lags -max_lag..max_lag frames (positive = dF/F rises after the spike).
    A peak at 0 means a spike is assigned to the frame where the rise first
    shows, i.e. it happened during the preceding inter-sample interval.
    """

    d = np.diff(dff, axis=1)
    sp = spk[:, 1:]
    lags = np.arange(-max_lag, max_lag + 1)
    cc = np.full((spk.shape[0], len(lags)), np.nan)
    for i, L in enumerate(lags):
        if L >= 0:
            a, b = sp[:, :sp.shape[1] - L], d[:, L:]
        else:
            a, b = sp[:, -L:], d[:, :d.shape[1] + L]
        a = a - a.mean(1, keepdims=True)
        b = b - np.nanmean(b, 1, keepdims=True)
        cc[:, i] = np.nansum(a * b, 1) / np.sqrt(np.nansum(a ** 2, 1) * np.nansum(b ** 2, 1))
    return {'lags': lags, 'cc_mean': np.nanmean(cc, 0), 'peak_lag': int(lags[np.nanargmax(np.nanmean(cc, 0))])}


def onset_timing(rates, samp, dt, gev, hev, comp_pairs, seed=0):
    """ Population onset relative to gaze and head onset, with bootstrap CIs.

    Events need ISO_TIMING s free of other gaze shifts before them, since the
    pre-onset period is what is being read.  Population = spikes pooled over
    cells, as a fractional change from the pooled baseline (tau < -0.5 s):
    sum_c (PETH_c - b_c) / sum_c b_c.  Two cell sets: every cell, and cells
    enhanced by gaze shifts (p < 0.05, picked on the other parity of events,
    so a cell is never measured on the events that picked it).

    Returns
    -------
    out : dict keyed by (spike key, alignment, cell set) with tau, the
        population PETH, its bootstrap band, onset fit and pre-onset statistic.
    """

    rng = np.random.default_rng(seed)
    nb = int(round(2 * PETH_HALF / PETH_BIN))
    iso = np.flatnonzero(gev['gap'] > ISO_TIMING)
    parity = np.full(len(gev['t_on']), -1)
    parity[iso] = np.arange(len(iso)) % 2
    sets = {}
    for cond, lt in (('light', True), ('dark', False)):
        e = iso[gev['light'][iso] == lt]
        sets['gaze_' + cond] = (gev['t_on'][e], parity[e])
        eh = e[np.isfinite(gev['t_head_on'][e]) & (gev['t_head_on'][e] < gev['t_on'][e])]
        sets['head_' + cond] = (gev['t_head_on'][eh], parity[eh])
    for cond in ('light', 'dark'):
        if cond in comp_pairs and comp_pairs[cond]['n_pairs'] >= 10:
            pr = comp_pairs[cond]['pairs']
            sets['comp_' + cond] = (hev['t_on'][pr[:, 1]], np.full(len(pr), -1))
            sets['matched_shift_' + cond] = (hev['t_on'][pr[:, 0]], np.full(len(pr), -1))

    # Enhanced cells on OMSI, cross-validated by event parity.
    ref = rates['norm_spikes'] if 'norm_spikes' in rates else next(iter(rates.values()))
    T_sel = gev['t_on'][iso]
    r1, _ = window_mean(ref, samp, dt, T_sel, *MATCH_WIN)
    r0, _ = window_mean(ref, samp, dt, T_sel, *BASE_WIN)
    d = r1 - r0
    par = parity[iso]
    picks = []
    for k in (0, 1):
        with np.errstate(invalid='ignore'):
            tt = ttest_1samp(d[:, par == k], 0, axis=1, nan_policy='omit')
        picks.append((np.asarray(tt.pvalue) < 0.05) & (np.nanmean(d[:, par == k], 1) > 0))
    out = {'n_selected': [int(p.sum()) for p in picks], 'sets_n': {k: len(v[0]) for k, v in sets.items()}}
    C = ref.shape[0]

    for key, rate in rates.items():
        active = np.nanmean(np.abs(rate), 1) > 0
        for name, (T, Tpar) in sets.items():
            if len(T) < 20:
                continue
            S, tau = peth_samples(rate, samp, T)
            basew = tau < -0.5

            def pop(w, cellset):
                """ Pooled fractional change; enhanced set averages both parity picks. """
                if cellset == 'all':
                    groups = [(np.flatnonzero(active), w)]
                else:
                    groups = [(np.flatnonzero(picks[k] & active), w * (Tpar != k)) for k in (0, 1)]
                ys = []
                for cells, ww in groups:
                    if len(cells) < 3:
                        continue
                    P = peth_from_samples(S, nb, ww, cells)
                    b = np.nanmean(P[:, basew], 1)
                    ys.append(np.nansum(P - b[:, None], 0) / np.nansum(np.abs(b)))
                return np.nanmean(np.stack(ys), 0) if ys else np.full(nb, np.nan)

            for cellset in ('all', 'enhanced'):
                y = pop(np.ones(len(T)), cellset)
                if not np.isfinite(y).any():
                    continue
                boots = np.stack([pop(rng.multinomial(len(T), np.ones(len(T)) / len(T)).astype(float), cellset)
                                  for _ in range(N_BOOT // 3)])
                fit = ramp_onset_fit(tau, y, dt)
                bt0 = np.array([ramp_onset_fit(tau, bb, dt)['t0'] for bb in boots[:60]])
                pre = (tau > -dt) & (tau <= 0)
                pre_stat = float(np.nanmean(y[pre]) - np.nanmean(y[basew]))
                pre_boot = np.nanmean(boots[:, pre], 1) - np.nanmean(boots[:, basew], 1)
                post = (tau > dt / 2) & (tau <= 0.5)
                out[(key, name, cellset)] = {
                    'tau': tau, 'peth': y, 'band': np.nanpercentile(boots, [2.5, 97.5], 0),
                    'fit': fit, 't0_ci': np.nanpercentile(bt0, [2.5, 97.5]),
                    'pre': pre_stat, 'pre_ci': np.nanpercentile(pre_boot, [2.5, 97.5]),
                    'post': float(np.nanmean(y[post]) - np.nanmean(y[basew])),
                    'post_ci': np.nanpercentile(np.nanmean(boots[:, post], 1) - np.nanmean(boots[:, basew], 1), [2.5, 97.5]),
                    'n': len(T)}
    lag = gev['t_on'] - gev['t_head_on']
    out['head_lead_ms'] = 1000 * np.nanmedian(lag[np.isfinite(lag) & (lag > 0)])
    return out


def _cell_interval_means(t40, x40, samp, dt):
    """ Mean of a 40 Hz trace over each cell's sample intervals (s - dt, s].

    x40 : (N,) shared or (N, R) with `rows` mapping cells to columns.
    Returns (C, F); NaN where any 40 Hz sample in the interval is NaN.
    """

    fin = np.isfinite(x40)
    cs = np.concatenate([[0.0], np.cumsum(np.where(fin, x40, 0.0))])
    cn = np.concatenate([[0], np.cumsum(fin)])
    lo = np.searchsorted(t40, samp - dt)
    hi = np.searchsorted(t40, samp)
    cnt = cn[hi] - cn[lo]
    full = (hi - lo)
    with np.errstate(invalid='ignore', divide='ignore'):
        m = (cs[hi] - cs[lo]) / cnt
    m[(cnt < full) | (full == 0)] = np.nan
    return m


def direction_preference(rate, samp, dt, s, v, gev, u_rows, row_of, seed=0):
    """ Gaze-shift vs visual-motion direction preference and their correlation.

    Gaze shift: d' between baseline-subtracted responses to nasal and temporal
    shifts (SHIFT_WIN after gaze onset); positive = prefers nasal, which moves
    the image temporally.  Miura & Scanziani's raw index (R_n - R_t)/(R_n + R_t)
    is kept alongside.  Visual motion: 2P samples outside gaze shifts
    (EXCL_PRE before to EXCL_POST after every episode), horizontal image
    velocity in the cell's RF averaged over each sample interval; temporal (u >
    U0) vs nasal (u < -U0) image motion compared inside head-velocity strata
    and averaged with harmonic-mean weights; positive = prefers temporal image
    motion.  Light is the visual measurement; dark, where the same image motion
    is computed but nothing is visible, is its non-visual control.

    Returns
    -------
    out : dict with per-cell d' (and raw indices), permutation p-values,
        split-half reliabilities, and correlations with ceilings.
    """

    rng = np.random.default_rng(seed)
    C = rate.shape[0]
    t40 = s['t']
    out = {}

    # Gaze-shift direction preference.
    for cond, lt in (('light', True), ('dark', False)):
        e = np.flatnonzero(gev['isolated'] & (gev['light'] == lt))
        T = gev['t_on'][e]
        r1, _ = window_mean(rate, samp, dt, T, *SHIFT_WIN)
        r0, _ = window_mean(rate, samp, dt, T, *BASE_WIN)
        ok = np.isfinite(r1).all(0) & np.isfinite(r0).all(0)
        r1, r0, dirs = r1[:, ok], r0[:, ok], gev['direction'][e][ok]
        d = r1 - r0

        def dprime(dd, lab):
            a, b = dd[:, lab == 1], dd[:, lab == -1]
            sd = np.sqrt(0.5 * (a.var(1) + b.var(1)))
            with np.errstate(invalid='ignore', divide='ignore'):
                return (a.mean(1) - b.mean(1)) / np.where(sd > 0, sd, np.nan)

        dp = dprime(d, dirs)
        null = np.stack([dprime(d, rng.permutation(dirs)) for _ in range(N_PERM // 4)])
        p = (1 + np.sum(np.abs(null) >= np.abs(dp), 0)) / (1 + len(null))
        rn, rt = r1[:, dirs == 1].mean(1), r1[:, dirs == -1].mean(1)
        with np.errstate(invalid='ignore', divide='ignore'):
            raw = (rn - rt) / (rn + rt)
        par = np.arange(len(dirs)) % 2
        dp_h = [dprime(d[:, par == k], dirs[par == k]) for k in (0, 1)]
        out['shift_' + cond] = {'dprime': dp, 'p': p, 'raw_index': raw, 'halves': dp_h,
                                'n_nasal': int((dirs == 1).sum()), 'n_temporal': int((dirs == -1).sum())}

    # Visual motion direction preference outside gaze shifts.
    excl = np.zeros(len(t40), bool)
    for a, b in zip(gev['ep_start'], gev['ep_end']):
        excl[max(0, a - int(EXCL_PRE / na.DT)):min(len(t40), b + int(EXCL_POST / na.DT))] = True
    head = v['head'].copy()
    U = np.full((C, samp.shape[1]), np.nan)
    H = np.full((C, samp.shape[1]), np.nan)
    Lt = np.full((C, samp.shape[1]), np.nan)
    ex = np.full((C, samp.shape[1]), np.nan)
    for c in range(C):
        U[c] = _cell_interval_means(t40, u_rows[:, row_of[c]], samp[c], dt)
        H[c] = _cell_interval_means(t40, head, samp[c], dt)
        Lt[c] = _cell_interval_means(t40, s['light'].astype(float), samp[c], dt)
        ex[c] = _cell_interval_means(t40, excl.astype(float), samp[c], dt)
    hs = np.digitize(H, HEAD_STRATA) - 1
    blk = np.floor((samp - samp[:, :1]) / 60.0).astype(int)
    n_str = len(HEAD_STRATA) - 1

    def codes(u, hsx, use):
        """ stratum * 2 + (0 nasal, 1 temporal) for usable samples, else -1. """
        cls = np.where(u > U0, 1, np.where(u < -U0, 0, -1))
        return np.where(use & (cls >= 0) & (hsx >= 0) & (hsx < n_str), hsx * 2 + cls, -1)

    def vpref(r, code):
        """ Stratified d' of temporal vs nasal image motion for one cell. """
        m = code >= 0
        cnt = np.bincount(code[m], minlength=2 * n_str).reshape(n_str, 2)
        s1 = np.bincount(code[m], weights=r[m], minlength=2 * n_str).reshape(n_str, 2)
        s2 = np.bincount(code[m], weights=r[m] ** 2, minlength=2 * n_str).reshape(n_str, 2)
        ok = (cnt >= MIN_STRATUM).all(1)
        if not ok.any():
            return np.nan, 0.0
        c_, a_, b_ = cnt[ok], s1[ok], s2[ok]
        mu = a_ / c_
        var = b_ / c_ - mu ** 2
        w = 2.0 / (1.0 / c_[:, 0] + 1.0 / c_[:, 1])
        diff = np.sum(w * (mu[:, 1] - mu[:, 0])) / w.sum()
        sd = np.sqrt(np.sum(w * 0.5 * (var[:, 0] + var[:, 1])) / w.sum())
        return (diff / sd if sd > 0 else np.nan), float(w.sum())

    for cond, lt in (('light', 1.0), ('dark', 0.0)):
        dp = np.full(C, np.nan)
        wt = np.zeros(C)
        halves = np.full((2, C), np.nan)
        pvals = np.full(C, np.nan)
        for c in range(C):
            use = (Lt[c] == lt) & (ex[c] == 0) & np.isfinite(U[c]) & np.isfinite(H[c]) & np.isfinite(rate[c])
            code = codes(U[c], hs[c], use)
            dp[c], wt[c] = vpref(rate[c], code)
            for k in (0, 1):
                halves[k, c], _ = vpref(rate[c], np.where(blk[c] % 2 == k, code, -1))
            if np.isfinite(dp[c]):
                # Circular shifts of at least 60 s keep each trace's own
                # autocorrelation and break the rate-behaviour alignment.
                n = len(rate[c])
                sh = rng.integers(int(60 / dt), n - int(60 / dt), N_PERM // 10)
                r0 = np.nan_to_num(rate[c])
                nl = np.array([vpref(np.roll(r0, k), code)[0] for k in sh])
                pvals[c] = (1 + np.sum(np.abs(nl) >= abs(dp[c]))) / (1 + np.sum(np.isfinite(nl)))
        out['visual_' + cond] = {'dprime': dp, 'weight': wt, 'halves': halves, 'p': pvals}

    def rel(h):
        a, b = h[0], h[1]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 10:
            return np.nan
        r = spearmanr(a[m], b[m]).correlation
        return 2 * r / (1 + r) if r > -1 else np.nan

    for k in ('shift_light', 'shift_dark', 'visual_light', 'visual_dark'):
        out[k]['reliability'] = rel(np.stack(out[k]['halves']))

    corr = {}
    for a, b in (('shift_dark', 'visual_light'), ('shift_light', 'visual_light'),
                 ('shift_light', 'shift_dark'), ('visual_light', 'visual_dark'),
                 ('shift_dark', 'visual_dark')):
        x, y = out[a]['dprime'], out[b]['dprime']
        for subset in ('all', 'selective'):
            m = np.isfinite(x) & np.isfinite(y)
            if subset == 'selective':
                m &= (out[a]['p'] < 0.05) & (out[b]['p'] < 0.05)
            if m.sum() < 5:
                corr[(a, b, subset)] = {'n': int(m.sum()), 'r': np.nan, 'rho': np.nan,
                                        'ci': np.array([np.nan, np.nan]), 'p': np.nan}
                continue
            r = float(np.corrcoef(x[m], y[m])[0, 1])
            sr = spearmanr(x[m], y[m])
            ii = np.flatnonzero(m)
            bs = []
            for _ in range(N_BOOT):
                jj = rng.choice(ii, len(ii))
                bs.append(spearmanr(x[jj], y[jj]).correlation)
            ra, rb = out[a]['reliability'], out[b]['reliability']
            ceil = np.sqrt(ra * rb) if (np.isfinite(ra) and np.isfinite(rb) and ra > 0 and rb > 0) else np.nan
            corr[(a, b, subset)] = {'n': int(m.sum()), 'r': r, 'rho': float(sr.correlation),
                                    'p': float(sr.pvalue), 'ci': np.nanpercentile(bs, [2.5, 97.5]),
                                    'ceiling': ceil}
    out['corr'] = corr
    return out


def run_probe(h5_path, npz_path, sn_dir=None, spike_keys=SPIKE_KEYS, mirror=False, seed=0):
    """ Every test for one session.

    Returns
    -------
    R_ : dict of everything the PDF and the HDF5 need.
    """

    t0 = time.time()
    print('Loading session...')
    s = na.load_session(h5_path, npz_path)
    v = na.eye_head_velocity(s)
    lab = na.classify_gated(v)
    print('Optic flow...')
    fl = na.optic_flow(s)
    twopT, dt, off = scan_offsets(h5_path)
    rates = load_rates(h5_path, spike_keys)
    n_cells = next(iter(rates.values())).shape[0]
    F = min(len(twopT), next(iter(rates.values())).shape[1])
    rates = {k: r[:, :F] for k, r in rates.items()}
    samp = twopT[None, :F] + off[:, None]
    # OMSI timestamps a spike one frame before the dF/F jump that OASIS and
    # suite2p put it on.  This is an offset in OMSI's output, not biology, and
    # left in it manufactures pre-onset activity, so every test uses OMSI moved
    # onto the jump; the timing page also shows it as stored for comparison.
    lag_check, omsi_asis = None, None
    if 'norm_spikes' in rates and 'norm_dFF' in rates:
        lag_check = omsi_lag_check(rates['norm_spikes'], rates['norm_dFF'])
        L = lag_check['peak_lag']
        if L > 0:
            omsi_asis = rates['norm_spikes']
            shifted = np.zeros_like(omsi_asis)
            shifted[:, L:] = omsi_asis[:, :-L]
            rates['norm_spikes'] = shifted
            print('  OMSI spikes sit {} frame(s) before the dF/F jump (OASIS: {}); OMSI moved {} frame(s) '
                  'later for every test.'.format(L, omsi_lag_check(rates['oasis_spks'], rates['norm_dFF'])['peak_lag']
                                                 if 'oasis_spks' in rates else 'n/a', L))
    print('  2P: {} cells, {} frames, frame {:.0f} ms; scan offsets {:.0f}-{:.0f} ms.'.format(
        n_cells, F, 1000 * dt, 1000 * off.min(), 1000 * off.max()))

    print('Events...')
    gev = gaze_shift_events(s, v, lab)
    hev = head_events(s, v, lab, gaze_onsets=gev['t_on'])
    for cond, lt in (('light', True), ('dark', False)):
        print('  {}: {} gaze shifts ({} isolated); head movements: {} with a shift, {} compensatory '
              '(isolated).'.format(cond, int((gev['light'] == lt).sum()),
                                   int((gev['isolated'] & (gev['light'] == lt)).sum()),
                                   int(((hev['kind'] == 1) & hev['isolated'] & (hev['light'] == lt)).sum()),
                                   int(((hev['kind'] == 0) & hev['isolated'] & (hev['light'] == lt)).sum())))

    print('Receptive fields...')
    sn_dir = sn_dir or os.path.join(os.path.dirname(os.path.dirname(h5_path)), 'sn1')
    rf = load_rfs(sn_dir, n_cells, mirror=mirror)
    rf_m = load_rfs(sn_dir, n_cells, mirror=not mirror)
    print('  {} cells with a sparse-noise RF fit; population RF az {:+.0f}, el {:+.0f}, sigma {:.0f} deg '
          '(+{:.0f} deg mapping uncertainty).'.format(rf['n_fit'], rf['pop_az'], rf['pop_el'],
                                                      rf['pop_sig'], MAP_SIGMA))
    # Unique RF rows: one per fitted cell plus the population RF.
    keys = np.stack([rf['az'], rf['el'], rf['sig']], 1)
    uk, row_of = np.unique(keys, axis=0, return_inverse=True)
    row_of = row_of.ravel()
    dirs, gaz, gel = _grid_dirs()
    W = rf_weights(gaz, gel, uk[:, 0], uk[:, 1], uk[:, 2])
    keys_m = np.stack([rf_m['az'], rf_m['el'], rf_m['sig']], 1)
    W_m = rf_weights(gaz, gel, keys_m[:, 0], keys_m[:, 1], keys_m[:, 2])
    W_field = np.cos(np.radians(gel))[None] / np.cos(np.radians(gel)).sum()
    Wf = rf_weights(fl['grid_az'], fl['grid_el'], uk[:, 0], uk[:, 1], uk[:, 2])
    u_rows = np.einsum('np,rp->nr', np.nan_to_num(fl['field'][:, :, 0], nan=np.nan), Wf)
    print('  RF image velocity vs gaze velocity: r = {:.2f} (positive = a nasal gaze movement '
          'drifts the image temporally across the RF).'.format(
              float(np.corrcoef(u_rows[np.isfinite(u_rows[:, 0]) & np.isfinite(v['gaze']), 0],
                                v['gaze'][np.isfinite(u_rows[:, 0]) & np.isfinite(v['gaze'])])[0, 1])))

    print('Rendering RF patches at event frames...')
    origin, rot, _ = na.pose_matrices(s)
    n40 = len(s['t'])
    g_pre = np.clip(gev['on_idx'] - 2, 0, n40 - 1)
    g_post = np.clip(gev['ep_end'] + 2, 0, n40 - 1)
    h_pre = np.clip(hev['on_idx'] - 2, 0, n40 - 1)
    h_post = np.clip(hev['off_idx'] + 2, 0, n40 - 1)
    allidx = np.unique(np.r_[g_pre, g_post, h_pre, h_post])
    labs, lok = render_labels(s, origin, rot, allidx, dirs)
    pos = {i: k for k, i in enumerate(allidx)}

    def imc_for(pre, post, Wx):
        a = labs[[pos[i] for i in pre]]
        b = labs[[pos[i] for i in post]]
        ok = lok[[pos[i] for i in pre]] & lok[[pos[i] for i in post]]
        m = image_change(a, b, Wx)
        for k in m:
            m[k][:, ~ok] = np.nan
        return m

    imc_g = imc_for(g_pre, g_post, W)
    imc_g_m = imc_for(g_pre, g_post, W_m)
    imc_g_f = imc_for(g_pre, g_post, W_field)
    imc_h = imc_for(h_pre, h_post, W)
    pop_row = int(row_of[np.flatnonzero(~rf['has_fit'])[0]]) if (~rf['has_fit']).any() else 0
    # Per-cell views (C, E).
    cellify = lambda m, rows: {k: x[rows] for k, x in m.items()}
    imc_g_cells = cellify(imc_g, row_of)
    imc_g_m_cells = imc_g_m
    imc_g_f_cells = {k: np.repeat(x, n_cells, 0) for k, x in imc_g_f.items()}
    imc_h_pop = {k: x[pop_row] for k, x in imc_h.items()}

    R_ = {'session': os.path.basename(h5_path).replace('_preproc.h5', ''), 'h5_path': h5_path,
          'npz_path': npz_path, 'date': time.strftime('%Y-%m-%d'), 'n_cells': n_cells, 'dt': dt,
          'scan_offset': off, 'gev': gev, 'hev': hev, 'rf': rf, 'eye_scale_px': s.get('eye_scale_px', np.nan),
          'vor_light': v['vor_slope_light'], 'vor_dark': v['vor_slope_dark'],
          'imc_g_pop': {k: x[pop_row] for k, x in imc_g.items()}, 'imc_h_pop': imc_h_pop,
          'spike_keys': list(rates.keys()) + (['omsi_asis'] if omsi_asis is not None else [])}

    print('Compensatory probe...')
    R_['comp'] = {}
    for key, rate in rates.items():
        R_['comp'][key] = compensatory_probe(rate, samp, dt, hev, imc_h_pop, seed=seed)
        for cond, c in R_['comp'][key].items():
            print('  [{}] {}: {} matched pairs; pooled response shift {:+.3f} [{:+.3f}, {:+.3f}], comp {:+.3f} '
                  '[{:+.3f}, {:+.3f}], ratio {:.2f} [{:.2f}, {:.2f}]; {} responsive cells.'.format(
                      KEY_LABEL[key], cond, c['n_pairs'], c['pooled_shift'], *c['pooled_shift_ci'],
                      c['pooled_comp'], *c['pooled_comp_ci'], c['pooled_ratio'], *c['pooled_ratio_ci'],
                      int(c['selected'].sum())))

    print('Regression within gaze shifts...')
    R_['reg'] = {}
    for key in [k for k in ('norm_spikes', 'oasis_spks') if k in rates]:
        counts = key == 'norm_spikes'
        R_['reg'][key] = shift_regression(rates[key], samp, dt, gev, imc_g_cells, counts=counts, seed=seed)
        for cond, r in R_['reg'][key].items():
            print('  [{}] {}: {} shifts, {} cells; mean unique R2 visual {:.4f} (null {:.4f}), kinematic '
                  '{:.4f} (null {:.4f}); cells p<0.05 visual {}, kinematic {}.'.format(
                      KEY_LABEL[key], cond, r['n_events'], len(r['cells']), r['pop_vis'],
                      np.mean(r['pop_vis_null']), r['pop_kin'], np.mean(r['pop_kin_null']),
                      int((r['p_vis'] < 0.05).sum()), int((r['p_kin'] < 0.05).sum())))
    if 'norm_spikes' in rates:
        R_['reg_mirror'] = shift_regression(rates['norm_spikes'], samp, dt, gev, imc_g_m_cells, seed=seed + 1)
        R_['reg_field'] = shift_regression(rates['norm_spikes'], samp, dt, gev, imc_g_f_cells, seed=seed + 2)
        for nm, rr in (('mirrored RF', R_['reg_mirror']), ('whole field', R_['reg_field'])):
            if 'light' in rr:
                print('  [OMSI, {}] light: mean unique R2 visual {:.4f} (null {:.4f}).'.format(
                    nm, rr['light']['pop_vis'], np.mean(rr['light']['pop_vis_null'])))

    print('Onset timing...')
    comp_ref = R_['comp'].get('norm_spikes', next(iter(R_['comp'].values())))
    t_rates = dict(rates)
    if omsi_asis is not None:
        t_rates['omsi_asis'] = omsi_asis
    R_['timing'] = onset_timing(t_rates, samp, dt, gev, hev, comp_ref, seed=seed)
    R_['lag_check'] = lag_check
    print('  Enhanced cells per parity half: {}.'.format(R_['timing']['n_selected']))
    for (key, name, cs), r in [(k, x) for k, x in R_['timing'].items() if isinstance(k, tuple)]:
        print('  [{}] {} {} cells (n={}): response {:+.3f} [{:+.3f}, {:+.3f}]; ramp start {:+.0f} ms '
              '[{:+.0f}, {:+.0f}]; pre-onset (-133..0 ms) {:+.3f} [{:+.3f}, {:+.3f}].'.format(
                  KEY_LABEL[key], name, cs, r['n'], r['post'], *r['post_ci'], 1000 * r['fit']['t0'],
                  *(1000 * r['t0_ci']), r['pre'], *r['pre_ci']))

    print('Direction preferences...')
    R_['dir'] = {}
    for key in [k for k in ('norm_spikes', 'oasis_spks') if k in rates]:
        R_['dir'][key] = direction_preference(rates[key], samp, dt, s, v, gev, u_rows, row_of, seed=seed)
        D = R_['dir'][key]
        print('  [{}] reliability: shift light {:.2f}, shift dark {:.2f}, visual light {:.2f}, '
              'visual dark {:.2f}.'.format(KEY_LABEL[key], D['shift_light']['reliability'],
                                           D['shift_dark']['reliability'], D['visual_light']['reliability'],
                                           D['visual_dark']['reliability']))
        for (a, b, sub), c in D['corr'].items():
            print('    {} vs {} [{}]: n={} rho={:+.2f} [{:+.2f}, {:+.2f}] (ceiling {:.2f}).'.format(
                a, b, sub, c['n'], c['rho'], *c['ci'], c.get('ceiling', np.nan)))
    _A = na.load_area_labels(h5_path, n_cells)
    R_['areas'] = _A['area_id']
    R_['area_names'] = ','.join(np.asarray(_A['area']).astype(str))
    R_['runtime_s'] = time.time() - t0
    return R_


def save_probe(path, R_):
    """ One-session HDF5 with the per-cell results, for pooling across areas. """

    out = {'session': R_['session'], 'n_cells': R_['n_cells'], 'area_id': R_['areas'],
           'area': R_.get('area_names', ''),
           'scan_offset_s': R_['scan_offset'], 'eye_scale_px': R_['eye_scale_px'],
           'rf': {k: R_['rf'][k] for k in ('az', 'el', 'sig', 'has_fit')},
           'gaze_events': {k: np.asarray(x, float) for k, x in R_['gev'].items()},
           'head_events': {k: np.asarray(x, float) for k, x in R_['hev'].items()}}
    # Figure-only extras, so the figure notebook reads files instead of rerunning.
    out['meta'] = {'date': R_['date'], 'h5_path': R_['h5_path'], 'npz_path': R_['npz_path'], 'dt': R_['dt'],
                   'spike_keys': ','.join(R_['spike_keys']), 'vor_light': R_['vor_light'],
                   'vor_dark': R_['vor_dark'], 'runtime_s': R_.get('runtime_s', np.nan),
                   'rf_n_fit': R_['rf']['n_fit'], 'rf_pop_az': R_['rf']['pop_az'], 'rf_pop_el': R_['rf']['pop_el'],
                   'rf_pop_sig': R_['rf']['pop_sig']}
    out['const'] = {'HEAD_RUN': HEAD_RUN, 'HEAD_ON': HEAD_ON, 'GAZE_ON': GAZE_ON, 'ISO_S': ISO_S,
                    'ISO_TIMING': ISO_TIMING, 'BASE_WIN': np.array(BASE_WIN), 'MATCH_WIN': np.array(MATCH_WIN),
                    'SHIFT_WIN': np.array(SHIFT_WIN), 'CALIPER': CALIPER, 'U0': U0, 'HEAD_STRATA': HEAD_STRATA,
                    'EXCL_PRE': EXCL_PRE, 'EXCL_POST': EXCL_POST, 'MIN_STRATUM': MIN_STRATUM,
                    'MON_SHAPE': np.array(MON_SHAPE), 'MON_F_PX': MON_F_PX, 'MAP_SIGMA': MAP_SIGMA,
                    'GRID_HALF': GRID_HALF, 'GRID_STEP': GRID_STEP, 'N_SHUF': N_SHUF, 'N_BOOT': N_BOOT,
                    'N_PERM': N_PERM, 'SHIFTED_GAZE': na.SHIFTED_GAZE, 'STILL_GAZE': na.STILL_GAZE}
    for key, comp in R_['comp'].items():
        for cond, c in comp.items():
            out['comp_{}_{}'.format(key, cond)] = {k: c[k] for k in (
                'pairs', 'resp_shift', 'resp_comp', 'p_shift', 'q_shift', 'selected', 'ratio_cell')}
            out['comp_{}_{}'.format(key, cond)].update({k: c[k] for k in (
                'pop_ratio', 'pop_ratio_ci', 'pooled_shift', 'pooled_shift_ci', 'pooled_comp', 'pooled_comp_ci',
                'pooled_ratio', 'pooled_ratio_ci', 'n_pairs')})
            out['comp_{}_{}'.format(key, cond)]['balance'] = {
                k: {'shift': np.asarray(a, float), 'comp': np.asarray(b, float)} for k, (a, b) in c['balance'].items()}
            for k in ('dI_shift', 'dI_comp'):
                if c[k] is not None:
                    out['comp_{}_{}'.format(key, cond)][k] = np.asarray(c[k], float)
    for key, reg in R_['reg'].items():
        for cond, r in reg.items():
            out['reg_{}_{}'.format(key, cond)] = {k: r[k] for k in (
                'cells', 'r2_full', 'unique_vis', 'unique_kin', 'p_vis', 'p_kin', 'pop_vis_null',
                'pop_kin_null', 'cov_corr', 'n_events', 'pop_vis', 'pop_kin')}
            out['reg_{}_{}'.format(key, cond)]['cov_names'] = ','.join(r['cov_names'])
    for nm in ('reg_mirror', 'reg_field'):
        for cond, r in R_.get(nm, {}).items():
            out['{}_{}'.format(nm, cond)] = {k: r[k] for k in ('pop_vis', 'pop_vis_null', 'pop_kin', 'pop_kin_null',
                                                               'n_events')}
    for (key, name, cs), r in [(k, x) for k, x in R_['timing'].items() if isinstance(k, tuple)]:
        out['peth_{}_{}_{}'.format(key, name, cs)] = {
            'tau': r['tau'], 'peth': r['peth'], 'band': r['band'], 't0': r['fit']['t0'], 't0_ci': r['t0_ci'],
            'pre': r['pre'], 'pre_ci': r['pre_ci'], 'post': r['post'], 'post_ci': r['post_ci'], 'n': r['n'],
            'fit': {k: v for k, v in r['fit'].items()}}
    out['timing'] = {'head_lead_ms': R_['timing'].get('head_lead_ms', np.nan),
                     'n_selected': np.asarray(R_['timing']['n_selected']),
                     'sets_n': {k: int(v) for k, v in R_['timing']['sets_n'].items()}}
    if R_.get('lag_check') is not None:
        out['lag_check'] = dict(R_['lag_check'])
    for key, D in R_['dir'].items():
        for k in ('shift_light', 'shift_dark', 'visual_light', 'visual_dark'):
            out['dir_{}_{}'.format(key, k)] = {kk: np.asarray(D[k][kk], float) for kk in ('dprime', 'p')}
            out['dir_{}_{}'.format(key, k)]['reliability'] = D[k]['reliability']
            out['dir_{}_{}'.format(key, k)]['halves'] = np.asarray(D[k]['halves'], float)
            for kk in ('raw_index', 'weight', 'n_nasal', 'n_temporal'):
                if kk in D[k]:
                    out['dir_{}_{}'.format(key, k)][kk] = np.asarray(D[k][kk], float)
        out['dircorr_{}'.format(key)] = {'{}__{}__{}'.format(a, b, sub): {kk: np.asarray(vv, float) for kk, vv in c.items()}
                                         for (a, b, sub), c in D['corr'].items()}
    write_h5(path, out)


def _pc(ax, x, y, c, lab=None, s=10):
    """ Scatter with a small surface ring. """

    ax.scatter(x, y, s=s, c=c, edgecolors='white', linewidths=0.4, label=lab, zorder=3)


def _page_summary(pdf, R_, verdicts):
    """ Headline numbers and verdicts. """

    fig = na._new_page('Is the gaze-shift response visual?  {}'.format(R_['session']),
                       'Eye scale {:.0f} px; slow-phase gain light {:.2f}, dark {:.2f}.  Spike measure for the '
                       'headline: OMSI.  {} cells, {} with a sparse-noise RF.'.format(
                           R_['eye_scale_px'], abs(R_['vor_light']), abs(R_['vor_dark']), R_['n_cells'],
                           R_['rf']['n_fit']))
    y = 0.86
    for title, body in verdicts:
        fig.text(0.05, y, title, fontsize=11, weight='bold', color=INK, va='top')
        y -= 0.03
        for line in textwrap.wrap(body, 150):
            fig.text(0.07, y, line, fontsize=9, color=INK2, va='top')
            y -= 0.022
        y -= 0.015
    pdf.savefig(fig)
    plt.close(fig)


def _page_comp(pdf, R_):
    """ Matching balance, residual image change and the response ratio. """

    comp = R_['comp'].get('norm_spikes', {})
    fig = na._new_page('1. Compensatory movements as the non-visual probe',
                       'Head movements with a gaze shift matched 1:1 to compensatory ones on peak head velocity, '
                       'amplitude and time since the previous movement, exactly on direction and light.  Responses: mean rate 0-500 ms after head onset '
                       'minus -800..-200 ms.  Cells picked on half the pairs, ratio read on the other half.')
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.97, top=0.86, bottom=0.08, hspace=0.45, wspace=0.3)
    for j, cond in enumerate(('light', 'dark')):
        if cond not in comp:
            continue
        c = comp[cond]
        ax = fig.add_subplot(gs[j, 0])
        for nm, col, k in (('gaze shift', COND_COLOR['shift'], 0), ('compensatory', COND_COLOR['comp'], 1)):
            _pc(ax, c['balance']['head_peak'][k], c['balance']['head_amp'][k], col, nm, s=8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('peak head velocity (deg/s)', fontsize=8)
        ax.set_ylabel('head amplitude (deg)', fontsize=8)
        ax.set_title('{}: {} matched pairs'.format(cond, c['n_pairs']), fontsize=9, color=INK)
        ax.legend(fontsize=7, frameon=False)
        na._style(ax)
        ax = fig.add_subplot(gs[j, 1])
        gd = [np.abs(c['balance']['gaze_disp'][0]), np.abs(c['balance']['gaze_disp'][1])]
        bins = np.linspace(0, np.nanpercentile(gd[0], 98), 30)
        ax.hist(gd[0], bins, color=COND_COLOR['shift'], alpha=0.7, label='gaze shift')
        ax.hist(gd[1], bins, color=COND_COLOR['comp'], alpha=0.7, label='compensatory')
        ax.set_xlabel('|gaze displacement| over the movement (deg)', fontsize=8)
        ax.set_ylabel('events', fontsize=8)
        dI = c['dI_shift'], c['dI_comp']
        ax.set_title('median gaze disp. {:.1f} vs {:.1f} deg; RF lum. change {:.3f} vs {:.3f}'.format(
            np.nanmedian(gd[0]), np.nanmedian(gd[1]), np.nanmedian(dI[0]), np.nanmedian(dI[1])), fontsize=8, color=INK)
        ax.legend(fontsize=7, frameon=False)
        na._style(ax)
        ax = fig.add_subplot(gs[j, 2])
        sel = c['selected']
        lim = np.nanpercentile(np.abs(np.r_[c['resp_shift'], c['resp_comp']]), 99)
        _pc(ax, c['resp_shift'][~sel], c['resp_comp'][~sel], '#c3c2bd', 'not selected', s=6)
        _pc(ax, c['resp_shift'][sel], c['resp_comp'][sel], COND_COLOR[cond], 'responsive', s=12)
        ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=0.8, ls='--')
        ax.axhline(0, color=GRID, lw=0.8)
        ax.axvline(0, color=GRID, lw=0.8)
        ax.set_xlabel('gaze-shift response (Hz)', fontsize=8)
        ax.set_ylabel('compensatory response (Hz)', fontsize=8)
        ax.set_title('pooled: shift {:+.2f}, comp {:+.2f}, ratio {:.2f} [{:.2f}, {:.2f}]'.format(
            c['pooled_shift'], c['pooled_comp'], c['pooled_ratio'], *c['pooled_ratio_ci']), fontsize=8, color=INK)
        ax.legend(fontsize=7, frameon=False)
        na._style(ax)
    pdf.savefig(fig)
    plt.close(fig)


def _page_regression(pdf, R_):
    """ Unique visual vs kinematic variance within gaze shifts. """

    reg = R_['reg'].get('norm_spikes', {})
    fig = na._new_page('2. Regression within gaze shifts (OMSI)',
                       'Response = spike count 0-400 ms after gaze onset (Anscombe).  Visual: RF luminance change, '
                       'pillar entering, pillar leaving.  Kinematic: head peak and amplitude, eye amplitude and peak, '
                       'direction.  Unique R^2 = CV R^2(full) - CV R^2(without the group).  Dark is the control: '
                       'image-change numbers exist but nothing is visible.')
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.97, top=0.86, bottom=0.08, hspace=0.45, wspace=0.32)
    for j, cond in enumerate(('light', 'dark')):
        if cond not in reg:
            continue
        r = reg[cond]
        ax = fig.add_subplot(gs[j, 0])
        sv, sk = r['p_vis'] < 0.05, r['p_kin'] < 0.05
        _pc(ax, r['unique_kin'], r['unique_vis'], '#c3c2bd', 'neither', s=6)
        _pc(ax, r['unique_kin'][sk & ~sv], r['unique_vis'][sk & ~sv], '#eb6834', 'kinematic p<0.05', s=12)
        _pc(ax, r['unique_kin'][sv & ~sk], r['unique_vis'][sv & ~sk], '#2a78d6', 'visual p<0.05', s=12)
        _pc(ax, r['unique_kin'][sv & sk], r['unique_vis'][sv & sk], '#1baf7a', 'both', s=12)
        ax.axhline(0, color=GRID, lw=0.8)
        ax.axvline(0, color=GRID, lw=0.8)
        ax.set_xlabel('unique R^2, kinematic', fontsize=8)
        ax.set_ylabel('unique R^2, visual', fontsize=8)
        ax.set_title('{}: {} shifts, {} cells'.format(cond, r['n_events'], len(r['cells'])), fontsize=9, color=INK)
        ax.legend(fontsize=7, frameon=False)
        na._style(ax)
        ax = fig.add_subplot(gs[j, 1])
        for k, (nm, obs, nul, col) in enumerate((('visual', r['pop_vis'], r['pop_vis_null'], '#2a78d6'),
                                                 ('kinematic', r['pop_kin'], r['pop_kin_null'], '#eb6834'))):
            ax.scatter(np.full(len(nul), k) + np.random.default_rng(k).uniform(-0.15, 0.15, len(nul)), nul,
                       s=6, color='#c3c2bd', zorder=2)
            ax.scatter([k], [obs], s=40, color=col, zorder=3, edgecolors='white')
            pv = (1 + np.sum(nul >= obs)) / (1 + len(nul))
            ax.text(k, obs, '  p={:.3g}'.format(pv), fontsize=7, color=INK2, va='center')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['visual', 'kinematic'], fontsize=8)
        ax.set_ylabel('median unique R^2 over cells', fontsize=8)
        ax.set_title('population vs permutation null', fontsize=9, color=INK)
        na._style(ax)
        ax = fig.add_subplot(gs[j, 2])
        im = ax.imshow(r['cov_corr'], vmin=-1, vmax=1, cmap='RdBu_r')
        ax.set_xticks(range(len(r['cov_names'])))
        ax.set_yticks(range(len(r['cov_names'])))
        ax.set_xticklabels(r['cov_names'], fontsize=6, rotation=60)
        ax.set_yticklabels(r['cov_names'], fontsize=6)
        ax.set_title('covariate correlation (one cell)', fontsize=9, color=INK)
        fig.colorbar(im, ax=ax, fraction=0.04)
    extra = []
    for nm, rr in (('mirrored RF', R_.get('reg_mirror', {})), ('whole field', R_.get('reg_field', {}))):
        if 'light' in rr:
            extra.append('{}: visual {:.4f} (null {:.4f})'.format(nm, rr['light']['pop_vis'],
                                                                  np.mean(rr['light']['pop_vis_null'])))
    if extra:
        fig.text(0.06, 0.03, 'RF placement sensitivity, light: ' + '; '.join(extra), fontsize=8, color=INK2)
    pdf.savefig(fig)
    plt.close(fig)


def _page_timing(pdf, R_):
    """ Population PETHs at sub-frame resolution with onset fits. """

    T = R_['timing']
    fig = na._new_page('3. Onset timing',
                       'Sub-frame PETH: each sample placed at its own scan-line time; the bin at tau holds spikes '
                       'from (tau - 133 ms, tau], so tau <= 0 is purely pre-onset.  Cells enhanced by gaze shifts, '
                       'pooled over all cells (dotted: enhanced cells, picked on alternate events).  Fractional change of the pooled rate from baseline; band = bootstrap 95%.  Head '
                       'leads gaze onset by a median {:.0f} ms in head-involved shifts.'.format(T.get('head_lead_ms', np.nan)))
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.97, top=0.85, bottom=0.08, hspace=0.45, wspace=0.3)
    panels = (('gaze_dark', 'dark shifts, gaze onset'), ('gaze_light', 'light shifts, gaze onset'),
              ('head_dark', 'dark shifts, head onset'), ('head_light', 'light shifts, head onset'))
    for k, (name, title) in enumerate(panels):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        for key in R_['spike_keys']:
            r = T.get((key, name, 'all'))
            if r is None:
                continue
            tau = 1000 * r['tau']
            ax.fill_between(tau, r['band'][0], r['band'][1], color=KEY_COLOR[key], alpha=0.15, lw=0)
            ax.plot(tau, r['peth'], color=KEY_COLOR[key], lw=1.6,
                    label='{} start {:+.0f} ms [{:+.0f},{:+.0f}]'.format(KEY_LABEL[key], 1000 * r['fit']['t0'],
                                                                        *(1000 * r['t0_ci'])))
        ax.axvline(0, color=INK2, lw=0.8)
        ax.axvspan(-1000 * R_['dt'], 0, color=GRID, alpha=0.6, lw=0)
        ax.set_xlim(-600, 800)
        ax.set_xlabel('time from onset (ms)', fontsize=8)
        ax.set_ylabel('fractional change', fontsize=8)
        re_ = T.get(('norm_spikes', name, 'enhanced'))
        if re_ is not None:
            ax.plot(1000 * re_['tau'], re_['peth'], color=KEY_COLOR['norm_spikes'], lw=1.0, ls=':',
                    label='OMSI, enhanced cells only')
        ax.set_title(title + ' (n={})'.format(T.get(('norm_spikes', name, 'all'), {}).get('n', 0)), fontsize=9, color=INK)
        ax.legend(fontsize=6.5, frameon=False, loc='upper left')
        na._style(ax)
    ax = fig.add_subplot(gs[0, 2])
    for cond, col in (('comp_light', COND_COLOR['comp']), ('matched_shift_light', COND_COLOR['shift'])):
        r = T.get(('norm_spikes', cond, 'all'))
        if r is None:
            continue
        ax.fill_between(1000 * r['tau'], r['band'][0], r['band'][1], color=col, alpha=0.15, lw=0)
        ax.plot(1000 * r['tau'], r['peth'], color=col, lw=1.6, label=cond.replace('_', ' '))
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(-600, 800)
    ax.set_xlabel('time from head onset (ms)', fontsize=8)
    ax.set_title('matched pairs, light, OMSI', fontsize=9, color=INK)
    ax.legend(fontsize=7, frameon=False)
    na._style(ax)
    lc = R_.get('lag_check')
    if lc is not None:
        ax = fig.add_subplot(gs[1, 2])
        ax.bar(lc['lags'], lc['cc_mean'], color='#2a78d6', width=0.6)
        ax.set_xlabel('lag of dF/F rise after OMSI spike (frames)', fontsize=8)
        ax.set_ylabel('mean correlation', fontsize=8)
        ax.set_title('OMSI frame-assignment check: peak lag {}'.format(lc['peak_lag']), fontsize=9, color=INK)
        na._style(ax)
    pdf.savefig(fig)
    plt.close(fig)


def _page_direction(pdf, R_):
    """ Gaze-shift vs visual-motion direction preference. """

    D = R_['dir'].get('norm_spikes')
    if D is None:
        return
    fig = na._new_page('4. Direction preferences (after Miura & Scanziani 2022)',
                       "d' of nasal vs temporal gaze shifts against d' of temporal vs nasal image motion in the RF "
                       'outside gaze shifts (head velocity stratified).  Both signed so a purely visual cell sits on '
                       'the diagonal.  Ceiling = sqrt(product of split-half reliabilities, Spearman-Brown).')
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.97, top=0.85, bottom=0.08, hspace=0.45, wspace=0.3)
    pairs = (('shift_dark', 'visual_light', 'dark gaze shift (non-visual)', 'visual motion, light'),
             ('shift_light', 'visual_light', 'light gaze shift', 'visual motion, light'),
             ('shift_light', 'shift_dark', 'light gaze shift', 'dark gaze shift'),
             ('visual_light', 'visual_dark', 'visual motion, light', 'same image motion, dark'))
    for k, (a, b, la, lb) in enumerate(pairs):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        x, y = D[a]['dprime'], D[b]['dprime']
        sel = (D[a]['p'] < 0.05) & (D[b]['p'] < 0.05)
        _pc(ax, x, y, '#c3c2bd', 'all', s=6)
        _pc(ax, x[sel], y[sel], '#2a78d6', 'both p<0.05', s=14)
        lim = np.nanpercentile(np.abs(np.r_[x, y]), 99)
        ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=0.8, ls='--')
        ax.axhline(0, color=GRID, lw=0.8)
        ax.axvline(0, color=GRID, lw=0.8)
        ca, cs = D['corr'][(a, b, 'all')], D['corr'][(a, b, 'selective')]
        ax.set_title('rho {:+.2f} [{:+.2f},{:+.2f}] n={}; ceiling {:.2f}\nselective: rho {:+.2f} n={}'.format(
            ca['rho'], *ca['ci'], ca['n'], ca.get('ceiling', np.nan), cs['rho'], cs['n']), fontsize=8, color=INK)
        ax.set_xlabel(la + " d'", fontsize=8)
        ax.set_ylabel(lb + " d'", fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        na._style(ax)
    ax = fig.add_subplot(gs[1, 1])
    names = ('shift_light', 'shift_dark', 'visual_light', 'visual_dark')
    rel = [D[n]['reliability'] for n in names]
    ax.bar(range(4), rel, color=['#eda100', '#2b2b6b', '#1baf7a', '#8a8986'], width=0.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=7)
    ax.set_ylabel('split-half reliability (S-B)', fontsize=8)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title('Can these indices correlate at all?', fontsize=9, color=INK)
    na._style(ax)
    ax = fig.add_subplot(gs[1, 2])
    fr = [np.mean(D[n]['p'][np.isfinite(D[n]['p'])] < 0.05) for n in names]
    ax.bar(range(4), fr, color='#52514e', width=0.6)
    ax.axhline(0.05, color='#d03b3b', lw=0.8, ls='--')
    ax.set_xticks(range(4))
    ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=7)
    ax.set_ylabel('fraction of cells p < 0.05', fontsize=8)
    ax.set_title('Direction-selective cells (dashed = chance)', fontsize=9, color=INK)
    na._style(ax)
    pdf.savefig(fig)
    plt.close(fig)


def _page_methods(pdf, R_):
    """ Assumptions and parameters. """

    fig = na._new_page('Methods and assumptions')
    txt = [
        'Eye: theta and phi recomputed at the measured eye scale ({:.0f} px) from <prefix>_eye_scale.json; '
        'pupil-from-head rebuilt as ang_offset - theta, so pose and optic flow use the same eye.'.format(R_['eye_scale_px']),
        'Gaze shift: |gaze| > {:.0f} deg/s (new_analysis.classify_gated); onset where |gaze| last rose through {:.0f} '
        'deg/s, interpolated.  Head movement: |head| > {:.0f} deg/s, onset/offset at {:.0f} deg/s.  Compensatory: no '
        'gaze-shift frame and |gaze| < {:.0f} deg/s from 100 ms before to 100 ms after.  Isolated: no other head-movement '
        'or gaze-shift onset in the preceding {:.1f} s ({:.1f} s for the timing test, which reads the pre-onset '
        'period).'.format(na.SHIFTED_GAZE, GAZE_ON, HEAD_RUN, HEAD_ON, na.STILL_GAZE, ISO_S, ISO_TIMING),
        'Matching: log peak head velocity, log head amplitude and time since the previous movement (cap 2 s), standardized, nearest neighbour without replacement '
        'inside direction x light strata, caliper {:.1f} SD.'.format(CALIPER),
        '2P timing: twopT is the frame start; a cell at mean row y is sampled y/{1} of a frame later.  A sample at s '
        'carries the spikes inferred for (s - dt, s], i.e. a spike sits on the first sample its fluorescence jump shows '
        'in -- where OASIS and suite2p put it.  OMSI timestamps it {0} frame(s) earlier, an offset in OMSI\'s output, '
        'so OMSI is moved onto the jump for every test; the timing page also shows it as stored.  '
        'The stored OMSI spike_times key is NOT used: it is frame-quantized and does not match omsi_spike_rate.'.format(
            (R_.get('lag_check') or {}).get('peak_lag', 'n/a'), 512),
        'RF: sparse-noise Gaussian fits (sn1/sparse_noise_labels_gaussfit.npz), monitor {}x{} px, flat screen, '
        'centre on the optic axis, focal length {:.0f} px (~{:.1f} px/deg at the centre, from the stimulus script).  '
        'RF width + {:.0f} deg mapping uncertainty in quadrature.  Cells without a fit use the population RF.  Mirrored '
        'and whole-field versions test the placement.'.format(MON_SHAPE[1], MON_SHAPE[0], MON_F_PX,
                                                              MON_F_PX * np.pi / 180, MAP_SIGMA),
        'Image change: surface labels ray-cast on a {:.0f} deg grid over +-{:.0f} deg at 50 ms before onset and 50 ms '
        'after the movement ends, luminances as rendered (sky 255, wall 170, floor 100, pillar 237).'.format(GRID_STEP, GRID_HALF),
        'Visual motion: horizontal analytic optic flow (rotation + translation, new_analysis.optic_flow) weighted '
        'by the RF; samples within {:.1f} s before to {:.1f} s after any gaze shift excluded; |u| > {:.0f} deg/s; '
        'head-velocity strata {}; at least {} samples per side.'.format(EXCL_PRE, EXCL_POST, U0,
                                                                       list(HEAD_STRATA[1:-1].astype(int)), MIN_STRATUM),
        'Statistics: Benjamini-Hochberg q < 0.1 for responsive cells; {} bootstrap resamples; {} permutations for '
        'gaze-shift direction, {} circular shifts (>= 60 s) for visual direction; regression nulls permute the group '
        'rows within direction ({} permutations).'.format(N_BOOT, N_PERM // 4, N_PERM // 10, N_SHUF // 4),
        'Runtime {:.0f} s.'.format(R_.get('runtime_s', np.nan)),
    ]
    y = 0.88
    for t in txt:
        for line in textwrap.wrap(t, 160):
            fig.text(0.05, y, line, fontsize=8.5, color=INK2, va='top')
            y -= 0.022
        y -= 0.012
    pdf.savefig(fig)
    plt.close(fig)


def verdicts(R_):
    """ One paragraph per test, OMSI. """

    out = []
    comp = R_['comp'].get('norm_spikes', {})
    parts = []
    for cond in ('light', 'dark'):
        if cond in comp:
            c = comp[cond]
            parts.append('{}: {} pairs; pooled response to shifts {:+.2f} [{:+.2f}, {:+.2f}], to compensatory '
                         'movements {:+.2f} [{:+.2f}, {:+.2f}], ratio {:.2f} [{:.2f}, {:.2f}]; {} individually responsive '
                         'cells; residual RF luminance change {:.3f} vs {:.3f} in the shifts, '
                         '|gaze displacement| {:.1f} vs {:.1f} deg'.format(
                             cond, c['n_pairs'], c['pooled_shift'], *c['pooled_shift_ci'], c['pooled_comp'],
                             *c['pooled_comp_ci'], c['pooled_ratio'], *c['pooled_ratio_ci'], int(c['selected'].sum()),
                             np.nanmedian(c['dI_comp']), np.nanmedian(c['dI_shift']),
                             np.nanmedian(np.abs(c['balance']['gaze_disp'][1])),
                             np.nanmedian(np.abs(c['balance']['gaze_disp'][0]))))
    out.append(('1. Compensatory probe', '; '.join(parts) + '.  A ratio near 1 means the response does not need '
                'the image to move; near 0 means it does.  In the dark both movements are non-visual, so the dark '
                'ratio is the kinematic reference for the light one.'))
    reg = R_['reg'].get('norm_spikes', {})
    parts = []
    for cond in ('light', 'dark'):
        if cond in reg:
            r = reg[cond]
            pv = (1 + np.sum(r['pop_vis_null'] >= r['pop_vis'])) / (1 + len(r['pop_vis_null']))
            pk = (1 + np.sum(r['pop_kin_null'] >= r['pop_kin'])) / (1 + len(r['pop_kin_null']))
            parts.append('{}: median unique R^2 visual {:.4f} (p={:.3g}), kinematic {:.4f} (p={:.3g}); cells p<0.05 '
                         'visual {}/{}, kinematic {}/{}'.format(cond, r['pop_vis'], pv, r['pop_kin'], pk,
                                                                int((r['p_vis'] < 0.05).sum()), len(r['cells']),
                                                                int((r['p_kin'] < 0.05).sum()), len(r['cells'])))
    out.append(('2. Regression within gaze shifts', '; '.join(parts) + '.  Chance is 5% of cells.'))
    T = R_['timing']
    parts = []
    for name in ('gaze_dark', 'gaze_light', 'head_dark', 'head_light'):
        r = T.get(('norm_spikes', name, 'all'))
        if r is not None:
            parts.append('{} (n={}): response {:+.2f} [{:+.2f}, {:+.2f}], ramp start {:+.0f} ms [{:+.0f}, {:+.0f}], '
                         'pre-onset {:+.3f} [{:+.3f}, {:+.3f}]'.format(
                             name.replace('_', ' '), r['n'], r['post'], *r['post_ci'], 1000 * r['fit']['t0'],
                             *(1000 * r['t0_ci']), r['pre'], *r['pre_ci']))
    lc = R_.get('lag_check')
    out.append(('3. Onset timing (OMSI, all cells pooled)', '; '.join(parts) + '.  Response and pre-onset are '
                'fractional changes of the pooled rate.  Enhanced cells per parity half: {}.  OMSI spikes '
                'peak against the dF/F rise at lag {} frames.'.format(T['n_selected'], lc['peak_lag'] if lc else 'n/a')))
    D = R_['dir'].get('norm_spikes')
    if D is not None:
        c1 = D['corr'][('shift_dark', 'visual_light', 'all')]
        c2 = D['corr'][('shift_light', 'visual_light', 'all')]
        c3 = D['corr'][('shift_light', 'shift_dark', 'all')]
        out.append(('4. Direction preferences', 'dark shift vs light visual motion rho {:+.2f} [{:+.2f}, {:+.2f}] '
                    '(n={}, ceiling {:.2f}); light shift vs visual motion rho {:+.2f} [{:+.2f}, {:+.2f}] (ceiling '
                    '{:.2f}); light vs dark shift rho {:+.2f}.  Reliability: shift light {:.2f}, shift dark {:.2f}, '
                    'visual light {:.2f}, visual dark {:.2f}.  An r near 0 only means "non-visual" when the ceiling '
                    'is well above 0.'.format(c1['rho'], *c1['ci'], c1['n'], c1.get('ceiling', np.nan), c2['rho'],
                                             *c2['ci'], c2.get('ceiling', np.nan), c3['rho'],
                                             D['shift_light']['reliability'], D['shift_dark']['reliability'],
                                             D['visual_light']['reliability'], D['visual_dark']['reliability'])))
    return out


def make_pdf(path, R_):
    """ Multi-page summary PDF. """

    with PdfPages(path) as pdf:
        _page_summary(pdf, R_, verdicts(R_))
        _page_comp(pdf, R_)
        _page_regression(pdf, R_)
        _page_timing(pdf, R_)
        _page_direction(pdf, R_)
        _page_methods(pdf, R_)


def main():
    """ Command-line entry point. """

    ap = argparse.ArgumentParser(description='Visual vs non-visual gaze-shift responses.')
    ap.add_argument('--h5_path', required=True)
    ap.add_argument('--npz_path', required=True,
                    help='Reconstruction pose npz (get_retinal_image --pose_only is enough).')
    ap.add_argument('--sn_dir', default=None, help='Sparse-noise directory; defaults to ../sn1.')
    ap.add_argument('--spike_keys', nargs='+', default=list(SPIKE_KEYS))
    ap.add_argument('--mirror_rf', action='store_true',
                    help='Use the mirrored monitor-to-retina mapping as the primary one.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_prefix', default=None,
                    help='Defaults to the h5 path minus _preproc.h5, plus _nonvisual.')
    args = ap.parse_args()

    R_ = run_probe(args.h5_path, args.npz_path, sn_dir=args.sn_dir, spike_keys=args.spike_keys,
                   mirror=args.mirror_rf, seed=args.seed)
    prefix = args.out_prefix or args.h5_path.replace('_preproc.h5', '') + '_nonvisual'
    make_pdf(prefix + '.pdf', R_)
    save_probe(prefix + '.h5', R_)
    summ = {'session': R_['session'], 'verdicts': dict(verdicts(R_))}
    with open(prefix + '.json', 'w') as f:
        json.dump(summ, f, indent=2, default=float)
    print('Saved {0}.pdf, {0}.h5 and {0}.json.  Total {1:.0f} s.'.format(prefix, R_['runtime_s']))


if __name__ == '__main__':
    main()
