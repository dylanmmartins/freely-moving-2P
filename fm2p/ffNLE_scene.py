# -*- coding: utf-8 -*-
"""
fm2p/ffNLE_scene.py

Scene-based encoding model: does the reconstructed retinal image predict
neural activity, and does the retinocentric version do it better than the
egocentric one?

Takes two `<prefix>_retinal_images.npz` files from get_retinal_image -- one
rendered with the eye frozen (egocentric) and one with the eye driving the
camera (retinocentric) -- bins both onto the 2P frame times, and fits a
cross-validated ridge encoding model per cell from a low-dimensional basis of
the scene.  Both conditions share the frame set, the folds, the basis
dimension and the penalty grid, so the ego-vs-retino difference is a paired
1-df comparison rather than two independent fits.

Why this and not a receptive field: the pillar bearing on the retina has an
integrated autocorrelation time of ~35 s, so 30 min of data is on the order of
30-60 independent looks at the scene.  Fitting a 3600-pixel map from that many
samples cannot work (see rf_from_reconstruction).  A 20-component encoding
model asks a far better-powered question of the same data, and the ego-vs-retino
contrast is 1 df rather than 3600.

Everything here is aimed at not fooling yourself:

  * the null is a circular shift of the rate trace, so it carries the same
    calcium autocorrelation the observed statistic does;
  * folds are contiguous blocks with an embargo, so a test block never sits
    inside the training set's autocorrelation;
  * an acausal arm (negative lags) runs alongside the causal one -- a scene
    model that predicts equally well from the future is a behavioral-epoch
    signature, not a visual response;
  * a behavior-only model runs alongside, and the headline number is the
    scene's UNIQUE variance over it;
  * a dark-epoch arm runs alongside, where the reconstruction still renders a
    scene the animal cannot see.

Ablations drop large regions of the scene, refit, and measure the loss.  A
region's drop only means something against a matched control, since half a
scene is highly redundant with the whole -- so a calibration curve of drop
versus the fraction of pixels removed at random is fit alongside, and each
region is reported as its excess over that curve.

Classes
-------
SceneFeatures
    Scene pixels to PCA scores with temporal lags; PCA fit on train rows only.
BehaviorFeatures
    Head and eye channels with temporal lags, for the nuisance baseline.
ConcatFeatures
    Two featurizers side by side, for the scene-plus-behavior model.

Functions
---------
_triangle_smooth
    Triangle-smooth each column of the target, as ffNLE does.
_fmt
    Format a float for printing, tolerating NaN.
bin_scene_to_2p
    Chunked average of the retinal movie within each 2P frame.
region_masks
    Named pixel masks over the retinal field for the ablations.
block_folds
    Contiguous-block cross-validation folds with an embargo.
_r2_cols
    Column-wise R^2 against the column mean of Y (or a supplied mean).
_corr_cols
    Column-wise Pearson r between two matrices of the same shape.
_svd_prep
    Thin SVD of a design matrix with near-zero singular values dropped.
_ridge_coefs
    Ridge coefficients from a prepared SVD, with a penalty per column of Y.
_pick_lambda
    Per-cell ridge penalty from an inner blockwise split of a training fold.
_fit_static_nl
    Monotone lookup from a linear prediction to the target, per cell.
_apply_static_nl
    Apply a per-cell monotone lookup, falling back to the linear value.
cv_ridge
    Cross-validated multi-output ridge with a circular-shift null.
_apply_lags
    Stack lagged copies of a feature matrix.
_lag_valid
    Samples whose every lagged source frame is itself usable.
load_behavior_channels
    Pull the self-motion and eye channels off a preproc h5, on the 2P clock.
load_scene
    Load one retinal-image npz and bin it onto the 2P frame times.
_ablation_regions
    Refit the model on the complement of each named region.
_random_calibration
    What removing a given fraction of the field at random costs.
_excess_drop
    Each region's drop above what removing that much field at random costs.
run_condition
    Every model arm for one npz: full, acausal, behavior, dark, ablations.
compare_conditions
    Paired ego-vs-retino statistics across cells.
_unique_scene_variance
    Scene variance over and above the behavior-only fit, per cell.
ffNLE_scene
    Top-level driver: load, fit, compare, save, plot.
print_verdict
    Plain-language readout of what the control arms decided.
_sanitize
    Recursively drop Nones and unsaveable values so write_h5 can take it.
_save
    Write the result tree to HDF5, minus the out-of-fold prediction matrices.
_save_json
    Write a scalar summary next to the h5, for reading without h5py.
_sig_mask
    Cells whose scene fit beat the circular-shift null.
_imshow_scene
    Show a scene-shaped map with degree ticks.
make_summary_pdf
    Figures for the whole run.
_ablation_pages
    Region-drop heatmap, the random-area calibration, and the tile maps.
_cell_pages
    Per-cell out-of-fold prediction traces for the best-fit cells.
main
    CLI entry point.


DMM, September 2026
"""

if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

import argparse
import json
import os
import time

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

from .utils.files import write_h5
from .rf_from_reconstruction import retinal_pixel_angles, RET_RES, RET_FOV_DEG

# Speed above which a frame counts as locomotion, cm/s.  Matches the motion
# filter used elsewhere in the package.
SPEED_THRESH_CM_S = 2.0

# Scene-to-response offsets in 2P frames at 7.5 Hz (133 ms per frame).  Zero
# through three covers the calcium rise; the acausal arm is these negated.
DEFAULT_LAGS = (0, 1, 2, 3)

# Ridge penalties, swept per cell on an inner split of each training fold.
LAM_GRID = np.logspace(-2, 6, 17)

# Fractions of the field removed at random to calibrate what an ablation of a
# given size costs by itself.
RANDOM_CALIB_FRACS = (0.25, 0.5, 0.75)


def _triangle_smooth(Y, half_width_s, dt):
    """ Triangle-smooth each column of the target, as ffNLE does.

    Deconvolved spikes at 7.5 Hz are mostly shot noise frame to frame, and no
    scene model of any kind can predict that -- fitting them raw pins every R^2
    near zero and leaves the ego-vs-retino contrast with no power at all.  The
    kernel matches ffNLE's TRIANGLE_HALF_WIDTH_S so the numbers here sit on the
    same scale as the behavioral models.

    Smoothing raises the target's autocorrelation, which inflates R^2 -- the
    circular-shift null is computed on the smoothed trace too, so it inflates by
    the same amount and the corrected number stays honest.
    """

    hw = max(1, int(round(float(half_width_s) / dt)))
    ramp = np.arange(1, hw + 2, dtype=np.float64)
    kern = np.concatenate([ramp, ramp[-2::-1]])
    kern /= kern.sum()
    out = np.empty_like(Y)
    for c in range(Y.shape[1]):
        out[:, c] = np.convolve(Y[:, c], kern, mode='same')
    return out


def _fmt(x, nd=3):
    """ Format a float for printing, tolerating NaN. """

    return 'nan' if not np.isfinite(x) else '{:.{}f}'.format(float(x), nd)


def bin_scene_to_2p(retinal_images, eyeT, twopT, rendered=None, bin_px=4,
                    chunk=8000, verbose=True):
    """ Average the retinal movie within each 2P frame, in chunks.

    A 2P frame integrates for ~133 ms and the retinal image moves a long way in
    that time, so what a frame's calcium saw is the mean image over the
    interval, not the nearest eye frame.  Bin edges sit midway between 2P frame
    times.  Chunked because the full movie is (120k, 120, 120) and casting it
    to float in one go is ~7 GB.

    Parameters
    ----------
    retinal_images : (n_eye, 120, 120) uint8
    eyeT : (n_eye,) float
        Eye-camera times on the same origin as twopT.
    twopT : (n_2p,) float
    rendered : (n_eye,) bool or None
        Which eye frames actually got a render.  Unrendered frames are all-zero
        and would drag the bin mean down, so they are dropped rather than
        averaged in.
    bin_px : int
        Spatial block-mean factor.  4 gives 30x30, about 4 deg per bin.

    Returns
    -------
    S : (n_2p, h, w) float32 in [0, 1]; bins with no eye frame are 0
    n_per_bin : (n_2p,) int
    """

    eyeT = np.asarray(eyeT, dtype=float)
    twopT = np.asarray(twopT, dtype=float)
    n_2p = len(twopT)
    bin_px = max(1, int(bin_px))

    mids = 0.5 * (twopT[1:] + twopT[:-1])
    edges = np.concatenate([[twopT[0] - (mids[0] - twopT[0])], mids,
                            [twopT[-1] + (twopT[-1] - mids[-1])]])

    ok = np.isfinite(eyeT)
    if rendered is not None:
        ok &= np.asarray(rendered, dtype=bool)[:len(eyeT)]
    idx = np.searchsorted(edges, eyeT[ok], side='right') - 1
    keep = (idx >= 0) & (idx < n_2p)
    idx = idx[keep]
    src = np.flatnonzero(ok)[keep]

    h_full, w_full = retinal_images.shape[1], retinal_images.shape[2]
    h2, w2 = (h_full // bin_px) * bin_px, (w_full // bin_px) * bin_px
    h, w = h2 // bin_px, w2 // bin_px

    S = np.zeros((n_2p, h, w), dtype=np.float32)
    n_per_bin = np.zeros(n_2p, dtype=np.int64)

    t0 = time.time()
    for start in range(0, len(src), chunk):
        sl = slice(start, start + chunk)
        block = retinal_images[src[sl]].astype(np.float32) / 255.0
        block = block[:, :h2, :w2].reshape(
            block.shape[0], h, bin_px, w, bin_px).mean(axis=(2, 4))
        np.add.at(S, idx[sl], block)
        np.add.at(n_per_bin, idx[sl], 1)
        del block
    nz = n_per_bin > 0
    S[nz] /= n_per_bin[nz, None, None]

    if verbose:
        print('  Binned to {} in {:.1f}s -- {:.1f}% of 2P frames got an eye '
              'frame (median {:.0f} per bin)...'.format(
                  S.shape, time.time() - t0, 100.0 * nz.mean(),
                  np.median(n_per_bin[nz]) if nz.any() else 0))
    return S, n_per_bin


def region_masks(h, w, bin_px=4, center_deg=30.0, horizon_deg=15.0, n_tiles=4):
    """ Named boolean pixel masks over the retinal field, for ablations.

    Each mask marks the pixels that belong to the region; the ablation REMOVES
    them and refits on the complement.

    Positive azimuth is the animal's right and positive elevation is up,
    following get_retinal_image's pinhole convention.  Names are left in those
    terms rather than nasal/temporal because which is which depends on the eye,
    and both npz files are rendered for the same one.

    Parameters
    ----------
    h, w : int
        Shape of the binned scene.
    bin_px : int
        Binning factor used, so pixel angles match.
    center_deg : float
        Radius of the central disk.
    horizon_deg : float
        Half-height of the horizon band.
    n_tiles : int
        Side of the square tile grid used for the coarse importance map.

    Returns
    -------
    masks : dict of str to (h, w) bool
    az, el : (h, w) float
        Pixel-center angles in degrees.
    """

    az_f, el_f = retinal_pixel_angles(RET_RES, RET_FOV_DEG)
    h2, w2 = h * bin_px, w * bin_px
    az = az_f[:h2, :w2].reshape(h, bin_px, w, bin_px).mean(axis=(1, 3))
    el = el_f[:h2, :w2].reshape(h, bin_px, w, bin_px).mean(axis=(1, 3))

    masks = {}
    masks['az_right'] = az > 0
    masks['az_left'] = az <= 0
    masks['el_upper'] = el > 0
    masks['el_lower'] = el <= 0
    masks['quad_upper_right'] = (az > 0) & (el > 0)
    masks['quad_upper_left'] = (az <= 0) & (el > 0)
    masks['quad_lower_right'] = (az > 0) & (el <= 0)
    masks['quad_lower_left'] = (az <= 0) & (el <= 0)
    masks['center'] = np.hypot(az, el) < center_deg
    masks['periphery'] = ~masks['center']
    masks['horizon_band'] = np.abs(el) < horizon_deg

    # Coarse tile grid: ablating one tile at a time gives an importance map
    # without pretending to per-pixel resolution the data cannot support.
    rows = np.array_split(np.arange(h), n_tiles)
    cols = np.array_split(np.arange(w), n_tiles)
    for ri, rr in enumerate(rows):
        for ci, cc in enumerate(cols):
            m = np.zeros((h, w), dtype=bool)
            m[np.ix_(rr, cc)] = True
            masks['tile_r{}c{}'.format(ri, ci)] = m

    return masks, az, el


def block_folds(twopT, use, n_folds=5, block_s=120.0, embargo_s=40.0, seed=0):
    """ Contiguous-block folds with an embargo around each test block.

    Blocks are contiguous in time and dealt round-robin to folds, so no fold is
    a single stretch of the recording (which would confound the model with
    drift).  The embargo then drops training frames within embargo_s of any
    test block: at a ~35 s stimulus autocorrelation, a training frame right
    beside a test frame is not an independent sample of anything.

    Parameters
    ----------
    twopT : (n_2p,) float
    use : (n_2p,) bool
        Frames eligible for the model.
    n_folds : int
    block_s : float
        Block length in seconds.  Should comfortably exceed the stimulus
        autocorrelation time.
    embargo_s : float
        Training frames this close in time to a test block are dropped.

    Returns
    -------
    fold_id : (n_2p,) int
        Fold index per frame, -1 where not used.
    train_masks : list of (n_2p,) bool
    test_masks : list of (n_2p,) bool
    """

    t = np.asarray(twopT, dtype=float)
    t0 = t[0]
    block = np.floor((t - t0) / float(block_s)).astype(int)
    n_blocks = block.max() + 1

    # Deal blocks greedily to whichever fold is currently lightest in USABLE
    # frames.  A plain round robin balances wall-clock time, which is not the
    # same thing once the motion filter and the light epochs have taken their
    # cut -- on a typical session that leaves one fold with a few hundred test
    # frames and another with a thousand, and the small fold's R^2 is noise.
    n_use_block = np.array([int(np.sum(use & (block == b))) for b in range(n_blocks)])
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_blocks)
    order = order[np.argsort(-n_use_block[order], kind='stable')]
    block_fold = np.zeros(n_blocks, dtype=int)
    load = np.zeros(int(n_folds))
    for b in order:
        k = int(np.argmin(load))
        block_fold[b] = k
        load[k] += n_use_block[b]

    fold_id = np.where(use, block_fold[block], -1)

    train_masks, test_masks = [], []
    for k in range(int(n_folds)):
        te = use & (fold_id == k)
        tr = use & (fold_id != k)
        if embargo_s > 0 and te.any():
            te_t = t[te]
            # Distance from each frame to the nearest test frame in time.
            pos = np.searchsorted(te_t, t)
            lo = np.clip(pos - 1, 0, len(te_t) - 1)
            hi = np.clip(pos, 0, len(te_t) - 1)
            dist = np.minimum(np.abs(t - te_t[lo]), np.abs(t - te_t[hi]))
            tr &= dist > float(embargo_s)
        train_masks.append(tr)
        test_masks.append(te)

    return fold_id, train_masks, test_masks


def _r2_cols(Y, Yhat, mu=None):
    """ Column-wise R^2 against the column mean of Y (or a supplied mean). """

    Y = np.asarray(Y, dtype=np.float64)
    Yhat = np.asarray(Yhat, dtype=np.float64)
    mu = Y.mean(axis=0) if mu is None else np.asarray(mu, dtype=np.float64)
    ss_res = np.sum((Y - Yhat) ** 2, axis=0)
    ss_tot = np.sum((Y - mu) ** 2, axis=0)
    out = 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)
    return out


def _corr_cols(Y, Yhat):
    """ Column-wise Pearson r between two matrices of the same shape. """

    a = Y - Y.mean(axis=0)
    b = Yhat - Yhat.mean(axis=0)
    den = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
    return np.sum(a * b, axis=0) / np.where(den > 0, den, np.nan)


def _svd_prep(X):
    """ Thin SVD of a design matrix with near-zero singular values dropped. """

    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    keep = s > (s.max() * 1e-10 if s.size and s.max() > 0 else 0.0)
    return U[:, keep], s[keep], Vt[keep]


def _ridge_coefs(U, s, Vt, Yc, lam):
    """ Ridge coefficients from a prepared SVD, with a penalty per column of Y.

    Parameters
    ----------
    U, s, Vt : arrays
        Thin SVD of the centered training design matrix.
    Yc : (n_train, C)
        Training targets, already centered.
    lam : float or (C,) array
        Ridge penalty, per cell when an array.

    Returns
    -------
    B : (n_features, C)
    """

    A = U.T @ Yc
    if np.isscalar(lam):
        D = (s / (s ** 2 + float(lam)))[:, None]
    else:
        lam = np.asarray(lam, dtype=np.float64)[None, :]
        D = s[:, None] / (s[:, None] ** 2 + lam)
    return Vt.T @ (D * A)


def _pick_lambda(Xtr, Ytr, block_tr, lam_grid):
    """ Per-cell ridge penalty from an inner blockwise split of a training fold.

    The inner split uses alternate blocks, so the validation rows are as far
    from the fitting rows as the outer folds are.  Only the penalty is chosen
    here -- both halves live inside the outer training set, so nothing about
    the held-out fold leaks in.

    Returns
    -------
    lam : (C,) float
    """

    ub = np.unique(block_tr)
    if len(ub) < 2:
        return np.full(Ytr.shape[1], float(np.median(lam_grid)))
    inner_val = np.isin(block_tr, ub[::2])
    inner_tr = ~inner_val
    if inner_tr.sum() < 20 or inner_val.sum() < 20:
        return np.full(Ytr.shape[1], float(np.median(lam_grid)))

    Xi, Xv = Xtr[inner_tr], Xtr[inner_val]
    Yi, Yv = Ytr[inner_tr], Ytr[inner_val]
    xm, ym = Xi.mean(axis=0), Yi.mean(axis=0)
    U, s, Vt = _svd_prep(Xi - xm)
    A = U.T @ (Yi - ym)
    Xv_c = Xv - xm

    best = np.full(Yv.shape[1], -np.inf)
    lam_out = np.full(Yv.shape[1], float(lam_grid[-1]))
    for lam in lam_grid:
        B = Vt.T @ ((s / (s ** 2 + lam))[:, None] * A)
        r2 = _r2_cols(Yv, Xv_c @ B + ym)
        better = np.isfinite(r2) & (r2 > best)
        best[better] = r2[better]
        lam_out[better] = float(lam)

    # Cells no penalty helps get the heaviest one, which shrinks the fit to the
    # training mean.  Without this the inner split -- adjacent blocks, no
    # embargo -- picks a penalty that is too light for the outer fold, and every
    # unpredictable cell reports a large negative R^2 instead of about zero.
    # The null reuses these penalties, so the comparison stays symmetric.
    lam_out[~(best > 0)] = float(lam_grid[-1])
    return lam_out


def _fit_static_nl(yhat, y, n_bins=12):
    """ Monotone lookup from a linear prediction to the target, per cell.

    Turns the ridge fit into a proper LN model: the linear stage sets the
    ordering, the lookup sets the scale.  Fit on training rows only.

    Returns
    -------
    knots : (C, n_bins) float
        Bin centers in prediction units.
    vals : (C, n_bins) float
        Mean target in each bin, made monotone by a running maximum of the
        cumulative isotonic-lite pass (plain means, then sorted).
    """

    C = yhat.shape[1]
    knots = np.full((C, n_bins), np.nan)
    vals = np.full((C, n_bins), np.nan)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    for c in range(C):
        e = np.quantile(yhat[:, c], qs)
        e = np.unique(e)
        if len(e) < 3:
            continue
        b = np.clip(np.searchsorted(e, yhat[:, c], side='right') - 1, 0, len(e) - 2)
        nb = len(e) - 1
        cnt = np.bincount(b, minlength=nb).astype(float)
        sm = np.bincount(b, weights=y[:, c], minlength=nb)
        sk = np.bincount(b, weights=yhat[:, c], minlength=nb)
        ok = cnt > 0
        if ok.sum() < 3:
            continue
        k = (sk[ok] / cnt[ok])
        v = (sm[ok] / cnt[ok])
        # Enforce monotone increasing so the lookup cannot invent structure the
        # linear stage did not order.
        v = np.maximum.accumulate(v) if np.corrcoef(k, v)[0, 1] >= 0 \
            else np.minimum.accumulate(v)
        knots[c, :len(k)] = k
        vals[c, :len(v)] = v
    return knots, vals


def _apply_static_nl(yhat, knots, vals):
    """ Apply a per-cell monotone lookup, falling back to the linear value. """

    out = yhat.copy()
    for c in range(yhat.shape[1]):
        k = knots[c][np.isfinite(knots[c])]
        v = vals[c][np.isfinite(vals[c])]
        if len(k) < 3:
            continue
        out[:, c] = np.interp(yhat[:, c], k, v)
    return out


def cv_ridge(build_X, Y, train_masks, test_masks, block, lam_grid=LAM_GRID,
             n_shuffles=0, min_shift=200, seed=0, static_nl=False,
             return_oof=False, verbose=False):
    """ Cross-validated multi-output ridge with a circular-shift null.

    Parameters
    ----------
    build_X : callable
        `build_X(train_rows_bool) to (n_2p, n_features)` design matrix.  Given
        the training mask so any fitting the featurizer does (scene PCA, z-score
        statistics) sees only training rows.
    Y : (n_2p, C) float
        Neural activity on the 2P timebase, NaN-free.
    train_masks, test_masks : list of (n_2p,) bool
    block : (n_2p,) int
        Time-block index per frame, used for the inner penalty split.
    n_shuffles : int
        Circular shifts of Y for the null.  Shifts are applied to the full-length
        trace before subsetting, so the calcium autocorrelation is preserved and
        the null R^2 carries the same inflation the observed one does.
    min_shift : int
        Smallest shift in 2P frames, so a shift never leaves the trace aligned
        with itself through its own autocorrelation.
    static_nl : bool
        Fit a per-cell monotone output nonlinearity on the training rows.

    Returns
    -------
    dict with 'r2', 'r', 'r2_folds', 'lam', 'null_r2', 'p', 'r2_corrected',
    'n_features', 'n_test', and optionally 'oof'.

    Notes
    -----
    The null reuses each fold's penalty from the real fit rather than re-tuning
    per shift.  That hands the shifted data a penalty tuned on real structure,
    which if anything widens the null -- the conservative direction -- and it
    keeps the cost of 50 shuffles at a few matrix products instead of 50 full
    penalty sweeps.
    """

    Y = np.asarray(Y, dtype=np.float64)
    n_2p, C = Y.shape
    n_folds = len(train_masks)

    used = np.zeros(n_2p, dtype=bool)

    oof = np.full((n_2p, C), np.nan) if return_oof else None
    ss_res = np.zeros(C)
    sy = np.zeros(C)
    syy = np.zeros(C)
    r2_folds = np.full((n_folds, C), np.nan)
    lam_all = np.full((n_folds, C), np.nan)

    shifts = np.array([], dtype=int)
    if n_shuffles and n_shuffles > 0 and (n_2p - 2 * min_shift) > 0:
        rng = np.random.default_rng(seed)
        shifts = rng.integers(min_shift, n_2p - min_shift, size=int(n_shuffles))
    n_sh = len(shifts)
    null_res = np.zeros((n_sh, C))
    null_sy = np.zeros((n_sh, C))
    null_syy = np.zeros((n_sh, C))

    n_features = 0
    for k in range(n_folds):
        tr, te = train_masks[k], test_masks[k]
        if tr.sum() < 50 or te.sum() < 20:
            continue
        X = build_X(tr)
        n_features = X.shape[1]
        Xtr, Xte = X[tr].astype(np.float64), X[te].astype(np.float64)
        Ytr, Yte = Y[tr], Y[te]

        lam = _pick_lambda(Xtr, Ytr, block[tr], lam_grid)
        lam_all[k] = lam

        xm = Xtr.mean(axis=0)
        ym = Ytr.mean(axis=0)
        U, s, Vt = _svd_prep(Xtr - xm)
        Xte_c = Xte - xm

        B = _ridge_coefs(U, s, Vt, Ytr - ym, lam)
        pred = Xte_c @ B + ym
        if static_nl:
            knots, vals = _fit_static_nl((Xtr - xm) @ B + ym, Ytr)
            pred = _apply_static_nl(pred, knots, vals)

        used |= te
        if return_oof:
            oof[te] = pred
        ss_res += np.sum((Yte - pred) ** 2, axis=0)
        sy += Yte.sum(axis=0)
        syy += (Yte ** 2).sum(axis=0)
        r2_folds[k] = _r2_cols(Yte, pred)

        # Nulls reuse this fold's SVD and penalty; only U.T @ Y changes.
        # Index the shifted rows directly rather than rolling the whole trace:
        # the fold only ever needs its own train and test rows.
        tr_rows = np.flatnonzero(tr)
        te_rows = np.flatnonzero(te)
        for j, sh in enumerate(shifts):
            Ys_tr = Y[(tr_rows - int(sh)) % n_2p]
            Ys_te = Y[(te_rows - int(sh)) % n_2p]
            ym_s = Ys_tr.mean(axis=0)
            Bs = _ridge_coefs(U, s, Vt, Ys_tr - ym_s, lam)
            pred_s = Xte_c @ Bs + ym_s
            null_res[j] += np.sum((Ys_te - pred_s) ** 2, axis=0)
            null_sy[j] += Ys_te.sum(axis=0)
            null_syy[j] += (Ys_te ** 2).sum(axis=0)

        if verbose:
            print('    Fold {}/{}: {} train, {} test, median R2 {}'.format(
                k + 1, n_folds, int(tr.sum()), int(te.sum()),
                _fmt(np.nanmedian(r2_folds[k]))))

    n_used = int(used.sum())
    ss_tot = syy - sy ** 2 / max(n_used, 1)
    r2 = 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)

    null_r2 = np.full((n_sh, C), np.nan)
    p = np.full(C, np.nan)
    if n_sh:
        nt = null_syy - null_sy ** 2 / max(n_used, 1)
        null_r2 = 1.0 - null_res / np.where(nt > 0, nt, np.nan)
        ge = np.nansum(null_r2 >= r2[None, :], axis=0)
        p = (1.0 + ge) / (1.0 + n_sh)

    out = {
        'r2': r2.astype(np.float64),
        'r': np.full(C, np.nan),
        'r2_folds': r2_folds,
        'lam': lam_all,
        'null_r2': null_r2,
        'null_r2_median': (np.nanmedian(null_r2, axis=0) if n_sh
                           else np.full(C, np.nan)),
        'p': p,
        'n_features': int(n_features),
        'n_test': n_used,
    }
    out['r2_corrected'] = out['r2'] - out['null_r2_median'] if n_sh else out['r2']
    if return_oof:
        out['oof'] = oof
        m = used
        out['r'] = _corr_cols(Y[m], oof[m])
    return out


def _apply_lags(Z, lags):
    """ Stack lagged copies of a feature matrix.

    Lag L takes the value from L frames earlier, so a positive lag is the
    physiological direction: the scene leads the response.  Wrapped rows are
    zeroed; the driver separately masks samples whose lagged source frames were
    not usable, so those zeros never reach a fit.

    Parameters
    ----------
    Z : (n, F) float
    lags : sequence of int

    Returns
    -------
    (n, F * len(lags)) float32
    """

    out = []
    n = Z.shape[0]
    for L in lags:
        sh = np.roll(Z, int(L), axis=0)
        if L > 0:
            sh[:L] = 0.0
        elif L < 0:
            sh[L:] = 0.0
        out.append(sh)
    return np.concatenate(out, axis=1).astype(np.float32)


def _lag_valid(use, lags):
    """ Samples whose every lagged source frame is itself usable. """

    ok = np.ones(len(use), dtype=bool)
    for L in lags:
        sh = np.roll(use, int(L))
        if L > 0:
            sh[:L] = False
        elif L < 0:
            sh[L:] = False
        ok &= sh
    return ok


class SceneFeatures:
    """ Scene pixels to lagged PCA scores, with the basis fit on train rows.

    The basis has to be refit inside every fold.  Twenty components out of ~900
    pixels is not a detail: at ~35 s of stimulus autocorrelation the recording
    holds tens of independent looks at the scene, so anything much richer is
    fitting the fold, not the cell.

    Parameters
    ----------
    S : (n_2p, h, w) float32
        Scene binned onto the 2P timebase.
    pix_mask : (h, w) bool or None
        Pixels to keep.  None keeps all.
    n_pcs : int
    lags : sequence of int
    """

    def __init__(self, S, pix_mask=None, n_pcs=20, lags=DEFAULT_LAGS):
        """ Flatten scene stack and keep only masked pixels. """

        self.shape = S.shape[1:]
        flat = S.reshape(S.shape[0], -1)
        self.keep = (np.ones(flat.shape[1], dtype=bool) if pix_mask is None
                     else np.asarray(pix_mask, dtype=bool).ravel())
        self.P = flat[:, self.keep].astype(np.float64)
        self.n_pcs = int(n_pcs)
        self.lags = list(lags)
        self.last_evals = None
        self.last_components = None

    def build(self, train_rows):
        """ Fit the PCA basis on train_rows, return the lagged design matrix. """

        Xp = self.P
        mu = Xp[train_rows].mean(axis=0)
        Xc = Xp - mu
        A = Xc[train_rows]

        # Covariance eigendecomposition rather than an SVD of A: the pixel
        # count is a few hundred and the frame count is thousands, so the
        # (P, P) route is the cheap one and hands back the whole spectrum for
        # the participation ratio.
        cov = (A.T @ A) / max(A.shape[0] - 1, 1)
        w, V = np.linalg.eigh(cov)
        order = np.argsort(w)[::-1]
        w, V = w[order], V[:, order]
        self.last_evals = w
        k = min(self.n_pcs, V.shape[1])
        comps = V[:, :k]
        self.last_components = comps

        Z = Xc @ comps
        sd = Z[train_rows].std(axis=0)
        Z = Z / np.where(sd > 0, sd, 1.0)
        return _apply_lags(Z, self.lags)

    def participation_ratio(self):
        """ (sum w)^2 / sum(w^2) of the last fitted spectrum: effective rank. """

        if self.last_evals is None:
            return np.nan
        w = np.clip(self.last_evals, 0.0, None)
        return float(w.sum() ** 2 / np.sum(w ** 2)) if w.sum() > 0 else np.nan


class BehaviorFeatures:
    """ Head and eye channels with temporal lags, as the nuisance baseline.

    Deliberately excludes anything derived from the pillar's position
    (`egocentric`, `retinocentric`, `dist_to_pillar`, `pillar_size`): those are
    the scene by another name, and the point of this model is to be a scene-free
    account of the same frames.

    Parameters
    ----------
    chans : dict of str to (n_2p,) float
    lags : sequence of int
    """

    def __init__(self, chans, lags=DEFAULT_LAGS):
        """ Stack behavior channels into one matrix, NaNs zeroed. """

        self.names = sorted(chans.keys())
        self.M = np.stack([np.asarray(chans[k], dtype=np.float64)
                           for k in self.names], axis=1)
        self.M[~np.isfinite(self.M)] = 0.0
        self.lags = list(lags)

    def build(self, train_rows):
        """ Z-score on train_rows, then lag. """

        mu = self.M[train_rows].mean(axis=0)
        sd = self.M[train_rows].std(axis=0)
        Z = (self.M - mu) / np.where(sd > 0, sd, 1.0)
        return _apply_lags(Z, self.lags)


class ConcatFeatures:
    """ Two featurizers side by side, for the scene-plus-behavior model. """

    def __init__(self, a, b):
        """ Hold the two featurizers to concatenate. """

        self.a, self.b = a, b

    def build(self, train_rows):
        """ Horizontal stack of both design matrices. """

        return np.concatenate([self.a.build(train_rows),
                               self.b.build(train_rows)], axis=1)


def load_behavior_channels(h5_path, n_2p):
    """ Pull the self-motion and eye channels off a preproc h5, on the 2P clock.

    Missing keys are skipped rather than filled: sessions differ in which IMU
    channels survived, and a column of zeros would only dilute the baseline.

    Returns
    -------
    chans : dict of str to (n_2p,) float
    """

    chans = {}
    with h5py.File(h5_path, 'r') as f:
        def _get(key):
            """ Read one channel, padded or cut to 2P frame count; None if absent. """

            if key not in f:
                return None
            v = np.asarray(f[key][:], dtype=float)
            if len(v) < n_2p:
                v = np.concatenate([v, np.full(n_2p - len(v), v[-1] if len(v) else 0.0)])
            return v[:n_2p]

        yaw = _get('head_yaw_deg')
        if yaw is not None:
            chans['yaw_sin'] = np.sin(np.radians(yaw))
            chans['yaw_cos'] = np.cos(np.radians(yaw))
        for key, name in [('pitch_twop_interp', 'pitch'),
                          ('roll_twop_interp', 'roll'),
                          ('gyro_x_twop_interp', 'gyro_x'),
                          ('gyro_y_twop_interp', 'gyro_y'),
                          ('gyro_z_twop_interp', 'gyro_z'),
                          ('speed', 'speed'),
                          ('pupil_from_head', 'pupil_from_head')]:
            v = _get(key)
            if v is not None:
                chans[name] = v
        for key, name in [('theta_interp', 'theta'), ('phi_interp', 'phi')]:
            v = _get(key)
            if v is None:
                continue
            # Theta/phi are stored in radians upstream; degrees only matter for
            # readability since everything is z-scored, but keep them consistent.
            if np.nanmax(np.abs(v)) < 2.0:
                v = np.degrees(v)
            chans[name] = v
            chans['d' + name] = np.gradient(np.nan_to_num(v, nan=0.0))
    return chans


def load_scene(npz_path, twopT, bin_px=4, verbose=True):
    """ Load one retinal-image npz and bin it onto the 2P frame times.

    Returns
    -------
    S : (n_2p, h, w) float32
    coverage : (n_2p,) bool
        2P frames that got at least one rendered eye frame.
    meta : dict
        Render flags worth carrying into the output and printing as a warning.
    """

    if verbose:
        print('Loading {}...'.format(os.path.basename(npz_path)))
    npz = np.load(npz_path, allow_pickle=False)
    ri = npz['retinal_images']
    eyeT = npz['eyeT']
    rendered = npz['rendered'] if 'rendered' in npz.files else None

    meta = {}
    for k in ('fixed_eye', 'fixed_pupil', 'walls', 'worldcam',
              'anatomical_eye_angle_deg', 'socket_elev_deg', 'vor_gain'):
        if k in npz.files:
            meta[k] = float(np.asarray(npz[k]).astype(float).ravel()[0])
    if verbose:
        print('  Movie {}, {:.1f}% of eye frames rendered, socket az {} el {}, '
              'fixed_eye={} fixed_pupil={}...'.format(
                  ri.shape,
                  100.0 * (np.mean(rendered) if rendered is not None else 1.0),
                  _fmt(meta.get('anatomical_eye_angle_deg', np.nan), 1),
                  _fmt(meta.get('socket_elev_deg', np.nan), 1),
                  bool(meta.get('fixed_eye', 0)), bool(meta.get('fixed_pupil', 0))))

    S, n_per_bin = bin_scene_to_2p(ri, eyeT, twopT, rendered=rendered,
                                   bin_px=bin_px, verbose=verbose)
    del ri
    npz.close()
    return S, (n_per_bin > 0), meta


def _ablation_regions(S, masks, Y, train_masks, test_masks, block,
                      n_pcs, lags, r2_full, lam_grid, static_nl, verbose):
    """ Refit the model on the complement of each named region.

    Refit rather than knockout: the question is how much of the prediction the
    rest of the scene can still carry, not how much a model already committed to
    one region leans on it.

    Returns
    -------
    out : dict of str to dict
    """

    out = {}
    P = int(masks[list(masks)[0]].size)
    for name, m in masks.items():
        keep = ~np.asarray(m, dtype=bool)
        n_keep = int(keep.sum())
        if n_keep < 16:
            continue
        feat = SceneFeatures(S, keep, n_pcs=min(n_pcs, n_keep), lags=lags)
        res = cv_ridge(feat.build, Y, train_masks, test_masks, block,
                       lam_grid=lam_grid, n_shuffles=0, static_nl=static_nl)
        drop = r2_full - res['r2']
        out[name] = {
            'r2': res['r2'],
            'drop': drop,
            'ablation_index': np.clip(drop / (np.abs(r2_full) + 1e-8), 0.0, 1.0),
            'frac_removed': float(1.0 - n_keep / float(P)),
            'n_px_kept': n_keep,
        }
        if verbose:
            print('    drop {:>18s}  ({:.0f}% of field)  median R2 {}  '
                  'median drop {}'.format(
                      name, 100.0 * out[name]['frac_removed'],
                      _fmt(np.nanmedian(res['r2'])), _fmt(np.nanmedian(drop))))
    return out


def _random_calibration(S, Y, train_masks, test_masks, block, n_pcs, lags,
                        r2_full, lam_grid, static_nl, n_random=3, seed=0,
                        verbose=True):
    """ What removing a given fraction of the field at random costs.

    Half a scene is highly redundant with the whole, so a region's drop is only
    interpretable against this curve.  Random subsets are scattered rather than
    contiguous, which if anything makes them a gentler ablation than a
    contiguous region of the same size -- so a region that only matches this
    curve is not carrying anything special.

    Returns
    -------
    dict with 'fracs', 'r2' (n_fracs, n_random, C), 'drop_median' (n_fracs, C)
    """

    P = S.shape[1] * S.shape[2]
    rng = np.random.default_rng(seed)
    fracs = np.array(RANDOM_CALIB_FRACS, dtype=float)
    r2 = np.full((len(fracs), int(n_random), Y.shape[1]), np.nan)
    for fi, fr in enumerate(fracs):
        n_keep = max(16, int(round((1.0 - fr) * P)))
        for rep in range(int(n_random)):
            keep = np.zeros(P, dtype=bool)
            keep[rng.choice(P, size=n_keep, replace=False)] = True
            feat = SceneFeatures(S, keep.reshape(S.shape[1:]),
                                 n_pcs=min(n_pcs, n_keep), lags=lags)
            res = cv_ridge(feat.build, Y, train_masks, test_masks, block,
                           lam_grid=lam_grid, n_shuffles=0, static_nl=static_nl)
            r2[fi, rep] = res['r2']
        if verbose:
            print('    random {:.0f}% removed: median R2 {}'.format(
                100 * fr, _fmt(np.nanmedian(r2[fi]))))
    drop_median = r2_full[None, :] - np.nanmedian(r2, axis=1)
    return {'fracs': fracs, 'r2': r2, 'drop_median': drop_median}


def _excess_drop(ab, calib, r2_full):
    """ Each region's drop above what removing that much field at random costs.

    Adds 'excess_drop' in place.  A positive value means the region carried
    something a random subset of the same size did not.
    """

    fr = np.concatenate([[0.0], calib['fracs']])
    for name, d in ab.items():
        curve = np.concatenate([np.zeros((1, len(r2_full))),
                                calib['drop_median']], axis=0)
        expect = np.array([np.interp(d['frac_removed'], fr, curve[:, c])
                           for c in range(len(r2_full))])
        d['expected_drop'] = expect
        d['excess_drop'] = d['drop'] - expect
    return ab


def run_condition(label, S, Y, twopT, use, dark_use, train_masks, test_masks,
                  block, beh_feat, n_pcs=20, lags=DEFAULT_LAGS, n_shuffles=50,
                  lam_grid=LAM_GRID, static_nl=False, do_ablations=True,
                  n_random=3, n_tiles=4, bin_px=4, seed=0, min_shift=200,
                  n_folds=5, fold_block_s=120.0, embargo_s=40.0, verbose=True):
    """ Every model arm for one scene reconstruction.

    Arms
    ----
    scene
        Causal lags, full field.  The headline model.
    scene_acausal
        The same lags negated.  A scene model that predicts as well from the
        future is a behavioral-epoch signature, not a visual response, so this
        is the control that decides whether anything here is worth reporting.
    scene_plus_behavior
        Scene and self-motion together; the difference against the
        behavior-only fit is the scene's unique variance.
    scene_dark
        Fit on dark frames only, where the reconstruction still renders a scene
        the animal cannot see.  Should collapse if the model is visual.
    ablation, random_calib
        Region drops and the matched-area calibration curve.

    Returns
    -------
    dict of arm name to result dict
    """

    out = {}
    feat = SceneFeatures(S, None, n_pcs=n_pcs, lags=lags)

    if verbose:
        print('  [{}] Scene, {} PCs x {} lags...'.format(label, n_pcs, len(lags)))
    out['scene'] = cv_ridge(feat.build, Y, train_masks, test_masks, block,
                            lam_grid=lam_grid, n_shuffles=n_shuffles,
                            min_shift=min_shift, seed=seed, static_nl=static_nl,
                            return_oof=True, verbose=verbose)
    out['scene_pca'] = {
        'participation_ratio': feat.participation_ratio(),
        'evals': (feat.last_evals[:60] if feat.last_evals is not None
                  else np.array([])),
        'var_explained': (float(np.sum(np.clip(feat.last_evals, 0, None)[:n_pcs])
                                / np.sum(np.clip(feat.last_evals, 0, None)))
                          if feat.last_evals is not None else np.nan),
    }
    r2_full = out['scene']['r2']
    if verbose:
        _pass = (np.nan_to_num(out['scene']['p'], nan=1.0) < 0.05) & (r2_full > 0)
        print('    median R2 {}   median null R2 {}   cells beating the null: '
              '{}/{} ({} of them with R2 > 0)'.format(
                  _fmt(np.nanmedian(r2_full)),
                  _fmt(np.nanmedian(out['scene']['null_r2_median'])),
                  int(np.nansum(np.nan_to_num(out['scene']['p'], nan=1.0) < 0.05)),
                  len(r2_full), int(_pass.sum())))

    if verbose:
        print('  [{}] Scene, acausal control...'.format(label))
    feat_ac = SceneFeatures(S, None, n_pcs=n_pcs, lags=[-L for L in lags])
    out['scene_acausal'] = cv_ridge(feat_ac.build, Y, train_masks, test_masks,
                                    block, lam_grid=lam_grid,
                                    n_shuffles=n_shuffles, min_shift=min_shift,
                                    seed=seed, static_nl=static_nl)
    if verbose:
        print('    median R2 {} (causal {})'.format(
            _fmt(np.nanmedian(out['scene_acausal']['r2'])), _fmt(np.nanmedian(r2_full))))

    if beh_feat is not None:
        if verbose:
            print('  [{}] Scene + behavior...'.format(label))
        both = ConcatFeatures(SceneFeatures(S, None, n_pcs=n_pcs, lags=lags),
                              beh_feat)
        out['scene_plus_behavior'] = cv_ridge(
            both.build, Y, train_masks, test_masks, block, lam_grid=lam_grid,
            n_shuffles=0, static_nl=static_nl)

    if dark_use is not None and dark_use.sum() > 400:
        if verbose:
            print('  [{}] Scene, dark frames only ({} frames)...'.format(
                label, int(dark_use.sum())))
        _, tr_d, te_d = block_folds(twopT, dark_use, n_folds=n_folds,
                                    block_s=fold_block_s, embargo_s=embargo_s,
                                    seed=seed)
        out['scene_dark'] = cv_ridge(feat.build, Y, tr_d, te_d, block,
                                     lam_grid=lam_grid, n_shuffles=n_shuffles,
                                     min_shift=min_shift, seed=seed,
                                     static_nl=static_nl)
        if verbose:
            print('    median R2 {} (light {})'.format(
                _fmt(np.nanmedian(out['scene_dark']['r2'])), _fmt(np.nanmedian(r2_full))))

    if do_ablations:
        masks, az, el = region_masks(S.shape[1], S.shape[2], bin_px=bin_px,
                                     n_tiles=n_tiles)
        if verbose:
            print('  [{}] random-area calibration...'.format(label))
        calib = _random_calibration(S, Y, train_masks, test_masks, block, n_pcs,
                                    lags, r2_full, lam_grid, static_nl,
                                    n_random=n_random, seed=seed, verbose=verbose)
        if verbose:
            print('  [{}] {} region ablations...'.format(label, len(masks)))
        ab = _ablation_regions(S, masks, Y, train_masks, test_masks, block,
                               n_pcs, lags, r2_full, lam_grid, static_nl, verbose)
        out['ablation'] = _excess_drop(ab, calib, r2_full)
        out['random_calib'] = calib
        out['az'] = az
        out['el'] = el

    out['scene_mean_image'] = S[use].mean(axis=0) if use.any() else S.mean(axis=0)
    out['scene_sd_image'] = S[use].std(axis=0) if use.any() else S.std(axis=0)
    return out


def compare_conditions(res_a, res_b, label_a, label_b, alpha=0.05,
                       n_boot=2000, seed=0):
    """ Paired ego-vs-retino statistics across cells.

    Both arms saw the same frames, folds, basis size and penalty grid, so the
    per-cell difference is paired and the population question is 1 df.  Cells
    are only counted where at least one arm beat its own circular-shift null;
    a cell neither arm predicts contributes noise to a difference score and
    nothing to the question.

    Parameters
    ----------
    res_a, res_b : dict
        `run_condition` outputs.
    alpha : float
        Per-cell significance against the shift null.

    Returns
    -------
    dict of comparison statistics
    """

    a, b = res_a['scene'], res_b['scene']
    r2a, r2b = a['r2_corrected'], b['r2_corrected']
    sig = ((np.nan_to_num(a['p'], nan=1.0) < alpha) |
           (np.nan_to_num(b['p'], nan=1.0) < alpha))

    dr2 = r2b - r2a
    pref = dr2 / (np.abs(r2a) + np.abs(r2b) + 1e-8)

    d_sel = dr2[sig & np.isfinite(dr2)]
    stat = {
        'label_a': label_a,
        'label_b': label_b,
        'alpha': float(alpha),
        'dr2': dr2,
        'pref_index': pref,
        'sig_either': sig,
        'n_sig_a': int(np.nansum(np.nan_to_num(a['p'], nan=1.0) < alpha)),
        'n_sig_b': int(np.nansum(np.nan_to_num(b['p'], nan=1.0) < alpha)),
        'n_cells': int(len(r2a)),
        'n_sig_either': int(sig.sum()),
        'median_r2_a': float(np.nanmedian(r2a)),
        'median_r2_b': float(np.nanmedian(r2b)),
        'median_dr2_sig': float(np.nanmedian(d_sel)) if len(d_sel) else np.nan,
        'median_dr2_all': float(np.nanmedian(dr2[np.isfinite(dr2)])),
        'frac_b_wins_sig': (float(np.mean(d_sel > 0)) if len(d_sel) else np.nan),
    }

    if len(d_sel) >= 8:
        try:
            w = stats.wilcoxon(d_sel, alternative='two-sided')
            stat['wilcoxon_stat'] = float(w.statistic)
            stat['wilcoxon_p'] = float(w.pvalue)
        except ValueError:
            stat['wilcoxon_stat'] = np.nan
            stat['wilcoxon_p'] = np.nan
        rng = np.random.default_rng(seed)
        boot = np.array([np.median(rng.choice(d_sel, size=len(d_sel), replace=True))
                         for _ in range(int(n_boot))])
        stat['dr2_ci_lo'] = float(np.percentile(boot, 2.5))
        stat['dr2_ci_hi'] = float(np.percentile(boot, 97.5))
    else:
        stat['wilcoxon_p'] = np.nan
        stat['dr2_ci_lo'] = np.nan
        stat['dr2_ci_hi'] = np.nan

    # Per-fold sign consistency: a real preference should hold in most folds,
    # not ride on one stretch of the session.
    fa, fb = a['r2_folds'], b['r2_folds']
    with np.errstate(invalid='ignore'):
        sgn = np.sign(fb - fa)
    stat['fold_sign_consistency'] = np.nanmean(sgn == np.sign(dr2)[None, :], axis=0)
    stat['median_fold_consistency_sig'] = float(
        np.nanmedian(stat['fold_sign_consistency'][sig])) if sig.any() else np.nan
    return stat


def _unique_scene_variance(cond, beh):
    """ Scene variance over and above the behavior-only fit, per cell. """

    if beh is None or 'scene_plus_behavior' not in cond:
        return None
    return {
        'r2_behavior': beh['r2'],
        'r2_scene_plus_behavior': cond['scene_plus_behavior']['r2'],
        'unique_scene': cond['scene_plus_behavior']['r2'] - beh['r2'],
        'unique_behavior': cond['scene_plus_behavior']['r2'] - cond['scene']['r2'],
    }


def ffNLE_scene(ego_npz, retino_npz=None, h5_path=None, out_h5=None,
                spike_key='norm_spikes', smooth_s=1.2, bin_px=4, n_pcs=20,
                lags=DEFAULT_LAGS, n_folds=5, fold_block_s=120.0,
                embargo_s=40.0, n_shuffles=50, speed_thresh=SPEED_THRESH_CM_S,
                light_only=True, alpha=0.05, max_cells=None, seed=0,
                do_ablations=True, n_random=3, n_tiles=4, static_nl=False,
                use_behavior=True, make_pdf=True, cell_pages=0, verbose=True):
    """ Fit the scene encoding model on two reconstructions and compare them.

    Parameters
    ----------
    ego_npz, retino_npz : str
        `<prefix>_retinal_images*.npz` files.  Both optional individually but at
        least one is needed; the comparison only runs when both are given.
    h5_path : str or None
        `<prefix>_preproc.h5`; derived from ego_npz when omitted.
    spike_key : str
        Target trace.  `norm_spikes` by default -- `norm_dFF` gives higher R^2
        for the wrong reason, since the calcium autocorrelation alone predicts
        itself across a fold boundary.
    smooth_s : float
        Half-width of the triangle kernel applied to the target, seconds.  0
        turns it off.  Raw deconvolved spikes at 7.5 Hz are dominated by shot
        noise and pin every R^2 at zero.
    bin_px : int
        Scene binning.  4 gives 30x30 (about 4 deg per bin).
    n_pcs : int
        Scene basis size.  Keep it small: the stimulus has tens of independent
        samples, not thousands.
    lags : sequence of int
        Scene-to-response offsets in 2P frames.
    fold_block_s, embargo_s : float
        Block length and purge width for the cross-validation.

    Returns
    -------
    res : dict
        Everything that gets written to the output h5.
    """

    npz_paths = [('egocentric', ego_npz), ('retinocentric', retino_npz)]
    npz_paths = [(k, v) for k, v in npz_paths if v]
    if not npz_paths:
        raise ValueError('Need at least one npz path.')

    ref = npz_paths[0][1]
    stem, sep, suffix = ref.rpartition('_retinal_images')
    if h5_path is None:
        h5_path = (stem + '_preproc.h5' if sep else ref.replace('.npz', '_preproc.h5'))
    if h5_path.endswith('.npz') or not os.path.isfile(h5_path):
        raise FileNotFoundError(
            'Could not find the preproc file for {}; looked for {}. Pass '
            'h5_path explicitly.'.format(os.path.basename(ref), h5_path))
    if out_h5 is None:
        out_h5 = (stem + '_ffNLE_scene.h5' if sep
                  else ref.replace('.npz', '_ffNLE_scene.h5'))

    print('Preproc: {}'.format(os.path.basename(h5_path)))
    with h5py.File(h5_path, 'r') as f:
        if spike_key not in f:
            raise KeyError('{!r} not in {}; spike-like keys present: {}'.format(
                spike_key, os.path.basename(h5_path),
                [k for k in f.keys() if 'spk' in k or 'spike' in k or 'dFF' in k]))
        rate = np.asarray(f[spike_key][:], dtype=float)
        twopT = np.asarray(f['twopT'][:], dtype=float)
        speed = np.asarray(f['speed'][:], dtype=float) if 'speed' in f else None
        ltdk = (np.asarray(f['ltdk_state_vec'][:]).astype(bool)
                if 'ltdk_state_vec' in f else None)

    # The permutation p can never go below 1/(n_shuffles + 1), so too few
    # shuffles silently makes every cell fail no matter how well it fits.
    if n_shuffles > 0 and 1.0 / (n_shuffles + 1.0) > alpha:
        print('  WARNING: {} shuffles floors the p-value at {:.3f}, above '
              'alpha {:g} -- no cell can pass. Use at least {} shuffles.'.format(
                  n_shuffles, 1.0 / (n_shuffles + 1.0), alpha,
                  int(np.ceil(1.0 / alpha))))

    n_fr = min(rate.shape[1], len(twopT))
    rate, twopT = rate[:, :n_fr], twopT[:n_fr]
    if max_cells is not None:
        rate = rate[:int(max_cells)]
    Y = rate.T.copy()
    Y[~np.isfinite(Y)] = 0.0
    # Z-score per cell so R^2 and the ridge penalty mean the same thing across
    # cells with very different firing rates.
    Y = (Y - Y.mean(axis=0)) / np.where(Y.std(axis=0) > 0, Y.std(axis=0), 1.0)
    n_cells = Y.shape[1]
    dt_2p = float(np.nanmedian(np.diff(twopT)))
    if smooth_s and smooth_s > 0:
        Y = _triangle_smooth(Y, smooth_s, dt_2p)
        Y = (Y - Y.mean(axis=0)) / np.where(Y.std(axis=0) > 0, Y.std(axis=0), 1.0)
    print('  {} cells, {} 2P frames at {:.1f} Hz, target {!r} '
          '(triangle smoothed, half-width {:g} s)'.format(
              n_cells, n_fr, 1.0 / dt_2p, spike_key, smooth_s))

    scenes, coverage, metas = {}, {}, {}
    for label, path in npz_paths:
        S, cov, meta = load_scene(path, twopT, bin_px=bin_px, verbose=verbose)
        scenes[label], coverage[label], metas[label] = S, cov, meta
        metas[label]['path'] = os.path.basename(path)
    if metas.get('egocentric', {}).get('fixed_eye', 0):
        print('  NOTE: the egocentric npz was rendered with fixed_eye, which '
              'also zeroes the anatomical socket -- so it differs from the '
              'retinocentric render in WHERE it looks as well as in whether '
              'the eye moves. For a clean 1-df control render the ego arm with '
              '--fixed_pupil instead.')

    # Shared frame selection.  Both arms must see exactly the same frames or the
    # paired comparison is not paired.
    use = np.ones(n_fr, dtype=bool)
    for label in scenes:
        use &= coverage[label]
    use &= np.isfinite(rate).all(axis=0)
    if speed is not None and speed_thresh > 0:
        sp = np.asarray(speed, dtype=float)
        if len(sp) < n_fr:
            sp = np.concatenate([sp, np.full(n_fr - len(sp), sp[-1] if len(sp) else 0.0)])
        moving = sp[:n_fr] > speed_thresh
        use &= moving
        print('  Motion filter (>{:g} cm/s) keeps {:.1f}% of frames...'.format(
            speed_thresh, 100.0 * moving.mean()))
    dark_use = None
    if ltdk is not None and len(ltdk) >= n_fr:
        if light_only:
            dark_use = use & ~ltdk[:n_fr]
            use = use & ltdk[:n_fr]
            print('  Light-only: {} light frames used, {} dark frames held for '
                  'the dark control...'.format(int(use.sum()), int(dark_use.sum())))
    _all_lags = list(lags) + [-L for L in lags]
    use &= _lag_valid(use.copy(), _all_lags)
    if dark_use is not None:
        dark_use &= _lag_valid(dark_use.copy(), _all_lags)
    print('  {}/{} 2P frames usable ({:.1f} min)...'.format(
        int(use.sum()), n_fr, use.sum() * dt_2p / 60.0))
    if use.sum() < 300:
        raise RuntimeError(
            'Only {} usable frames -- too few to cross-validate. Check that '
            'the reconstruction covers this recording.'.format(int(use.sum())))

    block = np.floor((twopT - twopT[0]) / float(fold_block_s)).astype(int)
    fold_id, train_masks, test_masks = block_folds(
        twopT, use, n_folds=n_folds, block_s=fold_block_s,
        embargo_s=embargo_s, seed=seed)
    print('  {} folds of {:.0f} s blocks, {:.0f} s embargo -- '
          'median {} train / {} test frames per fold...'.format(
              n_folds, fold_block_s, embargo_s,
              int(np.median([m.sum() for m in train_masks])),
              int(np.median([m.sum() for m in test_masks]))))

    beh_feat, beh_res = None, None
    if use_behavior:
        chans = load_behavior_channels(h5_path, n_fr)
        if chans:
            beh_feat = BehaviorFeatures(chans, lags=lags)
            print('Fitting the behavior-only baseline ({} channels)...'.format(
                len(chans)))
            beh_res = cv_ridge(beh_feat.build, Y, train_masks, test_masks, block,
                               n_shuffles=n_shuffles, min_shift=int(round(60.0 / dt_2p)),
                               seed=seed, static_nl=static_nl)
            print('  median R2 {}, cells p<{}: {}/{}'.format(
                _fmt(np.nanmedian(beh_res['r2'])), alpha,
                int(np.nansum(beh_res['p'] < alpha)), n_cells))
        else:
            print('  No behavior channels found in the h5 -- skipping the '
                  'baseline. The scene R2 then has no nuisance control.')

    conditions = {}
    for label in scenes:
        print('Fitting {}...'.format(label))
        t0 = time.time()
        conditions[label] = run_condition(
            label, scenes[label], Y, twopT, use, dark_use, train_masks,
            test_masks, block, beh_feat, n_pcs=n_pcs, lags=lags,
            n_shuffles=n_shuffles, static_nl=static_nl,
            do_ablations=do_ablations, n_random=n_random, n_tiles=n_tiles,
            bin_px=bin_px, seed=seed, min_shift=int(round(60.0 / dt_2p)),
            n_folds=n_folds, fold_block_s=fold_block_s, embargo_s=embargo_s,
            verbose=verbose)
        uv = _unique_scene_variance(conditions[label], beh_res)
        if uv is not None:
            conditions[label]['variance_partition'] = uv
            print('    unique scene variance over behavior: median {}'.format(
                _fmt(np.nanmedian(uv['unique_scene']))))
        print('  Done in {:.0f}s'.format(time.time() - t0))

    comparison = None
    if 'egocentric' in conditions and 'retinocentric' in conditions:
        comparison = compare_conditions(
            conditions['egocentric'], conditions['retinocentric'],
            'egocentric', 'retinocentric', alpha=alpha, seed=seed)
        _fw = comparison['frac_b_wins_sig']
        _nw = (int(round(_fw * comparison['n_sig_either']))
               if np.isfinite(_fw) else 0)
        print('Retinocentric minus egocentric: median dR2 {} '
              '(95% CI {} to {}), Wilcoxon p {}, {}/{} cells favor '
              'retinocentric...'.format(
                  _fmt(comparison['median_dr2_sig'], 4),
                  _fmt(comparison['dr2_ci_lo'], 4), _fmt(comparison['dr2_ci_hi'], 4),
                  _fmt(comparison['wilcoxon_p'], 4),
                  _nw, comparison['n_sig_either']))
        if comparison['n_sig_either'] == 0:
            print('  No cell beat its own circular-shift null in either arm, so '
                  'there is nothing to compare. Check the acausal and dark arms '
                  'in the PDF before concluding anything about the eye.')

    res = {
        'meta': {
            'h5_path': h5_path,
            'spike_key': spike_key,
            'smooth_s': float(smooth_s),
            'bin_px': int(bin_px),
            'n_pcs': int(n_pcs),
            'lags': np.asarray(lags, dtype=int),
            'n_folds': int(n_folds),
            'fold_block_s': float(fold_block_s),
            'embargo_s': float(embargo_s),
            'n_shuffles': int(n_shuffles),
            'speed_thresh': float(speed_thresh),
            'light_only': bool(light_only),
            'alpha': float(alpha),
            'static_nl': bool(static_nl),
            'dt_2p': dt_2p,
            'n_cells': int(n_cells),
            'n_frames_used': int(use.sum()),
            'minutes_used': float(use.sum() * dt_2p / 60.0),
            'npz': {k: v for k, v in metas.items()},
        },
        'use': use,
        'fold_id': fold_id,
        'twopT': twopT,
        'conditions': conditions,
        'behavior': beh_res,
    }
    if comparison is not None:
        res['comparison'] = comparison

    print_verdict(res)

    _save(res, out_h5)
    print('Saved {}'.format(out_h5))
    _save_json(res, out_h5.replace('.h5', '.json'))
    print('Saved {}'.format(out_h5.replace('.h5', '.json')))

    if make_pdf:
        pdf_path = out_h5.replace('.h5', '.pdf')
        make_summary_pdf(res, scenes, pdf_path, max_cell_pages=cell_pages,
                         Y_obs=Y)
        print('Saved {}'.format(pdf_path))
    return res


def print_verdict(res):
    """ Say in plain terms what the control arms decided, before anyone squints.

    The scene R^2 on its own settles nothing.  A model that predicts the same
    from the future, or in the dark, or no better than the self-motion channels,
    is reporting a behavioral-epoch signature -- and the ego-vs-retino contrast
    on top of one is meaningless.
    """

    alpha = res['meta']['alpha']
    print('')
    print('Verdict')
    print('-------')
    for label, c in res['conditions'].items():
        sig = _sig_mask(c, alpha)
        pick = sig if sig.any() else np.ones(len(sig), dtype=bool)
        r2 = np.nanmedian(c['scene']['r2'][pick])
        ac = np.nanmedian(c['scene_acausal']['r2'][pick])
        print('  {}:'.format(label))
        print('    Scene R2 {}  vs null {}  (medians over the cells the scene '
              'model predicts) -- {}/{} beat their own shift null with '
              'R2 > 0'.format(
                  _fmt(r2), _fmt(np.nanmedian(c['scene']['null_r2_median'][pick])),
                  int(np.sum(sig & (c['scene']['r2'] > 0))), len(sig)))
        print('    Acausal R2 {}  -- {}'.format(
            _fmt(ac),
            'causal wins, consistent with a visual response'
            if r2 > ac + 0.005 else
            'the future predicts as well as the past, so this is not visual '
            'drive'))
        if 'scene_dark' in c:
            dk = np.nanmedian(c['scene_dark']['r2'][pick])
            print('    Dark R2 {}  -- {}'.format(
                _fmt(dk),
                'light beats dark, as a visual response should'
                if r2 > dk + 0.005 else
                'the dark scene predicts as well, so the model is tracking '
                'behavior not light'))
        if 'variance_partition' in c:
            uq = np.nanmedian(c['variance_partition']['unique_scene'][pick])
            print('    unique over behavior {}  -- {}'.format(
                _fmt(uq),
                'the scene adds something self-motion cannot'
                if uq > 0.005 else
                'self-motion already accounts for it'))

    if 'comparison' in res:
        cm = res['comparison']
        print('  Retinocentric minus egocentric: median dR2 {} '
              '(95% CI {} to {}), Wilcoxon p {}'.format(
                  _fmt(cm['median_dr2_sig'], 4), _fmt(cm['dr2_ci_lo'], 4),
                  _fmt(cm['dr2_ci_hi'], 4), _fmt(cm['wilcoxon_p'], 4)))
        if cm['n_sig_either'] == 0:
            print('    Nothing to compare -- no cell was predicted by either arm.')
        elif not np.isfinite(cm['wilcoxon_p']) or cm['wilcoxon_p'] > 0.05:
            print('    No reliable difference: the eye signal buys nothing here.')
        else:
            print('    {} of {} cells favor retinocentric, folds agree {}% of '
                  'the time.'.format(
                      int(round(cm['frac_b_wins_sig'] * cm['n_sig_either'])),
                      cm['n_sig_either'],
                      _fmt(100 * cm['median_fold_consistency_sig'], 0)))
    print('')


def _sanitize(obj, drop_keys=('oof',)):
    """ Recursively drop Nones and unsaveable values so write_h5 can take it. """

    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in drop_keys or v is None:
                continue
            sv = _sanitize(v, drop_keys)
            if sv is not None:
                out[str(k)] = sv
        return out
    if isinstance(obj, (np.ndarray, np.number, np.bool_, int, float, str, bytes, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return np.asarray(obj)
    return None


def _save(res, out_h5):
    """ Write the result tree to HDF5, minus the out-of-fold prediction matrices. """

    write_h5(out_h5, _sanitize(res))


def _save_json(res, out_json):
    """ Write a scalar summary next to the h5, for reading without h5py. """

    def _med(x):
        """ NaN-median as float, None for empty input. """

        x = np.asarray(x, dtype=float)
        return float(np.nanmedian(x)) if x.size else None

    meta = dict(res['meta'])
    meta['lags'] = [int(v) for v in np.asarray(meta['lags']).ravel()]
    out = {'meta': meta, 'conditions': {}}

    if res.get('behavior') is not None:
        out['behavior'] = {
            'median_r2': _med(res['behavior']['r2']),
            'n_sig': int(np.nansum(res['behavior']['p'] < res['meta']['alpha'])),
        }

    for label, c in res['conditions'].items():
        alpha = res['meta']['alpha']
        sig = np.nan_to_num(c['scene']['p'], nan=1.0) < alpha
        d = {
            'median_r2': _med(c['scene']['r2']),
            'median_r2_corrected': _med(c['scene']['r2_corrected']),
            'median_r2_sig_cells': (_med(c['scene']['r2_corrected'][sig])
                                    if sig.any() else None),
            'median_null_r2': _med(c['scene']['null_r2_median']),
            'n_sig': int(sig.sum()),
            'n_sig_positive_r2': int(np.sum(sig & (c['scene']['r2'] > 0))),
            'n_cells': int(len(sig)),
            'median_r2_acausal': _med(c['scene_acausal']['r2']),
            'scene_participation_ratio': float(c['scene_pca']['participation_ratio']),
            'scene_var_explained_by_pcs': float(c['scene_pca']['var_explained']),
            'n_features': int(c['scene']['n_features']),
        }
        if 'scene_dark' in c:
            d['median_r2_dark'] = _med(c['scene_dark']['r2'])
        if 'variance_partition' in c:
            d['median_unique_scene_r2'] = _med(c['variance_partition']['unique_scene'])
            d['median_unique_behavior_r2'] = _med(c['variance_partition']['unique_behavior'])
        if 'ablation' in c:
            d['ablation'] = {}
            for name, ab in c['ablation'].items():
                d['ablation'][name] = {
                    'frac_removed': float(ab['frac_removed']),
                    'median_drop': _med(ab['drop'][sig]) if sig.any() else _med(ab['drop']),
                    'median_excess_drop': (_med(ab['excess_drop'][sig]) if sig.any()
                                           else _med(ab['excess_drop'])),
                    'median_ablation_index': (_med(ab['ablation_index'][sig]) if sig.any()
                                              else _med(ab['ablation_index'])),
                }
        out['conditions'][label] = d

    if 'comparison' in res:
        cm = res['comparison']
        out['comparison'] = {k: (float(v) if isinstance(v, (float, int, np.floating,
                                                            np.integer)) else v)
                             for k, v in cm.items()
                             if not isinstance(v, np.ndarray)}

    with open(out_json, 'w') as fh:
        json.dump(out, fh, indent=2, default=str)


def _sig_mask(cond, alpha):
    """ Cells whose scene fit beat the circular-shift null. """

    return np.nan_to_num(cond['scene']['p'], nan=1.0) < alpha


def _imshow_scene(ax, img, az, el, title, cmap='gray', vmin=None, vmax=None):
    """ Show a scene-shaped map with degree ticks. """

    im = ax.imshow(img, origin='upper', cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[az.min(), az.max(), el.min(), el.max()],
                   aspect='equal')
    ax.set_xlabel('azimuth (deg)')
    ax.set_ylabel('elevation (deg)')
    ax.set_title(title, fontsize=9)
    return im


def make_summary_pdf(res, scenes, out_pdf, max_cell_pages=0, Y_obs=None):
    """ Figures for the whole run: QC, model comparison, controls, ablations.

    Parameters
    ----------
    res : dict
        `ffNLE_scene` output.
    scenes : dict of str to (n_2p, h, w) float32
        Binned scenes, kept out of `res` because of their size.
    max_cell_pages : int
        Per-cell prediction pages to append.  Needs the out-of-fold predictions,
        so only works on a fresh result, not one read back from disk.
    Y_obs : (n_2p, C) float or None
        Observed activity, for the per-cell pages.
    """

    alpha = res['meta']['alpha']
    labels = list(res['conditions'].keys())
    colors = {'egocentric': '#D96459', 'retinocentric': '#03A9F4'}

    with PdfPages(out_pdf) as pdf:
        # Page 1: what went in.
        fig, axs = plt.subplots(len(labels), 3, figsize=(12, 3.6 * len(labels)),
                                squeeze=False)
        for i, label in enumerate(labels):
            c = res['conditions'][label]
            az, el = c.get('az'), c.get('el')
            if az is None:
                h, w = c['scene_mean_image'].shape
                _, az, el = region_masks(h, w, bin_px=res['meta']['bin_px'])
            _imshow_scene(axs[i][0], c['scene_mean_image'], az, el,
                          '{}: mean scene'.format(label))
            _imshow_scene(axs[i][1], c['scene_sd_image'], az, el,
                          '{}: scene SD (what the PCA sees)'.format(label),
                          cmap='magma')
            ev = np.clip(np.asarray(c['scene_pca']['evals'], dtype=float), 0, None)
            if ev.size:
                axs[i][2].semilogy(np.arange(1, len(ev) + 1), ev / ev.sum(), 'o-',
                                   ms=3, color=colors.get(label, 'k'))
            axs[i][2].axvline(res['meta']['n_pcs'], color='0.6', ls='--')
            axs[i][2].set_xlabel('scene PC')
            axs[i][2].set_ylabel('variance fraction')
            axs[i][2].set_title(
                'PR {:.1f}, top {} PCs hold {:.0f}% of scene variance'.format(
                    c['scene_pca']['participation_ratio'], res['meta']['n_pcs'],
                    100 * c['scene_pca']['var_explained']), fontsize=8)
        fig.suptitle('{}  |  {} cells, {} frames ({:.1f} min), target {} '
                     'smoothed {:g} s'.format(
                         os.path.basename(res['meta']['h5_path']),
                         res['meta']['n_cells'], res['meta']['n_frames_used'],
                         res['meta']['minutes_used'], res['meta']['spike_key'],
                         res['meta']['smooth_s']), fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: how well each arm does, and the paired difference.
        fig, axs = plt.subplots(2, 2, figsize=(11, 8))
        ax = axs[0][0]
        _all = np.concatenate([res['conditions'][l]['scene']['r2_corrected']
                               for l in labels])
        _lo, _hi = np.nanpercentile(_all[np.isfinite(_all)], [0.5, 99.5])
        _pad = 0.15 * max(_hi - _lo, 1e-6)
        rng_hist = (_lo - _pad, _hi + _pad)
        for label in labels:
            c = res['conditions'][label]
            ax.hist(c['scene']['r2_corrected'], bins=60, range=rng_hist,
                    histtype='step', lw=1.5,
                    color=colors.get(label, 'k'), label='{} (n sig {})'.format(
                        label, int(_sig_mask(c, alpha).sum())))
            if c['scene']['null_r2'].size:
                ax.hist(np.ravel(c['scene']['null_r2']), bins=60, range=rng_hist,
                        histtype='stepfilled',
                        alpha=0.15, color=colors.get(label, 'k'), density=False,
                        weights=np.full(c['scene']['null_r2'].size,
                                        1.0 / max(res['meta']['n_shuffles'], 1)))
        ax.axvline(0, color='0.5', lw=0.8)
        ax.set_xlabel('cross-validated R2 (shuffle corrected)')
        ax.set_ylabel('cells')
        ax.set_title('Scene model vs its own circular-shift null', fontsize=9)
        ax.legend(fontsize=7)

        ax = axs[0][1]
        if 'comparison' in res:
            cm = res['comparison']
            a = res['conditions']['egocentric']['scene']['r2_corrected']
            b = res['conditions']['retinocentric']['scene']['r2_corrected']
            sig = cm['sig_either']
            ax.scatter(a[~sig], b[~sig], s=6, c='0.8', lw=0)
            ax.scatter(a[sig], b[sig], s=10, c='#2ECC71', lw=0)
            lim = [np.nanmin([a.min(), b.min(), 0]), np.nanmax([a.max(), b.max()])]
            ax.plot(lim, lim, 'k-', lw=0.8)
            ax.set_xlabel('egocentric R2')
            ax.set_ylabel('retinocentric R2')
            ax.set_title('{}/{} cells above the line'.format(
                int(np.nansum(b[sig] > a[sig])), int(sig.sum())), fontsize=9)
        else:
            ax.axis('off')

        ax = axs[1][0]
        if 'comparison' in res:
            cm = res['comparison']
            d = cm['dr2'][cm['sig_either']]
            ax.hist(d[np.isfinite(d)], bins=40, color='#9C27B0', alpha=0.8)
            ax.axvline(0, color='k', lw=0.8)
            ax.axvline(cm['median_dr2_sig'], color='#D96459', lw=1.5)
            ax.set_xlabel('retinocentric minus egocentric R2')
            ax.set_ylabel('cells')
            ax.set_title('median {} (95% CI {} to {}), Wilcoxon p {}'.format(
                _fmt(cm['median_dr2_sig'], 4), _fmt(cm['dr2_ci_lo'], 4),
                _fmt(cm['dr2_ci_hi'], 4), _fmt(cm['wilcoxon_p'], 4)), fontsize=8)
        else:
            ax.axis('off')

        ax = axs[1][1]
        arms, vals, cols = [], [], []
        for label in labels:
            c = res['conditions'][label]
            sig = _sig_mask(c, alpha)
            pick = sig if sig.any() else np.ones(len(sig), dtype=bool)
            for arm, key in [('scene', 'scene'), ('acausal', 'scene_acausal'),
                             ('scene+beh', 'scene_plus_behavior'),
                             ('dark', 'scene_dark')]:
                if key in c:
                    arms.append('{}\n{}'.format(label[:6], arm))
                    vals.append(np.nanmedian(c[key]['r2'][pick]))
                    cols.append(colors.get(label, 'k'))
        if res.get('behavior') is not None:
            arms.append('behavior\nonly')
            vals.append(np.nanmedian(res['behavior']['r2']))
            cols.append('0.5')
        ax.bar(np.arange(len(arms)), vals, color=cols)
        ax.set_xticks(np.arange(len(arms)))
        ax.set_xticklabels(arms, fontsize=6, rotation=45, ha='right')
        ax.axhline(0, color='k', lw=0.8)
        ax.set_ylabel('median R2 over significant cells')
        ax.set_title('Every arm, same frames and folds', fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: the controls that decide whether any of this is visual.
        fig, axs = plt.subplots(2, 2, figsize=(11, 8))
        ax = axs[0][0]
        for label in labels:
            c = res['conditions'][label]
            sig = _sig_mask(c, alpha)
            ax.scatter(c['scene_acausal']['r2'][sig], c['scene']['r2'][sig], s=10,
                       color=colors.get(label, 'k'), lw=0, label=label)
        lim = ax.get_xlim()
        ax.plot(lim, lim, 'k-', lw=0.8)
        ax.set_xlabel('acausal R2 (scene from the future)')
        ax.set_ylabel('causal R2')
        ax.set_title('Above the line means the scene leads the response',
                     fontsize=9)
        ax.legend(fontsize=7)

        ax = axs[0][1]
        any_dark = False
        for label in labels:
            c = res['conditions'][label]
            if 'scene_dark' not in c:
                continue
            any_dark = True
            sig = _sig_mask(c, alpha)
            ax.scatter(c['scene_dark']['r2'][sig], c['scene']['r2'][sig], s=10,
                       color=colors.get(label, 'k'), lw=0, label=label)
        if any_dark:
            lim = ax.get_xlim()
            ax.plot(lim, lim, 'k-', lw=0.8)
            ax.set_xlabel('dark R2 (scene the animal cannot see)')
            ax.set_ylabel('light R2')
            ax.set_title('Above the line means the light matters', fontsize=9)
            ax.legend(fontsize=7)
        else:
            ax.axis('off')
            ax.set_title('No dark epochs in this session', fontsize=9)

        ax = axs[1][0]
        for label in labels:
            c = res['conditions'][label]
            if 'variance_partition' not in c:
                continue
            ax.hist(c['variance_partition']['unique_scene'], bins=50,
                    histtype='step', lw=1.5, color=colors.get(label, 'k'),
                    label=label)
        ax.axvline(0, color='k', lw=0.8)
        ax.set_xlabel('R2(scene + behavior) minus R2(behavior)')
        ax.set_ylabel('cells')
        ax.set_title('Scene variance the self-motion model cannot reach',
                     fontsize=9)
        ax.legend(fontsize=7)

        ax = axs[1][1]
        if 'comparison' in res:
            cm = res['comparison']
            fc = cm['fold_sign_consistency'][cm['sig_either']]
            ax.hist(fc[np.isfinite(fc)], bins=np.linspace(0, 1, 12), color='#FF9800')
            ax.axvline(0.5, color='k', lw=0.8, ls='--')
            ax.set_xlabel('fraction of folds agreeing with the cell dR2 sign')
            ax.set_ylabel('cells')
            ax.set_title('median {} -- a preference riding on one stretch of '
                         'the session sits at 0.5'.format(
                             _fmt(cm['median_fold_consistency_sig'], 2)), fontsize=8)
        else:
            ax.axis('off')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        _ablation_pages(res, pdf, alpha, colors)

        if max_cell_pages > 0:
            _cell_pages(res, pdf, alpha, colors, max_cell_pages, Y_obs)


def _ablation_pages(res, pdf, alpha, colors):
    """ Region-drop heatmap, the random-area calibration, and the tile maps. """

    labels = [l for l in res['conditions'] if 'ablation' in res['conditions'][l]]
    if not labels:
        return

    named = [n for n in res['conditions'][labels[0]]['ablation']
             if not n.startswith('tile_')]
    _ab0 = res['conditions'][labels[0]]['ablation']
    named = sorted(named, key=lambda n: _ab0[n]['frac_removed'])

    fig, axs = plt.subplots(1, 3, figsize=(15, 5.5),
                            gridspec_kw={'width_ratios': [1.1, 1.1, 1.0]})

    for ax, key, ttl in [(axs[0], 'drop', 'raw drop in R2'),
                         (axs[1], 'excess_drop',
                          'drop above a random\nablation of the same size')]:
        M = np.full((len(named), len(labels)), np.nan)
        for j, label in enumerate(labels):
            c = res['conditions'][label]
            sig = _sig_mask(c, alpha)
            pick = sig if sig.any() else np.ones(len(sig), dtype=bool)
            for i, n in enumerate(named):
                M[i, j] = np.nanmedian(c['ablation'][n][key][pick])
        v = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
        im = ax.imshow(M, cmap='coolwarm', vmin=-v, vmax=v, aspect='auto')
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
        ax.set_yticks(np.arange(len(named)))
        ax.set_yticklabels(['{}  ({:.0f}%)'.format(n, 100 * _ab0[n]['frac_removed'])
                            for n in named], fontsize=7)
        ax.set_title(ttl, fontsize=9)
        for i in range(len(named)):
            for j in range(len(labels)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, '{:.3f}'.format(M[i, j]), ha='center',
                            va='center', fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.04)

    ax = axs[2]
    for label in labels:
        c = res['conditions'][label]
        sig = _sig_mask(c, alpha)
        pick = sig if sig.any() else np.ones(len(sig), dtype=bool)
        cal = c['random_calib']
        x = np.concatenate([[0.0], cal['fracs']])
        y = np.concatenate([[0.0], np.nanmedian(cal['drop_median'][:, pick], axis=1)])
        ax.plot(x, y, 'o-', color=colors.get(label, 'k'),
                label='{}: random'.format(label))
        for n in named:
            ab = c['ablation'][n]
            ax.scatter(ab['frac_removed'], np.nanmedian(ab['drop'][pick]),
                       s=18, facecolors='none', edgecolors=colors.get(label, 'k'))
    ax.set_xlabel('fraction of the field removed')
    ax.set_ylabel('median drop in R2')
    ax.set_title('Region drops (open circles) against the\nrandom-area '
                 'calibration (lines)', fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle('Scene ablations: refit on the complement of each region',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close(fig)

    # Tile maps: the coarse importance image the ablations add up to.
    tiles = sorted([n for n in res['conditions'][labels[0]]['ablation']
                    if n.startswith('tile_')])
    if not tiles:
        return
    n_t = int(np.sqrt(len(tiles)))
    fig, axs = plt.subplots(2, len(labels), figsize=(5.4 * len(labels), 9),
                            squeeze=False)
    for j, label in enumerate(labels):
        c = res['conditions'][label]
        sig = _sig_mask(c, alpha)
        pick = sig if sig.any() else np.ones(len(sig), dtype=bool)
        az, el = c['az'], c['el']
        M = np.full((n_t, n_t), np.nan)
        for n in tiles:
            r, cc = n.replace('tile_r', '').split('c')
            M[int(r), int(cc)] = np.nanmedian(c['ablation'][n]['excess_drop'][pick])
        _imshow_scene(axs[0][j], c['scene_mean_image'], az, el,
                      '{}: mean scene'.format(label))
        v = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
        im = axs[1][j].imshow(M, cmap='coolwarm', vmin=-v, vmax=v,
                              extent=[az.min(), az.max(), el.min(), el.max()],
                              aspect='equal', origin='upper')
        axs[1][j].set_xlabel('azimuth (deg)')
        axs[1][j].set_ylabel('elevation (deg)')
        axs[1][j].set_title('{}: excess drop when this tile is removed'.format(label),
                            fontsize=9)
        fig.colorbar(im, ax=axs[1][j], fraction=0.046)
    fig.suptitle('Tile ablation map -- median over cells the scene model '
                 'predicts', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def _cell_pages(res, pdf, alpha, colors, max_pages, Y_obs=None):
    """ Per-cell out-of-fold prediction traces for the best-fit cells. """

    labels = list(res['conditions'].keys())
    have_oof = [l for l in labels if 'oof' in res['conditions'][l]['scene']]
    if not have_oof or Y_obs is None:
        return
    ref = res['conditions'][have_oof[0]]['scene']
    order = np.argsort(-np.nan_to_num(ref['r2_corrected'], nan=-np.inf))
    use = res['use']
    t = res['twopT']

    for cell in order[:int(max_pages)]:
        fig, axs = plt.subplots(2, 1, figsize=(11, 5.5))
        seg = np.flatnonzero(use)[:600]
        ax = axs[0]
        for label in have_oof:
            oof = res['conditions'][label]['scene']['oof']
            ax.plot(t[seg], oof[seg, cell], lw=1.0, color=colors.get(label, 'k'),
                    label='{} R2 {}'.format(
                        label, _fmt(res['conditions'][label]['scene']['r2'][cell])))
        ax.set_ylabel('predicted (z)')
        ax.set_title('cell {} -- held-out prediction, first 600 usable frames'.format(cell),
                     fontsize=9)
        ax.legend(fontsize=7)
        ax.plot(t[seg], Y_obs[seg, cell], lw=0.8, color='0.4', label='observed')
        ax.legend(fontsize=7)

        ax2 = axs[1]
        for label in have_oof:
            ax2.plot(res['conditions'][label]['scene']['r2_folds'][:, cell], 'o-',
                     color=colors.get(label, 'k'), label=label)
        ax2.axhline(0, color='k', lw=0.8)
        ax2.set_xlabel('fold')
        ax2.set_ylabel('held-out R2')
        ax2.legend(fontsize=7)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main():
    """ CLI entry point. """

    p = argparse.ArgumentParser(
        description='Scene encoding model from a reconstructed retinal image, '
                    'egocentric versus retinocentric, with region ablations.')
    p.add_argument('--ego_npz', default=None,
                   help='Egocentric <prefix>_retinal_images*.npz. Ideally the '
                        '--fixed_pupil render, which keeps the anatomical '
                        'socket and only freezes the eye.')
    p.add_argument('--retino_npz', default=None,
                   help='Retinocentric <prefix>_retinal_images*.npz')
    p.add_argument('--h5_path', default=None,
                   help='<prefix>_preproc.h5 (default: derived from ego_npz)')
    p.add_argument('--out_h5', default=None,
                   help='Output path (default: <prefix>_ffNLE_scene.h5)')
    p.add_argument('--spike_key', default='norm_spikes',
                   help='Target trace (default: %(default)s). norm_dFF inflates '
                        'R2 through its own autocorrelation.')
    p.add_argument('--smooth_s', type=float, default=1.2,
                   help='Triangle-kernel half-width applied to the target, '
                        'seconds; 0 disables (default: %(default)s). Matches '
                        'ffNLE. Raw deconvolved spikes at 7.5 Hz are shot noise '
                        'frame to frame and pin every R2 at zero.')
    p.add_argument('--bin_px', type=int, default=4,
                   help='Scene binning; 4 gives 30x30 (default: %(default)s)')
    p.add_argument('--n_pcs', type=int, default=20,
                   help='Scene basis size (default: %(default)s)')
    p.add_argument('--lags', type=int, nargs='+', default=list(DEFAULT_LAGS),
                   help='Scene-to-response lags in 2P frames (default: %(default)s)')
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--fold_block_s', type=float, default=120.0,
                   help='CV block length in seconds. Must exceed the stimulus '
                        'autocorrelation time, ~35 s here (default: %(default)s)')
    p.add_argument('--embargo_s', type=float, default=40.0,
                   help='Training frames this close to a test block are purged '
                        '(default: %(default)s)')
    p.add_argument('--n_shuffles', type=int, default=50,
                   help='Circular shifts for the null (default: %(default)s). '
                        'The smallest attainable p is 1/(n+1).')
    p.add_argument('--speed_thresh', type=float, default=SPEED_THRESH_CM_S,
                   help='Motion filter in cm/s; 0 disables (default: %(default)s)')
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--max_cells', type=int, default=None,
                   help='Only fit the first N cells, for quick tests')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n_random', type=int, default=3,
                   help='Repeats per fraction in the random-area calibration '
                        '(default: %(default)s)')
    p.add_argument('--n_tiles', type=int, default=4,
                   help='Side of the tile grid for the ablation map '
                        '(default: %(default)s)')
    p.add_argument('--no_ablations', action='store_true')
    p.add_argument('--no_behavior', action='store_true',
                   help='Skip the self-motion baseline. Without it the scene R2 '
                        'has no nuisance control.')
    p.add_argument('--static_nl', action='store_true',
                   help='Fit a per-cell monotone output nonlinearity on the '
                        'training rows, making the model LN rather than linear.')
    p.add_argument('--all_light_dark', action='store_true',
                   help='Fit on all frames instead of light only. The default '
                        'is light-only, with the dark frames kept as a control '
                        'arm: the reconstruction renders a scene in darkness '
                        'that the animal cannot see.')
    p.add_argument('--cell_pages', type=int, default=0,
                   help='Append this many per-cell prediction pages to the '
                        'PDF, best-fitting cells first (default: none)')
    p.add_argument('--no_pdf', action='store_true')
    args = p.parse_args()

    if not args.ego_npz and not args.retino_npz:
        p.error('Give at least one of --ego_npz / --retino_npz.')

    ffNLE_scene(
        ego_npz=args.ego_npz, retino_npz=args.retino_npz, h5_path=args.h5_path,
        out_h5=args.out_h5, spike_key=args.spike_key, smooth_s=args.smooth_s,
        bin_px=args.bin_px,
        n_pcs=args.n_pcs, lags=tuple(args.lags), n_folds=args.n_folds,
        fold_block_s=args.fold_block_s, embargo_s=args.embargo_s,
        n_shuffles=args.n_shuffles, speed_thresh=args.speed_thresh,
        light_only=not args.all_light_dark, alpha=args.alpha,
        max_cells=args.max_cells, seed=args.seed,
        do_ablations=not args.no_ablations, n_random=args.n_random,
        n_tiles=args.n_tiles, static_nl=args.static_nl,
        use_behavior=not args.no_behavior, make_pdf=not args.no_pdf,
        cell_pages=args.cell_pages)


if __name__ == '__main__':
    main()



# python -m fm2p.ffNLE_scene --ego_npz /home/dylan/Storage/freely_moving_data/V1PPC/cohort02_recordings/251016_DMM_DMM061_pos18/fm1/251016_DMM_DMM061_fm_01_retinal_images_fixed_pupil_ego.npz --retino_npz /home/dylan/Storage/freely_moving_data/V1PPC/cohort02_recordings/251016_DMM_DMM061_pos18/fm1/251016_DMM_DMM061_fm_01_retinal_images_retinocentric.npz --spike_key norm_spikes_s2p
