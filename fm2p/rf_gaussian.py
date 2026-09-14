# -*- coding: utf-8 -*-
"""Parametric receptive fields from the reconstructed retinal image.

`rf_from_reconstruction` estimates a free value at every pixel: 3600 parameters
at bin_px=2.  The stimulus cannot support that.  The pillar's retinal bearing
has an integrated autocorrelation time of ~35 s, so a 33 min recording carries
roughly **32 independent samples**, and the two halves of a split simply sample
different parts of a low-dimensional stimulus space.  That is why the split-half
maps disagree, and no estimator applied to a 3600-parameter fit can rescue it.

This module fits a *shape* instead.  A 2D Gaussian has five spatial parameters
(center, two widths, orientation) plus a gain and a baseline that are solved in
closed form, so the model is seven parameters against ~32 independent samples --
tight, but tractable in a way the pixel map is not.

The point is not that a Gaussian is the true receptive-field shape.  It is that
a model this small can be **cross-validated**, and cross-validated prediction is
a claim about the data rather than a picture to be eyeballed.  Every fit here is
scored on held-out blocks and against two nested nulls:

    constant            no visual drive at all
    mean luminance      the cell tracks total light in the field, no location
    Gaussian RF         the cell tracks light at a particular place

A Gaussian that does not beat *mean luminance* on held-out data is not evidence
of a receptive field, however clean its map looks.  That comparison is the
whole reason this module exists, and it is reported for every cell.

Folds are contiguous blocks, never interleaved frames: with a stimulus
autocorrelated over tens of seconds, random frame-wise folds put near-duplicate
samples in train and test and report a cross-validated score that is really a
training score.

Nothing here modifies or imports the STA path; `rf_from_reconstruction` is
untouched and the two can be run on the same npz independently.

Entry point: `gaussian_rf_from_reconstruction(npz_path, h5_path, ...)`.

Usage
-----
    python -m fm2p.rf_gaussian --npz_path /path/to/<prefix>_retinal_images.npz
    python -m fm2p.rf_gaussian --npz_path ... --pdf --n_shuffles 200
"""

if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

import argparse
import os
import time

import h5py
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .utils.files import write_h5
from .rf_from_reconstruction import (retinal_pixel_angles, bin_stimulus_to_2p,
                                     _bin_image_stack, SPEED_THRESH_CM_S)

# Cross-validation folds, as contiguous blocks of this many seconds.  Long
# enough that a train block and a test block are close to independent given the
# stimulus autocorrelation, short enough to leave several folds in a session.
CV_BLOCK_S = 60.0
N_FOLDS = 5

# Bounds on the Gaussian, in degrees.  The lower width bound is one bin at
# bin_px=2; below that the kernel falls between pixels and the fit is
# unconstrained.  The upper bound is a third of the field -- wider than that and
# the Gaussian is indistinguishable from the mean-luminance null it is being
# compared against, so the comparison would be vacuous.
SIGMA_MIN_DEG = 2.0
SIGMA_MAX_DEG = 40.0

_CMAP = 'plasma'


def gaussian_kernel(az, el, x0, y0, sx, sy, theta):
    """Unit-sum 2D Gaussian over the retinal pixel grid.

    Unit sum, not unit peak: the kernel is applied as `S @ k`, so normalizing
    the sum makes the predictor a weighted MEAN luminance over the field. The
    gain then carries the amplitude and does not trade off against the widths,
    which keeps the optimizer away from the degenerate direction where a
    shrinking kernel and a growing gain cancel.
    """

    ct, st = np.cos(theta), np.sin(theta)
    dx, dy = az - x0, el - y0
    u = ct * dx + st * dy
    v = -st * dx + ct * dy
    g = np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))
    ssum = g.sum()
    return (g / ssum) if ssum > 0 else g


def blocked_folds(n_frames, dt, block_s=CV_BLOCK_S, n_folds=N_FOLDS):
    """Assign each frame to a fold, in contiguous blocks.

    Blocks are dealt round-robin so every fold spans the whole session rather
    than one stretch of it; a fold taken from a single contiguous stretch would
    be scored under whatever behavioral state happened to fill it.
    """

    bl = max(1, int(round(float(block_s) / dt)))
    block_id = np.arange(n_frames) // bl
    return (block_id % int(n_folds)).astype(int)


def stimulus_basis(S, n_comp=60):
    """Low-rank basis for the stimulus, so a kernel can be applied cheaply.

    Evaluating `S @ k` for every optimizer step costs n_frames x n_pixels, which
    dominates the fit.  The stimulus is nowhere near full rank -- its
    participation ratio is ~10 (scene) to ~30 (pillar) -- so projecting onto its
    leading components loses nothing and makes each evaluation
    n_frames x n_comp + n_comp x n_pixels instead, ~30x faster at 3600 pixels.

    The basis is computed from the STIMULUS ONLY and never sees the firing rate,
    so sharing it across cross-validation folds leaks nothing about the response.

    Returns (Sb, V) with Sb = S @ V.T, so `S @ k` is `Sb @ (V @ k)`.
    """

    n_comp = int(min(n_comp, min(S.shape) - 1))
    # Randomized range finding: cheaper than a full SVD and exact enough for a
    # basis that only has to span the stimulus.
    rng = np.random.default_rng(0)
    Om = rng.standard_normal((S.shape[1], n_comp + 10))
    Q, _ = np.linalg.qr(S @ Om)
    B = Q.T @ S
    _, _, Vt = np.linalg.svd(B, full_matrices=False)
    V = Vt[:n_comp]
    return S @ V.T, V


def _linear_solve(pred, rate):
    """Least-squares gain and baseline for a fixed predictor. Returns (a, b)."""

    A = np.column_stack([pred, np.ones_like(pred)])
    coef, *_ = np.linalg.lstsq(A, rate, rcond=None)
    return float(coef[0]), float(coef[1])


def _cv_r(pred, rate, folds):
    """Held-out correlation, pooled over folds.

    Pooled rather than averaged per fold: with five folds a per-fold mean is
    noisy, and pooling the held-out predictions gives one correlation over the
    whole session that is not dominated by whichever fold happened to be
    quietest.
    """

    out = np.full_like(rate, np.nan, dtype=float)
    for k in np.unique(folds):
        te = folds == k
        tr = ~te
        if tr.sum() < 20 or te.sum() < 5:
            continue
        a, b = _linear_solve(pred[tr], rate[tr])
        out[te] = a * pred[te] + b
    m = np.isfinite(out)
    if m.sum() < 20 or np.std(out[m]) < 1e-12 or np.std(rate[m]) < 1e-12:
        return np.nan
    return float(np.corrcoef(out[m], rate[m])[0, 1])


def fit_gaussian_rf(S, rate, az, el, folds, n_starts=8, seed=0, basis=None,
                    mean_lum=None):
    """Fit one cell's Gaussian RF and score it against the nested nulls.

    S    : (n_frames, n_pixels) stimulus, already lagged and frame-selected
    rate : (n_frames,) spike rate
    az, el : (n_pixels,) pixel center angles in degrees

    The five spatial parameters are optimized on TRAINING folds only, one fit
    per fold, and the reported `cv_r` is the held-out correlation pooled across
    folds.  Fitting the shape on all frames and then cross-validating only the
    gain would leak the test set into the parameter that matters most.

    Returns a dict of parameters (from a final fit on all frames, for display)
    and the three cross-validated scores.
    """

    rng = np.random.default_rng(seed)
    az_r, el_r = az.ravel(), el.ravel()
    # With `basis` and `mean_lum` supplied, S is never touched -- that is what
    # lets a worker process fit a cell while holding a few MB instead of the
    # whole stimulus movie.
    if basis is None:
        basis = stimulus_basis(S)
    Sb, V = basis
    if mean_lum is None:
        mean_lum = S.mean(axis=1)

    def apply_k(k, rows=None):
        """S @ k through the basis; `rows` selects frames without re-projecting."""
        kv = V @ k
        return (Sb if rows is None else Sb[rows]) @ kv

    # Null 1: constant.  Null 2: mean luminance over the whole field -- the
    # cell responds to how bright the world is, with no preferred location.
    r_mean = _cv_r(mean_lum, rate, folds)

    lo = np.array([az_r.min(), el_r.min(), SIGMA_MIN_DEG, SIGMA_MIN_DEG, 0.0])
    hi = np.array([az_r.max(), el_r.max(), SIGMA_MAX_DEG, SIGMA_MAX_DEG, np.pi])

    def neg_r2(p, rows, rx):
        x0, y0, sx, sy, th = p
        if not (lo[2] <= sx <= hi[2] and lo[3] <= sy <= hi[3]):
            return 1e6
        k = gaussian_kernel(az_r, el_r, x0, y0, sx, sy, th)
        pred = apply_k(k, rows)
        if np.std(pred) < 1e-12:
            return 1e6
        a, b = _linear_solve(pred, rx)
        resid = rx - (a * pred + b)
        return float(np.sum(resid ** 2))

    def one_fit(rows, rx):
        best, bp = np.inf, None
        for _ in range(int(n_starts)):
            p0 = np.array([rng.uniform(az_r.min(), az_r.max()),
                           rng.uniform(el_r.min(), el_r.max()),
                           rng.uniform(4.0, 20.0), rng.uniform(4.0, 20.0),
                           rng.uniform(0, np.pi)])
            try:
                res = minimize(neg_r2, p0, args=(rows, rx),
                               method='Nelder-Mead',
                               options=dict(maxiter=300, xatol=0.5, fatol=1e-8))
            except Exception:
                continue
            if res.fun < best:
                best, bp = res.fun, res.x
        return bp

    # Cross-validated Gaussian: shape refit on each training set.
    held = np.full_like(rate, np.nan, dtype=float)
    for k in np.unique(folds):
        te = folds == k
        tr = ~te
        if tr.sum() < 50 or te.sum() < 5:
            continue
        p = one_fit(tr, rate[tr])
        if p is None:
            continue
        kk = gaussian_kernel(az_r, el_r, *p)
        a, b = _linear_solve(apply_k(kk, tr), rate[tr])
        held[te] = a * apply_k(kk, te) + b
    m = np.isfinite(held)
    r_gauss = (float(np.corrcoef(held[m], rate[m])[0, 1])
               if m.sum() > 20 and np.std(held[m]) > 1e-12 else np.nan)

    # Display fit on all frames -- reported for its shape, never for its score.
    p_all = one_fit(None, rate)
    out = dict(cv_r_gauss=r_gauss, cv_r_mean=r_mean,
               gain=np.nan, x0=np.nan, y0=np.nan,
               sigma_x=np.nan, sigma_y=np.nan, theta=np.nan)
    if p_all is not None:
        # (sx, sy, th) and (sy, sx, th + 90 deg) describe the same ellipse, so
        # the optimizer returns either at random.  Canonicalize to major axis
        # first, theta in [0, pi), or sigma_x/sigma_y are not comparable across
        # cells and any population summary of RF shape is meaningless.
        if p_all[2] < p_all[3]:
            p_all = np.array([p_all[0], p_all[1], p_all[3], p_all[2],
                              p_all[4] + np.pi / 2.0])
        p_all[4] = np.mod(p_all[4], np.pi)
        kk = gaussian_kernel(az_r, el_r, *p_all)
        a, b = _linear_solve(apply_k(kk), rate)
        out.update(x0=p_all[0], y0=p_all[1], sigma_x=p_all[2],
                   sigma_y=p_all[3], theta=p_all[4], gain=a)
    return out


def _fit_cells_serial(cell_idx, R, az, el, folds, basis, mean_lum,
                      n_starts, seed, n_shuffles, rng_seed):
    """Fit a list of cells in this process.  Shared by the serial and Ray paths
    so the two cannot drift apart."""

    rng = np.random.default_rng(rng_seed)
    n = basis[0].shape[0]
    out = []
    for c in cell_idx:
        r = fit_gaussian_rf(None, R[c], az, el, folds, n_starts=n_starts,
                            seed=seed + int(c), basis=basis, mean_lum=mean_lum)
        nulls = []
        for j in range(int(n_shuffles)):
            sh = int(rng.integers(max(1, n // 10), n - max(1, n // 10)))
            rs = fit_gaussian_rf(None, np.roll(R[c], sh), az, el, folds,
                                 n_starts=max(2, n_starts // 3),
                                 seed=seed + int(c) + j, basis=basis,
                                 mean_lum=mean_lum)
            nulls.append(rs['cv_r_gauss'])
        out.append((int(c), r, nulls))
    return out


def _fit_cells_ray(n_workers, cell_ids, R, az, el, folds, basis, mean_lum,
                   n_starts, seed, n_shuffles, chunk=6):
    """Same fits, spread over Ray workers.

    The shared arrays go into the object store ONCE -- after the basis
    projection they are only a few MB, so every worker gets the stimulus for
    free and only the per-cell rate travels.  Each worker is pinned to a single
    BLAS thread: without that, 16 processes each spawning 16 threads oversubscribe
    the machine and the parallel version runs slower than the serial one.
    """

    import ray
    if not ray.is_initialized():
        ray.init(num_cpus=int(n_workers), include_dashboard=False,
                 log_to_driver=False,
                 runtime_env={'env_vars': {'OMP_NUM_THREADS': '1',
                                           'MKL_NUM_THREADS': '1',
                                           'OPENBLAS_NUM_THREADS': '1'}})
    ref = ray.put((R, az, el, folds, basis, mean_lum))

    @ray.remote
    def _work(ref_bundle, ids, n_starts, seed, n_shuffles, rng_seed):
        Rw, azw, elw, fw, bw, mw = ref_bundle
        return _fit_cells_serial(ids, Rw, azw, elw, fw, bw, mw,
                                 n_starts, seed, n_shuffles, rng_seed)

    chunks = [cell_ids[i:i + chunk] for i in range(0, len(cell_ids), chunk)]
    futures = [_work.remote(ref, ch, n_starts, seed, n_shuffles, seed + 7919 * i)
               for i, ch in enumerate(chunks)]
    out, done_n, t0 = [], 0, time.time()
    while futures:
        ready, futures = ray.wait(futures, num_returns=1)
        out.extend(ray.get(ready[0]))
        done_n += 1
        if done_n % 10 == 0 or not futures:
            print(f'    {len(out)}/{len(cell_ids)} cells '
                  f'({time.time() - t0:.0f}s)', flush=True)
    return out


def gaussian_rf_from_reconstruction(
        npz_path, h5_path=None, out_h5=None, lag_s=0.133, bin_px=2,
        spike_key='norm_spikes', speed_thresh=SPEED_THRESH_CM_S,
        light_only=True, max_cells=None, n_starts=8, n_shuffles=0,
        cv_block_s=CV_BLOCK_S, n_folds=N_FOLDS, seed=0, make_pdf=False,
        n_workers=0):
    """Fit a 2D Gaussian receptive field per cell, cross-validated.

    Frame selection matches the STA path (covered, light, moving) so the two are
    comparable on the same recording.  `n_shuffles` > 0 adds a circular-shift
    null on the cross-validated score, which is the honest significance test
    here: it asks whether a Gaussian fitted to this cell predicts held-out
    activity better than the same fit to a time-shifted copy of the same trace.
    """

    # Split on the '_retinal_images' stem rather than the whole default file
    # name: get_retinal_image tags its npz with the render mode
    # (_retinocentric, _nose_ego, ...), and a plain .replace on those is a
    # no-op that leaves h5_path pointing at the npz.
    _stem, _sep, _suffix = npz_path.rpartition('_retinal_images')
    if h5_path is None:
        h5_path = (_stem + '_preproc.h5' if _sep
                   else npz_path.replace('.npz', '_preproc.h5'))
    if out_h5 is None:
        out_h5 = (_stem + f'_retinal_rf_gauss{_suffix[:-4]}.h5' if _sep
                  else npz_path.replace('.npz', '_retinal_rf_gauss.h5'))

    print(f'Loading {os.path.basename(npz_path)} ...')
    npz = np.load(npz_path, allow_pickle=False)
    with h5py.File(h5_path, 'r') as f:
        rate_all = np.asarray(f[spike_key][:], dtype=float)
        twopT = np.asarray(f['twopT'][:], dtype=float)
        speed = np.asarray(f['speed'][:], dtype=float) if 'speed' in f else None
        ltdk = (np.asarray(f['ltdk_state_vec'][:]).astype(bool)
                if 'ltdk_state_vec' in f else None)
    if max_cells is not None:
        rate_all = rate_all[:int(max_cells)]
    n_fr = min(rate_all.shape[1], len(twopT))
    rate_all, twopT = rate_all[:, :n_fr], twopT[:n_fr]
    dt = float(np.nanmedian(np.diff(twopT)))

    S, n_per_bin = bin_stimulus_to_2p(npz['retinal_images'], npz['eyeT'],
                                      twopT, bin_px=bin_px)
    use = n_per_bin > 0
    if speed is not None and speed_thresh > 0:
        sp = np.asarray(speed, dtype=float)
        if len(sp) < n_fr:
            sp = np.concatenate([sp, np.full(n_fr - len(sp), sp[-1])])
        use &= sp[:n_fr] > speed_thresh
        print(f'  motion filter (>{speed_thresh:g} cm/s): '
              f'{100 * np.mean(sp[:n_fr] > speed_thresh):.1f}% of frames moving')
    if light_only and ltdk is not None and len(ltdk) >= n_fr:
        use &= ltdk[:n_fr]
    use &= np.isfinite(rate_all).all(axis=0)
    print(f'  {use.sum()}/{n_fr} 2P frames used')
    if use.sum() < 200:
        raise RuntimeError(f'only {use.sum()} usable frames -- too few to fit.')

    idx = np.flatnonzero(use)
    lag = int(round(float(lag_s) / dt))
    S_use = S[idx].reshape(len(idx), -1)
    R_use = rate_all[:, idx]
    if lag > 0:
        S_use, R_use = S_use[:-lag], R_use[:, lag:]
    elif lag < 0:
        S_use, R_use = S_use[-lag:], R_use[:, :lag]

    az_f, el_f = retinal_pixel_angles()
    if bin_px and int(bin_px) > 1:
        az = _bin_image_stack(az_f[None], bin_px)[0]
        el = _bin_image_stack(el_f[None], bin_px)[0]
    else:
        az, el = az_f, el_f
    grid = az.shape

    folds = blocked_folds(S_use.shape[0], dt, cv_block_s, n_folds)
    print(f'  {S_use.shape[0]} frames, lag {lag:+d} ({lag * dt:+.3f} s), '
          f'{len(np.unique(folds))} folds of {cv_block_s:g} s blocks')

    print('  building stimulus basis ...')
    basis = stimulus_basis(S_use)
    print(f'  basis {basis[1].shape[0]} components of {S_use.shape[1]} pixels')

    mean_lum = S_use.mean(axis=1)
    n_cells = R_use.shape[0]
    keys = ('cv_r_gauss', 'cv_r_mean', 'gain', 'x0', 'y0',
            'sigma_x', 'sigma_y', 'theta')
    res = {k: np.full(n_cells, np.nan) for k in keys}
    null = np.full((int(n_shuffles), n_cells), np.nan) if n_shuffles else None

    # Fitting is embarrassingly parallel across cells: they share the stimulus
    # basis and touch nothing of each other's.  n_workers=0 asks for one worker
    # per core bar two, 1 forces the serial path.
    if n_workers == 0:
        n_workers = max(1, (os.cpu_count() or 2) - 2)
    n_workers = int(n_workers)
    if n_workers > 1:
        try:
            import ray  # noqa: F401
        except ImportError:
            print('  ray not installed -- falling back to a single process')
            n_workers = 1

    t0 = time.time()
    ids = list(range(n_cells))
    if n_workers > 1:
        print(f'  fitting {n_cells} cells on {n_workers} ray workers ...')
        got = _fit_cells_ray(n_workers, ids, R_use, az, el, folds, basis,
                             mean_lum, n_starts, seed, n_shuffles)
    else:
        print(f'  fitting {n_cells} cells in one process ...')
        got = _fit_cells_serial(ids, R_use, az, el, folds, basis, mean_lum,
                                n_starts, seed, n_shuffles, seed)
    for c, r, nulls in got:
        for k in keys:
            res[k][c] = r[k]
        if n_shuffles and nulls:
            null[:len(nulls), c] = nulls
    print(f'  done in {time.time() - t0:.0f}s '
          f'({(time.time() - t0) / max(n_cells, 1):.2f}s/cell)')

    # The comparison that matters: does a localized Gaussian beat a cell that
    # simply tracks total luminance?  A positive cv_r on its own does not show
    # a receptive field -- both models are driven by the same scene.
    gain_over_mean = res['cv_r_gauss'] - res['cv_r_mean']
    if n_shuffles:
        ge = np.nansum(null >= res['cv_r_gauss'][None, :], axis=0)
        p_value = (1.0 + ge) / (1.0 + int(n_shuffles))
    else:
        p_value = np.full(n_cells, np.nan)

    fin = np.isfinite(res['cv_r_gauss'])
    print(f'\n  cross-validated r, Gaussian : median {np.nanmedian(res["cv_r_gauss"]):+.3f}')
    print(f'  cross-validated r, mean lum : median {np.nanmedian(res["cv_r_mean"]):+.3f}')
    print(f'  Gaussian beats mean-luminance in {100 * np.nanmean(gain_over_mean > 0):.0f}% '
          f'of {fin.sum()} fitted cells')
    if n_shuffles:
        print(f'  p <= 0.05 (circular-shift null): '
              f'{100 * np.nanmean(p_value <= 0.05):.0f}% of cells')

    out = {k: res[k].astype(np.float64) for k in keys}
    out.update(gain_over_mean=gain_over_mean.astype(np.float64),
               p_value=p_value.astype(np.float64),
               az=az.astype(np.float32), el=el.astype(np.float32),
               grid=np.array(grid), lag_s=np.array(lag * dt),
               dt_2p=np.array(dt), bin_px=np.array(bin_px),
               n_frames_used=np.array(S_use.shape[0]),
               cv_block_s=np.array(cv_block_s), n_folds=np.array(n_folds),
               speed_thresh=np.array(speed_thresh),
               light_only=np.array(light_only), spike_key=spike_key)
    if null is not None:
        out['null_cv_r'] = null.astype(np.float32)
    write_h5(out_h5, out)
    print(f'Saved -> {out_h5}')
    if make_pdf:
        make_gaussian_pdf(out, S_use, R_use, out_h5.replace('.h5', '.pdf'))
    return out, out_h5


def make_gaussian_pdf(res, S_use, R_use, out_pdf, max_pages=60):
    """One page per cell, best first: the fitted kernel and its CV scores."""

    az, el = res['az'], res['el']
    g = tuple(int(v) for v in res['grid'])
    order = np.argsort(np.where(np.isfinite(res['gain_over_mean']),
                                res['gain_over_mean'], -np.inf))[::-1]
    order = order[:max_pages]
    with PdfPages(out_pdf) as pdf:
        for c in order:
            if not np.isfinite(res['x0'][c]):
                continue
            k = gaussian_kernel(az.ravel(), el.ravel(), res['x0'][c],
                                res['y0'][c], res['sigma_x'][c],
                                res['sigma_y'][c], res['theta'][c])
            fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
            im = axs[0].imshow(k.reshape(g), cmap=_CMAP, origin='upper',
                               extent=(az.min(), az.max(), el.min(), el.max()))
            plt.colorbar(im, ax=axs[0], fraction=0.046)
            axs[0].set_title(f'cell {c} fitted kernel')
            axs[0].set_xlabel('azimuth (deg)'); axs[0].set_ylabel('elevation (deg)')
            pred = S_use @ k
            a, b = _linear_solve(pred, R_use[c])
            axs[1].plot(R_use[c][:600], lw=0.8, label='rate')
            axs[1].plot((a * pred + b)[:600], lw=0.8, label='Gaussian fit')
            axs[1].legend(fontsize=7); axs[1].set_title('first 600 frames')
            axs[1].set_xlabel('2P frame')
            axs[2].axis('off')
            axs[2].text(0.02, 0.95,
                        f'cv r  Gaussian     {res["cv_r_gauss"][c]:+.3f}\n'
                        f'cv r  mean lum     {res["cv_r_mean"][c]:+.3f}\n'
                        f'difference         {res["gain_over_mean"][c]:+.3f}\n'
                        f'p (shift null)     {res["p_value"][c]:.3f}\n\n'
                        f'center  ({res["x0"][c]:+.1f}, {res["y0"][c]:+.1f}) deg\n'
                        f'sigma   {res["sigma_x"][c]:.1f} x {res["sigma_y"][c]:.1f} deg\n'
                        f'gain    {res["gain"][c]:+.3g}',
                        va='top', family='monospace', fontsize=9)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)
    print(f'Saved -> {out_pdf}')


def main():
    p = argparse.ArgumentParser(
        description='Cross-validated 2D Gaussian receptive fields from the '
                    'reconstructed retinal image.')
    p.add_argument('--npz_path', required=True)
    p.add_argument('--h5_path', default=None)
    p.add_argument('--out_h5', default=None)
    p.add_argument('--lag_s', type=float, default=0.133)
    p.add_argument('--bin_px', type=int, default=2)
    p.add_argument('--spike_key', default='norm_spikes')
    p.add_argument('--speed_thresh', type=float, default=SPEED_THRESH_CM_S)
    p.add_argument('--all_light_dark', action='store_true')
    p.add_argument('--max_cells', type=int, default=None)
    p.add_argument('--n_starts', type=int, default=8,
                   help='Random restarts per fit (the objective is multimodal).')
    p.add_argument('--n_shuffles', type=int, default=0,
                   help='Circular-shift null on the cross-validated score. '
                        'Expensive: each shuffle refits every fold.')
    p.add_argument('--cv_block_s', type=float, default=CV_BLOCK_S)
    p.add_argument('--n_folds', type=int, default=N_FOLDS)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--pdf', action='store_true')
    p.add_argument('--n_workers', type=int, default=0,
                   help='Ray workers for the per-cell fits. 0 (default) uses '
                        'one per core bar two; 1 forces a single process.')
    a = p.parse_args()
    gaussian_rf_from_reconstruction(
        npz_path=a.npz_path, h5_path=a.h5_path, out_h5=a.out_h5,
        lag_s=a.lag_s, bin_px=a.bin_px, spike_key=a.spike_key,
        speed_thresh=a.speed_thresh, light_only=not a.all_light_dark,
        max_cells=a.max_cells, n_starts=a.n_starts, n_shuffles=a.n_shuffles,
        cv_block_s=a.cv_block_s, n_folds=a.n_folds, seed=a.seed,
        make_pdf=a.pdf, n_workers=a.n_workers)


if __name__ == '__main__':
    main()



# python -m fm2p.rf_gaussian --h5_path /home/dylan/Storage/freely_moving_data/V1PPC/cohort02_recordings/251016_DMM_DMM061_pos18/fm1/251016_DMM_DMM061_fm_01_preproc.h5 --npz_path /home/dylan/Storage/freely_moving_data/V1PPC/cohort02_recordings/251016_DMM_DMM061_pos18/fm1/251016_DMM_DMM061_fm_01_retinal_images.npz --pdf