# -*- coding: utf-8 -*-
"""
fm2p/rf_from_reconstruction.py

Spatial receptive fields from the reconstructed retinal image.

`get_retinal_image.py` writes a `<prefix>_retinal_images.npz` holding a
(N_eye, 120, 120) binary movie of where the pillar sat on the retina, at the
eye-camera rate.  This module treats that movie as the stimulus and computes a
spike-triggered average against the 2P calcium traces, giving a receptive field
in retinal (azimuth, elevation) coordinates.

The important difference from `sparse_noise_mapping.py` is that this stimulus is
**not** balanced.  Sparse noise visits every location equally by construction, so
a raw STA is already an unbiased estimate.  Here the animal chooses where to put
the pillar on its retina, and it does so very unevenly -- so the raw STA mostly
reports occupancy, not tuning.  Two things follow, and both are done below:

  1. the occupancy map is subtracted, leaving the *excess* visibility around
     spikes rather than the absolute visibility, and
  2. significance is judged against a circular-shift null, which preserves both
     the spike train's autocorrelation and the stimulus's spatial statistics.
     A pixel is only interesting if it beats a shuffled version of the *same*
     rate trace against the *same* movie.

Entry point: `rf_from_reconstruction(npz_path, h5_path, ...)`.

Usage
-----
    python -m fm2p.rf_from_reconstruction \\
        --npz_path /path/to/<prefix>_retinal_images.npz

    # explicit lag scan and a PDF of the significant cells
    python -m fm2p.rf_from_reconstruction \\
        --npz_path /path/to/<prefix>_retinal_images.npz \\
        --lags 0 0.13 0.27 0.40 0.53 --pdf

Functions
---------
retinal_pixel_angles
    (az, el) in degrees for every pixel center of the retinal image.
_bin_image_stack
    Block-mean an (N, H, W) stack by an integer factor.
bin_stimulus_to_2p
    Average the retinal movie within each 2P frame.
compute_rf
    Occupancy-corrected STA, a scaled map, and an FWER-controlled p-value.
summarize_rf
    Per-cell RF peak location and extent.
rf_from_reconstruction
    Receptive fields from a reconstructed retinal movie.
find_sparse_noise_dir
    Head-fixed sparse noise directory beside the freely moving recording.
load_sparse_noise_rfs
    Open sparse noise STAs and labels for per-cell lookup.
make_rf_pdf
    One page per reliable sparse noise RF cell: full map, both halves, lag, sparse noise RF.
main
    Command-line entry point.


DMM, September 2026
"""

if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

import argparse
import glob
import os
import time

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap

from .utils.files import write_h5


# Retinal image geometry, matching simulate_retinal_projection.  The projection
# is rectilinear, not equiangular, so a pixel is 1 deg only on average across the
# full width -- 1.65 deg/px at the center and 0.41 deg/px at the edge.  Pixel
# centers are converted to true angles here so that RF centers and areas are
# reported in degrees rather than in pixels.
RET_RES     = 120
RET_FOV_DEG = 120.0

# Speed above which a frame counts as locomotion, cm/s.  Matches the motion
# filter used elsewhere in the package (boundary_tuning, tuning).
SPEED_THRESH_CM_S = 2.0

# Default temporal offsets between stimulus and calcium, in seconds.  0.133 s
# is one 2P frame at 7.5 Hz.
#
# NEGATIVE LAGS ARE NOT OPTIONAL.  A negative lag asks what the stimulus does
# AFTER the response, which no visual drive can explain, so the negative arm is
# a within-cell control for everything that is slow and shared between firing
# rate and the animal's behavior.  Scanning only positive lags hides the single
# most informative thing about this analysis: on 251016_DMM_DMM056_pos13 the
# number of "significant" cells is 25 at -0.27 s, 29 at 0 s and 24 at +0.27 s --
# flat, i.e. not a visual response at all.  See `causal_index` below.
# DEFAULT_LAGS_S = (-0.533, -0.400, -0.267, -0.133, 0.0,
#                   0.133, 0.267, 0.400, 0.533, 0.667, 0.800)
DEFAULT_LAGS_S = (-0.267, -0.133, 0.0, 0.133, 0.267, 0.400, 0.533)

# Coolwarm throughout, always symmetric about zero so the sign is readable.
_RF_CMAP = 'coolwarm'

# Sparse noise STAs are stored flat at monitor resolution.
SN_STA_SHAPE = (768, 1360)
# Display-only downsampling of the sparse noise STA; full res is ~1M px per cell.
SN_DISPLAY_STEP = 4


def retinal_pixel_angles(res=RET_RES, fov_deg=RET_FOV_DEG):
    """(az, el) in degrees for every pixel center of the retinal image.

    Returns two (res, res) arrays.  Column index maps to azimuth (positive =
    the animal's right), row index to elevation (positive = up), following
    get_retinal_image's pinhole convention.
    """

    f = (res / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    c = res / 2.0
    u = np.arange(res) + 0.5 - c
    v = np.arange(res) + 0.5 - c
    U, V = np.meshgrid(u, v)                      # U: column, V: row
    az = np.degrees(np.arctan2(U, f))
    el = np.degrees(np.arctan2(-V, np.hypot(f, U)))
    return az, el


def _bin_image_stack(stack, factor):
    """Block-mean an (N, H, W) stack by an integer factor."""

    if factor is None or int(factor) <= 1:
        return stack
    factor = int(factor)
    n, h, w = stack.shape
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    s = stack[:, :h2, :w2]
    return s.reshape(n, h2 // factor, factor, w2 // factor, factor).mean(axis=(2, 4))


def bin_stimulus_to_2p(retinal_images, eyeT, twopT, bin_px=1):
    """Average the retinal movie within each 2P frame.

    A 2P frame integrates for ~133 ms, over which the retinal image moves a
    great deal, so the stimulus that a frame's calcium signal actually saw is
    the *mean* image across that interval -- not the single nearest eye frame.
    Bin edges are placed midway between 2P frame times, so each bin is centered
    on its frame.

    Returns
    -------
    S : (n_2p, h, w) float32 in [0, 1], NaN-free; bins with no eye frame are 0
    n_per_bin : (n_2p,) int, how many eye frames each bin averaged
    """

    eyeT = np.asarray(eyeT, dtype=float)
    twopT = np.asarray(twopT, dtype=float)
    n_2p = len(twopT)

    mids = 0.5 * (twopT[1:] + twopT[:-1])
    edges = np.concatenate([[twopT[0] - (mids[0] - twopT[0])], mids,
                            [twopT[-1] + (twopT[-1] - mids[-1])]])

    ok = np.isfinite(eyeT)
    idx = np.searchsorted(edges, eyeT[ok], side='right') - 1
    keep = (idx >= 0) & (idx < n_2p)
    idx = idx[keep]
    src = np.flatnonzero(ok)[keep]

    imgs = retinal_images[src]
    if imgs.dtype == np.uint8:
        imgs = imgs.astype(np.float32) / 255.0
    else:
        imgs = imgs.astype(np.float32)
    imgs = _bin_image_stack(imgs, bin_px)

    h, w = imgs.shape[1], imgs.shape[2]
    S = np.zeros((n_2p, h, w), dtype=np.float32)
    n_per_bin = np.zeros(n_2p, dtype=np.int64)
    np.add.at(S, idx, imgs)
    np.add.at(n_per_bin, idx, 1)
    nz = n_per_bin > 0
    S[nz] /= n_per_bin[nz, None, None]
    return S, n_per_bin


def compute_rf(S, rate, lag_frames=0, n_shuffles=100, dt_2p=None,
               min_shift_s=5.0, seed=0, verbose=True):
    """Occupancy-corrected STA, a scaled map, and an FWER-controlled p-value.

    Parameters
    ----------
    S : (n_frames, h, w) float32
        Stimulus already binned onto the 2P timebase.
    rate : (n_cells, n_frames) float
        Spike rate per 2P frame.
    lag_frames : int
        Shift applied to the rate relative to the stimulus.  Positive means the
        response follows the stimulus, which is the physiological direction.
    n_shuffles : int
        Circular shifts used to build the null.  Shifts are shared across cells,
        so the null costs `n_shuffles` matrix products rather than
        `n_cells * n_shuffles`.
    min_shift_s : float
        Smallest circular shift, in seconds, so a shift never leaves the rate
        trace aligned with itself through the calcium autocorrelation.

    Returns
    -------
    dict with 'sta', 'occupancy', 'delta', 't', 'z', 'peak_t', 'null_peak',
    'p_value', 'n_used', 'lag_frames'

    Notes
    -----
    Three maps come out of this and they answer different questions.

    `delta` is STA minus occupancy: how much more (or less) the pillar was at
    each retinal location around spikes than on average.  It is in stimulus
    units and is the quantity to plot if you want an effect size.

    `t` is `delta` divided by an analytic standard error of the weighted mean,
    `sqrt(sum(w^2)) * sd_t(S_pixel)` with `w = r / sum(r)`.  This rescaling
    matters because the stimulus is one moving object: a pixel the pillar
    crosses often has a far larger sampling variance than one at the edge of the
    field, so an unscaled maximum would simply find the highest-occupancy pixel.

    `t` is NOT a t-statistic in the textbook sense -- 2P frames are strongly
    autocorrelated, so its null is much wider than Student's.  That is exactly
    why significance is decided by permutation instead: `p_value` compares the
    observed `max_pixels(t)` against the same statistic computed on circularly
    shifted rate traces.  Circular shifts preserve both the calcium
    autocorrelation and the stimulus, so the null carries the same inflation the
    observed statistic does.  Taking the maximum over pixels before comparing is
    what controls the family-wise error across the whole map; thresholding a
    per-pixel z at 4 does not, because the peak of ~3600 correlated pixels is
    routinely 4 or more under the null.

    `z` is the per-pixel empirical z against the shuffles.  It is descriptive --
    useful for seeing the shape of an RF whose p-value already passed -- and
    must not be thresholded to decide significance.
    """

    n_fr, h, w = S.shape
    P = h * w
    S2 = S.reshape(n_fr, P)

    rate = np.atleast_2d(np.asarray(rate, dtype=np.float64))

    # Apply the lag by trimming both traces, so no wrap-around is introduced.
    if lag_frames > 0:
        S_use = S2[:-lag_frames]
        R_use = rate[:, lag_frames:]
    elif lag_frames < 0:
        S_use = S2[-lag_frames:]
        R_use = rate[:, :lag_frames]
    else:
        S_use, R_use = S2, rate
    n_use = S_use.shape[0]
    n_cells = R_use.shape[0]

    R_use = np.where(np.isfinite(R_use), R_use, 0.0)
    occupancy = S_use.mean(axis=0)
    sd_S = S_use.std(axis=0)
    sd_S_safe = np.where(sd_S > 0, sd_S, np.nan)

    def _stat(R):
        """ Delta and t for a rate matrix, sharing the fixed pixel scale. """

        ws = R.sum(axis=1)
        ws = np.where(ws > 0, ws, np.nan)
        d = (R @ S_use) / ws[:, None] - occupancy[None, :]
        ess = np.sqrt((R ** 2).sum(axis=1)) / ws          # sqrt(sum w^2)
        return d, d / (ess[:, None] * sd_S_safe[None, :])

    delta, t = _stat(R_use)
    with np.errstate(invalid='ignore'):
        peak_t = np.nanmax(t, axis=1)

    z = np.full_like(delta, np.nan)
    null_peak = np.full((int(n_shuffles), n_cells), np.nan)
    p_value = np.full(n_cells, np.nan)

    min_shift = max(1, int(round((min_shift_s / dt_2p) if dt_2p else 40)))
    if n_shuffles and n_shuffles > 0 and (n_use - min_shift) > min_shift:
        rng = np.random.default_rng(seed)
        shifts = rng.integers(min_shift, n_use - min_shift, size=int(n_shuffles))
        mean_acc = np.zeros_like(delta)
        m2_acc = np.zeros_like(delta)
        t0 = time.time()
        for j, sh in enumerate(shifts):
            d_s, t_s = _stat(np.roll(R_use, int(sh), axis=1))
            with np.errstate(invalid='ignore'):
                null_peak[j] = np.nanmax(t_s, axis=1)
            prev = mean_acc.copy()
            mean_acc += (d_s - mean_acc) / (j + 1)
            m2_acc += (d_s - prev) * (d_s - mean_acc)
            if verbose and (j + 1) % 25 == 0:
                print('    Shuffle {}/{} '
                      '({:.0f}s)'.format(j + 1, len(shifts), time.time() - t0))
        sd = np.sqrt(m2_acc / max(len(shifts) - 1, 1))
        with np.errstate(invalid='ignore'):
            z = (delta - mean_acc) / np.where(sd > 0, sd, np.nan)
        # +1 in numerator and denominator: the observed arrangement is itself
        # one of the possible ones, so p can never be exactly zero.
        ge = np.nansum(null_peak >= peak_t[None, :], axis=0)
        p_value = (1.0 + ge) / (1.0 + int(n_shuffles))
    elif verbose:
        print('    Too few frames to shuffle; p-values left as NaN')

    shp = (n_cells, h, w)
    return {
        'sta': ((R_use @ S_use) /
                np.where(R_use.sum(1) > 0, R_use.sum(1), np.nan)[:, None]
                ).reshape(shp).astype(np.float32),
        'occupancy': occupancy.reshape(h, w).astype(np.float32),
        'delta': delta.reshape(shp).astype(np.float32),
        't': t.reshape(shp).astype(np.float32),
        'z': z.reshape(shp).astype(np.float32),
        'peak_t': peak_t.astype(np.float32),
        'null_peak': null_peak.astype(np.float32),
        'p_value': p_value.astype(np.float64),
        'n_used': int(n_use),
        'lag_frames': int(lag_frames),
    }


def summarize_rf(tmap, az, el, null_peak=None, frac_of_peak=0.5):
    """Per-cell RF peak location and extent.

    The RF outline is taken at a fraction of that cell's own peak rather than at
    a fixed statistic threshold, because the useful cutoff depends on how much
    the cell fired and how much the pillar moved; the *existence* of an RF is
    settled separately by the permutation p-value.  When `null_peak` is given,
    the contour is additionally floored at the 95th percentile of the null peak
    statistic, so a cell can never report area from values a shuffle reaches
    routinely.

    Area is the solid angle of the region, computed from the per-pixel angular
    area rather than a pixel count, because the projection is rectilinear
    (see retinal_pixel_angles).
    """

    n_cells = tmap.shape[0]
    out = {k: np.full(n_cells, np.nan) for k in
           ('peak_t', 'peak_az', 'peak_el', 'area_deg2', 'n_rf_px', 'thresh')}

    # Angular area of each pixel, from the local gradient of the angle maps.
    daz = np.gradient(az, axis=1)
    dele = np.gradient(el, axis=0)
    px_area = np.abs(daz * dele)

    floor = None
    if null_peak is not None and np.isfinite(null_peak).any():
        floor = np.nanpercentile(null_peak, 95, axis=0)

    for c in range(n_cells):
        tc = tmap[c]
        if not np.isfinite(tc).any():
            continue
        i_pk = int(np.nanargmax(tc))
        r, cc = np.unravel_index(i_pk, tc.shape)
        pk = float(tc[r, cc])
        out['peak_t'][c] = pk
        out['peak_az'][c] = az[r, cc]
        out['peak_el'][c] = el[r, cc]
        thr = frac_of_peak * pk
        if floor is not None and np.isfinite(floor[c]):
            thr = max(thr, float(floor[c]))
        out['thresh'][c] = thr
        sig = np.isfinite(tc) & (tc >= thr)
        out['n_rf_px'][c] = sig.sum()
        out['area_deg2'][c] = float(px_area[sig].sum())
    return out


def rf_from_reconstruction(npz_path, h5_path=None, out_h5=None,
                           lags_s=DEFAULT_LAGS_S, bin_px=2, n_shuffles=100,
                           spike_key='norm_spikes', speed_thresh=SPEED_THRESH_CM_S,
                           min_occupancy=0.002, alpha=0.05, max_cells=None,
                           seed=0, make_pdf=False, light_only=True,
                           split_mode='interleaved', split_block_s=60.0,
                           split_stat='delta'):
    """Receptive fields from a reconstructed retinal movie.

    Parameters
    ----------
    npz_path : str
        `<prefix>_retinal_images.npz` from get_retinal_image.
    h5_path : str or None
        `<prefix>_preproc.h5`.  Derived from npz_path when omitted.
    lags_s : sequence of float
        Stimulus-to-response offsets to try.  Every lag is saved; the best per
        cell is chosen by peak statistic, and the lag search is folded into the
        null so that choosing it does not inflate significance.
    bin_px : int
        Spatial binning of the retinal image before the STA.  2 gives 60x60
        (~2 deg bins on average), a sensible trade against the noise in a
        single-pillar stimulus.
    min_occupancy : float
        Pixels seen less than this fraction of the peak occupancy are masked
        out.  With one pillar the far periphery is visited a handful of times,
        and those pixels would otherwise dominate every map through a
        near-zero denominator.
    alpha : float
        Family-wise significance level for the permutation test.
    """

    # Split on the '_retinal_images' stem rather than matching the default file
    # name exactly.  get_retinal_image is routinely run with a suffix
    # (_retinocentric, _fixedpupil, _retinocentric_fixed, ...), and on those a
    # plain .replace('_retinal_images.npz', ...) is a no-op that leaves h5_path
    # pointing at the npz.  The isfile guard below then PASSES -- the npz does
    # exist -- and h5py fails several frames later with 'file signature not
    # found', which says nothing about the actual mistake.
    _stem, _sep, _suffix = npz_path.rpartition('_retinal_images')
    if h5_path is None:
        h5_path = (_stem + '_preproc.h5' if _sep
                   else npz_path.replace('.npz', '_preproc.h5'))
    if h5_path.endswith('.npz') or not os.path.isfile(h5_path):
        raise FileNotFoundError(
            f'Could not find the preproc file for {os.path.basename(npz_path)}; '
            f'looked for {h5_path}. Pass h5_path= explicitly.')
    if out_h5 is None:
        # Keep the npz's suffix in the output name so the variants of one
        # session do not overwrite each other.
        out_h5 = (_stem + '_retinal_rf' + _suffix[:-len('.npz')] + '.h5' if _sep
                  else npz_path.replace('.npz', '_retinal_rf.h5'))

    print('Loading {}...'.format(os.path.basename(npz_path)))
    npz = np.load(npz_path, allow_pickle=False)
    retinal_images = npz['retinal_images']
    eyeT = npz['eyeT']
    print('  Retinal movie {} '
          '({:.1f}% of frames '
          'have the pillar in view)'.format(retinal_images.shape, np.mean(retinal_images.any(axis=(1, 2))) * 100))

    with h5py.File(h5_path, 'r') as f:
        if spike_key not in f:
            raise KeyError(
                f'{spike_key!r} not in {os.path.basename(h5_path)}; '
                f'available spike-like keys: '
                f'{[k for k in f.keys() if "spk" in k or "spike" in k]}')
        rate = np.asarray(f[spike_key][:], dtype=float)
        twopT = np.asarray(f['twopT'][:], dtype=float)
        speed = np.asarray(f['speed'][:], dtype=float) if 'speed' in f else None
        # Light/dark state.  The reconstruction is a geometric model: it renders
        # the pillar in darkness too, because it has no idea the lights are off.
        # Fitting an RF on dark frames therefore regresses spikes against a
        # stimulus the animal cannot see, which can only add noise and, if
        # firing differs between light and dark, bias.
        ltdk = (np.asarray(f['ltdk_state_vec'][:]).astype(bool)
                if 'ltdk_state_vec' in f else None)
    n_cells_all = rate.shape[0]
    if max_cells is not None:
        rate = rate[:int(max_cells)]
    print('  Spikes {} from {!r}, twopT n={}'.format(rate.shape, spike_key, len(twopT)))

    n_fr = min(rate.shape[1], len(twopT))
    rate, twopT = rate[:, :n_fr], twopT[:n_fr]

    print('Binning the retinal movie onto the 2P frame times...')
    t0 = time.time()
    S, n_per_bin = bin_stimulus_to_2p(retinal_images, eyeT, twopT, bin_px=bin_px)
    print('  {} in {:.1f}s; '
          '{:.1f}% of 2P frames got at least one '
          'eye frame (median {:.0f} per bin)'.format(S.shape, time.time() - t0, np.mean(n_per_bin > 0) * 100, np.median(n_per_bin[n_per_bin > 0])))

    # Frame selection.  A 2P frame is usable when the reconstruction covered it
    # and the animal was moving; stationary frames are dropped for the same
    # reason the rest of the package drops them -- during grooming and rest the
    # retinal image is nearly static, so those frames form a large, highly
    # correlated block that a circular shift cannot break up.
    use = n_per_bin > 0
    # `speed` is a diff-based quantity and is ALWAYS one sample shorter than
    # twopT, so the old `len(speed) >= n_fr` guard was never true and this
    # filter silently never ran -- speed_thresh was ignored on every session.
    # Pad the missing last sample rather than widen the guard, so the filter
    # either applies to every frame or is explicitly off.
    if speed is not None and len(speed) >= n_fr - 1 and speed_thresh > 0:
        _sp = np.asarray(speed, dtype=float)
        if len(_sp) < n_fr:
            _sp = np.concatenate([_sp, np.full(n_fr - len(_sp), _sp[-1])])
        moving = _sp[:n_fr] > speed_thresh
        use &= moving
        print('  Motion filter (>{:g} cm/s): '
              '{:.1f}% of frames moving'.format(speed_thresh, np.mean(moving) * 100))
    use &= np.isfinite(rate).all(axis=0)
    if light_only:
        if ltdk is not None and len(ltdk) >= n_fr:
            use &= ltdk[:n_fr]
            print('  light-only: {:.1f}% of frames '
                  'are light'.format(np.mean(ltdk[:n_fr]) * 100))
        else:
            print('  light_only requested but no ltdk_state_vec in the h5; '
                  'using all frames (fine for a recording with no dark epochs)')
    print('  {}/{} 2P frames used'.format(use.sum(), n_fr))
    if use.sum() < 100:
        raise RuntimeError(
            f'Only {use.sum()} usable 2P frames -- too few for an RF. Check that '
            f'the reconstruction covers this recording (get_retinal_image prints '
            f'a frames-rendered count) and that speed/spikes are present.')

    dt_2p = float(np.nanmedian(np.diff(twopT)))
    S = S[use]
    rate = rate[:, use]

    az_full, el_full = retinal_pixel_angles()
    if bin_px and int(bin_px) > 1:
        az = _bin_image_stack(az_full[None], bin_px)[0]
        el = _bin_image_stack(el_full[None], bin_px)[0]
    else:
        az, el = az_full, el_full

    occ_all = S.mean(axis=0)
    occ_mask = occ_all >= (min_occupancy * occ_all.max())
    print('  occupancy mask keeps {}/{} pixels '
          '(>= {:g} x peak occupancy)'.format(occ_mask.sum(), occ_mask.size, min_occupancy))

    lag_frames_list, per_lag = [], []
    for lag_s in lags_s:
        lf = int(round(float(lag_s) / dt_2p))
        if lf in lag_frames_list:
            continue
        lag_frames_list.append(lf)
        print('  Lag {:+.3f} s ({:+d} frames): STA + '
              '{} shuffles for {} cells...'.format(lag_s, lf, n_shuffles, rate.shape[0]))
        r = compute_rf(S, rate, lag_frames=lf, n_shuffles=n_shuffles,
                       dt_2p=dt_2p, seed=seed)
        for k in ('delta', 't', 'z'):
            r[k][:, ~occ_mask] = np.nan
        # The mask can only lower a peak, so recompute it after masking.
        with np.errstate(invalid='ignore'):
            r['peak_t'] = np.nanmax(r['t'].reshape(r['t'].shape[0], -1), axis=1)
        per_lag.append(r)

    lags_used = np.array(lag_frames_list, dtype=int)
    T = np.stack([r['t'] for r in per_lag], axis=1)
    D = np.stack([r['delta'] for r in per_lag], axis=1)
    Z = np.stack([r['z'] for r in per_lag], axis=1)
    STA = np.stack([r['sta'] for r in per_lag], axis=1)

    # Best lag per cell.  The same maximization is applied to the null, so the
    # p-value already pays for the lag search rather than being optimistic by
    # the number of lags tried.
    peak_per_lag = np.stack([r['peak_t'] for r in per_lag], axis=1)
    best_i = np.nanargmax(np.where(np.isfinite(peak_per_lag),
                                   peak_per_lag, -np.inf), axis=1)
    obs_peak = peak_per_lag[np.arange(peak_per_lag.shape[0]), best_i]

    null_over_lags = np.nanmax(
        np.stack([r['null_peak'] for r in per_lag], axis=0), axis=0)   # (n_shuf, cells)
    ge = np.nansum(null_over_lags >= obs_peak[None, :], axis=0)
    p_value = (1.0 + ge) / (1.0 + int(n_shuffles))

    Tb = T[np.arange(T.shape[0]), best_i]
    Db = D[np.arange(D.shape[0]), best_i]
    Zb = Z[np.arange(Z.shape[0]), best_i]

    # Split-half maps, each computed on one temporal half of the retained
    # (light, moving) frames at that cell's own best lag.  Nothing is tested
    # against these -- they are for eyeballing whether a map is reproducible.
    # A real RF looks like itself in both halves; a map assembled out of a slow
    # drift usually does not.
    # Split definition.  A contiguous first-vs-second half confounds the RF with
    # anything that drifts over the session -- z drift, bleaching, behavioral
    # state -- and on 251016_DMM_DMM061_pos18 mean firing rate correlates 0.73
    # across contiguous halves versus 0.90 across interleaved 60 s blocks.
    # Measured effect on the median split-half map correlation: 0.130 to
    # 0.264 with `t`, 0.183 to 0.403 with `delta`.  Pass
    # split_mode='contiguous' to recover the old behavior.
    if split_mode == 'contiguous':
        half = np.zeros(S.shape[0], dtype=bool); half[:S.shape[0] // 2] = True
    elif split_mode == 'interleaved':
        _bl = max(1, int(round(float(split_block_s) / dt_2p)))
        half = (np.arange(S.shape[0]) // _bl) % 2 == 0
    else:
        raise ValueError(f"split_mode must be 'interleaved' or 'contiguous', "
                         f"got {split_mode!r}")
    # Statistic.  `t` divides by sd_S, which is a per-PIXEL scale attached to a
    # per-CELL effective N, so rarely-visited pixels get standard errors that
    # are far too small: RF peaks land where occupancy is 0.42x the field
    # median, and 48% of them beyond |az| > 45 deg in a +-60 deg field.  On
    # `delta` the same peaks sit at 1.66x median occupancy, and split-half
    # agreement is 0.403 against 0.264.  The permutation test above still uses
    # `t`; this only changes the maps shown for eyeballing.
    if split_stat not in ('delta', 't'):
        raise ValueError(f"split_stat must be 'delta' or 't', got {split_stat!r}")
    T_h = []
    for hmask in (half, ~half):
        per = {}
        for lf in set(lags_used[best_i].tolist()):
            r = compute_rf(S[hmask], rate[:, hmask], lag_frames=int(lf),
                           n_shuffles=0, dt_2p=dt_2p, verbose=False)
            r[split_stat][:, ~occ_mask] = np.nan
            per[int(lf)] = r[split_stat]
        T_h.append(np.stack([per[int(lags_used[best_i[c]])][c]
                             for c in range(Tb.shape[0])]))
    T_h1, T_h2 = T_h
    print('  split-half maps computed ({} / {} frames, '
          '{}'.format(half.sum(), (~half).sum(), split_mode)
          + (' {:g}s blocks'.format(split_block_s) if split_mode == 'interleaved' else '')
          + ', statistic={})'.format(split_stat))
    summary = summarize_rf(Tb, az, el, null_peak=null_over_lags)
    summary['p_value'] = p_value

    # Causality.  A visual receptive field must be driven by a stimulus that
    # came first, so its statistic has to be larger at positive lags than at
    # negative ones.  Anything symmetric in lag is a slow covariation between
    # firing rate and where the animal happened to be pointing -- statistically
    # real, and not a receptive field.
    lag_s_arr = lags_used * dt_2p
    pos = lag_s_arr > 0
    neg = lag_s_arr < 0
    if pos.any() and neg.any():
        max_pos = np.nanmax(peak_per_lag[:, pos], axis=1)
        max_neg = np.nanmax(peak_per_lag[:, neg], axis=1)
        causal_index = max_pos - max_neg
        causal = (lag_s_arr[best_i] > 0) & (causal_index > 0)
    else:
        causal_index = np.full(peak_per_lag.shape[0], np.nan)
        causal = np.ones(peak_per_lag.shape[0], dtype=bool)
        print('  WARNING: no negative lags scanned, so the causality control is '
              'not available. Pass negative values in --lags; a symmetric lag '
              'profile is the main way this analysis fails.')
    summary['causal_index'] = causal_index
    summary['significant'] = ((p_value <= alpha) & causal).astype(np.uint8)
    summary['significant_nocausal'] = (p_value <= alpha).astype(np.uint8)

    # A peak at the edge of the scanned range means the true lag is outside it.
    _sig = summary['significant'].astype(bool)
    if _sig.any() and len(lags_used) > 1:
        _edge = np.mean(best_i[_sig] == len(lags_used) - 1)
        if _edge > 0.25:
            print('  WARNING: {:.0f}% of significant cells peak at the '
                  'LAST lag ({:+.2f} s). The response is '
                  'still rising there, so the true lag is probably outside the '
                  'scanned range -- extend --lags.'.format(_edge * 100, lags_used[-1] * dt_2p))

    n_p = int(summary['significant_nocausal'].sum())
    n_sig = int(summary['significant'].sum())
    print('  {}/{} cells pass the permutation test '
          '(p <= {:g}, family-wise over {} pixels and '
          '{} lags; smallest attainable p = '
          '{:.3g})'.format(n_p, Tb.shape[0], alpha, occ_mask.sum(), len(lags_used), 1.0 / (1 + n_shuffles)))
    print('  {}/{} of those also peak at a POSITIVE lag '
          '(the causality control)'.format(n_sig, Tb.shape[0]))

    # Two sanity checks that decide whether any of this is worth believing.
    if pos.any() and neg.any():
        n_by_lag = [(l, int(np.sum((peak_per_lag[:, k] >= np.nanpercentile(
            null_over_lags, 95, axis=0)))))
            for k, l in enumerate(lag_s_arr)]
        print('  cells above the null 95th pct, by lag: ' +
              '  '.join('{:+.2f}s:{}'.format(l, c) for l, c in n_by_lag))
        _p = np.nanmax(peak_per_lag[:, pos], axis=1)
        _n = np.nanmax(peak_per_lag[:, neg], axis=1)
        _asym = np.nanmedian(_p) / max(np.nanmedian(_n), 1e-9)
        print('  Lag asymmetry (median positive peak / median negative peak): '
              '{:.2f}'.format(_asym))
        if _asym < 1.15:
            print('  WARNING: the lag profile is close to symmetric. A visual '
                  'response cannot precede its stimulus, so this is a slow '
                  'non-causal covariation -- most likely the pillar\'s retinal '
                  'position being a deterministic function of the animal\'s '
                  'position and heading, which any place- or heading-tuned cell '
                  'will reproduce. Do NOT read these maps as receptive fields.')

    _sel = summary['significant'].astype(bool)
    if _sel.sum() >= 3:
        _M = Tb[_sel].reshape(int(_sel.sum()), -1)
        _M = _M[np.isfinite(_M).all(axis=1)]
        if len(_M) >= 3:
            _Mz = (_M - _M.mean(1, keepdims=True)) / np.where(
                _M.std(1, keepdims=True) > 0, _M.std(1, keepdims=True), np.nan)
            _C = (_Mz @ _Mz.T) / _Mz.shape[1]
            _iu = np.triu_indices(len(_C), 1)
            _med = float(np.nanmedian(_C[_iu]))
            print('  median pairwise correlation between the selected cells\' '
                  'maps: {:.3f}'.format(_med))
            if _med > 0.4:
                print('  WARNING: the selected maps are largely the SAME map. '
                      'Real receptive fields tile the visual field and should be '
                      'weakly correlated across cells; a high value means the '
                      'analysis is reporting one session-wide pattern once per '
                      'cell.')
            summary_map_corr = _med
        else:
            summary_map_corr = np.nan
    else:
        summary_map_corr = np.nan

    out = {
        'sta': STA, 'delta': D, 't': T, 'z': Z,
        't_best': Tb.astype(np.float32), 'delta_best': Db.astype(np.float32),
        't_half1': T_h1.astype(np.float32), 't_half2': T_h2.astype(np.float32),
        'light_only': np.array(bool(light_only)),
        'z_best': Zb.astype(np.float32),
        'occupancy': per_lag[0]['occupancy'],
        'occupancy_mask': occ_mask.astype(np.uint8),
        'null_peak': null_over_lags.astype(np.float32),
        'lag_frames': lags_used,
        'lag_s': lags_used * dt_2p,
        'best_lag_index': best_i.astype(np.int32),
        'best_lag_s': (lags_used[best_i] * dt_2p).astype(np.float32),
        'az': az.astype(np.float32), 'el': el.astype(np.float32),
        'n_frames_used': np.array(int(use.sum())),
        'n_cells_total': np.array(n_cells_all),
        'bin_px': np.array(int(bin_px)),
        'n_shuffles': np.array(int(n_shuffles)),
        'alpha': np.array(float(alpha)),
        'dt_2p': np.array(dt_2p),
        'spike_key': str(spike_key),
        'speed_thresh': np.array(float(speed_thresh)),
        'map_similarity': np.array(float(summary_map_corr)),
    }
    out.update({f'summary_{k}': v for k, v in summary.items()})

    print('Writing {}...'.format(out_h5))
    write_h5(out_h5, out)

    if make_pdf:
        make_rf_pdf(out, out_h5.replace('.h5', '.pdf'), alpha=alpha,
                    sn_dir=find_sparse_noise_dir(h5_path))

    return out, out_h5


def find_sparse_noise_dir(h5_path):
    """ Head-fixed sparse noise directory beside the freely moving recording.

    Looks for sn* directories holding a sparse_noise.h5 in the parent of the fm
    directory, e.g. <pos>/fm1/..._preproc.h5 gives <pos>/sn1.

    Parameters
    ----------
    h5_path : str
        Freely moving preproc .h5.

    Returns
    -------
    sn_dir : str or None
        First match in sorted order, None if there is none.
    """

    parent = os.path.dirname(os.path.dirname(os.path.abspath(h5_path)))
    hits = sorted(d for d in glob.glob(os.path.join(parent, 'sn*'))
                  if os.path.isfile(os.path.join(d, 'sparse_noise.h5')))
    if not hits:
        print('  No sparse noise directory with sparse_noise.h5 in {} -- '
              'PDF will skip the head-fixed RF panel.'.format(parent))
        return None
    print('  Head-fixed sparse noise: {}'.format(hits[0]))
    return hits[0]


def load_sparse_noise_rfs(sn_dir):
    """ Open sparse noise STAs and labels for per-cell lookup.

    STAs stay on disk and are read one cell at a time; the full array is several
    GB. Labels, split-half correlations and Gaussian fits are optional and
    only used for the panel title and centroid marker.

    Parameters
    ----------
    sn_dir : str
        Directory holding sparse_noise.h5.

    Returns
    -------
    sn : dict
        'file' (open h5py.File), 'n_cells', and optional 'labels', 'isresp',
        'corr1', 'corr2', 'centroids' ({cell: (x, y)} in monitor px).
    """

    f = h5py.File(os.path.join(sn_dir, 'sparse_noise.h5'), 'r')
    sn = {'file': f, 'n_cells': int(f['STA'].shape[0]), 'dir': sn_dir}

    lab_path = os.path.join(sn_dir, 'sparse_noise_labels_gaussfit.npz')
    if os.path.isfile(lab_path):
        z = np.load(lab_path, allow_pickle=True)
        if 'labels' in z.files:
            sn['labels'] = np.asarray(z['labels'])
        if 'true_indices' in z.files and 'pos_centroids' in z.files:
            sn['centroids'] = {int(i): tuple(xy) for i, xy
                               in zip(z['true_indices'], z['pos_centroids'])}

    rel_path = os.path.join(sn_dir, 'reliability_stats_v2.npy')
    if os.path.isfile(rel_path):
        rel = np.load(rel_path)
        # Rows: corr of full STA with half 1, with half 2, responsive flag.
        sn['corr1'], sn['corr2'], sn['isresp'] = rel[0], rel[1], rel[2]
    return sn


def _draw_sn_panel(ax, sn, c):
    """ Head-fixed sparse noise STA for one cell, or a note if unavailable. """

    ax.set_title('head-fixed sparse noise', fontsize=9)
    if sn is None or c >= sn['n_cells']:
        ax.text(0.5, 0.5, 'no sparse noise RF', ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color='0.4')
        ax.set_xticks([]); ax.set_yticks([])
        return None

    sta = np.asarray(sn['file']['STA'][c], dtype=float).reshape(SN_STA_SHAPE)
    step = SN_DISPLAY_STEP
    sta = sta[::step, ::step]
    v = np.nanmax(np.abs(sta)) or 1.0
    h, w = SN_STA_SHAPE
    im = ax.imshow(sta, origin='upper', extent=[0, w, h, 0], aspect='equal',
                   cmap=_RF_CMAP, vmin=-v, vmax=v)
    ax.set_xlabel('Monitor x (px)', fontsize=8)
    ax.set_ylabel('Monitor y (px)', fontsize=8)

    if 'centroids' in sn and c in sn['centroids']:
        cx, cy = sn['centroids'][c]
        if 0 <= cx <= w and 0 <= cy <= h:
            ax.plot(cx, cy, '+', color='k', ms=10, mew=1.5)

    notes = []
    if 'labels' in sn and c < len(sn['labels']):
        lab = int(sn['labels'][c])
        notes.append({1: 'labeled RF', 0: 'labeled no RF'}.get(lab, 'unlabeled'))
    if 'corr1' in sn and c < len(sn['corr1']):
        notes.append('split r {:.2f}/{:.2f}'.format(sn['corr1'][c], sn['corr2'][c]))
    if notes:
        ax.set_title('head-fixed sparse noise\n' + ', '.join(notes), fontsize=9)
    return im


def make_rf_pdf(res, out_pdf, alpha=0.05, max_pages=200, sn_dir=None, n_best_png=15):
    """ One page per reliable sparse noise RF cell: full map, both halves, lag, sparse noise RF.

    Parameters
    ----------
    res : dict
        Output of rf_from_reconstruction.
    out_pdf : str
    alpha : float
        Significance level reported per page; only selects cells when there
        are no sparse noise labels.
    max_pages : int
    sn_dir : str or None
        Head-fixed sparse noise directory.  Pages are the cells labeled 1 in its
        sparse_noise_labels_gaussfit.npz; last panel is left blank if None.
    n_best_png : int
        The top cells (same order as the PDF) also go to <pdf stem>_best<N>.png,
        without the lag panel.
        Cell indices are assumed to match (suite2p run jointly, then split).
    """

    az, el = res['az'], res['el']
    Tb, T = res['t_best'], res['t']
    H1 = res.get('t_half1'); H2 = res.get('t_half2')
    lag_s = np.asarray(res['lag_s'], dtype=float)
    pv = res['summary_p_value']; peak = res['summary_peak_t']

    sn = load_sparse_noise_rfs(sn_dir) if sn_dir is not None else None
    if sn is not None and sn['n_cells'] != len(pv):
        print('  WARNING: sparse noise has {} cells, freely moving has {} -- '
              'cell indices may not correspond.'.format(sn['n_cells'], len(pv)))

    # Cells come from the head-fixed labels: only those hand-labeled as having a
    # reliable sparse noise RF, whatever their freely moving p-value.  Without
    # labels, fall back to the freely moving significance test.
    order = np.argsort(np.where(np.isfinite(peak), peak, -np.inf))[::-1]
    use_labels = sn is not None and 'labels' in sn
    if use_labels:
        order = [c for c in order if c < len(sn['labels']) and sn['labels'][c] == 1]
        n_sig = sum(1 for c in order if np.isfinite(pv[c]) and pv[c] <= alpha)
        print('  {} cells labeled with a reliable sparse noise RF ({} also at '
              'freely moving p <= {:g}).'.format(len(order), n_sig, alpha))
        if not order:
            print('  No labeled sparse noise RF cells; PDF not written')
            if sn is not None:
                sn['file'].close()
            return
    else:
        print('  WARNING: no sparse noise labels found -- selecting cells by '
              'freely moving p <= {:g} instead.'.format(alpha))
        order = [c for c in order if np.isfinite(pv[c]) and pv[c] <= alpha]
        if not order:
            print('  No cells at p <= {:g}; PDF not written'.format(alpha))
            if sn is not None:
                sn['file'].close()
            return
    order = order[:max_pages]
    ext = [az.min(), az.max(), el.min(), el.max()]

    def _show(ax, M, title):
        """ imshow a map with symmetric color limits and az/el axes. """

        v = np.nanmax(np.abs(M)) or 1.0
        im = ax.imshow(M, origin='upper', extent=ext, aspect='auto',
                       cmap=_RF_CMAP, vmin=-v, vmax=v)   # symmetric: sign readable
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Azimuth (deg)', fontsize=8)
        ax.set_ylabel('Elevation (deg)', fontsize=8)
        ax.axhline(0, color='0.4', lw=0.6); ax.axvline(0, color='0.4', lw=0.6)
        return im

    def _full(fig, ax, c):
        """ Full-recording map with its RF threshold contour. """

        im = _show(ax, Tb[c], f'cell {c}  full (peak {peak[c]:.1f}, p={pv[c]:.3g})')
        fig.colorbar(im, ax=ax)
        thr = res['summary_thresh'][c]
        if np.isfinite(thr):
            ax.contour(Tb[c], levels=[thr], colors='k', linewidths=0.9,
                       extent=ext, origin='upper')

    def _halves(fig, ax1, ax2, c):
        """ Split-half maps on one shared scale, with their correlation. """

        if H1 is None:
            for a in (ax1, ax2):
                a.set_axis_off()
            return
        # Both halves on ONE shared scale, so they are comparable by eye.
        v = np.nanmax([np.nanmax(np.abs(H1[c])), np.nanmax(np.abs(H2[c])), 1e-9])
        for a, M, t in ((ax1, H1[c], 'first half'), (ax2, H2[c], 'second half')):
            imh = a.imshow(M, origin='upper', extent=ext, aspect='auto',
                           cmap=_RF_CMAP, vmin=-v, vmax=v)
            a.set_title(t, fontsize=9)
            a.set_xlabel('Azimuth (deg)', fontsize=8)
            a.axhline(0, color='0.4', lw=0.6); a.axvline(0, color='0.4', lw=0.6)
        fig.colorbar(imh, ax=ax2)
        g = np.isfinite(H1[c]) & np.isfinite(H2[c])
        r = (np.corrcoef(H1[c][g], H2[c][g])[0, 1] if g.sum() > 10 else np.nan)
        ax1.set_ylabel('Elevation (deg)', fontsize=8)
        ax2.text(0.02, 0.02, f'half-half r = {r:+.2f}',
                 transform=ax2.transAxes, fontsize=9, color='k',
                 bbox=dict(fc='w', alpha=0.7, ec='none'))

    print('Writing {} ({} cells)...'.format(out_pdf, len(order)))
    with PdfPages(out_pdf) as pdf:
        for c in order:
            fig, axs = plt.subplots(1, 5, figsize=(22, 3.9),
                                    gridspec_kw={'width_ratios': [1, 1, 1, 1, 1.4]})
            _full(fig, axs[0], c)
            _halves(fig, axs[1], axs[2], c)

            with np.errstate(invalid='ignore'):
                prof = np.nanmax(T[c].reshape(T.shape[1], -1), axis=1)
            axs[3].plot(lag_s, prof, 'o-', color='#c0392b', label='observed peak')
            if 'null_peak' in res and np.isfinite(res['null_peak']).any():
                axs[3].axhline(np.nanpercentile(res['null_peak'][:, c], 95),
                               color='0.5', ls='--', lw=1, label='null 95th pct')
            axs[3].axvline(0, color='0.7', lw=0.8)
            axs[3].set_xlabel('Lag (s)'); axs[3].set_ylabel('peak statistic')
            axs[3].set_title(f'best lag {res["best_lag_s"][c]:+.2f} s', fontsize=9)
            axs[3].legend(fontsize=7)

            im4 = _draw_sn_panel(axs[4], sn, c)
            if im4 is not None:
                fig.colorbar(im4, ax=axs[4])
            fig.suptitle(f'cell {c} -- {res["summary_area_deg2"][c]:.0f} deg^2 at '
                         f'({res["summary_peak_az"][c]:+.0f}, '
                         f'{res["summary_peak_el"][c]:+.0f}) deg', fontsize=10)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print('  Wrote {}'.format(out_pdf))

    # Best cells as one PNG, one row per cell: full map, both halves, sparse
    # noise RF.  No lag profile.
    best = order[:n_best_png]
    if best:
        out_png = os.path.splitext(out_pdf)[0] + '_best{}.png'.format(len(best))
        fig, axs = plt.subplots(len(best), 4, figsize=(20, 3.6 * len(best)),
                                gridspec_kw={'width_ratios': [1, 1, 1, 1.4]},
                                squeeze=False)
        for row, c in enumerate(best):
            _full(fig, axs[row, 0], c)
            _halves(fig, axs[row, 1], axs[row, 2], c)
            im = _draw_sn_panel(axs[row, 3], sn, c)
            if im is not None:
                fig.colorbar(im, ax=axs[row, 3])
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print('  Wrote {}'.format(out_png))

    if sn is not None:
        sn['file'].close()


def main():
    """ Command-line entry point. """

    p = argparse.ArgumentParser(
        description='Spatial receptive fields from the reconstructed retinal image.')
    p.add_argument('--npz_path', required=True,
                   help='<prefix>_retinal_images.npz from get_retinal_image')
    p.add_argument('--h5_path', default=None,
                   help='<prefix>_preproc.h5 (default: derived from npz_path)')
    p.add_argument('--out_h5', default=None,
                   help='Output path (default: <prefix>_retinal_rf.h5)')
    p.add_argument('--lags', type=float, nargs='+', default=list(DEFAULT_LAGS_S),
                   help='Stimulus-to-response lags in seconds (default: %(default)s)')
    p.add_argument('--bin_px', type=int, default=2,
                   help='Spatial binning of the retinal image (default: 2 -> 60x60)')
    p.add_argument('--n_shuffles', type=int, default=100,
                   help='Circular shifts for the null (default: %(default)s). '
                        'The smallest attainable p is 1/(n+1).')
    p.add_argument('--spike_key', default='norm_spikes',
                   help='HDF5 key for the spike rate (default: %(default)s)')
    p.add_argument('--speed_thresh', type=float, default=SPEED_THRESH_CM_S,
                   help='Motion filter in cm/s; 0 disables (default: %(default)s)')
    p.add_argument('--alpha', type=float, default=0.05,
                   help='Family-wise significance level (default: %(default)s)')
    p.add_argument('--min_occupancy', type=float, default=0.002,
                   help='Mask pixels below this fraction of peak occupancy '
                        '(default: %(default)s)')
    p.add_argument('--max_cells', type=int, default=None,
                   help='Only process the first N cells (for quick tests)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--all_light_dark', action='store_true',
                   help='Fit on all frames instead of light only. The default '
                        'is light-only: the reconstruction renders the pillar '
                        'in darkness too, so dark frames regress spikes against '
                        'a stimulus the animal cannot see.')
    p.add_argument('--pdf', action='store_true',
                   help='Also write a PDF, one page per significant cell')
    p.add_argument('--split_mode', choices=['interleaved', 'contiguous'],
                   default='interleaved',
                   help='How the split-half maps divide the frames. '
                        'interleaved (default) alternates blocks so drift '
                        'cannot masquerade as an RF; contiguous is the old '
                        'first-vs-second-half behavior.')
    p.add_argument('--split_block_s', type=float, default=60.0,
                   help='Block length for --split_mode interleaved, seconds.')
    p.add_argument('--split_stat', choices=['delta', 't'], default='delta',
                   help='Statistic shown in the split-half maps. delta '
                        '(default) is the occupancy-subtracted effect size; t '
                        'additionally divides by sd_S, which inflates '
                        'rarely-visited pixels.')
    args = p.parse_args()

    rf_from_reconstruction(
        npz_path=args.npz_path, h5_path=args.h5_path, out_h5=args.out_h5,
        lags_s=args.lags, bin_px=args.bin_px, n_shuffles=args.n_shuffles,
        spike_key=args.spike_key, speed_thresh=args.speed_thresh,
        alpha=args.alpha, min_occupancy=args.min_occupancy,
        max_cells=args.max_cells, seed=args.seed, make_pdf=args.pdf,
        light_only=not args.all_light_dark,
        split_mode=args.split_mode, split_block_s=args.split_block_s,
        split_stat=args.split_stat,
    )


if __name__ == '__main__':
    main()


# python -m fm2p.rf_from_reconstruction --npz_path /home/dylan/Storage/freely_moving_data/V1PPC/cohort02_recordings/251016_DMM_DMM061_pos18/fm1/251016_DMM_DMM061_fm_01_retinal_images_retinotopic_fixed.npz --pdf --spike_key norm_spikes_s2p
