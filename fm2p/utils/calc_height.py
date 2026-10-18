# -*- coding: utf-8 -*-
"""
fm2p/utils/calc_height.py

Head height above the arena floor, from topdown headbar length and IMU tilt.

The headbar is rigid and a known length, so how long it looks from the
overhead camera is a distance measurement: a raised head sits closer to the
lens and the bar covers more pixels.  What breaks that on its own is tilt --
a rolled head foreshortens the bar exactly the way a raised one lengthens it
-- and tilt is the one thing the accelerometer measures absolutely.  So this
is a camera measurement with an IMU correction, not an IMU measurement.

Nothing here integrates the accelerometer.  Height is not recoverable from a
6-axis IMU alone at any useful timescale: it takes two integrations, the
vertical channel is formally unstable, and no part of the signal ever refers
to the floor.  See Woodman (2007), UCAM-CL-TR-696, and Fayat et al. (2021),
Sensors 21(18):6318.  The closest published version of what this module does
instead is Vagvolgyi et al. (2022), J Neurosci Methods 368:109453, who
recover 6-DOF rodent head pose from one camera and a known marker target --
they report 9.75 mm mean error on the depth axis against a commercial
tracker, which is about what to expect here.

Precision depends entirely on how many pixels the bar covers, so
calc_head_height() measures that and reports the achievable error before you
trust anything it returns.

Functions
---------
calc_head_height
    Head height per 2P frame, with a feasibility report.
add_head_height
    Run calc_head_height and write its outputs back into a data dict.
detect_rearing
    Boolean rearing mask from a height trace.
calibrate_bar_constant
    Solve the focal-length times bar-length product from reference frames.
foreshortening
    Fraction of bar length surviving projection at a given roll and pitch.
bar_geometry
    Headbar endpoints, length, midpoint and direction in topdown pixels.
plot_height_diagnostics
    Four-panel diagnostic figure for a fitted height trace.
_solve_depth
    Camera-to-bar distance from the exact pinhole projection of a segment.
_estimate_noise
    Robust per-frame noise on a height trace.
_principal_point
    Optical axis position in topdown pixels.
main
    CLI: fit head height for one session and plot it.


DMM, September 2026
"""


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .filter import nanmedfilt
from .files import read_h5


# Overhead camera height above the arena floor, mm.  Matches
# TOPDOWN_CAMERA_HEIGHT_MM in get_retinal_image.py; the two should stay equal.
CAMERA_HEIGHT_MM = 42 * 25.4

# Height the headbar rides at during normal locomotion, mm.  Same 25 mm
# get_retinal_image assumes for the whole recording, used here only to pin the
# calibration constant when the true bar length is unknown.
WALK_HEIGHT_MM = 25.0

# Frames used to calibrate: moving, level, both markers tracked.  Roll has to
# be small because the calibration cannot separate a short bar from a rolled
# one.
WALK_SPEED_MIN_CM_S = 5.0
WALK_ROLL_MAX_DEG = 10.0

# Rearing threshold, mm above the walking height, and the shortest run of
# frames that counts.  A rear is 50-75 mm so 20 mm sits well clear of noise.
REAR_THRESH_MM = 20.0
REAR_MIN_FRAMES = 2

# lear/rear are already nanmedfilt'd with a width of 7 in topcam.track_body,
# so extra filtering here is off by default.
BAR_MEDFILT_N = 1


def foreshortening(roll_deg, pitch_deg):
    """ Fraction of bar length surviving projection at a given roll and pitch.

    Bar runs along the head's lateral axis.  Under the intrinsic Z-Y'-X''
    convention the renderer uses, that axis is the second column of
    Rz(yaw) Ry(pitch) Rx(roll), whose vertical component is
    cos(pitch) sin(roll).  Horizontal extent -- all an overhead camera sees --
    is what is left.

    Pitch alone does nothing: pitching rotates the bar about its own axis.
    Roll is the whole effect, and it is not small -- 20 deg of roll shortens
    the bar 6%, which is a whole rear's worth of apparent height.

    Parameters
    ----------
    roll_deg : np.ndarray
        Head roll, degrees.
    pitch_deg : np.ndarray
        Head pitch, degrees.

    Returns
    -------
    c : np.ndarray
        Foreshortening factor in [0, 1].
    b_z : np.ndarray
        Vertical component of the bar's unit vector, cos(pitch) sin(roll).
    """

    roll = np.radians(np.asarray(roll_deg, dtype=float))
    pitch = np.radians(np.asarray(pitch_deg, dtype=float))

    b_z = np.cos(pitch) * np.sin(roll)
    c = np.sqrt(np.clip(1.0 - b_z ** 2, 0.0, 1.0))

    return c, b_z


def bar_geometry(data, medfilt_n=BAR_MEDFILT_N):
    """ Headbar endpoints, length, midpoint and direction in topdown pixels.

    Parameters
    ----------
    data : dict
        Needs 'lear_x', 'lear_y', 'rear_x', 'rear_y' (topdown px).
    medfilt_n : int
        Extra median filter width on the endpoints.  Off by default, since
        topcam.track_body already filtered these with a width of 7.

    Returns
    -------
    geom : dict
        'L_px' (length), 'mid_x'/'mid_y' (midpoint), 'ux'/'uy' (unit vector
        from right marker to left), all length n_frames.
    """

    def _get(key):
        v = np.asarray(data[key], dtype=float).squeeze()
        if medfilt_n and int(medfilt_n) > 1:
            v = np.asarray(nanmedfilt(v, int(medfilt_n))).squeeze()
        return v

    lx, ly = _get('lear_x'), _get('lear_y')
    rx, ry = _get('rear_x'), _get('rear_y')

    n = min(len(lx), len(ly), len(rx), len(ry))
    lx, ly, rx, ry = lx[:n], ly[:n], rx[:n], ry[:n]

    dx = lx - rx
    dy = ly - ry
    L_px = np.hypot(dx, dy)

    # Unit vector along the bar.  Zero-length frames would divide by zero;
    # they are untracked anyway, so send them to NaN.
    with np.errstate(invalid='ignore', divide='ignore'):
        ux = np.where(L_px > 0, dx / L_px, np.nan)
        uy = np.where(L_px > 0, dy / L_px, np.nan)

    return {
        'L_px':  L_px,
        'mid_x': 0.5 * (lx + rx),
        'mid_y': 0.5 * (ly + ry),
        'ux':    ux,
        'uy':    uy,
    }


def _principal_point(data, geom):
    """ Optical axis position in topdown pixels.

    Only matters for the off-axis term in _solve_depth, which needs the bar's
    offset from the axis rather than from an arbitrary image corner.  Arena
    corners are the best available proxy for where the camera was aimed;
    falling back to the mean bar position is fine when the head covers the
    arena roughly evenly.
    """

    corners = [k for k in ('arenaTL', 'arenaTR', 'arenaBR', 'arenaBL') if k in data]
    if len(corners) == 4:
        try:
            cx = float(np.mean([float(np.asarray(data[k]['x']).squeeze()) for k in corners]))
            cy = float(np.mean([float(np.asarray(data[k]['y']).squeeze()) for k in corners]))
            return cx, cy, 'arena corners'
        except (KeyError, TypeError, ValueError):
            pass

    return (float(np.nanmean(geom['mid_x'])),
            float(np.nanmean(geom['mid_y'])),
            'mean bar position')


def _solve_depth(dpx, dpy, offx, offy, c, b_z, f_px, bar_mm):
    """ Camera-to-bar distance from the exact pinhole projection of a segment.

    Naively the bar covers f * D * c / R pixels at distance R, which inverts
    to R = f * D * c / L.  That is exact only when the bar is horizontal.
    Once it rolls, its two ends sit at different depths, and a segment with
    ends at different depths projects differently depending on how far off the
    optical axis it is.

    Writing the ends as the bar midpoint plus and minus (D/2) b, with b the
    bar's unit vector, and projecting both through a pinhole at height H, the
    image separation dp and image midpoint p_mid satisfy exactly

        R * dp / 2  -  (D/2) * b_z * p_mid  =  f * (D/2) * b_xy

    as a vector identity.  Taking its norm kills the unknown direction of
    b_xy -- only its length (D/2) c is needed, which the IMU gives -- and
    leaves a quadratic in R:

        R^2 |A|^2  -  2 R (A.B)  +  |B|^2 - k^2 = 0

    with A = dp/2, B = (D/2) b_z p_mid, k = f (D/2) c.  Positive root is the
    physical one, and it reduces to R = f D c / L when b_z is zero.

    Projecting onto the measured bar direction instead would be a whole
    degree cheaper and wrong by several mm, because roll skews the image
    direction of the bar away from b_xy.

    Parameters
    ----------
    dpx, dpy : np.ndarray
        Left marker minus right marker, px.
    offx, offy : np.ndarray
        Bar midpoint offset from the optical axis, px.
    c : np.ndarray
        Foreshortening factor from foreshortening().
    b_z : np.ndarray
        Vertical component of the bar unit vector.
    f_px : float
        Camera focal length in pixels.
    bar_mm : float
        True bar length, mm.

    Returns
    -------
    R_mm : np.ndarray
        Distance from the lens to the bar midpoint, mm.
    """

    half = 0.5 * float(bar_mm)

    Ax, Ay = 0.5 * dpx, 0.5 * dpy
    Bx, By = half * b_z * offx, half * b_z * offy
    k = f_px * half * c

    AA = Ax ** 2 + Ay ** 2
    AB = Ax * Bx + Ay * By
    BB = Bx ** 2 + By ** 2

    with np.errstate(invalid='ignore', divide='ignore'):
        disc = AB ** 2 - AA * (BB - k ** 2)
        R_mm = (AB + np.sqrt(disc)) / AA
        R_mm = np.where((AA > 0) & (disc >= 0), R_mm, np.nan)

    return R_mm


def _estimate_noise(z_mm):
    """ Robust per-frame noise on a height trace, in mm.

    Second differences, scaled by the usual MAD constants (var is 6 sigma^2
    for white noise).  Real height changes are smooth across three frames at
    7.5 Hz -- a rear takes about a second -- so what survives differencing
    twice is mostly jitter.

    Two things this deliberately does not do.  It does not difference L_px:
    roll moves the projected bar length around on its own, and measuring
    there charges real head rotation to the noise budget.  And it does not
    gate on low speed to find "quiet" frames, because mice stop to rear, so
    that selects exactly the frames where height changes fastest.

    Still an overestimate, since genuine fast head bobs land in it, and an
    underestimate of raw DLC error, since topcam already median filtered the
    endpoints and that correlates neighbouring frames.

    Parameters
    ----------
    z_mm : np.ndarray
        Height trace, mm.

    Returns
    -------
    sigma_z : float
        Estimated per-frame noise, mm.
    """

    d = np.diff(np.asarray(z_mm, dtype=float), 2)
    d = d[np.isfinite(d)]
    if d.size < 10:
        return np.nan

    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(6.0))


def calibrate_bar_constant(L_px, c, speed, roll_deg, walk_height_mm=WALK_HEIGHT_MM,
                           camera_height_mm=CAMERA_HEIGHT_MM,
                           speed_min=WALK_SPEED_MIN_CM_S, roll_max=WALK_ROLL_MAX_DEG):
    """ Solve the focal-length times bar-length product from reference frames.

    f and D only ever appear multiplied, so the caliper measurement is
    optional: pin the product instead by asserting a height the animal is
    known to be at.  Frames where it is walking and level are the ones where
    that assertion holds.

    Only the mean is being fixed here.  Absolute height is as good as the
    walk_height_mm assumption; height *changes* do not depend on it at all.

    Parameters
    ----------
    L_px : np.ndarray
        Measured bar length, px.
    c : np.ndarray
        Foreshortening factor.
    speed : np.ndarray or None
        Speed, cm/s.  All frames are used if None.
    roll_deg : np.ndarray
        Head roll, degrees.
    walk_height_mm : float
        Height the bar is assumed to ride at on the reference frames.
    camera_height_mm : float
        Lens height above the floor, mm.
    speed_min : float
        Lower speed bound for a reference frame, cm/s.
    roll_max : float
        Upper |roll| bound for a reference frame, degrees.

    Returns
    -------
    K : float
        Product f * D, in px * mm.
    n_ref : int
        Reference frames used.
    """

    ok = np.isfinite(L_px) & np.isfinite(c) & (L_px > 0)
    ok &= np.isfinite(roll_deg) & (np.abs(roll_deg) <= roll_max)
    if speed is not None:
        ok &= np.isfinite(speed) & (speed >= speed_min)

    # Level, moving frames can be scarce in a short session; fall back to
    # every tracked frame rather than returning nothing.
    if ok.sum() < 20:
        ok = np.isfinite(L_px) & np.isfinite(c) & (L_px > 0)

    if ok.sum() < 5:
        return np.nan, int(ok.sum())

    K = float(np.median(L_px[ok] * (camera_height_mm - walk_height_mm) / c[ok]))

    return K, int(ok.sum())


def detect_rearing(z_mm, baseline_mm=None, thresh_mm=REAR_THRESH_MM,
                   min_frames=REAR_MIN_FRAMES):
    """ Boolean rearing mask from a height trace.

    Parameters
    ----------
    z_mm : np.ndarray
        Head height above the floor, mm.
    baseline_mm : float or None
        Walking height.  Median of the trace when None.
    thresh_mm : float
        How far above baseline counts as a rear.
    min_frames : int
        Runs shorter than this are dropped.

    Returns
    -------
    is_rearing : np.ndarray
        Boolean, same length as z_mm.
    """

    z = np.asarray(z_mm, dtype=float)
    if baseline_mm is None:
        baseline_mm = float(np.nanmedian(z))

    is_rearing = np.zeros(z.shape, dtype=bool)
    is_rearing[np.isfinite(z)] = z[np.isfinite(z)] > (baseline_mm + thresh_mm)

    if min_frames <= 1:
        return is_rearing

    # Drop runs shorter than min_frames.
    edges = np.diff(np.concatenate([[0], is_rearing.astype(int), [0]]))
    starts = np.where(edges == 1)[0]
    stops = np.where(edges == -1)[0]
    for a, b in zip(starts, stops):
        if (b - a) < min_frames:
            is_rearing[a:b] = False

    return is_rearing


def calc_head_height(data, bar_length_mm=None, camera_height_mm=CAMERA_HEIGHT_MM,
                     walk_height_mm=WALK_HEIGHT_MM, medfilt_n=BAR_MEDFILT_N,
                     smooth_n=1, roll_sign='auto', offaxis_correct=True,
                     principal_point=None, verbose=True):
    """ Head height per 2P frame, with a feasibility report.

    Parameters
    ----------
    data : dict, str or Path
        Preprocessed session dict, or a path to one.  Needs 'lear_x',
        'lear_y', 'rear_x', 'rear_y', 'pxls2cm', 'roll_twop_interp' and
        'pitch_twop_interp'.
    bar_length_mm : float or None
        True marker separation on the headbar.  When None the calibration
        constant is pinned to walk_height_mm instead and absolute height
        inherits that assumption.
    camera_height_mm : float
        Lens height above the arena floor, mm.
    walk_height_mm : float
        Height the bar rides at during locomotion, mm.
    medfilt_n : int
        Extra median filter on the bar endpoints.  Off by default.
    smooth_n : int
        Median filter width applied to the height trace itself.  Trading
        time resolution for precision here is usually worth it.
    roll_sign : {'auto', 1, -1}
        Sign relating stored roll to the bar's vertical component.  Depends
        on which marker is 'lear' and on the IMU mount convention, neither of
        which is worth assuming; 'auto' picks whichever leaves the height
        trace less correlated with the off-axis regressor.
    offaxis_correct : bool
        Use the exact projective solve.  False falls back to the scalar
        R = f D c / L, which is what most descriptions of this method give.
    principal_point : (float, float) or None
        Optical axis in topdown px.  Inferred from arena corners when None.
    verbose : bool
        Print the feasibility report.

    Returns
    -------
    z_mm : np.ndarray
        Head height above the floor, mm, one per 2P frame.
    info : dict
        Calibration constants, the noise budget, and 'is_rearing'.
    """

    if isinstance(data, (str, Path)):
        data = read_h5(data)

    geom = bar_geometry(data, medfilt_n=medfilt_n)
    n = len(geom['L_px'])

    roll = np.asarray(data['roll_twop_interp'], dtype=float).squeeze()[:n]
    pitch = np.asarray(data['pitch_twop_interp'], dtype=float).squeeze()[:n]
    if len(roll) < n or len(pitch) < n:
        n = min(len(roll), len(pitch))
        geom = {k: v[:n] for k, v in geom.items()}
        roll, pitch = roll[:n], pitch[:n]

    # Speed is one short of the frame count: it comes from a diff in
    # topcam.track_body.
    speed = np.asarray(data['speed'], dtype=float).squeeze() if 'speed' in data else None
    if speed is not None:
        speed = np.concatenate([speed, [np.nan]])[:n] if len(speed) < n else speed[:n]

    # px per mm at the floor; f follows from the pinhole model since the
    # corner calibration that sets pxls2cm was done at floor level.
    s_floor = float(data['pxls2cm']) / 10.0
    f_px = s_floor * camera_height_mm

    if principal_point is None:
        cx, cy, pp_src = _principal_point(data, geom)
    else:
        cx, cy = float(principal_point[0]), float(principal_point[1])
        pp_src = 'caller'

    # Bar midpoint offset from the optical axis.  Its component along the
    # bar is what the roll-sign check keys on.
    offx = geom['mid_x'] - cx
    offy = geom['mid_y'] - cy
    mid_off = offx * geom['ux'] + offy * geom['uy']

    dpx = geom['L_px'] * geom['ux']
    dpy = geom['L_px'] * geom['uy']

    L_px = geom['L_px']

    # Foreshortening is even in roll, so c and the calibration constant do
    # not depend on the sign convention and can be settled first.
    c, b_z_raw = foreshortening(roll, pitch)

    K, n_ref = calibrate_bar_constant(L_px, c, speed, roll,
                                      walk_height_mm=walk_height_mm,
                                      camera_height_mm=camera_height_mm)
    if not np.isfinite(K):
        raise ValueError('Could not calibrate: no usable headbar frames.')

    bar_mm = float(bar_length_mm) if bar_length_mm is not None else K / f_px

    # Scalar estimate, which the off-axis term has not been removed from.
    with np.errstate(invalid='ignore', divide='ignore'):
        z_scalar = camera_height_mm - f_px * bar_mm * c / np.where(L_px > 0, L_px, np.nan)

    # Sign convention relating stored roll to the bar's vertical component
    # depends on which marker is 'lear' and on the IMU mount, neither worth
    # assuming.  To first order the scalar estimate carries the off-axis term
    # additively, z_scalar = z_true + s * q, and true height has no reason to
    # track q, so the sign of cov(z_scalar, q) is s.  Keying this on the
    # corrected estimate instead does not work: both it and q flip together,
    # and the correlation cannot tell the two cases apart.
    with np.errstate(invalid='ignore', divide='ignore'):
        q = bar_mm * b_z_raw * mid_off / np.where(L_px > 0, L_px, np.nan)

    m = np.isfinite(z_scalar) & np.isfinite(q)
    if roll_sign == 'auto':
        if m.sum() > 10 and np.std(q[m]) > 0:
            cov = float(np.cov(z_scalar[m], q[m])[0, 1])
            sign = 1.0 if cov >= 0 else -1.0
            sign_score = abs(float(np.corrcoef(z_scalar[m], q[m])[0, 1]))
        else:
            sign, sign_score = 1.0, np.nan
    else:
        sign, sign_score = float(roll_sign), np.nan

    if offaxis_correct:
        R_mm = _solve_depth(dpx, dpy, offx, offy, c, sign * b_z_raw, f_px, bar_mm)
        z_mm = camera_height_mm - R_mm
    else:
        z_mm = z_scalar

    if smooth_n and int(smooth_n) > 1:
        z_mm = np.asarray(nanmedfilt(z_mm, int(smooth_n))).squeeze()

    # Noise budget.  dL/dz = L / (H - z), so one pixel of length error is
    # (H - z) / L millimetres of height.
    L_med = float(np.nanmedian(L_px))
    R_med = camera_height_mm - walk_height_mm
    px_per_mm_z = L_med / R_med
    sigma_z = _estimate_noise(z_mm)
    sigma_L = sigma_z * px_per_mm_z if np.isfinite(sigma_z) else np.nan
    sigma_z_sm = sigma_z / np.sqrt(max(int(smooth_n), 1)) if np.isfinite(sigma_z) else np.nan

    is_rearing = detect_rearing(z_mm, baseline_mm=walk_height_mm)

    info = {
        'f_px':              f_px,
        'K_px_mm':           K,
        'n_calib_frames':    n_ref,
        'bar_length_mm':     bar_mm,
        'bar_length_source': 'measured' if bar_length_mm is not None else 'calibrated',
        'roll_sign':         sign,
        'roll_sign_score':   sign_score,
        'principal_point':   (cx, cy),
        'principal_source':  pp_src,
        'L_px_median':       L_med,
        'px_per_mm_z':       px_per_mm_z,
        'sigma_L_px':        sigma_L,
        'sigma_z_mm':        sigma_z,
        'sigma_z_smoothed_mm': sigma_z_sm,
        'n_frames':          int(n),
        'n_tracked':         int(np.isfinite(z_mm).sum()),
        'is_rearing':        is_rearing,
        'offaxis_correct':   bool(offaxis_correct),
    }

    if verbose:
        print('[fm2p] Headbar height estimate')
        print('  Camera {:.0f} mm up, floor scale {:.3f} px/mm, '
              'f = {:.0f} px'.format(camera_height_mm, s_floor, f_px))
        print('  Bar {:.2f} mm ({} on {} frames), optical axis from {}'.format(
            bar_mm, info['bar_length_source'], n_ref, pp_src))
        print('  Bar covers {:.1f} px, so 1 px of length error is '
              '{:.1f} mm of height'.format(L_med, 1.0 / px_per_mm_z))
        if np.isfinite(sigma_z):
            print('  Measured sigma_z = {:.1f} mm/frame ({:.2f} px of bar '
                  'length), {:.1f} mm after smoothing'.format(
                      sigma_z, sigma_L, sigma_z_sm))
            snr = 60.0 / sigma_z if sigma_z > 0 else np.inf
            print('  A 60 mm rear is {:.1f} sigma per frame -- {}'.format(
                snr, 'detectable' if snr > 2 else 'NOT reliably detectable'))
        if roll_sign == 'auto':
            print('  Roll sign {:+.0f} chosen (off-axis corr {:.3f})'.format(
                sign, sign_score))
        print('  Tracked on {}/{} frames, rearing on {:.1f}%'.format(
            info['n_tracked'], n, 100.0 * is_rearing.mean()))
        print('  Height: median {:.1f} mm, 5-95 pct [{:.1f}, {:.1f}] mm'.format(
            np.nanmedian(z_mm), np.nanpercentile(z_mm, 5), np.nanpercentile(z_mm, 95)))

    return z_mm, info


def add_head_height(data, **kwargs):
    """ Run calc_head_height and write its outputs back into a data dict.

    Adds 'head_z_mm' and 'is_rearing' at the 2P rate, padded with NaN or False
    if the tracking ran short of twopT.

    Parameters
    ----------
    data : dict
        Preprocessed session dict, modified in place.
    **kwargs
        Passed through to calc_head_height.

    Returns
    -------
    data : dict
        Same dict, with the two new keys.
    """

    z_mm, info = calc_head_height(data, **kwargs)

    n_twop = len(np.asarray(data['twopT']).squeeze()) if 'twopT' in data else len(z_mm)
    if len(z_mm) < n_twop:
        pad = n_twop - len(z_mm)
        z_mm = np.concatenate([z_mm, np.full(pad, np.nan)])
        info['is_rearing'] = np.concatenate([info['is_rearing'], np.zeros(pad, bool)])

    data['head_z_mm'] = z_mm[:n_twop]
    data['is_rearing'] = info['is_rearing'][:n_twop]

    return data


def plot_height_diagnostics(z_mm, info, data=None, savepath=None):
    """ Four-panel diagnostic figure for a fitted height trace.

    Panels: height over time with rears marked, the height histogram, height
    against roll (should be flat if the foreshortening correction worked),
    and height against distance off the optical axis (should also be flat).

    Parameters
    ----------
    z_mm : np.ndarray
        Height trace from calc_head_height.
    info : dict
        Its info dict.
    data : dict or None
        Session dict, for roll and the bar position.
    savepath : str, Path or None
        Written here when given, otherwise the figure is returned.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """

    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    n = len(z_mm)
    rear = info.get('is_rearing')

    ax = axs[0, 0]
    ax.plot(z_mm, lw=0.6, color='k')
    if rear is not None and rear.any():
        ax.fill_between(np.arange(n), *ax.get_ylim(), where=rear[:n],
                        color='tab:red', alpha=0.25, lw=0)
    ax.set_xlabel('2P frame')
    ax.set_ylabel('head height (mm)')
    ax.set_title('Height, rears shaded')

    ax = axs[0, 1]
    finite = z_mm[np.isfinite(z_mm)]
    if finite.size:
        ax.hist(finite, bins=60, color='0.4')
    if np.isfinite(info.get('sigma_z_mm', np.nan)):
        ax.set_title('Distribution (sigma {:.1f} mm/frame)'.format(info['sigma_z_mm']))
    else:
        ax.set_title('Distribution')
    ax.set_xlabel('head height (mm)')
    ax.set_ylabel('frames')

    ax = axs[1, 0]
    if data is not None and 'roll_twop_interp' in data:
        roll = np.asarray(data['roll_twop_interp'], dtype=float).squeeze()[:n]
        m = np.isfinite(roll) & np.isfinite(z_mm)
        ax.plot(roll[m], z_mm[m], '.', ms=1, alpha=0.3, color='tab:blue')
        if m.sum() > 10 and np.std(roll[m]) > 0:
            r = float(np.corrcoef(roll[m], z_mm[m])[0, 1])
            ax.set_title('Height vs roll (r = {:.3f}, want ~0)'.format(r))
        ax.set_xlabel('roll (deg)')
        ax.set_ylabel('head height (mm)')
    else:
        ax.set_axis_off()

    ax = axs[1, 1]
    if data is not None and all(k in data for k in ('lear_x', 'lear_y', 'rear_x', 'rear_y')):
        geom = bar_geometry(data)
        cx, cy = info.get('principal_point', (np.nan, np.nan))
        rad = np.hypot(geom['mid_x'][:n] - cx, geom['mid_y'][:n] - cy)
        m = np.isfinite(rad) & np.isfinite(z_mm)
        ax.plot(rad[m], z_mm[m], '.', ms=1, alpha=0.3, color='tab:green')
        if m.sum() > 10 and np.std(rad[m]) > 0:
            r = float(np.corrcoef(rad[m], z_mm[m])[0, 1])
            ax.set_title('Height vs off-axis distance (r = {:.3f})'.format(r))
        ax.set_xlabel('distance from optical axis (px)')
        ax.set_ylabel('head height (mm)')
    else:
        ax.set_axis_off()

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(str(savepath), dpi=150)
        print('Saved height diagnostics to {}.'.format(savepath))

    return fig


def main():
    """ CLI: fit head height for one preprocessed session and plot it. """

    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split('\n')[3])
    parser.add_argument('h5_path', type=str,
                        help='Preprocessed session HDF5.')
    parser.add_argument('--bar_length_mm', type=float, default=None,
                        help='True marker separation on the headbar. Without '
                             'it, height is pinned to --walk_height_mm.')
    parser.add_argument('--camera_height_mm', type=float, default=CAMERA_HEIGHT_MM)
    parser.add_argument('--walk_height_mm', type=float, default=WALK_HEIGHT_MM)
    parser.add_argument('--smooth_n', type=int, default=1,
                        help='Median filter width on the height trace.')
    parser.add_argument('--no_offaxis', action='store_true',
                        help='Use the scalar R = f D c / L instead of the '
                             'exact projective solve.')
    parser.add_argument('--savefig', type=str, default=None)
    args = parser.parse_args()

    data = read_h5(args.h5_path)
    z_mm, info = calc_head_height(data,
                                  bar_length_mm=args.bar_length_mm,
                                  camera_height_mm=args.camera_height_mm,
                                  walk_height_mm=args.walk_height_mm,
                                  smooth_n=args.smooth_n,
                                  offaxis_correct=not args.no_offaxis)

    if args.savefig is not None:
        plot_height_diagnostics(z_mm, info, data=data, savepath=args.savefig)


if __name__ == '__main__':
    main()
