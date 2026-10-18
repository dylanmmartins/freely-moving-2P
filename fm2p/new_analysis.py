# -*- coding: utf-8 -*-
"""
fm2p/new_analysis.py

Behavior first: pillar on the retina, retinal optic flow, gaze-shift motifs,
object-directed gaze and behavioral states for one freely moving session.

Everything is computed on a common 25 ms (40 Hz) time base, the one Hou et al.
(2026, bioRxiv 10.64898/2026.06.13.732086) classify gaze shifts on.  The pose is
taken from the retinal reconstruction `.npz` and pushed through the renderer's
own rotation chain (get_retinal_image.scene_pose_matrices), so every pillar
angle and flow vector here is the one the reconstruction actually drew.  That
is checked, not assumed: `validate_pose` re-renders a sample of frames and
compares them to the stored images.

Coordinate frames
-----------------
world    mm, x right, y up (top-down image y flipped), z up off the floor.
head     x forward, y left, z up.  Azimuth +left, elevation +up.
optical  right eye, x right (temporal), y down, z along the optic axis.
cyclo    head azimuth minus the horizontal eye rotation from rest.  Mouse
         horizontal eye movements are conjugate (Meyer et al. 2020), so the
         binocular overlap moves with the eye; this is the frame the binocular
         zone is fixed in.  Only the right eye is tracked, so conjugacy is an
         assumption, not a measurement.

Sign conventions (all checked on DMM061 pos18 before writing this)
------------------------------------------------------------------
heading        yaw_eye carries one-frame spikes where get_retinal_image
               interpolated the wrapped IMU yaw across 0/360; removed on load.
head velocity  -gyro_z, leftward positive.  Slope 0.91 against the derivative
               of the renderer heading.
eye velocity   +dtheta/dt, leftward positive.  Anti-correlated with head
               velocity at zero lag (r -0.42, gain 0.38) as VOR requires.
nasal          leftward, for the tracked right eye (as in Hou et al.).
pitch          renderer pitch_eye is nose-DOWN positive; nose-up = -pitch_eye.
               Ear separation in the top-down view grows with nose-up
               (head closer to the camera), which confirms it.

DMM, October 2026
"""

if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

import argparse
import json
import os
import time
import warnings

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
from scipy.stats import binomtest

from .utils.files import write_h5
from .get_retinal_image import (_scene_ray_dirs, simulate_retinal_scene, _interp_short_gaps,
                                vertical_rest_offset)

FS = 40.0
DT = 1.0 / FS

EYE_OFFSET_MM = np.array([3.5, -5.0, 3.5])
R_ALIGN = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
RECON_FOV_DEG = 120.0

BINO_HALF_DEG = 20.0
FIELD_EDGE_DEG = 150.0
ZONES = ('binocular', 'monoc_R', 'monoc_L', 'blind')

SG_WIN, SG_ORD = 5, 3
MOTIF_HALF = 4
MOTIF_NAMES = ('HbE', 'HwE', 'HD', 'ED')

SHIFTED_HEAD = 60.0
STILL_GAZE = 120.0
SHIFTED_GAZE = 240.0

LAND_PAD = 2
N_NULL = 2000

STATE_NAMES = ('approach', 'investigate', 'rear_pillar', 'rear_other',
               'wall_follow', 'turn', 'locomote', 'rest', 'other')
STATE_COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4',
                '#008300', '#6250d6', '#e34948', '#b9b8b2')
MOTIF_COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#6250d6')
CLASS_COLORS = {'comp': '#b158c3', 'nasal': '#2a78d6', 'temporal': '#eb6834',
                'none': '#c9c8c2'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def _interp_gappy(t_src, x, t_dst, max_gap):

    x = np.asarray(x, float)
    ok = np.isfinite(x) & np.isfinite(t_src)
    out = np.full(len(t_dst), np.nan)
    if ok.sum() < 2:
        return out
    ts, xs = t_src[ok], x[ok]
    out = np.interp(t_dst, ts, xs, left=np.nan, right=np.nan)
    # Distance to the nearest finite source sample on either side.
    j = np.clip(np.searchsorted(ts, t_dst), 1, len(ts) - 1)
    gap = ts[j] - ts[j - 1]
    out[gap > max_gap] = np.nan
    return out


def _runs(mask):

    m = np.concatenate([[0], np.asarray(mask, int), [0]])
    d = np.diff(m)
    return np.stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1)], axis=1)


def _min_bout(mask, n):

    out = np.zeros(len(mask), bool)
    for a, b in _runs(mask):
        if b - a >= n:
            out[a:b] = True
    return out


def _wrap(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def rescaled_eye_angles(h5_path, scale_px, cam_center_x=None):

    with h5py.File(h5_path, 'r') as f:
        X0 = np.asarray(f['X0'][:], float)
        Y0 = np.asarray(f['Y0'][:], float)
        cx = float(f['cam_center_x'][()]) if cam_center_x is None else float(cam_center_x)
        cy = float(f['cam_center_y'][()])
        i0, i1 = int(f['eyeT_startInd'][()]), int(f['eyeT_endInd'][()])
        lag = int(f['eye_imu_lag_samples'][()]) if 'eye_imu_lag_samples' in f else 0

    with np.errstate(invalid='ignore'):
        th = np.arcsin((X0 - cx) / scale_px)
        ph = np.arcsin((Y0 - cy) / np.cos(th) / scale_px)

    def shift(x):
        out = np.full_like(x, np.nan)
        if lag < 0:
            out[:lag] = x[-lag:]
        elif lag > 0:
            out[lag:] = x[:-lag]
        else:
            out[:] = x
        return out

    return shift(np.degrees(th))[i0:i1], shift(-np.degrees(ph))[i0:i1]


def _eye_scale_from_json(h5_path):

    path = h5_path.replace('_preproc.h5', '_eye_scale.json')
    if not os.path.isfile(path):
        return None, None, None
    with open(path, 'r') as f:
        meas = json.load(f)
    return float(meas['recommended_scale_px']), meas.get('cam_center_x'), path


def load_session(h5_path, npz_path, eye_scale='auto'):

    npz = np.load(npz_path, allow_pickle=False)
    for flag in ('fixed_pupil', 'fixed_eye', 'worldcam'):
        if flag in npz and bool(npz[flag]):
            print('  WARNING: npz rendered with {} -- pass the eye-free '
                  'retinocentric render.'.format(flag))

    with h5py.File(h5_path, 'r') as f:
        eyeT = np.asarray(f['eyeT_trim'][:], float)
        theta = np.asarray(f['theta_trim'][:], float)
        gyro_z = np.asarray(f['gyro_z_eye_interp'][:], float)
        twopT = np.asarray(f['twopT'][:], float)
        ltdk = (np.asarray(f['ltdk_state_vec'][:]).astype(float)
                if 'ltdk_state_vec' in f else np.ones(len(twopT)))
        ear_sep = None
        if 'lear_x' in f and 'rear_x' in f:
            ear_sep = np.hypot(f['lear_x'][:] - f['rear_x'][:],
                               f['lear_y'][:] - f['rear_y'][:])
        lag = (float(f['eye_imu_lag_samples'][()])
               if 'eye_imu_lag_samples' in f else np.nan)

    # Eye angles at a measured eye scale.  The reconstruction's pupil-from-head
    # angle is ang_offset - theta_trim exactly, so it is rebuilt the same way.
    pfh_src = np.asarray(npz['pfh_eye'], float)
    phi_src = np.asarray(npz['phi_trim'], float)
    eye_scale_used = np.nan
    if eye_scale != 'stored':
        if eye_scale == 'auto':
            sc, cx, src = _eye_scale_from_json(h5_path)
        else:
            sc, cx, src = float(eye_scale), None, 'argument'
        if sc is not None:
            theta, phi_new = rescaled_eye_angles(h5_path, sc, cx)
            m = min(len(theta), len(pfh_src))
            # The reconstruction bridged short tracking gaps; do the same so the
            # valid-eye mask does not shrink.  theta stays raw for velocity.
            gap = float(npz['eye_max_gap_frames']) if 'eye_max_gap_frames' in npz else np.nan
            th_f, ph_f = theta[:m], phi_new[:m]
            # A render with phi_rest_ref subtracted the resting vertical angle;
            # take it again at this scale, by the renderer's own rule.
            if 'phi_rest_ref' in npz and bool(npz['phi_rest_ref']):
                ph_f = ph_f - vertical_rest_offset(phi_new, gyro_z)
            if np.isfinite(gap):
                th_f = _interp_short_gaps(th_f, max_gap=int(gap))
                ph_f = _interp_short_gaps(ph_f, max_gap=int(gap))
            pfh_src = np.where(np.isfinite(pfh_src[:m]), float(npz['ang_offset']) - th_f, np.nan)
            phi_src = np.where(np.isfinite(phi_src[:m]), ph_f, np.nan)
            eye_scale_used = sc
            print('  Eye angles recomputed at scale {:.1f} px (centre x {}) from {}.'.format(
                sc, 'stored' if cx is None else '{:.1f}'.format(float(cx)), os.path.basename(src)))

    n = min(len(eyeT), len(npz['yaw_eye']), len(theta), len(gyro_z), len(pfh_src))
    eyeT = eyeT[:n]
    if not np.isfinite(eyeT).all():
        print('  WARNING: {} NaN eye timestamps.'.format(int((~np.isfinite(eyeT)).sum())))
    # Long twopT/eyeT mismatch is the signature of contaminated top-down tracking.
    if abs((twopT[-1] - twopT[0]) - (np.nanmax(eyeT) - np.nanmin(eyeT))) > 1.0:
        print('  WARNING: twopT and eyeT_trim spans differ by > 1 s.')

    t = np.arange(np.nanmin(eyeT), np.nanmax(eyeT), DT)
    g = 2.5 / 60.0
    rendered = np.asarray(npz['rendered'][:n], bool)

    def rs(x, gap=g):
        return _interp_gappy(eyeT, np.asarray(x, float)[:n], t, gap)

    # Renderer heading: Rz(-yaw_eye), CCW-positive.  Unwrap before resampling.
    # get_retinal_image interpolates the WRAPPED IMU yaw onto eye frames, so an
    # eye frame between IMU samples at ~359 and ~1 deg comes out anywhere in
    # between -- spikes of up to 180 deg, in runs when the heading hovers near
    # north (96% of spikes on DMM061 pos18 sit beside the wrap).  Find them on
    # the circle (median of cos and sin over 7 frames survives a 3-frame run),
    # drop them, and unwrap only what is left.
    yaw_w = np.radians(-np.asarray(npz['yaw_eye'][:n], float))
    ref = np.arctan2(median_filter(np.sin(yaw_w), 7, mode='nearest'),
                     median_filter(np.cos(yaw_w), 7, mode='nearest'))
    # A monotonic turn sits exactly on its running median, so 8 deg is safe.
    spikes = np.abs(_wrap(np.degrees(yaw_w - ref))) > 8.0
    good = ~spikes & rendered & np.isfinite(yaw_w)
    yaw_r = np.full(n, np.nan)
    yaw_r[good] = np.degrees(np.unwrap(yaw_w[good]))
    if spikes.any():
        print('  Removed {} yaw spikes ({:.2f}% of native frames) -- wrap-interpolation '
              'artifact in the reconstruction pose.'.format(int(spikes.sum()), 100 * spikes.mean()))

    s = {
        't': t,
        'yaw_r': rs(yaw_r, 0.1),
        'pitch': rs(npz['pitch_eye']),
        'roll': rs(npz['roll_eye']),
        'pfh': rs(pfh_src),
        'phi': rs(phi_src),
        'hx': rs(npz['head_x_eye']),
        'hy': rs(npz['head_y_eye']),
        'theta': rs(theta),
        'gyro_z': rs(gyro_z),
        'light': np.interp(t, twopT[:len(ltdk)], ltdk[:len(twopT)]) > 0.5,
        'ear_sep': (np.interp(t, twopT[:len(ear_sep)], ear_sep[:len(twopT)])
                    if ear_sep is not None else np.full(len(t), np.nan)),
        # The renderer's resting pupil moves by the mean change in each angle.
        'mean_pfh': float(npz['mean_pfh']) + float(np.nanmean(pfh_src[:n] - np.asarray(npz['pfh_eye'][:n], float))),
        'mean_phi': float(npz['mean_phi']) + float(np.nanmean(phi_src[:n] - np.asarray(npz['phi_trim'][:n], float))),
        'eye_scale_px': eye_scale_used,
        'alpha': float(npz['anatomical_eye_angle_deg']),
        'beta': float(npz['socket_elev_deg']),
        'head_z': float(npz['head_z_mm']) if 'head_z_mm' in npz else 25.0,
        'pillar_xy': np.array([float(npz['pillar_x_mm']), float(npz['pillar_y_mm'])]),
        'pillar_r': float(npz['pillar_d_mm']) / 2.0,
        'pillar_h': float(npz['pillar_h_mm']),
        'wall_h': float(npz['wall_h_mm']),
        'arena': np.asarray(npz['arena_corners_mm'], float),
        'eye_imu_lag': lag,
        'n_native': n,
        'native_fps': 1.0 / float(np.nanmedian(np.diff(eyeT))),
    }
    # Pose valid where the renderer could draw it, eye valid where tracked.
    s['pose_ok'] = np.isfinite(s['yaw_r']) & np.isfinite(s['hx']) & np.isfinite(s['pitch'])
    s['eye_ok'] = s['pose_ok'] & np.isfinite(s['pfh']) & np.isfinite(s['phi'])
    print('  {} frames at {:.0f} Hz ({:.1f} min); pose {:.1f}%, eye {:.1f}%, '
          'light {:.1f}%.'.format(len(t), FS, len(t) * DT / 60, 100 * s['pose_ok'].mean(),
                                  100 * s['eye_ok'].mean(), 100 * s['light'].mean()))
    return s


def pose_matrices(s, freeze_pupil=False):
    """ Eye origin and optical-to-world rotation per frame, vectorized.

    Same chain as get_retinal_image.scene_pose_matrices: rot = R_head @ R_eye @
    R_align^T, origin = R_head @ t_eye + t_head, with the renderer's sign flips
    on both pupil angles (tilt_h = -pfh, tilt_v = -phi).

    Returns
    -------
    origin : (N, 3)
    rot : (N, 3, 3)
        Optical-to-world.  NaN where the pose (or, unfrozen, the eye) is missing.
    R_head : (N, 3, 3)
    """

    n = len(s['t'])
    pfh = np.full(n, s['mean_pfh']) if freeze_pupil else s['pfh']
    phi = np.full(n, s['mean_phi']) if freeze_pupil else s['phi']
    ok = s['pose_ok'] & np.isfinite(pfh) & np.isfinite(phi)

    Rh = np.full((n, 3, 3), np.nan)
    rot = np.full((n, 3, 3), np.nan)
    origin = np.full((n, 3), np.nan)
    Rh[s['pose_ok']] = R.from_euler('ZYX', np.stack(
        [s['yaw_r'], s['pitch'], s['roll']], 1)[s['pose_ok']], degrees=True).as_matrix()
    Rs = (R.from_euler('z', s['alpha'], degrees=True).as_matrix()
          @ R.from_euler('y', s['beta'], degrees=True).as_matrix())
    Rsac = R.from_euler('zyx', np.stack([-pfh[ok], -phi[ok], np.zeros(ok.sum())], 1),
                        degrees=True).as_matrix()
    rot[ok] = Rh[ok] @ (Rs @ Rsac) @ R_ALIGN.T
    th = np.stack([s['hx'], s['hy'], np.full(n, s['head_z'])], 1)
    origin[ok] = np.einsum('nij,j->ni', Rh[ok], EYE_OFFSET_MM) + th[ok]
    return origin, rot, Rh


def validate_pose(npz_path, n_frames=150, seed=0):
    """ Re-render sampled frames and compare to the stored reconstruction.

    Uses the native-rate pose with the same sign flips pose_matrices applies.
    A mean absolute luminance error of 0 means the geometry here is the
    geometry the images were drawn with.
    """

    z = np.load(npz_path, allow_pickle=False)
    ok = np.asarray(z['rendered'], bool) & np.isfinite(z['pfh_eye']) & np.isfinite(z['phi_trim'])
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(np.flatnonzero(ok), min(n_frames, int(ok.sum())), replace=False))
    err = np.zeros(len(idx))
    ss = int(z['supersample']) if 'supersample' in z else 1
    # Indexing an NpzFile key re-reads the whole array, so pull it once.
    stored = z['retinal_images']
    for k, i in enumerate(idx):
        img = simulate_retinal_scene(
            z['arena_corners_mm'], float(z['pillar_x_mm']), float(z['pillar_y_mm']),
            float(z['head_x_eye'][i]), float(z['head_y_eye'][i]), -float(z['yaw_eye'][i]),
            float(z['pitch_eye'][i]), float(z['roll_eye'][i]),
            -float(z['pfh_eye'][i]), -float(z['phi_trim'][i]),
            pillar_h=float(z['pillar_h_mm']), pillar_d=float(z['pillar_d_mm']),
            wall_h=float(z['wall_h_mm']),
            anatomical_eye_angle_deg=float(z['anatomical_eye_angle_deg']),
            socket_elev_deg=float(z['socket_elev_deg']),
            head_z=float(z['head_z_mm']), supersample=ss)
        err[k] = np.mean(np.abs(img.astype(float) - stored[i]))
    out = {'n': len(idx), 'mean_abs_err': float(err.mean()),
           'frac_exact': float((err < 0.5).mean())}
    print('  Re-render check: {} frames, mean |lum err| {:.3f}, '
          '{:.1f}% exact.'.format(out['n'], out['mean_abs_err'], 100 * out['frac_exact']))
    return out


def check_pose_match(pose_npz, images_npz, tol_deg=1e-3):
    """ Largest per-frame difference in the pose two reconstructions were made from.

    validate_pose re-renders the images npz from ITS OWN stored pose, so it
    passes even when the pose npz used for geometry comes from another render
    (e.g. a pose-only npz made before the vertical eye rest reference, beside
    images made after it -- DMM061 pos18 on 2026-10-14).  This compares the
    two directly.

    Returns
    -------
    diffs : dict
        key -> max |difference| (deg or mm) over frames finite in both;
        'match' is True when every key is within tol_deg.
    """

    keys = ('yaw_eye', 'pitch_eye', 'roll_eye', 'pfh_eye', 'phi_trim', 'head_x_eye', 'head_y_eye')
    out = {}
    with np.load(pose_npz, allow_pickle=False) as a, np.load(images_npz, allow_pickle=False) as b:
        for k in keys:
            if k not in a.files or k not in b.files:
                out[k] = np.nan
                continue
            x, y = np.asarray(a[k], float), np.asarray(b[k], float)
            n = min(len(x), len(y))
            d = np.abs(x[:n] - y[:n])
            if k == 'yaw_eye':
                d = np.abs(_wrap(x[:n] - y[:n]))
            f = np.isfinite(d)
            out[k] = float(d[f].max()) if f.any() else np.nan
        ra = bool(a['phi_rest_ref']) if 'phi_rest_ref' in a.files else False
        rb = bool(b['phi_rest_ref']) if 'phi_rest_ref' in b.files else False
    out['phi_rest_ref'] = (ra, rb)
    out['match'] = bool(all(not (v > tol_deg) for k, v in out.items() if k in keys) and ra == rb)
    if not out['match']:
        print('  WARNING: {} and {} were made from DIFFERENT poses (max diffs: {}; phi rest-referenced '
              '{} / {}).  Geometry comes from the first, image checks from the second -- re-run the '
              'pose-only render.'.format(os.path.basename(pose_npz), os.path.basename(images_npz),
                                         ', '.join('{} {:.3g}'.format(k, out[k]) for k in keys
                                                   if out[k] > tol_deg), ra, rb))
    return out


def zone_of(az_cyc, half=BINO_HALF_DEG, edge=FIELD_EDGE_DEG):
    """ Visual-field zone code (index into ZONES) from cyclopean azimuth.

    -1 where az is NaN.  Azimuth is +left, so the tracked right eye's monocular
    field is negative azimuth.
    """

    a = np.asarray(az_cyc, float)
    z = np.full(a.shape, -1, int)
    f = np.isfinite(a)
    z[f & (np.abs(a) <= half)] = 0
    z[f & (a < -half) & (a >= -edge)] = 1
    z[f & (a > half) & (a <= edge)] = 2
    z[f & (np.abs(a) > edge)] = 3
    return z


def pillar_geometry(s):
    """ Pillar azimuth, elevation, angular size and visual-field zone per frame.

    Optical-frame angles are off the right eye's optic axis (az +temporal, el
    +up), i.e. retinal coordinates.  Head-frame angles are off the nose (+left,
    +up).  The cyclopean azimuth subtracts the horizontal eye deviation from
    rest and is what the zone is read from.

    Returns
    -------
    p : dict of (N,) arrays.
    """

    n = len(s['t'])
    origin, rot, Rh = pose_matrices(s)
    _, rot0, _ = pose_matrices(s, freeze_pupil=True)
    px, py = s['pillar_xy']

    # Eye-level point on the pillar axis, so the bearing is that of the axis.
    tgt = np.stack([np.full(n, px), np.full(n, py), origin[:, 2]], 1)
    v_w = tgt - origin
    v_o = np.einsum('nji,nj->ni', rot, v_w)
    az_eye = np.degrees(np.arctan2(v_o[:, 0], v_o[:, 2]))
    el_eye = np.degrees(np.arctan2(-v_o[:, 1], np.hypot(v_o[:, 0], v_o[:, 2])))

    # Head frame, from the eye origin.
    v_h = np.einsum('nji,nj->ni', Rh, v_w)
    az_head = np.degrees(np.arctan2(v_h[:, 1], v_h[:, 0]))
    el_head = np.degrees(np.arctan2(v_h[:, 2], np.hypot(v_h[:, 0], v_h[:, 1])))

    # Horizontal eye deviation: optic axis azimuth in the head frame, free
    # minus at rest.  Optic axis in optical frame is +z.
    def axis_az(rr):
        a_w = rr[:, :, 2]
        a_h = np.einsum('nji,nj->ni', Rh, a_w)
        return np.degrees(np.arctan2(a_h[:, 1], a_h[:, 0])), a_w

    az_ax, ax_w = axis_az(rot)
    az_ax0, _ = axis_az(rot0)
    # Two measures of horizontal eye position.  The optic axis sits ~45 deg
    # above the horizon (socket + mean pupil elevation), so its AZIMUTH moves
    # 1/cos(el) ~ 1.5x faster than the eye actually rotates and also picks up
    # the noisy vertical trace.  Points on the horizon -- where the binocular
    # zone and the pillar are -- move by about the rotation itself, so the
    # cyclopean frame uses the horizontal rotation (leftward = -(pfh - rest)).
    eye_dev_axis = _wrap(az_ax - az_ax0)
    eye_dev = _wrap(-(s['pfh'] - s['mean_pfh']))
    az_cyc = _wrap(az_head - eye_dev)

    # World bearing of the pillar from the eye and the cyclopean gaze direction
    # in the world; az_cyc = bearing - gaze_cyc by construction.
    bearing = np.degrees(np.arctan2(v_w[:, 1], v_w[:, 0]))
    gaze_cyc = _wrap(bearing - az_cyc)

    d_h = np.hypot(v_w[:, 0], v_w[:, 1])
    r = s['pillar_r']
    width = 2.0 * np.degrees(np.arcsin(np.clip(r / np.maximum(d_h, r), 0, 1)))
    ez = origin[:, 2]
    near = np.maximum(d_h - r, 1.0)
    height = (np.degrees(np.arctan2(s['pillar_h'] - ez, near))
              - np.degrees(np.arctan2(-ez, near)))

    f = np.radians(RECON_FOV_DEG / 2)
    in_recon = (v_o[:, 2] > 0) & (np.abs(v_o[:, 0] / v_o[:, 2]) < np.tan(f)) \
        & (np.abs(v_o[:, 1] / v_o[:, 2]) < np.tan(f))
    zone = zone_of(az_cyc)

    # Pillar angular speed on the retina, eye free vs frozen (deg/s).
    def ang_speed(rr):
        vo = np.einsum('nji,nj->ni', rr, v_w)
        u = vo / np.linalg.norm(vo, axis=1, keepdims=True)
        sp = np.full(n, np.nan)
        dots = np.clip(np.einsum('ij,ij->i', u[:-2], u[2:]), -1, 1)
        sp[1:-1] = np.degrees(np.arccos(dots)) / (2 * DT)
        return sp

    p = {'az_eye': az_eye, 'el_eye': el_eye, 'az_head': az_head, 'el_head': el_head,
         'eye_dev': eye_dev, 'eye_dev_axis': eye_dev_axis, 'az_cyc': az_cyc, 'bearing': bearing, 'gaze_cyc': gaze_cyc,
         'dist_mm': d_h - r, 'width_deg': width, 'height_deg': height,
         'solid_deg2': width * height, 'in_recon': in_recon & np.isfinite(az_eye),
         'zone': zone, 'slip': ang_speed(rot), 'slip_frozen': ang_speed(rot0),
         'axis_w': ax_w}
    for k in ('az_eye', 'el_eye', 'dist_mm', 'width_deg', 'height_deg', 'solid_deg2'):
        p[k] = np.where(np.isfinite(az_cyc), p[k], np.nan)
    v = zone >= 0
    print('  Pillar: {:.1f}% of eye-valid frames in the 120 deg reconstruction; '
          'zones {}.'.format(100 * p['in_recon'][v].mean(),
                             ', '.join('{} {:.1f}%'.format(z, 100 * (zone[v] == i).mean())
                                       for i, z in enumerate(ZONES))))
    return p


def _raycast_depth(o, d, arena, pillar_xy, pillar_r, pillar_h, wall_h):
    """ Distance to the first surface along each ray, many frames at once.

    Same scene as simulate_retinal_scene: floor inside the arena outline, one
    vertical quad per edge, a capped cylinder.  Sky is +inf.

    Parameters
    ----------
    o : (F, 3)
        Ray origins.
    d : (F, P, 3)
        Unit directions, world frame.

    Returns
    -------
    t : (F, P)
    lab : (F, P) uint8
        0 sky, 1 wall, 2 floor, 3 pillar.
    """

    eps = 1e-9
    F, P, _ = d.shape
    best = np.full((F, P), np.inf)
    lab = np.zeros((F, P), np.uint8)
    ox, oy, oz = (o[:, i][:, None] for i in range(3))

    with np.errstate(divide='ignore', invalid='ignore'):
        tf = -oz / d[..., 2]
    hit = (d[..., 2] < -eps) & (tf > 0)
    fx, fy = ox + tf * d[..., 0], oy + tf * d[..., 1]
    c = arena
    area2 = np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1])
    sg = 1.0 if area2 > 0 else -1.0
    for k in range(len(c)):
        a, b = c[k], c[(k + 1) % len(c)]
        hit &= sg * ((b[0] - a[0]) * (fy - a[1]) - (b[1] - a[1]) * (fx - a[0])) >= -eps
    best[hit] = tf[hit]
    lab[hit] = 2

    for k in range(len(c)):
        a, b = c[k], c[(k + 1) % len(c)]
        ab = b - a
        L2 = float(ab @ ab)
        nx, ny = -ab[1], ab[0]
        den = nx * d[..., 0] + ny * d[..., 1]
        with np.errstate(divide='ignore', invalid='ignore'):
            tw = (nx * (a[0] - ox) + ny * (a[1] - oy)) / den
        wx, wy, wz = ox + tw * d[..., 0], oy + tw * d[..., 1], oz + tw * d[..., 2]
        sp = ((wx - a[0]) * ab[0] + (wy - a[1]) * ab[1]) / L2
        ok = ((np.abs(den) > eps) & (tw > 0) & (tw < best) & (sp >= 0) & (sp <= 1)
              & (wz >= 0) & (wz <= wall_h))
        best[ok] = tw[ok]
        lab[ok] = 1

    cx, cy = ox - pillar_xy[0], oy - pillar_xy[1]
    qa = d[..., 0] ** 2 + d[..., 1] ** 2
    qb = 2 * (cx * d[..., 0] + cy * d[..., 1])
    qc = cx ** 2 + cy ** 2 - pillar_r ** 2
    disc = qb ** 2 - 4 * qa * qc
    with np.errstate(invalid='ignore', divide='ignore'):
        sq = np.sqrt(np.where(disc > 0, disc, np.nan))
        tn = (-qb - sq) / (2 * qa)
        tfar = (-qb + sq) / (2 * qa)
    tc = np.where(tn > eps, tn, tfar)
    zc = oz + tc * d[..., 2]
    ok = (disc > 0) & (qa > eps) & (tc > eps) & (tc < best) & (zc >= 0) & (zc <= pillar_h)
    best[ok] = tc[ok]
    lab[ok] = 3
    return best, lab


def optic_flow(s, grid_res=15, chunk=4000):
    """ Analytic retinal optic flow on a grid, split into rotation and translation.

    For a static point seen along unit direction d (optical frame) at distance
    rho, with the eye turning at omega and translating at v (both in the optical
    frame):  d_dot = -omega x d - (v - (v.d) d) / rho.  Depth comes from casting
    each grid ray into the arena; sky rays have no translational flow.  The
    eye-frozen counterfactual uses the resting pupil and the same head motion.

    Returns
    -------
    fl : dict
        Per-frame summaries (deg/s): total/rot/trans mean speed over the field,
        frozen-eye total/rot, flow at the optic axis, and the per-frame field
        (N, P, 2) in tangent (az, el) components for averaging later.
    """

    n = len(s['t'])
    dirs = _scene_ray_dirs(grid_res, RECON_FOV_DEG)
    P = len(dirs)
    az = np.arctan2(dirs[:, 0], dirs[:, 2])
    el = np.arctan2(-dirs[:, 1], np.hypot(dirs[:, 0], dirs[:, 2]))
    e_az = np.stack([np.cos(az), np.zeros(P), -np.sin(az)], 1)
    e_el = np.stack([-np.sin(az) * np.sin(el), -np.cos(el), -np.cos(az) * np.sin(el)], 1)
    centre = np.argmin(az ** 2 + el ** 2)

    out = {k: np.full(n, np.nan) for k in
           ('total', 'rot', 'trans', 'total_frozen', 'rot_frozen', 'trans_frozen',
            'axis', 'omega', 'speed_mm_s', 'frac_sky')}
    field = np.full((n, P, 2), np.nan, np.float32)
    deg = np.degrees(1.0)
    for frozen in (False, True):
        origin, rot, _ = pose_matrices(s, freeze_pupil=frozen)
        # Central differences.  Rotation increment in the CURRENT optical frame.
        w = np.full((n, 3), np.nan)
        v = np.full((n, 3), np.nan)
        ok = np.isfinite(rot[:-2, 0, 0]) & np.isfinite(rot[2:, 0, 0]) & np.isfinite(rot[1:-1, 0, 0])
        ii = np.flatnonzero(ok) + 1
        dR = np.einsum('nji,njk->nik', rot[ii - 1], rot[ii + 1])
        w[ii] = R.from_matrix(dR).as_rotvec() / (2 * DT)
        vw = (origin[ii + 1] - origin[ii - 1]) / (2 * DT)
        v[ii] = np.einsum('nji,nj->ni', rot[ii], vw)
        sfx = '_frozen' if frozen else ''
        for a in range(0, len(ii), chunk):
            jj = ii[a:a + chunk]
            dw = np.einsum('nij,pj->npi', rot[jj], dirs)
            depth, lab = _raycast_depth(origin[jj], dw, s['arena'], s['pillar_xy'],
                                        s['pillar_r'], s['pillar_h'], s['wall_h'])
            f_rot = -np.cross(w[jj][:, None, :], dirs[None, :, :])
            vd = np.einsum('nj,pj->np', v[jj], dirs)
            f_tr = -(v[jj][:, None, :] - vd[..., None] * dirs[None]) / depth[..., None]
            f_tot = f_rot + f_tr
            out['total' + sfx][jj] = deg * np.linalg.norm(f_tot, axis=2).mean(1)
            out['rot' + sfx][jj] = deg * np.linalg.norm(f_rot, axis=2).mean(1)
            out['trans' + sfx][jj] = deg * np.linalg.norm(f_tr, axis=2).mean(1)
            if not frozen:
                field[jj, :, 0] = deg * np.einsum('npi,pi->np', f_tot, e_az)
                field[jj, :, 1] = deg * np.einsum('npi,pi->np', f_tot, e_el)
                out['axis'][jj] = deg * np.linalg.norm(f_tot[:, centre], axis=1)
                out['omega'][jj] = deg * np.linalg.norm(w[jj], axis=1)
                out['speed_mm_s'][jj] = np.linalg.norm(v[jj], axis=1)
                out['frac_sky'][jj] = (lab == 0).mean(1)
    out['field'] = field
    out['grid_az'] = np.degrees(az)
    out['grid_el'] = np.degrees(el)
    v = np.isfinite(out['total'])
    print('  Optic flow on {} frames: median total {:.0f}, rot {:.0f}, trans {:.0f} deg/s; '
          'eye-frozen total {:.0f}.'.format(int(v.sum()), np.nanmedian(out['total']),
                                            np.nanmedian(out['rot']), np.nanmedian(out['trans']),
                                            np.nanmedian(out['total_frozen'])))
    return out


def flow_formula_check(s, n_frames=2000, seed=0):
    """ Analytic flow against a finite difference of reprojected world points.

    For sampled frames, each grid ray's world point (from the ray cast) is
    projected into the optical frames at t-1 and t+1; the central difference of
    those unit vectors is the true image motion of that point.  This tests the
    flow formula, the frame conventions and the rotation-increment frame
    together, independently of how the formula was derived.
    """

    origin, rot, _ = pose_matrices(s)
    ok = (np.isfinite(rot[:-2, 0, 0]) & np.isfinite(rot[1:-1, 0, 0]) & np.isfinite(rot[2:, 0, 0]))
    ii = np.flatnonzero(ok) + 1
    rng = np.random.default_rng(seed)
    ii = np.sort(rng.choice(ii, min(n_frames, len(ii)), replace=False))
    dirs = _scene_ray_dirs(15, RECON_FOV_DEG)
    dw = np.einsum('nij,pj->npi', rot[ii], dirs)
    depth, _ = _raycast_depth(origin[ii], dw, s['arena'], s['pillar_xy'], s['pillar_r'],
                              s['pillar_h'], s['wall_h'])
    fin = np.isfinite(depth)
    X = origin[ii][:, None, :] + np.where(fin, depth, 0)[..., None] * dw

    def proj(j):
        rel = np.where(fin[..., None], X - origin[j][:, None, :], dw)
        q = np.einsum('nji,npj->npi', rot[j], rel)
        return q / np.linalg.norm(q, axis=2, keepdims=True)

    num = (proj(ii + 1) - proj(ii - 1)) / (2 * DT)
    w = R.from_matrix(np.einsum('nji,njk->nik', rot[ii - 1], rot[ii + 1])).as_rotvec() / (2 * DT)
    v = np.einsum('nji,nj->ni', rot[ii], (origin[ii + 1] - origin[ii - 1]) / (2 * DT))
    vd = np.einsum('nj,pj->np', v, dirs)
    ana = (-np.cross(w[:, None, :], dirs[None]) -
           (v[:, None, :] - vd[..., None] * dirs[None]) / np.where(fin, depth, np.inf)[..., None])
    a, b = np.degrees(ana).reshape(-1, 3), np.degrees(num).reshape(-1, 3)
    err = np.linalg.norm(a - b, axis=1)
    mag = np.linalg.norm(b, axis=1)
    out = {'n_frames': len(ii), 'median_abs_err': float(np.median(err)),
           'median_speed': float(np.median(mag)),
           'r': float(np.corrcoef(a.ravel(), b.ravel())[0, 1])}
    print('  Flow formula check: r = {:.4f}, median |error| {:.2f} deg/s at median speed '
          '{:.1f} deg/s.'.format(out['r'], out['median_abs_err'], out['median_speed']))
    return out


def warp_check(npz_path, n_frames=300, gap=3, min_change=2.0, seed=0):
    """ How much frame-to-frame image change does the motion model explain?

    Native-rate frames.  Every pixel of frame i is cast into the arena, its world
    point re-projected into frame i+gap with that frame's pose, and frame i+gap
    sampled there.  Warp error = mean |I_{i+gap}(warped) - I_i|; identity error =
    mean |I_{i+gap} - I_i| over the same pixels.  The counterfactual target pose
    keeps the head motion but holds the eye where it was at frame i, i.e. asks
    how well the image change is predicted if the eye had not moved in between.
    A ratio near 0 means the pose explains the image motion.
    """

    from scipy.ndimage import map_coordinates

    z = np.load(npz_path, allow_pickle=False)
    imgs = z['retinal_images']
    n = len(z['yaw_eye'])
    nat = {'t': np.arange(n) / 60.0,
           'yaw_r': np.degrees(np.unwrap(np.radians(-np.asarray(z['yaw_eye'], float)))),
           'pitch': np.asarray(z['pitch_eye'], float), 'roll': np.asarray(z['roll_eye'], float),
           'pfh': np.asarray(z['pfh_eye'], float), 'phi': np.asarray(z['phi_trim'], float),
           'hx': np.asarray(z['head_x_eye'], float), 'hy': np.asarray(z['head_y_eye'], float),
           'mean_pfh': float(z['mean_pfh']), 'mean_phi': float(z['mean_phi']),
           'alpha': float(z['anatomical_eye_angle_deg']), 'beta': float(z['socket_elev_deg']),
           'head_z': float(z['head_z_mm'])}
    nat['pose_ok'] = np.asarray(z['rendered'], bool) & np.isfinite(nat['pfh']) & np.isfinite(nat['phi'])
    ok = nat['pose_ok'][:-gap] & nat['pose_ok'][gap:]
    cand = np.flatnonzero(ok)
    rng = np.random.default_rng(seed)
    rng.shuffle(cand)
    res = 120
    dirs = _scene_ray_dirs(res, RECON_FOV_DEG)
    fpx = (res / 2) / np.tan(np.radians(RECON_FOV_DEG / 2))
    o, rt, Rh = pose_matrices(nat)
    out = {'warp': [], 'warp_frozen': [], 'ident': [], 'ident_frozen': [], 'eye_move_deg': []}
    for i in cand:
        A = imgs[i].astype(float)
        B = imgs[i + gap].astype(float)
        if np.mean(np.abs(A - B)) < min_change:
            continue
        errs = {}
        dw = dirs @ rt[i].T
        dep, _ = _raycast_depth(o[i][None], dw[None], np.asarray(z['arena_corners_mm'], float),
                                np.array([float(z['pillar_x_mm']), float(z['pillar_y_mm'])]),
                                float(z['pillar_d_mm']) / 2, float(z['pillar_h_mm']),
                                float(z['wall_h_mm']))
        dep = dep[0]
        fin = np.isfinite(dep)
        rel = np.where(fin[:, None], o[i] + np.where(fin, dep, 0)[:, None] * dw - o[i + gap], dw)
        # rot = R_head @ (R_eye @ R_align^T); swap in frame i's eye term.
        tgt = {False: rt[i + gap], True: Rh[i + gap] @ Rh[i].T @ rt[i]}
        for frozen in (False, True):
            q = rel @ tgt[frozen]
            vis = q[:, 2] > 1e-6
            u = fpx * q[:, 0] / np.where(vis, q[:, 2], 1) + res / 2 - 0.5
            vv = fpx * q[:, 1] / np.where(vis, q[:, 2], 1) + res / 2 - 0.5
            inb = vis & (u >= 0) & (u <= res - 1) & (vv >= 0) & (vv <= res - 1)
            Bw = map_coordinates(B, [vv[inb], u[inb]], order=1)
            errs[frozen] = (np.mean(np.abs(Bw - A.ravel()[inb])), np.mean(np.abs(B.ravel()[inb] - A.ravel()[inb])))
        out['warp'].append(errs[False][0])
        out['ident'].append(errs[False][1])
        out['warp_frozen'].append(errs[True][0])
        out['ident_frozen'].append(errs[True][1])
        dR = tgt[True].T @ tgt[False]
        out['eye_move_deg'].append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
        if len(out['warp']) >= n_frames:
            break
    for k in out:
        out[k] = np.asarray(out[k])
    out['ratio'] = float(np.median(out['warp'] / out['ident']))
    out['ratio_frozen'] = float(np.median(out['warp_frozen'] / out['ident_frozen']))
    big = out['eye_move_deg'] > 3.0
    out['n_eye_moved'] = int(big.sum())
    out['ratio_eye_moved'] = float(np.median(out['warp'][big] / out['ident'][big])) if big.any() else np.nan
    out['ratio_frozen_eye_moved'] = (float(np.median(out['warp_frozen'][big] / out['ident_frozen'][big]))
                                     if big.any() else np.nan)
    out['gap_ms'] = 1000.0 * gap * float(np.nanmedian(np.diff(z['eyeT'])))
    print('  Warp check ({} frame pairs, {:.0f} ms apart): warp/identity error {:.3f} with the '
          'eye, {:.3f} eye held; on {} pairs where the eye moved > 3 deg: {:.3f} vs '
          '{:.3f}.'.format(len(out['warp']), out['gap_ms'], out['ratio'], out['ratio_frozen'],
                           out['n_eye_moved'], out['ratio_eye_moved'], out['ratio_frozen_eye_moved']))
    return out


def _sg_nan(x, deriv=0):
    """ Savitzky-Golay on a gappy trace: fill short gaps, filter, re-mask. """

    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < SG_WIN:
        return np.full(len(x), np.nan)
    xf = np.interp(np.arange(len(x)), np.flatnonzero(ok), x[ok])
    y = savgol_filter(xf, SG_WIN, SG_ORD, deriv=deriv, delta=DT)
    # A sample is only as good as the window it was fitted from.
    bad = np.convolve((~ok).astype(float), np.ones(SG_WIN), mode='same') > 0
    y[bad] = np.nan
    return y


def eye_head_velocity(s):
    """ Savitzky-Golay smoothed head, eye and gaze velocity and acceleration.

    Head: -gyro_z (IMU, leftward positive).  Eye: d(theta)/dt (leftward
    positive).  Gaze = head + eye, as in Hou et al.  Positive = nasal for the
    tracked right eye.

    Returns
    -------
    v : dict of (N,) arrays, deg/s and deg/s^2.
    """

    head = _sg_nan(-s['gyro_z'])
    eye = _sg_nan(s['theta'], deriv=1)
    eye[~s['eye_ok']] = np.nan
    v = {'head': head, 'eye': eye, 'gaze': head + eye,
         'head_acc': np.gradient(head, DT), 'eye_acc': np.gradient(eye, DT)}
    # Counter-rotation gain on slow phases only.  A plain regression over all
    # head-moving frames mixes in saccades (eye WITH head) and is attenuated by
    # noise in the head trace, which on DMM061 pos18 read 0.26 against 0.41
    # here.  Saccade frames (|eye| > 150 deg/s) are dropped with +-50 ms
    # margins, and the slope is the total-least-squares one.
    sac = np.convolve((np.abs(np.nan_to_num(eye)) > 150).astype(float), np.ones(5), mode='same') > 0
    base = np.isfinite(v['gaze']) & (np.abs(head) > 30) & (np.abs(head) < 300) & ~sac

    def tls(m):
        if m.sum() < 100:
            return np.nan
        w, V = np.linalg.eigh(np.cov(np.stack([head[m], eye[m]])))
        return float(V[1, -1] / V[0, -1])

    v['vor_slope'] = tls(base)
    v['vor_slope_light'] = tls(base & s['light'])
    v['vor_slope_dark'] = tls(base & ~s['light'])
    print('  Eye/head velocity: {:.1f}% frames usable; slow-phase eye-on-head gain {:.2f} '
          '(light {:.2f}, dark {:.2f}).'.format(100 * np.isfinite(v['gaze']).mean(), v['vor_slope'],
                                                v['vor_slope_light'], v['vor_slope_dark']))
    return v


def classify_paper(v, seed=0):
    """ Hou et al. 3-way frame classification: compensatory / nasal / temporal.

    (head, eye) velocity projected to 4-D with a Gaussian random projection,
    k-means k=3, clusters named by the sign of their mean gaze velocity (the
    one nearest zero is compensatory).  Gaze-shift frames whose previous frame
    is compensatory are then dropped, as in their Methods 4.3.

    Returns
    -------
    lab : (N,) int
        -1 invalid, 0 compensatory, 1 nasal, 2 temporal.
    info : dict
        Cluster centres and the frames dropped by the exclusion rule.
    """

    from sklearn.random_projection import GaussianRandomProjection
    from sklearn.cluster import KMeans

    ok = np.isfinite(v['head']) & np.isfinite(v['eye'])
    X = np.stack([v['head'][ok], v['eye'][ok]], 1)
    with warnings.catch_warnings():
        # Projecting 2 features up to 4 trips sklearn's dimensionality warning.
        warnings.simplefilter('ignore')
        Z = GaussianRandomProjection(n_components=4, random_state=seed).fit_transform(X)
    km = KMeans(3, n_init=10, random_state=seed).fit(Z)
    g = X.sum(1)
    mg = np.array([g[km.labels_ == k].mean() for k in range(3)])
    comp_k = int(np.argmin(np.abs(mg)))
    rest = [k for k in range(3) if k != comp_k]
    nas_k = rest[int(np.argmax(mg[rest]))]
    tmp_k = rest[int(np.argmin(mg[rest]))]
    remap = {comp_k: 0, nas_k: 1, tmp_k: 2}
    lab = np.full(len(ok), -1, int)
    lab[ok] = np.vectorize(remap.get)(km.labels_)

    prev = np.r_[-1, lab[:-1]]
    drop = (lab > 0) & (prev == 0)
    lab_x = lab.copy()
    lab_x[drop] = -2  # excluded gaze-shift onset frame, not compensatory either
    cent = np.array([[X[km.labels_ == k, 0].mean(), X[km.labels_ == k, 1].mean()]
                     for k in (comp_k, nas_k, tmp_k)])
    info = {'centres_head_eye': cent, 'n_dropped': int(drop.sum()), 'lab_raw': lab}
    print('  Paper classifier: comp {:.1f}%, nasal {:.1f}%, temporal {:.1f}% of valid '
          'frames; {} onset frames dropped by the prev-frame rule.'.format(
              *(100 * np.mean(lab[ok] == k) for k in range(3)), info['n_dropped']))
    return lab_x, info


def classify_gated(v):
    """ Instantaneous gaze-velocity classes (Parker et al. 2023), used downstream.

    Hou et al. say their k-means classes reproduce this split on their data.
    On this session the k-means boundary falls on HEAD speed instead (eye/head
    gain ~0.3, so the eye barely changes where a frame lands), labelling slow
    compensated head turns as gaze shifts.  So: gaze shift = |gaze| > 240 deg/s,
    direction by gaze sign, no head requirement so eye-led (ED-type) shifts
    survive; compensatory = |head| > 60 and |gaze| < 120.  Hou et al.'s
    previous-frame rule is then applied unchanged.

    Returns (N,) int: -1 invalid, -2 dropped onset frame, 0 compensatory,
    1 nasal, 2 temporal, 3 other.
    """

    h, gz = v['head'], v['gaze']
    ok = np.isfinite(gz)
    lab = np.where(ok, 3, -1)
    lab[ok & (np.abs(h) > SHIFTED_HEAD) & (np.abs(gz) < STILL_GAZE)] = 0
    lab[ok & (gz > SHIFTED_GAZE)] = 1
    lab[ok & (gz < -SHIFTED_GAZE)] = 2
    prev = np.r_[-1, lab[:-1]]
    lab[((lab == 1) | (lab == 2)) & (prev == 0)] = -2
    print('  Gated classifier: comp {:.1f}%, nasal {:.1f}%, temporal {:.1f}% of valid frames; '
          '{} onset frames dropped by the prev-frame rule.'.format(
              *(100 * np.mean(lab[ok] == k) for k in range(3)), int((lab == -2).sum())))
    return lab


def _episodes(lab, gaze):
    """ Direction-pure gaze-shift episodes as (start, stop, direction) rows.

    A dropped onset frame (-2) still belongs to its episode -- the rule removes
    it from clustering, not from the movement.
    """

    nas = (lab == 1) | ((lab == -2) & (gaze > 0))
    tmp = (lab == 2) | ((lab == -2) & (gaze < 0))
    rows = [(a, b, 1) for a, b in _runs(nas)] + [(a, b, -1) for a, b in _runs(tmp)]
    return sorted(rows)


def classify_threshold(v):
    """ The package's threshold classification (PETH.py), for comparison.

    Returns (N,) int: -1 invalid, 0 compensatory, 1 nasal, 2 temporal, 3 other.
    """

    h, gz = v['head'], v['gaze']
    ok = np.isfinite(gz)
    moving = ok & (np.abs(h) > SHIFTED_HEAD)
    comp = moving & (np.abs(gz) < STILL_GAZE)
    shift = moving & (np.abs(gz) > SHIFTED_GAZE) & (np.sign(gz) == np.sign(h))
    lab = np.where(ok, 3, -1)
    lab[comp] = 0
    lab[shift & (gz > 0)] = 1
    lab[shift & (gz < 0)] = 2
    return lab


def _onset_lag(h, e):
    """ Head-minus-eye onset lag (ms) from the 50%-of-peak crossing.

    Traces are in the nasal frame and centred on the event.  NaN for a trace
    with no positive peak (only-eye or only-head events).
    """

    def onset(x):
        pk = np.nanmax(x)
        if not np.isfinite(pk) or pk <= 0:
            return np.nan
        return float(np.argmax(x >= 0.5 * pk))

    th, te = onset(h), onset(e)
    return (th - te) * DT * 1000.0


def _name_motifs(H, E):
    """ Name clusters HbE / HwE / HD / ED from their mean kinematics.

    H, E : (4, W) mean head and eye velocity per cluster, nasal frame.
    ED = largest eye/head ratio, HD = smallest, and of the other two the one
    whose head leads (more negative onset lag) is HbE.
    """

    ratio = np.abs(E).mean(1) / np.maximum(np.abs(H).mean(1), 1e-6)
    order = list(np.argsort(ratio))
    hd, ed = order[0], order[-1]
    mid = order[1:3]
    lags = [_onset_lag(H[k], E[k]) for k in mid]
    lags = [l if np.isfinite(l) else 0.0 for l in lags]
    hbe = mid[int(np.argmin(lags))]
    hwe = mid[1 - int(np.argmin(lags))]
    name = {hbe: 0, hwe: 1, hd: 2, ed: 3}
    return name, ratio


def find_motifs(v, lab, seed=0):
    """ Cluster gaze-shift frames into four head-eye motifs and extract events.

    Hou et al. Methods 4.3: head/eye velocity and acceleration sampled at
    +-100 ms around each gaze-shift frame, z-scored, k-means k=4 per direction;
    clusters matched across directions; frames with a differently-labelled
    gaze-shift neighbour within 100 ms dropped; the first frame of each
    remaining same-motif run is the event onset.

    Temporal events are sign-flipped into the nasal frame before clustering, so
    the per-direction centroids are directly comparable for matching.

    Returns
    -------
    ev : dict of (E,) arrays
        onset, ep_start, ep_end (gaze-shift episode, end exclusive), direction
        (+1 nasal, -1 temporal), motif, lag_ms, head/eye/gaze amplitude, peak gaze.
    mi : dict
        Mean traces per motif x direction, naming metrics, frame labels.
    """

    from sklearn.cluster import KMeans

    n = len(lab)
    W = 2 * MOTIF_HALF + 1
    feats_src = [v['head'], v['eye'], v['head_acc'], v['eye_acc']]
    mot = np.full(n, -1, int)
    cent_raw = {}
    for dcode, sgn in ((1, 1.0), (2, -1.0)):
        idx = np.flatnonzero(lab == dcode)
        idx = idx[(idx >= MOTIF_HALF) & (idx < n - MOTIF_HALF)]
        F = np.stack([np.concatenate([sgn * x[i - MOTIF_HALF:i + MOTIF_HALF + 1]
                                      for x in feats_src]) for i in idx]) if len(idx) else np.zeros((0, 4 * W))
        good = np.all(np.isfinite(F), axis=1)
        idx, F = idx[good], F[good]
        if len(idx) < 40:
            continue
        mu, sd = F.mean(0), F.std(0)
        km = KMeans(4, n_init=20, random_state=seed).fit((F - mu) / np.where(sd > 0, sd, 1))
        mot[idx] = km.labels_
        cent_raw[dcode] = np.stack([F[km.labels_ == k].mean(0) for k in range(4)])

    # Match temporal clusters onto nasal ones, then name the nasal set.
    if 1 not in cent_raw or 2 not in cent_raw:
        raise RuntimeError('Too few gaze-shift frames to cluster both directions.')
    cost = np.linalg.norm(cent_raw[1][:, None, :] - cent_raw[2][None, :, :], axis=2)
    r_i, c_i = linear_sum_assignment(cost)
    t2n = {int(c): int(r) for r, c in zip(r_i, c_i)}
    match_ratio = float(np.mean([cost[r, c] / np.mean(np.delete(cost[r], c))
                                 for r, c in zip(r_i, c_i)]))
    m2 = mot.copy()
    for c, r in t2n.items():
        m2[(lab == 2) & (mot == c)] = r
    pooled = np.stack([(cent_raw[1][k] + cent_raw[2][[c for c, r in t2n.items() if r == k][0]]) / 2
                       for k in range(4)])
    name, ratio = _name_motifs(pooled[:, :W], pooled[:, W:2 * W])
    motif = np.full(n, -1, int)
    has = m2 >= 0
    motif[has] = np.vectorize(name.get)(m2[has])

    # Neighbour exclusion within +-100 ms.
    keep = motif >= 0
    for k in range(1, MOTIF_HALF + 1):
        for sh in (k, -k):
            nb = np.roll(motif, sh)
            if sh > 0:
                nb[:sh] = -1
            else:
                nb[sh:] = -1
            keep &= ~((nb >= 0) & (nb != motif))
    motif_k = np.where(keep, motif, -1)

    # Events: first frame of each same-motif run; episode = enclosing run of
    # same-direction gaze-shift frames (the excluded onset frame counts too).
    ep_id = np.full(n, -1, int)
    eps = _episodes(lab, v['gaze'])
    for k, (a, b, _) in enumerate(eps):
        ep_id[a:b] = k
    rows = []
    for mk in range(4):
        for a, b in _runs(motif_k == mk):
            # Runs are already direction-pure: a motif run cannot straddle a
            # direction change without a non-gaze-shift frame in between.
            d = 1 if lab[a] == 1 else -1
            e = ep_id[a]
            ea, eb = (eps[e][:2] if e >= 0 else (a, b))
            sg = float(d)
            h = sg * v['head'][a - MOTIF_HALF:a + MOTIF_HALF + 1]
            ey = sg * v['eye'][a - MOTIF_HALF:a + MOTIF_HALF + 1]
            seg = slice(ea, eb)
            rows.append((a, ea, eb, d, mk, _onset_lag(h, ey),
                         float(np.nansum(v['head'][seg]) * DT),
                         float(np.nansum(v['eye'][seg]) * DT),
                         float(np.nansum(v['gaze'][seg]) * DT),
                         float(np.nanmax(np.abs(v['gaze'][seg])))))
    rows.sort()
    cols = ('onset', 'ep_start', 'ep_end', 'direction', 'motif', 'lag_ms',
            'head_amp', 'eye_amp', 'gaze_amp', 'gaze_peak')
    ev = {c: np.array([r[i] for r in rows]) for i, c in enumerate(cols)}
    for c in ('onset', 'ep_start', 'ep_end', 'direction', 'motif'):
        ev[c] = ev[c].astype(int)

    # Mean peri-onset traces, +-250 ms, per motif x direction.
    half = 10
    tr = {}
    for mk in range(4):
        for d in (1, -1):
            on = ev['onset'][(ev['motif'] == mk) & (ev['direction'] == d)]
            on = on[(on >= half) & (on < n - half)]
            for key in ('head', 'eye', 'gaze'):
                seg = np.stack([v[key][i - half:i + half + 1] for i in on]) if len(on) else np.full((1, 2 * half + 1), np.nan)
                tr[(mk, d, key)] = (np.nanmean(seg, 0), np.nanstd(seg, 0), len(on))
    mi = {'motif_frames': motif_k, 'traces': tr, 'trace_half': half,
          'eye_head_ratio': {MOTIF_NAMES[name[k]]: float(ratio[k]) for k in range(4)},
          'match_ratio': match_ratio}
    counts = [int(((ev['motif'] == k)).sum()) for k in range(4)]
    print('  Motifs: {} events -- {}.  Cross-direction match distance ratio {:.2f} '
          '(matched / mean unmatched).'.format(len(ev['onset']),
                                               ', '.join('{} {}'.format(MOTIF_NAMES[k], counts[k]) for k in range(4)),
                                               match_ratio))
    return ev, mi


def motif_transitions(ev, max_gap_s=0.5, n_perm=N_NULL, seed=0):
    """ Lift matrix and permutation z-scores for consecutive motifs.

    Consecutive events <= 500 ms apart.  Lift = P(j | i) / P(j).  Null: motif
    labels shuffled across events with the pair structure kept (Hou et al. 4.4).
    """

    on, m = ev['onset'], ev['motif']
    pair = np.flatnonzero(np.diff(on) * DT <= max_gap_s)
    a, b = m[pair], m[pair + 1]
    C = np.zeros((4, 4))
    np.add.at(C, (a, b), 1)
    marg = np.bincount(m, minlength=4) / max(len(m), 1)
    P = C / np.maximum(C.sum(1, keepdims=True), 1)
    lift = P / np.where(marg > 0, marg, np.nan)[None, :]
    rng = np.random.default_rng(seed)
    null = np.zeros((n_perm, 4, 4))
    for k in range(n_perm):
        mm = rng.permutation(m)
        np.add.at(null[k], (mm[pair], mm[pair + 1]), 1)
    z = (C - null.mean(0)) / np.where(null.std(0) > 0, null.std(0), np.nan)
    return {'count': C, 'lift': lift, 'z': z, 'n_pairs': len(pair)}


def _smooth_nan(x, sigma):
    """ Gaussian smoothing over a gappy trace; gaps stay NaN. """

    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return np.full(len(x), np.nan)
    xf = np.interp(np.arange(len(x)), np.flatnonzero(ok), x[ok])
    y = gaussian_filter1d(xf, sigma)
    y[~ok] = np.nan
    return y


def behavior_states(s):
    """ Segment approach, investigation, rearing, wall-following, turning, rest.

    Mutually exclusive labels on the 40 Hz grid, assigned in priority order
    (STATE_NAMES).  Kinematics from the reconstruction's head position (mm) and
    heading; rearing from nose-up head pitch, which the ear-separation check in
    the PDF validates.  Thresholds are in the returned `params`.

    Returns
    -------
    st : dict
        `state` (N,) int index into STATE_NAMES (-1 no pose), kinematic traces,
        and `params`.
    """

    prm = {'smooth_s': 0.15, 'rear_pitch_deg': 25.0, 'rear_min_s': 0.4,
           'rear_max_speed': 80.0, 'pillar_near_mm': 70.0, 'invest_max_speed': 80.0,
           'invest_bearing_deg': 90.0, 'approach_min_speed': 30.0,
           'approach_closing': 30.0, 'approach_bearing_deg': 45.0,
           'approach_min_s': 0.5, 'approach_reach_s': 1.0, 'wall_near_mm': 50.0,
           'wall_min_speed': 30.0, 'wall_angle_deg': 30.0, 'wall_min_s': 0.5,
           'turn_rate': 60.0, 'turn_min_s': 0.3, 'loco_speed': 30.0, 'loco_min_s': 0.25,
           'rest_speed': 15.0, 'rest_rate': 20.0, 'rest_min_s': 2.0,
           'rear_pillar_mm': 100.0}
    n = len(s['t'])
    sig = prm['smooth_s'] * FS
    hx, hy = _smooth_nan(s['hx'], sig), _smooth_nan(s['hy'], sig)
    vx, vy = np.gradient(hx, DT), np.gradient(hy, DT)
    speed = np.hypot(vx, vy)
    yaw_rate = np.gradient(_smooth_nan(s['yaw_r'], sig), DT)
    nose_up = -s['pitch']

    px, py = s['pillar_xy']
    d_pil = np.hypot(hx - px, hy - py) - s['pillar_r']
    closing = -np.gradient(d_pil, DT)
    ego = _wrap(np.degrees(np.arctan2(py - hy, px - hx)) - s['yaw_r'])

    c = s['arena']
    dw = np.full((n, len(c)), np.inf)
    cosw = np.zeros((n, len(c)))
    mv = np.stack([vx, vy], 1) / np.maximum(speed, 1e-9)[:, None]
    for k in range(len(c)):
        a, b = c[k], c[(k + 1) % len(c)]
        ab = b - a
        L = np.linalg.norm(ab)
        tt = np.clip(((hx - a[0]) * ab[0] + (hy - a[1]) * ab[1]) / L ** 2, 0, 1)
        dw[:, k] = np.hypot(hx - (a[0] + tt * ab[0]), hy - (a[1] + tt * ab[1]))
        cosw[:, k] = np.abs(mv @ (ab / L))
    kw = np.argmin(dw, 1)
    d_wall = dw[np.arange(n), kw]
    para = cosw[np.arange(n), kw] > np.cos(np.radians(prm['wall_angle_deg']))

    ms = lambda sec: max(1, int(round(sec * FS)))
    rear = _min_bout((nose_up > prm['rear_pitch_deg']) & (speed < prm['rear_max_speed']),
                     ms(prm['rear_min_s']))
    rear_p = rear & (d_pil < prm['rear_pillar_mm'])
    invest = ((d_pil < prm['pillar_near_mm']) & (speed < prm['invest_max_speed'])
              & (np.abs(ego) < prm['invest_bearing_deg']))
    cand = _min_bout((speed > prm['approach_min_speed']) & (closing > prm['approach_closing'])
                     & (np.abs(ego) < prm['approach_bearing_deg'])
                     & (d_pil >= prm['pillar_near_mm']), ms(prm['approach_min_s']))
    # Approach only counts if the bout actually ends at the pillar.
    approach = np.zeros(n, bool)
    reach = ms(prm['approach_reach_s'])
    for a, b in _runs(cand):
        if np.nanmin(d_pil[a:min(n, b + reach)]) < prm['pillar_near_mm']:
            approach[a:b] = True
    wall = _min_bout((d_wall < prm['wall_near_mm']) & (speed > prm['wall_min_speed']) & para,
                     ms(prm['wall_min_s']))
    turn = _min_bout(np.abs(yaw_rate) > prm['turn_rate'], ms(prm['turn_min_s']))
    loco = _min_bout(speed > prm['loco_speed'], ms(prm['loco_min_s']))
    rest = _min_bout((speed < prm['rest_speed']) & (np.abs(yaw_rate) < prm['rest_rate']),
                     ms(prm['rest_min_s']))

    state = np.full(n, 8, int)
    # Lowest priority first, so later writes win.
    for k, m in ((7, rest), (6, loco), (5, turn), (4, wall), (0, approach),
                 (1, invest), (3, rear & ~rear_p), (2, rear_p)):
        state[m] = k
    state[~np.isfinite(speed) | ~s['pose_ok']] = -1
    st = {'state': state, 'speed': speed / 10.0, 'yaw_rate': yaw_rate, 'nose_up': nose_up,
          'd_pillar': d_pil, 'd_wall': d_wall, 'ego_pillar': ego, 'closing': closing,
          'hx_s': hx, 'hy_s': hy, 'params': prm}
    v = state >= 0
    print('  States: ' + ', '.join('{} {:.1f}%'.format(nm, 100 * np.mean(state[v] == k))
                                   for k, nm in enumerate(STATE_NAMES)))
    return st


def _net_orienting(az_pre, dB, dG, rng, k_pairs=48):
    """ Toward rate and eccentricity change, each minus its shift-shuffle chance.

    Events are gaze shifts with the pillar in a monocular field.  Chance is the
    shift-shuffle of _landing_stats restricted to these events: every event's
    start and world-bearing change kept, its gaze vector drawn from another
    event.  The toward chance is exact (sum over shift sign of P(shift sign) x
    P(pillar side)); the eccentricity chance is the mean over k_pairs random
    (start, shift) pairings per event.  The MEAN, not the median: a shift
    either brings the pillar ~20 deg nearer the nose or takes it ~20 deg away,
    so the distribution is bimodal and its median jumps between the modes (on
    DMM061 pos18 a label-permutation null of the median contrast had SD 13 deg
    and a -8.5 deg offset).

    Returns
    -------
    net_toward, net_ecc : float
        Fraction toward minus chance; mean eccentricity change (deg) minus chance.
    """

    n = len(az_pre)
    if n < 2:
        return np.nan, np.nan
    sg, sa = np.sign(dG), np.sign(az_pre)
    toward = np.mean(sg == sa)
    chance = sum(np.mean(sg == v) * np.mean(sa == v) for v in (-1.0, 0.0, 1.0))
    ecc = np.mean(np.abs(_wrap(az_pre + dB - dG)) - np.abs(az_pre))
    j = rng.integers(0, n, size=(n, k_pairs))
    ecc_null = np.mean(np.abs(_wrap(az_pre[:, None] + dB[:, None] - dG[j])) - np.abs(az_pre)[:, None])
    return float(toward - chance), float(ecc - ecc_null)


def _landing_stats(az_pre, az_post, dB, dG, occ_pool, occ_keys, half, rng):
    """ Observed landing statistics plus three null distributions.

    Parameters
    ----------
    az_pre, az_post : (E,)
        Pillar cyclopean azimuth 50 ms before / after the gaze-shift episode.
    dB, dG : (E,)
        Change in the pillar's world bearing and in cyclopean gaze direction
        across the episode, so az_post = az_pre + dB - dG.
    occ_pool : dict
        key -> frame-level az_cyc samples for the occupancy null.
    occ_keys : (E,)
        Pool key per event (light x state at onset).
    """

    E = len(az_pre)
    out = {'n': E}
    if E < 5:
        return out
    bino = lambda a: np.abs(a) <= half
    pre_b, post_b = bino(az_pre), bino(az_post)
    out['bino_pre'] = float(pre_b.mean())
    out['bino_post'] = float(post_b.mean())
    out_pre = ~pre_b
    out['n_out_pre'] = int(out_pre.sum())
    out['enter'] = float(post_b[out_pre].mean()) if out_pre.any() else np.nan
    out['d_abs'] = float(np.median(np.abs(az_post) - np.abs(az_pre)))

    # Null 1: occupancy at random moments of the same light x state.
    occ = np.zeros(N_NULL)
    for key in np.unique(occ_keys):
        pool = occ_pool.get(key)
        k = int((occ_keys == key).sum())
        if pool is None or len(pool) == 0:
            continue
        draw = pool[rng.integers(0, len(pool), size=(N_NULL, k))]
        occ += bino(draw).sum(1)
    occ /= E
    out['occ_null'] = occ
    out['p_occ'] = float((1 + np.sum(occ >= out['bino_post'])) / (1 + N_NULL))

    # Null 2: shift vectors shuffled across events (direction + amplitude).
    # Null 3: amplitudes shuffled within direction (direction kept).
    sh, sh_enter, sh_d = np.zeros(N_NULL), np.zeros(N_NULL), np.zeros(N_NULL)
    am, tw_null = np.zeros(N_NULL), np.zeros(N_NULL)
    mono = (np.abs(az_pre) > half) & (np.abs(az_pre) <= FIELD_EDGE_DEG)
    sgn = np.sign(dG)
    pos, neg = np.flatnonzero(sgn > 0), np.flatnonzero(sgn <= 0)
    for k in range(N_NULL):
        perm = rng.permutation(E)
        a = _wrap(az_pre + dB - dG[perm])
        tw_null[k] = np.mean(np.sign(dG[perm][mono]) == np.sign(az_pre[mono])) if mono.any() else np.nan
        sh[k] = bino(a).mean()
        sh_enter[k] = bino(a[out_pre]).mean() if out_pre.any() else np.nan
        sh_d[k] = np.median(np.abs(a) - np.abs(az_pre))
        mag = np.abs(dG).copy()
        mag[pos] = mag[pos][rng.permutation(len(pos))]
        mag[neg] = mag[neg][rng.permutation(len(neg))]
        am[k] = bino(_wrap(az_pre + dB - sgn * mag)).mean()
    out['shift_null'] = sh
    out['p_shift'] = float((1 + np.sum(sh >= out['bino_post'])) / (1 + N_NULL))
    out['enter_null'] = sh_enter
    out['p_enter'] = float((1 + np.sum(sh_enter >= out['enter'])) / (1 + N_NULL))
    out['d_abs_null'] = sh_d
    out['p_d_abs'] = float((1 + np.sum(sh_d <= out['d_abs'])) / (1 + N_NULL))
    out['amp_null'] = am
    out['p_amp'] = float((1 + np.sum(am >= out['bino_post'])) / (1 + N_NULL))

    # Direction: pillar in a monocular field before, does gaze turn toward it?
    # Chance is NOT 0.5: nasal shifts outnumber temporal and the pillar sits on
    # one side more often, so the null is the same shift-shuffle.
    tow = np.sign(dG[mono]) == np.sign(az_pre[mono])
    out['n_dir'] = int(mono.sum())
    out['toward'] = float(tow.mean()) if mono.any() else np.nan
    out['toward_null'] = tw_null
    hi = (1 + np.sum(tw_null >= out['toward'])) / (1 + N_NULL)
    lo = (1 + np.sum(tw_null <= out['toward'])) / (1 + N_NULL)
    out['p_toward'] = float(min(1.0, 2 * min(hi, lo)))
    out['p_toward_binom'] = (float(binomtest(int(tow.sum()), int(mono.sum()), 0.5).pvalue)
                             if mono.sum() > 0 else np.nan)
    hi = (1 + np.sum(sh_d <= out['d_abs'])) / (1 + N_NULL)
    lo = (1 + np.sum(sh_d >= out['d_abs'])) / (1 + N_NULL)
    out['p_d_abs_2s'] = float(min(1.0, 2 * min(hi, lo)))
    return out


def landing_analysis(ev, p, st, light, half=BINO_HALF_DEG, seed=0, label=''):
    """ Does a gaze shift put the pillar in the binocular zone more than chance?

    Landing is read LAND_PAD frames (50 ms) after the gaze-shift episode ends,
    against LAND_PAD before it starts.  Three nulls (see _landing_stats):
    state-matched occupancy, shift-vector shuffle, amplitude shuffle.  Run per
    light condition, and in the light per motif, state and pillar distance.

    Returns
    -------
    res : dict
        Nested results plus per-event arrays (`az_pre`, `az_post`, `keep`).
    """

    rng = np.random.default_rng(seed)
    n = len(p['az_cyc'])
    pre = ev['ep_start'] - LAND_PAD
    post = ev['ep_end'] - 1 + LAND_PAD
    ok = (pre >= 0) & (post < n)
    pre_c, post_c = np.clip(pre, 0, n - 1), np.clip(post, 0, n - 1)
    az_pre, az_post = p['az_cyc'][pre_c], p['az_cyc'][post_c]
    dB = _wrap(p['bearing'][post_c] - p['bearing'][pre_c])
    dG = _wrap(p['gaze_cyc'][post_c] - p['gaze_cyc'][pre_c])
    ok &= np.isfinite(az_pre) & np.isfinite(az_post) & np.isfinite(dG)
    on = ev['onset']
    lt = light[on]
    stt = st['state'][on]
    key = lt.astype(int) * 100 + stt
    pool = {}
    fin = np.isfinite(p['az_cyc']) & (st['state'] >= 0)
    for kk in np.unique(key):
        m = fin & (light == (kk // 100 == 1)) & (st['state'] == kk % 100)
        pool[kk] = p['az_cyc'][m]

    def run(mask):
        m = mask & ok
        return _landing_stats(az_pre[m], az_post[m], dB[m], dG[m], pool, key[m], half, rng)

    res = {'light': run(lt), 'dark': run(~lt), 'by_motif': {}, 'by_state': {}, 'by_dist': {}}
    for k, nm in enumerate(MOTIF_NAMES):
        res['by_motif'][nm] = run(lt & (ev['motif'] == k))
    for k, nm in enumerate(STATE_NAMES):
        res['by_state'][nm] = run(lt & (stt == k))
    dist = p['dist_mm'][pre_c]
    edges = [0, 100, 250, 1e9]
    for a, b in zip(edges[:-1], edges[1:]):
        nm = '<{:.0f} mm'.format(b) if a == 0 else ('>{:.0f} mm'.format(a) if b > 1e8 else
                                                    '{:.0f}-{:.0f} mm'.format(a, b))
        res['by_dist'][nm] = run(lt & (dist >= a) & (dist < b))
    # Light vs dark on the same events, as NET effects: each condition's toward
    # rate and eccentricity change minus that condition's own shift-shuffle
    # chance.  Raw rates are not comparable -- in the dark of DMM061 pos18 79%
    # of shifts go left and the pillar sits left 35% of the time, so chance
    # toward is 0.41 and the eccentricity null +14 deg, and a raw contrast
    # counts that motor bias as orienting.  Permute the condition label WITHIN
    # behavioural state and recompute each pseudo-condition's own chance.  Then
    # a block-level test on each block's net effect, since light and dark come
    # in a handful of long blocks and events in one block are not independent.
    m = ok & (np.abs(az_pre) > half) & (np.abs(az_pre) <= FIELD_EDGE_DEG)
    if m.sum() > 20 and lt[m].any() and (~lt[m]).any():
        a_m, b_m, g_m = az_pre[m], dB[m], dG[m]
        tw = (np.sign(g_m) == np.sign(a_m)).astype(float)
        dd = np.abs(_wrap(a_m + b_m - g_m)) - np.abs(a_m)
        L_ = lt[m]
        strata = [np.flatnonzero(stt[m] == k) for k in np.unique(stt[m])]

        def contrast(lab_):
            n_l = _net_orienting(a_m[lab_], b_m[lab_], g_m[lab_], rng)
            n_d = _net_orienting(a_m[~lab_], b_m[~lab_], g_m[~lab_], rng)
            return n_l[0] - n_d[0], n_l[1] - n_d[1]

        def perm():
            out_ = L_.copy()
            for idx in strata:
                out_[idx] = L_[idx][rng.permutation(len(idx))]
            return out_

        obs = contrast(L_)
        null = np.array([contrast(perm()) for _ in range(N_NULL)])
        p2 = [float(min(1.0, (1 + np.sum(np.abs(null[:, j] - np.mean(null[:, j]))
                                         >= abs(obs[j] - np.mean(null[:, j])))) / (1 + N_NULL)))
              for j in range(2)]
        net_l = _net_orienting(a_m[L_], b_m[L_], g_m[L_], rng)
        net_d = _net_orienting(a_m[~L_], b_m[~L_], g_m[~L_], rng)
        blk = np.cumsum(np.r_[0, np.diff(light.astype(int)) != 0])[on][m]
        b_ids = np.unique(blk)
        b_tw = np.array([tw[blk == b].mean() for b in b_ids])
        b_dd = np.array([np.mean(dd[blk == b]) for b in b_ids])
        b_net = np.array([_net_orienting(a_m[blk == b], b_m[blk == b], g_m[blk == b], rng)
                          if np.sum(blk == b) >= 5 else (np.nan, np.nan) for b in b_ids])
        b_lt = np.array([L_[blk == b][0] for b in b_ids])
        b_n = np.array([int((blk == b).sum()) for b in b_ids])
        from scipy.stats import mannwhitneyu

        def mw(x):
            f_ = np.isfinite(x)
            if not (b_lt & f_).any() or not (~b_lt & f_).any():
                return np.nan
            return float(mannwhitneyu(x[b_lt & f_], x[~b_lt & f_], alternative='two-sided',
                                      method='exact').pvalue)
        res['light_vs_dark'] = {'d_toward': float(obs[0]), 'p_toward': p2[0],
                                'd_ecc': float(obs[1]), 'p_ecc': p2[1], 'n': int(m.sum()),
                                'd_toward_raw': float(tw[L_].mean() - tw[~L_].mean()),
                                'd_ecc_raw': float(np.mean(dd[L_]) - np.mean(dd[~L_])),
                                'net_toward_light': float(net_l[0]), 'net_toward_dark': float(net_d[0]),
                                'net_ecc_light': float(net_l[1]), 'net_ecc_dark': float(net_d[1]),
                                'block_toward': b_tw, 'block_ecc': b_dd,
                                'block_toward_net': b_net[:, 0], 'block_ecc_net': b_net[:, 1],
                                'block_light': b_lt, 'block_n': b_n,
                                'p_block_toward': mw(b_net[:, 0]), 'p_block_ecc': mw(b_net[:, 1])}
        print('  Light vs dark (net of each condition\'s own chance): toward {:+.1f} points ({}; raw '
              '{:+.1f}), eccentricity change {:+.1f} deg ({}; raw {:+.1f}); block level p = {:.3g} / '
              '{:.3g}.'.format(100 * obs[0], _pfmt(p2[0]), 100 * res['light_vs_dark']['d_toward_raw'],
                               obs[1], _pfmt(p2[1]), res['light_vs_dark']['d_ecc_raw'],
                               res['light_vs_dark']['p_block_toward'], res['light_vs_dark']['p_block_ecc']))
    res['az_pre'], res['az_post'], res['dG'], res['dB'] = az_pre, az_post, dG, dB
    res['keep'] = ok
    L = res['light']
    if 'bino_post' in L:
        print('  Landing{} (light, n={}): binocular after {:.3f} vs before {:.3f}; '
              'occupancy null {:.3f} (p={:.4f}); shift-shuffle null {:.3f} (p={:.4f}); '
              'toward {:.3f} vs null {:.3f} of {} (p={:.2g}); eccentricity change {:+.1f} vs {:+.1f} deg '
              '(p={:.2g}).'.format(
                  label, L['n'], L['bino_post'], L['bino_pre'], L['occ_null'].mean(), L['p_occ'],
                  L['shift_null'].mean(), L['p_shift'], L['toward'], L['toward_null'].mean(), L['n_dir'],
                  L['p_toward'], L['d_abs'], L['d_abs_null'].mean(), L['p_d_abs_2s']))
    return res


def state_covariates(st, v, lab, ev_all, ev, p, fl, light):
    """ Per-state table of gaze, flow and pillar measures (light and dark).

    Returns
    -------
    tab : dict
        cond -> metric -> (n_states,) array, plus `metrics` (row order) and
        `labels` (pretty names).
    """

    metrics = [('frac_time', 'time (%)'), ('minutes', 'minutes'),
               ('shift_rate', 'gaze shifts / s'), ('comp_frac', 'comp. (% head-moving)'),
               ('HbE', 'HbE (%)'), ('HwE', 'HwE (%)'), ('HD', 'HD (%)'), ('ED', 'ED (%)'),
               ('flow_total', 'flow total (deg/s)'), ('flow_rot', 'flow rot. (deg/s)'),
               ('flow_trans', 'flow trans. (deg/s)'), ('slip_ratio', 'pillar slip eye/frozen'),
               ('pillar_bino', 'pillar binoc. (%)'), ('pillar_monoR', 'pillar monoc-R (%)'),
               ('pillar_dist', 'pillar dist (cm)'), ('pillar_size', 'pillar width (deg)'),
               ('speed', 'speed (cm/s)'), ('nose_up', 'nose-up (deg)')]
    tab = {'metrics': [m for m, _ in metrics], 'labels': [l for _, l in metrics]}
    S = st['state']
    for cond, lm in (('light', light), ('dark', ~light)):
        T = {m: np.full(len(STATE_NAMES), np.nan) for m, _ in metrics}
        tot = np.sum(lm & (S >= 0))
        for k in range(len(STATE_NAMES)):
            m = lm & (S == k)
            nf = int(m.sum())
            T['frac_time'][k] = 100 * nf / max(tot, 1)
            T['minutes'][k] = nf * DT / 60
            if nf < FS * 5:
                continue
            ea = (S[ev_all['onset']] == k) & lm[ev_all['onset']]
            T['shift_rate'][k] = ea.sum() / (nf * DT)
            e = (S[ev['onset']] == k) & lm[ev['onset']]
            for j, nm in enumerate(MOTIF_NAMES):
                T[nm][k] = 100 * np.mean(ev['motif'][e] == j) if e.sum() >= 5 else np.nan
            hm = m & (np.abs(v['head']) > SHIFTED_HEAD) & (lab >= 0)
            T['comp_frac'][k] = 100 * np.mean(lab[hm] == 0) if hm.sum() > 20 else np.nan
            T['flow_total'][k] = np.nanmedian(fl['total'][m])
            T['flow_rot'][k] = np.nanmedian(fl['rot'][m])
            T['flow_trans'][k] = np.nanmedian(fl['trans'][m])
            T['slip_ratio'][k] = np.nanmedian(p['slip'][m]) / np.nanmedian(p['slip_frozen'][m])
            z = p['zone'][m]
            z = z[z >= 0]
            if len(z):
                T['pillar_bino'][k] = 100 * np.mean(z == 0)
                T['pillar_monoR'][k] = 100 * np.mean(z == 1)
            T['pillar_dist'][k] = np.nanmedian(p['dist_mm'][m]) / 10
            T['pillar_size'][k] = np.nanmedian(p['width_deg'][m])
            T['speed'][k] = np.nanmedian(st['speed'][m])
            T['nose_up'][k] = np.nanmedian(st['nose_up'][m])
        tab[cond] = T
    return tab


def episode_events(lab, v):
    """ Every gaze-shift episode of a frame labelling, in find_motifs' format. """

    rows = []
    for a, b, d in _episodes(lab, v['gaze']):
        seg = slice(a, b)
        rows.append((a, a, b, d, -1, np.nan, float(np.nansum(v['head'][seg]) * DT),
                     float(np.nansum(v['eye'][seg]) * DT),
                     float(np.nansum(v['gaze'][seg]) * DT),
                     float(np.nanmax(np.abs(v['gaze'][seg])))))
    rows.sort()
    cols = ('onset', 'ep_start', 'ep_end', 'direction', 'motif', 'lag_ms',
            'head_amp', 'eye_amp', 'gaze_amp', 'gaze_peak')
    ev = {c: np.array([r[i] for r in rows]) for i, c in enumerate(cols)}
    for c in ('onset', 'ep_start', 'ep_end', 'direction', 'motif'):
        ev[c] = ev[c].astype(int)
    return ev


def _style(ax, grid_y=True):
    """ Recessive axes: no top/right spines, light grid. """

    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    if grid_y:
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.set_axisbelow(True)


def _new_page(title, subtitle=''):
    """ Landscape letter page with a title strip. """

    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.04, 0.965, title, fontsize=15, weight='bold', color=INK, va='top')
    if subtitle:
        import textwrap
        fig.text(0.04, 0.932, textwrap.fill(subtitle, 175), fontsize=9, color=INK2, va='top')
    return fig


def _pfmt(pv):
    """ p-value label. """

    if pv is None or not np.isfinite(pv):
        return 'p n/a'
    return 'p < {:.0e}'.format(1.0 / (N_NULL + 1)) if pv <= 1.0 / (N_NULL + 1) + 1e-12 else 'p = {:.3g}'.format(pv)


def _passes(L):
    """ True if a landing result beats both nulls at p < 0.05. """

    return 'bino_post' in L and L['p_shift'] < 0.05 and L['p_occ'] < 0.05


def _verdict(L, robust=None):
    """ One-line reading of a landing result.

    robust : list of landing results for the same question under other zone
    sizes / classifiers.  A pass that most of them do not share is called weak.
    """

    if 'bino_post' not in L:
        return 'too few events'
    a = L['p_shift'] < 0.05
    b = L['p_occ'] < 0.05
    if a and b:
        if robust:
            k = sum(_passes(x) for x in robust)
            if k < (len(robust) + 1) // 2:
                return ('WEAK -- beats both nulls at the default zone (+{:.1f} points), but only {} of {} '
                        'variants (zone size, classifier) agree'.format(
                            100 * (L['bino_post'] - L['shift_null'].mean()), k, len(robust)))
            return 'YES -- beats both nulls, and {} of {} variants agree'.format(k, len(robust))
        return 'YES -- beats both the shift-shuffle and the state-matched occupancy null'
    if a:
        return 'PARTIAL -- shifts are targeted given the start, but landing is no richer than state-matched occupancy'
    if b:
        return 'PARTIAL -- post-shift binocular occupancy exceeds the state baseline, but shuffled shift vectors do as well'
    return 'NO -- landing is not enriched against either null'


def _page_summary(pdf, R_):
    """ Page 1: what was run and the headline answers. """

    fig = _new_page('Behavior before neurons: {}'.format(R_['session']),
                    'Pillar on the retina, retinal optic flow, gaze-shift motifs (Hou et al. 2026), '
                    'object-directed gaze, behavioral states.  Generated {}.'.format(R_['date']))
    s, L, D = R_['s'], R_['land']['light'], R_['land']['dark']
    rob = [('motif events', R_['land_m']['light']), ('PETH thresholds', R_['land_thr']['light']),
           ('k-means literal', R_['land_k']['light'])]
    ifc, ffc = R_['ifc'], R_['ffc']
    lines = [
        ('Data', '{:.1f} min at 40 Hz; pose valid {:.1f}%, eye tracked {:.1f}%; light {:.1f}% of '
         'the session ({} light/dark blocks).'.format(len(s['t']) * DT / 60, 100 * s['pose_ok'].mean(),
                                                      100 * s['eye_ok'].mean(), 100 * s['light'].mean(),
                                                      R_['n_blocks'])),
        ('Checks', 'Re-rendering {} random frames from this pose reproduces the stored reconstruction '
         '(mean |luminance error| {:.3f}).  Analytic flow vs finite-differenced reprojection: r = {:.4f}.  '
         'Warping each frame onto the one {:.0f} ms later with pose + depth leaves {:.0f}% of the '
         'frame-to-frame image change ({:.0f}% if the eye is held still; {:.0f}% vs {:.0f}% on pairs where '
         'the eye moved > 3 deg).'.format(
             R_['val']['n'], R_['val']['mean_abs_err'], ffc['r'], ifc.get('gap_ms', np.nan),
             100 * ifc.get('ratio', np.nan), 100 * ifc.get('ratio_frozen', np.nan),
             100 * ifc.get('ratio_eye_moved', np.nan), 100 * ifc.get('ratio_frozen_eye_moved', np.nan))
         + ('  POSE MISMATCH: the pose npz and the images npz come from different renders, so geometry '
            'here does not describe the images -- re-run the pose-only render.'
            if R_['val'].get('pose_match') is False else '')),
        ('Pillar on retina', 'Within the 120 deg reconstruction on {:.1f}% of eye-valid frames.  '
         'Zone (light): binocular {:.1f}%, right-monocular {:.1f}%, left-monocular {:.1f}%, '
         'blind {:.1f}%.  Median width {:.1f} deg at {:.0f} mm.'.format(
             100 * R_['pil_sum']['in_recon'], *[100 * x for x in R_['pil_sum']['zone_light']],
             R_['pil_sum']['width_med'], R_['pil_sum']['dist_med'])),
        ('Optic flow', 'Median retinal flow {:.0f} deg/s (rotation {:.0f}, translation {:.0f}).  '
         'During compensatory movements the eye cuts rotational flow by {:.0f}% '
         '(eye-frozen {:.0f} vs eye-free {:.0f} deg/s); during gaze shifts it changes it by {:+.0f}%.'.format(
             R_['flow_sum']['total'], R_['flow_sum']['rot'], R_['flow_sum']['trans'],
             R_['flow_sum']['comp_stab'], R_['flow_sum']['comp_frozen'], R_['flow_sum']['comp_free'],
             -R_['flow_sum']['shift_stab'])),
        ('Movements', 'Eye-on-head gain while the head turns is only {:.2f}, so the paper\'s literal k-means '
         'splits on head speed and calls {:.0f}% of frames gaze shifts.  Downstream uses the gaze-velocity '
         'split Hou et al. say k-means reproduces (Parker et al. 2023): compensatory {:.1f}%, nasal {:.1f}%, '
         'temporal {:.1f}%; {} episodes ({:.2f}/s).  {} motif events: {}.'.format(
             abs(R_['v']['vor_slope']), R_['k_frac'], *R_['cls_frac'], len(R_['ev_all']['onset']),
             len(R_['ev_all']['onset']) / (np.sum(R_['lab'] != -1) * DT), len(R_['ev']['onset']),
             ', '.join('{} {:.1f}%'.format(nm, 100 * np.mean(R_['ev']['motif'] == k))
                       for k, nm in enumerate(MOTIF_NAMES)))),
        ('Object-directed gaze\n(light)', 'n = {} gaze shifts.  Pillar binocular {:.1f}% after vs {:.1f}% '
         'before.  Shift-shuffle null {:.1f}% ({}); state-matched occupancy null {:.1f}% ({}).  '
         'Shifts toward a monocular pillar {:.1f}% vs {:.1f}% shuffled (n={}, {}).  Pillar eccentricity '
         'change across a shift {:+.1f} deg vs {:+.1f} shuffled ({}).'.format(
             L.get('n', 0), 100 * L.get('bino_post', np.nan), 100 * L.get('bino_pre', np.nan),
             100 * np.mean(L.get('shift_null', [np.nan])), _pfmt(L.get('p_shift')),
             100 * np.mean(L.get('occ_null', [np.nan])), _pfmt(L.get('p_occ')),
             100 * L.get('toward', np.nan), 100 * np.mean(L.get('toward_null', [np.nan])), L.get('n_dir', 0),
             _pfmt(L.get('p_toward')), L.get('d_abs', np.nan), np.mean(L.get('d_abs_null', [np.nan])),
             _pfmt(L.get('p_d_abs_2s')))),
        ('Answer', 'Binocular landing: {}.  Orienting: {}'.format(
            _verdict(L, R_['robust']),
            ('YES -- in the light, shifts turn toward the pillar and cut its eccentricity; in the dark they do '
             'neither (block-level p = {:.3g})'.format(R_['land']['light_vs_dark']['p_block_ecc'])
             if (L.get('p_d_abs_2s', 1) < 0.05 and L.get('d_abs', 0) < 0
                 and R_['land'].get('light_vs_dark', {}).get('p_block_ecc', 1) < 0.05)
             else 'not established'))),
        ('Dark control', 'n = {}; binocular after {:.1f}% vs shift-shuffle {:.1f}% ({}), occupancy '
         '{:.1f}% ({}).  Toward {:.1f}% vs {:.1f}% shuffled ({}); eccentricity {:+.1f} vs {:+.1f} deg ({}).  '
         'Light minus dark, each net of its own chance: toward {:+.1f} points ({}; raw {:+.1f}), '
         'mean eccentricity change {:+.1f} deg ({}; raw {:+.1f}).'.format(
             D.get('n', 0), 100 * D.get('bino_post', np.nan),
             100 * np.mean(D.get('shift_null', [np.nan])), _pfmt(D.get('p_shift')),
             100 * np.mean(D.get('occ_null', [np.nan])), _pfmt(D.get('p_occ')),
             100 * D.get('toward', np.nan), 100 * np.mean(D.get('toward_null', [np.nan])),
             _pfmt(D.get('p_toward')), D.get('d_abs', np.nan), np.mean(D.get('d_abs_null', [np.nan])),
             _pfmt(D.get('p_d_abs_2s')),
             100 * R_['land'].get('light_vs_dark', {}).get('d_toward', np.nan),
             _pfmt(R_['land'].get('light_vs_dark', {}).get('p_toward')),
             100 * R_['land'].get('light_vs_dark', {}).get('d_toward_raw', np.nan),
             R_['land'].get('light_vs_dark', {}).get('d_ecc', np.nan),
             _pfmt(R_['land'].get('light_vs_dark', {}).get('p_ecc')),
             R_['land'].get('light_vs_dark', {}).get('d_ecc_raw', np.nan))
         + '  Block level (each block\'s net effect one point): toward p = {:.3g}, eccentricity p = {:.3g} '
         '(Mann-Whitney, {} vs {} blocks).'.format(
             R_['land'].get('light_vs_dark', {}).get('p_block_toward', np.nan),
             R_['land'].get('light_vs_dark', {}).get('p_block_ecc', np.nan),
             int(np.sum(R_['land'].get('light_vs_dark', {}).get('block_light', []))),
             int(np.sum(~np.asarray(R_['land'].get('light_vs_dark', {}).get('block_light', []), bool))))),
        ('Robustness', '  |  '.join('{}: n={}, {:.1f}% vs shift {:.1f}% ({}), occ {:.1f}% ({})'.format(
            nm, LT.get('n', 0), 100 * LT.get('bino_post', np.nan),
            100 * np.mean(LT.get('shift_null', [np.nan])), _pfmt(LT.get('p_shift')),
            100 * np.mean(LT.get('occ_null', [np.nan])), _pfmt(LT.get('p_occ'))) for nm, LT in rob)),
        ('States (light)', ', '.join('{} {:.1f}%'.format(nm, R_['cov']['light']['frac_time'][k])
                                     for k, nm in enumerate(STATE_NAMES))),
    ]
    import textwrap
    y = 0.885
    for head, body in lines:
        fig.text(0.04, y, head, fontsize=9.5, weight='bold', color=INK, va='top')
        wrapped = textwrap.fill(body, 135)
        fig.text(0.19, y, wrapped, fontsize=8.8, color=INK, va='top')
        y -= 0.0215 * (wrapped.count('\n') + 1) + 0.016
    fig.text(0.04, 0.03, 'Pages: 2 pillar geometry | 3 optic flow | 4 movement classes | 5 motifs | '
             '6-7 object-directed gaze | 8 behavioral states | 9 states as covariates | 10 methods & caveats | '
             '11+ neurons',
             fontsize=8.5, color=INK2)
    pdf.savefig(fig)
    plt.close(fig)


def _page_pillar(pdf, R_):
    """ Page 2: pillar azimuth, elevation, size and zone. """

    s, p = R_['s'], R_['p']
    fig = _new_page('Where the pillar falls on the retina',
                    'Angles from the reconstruction pose.  Retinal = right-eye optical frame (az + temporal); '
                    'cyclopean = head azimuth minus horizontal eye deviation (+ left).')
    gs = fig.add_gridspec(3, 4, left=0.06, right=0.98, top=0.88, bottom=0.07, hspace=0.55, wspace=0.38)
    t = s['t']
    # Example window: 90 s of light with the pillar moving through several zones.
    lt = s['light'] & np.isfinite(p['az_cyc'])
    win = int(90 * FS)
    cand = np.flatnonzero(lt[:-win])[::int(10 * FS)]
    if len(cand):
        sc = [np.nanstd(p['az_cyc'][c:c + win]) * np.mean(lt[c:c + win]) for c in cand]
        a0 = cand[int(np.nanargmax(sc))]
    else:
        a0 = 0
    sl = slice(a0, a0 + win)
    ax = fig.add_subplot(gs[0, :3])
    ax.axhspan(-BINO_HALF_DEG, BINO_HALF_DEG, color='#cde2fb', lw=0)
    for key, col, lw_, nm in (('az_cyc', '#2a78d6', 1.2, 'cyclopean az'),
                              ('az_eye', '#eb6834', 1.0, 'retinal az (R eye)')):
        y = p[key][sl].copy()
        # Break the line at +-180 wraps instead of drawing across the panel.
        y[1:][np.abs(np.diff(y)) > 180] = np.nan
        ax.plot(t[sl] - t[a0], y, color=col, lw=lw_, label=nm)
    ax.set_ylim(-180, 180)
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_ylabel('azimuth (deg)', fontsize=9)
    ax.set_title('Example 90 s (light); shaded = binocular zone (+-{:.0f} deg)'.format(BINO_HALF_DEG),
                 fontsize=9.5, loc='left')
    ax.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor='none', loc='lower right', ncol=2)
    _style(ax)
    ax = fig.add_subplot(gs[1, :3])
    ax.plot(t[sl] - t[a0], p['el_eye'][sl], color='#1baf7a', lw=1.0, label='retinal elevation (deg)')
    ax.plot(t[sl] - t[a0], p['width_deg'][sl], color='#6250d6', lw=1.0, label='angular width (deg)')
    ax.set_xlabel('time (s)', fontsize=9)
    ax.set_ylabel('deg', fontsize=9)
    ax.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor='none', loc='lower right', ncol=2)
    _style(ax)

    ax = fig.add_subplot(gs[0, 3])
    for cond, col, lm in (('light', '#2a78d6', s['light']), ('dark', '#52514e', ~s['light'])):
        z = p['zone'][lm]
        z = z[z >= 0]
        fr = [np.mean(z == k) * 100 for k in range(4)]
        off = -0.2 if cond == 'light' else 0.2
        ax.bar(np.arange(4) + off, fr, width=0.38, color=col, label=cond)
    ax.set_xticks(range(4))
    ax.set_xticklabels(['binoc', 'mono R', 'mono L', 'blind'], fontsize=8)
    ax.set_ylabel('% of frames', fontsize=9)
    ax.set_title('Pillar zone occupancy', fontsize=9.5, loc='left')
    ax.legend(fontsize=8, frameon=False)
    _style(ax)

    ax = fig.add_subplot(gs[1, 3])
    bins = np.linspace(-180, 180, 73)
    for lm, col, nm in ((s['light'], '#2a78d6', 'light'), (~s['light'], '#52514e', 'dark')):
        ax.hist(p['az_cyc'][lm & np.isfinite(p['az_cyc'])], bins, density=True, histtype='step',
                color=col, lw=1.5, label=nm)
    for x in (-FIELD_EDGE_DEG, -BINO_HALF_DEG, BINO_HALF_DEG, FIELD_EDGE_DEG):
        ax.axvline(x, color=INK2, lw=0.6, ls=':')
    ax.set_xlabel('cyclopean azimuth (deg, + left)', fontsize=9)
    ax.set_yticks([])
    ax.set_title('Pillar bearing', fontsize=9.5, loc='left')
    _style(ax, grid_y=False)

    ax = fig.add_subplot(gs[2, 0:2])
    m = s['light'] & np.isfinite(p['az_eye'])
    H, _, _ = np.histogram2d(p['az_eye'][m], p['el_eye'][m], bins=[np.arange(-180, 181, 5),
                                                                        np.arange(-90, 91, 5)])
    H = H.T / max(H.sum(), 1) * 100
    im = ax.imshow(np.log10(H + 1e-3), origin='lower', extent=[-180, 180, -90, 90], aspect='auto',
                   cmap='Blues', vmin=-3, vmax=np.log10(H.max() + 1e-3))
    ax.add_patch(plt.Rectangle((-60, -60), 120, 120, fill=False, ec='#eb6834', lw=1.4))
    ax.axvline(0, color=INK2, lw=0.5)
    ax.set_xlabel('retinal azimuth (deg, + temporal)', fontsize=9)
    ax.set_ylabel('retinal elevation (deg)', fontsize=9)
    ax.set_title('Pillar on the right retina, light (log10 % frames); box = reconstructed 120 deg',
                 fontsize=9.5, loc='left')
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
    cb.ax.tick_params(labelsize=7)
    _style(ax, grid_y=False)

    ax = fig.add_subplot(gs[2, 2])
    m = np.isfinite(p['dist_mm']) & np.isfinite(p['width_deg'])
    ax.hexbin(p['dist_mm'][m] / 10, p['width_deg'][m], gridsize=40, bins='log', cmap='Blues',
              mincnt=1, extent=(0, 70, 0, 60))
    ax.set_xlabel('distance to pillar surface (cm)', fontsize=9)
    ax.set_ylabel('angular width (deg)', fontsize=9)
    ax.set_title('Size vs distance', fontsize=9.5, loc='left')
    _style(ax, grid_y=False)

    ax = fig.add_subplot(gs[2, 3])
    ax.axis('off')
    ps = R_['pil_sum']
    txt = ('Median width {:.1f} deg (IQR {:.1f}-{:.1f})\n'
           'Median height {:.0f} deg\n'
           'Median distance {:.1f} cm\n'
           'In 120 deg reconstruction {:.1f}%\n'
           'In tracked (R) eye field {:.1f}%\n\n'
           'Re-render check: {} frames,\n'
           'mean |lum err| {:.3f}\n\n'
           'Zone model: binocular |az| <= {:.0f} deg,\n'
           'temporal edge {:.0f} deg, conjugate\n'
           'horizontal eyes assumed.').format(
        ps['width_med'], ps['width_q'][0], ps['width_q'][1], ps['height_med'], ps['dist_med'] / 10,
        100 * ps['in_recon'], 100 * ps['in_field'], R_['val']['n'], R_['val']['mean_abs_err'],
        BINO_HALF_DEG, FIELD_EDGE_DEG)
    ax.text(0, 1, txt, fontsize=8.5, va='top', color=INK)
    pdf.savefig(fig)
    plt.close(fig)


def _page_flow(pdf, R_):
    """ Page 3: retinal optic flow fields and the eye's effect on them. """

    s, fl, v, lab, st, ifc = R_['s'], R_['fl'], R_['v'], R_['lab'], R_['st'], R_['ifc']
    conds_note = 'shaded = gaze shift'
    fig = _new_page('Retinal optic flow',
                    'Analytic flow of the static scene (floor, walls, pillar) on a 15x15 grid over the '
                    'reconstructed 120 deg field.  Rotation + translation/depth; sky has no translational flow.')
    gs = fig.add_gridspec(3, 4, left=0.05, right=0.98, top=0.88, bottom=0.07, hspace=0.6, wspace=0.35)
    gaz, gel = fl['grid_az'], fl['grid_el']
    conds = [('compensatory, head turning left', (lab == 0) & (v['head'] > SHIFTED_HEAD)),
             ('nasal gaze shift', lab == 1), ('temporal gaze shift', lab == 2),
             ('forward run (>10 cm/s, |yaw|<30)', (st['speed'] > 10) & (np.abs(st['yaw_rate']) < 30))]
    for j, (nm, m) in enumerate(conds):
        ax = fig.add_subplot(gs[0, j])
        m = m & np.isfinite(fl['field'][:, 0, 0])
        if m.sum() > 10:
            mf = np.nanmean(fl['field'][m], 0)
            spd = np.hypot(mf[:, 0], mf[:, 1])
            ax.quiver(gaz, gel, mf[:, 0], mf[:, 1], spd, cmap='Blues', clim=(0, np.nanmax(spd)),
                      angles='xy', scale_units='xy', scale=np.nanmax(spd) / 7 + 1e-9, width=0.006)
            ax.set_title('{}\nn={}, |mean field| {:.0f} deg/s'.format(nm, int(m.sum()), np.nanmean(spd)),
                         fontsize=8.5, loc='left')
        ax.set_xlim(-65, 65)
        ax.set_ylim(-65, 65)
        ax.set_aspect('equal')
        ax.set_xlabel('az (+temporal)', fontsize=8)
        if j == 0:
            ax.set_ylabel('el (deg)', fontsize=8)
        _style(ax, grid_y=False)

    # Eye's effect on rotational flow.
    for j, (nm, m) in enumerate((('compensatory', (lab == 0) & (np.abs(v['head']) > SHIFTED_HEAD)),
                                 ('gaze shift', (lab == 1) | (lab == 2)))):
        ax = fig.add_subplot(gs[1, j])
        mm = m & np.isfinite(fl['rot']) & np.isfinite(fl['rot_frozen'])
        bins = np.linspace(0, 600, 61)
        ax.hist(fl['rot_frozen'][mm], bins, histtype='step', color='#52514e', lw=1.5, label='eye frozen')
        ax.hist(fl['rot'][mm], bins, histtype='step', color='#2a78d6', lw=1.5, label='eye free')
        ax.set_xlabel('rotational flow (deg/s)', fontsize=8.5)
        ax.set_title('{}: median {:.0f} vs {:.0f}'.format(nm, np.median(fl['rot_frozen'][mm]),
                                                         np.median(fl['rot'][mm])), fontsize=8.5, loc='left')
        ax.legend(fontsize=7.5, frameon=False)
        ax.set_yticks([])
        _style(ax, grid_y=False)

    ax = fig.add_subplot(gs[1, 2])
    sp = st['speed']
    bins = np.arange(0, 41, 2.5)
    for key, col, nm in (('trans', '#1baf7a', 'translation'), ('rot', '#2a78d6', 'rotation')):
        med = [np.nanmedian(fl[key][(sp >= a) & (sp < b)]) for a, b in zip(bins[:-1], bins[1:])]
        ax.plot((bins[:-1] + bins[1:]) / 2, med, color=col, lw=2, marker='o', ms=3, label=nm)
    ax.set_xlabel('speed (cm/s)', fontsize=8.5)
    ax.set_ylabel('median flow (deg/s)', fontsize=8.5)
    ax.set_title('Flow components vs running speed', fontsize=8.5, loc='left')
    ax.legend(fontsize=7.5, frameon=False)
    _style(ax)

    ax = fig.add_subplot(gs[1, 3])
    if 'warp' in ifc:
        bins = np.linspace(0, 1.5, 31)
        ax.hist(ifc['warp'] / ifc['ident'], bins, histtype='step', color='#2a78d6', lw=1.5,
                label='pose (median {:.2f})'.format(ifc['ratio']))
        ax.hist(ifc['warp_frozen'] / ifc['ident_frozen'], bins, histtype='step', color='#52514e', lw=1.5,
                label='eye held (median {:.2f})'.format(ifc['ratio_frozen']))
        ax.axvline(1, color=INK2, lw=0.6, ls=':')
        ax.legend(fontsize=7, frameon=False)
    ax.set_xlabel('warp error / no-motion error', fontsize=8.5)
    ax.set_yticks([])
    ax.set_title('Warp check vs rendered frames\nformula check r = {:.4f}'.format(R_['ffc']['r']),
                 fontsize=8.5, loc='left')
    _style(ax, grid_y=False)

    ax = fig.add_subplot(gs[2, :])
    win = int(20 * FS)
    act = np.convolve(((lab == 1) | (lab == 2)) & s['light'] & np.isfinite(fl['total']),
                      np.ones(win), mode='valid')
    a0 = int(np.argmax(act))
    sl = slice(a0, a0 + win)
    tt = s['t'][sl] - s['t'][a0]
    ax.plot(tt, fl['rot_frozen'][sl], color='#52514e', lw=1, label='rotation, eye frozen')
    ax.plot(tt, fl['rot'][sl], color='#2a78d6', lw=1.2, label='rotation, eye free')
    ax.plot(tt, fl['trans'][sl], color='#1baf7a', lw=1.2, label='translation')
    gs_ = (lab[sl] == 1) | (lab[sl] == 2)
    for a, b in _runs(gs_):
        ax.axvspan(tt[a], tt[min(b, len(tt) - 1)], color='#fde7c2', lw=0)
    ax.set_xlabel('time (s); {}'.format(conds_note), fontsize=8.5)
    ax.set_ylabel('deg/s', fontsize=8.5)
    ax.set_title('Example 20 s', fontsize=8.5, loc='left')
    ax.legend(fontsize=7.5, frameon=False, ncol=3, loc='upper right')
    _style(ax)
    pdf.savefig(fig)
    plt.close(fig)


def _page_classes(pdf, R_):
    """ Page 4: compensatory vs gaze-shift classification, literal k-means and gated. """

    v, lab, lab_k, s = R_['v'], R_['lab'], R_['lab_k'], R_['s']
    fig = _new_page('Compensatory movements and gaze shifts',
                    'Left: Hou et al. k-means as written.  Middle: gaze-velocity split used downstream '
                    '(|gaze| > 240 deg/s; compensatory |head| > 60, |gaze| < 120).  + = leftward = nasal (right eye).')
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.97, top=0.87, bottom=0.08, hspace=0.45, wspace=0.35)
    rng = np.random.default_rng(0)
    for j, (L, nm) in enumerate(((lab_k, 'k-means k=3 (literal)'), (lab, 'gaze-velocity split (used)'))):
        ax = fig.add_subplot(gs[0, j])
        ok = L != -1
        for code, cn in ((3, 'none'), (0, 'comp'), (1, 'nasal'), (2, 'temporal')):
            idx = np.flatnonzero(L == code)
            if not len(idx):
                continue
            idx = rng.choice(idx, min(5000, len(idx)), replace=False)
            ax.scatter(v['head'][idx], v['eye'][idx], s=2, alpha=0.35, color=CLASS_COLORS[cn],
                       label='{} {:.1f}%'.format('other' if code == 3 else cn, 100 * np.mean(L[ok] == code)))
        ax.plot([-600, 600], [600, -600], color=INK2, lw=0.6, ls='--')
        ax.axhline(0, color=INK2, lw=0.5)
        ax.axvline(0, color=INK2, lw=0.5)
        ax.set_xlim(-600, 600)
        ax.set_ylim(-400, 400)
        ax.set_xlabel('head velocity (deg/s)', fontsize=8.5)
        ax.set_ylabel('eye velocity (deg/s)', fontsize=8.5)
        ax.set_title(nm, fontsize=9.5, loc='left')
        ax.legend(fontsize=7, frameon=False, markerscale=5, loc='upper right')
        _style(ax, grid_y=False)

    ax = fig.add_subplot(gs[0, 2])
    bins = np.linspace(-800, 800, 81)
    for code, cn in ((0, 'comp'), (1, 'nasal'), (2, 'temporal')):
        ax.hist(v['gaze'][lab_k == code], bins, histtype='step', lw=1.0, ls='--', color=CLASS_COLORS[cn])
        ax.hist(v['gaze'][lab == code], bins, histtype='step', lw=1.6, color=CLASS_COLORS[cn], label=cn)
    ax.set_yscale('log')
    ax.set_xlabel('gaze velocity (deg/s)', fontsize=8.5)
    ax.set_title('Gaze velocity by class\n(solid used, dashed k-means)', fontsize=9, loc='left')
    ax.legend(fontsize=7.5, frameon=False)
    _style(ax, grid_y=False)

    ax = fig.add_subplot(gs[1, 0])
    rows, cols = [0, 1, 2], [0, 1, 2, 3]
    M = np.array([[np.sum((lab_k == r) & (np.where(lab == -2, 3, lab) == c)) for c in cols] for r in rows], float)
    Mn = M / np.maximum(M.sum(1, keepdims=True), 1) * 100
    ax.imshow(Mn, cmap='Blues', vmin=0, vmax=100, aspect='auto')
    for i in range(3):
        for j in range(4):
            ax.text(j, i, '{:.0f}%'.format(Mn[i, j]), ha='center', va='center', fontsize=8,
                    color='white' if Mn[i, j] > 55 else INK)
    ax.set_xticks(range(4))
    ax.set_xticklabels(['comp', 'nasal', 'temp', 'other'], fontsize=8)
    ax.set_yticks(range(3))
    ax.set_yticklabels(['comp', 'nasal', 'temp'], fontsize=8)
    ax.set_xlabel('gaze-velocity split', fontsize=8.5)
    ax.set_ylabel('k-means literal', fontsize=8.5)
    ax.set_title('Agreement (row %)', fontsize=9.5, loc='left')

    ax = fig.add_subplot(gs[1, 1])
    for lm, nm, off, col in ((s['light'], 'light', -0.2, '#2a78d6'), (~s['light'], 'dark', 0.2, '#52514e')):
        m = lm & (lab != -1)
        fr = [100 * np.mean(lab[m] == 0), 100 * np.mean(lab[m] == 1), 100 * np.mean(lab[m] == 2)]
        ax.bar(np.arange(3) + off, fr, width=0.38, color=col, label=nm)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['comp', 'nasal', 'temporal'], fontsize=8)
    ax.set_ylabel('% of valid frames', fontsize=8.5)
    ax.set_title('Class occupancy (used), light vs dark', fontsize=9, loc='left')
    ax.legend(fontsize=7.5, frameon=False)
    _style(ax)

    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    cen = R_['cls_info']['centres_head_eye']
    sk = (lab_k == 1) | (lab_k == 2)
    txt = ('k-means centres (head, eye deg/s):\n'
           '  comp     {:+6.1f}, {:+6.1f}\n  nasal    {:+6.1f}, {:+6.1f}\n  temporal {:+6.1f}, {:+6.1f}\n'
           'k-means "gaze shift" frames with\n|gaze| < 60 deg/s: {:.0f}%\n'
           'k-means motifs (literal):\n  {}\n\n'
           'Eye-on-head slope while head turns\n(30-300 deg/s): {:.2f}.  Mouse VOR\n'
           'gain is usually ~0.6-0.9; a gain this\nlow is worth checking against the\n'
           'eye-camera angle calibration.\n\n'
           'Smoothing: Savitzky-Golay {} samples,\norder {} (not given in the paper).').format(
        *cen.ravel(), 100 * np.mean(np.abs(v['gaze'][sk]) < 60),
        ', '.join('{} {}'.format(n_, c_) for n_, c_ in zip(MOTIF_NAMES, R_['k_counts'])),
        v['vor_slope'], SG_WIN, SG_ORD)
    ax.text(0, 1, txt, fontsize=8, va='top', family='monospace', color=INK)
    pdf.savefig(fig)
    plt.close(fig)


def _page_motifs(pdf, R_):
    """ Page 5: four head-eye motifs, timing and sequence. """

    ev, mi, tr = R_['ev'], R_['mi'], R_['trans']
    fig = _new_page('Gaze-shift motifs',
                    'Head/eye velocity + acceleration at +-100 ms, z-scored, k-means k=4 per direction, '
                    'matched across directions.  Mean +- SD, aligned to onset; temporal panels sign-flipped to nasal.')
    gs = fig.add_gridspec(3, 4, left=0.06, right=0.98, top=0.87, bottom=0.07, hspace=0.65, wspace=0.32)
    half = mi['trace_half']
    tt = np.arange(-half, half + 1) * DT * 1000
    paper_pct = {'HbE': 16.51, 'HwE': 11.60, 'HD': 21.94, 'ED': 49.94}
    for k, nm in enumerate(MOTIF_NAMES):
        for r, d in enumerate((1, -1)):
            ax = fig.add_subplot(gs[r, k])
            for key, col in (('head', '#2a78d6'), ('eye', '#eb6834')):
                mu, sd, nn = mi['traces'][(k, d, key)]
                mu, sd = d * mu, sd
                ax.fill_between(tt, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
                ax.plot(tt, mu, color=col, lw=1.6, label=key)
            ax.axvline(0, color=INK2, lw=0.5)
            ax.axhline(0, color=INK2, lw=0.5)
            ax.set_title('{} {}  n={}'.format(nm, 'nasal' if d == 1 else 'temporal', nn), fontsize=9, loc='left')
            if k == 0:
                ax.set_ylabel('deg/s (nasal +)', fontsize=8)
            if r == 1:
                ax.set_xlabel('ms from onset', fontsize=8)
            if k == 0 and r == 0:
                ax.legend(fontsize=7.5, frameon=False)
            _style(ax, grid_y=False)

    ax = fig.add_subplot(gs[2, 0])
    for k, nm in enumerate(MOTIF_NAMES):
        lg = ev['lag_ms'][(ev['motif'] == k) & np.isfinite(ev['lag_ms'])]
        if len(lg):
            xs = np.sort(lg)
            ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), color=MOTIF_COLORS[k], lw=1.5, label=nm)
    ax.axvline(0, color=INK2, lw=0.5)
    ax.set_xlabel('head - eye onset lag (ms; - = head first)', fontsize=8)
    ax.set_ylabel('cumulative', fontsize=8)
    ax.legend(fontsize=7.5, frameon=False)
    _style(ax)

    ax = fig.add_subplot(gs[2, 1])
    lift = tr['lift']
    ax.imshow(np.log2(lift), cmap='RdBu_r', vmin=-1.5, vmax=1.5)
    for i in range(4):
        for j in range(4):
            star = '*' if np.isfinite(tr['z'][i, j]) and tr['z'][i, j] > 1.96 else ''
            ax.text(j, i, '{:.2f}{}'.format(lift[i, j], star), ha='center', va='center', fontsize=7.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(MOTIF_NAMES, fontsize=8)
    ax.set_yticks(range(4))
    ax.set_yticklabels(MOTIF_NAMES, fontsize=8)
    ax.set_xlabel('next', fontsize=8)
    ax.set_ylabel('current', fontsize=8)
    ax.set_title('Transition lift (* z>1.96), {} pairs'.format(tr['n_pairs']), fontsize=8.5, loc='left')

    ax = fig.add_subplot(gs[2, 2:])
    ax.axis('off')
    hdr = '{:<5}{:>8}{:>9}{:>9}{:>10}{:>10}{:>10}{:>10}'.format('motif', 'events', 'this %', 'paper %',
                                                               'eye/head', 'gaze amp', 'eye amp', 'lag ms')
    lines = [hdr]
    for k, nm in enumerate(MOTIF_NAMES):
        m = ev['motif'] == k
        lines.append('{:<5}{:>8d}{:>9.1f}{:>9.1f}{:>10.2f}{:>10.1f}{:>10.1f}{:>10.0f}'.format(
            nm, int(m.sum()), 100 * m.mean(), paper_pct[nm], mi['eye_head_ratio'][nm],
            np.nanmedian(np.abs(ev['gaze_amp'][m])), np.nanmedian(np.abs(ev['eye_amp'][m])),
            np.nanmedian(ev['lag_ms'][m])))
    lines += ['', 'Naming: ED = highest eye/head energy ratio, HD = lowest; of the',
              'remaining two, the head-leading one is HbE.  Amplitudes are medians',
              '(deg) over the enclosing gaze-shift episode.  Cross-direction',
              'match distance ratio {:.2f} (Hou et al. report 2.2-4.5x separation,'.format(mi['match_ratio']),
              'i.e. ratio 0.22-0.45).']
    ax.text(0, 1, '\n'.join(lines), fontsize=8, va='top', family='monospace', color=INK)
    pdf.savefig(fig)
    plt.close(fig)


def _null_bar(ax, res, title):
    """ Observed vs null binocular-landing fractions for one condition set. """

    names = list(res.keys())
    x = np.arange(len(names))
    for i, nm in enumerate(names):
        L = res[nm]
        if 'bino_post' not in L:
            ax.text(i, 2, 'n={}'.format(L.get('n', 0)), ha='center', fontsize=7, color=INK2)
            continue
        for off, key, col in ((-0.18, 'occ_null', '#b9b8b2'), (0.0, 'shift_null', '#86b6ef')):
            q = np.percentile(L[key], [2.5, 50, 97.5]) * 100
            ax.plot([i + off, i + off], [q[0], q[2]], color=col, lw=3, solid_capstyle='round')
            ax.plot(i + off, q[1], 'o', color=col, ms=5)
        ax.plot(i + 0.18, 100 * L['bino_post'], 'D', color='#eb6834', ms=6)
        star = ''.join(['s' if L['p_shift'] < 0.05 else '', 'o' if L['p_occ'] < 0.05 else ''])
        if star:
            ax.text(i + 0.18, 100 * L['bino_post'] + 1.5, star, ha='center', fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(['{}\nn={}'.format(nm, res[nm].get('n', 0)) for nm in names], fontsize=7.5)
    ax.set_ylabel('% landing binocular', fontsize=8.5)
    ax.set_title(title, fontsize=9, loc='left')
    _style(ax)


def _page_landing(pdf, R_):
    """ Page 6: does the pillar land in the binocular zone after a shift? """

    land, s, p, ev = R_['land'], R_['s'], R_['p'], R_['ev_all']
    fig = _new_page('Object-directed gaze: does a shift land the pillar in the binocular zone?',
                    'Landing read 50 ms after the gaze-shift episode, start 50 ms before.  Nulls: shift-shuffle '
                    '(each shift\'s gaze vector given to another start), state-matched occupancy (random '
                    'moments, same light x state).')
    gs = fig.add_gridspec(2, 4, left=0.06, right=0.98, top=0.84, bottom=0.08, hspace=0.5, wspace=0.4)
    ax = fig.add_subplot(gs[0, 0])
    conds = [('light', land['light']), ('dark', land['dark']), ('motif ev.', R_['land_m']['light']),
             ('PETH', R_['land_thr']['light']), ('k-means', R_['land_k']['light'])]
    for i, (nm, L) in enumerate(conds):
        if 'bino_post' not in L:
            continue
        ax.bar(i - 0.27, 100 * L['bino_pre'], width=0.18, color='#c9c8c2', label='before' if i == 0 else None)
        ax.bar(i - 0.09, 100 * L['bino_post'], width=0.18, color='#eb6834', label='after' if i == 0 else None)
        ax.bar(i + 0.09, 100 * L['shift_null'].mean(), width=0.18, color='#86b6ef',
               label='shift-shuffle' if i == 0 else None,
               yerr=[[100 * (L['shift_null'].mean() - np.percentile(L['shift_null'], 2.5))],
                     [100 * (np.percentile(L['shift_null'], 97.5) - L['shift_null'].mean())]], capsize=2)
        ax.bar(i + 0.27, 100 * L['occ_null'].mean(), width=0.18, color='#52514e',
               label='occupancy' if i == 0 else None,
               yerr=[[100 * (L['occ_null'].mean() - np.percentile(L['occ_null'], 2.5))],
                     [100 * (np.percentile(L['occ_null'], 97.5) - L['occ_null'].mean())]], capsize=2)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c[0] for c in conds], fontsize=7.5, rotation=30, ha='right')
    ax.set_ylabel('% pillar binocular', fontsize=8.5)
    ax.set_title('Observed vs nulls (95% CI)', fontsize=9, loc='left')
    ax.legend(fontsize=7, frameon=False, loc='upper right')
    _style(ax)

    for j, (key, nm) in enumerate((('shift_null', 'shift-shuffle'), ('occ_null', 'occupancy'))):
        ax = fig.add_subplot(gs[0, 1 + j])
        for cond, col in (('light', '#2a78d6'), ('dark', '#52514e')):
            L = land[cond]
            if 'bino_post' not in L:
                continue
            ax.hist(100 * L[key], 40, color=col, alpha=0.45, label='{} null'.format(cond))
            ax.axvline(100 * L['bino_post'], color=col, lw=2,
                       label='{} obs ({})'.format(cond, _pfmt(L['p_' + ('shift' if j == 0 else 'occ')])))
        ax.set_xlabel('% landing binocular', fontsize=8.5)
        ax.set_yticks([])
        ax.set_title('{} null distribution'.format(nm), fontsize=9, loc='left')
        ax.legend(fontsize=6.5, frameon=False, loc='upper left', bbox_to_anchor=(0.0, -0.18), ncol=2)
        _style(ax, grid_y=False)

    ax = fig.add_subplot(gs[0, 3])
    for i, (cond, col) in enumerate((('light', '#2a78d6'), ('dark', '#52514e'))):
        L = land[cond]
        if 'toward' not in L:
            continue
        q = np.percentile(L['toward_null'], [2.5, 50, 97.5]) * 100
        ax.plot([i - 0.15, i - 0.15], [q[0], q[2]], color='#86b6ef', lw=3, solid_capstyle='round')
        ax.plot(i - 0.15, q[1], 'o', color='#86b6ef', ms=5)
        ax.plot(i + 0.15, 100 * L['toward'], 'D', color=col, ms=7)
        ax.text(i, max(q[2], 100 * L['toward']) + 2, 'n={}\n{}'.format(L['n_dir'], _pfmt(L['p_toward'])),
                ha='center', fontsize=7, color=INK2)
    lvd = land.get('light_vs_dark', {})
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['light', 'dark'], fontsize=8)
    ax.set_xlim(-0.6, 1.6)
    lo_ = min(100 * min(np.percentile(land[c]['toward_null'], 2.5), land[c]['toward'])
              for c in ('light', 'dark') if 'toward' in land[c])
    hi_ = max(100 * max(np.percentile(land[c]['toward_null'], 97.5), land[c]['toward'])
              for c in ('light', 'dark') if 'toward' in land[c])
    ax.set_ylim(lo_ - 3, hi_ + 8)
    ax.set_ylabel('% shifts toward pillar side', fontsize=8.5)
    ax.set_title('Toward a monocular pillar\n(blue = shift-shuffle; light-dark net {})'.format(
        _pfmt(lvd.get('p_toward'))), fontsize=8.5, loc='left')
    _style(ax)

    ax = fig.add_subplot(gs[1, 0:2])
    half = int(0.6 * FS)
    tt = np.arange(-half, half + 1) * DT * 1000
    for cond, lm, col in (('light', s['light'], '#2a78d6'), ('dark', ~s['light'], '#52514e')):
        on = ev['onset'][lm[ev['onset']]]
        on = on[(on >= half) & (on < len(s['t']) - half)]
        seg = np.stack([np.abs(p['az_cyc'][i - half:i + half + 1]) for i in on])
        mu = np.nanmean(seg, 0)
        se = np.nanstd(seg, 0) / np.sqrt(np.sum(np.isfinite(seg), 0))
        ax.fill_between(tt, mu - se, mu + se, color=col, alpha=0.2, lw=0)
        ax.plot(tt, mu, color=col, lw=1.8, label='{} (n={})'.format(cond, len(on)))
    ax.axvline(0, color=INK2, lw=0.6)
    ax.axhline(BINO_HALF_DEG, color='#eb6834', lw=0.8, ls='--', label='binocular edge')
    ax.set_xlabel('ms from gaze-shift onset', fontsize=8.5)
    ax.set_ylabel('|pillar cyclopean az| (deg)', fontsize=8.5)
    ax.set_title('Pillar eccentricity around gaze shifts (mean +- SEM)', fontsize=9, loc='left')
    ax.legend(fontsize=7.5, frameon=False)
    _style(ax)

    ax = fig.add_subplot(gs[1, 2])
    keep = land['keep'] & s['light'][ev['onset']]
    ax.axhspan(-BINO_HALF_DEG, BINO_HALF_DEG, color='#fde7c2', lw=0)
    ax.axvspan(-BINO_HALF_DEG, BINO_HALF_DEG, color='#eeeeea', lw=0)
    ax.scatter(land['az_pre'][keep], land['az_post'][keep], s=3, alpha=0.35,
               c=np.where(ev['direction'][keep] > 0, '#2a78d6', '#eb6834'))
    ax.plot([-180, 180], [-180, 180], color=INK2, lw=0.5)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xlabel('pillar az before (deg)', fontsize=8.5)
    ax.set_ylabel('pillar az after (deg)', fontsize=8.5)
    ax.set_title('Light: before vs after\n(blue nasal, orange temporal)', fontsize=8.5, loc='left')
    _style(ax, grid_y=False)

    ax = fig.add_subplot(gs[1, 3])
    sens = R_['sens']
    hw = sorted(sens.keys())
    ax.plot(hw, [100 * sens[h]['bino_post'] for h in hw], 'D-', color='#eb6834', label='observed')
    ax.plot(hw, [100 * sens[h]['shift_null'].mean() for h in hw], 'o-', color='#86b6ef', label='shift-shuffle')
    ax.plot(hw, [100 * sens[h]['occ_null'].mean() for h in hw], 's-', color='#52514e', label='occupancy')
    for h in hw:
        ax.text(h, 100 * sens[h]['bino_post'] + 1.5, 's' * (sens[h]['p_shift'] < 0.05) + 'o' * (sens[h]['p_occ'] < 0.05),
                ha='center', fontsize=7, color=INK2)
    ax.set_xlabel('binocular half-width (deg)', fontsize=8.5)
    ax.set_ylabel('% landing binocular (light)', fontsize=8.5)
    ax.set_title('Sensitivity to zone size\n(s/o: beats shift / occupancy)', fontsize=8.5, loc='left')
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    pdf.savefig(fig)
    plt.close(fig)


def _page_landing_breakdown(pdf, R_):
    """ Page 7: landing by motif, state and distance. """

    land = R_['land']
    fig = _new_page('Object-directed gaze, broken down',
                    'Orange diamond = observed; grey = state-matched occupancy null; blue = shift-shuffle null '
                    '(median, 95% interval).  Labels: s = beats shift-shuffle, o = beats occupancy (p<0.05, uncorrected).')
    gs = fig.add_gridspec(2, 3, left=0.07, right=0.98, top=0.84, bottom=0.12, hspace=0.6, wspace=0.35)
    _null_bar(fig.add_subplot(gs[0, 0]), land['by_motif'], 'By motif')
    _null_bar(fig.add_subplot(gs[0, 1]), land['by_dist'], 'By distance to pillar at shift start')
    _null_bar(fig.add_subplot(gs[1, :]), land['by_state'], 'By behavioral state at onset')
    ax = fig.add_subplot(gs[0, 2])
    lvd = land.get('light_vs_dark', {})
    if 'block_toward' in lvd:
        bl = lvd['block_light']
        for xx, sel, col in ((0, bl, '#2a78d6'), (1, ~bl, '#52514e')):
            yy = 100 * lvd['block_toward_net'][sel]
            ax.scatter(np.full(len(yy), xx) + np.linspace(-0.12, 0.12, len(yy)), yy, s=6 + lvd['block_n'][sel] / 2,
                       color=col, alpha=0.8)
        ax.axhline(0, color=INK2, lw=0.6, ls=':')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['light blocks', 'dark blocks'], fontsize=8)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel('% toward pillar minus block chance', fontsize=8.5)
        ax.set_title('Per 3-min block, net of chance (size = n)\nMann-Whitney p = {:.3g}'.format(lvd['p_block_toward']),
                     fontsize=8.5, loc='left')
    _style(ax)
    pdf.savefig(fig)
    plt.close(fig)


def _page_states(pdf, R_):
    """ Page 8: behavioral state segmentation. """

    s, st = R_['s'], R_['st']
    fig = _new_page('Behavioral states',
                    'Exclusive labels, priority rear > investigate > approach > wall-follow > turn > locomote > rest > other.  '
                    'Covariates and gates, not the result.')
    gs = fig.add_gridspec(3, 4, left=0.06, right=0.98, top=0.87, bottom=0.07, hspace=0.6, wspace=0.35)
    ax = fig.add_subplot(gs[0, :])
    cm = ListedColormap(list(STATE_COLORS))
    S = st['state'].astype(float)
    S[S < 0] = np.nan
    nb = 2000
    step = max(1, len(S) // nb)
    ax.imshow(S[::step][None, :], aspect='auto', cmap=cm, vmin=-0.5, vmax=len(STATE_NAMES) - 0.5,
              extent=[0, s['t'][-1] / 60, 0, 1], interpolation='nearest')
    L = s['light'][::step].astype(float)
    ax.imshow(L[None, :], aspect='auto', cmap=ListedColormap(['#52514e', '#fde7c2']), vmin=0, vmax=1,
              extent=[0, s['t'][-1] / 60, 1.05, 1.3], interpolation='nearest')
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0.5, 1.175])
    ax.set_yticklabels(['state', 'light'], fontsize=8)
    ax.set_xlabel('minutes', fontsize=8.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STATE_COLORS]
    ax.legend(handles, STATE_NAMES, ncol=9, fontsize=7.5, frameon=False, loc='upper center',
              bbox_to_anchor=(0.5, -0.35))
    ax.set_title('Ethogram', fontsize=9.5, loc='left')
    _style(ax, grid_y=False)

    ax = fig.add_subplot(gs[1:, 0:2])
    c = np.vstack([s['arena'], s['arena'][:1]])
    ax.plot(c[:, 0] / 10, c[:, 1] / 10, color=INK, lw=1)
    rng = np.random.default_rng(0)
    for k in range(len(STATE_NAMES)):
        idx = np.flatnonzero(st['state'] == k)
        if not len(idx):
            continue
        idx = rng.choice(idx, min(2500, len(idx)), replace=False)
        ax.scatter(st['hx_s'][idx] / 10, st['hy_s'][idx] / 10, s=2, color=STATE_COLORS[k], alpha=0.5)
    ax.add_patch(plt.Circle(s['pillar_xy'] / 10, s['pillar_r'] / 10, color=INK))
    ax.set_aspect('equal')
    ax.set_xlabel('x (cm)', fontsize=8.5)
    ax.set_ylabel('y (cm)', fontsize=8.5)
    ax.set_title('Head position by state (all epochs); black disc = pillar', fontsize=9, loc='left')
    _style(ax, grid_y=False)

    ax = fig.add_subplot(gs[1, 2:])
    T = R_['cov']
    x = np.arange(len(STATE_NAMES))
    ax.bar(x - 0.2, T['light']['frac_time'], width=0.38, color='#2a78d6', label='light')
    ax.bar(x + 0.2, T['dark']['frac_time'], width=0.38, color='#52514e', label='dark')
    ax.set_xticks(x)
    ax.set_xticklabels(STATE_NAMES, fontsize=7.5, rotation=30, ha='right')
    ax.set_ylabel('% of time', fontsize=8.5)
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title('Time in state', fontsize=9, loc='left')
    _style(ax)

    ax = fig.add_subplot(gs[2, 2])
    nu, es = st['nose_up'], s['ear_sep']
    bins = np.arange(-40, 46, 5)
    mid = (bins[:-1] + bins[1:]) / 2
    med = [np.nanmedian(es[(nu >= a) & (nu < b)]) if np.sum((nu >= a) & (nu < b)) > 50 else np.nan
           for a, b in zip(bins[:-1], bins[1:])]
    ax.plot(mid, med, 'o-', color='#2a78d6', ms=3)
    ax.axvline(st['params']['rear_pitch_deg'], color='#eb6834', lw=1, ls='--')
    ax.set_xlabel('nose-up pitch (deg)', fontsize=8.5)
    ax.set_ylabel('ear separation (px)', fontsize=8.5)
    ax.set_title('Rear check: head nearer camera', fontsize=8.5, loc='left')
    _style(ax)

    ax = fig.add_subplot(gs[2, 3])
    data, lbl = [], []
    for k, nm in enumerate(STATE_NAMES):
        r = _runs(st['state'] == k)
        if len(r):
            data.append((r[:, 1] - r[:, 0]) * DT)
            lbl.append(nm)
    ax.boxplot(data, vert=True, showfliers=False, widths=0.6,
               medianprops={'color': '#eb6834'}, boxprops={'color': INK2},
               whiskerprops={'color': INK2}, capprops={'color': INK2})
    ax.set_yscale('log')
    ax.set_xticks(range(1, len(lbl) + 1))
    ax.set_xticklabels(lbl, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('bout duration (s)', fontsize=8.5)
    ax.set_title('Bout durations', fontsize=8.5, loc='left')
    _style(ax)
    pdf.savefig(fig)
    plt.close(fig)


def _page_covariates(pdf, R_):
    """ Page 9: per-state covariate tables, light and dark. """

    T = R_['cov']
    fig = _new_page('States as covariates',
                    'Per state: gaze-shift rate and motif mix, compensation, retinal flow, pillar geometry.  '
                    'Cell colour = z-score across states within a row; blank = < 5 s in state or < 5 events.')
    gs = fig.add_gridspec(1, 2, left=0.13, right=0.99, top=0.86, bottom=0.12, wspace=0.42)
    for j, cond in enumerate(('light', 'dark')):
        ax = fig.add_subplot(gs[0, j])
        M = np.array([T[cond][m] for m in T['metrics']])
        Z = (M - np.nanmean(M, 1, keepdims=True)) / np.where(np.nanstd(M, 1, keepdims=True) > 0,
                                                               np.nanstd(M, 1, keepdims=True), 1)
        ax.imshow(np.nan_to_num(Z), cmap='RdBu_r', vmin=-2.5, vmax=2.5, aspect='auto')
        for i in range(M.shape[0]):
            for k in range(M.shape[1]):
                if np.isfinite(M[i, k]):
                    val = M[i, k]
                    fmt = '{:.2f}' if abs(val) < 10 else '{:.0f}'
                    ax.text(k, i, fmt.format(val), ha='center', va='center', fontsize=6.5, color=INK)
        ax.set_xticks(range(len(STATE_NAMES)))
        ax.set_xticklabels(STATE_NAMES, fontsize=7.5, rotation=40, ha='right')
        ax.set_yticks(range(len(T['labels'])))
        ax.set_yticklabels(T['labels'] if j == 0 else [''] * len(T['labels']), fontsize=7.5)
        ax.set_title(cond, fontsize=10, loc='left')
    pdf.savefig(fig)
    plt.close(fig)


def _page_methods(pdf, R_):
    """ Page 10: parameters and caveats. """

    fig = _new_page('Methods, parameters and caveats')
    st = R_['st']['params']
    txt = [
        'TIME BASE  Everything on a 25 ms (40 Hz) grid.  Eye camera {:.1f} Hz, IMU ~37 Hz, top-down 7.5 Hz '
        '(head position is the reconstruction\'s arc-interpolated track, so translational quantities are '
        'smooth below ~4 Hz).'.format(R_['s']['native_fps']),
        'POSE  From {} (eye-free retinocentric render), passed through the renderer\'s rotation chain with '
        'tilt_h = -pfh and tilt_v = -phi.  Re-rendered frames match the stored images exactly.'.format(
            os.path.basename(R_['npz_path'])),
        'PILLAR  Bearing of the pillar axis at eye height.  Width = 2 asin(r/d); height = floor-to-top '
        'angle from the eye.  Zones in the cyclopean frame: binocular |az| <= {:.0f}, monocular to {:.0f}, '
        'blind beyond.  Conjugate horizontal eye movement is assumed for the untracked left eye; vertical '
        'eye position does not enter the zone.'.format(BINO_HALF_DEG, FIELD_EDGE_DEG),
        'OPTIC FLOW  d_dot = -omega x d - (v - (v.d)d)/rho on a 15x15 grid over the 120 deg reconstructed '
        'field; rho by ray casting the arena (floor, 4 walls of {:.0f} mm, pillar).  omega and v by central '
        'differences of the optical-frame pose.  Eye-frozen counterfactual = resting pupil.'.format(R_['s']['wall_h']),
        'CLASSIFICATION  Hou et al. 2026 Methods 4.3 run as written (random projection to 4-D, k-means k=3, '
        'previous-frame rule) is reported on page 4 but not used downstream: with an eye-on-head gain of ~{:.2f} '
        'its boundary falls on head speed.  Downstream frames use the instantaneous split Hou et al. say their '
        'classes reproduce (Parker et al. 2023): gaze shift |gaze| > {:.0f} deg/s (no head requirement), '
        'compensatory |head| > {:.0f} and |gaze| < {:.0f}, then the previous-frame rule.  Motifs then follow '
        'Hou et al. exactly (+-100 ms head/eye velocity+acceleration, z-score, k-means k=4 per direction, '
        'Hungarian matching across directions, 100 ms neighbour exclusion, run onsets).  Not specified there '
        'and chosen here: SG window {} samples, order {}; seeds 0.  Motif naming is by rule (page 5).'.format(
            abs(R_['v']['vor_slope']), SHIFTED_GAZE, SHIFTED_HEAD, STILL_GAZE, SG_WIN, SG_ORD),
        'FLOW CHECKS  (a) analytic flow vs the finite difference of each grid point re-projected at t+-1; '
        '(b) every pixel of a rendered frame warped onto the frame {:.0f} ms later with pose + ray-cast depth, '
        'error relative to no warp; the eye-held variant keeps the head motion but not the eye\'s.'.format(
            R_['ifc'].get('gap_ms', np.nan)),
        'LANDING  Pillar cyclopean azimuth 50 ms before episode start vs 50 ms after episode end.  '
        'az_post = az_pre + d(bearing) - d(gaze), which the shift-shuffle null uses to give each start '
        'another event\'s gaze vector while keeping its own translation.  {} shuffles; p = (1+#null>=obs)/(1+N).  '
        'Breakdown p-values are uncorrected.  Light vs dark compares NET effects (toward rate and '
        'mean eccentricity change minus each condition\'s own shift-shuffle chance), with condition labels '
        'permuted within state and each pseudo-condition\'s chance recomputed; block test on each '
        'block\'s net effect.'.format(N_NULL),
        'STATES  smoothing {smooth_s} s; rear: nose-up > {rear_pitch_deg} deg for >= {rear_min_s} s, speed < '
        '{rear_max_speed} mm/s, "at pillar" if within {rear_pillar_mm} mm; investigate: head < {pillar_near_mm} mm '
        'from pillar surface, speed < {invest_max_speed} mm/s, pillar within +-{invest_bearing_deg} deg; approach: '
        'speed > {approach_min_speed} mm/s, closing > {approach_closing} mm/s, pillar within +-{approach_bearing_deg} deg, '
        '>= {approach_min_s} s, and must reach the pillar within {approach_reach_s} s; wall-follow: < {wall_near_mm} mm '
        'from a wall, speed > {wall_min_speed} mm/s, within {wall_angle_deg} deg of parallel, >= {wall_min_s} s; turn: '
        '|yaw rate| > {turn_rate} deg/s for >= {turn_min_s} s; locomote: speed > {loco_speed} mm/s; rest: speed < '
        '{rest_speed} mm/s and |yaw rate| < {rest_rate} deg/s for >= {rest_min_s} s.'.format(**st),
        'CAVEATS  (1) One session, one animal: every p-value here is within-session.  (2) Only the right eye is '
        'tracked; the binocular zone assumes conjugacy.  (3) Rearing is from IMU pitch, not head height '
        '(calc_height.py is not validated).  (4) Head position is 7.5 Hz; translational flow and approach '
        'kinematics inherit that.  (5) In the dark the pillar is invisible, so a dark-epoch effect points to '
        'non-visual (spatial memory, whisking) or behavioural confounds rather than visual targeting.  '
        '(6) Gaze shifts are mostly head-driven; with an eye/head gain ~{:.2f} the eye contributes little to '
        'the landing.'.format(abs(R_['v']['vor_slope'])),
    ]
    y = 0.88
    import textwrap
    for para in txt:
        w = textwrap.fill(para, 150)
        fig.text(0.04, y, w, fontsize=8.8, va='top', color=INK)
        y -= 0.022 * (w.count('\n') + 1) + 0.02
    pdf.savefig(fig)
    plt.close(fig)


# Neural analyses.  Everything below runs on the 2P frame clock, with the
# covariates above binned onto it.  The fitting machinery (contiguous-block
# folds with an embargo, nested per-cell ridge, circular-shift null, triangle
# smoothing of the target) is ffNLE_scene's, so R^2 values here sit on the same
# scale as the scene encoding model.

SPIKE_KEY = 'norm_spikes'
SMOOTH_S = 1.2
N_VM = 8
VM_KAPPA = 2.0
BLOCK_S = 120.0
EMBARGO_S = 40.0
N_FOLDS = 5
GROUPS = ('retinal_pillar', 'retinal_flow', 'eye', 'head', 'speed', 'pupil',
          'ego_pillar', 'allo', 'state')
GROUP_LABELS = {'retinal_pillar': 'retinal pillar', 'retinal_flow': 'retinal flow',
                'eye': 'eye pos/vel', 'head': 'head ang/vel', 'speed': 'speed',
                'pupil': 'pupil', 'ego_pillar': 'ego pillar', 'allo': 'allo pos/hd',
                'state': 'state'}
# Labels that are not areas: kept in session-wide analyses, skipped per area.
NON_AREAS = ('border', 'none', 'unassigned')
AREA_LABELS = {0: 'none', 2: 'RL', 3: 'AM', 4: 'PM', 5: 'V1', 7: 'AL', 8: 'LM',
               9: 'P', 10: 'A'}
AREA_COLORS = {'V1': '#2a78d6', 'AM': '#eb6834', 'A': '#1baf7a', 'RL': '#6250d6',
               'PM': '#e87ba4', 'AL': '#eda100', 'LM': '#008300', 'P': '#e34948',
               'unassigned': '#b9b8b2'}
BETA_GRID = np.round(np.arange(-1.0, 2.01, 0.1), 2)


def _vm_basis(ang_deg, n=N_VM, kappa=VM_KAPPA):
    """ von Mises bumps tiling the circle, (N, n). """

    c = np.radians(np.arange(n) * 360.0 / n)
    a = np.radians(np.asarray(ang_deg, float))[:, None] - c[None, :]
    return np.exp(kappa * (np.cos(a) - 1.0))


def _gauss_basis(x, centers, width):
    """ Gaussian bumps at the given centers, (N, len(centers)). """

    x = np.asarray(x, float)[:, None]
    return np.exp(-0.5 * ((x - np.asarray(centers, float)[None, :]) / float(width)) ** 2)


def _twop_edges(twopT):
    """ Bin edges around each 2P frame time. """

    d = np.diff(twopT)
    return np.r_[twopT[0] - d[0] / 2, twopT[:-1] + d / 2, twopT[-1] + d[-1] / 2]


def _bin_to_2p(t, x, twopT, circular=False):
    """ Mean of a fast trace inside each 2P frame; circular mean for angles (deg).

    Returns
    -------
    out : (n_2p,) or (n_2p, k)
    count : (n_2p,)
        Finite samples that went into each bin.
    """

    edges = _twop_edges(twopT)
    n = len(twopT)
    idx = np.searchsorted(edges, t, side='right') - 1
    inb = (idx >= 0) & (idx < n)
    x = np.asarray(x, float)

    def mean1(v):
        f = inb & np.isfinite(v)
        s = np.bincount(idx[f], weights=v[f], minlength=n)
        c = np.bincount(idx[f], minlength=n).astype(float)
        return s / np.where(c > 0, c, np.nan), c

    if x.ndim == 2:
        cols = [mean1(x[:, j]) for j in range(x.shape[1])]
        return np.stack([c[0] for c in cols], 1), cols[0][1]
    if circular:
        r = np.radians(x)
        cm, cnt = mean1(np.cos(r))
        sm, _ = mean1(np.sin(r))
        return np.degrees(np.arctan2(sm, cm)), cnt
    return mean1(x)


def load_area_labels(h5_path, n_cells, composites_dir=None):
    """ Visual area per cell, from area_labels.assign_areas (widefield registration).

    Replaces the stored visual_area_id of <animal>_merged_essentials_v*.h5,
    which is in string cell-key order (row i is not cell i) and keyed by
    position rather than recording -- see fm2p/area_labels.py.  Falls back to
    'unassigned' with a warning, so a session without registration still runs.

    Returns
    -------
    A : dict
        'area' (C,) analysis label (atlas area, 'border' within BORDER_PX of
        another area, 'none' outside the atlas, 'unassigned' on failure);
        'area_atlas', 'area_id', 'border_dist', 'wf_pos', 'vfs_pos',
        'own_sign', 'source' (str), 'info' (dict).
    """

    from .area_labels import assign_areas

    A = {'area': np.array(['unassigned'] * n_cells, dtype=object),
         'area_atlas': np.array(['unassigned'] * n_cells, dtype=object),
         'area_id': np.zeros(n_cells, int), 'border_dist': np.full(n_cells, np.nan),
         'wf_pos': np.full((n_cells, 2), np.nan), 'vfs_pos': np.full((n_cells, 2), np.nan),
         'own_sign': np.full(n_cells, np.nan), 'source': '', 'info': {}}
    try:
        out = assign_areas(h5_path, composites_dir=composites_dir)
    except Exception as e:
        print('  WARNING: no area labels for {} ({}) -- every cell is "unassigned".'.format(
            os.path.basename(h5_path), e))
        return A
    if len(out['area']) != n_cells:
        print('  WARNING: {} cells in cell_x_pix, {} in the spike array -- areas left unassigned.'.format(
            len(out['area']), n_cells))
        return A
    A.update({k: out[k] for k in ('area', 'area_atlas', 'area_id', 'border_dist', 'wf_pos', 'vfs_pos',
                                  'own_sign', 'info')})
    A['source'] = '{} + {}'.format(out['info']['tile_file'], out['info']['registration_file'])
    u, c = np.unique(A['area'], return_counts=True)
    print('  Areas (atlas, border < {:.0f} px): {}.'.format(out['info']['border_px'],
                                                         ', '.join('{} {}'.format(a, n) for a, n in zip(u, c))))
    return A


def build_neural_dataset(h5_path, s, p, fl, v, st, spike_key=SPIKE_KEY, smooth_s=SMOOTH_S):
    """ Neural target and every covariate group, on the 2P frame clock.

    Returns
    -------
    D : dict
        't', 'Y' (n_2p, C) triangle-smoothed rate, 'Y_raw', 'use', 'light',
        'X' (n_2p, F), 'names', 'groups' {group: column indices}, 'base' (always-on
        columns), and the raw angles the reference-frame analyses need
        ('rho' retinal/eye-centred pillar az, 'eta' head-centred, 'e' horizontal
        eye-in-head, 'hb' head-on-body, 'allo_dir', 'dist_cm', 'heading', ...).
    """

    from .ffNLE_scene import _triangle_smooth

    with h5py.File(h5_path, 'r') as f:
        twopT = np.asarray(f['twopT'][()], float)
        Y_raw = np.asarray(f[spike_key][()], float).T
        light = np.asarray(f['ltdk_state_vec'][()], bool) if 'ltdk_state_vec' in f else np.ones(len(twopT), bool)
        i0 = int(f['eyeT_startInd'][()]) if 'eyeT_startInd' in f else 0
        i1 = int(f['eyeT_endInd'][()]) if 'eyeT_endInd' in f else None
        eyeT = np.asarray(f['eyeT_trim'][()], float)
        la = np.asarray(f['longaxis'][()], float)[i0:i1]
        sa = np.asarray(f['shortaxis'][()], float)[i0:i1]
    n = min(len(twopT), Y_raw.shape[0], len(light))
    twopT, Y_raw, light = twopT[:n], Y_raw[:n], light[:n]
    dt = float(np.median(np.diff(twopT)))
    t = s['t']

    def B(x, circ=False):
        return _bin_to_2p(t, x, twopT, circular=circ)[0]

    # 40 Hz derived quantities.
    phi_dev = s['phi'] - s['mean_phi']
    eye_v_vert = np.gradient(_sg_nan(s['phi']), DT)
    pitch_v = np.gradient(_sg_nan(s['pitch']), DT)
    roll_v = np.gradient(_sg_nan(s['roll']), DT)
    vx, vy = np.gradient(st['hx_s'], DT), np.gradient(st['hy_s'], DT)
    move_dir = np.degrees(np.arctan2(vy, vx))
    heading = _wrap(s['yaw_r'])
    hb = _wrap(heading - move_dir)
    # Travel direction from 7.5 Hz positions is noise below ~8 cm/s (median
    # |hb| 28 deg at > 5 cm/s vs 25 at > 8), and |hb| > 90 deg is a tracking
    # failure, not a head turned backwards on the body.
    hb[(st['speed'] < 8.0) | (np.abs(hb) > 90.0)] = np.nan
    px, py = s['pillar_xy']
    allo_dir = np.degrees(np.arctan2(py - st['hy_s'], px - st['hx_s']))
    pose_frac, _ = _bin_to_2p(t, s['pose_ok'].astype(float), twopT)
    eye_frac, _ = _bin_to_2p(t, s['eye_ok'].astype(float), twopT)

    pup = np.pi * la * sa
    m = min(len(eyeT), len(pup))
    pupil, _ = _bin_to_2p(eyeT[:m], pup[:m], twopT)
    pupil_z = (pupil - np.nanmean(pupil)) / np.nanstd(pupil)

    D = {'t': twopT, 'dt': dt, 'light': light, 'Y_raw': Y_raw}
    D['rho'] = B(p['az_cyc'], True)
    D['eta'] = B(p['az_head'], True)
    D['e'] = B(p['eye_dev'])
    D['az_eye'] = B(p['az_eye'], True)
    D['el_eye'] = B(p['el_eye'])
    D['heading'] = B(heading, True)
    D['hb'] = B(hb, True)
    D['allo_dir'] = B(allo_dir, True)
    D['ego_dir'] = B(st['ego_pillar'], True)
    D['dist_cm'] = B(st['d_pillar']) / 10.0
    D['speed'] = B(st['speed'])
    D['hx'], D['hy'] = B(st['hx_s']), B(st['hy_s'])
    D['pitch'], D['roll'] = B(s['pitch']), B(s['roll'])
    # State at the frame time (labels cannot be averaged).
    j = np.clip(np.searchsorted(t, twopT), 0, len(t) - 1)
    D['state'] = st['state'][j]

    cols, names, groups = [], [], {}

    def add(group, arr, nm):
        arr = np.asarray(arr, float)
        arr = arr[:, None] if arr.ndim == 1 else arr
        k0 = sum(c.shape[1] for c in cols)
        cols.append(arr)
        names.extend(nm if isinstance(nm, list) else [nm])
        groups.setdefault(group, []).extend(range(k0, k0 + arr.shape[1]))

    # Retinal pillar: where it falls on the retina, how big, how fast it slips.
    add('retinal_pillar', _vm_basis(D['az_eye']), ['ret_az_vm{}'.format(i) for i in range(N_VM)])
    add('retinal_pillar', _gauss_basis(D['el_eye'], [-60, -20, 20, 60], 25.0),
        ['ret_el_g{}'.format(i) for i in range(4)])
    add('retinal_pillar', np.log(B(p['width_deg'])), 'log_width')
    add('retinal_pillar', B(p['in_recon'].astype(float)), 'in_view')
    add('retinal_pillar', np.log1p(B(p['slip'])), 'log_slip')

    # Retinal flow: magnitude by component, plus the field averaged into 3x3
    # sectors (az, el components each).
    add('retinal_flow', np.log1p(np.stack([B(fl['total']), B(fl['rot']), B(fl['trans'])], 1)),
        ['log_flow_total', 'log_flow_rot', 'log_flow_trans'])
    g = int(round(np.sqrt(fl['field'].shape[1])))
    sect = np.minimum(np.arange(g) * 3 // g, 2)
    sec_id = (sect[:, None] * 3 + sect[None, :]).ravel()
    sec = np.stack([np.nanmean(fl['field'][:, sec_id == k, c], axis=1)
                    for k in range(9) for c in range(2)], 1)
    add('retinal_flow', _bin_to_2p(t, sec, twopT)[0],
        ['flow_s{}_{}'.format(k, 'az' if c == 0 else 'el') for k in range(9) for c in range(2)])

    # Eye position and velocity.
    e = D['e']
    eq = np.nanpercentile(e, [10, 37, 63, 90])
    add('eye', np.stack([e, B(phi_dev), B(v['eye']), np.abs(B(v['eye'])), B(eye_v_vert)], 1),
        ['eye_h', 'eye_v', 'eye_vel_h', 'eye_speed_h', 'eye_vel_v'])
    add('eye', _gauss_basis(e, eq, max(np.nanstd(e) / 1.5, 1.0)), ['eye_h_g{}'.format(i) for i in range(4)])

    # Head angles and velocities.
    hv = B(v['head'])
    add('head', np.stack([D['pitch'], D['roll'], D['pitch'] ** 2, D['roll'] ** 2, hv, np.abs(hv),
                          B(pitch_v), B(roll_v)], 1),
        ['pitch', 'roll', 'pitch2', 'roll2', 'yaw_vel', 'yaw_speed', 'pitch_vel', 'roll_vel'])

    sp = D['speed']
    add('speed', np.stack([sp, np.log1p(np.clip(sp, 0, None)), np.gradient(sp, dt)], 1),
        ['speed', 'log_speed', 'accel'])

    add('pupil', np.stack([pupil_z, np.gradient(pupil_z, dt)], 1), ['pupil', 'pupil_vel'])

    # Egocentric pillar: bearing from the head and distance.
    ld = np.log(np.clip(D['dist_cm'], 0.5, None))
    lq = np.nanpercentile(ld, [5, 35, 65, 95])
    add('ego_pillar', _vm_basis(D['ego_dir']), ['ego_vm{}'.format(i) for i in range(N_VM)])
    add('ego_pillar', ld, 'log_dist')
    add('ego_pillar', _gauss_basis(ld, lq, (lq[-1] - lq[0]) / 4), ['dist_g{}'.format(i) for i in range(4)])

    # Allocentric position (5x5 grid over the arena) and heading.
    ax = np.linspace(s['arena'][:, 0].min(), s['arena'][:, 0].max(), 7)[1:-1]
    ay = np.linspace(s['arena'][:, 1].min(), s['arena'][:, 1].max(), 7)[1:-1]
    wpos = 0.75 * (ax[1] - ax[0])
    pos = np.stack([np.exp(-0.5 * (((D['hx'] - cx) ** 2 + (D['hy'] - cy) ** 2) / wpos ** 2))
                    for cx in ax for cy in ay], 1)
    add('allo', pos, ['pos{}'.format(i) for i in range(pos.shape[1])])
    add('allo', _vm_basis(D['heading']), ['hd_vm{}'.format(i) for i in range(N_VM)])

    add('state', np.stack([(D['state'] == k).astype(float) for k in range(len(STATE_NAMES) - 1)], 1),
        ['state_{}'.format(nm) for nm in STATE_NAMES[:-1]])

    # Always-on nuisance: light, and slow drift (cosines with periods >= T/2.5).
    T = twopT[-1] - twopT[0]
    drift = np.stack([np.cos(np.pi * k * (twopT - twopT[0]) / T) for k in range(1, 6)], 1)
    add('base', np.c_[light.astype(float), drift], ['light'] + ['drift{}'.format(k) for k in range(1, 6)])

    X = np.concatenate(cols, 1)
    Yr = np.where(np.isfinite(Y_raw), Y_raw, 0.0)
    D['Y'] = _triangle_smooth(Yr, smooth_s, dt)
    D['X'] = X
    D['names'] = names
    D['groups'] = {k: np.asarray(v_, int) for k, v_ in groups.items()}
    D['use'] = (np.all(np.isfinite(X), 1) & np.all(np.isfinite(Y_raw), 1)
                & (pose_frac > 0.5) & (eye_frac > 0.5))
    D['spike_key'] = spike_key
    print('  Neural dataset: {} cells x {} frames ({:.1f} Hz); {} covariates in {} groups; '
          '{:.1f}% of frames usable ({:.1f}% of those in the light).'.format(
              Y_raw.shape[1], n, 1 / dt, X.shape[1], len(GROUPS), 100 * D['use'].mean(),
              100 * light[D['use']].mean()))
    return D


def _folds(D, use):
    """ ffNLE_scene's block folds plus the inner-split block index. """

    from .ffNLE_scene import block_folds

    _, trm, tem = block_folds(D['t'], use, n_folds=N_FOLDS, block_s=BLOCK_S, embargo_s=EMBARGO_S)
    block = np.floor((D['t'] - D['t'][0]) / BLOCK_S).astype(int)
    return trm, tem, block


def _zbuilder(X):
    """ build_X for cv_ridge: z-score on the training rows, zero the rest. """

    def build(tr):
        mu = np.nanmean(X[tr], 0)
        sd = np.nanstd(X[tr], 0)
        sd[~(sd > 0)] = 1.0
        return np.nan_to_num((X - mu) / sd)
    return build


def _cv(X, Y, folds, n_shuffles=0, seed=0):
    """ One cross-validated ridge fit through ffNLE_scene.cv_ridge. """

    from .ffNLE_scene import cv_ridge

    trm, tem, block = folds
    return cv_ridge(_zbuilder(X), Y, trm, tem, block, n_shuffles=n_shuffles,
                    min_shift=400, seed=seed)


def encoding_models(D, n_shuffles=20, seed=0):
    """ Full model, leave-one-group-out and group-alone fits; unique variance.

    Unique variance of a group = R^2(full) - R^2(full without that group), both
    cross-validated on the same folds.  A cell carries a group uniquely when the
    drop is positive in at least 4 of 5 folds.

    Returns
    -------
    enc : dict
        'r2_full', 'p_full', 'null_full', 'unique' (C, G), 'alone' (C, G),
        'unique_folds_pos' (C, G), 'sig' (C,) bool.
    """

    folds = _folds(D, D['use'])
    base = D['groups']['base']
    allc = np.concatenate([D['groups'][g] for g in GROUPS] + [base])
    print('  Full model ({} covariates, {} shuffles)...'.format(len(allc), n_shuffles))
    full = _cv(D['X'][:, allc], D['Y'], folds, n_shuffles=n_shuffles, seed=seed)
    C = D['Y'].shape[1]
    uniq = np.full((C, len(GROUPS)), np.nan)
    alone = np.full((C, len(GROUPS)), np.nan)
    upos = np.zeros((C, len(GROUPS)), int)
    for gi, gname in enumerate(GROUPS):
        keep = np.concatenate([D['groups'][g] for g in GROUPS if g != gname] + [base])
        red = _cv(D['X'][:, keep], D['Y'], folds)
        uniq[:, gi] = full['r2'] - red['r2']
        upos[:, gi] = np.sum((full['r2_folds'] - red['r2_folds']) > 0, axis=0)
        alo = _cv(D['X'][:, np.concatenate([D['groups'][gname], base])], D['Y'], folds)
        alone[:, gi] = alo['r2']
    base_only = _cv(D['X'][:, base], D['Y'], folds)
    sig = (full['p'] < 0.05) & (full['r2'] > base_only['r2'])
    enc = {'r2_full': full['r2'], 'r2_folds': full['r2_folds'], 'p_full': full['p'],
           'null_full': full['null_r2_median'], 'r2_base': base_only['r2'],
           'unique': uniq, 'alone': alone, 'unique_folds_pos': upos, 'sig': sig,
           'groups': list(GROUPS)}
    print('  Encoding: {} / {} cells beat the shuffle and the nuisance-only model; median full '
          'R2 {:.4f} (null {:.4f}).'.format(int(sig.sum()), C, np.nanmedian(full['r2']),
                                            np.nanmedian(full['null_r2_median'])))
    for gi, gname in enumerate(GROUPS):
        k = int(np.sum(sig & (upos[:, gi] >= 4) & (uniq[:, gi] > 0)))
        print('    {:15s} unique: median {:+.4f} (sig cells), {} cells with a consistent '
              'unique share'.format(gname, np.nanmedian(uniq[sig, gi]) if sig.any() else np.nan, k))
    return enc


def _circ_smooth(M, sigma_bins):
    """ Circular Gaussian smoothing along axis 0 of (nb, C), via FFT. """

    nb = M.shape[0]
    k = np.arange(nb)
    k = np.minimum(k, nb - k)
    ker = np.exp(-0.5 * (k / sigma_bins) ** 2)
    ker /= ker.sum()
    return np.real(np.fft.ifft(np.fft.fft(M, axis=0) * np.fft.fft(ker)[:, None], axis=0))


def _circ_fields(ang, Y, w, dt, nbins=36, sigma_deg=20.0, min_occ_s=2.0):
    """ Occupancy-normalized circular response fields for every cell.

    Numerator and occupancy are smoothed separately and divided, so empty bins
    neither spread NaN nor drag the field to zero.  Bins with less than
    min_occ_s of smoothed weighted occupancy are NaN.

    Returns (nbins, C).
    """

    ok = np.isfinite(ang) & (w > 0)
    b = (np.floor(((np.asarray(ang)[ok] + 180.0) % 360.0) / (360.0 / nbins))).astype(int) % nbins
    W = np.zeros((nbins,))
    np.add.at(W, b, w[ok])
    num = np.zeros((nbins, Y.shape[1]))
    np.add.at(num, b, w[ok][:, None] * Y[ok])
    sb = sigma_deg / (360.0 / nbins)
    Ws = _circ_smooth(W[:, None], sb)[:, 0]
    F = _circ_smooth(num, sb) / np.where(Ws > 0, Ws, np.nan)[:, None]
    F[Ws * dt < min_occ_s] = np.nan
    return F


def _best_shift(Fa, Fb, max_shift=90):
    """ Shift s (deg) maximizing corr(Fb(x), Fa(x - s)), per cell.

    Fields are upsampled to 1 deg on the circle.  Returns (shift, r_at_best).
    """

    nb, C = Fa.shape
    x = np.arange(nb) * 360.0 / nb
    xx = np.arange(360.0)

    def up(F):
        out = np.full((360, C), np.nan)
        for c in range(C):
            f = F[:, c]
            ok = np.isfinite(f)
            if ok.sum() < nb // 3:
                continue
            xs, fs = x[ok], f[ok]
            out[:, c] = np.interp(xx, np.r_[xs - 360, xs, xs + 360], np.r_[fs, fs, fs])
            # Do not invent field across long unsampled gaps.
            gap = np.interp(xx, np.r_[x - 360, x, x + 360], np.r_[~ok, ~ok, ~ok].astype(float))
            out[gap > 0.5, c] = np.nan
        return out

    A, Bm = up(Fa), up(Fb)
    shifts = np.arange(-max_shift, max_shift + 1)
    R = np.full((len(shifts), C), np.nan)
    for i, sft in enumerate(shifts):
        As = np.roll(A, sft, axis=0)
        m = np.isfinite(As) & np.isfinite(Bm)
        n = m.sum(0)
        a = np.where(m, As, 0.0)
        bb = np.where(m, Bm, 0.0)
        ma = a.sum(0) / np.maximum(n, 1)
        mb = bb.sum(0) / np.maximum(n, 1)
        da = np.where(m, As - ma, 0.0)
        db = np.where(m, Bm - mb, 0.0)
        den = np.sqrt((da ** 2).sum(0) * (db ** 2).sum(0))
        r = (da * db).sum(0) / np.where(den > 0, den, np.nan)
        r[n < 120] = np.nan
        R[i] = r
    best = np.full(C, np.nan)
    rbest = np.full(C, np.nan)
    ok = np.any(np.isfinite(R), 0)
    if ok.any():
        k = np.nanargmax(np.where(np.isfinite(R[:, ok]), R[:, ok], -np.inf), axis=0)
        best[ok] = shifts[k]
        rbest[ok] = R[k, np.flatnonzero(ok)]
    return best, rbest


def _match_weights(groups, strata):
    """ Importance weights making each group's stratum distribution the pooled one. """

    w = np.zeros(len(strata))
    pooled = np.bincount(strata[groups >= 0], minlength=strata.max() + 1).astype(float)
    pooled /= pooled.sum()
    for g in np.unique(groups[groups >= 0]):
        m = groups == g
        pg = np.bincount(strata[m], minlength=len(pooled)).astype(float)
        pg /= max(pg.sum(), 1)
        ratio = np.where(pg > 0, pooled / np.where(pg > 0, pg, 1), 0.0)
        w[m] = np.clip(ratio[strata[m]], 0, 5)
    return w


def frame_shift_analysis(D, ang_key, mod_key, mask, strata, n_boot=100, n_null=50,
                         seed=0, label='', min_occ_s=2.0):
    """ Field shift between low and high values of a modulating angle.

    Fields are estimated over `ang_key` (e.g. retinal pillar azimuth rho) in the
    lowest and highest terciles of `mod_key` (e.g. eye-in-head e), with
    importance weights matching the distribution of `strata` (head angle) across
    the two, i.e. head angle held fixed.  If the cell's field is fixed in
    ang_key coordinates the shift is 0 (ratio 1); if it is fixed in
    ang_key + mod_key coordinates the field moves by -delta (ratio 0).

        ratio = 1 + shift / delta_mod

    Returns
    -------
    out : dict with 'ratio', 'ci' (C, 2), 'shift', 'r_shift', 'delta', 'F_low',
          'F_high', 'F_all', 'split_r', 'p_field', 'n_frames'.
    """

    rng = np.random.default_rng(seed)
    a, mod = D[ang_key], D[mod_key]
    m = mask & np.isfinite(a) & np.isfinite(mod)
    Y = D['Y']
    dt = D['dt']
    q = np.nanpercentile(mod[m], [100 / 3, 200 / 3])
    grp = np.full(len(a), -1)
    grp[m & (mod <= q[0])] = 0
    grp[m & (mod >= q[1])] = 1
    w = _match_weights(grp, strata)
    delta = (np.average(mod[grp == 1], weights=np.maximum(w[grp == 1], 1e-9))
             - np.average(mod[grp == 0], weights=np.maximum(w[grp == 0], 1e-9)))
    lo, hi = grp == 0, grp == 1
    F_lo = _circ_fields(np.where(lo, a, np.nan), Y, w, dt, min_occ_s=min_occ_s)
    F_hi = _circ_fields(np.where(hi, a, np.nan), Y, w, dt, min_occ_s=min_occ_s)
    F_all = _circ_fields(np.where(m, a, np.nan), Y, m.astype(float), dt, min_occ_s=min_occ_s)
    shift, rsh = _best_shift(F_lo, F_hi)
    ratio = 1.0 + shift / delta

    # Field reliability: alternate 60 s blocks.
    blk = np.floor((D['t'] - D['t'][0]) / 60.0).astype(int)
    odd = (blk % 2).astype(bool)
    Fo = _circ_fields(np.where(m & odd, a, np.nan), Y, np.ones(len(a)), dt, min_occ_s=min_occ_s)
    Fe = _circ_fields(np.where(m & ~odd, a, np.nan), Y, np.ones(len(a)), dt, min_occ_s=min_occ_s)
    split_r = np.array([_nancorr(Fo[:, c], Fe[:, c]) for c in range(Y.shape[1])])

    # Field modulation vs circular shifts of the activity (same autocorrelation).
    def depth(F):
        return (np.nanmax(F, 0) - np.nanmin(F, 0)) / np.maximum(np.nanmean(F, 0), 1e-9)
    obs_depth = depth(F_all)
    null = np.zeros((n_null, Y.shape[1]))
    n = len(a)
    for k in range(n_null):
        sh = int(rng.integers(400, n - 400))
        null[k] = depth(_circ_fields(np.where(m, a, np.nan), np.roll(Y, sh, 0), m.astype(float), dt, min_occ_s=min_occ_s))
    p_field = (1 + np.sum(null >= obs_depth[None, :], 0)) / (1 + n_null)

    # Block bootstrap of the shift (60 s blocks resampled within each group).
    ub = np.unique(blk[m])
    boots = np.full((n_boot, Y.shape[1]), np.nan)
    for k in range(n_boot):
        pick = rng.choice(ub, len(ub), replace=True)
        cnt = np.bincount(pick, minlength=blk.max() + 1)
        wb = w * cnt[blk]
        Fl = _circ_fields(np.where(lo, a, np.nan), Y, wb, dt, min_occ_s=min_occ_s)
        Fh = _circ_fields(np.where(hi, a, np.nan), Y, wb, dt, min_occ_s=min_occ_s)
        boots[k] = 1.0 + _best_shift(Fl, Fh)[0] / delta
    ci = np.nanpercentile(boots, [2.5, 97.5], axis=0).T
    print('  {}: delta {:.1f} deg, {} frames; {} cells with a reliable field (split r > 0.5, '
          'p < 0.05).'.format(label, delta, int(m.sum()),
                              int(np.sum((split_r > 0.5) & (p_field < 0.05)))))
    return {'ratio': ratio, 'ci': ci, 'shift': shift, 'r_shift': rsh, 'delta': float(delta),
            'F_low': F_lo, 'F_high': F_hi, 'F_all': F_all, 'split_r': split_r,
            'p_field': p_field, 'depth': obs_depth, 'n_frames': int(m.sum()),
            'reliable': (split_r > 0.5) & (p_field < 0.05)}


def _nancorr(a, b):
    """ Pearson r over the jointly finite entries. """

    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def beta_scan(D, ang_key, mod_key, mask, nuis_cols, label=''):
    """ Cross-validated R^2 of a pillar field in coordinate ang + beta * mod.

    beta = 0: field fixed in ang_key coordinates; beta = 1: fixed in
    ang + mod.  The modulator itself and the nuisance columns (head angle) are
    in every model, so beta only decides WHERE the field lives.  Index = 1 -
    beta*, on the same scale as frame_shift_analysis' ratio.
    """

    use = mask & np.isfinite(D[ang_key]) & np.isfinite(D[mod_key])
    folds = _folds(D, use)
    modz = (D[mod_key] - np.nanmean(D[mod_key][use])) / np.nanstd(D[mod_key][use])
    nuis = np.c_[D['X'][:, nuis_cols], modz]
    nuis_fit = _cv(nuis, D['Y'], folds)
    R = np.full((len(BETA_GRID), D['Y'].shape[1]), np.nan)
    Rf = []
    for i, b in enumerate(BETA_GRID):
        x = _wrap(D[ang_key] + b * D[mod_key])
        fit = _cv(np.c_[_vm_basis(x), nuis], D['Y'], folds)
        R[i] = fit['r2']
        Rf.append(fit['r2_folds'])
    Rs = R.copy()
    Rs[1:-1] = (R[:-2] + R[1:-1] + R[2:]) / 3.0
    k = np.nanargmax(np.where(np.isfinite(Rs), Rs, -np.inf), axis=0)
    best = BETA_GRID[k]
    gain_over_nuis = np.nanmax(R, 0) - nuis_fit['r2']
    fpos = np.sum(np.stack(Rf)[k, :, np.arange(len(k))] - nuis_fit['r2_folds'].T > 0, axis=1)
    tuned = (gain_over_nuis > 0) & (fpos >= 4)
    # Per-cell frame call: which end of the scan wins in >= 4 of 5 folds.
    i0, i1 = int(np.argmin(np.abs(BETA_GRID - 0.0))), int(np.argmin(np.abs(BETA_GRID - 1.0)))
    d01 = Rf[i0] - Rf[i1]
    first = tuned & (np.sum(d01 > 0, 0) >= 4) & (R[i0] > R[i1])
    second = tuned & (np.sum(d01 < 0, 0) >= 4) & (R[i1] > R[i0])
    print('  {} beta scan: {} cells whose pillar field adds to the nuisance model in >= 4/5 '
          'folds; median index (1 - beta*) {:.2f}.'.format(label, int(tuned.sum()),
                                                         np.nanmedian(1 - best[tuned]) if tuned.any() else np.nan))
    print('    frame call (>= 4/5 folds): {} prefer beta=0 ({} frame), {} prefer beta=1 ({} + {}), '
          '{} ambiguous.'.format(int(first.sum()), ang_key, int(second.sum()), ang_key, mod_key,
                                 int((tuned & ~first & ~second).sum())))
    return {'beta': BETA_GRID, 'r2': R, 'beta_best': best, 'index': 1.0 - best,
            'r2_nuis': nuis_fit['r2'], 'gain_over_nuis': gain_over_nuis, 'tuned': tuned,
            'prefer_first': first, 'prefer_second': second, 'n_frames': int(use.sum())}


def gain_field_models(D, mask, nuis_cols, ang='rho', head_ang='eta', mod='e', label=''):
    """ Retinal vs additive vs gain-field vs head-centred pillar models per cell.

    RET  nuisance + vm(rho)
    ADD  RET + eye-position bumps                 (eye shifts the baseline)
    GAIN ADD + vm(rho) x eye_z                    (eye scales the retinal field)
    HEAD nuisance + vm(eta) + eye-position bumps  (field lives in head coords)

    A gain-modulated basis function looks like GAIN > ADD > HEAD with a retinal
    field; a readout looks like HEAD >= GAIN.  Also returns the GAIN model
    refit on every usable frame, so its coefficients can rebuild this cell's
    gain field on another session's covariates.
    """

    use = mask & np.isfinite(D[ang]) & np.isfinite(D[head_ang]) & np.isfinite(D[mod])
    folds = _folds(D, use)
    mu, sd = np.nanmean(D[mod][use]), np.nanstd(D[mod][use])
    mz = (D[mod] - mu) / sd
    eq = np.nanpercentile(D[mod][use], [10, 37, 63, 90])
    ew = max(sd / 1.5, 1.0)
    nuis = np.c_[D['X'][:, nuis_cols]]
    vr = _vm_basis(D[ang])
    vh = _vm_basis(D[head_ang])
    eb = _gauss_basis(D[mod], eq, ew)
    designs = {
        'NUIS': np.c_[nuis, mz],
        'RET': np.c_[nuis, mz, vr],
        'ADD': np.c_[nuis, mz, vr, eb],
        'GAIN': np.c_[nuis, mz, vr, eb, vr * mz[:, None]],
        'HEAD': np.c_[nuis, mz, vh, eb],
    }
    fits = {k: _cv(X, D['Y'], folds) for k, X in designs.items()}
    r2 = {k: f['r2'] for k, f in fits.items()}

    def beats(a, b):
        return (r2[a] > r2[b]) & (np.sum(fits[a]['r2_folds'] - fits[b]['r2_folds'] > 0, 0) >= 4)

    ret_tuned = beats('RET', 'NUIS') | beats('HEAD', 'NUIS')
    gain = ret_tuned & beats('GAIN', 'ADD')
    head_better = ret_tuned & beats('HEAD', 'ADD')
    eye_better = ret_tuned & beats('ADD', 'HEAD')

    # Full-data GAIN fit (z-scored design), median fold penalty.
    from .ffNLE_scene import _svd_prep, _ridge_coefs
    X = designs['GAIN']
    xm, xs = np.nanmean(X[use], 0), np.nanstd(X[use], 0)
    xs[~(xs > 0)] = 1.0
    Z = (X[use] - xm) / xs
    Yu = D['Y'][use]
    ym = Yu.mean(0)
    U, s_, Vt = _svd_prep(Z)
    lam = np.nanmedian(fits['GAIN']['lam'], 0)
    coef = _ridge_coefs(U, s_, Vt, Yu - ym, np.where(np.isfinite(lam), lam, 1e3))
    print('  {} gain-field models: {} pillar-tuned cells; gain-modulated {}, eye-centred-better '
          '{}, head-centred-better {}.'.format(label, int(ret_tuned.sum()), int(gain.sum()),
                                               int(eye_better.sum()), int(head_better.sum())))
    return {'r2': r2, 'tuned': ret_tuned, 'gain': gain, 'eye_better': eye_better,
            'head_better': head_better, 'n_frames': int(use.sum()),
            'gain_model': {'coef': coef, 'x_mean': xm, 'x_sd': xs, 'y_mean': ym,
                           'mod_mean': mu, 'mod_sd': sd, 'mod_centers': eq, 'mod_width': ew,
                           'nuis_names': [D['names'][i] for i in nuis_cols],
                           'layout': 'nuis, mod_z, vm(ang) x{0}, mod_gauss x4, vm(ang)*mod_z x{0}'.format(N_VM)}}


def reference_frames(D, seed=0):
    """ Eye- vs head-centred (light) and head- vs body-centred pillar coding.

    Eye/head: fields over retinal azimuth rho at low vs high eye-in-head e,
    head angle (heading x pitch) held fixed by reweighting.  Head/body: fields
    over head-centred azimuth eta at low vs high head-on-body hb, body axis =
    direction of travel (> 8 cm/s), heading held fixed.  The eye/head models
    are repeated in the dark, where the pillar is invisible: a "pillar field"
    that is just as common there is not visual.
    """

    light = D['light'] & D['use']
    hd_bin = (np.floor(((np.nan_to_num(D['heading']) + 180) % 360) / 45)).astype(int) % 8
    pq = np.nanpercentile(D['pitch'][light], [33.3, 66.7])
    p_bin = np.digitize(np.nan_to_num(D['pitch']), pq)
    strata_eye = hd_bin * 3 + p_bin
    nuis = np.concatenate([np.flatnonzero(np.char.startswith(np.array(D['names']), 'hd_vm')),
                           [D['names'].index('pitch'), D['names'].index('roll'),
                            D['names'].index('speed'), D['names'].index('log_dist')]])
    out = {}
    out['eye_shift'] = frame_shift_analysis(D, 'rho', 'e', light, strata_eye, seed=seed,
                                            label='Eye vs head (field shift)')
    out['eye_beta'] = beta_scan(D, 'rho', 'e', light, nuis, label='Eye vs head')
    out['eye_gain'] = gain_field_models(D, light, nuis, label='Eye vs head')
    dark = ~D['light'] & D['use']
    out['eye_beta_dark'] = beta_scan(D, 'rho', 'e', dark, nuis, label='Eye vs head [DARK control]')
    out['eye_gain_dark'] = gain_field_models(D, dark, nuis, label='Eye vs head [DARK control]')

    loco = light & np.isfinite(D['hb'])
    nuis_b = np.concatenate([np.flatnonzero(np.char.startswith(np.array(D['names']), 'hd_vm')),
                             [D['names'].index('speed'), D['names'].index('log_dist')]])
    if loco.sum() > 600:
        # ~150 s of locomotion spread over 36 bins and two terciles: a 2 s floor
        # blanks almost every bin, so this one runs at 0.75 s and is flagged
        # as underpowered.
        out['body_shift'] = frame_shift_analysis(D, 'eta', 'hb', loco, hd_bin, seed=seed,
                                                 label='Head vs body (field shift)', min_occ_s=0.75)
        out['body_beta'] = beta_scan(D, 'eta', 'hb', loco, nuis_b, label='Head vs body')
    else:
        print('  Head vs body: only {} locomotion frames in the light -- skipped.'.format(int(loco.sum())))
    return out


def _ridge_angle(Ztr, ang_tr, Zte, block_tr):
    """ Ridge from population activity to (cos, sin) of an angle; predicted angle in deg. """

    from .ffNLE_scene import _pick_lambda, _svd_prep, _ridge_coefs, LAM_GRID

    T = np.c_[np.cos(np.radians(ang_tr)), np.sin(np.radians(ang_tr))]
    lam = _pick_lambda(Ztr, T, block_tr, LAM_GRID)
    xm, tm = Ztr.mean(0), T.mean(0)
    U, s_, Vt = _svd_prep(Ztr - xm)
    Bc = _ridge_coefs(U, s_, Vt, T - tm, lam)
    P = (Zte - xm) @ Bc + tm
    return np.degrees(np.arctan2(P[:, 1], P[:, 0]))


def _circ_err(pred, true):
    """ MAE and circular-mean signed error (deg). """

    e = _wrap(pred - true)
    return float(np.mean(np.abs(e))), float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians(e))))))


def cross_position_decoders(D, cells, mask=None, n_null=20, seed=0):
    """ Decode pillar bearing at one eye position, test at another.

    Two decoders -- head-centred (eta) and retinal (rho) -- trained on frames in
    the low eye-position tercile and tested on the high one (and the reverse),
    against the same decoder cross-validated within the test tercile with a
    matched training size.  Performance P = 1 - MAE / MAE_chance, chance from
    the same pipeline on time-shifted labels.  Invariance = P_cross / P_within.

    The cross-condition bias is a second reference-frame readout: an
    eye-centred population makes a head decoder err by -delta_e when moved from
    low to high eye position, and a head-centred population makes a retinal
    decoder err by +delta_e.

    Both terciles are subsampled to a common distribution of the decoded
    angle first (see match), so a decoder moved between them meets the same
    prior; delta_e and the bias index are taken on the matched frames.
    """

    rng = np.random.default_rng(seed)
    m = (D['light'] & D['use'] if mask is None else mask) & np.isfinite(D['e']) \
        & np.isfinite(D['rho']) & np.isfinite(D['eta'])
    q = np.nanpercentile(D['e'][m], [100 / 3, 200 / 3])
    G = {0: m & (D['e'] <= q[0]), 1: m & (D['e'] >= q[1])}
    de = float(np.mean(D['e'][G[1]]) - np.mean(D['e'][G[0]]))
    Y = D['Y'][:, cells]
    blk = np.floor((D['t'] - D['t'][0]) / BLOCK_S).astype(int)
    out = {'delta_e': de, 'n_cells': int(len(cells))}
    if len(cells) < 5:
        return out

    def match(tgt, Gs, nb=12):
        """ Subsample both groups to a common distribution of the decoded angle.

        Where the pillar sits differs with eye position (gaze shifts orient
        toward it), so on DMM061 pos18 the low-e tercile has the pillar spread
        round the circle (R 0.10) and the high-e tercile has it ahead (R 0.60).
        A decoder moved between them then fails by the shift in its prior, not
        by any reference-frame change.  Each angle bin keeps min(n_low, n_high)
        frames from each group.
        """
        b = (np.floor((np.nan_to_num(tgt) + 180.0) % 360.0 / (360.0 / nb))).astype(int) % nb
        out = {k: np.zeros(len(tgt), bool) for k in Gs}
        for k in range(nb):
            idx = {g_: np.flatnonzero(Gs[g_] & (b == k)) for g_ in Gs}
            nk = min(len(v_) for v_ in idx.values())
            for g_, v_ in idx.items():
                if nk > 0:
                    out[g_][rng.choice(v_, nk, replace=False)] = True
        return out

    def run(target, shift=0):
        tgt = np.roll(D[target], shift) if shift else D[target]
        # A time-shifted label is missing wherever the shift brought in an
        # unusable frame; keep each group to frames with a label.
        Gs = match(tgt, {k: g_ & np.isfinite(tgt) for k, g_ in G.items()})
        res = {'n_matched': int(Gs[0].sum()),
               'de': float(np.mean(D['e'][Gs[1]]) - np.mean(D['e'][Gs[0]])) if Gs[0].any() else np.nan}
        for a, b in ((0, 1), (1, 0)):
            folds = _folds(D, Gs[b])
            errs_w, n_tr = [], []
            pred_w = np.full(len(tgt), np.nan)
            for tr, te in zip(folds[0], folds[1]):
                if tr.sum() < 100 or te.sum() < 20:
                    continue
                mu, sd = Y[tr].mean(0), Y[tr].std(0) + 1e-9
                pred_w[te] = _ridge_angle((Y[tr] - mu) / sd, tgt[tr], (Y[te] - mu) / sd, blk[tr])
                n_tr.append(int(tr.sum()))
            okw = np.isfinite(pred_w) & Gs[b]
            mae_w, bias_w = _circ_err(pred_w[okw], tgt[okw])
            # Cross: train on a block-contiguous subset of A matched in size.
            ia = np.flatnonzero(Gs[a])
            ntr = int(np.mean(n_tr)) if n_tr else len(ia)
            ub = np.unique(blk[ia])
            rng.shuffle(ub)
            keep = np.zeros(len(tgt), bool)
            for bb in ub:
                keep |= Gs[a] & (blk == bb)
                if keep.sum() >= ntr:
                    break
            mu, sd = Y[keep].mean(0), Y[keep].std(0) + 1e-9
            pc = _ridge_angle((Y[keep] - mu) / sd, tgt[keep], (Y[Gs[b]] - mu) / sd, blk[keep])
            mae_c, bias_c = _circ_err(pc, tgt[Gs[b]])
            res[(a, b)] = (mae_w, bias_w, mae_c, bias_c)
        return res

    PAIRS = ((0, 1), (1, 0))
    for target in ('eta', 'rho'):
        obs = run(target)
        null = [run(target, int(rng.integers(600, len(D['t']) - 600))) for _ in range(n_null)]
        o = {}
        for key, i_mae in (('within', 0), ('cross', 2)):
            mae = np.mean([obs[k][i_mae] for k in PAIRS])
            mae_null = np.mean([[nn[k][i_mae] for k in PAIRS] for nn in null])
            o['mae_' + key] = float(mae)
            o['mae_chance_' + key] = float(mae_null)
            o['P_' + key] = float(1 - mae / mae_null)
            o['P_null_' + key] = np.array([1 - np.mean([nn[k][i_mae] for k in PAIRS]) / mae_null for nn in null])
        o['n_matched'] = obs['n_matched']
        o['frac_matched'] = float(obs['n_matched'] / max(1, min(G[0].sum(), G[1].sum())))
        o['invariance'] = float(o['P_cross'] / o['P_within']) if o['P_within'] > 0 else np.nan
        # Signed cross bias, low->high minus high->low, over 2 delta_e.
        o['bias_lh'] = float(obs[(0, 1)][3])
        o['bias_hl'] = float(obs[(1, 0)][3])
        o['delta_e_matched'] = obs['de']
        o['bias_index'] = float((o['bias_lh'] - o['bias_hl']) / (2 * obs['de']))
        o['p_within'] = float((1 + np.sum(o['P_null_within'] >= o['P_within'])) / (1 + n_null))
        out[target] = o
    print('  Decoders ({} cells; bearing-matched terciles, {} / {} frames each): head-centred P within '
          '{:.3f} / cross {:.3f} (invariance {:.2f}, bias/de {:+.2f}); retinal P within {:.3f} / cross {:.3f} '
          '(invariance {:.2f}, bias/de {:+.2f}).'.format(len(cells), out['eta']['n_matched'], out['rho']['n_matched'],
                             out['eta']['P_within'], out['eta']['P_cross'],
                             out['eta']['invariance'], out['eta']['bias_index'],
                             out['rho']['P_within'], out['rho']['P_cross'], out['rho']['invariance'],
                             out['rho']['bias_index']))
    return out


def two_layer_model(D, v1_cells, gainfit=None, gainfit_noisy=None, targets=None, seed=0):
    """ Gain-field layer 1, linear readout layer 2.

    Layer 1 candidates:
      gain_basis    vm(rho) x [1, e, e^2]: planar + quadratic gain fields, the
                    Zipser-Andersen / Pouget basis.  Positive control -- a
                    shifted curve vm(rho + e) is, to second order, a linear
                    combination of exactly these.
      retinal_basis vm(rho) plus e, no products.  Negative control: it can
                    express retinal tuning but not head-centred tuning.
      v1_recorded   smoothed activity of the V1 cells.
      v1_gainfit    each pillar-tuned V1 cell's fitted gain-field model, i.e.
                    the measured gain fields with the noise taken out.  An
                    UPPER BOUND only: a linear readout ignores how weak a gain
                    term is, so noise-free model units span the product space
                    no matter how small the measured modulation.
      v1_gainfit_noisy  the same plus each cell's own residual, circularly
                    shifted -- the measured gain fields at the measured SNR.
    Layer 2: linear ridge readout, scored by block cross-validation over all
    light frames.  Targets are head-centred tuning curves vm(eta) -- what a
    head-centred readout area would carry -- and, as the contrast, retinal
    curves vm(rho).  Pass `targets` (a higher area's smoothed activity) to fit
    real cells instead.

    Training on one eye position and testing on another is NOT the test here:
    the eye terciles do not overlap, so units that only switch on at the other
    eye position never get a weight.  The basis-function claim is that one
    linear readout serves every eye position, so readout quality is also
    reported separately in each eye-position tercile of the held-out data.
    """

    from .ffNLE_scene import _r2_cols, cv_ridge

    m = D['light'] & D['use'] & np.isfinite(D['rho']) & np.isfinite(D['e']) & np.isfinite(D['eta'])
    e = D['e']
    ez = (e - np.nanmean(e[m])) / np.nanstd(e[m])
    vr = _vm_basis(D['rho'])
    L1 = {'gain_basis': np.c_[vr, vr * ez[:, None], vr * ez[:, None] ** 2],
          'retinal_basis': np.c_[vr, ez],
          'v1_recorded': D['Y'][:, v1_cells]}
    if gainfit is not None:
        L1['v1_gainfit'] = gainfit
    if gainfit_noisy is not None:
        L1['v1_gainfit_noisy'] = gainfit_noisy
    # With eye position spanning ~+-11 deg and vm tuning ~60 deg wide, head-
    # and retinal-centred curves are nearly the same function, so a retinal
    # layer reads out vm(eta) almost perfectly and the plain test has no power.
    # The diagnostic target is the HEAD-SPECIFIC part: vm(eta) residualized on
    # [vm(rho), e] over all usable frames.  A layer without rho x e products
    # cannot carry it.
    H = _vm_basis(D['eta'])
    A = np.c_[np.ones(len(e)), vr, ez]
    coef = np.linalg.lstsq(A[m], H[m], rcond=None)[0]
    tg = {'head': H, 'head_specific': H - A @ coef, 'retinal': vr}
    if targets is not None:
        tg = {'cells': targets}
    q = np.nanpercentile(e[m], [100 / 3, 200 / 3])
    terc = {'low': e <= q[0], 'mid': (e > q[0]) & (e < q[1]), 'high': e >= q[1]}
    out = {}
    for name, X in L1.items():
        use = m & np.all(np.isfinite(X), 1)
        folds = _folds(D, use)
        res = {'n_units': int(X.shape[1])}
        for tname, T in tg.items():
            # Out-of-fold predictions, for the per-tercile scores.
            trm, tem, block = folds
            fit = cv_ridge(_zbuilder(np.nan_to_num(X)), np.nan_to_num(T), trm, tem, block,
                           return_oof=True)
            res['r2_' + tname] = float(np.nanmean(fit['r2']))
            oof = fit['oof']
            for k, tm in terc.items():
                mm = use & tm & np.all(np.isfinite(oof), 1)
                res['r2_{}_{}'.format(tname, k)] = (float(np.nanmean(_r2_cols(T[mm], oof[mm])))
                                                    if mm.sum() > 50 else np.nan)
        if 'r2_head' in res:
            print('  Two-layer [{}] ({} units): readout R2 head-specific {:.3f} (eye low/mid/high '
                  '{:.3f}/{:.3f}/{:.3f}); head-centred {:.3f}; retinal {:.3f}.'.format(
                      name, X.shape[1], res['r2_head_specific'], res['r2_head_specific_low'],
                      res['r2_head_specific_mid'], res['r2_head_specific_high'], res['r2_head'],
                      res['r2_retinal']))
        out[name] = res
    return out


def gainfit_predictions(D, gm, mask):
    """ Each cell's fitted GAIN model evaluated on this session's covariates. """

    mz = (D['e'] - gm['mod_mean']) / gm['mod_sd']
    nuis = np.c_[D['X'][:, [D['names'].index(nm) for nm in gm['nuis_names']]]]
    vr = _vm_basis(D['rho'])
    eb = _gauss_basis(D['e'], gm['mod_centers'], gm['mod_width'])
    X = np.c_[nuis, mz, vr, eb, vr * mz[:, None]]
    P = ((X - gm['x_mean']) / gm['x_sd']) @ gm['coef'] + gm['y_mean']
    P[~mask] = np.nan
    return P


def object_vector_tuning(D, n_null=50, seed=0):
    """ Object-vector-like tuning: allocentric direction to the pillar x distance.

    Per cell, in the light: an allocentric-vector map (direction from the animal
    to the pillar in the room frame, x log distance), its split-half stability
    (alternate 60 s blocks) and a circular-shift null; then CV model comparison
    of ALLO = vm(allo dir) (x) distance bumps against EGO = vm(egocentric
    bearing) (x) distance, PLACE (5x5 position grid) and HD (heading).

    With one fixed pillar, allocentric direction + distance to it is the animal's
    position in polar coordinates about the pillar, so ALLO and PLACE describe
    the same variable in different bases; telling an object-vector cell from a
    place cell needs the pillar moved.  What this session CAN ask: is the
    direction part allocentric (ALLO > EGO, which differ by heading), and does
    it survive the dark, where the walls -- the only cue that can fix allocentric
    direction around a rotationally symmetric pillar -- are not visible?
    """

    rng = np.random.default_rng(seed)
    Y = D['Y']
    ld = np.log(np.clip(D['dist_cm'], 0.5, None))
    ok = D['use'] & np.isfinite(D['allo_dir']) & np.isfinite(ld)
    light = ok & D['light']
    dark = ok & ~D['light']
    de = np.nanpercentile(ld[ok], np.linspace(2, 98, 7))
    nd, na = len(de) - 1, 12

    def vmap(mask, Yx=Y):
        ai = (np.floor(((D['allo_dir'][mask] + 180) % 360) / (360 / na))).astype(int) % na
        di = np.clip(np.searchsorted(de, ld[mask]) - 1, 0, nd - 1)
        occ = np.zeros((na, nd))
        np.add.at(occ, (ai, di), 1.0)
        num = np.zeros((na, nd, Yx.shape[1]))
        np.add.at(num, (ai, di), Yx[mask])
        M = num / np.where(occ > 0, occ, np.nan)[..., None]
        M[occ * D['dt'] < 2.0] = np.nan
        return M

    blk = np.floor((D['t'] - D['t'][0]) / 60.0).astype(int)
    odd = (blk % 2).astype(bool)
    M_l, M_o, M_e, M_d = vmap(light), vmap(light & odd), vmap(light & ~odd), vmap(dark)
    C = Y.shape[1]
    stab = np.array([_nancorr(M_o[..., c].ravel(), M_e[..., c].ravel()) for c in range(C)])
    ld_r = np.array([_nancorr(M_l[..., c].ravel(), M_d[..., c].ravel()) for c in range(C)])
    null = np.zeros((n_null, C))
    for k in range(n_null):
        Ys = np.roll(Y, int(rng.integers(400, len(Y) - 400)), 0)
        a_, b_ = vmap(light & odd, Ys), vmap(light & ~odd, Ys)
        null[k] = [_nancorr(a_[..., c].ravel(), b_[..., c].ravel()) for c in range(C)]
    p_stab = (1 + np.sum(null >= stab[None, :], 0)) / (1 + n_null)

    dg = _gauss_basis(ld, np.linspace(de[0], de[-1], 4), (de[-1] - de[0]) / 4)
    va = _vm_basis(D['allo_dir'])
    ve = _vm_basis(D['ego_dir'])
    base = D['groups']['base']
    pos = D['groups']['allo'][:25]
    hd = D['groups']['allo'][25:]
    designs = {'ALLO': np.c_[(va[:, :, None] * dg[:, None, :]).reshape(len(ld), -1), D['X'][:, base]],
               'EGO': np.c_[(ve[:, :, None] * dg[:, None, :]).reshape(len(ld), -1), D['X'][:, base]],
               'PLACE': np.c_[D['X'][:, pos], D['X'][:, base]],
               'HD': np.c_[D['X'][:, hd], D['X'][:, base]],
               'BASE': D['X'][:, base]}
    folds = _folds(D, light)
    fits = {k: _cv(X, Y, folds) for k, X in designs.items()}
    r2 = {k: f['r2'] for k, f in fits.items()}

    def beats(a, b):
        return (r2[a] > r2[b]) & (np.sum(fits[a]['r2_folds'] - fits[b]['r2_folds'] > 0, 0) >= 4)

    # Light-trained ALLO model applied in the dark, against dark-trained.
    from .ffNLE_scene import _svd_prep, _ridge_coefs, _r2_cols
    X = np.nan_to_num(designs['ALLO'])
    xm, xs = X[light].mean(0), X[light].std(0) + 1e-9
    Z = (X - xm) / xs
    ym = Y[light].mean(0)
    U, s_, Vt = _svd_prep(Z[light])
    lam = np.nanmedian(fits['ALLO']['lam'], 0)
    Bc = _ridge_coefs(U, s_, Vt, Y[light] - ym, np.where(np.isfinite(lam), lam, 1e3))
    r2_l2d = _r2_cols(Y[dark], Z[dark] @ Bc + ym, mu=Y[dark].mean(0))
    folds_d = _folds(D, dark)
    r2_dd = _cv(designs['ALLO'], Y, folds_d)['r2']

    ovc = (stab > 0.5) & (p_stab < 0.05) & beats('ALLO', 'BASE') & beats('ALLO', 'EGO')
    print('  Object-vector: {} cells with stable allo-vector maps (split r > 0.5, p < 0.05); '
          '{} where ALLO also beats EGO and BASE in >= 4/5 folds; median light-dark map r '
          '{:.2f} for the stable ones.'.format(int(np.sum((stab > 0.5) & (p_stab < 0.05))), int(ovc.sum()),
                                             np.nanmedian(ld_r[(stab > 0.5) & (p_stab < 0.05)])
                                             if np.any((stab > 0.5) & (p_stab < 0.05)) else np.nan))
    return {'stab': stab, 'p_stab': p_stab, 'light_dark_r': ld_r, 'r2': r2,
            'allo_beats_ego': beats('ALLO', 'EGO'), 'allo_beats_base': beats('ALLO', 'BASE'),
            'r2_light_to_dark': r2_l2d, 'r2_dark_to_dark': r2_dd, 'ovc': ovc,
            'map_light': M_l, 'map_dark': M_d, 'dist_edges_log': de, 'n_dir_bins': na}


def run_neural(h5_path, s, p, fl, v, st, spike_key=SPIKE_KEY, composites_dir=None,
               n_shuffles=20, seed=0):
    """ Every neural analysis for one session; returns the result tree. """

    t0 = time.time()
    D = build_neural_dataset(h5_path, s, p, fl, v, st, spike_key=spike_key)
    C = D['Y'].shape[1]
    A = load_area_labels(h5_path, C, composites_dir)
    names = np.asarray(A['area']).astype(str)
    # Per-area analyses run on real areas only; border and unregistered cells
    # stay in the session-wide analyses.
    real = [a for a in sorted(set(names)) if a not in NON_AREAS]
    N = {'D': D, 'area_id': A['area_id'], 'area': names, 'area_atlas': np.asarray(A['area_atlas']).astype(str),
         'border_dist': A['border_dist'], 'own_sign': A['own_sign'],
         'cell_pos': A['wf_pos'], 'vfs_cell_pos': A['vfs_pos'], 'area_source': A['source']}
    print('Encoding models...')
    N['enc'] = encoding_models(D, n_shuffles=n_shuffles, seed=seed)
    print('Reference frames...')
    N['rf'] = reference_frames(D, seed=seed)
    print('Decoders, per area...')
    N['dec'] = {}
    for a in real:
        cells = np.flatnonzero(names == a)
        if len(cells) >= 10:
            N['dec'][a] = cross_position_decoders(D, cells, seed=seed)
    print('Two-layer model...')
    gm = N['rf']['eye_gain']['gain_model']
    light = D['light'] & D['use']
    G = gainfit_predictions(D, gm, light)
    N['two_layer'] = {}
    for a in real:
        cells = np.flatnonzero(names == a)
        if len(cells) < 10:
            continue
        tuned = cells[N['rf']['eye_gain']['tuned'][cells]]
        tc = tuned if len(tuned) >= 5 else cells
        gf = G[:, tc]
        # Add each cell's own residual back, circularly shifted within the
        # usable light frames, so the model units carry the measured noise
        # with its autocorrelation.
        rng = np.random.default_rng(seed)
        idx = np.flatnonzero(light & np.all(np.isfinite(gf), 1))
        noisy = np.full_like(gf, np.nan)
        for j, c in enumerate(tc):
            r = D['Y'][idx, c] - gf[idx, j]
            noisy[idx, j] = gf[idx, j] + np.roll(r, int(rng.integers(400, max(401, len(idx) - 400))))
        N['two_layer'][a] = two_layer_model(D, cells, gainfit=gf, gainfit_noisy=noisy, seed=seed)
    print('Object-vector tuning...')
    N['ovc'] = object_vector_tuning(D, seed=seed)
    print('  Neural analyses done in {:.0f} s.'.format(time.time() - t0))
    return N


def save_neural(path, N, R_):
    """ Everything a cross-area comparison needs, one session per file.

    Layout: /meta, /cells (area, positions), /encoding, /refframe/{eye,body},
    /gain_model (coefficients that rebuild each cell's eye gain field on any
    session's covariates), /decoders/<area>, /two_layer/<area>, /ovc, and
    /covariates (the 2P design matrix, names, groups, masks and raw angles), so a
    pooled analysis can refit or cross-evaluate without rerunning this module.
    """

    D = N['D']
    rf = N['rf']

    def shift_block(x):
        return {k: x[k] for k in ('ratio', 'ci', 'shift', 'r_shift', 'delta', 'split_r', 'p_field',
                                  'depth', 'reliable', 'n_frames', 'F_low', 'F_high', 'F_all')}

    def beta_block(x):
        return {k: x[k] for k in ('beta', 'r2', 'beta_best', 'index', 'r2_nuis', 'gain_over_nuis',
                                  'tuned', 'prefer_first', 'prefer_second', 'n_frames')}

    eg = rf['eye_gain']
    out = {
        'meta': {'session': R_['session'], 'h5_path': R_['h5_path'], 'npz_path': R_['npz_path'],
                 'spike_key': D['spike_key'], 'smooth_s': SMOOTH_S, 'n_frames': len(D['t']),
                 'frame_rate': 1 / D['dt'], 'light_frac': float(D['light'].mean()),
                 'use_frac': float(D['use'].mean()), 'area_source': N['area_source'],
                 'groups': ','.join(GROUPS), 'vor_gain_slow_phase': R_['v']['vor_slope'],
                 'date': R_['date']},
        'cells': {'area_id': N['area_id'], 'area': ','.join(N['area']),
                  'area_atlas': ','.join(N['area_atlas']), 'border_dist': N['border_dist'],
                  'own_sign': N['own_sign'], 'cell_pos': N['cell_pos'], 'vfs_cell_pos': N['vfs_cell_pos'],
                  'mean_rate': np.nanmean(D['Y_raw'], 0),
                  'mean_rate_light': np.nanmean(D['Y_raw'][D['light'] & D['use']], 0),
                  'mean_rate_dark': np.nanmean(D['Y_raw'][~D['light'] & D['use']], 0)},
        'encoding': {k: N['enc'][k] for k in ('r2_full', 'r2_folds', 'p_full', 'null_full', 'r2_base',
                                              'unique', 'alone', 'unique_folds_pos', 'sig')},
        'refframe': {'eye_shift': shift_block(rf['eye_shift']), 'eye_beta': beta_block(rf['eye_beta']),
                     'eye_models': {'r2_' + k: x for k, x in eg['r2'].items()}},
        'gain_model': {k: (np.asarray(x) if not isinstance(x, (str, list)) else
                           (','.join(x) if isinstance(x, list) else x))
                       for k, x in eg['gain_model'].items()},
        'decoders': {}, 'two_layer': {},
        'ovc': {k: N['ovc'][k] for k in ('stab', 'p_stab', 'light_dark_r', 'allo_beats_ego',
                                         'allo_beats_base', 'r2_light_to_dark', 'r2_dark_to_dark',
                                         'ovc', 'map_light', 'map_dark', 'dist_edges_log')},
        'covariates': {'t': D['t'], 'X': D['X'].astype(np.float32), 'names': ','.join(D['names']),
                       'use': D['use'], 'light': D['light'],
                       **{'group_' + g: idx for g, idx in D['groups'].items()},
                       **{k: D[k] for k in ('rho', 'eta', 'e', 'hb', 'heading', 'allo_dir', 'ego_dir',
                                            'dist_cm', 'speed', 'pitch', 'roll', 'state')}},
    }
    for k in ('tuned', 'gain', 'eye_better', 'head_better'):
        out['refframe']['eye_models'][k] = eg[k]
    if 'eye_gain_dark' in rf:
        egd = rf['eye_gain_dark']
        out['refframe']['eye_models_dark'] = {**{'r2_' + k: x for k, x in egd['r2'].items()},
                                              **{k: egd[k] for k in ('tuned', 'gain', 'eye_better', 'head_better')}}
        out['refframe']['eye_beta_dark'] = beta_block(rf['eye_beta_dark'])
    for k, x in N['ovc']['r2'].items():
        out['ovc']['r2_' + k] = x
    if 'body_shift' in rf:
        out['refframe']['body_shift'] = shift_block(rf['body_shift'])
        out['refframe']['body_beta'] = beta_block(rf['body_beta'])
    for a, d in N['dec'].items():
        out['decoders'][a] = {'delta_e': d['delta_e'], 'n_cells': d['n_cells']}
        for tg in ('eta', 'rho'):
            if tg in d:
                out['decoders'][a][tg] = d[tg]
    for a, d in N['two_layer'].items():
        out['two_layer'][a] = d
    write_h5(path, out)
    print('Saved {}.'.format(path))


def pool_neural(h5_paths, out_pdf=None):
    """ Pool _neural.h5 files across sessions and summarize per area.

    Returns a dict area -> summary (cell counts, median unique variance per
    group, reference-frame indices for reliable / tuned cells, gain-field
    fraction, decoder invariance per session, OVC counts).
    """

    from .utils.files import read_h5

    def as_str(x):
        x = x.item() if isinstance(x, np.ndarray) and x.size == 1 else x
        return x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)

    rows = {}
    for path in h5_paths:
        r = read_h5(path)
        areas = as_str(r['cells']['area']).split(',')
        sig = np.asarray(r['encoding']['sig'], bool)
        uniq = np.asarray(r['encoding']['unique'])
        ups = np.asarray(r['encoding']['unique_folds_pos'])
        es = r['refframe']['eye_shift']
        eb = r['refframe']['eye_beta']
        em = r['refframe']['eye_models']
        for a in sorted(set(areas)):
            m = np.array(areas) == a
            d = rows.setdefault(a, {'n_cells': 0, 'n_sig': 0, 'unique_sig': [], 'unique_frac': [],
                                    'eye_ratio': [], 'eye_index': [], 'n_tuned': 0, 'n_gain': 0,
                                    'n_head_better': 0, 'n_eye_better': 0, 'dec_inv_eta': [],
                                    'dec_inv_rho': [], 'dec_bias_eta': [], 'n_ovc': 0, 'sessions': []})
            d['sessions'].append(as_str(r['meta']['session']))
            d['n_cells'] += int(m.sum())
            d['n_sig'] += int((m & sig).sum())
            d['unique_sig'].append(uniq[m & sig])
            d['unique_frac'].append(((ups[m] >= 4) & (uniq[m] > 0) & sig[m, None]))
            rel = m & np.asarray(es['reliable'], bool)
            d['eye_ratio'].append(np.asarray(es['ratio'])[rel])
            tun = m & np.asarray(eb['tuned'], bool)
            d['eye_index'].append(np.asarray(eb['index'])[tun])
            d['n_tuned'] += int((m & np.asarray(em['tuned'], bool)).sum())
            d['n_gain'] += int((m & np.asarray(em['gain'], bool)).sum())
            d['n_head_better'] += int((m & np.asarray(em['head_better'], bool)).sum())
            d['n_eye_better'] += int((m & np.asarray(em['eye_better'], bool)).sum())
            d['n_ovc'] += int((m & np.asarray(r['ovc']['ovc'], bool)).sum())
            dec = r.get('decoders', {}).get(a)
            if dec:
                d['dec_inv_eta'].append(float(dec['eta']['invariance']))
                d['dec_inv_rho'].append(float(dec['rho']['invariance']))
                d['dec_bias_eta'].append(float(dec['eta']['bias_index']))
    summary = {}
    for a, d in rows.items():
        U = np.concatenate(d['unique_sig'], 0) if d['unique_sig'] else np.zeros((0, len(GROUPS)))
        F = np.concatenate(d['unique_frac'], 0)
        er = np.concatenate(d['eye_ratio'])
        ei = np.concatenate(d['eye_index'])
        summary[a] = {'sessions': d['sessions'], 'n_cells': d['n_cells'], 'n_sig': d['n_sig'],
                      'unique_median': dict(zip(GROUPS, np.nanmedian(U, 0) if len(U) else [np.nan] * len(GROUPS))),
                      'unique_frac_cells': dict(zip(GROUPS, F.mean(0) if len(F) else [np.nan] * len(GROUPS))),
                      'eye_ratio_median': float(np.nanmedian(er)) if len(er) else np.nan,
                      'eye_ratio_n': int(len(er)),
                      'eye_index_median': float(np.nanmedian(ei)) if len(ei) else np.nan,
                      'eye_index_n': int(len(ei)),
                      'frac_gain_of_tuned': d['n_gain'] / d['n_tuned'] if d['n_tuned'] else np.nan,
                      'n_tuned': d['n_tuned'], 'n_head_better': d['n_head_better'],
                      'n_eye_better': d['n_eye_better'],
                      'decoder_invariance_eta': d['dec_inv_eta'], 'decoder_invariance_rho': d['dec_inv_rho'],
                      'decoder_bias_index_eta': d['dec_bias_eta'], 'n_ovc': d['n_ovc']}
        print('  {}: {} cells in {} session(s), {} beat their null; eye-frame ratio median {:.2f} '
              '(n={}), beta index {:.2f} (n={}); gain-modulated {}/{} tuned.'.format(
                  a, d['n_cells'], len(d['sessions']), d['n_sig'], summary[a]['eye_ratio_median'],
                  summary[a]['eye_ratio_n'], summary[a]['eye_index_median'], summary[a]['eye_index_n'],
                  d['n_gain'], d['n_tuned']))
    return summary


def _area_colors(names):
    """ Colour per cell by area. """

    return [AREA_COLORS.get(a, '#b9b8b2') for a in names]


def _page_neural_summary(pdf, R_):
    """ Neural section opener: data, headline numbers, caveats. """

    N = R_['neural']
    enc, rf = N['enc'], N['rf']
    D = N['D']
    fig = _new_page('Neurons: encoding, reference frames, readout, object vectors',
                    'Spike source: {} ({}), triangle-smoothed {:.1f} s.  Areas: {}.  All p-values are '
                    'within this one session.'.format(D['spike_key'],
                                                      'OMSI rate' if D['spike_key'] == 'norm_spikes' else 'see key',
                                                      SMOOTH_S,
                                                      ', '.join('{} {}'.format(a, int(np.sum(N['area'] == a)))
                                                                for a in sorted(set(N['area'])))))
    import textwrap
    es, eb, eg = rf['eye_shift'], rf['eye_beta'], rf['eye_gain']
    rel = es['reliable']
    lines = [
        ('Encoding', '{} / {} cells beat both a circular-shift null and the light+drift model; median full R2 '
         '{:.4f}.  Cells with a consistent unique share (>= 4/5 folds): {}.'.format(
             int(enc['sig'].sum()), len(enc['sig']), np.nanmedian(enc['r2_full']),
             ', '.join('{} {}'.format(GROUP_LABELS[g], int(np.sum(enc['sig'] & (enc['unique_folds_pos'][:, i] >= 4)
                                                                  & (enc['unique'][:, i] > 0))))
                       for i, g in enumerate(GROUPS)))),
        ('Eye vs head', 'Field-shift ratio (1 eye-centred, 0 head-centred) on {} cells with a reliable '
         'retinal field: median {:.2f} [IQR {:.2f}-{:.2f}], eye-position contrast {:.1f} deg.  Beta scan on {} '
         'pillar-tuned cells: median index {:.2f}; {} prefer the eye-centred end and {} the head-centred end in '
         '>= 4/5 folds.  Model comparison: {} tuned, {} gain-modulated, {} '
         'eye-centred-better, {} head-centred-better.'.format(
             int(rel.sum()), np.nanmedian(es['ratio'][rel]) if rel.any() else np.nan,
             *(np.nanpercentile(es['ratio'][rel], [25, 75]) if rel.any() else [np.nan, np.nan]),
             es['delta'], int(eb['tuned'].sum()),
             np.nanmedian(eb['index'][eb['tuned']]) if eb['tuned'].any() else np.nan,
             int(eb['prefer_first'].sum()), int(eb['prefer_second'].sum()),
             int(eg['tuned'].sum()), int(eg['gain'].sum()), int(eg['eye_better'].sum()),
             int(eg['head_better'].sum()))),
    ]
    if 'eye_gain_dark' in rf:
        gd, bd = rf['eye_gain_dark'], rf['eye_beta_dark']
        lines.append(('Dark control', 'Same eye/head models in the dark (pillar invisible): {} "tuned", {} '
                      'gain-modulated, {} eye-centred-better, {} head-centred-better; beta scan {} tuned ({} / {} '
                      'prefer eye / head end).  Light counts above that do not exceed these are not visual.'.format(
                          int(gd['tuned'].sum()), int(gd['gain'].sum()), int(gd['eye_better'].sum()),
                          int(gd['head_better'].sum()), int(bd['tuned'].sum()), int(bd['prefer_first'].sum()),
                          int(bd['prefer_second'].sum()))))
    if 'body_shift' in rf:
        bs, bb = rf['body_shift'], rf['body_beta']
        lines.append(('Head vs body', 'Body axis = direction of travel (no body points are tracked).  {} '
                      'locomotion frames; {} reliable cells, median ratio {:.2f} (1 head-centred, 0 '
                      'body-centred); beta scan {} tuned, median index {:.2f}; {} prefer head-centred, {} '
                      'body-centred in >= 4/5 folds.'.format(
                          bs['n_frames'], int(bs['reliable'].sum()),
                          np.nanmedian(bs['ratio'][bs['reliable']]) if bs['reliable'].any() else np.nan,
                          int(bb['tuned'].sum()),
                          np.nanmedian(bb['index'][bb['tuned']]) if bb['tuned'].any() else np.nan,
                          int(bb['prefer_first'].sum()), int(bb['prefer_second'].sum()))))
    for a, d in N['dec'].items():
        if 'eta' in d:
            lines.append(('Decoders ({})'.format(a),
                          'Head-centred bearing: P within {:.3f}, across eye positions {:.3f}, invariance '
                          '{:.2f}, bias/de {:+.2f}.  Retinal bearing: P within {:.3f}, across {:.3f}, '
                          'invariance {:.2f}, bias/de {:+.2f}.  (P = 1 - MAE/chance; within-tercile p = '
                          '{:.3g} / {:.3g}; bearing distribution matched across the eye terciles, {} / {} '
                          'frames each.)'.format(d['eta']['P_within'], d['eta']['P_cross'],
                                                       d['eta']['invariance'], d['eta']['bias_index'],
                                                       d['rho']['P_within'], d['rho']['P_cross'],
                                                       d['rho']['invariance'], d['rho']['bias_index'],
                                                       d['eta']['p_within'], d['rho']['p_within'],
                                                       d['eta'].get('n_matched', -1), d['rho'].get('n_matched', -1))))
    for a, d in N['two_layer'].items():
        lines.append(('Two-layer ({})'.format(a), 'Readout CV R2 of the head-SPECIFIC component (head tuning '
                      'residualized on retinal tuning + eye) / plain head-centred / retinal: ' + '  |  '.join(
            '{} {:.3f} / {:.3f} / {:.3f}'.format(k, x['r2_head_specific'], x['r2_head'], x['r2_retinal'])
            for k, x in d.items())))
    o = N['ovc']
    st_ = (o['stab'] > 0.5) & (o['p_stab'] < 0.05)
    lines.append(('Object vectors', '{} cells with a stable allocentric-vector map; {} also beat the '
                  'egocentric-vector and nuisance models.  Median light-dark map r for stable cells '
                  '{:.2f}.  With one fixed pillar this is not separable from place tuning.'.format(
                      int(st_.sum()), int(o['ovc'].sum()),
                      np.nanmedian(o['light_dark_r'][st_]) if st_.any() else np.nan)))
    lines.append(('Area claim', 'This session is {} only, so "V1/AM = gain-modulated basis, A/RL = readout" '
                  'cannot be tested here; the _neural.h5 carries everything pool_neural() needs once AM, A '
                  'and RL sessions are run.'.format('/'.join(sorted(set(N['area']))))))
    y = 0.86
    for head, body in lines:
        fig.text(0.04, y, head, fontsize=9.5, weight='bold', color=INK, va='top')
        w = textwrap.fill(body, 132)
        fig.text(0.19, y, w, fontsize=8.8, color=INK, va='top')
        y -= 0.0215 * (w.count('\n') + 1) + 0.017
    pdf.savefig(fig)
    plt.close(fig)


def _page_encoding(pdf, R_):
    """ Encoding models: full R^2 and unique variance per group. """

    N = R_['neural']
    enc = N['enc']
    fig = _new_page('Encoding models: unique variance per covariate group',
                    'Cross-validated ridge (5 contiguous-block folds, 40 s embargo, nested penalty).  Unique '
                    'variance = R2(full) - R2(full minus group).  Points = cells that beat their null.')
    gs = fig.add_gridspec(2, 3, left=0.07, right=0.98, top=0.85, bottom=0.12, hspace=0.55, wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(min(-0.02, np.nanpercentile(enc['r2_full'], 1)), np.nanpercentile(enc['r2_full'], 99.5), 50)
    ax.hist(enc['null_full'], bins, color='#b9b8b2', label='shuffle null (median per cell)')
    ax.hist(enc['r2_full'], bins, histtype='step', color='#2a78d6', lw=1.6, label='observed')
    ax.set_xlabel('full-model CV R2', fontsize=8.5)
    ax.set_title('{} / {} cells beat null'.format(int(enc['sig'].sum()), len(enc['sig'])), fontsize=9, loc='left')
    ax.legend(fontsize=7, frameon=False)
    _style(ax, grid_y=False)

    ax = fig.add_subplot(gs[0, 1:])
    sig = enc['sig']
    rng = np.random.default_rng(0)
    areas = sorted(set(N['area']))
    for i, g in enumerate(GROUPS):
        for j, a in enumerate(areas):
            m = sig & (N['area'] == a)
            yv = enc['unique'][m, i]
            xv = i + (j - (len(areas) - 1) / 2) * 0.25 + rng.uniform(-0.06, 0.06, len(yv))
            ax.scatter(xv, yv, s=6, color=AREA_COLORS.get(a, '#b9b8b2'), alpha=0.6,
                       label=a if i == 0 else None)
            if len(yv):
                ax.plot([xv.mean() - 0.1, xv.mean() + 0.1], [np.median(yv)] * 2, color=INK, lw=1.6)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUPS], fontsize=8, rotation=25, ha='right')
    ax.set_ylabel('unique R2', fontsize=8.5)
    ax.set_title('Unique variance (black bar = median)', fontsize=9, loc='left')
    ax.legend(fontsize=7, frameon=False, markerscale=2)
    _style(ax)

    ax = fig.add_subplot(gs[1, 0:2])
    w = 0.8 / max(len(areas), 1)
    for j, a in enumerate(areas):
        m = sig & (N['area'] == a)
        fr = [100 * np.mean((enc['unique_folds_pos'][m, i] >= 4) & (enc['unique'][m, i] > 0)) if m.any() else 0
              for i in range(len(GROUPS))]
        ax.bar(np.arange(len(GROUPS)) + (j - (len(areas) - 1) / 2) * w, fr, width=w,
               color=AREA_COLORS.get(a, '#b9b8b2'), label=a)
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUPS], fontsize=8, rotation=25, ha='right')
    ax.set_ylabel('% of null-beating cells', fontsize=8.5)
    ax.set_title('Cells carrying the group uniquely (>= 4/5 folds)', fontsize=9, loc='left')
    ax.legend(fontsize=7, frameon=False)
    _style(ax)

    ax = fig.add_subplot(gs[1, 2])
    med = [np.nanmedian(enc['alone'][sig, i]) if sig.any() else np.nan for i in range(len(GROUPS))]
    ax.barh(np.arange(len(GROUPS)), med, color='#86b6ef')
    ax.axvline(np.nanmedian(enc['r2_base'][sig]) if sig.any() else 0, color=INK2, lw=0.8, ls='--')
    ax.set_yticks(range(len(GROUPS)))
    ax.set_yticklabels([GROUP_LABELS[g] for g in GROUPS], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('median CV R2, group + nuisance alone', fontsize=8.5)
    ax.set_title('Group alone (dashed = nuisance only)', fontsize=9, loc='left')
    _style(ax, grid_y=False)
    pdf.savefig(fig)
    plt.close(fig)


def _page_frames(pdf, R_, kind):
    """ Eye-vs-head or head-vs-body reference frame page. """

    N = R_['neural']
    rf = N['rf']
    sh, be = rf[kind + '_shift'], rf[kind + '_beta']
    under = sh['n_frames'] < 3000
    if kind == 'eye':
        title = 'Eye-centred or head-centred? Pillar fields across eye-in-head positions'
        sub = ('Fields over retinal azimuth (rho) at low vs high eye-in-head (lower/upper tercile, delta {:.1f} deg), '
               'heading x pitch held fixed by reweighting; light only.  Ratio = 1 + shift/delta: 1 eye-centred, '
               '0 head-centred.'.format(sh['delta']))
        xl, ml = 'retinal pillar azimuth (deg)', ('eye low', 'eye high')
    else:
        title = 'Head-centred or body-centred? Pillar fields across head-on-body angles'
        sub = ('Fields over head-centred azimuth (eta) at low vs high head-on-body (delta {:.1f} deg).  Body axis = '
               'direction of travel during locomotion (> 8 cm/s); no body points are tracked.  Ratio 1 '
               'head-centred, 0 body-centred.'.format(sh['delta']))
        xl, ml = 'head-centred pillar azimuth (deg)', ('head-on-body low', 'high')
    if under:
        sub += '  UNDERPOWERED: {} frames ({:.0f} s).'.format(sh['n_frames'], sh['n_frames'] * N['D']['dt'])
    fig = _new_page(title, sub)
    gs = fig.add_gridspec(2, 4, left=0.06, right=0.98, top=0.84, bottom=0.08, hspace=0.55, wspace=0.38)
    rel = sh['reliable']
    ax = fig.add_subplot(gs[0, 0])
    r = np.clip(sh['ratio'][rel], -1.5, 2.5)
    ax.hist(r, np.linspace(-1.5, 2.5, 33), color='#2a78d6')
    for x_ in (0, 1):
        ax.axvline(x_, color=INK2, lw=0.8, ls='--')
    ax.set_xlabel('field-shift ratio', fontsize=8.5)
    ax.set_title('Reliable fields: n={}, median {:.2f}'.format(int(rel.sum()), np.nanmedian(r) if len(r) else np.nan),
                 fontsize=8.5, loc='left')
    _style(ax)

    ax = fig.add_subplot(gs[0, 1])
    if rel.any():
        ci = sh['ci'][rel]
        order = np.argsort(sh['ratio'][rel])
        yy = np.arange(len(order))
        ax.errorbar(np.clip(sh['ratio'][rel][order], -2, 3), yy,
                    xerr=[np.clip(sh['ratio'][rel][order] - ci[order, 0], 0, 5),
                          np.clip(ci[order, 1] - sh['ratio'][rel][order], 0, 5)],
                    fmt='o', ms=3, color='#2a78d6', ecolor='#86b6ef', elinewidth=1)
        for x_ in (0, 1):
            ax.axvline(x_, color=INK2, lw=0.8, ls='--')
        ax.set_xlim(-2, 3)
    ax.set_yticks([])
    ax.set_xlabel('ratio with 95% block-bootstrap CI', fontsize=8.5)
    ax.set_title('Per-cell uncertainty', fontsize=8.5, loc='left')
    _style(ax, grid_y=False)

    ax = fig.add_subplot(gs[0, 2])
    tun = be['tuned']
    ax.hist(be['index'][tun], np.arange(-1.05, 2.1, 0.1), color='#eb6834')
    for x_ in (0, 1):
        ax.axvline(x_, color=INK2, lw=0.8, ls='--')
    ax.set_xlabel('beta-scan index (1 - beta*)', fontsize=8.5)
    edge = tun & ((be['beta_best'] <= be['beta'][0]) | (be['beta_best'] >= be['beta'][-1]))
    ax.set_title('Tuned n={}; {} at grid edge\ncalls: {} first frame, {} second (4/5 folds)'.format(
        int(tun.sum()), int(edge.sum()), int(be['prefer_first'].sum()), int(be['prefer_second'].sum())),
        fontsize=8, loc='left')
    _style(ax)

    ax = fig.add_subplot(gs[0, 3])
    if tun.any():
        R = be['r2'][:, tun] - np.nanmax(be['r2'][:, tun], 0)
        ax.plot(1 - be['beta'], R, color='#b9b8b2', lw=0.6, alpha=0.6)
        ax.plot(1 - be['beta'], np.nanmedian(R, 1), color='#eb6834', lw=2)
    ax.set_xlabel('index (1 = eye-/head-centred frame)' if kind == 'eye' else 'index', fontsize=8.5)
    ax.set_ylabel('R2 - max', fontsize=8.5)
    ax.set_title('CV R2 profiles over frames', fontsize=8.5, loc='left')
    _style(ax)

    # Example fields.
    score = np.where(rel, sh['split_r'], -np.inf)
    ex = [c for c in np.argsort(-score)[:4] if np.isfinite(score[c])]
    xb = np.arange(36) * 10 - 180 + 5
    for k in range(4):
        ax = fig.add_subplot(gs[1, k])
        if k < len(ex):
            c = ex[k]
            ax.plot(xb, sh['F_low'][:, c], color='#2a78d6', lw=1.6, label=ml[0])
            ax.plot(xb, sh['F_high'][:, c], color='#eb6834', lw=1.6, label=ml[1])
            ax.set_title('cell {} ({})  ratio {:.2f} [{:.2f}, {:.2f}]'.format(
                c, N['area'][c], sh['ratio'][c], sh['ci'][c, 0], sh['ci'][c, 1]), fontsize=8, loc='left')
            if k == 0:
                ax.legend(fontsize=7, frameon=False)
        ax.set_xlabel(xl, fontsize=8)
        ax.set_xlim(-180, 180)
        ax.set_xticks([-180, -90, 0, 90, 180])
        _style(ax)
    pdf.savefig(fig)
    plt.close(fig)


def _page_gain_models(pdf, R_):
    """ Gain-field model comparison and two-layer readout. """

    N = R_['neural']
    eg = N['rf']['eye_gain']
    r2 = eg['r2']
    fig = _new_page('Gain fields and the readout',
                    'Left: per-cell model comparison (light).  Right: decoders trained at one eye position and '
                    'tested at the other, and a two-layer model (gain-field layer -> linear head-centred readout).')
    gs = fig.add_gridspec(2, 3, left=0.07, right=0.98, top=0.85, bottom=0.1, hspace=0.55, wspace=0.38)
    ax = fig.add_subplot(gs[0, 0])
    t_ = eg['tuned']
    ax.scatter(r2['GAIN'][t_] - r2['ADD'][t_], r2['HEAD'][t_] - r2['ADD'][t_], s=10,
               c=_area_colors(N['area'][t_]))
    ax.axhline(0, color=INK2, lw=0.6)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlabel('R2 gain-field - additive', fontsize=8.5)
    ax.set_ylabel('R2 head-centred - additive', fontsize=8.5)
    ax.set_title('Pillar-tuned cells (n={})'.format(int(t_.sum())), fontsize=9, loc='left')
    _style(ax, grid_y=False)

    ax = fig.add_subplot(gs[1, 0])
    areas = sorted(set(N['area']))
    keys = (('tuned', 'tuned'), ('gain', 'gain-mod.'), ('eye_better', 'eye-centred >'), ('head_better', 'head-centred >'))
    w = 0.8 / max(len(areas), 1)
    for j, a in enumerate(areas):
        m = N['area'] == a
        ax.bar(np.arange(4) + (j - (len(areas) - 1) / 2) * w, [int(np.sum(eg[k][m])) for k, _ in keys],
               width=w, color=AREA_COLORS.get(a, '#b9b8b2'), label=a)
    ax.set_xticks(range(4))
    ax.set_xticklabels([l for _, l in keys], fontsize=8)
    ax.set_ylabel('cells', fontsize=8.5)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)

    ax = fig.add_subplot(gs[0, 1:])
    labels, wv, cv, cols = [], [], [], []
    for a, d in N['dec'].items():
        for tg, nm in (('eta', 'head-centred'), ('rho', 'retinal')):
            if tg in d:
                labels.append('{}\n{}'.format(a, nm))
                wv.append(d[tg]['P_within'])
                cv.append(d[tg]['P_cross'])
    x = np.arange(len(labels))
    ax.bar(x - 0.18, wv, width=0.34, color='#2a78d6', label='within eye position')
    ax.bar(x + 0.18, cv, width=0.34, color='#eb6834', label='across eye positions')
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('P = 1 - MAE / chance', fontsize=8.5)
    ax.set_title('Population decoders of pillar bearing', fontsize=9, loc='left')
    ax.legend(fontsize=7, frameon=False)
    _style(ax)

    ax = fig.add_subplot(gs[1, 1:])
    labels, sv, hv, rv, terc = [], [], [], [], []
    for a, d in N['two_layer'].items():
        for k, x_ in d.items():
            labels.append('{}\n{}'.format(a, k.replace('_', ' ')))
            sv.append(x_['r2_head_specific'])
            hv.append(x_['r2_head'])
            rv.append(x_['r2_retinal'])
            terc.append([x_['r2_head_specific_low'], x_['r2_head_specific_mid'], x_['r2_head_specific_high']])
    x = np.arange(len(labels))
    ax.bar(x - 0.27, sv, width=0.26, color='#eb6834', label='head-SPECIFIC readout')
    ax.bar(x, hv, width=0.26, color='#fbc5a7', label='head-centred readout')
    ax.bar(x + 0.27, rv, width=0.26, color='#2a78d6', label='retinal readout')
    for i, tv in enumerate(terc):
        ax.plot([i - 0.36, i - 0.27, i - 0.18], tv, 'o-', color=INK, ms=3, lw=0.8,
                label='head-specific at eye low/mid/high' if i == 0 else None)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel('readout CV R2', fontsize=8.5)
    ax.set_title('Two-layer model: layer 1 -> one linear readout for all eye positions', fontsize=9, loc='left')
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    pdf.savefig(fig)
    plt.close(fig)


def _page_ovc(pdf, R_):
    """ Object-vector-like tuning. """

    N = R_['neural']
    o = N['ovc']
    fig = _new_page('Object-vector-like tuning: allocentric direction to the pillar x distance',
                    'A rotationally symmetric pillar cannot give allocentric direction by itself; any such signal '
                    'must come from walls or distal cues -- so it should weaken in the dark.  With one fixed pillar, '
                    'allocentric vector = position in polar coordinates about the pillar.')
    gs = fig.add_gridspec(2, 4, left=0.06, right=0.98, top=0.84, bottom=0.08, hspace=0.55, wspace=0.4)
    stable = (o['stab'] > 0.5) & (o['p_stab'] < 0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(o['stab'][np.isfinite(o['stab'])], 40, color='#2a78d6')
    ax.axvline(0.5, color=INK2, lw=0.8, ls='--')
    ax.set_xlabel('split-half map r (light)', fontsize=8.5)
    ax.set_title('{} stable (r > 0.5, p < 0.05)'.format(int(stable.sum())), fontsize=8.5, loc='left')
    _style(ax)
    ax = fig.add_subplot(gs[0, 1])
    d_ = o['r2']['ALLO'] - o['r2']['EGO']
    d_ = d_[np.isfinite(d_)]
    ax.hist(np.clip(d_, -0.05, 0.05), np.linspace(-0.05, 0.05, 41), color='#eb6834')
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel('R2 allo-vector - ego-vector', fontsize=8.5)
    ax.set_title('{} cells ALLO > EGO (>= 4/5 folds)'.format(int(o['allo_beats_ego'].sum())), fontsize=8.5, loc='left')
    _style(ax)
    ax = fig.add_subplot(gs[0, 2])
    xd = np.clip(o['r2_dark_to_dark'], -0.2, None)
    yd = np.clip(o['r2_light_to_dark'], -0.2, None)
    ax.scatter(xd, yd, s=6, color='#52514e', alpha=0.5)
    ax.scatter(xd[stable], yd[stable], s=14, color='#eb6834', label='stable (clipped at -0.2)')
    lim = [-0.2, max(0.05, np.nanmax([xd, yd]))]
    ax.plot(lim, lim, color=INK2, lw=0.6)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('ALLO R2, dark-trained (CV)', fontsize=8.5)
    ax.set_ylabel('ALLO R2, light-trained, tested in dark', fontsize=8.5)
    ax.legend(fontsize=7, frameon=False)
    _style(ax, grid_y=False)
    ax = fig.add_subplot(gs[0, 3])
    ax.hist(o['light_dark_r'][stable & np.isfinite(o['light_dark_r'])], np.linspace(-1, 1, 21), color='#1baf7a')
    ax.set_xlabel('light vs dark map r (stable cells)', fontsize=8.5)
    _style(ax)
    ex = [c for c in np.argsort(-np.where(stable, o['stab'], -np.inf))[:2] if stable[c]]
    na = o['map_light'].shape[0]
    th = np.radians(np.arange(na + 1) * 360 / na - 180)
    rr = np.exp(o['dist_edges_log'])
    for k, c in enumerate(ex):
        for j, (M, nm) in enumerate(((o['map_light'], 'light'), (o['map_dark'], 'dark'))):
            ax = fig.add_subplot(gs[1, 2 * k + j], projection='polar')
            Z = M[..., c]
            ax.pcolormesh(th, rr, np.ma.masked_invalid(Z).T, cmap='Blues', shading='flat')
            ax.set_title('cell {} {} (r {:.2f})'.format(c, nm, o['stab'][c]), fontsize=8, loc='left')
            ax.tick_params(labelsize=6)
    pdf.savefig(fig)
    plt.close(fig)


def make_pdf(path, R_):
    """ Multi-page summary PDF. """

    with PdfPages(path) as pdf:
        pages = [_page_summary, _page_pillar, _page_flow, _page_classes, _page_motifs,
                 _page_landing, _page_landing_breakdown, _page_states, _page_covariates,
                 _page_methods]
        if 'neural' in R_:
            pages += [_page_neural_summary, _page_encoding, lambda p_, r_: _page_frames(p_, r_, 'eye')]
            if 'body_shift' in R_['neural']['rf']:
                pages.append(lambda p_, r_: _page_frames(p_, r_, 'body'))
            pages += [_page_gain_models, _page_ovc]
        for fn in pages:
            try:
                fn(pdf, R_)
            except Exception as e:
                # One broken panel should not cost the whole report.
                import traceback
                fig = _new_page('{} failed'.format(getattr(fn, '__name__', 'page')), repr(e))
                pdf.savefig(fig)
                plt.close(fig)
                print('  WARNING: {} failed: {!r}'.format(getattr(fn, '__name__', 'page'), e))
                traceback.print_exc()
    print('Saved {}.'.format(path))


def _h5_safe(x):
    """ Make a nested result write_h5-able: drop None, tuples to arrays, str lists joined. """

    if isinstance(x, dict):
        return {str(k): _h5_safe(v_) for k, v_ in x.items() if v_ is not None}
    if isinstance(x, (list, tuple)):
        if len(x) and all(isinstance(e, str) for e in x):
            return ','.join(x)
        if any(isinstance(e, dict) for e in x):
            return {str(i): _h5_safe(e) for i, e in enumerate(x)}
        return np.asarray(x)
    if isinstance(x, np.ndarray) and x.dtype == object:
        return ','.join(str(e) for e in x.ravel())
    return x


def figure_data(R_):
    """ Everything the PDF pages draw that the per-frame arrays don't already hold.

    Nulls, fitted summaries, flow-field means and motif traces, so the figure
    notebook only reads files.  Saved as /fig in the _behavior.h5.
    """

    s, fl, v, lab, st = R_['s'], R_['fl'], R_['v'], R_['lab'], R_['st']
    # Mean flow field per condition, as drawn on the flow page.
    conds = (('comp_left', (lab == 0) & (v['head'] > SHIFTED_HEAD)),
             ('nasal', lab == 1), ('temporal', lab == 2),
             ('forward', (st['speed'] > 10) & (np.abs(st['yaw_rate']) < 30)))
    ffield = {}
    for nm, m in conds:
        m = m & np.isfinite(fl['field'][:, 0, 0])
        ffield[nm] = {'n': int(m.sum()),
                      'mean': np.nanmean(fl['field'][m], 0) if m.sum() > 10 else np.full(fl['field'].shape[1:], np.nan)}
    mi = R_['mi']
    W = 2 * mi['trace_half'] + 1
    tr_mu = np.full((4, 2, 3, W), np.nan)
    tr_sd = np.full((4, 2, 3, W), np.nan)
    tr_n = np.zeros((4, 2), int)
    for k in range(4):
        for j, d in enumerate((1, -1)):
            for i, key in enumerate(('head', 'eye', 'gaze')):
                mu, sd, nn = mi['traces'][(k, d, key)]
                tr_mu[k, j, i], tr_sd[k, j, i], tr_n[k, j] = mu, sd, nn
    land = R_['land']
    fig = {
        'meta': {'session': R_['session'], 'date': R_['date'], 'h5_path': R_['h5_path'],
                 'npz_path': R_['npz_path'], 'n_blocks': R_['n_blocks']},
        'const': {'FS': FS, 'BINO_HALF_DEG': BINO_HALF_DEG, 'FIELD_EDGE_DEG': FIELD_EDGE_DEG,
                  'SG_WIN': SG_WIN, 'SG_ORD': SG_ORD, 'SHIFTED_HEAD': SHIFTED_HEAD, 'STILL_GAZE': STILL_GAZE,
                  'SHIFTED_GAZE': SHIFTED_GAZE, 'LAND_PAD': LAND_PAD, 'N_NULL': N_NULL,
                  'RECON_FOV_DEG': RECON_FOV_DEG, 'MOTIF_NAMES': list(MOTIF_NAMES),
                  'STATE_NAMES': list(STATE_NAMES)},
        'session': {k: s[k] for k in ('native_fps', 'wall_h', 'arena', 'pillar_xy', 'pillar_r', 'pillar_h',
                                      'eye_scale_px', 'mean_pfh', 'mean_phi')},
        'traces': {'hx_s': st['hx_s'], 'hy_s': st['hy_s'], 'ear_sep': s['ear_sep']},
        'checks': {'val': R_['val'], 'ffc': R_['ffc'], 'ifc': R_['ifc']},
        'pil_sum': R_['pil_sum'], 'flow_sum': R_['flow_sum'],
        'flow_field': {'grid_az': fl['grid_az'], 'grid_el': fl['grid_el'], **ffield},
        'vor': {k: v[k] for k in ('vor_slope', 'vor_slope_light', 'vor_slope_dark')},
        'classes': {'centres_head_eye': R_['cls_info']['centres_head_eye'], 'n_dropped': R_['cls_info']['n_dropped'],
                    'k_counts': np.asarray(R_['k_counts']), 'k_frac': R_['k_frac'],
                    'cls_frac': np.asarray(R_['cls_frac'])},
        'motifs': {'trace_mean': tr_mu, 'trace_sd': tr_sd, 'trace_n': tr_n, 'trace_half': mi['trace_half'],
                   'trace_dirs': np.array([1, -1]), 'trace_keys': 'head,eye,gaze',
                   'eye_head_ratio': mi['eye_head_ratio'], 'match_ratio': mi['match_ratio'],
                   'transitions': R_['trans']},
        'landing': {'light': land['light'], 'dark': land['dark'], 'light_motif_events': R_['land_m']['light'],
                    'light_peth_thresholds': R_['land_thr']['light'], 'light_kmeans_literal': R_['land_k']['light'],
                    'by_motif': land['by_motif'], 'by_state': land['by_state'], 'by_dist': land['by_dist'],
                    'by_dist_order': list(land['by_dist'].keys()),
                    'light_vs_dark': land.get('light_vs_dark', {}),
                    'sens': {'{:.0f}'.format(h): x for h, x in R_['sens'].items()},
                    'events': {k: land[k] for k in ('az_pre', 'az_post', 'dG', 'dB', 'keep')}},
        'robust_names': 'half15,half25,half30,peth_thresholds,motif_events,kmeans_literal',
        'robust': {str(i): x for i, x in enumerate(R_['robust'])},
        'cov': {'metrics': R_['cov']['metrics'], 'labels': R_['cov']['labels'],
                'light': R_['cov']['light'], 'dark': R_['cov']['dark']},
        'state_params': st['params'],
    }
    return _h5_safe(fig)


def main():
    """ Command-line entry point. """

    ap = argparse.ArgumentParser(description=__doc__.split('\n')[3])
    ap.add_argument('--h5_path')
    ap.add_argument('--npz_path')
    ap.add_argument('--out_prefix', default=None,
                    help='Defaults to the h5 path minus _preproc.h5, plus _behavior.')
    ap.add_argument('--skip_warp_check', action='store_true')
    ap.add_argument('--skip_neural', action='store_true')
    ap.add_argument('--spike_key', default=SPIKE_KEY,
                    help='norm_spikes = OMSI rate (default); oasis_spks = OASIS; norm_spikes_s2p = suite2p.')
    ap.add_argument('--composites_dir', default=None,
                    help='mouse_composites directory for area labels (fm2p.area_labels); defaults to '
                         '<V1PPC>/mouse_composites beside the cohort.')
    ap.add_argument('--n_shuffles', type=int, default=20)
    ap.add_argument('--pool', nargs='+', default=None,
                    help='Instead of running a session, pool these _neural.h5 files by area.')
    ap.add_argument('--images_npz', default=None,
                    help='A rendered reconstruction for the image checks, when --npz_path is '
                         'pose-only (get_retinal_image --pose_only).')
    ap.add_argument('--eye_scale', default='auto',
                    help="'auto' uses <prefix>_eye_scale.json if present, 'stored' the preproc "
                         "eye angles, or a scale in px.")
    args = ap.parse_args()
    if args.pool:
        pool_neural(args.pool)
        return
    if not (args.h5_path and args.npz_path):
        ap.error('--h5_path and --npz_path are required unless --pool is given')

    t0 = time.time()
    prefix = args.out_prefix or args.h5_path.replace('_preproc.h5', '') + '_behavior'
    print('Loading session...')
    s = load_session(args.h5_path, args.npz_path,
                     eye_scale=args.eye_scale if args.eye_scale in ('auto', 'stored') else float(args.eye_scale))
    img_npz = args.images_npz or args.npz_path
    with np.load(img_npz, allow_pickle=False) as _z:
        has_img = _z['retinal_images'].shape[0] > 0
    pose_match = None
    if os.path.abspath(img_npz) != os.path.abspath(args.npz_path):
        pose_match = check_pose_match(args.npz_path, img_npz)
    if has_img:
        print('Checking pose against the stored reconstruction...')
        val = validate_pose(img_npz)
    else:
        print('No rendered images -- skipping the re-render and warp checks.')
        val = {'n': 0, 'mean_abs_err': np.nan, 'frac_exact': np.nan}
    if pose_match is not None:
        val['pose_match'] = pose_match['match']
    print('Pillar geometry...')
    p = pillar_geometry(s)
    print('Optic flow...')
    fl = optic_flow(s)
    ffc = flow_formula_check(s)
    ifc = {} if (args.skip_warp_check or not has_img) else warp_check(img_npz)
    print('Eye/head movements...')
    v = eye_head_velocity(s)
    lab_k, cls_info = classify_paper(v)
    lab = classify_gated(v)
    lab_t = classify_threshold(v)
    ev, mi = find_motifs(v, lab)
    trans = motif_transitions(ev)
    try:
        ev_k, _ = find_motifs(v, lab_k)
        k_counts = [int(np.sum(ev_k['motif'] == k)) for k in range(4)]
    except RuntimeError:
        k_counts = [0, 0, 0, 0]
    print('Behavioral states...')
    st = behavior_states(s)
    print('Landing analysis...')
    ev_all = episode_events(lab, v)
    land = landing_analysis(ev_all, p, st, s['light'], label=' [all gaze shifts]')
    land_m = landing_analysis(ev, p, st, s['light'], label=' [motif events]')
    land['by_motif'] = land_m['by_motif']
    land_thr = landing_analysis(episode_events(lab_t, v), p, st, s['light'], label=' [PETH thresholds]')
    land_k = landing_analysis(episode_events(lab_k, v), p, st, s['light'], label=' [k-means literal]')
    sens = {}
    for hw in (10.0, 15.0, 20.0, 25.0, 30.0):
        sens[hw] = landing_analysis(ev_all, p, st, s['light'], half=hw,
                                    label=' [half {:.0f}]'.format(hw))['light']
    cov = state_covariates(st, v, lab, ev_all, ev, p, fl, s['light'])
    robust = [sens[15.0], sens[25.0], sens[30.0], land_thr['light'], land_m['light'], land_k['light']]

    v_ok = p['zone'] >= 0
    lm = s['light'] & v_ok
    comp = (lab == 0) & (np.abs(v['head']) > SHIFTED_HEAD)
    shf = (lab == 1) | (lab == 2)
    fs_ = {'total': np.nanmedian(fl['total']), 'rot': np.nanmedian(fl['rot']),
           'trans': np.nanmedian(fl['trans']),
           'comp_free': np.nanmedian(fl['rot'][comp]), 'comp_frozen': np.nanmedian(fl['rot_frozen'][comp]),
           'shift_free': np.nanmedian(fl['rot'][shf]), 'shift_frozen': np.nanmedian(fl['rot_frozen'][shf])}
    fs_['comp_stab'] = 100 * (fs_['comp_frozen'] - fs_['comp_free']) / fs_['comp_frozen']
    fs_['shift_stab'] = 100 * (fs_['shift_frozen'] - fs_['shift_free']) / fs_['shift_frozen']
    pil_sum = {'in_recon': float(p['in_recon'][v_ok].mean()),
               'in_field': float(np.isin(p['zone'][v_ok], [0, 1]).mean()),
               'zone_light': [float(np.mean(p['zone'][lm] == k)) for k in range(4)],
               'width_med': float(np.nanmedian(p['width_deg'])),
               'width_q': np.nanpercentile(p['width_deg'], [25, 75]),
               'height_med': float(np.nanmedian(p['height_deg'])),
               'dist_med': float(np.nanmedian(p['dist_mm']))}
    vf = lab != -1
    R_ = {'session': os.path.basename(args.h5_path).replace('_preproc.h5', ''),
          'date': time.strftime('%Y-%m-%d'), 'npz_path': args.npz_path, 'h5_path': args.h5_path,
          's': s, 'p': p, 'fl': fl, 'ifc': ifc, 'ffc': ffc, 'val': val, 'v': v, 'lab': lab,
          'lab_k': lab_k, 'lab_t': lab_t, 'k_counts': k_counts, 'ev_all': ev_all,
          'cls_info': cls_info, 'ev': ev, 'mi': mi, 'trans': trans, 'st': st, 'land': land,
          'land_thr': land_thr, 'land_k': land_k, 'land_m': land_m, 'robust': robust, 'sens': sens, 'cov': cov, 'flow_sum': fs_, 'pil_sum': pil_sum,
          'cls_frac': [100 * np.mean(lab[vf] == k) for k in range(3)],
          'k_frac': 100 * np.mean(np.isin(lab_k[lab_k != -1], [1, 2])),
          'n_blocks': len(_runs(s['light'])) + len(_runs(~s['light']))}

    if not args.skip_neural:
        print('Neural analyses ({})...'.format(args.spike_key))
        R_['neural'] = run_neural(args.h5_path, s, p, fl, v, st, spike_key=args.spike_key,
                                  composites_dir=args.composites_dir, n_shuffles=args.n_shuffles)
        save_neural(prefix.replace('_behavior', '') + '_neural{}.h5'.format(
            '' if args.spike_key == SPIKE_KEY else '_' + args.spike_key), R_['neural'], R_)

    make_pdf(prefix + '_summary{}.pdf'.format(
        '' if (args.skip_neural or args.spike_key == SPIKE_KEY) else '_' + args.spike_key), R_)

    # Per-frame arrays and events.  Flow field and null distributions stay out.
    def scal(L):
        return {k: (float(np.mean(x)) if isinstance(x, np.ndarray) else x) for k, x in L.items()
                if np.ndim(x) == 0 or isinstance(x, np.ndarray) and x.size == N_NULL}
    out = {'t': s['t'], 'light': s['light'], 'pose_ok': s['pose_ok'], 'eye_ok': s['eye_ok'],
           'pillar': {k: p[k] for k in ('az_eye', 'el_eye', 'az_head', 'el_head', 'az_cyc', 'eye_dev',
                                         'dist_mm', 'width_deg', 'height_deg', 'solid_deg2', 'in_recon',
                                         'zone', 'slip', 'slip_frozen')},
           'flow': {k: fl[k] for k in ('total', 'rot', 'trans', 'total_frozen', 'rot_frozen',
                                       'trans_frozen', 'axis', 'omega', 'speed_mm_s', 'frac_sky')},
           'velocity': {k: v[k] for k in ('head', 'eye', 'gaze', 'head_acc', 'eye_acc')},
           'class_gated': lab, 'class_kmeans': lab_k, 'class_threshold': lab_t, 'motif_frames': mi['motif_frames'],
           'events_motif': ev, 'events_all': ev_all,
           'state': st['state'], 'state_names': ','.join(STATE_NAMES),
           'kin': {k: st[k] for k in ('speed', 'yaw_rate', 'nose_up', 'd_pillar', 'd_wall', 'ego_pillar')},
           'landing_light': scal(land['light']), 'landing_dark': scal(land['dark']),
           'fig': figure_data(R_)}
    write_h5(prefix + '.h5', out)

    summ = {'session': R_['session'], 'validation': val, 'pillar': {k: (list(np.atleast_1d(x).astype(float)) if np.ndim(x) else float(x)) for k, x in pil_sum.items()},
            'flow': {k: float(x) for k, x in fs_.items()},
            'warp_check': {k: x for k, x in ifc.items() if not isinstance(x, np.ndarray)},
            'flow_formula_check': ffc,
            'class_frac_gated': R_['cls_frac'],
            'motif_counts_kmeans_literal': dict(zip(MOTIF_NAMES, k_counts)),
            'motif_counts': {nm: int(np.sum(ev['motif'] == k)) for k, nm in enumerate(MOTIF_NAMES)},
            'landing': {}}
    for nm, L in (('light', land['light']), ('dark', land['dark']),
                  ('light_motif_events', land_m['light']), ('light_peth_thresholds', land_thr['light']),
                  ('light_kmeans_literal', land_k['light'])):
        summ['landing'][nm] = {k: (float(np.mean(x)) if isinstance(x, np.ndarray) else x)
                               for k, x in L.items()}
        summ['landing'][nm]['verdict'] = _verdict(L, robust if nm == 'light' else None)
    summ['landing']['light_vs_dark'] = {k: (x.tolist() if isinstance(x, np.ndarray) else x)
                                        for k, x in land.get('light_vs_dark', {}).items()}
    with open(prefix + '_summary.json', 'w') as f:
        json.dump(summ, f, indent=2, default=float)
    print('Saved {}.h5 and {}_summary.json.  Total {:.0f} s.'.format(prefix, prefix, time.time() - t0))


if __name__ == '__main__':
    main()
