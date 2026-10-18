# -*- coding: utf-8 -*-
"""
fm2p/gaze_stabilization.py

Does the eye stabilize the reconstructed retinal image?

The retinocentric reconstruction differs from an egocentric one only by the eye
term, so the pair isolates what eye movement contributes.  This module
quantifies that per recording, using the package's own definitions of
compensatory and gaze-shifting movements (PETH.py).

The test is a prediction, not a correlation.  A compensatory eye movement
cancels head rotation, so during compensatory epochs the image should move LESS
across the retina when the eye is included than when it is frozen.  During gaze
shifts the eye moves with the head and the opposite must hold.  If the two
classes look alike, the eye term is carrying nothing.

Two measures, both computed analytically from the pose the renderer used:

  gaze axis   the optic axis in world coordinates.  Image stabilization is a
              property of where the eye points, so this needs no object at all.
  pillar      the pillar's bearing in the eye's optical frame.  Note this is
              defined on EVERY frame, including when the pillar falls outside
              the rendered 120 deg window -- the object still has a bearing
              relative to the retina, it is only the rasterized image that
              clips.  An earlier version measured the pillar's centroid in the
              rendered image instead, which forced a visibility filter and threw
              away most of the data (41.3% to 8.8% of frames) for no reason.

Because the eye term is toggled analytically, a single `.npz` is enough: the
egocentric control is the same pose with the pupil held at rest, which is what
`get_retinal_image --fixed_pupil` renders.

Usage
-----
    python -m fm2p.gaze_stabilization \\
        --h5_path <prefix>_preproc.h5 \\
        --npz_path <prefix>_retinal_images.npz

Functions
---------
_pose
    Rotations and head position per frame, as the renderer builds them.
gaze_axis
    Optic axis in world coordinates, (N, 3) unit vectors.
pillar_retinal_direction
    Pillar bearing in the eye's optical frame, (N, 3) unit vectors.
angular_speed
    Total angular speed between consecutive unit vectors, deg/s.
horizontal_speed
    Speed of the HORIZONTAL (azimuth) component only, deg/s.
vertical_speed
    Speed of the ELEVATION component only, deg/s.
classify_movements
    Compensatory / gaze-shifting, per eye frame.
_stats
    Median speeds, stabilization, block-bootstrap CI and event count.
analyze
    Stabilization by the eye, for both measures and every movement class.
make_figure
    Two rows: azimuth on top, elevation below, with the summary at bottom-left.
main
    Command-line entry point.


DMM, September 2026
"""

if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

import argparse
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from .utils.files import write_h5

# Movement thresholds, deg/s -- the package's existing definitions (PETH.py),
# kept identical so the classes mean the same thing everywhere.
SHIFTED_HEAD = 60.0     # head must exceed this to count as moving at all
# Gaze slower than this while the head moves = compensatory.
STILL_GAZE = 120.0
SHIFTED_GAZE = 240.0    # gaze faster than this, with the head = gaze shift

HEAD_Z_MM = 25.0
EYE_OFFSET_MM = (3.5, -5.0, 3.5)
# Eye frame (x fwd, y left, z up) to optical frame (x right, y down, z fwd).
R_ALIGN = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])


def _pose(npz, freeze_pupil=False, pitch_gain=1.0):
    """Rotations and head position per frame, as the renderer builds them."""

    yaw = np.asarray(npz['yaw_eye'], float)
    pitch = np.asarray(npz['pitch_eye'], float)
    roll = np.asarray(npz['roll_eye'], float)
    pfh = np.asarray(npz['pfh_eye'], float)
    phi = np.asarray(npz['phi_trim'], float)
    hx = np.asarray(npz['head_x_eye'], float)
    hy = np.asarray(npz['head_y_eye'], float)
    n = min(map(len, (yaw, pitch, roll, pfh, phi, hx, hy)))
    yaw, pitch, roll, pfh, phi, hx, hy = (a[:n] for a in
                                          (yaw, pitch, roll, pfh, phi, hx, hy))

    mean_pfh = float(npz['mean_pfh']) if 'mean_pfh' in npz else np.nanmean(pfh)
    mean_phi = float(npz['mean_phi']) if 'mean_phi' in npz else np.nanmean(phi)
    alpha = float(npz['anatomical_eye_angle_deg'])
    beta = float(npz['socket_elev_deg'])

    tilt_h = np.where(np.isfinite(pfh), pfh, mean_pfh)
    tilt_v = np.where(np.isfinite(phi), phi, mean_phi)
    if freeze_pupil:
        tilt_h = np.full(n, mean_pfh)
        tilt_v = np.full(n, mean_phi)
    # Renderer sign convention (see get_retinal_image): BOTH pupil angles are
    # negated.  tilt_v used to keep +phi, so the vertical eye term pointed the
    # wrong way; re-rendering frames with +phi misses the stored images by a
    # mean 43 luminance levels, with -phi it is pixel-exact.
    tilt_h = -tilt_h
    tilt_v = -tilt_v

    ok = np.isfinite(yaw) & np.isfinite(pitch) & np.isfinite(roll)
    Rh = np.full((n, 3, 3), np.nan)
    Re = np.full((n, 3, 3), np.nan)
    if ok.any():
        Rh[ok] = R.from_euler('ZYX', np.stack(
            [-yaw[ok], pitch[ok] * pitch_gain, roll[ok]], axis=1),
            degrees=True).as_matrix()
        Rs = (R.from_euler('z', alpha, degrees=True).as_matrix() @
              R.from_euler('y', beta, degrees=True).as_matrix())
        Rsac = R.from_euler('zyx', np.stack(
            [tilt_h[ok], tilt_v[ok], np.zeros(int(ok.sum()))], axis=1),
            degrees=True).as_matrix()
        Re[ok] = Rs @ Rsac
    return {'R_head': Rh, 'R_eye': Re, 'ok': ok, 'hx': hx, 'hy': hy, 'n': n}


def gaze_axis(npz, freeze_pupil=False):
    """Optic axis in world coordinates, (N, 3) unit vectors."""

    p = _pose(npz, freeze_pupil)
    a = np.full((p['n'], 3), np.nan)
    m = p['ok']
    if m.any():
        a[m] = np.einsum('nij,njk,k->ni', p['R_head'][m], p['R_eye'][m],
                         np.array([1.0, 0.0, 0.0]))
    return a


def pillar_retinal_direction(npz, freeze_pupil=False):
    """Pillar bearing in the eye's optical frame, (N, 3) unit vectors.

    Defined on every frame with a tracked pose, including frames where the
    pillar falls outside the rendered field -- being off-frame does not make the
    bearing undefined, it only means the rasterizer clipped it.
    """

    p = _pose(npz, freeze_pupil)
    px, py = float(npz['pillar_x_mm']), float(npz['pillar_y_mm'])
    target = np.array([px, py, HEAD_Z_MM])   # level with the eye

    out = np.full((p['n'], 3), np.nan)
    m = p['ok'] & np.isfinite(p['hx']) & np.isfinite(p['hy'])
    if m.any():
        head = np.stack([p['hx'][m], p['hy'][m],
                         np.full(int(m.sum()), HEAD_Z_MM)], axis=1)
        v_world = target[None, :] - head
        v_head = np.einsum('nji,nj->ni', p['R_head'][m], v_world)   # R_head^T v
        v_head = v_head - np.asarray(EYE_OFFSET_MM)[None, :]
        v_eye = np.einsum('nji,nj->ni', p['R_eye'][m], v_head)      # R_eye^T v
        v_opt = v_eye @ R_ALIGN.T
        nrm = np.linalg.norm(v_opt, axis=1, keepdims=True)
        out[m] = v_opt / np.where(nrm > 0, nrm, np.nan)
    return out


def angular_speed(vecs, dt):
    """Total angular speed between consecutive unit vectors, deg/s.

    Works for directions anywhere on the sphere, so there is no wrapping problem
    when the target passes behind the animal.
    """

    d = np.full(len(vecs), np.nan)
    dot = np.clip(np.einsum('ij,ij->i', vecs[:-1], vecs[1:]), -1.0, 1.0)
    d[1:] = np.degrees(np.arccos(dot)) / dt
    return d


def horizontal_speed(vecs, dt, x=0, y=1):
    """Speed of the HORIZONTAL (azimuth) component only, deg/s.

    This is the measure that matches the movement classification, and the
    distinction matters a great deal.  `compensatory` and `gaze_shift` are
    defined purely on yaw (dGaze = -gyro_z + dtheta/dt), so the eye term being
    tested is horizontal.  Scoring those classes with the TOTAL 3-D angular
    speed also folds in the vertical eye movement, which is unconstrained by the
    classification and on this data swamps the horizontal effect: the same
    session reads +33% stabilization in azimuth and -12% in total speed.  Report
    the matched component as the headline and keep the total as context.
    """

    az = np.degrees(np.unwrap(np.arctan2(vecs[:, y], vecs[:, x])))
    return np.gradient(az, dt)


def vertical_speed(vecs, dt, up=2, sign=1.0):
    """Speed of the ELEVATION component only, deg/s.

    Elevation is bounded to +-90 deg so it never wraps; no unwrapping needed.
    `up` is the component that points up in that frame and `sign` orients it:
    the world frame has z up (up=2, sign=+1), the optical frame has y DOWN
    (up=1, sign=-1).

    Read this as a control on the azimuth result rather than a second version of
    it.  The movement classes are defined purely on yaw, so they say nothing
    about what the eye should be doing vertically.  A large elevation effect in
    the yaw-defined compensatory class therefore does not mean the eye is
    stabilizing elevation -- it means the vertical eye term covaries with the
    horizontal one, whether through real coupling or through shared noise.
    """

    el = np.degrees(np.arcsin(np.clip(sign * vecs[:, up], -1.0, 1.0)))
    return np.gradient(el, dt)


def classify_movements(theta_deg, gyro_z, eyeT,
                       shifted_head=SHIFTED_HEAD, still_gaze=STILL_GAZE,
                       shifted_gaze=SHIFTED_GAZE):
    """Compensatory / gaze-shifting, per eye frame.

    Package convention: dHead = -gyro_z, dEye = +dtheta/dt, dGaze = dHead + dEye.
    Under it a perfectly compensating eye gives dGaze = 0, so `compensatory` is
    literally "the head moved and gaze did not".
    """

    dt = float(np.nanmedian(np.diff(eyeT)))
    dHead = -np.asarray(gyro_z, dtype=float)
    dEye = np.gradient(np.asarray(theta_deg, dtype=float), dt)
    dGaze = dHead + dEye
    moving = np.isfinite(dGaze) & (np.abs(dHead) > shifted_head)
    comp = moving & (np.abs(dGaze) < still_gaze)
    shift = (moving & (np.abs(dGaze) > shifted_gaze)
             & (np.sign(dGaze) == np.sign(dHead)))
    return {'dHead': dHead, 'dEye': dEye, 'dGaze': dGaze,
            'moving': moving, 'compensatory': comp, 'gaze_shift': shift}


def _stats(vr, ve, mask, dt, rng_seed=0):
    """Median speeds, stabilization, block-bootstrap CI and event count."""

    m = mask & np.isfinite(vr) & np.isfinite(ve)
    runs = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    out = {'n': int(m.sum()), 'n_events': int((runs == 1).sum())}
    if m.sum() < 50:
        return out
    se, sr = float(np.median(ve[m])), float(np.median(vr[m]))
    out['speed_ego'], out['speed_retino'] = se, sr
    out['stabilization'] = (se - sr) / se if se > 0 else np.nan

    # Frames are heavily autocorrelated, so resample contiguous blocks; a
    # frame-wise interval would be far too narrow.
    idx = np.flatnonzero(m)
    blk = max(1, int(round(0.5 / dt)))
    starts = np.arange(0, max(len(idx) - blk, 1), blk)
    if len(starts) >= 10:
        rng = np.random.default_rng(rng_seed)
        boot = []
        for _ in range(400):
            sel = np.concatenate([idx[s:s + blk]
                                  for s in rng.choice(starts, len(starts))])
            a, b = np.median(ve[sel]), np.median(vr[sel])
            boot.append((a - b) / a if a > 0 else np.nan)
        out['ci'] = tuple(np.nanpercentile(boot, [2.5, 97.5]))
    return out


def analyze(h5_path, npz_path, speed_thresh=2.0, light_only=True):
    """Stabilization by the eye, for both measures and every movement class."""

    with h5py.File(h5_path, 'r') as f:
        theta = np.asarray(f['theta_trim'][:], float)
        eyeT = np.asarray(f['eyeT_trim'][:], float)
        gyro_z = np.asarray(f['gyro_z_eye_interp'][:], float)
        twopT = np.asarray(f['twopT'][:], float)
        speed = np.asarray(f['speed'][:], float) if 'speed' in f else None
        ltdk = (np.asarray(f['ltdk_state_vec'][:]).astype(bool)
                if 'ltdk_state_vec' in f else None)
        lag = (float(f['eye_imu_lag_samples'][()])
               if 'eye_imu_lag_samples' in f else np.nan)
    if not np.isfinite(lag):
        print('  WARNING: no eye_imu_lag_samples stamp -- this h5 predates the '
              'eye/IMU timing correction. Re-run preprocessing.')

    npz = np.load(npz_path, allow_pickle=False)
    if 'fixed_pupil' in npz and bool(npz['fixed_pupil']):
        print('  WARNING: this npz was rendered with --fixed_pupil, so its pose '
              'already has the eye frozen. Pass the normal (eye-free) run.')

    n = min(len(theta), len(eyeT), len(gyro_z), len(npz['yaw_eye']))
    theta, eyeT, gyro_z = theta[:n], eyeT[:n], gyro_z[:n]
    dt = float(np.nanmedian(np.diff(eyeT)))

    # For the gaze axis (world frame, y-up) azimuth is atan2(y, x); for the
    # pillar bearing (optical frame, x right / z forward) it is atan2(x, z).
    meas = {}
    for name, fn, xy, upn in (('gaze', gaze_axis, (0, 1), (2, 1.0)),
                              ('pillar', pillar_retinal_direction, (2, 0), (1, -1.0))):
        vr, ve = fn(npz, False)[:n], fn(npz, True)[:n]
        meas[f'{name}_az'] = (np.abs(horizontal_speed(vr, dt, *xy)),
                              np.abs(horizontal_speed(ve, dt, *xy)))
        meas[f'{name}_el'] = (np.abs(vertical_speed(vr, dt, *upn)),
                              np.abs(vertical_speed(ve, dt, *upn)))
        meas[f'{name}_total'] = (angular_speed(vr, dt), angular_speed(ve, dt))

    cls = classify_movements(theta, gyro_z, eyeT)
    keep = np.ones(n, dtype=bool)
    if speed is not None and len(twopT) >= 2:
        keep &= np.interp(eyeT, twopT[:len(speed)],
                          speed[:len(twopT)]) > speed_thresh
    if light_only and ltdk is not None and len(twopT) >= 2:
        keep &= np.interp(eyeT, twopT[:len(ltdk)],
                          ltdk[:len(twopT)].astype(float)) > 0.5

    print('  {} eye frames; {} ({:.1f}%) pass '
          'the light/speed filters'.format(n, int(keep.sum()), 100 * keep.mean()))

    out = {'eyeT': eyeT, 'keep': keep, 'dt': dt, 'lag_stamp': lag, **cls}
    for mname, (vr, ve) in meas.items():
        out[f'{mname}_retino'], out[f'{mname}_ego'] = vr, ve
        for cname in ('compensatory', 'gaze_shift', 'moving'):
            out[f'{mname}_{cname}'] = _stats(vr, ve, keep & cls[cname], dt)
    return out


def make_figure(res, out_png, title=''):
    """ 2x2 azimuth-only summary: movement classes, stabilization bars, image speed histograms.

    Top-left: dHead vs dEye with the compensatory and gaze-shift classes.
    Top-right: stabilization by the eye for each class.  Bottom: horizontal
    image speed with the eye frozen vs free, compensatory then gaze shift.

    Everything plotted is the `pillar` measure -- retinal image speed of a real
    scene feature, parallax included -- not the `gaze` measure, which is the
    optic axis's rotation in the world and therefore assumes a scene at
    infinity.  Gaze-axis azimuth bars would also be circular: the movement
    classes are thresholds on dGaze, and the eye-free gaze-axis azimuth speed is
    dGaze, so a compensatory bar there restates the selection rule.

    Parameters
    ----------
    res : dict
        Output of analyze.
    out_png : str
    title : str
        Unused; kept for callers.
    """

    # Sized for reading off a slide or a printed page, not a zoomed-in screen.
    FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 11, 11, 11, 11
    CLASSES = (('compensatory', 'compensatory', "#b158c3"),
               ('gaze_shift', 'gaze-shifting', "#4cb6c7"))

    fig, axs = plt.subplots(2, 2, figsize=(9,7))
    keep = res['keep']

    # Movement classes, in the eye-vs-head plane the thresholds carve up.
    ax = axs[0, 0]
    m = keep & np.isfinite(res['dEye']) & np.isfinite(res['dHead'])
    sub = np.flatnonzero(m)[::max(1, int(m.sum()) // 20000)]
    ax.scatter(res['dHead'][sub], res['dEye'][sub], s=1, alpha=0.12, color='0.6')
    for nm, lbl, col in CLASSES:
        mm = keep & res[nm]
        s2 = np.flatnonzero(mm)[::max(1, int(mm.sum()) // 8000)]
        ax.scatter(res['dHead'][s2], res['dEye'][s2], s=3, alpha=0.3,
                   color=col, label=lbl)
    ax.axhline(0, color='k', lw=0.6); ax.axvline(0, color='k', lw=0.6)
    ax.set_xlabel('dHead (deg/s)', fontsize=FS_LABEL)
    ax.set_ylabel('dEye (deg/s)', fontsize=FS_LABEL)
    # set_aspect rather than axis('equal'): the latter widens the limits.
    ax.set_xlim(-600, 600); ax.set_ylim(-600, 600)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize=FS_LEGEND, markerscale=6)

    # Stabilization bars, azimuth only.
    ax = axs[0, 1]
    labels, vals, cis, cols = ['compensatory', 'gaze'], [], [], []
    for cname, lbl, _ in CLASSES:
        st = res['pillar_az_{}'.format(cname)]
        v = 100 * st.get('stabilization', np.nan)
        # labels.append('{}\nn={}'.format(lbl, st.get('n', 0)))
        vals.append(v); cis.append(st.get('ci'))
        cols.append('#b158c3' if (np.isfinite(v) and v > 0) else '#4cb6c7')
    ax.bar(range(len(vals)), vals, color=cols, width=0.6)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels)
    for i, ci in enumerate(cis):
        if ci is not None:
            ax.plot([i, i], [100 * ci[0], 100 * ci[1]], color='k', lw=1.8)
    ax.axhline(0, color='k', lw=1)
    ax.set_xlim(-0.6, len(vals) - 0.4)
    # Units: percent OF the eye-frozen speed.  +33 means that including the eye
    # makes the image sweep 33% more slowly (median).
    ax.set_ylabel('median horizontal image speed\nreduction with eye\n'
                  '(% of eye-frozen)', fontsize=FS_LABEL)
    # ax.set_title('+ = eye stabilizes, - = adds motion', fontsize=FS_TITLE)

    # Horizontal image speed histograms.  The `pillar` measure, not `gaze`:
    # gaze speed is blind to parallax, and translation alone sweeps the pillar
    # across the retina at v/d -- small next to a gaze shift but not next to the
    # compensatory class the test turns on.
    vr = res['pillar_az_retino']; ve = res['pillar_az_ego']
    for k, (cname, lbl, _) in enumerate(CLASSES):
        ax = axs[1, k]
        m = keep & res[cname] & np.isfinite(vr) & np.isfinite(ve)
        if m.sum() > 50:
            # Wide bins: these classes are thin.
            bins = np.linspace(0, 1100, 50) # 600 if cname == 'compensatory' else 1100, 50)
            for v, col, lab in ((ve[m], '#b2182b', 'eye frozen'),
                                (vr[m], '#2166ac', 'eye free')):
                # Clip into the top bin rather than let the range drop it:
                # density=True renormalizes over the given bins, so anything
                # past the last edge would vanish AND inflate the rest.
                ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins,
                        histtype='stepfilled', color=col,
                        density=True, label=lab, alpha=0.5)
                ax.set_xlim([0,1000])
            if k == 0:
                ax.legend(fontsize=FS_LEGEND)
        ax.set_ylabel('density (per deg/s)', fontsize=FS_LABEL)
        ax.set_xlabel('horizontal image speed (deg/s)', fontsize=FS_LABEL)
        ax.set_title('{}'.format(lbl), fontsize=FS_TITLE)

    for a in axs.ravel():
        a.tick_params(axis='both', labelsize=FS_TICK)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    fig.savefig('gaze_stabalization', dpi=130)
    plt.close(fig)
    print('  Wrote {}'.format(out_png))


def main():
    """ Command-line entry point. """

    p = argparse.ArgumentParser(
        description='Does the eye stabilize the reconstructed retinal image?')
    p.add_argument('--h5_path', required=True)
    p.add_argument('--npz_path', required=True,
                   help='the normal (eye-free) get_retinal_image output; the '
                        'egocentric control is derived analytically')
    p.add_argument('--out_prefix', default=None)
    p.add_argument('--speed_thresh', type=float, default=2.0)
    p.add_argument('--all_light_dark', action='store_true')
    args = p.parse_args()

    res = analyze(args.h5_path, args.npz_path, speed_thresh=args.speed_thresh,
                  light_only=not args.all_light_dark)
    for mname in ('gaze_az', 'pillar_az'):
        for cname in ('compensatory', 'gaze_shift', 'moving'):
            st = res[f'{mname}_{cname}']
            if 'stabilization' not in st:
                print('  {:<13} {:<13} n={} -- too few'.format(mname, cname, st.get("n", 0)))
                continue
            ci = st.get('ci')
            cis = (f'  [95% CI {100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]'
                   if ci else '')
            print('  {:<13} {:<13} n={:6d} fr / '
                  '{:5d} ev   {:6.1f} to '
                  '{:6.1f} deg/s   '
                  '{:+.1f}%{}'.format(mname, cname, st["n"], st["n_events"], st["speed_ego"], st["speed_retino"], 100 * st["stabilization"], cis))
    pre = args.out_prefix or args.h5_path.replace('_preproc.h5', '_gaze_stab')
    make_figure(res, pre + '.png', title=os.path.basename(args.h5_path))
    write_h5(pre + '.h5', {k: v for k, v in res.items()
                           if isinstance(v, np.ndarray)})


if __name__ == '__main__':
    
    main()


# python -m fm2p.gaze_stabilization --h5_path /home/dylan/Storage/freely_moving_data/V1PPC/cohort02_recordings/251016_DMM_DMM061_pos18/fm1/251016_DMM_DMM061_fm_01_preproc.h5 --npz_path /home/dylan/Storage/freely_moving_data/V1PPC/cohort02_recordings/251016_DMM_DMM061_pos18/fm1/251016_DMM_DMM061_fm_01_retinal_images.npz
