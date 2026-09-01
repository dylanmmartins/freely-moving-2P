
"""
Transform 3D pillar from world coordinates to mouse 2D retinal projection.
linear units in mm, ang in deg.

DMM April 2026
"""

import glob
import os
import subprocess
import sys
import time
from multiprocessing import Pool, cpu_count

import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

np.random.seed(0)

import fm2p

# The pillar is a cylinder, but it used to be modelled as its axis-aligned
# bounding box.  A box circumscribes the circle, so its silhouette is up to
# sqrt(2) wider than the cylinder's and the excess depends on viewing angle --
# measured against the annotated masks, the rendered pillar came out 1.29-1.32x
# too wide.  Sampling the actual circular cross-section removes that bias.
_N_CYL_SIDES = 16


def _cylinder_points(pillar_x, pillar_y, r, h, n=_N_CYL_SIDES):
    """(4, 2n) homogeneous points: n around the base circle, n around the top."""
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    cx = pillar_x + r * np.cos(a)
    cy = pillar_y + r * np.sin(a)
    return np.concatenate([
        np.stack([cx, cy, np.zeros(n), np.ones(n)]),
        np.stack([cx, cy, np.full(n, h), np.ones(n)]),
    ], axis=1)


def _cylinder_edges(n=_N_CYL_SIDES):
    """Ring + vertical edges, for near-plane clipping."""
    e = [(i, (i + 1) % n) for i in range(n)]
    e += [(n + i, n + (i + 1) % n) for i in range(n)]
    e += [(i, n + i) for i in range(n)]
    return e


_BBOX_EDGES = _cylinder_edges()
_Z_NEAR = 0.1   # mm -- near clipping plane in optical space

# Worldcam mount orientation in the head frame, azimuth then elevation.
# Fitted by maximising overlap with the hand-annotated polygon masks on
# 251028_DMM_DMM000_fm_04 (50/50 train/held-out), held-out aggregate IoU 0.231.
#
# An earlier fit put azimuth at +18 deg and could not distinguish a camera mount
# angle from a constant bias in head_yaw_deg.  It turned out to be neither: the
# gyro-integrated yaw was drifting -3.9 deg/s because the drift correction
# smoothed over 5 s, and the fit was absorbing that drift into the mount angle.
# With the correction window at 0.5 s (see detrend_gyroz_weighted_gaussian) the
# drift falls to -0.07 deg/s and the azimuth collapses to 0.00 deg -- i.e. the
# camera points straight ahead, which is what a forward-facing head mount should
# give.  Elevation (-4.1 -> -4.2) and focal length (1475 -> 1499) barely moved,
# so those were sound all along.
WORLDCAM_SOCKET_AZ_DEG = 0.0
WORLDCAM_SOCKET_EL_DEG = -4.58

# World-camera intrinsics.  The "worldcam" is the eye-camera optic pointed
# outward, so its field of view is far narrower than the modelled retina:
# ~24 deg against the retinal image's 120 deg.  Three independent estimates
# agree -- GT blob width against the known 4 cm pillar at the tracked distance
# (20.7 deg), the direct fit above (24.0 deg), and the scale the old
# scale+translate fit had been absorbing (60/2.628 = 22.8 deg).
# Overhead camera geometry, for the parallax correction below.
TOPDOWN_CAMERA_HEIGHT_MM = 42 * 25.4     # 42 inches above the arena floor
TOPDOWN_W, TOPDOWN_H     = 2448, 2048    # native topdown frame, px

# Scale on head pitch before it enters the head rotation.  1.0 = as measured.
# After the Euler fix a residual dependence of elevation error on pitch remains
# (slope +0.24 deg per deg, R2 0.23), i.e. pitch still over-applied ~24%; this
# is the knob for it.  Left at 1.0 until fitted on more than one session --
# a gain fitted to a single recording would bake that session's IMU
# calibration into every other one.
PITCH_GAIN = 1.0

# Extra smoothing applied on top of the Kalman filters that already produce
# these signals.  Both are OFF.
#
# The behaviour inputs are Kalman-filtered before they ever reach this module:
# yaw comes from _kalman_rts_upsample_angle inside
# detrend_gyroz_weighted_gaussian, and head position is fused by
# _imu_dlc_kalman_position below.  A median filter and a second RTS pass used to
# sit on top of that, so the model saw a signal smoothed three times.  It showed
# up as a reconstruction that lagged real movement and a top-down marker that
# slid rather than tracked.
#
# ANGLE_SMOOTH_Q: None disables the extra RTS pass; set a float (e.g. 1e-2) to
# re-enable it.  BEHAV_MEDFILT_N: 1 disables the median filter.
ANGLE_SMOOTH_Q = None
BEHAV_MEDFILT_N = 1

# Process-noise fraction for the IMU/DLC position fusion.  Higher = trusts the
# tracked anchors more = less smoothing.
#
# The old default of 1e-2 was far too stiff, for a structural reason: the filter
# scales Q by dt**2 but leaves R unscaled, so at a 62 Hz eye rate the effective
# ratio was R/Q ~ 19,000:1 and the fusion essentially ignored its own
# measurements.  Measured against the DLC anchors it was landing a median
# 114 px (3.3 cm) away from them and retaining only 8 of the 41.6 px of real
# inter-anchor motion -- the tracked point visibly slid instead of tracking.
#
#   Q_frac    median px to anchor    inter-anchor motion kept
#     0.01          114.4                    8.0 px
#     1.0            23.4                   25.3 px
#     10             7.9                   32.0 px
#     100            3.0                   37.8 px
#
# 10 keeps the trace within ~2 mm of every anchor while still rejecting some
# DLC jitter; 100 is nearer pure interpolation and passes the noise through.
POSITION_KALMAN_Q = 10.0

WORLDCAM_W, WORLDCAM_H = 640, 480
WORLDCAM_F_PX = 1584.0     # native px; HFOV 22.9 deg, VFOV 17.3 deg


def _clip_corners_to_near(corners_opt):
    """Return list of 3-vectors: in-front corners + near-plane intersection pts.

    corners_opt : (3, 8) array in optical frame.  Any corner with optical_z >
    _Z_NEAR is kept; edges that cross the near plane contribute an intersection
    point.  This ensures that when the viewer is inside the bounding box the
    projected hull covers the full visible extent instead of just the base.
    """
    pts = []
    for i in range(corners_opt.shape[1]):
        if corners_opt[2, i] > _Z_NEAR:
            pts.append(corners_opt[:, i])
    for a, b in _BBOX_EDGES:
        za, zb = corners_opt[2, a], corners_opt[2, b]
        if (za > _Z_NEAR) != (zb > _Z_NEAR):
            t = (_Z_NEAR - za) / (zb - za)
            pts.append(corners_opt[:, a] + t * (corners_opt[:, b] - corners_opt[:, a]))
    return pts


def simulate_retinal_projection(
    pillar_x, pillar_y, mouse_x, mouse_y, mouse_yaw, mouse_pitch, mouse_roll,
    pupil_tilt_h, pupil_tilt_v,
    pillar_h=210.0, pillar_d=40.0,
    eye_offset_x=3.5, eye_offset_y=-5.0, eye_offset_z=3.5,
    anatomical_eye_angle_deg=-65.0,
    fixed_eye=False,
    fixed_pitch=False,
    fixed_roll=False,
    resting_tilt_h=0.0,
    resting_tilt_v=0.0,
    resting_pitch=0.0,
    resting_roll=0.0,
    worldcam=False,
    worldcam_socket_deg=WORLDCAM_SOCKET_AZ_DEG,
    worldcam_elev_deg=WORLDCAM_SOCKET_EL_DEG,
    socket_elev_deg=0.0,
    pitch_gain=PITCH_GAIN,
):
    # worldcam: camera at head centre, no eye offset and no pupil movement.
    # The mount orientation is a azimuth/elevation pair in the head frame (see
    # WORLDCAM_SOCKET_AZ_DEG / WORLDCAM_SOCKET_EL_DEG).  It was formerly
    # hard-coded to -90 deg azimuth to make the retinal image agree with the
    # panoramic diagnostic's -90 deg azimuth offset; that pointed the virtual
    # camera 90 deg to the animal's right and left the reconstruction
    # anti-correlated with the hand-annotated masks (MCC -0.41).
    if worldcam:
        eye_offset_x = eye_offset_y = eye_offset_z = 0.0
        anatomical_eye_angle_deg = worldcam_socket_deg
        socket_elev_deg = worldcam_elev_deg
        pupil_tilt_h = pupil_tilt_v = 0.0

    # fixed_eye: camera glued to the head axis.  This used only to freeze the
    # pupil at its resting position while leaving the -65 deg anatomical socket
    # in place, so the view still pointed 65 deg to the animal's right -- with
    # a 120 deg field that puts head-forward off the edge of the image, which
    # reads as "the reconstruction ignores the heading".  Zeroing the socket as
    # well makes this the exact geometry the worldcam path uses, so it is the
    # control for separating head-pose errors from eye errors on any session.
    if fixed_eye:
        eye_offset_x = eye_offset_y = eye_offset_z = 0.0
        anatomical_eye_angle_deg = 0.0
        socket_elev_deg = 0.0
        pupil_tilt_h = pupil_tilt_v = 0.0
    if fixed_pitch:
        mouse_pitch = resting_pitch
    if fixed_roll:
        mouse_roll = resting_roll

    fov_deg = 120.0
    res_w, res_h = 120, 120 # 1 pixel is 1 visual deg
    
    # focal length
    f = (res_w / 2) / np.tan(np.radians(fov_deg / 2))
    cx, cy = res_w / 2, res_h / 2
    
    # pinhole mdoel
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1]
    ])

    head_z = 25.0 # head height

    r = pillar_d / 2.0
    # Points around the cylinder itself, not its bounding box (see _BBOX_EDGES).
    corners_world = _cylinder_points(pillar_x, pillar_y, r, pillar_h)

    # Yaw, pitch, roll as an INTRINSIC Z-Y'-X'' sequence: R = Rz(yaw) Ry(pitch)
    # Rx(roll), i.e. pitch about the head's own (already yawed) lateral axis.
    # SciPy spells intrinsic with upper case.  This was lower-case 'zyx' --
    # extrinsic, Rx(roll) Ry(pitch) Rz(yaw) -- which pitches about the fixed
    # world axis instead.  The two agree only near zero pitch/roll; at the
    # +-24 deg pitch and -26 deg roll seen here they do not.  Measured against
    # the annotated masks, the fix cut elevation MAD 2.35 -> 1.77 deg and
    # halved the residual pitch dependence (slope 0.346 -> 0.239, R2 0.39 ->
    # 0.23).  See _pillar_optical_corners for the matching change.
    R_head = R.from_euler('ZYX', [mouse_yaw, mouse_pitch * pitch_gain, mouse_roll],
                          degrees=True).as_matrix()
    t_head = np.array([[mouse_x], [mouse_y], [head_z]])
    
    M_head = np.eye(4)
    M_head[:3, :3] = R_head
    M_head[:3, 3:] = t_head
    
    # invert so world -> head
    M_head_inv = np.linalg.inv(M_head)
    corners_head = M_head_inv @ corners_world

    # Saccade rotation (dynamic, eye-in-socket).  Still extrinsic: unlike the
    # head rotation above this one has NOT been validated against ground truth
    # (worldcam mode zeroes both tilts), so it is left as-is rather than changed
    # blind.  Worth revisiting for the eye-camera path.
    R_saccade = R.from_euler('zyx', [pupil_tilt_h, pupil_tilt_v, 0], degrees=True).as_matrix()
    # Anatomical socket rotation (fixed)
    # Socket: azimuth about head +Z, then elevation about the yawed lateral
    # axis (intrinsic Z-Y').  Built explicitly rather than via a sequence string
    # so the composition order is unambiguous.
    R_socket = (R.from_euler('z', anatomical_eye_angle_deg, degrees=True).as_matrix()
                @ R.from_euler('y', socket_elev_deg, degrees=True).as_matrix())
    R_eye = R_socket @ R_saccade  # Compose fixed socket rotation and dynamic saccade
    t_eye = np.array([[eye_offset_x], [eye_offset_y], [eye_offset_z]])
    
    M_eye = np.eye(4)
    M_eye[:3, :3] = R_eye
    M_eye[:3, 3:] = t_eye
    
    # invert so head -> eye
    M_eye_inv = np.linalg.inv(M_eye)
    corners_eye = M_eye_inv @ corners_head

    # optical_X = +eye_Y: rightward objects map to the right side of the image
    # (right eye: temporal field on right, nasal on left)
    R_align = np.array([
        [ 0, -1,  0], # tried changing to [ 0,  1,  0],
        [ 0,  0, -1],
        [ 1,  0,  0]
    ])

    corners_optical = R_align @ corners_eye[:3, :]

    retina_image = np.zeros((res_h, res_w), dtype=np.uint8)

    clipped = _clip_corners_to_near(corners_optical)
    if clipped:
        points_2d = []
        for pt_3d in clipped:
            pt_proj = K @ pt_3d
            u = int(np.clip(pt_proj[0] / pt_proj[2], -10000, 10000))
            v = int(np.clip(pt_proj[1] / pt_proj[2], -10000, 10000))
            points_2d.append([u, v])
        pts_arr = np.array(points_2d, dtype=np.int32)
        hull = cv2.convexHull(pts_arr)
        cv2.fillPoly(retina_image, [hull], color=255)

    return retina_image


def parallax_correct_px(x_px, y_px, object_height_mm,
                       camera_height_mm=TOPDOWN_CAMERA_HEIGHT_MM,
                       principal_point=None):
    """Pull tracked points in to their floor-plane position.

    The overhead camera is a pinhole, not telecentric, so a marker sitting
    `object_height_mm` above the floor images further from the optical axis
    than the floor point beneath it.  Treating that pixel as a floor point puts
    the animal too far out, by a factor that grows with eccentricity:

        r_floor = r_measured * (H - h) / H

    With H = 42 in and head markers at 25 mm this is only a 2.3 % pull-in --
    0.8 deg of pillar bearing across the observed eccentricity range, against
    the ~12.5 deg of position-dependent azimuth error actually measured.  So it
    is applied because it is correct, not because it explains that error; the
    dominant position-dependent term is elsewhere (see fit_pillar_position).
    """
    if principal_point is None:
        principal_point = (TOPDOWN_W / 2.0, TOPDOWN_H / 2.0)
    cx, cy = principal_point
    s = (camera_height_mm - object_height_mm) / float(camera_height_mm)
    return cx + (np.asarray(x_px, float) - cx) * s, \
           cy + (np.asarray(y_px, float) - cy) * s


def _opt_medfilt(x, n):
    """Median filter, or a plain float copy when n <= 1 (filter disabled)."""
    x = np.asarray(x, dtype=float)
    if n is None or int(n) <= 1:
        return x.squeeze()
    return fm2p.nanmedfilt(x, int(n)).squeeze()


# Gaussian half-width, in seconds, of the drift correction applied to the
# integrated gyro.  It sets the crossover of the complementary filter: below it
# the yaw follows the gyro, above it the top-down ear vector.  Held-out
# cross-validation over 37 sessions (top-down anchors split into 4 s blocks,
# one block in four withheld) is flat from 0.05 to ~1 s and degrades beyond
# that, so the exact value does not matter much inside that range.
YAW_FUSION_SIGMA_S = 0.5


def _fuse_head_yaw(gyro_z, imuT, cam_yaw_deg, camT, sigma_s=YAW_FUSION_SIGMA_S):
    """Allocentric yaw on the IMU clock, from gyro-z plus top-down anchors.

    The gyro is precise but its integral drifts; the top-down ear vector is
    drift-free but noisy and gappy.  Combining them is a complementary filter:
    integrate the gyro, measure the residual against the camera at the anchor
    times, low-pass that residual, and add it back.

    This recomputes what `upsampled_yaw/igyro_corrected_deg` holds, and it is
    worth recomputing for two reasons.

    First, the stored copy is only as current as the preprocessing run that
    wrote it, and several of the inputs have changed (the top-down median
    filter, the drift-correction window, the top-down timebase).  Across the 37
    sessions in cohort02 the stored yaw sits a median 22.2 deg from the ear
    vector where a fresh fusion of the same h5 sits 18.0 deg from it.

    Second, `detrend_gyroz_weighted_gaussian` forms the residual against a
    Kalman-upsampled *copy* of the camera yaw resampled onto the IMU grid,
    which adds its own lag and jitter before the residual is even taken.  Doing
    it at the anchor times instead is both simpler and measurably better: on a
    held-out quarter of the anchors (4 s blocks, so DLC error is decorrelated
    between train and test), median error 18.0 vs 27.9 deg and 90th percentile
    76.7 vs 116.2 deg, better on 33 of 37 sessions.

    Outlier rejection on the anchors was tried and dropped.  Gating on a robust
    running median of the residual improves the median slightly and costs a lot
    at the tail, because every rejected anchor is an interval the gyro has to
    free-run across: on 251016_DMM_DMM061_pos18 it discarded 39 % of the
    anchors and took the 90th percentile from 22 to 137 deg.

    Parameters
    ----------
    gyro_z : array
        Head yaw rate on the IMU clock, deg/s (`gyro_z_trim`).
    imuT : array
        IMU timestamps, s (`imuT_trim`).
    cam_yaw_deg : array
        Top-down ear-vector yaw, deg (`head_yaw_deg`), NaN where untracked.
    camT : array
        Timestamps for `cam_yaw_deg`, s (`twopT`).

    Returns
    -------
    yaw_deg : np.ndarray
        Fused yaw in [0, 360) at `imuT`, or None if there is nothing to fuse.
    info : dict
        'n_anchors', 'drift_total_deg' (net correction applied over the
        recording) and 'drift_rate_deg_s'.
    """
    imuT = np.asarray(imuT, dtype=float)
    gyro_z = np.asarray(gyro_z, dtype=float)
    if len(imuT) < 2 or len(gyro_z) != len(imuT):
        return None, {}
    if not np.isfinite(gyro_z).any():
        return None, {}

    imuT = fm2p.nan_interp(imuT) if not np.isfinite(imuT).all() else imuT
    dt = float(np.nanmedian(np.diff(imuT)))
    if not np.isfinite(dt) or dt <= 0:
        return None, {}

    # cumsum turns a single NaN into an all-NaN tail, so bridge the gaps first.
    gz = gyro_z.copy()
    if np.isnan(gz).any():
        gz = fm2p.nan_interp(gz)
    yaw_gyro = np.cumsum(gz) * dt          # unwrapped, arbitrary origin

    camT = np.asarray(camT, dtype=float)
    cam = np.asarray(cam_yaw_deg, dtype=float)
    m = min(len(camT), len(cam))
    camT, cam = camT[:m], cam[:m]
    ok = np.isfinite(cam) & np.isfinite(camT)
    if ok.sum() < 5:
        return None, {}

    tk = camT[ok]
    resid = ((cam[ok] - np.interp(tk, imuT, yaw_gyro)) + 180.0) % 360.0 - 180.0
    resid = np.degrees(np.unwrap(np.radians(resid)))

    drift = np.interp(imuT, tk, resid)
    if sigma_s and sigma_s > 0:
        drift = gaussian_filter1d(drift, sigma_s / dt, mode='nearest')

    span = float(imuT[-1] - imuT[0])
    info = dict(
        n_anchors=int(ok.sum()),
        drift_total_deg=float(drift[-1] - drift[0]),
        drift_rate_deg_s=float((drift[-1] - drift[0]) / span) if span > 0 else np.nan,
    )
    return (yaw_gyro + drift) % 360.0, info


def _read_pillar_centroid_px(f):

    import h5py as _h5py
    try:
        px = f['pillar_x']
        if isinstance(px, _h5py.Group):
            pillar_x_px = float(np.mean([float(px[str(k)][()]) for k in range(len(px))]))
            pillar_y_px = float(np.mean([float(f['pillar_y'][str(k)][()]) for k in range(len(f['pillar_y']))]))
        else:
            pillar_x_px = float(np.mean(px[:]))
            pillar_y_px = float(np.mean(f['pillar_y'][:]))
    except (KeyError, ValueError):
        pillar_x_px = float(f['pillar_centroid']['x'][()])
        pillar_y_px = float(f['pillar_centroid']['y'][()])
    return pillar_x_px, pillar_y_px


def get_retinal_image(
    h5_path,
    out_npz=None,
    eye_offset_x=3.5,
    eye_offset_y=-5.0,
    eye_offset_z=3.5,
    pillar_d=None,
    pillar_h=210.0,
    ang_imoffset_override=None,
    anatomical_eye_angle_deg=-65.0,
    fixed_eye=False,
    fixed_pitch=False,
    fixed_roll=False,
    arena_width_cm=None,
    worldcam=False,
    worldcam_video_path=None,
    worldcam_socket_deg=WORLDCAM_SOCKET_AZ_DEG,
    worldcam_elev_deg=WORLDCAM_SOCKET_EL_DEG,
    parallax_correct=True,
    angle_smooth_Q=ANGLE_SMOOTH_Q,
    medfilt_n=BEHAV_MEDFILT_N,
    position_kalman_Q=POSITION_KALMAN_Q,
    camera_height_mm=TOPDOWN_CAMERA_HEIGHT_MM,
    head_z_mm=25.0,
    pillar_parallax_height_mm=0.0,
    refit_yaw=True,
    yaw_sigma_s=YAW_FUSION_SIGMA_S,
):

    _root = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from fm2p.utils.ref_frame import calc_vor_eye_offset

    with h5py.File(h5_path, 'r') as f:
        eyeT        = f['eyeT_trim'][:]
        theta_trim  = f['theta_trim'][:]
        phi_trim    = f['phi_trim'][:]
        pitch_eye   = _opt_medfilt(f['pitch_eye_interp'][:], medfilt_n)
        roll_eye    = _opt_medfilt(f['roll_eye_interp'][:], medfilt_n)
        gyro_z_eye  = f['gyro_z_eye_interp'][:]

        imuT        = f['imuT_trim'][:]
        yaw_imu     = _opt_medfilt(f['upsampled_yaw']['igyro_corrected_deg'][:], medfilt_n)
        # Raw yaw rate, so the drift correction can be redone here rather than
        # inherited from whenever preprocessing last ran (see _fuse_head_yaw).
        gyro_z_imu  = f['gyro_z_trim'][:] if 'gyro_z_trim' in f else None

        twopT       = f['twopT'][:]
        head_x_2p   = f['head_x'][:]
        head_y_2p   = f['head_y'][:]
        nose_likelihood_2p = f['nose_likelihood'][:] if 'nose_likelihood' in f else None
        # Topdown ear-vector yaw at the 2P rate.  Only used as a fallback when
        # the gyro-integrated yaw is missing, and as the reference the fused
        # yaw is checked against below.
        head_yaw_2p = (f['head_yaw_deg'][:][:len(twopT)]
                       if 'head_yaw_deg' in f else None)

        pxls2cm        = float(f['pxls2cm'][()])
        eyeT_startInd  = int(f['eyeT_startInd'][()]) if 'eyeT_startInd' in f else 0
        eyeT_endInd    = int(f['eyeT_endInd'][()])   if 'eyeT_endInd'   in f else eyeT_startInd + len(eyeT)
        pillar_x_px, pillar_y_px = _read_pillar_centroid_px(f)
        _pillar_rad_px = float(f['pillar_radius'][()]) if 'pillar_radius' in f else None
        if arena_width_cm is not None:
            try:
                arenaTL_x = float(f['arenaTL']['x'][()])
                arenaTR_x = float(f['arenaTR']['x'][()])
                arenaBL_x = float(f['arenaBL']['x'][()])
                arenaBR_x = float(f['arenaBR']['x'][()])
                px_width = np.mean([arenaTR_x - arenaTL_x, arenaBR_x - arenaBL_x])
                pxls2cm = px_width / arena_width_cm
                print(f'  Overriding pxls2cm with value calculated from arena_width_cm={arena_width_cm:.1f} -> {pxls2cm:.3f}')
            except KeyError:
                print('  WARNING: Could not find arena corner keys in HDF5 to override pxls2cm.')

    px2mm       = 10.0 / pxls2cm

    if parallax_correct:
        # Head markers ride at head height, so they need the correction.  The
        # pillar defaults to 0 (no correction): its annotated centre is off by
        # ~32 mm on the reference session, but the fitted correction does not
        # point along the parallax direction, so that error is hand-annotation
        # on an obliquely-viewed cylinder rather than optics.  Fit the pillar
        # position per session instead of modelling it -- and note it trades off
        # against the mount azimuth, so fit one with the other held fixed.
        _pcx, _pcy = parallax_correct_px(pillar_x_px, pillar_y_px,
                                         pillar_parallax_height_mm,
                                         camera_height_mm=camera_height_mm)
        print(f'  Parallax: pillar ({pillar_x_px:.1f}, {pillar_y_px:.1f}) -> '
              f'({float(_pcx):.1f}, {float(_pcy):.1f}) px '
              f'(h={pillar_parallax_height_mm:.0f} mm, cam={camera_height_mm:.0f} mm)')
        pillar_x_px, pillar_y_px = float(_pcx), float(_pcy)

    pillar_x_mm =  pillar_x_px * px2mm
    pillar_y_mm = -pillar_y_px * px2mm

    if _pillar_rad_px is not None:
        pillar_d_mm = 2.0 * _pillar_rad_px * px2mm
        if pillar_d is None:
            print(f'  pillar_d from h5 pillar_radius: {pillar_d_mm:.1f} mm')
            pillar_d = pillar_d_mm
        else:
            # An explicitly supplied physical diameter is the measurement;
            # the annotation polygon is drawn generously around the pillar and
            # over-estimates it (54 mm vs a true 40 mm on the reference session).
            print(f'  pillar_d: using supplied {pillar_d:.1f} mm '
                  f'(h5 pillar_radius implies {pillar_d_mm:.1f} mm)')
    elif pillar_d is None:
        pillar_d = 40.0

    valid_hx = head_x_2p[np.isfinite(head_x_2p)]
    valid_hy = head_y_2p[np.isfinite(head_y_2p)]
    print(f'  pxls2cm={pxls2cm:.3f}  px2mm={px2mm:.4f}')
    print(f'  pillar_d={pillar_d:.1f} mm  pillar_h={pillar_h:.1f} mm')
    print(f'  Pillar centroid: ({pillar_x_px:.1f}, {pillar_y_px:.1f}) px  '
          f'->  ({pillar_x_mm:.1f}, {pillar_y_mm:.1f}) mm')
    print(f'  Head X range: [{valid_hx.min():.1f}, {valid_hx.max():.1f}] px  '
          f'->  [{valid_hx.min()*px2mm:.1f}, {valid_hx.max()*px2mm:.1f}] mm')
    print(f'  Head Y range: [{valid_hy.min():.1f}, {valid_hy.max():.1f}] px  '
          f'->  [{valid_hy.min()*px2mm:.1f}, {valid_hy.max()*px2mm:.1f}] mm')
    pillar_in_x = valid_hx.min() <= pillar_x_px <= valid_hx.max()
    pillar_in_y = valid_hy.min() <= pillar_y_px    <= valid_hy.max()
    if not (pillar_in_x and pillar_in_y):
        print(f'  WARNING: pillar is OUTSIDE the head position range -- '
              f'coordinate system mismatch? (in_x={pillar_in_x}, in_y={pillar_in_y})')

    fps_eye    = float(1.0 / np.nanmedian(np.diff(eyeT)))
    vor        = calc_vor_eye_offset(theta_trim, None, fps_eye, head_vel_deg_s=gyro_z_eye)
    ang_offset = float(vor['ang_offset_vor_null'])
    print(f'ang_offset (VOR null): {ang_offset:.2f} deg  |  VOR gain: {vor["vor_gain"]:.3f}')

    # Compute resting (mean) values for fixed_* modes
    mean_pfh   = float(np.nanmean(ang_offset - theta_trim))
    mean_phi   = float(np.nanmean(phi_trim))
    mean_pitch = float(np.nanmean(pitch_eye))
    mean_roll  = float(np.nanmean(roll_eye))
    print(f'  Resting eye: tilt_h={mean_pfh:.2f} deg  tilt_v={mean_phi:.2f} deg  '
          f'pitch={mean_pitch:.2f} deg  roll={mean_roll:.2f} deg')
    if fixed_eye:
        print('  fixed_eye=True: camera on the head axis '
              '(socket az/el = 0, no eye offset, no pupil movement) -- '
              'same geometry as worldcam mode')

    N = len(eyeT)

    # Re-derive the allocentric yaw from gyro-z and the top-down ear vector.
    # The stored upsampled_yaw is whatever preprocessing produced at the time,
    # and it is measurably worse than a fresh fusion of the same inputs -- see
    # _fuse_head_yaw for the held-out numbers.
    if refit_yaw and gyro_z_imu is not None and head_yaw_2p is not None:
        _yaw_new, _yinfo = _fuse_head_yaw(gyro_z_imu, imuT, head_yaw_2p, twopT,
                                          sigma_s=yaw_sigma_s)
        if _yaw_new is not None:
            _d_old = np.abs(((np.interp(twopT, imuT, yaw_imu) - head_yaw_2p
                              + 180.0) % 360.0) - 180.0)
            _d_new = np.abs(((np.interp(twopT, imuT, _yaw_new) - head_yaw_2p
                              + 180.0) % 360.0) - 180.0)
            yaw_imu = _yaw_new
            print(f'  Yaw refit from gyro_z_trim + head_yaw_deg '
                  f'(sigma={yaw_sigma_s:g} s, {_yinfo["n_anchors"]} anchors, '
                  f'drift removed {_yinfo["drift_total_deg"]:+.0f} deg = '
                  f'{_yinfo["drift_rate_deg_s"]:+.2f} deg/s)')
            print(f'    |yaw - head_yaw_deg| median: stored '
                  f'{np.nanmedian(_d_old):.1f} deg -> refit '
                  f'{np.nanmedian(_d_new):.1f} deg')
        else:
            print('  Yaw refit skipped (no usable gyro_z_trim / head_yaw_deg); '
                  'using the stored upsampled_yaw')
    elif not refit_yaw:
        print('  Yaw refit disabled: using the stored upsampled_yaw')

    yaw_eye = np.interp(eyeT, imuT, yaw_imu)

    # ---- Pose sanity checks -------------------------------------------------
    # Every one of these used to fail silently: a NaN yaw became yaw=0 in the
    # render loop (a static east-facing view for the whole recording), and a
    # topdown trace that covers a fraction of the eye timebase just produced
    # black frames.  Both are indistinguishable from "the reconstruction is
    # wrong" when you are looking at the output video, so say so out loud.
    _yaw_bad = ~np.isfinite(yaw_eye)
    if _yaw_bad.any():
        print(f'  WARNING: gyro-integrated yaw is non-finite on '
              f'{_yaw_bad.mean() * 100:.1f}% of eye frames '
              f'({int(_yaw_bad.sum())}/{N}).')
        if head_yaw_2p is not None and np.isfinite(head_yaw_2p).sum() >= 2:
            _ok2p = np.isfinite(head_yaw_2p)
            _dlc_at_eye = np.interp(
                eyeT, twopT[_ok2p],
                np.unwrap(np.radians(head_yaw_2p[_ok2p])) * 180.0 / np.pi,
                left=np.nan, right=np.nan) % 360.0
            yaw_eye = np.where(_yaw_bad, _dlc_at_eye, yaw_eye)
            _still_bad = ~np.isfinite(yaw_eye)
            print(f'           Filled from topdown head_yaw_deg; '
                  f'{_still_bad.mean() * 100:.1f}% still non-finite '
                  f'(those frames are skipped, not rendered at yaw=0).')
        else:
            print('           No usable head_yaw_deg to fall back on -- '
                  'those frames will be SKIPPED, not rendered at yaw=0. '
                  'Re-run preprocessing: gyro_z_trim with NaN gaps used to '
                  'integrate to an all-NaN yaw (fixed in imu.py).')

    # Does the topdown trace that anchors both position and yaw actually cover
    # the window we are about to render?
    if head_yaw_2p is not None:
        _cov = float(np.isfinite(head_yaw_2p).mean())
        if _cov < 0.5:
            print(f'  WARNING: head_yaw_deg is {(1 - _cov) * 100:.1f}% NaN '
                  f'({int(np.isfinite(head_yaw_2p).sum())}/{len(head_yaw_2p)} '
                  f'tracked frames). The gyro drift correction is anchored to '
                  f'this trace, so the yaw driving the render is close to a '
                  f'free-running integral. Fix the topdown DLC tracking first.')
    _wrong_video = False
    _span_2p, _span_eye = float(twopT[-1] - twopT[0]), float(eyeT[-1] - eyeT[0])
    if _span_2p < 0.9 * _span_eye:
        print(f'  WARNING: topdown/2P timebase spans {_span_2p:.0f} s but the '
              f'eye timebase spans {_span_eye:.0f} s. Only the overlapping '
              f'{_span_2p / _span_eye * 100:.0f}% can be rendered; the rest '
              f'has no head position and comes out blank.')
        # On a sound session these two spans agree to within a second, because
        # the top-down camera and the eye camera run for the same recording.
        # A large mismatch usually means preprocessing tracked the WRONG VIDEO:
        # the topdown_DLC_search_key is a network-name glob, so it also matches
        # the DLC output for diagnostic MP4s written into the same folder, and
        # `find(..., MR=True)` took whichever was newest.  Everything downstream
        # is then tracking of a figure panel, and the 2P arrays were truncated
        # to match it.  The give-away is that the head coordinates fit inside a
        # figure rather than the 2448x2048 arena frame.
        _hx_hi = float(np.nanmax(head_x_2p)) if np.isfinite(head_x_2p).any() else np.nan
        _hy_hi = float(np.nanmax(head_y_2p)) if np.isfinite(head_y_2p).any() else np.nan
        if np.isfinite(_hx_hi) and _hx_hi < 0.75 * TOPDOWN_W and _hy_hi < 0.75 * TOPDOWN_H:
            _wrong_video = True
            print(f'           head_x/head_y peak at ({_hx_hi:.0f}, {_hy_hi:.0f}) px, '
                  f'well inside the {TOPDOWN_W}x{TOPDOWN_H} top-down frame. '
                  f'Together with the span mismatch and {len(twopT)} tracked '
                  f'frames this is the signature of tracking run on a DIAGNOSTIC '
                  f'video rather than the arena camera. Check which .h5 '
                  f'topdown_DLC_search_key resolved to, delete the DLC output '
                  f'for the diagnostic MP4s in that folder, and re-preprocess '
                  f'(fixed in preprocess._resolve_topdown_dlc).')

    # How far the yaw the renderer uses sits from the topdown ear vector.  On a
    # sound session this is a few degrees (4.4 deg median on the worldcam
    # reference); tens of degrees means the drift correction has lost its
    # anchors and the reconstruction will not track the animal.
    if head_yaw_2p is not None and np.isfinite(head_yaw_2p).any():
        _yaw_at_2p = np.interp(twopT, eyeT, yaw_eye)
        _dyaw = np.abs(((_yaw_at_2p - head_yaw_2p + 180.0) % 360.0) - 180.0)
        if np.isfinite(_dyaw).any():
            _p50, _p90 = np.nanpercentile(_dyaw, [50, 90])
            _flag = '  <-- reconstruction will not track the animal' if _p50 > 20 else ''
            print(f'  Yaw check vs topdown head_yaw_deg: median {_p50:.1f} deg, '
                  f'p90 {_p90:.1f} deg{_flag}')

    # Idea 2: pre-smooth DLC anchor positions at 2P rate before fusion
    if medfilt_n is not None and int(medfilt_n) > 1:
        head_x_2p_sm, head_y_2p_sm = _smooth_2p_positions(head_x_2p, head_y_2p,
                                                          window=int(medfilt_n))
    else:
        # Straight to the Kalman fusion -- it is the only smoothing position gets.
        head_x_2p_sm = np.asarray(head_x_2p, dtype=float)
        head_y_2p_sm = np.asarray(head_y_2p, dtype=float)

    # Idea 3+5: IMU-DLC Kalman fusion with optional adaptive R from DLC confidence.
    # Idea 1 (RTS backward smoothing pass) is applied inside _imu_dlc_kalman_position.
    _n_dlc_meas = int(np.sum(np.isfinite(head_x_2p)))
    _lik_msg = ', DLC confidence available' if nose_likelihood_2p is not None else ', no DLC confidence'
    print(f'  Position fusion: {_n_dlc_meas} valid DLC anchor frames{_lik_msg}')
    if parallax_correct:
        _hx, _hy = parallax_correct_px(head_x_2p_sm, head_y_2p_sm, head_z_mm,
                                       camera_height_mm=camera_height_mm)
        _shift = np.nanmedian(np.hypot(_hx - head_x_2p_sm, _hy - head_y_2p_sm))
        print(f'  Parallax: head trace pulled in by a median {_shift:.1f} px '
              f'({_shift * px2mm:.1f} mm)')
        head_x_2p_sm, head_y_2p_sm = _hx, _hy

    head_x_eye, head_y_eye = _imu_dlc_kalman_position(
        head_x_2p_sm * px2mm,
        -head_y_2p_sm * px2mm,   # flip image y-down -> math y-up
        twopT,
        eyeT,
        nose_likelihood_2p=nose_likelihood_2p,
        Q_frac=position_kalman_Q,
    )
    print(f'  Position fusion Q_frac={position_kalman_Q:g}')

    # The fused trace expressed back in native top-down pixels, so the
    # diagnostic video can draw the position the renderer actually used instead
    # of the raw 7.5 Hz DLC anchors.  Undo the y-flip, the mm scaling and the
    # parallax pull-in of section 3.4, so the point lands on the animal in the
    # video frame rather than on its floor projection.
    _hx_px = np.asarray(head_x_eye, dtype=float) / px2mm
    _hy_px = -np.asarray(head_y_eye, dtype=float) / px2mm
    if parallax_correct:
        _cx0, _cy0 = TOPDOWN_W / 2.0, TOPDOWN_H / 2.0
        _s = (camera_height_mm - head_z_mm) / float(camera_height_mm)
        _hx_px = _cx0 + (_hx_px - _cx0) / _s
        _hy_px = _cy0 + (_hy_px - _cy0) / _s
    _cov_raw = float(np.isfinite(head_x_2p).mean())
    _cov_fus = float(np.isfinite(_hx_px).mean())
    print(f'  Head position coverage: {_cov_raw * 100:.1f}% of top-down frames '
          f'tracked -> {_cov_fus * 100:.1f}% of rendered frames after fusion')
    if _cov_raw < 0.5:
        # head_x/head_y are the midpoint of the two ear bars, so they exist only
        # where BOTH bars clear cfg['likelihood_thresh'].  That is the usual
        # reason the top-down overlay is empty, and it is a tracking problem,
        # not a reconstruction one -- say which marker is failing.
        _lik = {}
        with h5py.File(h5_path, 'r') as _f:
            for _k in ('leftbar_likelihood', 'rightbar_likelihood',
                       'nose_likelihood'):
                if _k in _f:
                    _lik[_k] = np.asarray(_f[_k][:], dtype=float).ravel()
        if _lik:
            print('    DLC likelihood on the markers head_x/head_y are built '
                  'from (median, frac > 0.9):')
            for _k, _v in _lik.items():
                print(f'      {_k:<22} {np.nanmedian(_v):.3f}   '
                      f'{np.mean(_v > 0.9) * 100:5.1f}%')
            _bars = [v for k, v in _lik.items() if 'bar' in k]
            if _wrong_video:
                # These likelihoods describe how confidently the network found
                # "ear bars" in a figure panel, so they say nothing about the
                # top-down tracking.  Do not read them as a tracking problem --
                # on the real video for this session the bars clear 0.7 on 95%
                # of frames.
                print('    NOTE: these likelihoods are meaningless while the '
                      'tracking comes from the wrong video (see above). They '
                      'describe the network run on a figure, not on the animal. '
                      'Re-preprocess before drawing any conclusion about '
                      'top-down tracking quality.')
            elif _bars and 'nose_likelihood' in _lik:
                _bar_med = float(np.nanmedian([np.nanmedian(b) for b in _bars]))
                _nose_med = float(np.nanmedian(_lik['nose_likelihood']))
                if _bar_med < 0.7 <= _nose_med:
                    print(f'    The ear bars are the failing markers (median '
                          f'{_bar_med:.2f}, peak '
                          f'{max(float(np.nanmax(b)) for b in _bars):.2f}) while '
                          f'the nose tracks fine (median {_nose_med:.2f}). '
                          f'head_x, head_y AND head_yaw_deg are all built from '
                          f'the two bars and from nothing else, so they drop out '
                          f'together and the nose cannot rescue them. Retrain '
                          f'the top-down network on the bars, or lower '
                          f'likelihood_thresh in the run config '
                          f'(config/config.yaml; fm2p/utils/internals.yaml is '
                          f'only the fallback) and re-preprocess.')

    pfh_eye = ang_offset - theta_trim

    # Idea 1: RTS smoother (replaces forward-only Kalman) for angular signals
    _kfps = fps_eye
    _q = angle_smooth_Q
    if _q is not None:
        yaw_eye   = _kalman_rts_smooth(yaw_eye,   _kfps, Q_frac=_q)
        pitch_eye = _kalman_rts_smooth(pitch_eye, _kfps, Q_frac=_q)
        roll_eye  = _kalman_rts_smooth(roll_eye,  _kfps, Q_frac=_q)
        pfh_eye   = _kalman_rts_smooth(pfh_eye,   _kfps, Q_frac=_q)
        phi_trim  = _kalman_rts_smooth(phi_trim,  _kfps, Q_frac=_q)
        print(f'  Angular smoothing: extra RTS pass Q_frac={_q:.0e}, '
              f'median filter n={medfilt_n}')
    else:
        print(f'  Angular smoothing: OFF (upstream Kalman only), '
              f'median filter n={medfilt_n}')

    retinal_images = np.zeros((N, 120, 120), dtype=np.uint8)

    n = min(len(head_x_eye), len(yaw_eye), len(pitch_eye), len(roll_eye),
            len(pfh_eye), len(phi_trim), N)

    print(f'Rendering {n} retinal frames at {fps_eye:.1f} Hz ...')
    t0 = time.time()
    _n_skip_pos = _n_skip_yaw = 0
    for i in range(n):
        mx = float(head_x_eye[i])
        my = float(head_y_eye[i])
        if not np.isfinite(mx) or not np.isfinite(my):
            _n_skip_pos += 1
            continue
        # A missing yaw is a missing pose, not a heading of zero.  Rendering it
        # at yaw=0 put a whole recording's worth of frames facing due east.
        if not np.isfinite(yaw_eye[i]):
            _n_skip_yaw += 1
            continue

        yaw    = -float(yaw_eye[i])
        p      = float(pitch_eye[i]) if np.isfinite(pitch_eye[i]) else mean_pitch
        r_     = float(roll_eye[i])  if np.isfinite(roll_eye[i])  else mean_roll
        # pfh_eye = ang_offset - theta, so its resting value is ~0, not
        # ang_offset.  The old fallback stepped the eye by a full ang_offset
        # (39 deg on some sessions) on every frame where theta was untracked --
        # 28-88% of frames, depending on the recording.
        tilt_h = float(pfh_eye[i])   if np.isfinite(pfh_eye[i])   else mean_pfh
        tilt_v = float(phi_trim[i])  if np.isfinite(phi_trim[i])  else mean_phi

        retinal_images[i] = simulate_retinal_projection(
            pillar_x_mm, pillar_y_mm, mx, my, yaw, p, r_,
            tilt_h, tilt_v,
            pillar_d=pillar_d,
            pillar_h=pillar_h,
            eye_offset_x=eye_offset_x,
            eye_offset_y=eye_offset_y,
            eye_offset_z=eye_offset_z,
            anatomical_eye_angle_deg=anatomical_eye_angle_deg,
            fixed_eye=fixed_eye,
            fixed_pitch=fixed_pitch,
            fixed_roll=fixed_roll,
            resting_tilt_h=mean_pfh,
            resting_tilt_v=mean_phi,
            resting_pitch=mean_pitch,
            resting_roll=mean_roll,
            worldcam=worldcam,
            worldcam_socket_deg=worldcam_socket_deg,
            worldcam_elev_deg=worldcam_elev_deg,
        )

    print(f'  done in {time.time() - t0:.1f}s')
    _n_rendered = n - _n_skip_pos - _n_skip_yaw
    print(f'  {_n_rendered}/{n} frames rendered '
          f'({_n_skip_pos} skipped for missing head position, '
          f'{_n_skip_yaw} for missing yaw)')
    if _n_rendered < 0.5 * n:
        print(f'  WARNING: over half the output is blank. The reconstruction '
              f'cannot be judged from this session until the topdown tracking '
              f'covers the eye-camera window.')

    worldcam_frames  = None
    worldcam_metrics = None
    gt_eval          = None
    if worldcam:
        if worldcam_video_path is None:
            _rec_dir = os.path.dirname(h5_path)
            _prefix  = os.path.basename(h5_path).replace('_preproc.h5', '')
            worldcam_video_path = os.path.join(_rec_dir, f'{_prefix}_eyecam_deinter.avi')
        if os.path.exists(worldcam_video_path):
            print(f'Loading worldcam frames from {os.path.basename(worldcam_video_path)} '
                  f'(start_frame={eyeT_startInd}, N={N}) ...')
            t0 = time.time()
            worldcam_frames = np.zeros((N, 120, 120), dtype=np.uint8)
            cap = cv2.VideoCapture(worldcam_video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, eyeT_startInd)
            for i in range(N):
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    worldcam_frames[i] = cv2.resize(gray, (120, 120))
            cap.release()
            print(f'  done in {time.time() - t0:.1f}s')

            print('Computing worldcam alignment metrics ...')
            t0 = time.time()
            worldcam_metrics = _compute_worldcam_metrics(worldcam_frames, retinal_images)
            print(f'  done in {time.time() - t0:.1f}s  '
                  f'(mean tc_score={float(np.nanmean(worldcam_metrics["tc_score"])):.2f}  '
                  f'mean bg_iou={float(np.nanmean(worldcam_metrics["bg_iou"])):.3f}  '
                  f'mean flow_error={float(np.nanmean(worldcam_metrics["flow_error"])):.2f} px)')
        else:
            print(f'  WARNING: worldcam video not found at {worldcam_video_path} — skipping frame save')
            worldcam_metrics = None

        # Score against the hand-annotated masks, in the world camera's own
        # pixel space and at its own resolution (see
        # evaluate_worldcam_reconstruction for why not in retinal space).
        _mask_path = os.path.join(
            os.path.dirname(h5_path),
            os.path.basename(h5_path).replace('_preproc.h5', '_eyecam_deinter_polygon_masks.npy'),
        )
        if os.path.exists(_mask_path):
            print(f'Validating against {os.path.basename(_mask_path)} ...')
            t0 = time.time()
            _gt_masks = np.load(_mask_path, mmap_mode='r')
            _pose = dict(
                head_x_eye=head_x_eye, head_y_eye=head_y_eye, yaw_eye=yaw_eye,
                pitch_eye=pitch_eye, roll_eye=roll_eye,
                pillar_x_mm=np.array(pillar_x_mm), pillar_y_mm=np.array(pillar_y_mm),
                pillar_d_mm=np.array(pillar_d), pillar_h_mm=np.array(pillar_h),
                eyeT_startInd=np.array(eyeT_startInd),
            )
            try:
                gt_eval = evaluate_worldcam_reconstruction(
                    _gt_masks, _pose,
                    socket_az_deg=worldcam_socket_deg,
                    socket_el_deg=worldcam_elev_deg,
                )
            except Exception as exc:
                print(f'  WARNING: worldcam validation failed ({exc})')
                gt_eval = None
            print(f'  done in {time.time() - t0:.1f}s')
        else:
            print(f'  No GT polygon mask found at {os.path.basename(_mask_path)} — '
                  f'skipping validation')

    if out_npz is None:
        out_npz = h5_path.replace('_preproc.h5', '_retinal_images.npz')

    save_dict = dict(
        retinal_images=retinal_images,
        eyeT=eyeT,
        ang_offset=np.array(ang_offset),
        vor_gain=np.array(vor['vor_gain']),
        pillar_x_mm=np.array(pillar_x_mm),
        pillar_y_mm=np.array(pillar_y_mm),
        pillar_d_mm=np.array(pillar_d),
        pillar_h_mm=np.array(pillar_h),
        worldcam=np.array(worldcam),
        mean_pfh=np.array(mean_pfh),
        mean_phi=np.array(mean_phi),
        mean_pitch=np.array(mean_pitch),
        mean_roll=np.array(mean_roll),
        # The socket the frames were actually rendered through, so the
        # diagnostic panels centre on the same axis (fixed_eye zeroes it).
        anatomical_eye_angle_deg=np.array(
            worldcam_socket_deg if worldcam else
            (0.0 if fixed_eye else anatomical_eye_angle_deg)),
        socket_elev_deg=np.array(worldcam_elev_deg if worldcam else 0.0),
        fixed_eye=np.array(fixed_eye),
        head_x_eye=head_x_eye,
        head_y_eye=head_y_eye,
        yaw_eye=yaw_eye,
        pitch_eye=pitch_eye,
        roll_eye=roll_eye,
        pfh_eye=pfh_eye,
        phi_trim=phi_trim,
        # Fused position in native top-down pixels, for the video overlay.
        head_x_top_px=_hx_px,
        head_y_top_px=_hy_px,
        eyeT_startInd=np.array(eyeT_startInd),
        eyeT_endInd=np.array(eyeT_endInd),
    )
    if worldcam_frames is not None:
        save_dict['worldcam_frames'] = worldcam_frames
    if worldcam_metrics is not None:
        for _k, _v in worldcam_metrics.items():
            save_dict[f'wc_{_k}'] = _v
    save_dict['worldcam_f_px'] = np.array(WORLDCAM_F_PX)
    if gt_eval is not None:
        for _k in ('agg_iou', 'mean_iou', 'mcc', 'pred_visible', 'gt_visible',
                   'false_pos', 'false_neg', 'hfov_deg', 'vfov_deg'):
            save_dict[f'gt_{_k}'] = np.array(gt_eval[_k])
    np.savez_compressed(out_npz, **save_dict)
    print(f'Saved -> {out_npz}')

    return {
        'retinal_images':          retinal_images,
        'eyeT':                    eyeT,
        'ang_offset':              ang_offset,
        'vor_gain':                vor['vor_gain'],
        'pillar_x_mm':             pillar_x_mm,
        'pillar_y_mm':             pillar_y_mm,
        'mean_pfh':                mean_pfh,
        'mean_phi':                mean_phi,
        'mean_pitch':              mean_pitch,
        'mean_roll':               mean_roll,
        'anatomical_eye_angle_deg': (worldcam_socket_deg if worldcam else
                                     (0.0 if fixed_eye else anatomical_eye_angle_deg)),
        'socket_elev_deg':          (worldcam_elev_deg if worldcam else 0.0),
        'fixed_eye':                bool(fixed_eye),
    }, out_npz


_RET_CMAP   = LinearSegmentedColormap.from_list('retinal', ['#060e06', '#00ff55'])
_VIDEO_FPS  = 30
_STRIDE     = 2    # sub-sample eye-camera frames (60 fps -> 30 fps effective)
_IMU_WIN_S  = 20.0
_FIGSIZE    = (14, 8)
_DPI        = 120

_TOP_H, _TOP_W   = 360, round(360 * 2448 / 2048)
_EYE_W, _EYE_H   = 640, 480
_EYE_CX0 = int(_EYE_W * 0.25)
_EYE_CX1 = int(_EYE_W * 0.75)
_EYE_CY0 = int(_EYE_H * 0.40)
_EYE_CY1 = _EYE_H

_FIG_BG   = 'k'
_TRC_BG   = '#0a0a0a'

_RV: dict = {}

def _interp_short_gaps(x, max_gap=5):
    x   = np.asarray(x, dtype=float)
    out = x.copy()
    n, i = len(x), 0
    while i < n:
        if np.isnan(out[i]):
            start = i
            while i < n and np.isnan(out[i]):
                i += 1
            end = i
            if (end - start) <= max_gap and start > 0 and end < n:
                out[start:end] = np.interp(
                    np.arange(start, end), [start - 1, end],
                    [out[start - 1], out[end]])
        else:
            i += 1
    return out


def _nan_wrap(sig, threshold=180.0):
    out   = sig.copy().astype(float)
    jumps = np.where(np.abs(np.diff(out)) > threshold)[0]
    out[jumps + 1] = np.nan
    return out


def _kalman_smooth(signal, fps, Q_frac=1e-3, R_frac=1e-2):

    sig = np.asarray(signal, dtype=float)
    n = len(sig)
    finite = sig[np.isfinite(sig)]
    if len(finite) < 2:
        return sig.copy()
    sv = float(np.var(finite))
    if sv == 0.0:
        return sig.copy()

    dt = 1.0 / fps
    # State transition: position += velocity * dt
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    # Process noise: small on position, larger on velocity
    Q = np.array([[Q_frac * sv * dt ** 2, 0.0],
                  [0.0,                   Q_frac * sv]])
    R_var = R_frac * sv

    i0 = int(np.where(np.isfinite(sig))[0][0])
    x = np.array([[sig[i0]], [0.0]])   # [pos, vel]
    P = np.eye(2) * sv

    out = sig.copy()
    for i in range(n):
        x = F @ x
        P = F @ P @ F.T + Q
        if np.isfinite(sig[i]):
            innov = sig[i] - float(H @ x)
            S = float(H @ P @ H.T) + R_var
            K = (P @ H.T) / S
            x = x + K * innov
            P = (np.eye(2) - K @ H) @ P
        out[i] = float(x[0])
    return out


def _kalman_rts_smooth(signal, fps, Q_frac=1e-3, R_frac=1e-2, R_per_frame=None):

    sig = np.asarray(signal, dtype=float)
    n = len(sig)
    finite = sig[np.isfinite(sig)]
    if len(finite) < 2:
        return sig.copy()
    sv = float(np.var(finite))
    if sv == 0.0:
        return sig.copy()

    dt = 1.0 / fps
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[Q_frac * sv * dt**2, 0.0],
                  [0.0,                  Q_frac * sv]])
    R_base = R_frac * sv

    i0 = int(np.where(np.isfinite(sig))[0][0])
    x = np.array([[sig[i0]], [0.0]])
    P = np.eye(2) * sv

    xs   = np.zeros((n, 2))
    Ps   = np.zeros((n, 2, 2))
    xprs = np.zeros((n, 2))
    Pprs = np.zeros((n, 2, 2))

    for i in range(n):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        xprs[i] = x_pred[:, 0]
        Pprs[i] = P_pred
        x = x_pred
        P = P_pred
        if np.isfinite(sig[i]):
            R_k = float(R_per_frame[i]) if R_per_frame is not None else R_base
            innov = sig[i] - float((H @ x)[0, 0])
            S = float((H @ P @ H.T)[0, 0]) + R_k
            K = (P @ H.T) / S
            x = x + K * innov
            P = (np.eye(2) - K @ H) @ P
        xs[i] = x[:, 0]
        Ps[i] = P

    FT = F.T
    for i in range(n - 2, -1, -1):
        G      = Ps[i] @ FT @ np.linalg.inv(Pprs[i + 1])
        xs[i]  = xs[i]  + G @ (xs[i + 1]  - xprs[i + 1])
        Ps[i]  = Ps[i]  + G @ (Ps[i + 1]  - Pprs[i + 1]) @ G.T

    return xs[:, 0]


def _arc_interpolate_position(x_2p_mm, y_2p_mm, twopT, yaw_deg_renderer, eyeT):

    x_out = np.full(len(eyeT), np.nan)
    y_out = np.full(len(eyeT), np.nan)

    valid = np.isfinite(x_2p_mm) & np.isfinite(y_2p_mm)
    vi = np.where(valid)[0]
    if len(vi) < 2:
        return x_out, y_out

    yaw_rad = np.radians(yaw_deg_renderer)

    for ii in range(len(vi) - 1):
        k0, k1 = vi[ii], vi[ii + 1]
        t0, t1 = float(twopT[k0]), float(twopT[k1])
        x0, y0 = float(x_2p_mm[k0]), float(y_2p_mm[k0])
        x1, y1 = float(x_2p_mm[k1]), float(y_2p_mm[k1])

        T = t1 - t0
        if T <= 0:
            continue

        mask = (eyeT >= t0) & (eyeT <= t1)
        if not mask.any():
            continue

        t_sub = eyeT[mask]
        psi   = yaw_rad[mask]

        dx_actual = x1 - x0
        dy_actual = y1 - y0
        dist = np.sqrt(dx_actual ** 2 + dy_actual ** 2)

        if dist < 1.0:  # near-stationary: linear is exact
            alpha = (t_sub - t0) / T
            x_out[mask] = x0 + alpha * dx_actual
            y_out[mask] = y0 + alpha * dy_actual
            continue

        # Constant-speed dead-reckoning along heading direction.
        # dt_steps[i] = duration from the previous sample to t_sub[i];
        # the first step spans from t0 to t_sub[0].
        speed = dist / T
        dt_steps = np.diff(np.concatenate([[t0], t_sub]))
        dx_dr = np.cumsum(speed * np.cos(psi) * dt_steps)
        dy_dr = np.cumsum(speed * np.sin(psi) * dt_steps)

        # Ramp boundary correction: distribute the endpoint residual linearly
        # over [t0, t1] so the path closes on (x1, y1) exactly.
        alpha = (t_sub - t0) / T
        x_out[mask] = x0 + dx_dr + alpha * (dx_actual - dx_dr[-1])
        y_out[mask] = y0 + dy_dr + alpha * (dy_actual - dy_dr[-1])

    return x_out, y_out


def _smooth_2p_positions(x_2p, y_2p, window=5):

    x_sm = fm2p.nanmedfilt(np.asarray(x_2p, dtype=float), window).squeeze()
    y_sm = fm2p.nanmedfilt(np.asarray(y_2p, dtype=float), window).squeeze()
    return x_sm, y_sm


def _imu_dlc_kalman_position(head_x_2p_mm, head_y_2p_mm, twopT, eyeT,
                              nose_likelihood_2p=None,
                              Q_frac=POSITION_KALMAN_Q, R_frac=5e-2):

    N_eye = len(eyeT)
    x_out = np.full(N_eye, np.nan)
    y_out = np.full(N_eye, np.nan)

    valid_2p = np.isfinite(head_x_2p_mm) & np.isfinite(head_y_2p_mm)
    vi = np.where(valid_2p)[0]
    if len(vi) < 2:
        return x_out, y_out

    sv = max(float(np.nanvar(head_x_2p_mm)), float(np.nanvar(head_y_2p_mm)), 1.0)

    x_meas   = head_x_2p_mm[vi]
    y_meas   = head_y_2p_mm[vi]
    t_meas   = twopT[vi]
    lik_meas = nose_likelihood_2p[vi] if nose_likelihood_2p is not None else None

    # Map each 2P frame to its nearest eye frame (each measurement used exactly
    # once).
    #
    # This was `np.argmin(np.abs(eyeT - t_k))`, which is wrong whenever eyeT
    # contains a NaN: np.argmin returns the index of the NaN rather than of the
    # nearest time, and it does so for EVERY t_k.  All the DLC anchors then
    # collapsed onto that one eye frame, the filter saw a single measurement,
    # and the fused position came out as a constant -- a head that never moves,
    # for the whole recording.  eyeT_trim carries 3-9 interior NaNs on 20 of the
    # 37 cohort02 sessions, so this silently pinned the reconstruction in place
    # on more than half the dataset.  Search the finite subsequence instead,
    # which is sorted, and map back; it is also O(log n) per measurement rather
    # than O(n).
    fin = np.flatnonzero(np.isfinite(eyeT))
    if len(fin) < 2:
        return x_out, y_out
    eyeT_f = np.asarray(eyeT, dtype=float)[fin]
    if not np.all(np.diff(eyeT_f) >= 0):
        _o = np.argsort(eyeT_f, kind='stable')
        fin, eyeT_f = fin[_o], eyeT_f[_o]
    if len(fin) < N_eye:
        print(f'  NOTE: eyeT has {N_eye - len(fin)} non-finite sample(s); '
              f'anchors are matched against the {len(fin)} finite ones.')

    _r = np.searchsorted(eyeT_f, t_meas)
    _lo = np.clip(_r - 1, 0, len(eyeT_f) - 1)
    _hi = np.clip(_r,     0, len(eyeT_f) - 1)
    _pick = np.where(np.abs(eyeT_f[_lo] - t_meas) <= np.abs(eyeT_f[_hi] - t_meas),
                     _lo, _hi)
    meas_at_eye = {}
    for k, i_eye in enumerate(fin[_pick]):
        i_eye = int(i_eye)
        if i_eye not in meas_at_eye:
            meas_at_eye[i_eye] = k

    H      = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    R_base = R_frac * sv

    eye_start = int(fin[np.searchsorted(eyeT_f, float(t_meas[0]))]
                    if np.searchsorted(eyeT_f, float(t_meas[0])) < len(fin) else N_eye)
    _e = int(np.searchsorted(eyeT_f, float(t_meas[-1])))
    eye_end   = min(int(fin[_e]) + 1 if _e < len(fin) else N_eye, N_eye)
    if eye_start >= eye_end:
        return x_out, y_out

    dt_eye = float(np.nanmedian(np.diff(eyeT[eye_start:eye_end])))
    if dt_eye <= 0:
        dt_eye = 1.0 / 60.0

    F = np.array([[1, 0, dt_eye, 0],
                   [0, 1, 0, dt_eye],
                   [0, 0, 1,      0],
                   [0, 0, 0,      1]], dtype=float)
    Q = np.diag([Q_frac * sv * dt_eye**2,
                  Q_frac * sv * dt_eye**2,
                  Q_frac * sv * dt_eye,
                  Q_frac * sv * dt_eye])

    n_run = eye_end - eye_start
    xs   = np.zeros((n_run, 4))
    Ps   = np.zeros((n_run, 4, 4))
    xprs = np.zeros((n_run, 4))
    Pprs = np.zeros((n_run, 4, 4))

    x_state = np.array([float(head_x_2p_mm[vi[0]]),
                         float(head_y_2p_mm[vi[0]]),
                         0.0, 0.0])
    P_state = np.diag([sv, sv, sv * 0.1, sv * 0.1])

    for j in range(n_run):
        i = eye_start + j
        x_pred = F @ x_state
        P_pred = F @ P_state @ F.T + Q
        xprs[j] = x_pred
        Pprs[j] = P_pred
        x_state = x_pred
        P_state = P_pred
        if i in meas_at_eye:
            k = meas_at_eye[i]
            z    = np.array([x_meas[k], y_meas[k]])
            lik  = float(np.clip(lik_meas[k], 0.01, 1.0)) if lik_meas is not None else 1.0
            R_k  = R_base / (lik ** 2)
            S    = H @ P_state @ H.T + np.diag([R_k, R_k])
            K    = P_state @ H.T @ np.linalg.inv(S)
            x_state = x_state + K @ (z - H @ x_state)
            P_state = (np.eye(4) - K @ H) @ P_state
        xs[j]  = x_state
        Ps[j]  = P_state

    # RTS backward pass (eliminates one-sided Kalman lag)
    # should try without this at some point...
    FT = F.T
    for j in range(n_run - 2, -1, -1):
        G      = Ps[j] @ FT @ np.linalg.inv(Pprs[j + 1])
        xs[j]  = xs[j]  + G @ (xs[j + 1]  - xprs[j + 1])
        Ps[j]  = Ps[j]  + G @ (Ps[j + 1]  - Pprs[j + 1]) @ G.T

    x_out[eye_start:eye_end] = xs[:, 0]
    y_out[eye_start:eye_end] = xs[:, 1]
    return x_out, y_out


def _compute_worldcam_metrics(worldcam_frames, retinal_images, tau_bg=30, bg_threshold=15):

    N = len(worldcam_frames)
    eps = 1e-6

    tc_score = np.full(N, np.nan)
    for t in range(N - 1):
        diff = np.abs(worldcam_frames[t + 1].astype(np.float32) -
                      worldcam_frames[t].astype(np.float32))
        mask  = retinal_images[t] > 0
        n_in  = int(mask.sum())
        n_out = int((~mask).sum())
        if n_in > 0 and n_out > 0:
            tc_score[t] = float(diff[mask].mean()) / (float(diff[~mask].mean()) + eps)

    init_n = min(tau_bg * 3, N)
    bg     = worldcam_frames[:init_n].astype(np.float32).mean(axis=0)
    alpha  = 1.0 - np.exp(-1.0 / tau_bg)

    bg_iou          = np.full(N, np.nan)
    bg_centroid_err = np.full((N, 2), np.nan)

    for t in range(N):
        fg   = np.abs(worldcam_frames[t].astype(np.float32) - bg) > bg_threshold
        mask = retinal_images[t] > 0

        inter = np.logical_and(fg, mask).sum()
        union = np.logical_or(fg, mask).sum()
        if union > 0:
            bg_iou[t] = float(inter) / float(union)

        if fg.any() and mask.any():
            fg_r, fg_c = np.where(fg)
            m_r,  m_c  = np.where(mask)
            bg_centroid_err[t, 0] = float(fg_c.mean()) - float(m_c.mean())
            bg_centroid_err[t, 1] = float(fg_r.mean()) - float(m_r.mean())

        bg = (1.0 - alpha) * bg + alpha * worldcam_frames[t].astype(np.float32)

    # Pre-compute retinal mask centroids [col, row]
    centroids = np.full((N, 2), np.nan)
    for t in range(N):
        mask = retinal_images[t] > 0
        if mask.any():
            r, c = np.where(mask)
            centroids[t] = [float(c.mean()), float(r.mean())]

    flow_pred_disp = np.full((N, 2), np.nan)
    flow_obs       = np.full((N, 2), np.nan)
    flow_error     = np.full(N, np.nan)

    for t in range(N - 1):
        if not np.isnan(centroids[t]).any() and not np.isnan(centroids[t + 1]).any():
            flow_pred_disp[t] = centroids[t + 1] - centroids[t]

        mask_t = retinal_images[t] > 0
        if mask_t.sum() >= 4:
            flow = cv2.calcOpticalFlowFarneback(
                worldcam_frames[t], worldcam_frames[t + 1],
                None, 0.5, 3, 15, 3, 5, 1.2, 0,
            )
            flow_obs[t, 0] = float(flow[mask_t, 0].mean())
            flow_obs[t, 1] = float(flow[mask_t, 1].mean())

        if not np.isnan(flow_pred_disp[t]).any() and not np.isnan(flow_obs[t]).any():
            flow_error[t] = float(np.sqrt(np.sum((flow_pred_disp[t] - flow_obs[t]) ** 2)))

    return dict(
        tc_score=tc_score,
        bg_iou=bg_iou,
        bg_centroid_err=bg_centroid_err,
        flow_pred_disp=flow_pred_disp,
        flow_obs=flow_obs,
        flow_error=flow_error,
    )


def _subtract_band(frame_gray):
    left    = frame_gray[:, :10].mean(axis=1).astype(float)
    right   = frame_gray[:, -10:].mean(axis=1).astype(float)
    profile = gaussian_filter1d(0.5 * (left + right), sigma=15)
    corrected = frame_gray.astype(float) - profile[:, np.newaxis]
    corrected -= corrected.min()
    mx = corrected.max()
    if mx > 0:
        corrected = corrected / mx * 255.0
    return np.clip(corrected, 0, 255).astype(np.uint8)


def _pillar_optical_corners(
    pillar_x, pillar_y, mouse_x, mouse_y, mouse_yaw,
    mouse_pitch, mouse_roll, pupil_tilt_h, pupil_tilt_v,
    pillar_h=210.0, pillar_d=40.0,
    eye_offset_x=3.5, eye_offset_y=-5.0, eye_offset_z=3.5,
    anatomical_eye_angle_deg=-65.0,
    socket_elev_deg=0.0,
    head_z=25.0,
    pitch_gain=PITCH_GAIN,
):
    
    r = pillar_d / 2.0
    corners_world = _cylinder_points(pillar_x, pillar_y, r, pillar_h)
    # Intrinsic Z-Y'-X'' -- must match simulate_retinal_projection, see the
    # comment there for why this is upper case.
    R_head = R.from_euler('ZYX', [mouse_yaw, mouse_pitch * pitch_gain, mouse_roll],
                          degrees=True).as_matrix()
    t_head = np.array([[mouse_x], [mouse_y], [head_z]])
    M_head = np.eye(4)
    M_head[:3, :3] = R_head
    M_head[:3, 3:] = t_head
    corners_head = np.linalg.inv(M_head) @ corners_world

    R_saccade = R.from_euler('zyx', [pupil_tilt_h, pupil_tilt_v, 0], degrees=True).as_matrix()
    R_socket  = (R.from_euler('z', anatomical_eye_angle_deg, degrees=True).as_matrix()
                 @ R.from_euler('y', socket_elev_deg, degrees=True).as_matrix())
    R_eye     = R_socket @ R_saccade
    t_eye     = np.array([[eye_offset_x], [eye_offset_y], [eye_offset_z]])
    M_eye     = np.eye(4)
    M_eye[:3, :3] = R_eye
    M_eye[:3, 3:] = t_eye
    corners_eye = np.linalg.inv(M_eye) @ corners_head

    R_align = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)
    # tried changing to R_align = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)
    return R_align @ corners_eye[:3, :]


def _render_panoramic(corners_opt, pan_w=360, pan_h=120, az_offset=0.0):

    img = np.zeros((pan_h, pan_w), dtype=np.uint8)

    # Elevation: min/max over the sampled cylinder points
    el_list = []
    for i in range(corners_opt.shape[1]):
        pt = corners_opt[:, i]
        lateral = np.sqrt(pt[0] ** 2 + pt[2] ** 2)
        el = np.degrees(np.arctan2(-pt[1], lateral))
        el_list.append(el)
    el_arr = np.array(el_list)

    center = corners_opt.mean(axis=1)
    az_c = np.degrees(np.arctan2(center[0], center[2])) + az_offset

    az_corners = np.degrees(np.arctan2(corners_opt[0, :], corners_opt[2, :]))
    az_delta = ((az_corners - (az_c - az_offset) + 180.0) % 360.0) - 180.0
    az_half = min(float(np.max(np.abs(az_delta))), 90.0)

    u_min = int(np.floor(az_c - az_half + pan_w / 2))
    u_max = int(np.ceil( az_c + az_half + pan_w / 2))
    v_min = int(np.floor(pan_h / 2 - el_arr.max()))
    v_max = int(np.ceil( pan_h / 2 - el_arr.min()))
    v_min = max(0, v_min)
    v_max = min(pan_h - 1, v_max)
    for u in range(u_min, u_max + 1):
        img[v_min:v_max + 1, u % pan_w] = 255
    return img


def _find_ffmpeg():
    for ff in ('/usr/bin/ffmpeg', 'ffmpeg'):
        try:
            out = subprocess.run([ff, '-encoders'], capture_output=True, text=True)
            if 'libx264' in out.stdout:
                return ff, ['-vcodec', 'libx264', '-preset', 'faster', '-crf', '20', '-bf', '0']
            if 'libopenh264' in out.stdout:
                return ff, ['-vcodec', 'libopenh264', '-b:v', '8M', '-bf', '0']
            if 'libvpx' in out.stdout:
                return ff, ['-vcodec', 'libvpx', '-b:v', '8M']
        except FileNotFoundError:
            continue
    return 'ffmpeg', ['-vcodec', 'libx264', '-preset', 'faster', '-crf', '20', '-bf', '0']


def _rv_worker_init(init_data: dict) -> None:
    global _RV
    _RV = init_data

    # Determine early whether worldcam flow data is available so layout is conditional
    _wc_eyeT_pre = init_data.get('wc_eyeT', np.array([]))
    _wc_flow_pre = init_data.get('wc_flow', np.array([]))
    has_err      = len(_wc_eyeT_pre) > 0 and len(_wc_flow_pre) > 0

    fig = plt.figure(figsize=_FIGSIZE, dpi=_DPI, facecolor=_FIG_BG)

    gs = GridSpec(
        3, 2, figure=fig,
        width_ratios=[1, 2.2],
        height_ratios=[1, 1, 1.2],
        hspace=0.08, wspace=0.08,
        left=0.10, right=0.97, top=0.97, bottom=0.05,
    )

    ax_td  = fig.add_subplot(gs[0, 0])
    ax_eye = fig.add_subplot(gs[1, 0])

    # Right column: retina full-height when no worldcam; split with flow trace when worldcam
    if has_err:
        gs_right  = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[0:3, 1],
            height_ratios=[3, 1], hspace=0.14,
        )
        ax_retina = fig.add_subplot(gs_right[0])
        ax_flow   = fig.add_subplot(gs_right[1])
    else:
        ax_retina = fig.add_subplot(gs[0:3, 1])
        ax_flow   = None

    gs_tr = GridSpecFromSubplotSpec(5, 1, subplot_spec=gs[2, 0], hspace=0.06)
    ax_pitch = fig.add_subplot(gs_tr[0])
    ax_roll  = fig.add_subplot(gs_tr[1])
    ax_yaw   = fig.add_subplot(gs_tr[2])
    ax_theta = fig.add_subplot(gs_tr[3])
    ax_phi   = fig.add_subplot(gs_tr[4])

    trace_axes = (ax_pitch, ax_roll, ax_yaw, ax_theta, ax_phi)

    for ax in (ax_td, ax_eye, ax_retina):
        ax.set_facecolor(_FIG_BG)
        ax.axis('off')

    style_axes = list(trace_axes) + ([ax_flow] if ax_flow is not None else [])
    for ax in style_axes:
        ax.set_facecolor(_TRC_BG)
        ax.tick_params(colors='0.6', labelsize=8)
        for sp in ax.spines.values():
            sp.set_color('0.3')

    im_td = ax_td.imshow(
        np.zeros((_TOP_H, _TOP_W, 3), dtype=np.uint8),
        aspect='equal', interpolation='nearest',
    )
    ax_td.set_xlim(0, _TOP_W - 1)
    ax_td.set_ylim(_TOP_H - 1, 0)
    ax_td.plot(init_data['pillar_x_top'], init_data['pillar_y_top'],
               '*', color='#ff4444', markersize=14, markeredgewidth=1,
               markeredgecolor='white', zorder=10)

    # Hollow: the raw DLC ear-bar midpoint, present only on frames where both
    # bars cleared the likelihood threshold.  Filled: the fused position the
    # renderer used.  Seeing them separately is the whole point -- a filled dot
    # with no hollow one around it means the pose is interpolated, and no dot at
    # all means the top-down tracking never covered this part of the recording.
    head_anchor, = ax_td.plot([], [], 'o', color='none', markersize=11,
                              markeredgewidth=1.2, markeredgecolor='#ffff00',
                              alpha=0.8, zorder=19)
    head_dot, = ax_td.plot([], [], 'o', color='#ffff00', markersize=7,
                           markeredgewidth=1, markeredgecolor='k', zorder=20)
    head_dir, = ax_td.plot([], [], '-', color='#ffff00', lw=2, zorder=21)
    gaze_dir, = ax_td.plot([], [], '-', color='#00ff55', lw=2, zorder=22)

    im_eye = ax_eye.imshow(
        np.zeros((_EYE_H, _EYE_W), dtype=np.uint8),
        cmap='gray', vmin=0, vmax=255,
        aspect='equal', interpolation='nearest',
    )
    ax_eye.set_xlim(-0.5, _EYE_W - 0.5)
    ax_eye.set_ylim(_EYE_H - 0.5, -0.5)

    im_ret = ax_retina.imshow(
        np.zeros((120, 120), dtype=np.uint8),
        cmap=_RET_CMAP, vmin=0, vmax=255,
        aspect='equal', interpolation='nearest',
        extent=[-60, 60, -60, 60],
    )
    # 2x the world camera's field, centred on its axis (computed by the caller).
    # This panel used to show retinal_images[:60, 60:] -- the upper-right
    # quadrant, az 0-60 / el 0-60 -- which assumed the camera covered that
    # quadrant; it covers ~23 x 17 deg.
    _rx, _ry = init_data['ret_xlim'], init_data['ret_ylim']
    ax_retina.set_xlim(*_rx)
    ax_retina.set_ylim(*_ry)
    ax_retina.set_title('Estimated retinal image', color='0.6', fontsize=11, pad=4)
    ax_retina.tick_params(colors='0.4', labelsize=8)
    ax_retina.set_xlabel('Azimuth (deg)', color='0.5', fontsize=9)
    ax_retina.set_ylabel('Elevation (deg)', color='0.5', fontsize=9)
    for sp in ax_retina.spines.values():
        sp.set_color('0.3')
    ax_retina.axis('on')

    twopT   = init_data['twopT']
    t_start = float(init_data['t_start'])
    blk_lo  = init_data['twop_lo']
    blk_nd  = init_data['twop_nd']

    fps_2p       = (len(twopT) - 1) / float(twopT[-1] - twopT[0]) if len(twopT) > 1 else 7.5
    extra_frames = int(_IMU_WIN_S / 2.0 * fps_2p)
    ext_lo       = max(0, blk_lo - extra_frames)

    tt_rel  = twopT[ext_lo:blk_nd] - t_start
    pitch_b = _interp_short_gaps(init_data['pitch'][ext_lo:blk_nd].astype(float))
    roll_b  = _interp_short_gaps(init_data['roll'][ext_lo:blk_nd].astype(float))
    # Yaw trace: the fused signal that actually drives the render, sampled at
    # the 2P times so it shares an x-axis with the other traces.  It used to be
    # the raw ear-vector yaw, which is NaN wherever the top-down ear bars drop
    # out -- 93.4 % of frames on 251016_DMM_DMM061_pos18, so the panel was
    # simply blank on exactly the sessions where you most want to see it.
    _yaw_fused = init_data.get('head_yaw_fused_out')
    if _yaw_fused is not None:
        _yb = np.interp(twopT[ext_lo:blk_nd], init_data['times'],
                        np.unwrap(np.radians(_yaw_fused)) * 180.0 / np.pi) % 360.0
    else:
        _yb = init_data['yaw'][ext_lo:blk_nd].copy().astype(float)
    yaw_b   = _nan_wrap(_interp_short_gaps(_yb))
    # theta/phi at the EYE-CAMERA rate, which is the trace the render uses.
    #
    # These panels used to plot theta_interp / phi_interp -- the same signals
    # resampled onto the 7.5 Hz 2P grid by interpT, i.e. scipy interp1d with
    # linear interpolation.  Linear interpolation returns NaN whenever either
    # bracketing sample is NaN, and pupil tracking is sparse (theta_trim is
    # 25.4 % finite on 251016_DMM_DMM056_pos13), so nearly every finite sample
    # has a NaN neighbour and the resampled copy is eroded down to 17.3 %.
    # Measured on that session: 48.8 % of 2P frames have a finite eye sample
    # within half a frame time, and theta_interp is NaN on 64.5 % of those.  The
    # gaps in the panel were therefore mostly an artefact of the resampling,
    # not missing data -- and they did not correspond to the gaps in what was
    # being rendered, which is why gaze kept moving through them.
    _eyeT_tr = init_data.get('eyeT_trim_full')
    _th_tr   = init_data.get('theta_trim')
    _ph_tr   = init_data.get('phi_trim')
    if _eyeT_tr is not None and _th_tr is not None:
        _w = ((_eyeT_tr >= float(twopT[ext_lo])) &
              (_eyeT_tr <= float(twopT[blk_nd - 1])) & np.isfinite(_eyeT_tr))
        te_rel  = _eyeT_tr[_w] - t_start
        theta_b = _interp_short_gaps(np.asarray(_th_tr, float)[_w])
        phi_b   = _interp_short_gaps(np.asarray(_ph_tr, float)[_w])
    else:
        te_rel  = tt_rel
        theta_b = _interp_short_gaps(init_data['theta_2p'][ext_lo:blk_nd].astype(float))
        phi_b   = _interp_short_gaps(init_data['phi_2p'][ext_lo:blk_nd].astype(float))

    colors  = ['#4a9eff', '#4aff88', '#ffaa44', '#ff4aaa', '#bb88ff']
    labels  = ['Pitch', 'Roll', 'Yaw', 'θ', 'φ']
    signals = [pitch_b, roll_b, yaw_b, theta_b, phi_b]
    sig_t   = [tt_rel, tt_rel, tt_rel, te_rel, te_rel]

    for ax, sig, col, lbl, _t in zip(trace_axes, signals, colors, labels, sig_t):
        ax.plot(_t, sig, color=col, lw=1.2)
        ax.set_ylabel(lbl, color='0.6', fontsize=8, labelpad=2)
        fin = sig[np.isfinite(sig)]
        if len(fin):
            p1, p99 = np.nanpercentile(fin, [1, 99])
            mg = max(0.05 * abs(p99 - p1), 0.5)
            ax.set_ylim(p1 - mg, p99 + mg)
        ax.set_xlim(-_IMU_WIN_S / 2.0, _IMU_WIN_S / 2.0)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: str(int(x))))

    ax_phi.set_xlabel('Time (s)', color='0.6', fontsize=8)
    for ax in (ax_pitch, ax_roll, ax_yaw, ax_theta):
        ax.tick_params(labelbottom=False)

    cursors = [ax.axvline(0.0, color='w', lw=0.8, alpha=0.8)
               for ax in trace_axes]

    # ---- Optical flow error trace (worldcam only) ----
    _wc_eyeT = _wc_eyeT_pre
    _wc_flow = _wc_flow_pre

    err_cursors = []
    if has_err:
        _eye_lo = int(np.searchsorted(_wc_eyeT, twopT[ext_lo]))
        _eye_nd = int(np.searchsorted(_wc_eyeT, twopT[blk_nd]))
        err_tt  = _wc_eyeT[_eye_lo:_eye_nd] - t_start
        sig     = _wc_flow[_eye_lo:_eye_nd]
        ax_flow.plot(err_tt, sig, color='#ff44bb', lw=1.0)
        ax_flow.set_ylabel('Flow\nerr (px)', color='0.6', fontsize=7,
                           labelpad=2, rotation=0, va='center', ha='right')
        fin = sig[np.isfinite(sig)]
        if len(fin):
            p1, p99 = np.nanpercentile(fin, [1, 99])
            mg = max(0.05 * abs(p99 - p1), 0.01)
            ax_flow.set_ylim(max(0, p1 - mg), p99 + mg)
        ax_flow.set_xlim(-_IMU_WIN_S / 2.0, _IMU_WIN_S / 2.0)
        ax_flow.xaxis.set_major_locator(mticker.MultipleLocator(5))
        ax_flow.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: str(int(x))))
        ax_flow.set_xlabel('Time (s)', color='0.6', fontsize=8)
        ax_flow.set_title('Optic flow reconstruction error', color='0.5',
                          fontsize=8, pad=3)
        err_cursors = [ax_flow.axvline(0.0, color='w', lw=0.8, alpha=0.8)]

    time_txt = fig.text(0.50, 0.004, '', color='0.6', fontsize=10,
                        ha='center', va='bottom')

    _RV['fig']         = fig
    _RV['im_td']       = im_td
    _RV['im_eye']      = im_eye
    _RV['im_ret']      = im_ret
    _RV['trace_axes']  = trace_axes
    _RV['cursors']     = cursors
    _RV['ax_flow']     = ax_flow
    _RV['err_cursors'] = err_cursors
    _RV['has_err']     = has_err
    _RV['time_txt']    = time_txt
    _RV['head_anchor'] = head_anchor
    _RV['head_dot']    = head_dot
    _RV['head_dir']    = head_dir
    _RV['gaze_dir']    = gaze_dir
    _RV['FIG_H']       = int(fig.get_figheight() * _DPI)
    _RV['FIG_W']       = int(fig.get_figwidth()  * _DPI)


def _rv_render_frame(out_idx: int) -> bytes:
    
    t_abs        = float(_RV['times'][out_idx])
    t_rel        = t_abs - float(_RV['t_start'])
    twop_abs_idx = int(_RV['twop_idx'][out_idx])

    _RV['im_eye'].set_data(_RV['eye_frames'][out_idx])

    # Topdown frame: use the precomputed time-matched index so the video
    # frame is always the one closest in time to twopT[twop_abs_idx],
    # regardless of the topdown camera fps or container header accuracy.
    top_bi = int(_RV['top_idx_for_output'][out_idx])
    _RV['im_td'].set_data(
        cv2.cvtColor(_RV['top_frames'][top_bi], cv2.COLOR_BGR2RGB))

    # Head position and direction overlay on topdown
    hx = _RV['head_x_arr']
    hy = _RV['head_y_arr']
    hyaw = _RV['head_yaw_arr']
    sx = _RV['top_scale_x']
    sy = _RV['top_scale_y']
    # Index the raw tracking by the topdown frame that is actually on screen, so
    # the dot and the animal cannot drift apart even if twopT and the video
    # timestamps disagree.
    tidx = min(int(_RV['top_buf_track_idx'][top_bi]), len(hx) - 1)

    # The filled marker shows the position the RENDERER used -- the fused trace
    # at the eye-camera rate -- not the raw DLC anchors.  Drawing the anchors
    # made the overlay blank wherever a marker dropped out (93 % of frames on
    # 251016_DMM_DMM061_pos18), which reads as "the reconstruction is broken"
    # when the reconstruction was in fact tracking a Kalman-interpolated
    # position perfectly well.  The raw anchors are still drawn, hollow, so
    # measured frames stay distinguishable from coasted ones.
    _fx = _RV.get('head_x_top_px')
    if _fx is not None:
        _ri = min(int(_RV['eye_trim_idx'][out_idx]), len(_fx) - 1)
        px, py = float(_fx[_ri]), float(_RV['head_y_top_px'][_ri])
    else:
        px, py = float(hx[tidx]), float(hy[tidx])

    _ax, _ay = float(hx[tidx]), float(hy[tidx])
    if np.isfinite(_ax) and np.isfinite(_ay):
        _RV['head_anchor'].set_data([_ax * sx], [_ay * sy])
    else:
        _RV['head_anchor'].set_data([], [])

    _fused = _RV.get('head_yaw_fused_out')
    yaw_deg = (float(_fused[out_idx]) if _fused is not None
               else float(hyaw[min(tidx, len(hyaw) - 1)]))
    if np.isfinite(px) and np.isfinite(py):
        dx_top = px * sx
        dy_top = py * sy  # image y-down convention (no negation needed)
        _RV['head_dot'].set_data([dx_top], [dy_top])
        if np.isfinite(yaw_deg):
            arrow_len = 20.0
            yaw_rad = np.radians(yaw_deg)
            ex = dx_top + arrow_len * np.cos(yaw_rad)
            ey = dy_top + arrow_len * np.sin(yaw_rad)
            _RV['head_dir'].set_data([dx_top, ex], [dy_top, ey])

            # Gaze direction. For worldcam the camera is co-aligned with the head,
            # so gaze = head yaw regardless of the anatomical socket convention.
            # fixed_eye renders through the same zeroed socket.
            if _RV.get('head_aligned', _RV.get('worldcam', False)):
                gaze_deg = yaw_deg
            else:
                gaze_deg = yaw_deg - _RV['anatomical_eye_angle_deg'] - _RV['resting_tilt_h']
            gaze_rad = np.radians(gaze_deg)
            gex = dx_top + arrow_len * np.cos(gaze_rad)
            gey = dy_top + arrow_len * np.sin(gaze_rad)
            _RV['gaze_dir'].set_data([dx_top, gex], [dy_top, gey])
        else:
            _RV['head_dir'].set_data([], [])
            _RV['gaze_dir'].set_data([], [])
    else:
        _RV['head_dot'].set_data([], [])
        _RV['head_dir'].set_data([], [])
        _RV['gaze_dir'].set_data([], [])

    ret_idx = min(int(_RV['eye_trim_idx'][out_idx]), len(_RV['retinal_images']) - 1)
    _RV['im_ret'].set_data(_RV['retinal_images'][ret_idx])

    for ax, cur in zip(_RV['trace_axes'], _RV['cursors']):
        ax.set_xlim(t_rel - _IMU_WIN_S / 2.0, t_rel + _IMU_WIN_S / 2.0)
        cur.set_xdata([t_rel, t_rel])

    if _RV.get('has_err', False):
        _RV['ax_flow'].set_xlim(t_rel - _IMU_WIN_S / 2.0, t_rel + _IMU_WIN_S / 2.0)
        _RV['err_cursors'][0].set_xdata([t_rel, t_rel])

    _RV['time_txt'].set_text(f't = {t_rel:.2f} s')

    _RV['fig'].canvas.draw()
    buf = _RV['fig'].canvas.buffer_rgba()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(
        _RV['FIG_H'], _RV['FIG_W'], 4)
    return arr[:, :, :3].tobytes()


# Videos in the recording folder that are outputs, not the arena camera.  The
# diagnostic MP4s this module writes land in the same directory, and one of them
# being picked up as the "top-down video" is not hypothetical -- see
# preprocess._resolve_topdown_dlc for the same mistake made with DLC files.
_TOP_VIDEO_PATTERNS = ('fm*_000?.mp4', 'fm*.mp4', 'rec*_000?.mp4',
                       '*_top.mp4', '*top*.mp4')
_TOP_VIDEO_REJECT   = ('diagnostic', 'labeled', 'decoding', 'resnet50',
                       'retinal', 'eyecam', 'dlc')


def _find_topdown_video(rec_dir):
    """The arena camera video in `rec_dir`.

    This used to be

        for cand in ('fm*_000?.mp4'):

    where the parentheses do NOT make a tuple -- it is a plain string, so the
    loop ran over its characters.  The second character to reach glob is '*',
    which matches every file in the directory, and the code then took
    `matches[0]`: whatever `os.scandir` happened to return first.  On
    251016_DMM_DMM056_pos13 that was `..._eye_tracking_figs.pdf`, on
    251016_DMM_DMM061_pos18 `model_results_cell_93.png`.  Because the order is
    filesystem-dependent it also changed between runs on the same folder, so the
    same session could work once and pick a PNG the next time.
    """

    seen = []
    for pat in _TOP_VIDEO_PATTERNS:
        for m in sorted(glob.glob(os.path.join(rec_dir, pat))):
            if m in seen:
                continue
            seen.append(m)
            low = os.path.basename(m).lower()
            if any(bad in low for bad in _TOP_VIDEO_REJECT):
                continue
            cap = cv2.VideoCapture(m)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  if cap.isOpened() else 0
            cap.release()
            # A still image opens fine as a 1-frame "video", so require a real
            # sequence before believing this is the arena camera.
            if n > 1 and w > 0:
                return m

    raise FileNotFoundError(
        f'Cannot find a top-down video in {rec_dir}. Looked for '
        f'{list(_TOP_VIDEO_PATTERNS)}, rejected names containing '
        f'{list(_TOP_VIDEO_REJECT)}, and required a readable multi-frame video. '
        f'Candidates considered: {[os.path.basename(x) for x in seen]}. '
        f'Pass top_path= explicitly if the arena video is named unusually.')


def make_retinal_diagnostic_video(
    h5_path,
    npz_path,
    eye_path=None,
    top_path=None,
    out_path=None,
    arena_width_cm=None,
    pillar_d=40.0,
    pillar_h=210.0,
    eye_offset_x=3.5,
    eye_offset_y=-5.0,
    eye_offset_z=3.5,
    panoramic_yaw_only=False,
):

    rec_dir = os.path.dirname(h5_path)
    prefix  = os.path.basename(h5_path).replace('_preproc.h5', '')

    if eye_path is None:
        eye_path = os.path.join(rec_dir, f'{prefix}_eyecam_deinter.avi')
    if top_path is None:
        top_path = _find_topdown_video(rec_dir)
    if out_path is None:
        out_path = os.path.join(rec_dir, f'{prefix}_retinal_diagnostic.mp4')

    print(f'Topdown video: {os.path.basename(top_path)}')


    print('Loading preproc ...')
    with h5py.File(h5_path, 'r') as f:
        eyeT_trim    = f['eyeT_trim'][:]
        twopT        = f['twopT'][:]
        light_onsets = f['light_onsets'][:]
        dark_onsets  = f['dark_onsets'][:]
        startInd     = int(f['eyeT_startInd'][()])
        theta_raw    = f['theta'][:]
        phi_raw      = f['phi'][:]
        theta_2p     = f['theta_interp'][:]
        phi_2p       = f['phi_interp'][:]
        # Eye-rate pupil angles: what the reconstruction is actually driven by.
        theta_trim_v = f['theta_trim'][:] if 'theta_trim' in f else None
        phi_trim_v   = f['phi_trim'][:]   if 'phi_trim'   in f else None
        head_x       = f['head_x'][:]
        head_y       = f['head_y'][:]
        head_yaw     = f['head_yaw_deg'][:]
        try:
            pitch = f['pitch_twop_interp'][:]
            roll  = f['roll_twop_interp'][:]
        except KeyError:
            pitch = np.zeros(len(twopT))
            roll  = np.zeros(len(twopT))
            print('  WARNING: pitch_twop_interp / roll_twop_interp not in H5 — pitch/roll set to zero.')
        pxls2cm      = float(f['pxls2cm'][()])
        if arena_width_cm is not None:
            try:
                arenaTL_x = float(f['arenaTL']['x'][()])
                arenaTR_x = float(f['arenaTR']['x'][()])
                arenaBL_x = float(f['arenaBL']['x'][()])
                arenaBR_x = float(f['arenaBR']['x'][()])
                px_width = np.mean([arenaTR_x - arenaTL_x, arenaBR_x - arenaBL_x])
                pxls2cm = px_width / arena_width_cm
                print(f'  [Video] Overriding pxls2cm with value calculated from arena_width_cm={arena_width_cm:.1f} -> {pxls2cm:.3f}')
            except KeyError:
                print('  [Video] WARNING: Could not find arena corner keys in HDF5 to override pxls2cm.')
        # Average the annotation polygon rather than trusting
        # pillar_centroid, which is wrong on some sessions -- on
        # 251016_DMM_DMM061_pos18 it holds (-1155, 1239) px against polygon
        # vertices spanning x 1374-1549, so the red pillar marker landed off the
        # arena entirely while the renderer (which already averages the polygon)
        # used the right place.  Same helper as get_retinal_image, so the two
        # panels can no longer disagree about where the pillar is.
        pillar_x_px, pillar_y_px = _read_pillar_centroid_px(f)
        _pillar_rad_px2 = float(f['pillar_radius'][()]) if 'pillar_radius' in f else None
        try:
            imuT_trim_arr = f['imuT_trim'][:]
            yaw_imu_arr   = f['upsampled_yaw']['igyro_corrected_deg'][:]
        except KeyError:
            print('  WARNING: imuT_trim / upsampled_yaw not found -- falling back to head_yaw_deg for panoramic.')
            imuT_trim_arr = None
            yaw_imu_arr   = None


    print('Loading retinal images ...')
    npz            = np.load(npz_path, allow_pickle=False)
    retinal_images = npz['retinal_images']
    resting_tilt_h          = float(npz['mean_pfh'])          if 'mean_pfh'                   in npz else 0.0
    anatomical_eye_angle_deg = float(npz['anatomical_eye_angle_deg']) if 'anatomical_eye_angle_deg' in npz else -65.0
    _npz_pillar_x_mm = float(npz['pillar_x_mm']) if 'pillar_x_mm' in npz else None
    _npz_pillar_y_mm = float(npz['pillar_y_mm']) if 'pillar_y_mm' in npz else None
    _npz_ang_offset  = float(npz['ang_offset'])  if 'ang_offset'  in npz else 0.0
    _npz_mean_pfh    = float(npz['mean_pfh'])     if 'mean_pfh'    in npz else 0.0
    _npz_mean_phi    = float(npz['mean_phi'])     if 'mean_phi'    in npz else 0.0
    _npz_mean_pitch  = float(npz['mean_pitch'])   if 'mean_pitch'  in npz else 0.0
    _npz_mean_roll   = float(npz['mean_roll'])    if 'mean_roll'   in npz else 0.0
    _npz_worldcam    = bool(npz['worldcam'])       if 'worldcam'    in npz else False
    # fixed_eye renders through a zeroed socket, so the camera axis is the head
    # axis -- the gaze arrow and the panel crop follow the same rule as worldcam.
    _npz_fixed_eye   = bool(npz['fixed_eye'])      if 'fixed_eye'   in npz else False
    _head_aligned    = _npz_worldcam or _npz_fixed_eye
    # The fused head position in native top-down pixels, at the render rate.
    # Older npz files predate it; the overlay then falls back to the raw 7.5 Hz
    # DLC anchors, which is what it always used to draw.
    head_x_top_px = npz['head_x_top_px'] if 'head_x_top_px' in npz else None
    head_y_top_px = npz['head_y_top_px'] if 'head_y_top_px' in npz else None
    if head_x_top_px is None:
        print('  Topdown position overlay: raw DLC anchors '
              '(npz predates head_x_top_px -- re-run get_retinal_image)')
    else:
        print(f'  Topdown position overlay: fused render position, finite on '
              f'{np.isfinite(head_x_top_px).mean() * 100:.1f}% of frames '
              f'(raw DLC anchors drawn hollow, finite on '
              f'{np.isfinite(head_x).mean() * 100:.1f}% of top-down frames)')
    # Worldcam alignment metrics (present only when get_retinal_image was run with --worldcam)
    _wc_eyeT = npz['eyeT']          if 'eyeT'          in npz else np.array([])
    _wc_flow = npz['wc_flow_error'] if 'wc_flow_error' in npz else np.array([])
    if len(_wc_flow) > 0:
        print(f'  worldcam flow error loaded: {len(_wc_flow)} samples')
    if _npz_worldcam:
        # Socket angle comes from the npz (worldcam_socket_deg, default 0 =
        # camera facing along head +X).  Older npz files predate this and
        # store the unused eye default; regenerate them.
        # Effective socket angle is stored in the npz by get_retinal_image.
        resting_tilt_h           = 0.0
        eye_offset_x = eye_offset_y = eye_offset_z = 0.0
        _npz_mean_pfh = _npz_mean_phi = 0.0
        print(f'  worldcam NPZ: anatomical_eye_angle_deg='
              f'{anatomical_eye_angle_deg:.1f} deg, no az_offset')
    _pan_az_offset = 0.0
    _pitch_finite = np.sum(np.isfinite(pitch))
    _roll_finite  = np.sum(np.isfinite(roll))
    print(f'  pitch: {_pitch_finite}/{len(pitch)} finite  range=[{np.nanmin(pitch):.1f}, {np.nanmax(pitch):.1f}] deg')
    print(f'  roll:  {_roll_finite}/{len(roll)} finite  range=[{np.nanmin(roll):.1f}, {np.nanmax(roll):.1f}] deg')

    theta_trim = theta_raw[startInd: startInd + len(eyeT_trim)]
    phi_trim   = phi_raw[startInd:   startInd + len(eyeT_trim)]

    px2cm  = 1.0 / pxls2cm
    px2mm_ = 10.0 / pxls2cm
    if _pillar_rad_px2 is not None:
        pillar_d = 2.0 * _pillar_rad_px2 * px2mm_
    hx_f  = head_x.astype(float)
    hy_f  = head_y.astype(float)
    dt_tp = np.diff(twopT)
    dx_cm = np.diff(hx_f) * px2cm
    dy_cm = np.diff(hy_f) * px2cm
    speed_twop = np.full(len(twopT), np.nan)
    ok = np.isfinite(dx_cm) & np.isfinite(dy_cm) & (dt_tp > 0)
    speed_twop[:-1][ok] = np.sqrt((dx_cm[ok] / dt_tp[ok])**2 +
                                   (dy_cm[ok] / dt_tp[ok])**2)

    theta_at_twop = np.interp(twopT, eyeT_trim, theta_trim.astype(float))
    phi_at_twop   = np.interp(twopT, eyeT_trim, phi_trim.astype(float))
    eye_ok_twop   = np.isfinite(theta_at_twop) & np.isfinite(phi_at_twop)

    best_idx, best_score = -1, -1
    for i in range(len(light_onsets)):
        lo  = int(light_onsets[i])
        nxt = dark_onsets[dark_onsets > lo]
        if len(nxt) == 0:
            continue
        nd  = int(nxt[0])
        nd  = min(nd, len(twopT) - 1)
        if nd <= lo:
            continue
        moving_block = speed_twop[lo:nd] > 2.0
        score = int(np.sum(moving_block & eye_ok_twop[lo:nd]))
        if score > best_score:
            best_score, best_idx = score, i

    if best_idx < 0:
        raise RuntimeError('No valid light block found.')

    lo      = int(light_onsets[best_idx])
    nd      = int(dark_onsets[dark_onsets > lo][0])
    nd      = min(nd, len(twopT) - 1)
    t_start = float(twopT[lo])
    t_end   = float(twopT[nd])

    nd      = min(nd, int(np.searchsorted(twopT, t_start + 80.0)))
    t_end   = float(twopT[nd])
    print(f'  Light block {best_idx}: 2P [{lo}:{nd}]  t=[{t_start:.1f}-{t_end:.1f}]s  '
          f'score={best_score} frames (moving + eye tracked)')

    eye_start    = int(np.searchsorted(eyeT_trim, t_start))
    eye_end      = int(np.searchsorted(eyeT_trim, t_end))
    eye_trim_idx = np.arange(eye_start, eye_end, _STRIDE)
    eye_full_idx = eye_trim_idx + startInd
    times        = eyeT_trim[eye_trim_idx]

    twop_idx      = np.searchsorted(twopT, times).clip(0, len(twopT) - 1)
    twop_idx_prev = (twop_idx - 1).clip(0)
    closer_prev   = (np.abs(twopT[twop_idx_prev] - times)
                     < np.abs(twopT[twop_idx] - times))
    twop_idx      = np.where(closer_prev, twop_idx_prev, twop_idx)

    n_frames = len(eye_trim_idx)
    print(f'  {n_frames} output frames  (stride={_STRIDE})')

    print('Pre-loading eye frames ...')
    t0 = time.time()
    eye_frames = np.empty((n_frames, _EYE_H, _EYE_W), dtype=np.uint8)
    cap = cv2.VideoCapture(eye_path)
    
    target_start = int(eye_full_idx[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(target_start):
        cap.grab()
            
    current = target_start
    for i, target in enumerate(eye_full_idx.astype(int)):
        skip = target - current
        for _ in range(skip):
            cap.grab()
        ret, frame = cap.read()
        if ret:
            _gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)

            # Band subtraction: remove per-row illumination gradient using edge columns
            _pr = gaussian_filter1d(
                0.5 * (_gr[:, :10].mean(axis=1) + _gr[:, -10:].mean(axis=1)), sigma=15
            )
            _gr -= _pr[:, np.newaxis]
            _gr -= _gr.min()
            _mx = _gr.max()
            if _mx > 0:
                _gr = _gr / _mx * 255.0
            _gu = np.clip(_gr, 0, 255).astype(np.uint8)
            # _fr = np.stack([_gu, _gu, _gu], axis=-1)
            eye_frames[i] = cv2.resize(_gu, (_EYE_W, _EYE_H))

        current = target + 1
    cap.release()
    print(f'  done in {time.time() - t0:.1f}s')

    print('Pre-loading topdown frames ...')
    t0  = time.time()
    cap = cv2.VideoCapture(top_path)
    top_fps_reported = cap.get(cv2.CAP_PROP_FPS)
    if top_fps_reported <= 0:
        top_fps_reported = 30.0
    print(f'  Topdown video reported fps: {top_fps_reported:.2f}')

    # Seek by time (POS_MSEC) rather than by frame number so we are
    # independent of cv2.CAP_PROP_FPS accuracy.  We record the actual
    # POS_MSEC timestamp of every frame we load, then build an index
    # that maps each output frame to its nearest topdown frame by time.
    seek_ms = max(0.0, t_start - 0.5) * 1000.0
    cap.set(cv2.CAP_PROP_POS_MSEC, seek_ms)

    # Pre-allocate generously; cap fps estimate to avoid huge allocs
    top_fps_est = min(max(top_fps_reported, 5.0), 120.0)
    n_top_est   = int((t_end - t_start + 2.0) * top_fps_est) + 20
    top_frames     = np.zeros((n_top_est, _TOP_H, _TOP_W, 3), dtype=np.uint8)
    top_buf_times  = np.full(n_top_est, np.nan)
    n_top_loaded   = 0
    orig_top_h, orig_top_w = _TOP_H, _TOP_W

    while n_top_loaded < n_top_est:
        ret, frame = cap.read()
        if not ret:
            break
        # POS_MSEC reports the presentation time of the frame that was just
        # decoded, so it must be read AFTER cap.read().  Reading it before the
        # read returns the *previous* frame's timestamp, which mislabels every
        # buffered frame by one topdown frame (133 ms at 7.5 fps) and makes the
        # tracking overlay appear to lag the topdown video.
        t_vid = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t_vid > t_end + 0.5:
            break
        if n_top_loaded == 0:
            orig_top_h, orig_top_w = frame.shape[:2]
        top_buf_times[n_top_loaded] = t_vid
        _fr = cv2.resize(frame, (_TOP_W, _TOP_H))
        # Band subtraction: estimate per-row illumination from edge columns and remove it,
        # then gamma-correct for visibility.
        _gr = cv2.cvtColor(_fr, cv2.COLOR_BGR2GRAY).astype(float)
        _pr = gaussian_filter1d(0.5 * (_gr[:, :10].mean(axis=1) +
                                        _gr[:, -10:].mean(axis=1)), sigma=15)
        _gr -= _pr[:, np.newaxis]
        _gr -= _gr.min()
        _mx = _gr.max()
        if _mx > 0:
            _gr = (_gr / _mx) ** 0.8 * 255.0
        _gu = np.clip(_gr, 0, 255).astype(np.uint8)
        top_frames[n_top_loaded] = np.stack([_gu, _gu, _gu], axis=-1)
        n_top_loaded += 1

    cap.release()

    top_buf_times = top_buf_times[:n_top_loaded]
    top_frames    = top_frames[:n_top_loaded]
    top_scale_x   = _TOP_W / orig_top_w
    top_scale_y   = _TOP_H / orig_top_h

    if n_top_loaded == 0:
        raise RuntimeError('Failed to load any topdown frames -- check top_path and t_start/t_end.')

    print(f'  done in {time.time() - t0:.1f}s  '
          f'({n_top_loaded} frames, '
          f't=[{top_buf_times[0]:.2f}–{top_buf_times[-1]:.2f}]s, '
          f'original {orig_top_w}x{orig_top_h})')

    # Build a direct lookup: for each output frame find the nearest topdown frame
    # by matching 2P timestamps (twopT[twop_idx]) to the actual video timestamps.
    target_top_times   = twopT[twop_idx].astype(float)
    top_idx_for_output = np.searchsorted(top_buf_times, target_top_times, side='left')
    top_idx_for_output = np.clip(top_idx_for_output, 0, n_top_loaded - 1)
    # Prefer left neighbour when it is closer
    left_idx     = np.maximum(top_idx_for_output - 1, 0)
    prefer_left  = (np.abs(top_buf_times[left_idx] - target_top_times) <
                    np.abs(top_buf_times[top_idx_for_output] - target_top_times))
    top_idx_for_output = np.where(prefer_left, left_idx, top_idx_for_output)

    # Map each buffered topdown frame to the tracking sample that belongs to it.
    # The overlay must be indexed by the frame actually being displayed, not by
    # the 2P index: any mismatch between twopT and the video's real timestamps
    # (e.g. twopT synthesised as arange(n)/fps) otherwise shows up as the
    # tracking dot trailing the animal.
    top_buf_track_idx = np.searchsorted(twopT, top_buf_times).clip(0, len(twopT) - 1)
    _prev = (top_buf_track_idx - 1).clip(0)
    _use_prev = (np.abs(twopT[_prev] - top_buf_times)
                 < np.abs(twopT[top_buf_track_idx] - top_buf_times))
    top_buf_track_idx = np.where(_use_prev, _prev, top_buf_track_idx)

    _resid = np.abs(twopT[top_buf_track_idx] - top_buf_times)
    print(f'  Topdown frame <-> tracking sample: max time residual '
          f'{np.max(_resid) * 1000:.1f} ms, median {np.median(_resid) * 1000:.1f} ms')

    pillar_x_top = pillar_x_px * top_scale_x
    pillar_y_top = pillar_y_px * top_scale_y
    print(f'  Pillar centroid: ({pillar_x_px:.1f}, {pillar_y_px:.1f}) px  ->  '
          f'scaled ({pillar_x_top:.1f}, {pillar_y_top:.1f})')

    # Head direction for the topdown overlay.  The renderer drives the retinal
    # image from the gyro-integrated yaw that has been drift-corrected against
    # the topdown ear vector (upsampled_yaw/igyro_corrected_deg), so the arrow
    # must be drawn from that same fused signal at the eye-camera rate -- not
    # from the raw 7.5 Hz DLC yaw, which is a different, coarser trace.
    # Prefer yaw_eye from the npz: that is the yaw the frames were rendered
    # through, and since get_retinal_image re-fuses gyro_z_trim with
    # head_yaw_deg it is NOT the same as the h5's stored upsampled_yaw (55.3 vs
    # 9.0 deg median error against the ear vector on 251016_DMM_DMM061_pos18).
    # Reading the stored copy here made the arrow disagree with the retinal
    # image it sits next to.
    _npz_yaw = npz['yaw_eye'] if 'yaw_eye' in npz else None
    if _npz_yaw is not None and len(_wc_eyeT) == len(_npz_yaw):
        _ok = np.isfinite(_npz_yaw)
        head_yaw_fused_out = np.interp(
            times, np.asarray(_wc_eyeT, float)[_ok],
            np.unwrap(np.radians(np.asarray(_npz_yaw, float)[_ok])) * 180.0 / np.pi,
        ) % 360.0
        print(f'  Topdown head-direction arrow: yaw_eye from the npz '
              f'(the yaw the frames were rendered through, '
              f'{int(_ok.sum())} finite samples)')
    elif imuT_trim_arr is not None and yaw_imu_arr is not None:
        _yaw_src = np.asarray(yaw_imu_arr, dtype=float)
        _n_yaw = min(len(imuT_trim_arr), len(_yaw_src))
        head_yaw_fused_out = np.interp(
            times, np.asarray(imuT_trim_arr, dtype=float)[:_n_yaw],
            np.unwrap(np.radians(_yaw_src[:_n_yaw])) * 180.0 / np.pi,
        ) % 360.0
        print('  Topdown head-direction arrow: WARNING stored upsampled_yaw '
              f'({_n_yaw} samples) -- npz has no yaw_eye, so the arrow may not '
              f'match the render. Re-run get_retinal_image.')
    else:
        head_yaw_fused_out = None
        print('  Topdown head-direction arrow: WARNING falling back to raw '
              'topdown yaw (upsampled_yaw not in H5)')

    # Panel crop.  The 2x-worldcam-field crop only means anything when there IS
    # a worldcam to compare against: it is a +-22.8 x +-17.3 deg window built
    # from that camera's intrinsics.  Applying it to an eye-camera render threw
    # away the rest of the 120 deg field for nothing -- on 251024_DMM_DMM061
    # the pillar centre falls inside that window on 6.9% of frames, so the
    # panel is blank almost always even when the geometry is right.
    if _npz_worldcam:
        _ret_xlim, _ret_ylim = retinal_display_limits(
            socket_az_deg=anatomical_eye_angle_deg,
            socket_el_deg=(float(npz['socket_elev_deg'])
                           if 'socket_elev_deg' in npz else WORLDCAM_SOCKET_EL_DEG),
            f_px=(float(npz['worldcam_f_px'])
                  if 'worldcam_f_px' in npz else WORLDCAM_F_PX))
        print(f'  Retinal panel cropped to 2x the world camera field: '
              f'az [{_ret_xlim[0]:.1f}, {_ret_xlim[1]:.1f}], '
              f'el [{_ret_ylim[0]:.1f}, {_ret_ylim[1]:.1f}] deg')
    else:
        _ret_xlim, _ret_ylim = (-60.0, 60.0), (-60.0, 60.0)
        print('  Retinal panel showing the full 120 deg modelled field')

    init_data = {
        'ret_xlim':                 _ret_xlim,
        'ret_ylim':                 _ret_ylim,
        'head_aligned':             _head_aligned,
        'head_yaw_fused_out':       head_yaw_fused_out,
        'top_buf_track_idx':        top_buf_track_idx,
        'eye_frames':               eye_frames,
        'top_frames':               top_frames,
        'retinal_images':           retinal_images,
        'eye_trim_idx':             eye_trim_idx,
        'twop_idx':                 twop_idx,
        'times':                    times,
        'twopT':                    twopT,
        'pitch':                    pitch,
        'roll':                     roll,
        'yaw':                      head_yaw[:len(twopT)],
        'theta_2p':                 theta_2p,
        'phi_2p':                   phi_2p,
        'twop_lo':                  lo,
        'twop_nd':                  nd,
        't_start':                  t_start,
        'pillar_x_top':             pillar_x_top,
        'pillar_y_top':             pillar_y_top,
        'head_x_arr':               head_x.astype(float),
        'head_y_arr':               head_y.astype(float),
        'head_x_top_px':            head_x_top_px,
        'head_y_top_px':            head_y_top_px,
        'eyeT_trim_full':           eyeT_trim,
        'theta_trim':               theta_trim_v,
        'phi_trim':                 phi_trim_v,
        'head_yaw_arr':             head_yaw[:len(twopT)].astype(float),
        'top_scale_x':              top_scale_x,
        'top_scale_y':              top_scale_y,
        'top_idx_for_output':       top_idx_for_output,
        'resting_tilt_h':           resting_tilt_h,
        'anatomical_eye_angle_deg': anatomical_eye_angle_deg,
        'worldcam':                 _npz_worldcam,
        'wc_eyeT':                  _wc_eyeT,
        'wc_flow':                  _wc_flow,
    }

    _probe = plt.figure(figsize=_FIGSIZE, dpi=_DPI)
    FIG_H  = int(_probe.get_figheight() * _DPI)
    FIG_W  = int(_probe.get_figwidth()  * _DPI)
    plt.close(_probe)
    FIG_H += FIG_H % 2
    FIG_W += FIG_W % 2
    print(f'Output: {FIG_W}x{FIG_H} px  |  {n_frames} frames @ {_VIDEO_FPS} fps')

    ffmpeg_bin, codec_args = _find_ffmpeg()
    ffmpeg_cmd = [
        ffmpeg_bin, '-y',
        '-f',        'rawvideo',
        '-vcodec',   'rawvideo',
        '-s',        f'{FIG_W}x{FIG_H}',
        '-pix_fmt',  'rgb24',
        '-r',        str(_VIDEO_FPS),
        '-i',        'pipe:0',
        *codec_args,
        '-pix_fmt',  'yuv420p',
        '-movflags', '+faststart',
        out_path,
    ]
    print(f'Writing -> {out_path}  (codec: {codec_args[1]})')
    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    valid_frames = np.arange(n_frames)

    n_workers = max(1, min(cpu_count() - 1, 8))
    print(f'Rendering with {n_workers} worker(s) ...')
    t0, n_written = time.time(), 0
    n_to_write = len(valid_frames)

    try:
        with Pool(
            processes=n_workers,
            initializer=_rv_worker_init,
            initargs=(init_data,),
        ) as pool:
            for frame_bytes in pool.imap(_rv_render_frame, valid_frames.tolist(), chunksize=4):
                ffmpeg_proc.stdin.write(frame_bytes)
                n_written += 1
                if n_written % 150 == 0:
                    elapsed = time.time() - t0
                    print(f'  {n_written}/{n_to_write}  ({n_written / elapsed:.1f} fps)')
    except BrokenPipeError:
        print('BrokenPipeError -- ffmpeg stderr:')
        print(ffmpeg_proc.stderr.read().decode(errors='replace'))
        raise

    ffmpeg_proc.stdin.close()
    retcode = ffmpeg_proc.wait()
    elapsed = time.time() - t0
    if retcode != 0:
        print(f'WARNING: ffmpeg exited {retcode}:')
        print(ffmpeg_proc.stderr.read().decode(errors='replace')[-3000:])
    print(f'Done in {elapsed:.1f}s  ({n_frames / elapsed:.1f} fps)  ->  {out_path}')


def make_retinal_diagnostic_pdf(h5_path, npz_path, out_pdf=None, arena_width_cm=None):

    if out_pdf is None:
        out_pdf = npz_path.replace('_retinal_images.npz', '_retinal_diagnostic.pdf')
        if out_pdf == npz_path:
            out_pdf = npz_path.replace('.npz', '_diagnostic.pdf')

    print(f'Building diagnostic PDF -> {out_pdf}')

    with h5py.File(h5_path, 'r') as f:
        eyeT_trim   = f['eyeT_trim'][:]
        theta_trim  = f['theta_trim'][:]
        phi_trim    = f['phi_trim'][:]
        gyro_z      = f['gyro_z_eye_interp'][:]
        imuT        = f['imuT_trim'][:]
        yaw_imu     = f['upsampled_yaw']['igyro_corrected_deg'][:]
        twopT       = f['twopT'][:]
        head_x_2p   = f['head_x'][:]
        head_y_2p   = f['head_y'][:]
        pxls2cm     = float(f['pxls2cm'][()])
        if arena_width_cm is not None:
            try:
                arenaTL_x = float(f['arenaTL']['x'][()])
                arenaTR_x = float(f['arenaTR']['x'][()])
                arenaBL_x = float(f['arenaBL']['x'][()])
                arenaBR_x = float(f['arenaBR']['x'][()])
                px_width = np.mean([arenaTR_x - arenaTL_x, arenaBR_x - arenaBL_x])
                pxls2cm = px_width / arena_width_cm
                print(f'  [PDF] Overriding pxls2cm with value calculated from arena_width_cm={arena_width_cm:.1f} -> {pxls2cm:.3f}')
            except KeyError:
                print('  [PDF] WARNING: Could not find arena corner keys in HDF5 to override pxls2cm.')
        pillar_x_px, pillar_y_px = _read_pillar_centroid_px(f)

    npz            = np.load(npz_path)
    retinal_images = npz['retinal_images']
    ang_offset     = float(npz['ang_offset'])
    vor_gain       = float(npz['vor_gain'])
    pillar_x_mm    = float(npz['pillar_x_mm'])
    pillar_y_mm    = float(npz['pillar_y_mm'])

    px2mm = 10.0 / pxls2cm
    N = min(len(eyeT_trim), len(retinal_images))

    yaw_eye = np.interp(eyeT_trim[:N], imuT, yaw_imu)
    pfh_eye = ang_offset - theta_trim[:N]

    twop_valid_x = np.isfinite(head_x_2p)
    twop_valid_y = np.isfinite(head_y_2p)
    head_x_eye =  np.interp(eyeT_trim[:N],
                             twopT[twop_valid_x], head_x_2p[twop_valid_x]) * px2mm
    head_y_eye = -np.interp(eyeT_trim[:N],
                             twopT[twop_valid_y], head_y_2p[twop_valid_y]) * px2mm

    dx = pillar_x_mm - head_x_eye
    dy = pillar_y_mm - head_y_eye
    world_ang      = np.degrees(np.arctan2(dy, dx))
    dist_to_pillar = np.sqrt(dx**2 + dy**2)

    ego_ang    = (world_ang - yaw_eye + 180) % 360 - 180
    gaze       = yaw_eye + pfh_eye
    retino_ang = (world_ang - gaze + 180) % 360 - 180

    has_retinal = np.array([retinal_images[i].sum() > 0 for i in range(N)], dtype=bool)
    nz_indices  = np.where(has_retinal)[0]
    pct_nz      = 100.0 * has_retinal.mean()
    print(f'  Non-zero retinal frames: {has_retinal.sum()} / {N}  ({pct_nz:.1f}%)')

    with PdfPages(out_pdf) as pdf:

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Fig 1: Mouse Trajectory & Pillar Position', fontsize=12)

        ax = axes[0]
        valid = np.isfinite(head_x_eye) & np.isfinite(head_y_eye)
        sc = ax.scatter(head_x_eye[valid], head_y_eye[valid],
                        c=eyeT_trim[:N][valid], cmap='viridis', s=1, alpha=0.4)
        # ax.plot(pillar_x_mm, pillar_y_mm, 'r*', ms=14, label='Pillar', zorder=10)
        plt.colorbar(sc, ax=ax, label='Time (s)')
        ax.set_xlabel('Head X (mm)')
        ax.set_ylabel('Head Y (mm)')
        ax.set_title('Head trajectory (mm, eye-cam rate)')
        ax.legend()
        ax.set_aspect('equal')

        ax = axes[1]
        valid2 = np.isfinite(head_x_2p) & np.isfinite(head_y_2p)
        sc2 = ax.scatter(head_x_2p[valid2], -head_y_2p[valid2],
                         c=twopT[valid2], cmap='viridis', s=1, alpha=0.3)
        # ax.plot(pillar_x_px, -pillar_y_px, 'r*', ms=14, label='Pillar', zorder=10)
        plt.colorbar(sc2, ax=ax, label='Time (s)')
        ax.set_xlabel('Head X (px, x-flipped)')
        ax.set_ylabel('Head Y (px, y-flipped to math convention)')
        ax.set_title('Head trajectory (pixels, 2-P rate)')
        ax.legend()
        ax.set_aspect('equal')

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(
            f'Fig 2: Eye Angle Distributions  '
            f'(ang_offset={ang_offset:.2f}°,  VOR gain={vor_gain:.3f})',
            fontsize=12)

        theta_t = theta_trim[:N]
        phi_t   = phi_trim[:N]
        gyro_t  = gyro_z[:N]

        ax = axes[0, 0]
        fin = theta_t[np.isfinite(theta_t)]
        ax.hist(fin, bins=80, color='steelblue', alpha=0.8)
        ax.axvline(np.nanmedian(theta_t), color='r',      lw=2,
                   label=f'median={np.nanmedian(theta_t):.1f}°')
        ax.axvline(ang_offset,            color='orange', lw=2, ls='--',
                   label=f'ang_offset={ang_offset:.1f}°')
        ax.set_xlabel('theta -- raw pupil horiz angle (deg)')
        ax.set_title('theta distribution')
        ax.legend(fontsize=8)

        ax = axes[0, 1]
        fin = phi_t[np.isfinite(phi_t)]
        ax.hist(fin, bins=80, color='coral', alpha=0.8)
        ax.axvline(np.nanmedian(phi_t), color='r', lw=2,
                   label=f'median={np.nanmedian(phi_t):.1f}°')
        ax.set_xlabel('phi -- pupil elevation (deg)')
        ax.set_title('phi distribution')
        ax.legend(fontsize=8)

        ax = axes[1, 0]
        pfh_fin = pfh_eye[np.isfinite(pfh_eye)]
        ax.hist(pfh_fin, bins=80, color='mediumseagreen', alpha=0.8)
        ax.axvline(np.nanmedian(pfh_eye), color='r', lw=2,
                   label=f'median={np.nanmedian(pfh_eye):.1f}°')
        ax.axvline(0, color='k', lw=1, ls='--', label='zero')
        ax.set_xlabel('pfh = ang_offset - theta  (eye-in-head horiz, deg)')
        ax.set_title('pfh distribution')
        ax.legend(fontsize=8)

        ax = axes[1, 1]
        stride = max(1, N // 4000)
        ax.scatter(theta_t[::stride], pfh_eye[::stride], s=1, alpha=0.2, color='purple')
        ax.axhline(0,          color='k',      lw=0.5, ls='--')
        ax.axvline(ang_offset, color='orange', lw=1,   ls='--',
                   label=f'ang_offset={ang_offset:.1f}°')
        ax.set_xlabel('theta (deg)')
        ax.set_ylabel('pfh (deg)')
        ax.set_title('theta vs pfh')
        ax.legend(fontsize=8)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Fig 3: VOR Analysis', fontsize=12)

        ax = axes[0]
        valid_vor = np.isfinite(theta_t) & np.isfinite(gyro_t)
        stride2 = max(1, valid_vor.sum() // 4000)
        ax.scatter(gyro_t[valid_vor][::stride2], theta_t[valid_vor][::stride2],
                   s=1, alpha=0.2, color='royalblue')
        ax.set_xlabel('gyro_z (deg/s)')
        ax.set_ylabel('theta (deg)')
        ax.set_title(f'theta vs gyro_z\n(VOR gain={vor_gain:.3f},  null={ang_offset:.1f}°)')

        ax = axes[1]
        stride3 = max(1, N // 2000)
        t_sub = eyeT_trim[:N][::stride3]
        l1, = ax.plot(t_sub, yaw_eye[::stride3] % 360, 'b-', lw=0.8, label='yaw (°)')
        ax2r = ax.twinx()
        l2, = ax2r.plot(t_sub, pfh_eye[::stride3], 'g-', lw=0.8, alpha=0.7, label='pfh (°)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Head yaw (deg)', color='b')
        ax2r.set_ylabel('pfh (deg)', color='g')
        ax.set_title('Yaw and pfh over time')
        ax.legend(handles=[l1, l2], fontsize=8)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Fig 4: Pillar Angle in Head / Gaze Frame', fontsize=12)

        ax = axes[0]
        valid_ang = np.isfinite(ego_ang)
        ax.hist(ego_ang[valid_ang], bins=90, range=(-180, 180),
                color='steelblue', alpha=0.8)
        ax.axvline(0, color='r', lw=1.5, ls='--', label='ahead')
        ax.set_xlabel('Pillar angle egocentric (deg)')
        ax.set_title('Pillar in head frame\n(0 = ahead, + = CCW)')
        ax.legend()

        ax = axes[1]
        valid_r = np.isfinite(retino_ang)
        ax.hist(retino_ang[valid_r], bins=90, range=(-180, 180),
                color='coral', alpha=0.8)
        ax.axvline(0, color='r', lw=1.5, ls='--', label='on gaze axis')
        ax.set_xlabel('Pillar angle retinocentric (deg)')
        ax.set_title('Pillar in gaze frame\n(0 = fovea, + = CCW)')
        ax.legend()

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle('Fig 5: Retinal Image Coverage Summary', fontsize=12)

        ret_float   = retinal_images[:N].astype(float)
        mean_img    = ret_float.mean(axis=0)
        nonzero_frac = (ret_float > 0).mean(axis=0)

        ax = axes[0]
        im = ax.imshow(mean_img, cmap='hot', aspect='equal')
        plt.colorbar(im, ax=ax, label='Mean intensity')
        ax.axhline(60, color='w', lw=0.5, ls='--')
        ax.axvline(60, color='w', lw=0.5, ls='--')
        ax.set_title('Mean retinal image (all frames)')
        ax.set_xlabel('Azimuth (deg)')
        ax.set_ylabel('Elevation (deg)')

        ax = axes[1]
        im2 = ax.imshow(nonzero_frac, cmap='plasma', aspect='equal', vmin=0, vmax=1)
        plt.colorbar(im2, ax=ax, label='Fraction of frames')
        ax.axhline(60, color='w', lw=0.5, ls='--')
        ax.axvline(60, color='w', lw=0.5, ls='--')
        ax.set_title('Fraction of frames with pillar at each pixel')
        ax.set_xlabel('Azimuth (deg)')

        ax = axes[2]
        win = min(500, max(N // 20, 1))
        if N > win:
            kernel = np.ones(win) / win
            frac_nz = np.convolve(has_retinal[:N].astype(float), kernel, mode='valid')
            t_frac  = eyeT_trim[:len(frac_nz)]
            ax.plot(t_frac, frac_nz, 'g-', lw=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Fraction non-zero')
        ax.set_title(f'Temporal coverage\n(overall {pct_nz:.1f}% non-zero frames)')
        ax.set_ylim(0, 1)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if len(nz_indices) > 0:
            n_show  = min(24, len(nz_indices))
            sel_idx = nz_indices[
                np.linspace(0, len(nz_indices) - 1, n_show).astype(int)
            ]
            ncols = 6
            nrows = (n_show + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.4 * nrows))
            fig.suptitle(
                f'Fig 6: Sample Retinal Frames -- non-zero only  '
                f'({len(nz_indices)} total,  {n_show} shown)',
                fontsize=12)

            axes_flat = np.array(axes).ravel()
            for k, idx in enumerate(sel_idx):
                ax = axes_flat[k]
                ax.imshow(retinal_images[idx], cmap='hot', vmin=0, vmax=255,
                          aspect='equal', interpolation='nearest')
                ax.axhline(60, color='c', lw=0.5, ls='--')
                ax.axvline(60, color='c', lw=0.5, ls='--')
                t_lbl = eyeT_trim[idx] if idx < len(eyeT_trim) else idx
                ax.set_title(f't={t_lbl:.1f}s', fontsize=7)
                ax.axis('off')

            for k in range(n_show, len(axes_flat)):
                axes_flat[k].axis('off')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        if len(nz_indices) > 0:
            cx_list, cy_list, t_list = [], [], []
            for idx in nz_indices:
                img = retinal_images[idx]
                ys, xs = np.where(img > 0)
                cx_list.append(xs.mean() - 60)
                cy_list.append(-(ys.mean() - 60))
                t_list.append(eyeT_trim[idx] if idx < len(eyeT_trim) else float(idx))

            cx_arr = np.array(cx_list)
            cy_arr = np.array(cy_list)
            t_arr  = np.array(t_list)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Fig 7: Pillar Centroid Position in Retinal Image', fontsize=12)

            ax = axes[0]
            sc = ax.scatter(cx_arr, cy_arr, c=t_arr, cmap='viridis', s=4, alpha=0.5)
            plt.colorbar(sc, ax=ax, label='Time (s)')
            ax.axhline(0, color='k', lw=0.5, ls='--')
            ax.axvline(0, color='k', lw=0.5, ls='--')
            ax.set_xlim(-60, 60)
            ax.set_ylim(-60, 60)
            ax.set_xlabel('Azimuth offset from center (deg, +right)')
            ax.set_ylabel('Elevation offset from center (deg, +up)')
            ax.set_title('Pillar centroid -- scatter (colored by time)')
            ax.set_aspect('equal')

            ax = axes[1]
            ax.hist2d(cx_arr, cy_arr, bins=30, range=[[-60, 60], [-60, 60]], cmap='hot')
            ax.axhline(0, color='c', lw=0.5, ls='--')
            ax.axvline(0, color='c', lw=0.5, ls='--')
            ax.set_xlabel('Azimuth (deg, +right)')
            ax.set_ylabel('Elevation (deg, +up)')
            ax.set_title('Pillar centroid -- 2-D histogram')
            ax.set_aspect('equal')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f'Saved -> {out_pdf}')
    return out_pdf


def make_synthetic_retinal_diagnostic_pdf(
    out_pdf,
    h5_path,
    npz_path=None,
    arena_w_mm=600.0,
    arena_h_mm=600.0,
    pillar_d=40.0,
    pillar_h=210.0,
    eye_offset_x=3.5,
    eye_offset_y=-5.0,
    eye_offset_z=3.5,
    anatomical_eye_angle_deg=-65.0,
    n_pages=10,
    frames_per_page=5,
    seed=0
):

    _npz_path = npz_path
    if _npz_path is None:
        _npz_path = h5_path.replace('_preproc.h5', '_retinal_images.npz')
    print(f'  Reading resting orientations from {os.path.basename(h5_path)}')
    with h5py.File(h5_path, 'r') as f:
        theta_trim = f['theta_trim'][:]
        phi_trim   = f['phi_trim'][:]
        pitch_eye  = f['pitch_eye_interp'][:]
        roll_eye   = f['roll_eye_interp'][:]
    ang_offset     = float(np.load(_npz_path)['ang_offset'])
    pfh            = ang_offset - theta_trim
    resting_tilt_h = float(np.nanmean(pfh))
    resting_tilt_v = float(np.nanmean(phi_trim))
    resting_pitch  = float(np.nanmean(pitch_eye))
    resting_roll   = float(np.nanmean(roll_eye))
    print(f'    ang_offset={ang_offset:.2f}°  mean_theta={np.nanmean(theta_trim):.2f}°')
    print(f'    -> tilt_h={resting_tilt_h:.2f}°  tilt_v={resting_tilt_v:.2f}°  '
            f'pitch={resting_pitch:.2f}°  roll={resting_roll:.2f}°')

    rng    = np.random.default_rng(seed)
    margin = max(pillar_d, 50.0)

    def _optical_corners(pillar_x, pillar_y, mouse_x, mouse_y, mouse_yaw_ccw,
                         tilt_h=0.0, tilt_v=0.0, pitch=0.0, roll=0.0,
                         anatomical_eye_angle_deg=-65.0):

        head_z = 25.0
        r = pillar_d / 2.0
        corners_w = np.array([
            [pillar_x + r, pillar_y + r, 0,        1],
            [pillar_x + r, pillar_y - r, 0,        1],
            [pillar_x - r, pillar_y + r, 0,        1],
            [pillar_x - r, pillar_y - r, 0,        1],
            [pillar_x + r, pillar_y + r, pillar_h, 1],
            [pillar_x + r, pillar_y - r, pillar_h, 1],
            [pillar_x - r, pillar_y + r, pillar_h, 1],
            [pillar_x - r, pillar_y - r, pillar_h, 1],
        ]).T

  
        R_head = R.from_euler('zyx', [mouse_yaw_ccw, pitch, roll], degrees=True).as_matrix()
        t_head = np.array([[mouse_x], [mouse_y], [head_z]])
        M_head = np.eye(4)
        M_head[:3, :3] = R_head
        M_head[:3, 3:] = t_head
        corners_head = np.linalg.inv(M_head) @ corners_w


        R_saccade = R.from_euler('zyx', [tilt_h, tilt_v, 0.0], degrees=True).as_matrix()
        R_socket = R.from_euler('z', anatomical_eye_angle_deg, degrees=True).as_matrix()
        R_eye = R_socket @ R_saccade
        t_eye = np.array([[eye_offset_x], [eye_offset_y], [eye_offset_z]])
        M_eye = np.eye(4)
        M_eye[:3, :3] = R_eye
        M_eye[:3, 3:] = t_eye
        corners_eye = np.linalg.inv(M_eye) @ corners_head

        R_align = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)
        # tried changing to R_align = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)
        return R_align @ corners_eye[:3, :]

    def _render_retinal(corners_opt):

        fov_deg = 120.0
        res = 120
        f  = (res / 2) / np.tan(np.radians(fov_deg / 2))
        cx = cy = res / 2
        K  = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=float)

        img = np.zeros((res, res), dtype=np.uint8)
        pts = []
        for i in range(8):
            pt = corners_opt[:, i]
            if pt[2] <= 0:
                continue
            proj = K @ pt
            pts.append([int(proj[0] / proj[2]), int(proj[1] / proj[2])])

        if pts:
            arr = np.array(pts, dtype=np.int32)
            hull = cv2.convexHull(arr)
            cv2.fillPoly(img, [hull], color=255)
        return img

    def _render_panoramic(corners_opt, pan_w=360, pan_h=120, az_offset=0.0):
        img = np.zeros((pan_h, pan_w), dtype=np.uint8)
        el_list = []
        for i in range(8):
            pt = corners_opt[:, i]
            lateral = np.sqrt(pt[0] ** 2 + pt[2] ** 2)
            el = np.degrees(np.arctan2(-pt[1], lateral))
            el_list.append(el)
        el_arr = np.array(el_list)
        center = corners_opt.mean(axis=1)
        az_c = np.degrees(np.arctan2(center[0], center[2])) + az_offset
        az_corners = np.degrees(np.arctan2(corners_opt[0, :], corners_opt[2, :]))
        az_delta = ((az_corners - (az_c - az_offset) + 180.0) % 360.0) - 180.0
        az_half = min(float(np.max(np.abs(az_delta))), 90.0)
        u_min = int(np.floor(az_c - az_half + pan_w / 2))
        u_max = int(np.ceil( az_c + az_half + pan_w / 2))
        v_min = int(np.floor(pan_h / 2 - el_arr.max()))
        v_max = int(np.ceil( pan_h / 2 - el_arr.min()))
        v_min = max(0, v_min)
        v_max = min(pan_h - 1, v_max)
        for u in range(u_min, u_max + 1):
            img[v_min:v_max + 1, u % pan_w] = 255
        return img

    print(f'Building synthetic retinal diagnostic PDF -> {out_pdf}')
    print(f'  {n_pages} pages x {frames_per_page} frames/page  seed={seed}')
    print(f'  resting: pitch={resting_pitch}°  roll={resting_roll}°  '
          f'tilt_h={resting_tilt_h}°  tilt_v={resting_tilt_v}°')

    col_w = 4.0
    fig_w = col_w * frames_per_page
    fig_h = col_w * 2.5

    with PdfPages(out_pdf) as pdf:
        for page in range(n_pages):
            fig = plt.figure(figsize=(fig_w, fig_h), facecolor='k')
            gs  = GridSpec(
                3, frames_per_page, figure=fig,
                height_ratios=[2, 2, 1],
                hspace=0.15, wspace=0.08,
                left=0.03, right=0.97, top=0.91, bottom=0.04,
            )

            for col in range(frames_per_page):

                px  = float(rng.uniform(margin, arena_w_mm - margin))
                py  = float(rng.uniform(margin, arena_h_mm - margin))
                mx  = float(rng.uniform(margin, arena_w_mm - margin))
                my  = float(rng.uniform(margin, arena_h_mm - margin))
                yaw = float(rng.uniform(0.0, 360.0))   # CCW from east (math convention)
                yaw_rad  = np.radians(yaw)

                corners_opt = _optical_corners(
                    px, py, mx, my, yaw,
                    tilt_h=resting_tilt_h, tilt_v=resting_tilt_v,
                    pitch=resting_pitch,   roll=resting_roll,
                    anatomical_eye_angle_deg=anatomical_eye_angle_deg,
                )

                ax0 = fig.add_subplot(gs[0, col])
                ax0.set_facecolor('k')
                ax0.set_aspect('equal')

                # arena box
                ax0.add_patch(plt.Polygon(
                    [[0, 0], [arena_w_mm, 0],
                     [arena_w_mm, arena_h_mm], [0, arena_h_mm]],
                    closed=True, fill=False, edgecolor='white', lw=1.5, zorder=1,
                ))

                ax0.add_patch(plt.Circle(
                    (px, py), radius=pillar_d / 2,
                    color='red', alpha=0.9, zorder=5,
                ))

                arrow_len  = arena_w_mm * 0.09
                gaze_angle = np.radians(yaw + resting_tilt_h)
                gaze_angle = np.radians(yaw + anatomical_eye_angle_deg + resting_tilt_h)
                ax0.plot(mx, my, 'o', color='#ffff00', markersize=7,
                         markeredgecolor='k', markeredgewidth=0.8, zorder=10)
                ax0.plot(
                    [mx, mx + arrow_len * np.cos(yaw_rad)],
                    [my, my + arrow_len * np.sin(yaw_rad)],
                    '-', color='#ffff00', lw=2, zorder=11,
                )

                gaze_len = arrow_len * 0.70
                ax0.plot(
                    [mx, mx + gaze_len * np.cos(gaze_angle)],
                    [my, my + gaze_len * np.sin(gaze_angle)],
                    '-', color='#00cc44', lw=2, zorder=12,
                )

                pad = arena_w_mm * 0.06
                ax0.set_xlim(-pad, arena_w_mm + pad)
                ax0.set_ylim(-pad, arena_h_mm + pad)
                ax0.axis('off')
                ax0.set_title(
                    f'yaw={yaw:.0f}°  θ={resting_tilt_h:.0f}°',
                    color='0.65', fontsize=8, pad=2,
                )

                ax1 = fig.add_subplot(gs[1, col])
                ax1.set_facecolor('k')
                ret_disp = _render_retinal(corners_opt)
                ax1.imshow(
                    ret_disp, cmap=_RET_CMAP, vmin=0, vmax=255,
                    aspect='equal', interpolation='nearest',
                    extent=[-60, 60, -60, 60],
                )
                ax1.axhline(0, color='0.35', lw=0.5, ls='--')
                ax1.axvline(0, color='0.35', lw=0.5, ls='--')
                ax1.tick_params(colors='0.45', labelsize=6)
                ax1.set_xlabel('Az (°)', color='0.5', fontsize=7)
                ax1.set_ylabel('El (°)', color='0.5', fontsize=7)
                for sp in ax1.spines.values():
                    sp.set_color('0.3')

                ax2 = fig.add_subplot(gs[2, col])
                ax2.set_facecolor('k')
                pan_disp = _render_panoramic(corners_opt, pan_w=360, pan_h=120)
                ax2.imshow(
                    pan_disp, cmap=_RET_CMAP, vmin=0, vmax=255,
                    aspect='auto', interpolation='nearest',
                    extent=[-180, 180, -60, 60],
                )

                ax2.axvline(-60, color='white', lw=1.0, ls='--', alpha=0.75)
                ax2.axvline( 60, color='white', lw=1.0, ls='--', alpha=0.75)
                ax2.axhline(0, color='0.3', lw=0.4, ls='--')
                ax2.set_xlim(-180, 180)
                ax2.set_ylim(-60, 60)
                ax2.tick_params(colors='0.45', labelsize=6)
                ax2.set_xlabel('Az (°)', color='0.5', fontsize=7)
                for sp in ax2.spines.values():
                    sp.set_color('0.3')

            pdf.savefig(fig, facecolor='k')
            plt.close(fig)

    print(f'Saved -> {out_pdf}')
    return out_pdf


def _render_eye_schematic(theta_deg, phi_deg, w=_EYE_W, h=_EYE_H):

    img = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    pxd = 5.0
    cv2.circle(img, (cx, cy), min(cx, cy) - 20, 50, 2)
    pu = int(np.clip(cx + theta_deg * pxd, 40, w - 40))
    pv = int(np.clip(cy - phi_deg   * pxd, 40, h - 40))
    cv2.ellipse(img, (pu, pv), (32, 32), 0, 0, 360, 200, -1)
    cv2.ellipse(img, (pu, pv), (36, 36), 0, 0, 360, 240, 2)
    cv2.line(img, (cx - 14, cy), (cx + 14, cy), 30, 1)
    cv2.line(img, (cx, cy - 14), (cx, cy + 14), 30, 1)
    return img


def make_dummy_diagnostic(
    out_dir='.',
    out_video=None,
    out_pdf=None,
    pillar_x_mm=0.0,
    pillar_y_mm=0.0,
    arena_w_mm=600.0,
    arena_h_mm=600.0,
    pillar_d=40.0,
    pillar_h=210.0,
    eye_offset_x=3.5,
    eye_offset_y=-5.0,
    eye_offset_z=3.5,
    anatomical_eye_angle_deg=-65.0,
    seed=42,
):

    ts = time.strftime('%Y%m%d_%H%M%S')
    if out_video is None:
        out_video = os.path.join(out_dir, f'retinal_dummy_{ts}.mp4')
    if out_pdf is None:
        out_pdf   = os.path.join(out_dir, f'retinal_dummy_{ts}_sweeps.pdf')

    rng    = np.random.default_rng(seed)
    margin = 80.0
    mx = my = 0.0
    for _ in range(400):
        dist = rng.uniform(100.0, min(arena_w_mm, arena_h_mm) / 3.0)
        ang  = rng.uniform(0, 2 * np.pi)
        mx   = pillar_x_mm + dist * np.cos(ang)
        my   = pillar_y_mm + dist * np.sin(ang)
        if (-arena_w_mm / 2 + margin < mx < arena_w_mm / 2 - margin and
                -arena_h_mm / 2 + margin < my < arena_h_mm / 2 - margin):
            break
    dx, dy = pillar_x_mm - mx, pillar_y_mm - my
    yaw_to_pillar = np.degrees(np.arctan2(dy, dx))
    yaw_for_gaze  = yaw_to_pillar - anatomical_eye_angle_deg
    print(f'Mouse:  ({mx:.1f}, {my:.1f}) mm   Pillar: ({pillar_x_mm:.1f}, {pillar_y_mm:.1f}) mm')
    print(f'yaw_to_pillar={yaw_to_pillar:.1f} deg  yaw_for_gaze={yaw_for_gaze:.1f} deg  '
          f'(anat_eye_angle={anatomical_eye_angle_deg:.0f} deg)')

    n_yaw   = int(30 * _VIDEO_FPS)
    n_other = int(15 * _VIDEO_FPS)
    n_total = n_yaw + 4 * n_other

    sweep_names  = ['Yaw', 'Pitch', 'Roll', 'Theta', 'Phi']
    sweep_frames = [n_yaw, n_other, n_other, n_other, n_other]
    sweep_ranges = [(0.0, 360.0), (-30.0, 30.0), (-30.0, 30.0), (-40.0, 40.0), (-30.0, 30.0)]

    yaw_arr   = np.full(n_total, yaw_for_gaze)
    pitch_arr = np.zeros(n_total)
    roll_arr  = np.zeros(n_total)
    theta_arr = np.zeros(n_total)
    phi_arr   = np.zeros(n_total)
    sweep_id  = np.zeros(n_total, dtype=int)

    starts = np.cumsum([0] + sweep_frames[:-1])
    for si, (name, (lo, hi), nf, s0) in enumerate(
            zip(sweep_names, sweep_ranges, sweep_frames, starts)):
        sl   = slice(s0, s0 + nf)
        vals = np.linspace(lo, hi, nf, endpoint=(si != 0))
        if   name == 'Yaw':   yaw_arr[sl]   = vals
        elif name == 'Pitch': pitch_arr[sl]  = vals
        elif name == 'Roll':  roll_arr[sl]   = vals
        elif name == 'Theta': theta_arr[sl]  = vals
        elif name == 'Phi':   phi_arr[sl]    = vals
        sweep_id[sl] = si

    sweep_slices = [slice(int(starts[i]), int(starts[i]) + sweep_frames[i])
                    for i in range(len(sweep_names))]
    sweep_arrs   = [yaw_arr, pitch_arr, roll_arr, theta_arr, phi_arr]
    times        = np.arange(n_total) / _VIDEO_FPS

    print(f'Computing {n_total} retinal frames ...')
    t0 = time.time()
    retinal_images   = np.zeros((n_total, 120, 120), dtype=np.uint8)
    panoramic_images = np.zeros((n_total, 120, 360), dtype=np.uint8)
    for i in range(n_total):
        retinal_images[i] = simulate_retinal_projection(
            pillar_x_mm, pillar_y_mm, mx, my,
            yaw_arr[i], pitch_arr[i], roll_arr[i],
            theta_arr[i], phi_arr[i],
            pillar_h=pillar_h, pillar_d=pillar_d,
            eye_offset_x=eye_offset_x, eye_offset_y=eye_offset_y,
            eye_offset_z=eye_offset_z,
            anatomical_eye_angle_deg=anatomical_eye_angle_deg,
        )
        panoramic_images[i] = _render_panoramic(
            _pillar_optical_corners(
                pillar_x_mm, pillar_y_mm, mx, my,
                yaw_arr[i], pitch_arr[i], roll_arr[i],
                theta_arr[i], phi_arr[i],
                pillar_h=pillar_h, pillar_d=pillar_d,
                eye_offset_x=eye_offset_x, eye_offset_y=eye_offset_y,
                eye_offset_z=eye_offset_z,
                anatomical_eye_angle_deg=anatomical_eye_angle_deg,
            )
        )
        if (i + 1) % 600 == 0:
            print(f'  {i+1}/{n_total}')
    # retinal_images = np.flip(retinal_images, axis=2)
    print(f'  done in {time.time() - t0:.1f}s')

    fig = plt.figure(figsize=_FIGSIZE, dpi=_DPI, facecolor=_FIG_BG)
    gs  = GridSpec(
        3, 2, figure=fig,
        width_ratios=[1, 2.2], height_ratios=[1, 1, 1.2],
        hspace=0.08, wspace=0.08,
        left=0.10, right=0.97, top=0.97, bottom=0.05,
    )
    ax_td  = fig.add_subplot(gs[0, 0])
    ax_eye = fig.add_subplot(gs[1, 0])
    gs_right = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[0:3, 1],
        height_ratios=[1.8, 1.0], hspace=0.30,
    )
    ax_retina = fig.add_subplot(gs_right[0])
    ax_pano   = fig.add_subplot(gs_right[1])
    gs_tr = GridSpecFromSubplotSpec(5, 1, subplot_spec=gs[2, 0], hspace=0.06)
    tr_axes = tuple(fig.add_subplot(gs_tr[k]) for k in range(5))
    ax_pitch_tr, ax_roll_tr, ax_yaw_tr, ax_theta_tr, ax_phi_tr = tr_axes

    pad_td = max(arena_w_mm, arena_h_mm) * 0.08
    ax_td.set_facecolor('#0a0a0a')
    ax_td.set_aspect('equal')
    ax_td.axis('off')
    ax_td.set_xlim(-arena_w_mm / 2 - pad_td,  arena_w_mm / 2 + pad_td)
    ax_td.set_ylim(-arena_h_mm / 2 - pad_td,  arena_h_mm / 2 + pad_td)
    from matplotlib.patches import Rectangle as _DRect, Circle as _DCirc
    ax_td.add_patch(_DRect(
        (-arena_w_mm / 2, -arena_h_mm / 2), arena_w_mm, arena_h_mm,
        fill=False, edgecolor='0.4', lw=1.5, zorder=2,
    ))
    ax_td.add_patch(_DCirc(
        (pillar_x_mm, pillar_y_mm), max(pillar_d / 2, pad_td * 0.14),
        color='red', zorder=8,
    ))
    arrow_td = min(arena_w_mm, arena_h_mm) * 0.09
    _td_head, = ax_td.plot([mx], [my], 'o', color='#ffff00', markersize=8,
                            markeredgewidth=1, markeredgecolor='k', zorder=20)
    _td_hdir, = ax_td.plot([], [], '-', color='#ffff00', lw=2, zorder=21)
    _td_gdir, = ax_td.plot([], [], '-', color='#00ff55', lw=2, zorder=22)
    _td_lbl   = ax_td.text(
        -arena_w_mm / 2 + pad_td * 0.3,
         arena_h_mm / 2 - pad_td * 0.3,
        '', color='0.65', fontsize=7, va='top', zorder=30,
    )

    ax_eye.set_facecolor(_FIG_BG)
    ax_eye.axis('off')
    im_eye_d = ax_eye.imshow(
        _render_eye_schematic(0.0, 0.0),
        cmap='gray', vmin=0, vmax=255, aspect='equal', interpolation='nearest',
    )

    ax_retina.set_facecolor(_FIG_BG)
    im_ret_d = ax_retina.imshow(
        np.zeros((120, 120), dtype=np.uint8),
        cmap=_RET_CMAP, vmin=0, vmax=255, aspect='equal',
        interpolation='nearest', extent=[-60, 60, -60, 60],
    )
    ax_retina.set_xlim(-60, 60)
    ax_retina.set_ylim(-60, 60)
    ax_retina.set_title('Estimated retinal image', color='0.6', fontsize=11, pad=4)
    ax_retina.axhline(0, color='0.3', lw=0.6, ls='--')
    ax_retina.axvline(0, color='0.3', lw=0.6, ls='--')
    ax_retina.tick_params(colors='0.4', labelsize=8)
    ax_retina.set_xlabel('Azimuth (deg)', color='0.5', fontsize=9)
    ax_retina.set_ylabel('Elevation (deg)', color='0.5', fontsize=9)
    for sp in ax_retina.spines.values():
        sp.set_color('0.3')
    ax_retina.axis('on')

    im_pan_d = ax_pano.imshow(
        np.zeros((120, 360), dtype=np.uint8),
        cmap=_RET_CMAP, vmin=0, vmax=255, aspect='auto',
        interpolation='nearest', extent=[-180, 180, -60, 60],
    )
    ax_pano.set_xlim(-180, 180)
    ax_pano.set_ylim(-60, 60)
    ax_pano.axvline(-60, color='white', lw=1.0, ls='--', alpha=0.75)
    ax_pano.axvline( 60, color='white', lw=1.0, ls='--', alpha=0.75)
    ax_pano.axhline(0, color='0.3', lw=0.4, ls='--')
    ax_pano.tick_params(colors='0.4', labelsize=8)
    ax_pano.set_xlabel('Azimuth (deg)', color='0.5', fontsize=9)
    ax_pano.set_ylabel('El (deg)', color='0.5', fontsize=8)
    for sp in ax_pano.spines.values():
        sp.set_color('0.3')
    ax_pano.axis('on')

    _tr_colors  = ['#4a9eff', '#4aff88', '#ffaa44', '#ff4aaa', '#bb88ff']
    _tr_labels  = ['Pitch', 'Roll', 'Yaw', 'theta', 'phi']
    _tr_signals = [pitch_arr, roll_arr, yaw_arr, theta_arr, phi_arr]
    for ax in tr_axes:
        ax.set_facecolor(_TRC_BG)
        ax.tick_params(colors='0.6', labelsize=8)
        for sp in ax.spines.values():
            sp.set_color('0.3')
    for ax, sig, col, lbl in zip(tr_axes, _tr_signals, _tr_colors, _tr_labels):
        ax.plot(times, sig, color=col, lw=1.0)
        ax.set_ylabel(lbl, color='0.6', fontsize=8, labelpad=2)
        fin = sig[np.isfinite(sig)]
        if len(fin):
            p1, p99 = np.nanpercentile(fin, [1, 99])
            mg = max(0.05 * abs(p99 - p1), 0.5)
            ax.set_ylim(p1 - mg, p99 + mg)
        ax.set_xlim(-_IMU_WIN_S / 2.0, _IMU_WIN_S / 2.0)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: str(int(x))))
    ax_phi_tr.set_xlabel('Time (s)', color='0.6', fontsize=8)
    for ax in (ax_pitch_tr, ax_roll_tr, ax_yaw_tr, ax_theta_tr):
        ax.tick_params(labelbottom=False)
    cursors_tr = [ax.axvline(0.0, color='w', lw=0.8, alpha=0.8) for ax in tr_axes]
    time_txt_d = fig.text(0.50, 0.004, '', color='0.6', fontsize=10,
                          ha='center', va='bottom')

    FIG_H = int(fig.get_figheight() * _DPI)
    FIG_W = int(fig.get_figwidth()  * _DPI)
    FIG_H += FIG_H % 2
    FIG_W += FIG_W % 2
    print(f'Output: {FIG_W}x{FIG_H} px  |  {n_total} frames @ {_VIDEO_FPS} fps')

    ffmpeg_bin, codec_args = _find_ffmpeg()
    ffmpeg_cmd = [
        ffmpeg_bin, '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{FIG_W}x{FIG_H}',
        '-pix_fmt', 'rgb24',
        '-r', str(_VIDEO_FPS),
        '-i', 'pipe:0',
        *codec_args,
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        out_video,
    ]
    print(f'Writing -> {out_video}  (codec: {codec_args[1]})')
    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    t0 = time.time()
    try:
        for i in range(n_total):
            t_rel = float(times[i])
            sid   = int(sweep_id[i])

            yr = np.radians(yaw_arr[i])
            _td_hdir.set_data(
                [mx, mx + arrow_td * np.cos(yr)],
                [my, my + arrow_td * np.sin(yr)],
            )
            gr = np.radians(yaw_arr[i] + anatomical_eye_angle_deg + theta_arr[i])
            _td_gdir.set_data(
                [mx, mx + arrow_td * 0.7 * np.cos(gr)],
                [my, my + arrow_td * 0.7 * np.sin(gr)],
            )
            _td_lbl.set_text(f'{sweep_names[sid]}: {sweep_arrs[sid][i]:.1f} deg')

            im_eye_d.set_data(_render_eye_schematic(theta_arr[i], phi_arr[i]))
            im_ret_d.set_data(retinal_images[i])
            im_pan_d.set_data(panoramic_images[i])

            for ax, cur in zip(tr_axes, cursors_tr):
                ax.set_xlim(t_rel - _IMU_WIN_S / 2.0, t_rel + _IMU_WIN_S / 2.0)
                cur.set_xdata([t_rel, t_rel])
            time_txt_d.set_text(f't = {t_rel:.2f} s')

            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(FIG_H, FIG_W, 4)
            ffmpeg_proc.stdin.write(arr[:, :, :3].tobytes())

            if (i + 1) % 300 == 0:
                elapsed = time.time() - t0
                print(f'  {i+1}/{n_total}  ({(i+1)/elapsed:.1f} fps)')

    except BrokenPipeError:
        print('BrokenPipeError -- ffmpeg stderr:')
        print(ffmpeg_proc.stderr.read().decode(errors='replace'))
        raise
    finally:
        ffmpeg_proc.stdin.close()
        retcode = ffmpeg_proc.wait()
    elapsed = time.time() - t0
    if retcode != 0:
        print(f'WARNING: ffmpeg exited {retcode}')
        print(ffmpeg_proc.stderr.read().decode(errors='replace')[-2000:])
    print(f'Video done in {elapsed:.1f}s  ({n_total/elapsed:.1f} fps)  -> {out_video}')
    plt.close(fig)

    _make_dummy_sweeps_pdf(
        retinal_images,
        sweep_names, sweep_slices, sweep_arrs,
        mx, my, dx, dy,
        pillar_x_mm, pillar_y_mm,
        arena_w_mm, arena_h_mm, pillar_d,
        anatomical_eye_angle_deg, yaw_for_gaze,
        out_pdf,
    )


def _make_dummy_sweeps_pdf(
    retinal_images,
    sweep_names, sweep_slices, sweep_arrs,
    mx, my, dx, dy,
    pillar_x_mm, pillar_y_mm,
    arena_w_mm, arena_h_mm, pillar_d,
    anatomical_eye_angle_deg, yaw_for_gaze,
    out_pdf,
):
    
    print(f'Building PDF -> {out_pdf}')
    N_COLS = 15
    N_ROWS = len(sweep_names)

    with PdfPages(out_pdf) as pdf:

        fig1, axes1 = plt.subplots(
            N_ROWS, N_COLS,
            figsize=(N_COLS * 1.15, N_ROWS * 1.6),
            facecolor='k',
        )
        fig1.subplots_adjust(
            hspace=0.55, wspace=0.04,
            left=0.05, right=0.995, top=0.93, bottom=0.03,
        )
        for ri, (sname, sslice) in enumerate(zip(sweep_names, sweep_slices)):
            n_sweep = sslice.stop - sslice.start
            col_idx = (np.round(np.linspace(0, n_sweep - 1, N_COLS))
                       .astype(int) + sslice.start)
            for ci, gi in enumerate(col_idx):
                ax = axes1[ri, ci]
                ax.set_facecolor('k')
                ax.imshow(
                    retinal_images[gi],
                    cmap=_RET_CMAP, vmin=0, vmax=255,
                    aspect='equal', interpolation='nearest',
                    extent=[-60, 60, -60, 60],
                )
                ax.set_xlim(-60, 60)
                ax.set_ylim(-60, 60)
                ax.set_title(f'{sweep_arrs[ri][gi]:.0f} deg',
                              color='0.75', fontsize=7, pad=2)
                ax.axis('off')
            axes1[ri, 0].set_ylabel(sname, color='0.85', fontsize=9, labelpad=3)
            axes1[ri, 0].axis('on')
            axes1[ri, 0].tick_params(left=False, bottom=False,
                                       labelleft=False, labelbottom=False)
            for sp in axes1[ri, 0].spines.values():
                sp.set_visible(False)
        fig1.suptitle(
            'Retinal projections -- variable sweep  (15 evenly-spaced frames per row)',
            color='0.85', fontsize=11,
        )
        pdf.savefig(fig1, facecolor='k')
        plt.close(fig1)

        fig2, ax2 = plt.subplots(1, 1, figsize=(7, 7), facecolor='#0a0a0a')
        ax2.set_facecolor('#0a0a0a')
        ax2.set_aspect('equal')
        ax2.axis('off')

        from matplotlib.patches import Rectangle as _SR, Circle as _SC
        pad2 = max(arena_w_mm, arena_h_mm) * 0.06
        ax2.set_xlim(-arena_w_mm / 2 - pad2,  arena_w_mm / 2 + pad2)
        ax2.set_ylim(-arena_h_mm / 2 - pad2,  arena_h_mm / 2 + pad2)
        ax2.add_patch(_SR(
            (-arena_w_mm / 2, -arena_h_mm / 2), arena_w_mm, arena_h_mm,
            fill=False, edgecolor='0.5', lw=1.5,
        ))
        ax2.add_patch(_SC(
            (pillar_x_mm, pillar_y_mm), pillar_d / 2, color='red', zorder=10,
        ))
        ax2.text(pillar_x_mm, pillar_y_mm - pillar_d * 0.9,
                 'Pillar', color='red', fontsize=9, ha='center', va='top')

        arrow2 = min(arena_w_mm, arena_h_mm) * 0.10
        ax2.plot(mx, my, 'o', color='#ffff00', markersize=11,
                 markeredgewidth=1, markeredgecolor='k', zorder=20)
        ax2.text(mx + arrow2 * 0.25, my - arrow2 * 0.35,
                 'Mouse', color='#ffff00', fontsize=9)

        yr = np.radians(yaw_for_gaze)
        ax2.annotate(
            '', xy=(mx + arrow2 * np.cos(yr), my + arrow2 * np.sin(yr)),
            xytext=(mx, my),
            arrowprops=dict(arrowstyle='->', color='#ffff00', lw=2.5),
        )
        ax2.text(mx + arrow2 * 1.08 * np.cos(yr),
                 my + arrow2 * 1.08 * np.sin(yr),
                 'Head', color='#ffff00', fontsize=8, ha='center')

        gr = np.radians(yaw_for_gaze + anatomical_eye_angle_deg)
        ax2.annotate(
            '', xy=(mx + arrow2 * 0.72 * np.cos(gr),
                    my + arrow2 * 0.72 * np.sin(gr)),
            xytext=(mx, my),
            arrowprops=dict(arrowstyle='->', color='#00ff55', lw=2.5),
        )
        ax2.text(mx + arrow2 * 0.78 * np.cos(gr),
                 my + arrow2 * 0.78 * np.sin(gr),
                 'Gaze', color='#00ff55', fontsize=8, ha='center')

        ax2.plot([mx, pillar_x_mm], [my, pillar_y_mm],
                 '--', color='0.3', lw=1.0, zorder=1)

        dist_mm = np.sqrt(dx ** 2 + dy ** 2)
        info = (
            f'Mouse:  ({mx:.0f}, {my:.0f}) mm\n'
            f'Pillar: ({pillar_x_mm:.0f}, {pillar_y_mm:.0f}) mm\n'
            f'Distance: {dist_mm:.0f} mm\n'
            f'yaw_for_gaze: {yaw_for_gaze:.1f} deg\n'
            f'anat. eye angle: {anatomical_eye_angle_deg:.0f} deg'
        )
        fig2.text(0.02, 0.02, info, color='0.55', fontsize=8,
                  va='bottom', family='monospace')
        fig2.suptitle(
            'Geometry schematic -- head/gaze orientation when eye faces pillar',
            color='0.8', fontsize=11,
        )
        pdf.savefig(fig2, facecolor='#0a0a0a')
        plt.close(fig2)

    print(f'Saved -> {out_pdf}')


def render_pillar_worldcam(
    pillar_x, pillar_y, mouse_x, mouse_y, mouse_yaw, mouse_pitch, mouse_roll,
    pillar_d=40.0, pillar_h=210.0, head_z=25.0,
    socket_az_deg=WORLDCAM_SOCKET_AZ_DEG,
    socket_el_deg=WORLDCAM_SOCKET_EL_DEG,
    f_px=WORLDCAM_F_PX, cx=None, cy=None,
    out_w=WORLDCAM_W, out_h=WORLDCAM_H,
):
    """Rasterise the pillar into world-camera pixel space.

    This is the counterpart to simulate_retinal_projection for validation: the
    retinal image models the animal's ~120 deg visual field, whereas the world
    camera sees only ~24 deg (it is eye-camera optics pointed outward).  The two
    must not be compared as if they shared a field of view -- the camera's
    footprint is 1.4 % of the retinal image's area.  Comparing here, at the
    camera's own resolution, keeps all 640x480 annotated pixels in play instead
    of squashing them into the ~16x12 retinal pixels the footprint covers.

    Parameters
    ----------
    mouse_yaw : float
        Renderer-convention yaw (CCW-positive, i.e. -yaw_imu), matching what
        get_retinal_image passes to simulate_retinal_projection.
    f_px, cx, cy : float
        World-camera pinhole intrinsics in output pixels.  cx/cy default to the
        image centre.  f_px is scaled automatically if out_w differs from
        WORLDCAM_W, so a downsampled render stays geometrically consistent.

    Returns
    -------
    mask : np.ndarray, shape (out_h, out_w), bool
    """

    scale = out_w / float(WORLDCAM_W)
    f = f_px * scale
    if cx is None:
        cx = out_w / 2.0
    if cy is None:
        cy = out_h / 2.0

    corners_opt = _pillar_optical_corners(
        pillar_x, pillar_y, mouse_x, mouse_y, mouse_yaw, mouse_pitch, mouse_roll,
        0.0, 0.0, pillar_h=pillar_h, pillar_d=pillar_d,
        eye_offset_x=0.0, eye_offset_y=0.0, eye_offset_z=0.0,
        anatomical_eye_angle_deg=socket_az_deg,
        socket_elev_deg=socket_el_deg,
        head_z=head_z,
    )

    mask = np.zeros((out_h, out_w), dtype=bool)
    clipped = _clip_corners_to_near(corners_opt)
    if not clipped:
        return mask

    pts = []
    for pt in clipped:
        u = np.clip(f * pt[0] / pt[2] + cx, -1e4, 1e4)
        v = np.clip(f * pt[1] / pt[2] + cy, -1e4, 1e4)
        pts.append([u, v])
    hull = cv2.convexHull(np.array(pts, dtype=np.int32))
    img = np.zeros((out_h, out_w), dtype=np.uint8)
    cv2.fillPoly(img, [hull], 1)
    return img.astype(bool)


def worldcam_frustum_in_retinal_px(
    socket_az_deg=WORLDCAM_SOCKET_AZ_DEG,
    socket_el_deg=WORLDCAM_SOCKET_EL_DEG,
    f_px=WORLDCAM_F_PX, res=120, fov_deg=120.0, fov_scale=1.0,
    retinal_socket_az_deg=None, retinal_socket_el_deg=None,
):
    """Corners of the world-camera footprint inside the retinal image.

    Returns (col0, row0, col1, row1) in retinal-image pixels, for drawing the
    camera's field of view as a box on the reconstruction.

    The box is placed at the camera axis *relative to the retinal image's own
    optical axis*.  In worldcam mode those are the same thing -- the render goes
    through the very socket given by socket_az/el_deg -- so the box is centred,
    and retinal_socket_* default to the camera's own values to express that.
    Placing it at the absolute mount angle instead applies the rotation twice
    and drops the box ~2.9 px low on this mount, which shows up as annotation
    contours sitting outside a box that is supposed to be the whole camera
    frame.  For the eye-camera path, pass the eye's socket so the offset is the
    genuine difference between the two axes.
    """
    if retinal_socket_az_deg is None:
        retinal_socket_az_deg = socket_az_deg
    if retinal_socket_el_deg is None:
        retinal_socket_el_deg = socket_el_deg
    socket_az_deg = socket_az_deg - retinal_socket_az_deg
    socket_el_deg = socket_el_deg - retinal_socket_el_deg

    f_ret = (res / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    c = res / 2.0
    # fov_scale > 1 widens the box about the camera axis -- fov_scale=2 spans
    # twice the camera's field in each dimension, which is the crop the display
    # panels use: wide enough to show where the pillar sits relative to the
    # camera window, without the 120 deg field shrinking that window to nothing.
    half_h = np.degrees(np.arctan((WORLDCAM_W / 2.0) / f_px)) * fov_scale
    half_v = np.degrees(np.arctan((WORLDCAM_H / 2.0) / f_px)) * fov_scale

    def _uv(az, el):
        az, el = np.radians(az), np.radians(el)
        Z = np.cos(el) * np.cos(az)
        X = np.cos(el) * np.sin(az)
        Y = -np.sin(el)
        return f_ret * X / Z + c, f_ret * Y / Z + c

    u0, v0 = _uv(socket_az_deg - half_h, socket_el_deg + half_v)
    u1, v1 = _uv(socket_az_deg + half_h, socket_el_deg - half_v)
    return float(u0), float(v0), float(u1), float(v1)


# How much wider than the camera's own field the reconstruction panels are
# cropped to.  Display only -- retinal_images itself is always the full 120 deg.
DISPLAY_FOV_SCALE = 2.0


def retinal_display_limits(socket_az_deg=WORLDCAM_SOCKET_AZ_DEG,
                           socket_el_deg=WORLDCAM_SOCKET_EL_DEG,
                           f_px=WORLDCAM_F_PX, res=120, fov_deg=120.0,
                           fov_scale=DISPLAY_FOV_SCALE,
                           retinal_socket_az_deg=None,
                           retinal_socket_el_deg=None):
    """(xlim, ylim) for a retinal panel drawn with extent [-60, 60, -60, 60].

    Returns the crop spanning `fov_scale` times the world camera's field in each
    dimension, centred on the camera axis, in the centred-pixel coordinates the
    panels use.
    """
    u0, v0, u1, v1 = worldcam_frustum_in_retinal_px(
        socket_az_deg=socket_az_deg, socket_el_deg=socket_el_deg,
        f_px=f_px, res=res, fov_deg=fov_deg, fov_scale=fov_scale,
        retinal_socket_az_deg=retinal_socket_az_deg,
        retinal_socket_el_deg=retinal_socket_el_deg)
    c = res / 2.0
    return (u0 - c, u1 - c), (c - v1, c - v0)


def _wc_pose_from_npz(npz):
    """Pull the pose the reconstruction was rendered from, plus a validity mask."""

    hx, hy = npz['head_x_eye'], npz['head_y_eye']
    yaw, pit, rol = npz['yaw_eye'], npz['pitch_eye'], npz['roll_eye']
    ok = np.isfinite(hx) & np.isfinite(hy) & np.isfinite(yaw)
    pit = np.where(np.isfinite(pit), pit, 0.0)
    rol = np.where(np.isfinite(rol), rol, 0.0)
    return hx, hy, yaw, pit, rol, ok


def _wc_scores(pred, gt):
    """Aggregate IoU, mean per-frame IoU, detection MCC, predicted-visible rate."""

    n = len(pred)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    agg = float(inter) / float(union) if union else 0.0

    pe = pred.reshape(n, -1).any(1)
    ge = gt.reshape(n, -1).any(1)
    both = np.where(pe | ge)[0]
    per = []
    for k in both:
        u = np.logical_or(pred[k], gt[k]).sum()
        per.append(float(np.logical_and(pred[k], gt[k]).sum()) / u if u else 0.0)

    tp = int((pe & ge).sum()); tn = int(((~pe) & (~ge)).sum())
    fp = int((pe & ~ge).sum()); fn = int(((~pe) & ge).sum())
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(
        agg_iou=agg,
        mean_iou=float(np.mean(per)) if per else 0.0,
        mcc=float((tp * tn - fp * fn) / den) if den else 0.0,
        pred_visible=float(pe.mean()), gt_visible=float(ge.mean()),
        false_pos=float((pe & ~ge).mean()), false_neg=float(((~pe) & ge).mean()),
        n_frames=int(n),
    )


def _wc_load_gt(gt_masks, start_ind, frame_idx, out_w, out_h):
    out = np.zeros((len(frame_idx), out_h, out_w), dtype=bool)
    for k, i in enumerate(frame_idx):
        g = np.asarray(gt_masks[start_ind + i])
        if g.ndim == 3:
            g = g[:, :, 0]
        out[k] = cv2.resize(g.astype(np.float32), (out_w, out_h),
                            interpolation=cv2.INTER_AREA) > 0.3
    return out


def evaluate_worldcam_reconstruction(
    gt_masks, npz, eyeT_startInd=None,
    socket_az_deg=WORLDCAM_SOCKET_AZ_DEG,
    socket_el_deg=WORLDCAM_SOCKET_EL_DEG,
    f_px=WORLDCAM_F_PX, pillar_d=None, pillar_h=None, head_z=25.0,
    downsample=8, max_frames=None, verbose=True,
):
    """Score the reconstruction against hand-annotated masks in worldcam space.

    Replaces estimate_retinal_to_eyecam_transform, which compared the retinal
    image's upper-right quadrant (az 0-60, el 0-60 deg) against the masks after
    an isotropic scale + translation.  That crop is ~18x the camera's true
    footprint area and centred in the wrong place, and a scale+shift in image
    space cannot represent the mount rotation.
    """

    if isinstance(npz, str):
        npz = np.load(npz, allow_pickle=False)
    if eyeT_startInd is None:
        eyeT_startInd = int(npz['eyeT_startInd'])
    if pillar_d is None:
        pillar_d = float(npz['pillar_d_mm'])
    if pillar_h is None:
        pillar_h = float(npz['pillar_h_mm'])

    px, py = float(npz['pillar_x_mm']), float(npz['pillar_y_mm'])
    hx, hy, yaw, pit, rol, ok = _wc_pose_from_npz(npz)
    idx = np.where(ok)[0]
    if max_frames is not None and len(idx) > max_frames:
        idx = idx[np.linspace(0, len(idx) - 1, max_frames).astype(int)]
    if len(idx) == 0:
        raise ValueError('no frames with a usable head pose')

    ow, oh = WORLDCAM_W // downsample, WORLDCAM_H // downsample
    gt = _wc_load_gt(gt_masks, eyeT_startInd, idx, ow, oh)

    pred = np.zeros_like(gt)
    for k, i in enumerate(idx):
        pred[k] = render_pillar_worldcam(
            px, py, float(hx[i]), float(hy[i]), -float(yaw[i]),
            float(pit[i]), float(rol[i]),
            pillar_d=pillar_d, pillar_h=pillar_h, head_z=head_z,
            socket_az_deg=socket_az_deg, socket_el_deg=socket_el_deg,
            f_px=f_px, out_w=ow, out_h=oh)

    m = _wc_scores(pred, gt)
    m.update(socket_az_deg=socket_az_deg, socket_el_deg=socket_el_deg, f_px=f_px,
             hfov_deg=2 * np.degrees(np.arctan((WORLDCAM_W / 2.0) / f_px)),
             vfov_deg=2 * np.degrees(np.arctan((WORLDCAM_H / 2.0) / f_px)),
             n_pose_valid=int(ok.sum()), n_total=int(len(hx)))
    if verbose:
        print(f'  Worldcam-space validation on {m["n_frames"]} frames '
              f'({m["n_pose_valid"]}/{m["n_total"]} have a usable head pose):')
        print(f'    aggregate IoU {m["agg_iou"]:.4f}   mean per-frame IoU '
              f'{m["mean_iou"]:.4f}   detection MCC {m["mcc"]:+.3f}')
        print(f'    predicted visible {m["pred_visible"]*100:.1f}%  vs GT '
              f'{m["gt_visible"]*100:.1f}%   (FP {m["false_pos"]*100:.1f}%, '
              f'FN {m["false_neg"]*100:.1f}%)')
    return m


def optical_dirs_to_angles(P):
    """Optical-frame directions -> (azimuth, elevation) in deg off the optic axis.

    P : (3, N) array of optical-frame points (X right, Y down, Z forward).
    """
    P = np.asarray(P, dtype=float)
    X, Y, Z = P[0], P[1], P[2]
    az = np.degrees(np.arctan2(X, Z))
    el = np.degrees(np.arctan2(-Y, np.hypot(X, Z)))
    return az, el


def worldcam_px_to_retinal_px(pts, f_px=WORLDCAM_F_PX, res=120, fov_deg=120.0,
                              out_w=WORLDCAM_W, out_h=WORLDCAM_H):
    """Map world-camera pixels onto the retinal image.

    Both renders share an optical frame (same mount), so the two image planes
    differ only in focal length: the map is an exact centred scale, verified to
    1e-11 px.  A ground-truth *polygon* therefore transfers onto the
    reconstruction with no resampling loss -- unlike a filled mask, which is
    clipped at the camera's ~24 deg instead of the retina's 120 deg.

    pts : (N, 2) array of (col, row) in world-camera pixels.
    """
    pts = np.asarray(pts, dtype=float)
    f_ret = (res / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    s = f_ret / (f_px * out_w / float(WORLDCAM_W))
    out = np.empty_like(pts)
    out[:, 0] = (pts[:, 0] - out_w / 2.0) * s + res / 2.0
    out[:, 1] = (pts[:, 1] - out_h / 2.0) * s + res / 2.0
    return out


def pillar_outline_retinal_px(
    pillar_x, pillar_y, mouse_x, mouse_y, mouse_yaw, mouse_pitch, mouse_roll,
    pillar_d=40.0, pillar_h=210.0, head_z=25.0,
    socket_az_deg=WORLDCAM_SOCKET_AZ_DEG,
    socket_el_deg=WORLDCAM_SOCKET_EL_DEG,
    res=120, fov_deg=120.0, centred=True,
):
    """Pillar silhouette as a polygon in retinal-image pixel coordinates.

    The rasterised retinal image is only 120 px across, so a panel cropped to
    twice the camera's field shows the pillar over ~29 px and it reads as a
    staircase.  The silhouette is a projected convex hull -- it is vector
    geometry before it is ever rasterised -- so display panels should draw this
    polygon instead of zooming into the bitmap.  Same numbers, no pixel edges.

    Returns (N, 2) in PLOTTING coordinates: x right-positive, y UP-positive,
    origin at the image centre when centred=True.  That matches an imshow drawn
    with extent [-60, 60, -60, 60] and the limits from retinal_display_limits(),
    both of which are y-up.  The optical frame's row index is y-down, so the row
    component is negated here -- getting that wrong flips the silhouette
    vertically against the video it is drawn over.
    """
    C = _pillar_optical_corners(
        pillar_x, pillar_y, mouse_x, mouse_y, mouse_yaw, mouse_pitch, mouse_roll,
        0.0, 0.0, pillar_h=pillar_h, pillar_d=pillar_d,
        eye_offset_x=0.0, eye_offset_y=0.0, eye_offset_z=0.0,
        anatomical_eye_angle_deg=socket_az_deg, socket_elev_deg=socket_el_deg,
        head_z=head_z)
    pts = _clip_corners_to_near(C)
    if not pts:
        return None
    f = (res / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    c = 0.0 if centred else res / 2.0
    uv = []
    for p in pts:
        uv.append([f * p[0] / p[2] + c, f * p[1] / p[2] + c])
    uv = np.array(uv, dtype=np.float32)
    if len(uv) < 3:
        return None
    hull = cv2.convexHull(uv).reshape(-1, 2).astype(float)
    if centred:
        hull[:, 1] = -hull[:, 1]                 # row (y-down) -> y-up
    else:
        hull[:, 1] = res - hull[:, 1]
    return np.vstack([hull, hull[:1]])


def worldcam_mask_outline_retinal_px(mask, f_px=WORLDCAM_F_PX,
                                     res=120, fov_deg=120.0, centred=True):
    """Ground-truth mask contour mapped into retinal-image pixel coordinates.

    Exact: the two image planes share an optical frame and differ only in focal
    length, so this is a centred scale (see worldcam_px_to_retinal_px).  Drawing
    the annotation this way puts it on the reconstruction's own angular scale,
    rather than beside it at whatever size its own panel happens to be.
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if not m.any():
        return None
    cnts = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    out = worldcam_px_to_retinal_px(c, f_px=f_px, res=res, fov_deg=fov_deg,
                                    out_w=m.shape[1], out_h=m.shape[0])
    out = np.asarray(out, dtype=float).copy()
    # Same y-down -> y-up flip as pillar_outline_retinal_px; see its docstring.
    if centred:
        out[:, 0] -= res / 2.0
        out[:, 1] = res / 2.0 - out[:, 1]
    else:
        out[:, 1] = res - out[:, 1]
    return np.vstack([out, out[:1]])


def worldcam_px_to_angles(pts, f_px=WORLDCAM_F_PX,
                          out_w=WORLDCAM_W, out_h=WORLDCAM_H):
    """World-camera pixels -> (az, el) in deg off the optic axis."""
    pts = np.asarray(pts, dtype=float)
    f = f_px * out_w / float(WORLDCAM_W)
    X = pts[:, 0] - out_w / 2.0
    Y = pts[:, 1] - out_h / 2.0
    return optical_dirs_to_angles(np.stack([X, Y, np.full(len(pts), f)]))


def mask_outline_angles(mask, f_px=WORLDCAM_F_PX):
    """Largest contour of a world-camera mask, as (az, el) degrees.

    Returns (az, el) closed polylines, or (None, None) if the mask is empty.
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if not m.any():
        return None, None
    cnts = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    c = np.vstack([c, c[:1]])
    return worldcam_px_to_angles(c, f_px=f_px,
                                 out_w=m.shape[1], out_h=m.shape[0])


def pillar_outline_angles(
    pillar_x, pillar_y, mouse_x, mouse_y, mouse_yaw, mouse_pitch, mouse_roll,
    pillar_d=40.0, pillar_h=210.0, head_z=25.0,
    socket_az_deg=WORLDCAM_SOCKET_AZ_DEG,
    socket_el_deg=WORLDCAM_SOCKET_EL_DEG,
):
    """The modelled pillar's silhouette as (az, el) degrees, unclipped.

    Unclipped matters: the camera window holds a median of only 8 % of the
    reconstructed pillar, so anything that compares filled masks is comparing a
    sliver against the whole object.
    """
    C = _pillar_optical_corners(
        pillar_x, pillar_y, mouse_x, mouse_y, mouse_yaw, mouse_pitch, mouse_roll,
        0.0, 0.0, pillar_h=pillar_h, pillar_d=pillar_d,
        eye_offset_x=0.0, eye_offset_y=0.0, eye_offset_z=0.0,
        anatomical_eye_angle_deg=socket_az_deg, socket_elev_deg=socket_el_deg,
        head_z=head_z)
    pts = _clip_corners_to_near(C)
    if not pts:
        return None, None
    az, el = optical_dirs_to_angles(np.array(pts).T)
    pa = np.stack([az, el], 1).astype(np.float32)
    hull = cv2.convexHull(pa).reshape(-1, 2)
    hull = np.vstack([hull, hull[:1]])
    return hull[:, 0], hull[:, 1]


def angular_error_vs_gt(
    gt_masks, npz, eyeT_startInd=None,
    socket_az_deg=WORLDCAM_SOCKET_AZ_DEG,
    socket_el_deg=WORLDCAM_SOCKET_EL_DEG,
    f_px=WORLDCAM_F_PX, pillar_d=None, pillar_h=None, head_z=25.0,
    downsample=2, verbose=True,
):
    """Per-frame angular offset between predicted and annotated pillar, in degrees.

    This is the scale-free residual.  IoU in the camera frame is not: a 6 deg
    error is a quarter of the camera's 24 deg field but only 5 % of the modelled
    120 deg retina, so the same reconstruction looks four times worse when
    scored through the camera.  Angular error says the same thing at any zoom.

    Returns per-frame arrays (NaN where either mask is empty) plus summary stats.
    """

    if isinstance(npz, str):
        npz = np.load(npz, allow_pickle=False)
    if eyeT_startInd is None:
        eyeT_startInd = int(npz['eyeT_startInd'])
    if pillar_d is None:
        pillar_d = float(npz['pillar_d_mm'])
    if pillar_h is None:
        pillar_h = float(npz['pillar_h_mm'])

    px, py = float(npz['pillar_x_mm']), float(npz['pillar_y_mm'])
    hx, hy, yaw, pit, rol, ok = _wc_pose_from_npz(npz)
    n = len(hx)
    ow, oh = WORLDCAM_W // downsample, WORLDCAM_H // downsample

    d_az = np.full(n, np.nan)
    d_el = np.full(n, np.nan)
    for i in range(n):
        if not ok[i]:
            continue
        g = np.asarray(gt_masks[eyeT_startInd + i])
        if g.ndim == 3:
            g = g[:, :, 0]
        g = cv2.resize(g.astype(np.float32), (ow, oh),
                       interpolation=cv2.INTER_AREA) > 0.3
        if not g.any():
            continue
        p = render_pillar_worldcam(
            px, py, float(hx[i]), float(hy[i]), -float(yaw[i]),
            float(pit[i]), float(rol[i]), pillar_d=pillar_d, pillar_h=pillar_h,
            head_z=head_z, socket_az_deg=socket_az_deg,
            socket_el_deg=socket_el_deg, f_px=f_px, out_w=ow, out_h=oh)
        if not p.any():
            continue
        yg, xg = np.where(g)
        yp, xp = np.where(p)
        a_g, e_g = worldcam_px_to_angles(np.array([[xg.mean(), yg.mean()]]),
                                         f_px=f_px, out_w=ow, out_h=oh)
        a_p, e_p = worldcam_px_to_angles(np.array([[xp.mean(), yp.mean()]]),
                                         f_px=f_px, out_w=ow, out_h=oh)
        d_az[i] = float(a_p[0] - a_g[0])
        d_el[i] = float(e_p[0] - e_g[0])

    radial = np.hypot(d_az, d_el)
    fin = np.isfinite(radial)
    out = dict(d_az=d_az, d_el=d_el, radial=radial, n_scored=int(fin.sum()))
    if fin.any():
        out.update(
            median_az=float(np.nanmedian(d_az)),
            median_el=float(np.nanmedian(d_el)),
            median_radial=float(np.nanmedian(radial)),
            iqr_radial=(float(np.nanpercentile(radial, 25)),
                        float(np.nanpercentile(radial, 75))),
            frac_within_5=float(np.mean(radial[fin] < 5.0)),
            frac_within_10=float(np.mean(radial[fin] < 10.0)),
        )
        if verbose:
            print(f'  Angular error on {out["n_scored"]} co-visible frames:')
            print(f'    bias  az {out["median_az"]:+.2f} deg   '
                  f'el {out["median_el"]:+.2f} deg')
            print(f'    radial median {out["median_radial"]:.2f} deg  '
                  f'IQR [{out["iqr_radial"][0]:.2f}, {out["iqr_radial"][1]:.2f}]')
            print(f'    within 5 deg {out["frac_within_5"]*100:.0f}%   '
                  f'within 10 deg {out["frac_within_10"]*100:.0f}%')
    return out


def fit_worldcam_mount(
    gt_masks, npz, eyeT_startInd=None, pillar_d=40.0, pillar_h=210.0,
    head_z=25.0, downsample=8, max_frames=1200, random_seed=0, verbose=True,
):
    """Fit mount azimuth, elevation and focal length to the annotated masks.

    Pillar dimensions and head height are held at their measured values: freeing
    them lets the optimiser absorb pose noise by inflating the pillar (an
    unconstrained fit ran away to a 177 mm pillar and a principal point off the
    sensor).  Reports a held-out score on a random half of the frames.
    """

    from scipy.optimize import minimize

    if isinstance(npz, str):
        npz = np.load(npz, allow_pickle=False)
    if eyeT_startInd is None:
        eyeT_startInd = int(npz['eyeT_startInd'])

    px, py = float(npz['pillar_x_mm']), float(npz['pillar_y_mm'])
    hx, hy, yaw, pit, rol, ok = _wc_pose_from_npz(npz)
    idx = np.where(ok)[0]
    if len(idx) > max_frames:
        idx = idx[np.linspace(0, len(idx) - 1, max_frames).astype(int)]

    ow, oh = WORLDCAM_W // downsample, WORLDCAM_H // downsample
    gt = _wc_load_gt(gt_masks, eyeT_startInd, idx, ow, oh)

    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(len(idx))
    train, test = np.sort(perm[:len(idx) // 2]), np.sort(perm[len(idx) // 2:])

    def _pred(az, el, f, sub):
        out = np.zeros((len(sub), oh, ow), dtype=bool)
        for k, j in enumerate(sub):
            i = idx[j]
            out[k] = render_pillar_worldcam(
                px, py, float(hx[i]), float(hy[i]), -float(yaw[i]),
                float(pit[i]), float(rol[i]),
                pillar_d=pillar_d, pillar_h=pillar_h, head_z=head_z,
                socket_az_deg=az, socket_el_deg=el, f_px=f,
                out_w=ow, out_h=oh)
        return out

    def _neg(p, sub):
        az, el, f = p
        if not (-60 <= az <= 60 and -50 <= el <= 50 and 200 <= f <= 6000):
            return 0.0
        pr = _pred(az, el, f, sub)
        inter = np.logical_and(pr, gt[sub]).sum()
        union = np.logical_or(pr, gt[sub]).sum()
        return -(float(inter) / float(union)) if union else 0.0

    best = (-np.inf, None)
    for az in np.arange(-24, 25, 4.0):
        for el in np.arange(-28, 13, 4.0):
            for f in (600., 900., 1200., 1500., 1800., 2200.):
                s = -_neg([az, el, f], train)
                if s > best[0]:
                    best = (s, [az, el, f])
    if verbose:
        print(f'  grid: aggIoU(train)={best[0]:.4f}  az={best[1][0]:.0f} '
              f'el={best[1][1]:.0f} f={best[1][2]:.0f}')

    res = minimize(_neg, best[1], args=(train,), method='Nelder-Mead',
                   options=dict(maxiter=1200, xatol=1e-3, fatol=1e-6, adaptive=True))
    az, el, f = float(res.x[0]), float(res.x[1]), float(res.x[2])
    out = dict(socket_az_deg=az, socket_el_deg=el, f_px=f,
               hfov_deg=2 * np.degrees(np.arctan((WORLDCAM_W / 2.0) / f)),
               vfov_deg=2 * np.degrees(np.arctan((WORLDCAM_H / 2.0) / f)),
               agg_iou_train=-_neg(res.x, train),
               agg_iou_test=-_neg(res.x, test),
               n_frames=int(len(idx)))
    if verbose:
        print(f'  fitted: az={az:+.2f} deg  el={el:+.2f} deg  f={f:.0f} px  '
              f'-> HFOV {out["hfov_deg"]:.1f} deg / VFOV {out["vfov_deg"]:.1f} deg')
        print(f'          aggIoU train={out["agg_iou_train"]:.4f}  '
              f'held-out={out["agg_iou_test"]:.4f}')
    return out


def align_gt_masks(gt_masks, h5_path=None, npz_path=None):
 
    start_ind = end_ind = None

    if h5_path is not None:
        with h5py.File(h5_path, 'r') as f:
            if 'eyeT_startInd' in f:
                start_ind = int(f['eyeT_startInd'][()])
            if 'eyeT_endInd' in f:
                end_ind = int(f['eyeT_endInd'][()])

    if (start_ind is None or end_ind is None) and npz_path is not None:
        try:
            npz = np.load(npz_path, allow_pickle=False)
            if start_ind is None and 'eyeT_startInd' in npz:
                start_ind = int(npz['eyeT_startInd'])
            if end_ind is None and 'eyeT_endInd' in npz:
                end_ind = int(npz['eyeT_endInd'])
            if end_ind is None and start_ind is not None and 'retinal_images' in npz:
                end_ind = start_ind + int(npz['retinal_images'].shape[0])
        except Exception as exc:
            print(f'  align_gt_masks: could not read npz ({exc})')

    if start_ind is None:
        n_full  = len(gt_masks)
        n_recon = n_full
        if npz_path is not None:
            try:
                npz = np.load(npz_path, allow_pickle=False)
                if 'retinal_images' in npz:
                    n_recon = int(npz['retinal_images'].shape[0])
            except Exception:
                pass
        start_ind = n_full - n_recon
        end_ind   = n_full
        print(f'  align_gt_masks WARNING: eyeT_startInd not found; '
              f'inferred start_ind={start_ind} from frame-count difference '
              f'(may be inaccurate -- re-run get_retinal_image to fix)')

    end_ind    = min(end_ind, len(gt_masks))
    gt_aligned = gt_masks[start_ind:end_ind]
    print(f'  align_gt_masks: GT[{start_ind}:{end_ind}] -> {len(gt_aligned)} frames '
          f'(of {len(gt_masks)} total GT frames)')
    return gt_aligned, start_ind, end_ind


if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser(description='Retinal image utilities')

    # Dummy sweep mode (no real data)
    parser.add_argument('--dummy', action='store_true',
                        help='Run synthetic sweep diagnostic video + PDF (no real data required)')
    parser.add_argument('--out_dir', default='.',
                        help='Output directory for dummy video + PDF')
    parser.add_argument('--pillar_x_mm', type=float, default=0.0)
    parser.add_argument('--pillar_y_mm', type=float, default=0.0)
    parser.add_argument('--arena_w_mm',  type=float, default=600.0)
    parser.add_argument('--arena_h_mm',  type=float, default=600.0)
    parser.add_argument('--seed', type=int, default=42)

    # Synthetic PDF mode (no real data)
    parser.add_argument('--synthetic_pdf', metavar='OUT_PDF',
                        help='Generate a synthetic diagnostic PDF. Provide output path.')
    parser.add_argument('--synthetic_pages', type=int, default=10)
    parser.add_argument('--synthetic_arena_w_mm', type=float, default=560.0)
    parser.add_argument('--synthetic_arena_h_mm', type=float, default=560.0)
    parser.add_argument('--synthetic_seed', type=int, default=0)

    # Real data
    parser.add_argument('--h5_path', help='Path to the _preproc.h5 file')
    parser.add_argument('--out_npz', help='Path to save the output .npz file (default: same dir as input)')
    parser.add_argument('--arena_width_cm', type=float, default=None,
                        help='Override arena width in cm to recalculate pxls2cm')
    parser.add_argument('--fixed_eye', action='store_true',
                        help='Render from the head axis: socket azimuth/elevation zeroed, '
                             'no eye offset, no pupil movement. Same geometry as --worldcam, '
                             'so it isolates head-pose error from eye error.')
    parser.add_argument('--use_stored_yaw', action='store_true',
                        help='Use upsampled_yaw/igyro_corrected_deg from the h5 '
                             'as written by preprocessing, instead of re-fusing '
                             'gyro_z_trim with head_yaw_deg here (default is to '
                             're-fuse; see _fuse_head_yaw).')
    parser.add_argument('--yaw_sigma_s', type=float, default=YAW_FUSION_SIGMA_S,
                        help='Gaussian half-width (s) of the yaw drift '
                             'correction (default: %(default)s).')
    parser.add_argument('--fixed_pitch', action='store_true',
                        help='Hold pitch at resting mean')
    parser.add_argument('--fixed_roll', action='store_true',
                        help='Hold roll at resting mean')

    # Shared
    parser.add_argument('--pillar_d', type=float, default=40.0,
                        help='Pillar diameter in mm (default: 40 = 4 cm). '
                             'Overrides the annotation-derived value.')
    parser.add_argument('--pillar_h', type=float, default=210.0,
                        help='Pillar height in mm (default: 210 = 21 cm).')
    parser.add_argument('--anatomical_eye_angle_deg', type=float, default=-65.0)
    parser.add_argument('--worldcam', action='store_true',
                        help='Camera at head centre facing forward: no eye offset, '
                             'no anatomical socket rotation, no pupil movement')
    parser.add_argument('--worldcam_socket_deg', type=float,
                        default=WORLDCAM_SOCKET_AZ_DEG,
                        help='Azimuth of the worldcam optic axis in the head '
                             'frame, deg (default: %(default)s, fitted to the '
                             'annotated masks). 0 = straight ahead, + = toward '
                             "the animal's left.")
    parser.add_argument('--worldcam_elev_deg', type=float,
                        default=WORLDCAM_SOCKET_EL_DEG,
                        help='Elevation of the worldcam optic axis in the head '
                             'frame, deg (default: %(default)s, fitted). '
                             'Negative tilts the optic axis up, which moves '
                             'the scene down in the frame.')
    parser.add_argument('--worldcam_video_path', type=str, default=None,
                        help='Path to worldcam AVI (default: auto-detected as '
                             '<prefix>_eyecam_deinter.avi in the same directory as h5_path)')
    parser.add_argument('--panoramic_yaw_only', action='store_true',
                        help='Panoramic plot uses only yaw; pitch and roll are fixed at '
                             'their session means instead of per-frame values')

    args = parser.parse_args()

    if args.dummy:
        make_dummy_diagnostic(
            out_dir=args.out_dir,
            pillar_x_mm=args.pillar_x_mm,
            pillar_y_mm=args.pillar_y_mm,
            arena_w_mm=args.arena_w_mm,
            arena_h_mm=args.arena_h_mm,
            pillar_d=args.pillar_d,
            pillar_h=args.pillar_h,
            anatomical_eye_angle_deg=args.anatomical_eye_angle_deg,
            seed=args.seed,
        )
    elif args.synthetic_pdf:
        make_synthetic_retinal_diagnostic_pdf(
            out_pdf=args.synthetic_pdf,
            h5_path=args.h5_path,
            pillar_d=args.pillar_d,
            pillar_h=args.pillar_h,
            arena_w_mm=args.synthetic_arena_w_mm,
            arena_h_mm=args.synthetic_arena_h_mm,
            anatomical_eye_angle_deg=args.anatomical_eye_angle_deg,
            n_pages=args.synthetic_pages,
            seed=args.synthetic_seed,
        )
    elif args.h5_path:
        npz_path = args.out_npz
        _, npz_path = get_retinal_image(
            h5_path=args.h5_path,
            out_npz=npz_path,
            pillar_d=args.pillar_d,
            pillar_h=args.pillar_h,
            arena_width_cm=args.arena_width_cm,
            anatomical_eye_angle_deg=args.anatomical_eye_angle_deg,
            fixed_eye=args.fixed_eye,
            refit_yaw=not args.use_stored_yaw,
            yaw_sigma_s=args.yaw_sigma_s,
            fixed_pitch=args.fixed_pitch,
            fixed_roll=args.fixed_roll,
            worldcam=args.worldcam,
            worldcam_video_path=args.worldcam_video_path,
            worldcam_socket_deg=args.worldcam_socket_deg,
            worldcam_elev_deg=args.worldcam_elev_deg,
        )
        make_retinal_diagnostic_video(
            h5_path=args.h5_path,
            npz_path=npz_path,
            arena_width_cm=args.arena_width_cm,
            pillar_d=args.pillar_d,
            pillar_h=args.pillar_h,
            panoramic_yaw_only=args.panoramic_yaw_only,
        )
        make_retinal_diagnostic_pdf(
            h5_path=args.h5_path,
            npz_path=npz_path,
            arena_width_cm=args.arena_width_cm,
        )
    else:
        parser.print_help()


# python -m fm2p.get_retinal_image --h5_path /home/dylan/Fast1/ret2ego_reconstruction/251028_DMM_worldcam/fm4_251028_121027_776/251028_DMM_DMM000_fm_04_preproc.h5 --worldcam

# python -m fm2p.get_retinal_image --h5_path /home/dylan/Storage/freely_moving_data/V1PPC/cohort02_recordings/251016_DMM_DMM061_pos18/fm1/251016_DMM_DMM061_fm_01_preproc.h5

