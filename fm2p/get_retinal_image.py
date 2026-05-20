
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

# Edges of the pillar bounding box (indices into the 8-corner array defined in
# simulate_retinal_projection / _pillar_optical_corners).
_BBOX_EDGES = [
    (0, 1), (1, 3), (3, 2), (2, 0),   # bottom face
    (4, 5), (5, 7), (7, 6), (6, 4),   # top face
    (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
]
_Z_NEAR = 0.1   # mm -- near clipping plane in optical space


def _clip_corners_to_near(corners_opt):
    """Return list of 3-vectors: in-front corners + near-plane intersection pts.

    corners_opt : (3, 8) array in optical frame.  Any corner with optical_z >
    _Z_NEAR is kept; edges that cross the near plane contribute an intersection
    point.  This ensures that when the viewer is inside the bounding box the
    projected hull covers the full visible extent instead of just the base.
    """
    pts = []
    for i in range(8):
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
):
    # worldcam: camera at head center.  No eye offset or pupil movement.
    # -90° socket rotates the camera to face the same direction as the panoramic
    # center (after the -90° az_offset convention): head -Y axis.  Using 0° here
    # and -90° az_offset on the panoramic gives the same panoramic visuals but
    # inconsistent retinal-image orientation; -90° socket makes both agree.
    if worldcam:
        eye_offset_x = eye_offset_y = eye_offset_z = 0.0
        anatomical_eye_angle_deg = -90.0
        pupil_tilt_h = pupil_tilt_v = 0.0

    if fixed_eye:
        pupil_tilt_h = resting_tilt_h
        pupil_tilt_v = resting_tilt_v
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
    # 8 corners of the bounding box of the cylinder
    corners_world = np.array([
        [pillar_x + r, pillar_y + r, 0, 1],
        [pillar_x + r, pillar_y - r, 0, 1],
        [pillar_x - r, pillar_y + r, 0, 1],
        [pillar_x - r, pillar_y - r, 0, 1],
        [pillar_x + r, pillar_y + r, pillar_h, 1],
        [pillar_x + r, pillar_y - r, pillar_h, 1],
        [pillar_x - r, pillar_y + r, pillar_h, 1],
        [pillar_x - r, pillar_y - r, pillar_h, 1]
    ]).T

    # yaw, Pitch, Roll)
    R_head = R.from_euler('zyx', [mouse_yaw, mouse_pitch, mouse_roll], degrees=True).as_matrix()
    t_head = np.array([[mouse_x], [mouse_y], [head_z]])
    
    M_head = np.eye(4)
    M_head[:3, :3] = R_head
    M_head[:3, 3:] = t_head
    
    # invert so world -> head
    M_head_inv = np.linalg.inv(M_head)
    corners_head = M_head_inv @ corners_world

    # Saccade rotation (dynamic, eye-in-socket)
    R_saccade = R.from_euler('zyx', [pupil_tilt_h, pupil_tilt_v, 0], degrees=True).as_matrix()
    # Anatomical socket rotation (fixed)
    R_socket = R.from_euler('z', anatomical_eye_angle_deg, degrees=True).as_matrix()
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
    pillar_d=40.0,
    pillar_h=210.0,
    ang_imoffset_override=None,
    anatomical_eye_angle_deg=-65.0,
    fixed_eye=False,
    fixed_pitch=False,
    fixed_roll=False,
    arena_width_cm=None,
    worldcam=False,
    worldcam_video_path=None,
):

    _root = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from fm2p.utils.ref_frame import calc_vor_eye_offset

    with h5py.File(h5_path, 'r') as f:
        eyeT        = f['eyeT_trim'][:]
        theta_trim  = f['theta_trim'][:]
        phi_trim    = f['phi_trim'][:]
        pitch_eye   = fm2p.nanmedfilt(f['pitch_eye_interp'][:], 5).squeeze()
        roll_eye    = fm2p.nanmedfilt(f['roll_eye_interp'][:], 5).squeeze()
        gyro_z_eye  = f['gyro_z_eye_interp'][:]

        imuT        = f['imuT_trim'][:]
        yaw_imu     = fm2p.nanmedfilt(f['upsampled_yaw']['igyro_corrected_deg'][:], 5).squeeze()

        twopT       = f['twopT'][:]
        head_x_2p   = f['head_x'][:]
        head_y_2p   = f['head_y'][:]

        pxls2cm        = float(f['pxls2cm'][()])
        eyeT_startInd  = int(f['eyeT_startInd'][()]) if 'eyeT_startInd' in f else 0
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

    pillar_x_mm =  pillar_x_px * px2mm
    pillar_y_mm = -pillar_y_px * px2mm

    if _pillar_rad_px is not None:
        pillar_d_mm = 2.0 * _pillar_rad_px * px2mm
        if pillar_d != pillar_d_mm:
            print(f'  pillar_d overridden by h5 pillar_radius: '
                  f'{pillar_d:.1f} -> {pillar_d_mm:.1f} mm')
        pillar_d = pillar_d_mm

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
        print(f'  fixed_eye=True: using resting tilt_h={mean_pfh:.2f}, tilt_v={mean_phi:.2f}')

    N = len(eyeT)

    yaw_eye = np.interp(eyeT, imuT, yaw_imu)

    twop_valid_x = np.isfinite(head_x_2p)
    twop_valid_y = np.isfinite(head_y_2p)

    head_x_eye =  np.interp(eyeT, twopT[twop_valid_x], head_x_2p[twop_valid_x]) * px2mm
    head_y_eye = -np.interp(eyeT, twopT[twop_valid_y], head_y_2p[twop_valid_y]) * px2mm  # negate: image y-down -> math y-up

    eye_in_range = (eyeT >= twopT[0]) & (eyeT <= twopT[-1])
    head_x_eye[~eye_in_range] = np.nan
    head_y_eye[~eye_in_range] = np.nan

    pfh_eye = ang_offset - theta_trim

    # Kalman-filter all behavioural inputs to suppress single-frame tracking
    # transients. NaN frames are prediction-only steps, so gaps are handled
    # gracefully. Tune R_frac (measurement noise fraction) to control the
    # smoothing strength: larger R_frac -> more smoothing, more lag.
    _kfps = fps_eye
    head_x_eye = _kalman_smooth(head_x_eye, _kfps)
    head_y_eye = _kalman_smooth(head_y_eye, _kfps)
    yaw_eye    = _kalman_smooth(yaw_eye,    _kfps)
    pitch_eye  = _kalman_smooth(pitch_eye,  _kfps)
    roll_eye   = _kalman_smooth(roll_eye,   _kfps)
    pfh_eye    = _kalman_smooth(pfh_eye,    _kfps)
    phi_trim   = _kalman_smooth(phi_trim,   _kfps)

    retinal_images = np.zeros((N, 120, 120), dtype=np.uint8)

    n = min(len(head_x_eye), len(yaw_eye), len(pitch_eye), len(roll_eye),
            len(pfh_eye), len(phi_trim), N)

    print(f'Rendering {n} retinal frames at {fps_eye:.1f} Hz ...')
    t0 = time.time()
    for i in range(n):
        mx = float(head_x_eye[i])
        my = float(head_y_eye[i])
        if not np.isfinite(mx) or not np.isfinite(my):
            continue

        yaw    = -float(yaw_eye[i])  if np.isfinite(yaw_eye[i])   else 0.0
        p      = float(pitch_eye[i]) if np.isfinite(pitch_eye[i]) else 0.0
        r_     = float(roll_eye[i])  if np.isfinite(roll_eye[i])  else 0.0
        tilt_h = float(pfh_eye[i])   if np.isfinite(pfh_eye[i])   else ang_offset
        tilt_v = float(phi_trim[i])  if np.isfinite(phi_trim[i])  else 0.0

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
        )

    print(f'  done in {time.time() - t0:.1f}s')

    # Load worldcam frames (head-mounted camera) when running in worldcam mode.
    # Frames are resized to 120×120 grayscale and saved alongside retinal_images
    # so downstream code does not need to re-read the video.
    worldcam_frames  = None
    worldcam_metrics = None
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
        anatomical_eye_angle_deg=np.array(anatomical_eye_angle_deg),
        head_x_eye=head_x_eye,
        head_y_eye=head_y_eye,
        yaw_eye=yaw_eye,
        pitch_eye=pitch_eye,
        roll_eye=roll_eye,
        pfh_eye=pfh_eye,
        phi_trim=phi_trim,
    )
    if worldcam_frames is not None:
        save_dict['worldcam_frames'] = worldcam_frames
    if worldcam_metrics is not None:
        for _k, _v in worldcam_metrics.items():
            save_dict[f'wc_{_k}'] = _v
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
        'anatomical_eye_angle_deg': anatomical_eye_angle_deg,
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
    """
    Constant-velocity Kalman filter for a 1-D scalar signal.

    State: [position, velocity]; measurement: position only.
    NaN frames receive a prediction-only step (no measurement update),
    so gaps are bridged by the velocity estimate without corrupting the
    filter state.

    Parameters
    ----------
    signal : (N,) float array
    fps    : sample rate in Hz
    Q_frac : process-noise variance as a fraction of signal variance.
             Governs how fast the true state can accelerate.
    R_frac : measurement-noise variance as a fraction of signal variance.
             R_frac / Q_frac ~= 10 (default) -> moderate smoothing with
             ~3-frame lag; raise R_frac to increase smoothing.
    """
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


def _compute_worldcam_metrics(worldcam_frames, retinal_images, tau_bg=30, bg_threshold=15):
    """
    Three frame-wise alignment metrics between worldcam (uint8) and binary retinal images.

    All output arrays are length N (or N, 2) aligned to the common eyeT axis.
    The last frame of temporal/flow metrics is NaN (no t+1 frame).

    Metric 1 — temporal contrast score (tc_score)
        Ratio of mean |Δframe| inside the retinal mask to mean |Δframe| outside.
        Values >> 1 mean the mask covers the high-motion (moving pillar) region.

    Metric 2 — background-subtraction IoU (bg_iou, bg_centroid_err)
        Causal EMA background (time-constant tau_bg frames) is subtracted;
        residual > bg_threshold is treated as foreground. IoU between foreground
        and retinal mask.  bg_centroid_err[t] = [fg_col - mask_col, fg_row - mask_row]
        in pixels.

    Metric 3 — optical flow displacement error (flow_pred_disp, flow_obs, flow_error)
        flow_pred_disp[t] = centroid shift of retinal mask from t->t+1 (col, row).
        flow_obs[t]       = mean Farneback flow inside the mask at frame t.
        flow_error[t]     = L2(flow_pred_disp[t] - flow_obs[t]).
    """
    N = len(worldcam_frames)
    eps = 1e-6

    # ---- Metric 1: temporal contrast score ----
    tc_score = np.full(N, np.nan)
    for t in range(N - 1):
        diff = np.abs(worldcam_frames[t + 1].astype(np.float32) -
                      worldcam_frames[t].astype(np.float32))
        mask  = retinal_images[t] > 0
        n_in  = int(mask.sum())
        n_out = int((~mask).sum())
        if n_in > 0 and n_out > 0:
            tc_score[t] = float(diff[mask].mean()) / (float(diff[~mask].mean()) + eps)

    # ---- Metric 2: background subtraction -> IoU ----
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

        # Update background *after* computing residual so it stays causal
        bg = (1.0 - alpha) * bg + alpha * worldcam_frames[t].astype(np.float32)

    # ---- Metric 3: optical flow vs. predicted displacement ----
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
):
    
    r = pillar_d / 2.0
    head_z = 25.0
    corners_world = np.array([
        [pillar_x + r, pillar_y + r, 0,        1],
        [pillar_x + r, pillar_y - r, 0,        1],
        [pillar_x - r, pillar_y + r, 0,        1],
        [pillar_x - r, pillar_y - r, 0,        1],
        [pillar_x + r, pillar_y + r, pillar_h, 1],
        [pillar_x + r, pillar_y - r, pillar_h, 1],
        [pillar_x - r, pillar_y + r, pillar_h, 1],
        [pillar_x - r, pillar_y - r, pillar_h, 1],
    ]).T
    R_head = R.from_euler('zyx', [mouse_yaw, mouse_pitch, mouse_roll], degrees=True).as_matrix()
    t_head = np.array([[mouse_x], [mouse_y], [head_z]])
    M_head = np.eye(4)
    M_head[:3, :3] = R_head
    M_head[:3, 3:] = t_head
    corners_head = np.linalg.inv(M_head) @ corners_world

    R_saccade = R.from_euler('zyx', [pupil_tilt_h, pupil_tilt_v, 0], degrees=True).as_matrix()
    R_socket  = R.from_euler('z', anatomical_eye_angle_deg, degrees=True).as_matrix()
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
    """Render azimuth/elevation silhouette of the pillar bounding box.

    Uses the bounding-box centroid for azimuth to avoid wrap-around blowup
    when corners straddle the ±180° line (e.g. pillar nearly behind the eye).
    Elevation uses min/max of all 8 corners (no wrap issue for elevation).

    az_offset : degrees added to the centroid azimuth before rasterising.
        Use -90.0 for worldcam recordings to align the camera-forward axis
        with the panoramic display's zero-azimuth convention.
    """
    img = np.zeros((pan_h, pan_w), dtype=np.uint8)

    # Elevation: min/max of all 8 corners
    el_list = []
    for i in range(8):
        pt = corners_opt[:, i]
        lateral = np.sqrt(pt[0] ** 2 + pt[2] ** 2)
        el = np.degrees(np.arctan2(-pt[1], lateral))
        el_list.append(el)
    el_arr = np.array(el_list)

    # Azimuth: centroid position + effective bounding-box radius.
    # This prevents the span from blowing up to ~360° when some corners wrap
    # across ±180° (happens when the pillar is nearly behind the eye or when
    # the mouse is inside the rectangular bounding box).
    center = corners_opt.mean(axis=1)
    az_c = np.degrees(np.arctan2(center[0], center[2])) + az_offset

    # Per-corner azimuth delta relative to centroid — perspective-correct width.
    # Unwrapping relative to the centroid handles the ±180° wrap case.
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
        np.zeros((60, 60), dtype=np.uint8),
        cmap=_RET_CMAP, vmin=0, vmax=255,
        aspect='equal', interpolation='nearest',
        extent=[0, 60, 0, 60],
    )
    ax_retina.set_xlim(0, 60)
    ax_retina.set_ylim(0, 60)
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
    yaw_b   = _nan_wrap(_interp_short_gaps(
        init_data['yaw'][ext_lo:blk_nd].copy().astype(float)))
    theta_b = _interp_short_gaps(init_data['theta_2p'][ext_lo:blk_nd].astype(float))
    phi_b   = _interp_short_gaps(init_data['phi_2p'][ext_lo:blk_nd].astype(float))

    colors  = ['#4a9eff', '#4aff88', '#ffaa44', '#ff4aaa', '#bb88ff']
    labels  = ['Pitch', 'Roll', 'Yaw', 'θ', 'φ']
    signals = [pitch_b, roll_b, yaw_b, theta_b, phi_b]

    for ax, sig, col, lbl in zip(trace_axes, signals, colors, labels):
        ax.plot(tt_rel, sig, color=col, lw=1.2)
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
    tidx = min(twop_abs_idx, len(hx) - 1)
    px, py, yaw_deg = float(hx[tidx]), float(hy[tidx]), float(hyaw[tidx])
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
            if _RV.get('worldcam', False):
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
    _RV['im_ret'].set_data(_RV['retinal_images'][ret_idx][:60, 60:])

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

        for cand in ('fm*_0001.mp4', 'fm*0001.mp4'):
            matches = glob.glob(os.path.join(rec_dir, cand))
            if matches:
                top_path = matches[0]
                break
        if top_path is None:
            raise FileNotFoundError(f'Cannot find topdown video in {rec_dir}')
    if out_path is None:
        out_path = os.path.join(rec_dir, f'{prefix}_retinal_diagnostic.mp4')


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
        pillar_x_px     = float(f['pillar_centroid']['x'][()])
        pillar_y_px     = float(f['pillar_centroid']['y'][()])
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
    # Worldcam alignment metrics (present only when get_retinal_image was run with --worldcam)
    _wc_eyeT = npz['eyeT']          if 'eyeT'          in npz else np.array([])
    _wc_flow = npz['wc_flow_error'] if 'wc_flow_error' in npz else np.array([])
    if len(_wc_flow) > 0:
        print(f'  worldcam flow error loaded: {len(_wc_flow)} samples')
    if _npz_worldcam:
        # Use -90° socket so the panoramic and retinal image share the same
        # camera-forward direction (head -Y axis).  No az_offset needed.
        anatomical_eye_angle_deg = -90.0
        resting_tilt_h           = 0.0
        eye_offset_x = eye_offset_y = eye_offset_z = 0.0
        _npz_mean_pfh = _npz_mean_phi = 0.0
        print('  worldcam NPZ: anatomical_eye_angle_deg=-90°, no az_offset')
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
        t_vid = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0   # timestamp BEFORE read
        if t_vid > t_end + 0.5:
            break
        ret, frame = cap.read()
        if not ret:
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

    pillar_x_top = pillar_x_px * top_scale_x
    pillar_y_top = pillar_y_px * top_scale_y
    print(f'  Pillar centroid: ({pillar_x_px:.1f}, {pillar_y_px:.1f}) px  ->  '
          f'scaled ({pillar_x_top:.1f}, {pillar_y_top:.1f})')

    init_data = {
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
                        help='Simulate eye fixed at resting position (no pupil movement)')
    parser.add_argument('--fixed_pitch', action='store_true',
                        help='Hold pitch at resting mean')
    parser.add_argument('--fixed_roll', action='store_true',
                        help='Hold roll at resting mean')

    # Shared
    parser.add_argument('--pillar_d', type=float, default=40.0)
    parser.add_argument('--pillar_h', type=float, default=2100.0)
    parser.add_argument('--anatomical_eye_angle_deg', type=float, default=-65.0)
    parser.add_argument('--worldcam', action='store_true',
                        help='Camera at head centre facing forward: no eye offset, '
                             'no anatomical socket rotation, no pupil movement')
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
            fixed_pitch=args.fixed_pitch,
            fixed_roll=args.fixed_roll,
            worldcam=args.worldcam,
            worldcam_video_path=args.worldcam_video_path,
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

# python -m fm2p.get_retinal_image --h5_path /home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings/cohort02_recordings/251020_DMM_DMM056_pos08/fm1/251020_DMM_DMM056_fm_01_preproc.h5