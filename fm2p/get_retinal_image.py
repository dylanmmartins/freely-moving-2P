
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


def simulate_retinal_projection(
    pillar_x, pillar_y, mouse_x, mouse_y, mouse_yaw, mouse_pitch, mouse_roll,
    pupil_tilt_h, pupil_tilt_v,
    pillar_h=210.0, pillar_d=40.0,
    eye_offset_x=3.5, eye_offset_y=-5.0, eye_offset_z=3.5,
    fixed_eye=False,
    fixed_pitch=False,
    fixed_roll=False,
):

    if fixed_eye:
        pupil_tilt_h = 0.0
        pupil_tilt_v = 0.0
    if fixed_pitch:
        mouse_pitch = 0.0
    if fixed_roll:
        mouse_roll = 0.0

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

    # head to eye transform... theta/yaw and phi/pitch
    R_eye = R.from_euler('zyx', [pupil_tilt_h, pupil_tilt_v, 0], degrees=True).as_matrix()
    t_eye = np.array([[eye_offset_x], [eye_offset_y], [eye_offset_z]])
    
    M_eye = np.eye(4)
    M_eye[:3, :3] = R_eye
    M_eye[:3, 3:] = t_eye
    
    # invert so head -> eye
    M_eye_inv = np.linalg.inv(M_eye)
    corners_eye = M_eye_inv @ corners_head

    # optical_X = +eye_Y so that rightward objects map to the right side of the image
    # (right eye convention: temporal field on right, nasal on left)
    R_align = np.array([
        [ 0, -1,  0],
        [ 0,  0, -1],
        [ 1,  0,  0]
    ])

    corners_optical = R_align @ corners_eye[:3, :]

    # now the retinal projection (thjis is going to 2D now)
    retina_image = np.zeros((res_h, res_w), dtype=np.uint8)
    points_2d = []

    for i in range(8):
        pt_3d = corners_optical[:, i]
        
        # is pt behind eye?
        if pt_3d[2] <= 0:
            continue 
            
        pt_proj = K @ pt_3d
        
        u = int(pt_proj[0] / pt_proj[2])
        v = int(pt_proj[1] / pt_proj[2])
        
        points_2d.append([u, v])

    if len(points_2d) > 0:
        points_2d = np.array(points_2d, dtype=np.int32)

        hull = cv2.convexHull(points_2d)

        cv2.fillPoly(retina_image, [hull], color=255)

    return retina_image


def get_retinal_image(
    h5_path,
    out_npz=None,
    eye_offset_x=3.5,
    eye_offset_y=-5.0,
    eye_offset_z=3.5,
    pillar_d=40.0,
    pillar_h=210.0,
    ang_imoffset_override=None,
    fixed_eye=False,
    fixed_pitch=False,
    fixed_roll=False,
):

    _root = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from fm2p.utils.ref_frame import calc_vor_eye_offset

    with h5py.File(h5_path, 'r') as f:
        eyeT        = f['eyeT_trim'][:]
        theta_trim  = f['theta_trim'][:]
        phi_trim    = f['phi_trim'][:]
        pitch_eye   = f['pitch_eye_interp'][:]
        roll_eye    = f['roll_eye_interp'][:]
        gyro_z_eye  = f['gyro_z_eye_interp'][:]

        imuT        = f['imuT_trim'][:]
        yaw_imu     = f['upsampled_yaw']['igyro_corrected_deg'][:]

        twopT       = f['twopT'][:]
        head_x_2p   = f['head_x'][:]
        head_y_2p   = f['head_y'][:]

        pxls2cm     = float(f['pxls2cm'][()])
        pillar_x_px = float(f['pillar_centroid']['x'][()])
        pillar_y_px = float(f['pillar_centroid']['y'][()])

    px2mm       = 10.0 / pxls2cm
    # y is negated to convert from image convention (y-down)
    # to the right-hand world frame expected by the rotation math (y-up).
    pillar_x_mm = pillar_x_px * px2mm
    pillar_y_mm = -pillar_y_px * px2mm

    valid_hx = head_x_2p[np.isfinite(head_x_2p)]
    valid_hy = head_y_2p[np.isfinite(head_y_2p)]
    print(f'  pxls2cm={pxls2cm:.3f}  px2mm={px2mm:.4f}')
    print(f'  Pillar centroid: ({pillar_x_px:.1f}, {pillar_y_px:.1f}) px  '
          f'->  ({pillar_x_mm:.1f}, {pillar_y_mm:.1f}) mm')
    print(f'  Head X range: [{valid_hx.min():.1f}, {valid_hx.max():.1f}] px  '
          f'->  [{valid_hx.min()*px2mm:.1f}, {valid_hx.max()*px2mm:.1f}] mm')
    print(f'  Head Y range: [{valid_hy.min():.1f}, {valid_hy.max():.1f}] px  '
          f'->  [{valid_hy.min()*px2mm:.1f}, {valid_hy.max()*px2mm:.1f}] mm')
    pillar_in_x = valid_hx.min() <= pillar_x_px <= valid_hx.max()
    pillar_in_y = valid_hy.min() <= pillar_y_px <= valid_hy.max()
    if not (pillar_in_x and pillar_in_y):
        print(f'  WARNING: pillar is OUTSIDE the head position range — '
              f'coordinate system mismatch? (in_x={pillar_in_x}, in_y={pillar_in_y})')

    fps_eye    = float(1.0 / np.nanmedian(np.diff(eyeT)))
    vor        = calc_vor_eye_offset(theta_trim, None, fps_eye, head_vel_deg_s=gyro_z_eye)
    ang_offset = float(vor['ang_offset_vor_null'])
    print(f'ang_offset (VOR null): {ang_offset:.2f} deg  |  VOR gain: {vor["vor_gain"]:.3f}')
    if fixed_eye:
        print('  fixed_eye=True: pupil input ignored, eye treated as stationary at (0,0)')

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

        yaw    = float(yaw_eye[i])   if np.isfinite(yaw_eye[i])   else 0.0
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
            fixed_eye=fixed_eye,
            fixed_pitch=fixed_pitch,
            fixed_roll=fixed_roll,
        )

    print(f'  done in {time.time() - t0:.1f}s')

    if out_npz is None:
        out_npz = h5_path.replace('_preproc.h5', '_retinal_images.npz')

    np.savez_compressed(
        out_npz,
        retinal_images=retinal_images,
        eyeT=eyeT,
        ang_offset=np.array(ang_offset),
        vor_gain=np.array(vor['vor_gain']),
        pillar_x_mm=np.array(pillar_x_mm),
        pillar_y_mm=np.array(pillar_y_mm),
    )
    print(f'Saved -> {out_npz}')

    return {
        'retinal_images': retinal_images,
        'eyeT':           eyeT,
        'ang_offset':     ang_offset,
        'vor_gain':       vor['vor_gain'],
        'pillar_x_mm':    pillar_x_mm,
        'pillar_y_mm':    pillar_y_mm,
    }, out_npz


_RET_CMAP   = LinearSegmentedColormap.from_list('retinal', ['#060e06', '#00ff55'])
_VIDEO_FPS  = 30
_STRIDE     = 1          # sub-sample eye-camera frames (60 fps -> 7.5 fps effective)
_IMU_WIN_S  = 20.0       # seconds of IMU/eye trace visible at once
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

    fig = plt.figure(figsize=_FIGSIZE, dpi=_DPI, facecolor=_FIG_BG)

    gs = GridSpec(
        3, 2, figure=fig,
        width_ratios=[1, 2.2],
        height_ratios=[1, 1, 1.2],
        hspace=0.08, wspace=0.08,
        left=0.10, right=0.97, top=0.97, bottom=0.05,
    )

    ax_td     = fig.add_subplot(gs[0, 0])
    ax_eye    = fig.add_subplot(gs[1, 0])
    ax_retina = fig.add_subplot(gs[0:3, 1])

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

    for ax in trace_axes:
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

    # Head position dot and direction arrow (updated per frame)
    head_dot, = ax_td.plot([], [], 'o', color='#ffff00', markersize=7,
                           markeredgewidth=1, markeredgecolor='k', zorder=20)
    head_dir, = ax_td.plot([], [], '-', color='#ffff00', lw=2, zorder=21)

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
    )
    ax_retina.set_xlim(-0.5, 119.5)
    ax_retina.set_ylim(-0.5, 119.5)
    ax_retina.set_title('Estimated retinal image', color='0.6', fontsize=11, pad=4)

    ax_retina.axhline(60, color='0.3', lw=0.6, ls='--')
    ax_retina.axvline(60, color='0.3', lw=0.6, ls='--')
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

    fps_2p       = len(twopT) / float(twopT[-1])
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

    time_txt = fig.text(0.50, 0.004, '', color='0.6', fontsize=10,
                        ha='center', va='bottom')

    _RV['fig']         = fig
    _RV['im_td']       = im_td
    _RV['im_eye']      = im_eye
    _RV['im_ret']      = im_ret
    _RV['trace_axes']  = trace_axes
    _RV['cursors']     = cursors
    _RV['time_txt']    = time_txt
    _RV['head_dot']    = head_dot
    _RV['head_dir']    = head_dir
    _RV['FIG_H']       = int(fig.get_figheight() * _DPI)
    _RV['FIG_W']       = int(fig.get_figwidth()  * _DPI)


def _rv_render_frame(out_idx: int) -> bytes:
    
    t_abs        = float(_RV['times'][out_idx])
    t_rel        = t_abs - float(_RV['t_start'])
    twop_abs_idx = int(_RV['twop_idx'][out_idx])

    _RV['im_eye'].set_data(_RV['eye_frames'][out_idx])

    top_bi = (twop_abs_idx - _RV['twop_lo'])
    top_bi = max(0, min(top_bi, len(_RV['top_frames']) - 1))
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
            # yaw_deg is in allocentric/world frame (math: y-up, x-right).
            # Topdown display has y-down, so negate the y-component.
            arrow_len = 20.0  # pixels in resized topdown frame
            yaw_rad = np.radians(yaw_deg)
            ex = dx_top + arrow_len * np.cos(yaw_rad)
            ey = dy_top - arrow_len * np.sin(yaw_rad)
            _RV['head_dir'].set_data([dx_top, ex], [dy_top, ey])
        else:
            _RV['head_dir'].set_data([], [])
    else:
        _RV['head_dot'].set_data([], [])
        _RV['head_dir'].set_data([], [])

    ret_idx = min(int(_RV['eye_trim_idx'][out_idx]), len(_RV['retinal_images']) - 1)
    _RV['im_ret'].set_data(_RV['retinal_images'][ret_idx])

    for ax, cur in zip(_RV['trace_axes'], _RV['cursors']):
        ax.set_xlim(t_rel - _IMU_WIN_S / 2.0, t_rel + _IMU_WIN_S / 2.0)
        cur.set_xdata([t_rel, t_rel])

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
        pitch        = f['pitch_twop_interp'][:]
        roll         = f['roll_twop_interp'][:]
        pxls2cm      = float(f['pxls2cm'][()])
        pillar_x_px  = float(f['pillar_centroid']['x'][()])
        pillar_y_px  = float(f['pillar_centroid']['y'][()])


    print('Loading retinal images ...')
    npz            = np.load(npz_path)
    retinal_images = np.flip(npz['retinal_images'], axis=1)  # (N_eye, 120, 120) uint8 at eye camera rate
                                    # flip rows of each frame so below-horizon objects (pillar on floor)
                                    # appear in the top half of the display image


    theta_trim = theta_raw[startInd: startInd + len(eyeT_trim)]
    phi_trim   = phi_raw[startInd:   startInd + len(eyeT_trim)]

    # Precompute per-2P-frame speed (cm/s) from head position
    px2cm = 1.0 / pxls2cm
    hx_f  = head_x.astype(float)
    hy_f  = head_y.astype(float)
    dt_tp = np.diff(twopT)
    dx_cm = np.diff(hx_f) * px2cm
    dy_cm = np.diff(hy_f) * px2cm
    speed_twop = np.full(len(twopT), np.nan)
    ok = np.isfinite(dx_cm) & np.isfinite(dy_cm) & (dt_tp > 0)
    speed_twop[:-1][ok] = np.sqrt((dx_cm[ok] / dt_tp[ok])**2 +
                                   (dy_cm[ok] / dt_tp[ok])**2)

    # Eye-tracking validity resampled to 2P timebase
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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eye_frames[i] = cv2.resize(gray, (_EYE_W, _EYE_H))
        current = target + 1
    cap.release()
    print(f'  done in {time.time() - t0:.1f}s')

    print('Pre-loading topdown frames ...')
    t0         = time.time()
    n_top      = nd - lo
    cap        = cv2.VideoCapture(top_path)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(lo):
        cap.grab()
        
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read topdown frame")
        
    orig_top_h, orig_top_w = first_frame.shape[:2]
    top_scale_x = _TOP_W / orig_top_w
    top_scale_y = _TOP_H / orig_top_h
    
    top_frames = np.empty((n_top, _TOP_H, _TOP_W, 3), dtype=np.uint8)
    top_frames[0] = cv2.resize(first_frame, (_TOP_W, _TOP_H))
            
    for i in range(1, n_top):
        ret, frame = cap.read()
        if ret:
            top_frames[i] = cv2.resize(frame, (_TOP_W, _TOP_H))
    cap.release()
    print(f'  done in {time.time() - t0:.1f}s  (original {int(orig_top_w)}x{int(orig_top_h)})')

    pillar_x_top = pillar_x_px * top_scale_x
    pillar_y_top = pillar_y_px * top_scale_y
    print(f'  Pillar centroid: ({pillar_x_px:.1f}, {pillar_y_px:.1f}) px  ->  '
          f'scaled ({pillar_x_top:.1f}, {pillar_y_top:.1f})')

    init_data = {
        'eye_frames':      eye_frames,
        'top_frames':      top_frames,
        'retinal_images':  retinal_images,   # indexed by eye_trim_idx
        'eye_trim_idx':    eye_trim_idx,
        'twop_idx':        twop_idx,
        'times':           times,
        'twopT':           twopT,
        'pitch':           pitch,
        'roll':            roll,
        'yaw':             head_yaw[:len(twopT)],
        'theta_2p':        theta_2p,
        'phi_2p':          phi_2p,
        'twop_lo':         lo,
        'twop_nd':         nd,
        't_start':         t_start,
        'pillar_x_top':    pillar_x_top,
        'pillar_y_top':    pillar_y_top,
        'head_x_arr':      head_x.astype(float),
        'head_y_arr':      head_y.astype(float),
        'head_yaw_arr':    head_yaw[:len(twopT)].astype(float),
        'top_scale_x':     top_scale_x,
        'top_scale_y':     top_scale_y,
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

    n_workers = max(1, min(cpu_count() - 1, 8))
    print(f'Rendering with {n_workers} worker(s) ...')
    t0, n_written = time.time(), 0

    try:
        with Pool(
            processes=n_workers,
            initializer=_rv_worker_init,
            initargs=(init_data,),
        ) as pool:
            for frame_bytes in pool.imap(_rv_render_frame, range(n_frames), chunksize=4):
                ffmpeg_proc.stdin.write(frame_bytes)
                n_written += 1
                if n_written % 150 == 0:
                    elapsed = time.time() - t0
                    print(f'  {n_written}/{n_frames}  ({n_written / elapsed:.1f} fps)')
    except BrokenPipeError:
        print('BrokenPipeError — ffmpeg stderr:')
        print(ffmpeg_proc.stderr.read().decode(errors='replace'))
        raise

    ffmpeg_proc.stdin.close()
    retcode = ffmpeg_proc.wait()
    elapsed = time.time() - t0
    if retcode != 0:
        print(f'WARNING: ffmpeg exited {retcode}:')
        print(ffmpeg_proc.stderr.read().decode(errors='replace')[-3000:])
    print(f'Done in {elapsed:.1f}s  ({n_frames / elapsed:.1f} fps)  ->  {out_path}')


def make_retinal_diagnostic_pdf(h5_path, npz_path, out_pdf=None):
    """
    Generate 7-figure diagnostic PDF alongside the retinal video.

    Figures
    -------
    1  Mouse trajectory + pillar location (mm and px)
    2  Eye angle distributions (theta, phi, pfh, scatter)
    3  VOR analysis (theta vs gyro_z, yaw+pfh trace)
    4  Pillar angle distribution in head-frame and gaze-frame
    5  Mean retinal image, per-pixel coverage fraction, temporal non-zero fraction
    6  Grid of sample retinal frames (non-zero frames only)
    7  Pillar centroid scatter + 2-D histogram in retinal image
    """
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
        pillar_x_px = float(f['pillar_centroid']['x'][()])
        pillar_y_px = float(f['pillar_centroid']['y'][()])

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
                             twopT[twop_valid_y], head_y_2p[twop_valid_y]) * px2mm  # match simulation y-up

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

        # ── Fig 1: Mouse trajectory & pillar ────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Fig 1: Mouse Trajectory & Pillar Position', fontsize=12)

        ax = axes[0]
        valid = np.isfinite(head_x_eye) & np.isfinite(head_y_eye)
        sc = ax.scatter(head_x_eye[valid], head_y_eye[valid],
                        c=eyeT_trim[:N][valid], cmap='viridis', s=1, alpha=0.4)
        ax.plot(pillar_x_mm, pillar_y_mm, 'r*', ms=14, label='Pillar', zorder=10)
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
        ax.plot(pillar_x_px, -pillar_y_px, 'r*', ms=14, label='Pillar', zorder=10)
        plt.colorbar(sc2, ax=ax, label='Time (s)')
        ax.set_xlabel('Head X (px, x-flipped)')
        ax.set_ylabel('Head Y (px, y-flipped to math convention)')
        ax.set_title('Head trajectory (pixels, 2-P rate)')
        ax.legend()
        ax.set_aspect('equal')

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ── Fig 2: Eye angle distributions ──────────────────────────────────
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
        ax.set_xlabel('theta — raw pupil horiz angle (deg)')
        ax.set_title('theta distribution')
        ax.legend(fontsize=8)

        ax = axes[0, 1]
        fin = phi_t[np.isfinite(phi_t)]
        ax.hist(fin, bins=80, color='coral', alpha=0.8)
        ax.axvline(np.nanmedian(phi_t), color='r', lw=2,
                   label=f'median={np.nanmedian(phi_t):.1f}°')
        ax.set_xlabel('phi — pupil elevation (deg)')
        ax.set_title('phi distribution')
        ax.legend(fontsize=8)

        ax = axes[1, 0]
        pfh_fin = pfh_eye[np.isfinite(pfh_eye)]
        ax.hist(pfh_fin, bins=80, color='mediumseagreen', alpha=0.8)
        ax.axvline(np.nanmedian(pfh_eye), color='r', lw=2,
                   label=f'median={np.nanmedian(pfh_eye):.1f}°')
        ax.axvline(0, color='k', lw=1, ls='--', label='zero')
        ax.set_xlabel('pfh = ang_offset − theta  (eye-in-head horiz, deg)')
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

        # ── Fig 3: VOR analysis ──────────────────────────────────────────────
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

        # ── Fig 4: Pillar angle in head-frame and gaze-frame ────────────────
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

        # ── Fig 5: Mean retinal image, pixel coverage, temporal coverage ────
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

        # ── Fig 6: Grid of sample retinal frames (non-zero only) ─────────────
        if len(nz_indices) > 0:
            n_show  = min(24, len(nz_indices))
            sel_idx = nz_indices[
                np.linspace(0, len(nz_indices) - 1, n_show).astype(int)
            ]
            ncols = 6
            nrows = (n_show + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.4 * nrows))
            fig.suptitle(
                f'Fig 6: Sample Retinal Frames — non-zero only  '
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

        # ── Fig 7: Pillar centroid position within retinal image ─────────────
        if len(nz_indices) > 0:
            cx_list, cy_list, t_list = [], [], []
            for idx in nz_indices:
                img = retinal_images[idx]
                ys, xs = np.where(img > 0)
                cx_list.append(xs.mean() - 60)   # + = rightward
                cy_list.append(-(ys.mean() - 60)) # + = upward (flip y)
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
            ax.set_title('Pillar centroid — scatter (colored by time)')
            ax.set_aspect('equal')

            ax = axes[1]
            ax.hist2d(cx_arr, cy_arr, bins=30, range=[[-60, 60], [-60, 60]], cmap='hot')
            ax.axhline(0, color='c', lw=0.5, ls='--')
            ax.axvline(0, color='c', lw=0.5, ls='--')
            ax.set_xlabel('Azimuth (deg, +right)')
            ax.set_ylabel('Elevation (deg, +up)')
            ax.set_title('Pillar centroid — 2-D histogram')
            ax.set_aspect('equal')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f'Saved -> {out_pdf}')
    return out_pdf


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate retinal image estimates from preproc data.')
    parser.add_argument('--h5_path', help='Path to the _preproc.h5 file')
    parser.add_argument('--out_npz', help='Path to save the output .npz file (default: same dir as input)')
    parser.add_argument('--pillar_d', type=float, default=40.0, help='Pillar diameter in mm (default: 40)')
    parser.add_argument('--pillar_h', type=float, default=210.0, help='Pillar height in mm (default: 210)')
    parser.add_argument('--fixed_eye', action='store_true',
                        help='Ignore pupil input; simulate eye fixed at (0,0) in head frame '
                             '(right eye position only, no socket movement)')
    parser.add_argument('--fixed_pitch', action='store_true',
                        help='Hold pitch at 0 — only yaw (and roll unless also fixed) affects the image')
    parser.add_argument('--fixed_roll', action='store_true',
                        help='Hold roll at 0 — only yaw (and pitch unless also fixed) affects the image')

    args = parser.parse_args()

    npz_path = args.out_npz

    _, npz_path = get_retinal_image(
        h5_path=args.h5_path,
        out_npz=npz_path,
        pillar_d=args.pillar_d,
        pillar_h=args.pillar_h,
        fixed_eye=args.fixed_eye,
        fixed_pitch=args.fixed_pitch,
        fixed_roll=args.fixed_roll,
    )

    make_retinal_diagnostic_video(
        h5_path=args.h5_path,
        npz_path=npz_path,
    )

    make_retinal_diagnostic_pdf(
        h5_path=args.h5_path,
        npz_path=npz_path,
    )



# python fm2p/get_retinal_image.py --h5_path /home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings/cohort02_recordings/251020_DMM_DMM056_pos08/fm1/251020_DMM_DMM056_fm_01_preproc.h5 --fixed_eye --fixed_pitch --fixed_roll