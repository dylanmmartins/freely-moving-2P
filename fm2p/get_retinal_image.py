
"""
Transform 3D pillar from world coordinates to mouse 2D retinal projection.
linear units in mm, ang in deg.

DMM April 2026
"""

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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R


def simulate_retinal_projection(
    pillar_x, pillar_y, mouse_x, mouse_y, mouse_yaw, mouse_pitch, mouse_roll,
    pupil_tilt_h, pupil_tilt_v,
    pillar_h=210.0, pillar_d=40.0,
    eye_offset_x=3.5, eye_offset_y=-5.0, eye_offset_z=3.5
):

    
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
    pillar_x_mm = pillar_x_px * px2mm
    pillar_y_mm = pillar_y_px * px2mm

    fps_eye    = float(1.0 / np.nanmedian(np.diff(eyeT)))
    vor        = calc_vor_eye_offset(theta_trim, None, fps_eye, head_vel_deg_s=gyro_z_eye)
    ang_offset = float(vor['ang_offset_vor_null'])
    print(f'ang_offset (VOR null): {ang_offset:.2f} deg  |  VOR gain: {vor["vor_gain"]:.3f}')

    N = len(eyeT)

    yaw_eye = np.interp(eyeT, imuT, yaw_imu)

    twop_valid_x = np.isfinite(head_x_2p)
    twop_valid_y = np.isfinite(head_y_2p)

    head_x_eye = np.interp(eyeT,
                           twopT[twop_valid_x], head_x_2p[twop_valid_x]) * px2mm
    head_y_eye = np.interp(eyeT,
                           twopT[twop_valid_y], head_y_2p[twop_valid_y]) * px2mm

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
            eye_offset_x=eye_offset_x,
            eye_offset_y=eye_offset_y,
            eye_offset_z=eye_offset_z,
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
_ROLL_WIN   = 20         # frames to rolling-average for topdown brightness
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

    im_eye = ax_eye.imshow(
        np.zeros((_EYE_H, _EYE_W), dtype=np.uint8),
        cmap='gray', vmin=0, vmax=300,
        aspect='equal', interpolation='nearest',
    )
    ax_eye.set_xlim(_EYE_CX0 - 0.5, _EYE_CX1 - 0.5)
    ax_eye.set_ylim(_EYE_CY1 - 0.5, _EYE_CY0 - 0.5)

    im_ret = ax_retina.imshow(
        np.zeros((120, 120), dtype=np.uint8),
        cmap=_RET_CMAP, vmin=0, vmax=255,
        aspect='equal', interpolation='nearest',
    )
    ax_retina.set_xlim(-0.5, 119.5)
    ax_retina.set_ylim(119.5, -0.5)
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
    labels  = ['Pitch', 'Roll', 'Yaw', 'θ (eye H)', 'φ (eye V)']
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
    _RV['FIG_H']       = int(fig.get_figheight() * _DPI)
    _RV['FIG_W']       = int(fig.get_figwidth()  * _DPI)


def _rv_render_frame(out_idx: int) -> bytes:
    t_abs        = float(_RV['times'][out_idx])
    t_rel        = t_abs - float(_RV['t_start'])
    twop_abs_idx = int(_RV['twop_idx'][out_idx])

    _RV['im_eye'].set_data(_subtract_band(_RV['eye_frames'][out_idx]))

    top_bi = (twop_abs_idx - _RV['twop_lo'])
    top_bi = max(0, min(top_bi, len(_RV['top_frames']) - 1))
    _RV['im_td'].set_data(
        cv2.cvtColor(_RV['top_frames'][top_bi], cv2.COLOR_BGR2RGB))

    ret_idx = min(int(_RV['eye_trim_idx'][out_idx]), len(_RV['retinal_images']) - 1)
    _RV['im_ret'].set_data(_RV['retinal_images'][ret_idx])

    # scroll traces
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

        for cand in ('fm1_0001.mp4', 'fm01_0001.mp4'):
            p = os.path.join(rec_dir, cand)
            if os.path.exists(p):
                top_path = p
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


    print('Loading retinal images ...')
    npz            = np.load(npz_path)
    retinal_images = npz['retinal_images']      # (N_eye, 120, 120) uint8 at eye camera rate


    theta_trim = theta_raw[startInd: startInd + len(eyeT_trim)]
    phi_trim   = phi_raw[startInd:   startInd + len(eyeT_trim)]

    best_idx, best_score = -1, -1.0
    for i in range(1, len(light_onsets)):
        lo  = int(light_onsets[i])
        nxt = dark_onsets[dark_onsets > lo]
        if len(nxt) == 0:
            continue
        nd     = int(nxt[0])
        t_s    = twopT[lo]
        t_e    = twopT[nd]
        mask   = (eyeT_trim >= t_s) & (eyeT_trim <= t_e)
        n_eye  = int(mask.sum())
        if n_eye == 0:
            continue
        ep = 100.0 * int(np.sum(mask & np.isfinite(theta_trim) & np.isfinite(phi_trim))) / n_eye
        tp = 100.0 * (np.isfinite(head_x[lo:nd]) & np.isfinite(head_y[lo:nd])).sum() / max(nd - lo, 1)
        score = (ep + tp) / 2.0
        if score > best_score:
            best_score, best_idx = score, i

    if best_idx < 0:
        raise RuntimeError('No valid light block found.')

    lo      = int(light_onsets[best_idx])
    nd      = int(dark_onsets[dark_onsets > twopT[lo]][0])
    t_start = float(twopT[lo])
    t_end   = float(twopT[nd])

    nd      = min(nd, int(np.searchsorted(twopT, t_start + 80.0)))
    t_end   = float(twopT[nd])
    print(f'  Light block {best_idx}: 2P [{lo}:{nd}]  t=[{t_start:.1f}-{t_end:.1f}]s  '
          f'score={best_score:.1f}%')

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
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(eye_full_idx[0]))
    current = int(eye_full_idx[0])
    for i, target in enumerate(eye_full_idx.astype(int)):
        skip = target - current
        for _ in range(skip - 1):
            cap.read()
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
    top_frames = np.empty((n_top, _TOP_H, _TOP_W, 3), dtype=np.uint8)
    cap        = cv2.VideoCapture(top_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    for i in range(n_top):
        ret, frame = cap.read()
        if ret:
            top_frames[i] = cv2.resize(frame, (_TOP_W, _TOP_H))
    cap.release()
    print(f'  done in {time.time() - t0:.1f}s')

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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate retinal image estimates from preproc data.')
    parser.add_argument('--h5_path', help='Path to the _preproc.h5 file')
    parser.add_argument('--out_npz', help='Path to save the output .npz file (default: same dir as input)')

    args = parser.parse_args()

    npz_path = args.out_npz

    _, npz_path = get_retinal_image(
        h5_path=args.h5_path,
        out_npz=npz_path
    )

    make_retinal_diagnostic_video(
        h5_path=args.h5_path,
        npz_path=npz_path,
    )
