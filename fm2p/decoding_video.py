

if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

import argparse
import glob
import os
import subprocess
import time
from multiprocessing import Pool, cpu_count

import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


DEFAULT_REC_DIR = (
    '/home/dylan/Storage/freely_moving_data/_V1PPC/'
    'cohort02_recordings/cohort02_recordings/'
    '251020_DMM_DMM056_pos08/fm1'
)
DEFAULT_PREFIX = '251020_DMM_DMM056_fm_01'


TRACE_WIN_S   = 20.0     # scrolling trace window width (seconds)
TRACE_SMOOTH  = 0.8      # Gaussian smoothing sigma (2P frames) for trace display
FIGSIZE       = (18, 12)
DPI           = 80
FS            = 27       # base font size (all text)

GT_COLOR      = '#cccccc'   # light grey — visible on dark background
PRED_COLOR    = 'tab:red'
FIG_BG        = 'k'
TRACE_BG      = 'k'

EYE_W, EYE_H  = 640, 480

EYE_CROP_X1 = int(EYE_W * 2 / 3)    # 426
EYE_CROP_Y0 = int(EYE_H * 0.20)     # 96
EYE_CROP_Y1 = int(EYE_H * 0.70)     # 336



def load_data(h5_path: str) -> dict:
    data = {}
    with h5py.File(h5_path, 'r') as f:
        for key in ('eyeT_trim', 'twopT',
                    'light_onsets', 'dark_onsets',
                    'theta', 'phi', 'ellipse_phi',
                    'longaxis', 'shortaxis', 'X0', 'Y0'):
            if key in f:
                data[key] = f[key][:]
            else:
                raise KeyError(f'Required key {key!r} missing from {h5_path}')
        data['eyeT_startInd'] = int(f['eyeT_startInd'][()])

        # Prefer deconvolved spikes; fall back to dF/F
        for neural_key in ('norm_spikes', 'norm_dFF'):
            if neural_key in f:
                data['neural']     = f[neural_key][:]
                data['neural_key'] = neural_key
                break
        if 'neural' not in data:
            raise KeyError('Neither norm_spikes nor norm_dFF found.')

    print(f'  Neural: {data["neural_key"]}  shape={data["neural"].shape}')
    return data



def select_best_block(data: dict) -> dict:

    eyeT_trim    = data['eyeT_trim']
    twopT        = data['twopT']
    light_onsets = data['light_onsets'].astype(int)
    dark_onsets  = data['dark_onsets'].astype(int)
    startInd     = data['eyeT_startInd']
    n_trim       = len(eyeT_trim)

    theta_trim = data['theta'][startInd:startInd + n_trim].astype(float)
    phi_trim   = data['phi'  ][startInd:startInd + n_trim].astype(float)

    best_idx, best_pct = -1, -1.0
    for i in range(1, len(light_onsets)):
        lo  = light_onsets[i]
        nxt = dark_onsets[dark_onsets > lo]
        if len(nxt) == 0:
            continue
        nd   = int(nxt[0])
        mask = (eyeT_trim >= twopT[lo]) & (eyeT_trim <= twopT[nd])
        n_eye = int(mask.sum())
        if n_eye == 0:
            continue
        pct = 100.0 * int(np.sum(
            mask & np.isfinite(theta_trim) & np.isfinite(phi_trim))) / n_eye
        if pct > best_pct:
            best_pct, best_idx = pct, i

    if best_idx < 0:
        raise RuntimeError('No valid light block found.')

    lo = int(light_onsets[best_idx])
    nd = int(dark_onsets[dark_onsets > lo][0])
    n_cells = data['neural'].shape[0]
    print(f'  Best block: idx={best_idx}, 2P frames [{lo}:{nd}] '
          f'({nd-lo} frames), eye tracking={best_pct:.1f}%, '
          f'cells={n_cells}')
    return {'lo': lo, 'nd': nd,
            't_start': float(twopT[lo]), 't_end': float(twopT[nd])}


def decode(data: dict, lo: int, nd: int) -> dict:

    twopT     = data['twopT']
    eyeT_trim = data['eyeT_trim']
    startInd  = data['eyeT_startInd']
    n_trim    = len(eyeT_trim)

    def eye_arr(key):
        return data[key][startInd:startInd + n_trim].astype(float)

    theta_e = eye_arr('theta')
    phi_e   = eye_arr('phi')
    X0_e    = eye_arr('X0')
    Y0_e    = eye_arr('Y0')
    la_e    = eye_arr('longaxis')
    sa_e    = eye_arr('shortaxis')
    ephi_e  = eye_arr('ellipse_phi')

    def to_2p(sig):
        return np.interp(twopT, eyeT_trim, sig)

    theta_2p = to_2p(theta_e)
    phi_2p   = to_2p(phi_e)
    X0_2p    = to_2p(X0_e)
    Y0_2p    = to_2p(Y0_e)
    la_2p    = to_2p(la_e)
    sa_2p    = to_2p(sa_e)
    ephi_2p  = to_2p(ephi_e)

    bt    = theta_2p[lo:nd]
    bp    = phi_2p  [lo:nd]
    bX    = X0_2p   [lo:nd]
    bY    = Y0_2p   [lo:nd]
    bla   = la_2p   [lo:nd]
    bsa   = sa_2p   [lo:nd]
    bephi = ephi_2p [lo:nd]
    n_block = nd - lo

    neural_T = data['neural'][:, lo:nd].T.astype(float)

    valid = (  np.isfinite(bt)
             & np.isfinite(bp)
             & np.isfinite(bX)
             & np.isfinite(bY)
             & np.isfinite(neural_T).all(axis=1))
    print(f'  Valid frames for ridge regression: {valid.sum()}/{n_block}')

    X_fit = neural_T[valid]
    print(f'  Ridge fit: {X_fit.shape[0]} frames × {X_fit.shape[1]} cells …')
    ridge_theta = Ridge(alpha=1.0).fit(X_fit, bt[valid])
    ridge_phi   = Ridge(alpha=1.0).fit(X_fit, bp[valid])

    pred_theta = ridge_theta.predict(neural_T)
    pred_phi   = ridge_phi.predict(neural_T)

    r_theta = float(np.corrcoef(bt[valid], pred_theta[valid])[0, 1])
    r_phi   = float(np.corrcoef(bp[valid], pred_phi  [valid])[0, 1])
    print(f'  Decoded correlations: theta r={r_theta:.3f}, phi r={r_phi:.3f}')

    valid_e = (  np.isfinite(theta_e) & np.isfinite(phi_e)
               & np.isfinite(X0_e)   & np.isfinite(Y0_e))
    poly = PolynomialFeatures(degree=2, include_bias=True)
    feat_eye = poly.fit_transform(
        np.column_stack([theta_e[valid_e], phi_e[valid_e]]))
    reg_x0 = Ridge(alpha=1.0).fit(feat_eye, X0_e[valid_e])
    reg_y0 = Ridge(alpha=1.0).fit(feat_eye, Y0_e[valid_e])


    feat_pred = poly.transform(np.column_stack([pred_theta, pred_phi]))
    pred_X0   = reg_x0.predict(feat_pred)
    pred_Y0   = reg_y0.predict(feat_pred)

    r_X0 = float(np.corrcoef(bX[valid], pred_X0[valid])[0, 1])
    r_Y0 = float(np.corrcoef(bY[valid], pred_Y0[valid])[0, 1])
    print(f'  Position correlations: X0 r={r_X0:.3f},  Y0 r={r_Y0:.3f}')

    return dict(
        gt_theta=np.degrees(bt),         gt_phi=np.degrees(bp),
        gt_X0=bX,                        gt_Y0=bY,
        pred_theta=np.degrees(pred_theta), pred_phi=np.degrees(pred_phi),
        pred_X0=pred_X0,                 pred_Y0=pred_Y0,
        gt_longaxis=bla, gt_shortaxis=bsa, gt_ellipse_phi=bephi,
    )


def find_eye_video(rec_dir: str, prefix: str) -> str:

    exact = os.path.join(rec_dir, f'{prefix}_eyecam_deinter.avi')
    if os.path.isfile(exact):
        return exact
    hits = sorted(glob.glob(os.path.join(rec_dir, '*_deinter.avi')))
    if hits:
        print(f'  Eye video: {hits[-1]}')
        return hits[-1]
    raise FileNotFoundError(f'No *_deinter.avi found in {rec_dir}')


def eye_frame_indices(data: dict, lo: int, nd: int) -> np.ndarray:
    twopT     = data['twopT']
    eyeT_trim = data['eyeT_trim']
    startInd  = data['eyeT_startInd']

    times    = twopT[lo:nd]
    trim_idx = np.searchsorted(eyeT_trim, times).clip(0, len(eyeT_trim) - 1)
    prev_idx = (trim_idx - 1).clip(0)
    closer   = (np.abs(eyeT_trim[prev_idx] - times)
                < np.abs(eyeT_trim[trim_idx] - times))
    trim_idx = np.where(closer, prev_idx, trim_idx)
    return (trim_idx + startInd).astype(int)


def preload_eye_frames(cap_path: str, full_idx: np.ndarray) -> np.ndarray:

    n      = len(full_idx)
    frames = np.zeros((n, EYE_H, EYE_W), dtype=np.uint8)
    cap    = cv2.VideoCapture(cap_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(full_idx[0]))
    current = int(full_idx[0])
    for i, target in enumerate(full_idx):
        skip = int(target) - current
        for _ in range(max(0, skip - 1)):
            cap.read()
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames[i] = cv2.resize(gray, (EYE_W, EYE_H))
        current = int(target) + 1
    cap.release()
    return frames


_W: dict = {}


def interp_short_gaps(x: np.ndarray, max_gap: int = 5) -> np.ndarray:

    x = np.asarray(x, dtype=float)
    isnan = np.isnan(x)
    if not isnan.any():
        return x.copy()
    out = x.copy()
    n, i = len(x), 0
    while i < n:
        if isnan[i]:
            start = i
            while i < n and isnan[i]:
                i += 1
            end = i
            if (end - start) <= max_gap and start > 0 and end < n:
                out[start:end] = np.interp(
                    np.arange(start, end), [start - 1, end],
                    [out[start - 1], out[end]])
        else:
            i += 1
    return out


def _smooth_trace(s: np.ndarray) -> np.ndarray:

    out  = np.array(s, dtype=float)
    nans = ~np.isfinite(out)
    if nans.all():
        return out
    if nans.any():
        fill = np.nanmedian(out)
        out[nans] = fill
    if TRACE_SMOOTH > 0:
        out = gaussian_filter1d(out, TRACE_SMOOTH)
    out[nans] = np.nan
    return out


def worker_init(init_data: dict) -> None:
    global _W
    _W = init_data

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor=FIG_BG)

    gs = GridSpec(3, 2, figure=fig,
                  height_ratios=[1, 1, 1.4],
                  hspace=0.40, wspace=0.30,
                  left=0.10, right=0.97, top=0.97, bottom=0.07)

    ax_theta = fig.add_subplot(gs[0, 0])
    ax_phi   = fig.add_subplot(gs[0, 1])
    ax_X     = fig.add_subplot(gs[1, 0])
    ax_Y     = fig.add_subplot(gs[1, 1])
    ax_ell   = fig.add_subplot(gs[2, 0])
    ax_eye   = fig.add_subplot(gs[2, 1])

    def _style(ax):
        ax.set_facecolor(TRACE_BG)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('0.4')
        ax.spines['bottom'].set_color('0.4')
        ax.tick_params(colors='0.6', labelcolor='0.6', labelsize=FS, length=4)

    for ax in (ax_theta, ax_phi, ax_X, ax_Y):
        _style(ax)

    twopT  = init_data['twopT']
    lo     = init_data['lo']
    nd     = init_data['nd']
    t_zero = init_data['t_start']
    tt     = twopT[lo:nd] - t_zero

    trace_specs = [
        (init_data['gt_theta'], init_data['pred_theta'],
         r'$\theta$ (°)',  r'$\theta$',    r'$\hat{\theta}$', ax_theta),
        (init_data['gt_phi'],   init_data['pred_phi'],
         r'$\phi$ (°)',    r'$\phi$',      r'$\hat{\phi}$',   ax_phi),
        (init_data['gt_X0'],    init_data['pred_X0'],
         r'$X_0$ (px)',    r'$X_0$',       r'$\hat{X}_0$',    ax_X),
        (init_data['gt_Y0'],    init_data['pred_Y0'],
         r'$Y_0$ (px)',    r'$Y_0$',       r'$\hat{Y}_0$',    ax_Y),
    ]

    trace_axes    = []
    trace_cursors = []

    for gt, pred, ylabel, lbl_gt, lbl_pred, ax in trace_specs:
        gt_s   = _smooth_trace(interp_short_gaps(gt))
        pred_s = _smooth_trace(pred)

        ax.plot(tt, gt_s,   color=GT_COLOR,   lw=2.7, alpha=0.9,  label=lbl_gt)
        ax.plot(tt, pred_s, color=PRED_COLOR, lw=2.7, alpha=0.85, label=lbl_pred)

        ax.set_ylabel(ylabel, fontsize=FS, color='0.6')
        ax.set_xlabel('time (s)', fontsize=FS, color='0.6')
        ax.set_xlim(-TRACE_WIN_S / 2, TRACE_WIN_S / 2)

        finite = gt_s[np.isfinite(gt_s)]
        if len(finite) > 2:
            lo_y, hi_y = np.nanpercentile(finite, [1, 99])
            mg = max(0.05 * abs(hi_y - lo_y), 0.005)
            ax.set_ylim(lo_y - mg, hi_y + mg)

        cursor = ax.axvline(0, color='w', lw=3.0, ls='--', alpha=0.7)
        trace_axes.append(ax)
        trace_cursors.append(cursor)

    ax_ell.set_facecolor('k')
    ax_ell.set_aspect('equal')
    ax_ell.spines['top'].set_visible(False)
    ax_ell.spines['right'].set_visible(False)
    ax_ell.spines['left'].set_color('0.4')
    ax_ell.spines['bottom'].set_color('0.4')
    ax_ell.tick_params(colors='0.6', labelcolor='0.6', labelsize=FS, length=4)
    ax_ell.set_xlabel('X (px)', fontsize=FS, color='0.6')
    ax_ell.set_ylabel('Y (px)', fontsize=FS, color='0.6')

    gt_X = init_data['gt_X0']
    gt_Y = init_data['gt_Y0']
    la   = init_data['gt_longaxis']
    x_fin  = gt_X[np.isfinite(gt_X)]
    y_fin  = gt_Y[np.isfinite(gt_Y)]
    la_med = float(np.nanmedian(la[np.isfinite(la)])) if np.any(np.isfinite(la)) else 30.0
    if len(x_fin) > 0:
        x_ctr = float(np.nanmedian(x_fin))
        y_ctr = float(np.nanmedian(y_fin))
        half  = max(float(np.nanstd(x_fin)) * 4, float(np.nanstd(y_fin)) * 4,
                    la_med * 4, 40.0)
    else:
        x_ctr, y_ctr, half = EYE_W / 2, EYE_H / 2, 100.0
    ax_ell.set_xlim(x_ctr - half, x_ctr + half)
    ax_ell.set_ylim(y_ctr + half, y_ctr - half)

    from matplotlib.patches import Ellipse as _Ellipse
    ell_gt   = _Ellipse((x_ctr, y_ctr), la_med * 2, la_med, angle=0,
                        fill=False, edgecolor=GT_COLOR,   linewidth=2.7, zorder=3)
    ell_pred = _Ellipse((x_ctr, y_ctr), la_med * 2, la_med, angle=0,
                        fill=False, edgecolor=PRED_COLOR, linewidth=2.7, zorder=4)
    ax_ell.add_patch(ell_gt)
    ax_ell.add_patch(ell_pred)
    ell_gt.set_visible(False)
    ell_pred.set_visible(False)

    ax_eye.set_facecolor('k')
    ax_eye.axis('off')

    im_eye = ax_eye.imshow(
        np.zeros((EYE_H, EYE_W), dtype=np.uint8),
        cmap='gray', vmin=0, vmax=255,
        aspect='equal', interpolation='nearest',
    )
    ax_eye.set_xlim(-0.5, EYE_CROP_X1 - 0.5)
    ax_eye.set_ylim(EYE_CROP_Y1 - 0.5, EYE_CROP_Y0 - 0.5)


    time_txt = fig.text(0.50, 0.005, '', color='0.6', fontsize=FS,
                        ha='center', va='bottom')

    _W['fig']           = fig
    _W['trace_axes']    = trace_axes
    _W['trace_cursors'] = trace_cursors
    _W['ell_gt']        = ell_gt
    _W['ell_pred']      = ell_pred
    _W['im_eye']        = im_eye
    _W['time_txt']      = time_txt
    _W['FIG_H']         = int(fig.get_figheight() * DPI)
    _W['FIG_W']         = int(fig.get_figwidth()  * DPI)


def render_frame(fi: int) -> bytes:
    """Render eye-camera frame index fi and return raw RGB bytes.

    fi indexes into eye_frames / eye_times (60 Hz eye timebase).
    eye_to_2p[fi] gives the corresponding 2P block frame for ellipse data.
    """
    t_zero  = _W['t_start']
    t_rel   = float(_W['eye_times'][fi]) - t_zero
    half    = TRACE_WIN_S / 2.0
    twop_fi = int(_W['eye_to_2p'][fi])
    for ax, cur in zip(_W['trace_axes'], _W['trace_cursors']):
        ax.set_xlim(t_rel - half, t_rel + half)
        cur.set_xdata([t_rel, t_rel])

    x0_gt  = float(_W['gt_X0'][twop_fi])
    y0_gt  = float(_W['gt_Y0'][twop_fi])
    la     = float(_W['gt_longaxis'][twop_fi])
    sa     = float(_W['gt_shortaxis'][twop_fi])
    phi    = float(_W['gt_ellipse_phi'][twop_fi])
    x0_p   = float(_W['pred_X0'][twop_fi])
    y0_p   = float(_W['pred_Y0'][twop_fi])

    gt_ok = np.isfinite(x0_gt + y0_gt + la + sa + phi) and la > 0 and sa > 0
    if gt_ok:
        _W['ell_gt'].set_center((x0_gt, y0_gt))
        _W['ell_gt'].width  = 2.0 * la
        _W['ell_gt'].height = 2.0 * sa
        _W['ell_gt'].angle  = float(np.degrees(phi))
        _W['ell_gt'].set_visible(True)
    else:
        _W['ell_gt'].set_visible(False)

    pred_ok = gt_ok and np.isfinite(x0_p + y0_p)
    if pred_ok:
        _W['ell_pred'].set_center((x0_p, y0_p))
        _W['ell_pred'].width  = 2.0 * la
        _W['ell_pred'].height = 2.0 * sa
        _W['ell_pred'].angle  = float(np.degrees(phi))
        _W['ell_pred'].set_visible(True)
    else:
        _W['ell_pred'].set_visible(False)


    _W['im_eye'].set_data(_W['eye_frames'][fi])


    _W['time_txt'].set_text(f't = {t_rel:.2f} s')

    _W['fig'].canvas.draw()
    buf = _W['fig'].canvas.buffer_rgba()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(
        _W['FIG_H'], _W['FIG_W'], 4)
    return arr[:, :, :3].tobytes()



def _find_ffmpeg():
    for ff in ('/usr/bin/ffmpeg', 'ffmpeg'):
        try:
            out = subprocess.run([ff, '-encoders'],
                                 capture_output=True, text=True)
            if 'libx264' in out.stdout:
                return ff, ['-vcodec', 'libx264', '-preset', 'faster',
                            '-crf', '20', '-bf', '0']
            if 'libopenh264' in out.stdout:
                return ff, ['-vcodec', 'libopenh264', '-b:v', '8M', '-bf', '0']
            if 'libvpx' in out.stdout:
                return ff, ['-vcodec', 'libvpx', '-b:v', '8M']
        except FileNotFoundError:
            continue
    return 'ffmpeg', ['-vcodec', 'libx264', '-preset', 'faster',
                      '-crf', '20', '-bf', '0']



def main(rec_dir: str = DEFAULT_REC_DIR, prefix: str = DEFAULT_PREFIX) -> None:

    h5_path     = os.path.join(rec_dir, f'{prefix}_preproc.h5')
    output_path = os.path.join(rec_dir, f'{prefix}_decoding.mp4')

    print('Loading data …')
    data = load_data(h5_path)

    print('Selecting best light block …')
    block = select_best_block(data)
    lo, nd = block['lo'], block['nd']

    print('Running neural decoding …')
    decoded = decode(data, lo, nd)

    print('Finding eye camera video …')
    eye_path = find_eye_video(rec_dir, prefix)

    eyeT_trim = data['eyeT_trim']
    startInd  = data['eyeT_startInd']
    t_start, t_end = block['t_start'], block['t_end']
    eye_mask     = (eyeT_trim >= t_start) & (eyeT_trim <= t_end)
    eye_trim_idx = np.where(eye_mask)[0]
    eye_full_idx = (eye_trim_idx + startInd).astype(int)
    eye_times    = eyeT_trim[eye_trim_idx]
    n_eye_frames = len(eye_full_idx)

    twop_block_t = data['twopT'][lo:nd]
    eye_to_2p    = np.searchsorted(twop_block_t, eye_times).clip(0, len(twop_block_t) - 1)
    prev         = (eye_to_2p - 1).clip(0)
    closer       = (np.abs(twop_block_t[prev] - eye_times)
                    < np.abs(twop_block_t[eye_to_2p] - eye_times))
    eye_to_2p    = np.where(closer, prev, eye_to_2p).astype(np.int32)

    output_fps = 60
    print(f'  Eye frames in block: {n_eye_frames}  →  {output_fps} fps real-time output')

    print(f'Pre-loading {n_eye_frames} eye frames …')
    t0 = time.time()
    eye_frames = preload_eye_frames(eye_path, eye_full_idx)
    print(f'  done in {time.time()-t0:.1f}s  ({eye_frames.nbytes / 1e6:.0f} MB)')

    init_data = dict(
        twopT          = data['twopT'],
        lo             = lo,
        nd             = nd,
        t_start        = t_start,
        eye_times      = eye_times,
        eye_to_2p      = eye_to_2p,
        gt_theta       = decoded['gt_theta'],
        gt_phi         = decoded['gt_phi'],
        gt_X0          = decoded['gt_X0'],
        gt_Y0          = decoded['gt_Y0'],
        pred_theta     = decoded['pred_theta'],
        pred_phi       = decoded['pred_phi'],
        pred_X0        = decoded['pred_X0'],
        pred_Y0        = decoded['pred_Y0'],
        gt_longaxis    = decoded['gt_longaxis'],
        gt_shortaxis   = decoded['gt_shortaxis'],
        gt_ellipse_phi = decoded['gt_ellipse_phi'],
        eye_frames     = eye_frames,
    )

    _probe = plt.figure(figsize=FIGSIZE, dpi=DPI)
    FIG_H  = int(_probe.get_figheight() * DPI)
    FIG_W  = int(_probe.get_figwidth()  * DPI)
    plt.close(_probe)
    FIG_H += FIG_H % 2
    FIG_W += FIG_W % 2

    print(f'Output resolution: {FIG_W}x{FIG_H} px  |  '
          f'{n_eye_frames} frames at {output_fps} fps')

    ffmpeg_bin, codec_args = _find_ffmpeg()
    ffmpeg_cmd = [
        ffmpeg_bin, '-y',
        '-f',       'rawvideo',
        '-vcodec',  'rawvideo',
        '-s',       f'{FIG_W}x{FIG_H}',
        '-pix_fmt', 'rgb24',
        '-r',       str(output_fps),
        '-i',       'pipe:0',
        *codec_args,
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        output_path,
    ]
    print(f'Writing: {output_path}  (ffmpeg codec: {codec_args[1]})')
    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    n_workers = max(1, min(cpu_count() - 1, 6))
    print(f'Rendering with {n_workers} worker(s) …')
    t0, n_written = time.time(), 0

    try:
        with Pool(processes=n_workers, initializer=worker_init,
                  initargs=(init_data,)) as pool:
            for frame_bytes in pool.imap(render_frame, range(n_eye_frames), chunksize=4):
                ffmpeg_proc.stdin.write(frame_bytes)
                n_written += 1
                if n_written % 300 == 0:
                    elapsed = time.time() - t0
                    print(f'  {n_written}/{n_eye_frames} frames  '
                          f'({n_written/elapsed:.1f} fps)')
    except BrokenPipeError:
        print('BrokenPipeError — ffmpeg stderr:')
        print(ffmpeg_proc.stderr.read().decode(errors='replace'))
        raise

    ffmpeg_proc.stdin.close()
    retcode = ffmpeg_proc.wait()
    if retcode != 0:
        print(f'ffmpeg exited with code {retcode}:')
        print(ffmpeg_proc.stderr.read().decode(errors='replace')[-3000:])
    elapsed = time.time() - t0
    print(f'Done in {elapsed:.1f}s  ({n_eye_frames / elapsed:.1f} frames/s)')
    print(f'Saved: {output_path}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Decoding video: population-code decoding of pupil position.')
    parser.add_argument('--rec_dir', default=DEFAULT_REC_DIR,
                        help='Recording directory (contains preproc.h5 and *_deinter.avi)')
    parser.add_argument('--prefix',  default=DEFAULT_PREFIX,
                        help='File prefix (e.g. 251020_DMM_DMM056_fm_01)')
    args = parser.parse_args()
    main(rec_dir=args.rec_dir, prefix=args.prefix)
