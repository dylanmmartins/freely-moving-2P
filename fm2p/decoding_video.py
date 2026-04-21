

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

from .utils.paths import find


DEFAULT_REC_DIR = (
    '/home/dylan/Storage/freely_moving_data/_V1PPC/'
    'cohort02_recordings/cohort02_recordings/'
    '251020_DMM_DMM056_pos08/fm1'
)
DEFAULT_PREFIX = '251020_DMM_DMM056_fm_01'


T_OFFSET_S    = 5.0
TRACE_WIN_S   = 20.0
TRACE_SMOOTH  = 0.8
FIGSIZE       = (18, 8.5)
DPI           = 80
FS            = 27

GT_COLOR      = '#cccccc'
PRED_COLOR    = 'tab:red'
FIG_BG        = 'k'
TRACE_BG      = 'k'

EYE_W, EYE_H  = 640, 480

EYE_CROP_X1 = int(EYE_W * 2 / 3)
EYE_CROP_Y0 = int(EYE_H * 0.20)
EYE_CROP_Y1 = int(EYE_H * 0.70)


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
        pct = 95.0 * int(np.sum(
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
    print(f'  Ridge fit: {X_fit.shape[0]} frames × {X_fit.shape[1]} cells ...')
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


def find_eye_video(rec_dir: str, prefix: str = '') -> str:

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

    gs = GridSpec(2, 2, figure=fig,
                  height_ratios=[1, 1.4],
                  hspace=0.40, wspace=0.30,
                  left=0.10, right=0.97, top=0.97, bottom=0.12)

    ax_theta = fig.add_subplot(gs[0, 0])
    ax_phi   = fig.add_subplot(gs[0, 1])
    ax_eye   = fig.add_subplot(gs[1, 0])
    ax_ell   = fig.add_subplot(gs[1, 1])

    def _style(ax):
        ax.set_facecolor(TRACE_BG)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('0.4')
        ax.spines['bottom'].set_color('0.4')
        ax.tick_params(colors='0.6', labelcolor='0.6', labelsize=FS, length=4)

    for ax in (ax_theta, ax_phi):
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
        x_ctr, y_ctr, half = EYE_W / 2, EYE_H / 2, 95.0
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

    from matplotlib.lines import Line2D as _Line2D
    ax_ell.legend(
        handles=[
            _Line2D([0], [0], color=GT_COLOR,   lw=2.7, label='measured'),
            _Line2D([0], [0], color=PRED_COLOR, lw=2.7, label='decoded'),
        ],
        loc='upper right', facecolor='k', edgecolor='0.4',
        labelcolor='0.6', fontsize=int(FS * 0.7),
    )

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



def save_diagnostic_figs(pairs: list, output_path: str, title: str = '') -> None:

    from matplotlib.backends.backend_pdf import PdfPages

    n = len(pairs)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    with PdfPages(output_path) as pdf:


        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4.5 * ncols, 4.2 * nrows),
                                 squeeze=False)
        if title:
            fig.suptitle(title, fontsize=12)
        for i, (label, gt, pred) in enumerate(pairs):
            ax = axes[i // ncols][i % ncols]
            valid = np.isfinite(gt) & np.isfinite(pred)
            if valid.sum() > 1:
                ax.scatter(gt[valid], pred[valid], s=2, alpha=0.3,
                           color='steelblue', rasterized=True)
                lo_v = min(gt[valid].min(), pred[valid].min())
                hi_v = max(gt[valid].max(), pred[valid].max())
                ax.plot([lo_v, hi_v], [lo_v, hi_v], 'k--', lw=1.2, alpha=0.6)
                r = float(np.corrcoef(gt[valid], pred[valid])[0, 1])
                ax.set_title(f'{label}   r = {r:.3f}', fontsize=10)
            else:
                ax.set_title(f'{label}   (no data)', fontsize=10)
            ax.set_xlabel('measured', fontsize=9)
            ax.set_ylabel('decoded',  fontsize=9)
            ax.tick_params(labelsize=8)
        for j in range(i + 1, nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)
        fig.tight_layout(rect=[0, 0, 1, 0.96] if title else [0, 0, 1, 1])
        pdf.savefig(fig, dpi=120)
        plt.close(fig)

   
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5.5 * ncols, 3.0 * nrows),
                                 squeeze=False)
        if title:
            fig.suptitle(title + ' — time series', fontsize=12)
        for i, (label, gt, pred) in enumerate(pairs):
            ax = axes[i // ncols][i % ncols]
            t = np.arange(len(gt))
            gt_s   = _smooth_trace(interp_short_gaps(gt))
            pred_s = _smooth_trace(pred)
            ax.plot(t, gt_s,   lw=1.2, alpha=0.8, color='#555555', label='measured')
            ax.plot(t, pred_s, lw=1.2, alpha=0.8, color='tab:red',  label='decoded')
            ax.set_title(label, fontsize=10)
            ax.set_xlabel('frame', fontsize=9)
            ax.tick_params(labelsize=8)
            if i == 0:
                ax.legend(fontsize=8, loc='upper right')
        for j in range(i + 1, nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)
        fig.tight_layout(rect=[0, 0, 1, 0.96] if title else [0, 0, 1, 1])
        pdf.savefig(fig, dpi=120)
        plt.close(fig)

    print(f'  Diagnostics saved: {output_path}')


HEAD_FIGSIZE = (22, 10)

_W2: dict = {}


def _rotate_pts(pts: np.ndarray, angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return pts @ np.array([[c, -s], [s, c]]).T


def load_head_decoding(preproc_path: str, traces_path: str,
                       ap_key: str,
                       region: str = 'V1', cond: str = 'l') -> dict:

    with h5py.File(traces_path, 'r') as f:
        grp = f[ap_key][region]

        def _get(bname):
            key = f'{bname}_{cond}'
            if key not in grp:
                return None, None, None
            return (grp[key]['y_true'][()].astype(float),
                    grp[key]['y_pred'][()].astype(float),
                    grp[key]['test_twop_idx'][()].astype(int))

        theta_t, theta_p, theta_idx = _get('theta')
        phi_t,   phi_p,   phi_idx   = _get('phi')
        pitch_t, pitch_p, pitch_idx = _get('pitch')
        roll_t,  roll_p,  roll_idx  = _get('roll')

    all_idx = np.concatenate([
        ix for ix in (theta_idx, phi_idx, pitch_idx, roll_idx) if ix is not None
    ])
    if len(all_idx) == 0:
        raise ValueError(f'No decoded results for {ap_key}/{region}/{cond}')
    lo      = int(all_idx.min())
    nd      = int(all_idx.max()) + 1
    n_block = nd - lo
    block_t = np.arange(n_block, dtype=float)

    with h5py.File(preproc_path, 'r') as f:
        twopT     = f['twopT'][()].astype(float)
        eyeT_trim = f['eyeT_trim'][()].astype(float)
        startInd  = int(f['eyeT_startInd'][()])
        n_trim    = len(eyeT_trim)

        def _eye(key):
            return f[key][()][startInd:startInd + n_trim].astype(float)

        theta_e = _eye('theta')
        phi_e   = _eye('phi')
        X0_e    = _eye('X0')
        Y0_e    = _eye('Y0')
        la_e    = _eye('longaxis')
        sa_e    = _eye('shortaxis')
        ephi_e  = _eye('ellipse_phi')
        pitch_full = f['pitch_twop_interp'][()].astype(float)
        roll_full  = f['roll_twop_interp' ][()].astype(float)

    def to_2p(sig):
        return np.interp(twopT, eyeT_trim, sig)

    theta_2p = to_2p(theta_e)
    phi_2p   = to_2p(phi_e)

    gt_theta_rad = theta_2p   [lo:nd]
    gt_phi_rad   = phi_2p     [lo:nd]
    gt_pitch     = pitch_full [lo:nd]
    gt_roll      = roll_full  [lo:nd]
    gt_X0        = to_2p(X0_e)   [lo:nd]
    gt_Y0        = to_2p(Y0_e)   [lo:nd]
    gt_la        = to_2p(la_e)   [lo:nd]
    gt_sa        = to_2p(sa_e)   [lo:nd]
    gt_ephi      = to_2p(ephi_e) [lo:nd]

    def _cont(idx, y_pred):
        if idx is None:
            return np.full(n_block, np.nan)
        rel = idx - lo
        in_blk = (rel >= 0) & (rel < n_block)
        rel    = rel[in_blk]
        y_pred = y_pred[in_blk]
        if len(rel) < 2:
            return np.full(n_block, np.nan)
        order  = np.argsort(rel)
        rel    = rel[order];  y_pred = y_pred[order]
        out    = np.interp(block_t, rel, y_pred)
        out[block_t < rel[0]]  = np.nan
        out[block_t > rel[-1]] = np.nan
        return out

    pred_theta_rad = _cont(theta_idx, theta_p)
    pred_phi_rad   = _cont(phi_idx,   phi_p)
    pred_pitch     = _cont(pitch_idx, pitch_p)
    pred_roll      = _cont(roll_idx,  roll_p)

    valid_e = (np.isfinite(theta_e) & np.isfinite(phi_e)
               & np.isfinite(X0_e)  & np.isfinite(Y0_e))
    poly   = PolynomialFeatures(degree=2, include_bias=True)
    feat_e = poly.fit_transform(np.column_stack([theta_e[valid_e], phi_e[valid_e]]))
    reg_x0 = Ridge(alpha=1.0).fit(feat_e, X0_e[valid_e])
    reg_y0 = Ridge(alpha=1.0).fit(feat_e, Y0_e[valid_e])

    pred_X0 = np.full(n_block, np.nan)
    pred_Y0 = np.full(n_block, np.nan)
    ok_pred = np.isfinite(pred_theta_rad) & np.isfinite(pred_phi_rad)
    if ok_pred.any():
        feat_p = poly.transform(
            np.column_stack([pred_theta_rad[ok_pred], pred_phi_rad[ok_pred]]))
        pred_X0[ok_pred] = reg_x0.predict(feat_p)
        pred_Y0[ok_pred] = reg_y0.predict(feat_p)

    print(f'  Block: 2P frames [{lo}:{nd}]  ({n_block} frames)')
    for name, gt_b, pred_b in [
        ('theta', gt_theta_rad, pred_theta_rad), ('phi', gt_phi_rad, pred_phi_rad),
        ('pitch', gt_pitch,     pred_pitch),     ('roll', gt_roll,   pred_roll),
    ]:
        ok = np.isfinite(gt_b) & np.isfinite(pred_b)
        if ok.sum() > 1:
            r = float(np.corrcoef(gt_b[ok], pred_b[ok])[0, 1])
            print(f'  {name}: r={r:.3f}  ({ok.sum()} frames)')

    return dict(
        twopT          = twopT,
        lo             = lo,
        nd             = nd,
        t_start        = float(twopT[lo]),
        gt_theta       = np.degrees(gt_theta_rad),
        gt_phi         = np.degrees(gt_phi_rad),
        gt_pitch       = gt_pitch,
        gt_roll        = gt_roll,
        gt_X0          = gt_X0,
        gt_Y0          = gt_Y0,
        pred_theta     = np.degrees(pred_theta_rad),
        pred_phi       = np.degrees(pred_phi_rad),
        pred_pitch     = pred_pitch,
        pred_roll      = pred_roll,
        pred_X0        = pred_X0,
        pred_Y0        = pred_Y0,
        gt_longaxis    = gt_la,
        gt_shortaxis   = gt_sa,
        gt_ellipse_phi = gt_ephi,
    )


def worker_init_head(init_data: dict) -> None:
    global _W2
    _W2 = init_data

    fig = plt.figure(figsize=HEAD_FIGSIZE, dpi=DPI, facecolor=FIG_BG)

    gs = GridSpec(2, 12, figure=fig,
                  height_ratios=[1, 1.5],
                  hspace=0.45, wspace=0.55,
                  left=0.07, right=0.97, top=0.96, bottom=0.10)

    ax_theta  = fig.add_subplot(gs[0, 0:3])
    ax_phi    = fig.add_subplot(gs[0, 3:6])
    ax_pitch  = fig.add_subplot(gs[0, 6:9])
    ax_roll   = fig.add_subplot(gs[0, 9:12])

    ax_ell       = fig.add_subplot(gs[1, 0:4])
    ax_pitch_vis = fig.add_subplot(gs[1, 4:8])
    ax_roll_vis  = fig.add_subplot(gs[1, 8:12])

    def _style(ax):
        ax.set_facecolor(TRACE_BG)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('0.4')
        ax.spines['bottom'].set_color('0.4')
        ax.tick_params(colors='0.6', labelcolor='0.6',
                       labelsize=int(FS * 0.75), length=4)

    twopT  = init_data['twopT']
    lo     = init_data['lo']
    nd     = init_data['nd']
    t_zero = init_data['t_start']
    tt     = twopT[lo:nd] - t_zero

    trace_specs = [
        (init_data['gt_theta'],  init_data['pred_theta'],  r'$\theta$ (°)',  ax_theta),
        (init_data['gt_phi'],    init_data['pred_phi'],    r'$\phi$ (°)',    ax_phi),
        (init_data['gt_pitch'],  init_data['pred_pitch'],  'pitch (°)',      ax_pitch),
        (init_data['gt_roll'],   init_data['pred_roll'],   'roll (°)',       ax_roll),
    ]

    trace_axes    = []
    trace_cursors = []

    for gt, pred, ylabel, ax in trace_specs:
        _style(ax)
        gt_s   = _smooth_trace(interp_short_gaps(gt))
        pred_s = _smooth_trace(pred)

        ax.plot(tt, gt_s,   color=GT_COLOR,   lw=2.2, alpha=0.9)
        ax.plot(tt, pred_s, color=PRED_COLOR, lw=2.2, alpha=0.85)

        ax.set_ylabel(ylabel, fontsize=int(FS * 0.75), color='0.6')
        ax.set_xlabel('time (s)', fontsize=int(FS * 0.75), color='0.6')
        ax.set_xlim(-TRACE_WIN_S / 2, TRACE_WIN_S / 2)

        finite = gt_s[np.isfinite(gt_s)]
        if len(finite) > 2:
            lo_y, hi_y = np.nanpercentile(finite, [1, 99])
            mg = max(0.05 * abs(hi_y - lo_y), 0.005)
            ax.set_ylim(lo_y - mg, hi_y + mg)

        cursor = ax.axvline(0, color='w', lw=2.5, ls='--', alpha=0.7)
        trace_axes.append(ax)
        trace_cursors.append(cursor)

    ax_ell.set_facecolor('k')
    ax_ell.set_aspect('equal')
    ax_ell.spines['top'].set_visible(False)
    ax_ell.spines['right'].set_visible(False)
    ax_ell.spines['left'].set_color('0.4')
    ax_ell.spines['bottom'].set_color('0.4')
    ax_ell.tick_params(colors='0.6', labelcolor='0.6',
                       labelsize=int(FS * 0.7), length=4)
    ax_ell.set_xlabel('X (px)', fontsize=int(FS * 0.7), color='0.6')
    ax_ell.set_ylabel('Y (px)', fontsize=int(FS * 0.7), color='0.6')

    gt_X  = init_data['gt_X0']
    gt_Y  = init_data['gt_Y0']
    la    = init_data['gt_longaxis']
    x_fin = gt_X[np.isfinite(gt_X)]
    y_fin = gt_Y[np.isfinite(gt_Y)]
    la_med = float(np.nanmedian(la[np.isfinite(la)])) if np.any(np.isfinite(la)) else 30.0
    if len(x_fin) > 0:
        x_ctr = float(np.nanmedian(x_fin))
        y_ctr = float(np.nanmedian(y_fin))
        half  = max(float(np.nanstd(x_fin)) * 4, float(np.nanstd(y_fin)) * 4,
                    la_med * 4, 40.0)
    else:
        x_ctr, y_ctr, half = EYE_W / 2, EYE_H / 2, 95.0
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

    from matplotlib.lines import Line2D as _Line2D
    ax_ell.legend(
        handles=[
            _Line2D([0], [0], color=GT_COLOR,   lw=2.7, label='measured'),
            _Line2D([0], [0], color=PRED_COLOR, lw=2.7, label='decoded'),
        ],
        loc='upper right', facecolor='k', edgecolor='0.4',
        labelcolor='0.6', fontsize=int(FS * 0.6),
    )

    for ax in (ax_pitch_vis, ax_roll_vis):
        ax.set_facecolor('k')
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)

    ax_pitch_vis.set_title('pitch', color='0.6', fontsize=int(FS * 0.8), pad=4)
    ax_roll_vis .set_title('roll',  color='0.6', fontsize=int(FS * 0.8), pad=4)

    _tri_pts = np.array([[-0.8, -0.45], [-0.8, 0.45], [1.6, 0.0]])

    from matplotlib.patches import Polygon as _Polygon
    tri_patch = _Polygon(_tri_pts, closed=True,
                         facecolor='0.45', edgecolor='0.7', linewidth=1.5, zorder=3)
    ax_pitch_vis.add_patch(tri_patch)

    pred_pitch_line, = ax_pitch_vis.plot([0.0, 1.8], [0.0, 0.0],
                                          color=PRED_COLOR, lw=2.5, zorder=4)

    from matplotlib.patches import Circle as _Circle
    roll_circle = _Circle((0.0, 0.0), 0.72,
                           facecolor='0.45', edgecolor='0.7', linewidth=1.5, zorder=3)
    ax_roll_vis.add_patch(roll_circle)

    _ear_L_pts = np.array([[-0.58, 0.58], [-0.12, 0.58], [-0.35, 1.12]])
    _ear_R_pts = np.array([[ 0.12, 0.58], [ 0.58, 0.58], [ 0.35, 1.12]])

    ear_L = _Polygon(_ear_L_pts, closed=True,
                     facecolor='0.45', edgecolor='0.7', linewidth=1.5, zorder=3)
    ear_R = _Polygon(_ear_R_pts, closed=True,
                     facecolor='0.45', edgecolor='0.7', linewidth=1.5, zorder=3)
    ax_roll_vis.add_patch(ear_L)
    ax_roll_vis.add_patch(ear_R)

    pred_roll_line, = ax_roll_vis.plot([-1.8, 1.8], [0.0, 0.0],
                                        color=PRED_COLOR, lw=2.5, zorder=4)

    time_txt = fig.text(0.50, 0.005, '', color='0.6', fontsize=int(FS * 0.8),
                        ha='center', va='bottom')

    _W2['fig']             = fig
    _W2['trace_axes']      = trace_axes
    _W2['trace_cursors']   = trace_cursors
    _W2['ell_gt']          = ell_gt
    _W2['ell_pred']        = ell_pred
    _W2['tri_patch']       = tri_patch
    _W2['tri_pts']         = _tri_pts
    _W2['pred_pitch_line'] = pred_pitch_line
    _W2['roll_circle']     = roll_circle
    _W2['ear_L']           = ear_L
    _W2['ear_R']           = ear_R
    _W2['ear_L_pts']       = _ear_L_pts
    _W2['ear_R_pts']       = _ear_R_pts
    _W2['pred_roll_line']  = pred_roll_line
    _W2['time_txt']        = time_txt
    _W2['FIG_H']           = int(fig.get_figheight() * DPI)
    _W2['FIG_W']           = int(fig.get_figwidth()  * DPI)


def render_frame_head(fi: int) -> bytes:
    t_zero  = _W2['t_start']
    lo      = _W2['lo']
    t_rel   = float(_W2['twopT'][lo + fi]) - t_zero
    half    = TRACE_WIN_S / 2.0

    for ax, cur in zip(_W2['trace_axes'], _W2['trace_cursors']):
        ax.set_xlim(t_rel - half, t_rel + half)
        cur.set_xdata([t_rel, t_rel])

    x0_gt = float(_W2['gt_X0'][fi])
    y0_gt = float(_W2['gt_Y0'][fi])
    la    = float(_W2['gt_longaxis'][fi])
    sa    = float(_W2['gt_shortaxis'][fi])
    phi   = float(_W2['gt_ellipse_phi'][fi])
    x0_p  = float(_W2['pred_X0'][fi])
    y0_p  = float(_W2['pred_Y0'][fi])

    gt_ok = np.isfinite(x0_gt + y0_gt + la + sa + phi) and la > 0 and sa > 0
    if gt_ok:
        _W2['ell_gt'].set_center((x0_gt, y0_gt))
        _W2['ell_gt'].width  = 2.0 * la
        _W2['ell_gt'].height = 2.0 * sa
        _W2['ell_gt'].angle  = float(np.degrees(phi))
        _W2['ell_gt'].set_visible(True)
    else:
        _W2['ell_gt'].set_visible(False)

    pred_ok = gt_ok and np.isfinite(x0_p + y0_p)
    if pred_ok:
        _W2['ell_pred'].set_center((x0_p, y0_p))
        _W2['ell_pred'].width  = 2.0 * la
        _W2['ell_pred'].height = 2.0 * sa
        _W2['ell_pred'].angle  = float(np.degrees(phi))
        _W2['ell_pred'].set_visible(True)
    else:
        _W2['ell_pred'].set_visible(False)

    true_pitch = float(_W2['gt_pitch'][fi])
    pred_pitch = float(_W2['pred_pitch'][fi])

    _W2['tri_patch'].set_visible(np.isfinite(true_pitch))
    if np.isfinite(true_pitch):
        _W2['tri_patch'].set_xy(_rotate_pts(_W2['tri_pts'], true_pitch))

    _W2['pred_pitch_line'].set_visible(np.isfinite(pred_pitch))
    if np.isfinite(pred_pitch):
        a = np.radians(pred_pitch)
        llen = 1.8
        _W2['pred_pitch_line'].set_data(
            [0.0, llen * np.cos(a)], [0.0, llen * np.sin(a)])

    true_roll = float(_W2['gt_roll'][fi])
    pred_roll = float(_W2['pred_roll'][fi])

    ears_ok = np.isfinite(true_roll)
    _W2['ear_L'].set_visible(ears_ok)
    _W2['ear_R'].set_visible(ears_ok)
    if ears_ok:
        _W2['ear_L'].set_xy(_rotate_pts(_W2['ear_L_pts'], true_roll))
        _W2['ear_R'].set_xy(_rotate_pts(_W2['ear_R_pts'], true_roll))

    _W2['pred_roll_line'].set_visible(np.isfinite(pred_roll))
    if np.isfinite(pred_roll):
        a = np.radians(pred_roll)
        llen = 1.8
        _W2['pred_roll_line'].set_data(
            [-llen * np.cos(a), llen * np.cos(a)],
            [-llen * np.sin(a), llen * np.sin(a)])

    _W2['time_txt'].set_text(f't = {t_rel:.2f} s')

    _W2['fig'].canvas.draw()
    buf = _W2['fig'].canvas.buffer_rgba()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(
        _W2['FIG_H'], _W2['FIG_W'], 4)
    return arr[:, :, :3].tobytes()


def main_head(preproc_path: str) -> None:

    rec_dir = os.path.dirname(os.path.abspath(preproc_path))
    output_path = os.path.join(rec_dir, 'head_decoding.mp4')

    print('Loading data ...')
    data = load_data(preproc_path)

    with h5py.File(preproc_path, 'r') as f:
        n_twop = len(data['twopT'])
        pitch_full = (f['pitch_twop_interp'][()].astype(float)
                      if 'pitch_twop_interp' in f
                      else np.full(n_twop, np.nan))
        roll_full  = (f['roll_twop_interp'][()].astype(float)
                      if 'roll_twop_interp' in f
                      else np.full(n_twop, np.nan))

    print('Selecting best light block ...')
    block = select_best_block(data)
    lo, nd = block['lo'], block['nd']

    print('Running neural decoding ...')
    decoded = decode(data, lo, nd)

    neural_block = data['neural'][:, lo:nd].T.astype(float)
    gt_pitch = pitch_full[lo:nd]
    gt_roll  = roll_full [lo:nd]

    valid_pr = (np.isfinite(gt_pitch) & np.isfinite(gt_roll)
                & np.isfinite(neural_block).all(axis=1))
    if valid_pr.sum() > 1:
        ridge_pitch = Ridge(alpha=1.0).fit(neural_block[valid_pr], gt_pitch[valid_pr])
        ridge_roll  = Ridge(alpha=1.0).fit(neural_block[valid_pr], gt_roll [valid_pr])
        pred_pitch  = ridge_pitch.predict(neural_block)
        pred_roll   = ridge_roll .predict(neural_block)
        r_p = float(np.corrcoef(gt_pitch[valid_pr], pred_pitch[valid_pr])[0, 1])
        r_r = float(np.corrcoef(gt_roll [valid_pr], pred_roll [valid_pr])[0, 1])
        print(f'  Decoded: pitch r={r_p:.3f}, roll r={r_r:.3f}')
    else:
        pred_pitch = np.full(nd - lo, np.nan)
        pred_roll  = np.full(nd - lo, np.nan)
        print('  Insufficient valid pitch/roll data for decoding.')

    print('Saving diagnostic figures ...')
    diag_pairs = [
        (r'$\theta$ (°)',  decoded['gt_theta'],  decoded['pred_theta']),
        (r'$\phi$ (°)',    decoded['gt_phi'],     decoded['pred_phi']),
        ('X0 (px)',        decoded['gt_X0'],      decoded['pred_X0']),
        ('Y0 (px)',        decoded['gt_Y0'],      decoded['pred_Y0']),
        ('pitch (°)',      gt_pitch,              pred_pitch),
        ('roll (°)',       gt_roll,               pred_roll),
    ]
    save_diagnostic_figs(diag_pairs,
                         os.path.join(rec_dir, 'head_decoding_diagnostics.pdf'),
                         title='Head decoding diagnostics')

    twopT   = data['twopT']
    n_video = nd - lo

    twop_fps   = float(np.median(1.0 / np.diff(twopT[lo:nd])))
    output_fps = max(1, min(60, int(round(twop_fps))))
    print(f'  2P frame rate: {twop_fps:.1f} Hz  →  output {output_fps} fps')

    _probe = plt.figure(figsize=HEAD_FIGSIZE, dpi=DPI)
    FIG_H  = int(_probe.get_figheight() * DPI)
    FIG_W  = int(_probe.get_figwidth()  * DPI)
    plt.close(_probe)
    FIG_H += FIG_H % 2
    FIG_W += FIG_W % 2

    print(f'Output resolution: {FIG_W}x{FIG_H} px  |  '
          f'{n_video} frames at {output_fps} fps')

    isg = interp_short_gaps
    init_data = dict(
        twopT          = twopT,
        lo             = lo,
        nd             = nd,
        t_start        = block['t_start'],
        gt_theta       = decoded['gt_theta'],
        gt_phi         = decoded['gt_phi'],
        gt_pitch       = gt_pitch,
        gt_roll        = gt_roll,
        gt_X0          = isg(decoded['gt_X0']),
        gt_Y0          = isg(decoded['gt_Y0']),
        pred_theta     = decoded['pred_theta'],
        pred_phi       = decoded['pred_phi'],
        pred_pitch     = pred_pitch,
        pred_roll      = pred_roll,
        pred_X0        = isg(decoded['pred_X0']),
        pred_Y0        = isg(decoded['pred_Y0']),
        gt_longaxis    = isg(decoded['gt_longaxis']),
        gt_shortaxis   = isg(decoded['gt_shortaxis']),
        gt_ellipse_phi = isg(decoded['gt_ellipse_phi']),
    )

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

    n_workers = max(1, cpu_count() - 1)
    print(f'Rendering with {n_workers} worker(s) ...')
    t0, n_written = time.time(), 0

    try:
        with Pool(processes=n_workers, initializer=worker_init_head,
                  initargs=(init_data,)) as pool:
            for frame_bytes in pool.imap(render_frame_head, range(n_video), chunksize=4):
                ffmpeg_proc.stdin.write(frame_bytes)
                n_written += 1
                if n_written % 300 == 0:
                    elapsed = time.time() - t0
                    print(f'  {n_written}/{n_video} frames  '
                          f'({n_written/elapsed:.1f} fps)')
    except BrokenPipeError:
        print('BrokenPipeError -- ffmpeg stderr:')
        print(ffmpeg_proc.stderr.read().decode(errors='replace'))
        raise

    ffmpeg_proc.stdin.close()
    retcode = ffmpeg_proc.wait()
    if retcode != 0:
        print(f'ffmpeg exited with code {retcode}:')
        print(ffmpeg_proc.stderr.read().decode(errors='replace')[-3000:])
    elapsed = time.time() - t0
    print(f'Done in {elapsed:.1f}s  ({n_video / elapsed:.1f} frames/s)')
    print(f'Saved: {output_path}')


def main(rec_dir) -> None:

    h5_path     = os.path.join(find('*_preproc.h5', rec_dir, MR=True))
    output_path = os.path.join(rec_dir, f'decoding_video.mp4')

    print('Loading data ...')
    data = load_data(h5_path)

    print('Selecting best light block ...')
    block = select_best_block(data)
    lo, nd = block['lo'], block['nd']

    print('Running neural decoding ...')
    decoded = decode(data, lo, nd)

    twopT   = data['twopT']
    lo_video = int(np.searchsorted(twopT, block['t_start'] + T_OFFSET_S))
    lo_video = max(lo, min(lo_video, nd - 1))
    trim     = lo_video - lo
    t_start  = float(twopT[lo_video])

    for key in ('gt_theta', 'gt_phi', 'gt_X0', 'gt_Y0',
                'pred_theta', 'pred_phi', 'pred_X0', 'pred_Y0',
                'gt_longaxis', 'gt_shortaxis', 'gt_ellipse_phi'):
        decoded[key] = decoded[key][trim:]

    lo = lo_video

    nd_cap   = int(np.searchsorted(twopT, t_start + 95.0))
    nd       = min(nd, nd_cap)
    n_video  = nd - lo

    for key in ('gt_theta', 'gt_phi', 'gt_X0', 'gt_Y0',
                'pred_theta', 'pred_phi', 'pred_X0', 'pred_Y0',
                'gt_longaxis', 'gt_shortaxis', 'gt_ellipse_phi'):
        decoded[key] = decoded[key][:n_video]

    print('Saving diagnostic figures ...')
    diag_pairs = [
        (r'$\theta$ (°)',  decoded['gt_theta'], decoded['pred_theta']),
        (r'$\phi$ (°)',    decoded['gt_phi'],   decoded['pred_phi']),
        ('X0 (px)',        decoded['gt_X0'],    decoded['pred_X0']),
        ('Y0 (px)',        decoded['gt_Y0'],    decoded['pred_Y0']),
    ]
    save_diagnostic_figs(diag_pairs,
                         os.path.join(rec_dir, 'decoding_diagnostics.pdf'),
                         title='Eye decoding diagnostics')

    print('Finding eye camera video ...')
    eye_path = find_eye_video(rec_dir)

    eyeT_trim = data['eyeT_trim']
    startInd  = data['eyeT_startInd']
    t_end     = min(block['t_end'], t_start + 95.0)
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
    print(f'  Eye frames in block: {n_eye_frames}  ->  {output_fps} fps real-time output')

    print(f'Pre-loading {n_eye_frames} eye frames ...')
    t0 = time.time()
    eye_frames = preload_eye_frames(eye_path, eye_full_idx)
    print(f'  done in {time.time()-t0:.1f}s  ({eye_frames.nbytes / 1e6:.0f} MB)')

    isg = interp_short_gaps
    init_data = dict(
        twopT          = data['twopT'],
        lo             = lo,
        nd             = nd,
        t_start        = t_start,
        eye_times      = eye_times,
        eye_to_2p      = eye_to_2p,
        gt_theta       = decoded['gt_theta'],
        gt_phi         = decoded['gt_phi'],
        gt_X0          = isg(decoded['gt_X0']),
        gt_Y0          = isg(decoded['gt_Y0']),
        pred_theta     = decoded['pred_theta'],
        pred_phi       = decoded['pred_phi'],
        pred_X0        = isg(decoded['pred_X0']),
        pred_Y0        = isg(decoded['pred_Y0']),
        gt_longaxis    = isg(decoded['gt_longaxis']),
        gt_shortaxis   = isg(decoded['gt_shortaxis']),
        gt_ellipse_phi = isg(decoded['gt_ellipse_phi']),
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

    n_workers = max(1, cpu_count()-1)
    print(f'Rendering with {n_workers} worker(s) ...')
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
        print('BrokenPipeError -- ffmpeg stderr:')
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
    parser.add_argument('--preproc', default=None,
                        help='Path to preproc.h5 (for --head mode)')
    parser.add_argument('--head', action='store_true', default=True)
    args = parser.parse_args()

    if args.head:
        preproc = args.preproc or find('*preproc.h5', args.rec_dir, MR=True)
        main_head(preproc)
    main(rec_dir=args.rec_dir)
