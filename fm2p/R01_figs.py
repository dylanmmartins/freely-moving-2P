
if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

import argparse
import glob
import os

import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.ndimage import gaussian_filter1d
from scipy.signal import lfilter, medfilt
from scipy.stats import kurtosis as scipy_kurtosis

import matplotlib as mpl
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 7

from fm2p.utils.helper import interp_short_gaps, interp_short_gaps_circ


DEFAULT_REC_DIR = (
    '/home/dylan/Storage/freely_moving_data/_V1PPC/'
    'cohort02_recordings/cohort02_recordings/'
    '251020_DMM_DMM056_pos08/fm1'
)
DEFAULT_PREFIX = '251020_DMM_DMM056_fm_01'


FIGSIZE      = (5, 6.5)
DPI          = 300
N_CELLS      = 40
TRACE_WIN_S  = 120.0    # 1-minute window for dF/F and IMU panels
DFF_OFFSET   = 0.65    # y-spacing between traces — < 1 gives overlap
MIP_STRIDE   = 4       # subsample every Nth tif frame for MIP (speed)
EYE_CROP_R0  = int(480 * 0.10)
EYE_CROP_R1  = int(480 * 0.80)
EYE_CROP_C0  = 0
EYE_CROP_C1  = int(640 * 0.80)

# axon eye camera: pupil sits lower-right — crop against bottom-right of frame
EYE_CROP_AXON_R0 = int(480 * 0.30)
EYE_CROP_AXON_R1 = 480
EYE_CROP_AXON_C0 = int(640 * 0.20)
EYE_CROP_AXON_C1 = 640

matplotlib.rcParams.update({
    'font.family':       'sans-serif',
    'font.size':         8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})



def _load_data(h5_path):
    data = {}
    with h5py.File(h5_path, 'r') as f:
        for key in [
            'eyeT_trim', 'twopT',
            'light_onsets', 'dark_onsets',
            'theta', 'phi',
            'head_x', 'head_y', 'head_yaw_deg',
            'pitch_twop_interp', 'roll_twop_interp',
            'norm_dFF',
            'twop_mean_img',
        ]:
            data[key] = f[key][:]
        data['eyeT_startInd'] = int(f['eyeT_startInd'][()])
    return data


def _select_best_light_block(data):
    eyeT_trim    = data['eyeT_trim']
    twopT        = data['twopT']
    light_onsets = data['light_onsets']
    dark_onsets  = data['dark_onsets']
    startInd     = data['eyeT_startInd']
    theta        = data['theta'][startInd: startInd + len(eyeT_trim)]
    phi          = data['phi'][startInd:   startInd + len(eyeT_trim)]

    hx  = data['head_x'].astype(float)
    hy  = data['head_y'].astype(float)
    spd = np.concatenate([[np.nan], np.sqrt(np.diff(hx)**2 + np.diff(hy)**2)])
    spd_thresh = np.nanpercentile(spd, 25)

    best_idx, best_score = -1, -1.0
    for i in range(1, len(light_onsets)):
        lo = light_onsets[i]
        next_darks = dark_onsets[dark_onsets > lo]
        if len(next_darks) == 0:
            continue
        nd      = next_darks[0]
        t_start = twopT[lo]
        t_end   = twopT[nd]
        mask    = (eyeT_trim >= t_start) & (eyeT_trim <= t_end)
        n_eye   = mask.sum()
        if n_eye == 0:
            continue
        good       = np.sum(mask & ~np.isnan(theta) & ~np.isnan(phi))
        eye_pct    = 100.0 * good / n_eye
        active_pct = 100.0 * float(np.nanmean(spd[lo:nd] > spd_thresh))
        score      = eye_pct * active_pct
        if score > best_score:
            best_score, best_idx = score, i

    lo      = light_onsets[best_idx]
    nd      = dark_onsets[dark_onsets > lo][0]
    t_start = float(twopT[lo])
    t_end   = float(twopT[nd])
    print(f'  Best block: [{lo}:{nd}]  t=[{t_start:.1f}–{t_end:.1f}]s')
    return dict(twop_lo=int(lo), twop_nd=int(nd), t_start=t_start, t_end=t_end)


def _select_top_kurtosis_cells(norm_dff, lo, nd, n=N_CELLS):
    block = norm_dff[:, lo:nd]
    kurts = scipy_kurtosis(block, axis=1, nan_policy='omit')
    kurts = np.where(np.isfinite(kurts), kurts, -np.inf)
    return np.argsort(kurts)[-n:][::-1]


def _get_cell_colors(n):
    rng   = np.random.default_rng(0)
    idxs  = rng.permutation(n)
    return plt.cm.plasma(np.linspace(0.05, 0.92, n))[idxs]


def _smooth_decay_only(tr, tau_frames):
    """Smooth calcium decay without blurring sharp rises.

    Step 1 — median filter (kernel=3): kills single-sample noise spikes
    in flat periods while preserving multi-frame transient edges (median
    is edge-preserving).

    Step 2 — causal exponential moving average: lags behind fast rises
    so raw > smoothed during onset → raw value kept; on the decay
    smoothed > raw → smoothed value used (pointwise max).
    """
    nan_mask = np.isnan(tr)
    tr_fill  = np.where(nan_mask, 0.0, tr)
    tr_fill  = medfilt(tr_fill, kernel_size=3).astype(float)
    tr_fill[nan_mask] = np.nan

    alpha = 1.0 - np.exp(-1.0 / max(float(tau_frames), 1e-6))
    tr_sm = lfilter([alpha], [1.0, -(1.0 - alpha)], np.where(nan_mask, 0.0, tr_fill))
    tr_sm[nan_mask] = np.nan

    out = np.where(tr_fill > tr_sm, tr_fill, tr_sm)
    out[nan_mask] = np.nan
    return out


def _draw_dff_panel(ax, dff_mat, t_dff, title='ΔF/F', smooth_tau=0):
    """Stacked dF/F traces: jet colormap, auto ylim, L-shaped scalebar, no x-axis."""
    dff_mat = np.atleast_2d(np.asarray(dff_mat, dtype=float))
    n_cells, n_t = dff_mat.shape

    # gracefully handle recordings with no usable dF/F window
    if n_cells == 0 or n_t == 0 or len(t_dff) == 0:
        ax.text(0.5, 0.5, 'no dF/F data in window', transform=ax.transAxes,
                ha='center', va='center', fontsize=8, color='0.5')
        ax.set_xlim(0, TRACE_WIN_S)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(bottom=False, labelbottom=False)
        ax.set_title(title, fontsize=9, pad=4, loc='left')
        return

    cell_colors = _get_cell_colors(n_cells)

    traces, dff_ranges = [], []
    for i in range(n_cells):
        tr  = np.asarray(dff_mat[i]).ravel()
        if smooth_tau > 0:
            tr = _smooth_decay_only(tr, smooth_tau)
        finite = tr[np.isfinite(tr)]
        if len(finite) == 0:
            p5, p95 = 0.0, 1.0
        else:
            p5  = float(np.nanpercentile(tr, 5))
            p95 = float(np.nanpercentile(tr, 95))
        rng = max(p95 - p5, 1e-6)
        dff_ranges.append(rng)
        row_offset = (n_cells - 1 - i) * DFF_OFFSET
        traces.append((tr - p5) / rng + row_offset)

    all_vals = np.concatenate([t[np.isfinite(t)] for t in traces if np.any(np.isfinite(t))])
    if len(all_vals):
        # use actual max of the topmost trace so its peaks are never clipped;
        # use a loose percentile for the lower bound to ignore downward outliers
        top_finite = traces[0][np.isfinite(traces[0])]
        y_max = float(np.nanmax(top_finite)) if len(top_finite) else float(n_cells * DFF_OFFSET)
        y_min = float(np.percentile(all_vals, 0.5))
    else:
        y_max = float(n_cells * DFF_OFFSET)
        y_min = 0.0

    for trace, color in zip(traces, cell_colors):
        ax.plot(t_dff, trace, color=color, lw=0.65, alpha=0.9, rasterized=True)

    # L-shaped scalebar — bottom-right corner
    median_rng = float(np.median(dff_ranges))
    sb_t  = 10.0                                  # 10-second horizontal arm
    sb_h  = 100.0 / max(median_rng, 1e-6)         # height = 100 raw dF/F units
    sb_x1 = TRACE_WIN_S * 0.97
    sb_x0 = sb_x1 - sb_t
    sb_y0 = y_min - DFF_OFFSET * 0.5

    ax.plot([sb_x0, sb_x1], [sb_y0, sb_y0], 'k-', lw=1.5,
            solid_capstyle='butt', clip_on=False)
    ax.plot([sb_x0, sb_x0], [sb_y0, sb_y0 + sb_h], 'k-', lw=1.5,
            solid_capstyle='butt', clip_on=False)
    ax.text((sb_x0 + sb_x1) / 2, sb_y0 - 0.25,
            f'{sb_t:.0f} s', ha='center', va='top', fontsize=6)
    ax.text(sb_x0 - 1.5, sb_y0 + sb_h / 2,
            '100 ΔF/F', ha='right', va='center', fontsize=6)

    ax.set_xlim(0, TRACE_WIN_S)
    ax.set_ylim(sb_y0 - 0.3, y_max + DFF_OFFSET * 0.3)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(bottom=False, labelbottom=False)
    ax.set_title(title, fontsize=9, pad=4, loc='left')


def _subtract_band(frame_gray, r0=EYE_CROP_R0, r1=EYE_CROP_R1,
                   c0=EYE_CROP_C0, c1=EYE_CROP_C1):
    left    = frame_gray[:, :10].mean(axis=1).astype(float)
    right   = frame_gray[:, -10:].mean(axis=1).astype(float)
    profile = gaussian_filter1d(0.5 * (left + right), sigma=15)
    corr    = frame_gray.astype(float) - profile[:, np.newaxis]
    # percentile stretch on full frame so dark border pixels anchor the low end
    # and specular reflections don't pin the max
    lo_val = float(np.percentile(corr, 1))
    hi_val = float(np.percentile(corr, 99))
    if hi_val > lo_val:
        corr = (corr - lo_val) / (hi_val - lo_val) * 255.0
    clipped = np.clip(corr, 0, 255).astype(np.uint8)
    return clipped[r0:r1, c0:c1]


def _read_topdown_frame(top_path, frame_idx):
    cap   = cv2.VideoCapture(top_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(0, frame_idx), total - 1))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    fr = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_BGR2RGB)
    return fr


def _read_eye_frame(eye_path, frame_idx, r0=EYE_CROP_R0, r1=EYE_CROP_R1,
                    c0=EYE_CROP_C0, c1=EYE_CROP_C1):
    cap   = cv2.VideoCapture(eye_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(0, frame_idx), total - 1))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return np.zeros((r1 - r0, c1 - c0), dtype=np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (640, 480))
    return _subtract_band(gray, r0=r0, r1=r1, c0=c0, c1=c1)


def _compute_mean_img(tif_path, lo, nd):
    """Mean image over tif frames [lo:nd:MIP_STRIDE]."""
    acc, count = None, 0
    with tifffile.TiffFile(tif_path) as tif:
        n_pages = len(tif.pages)
        for i in range(lo, min(nd, n_pages), MIP_STRIDE):
            frame = tif.pages[i].asarray().astype(np.float32)
            acc   = frame.copy() if acc is None else acc + frame
            count += 1
    if acc is None or count == 0:
        return np.zeros((512, 512), dtype=np.float32)
    return acc / count

def _smooth_circ_deg(arr, sigma):
    """Gaussian smooth circular data in [0, 360) by smoothing sin/cos separately.

    Handles both hard wrap-around jumps (358->2 with no NaN) and NaN gaps
    correctly — sin/cos have no discontinuity at the 0/360 boundary.
    """
    valid = ~np.isnan(arr)
    if not np.any(valid):
        return arr.copy()
    rad  = np.deg2rad(np.where(valid, arr, 0.0))
    c_sm = gaussian_filter1d(np.where(valid, np.cos(rad), 0.0), sigma)
    s_sm = gaussian_filter1d(np.where(valid, np.sin(rad), 0.0), sigma)
    w_sm = gaussian_filter1d(valid.astype(float), sigma)
    angle = np.rad2deg(np.arctan2(s_sm, c_sm)) % 360
    angle[w_sm < 0.05] = np.nan  # mask regions far from any valid sample
    return angle


def _smooth_imu(data):
    n   = len(data['twopT'])
    sig = 1.5
    pitch = gaussian_filter1d(
        interp_short_gaps(data['pitch_twop_interp'][:n].astype(float)), sig)
    roll  = gaussian_filter1d(
        interp_short_gaps(data['roll_twop_interp'][:n].astype(float)), sig)
    # fill short NaN gaps circularly first, then smooth via sin/cos
    yaw_filled = interp_short_gaps_circ(data['head_yaw_deg'][:n].astype(float))
    yaw        = _smooth_circ_deg(yaw_filled, sig)
    # Break the line at each 0°/360° crossing so matplotlib doesn't draw a
    # straight segment from e.g. 359° to 1° through 180°.
    wraps = np.where(np.abs(np.diff(yaw)) > 180)[0]
    yaw[wraps]     = np.nan
    yaw[wraps + 1] = np.nan
    # Snap the last point before and the first point after each break to exactly
    # 0° or 360° so segments cleanly terminate at the axis boundary.
    n_yaw = len(yaw)
    for w in wraps:
        if w > 0 and not np.isnan(yaw[w - 1]):
            yaw[w - 1] = 360.0 if yaw[w - 1] > 180 else 0.0
        if w + 2 < n_yaw and not np.isnan(yaw[w + 2]):
            yaw[w + 2] = 360.0 if yaw[w + 2] > 180 else 0.0
    return pitch, roll, yaw


def make_r01_figure(rec_dir=DEFAULT_REC_DIR, prefix=DEFAULT_PREFIX, out_path=None):

    h5_path  = os.path.join(rec_dir, f'{prefix}_preproc.h5')
    eye_path = os.path.join(rec_dir, f'{prefix}_eyecam_deinter.avi')
    top_path = os.path.join(rec_dir, 'fm1_0001.mp4')
    tif_path = os.path.join(rec_dir, 'file_00001.tif')

    if out_path is None:
        out_path = os.path.join(rec_dir, f'{prefix}_R01_fig1.svg')

    print('Loading preproc data ...')
    data  = _load_data(h5_path)
    block = _select_best_light_block(data)
    lo    = block['twop_lo']
    nd    = block['twop_nd']
    t0    = block['t_start']

    twopT = data['twopT']
    t_rel = twopT - t0   # seconds from block start

    # cap to 1-minute window
    nd_win = int(np.searchsorted(twopT, t0 + TRACE_WIN_S))
    nd_win = min(nd_win, nd, data['norm_dFF'].shape[1])

    print('Selecting cells ...')
    top_cells = _select_top_kurtosis_cells(data['norm_dFF'], lo, nd_win)

 
    mid_twop     = (lo + nd_win) // 2
    t_mid        = float(twopT[mid_twop])
    eye_trim_mid = int(np.searchsorted(data['eyeT_trim'], t_mid))
    eye_full_mid = data['eyeT_startInd'] + eye_trim_mid + 20

    print('Reading topdown frame ...')
    top_frame = _read_topdown_frame(top_path, mid_twop)

    print('Reading eye frame ...')
    eye_frame = _read_eye_frame(eye_path, eye_full_mid)

    print('Computing 2-photon mean image ...')
    if os.path.exists(tif_path):
        mip = _compute_mean_img(tif_path, lo, nd_win)
    else:
        print(f'  tif not found, using mean image from h5')
        mip = data['twop_mean_img'].astype(np.float32)


    pitch, roll, yaw = _smooth_imu(data)
    win  = (t_rel >= 0) & (t_rel <= TRACE_WIN_S)
    t_imu = t_rel[win]

    dff_mat = data['norm_dFF'][top_cells, lo:nd_win].astype(float)
    n_t     = dff_mat.shape[1]
    t_dff   = t_rel[lo: lo + n_t]
    dff_lim = t_dff <= TRACE_WIN_S
    t_dff   = t_dff[dff_lim]
    dff_mat = dff_mat[:, dff_lim]

 
    print('Building figure ...')

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    gs  = GridSpec(
        3, 2, figure=fig,
        height_ratios=[1.5, 1.0, 2.0],
        width_ratios=[1.0, 1.2],
        hspace=0.18, wspace=0.25,
        left=0.06, right=0.97, top=0.96, bottom=0.07,
    )

    ax_td  = fig.add_subplot(gs[0, 0])   # topdown frame — top-left
    ax_2p  = fig.add_subplot(gs[0, 1])   # 2P MIP — top-right (larger, swapped)
    ax_eye = fig.add_subplot(gs[1, 0])   # eye-camera — middle-left
    ax_dff = fig.add_subplot(gs[2, :])   # dF/F — bottom full width

    # stacked IMU axes now in middle-right (swapped from top-right)
    gs_imu   = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 1], hspace=0.10)
    ax_pitch = fig.add_subplot(gs_imu[0])
    ax_roll  = fig.add_subplot(gs_imu[1])
    ax_yaw   = fig.add_subplot(gs_imu[2])

    top_frame_width = np.size(top_frame, 1)
    top_frame_height = np.size(top_frame, 0)
    cropval = (top_frame_width - top_frame_height) // 2
    top_frame_to_show = top_frame[:, cropval:-cropval]
    lo_val = float(np.percentile(top_frame_to_show, 1))
    hi_val = float(np.percentile(top_frame_to_show, 99))
    if hi_val > lo_val:
        top_frame_to_show = (top_frame_to_show - lo_val) / (hi_val - lo_val) * 255.0
    top_frame_to_show = np.clip(top_frame_to_show, 0, 255).astype(np.uint8)
    ax_td.imshow(top_frame_to_show, aspect='equal', interpolation='nearest', vmin=0, vmax=255)
    ax_td.axis('off')
    ax_td.set_title('Top-down camera', fontsize=9, pad=4, loc='left')

    ax_eye.imshow(eye_frame, cmap='gray', vmin=0, vmax=255,
                  aspect='equal', interpolation='nearest')
    ax_eye.axis('off')
    ax_eye.set_title('Eye camera', fontsize=9, pad=4, loc='left')
    _p = ax_eye.get_position()
    ax_eye.set_position([_p.x0 + 0.03, _p.y0 - 0.025, _p.width, _p.height])

    vlo = float(np.percentile(mip, 1))
    vhi = float(np.percentile(mip, 99.5)) * 1.05
    ax_2p.imshow(mip, cmap='gray', vmin=vlo, vmax=vhi,
                 aspect='equal', interpolation='nearest')
    ax_2p.axis('off')
    ax_2p.set_title('2-photon FOV', fontsize=9, pad=4, loc='left')

    imu_specs = [
        (ax_pitch, pitch[win], '#1f77b4', 'pitch (°)'),
        (ax_roll,  roll[win],  '#2ca02c', 'roll (°)'),
        (ax_yaw,   yaw[win],   '#d62728', 'yaw (°)'),
    ]
    for ax, sig, col, lbl in imu_specs:
        ax.plot(t_imu, sig, color=col, lw=0.9)
        ax.set_ylabel(lbl, fontsize=7, labelpad=2)
        ax.set_xlim(0, TRACE_WIN_S)
        ax.tick_params(labelsize=6, length=3)
        finite = sig[np.isfinite(sig)]
        if len(finite) > 0:
            p2, p98 = np.nanpercentile(finite, [2, 98])
            mg = max(0.05 * abs(p98 - p2), 0.5)
            # ax.set_ylim(p2 - mg, p98 + mg)

    ax_pitch.set_ylim([-40,40])
    ax_roll.set_ylim([-30,30])
    ax_yaw.set_ylim([0,360])
    ax_yaw.set_yticks([0,180,360])

    ax_pitch.tick_params(labelbottom=False)
    ax_roll.tick_params(labelbottom=False)
    ax_yaw.set_xlabel('time (sec)', fontsize=7)
    ax_pitch.set_title('Head kinematics', fontsize=9, pad=4, loc='left')

    _draw_dff_panel(ax_dff, dff_mat, t_dff)


    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {out_path}')



DEFAULT_REC_DIR_AXON = (
    '/home/dylan/Storage/freely_moving_data/LP/'
    '250514_DMM_DMM046_LPaxons/fm1'
)
DEFAULT_PREFIX_AXON = '250514_DMM_DMM046_fm_1'


def _load_data_axon(h5_path):
    """Load axon preproc h5, tolerating recording-to-recording key variations.

    Variations handled:
    - eyeT_trim absent → reconstructed from eyeT[startInd:endInd]
    - light_onsets / dark_onsets absent → set to None (single-block recording)
    - twop_mean_img absent → caller falls back to FOV_mean.npy
    - raw_dFF may be 2-D (n,t) or 3-D (1,n,t); squeeze either way
    """
    data = {}
    with h5py.File(h5_path, 'r') as f:
        data['twopT']        = f['twopT'][:]
        data['theta']        = f['theta'][:]
        data['phi']          = f['phi'][:]
        data['head_x']       = f['head_x'][:]
        data['head_y']       = f['head_y'][:]
        data['eyeT_startInd'] = int(f['eyeT_startInd'][()])

        # eyeT_trim: use stored array or reconstruct from full eyeT
        if 'eyeT_trim' in f:
            data['eyeT_trim'] = f['eyeT_trim'][:]
        else:
            end_ind = int(f['eyeT_endInd'][()]) if 'eyeT_endInd' in f else len(f['eyeT'][:])
            data['eyeT_trim'] = f['eyeT'][:][data['eyeT_startInd']:end_ind]

        # light / dark onsets — absent in continuous (non-structured) recordings
        data['light_onsets'] = f['light_onsets'][:] if 'light_onsets' in f else None
        data['dark_onsets']  = f['dark_onsets'][:]  if 'dark_onsets'  in f else None

        # IMU — may be absent in some axon recordings
        for imu_key in ('head_yaw_deg', 'pitch_twop_interp', 'roll_twop_interp'):
            data[imu_key] = f[imu_key][:] if imu_key in f else None

        # dF/F: prefer denoised_dFF (better SNR), fall back to raw_dFF
        # raw_dFF may be 3-D (1, n_groups, n_frames) — squeeze to 2-D
        if 'denoised_dFF' in f:
            dff = np.squeeze(f['denoised_dFF'][:])
            data['norm_dFF'] = dff if dff.ndim == 2 else None
        elif 'raw_dFF' in f:
            dff = np.squeeze(f['raw_dFF'][:])
            data['norm_dFF'] = dff if dff.ndim == 2 else None
        else:
            data['norm_dFF'] = None

        data['twop_mean_img'] = f['twop_mean_img'][:] if 'twop_mean_img' in f else None
    return data


def make_r01_figure_axon(
    rec_dir=DEFAULT_REC_DIR_AXON,
    prefix=DEFAULT_PREFIX_AXON,
    out_path=None,
    som_rec_dir=DEFAULT_REC_DIR,
    som_prefix=DEFAULT_PREFIX,
):
    """Fig 2: same layout as make_r01_figure() but using axon recording data.

    If the axon h5 lacks IMU signals, a segment of the somatic recording
    (som_rec_dir / som_prefix) is used as a stand-in for the IMU panel.
    """
    h5_path  = os.path.join(rec_dir, f'{prefix}_preproc.h5')
    eye_path = os.path.join(rec_dir, f'{prefix}_eyecam_deinter.avi')

    # topdown video name varies across recordings
    _top_candidates = (
        glob.glob(os.path.join(rec_dir, 'fm01_0001.mp4')) or
        glob.glob(os.path.join(rec_dir, 'fm1_0001.mp4'))
    )
    top_path = _top_candidates[0] if _top_candidates else ''

    # axon tifs: prefer denoised_registered, fall back to any file_*.tif
    _tif_candidates = (
        sorted(glob.glob(os.path.join(rec_dir, 'file_*_denoised_registered.tif'))) or
        sorted(glob.glob(os.path.join(rec_dir, 'file_*.tif')))
    )
    tif_path     = _tif_candidates[0] if _tif_candidates else None
    fov_mean_npy = os.path.join(rec_dir, 'FOV_mean.npy')

    if out_path is None:
        out_path = os.path.join(rec_dir, f'{prefix}_R01_fig2.pdf')

    print('Loading axon preproc data ...')
    data  = _load_data_axon(h5_path)

    twopT = data['twopT']

    # block selection: use light/dark onsets when present, else whole recording
    if data['light_onsets'] is not None and data['dark_onsets'] is not None:
        block = _select_best_light_block(data)
        lo    = block['twop_lo']
        nd    = block['twop_nd']
        t0    = block['t_start']
    else:
        lo = 0
        nd = len(twopT) - 1
        t0 = float(twopT[0])
        print(f'  No light/dark onsets — using full recording [{lo}:{nd}]  t=[{t0:.1f}s]')

    t_rel = twopT - t0

    nd_win = int(np.searchsorted(twopT, t0 + TRACE_WIN_S))
    nd_win = min(nd_win, nd, (data['norm_dFF'].shape[1]
                               if data['norm_dFF'] is not None else nd_win))

    have_dff = data['norm_dFF'] is not None
    if have_dff:
        print('Selecting cells ...')
        top_cells = _select_top_kurtosis_cells(data['norm_dFF'], lo, nd_win)
        dff_mat   = data['norm_dFF'][top_cells, lo:nd_win].astype(float)
        n_t       = dff_mat.shape[1]
        t_dff     = t_rel[lo: lo + n_t]
        dff_lim   = t_dff <= TRACE_WIN_S
        t_dff     = t_dff[dff_lim]
        dff_mat   = dff_mat[:, dff_lim]
    else:
        print('  norm_dFF not found in axon h5 — dF/F panel will be blank')

    mid_twop     = (lo + nd_win) // 2
    t_mid        = float(twopT[mid_twop])
    eye_trim_mid = int(np.searchsorted(data['eyeT_trim'], t_mid))
    eye_full_mid = data['eyeT_startInd'] + eye_trim_mid

    print('Reading topdown frame ...')
    top_frame = _read_topdown_frame(top_path, mid_twop)

    print('Reading eye frame ...')
    eye_frame = _read_eye_frame(eye_path, eye_full_mid,
                                r0=EYE_CROP_AXON_R0, r1=EYE_CROP_AXON_R1,
                                c0=EYE_CROP_AXON_C0, c1=EYE_CROP_AXON_C1)

    print('Computing 2-photon mean image ...')
    if tif_path and os.path.exists(tif_path):
        print(f'  using tif: {os.path.basename(tif_path)}')
        mip = _compute_mean_img(tif_path, lo, nd_win)
    elif data['twop_mean_img'] is not None:
        mip = data['twop_mean_img'].astype(np.float32)
    elif os.path.exists(fov_mean_npy):
        print(f'  using FOV_mean.npy')
        mip = np.load(fov_mean_npy).astype(np.float32)
    else:
        mip = np.zeros((512, 512), dtype=np.float32)

    have_imu = all(data[k] is not None
                   for k in ('head_yaw_deg', 'pitch_twop_interp', 'roll_twop_interp'))

    if have_imu:
        print('Using axon IMU data ...')
        pitch, roll, yaw = _smooth_imu(data)
        win   = (t_rel >= 0) & (t_rel <= TRACE_WIN_S)
        t_imu = t_rel[win]
    else:
        print('No axon IMU — loading somatic recording for IMU panel ...')
        som_h5 = os.path.join(som_rec_dir, f'{som_prefix}_preproc.h5')
        som    = _load_data(som_h5)
        s_block = _select_best_light_block(som)
        s_t0    = s_block['t_start']
        s_trel  = som['twopT'] - s_t0
        pitch, roll, yaw = _smooth_imu(som)
        win   = (s_trel >= 0) & (s_trel <= TRACE_WIN_S)
        t_imu = s_trel[win]


    print('Building figure ...')
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    gs  = GridSpec(
        3, 2, figure=fig,
        height_ratios=[1.5, 1.0, 2.0],
        width_ratios=[1.0, 1.2],
        hspace=0.18, wspace=0.25,
        left=0.06, right=0.97, top=0.96, bottom=0.07,
    )

    ax_td  = fig.add_subplot(gs[0, 0])   # topdown frame — top-left
    ax_2p  = fig.add_subplot(gs[0, 1])   # 2P MIP — top-right (larger, swapped)
    ax_eye = fig.add_subplot(gs[1, 0])   # eye-camera — middle-left
    ax_dff = fig.add_subplot(gs[2, :])   # dF/F — bottom full width

    # stacked IMU axes now in middle-right (swapped from top-right)
    gs_imu   = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 1], hspace=0.10)
    ax_pitch = fig.add_subplot(gs_imu[0])
    ax_roll  = fig.add_subplot(gs_imu[1])
    ax_yaw   = fig.add_subplot(gs_imu[2])

    # topdown
    top_w = np.size(top_frame, 1)
    top_h = np.size(top_frame, 0)
    crop  = (top_w - top_h) // 2
    top_show = top_frame[:, crop: crop + top_h]
    lo_val = float(np.percentile(top_show, 1))
    hi_val = float(np.percentile(top_show, 99))
    if hi_val > lo_val:
        top_show = (top_show - lo_val) / (hi_val - lo_val) * 255.0
    ax_td.imshow(np.clip(top_show, 0, 255).astype(np.uint8),
                 aspect='equal', interpolation='nearest', vmin=0, vmax=255)
    ax_td.axis('off')
    ax_td.set_title('Top-down camera', fontsize=9, pad=4, loc='left')

    # eye camera
    ax_eye.imshow(eye_frame, cmap='gray', vmin=0, vmax=255,
                  aspect='equal', interpolation='nearest')
    ax_eye.axis('off')
    ax_eye.set_title('Eye camera', fontsize=9, pad=4, loc='left')
    _p = ax_eye.get_position()
    ax_eye.set_position([_p.x0 + 0.03, _p.y0 - 0.025, _p.width, _p.height])

    # 2P MIP
    vlo = float(np.percentile(mip, 1))
    vhi = float(np.percentile(mip, 99.5))
    if vhi <= vlo:
        vhi = vlo + 1.0
    ax_2p.imshow(mip, cmap='gray', vmin=vlo, vmax=vhi * 1.05,
                 aspect='equal', interpolation='nearest')
    ax_2p.axis('off')
    ax_2p.set_title('2-photon FOV (axons)', fontsize=9, pad=4, loc='left')

    # IMU
    imu_label = 'Head kinematics' if have_imu else 'Head kinematics (somatic)'
    imu_specs = [
        (ax_pitch, pitch[win], '#1f77b4', 'pitch (°)'),
        (ax_roll,  roll[win],  '#2ca02c', 'roll (°)'),
        (ax_yaw,   yaw[win],   '#d62728', 'yaw (°)'),
    ]
    for ax, sig, col, lbl in imu_specs:
        ax.plot(t_imu, sig, color=col, lw=0.9)
        ax.set_ylabel(lbl, fontsize=7, labelpad=2)
        ax.set_xlim(0, TRACE_WIN_S)
        ax.tick_params(labelsize=6, length=3)

    ax_pitch.set_ylim([-40, 40])
    ax_roll.set_ylim([-30, 30])
    ax_yaw.set_ylim([0, 360])
    ax_yaw.set_yticks([0, 180, 360])
    ax_pitch.tick_params(labelbottom=False)
    ax_roll.tick_params(labelbottom=False)
    ax_yaw.set_xlabel('time (sec)', fontsize=7)
    ax_pitch.set_title(imu_label, fontsize=9, pad=4, loc='left')

    if have_dff:
        _draw_dff_panel(ax_dff, dff_mat, t_dff, title='ΔF/F (axons)', smooth_tau=12)
    else:
        ax_dff.text(0.5, 0.5, 'dF/F not available in axon h5',
                    transform=ax_dff.transAxes, ha='center', va='center',
                    fontsize=8, color='0.5')
        ax_dff.set_xlim(0, TRACE_WIN_S)
        ax_dff.set_yticks([])
        ax_dff.spines['left'].set_visible(False)
        ax_dff.spines['bottom'].set_visible(False)
        ax_dff.tick_params(bottom=False, labelbottom=False)
        ax_dff.set_title('ΔF/F (axons)', fontsize=9, pad=4, loc='left')

    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {out_path}')


def main():
    parser = argparse.ArgumentParser(description='R01 static diagnostic figure.')
    parser.add_argument('--rec_dir', default=DEFAULT_REC_DIR)
    parser.add_argument('--prefix',  default=DEFAULT_PREFIX)
    parser.add_argument('--fig',     default='1', choices=['1', '2'],
                        help='1 = somatic (default), 2 = axon')
    parser.add_argument('--axon_rec_dir', default=DEFAULT_REC_DIR_AXON)
    parser.add_argument('--axon_prefix',  default=DEFAULT_PREFIX_AXON)
    parser.add_argument('--out',     default=None,
                        help='Output path (.pdf or .png).')
    args = parser.parse_args()
    
    outpath = args.out
    if outpath is None:
        outpath = './R01_fig{}.svg'.format(args.fig)

    
    if args.fig == '2':
        make_r01_figure_axon(
            rec_dir=args.axon_rec_dir, prefix=args.axon_prefix,
            out_path=args.out,
            som_rec_dir=args.rec_dir, som_prefix=args.prefix,
        )
    else:
        make_r01_figure(rec_dir=args.rec_dir, prefix=args.prefix, out_path=args.out)


if __name__ == '__main__':
    main()
