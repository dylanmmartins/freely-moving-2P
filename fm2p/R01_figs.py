
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
N_CELLS      = 80
TRACE_WIN_S  = 120.0    # 1-minute window for dF/F and IMU panels
DFF_OFFSET   = 0.65    # y-spacing between traces — < 1 gives overlap
MIP_STRIDE   = 4       # subsample every Nth tif frame for MIP (speed)
EYE_CROP_R0  = int(480 * 0.10)
EYE_CROP_R1  = int(480 * 0.80)
EYE_CROP_C0  = 0
EYE_CROP_C1  = int(640 * 0.80)

# axon eye camera: pupil sits lower-right — crop against bottom-right of frame
EYE_CROP_AXON_R0 = int(480 * 0.10)
EYE_CROP_AXON_R1 = 480
EYE_CROP_AXON_C0 = int(640 * 0.10)
EYE_CROP_AXON_C1 = int(640 * 0.8)

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
    so raw > smoothed during onset -> raw value kept; on the decay
    smoothed > raw -> smoothed value used (pointwise max).
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
    - eyeT_trim absent -> reconstructed from eyeT[startInd:endInd]
    - light_onsets / dark_onsets absent -> set to None (single-block recording)
    - twop_mean_img absent -> caller falls back to FOV_mean.npy
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
        out_path = os.path.join(rec_dir, f'{prefix}_R01_fig2.svg')

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
    eye_trim_mid = int(np.searchsorted(data['eyeT_trim'], t_mid)) + 1000
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


def figure_3(rec_dir=DEFAULT_REC_DIR, prefix=DEFAULT_PREFIX, out_path=None):

    from fm2p.decoding_video import (
        load_data, get_all_light_blocks, decode,
        _smooth_trace, _neural_features, _make_decoder, _rotate_pts,
        eye_frame_indices,
        TRACE_WIN_S, EYE_W, EYE_H,
    )
    from fm2p.decoding_video import interp_short_gaps as _dv_interp
    from matplotlib.patches import Ellipse as _Ellipse, Polygon as _Polygon, Circle as _Circle
    from matplotlib.lines import Line2D as _Line2D

    CHOOSE_BEST_FRAME = True   # True -> cursor/shapes at lowest-error frame; False -> window centre

    GT_COLOR    = '#444444'
    PRED_COLOR  = 'tab:red'
    SHAPE_FACE  = '0.78'
    SHAPE_EDGE  = '0.40'
    FS2         = 7
    LLEN        = 1.8

    h5_path = os.path.join(rec_dir, f'{prefix}_preproc.h5')
    if out_path is None:
        out_path = os.path.join(rec_dir, f'{prefix}_R01_fig3.svg')

    print('Figure 3 — loading data ...')
    data = load_data(h5_path)

    with h5py.File(h5_path, 'r') as _f:
        n_twop     = len(data['twopT'])
        pitch_full = (_f['pitch_twop_interp'][()].astype(float)
                      if 'pitch_twop_interp' in _f else np.full(n_twop, np.nan))
        roll_full  = (_f['roll_twop_interp'][()].astype(float)
                      if 'roll_twop_interp'  in _f else np.full(n_twop, np.nan))
        if 'head_yaw_deg' in _f:
            yaw_full = _f['head_yaw_deg'][()].astype(float)
        elif 'upsampled_yaw' in _f:
            yaw_full = _f['upsampled_yaw'][()].astype(float)
        else:
            yaw_full = np.full(n_twop, np.nan)

    blocks, best_idx = get_all_light_blocks(data)
    if best_idx < 0:
        raise RuntimeError('No valid light block found.')
    b            = blocks[best_idx]
    lo, nd       = b['lo'], b['nd']
    train_blocks = [blocks[i] for i in range(len(blocks)) if i != best_idx]

    print('Figure 3 — decoding eye ...')
    dec = decode(data, lo, nd, train_blocks, smooth_features=False)

    print('Figure 3 — decoding head ...')
    neural_feat = _neural_features(data['neural'][:, lo:nd].T.astype(float), smooth=False)
    gt_pitch    = pitch_full[lo:nd]
    gt_roll     = roll_full[lo:nd]
    gt_yaw      = yaw_full[lo:nd]
    n_feat      = neural_feat.shape[1]

    def _build_train(signal_full):
        Xp, yp = [], []
        for tb in train_blocks:
            t_lo, t_nd = tb['lo'], tb['nd']
            nt  = _neural_features(data['neural'][:, t_lo:t_nd].T.astype(float), smooth=False)
            sig = signal_full[t_lo:t_nd]
            v   = np.isfinite(sig) & np.isfinite(nt).all(axis=1)
            if v.sum() > 0:
                Xp.append(nt[v]); yp.append(sig[v])
        return (np.concatenate(Xp), np.concatenate(yp)) if Xp else (None, None)

    vpr = (np.isfinite(gt_pitch) & np.isfinite(gt_roll)
           & np.isfinite(neural_feat).all(axis=1))
    if vpr.sum() > 1:
        Xtp, ytp = _build_train(pitch_full)
        Xtr, ytr = _build_train(roll_full)
        if Xtp is not None:
            pred_pitch = _make_decoder(n_feat).fit(Xtp, ytp).predict(neural_feat)
            pred_roll  = _make_decoder(n_feat).fit(Xtr, ytr).predict(neural_feat)
        else:
            pred_pitch = _make_decoder(n_feat).fit(
                neural_feat[vpr], gt_pitch[vpr]).predict(neural_feat)
            pred_roll  = _make_decoder(n_feat).fit(
                neural_feat[vpr], gt_roll[vpr]).predict(neural_feat)
    else:
        pred_pitch = np.full(nd - lo, np.nan)
        pred_roll  = np.full(nd - lo, np.nan)

    vy = np.isfinite(gt_yaw) & np.isfinite(neural_feat).all(axis=1)
    if vy.sum() > 1:
        Xts, yts = _build_train(np.sin(np.radians(yaw_full)))
        Xtc, ytc = _build_train(np.cos(np.radians(yaw_full)))
        if Xts is not None:
            ps = _make_decoder(n_feat).fit(Xts, yts)
            pc = _make_decoder(n_feat).fit(Xtc, ytc)
        else:
            yaw_rad = np.radians(gt_yaw[vy])
            ps = _make_decoder(n_feat).fit(neural_feat[vy], np.sin(yaw_rad))
            pc = _make_decoder(n_feat).fit(neural_feat[vy], np.cos(yaw_rad))
        pred_yaw = np.degrees(np.arctan2(
            ps.predict(neural_feat), pc.predict(neural_feat)))
    else:
        pred_yaw = np.full(nd - lo, np.nan)

    twopT  = data['twopT']
    tt     = twopT[lo:nd] - b['t_start']
    fps_2p = float(1.0 / np.nanmedian(np.diff(twopT)))
    _WIN_S = 10.0
    win_n  = max(10, int(_WIN_S * fps_2p))

    gt_t    = _dv_interp(dec['gt_theta'])
    pred_t  = np.array(dec['pred_theta'], dtype=float)
    gt_p    = _dv_interp(dec['gt_phi'])
    pred_p  = np.array(dec['pred_phi'], dtype=float)
    gt_pi   = _dv_interp(gt_pitch)
    pred_pi = np.array(pred_pitch, dtype=float)
    gt_ro   = _dv_interp(gt_roll)
    pred_ro = np.array(pred_roll, dtype=float)
    gt_ya   = _dv_interp(gt_yaw % 360)
    pred_ya = np.array(pred_yaw % 360, dtype=float)

    # Scan all candidate windows. Score = sum of |gt_theta - pred_theta| over observed
    # frames. Discard windows where more than 20% of raw theta frames are NaN.
    raw_theta = np.array(dec['gt_theta'], dtype=float)
    _NAN_MAX  = 0.20   # up to 20% NaN allowed per window

    n = len(gt_t)
    all_windows = []   # (sum_err, center_idx) for windows passing the NaN gate
    stride = max(1, win_n // 20)
    for i in range(win_n // 2, n - win_n // 2, stride):
        sl       = slice(i - win_n // 2, i + win_n // 2)
        raw_sl   = raw_theta[sl]
        nan_frac = float(np.mean(~np.isfinite(raw_sl)))
        if nan_frac > _NAN_MAX:
            continue
        valid = np.isfinite(raw_sl) & np.isfinite(pred_t[sl])
        if valid.sum() < 5:
            continue
        sum_err = float(np.sum(np.abs(raw_sl[valid] - pred_t[sl][valid])))
        all_windows.append((sum_err, i))

    # Top 20 by lowest residual sum; fall back to midpoint if nothing qualifies
    all_windows.sort(key=lambda x: x[0])
    top_windows = all_windows[:20]
    if top_windows:
        chosen_centers = [idx for _, idx in top_windows]
        print(f'  Top {len(chosen_centers)} windows by theta residual sum '
              f'(from {len(all_windows)} with ≤{_NAN_MAX:.0%} NaN); '
              f'best sum={top_windows[0][0]:.2f}°, worst={top_windows[-1][0]:.2f}°')
    else:
        chosen_centers = [n // 2]
        print('  No window passed the NaN gate; using midpoint')

    # Precompute ellipse arrays (same for every window, the decode block is fixed)
    gt_X  = _dv_interp(dec['gt_X0']);      gt_Y  = _dv_interp(dec['gt_Y0'])
    la    = _dv_interp(dec['gt_longaxis']); sa    = _dv_interp(dec['gt_shortaxis'])
    ephi  = _dv_interp(dec['gt_ellipse_phi'])
    px    = _dv_interp(dec['pred_X0']);    py    = _dv_interp(dec['pred_Y0'])

    x_fin  = gt_X[np.isfinite(gt_X)];  y_fin = gt_Y[np.isfinite(gt_Y)]
    la_med = float(np.nanmedian(la[np.isfinite(la)])) if np.any(np.isfinite(la)) else 30.0
    if len(x_fin) > 0:
        x_ctr   = float(np.nanmedian(x_fin));  y_ctr = float(np.nanmedian(y_fin))
        half_px = max(np.nanstd(x_fin) * 4, np.nanstd(y_fin) * 4, la_med * 4, 40.0)
    else:
        x_ctr, y_ctr, half_px = EYE_W / 2, EYE_H / 2, 95.0

    _eyecam_path = os.path.join(rec_dir, f'{prefix}_eyecam_deinter.avi')
    _vid_indices = eye_frame_indices(data, lo, nd)

    _ear_L = np.array([[-0.58, 0.58], [-0.12, 0.58], [-0.35, 1.12]])
    _ear_R = np.array([[ 0.12, 0.58], [ 0.58, 0.58], [ 0.35, 1.12]])
    _tri_pts = np.array([[-0.8, -0.45], [-0.8, 0.45], [1.6, 0.0]])

    # Base path for multi-version output
    if out_path is None:
        _base_path = os.path.join(rec_dir, f'{prefix}_R01_fig3.svg')
    else:
        _base_path = out_path

    for v_idx, center_idx in enumerate(chosen_centers):

        win_sl      = slice(center_idx - win_n // 2, center_idx + win_n // 2)
        win_start_t = float(tt[win_sl.start])

        if CHOOSE_BEST_FRAME:
            nw  = win_sl.stop - win_sl.start
            err = np.zeros(nw, dtype=float)
            for _sg, _sp in [(gt_t, pred_t), (gt_p, pred_p)]:
                _rng = max(float(np.nanpercentile(_sg[win_sl], 95) -
                                 np.nanpercentile(_sg[win_sl],  5)), 1e-6)
                err += np.abs(_sg[win_sl] - _sp[win_sl]) / _rng
            err[~np.isfinite(err)] = np.inf
            fi = win_sl.start + int(np.argmin(err))
        else:
            fi = center_idx

        t_display     = tt - win_start_t
        cursor_disp_t = float(tt[fi]) - win_start_t
        center_t      = float(tt[center_idx])
        print(f'  v{v_idx:02d}: window t={center_t:.1f} s  cursor={cursor_disp_t:.1f} s into window')

        fig = plt.figure(figsize=(14, 5.5), dpi=DPI, facecolor='w')
        gs  = GridSpec(2, 20, figure=fig,
                       height_ratios=[1, 1.5],
                       hspace=0.55, wspace=3.0,
                       left=0.05, right=0.98, top=0.93, bottom=0.10)

        ax_theta  = fig.add_subplot(gs[0,  0: 4])
        ax_phi    = fig.add_subplot(gs[0,  4: 8])
        ax_pitch  = fig.add_subplot(gs[0,  8:12])
        ax_roll   = fig.add_subplot(gs[0, 12:16])
        ax_yaw    = fig.add_subplot(gs[0, 16:20])
        ax_ell    = fig.add_subplot(gs[1,  0: 5])
        ax_pit_v  = fig.add_subplot(gs[1,  5:10])
        ax_rol_v  = fig.add_subplot(gs[1, 10:15])
        ax_yaw_v  = fig.add_subplot(gs[1, 15:20])

        for gt_s, pred_s, ylabel, ax in [
            (gt_t,  pred_t,  r'$\theta$ (°)', ax_theta),
            (gt_p,  pred_p,  r'$\phi$ (°)',   ax_phi),
            (gt_pi, pred_pi, 'pitch (°)',      ax_pitch),
            (gt_ro, pred_ro, 'roll (°)',       ax_roll),
            (gt_ya, pred_ya, 'yaw (°)',        ax_yaw),
        ]:
            ax.plot(t_display[win_sl], gt_s[win_sl],   color=GT_COLOR,   lw=1.3, alpha=0.9)
            ax.plot(t_display[win_sl], pred_s[win_sl], color=PRED_COLOR, lw=1.3, alpha=0.85)
            ax.axvline(cursor_disp_t, color='k', lw=1.0, ls='--', alpha=0.4)
            ax.set_xlim(0, _WIN_S)
            ax.set_xlabel('time (s)', fontsize=FS2)
            ax.set_ylabel(ylabel,     fontsize=FS2)
            ax.tick_params(labelsize=FS2 - 1)
            finite = gt_s[win_sl][np.isfinite(gt_s[win_sl])]
            if len(finite) > 2:
                lo_y, hi_y = np.nanpercentile(finite, [1, 99])
                mg = max(0.05 * abs(hi_y - lo_y), 0.005)
                ax.set_ylim(lo_y - mg, hi_y + mg)

        ax_theta.legend(
            handles=[_Line2D([0], [0], color=GT_COLOR,   lw=1.3, label='measured'),
                     _Line2D([0], [0], color=PRED_COLOR, lw=1.3, label='decoded')],
            fontsize=FS2 - 1, loc='upper left', frameon=False)

        ax_ell.set_aspect('equal')
        ax_ell.set_xlabel('X (px)', fontsize=FS2)
        ax_ell.set_ylabel('Y (px)', fontsize=FS2)
        ax_ell.tick_params(labelsize=FS2 - 1)
        ax_ell.set_xlim(x_ctr - half_px, x_ctr + half_px)
        ax_ell.set_ylim(y_ctr + half_px, y_ctr - half_px)

        _vid_fi = int(_vid_indices[fi])
        _cap = cv2.VideoCapture(_eyecam_path)
        _cap.set(cv2.CAP_PROP_POS_FRAMES, _vid_fi)
        _ret, _raw_frame = _cap.read()
        _cap.release()
        if _ret:
            _eye_gray = cv2.cvtColor(_raw_frame, cv2.COLOR_BGR2GRAY)
            _eye_gray = cv2.resize(_eye_gray, (EYE_W, EYE_H))
        else:
            _eye_gray = np.zeros((EYE_H, EYE_W), dtype=np.uint8)
        ax_ell.imshow(_eye_gray, cmap='gray', vmin=0, vmax=255,
                      extent=[0, EYE_W, EYE_H, 0], aspect='auto', zorder=0)

        ok_g = (np.isfinite(gt_X[fi] + gt_Y[fi] + la[fi] + sa[fi] + ephi[fi])
                and la[fi] > 0 and sa[fi] > 0)
        if ok_g:
            ax_ell.add_patch(_Ellipse(
                (gt_X[fi], gt_Y[fi]), 2*la[fi], 2*sa[fi],
                angle=float(np.degrees(ephi[fi])),
                fill=False, edgecolor=GT_COLOR, linewidth=2.0, zorder=3))
        if ok_g and np.isfinite(px[fi] + py[fi]):
            ax_ell.add_patch(_Ellipse(
                (px[fi], py[fi]), 2*la[fi], 2*sa[fi],
                angle=float(np.degrees(ephi[fi])),
                fill=False, edgecolor=PRED_COLOR, linewidth=2.0, zorder=4))

        true_pitch = float(gt_pi[fi])   if np.isfinite(gt_pi[fi])   else 0.0
        pred_pval  = float(pred_pi[fi]) if np.isfinite(pred_pi[fi]) else 0.0

        for _axv, _title, _true, _pred, _tri in [
            (ax_pit_v, 'pitch', true_pitch, pred_pval, _tri_pts),
            (ax_yaw_v, 'yaw',
             float(gt_ya[fi])   if np.isfinite(gt_ya[fi])   else 0.0,
             float(pred_ya[fi]) if np.isfinite(pred_ya[fi]) else 0.0,
             np.array([[-0.8, -0.45], [-0.8, 0.45], [1.6, 0.0]])),
        ]:
            _axv.set_aspect('equal'); _axv.axis('off')
            _axv.set_xlim(-2.0, 2.0); _axv.set_ylim(-2.0, 2.0)
            _axv.set_title(_title, fontsize=FS2, pad=3)
            _axv.add_patch(_Polygon(_rotate_pts(_tri, _true), closed=True,
                                    facecolor=SHAPE_FACE, edgecolor=SHAPE_EDGE,
                                    linewidth=1.2, zorder=3))
            a_p = np.radians(_pred)
            _axv.plot([0.0, LLEN*np.cos(a_p)], [0.0, LLEN*np.sin(a_p)],
                      color=PRED_COLOR, lw=2.0, zorder=4)
            a_g = np.radians(_true)
            _axv.plot([0.0, LLEN*np.cos(a_g)], [0.0, LLEN*np.sin(a_g)],
                      color=GT_COLOR,   lw=2.0, zorder=5)

        true_roll = float(gt_ro[fi])   if np.isfinite(gt_ro[fi])   else 0.0
        pred_rval = float(pred_ro[fi]) if np.isfinite(pred_ro[fi]) else 0.0

        ax_rol_v.set_aspect('equal'); ax_rol_v.axis('off')
        ax_rol_v.set_xlim(-2.0, 2.0); ax_rol_v.set_ylim(-2.0, 2.0)
        ax_rol_v.set_title('roll', fontsize=FS2, pad=3)
        ax_rol_v.add_patch(_Circle((0.0, 0.0), 0.72,
                                   facecolor=SHAPE_FACE, edgecolor=SHAPE_EDGE,
                                   linewidth=1.2, zorder=3))
        for _ear in (_ear_L, _ear_R):
            ax_rol_v.add_patch(_Polygon(_rotate_pts(_ear, true_roll), closed=True,
                                        facecolor=SHAPE_FACE, edgecolor=SHAPE_EDGE,
                                        linewidth=1.2, zorder=3))
        a_pr = np.radians(pred_rval)
        ax_rol_v.plot([-LLEN*np.cos(a_pr), LLEN*np.cos(a_pr)],
                      [-LLEN*np.sin(a_pr), LLEN*np.sin(a_pr)],
                      color=PRED_COLOR, lw=2.0, zorder=4)
        a_gr = np.radians(true_roll)
        ax_rol_v.plot([-LLEN*np.cos(a_gr), LLEN*np.cos(a_gr)],
                      [-LLEN*np.sin(a_gr), LLEN*np.sin(a_gr)],
                      color=GT_COLOR,   lw=2.0, zorder=5)

        fig.suptitle('Neural decoding of eye and head position', fontsize=9)

        if len(chosen_centers) == 1:
            save_path = _base_path
        else:
            _b, _e = os.path.splitext(_base_path)
            save_path = f'{_b}_v{v_idx:02d}{_e}'

        fig.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor='w')
        plt.close(fig)
        print(f'  Saved -> {save_path}')


DEFAULT_REC_DIR_FIG4 = (
    '/home/dylan/Fast1/ret2ego_reconstruction/251028_DMM_worldcam/'
    'fm4_251028_121027_776'
)
DEFAULT_PREFIX_FIG4 = '251028_DMM_DMM000_fm_04'


def figure_4(
    rec_dir=DEFAULT_REC_DIR_FIG4,
    prefix=DEFAULT_PREFIX_FIG4,
    out_path=None,
):

    from matplotlib.colors import LinearSegmentedColormap

    _RET_CMAP4  = LinearSegmentedColormap.from_list('retinal', ['#060e06', '#00ff55'])
    _MASK_COLOR = np.array([0, 255, 136], dtype=float)   # #00ff88 (annotate_worldcam fill)
    _MASK_ALPHA = 0.40
    FS          = 8

    h5_path   = os.path.join(rec_dir, f'{prefix}_preproc.h5')
    wc_path   = os.path.join(rec_dir, f'{prefix}_eyecam_deinter.avi')
    mask_path = os.path.join(rec_dir, f'{prefix}_eyecam_deinter_polygon_masks.npy')
    npz_path  = os.path.join(rec_dir, f'{prefix}_retinal_images.npz')
    top_cands = (glob.glob(os.path.join(rec_dir, 'fm*_0001.mp4')) or
                 glob.glob(os.path.join(rec_dir, 'fm*0001.mp4')))
    top_path  = top_cands[0] if top_cands else ''
    if out_path is None:
        out_path = os.path.join(rec_dir, f'{prefix}_R01_fig4.svg')

    print('Figure 4 — loading h5 ...')
    with h5py.File(h5_path, 'r') as f:
        twopT         = f['twopT'][:]
        eyeT_startInd = int(f['eyeT_startInd'][()])
        pitch_2p = (f['pitch_twop_interp'][:]
                    if 'pitch_twop_interp' in f else np.full(len(twopT), np.nan))
        roll_2p  = (f['roll_twop_interp'][:]
                    if 'roll_twop_interp'  in f else np.full(len(twopT), np.nan))
        yaw_2p   = f['head_yaw_deg'][:][:len(twopT)]

    pitch, roll, yaw = _smooth_imu({
        'twopT':              twopT,
        'pitch_twop_interp':  pitch_2p,
        'roll_twop_interp':   roll_2p,
        'head_yaw_deg':       yaw_2p,
    })

    print('Figure 4 — loading retinal npz ...')
    npz            = np.load(npz_path, allow_pickle=False)
    retinal_images = npz['retinal_images']   # (N_eye, 120, 120)
    eyeT           = npz['eyeT']             # same epoch as twopT

    print('Figure 4 — loading masks ...')
    masks = np.load(mask_path, allow_pickle=True)   # (N_vid, 480, 640) uint8 0/1

    best_eye_idx  = int(np.argmin(np.abs(eyeT - 21.0)))
    best_t        = float(eyeT[best_eye_idx])
    best_vid_idx  = eyeT_startInd + best_eye_idx
    best_twop_idx = int(np.clip(np.searchsorted(twopT, best_t), 0, len(twopT) - 1))
    print(f'  Frame at t=22.45s: eye_idx={best_eye_idx}  actual_t={best_t:.3f}s')

    print('Figure 4 — reading worldcam frame ...')
    cap = cv2.VideoCapture(wc_path)
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(best_vid_idx, n_vid - 1))
    ret_ok, wc_raw = cap.read()
    cap.release()
    if ret_ok:
        wc_gray = cv2.cvtColor(wc_raw, cv2.COLOR_BGR2GRAY)
    else:
        wc_gray = np.zeros((480, 640), dtype=np.uint8)
    wc_gray = _subtract_band(wc_gray, r0=0, r1=480, c0=0, c1=640)

    mask_frame = masks[min(best_vid_idx, len(masks) - 1)]   # (480, 640) uint8 0/1
    wc_rgb     = np.stack([wc_gray, wc_gray, wc_gray], axis=-1).astype(float)
    mbool      = mask_frame > 0
    wc_masked  = wc_rgb.copy()
    wc_masked[mbool] = np.clip(
        _MASK_ALPHA * _MASK_COLOR + (1.0 - _MASK_ALPHA) * wc_rgb[mbool],
        0, 255)
    wc_masked = wc_masked.astype(np.uint8)

    print('Figure 4 — reading topdown frame ...')
    top_frame = _read_topdown_frame(top_path, best_twop_idx)   # (480, 640, 3) RGB
    top_h, top_w = top_frame.shape[:2]
    cropval   = (top_w - top_h) // 2
    top_show  = top_frame[:, cropval: cropval + top_h]
    lo_v, hi_v = float(np.percentile(top_show, 1)), float(np.percentile(top_show, 99))
    if hi_v > lo_v:
        top_show = np.clip(
            (top_show.astype(float) - lo_v) / (hi_v - lo_v) * 255, 0, 255
        ).astype(np.uint8)

    print('Figure 4 — building figure ...')
    fig = plt.figure(figsize=(12, 5.5), dpi=DPI, facecolor='w')

    # Left 2×2 | right retinal
    gs_main = GridSpec(1, 2, figure=fig,
                       width_ratios=[1.35, 1.0],
                       left=0.05, right=0.97, top=0.94, bottom=0.08, wspace=0.28)

    gs_left  = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_main[0],
                                       hspace=0.35, wspace=0.22)
    ax_td    = fig.add_subplot(gs_left[0, 0])
    ax_wc    = fig.add_subplot(gs_left[1, 0])
    ax_mask_ = fig.add_subplot(gs_left[1, 1])

    gs_imu    = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_left[0, 1], hspace=0.12)
    ax_pitch_ = fig.add_subplot(gs_imu[0])
    ax_roll_  = fig.add_subplot(gs_imu[1])
    ax_yaw_   = fig.add_subplot(gs_imu[2])

    # Retinal panel centred at 2/3 figure height (height_ratios 1:4:1 -> middle = 4/6 ≈ 2/3)
    gs_right = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_main[1],
                                       height_ratios=[1, 4, 1], hspace=0)
    ax_ret   = fig.add_subplot(gs_right[1])

    # topdown
    ax_td.imshow(top_show, aspect='equal', interpolation='nearest')
    ax_td.axis('off')
    ax_td.set_title('Top-down camera', fontsize=FS + 1, pad=4, loc='left')

    # worldcam
    ax_wc.imshow(wc_gray, cmap='gray', vmin=0, vmax=255,
                 aspect='equal', interpolation='nearest')
    ax_wc.axis('off')
    ax_wc.set_title('Worldcam', fontsize=FS + 1, pad=4, loc='left')

    # mask overlay
    ax_mask_.imshow(wc_masked, aspect='equal', interpolation='nearest')
    ax_mask_.axis('off')
    ax_mask_.set_title('Field-of-view mask', fontsize=FS + 1, pad=4, loc='left')

    # IMU traces — full recording with cursor at chosen frame
    t_rel = twopT - twopT[0]
    t_end = float(t_rel[-1])
    t_cur = best_t - float(twopT[0])
    for ax, sig, col, lbl in [
        (ax_pitch_, pitch, '#1f77b4', 'pitch (°)'),
        (ax_roll_,  roll,  '#2ca02c', 'roll (°)'),
        (ax_yaw_,   yaw,   '#d62728', 'yaw (°)'),
    ]:
        ax.plot(t_rel, sig, color=col, lw=0.9)
        ax.axvline(t_cur, color='k', lw=0.8, ls='--', alpha=0.5)
        ax.set_ylabel(lbl, fontsize=FS - 1, labelpad=2)
        ax.set_xlim(0, t_end)
        ax.tick_params(labelsize=FS - 2, length=3)

    ax_pitch_.set_ylim([-40,  40])
    ax_roll_.set_ylim([-30,  30])
    ax_yaw_.set_ylim([  0, 360])
    ax_yaw_.set_yticks([0, 180, 360])
    ax_pitch_.tick_params(labelbottom=False)
    ax_roll_.tick_params(labelbottom=False)
    ax_yaw_.set_xlabel('time (s)', fontsize=FS - 1)
    ax_pitch_.set_title('Head kinematics', fontsize=FS + 1, pad=4, loc='left')

    # retinal reconstruction — crop to first quadrant (az 0–60°, elev 0–60°)
    # pixel layout: row 0 = elev +60°, row 119 = elev -60°; col 0 = az -60°, col 119 = az +60°
    ret_full  = retinal_images[best_eye_idx]        # (120, 120) uint8
    ret_frame = ret_full[0:60, 60:120]              # elev 0–60°, az 0–60°
    ax_ret.set_facecolor('#060e06')
    ax_ret.imshow(ret_frame,
                  cmap=_RET_CMAP4, vmin=0, vmax=255,
                  aspect='equal', interpolation='nearest',
                  extent=[0, 60, 0, 60])
    ax_ret.set_xlim(0, 60)
    ax_ret.set_ylim(0, 60)
    ax_ret.set_xlabel('Azimuth (°)', fontsize=FS)
    ax_ret.set_ylabel('Elevation (°)', fontsize=FS)
    ax_ret.tick_params(labelsize=FS - 1, colors='0.5')
    ax_ret.set_title('Estimated retinal image', fontsize=FS + 1, pad=4, loc='left')
    for sp in ax_ret.spines.values():
        sp.set_color('0.35')

    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor='w')
    plt.close(fig)
    print(f'Saved -> {out_path}')


def figure_5(
    rec_dir=DEFAULT_REC_DIR_FIG4,
    prefix=DEFAULT_PREFIX_FIG4,
    out_path=None,
):

    import subprocess as _sp
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D as _Line2D5

    USE_TRANSFORM_ERROR = True  # True -> best-fit x/y/θ shift distance; False -> direct mask–retinal corr

    _RET_CMAP5  = LinearSegmentedColormap.from_list('retinal', ['#060e06', '#00ff55'])
    _MASK_CMAP5 = LinearSegmentedColormap.from_list('mask',    ['#0a0a0a', '#00ff88'])
    _FIG_BG     = 'k'
    _TRC_BG     = '#0a0a0a'
    _STRIDE     = 2          # 60 fps eye-cam -> 30 fps video
    _VID_FPS    = 30
    _IMU_WIN_S  = 20.0       # scrolling window (±10 s) for both bottom panels
    _FIG_DPI    = 100
    _WC_W, _WC_H = 320, 240  # worldcam display resolution

    h5_path   = os.path.join(rec_dir, f'{prefix}_preproc.h5')
    wc_path   = os.path.join(rec_dir, f'{prefix}_eyecam_deinter.avi')
    mask_path = os.path.join(rec_dir, f'{prefix}_eyecam_deinter_polygon_masks.npy')
    npz_path  = os.path.join(rec_dir, f'{prefix}_retinal_images.npz')
    if out_path is None:
        out_path = os.path.join(rec_dir, f'{prefix}_R01_fig5.mp4')

    print('Figure 5 — loading h5 ...')
    with h5py.File(h5_path, 'r') as f:
        twopT         = f['twopT'][:]
        eyeT_startInd = int(f['eyeT_startInd'][()])
        eyeT_trim     = f['eyeT_trim'][:]
        pitch_2p = (f['pitch_twop_interp'][:]
                    if 'pitch_twop_interp' in f else np.full(len(twopT), np.nan))
        roll_2p  = (f['roll_twop_interp'][:]
                    if 'roll_twop_interp'  in f else np.full(len(twopT), np.nan))
        yaw_2p   = f['head_yaw_deg'][:][:len(twopT)]

    pitch_s, roll_s, yaw_s = _smooth_imu({
        'twopT': twopT, 'pitch_twop_interp': pitch_2p,
        'roll_twop_interp': roll_2p, 'head_yaw_deg': yaw_2p,
    })
    t_twop    = twopT    - twopT[0]
    t_eye_rel = eyeT_trim - eyeT_trim[0]

    # Interpolate IMU signals to eye-camera timebase
    pitch_eye = np.interp(t_eye_rel, t_twop, pitch_s)
    roll_eye  = np.interp(t_eye_rel, t_twop, roll_s)
    _fy = np.isfinite(yaw_s)
    yaw_eye = (np.interp(t_eye_rel, t_twop[_fy], yaw_s[_fy])
               if _fy.sum() > 1 else np.full(len(t_eye_rel), np.nan))

    print('Figure 5 — loading retinal npz ...')
    npz            = np.load(npz_path, allow_pickle=False)
    retinal_images = npz['retinal_images']   # (N_eye, 120, 120) uint8
    N_EYE          = len(npz['eyeT'])

    _metric_label = 'transform error (px)' if USE_TRANSFORM_ERROR else 'mask-retinal corr.'
    print(f'Figure 5 — precomputing masks and {_metric_label} ({N_EYE} frames) ...')
    masks      = np.load(mask_path, mmap_mode='r')   # (N_vid, 480, 640) lazy
    mask_disp  = np.zeros((N_EYE, 120, 120), dtype=np.uint8)
    mask_corrs = np.zeros(N_EYE, dtype=float)

    if USE_TRANSFORM_ERROR:
        from scipy.ndimage import rotate as _nd_rotate
        _ANGLES = np.arange(-20.0, 21.0, 5.0)   # rotation search grid (degrees)
        print(f'  transform search: {len(_ANGLES)} angles × {N_EYE} frames')

    for i in range(N_EYE):
        m    = masks[eyeT_startInd + i].astype(np.float32) * 255.0
        m_sm = cv2.resize(m, (120, 120), interpolation=cv2.INTER_AREA)
        mask_disp[i] = np.clip(m_sm, 0, 255).astype(np.uint8)

        m_c    = m_sm - m_sm.mean()
        m_norm = float(np.sqrt((m_c ** 2).sum()))
        r      = retinal_images[i].astype(float)

        if USE_TRANSFORM_ERROR:
            best_corr = -np.inf
            best_dx = 0.0
            best_dy = 0.0
            for theta in _ANGLES:
                rot   = (_nd_rotate(r, theta, reshape=False, order=1)
                         if theta != 0.0 else r)
                rot_c = rot - rot.mean()
                r_norm = float(np.sqrt((rot_c ** 2).sum()))
                if r_norm < 1e-9 or m_norm < 1e-9:
                    continue
                xcorr = np.fft.fftshift(np.real(
                    np.fft.ifft2(np.fft.fft2(m_c) * np.conj(np.fft.fft2(rot_c)))
                ))
                peak  = np.unravel_index(np.argmax(xcorr), xcorr.shape)
                pcorr = xcorr[peak] / (m_norm * r_norm)
                if pcorr > best_corr:
                    best_corr = pcorr
                    best_dy   = float(peak[0] - m_sm.shape[0] // 2)
                    best_dx   = float(peak[1] - m_sm.shape[1] // 2)
            mask_corrs[i] = float(np.sqrt(best_dx ** 2 + best_dy ** 2))
        else:
            r_c   = r.ravel() - r.mean()
            r_norm = float(np.sqrt((r_c ** 2).sum()))
            den    = m_norm * r_norm
            mask_corrs[i] = float(np.dot(m_c.ravel(), r_c) / den) if den > 0 else 0.0

        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{N_EYE}')
    del masks   # release mmap

    print('Figure 5 — preloading worldcam frames ...')
    out_eye_idxs = np.arange(0, N_EYE, _STRIDE)
    n_out        = len(out_eye_idxs)
    wc_frames    = np.zeros((n_out, _WC_H, _WC_W), dtype=np.uint8)

    cap = cv2.VideoCapture(wc_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, eyeT_startInd)
    for i in range(N_EYE):
        ok, frame = cap.read()
        if not ok:
            break
        if i % _STRIDE == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = _subtract_band(gray, r0=0, r1=480, c0=0, c1=640)
            wc_frames[i // _STRIDE] = cv2.resize(gray, (_WC_W, _WC_H))
    cap.release()
    print(f'  {n_out} output frames')

    print('Figure 5 — building figure ...')
    fig = plt.figure(figsize=(12, 5), dpi=_FIG_DPI, facecolor=_FIG_BG)
    gs  = GridSpec(2, 3, figure=fig,
                   height_ratios=[1.1, 1.0],
                   hspace=0.40, wspace=0.22,
                   left=0.08, right=0.96, top=0.92, bottom=0.12)

    ax_wc   = fig.add_subplot(gs[0, 0])
    ax_msk  = fig.add_subplot(gs[0, 1])
    ax_ret  = fig.add_subplot(gs[0, 2])
    ax_imu  = fig.add_subplot(gs[1, 0])
    ax_corr = fig.add_subplot(gs[1, 1:3])

    # image axes — no axes decorations, dark bg
    for ax in (ax_wc, ax_msk, ax_ret):
        ax.set_facecolor(_FIG_BG)
        ax.axis('off')
    ax_ret.set_facecolor('#060e06')

    # trace axes
    for ax in (ax_imu, ax_corr):
        ax.set_facecolor(_TRC_BG)
        ax.tick_params(colors='0.6', labelsize=7)
        for sp in ax.spines.values():
            sp.set_color('0.3')

    ax_wc.set_title('Worldcam',                color='0.7', fontsize=8, pad=2, loc='left')
    ax_msk.set_title('FOV mask',               color='0.7', fontsize=8, pad=2, loc='left')
    ax_ret.set_title('Retinal reconstruction', color='0.7', fontsize=8, pad=2, loc='left')

    im_wc  = ax_wc.imshow(np.zeros((_WC_H, _WC_W), dtype=np.uint8),
                           cmap='gray', vmin=0, vmax=255,
                           aspect='equal', interpolation='nearest')
    im_msk = ax_msk.imshow(np.zeros((120, 120), dtype=np.uint8),
                            cmap=_MASK_CMAP5, vmin=0, vmax=255,
                            aspect='equal', interpolation='nearest')
    im_ret = ax_ret.imshow(np.zeros((60, 60), dtype=np.uint8),
                            cmap=_RET_CMAP5, vmin=0, vmax=255,
                            aspect='equal', interpolation='nearest',
                            extent=[0, 60, 0, 60])

    # IMU: pitch + roll on left y-axis, yaw on right
    ax_yaw_r = ax_imu.twinx()
    ax_yaw_r.set_facecolor(_TRC_BG)
    ax_yaw_r.tick_params(colors='#ffaa44', labelsize=7)
    for sp in ax_yaw_r.spines.values():
        sp.set_color('0.3')

    ax_imu.plot(t_eye_rel, pitch_eye, color='#4a9eff', lw=1.0)
    ax_imu.plot(t_eye_rel, roll_eye,  color='#4aff88', lw=1.0)
    ax_yaw_r.plot(t_eye_rel, yaw_eye, color='#ffaa44', lw=1.0)

    _pr = np.concatenate([pitch_eye[np.isfinite(pitch_eye)],
                          roll_eye[np.isfinite(roll_eye)]])
    if len(_pr):
        _lo, _hi = np.nanpercentile(_pr, [1, 99])
        ax_imu.set_ylim(_lo - 5, _hi + 5)
    ax_yaw_r.set_ylim(0, 360)
    ax_yaw_r.set_yticks([0, 180, 360])

    ax_imu.set_ylabel('pitch / roll (°)', color='0.6', fontsize=7, labelpad=2)
    ax_yaw_r.set_ylabel('yaw (°)',         color='#ffaa44', fontsize=7, labelpad=2)
    ax_imu.set_xlabel('Time (s)',          color='0.6', fontsize=7)
    ax_imu.legend(
        handles=[_Line2D5([0],[0], color='#4a9eff', lw=1.2, label='pitch'),
                 _Line2D5([0],[0], color='#4aff88', lw=1.2, label='roll'),
                 _Line2D5([0],[0], color='#ffaa44', lw=1.2, label='yaw')],
        fontsize=6, loc='upper left', frameon=False, labelcolor='0.7')
    imu_cursor = ax_imu.axvline(0.0, color='w', lw=0.8, alpha=0.7)

    # error/correlation panel — scrolling trace + animated cursor
    _mc = mask_corrs[np.isfinite(mask_corrs)]
    ax_corr.plot(t_eye_rel, mask_corrs, color='#ff4aaa', lw=1.0)
    _corr_ylabel = _metric_label
    ax_corr.set_ylabel(_corr_ylabel, color='0.6', fontsize=7, labelpad=2)
    ax_corr.set_xlabel('Time (s)', color='0.6', fontsize=7)
    if len(_mc):
        _ylo, _yhi = np.nanpercentile(_mc, [1, 99])
        _mg = max(0.02, 0.05 * abs(_yhi - _ylo))
        ax_corr.set_ylim(_ylo - _mg, _yhi + _mg)
    ax_corr.set_title(_metric_label.capitalize(), color='0.7', fontsize=7, pad=2, loc='left')
    corr_cursor = ax_corr.axvline(0.0, color='w', lw=0.8, alpha=0.7)

    time_txt = fig.text(0.50, 0.003, '', color='0.5', fontsize=8,
                        ha='center', va='bottom')

    FIG_H = int(fig.get_figheight() * _FIG_DPI)
    FIG_W = int(fig.get_figwidth()  * _FIG_DPI)
    FIG_H -= FIG_H % 2   # most encoders require even dimensions
    FIG_W -= FIG_W % 2

    from fm2p.get_retinal_image import _find_ffmpeg as _gri_ffmpeg
    _ff_path, _ff_enc = _gri_ffmpeg()
    # Ensure output is in a playable pixel format regardless of codec
    if '-pix_fmt' not in _ff_enc:
        _ff_enc = list(_ff_enc) + ['-pix_fmt', 'yuv420p']
    print(f'  ffmpeg encoder: {_ff_enc}')

    proc = _sp.Popen(
        [_ff_path, '-y',
         '-f', 'rawvideo', '-vcodec', 'rawvideo',
         '-s', f'{FIG_W}x{FIG_H}', '-pix_fmt', 'rgb24', '-r', str(_VID_FPS),
         '-i', 'pipe:0', *_ff_enc, out_path],
        stdin=_sp.PIPE, stdout=_sp.DEVNULL, stderr=_sp.PIPE,
    )
    print(f'Figure 5 — rendering {n_out} frames ...')
    _half = _IMU_WIN_S / 2.0
    for k, i in enumerate(out_eye_idxs):
        t_cur = float(t_eye_rel[i])

        im_wc.set_data(wc_frames[k])
        im_msk.set_data(mask_disp[i])
        im_ret.set_data(retinal_images[i][0:60, 60:120])   # az 0–60°, elev 0–60°

        ax_imu.set_xlim(t_cur - _half, t_cur + _half)
        ax_corr.set_xlim(t_cur - _half, t_cur + _half)
        imu_cursor.set_xdata([t_cur, t_cur])
        corr_cursor.set_xdata([t_cur, t_cur])
        time_txt.set_text(f't = {t_cur:.2f} s')

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(FIG_H, FIG_W, 4)
        try:
            proc.stdin.write(arr[:, :, :3].tobytes())
        except BrokenPipeError:
            _, ff_err = proc.communicate()
            raise RuntimeError(
                f'ffmpeg pipe broke at frame {k}:\n'
                + (ff_err.decode(errors='replace') if ff_err else '(no stderr)')
            )

        if (k + 1) % 200 == 0:
            print(f'  frame {k + 1}/{n_out}')

    proc.stdin.close()
    ret_code = proc.wait()
    if ret_code != 0:
        ff_err = proc.stderr.read().decode(errors='replace')
        raise RuntimeError(f'ffmpeg exited {ret_code}:\n{ff_err}')
    plt.close(fig)
    print(f'Saved -> {out_path}')


def figure_6(
    rec_dir=DEFAULT_REC_DIR_FIG4,
    prefix=DEFAULT_PREFIX_FIG4,
    out_path=None,
):

    import subprocess as _sp
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D as _Line2D6

    USE_TRANSFORM_ERROR = True

    _RET_CMAP6  = LinearSegmentedColormap.from_list('retinal', ['#060e06', '#00ff55'])
    _MASK_CMAP6 = LinearSegmentedColormap.from_list('mask',    ['#0a0a0a', '#00ff88'])
    _FIG_BG     = 'k'
    _TRC_BG     = '#0a0a0a'
    _STRIDE     = 2
    _VID_FPS    = 30
    _IMU_WIN_S  = 20.0
    _FIG_DPI    = 100
    _WC_W, _WC_H = 320, 240
    _KER3       = np.ones((3, 3), dtype=np.uint8)   # kernel for outline computation

    h5_path   = os.path.join(rec_dir, f'{prefix}_preproc.h5')
    wc_path   = os.path.join(rec_dir, f'{prefix}_eyecam_deinter.avi')
    mask_path = os.path.join(rec_dir, f'{prefix}_eyecam_deinter_polygon_masks.npy')
    npz_path  = os.path.join(rec_dir, f'{prefix}_retinal_images.npz')
    if out_path is None:
        out_path = os.path.join(rec_dir, f'{prefix}_R01_fig6.mp4')

    print('Figure 6 — loading h5 ...')
    with h5py.File(h5_path, 'r') as f:
        twopT         = f['twopT'][:]
        eyeT_startInd = int(f['eyeT_startInd'][()])
        eyeT_trim     = f['eyeT_trim'][:]
        pitch_2p = (f['pitch_twop_interp'][:]
                    if 'pitch_twop_interp' in f else np.full(len(twopT), np.nan))
        roll_2p  = (f['roll_twop_interp'][:]
                    if 'roll_twop_interp'  in f else np.full(len(twopT), np.nan))
        yaw_2p   = f['head_yaw_deg'][:][:len(twopT)]

    pitch_s, roll_s, yaw_s = _smooth_imu({
        'twopT': twopT, 'pitch_twop_interp': pitch_2p,
        'roll_twop_interp': roll_2p, 'head_yaw_deg': yaw_2p,
    })
    t_twop    = twopT    - twopT[0]
    t_eye_rel = eyeT_trim - eyeT_trim[0]

    pitch_eye = np.interp(t_eye_rel, t_twop, pitch_s)
    roll_eye  = np.interp(t_eye_rel, t_twop, roll_s)
    _fy = np.isfinite(yaw_s)
    yaw_eye = (np.interp(t_eye_rel, t_twop[_fy], yaw_s[_fy])
               if _fy.sum() > 1 else np.full(len(t_eye_rel), np.nan))

    print('Figure 6 — loading retinal npz ...')
    npz            = np.load(npz_path, allow_pickle=False)
    retinal_images = npz['retinal_images']
    N_EYE          = len(npz['eyeT'])

    _metric_label = 'transform error (px)' if USE_TRANSFORM_ERROR else 'mask-retinal corr.'
    print(f'Figure 6 — precomputing masks and {_metric_label} ({N_EYE} frames) ...')
    masks      = np.load(mask_path, mmap_mode='r')
    mask_disp  = np.zeros((N_EYE, 120, 120), dtype=np.uint8)
    mask_corrs = np.zeros(N_EYE, dtype=float)

    if USE_TRANSFORM_ERROR:
        from scipy.ndimage import rotate as _nd_rotate
        _ANGLES = np.arange(-20.0, 21.0, 5.0)
        print(f'  transform search: {len(_ANGLES)} angles × {N_EYE} frames')

    for i in range(N_EYE):
        m    = masks[eyeT_startInd + i].astype(np.float32) * 255.0
        m_sm = cv2.resize(m, (120, 120), interpolation=cv2.INTER_AREA)
        mask_disp[i] = np.clip(m_sm, 0, 255).astype(np.uint8)

        m_c    = m_sm - m_sm.mean()
        m_norm = float(np.sqrt((m_c ** 2).sum()))
        r      = retinal_images[i].astype(float)

        if USE_TRANSFORM_ERROR:
            best_corr = -np.inf
            best_dx = 0.0
            best_dy = 0.0
            for theta in _ANGLES:
                rot   = (_nd_rotate(r, theta, reshape=False, order=1)
                         if theta != 0.0 else r)
                rot_c = rot - rot.mean()
                r_norm = float(np.sqrt((rot_c ** 2).sum()))
                if r_norm < 1e-9 or m_norm < 1e-9:
                    continue
                xcorr = np.fft.fftshift(np.real(
                    np.fft.ifft2(np.fft.fft2(m_c) * np.conj(np.fft.fft2(rot_c)))
                ))
                peak  = np.unravel_index(np.argmax(xcorr), xcorr.shape)
                pcorr = xcorr[peak] / (m_norm * r_norm)
                if pcorr > best_corr:
                    best_corr = pcorr
                    best_dy   = float(peak[0] - m_sm.shape[0] // 2)
                    best_dx   = float(peak[1] - m_sm.shape[1] // 2)
            mask_corrs[i] = float(np.sqrt(best_dx ** 2 + best_dy ** 2))
        else:
            r_c   = r.ravel() - r.mean()
            r_norm = float(np.sqrt((r_c ** 2).sum()))
            den    = m_norm * r_norm
            mask_corrs[i] = float(np.dot(m_c.ravel(), r_c) / den) if den > 0 else 0.0

        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{N_EYE}')
    del masks

    print('Figure 6 — preloading worldcam frames ...')
    out_eye_idxs = np.arange(0, N_EYE, _STRIDE)
    n_out        = len(out_eye_idxs)
    wc_frames    = np.zeros((n_out, _WC_H, _WC_W), dtype=np.uint8)

    cap = cv2.VideoCapture(wc_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, eyeT_startInd)
    for i in range(N_EYE):
        ok, frame = cap.read()
        if not ok:
            break
        if i % _STRIDE == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = _subtract_band(gray, r0=0, r1=480, c0=0, c1=640)
            wc_frames[i // _STRIDE] = cv2.resize(gray, (_WC_W, _WC_H))
    cap.release()
    print(f'  {n_out} output frames')

    print('Figure 6 — building figure ...')
    fig = plt.figure(figsize=(10, 5), dpi=_FIG_DPI, facecolor=_FIG_BG)
    gs  = GridSpec(2, 2, figure=fig,
                   height_ratios=[1.1, 1.0],
                   hspace=0.40, wspace=0.28,
                   left=0.08, right=0.97, top=0.92, bottom=0.12)

    ax_wc    = fig.add_subplot(gs[0, 0])
    ax_combo = fig.add_subplot(gs[0, 1])
    ax_imu   = fig.add_subplot(gs[1, 0])
    ax_corr  = fig.add_subplot(gs[1, 1])

    for ax in (ax_wc, ax_combo):
        ax.set_facecolor(_FIG_BG)
        ax.axis('off')
    ax_combo.set_facecolor('#060e06')

    for ax in (ax_imu, ax_corr):
        ax.set_facecolor(_TRC_BG)
        ax.tick_params(colors='0.6', labelsize=7)
        for sp in ax.spines.values():
            sp.set_color('0.3')

    ax_wc.set_title('Worldcam',
                    color='0.7', fontsize=8, pad=2, loc='left')
    ax_combo.set_title('Reconstruction (green) + GT mask outline (white)',
                       color='0.7', fontsize=8, pad=2, loc='left')

    im_wc = ax_wc.imshow(np.zeros((_WC_H, _WC_W), dtype=np.uint8),
                          cmap='gray', vmin=0, vmax=255,
                          aspect='equal', interpolation='nearest')

    # Bottom layer: retinal reconstruction in green (zorder=1)
    im_ret_combo = ax_combo.imshow(np.zeros((60, 60), dtype=np.uint8),
                                    cmap=_RET_CMAP6, vmin=0, vmax=255,
                                    aspect='equal', interpolation='nearest',
                                    extent=[0, 60, 0, 60], zorder=1)
    # Top layer: GT mask boundary as white RGBA (zorder=2, white always on top)
    im_msk_combo = ax_combo.imshow(np.zeros((60, 60, 4), dtype=np.uint8),
                                    aspect='equal', interpolation='nearest',
                                    extent=[0, 60, 0, 60], zorder=2)
    ax_combo.set_xlim(0, 60)
    ax_combo.set_ylim(0, 60)

    # IMU panel
    ax_yaw_r = ax_imu.twinx()
    ax_yaw_r.set_facecolor(_TRC_BG)
    ax_yaw_r.tick_params(colors='#ffaa44', labelsize=7)
    for sp in ax_yaw_r.spines.values():
        sp.set_color('0.3')

    ax_imu.plot(t_eye_rel, pitch_eye, color='#4a9eff', lw=1.0)
    ax_imu.plot(t_eye_rel, roll_eye,  color='#4aff88', lw=1.0)
    ax_yaw_r.plot(t_eye_rel, yaw_eye, color='#ffaa44', lw=1.0)

    _pr = np.concatenate([pitch_eye[np.isfinite(pitch_eye)],
                          roll_eye[np.isfinite(roll_eye)]])
    if len(_pr):
        _lo, _hi = np.nanpercentile(_pr, [1, 99])
        ax_imu.set_ylim(_lo - 5, _hi + 5)
    ax_yaw_r.set_ylim(0, 360)
    ax_yaw_r.set_yticks([0, 180, 360])

    ax_imu.set_ylabel('pitch / roll (°)', color='0.6', fontsize=7, labelpad=2)
    ax_yaw_r.set_ylabel('yaw (°)',         color='#ffaa44', fontsize=7, labelpad=2)
    ax_imu.set_xlabel('Time (s)',          color='0.6', fontsize=7)
    ax_imu.legend(
        handles=[_Line2D6([0],[0], color='#4a9eff', lw=1.2, label='pitch'),
                 _Line2D6([0],[0], color='#4aff88', lw=1.2, label='roll'),
                 _Line2D6([0],[0], color='#ffaa44', lw=1.2, label='yaw')],
        fontsize=6, loc='upper left', frameon=False, labelcolor='0.7')
    imu_cursor = ax_imu.axvline(0.0, color='w', lw=0.8, alpha=0.7)

    # Error/correlation panel
    _mc = mask_corrs[np.isfinite(mask_corrs)]
    ax_corr.plot(t_eye_rel, mask_corrs, color='#ff4aaa', lw=1.0)
    ax_corr.set_ylabel(_metric_label, color='0.6', fontsize=7, labelpad=2)
    ax_corr.set_xlabel('Time (s)',    color='0.6', fontsize=7)
    if len(_mc):
        _ylo, _yhi = np.nanpercentile(_mc, [1, 99])
        _mg = max(0.02, 0.05 * abs(_yhi - _ylo))
        ax_corr.set_ylim(_ylo - _mg, _yhi + _mg)
    ax_corr.set_title(_metric_label.capitalize(), color='0.7', fontsize=7, pad=2, loc='left')
    corr_cursor = ax_corr.axvline(0.0, color='w', lw=0.8, alpha=0.7)

    time_txt = fig.text(0.50, 0.003, '', color='0.5', fontsize=8,
                        ha='center', va='bottom')

    FIG_H = int(fig.get_figheight() * _FIG_DPI)
    FIG_W = int(fig.get_figwidth()  * _FIG_DPI)
    FIG_H -= FIG_H % 2
    FIG_W -= FIG_W % 2

    from fm2p.get_retinal_image import _find_ffmpeg as _gri_ffmpeg
    _ff_path, _ff_enc = _gri_ffmpeg()
    if '-pix_fmt' not in _ff_enc:
        _ff_enc = list(_ff_enc) + ['-pix_fmt', 'yuv420p']
    print(f'  ffmpeg encoder: {_ff_enc}')

    proc = _sp.Popen(
        [_ff_path, '-y',
         '-f', 'rawvideo', '-vcodec', 'rawvideo',
         '-s', f'{FIG_W}x{FIG_H}', '-pix_fmt', 'rgb24', '-r', str(_VID_FPS),
         '-i', 'pipe:0', *_ff_enc, out_path],
        stdin=_sp.PIPE, stdout=_sp.DEVNULL, stderr=_sp.PIPE,
    )
    print(f'Figure 6 — rendering {n_out} frames ...')
    _half = _IMU_WIN_S / 2.0
    for k, i in enumerate(out_eye_idxs):
        t_cur = float(t_eye_rel[i])

        im_wc.set_data(wc_frames[k])

        # Green retinal reconstruction (top-right quadrant)
        im_ret_combo.set_data(retinal_images[i][0:60, 60:120])

        # White outline of GT mask (top-right quadrant), drawn on top
        m_bin = (mask_disp[i][0:60, 60:120] > 127).astype(np.uint8)
        dilated = cv2.dilate(m_bin, _KER3, iterations=1)
        eroded  = cv2.erode( m_bin, _KER3, iterations=1)
        outline_px = (dilated - eroded).astype(bool)
        outline_rgba = np.zeros((60, 60, 4), dtype=np.uint8)
        outline_rgba[outline_px] = [255, 255, 255, 255]
        im_msk_combo.set_data(outline_rgba)

        ax_imu.set_xlim(t_cur - _half, t_cur + _half)
        ax_corr.set_xlim(t_cur - _half, t_cur + _half)
        imu_cursor.set_xdata([t_cur, t_cur])
        corr_cursor.set_xdata([t_cur, t_cur])
        time_txt.set_text(f't = {t_cur:.2f} s')

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(FIG_H, FIG_W, 4)
        try:
            proc.stdin.write(arr[:, :, :3].tobytes())
        except BrokenPipeError:
            _, ff_err = proc.communicate()
            raise RuntimeError(
                f'ffmpeg pipe broke at frame {k}:\n'
                + (ff_err.decode(errors='replace') if ff_err else '(no stderr)')
            )

        if (k + 1) % 200 == 0:
            print(f'  frame {k + 1}/{n_out}')

    proc.stdin.close()
    ret_code = proc.wait()
    if ret_code != 0:
        ff_err = proc.stderr.read().decode(errors='replace')
        raise RuntimeError(f'ffmpeg exited {ret_code}:\n{ff_err}')
    plt.close(fig)
    print(f'Saved -> {out_path}')


def main():
    parser = argparse.ArgumentParser(description='R01 static diagnostic figure.')
    parser.add_argument('--rec_dir', default=DEFAULT_REC_DIR)
    parser.add_argument('--prefix',  default=DEFAULT_PREFIX)
    parser.add_argument('--fig',     default='1', choices=['1', '2', '3', '4', '5', '6'],
                        help='1 = somatic, 2 = axon, 3 = decoding still, '
                             '4 = retinal still, 5 = retinal video, '
                             '6 = retinal video (combined mask+recon)')
    parser.add_argument('--axon_rec_dir', default=DEFAULT_REC_DIR_AXON)
    parser.add_argument('--axon_prefix',  default=DEFAULT_PREFIX_AXON)
    parser.add_argument('--out',     default=None,
                        help='Output path (.pdf or .png).')
    args = parser.parse_args()

    outpath = args.out
    if outpath is None:
        ext     = 'mp4' if args.fig in ('5', '6') else 'svg'
        outpath = f'./R01_fig{args.fig}.{ext}'

    if args.fig == '6':
        figure_6(out_path=outpath)
    elif args.fig == '5':
        figure_5(out_path=outpath)
    elif args.fig == '4':
        figure_4(out_path=outpath)
    elif args.fig == '3':
        figure_3(rec_dir=args.rec_dir, prefix=args.prefix, out_path=outpath)
    elif args.fig == '2':
        make_r01_figure_axon(
            rec_dir=args.axon_rec_dir, prefix=args.axon_prefix,
            out_path=args.out,
            som_rec_dir=args.rec_dir, som_prefix=args.prefix,
        )
    else:
        make_r01_figure(rec_dir=args.rec_dir, prefix=args.prefix, out_path=args.out)


if __name__ == '__main__':
    main()
