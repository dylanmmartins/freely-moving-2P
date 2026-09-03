# -*- coding: utf-8 -*-
"""
fm2p/show_fm_reconstruction.py

Egocentric vs retinocentric reconstruction video: same session, same pose, the
eye term the only difference between the two rendered scenes.

The top row is the diagnostic video's top row (top-down camera, eye camera, IMU
and pupil traces).  The bottom row is the comparison: egocentric scene,
retinocentric scene, and the frame-wise optic flow of the second minus the
first.  Because the two renders differ only by pupil movement, that difference
field is the retinal image motion the eye adds -- a full-field sweep on every
saccade, near zero in between.  Whichever sequence carries the smaller flow is
the more stable one, and the per-frame means printed in the flow panel say
which.

Reads the `.npz` files `get_retinal_image` already wrote; renders nothing.  Two
are needed, from the same recording:

    ego     `get_retinal_image ... --fixed_pupil`  (pupil frozen at rest)
    retino  `get_retinal_image ...`                (pupil free)

`--fixed_eye` works as the control too, and the bottom-left title follows
whichever file is passed.  It zeroes the anatomical socket as well, so the two
panels then differ by a constant 65 deg rotation on top of the eye movement --
a constant rotation carries no flow, so the flow panel means the same thing
either way, but the two scenes no longer overlap on the page.

A binary silhouette render (no `--walls`) gives the flow panel almost nothing
to work with: outside the pillar the field is uniform, and uniform pixels have
no measurable flow.  The full-scene `--walls` pair is what the comparison is
for.

Usage
-----
    python -m fm2p.show_fm_reconstruction \\
        --h5_path <prefix>_preproc.h5 \\
        --ego_npz <prefix>_retinal_images_fixedpupil.npz \\
        --retino_npz <prefix>_retinal_images.npz

`--h5_path` can be left off when the retinocentric npz still has its default
name.  `--preview` writes a single PNG instead of the video, which is the quick
way to check a window before committing to a render.

Functions
---------
show_fm_reconstruction
    Write the two-row comparison video for one recording.
_load_flow_pair
    Load and cross-check the egocentric and retinocentric image stacks.
_farneback
    Dense optic flow between two rendered frames, in degrees of visual angle.
_hann
    Cached Hann window for the phase-correlation stage.
_flow_mean
    Mean flow magnitude over pixels that carry any image structure.
_sf_worker_init
    Build the figure once per render worker.
_sf_render_frame
    Draw one output frame and hand back its RGB bytes.


DMM, September 2026
"""

if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

import argparse
import os
import subprocess
import time
from multiprocessing import Pool, cpu_count

import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .get_retinal_image import (
    _EYE_H, _EYE_W, _FIG_BG, _IMU_WIN_S, _RET_CMAP, _STRIDE, _TOP_H, _TOP_W,
    _TRC_BG, _VIDEO_FPS, _find_ffmpeg, _find_topdown_video, _interp_short_gaps,
    _nan_wrap, _read_pillar_centroid_px, reconstruction_panel_label,
)

# 1920x1080 at this dpi, which the 2x3 layout needs -- the diagnostic video's
# 14x8 leaves the six panels too small to read.
_FIGSIZE = (16, 9)
_DPI = 120

# Rendered field is 120 px over 120 deg, so a pixel of flow is a degree of
# visual angle and no conversion is needed anywhere below.
_RET_RES = 120
_DEG_PER_PX = 1.0

# Quiver grid spacing in pixels, and the gradient magnitude a pixel needs
# before its flow counts toward the reported means.  Flat surfaces (floor,
# wall, sky) carry no flow information at all: including them would drag both
# means toward zero by however much of the field happens to be featureless.
_FLOW_STEP = 10
_FLOW_TEX_THRESH = 2.0

# Phase-correlation sanity limits: minimum response to believe a global shift,
# and the largest per-frame shift worth believing, in pixels (= degrees).
_PHASE_MIN_RESP = 1e-3
_PHASE_MAX_SHIFT = 60.0

# Gaussian sigma, px, applied to the difference field before it is sampled on
# the quiver grid.  Half the grid spacing, so each arrow is the mean flow over
# the neighbourhood it stands for rather than one pixel's estimate, which on a
# nearly featureless render is mostly noise.
_FLOW_SMOOTH_PX = 5.0

_SF: dict = {}
_HANN = None


def _hann():
    """ Cached Hann window for phase correlation, built on first use. """

    global _HANN
    if _HANN is None:
        _HANN = cv2.createHanningWindow((_RET_RES, _RET_RES), cv2.CV_32F)
    return _HANN


def _farneback(prev, cur):
    """ Dense optic flow prev -> cur, as (u, v) in degrees per frame.

    Two stages.  Phase correlation first, for the global part of the
    displacement, then Farneback on the globally aligned pair for what is left.
    Farneback alone cannot do this: measured on a synthetic pillar-and-horizon
    scene it tracks a shift of 10 px fine and then returns the wrong sign at
    20 px and above, whatever the pyramid depth or window size, because the
    pillar edge has moved past every candidate match in its search band.  A
    saccade is 20-30 deg between frames, which is 20-30 px here, so that is
    precisely the regime the panel exists to show.  Phase correlation recovers
    those shifts exactly, since an eye movement is close to a rigid translation
    of the whole field.

    Parameters
    ----------
    prev, cur : np.ndarray
        (120, 120) uint8 rendered frames.

    Returns
    -------
    flow : np.ndarray
        (120, 120, 2), column and row displacement in degrees.
    """

    p32 = prev.astype(np.float32)
    c32 = cur.astype(np.float32)
    try:
        (dx, dy), resp = cv2.phaseCorrelate(p32, c32, _hann())
    except cv2.error:
        dx, dy, resp = 0.0, 0.0, 0.0
    # A blank frame pair has no phase to correlate: it comes back as the field
    # centre (60, 60) with zero response, which would otherwise be read as a
    # 60 deg saccade every frame of a dark block.  The magnitude cap is the
    # same idea for a high-response mismatch -- nothing physiological moves the
    # retinal image by half the modelled field in one 60 Hz frame.
    if (not np.isfinite(dx) or not np.isfinite(dy) or resp < _PHASE_MIN_RESP
            or max(abs(dx), abs(dy)) > _PHASE_MAX_SHIFT):
        dx = dy = 0.0

    if dx or dy:
        M = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
        base = cv2.warpAffine(prev, M, (_RET_RES, _RET_RES),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    else:
        base = prev

    flow = cv2.calcOpticalFlowFarneback(
        base, cur, None, 0.5, 4, 21, 3, 5, 1.2, 0)
    flow[..., 0] += dx
    flow[..., 1] += dy
    return flow * _DEG_PER_PX


def _flow_mean(img, flow):
    """ Mean flow magnitude over textured pixels of `img`, in degrees. """

    gy, gx = np.gradient(img.astype(np.float32))
    tex = np.hypot(gx, gy) > _FLOW_TEX_THRESH
    if not tex.any():
        return 0.0
    return float(np.mean(np.hypot(flow[..., 0], flow[..., 1])[tex]))


def _load_flow_pair(ego_npz_path, retino_npz_path):
    """ Load both reconstructions and check they describe the same recording.

    Parameters
    ----------
    ego_npz_path, retino_npz_path : str
        Paths to the `_retinal_images.npz` files written by `get_retinal_image`.

    Returns
    -------
    ego_npz, retino_npz : np.lib.npyio.NpzFile
        Open handles, with the image stacks trimmed to a common length.
    ego_imgs, retino_imgs : np.ndarray
        (N, 120, 120) uint8 stacks, indexed by eye-trim frame.
    """

    ego_npz = np.load(ego_npz_path, allow_pickle=False)
    retino_npz = np.load(retino_npz_path, allow_pickle=False)

    ego_imgs = ego_npz['retinal_images']
    retino_imgs = retino_npz['retinal_images']

    def _flag(npz, name):
        try:
            return bool(npz[name])
        except (KeyError, IndexError, TypeError):
            return False

    # The pair only isolates eye movement if the poses match and exactly one
    # side has the pupil frozen.  Warn rather than refuse: someone comparing
    # --fixed_pitch against the full render has a reason to.
    if not (_flag(ego_npz, 'fixed_pupil') or _flag(ego_npz, 'fixed_eye')):
        print('  WARNING: --ego_npz has neither fixed_pupil nor fixed_eye set. '
              'Re-run get_retinal_image with --fixed_pupil for the egocentric '
              'control.')
    if _flag(retino_npz, 'fixed_pupil') or _flag(retino_npz, 'fixed_eye'):
        print('  WARNING: --retino_npz has the pupil frozen, so both panels '
              'show the same thing and the flow panel will be empty.')

    for key in ('pillar_x_mm', 'pillar_y_mm', 'walls', 'invert_pitch',
                'anatomical_eye_angle_deg', 'socket_elev_deg'):
        if key not in ego_npz or key not in retino_npz:
            continue
        if not np.allclose(np.asarray(ego_npz[key], float),
                           np.asarray(retino_npz[key], float),
                           equal_nan=True):
            print(f'  WARNING: {key} differs between the two npz files '
                  f'({ego_npz[key]} vs {retino_npz[key]}) -- the pair does not '
                  f'isolate eye movement.')

    n = min(len(ego_imgs), len(retino_imgs))
    if len(ego_imgs) != len(retino_imgs):
        print(f'  WARNING: image stacks differ in length '
              f'({len(ego_imgs)} vs {len(retino_imgs)}) -- trimming to {n}.')

    print(f'  ego:    {os.path.basename(ego_npz_path)}  '
          f'"{reconstruction_panel_label(ego_npz)}"  {len(ego_imgs)} frames')
    print(f'  retino: {os.path.basename(retino_npz_path)}  '
          f'"{reconstruction_panel_label(retino_npz)}"  {len(retino_imgs)} frames')

    return ego_npz, retino_npz, ego_imgs[:n], retino_imgs[:n]


def _sf_worker_init(init_data: dict) -> None:
    """ Build the 2x3 figure once per worker and cache every artist. """

    global _SF
    _SF = init_data

    fig = plt.figure(figsize=_FIGSIZE, dpi=_DPI, facecolor=_FIG_BG)

    gs = GridSpec(
        2, 3, figure=fig,
        width_ratios=[1, 1, 1.05],
        height_ratios=[1, 1],
        hspace=0.16, wspace=0.14,
        left=0.045, right=0.985, top=0.94, bottom=0.06,
    )

    ax_td = fig.add_subplot(gs[0, 0])
    ax_eye = fig.add_subplot(gs[0, 1])
    ax_ego = fig.add_subplot(gs[1, 0])
    ax_ret = fig.add_subplot(gs[1, 1])
    ax_flow = fig.add_subplot(gs[1, 2])

    gs_tr = GridSpecFromSubplotSpec(5, 1, subplot_spec=gs[0, 2], hspace=0.08)
    ax_pitch = fig.add_subplot(gs_tr[0])
    ax_roll = fig.add_subplot(gs_tr[1])
    ax_yaw = fig.add_subplot(gs_tr[2])
    ax_theta = fig.add_subplot(gs_tr[3])
    ax_phi = fig.add_subplot(gs_tr[4])
    trace_axes = (ax_pitch, ax_roll, ax_yaw, ax_theta, ax_phi)

    for ax in (ax_td, ax_eye):
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
    ax_td.set_title('Top-down camera', color='0.6', fontsize=11, pad=4)
    ax_td.plot(init_data['pillar_x_top'], init_data['pillar_y_top'],
               '*', color='#ff4444', markersize=14, markeredgewidth=1,
               markeredgecolor='white', zorder=10)

    # Hollow marker is the raw DLC ear-bar midpoint, filled is the fused
    # position the renderer used.  A filled dot with no hollow one around it
    # means the pose is interpolated.
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
    ax_eye.set_title('Eye camera', color='0.6', fontsize=11, pad=4)

    # Both scene panels get the same extent, colour scale and limits: the
    # comparison is only readable if a pixel means the same thing in each.
    scene_ims = []
    for ax, role, lbl in (
            (ax_ego, 'Egocentric (eye frozen)', init_data['ego_label']),
            (ax_ret, 'Retinocentric (eye free)', init_data['retino_label'])):
        im = ax.imshow(
            np.zeros((_RET_RES, _RET_RES), dtype=np.uint8),
            cmap=_RET_CMAP, vmin=0, vmax=255,
            aspect='equal', interpolation='nearest',
            extent=[-60, 60, -60, 60],
        )
        ax.set_facecolor(_FIG_BG)
        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)
        # Role on top, then what the npz says it actually is.  The two are not
        # the same sentence: --fixed_pupil keeps the anatomical socket, so
        # reconstruction_panel_label calls it a fixed-eye RETINOCENTRIC image
        # even though it is the egocentric control here.
        ax.set_title(role, color='0.75', fontsize=11, pad=17)
        ax.text(0.5, 1.008, lbl, transform=ax.transAxes, ha='center',
                va='bottom', color='0.45', fontsize=8)
        ax.set_xlabel('Azimuth (deg)', color='0.5', fontsize=9)
        ax.set_ylabel('Elevation (deg)', color='0.5', fontsize=9)
        ax.tick_params(colors='0.4', labelsize=8)
        for sp in ax.spines.values():
            sp.set_color('0.3')
        scene_ims.append(im)
    im_ego, im_ret = scene_ims

    # Flow panel.  angles/scale_units='xy' with scale=1/gain draws each arrow
    # at its true angular length, so a saccade's arrows visibly cross the
    # panel while inter-saccade flow stays a stipple.
    gain = float(init_data['flow_gain'])
    grid = np.arange(_FLOW_STEP // 2, _RET_RES, _FLOW_STEP)
    gx, gy = np.meshgrid(grid, grid)
    qx = gx - (_RET_RES / 2.0 - 0.5)
    qy = (_RET_RES / 2.0 - 0.5) - gy
    zeros = np.zeros_like(qx, dtype=float)
    quiv = ax_flow.quiver(
        qx, qy, zeros, zeros, zeros,
        cmap='cool', clim=(0, float(init_data['flow_clim'])),
        angles='xy', scale_units='xy', scale=1.0 / gain,
        width=0.005, headwidth=3.5, headlength=4.0, zorder=5,
    )
    quiv_mean = ax_flow.quiver(
        [0.0], [0.0], [0.0], [0.0],
        color='w', angles='xy', scale_units='xy', scale=1.0 / gain,
        width=0.011, headwidth=3.0, headlength=3.5, alpha=0.85, zorder=6,
    )
    ax_flow.set_facecolor(_FIG_BG)
    ax_flow.set_xlim(-60, 60)
    ax_flow.set_ylim(-60, 60)
    ax_flow.set_aspect('equal')
    ax_flow.set_title('Retinocentric minus egocentric flow',
                      color='0.6', fontsize=11, pad=4)
    ax_flow.set_xlabel('Azimuth (deg)', color='0.5', fontsize=9)
    ax_flow.tick_params(colors='0.4', labelsize=8)
    for sp in ax_flow.spines.values():
        sp.set_color('0.3')
    flow_txt = ax_flow.text(
        -57, 57, '', color='0.85', fontsize=9, ha='left', va='top',
        family='monospace', zorder=8,
        bbox=dict(facecolor=_FIG_BG, edgecolor='0.3', alpha=0.8, pad=3),
    )
    # Arrow colour carries the same quantity as arrow length, so the bar is
    # what makes a short arrow at the field edge readable.
    # Backing panel plus an explicit zorder: the inset axes would otherwise be
    # drawn under the quiver, which sits right on top of it.
    ax_flow.add_patch(Rectangle(
        (0.555, 0.015), 0.435, 0.145, transform=ax_flow.transAxes,
        facecolor=_FIG_BG, edgecolor='0.3', zorder=7))
    cax = ax_flow.inset_axes([0.60, 0.085, 0.36, 0.028], zorder=8)
    cbar = fig.colorbar(quiv, cax=cax, orientation='horizontal')
    cbar.set_label('|flow| (deg/frame)', color='0.7', fontsize=7, labelpad=1)
    cbar.ax.tick_params(colors='0.7', labelsize=7, length=2, pad=1)
    cbar.outline.set_edgecolor('0.4')

    # Traces, all sharing the recording clock so one cursor serves all five.
    twopT = init_data['twopT']
    t_start = float(init_data['t_start'])
    blk_lo = init_data['twop_lo']
    blk_nd = init_data['twop_nd']

    fps_2p = ((len(twopT) - 1) / float(twopT[-1] - twopT[0])
              if len(twopT) > 1 else 7.5)
    ext_lo = max(0, blk_lo - int(_IMU_WIN_S / 2.0 * fps_2p))

    tt_rel = twopT[ext_lo:blk_nd] - t_start
    pitch_b = _interp_short_gaps(init_data['pitch'][ext_lo:blk_nd].astype(float))
    roll_b = _interp_short_gaps(init_data['roll'][ext_lo:blk_nd].astype(float))

    _yaw_fused = init_data.get('head_yaw_fused_out')
    if _yaw_fused is not None:
        _yb = np.interp(twopT[ext_lo:blk_nd], init_data['times'],
                        np.unwrap(np.radians(_yaw_fused)) * 180.0 / np.pi) % 360.0
    else:
        _yb = init_data['yaw'][ext_lo:blk_nd].copy().astype(float)
    yaw_b = _nan_wrap(_interp_short_gaps(_yb))

    # theta/phi at the eye-camera rate: resampling them onto the 7.5 Hz 2P grid
    # erodes sparse pupil tracking to almost nothing, and the gaps that opens
    # do not correspond to the gaps in what was rendered.
    _eyeT_tr = init_data.get('eyeT_trim_full')
    _th_tr = init_data.get('theta_trim')
    _ph_tr = init_data.get('phi_trim')
    if _eyeT_tr is not None and _th_tr is not None:
        _w = ((_eyeT_tr >= float(twopT[ext_lo])) &
              (_eyeT_tr <= float(twopT[blk_nd - 1])) & np.isfinite(_eyeT_tr))
        te_rel = _eyeT_tr[_w] - t_start
        theta_b = _interp_short_gaps(np.asarray(_th_tr, float)[_w])
        phi_b = _interp_short_gaps(np.asarray(_ph_tr, float)[_w])
    else:
        te_rel = tt_rel
        theta_b = _interp_short_gaps(init_data['theta_2p'][ext_lo:blk_nd].astype(float))
        phi_b = _interp_short_gaps(init_data['phi_2p'][ext_lo:blk_nd].astype(float))

    colors = ['#4a9eff', '#4aff88', '#ffaa44', '#ff4aaa', '#bb88ff']
    labels = ['Pitch', 'Roll', 'Yaw', 'θ', 'φ']
    signals = [pitch_b, roll_b, yaw_b, theta_b, phi_b]
    sig_t = [tt_rel, tt_rel, tt_rel, te_rel, te_rel]

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

    ax_pitch.set_title('Head and eye', color='0.6', fontsize=11, pad=4)
    ax_phi.set_xlabel('Time (s)', color='0.6', fontsize=8)
    for ax in (ax_pitch, ax_roll, ax_yaw, ax_theta):
        ax.tick_params(labelbottom=False)

    cursors = [ax.axvline(0.0, color='w', lw=0.8, alpha=0.8)
               for ax in trace_axes]

    time_txt = fig.text(0.010, 0.984, '', color='0.75', fontsize=11,
                        ha='left', va='top', family='monospace')

    _SF['fig'] = fig
    _SF['im_td'] = im_td
    _SF['im_eye'] = im_eye
    _SF['im_ego'] = im_ego
    _SF['im_ret'] = im_ret
    _SF['quiv'] = quiv
    _SF['quiv_mean'] = quiv_mean
    _SF['flow_txt'] = flow_txt
    _SF['grid'] = grid
    _SF['trace_axes'] = trace_axes
    _SF['cursors'] = cursors
    _SF['time_txt'] = time_txt
    _SF['head_anchor'] = head_anchor
    _SF['head_dot'] = head_dot
    _SF['head_dir'] = head_dir
    _SF['gaze_dir'] = gaze_dir
    _SF['FIG_H'] = int(fig.get_figheight() * _DPI)
    _SF['FIG_W'] = int(fig.get_figwidth() * _DPI)


def _sf_render_frame(out_idx: int) -> bytes:
    """ Draw output frame `out_idx` and return it as packed RGB bytes. """

    t_abs = float(_SF['times'][out_idx])
    t_rel = t_abs - float(_SF['t_start'])

    _SF['im_eye'].set_data(_SF['eye_frames'][out_idx])

    # Top-down frame comes from the precomputed time match, so it is always the
    # frame nearest in time regardless of the container's fps header.
    top_bi = int(_SF['top_idx_for_output'][out_idx])
    _SF['im_td'].set_data(
        cv2.cvtColor(_SF['top_frames'][top_bi], cv2.COLOR_BGR2RGB))

    hx = _SF['head_x_arr']
    hy = _SF['head_y_arr']
    sx = _SF['top_scale_x']
    sy = _SF['top_scale_y']
    tidx = min(int(_SF['top_buf_track_idx'][top_bi]), len(hx) - 1)

    _fx = _SF.get('head_x_top_px')
    if _fx is not None:
        _ri = min(int(_SF['eye_trim_idx'][out_idx]), len(_fx) - 1)
        px, py = float(_fx[_ri]), float(_SF['head_y_top_px'][_ri])
    else:
        px, py = float(hx[tidx]), float(hy[tidx])

    _ax, _ay = float(hx[tidx]), float(hy[tidx])
    if np.isfinite(_ax) and np.isfinite(_ay):
        _SF['head_anchor'].set_data([_ax * sx], [_ay * sy])
    else:
        _SF['head_anchor'].set_data([], [])

    _fused = _SF.get('head_yaw_fused_out')
    yaw_deg = (float(_fused[out_idx]) if _fused is not None
               else float(_SF['head_yaw_arr'][min(tidx, len(hx) - 1)]))
    if np.isfinite(px) and np.isfinite(py):
        dx_top = px * sx
        dy_top = py * sy
        _SF['head_dot'].set_data([dx_top], [dy_top])
        if np.isfinite(yaw_deg):
            arrow_len = 20.0
            yaw_rad = np.radians(yaw_deg)
            _SF['head_dir'].set_data(
                [dx_top, dx_top + arrow_len * np.cos(yaw_rad)],
                [dy_top, dy_top + arrow_len * np.sin(yaw_rad)])

            # Yellow is the head axis the egocentric panel is drawn from, green
            # the gaze axis the retinocentric panel is drawn from: the flow
            # panel is what separates them.  fixed_eye and worldcam render
            # through a zeroed socket, so there the two coincide.
            if _SF.get('head_aligned', False):
                gaze_deg = yaw_deg
            else:
                gaze_deg = (yaw_deg - _SF['anatomical_eye_angle_deg']
                            + _SF['resting_tilt_h'])
            gaze_rad = np.radians(gaze_deg)
            _SF['gaze_dir'].set_data(
                [dx_top, dx_top + arrow_len * np.cos(gaze_rad)],
                [dy_top, dy_top + arrow_len * np.sin(gaze_rad)])
        else:
            _SF['head_dir'].set_data([], [])
            _SF['gaze_dir'].set_data([], [])
    else:
        _SF['head_dot'].set_data([], [])
        _SF['head_dir'].set_data([], [])
        _SF['gaze_dir'].set_data([], [])

    k_cur = int(_SF['ret_idx'][out_idx])
    k_prev = int(_SF['ret_idx'][max(out_idx - 1, 0)])
    ego_cur = _SF['ego_imgs'][k_cur]
    ret_cur = _SF['retino_imgs'][k_cur]
    _SF['im_ego'].set_data(ego_cur)
    _SF['im_ret'].set_data(ret_cur)

    # Flow is measured between successive DISPLAYED frames, so the arrows are
    # the motion the viewer just saw rather than motion at some other rate.
    if k_cur == k_prev:
        d = np.zeros((_RET_RES, _RET_RES, 2), dtype=np.float32)
        m_ego = m_ret = 0.0
    else:
        ego_prev = _SF['ego_imgs'][k_prev]
        ret_prev = _SF['retino_imgs'][k_prev]
        f_ego = _farneback(ego_prev, ego_cur)
        f_ret = _farneback(ret_prev, ret_cur)
        d = f_ret - f_ego
        m_ego = _flow_mean(ego_prev, f_ego)
        m_ret = _flow_mean(ret_prev, f_ret)

    sm = _SF['flow_smooth']
    if sm > 0:
        d = cv2.GaussianBlur(d, (0, 0), sm)

    g = _SF['grid']
    sub = d[np.ix_(g, g)]
    # Rows run downward in the image but elevation runs upward on the axis, so
    # the row component flips sign before it becomes an arrow.
    u = sub[..., 0]
    v = -sub[..., 1]
    _SF['quiv'].set_UVC(u, v, np.hypot(u, v))
    _SF['quiv_mean'].set_UVC(
        np.array([np.nanmean(u)]), np.array([np.nanmean(v)]))
    _SF['flow_txt'].set_text(
        f'egocentric    {m_ego:6.2f} deg/frame\n'
        f'retinocentric {m_ret:6.2f} deg/frame')

    for ax, cur in zip(_SF['trace_axes'], _SF['cursors']):
        ax.set_xlim(t_rel - _IMU_WIN_S / 2.0, t_rel + _IMU_WIN_S / 2.0)
        cur.set_xdata([t_rel, t_rel])

    _SF['time_txt'].set_text(f't = {t_rel:.2f} s')

    _SF['fig'].canvas.draw()
    buf = _SF['fig'].canvas.buffer_rgba()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(
        _SF['FIG_H'], _SF['FIG_W'], 4)
    return arr[:, :, :3].tobytes()


def show_fm_reconstruction(
    h5_path,
    ego_npz_path,
    retino_npz_path,
    eye_path=None,
    top_path=None,
    out_path=None,
    arena_width_cm=None,
    stride=_STRIDE,
    max_seconds=80.0,
    t_start_override=None,
    flow_gain=1.0,
    flow_clim=20.0,
    flow_smooth=_FLOW_SMOOTH_PX,
    preview=False,
):
    """ Write the egocentric vs retinocentric comparison video for one recording.

    Parameters
    ----------
    h5_path : str
        The `_preproc.h5` for the recording.
    ego_npz_path, retino_npz_path : str
        `get_retinal_image` outputs, pupil frozen and pupil free.
    eye_path, top_path : str or None
        Eye and arena videos.  Found next to `h5_path` when not given.
    out_path : str or None
        Output MP4.  Defaults to `<prefix>_fm_reconstruction.mp4`.
    arena_width_cm : float or None
        Recompute pxls2cm from the annotated corners and this width.
    stride : int
        Sub-sample eye-camera frames.  2 turns 60 fps into 30 fps output.
    max_seconds : float
        Longest stretch of the chosen light block to render.
    t_start_override : float or None
        Start time in recording seconds, instead of the best light block.
    flow_gain : float
        Arrow length multiplier.  1.0 draws flow at its true angular length.
    flow_clim : float
        Flow magnitude, deg/frame, at the top of the arrow colour scale.
    flow_smooth : float
        Gaussian sigma in pixels for the difference field.  0 turns it off.
    preview : bool
        Render one frame from the middle of the block to PNG and stop.

    Returns
    -------
    out_path : str
        Path written.
    """

    rec_dir = os.path.dirname(h5_path)
    prefix = os.path.basename(h5_path).replace('_preproc.h5', '')

    if eye_path is None:
        eye_path = os.path.join(rec_dir, f'{prefix}_eyecam_deinter.avi')
    if top_path is None:
        top_path = _find_topdown_video(rec_dir)
    if out_path is None:
        out_path = os.path.join(rec_dir, f'{prefix}_fm_reconstruction.mp4')

    print(f'Topdown video: {os.path.basename(top_path)}')

    print('Loading preproc ...')
    with h5py.File(h5_path, 'r') as f:
        eyeT_trim = f['eyeT_trim'][:]
        twopT = f['twopT'][:]
        light_onsets = f['light_onsets'][:]
        dark_onsets = f['dark_onsets'][:]
        startInd = int(f['eyeT_startInd'][()])
        theta_raw = f['theta'][:]
        phi_raw = f['phi'][:]
        theta_2p = f['theta_interp'][:]
        phi_2p = f['phi_interp'][:]
        theta_trim_v = f['theta_trim'][:] if 'theta_trim' in f else None
        phi_trim_v = f['phi_trim'][:] if 'phi_trim' in f else None
        head_x = f['head_x'][:]
        head_y = f['head_y'][:]
        head_yaw = f['head_yaw_deg'][:]
        try:
            pitch = f['pitch_twop_interp'][:]
            roll = f['roll_twop_interp'][:]
        except KeyError:
            pitch = np.zeros(len(twopT))
            roll = np.zeros(len(twopT))
            print('  WARNING: pitch_twop_interp / roll_twop_interp not in H5 '
                  '-- pitch/roll set to zero.')
        pxls2cm = float(f['pxls2cm'][()])
        if arena_width_cm is not None:
            try:
                px_width = np.mean([
                    float(f['arenaTR']['x'][()]) - float(f['arenaTL']['x'][()]),
                    float(f['arenaBR']['x'][()]) - float(f['arenaBL']['x'][()])])
                pxls2cm = px_width / arena_width_cm
                print(f'  Overriding pxls2cm from arena_width_cm='
                      f'{arena_width_cm:.1f} to {pxls2cm:.3f}')
            except KeyError:
                print('  WARNING: arena corner keys missing, keeping stored '
                      'pxls2cm.')
        # Average the annotation polygon: pillar_centroid is wrong on some
        # sessions, and the renderer already averages the polygon.
        pillar_x_px, pillar_y_px = _read_pillar_centroid_px(f)
        try:
            imuT_trim_arr = f['imuT_trim'][:]
            yaw_imu_arr = f['upsampled_yaw']['igyro_corrected_deg'][:]
        except KeyError:
            imuT_trim_arr = None
            yaw_imu_arr = None

    print('Loading reconstructions ...')
    ego_npz, retino_npz, ego_imgs, retino_imgs = _load_flow_pair(
        ego_npz_path, retino_npz_path)

    def _flag(npz, name):
        try:
            return bool(npz[name])
        except (KeyError, IndexError, TypeError):
            return False

    resting_tilt_h = (float(retino_npz['mean_pfh'])
                      if 'mean_pfh' in retino_npz else 0.0)
    anatomical_eye_angle_deg = (float(retino_npz['anatomical_eye_angle_deg'])
                                if 'anatomical_eye_angle_deg' in retino_npz
                                else -65.0)
    head_aligned = _flag(retino_npz, 'worldcam') or _flag(retino_npz, 'fixed_eye')
    if head_aligned:
        resting_tilt_h = 0.0

    head_x_top_px = (retino_npz['head_x_top_px']
                     if 'head_x_top_px' in retino_npz else None)
    head_y_top_px = (retino_npz['head_y_top_px']
                     if 'head_y_top_px' in retino_npz else None)
    if head_x_top_px is None:
        print('  Topdown position overlay: raw DLC anchors '
              '(npz predates head_x_top_px -- re-run get_retinal_image)')

    theta_trim = theta_raw[startInd: startInd + len(eyeT_trim)]
    phi_trim = phi_raw[startInd: startInd + len(eyeT_trim)]

    px2cm = 1.0 / pxls2cm
    hx_f = head_x.astype(float)
    hy_f = head_y.astype(float)
    dt_tp = np.diff(twopT)
    dx_cm = np.diff(hx_f) * px2cm
    dy_cm = np.diff(hy_f) * px2cm
    speed_twop = np.full(len(twopT), np.nan)
    ok = np.isfinite(dx_cm) & np.isfinite(dy_cm) & (dt_tp > 0)
    speed_twop[:-1][ok] = np.sqrt((dx_cm[ok] / dt_tp[ok])**2 +
                                  (dy_cm[ok] / dt_tp[ok])**2)

    theta_at_twop = np.interp(twopT, eyeT_trim, theta_trim.astype(float))
    phi_at_twop = np.interp(twopT, eyeT_trim, phi_trim.astype(float))
    eye_ok_twop = np.isfinite(theta_at_twop) & np.isfinite(phi_at_twop)

    # Pick the light block with the most frames that are both moving and
    # eye-tracked: without pupil tracking the two bottom panels are identical.
    best_idx, best_score = -1, -1
    for i in range(len(light_onsets)):
        lo_i = int(light_onsets[i])
        nxt = dark_onsets[dark_onsets > lo_i]
        if len(nxt) == 0:
            continue
        nd_i = min(int(nxt[0]), len(twopT) - 1)
        if nd_i <= lo_i:
            continue
        score = int(np.sum((speed_twop[lo_i:nd_i] > 2.0) &
                           eye_ok_twop[lo_i:nd_i]))
        if score > best_score:
            best_score, best_idx = score, i

    if best_idx < 0:
        raise RuntimeError('No valid light block found.')

    lo = int(light_onsets[best_idx])
    nd = min(int(dark_onsets[dark_onsets > lo][0]), len(twopT) - 1)
    if t_start_override is not None:
        # An explicit start is taken at face value, block boundaries and all:
        # the window it opens runs max_seconds from there, not to whatever the
        # automatically chosen block happened to end at.
        lo = int(np.clip(np.searchsorted(twopT, t_start_override),
                         0, len(twopT) - 3))
        nd = len(twopT) - 1
    t_start = float(twopT[lo])

    nd = min(nd, int(np.searchsorted(twopT, t_start + max_seconds)))
    nd = max(nd, lo + 2)
    t_end = float(twopT[nd])
    if t_start_override is None:
        print(f'  Light block {best_idx}: 2P [{lo}:{nd}]  '
              f't=[{t_start:.1f}-{t_end:.1f}]s  score={best_score} frames '
              f'(moving + eye tracked)')
    else:
        print(f'  Requested window: 2P [{lo}:{nd}]  '
              f't=[{t_start:.1f}-{t_end:.1f}]s')

    eye_start = int(np.searchsorted(eyeT_trim, t_start))
    eye_end = int(np.searchsorted(eyeT_trim, t_end))
    eye_trim_idx = np.arange(eye_start, eye_end, stride)
    if len(eye_trim_idx) < 2:
        raise RuntimeError('Chosen window holds fewer than two eye frames.')
    eye_full_idx = eye_trim_idx + startInd
    times = eyeT_trim[eye_trim_idx]

    twop_idx = np.searchsorted(twopT, times).clip(0, len(twopT) - 1)
    twop_idx_prev = (twop_idx - 1).clip(0)
    closer_prev = (np.abs(twopT[twop_idx_prev] - times)
                   < np.abs(twopT[twop_idx] - times))
    twop_idx = np.where(closer_prev, twop_idx_prev, twop_idx)

    n_frames = len(eye_trim_idx)
    print(f'  {n_frames} output frames  (stride={stride})')

    # Only the block's images travel to the workers.  Handing over the whole
    # session would copy two multi-GB stacks per process for a 80 s clip.
    n_img = len(ego_imgs)
    k0 = int(min(eye_trim_idx[0], n_img - 1))
    k1 = int(min(eye_trim_idx[-1], n_img - 1)) + 1
    ret_idx = np.clip(eye_trim_idx, 0, n_img - 1) - k0
    ego_block = np.ascontiguousarray(ego_imgs[k0:k1])
    retino_block = np.ascontiguousarray(retino_imgs[k0:k1])
    print(f'  Image block [{k0}:{k1}]  '
          f'{ego_block.nbytes / 1e6:.0f} MB per reconstruction')
    # A full session is ~1.5 GB per stack, and everything in init_data is
    # pickled once per worker.  Drop the originals now that the block is copied.
    del ego_imgs, retino_imgs

    print('Pre-loading eye frames ...')
    t0 = time.time()
    eye_frames = np.zeros((n_frames, _EYE_H, _EYE_W), dtype=np.uint8)
    cap = cv2.VideoCapture(eye_path)
    target_start = int(eye_full_idx[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(target_start):
        cap.grab()

    current = target_start
    for i, target in enumerate(eye_full_idx.astype(int)):
        for _ in range(target - current):
            cap.grab()
        ret, frame = cap.read()
        if ret:
            _gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
            # Band subtraction: per-row illumination estimated from the edge
            # columns, where the eye never is.
            _pr = gaussian_filter1d(
                0.5 * (_gr[:, :10].mean(axis=1) + _gr[:, -10:].mean(axis=1)),
                sigma=15)
            _gr -= _pr[:, np.newaxis]
            _gr -= _gr.min()
            _mx = _gr.max()
            if _mx > 0:
                _gr = _gr / _mx * 255.0
            eye_frames[i] = cv2.resize(
                np.clip(_gr, 0, 255).astype(np.uint8), (_EYE_W, _EYE_H))
        current = target + 1
    cap.release()
    print(f'  done in {time.time() - t0:.1f}s')

    print('Pre-loading topdown frames ...')
    t0 = time.time()
    cap = cv2.VideoCapture(top_path)
    top_fps_reported = cap.get(cv2.CAP_PROP_FPS)
    if top_fps_reported <= 0:
        top_fps_reported = 30.0

    # Seek by time, not frame number: CAP_PROP_FPS is not reliable on these
    # containers, so every frame's real POS_MSEC is recorded instead.
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_start - 0.5) * 1000.0)
    top_fps_est = min(max(top_fps_reported, 5.0), 120.0)
    n_top_est = int((t_end - t_start + 2.0) * top_fps_est) + 20
    top_frames = np.zeros((n_top_est, _TOP_H, _TOP_W, 3), dtype=np.uint8)
    top_buf_times = np.full(n_top_est, np.nan)
    n_top_loaded = 0
    orig_top_h, orig_top_w = _TOP_H, _TOP_W

    while n_top_loaded < n_top_est:
        ret, frame = cap.read()
        if not ret:
            break
        # POS_MSEC has to be read AFTER the read: before it, it reports the
        # previous frame and every buffered frame is mislabelled by one.
        t_vid = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t_vid > t_end + 0.5:
            break
        if n_top_loaded == 0:
            orig_top_h, orig_top_w = frame.shape[:2]
        top_buf_times[n_top_loaded] = t_vid
        _fr = cv2.resize(frame, (_TOP_W, _TOP_H))
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

    if n_top_loaded == 0:
        raise RuntimeError('Failed to load any topdown frames -- check '
                           'top_path and the chosen time window.')

    top_buf_times = top_buf_times[:n_top_loaded]
    top_frames = top_frames[:n_top_loaded]
    top_scale_x = _TOP_W / orig_top_w
    top_scale_y = _TOP_H / orig_top_h
    print(f'  done in {time.time() - t0:.1f}s  ({n_top_loaded} frames, '
          f't=[{top_buf_times[0]:.2f}-{top_buf_times[-1]:.2f}]s, '
          f'original {orig_top_w}x{orig_top_h})')

    target_top_times = twopT[twop_idx].astype(float)
    top_idx_for_output = np.searchsorted(
        top_buf_times, target_top_times, side='left')
    top_idx_for_output = np.clip(top_idx_for_output, 0, n_top_loaded - 1)
    left_idx = np.maximum(top_idx_for_output - 1, 0)
    prefer_left = (np.abs(top_buf_times[left_idx] - target_top_times) <
                   np.abs(top_buf_times[top_idx_for_output] - target_top_times))
    top_idx_for_output = np.where(prefer_left, left_idx, top_idx_for_output)

    # Overlay is indexed by the top-down frame on screen, not by the 2P index:
    # any disagreement between twopT and the video clock otherwise shows up as
    # the tracking dot trailing the animal.
    top_buf_track_idx = np.searchsorted(
        twopT, top_buf_times).clip(0, len(twopT) - 1)
    _prev = (top_buf_track_idx - 1).clip(0)
    _use_prev = (np.abs(twopT[_prev] - top_buf_times)
                 < np.abs(twopT[top_buf_track_idx] - top_buf_times))
    top_buf_track_idx = np.where(_use_prev, _prev, top_buf_track_idx)

    pillar_x_top = pillar_x_px * top_scale_x
    pillar_y_top = pillar_y_px * top_scale_y

    # Head-direction arrow comes from yaw_eye in the npz -- the yaw the frames
    # were actually rendered through.  get_retinal_image re-fuses gyro_z_trim
    # with head_yaw_deg, so the h5's stored upsampled_yaw is a different trace.
    _npz_yaw = retino_npz['yaw_eye'] if 'yaw_eye' in retino_npz else None
    _npz_eyeT = retino_npz['eyeT'] if 'eyeT' in retino_npz else np.array([])
    if _npz_yaw is not None and len(_npz_eyeT) == len(_npz_yaw):
        _ok = np.isfinite(_npz_yaw)
        head_yaw_fused_out = np.interp(
            times, np.asarray(_npz_eyeT, float)[_ok],
            np.unwrap(np.radians(np.asarray(_npz_yaw, float)[_ok])) * 180.0 / np.pi,
        ) % 360.0
        print(f'  Head-direction arrow: yaw_eye from the npz '
              f'({int(_ok.sum())} finite samples)')
    elif imuT_trim_arr is not None and yaw_imu_arr is not None:
        _yaw_src = np.asarray(yaw_imu_arr, dtype=float)
        _n_yaw = min(len(imuT_trim_arr), len(_yaw_src))
        head_yaw_fused_out = np.interp(
            times, np.asarray(imuT_trim_arr, dtype=float)[:_n_yaw],
            np.unwrap(np.radians(_yaw_src[:_n_yaw])) * 180.0 / np.pi,
        ) % 360.0
        print('  Head-direction arrow: WARNING stored upsampled_yaw -- npz has '
              'no yaw_eye, so the arrow may not match the render.')
    else:
        head_yaw_fused_out = None
        print('  Head-direction arrow: WARNING falling back to raw topdown yaw')

    init_data = {
        'ego_label': reconstruction_panel_label(ego_npz),
        'retino_label': reconstruction_panel_label(retino_npz),
        'ego_imgs': ego_block,
        'retino_imgs': retino_block,
        'ret_idx': ret_idx,
        'flow_gain': flow_gain,
        'flow_clim': flow_clim,
        'flow_smooth': float(flow_smooth),
        'head_aligned': head_aligned,
        'head_yaw_fused_out': head_yaw_fused_out,
        'top_buf_track_idx': top_buf_track_idx,
        'eye_frames': eye_frames,
        'top_frames': top_frames,
        'eye_trim_idx': eye_trim_idx,
        'twop_idx': twop_idx,
        'times': times,
        'twopT': twopT,
        'pitch': pitch,
        'roll': roll,
        'yaw': head_yaw[:len(twopT)],
        'theta_2p': theta_2p,
        'phi_2p': phi_2p,
        'twop_lo': lo,
        'twop_nd': nd,
        't_start': t_start,
        'pillar_x_top': pillar_x_top,
        'pillar_y_top': pillar_y_top,
        'head_x_arr': hx_f,
        'head_y_arr': hy_f,
        'head_x_top_px': head_x_top_px,
        'head_y_top_px': head_y_top_px,
        'eyeT_trim_full': eyeT_trim,
        'theta_trim': theta_trim_v,
        'phi_trim': phi_trim_v,
        'head_yaw_arr': head_yaw[:len(twopT)].astype(float),
        'top_scale_x': top_scale_x,
        'top_scale_y': top_scale_y,
        'top_idx_for_output': top_idx_for_output,
        'resting_tilt_h': resting_tilt_h,
        'anatomical_eye_angle_deg': anatomical_eye_angle_deg,
    }

    if preview:
        png_path = os.path.splitext(out_path)[0] + '_preview.png'
        _sf_worker_init(init_data)
        _sf_render_frame(n_frames // 2)
        _SF['fig'].savefig(png_path, dpi=_DPI, facecolor=_FIG_BG)
        plt.close(_SF['fig'])
        print(f'Preview frame {n_frames // 2} saved to {png_path}')
        return png_path

    _probe = plt.figure(figsize=_FIGSIZE, dpi=_DPI)
    FIG_H = int(_probe.get_figheight() * _DPI)
    FIG_W = int(_probe.get_figwidth() * _DPI)
    plt.close(_probe)
    FIG_H += FIG_H % 2
    FIG_W += FIG_W % 2
    print(f'Output: {FIG_W}x{FIG_H} px  |  {n_frames} frames @ {_VIDEO_FPS} fps')

    ffmpeg_bin, codec_args = _find_ffmpeg()
    ffmpeg_cmd = [
        ffmpeg_bin, '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{FIG_W}x{FIG_H}',
        '-pix_fmt', 'rgb24',
        '-r', str(_VIDEO_FPS),
        '-i', 'pipe:0',
        *codec_args,
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        out_path,
    ]
    print(f'Writing to {out_path}  (codec: {codec_args[1]})')
    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    n_workers = max(1, min(cpu_count() - 1, 8))
    print(f'Rendering with {n_workers} worker(s) ...')
    t0, n_written = time.time(), 0

    try:
        with Pool(processes=n_workers, initializer=_sf_worker_init,
                  initargs=(init_data,)) as pool:
            for frame_bytes in pool.imap(_sf_render_frame,
                                         list(range(n_frames)), chunksize=4):
                ffmpeg_proc.stdin.write(frame_bytes)
                n_written += 1
                if n_written % 150 == 0:
                    print(f'  {n_written}/{n_frames}  '
                          f'({n_written / (time.time() - t0):.1f} fps)')
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
    print(f'Done in {elapsed:.1f}s  ({n_frames / elapsed:.1f} fps)  '
          f'to {out_path}')

    return out_path


def main():
    p = argparse.ArgumentParser(
        description='Egocentric vs retinocentric reconstruction video.')
    p.add_argument('--h5_path', default=None,
                   help='The _preproc.h5 for the recording. Derived from '
                        '--retino_npz when omitted.')
    p.add_argument('--ego_npz', required=True,
                   help='npz from get_retinal_image --fixed_pupil.')
    p.add_argument('--retino_npz', required=True,
                   help='npz from get_retinal_image with the pupil free.')
    p.add_argument('--eye_path', default=None)
    p.add_argument('--top_path', default=None)
    p.add_argument('--out_path', default=None)
    p.add_argument('--arena_width_cm', type=float, default=None)
    p.add_argument('--stride', type=int, default=_STRIDE,
                   help='Eye-frame sub-sampling (default: %(default)s, which '
                        'turns 60 fps into 30 fps output).')
    p.add_argument('--max_seconds', type=float, default=80.0)
    p.add_argument('--t_start', type=float, default=None,
                   help='Start time in recording seconds, instead of the '
                        'automatically chosen light block.')
    p.add_argument('--flow_gain', type=float, default=1.0,
                   help='Arrow length multiplier (default: %(default)s, i.e. '
                        'true angular length).')
    p.add_argument('--flow_clim', type=float, default=20.0,
                   help='Flow magnitude in deg/frame at the top of the arrow '
                        'colour scale (default: %(default)s).')
    p.add_argument('--flow_smooth', type=float, default=_FLOW_SMOOTH_PX,
                   help='Gaussian sigma in px for the difference field before '
                        'it is sampled on the arrow grid (default: '
                        '%(default)s, 0 turns it off).')
    p.add_argument('--preview', action='store_true',
                   help='Render one frame to PNG and stop.')
    args = p.parse_args()

    h5_path = args.h5_path
    if h5_path is None:
        h5_path = args.retino_npz.replace('_retinal_images.npz', '_preproc.h5')
        if not os.path.exists(h5_path):
            p.error(f'Could not derive --h5_path from --retino_npz '
                    f'(tried {h5_path}).')
        print(f'Using h5_path {h5_path}')

    show_fm_reconstruction(
        h5_path=h5_path,
        ego_npz_path=args.ego_npz,
        retino_npz_path=args.retino_npz,
        eye_path=args.eye_path,
        top_path=args.top_path,
        out_path=args.out_path,
        arena_width_cm=args.arena_width_cm,
        stride=args.stride,
        max_seconds=args.max_seconds,
        t_start_override=args.t_start,
        flow_gain=args.flow_gain,
        flow_clim=args.flow_clim,
        flow_smooth=args.flow_smooth,
        preview=args.preview,
    )


if __name__ == '__main__':
    main()


# python -m fm2p.show_fm_reconstruction --h5_path /home/dylan/Storage/freely_moving_data/V1PPC/cohort02_recordings/251016_DMM_DMM061_pos18/fm1/251016_DMM_DMM061_fm_01_preproc.h5 --ego_npz /home/dylan/Storage/freely_moving_data/V1PPC/cohort02_recordings/251016_DMM_DMM061_pos18/fm1/251016_DMM_DMM061_fm_01_retinal_images_egocentric.npz --retino_npz /home/dylan/Storage/freely_moving_data/V1PPC/cohort02_recordings/251016_DMM_DMM061_pos18/fm1/251016_DMM_DMM061_fm_01_retinal_images_retinocentric.npz

# python -m fm2p.show_fm_reconstruction --h5_path /home/dylan/Storage/freely_moving_data/V1PPC/cohort02_recordings/251016_DMM_DMM061_pos18/fm1/251016_DMM_DMM061_fm_01_preproc.h5 --ego_npz /home/dylan/Storage/freely_moving_data/V1PPC/cohort02_recordings/251016_DMM_DMM061_pos18/fm1/251016_DMM_DMM061_fm_01_retinal_images_ego_fixed_pupil.npz --retino_npz /home/dylan/Storage/freely_moving_data/V1PPC/cohort02_recordings/251016_DMM_DMM061_pos18/fm1/251016_DMM_DMM061_fm_01_retinal_images_retinocentric.npz