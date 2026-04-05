#!/usr/bin/env python3
"""
Diagnostic video for a freely-moving 2P recording.

Layout  (GridSpec 3 x 6, height_ratios [1, 2, 2]):
  Top row  (row 0):
    cols 0-1  topdown camera + head-center dot + direction arrow
    cols 2-3  eye camera (band-subtracted) + ellipse fit overlay
    cols 4-5  IMU -- 3 stacked sub-panels (pitch / roll / yaw)
  Bottom   (rows 1-2):
    cols 0-2  2P tif (rolling average) + coloured ROI outlines
    cols 3-5  dF/F traces -- top-30 kurtosis cells (moving cursor)

Output: 4x real-time MP4 (stride-8 from 60 Hz eye cam -> 30 fps out).
Usage:
    python -m fm2p.diagnostic_video
    python -m fm2p.diagnostic_video --rec_dir /path/to/fm1
"""

import argparse
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
import numpy as np
import tifffile
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Ellipse
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kurtosis as scipy_kurtosis

DEFAULT_REC_DIR = (
    "/home/dylan/Storage/freely_moving_data/_V1PPC/"
    "cohort02_recordings/cohort02_recordings/"
    "251020_DMM_DMM056_pos08/fm1"
)
DEFAULT_PREFIX = "251020_DMM_DMM056_fm_01"

EYE_FPS     = 60.0
STRIDE      = 8
OUTPUT_FPS  = 30
N_CELLS     = 30
ROLL_AVG    = 5
FIGSIZE     = (22, 13)
DPI         = 80
IMU_WIN_S   = 20.0
TOPDOWN_W   = 640
TOPDOWN_H   = 480
EYE_W       = 640
EYE_H       = 480


def get_cell_colors(n: int) -> np.ndarray:
    """Return (n, 3) float32 RGB colours from tab20 + tab20b."""
    cmap1 = plt.cm.tab20(np.linspace(0, 1, 20))[:, :3]
    cmap2 = plt.cm.tab20b(np.linspace(0, 1, 20))[:, :3]
    palette = np.concatenate([cmap1, cmap2], axis=0)
    return palette[:n].astype(np.float32)


def subtract_band(frame_gray: np.ndarray) -> np.ndarray:

    left    = frame_gray[:, :10].mean(axis=1).astype(float)
    right   = frame_gray[:, -10:].mean(axis=1).astype(float)
    profile = gaussian_filter1d(0.5 * (left + right), sigma=15)
    corrected = frame_gray.astype(float) - profile[:, np.newaxis]
    corrected -= corrected.min()
    mx = corrected.max()
    if mx > 0:
        corrected = corrected / mx * 255.0
    return np.clip(corrected, 0, 255).astype(np.uint8)


def build_roi_overlay(
    cell_xs: list,
    cell_ys: list,
    colors: np.ndarray,
    H: int = 512,
    W: int = 512,
) -> np.ndarray:

    overlay = np.zeros((H, W, 4), dtype=np.float32)
    for i, (xs, ys) in enumerate(zip(cell_xs, cell_ys)):
        xi = xs.astype(int)
        yi = ys.astype(int)
        valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        overlay[yi[valid], xi[valid], :3] = colors[i]
        overlay[yi[valid], xi[valid],  3] = 1.0
    return overlay


def load_data(h5_path: str) -> dict:
    data = {}
    with h5py.File(h5_path, "r") as f:
        for key in [
            "eyeT_trim", "twopT",
            "light_onsets", "dark_onsets",
            "theta", "phi", "ellipse_phi", "longaxis", "shortaxis",
            "X0", "Y0",
            "head_x", "head_y", "head_yaw_deg",
            "pitch_twop_interp", "roll_twop_interp",
            "norm_dFF",
            "twop_mean_img",
        ]:
            data[key] = f[key][:]
        data["eyeT_startInd"] = int(f["eyeT_startInd"][()])

        # Cell ROI pixel coordinates
        n_cells = max(int(k) for k in f["cell_x_pix"].keys()) + 1
        data["cell_x_pix"] = [f[f"cell_x_pix/{i}"][:] for i in range(n_cells)]
        data["cell_y_pix"] = [f[f"cell_y_pix/{i}"][:] for i in range(n_cells)]
    return data


def select_best_light_block(data: dict) -> dict:

    eyeT_trim     = data["eyeT_trim"]
    twopT         = data["twopT"]
    light_onsets  = data["light_onsets"]
    dark_onsets   = data["dark_onsets"]
    startInd      = data["eyeT_startInd"]
    theta         = data["theta"][startInd : startInd + len(eyeT_trim)]
    phi           = data["phi"][startInd   : startInd + len(eyeT_trim)]

    best_idx = -1
    best_pct = -1.0
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
        good = np.sum(mask & ~np.isnan(theta) & ~np.isnan(phi))
        pct  = 100.0 * good / n_eye
        if pct > best_pct:
            best_pct = pct
            best_idx = i

    lo      = light_onsets[best_idx]
    nd      = dark_onsets[dark_onsets > lo][0]
    t_start = float(twopT[lo])
    t_end   = float(twopT[nd])
    print(
        f"  Best light block: index={best_idx}, "
        f"twop [{lo}:{nd}], t=[{t_start:.1f}-{t_end:.1f}]s, "
        f"eye-tracking={best_pct:.1f}%"
    )
    return {
        "block_idx": best_idx,
        "twop_lo":   int(lo),
        "twop_nd":   int(nd),
        "t_start":   t_start,
        "t_end":     t_end,
    }


def compute_output_indices(data: dict, block: dict) -> dict:

    eyeT_trim = data["eyeT_trim"]
    twopT     = data["twopT"]
    startInd  = data["eyeT_startInd"]
    t_start   = block["t_start"]
    t_end     = block["t_end"]

    eye_start = int(np.searchsorted(eyeT_trim, t_start))
    eye_end   = int(np.searchsorted(eyeT_trim, t_end))

    eye_trim_idx = np.arange(eye_start, eye_end, STRIDE)
    eye_full_idx = eye_trim_idx + startInd
    times        = eyeT_trim[eye_trim_idx]

    twop_idx      = np.searchsorted(twopT, times).clip(0, len(twopT) - 1)
    twop_idx_prev = (twop_idx - 1).clip(0, len(twopT) - 1)
    closer_prev   = np.abs(twopT[twop_idx_prev] - times) < np.abs(twopT[twop_idx] - times)
    twop_idx      = np.where(closer_prev, twop_idx_prev, twop_idx)

    twop_lo       = block["twop_lo"]
    twop_nd       = block["twop_nd"]
    twop_block_idx = (twop_idx - twop_lo).clip(0, twop_nd - twop_lo - 1)

    return {
        "eye_full_idx":   eye_full_idx,
        "eye_trim_idx":   eye_trim_idx,
        "twop_idx":       twop_idx,
        "twop_block_idx": twop_block_idx,
        "times":          times,
        "n_frames":       len(eye_trim_idx),
    }


def select_top_kurtosis_cells(
    norm_dff: np.ndarray, lo: int, nd: int, n: int = 30
) -> np.ndarray:
    block = norm_dff[:, lo:nd]
    kurts = scipy_kurtosis(block, axis=1, nan_policy="omit")
    kurts = np.where(np.isfinite(kurts), kurts, -np.inf)
    return np.argsort(kurts)[-n:][::-1]


def preload_eye_frames(cap_path: str, eye_full_idx: np.ndarray) -> np.ndarray:

    n      = len(eye_full_idx)
    frames = np.empty((n, EYE_H, EYE_W), dtype=np.uint8)
    cap    = cv2.VideoCapture(cap_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(eye_full_idx[0]))
    current = int(eye_full_idx[0])
    for i, target in enumerate(eye_full_idx.astype(int)):
        skip = target - current
        for _ in range(skip - 1):
            cap.read()
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames[i] = cv2.resize(gray, (EYE_W, EYE_H))
        current = target + 1
    cap.release()
    return frames


def preload_topdown_frames(cap_path: str, lo: int, nd: int) -> np.ndarray:

    n      = nd - lo
    frames = np.empty((n, TOPDOWN_H, TOPDOWN_W, 3), dtype=np.uint8)
    cap    = cv2.VideoCapture(cap_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    for i in range(n):
        ret, frame = cap.read()
        if ret:
            frames[i] = cv2.resize(frame, (TOPDOWN_W, TOPDOWN_H))
    cap.release()
    return frames


def preload_twop_block(
    tif_path: str, lo: int, nd: int, extra: int = 3
) -> tuple:

    start = max(0, lo - extra)
    end   = nd + extra
    tif   = tifffile.TiffFile(tif_path)
    block = np.stack([tif.pages[i].asarray() for i in range(start, end)])
    tif.close()
    return block, start


_W: dict = {}


def worker_init(init_data: dict) -> None:

    global _W
    _W = init_data

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor="k")
    gs  = GridSpec(
        3, 6, figure=fig,
        height_ratios=[1, 2, 2],
        hspace=0.12, wspace=0.12,
        left=0.04, right=0.97, top=0.97, bottom=0.04,
    )

    ax_td  = fig.add_subplot(gs[0, 0:2])
    ax_eye = fig.add_subplot(gs[0, 2:4])

    gs_imu   = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0, 4:6], hspace=0.08)
    ax_pitch = fig.add_subplot(gs_imu[0])
    ax_roll  = fig.add_subplot(gs_imu[1])
    ax_yaw   = fig.add_subplot(gs_imu[2])

    ax_2p  = fig.add_subplot(gs[1:3, 0:3])
    ax_dff = fig.add_subplot(gs[1:3, 3:6])

    for ax in (ax_td, ax_eye, ax_2p):
        ax.set_facecolor("k")
        ax.axis("off")

    for ax in (ax_pitch, ax_roll, ax_yaw, ax_dff):
        ax.set_facecolor("#0a0a0a")
        ax.tick_params(colors="0.6", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("0.3")

    im_td = ax_td.imshow(
        np.zeros((TOPDOWN_H, TOPDOWN_W, 3), dtype=np.uint8),
        aspect="auto", interpolation="nearest",
    )
    head_dot, = ax_td.plot([], [], "o", color="#ff6666", ms=8, zorder=5)
    head_arr, = ax_td.plot([], [], "-",  color="#ffcc66", lw=2,  zorder=5)
    ax_td.set_xlim(0, TOPDOWN_W - 1)
    ax_td.set_ylim(TOPDOWN_H - 1, 0)
    ax_td.set_title("Topdown", color="0.7", fontsize=8, pad=2)

    im_eye = ax_eye.imshow(
        np.zeros((EYE_H, EYE_W), dtype=np.uint8),
        cmap="gray", vmin=0, vmax=255,
        aspect="auto", interpolation="nearest",
    )
    ell_patch = Ellipse(
        xy=(EYE_W // 2, EYE_H // 2), width=40, height=20, angle=0,
        fill=False, edgecolor="yellow", linewidth=1.5,
    )
    ax_eye.add_patch(ell_patch)
    ax_eye.set_xlim(0, EYE_W - 1)
    ax_eye.set_ylim(EYE_H - 1, 0)
    ax_eye.set_title("Eye camera", color="0.7", fontsize=8, pad=2)

    mean_img = init_data["mean_img"].astype(float)
    lo_p, hi_p = np.percentile(mean_img, [1, 99])
    mean_img_norm = np.clip((mean_img - lo_p) / max(hi_p - lo_p, 1.0), 0, 1)
    im_2p = ax_2p.imshow(
        mean_img_norm, cmap="gray", vmin=0, vmax=1,
        aspect="equal", interpolation="nearest",
    )
    ax_2p.imshow(
        init_data["roi_overlay"],
        aspect="equal", interpolation="nearest",
    )
    ax_2p.set_xlim(0, 511)
    ax_2p.set_ylim(511, 0)
    ax_2p.set_title("2P (rolling avg)", color="0.7", fontsize=8, pad=2)

    twopT   = init_data["twopT"]
    blk_lo  = init_data["twop_lo"]
    blk_nd  = init_data["twop_nd"]
    tt      = twopT[blk_lo:blk_nd]
    pitch_b = init_data["pitch"][blk_lo:blk_nd]
    roll_b  = init_data["roll"][blk_lo:blk_nd]
    yaw_b   = init_data["yaw"][blk_lo:blk_nd]

    ax_pitch.plot(tt, pitch_b, color="#4a9eff", lw=0.6)
    ax_roll.plot( tt, roll_b,  color="#4aff88", lw=0.6)
    ax_yaw.plot(  tt, yaw_b,   color="#ffaa44", lw=0.6)

    ax_pitch.set_ylabel("Pitch", color="0.6", fontsize=7)
    ax_roll.set_ylabel("Roll",   color="0.6", fontsize=7)
    ax_yaw.set_ylabel("Yaw",     color="0.6", fontsize=7)
    ax_yaw.set_xlabel("Time (s)", color="0.6", fontsize=7)
    ax_pitch.set_title("IMU", color="0.7", fontsize=8, pad=2)

    for ax, sig in [(ax_pitch, pitch_b), (ax_roll, roll_b), (ax_yaw, yaw_b)]:
        ax.set_xlim(tt[0], tt[0] + IMU_WIN_S)
        p2, p98 = np.nanpercentile(sig, [1, 99])
        mg = max(0.05 * (p98 - p2), 0.5)
        ax.set_ylim(p2 - mg, p98 + mg)

    imu_cursors = [ax.axvline(tt[0], color="w", lw=0.8, alpha=0.8)
                   for ax in (ax_pitch, ax_roll, ax_yaw)]

    colors     = init_data["cell_colors"]    # (N_CELLS, 3)
    dff_block  = init_data["dff_block"]      # (N_CELLS, n_block_frames)
    n_c        = dff_block.shape[0]

    dff_traces = np.empty_like(dff_block, dtype=float)
    for i in range(n_c):
        tr = dff_block[i].astype(float)
        p2, p98 = np.nanpercentile(tr, [2, 98])
        rng = p98 - p2
        dff_traces[i] = (tr - p2) / (rng if rng > 0 else 1.0)

    dff_tt = tt

    for i in range(n_c - 1, -1, -1):
        offset = n_c - 1 - i
        ax_dff.plot(dff_tt, dff_traces[i] * 0.9 + offset,
                    color=colors[i], lw=0.5, alpha=0.85)

    ax_dff.set_xlim(tt[0], tt[-1])
    ax_dff.set_ylim(-0.5, n_c + 0.5)
    ax_dff.set_yticks([])
    ax_dff.set_xlabel("Time (s)", color="0.6", fontsize=7)
    ax_dff.set_title(f"dF/F (top-{n_c} kurtosis cells)",
                     color="0.7", fontsize=8, pad=2)

    dff_cursor = ax_dff.axvline(tt[0], color="w", lw=0.9, alpha=0.9)

    time_txt = fig.text(
        0.50, 0.005, "", color="0.6", fontsize=8, ha="center", va="bottom",
    )

    _W["fig"]          = fig
    _W["im_td"]        = im_td
    _W["head_dot"]     = head_dot
    _W["head_arr"]     = head_arr
    _W["im_eye"]       = im_eye
    _W["ell_patch"]    = ell_patch
    _W["im_2p"]        = im_2p
    _W["imu_cursors"]  = imu_cursors
    _W["axes_imu"]     = [ax_pitch, ax_roll, ax_yaw]
    _W["dff_cursor"]   = dff_cursor
    _W["time_txt"]     = time_txt
    _W["twopT_block"]  = tt
    _W["FIG_H"]        = int(fig.get_figheight() * DPI)
    _W["FIG_W"]        = int(fig.get_figwidth()  * DPI)


def render_frame(out_idx: int) -> bytes:

    eye_frame    = _W["eye_frames"][out_idx]
    t            = float(_W["times"][out_idx])
    twop_abs_idx = int(_W["twop_idx"][out_idx])
    eye_trim_i   = int(_W["eye_trim_idx"][out_idx])

    twop_lo = _W["twop_lo"]
    top_bi  = (twop_abs_idx - twop_lo)
    top_bi  = max(0, min(top_bi, len(_W["top_frames"]) - 1))

    eye_corr = subtract_band(eye_frame)
    _W["im_eye"].set_data(eye_corr)

    full_i = eye_trim_i + _W["eyeT_startInd"]
    x0  = float(_W["X0"][full_i])
    y0  = float(_W["Y0"][full_i])
    la  = float(_W["longaxis"][full_i])
    sa  = float(_W["shortaxis"][full_i])
    phi = float(_W["ellipse_phi"][full_i])
    if np.isfinite(x0 + y0 + la + sa + phi) and la > 0 and sa > 0:
        _W["ell_patch"].set_center((x0, y0))
        _W["ell_patch"].width  = 2.0 * la
        _W["ell_patch"].height = 2.0 * sa
        _W["ell_patch"].angle  = np.degrees(phi)
        _W["ell_patch"].set_visible(True)
    else:
        _W["ell_patch"].set_visible(False)

    top_frame = _W["top_frames"][top_bi]
    _W["im_td"].set_data(cv2.cvtColor(top_frame, cv2.COLOR_BGR2RGB))

    hx  = float(_W["head_x"][twop_abs_idx])
    hy  = float(_W["head_y"][twop_abs_idx])
    ang = float(_W["head_yaw_deg"][twop_abs_idx])
    if np.isfinite(hx + hy + ang):
        cx  = hx * (TOPDOWN_W / 2448.0)
        cy  = hy * (TOPDOWN_H / 2048.0)
        rad = np.radians(ang)
        L   = 30
        ex  = cx + L * np.cos(rad)
        ey  = cy - L * np.sin(rad)
        _W["head_dot"].set_data([cx], [cy])
        _W["head_arr"].set_data([cx, ex], [cy, ey])
    else:
        _W["head_dot"].set_data([], [])
        _W["head_arr"].set_data([], [])

    block  = _W["twop_block"]
    offset = _W["twop_block_offset"]
    idx_b  = twop_abs_idx - offset
    half   = ROLL_AVG // 2
    s      = max(0, idx_b - half)
    e      = min(len(block), idx_b + half + 1)
    twop_avg = block[s:e].mean(axis=0).astype(float)

    lo_p, hi_p = np.percentile(twop_avg, [2, 98])
    twop_norm  = np.clip((twop_avg - lo_p) / max(hi_p - lo_p, 1.0), 0, 1)
    twop_rgb   = plt.cm.gray(twop_norm)[:, :, :3].astype(np.float32)
    roi        = _W["roi_overlay"]
    alpha      = roi[:, :, 3:4]
    composite  = twop_rgb * (1.0 - alpha * 0.65) + roi[:, :, :3] * alpha * 0.65
    _W["im_2p"].set_data(np.clip(composite, 0, 1))

    for ax, cur in zip(_W["axes_imu"], _W["imu_cursors"]):
        ax.set_xlim(t - IMU_WIN_S / 2.0, t + IMU_WIN_S / 2.0)
        cur.set_xdata([t, t])

    _W["dff_cursor"].set_xdata([t, t])

    elapsed = t - _W["t_start"]
    _W["time_txt"].set_text(f"t = {elapsed:.2f} s  (block-relative)")

    _W["fig"].canvas.draw()
    buf = _W["fig"].canvas.buffer_rgba()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(
        _W["FIG_H"], _W["FIG_W"], 4
    )
    return arr[:, :, :3].tobytes()


def main(rec_dir: str = DEFAULT_REC_DIR, prefix: str = DEFAULT_PREFIX) -> None:
    h5_path     = os.path.join(rec_dir, f"{prefix}_preproc.h5")
    eye_path    = os.path.join(rec_dir, f"{prefix}_eyecam_deinter.avi")
    top_path    = os.path.join(rec_dir, "fm1_0001.mp4")
    tif_path    = os.path.join(rec_dir, "file_00001.tif")
    output_path = os.path.join(rec_dir, f"{prefix}_diagnostic.mp4")

    print("Loading preproc data ...")
    t0   = time.time()
    data = load_data(h5_path)
    print(f"  done in {time.time()-t0:.1f}s")

    print("Selecting best light block ...")
    block = select_best_light_block(data)
    lo, nd = block["twop_lo"], block["twop_nd"]

    print("Computing output frame indices ...")
    idx_info = compute_output_indices(data, block)
    n_frames = idx_info["n_frames"]
    print(f"  {n_frames} output frames  (stride={STRIDE}, fps={OUTPUT_FPS})")

    print("Selecting top-kurtosis cells ...")
    top_cells = select_top_kurtosis_cells(data["norm_dFF"], lo, nd, N_CELLS)
    dff_block = data["norm_dFF"][top_cells, lo:nd].astype(np.float32)
    colors    = get_cell_colors(N_CELLS)

    print("Building ROI overlay ...")
    cell_xs = [data["cell_x_pix"][i] for i in top_cells]
    cell_ys = [data["cell_y_pix"][i] for i in top_cells]
    roi_overlay = build_roi_overlay(cell_xs, cell_ys, colors)

    print("Pre-loading eye-camera frames ...")
    t0         = time.time()
    eye_frames = preload_eye_frames(eye_path, idx_info["eye_full_idx"])
    print(f"  {len(eye_frames)} frames in {time.time()-t0:.1f}s")

    print("Pre-loading topdown frames ...")
    t0         = time.time()
    top_frames = preload_topdown_frames(top_path, lo, nd)
    print(f"  {len(top_frames)} frames in {time.time()-t0:.1f}s")

    print("Pre-loading 2P tif block ...")
    t0 = time.time()
    twop_block, twop_offset = preload_twop_block(tif_path, lo, nd)
    print(
        f"  {len(twop_block)} frames in {time.time()-t0:.1f}s, "
        f"{twop_block.nbytes/1e6:.0f} MB"
    )

    init_data = {

        "eye_trim_idx":      idx_info["eye_trim_idx"],
        "twop_idx":          idx_info["twop_idx"],
        "twop_block_idx":    idx_info["twop_block_idx"],
        "times":             idx_info["times"],

        "eye_frames":        eye_frames,
        "top_frames":        top_frames,
        "twop_block":        twop_block,
        "twop_block_offset": twop_offset,

        "twopT":             data["twopT"],
        "eyeT_startInd":     data["eyeT_startInd"],
        "X0":                data["X0"],
        "Y0":                data["Y0"],
        "longaxis":          data["longaxis"],
        "shortaxis":         data["shortaxis"],
        "ellipse_phi":       data["ellipse_phi"],
        "head_x":            data["head_x"],
        "head_y":            data["head_y"],
        "head_yaw_deg":      data["head_yaw_deg"],
        "pitch":             data["pitch_twop_interp"],
        "roll":              data["roll_twop_interp"],
        "yaw":               data["head_yaw_deg"][: len(data["twopT"])],

        "mean_img":          data["twop_mean_img"].astype(np.float32),
        "roi_overlay":       roi_overlay,
        "top_cell_idx":      top_cells,
        "cell_colors":       colors,
        "dff_block":         dff_block,

        "twop_lo":           lo,
        "twop_nd":           nd,
        "t_start":           block["t_start"],
    }


    _fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    FIG_H = int(_fig.get_figheight() * DPI)
    FIG_W = int(_fig.get_figwidth()  * DPI)
    plt.close(_fig)

    FIG_H = FIG_H + (FIG_H % 2)
    FIG_W = FIG_W + (FIG_W % 2)
    print(f"Output resolution: {FIG_W}x{FIG_H}  |  {n_frames} frames at {OUTPUT_FPS} fps")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f",       "rawvideo",
        "-vcodec",  "rawvideo",
        "-s",       f"{FIG_W}x{FIG_H}",
        "-pix_fmt", "rgb24",
        "-r",       str(OUTPUT_FPS),
        "-i",       "pipe:0",
        "-vcodec",  "libx264",
        "-preset",  "faster",
        "-crf",     "20",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]
    print(f"Writing: {output_path}")
    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    n_workers = max(1, min(cpu_count() - 1, 8))
    print(f"Rendering with {n_workers} worker(s) ...")
    t0 = time.time()

    n_written = 0
    try:
        with Pool(
            processes=n_workers,
            initializer=worker_init,
            initargs=(init_data,),
        ) as pool:
            for frame_bytes in pool.imap(render_frame, range(n_frames), chunksize=4):
                ffmpeg_proc.stdin.write(frame_bytes)
                n_written += 1
                if n_written % 100 == 0:
                    print(f"  {n_written}/{n_frames} frames written")
    except BrokenPipeError:
        print(f"BrokenPipeError after {n_written} frames — ffmpeg stderr:")
        print(ffmpeg_proc.stderr.read().decode(errors="replace"))
        raise

    ffmpeg_proc.stdin.close()
    retcode = ffmpeg_proc.wait()
    stderr_out = ffmpeg_proc.stderr.read().decode(errors="replace")
    elapsed = time.time() - t0
    if retcode != 0:
        print(f"WARNING: ffmpeg exited with code {retcode}")
        if stderr_out:
            print(stderr_out[-3000:])  # last 3000 chars
    print(f"Done in {elapsed:.1f}s  ({n_frames/elapsed:.1f} frames/s)")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Diagnostic video for fm2p recording.")
    parser.add_argument("--rec_dir", default=DEFAULT_REC_DIR)
    parser.add_argument("--prefix",  default=DEFAULT_PREFIX)
    args = parser.parse_args()
    main(rec_dir=args.rec_dir, prefix=args.prefix)
