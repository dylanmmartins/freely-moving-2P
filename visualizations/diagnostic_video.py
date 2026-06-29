


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
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kurtosis as scipy_kurtosis

from fm2p.utils.helper import interp_short_gaps

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
ROLL_AVG    = 20
FIGSIZE     = (16, 9)
DPI         = 120
IMU_WIN_S   = 20.0
TOPDOWN_W   = 640
TOPDOWN_H   = 480
EYE_W       = 640
EYE_H       = 480

EYE_CROP_R0 = int(EYE_H * 0.20)
EYE_CROP_R1 = int(EYE_H * 0.70)
EYE_CROP_C1 = int(EYE_W * 0.60)
CROP_EYE_H  = EYE_CROP_R1 - EYE_CROP_R0
CROP_EYE_W  = EYE_CROP_C1


def get_cell_colors(n: int) -> np.ndarray:

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

    hx  = data["head_x"].astype(float)
    hy  = data["head_y"].astype(float)
    spd = np.concatenate([[np.nan], np.sqrt(np.diff(hx)**2 + np.diff(hy)**2)])
    spd_thresh = np.nanpercentile(spd, 25)

    best_idx    = -1
    best_score  = -1.0
    best_eye    = 0.0
    best_active = 0.0
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
            best_score  = score
            best_idx    = i
            best_eye    = eye_pct
            best_active = active_pct

    lo      = light_onsets[best_idx]
    nd      = dark_onsets[dark_onsets > lo][0]
    t_start = float(twopT[lo])
    t_end   = float(twopT[nd])
    print(
        f"  Best light block: index={best_idx}, "
        f"twop [{lo}:{nd}], t=[{t_start:.1f}-{t_end:.1f}]s, "
        f"eye-tracking={best_eye:.1f}%  active={best_active:.1f}%"
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
        height_ratios=[2, 1, 1],
        hspace=0.35, wspace=0.35,
        left=0.06, right=0.97, top=0.97, bottom=0.05,
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
        ax.tick_params(colors="0.6", labelsize=10)
        for sp in ax.spines.values():
            sp.set_color("0.3")

    im_td = ax_td.imshow(
        np.zeros((TOPDOWN_H, TOPDOWN_W, 3), dtype=np.uint8),
        aspect="equal", interpolation="nearest",
    )
    ax_td.set_xlim(0, TOPDOWN_W - 1)
    ax_td.set_ylim(TOPDOWN_H - 1, 0)

    im_eye = ax_eye.imshow(
        np.zeros((CROP_EYE_H, CROP_EYE_W), dtype=np.uint8),
        cmap="gray", vmin=0, vmax=260,
        aspect="equal", interpolation="nearest",
    )
    ax_eye.set_xlim(0, CROP_EYE_W - 1)
    ax_eye.set_ylim(CROP_EYE_H - 1, 0)

    mean_img = init_data["mean_img"].astype(float)
    vmin_2p  = float(np.percentile(mean_img, 1))
    vmax_2p  = float(np.percentile(mean_img, 100)) * 1.1
    im_2p = ax_2p.imshow(
        mean_img, cmap="gray", vmin=vmin_2p, vmax=vmax_2p,
        aspect="equal", interpolation="nearest",
    )
    ax_2p.set_xlim(0, 511)
    ax_2p.set_ylim(511, 0)

    twopT_full  = init_data["twopT"]
    blk_lo      = init_data["twop_lo"]
    blk_nd      = init_data["twop_nd"]
    t_start_abs = float(twopT_full[blk_lo])
    tt_full     = twopT_full - t_start_abs

    n_tp      = len(twopT_full)
    _sigma    = 0.5
    pitch_f   = gaussian_filter1d(
        interp_short_gaps(init_data["pitch"][:n_tp].astype(float)), _sigma)
    roll_f    = gaussian_filter1d(
        interp_short_gaps(init_data["roll"][:n_tp].astype(float)),  _sigma)
    yaw_isg   = interp_short_gaps(init_data["yaw"][:n_tp].astype(float))
    yaw_sm    = gaussian_filter1d(yaw_isg, _sigma)

    wrap_inds = np.where(np.abs(np.diff(yaw_isg)) > 180)[0]
    yaw_f     = yaw_sm.copy()
    yaw_f[wrap_inds]     = np.nan
    yaw_f[wrap_inds + 1] = np.nan

    ax_pitch.plot(tt_full, pitch_f, color="#4a9eff", lw=1.5)
    ax_roll.plot( tt_full, roll_f,  color="#4aff88", lw=1.5)
    ax_yaw.plot(  tt_full, yaw_f,   color="#ffaa44", lw=1.5)

    ax_pitch.set_ylabel("Pitch", color="0.6", fontsize=10)
    ax_roll.set_ylabel("Roll",   color="0.6", fontsize=10)
    ax_yaw.set_ylabel("Yaw",     color="0.6", fontsize=10)
    for ax in (ax_pitch, ax_roll, ax_yaw):
        ax.yaxis.set_label_coords(-0.08, 0.5)
    ax_yaw.set_xlabel("Time (s)", color="0.6", fontsize=10)

    pitch_b = pitch_f[blk_lo:blk_nd]
    roll_b  = roll_f[blk_lo:blk_nd]
    yaw_b   = yaw_f[blk_lo:blk_nd]
    for ax, sig in [(ax_pitch, pitch_b), (ax_roll, roll_b), (ax_yaw, yaw_b)]:
        ax.set_xlim(-IMU_WIN_S / 2.0, IMU_WIN_S / 2.0)
        p2, p98 = np.nanpercentile(sig, [1, 99])
        mg = max(0.05 * (p98 - p2), 0.5)
        ax.set_ylim(p2 - mg, p98 + mg)

    imu_cursors = [ax.axvline(0, color="w", lw=0.8, alpha=0.8)
                   for ax in (ax_pitch, ax_roll, ax_yaw)]

    colors    = init_data["cell_colors"]
    dff_block = init_data["dff_block"]
    dff_full  = init_data["dff_full"]
    n_c       = dff_block.shape[0]
    n_dff     = min(dff_full.shape[1], n_tp)
    tt_dff    = tt_full[:n_dff]

    for i in range(n_c - 1, -1, -1):
        row_offset = n_c - 1 - i
        tr_blk  = dff_block[i].astype(float)
        p2, p98 = np.nanpercentile(tr_blk, [2, 98])
        rng     = max(p98 - p2, 1e-6)
        tr_norm = (dff_full[i, :n_dff].astype(float) - p2) / rng
        ax_dff.plot(tt_dff, tr_norm * 0.9 + row_offset,
                    color=colors[i], lw=1.2, alpha=0.85)

    ax_dff.set_xlim(-IMU_WIN_S / 2.0, IMU_WIN_S / 2.0)
    ax_dff.set_ylim(-0.5, n_c + 0.5)
    ax_dff.set_yticks([])
    ax_dff.set_xlabel("Time (s)", color="0.6", fontsize=10)
    ax_dff.set_ylabel(r"$\Delta$F/F", color="0.6", fontsize=10)

    dff_cursor = ax_dff.axvline(0, color="w", lw=0.9, alpha=0.9)

    time_txt = fig.text(
        0.50, 0.005, "", color="0.6", fontsize=11, ha="center", va="bottom",
    )

    _W["fig"]          = fig
    _W["im_td"]        = im_td
    _W["im_eye"]       = im_eye
    _W["im_2p"]        = im_2p
    _W["axes_imu"]     = [ax_pitch, ax_roll, ax_yaw]
    _W["imu_cursors"]  = imu_cursors
    _W["ax_dff"]       = ax_dff
    _W["dff_cursor"]   = dff_cursor
    _W["time_txt"]     = time_txt
    _W["t_start_abs"]  = t_start_abs
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

    eye_corr   = subtract_band(eye_frame)
    eye_cropped = eye_corr[EYE_CROP_R0:EYE_CROP_R1, :EYE_CROP_C1]
    _W["im_eye"].set_data(eye_cropped)

    top_frame = _W["top_frames"][top_bi]
    _W["im_td"].set_data(cv2.cvtColor(top_frame, cv2.COLOR_BGR2RGB))

    block  = _W["twop_block"]
    offset = _W["twop_block_offset"]
    idx_b  = twop_abs_idx - offset
    half   = ROLL_AVG // 2
    s      = max(0, idx_b - half)
    e      = min(len(block), idx_b + half + 1)
    twop_avg = block[s:e].mean(axis=0).astype(float)
    _W["im_2p"].set_data(twop_avg)

    t_rel = t - _W["t_start_abs"]

    for ax, cur in zip(_W["axes_imu"], _W["imu_cursors"]):
        ax.set_xlim(t_rel - IMU_WIN_S / 2.0, t_rel + IMU_WIN_S / 2.0)
        cur.set_xdata([t_rel, t_rel])

    _W["dff_cursor"].set_xdata([t_rel, t_rel])
    _W["ax_dff"].set_xlim(t_rel - IMU_WIN_S / 2.0, t_rel + IMU_WIN_S / 2.0)

    _W["time_txt"].set_text(f"t = {t_rel:.2f} s")

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
        "dff_full":          data["norm_dFF"][top_cells, :].astype(np.float32),

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
        "-vcodec",  "libopenh264",
        "-b:v",     "4M",
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
            print(stderr_out[-3000:])
    print(f"Done in {elapsed:.1f}s  ({n_frames/elapsed:.1f} frames/s)")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Diagnostic video for fm2p recording.")
    parser.add_argument("--rec_dir", default=DEFAULT_REC_DIR)
    parser.add_argument("--prefix",  default=DEFAULT_PREFIX)
    args = parser.parse_args()
    main(rec_dir=args.rec_dir, prefix=args.prefix)
