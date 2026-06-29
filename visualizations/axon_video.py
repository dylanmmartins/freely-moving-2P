

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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import tifffile
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.ndimage import gaussian_filter1d


DEFAULT_REC_DIR = (
    "/home/dylan/Storage/freely_moving_data/_LGN/"
    "250915_DMM_DMM052_lgnaxons/fm1"
)
DEFAULT_PREFIX = "250915_DMM_DMM052_fm_01"

EYE_FPS    = 60.0
STRIDE     = 8
OUTPUT_FPS = 30
ROLL_AVG   = 20
FIGSIZE    = (12, 8)
DPI        = 120
IMU_WIN_S  = 20.0

TOPDOWN_H  = 480
TOPDOWN_W  = round(480 * 2448 / 2048)

EYE_W      = 640
EYE_H      = 480

EYE_CROP_X0 = int(EYE_W * 0.25)
EYE_CROP_X1 = int(EYE_W * 0.75)
EYE_CROP_Y0 = int(EYE_H * 0.40)
EYE_CROP_Y1 = EYE_H

FIG_BG     = "k"
TRACE_BG   = "#0a0a0a"


def interp_short_gaps(x: np.ndarray, max_gap: int = 5) -> np.ndarray:

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


def nan_wrap(sig: np.ndarray, threshold: float = 180.0) -> np.ndarray:

    out   = sig.copy().astype(float)
    jumps = np.where(np.abs(np.diff(out)) > threshold)[0]
    out[jumps + 1] = np.nan
    return out


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

def load_data(h5_path: str) -> dict:
    data = {}
    with h5py.File(h5_path, "r") as f:
        for key in [
            "eyeT_trim", "twopT",
            "light_onsets", "dark_onsets",
            "theta", "phi",
            "head_x", "head_y", "head_yaw_deg",
            "pitch_twop_interp", "roll_twop_interp",
        ]:
            data[key] = f[key][:]
        data["eyeT_startInd"] = int(f["eyeT_startInd"][()])
    return data


def select_best_light_block(data: dict) -> dict:

    eyeT_trim    = data["eyeT_trim"]
    twopT        = data["twopT"]
    light_onsets = data["light_onsets"]
    dark_onsets  = data["dark_onsets"]
    startInd     = data["eyeT_startInd"]
    theta        = data["theta"][startInd : startInd + len(eyeT_trim)]
    phi          = data["phi"][startInd   : startInd + len(eyeT_trim)]
    head_x       = data["head_x"]
    head_y       = data["head_y"]

    best_idx, best_score = -1, -1.0
    for i in range(1, len(light_onsets)):
        lo  = int(light_onsets[i])
        nxt = dark_onsets[dark_onsets > lo]
        if len(nxt) == 0:
            continue
        nd      = int(nxt[0])
        t_start = twopT[lo]
        t_end   = twopT[nd]

        mask  = (eyeT_trim >= t_start) & (eyeT_trim <= t_end)
        n_eye = int(mask.sum())
        if n_eye == 0:
            continue
        eye_pct = 100.0 * int(np.sum(mask & np.isfinite(theta) & np.isfinite(phi))) / n_eye

        top_block = (np.isfinite(head_x[lo:nd]) & np.isfinite(head_y[lo:nd]))
        top_pct   = 100.0 * top_block.sum() / max(len(top_block), 1)

        score = (eye_pct + top_pct) / 2.0
        if score > best_score:
            best_score, best_idx = score, i

    if best_idx < 0:
        raise RuntimeError("No valid light block found.")

    lo      = int(light_onsets[best_idx])
    nd      = int(dark_onsets[dark_onsets > lo][0])
    t_start = float(twopT[lo])
    t_end   = float(twopT[nd])
    print(
        f"  Best light block: index={best_idx}, "
        f"2P [{lo}:{nd}] ({nd - lo} frames), "
        f"t=[{t_start:.1f}–{t_end:.1f}] s, score={best_score:.1f}%"
    )
    return {"twop_lo": lo, "twop_nd": nd, "t_start": t_start, "t_end": t_end}


def compute_output_indices(data: dict, block: dict) -> dict:

    eyeT_trim = data["eyeT_trim"]
    twopT     = data["twopT"]
    startInd  = data["eyeT_startInd"]
    t_start   = block["t_start"]
    t_end     = block["t_end"]

    eye_start    = int(np.searchsorted(eyeT_trim, t_start))
    eye_end      = int(np.searchsorted(eyeT_trim, t_end))
    eye_trim_idx = np.arange(eye_start, eye_end, STRIDE)
    eye_full_idx = eye_trim_idx + startInd
    times        = eyeT_trim[eye_trim_idx]

    twop_idx      = np.searchsorted(twopT, times).clip(0, len(twopT) - 1)
    twop_idx_prev = (twop_idx - 1).clip(0)
    closer_prev   = (np.abs(twopT[twop_idx_prev] - times)
                     < np.abs(twopT[twop_idx] - times))
    twop_idx      = np.where(closer_prev, twop_idx_prev, twop_idx)

    twop_lo        = block["twop_lo"]
    twop_nd        = block["twop_nd"]
    twop_block_idx = (twop_idx - twop_lo).clip(0, twop_nd - twop_lo - 1)

    return {
        "eye_full_idx":   eye_full_idx,
        "eye_trim_idx":   eye_trim_idx,
        "twop_idx":       twop_idx,
        "twop_block_idx": twop_block_idx,
        "times":          times,
        "n_frames":       len(eye_trim_idx),
    }


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


def preload_twop_block(tif_path: str, lo: int, nd: int,
                       extra: int = ROLL_AVG) -> tuple:

    start   = max(0, lo - extra)
    tif     = tifffile.TiffFile(tif_path)
    n_pages = len(tif.pages)
    end     = min(nd + extra, n_pages)
    block = np.stack([tif.pages[i].asarray() for i in range(start, end)])
    tif.close()
    return block, start


def _find_ffmpeg():
    for ff in ('/usr/bin/ffmpeg', 'ffmpeg'):
        try:
            out = subprocess.run([ff, '-encoders'], capture_output=True, text=True)
            if 'libx264' in out.stdout:
                return ff, ['-vcodec', 'libx264', '-preset', 'faster',
                            '-crf', '20', '-bf', '0']
            if 'libopenh264' in out.stdout:
                return ff, ['-vcodec', 'libopenh264', '-b:v', '8M', '-bf', '0']
            if 'libvpx' in out.stdout:
                return ff, ['-vcodec', 'libvpx', '-b:v', '8M']
        except FileNotFoundError:
            continue
    return 'ffmpeg', ['-vcodec', 'libx264', '-preset', 'faster', '-crf', '20', '-bf', '0']

_W: dict = {}


def worker_init(init_data: dict) -> None:

    global _W
    _W = init_data

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor=FIG_BG)

    gs = GridSpec(
        3, 2, figure=fig,
        width_ratios=[1, 2],
        height_ratios=[1, 1, 1],
        hspace=0.10, wspace=0.10,
        left=0.14, right=0.97, top=0.97, bottom=0.06,
    )

    ax_td  = fig.add_subplot(gs[0, 0])
    ax_eye = fig.add_subplot(gs[1, 0])
    ax_2p  = fig.add_subplot(gs[0:3, 1])

    gs_imu   = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[2, 0], hspace=0.08)
    ax_pitch = fig.add_subplot(gs_imu[0])
    ax_roll  = fig.add_subplot(gs_imu[1])
    ax_yaw   = fig.add_subplot(gs_imu[2])

    for ax in (ax_td, ax_eye, ax_2p):
        ax.set_facecolor(FIG_BG)
        ax.axis("off")

    for ax in (ax_pitch, ax_roll, ax_yaw):
        ax.set_facecolor(TRACE_BG)
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
        np.zeros((EYE_H, EYE_W), dtype=np.uint8),
        cmap="gray", vmin=0, vmax=300,
        aspect="equal", interpolation="nearest",
    )
    ax_eye.set_xlim(EYE_CROP_X0 - 0.5, EYE_CROP_X1 - 0.5)
    ax_eye.set_ylim(EYE_CROP_Y1 - 0.5, EYE_CROP_Y0 - 0.5)

    vmin_2p  = float(init_data["vmin_2p"])
    vmax_2p  = float(init_data["vmax_2p"])
    mean_img = init_data["mean_img"].astype(float)
    mean_norm = np.clip((mean_img - vmin_2p) / max(vmax_2p - vmin_2p, 1.0), 0, 1)
    im_2p = ax_2p.imshow(
        mean_norm, cmap="gray", vmin=0, vmax=1.5,
        aspect="equal", interpolation="nearest",
    )
    H2p, W2p = mean_img.shape
    ax_2p.set_xlim(0, W2p - 1)
    ax_2p.set_ylim(H2p - 1, 0)

    twopT   = init_data["twopT"]
    t_start = float(init_data["t_start"])
    blk_lo  = init_data["twop_lo"]
    blk_nd  = init_data["twop_nd"]

    fps_2p       = len(twopT) / float(twopT[-1])
    extra_frames = int(IMU_WIN_S / 2.0 * fps_2p)
    ext_lo       = max(0, blk_lo - extra_frames)

    tt_rel  = twopT[ext_lo:blk_nd] - t_start
    pitch_b = interp_short_gaps(init_data["pitch"][ext_lo:blk_nd].astype(float))
    roll_b  = interp_short_gaps(init_data["roll"][ext_lo:blk_nd].astype(float))

    yaw_raw = init_data["yaw"][ext_lo:blk_nd].copy().astype(float)
    yaw_b   = nan_wrap(interp_short_gaps(yaw_raw))

    ax_pitch.plot(tt_rel, pitch_b, color="#4a9eff", lw=1.5)
    ax_roll.plot( tt_rel, roll_b,  color="#4aff88", lw=1.5)
    ax_yaw.plot(  tt_rel, yaw_b,   color="#ffaa44", lw=1.5)

    ax_pitch.set_ylabel("Pitch",  color="0.6", fontsize=10)
    ax_roll.set_ylabel("Roll",    color="0.6", fontsize=10)
    ax_yaw.set_ylabel("Yaw",      color="0.6", fontsize=10)
    ax_yaw.set_xlabel("Time (s)", color="0.6", fontsize=10)
    ax_pitch.tick_params(labelbottom=False)
    ax_roll.tick_params(labelbottom=False)

    for ax in (ax_pitch, ax_roll, ax_yaw):
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: str(int(x)))
        )

    for ax, sig in [(ax_pitch, pitch_b), (ax_roll, roll_b), (ax_yaw, yaw_b)]:
        fin = sig[np.isfinite(sig)]
        p1, p99 = np.nanpercentile(fin, [1, 99]) if len(fin) else (-1, 1)
        mg = max(0.05 * abs(p99 - p1), 0.5)
        ax.set_ylim(p1 - mg, p99 + mg)

    for ax in (ax_pitch, ax_roll, ax_yaw):
        ax.set_xlim(-IMU_WIN_S / 2.0, IMU_WIN_S / 2.0)

    imu_cursors = [ax.axvline(0.0, color="w", lw=0.8, alpha=0.8)
                   for ax in (ax_pitch, ax_roll, ax_yaw)]

    time_txt = fig.text(
        0.50, 0.005, "", color="0.6", fontsize=11, ha="center", va="bottom",
    )

    _W["fig"]          = fig
    _W["im_td"]        = im_td
    _W["im_eye"]       = im_eye
    _W["im_2p"]        = im_2p
    _W["vmin_2p"]      = vmin_2p
    _W["vmax_2p"]      = vmax_2p
    _W["axes_imu"]     = [ax_pitch, ax_roll, ax_yaw]
    _W["imu_cursors"]  = imu_cursors
    _W["time_txt"]     = time_txt
    _W["FIG_H"]        = int(fig.get_figheight() * DPI)
    _W["FIG_W"]        = int(fig.get_figwidth()  * DPI)


def render_frame(out_idx: int) -> bytes:

    eye_frame    = _W["eye_frames"][out_idx]
    t_abs        = float(_W["times"][out_idx])
    t_rel        = t_abs - float(_W["t_start"])
    twop_abs_idx = int(_W["twop_idx"][out_idx])

    _W["im_eye"].set_data(subtract_band(eye_frame))

    top_bi = (twop_abs_idx - _W["twop_lo"])
    top_bi = max(0, min(top_bi, len(_W["top_frames"]) - 1))
    _W["im_td"].set_data(
        cv2.cvtColor(_W["top_frames"][top_bi], cv2.COLOR_BGR2RGB)
    )

    block  = _W["twop_block"]
    offset = _W["twop_block_offset"]
    idx_b  = twop_abs_idx - offset
    half   = ROLL_AVG // 2
    s      = max(0, idx_b - half)
    e      = min(len(block), idx_b + half + 1)
    twop_avg  = block[s:e].mean(axis=0).astype(float)
    twop_norm = np.clip(
        (twop_avg - _W["vmin_2p"]) / max(_W["vmax_2p"] - _W["vmin_2p"], 1.0),
        0, 1,
    )
    _W["im_2p"].set_data(twop_norm)

    for ax, cur in zip(_W["axes_imu"], _W["imu_cursors"]):
        ax.set_xlim(t_rel - IMU_WIN_S / 2.0, t_rel + IMU_WIN_S / 2.0)
        cur.set_xdata([t_rel, t_rel])

    _W["time_txt"].set_text(f"t = {t_rel:.2f} s")

    _W["fig"].canvas.draw()
    buf = _W["fig"].canvas.buffer_rgba()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(
        _W["FIG_H"], _W["FIG_W"], 4
    )
    return arr[:, :, :3].tobytes()


def main(rec_dir: str = DEFAULT_REC_DIR,
         prefix:  str = DEFAULT_PREFIX) -> None:

    h5_path     = os.path.join(rec_dir, f"{prefix}_preproc.h5")
    eye_path    = os.path.join(rec_dir, f"{prefix}_eyecam_deinter.avi")
    top_path    = os.path.join(rec_dir, "fm01_0001.mp4")
    tif_path    = os.path.join(rec_dir, "file_00001.tif")
    output_path = os.path.join(rec_dir, f"{prefix}_axon_diagnostic.mp4")

    print("Loading preproc data ...")
    t0   = time.time()
    data = load_data(h5_path)
    print(f"  done in {time.time() - t0:.1f}s")

    print("Selecting best light block ...")
    block  = select_best_light_block(data)
    lo, nd = block["twop_lo"], block["twop_nd"]

    twopT  = data["twopT"]
    nd_cap = int(np.searchsorted(twopT, block["t_start"] + 80.0))
    nd     = min(nd, nd_cap)
    block["twop_nd"] = nd
    block["t_end"]   = float(twopT[nd])

    print("Computing output frame indices ...")
    idx_info = compute_output_indices(data, block)
    n_frames = idx_info["n_frames"]
    print(f"  {n_frames} output frames  (stride={STRIDE}, output={OUTPUT_FPS} fps)")

    print("Pre-loading eye-camera frames ...")
    t0         = time.time()
    eye_frames = preload_eye_frames(eye_path, idx_info["eye_full_idx"])
    print(f"  {len(eye_frames)} frames in {time.time() - t0:.1f}s")

    print("Pre-loading topdown frames ...")
    t0         = time.time()
    top_frames = preload_topdown_frames(top_path, lo, nd)
    print(f"  {len(top_frames)} frames in {time.time() - t0:.1f}s")

    print("Pre-loading 2P tif block ...")
    t0 = time.time()
    twop_block, twop_offset = preload_twop_block(tif_path, lo, nd)
    print(
        f"  {len(twop_block)} frames in {time.time() - t0:.1f}s, "
        f"{twop_block.nbytes / 1e6:.0f} MB"
    )

    pad = ROLL_AVG
    mean_img = twop_block[pad:-pad].mean(axis=0).astype(np.float32) if len(twop_block) > 2 * pad else twop_block.mean(axis=0).astype(np.float32)

    vmin_2p = float(np.percentile(mean_img, 1.0))
    vmax_2p = float(np.percentile(mean_img, 99.9))
    print(f"  2P normalization: vmin={vmin_2p:.1f}, vmax={vmax_2p:.1f}")

    twopT = data["twopT"]

    head_yaw = data["head_yaw_deg"][:len(twopT)]

    init_data = {
        "eye_trim_idx":      idx_info["eye_trim_idx"],
        "twop_idx":          idx_info["twop_idx"],
        "times":             idx_info["times"],

        "eye_frames":        eye_frames,
        "top_frames":        top_frames,
        "twop_block":        twop_block,
        "twop_block_offset": twop_offset,

        "twopT":             twopT,
        "pitch":             data["pitch_twop_interp"],
        "roll":              data["roll_twop_interp"],
        "yaw":               head_yaw,

        "mean_img":          mean_img,
        "vmin_2p":           vmin_2p,
        "vmax_2p":           vmax_2p,
        "twop_lo":           lo,
        "twop_nd":           nd,
        "t_start":           block["t_start"],
    }

    _probe = plt.figure(figsize=FIGSIZE, dpi=DPI)
    FIG_H  = int(_probe.get_figheight() * DPI)
    FIG_W  = int(_probe.get_figwidth()  * DPI)
    plt.close(_probe)
    FIG_H += FIG_H % 2
    FIG_W += FIG_W % 2
    print(f"Output resolution: {FIG_W}×{FIG_H}  |  {n_frames} frames at {OUTPUT_FPS} fps")

    ffmpeg_bin, codec_args = _find_ffmpeg()
    ffmpeg_cmd = [
        ffmpeg_bin, "-y",
        "-f",       "rawvideo",
        "-vcodec",  "rawvideo",
        "-s",       f"{FIG_W}x{FIG_H}",
        "-pix_fmt", "rgb24",
        "-r",       str(OUTPUT_FPS),
        "-i",       "pipe:0",
        *codec_args,
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]
    print(f"Writing: {output_path}  (codec: {codec_args[1]})")
    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    n_workers = max(1, min(cpu_count() - 1, 8))
    print(f"Rendering with {n_workers} worker(s) ...")
    t0, n_written = time.time(), 0

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
                    elapsed = time.time() - t0
                    print(f"  {n_written}/{n_frames} frames  "
                          f"({n_written / elapsed:.1f} fps)")
    except BrokenPipeError:
        print("BrokenPipeError — ffmpeg stderr:")
        print(ffmpeg_proc.stderr.read().decode(errors="replace"))
        raise

    ffmpeg_proc.stdin.close()
    retcode = ffmpeg_proc.wait()
    elapsed = time.time() - t0
    if retcode != 0:
        print(f"WARNING: ffmpeg exited with code {retcode}:")
        print(ffmpeg_proc.stderr.read().decode(errors="replace")[-3000:])
    print(f"Done in {elapsed:.1f}s  ({n_frames / elapsed:.1f} frames/s)")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Axon diagnostic video for a freely-moving 2P recording."
    )
    parser.add_argument("--rec_dir", default=DEFAULT_REC_DIR,
                        help="Recording directory")
    parser.add_argument("--prefix",  default=DEFAULT_PREFIX,
                        help="File prefix (e.g. 250915_DMM_DMM052_fm_01)")
    args = parser.parse_args()
    main(rec_dir=args.rec_dir, prefix=args.prefix)
