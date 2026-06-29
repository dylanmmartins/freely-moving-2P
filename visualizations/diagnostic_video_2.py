
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d

SN_PATH = (
    "/home/dylan/Fast0/Dropbox/dyns_visnav_talks_feb2026/"
    "251013_DMM_DMM056_sn_01_eyecam_deinter.avi"
)
FM_PATH = (
    "/home/dylan/Fast0/Dropbox/dyns_visnav_talks_feb2026/"
    "251013_DMM_DMM056_fm_05_eyecam_deinter.avi"
)
OUTPUT_PATH = (
    "/home/dylan/Fast0/Dropbox/dyns_visnav_talks_feb2026/"
    "sn_vs_fm_eye_comparison.mp4"
)

VIDEO_W    = 640
VIDEO_H    = 480
TITLE_H    = 72
INPUT_FPS  = 60.0
SPEED      = 4
OUTPUT_FPS = INPUT_FPS * SPEED
DURATION_S = 120.0
N_FRAMES   = int(DURATION_S * INPUT_FPS)
START_PCT  = 0.30

START_OFFSET_S = -120.0

CANVAS_W = VIDEO_W * 2
CANVAS_H = VIDEO_H + TITLE_H

ARIAL_BOLD_PATH = "/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf"
FONT_SIZE_PT    = 36

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


def subtract_band_batch(frames: np.ndarray) -> np.ndarray:

    f = frames.astype(np.float32)
    left  = f[:, :, :10].mean(axis=2)   # (N, H)
    right = f[:, :, -10:].mean(axis=2)  # (N, H)
    profile = gaussian_filter1d(0.5 * (left + right), sigma=15, axis=1)  # (N, H)
    corrected = f - profile[:, :, np.newaxis]
    corrected -= corrected.min(axis=(1, 2), keepdims=True)
    mx = corrected.max(axis=(1, 2), keepdims=True)
    mx = np.where(mx > 0, mx, 1.0)
    corrected = corrected / mx * 255.0
    return np.clip(corrected, 0, 255).astype(np.uint8)


def load_video_frames(path: str, start_pct: float, n_frames: int,
                      label: str) -> np.ndarray:

    cap   = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or INPUT_FPS
    start = int(total * start_pct) + int(START_OFFSET_S * fps)
    start = max(0, start)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frames = np.empty((n_frames, VIDEO_H, VIDEO_W), dtype=np.uint8)
    t0 = time.time()
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"  [{label}] video ended early at frame {i}")
            frames = frames[:i]
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape != (VIDEO_H, VIDEO_W):
            gray = cv2.resize(gray, (VIDEO_W, VIDEO_H))
        frames[i] = gray
        if (i + 1) % 1000 == 0:
            print(f"  [{label}] loaded {i+1}/{n_frames} frames "
                  f"({time.time()-t0:.1f}s)")
    cap.release()
    print(f"  [{label}] done: {len(frames)} frames in {time.time()-t0:.1f}s")
    return frames


def make_title_bar() -> np.ndarray:

    img  = Image.new("RGB", (CANVAS_W, TITLE_H), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(ARIAL_BOLD_PATH, FONT_SIZE_PT)
    except OSError:
        font = ImageFont.load_default()

    for text, x_offset in [("head-fixed", 0), ("freely moving", VIDEO_W)]:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw   = bbox[2] - bbox[0]
        th   = bbox[3] - bbox[1]
        x    = x_offset + (VIDEO_W - tw) // 2 - bbox[0]
        y    = (TITLE_H - th) // 2 - bbox[1]
        draw.text((x, y), text, font=font, fill=(255, 255, 255))

    bar_rgb = np.array(img)
    return bar_rgb[:, :, ::-1].copy()


def best_encoder() -> tuple[str, list[str]]:

    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        if "h264_nvenc" in out:
            return "h264_nvenc", ["-preset", "p1", "-rc", "vbr", "-cq", "20"]
        if "h264_videotoolbox" in out:
            return "h264_videotoolbox", ["-q:v", "50"]
    except Exception:
        pass
    return "libx264", ["-preset", "ultrafast", "-crf", "20"]



def main() -> None:
    t_global = time.time()

    print(f"Loading {N_FRAMES} frames per video ({DURATION_S:.0f}s @ {INPUT_FPS:.0f}fps) ...")

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_sn = ex.submit(load_video_frames, SN_PATH, START_PCT, N_FRAMES, "sn_01")
        fut_fm = ex.submit(load_video_frames, FM_PATH, START_PCT, N_FRAMES, "fm_05")
        sn_raw = fut_sn.result()
        fm_raw = fut_fm.result()

    n_frames = min(len(sn_raw), len(fm_raw))
    sn_raw   = sn_raw[:n_frames]
    fm_raw   = fm_raw[:n_frames]
    print(f"Loaded {n_frames} frames each in {time.time()-t_global:.1f}s total")

    print("Applying band subtraction ...")
    t0 = time.time()
    sn_corr = subtract_band_batch(sn_raw)
    fm_corr = subtract_band_batch(fm_raw)
    del sn_raw, fm_raw
    print(f"  done in {time.time()-t0:.1f}s")

    title_bar = make_title_bar()

    encoder, enc_flags = best_encoder()
    print(f"Encoder: {encoder}  |  output {CANVAS_W}x{CANVAS_H} @ {int(OUTPUT_FPS)}fps")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{CANVAS_W}x{CANVAS_H}",
        "-pix_fmt", "bgr24",
        "-r", str(int(OUTPUT_FPS)),
        "-i", "pipe:0",
        "-vcodec", encoder,
        *enc_flags,
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        OUTPUT_PATH,
    ]
    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:TITLE_H] = title_bar

    print(f"Piping {n_frames} frames to ffmpeg ...")
    t0 = time.time()
    n_written = 0
    try:
        for i in range(n_frames):

            canvas[TITLE_H:, :VIDEO_W, 0] = sn_corr[i]
            canvas[TITLE_H:, :VIDEO_W, 1] = sn_corr[i]
            canvas[TITLE_H:, :VIDEO_W, 2] = sn_corr[i]
            canvas[TITLE_H:, VIDEO_W:, 0] = fm_corr[i]
            canvas[TITLE_H:, VIDEO_W:, 1] = fm_corr[i]
            canvas[TITLE_H:, VIDEO_W:, 2] = fm_corr[i]
            ffmpeg_proc.stdin.write(canvas.tobytes())
            n_written += 1
            if n_written % 500 == 0:
                elapsed   = time.time() - t0
                fps_rate  = n_written / elapsed
                remaining = (n_frames - n_written) / max(fps_rate, 1)
                print(f"  {n_written}/{n_frames}  "
                      f"({100*n_written/n_frames:.0f}%)  "
                      f"{fps_rate:.0f} fr/s  ETA {remaining:.0f}s")
    except BrokenPipeError:
        print(f"BrokenPipeError at frame {n_written}")
        print(ffmpeg_proc.stderr.read().decode(errors="replace"))
        raise
    finally:
        ffmpeg_proc.stdin.close()

    retcode    = ffmpeg_proc.wait()
    stderr_out = ffmpeg_proc.stderr.read().decode(errors="replace")
    elapsed    = time.time() - t0

    if retcode != 0:
        print(f"ffmpeg exited with code {retcode}:")
        print(stderr_out[-3000:])
        sys.exit(retcode)

    total_elapsed = time.time() - t_global
    print(
        f"\nDone — {n_written} frames in {elapsed:.1f}s "
        f"({n_written/elapsed:.0f} fr/s)\n"
        f"Total wall time: {total_elapsed:.1f}s\n"
        f"Saved: {OUTPUT_PATH}\n"
        f"Duration: {n_written/OUTPUT_FPS:.1f}s "
        f"(= {DURATION_S:.0f}s real-time at {SPEED}×)"
    )


if __name__ == "__main__":
    main()

