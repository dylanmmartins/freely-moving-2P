"""
Sanity-check video for head-yaw tracking: overlays the tracked head position
and two yaw estimates -- the raw topdown-camera-derived yaw ('head_yaw_deg')
and the IMU-derived, drift-corrected yaw from
fm2p.utils.imu.detrend_gyroz_weighted_gaussian() -- directly on top of the
recording's raw topdown video. Built to answer one question: is the IMU
correction making the yaw tracking better or worse, by eye, against the
camera as ground truth.

Alignment notes (read this before trusting the output):
- The topdown video is hardware frame-locked to the 2P scope (one video
  frame per twopT sample, see fm2p/preprocess.py's module docstring) -- so
  video frame i and head_x[i]/head_y[i]/twopT[i] refer to the same instant
  with no resampling or shift needed.
- All overlay drawing happens at the video's NATIVE resolution, directly in
  the same pixel coordinates as head_x/head_y (no unit conversion, no
  pxls2cm). The composited frame is resized (if --scale != 1) only as the
  last step before writing to the output video -- never before drawing.
  Resizing the frame without correspondingly rescaling the overlay
  coordinates is a classic way to make tracking look like it has a
  "position lag" that grows with movement speed, when the underlying
  tracking is actually fine; this script structurally can't make that
  mistake.
- head_yaw_deg as stored in preproc.h5 is reliably 1 sample longer than
  head_x/head_y/twopT (see the trimming loop in fm2p/preprocess.py around
  the `_len_diff` loop -- it trims a *local* `yaw` variable from the end,
  but that trimmed copy is never written back to the dict that gets saved,
  so the untrimmed original survives). We trim the trailing sample to match
  -- i.e. keep head_yaw_deg[:n] -- consistent with that trimming logic.
- Frame-accurate seeking via cv2's CAP_PROP_POS_FRAMES is unreliable for at
  least some of these topdown video files (confirmed: it reports success
  but lands on the wrong frame, which then fails to decode). This script
  seeks by grabbing frames sequentially from 0 instead -- slower for a
  large --start_frame, but actually correct. Don't "optimize" this back to
  cap.set() without re-verifying against the actual video file.
- gyro_z_trim has an exact alternating (every-other-sample) NaN pattern in
  a meaningful fraction of IMU recordings -- a real data-quality issue, not
  something introduced by this script. detrend_gyroz_weighted_gaussian()
  does an unguarded np.cumsum() over gyro_z_trim, so a single NaN poisons
  every cumulative sum after it; for affected recordings the *stored*
  preproc.h5 'upsampled_yaw' is ~100% NaN already, in production, before
  this script ever touches it. This script patches short gaps for display
  (see load_yaw_traces) and prints a warning when it does -- it does not
  fix the underlying issue, which lives in detrend_gyroz_weighted_gaussian
  or further upstream in how gyro_z_trim gets built.
- The IMU-corrected yaw is computed at IMU sample resolution (imuT_trim,
  much higher rate than twopT) and is interpolated onto twopT for display.
  We interpolate the *unwrapped* radian signal (igyro_corrected_rad), not
  the wrapped 0-360 degree signal, to avoid wraparound interpolation
  artifacts at the 359->0 boundary; wrapping doesn't matter afterward since
  the angle is only ever used through cos/sin.

Usage:
    python -m fm2p.yaw_correction_diagnostic_video /path/to/preproc.h5
    python -m fm2p.yaw_correction_diagnostic_video /path/to/preproc.h5 \\
        --start_frame 500 --n_frames 600 --scale 0.4
"""

if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

from ..fm2p.utils.files import read_h5
from ..fm2p.utils.paths import find
from ..fm2p.utils.imu import complementary_filter_yaw
from ..fm2p.utils.time import interpT

CAM_COLOR     = (255, 255, 0)    # cyan, BGR  -- raw topdown-camera yaw
IMU_COLOR     = (0, 0, 255)      # red,  BGR  -- IMU-corrected ("upsampled") yaw
POS_COLOR     = (0, 255, 0)      # green, BGR -- tracked head position
ARROW_LEN_PX  = 60
POS_RADIUS_PX = 6
TOPDOWN_GLOB  = 'fm*_0001.mp4'   # see fm2p/preprocess.py docstring for this naming convention


def _find_topdown_video(preproc_path):
    rec_dir = os.path.dirname(os.path.abspath(preproc_path))
    hits = find(TOPDOWN_GLOB, rec_dir)
    if not hits:
        raise FileNotFoundError(
            f'No topdown video matching "{TOPDOWN_GLOB}" found in {rec_dir}. '
            'Pass --video explicitly.')
    return sorted(hits)[0]


def load_yaw_traces(data, tau=1.0):
    """Returns (head_x, head_y, yaw_cam_deg, yaw_imu_deg), all aligned to
    twopT / topdown-video-frame index. See module docstring for the
    alignment reasoning behind the head_yaw_deg trim and the unwrapped-angle
    interpolation."""
    twopT = np.asarray(data['twopT'], dtype=float)
    n = len(twopT)

    head_x = np.asarray(data['head_x'], dtype=float)[:n]
    head_y = np.asarray(data['head_y'], dtype=float)[:n]
    yaw_cam_deg = np.asarray(data['head_yaw_deg'], dtype=float)[:n]

    # gyro_z_trim is NaN in an exact alternating (every-other-sample)
    # pattern in a meaningful fraction of IMU recordings. Print a warning
    # and patch short gaps so the filter can coast through them.
    gyro_z = np.asarray(data['gyro_z_trim'], dtype=float).copy()
    nan_frac = np.mean(~np.isfinite(gyro_z))
    if nan_frac > 0.01:
        print(f'  WARNING: gyro_z_trim is {nan_frac:.1%} NaN for this recording -- '
              'this is a known data-quality issue (alternating-sample dropout). '
              'Patching short gaps with interp_short_gaps() for display purposes only.')
        from ..fm2p.utils.helper import interp_short_gaps
        gyro_z = interp_short_gaps(gyro_z, max_gap=5)
        data = {**data, 'gyro_z_trim': gyro_z}

    imu_result = complementary_filter_yaw(data, tau=tau)
    imuT = np.asarray(data['imuT_trim'], dtype=float)
    yaw_imu_rad_unwrapped = interpT(imu_result['igyro_corrected_rad'], imuT, twopT)
    yaw_imu_deg = np.rad2deg(yaw_imu_rad_unwrapped) % 360.0

    return head_x, head_y, yaw_cam_deg, yaw_imu_deg


def _draw_overlay(frame, x, y, yaw_cam_deg, yaw_imu_deg):
    """Draws directly on `frame` at its native resolution, in the same
    pixel coordinates as head_x/head_y -- do not resize `frame` before
    calling this, and do not rescale x/y to match a resized frame; resize
    the already-composited output of this function instead."""
    if np.isfinite(x) and np.isfinite(y):
        pt = (int(round(x)), int(round(y)))
        cv2.circle(frame, pt, POS_RADIUS_PX, POS_COLOR, -1)
        for yaw_deg, color in ((yaw_cam_deg, CAM_COLOR), (yaw_imu_deg, IMU_COLOR)):
            if np.isfinite(yaw_deg):
                rad = np.deg2rad(yaw_deg)
                end = (int(round(x + ARROW_LEN_PX * np.cos(rad))),
                       int(round(y + ARROW_LEN_PX * np.sin(rad))))
                cv2.arrowedLine(frame, pt, end, color, 2, tipLength=0.3)

    cv2.putText(frame, 'camera yaw',  (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, CAM_COLOR, 2)
    cv2.putText(frame, 'IMU yaw',     (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, IMU_COLOR, 2)
    cv2.putText(frame, 'head pos',    (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, POS_COLOR, 2)
    return frame


def make_yaw_diagnostic_video(preproc_path, video_path=None, out_path=None,
                               start_frame=0, n_frames=450, scale=0.5,
                               tau=1.0):
    print(f'Loading {preproc_path}')
    data = read_h5(preproc_path)
    if 'imuT_trim' not in data or 'gyro_z_trim' not in data:
        raise ValueError(f'{preproc_path} has no IMU data -- cannot compute IMU-corrected yaw.')

    head_x, head_y, yaw_cam_deg, yaw_imu_deg = load_yaw_traces(data, tau=tau)
    n_total = len(head_x)

    if video_path is None:
        video_path = _find_topdown_video(preproc_path)
    print(f'Topdown video: {video_path}')

    start_frame = max(0, min(start_frame, n_total - 1))
    end_frame   = min(start_frame + n_frames, n_total)

    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(preproc_path)),
            f'yaw_diagnostic_{start_frame}_{end_frame}.mp4')

    cap = cv2.VideoCapture(video_path)
    n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_video_frames < end_frame:
        print(f'  Warning: video has {n_video_frames} frames, requested range '
              f'goes to {end_frame}; clipping.')
        end_frame = min(end_frame, n_video_frames)

    # cap.set(CAP_PROP_POS_FRAMES, ...) is unreliable on this video file --
    # verified it silently seeks to the wrong position and then fails to
    # read at all (reproduced: set() to frame 4975 "succeeds" but
    # get(POS_FRAMES) afterward reports 3655, and the following read()
    # returns False). Long-GOP/irregular-timestamp topdown captures are a
    # known case where frame-accurate seeking breaks in OpenCV/ffmpeg.
    # grab()-ing sequentially from frame 0 is slower but actually correct.
    if start_frame > 0:
        print(f'  Seeking to frame {start_frame} (sequential grab -- '
              'cap.set() is unreliable for this video format)...')
        for _ in tqdm(range(start_frame), desc='Seeking'):
            cap.grab()

    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = max(1, int(round(native_w * scale))), max(1, int(round(native_h * scale)))

    twopT_window = np.asarray(data['twopT'], dtype=float)[start_frame:end_frame]
    fps_out = 1.0 / float(np.nanmedian(np.diff(twopT_window))) if len(twopT_window) > 1 else 7.5

    fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(out_path, fourcc, fps_out, (out_w, out_h))

    print(f'Rendering frames {start_frame}:{end_frame} (fps={fps_out:.2f}, '
          f'native={native_w}x{native_h} -> out={out_w}x{out_h})')
    for i in tqdm(range(start_frame, end_frame), desc='Rendering'):
        ret, frame = cap.read()
        if not ret:
            print(f'  Video ended early at frame {i}.')
            break
        frame = _draw_overlay(frame, head_x[i], head_y[i], yaw_cam_deg[i], yaw_imu_deg[i])
        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        out_vid.write(frame)

    cap.release()
    out_vid.release()
    print(f'Saved: {out_path}')
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description='Overlay tracked head position + camera/IMU yaw on the topdown video.')
    parser.add_argument('--preproc', help='Path to preproc.h5', default='/home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings/cohort02_recordings/251021_DMM_DMM061_pos04/fm1/251021_DMM_DMM061_fm_01_preproc_v2.h5')
    parser.add_argument('--video', default='/home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings/cohort02_recordings/251021_DMM_DMM061_pos04/fm1/fm1_0001.mp4',
                        help=f'Path to topdown mp4 (auto-detected via "{TOPDOWN_GLOB}" if omitted)')
    parser.add_argument('--out', default=None, help='Output video path')
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--n_frames', type=int, default=3600,
                        help='~60s at the typical ~7.5 Hz topdown/2P frame rate')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='Output resize factor (applied after drawing, never before)')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='Complementary filter time constant (s): camera pulls the '
                             'gyro estimate back over this timescale. 0.5 s snaps fast, '
                             '5 s is more gyro-like. Default 1.0 s.')
    args = parser.parse_args()

    make_yaw_diagnostic_video(
        args.preproc, video_path=args.video, out_path=args.out,
        start_frame=args.start_frame, n_frames=args.n_frames, scale=args.scale,
        tau=args.tau,
    )


if __name__ == '__main__':
    main()
