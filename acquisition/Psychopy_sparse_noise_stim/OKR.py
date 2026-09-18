# -*- coding: utf-8 -*-
"""
acquisition/Psychopy_sparse_noise_stim/OKR.py

OKR stimulus w/ vertical square wave gratings drifting l-to-r and r-to-l.
2 min per direction, 6 blocks where width of bar is halved each time
a new block starts. 30 sec ISI between blocks.
Got rid of all TTL triggers.

need to convert to cycles/degree later...
pxls/deg = screen_width_px / (2 * np.arctan(W_cm / (2 * D_cm)) * (180 / pi))

Functions
---------
run_drift
    Drift a grating for a fixed duration, logging every flip.
run_isi
    Blank screen between drifts, logging every flip.


DMM, April 2026
"""

from psychopy import visual, core, event
import numpy as np
import csv
import time

monitor_x     = 1360
monitor_y     = 768
refresh_rate  = 60.0
temporal_freq = 3.0
dur_sweep_s   = 60
dur_isi_s     = 10
n_sf_steps    = 7


win = visual.Window(
    size=[monitor_x, monitor_y],
    color=[-1, -1, -1],
    units='pix',
    fullscr=True,
    checkTiming=False,
    screen=1
)
monitor_x, monitor_y = win.size[0] // 2, win.size[1] // 2


screen_width_px = monitor_x * 2
sf_initial      = 2.0 / screen_width_px
sf_values       = [sf_initial * (2 ** i) for i in range(n_sf_steps)]

phase_rtr = +(temporal_freq / refresh_rate)
phase_ltr = -(temporal_freq / refresh_rate)

frame_data    = []
history_clock = core.MonotonicClock()


def run_drift(stim, phase_step, duration_s, sf_cyc_per_pix, direction):
    """ Drift a grating for a fixed duration, logging every flip.

    Parameters
    ----------
    stim : psychopy.visual.GratingStim
        Grating to draw.
    phase_step : float
        Phase increment per frame (cycles).
    duration_s : float
        How long to drift, in seconds.
    sf_cyc_per_pix : float
        Spatial frequency, logged with each frame.
    direction : str
        Drift direction label, logged with each frame.
    """

    clock      = core.Clock()
    stim.phase = 0.0
    onset      = history_clock.getTime()
    frame_idx  = 0
    while clock.getTime() < duration_s:
        if event.getKeys(['escape']):
            win.close()
            core.quit()
        stim.phase += phase_step
        stim.draw()
        flip_time = win.flip()
        frame_data.append((
            direction,
            sf_cyc_per_pix,
            frame_idx,
            flip_time,
            time.time()
        ))
        frame_idx += 1
    print(' {} sf={:.6f}: {} frames, onset={:.3f}s'.format(direction, sf_cyc_per_pix, frame_idx, onset))


def run_isi(duration_s):
    """ Blank screen between drifts, logging every flip.

    Parameters
    ----------
    duration_s : float
        ISI length in seconds.
    """

    clock     = core.Clock()
    onset     = history_clock.getTime()
    frame_idx = 0
    while clock.getTime() < duration_s:
        if event.getKeys(['escape']):
            win.close()
            core.quit()
        win.flip()
        frame_data.append((
            'isi',
            np.nan,
            frame_idx,
            history_clock.getTime(),
            time.time()
        ))
        frame_idx += 1
    print('  ISI {}s, onset={:.3f}s'.format(duration_s, onset))


for sf_idx, sf in enumerate(sf_values):
    period_px = 1.0 / sf
    n_cycles  = screen_width_px * sf
    print('\nBlock {}/{}: '
          'sf={:.6f} cyc/pix,  period={:.1f} px  '
          '({:.0f} cycles on screen)'.format(sf_idx + 1, n_sf_steps, sf, period_px, n_cycles))

    stim = visual.GratingStim(
        win,
        tex='sqr',
        sf=sf,
        ori=180,
        size=(monitor_x * 2, monitor_y * 2),
        units='pix',
        contrast=1.0,
    )

    print('  Right-to-left...')
    run_drift(stim, phase_rtr, dur_sweep_s, sf, 'right_to_left')

    # print('  Left-to-right ...')
    # run_drift(stim, phase_ltr, dur_sweep_s, sf, 'left_to_right')

    if sf_idx < n_sf_steps - 1:
        print('  ISI...')
        run_isi(dur_isi_s)

win.close()

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
root.lift()

while True:

    timestamp_file = filedialog.asksaveasfilename(
        title='Save OKR metadata CSV',
        defaultextension='.csv',
        filetypes=[('CSV', '*.csv'),]
    )
    if timestamp_file:
        break

root.destroy()

with open(timestamp_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        'direction',
        'sf_cyc_per_pix',
        'frame_number',
        'psychopy_time',
        'system_time'
    ])
    writer.writerows(frame_data)

core.quit()
