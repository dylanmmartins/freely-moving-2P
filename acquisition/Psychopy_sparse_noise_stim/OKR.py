# -*- coding: utf-8 -*-
"""
OKR stimulus w/ vertical square wave gratings drifting l->r and r->l.
2 min per direction, 6 blocks where width of bar is halved each time
a new block starts. 30 sec ISI between blocks.
Got rid of all TTL triggers.

need to convert to cycles/degree later...
pxls/deg = screen_width_px / (2 * np.arctan(W_cm / (2 * D_cm)) * (180 / pi))

Author: DMM, April 22, 2026
"""

from psychopy import visual, core, event
import numpy as np
import csv
import time

monitor_x     = 1360
monitor_y     = 768
refresh_rate  = 60.0
temporal_freq = 1.0
dur_sweep_s   = 2 * 60
dur_isi_s     = 30
n_sf_steps    = 6


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
sf_initial      = 1.0 / screen_width_px
sf_values       = [sf_initial * (2 ** i) for i in range(n_sf_steps)]

phase_rtr = +(temporal_freq / refresh_rate)
phase_ltr = -(temporal_freq / refresh_rate)

frame_data    = []
history_clock = core.MonotonicClock()


def run_drift(stim, phase_step, duration_s, sf_cyc_per_pix, direction):
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
    print(f' {direction} sf={sf_cyc_per_pix:.6f}: {frame_idx} frames, onset={onset:.3f}s')


def run_isi(duration_s):

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
    print(f'  ISI {duration_s}s, onset={onset:.3f}s')


for sf_idx, sf in enumerate(sf_values):
    period_px = 1.0 / sf
    n_cycles  = screen_width_px * sf
    print(f'\nBlock {sf_idx + 1}/{n_sf_steps}: '
          f'sf={sf:.6f} cyc/pix,  period={period_px:.1f} px  '
          f'({n_cycles:.0f} cycles on screen)')

    stim = visual.GratingStim(
        win,
        tex='sqr',
        sf=sf,
        ori=180,
        size=(monitor_x * 2, monitor_y * 2),
        units='pix',
        contrast=1.0,
    )

    print('  Right-to-left ...')
    run_drift(stim, phase_rtr, dur_sweep_s, sf, 'right_to_left')

    print('  Left-to-right ...')
    run_drift(stim, phase_ltr, dur_sweep_s, sf, 'left_to_right')

    if sf_idx < n_sf_steps - 1:
        print('  ISI ...')
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
