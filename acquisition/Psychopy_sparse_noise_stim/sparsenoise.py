# -*- coding: utf-8 -*-
"""
acquisition/Psychopy_sparse_noise_stim/sparsenoise.py

Download the stable windows release of Psychpy. Tested with
version 2025.1.1 from https://www.psychopy.org/download.html
Arduino Uno R3 must be attached to COM3 and have a BNC jack
connected to Analog pin 0.
Run the script from the Psychopy GUI.
Uses non-overlapping spots. ISI removed.

Functions
---------
non_overlapping_pos
    Random dot position that doesn't overlap existing dots.
check_illegal_transitions
    Flag frame pairs where an overlapping dot flips polarity.
generate_frame
    Random sparse-noise frame of non-overlapping black/white dots.
legal_sparse_frames
    Sequence of sparse-noise frames with no illegal polarity flips between neighbors.


DMM, October 2025
"""


from psychopy import visual, core, event
import numpy as np
import serial
import csv
import time


output_file = 'D:/sparse_noise_sequence_scope_test_v1.npy'
timestamp_file = 'D:/timestamps_251120_DMM056_sparsenoise.csv'
num_frames = 4000
max_dots = 6
diameter_range = (15, 350) # bwtween 1 and 40 visual degrees
on_time = 0.500 # 3 Hz
num_repeats = 1
shuffle = True
save_frames = False
use_trigger = True
monitor_x = 1920
monitor_y = 1080


def non_overlapping_pos(existing_dots, new_diameter, max_attempts=1000):
    """ Random dot position that doesn't overlap existing dots.

    Gives up after max_attempts and returns last draw anyway.

    Parameters
    ----------
    existing_dots : list of dict
        Dots already in frame, each with 'pos' and 'diameter'.
    new_diameter : float
        Diameter of dot being placed.
    max_attempts : int
        Draws to try before giving up.

    Returns
    -------
    pos_x, pos_y : float
        Dot center in monitor units.
    """

    new_radius = new_diameter / 2
    for _ in range(max_attempts):
        pos_x = np.random.uniform(-monitor_x + new_radius, monitor_x - new_radius)
        pos_y = np.random.uniform(-monitor_y + new_radius, monitor_y - new_radius)
        # Check overlap
        overlap = False
        for dot in existing_dots:
            old_x, old_y = dot['pos']
            old_r = dot['diameter'] / 2
            dist = np.hypot(pos_x - old_x, pos_y - old_y)
            if dist < (new_radius + old_r):
                overlap = True
                break
        if not overlap:
            return pos_x, pos_y

    return pos_x, pos_y


def check_illegal_transitions(prev_dots, curr_dots, tol=1e-6):
    """ Flag frame pairs where an overlapping dot flips polarity.

    Parameters
    ----------
    prev_dots : list of dict
        Dots in previous frame.
    curr_dots : list of dict
        Dots in current frame.
    tol : float
        Tolerance on color difference; black-to-white is a difference of 2.

    Returns
    -------
    illegal : bool
        True if any overlapping dot pair has opposite color.
    """

    illegal = False

    for prev_dot in prev_dots:
        px, py = prev_dot['pos']
        pr = prev_dot['diameter'] / 2
        pcol = prev_dot['color']

        for curr_dot in curr_dots:
            cx, cy = curr_dot['pos']
            cr = curr_dot['diameter'] / 2
            ccol = curr_dot['color']

            dist = np.hypot(px - cx, py - cy)
            if dist < (pr + cr):
                if np.abs(pcol - ccol) > (2 - tol):
                    illegal = True
                    break
        if illegal:
            break

    return illegal


def generate_frame(max_dots, diameter_range):
    """ Random sparse-noise frame of non-overlapping black/white dots.

    Parameters
    ----------
    max_dots : int
        Upper bound on dots per frame.
    diameter_range : tuple of float
        (min, max) dot diameter.

    Returns
    -------
    frame_dots : list of dict
        One dict per dot with 'diameter', 'color', 'pos'.
    """

    n_dots = np.random.randint(1, max_dots + 1)
    frame_dots = []

    for _ in range(n_dots):
        diameter = np.random.uniform(*diameter_range)
        color = np.random.choice([1, -1])
        pos_x, pos_y = non_overlapping_pos(frame_dots, diameter)
        frame_dots.append({
            'diameter': diameter,
            'color': color,
            'pos': (pos_x, pos_y)
        })
    return frame_dots


def legal_sparse_frames(num_frames, max_dots, diameter_range,
                        monitor_x, monitor_y, shuffle=False,
                        max_attempts=1000):
    """ Sequence of sparse-noise frames with no illegal polarity flips between neighbors.

    Parameters
    ----------
    num_frames : int
        Frames to generate.
    max_dots : int
        Upper bound on dots per frame.
    diameter_range : tuple of float
        (min, max) dot diameter.
    monitor_x, monitor_y : float
        Monitor half-extent; unused here, positions use module globals.
    shuffle : bool
        Shuffle frame order after generation.
    max_attempts : int
        Redraws allowed per frame before raising.

    Returns
    -------
    stim_instructions : list of list of dict
        One dot list per frame.
    """

    stim_instructions = []

    prev_frame = generate_frame(max_dots, diameter_range)
    stim_instructions.append(prev_frame)

    for _ in range(1, num_frames):
        for attempt in range(max_attempts):
            curr_frame = generate_frame(max_dots, diameter_range)
            if not check_illegal_transitions(prev_frame, curr_frame):
                stim_instructions.append(curr_frame)
                prev_frame = curr_frame
                break
        else:
            raise RuntimeError(f"Failed to generate legal frame after {max_attempts} attempts.")

    if shuffle:
        np.random.shuffle(stim_instructions)

    return stim_instructions


if use_trigger:

    arduino_port = 'COM3'
    baud_rate = 115200
    trigger_threshold = 512

win = visual.Window(
    size=[monitor_x, monitor_y],
    color=[0, 0, 0],
    units='pix',
    fullscr=True,
    checkTiming=False,
    screen=1
)
monitor_x, monitor_y = win.size[0] // 2, win.size[1] // 2

np.random.seed(42)

stim_instructions = legal_sparse_frames(
    num_frames = num_frames,
    max_dots = max_dots,
    diameter_range = diameter_range,
    monitor_x = monitor_x,
    monitor_y = monitor_y,
    shuffle = shuffle
)

if use_trigger:

    ser = serial.Serial(arduino_port, baud_rate, timeout=1)
    core.wait(2.0)

    print('Waiting for analog trigger on Arduino A0...')
    triggered = False

    while not triggered:
        if event.getKeys(['escape']):
            win.close()
            core.quit()
        line = ser.readline().decode('utf-8').strip()
        if line.isdigit():
            value = int(line)
            if value <= trigger_threshold:
                triggered = True
                print('Trigger received (value={}). Starting stimulus.'.format(value))

recorded_frames = []
frame_data = []

history_clock = core.MonotonicClock()

for rep in range(num_repeats):
    for i, frame_dots in enumerate(stim_instructions):
        on_clock = core.Clock()
        stim_onset = history_clock.getTime()

        while on_clock.getTime() < on_time:
            if event.getKeys(['escape']):
                win.close()
                core.quit()

            for dot in frame_dots:
                stim = visual.Circle(
                    win,
                    radius=dot['diameter'] / 2,
                    pos=dot['pos'],
                    fillColor=[dot['color']]*3,
                    lineColor=[dot['color']]*3,
                    units='pix'
                )
                stim.draw()

            flip_time = win.flip()
            system_time = time.time()

        frame_data.append((i, flip_time, system_time))

        stim_offset = history_clock.getTime()

        if save_frames:
            frame = win.getMovieFrame(buffer='front')
            frame_np = np.asarray(frame)
            recorded_frames.append(frame_np)

        print('Frame {}: {} dots, '
              'onset={:.3f}, offset={:.3f}'.format(i, len(frame_dots), stim_onset, stim_offset))

win.close()
if use_trigger:
    ser.close()

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
root.lift()

while True:
    print('Prompting for timestamp CSV save path...')
    timestamp_file = filedialog.asksaveasfilename(
        title='Save timestamp CSV (required)',
        defaultextension='.csv',
        filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
    )
    if timestamp_file:
        break
    print('Timestamp save path is required. Please choose a file.')

if save_frames and recorded_frames:
    print('Prompting for recorded frames save path...')
    output_file = filedialog.asksaveasfilename(
        title='Save recorded frames (.npy)',
        defaultextension='.npy',
        filetypes=[('NumPy files', '*.npy'), ('All files', '*.*')]
    )

root.destroy()

with open(timestamp_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["frame_number", "psychopy_time", "system_time"])
    writer.writerows(frame_data)
print('Saved timestamps to {}'.format(timestamp_file))

if save_frames and recorded_frames:
    if output_file:
        recorded_frames = np.stack(recorded_frames, axis=0)
        np.save(output_file, recorded_frames)
        print('Saved {} frames to {}'.format(recorded_frames.shape[0], output_file))
    else:
        print('No frame save path selected; frames not saved.')

core.quit()
