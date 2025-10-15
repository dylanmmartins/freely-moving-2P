# -*- coding: utf-8 -*-
"""
Download the stable windows release of Psychpy. Tested with
version 2025.1.1 from https://www.psychopy.org/download.html
Arduino Uno R3 must be attached to COM3 and have a BNC jack
connected to Analog pin 0.
Run the script from the Psychopy GUI.
Uses non-overlapping spots. ISI removed.

Author: DMM, last modified Oct. 2025
"""


from psychopy import visual, core, event
import numpy as np
import serial
import csv
import time


def non_overlapping_pos(existing_dots, new_diameter, max_attempts=1000):
    """ Generate a random (x, y) position that doesn't overlap with existing dots.
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
    """
    Check if any dots transition directly from black→white or white→black
    between frames (illegal transitions).

    prev_dots and curr_dots are lists of dicts like:
        {'pos': (x, y), 'diameter': d, 'color': ±1}
    """
    illegal = False

    # Compare all dots that occupy roughly the same position
    for prev_dot in prev_dots:
        px, py = prev_dot['pos']
        pr = prev_dot['diameter'] / 2
        pcol = prev_dot['color']

        for curr_dot in curr_dots:
            cx, cy = curr_dot['pos']
            cr = curr_dot['diameter'] / 2
            ccol = curr_dot['color']

            # If they overlap, check if color flips illegally
            dist = np.hypot(px - cx, py - cy)
            if dist < (pr + cr):  # overlapping dots (same region)
                if np.abs(pcol - ccol) > (2 - tol):  # black↔white flip
                    illegal = True
                    break
        if illegal:
            break

    return illegal


def generate_frame(max_dots, diameter_range):
    """
    Generate one sparse noise frame: a list of dot dictionaries.
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
    """
    Generate a list of sparse noise stimulus instructions,
    ensuring that no frame transitions directly from black↔white.

    Returns
    -------
    stim_instructions : list of list of dicts
    """

    stim_instructions = []

    # First frame: anything goes (starts from grey)
    prev_frame = generate_frame(max_dots, diameter_range)
    stim_instructions.append(prev_frame)

    # Generate subsequent frames, ensuring legal transitions
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


# Parameters
num_frames = 4000 # 5000 frames == ~41 minutes
max_dots = 6
diameter_range = (15, 350) # bwtween 1 and 40 visual degrees
on_time = 0.500 # 3 Hz
num_repeats = 1
shuffle = True
save_frames = True
output_file = 'D:/sparse_noise_sequence_v6.npy'
timestamp_file = 'D:/timestamps_251014_DMM000_ltdk.csv'
use_trigger = False
monitor_x = 1920
monitor_y = 1080

if use_trigger:
    # Arduino serial settings
    arduino_port = 'COM3'
    baud_rate = 115200
    trigger_threshold = 512 # analog value threshold (0-1023)

# Setup window
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

# Generate stimulus instructions
stim_instructions = legal_sparse_frames(
    num_frames = num_frames,
    max_dots = max_dots,
    diameter_range = diameter_range,
    monitor_x = monitor_x,
    monitor_y = monitor_y,
    shuffle = shuffle
)

if use_trigger:
    # Open serial to Arduino
    ser = serial.Serial(arduino_port, baud_rate, timeout=1)
    core.wait(2.0)  # wait for Arduino reset

    print('Waiting for analog trigger on Arduino A0...')
    triggered = False
    # triggered = True
    while not triggered:
        if event.getKeys(['escape']):
            win.close()
            core.quit()
        line = ser.readline().decode('utf-8').strip()
        if line.isdigit():
            value = int(line)
            if value <= trigger_threshold:
                triggered = True
                print(f'Trigger received (value={value}). Starting stimulus.')

recorded_frames = []
frame_data = []

history_clock = core.MonotonicClock()

frameN = 0
for rep in range(num_repeats):
    for i, frame_dots in enumerate(stim_instructions):
        on_clock = core.Clock()
        stim_onset = history_clock.getTime()

        while on_clock.getTime() < on_time:
            if event.getKeys(['escape']):
                win.close()
                core.quit()

            # Draw all dots
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

            frame_data.append((frameN, flip_time, system_time))
            frameN += 1

        stim_offset = history_clock.getTime()

        # Record frame if enabled
        if save_frames:
            frame = win.getMovieFrame(buffer='front')
            frame_np = np.asarray(frame)
            recorded_frames.append(frame_np)

        # ISI removed
        # Inter-frame gray screen
        # if off_time > 0:
        #     if event.getKeys(['escape']):
        #         win.close()
        #         core.quit()
        #     win.flip()
        #     core.wait(off_time)

        print(f'Frame {i}: {len(frame_dots)} dots, '
              f'onset={stim_onset:.3f}, offset={stim_offset:.3f}')

with open(timestamp_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["frame_number", "psychopy_time", "system_time"])
    writer.writerows(frame_data)

if save_frames and recorded_frames:
   recorded_frames = np.stack(recorded_frames, axis=0)
   np.save(output_file, recorded_frames)
   print(f'Saved {recorded_frames.shape[0]} frames to {output_file}')

win.close()
ser.close()
core.quit()

# cross correlation between stimulus and population spiking activity

# stim_flat = stimarr.reshape(np.size(stimarr,0), -1)
# stim_drive = stim_flat.mean(axis=1)
# stim_drive_interp = fm2p.interpT(stim_drive, stimT, twopT)

# pop_resp = np.nansum(data['s2p_spks'], axis=0)
# # z-score
# pop_resp = (pop_resp - np.mean(pop_resp)) / np.std(pop_resp)

# cc, lags = fm2p.nanxcorr(stim_drive_interp, pop_resp, maxlag=40)
# best_lag = lags[np.argmax(cc)]

