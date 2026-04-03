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


# Parameters
num_frames = 4000
max_dots = 6
diameter_range = (15, 350) # bwtween 1 and 40 visual degrees
on_time = 0.500 # 3 Hz
num_repeats = 1
shuffle = True
save_frames = False
output_file = 'D:/sparse_noise_sequence_v9.npy'
timestamp_file = 'D:/timestamps_251020_DMM056_sparsenoise.csv'
use_trigger = True
monitor_x = 1920
monitor_y = 1080


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
    Check if any dots transition directly from black->white or white->black
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
                if np.abs(pcol - ccol) > (2 - tol):  # black/white flip
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

# switch to pre-rendered frames
pre_rendered_frames = []

for i, frame_dots in enumerate(stim_instructions):
    # Draw everything for this frame once
    for dot in frame_dots:
        stim = visual.Circle(
            win,
            radius=dot['diameter'] / 2,
            pos=dot['pos'],
            fillColor=[dot['color']] * 3,
            lineColor=[dot['color']] * 3,
            units='pix'
        )
        stim.draw()

    # Capture the current drawn buffer as an image
    img_stim = visual.BufferImageStim(win, buffer='back')
    pre_rendered_frames.append(img_stim)

    # Clear window for next pre-render
    win.flip(clearBuffer=True)

print(f"Pre-rendered {len(pre_rendered_frames)} frames.")


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

for rep in range(num_repeats):
    for i, img_stim in enumerate(pre_rendered_frames):
        # Draw pre-rendered frame
        img_stim.draw()

        # Flip buffer — this returns the **actual onset time**
        onset_time = win.flip()
        stim_onset = history_clock.getTime()

        # Compute the target offset time (desired duration)
        desired_offset_time = onset_time + on_time

        # Wait for the frame duration, allowing escape check
        while history_clock.getTime() < stim_onset + on_time:
            if event.getKeys(['escape']):
                win.close()
                core.quit()

            # Avoid busy waiting
            core.wait(0.001)

        # Prepare for next frame (draw next or blank)
        if i < len(pre_rendered_frames) - 1:
            next_stim = pre_rendered_frames[i + 1]
            next_stim.draw()
        else:
            win.color = [0, 0, 0]
            win.flip(clearBuffer=True)

        # The **offset** is the next flip (when current frame is replaced)
        offset_time = win.flip()

        # Record frame if enabled
        if save_frames:
            frame = win.getMovieFrame(buffer='front')
            frame_np = np.asarray(frame)
            recorded_frames.append(frame_np)

        frame_data.append({
            'rep': rep,
            'frame_index': i,
            'onset_time': onset_time,
            'offset_time': offset_time,
            'stim_onset_clock': stim_onset,
            'duration_actual': offset_time - onset_time,
        })

        print(f'Frame {i}: {len(frame_dots)} dots, '
              f'onset={onset_time:.3f}, offset={offset_time:.3f}')

with open(timestamp_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(frame_data)

if save_frames and recorded_frames:
   recorded_frames = np.stack(recorded_frames, axis=0)
   np.save(output_file, recorded_frames)
   print(f'Saved {recorded_frames.shape[0]} frames to {output_file}')

win.close()
ser.close()
core.quit()