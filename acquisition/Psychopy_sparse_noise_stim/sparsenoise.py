# -*- coding: utf-8 -*-
"""
Download the stable windows release of Psychpy. Tested with
version 2025.1.1 from https://www.psychopy.org/download.html
Arduino Uno R3 must be attached to COM3 and have a BNC jack
connected to Analog pin 0.
Run the script from the Psychopy GUI.

Author: DMM, last modified Sept. 2025
"""


from psychopy import visual, core, event
import numpy as np
import serial


# Parameters
num_frames = 1000 # ~16 minutes
max_dots = 10
diameter_range = (10, 400)
on_time = 0.5
off_time = 0.5
num_repeats = 1
shuffle = True
save_frames = True
output_file = 'sparse_noise_sequence.npy'

# Arduino serial settings
arduino_port = 'COM3'
baud_rate = 115200
trigger_threshold = 512 # analog value threshold (0-1023)

# Setup window
win = visual.Window(
    size=[1920, 1080],
    color=[0, 0, 0],
    units='pix',
    fullscr=True,
    checkTiming=False
)
monitor_x, monitor_y = win.size[0] // 2, win.size[1] // 2

# Deterministic RNG
np.random.seed(42)

# Generate stimulus instructions
stim_instructions = []
for _ in range(num_frames):
    n_dots = np.random.randint(1, max_dots + 1)
    frame_dots = []
    for _ in range(n_dots):
        diameter = np.random.uniform(*diameter_range)
        color = np.random.choice([1, -1])
        pos_x = np.random.uniform(-monitor_x, monitor_x)
        pos_y = np.random.uniform(-monitor_y, monitor_y)
        frame_dots.append({
            'diameter': diameter,
            'color': color,
            'pos': (pos_x, pos_y)
        })
    stim_instructions.append(frame_dots)

if shuffle:
    np.random.shuffle(stim_instructions)

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

# Prepare frame recording
recorded_frames = []

# Run stimulus
history_clock = core.MonotonicClock()

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
            win.flip()

        stim_offset = history_clock.getTime()

        # Record frame if enabled
        if save_frames:
            frame = win.getMovieFrame(buffer='back')
            frame_np = np.array(frame)
            recorded_frames.append(frame_np)

        # Inter-frame gray screen
        if off_time > 0:
            if event.getKeys(['escape']):
                win.close()
                core.quit()
            win.flip()
            core.wait(off_time)

        print(f'Frame {i}: {len(frame_dots)} dots, '
              f'onset={stim_onset:.3f}, offset={stim_offset:.3f}')

# Save frames to .npy
if save_frames and recorded_frames:
    recorded_frames = np.stack(recorded_frames, axis=0)
    np.save(output_file, recorded_frames)
    print(f'Saved {recorded_frames.shape[0]} frames to {output_file}')

win.close()
ser.close()
core.quit()
