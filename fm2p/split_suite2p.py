# -*- coding: utf-8 -*-
"""
fm2p/split_suite2p.py

Split suite2p outputs from a concatenated run back into per-recording folders.

Functions
---------
ask_integer
    Ask for an integer in a Tk dialog.
count_tif_frames
    Count pages in a tif by seeking until EOF.
split_suite2p_npy_multi
    Split a suite2p (cells, frames) npy along time into several output dirs.
split_suite2p
    Split a multi-recording suite2p run back into per-recording plane0 folders.


DMM, April 2026
"""


if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

from PIL import Image
import numpy as np
import os
import shutil
import tkinter as tk
from tkinter import simpledialog

from .utils.gui_funcs import select_directory, select_file


def ask_integer(prompt: str, min_value: int = 2) -> int:
    """ Ask for an integer in a Tk dialog.

    Parameters
    ----------
    prompt : str
    min_value : int

    Returns
    -------
    value : int or None
        None if canceled.
    """

    root = tk.Tk()
    root.withdraw()
    value = simpledialog.askinteger("Input", prompt, minvalue=min_value)
    root.destroy()
    return value


def count_tif_frames(file_path: str) -> int:
    """ Count pages in a tif by seeking until EOF.

    Parameters
    ----------
    file_path : str

    Returns
    -------
    count : int
    """

    with Image.open(file_path) as img:
        count = 0
        try:
            while True:
                img.seek(count)
                count += 1
        except EOFError:
            pass
    return count


def split_suite2p_npy_multi(file_path: str, split_indices: list, out_dirs: list):
    """ Split a suite2p (cells, frames) npy along time into several output dirs.

    Parameters
    ----------
    file_path : str
        suite2p .npy, e.g. F.npy.
    split_indices : list of int
        Frame indices where each new recording starts.
    out_dirs : list of str
        One directory per segment.
    """

    data = np.load(file_path, allow_pickle=True)
    filename = os.path.basename(file_path)

    boundaries = [0] + split_indices + [data.shape[1]]

    for i, out_dir in enumerate(out_dirs):
        os.makedirs(out_dir, exist_ok=True)
        segment = data[:, boundaries[i]:boundaries[i + 1]]
        np.save(os.path.join(out_dir, filename), segment)
        print("  [{}/{}] {}: frames {}-{} to {}".format(i+1, len(out_dirs), filename, boundaries[i], boundaries[i+1]-1, out_dir))


def split_suite2p():
    """ Split a multi-recording suite2p run back into per-recording plane0 folders.

    Asks for recording count and each tif to get frame counts, checks the total
    against F.npy, then splits every per-frame npy.
    """

    n_recordings = ask_integer(
        'How many recordings were run together through suite2p?',
        min_value=2
    )
    if n_recordings is None:
        print('Canceled.')
        return

    s2p_dir = select_directory('Select suite2p plane0 directory.')

    frame_counts = []
    for i in range(n_recordings):
        tif = select_file(
            f'Select tif stack for recording {i + 1} of {n_recordings}.',
            filetypes=[('TIF', '.tif'), ('TIFF', '.tiff')]
        )
        print('Counting frames in {}...'.format(tif))
        n_frames = count_tif_frames(tif)
        print('  {} frames'.format(n_frames))
        frame_counts.append(n_frames)

    total_tif_frames = sum(frame_counts)
    print('\nTif frame counts: {}  (total: {})'.format(frame_counts, total_tif_frames))

    f_npy_path = os.path.join(s2p_dir, 'F.npy')
    f_data = np.load(f_npy_path, allow_pickle=True)
    total_npy_frames = f_data.shape[1]
    if total_tif_frames != total_npy_frames:
        print(
            '\nWARNING: frame count mismatch!\n'
            '  Sum of tif frames : {}  {}\n'
            '  F.npy time axis   : {}\n'
            'Make sure you selected the correct tif stacks and suite2p directory.'.format(total_tif_frames, frame_counts, total_npy_frames)
        )
        raise ValueError(
            f'Frame count mismatch: tifs sum to {total_tif_frames} '
            f'but F.npy has {total_npy_frames} frames.'
        )

    split_indices = list(np.cumsum(frame_counts[:-1]).astype(int))

    save_dirs = []
    for i in range(n_recordings):
        d = select_directory(f'Select save directory for recording {i + 1} of {n_recordings}.')
        if 'suite2p' not in d:
            d = os.path.join(d, 'suite2p', 'plane0')
        save_dirs.append(d)

    for d in save_dirs:
        os.makedirs(d, exist_ok=True)

    print('\nSplitting into {} recordings at cumulative indices: {}'.format(n_recordings, split_indices))

    for key in ['F.npy', 'Fneu.npy', 'spks.npy']:
        print('\nSplitting {}:'.format(key))
        split_suite2p_npy_multi(
            os.path.join(s2p_dir, key),
            split_indices,
            save_dirs
        )

    for key in ['iscell.npy', 'ops.npy', 'stat.npy']:
        source_file = os.path.join(s2p_dir, key)
        print('Copying {} to all {} directories...'.format(key, n_recordings))
        for d in save_dirs:
            shutil.copyfile(source_file, os.path.join(d, key))

    print('\nDone.')


if __name__ == '__main__':

    split_suite2p()
