# -*- coding: utf-8 -*-


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
    root = tk.Tk()
    root.withdraw()
    value = simpledialog.askinteger("Input", prompt, minvalue=min_value)
    root.destroy()
    return value


def count_tif_frames(file_path: str) -> int:
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

    data = np.load(file_path, allow_pickle=True)
    filename = os.path.basename(file_path)

    boundaries = [0] + split_indices + [data.shape[1]]

    for i, out_dir in enumerate(out_dirs):
        os.makedirs(out_dir, exist_ok=True)
        segment = data[:, boundaries[i]:boundaries[i + 1]]
        np.save(os.path.join(out_dir, filename), segment)
        print(f"  [{i+1}/{len(out_dirs)}] {filename}: frames {boundaries[i]}–{boundaries[i+1]-1} -> {out_dir}")


def split_suite2p():

    n_recordings = ask_integer(
        'How many recordings were run together through suite2p?',
        min_value=2
    )
    if n_recordings is None:
        print('Cancelled.')
        return

    s2p_dir = select_directory('Select suite2p plane0 directory.')

    frame_counts = []
    for i in range(n_recordings):
        tif = select_file(
            f'Select tif stack for recording {i + 1} of {n_recordings}.',
            filetypes=[('TIF', '.tif'), ('TIFF', '.tiff')]
        )
        print(f'Counting frames in {tif}')
        n_frames = count_tif_frames(tif)
        print(f'  -> {n_frames} frames')
        frame_counts.append(n_frames)

    total_tif_frames = sum(frame_counts)
    print(f'\nTif frame counts: {frame_counts}  (total: {total_tif_frames})')

    f_npy_path = os.path.join(s2p_dir, 'F.npy')
    f_data = np.load(f_npy_path, allow_pickle=True)
    total_npy_frames = f_data.shape[1]
    if total_tif_frames != total_npy_frames:
        print(
            f'\nWARNING: frame count mismatch!\n'
            f'  Sum of tif frames : {total_tif_frames}  {frame_counts}\n'
            f'  F.npy time axis   : {total_npy_frames}\n'
            f'Make sure you selected the correct tif stacks and suite2p directory.'
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

    print(f'\nSplitting into {n_recordings} recordings at cumulative indices: {split_indices}')

    for key in ['F.npy', 'Fneu.npy', 'spks.npy']:
        print(f'\nSplitting {key}:')
        split_suite2p_npy_multi(
            os.path.join(s2p_dir, key),
            split_indices,
            save_dirs
        )

    for key in ['iscell.npy', 'ops.npy', 'stat.npy']:
        source_file = os.path.join(s2p_dir, key)
        print(f'Copying {key} to all {n_recordings} directories')
        for d in save_dirs:
            shutil.copyfile(source_file, os.path.join(d, key))

    print('\nDone.')


if __name__ == '__main__':

    split_suite2p()
