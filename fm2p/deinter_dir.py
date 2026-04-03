# -*- coding: utf-8 -*-
"""
Batch deinterlacing utility for a directory of AVI videos.

This script allows the user to select a directory, finds all .avi files within it,
and applies deinterlacing (and optional rotation) to each file using fm2p utilities.

Functions:
    deinter_dir(): Prompts user for a directory, finds all .avi files, and deinterlaces them.

Example usage:
    $ python deinter_dir.py

Author: DMM, lat modified June 2025
"""


if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

import os
from tqdm import tqdm

from .utils.gui_funcs import select_directory
from .utils.paths import find
from .utils.cameras import deinterlace


def deinter_dir(dir=None):

    if dir is None:
        dir = select_directory('Select a directory of videos.')

    file_list = find('*.avi', dir)

    for f in tqdm(file_list):
        f_ = os.path.join(dir, f)
        print(f_)
        print(os.path.isfile(f_))
        _ = deinterlace(f_, do_rotation=True)


if __name__ == '__main__':
    deinter_dir()

