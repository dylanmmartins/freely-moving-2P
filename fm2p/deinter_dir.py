# -*- coding: utf-8 -*-
"""
Batch deinterlacing utility for a directory of AVI videos.

This script allows the user to select a directory, finds all .avi files within it,
and applies deinterlacing (and optional rotation) to each file using fm2p utilities.

Functions:
    deinter_dir(): Prompts user for a directory, finds all .avi files, and deinterlaces them.

Example usage:
    $ python deinter_dir.py

Author: DMM, June 2025
"""

import os
import fm2p
from tqdm import tqdm



def deinter_dir(dir=None):
    """
    Prompt the user to select a directory, find all .avi files, and deinterlace each one.
    Uses fm2p utilities for file selection, searching, and deinterlacing.
    """
    # Prompt user to select a directory containing videos
    if dir is None:
        dir = fm2p.select_directory('Select a directory of videos.')

    # Find all .avi files in the selected directory
    file_list = fm2p.find('*.avi', dir)

    # Process each file
    for f in tqdm(file_list):
        f_ = os.path.join(dir, f)
        print(f_)
        print(os.path.isfile(f_))
        _ = fm2p.deinterlace(f_, do_rotation=True)


if __name__ == '__main__':
    deinter_dir()

