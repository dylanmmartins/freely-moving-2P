# -*- coding: utf-8 -*-
"""
fm2p/utils/check_tif_dims.py

Utility for inspecting the dimensions and page count of a TIFF file.

Reads every page sequentially rather than using PIL's n_frames property,
which can silently mis-count for certain multi-page TIFFs.

Functions
---------
check_tiff_dims
    Print dimensions and total page count for a TIFF file.


DMM, October 2025
"""

from PIL import Image
import os

from .gui_funcs import select_file


def check_tiff_dims(tiff_path=None):
    """ Print width, height, and total page count for a TIFF file.

    Parameters
    ----------
    tiff_path : str or None
        Path to the TIF. If None, opens a file-chooser dialog.
    """

    if tiff_path is None:
        tiff_path = select_file(
            'Choose tif file',
            [('TIF', '.tif'), ('TIFF', '.tiff')]
        )
    if not os.path.exists(tiff_path):
        print('Error -- file not found: {}'.format(tiff_path))
        return
    try:
        with Image.open(tiff_path) as img:
            page_count = 0
            print('File: {}'.format(tiff_path))
            while True:
                page_count += 1
                width, height = img.size
                if page_count == 1:
                    print('Page {}: {} x {} pixels'.format(page_count, width, height))
                try:
                    img.seek(img.tell() + 1)
                except EOFError:
                    break
            print('Total pages: {}'.format(page_count))
    except Exception as e:
        print('Error reading TIFF file: {}'.format(e))


if __name__ == '__main__':

    check_tiff_dims()
