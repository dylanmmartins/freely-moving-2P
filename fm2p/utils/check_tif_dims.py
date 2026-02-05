# -*- coding: utf-8 -*-
"""
Check the dimensions of a tif. Currently reads every page until there are
no remaining pages. Not an efficient way to do it, but safe.

Author: DMM, last modified Oct 2025
"""


from PIL import Image
import os

import fm2p


def check_tiff_dims(tiff_path=None):
    """Check and print dimensions of each page in a multi-page TIFF file."""

    if tiff_path is None:
        tiff_path = fm2p.select_file(
            'Choose tif file',
            [('TIF', '.tif'), ('TIFF', '.tiff')]
        )

    if not os.path.exists(tiff_path):
        print(f"Error: File not found: {tiff_path}")
        return

    try:
        with Image.open(tiff_path) as img:
            page_count = 0
            print(f"File: {tiff_path}")

            while True:
                page_count += 1
                width, height = img.size
                if page_count == 0:
                    print(f"Page {page_count}: {width} x {height} pixels")

                try:
                    img.seek(img.tell() + 1)
                except EOFError:
                    break  # no more pages

            print(f"\nTotal pages: {page_count}")

    except Exception as e:
        print(f"Error reading TIFF file: {e}")


if __name__ == "__main__":

    check_tiff_dims()
