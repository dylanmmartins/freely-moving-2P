# -*- coding: utf-8 -*-
"""
fm2p/utils/img_stacks.py

General image stack operations: loading, registration, normalization, and conversion.

Functions
---------
norm_arr
    Normalize an array to a specified [min, max] range.
register_stack_to_template
    Phase-cross-correlation registration of a stack to a template frame.
load_tif_stack
    Load a multi-page TIFF stack into a numpy array.
multipart_tif_to_avi
    Concatenate multi-part TIF files and write an AVI video.
read_tif_frame
    Read a single frame from a TIFF stack without loading the whole file.
read_tif_until
    Read the first N frames from a TIFF stack without loading beyond them.


DMM, February 2026
"""

import tifffile
import os
import cv2
import numpy as np
from tqdm import tqdm
import tifffile as tiff
import skimage.registration
import scipy.ndimage


def norm_arr(A, min_=None, max_=None):
    """ Normalize an array to a specified range.

    Parameters
    ----------
    A : np.ndarray
        Input array.
    min_ : float or None
        Lower bound; defaults to A.min().
    max_ : float or None
        Upper bound; defaults to A.max().

    Returns
    -------
    _a : np.ndarray
        Normalized array with the same shape as A.
    """

    if min_ is None:
        min_ = np.nanmin(A)
    if max_ is None:
        max_ = np.nanmax(A)

    _a = A + np.abs(min_)
    _a = _a / max_

    return _a


def register_stack_to_template(stack, template=None):
    """ Phase-cross-correlation registration of each frame to a template.

    Parameters
    ----------
    stack : np.ndarray, shape (N_frames, H, W)
    template : np.ndarray or None
        Reference frame; defaults to the first frame.

    Returns
    -------
    stack : np.ndarray
        Stack with each frame shifted to align with the template.
    extras : dict
        'x_shift', 'y_shift', 'shifterr' -- per-frame shift values and error.
    """

    print('Registering image stack to template.')

    if template is None:
        template = stack[0, :, :].copy()

    x_shift = np.zeros(np.size(stack, axis=0))
    y_shift = np.zeros(np.size(stack, axis=0))
    shifterr = np.zeros(np.size(stack, axis=0))

    for i in tqdm(range(np.size(stack, axis=0))):
        shift, error, _ = skimage.registration.phase_cross_correlation(
            reference_image=template,
            moving_image=stack[i, :, :],
            upsample_factor=4
        )

        x_shift[i] = shift[0]
        y_shift[i] = shift[1]
        shifterr[i] = error

        stack[i, :, :] = scipy.ndimage.shift(
            stack[i, :, :],
            shift,
            mode='constant',
            cval=np.nan
        )

    extras = {
        'x_shift': x_shift,
        'y_shift': y_shift,
        'shifterr': shifterr
    }

    return stack, extras


def load_tif_stack(path, rotate=False, ds=1.0, doReg=False, doNorm=False):
    """ Load a multi-page TIFF stack into a numpy array.

    Parameters
    ----------
    path : str
        Path to the multi-page TIF file.
    rotate : bool
        If True, rotate each frame 180 degrees.
    ds : float
        Downsample factor (e.g. 0.25 gives quarter-resolution). 1 = no downsampling.
    doReg : bool
        If True, register frames to the first frame via phase cross-correlation.
    doNorm : bool
        If True, normalize the stack to its global [min, max].

    Returns
    -------
    tif_array : np.ndarray
    """

    tif_array = tiff.imread(path)

    if rotate is True:
        for i in range(np.size(tif_array, axis=0)):
            tif_array[i, :, :] = np.flipud(np.fliplr(tif_array[i, :, :]))

    if ds != 1:
        tif_array = tif_array[:, ::int(1 / ds), ::int(1 / ds)]

    if doReg is True:
        tif_array, _ = register_stack_to_template(tif_array)

    if doNorm is True:
        tif_array = norm_arr(tif_array)

    return tif_array


def multipart_tif_to_avi(searchpath):
    """ Concatenate multi-part TIF files and write an AVI video.

    Parameters
    ----------
    searchpath : str
        Directory containing the individual TIF files.

    Returns
    -------
    video_savepath : str
        Path to the written AVI file.
    """

    filelist = [os.path.join(searchpath, f) for f in os.listdir(searchpath)]

    f = filelist[0]
    imgs = load_tif_stack(f, doReg=False, doNorm=False)
    total_frames = 0
    for f in filelist:
        total_frames += np.size(load_tif_stack(f, doReg=False, doNorm=False), 0)

    print('Found {} frames.'.format(total_frames))

    imgstack = np.empty([total_frames, 512, 640, 3])

    filled_to = 0
    print('Reading tif blocks...')
    for f in tqdm(filelist):
        im = load_tif_stack(f, doReg=False, doNorm=False)
        will_add = np.size(im, 0)
        imgstack[filled_to:filled_to + will_add, :, :, :] = im.copy()
        filled_to += will_add

    video_savepath = os.path.join(searchpath, 'full_video.avi')

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    video = cv2.VideoWriter(
        video_savepath,
        fourcc, 60.,
        (
            np.size(imgstack, 2),
            np.size(imgstack, 1)
        )
    )

    print('Writing avi file...')
    for i in tqdm(range(np.size(imgstack, 0))):

        im = imgstack[i, :, :, :]
        im = im.astype(np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        video.write(im)

    cv2.destroyAllWindows()
    video.release()

    return video_savepath


def read_tif_frame(file_path, frame_index):
    """ Read a single frame from a TIFF stack without loading the entire file.

    Parameters
    ----------
    file_path : str
        Path to the TIFF file.
    frame_index : int
        Zero-based frame index.

    Returns
    -------
    np.ndarray
        The requested frame.
    """

    with tifffile.TiffFile(file_path) as tif:
        num_pages = len(tif.pages)
        if frame_index < 0 or frame_index >= num_pages:
            raise IndexError('Frame index {} out of range (0-{})'.format(frame_index, num_pages - 1))
        frame = tif.pages[frame_index].asarray()
    return frame


def read_tif_until(file_path, last_frame=3600):
    """ Read the first N frames from a TIFF stack without loading beyond last_frame.

    Parameters
    ----------
    file_path : str
        Path to the TIFF file.
    last_frame : int
        Zero-based index of the last frame to read.

    Returns
    -------
    np.ndarray, shape (N_frames, H, W)
    """

    frames = []
    with tifffile.TiffFile(file_path) as tif:
        num_pages = len(tif.pages)
        stop = min(last_frame + 1, num_pages)
        for i in range(stop):
            frames.append(tif.pages[i].asarray())
    return np.stack(frames, axis=0)
