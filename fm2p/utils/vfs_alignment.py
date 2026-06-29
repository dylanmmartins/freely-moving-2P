# -*- coding: utf-8 -*-
"""
fm2p/utils/vfs_alignment.py

Convenience wrapper for aligning a single animal's VFS map to the shared
reference and transforming area contours back into widefield space.

Functions
---------
vfs_alignment
    Align a .mat VFS file to the reference, return transform and contours.


DMM, March 2026
"""

import os
from .vfs_composite import align_single_vfs_to_reference
from .vfs_contours import getMapsFromMATFile, contours_to_aligned_signmap, resize_contours


def vfs_alignment(mat_path, ref_vfs_path=None, contours_path=None):
    """ Align a single VFS .mat file to the reference and transform contours.

    Parameters
    ----------
    mat_path : str
        Path to the animal's VFS .mat file.
    ref_vfs_path : str or None
        Path to the reference VFS composite TIF. Defaults to
        vfs_mean_composite.tif in the current directory.
    contours_path : str or None
        Path to the contours pkl file. Defaults to vfs_contours.pkl in the
        current directory.

    Returns
    -------
    transform_params : dict
        Affine transform parameters (dx, dy, rotation_deg, scale_factor).
    reverse_contours_resized : dict
        Area contours resized to 2048x2048 widefield space.
    """

    if ref_vfs_path is None:
        ref_vfs_path = os.path.join(os.path.curdir, 'vfs_mean_composite.tif')
    if contours_path is None:
        contours_path = os.path.join(os.path.curdir, 'vfs_contours.pkl')

    params = {
        'phaseMapFilterSigma': 0.5,
        'signMapFilterSigma': 2.
    }

    _, _, signMapf = getMapsFromMATFile(mat_path, params, filter=True)

    transform_params, aligned_vfs = align_single_vfs_to_reference(
        mat_path,
        ref_vfs_path
    )

    reverse_transformed_contours = contours_to_aligned_signmap(
        contours_path,
        transform_params
    )

    reverse_contours_resized = resize_contours(
        reverse_transformed_contours,
        source_shape=signMapf.shape,
        target_shape=(2048, 2048),
    )

    return transform_params, reverse_contours_resized
