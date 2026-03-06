

import os

from .vfs_composite import align_single_vfs_to_reference
from .vfs_contours import getMapsFromMATFile, contours_to_aligned_signmap, resize_contours


def vfs_alignment(mat_path, ref_vfs_path=None, contours_path=None):
    """Align an animal's VFS to the shared reference VFS.

    Parameters
    ----------
    mat_path : str
        Path to the additional_maps.mat file for this animal.
    ref_vfs_path : str, optional
        Path to the reference VFS tif (mean composite). Defaults to
        'vfs_mean_composite.tif' in the current directory.
    contours_path : str, optional
        Path to the reference contours pickle file. Defaults to
        'vfs_contours.pkl' in the current directory.

    Returns
    -------
    transform_params : dict
        Transform parameters with keys: dx, dy, rotation_deg, scale_factor,
        pearson_r, animal_id, reference_vfs_path.
    reverse_contours_resized : dict
        Visual area contours in the animal's widefield coordinate space
        (2048 x 2048). Keys are area names (e.g. 'V1', 'RL'), values are
        lists of (x, y) tuples.
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
