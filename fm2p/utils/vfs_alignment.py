

import os
from .vfs_composite import align_single_vfs_to_reference
from .vfs_contours import getMapsFromMATFile, contours_to_aligned_signmap, resize_contours


def vfs_alignment(mat_path, ref_vfs_path=None, contours_path=None):

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
