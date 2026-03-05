

import os

from .vfs_composite import *
from .vfs_contours import *


def vfs_alignment(mat_path):
    # mat path should be the additional maps .mat file

    REF_VFS_PATH = os.path.join(os.path.curdir(), 'vfs_mean_composite.tif')
    CONTOUR_PATH = os.path.join(os.path.curdir(), 'vfs_contours.json')

    params = {
        'phaseMapFilterSigma': 0.5,
        'signMapFilterSigma': 2.
    }

    altPosMapf, aziPosMapf, signMapf = getMapsFromMATFile(
        mat_path,
        params,
        filter=True
    )

    # Align animal-specific signmap to reference vfs, save parameters
    transform_params, aligned_vfs = align_single_vfs_to_reference(
        mat_path,
        REF_VFS_PATH
    )


    # Reverse-apply the transformation parameters to the contours
    # Now the contours are in the original image space of the animal
    reverse_transformed_contours = contours_to_aligned_signmap(
        CONTOUR_PATH,
        transform_params
    )

    # If you need to resize the contours to the scale of the widefield image
    # (original contours will be on the scale of the signmap, i.e. (400, 400)):
    reverse_transformed_contours_resized = resize_contours(
        reverse_transformed_contours,
        source_shape=signMapf.shape,
        widefield_path=None, # pass the path to the animal's widefield image or...
        target_shape=(2048, 2048), # provide the target shape to resize to
    )

    return reverse_transformed_contours_resized





