

import os
import numpy as np
import json
from .utils.files import read_h5, write_h5
from .utils.paths import find


def make_pooled_dataset(ref_contours_path=None, cohort_basepath=None):
    """Build the pooled dataset read by topography.py.

    Parameters
    ----------
    ref_contours_path : str, optional
        Path to the reference area-contours pickle file (in reference VFS
        space).  When provided the contours are embedded in the pooled data
        under keys 'ref_contour_<area_name>' so that topography.py can build
        its labeled_array without loading a separate file.
    cohort_basepath : str, optional
        Root directory containing cohort recording subdirectories.  Used to
        locate per-recording ffNLE output files
        ('pytorchGLM_predictions_v09b.h5').  Defaults to the standard cohort02
        path.
    """

    if cohort_basepath is None:
        cohort_basepath = '/home/dylan/Storage/freely_moving_data/_V1PPC/'

    uniref = read_h5('/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites/DMM056/animal_reference_260115_10h-06m-52s.h5')

    pooled = {
        'uniref': uniref
    }

    animal_dirs = ['DMM037', 'DMM041', 'DMM042', 'DMM056', 'DMM061']
    main_basepath = '/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites'

    for animal_dir in animal_dirs:
        pooled[animal_dir] = {}

    for animal_dir in animal_dirs:

        basepath = os.path.join(main_basepath, animal_dir)

        # Prefer the VFS-aligned composite produced by
        # register_animals_using_shared_template(); fall back to the legacy
        # manually-aligned composite if the new file is not present.
        try:
            transform_g2u = read_h5(
                find('vfs_aligned_composite_*.h5', basepath, MR=True))
        except Exception:
            if animal_dir == 'DMM056':
                transform_g2u = read_h5(find(
                    '*aligned_composite_local_to_global_transform.h5',
                    basepath, MR=True))
            else:
                transform_g2u = read_h5(
                    find('aligned_composite_*.h5', basepath, MR=True))

        messentials = fm2p.read_h5(
            fm2p.find('*_merged_essentials_v10.h5', basepath, MR=True))

        pooled[animal_dir]['messentials'] = messentials
        pooled[animal_dir]['transform'] = transform_g2u

        # Load ffNLE outputs (v09b) for each recording position if not already
        # present in the merged essentials.  The check looks for 'full_r2' in
        # any position's data, which is the primary output key written by
        # fit_test_ffNLE().
        ffnle_already_in_messentials = any(
            isinstance(messentials.get(k), dict) and 'full_r2' in messentials[k]
            for k in messentials
            if isinstance(k, str) and k.startswith('pos')
        )

        if not ffnle_already_in_messentials:
            ffnle_paths = fm2p.find(
                'pytorchGLM_predictions_v09b.h5',
                cohort_basepath
            )
            ffnle_paths = [f for f in ffnle_paths if animal_dir in f]
            if isinstance(ffnle_paths, str):
                ffnle_paths = [ffnle_paths]
            ffnle_by_pos = {}
            for fp in ffnle_paths:
                # Derive pos_key using the same logic as merge_animal_essentials():
                # grandparent directory name → last underscore-separated token.
                rec_dir = os.path.split(os.path.split(os.path.split(fp)[0])[0])[1]
                pos_key = rec_dir.split('_')[-1]
                try:
                    ffnle_by_pos[pos_key] = read_h5(fp)
                except Exception as e:
                    print(f'  Warning: could not load ffNLE for {animal_dir}/{pos_key}: {e}')
            if ffnle_by_pos:
                pooled[animal_dir]['ffnle'] = ffnle_by_pos

    # Embed reference contours (in reference VFS space) in the pooled data so
    # that topography.py can build its labeled_array without a separate file.
    if ref_contours_path is not None and os.path.isfile(ref_contours_path):

        # read in json file as ref_contours -- it is not a pickle file
        with open(ref_contours_path, 'r') as f:
            ref_contours = json.load(f)

        for area_name, coords in ref_contours.items():
            if coords is not None and len(coords) >= 3:
                pooled[f'ref_contour_{area_name}'] = np.array(coords)
        # Determine the reference VFS shape from the first available animal.
        ref_vfs_shape = np.array([400, 400])
        for animal_dir in animal_dirs:
            try:
                basepath = os.path.join(main_basepath, animal_dir)
                align_data = read_h5(
                    find('vfs_aligned_composite_*.h5', basepath, MR=True))
                ref_vfs_shape = align_data.get(
                    '_ref_vfs_shape', ref_vfs_shape)
                break
            except Exception:
                continue
        pooled['ref_vfs_shape'] = ref_vfs_shape

    savepath = '/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites/pooled_260318a.h5'
    print('Writing {}'.format(savepath))
    write_h5(savepath, pooled)


if __name__ == '__main__':

    make_pooled_dataset()

