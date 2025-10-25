# split one suite2p output into two directories based on the length of the first tif stack

from PIL import Image
import numpy as np
import os
import shutil

import fm2p


def count_tif_frames(file_path: str) -> int:
    """
    Count the number of frames in a TIFF stack using Pillow.
    
    Args:
        file_path (str): Path to the TIFF file.
    
    Returns:
        int: Number of frames in the TIFF stack.
    """
    with Image.open(file_path) as img:
        count = 0
        try:
            while True:
                img.seek(count)
                count += 1
        except EOFError:
            pass
    return count


def split_suite2p_npy(file_path: str, split_index: int, out_dir_first: str, out_dir_second: str):
    """
    Splits a Suite2P .npy file (e.g., F.npy, Fneu.npy, spks.npy) into two parts 
    based on a given frame index, and saves them into two output directories 
    with the same filename.

    Args:
        file_path (str): Path to the input .npy file (e.g., 'F.npy').
        split_index (int): Frame index to split the recording.
        out_dir_first (str): Directory to save the first half.
        out_dir_second (str): Directory to save the second half.
    """

    # Load the array
    data = np.load(file_path, allow_pickle=True)

    # Split into two halves
    first_half = data[:, :split_index]
    second_half = data[:, split_index:]

    # Ensure output directories exist
    os.makedirs(out_dir_first, exist_ok=True)
    os.makedirs(out_dir_second, exist_ok=True)

    # Extract filename (e.g., 'F.npy', 'Fneu.npy', 'spks.npy')
    filename = os.path.basename(file_path)

    # Save out the two parts
    np.save(os.path.join(out_dir_first, filename), first_half)
    np.save(os.path.join(out_dir_second, filename), second_half)

    print(f"Saved {filename} split at index {split_index} into:\n - {out_dir_first}\n - {out_dir_second}")


def split_suite2p():

    s2p_dir = fm2p.select_directory('Select starting suite2p directory.')

    firsttif = fm2p.select_file(
        'Select first tif stack in merged data.',
        filetypes=[('TIF','.tif'),('TIFF','.tiff'),]
    )
    print('Counting frames in {}'.format(firsttif))
    split_ind = count_tif_frames(firsttif)
    
    save1_dir = fm2p.select_directory('Select first save directory (base).')
    save2_dir = fm2p.select_directory('Select second save directory (base).')

    if 'suite2p' not in save1_dir:
        save1_dir = os.path.join(save1_dir, 'suite2p/plane0')
        save2_dir = os.path.join(save2_dir, 'suite2p/plane0')
        os.makedirs(save1_dir)
        os.makedirs(save2_dir)

    for key in ['F.npy','Fneu.npy','spks.npy']:
        split_suite2p_npy(
            os.path.join(s2p_dir, key),
            split_ind,
            save1_dir,
            save2_dir
        )

    for key in ['iscell.npy','ops.npy','stat.npy']:
        source_file = os.path.join(s2p_dir, key)
        print('Copying {}'.format(key))
        shutil.copyfile(
            source_file,
            os.path.join(save1_dir, key)
        )
        shutil.copyfile(
            source_file,
            os.path.join(save2_dir, key)
        )

if __name__ == '__main__':
    split_suite2p()