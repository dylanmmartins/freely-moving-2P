

import os
import fm2p
from tqdm import tqdm



def deinter_dir():

    dir = fm2p.select_directory('Select a directory of videos.')

    file_list = fm2p.find('*.avi', dir)

    for f in tqdm(file_list):
        f_ = os.path.join(dir, f)
        print(f_)
        print(os.path.isfile(f_))
        _ = fm2p.deinterlace(f_, do_rotation=True)


if __name__ == '__main__':
    deinter_dir()

