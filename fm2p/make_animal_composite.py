# -*- coding: utf-8 -*-


import os
import argparse
import numpy as np

import fm2p


def make_animal_composite():

    parser = argparse.ArgumentParser()
    parser.add_argument('-animal', '--animal', type=str)
    args = parser.parse_args()

    cohort_dir = fm2p.select_directory(
        'Select cohort directory.'
    )

    animalID = args.animal

    animal_dict = {}

    preproc_paths = fm2p.find(
        '*{}*preproc.h5'.format(animalID),
        cohort_dir
    )
    for p in preproc_paths:
        main_key = os.path.split(os.path.split(os.path.split(p)[0])[0])[1]
        pos_key = main_key.split('_')[-1]
        r = fm2p.find('eyehead_revcorrs.h5', os.path.split(p)[0], MR=True)
        sn = os.path.join(os.path.split(os.path.split(p)[0])[0], 'sn1/sparse_noise_labels_gaussfit.npz')

        animal_dict[pos_key] = {
            'preproc': p,
            'revcorr': r,
            'sparsenoise': sn,
            'name': main_key
        }


    full_dict = {}

    all_pdata = []
    all_rdata = []
    all_pos = []
    all_cell_positions = []
    full_map = np.zeros([512*5, 512*5]) * np.nan

    row = 0
    col = 0
    for pos in range(1,26):
        pos_str = 'pos{:02d}'.format(pos)

        if pos_str not in list(animal_dict.keys()):
            if (pos%5)==0: # if you're at the end of a row
                col = 0
                row += 1
            else:
                col += 1
            continue

        pdata = fm2p.read_h5(animal_dict[pos_str]['preproc'])
        rdata = fm2p.read_h5(animal_dict[pos_str]['revcorr'])
        if os.path.isfile(animal_dict[pos_str]['sparsenoise']):
            sndata = np.load(animal_dict[pos_str]['sparsenoise'])
            snarr = np.concatenate([sndata['true_indices'][:,np.newaxis], sndata['pos_centroids']], axis=1)
        else:
            snarr = np.nan

        all_pdata.append(pdata)
        all_rdata.append(rdata)
        all_pos.append((row, col))

        singlemap = pdata['twop_ref_img']

        full_map[row*512 : (row*512)+512, col*512 : (col*512)+512] = singlemap

        cell_positions = np.zeros([len(pdata['cell_x_pix'].keys()), 2]) * np.nan

        for ki, k in enumerate(pdata['cell_x_pix'].keys()):
            # cellx = np.median(512 - pdata['cell_x_pix'][k]) + col*512
            cellx = np.median(pdata['cell_x_pix'][k]) + col*512
            celly = np.median(pdata['cell_y_pix'][k]) + row*512

            cell_positions[ki,:] = np.array([cellx, celly])

        full_dict[pos_str] = {
            'rdata': rdata,
            'tile_pos': np.array([row,col]),
            'cell_pos': cell_positions,
            'sn_cents': snarr
        }

        all_cell_positions.append(cell_positions)

        col += 1

        if (pos%5)==0: # if you're at the end of a row
            col = 0
            row += 1

    full_dict['full_map'] = full_map

    savepath = os.path.join(os.path.split(cohort_dir)[0], '{}_merged_essentials_v4.h5'.format(animalID))
    fm2p.write_h5(savepath, full_dict)



if __name__ == '__main__':

    make_animal_composite()

