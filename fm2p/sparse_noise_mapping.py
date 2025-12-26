import os
import fm2p
import argparse
import numpy as np
import gc
import platform
from tqdm import tqdm

os_name = platform.system()

def calc_sparse_noise_STAs(preproc_path=None, stimpath=None):

    if preproc_path is None:
        preproc_path = fm2p.select_file(
            'Select preprocessed HDF file.',
            filetypes=[('HDF','.h5'),]
        )

    if stimpath is None:
        if os_name == "Linux":
            stimpath = '/home/dylan/Documents/sparse_noise_sequence_v7.npy'
        elif os_name == "Windows":
            stimpath = r'J:\sparse_noise\sparse_noise_sequence_v7.npy'

    stimulus = np.load(stimpath)[:,:,:,0]

    data = fm2p.read_h5(preproc_path)

    norm_spikes = data['s2p_spks']
    stimT = data['stimT']
    stimT = stimT - stimT[0]
    twopT = data['twopT']

    if stimulus.max() <= 1.0:
        stimulus = stimulus * 255.0

    n_cells = np.size(norm_spikes, 0)

    sta_all, lag_axis, delay = fm2p.compute_calcium_sta_spatial(
        stimulus,
        norm_spikes,
        stimT,
        twopT,
        window=15,
        delay=np.zeros(n_cells)
    )

    dict_out = {
        'STAs': sta_all,
        'lag_axis': lag_axis,
        'delay': delay
    }

    savepath = os.path.join(os.path.split(preproc_path)[0], 'sparse_noise_receptive_fields.h5')
    fm2p.write_h5(savepath, dict_out)

    return dict_out


def calc_sparse_noise_STA_reliability(preproc_path=None, stimpath=None):

    if preproc_path is None:
        preproc_path = fm2p.select_file(
            'Select preprocessed HDF file.',
            filetypes=[('HDF','.h5'),]
        )

    if stimpath is None:
        if os_name == 'Linux':
            stimpath = '/home/dylan/Documents/sparse_noise_sequence_v7.npy'
        elif os_name == 'Windows':
            stimpath = r'J:\sparse_noise\sparse_noise_sequence_v7.npy'

    print('  -> Loading preprocessed data.')
    data = fm2p.read_h5(preproc_path)

    print('  -> Loading stimulus.')

    stimulus = np.load(stimpath)[:,:,:,0]
    spikes = data['s2p_spks']
    stimT = data['stimT']
    stimT = stimT - stimT[0]
    twopT = data['twopT']

    if stimulus.max() <= 1.0:
        stimulus = stimulus * 255.0

    n_cells = np.size(spikes, 0)

    STA, STA1, STA2, r, lags = fm2p.compute_split_STAs(
        stimulus,
        spikes,
        stimT,
        twopT,
        window=13,
        delay=np.zeros(n_cells)
    )

    dict_out = {
        'STA': STA,
        'STA1': STA1,
        'STA2': STA2,
        'lags': lags,
        'jcorr': r
    }

    savepath = os.path.join(os.path.split(preproc_path)[0], 'sparse_noise.h5')
    print('  -> Writing {}'.format(savepath))
    fm2p.write_h5(savepath, dict_out)


def calc_STA_correlation(splitSTAs_path):

    dict_in = fm2p.read_h5(splitSTAs_path)

    STA = dict_in['STA']
    STA1 = dict_in['STA1']
    STA2 = dict_in['STA2']

    n_cells = np.size(STA,0)

    corr1 = np.zeros(n_cells) * np.nan
    corr2 = np.zeros(n_cells) * np.nan
    isresp = np.zeros_like(corr1) * np.nan

    for c in tqdm(range(n_cells)):
        corr1[c] = fm2p.corr2_coeff(
            STA[c].reshape([768,1360]), STA1[c].reshape([768,1360])
        )
        corr2[c] = fm2p.corr2_coeff(
            STA[c].reshape([768,1360]), STA2[c].reshape([768,1360])
        )
        isresp[c] = int((corr1[c] >= 0.10) and (corr2[c] >= 0.10))
        
    arr_out = np.vstack([corr1, corr2, isresp])

    savepath = os.path.join(os.path.split(splitSTAs_path)[0], 'reliability_stats_v2.npy')
    np.save(savepath, arr_out)


def sparse_noise_mapping(method, prepath=None):

    if prepath is None:
        prepath = fm2p.select_file(
            'Select sparse noise HDF file.',
            [('HDF','.h5'),]
        )

    if method == 'splits':
        calc_sparse_noise_STA_reliability(
            prepath
        )

    elif method == 'rel':
        calc_STA_correlation(
            prepath
        )

    elif method == 'single': # don't use this anymore
        calc_sparse_noise_STAs(
            prepath
        )
     

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-method', '--method', type=str, default='splits') # splits
    parser.add_argument('-path', '--path', type=str, default=None)
    args = parser.parse_args()
    
    sparse_noise_mapping(args.method, args.path)

