import os
import fm2p
import argparse
import numpy as np
import gc

def calc_sparse_noise_STAs(preproc_path=None, stimpath=None):

    if preproc_path is None:
        preproc_path = fm2p.select_file(
            'Select preprocessed HDF file.',
            filetypes=[('HDF','.h5'),]
        )

    if stimpath is None:
        stimpath = r'T:\goard_lab\sparse_noise_stimuli\sparse_noise_sequence_v7.npy'
    stimulus = np.load(stimpath)[:,:,:,0]

    data = fm2p.read_h5(preproc_path)

    norm_spikes = data['s2p_spks']
    stimT = data['stimT']
    stimT = stimT - stimT[0]
    twopT = data['twopT']

    if stimulus.max() <= 1.0:
        stimulus = stimulus * 255.0

    n_cells = np.size(norm_spikes, 0)

    # calculate the STAs
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
        stimpath = r'T:\dylan\sparse_noise_sequence_v7.npy'

    data = fm2p.read_h5(preproc_path)
    stimulus = np.load(stimpath)[:,:,:,0]
    spikes = data['s2p_spks']
    stimT = data['stimT']
    stimT = stimT - stimT[0]
    twopT = data['twopT']

    if stimulus.max() <= 1.0:
        stimulus = stimulus * 255.0

    n_cells = np.size(spikes, 0)
    # STAs = fm2p.read_h5(sta_path)['STAs']

    # best_lags = np.zeros(n_cells)
    # for c in range(n_cells):
    #     lagmax = np.zeros(np.size(STAs, 1)) * np.nan
    #     for l in range(np.size(STAs, 1)):
    #         lagmax[l] = np.nanmax(np.abs(STAs[c,l,:]))
    #     best_lags[c] = np.nanargmax(lagmax)

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
        'corr': r
    }

    savepath = os.path.join(os.path.split(preproc_path)[0], 'sparse_noise.h5')
    print('  -> Writing {}'.format(savepath))
    fm2p.write_h5(savepath, dict_out)


def sparse_noise_mapping():
     
    parser = argparse.ArgumentParser()
    parser.add_argument('-cr', '--check_reliability', type=fm2p.str_to_bool, default=True) ### CHANGE BACK TO DEFAULT AS FALSE
    args = parser.parse_args()
    check_reliability = args.check_reliability

    if check_reliability:
        calc_sparse_noise_STA_reliability(
        )

    elif not check_reliability:
        calc_sparse_noise_STAs(
        )
     

if __name__ == '__main__':
    
    sparse_noise_mapping()