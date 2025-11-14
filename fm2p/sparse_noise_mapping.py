import os
import fm2p
import argparse
import numpy as np

def calc_sparse_noise_STAs(preproc_path=None, stimpath=None):

    if preproc_path is None:
        preproc_path = fm2p.select_file(
            'Select preprocessed HDF file.',
            filetypes=[('HDF','.h5'),]
        )

    if stimpath is None:
        stimpath = r'T:\dylan\sparse_noise_sequence_v7.npy'
    stimulus = np.load(stimpath)[:,:,:,0]

    data = fm2p.read_h5(preproc_path)

    norm_spikes = data['s2p_spks']
    stimT = data['stimT']
    stimT = stimT - stimT[0]
    twopT = data['twopT']

    if stimulus.max() <= 1.0:
        stimulus = stimulus * 255.0

    # calculate the STAs
    sta_all, lag_axis, delay = fm2p.compute_calcium_sta_spatial(
        stimulus,
        norm_spikes,
        stimT,
        twopT,
        window=15,
        auto_delay=False
    )

    dict_out = {
        'STAs': sta_all,
        'lag_axis': lag_axis,
        'delay': delay
    }

    savepath = os.path.join(os.path.split(preproc_path)[0], 'sparse_noise_receptive_fields.h5')
    fm2p.write_h5(savepath, dict_out)

    return dict_out


def calc_sparse_noise_STA_reliability(preproc_path=None, sta_path=None, stimpath=None):

    if preproc_path is None:
        preproc_path = fm2p.select_file(
            'Select preprocessed HDF file.',
            filetypes=[('HDF','.h5'),]
        )

    if sta_path is None:
        sta_path = fm2p.select_file(
            'Select sparse noise STA HDF file.',
            filetypes=[('HDF','.h5'),]
        )

    if stimpath is None:
        stimpath = r'T:\dylan\sparse_noise_sequence_v7.npy'
    stimulus = np.load(stimpath)[:,:,:,0]

    STAs = fm2p.read_h5(sta_path)['STAs']
    data = fm2p.read_h5(preproc_path)

    norm_spikes = data['s2p_spks']
    stimT = data['stimT']
    stimT = stimT - stimT[0]
    twopT = data['twopT']

    if stimulus.max() <= 1.0:
        stimulus = stimulus * 255.0

    # calculate the STAs
    sta_all_a, sta_all_b, best_lags, split_corr = fm2p.compute_sta_chunked_reliability(
        stimulus,
        norm_spikes,
        stimT,
        twopT,
        STAs,
        thresh=None
    )

    dict_out = {
        'STAs_1': sta_all_a,
        'STAs_2': sta_all_b,
        'best_lags': best_lags,
        'split_corr': split_corr
    }

    savepath = os.path.join(os.path.split(preproc_path)[0], 'sparse_noise_reliability.h5')
    fm2p.write_h5(savepath, dict_out)

    return dict_out


def sparse_noise_mapping():
     
    parser = argparse.ArgumentParser()
    parser.add_argument('-cr', '--check_reliability', type=fm2p.str_to_bool, default=True) ### CHANGE BACK TO DEFAULT AS FALSE
    args = parser.parse_args()
    check_reliability = args.check_reliability

    if check_reliability:
        calc_sparse_noise_STA_reliability()

    elif not check_reliability:
        calc_sparse_noise_STAs()
     

if __name__ == '__main__':
    
    sparse_noise_mapping()