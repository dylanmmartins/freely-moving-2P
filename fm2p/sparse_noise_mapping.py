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

    norm_spikes = data['s2p_spks'][[14,15,16],:]
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
        delay='none'
    )

    dict_out = {
        'STAs': sta_all,
        'lag_axis': lag_axis,
        'delay': delay
    }

    # savepath = os.path.join(os.path.split(preproc_path)[0], 'sparse_noise_receptive_fields.h5')
    # fm2p.write_h5(savepath, dict_out)

    return dict_out


def calc_sparse_noise_STA_reliability(preproc_path=None, sta_path=None, stimpath=None):

    D = fm2p.read_h5(preproc_path)
    stimulus = np.load(stimpath)[:,:,:,0]
    spikes = D['s2p_spks']
    stimT = D['stimT']
    twopT = D['twopT']

    if stimulus.max() <= 1.0:
        stimulus = stimulus * 255.0

    n_cells = np.size(spikes, 0)
    STAs = fm2p.read_h5(sta_path)['STAs']

    best_lags = np.zeros(n_cells)
    for c in range(n_cells):
        lagmax = np.zeros(np.size(STAs, 1)) * np.nan
        for l in range(np.size(STAs, 1)):
            lagmax[l] = np.nanmax(np.abs(STAs[c,l,:]))
        best_lags[c] = np.nanargmax(lagmax)

    STA1, STA2, r = fm2p.compute_split_STAs(
        stimulus,
        spikes[[14,15],:],
        stimT,
        twopT,
        window=0,
        delay=best_lags
    )

#     if preproc_path is None:
#         preproc_path = fm2p.select_file(
#             'Select preprocessed HDF file.',
#             filetypes=[('HDF','.h5'),]
#         )

#     if sta_path is None:
#         sta_path = fm2p.select_file(
#             'Select sparse noise STA HDF file.',
#             filetypes=[('HDF','.h5'),]
#         )

#     print('  -> Loading sparse noise stimulus.')
#     if stimpath is None:
#         stimpath = r'T:\goard_lab\sparse_noise_stimuli\sparse_noise_sequence_v7.npy'
#     stimulus = np.load(stimpath)[:,:,:,0]

#     print('  -> Loading full STAs and preprocessed data.')
#     STAs = fm2p.read_h5(sta_path)['STAs']
#     data = fm2p.read_h5(preproc_path)

#     spikes = data['s2p_spks']

#     stimT = data['stimT']
#     stimT = stimT - stimT[0]

#     twopT = data['twopT']

#     if stimulus.max() <= 1.0:
#         stimulus = stimulus * 255.0

#     n_cells = np.size(spikes, 0)

#     print('  -> Calculating best lag for each cell.')
#     # find best STA lag so there is only one STA per cell to deal with
#     best_lags = np.zeros(n_cells)

#     for c in range(n_cells):
#         lagmax = np.zeros(16) * np.nan
#         for l in range(16):
#             lagmax[l] = np.nanmax(np.abs(STAs[c,l,:]))
#         best_lags[c] = np.nanargmax(lagmax)

#     best_sta = np.zeros([
#         n_cells,
#         np.size(STAs,2)
#     ])
#     for c in range(n_cells):
#         best_sta[c,:] = STAs[c, int(best_lags[c]), :]

#     del STAs
#     print('  -> Clearing space in memory.')
#     gc.collect()

#     # calculate the STAs
#     sta_all_a, sta_all_b, split_corr = fm2p.compute_split_STAs(
#         stimulus,
#         spikes[[14,15,16],:],
#         stimT,
#         twopT,
#         best_lags[[14,15,16]]
#     )

#     dict_out = {
#         'STAs_1': sta_all_a,
#         'STAs_2': sta_all_b,
#         'best_lags': best_lags,
#         'split_corr': split_corr
#     }

#     savepath = os.path.join(os.path.split(preproc_path)[0], 'sparse_noise_reliability.h5')
#     print('  -> Saving results to {}'.format(savepath))
#     fm2p.write_h5(savepath, dict_out)

#     return dict_out





def sparse_noise_mapping():
     
    parser = argparse.ArgumentParser()
    parser.add_argument('-cr', '--check_reliability', type=fm2p.str_to_bool, default=True) ### CHANGE BACK TO DEFAULT AS FALSE
    args = parser.parse_args()
    check_reliability = args.check_reliability

    if check_reliability:
        calc_sparse_noise_STA_reliability(
            r'K:\goard_lab\Mini2P_V1PPC_cohort02\251016_DMM_DMM061_pos18\sn1\sn1_preproc.h5',
            r'K:\goard_lab\Mini2P_V1PPC_cohort02\251016_DMM_DMM061_pos18\sn1\sparse_noise_receptive_fields.h5',
            r'T:\goard_lab\sparse_noise_stimuli\sparse_noise_sequence_v7.npy'
        )

    elif not check_reliability:
        calc_sparse_noise_STAs()
     

if __name__ == '__main__':
    
    sparse_noise_mapping()