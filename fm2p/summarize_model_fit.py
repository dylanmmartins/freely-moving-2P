

import os
import numpy as np
import argparse

import fm2p

def summarize_model_fit():

    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', type=str, default=None)
    parser.add_argument('--preproc', type=str, default=None)
    parser.add_argument('--nulldir', type=str, default=None)
    parser.add_argument('-v', '--version', type=str, default='00')
    args = parser.parse_args()

    if args.modeldir is None:
        print('Choose model fit directory (subdirectory within a recording directory).')
        model_dir = fm2p.select_directory(
            title='Choose a model fit directory.'
        )
    else:
        model_dir = args.modeldir

    if args.preproc is None:
        print('Choose a preprocessing file.')
        preproc_path = fm2p.select_file(
            title='Choose a preprocessing file.',
            filetypes=[('H5','.h5')]
        )
    else:
        preproc_path = args.preproc

    if args.nulldir is None:
        print('Select the null model fit directory.')
        null_dir = fm2p.select_directory(
            title='Select the null model fit directory.'
        )
    else:
        null_dir = args.nulldir


    print('Reading in model fit results.')

    model = fm2p.read_models(model_dir)
    savepath = os.path.join(model_dir, 'cell_summary_LNP_v{}.pdf'.format(args.version))

    ego_bins = np.linspace(-180, 180, 19)
    retino_bins = np.linspace(-180, 180, 19) # 20 deg bins
    pupil_bins = np.linspace(45, 95, 11) # 5 deg bins

    var_bins = [pupil_bins, retino_bins, ego_bins]

    print('Reading null model fit results.')
    model_null = fm2p.read_models(null_dir)

    print('Reading in preprocessed experiment data.')
    preprocdata = fm2p.read_h5(preproc_path)

    model_save_key = os.path.split(model_dir)[1]
    if 'neg' in model_save_key:
        shift_val = -int(model_save_key.split('neg')[1])
    elif 'pos' in model_save_key:
        shift_val = int(model_save_key.split('pos')[1])
    else:
        print('Could not find pos/neg key in model directory.')

    print('Writing summary file.')
    fm2p.write_detailed_cell_summary(
        model,
        var_bins=var_bins,
        savepath=savepath,
        preprocdata=preprocdata,
        null_data=model_null,
        lag_val=shift_val
    )

if __name__ == '__main__':

    summarize_model_fit()
