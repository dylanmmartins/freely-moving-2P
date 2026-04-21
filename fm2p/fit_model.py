# -*- coding: utf-8 -*-
"""
Fit the linear-nonlinear-Poisson model to neural/behavior data.

Functions
---------
fit_model(cfg_path=None)
    Fit the linear-nonlinear-Poisson model to neural/behavior data.
    
Example usage
-------------
    $ python -m fm2p.fit_model.py -cfg config.yaml
or alternatively, leave out the -cfg flag and select the config file from a file dialog box.
    $ python -m fm2p.fit_model.py

Authors: DMM, 2024
"""


if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

import os
import argparse
import numpy as np
from tqdm import tqdm

from .utils.gui_funcs import select_file

from .utils.ffNLE import fit_test_ffNLE
# from .utils.ffNLE_cell_summary import 



if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--rec', type='str', default=None)
    args.add_argument('--model' , type='str', default=None)
    args.parse_args()

    assert args.model is not None, 'Must specify model with --model flag.'

    if args.rec is None:
        rec = select_file('Select recording preproc.h5 file', filetypes=[('HDF5 files', '*.h5')])
    else:
        rec = args.rec

    if args.model == 'encoder':
        savepath = rec.replace('preproc.h5', 'encoder_model_v01.h5')
        fit_test_ffNLE(rec, savepath)

    elif args.model == 'encoder_summary':

        savepath = rec.replace('preproc.h5', 'encoder_cell_summary_v01.h5')
        fit_test_ffNLE(rec, savepath, summary_only=True)

    elif args.model == 'decoder':
        savepath = rec.replace('preproc.h5', 'decoder_model_v01.h5')
