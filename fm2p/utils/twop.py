# -*- coding: utf-8 -*-
"""
Two-photon calcium imaging data processing.

Classes
-------
TwoP
    Class for processing two-photon calcium imaging data.

Functions
---------
calc_dFF1
    Calculate dF/F and denoised fluorescence signal using Oasis.

Author: DMM, 2024
"""


import os
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.stats
import oasis

import fm2p


class TwoP():

    def __init__(self, recording_path='', recording_name='', cfg=None):
        """ Initialize the TwoP class.
        
        Parameters
        ----------
        recording_path : str, optional
            Path to the recording directory.
        recording_name : str, optional
            Name of the recording.
        cfg : str or dict, optional
            Path to the configuration file or a dictionary containing configuration parameters.
            If None, the default configuration file will be used.
        """
        
        self.recording_path = recording_path
        self.recording_name = recording_name

        if cfg is None:
            internals_config_path = os.path.join(fm2p.up_dir(__file__, 1), 'internals.yaml')
            with open(internals_config_path, 'r') as infile:
                cfg = yaml.load(infile, Loader=yaml.FullLoader)
        elif type(cfg)==str:
            with open(cfg, 'r') as infile:
                cfg = yaml.load(infile, Loader=yaml.FullLoader)

        self.cfg = cfg

        self.dt = 1. / cfg['twop_rate']

        self.dFF = None
        self.nCells = None

    def find_files(self):
        """ Find the files in the recording directory.
        """

        self.F = np.load(os.path.join(self.recording_path, r'suite2p/plane0/F.npy'), allow_pickle=True)
        self.Fneu = np.load(os.path.join(self.recording_path, r'suite2p/plane0/Fneu.npy'), allow_pickle=True)
        iscell = np.load(os.path.join(self.recording_path, r'suite2p/plane0/iscell.npy'), allow_pickle=True)
        spks = np.load(os.path.join(self.recording_path, r'suite2p/plane0/spks.npy'), allow_pickle=True)

        usecells = iscell[:,0]==1

        self.F = self.F[usecells, :]
        self.Fneu = self.Fneu[usecells, :]
        self.s2p_spks = spks[usecells, :]

    def add_files(self, F_path=None, Fneu_path=None, spikes_path=None, iscell_path=None, base_path=None):
        """ Add the files to the TwoP class manually from file paths.

        Either give a value for `base_path`, the suite2p directory i.e., /.../suite2p/plane0/
        Or, give the path to each of the individual files, i.e., F.npy, Fneu.npy, spks.npy, iscell.npy.

        Parameters
        ----------
        F_path : str, optional
            Path to the F.npy file.
        Fneu_path : str, optional
            Path to the Fneu.npy file.
        spikes_path : str, optional
            Path to the spks.npy file.
        iscell_path : str, optional
            Path to the iscell.npy file.
        base_path : str, optional
            Path to the base directory containing the suite2p files.
        """

        if (base_path is not None) and (F_path is None) and (Fneu_path is None) and (iscell_path is None) and (spikes_path is None):
            F_path = os.path.join(base_path, 'F.npy')
            Fneu_path = os.path.join(base_path, 'Fneu.npy')
            iscell_path = os.path.join(base_path, 'iscell.npy')
            spikes_path = os.path.join(base_path, 'spks.npy')

        self.F = np.load(F_path, allow_pickle=True)
        self.Fneu = np.load(Fneu_path, allow_pickle=True)
        iscell = np.load(iscell_path, allow_pickle=True)
        spks = np.load(spikes_path, allow_pickle=True)

        usecells = iscell[:,0]==1

        self.F = self.F[usecells, :]
        self.Fneu = self.Fneu[usecells, :]
        self.s2p_spks = spks[usecells, :]

        self.usecells = usecells

    def add_data(self, F, Fneu, spikes, iscell):
        """ Add the data to the TwoP class manually from numpy arrays.

        Parameters
        ----------
        F : np.ndarray
            Fluorescence data.
        Fneu : np.ndarray
            Neuropil fluorescence data.
        spikes : np.ndarray
            Spikes data.
        iscell : np.ndarray
            Cell mask data.
        """

        usecells = iscell[:,0]==1

        self.F = F[usecells, :]
        self.Fneu = Fneu[usecells, :]
        self.s2p_spks = spikes[usecells, :]
        self.nCells = np.size(self.F, 0)

        self.usecells = usecells


    def calc_dFF(self, neu_correction=0.7, oasis=True):
        """ Calculate dF/F and denoised fluorescence signal using Oasis.
        
        Parameters
        ----------
        neu_correction : float, optional
            Neuropil correction factor. Default is 0.7.
        oasis : bool, optional
            If True, use Oasis to denoise the fluorescence signal. Default is True.
            If False, spike inference and denoising is not performed and those keys will
            not be included in the output dictionary.

        Returns
        -------
        twop_dict : dict
            A dictionary containing the following keys:
                - 'raw_F0': Raw F0 values for each cell.
                - 'norm_F0': Normalized F0 values for each cell.
                - 'raw_F': Raw fluorescence data.
                - 'norm_F': Normalized fluorescence data.
                - 'raw_Fneu': Raw neuropil fluorescence data.
                - 'raw_dFF': Raw dF/F values for each cell.
                - 'norm_dFF': Normalized dF/F values for each cell.
                - 's2p_spks': Spikes data from Suite2P.
            Optionally, if `oasis` is True:
                - 'oasis_spks': Spikes data from Oasis.
                - 'denoised_dFF': Denoised dF/F values for each cell.
        """

        F = self.F
        Fneu = self.Fneu

        nCells, lenT = np.shape(F)

        norm_F = np.zeros([nCells, lenT])
        raw_dFF = np.zeros([nCells, lenT])
        norm_dFF = np.zeros([nCells, lenT])
        norm_F0 = np.zeros(nCells)
        raw_F0 = np.zeros(nCells)
        denoised_dFF = np.zeros([nCells, lenT])
        sps = np.zeros([nCells, lenT])

        for c in range(nCells):
            
            F_cell = F[c,:].copy()
            F_cell_neu = Fneu[c,:].copy()

            _f0_raw = scipy.stats.mode(F_cell, nan_policy='omit').mode

            # Raw DF/F
            _raw_dFF = (F_cell - _f0_raw) / _f0_raw * 100

            # Subtract neuropil
            _normF = F_cell - neu_correction * F_cell_neu + neu_correction * np.nanmean(F_cell_neu)

            _f0_norm = scipy.stats.mode(_normF, nan_policy='omit').mode

            # dF/F with neuropil correction
            norm_dFF[c,:] = (_normF - _f0_norm) / _f0_norm * 100

            if oasis:
                # Deconvolved spiking activity and denoised fluorescence signal
                g = oasis.functions.estimate_time_constant(norm_dFF[c,:].copy(), 1)
                denoised_dFF[c,:], sps[c,:] = oasis.oasisAR1(norm_dFF[c,:].copy(), g)

            norm_F[c,:] = _normF
            raw_dFF[c,:] = _raw_dFF
            norm_F0[c] = _f0_norm
            raw_F0[c] = _f0_raw

        twop_dict = {
            'raw_F0': raw_F0,
            'norm_F0': norm_F0,
            'raw_F': F,
            'norm_F': norm_F,
            'raw_Fneu': Fneu,
            'raw_dFF': raw_dFF,
            'norm_dFF': norm_dFF,
            's2p_spks': self.s2p_spks
        }

        if oasis:
            twop_dict['oasis_spks'] = sps
            twop_dict['denoised_dFF'] = denoised_dFF


        self.dFF = norm_dFF
        self.spikes = self.s2p_spks
        self.nCells = nCells

        return twop_dict


    def save_fluor(self, twop_dict):
        """ Save the fluorescence data to a HDF5 file.
        
        Parameters
        ----------
        twop_dict : dict
            A dictionary containing the fluorescence data to be saved.
            
        Returns
        -------
        _savepath : str
            The path to the saved HDF5 file.
        """

        savedir = os.path.join(self.recording_path, self.recording_name)
        _savepath = os.path.join(savedir, '{}_twophoton.h5'.format(self.recording_name))
        fm2p.write_h5(_savepath, twop_dict)

        return _savepath


    def calc_dFF_transients(self):

        sd_thresh = self.cfg['cell_sd_thresh']

        assert self.dFF is not None, "dFF must be calculated before calling this method."

        dFF = self.dFF.copy()

        dFF_transients = np.zeros_like(dFF)
        
        for c in range(self.nCells):
            sd = np.std(dFF[c,:])
            baseline_times = np.where(dFF[c,:] < (sd * sd_thresh))[0]
            mean_baseline = np.mean(dFF[c, baseline_times])
            sd_baseline = np.std(dFF[c, baseline_times])
            transient_times = np.where(dFF[c,:] > (sd_thresh * sd_baseline + mean_baseline))[0]
            dFF_transients[c, transient_times] = dFF[c, transient_times]

        self.dFF_transients = dFF_transients
        return dFF_transients
    

    def normalize_spikes(self):
        # set a maximum spike rate for each cell
        # then, normalize

        sd_thresh = self.cfg['cell_sd_thresh']

        assert self.nCells > 0
        assert self.spikes is not None

        spikes = self.spikes.copy()

        for c in range(self.nCells):
            sp_ = self.spikes[c, :]
            std_ = np.std(sp_)
            mean_ = np.mean(sp_)

            sp_[sp_ > (mean_ + std_ * sd_thresh)] = mean_ + std_ * sd_thresh

            spikes[c, :] = sp_

        # Normalize
        spikes = spikes / np.max(spikes, axis=1, keepdims=True)
        
        self.cleanspikes = spikes
        
        return spikes
    
    
    def get_recording_props(self, stat, ops):
        # inputs should be paths

        if type(stat)==str and type(ops)==str:
            stat = np.load(stat, allow_pickle=True)
            ops = np.load(ops, allow_pickle=True)
        elif type(stat)==np.ndarray and type(ops)==np.ndarray:
            pass

        recording_props = {
            'twop_mean_img': ops.item()['meanImg'],
            'twop_ref_img': ops.item()['refImg'],
            'twop_max_proj': ops.item()['max_proj'],
            'twop_enhanced_mean_img': ops.item()['meanImgE']
        }

        cell_x_pix = []
        cell_y_pix = []

        itercells = np.arange(len(stat))[self.usecells]

        for c in itercells:
            x = stat[c]['xpix']
            y = stat[c]['ypix']

            cell_x_pix.append(x)
            cell_y_pix.append(y)

        recording_props['cell_x_pix'] = cell_x_pix
        recording_props['cell_y_pix'] = cell_y_pix

        self.recording_props = recording_props

        return recording_props


def calc_dFF1(dFF, neu_correction=0.7, fps=7.49):
    """ Calculate dF/F and denoised fluorescence signal using Oasis.
    
    This is a simplified version of the calc_dFF method in the TwoP class,
    and can be used for dFF data preprocessed in Matlab.
    
    Parameters
    ----------
    dFF : np.ndarray
        dF/F data for each cell. Shape: (cells, time).
    neu_correction : float, optional
        Neuropil correction factor. Default is 0.7.
    fps : float, optional
        Frames per second. Default is 7.49.
    
    Returns
    -------
    denoised_dFF : np.ndarray
        Denoised dF/F data for each cell. Shape: (cells, time).
    sps : np.ndarray
        Spikes data for each cell. Shape: (cells, time).
    """

    nCells, lenT = np.shape(dFF)

    denoised_dFF = np.zeros([nCells, lenT])
    sps = np.zeros([nCells, lenT])

    for c in range(nCells):

        g = oasis.functions.estimate_time_constant(dFF[c,:].copy(), 1)
        denoised_dFF[c,:], sps[c,:] = oasis.oasisAR1(dFF[c,:].copy(), g)

    return denoised_dFF, sps