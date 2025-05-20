# -*- coding: utf-8 -*-
"""
Utility functions for working with axonal two-photon calcium data.

It includes functions for identifying independent axons based on correlation coefficients,
removing correlated axons, and filtering dF/F traces.

Functions
---------
get_independent_axons(matpath, cc_thresh=0.5, gcc_thresh=0.5, apply_dFF_filter=False)
    Identifies independent axons from a .mat file containing calcium imaging data.

Author: DMM, May 2025
"""


import numpy as np
from scipy import io
import itertools
import oasis

import fm2p
import imgtools


def get_independent_axons(matpath, cc_thresh=0.5, gcc_thresh=0.5, apply_dFF_filter=False):
    """ Identify independent axons from a .mat file containing calcium imaging data.
    
    Parameters
    ----------
    matpath : str
        Path to the .mat file containing calcium imaging data written by Matlab
        two-photon-calcium-post-processing pipeline (see README).
    cc_thresh : float, optional
        Threshold for between-cell correlation coefficient. Default is 0.5.
    gcc_thresh : float, optional
        Threshold for global frame correlation coefficient. Default is 0.5.
    apply_dFF_filter : bool, optional
        If True, apply a filter to the dF/F traces before calculating correlation coefficients.
        Default is False.

    Returns
    -------
    dFF_out : np.ndarray
        Filtered dF/F traces of independent axons.
    denoised_dFF : np.ndarray
        Denoised dF/F traces of independent axons.
    sps : np.ndarray
        Spike times of independent axons.
    usecells : list
        List of indices of independent axons.
    """

    fps = 7.49

    mat = io.loadmat(matpath)
    dff_ind = int(np.argwhere(np.asarray(mat['data'][0].dtype.names)=='DFF')[0])
    dFF = mat['data'].item()[dff_ind].copy()

    if apply_dFF_filter:
        # Smooth dFF traces of all cells
        all_smoothed_units = []
        for c in range(np.size(dFF, 0)):
            y = imgtools.nanmedfilt(
                    imgtools.rolling_average_1d(dFF[c,:], 11),
            25).flatten()
            all_smoothed_units.append(y)
        all_smoothed_units = np.array(all_smoothed_units)

    # Calculate all between-cell correlation coeffients
    perm_mat = np.array(list(itertools.combinations(range(np.size(dFF, 0)), 2)))
    cc_vec = np.zeros([np.size(perm_mat,0)])
    if apply_dFF_filter:
        for i in range(np.size(perm_mat,0)):
            cc_vec[i] = fm2p.corr2_coeff(
                all_smoothed_units[perm_mat[i,0]][np.newaxis,:],
                all_smoothed_units[perm_mat[i,1]][np.newaxis,:]
            )
    elif not apply_dFF_filter:
        for i in range(np.size(perm_mat,0)):
            cc_vec[i] = fm2p.corr2_coeff(
                dFF[perm_mat[i,0]][np.newaxis,:],
                dFF[perm_mat[i,1]][np.newaxis,:]
            )

    # Find axon pairs with cc above threshold
    check_index = np.where(cc_vec > cc_thresh)[0]
    exclude_inds = []

    for c in check_index:

        axon1 = perm_mat[c,0]
        axon2 = perm_mat[c,1]

        # Exclude the neuron with the lower integrated dFF
        if (np.sum(dFF[axon1,:]) < np.sum(dFF[axon2,:])):
            exclude_inds.append(axon1)
        elif (np.sum(dFF[axon1,:]) > np.sum(dFF[axon2,:])):
            exclude_inds.append(axon2)

    exclude_inds = list(set(exclude_inds))
    usecells = [c for c in list(np.arange(np.size(dFF,0))) if c not in exclude_inds]

    # Check correlation between global frame fluorescence and the dF/F of each axon.
    framef_ind = int(np.argwhere(np.asarray(mat['data'][0].dtype.names)=='frame_F')[0])
    frameF = mat['data'].item()[framef_ind].copy()

    gcc_vec = np.zeros([len(usecells)])
    for i,c in enumerate(usecells):
        gcc_vec[i] = fm2p.corr2_coeff(
            dFF[c,:][np.newaxis,:],
            frameF
        )

    # Find axons with gcc above threshold.
    axon_correlates_with_globalF = np.where(gcc_vec > gcc_thresh)[0]
    usecells_gcc = [c for c in usecells if c not in axon_correlates_with_globalF]

    # Remove axons with high correlation with global frame fluorescence.
    dFF_out = dFF.copy()[usecells_gcc, :]

    # Remove axons with high correlation with other axons.
    denoised_dFF, sps = fm2p.calc_dFF1(dFF_out, fps=fps)

    return dFF_out, denoised_dFF, sps, usecells

