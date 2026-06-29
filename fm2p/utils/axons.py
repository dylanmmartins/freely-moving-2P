# -*- coding: utf-8 -*-
"""
fm2p/utils/axons.py

Functions for identifying independent axons from two-photon calcium data.

Axonal recordings often contain many overlapping ROIs from the same axon.
These functions deduplicate them either by dropping the lower-fluorescence
member of each correlated pair, or by grouping them and averaging.

Functions
---------
get_single_independent_axons
    Drop the lower-fluorescence member of each correlated pair.
get_grouped_independent_axons
    Group correlated ROIs by connected components, average each group.
get_independent_axons
    Top-level entry point: loads data, runs kurtosis threshold, then dispatches
    to single or grouped deduplication.
threshold_kurtosis
    Return indices of ROIs whose kurtosis exceeds a threshold.


DMM, May 2025
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy import io
import itertools
from collections import defaultdict

from .filter import nanmedfilt, rolling_average_1d
from .correlation import corr2_coeff
from .twop import TwoP, calc_inf_spikes
from .helper import compute_kurtosis


def get_single_independent_axons(
        dFF, cc_thresh=0.5, gcc_thresh=0.5, apply_dFF_filter=False,
        fps=7.5, frame_means=None):
    """ Drop the lower-fluorescence ROI from each correlated pair.

    For each pair whose correlation exceeds cc_thresh, keeps the one with the
    higher integrated dFF. Optionally also removes ROIs that correlate too
    strongly with the global frame fluorescence (neuropil proxy).

    Parameters
    ----------
    dFF : np.ndarray, shape (N_rois, N_frames)
        Raw dF/F traces.
    cc_thresh : float
        Pairwise correlation threshold above which one ROI is dropped.
    gcc_thresh : float
        Global-fluorescence correlation threshold above which an ROI is dropped.
        Only applied when frame_means is provided.
    apply_dFF_filter : bool
        Smooth traces with a rolling median before computing correlations.
    fps : float
        Frame rate; passed to calc_inf_spikes.
    frame_means : np.ndarray or None
        Mean fluorescence per frame across all pixels; used as neuropil proxy.

    Returns
    -------
    dFF_out : np.ndarray
        dF/F traces of kept ROIs.
    denoised_dFF : np.ndarray
        Denoised traces from cascade/OASIS.
    sps : np.ndarray
        Inferred spike rates.
    usecells : list of int
        Original ROI indices that survived.
    """

    if apply_dFF_filter:
        all_smoothed_units = []
        for c in range(np.size(dFF, 0)):
            y = nanmedfilt(
                    rolling_average_1d(dFF[c, :], 11),
            25).flatten()
            all_smoothed_units.append(y)
        all_smoothed_units = np.array(all_smoothed_units)

    perm_mat = np.array(list(itertools.combinations(range(np.size(dFF, 0)), 2)))
    cc_vec = np.zeros([np.size(perm_mat, 0)])

    if apply_dFF_filter:
        for i in range(np.size(perm_mat, 0)):
            cc_vec[i] = corr2_coeff(
                all_smoothed_units[perm_mat[i, 0]][np.newaxis, :],
                all_smoothed_units[perm_mat[i, 1]][np.newaxis, :]
            )
    elif not apply_dFF_filter:
        for i in range(np.size(perm_mat, 0)):
            cc_vec[i] = corr2_coeff(
                dFF[perm_mat[i, 0]][np.newaxis, :],
                dFF[perm_mat[i, 1]][np.newaxis, :]
            )

    check_index = np.where(cc_vec > cc_thresh)[0]
    exclude_inds = []

    for c in check_index:

        axon1 = perm_mat[c, 0]
        axon2 = perm_mat[c, 1]

        # Keep the ROI with higher integrated fluorescence; it is more likely
        # to represent the actual axon rather than a partial overlap.
        if (np.sum(dFF[axon1, :]) < np.sum(dFF[axon2, :])):
            exclude_inds.append(axon1)
        elif (np.sum(dFF[axon1, :]) > np.sum(dFF[axon2, :])):
            exclude_inds.append(axon2)

    exclude_inds = list(set(exclude_inds))
    usecells = [c for c in list(np.arange(np.size(dFF, 0))) if c not in exclude_inds]

    if frame_means is not None:
        # Remove ROIs that track global fluorescence -- likely neuropil contamination.
        gcc_vec = np.zeros([len(usecells)])
        for i, c in enumerate(usecells):
            gcc_vec[i] = corr2_coeff(
                dFF[c, :][np.newaxis, :],
                frame_means
            )

        axon_correlates_with_globalF = np.where(gcc_vec > gcc_thresh)[0]
        usecells_gcc = [c for c in usecells if c not in axon_correlates_with_globalF]

        dFF_out = dFF.copy()[usecells_gcc, :]

    elif frame_means is None:
        dFF_out = dFF.copy()[usecells, :]

    denoised_dFF, sps = calc_inf_spikes(dFF_out, fps=fps)

    return dFF_out, denoised_dFF, sps, usecells


def get_grouped_independent_axons(
        dFF, cc_thresh=0.25, gcc_thresh=0.70, apply_dFF_filter=False,
        fps=7.5
    ):
    """ Group correlated ROIs by connected components and average each group.

    Builds an adjacency graph where ROI pairs with correlation > cc_thresh are
    connected, finds connected components (axonal groups), averages traces
    within each group, then drops groups whose mean trace correlates too
    strongly with global fluorescence.

    Parameters
    ----------
    dFF : np.ndarray, shape (N_rois, N_frames)
        Raw dF/F traces.
    cc_thresh : float
        Pairwise correlation threshold for building the adjacency graph.
    gcc_thresh : float
        Global-fluorescence correlation threshold; groups above this are dropped.
    apply_dFF_filter : bool
        Smooth traces before computing pairwise correlations.
    fps : float
        Frame rate; passed to calc_inf_spikes.

    Returns
    -------
    dFF_out : np.ndarray
        Averaged dF/F trace per kept group.
    denoised_dFF : np.ndarray
        Denoised traces.
    sps : np.ndarray
        Inferred spike rates.
    kept_groups : list of list of int
        Original ROI indices belonging to each kept group.
    """

    if apply_dFF_filter:
        all_smoothed_units = []
        for c in range(np.size(dFF, 0)):
            y = nanmedfilt(
                rolling_average_1d(dFF[c, :], 11),
                25
            ).flatten()
            all_smoothed_units.append(y)
        all_smoothed_units = np.array(all_smoothed_units)
    else:
        all_smoothed_units = dFF

    perm_mat = np.array(list(itertools.combinations(range(np.size(dFF, 0)), 2)))
    cc_vec = np.zeros([np.size(perm_mat, 0)])
    for i in range(np.size(perm_mat, 0)):
        cc_vec[i] = corr2_coeff(
            all_smoothed_units[perm_mat[i, 0]][np.newaxis, :],
            all_smoothed_units[perm_mat[i, 1]][np.newaxis, :]
        )

    # Build undirected adjacency from correlated pairs, then find connected
    # components via depth-first search.
    adjacency = defaultdict(set)
    for idx, c in enumerate(perm_mat):
        if cc_vec[idx] > cc_thresh:
            adjacency[c[0]].add(c[1])
            adjacency[c[1]].add(c[0])

    visited = set()
    groups = []
    for node in range(np.size(dFF, 0)):
        if node not in visited:
            stack = [node]
            group = set()
            while stack:
                n = stack.pop()
                if n not in visited:
                    visited.add(n)
                    group.add(n)
                    stack.extend(adjacency[n] - visited)
            groups.append(sorted(list(group)))

    averaged_traces = []
    for group in groups:
        avg_trace = np.mean(dFF[group, :], axis=0)
        averaged_traces.append(avg_trace)
    averaged_traces = np.array(averaged_traces)

    frame_means = np.mean(dFF, axis=0)

    # Drop groups that track global fluorescence (neuropil).
    gcc_vec = np.zeros([len(averaged_traces)])
    for i, trace in enumerate(averaged_traces):
        try:
            gcc_vec[i] = corr2_coeff(
                trace[np.newaxis, :],
                frame_means[np.newaxis, :]
            )
        except ValueError:
            gcc_vec[i] = corr2_coeff(
                trace,
                frame_means
            )

    keep_inds = [i for i in range(len(averaged_traces)) if gcc_vec[i] <= gcc_thresh]

    dFF_out = averaged_traces[keep_inds, :]
    kept_groups = [groups[i] for i in keep_inds]

    if len(keep_inds) < 1:
        print('No independent axons found.')
        print('{} ROIs evaluated, grouped into {} clusters.'.format(
            np.size(dFF, 0), len(averaged_traces)
        ))
        print('Check cell segmentation.')
        print('Exiting -- preprocessing cannot continue without at least one independent axon.')
        quit()

    denoised_dFF, sps = calc_inf_spikes(dFF_out, fps=fps, neu_correction=0)

    return dFF_out, denoised_dFF, sps, kept_groups


def get_independent_axons(
        cfg, s2p_dict=None, matpath=None, merge_duplicates=True,
        cc_thresh=0.25, gcc_thresh=1.0, apply_dFF_filter=False
    ):
    """ Load axonal data, apply kurtosis filter, and deduplicate ROIs.

    Top-level entry point. Dispatches to get_single_independent_axons or
    get_grouped_independent_axons depending on merge_duplicates.

    Parameters
    ----------
    cfg : dict
        Config dict; must contain 'twop_rate' (float).
    s2p_dict : dict or None
        Suite2p output dict with keys 'F', 'Fneu', 'spks', 'iscell',
        'ops_path', 'bin_path'. Mutually exclusive with matpath.
    matpath : str or None
        Path to legacy .mat file with 'DFF' and 'frame_F' fields.
    merge_duplicates : bool
        If True, group and average correlated ROIs. If False, drop the
        lower-fluorescence member of each pair.
    cc_thresh : float
        Pairwise correlation threshold.
    gcc_thresh : float
        Global-fluorescence correlation threshold.
    apply_dFF_filter : bool
        Smooth before computing correlations.

    Returns
    -------
    Same returns as get_single_independent_axons or get_grouped_independent_axons.
    """

    fps = cfg['twop_rate']

    if s2p_dict is not None:
        twop_data = TwoP(cfg)
        twop_data.add_data(
            s2p_dict['F'],
            s2p_dict['Fneu'],
            s2p_dict['spks'],
            s2p_dict['iscell'],
        )
        twop_dict_out = twop_data.calc_dFF(neu_correction=0.3)
        dFF = twop_dict_out['raw_dFF']

        frame_means = twop_data.calc_frame_mean_across_time(
            s2p_dict['ops_path'],
            s2p_dict['bin_path']
        )

    elif matpath is not None:
        mat = io.loadmat(matpath)
        try:
            dff_ind = int(np.argwhere(np.asarray(mat['data'][0].dtype.names) == 'DFF')[0])
        except IndexError as e:
            print(e)
            print('No cells found in this recording -- check cell segmentation.')
            quit()
        dFF = mat['data'].item()[dff_ind].copy()

        framef_ind = int(np.argwhere(np.asarray(mat['data'][0].dtype.names) == 'frame_F')[0])
        frame_means = mat['data'].item()[framef_ind].copy().T

    # Kurtosis < thresh indicates a flat, low-activity ROI -- likely noise.
    useinds = threshold_kurtosis(dFF, thresh=1.5)
    dFF = dFF[useinds, :]

    if not merge_duplicates:
        return get_single_independent_axons(dFF, cc_thresh, gcc_thresh, apply_dFF_filter, fps=fps, frame_means=frame_means)

    elif merge_duplicates:
        return get_grouped_independent_axons(dFF, cc_thresh, gcc_thresh, apply_dFF_filter, fps=fps)


def threshold_kurtosis(dFF, thresh=2.):
    """ Return indices of ROIs with kurtosis above thresh.

    Parameters
    ----------
    dFF : np.ndarray, shape (N_rois, N_frames)
        dF/F traces.
    thresh : float
        Minimum kurtosis to keep an ROI.

    Returns
    -------
    use_ROIs : np.ndarray of int
        Indices of ROIs that passed the threshold.
    """

    kurtosis_ = compute_kurtosis(dFF)
    use_ROIs = np.where(kurtosis_.ravel() > thresh)[0]

    return use_ROIs
