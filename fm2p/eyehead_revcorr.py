# -*- coding: utf-8 -*-
"""
fm2p/eyehead_revcorr.py

1D tuning curves and cross-validated reliability for eye and head variables.

Functions
---------
valid_mask
    Mask of samples that are non-NaN in every array.
_find_consecutive_extremes
    Start indices of lowest and highest n-bin windows of a tuning curve.
calc_reliability_over
    Cross-validated modulation over repeated split-halves of shuffled micro-chunks.
calc_1d_tuning
    Tuning curves for one variable, split by light and dark.
tuning2d
    2D tuning heatmap over the 10-90th percentile range of two variables.
_compute_var_results
    Process one behavioral variable in a worker process.
eyehead_revcorr
    1D tuning and reliability to all eye and head variables, light and dark.
eyehead_revcorr_eye_only
    Eye-only variant of eyehead_revcorr for recordings without IMU.


DMM, September 2026
"""


if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

import os
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

import warnings
warnings.filterwarnings('ignore')

from .utils.gui_funcs import select_file
from .utils.files import read_h5, write_h5
from .utils.helper import interp_short_gaps, nan_interp
from .utils.time import interpT
from .utils.tuning import tuning_curve, calc_multicell_modulation
from .utils.paths import find


_g_spikes = None
_g_ltdk   = None
_g_twopT  = None


def valid_mask(items):
    """ Mask of samples that are non-NaN in every array.

    Parameters
    ----------
    items : list of np.ndarray
        Arrays of equal shape.

    Returns
    -------
    mask : np.ndarray of bool
    """

    mask = np.ones_like(items[0]).astype(bool)
    for x in items:
        mask &= ~np.isnan(x)
    return mask.astype(bool)


def _find_consecutive_extremes(tc, n=3):
    """ Start indices of lowest and highest n-bin windows of a tuning curve. """

    n_bins = len(tc)
    low_val  =  np.inf
    high_val = -np.inf
    low_start  = None
    high_start = None

    for i in range(n_bins - n + 1):
        window = tc[i:i + n]
        if not np.all(np.isfinite(window)):
            continue
        wmean = np.mean(window)
        if wmean < low_val:
            low_val   = wmean
            low_start = i
        if wmean > high_val:
            high_val   = wmean
            high_start = i

    return low_start, high_start


def calc_reliability_over(spikes, behavior, n_micro=20, n_bins=13, bound=2,
                          cv_thresh=0.1, n_repeats=50, rng_seed=None):
    """ Cross-validated modulation over repeated split-halves of shuffled micro-chunks.

    Parameters
    ----------
    spikes : np.ndarray, shape (n_cells, n_frames)
        Spike rates.
    behavior : np.ndarray, shape (n_frames,)
        Behavioral variable.
    n_micro : int
        Number of micro-chunks; split evenly into two halves.
    n_bins : int
        Tuning curve bins.
    bound : float
        Percentile trimmed off each end when setting bin range.
    cv_thresh : float
        Minimum CV modulation to count as reliable.
    n_repeats : int
        Random split repeats averaged.
    rng_seed : int or None
        Seed for chunk shuffling.

    Returns
    -------
    cv_mi : np.ndarray, shape (n_cells,)
        Mean cross-validated modulation.
    reliable_inds : np.ndarray of bool, shape (n_cells,)
    """

    N_EXTREME = 3
    n_cells   = spikes.shape[0]
    n_frames  = spikes.shape[1]
    sec_sz    = n_micro // 2          # micro-chunks per half

    if behavior is None or n_frames < 2 * n_micro or sec_sz < 1:
        return np.zeros(n_cells), np.zeros(n_cells, dtype=bool)

    beh = np.asarray(behavior, dtype=float)
    if np.sum(np.isfinite(beh)) < n_bins + 1:
        return np.zeros(n_cells), np.zeros(n_cells, dtype=bool)

    bins = np.linspace(
        np.nanpercentile(beh, bound),
        np.nanpercentile(beh, 100 - bound),
        n_bins + 1,
    )

    rng        = np.random.default_rng(rng_seed)
    chunk_size = n_frames // n_micro
    all_inds   = np.arange(n_frames)

    cv_mi_all = np.zeros((n_repeats, n_cells))

    for rep in range(n_repeats):
        chunk_order = np.arange(n_micro)
        rng.shuffle(chunk_order)

        # Build 2 halves from the shuffled micro-chunks
        halves = []
        for s in range(2):
            idx = []
            for c in chunk_order[s * sec_sz : (s + 1) * sec_sz]:
                idx.extend(all_inds[chunk_size * c : chunk_size * (c + 1)])
            halves.append(np.sort(idx).astype(int))

        # Train-half tuning curves -- used only for peak/trough location
        _, tc_train, _ = tuning_curve(
            spikes[:, halves[0]], beh[halves[0]], bins
        )

        # Held-out half tuning curve (unsmoothed)
        _, tc_eval, _ = tuning_curve(
            spikes[:, halves[1]], beh[halves[1]], bins
        )

        mi = np.zeros(n_cells)

        for c in range(n_cells):
            # Peak and trough identified from train half, evaluated on held-out half
            _, high_A = _find_consecutive_extremes(tc_train[c], n=N_EXTREME)
            low_A, _  = _find_consecutive_extremes(tc_train[c], n=N_EXTREME)
            if high_A is not None and low_A is not None:
                hi    = np.nanmean(tc_eval[c, high_A : high_A + N_EXTREME])
                lo    = np.nanmean(tc_eval[c, low_A  : low_A  + N_EXTREME])
                denom = hi + lo
                if denom > 0 and np.isfinite(denom):
                    mi[c] = max((hi - lo) / denom, 0.0)

        cv_mi_all[rep] = mi

    cv_mi = np.nanmean(cv_mi_all, axis=0)
    reliable_inds = cv_mi > cv_thresh
    return cv_mi, reliable_inds


def calc_1d_tuning(spikes, var, ltdk, bound=2, n_bins=13, fixed_bins=None):
    """ Tuning curves for one variable, split by light and dark.

    Parameters
    ----------
    spikes : np.ndarray, shape (n_cells, n_frames)
        Spike rates.
    var : np.ndarray, shape (n_frames,)
        Behavioral variable; NaN frames dropped.
    ltdk : np.ndarray of bool, shape (n_frames,)
        True when lights are on.
    bound : float
        Percentile trimmed off each end when setting bin range.
    n_bins : int
        Number of bin edges.
    fixed_bins : np.ndarray or None
        Explicit bin edges, overrides bound and n_bins.

    Returns
    -------
    bin_edges : np.ndarray
    tuning_out : np.ndarray, shape (n_cells, n_bins-1, 2)
        Last axis is [dark, light].
    err_out : np.ndarray
        SEM, same shape as tuning_out.
    """

    # spikes should be 2D and have shape (cells, time)
    # var should be 1d
    # ltdk is the light/dark state vector, bool, 1==lights on

    # Skip over any NaNs so that all bins will be valid.
    mask = ~np.isnan(var)
    var = var[mask]
    spikes = spikes[:, mask]
    ltdk = ltdk[mask]

    if fixed_bins is not None:
        bins = fixed_bins
    else:
        bins = np.linspace(
            np.nanpercentile(var, bound),
            np.nanpercentile(var, 100-bound),
            n_bins
        )
    n_cells = np.size(spikes, 0)

    tuning_out = np.zeros([
        n_cells,
        len(bins)-1,
        2
    ]) * np.nan
    err_out = np.zeros_like(tuning_out) * np.nan

    for state in range(2):
        # state 0 is dark, state 1 is light
        if state == 0:
            usesamp = ~ltdk
        elif state==1:
            usesamp = ltdk

        # usespikes = fm2p.zscore_spikes(spikes[:,usesamp]) # stopped z-scoring spikes... jan 21 2026
        usespikes = spikes[:, usesamp]
        usevar = var[usesamp]

        bin_edges, tuning_out[:,:,state], err_out[:,:,state] = tuning_curve(usespikes, usevar, bins)

    # for output's last dimension, 0 is dark, 1 is light
    return bin_edges, tuning_out, err_out


def tuning2d(x_vals, y_vals, rates, n_x=13, n_y=13):
    """ 2D tuning heatmap over the 10-90th percentile range of two variables.

    Parameters
    ----------
    x_vals, y_vals : np.ndarray, shape (n_frames,)
        Variables on each axis.
    rates : np.ndarray, shape (n_cells, n_frames)
        Spike rates.
    n_x, n_y : int
        Bins per axis.

    Returns
    -------
    x_bins, y_bins : np.ndarray
        Bin edges.
    heatmap : np.ndarray, shape (n_cells, n_y, n_x)
    """

    x_bins = np.linspace(np.nanpercentile(x_vals, 10), np.nanpercentile(x_vals, 90), num=n_x+1)
    y_bins = np.linspace(np.nanpercentile(y_vals, 10), np.nanpercentile(y_vals, 90), num=n_y+1)

    n_cells = np.size(rates, 0)

    heatmap = np.zeros((n_cells, n_y, n_x)) * np.nan

    for c in range(n_cells):
        for i in range(n_x):
            for j in range(n_y):
                in_bin = (x_vals >= x_bins[i]) & (x_vals < x_bins[i+1]) & \
                        (y_vals >= y_bins[j]) & (y_vals < y_bins[j+1])
                heatmap[c, j, i] = np.nanmean(rates[c, in_bin])

    return x_bins, y_bins, heatmap


def _compute_var_results(item):
    """Process one behavioral variable in a worker process.

    Reads spikes / ltdk / twopT from module-level globals populated before
    the Pool is created (copy-on-write via Linux fork).
    """

    behavior_k, behavior_v = item
    spikes = _g_spikes
    ltdk   = _g_ltdk
    twopT  = _g_twopT

    behavior_v = np.asarray(behavior_v, dtype=float)

    print(behavior_k)

    yaw_bins = np.linspace(0, 360, 14) if behavior_k == 'yaw' else None
    try:
        b, t, e = calc_1d_tuning(spikes, behavior_v, ltdk, fixed_bins=yaw_bins)
    except IndexError:
        if len(behavior_v) != len(np.arange(len(behavior_v) * (1 / 7.5), 1 / 7.5)):
            try:
                behavior_v = interpT(
                    nan_interp(behavior_v),
                    np.arange(0, len(behavior_v) * (1 / 7.5), 1 / 7.5)[:-1],
                    twopT,
                )
            except Exception:
                behavior_v = interpT(
                    nan_interp(behavior_v),
                    np.arange(0, len(behavior_v) * (1 / 7.5), 1 / 7.5),
                    twopT,
                )
        else:
            behavior_v = interpT(
                nan_interp(behavior_v),
                np.arange(0, len(behavior_v) * (1 / 7.5), 1 / 7.5),
                twopT,
            )
        b, t, e = calc_1d_tuning(spikes, behavior_v, ltdk, fixed_bins=yaw_bins)

    mod_l, ismod_l = calc_multicell_modulation(t[:, :, 1])
    mod_d, ismod_d = calc_multicell_modulation(t[:, :, 0])

    relL, isrelL = calc_reliability_over(spikes[:, ltdk],  behavior_v[ltdk])
    relD, isrelD = calc_reliability_over(spikes[:, ~ltdk], behavior_v[~ltdk])

    return (behavior_k, b, t, e,
            mod_l, ismod_l, mod_d, ismod_d,
            relL, isrelL, relD, isrelD)


def eyehead_revcorr(preproc_path=None):
    """ 1D tuning and reliability to all eye and head variables, light and dark.

    Writes eyehead_revcorrs_v06.h5 beside the preproc file.

    Parameters
    ----------
    preproc_path : str, dict or None
        Preproc .h5 path, already-loaded dict, or None to open a picker.
    """

    if preproc_path is None:
        preproc_path = select_file(
            'Select preprocessing HDF file.',
            filetypes=[('HDF','.h5'),]
        )
        data = read_h5(preproc_path)
    elif type(preproc_path) == str:
        data = read_h5(preproc_path)
    elif type(preproc_path) == dict:
        data = preproc_path

    print('  Loading data.')

    eyeT = data['eyeT'][data['eyeT_startInd']:data['eyeT_endInd']]
    eyeT = eyeT - eyeT[0]

    if 'dPhi' not in data.keys():
        phi_full = np.rad2deg(data['phi'][data['eyeT_startInd']:data['eyeT_endInd']])
        dPhi  = np.diff(interp_short_gaps(phi_full, 5)) / np.diff(eyeT)
        # Timing offset applied once in preprocessing (eye_imu_lag_samples).
        data['dPhi'] = dPhi

    if 'dTheta' not in data.keys():

        if 'dEye' not in data.keys():
            t = eyeT.copy()[:-1]
            t1 = t + (np.diff(eyeT) / 2)
            theta_full = np.rad2deg(data['theta'][data['eyeT_startInd']:data['eyeT_endInd']])
            dEye  = np.diff(interp_short_gaps(theta_full, 5)) / np.diff(eyeT)
            # Timing offset applied once in preprocessing (eye_imu_lag_samples).
            data['dTheta'] = dEye
            data['eyeT1'] = t1

        else:
            data['dTheta'] = data['dEye'].copy()

    # Interpolate dEye values to twop data.
    dTheta = interp_short_gaps(data['dTheta'])
    dTheta = interpT(dTheta, data['eyeT1'], data['twopT'])
    dPhi = interp_short_gaps(data['dPhi'])
    dPhi = interpT(dPhi, data['eyeT1'], data['twopT'])

    spikes = data['norm_spikes'].copy()

    ltdk = data['ltdk_state_vec'].copy()

    # Compute allocentric gaze direction: head yaw + pupil offset.
    # pupil_from_head = ang_offset - theta is stored in preproc.h5 by
    # calc_reference_frames and works for all recording types (somatic/axonal).
    # if 'pupil_from_head' in data.keys() and 'head_yaw_deg' in data.keys():
    #     gaze  = data['head_yaw_deg'].copy() + data['pupil_from_head'].copy()
    #     dGaze = np.gradient(gaze)  # deg/frame, same length as other vars
    # elif 'dGaze' in data.keys():
    #     # Legacy: integrate eye-camera-derived gaze velocity
    #     _dGaze_raw = data['dGaze'].copy()
    #     gazeT = data['eyeT_trim'] + (np.nanmedian(data['eyeT_trim']) / 2)
    #     gazeT = gazeT[:-1]
    #     gaze  = interpT(np.cumsum(_dGaze_raw), gazeT, data['twopT'])
    #     dGaze = interpT(_dGaze_raw, gazeT, data['twopT'])
    # else:
    #     gaze  = None
    #     dGaze = None

    # _gaze_entries = {'gaze': gaze, 'dGaze': dGaze} if gaze is not None else {}

    if 'gyro_x_twop_interp' in data.keys():
        behavior_vars = {
            # Head positions
            'yaw': data['head_yaw_deg'].copy(),
            'pitch': data['pitch_twop_interp'].copy(),
            'roll': data['roll_twop_interp'].copy(),
            # Eye positions
            'theta': data['theta_interp'].copy(),# - np.nanmean(data['theta_interp']),
            'phi': data['phi_interp'].copy(),# - np.nanmean(data['phi_interp']),
            # Eye speeds
            'dTheta': dTheta,
            'dPhi': dPhi,
            # Head angular rotation speeds.
            'gyro_x': data['gyro_x_twop_interp'].copy(),
            'gyro_y': data['gyro_y_twop_interp'].copy(),
            'gyro_z': data['gyro_z_twop_interp'].copy(),
            # Head accelerations
            'acc_x': data['acc_x_twop_interp'].copy(),
            'acc_y': data['acc_y_twop_interp'].copy(),
            'acc_z': data['acc_z_twop_interp'].copy(),
            # gaze_entries,
        }
    else:
        behavior_vars = {
            # Eye positions
            'theta': data['theta_interp'].copy(),# - np.nanmean(data['theta_interp']),
            'phi': data['phi_interp'].copy(),# - np.nanmean(data['phi_interp']),
            # Eye speeds
            'dTheta': dTheta,
            'dPhi': dPhi,
            # gaze_entries,
        }

    dict_out = {}

    # Populate module-level globals before forking so workers inherit them.
    global _g_spikes, _g_ltdk, _g_twopT
    _g_spikes = spikes
    _g_ltdk   = ltdk
    _g_twopT  = data['twopT']

    print('  Measuring 1D tuning to all eye/head variables.')
    n_workers = min(len(behavior_vars), mp.cpu_count())
    with mp.Pool(n_workers) as pool:
        results = pool.map(_compute_var_results, list(behavior_vars.items()))

    # shape of tuning and error will be (cells, bins, ltdk_state)
    for (behavior_k, b, t, e,
         mod_l, ismod_l, mod_d, ismod_d,
         relL, isrelL, relD, isrelD) in results:
        dict_out['{}_1dbins'.format(behavior_k)]   = b
        dict_out['{}_1dtuning'.format(behavior_k)] = t
        dict_out['{}_1derr'.format(behavior_k)]    = e
        dict_out['{}_l_mod'.format(behavior_k)]    = mod_l
        dict_out['{}_l_ismod'.format(behavior_k)]  = ismod_l
        dict_out['{}_d_mod'.format(behavior_k)]    = mod_d
        dict_out['{}_d_ismod'.format(behavior_k)]  = ismod_d
        dict_out['{}_l_rel'.format(behavior_k)]    = relL
        dict_out['{}_l_isrel'.format(behavior_k)]  = isrelL
        dict_out['{}_d_rel'.format(behavior_k)]    = relD
        dict_out['{}_d_isrel'.format(behavior_k)]  = isrelD

    basedir, _ = os.path.split(preproc_path)
    savename = os.path.join(basedir, 'eyehead_revcorrs_v06.h5')
    print('  Writing {}...'.format(savename))
    write_h5(savename, dict_out)


def eyehead_revcorr_eye_only(preproc_path=None):
    """ Eye-only variant of eyehead_revcorr for recordings without IMU.

    Writes eyehead_revcorrs_v07.h5 beside the preproc file.

    Parameters
    ----------
    preproc_path : str, dict or None
        Preproc .h5 path, already-loaded dict, or None to open a picker.
    """

    if preproc_path is None:
        preproc_path = select_file(
            'Select preprocessing HDF file.',
            filetypes=[('HDF','.h5'),]
        )
        data = read_h5(preproc_path)
    elif type(preproc_path) == str:
        data = read_h5(preproc_path)
    elif type(preproc_path) == dict:
        data = preproc_path

    print('  Loading data (eye-only mode).')

    eyeT = data['eyeT'][data['eyeT_startInd']:data['eyeT_endInd']]
    eyeT = eyeT - eyeT[0]

    if 'dPhi' not in data.keys():
        phi_full = np.rad2deg(data['phi'][data['eyeT_startInd']:data['eyeT_endInd']])
        dPhi  = np.diff(interp_short_gaps(phi_full, 5)) / np.diff(eyeT)
        # Timing offset applied once in preprocessing (eye_imu_lag_samples).
        data['dPhi'] = dPhi

    if 'dTheta' not in data.keys():

        if 'dEye' not in data.keys():
            t = eyeT.copy()[:-1]
            t1 = t + (np.diff(eyeT) / 2)
            theta_full = np.rad2deg(data['theta'][data['eyeT_startInd']:data['eyeT_endInd']])
            dEye  = np.diff(interp_short_gaps(theta_full, 5)) / np.diff(eyeT)
            data['dTheta'] = dEye
            data['eyeT1'] = t1

        else:
            data['dTheta'] = data['dEye'].copy()

    dTheta = interp_short_gaps(data['dTheta'])
    dTheta = interpT(dTheta, data['eyeT1'], data['twopT'])
    dPhi = interp_short_gaps(data['dPhi'])
    dPhi = interpT(dPhi, data['eyeT1'], data['twopT'])

    spikes = data['norm_spikes'].copy()
    n_frames = spikes.shape[1]

    # All frames are treated as light-on (no dark condition).
    ltdk = np.ones(n_frames, dtype=bool)

    behavior_vars = {
        'theta':  data['theta_interp'].copy(),# - np.nanmean(data['theta_interp']),
        'phi':    data['phi_interp'].copy(),# - np.nanmean(data['phi_interp']),
        'dTheta': dTheta,
        'dPhi':   dPhi,
    }

    dict_out = {}

    global _g_spikes, _g_ltdk, _g_twopT
    _g_spikes = spikes
    _g_ltdk   = ltdk
    _g_twopT  = data['twopT']

    print('  Measuring 1D tuning to eye variables.')
    n_workers = min(len(behavior_vars), mp.cpu_count())
    with mp.Pool(n_workers) as pool:
        results = pool.map(_compute_var_results, list(behavior_vars.items()))

    for (behavior_k, b, t, e,
         mod_l, ismod_l, mod_d, ismod_d,
         relL, isrelL, relD, isrelD) in results:
        dict_out['{}_1dbins'.format(behavior_k)]   = b
        dict_out['{}_1dtuning'.format(behavior_k)] = t
        dict_out['{}_1derr'.format(behavior_k)]    = e
        dict_out['{}_l_mod'.format(behavior_k)]    = mod_l
        dict_out['{}_l_ismod'.format(behavior_k)]  = ismod_l
        dict_out['{}_d_mod'.format(behavior_k)]    = mod_d
        dict_out['{}_d_ismod'.format(behavior_k)]  = ismod_d
        dict_out['{}_l_rel'.format(behavior_k)]    = relL
        dict_out['{}_l_isrel'.format(behavior_k)]  = isrelL
        dict_out['{}_d_rel'.format(behavior_k)]    = relD
        dict_out['{}_d_isrel'.format(behavior_k)]  = isrelD

    basedir, _ = os.path.split(preproc_path)
    savename = os.path.join(basedir, 'eyehead_revcorrs_v07.h5')
    print('  Writing {}...'.format(savename))
    write_h5(savename, dict_out)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', action='store_true', default=False)
    # Either directory or specific hdf file.
    parser.add_argument('--path', type=str, default=None)
    args = parser.parse_args()


    if args.batch:
        all_fm_preproc_files = find('*DMM*fm*preproc.h5', args.path)

        for f in tqdm(all_fm_preproc_files):
            _probe = read_h5(f)
            has_ltdk = 'ltdk_state_vec' in _probe
            if has_ltdk:
                eyehead_revcorr(f)
            else:
                has_imu = 'gyro_x_twop_interp' in _probe
                print('  No ltdk_state_vec found (has_imu={}). Using eye-only mode.'.format(has_imu))
                eyehead_revcorr_eye_only(f)


    elif not args.batch:
        eyehead_revcorr(args.path)
