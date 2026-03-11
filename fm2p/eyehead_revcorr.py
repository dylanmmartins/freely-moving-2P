# -*- coding: utf-8 -*-

import os
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

import warnings
warnings.filterwarnings('ignore')

import fm2p


# ---------------------------------------------------------------------------
# Module-level state shared with multiprocessing workers via fork.
# Set by eyehead_revcorr() before Pool creation; workers inherit via COW.
# ---------------------------------------------------------------------------
_g_spikes = None
_g_ltdk   = None
_g_twopT  = None


def valid_mask(items):
    mask = np.ones_like(items[0]).astype(bool)
    for x in items:
        mask &= ~np.isnan(x)
    return mask.astype(bool)


def _find_consecutive_extremes(tc, n=3):
    """Return start indices of the lowest and highest n-consecutive-bin windows.

    Parameters
    ----------
    tc : (n_bins,) array -- tuning curve (may contain NaN)
    n  : int             -- window size

    Returns
    -------
    low_start, high_start : int or None
        Start index of the lowest / highest window, or None if no valid window
        of length n exists.
    """
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


def calc_reliability_over(spikes, behavior, n_micro=20, n_bins=13, bound=10,
                          cv_thresh=0.1, n_repeats=10, rng_seed=None):
    """Cross-validated modulation index using a 4-section scheme.

    Splits the recording into ``n_micro`` micro-chunks (default 20), randomly
    assigns them into 4 equal sections A, B, C, D.  Two cross-validation
    rounds per repeat:

      Round 1 — peak bin identified from section A, trough bin from section B;
                MI = (rate_peak - rate_trough) / (rate_peak + rate_trough)
                evaluated on the held-out sections C + D.
      Round 2 — peak from C, trough from D; MI evaluated on sections A + B.

    The CV-MI per repeat is the mean of both rounds, clipped to [0, 1].
    The final CV-MI is the mean over ``n_repeats`` independent random splits.

    Peak/trough positions are found on a lightly smoothed per-section tuning
    curve (sigma = 1 bin) using ``_find_consecutive_extremes`` (3 consecutive
    bins).  The MI itself is evaluated on the raw (unsmoothed) held-out curve.

    A noisy cell whose apparent peak in section A is noise-driven will not
    reproduce in C+D, yielding a low CV-MI despite large within-section
    modulation.

    Parameters
    ----------
    spikes    : (n_cells, n_frames)
    behavior  : (n_frames,)
    n_micro   : int   -- micro-chunk count, must be divisible by 4 (default 20)
    n_bins    : int   -- number of tuning bins
    bound     : float -- percentile used to clip the bin range
    cv_thresh : float -- CV-MI threshold for "reliable" classification
    n_repeats : int   -- independent random splits to average (default 10)
    rng_seed  : int or None

    Returns
    -------
    cv_mi         : (n_cells,) float -- cross-validated modulation index
    reliable_inds : (n_cells,) bool  -- True where cv_mi > cv_thresh
    """
    N_EXTREME = 3
    n_cells   = spikes.shape[0]
    n_frames  = spikes.shape[1]
    sec_sz    = n_micro // 4          # micro-chunks per section

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

        # Build 4 sections from the shuffled micro-chunks
        sections = []
        for s in range(4):
            idx = []
            for c in chunk_order[s * sec_sz : (s + 1) * sec_sz]:
                idx.extend(all_inds[chunk_size * c : chunk_size * (c + 1)])
            sections.append(np.sort(idx).astype(int))

        # Per-section tuning curves — used only for peak/trough location
        # No additional smoothing: each section already averages 1/4 of frames,
        # and _find_consecutive_extremes uses a 3-bin window for robustness.
        tcs_smooth = []
        for s in range(4):
            _, tc, _ = fm2p.tuning_curve(
                spikes[:, sections[s]], beh[sections[s]], bins
            )
            tcs_smooth.append(tc)

        # Held-out evaluation tuning curves (unsmoothed)
        pair1_inds = np.sort(np.concatenate([sections[0], sections[1]]))
        pair2_inds = np.sort(np.concatenate([sections[2], sections[3]]))
        _, tc_eval_pair2, _ = fm2p.tuning_curve(
            spikes[:, pair2_inds], beh[pair2_inds], bins
        )
        _, tc_eval_pair1, _ = fm2p.tuning_curve(
            spikes[:, pair1_inds], beh[pair1_inds], bins
        )

        mi1 = np.zeros(n_cells)
        mi2 = np.zeros(n_cells)

        for c in range(n_cells):
            # Round 1: peak bin from section A, trough bin from section B
            #          evaluate in sections C+D
            _, high_A = _find_consecutive_extremes(tcs_smooth[0][c], n=N_EXTREME)
            low_B, _  = _find_consecutive_extremes(tcs_smooth[1][c], n=N_EXTREME)
            if high_A is not None and low_B is not None:
                hi    = np.nanmean(tc_eval_pair2[c, high_A : high_A + N_EXTREME])
                lo    = np.nanmean(tc_eval_pair2[c, low_B  : low_B  + N_EXTREME])
                denom = hi + lo
                if denom > 0 and np.isfinite(denom):
                    mi1[c] = max((hi - lo) / denom, 0.0)

            # Round 2: peak bin from section C, trough bin from section D
            #          evaluate in sections A+B
            _, high_C = _find_consecutive_extremes(tcs_smooth[2][c], n=N_EXTREME)
            low_D, _  = _find_consecutive_extremes(tcs_smooth[3][c], n=N_EXTREME)
            if high_C is not None and low_D is not None:
                hi    = np.nanmean(tc_eval_pair1[c, high_C : high_C + N_EXTREME])
                lo    = np.nanmean(tc_eval_pair1[c, low_D  : low_D  + N_EXTREME])
                denom = hi + lo
                if denom > 0 and np.isfinite(denom):
                    mi2[c] = max((hi - lo) / denom, 0.0)

        cv_mi_all[rep] = (mi1 + mi2) / 2.0

    cv_mi = np.nanmean(cv_mi_all, axis=0)
    reliable_inds = cv_mi > cv_thresh
    return cv_mi, reliable_inds
    

def calc_1d_tuning(spikes, var, ltdk, bound=10, n_bins=13):
    # spikes should be 2D and have shape (cells, time)
    # var should be 1d
    # ltdk is the light/dark state vector, bool, 1==lights on

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

        bin_edges, tuning_out[:,:,state], err_out[:,:,state] = fm2p.tuning_curve(usespikes, usevar, bins)

    # for output's last dimension, 0 is dark, 1 is light
    return bin_edges, tuning_out, err_out


def tuning2d(x_vals, y_vals, rates, n_x=13, n_y=13):

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

    try:
        b, t, e = calc_1d_tuning(spikes, behavior_v, ltdk)
    except IndexError:
        if len(behavior_v) != len(np.arange(len(behavior_v) * (1 / 7.5), 1 / 7.5)):
            try:
                behavior_v = fm2p.interpT(
                    fm2p.nan_interp(behavior_v),
                    np.arange(0, len(behavior_v) * (1 / 7.5), 1 / 7.5)[:-1],
                    twopT,
                )
            except Exception:
                behavior_v = fm2p.interpT(
                    fm2p.nan_interp(behavior_v),
                    np.arange(0, len(behavior_v) * (1 / 7.5), 1 / 7.5),
                    twopT,
                )
        else:
            behavior_v = fm2p.interpT(
                fm2p.nan_interp(behavior_v),
                np.arange(0, len(behavior_v) * (1 / 7.5), 1 / 7.5),
                twopT,
            )
        b, t, e = calc_1d_tuning(spikes, behavior_v, ltdk)

    mod_l, ismod_l = fm2p.calc_multicell_modulation(t[:, :, 1])
    mod_d, ismod_d = fm2p.calc_multicell_modulation(t[:, :, 0])

    relL, isrelL = calc_reliability_over(spikes[:, ltdk],  behavior_v[ltdk])
    relD, isrelD = calc_reliability_over(spikes[:, ~ltdk], behavior_v[~ltdk])

    return (behavior_k, b, t, e,
            mod_l, ismod_l, mod_d, ismod_d,
            relL, isrelL, relD, isrelD)


def eyehead_revcorr(preproc_path=None):

    if preproc_path is None:
        preproc_path = fm2p.select_file(
            'Select preprocessing HDF file.',
            filetypes=[('HDF','.h5'),]
        )
        data = fm2p.read_h5(preproc_path)
    elif type(preproc_path) == str:
        data = fm2p.read_h5(preproc_path)
    elif type(preproc_path) == dict:
        data = preproc_path

    print('  -> Loading data.')

    eyeT = data['eyeT'][data['eyeT_startInd']:data['eyeT_endInd']]
    eyeT = eyeT - eyeT[0]

    if 'dPhi' not in data.keys():
        phi_full = np.rad2deg(data['phi'][data['eyeT_startInd']:data['eyeT_endInd']])
        dPhi  = np.diff(fm2p.interp_short_gaps(phi_full, 5)) / np.diff(eyeT)
        dPhi = np.roll(dPhi, -2)
        data['dPhi'] = dPhi

    if 'dTheta' not in data.keys():

        if 'dEye' not in data.keys():
            t = eyeT.copy()[:-1]
            t1 = t + (np.diff(eyeT) / 2)
            theta_full = np.rad2deg(data['theta'][data['eyeT_startInd']:data['eyeT_endInd']])
            dEye  = np.diff(fm2p.interp_short_gaps(theta_full, 5)) / np.diff(eyeT)
            data['dTheta'] = np.roll(dEye, -2) # static offset correction
            data['eyeT1'] = t1

        else:
            data['dTheta'] = data['dEye'].copy()

    # interpolate dEye values to twop data
    dTheta = fm2p.interp_short_gaps(data['dTheta'])
    dTheta = fm2p.interpT(dTheta, data['eyeT1'], data['twopT'])
    dPhi = fm2p.interp_short_gaps(data['dPhi'])
    dPhi = fm2p.interpT(dPhi, data['eyeT1'], data['twopT'])

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
    #     gaze  = fm2p.interpT(np.cumsum(_dGaze_raw), gazeT, data['twopT'])
    #     dGaze = fm2p.interpT(_dGaze_raw, gazeT, data['twopT'])
    # else:
    #     gaze  = None
    #     dGaze = None

    # _gaze_entries = {'gaze': gaze, 'dGaze': dGaze} if gaze is not None else {}

    if 'gyro_x_twop_interp' in data.keys():
        behavior_vars = {
            # head positions
            'yaw': data['head_yaw_deg'].copy(),
            'pitch': data['pitch_twop_interp'].copy(),
            'roll': data['roll_twop_interp'].copy(),
            # eye positions
            'theta': data['theta_interp'].copy() - np.nanmean(data['theta_interp']),
            'phi': data['phi_interp'].copy() - np.nanmean(data['phi_interp']),
            # eye speeds
            'dTheta': dTheta,
            'dPhi': dPhi,
            # head angular rotation speeds
            'gyro_x': data['gyro_x_twop_interp'].copy(),
            'gyro_y': data['gyro_y_twop_interp'].copy(),
            'gyro_z': data['gyro_z_twop_interp'].copy(),
            # head accelerations
            'acc_x': data['acc_x_twop_interp'].copy(),
            'acc_y': data['acc_y_twop_interp'].copy(),
            'acc_z': data['acc_z_twop_interp'].copy(),
            # **_gaze_entries,
        }
    else:
        behavior_vars = {
            # eye positions
            'theta': data['theta_interp'].copy() - np.nanmean(data['theta_interp']),
            'phi': data['phi_interp'].copy() - np.nanmean(data['phi_interp']),
            # eye speeds
            'dTheta': dTheta,
            'dPhi': dPhi,
            # **_gaze_entries,
        }

    dict_out = {}

    # Populate module-level globals before forking so workers inherit them.
    global _g_spikes, _g_ltdk, _g_twopT
    _g_spikes = spikes
    _g_ltdk   = ltdk
    _g_twopT  = data['twopT']

    print('  -> Measuring 1D tuning to all eye/head variables.')
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
    print('  -> Writing {}'.format(savename))
    fm2p.write_h5(savename, dict_out)


if __name__ == '__main__':


    all_fm_preproc_files = fm2p.find('*DMM*fm*preproc.h5', '/home/dylan/Storage/freely_moving_data/_V1PPC')
    
    for f in tqdm(all_fm_preproc_files):
        fm2p.eyehead_revcorr(f)


    # filepath = fm2p.select_file(
    #     'Select preprocessed HDF file.',
    #     [('H5','.h5'),]
    # )

    # eyehead_revcorr(filepath)