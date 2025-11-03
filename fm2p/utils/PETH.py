import numpy as np
import matplotlib.pyplot as plt

import fm2p


def norm_psth(mean_psth):
    psth_norm = np.zeros_like(mean_psth)*np.nan
    for c in range(np.size(mean_psth,0)):
        x = mean_psth[c].copy()
        psth_norm[c,:] = (x - np.nanmean(x[:10])) / np.nanmax(x)
    return psth_norm


def calc_hist_PETH(spikes, event_frames, window_bins):
    spikes = np.asarray(spikes)
    event_frames = np.asarray(event_frames)
    window_bins = np.asarray(window_bins)

    n_cells, n_frames = spikes.shape
    n_events = len(event_frames)
    n_bins = len(window_bins)

    psth = np.zeros((n_cells, n_events, n_bins))

    for i, event in enumerate(event_frames):
        # Calculate absolute indices for this event
        indices = event + window_bins
        # Clip indices to stay within valid range
        valid_mask = (indices >= 0) & (indices < n_frames)
        valid_indices = (indices[valid_mask]).astype(int)
        if len(valid_indices) > 0:
            psth[:, i, valid_mask] = spikes[:, valid_indices]

    mean_psth = psth.mean(axis=1)  # average across events
    stderr = np.std(psth, axis=1) / np.sqrt(np.size(psth, axis=1))

    norm_psths = np.zeros_like(psth) * np.nan
    for i in range(np.size(psth, 1)):
        norm_psths[:,i] = norm_psth(psth[:,i])

    mean_psth_norm = norm_psths.mean(axis=1)
    stderr_norm = np.std(norm_psths, axis=1) / np.sqrt(np.size(norm_psths, axis=1))

    return mean_psth, stderr, mean_psth_norm, stderr_norm


def norm_psth_paired(mean_psth1, mean_psth2):
    psth1_norm = np.zeros_like(mean_psth1)*np.nan
    psth2_norm = np.zeros_like(mean_psth2)*np.nan
    for c in range(np.size(mean_psth1,0)):
        x1 = mean_psth1[c].copy()
        x2 = mean_psth2[c].copy()
        max_val = np.nanmax([np.nanmax(x1), np.nanmax(x2)])
        psth1_norm[c,:] = (x1 - np.nanmean(x1[:10])) / max_val
        psth2_norm[c,:] = (x2 - np.nanmean(x2[:10])) / max_val
    return psth1_norm, psth2_norm


def find_trajectory_initiation(signal, time, peak_times, smoothing_window=2):

    signal = np.asarray(signal)
    time = np.asarray(time)
    peak_times = np.asarray(peak_times)

    # simple smoothing to suppress jitter (moving average)
    if smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        smoothed = np.convolve(signal, kernel, mode='same')
    else:
        smoothed = signal

    onsets = []
    for pt in peak_times:
        # index of the peak
        peak_idx = np.nanargmin(np.abs(time - pt))
        
        # walk backwards until the signal stops decreasing
        idx = peak_idx
        while idx > 0 and smoothed[idx-1] <= smoothed[idx]:
            idx -= 1
        trough_idx = idx
        
        # the onset is the trough before the rise
        onsets.append(time[idx])

        # look between trough and peak for point closest to zero (i.e., when the
        # direction reverses and the velocity sign changes, which should be the
        # onset of the new direciotn of movement).
        segment = signal[trough_idx:peak_idx+1]
        rel_idx = np.argmin(np.abs(segment))
        onset_idx = trough_idx + rel_idx

        onsets.append(time[onset_idx])
    
    return np.array(onsets)


def get_event_onsets(event_times, sample_rate=7.49, min_frames=4):

    event_times = np.sort(np.asarray(event_times))  # ensure sorted
    min_gap = min_frames / sample_rate  # minimum time between events
    
    onsets = [event_times[0]]  # always keep first event
    for t in event_times[1:]:
        if t - onsets[-1] >= min_gap:
            onsets.append(t)
    
    return np.array(onsets)


def get_event_offsets(event_times, sample_rate=7.49, min_frames=4):

    event_times = np.sort(np.asarray(event_times))  # ensure sorted
    min_gap = min_frames / sample_rate  # minimum time between events
    
    onsets = [event_times[0]]
    for t in event_times[1:]:
        if t - onsets[-1] < min_gap:
            # replace previous with the later one (keep last in cluster)
            onsets[-1] = t
        else:
            # start a new cluster
            onsets.append(t)
    
    return np.array(onsets)


def drop_nearby_events(thin, avoid, win=0.25):
    to_drop = np.array([c for c in thin for g in avoid if ((g>(c-win)) & (g<(c+win)))])
    thinned = np.delete(thin, np.isin(thin, to_drop))
    return thinned


def drop_repeat_events(eventT, onset=True, win=0.020):
    duplicates = set([])
    for t in eventT:
        if onset:
            # keep first
            new = eventT[((eventT-t)<win) & ((eventT-t)>0)]
        else:
            # keep last
            new = eventT[((t-eventT)<win) & ((t-eventT)>0)]
        duplicates.update(list(new))
    thinned = np.sort(np.setdiff1d(eventT, np.array(list(duplicates)), assume_unique=True))
    return thinned


def balanced_index_resample(signal, bin_edges, random_state=None):
    rng = np.random.default_rng(random_state)

    # Digitize signal values into bins
    bin_ids = np.digitize(signal, bin_edges) - 1  # bin index for each sample
    unique_bins = np.arange(len(bin_edges) - 1)

    # Collect indices per bin
    bin_to_indices = {b: np.where(bin_ids == b)[0] for b in unique_bins}

    # Find minimum bin population (ignore empty bins)
    bin_counts = {b: len(idxs) for b, idxs in bin_to_indices.items()}
    nonempty_bins = {b: c for b, c in bin_counts.items() if c > 0}
    if not nonempty_bins:
        return np.array([], dtype=int)  # nothing to return
    min_count = min(nonempty_bins.values())

    # Randomly select min_count indices from each non-empty bin
    selected_indices = []
    for b, idxs in bin_to_indices.items():
        if len(idxs) > 0:
            chosen = rng.choice(idxs, size=min_count, replace=False)
            selected_indices.append(chosen)

    # Concatenate and shuffle final indices
    final_indices = np.concatenate(selected_indices)
    rng.shuffle(final_indices)

    return final_indices


def calc_PETH_mod_ind(psth):
    baseline = np.nanmean(psth[:8])
    modind = (np.nanmax(psth) - baseline) / (np.nanmax(psth) + baseline)
    return modind


def drop_redundant_saccades(mov, to_avoid=None, near_win=0.25, repeat_win=0.20, onset=True):

    # drop nearby events
    if to_avoid is not None:
        to_drop = np.array([c for c in mov for g in to_avoid if ((g>(c-near_win)) & (g<(c+near_win)))])
        eventT = np.delete(mov, np.isin(mov, to_drop))
    else:
        eventT = mov

    # drop repeat events
    duplicates = set([])
    for t in eventT:
        if onset:
            # keep first
            new = eventT[((eventT-t)<repeat_win) & ((eventT-t)>0)]
        else:
            # keep last
            new = eventT[((t-eventT)<repeat_win) & ((t-eventT)>0)]
        duplicates.update(list(new))
    thinned = np.sort(np.setdiff1d(eventT, np.array(list(duplicates)), assume_unique=True))
    return thinned


def calc_eye_head_movement_times(data):

    eyeT = data['eyeT'][data['eyeT_startInd']:data['eyeT_endInd']]
    eyeT = eyeT - eyeT[0]
    t = eyeT.copy()[:-1]
    t1 = t + (np.diff(eyeT) / 2)
    imuT = data['imuT_trim']
    dHead = - fm2p.interpT(data['gyro_z_trim'], imuT, t1)
    theta_full = np.rad2deg(data['theta'][data['eyeT_startInd']:data['eyeT_endInd']])
    dEye  = np.diff(fm2p.interp_short_gaps(theta_full, 5)) / np.diff(eyeT)
    dEye = np.roll(dEye, -2) # static offset correction

    dGaze = dHead + dEye

    shifted_head = 60
    still_gaze = 120
    shifted_gaze = 240

    # gaze-shifting saccades
    gaze_left = eyeT[(
        (dHead > shifted_head) &
        (dGaze > shifted_gaze)
    )]
    gaze_right = eyeT[(
        (dHead < -shifted_gaze) &
        (dGaze < -shifted_gaze)
    )]

    # compensatory eye/head movements
    comp_left = eyeT[(
        (dHead > shifted_head) &
        (dGaze < still_gaze)   &
        (dGaze > -still_gaze)
    )]
    comp_right = eyeT[(
        (dHead < -shifted_head) &
        (dGaze < still_gaze)    &
        (dGaze > -still_gaze)
    )]

    gaze_left = drop_redundant_saccades(gaze_left)
    gaze_right = drop_redundant_saccades(gaze_right)

    # with two arguments, it also removes nearby events. otherwise, just the repeated events
    comp_left = drop_redundant_saccades(comp_left, comp_right)
    comp_right = drop_redundant_saccades(comp_right, comp_left)

    saccade_dict = {
        'gaze_left': gaze_left,
        'gaze_right': gaze_right,
        'comp_left': comp_left,
        'comp_right': comp_right,
        'dEye': dEye,
        'dHead': dHead,
        'dGaze': dGaze
    }

    return saccade_dict


def calc_PETHs(data):

    saccade_dict = calc_eye_head_movement_times(data)
    sps = data['norm_spikes']
    dFF = data['raw_dFF']

    win_frames = np.arange(-15,16)
    win_times = win_frames*(1/7.52)

    gaze_right_sps, gaze_right_sps_stderr, gaze_right_norm_sps, gaze_right_norm_stderr_sps = calc_hist_PETH(
        sps, saccade_dict['gaze_right'], win_frames
    )
    gaze_left_sps, gaze_left_sps_stderr, gaze_left_norm_sps, gaze_left_norm_stderr_sps = calc_hist_PETH(
        sps, saccade_dict['gaze_left'], win_frames
    )
    comp_right_sps, comp_right_sps_stderr, comp_right_norm_sps, comp_right_norm_stderr_sps = calc_hist_PETH(
        sps, saccade_dict['comp_right'], win_frames
    )
    comp_left_sps, comp_left_sps_stderr, comp_left_norm_sps, comp_left_norm_stderr_sps = calc_hist_PETH(
        sps, saccade_dict['comp_right'], win_frames
    )

    gaze_right_dFF, gaze_right_dFF_stderr, gaze_right_norm_dFF, gaze_right_norm_stderr_dFF = calc_hist_PETH(
        dFF, saccade_dict['gaze_right'], win_frames
    )
    gaze_left_dFF, gaze_left_dFF_stderr, gaze_left_norm_dFF, gaze_left_norm_stderr_dFF = calc_hist_PETH(
        dFF, saccade_dict['gaze_left'], win_frames
    )
    comp_right_dFF, comp_right_dFF_stderr, comp_right_norm_dFF, comp_right_norm_stderr_dFF = calc_hist_PETH(
        dFF, saccade_dict['comp_right'], win_frames
    )
    comp_left_dFF, comp_left_dFF_stderr, comp_left_norm_dFF, comp_left_norm_stderr_dFF = calc_hist_PETH(
        dFF, saccade_dict['comp_right'], win_frames
    )

    norm_comp__sps = norm_psth(right_PETHs_sps)
    normL_sps = norm_psth(left_PETHs_sps)
    normU_sps = norm_psth(up_PETHs_sps)
    normD_sps = norm_psth(down_PETHs_sps)

    normR_dFF = norm_psth(right_PETHs_dFF)
    normL_dFF = norm_psth(left_PETHs_dFF)
    normU_dFF = norm_psth(up_PETHs_dFF)
    normD_dFF = norm_psth(down_PETHs_dFF)
 
    peth_dict = {
        'win_frames': win_frames,
        'win_times': win_times,
        'right_PETHs_dFF': right_PETHs_dFF,
        'left_PETHs_dFF': left_PETHs_dFF,
        'up_PETHs_dFF': up_PETHs_dFF,
        'down_PETHs_dFF': down_PETHs_dFF,
        'norm_right_PETHs_dFF': normR_dFF,
        'norm_left_PETHs_dFF': normL_dFF,
        'norm_up_PETHs_dFF': normU_dFF,
        'norm_down_PETHs_dFF': normD_dFF,
        'right_PETHs_sps': right_PETHs_sps,
        'left_PETHs_sps': left_PETHs_sps,
        'up_PETHs_sps': up_PETHs_sps,
        'down_PETHs_sps': down_PETHs_sps,
        'norm_right_PETHs_sps': normR_sps,
        'norm_left_PETHs_sps': normL_sps,
        'norm_up_PETHs_sps': normU_sps,
        'norm_down_PETHs_sps': normD_sps
    }

    return peth_dict

