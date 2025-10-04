import numpy as np
import matplotlib.pyplot as plt
import fm2p


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

    return psth, mean_psth


def norm_psth(mean_psth):
    psth_norm = np.zeros_like(mean_psth)*np.nan
    for c in range(np.size(mean_psth,0)):
        x = mean_psth[c].copy()
        psth_norm[c,:] = (x - np.nanmean(x[:10])) / np.nanmax(x)
    return psth_norm


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


def calc_PETHs(data):

    # theta_interp = data['theta']
    # phi_interp = data['phi']

    theta_full = np.rad2deg(data['theta'][data['eyeT_startInd']:data['eyeT_endInd']])
    phi_full = np.rad2deg(data['phi'][data['eyeT_startInd']:data['eyeT_endInd']])
    eyeT = data['eyeT'][data['eyeT_startInd']:data['eyeT_endInd']]
    eyeT = eyeT - eyeT[0]
    twopT = data['twopT']
    dt = 1/60
    dTheta = np.diff(theta_full) / dt
    dPhi = np.diff(phi_full) / dt
    
    rightward_onsets = get_event_onsets(eyeT[np.where(dTheta > 300)[0]], min_frames=4)
    rightward_onsets = find_trajectory_initiation(dTheta, eyeT[:-1], rightward_onsets)
    rightward_onsets = get_event_offsets(rightward_onsets, min_frames=4)
    right_theta_movement_inds = np.array([fm2p.find_closest_timestamp(twopT, t)[0] for t in rightward_onsets if not np.isnan(t)])
    # movR = np.zeros(len(theta_interp))
    # movR[right_theta_movement_inds] = 1
    # movR = np.concatenate([(np.diff(movR)>0), np.array([0])])

    leftward_onsets = get_event_onsets(eyeT[np.where(dTheta < -300)[0]], min_frames=4)
    leftward_onsets = find_trajectory_initiation(dTheta, eyeT[:-1], leftward_onsets)
    leftward_onsets = get_event_offsets(leftward_onsets, min_frames=4)
    left_theta_movement_inds = np.array([fm2p.find_closest_timestamp(twopT, t)[0] for t in leftward_onsets if not np.isnan(t)])
    # movL = np.zeros(len(theta_interp))
    # movL[left_theta_movement_inds] = 1
    # movL = np.concatenate([(np.diff(movL)>0), np.array([0])])

    downward_onsets = get_event_onsets(eyeT[np.where(dPhi < -300)[0]], min_frames=4)
    downward_onsets = find_trajectory_initiation(dPhi, eyeT[:-1], downward_onsets)
    downward_onsets = get_event_offsets(downward_onsets, min_frames=4)
    down_phi_movement_inds = np.array([fm2p.find_closest_timestamp(twopT, t)[0] for t in downward_onsets if not np.isnan(t)])

    upward_onsets = get_event_onsets(eyeT[np.where(dPhi > 300)[0]], min_frames=4)
    upward_onsets = find_trajectory_initiation(dPhi, eyeT[:-1], upward_onsets)
    upward_onsets = get_event_offsets(upward_onsets, min_frames=4)
    up_phi_movement_inds = np.array([fm2p.find_closest_timestamp(twopT, t)[0] for t in upward_onsets if not np.isnan(t)])

    win_frames = np.arange(-15,16)
    win_times = win_frames*(1/7.49)

    _, right_PETHs = calc_hist_PETH(data['norm_spikes'], rightward_onsets, win_frames)

    _, left_PETHs = calc_hist_PETH(data['norm_spikes'], leftward_onsets, win_frames)

    _, up_PETHs = calc_hist_PETH(data['norm_spikes'], upward_onsets, win_frames)

    _, down_PETHs = calc_hist_PETH(data['norm_spikes'], downward_onsets, win_frames)

    normR = norm_psth(right_PETHs)
    normL = norm_psth(left_PETHs)
    normU = norm_psth(up_PETHs)
    normD = norm_psth(down_PETHs)

    # fig, axs = plt.subplots(10,10,dpi=300,figsize=(10,10), sharex=True, sharey=True)
    # axs = axs.flatten()
    # for c in range(100):
    #     if c == 9:
    #         continue
    #     axs[c].plot(win_times, normR[c], 'tab:red')
    #     axs[c].plot(win_times, normL[c], 'tab:blue')
    #     axs[c].hlines(0, win_times[0], win_times[-1], ls='--', color='k', alpha=0.3)
    #     axs[c].hlines(0, -0.1, 0.1, ls='--', color='k', alpha=0.3)
    # fig.suptitle('rightwards saccadic PETH (spikes)')
    # fig.tight_layout()
    # fig.savefig('saccadic_PETH_spikes_right.png')

    # fig, axs = plt.subplots(10,10,dpi=300,figsize=(10,10), sharex=True, sharey=True)
    # axs = axs.flatten()
    # for c in range(100):
    #     if c == 9:
    #         continue
    #     axs[c].plot(win_times, normR[c], 'tab:red')
    #     axs[c].plot(win_times, normL[c], 'tab:blue')
    #     axs[c].hlines(0, win_times[0], win_times[-1], ls='--', color='k', alpha=0.3)
    #     axs[c].hlines(0, -0.1, 0.1, ls='--', color='k', alpha=0.3)
    # fig.suptitle('leftward saccadic PETH')
    # fig.tight_layout()
    # fig.savefig('saccadic_PETH_spikes_left.png')

    peth_dict = {
        'leftward_onsets': leftward_onsets,
        'rightward_onsets': rightward_onsets,
        'upward_onsets': upward_onsets,
        'downward_onsets': downward_onsets,
        'right_theta_movement_inds': right_theta_movement_inds,
        'left_theta_movement_inds': left_theta_movement_inds,
        'down_phi_movement_inds': down_phi_movement_inds,
        'up_phi_movement_inds': up_phi_movement_inds,
        'right_PETHs': right_PETHs,
        'left_PETHs': left_PETHs,
        'up_PETHs': up_PETHs,
        'down_PETHs': down_PETHs,
        'norm_right_PETHs': normR,
        'norm_left_PETHs': normL,
        'norm_up_PETHs': normU,
        'norm_down_PETHs': normD
    }

    return peth_dict



def calc_PETHs_IMU(data):

    theta_full = np.rad2deg(data['theta'][data['eyeT_startInd']:data['eyeT_endInd']])
    phi_full = np.rad2deg(data['phi'][data['eyeT_startInd']:data['eyeT_endInd']])
    eyeT = data['eyeT'][data['eyeT_startInd']:data['eyeT_endInd']]
    eyeT = eyeT - eyeT[0]
    twopT = data['twopT']
    dt = 1/60
    dTheta = np.diff(theta_full) / dt
    dPhi = np.diff(phi_full) / dt

    dHead = data['gyro_z_eye_interp'].copy()[:-1]
    dGaze = dTheta.copy() + dHead
    eyeT = eyeT[:-1]

    leftward_gazeshift_onsets = get_event_onsets(eyeT[(
        (dHead > 60) *
        (dGaze > 240)
        )],
        min_frames=4
    )
    leftward_gazeshift_inds = np.array([fm2p.find_closest_timestamp(twopT, t)[0] for t in leftward_gazeshift_onsets if not np.isnan(t)])

    rightward_gazeshift_onsets = get_event_onsets(eyeT[(
        (dHead < -60) *
        (dGaze < -240)
        )],
        min_frames=4
    )
    rightward_gazeshift_inds = np.array([fm2p.find_closest_timestamp(twopT, t)[0] for t in rightward_gazeshift_onsets if not np.isnan(t)])

    leftward_compensatory_onsets = get_event_onsets(eyeT[(
        (dHead > 60) *
        (dGaze < 120) *
        (dGaze > -120)
        )],
        min_frames=4
    )
    leftward_compensatory_inds = np.array([fm2p.find_closest_timestamp(twopT, t)[0] for t in leftward_compensatory_onsets if not np.isnan(t)])

    rightward_compensatory_onsets = get_event_onsets(eyeT[(
        (dHead < -60) *
        (dGaze < 120) *
        (dGaze > -120)
        )],
        min_frames=4
    )
    rightward_compensatory_inds = np.array([fm2p.find_closest_timestamp(twopT, t)[0] for t in rightward_compensatory_onsets if not np.isnan(t)])

    win_frames = np.arange(-15,16)
    _, left_gaze_PETHs = calc_hist_PETH(data['norm_spikes'], leftward_gazeshift_inds, win_frames)
    _, right_gaze_PETHs = calc_hist_PETH(data['norm_spikes'], rightward_gazeshift_inds, win_frames)
    _, left_comp_PETHs = calc_hist_PETH(data['norm_spikes'], leftward_compensatory_inds, win_frames)
    _, right_comp_PETHs = calc_hist_PETH(data['norm_spikes'], rightward_compensatory_inds, win_frames)

    norm_left_gaze = norm_psth(left_gaze_PETHs)
    norm_right_gaze = norm_psth(right_gaze_PETHs)
    norm_left_comp = norm_psth(left_comp_PETHs)
    norm_right_comp = norm_psth(right_comp_PETHs)

    peth_dict = {
        'left_gaze_onsets': leftward_gazeshift_inds,
        'right_gaze_onsets': rightward_gazeshift_inds,
        'left_comp_onsets': leftward_compensatory_inds,
        'right_comp_onsets': rightward_compensatory_inds,
        'right_gaze_PETHs': right_gaze_PETHs,
        'left_gaze_PETHs': left_gaze_PETHs,
        'right_comp_PETHs': right_comp_PETHs,
        'left_comp_PETHs': left_comp_PETHs,
        'norm_right_gaze_PETHs': norm_left_gaze,
        'norm_left_gaze_PETHs': norm_right_gaze,
        'norm_right_comp_PETHs': norm_left_comp,
        'norm_left_comp_PETHs': norm_right_comp
    }

    return peth_dict

