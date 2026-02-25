# -*- coding: utf-8 -*-



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import time
import json
from tqdm import tqdm

import fm2p


def norm_psth(mean_psth):
    psth_norm = np.zeros_like(mean_psth)*np.nan
    for c in range(np.size(mean_psth,0)):
        x = mean_psth[c,:].copy()
        # index into first ten so that i'm normalizing by the baseline not the responsive period
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

    mean_psth = np.nanmean(psth, axis=1)  # average across events
    stderr = np.nanstd(psth, axis=1) / np.sqrt(np.size(psth, axis=1))

    mean_psth_norm = norm_psth(mean_psth)
    stderr_norm = norm_psth(stderr)

    return mean_psth, stderr, mean_psth_norm, stderr_norm


def calc_cont_PETH(spike_times_list, event_times, window=[-0.75, 0.75], sigma=0.04, resolution=0.02):
    event_times = np.asarray(event_times)
    n_cells = len(spike_times_list)
    
    t_start, t_end = window
    # Ensure we cover the full window
    time_axis = np.arange(t_start, t_end + resolution/100.0, resolution)
    n_points = len(time_axis)
    
    mean_peth = np.zeros((n_cells, n_points))
    stderr_peth = np.zeros((n_cells, n_points))
    
    norm_factor = 1.0 / (np.sqrt(2 * np.pi) * sigma)
    margin = 4 * sigma 
    
    for c, cell_spikes in enumerate(tqdm(spike_times_list, desc="Calculating continuous PETH")):
        cell_spikes = np.sort(np.asarray(cell_spikes))
        if len(cell_spikes) == 0:
            continue
            
        trial_traces = np.zeros((len(event_times), n_points))
        
        for i, t_ev in enumerate(event_times):
            t_min = t_ev + t_start - margin
            t_max = t_ev + t_end + margin
            
            idx_start = np.searchsorted(cell_spikes, t_min)
            idx_end = np.searchsorted(cell_spikes, t_max)
            
            spikes_in_window = cell_spikes[idx_start:idx_end]
            
            if len(spikes_in_window) == 0:
                continue
                
            rel_times = spikes_in_window - t_ev
            
            # Vectorized kernel addition
            diffs = rel_times[:, np.newaxis] - time_axis[np.newaxis, :]
            kernels = np.exp(-0.5 * (diffs / sigma)**2)
            trial_traces[i, :] = np.sum(kernels, axis=0) * norm_factor

        mean_peth[c, :] = np.mean(trial_traces, axis=0)
        stderr_peth[c, :] = np.std(trial_traces, axis=0) / np.sqrt(len(event_times))
        
    return mean_peth, stderr_peth, time_axis


def calc_binned_PETH(spike_times_list, event_times, window=[-0.5, 0.5], bin_size=0.05):
    """
    Calculate PETH using binned histograms.
    
    Parameters
    ----------
    spike_times_list : list of np.ndarray
        List where each element is an array of spike times (in seconds) for a cell.
    event_times : np.ndarray
        Array of event times (in seconds).
    window : list or tuple
        [start, end] time relative to event (seconds).
    bin_size : float
        Size of bins in seconds.
        
    Returns
    -------
    mean_peth : np.ndarray
        (n_cells, n_bins) Firing rate in Hz.
    stderr_peth : np.ndarray
        (n_cells, n_bins) Standard error of the mean.
    time_axis : np.ndarray
        (n_bins,) Time points (centers of bins).
    """
    event_times = np.asarray(event_times)
    n_cells = len(spike_times_list)
    
    bins = np.arange(window[0], window[1] + bin_size/1000.0, bin_size)
    time_axis = bins[:-1] + bin_size/2
    n_bins = len(time_axis)
    
    mean_peth = np.zeros((n_cells, n_bins))
    stderr_peth = np.zeros((n_cells, n_bins))
    
    for c, cell_spikes in enumerate(tqdm(spike_times_list, desc="Calculating binned PETH")):
        cell_spikes = np.sort(np.asarray(cell_spikes))
        if len(cell_spikes) == 0:
            continue
            
        trial_counts = np.zeros((len(event_times), n_bins))
        
        for i, t_ev in enumerate(event_times):
            t_min = t_ev + window[0]
            t_max = t_ev + window[1]
            
            # Find spikes in window
            idx_start = np.searchsorted(cell_spikes, t_min)
            idx_end = np.searchsorted(cell_spikes, t_max)
            
            spikes_in_window = cell_spikes[idx_start:idx_end]
            rel_times = spikes_in_window - t_ev
            
            counts, _ = np.histogram(rel_times, bins=bins)
            trial_counts[i, :] = counts
        
        # Convert counts to rate (Hz)
        trial_rates = trial_counts / bin_size
        
        mean_peth[c, :] = np.mean(trial_rates, axis=0)
        stderr_peth[c, :] = np.std(trial_rates, axis=0) / np.sqrt(len(event_times))
        
    return mean_peth, stderr_peth, time_axis


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


def get_event_onsets(event_times, sample_rate=7.5, min_frames=4):

    event_times = np.sort(np.asarray(event_times))  # ensure sorted
    min_gap = min_frames / sample_rate  # minimum time between events
    
    onsets = [event_times[0]]  # always keep first event
    for t in event_times[1:]:
        if t - onsets[-1] >= min_gap:
            onsets.append(t)
    
    return np.array(onsets)


def get_event_offsets(event_times, sample_rate=7.5, min_frames=4):

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


def drop_redundant_saccades(mov, to_avoid=None, near_win=0.20, repeat_win=0.15, onset=True):

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

    # also calculate dPhi
    phi_full = np.rad2deg(data['phi'][data['eyeT_startInd']:data['eyeT_endInd']])
    dPhi  = np.diff(fm2p.interp_short_gaps(phi_full, 5)) / np.diff(eyeT)
    dPhi = np.roll(dPhi, -2)

    dGaze = dHead + dEye

    shifted_head = 60
    still_gaze = 120
    shifted_gaze = 240

    # gaze-shifting saccades
    gaze_left = t1[(
        (dHead > shifted_head) &
        (dGaze > shifted_gaze)
    )]
    gaze_right = t1[(
        (dHead < -shifted_gaze) &
        (dGaze < -shifted_gaze)
    )]

    # compensatory eye/head movements
    comp_left = t1[(
        (dHead > shifted_head) &
        (dGaze < still_gaze)   &
        (dGaze > -still_gaze)
    )]
    comp_right = t1[(
        (dHead < -shifted_head) &
        (dGaze < still_gaze)    &
        (dGaze > -still_gaze)
    )]

    gaze_left = drop_redundant_saccades(gaze_left)
    gaze_right = drop_redundant_saccades(gaze_right)

    # with two arguments, it also removes nearby events.
    # otherwise, just the repeated events
    comp_left = drop_redundant_saccades(comp_left, comp_right)
    comp_right = drop_redundant_saccades(comp_right, comp_left)

    saccade_dict = {
        'gaze_left': gaze_left,
        'gaze_right': gaze_right,
        'comp_left': comp_left,
        'comp_right': comp_right,
        'dTheta': dEye,
        'dHead': dHead,
        'dGaze': dGaze,
        'eyeT1': t1,
        'dPhi': dPhi
    }

    return saccade_dict


def calc_PETHs(data):

    # Trim data for IMU disconnects
    data = fm2p.check_and_trim_imu_disconnect(data)

    saccade_dict = calc_eye_head_movement_times(data)
    sps = data['norm_spikes']
    dFF = data['raw_dFF']

    win_frames = np.arange(-15,16)
    win_times = win_frames*(1/7.52)

    peth_dict= {
        'win_frames': win_frames,
        'win_times': win_times
    }

    vars = ['gaze_left', 'gaze_right', 'comp_left', 'comp_right']

    for i, varname in enumerate(vars):

        peth_sps, petherr_sps, norm_sps, norm_err = calc_hist_PETH(
            sps,
            saccade_dict[varname],
            win_frames
        )
        peth_dict['{}_peth_sps'.format(varname)] = peth_sps
        peth_dict['{}_peth_err_sps'.format(varname)] = petherr_sps
        peth_dict['{}_norm_peth_sps'.format(varname)] = norm_sps
        peth_dict['{}_norm_peth_err_sps'.format(varname)] = norm_err

        peth_dff, petherr_dff, norm_dff, norm_dff = calc_hist_PETH(
            dFF,
            saccade_dict[varname],
            win_frames
        )
        peth_dict['{}_peth_dff'.format(varname)] = peth_dff
        peth_dict['{}_peth_err_dff'.format(varname)] = petherr_dff
        peth_dict['{}_norm_peth_dff'.format(varname)] = norm_dff
        peth_dict['{}_norm_peth_err_dff'.format(varname)] = norm_dff

    dict_out = {**saccade_dict, **peth_dict,}

    return dict_out


def analyze_gaze_state_changes(data, savepath=None, use_mcmc=True, spike_times=None):
    """
    Analyze neural activity around gaze shifts and compensatory movements.
    Categorize cells as position or motor cells based on step vs decay.
    """
    
    # Trim data for IMU disconnects
    data = fm2p.check_and_trim_imu_disconnect(data)
    
    # Recalculate movement onsets with trimmed data
    saccade_dict = calc_eye_head_movement_times(data)
    data.update(saccade_dict)
    
    # Check event counts before proceeding
    n_gl = len(data.get('gaze_left', []))
    n_gr = len(data.get('gaze_right', []))
    n_cl = len(data.get('comp_left', []))
    n_cr = len(data.get('comp_right', []))
    
    if n_gl < 50 or n_gr < 50 or n_cl < 50 or n_cr < 50:
        print(f"Skipping analysis: insufficient events (L={n_gl}, R={n_gr}, cL={n_cl}, cR={n_cr})")
        return None, None, None

    twopT = data['twopT']
    dff = data['norm_dFF']
    n_cells = dff.shape[0]
    
    dt = np.median(np.diff(twopT))
    fs = 1/dt
    
    if spike_times is None:
        if use_mcmc:
            print('Calculating spike times (MCMC)...')
            spike_times, _, _ = fm2p.get_discrete_spike_times(dff, fs=fs)
        else:
            print('Calculating spike times (OASIS)...')
            spike_times = []
            try:
                from oasis.functions import deconvolve
                t0 = time.time()
                for i in range(n_cells):
                    # Estimate gamma for tau=0.5
                    g = np.exp(-1 / (fs * 0.5))
                    c, s, b, g_est, lam = deconvolve(dff[i], g=(g,), penalty=1)
                    # Simple thresholding for OASIS spikes
                    spk_indices = np.where(s > 0.05)[0]
                    spike_times.append(spk_indices / fs)
                print(f"OASIS took {time.time() - t0:.2f} s")
            except ImportError:
                print("OASIS package not found. Skipping OASIS comparison.")
                spike_times = [np.array([]) for _ in range(n_cells)]
    else:
        print('Using provided spike times.')
    
    
    # Event keys
    event_keys = ['gaze_left', 'gaze_right', 'comp_left', 'comp_right']
    
    # Windows for calculation (seconds)
    pre_win =  [-0.500, -0.100]
    post_win = [0.100,   0.500]
    
    # PETH window for plotting and analysis (seconds)
    peth_win_start = -0.750
    peth_win_end =    0.750
    
    bin_size = 0.02 # 20ms bins
    peth_time = np.arange(peth_win_start, peth_win_end + bin_size/1000.0, bin_size)[:-1] + bin_size/2
    
    # Indices for calculation relative to PETH window
    idx_pre = np.where((peth_time >= pre_win[0]) & (peth_time <= pre_win[1]))[0]
    idx_post = np.where((peth_time >= post_win[0]) & (peth_time <= post_win[1]))[0]
    # Late window for categorization (last 1s of post window) to check for decay
    idx_late = np.where((peth_time >= (post_win[1] - 1.0)) & (peth_time <= post_win[1]))[0]
    
    results = {}
    
    for key in event_keys:
        if key not in data:
            continue
        
        events = data[key]
        if len(events) == 0:
            continue
            
        mean_psth, stderr, time_axis = calc_binned_PETH(
            spike_times, 
            events, 
            window=[peth_win_start, peth_win_end]
        )
        
        if len(time_axis) != len(peth_time):
            peth_time = time_axis
            idx_pre = np.where((peth_time >= pre_win[0]) & (peth_time <= pre_win[1]))[0]
            idx_post = np.where((peth_time >= post_win[0]) & (peth_time <= post_win[1]))[0]
            idx_late = np.where((peth_time >= (post_win[1] - 1.0)) & (peth_time <= post_win[1]))[0]
        
        results[key] = mean_psth

    # Categorize cells
    cell_cats = [] # 0: None, 1: Motor, 2: Position
    
    for c in range(n_cells):
        is_pos = False
        is_motor = False
        
        # Check Gaze (Left/Right) for categorization
        psth_list = []
        if 'gaze_left' in results: psth_list.append(results['gaze_left'][c])
        if 'gaze_right' in results: psth_list.append(results['gaze_right'][c])
        
        if psth_list:
            # Pick max response (deviation from baseline)
            best_psth = max(psth_list, key=lambda x: np.max(np.abs(x - np.mean(x[idx_pre]))))
            
            base = np.mean(best_psth[idx_pre])
            post = np.mean(best_psth[idx_post])
            late = np.mean(best_psth[idx_late])
            
            step = post - base
            late_step = late - base
            
            # Thresholds (arbitrary, based on normalized spikes 0-1)
            if abs(step) > 0.01: # Responsive
                # Ratio of late response to overall post response
                if abs(late_step) > 0.33 * abs(step): # Sustained
                    is_pos = True
                else:
                    is_motor = True
        
        cell_cats.append(2 if is_pos else (1 if is_motor else 0))

    if savepath:
        # Get event counts for title
        n_gl = len(data.get('gaze_left', []))
        n_gr = len(data.get('gaze_right', []))
        n_cl = len(data.get('comp_left', []))
        n_cr = len(data.get('comp_right', []))
        title_str = f"Gaze Shifts: L={n_gl}, R={n_gr}  |  Compensatory: L={n_cl}, R={n_cr}"

        with PdfPages(savepath) as pdf:
            # Sort by category
            sort_idx = np.argsort(cell_cats)[::-1]
            
            cells_per_page = 10
            n_pages = int(np.ceil(n_cells / cells_per_page))
            
            for page in range(n_pages):
                fig, axs = plt.subplots(cells_per_page, 2, figsize=(8.5, 11))
                fig.suptitle(title_str, fontsize=10)
                
                start_idx = page * cells_per_page
                end_idx = min((page + 1) * cells_per_page, n_cells)
                page_cells = sort_idx[start_idx:end_idx]
                
                for i, c in enumerate(page_cells):
                    # Gaze Shifts (Left column)
                    ax_gaze = axs[i, 0]
                    if 'gaze_left' in results:
                        ax_gaze.plot(peth_time, results['gaze_left'][c], color='blue', label='Left', linewidth=1)
                    if 'gaze_right' in results:
                        ax_gaze.plot(peth_time, results['gaze_right'][c], color='red', label='Right', linewidth=1)
                    ax_gaze.axvline(0, color='k', linestyle='--', linewidth=0.5)
                    
                    # Compensatory (Right column)
                    ax_comp = axs[i, 1]
                    if 'comp_left' in results:
                        ax_comp.plot(peth_time, results['comp_left'][c], color='blue', label='Left', linewidth=1)
                    if 'comp_right' in results:
                        ax_comp.plot(peth_time, results['comp_right'][c], color='red', label='Right', linewidth=1)
                    ax_comp.axvline(0, color='k', linestyle='--', linewidth=0.5)
                    
                    cat_str = ["None", "Motor", "Position"][cell_cats[c]]
                    ax_gaze.set_ylabel(f'Cell {c}\n{cat_str}', fontsize=8)
                    
                    # Hide x labels for all but the last visible row
                    if i < len(page_cells) - 1:
                        ax_gaze.tick_params(labelbottom=False)
                        ax_comp.tick_params(labelbottom=False)
                
                # Hide unused axes
                for i in range(len(page_cells), cells_per_page):
                    axs[i, 0].axis('off')
                    axs[i, 1].axis('off')
                
                axs[0, 0].set_title('Gaze Shifts')
                axs[0, 1].set_title('Compensatory')
                
                axs[len(page_cells)-1, 0].set_xlabel('Time (s)')
                axs[len(page_cells)-1, 1].set_xlabel('Time (s)')
                
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                pdf.savefig(fig)
                plt.close(fig)
        
    return results, cell_cats, spike_times


def run_gaze_analysis(data, animal_dirs, root_dir, use_mcmc=True):
    
    print("Starting gaze state change analysis.")
    
    json_path = os.path.join(root_dir, 'mcmc_spike_times.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            try:
                all_spike_times = json.load(f)
            except json.JSONDecodeError:
                all_spike_times = {}
    else:
        all_spike_times = {}

    for animal in animal_dirs:
        if animal not in data: 
            continue
        print('Analyzing {}.'.format(animal))
        
        for poskey in tqdm(data[animal]['transform']):
            
            try:
                pos_num = int(poskey.replace('pos', ''))
                pos_str = f'pos{pos_num:02d}'
            except:
                continue
            
            filename_pattern = f'*{animal}*preproc.h5'
            try:
                candidates = fm2p.find(filename_pattern, root_dir, MR=False)
            except:
                continue
            
            valid_candidates = [c for c in candidates if pos_str in c]
            
            if not valid_candidates:
                continue
            
            ppath = fm2p.choose_most_recent(valid_candidates)

            try:
                pdata = fm2p.read_h5(ppath)
                
                savepath = os.path.join('/home/dylan/Documents/Github/freely-moving-2P', f'{animal}_{poskey}_gaze_state_changes.pdf')
                
                print(f"Analyzing {animal} {poskey}")
                
                current_spike_times = None
                if animal in all_spike_times and poskey in all_spike_times[animal]:
                    print(f"Loading spike times from JSON for {animal} {poskey}")
                    current_spike_times = [np.array(x) for x in all_spike_times[animal][poskey]]

                _, _, spike_times = fm2p.analyze_gaze_state_changes(pdata, savepath=savepath, use_mcmc=use_mcmc, spike_times=current_spike_times)
                
                if spike_times is None:
                    continue
                
                if current_spike_times is None:
                    if animal not in all_spike_times:
                        all_spike_times[animal] = {}
                    
                    all_spike_times[animal][poskey] = [st.tolist() for st in spike_times]
                    
                    with open(json_path, 'w') as f:
                        json.dump(all_spike_times, f)

            except Exception as e:
                print(f"Error analyzing {animal} {poskey}: {e}")


if __name__ == '__main__':

    data = fm2p.read_h5('/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites/pooled_260210.h5')
    root_dir = '/home/dylan/Storage/freely_moving_data/_V1PPC'
    animal_dirs = ['DMM056', 'DMM061']

    run_gaze_analysis(data, animal_dirs, root_dir)
