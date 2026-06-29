# -*- coding: utf-8 -*-
"""
fm2p/utils/PETH.py

Peri-event time histogram (PETH) computation and gaze-saccade analysis.

Functions
---------
get_discrete_spike_times
    Run MCMC spike deconvolution and return spike times via subprocess.
norm_psth
    Normalize a PSTH by subtracting pre-event baseline and dividing by peak.
calc_hist_PETH
    Frame-aligned PETH from a discretized spike array.
calc_cont_PETH
    Continuous Gaussian-kernel PETH from a spike-times list.
calc_dff_peth
    Event-triggered dF/F PETH without spike inference.
calc_binned_PETH
    Binned-histogram PETH from a spike-times list.
norm_psth_paired
    Normalise two PSTHs together using the shared maximum.
find_trajectory_initiation
    Walk backwards from a peak to find the motion onset.
get_event_onsets
    Keep only the first event in each cluster of nearby events.
get_event_offsets
    Keep only the last event in each cluster of nearby events.
drop_nearby_events
    Remove events from one list that fall too close to another list.
drop_repeat_events
    Remove duplicated events within a short time window.
balanced_index_resample
    Undersample a signal to equalise occupancy across bins.
calc_PETH_mod_ind
    Modulation index of a PSTH relative to the pre-event baseline.
drop_redundant_saccades
    Remove saccades that are too close to other events or to each other.
calc_eye_head_movement_times
    Classify eye/head movements into gaze-shifting and compensatory types.
analyze_gaze_state_changes
    PETH analysis around gaze shifts and compensatory movements.


DMM, September 2025
"""

if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
    __package__ = 'fm2p.utils'

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from .time import interpT
    from .helper import interp_short_gaps
    from .imu import check_and_trim_imu_disconnect
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from fm2p.utils.time import interpT
    from fm2p.utils.helper import interp_short_gaps
    from fm2p.utils.imu import check_and_trim_imu_disconnect


def get_discrete_spike_times(raw_F, raw_Fneu, fs):
    """Run MCMC spike deconvolution and return per-cell spike times in seconds.

    Passes raw fluorescence arrays directly to fMCSI so that it can compute
    dF/F internally using its own 8th-percentile baseline.  fMCSI's Poisson
    prior and MCMC likelihood were calibrated for that specific dF/F scale.
    Passing a pre-computed dF/F with a different normalisation (e.g.
    mode-based or median-based) changes the transient amplitudes and therefore
    the likelihood gain per spike proposal, requiring compensatory changes to
    lam_scale.  For Suite2p data, always prefer passing raw_F / raw_Fneu.

    Parameters
    ----------
    raw_F : (n_cells, n_frames) array
        Raw Suite2p fluorescence (F.npy).
    raw_Fneu : (n_cells, n_frames) array
        Raw Suite2p neuropil fluorescence (Fneu.npy).
    fs : float
        Imaging frame rate in Hz.

    Returns
    -------
    spike_times : list of np.ndarray
        Length n_cells.  Each element is a 1-D array of spike times in
        seconds, measured from frame 0 of the recording.
    """
    import tempfile, subprocess, os

    f_arr    = np.ascontiguousarray(np.atleast_2d(raw_F),    dtype=np.float32)
    fneu_arr = np.ascontiguousarray(np.atleast_2d(raw_Fneu), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        f_path    = os.path.join(tmpdir, 'f.npy')
        fneu_path = os.path.join(tmpdir, 'fneu.npy')
        out_path  = os.path.join(tmpdir, 'spikes.npy')
        np.save(f_path,    f_arr)
        np.save(fneu_path, fneu_arr)

        script = "\n".join([
            "import numpy as np, fMCSI",
            "f    = np.load({})".format(repr(f_path)),
            "fneu = np.load({})".format(repr(fneu_path)),
            "results = fMCSI.deconv_from_array(f=f, fneu=fneu, hz={})".format(float(fs)),
            "spikes = results['spikes']",
            "max_len = max((len(s) for s in spikes), default=1)",
            "out = np.full((len(spikes), max_len), np.nan)",
            "for i, s in enumerate(spikes): out[i, :len(s)] = s",
            "np.save({}, out)".format(repr(out_path)),
        ])

        result = subprocess.run(
            ['conda', 'run', '-n', 'spikeinf', 'python', '-c', script],
        )

        if result.returncode != 0:
            raise RuntimeError(
                'fMCSI subprocess failed (see output above for details).'
            )

        out = np.load(out_path)

    # Convert NaN-padded 2-D array back to list of 1-D arrays
    spike_times = [row[~np.isnan(row)] for row in out]
    return spike_times


def norm_psth(mean_psth):
    """ Normalise a PSTH by subtracting the pre-event baseline and dividing by the peak. """

    psth_norm = np.zeros_like(mean_psth) * np.nan
    for c in range(np.size(mean_psth,0)):
        x = mean_psth[c,:].copy()
        # index into first ten so that i'm normalizing by the baseline not the responsive period
        psth_norm[c,:] = (x - np.nanmean(x[:10])) / np.nanmax(x)
    return psth_norm


def calc_hist_PETH(spikes, event_frames, window_bins):
    """ Frame-aligned PETH from a discretized spike array.

    Parameters
    ----------
    spikes : np.ndarray, shape (N_cells, N_frames)
    event_frames : array-like
        Frame indices of events.
    window_bins : array-like
        Offsets (in frames) relative to each event frame.

    Returns
    -------
    mean_psth, stderr, mean_psth_norm, stderr_norm : np.ndarray
    """

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
    """ Gaussian-kernel PETH from a list of spike times per cell.

    Parameters
    ----------
    spike_times_list : list of np.ndarray
        One array of spike times (seconds) per cell.
    event_times : array-like
    window : [float, float]
        [start, end] in seconds relative to event.
    sigma : float
        Gaussian kernel width (seconds).
    resolution : float
        Time resolution of the output (seconds).

    Returns
    -------
    mean_peth, stderr_peth : np.ndarray, shape (N_cells, N_points)
    time_axis : np.ndarray
    """

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


def calc_dff_peth(dff, frame_times, event_times, window=(-0.75, 0.75)):
    """Compute event-triggered dF/F PETH without any spike inference.

    Parameters
    ----------
    dff : (n_cells, n_frames) array
        dF/F traces.
    frame_times : (n_frames,) array
        Timestamp of each frame in seconds.
    event_times : (n_events,) array
        Event times in seconds.
    window : tuple
        (pre, post) window in seconds.

    Returns
    -------
    mean_peth : (n_cells, n_bins) array
    stderr_peth : (n_cells, n_bins) array
    time_axis : (n_bins,) array
        Frame times relative to event onset, sampled at frame_times resolution.
    """
    frame_times = np.asarray(frame_times)
    event_times = np.asarray(event_times)
    dt = np.median(np.diff(frame_times))
    n_pre  = int(round(abs(window[0]) / dt))
    n_post = int(round(abs(window[1]) / dt))
    n_bins = n_pre + n_post
    time_axis = np.arange(-n_pre, n_post) * dt

    n_cells  = dff.shape[0]
    n_frames = dff.shape[1]

    trials = []
    for t_ev in event_times:
        center = int(np.argmin(np.abs(frame_times - t_ev)))
        i0 = center - n_pre
        i1 = center + n_post
        if i0 < 0 or i1 > n_frames:
            continue
        trials.append(dff[:, i0:i1])   # (n_cells, n_bins)

    if len(trials) == 0:
        return (np.zeros((n_cells, n_bins)),
                np.zeros((n_cells, n_bins)),
                time_axis)

    stack = np.stack(trials, axis=0)   # (n_trials, n_cells, n_bins)
    mean_peth   = np.mean(stack,  axis=0)
    stderr_peth = np.std(stack, axis=0) / np.sqrt(len(trials))
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
    """ Normalise two PSTHs jointly, using the shared maximum across both. """

    psth1_norm = np.zeros_like(mean_psth1) * np.nan
    psth2_norm = np.zeros_like(mean_psth2)*np.nan
    for c in range(np.size(mean_psth1,0)):
        x1 = mean_psth1[c].copy()
        x2 = mean_psth2[c].copy()
        max_val = np.nanmax([np.nanmax(x1), np.nanmax(x2)])
        psth1_norm[c,:] = (x1 - np.nanmean(x1[:10])) / max_val
        psth2_norm[c,:] = (x2 - np.nanmean(x2[:10])) / max_val
    return psth1_norm, psth2_norm


def find_trajectory_initiation(signal, time, peak_times, smoothing_window=2):
    """ Walk backwards from each peak to find the onset of the trajectory.

    Parameters
    ----------
    signal : array-like
        1D signal (e.g., angular velocity).
    time : array-like
        Timestamps corresponding to signal.
    peak_times : array-like
        Times of detected peaks.
    smoothing_window : int
        Moving-average kernel width for jitter suppression.

    Returns
    -------
    np.ndarray
        Onset time for each peak.
    """

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
    """ Keep only the first event in each cluster of nearby events. """

    event_times = np.sort(np.asarray(event_times))
    min_gap = min_frames / sample_rate  # minimum time between events
    
    onsets = [event_times[0]]  # always keep first event
    for t in event_times[1:]:
        if t - onsets[-1] >= min_gap:
            onsets.append(t)
    
    return np.array(onsets)


def get_event_offsets(event_times, sample_rate=7.5, min_frames=4):
    """ Keep only the last event in each cluster of nearby events. """

    event_times = np.sort(np.asarray(event_times))
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
    """ Remove events from `thin` that fall within `win` seconds of any event in `avoid`. """

    to_drop = np.array([c for c in thin for g in avoid if ((g > (c - win)) & (g < (c + win)))])
    thinned = np.delete(thin, np.isin(thin, to_drop))
    return thinned


def drop_repeat_events(eventT, onset=True, win=0.020):
    """ Remove duplicated events within a short time window.

    Parameters
    ----------
    eventT : array-like
    onset : bool
        If True, keep the first event in each cluster; if False, keep the last.
    win : float
        Time window (seconds) for clustering.

    Returns
    -------
    np.ndarray
    """

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
    """ Undersample a signal to equalise occupancy across behavioural bins.

    Parameters
    ----------
    signal : array-like
        Continuous variable to balance over (e.g., speed).
    bin_edges : array-like
        Histogram edges defining the bins.
    random_state : int or None

    Returns
    -------
    final_indices : np.ndarray, dtype int
        Selected frame indices; empty array if no non-empty bins found.
    """

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
    """ Modulation index of a single-cell PSTH relative to the pre-event baseline. """

    baseline = np.nanmean(psth[:8])
    modind = (np.nanmax(psth) - baseline) / (np.nanmax(psth) + baseline)
    return modind


def drop_redundant_saccades(mov, to_avoid=None, near_win=0.20, repeat_win=0.15, onset=True):
    """ Remove saccades too close to other events or to each other.

    Parameters
    ----------
    mov : array-like
        Event times to thin.
    to_avoid : array-like or None
        If given, remove events from `mov` within `near_win` seconds of these.
    near_win : float
        Minimum gap (seconds) to `to_avoid` events.
    repeat_win : float
        Minimum gap (seconds) between kept events in `mov`.
    onset : bool
        If True, keep the first of each repeat cluster; if False, keep the last.

    Returns
    -------
    np.ndarray
    """

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
    """ Classify eye/head movements into gaze-shifting and compensatory types.

    Parameters
    ----------
    data : dict
        Preprocessed session data containing eyeT, theta, phi, imuT_trim, gyro_z_trim,
        eyeT_startInd, and eyeT_endInd.

    Returns
    -------
    saccade_dict : dict
        'gaze_left', 'gaze_right', 'comp_left', 'comp_right', 'dTheta', 'dHead',
        'dGaze', 'eyeT1', 'dPhi'.
    """

    eyeT = data['eyeT'][data['eyeT_startInd']:data['eyeT_endInd']]
    eyeT = eyeT - eyeT[0]
    t = eyeT.copy()[:-1]
    t1 = t + (np.diff(eyeT) / 2)
    imuT = data['imuT_trim']
    dHead = - interpT(data['gyro_z_trim'], imuT, t1)
    theta_full = np.rad2deg(data['theta'][data['eyeT_startInd']:data['eyeT_endInd']])
    dEye  = np.diff(interp_short_gaps(theta_full, 5)) / np.diff(eyeT)
    dEye = np.roll(dEye, -2) # static offset correction

    # also calculate dPhi
    phi_full = np.rad2deg(data['phi'][data['eyeT_startInd']:data['eyeT_endInd']])
    dPhi  = np.diff(interp_short_gaps(phi_full, 5)) / np.diff(eyeT)
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
        (dHead < -shifted_head) &
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


def analyze_gaze_state_changes(data, savepath=None, use_mcmc=True, spike_times=None):
    """ Compute PETHs around gaze shifts and classify cells as motor vs position.

    Parameters
    ----------
    data : dict
        Session data; must contain raw_F, raw_Fneu, twopT, and all eyeT/IMU fields
        required by calc_eye_head_movement_times.
    savepath : str or None
        If given, write a summary PDF at this path.
    use_mcmc : bool
        If True, use fMCSI MCMC spike inference; if False, use OASIS.
    spike_times : list of np.ndarray or None
        Pre-computed spike times; if None they will be calculated.

    Returns
    -------
    results : dict
        Per-event-type PETH data (hist, kde, and time axes).
    cell_cats : list of int
        0=unclassified, 1=motor, 2=position, one per cell.
    spike_times : list of np.ndarray
    """
    
    # Trim data for IMU disconnects
    data = check_and_trim_imu_disconnect(data)
    
    # Recalculate movement onsets with trimmed data
    saccade_dict = calc_eye_head_movement_times(data)
    data.update(saccade_dict)
    
    # Check event counts before proceeding
    n_gl = len(data.get('gaze_left', []))
    n_gr = len(data.get('gaze_right', []))
    n_cl = len(data.get('comp_left', []))
    n_cr = len(data.get('comp_right', []))
    
    if n_gl < 50 or n_gr < 50 or n_cl < 50 or n_cr < 50:
        print('Skipping analysis -- insufficient events (L={}, R={}, cL={}, cR={})'.format(
            n_gl, n_gr, n_cl, n_cr))
        return None, None, None
    
    print('Sufficient events found. (L={}, R={}, cL={}, cR={})'.format(n_gl, n_gr, n_cl, n_cr))

    twopT = data['twopT']
    raw_F    = data['raw_F']
    raw_Fneu = data['raw_Fneu']
    n_cells = raw_F.shape[0]

    dt = np.median(np.diff(twopT))
    fs = 1/dt

    if spike_times is None:
        if use_mcmc:
            print('Calculating spike times (MCMC)...')
            spike_times = get_discrete_spike_times(raw_F, raw_Fneu, fs=fs)
        else:
            print('Calculating spike times (OASIS)...')
            spike_times = []
            try:
                from oasis.functions import deconvolve
                Fc  = raw_F - 0.7 * raw_Fneu
                F0  = np.percentile(Fc, 8, axis=1, keepdims=True)
                dff = (Fc - F0) / np.abs(F0)
                t0 = time.time()
                for i in range(n_cells):
                    # Estimate gamma for tau=0.5
                    g = np.exp(-1 / (fs * 0.5))
                    c, s, b, g_est, lam = deconvolve(dff[i], g=(g,), penalty=1)
                    # Simple thresholding for OASIS spikes
                    spk_indices = np.where(s > 0.05)[0]
                    spike_times.append(spk_indices / fs)
                print('OASIS took {:.2f} s'.format(time.time() - t0))
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
    
    bin_size  = 0.020  # 20 ms histogram bins
    kde_sigma = 0.040  # 40 ms Gaussian sigma
    kde_res   = 0.005  # 5 ms KDE resolution

    peth_time = np.arange(peth_win_start, peth_win_end + bin_size/1000.0, bin_size)[:-1] + bin_size/2

    # Indices for calculation relative to PETH window
    idx_pre  = np.where((peth_time >= pre_win[0])  & (peth_time <= pre_win[1]))[0]
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

        # --- histogram PETH (spikes/s) ---
        mean_hist, stderr_hist, time_hist = calc_binned_PETH(
            spike_times,
            events,
            window=[peth_win_start, peth_win_end],
            bin_size=bin_size,
        )

        if len(time_hist) != len(peth_time):
            peth_time = time_hist
            idx_pre  = np.where((peth_time >= pre_win[0])  & (peth_time <= pre_win[1]))[0]
            idx_post = np.where((peth_time >= post_win[0]) & (peth_time <= post_win[1]))[0]
            idx_late = np.where((peth_time >= (post_win[1] - 1.0)) & (peth_time <= post_win[1]))[0]

        # --- KDE PETH (spikes/s) ---
        mean_kde, stderr_kde, time_kde = calc_cont_PETH(
            spike_times,
            events,
            window=[peth_win_start, peth_win_end],
            sigma=kde_sigma,
            resolution=kde_res,
        )

        results[key] = {
            'hist':        mean_hist,    # (n_cells, n_bins)  spikes/s
            'hist_stderr': stderr_hist,
            'hist_time':   time_hist,
            'kde':         mean_kde,     # (n_cells, n_points) spikes/s
            'kde_stderr':  stderr_kde,
            'kde_time':    time_kde,
        }

    # Categorize cells — uses histogram PETH for consistency
    cell_cats = []  # 0: None, 1: Motor, 2: Position

    for c in range(n_cells):
        is_pos = False
        is_motor = False

        # Check Gaze (Left/Right) for categorization
        psth_list = []
        if 'gaze_left'  in results: psth_list.append(results['gaze_left']['hist'][c])
        if 'gaze_right' in results: psth_list.append(results['gaze_right']['hist'][c])
        
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
        title_str = 'Gaze Shifts: L={}, R={}  --  Compensatory: L={}, R={}'.format(
            n_gl, n_gr, n_cl, n_cr)

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
                        ax_gaze.plot(results['gaze_left']['hist_time'], results['gaze_left']['hist'][c], color='blue', label='Left', linewidth=1)
                    if 'gaze_right' in results:
                        ax_gaze.plot(results['gaze_right']['hist_time'], results['gaze_right']['hist'][c], color='red', label='Right', linewidth=1)
                    ax_gaze.axvline(0, color='k', linestyle='--', linewidth=0.5)

                    # Compensatory (Right column)
                    ax_comp = axs[i, 1]
                    if 'comp_left' in results:
                        ax_comp.plot(results['comp_left']['hist_time'], results['comp_left']['hist'][c], color='blue', label='Left', linewidth=1)
                    if 'comp_right' in results:
                        ax_comp.plot(results['comp_right']['hist_time'], results['comp_right']['hist'][c], color='red', label='Right', linewidth=1)
                    ax_comp.axvline(0, color='k', linestyle='--', linewidth=0.5)
                    
                    cat_str = ["None", "Motor", "Position"][cell_cats[c]]
                    ax_gaze.set_ylabel('Cell {}\n{}'.format(c, cat_str), fontsize=8)
                    
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


