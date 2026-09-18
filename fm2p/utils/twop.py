# -*- coding: utf-8 -*-
"""
fm2p/utils/twop.py

Two-photon calcium imaging data processing.

Classes
-------
TwoP
    Load suite2p outputs, compute dF/F, extract transients, normalize spikes.

Functions
---------
run_omsi
    MCMC spike inference with OMSI, run out-of-process.
bin_spike_times_to_frames
    Discrete spike times (s) to per-frame spike RATE on the 2P timebase.
calc_inf_spikes
    Run OASIS deconvolution on a dF/F array.
normalize_axonal_spikes
    Clip outliers and scale spike estimates to [0, 1] per cell.
zscore_spikes
    Z-score spike estimates across time for each cell.


DMM, December 2024
"""

import warnings
warnings.filterwarnings('ignore')

import os
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.stats
try:
    import oasis
    _oasis_available = True
except ModuleNotFoundError:
    _oasis_available = False

from .paths import up_dir
from .files import write_h5


class TwoP():
    """ Load suite2p outputs and compute dF/F, transients, and normalized spikes.

    Parameters are stored from a YAML config; the main workflow is:
    1. ``find_files`` or ``add_files`` or ``add_data`` to load raw arrays.
    2. ``calc_dFF`` to compute neuropil-corrected dF/F.
    3. ``calc_dFF_transients`` to extract supra-threshold events.
    4. ``normalize_spikes`` to produce [0, 1]-normalized spike estimates.
    """

    def __init__(self, recording_path='', recording_name='', cfg=None):
        """ Initialize with a recording path and optional config.

        Parameters
        ----------
        recording_path : str
        recording_name : str
        cfg : dict, str, or None
            Config dict, path to a YAML file, or None to use internals.yaml.
        """

        self.recording_path = recording_path
        self.recording_name = recording_name

        if cfg is None:
            internals_config_path = os.path.join(up_dir(__file__, 1), 'internals.yaml')
            with open(internals_config_path, 'r') as infile:
                cfg = yaml.load(infile, Loader=yaml.FullLoader)
        elif type(cfg) == str:
            with open(cfg, 'r') as infile:
                cfg = yaml.load(infile, Loader=yaml.FullLoader)

        self.cfg = cfg

        self.dt = 1. / cfg['twop_rate']

        self.dFF = None
        self.nCells = None

    def find_files(self):
        """ Load suite2p outputs from the canonical suite2p/plane0/ subfolder. """

        self.F = np.load(os.path.join(self.recording_path, r'suite2p/plane0/F.npy'), allow_pickle=True)
        self.Fneu = np.load(os.path.join(self.recording_path, r'suite2p/plane0/Fneu.npy'), allow_pickle=True)
        iscell = np.load(os.path.join(self.recording_path, r'suite2p/plane0/iscell.npy'), allow_pickle=True)
        spks = np.load(os.path.join(self.recording_path, r'suite2p/plane0/spks.npy'), allow_pickle=True)

        usecells = iscell[:, 0] == 1

        self.F = self.F[usecells, :]
        self.Fneu = self.Fneu[usecells, :]
        self.s2p_spks = spks[usecells, :]

    def add_files(self, F_path=None, Fneu_path=None, spikes_path=None, iscell_path=None, base_path=None):
        """ Load suite2p npy files from explicit paths or a common base directory.

        Parameters
        ----------
        F_path, Fneu_path, spikes_path, iscell_path : str or None
            Individual file paths. If all None and base_path is given, defaults
            to <base_path>/F.npy, Fneu.npy, iscell.npy, spks.npy.
        base_path : str or None
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

        usecells = iscell[:, 0] == 1

        self.F = self.F[usecells, :]
        self.Fneu = self.Fneu[usecells, :]
        self.s2p_spks = spks[usecells, :]

        self.usecells = usecells

    def add_data(self, F, Fneu, spikes, iscell):
        """ Attach pre-loaded numpy arrays instead of reading from disk.

        Parameters
        ----------
        F, Fneu, spikes, iscell : np.ndarray
            Raw suite2p arrays (all ROIs, before iscell filtering).
        """

        usecells = iscell[:, 0] == 1

        self.F = F[usecells, :]
        self.Fneu = Fneu[usecells, :]
        self.s2p_spks = spikes[usecells, :]
        self.nCells = np.size(self.F, 0)

        self.usecells = usecells

    def calc_dFF(self, neu_correction=0.7, use_oasis=True):
        """ Compute neuropil-corrected dF/F and optionally OASIS-deconvolved spikes.

        F0 is estimated as the histogram mode (peak bin center) of the
        fluorescence distribution, giving a robust baseline even with
        transient-skewed distributions.

        Parameters
        ----------
        neu_correction : float
            Neuropil subtraction coefficient (suite2p default 0.7).
        use_oasis : bool
            Run OASIS AR1 deconvolution to get denoised_dFF and oasis_spks.

        Returns
        -------
        twop_dict : dict
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

            F_cell = F[c, :].copy()
            F_cell_neu = Fneu[c, :].copy()

            _lo, _hi = np.nanpercentile(F_cell, [1, 99])
            _counts, _edges = np.histogram(F_cell, bins=300, range=(_lo, _hi))
            _pk = np.argmax(_counts)
            _f0_raw = 0.5 * (_edges[_pk] + _edges[_pk + 1])

            _raw_dFF = (F_cell - _f0_raw) / _f0_raw

            _normF = F_cell - neu_correction * F_cell_neu + neu_correction * np.nanmean(F_cell_neu)

            _lo_n, _hi_n = np.nanpercentile(_normF, [1, 99])
            _counts_n, _edges_n = np.histogram(_normF, bins=300, range=(_lo_n, _hi_n))
            _pk_n = np.argmax(_counts_n)
            _f0_norm = 0.5 * (_edges_n[_pk_n] + _edges_n[_pk_n + 1])

            norm_dFF[c, :] = (_normF - _f0_norm) / _f0_norm

            if use_oasis:

                g = oasis.functions.estimate_time_constant(norm_dFF[c, :].copy(), 1)
                denoised_dFF[c, :], sps[c, :] = oasis.oasisAR1(norm_dFF[c, :].copy(), g)

            norm_F[c, :] = _normF
            raw_dFF[c, :] = _raw_dFF
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
        """ Write the fluorescence dict to HDF5 in the recording directory. """

        savedir = os.path.join(self.recording_path, self.recording_name)
        _savepath = os.path.join(savedir, '{}_twophoton.h5'.format(self.recording_name))
        write_h5(_savepath, twop_dict)

        return _savepath

    def calc_frame_mean_across_time(self, ops_path, bin_path):
        """ Compute per-frame mean pixel intensity from the suite2p binary.

        Parameters
        ----------
        ops_path : str
            Path to suite2p ops.npy.
        bin_path : str
            Path to the data.bin file.

        Returns
        -------
        frame_means : np.ndarray, shape (N_frames,)
        """

        ops = np.load(ops_path, allow_pickle=True).item()

        Ly, Lx = ops['Ly'], ops['Lx']
        nframes = ops['nframes']

        data = np.memmap(bin_path, dtype=np.int16, mode='r')
        data = data.reshape((nframes, Ly, Lx))

        frame_means = np.zeros(nframes, dtype=np.float64)

        for i in range(nframes):
            frame = data[i, :, :].reshape(Ly * Lx)
            frame_means[i] = frame.mean()

        self.frame_means = frame_means

        return frame_means

    def calc_dFF_transients(self):
        """ Extract supra-threshold dF/F events by zeroing sub-threshold frames.

        Returns
        -------
        dFF_transients : np.ndarray, shape (N_cells, N_frames)
        """

        sd_thresh = self.cfg['cell_sd_thresh']

        assert self.dFF is not None, 'dFF must be calculated before calling this method.'

        dFF = self.dFF.copy()

        dFF_transients = np.zeros_like(dFF)

        for c in range(self.nCells):
            sd = np.std(dFF[c, :])
            baseline_times = np.where(dFF[c, :] < (sd * sd_thresh))[0]
            mean_baseline = np.mean(dFF[c, baseline_times])
            sd_baseline = np.std(dFF[c, baseline_times])
            transient_times = np.where(dFF[c, :] > (sd_thresh * sd_baseline + mean_baseline))[0]
            dFF_transients[c, transient_times] = dFF[c, transient_times]

        self.dFF_transients = dFF_transients
        return dFF_transients

    def normalize_spikes(self):
        """ Clip outlier spikes and scale each cell's trace to [0, 1].

        Returns
        -------
        spikes : np.ndarray, shape (N_cells, N_frames)
        """

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

        spikes = spikes / np.max(spikes, axis=1, keepdims=True)
        self.cleanspikes = spikes

        return spikes

    def get_recording_props(self, stat, ops):
        """ Extract cell pixel positions and summary images from suite2p outputs.

        Parameters
        ----------
        stat : str or np.ndarray
            Path to stat.npy or already-loaded stat array.
        ops : str or np.ndarray
            Path to ops.npy or already-loaded ops array.

        Returns
        -------
        recording_props : dict
            'twop_mean_img', 'twop_ref_img', 'twop_max_proj',
            'twop_enhanced_mean_img', 'cell_x_pix', 'cell_y_pix'.
        """

        if type(stat) == str and type(ops) == str:
            stat = np.load(stat, allow_pickle=True)
            ops = np.load(ops, allow_pickle=True)
        elif type(stat) == np.ndarray and type(ops) == np.ndarray:
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


OMSI_ENV = 'spikeinf'


def run_omsi(raw_F, raw_Fneu, fs, env=OMSI_ENV, f_corr=0.7, timeout_s=None):
    """MCMC spike inference with OMSI, run out-of-process.

    OMSI (formerly fMCSI) lives in its own conda environment because it pulls in
    Ray and a different numpy line, so it is driven through `conda run` rather
    than imported.  Raw F and Fneu are passed rather than a pre-computed dF/F:
    OMSI's Poisson prior and MCMC likelihood are calibrated against its own
    8th-percentile-baseline dF/F, and feeding it the mode-based `norm_dFF` this
    module computes would change transient amplitudes and therefore the
    likelihood gain per spike proposal.

    Returns
    -------
    dict
        'spike_times' : list of (n_spikes,) arrays, seconds from frame 0
        'prob_trace'  : (n_cells, n_frames) expected spikes per frame
        Both are None if OMSI is unavailable or the run fails.

    Notes
    -----
    `deconv` also returns a 'spike_train' key.  It is NOT a per-cell binned
    array despite the name -- it comes back as (n_spikes_of_one_cell, n_frames)
    of one-hot rows, i.e. the ragged per-cell list collapsed onto a single cell.
    Verified on a 3-cell x 500-frame synthetic run with 10/20/30 spikes per
    cell: 'spikes' had lengths 10/20/30, 'prob_trace' was (3, 500) with row
    sums 10/20/29.6, and 'spike_train' was (30, 500) with every row summing
    to 1.  Use 'spikes' or 'prob_trace'; never 'spike_train'.
    """

    import tempfile, subprocess, os as _os

    f_arr    = np.ascontiguousarray(np.atleast_2d(raw_F),    dtype=np.float32)
    fneu_arr = np.ascontiguousarray(np.atleast_2d(raw_Fneu), dtype=np.float32)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f'run_omsi needs a positive frame rate, got {fs!r}')

    with tempfile.TemporaryDirectory() as tmpdir:
        f_path    = _os.path.join(tmpdir, 'f.npy')
        fneu_path = _os.path.join(tmpdir, 'fneu.npy')
        spk_path  = _os.path.join(tmpdir, 'spikes.npy')
        prob_path = _os.path.join(tmpdir, 'prob.npy')
        np.save(f_path, f_arr)
        np.save(fneu_path, fneu_arr)

        script = "\n".join([
            "import numpy as np, OMSI",
            "f    = np.load({})".format(repr(f_path)),
            "fneu = np.load({})".format(repr(fneu_path)),
            "r = OMSI.deconv_from_array(f=f, fneu=fneu, hz={}, f_corr={})".format(
                float(fs), float(f_corr)),
            "sp = [np.asarray(x, dtype=float).ravel() for x in r['spikes']]",
            "n = max((s.size for s in sp), default=1)",
            "out = np.full((len(sp), n), np.nan)",
            "for i, s in enumerate(sp): out[i, :s.size] = s",
            "np.save({}, out)".format(repr(spk_path)),
            "np.save({}, np.asarray(r['prob_trace'], dtype=np.float32))".format(
                repr(prob_path)),
        ])

        try:
            result = subprocess.run(
                ['conda', 'run', '-n', env, 'python', '-c', script],
                timeout=timeout_s,
            )
        except FileNotFoundError:
            print('  OMSI: conda not on PATH -- skipping (env {!r})'.format(env))
            return {'spike_times': None, 'prob_trace': None}
        except subprocess.TimeoutExpired:
            print('  OMSI: timed out after {} s -- skipping'.format(timeout_s))
            return {'spike_times': None, 'prob_trace': None}

        if result.returncode != 0 or not _os.path.isfile(spk_path):
            print('  OMSI: subprocess failed (returncode {}) '
                  '-- see output above'.format(result.returncode))
            return {'spike_times': None, 'prob_trace': None}

        padded = np.load(spk_path)
        prob   = np.load(prob_path) if _os.path.isfile(prob_path) else None

    spike_times = [row[np.isfinite(row)] for row in padded]
    return {'spike_times': spike_times, 'prob_trace': prob}


def bin_spike_times_to_frames(spike_times, twopT, t0=None):
    """Discrete spike times (s) to per-frame spike RATE on the 2P timebase.

    OMSI reports spike times in seconds measured from frame 0 of the array it
    was given, which is the first 2P frame, so the bins are built by shifting
    `twopT` to start at zero.  Edges are placed midway between frame times, so
    each bin is centered on its frame and the mapping is exact for a uniform
    timebase and still correct for a jittered one.

    Returns
    -------
    rate : (n_cells, len(twopT)) float array, spikes per second
    """

    twopT = np.asarray(twopT, dtype=float)
    n_fr = len(twopT)
    if n_fr < 2:
        raise ValueError('need at least two 2P frame times to bin into')
    if t0 is None:
        t0 = twopT[0]
    t = twopT - t0

    mids  = 0.5 * (t[1:] + t[:-1])
    edges = np.concatenate([[t[0] - (mids[0] - t[0])], mids,
                            [t[-1] + (t[-1] - mids[-1])]])
    width = np.diff(edges)

    rate = np.zeros((len(spike_times), n_fr), dtype=float)
    for i, st in enumerate(spike_times):
        st = np.asarray(st, dtype=float)
        st = st[np.isfinite(st)]
        if st.size:
            rate[i] = np.histogram(st, bins=edges)[0] / width
    return rate


def calc_inf_spikes(dFF, neu_correction=0.7, fps=7.49):
    """ Run OASIS AR1 deconvolution on a dF/F array.

    Parameters
    ----------
    dFF : np.ndarray, shape (N_cells, N_frames) or (N_frames,)
    neu_correction : float
        Unused; retained for API compatibility.
    fps : float
        Acquisition rate (frames per second).

    Returns
    -------
    denoised_dFF : np.ndarray
    sps : np.ndarray
    """

    dFF = np.squeeze(dFF)
    if dFF.ndim == 1:
        dFF = dFF[np.newaxis, :]
    nCells, lenT = np.shape(dFF)

    denoised_dFF = np.zeros([nCells, lenT])
    sps = np.zeros([nCells, lenT])

    for c in range(nCells):

        g = oasis.functions.estimate_time_constant(dFF[c, :].copy(), 1)
        denoised_dFF[c, :], sps[c, :] = oasis.oasisAR1(dFF[c, :].copy(), g)

    return denoised_dFF, sps


def normalize_axonal_spikes(spikes, cfg):
    """ Clip outlier spikes and scale axonal spike estimates to [0, 1] per cell.

    Parameters
    ----------
    spikes : np.ndarray, shape (N_cells, N_frames)
    cfg : dict
        Must contain 'cell_sd_thresh'.

    Returns
    -------
    norm_spikes : np.ndarray
    """

    sd_thresh = cfg['cell_sd_thresh']

    nCells = np.size(spikes, 0)

    for c in range(nCells):
        sp_ = spikes[c, :]
        std_ = np.std(sp_)
        mean_ = np.mean(sp_)

        sp_[sp_ > (mean_ + std_ * sd_thresh)] = mean_ + std_ * sd_thresh

        spikes[c, :] = sp_

    norm_spikes = spikes / np.max(spikes, axis=1, keepdims=True)

    return norm_spikes


def zscore_spikes(spikes):
    """ Z-score spike estimates across time for each cell.

    Parameters
    ----------
    spikes : np.ndarray, shape (N_cells, N_frames)

    Returns
    -------
    zspikes : np.ndarray
    """

    zspikes = np.zeros_like(spikes) * np.nan
    n_cells = np.size(spikes, 0)
    for c in range(n_cells):
        zspikes[c] = (spikes[c] - np.nanmean(spikes[c])) / np.nanstd(spikes[c])
    return zspikes
