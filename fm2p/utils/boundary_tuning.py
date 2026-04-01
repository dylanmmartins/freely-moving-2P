# -*- coding: utf-8 -*-
"""
Boundary tuning analysis tools for freely-moving 2P experiments.

Computes egocentric boundary cell (EBC) and retinocentric boundary cell (RBC)
receptive fields for neurons recorded during navigation in a square arena.

Written 2025-2026, DMM
"""

import os
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import skew
from scipy.ndimage import label
from tqdm import tqdm
import multiprocessing

import matplotlib
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import warnings
warnings.filterwarnings('ignore')

import multiprocessing

if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
    __package__ = 'fm2p.utils'

from .cmap import make_parula
from .files import read_h5, write_h5
from .paths import find
from .correlation import corr2_coeff


def convert_bools_to_ints(data):

    new_dict = {}
    for key, value in data.items():
        if isinstance(value, dict):
            new_dict[key] = convert_bools_to_ints(value)
        elif isinstance(value, (bool, np.bool_)):
            new_dict[key] = int(value)
        elif isinstance(value, np.complex128):
            new_dict[key] = str(value)
        else:
            new_dict[key] = value
    return new_dict

_BT_ray_distances     = None
_BT_ray_dist_use_inds = None
_BT_spikes_all        = None
_BT_occupancy         = None
_BT_dist_bin_edges    = None
_BT_dist_bin_cents    = None
_BT_ray_width         = None
_BT_dist_bin_size     = None
_BT_USE_RF_CORRELATION         = True
_BT_USE_RF_SHUFFLE_CORRELATION = True



def _cast_frames_chunk(args):

    x_chunk, y_chunk, ang_chunk, ray_offsets_rad, walls = args
    N_frames = len(x_chunk)
    N_ang    = len(ray_offsets_rad)
    dists    = np.full((N_frames, N_ang), np.nan)
    for fi in range(N_frames):
        px, py   = x_chunk[fi], y_chunk[fi]
        base_ang = ang_chunk[fi]
        for ri, off in enumerate(ray_offsets_rad):
            ray_ang = base_ang + off
            cx, sy  = np.cos(ray_ang), np.sin(ray_ang)
            best    = np.inf
            for wall in walls:
                start = wall[0]
                vec   = wall[1] - wall[0]
                rel_x = px - start[0]
                rel_y = py - start[1]
                det   = vec[0] * sy - vec[1] * cx
                if det == 0:
                    continue
                t = (rel_x * sy - rel_y * cx) / det
                if t < 0 or t > 1:
                    continue
                ix = start[0] + t * vec[0] - px
                iy = start[1] + t * vec[1] - py
                if ix * cx + iy * sy < 0:
                    continue
                d = np.sqrt(ix * ix + iy * iy)
                if d < best:
                    best = d
            if best < np.inf:
                dists[fi, ri] = best
    return dists


def _ratemap_single_cell(args):
    """Compute the raw (unsmoothed) rate map for one cell using shared globals."""
    c, = args
    ray_distances     = _BT_ray_distances
    ray_dist_use_inds = _BT_ray_dist_use_inds # abs frame indices, aligned to ray_distances rows
    spikes_c          = _BT_spikes_all[c]
    occupancy         = _BT_occupancy
    dist_bin_edges    = _BT_dist_bin_edges
    ray_width         = _BT_ray_width
    N_ang   = int(360 / ray_width)
    N_dist  = len(dist_bin_edges) - 1
    min_occ = 8

    sp_frames = spikes_c[ray_dist_use_inds]   # (N_frames_rd,)

    rm = np.zeros((N_ang, N_dist))
    for a in range(N_ang):
        dists    = ray_distances[:, a]
        valid    = ~np.isnan(dists)
        bin_inds = np.digitize(dists[valid], dist_bin_edges) - 1
        inrange  = (bin_inds >= 0) & (bin_inds < N_dist)
        np.add.at(rm[a], bin_inds[inrange], sp_frames[valid][inrange])
    rm = rm / (occupancy + 1e-6)
    rm[occupancy < min_occ] = np.nan
    return rm


def _smooth_cell(args):

    rm, sigma = args
    nan_mask = np.isnan(rm)
    rm_fill  = rm.copy()
    rm_fill[nan_mask] = 0.
    weights  = (~nan_mask).astype(float)
    padded_v = np.vstack([rm_fill, rm_fill, rm_fill])
    padded_w = np.vstack([weights,  weights,  weights])
    sv = gaussian_filter(padded_v, sigma=sigma)
    sw = gaussian_filter(padded_w, sigma=sigma)
    N  = rm.shape[0]
    return sv[N: 2 * N, :] / (sw[N: 2 * N, :] + 1e-10)


def _process_cell(args):

    c, is_inverse_c, inv_crit_c, n_shfl = args

    ray_distances     = _BT_ray_distances
    ray_dist_use_inds = _BT_ray_dist_use_inds
    spikes_all        = _BT_spikes_all
    occupancy         = _BT_occupancy
    dist_bin_edges    = _BT_dist_bin_edges
    dist_bin_cents    = _BT_dist_bin_cents
    ray_width         = _BT_ray_width
    dist_bin_size     = _BT_dist_bin_size
    use_rf_corr  = _BT_USE_RF_CORRELATION
    use_rf_shfl  = _BT_USE_RF_SHUFFLE_CORRELATION

    N_ang    = int(360 / ray_width)
    N_dist   = len(dist_bin_edges) - 1
    min_occ  = 8
    max_sp   = spikes_all.shape[1]
    RF_CORR_THRESH = 0.6

    abs_inds = ray_dist_use_inds[:ray_distances.shape[0]]
    abs_inds = abs_inds[abs_inds < max_sp]
    spikes_c = spikes_all[c, :]

    angs_rad     = np.deg2rad(np.arange(0, 360, ray_width))
    angs_mesh, _ = np.meshgrid(angs_rad, dist_bin_cents, indexing='ij')
    nan_map      = np.full((N_ang, N_dist), np.nan)

    sp_frames = spikes_c[abs_inds]
    rm_raw    = np.zeros((N_ang, N_dist))
    for a in range(N_ang):
        dists    = ray_distances[:, a]
        valid    = ~np.isnan(dists)
        bin_inds = np.digitize(dists[valid], dist_bin_edges) - 1
        inrange  = (bin_inds >= 0) & (bin_inds < N_dist)
        np.add.at(rm_raw[a], bin_inds[inrange], sp_frames[valid][inrange])
    rm_raw = rm_raw / (occupancy + 1e-6)
    rm_raw[occupancy < min_occ] = np.nan

    rm_work = rm_raw.copy()
    if is_inverse_c:
        rm_work = np.nanmax(rm_work) - rm_work + np.nanmin(rm_work)
    total_w = np.nansum(rm_work)
    if total_w < 1e-10:
        mrl, mra = 0.0, 0.0
    else:
        mr  = np.nansum(rm_work * np.exp(1j * angs_mesh)) / total_w
        mrl = float(np.abs(mr))
        mra = float(np.arctan2(np.imag(mr), np.real(mr)))
        if mra < 0:
            mra += 2 * np.pi

    def _smooth(rm):
        nm = np.isnan(rm)
        rv = rm.copy();  rv[nm] = 0.
        wv = (~nm).astype(float)
        pv = np.vstack([rv, rv, rv])
        pw = np.vstack([wv, wv, wv])
        sv = gaussian_filter(pv, sigma=2.5)
        sw = gaussian_filter(pw, sigma=2.5)
        N  = rm.shape[0]
        return sv[N: 2*N, :] / (sw[N: 2*N, :] + 1e-10)

    def _subset_rm(subset_abs, sp_override=None):

        mask   = np.isin(abs_inds, subset_abs)
        rd_sub = ray_distances[mask, :]
        sp_sub = sp_frames[mask] if sp_override is None else sp_override[mask]
        rm_s   = np.zeros((N_ang, N_dist))
        for a in range(N_ang):
            d = rd_sub[:, a]
            v = ~np.isnan(d)
            b = np.digitize(d[v], dist_bin_edges) - 1
            ir = (b >= 0) & (b < N_dist)
            np.add.at(rm_s[a], b[ir], sp_sub[v][ir])
        occ = np.zeros((N_ang, N_dist))
        for di, lo in enumerate(dist_bin_edges[:-1]):
            hi = lo + dist_bin_size
            m2 = (rd_sub >= lo) & (rd_sub < hi)
            occ[:, di] = np.sum(m2, axis=0)
        rm_s = rm_s / (occ + 1e-6)
        rm_s[occ < min_occ] = np.nan
        return rm_s

    n_used = len(abs_inds)

    if n_used == 0:
        corr, corr_pass, rm1_s, rm2_s = np.nan, False, nan_map, nan_map
    else:
        mid  = n_used // 2
        rm1_s = _smooth(_subset_rm(abs_inds[:mid]))
        rm2_s = _smooth(_subset_rm(abs_inds[mid:]))

        valid = ~np.isnan(rm1_s) & ~np.isnan(rm2_s)
        if valid.sum() > 5:
            a_, b_ = rm1_s[valid], rm2_s[valid]
            denom  = np.std(a_) * np.std(b_)
            corr   = float(np.mean((a_ - a_.mean()) * (b_ - b_.mean())) / denom) \
                     if denom > 0 else np.nan
        else:
            corr = np.nan

        def _pref(rm):
            if np.all(np.isnan(rm)):
                return np.nan, np.nan
            fi = np.nanargmax(rm)
            ai, di = np.unravel_index(fi, rm.shape)
            return float(ai * ray_width), float(dist_bin_cents[di])

        ang1, dist1 = _pref(rm1_s)
        ang2, dist2 = _pref(rm2_s)
        if np.isnan(ang1) or np.isnan(ang2):
            corr_pass = False
        else:
            da = abs(((ang1 - ang2 + 180) % 360) - 180)
            md = (dist1 + dist2) / 2.0
            dd = abs(dist1 - dist2) / md if md > 0 else np.inf
            corr_pass = (da < 45.) and (dd < 0.5)

    rf_corr             = np.nan
    rf_corr_pass        = False
    rf_shfl_corrs       = np.zeros(n_shfl)
    rf_corr_shfl_thresh = np.nan
    rf_corr_shfl_pass   = False
    rm1_raw_blk         = nan_map.copy()
    rm2_raw_blk         = nan_map.copy()

    if n_used >= 10:

        block_size  = max(5, n_used // 20)
        n_blocks    = max(2, n_used // block_size)
        block_edges = np.linspace(0, n_used, n_blocks + 1).astype(int)
        block_list  = [abs_inds[block_edges[i]:block_edges[i + 1]]
                       for i in range(n_blocks)]
        rng_b = np.random.default_rng(seed=42 + c)
        perm  = rng_b.permutation(n_blocks)
        s1_inds = np.concatenate([block_list[b] for b in perm[: n_blocks // 2]])
        s2_inds = np.concatenate([block_list[b] for b in perm[n_blocks // 2 :]])

        rm1_raw_blk = _subset_rm(s1_inds)
        rm2_raw_blk = _subset_rm(s2_inds)

        valid_rf = ~np.isnan(rm1_raw_blk) & ~np.isnan(rm2_raw_blk)
        if valid_rf.sum() > 5:
            a_rf = rm1_raw_blk[valid_rf].reshape(1, -1)
            b_rf = rm2_raw_blk[valid_rf].reshape(1, -1)
            rf_corr = float(corr2_coeff(a_rf, b_rf))

        if use_rf_corr:
            rf_corr_pass = (not np.isnan(rf_corr)) and (rf_corr > RF_CORR_THRESH)

        if use_rf_shfl:
            for i in range(n_shfl):
                shift  = np.random.randint(1, max(n_used, 2))
                sp_sh  = np.roll(sp_frames, shift)
                rm1_sh = _subset_rm(s1_inds, sp_override=sp_sh)
                rm2_sh = _subset_rm(s2_inds, sp_override=sp_sh)
                v_sh   = ~np.isnan(rm1_sh) & ~np.isnan(rm2_sh)
                if v_sh.sum() > 5:
                    aa = rm1_sh[v_sh].reshape(1, -1)
                    bb = rm2_sh[v_sh].reshape(1, -1)
                    rf_shfl_corrs[i] = float(corr2_coeff(aa, bb))
            rf_corr_shfl_thresh = float(np.percentile(rf_shfl_corrs, 99))
            rf_corr_shfl_pass   = (not np.isnan(rf_corr)) and \
                                   (rf_corr > rf_corr_shfl_thresh)

    N_frames = ray_distances.shape[0]
    if N_frames < 10:
        mrl_thresh, mrl_pass = np.nan, False
        shfl_mrls = np.zeros(n_shfl)
    else:
        shfl_mrls = np.zeros(n_shfl)
        for i in range(n_shfl):
            shift = np.random.randint(1, N_frames)
            sp_sh = np.roll(sp_frames, shift)
            rm_sh = np.zeros((N_ang, N_dist))
            for a in range(N_ang):
                d  = ray_distances[:, a]
                v  = ~np.isnan(d)
                b  = np.digitize(d[v], dist_bin_edges) - 1
                ir = (b >= 0) & (b < N_dist)
                np.add.at(rm_sh[a], b[ir], sp_sh[v][ir])
            rm_sh = rm_sh / (occupancy + 1e-6)
            rm_sh[occupancy < min_occ] = np.nan
            if is_inverse_c:
                rm_sh = np.nanmax(rm_sh) - rm_sh + np.nanmin(rm_sh)
            tw = np.nansum(rm_sh)
            if tw > 1e-10:
                shfl_mrls[i] = float(np.abs(
                    np.nansum(rm_sh * np.exp(1j * angs_mesh)) / tw))
        mrl_thresh = float(np.percentile(shfl_mrls, 99))
        mrl_pass   = mrl > mrl_thresh

    is_bc = bool(rf_corr_pass and mrl_pass)

    cell_crit = {
        **inv_crit_c,
        'mean_resultant_length': float(mrl),
        'mean_resultant_angle':  float(mra),
        # legacy peak-stability split-half (not used for classification)
        'corr_coeff':   float(corr) if not np.isnan(corr) else float('nan'),
        'corr_pass':    int(corr_pass),
        # block-shuffle RF correlation (use this for selectivity)
        'rf_corr':              float(rf_corr) if not np.isnan(rf_corr) else float('nan'),
        'rf_corr_pass':         int(rf_corr_pass),
        'rf_corr_shfl_99pct':   float(rf_corr_shfl_thresh) if not np.isnan(rf_corr_shfl_thresh) else float('nan'),
        'rf_corr_shfl_pass':    int(rf_corr_shfl_pass),
        # MRL shuffle
        'mrl_99_pctl':  float(mrl_thresh) if not np.isnan(mrl_thresh) else float('nan'),
        'mrl_pass':     int(mrl_pass),
        'shuffled_mrls':    shfl_mrls,
        # old smoothed temporal halves
        'split_rate_map_1': rm1_s,
        'split_rate_map_2': rm2_s,
        # unsmoothed block-shuffle halves used by the RF correlation test
        'rf_split_map_1':   rm1_raw_blk,
        'rf_split_map_2':   rm2_raw_blk,
    }
    return is_bc, cell_crit


def rate_map_mp(spike_rate, occupancy, ray_distances, ray_width, dist_bin_edges, dist_bin_size):
    N_ang  = int(360 / ray_width)
    N_dist = len(dist_bin_edges) - 1
    rm     = np.zeros((N_ang, N_dist))
    for a in range(N_ang):
        dists    = ray_distances[:, a]
        valid    = ~np.isnan(dists)
        bin_inds = np.digitize(dists[valid], dist_bin_edges) - 1
        inrange  = (bin_inds >= 0) & (bin_inds < N_dist)
        np.add.at(rm[a], bin_inds[inrange], spike_rate[valid][inrange])
    rm /= occupancy + 1e-6
    return rm


def calc_MRL_mp(ratemap, ray_width, dist_bin_cents):
    angs_rad     = np.deg2rad(np.arange(0, 360, ray_width))
    angs_mesh, _ = np.meshgrid(angs_rad, dist_bin_cents, indexing='ij')
    total_weight = np.nansum(ratemap)
    if total_weight < 1e-10:
        return 0.0
    return float(np.abs(np.nansum(ratemap * np.exp(1j * angs_mesh)) / total_weight))


def calc_shfl_mean_resultant_mp(spikes, useinds, occupancy, ray_distances, ray_width,
                                dist_bin_edges, dist_bin_size, dist_bin_cents, is_inverse):
    N_frames     = int(np.sum(useinds))
    shift_amount = np.random.randint(1, max(N_frames, 2))
    shifted      = np.roll(spikes[useinds], shift_amount)
    rm           = rate_map_mp(shifted, occupancy, ray_distances,
                               ray_width, dist_bin_edges, dist_bin_size)
    if is_inverse:
        rm = np.max(rm) - rm + np.min(rm)
    return calc_MRL_mp(rm, ray_width, dist_bin_cents)


class BoundaryTuning:

    def __init__(self, preprocessed_data):
        self.data = preprocessed_data

        self.ray_width    = 3    # deg per angular bin
        self.max_dist     = 26   # cm
        self.dist_bin_size = 2.  # cm

        self.USE_RF_CORRELATION         = True  # corr2_coeff(half1, half2) > 0.6
        self.USE_RF_SHUFFLE_CORRELATION = True  # corr > 99th pctil of spike shuffles

        self.head_ang  = None
        self.pupil_ang = None
        self.ego_ang   = None

        self.criteria_out = {}
        for c in range(np.size(self.data['norm_spikes'], 0)):
            self.criteria_out['cell_{:03d}'.format(c)] = {}

        self.ebc_results = None
        self.rbc_results = None
        self.ebc_dark_results = None
        self.rbc_dark_results = None
        self.is_EBC = None
        self.is_RBC = None
        self.is_EBC_dark = None
        self.is_RBC_dark = None
        self.is_fully_reliable_EBC = None
        self.is_fully_reliable_RBC = None


    def calc_allo_yaw(self):

        self.head_ang = self.data['head_yaw_deg']

    def calc_allo_pupil(self):

        head  = self.data['head_yaw_deg'].copy()
        theta = self.data.get('theta_interp', None)

        if theta is not None:

            if 'ang_offset_vor_regression' in self.data:
                ang_offset = float(self.data['ang_offset_vor_regression'])
            elif 'ang_offset_vor_null' in self.data:
                ang_offset = float(self.data['ang_offset_vor_null'])
            else:
                from .ref_frame import calc_vor_eye_offset
                twopT = self.data.get('twopT', None)
                fps = (float(1.0 / np.nanmedian(np.diff(twopT)))
                       if twopT is not None and len(twopT) > 1 else 30.0)
                vor = calc_vor_eye_offset(theta, head, fps)
                ang_offset = vor['ang_offset_vor_regression']

            pfh = ang_offset - np.asarray(theta, dtype=float)
        else:
            pfh = self.data['pupil_from_head'].copy()

        pfh = np.asarray(pfh, dtype=float)

        nan_frac = float(np.isnan(pfh).mean())
        if nan_frac > 0.5:
            print(f'  [WARNING] Gaze offset (pfh) is {nan_frac*100:.0f}% NaN — '
                  f'eye tracking may have failed for this recording. '
                  f'RBC analysis will have few or no valid frames.')

        _pfh_range = float(np.nanmax(pfh) - np.nanmin(pfh))
        if 1e-6 < _pfh_range < 2.0:
            print(f'  [WARNING] Gaze offset (pfh) range = {_pfh_range:.4f} — '
                  f'consistent with radians (expected ≥ 20° for freely moving '
                  f'data). Auto-converting to degrees.')
            pfh = np.rad2deg(pfh)

        n = min(len(head), len(pfh))
        self.pupil_ang = (head[:n] + pfh[:n]) % 360

    def calc_ego(self):

        self.ego_ang = self.data['egocentric'] + 180.

    def _get_angle_trace(self, angle_type):

        if angle_type in ('head', 'egow'):
            if self.head_ang is None:
                self.calc_allo_yaw()
            return self.head_ang

        elif angle_type in ('gaze', 'pupil'):
            if self.pupil_ang is None:
                self.calc_allo_pupil()
            return self.pupil_ang

        elif angle_type in ('ego', 'egop'):
            if self.ego_ang is None:
                self.calc_ego()
            return self.ego_ang

        elif angle_type == 'retino':
            return self.data['retinocentric'] + 180.

        else:
            raise ValueError(f"Unknown angle_type '{angle_type}'. "
                             "Use 'head', 'gaze', 'ego', or 'retino'.")


    def _compute_ray_dists_from_trace(self, angle_trace_deg):

        p2c    = self.data['pxls2cm']
        x_full = self.data['head_x'].copy() / p2c
        y_full = self.data['head_y'].copy() / p2c

        max_valid = len(angle_trace_deg)
        use_inds  = np.where(self.useinds)[0]
        use_inds  = use_inds[use_inds < max_valid]

        nan_mask = np.isnan(angle_trace_deg[use_inds])
        if nan_mask.any():
            print(f'    [WARNING] {int(nan_mask.sum())}/{len(use_inds)} frames '
                  f'have NaN reference angle and will be excluded from ray casting.')
            use_inds = use_inds[~nan_mask]

        N_frames = len(use_inds)
        self._ray_dist_use_inds = use_inds.copy()

        if N_frames == 0:
            N_ang = int(360 / self.ray_width)
            print('    [WARNING] No valid frames remain after NaN filtering; '
                  'returning empty ray distance array.')
            return np.empty((0, N_ang))

        x_trace = x_full[use_inds]
        y_trace = y_full[use_inds]
        ang_rad = np.deg2rad(angle_trace_deg[use_inds])

        BL = (self.data['arenaBL']['x'] / p2c, self.data['arenaBL']['y'] / p2c)
        BR = (self.data['arenaBR']['x'] / p2c, self.data['arenaBR']['y'] / p2c)
        TR = (self.data['arenaTR']['x'] / p2c, self.data['arenaTR']['y'] / p2c)
        TL = (self.data['arenaTL']['x'] / p2c, self.data['arenaTL']['y'] / p2c)

        walls = [
            np.array([[BL[0], BL[1]], [BR[0], BR[1]]]),
            np.array([[BR[0], BR[1]], [TR[0], TR[1]]]),
            np.array([[TR[0], TR[1]], [TL[0], TL[1]]]),
            np.array([[TL[0], TL[1]], [BL[0], BL[1]]]),
        ]

        ray_offsets_rad = np.deg2rad(np.arange(0, 360, self.ray_width))

        self.dist_bin_edges = np.arange(0, self.max_dist + self.dist_bin_size,
                                        self.dist_bin_size)
        self.dist_bin_cents = self.dist_bin_edges[:-1] + self.dist_bin_size / 2

        n_workers  = max(1, multiprocessing.cpu_count() - 1)
        chunk_size = max(1, int(np.ceil(N_frames / n_workers)))
        chunks = [
            (x_trace[i: i + chunk_size],
             y_trace[i: i + chunk_size],
             ang_rad[i: i + chunk_size],
             ray_offsets_rad,
             walls)
            for i in range(0, N_frames, chunk_size)
        ]
        print(f'    ray casting: {N_frames} frames '
              f'across {len(chunks)} workers...')
        with multiprocessing.Pool(processes=n_workers) as pool:
            results = pool.map(_cast_frames_chunk, chunks)

        return np.vstack(results)


    def get_ray_distances(self, angle='head'):

        angle_trace = self._get_angle_trace(angle)
        self.ray_distances = self._compute_ray_dists_from_trace(angle_trace)
        return self.ray_distances

    def _compute_occupancy_from_raydists(self, ray_distances):

        N_ang  = int(360 / self.ray_width)
        N_dist = len(self.dist_bin_edges) - 1
        occupancy = np.zeros((N_ang, N_dist))

        for d, lo in enumerate(self.dist_bin_edges[:-1]):
            hi   = lo + self.dist_bin_size
            mask = (ray_distances >= lo) & (ray_distances < hi)
            occupancy[:, d] = np.sum(mask, axis=0)

        return occupancy


    def calc_occupancy(self, inds=None):

        if inds is None:
            return self._compute_occupancy_from_raydists(self.ray_distances)

        abs_inds = np.nonzero(self.useinds)[0]
        if isinstance(inds, np.ndarray) and inds.dtype == bool:
            target_inds = np.where(inds)[0]
        else:
            target_inds = np.asarray(inds)

        mask = np.isin(abs_inds, target_inds)
        rd_sub = self.ray_distances[mask, :]
        return self._compute_occupancy_from_raydists(rd_sub)


    def _compute_rate_maps_from_raydists(self, ray_distances, occupancy):
        """Compute rate maps for all cells in parallel (one worker per cell)."""
        global _BT_ray_distances, _BT_ray_dist_use_inds, _BT_spikes_all
        global _BT_occupancy, _BT_dist_bin_edges, _BT_dist_bin_cents
        global _BT_ray_width, _BT_dist_bin_size

        N_frames_rd      = ray_distances.shape[0]
        use_inds_clipped = self._ray_dist_use_inds[:N_frames_rd]
        max_sp           = self.data['norm_spikes'].shape[1]
        use_inds_clipped = use_inds_clipped[use_inds_clipped < max_sp]

        N_cells = self.data['norm_spikes'].shape[0]

        _BT_ray_distances     = ray_distances
        _BT_ray_dist_use_inds = use_inds_clipped
        _BT_spikes_all        = self.data['norm_spikes']
        _BT_occupancy         = occupancy
        _BT_dist_bin_edges    = self.dist_bin_edges
        _BT_dist_bin_cents    = self.dist_bin_cents
        _BT_ray_width         = self.ray_width
        _BT_dist_bin_size     = self.dist_bin_size

        n_workers = max(1, multiprocessing.cpu_count() - 1)
        with multiprocessing.Pool(processes=n_workers) as pool:
            results = pool.map(_ratemap_single_cell, [(c,) for c in range(N_cells)])

        return np.stack(results)

    def _compute_ratemap_for_cell_subset(self, c, split_abs_inds, ray_distances):

        max_sp = self.data['norm_spikes'].shape[1]
        use_inds_clipped = self._ray_dist_use_inds[:ray_distances.shape[0]]
        use_inds_clipped = use_inds_clipped[use_inds_clipped < max_sp]

        mask = np.isin(use_inds_clipped, split_abs_inds)
        rd_sub = ray_distances[mask, :]
        sp_sub = self.data['norm_spikes'][c, use_inds_clipped[mask]]

        N_ang  = int(360 / self.ray_width)
        N_dist = len(self.dist_bin_edges) - 1
        rm     = np.zeros((N_ang, N_dist))

        for a in range(N_ang):
            dists = rd_sub[:, a]
            valid = ~np.isnan(dists)
            bin_inds = np.digitize(dists[valid], self.dist_bin_edges) - 1
            inrange  = (bin_inds >= 0) & (bin_inds < N_dist)
            np.add.at(rm[a], bin_inds[inrange], sp_sub[valid][inrange])

        occ = self._compute_occupancy_from_raydists(rd_sub)
        min_occ = 8
        rm = rm / (occ + 1e-6)
        rm[occ < min_occ] = np.nan
        return rm

    def calc_rate_maps_mp(self):

        nCells     = np.size(self.data['norm_spikes'], 0)
        N_ang      = int(360 / self.ray_width)
        N_dist     = len(self.dist_bin_edges) - 1
        n_proc     = multiprocessing.cpu_count() - 1

        spikes = self.data['norm_spikes'].copy()[:, self.useinds.astype(bool)]
        pool = multiprocessing.Pool(processes=n_proc)
        mp_param_set = [
            pool.apply_async(
                rate_map_mp,
                args=(spikes[c], self.occupancy, self.ray_distances,
                      self.ray_width, self.dist_bin_edges, self.dist_bin_size),
            ) for c in range(nCells)
        ]
        outputs = [r.get() for r in mp_param_set]
        self.rate_maps = np.zeros((nCells, N_ang, N_dist))
        for c, rm in enumerate(outputs):
            self.rate_maps[c] = rm

        pool.close()
        return self.rate_maps

    def calc_rate_maps(self, use_mp=True):

        if use_mp:
            return self.calc_rate_maps_mp()
        self.rate_maps = self._compute_rate_maps_from_raydists(
            self.ray_distances, self.occupancy)
        return self.rate_maps

    def _smooth_single(self, rm, sigma=2.5):

        nan_mask   = np.isnan(rm)
        rm_fill    = rm.copy()
        rm_fill[nan_mask] = 0.
        weights    = (~nan_mask).astype(float)

        padded_v = np.vstack([rm_fill, rm_fill, rm_fill])
        padded_w = np.vstack([weights,  weights,  weights])

        sv = gaussian_filter(padded_v, sigma=sigma)
        sw = gaussian_filter(padded_w, sigma=sigma)

        N = rm.shape[0]
        smoothed = sv[N: 2 * N, :] / (sw[N: 2 * N, :] + 1e-10)
        return smoothed

    def _smooth_rate_maps_arr(self, rate_maps, sigma=2.5):

        n_workers = max(1, multiprocessing.cpu_count() - 1)
        args_list = [(rate_maps[c], sigma) for c in range(rate_maps.shape[0])]
        with multiprocessing.Pool(processes=n_workers) as pool:
            results = pool.map(_smooth_cell, args_list)
        return np.stack(results)

    def smooth_rate_maps(self):

        self.smoothed_rate_maps = self._smooth_rate_maps_arr(self.rate_maps)
        return self.smoothed_rate_maps

    def smooth_map_pair(self, map1, map2):

        return self._smooth_single(map1), self._smooth_single(map2)


    def _invert_ratemap(self, rm):

        return np.nanmax(rm) - rm + np.nanmin(rm)

    def _measure_skewness(self, rm):

        valid = rm[~np.isnan(rm)]
        sv = skew(valid) if len(valid) > 1 else np.nan
        return sv, bool(sv < 0.) if not np.isnan(sv) else False

    def _calc_dispersion(self, rm):

        N_ang, N_dist = rm.shape
        angs = np.deg2rad(np.arange(0, 360, self.ray_width))
        xc = np.zeros((N_ang, N_dist))
        yc = np.zeros((N_ang, N_dist))
        for a in range(N_ang):
            for d in range(N_dist):
                xc[a, d] = self.dist_bin_cents[d] * np.cos(angs[a])
                yc[a, d] = self.dist_bin_cents[d] * np.sin(angs[a])
        thresh  = np.nanpercentile(rm, 90)
        top     = rm >= thresh
        tx, ty  = xc[top], yc[top]
        if len(tx) < 2:
            return np.inf
        cx, cy = np.mean(tx), np.mean(ty)
        return np.mean(np.sqrt((tx - cx)**2 + (ty - cy)**2))

    def _measure_dispursion(self, rm):

        nd = self._calc_dispersion(rm)
        ni = self._calc_dispersion(self._invert_ratemap(rm))

        return nd, ni, ni < nd

    def _calc_receptive_field_size(self, rm):

        padded    = np.vstack([rm[-1, :], rm, rm[0, :]])
        binary    = padded >= np.nanpercentile(padded, 50)
        labeled, nfeat = label(binary, structure=np.ones((3, 3)))
        if nfeat == 0:
            return 0.
        labeled = labeled[1:-1, :]
        largest = max(np.sum(labeled == i) for i in range(1, nfeat + 1))

        return largest / rm.size

    def _measure_receptive_field_size(self, rm):

        nrm = self._calc_receptive_field_size(rm)
        nim = self._calc_receptive_field_size(self._invert_ratemap(rm))
        return nrm, nim, nim < nrm


    def _calc_mean_resultant(self, rm):

        N_ang, N_dist = rm.shape
        angs_rad      = np.deg2rad(np.arange(0, 360, self.ray_width))
        angs_mesh, _  = np.meshgrid(angs_rad, self.dist_bin_cents, indexing='ij')

        total_weight = np.nansum(rm)
        if total_weight < 1e-10:
            return 0 + 0j, 0.0, 0.0

        mr  = np.nansum(rm * np.exp(1j * angs_mesh)) / total_weight
        mrl = float(np.abs(mr))
        mra = float(np.arctan2(np.imag(mr), np.real(mr)))
        if mra < 0:
            mra += 2 * np.pi
        return mr, mrl, mra

    def _identify_inverse_responses_from(self, rate_maps, inv_thresh=2):

        N_cells    = rate_maps.shape[0]
        is_inverse = np.zeros(N_cells, dtype=bool)
        criteria   = [{}] * N_cells

        for c in range(N_cells):
            rm = rate_maps[c]
            sv, sp = self._measure_skewness(rm)
            _, _, dp = self._measure_dispursion(rm)
            _, _, rp = self._measure_receptive_field_size(rm)
            if sum([sp, dp, rp]) >= inv_thresh:
                is_inverse[c] = True
            criteria[c] = {
                'skewness_val':  float(sv),
                'skewness_pass': int(sp),
                'dispersion_pass': int(dp),
                'rf_size_pass':  int(rp),
                'is_inverse':    int(is_inverse[c]),
            }
        return is_inverse, criteria

    def identify_inverse_responses(self, inv_criteria_thresh=2):

        N_cells = self.rate_maps.shape[0]
        self.is_IEBC = np.zeros(N_cells, dtype=bool)
        is_inv, criteria = self._identify_inverse_responses_from(
            self.rate_maps, inv_criteria_thresh)
        self.is_IEBC = is_inv
        for c in range(N_cells):
            self.criteria_out['cell_{:03d}'.format(c)].update(criteria[c])
        return self.is_IEBC

    def _calc_correlation_across_split_v2(self, c, ray_distances,
                                          ncnk=20, corr_thresh=0.6):

        max_sp   = self.data['norm_spikes'].shape[1]
        abs_inds = self._ray_dist_use_inds[:ray_distances.shape[0]]
        abs_inds = abs_inds[abs_inds < max_sp]
        n_used   = len(abs_inds)

        if n_used == 0:
            nan_map = np.full((int(360 / self.ray_width),
                               len(self.dist_bin_edges) - 1), np.nan)
            return np.nan, False, nan_map, nan_map

        mid   = n_used // 2
        s1    = abs_inds[:mid]
        s2    = abs_inds[mid:]

        rm1 = self._compute_ratemap_for_cell_subset(c, s1, ray_distances)
        rm2 = self._compute_ratemap_for_cell_subset(c, s2, ray_distances)
        rm1_s, rm2_s = self.smooth_map_pair(rm1, rm2)


        valid = ~np.isnan(rm1_s) & ~np.isnan(rm2_s)
        if valid.sum() > 5:
            a, b = rm1_s[valid], rm2_s[valid]
            denom = np.std(a) * np.std(b)
            corr  = float(np.mean((a - a.mean()) * (b - b.mean())) / denom) if denom > 0 else np.nan
        else:
            corr = np.nan

        # Alexander 2020 reliability criteria
        def _pref_angle_dist(rm):

            rm_nan = rm.copy()
            if np.all(np.isnan(rm_nan)):
                return np.nan, np.nan
            flat_idx = np.nanargmax(rm_nan)
            ai, di   = np.unravel_index(flat_idx, rm_nan.shape)
            pref_ang = float(ai * self.ray_width) # deg
            pref_dist = float(self.dist_bin_cents[di]) # cm
            return pref_ang, pref_dist

        ang1, dist1 = _pref_angle_dist(rm1_s)
        ang2, dist2 = _pref_angle_dist(rm2_s)

        if np.isnan(ang1) or np.isnan(ang2):
            passes = False
        else:
            diff_ang  = abs(((ang1 - ang2 + 180) % 360) - 180)
            mean_dist = (dist1 + dist2) / 2.0
            diff_dist = abs(dist1 - dist2) / mean_dist if mean_dist > 0 else np.inf
            passes = (diff_ang < 45.0) and (diff_dist < 0.5)

        return corr, passes, rm1_s, rm2_s


    def _calc_single_ratemap_subsetting(self, c, inds):
        return self._compute_ratemap_for_cell_subset(c, inds, self.ray_distances)

    def _calc_correlation_across_split(self, c, ncnk=20, corr_thresh=0.6):
        corr, passes, rm1_s, rm2_s = self._calc_correlation_across_split_v2(
            c, self.ray_distances, ncnk, corr_thresh)
        self.criteria_out['cell_{:03d}'.format(c)].update({
            'split_rate_map_1': rm1_s,
            'split_rate_map_2': rm2_s,
        })
        return corr, passes

    def _test_mrl_against_shuffles(self, c, mrl, ray_distances, occupancy,
                                   is_inverse, n_shfl=100, pctl=99):

        N_frames = ray_distances.shape[0]

        max_sp = self.data['norm_spikes'].shape[1]
        use_inds_clipped = self._ray_dist_use_inds[:N_frames]
        use_inds_clipped = use_inds_clipped[use_inds_clipped < max_sp]

        spikes_cell = self.data['norm_spikes'][c, use_inds_clipped]
        N_ang  = int(360 / self.ray_width)
        N_dist = len(self.dist_bin_edges) - 1

        if N_frames < 10:
            nan_map = np.full((int(360 / self.ray_width),
                               len(self.dist_bin_edges) - 1), np.nan)
            return np.nan, False, np.zeros(n_shfl)

        shuffled_mrls = np.zeros(n_shfl)
        for i in range(n_shfl):

            shift = np.random.randint(1, N_frames)
            sp_sh = np.roll(spikes_cell, shift)

            rm_sh = np.zeros((N_ang, N_dist))
            for a in range(N_ang):
                dists = ray_distances[:, a]
                valid = ~np.isnan(dists)
                bins  = np.digitize(dists[valid], self.dist_bin_edges) - 1
                inrng = (bins >= 0) & (bins < N_dist)
                np.add.at(rm_sh[a], bins[inrng], sp_sh[valid][inrng])
            rm_sh = rm_sh / (occupancy + 1e-6)
            rm_sh[occupancy < 8] = np.nan

            if is_inverse:
                rm_sh = self._invert_ratemap(rm_sh)

            _, shf_mrl, _ = self._calc_mean_resultant(rm_sh)
            shuffled_mrls[i] = shf_mrl

        thresh  = float(np.percentile(shuffled_mrls, pctl))
        passes  = mrl > thresh
        return thresh, passes, shuffled_mrls

    def _test_mean_resultant_across_shuffles_mp(self, c, mrl, n_shfl=100,
                                                mrl_thresh_position=99):
        n_proc = multiprocessing.cpu_count() - 1
        pool   = multiprocessing.Pool(processes=n_proc)
        mp_set = [
            pool.apply_async(
                calc_shfl_mean_resultant_mp,
                args=(self.data['norm_spikes'][c].copy(),
                      self.useinds,
                      self.occupancy,
                      self.ray_distances,
                      self.ray_width,
                      self.dist_bin_edges,
                      self.dist_bin_size,
                      self.dist_bin_cents,
                      bool(self.is_IEBC[c]))
            ) for _ in range(n_shfl)
        ]
        shfl = np.array([r.get() for r in mp_set])
        thresh  = np.percentile(shfl, mrl_thresh_position)
        passes  = mrl > thresh
        self.criteria_out['cell_{:03d}'.format(c)]['shuffled_mrls'] = shfl
        pool.close()
        return thresh, passes

    def _test_mean_resultant_across_shuffles(self, c, mrl, n_shfl=100,
                                             mrl_thresh_position=99, use_mp=True):
        if use_mp:
            return self._test_mean_resultant_across_shuffles_mp(
                c, mrl, n_shfl, mrl_thresh_position)
        N_frames = int(np.sum(self.useinds))
        shfl = []
        for _ in range(n_shfl):
            sh   = np.random.randint(1, max(N_frames, 2))
            inds = np.roll(np.arange(N_frames), sh)
            rm   = self._calc_single_ratemap_subsetting(c, inds)
            if self.is_IEBC[c]:
                rm = self._invert_ratemap(rm)
            _, mrl_sh, _ = self._calc_mean_resultant(rm)
            shfl.append(mrl_sh)
        shfl   = np.array(shfl)
        thresh = np.percentile(shfl, mrl_thresh_position)
        return thresh, mrl > thresh

    def identify_boundary_cells(self, n_chunks=20, n_shuffles=20,
                                corr_thresh=0.6, mp=True):

        N_cells  = self.rate_maps.shape[0]
        self.is_EBC = np.zeros(N_cells, dtype=bool)

        for c in range(N_cells):
            rm = self.rate_maps[c].copy()
            if self.is_IEBC[c]:
                rm = self._invert_ratemap(rm)
            _, mrl, mra = self._calc_mean_resultant(rm)
            corr, corr_pass = self._calc_correlation_across_split(
                c, ncnk=n_chunks, corr_thresh=corr_thresh)
            thresh, mrl_pass = self._test_mean_resultant_across_shuffles(
                c, mrl, n_shfl=n_shuffles, use_mp=mp)
            if corr_pass and mrl_pass:
                self.is_EBC[c] = True
            self.criteria_out['cell_{:03d}'.format(c)].update({
                'mean_resultant_length': mrl,
                'mean_resultant_angle':  mra,
                'corr_coeff':   corr,
                'corr_pass':    int(corr_pass),
                'mrl_99_pctl':  thresh,
                'mrl_pass':     int(mrl_pass),
            })
        return self.criteria_out


    def _run_angle_pipeline(self, angle_type, n_chunks=20, n_shuffles=100,
                            corr_thresh=0.6):

        label = angle_type.upper()

        angle_trace  = self._get_angle_trace(angle_type)
        ray_distances = self._compute_ray_dists_from_trace(angle_trace)

        if ray_distances.shape[0] == 0:
            print(f'  [{label}] SKIPPED — no valid frames after NaN filtering '
                  f'(eye tracking likely failed for this recording).')
            N_cells = self.data['norm_spikes'].shape[0]
            N_ang   = int(360 / self.ray_width)
            N_dist  = len(self.dist_bin_edges) - 1
            nan_map = np.full((N_ang, N_dist), np.nan)
            return {
                'ray_distances':      ray_distances,
                'occupancy':          np.zeros((N_ang, N_dist)),
                'rate_maps':          np.full((N_cells, N_ang, N_dist), np.nan),
                'smoothed_rate_maps': np.full((N_cells, N_ang, N_dist), np.nan),
                'is_inverse':         np.zeros(N_cells, dtype=int),
                'is_bc':              np.zeros(N_cells, dtype=int),
                'criteria':           {},
                'angle_type':         angle_type,
                'ray_width':          self.ray_width,
                'dist_bin_edges':     self.dist_bin_edges,
                'dist_bin_cents':     self.dist_bin_cents,
                'angle_rad':          np.deg2rad(np.arange(0, 360, self.ray_width)),
            }

        occupancy = self._compute_occupancy_from_raydists(ray_distances)

        rate_maps = self._compute_rate_maps_from_raydists(ray_distances, occupancy)

        smoothed = self._smooth_rate_maps_arr(rate_maps)

        is_inverse, inv_crit = self._identify_inverse_responses_from(rate_maps)

        N_cells = rate_maps.shape[0]


        global _BT_ray_distances, _BT_ray_dist_use_inds, _BT_spikes_all
        global _BT_occupancy, _BT_dist_bin_edges, _BT_dist_bin_cents
        global _BT_ray_width, _BT_dist_bin_size
        global _BT_USE_RF_CORRELATION, _BT_USE_RF_SHUFFLE_CORRELATION

        _BT_ray_distances     = ray_distances
        _BT_ray_dist_use_inds = self._ray_dist_use_inds.copy()
        _BT_spikes_all        = self.data['norm_spikes']
        _BT_occupancy         = occupancy
        _BT_dist_bin_edges    = self.dist_bin_edges
        _BT_dist_bin_cents    = self.dist_bin_cents
        _BT_ray_width         = self.ray_width
        _BT_dist_bin_size     = self.dist_bin_size
        _BT_USE_RF_CORRELATION         = self.USE_RF_CORRELATION
        _BT_USE_RF_SHUFFLE_CORRELATION = self.USE_RF_SHUFFLE_CORRELATION

        args_list = [
            (c, bool(is_inverse[c]), inv_crit[c], n_shuffles)
            for c in range(N_cells)
        ]
        print(f'    [{label}] per-cell analysis: '
              f'{N_cells} cells across {max(1, multiprocessing.cpu_count()-1)} workers...')
        n_workers = max(1, multiprocessing.cpu_count() - 1)
        with multiprocessing.Pool(processes=n_workers) as pool:
            cell_results = pool.map(_process_cell, args_list)

        is_bc = np.zeros(N_cells, dtype=bool)
        cell_criteria = {}
        for c, (bc, crit) in enumerate(cell_results):
            is_bc[c] = bc
            cell_criteria['cell_{:03d}'.format(c)] = crit

        n_pass = int(np.sum(is_bc))

        return {
            'ray_distances':    ray_distances,
            'occupancy':        occupancy,
            'rate_maps':        rate_maps,
            'smoothed_rate_maps': smoothed,
            'is_inverse':       is_inverse.astype(int),
            'is_bc':            is_bc.astype(int),
            'criteria':         cell_criteria,
            'angle_type':       angle_type,
            'ray_width':        self.ray_width,
            'dist_bin_edges':   self.dist_bin_edges,
            'dist_bin_cents':   self.dist_bin_cents,
            'angle_rad':        np.deg2rad(np.arange(0, 360, self.ray_width)),
        }


    def _test_light_dark_stability(self, c):

        def _peak(rm):
            if rm is None or np.all(np.isnan(rm)):
                return np.nan, np.nan
            flat_idx = np.nanargmax(rm)
            ai, di   = np.unravel_index(flat_idx, rm.shape)
            return float(ai * self.ray_width), float(self.dist_bin_cents[di])

        def _compare(rm_light, rm_dark):
            ang_l, dist_l = _peak(rm_light)
            ang_d, dist_d = _peak(rm_dark)
            if np.isnan(ang_l) or np.isnan(ang_d):
                return False, np.nan, np.nan
            diff_ang  = abs(((ang_l - ang_d + 180) % 360) - 180)
            mean_dist = (dist_l + dist_d) / 2.0
            diff_dist = abs(dist_l - dist_d) / mean_dist if mean_dist > 0 else np.inf
            return (diff_ang < 45.) and (diff_dist < 0.5), diff_ang, diff_dist

        out = {}
        for key, light_res, dark_res in [
            ('ebc', self.ebc_results, self.ebc_dark_results),
            ('rbc', self.rbc_results, self.rbc_dark_results),
        ]:
            if dark_res is None:
                out[key] = {'passes': False, 'diff_ang': np.nan, 'diff_dist': np.nan}
                continue
            rm_l     = light_res['smoothed_rate_maps'][c]
            rm_d     = dark_res['smoothed_rate_maps'][c]
            passes, diff_ang, diff_dist = _compare(rm_l, rm_d)
            out[key] = {
                'passes':    passes,
                'diff_ang':  float(diff_ang),
                'diff_dist': float(diff_dist),
            }
        return out

    def identify_responses_both(self, n_chunks=20, n_shuffles=100, corr_thresh=0.6):

        N = self.data['norm_spikes'].shape[1]
        N = min(N, len(self.data['speed']))
        speed = self.data['speed'][:N]

        ltdk = self.data.get('ltdk_state_vec', None)
        if ltdk is not None:
            ltdk       = np.asarray(ltdk[:N])
            light_mask = (ltdk == 1)
            dark_mask  = (ltdk == 0)
        else:
            light_mask = np.ones(N, dtype=bool)
            dark_mask  = np.zeros(N, dtype=bool)

        self.calc_allo_yaw()
        self.calc_allo_pupil()


        print('  Running light-period EBC/RBC pipelines...')
        self.useinds = light_mask & (speed > 5.)
        self.ebc_results = self._run_angle_pipeline(
            'head', n_chunks, n_shuffles, corr_thresh)
        self.rbc_results = self._run_angle_pipeline(
            'gaze', n_chunks, n_shuffles, corr_thresh)
        self.is_EBC = self.ebc_results['is_bc'].astype(bool)
        self.is_RBC = self.rbc_results['is_bc'].astype(bool)


        dark_frames = int(dark_mask.sum())
        if dark_frames > 100:
            print(f'  Running dark-period EBC/RBC pipelines '
                  f'({dark_frames} frames)...')
            self.useinds = dark_mask & (speed > 5.)
            self.ebc_dark_results = self._run_angle_pipeline(
                'head', n_chunks, n_shuffles, corr_thresh)
            self.rbc_dark_results = self._run_angle_pipeline(
                'gaze', n_chunks, n_shuffles, corr_thresh)
            self.is_EBC_dark = self.ebc_dark_results['is_bc'].astype(bool)
            self.is_RBC_dark = self.rbc_dark_results['is_bc'].astype(bool)
        else:
            print(f'  Insufficient dark frames ({dark_frames}) — '
                  f'skipping dark-period analysis.')
            self.ebc_dark_results = None
            self.rbc_dark_results = None
            N_cells = len(self.is_EBC)
            self.is_EBC_dark = np.zeros(N_cells, dtype=bool)
            self.is_RBC_dark = np.zeros(N_cells, dtype=bool)


        self.is_fully_reliable_EBC = self.is_EBC & self.is_EBC_dark
        self.is_fully_reliable_RBC = self.is_RBC & self.is_RBC_dark

        return self.ebc_results, self.rbc_results


    def identify_responses(self, use_angle='head', use_light=False,
                           use_dark=False, skip_classification=False):
        
        N = self.data['norm_spikes'].shape[1]
        if use_light:
            useinds = (self.data['ltdk_state_vec'][:N].copy() == 1)
        elif use_dark:
            useinds = (self.data['ltdk_state_vec'][:N].copy() == 0)
        else:
            useinds = np.ones(N, dtype=bool)

        N = min(N, len(self.data['speed']))
        useinds = useinds[:N]
        self.useinds = useinds & (self.data['speed'][:N] > 5.)

        if use_angle == 'head':
            self.calc_allo_yaw()
        elif use_angle in ('pupil', 'gaze'):
            self.calc_allo_pupil()
        elif use_angle in ('ego', 'egop'):
            self.calc_ego()

        _ = self.get_ray_distances(angle=use_angle)

        self.occupancy = self.calc_occupancy(inds=self.useinds)

        _ = self.calc_rate_maps()

        _ = self.smooth_rate_maps()

        if not skip_classification:

            _ = self.identify_inverse_responses()

            _ = self.identify_boundary_cells()

        data_out = {
            'occupancy':          self.occupancy,
            'rate_maps':          self.rate_maps,
            'smoothed_rate_maps': self.smoothed_rate_maps,
            'ray_width':          self.ray_width,
            'max_dist':           self.max_dist,
            'dist_bin_size':      self.dist_bin_size,
            'bin_dist_edges':     self.dist_bin_edges,
            'dist_bin_cents':     self.dist_bin_cents,
            'ray_distances':      self.ray_distances,
            'angle_rad':          np.deg2rad(np.arange(0, 360, self.ray_width)),
        }
        if not skip_classification:
            data_out.update({
                'is_IEBC': self.is_IEBC.astype(int),
                'is_EBC':  self.is_EBC.astype(int),
                **self.criteria_out,
            })
        self.data_out = data_out
        return data_out


    def make_summary_pdf(self, savepath):

        assert self.ebc_results is not None and self.rbc_results is not None, \
            "Run identify_responses_both() before make_summary_pdf()."

        cmap        = make_parula()
        theta_edges = np.deg2rad(np.arange(0, 360 + self.ray_width, self.ray_width))
        r_edges     = self.dist_bin_edges
        has_dark    = self.ebc_dark_results is not None

        N_ang  = int(360 / self.ray_width)
        N_dist = len(self.dist_bin_edges) - 1
        _nan_rm  = np.full((N_ang, N_dist), np.nan)
        _nan_crit = {
            'mean_resultant_length': np.nan, 'mean_resultant_angle': np.nan,
            'rf_corr': np.nan, 'rf_corr_pass': 0,
            'corr_coeff': np.nan, 'corr_pass': 0,
            'mrl_99_pctl': np.nan, 'mrl_pass': 0,
            'shuffled_mrls': np.array([]),
        }

        def _get_crit(results, c):
            key = 'cell_{:03d}'.format(c)
            return results['criteria'].get(key, _nan_crit)

        def _get_rm(results, c):
            rm = results.get('smoothed_rate_maps')
            if rm is None or c >= len(rm):
                return _nan_rm
            return rm[c]

        show_cells = np.where(self.is_EBC | self.is_RBC)[0]
        if len(show_cells) == 0:
            print('  No reliable EBC or RBC cells found — no PDF generated.')
            return

        with PdfPages(savepath) as pdf:
            for c in show_cells:
                fig = plt.figure(figsize=(13, 11))


                ax_ebc_l = fig.add_axes([0.05, 0.56, 0.38, 0.38], projection='polar')
                ax_rbc_l = fig.add_axes([0.52, 0.56, 0.38, 0.38], projection='polar')
                ax_ebc_d = fig.add_axes([0.05, 0.12, 0.38, 0.38], projection='polar')
                ax_rbc_d = fig.add_axes([0.52, 0.12, 0.38, 0.38], projection='polar')
                cax      = fig.add_axes([0.93, 0.20, 0.012, 0.65])


                ebc_rm  = _get_rm(self.ebc_results, c)
                rbc_rm  = _get_rm(self.rbc_results, c)
                ck_ebc  = _get_crit(self.ebc_results, c)
                ck_rbc  = _get_crit(self.rbc_results, c)
                ebc_ok  = bool(self.is_EBC[c])
                rbc_ok  = bool(self.is_RBC[c])
                ebc_col = '#1a7f37' if ebc_ok else '#888888'
                rbc_col = '#1a5fa8' if rbc_ok else '#888888'


                all_vals = [ebc_rm.flatten(), rbc_rm.flatten()]
                if has_dark:
                    all_vals.append(
                        self.ebc_dark_results['smoothed_rate_maps'][c].flatten())
                    all_vals.append(
                        self.rbc_dark_results['smoothed_rate_maps'][c].flatten())
                vmax = np.nanpercentile(np.concatenate(all_vals), 99)
                vmax = max(vmax, 1e-6)

                im = ax_ebc_l.pcolormesh(theta_edges, r_edges, ebc_rm.T,
                                         cmap=cmap, shading='auto',
                                         vmin=0, vmax=vmax)
                ax_ebc_l.set_title(
                    f'EBC (light) — {"RELIABLE" if ebc_ok else "not reliable"}\n'
                    f'MRL={ck_ebc["mean_resultant_length"]:.3f}  '
                    f'RF_CC={ck_ebc.get("rf_corr", float("nan")):.3f}',
                    color=ebc_col, fontsize=9, pad=12)
                _polar_axes_style(ax_ebc_l,
                                  labels=['fwd', 'right', 'bkwd', 'left'],
                                  r_max=self.max_dist)

                ax_rbc_l.pcolormesh(theta_edges, r_edges, rbc_rm.T,
                                    cmap=cmap, shading='auto',
                                    vmin=0, vmax=vmax)
                ax_rbc_l.set_title(
                    f'RBC (light) — {"RELIABLE" if rbc_ok else "not reliable"}\n'
                    f'MRL={ck_rbc["mean_resultant_length"]:.3f}  '
                    f'RF_CC={ck_rbc.get("rf_corr", float("nan")):.3f}',
                    color=rbc_col, fontsize=9, pad=12)
                _polar_axes_style(ax_rbc_l,
                                  labels=['center', 'temporal', 'surround', 'nasal'],
                                  r_max=self.max_dist, shade_off_retina=True)

                if has_dark:
                    ebc_d_rm  = _get_rm(self.ebc_dark_results, c)
                    rbc_d_rm  = _get_rm(self.rbc_dark_results, c)
                    ck_ebc_d  = _get_crit(self.ebc_dark_results, c)
                    ck_rbc_d  = _get_crit(self.rbc_dark_results, c)
                    ebc_d_ok  = bool(self.is_EBC_dark[c])
                    rbc_d_ok  = bool(self.is_RBC_dark[c])
                    ebc_d_col = '#1a7f37' if ebc_d_ok else '#888888'
                    rbc_d_col = '#1a5fa8' if rbc_d_ok else '#888888'

                    ld = self._test_light_dark_stability(c)
                    def _ld_str(res):
                        if np.isnan(res['diff_ang']):
                            return 'L–D: n/a'
                        sym = '\u2713' if res['passes'] else '\u2717'
                        return (f'L\u2013D {sym}  \u0394ang={res["diff_ang"]:.0f}\u00b0  '
                                f'\u0394dist={res["diff_dist"]:.0%}')

                    ax_ebc_d.pcolormesh(theta_edges, r_edges, ebc_d_rm.T,
                                        cmap=cmap, shading='auto',
                                        vmin=0, vmax=vmax)
                    ax_ebc_d.set_title(
                        f'EBC (dark) — {"RELIABLE" if ebc_d_ok else "not reliable"}\n'
                        f'MRL={ck_ebc_d["mean_resultant_length"]:.3f}  '
                        f'RF_CC={ck_ebc_d.get("rf_corr", float("nan")):.3f}\n'
                        f'{_ld_str(ld["ebc"])}',
                        color=ebc_d_col, fontsize=9, pad=12)
                    _polar_axes_style(ax_ebc_d,
                                      labels=['fwd', 'right', 'bkwd', 'left'],
                                      r_max=self.max_dist)

                    ax_rbc_d.pcolormesh(theta_edges, r_edges, rbc_d_rm.T,
                                        cmap=cmap, shading='auto',
                                        vmin=0, vmax=vmax)
                    ax_rbc_d.set_title(
                        f'RBC (dark) — {"RELIABLE" if rbc_d_ok else "not reliable"}\n'
                        f'MRL={ck_rbc_d["mean_resultant_length"]:.3f}  '
                        f'RF_CC={ck_rbc_d.get("rf_corr", float("nan")):.3f}\n'
                        f'{_ld_str(ld["rbc"])}',
                        color=rbc_d_col, fontsize=9, pad=12)
                    _polar_axes_style(ax_rbc_d,
                                      labels=['center', 'temporal', 'surround', 'nasal'],
                                      r_max=self.max_dist, shade_off_retina=True)

                    _add_shuffle_inset(fig, ck_ebc_d, ebc_d_ok, color=ebc_d_col,
                                       rect=[0.51, 0.01, 0.16, 0.07])
                    _add_shuffle_inset(fig, ck_rbc_d, rbc_d_ok, color=rbc_d_col,
                                       rect=[0.74, 0.01, 0.16, 0.07])
                else:
                    for ax, lbl in [(ax_ebc_d, 'EBC (dark)'),
                                    (ax_rbc_d, 'RBC (dark)')]:
                        ax.set_title(lbl, fontsize=9, pad=12)
                        ax.text(0.5, 0.5, 'No dark data',
                                transform=ax.transAxes,
                                ha='center', va='center',
                                fontsize=10, color='gray')

                fig.colorbar(im, cax=cax, label='Rate (a.u.)')

                _add_shuffle_inset(fig, ck_ebc, ebc_ok, color=ebc_col,
                                   rect=[0.04, 0.01, 0.16, 0.07])
                _add_shuffle_inset(fig, ck_rbc, rbc_ok, color=rbc_col,
                                   rect=[0.27, 0.01, 0.16, 0.07])

                light_tag = ', '.join(
                    [x for x, ok in [('EBC', ebc_ok), ('RBC', rbc_ok)] if ok]
                ) or 'neither'
                full_parts = []
                if has_dark:
                    if self.is_fully_reliable_EBC[c]: full_parts.append('EBC')
                    if self.is_fully_reliable_RBC[c]: full_parts.append('RBC')
                full_tag = ', '.join(full_parts) or 'none'
                title = f'Cell {c:03d}   light reliable: {light_tag}'
                if has_dark:
                    title += f'   fully reliable (L+D): {full_tag}'
                fig.suptitle(title, fontsize=10)

                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

    def make_fully_reliable_pdf(self, savepath):

        assert self.ebc_results is not None and self.rbc_results is not None, \
            "Run identify_responses_both() before make_fully_reliable_pdf()."

        if self.ebc_dark_results is None:
            print('  No dark data available — skipping fully_reliable PDF.')
            return

        show_cells = np.where(
            self.is_fully_reliable_EBC | self.is_fully_reliable_RBC)[0]
        if len(show_cells) == 0:
            print('  No fully reliable (light + dark) cells found — '
                  'no fully_reliable PDF generated.')
            return

        cmap        = make_parula()
        theta_edges = np.deg2rad(np.arange(0, 360 + self.ray_width, self.ray_width))
        r_edges     = self.dist_bin_edges

        N_ang_fr  = int(360 / self.ray_width)
        N_dist_fr = len(self.dist_bin_edges) - 1
        _nan_rm_fr  = np.full((N_ang_fr, N_dist_fr), np.nan)
        _nan_crit_fr = {
            'mean_resultant_length': np.nan, 'mean_resultant_angle': np.nan,
            'rf_corr': np.nan, 'rf_corr_pass': 0,
            'corr_coeff': np.nan, 'corr_pass': 0,
            'mrl_99_pctl': np.nan, 'mrl_pass': 0,
            'shuffled_mrls': np.array([]),
        }

        def _gcrit(results, c):
            return results['criteria'].get('cell_{:03d}'.format(c), _nan_crit_fr)

        def _grm(results, c):
            rm = results.get('smoothed_rate_maps') if results else None
            return rm[c] if rm is not None and c < len(rm) else _nan_rm_fr

        print(f'  Writing fully-reliable PDF ({len(show_cells)} cells) -> {savepath}')
        with PdfPages(savepath) as pdf:
            for c in show_cells:
                fig = plt.figure(figsize=(13, 11))

                ax_ebc_l = fig.add_axes([0.05, 0.56, 0.38, 0.38], projection='polar')
                ax_rbc_l = fig.add_axes([0.52, 0.56, 0.38, 0.38], projection='polar')
                ax_ebc_d = fig.add_axes([0.05, 0.12, 0.38, 0.38], projection='polar')
                ax_rbc_d = fig.add_axes([0.52, 0.12, 0.38, 0.38], projection='polar')
                cax      = fig.add_axes([0.93, 0.20, 0.012, 0.65])

                ebc_rm   = _grm(self.ebc_results,      c)
                rbc_rm   = _grm(self.rbc_results,      c)
                ebc_d_rm = _grm(self.ebc_dark_results, c)
                rbc_d_rm = _grm(self.rbc_dark_results, c)
                ck_ebc   = _gcrit(self.ebc_results,      c)
                ck_rbc   = _gcrit(self.rbc_results,      c)
                ck_ebc_d = _gcrit(self.ebc_dark_results, c)
                ck_rbc_d = _gcrit(self.rbc_dark_results, c)
                ebc_ok   = bool(self.is_EBC[c])
                rbc_ok   = bool(self.is_RBC[c])
                ebc_d_ok = bool(self.is_EBC_dark[c])
                rbc_d_ok = bool(self.is_RBC_dark[c])

                vmax = np.nanpercentile(
                    np.concatenate([ebc_rm.flatten(), rbc_rm.flatten(),
                                    ebc_d_rm.flatten(), rbc_d_rm.flatten()]), 99)
                vmax = max(vmax, 1e-6)

                def _col(ok, base_col):
                    return base_col if ok else '#888888'

                ld = self._test_light_dark_stability(c)
                def _ld_str(res):
                    if np.isnan(res['diff_ang']):
                        return 'L-D: n/a'
                    sym = '\u2713' if res['passes'] else '\u2717'
                    return (f'L\u2013D {sym}  \u0394ang={res["diff_ang"]:.0f}\u00b0  '
                            f'\u0394dist={res["diff_dist"]:.0%}')

                im = ax_ebc_l.pcolormesh(theta_edges, r_edges, ebc_rm.T,
                                         cmap=cmap, shading='auto',
                                         vmin=0, vmax=vmax)
                ax_ebc_l.set_title(
                    f'EBC (light) — {"RELIABLE" if ebc_ok else "not reliable"}\n'
                    f'MRL={ck_ebc["mean_resultant_length"]:.3f}  '
                    f'RF_CC={ck_ebc.get("rf_corr", float("nan")):.3f}',
                    color=_col(ebc_ok, '#1a7f37'), fontsize=9, pad=12)
                _polar_axes_style(ax_ebc_l,
                                  labels=['fwd', 'right', 'bkwd', 'left'],
                                  r_max=self.max_dist)

                ax_rbc_l.pcolormesh(theta_edges, r_edges, rbc_rm.T,
                                    cmap=cmap, shading='auto',
                                    vmin=0, vmax=vmax)
                ax_rbc_l.set_title(
                    f'RBC (light) — {"RELIABLE" if rbc_ok else "not reliable"}\n'
                    f'MRL={ck_rbc["mean_resultant_length"]:.3f}  '
                    f'RF_CC={ck_rbc.get("rf_corr", float("nan")):.3f}',
                    color=_col(rbc_ok, '#1a5fa8'), fontsize=9, pad=12)
                _polar_axes_style(ax_rbc_l,
                                  labels=['center', 'temporal', 'surround', 'nasal'],
                                  r_max=self.max_dist, shade_off_retina=True)

                ax_ebc_d.pcolormesh(theta_edges, r_edges, ebc_d_rm.T,
                                    cmap=cmap, shading='auto',
                                    vmin=0, vmax=vmax)
                ax_ebc_d.set_title(
                    f'EBC (dark) — {"RELIABLE" if ebc_d_ok else "not reliable"}\n'
                    f'MRL={ck_ebc_d["mean_resultant_length"]:.3f}  '
                    f'RF_CC={ck_ebc_d.get("rf_corr", float("nan")):.3f}\n'
                    f'{_ld_str(ld["ebc"])}',
                    color=_col(ebc_d_ok, '#1a7f37'), fontsize=9, pad=12)
                _polar_axes_style(ax_ebc_d,
                                  labels=['fwd', 'right', 'bkwd', 'left'],
                                  r_max=self.max_dist)

                ax_rbc_d.pcolormesh(theta_edges, r_edges, rbc_d_rm.T,
                                    cmap=cmap, shading='auto',
                                    vmin=0, vmax=vmax)
                ax_rbc_d.set_title(
                    f'RBC (dark) — {"RELIABLE" if rbc_d_ok else "not reliable"}\n'
                    f'MRL={ck_rbc_d["mean_resultant_length"]:.3f}  '
                    f'RF_CC={ck_rbc_d.get("rf_corr", float("nan")):.3f}\n'
                    f'{_ld_str(ld["rbc"])}',
                    color=_col(rbc_d_ok, '#1a5fa8'), fontsize=9, pad=12)
                _polar_axes_style(ax_rbc_d,
                                  labels=['center', 'temporal', 'surround', 'nasal'],
                                  r_max=self.max_dist, shade_off_retina=True)

                fig.colorbar(im, cax=cax, label='Rate (a.u.)')

                _add_shuffle_inset(fig, ck_ebc,   ebc_ok,   color=_col(ebc_ok,   '#1a7f37'),
                                   rect=[0.04, 0.01, 0.16, 0.07])
                _add_shuffle_inset(fig, ck_rbc,   rbc_ok,   color=_col(rbc_ok,   '#1a5fa8'),
                                   rect=[0.27, 0.01, 0.16, 0.07])
                _add_shuffle_inset(fig, ck_ebc_d, ebc_d_ok, color=_col(ebc_d_ok, '#1a7f37'),
                                   rect=[0.51, 0.01, 0.16, 0.07])
                _add_shuffle_inset(fig, ck_rbc_d, rbc_d_ok, color=_col(rbc_d_ok, '#1a5fa8'),
                                   rect=[0.74, 0.01, 0.16, 0.07])

                fr_parts = []
                if self.is_fully_reliable_EBC[c]: fr_parts.append('EBC')
                if self.is_fully_reliable_RBC[c]: fr_parts.append('RBC')
                fig.suptitle(
                    f'Cell {c:03d}   FULLY RELIABLE: {", ".join(fr_parts)}\n'
                    f'(light EBC={ebc_ok}, RBC={rbc_ok} | '
                    f'dark EBC={ebc_d_ok}, RBC={rbc_d_ok})',
                    fontsize=10)

                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)


    def make_diagnostic_figs(self, savedir):

        assert self.ebc_results is not None and self.rbc_results is not None, \
            "Run identify_responses_both() before make_diagnostic_figs()."

        os.makedirs(savedir, exist_ok=True)
        cmap = make_parula()

        theta_edges = np.deg2rad(
            np.arange(0, 360 + self.ray_width, self.ray_width))
        r_edges = self.dist_bin_edges
        N_cells = self.ebc_results['rate_maps'].shape[0]

        _diag_nan_crit = {
            'mean_resultant_length': np.nan,
            'mrl_99_pctl': np.nan,
            'shuffled_mrls': np.array([]),
            'rf_corr': np.nan,
            'rf_corr_pass': 0,
            'rf_corr_shfl_99pct': np.nan,
            'corr_coeff': np.nan,
            'corr_pass': 0,
        }

        def _dc(res, c):

            return res['criteria'].get('cell_{:03d}'.format(c), _diag_nan_crit)

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for ax, res, lbl, col in zip(
                axs,
                [self.ebc_results, self.rbc_results],
                ['EBC (head dir.)', 'RBC (gaze dir.)'],
                ['#1a7f37', '#1a5fa8']):

            real_mrls   = np.array([
                _dc(res, c)['mean_resultant_length']
                for c in range(N_cells)])
            all_shfl    = np.concatenate([
                _dc(res, c)['shuffled_mrls']
                for c in range(N_cells)])
            thresholds  = np.array([
                _dc(res, c)['mrl_99_pctl']
                for c in range(N_cells)])

            real_mrls = real_mrls[np.isfinite(real_mrls)]
            all_shfl  = all_shfl[np.isfinite(all_shfl)]
            thresholds = thresholds[np.isfinite(thresholds)]

            if len(real_mrls) == 0 or len(all_shfl) == 0:
                ax.set_title(lbl + '\n(no valid data)')
                ax.text(0.5, 0.5, 'no valid data', transform=ax.transAxes,
                        ha='center', va='center', color='gray')
            else:
                bins = np.linspace(0, max(real_mrls.max(), np.percentile(all_shfl, 99.9)) * 1.05, 40)
                ax.hist(all_shfl, bins=bins, color='lightgray', label='shuffled null',
                        density=True, alpha=0.8)
                ax.hist(real_mrls, bins=bins, color=col, label='real MRL',
                        alpha=0.6, density=True)
                if len(thresholds):
                    ax.axvline(np.mean(thresholds), color='k', ls='--',
                               label=f'mean 99th pctil ({np.mean(thresholds):.3f})')
                ax.legend(fontsize=7)
            ax.set_xlabel('MRL'); ax.set_ylabel('Density')
            ax.set_title(lbl)
        fig.tight_layout()
        fig.savefig(os.path.join(savedir, '01_mrl_distributions.pdf'))
        plt.close(fig)


        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for ax, res, lbl, col in zip(
                axs,
                [self.ebc_results, self.rbc_results],
                ['EBC (head dir.)', 'RBC (gaze dir.)'],
                ['#1a7f37', '#1a5fa8']):
            rf_ccs = np.array([
                _dc(res, c).get('rf_corr', np.nan)
                for c in range(N_cells)], dtype=float)
            rf_pass_flags = np.array([
                _dc(res, c).get('rf_corr_pass', 0)
                for c in range(N_cells)], dtype=bool)
            shfl_threshs = np.array([
                _dc(res, c).get('rf_corr_shfl_99pct', np.nan)
                for c in range(N_cells)], dtype=float)
            rf_ccs_valid = rf_ccs[np.isfinite(rf_ccs)]
            if len(rf_ccs_valid):
                bins = np.linspace(np.nanmin(rf_ccs_valid) - 0.05,
                                   max(1.0, np.nanmax(rf_ccs_valid)) + 0.05, 35)
                ax.hist(rf_ccs_valid, bins=bins, color=col, alpha=0.7,
                        label='RF corr (block-shuffle halves)')
                ax.axvline(0.6, color='k', ls='--', lw=1, label='threshold (0.6)')
                finite_thresh = shfl_threshs[np.isfinite(shfl_threshs)]
                if len(finite_thresh):
                    ax.axvline(np.mean(finite_thresh), color='gray', ls=':',
                               lw=1, label=f'mean shuffle 99th ({np.mean(finite_thresh):.3f})')
            else:
                ax.text(0.5, 0.5, 'no valid data', transform=ax.transAxes,
                        ha='center', va='center', color='gray')
            ax.set_xlabel('RF split-half correlation (corr2_coeff, unsmoothed)')
            ax.set_ylabel('Count')
            ax.set_title(lbl + f'\n{int(np.sum(rf_pass_flags))}/{N_cells} pass '
                         f'(rf_corr > 0.6)')
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(savedir, '02_split_half_corr.pdf'))
        plt.close(fig)


        fig, axs = plt.subplots(1, 2, figsize=(10, 5),
                                subplot_kw={'projection': 'polar'})
        for ax, res, lbl, polar_labels, is_rbc in zip(
                axs,
                [self.ebc_results, self.rbc_results],
                ['EBC occupancy\n(head dir.)', 'RBC occupancy\n(gaze dir.)'],
                [['fwd', 'right', 'bkwd', 'left'],
                 ['center', 'temporal', 'surround', 'nasal']],
                [False, True]):
            occ = res['occupancy']
            ax.pcolormesh(theta_edges, r_edges, occ.T,
                          cmap='hot', shading='auto')
            _polar_axes_style(ax, labels=polar_labels,
                              r_max=self.max_dist, shade_off_retina=is_rbc)
            ax.set_title(lbl, fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(savedir, '03_occupancy_maps.pdf'))
        plt.close(fig)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5),
                                subplot_kw={'projection': 'polar'})
        for ax, res, lbl, polar_labels, is_rbc in zip(
                axs,
                [self.ebc_results, self.rbc_results],
                ['Mean EBC rate map', 'Mean RBC rate map'],
                [['fwd', 'right', 'bkwd', 'left'],
                 ['center', 'temporal', 'surround', 'nasal']],
                [False, True]):
            mean_rm = np.mean(res['smoothed_rate_maps'], axis=0)
            ax.pcolormesh(theta_edges, r_edges, mean_rm.T,
                          cmap=cmap, shading='auto')
            _polar_axes_style(ax, labels=polar_labels,
                              r_max=self.max_dist, shade_off_retina=is_rbc)
            ax.set_title(lbl, fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(savedir, '04_population_ratemaps.pdf'))
        plt.close(fig)

        ebc_mrls = np.array([
            _dc(self.ebc_results, c)['mean_resultant_length']
            for c in range(N_cells)])
        rbc_mrls = np.array([
            _dc(self.rbc_results, c)['mean_resultant_length']
            for c in range(N_cells)])
        ebc_thresh = np.array([
            _dc(self.ebc_results, c)['mrl_99_pctl']
            for c in range(N_cells)])
        rbc_thresh = np.array([
            _dc(self.rbc_results, c)['mrl_99_pctl']
            for c in range(N_cells)])

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        colors = np.array(['#888888'] * N_cells)
        colors[self.is_EBC & ~self.is_RBC] = '#1a7f37'
        colors[~self.is_EBC & self.is_RBC] = '#1a5fa8'
        colors[self.is_EBC & self.is_RBC]  = '#9b2dca'

        ax.scatter(ebc_mrls, rbc_mrls, c=colors, s=30, alpha=0.7, zorder=3)
        ax.axvline(np.mean(ebc_thresh), color='#1a7f37', ls='--', lw=1,
                   label='EBC 99th pctil threshold')
        ax.axhline(np.mean(rbc_thresh), color='#1a5fa8', ls='--', lw=1,
                   label='RBC 99th pctil threshold')
        ax.plot([0,0.2],[0,0.2], color='tab:red', ls='--', lw=1)

        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker='o', ls='', color='#1a7f37', label='EBC only'),
            Line2D([0], [0], marker='o', ls='', color='#1a5fa8', label='RBC only'),
            Line2D([0], [0], marker='o', ls='', color='#9b2dca', label='Both'),
            Line2D([0], [0], marker='o', ls='', color='#888888', label='Neither'),
        ]
        ax.legend(handles=handles, fontsize=8)
        ax.set_xlabel('EBC MRL (head direction)')
        ax.set_ylabel('RBC MRL (gaze direction)')
        ax.set_title(f'EBC vs RBC reliability\n'
                     f'EBC={np.sum(self.is_EBC)}, RBC={np.sum(self.is_RBC)}, '
                     f'Both={np.sum(self.is_EBC & self.is_RBC)}')
        fig.tight_layout()
        fig.savefig(os.path.join(savedir, '05_cell_scatter.pdf'))
        plt.close(fig)

        print(f'  Diagnostic figures saved to {savedir}/')

    def make_detailed_pdf(self, savepath_ebc, savepath_rbc):

        assert self.ebc_results is not None and self.rbc_results is not None, \
            "Run identify_responses_both() before make_detailed_pdf()."

        N_cells = len(self.is_EBC)

        _dd_nan = {'mean_resultant_length': np.nan, 'rf_corr': np.nan, 'corr_coeff': np.nan}

        def _dd(results, c):
            return results['criteria'].get(f'cell_{c:03d}', _dd_nan)

        def _get_metrics(results):
            mrls = np.array([_dd(results, c)['mean_resultant_length'] for c in range(N_cells)])
            ccs  = np.array([_dd(results, c).get('rf_corr',
                             _dd(results, c).get('corr_coeff', np.nan))
                             for c in range(N_cells)], dtype=float)
            return mrls, ccs

        def _write_pdf(target_indices, results, label, filename,
                       axis_labels, shade_off_retina=False):
            if len(target_indices) == 0:
                print(f"No reliable {label} cells found. Skipping {filename}.")
                return

            ebc_mrls, ebc_ccs = _get_metrics(self.ebc_results)
            rbc_mrls, rbc_ccs = _get_metrics(self.rbc_results)
            sort_metric = rbc_mrls if label == 'RBC' else ebc_mrls

            sorted_indices = target_indices[np.argsort(sort_metric[target_indices])[::-1]]
            
            theta_edges = np.deg2rad(np.arange(0, 360 + self.ray_width, self.ray_width))
            r_edges = self.dist_bin_edges
            cmap = make_parula()

            print(f"Writing {label} PDF to {filename}...")
            with PdfPages(filename) as pdf:
                for c in sorted_indices:
                    fig = plt.figure(figsize=(14, 8))
                    
                    ax_full = fig.add_axes([0.05, 0.55, 0.25, 0.4], projection='polar')
                    ax_s1   = fig.add_axes([0.35, 0.55, 0.25, 0.4], projection='polar')
                    ax_s2   = fig.add_axes([0.65, 0.55, 0.25, 0.4], projection='polar')
                    
                    ax_sc1 = fig.add_axes([0.05, 0.1, 0.25, 0.35])
                    ax_sc2 = fig.add_axes([0.35, 0.1, 0.25, 0.35])
                    ax_sc3 = fig.add_axes([0.65, 0.1, 0.25, 0.35])
                    
                    full_rm = results['smoothed_rate_maps'][c]
                    crit_c = _dd(results, c)

                    s1_rm = crit_c.get('rf_split_map_1',
                                       crit_c.get('split_rate_map_1'))
                    s2_rm = crit_c.get('rf_split_map_2',
                                       crit_c.get('split_rate_map_2'))
                    
                    vmax = np.nanmax([full_rm, s1_rm, s2_rm])
                    if vmax == 0: vmax = 1.0
                    
                    for ax, rm, title in zip([ax_full, ax_s1, ax_s2], [full_rm, s1_rm, s2_rm], ['Full', 'Split 1', 'Split 2']):
                        ax.pcolormesh(theta_edges, r_edges, rm.T, cmap=cmap, shading='auto', vmin=0, vmax=vmax)
                        _polar_axes_style(ax, labels=axis_labels,
                                          r_max=self.max_dist,
                                          shade_off_retina=shade_off_retina)
                        ax.set_title(title, fontsize=10)

                    ax_sc1.scatter(ebc_mrls, rbc_mrls, c='gray', s=10, alpha=0.5)
                    ax_sc1.scatter(ebc_mrls[c], rbc_mrls[c], c='red', s=50, marker='*', zorder=10)
                    ax_sc1.set_xlabel('EBC MRL')
                    ax_sc1.set_ylabel('RBC MRL')
                    ax_sc1.set_title('EBC vs RBC MRL')
                    
                    ax_sc2.scatter(ebc_ccs, ebc_mrls, c='gray', s=10, alpha=0.5)
                    ax_sc2.scatter(ebc_ccs[c], ebc_mrls[c], c='red', s=50, marker='*', zorder=10)
                    ax_sc2.set_xlabel('EBC RF corr (block-shuffle)')
                    ax_sc2.set_ylabel('EBC MRL')
                    ax_sc2.set_title('EBC Reliability')
                    ax_sc2.axvline(0.6, color='k', ls='--', lw=0.8)

                    ax_sc3.scatter(rbc_ccs, rbc_mrls, c='gray', s=10, alpha=0.5)
                    ax_sc3.scatter(rbc_ccs[c], rbc_mrls[c], c='red', s=50, marker='*', zorder=10)
                    ax_sc3.set_xlabel('RBC RF corr (block-shuffle)')
                    ax_sc3.set_ylabel('RBC MRL')
                    ax_sc3.set_title('RBC Reliability')
                    ax_sc3.axvline(0.6, color='k', ls='--', lw=0.8)
                    
                    fig.suptitle(f'Cell {c} ({label}) - MRL={sort_metric[c]:.3f}', fontsize=16)
                    
                    pdf.savefig(fig)
                    plt.close(fig)

        ebc_indices = np.where(self.is_EBC)[0]
        _write_pdf(ebc_indices, self.ebc_results, 'EBC', savepath_ebc,
                   axis_labels=['fwd', 'right', 'bkwd', 'left'],
                   shade_off_retina=False)

        rbc_indices = np.where(self.is_RBC)[0]
        _write_pdf(rbc_indices, self.rbc_results, 'RBC', savepath_rbc,
                   axis_labels=['center', 'temporal', 'surround', 'nasal'],
                   shade_off_retina=True)


    def load_results(self, results_path):

        print(f"Loading results from {results_path}...")
        res = read_h5(results_path)


        if 'params' in res:
            params = res['params']
            self.ray_width = params['ray_width']
            self.max_dist = params['max_dist']
            self.dist_bin_size = params['dist_bin_size']
            self.dist_bin_edges = params['dist_bin_edges']
            self.dist_bin_cents = params['dist_bin_cents']
        
        self.ebc_results = res.get('ebc')
        self.rbc_results = res.get('rbc')
        
        if 'classification' in res:
            cls = res['classification']
            self.is_EBC = cls['is_EBC'].astype(bool)
            self.is_RBC = cls['is_RBC'].astype(bool)
        
        print("Results loaded.")

    def save_results(self, savepath):

        write_h5(savepath, convert_bools_to_ints(self.data_out))

    def save_results_combined(self, savepath):

        assert self.ebc_results is not None and self.rbc_results is not None, \
            "Run identify_responses_both() before save_results_combined()."

        def _criteria_flat(crit_dict):
            out = {}
            for key, val in crit_dict.items():
                out[key] = {k: v for k, v in val.items()
                            if not isinstance(v, np.ndarray) or v.ndim <= 2}
            return out

        def _pipeline_dict(res, bc_key):
            return {
                'rate_maps':          res['rate_maps'],
                'smoothed_rate_maps': res['smoothed_rate_maps'],
                'occupancy':          res['occupancy'],
                'ray_distances':      res['ray_distances'],
                'is_inverse':         res['is_inverse'],
                bc_key:               res['is_bc'],
                'criteria':           _criteria_flat(res['criteria']),
            }

        data_out = {
            'params': {
                'ray_width':      self.ray_width,
                'max_dist':       self.max_dist,
                'dist_bin_size':  self.dist_bin_size,
                'dist_bin_edges': self.dist_bin_edges,
                'dist_bin_cents': self.dist_bin_cents,
                'angle_rad':      np.deg2rad(np.arange(0, 360, self.ray_width)),
            },
            'ebc':      _pipeline_dict(self.ebc_results, 'is_EBC'),
            'rbc':      _pipeline_dict(self.rbc_results, 'is_RBC'),
            'classification': {
                'is_EBC':              self.is_EBC.astype(int),
                'is_RBC':              self.is_RBC.astype(int),
                'is_EBC_dark':         self.is_EBC_dark.astype(int),
                'is_RBC_dark':         self.is_RBC_dark.astype(int),
                'is_fully_reliable_EBC': self.is_fully_reliable_EBC.astype(int),
                'is_fully_reliable_RBC': self.is_fully_reliable_RBC.astype(int),
                'is_either': (self.is_EBC | self.is_RBC).astype(int),
                'is_both':   (self.is_EBC & self.is_RBC).astype(int),
            },
        }

        if self.ebc_dark_results is not None:
            data_out['ebc_dark'] = _pipeline_dict(self.ebc_dark_results, 'is_EBC_dark')
            data_out['rbc_dark'] = _pipeline_dict(self.rbc_dark_results, 'is_RBC_dark')

        write_h5(savepath, convert_bools_to_ints(data_out))
        print(f'  Results saved -> {savepath}')


def _polar_axes_style(ax, labels=None, r_max=26, shade_off_retina=False):

    ax.set_theta_zero_location('E' if shade_off_retina else 'N')
    ax.set_theta_direction(-1)

    ax.set_yticks([r_max * 0.5, r_max])
    ax.set_yticklabels([f'{r_max * 0.5:.0f}', f'{r_max:.0f} cm'], fontsize=6)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    if labels is not None:
        ax.set_xticklabels(labels, fontsize=8)
    ax.tick_params(axis='both', labelsize=7)

    if shade_off_retina:

        off_theta = np.deg2rad(np.linspace(120, 240, 200))
        ax.fill_between(off_theta, 0, r_max, color='gray', alpha=0.4, zorder=3,
                        linewidth=0)

        for lbl in ax.get_xticklabels():
            if lbl.get_text() == 'temporal':
                lbl.set_ha('left')


def _add_shuffle_inset(fig, criteria, passes, color, rect):

    ax = fig.add_axes(rect)
    shfl = criteria.get('shuffled_mrls', np.array([]))
    mrl  = criteria.get('mean_resultant_length', 0.)
    thr  = criteria.get('mrl_99_pctl', 0.)

    if len(shfl) > 0:
        bins = np.linspace(shfl.min() * 0.9, max(shfl.max(), mrl) * 1.1, 20)
        ax.hist(shfl, bins=bins, color='lightgray', density=True)
        ax.axvline(thr, color='k', lw=0.8, ls='--')
        ax.axvline(mrl, color=color, lw=1.5,
                   label=f'MRL={mrl:.3f}')
    ax.set_xlabel('MRL', fontsize=5)
    ax.set_title('shuffle', fontsize=5)
    ax.tick_params(labelsize=4)
    ax.set_yticks([])


def boundary_tuning(path_in):

    if path_in is None:
        parser = argparse.ArgumentParser(description='Run boundary tuning analysis.')
        parser.add_argument('--path', type=str, help='Path to preprocessed .h5 file OR results .h5 file if --pdf_only is used')
        parser.add_argument('--pdf_only', action='store_true', help='Generate PDFs from existing results file')
        args = parser.parse_args()
        path = args.path
        pdf_only = args.pdf_only
    else:
        path = path_in
        pdf_only = False

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    version_key = 'v03'

    if pdf_only:
        print(f'Loading results from {path}...')
        bt = BoundaryTuning({}) 
        bt.load_results(args.path)
        
        base_path = os.path.splitext(path)[0]

        if base_path.endswith('_boundary_results'):
            base_path = base_path[:-17]


        bt.make_detailed_pdf(f"{base_path}_EBC_detailed_{version_key}.pdf", f"{base_path}_RBC_detailed_{version_key}.pdf")
        
    else:
        
        print(f'Loading preprocessed data from {path}...')
        data = read_h5(path)
        bt = BoundaryTuning(data)
        bt.identify_responses_both()

        base_path = os.path.splitext(path)[0]
        bt.save_results_combined(f"{base_path}_boundary_results_{version_key}.h5")
        bt.make_summary_pdf(f"{base_path}_boundary_summary_{version_key}.pdf")
        bt.make_fully_reliable_pdf(f"{base_path}_boundary_fully_reliable_{version_key}.pdf")
        bt.make_diagnostic_figs(f"{base_path}_boundary_diagnostics_{version_key}")
        bt.make_detailed_pdf(f"{base_path}_EBC_detailed_{version_key}.pdf",
                             f"{base_path}_RBC_detailed_{version_key}.pdf")


def _boundary_tuning_worker(f):
    try:
        boundary_tuning(f)
    except Exception as e:
        print('Error processing {}.... {}'.format(f, e))


if __name__ == '__main__':

    files = find('*DMM*fm*preproc.h5', '/home/dylan/Storage/freely_moving_data/_V1PPC')

    for i, f in enumerate(files):
        print('   => Processing file {} of {}'.format(i, len(files)))
        boundary_tuning(f)

    print('  => Done')
