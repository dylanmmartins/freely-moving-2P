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


def rate_map_mp(spike_rate, occupancy, ray_distances, ray_width, dist_bin_edges, dist_bin_size):
    
    N_angular_bins = int(360 / ray_width)
    N_distance_bins = len(dist_bin_edges) - 1

    rate_map = np.zeros((N_angular_bins, N_distance_bins))

    for a in range(N_angular_bins):
        dists = ray_distances[:, a]
        valid = ~np.isnan(dists)
        bin_inds = np.digitize(dists[valid], dist_bin_edges) - 1
        inrange = (bin_inds >= 0) & (bin_inds < N_distance_bins)
        np.add.at(rate_map[a], bin_inds[inrange], spike_rate[valid][inrange])

    rate_map /= occupancy + 1e-6
    return rate_map


def calc_MRL_mp(ratemap, ray_width, dist_bin_cents):

    angs_rad = np.deg2rad(np.arange(0, 360, ray_width))
    angs_mesh, _ = np.meshgrid(angs_rad, dist_bin_cents, indexing='ij')

    total_weight = np.nansum(ratemap)
    if total_weight < 1e-10:
        return 0.0
    mr = np.nansum(ratemap * np.exp(1j * angs_mesh)) / total_weight
    return float(np.abs(mr))


def calc_shfl_mean_resultant_mp(spikes, useinds, occupancy, ray_distances, ray_width,
                                dist_bin_edges, dist_bin_size, dist_bin_cents, is_inverse):

    N_frames = int(np.sum(useinds))
    shift_amount = np.random.randint(1, max(N_frames, 2))
    shifted_spikes = np.roll(spikes[useinds], shift_amount)

    shifted_ratemap = rate_map_mp(shifted_spikes, occupancy, ray_distances,
                                  ray_width, dist_bin_edges, dist_bin_size)
    if is_inverse:
        shifted_ratemap = np.max(shifted_ratemap) - shifted_ratemap + np.min(shifted_ratemap)

    return calc_MRL_mp(shifted_ratemap, ray_width, dist_bin_cents)


class BoundaryTuning:

    def __init__(self, preprocessed_data):
        self.data = preprocessed_data

        self.ray_width    = 3    # deg per angular bin
        self.max_dist     = 26   # cm
        self.dist_bin_size = 2.  # cm

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

        p2c = self.data['pxls2cm']
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

        N_frames  = len(use_inds)

        self._ray_dist_use_inds = use_inds.copy()

        x_trace = x_full[use_inds]
        y_trace = y_full[use_inds]
        ang_rad  = np.deg2rad(angle_trace_deg[use_inds])

        BL = (self.data['arenaBL']['x'] / p2c, self.data['arenaBL']['y'] / p2c)
        BR = (self.data['arenaBR']['x'] / p2c, self.data['arenaBR']['y'] / p2c)
        TR = (self.data['arenaTR']['x'] / p2c, self.data['arenaTR']['y'] / p2c)
        TL = (self.data['arenaTL']['x'] / p2c, self.data['arenaTL']['y'] / p2c)

        walls = [
            np.array([[BL[0], BL[1]], [BR[0], BR[1]]]),  # bottom
            np.array([[BR[0], BR[1]], [TR[0], TR[1]]]),  # right
            np.array([[TR[0], TR[1]], [TL[0], TL[1]]]),  # top
            np.array([[TL[0], TL[1]], [BL[0], BL[1]]]),  # left
        ]

        ray_offsets_rad = np.deg2rad(np.arange(0, 360, self.ray_width))
        N_ang = len(ray_offsets_rad)

        self.dist_bin_edges = np.arange(0, self.max_dist + self.dist_bin_size,
                                        self.dist_bin_size)
        self.dist_bin_cents = self.dist_bin_edges[:-1] + self.dist_bin_size / 2

        ray_distances = np.full((N_frames, N_ang), np.nan)

        for fr in tqdm(range(N_frames), leave=False, desc='    ray casting'):
            px, py = x_trace[fr], y_trace[fr]
            base_ang = ang_rad[fr]

            for ri, off in enumerate(ray_offsets_rad):
                ray_ang = base_ang + off
                rv = np.array([np.cos(ray_ang), np.sin(ray_ang)]) # unit vector

                best = np.inf
                for wall in walls:
                    start  = wall[0]
                    vec    = wall[1] - wall[0]
                    rel    = np.array([px, py]) - start

                    det = np.cross(vec, rv)
                    if det == 0:
                        continue

                    t = np.cross(rel, rv) / det
                    if t < 0 or t > 1:
                        continue

                    isect = start + t * vec
                    to_isect = isect - np.array([px, py])

                    if np.dot(to_isect, rv) < 0:
                        continue

                    dist = np.linalg.norm(to_isect)
                    if dist < best:
                        best = dist

                if best < np.inf:
                    ray_distances[fr, ri] = best

        return ray_distances


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

        N_frames_rd = ray_distances.shape[0]

        use_inds_clipped = self._ray_dist_use_inds[:N_frames_rd]
        max_sp = self.data['norm_spikes'].shape[1]
        use_inds_clipped = use_inds_clipped[use_inds_clipped < max_sp]

        spikes_used = self.data['norm_spikes'][:, use_inds_clipped]  # (N_cells, N_frames)

        N_cells = spikes_used.shape[0]
        N_ang   = int(360 / self.ray_width)
        N_dist  = len(self.dist_bin_edges) - 1

        rate_maps = np.zeros((N_cells, N_ang, N_dist))

        for a in tqdm(range(N_ang), leave=False, desc='    rate maps'):
            dists = ray_distances[:, a]
            valid = ~np.isnan(dists)
            bin_inds = np.digitize(dists[valid], self.dist_bin_edges) - 1
            inrange  = (bin_inds >= 0) & (bin_inds < N_dist)

            valid_frames   = np.where(valid)[0][inrange]
            valid_bin_inds = bin_inds[inrange]

            for c in range(N_cells):
                np.add.at(rate_maps[c, a], valid_bin_inds,
                          spikes_used[c, valid_frames])

        min_occ = 8
        occ_mask = occupancy < min_occ
        for c in range(N_cells):
            rm = rate_maps[c] / (occupancy + 1e-6)
            rm[occ_mask] = np.nan
            rate_maps[c] = rm

        return rate_maps

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
        # pbar.close()
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

    def _smooth_rate_maps_arr(self, rate_maps):

        smoothed = np.zeros_like(rate_maps)
        for c in range(rate_maps.shape[0]):
            smoothed[c] = self._smooth_single(rate_maps[c])
        return smoothed

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
        """Backward-compatible inverse-response classifier."""
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
            """Return (preferred_angle_deg, preferred_dist_cm) for a 2D rate map."""
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

        occupancy = self._compute_occupancy_from_raydists(ray_distances)

        rate_maps = self._compute_rate_maps_from_raydists(ray_distances, occupancy)

        smoothed = self._smooth_rate_maps_arr(rate_maps)

        is_inverse, inv_crit = self._identify_inverse_responses_from(rate_maps)

        N_cells = rate_maps.shape[0]
        is_bc   = np.zeros(N_cells, dtype=bool)
        cell_criteria = {}

        for c in range(N_cells):
            rm = rate_maps[c].copy()
            if is_inverse[c]:
                rm = self._invert_ratemap(rm)

            _, mrl, mra = self._calc_mean_resultant(rm)

            corr, corr_pass, rm1_s, rm2_s = self._calc_correlation_across_split_v2(
                c, ray_distances, ncnk=n_chunks, corr_thresh=corr_thresh)

            mrl_thresh, mrl_pass, shfl_mrls = self._test_mrl_against_shuffles(
                c, mrl, ray_distances, occupancy, bool(is_inverse[c]),
                n_shfl=n_shuffles)

            if corr_pass and mrl_pass:
                is_bc[c] = True

            cell_criteria['cell_{:03d}'.format(c)] = {
                **inv_crit[c],
                'mean_resultant_length': float(mrl),
                'mean_resultant_angle':  float(mra),
                'corr_coeff':            float(corr),
                'corr_pass':             int(corr_pass),
                'mrl_99_pctl':           float(mrl_thresh),
                'mrl_pass':              int(mrl_pass),
                'shuffled_mrls':         shfl_mrls,
                'split_rate_map_1':      rm1_s,
                'split_rate_map_2':      rm2_s,
            }

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
        """Compare light vs dark peak location for cell c.

        Uses the same criterion as split-half: angle diff < 45° AND
        fractional distance diff < 50%.  Returns a dict with keys 'ebc'
        and 'rbc', each containing {'passes', 'diff_ang', 'diff_dist'}.
        """
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
        """Run EBC and RBC pipelines separately for light and dark periods.

        Light frames (ltdk_state_vec == 1) are used for the primary results
        stored in ebc_results / rbc_results.  Dark frames (ltdk_state_vec == 0)
        are analysed in parallel and stored in ebc_dark_results / rbc_dark_results.
        Cells are classified as "fully reliable" when they pass both light and
        dark reliability criteria for a given coordinate system.
        """
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

        # ── Light period ──────────────────────────────────────────────────────
        print('  Running light-period EBC/RBC pipelines...')
        self.useinds = light_mask & (speed > 5.)
        self.ebc_results = self._run_angle_pipeline(
            'head', n_chunks, n_shuffles, corr_thresh)
        self.rbc_results = self._run_angle_pipeline(
            'gaze', n_chunks, n_shuffles, corr_thresh)
        self.is_EBC = self.ebc_results['is_bc'].astype(bool)
        self.is_RBC = self.rbc_results['is_bc'].astype(bool)

        # ── Dark period ───────────────────────────────────────────────────────
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

        # ── Full reliability: must pass in BOTH light and dark ─────────────
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
        """Summary PDF with four polar plots per cell (EBC/RBC × light/dark).

        Top row: light-period EBC and RBC maps.
        Bottom row: dark-period EBC and RBC maps (or a 'No dark data' label).
        Shuffle-MRL insets are shown below each column.
        The title includes light-vs-dark peak-stability results.
        """
        assert self.ebc_results is not None and self.rbc_results is not None, \
            "Run identify_responses_both() before make_summary_pdf()."

        cmap        = make_parula()
        theta_edges = np.deg2rad(np.arange(0, 360 + self.ray_width, self.ray_width))
        r_edges     = self.dist_bin_edges
        has_dark    = self.ebc_dark_results is not None

        show_cells = np.where(self.is_EBC | self.is_RBC)[0]
        if len(show_cells) == 0:
            print('  No reliable EBC or RBC cells found — no PDF generated.')
            return

        with PdfPages(savepath) as pdf:
            for c in show_cells:
                fig = plt.figure(figsize=(13, 11))

                # ── Axes layout ──────────────────────────────────────────────
                ax_ebc_l = fig.add_axes([0.05, 0.56, 0.38, 0.38], projection='polar')
                ax_rbc_l = fig.add_axes([0.52, 0.56, 0.38, 0.38], projection='polar')
                ax_ebc_d = fig.add_axes([0.05, 0.12, 0.38, 0.38], projection='polar')
                ax_rbc_d = fig.add_axes([0.52, 0.12, 0.38, 0.38], projection='polar')
                cax      = fig.add_axes([0.93, 0.20, 0.012, 0.65])

                # ── Light-period data ────────────────────────────────────────
                ebc_rm  = self.ebc_results['smoothed_rate_maps'][c]
                rbc_rm  = self.rbc_results['smoothed_rate_maps'][c]
                ck_ebc  = self.ebc_results['criteria']['cell_{:03d}'.format(c)]
                ck_rbc  = self.rbc_results['criteria']['cell_{:03d}'.format(c)]
                ebc_ok  = bool(self.is_EBC[c])
                rbc_ok  = bool(self.is_RBC[c])
                ebc_col = '#1a7f37' if ebc_ok else '#888888'
                rbc_col = '#1a5fa8' if rbc_ok else '#888888'

                # Shared colour scale across all four panels
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
                    f'CC={ck_ebc["corr_coeff"]:.3f}',
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
                    f'CC={ck_rbc["corr_coeff"]:.3f}',
                    color=rbc_col, fontsize=9, pad=12)
                _polar_axes_style(ax_rbc_l,
                                  labels=['center', 'temporal', 'surround', 'nasal'],
                                  r_max=self.max_dist, shade_off_retina=True)

                # ── Dark-period data ─────────────────────────────────────────
                if has_dark:
                    ebc_d_rm  = self.ebc_dark_results['smoothed_rate_maps'][c]
                    rbc_d_rm  = self.rbc_dark_results['smoothed_rate_maps'][c]
                    ck_ebc_d  = self.ebc_dark_results['criteria']['cell_{:03d}'.format(c)]
                    ck_rbc_d  = self.rbc_dark_results['criteria']['cell_{:03d}'.format(c)]
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
                        f'CC={ck_ebc_d["corr_coeff"]:.3f}\n'
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
                        f'CC={ck_rbc_d["corr_coeff"]:.3f}\n'
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
        """PDF restricted to cells reliable in BOTH light and dark.

        A cell appears here if it passes the reliability criteria in both the
        light and dark period for at least one coordinate system (EBC or RBC).
        Each page shows the same four-panel layout as make_summary_pdf, with
        the light-vs-dark peak-stability result annotated on the dark panels.
        """
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

        print(f'  Writing fully-reliable PDF ({len(show_cells)} cells) -> {savepath}')
        with PdfPages(savepath) as pdf:
            for c in show_cells:
                fig = plt.figure(figsize=(13, 11))

                ax_ebc_l = fig.add_axes([0.05, 0.56, 0.38, 0.38], projection='polar')
                ax_rbc_l = fig.add_axes([0.52, 0.56, 0.38, 0.38], projection='polar')
                ax_ebc_d = fig.add_axes([0.05, 0.12, 0.38, 0.38], projection='polar')
                ax_rbc_d = fig.add_axes([0.52, 0.12, 0.38, 0.38], projection='polar')
                cax      = fig.add_axes([0.93, 0.20, 0.012, 0.65])

                ebc_rm   = self.ebc_results['smoothed_rate_maps'][c]
                rbc_rm   = self.rbc_results['smoothed_rate_maps'][c]
                ebc_d_rm = self.ebc_dark_results['smoothed_rate_maps'][c]
                rbc_d_rm = self.rbc_dark_results['smoothed_rate_maps'][c]
                ck_ebc   = self.ebc_results['criteria']['cell_{:03d}'.format(c)]
                ck_rbc   = self.rbc_results['criteria']['cell_{:03d}'.format(c)]
                ck_ebc_d = self.ebc_dark_results['criteria']['cell_{:03d}'.format(c)]
                ck_rbc_d = self.rbc_dark_results['criteria']['cell_{:03d}'.format(c)]
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
                        return 'L–D: n/a'
                    sym = '\u2713' if res['passes'] else '\u2717'
                    return (f'L\u2013D {sym}  \u0394ang={res["diff_ang"]:.0f}\u00b0  '
                            f'\u0394dist={res["diff_dist"]:.0%}')

                # Light EBC
                im = ax_ebc_l.pcolormesh(theta_edges, r_edges, ebc_rm.T,
                                         cmap=cmap, shading='auto',
                                         vmin=0, vmax=vmax)
                ax_ebc_l.set_title(
                    f'EBC (light) — {"RELIABLE" if ebc_ok else "not reliable"}\n'
                    f'MRL={ck_ebc["mean_resultant_length"]:.3f}  '
                    f'CC={ck_ebc["corr_coeff"]:.3f}',
                    color=_col(ebc_ok, '#1a7f37'), fontsize=9, pad=12)
                _polar_axes_style(ax_ebc_l,
                                  labels=['fwd', 'right', 'bkwd', 'left'],
                                  r_max=self.max_dist)

                # Light RBC
                ax_rbc_l.pcolormesh(theta_edges, r_edges, rbc_rm.T,
                                    cmap=cmap, shading='auto',
                                    vmin=0, vmax=vmax)
                ax_rbc_l.set_title(
                    f'RBC (light) — {"RELIABLE" if rbc_ok else "not reliable"}\n'
                    f'MRL={ck_rbc["mean_resultant_length"]:.3f}  '
                    f'CC={ck_rbc["corr_coeff"]:.3f}',
                    color=_col(rbc_ok, '#1a5fa8'), fontsize=9, pad=12)
                _polar_axes_style(ax_rbc_l,
                                  labels=['center', 'temporal', 'surround', 'nasal'],
                                  r_max=self.max_dist, shade_off_retina=True)

                # Dark EBC
                ax_ebc_d.pcolormesh(theta_edges, r_edges, ebc_d_rm.T,
                                    cmap=cmap, shading='auto',
                                    vmin=0, vmax=vmax)
                ax_ebc_d.set_title(
                    f'EBC (dark) — {"RELIABLE" if ebc_d_ok else "not reliable"}\n'
                    f'MRL={ck_ebc_d["mean_resultant_length"]:.3f}  '
                    f'CC={ck_ebc_d["corr_coeff"]:.3f}\n'
                    f'{_ld_str(ld["ebc"])}',
                    color=_col(ebc_d_ok, '#1a7f37'), fontsize=9, pad=12)
                _polar_axes_style(ax_ebc_d,
                                  labels=['fwd', 'right', 'bkwd', 'left'],
                                  r_max=self.max_dist)

                # Dark RBC
                ax_rbc_d.pcolormesh(theta_edges, r_edges, rbc_d_rm.T,
                                    cmap=cmap, shading='auto',
                                    vmin=0, vmax=vmax)
                ax_rbc_d.set_title(
                    f'RBC (dark) — {"RELIABLE" if rbc_d_ok else "not reliable"}\n'
                    f'MRL={ck_rbc_d["mean_resultant_length"]:.3f}  '
                    f'CC={ck_rbc_d["corr_coeff"]:.3f}\n'
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


        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for ax, res, lbl, col in zip(
                axs,
                [self.ebc_results, self.rbc_results],
                ['EBC (head dir.)', 'RBC (gaze dir.)'],
                ['#1a7f37', '#1a5fa8']):

            real_mrls   = np.array([
                res['criteria']['cell_{:03d}'.format(c)]['mean_resultant_length']
                for c in range(N_cells)])
            all_shfl    = np.concatenate([
                res['criteria']['cell_{:03d}'.format(c)]['shuffled_mrls']
                for c in range(N_cells)])
            thresholds  = np.array([
                res['criteria']['cell_{:03d}'.format(c)]['mrl_99_pctl']
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
            ccs = np.array([
                res['criteria']['cell_{:03d}'.format(c)]['corr_coeff']
                for c in range(N_cells)])
            pass_flags = np.array([
                res['criteria']['cell_{:03d}'.format(c)]['corr_pass']
                for c in range(N_cells)], dtype=bool)
            ccs_valid = ccs[np.isfinite(ccs)]
            if len(ccs_valid):
                ax.hist(ccs_valid, bins=30, color=col, alpha=0.7)
            else:
                ax.text(0.5, 0.5, 'no valid data', transform=ax.transAxes,
                        ha='center', va='center', color='gray')
            ax.set_xlabel('Split-half CC (informational)')
            ax.set_ylabel('Count')
            ax.set_title(lbl + f'\n{int(np.sum(pass_flags))}/{N_cells} pass '
                         f'(peak stability: Δang<45°, Δdist<50%)')
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
            self.ebc_results['criteria']['cell_{:03d}'.format(c)]['mean_resultant_length']
            for c in range(N_cells)])
        rbc_mrls = np.array([
            self.rbc_results['criteria']['cell_{:03d}'.format(c)]['mean_resultant_length']
            for c in range(N_cells)])
        ebc_thresh = np.array([
            self.ebc_results['criteria']['cell_{:03d}'.format(c)]['mrl_99_pctl']
            for c in range(N_cells)])
        rbc_thresh = np.array([
            self.rbc_results['criteria']['cell_{:03d}'.format(c)]['mrl_99_pctl']
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

        def _get_metrics(results):
            mrls = np.array([results['criteria'][f'cell_{c:03d}']['mean_resultant_length'] for c in range(N_cells)])
            ccs  = np.array([results['criteria'][f'cell_{c:03d}']['corr_coeff'] for c in range(N_cells)])
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
                    s1_rm = results['criteria'][f'cell_{c:03d}']['split_rate_map_1']
                    s2_rm = results['criteria'][f'cell_{c:03d}']['split_rate_map_2']
                    
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
                    ax_sc2.set_xlabel('EBC Split-Half CC')
                    ax_sc2.set_ylabel('EBC MRL')
                    ax_sc2.set_title('EBC Reliability')
                    
                    ax_sc3.scatter(rbc_ccs, rbc_mrls, c='gray', s=10, alpha=0.5)
                    ax_sc3.scatter(rbc_ccs[c], rbc_mrls[c], c='red', s=50, marker='*', zorder=10)
                    ax_sc3.set_xlabel('RBC Split-Half CC')
                    ax_sc3.set_ylabel('RBC MRL')
                    ax_sc3.set_title('RBC Reliability')
                    
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

    # files = find('*DMM*fm*preproc.h5', '/home/dylan/Storage/freely_moving_data/_V1PPC')
    # # files = all_fm_preproc_files[8:]

    # n_workers = max(1, multiprocessing.cpu_count() - 1)
    # print(f'Processing {len(files)} recordings with {n_workers} workers.')

    # with multiprocessing.Pool(processes=n_workers) as pool:
    #     list(tqdm(pool.imap_unordered(_boundary_tuning_worker, files), total=len(files)))

    boundary_tuning(
        '/home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings/cohort02_recordings/251021_DMM_DMM061_pos04/fm1/251021_DMM_DMM061_fm_01_preproc.h5'
    )