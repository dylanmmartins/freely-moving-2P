# -*- coding: utf-8 -*-
"""
Boundary tuning analysis tools for freely-moving 2P experiments.

Computes egocentric boundary cell (EBC) and retinocentric boundary cell (RBC)
receptive fields for neurons recorded during navigation in a square arena.

EBC reference frame  : allocentric head direction (yaw)
RBC reference frame  : allocentric gaze direction (yaw + theta_eye)

Both use wall-ray-casting: for every frame a full 360-degree fan of rays is cast
from the animal's head position and the distance to the nearest wall is recorded.
Firing rate is accumulated into a 2D (angle × distance) rate map, then tested for
reliability via (a) split-half correlation and (b) spike-shuffle MRL test.

Classes
-------
BoundaryTuning
    Main class for EBC/RBC analysis.

Module-level helpers (kept for backward compatibility with multiprocessing)
---------------------------------------------------------------------------
convert_bools_to_ints(data)
rate_map_mp(...)
calc_MRL_mp(...)
calc_shfl_mean_resultant_mp(...)

Author: DMM, last modified 2025-2026
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

import fm2p


# ---------------------------------------------------------------------------
# Module-level helpers (kept for backward-compat multiprocessing pickling)
# ---------------------------------------------------------------------------

def convert_bools_to_ints(data):
    """Recursively convert all booleans in a dict to ints (for HDF5 saving)."""
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
    """
    Calculate a 2D rate map for a single cell (multiprocessing helper).

    Parameters
    ----------
    spike_rate : np.ndarray, shape (N_frames,)
    occupancy  : np.ndarray, shape (N_ang, N_dist)
    ray_distances : np.ndarray, shape (N_frames, N_ang)
    ray_width  : float  – angular bin width (deg)
    dist_bin_edges : np.ndarray
    dist_bin_size  : float  – cm

    Returns
    -------
    rate_map : np.ndarray, shape (N_ang, N_dist)
    """
    N_angular_bins = int(360 / ray_width)
    N_distance_bins = len(dist_bin_edges) - 1

    rate_map = np.zeros((N_angular_bins, N_distance_bins))

    # Vectorised inner loop: for each angular bin, digitise distances
    for a in range(N_angular_bins):
        dists = ray_distances[:, a]
        valid = ~np.isnan(dists)
        bin_inds = np.digitize(dists[valid], dist_bin_edges) - 1
        inrange = (bin_inds >= 0) & (bin_inds < N_distance_bins)
        np.add.at(rate_map[a], bin_inds[inrange], spike_rate[valid][inrange])

    rate_map /= occupancy + 1e-6
    return rate_map


def calc_MRL_mp(ratemap, ray_width, dist_bin_cents):
    """
    Calculate mean resultant length of a 2D rate map (multiprocessing helper).

    FIX: normalises by np.sum(ratemap) not N_bins so MRL ∈ [0, 1].
    """
    angs_rad = np.deg2rad(np.arange(0, 360, ray_width))
    angs_mesh, _ = np.meshgrid(angs_rad, dist_bin_cents, indexing='ij')

    total_weight = np.nansum(ratemap)
    if total_weight < 1e-10:
        return 0.0
    mr = np.nansum(ratemap * np.exp(1j * angs_mesh)) / total_weight
    return float(np.abs(mr))


def calc_shfl_mean_resultant_mp(spikes, useinds, occupancy, ray_distances, ray_width,
                                dist_bin_edges, dist_bin_size, dist_bin_cents, is_inverse):
    """
    Compute MRL for one circular-shifted spike train (multiprocessing helper).

    Parameters
    ----------
    spikes      : np.ndarray, shape (N_total_frames,)
    useinds     : np.ndarray, boolean mask of usable frames
    occupancy   : np.ndarray, shape (N_ang, N_dist)
    ray_distances : np.ndarray, shape (N_used_frames, N_ang)
    ray_width   : float
    dist_bin_edges : np.ndarray
    dist_bin_size  : float
    dist_bin_cents : np.ndarray
    is_inverse  : bool  - if True, invert the rate map before computing MRL

    Returns
    -------
    shf_mrl : float
    """
    N_frames = int(np.sum(useinds))
    shift_amount = np.random.randint(1, max(N_frames, 2))
    shifted_spikes = np.roll(spikes[useinds], shift_amount)

    shifted_ratemap = rate_map_mp(shifted_spikes, occupancy, ray_distances,
                                  ray_width, dist_bin_edges, dist_bin_size)
    if is_inverse:
        shifted_ratemap = np.max(shifted_ratemap) - shifted_ratemap + np.min(shifted_ratemap)

    return calc_MRL_mp(shifted_ratemap, ray_width, dist_bin_cents)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class BoundaryTuning:
    """
    Compute and classify egocentric (EBC) and retinocentric (RBC) boundary cells.

    Quick-start
    -----------
    >>> bt = BoundaryTuning(preprocessed_data)
    >>> ebc_res, rbc_res = bt.identify_responses_both()
    >>> bt.make_summary_pdf('boundary_summary.pdf')
    >>> bt.save_results_combined('boundary_results.h5')

    For backward compatibility the old API still works:
    >>> bt.identify_responses(use_angle='head')
    """

    def __init__(self, preprocessed_data):
        self.data = preprocessed_data

        self.ray_width    = 3    # degrees per angular bin
        self.max_dist     = 26   # cm
        self.dist_bin_size = 2.  # cm

        self.head_ang  = None
        self.pupil_ang = None
        self.ego_ang   = None

        # Legacy criteria storage (populate during identify_responses)
        self.criteria_out = {}
        for c in range(np.size(self.data['norm_spikes'], 0)):
            self.criteria_out['cell_{:03d}'.format(c)] = {}

        # New storage for dual EBC+RBC pipeline
        self.ebc_results = None
        self.rbc_results = None
        self.is_EBC = None
        self.is_RBC = None

    # ------------------------------------------------------------------
    # Angle computation
    # ------------------------------------------------------------------

    def calc_allo_yaw(self):
        """Allocentric head direction from preprocessed data."""
        self.head_ang = self.data['head_yaw_deg']

    def calc_allo_pupil(self):
        """
        Allocentric gaze direction = head_yaw + (theta - ang_offset).

        Sign convention fix
        -------------------
        The preprocessed ``pupil_from_head`` was stored as
        ``ang_offset_config - theta``, which has the *wrong sign* for gaze:
        adding it to head_yaw shifts gaze LEFT when the eye rotates RIGHT.
        The correct formula is ``gaze = head + (theta - ang_offset)``.

        Offset priority (highest to lowest)
        ------------------------------------
        1. ``ang_offset_vor_regression`` in data  (regression-based, preferred)
        2. ``ang_offset_vor_null`` in data         (VOR-null fallback)
        3. Compute on-the-fly from theta_interp + head_yaw_deg
        4. Flip sign of stored ``pupil_from_head`` (coarse fallback)
        """
        head  = self.data['head_yaw_deg'].copy()
        theta = self.data.get('theta_interp', None)

        if theta is not None:
            # Determine ang_offset from calibration keys or compute on-the-fly
            if 'ang_offset_vor_regression' in self.data:
                ang_offset = float(self.data['ang_offset_vor_regression'])
            elif 'ang_offset_vor_null' in self.data:
                ang_offset = float(self.data['ang_offset_vor_null'])
            else:
                import fm2p as _fm2p
                twopT = self.data.get('twopT', None)
                fps = (float(1.0 / np.nanmedian(np.diff(twopT)))
                       if twopT is not None and len(twopT) > 1 else 30.0)
                vor = _fm2p.calc_vor_eye_offset(theta, head, fps)
                ang_offset = vor['ang_offset_vor_regression']

            pfh = np.asarray(theta, dtype=float) - ang_offset  # sign-corrected
        else:
            # Coarse fallback: flip sign of stored pupil_from_head
            pfh = -self.data['pupil_from_head'].copy()

        pfh = np.asarray(pfh, dtype=float)

        # Radians/degrees guard -------------------------------------------------
        # pfh should span many degrees (typically 20–60° for freely moving mice).
        # Older preprocessed files stored theta_interp in radians (before the
        # np.rad2deg conversion was added to preprocess.py), making
        # ang_offset_vor_regression also in radians.  In that case pfh is in
        # radians too, typically 0.35–1.0 rad for ±20–30° mouse eye movements.
        # Any pfh range < 2.0 is physiologically implausible in degrees (< 2°
        # total eye deviation is essentially no movement) but plausible in
        # radians, so we auto-convert.  Values ≥ 2.0 are unambiguously degrees.
        _pfh_range = float(np.nanmax(pfh) - np.nanmin(pfh))
        if 1e-6 < _pfh_range < 2.0:
            print(f'  [WARNING] Gaze offset (pfh) range = {_pfh_range:.4f} — '
                  f'consistent with radians (expected ≥ 20° for freely moving '
                  f'data). Auto-converting to degrees.')
            pfh = np.rad2deg(pfh)
        # -----------------------------------------------------------------------

        n = min(len(head), len(pfh))
        self.pupil_ang = (head[:n] + pfh[:n]) % 360

    def calc_ego(self):
        """Egocentric angle to the pillar."""
        self.ego_ang = self.data['egocentric'] + 180.

    def _get_angle_trace(self, angle_type):
        """
        Return the allocentric reference angle trace (degrees) for a given type.

        angle_type options
        ------------------
        'head'  / 'egow'  : allocentric head direction (EBC)
        'gaze'  / 'pupil' : allocentric gaze = head + eye  (RBC)
        'ego'   / 'egop'  : egocentric angle to pillar
        'retino'          : retinocentric angle to pillar (legacy)
        """
        if angle_type in ('head', 'egow'):
            if self.head_ang is None:
                self.calc_allo_yaw()
            return self.head_ang

        elif angle_type in ('gaze', 'pupil'):
            if self.pupil_ang is None:
                self.calc_allo_pupil()
            return self.pupil_ang

        elif angle_type in ('ego', 'egop'):
            if self.ego_ang is None:  # BUG FIX: was checking pupil_ang
                self.calc_ego()
            return self.ego_ang

        elif angle_type == 'retino':
            return self.data['retinocentric'] + 180.

        else:
            raise ValueError(f"Unknown angle_type '{angle_type}'. "
                             "Use 'head', 'gaze', 'ego', or 'retino'.")

    # ------------------------------------------------------------------
    # Core ray casting
    # ------------------------------------------------------------------

    def _compute_ray_dists_from_trace(self, angle_trace_deg):
        """
        Cast 360-degree fan of rays from each frame's head position and return
        the distance to the nearest wall.

        Parameters
        ----------
        angle_trace_deg : np.ndarray
            Allocentric reference angle (degrees) for every frame in the
            recording.  May be shorter than norm_spikes axis-1 (e.g. gaze
            has one fewer frame); frames beyond its length are dropped.

        Returns
        -------
        ray_distances : np.ndarray, shape (N_used_frames, N_angular_bins)
            Distance to nearest wall (cm) for each ray.  NaN where no wall
            was intersected.
        """
        p2c = self.data['pxls2cm']
        x_full = self.data['head_x'].copy() / p2c
        y_full = self.data['head_y'].copy() / p2c

        # Clip use_inds to the valid range of angle_trace_deg
        max_valid = len(angle_trace_deg)
        use_inds  = np.where(self.useinds)[0]
        use_inds  = use_inds[use_inds < max_valid]

        # Exclude frames where the reference angle is NaN (e.g. failed eye
        # tracking).  NaN angles produce NaN ray directions; all wall-intersection
        # comparisons evaluate False for NaN, so best never updates from inf and
        # every ray_distance stays NaN → zero occupancy → all-zero rate maps.
        nan_mask = np.isnan(angle_trace_deg[use_inds])
        if nan_mask.any():
            print(f'    [WARNING] {int(nan_mask.sum())}/{len(use_inds)} frames '
                  f'have NaN reference angle and will be excluded from ray casting.')
            use_inds = use_inds[~nan_mask]

        N_frames  = len(use_inds)

        # Store so downstream functions know which frames map to which rows.
        self._ray_dist_use_inds = use_inds.copy()

        x_trace = x_full[use_inds]
        y_trace = y_full[use_inds]
        ang_rad  = np.deg2rad(angle_trace_deg[use_inds])

        # Arena walls defined by corner pairs (in cm)
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

        # Precompute distance bin edges (needed by callers)
        self.dist_bin_edges = np.arange(0, self.max_dist + self.dist_bin_size,
                                        self.dist_bin_size)
        self.dist_bin_cents = self.dist_bin_edges[:-1] + self.dist_bin_size / 2

        ray_distances = np.full((N_frames, N_ang), np.nan)

        for fr in tqdm(range(N_frames), leave=False, desc='    ray casting'):
            px, py = x_trace[fr], y_trace[fr]
            base_ang = ang_rad[fr]

            for ri, off in enumerate(ray_offsets_rad):
                ray_ang = base_ang + off
                rv = np.array([np.cos(ray_ang), np.sin(ray_ang)])  # ray unit vector

                best = np.inf
                for wall in walls:
                    start  = wall[0]
                    vec    = wall[1] - wall[0]    # wall direction vector
                    rel    = np.array([px, py]) - start

                    det = np.cross(vec, rv)
                    if det == 0:
                        continue  # parallel

                    t = np.cross(rel, rv) / det  # parameter along wall [0,1]
                    if t < 0 or t > 1:
                        continue

                    isect = start + t * vec
                    to_isect = isect - np.array([px, py])

                    if np.dot(to_isect, rv) < 0:
                        continue  # intersection is behind the ray

                    dist = np.linalg.norm(to_isect)
                    if dist < best:
                        best = dist

                if best < np.inf:
                    ray_distances[fr, ri] = best

        return ray_distances

    # ---- legacy wrapper (kept for backward compat) ----
    def get_ray_distances(self, angle='head'):
        """
        Backward-compatible wrapper.  Computes and stores self.ray_distances.
        """
        angle_trace = self._get_angle_trace(angle)
        self.ray_distances = self._compute_ray_dists_from_trace(angle_trace)
        return self.ray_distances

    # ------------------------------------------------------------------
    # Occupancy
    # ------------------------------------------------------------------

    def _compute_occupancy_from_raydists(self, ray_distances):
        """
        Count the number of frames in each (angle × distance) bin.

        Parameters
        ----------
        ray_distances : np.ndarray, shape (N_frames, N_ang)

        Returns
        -------
        occupancy : np.ndarray, shape (N_ang, N_dist)
        """
        N_ang  = int(360 / self.ray_width)
        N_dist = len(self.dist_bin_edges) - 1
        occupancy = np.zeros((N_ang, N_dist))

        for d, lo in enumerate(self.dist_bin_edges[:-1]):
            hi   = lo + self.dist_bin_size
            mask = (ray_distances >= lo) & (ray_distances < hi)
            occupancy[:, d] = np.sum(mask, axis=0)

        return occupancy

    # ---- legacy wrapper ----
    def calc_occupancy(self, inds=None):
        """
        Backward-compatible occupancy wrapper using self.ray_distances.
        `inds` can be a boolean mask or integer array of absolute frame indices.
        """
        if inds is None:
            return self._compute_occupancy_from_raydists(self.ray_distances)

        # Determine which rows of self.ray_distances to keep
        abs_inds = np.nonzero(self.useinds)[0]  # absolute indices with valid data
        if isinstance(inds, np.ndarray) and inds.dtype == bool:
            target_inds = np.where(inds)[0]
        else:
            target_inds = np.asarray(inds)

        mask = np.isin(abs_inds, target_inds)
        rd_sub = self.ray_distances[mask, :]
        return self._compute_occupancy_from_raydists(rd_sub)

    # ------------------------------------------------------------------
    # Rate maps
    # ------------------------------------------------------------------

    def _compute_rate_maps_from_raydists(self, ray_distances, occupancy):
        """
        Accumulate spike rates into 2D (angle × distance) bins for all cells.

        Uses vectorised np.digitize / np.add.at — much faster than triple loops.

        Parameters
        ----------
        ray_distances : np.ndarray, shape (N_frames, N_ang)
        occupancy     : np.ndarray, shape (N_ang, N_dist)

        Returns
        -------
        rate_maps : np.ndarray, shape (N_cells, N_ang, N_dist)
        """
        N_frames_rd = ray_distances.shape[0]

        # Use the exact frame indices stored during ray casting to avoid
        # misalignment when NaN-angle frames were removed.
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

        # Normalise by occupancy.
        # Bins with < min_occ frames (~267 ms at 30 Hz per Alexander 2020) are
        # set to NaN to avoid inflating rate values from noisy low-count bins.
        min_occ = 8  # frames (~267 ms at 30 Hz)
        occ_mask = occupancy < min_occ
        for c in range(N_cells):
            rm = rate_maps[c] / (occupancy + 1e-6)
            rm[occ_mask] = np.nan
            rate_maps[c] = rm

        return rate_maps

    def _compute_ratemap_for_cell_subset(self, c, split_abs_inds, ray_distances):
        """
        Compute a rate map for cell `c` using only the frames in split_abs_inds.

        Parameters
        ----------
        c              : int  cell index
        split_abs_inds : np.ndarray  absolute frame indices (from full recording)
        ray_distances  : np.ndarray  shape (N_used_frames, N_ang) – pre-computed
                         for the full useinds set corresponding to the current pipeline.

        Returns
        -------
        rate_map : np.ndarray, shape (N_ang, N_dist)
        """
        # Which rows of ray_distances correspond to split_abs_inds?
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

    # ---- legacy wrappers ----
    def calc_rate_maps_mp(self):
        """Backward-compatible multiprocessing rate-map wrapper."""
        nCells     = np.size(self.data['norm_spikes'], 0)
        N_ang      = int(360 / self.ray_width)
        N_dist     = len(self.dist_bin_edges) - 1
        n_proc     = multiprocessing.cpu_count() - 1

        pbar = tqdm(total=nCells)
        def _update(*_): pbar.update()

        spikes = self.data['norm_spikes'].copy()[:, self.useinds.astype(bool)]
        pool = multiprocessing.Pool(processes=n_proc)
        mp_param_set = [
            pool.apply_async(
                rate_map_mp,
                args=(spikes[c], self.occupancy, self.ray_distances,
                      self.ray_width, self.dist_bin_edges, self.dist_bin_size),
                callback=_update
            ) for c in range(nCells)
        ]
        outputs = [r.get() for r in mp_param_set]
        self.rate_maps = np.zeros((nCells, N_ang, N_dist))
        for c, rm in enumerate(outputs):
            self.rate_maps[c] = rm
        pbar.close(); pool.close()
        return self.rate_maps

    def calc_rate_maps(self, use_mp=True):
        """Backward-compatible rate-map wrapper."""
        if use_mp:
            return self.calc_rate_maps_mp()
        self.rate_maps = self._compute_rate_maps_from_raydists(
            self.ray_distances, self.occupancy)
        return self.rate_maps

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------

    def _smooth_single(self, rm, sigma=2.5):
        """
        NaN-aware 2D Gaussian smoothing with circular padding on the angle axis.

        ``gaussian_filter`` propagates NaN: with sigma=5 and only 13 distance
        bins, a single NaN from the min_occ mask contaminates the entire
        distance column.  Fix: fill NaN bins with 0, smooth both the filled
        values and a binary weight map, then divide — equivalent to computing
        a weighted local mean that ignores missing bins.
        """
        nan_mask   = np.isnan(rm)
        rm_fill    = rm.copy()
        rm_fill[nan_mask] = 0.
        weights    = (~nan_mask).astype(float)

        # Triple-wrap on angle axis for circular boundary conditions.
        padded_v = np.vstack([rm_fill, rm_fill, rm_fill])
        padded_w = np.vstack([weights,  weights,  weights])

        sv = gaussian_filter(padded_v, sigma=sigma)
        sw = gaussian_filter(padded_w, sigma=sigma)

        N = rm.shape[0]
        smoothed = sv[N: 2 * N, :] / (sw[N: 2 * N, :] + 1e-10)
        return smoothed

    def _smooth_rate_maps_arr(self, rate_maps):
        """
        Smooth an array of rate maps with angular-wrap padding.

        Parameters
        ----------
        rate_maps : np.ndarray, shape (N_cells, N_ang, N_dist)

        Returns
        -------
        smoothed : np.ndarray, same shape
        """
        smoothed = np.zeros_like(rate_maps)
        for c in range(rate_maps.shape[0]):
            smoothed[c] = self._smooth_single(rate_maps[c])
        return smoothed

    def smooth_rate_maps(self):
        """Backward-compatible smooth wrapper (operates on self.rate_maps)."""
        self.smoothed_rate_maps = self._smooth_rate_maps_arr(self.rate_maps)
        return self.smoothed_rate_maps

    def smooth_map_pair(self, map1, map2):
        """Smooth two rate maps without touching self.rate_maps."""
        return self._smooth_single(map1), self._smooth_single(map2)

    # ------------------------------------------------------------------
    # Rate-map quality metrics
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Mean resultant vector (FIX: normalise by total weight)
    # ------------------------------------------------------------------

    def _calc_mean_resultant(self, rm):
        """
        Compute mean resultant vector, length, and angle.

        FIX: normalises by np.sum(rm) so MRL ∈ [0, 1].

        Returns
        -------
        mr  : complex
        mrl : float  (mean resultant length)
        mra : float  (mean resultant angle, radians, [0, 2π])
        """
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

    # ------------------------------------------------------------------
    # Inverse-response classification
    # ------------------------------------------------------------------

    def _identify_inverse_responses_from(self, rate_maps, inv_thresh=2):
        """
        Classify each cell as inverse (IEBC/IRBC) if ≥ inv_thresh of three
        criteria pass: negative skewness, inverted map is less dispersed,
        inverted RF is smaller.

        Returns
        -------
        is_inverse : np.ndarray bool, shape (N_cells,)
        criteria   : list of dicts, one per cell
        """
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

    # ---- legacy wrapper ----
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

    # ------------------------------------------------------------------
    # Split-half reliability
    # ------------------------------------------------------------------

    def _calc_correlation_across_split_v2(self, c, ray_distances,
                                          ncnk=20, corr_thresh=0.6):
        """
        Split-half reliability following Alexander et al. 2020.

        Each half is the first / second half of the session (not interleaved
        chunks), matching the original paper.  Reliability is assessed by:
          (i)  absolute circular difference in preferred angle < 45°
          (ii) change in preferred distance < 50%

        `corr_thresh` is kept as a parameter for backward compatibility but is
        no longer used as a pass criterion.

        Returns
        -------
        corr      : float  2D Pearson r (computed on non-NaN bins, informational)
        passes    : bool   True if angle_diff < 45° AND dist_diff < 50%
        rm1_smooth, rm2_smooth : np.ndarray  smoothed split-half rate maps
        """
        max_sp   = self.data['norm_spikes'].shape[1]
        abs_inds = self._ray_dist_use_inds[:ray_distances.shape[0]]
        abs_inds = abs_inds[abs_inds < max_sp]
        n_used   = len(abs_inds)

        if n_used == 0:
            nan_map = np.full((int(360 / self.ray_width),
                               len(self.dist_bin_edges) - 1), np.nan)
            return np.nan, False, nan_map, nan_map

        # First and second half of session
        mid   = n_used // 2
        s1    = abs_inds[:mid]
        s2    = abs_inds[mid:]

        rm1 = self._compute_ratemap_for_cell_subset(c, s1, ray_distances)
        rm2 = self._compute_ratemap_for_cell_subset(c, s2, ray_distances)
        rm1_s, rm2_s = self.smooth_map_pair(rm1, rm2)

        # Informational Pearson r (using only bins valid in both halves)
        valid = ~np.isnan(rm1_s) & ~np.isnan(rm2_s)
        if valid.sum() > 5:
            a, b = rm1_s[valid], rm2_s[valid]
            denom = np.std(a) * np.std(b)
            corr  = float(np.mean((a - a.mean()) * (b - b.mean())) / denom) if denom > 0 else np.nan
        else:
            corr = np.nan

        # Alexander 2020 reliability criteria
        # Preferred angle: angular bin index of max across distance dimension
        def _pref_angle_dist(rm):
            """Return (preferred_angle_deg, preferred_dist_cm) for a 2D rate map."""
            rm_nan = rm.copy()
            if np.all(np.isnan(rm_nan)):
                return np.nan, np.nan
            flat_idx = np.nanargmax(rm_nan)
            ai, di   = np.unravel_index(flat_idx, rm_nan.shape)
            pref_ang = float(ai * self.ray_width)          # degrees
            pref_dist = float(self.dist_bin_cents[di])     # cm
            return pref_ang, pref_dist

        ang1, dist1 = _pref_angle_dist(rm1_s)
        ang2, dist2 = _pref_angle_dist(rm2_s)

        if np.isnan(ang1) or np.isnan(ang2):
            passes = False
        else:
            # Circular difference in preferred angle
            diff_ang  = abs(((ang1 - ang2 + 180) % 360) - 180)
            # Fractional change in preferred distance
            mean_dist = (dist1 + dist2) / 2.0
            diff_dist = abs(dist1 - dist2) / mean_dist if mean_dist > 0 else np.inf
            passes = (diff_ang < 45.0) and (diff_dist < 0.5)

        return corr, passes, rm1_s, rm2_s

    # ---- legacy wrapper ----
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

    # ------------------------------------------------------------------
    # Shuffle MRL test
    # ------------------------------------------------------------------

    def _test_mrl_against_shuffles(self, c, mrl, ray_distances, occupancy,
                                   is_inverse, n_shfl=100, pctl=99):
        """
        Compare observed MRL against a null distribution from circularly-shifted
        spike trains.

        Parameters
        ----------
        c           : int  cell index
        mrl         : float  observed mean resultant length
        ray_distances : np.ndarray, shape (N_used, N_ang)
        occupancy   : np.ndarray, shape (N_ang, N_dist)
        is_inverse  : bool
        n_shfl      : int  number of shuffles
        pctl        : float  percentile threshold (default 99)

        Returns
        -------
        thresh        : float  shuffle-distribution threshold
        passes        : bool
        shuffled_mrls : np.ndarray, shape (n_shfl,)
        """
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
            # Unrestricted circular shift (Alexander et al. 2020); minimum 1 frame.
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
            rm_sh[occupancy < 8] = np.nan  # same min_occ mask as full rate maps

            if is_inverse:
                rm_sh = self._invert_ratemap(rm_sh)

            _, shf_mrl, _ = self._calc_mean_resultant(rm_sh)
            shuffled_mrls[i] = shf_mrl

        thresh  = float(np.percentile(shuffled_mrls, pctl))
        passes  = mrl > thresh
        return thresh, passes, shuffled_mrls

    # ---- legacy MP wrapper ----
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

    # ------------------------------------------------------------------
    # Legacy classify methods (backward compat)
    # ------------------------------------------------------------------

    def identify_boundary_cells(self, n_chunks=20, n_shuffles=20,
                                corr_thresh=0.6, mp=True):
        """Backward-compatible EBC classifier (uses self.rate_maps etc.)."""
        N_cells  = self.rate_maps.shape[0]
        self.is_EBC = np.zeros(N_cells, dtype=bool)

        for c in tqdm(range(N_cells)):
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

    # ------------------------------------------------------------------
    # Unified angle-pipeline (new API)
    # ------------------------------------------------------------------

    def _run_angle_pipeline(self, angle_type, n_chunks=20, n_shuffles=100,
                            corr_thresh=0.6):
        """
        Run the complete RF pipeline for a single reference frame.

        Steps
        -----
        1. Cast rays using angle_type reference direction
        2. Compute occupancy
        3. Compute rate maps for all cells
        4. Smooth rate maps
        5. Identify inverse responses
        6. For each cell: split-half correlation + shuffle-MRL reliability test
        7. Classify as boundary cell (both criteria must pass)

        Parameters
        ----------
        angle_type  : str  'head' for EBC, 'gaze' for RBC
        n_chunks    : int  split-half chunk count
        n_shuffles  : int  number of spike-train shuffles for MRL null
        corr_thresh : float  split-half correlation pass threshold

        Returns
        -------
        results : dict  with keys:
            ray_distances, occupancy, rate_maps, smoothed_rate_maps,
            is_inverse, is_bc, criteria, angle_type,
            ray_width, dist_bin_edges, dist_bin_cents, angle_rad
        """
        label = angle_type.upper()

        print(f'  [{label}] Casting rays...')
        angle_trace  = self._get_angle_trace(angle_type)
        ray_distances = self._compute_ray_dists_from_trace(angle_trace)

        print(f'  [{label}] Computing occupancy...')
        occupancy = self._compute_occupancy_from_raydists(ray_distances)

        print(f'  [{label}] Computing rate maps...')
        rate_maps = self._compute_rate_maps_from_raydists(ray_distances, occupancy)

        print(f'  [{label}] Smoothing...')
        smoothed = self._smooth_rate_maps_arr(rate_maps)

        print(f'  [{label}] Identifying inverse responses...')
        is_inverse, inv_crit = self._identify_inverse_responses_from(rate_maps)

        print(f'  [{label}] Reliability tests for {rate_maps.shape[0]} cells...')
        N_cells = rate_maps.shape[0]
        is_bc   = np.zeros(N_cells, dtype=bool)
        cell_criteria = {}

        for c in tqdm(range(N_cells), desc=f'  [{label}]'):
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
        print(f'  [{label}] {n_pass}/{N_cells} cells classified as boundary cells.')

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

    # ------------------------------------------------------------------
    # Main dual-pipeline entry point
    # ------------------------------------------------------------------

    def identify_responses_both(self, use_light=False, use_dark=False,
                                n_chunks=20, n_shuffles=100, corr_thresh=0.6):
        """
        Run EBC (head direction) and RBC (gaze direction) pipelines.

        EBC reference : allocentric yaw
        RBC reference : allocentric gaze = yaw + theta_eye

        Parameters
        ----------
        use_light   : bool  restrict to lit epochs
        use_dark    : bool  restrict to dark epochs
        n_chunks    : int   split-half chunks (default 20)
        n_shuffles  : int   MRL shuffle count (default 100)
        corr_thresh : float split-half correlation pass threshold (default 0.6)

        Returns
        -------
        ebc_results, rbc_results : dict
            Each has keys: ray_distances, occupancy, rate_maps,
            smoothed_rate_maps, is_inverse, is_bc, criteria, ...
        """
        # Set up frame mask
        N = self.data['norm_spikes'].shape[1]
        if use_light:
            useinds = (self.data['ltdk_state_vec'][:N].copy() == 1)
        elif use_dark:
            useinds = (self.data['ltdk_state_vec'][:N].copy() == 0)
        else:
            useinds = np.ones(N, dtype=bool)

        self.useinds = useinds & (self.data['speed'][:N] > 5.)

        # Pre-compute angle traces
        self.calc_allo_yaw()
        self.calc_allo_pupil()

        print('=' * 60)
        print('EBC PIPELINE  (reference: head direction / yaw)')
        print('=' * 60)
        self.ebc_results = self._run_angle_pipeline(
            'head', n_chunks, n_shuffles, corr_thresh)

        print('=' * 60)
        print('RBC PIPELINE  (reference: gaze = yaw + theta_eye)')
        print('=' * 60)
        self.rbc_results = self._run_angle_pipeline(
            'gaze', n_chunks, n_shuffles, corr_thresh)

        self.is_EBC = self.ebc_results['is_bc'].astype(bool)
        self.is_RBC = self.rbc_results['is_bc'].astype(bool)

        N_cells = len(self.is_EBC)
        print('=' * 60)
        print(f'  EBC: {np.sum(self.is_EBC)}/{N_cells}')
        print(f'  RBC: {np.sum(self.is_RBC)}/{N_cells}')
        print(f'  Both: {np.sum(self.is_EBC & self.is_RBC)}/{N_cells}')
        print('=' * 60)

        return self.ebc_results, self.rbc_results

    # ------------------------------------------------------------------
    # Legacy pipeline (backward compat)
    # ------------------------------------------------------------------

    def identify_responses(self, use_angle='head', use_light=False,
                           use_dark=False, skip_classification=False):
        """
        Backward-compatible single-angle pipeline.

        Populates self.rate_maps, self.is_IEBC, self.is_EBC, self.criteria_out,
        self.data_out.
        """
        N = self.data['norm_spikes'].shape[1]
        if use_light:
            useinds = (self.data['ltdk_state_vec'][:N].copy() == 1)
        elif use_dark:
            useinds = (self.data['ltdk_state_vec'][:N].copy() == 0)
        else:
            useinds = np.ones(N, dtype=bool)

        self.useinds = useinds & (self.data['speed'][:N] > 5.)

        if use_angle == 'head':
            self.calc_allo_yaw()
        elif use_angle in ('pupil', 'gaze'):
            self.calc_allo_pupil()
        elif use_angle in ('ego', 'egop'):
            self.calc_ego()

        print('  -> Calculating ray distances.')
        _ = self.get_ray_distances(angle=use_angle)
        print('  -> Calculating occupancy.')
        self.occupancy = self.calc_occupancy(inds=self.useinds)
        print('  -> Calculating rate maps.')
        _ = self.calc_rate_maps()
        print('  -> Smoothing rate maps.')
        _ = self.smooth_rate_maps()

        if not skip_classification:
            print('  -> Identifying inverse boundary cells.')
            _ = self.identify_inverse_responses()
            print('  -> Identifying boundary cells.')
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

    # ------------------------------------------------------------------
    # Summary PDF
    # ------------------------------------------------------------------

    def make_summary_pdf(self, savepath):
        """
        Generate a multi-page PDF summarising EBC and RBC receptive fields.

        One page per cell that is reliable for at least one of EBC or RBC.
        Each page shows:
          • Left  : EBC polar rate map (reference = head direction)
          • Right : RBC polar rate map (reference = gaze direction)
        Maps are labelled with MRL, split-half CC, and reliability status.

        Parameters
        ----------
        savepath : str  path to output PDF
        """
        assert self.ebc_results is not None and self.rbc_results is not None, \
            "Run identify_responses_both() before make_summary_pdf()."

        cmap = fm2p.make_parula()

        theta_edges = np.deg2rad(
            np.arange(0, 360 + self.ray_width, self.ray_width))
        r_edges = self.dist_bin_edges

        n_ebc = int(np.sum(self.is_EBC))
        n_rbc = int(np.sum(self.is_RBC))
        if n_ebc == 0:
            print('  No reliable EBC cells found — skipping summary PDF.')
            return
        if n_rbc == 0:
            print('  No reliable RBC cells found — skipping summary PDF.')
            return

        show_cells = np.where(self.is_EBC | self.is_RBC)[0]
        if len(show_cells) == 0:
            print('  No reliable EBC or RBC cells found — no PDF generated.')
            return

        print(f'  Writing PDF with {len(show_cells)} pages → {savepath}')

        with PdfPages(savepath) as pdf:
            for c in show_cells:
                fig = plt.figure(figsize=(13, 6))
                # Two polar axes + a narrow colorbar axes
                ax_ebc = fig.add_axes([0.05, 0.10, 0.38, 0.75],
                                       projection='polar')
                ax_rbc = fig.add_axes([0.52, 0.10, 0.38, 0.75],
                                       projection='polar')
                cax    = fig.add_axes([0.93, 0.20, 0.015, 0.55])

                # ---- EBC panel ----
                ebc_rm  = self.ebc_results['smoothed_rate_maps'][c]
                ck_ebc  = self.ebc_results['criteria']['cell_{:03d}'.format(c)]
                mrl_ebc = ck_ebc['mean_resultant_length']
                cc_ebc  = ck_ebc['corr_coeff']
                ebc_ok  = bool(self.is_EBC[c])
                ebc_col = '#1a7f37' if ebc_ok else '#888888'

                vmax = np.nanpercentile(
                    np.concatenate([ebc_rm.flatten(),
                                    self.rbc_results['smoothed_rate_maps'][c].flatten()]),
                    99)
                vmax = max(vmax, 1e-6)

                im = ax_ebc.pcolormesh(theta_edges, r_edges, ebc_rm.T,
                                       cmap=cmap, shading='auto',
                                       vmin=0, vmax=vmax)
                ax_ebc.set_title(
                    f'EBC — {"RELIABLE" if ebc_ok else "not reliable"}\n'
                    f'MRL={mrl_ebc:.3f}  CC={cc_ebc:.3f}',
                    color=ebc_col, fontsize=10, pad=12)
                _polar_axes_style(ax_ebc,
                    labels=['fwd', 'right', 'bkwd', 'left'],
                    r_max=self.max_dist)

                # ---- RBC panel ----
                rbc_rm  = self.rbc_results['smoothed_rate_maps'][c]
                ck_rbc  = self.rbc_results['criteria']['cell_{:03d}'.format(c)]
                mrl_rbc = ck_rbc['mean_resultant_length']
                cc_rbc  = ck_rbc['corr_coeff']
                rbc_ok  = bool(self.is_RBC[c])
                rbc_col = '#1a5fa8' if rbc_ok else '#888888'

                ax_rbc.pcolormesh(theta_edges, r_edges, rbc_rm.T,
                                  cmap=cmap, shading='auto',
                                  vmin=0, vmax=vmax)
                ax_rbc.set_title(
                    f'RBC — {"RELIABLE" if rbc_ok else "not reliable"}\n'
                    f'MRL={mrl_rbc:.3f}  CC={cc_rbc:.3f}',
                    color=rbc_col, fontsize=10, pad=12)
                _polar_axes_style(ax_rbc,
                    labels=['center', 'temporal', 'surround', 'nasal'],
                    r_max=self.max_dist, shade_off_retina=True)

                # ---- Colorbar ----
                fig.colorbar(im, cax=cax, label='Rate (a.u.)')

                # ---- Shuffle distributions inset ----
                _add_shuffle_inset(fig, ck_ebc, ebc_ok, color=ebc_col,
                                   rect=[0.08, 0.02, 0.18, 0.12])
                _add_shuffle_inset(fig, ck_rbc, rbc_ok, color=rbc_col,
                                   rect=[0.55, 0.02, 0.18, 0.12])

                # ---- Overall title ----
                status = []
                if ebc_ok: status.append('EBC')
                if rbc_ok: status.append('RBC')
                tag = ', '.join(status) if status else 'neither'
                fig.suptitle(f'Cell {c:03d}   reliable: {tag}')
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        print(f'  Done -> {savepath}')

    # ------------------------------------------------------------------
    # Diagnostic figures
    # ------------------------------------------------------------------

    def make_diagnostic_figs(self, savedir):
        """
        Save a set of population-level diagnostic figures to savedir/.

        Figures generated
        -----------------
        01_mrl_distributions.pdf  — EBC and RBC MRL histograms (real vs shuffled null)
        02_split_half_corr.pdf    — split-half CC distributions
        03_occupancy_maps.pdf     — head-direction and gaze-direction occupancy
        04_population_ratemaps.pdf — mean rate map across all cells per type
        05_cell_scatter.pdf       — scatter of EBC MRL vs RBC MRL per cell
        """
        assert self.ebc_results is not None and self.rbc_results is not None, \
            "Run identify_responses_both() before make_diagnostic_figs()."

        os.makedirs(savedir, exist_ok=True)
        cmap = fm2p.make_parula()

        theta_edges = np.deg2rad(
            np.arange(0, 360 + self.ray_width, self.ray_width))
        r_edges = self.dist_bin_edges
        N_cells = self.ebc_results['rate_maps'].shape[0]

        # ---- 1. MRL distributions ----
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

        # ---- 2. Split-half correlations ----
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

        # ---- 3. Occupancy maps ----
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

        # ---- 4. Mean rate maps ----
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

        # ---- 5. EBC MRL vs RBC MRL scatter ----
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
        """
        Generate detailed PDFs for EBCs and RBCs.
        
        For each reliable cell, creates a page with:
        - Full polar rate map
        - Split-half polar rate maps
        - Scatter of EBC MRL vs RBC MRL (highlighting cell)
        - Scatter of Split-half CC vs MRL for EBC (highlighting cell)
        - Scatter of Split-half CC vs MRL for RBC (highlighting cell)
        
        Cells are sorted by MRL.
        """
        assert self.ebc_results is not None and self.rbc_results is not None, \
            "Run identify_responses_both() before make_detailed_pdf()."

        N_cells = len(self.is_EBC)

        def _get_metrics(results):
            mrls = np.array([results['criteria'][f'cell_{c:03d}']['mean_resultant_length'] for c in range(N_cells)])
            ccs  = np.array([results['criteria'][f'cell_{c:03d}']['corr_coeff'] for c in range(N_cells)])
            return mrls, ccs

        # Helper to make one PDF
        def _write_pdf(target_indices, results, label, filename,
                       axis_labels, shade_off_retina=False):
            if len(target_indices) == 0:
                print(f"No reliable {label} cells found. Skipping {filename}.")
                return

            # Compute population metrics here so each PDF is independent
            ebc_mrls, ebc_ccs = _get_metrics(self.ebc_results)
            rbc_mrls, rbc_ccs = _get_metrics(self.rbc_results)
            sort_metric = rbc_mrls if label == 'RBC' else ebc_mrls

            # Sort by MRL descending
            sorted_indices = target_indices[np.argsort(sort_metric[target_indices])[::-1]]
            
            theta_edges = np.deg2rad(np.arange(0, 360 + self.ray_width, self.ray_width))
            r_edges = self.dist_bin_edges
            cmap = fm2p.make_parula()

            print(f"Writing {label} PDF to {filename}...")
            with PdfPages(filename) as pdf:
                for c in tqdm(sorted_indices, desc=f"Writing {label} PDF"):
                    fig = plt.figure(figsize=(14, 8))
                    
                    # Layout: 
                    # Top row: Full RF, Split 1, Split 2 (Polar)
                    # Bottom row: Scatters
                    
                    # Polar axes
                    ax_full = fig.add_axes([0.05, 0.55, 0.25, 0.4], projection='polar')
                    ax_s1   = fig.add_axes([0.35, 0.55, 0.25, 0.4], projection='polar')
                    ax_s2   = fig.add_axes([0.65, 0.55, 0.25, 0.4], projection='polar')
                    
                    # Scatter axes
                    ax_sc1 = fig.add_axes([0.05, 0.1, 0.25, 0.35])
                    ax_sc2 = fig.add_axes([0.35, 0.1, 0.25, 0.35])
                    ax_sc3 = fig.add_axes([0.65, 0.1, 0.25, 0.35])
                    
                    # --- Polar Plots ---
                    full_rm = results['smoothed_rate_maps'][c]
                    s1_rm = results['criteria'][f'cell_{c:03d}']['split_rate_map_1']
                    s2_rm = results['criteria'][f'cell_{c:03d}']['split_rate_map_2']
                    
                    # Determine vmax from full map or all maps? Usually full map or consistent across splits.
                    # Let's use max of all three for consistent scaling on page
                    vmax = np.nanmax([full_rm, s1_rm, s2_rm])
                    if vmax == 0: vmax = 1.0
                    
                    for ax, rm, title in zip([ax_full, ax_s1, ax_s2], [full_rm, s1_rm, s2_rm], ['Full', 'Split 1', 'Split 2']):
                        ax.pcolormesh(theta_edges, r_edges, rm.T, cmap=cmap, shading='auto', vmin=0, vmax=vmax)
                        _polar_axes_style(ax, labels=axis_labels,
                                          r_max=self.max_dist,
                                          shade_off_retina=shade_off_retina)
                        ax.set_title(title, fontsize=10)

                    # --- Scatter 1: EBC MRL vs RBC MRL ---
                    ax_sc1.scatter(ebc_mrls, rbc_mrls, c='gray', s=10, alpha=0.5)
                    ax_sc1.scatter(ebc_mrls[c], rbc_mrls[c], c='red', s=50, marker='*', zorder=10)
                    ax_sc1.set_xlabel('EBC MRL')
                    ax_sc1.set_ylabel('RBC MRL')
                    ax_sc1.set_title('EBC vs RBC MRL')
                    
                    # --- Scatter 2: EBC CC vs MRL ---
                    ax_sc2.scatter(ebc_ccs, ebc_mrls, c='gray', s=10, alpha=0.5)
                    ax_sc2.scatter(ebc_ccs[c], ebc_mrls[c], c='red', s=50, marker='*', zorder=10)
                    ax_sc2.set_xlabel('EBC Split-Half CC')
                    ax_sc2.set_ylabel('EBC MRL')
                    ax_sc2.set_title('EBC Reliability')
                    
                    # --- Scatter 3: RBC CC vs MRL ---
                    ax_sc3.scatter(rbc_ccs, rbc_mrls, c='gray', s=10, alpha=0.5)
                    ax_sc3.scatter(rbc_ccs[c], rbc_mrls[c], c='red', s=50, marker='*', zorder=10)
                    ax_sc3.set_xlabel('RBC Split-Half CC')
                    ax_sc3.set_ylabel('RBC MRL')
                    ax_sc3.set_title('RBC Reliability')
                    
                    # Title
                    fig.suptitle(f'Cell {c} ({label}) - MRL={sort_metric[c]:.3f}', fontsize=16)
                    
                    pdf.savefig(fig)
                    plt.close(fig)

        # Generate EBC PDF
        ebc_indices = np.where(self.is_EBC)[0]
        _write_pdf(ebc_indices, self.ebc_results, 'EBC', savepath_ebc,
                   axis_labels=['fwd', 'right', 'bkwd', 'left'],
                   shade_off_retina=False)

        # Generate RBC PDF
        rbc_indices = np.where(self.is_RBC)[0]
        _write_pdf(rbc_indices, self.rbc_results, 'RBC', savepath_rbc,
                   axis_labels=['center', 'temporal', 'surround', 'nasal'],
                   shade_off_retina=True)

    # ------------------------------------------------------------------
    # HDF5 saving
    # ------------------------------------------------------------------

    def load_results(self, results_path):
        """
        Load analysis results from an HDF5 file (created by save_results_combined).
        Populates self.ebc_results, self.rbc_results, self.is_EBC, self.is_RBC, etc.
        """
        print(f"Loading results from {results_path}...")
        res = fm2p.read_h5(results_path)
        
        # Params
        if 'params' in res:
            params = res['params']
            self.ray_width = params['ray_width']
            self.max_dist = params['max_dist']
            self.dist_bin_size = params['dist_bin_size']
            self.dist_bin_edges = params['dist_bin_edges']
            self.dist_bin_cents = params['dist_bin_cents']
        
        # Results
        self.ebc_results = res.get('ebc')
        self.rbc_results = res.get('rbc')
        
        # Classification
        if 'classification' in res:
            cls = res['classification']
            self.is_EBC = cls['is_EBC'].astype(bool)
            self.is_RBC = cls['is_RBC'].astype(bool)
        
        print("Results loaded.")

    def save_results(self, savepath):
        """Save legacy (single-angle) results to HDF5."""
        fm2p.write_h5(savepath, convert_bools_to_ints(self.data_out))

    def save_results_combined(self, savepath):
        """
        Save EBC and RBC results to a single HDF5 file.

        Structure
        ---------
        /params/          — shared parameters
        /ebc/             — EBC maps, occupancy, ray distances
        /ebc/criteria/    — per-cell EBC metrics
        /rbc/             — RBC maps, occupancy, ray distances
        /rbc/criteria/    — per-cell RBC metrics
        /classification/  — is_EBC, is_RBC arrays
        """
        assert self.ebc_results is not None and self.rbc_results is not None, \
            "Run identify_responses_both() before save_results_combined()."

        def _criteria_flat(crit_dict):
            """Flatten criteria dict, removing large split-map arrays."""
            out = {}
            for key, val in crit_dict.items():
                # keep per-cell scalar metrics, shuffled MRLs, and split maps
                out[key] = {k: v for k, v in val.items()
                            if not isinstance(v, np.ndarray) or v.ndim <= 2}
            return out

        data_out = {
            'params': {
                'ray_width':      self.ray_width,
                'max_dist':       self.max_dist,
                'dist_bin_size':  self.dist_bin_size,
                'dist_bin_edges': self.dist_bin_edges,
                'dist_bin_cents': self.dist_bin_cents,
                'angle_rad':      np.deg2rad(np.arange(0, 360, self.ray_width)),
            },
            'ebc': {
                'rate_maps':          self.ebc_results['rate_maps'],
                'smoothed_rate_maps': self.ebc_results['smoothed_rate_maps'],
                'occupancy':          self.ebc_results['occupancy'],
                'ray_distances':      self.ebc_results['ray_distances'],
                'is_IEBC':            self.ebc_results['is_inverse'],
                'is_EBC':             self.ebc_results['is_bc'],
                'criteria':           _criteria_flat(self.ebc_results['criteria']),
            },
            'rbc': {
                'rate_maps':          self.rbc_results['rate_maps'],
                'smoothed_rate_maps': self.rbc_results['smoothed_rate_maps'],
                'occupancy':          self.rbc_results['occupancy'],
                'ray_distances':      self.rbc_results['ray_distances'],
                'is_IRBC':            self.rbc_results['is_inverse'],
                'is_RBC':             self.rbc_results['is_bc'],
                'criteria':           _criteria_flat(self.rbc_results['criteria']),
            },
            'classification': {
                'is_EBC': self.is_EBC.astype(int),
                'is_RBC': self.is_RBC.astype(int),
                'is_either': (self.is_EBC | self.is_RBC).astype(int),
                'is_both':   (self.is_EBC & self.is_RBC).astype(int),
            },
        }

        fm2p.write_h5(savepath, convert_bools_to_ints(data_out))
        print(f'  Results saved -> {savepath}')


# ---------------------------------------------------------------------------
# Module-level plot helpers
# ---------------------------------------------------------------------------

def _polar_axes_style(ax, labels=None, r_max=26, shade_off_retina=False):
    """Apply consistent styling to a polar rate-map axis.

    Parameters
    ----------
    ax               : matplotlib PolarAxes
    labels           : list of 4 str, tick labels at [0, π/2, π, 3π/2]
                       EBC: ['fwd', 'right', 'bkwd', 'left']
                       RBC: ['center', 'temporal', 'surround', 'nasal']
    r_max            : float  maximum radius (cm)
    shade_off_retina : bool   if True, shade the region beyond ±120° from
                       theta=0 (the ~40% of the polar plot that falls outside
                       the mouse's estimated visual field)
    """
    # Put theta=0 at the top (12 o'clock) so that 'fwd' / 'center' faces up.
    ax.set_theta_zero_location('N')
    # Use CW direction: head_yaw_deg is computed in image pixel coords (y-DOWN),
    # so angles increase clockwise (0=East, 90=South, 270=North).  Ray bin i
    # therefore sits i*ray_width degrees CW from the head/gaze direction.
    # CW plot direction ensures bin i lands on the correct (CW) side of the plot.
    ax.set_theta_direction(-1)

    ax.set_yticks([r_max * 0.5, r_max])
    ax.set_yticklabels([f'{r_max * 0.5:.0f}', f'{r_max:.0f} cm'], fontsize=6)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    if labels is not None:
        ax.set_xticklabels(labels, fontsize=8)
    ax.tick_params(axis='both', labelsize=7)

    if shade_off_retina:
        # Shade angles beyond ±120° from center-of-retina (theta=0).
        # In CCW convention that is the arc from 120° → 240° going through 180°.
        off_theta = np.deg2rad(np.linspace(120, 240, 200))
        ax.fill_between(off_theta, 0, r_max, color='gray', alpha=0.4, zorder=3,
                        linewidth=0)


def _add_shuffle_inset(fig, criteria, passes, color, rect):
    """
    Add a small inset axis showing observed MRL vs shuffled null distribution.

    Parameters
    ----------
    fig      : matplotlib Figure
    criteria : dict  from cell_criteria (has 'shuffled_mrls', 'mean_resultant_length',
                    'mrl_99_pctl', 'mrl_pass')
    passes   : bool
    color    : str  line colour for observed MRL
    rect     : list [left, bottom, width, height] in figure coordinates
    """
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

    # parser = argparse.ArgumentParser(description='Run boundary tuning analysis.')
    # parser.add_argument(
    #     'path', type=str, help='Path to preprocessed .h5 file')        default=)
    # args = parser.parse_args()
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
    
    version_key = 'v02'

    # if not os.path.exists(args.path):
    #     raise FileNotFoundError(f"File not found: {args.path}")
    if pdf_only:
        print(f'Loading results from {path}...')
        # Initialize with empty data since we are loading results
        bt = BoundaryTuning({}) 
        bt.load_results(args.path)
        
        base_path = os.path.splitext(path)[0]
        # Remove _boundary_results if present to avoid duplication in filename
        if base_path.endswith('_boundary_results'):
            base_path = base_path[:-17]

        # bt.make_summary_pdf(f"{base_path}_boundary_summary{version_key}.pdf")
        bt.make_detailed_pdf(f"{base_path}_EBC_detailed{version_key}.pdf", f"{base_path}_RBC_detailed{version_key}.pdf")
        
    else:
        
        print(f'Loading preprocessed data from {path}...')
        data = fm2p.read_h5(path)

        print(f'Loading {path}...')
        data = fm2p.read_h5(path)
        bt = BoundaryTuning(data)
        bt.identify_responses_both(use_light=True)

        base_path = os.path.splitext(path)[0]
        bt.save_results_combined(f"{base_path}_boundary_results{version_key}.h5")
        bt.make_summary_pdf(f"{base_path}_boundary_summary{version_key}.pdf")
        bt.make_diagnostic_figs(f"{base_path}_boundary_diagnostics{version_key}")
        bt.make_detailed_pdf(f"{base_path}_EBC_detailed.pdf", f"{base_path}_RBC_detailed{version_key}.pdf")


if __name__ == '__main__':

    all_fm_preproc_files = fm2p.find('*DMM*fm*preproc.h5', '/home/dylan/Storage/freely_moving_data/_V1PPC')
    for f in tqdm(all_fm_preproc_files):
        boundary_tuning(f)

