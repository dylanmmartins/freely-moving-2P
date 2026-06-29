# -*- coding: utf-8 -*-
"""
fm2p/utils/place_cells.py

Place-cell analysis: spatial rate maps, spatial information, and reliability scoring.

Classes
-------
SpatialCoding
    Compute 2D activity maps and score cells as place cells.

Functions
---------
plot_place_cell_maps
    Save a multi-page PDF of smoothed place-field maps.


DMM, June 2025
"""

import os
import numpy as np
from scipy.stats import binned_statistic_2d
from scipy.ndimage import uniform_filter
from numpy import log2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.backends.backend_pdf import PdfPages

from .time import fmt_now


class SpatialCoding():
    """ Compute 2D occupancy-normalized activity maps and classify place cells.

    Scoring requires three independent criteria to all pass:
    spatial information > shuffled 85th percentile, split-bout Cohen's d >
    threshold, and at least one 2x2 contiguous block above the firing threshold.
    """

    def __init__(self, cfg):
        """ Initialize thresholds from a config dict.

        Parameters
        ----------
        cfg : dict
            Must contain 'place_bin_size' (cm), 'place_sd_thresh', 'running_thresh',
            'likelihood_thresh', 'cohens_d', 'bout_duration', 'n_pc_shuffles', 'n_bouts'.
        """

        self.cfg = cfg

        self.bin_size = cfg['place_bin_size']  # cm
        self.sd_thresh = cfg['place_sd_thresh']
        # Displacement threshold in pixels below which is treated as not moving.
        self.move_thresh = cfg['running_thresh']
        self.likelihood_thresh = cfg['likelihood_thresh']

        self.nCells = 0
        self.x = None
        self.y = None
        self.spikes = None

    def add_data(self, topdown_dict, arena_dict, dFF_transients):
        """ Attach position and fluorescence data before computing maps.

        Parameters
        ----------
        topdown_dict : dict
            Must contain 'x', 'y', 'speed' arrays.
        arena_dict : dict
            Must contain 'pxl_size' (pixels per cm).
        dFF_transients : np.ndarray, shape (N_cells, N_frames)
        """

        self.x = topdown_dict['x']
        self.y = topdown_dict['y']
        # Speed array is one frame shorter than position; duplicate last value to match.
        self.speed = np.append([
            topdown_dict['speed'], topdown_dict['speed'][-1]
        ])
        self.useF = self.speed > self.move_thresh
        self.dFF_transients = dFF_transients
        self.nCells = np.size(dFF_transients, 0)
        self.arena = arena_dict

    def calc_place_cells(self):
        """ Compute 2D activity maps for all cells.

        Returns (sets self.occupancy_map and self.activity_maps as side effects).

        Returns
        -------
        occupancy_map : np.ndarray
        activity_maps : np.ndarray, shape (N_cells, N_x_bins, N_y_bins)
        """

        assert self.nCells > 0
        assert self.dFF_transients is not None
        assert self.x is not None
        assert self.y is not None
        assert self.spikes is not None
        assert self.arena is not None

        dFF_transients = self.dFF_transients.copy()
        x = self.x.copy()[self.useF]
        y = self.y.copy()[self.useF]

        bin_size_pxls = self.bin_size / self.arena['pxl_size']

        x_edges = np.linspace(
            np.floor(np.min(x)),
            np.ceil(np.max(x)),
            num=(np.ceil(np.max(x)) - np.floor(np.min(x))) / bin_size_pxls
        )
        y_edges = np.linspace(
            np.floor(np.min(y)),
            np.ceil(np.max(y)),
            num=(np.ceil(np.max(y)) - np.floor(np.min(y))) / bin_size_pxls
        )
        num_bins_x = len(x_edges) - 1
        num_bins_y = len(y_edges) - 1

        occupancy_map, occ_x, occ_y = np.histogram2d(
            x,
            y,
            bins=[x_edges, y_edges]
        )

        activity_maps = np.zeros([
            self.nCells,
            np.size(occ_x),
            np.size(occ_y)
        ])

        for c in range(self.nCells):

            actmap_ = np.histogram2d(
                x,
                y,
                bins=[x_edges, y_edges],
                weights=dFF_transients[c, self.useF]
            )

            # Replace zero-occupancy bins with NaN to avoid spurious rates.
            occupancy_map[occupancy_map == 0] = np.nan

            activity_maps[c, :, :] = actmap_ / occupancy_map

        self.occupancy_map = occupancy_map
        self.activity_maps = activity_maps

    def check_place_cell_reliability(self, dFF_transients=None, x=None, y=None):
        """ Score each cell on spatial information, split-bout reliability, and field contiguity.

        Parameters
        ----------
        dFF_transients : np.ndarray or None
            (N_cells, N_frames); defaults to self.dFF_transients.
        x : np.ndarray or None
            Position arrays; default to self.x/y filtered by self.useF.
        y : np.ndarray or None

        Returns
        -------
        place_cell_inds : np.ndarray of bool
        criteria_dict : dict
            Keys: 'place_cell_spatial_info', 'place_cell_reliability', 'has_place_field'.
        """

        if dFF_transients is None:
            dFF_transients = self.dFF_transients.copy()
        if x is None:
            x = self.x.copy()[self.useF]
        if y is None:
            y = self.y.copy()[self.useF]

        cohens_d = self.cfg['cohens_d']
        bout_duration = self.cfg['bout_duration']
        nShuffles = self.cfg['n_pc_shuffles']
        bin_size = self.bin_size / self.arena['pxl_size']  # pixels
        n_bouts = self.cfg['n_bouts']

        nCells, nFrames = np.shape(dFF_transients)

        xEdges = np.arange(np.floor(x.min()), np.ceil(x.max()) + bin_size, bin_size)
        yEdges = np.arange(np.floor(y.min()), np.ceil(y.max()) + bin_size, bin_size)
        nBinsX = len(xEdges) - 1
        nBinsY = len(yEdges) - 1
        nBins = nBinsX * nBinsY

        xBin = np.digitize(x, xEdges) - 1
        yBin = np.digitize(y, yEdges) - 1

        valid = (xBin >= 0) & (yBin >= 0) & (xBin < nBinsX) & (yBin < nBinsY)
        occupancyMap = np.zeros((nBinsY, nBinsX))
        for xb, yb in zip(xBin[valid], yBin[valid]):
            occupancyMap[yb, xb] += 1
        occupancyFlat = occupancyMap.flatten()
        p_i = occupancyFlat / np.sum(occupancyFlat)

        binIdx = np.zeros(nFrames, dtype=int)
        for i in range(nFrames):
            if 0 <= xBin[i] < nBinsX and 0 <= yBin[i] < nBinsY:
                binIdx[i] = yBin[i] * nBinsX + xBin[i]
            else:
                binIdx[i] = -1

        activityFlat = np.zeros((nBins, nCells))
        for c in range(nCells):
            r_i = np.zeros(nBins)
            for b in range(nBins):
                valid_idx = (binIdx == b)
                if occupancyFlat[b] > 0:
                    r_i[b] = np.sum(dFF_transients[c, valid_idx]) / occupancyFlat[b]
            activityFlat[:, c] = r_i

        spatialInfo = np.zeros(nCells)
        for c in range(nCells):
            r_i = activityFlat[:, c]
            r_i[r_i == 0] = np.finfo(float).eps
            r_bar = np.sum(p_i * r_i)
            spatialInfo[c] = np.sum(p_i * (r_i / r_bar) * log2(r_i / r_bar))

        shuffledSI = np.zeros((nShuffles, nCells))
        for s in range(nShuffles):
            for c in range(nCells):
                shuffled_trace = np.roll(dFF_transients[c, :], np.random.randint(nFrames))
                r_i = np.zeros(nBins)
                for b in range(nBins):
                    valid_idx = (binIdx == b)
                    if occupancyFlat[b] > 0:
                        r_i[b] = np.sum(shuffled_trace[valid_idx]) / occupancyFlat[b]
                r_i[r_i == 0] = np.finfo(float).eps
                r_bar = np.sum(p_i * r_i)
                shuffledSI[s, c] = np.sum(p_i * (r_i / r_bar) * log2(r_i / r_bar))

        sigSI = spatialInfo > np.percentile(shuffledSI, 85, axis=0)

        reliability = np.zeros(nCells)

        for c in range(nCells):
            d_values = []
            for _ in range(n_bouts):
                idxA_start = np.random.randint(nFrames - bout_duration + 1)
                idxB_start = np.random.randint(nFrames - bout_duration + 1)
                idxA = np.arange(idxA_start, idxA_start + bout_duration)
                idxB = np.arange(idxB_start, idxB_start + bout_duration)

                aBins = binIdx[idxA]
                bBins = binIdx[idxB]
                aVals = dFF_transients[c, idxA]
                bVals = dFF_transients[c, idxB]

                aAct = np.zeros(nBins)
                bAct = np.zeros(nBins)

                for b in range(nBins):
                    aMask = aBins == b
                    bMask = bBins == b
                    if np.any(aMask):
                        aAct[b] = np.sum(aVals[aMask])
                    if np.any(bMask):
                        bAct[b] = np.sum(bVals[bMask])

                diff_mean = np.mean(aAct) - np.mean(bAct)
                pooled_std = np.sqrt((np.std(aAct) ** 2 + np.std(bAct) ** 2) / 2)
                if pooled_std > 0:
                    d_values.append(diff_mean / pooled_std)
                else:
                    d_values.append(0)

            reliability[c] = np.mean(np.abs(d_values))

        sigRel = reliability > cohens_d

        hasPlaceField = np.zeros(nCells, dtype=bool)
        thresholdFrac = 0.4

        for c in range(nCells):
            rMap = activityFlat[:, c].reshape((nBinsY, nBinsX))
            rThresh = np.mean(rMap) * (1 + thresholdFrac)
            above = rMap > rThresh

            for i in range(nBinsY - 1):
                for j in range(nBinsX - 1):
                    block = above[i:i + 2, j:j + 2]
                    if np.all(block):
                        hasPlaceField[c] = True
                        break
                if hasPlaceField[c]:
                    break

        criteria_dict = {
            'place_cell_spatial_info': sigSI,
            'place_cell_reliability': sigRel,
            'has_place_field': hasPlaceField
        }
        place_cell_inds = sigSI & sigRel & hasPlaceField
        print('Identified {} place cells out of {}.'.format(np.sum(place_cell_inds), nCells))

        self.place_cell_inds = place_cell_inds
        self.criteria_dict = criteria_dict

        return place_cell_inds, criteria_dict


def plot_place_cell_maps(cellIndices, activity_maps, savedir, sigma=1):
    """ Save a multi-page PDF of smoothed place-field maps.

    Parameters
    ----------
    cellIndices : array-like of int
        Indices into activity_maps for cells to plot.
    activity_maps : np.ndarray, shape (N_cells, N_x_bins, N_y_bins)
    savedir : str
        Directory to write the PDF.
    sigma : float
        Standard deviation of the Gaussian smoothing kernel (bins).
    """

    pdf = PdfPages(os.path.join(savedir, 'place_cell_maps_{}.pdf').format(fmt_now(c=True)))

    panel_width = 4
    panel_height = 5

    nPlaceCells = len(cellIndices)

    for batchStart in range(0, nPlaceCells, panel_width * panel_height):
        batchEnd = min(batchStart + panel_width * panel_height, nPlaceCells)

        fig, axs = plt.subplots(panel_width, panel_height, figsize=(15, 10))
        axs = axs.flatten()

        for i, ax in enumerate(axs[:batchEnd - batchStart]):

            cell_idx = cellIndices[batchStart + i]
            smoothedMap = gaussian_filter(activity_maps[cell_idx, :, :], sigma=sigma)

            im = ax.imshow(smoothedMap, cmap='viridis')
            ax.axis('off')
            ax.set_title('Cell {}'.format(cell_idx))
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for j in range(batchEnd - batchStart, len(axs)):
            axs[j].axis('off')

        fig.suptitle('Place Cells {} to {} of {}'.format(
            cellIndices[batchStart], cellIndices[batchEnd - 1], nPlaceCells))
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        pdf.savefig()
