
import os
import numpy as np
from scipy.stats import binned_statistic_2d
from scipy.ndimage import uniform_filter
from numpy import log2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.backends.backend_pdf import PdfPages

import fm2p

class SpatialCoding():

    def __init__(self, cfg):

        self.cfg = cfg

        self.bin_size = cfg['place_bin_size'] # in cm
        self.sd_thresh = cfg['place_sd_thresh']
        # number of pixels between coordinates below which is considered 'not moving'
        self.move_thresh = cfg['running_thresh']
        self.likelihood_thresh = cfg['likelihood_thresh']

        self.nCells = 0
        self.x = None
        self.y = None
        self.spikes = None


    def add_data(self, topdown_dict, arena_dict, dFF_transients):

        self.x = topdown_dict['x']
        self.y = topdown_dict['y']
        # Ensure that speed is the same length as position data
        self.speed = np.append([
            topdown_dict['speed'], topdown_dict['speed'][-1]
        ])
        self.useF = self.speed > self.move_thresh
        self.dFF_transients = dFF_transients
        self.nCells = np.size(dFF_transients, 0)
        self.arena = arena_dict


    def calc_place_cells(self):

        assert self.nCells > 0
        assert self.dFF_transients is not None
        assert self.x is not None
        assert self.y is not None
        assert self.spikes is not None
        assert self.arena is not None

        dFF_transients = self.dFF_transients.copy()
        x = self.x.copy()[self.useF]
        y = self.y.copy()[self.useF]

        # bin size in units of pixels
        bin_size_pxls = self.bin_size / self.arena['pxl_size']

        x_edges = np.linspace(
            np.floor(np.min(x)),
            np.ceil(np.max(x)),
            num=(np.ceil(np.max(x))-np.floor(np.min(x))) / bin_size_pxls
        )
        y_edges = np.linspace(
            np.floor(np.min(y)),
            np.ceil(np.max(y)),
            num=(np.ceil(np.max(y))-np.floor(np.min(y))) / bin_size_pxls
        )
        num_bins_x = len(x_edges) - 1
        num_bins_y = len(y_edges) - 1

        # hist of occupancy
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

            # avoid dividing by zero
            occupancy_map[occupancy_map == 0] = np.nan

            activity_maps[c,:,:] = actmap_ / occupancy_map

        self.occupancy_map = occupancy_map
        self.activity_maps = activity_maps


    def check_place_cell_reliability(self, dFF_transients=None, x=None, y=None):
        
        if dFF_transients is None:
            dFF_transients = self.dFF_transients.copy()
        if x is None:
            x = self.x.copy()[self.useF]
        if y is None:
            y = self.y.copy()[self.useF]
        
        cohens_d = self.cfg['cohens_d']
        bout_duration = self.cfg['bout_duration']
        nShuffles = self.cfg['n_pc_shuffles']
        bin_size = self.bin_size / self.arena['pxl_size']  # convert to pixels
        n_bouts = self.cfg['n_bouts']

        nCells, nFrames = np.shape(dFF_transients)

        xEdges = np.arange(np.floor(x.min()), np.ceil(x.max()) + bin_size, bin_size)
        yEdges = np.arange(np.floor(y.min()), np.ceil(y.max()) + bin_size, bin_size)
        nBinsX = len(xEdges) - 1
        nBinsY = len(yEdges) - 1
        nBins = nBinsX * nBinsY

        xBin = np.digitize(x, xEdges) - 1
        yBin = np.digitize(y, yEdges) - 1

        # occupancy
        valid = (xBin >= 0) & (yBin >= 0) & (xBin < nBinsX) & (yBin < nBinsY)
        occupancyMap = np.zeros((nBinsY, nBinsX))
        for xb, yb in zip(xBin[valid], yBin[valid]):
            occupancyMap[yb, xb] += 1
        occupancyFlat = occupancyMap.flatten()
        p_i = occupancyFlat / np.sum(occupancyFlat)

        # bin index per frame
        binIdx = np.zeros(nFrames, dtype=int)
        for i in range(nFrames):
            if 0 <= xBin[i] < nBinsX and 0 <= yBin[i] < nBinsY:
                binIdx[i] = yBin[i] * nBinsX + xBin[i]
            else:
                binIdx[i] = -1  # invalid bin

        # spatial information
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
            spatialInfo[c] = np.sum(p_i * (r_i / r_bar) * np.log2(r_i / r_bar))

        # Shuffled SI
        shuffledSI = np.zeros((nShuffles, nCells))
        for s in range(nShuffles):
            for c in range(nCells):
                shuffled_trace = np.roll(dFF_transients[c,:], np.random.randint(nFrames))
                r_i = np.zeros(nBins)
                for b in range(nBins):
                    valid_idx = (binIdx == b)
                    if occupancyFlat[b] > 0:
                        r_i[b] = np.sum(shuffled_trace[valid_idx]) / occupancyFlat[b]
                r_i[r_i == 0] = np.finfo(float).eps
                r_bar = np.sum(p_i * r_i)
                shuffledSI[s, c] = np.sum(p_i * (r_i / r_bar) * np.log2(r_i / r_bar))

        sigSI = spatialInfo > np.percentile(shuffledSI, 85, axis=0)

        # Consistency via Cohen's d
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

        # place field contiguity
        hasPlaceField = np.zeros(nCells, dtype=bool)
        thresholdFrac = 0.4

        for c in range(nCells):
            rMap = activityFlat[:, c].reshape((nBinsY, nBinsX))
            rThresh = np.mean(rMap) * (1 + thresholdFrac)
            above = rMap > rThresh

            for i in range(nBinsY - 1):
                for j in range(nBinsX - 1):
                    block = above[i:i+2, j:j+2]
                    if np.all(block):
                        hasPlaceField[c] = True
                        break
                if hasPlaceField[c]:
                    break


        criteria_dict = {
            'spatial_info': sigSI,
            'reliability': sigRel,
            'has_place_field': hasPlaceField
        }
        place_cell_inds = sigSI & sigRel & hasPlaceField
        print(f'Identified {np.sum(place_cell_inds)} place cells out of {nCells}.')

        self.place_cell_inds = place_cell_inds
        self.criteria_dict = criteria_dict

        return place_cell_inds, criteria_dict
    

def plot_place_cell_maps(cellIndices, activity_maps, savedir, sigma=1):
    # sigma is std of gaussian filter


    pdf = PdfPages(os.path.join(savedir, 'place_cell_maps_{}.pdf').format(fm2p.fmt_now(c=True)))

    panel_width = 4
    panel_height = 5

    # valid_PCs is a boolean array; get indices of True values
    nPlaceCells = len(cellIndices)

    for batchStart in range(0, nPlaceCells, panel_width*panel_height):
        batchEnd = min(batchStart + panel_width*panel_height, nPlaceCells)

        fig, axs = plt.subplots(panel_width, panel_height, figsize=(15, 10))
        axs = axs.flatten()

        for i, ax in enumerate(axs[:batchEnd - batchStart]):

            cell_idx = cellIndices[batchStart + i]
            smoothedMap = gaussian_filter(activity_maps[cell_idx,:,:], sigma=sigma)

            im = ax.imshow(smoothedMap, cmap='viridis')
            ax.axis('off')
            ax.set_title(f'Cell {cell_idx}')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide any unused subplots
        for j in range(batchEnd - batchStart, len(axs)):
            axs[j].axis('off')

        fig.suptitle(f'Place Cells {cellIndices[batchStart]}–{cellIndices[batchEnd - 1]} of {nPlaceCells}')
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        pdf.savefig()


        


