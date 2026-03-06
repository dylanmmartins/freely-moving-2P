# Automatic visual field signmap HVA segmentation adapted from
# https://github.com/zhuangjun1981/retinotopic_mapping.git.

# The code here replicates the HVA segmentation pipeline used in:
# - Zhuang et al. eLife 2017 (https://doi.org/10.7554/eLife.18372)
# - Waters et al. Plos One 2019 (https://doi.org/10.1371/journal.pone.0213924)

# Their repo is written in Python 2.x. This is a self-contained
# Python 3.x implementation with additional functions and classes
# for working on the contours.

# LDR 02/26

import os
import copy
from io import BytesIO
from typing import Dict, Optional
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, Button
import cv2
import scipy.ndimage as ni
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat
import scipy.sparse as sparse
import re
from skimage import segmentation
from skimage import measure
import skimage.morphology as sm
from itertools import combinations
from operator import itemgetter
import tkinter as tk
from tkinter import messagebox, simpledialog

import requests
import tifffile


DEFAULT_PARAMS = {
    # These are the default parameters used in
    # Zhuang et al. 2017 and Waters et al. 2019
    "phaseMapFilterSigma": 0.5,
    "signMapFilterSigma": 8.0,
    "signMapThr": 0.23,
    "eccMapFilterSigma": 15.0,
    "splitLocalMinCutStep": 5.0,
    "closeIter": 3,
    "openIter": 3,
    "dilationIter": 15,
    "borderWidth": 1,
    "smallPatchThr": 100,
    "visualSpacePixelSize": 0.5,
    "visualSpaceCloseIter": 15,
    "splitOverlapThr": 1.1,
    "mergeOverlapThr": 0.1,
}


def default_params():
    return dict(DEFAULT_PARAMS)


def update_params(params, **tweaks):
    updated = dict(params)
    updated.update(tweaks)
    return updated


def int2str(num, length=None):
    """
    generate a string representation for a integer with a given length
    :param num: input number
    :param length: length of the string
    :return: string representation of the integer
    """
    rawstr = str(int(num))
    if length is None or length == len(rawstr):
        return rawstr
    elif length < len(rawstr):
        raise ValueError('Length of the number is longer than the defined display length!')
    else:
        return '0' * (length - len(rawstr)) + rawstr


def get_trace(movie, mask, maskMode='binary'):
    """
    get a trace across a movie with averaged value in a mask

    maskMode: 'binary': ones in roi, zeros outside
              'binaryNan': ones in roi, nans outside
              'weighted': weighted values in roi, zeros outside (note: all pixels equal to zero will be considered outside roi
              'weightedNan': weighted values in roi, nans outside
    """

    if maskMode == 'binary':
        if np.where(mask == 0)[0].size + np.where(mask == 1)[0].size < mask.size:
            raise ValueError('Binary mask should only contain zeros and ones!!')
        else:
            finalMask = np.array(mask.astype(float))
            pixelNum = np.sum(finalMask.flatten())
    elif maskMode == 'binaryNan':
        if np.sum(np.isnan(mask).flatten()) + np.where(mask == 1)[0].size < mask.size:
            raise ValueError('BinaryNan mask should only contain nans and ones!!')
        else:
            finalMask = np.ones(mask.shape, dtype=float)
            finalMask[np.isnan(mask)] = 0
            pixelNum = mask.size - np.sum(np.isnan(mask).flatten())
    elif maskMode == 'weighted':
        if np.isnan(mask).any():
            raise ValueError('Weighted mask should not contain nan(s)!!')
        else:
            finalMask = np.array(mask.astype(float))
            pixelNum = mask.size - np.where(mask == 0)[0].size
    elif maskMode == 'weightedNan':
        finalMask = np.array(mask.astype(float))
        finalMask[np.isnan(mask)] = 0
        pixelNum = mask.size - np.where(finalMask == 0)[0].size
    else:
        raise LookupError('maskMode not understood. Should be one of "binary", "binaryNan", "weighted", "weightedNan".')

    trace = np.sum(np.multiply(movie, finalMask), (1, 2)) / pixelNum

    return trace


def plotVisualCoverage(visualSpace, pixelSize, altStart=-40, aziStart=-20, tickSpace=10, plotAxis=None):
    """
    plot visual space in given plotAxis
    """

    pixelSize = float(pixelSize)

    altRange = np.arange(altStart, altStart + pixelSize * visualSpace.shape[0], pixelSize)
    aziRange = np.arange(aziStart, aziStart + pixelSize * visualSpace.shape[1], pixelSize)

    tickPixelSpace = int(tickSpace / pixelSize)
    xtickInd = np.arange(int((aziStart % tickSpace) / pixelSize),
                         visualSpace.shape[1],
                         tickPixelSpace)
    ytickInd = np.arange(int((altStart % tickSpace) / pixelSize),
                         visualSpace.shape[0],
                         tickPixelSpace)

    xtickLabel = [str(int(round(aziRange[x]))) for x in xtickInd]
    ytickLabel = [str(int(round(altRange[x]))) for x in ytickInd]

    if not plotAxis:
        f = plt.figure()
        ax = f.add_subplot(111)
    else:
        ax = plotAxis
    ax.imshow(visualSpace, cmap='hot_r', interpolation='nearest')
    ax.invert_yaxis()
    ax.set_aspect(1)
    ax.set_xticks(xtickInd)
    ax.set_xticklabels(xtickLabel)
    ax.set_yticks(ytickInd)
    ax.set_yticklabels(ytickLabel)


def localMin(eccMap, binSize):
    """
    find local minimum of eccenticity map (in degree), with binning by binSize
    in degree
    """

    eccMap2 = np.array(eccMap)
    cutStep = np.arange(np.nanmin(eccMap2[:]) - binSize,
                        np.nanmax(eccMap2[:]) + binSize * 2,
                        binSize)
    NumOfMin = 0
    i = 0
    while (NumOfMin <= 1) and (i < len(cutStep)):
        currThr = cutStep[i]
        marker = np.zeros(eccMap.shape, dtype=int)
        marker[eccMap2 <= (currThr)] = 1
        marker, NumOfMin = ni.measurements.label(marker)
        i = i + 1

    return marker


def array_nor(A):
    """
    normalize a np.array to the scale [0, 1]
    """

    B = A.astype(float)
    return (B - np.amin(B)) / (np.amax(B) - np.amin(B))


def dilationPatches2(rawPatches, dilationIter=20, borderWidth=1):  # pixel width of the border after dilation

    """
    dilation patched in a given area untill the border between them are as
    narrow as defined by 'borderWidth'.
    """

    total_area = ni.binary_dilation(rawPatches, iterations=dilationIter).astype(int)
    patchBorder = total_area - rawPatches

    # thinning patch borders
    patchBorder = sm.skeletonize(patchBorder)

    # thickening patch borders
    if borderWidth > 1:
        patchBorder = ni.binary_dilation(patchBorder, iterations=borderWidth - 1).astype(int)

    # genertating new patches
    newPatches = np.multiply(-1 * (patchBorder - 1), total_area)

    # removing small edges
    labeledPatches, patchNum = ni.label(newPatches)

    newPatches2 = np.zeros(newPatches.shape, dtype=int)

    for i in range(1, patchNum + 1):
        currPatch = np.zeros(labeledPatches.shape, dtype=int)
        currPatch[labeledPatches == i] = 1
        currPatch[labeledPatches != i] = 0

        if (np.sum(np.multiply(currPatch, rawPatches)[:]) > 0):
            newPatches2[currPatch == 1] = 1

    return newPatches2


def singleMergePatches(array1, array2, borderWidth=2):
    """
    merge two binary patches with borderWidth no greater than borderWidth
    """

    sp = array1 + array2
    spc = ni.binary_closing(sp, iterations=(borderWidth)).astype(np.int8)

    _, patchNum = ni.measurements.label(spc)
    if patchNum > 1:
        raise LookupError('These two two patches are too far apart!!!')
    else:
        return spc


def labelPatches(patchmap, signMap):
    """
    from a segregated patchmap generate a dictionary with each entry represents
    a single patch, sorted by area
    """

    labeledPatches, patchNum = ni.label(patchmap)

    # list of area of every patch, first column: patch label, second column: area
    patchArea = np.zeros((patchNum, 2), dtype=int)

    for i in range(1, patchNum + 1):
        currPatch = np.zeros(labeledPatches.shape, dtype=int)
        currPatch[labeledPatches == i] = 1
        currPatch[labeledPatches != i] = 0
        patchArea[i - 1] = [i, np.sum(currPatch[:])]

    # sort patches by the area, from largest to the smallest
    sortArea = patchArea[patchArea[:, 1].argsort(axis=0)][::-1, :]

    patches = {}
    for i, ind in enumerate(sortArea[:, 0]):
        currPatch = np.zeros(labeledPatches.shape, dtype=int)
        currPatch[labeledPatches == ind] = 1
        currPatch[labeledPatches != ind] = 0
        currSignPatch = np.multiply(currPatch, signMap)

        if np.sum(currSignPatch[:]) > 0:
            currSign = 1
        elif np.sum(currSignPatch[:]) < 0:
            currSign = -1
        else:
            raise LookupError('This patch has no visual Sign!!')

        patchname = 'patch' + int2str(i, 2)

        patches.update({patchname: Patch(currPatch, currSign)})

    return patches


def sortPatches(patchDict):
    """
    from a patch dictionary generate an new dictionary with patches sorted by there area
    """

    patches = []
    newPatchDict = {}

    for key, value in patchDict.items(): # change to python 3
        patches.append((value, value.getArea()))

    patches = sorted(patches, key=lambda a: a[1], reverse=True)

    for i, item in enumerate(patches):
        patchName = 'patch' + int2str(i + 1, 2)

        newPatchDict.update({patchName: item[0]})

    return newPatchDict


def plotPatches(patches, plotaxis=None, zoom=1, alpha=0.5, markersize=5):
    """
    plot a patches in a patch dictionary
    """

    if plotaxis == None:
        f = plt.figure()
        plotaxis = f.add_axes([1, 1, 1, 1])

    imageHandle = {}
    for key, value in patches.items():

        if zoom > 1:
            currPatch = Patch(ni.zoom(value.array, zoom, order=0), value.sign)
        else:
            currPatch = value

        h = plotaxis.imshow(currPatch.getSignedMask(), vmax=1, vmin=-1, interpolation='nearest', alpha=alpha, cmap='jet')
        plotaxis.plot(currPatch.getCenter()[1], currPatch.getCenter()[0], '.k', markersize=markersize * zoom)
        imageHandle.update({'handle_' + key: h})

    plotaxis.set_xlim([0, currPatch.array.shape[1] - 1])
    plotaxis.set_ylim([currPatch.array.shape[0] - 1, 0])
    # plotaxis.set_axis_off()
    return imageHandle


def plotVFSWithPatches(signMapf, finalPatches, labels=True):
    """
    Plot sign map with patch borders overlaid, matching the notebook cell.
    """

    border_mask = np.zeros(signMapf.shape)

    for key in finalPatches:
        area_mask = finalPatches[key].array
        border_mask = border_mask + area_mask - ni.binary_erosion(area_mask, iterations=1)

    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)
    ax1.imshow(signMapf + 10 * border_mask, clim=(-1, 1), cmap='jet')

    if labels:
        for key in finalPatches:
            center = finalPatches[key].getCenter()
            ax1.text(center[1], center[0], key, color='white', fontsize=8,
                     ha='center', va='center')
    plt.show()


def getPixelVisualCenter(self, altMap, aziMap):
        """
        get the center coordinates in visual response space for all pixels in
        this cortical patch
        """

        altPatch = self.array * altMap
        meanAlt = np.mean(altPatch[altPatch != 0])
        aziPatch = self.array * aziMap
        meanAzi = np.mean(aziPatch[aziPatch != 0])

        return meanAlt, meanAzi


def eccentricityMap(altMap, aziMap, altCenter, aziCenter):
    """
    calculate eccentricity map of with defined center

    altMap, aziMap, altCenter, aziCenter: in degree

    eccentricity map is returned in degree
    """

    altMap2 = altMap * np.pi / 180
    aziMap2 = aziMap * np.pi / 180

    altCenter2 = altCenter * np.pi / 180
    aziCenter2 = aziCenter * np.pi / 180

    eccMap = np.zeros(altMap.shape)
    eccMap[:] = np.nan
    #    for i in xrange(altMap.shape[0]):
    #        for j in xrange(altMap.shape[1]):
    #            alt = altMap2[i,j]
    #            azi = aziMap2[i,j]
    #            eccMap[i,j] = np.arctan(np.sqrt(np.tan(alt-altCenter2)**2 + ((np.tan(azi-aziCenter2)**2)/(np.cos(alt-altCenter2)**2))))
    eccMap = np.arctan(
        np.sqrt(
            np.square(np.tan(altMap2 - altCenter2))
            +
            np.square(np.tan(aziMap2 - aziCenter2)) / np.square(np.cos(altMap2 - altCenter2))
        )
    )

    eccMap = eccMap * 180 / np.pi
    return eccMap


def is_adjacent(array1, array2, borderWidth=2):
    """
    decide if two patches are adjacent within border width
    """

    p1d = ni.binary_dilation(array1, iterations=borderWidth - 1).astype(np.int8)
    p2d = ni.binary_dilation(array2, iterations=borderWidth - 1).astype(np.int8)

    if np.amax(p1d + p2d) > 1:
        return True
    else:
        return False


def adjacentPairs(patches, borderWidth=2):
    """
    return all the patch pairs with same visual sign and sharing border
    """

    keyList = patches.keys()
    pairKeyList = []

    for pair in combinations(keyList, 2):
        patch1 = patches[pair[0]]
        patch2 = patches[pair[1]]

        if (is_adjacent(patch1.array, patch2.array, borderWidth=borderWidth)) and (patch1.sign == patch2.sign):
            pairKeyList.append(pair)

    return pairKeyList


def plotPairedPatches(patch1, patch2, altMap, aziMap, title, pixelSize=1, closeIter=None):
    visualSpace1, area1, _, _ = patch1.getVisualSpace(altMap=altMap,
                                                      aziMap=aziMap,
                                                      pixelSize=pixelSize,
                                                      closeIter=closeIter)

    visualSpace2, area2, _, _ = patch2.getVisualSpace(altMap=altMap,
                                                      aziMap=aziMap,
                                                      pixelSize=pixelSize,
                                                      closeIter=closeIter)

    visualSpace1 = np.array(visualSpace1, dtype=np.float32)
    visualSpace2 = np.array(visualSpace2, dtype=np.float32)

    visualSpace1[visualSpace1 == 0] = np.nan
    visualSpace2[visualSpace2 == 0] = np.nan

    f = plt.figure()
    f.suptitle(title)
    f_121 = f.add_subplot(121)
    patchPlot1 = f_121.imshow(patch1.getMask(), interpolation='nearest', alpha=0.5, vmax=2, vmin=1)
    patchPlot2 = f_121.imshow(patch2.getMask() * 2, interpolation='nearest', alpha=0.5, vmax=2, vmin=1)
    f_121.set_title('patch1: blue, patch2: red')

    f_122 = f.add_subplot(122)
    areaPlot1 = f_122.imshow(visualSpace1, interpolation='nearest', alpha=0.5, vmax=2, vmin=1)
    areaPlot2 = f_122.imshow(visualSpace2 * 2, interpolation='nearest', alpha=0.5, vmax=2, vmin=1)
    f_122.set_title('area1: %.1f, area2: %.1f (deg^2)' % (area1, area2))
    f_122.invert_yaxis()

    # ---------------------------------------------------------------------------------------------
    # reorganize visual space axis label
    altRange = np.array([np.amin(altMap), np.amax(altMap)])
    aziRange = np.array([np.amin(aziMap), np.amax(aziMap)])
    xlist = np.arange(aziRange[0], aziRange[1], pixelSize)
    ylist = np.arange(altRange[0], altRange[1], pixelSize)

    xtick = []
    xticklabel = []
    i = 0
    while i < len(xlist):
        if int(np.floor(xlist[i])) % 10 == 0:
            xtick.append(i)
            xticklabel.append(str(int(np.floor(xlist[i]))))
            i = int(i + 9 / pixelSize)
        else:
            i = i + 1

    ytick = []
    yticklabel = []
    i = 0
    while i < len(ylist):
        if int(np.floor(ylist[i])) % 10 == 0:
            ytick.append(i)
            yticklabel.append(str(int(np.floor(ylist[i]))))
            i = int(i + 9 / pixelSize)
        else:
            i = i + 1

    f_122.set_xticks(xtick)
    f_122.set_xticklabels(xticklabel)
    f_122.set_yticks(ytick)
    f_122.set_yticklabels(yticklabel)


def getMapsFromMATFile(additional_maps_path, filter=True, params=None):
    """
    Calculate man azimuth and altitude maps from Goard Lab retinotopic mapping
    pipeline output file additional_maps.mat
    """
    maps_data = loadmat(additional_maps_path)

    # Extract data from loaded files - fix the nested structure
    maps_struct = maps_data['maps'][0, 0]

    # Extract individual maps
    altPosMap = maps_struct['VerticalRetinotopy']
    aziPosMap = maps_struct['HorizontalRetinotopy']
    signMap = maps_struct['VFS_raw']

    if filter:
        if params is None:
            # Default sigma fallbacks
            params = {
                      'phaseMapFilterSigma': 0.5,
                      'signMapFilterSigma': 8.
                      }
        altPosMapf = ni.gaussian_filter(altPosMap, params['phaseMapFilterSigma'])
        aziPosMapf = ni.gaussian_filter(aziPosMap, params['phaseMapFilterSigma'])
        signMapf = ni.gaussian_filter(signMap, params['signMapFilterSigma'])
        return altPosMapf, aziPosMapf, signMapf

    return altPosMap, aziPosMap, signMap


def getAltAziMapsFromWaters(ISI1_list,
                            ISI1_altitude_map_stack,
                            ISI1_azimuth_map_stack,
                            remove_nans=True,
                            filter=True,
                            params=None):
    """
    Calculate (filtered) mean azimuth and altitude maps from Water et al. 2019 data
    """

    if params is None:
        # default fallback for smoothening sigma
        params = {
          'phaseMapFilterSigma': 0.5
          }

    # Calculate mean azimuth and altitude maps
    mean_altitude_map = np.copy(ISI1_altitude_map_stack)
    mean_azimuth_map = np.copy(ISI1_azimuth_map_stack)

    for ii in range(len(ISI1_list)):
        altitude_map = mean_altitude_map[ii,:,:]
        azimuth_map = mean_azimuth_map[ii,:,:]

        # convert all values with no data to NaNs
        altitude_map[altitude_map == altitude_map[0,0]] = np.nan
        azimuth_map[azimuth_map == azimuth_map[0,0]] = np.nan

        mean_altitude_map[ii,:,:] = altitude_map
        mean_azimuth_map[ii,:,:] = azimuth_map

    mean_altitude_map = np.nanmean(mean_altitude_map, axis=0)
    mean_azimuth_map = np.nanmean(mean_azimuth_map, axis=0)

    if remove_nans:
        # remove nans before calculating sign maps
        mean_altitude_map[np.isnan(mean_altitude_map)] = 1000
        mean_azimuth_map[np.isnan(mean_azimuth_map)] = 1000

    if filter:
        mean_altitude_map = ni.filters.gaussian_filter(mean_altitude_map,
                                                       params['phaseMapFilterSigma'])
        mean_azimuth_map = ni.filters.gaussian_filter(mean_azimuth_map,
                                                       params['phaseMapFilterSigma'])

    return mean_altitude_map, mean_azimuth_map


def visualSignMap(phasemap1, phasemap2, filter=True, params=None):
    """
    calculate (filtered) visual sign map from two orthogonally oriented phase maps
    """
    # Smoothen altitude and azimuth phase maps
    if params is None:
        # default fallback for smoothening sigma
        params = {
          'phaseMapFilterSigma': 0.5,
          'signMapFilterSigma': 8.
          }

    if phasemap1.shape != phasemap2.shape:
        raise LookupError("'phasemap1' and 'phasemap2' should have same size.")

    gradmap1 = np.gradient(phasemap1)
    gradmap2 = np.gradient(phasemap2)

    # gradmap1 = ni.filters.median_filter(gradmap1,100.)
    # gradmap2 = ni.filters.median_filter(gradmap2,100.)

    graddir1 = np.zeros(np.shape(gradmap1[0]))
    # gradmag1 = np.zeros(np.shape(gradmap1[0]))

    graddir2 = np.zeros(np.shape(gradmap2[0]))
    # gradmag2 = np.zeros(np.shape(gradmap2[0]))

    for i in range(phasemap1.shape[0]):
        for j in range(phasemap2.shape[1]):
            graddir1[i, j] = np.arctan2(gradmap1[1][i, j], gradmap1[0][i, j])
            graddir2[i, j] = np.arctan2(gradmap2[1][i, j], gradmap2[0][i, j])

            # gradmag1[i,j] = np.sqrt((gradmap1[1][i,j]**2)+(gradmap1[0][i,j]**2))
            # gradmag2[i,j] = np.sqrt((gradmap2[1][i,j]**2)+(gradmap2[0][i,j]**2))

    vdiff = np.multiply(np.exp(1j * graddir1), np.exp(-1j * graddir2))

    areamap = np.sin(np.angle(vdiff))

    if filter:
        areamap = ni.filters.gaussian_filter(areamap, params['signMapFilterSigma'])

    return areamap


def getRawPatchMap(signMapf, params, isPlot=False):
    signMapThr = params['signMapThr']
    openIter = params['openIter']
    closeIter = params['closeIter']

    # thresholding filtered signmap
    patchmap = np.zeros(signMapf.shape)
    patchmap[signMapf >= signMapThr] = 1
    patchmap[signMapf <= -1 * signMapThr] = 1
    patchmap[(signMapf < signMapThr) & (signMapf > -1 * signMapThr)] = 0
    patchmap = ni.binary_opening(np.abs(patchmap), iterations=openIter).astype(int)
    patches, patchNum = ni.label(patchmap)

    # closing each patch, then put them together
    patchmap2 = np.zeros(patchmap.shape).astype(int)
    for i in range(patchNum):
        currPatch = np.zeros(patches.shape).astype(int)
        currPatch[patches == i + 1] = 1
        currPatch = ni.binary_closing(currPatch, iterations=closeIter).astype(int)
        patchmap2 = patchmap2 + currPatch

    if isPlot:
        plt.figure()
        plt.imshow(patchmap, vmin=0, vmax=1, cmap='jet', interpolation='nearest')
        plt.colorbar()
        plt.title('raw patchmap')
        plt.gca().set_axis_off()

    return patchmap2


def getRawPatches(signMapf, rawPatchMap, params, isPlot=False):
    rawPatchMap = rawPatchMap
    dilationIter = params['dilationIter']
    borderWidth = params['borderWidth']
    smallPatchThr = params['smallPatchThr']

    patchMapDilated = dilationPatches2(rawPatchMap, dilationIter=dilationIter, borderWidth=borderWidth)

    # generate raw patch dictionary
    rawPatches = labelPatches(patchMapDilated, signMapf)

    rawPatches2 = dict(rawPatches)
    # remove small patches
    for key, value in rawPatches2.items(): # needs to be updated to python 3 formatting
        if (value.getArea() < smallPatchThr):
            rawPatches.pop(key)

    # remove isolated Patches
    rawPatches2 = dict(rawPatches)
    for key in rawPatches2.keys(): # needs to be updated to python 3 formatting
        isTouching = 0
        for key2 in rawPatches2.keys(): # needs to be updated to python 3 formatting
            if key != key2:
                if rawPatches2[key].isTouching(rawPatches2[key2], borderWidth * 2):
                    isTouching = 1
                    break

        if isTouching == 0:
            rawPatches.pop(key)

    rawPatches = sortPatches(rawPatches)

    if isPlot:
        try:
            zoom = 2048 / rawPatches['patch01'].array.shape[0]
        except:
            zoom = 1
        f = plt.figure()
        f_axis = f.add_subplot(111)
        _ = plotPatches(rawPatches, plotaxis=f_axis, zoom=zoom)
        f_axis.set_title('raw patches')
        plt.gca().set_axis_off()
        del _

    return rawPatches


def getDeterminantMap(altPosMapf, aziPosMapf, isPlot=False):
    gradAltMap = np.gradient(altPosMapf)
    gradAziMap = np.gradient(aziPosMapf)

    detMap = np.array([[gradAltMap[0], gradAltMap[1]],
                        [gradAziMap[0], gradAziMap[1]]])

    detMap = detMap.transpose(2, 3, 0, 1)
    detMap = np.abs(np.linalg.det(detMap))

    if isPlot:
        plt.figure()
        plt.imshow(detMap, vmin=0, vmax=1, cmap='hsv', interpolation='nearest')
        plt.colorbar()
        plt.title('determinant map')
        plt.gca().set_axis_off()

    return detMap


def getEccentricityMap(altPosMapf, aziPosMapf, rawPatches, params=None, isPlot=False):

    if params is None:
        # default fallback for eccentricity map sigma
        params = {
            'eccMapFilterSigma': 15.0
        }

    eccMap = np.zeros(altPosMapf.shape)
    eccMapf = np.zeros(altPosMapf.shape)
    eccMap[:] = np.nan
    eccMapf[:] = np.nan

    for key, value in rawPatches.items():
        patchAltC, patchAziC = value.getPixelVisualCenter(altPosMapf, aziPosMapf)
        patchEccMap = eccentricityMap(altPosMapf, aziPosMapf, patchAltC, patchAziC)
        patchEccMapf = ni.filters.uniform_filter(patchEccMap, params['eccMapFilterSigma'])

        eccMap[value.array == 1] = patchEccMap[value.array == 1]
        eccMapf[value.array == 1] = patchEccMapf[value.array == 1]

    if isPlot:
        plt.figure()
        plt.imshow(eccMapf, cmap='jet', interpolation='nearest')
        plt.colorbar()
        plt.title('filtered eccentricity map')
        plt.gca().set_axis_off()

    return eccMap, eccMapf


def splitPatches(altPosMapf,
                 aziPosMapf,
                 eccMapf,
                 rawPatches,
                 detMap,
                 params=None,
                 isPlot=False):

    patches = dict(rawPatches)

    if params is None:
        # Default fallback for parameter values
        params = {
          'splitLocalMinCutStep': 5.,
          'borderWidth': 1,
          'visualSpacePixelSize': 0.5,
          'visualSpaceCloseIter': 15,
          'splitOverlapThr': 1.1
          }

    visualSpacePixelSize = params['visualSpacePixelSize']
    visualSpaceCloseIter = params['visualSpaceCloseIter']
    splitLocalMinCutStep = params['splitLocalMinCutStep']
    splitOverlapThr = params['splitOverlapThr']
    borderWidth = params['borderWidth']

    overlapPatches = []
    newPatchesDict = {}

    for key, value in patches.items():
        visualSpace, AU, _, _ = value.getVisualSpace(altPosMapf,
                                                        aziPosMapf,
                                                        pixelSize=visualSpacePixelSize,
                                                        closeIter=visualSpaceCloseIter)
        AS = value.getSigmaArea(detMap)
        print(key + 'AU=' + str(AU) + ' AS=' + str(AS) + ' ratio=' + str(AS / AU))

        if AS / AU >= splitOverlapThr:

            patchEccMapf = eccMapf * value.getMask()
            patchEccMapf[value.array == 0] = np.nan

            minMarker = localMin(patchEccMapf, splitLocalMinCutStep)
            NumOfMin = np.amax(minMarker)

            if NumOfMin == 1:
                print('Only one local minimum was found!!!')
            elif NumOfMin == 0:
                print('No local minimum was found!!!')
            else:
                print(str(NumOfMin) + ' local minima were found!!!')

                overlapPatches.append(key)

                newPatches = value.split2(patchEccMapf,
                                            patchName=key,
                                            cutStep=splitLocalMinCutStep,
                                            borderWidth=borderWidth,
                                            isPlot=False)

                # plotting splitted patches
                if len(newPatches) > 1:

                    f = plt.figure()
                    f121 = f.add_subplot(121)
                    f121.set_title(key)
                    f122 = f.add_subplot(122)
                    f122.set_title('visual space')
                    currPatchValue = 0
                    for key2, value2 in newPatches.items():
                        currPatchValue += 1
                        currArray = np.array(value2.array, dtype=np.float32)
                        currArray[currArray == 0] = np.nan
                        currArray[currArray == 1] = currPatchValue
                        f121.imshow(currArray, interpolation='nearest', vmin=0, vmax=len(newPatches.keys()))
                        f121.set_axis_off()
                        currVisualSpace, _, _, _ = value2.getVisualSpace(altPosMapf,
                                                                            aziPosMapf,
                                                                            pixelSize=visualSpacePixelSize,
                                                                            closeIter=visualSpaceCloseIter)
                        currVisualSpace = currVisualSpace.astype(np.float32)
                        currVisualSpace[currVisualSpace == 0] = np.nan
                        currVisualSpace[currVisualSpace == 1] = currPatchValue
                        f122.imshow(currVisualSpace, interpolation='nearest', alpha=0.5, vmin=0,
                                    vmax=len(newPatches.keys()))

                    xlabel = np.arange(-20, 120, visualSpacePixelSize)
                    ylabel = np.arange(60, -40, -visualSpacePixelSize)

                    indSpace = int(10. / visualSpacePixelSize)

                    xtickInd = range(0, len(xlabel), indSpace)
                    ytickInd = range(0, len(ylabel), indSpace)

                    xtickLabel = [str(int(xlabel[x])) for x in xtickInd]
                    ytickLabel = [str(int(ylabel[y])) for y in ytickInd]

                    f122.xaxis.set_ticks(xtickInd)
                    f122.xaxis.set_ticklabels(xtickLabel)
                    f122.yaxis.set_ticks(ytickInd)
                    f122.yaxis.set_ticklabels(ytickLabel)

                newPatchesDict.update(newPatches)

    for i in range(len(overlapPatches)):
        patches.pop(overlapPatches[i])

    patches.update(newPatchesDict)

    if isPlot:
        zoom = 1
        f2 = plt.figure()
        f2_111 = f2.add_subplot(111)
        h = plotPatches(patches, plotaxis=f2_111, zoom=zoom)
        f2_111.set_axis_off()
        f2_111.set_title('patches after split')
        f2_111.set_axis_off()

    # self.patchesAfterSplit = patches

    return patches


def mergePatches(patchesAfterSplit,
                 altPosMapf,
                 aziPosMapf,
                 params=None,
                 isPlot=False):

    patches = dict(patchesAfterSplit)

    if params is None:
        # Default fall back parameter values
        params = {
          'borderWidth': 1,
          'smallPatchThr': 100,
          'visualSpacePixelSize': 0.5,
          'visualSpaceCloseIter': 15,
          'mergeOverlapThr': 0.1
          }

    borderWidth = params['borderWidth']
    visualSpacePixelSize = params['visualSpacePixelSize']
    visualSpaceCloseIter = params['visualSpaceCloseIter']
    mergeOverlapThr = params['mergeOverlapThr']
    smallPatchThr = params['smallPatchThr']

    # merging non-overlaping patches
    mergeIter = 1

    # pairs of patches that meet the criterion of merging
    # have 5 columns:
    # first column: key of first patch of the pair
    # second column: key of second patch of the pair
    # third column: merged patch
    # forth column: sum of overlapping ratio of each patch
    # fifth column: negative of unique visual space area (AU) of the merged patch
    mergePairs = []

    while (mergeIter == 1) or (len(mergePairs) > 0):

        print('merge iteration: ' + str(mergeIter))

        mergePairs = []

        # get adjacent pairs
        adjPairs = adjacentPairs(patches, borderWidth=borderWidth + 1)

        for ind, pair in enumerate(adjPairs):  # for every adjacent pair
            patch1 = patches[pair[0]]
            patch2 = patches[pair[1]]

            try:
                # merge these two patches
                currMergedPatch = Patch(singleMergePatches(patch1.array, patch2.array, borderWidth=borderWidth),
                                        sign=patch1.sign)

                # calculate unique area of the merged patch
                _, AU, _, _ = currMergedPatch.getVisualSpace(altPosMapf,
                                                                aziPosMapf,
                                                                pixelSize=visualSpacePixelSize,
                                                                closeIter=visualSpaceCloseIter)

                # calculate the visual space and unique area of the first patch
                visualSpace1, AU1, _, _ = patch1.getVisualSpace(altPosMapf,
                                                                aziPosMapf,
                                                                pixelSize=visualSpacePixelSize,
                                                                closeIter=visualSpaceCloseIter)
                visualSpace1 = visualSpace1.astype(np.uint8)

                # calculate the visual space and unique area of the second patch
                visualSpace2, AU2, _, _ = patch2.getVisualSpace(altPosMapf,
                                                                aziPosMapf,
                                                                pixelSize=visualSpacePixelSize,
                                                                closeIter=visualSpaceCloseIter)
                visualSpace2 = visualSpace2.astype(np.uint8)

                # calculate the overlapping area of these two patches
                sumSpace = visualSpace1 + visualSpace2
                overlapSpace = np.zeros(sumSpace.shape, dtype=int)
                overlapSpace[sumSpace == 2] = 1
                Aoverlap = np.sum(overlapSpace[:]) * (visualSpacePixelSize ** 2)

                # calculate the ratio of overlaping area to the unique area of each patch
                overlapRatio1 = Aoverlap / AU1
                overlapRatio2 = Aoverlap / AU2

                # if both ratios are small than merge overlap threshold definded at the beginning of the file
                if (overlapRatio1 <= mergeOverlapThr) and (overlapRatio2 <= mergeOverlapThr):
                    # put this pair and related information to mergePairs list
                    mergePairs.append([pair[0],
                                        pair[1],
                                        currMergedPatch,
                                        np.max([overlapRatio1, overlapRatio2]),
                                        (-1 * AU)])

                del visualSpace1, visualSpace2, AU1, AU2, sumSpace, overlapSpace, Aoverlap



            except LookupError:
                pass

            del patch1, patch2

        if len(mergePairs) > 0:
            # for each identified patch pair to merge sort them with the sum of two
            # overlap ratios, from smallest to biggest and then sort them with the
            # unique area of merged patches from biggest to smallest
            mergePairs.sort(key=itemgetter(3, 4))

            for ind, value in enumerate(mergePairs):  # for each of these pairs
                patch1 = value[0]
                patch2 = value[1]

                # if both of these two patches are still in the 'patches' dictionary
                if (patch1 in patches.keys()) and (patch2 in patches.keys()):
                    # plot these patches and their visual space
                    # plotPairedPatches(patches[patch1],
                    #                   patches[patch2],
                    #                   altPosMapf,
                    #                   aziPosMapf,
                    #                   title='merge iteation:' + str(
                    #                       mergeIter) + ' patch1:' + patch1 + ' patch2:' + patch2,
                    #                   pixelSize=visualSpacePixelSize,
                    #                   closeIter=visualSpaceCloseIter)

                    # remove these two patches from the 'patches' dictionary
                    patches.pop(patch1)
                    patches.pop(patch2)

                    # add merged patches into the 'patches' dictionary
                    patches.update({patch1 + '+' + patch2[5:]: value[2]})

                    print('merging: ' + patch1 + ' & ' + patch2 + ', overlap ratio: ' + str(value[3]))

        mergeIter = mergeIter + 1

    # remove small patches
    patches2 = dict(patches)
    for key, value in patches2.items():
        if (value.getArea() < smallPatchThr):
            patches.pop(key)

    del patches2

    patchesAfterMerge = patches

    finalPatches = sortPatches(patches)

    if isPlot:
        zoom = 1
        f = plt.figure()
        f111 = f.add_subplot(111)
        h = plotPatches(finalPatches, plotaxis=f111, zoom=zoom)
        f111.set_axis_off()
        f111.set_title('final Patches')

    return patchesAfterMerge, finalPatches


class Patch(object):
    def __init__(self, patchArray, sign):

        if isinstance(patchArray, sparse.coo_matrix):
            self.sparseArray = patchArray.astype(np.uint8)
        else:
            arr = patchArray.astype(np.int8)
            arr[arr > 0] = 1
            arr[arr == 0] = 0
            self.sparseArray = sparse.coo_matrix(arr)

        if sign == 1 or sign == 0 or sign == -1:
            self.sign = int(sign)
        else:
            raise ValueError('Sign should be -1, 0 or 1!')

    @property
    def array(self):
        return self.sparseArray.toarray()

    def getCenter(self):
        """
        return the coordinates of the center of a patch
        [rowIndex, columnIndex]
        """
        pixels = np.argwhere(self.array)
        center = np.mean(pixels.astype(np.float32), axis=0)
        return np.round(center).astype(int)

    def getArea(self):
        """
        return pixel number in the patch
        """
        return np.sum(self.array[:])

    def getMask(self):
        """
        generating ploting mask for the patch
        """
        mask = np.array(self.array, dtype=np.float32)
        mask[mask == 0] = np.nan
        return mask

    def getSignedMask(self):
        """
        generating ploting mask with visual sign for the patch
        """
        signedMask = np.array(self.array * self.sign, dtype=np.float32)
        signedMask[signedMask == 0] = np.nan
        return signedMask

    def getDict(self):
        return {'sparseArray': self.sparseArray, 'sign': self.sign}

    def getTrace(self, mov):
        """
        return trace of this patch in a certain movie
        """
        return get_trace(mov, self.array)

    def isTouching(self, patch2, distance=1):
        """
        decide if this patch is adjacent to another patch within certain distance
        """

        if distance < 1:
            raise LookupError('distance should be integer no less than 1.')

        bigPatch = ni.binary_dilation(self.array,
                          iterations=distance).astype(int)

        if np.amax(bigPatch + patch2.array) > 1:
            return True
        else:
            return False

    def getVisualSpace(self, altMap, aziMap, visualFieldOrigin=None, pixelSize=1., closeIter=None, isPlot=False):
        """
        get the visual response space, visual response space center unique area and
        eccentricity map of a cortical patch
        """

        #        altRange = np.array([np.amin(altMap)-10., np.amax(altMap)+10.])
        #        aziRange = np.array([np.amin(aziMap)-10., np.amax(aziMap)+10.])

        pixelSize = float(pixelSize)

        altRange = np.array([-40., 60.])
        aziRange = np.array([-20., 120.])

        if visualFieldOrigin:
            altMap = altMap - visualFieldOrigin[0]
            aziMap = aziMap - visualFieldOrigin[1]

        gridAzi, gridAlt = np.meshgrid(np.arange(aziRange[0], aziRange[1], pixelSize),
                                       np.arange(altRange[0], altRange[1], pixelSize))

        visualSpace = np.zeros((int(np.ceil((altRange[1] - altRange[0]) / pixelSize)),
                                int(np.ceil((aziRange[1] - aziRange[0]) / pixelSize))))

        patchArray = self.array
        for i in range(patchArray.shape[0]):
            for j in range(patchArray.shape[1]):
                if patchArray[i, j]:
                    corAlt = altMap[i, j]
                    corAzi = aziMap[i, j]
                    if (corAlt >= altRange[0]) & (corAlt < altRange[1]) & (corAzi >= aziRange[0]) & (
                        corAzi < aziRange[1]):
                        indAlt = (corAlt - altRange[0]) // pixelSize
                        indAzi = (corAzi - aziRange[0]) // pixelSize
                        visualSpace[int(indAlt), int(indAzi)] = 1

        if closeIter >= 1:
            visualSpace = ni.binary_closing(visualSpace, iterations=closeIter).astype(int)

        uniqueArea = np.sum(visualSpace[:]) * (pixelSize ** 2)

        visualAltCenter = np.mean(gridAlt[visualSpace != 0])
        visualAziCenter = np.mean(gridAzi[visualSpace != 0])

        if isPlot:
            f = plt.figure()
            ax = f.add_subplot(111)
            plotVisualCoverage(visualSpace,
                               pixelSize=pixelSize,
                               plotAxis=ax)

        return visualSpace, uniqueArea, visualAltCenter, visualAziCenter

    def getSigmaArea(self, detMap):
        """
        calculate sigma area for the patch given altitude and azimuth maps
        """
        sigmaArea = np.sum((self.array * detMap)[:])
        return sigmaArea

    def getPixelVisualCenter(self, altMap, aziMap):
        """
        get the center coordinates in visual response space for all pixels in
        this cortical patch
        """

        altPatch = self.array * altMap
        meanAlt = np.mean(altPatch[altPatch != 0])
        aziPatch = self.array * aziMap
        meanAzi = np.mean(aziPatch[aziPatch != 0])

        return meanAlt, meanAzi

    def eccentricityMap(self, altMap, aziMap, altCenter, aziCenter):
        """
        calculate eccentricity map of this patch to a certain center in visual
        space

        altMap, aziMap, altCenter, aziCenter: in degree

        eccentricity map is returned in degree
        """

        altMap2 = altMap * np.pi / 180
        aziMap2 = aziMap * np.pi / 180

        altCenter2 = altCenter * np.pi / 180
        aziCenter2 = aziCenter * np.pi / 180

        eccMap = np.zeros(self.array.shape)
        eccMap = np.arctan(
            np.sqrt(
                np.square(np.tan(altMap2 - altCenter2))
                +
                np.square(np.tan(aziMap2 - aziCenter2)) / np.square(np.cos(altMap2 - altCenter2))
            )
        )
        eccMap = eccMap * 180 / np.pi
        eccMap[self.array == 0] = np.nan
        return eccMap

    def split2(self, eccMap, patchName='patch00', cutStep=1, borderWidth=2, isPlot=False):
        """
        split this patch into two or more patch, according to the eccentricity
        map (in degree). return a dictionary of patches after split

        patchName: str, original patch name
        """
        minMarker = localMin(eccMap, cutStep)

        connectivity = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

        newLabel = segmentation.watershed(eccMap, minMarker, connectivity=connectivity, mask=self.array)

        border = ni.binary_dilation(self.array).astype(np.int8) - self.array

        for i in range(1, np.amax(newLabel) + 1):
            currArray = np.zeros(self.array.shape, dtype=np.int8)
            currArray[newLabel == i] = 1
            currBorder = ni.binary_dilation(currArray).astype(np.int8) - currArray
            border = border + currBorder

        border[border > 1] = 1
        border = sm.skeletonize(border)

        if borderWidth > 1:
            border = ni.binary_dilation(border, iterations=borderWidth - 1).astype(np.int8)

        newPatchMap = ni.binary_dilation(self.array).astype(np.int8) * (-1 * (border - 1))

        labeledNewPatchMap, patchNum = ni.label(newPatchMap)

        newPatchDict = {}

        for j in range(1, patchNum + 1):

            currPatchName = patchName + '.' + str(j)
            currArray = np.zeros(self.array.shape, dtype=np.int8)
            currArray[labeledNewPatchMap == j] = 1
            currArray = currArray * self.array

            if np.sum(currArray[:]) > 0:
                newPatchDict.update({currPatchName: Patch(currArray, self.sign)})

        if isPlot:
            plt.figure()
            plt.subplot(121)
            plt.imshow(self.array, interpolation='nearest')
            plt.title(patchName + ': before split')
            plt.subplot(122)
            plt.imshow(labeledNewPatchMap, interpolation='nearest')
            plt.title(patchName + ': after split')

        return newPatchDict

    def split(self, eccMap, patchName='patch00', cutStep=1, borderWidth=2, isPlot=False):
        """
        split this patch into two or more patch, according to the eccentricity
        map (in degree). return a dictionary of patches after split

        patchName: str, original patch name
        """
        minMarker = localMin(eccMap, cutStep)

        plt.figure()
        plt.imshow(minMarker, vmin=0, interpolation='nearest')
        plt.colorbar()
        plt.title('markers 1')
        plt.show()

        minMarker = minMarker.astype(np.int32)
        selfArray = self.array.astype(np.int32)
        minMarker = minMarker + 1
        minMarker[minMarker == 1] = 0
        minMarker = minMarker + (-1 * (selfArray - 1))
        # minMarker: marker type for opencv watershed,
        # sure background = 1
        # unknow = 0
        # sure forgrand = 2,3,4... etc

        plt.figure()
        plt.imshow(minMarker, vmin=0, interpolation='nearest')
        plt.colorbar()
        plt.title('markers 2')
        plt.show()

        _ecc_min, _ecc_max = np.nanmin(eccMap), np.nanmax(eccMap)
        eccMapNor = (np.round((eccMap - _ecc_min) / (_ecc_max - _ecc_min + 1e-12) * 255)).astype(np.uint8)
        eccMapRGB = cv2.cvtColor(eccMapNor, cv2.COLOR_GRAY2RGB)
        # eccMapRGB: image type for opencv watershed, RGB, [uint8, uint8, uint8]

        newLabel = cv2.watershed(eccMapRGB, minMarker)

        plt.figure()
        plt.imshow(newLabel, vmin=0, interpolation='nearest')
        plt.colorbar()
        plt.title('markers 3')
        plt.show()

        newBorder = np.zeros(newLabel.shape).astype(int)

        newBorder[newLabel == -1] = 1

        border = ni.binary_dilation(self.array).astype(int) - self.array

        border = newBorder + border

        border[border > 1] = 1

        border = sm.skeletonize(border)

        if borderWidth > 1:
            border = ni.binary_dilation(border, iterations=borderWidth - 1).astype(np.int8)

        newPatchMap = ni.binary_dilation(self.array).astype(np.int8) * (-1 * (border - 1))

        labeledNewPatchMap, patchNum = ni.label(newPatchMap)

        #        if patchNum != np.amax(newLabel):
        #            print 'number of patches: ', patchNum, '; number of local minimum:', np.amax(newLabel)
        #            raise ValueError, "Number of patches after splitting does not equal to number of local minimum!"

        newPatchDict = {}

        for j in range(1, patchNum + 1):

            currPatchName = patchName + '.' + str(j)
            currArray = np.zeros(self.array.shape, dtype=np.int8)
            currArray[labeledNewPatchMap == j] = 1
            currArray = currArray * self.array

            if np.sum(currArray[:]) > 0:
                newPatchDict.update({currPatchName: Patch(currArray, self.sign)})

        if isPlot:
            plt.figure()
            plt.subplot(121)
            plt.imshow(self.array, interpolation='nearest')
            plt.title(patchName + ': before split')
            plt.subplot(122)
            plt.imshow(labeledNewPatchMap, interpolation='nearest')
            plt.title(patchName + ': after split')

        return newPatchDict

    def getBorder(self, borderWidth=2):
        """
        return boder of this patch with boder width defined by "borderWidth"
        """

        patchMap = np.array(self.array, dtype=np.float32)

        smallPatch = ni.binary_erosion(patchMap, iterations=borderWidth).astype(np.float32)

        border = patchMap - smallPatch

        border[border == 0] = np.nan

        return border

    def getCorticalPixelForVisualSpaceCenter(self, eccMap):
        """
        return the coordinates of the pixel representing the center of the
        visual space of the patch
        """
        eccMap2 = np.array(eccMap).astype(float)

        eccMap2[self.array == 0] = np.nan

        cor = np.array(np.where(eccMap2 == np.nanmin(eccMap2))).transpose()

        return cor


class VFSSegment:
    r"""
    GUI for drawing and labeling cortical area contours on a VFS image.

    Example usage:
    --------------
    segmenter = VFSSegment(
        vfs_path=r"D:\path\to\vfs.tif"
    )
    contours = segmenter.get_contours()

    # Or load from previously saved contours:
    segmenter = VFSSegment.from_saved_contours(
        contours_path=r"D:\path\to\vfs_contours.pkl"
    )
    contours = segmenter.contours
    """

    def __init__(self, vfs_path: Optional[str] = None,
                 vfs_array: Optional[np.ndarray] = None,
                 output_dir: Optional[str] = None, output_filename: str = 'vfs_contours.pkl',
                 auto_show: bool = True):
        """Create a VFSSegment. `vfs_path` or `vfs_array` is sufficient to launch the GUI."""
        self.vfs_path = vfs_path
        self.vfs_array = vfs_array

        # Determine output directory: prefer provided, else vfs folder, else cwd
        if output_dir is not None:
            self.output_dir = output_dir
        elif vfs_path is not None:
            self.output_dir = os.path.dirname(vfs_path)
        else:
            self.output_dir = os.getcwd()

        self.output_path = os.path.join(self.output_dir, output_filename)

        # Load VFS if provided
        self.vfs_img = None
        self.vfs_resized = None

        if self.vfs_array is not None:
            self.vfs_img = np.array(self.vfs_array, dtype=float)
        elif self.vfs_path is not None:
            self.vfs_img = self._load_image(self.vfs_path, as_color=True)

        if self.vfs_img is not None:
            self.vfs_resized = self.vfs_img
        else:
            self.vfs_resized = None

        # Contour storage
        self.contours = {}  # Smoothed contours (what gets displayed and saved)
        self.original_contours = {}  # Original unsmoothed contours
        self.area_lines = {}
        self.area_texts = {}  # Text labels for area names
        self.edit_target = None
        self.current_smoothing = 0  # Current smoothing sigma value
        self.pending_add_name = None
        self.brush_mode = False
        self.brush_mode_action = "add"
        self.brush_radius = 8.0
        self.brush_fill_gap_px = 2
        self.brush_active_area = None
        self.brush_is_dragging = False
        self.brush_masks = None
        self.brush_preview = None
        self.undo_stack = []
        self.redo_stack = []

        # Load existing contours if present
        self._load_existing_contours()

        # Tk root for dialogs
        self._tk_root = tk.Tk()
        self._tk_root.withdraw()

        if auto_show:
            if self.vfs_img is not None:
                # VFS is sufficient to launch the GUI; widefield is optional.
                self.launch_gui()
            else:
                print("Warning: vfs image not provided — GUI not launched.")

    @classmethod
    def from_saved_contours(cls, contours_path: str,
                           vfs_path: Optional[str] = None, vfs_array: Optional[np.ndarray] = None,
                           auto_show: bool = False):
        r"""
        Create a VFSSegment instance from a saved contours dict pickle file.

        Parameters:
        -----------
        contours_path : str
            Path to the saved contours pickle file
        vfs_path : str, optional
            Path to VFS image. If None, you can pass `vfs_array`.
        vfs_array : np.ndarray, optional
            VFS image array. Used if `vfs_path` is not provided.
        auto_show : bool, default False
            Whether to automatically launch the GUI after loading

        Returns:
        --------
        VFSSegment
            Instance with loaded contour data

        Example:
        --------
        # Load contours only (no images needed)
        segmenter = VFSSegment.from_saved_contours(
            contours_path=r"D:\path\to\vfs_contours.pkl"
        )
        contours = segmenter.contours

        # Load contours and images to edit in GUI
        segmenter = VFSSegment.from_saved_contours(
            contours_path=r"D:\path\to\vfs_contours.pkl",
            vfs_path=r"D:\path\to\vfs.tif",
            auto_show=True
        )
        """

        with open(contours_path, 'r') as file:
            contours = json.load(file)

        if not isinstance(contours, dict):
            raise ValueError(f"Expected contours to be a dict, got {type(contours)}")

        # Create instance without calling __init__
        instance = cls.__new__(cls)

        # Set basic attributes
        instance.contours = contours
        instance.original_contours = {name: coords.copy() for name, coords in contours.items()}
        instance.output_path = contours_path
        instance.output_dir = os.path.dirname(contours_path)
        instance.current_smoothing = 0

        # Handle optional image paths/arrays (vfs_path or vfs_array is sufficient to launch GUI)
        instance.vfs_path = vfs_path
        instance.vfs_array = vfs_array
        instance.vfs_img = None
        instance.vfs_resized = None

        if instance.vfs_array is not None:
            instance.vfs_img = np.array(instance.vfs_array, dtype=float)
        elif instance.vfs_path is not None:
            instance.vfs_img = instance._load_image(instance.vfs_path, as_color=True)

        if instance.vfs_img is not None:
            instance.vfs_resized = instance.vfs_img

        # Initialize GUI-related attributes
        instance.area_lines = {}
        instance.area_texts = {}
        instance.edit_target = None
        instance.pending_add_name = None
        instance.brush_mode = False
        instance.brush_mode_action = "add"
        instance.brush_radius = 8.0
        instance.brush_fill_gap_px = 2
        instance.brush_active_area = None
        instance.brush_is_dragging = False
        instance.brush_masks = None
        instance.brush_preview = None
        instance.undo_stack = []
        instance.redo_stack = []
        instance._tk_root = tk.Tk()
        instance._tk_root.withdraw()

        print(f"Loaded {len(contours)} contours from: {contours_path}")

        # Launch GUI if requested and images are available
        if auto_show:
            if instance.vfs_img is not None:
                instance.launch_gui()
            else:
                print("Warning: Cannot launch GUI without vfs_path or vfs_array")

        return instance

    def _load_image(self, path: str, as_color: bool = False) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not load image: {path}")
        if as_color:
            # Keep as color and convert BGR to RGB
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                # If grayscale, expand to RGB
                img = np.stack([img, img, img], axis=2)
        else:
            # Convert to grayscale
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.astype(float)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        if img.max() == img.min():
            return np.zeros_like(img, dtype=float)
        return (img - img.min()) / (img.max() - img.min())

    def _load_existing_contours(self):
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, 'r') as file:
                    data = json.load(file)
                if isinstance(data, dict):
                    self.contours = data
                    # Store as original contours (loaded contours are assumed unsmoothed)
                    self.original_contours = {name: coords.copy() for name, coords in data.items()}
                    print(f"Loaded {len(self.contours)} contours from {self.output_path}")
            except Exception as e:
                print(f"Warning: Failed to load contours from {self.output_path}: {e}")

    def _build_composite(self, vfs_alpha: float) -> np.ndarray:
        vfs_norm = self._normalize(self.vfs_resized)

        # Always render VFS with jet colormap
        if len(vfs_norm.shape) == 2:
            vfs_gray = vfs_norm
        else:
            vfs_gray = np.mean(vfs_norm, axis=2)
        vfs_rgb = plt.get_cmap('jet')(vfs_gray)[..., :3]

        white_bg = np.ones_like(vfs_rgb)
        composite = white_bg * (1 - vfs_alpha) + vfs_rgb * vfs_alpha
        return composite

    def _refresh_display(self):
        composite = self._build_composite(self.slider_vfs_alpha.val)
        self.img_display.set_data(composite)
        self.fig.canvas.draw_idle()

    def _smooth_contour(self, coords: list, sigma: float) -> list:
        """
        Apply Gaussian smoothing to contour coordinates.

        Parameters:
        -----------
        coords : list
            List of (x, y) coordinate tuples
        sigma : float
            Standard deviation for Gaussian kernel. 0 = no smoothing.

        Returns:
        --------
        list : Smoothed coordinates as list of (x, y) tuples
        """
        if sigma == 0 or len(coords) < 3:
            return coords

        # Check if contour is closed (first and last points are the same or very close)
        coords_array = np.array(coords)
        is_closed = np.allclose(coords_array[0], coords_array[-1], atol=1e-6)

        # If closed, remove the duplicate last point before smoothing
        if is_closed:
            coords_to_smooth = coords[:-1]
        else:
            coords_to_smooth = coords

        # Separate x and y coordinates
        xs = np.array([p[0] for p in coords_to_smooth])
        ys = np.array([p[1] for p in coords_to_smooth])

        # Apply Gaussian filter (mode='wrap' to handle closed contours)
        xs_smooth = gaussian_filter(xs, sigma=sigma, mode='wrap')
        ys_smooth = gaussian_filter(ys, sigma=sigma, mode='wrap')

        # Recombine into coordinate list
        smoothed = [(x, y) for x, y in zip(xs_smooth, ys_smooth)]

        # If originally closed, close the smoothed contour
        if is_closed:
            smoothed.append(smoothed[0])

        return smoothed

    def _update_smoothing(self, sigma: float):
        """
        Update all contours with new smoothing value.

        Parameters:
        -----------
        sigma : float
            Smoothing parameter (0 = no smoothing)
        """
        self.current_smoothing = sigma

        # Apply smoothing to all original contours
        for name, original_coords in self.original_contours.items():
            self.contours[name] = self._smooth_contour(original_coords, sigma)

        # Redraw all contour lines
        self._draw_existing_contours()
        self.fig.canvas.draw_idle()

    def _draw_existing_contours(self):
        # Remove existing lines and text labels
        for name, line in self.area_lines.items():
            line.remove()
        self.area_lines = {}

        for name, text in self.area_texts.items():
            text.remove()
        self.area_texts = {}

        # Draw (potentially smoothed) contours
        for name, coords in self.contours.items():
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            # Preserve edit highlighting if this area is being edited
            if name == self.edit_target:
                line, = self.ax.plot(xs, ys, color='red', linewidth=2.5)
            else:
                line, = self.ax.plot(xs, ys, color='black', linewidth=1.5)
            self.area_lines[name] = line

            # Calculate centroid of the contour for label placement
            if len(coords) > 0:
                xs_array = np.array(xs)
                ys_array = np.array(ys)
                centroid_x = np.mean(xs_array)
                centroid_y = np.mean(ys_array)

                # Add text label at centroid
                text = self.ax.text(centroid_x, centroid_y, name,
                                   ha='center', va='center',
                                   fontsize=10, fontweight='bold',
                                   color='white',
                                   bbox=dict(boxstyle='round,pad=0.3',
                                           facecolor='black',
                                           alpha=0.7,
                                           edgecolor='none'))
                self.area_texts[name] = text

    def _build_area_masks(self):
        target_shape = self._get_target_shape()
        if target_shape is None:
            return {}
        masks = {}
        for name, coords in self.original_contours.items():
            mask = self._contour_to_mask(coords, target_shape)
            if mask is not None:
                masks[name] = mask
        return masks

    def _find_area_at_point(self, x, y, masks):
        xi = int(round(x))
        yi = int(round(y))
        for name, mask in masks.items():
            if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1]:
                if mask[yi, xi]:
                    return name
        return None

    def _make_circle_mask(self, x, y, radius, shape):
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.circle(mask, (int(round(x)), int(round(y))), int(round(radius)), 1, -1)
        return mask.astype(bool)

    def _mask_to_contour(self, mask):
        contours = measure.find_contours(mask.astype(np.uint8), 0.5)
        if not contours:
            return None
        contour = max(contours, key=lambda c: c.shape[0])
        coords = [(float(p[1]), float(p[0])) for p in contour]
        if len(coords) >= 3 and not np.allclose(coords[0], coords[-1]):
            coords.append(coords[0])
        return coords

    def _apply_masks_to_contours(self, masks, names):
        for name in list(names):
            mask = masks.get(name)
            if mask is None or mask.sum() == 0:
                if name in self.contours:
                    del self.contours[name]
                if name in self.original_contours:
                    del self.original_contours[name]
                if name in self.area_lines:
                    try:
                        self.area_lines[name].remove()
                    except Exception:
                        pass
                    del self.area_lines[name]
                if name in self.area_texts:
                    try:
                        self.area_texts[name].remove()
                    except Exception:
                        pass
                    del self.area_texts[name]
                continue

            coords = self._mask_to_contour(mask)
            if coords is None:
                continue
            self.original_contours[name] = coords
            self.contours[name] = self._smooth_contour(coords.copy(), self.current_smoothing)

        self._draw_existing_contours()
        self.fig.canvas.draw_idle()

    def _snapshot_state(self):
        return (
            copy.deepcopy(self.original_contours),
            copy.deepcopy(self.contours),
        )

    def _restore_state(self, state):
        self.original_contours, self.contours = state
        self.edit_target = None
        self.pending_add_name = None
        self.brush_masks = None
        self._draw_existing_contours()
        self.fig.canvas.draw_idle()

    def _push_undo(self):
        self.undo_stack.append(self._snapshot_state())
        self.redo_stack = []

    def _undo(self):
        if not self.undo_stack:
            return
        self.redo_stack.append(self._snapshot_state())
        state = self.undo_stack.pop()
        self._restore_state(state)

    def _redo(self):
        if not self.redo_stack:
            return
        self.undo_stack.append(self._snapshot_state())
        state = self.redo_stack.pop()
        self._restore_state(state)

    def _prompt_area_name(self, default_name: Optional[str] = None) -> Optional[str]:
        return simpledialog.askstring(
            "Area name",
            "Enter area name:",
            parent=self._tk_root,
            initialvalue=default_name
        )

    def _parse_area_names(self, raw_input: Optional[str]) -> list:
        if not raw_input:
            return []
        parts = re.split(r"[\s,;\n]+", raw_input.strip())
        return [p.strip() for p in parts if p.strip()]

    def _get_target_shape(self) -> Optional[tuple]:
        if self.vfs_resized is not None:
            return self.vfs_resized.shape[:2]
        if self.vfs_img is not None:
            return self.vfs_img.shape[:2]
        return None

    def _contour_to_mask(self, coords: list, shape: tuple) -> Optional[np.ndarray]:
        if coords is None or len(coords) < 3:
            return None
        pts = np.array(coords, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 3:
            return None

        pts_int = np.round(pts).astype(np.int32)
        pts_int[:, 0] = np.clip(pts_int[:, 0], 0, shape[1] - 1)
        pts_int[:, 1] = np.clip(pts_int[:, 1], 0, shape[0] - 1)
        pts_int = pts_int.reshape((-1, 1, 2))

        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [pts_int], 1)
        return mask.astype(bool)

    def _merge_areas(self):
        if len(self.contours) < 2:
            messagebox.showinfo("Merge areas", "Need at least two areas to merge.")
            return

        name_input = simpledialog.askstring(
            "Merge areas",
            f"Enter area name(s) to merge (comma/space-separated):\n{', '.join(sorted(self.contours.keys()))}",
            parent=self._tk_root
        )
        names = self._parse_area_names(name_input)
        if len(names) < 2:
            messagebox.showwarning("Merge areas", "Please enter two or more area names.")
            return

        missing = [n for n in names if n not in self.contours]
        if missing:
            messagebox.showwarning("Merge areas", f"Area(s) not found: {', '.join(missing)}")
            return

        existing = [n for n in names if n in self.contours]

        new_name = simpledialog.askstring(
            "Merge areas",
            "Enter new merged area name:",
            parent=self._tk_root
        )
        if not new_name:
            return
        new_name = new_name.strip()
        if not new_name:
            return

        delete_names = list(existing)
        if new_name in self.contours and new_name not in existing:
            overwrite = messagebox.askyesno(
                "Merge areas",
                f"Area '{new_name}' already exists. Overwrite it with merged area?"
            )
            if not overwrite:
                return
            delete_names.append(new_name)

        target_shape = self._get_target_shape()
        if target_shape is None:
            messagebox.showwarning("Merge areas", "Cannot determine image shape for merging.")
            return

        masks = []
        for name in existing:
            coords = self.original_contours.get(name, self.contours.get(name))
            mask = self._contour_to_mask(coords, target_shape)
            if mask is None or mask.sum() == 0:
                messagebox.showwarning("Merge areas", f"Area '{name}' has an invalid contour.")
                return
            masks.append(mask)

        union_mask = np.zeros(target_shape, dtype=bool)
        for mask in masks:
            union_mask |= mask

        _, num_labels = ni.label(union_mask.astype(np.uint8))
        merge_mask = union_mask
        if num_labels > 1:
            border_tolerance = 10
            union_dilated = np.zeros(target_shape, dtype=bool)
            for mask in masks:
                union_dilated |= ni.binary_dilation(mask, iterations=border_tolerance)
            _, num_labels_dilated = ni.label(union_dilated.astype(np.uint8))
            if num_labels_dilated > 1:
                messagebox.showwarning(
                    "Merge areas",
                    f"Selected areas are not bordering within {border_tolerance} px."
                )
                return
            merge_mask = ni.binary_closing(union_mask, iterations=border_tolerance)

        contours = measure.find_contours(merge_mask.astype(np.uint8), 0.5)
        if not contours:
            messagebox.showwarning("Merge areas", "Failed to extract merged contour.")
            return

        merged_contour = max(contours, key=lambda c: c.shape[0])
        merged_coords = [(float(p[1]), float(p[0])) for p in merged_contour]
        if len(merged_coords) < 3:
            messagebox.showwarning("Merge areas", "Merged contour is too small.")
            return
        if not np.allclose(merged_coords[0], merged_coords[-1]):
            merged_coords.append(merged_coords[0])

        self._push_undo()

        for name in delete_names:
            if name in self.contours:
                del self.contours[name]
            if name in self.original_contours:
                del self.original_contours[name]
            if name in self.area_lines:
                self.area_lines[name].remove()
                del self.area_lines[name]
            if name in self.area_texts:
                self.area_texts[name].remove()
                del self.area_texts[name]
            if self.edit_target == name:
                self.edit_target = None

        self.original_contours[new_name] = merged_coords
        self.contours[new_name] = self._smooth_contour(merged_coords.copy(), self.current_smoothing)

        self._draw_existing_contours()
        self.status_text.set_text(f"Merged {len(existing)} area(s) -> {new_name}")
        self.fig.canvas.draw_idle()

    def _split_area_with_line(self, area_name: str, p1: tuple, p2: tuple):
        """
        Split an existing area by drawing a straight line between p1 and p2.

        area_name: name of area to split (must exist in self.original_contours)
        p1, p2: (x,y) coordinates in display/image space
        """
        if area_name not in self.original_contours and area_name not in self.contours:
            messagebox.showwarning("Split area", f"Area '{area_name}' not found.")
            return

        target_shape = self._get_target_shape()
        if target_shape is None:
            messagebox.showwarning("Split area", "Cannot determine image shape for splitting.")
            return

        coords = self.original_contours.get(area_name, self.contours.get(area_name))
        mask = self._contour_to_mask(coords, target_shape)
        if mask is None or mask.sum() == 0:
            messagebox.showwarning("Split area", f"Area '{area_name}' has an invalid contour.")
            return

        # Rasterize the cut line
        cut_mask = np.zeros(target_shape, dtype=np.uint8)
        pt1 = (int(round(p1[0])), int(round(p1[1])))
        pt2 = (int(round(p2[0])), int(round(p2[1])))
        # cv2.line takes (x,y) points as (col,row)
        cv2.line(cut_mask, pt1, pt2, color=1, thickness=1)

        new_mask = mask.copy()
        new_mask[cut_mask.astype(bool)] = False

        labeled, num = ni.label(new_mask.astype(np.uint8))

        # If the cut did not split, try thicker cut as fallback
        if num < 2:
            cut_mask = np.zeros(target_shape, dtype=np.uint8)
            cv2.line(cut_mask, pt1, pt2, color=1, thickness=3)
            new_mask = mask.copy()
            new_mask[cut_mask.astype(bool)] = False
            labeled, num = ni.label(new_mask.astype(np.uint8))

        if num < 2:
            messagebox.showwarning("Split area", "Line did not split the selected area. Try a different cut.")
            return

        self._push_undo()

        # Collect labeled components (ignore tiny pieces)
        components = []
        for i in range(1, num + 1):
            comp = (labeled == i)
            area = np.sum(comp)
            if area > 2:
                components.append((area, comp))

        if len(components) < 2:
            messagebox.showwarning("Split area", "Cut produced fewer than two usable components.")
            return

        components.sort(key=lambda x: x[0], reverse=True)

        # Remove original area and its visuals
        if area_name in self.contours:
            del self.contours[area_name]
        if area_name in self.original_contours:
            del self.original_contours[area_name]
        if area_name in self.area_lines:
            try:
                self.area_lines[area_name].remove()
            except Exception:
                pass
            del self.area_lines[area_name]
        if area_name in self.area_texts:
            try:
                self.area_texts[area_name].remove()
            except Exception:
                pass
            del self.area_texts[area_name]

        # Create new areas named area_name + '.1', '.2', ...
        for idx, (_, comp_mask) in enumerate(components, start=1):
            new_name = f"{area_name}.{idx}"
            contours = measure.find_contours(comp_mask.astype(np.uint8), 0.5)
            if not contours:
                # fallback: create bounding box contour
                ys, xs = np.where(comp_mask)
                if xs.size == 0:
                    continue
                minx, maxx = xs.min(), xs.max()
                miny, maxy = ys.min(), ys.max()
                contour_coords = [(float(minx), float(miny)), (float(maxx), float(miny)), (float(maxx), float(maxy)), (float(minx), float(maxy)), (float(minx), float(miny))]
            else:
                contour = max(contours, key=lambda c: c.shape[0])
                contour_coords = [(float(p[1]), float(p[0])) for p in contour]
                if len(contour_coords) >= 3 and not np.allclose(contour_coords[0], contour_coords[-1]):
                    contour_coords.append(contour_coords[0])

            self.original_contours[new_name] = contour_coords
            self.contours[new_name] = self._smooth_contour(contour_coords.copy(), self.current_smoothing)

        # Redraw
        self._draw_existing_contours()
        self.status_text.set_text(f"Split '{area_name}' -> {len(components)} pieces")
        self.save_contours()
        self.fig.canvas.draw_idle()

    def _add_area(self):
        name = simpledialog.askstring(
            "Add area",
            "Enter new area name:",
            parent=self._tk_root
        )
        if not name:
            return
        name = name.strip()
        if not name:
            return

        if name in self.contours:
            overwrite = messagebox.askyesno("Add area", f"Area '{name}' exists. Overwrite?")
            if not overwrite:
                return

        self.pending_add_name = name
        self.edit_target = None
        self.status_text.set_text(f"Draw new area: {name}")
        self.fig.canvas.draw_idle()

    def _prompt_edit_area(self):
        if not self.contours:
            messagebox.showinfo("Edit area", "No existing areas to edit.")
            return
        name = simpledialog.askstring(
            "Edit area",
            f"Enter area name to edit:\n{', '.join(sorted(self.contours.keys()))}",
            parent=self._tk_root
        )
        if name and name in self.contours:
            self.edit_target = name
            for area_name, line in self.area_lines.items():
                line.set_color('black')
                line.set_linewidth(1.5)
            if name in self.area_lines:
                self.area_lines[name].set_color('red')
                self.area_lines[name].set_linewidth(2.5)
            self.status_text.set_text(f"Editing area: {name}")
            self.fig.canvas.draw_idle()
        elif name:
            messagebox.showwarning("Edit area", f"Area '{name}' not found.")

    def _delete_area(self):
        if not self.contours:
            messagebox.showinfo("Delete area", "No existing areas to delete.")
            return
        name_input = simpledialog.askstring(
            "Delete area",
            f"Enter area name(s) to delete (comma/space-separated):\n{', '.join(sorted(self.contours.keys()))}",
            parent=self._tk_root
        )
        names = self._parse_area_names(name_input)
        if not names:
            return

        existing = [n for n in names if n in self.contours]
        missing = [n for n in names if n not in self.contours]

        if not existing:
            if missing:
                messagebox.showwarning("Delete area", f"Area(s) not found: {', '.join(missing)}")
            return

        confirm_list = ", ".join(existing)
        if not messagebox.askyesno("Delete area", f"Delete area(s): {confirm_list}?"):
            return

        self._push_undo()

        for name in existing:
            del self.contours[name]
            if name in self.original_contours:
                del self.original_contours[name]
            if name in self.area_lines:
                self.area_lines[name].remove()
                del self.area_lines[name]
            if name in self.area_texts:
                self.area_texts[name].remove()
                del self.area_texts[name]
            if self.edit_target == name:
                self.edit_target = None

        if missing:
            messagebox.showwarning("Delete area", f"Area(s) not found: {', '.join(missing)}")

        self.status_text.set_text(f"Deleted {len(existing)} area(s).")
        self.fig.canvas.draw_idle()

    def _rename_area(self):
        if not self.contours:
            messagebox.showinfo("Rename area", "No existing areas to rename.")
            return
        old_name = simpledialog.askstring(
            "Rename area",
            f"Enter area name to rename:\n{', '.join(sorted(self.contours.keys()))}",
            parent=self._tk_root
        )
        if not old_name:
            return
        if old_name not in self.contours:
            messagebox.showwarning("Rename area", f"Area '{old_name}' not found.")
            return

        new_name = simpledialog.askstring(
            "Rename area",
            "Enter new area name:",
            parent=self._tk_root,
            initialvalue=old_name
        )
        if not new_name:
            return
        if new_name in self.contours and new_name != old_name:
            messagebox.showwarning("Rename area", f"Area '{new_name}' already exists.")
            return

        self._push_undo()

        self.contours[new_name] = self.contours.pop(old_name)
        if old_name in self.original_contours:
            self.original_contours[new_name] = self.original_contours.pop(old_name)
        if self.edit_target == old_name:
            self.edit_target = new_name

        self._draw_existing_contours()
        self.status_text.set_text(f"Renamed area '{old_name}' -> '{new_name}'.")
        self.fig.canvas.draw_idle()

    def save_contours(self):
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(self.contours, f, indent=4)
        print(f"Contours saved to: {self.output_path}")

    def get_contours(self) -> Dict[str, list]:
        return self.contours

    def launch_gui(self):
        # Ensure interactive backend
        matplotlib.use('TkAgg')
        plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.subplots_adjust(left=0.1, bottom=0.30, right=0.95, top=0.95)

        composite = self._build_composite(1.0)
        self.img_display = self.ax.imshow(composite)
        self.ax.axis('off')

        # Sliders
        ax_vfs = plt.axes([0.15, 0.24, 0.7, 0.02])
        ax_smooth = plt.axes([0.15, 0.20, 0.7, 0.02])
        ax_brush = plt.axes([0.15, 0.16, 0.7, 0.02])
        self.slider_vfs_alpha = Slider(ax_vfs, 'VFS Alpha', 0.0, 1.0, valinit=1.0, valstep=0.01)
        self.slider_smooth = Slider(ax_smooth, 'Contour Smoothing', 0.0, 7.0, valinit=0.0, valstep=0.1)
        self.slider_brush = Slider(ax_brush, 'Brush Radius', 2.0, 30.0, valinit=self.brush_radius, valstep=1.0)

        # Buttons
        # Row 1: Rename, Merge, Split, Brush, Br. Mode
        ax_rename = plt.axes([0.255, 0.06, 0.09, 0.04])
        ax_merge = plt.axes([0.355, 0.06, 0.09, 0.04])
        ax_split = plt.axes([0.455, 0.06, 0.09, 0.04])
        ax_brush_btn = plt.axes([0.555, 0.06, 0.09, 0.04])
        ax_mode = plt.axes([0.655, 0.06, 0.09, 0.04])
        # Row 2: Add, Delete, Undo, Redo, Save, Done
        ax_add = plt.axes([0.205, 0.01, 0.09, 0.04])
        ax_delete = plt.axes([0.305, 0.01, 0.09, 0.04])
        ax_undo = plt.axes([0.405, 0.01, 0.09, 0.04])
        ax_redo = plt.axes([0.505, 0.01, 0.09, 0.04])
        ax_save = plt.axes([0.605, 0.01, 0.09, 0.04])
        ax_done = plt.axes([0.705, 0.01, 0.09, 0.04])

        btn_rename = Button(ax_rename, 'Rename')
        btn_merge = Button(ax_merge, 'Merge')
        btn_split = Button(ax_split, 'Split')
        btn_brush = Button(ax_brush_btn, 'Brush')
        btn_mode = Button(ax_mode, 'Br. Mode')
        btn_add = Button(ax_add, 'Add')
        btn_delete = Button(ax_delete, 'Delete')
        btn_undo = Button(ax_undo, 'Undo')
        btn_redo = Button(ax_redo, 'Redo')
        btn_save = Button(ax_save, 'Save')
        btn_done = Button(ax_done, 'Done')

        # Status text
        self.status_text = self.ax.text(0.02, 0.02, '', transform=self.ax.transAxes,
                                        ha='left', va='bottom', fontsize=10,
                                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

        # Brush preview circle
        self.brush_preview = Circle(
            (0, 0),
            radius=self.brush_radius,
            fill=True,
            facecolor=(0.5, 0.5, 0.5, 0.5),
            edgecolor='black',
            linewidth=1.5,
            visible=False,
        )
        self.ax.add_patch(self.brush_preview)

        # Drawing state
        self.drawing = False
        self.current_points = []
        self.current_line = None

        # Split-line state
        self.split_line_mode = False
        self.split_line_points = []
        self.split_preview_line = None
        self._split_target_name = None

        def on_slider_change(val):
            self._refresh_display()

        def on_press(event):
            if event.inaxes != self.ax or event.button != 1:
                return
            if event.xdata is None or event.ydata is None:
                return
            if self.brush_mode:
                if self.brush_masks is None:
                    self.brush_masks = self._build_area_masks()
                if self.brush_mode_action in ("add", "fill"):
                    if not self.brush_masks:
                        messagebox.showwarning("Brush edit", "No areas available to edit.")
                        return
                    active = self._find_area_at_point(event.xdata, event.ydata, self.brush_masks)
                    if active is None:
                        messagebox.showwarning("Brush edit", "Start the brush inside an existing area.")
                        return
                    self.brush_active_area = active
                else:
                    self.brush_active_area = None
                self._push_undo()
                self.brush_is_dragging = True
                _apply_brush(event.xdata, event.ydata)
                return
            # If in split-line mode, capture the first endpoint and draw preview
            if self.split_line_mode:
                self.split_line_points = [(event.xdata, event.ydata)]
                if self.split_preview_line is not None:
                    try:
                        self.split_preview_line.remove()
                    except Exception:
                        pass
                self.split_preview_line, = self.ax.plot([event.xdata, event.xdata], [event.ydata, event.ydata], color='yellow', linewidth=2.0)
                self.fig.canvas.draw_idle()
                return
            # Normal freehand drawing
            if self.pending_add_name is None and self.edit_target is None:
                return
            self.drawing = True
            self.current_points = [(event.xdata, event.ydata)]
            self.current_line, = self.ax.plot([event.xdata], [event.ydata], color='black', linewidth=1.5)
            self.fig.canvas.draw_idle()

        def on_move(event):
            if self.brush_mode:
                if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
                    self.brush_preview.center = (event.xdata, event.ydata)
                    self.brush_preview.set_radius(self.brush_radius)
                    if not self.brush_preview.get_visible():
                        self.brush_preview.set_visible(True)
                    self.fig.canvas.draw_idle()
                else:
                    if self.brush_preview.get_visible():
                        self.brush_preview.set_visible(False)
                        self.fig.canvas.draw_idle()
            # Update preview for split-line mode
            if self.split_line_mode:
                if not self.split_line_points:
                    return
                if event.inaxes != self.ax:
                    return
                if event.xdata is None or event.ydata is None:
                    return
                p1 = self.split_line_points[0]
                xs = [p1[0], event.xdata]
                ys = [p1[1], event.ydata]
                if self.split_preview_line is None:
                    self.split_preview_line, = self.ax.plot(xs, ys, color='yellow', linewidth=2.0)
                else:
                    self.split_preview_line.set_data(xs, ys)
                self.fig.canvas.draw_idle()
                return
            if self.brush_mode and self.brush_is_dragging:
                if event.inaxes != self.ax:
                    return
                if event.xdata is None or event.ydata is None:
                    return
                _apply_brush(event.xdata, event.ydata)
                return
            # Normal freehand drawing
            if not self.drawing or event.inaxes != self.ax:
                return
            if event.xdata is None or event.ydata is None:
                return
            self.current_points.append((event.xdata, event.ydata))
            xs = [p[0] for p in self.current_points]
            ys = [p[1] for p in self.current_points]
            self.current_line.set_data(xs, ys)
            self.fig.canvas.draw_idle()

        def on_release(event):
            if self.brush_mode and self.brush_is_dragging:
                self.brush_is_dragging = False
                self.brush_active_area = None
                return
            # Finalize split-line if active
            if self.split_line_mode:
                if not self.split_line_points:
                    return
                if event.xdata is None or event.ydata is None:
                    return
                self.split_line_points.append((event.xdata, event.ydata))
                p1, p2 = self.split_line_points[0], self.split_line_points[1]
                if self.split_preview_line is not None:
                    try:
                        self.split_preview_line.remove()
                    except Exception:
                        pass
                    self.split_preview_line = None
                target = self._split_target_name if self._split_target_name else self.edit_target
                if not target:
                    target = simpledialog.askstring("Split area", f"Enter area name to split:\n{', '.join(sorted(self.contours.keys()))}", parent=self._tk_root)
                if target:
                    target = target.strip()
                    if target:
                        self._split_area_with_line(target, p1, p2)
                self.split_line_mode = False
                self.split_line_points = []
                self._split_target_name = None
                return

            if not self.drawing:
                return
            self.drawing = False

            if len(self.current_points) < 3:
                if self.current_line is not None:
                    self.current_line.remove()
                self.current_line = None
                self.current_points = []
                self.fig.canvas.draw_idle()
                return

            # Close contour
            self.current_points.append(self.current_points[0])
            xs = [p[0] for p in self.current_points]
            ys = [p[1] for p in self.current_points]
            self.current_line.set_data(xs, ys)

            if self.pending_add_name:
                area_name = self.pending_add_name
                self.pending_add_name = None
            else:
                default_name = self.edit_target
                area_name = self._prompt_area_name(default_name=default_name)
            if area_name is None:
                # Cancelled
                self.current_line.remove()
                self.current_line = None
                self.current_points = []
                self.pending_add_name = None
                self.fig.canvas.draw_idle()
                return

            area_name = area_name.strip()
            if not area_name:
                self.current_line.remove()
                self.current_line = None
                self.current_points = []
                self.pending_add_name = None
                self.fig.canvas.draw_idle()
                return

            if area_name in self.contours and area_name != self.edit_target:
                overwrite = messagebox.askyesno("Overwrite area", f"Area '{area_name}' exists. Overwrite?")
                if not overwrite:
                    self.current_line.remove()
                    self.current_line = None
                    self.current_points = []
                    self.pending_add_name = None
                    self.fig.canvas.draw_idle()
                    return

            self._push_undo()

            # Remove old line if replacing existing area
            if area_name in self.area_lines:
                self.area_lines[area_name].remove()

            # Store original (unsmoothed) contour
            self.original_contours[area_name] = self.current_points.copy()
            # Apply current smoothing to get display contour
            self.contours[area_name] = self._smooth_contour(
                self.current_points.copy(),
                self.current_smoothing
            )

            # Update the line to use smoothed contour
            smoothed_xs = [p[0] for p in self.contours[area_name]]
            smoothed_ys = [p[1] for p in self.contours[area_name]]
            self.current_line.set_data(smoothed_xs, smoothed_ys)
            self.area_lines[area_name] = self.current_line
            self.current_line.set_color('black')
            self.current_line.set_linewidth(1.5)

            self.edit_target = None
            self.pending_add_name = None
            self.status_text.set_text(f"Saved area: {area_name}")
            self.current_line = None
            self.current_points = []
            self.save_contours()
            self.fig.canvas.draw_idle()

        def on_save(event):
            self.save_contours()
            self.status_text.set_text("Contours saved.")
            self.fig.canvas.draw_idle()

        def on_split(event):
            default = self.edit_target
            name = simpledialog.askstring("Split area", "Enter area name to split:", parent=self._tk_root, initialvalue=default)
            if not name:
                return
            name = name.strip()
            if name not in self.contours and name not in self.original_contours:
                messagebox.showwarning("Split area", f"Area '{name}' not found.")
                return
            self._split_target_name = name
            self.split_line_mode = True
            self.split_line_points = []
            self.status_text.set_text(f"Draw straight line across area: {name} (click-drag-release)")
            self.fig.canvas.draw_idle()

        def on_brush(event):
            self.brush_mode = not self.brush_mode
            if self.brush_mode:
                self.brush_masks = self._build_area_masks()
                label = "fill-to-neighbor" if self.brush_mode_action == "fill" else self.brush_mode_action
                self.status_text.set_text(
                    f"Brush mode ON ({label}). Drag to paint."
                )
            else:
                self.brush_masks = None
                if self.brush_preview.get_visible():
                    self.brush_preview.set_visible(False)
                self.status_text.set_text("Brush mode OFF.")
            self.fig.canvas.draw_idle()

        def on_mode(event):
            mode_cycle = ["add", "subtract", "fill"]
            curr_index = mode_cycle.index(self.brush_mode_action)
            self.brush_mode_action = mode_cycle[(curr_index + 1) % len(mode_cycle)]
            label = "fill-to-neighbor" if self.brush_mode_action == "fill" else self.brush_mode_action
            self.status_text.set_text(f"Brush mode: {label}")
            self.fig.canvas.draw_idle()

        def on_undo(event):
            self._undo()

        def on_redo(event):
            self._redo()

        def on_delete(event):
            self._delete_area()

        def on_add(event):
            self._add_area()

        def on_merge(event):
            self._merge_areas()

        def on_rename(event):
            self._rename_area()

        def on_done(event):
            self.save_contours()
            plt.close(self.fig)

        def on_smooth_change(val):
            self._update_smoothing(val)

        def on_brush_change(val):
            self.brush_radius = float(val)
            if self.brush_preview.get_visible():
                self.brush_preview.set_radius(self.brush_radius)
                self.fig.canvas.draw_idle()

        def _apply_brush(x, y):
            target_shape = self._get_target_shape()
            if target_shape is None:
                return

            if self.brush_masks is None:
                self.brush_masks = self._build_area_masks()
            circle = self._make_circle_mask(x, y, self.brush_radius, target_shape)
            affected = set()

            if self.brush_mode_action == "add":
                active = self.brush_active_area
                if active is None:
                    return
                if active not in self.brush_masks:
                    self.brush_masks[active] = np.zeros(target_shape, dtype=bool)
                self.brush_masks[active] = self.brush_masks[active] | circle
                affected.add(active)

                for name in list(self.brush_masks.keys()):
                    if name == active:
                        continue
                    before = self.brush_masks[name]
                    after = before & (~circle)
                    if not np.array_equal(before, after):
                        self.brush_masks[name] = after
                        affected.add(name)
            elif self.brush_mode_action == "fill":
                active = self.brush_active_area
                if active is None:
                    return
                if active not in self.brush_masks:
                    self.brush_masks[active] = np.zeros(target_shape, dtype=bool)

                # Grow only into empty space, preserving neighboring areas.
                other_union = np.zeros(target_shape, dtype=bool)
                for name, mask in self.brush_masks.items():
                    if name == active:
                        continue
                    other_union |= mask

                allowed = circle & (~other_union)
                before = self.brush_masks[active]
                after = before | allowed
                if self.brush_fill_gap_px > 0:
                    after = ni.binary_closing(after, iterations=self.brush_fill_gap_px)
                    after = after & (~other_union)
                if not np.array_equal(before, after):
                    self.brush_masks[active] = after
                    affected.add(active)
            else:
                for name in list(self.brush_masks.keys()):
                    before = self.brush_masks[name]
                    after = before & (~circle)
                    if not np.array_equal(before, after):
                        self.brush_masks[name] = after
                        affected.add(name)

            if affected:
                self._apply_masks_to_contours(self.brush_masks, affected)

        self.slider_vfs_alpha.on_changed(on_slider_change)
        self.slider_smooth.on_changed(on_smooth_change)
        self.slider_brush.on_changed(on_brush_change)
        # Button callbacks in order: rename, merge, split, edit, add, delete, save, done
        btn_rename.on_clicked(on_rename)
        btn_merge.on_clicked(on_merge)
        btn_split.on_clicked(on_split)
        btn_brush.on_clicked(on_brush)
        btn_mode.on_clicked(on_mode)
        btn_undo.on_clicked(on_undo)
        btn_redo.on_clicked(on_redo)
        btn_add.on_clicked(on_add)
        btn_delete.on_clicked(on_delete)
        btn_save.on_clicked(on_save)
        btn_done.on_clicked(on_done)

        self.fig.canvas.mpl_connect('button_press_event', on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', on_move)
        self.fig.canvas.mpl_connect('button_release_event', on_release)

        self._draw_existing_contours()
        if self.contours:
            self.status_text.set_text(f"Loaded {len(self.contours)} existing areas.")

        plt.show(block=True)


def load_composite_maps(
    mean_vfs_path,
    mean_alt_path,
    mean_azi_path,
    params=None,
    param_tweaks=None,
):
    if params is None:
        params = default_params()
    if param_tweaks:
        params = update_params(params, **param_tweaks)

    alt_pos_mapf = ni.gaussian_filter(
        tifffile.imread(mean_alt_path), sigma=params["phaseMapFilterSigma"]
    )
    azi_pos_mapf = ni.gaussian_filter(
        tifffile.imread(mean_azi_path), sigma=params["phaseMapFilterSigma"]
    )
    sign_mapf = ni.gaussian_filter(
        tifffile.imread(mean_vfs_path), sigma=params["signMapFilterSigma"]
    )
    return alt_pos_mapf, azi_pos_mapf, sign_mapf


def default_waters_isi1_list():
    return [
        "541622887",
        "512415468",
        "531337417",
        "531338127",
        "542117164",
        "542112455",
        "541520741",
        "535676912",
        "540992447",
        "541521621",
        "541621635",
        "543024492",
        "541623655",
        "542269558",
        "542118621",
        "543025312",
        "542267920",
        "542116435",
        "541626956",
        "543018195",
        "539320230",
        "542268849",
        "542266505",
        "542267213",
        "540986357",
        "504720222",
        "505121458",
        "509427495",
        "509435130",
        "509591479",
        "509592187",
        "510654077",
        "511438680",
        "509844206",
        "509842724",
        "512319446",
        "511439388",
        "511559662",
        "511558930",
        "511587340",
        "511940682",
        "512416186",
        "511816486",
        "513042927",
        "511817198",
        "512404385",
        "513244249",
        "513245031",
        "513045901",
        "513486087",
        "513045193",
        "514020745",
        "513245748",
        "514070360",
        "513246501",
        "513247246",
        "528692773",
        "513489967",
        "513044476",
        "539641847",
    ]


def default_waters_wkf_ids():
    return {
        "sign": 745546072,
        "altitude": 745544088,
        "azimuth": 745545159,
    }


def numpy_load_wkf(wkf_id, url="http://api.brain-map.org/api/v2/well_known_file_download/{}"):
    url = url.format(wkf_id)
    response = requests.get(url, timeout=60)
    if response.status_code != 200:
        raise ValueError(f"Error retrieving file from {url}")
    return np.load(BytesIO(response.content))


def load_waters_maps_from_wkf(
    isi1_list=None,
    wkf_ids=None,
    params=None,
    remove_nans=True,
    filter=True,
):
    if params is None:
        params = default_params()
    if isi1_list is None:
        isi1_list = default_waters_isi1_list()
    if wkf_ids is None:
        wkf_ids = default_waters_wkf_ids()

    sign_map_stack = numpy_load_wkf(wkf_ids["sign"])
    alt_map_stack = numpy_load_wkf(wkf_ids["altitude"])
    azi_map_stack = numpy_load_wkf(wkf_ids["azimuth"])

    alt_pos_mapf, azi_pos_mapf = getAltAziMapsFromWaters(
        isi1_list,
        alt_map_stack,
        azi_map_stack,
        remove_nans=remove_nans,
        filter=filter,
        params=params,
    )
    sign_mapf = visualSignMap(alt_pos_mapf, azi_pos_mapf, filter=False, params=params)
    return alt_pos_mapf, azi_pos_mapf, sign_mapf, sign_map_stack


def plot_maps(sign_mapf, alt_pos_mapf, azi_pos_mapf):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Sign Map")
    plt.imshow(sign_mapf, cmap="jet")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Altitude Map")
    plt.imshow(alt_pos_mapf, cmap="jet")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Azimuth Map")
    plt.imshow(azi_pos_mapf, cmap="jet")
    plt.colorbar()

    plt.show()


def run_segmentation_pipeline(alt_pos_mapf, azi_pos_mapf, sign_mapf, params=None):
    if params is None:
        params = default_params()

    raw_patch_map = getRawPatchMap(sign_mapf, params=params, isPlot=False)
    raw_patches = getRawPatches(sign_mapf, raw_patch_map, params=params, isPlot=False)
    det_map = getDeterminantMap(alt_pos_mapf, azi_pos_mapf, isPlot=False)
    ecc_map, ecc_mapf = getEccentricityMap(
        alt_pos_mapf, azi_pos_mapf, raw_patches, params=params, isPlot=False
    )
    patches_after_split = splitPatches(
        alt_pos_mapf,
        azi_pos_mapf,
        ecc_mapf,
        raw_patches,
        det_map,
        params=params,
        isPlot=False,
    )
    patches_after_merge, final_patches = mergePatches(
        patches_after_split,
        alt_pos_mapf,
        azi_pos_mapf,
        params=params,
        isPlot=False,
    )

    return {
        "rawPatchMap": raw_patch_map,
        "rawPatches": raw_patches,
        "detMap": det_map,
        "eccMap": ecc_map,
        "eccMapf": ecc_mapf,
        "patchesAfterSplit": patches_after_split,
        "patchesAfterMerge": patches_after_merge,
        "finalPatches": final_patches,
    }


def plot_vfs_with_patches(sign_mapf, final_patches, labels=True):
    plotVFSWithPatches(sign_mapf, final_patches, labels=labels)


def patch_to_contour(mask):
    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        return []
    contour = max(contours, key=len)
    coords = [(float(c[1]), float(c[0])) for c in contour]
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def build_contours_dict(patches):
    return {k: patch_to_contour(patches[k].array) for k in patches}


def contour_to_mask(contour_coords, shape):
    if contour_coords is None or len(contour_coords) < 3:
        return None
    pts = np.array(contour_coords, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return None

    pts_int = np.round(pts).astype(np.int32)
    pts_int[:, 0] = np.clip(pts_int[:, 0], 0, shape[1] - 1)
    pts_int[:, 1] = np.clip(pts_int[:, 1], 0, shape[0] - 1)
    pts_int = pts_int.reshape((-1, 1, 2))

    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts_int], 1)
    return mask.astype(bool)


def contours_to_patches(contours, shape, sign_map=None, default_sign=1):
    patches = {}
    for name, coords in contours.items():
        mask = contour_to_mask(coords, shape)
        if mask is None:
            continue
        if sign_map is None:
            sign = default_sign
        else:
            vals = sign_map[mask]
            mean_val = float(np.mean(vals)) if vals.size else 0.0
            sign = 1 if mean_val > 0 else -1 if mean_val < 0 else default_sign
        patches[name] = Patch(mask.astype(np.int8), sign)
    return patches


def contours_to_aligned_signmap(
    contours_path,
    transform_params,
    reference_vfs_path=None,
):
    """
    Reverse-apply transform parameters to contours stored in a pickle file.
    The transform parameters are from aligning a single-animal signmap to a reference
    signmap using vfs_composite.align_single_vfs_to_reference().
    """

    with open(contours_path, 'r') as file:
        contours = json.load(file)

    if reference_vfs_path is None:
        reference_vfs_path = transform_params.get("reference_vfs_path")
    if reference_vfs_path is None:
        raise ValueError("reference_vfs_path is required to compute the inverse transform.")

    ref_img = tifffile.imread(reference_vfs_path)
    h, w = ref_img.shape[:2]
    center = (w / 2.0, h / 2.0)

    rot = transform_params["rotation_deg"]
    scale = transform_params["scale_factor"]
    dx = transform_params["dx"]
    dy = transform_params["dy"]

    M = cv2.getRotationMatrix2D(center, rot, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    M3 = np.vstack([M, [0.0, 0.0, 1.0]])
    M_inv = np.linalg.inv(M3)

    def _apply_inverse(coords):
        if coords is None:
            return None
        arr = np.asarray(coords, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return coords
        ones = np.ones((arr.shape[0], 1), dtype=np.float32)
        pts = np.concatenate([arr, ones], axis=1)
        out = pts @ M_inv.T
        return [(float(x), float(y)) for x, y in out[:, :2]]

    return {name: _apply_inverse(coords) for name, coords in contours.items()}


def plot_patch_visual_spaces(
    patches,
    alt_pos_mapf,
    azi_pos_mapf,
    pixel_size=None,
    close_iter=None,
    max_cols=4,
):
    if pixel_size is None:
        pixel_size = DEFAULT_PARAMS["visualSpacePixelSize"]
    if close_iter is None:
        close_iter = DEFAULT_PARAMS["visualSpaceCloseIter"]

    if isinstance(patches, dict):
        items = list(patches.items())
    else:
        items = [(f"patch{i+1:02d}", p) for i, p in enumerate(patches)]

    n_items = len(items)
    if n_items == 0:
        raise ValueError("No patches provided for visual space plotting.")

    n_cols = min(max_cols, n_items)
    n_rows = int(np.ceil(n_items / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for idx, (name, patch) in enumerate(items):
        visual_space, _, _, _ = patch.getVisualSpace(
            alt_pos_mapf,
            azi_pos_mapf,
            pixelSize=pixel_size,
            closeIter=close_iter,
            isPlot=False,
        )
        plotVisualCoverage(visual_space, pixelSize=pixel_size, plotAxis=axes[idx])
        axes[idx].set_title(name)

    for ax in axes[n_items:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    return fig, axes


def plot_visual_spaces_from_contours(
    contours,
    alt_pos_mapf,
    azi_pos_mapf,
    sign_map=None,
    pixel_size=None,
    close_iter=None,
    max_cols=4,
):
    patches = contours_to_patches(
        contours,
        shape=alt_pos_mapf.shape,
        sign_map=sign_map,
    )
    return plot_patch_visual_spaces(
        patches,
        alt_pos_mapf,
        azi_pos_mapf,
        pixel_size=pixel_size,
        close_iter=close_iter,
        max_cols=max_cols,
    )


def resize_contours(vfs_contours, source_shape, widefield_path=None, target_shape=None):
    """Resize contour coordinates from source_shape to match dimensions of widefield image
    or target shape. Useful for when you need the contours to be on the same scale as 
    the widefield image."""
    if widefield_path is not None:
        wf = tifffile.imread(widefield_path)
        wf_h, wf_w = wf.shape[:2]

    elif target_shape is not None:
        wf_h, wf_w = target_shape
    
    else:
        print("No widefield path or target shape provided.")
        print("Returning unchanged contours dict.")
        return vfs_contours

    src_h, src_w = source_shape

    scale_x = wf_w / src_w
    scale_y = wf_h / src_h

    resized = {}
    for name, coords in vfs_contours.items():
        resized[name] = [(x * scale_x, y * scale_y) for x, y in coords]
    for name, coords in resized.items():
        resized[name] = [(int(round(x)), int(round(y))) for x, y in coords]
    return resized


def save_contours(vfs_contours, contours_path):
    os.makedirs(os.path.dirname(contours_path), exist_ok=True)
    with open(contours_path, 'w') as f:
        json.dump(vfs_contours, f, indent=4)
    return contours_path


def edit_contours_gui(contours_path, vfs_array=None, auto_show=True):
    return VFSSegment.from_saved_contours(
        contours_path=contours_path,
        vfs_array=vfs_array,
        auto_show=auto_show,
    )
