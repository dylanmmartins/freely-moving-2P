# -*- coding: utf-8 -*-
"""
Filter functions.

Functions
--------
convfilt(y, box_pts=10)
    Smooth values in an array using a convolutional window.
sub2ind(array_shape, rows, cols)
    Convert subscripts to linear indices.
nanmedfilt(A, sz=5)
    Median filtering of 1D or 2D array while ignoring NaNs.

Author: DMM, 2024
"""


import numpy as np


def convfilt(y, box_pts=10, circular=False):

    box = np.ones(box_pts) / box_pts
    if not circular:
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError('convfilt with circular=True supports 1D arrays only')
    valid = ~np.isnan(y)
    theta = np.deg2rad(np.where(valid, y, 0.0))
    cos_comp = np.cos(theta) * valid.astype(float)
    sin_comp = np.sin(theta) * valid.astype(float)
    cos_conv = np.convolve(cos_comp, box, mode='same')
    sin_conv = np.convolve(sin_comp, box, mode='same')
    norm = np.convolve(valid.astype(float), box, mode='same')
    with np.errstate(invalid='ignore', divide='ignore'):
        cos_smooth = np.where(norm > 0, cos_conv / norm, np.nan)
        sin_smooth = np.where(norm > 0, sin_conv / norm, np.nan)
    angle = np.rad2deg(np.arctan2(sin_smooth, cos_smooth))
    angle = ((angle + 180) % 360) - 180

    return angle


def sub2ind(array_shape, rows, cols):
    # Equivalent to Matlab's sub2ind function https://www.mathworks.com/help/matlab/ref/sub2ind.html

    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1

    return ind


def nanmedfilt(A, sz=5):
    # Median filtering of 1D or 2D array while ignoring NaNs.

    if type(sz) == int:
        sz = np.array([sz, sz])
    if any(sz % 2 == 0):
        print('kernel size must be odd')
    margin = np.array((sz-1) // 2)
    if len(np.shape(A)) == 1:
        A = np.expand_dims(A, axis=0)
    AA = np.zeros(np.squeeze(np.array(np.shape(A)) + 2*np.expand_dims(margin, 0)))
    AA[:] = np.nan
    AA[margin[0]:-margin[0], margin[1]:-margin[1]] = A

    iB, jB = np.mgrid[0:sz[0], 0:sz[1]]
    isB = sub2ind(np.shape(AA.T), jB, iB) + 1

    iA, jA = np.mgrid[0:np.size(A,0), 0:np.size(A,1)]
    iA += 1
    isA = sub2ind(np.shape(AA.T), jA, iA)
    idx = isA + np.expand_dims(isB.flatten('F')-1, 1)
    B = np.sort(AA.T.flatten()[idx-1], 0)
    j = np.any(np.isnan(B), 0)
    last = np.zeros([1, np.size(B,1)]) + np.size(B,0)
    last[:, j] = np.argmax(np.isnan(B[:, j]),0)
    
    M = np.zeros([1, np.size(B,1)])
    M[:] = np.nan
    valid = np.where(last > 0)[1]
    mid = (1 + last) / 2

    i1 = np.floor(mid[:, valid])
    i2 = np.ceil(mid[:, valid])
    i1 = sub2ind(np.shape(B.T), valid, i1)
    i2 = sub2ind(np.shape(B.T), valid, i2)

    M[:,valid] = 0.5*(B.flatten('F')[i1.astype(int)-1] + B.flatten('F')[i2.astype(int)-1])
    M = np.reshape(M, np.shape(A))

    return M


def convolve2d(image, kernel):

    kernel = np.flipud(np.fliplr(kernel))
    kH, kW = kernel.shape
    pad_h = kH // 2
    pad_w = kW // 2
    if image.ndim == 3:
        output = np.zeros_like(image)
        for c in range(image.shape[2]):
            output[..., c] = convolve2d(image[..., c], kernel)
        return output
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    H, W = image.shape
    result = np.zeros_like(image, dtype=float)
    for i in range(H):
        for j in range(W):
            region = padded[i:i+kH, j:j+kW]
            result[i, j] = np.sum(region * kernel)

    return result


def rolling_average(arr, window=8, ensure_shape_match=False):

    shape = (arr.shape[0] - window + 1, window) + arr.shape[1:]
    strides = (arr.strides[0],) + arr.strides
    windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    rolled_array = windows.mean(axis=1)
    if ensure_shape_match:
        band = np.zeros([
            window,
            np.size(rolled_array,1),
            np.size(rolled_array,2)
        ])
        return np.concatenate([band, rolled_array, band], axis=0)

    return rolled_array


def rolling_average_1d(data, window_size):

    if window_size < 1:
        raise ValueError("Window size must be at least 1")

    data = np.asarray(data)
    result = np.empty_like(data, dtype=float)

    half_w = window_size // 2

    for i in range(len(data)):
        start = max(0, i - half_w)
        end = min(len(data), i + half_w + 1)
        result[i] = np.mean(data[start:end])

    return result

