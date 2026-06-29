# -*- coding: utf-8 -*-
"""
fm2p/utils/helper.py

Miscellaneous helper functions used across the pipeline.

Functions
---------
compute_kurtosis
    Excess (Fisher) kurtosis for each row of a 2D array.
split_xyl
    Split a DLC xyl DataFrame into separate x, y, likelihood DataFrames.
apply_liklihood_thresh
    Zero out DLC keypoints below a likelihood threshold.
str_to_bool
    Parse argparse boolean string flags.
make_default_cfg
    Load the package internals.yaml config.
to_dict_of_arrays
    Convert a DataFrame to a plain dict of numpy arrays.
blockPrint
    Redirect stdout to /dev/null (silence output).
enablePrint
    Restore stdout to the terminal.
fix_dict_dtype
    Recursively cast all numeric values in a dict to a given dtype.
nan_filt
    Drop columns that contain NaN in any of the input 2D arrays.
nan_interp
    Linearly interpolate over NaN values in a 1D array.
nan_interp_circular
    Linearly interpolate over NaN values in a 1D angular array (deg or rad).
calc_r2
    R-squared between observed and predicted values.
mask_non_nan
    Return arrays masked to frames where all inputs are non-NaN.
interp_short_gaps
    Fill NaN gaps shorter than max_gap with linear interpolation.
interp_short_gaps_circ
    Fill NaN gaps in an angular (degree) array with circular interpolation.
angular_diff_deg
    Circular difference between successive angles in degrees.
step_interp
    Zero-order-hold (step) interpolation.
bootstrap_stderr
    Bootstrap standard error of the median.
array_to_pil
    Convert a numpy array to a PIL Image in uint8 format.


DMM, December 2024
"""

import sys
import os
import pandas as pd
import numpy as np
from PIL import Image

from .paths import up_dir
from .files import read_yaml


def compute_kurtosis(traces):
    """ Excess (Fisher) kurtosis for each row of a 2D array.

    Parameters
    ----------
    traces : np.ndarray, shape (N,) or (N_cells, N_frames)

    Returns
    -------
    kurt : np.ndarray, shape (N_cells,)
    """

    if traces.ndim == 1:
        traces = traces[None, :]

    mean = np.mean(traces, axis=1, keepdims=True)
    std = np.std(traces, axis=1, keepdims=True)
    # Avoid division by zero for silent cells.
    std[std < 1e-9] = 1.0

    fourth_moment = np.mean((traces - mean) ** 4, axis=1, keepdims=True)
    kurt = fourth_moment / (std ** 4)

    return (kurt - 3.0).flatten()


def split_xyl(xyl):
    """ Split a DLC xyl DataFrame into separate x, y, likelihood DataFrames.

    Parameters
    ----------
    xyl : pd.DataFrame
        Columns named '<keypoint>_x', '<keypoint>_y', '<keypoint>_likelihood'.

    Returns
    -------
    x_vals, y_vals, l_vals : pd.DataFrame
    """

    names = list(xyl.columns.values)

    x_locs = []
    y_locs = []
    l_locs = []

    for loc_num in range(0, len(names)):
        loc = names[loc_num]
        if '_x' in loc:
            x_locs.append(loc)
        elif '_y' in loc:
            y_locs.append(loc)
        elif 'likeli' in loc:
            l_locs.append(loc)

    x_vals = xyl[x_locs]
    y_vals = xyl[y_locs]
    l_vals = xyl[l_locs]

    return x_vals, y_vals, l_vals


def apply_liklihood_thresh(x, l, threshold=0.99):
    """ Zero out DLC keypoints where likelihood falls below threshold.

    Parameters
    ----------
    x : pd.DataFrame
        Keypoint positions.
    l : pd.DataFrame
        Corresponding likelihood values.
    threshold : float
        Minimum likelihood; positions below become NaN.

    Returns
    -------
    x_vals : pd.DataFrame
    """

    thresh_arr = (l > threshold).astype(float).values
    x_vals1 = x.copy().values

    x_vals2 = pd.DataFrame((x_vals1 * thresh_arr), columns=x.columns)
    x_vals2[x_vals2 == 0.] = np.nan

    x_vals = x_vals2.copy()

    return x_vals


def str_to_bool(value):
    """ Parse a string boolean argument (from argparse) to a Python bool. """

    if isinstance(value, bool):
        return value

    if value.lower() in {'False', 'false', 'f', '0', 'no', 'n'}:
        return False

    elif value.lower() in {'True', 'true', 't', '1', 'yes', 'y'}:
        return True

    raise ValueError('{} is not a valid boolean value'.format(value))


def make_default_cfg():
    """ Load the package internals.yaml config dict. """

    internals_config_path = os.path.join(up_dir(__file__, 1), 'internals.yaml')
    cfg = read_yaml(internals_config_path)

    return cfg


def to_dict_of_arrays(df):
    """ Convert a DataFrame to a plain dict of numpy arrays. """

    seriesdict = {}
    for key in df.keys():
        seriesdict[key] = df[key].to_numpy()
    return seriesdict


def blockPrint():
    """ Redirect stdout to /dev/null (silence all print output). """

    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    """ Restore stdout to the terminal after blockPrint. """

    sys.stdout = sys.__stdout__


def fix_dict_dtype(d, totype):
    """ Recursively cast all numeric values in a dict to totype. """

    for k, v in d.items():
        if type(v) == dict:
            d[k] = fix_dict_dtype(d[k], totype)
            continue
        if type(v) == list:
            d[k] = [x.astype(totype) for x in v]
            continue
        if type(v) == np.ndarray:
            d[k] = v.astype(totype).tolist()
            continue
        d[k] = float(v)

    return d


def nan_filt(items, circular=False):
    """ Drop frames (columns) that contain NaN in any input 2D array.

    Parameters
    ----------
    items : list of np.ndarray, each shape (N_traces, N_frames)
        Last element may optionally be a bool (circular flag) or a list of
        per-item booleans indicating which arrays are angular.
    circular : bool
        Default circular mode for all arrays.

    Returns
    -------
    items_out : list of np.ndarray
        Same arrays with NaN-containing columns removed.
    """

    if any([type(arr) != np.ndarray for arr in items]):
        items = [np.array(arr) for arr in items]

    shapes = [arr.shape for arr in items]
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError('All input arrays must have the same shape.')

    assert items[0].ndim == 2

    circular = False

    if len(items) > 0 and isinstance(items[-1], (bool, np.bool_)):
        circular = bool(items[-1])
        items = items[:-1]

    per_item_circular = None
    if len(items) > 0 and isinstance(items[-1], (list, tuple, np.ndarray)):

        cand = items[-1]
        if len(cand) == len(items) - 1:

            per_item_circular = [bool(x) for x in cand]
            items = items[:-1]

    n_items = len(items)
    if per_item_circular is None:
        per_item_circular = [circular] * n_items

    items = [np.array(arr) for arr in items]

    for idx, arr in enumerate(items):
        if not per_item_circular[idx]:
            continue
        for r in range(arr.shape[0]):
            row = arr[r, :]
            if np.all(np.isnan(row)):
                continue

            cos_row = np.cos(np.deg2rad(row))
            sin_row = np.sin(np.deg2rad(row))

            try:
                cos_filled = nan_interp(cos_row)
                sin_filled = nan_interp(sin_row)
            except Exception:
                continue

            angle = np.rad2deg(np.arctan2(sin_filled, cos_filled))
            angle = ((angle + 180) % 360) - 180
            arr[r, :] = angle

    mask = ~np.isnan(np.vstack(items)).any(axis=0)
    items_out = [arr[:, mask] for arr in items]

    return items_out


def nan_interp(y):
    """ Linearly interpolate over NaN values in a 1D array. """

    y_interp = y.copy()
    nan_mask, x = np.isnan(y), lambda z: z.nonzero()[0]
    y_interp[nan_mask] = np.interp(x(nan_mask), x(~nan_mask), y[~nan_mask])
    return y_interp


def nan_interp_circular(y, deg=True):
    """ Linearly interpolate over NaN values in a 1D angular array.

    Handles wrap-around by interpolating sin/cos components separately.

    Parameters
    ----------
    y : array-like
        Angular values (degrees if deg=True, else radians). May contain NaN.
    deg : bool
        If True, input/output in degrees. Otherwise radians.

    Returns
    -------
    np.ndarray
        Array with NaN values replaced by circularly interpolated angles.
    """

    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError('nan_interp_circular expects a 1D array')

    if deg:
        factor = np.pi / 180.0
        inv = 180.0 / np.pi
    else:
        factor = 1.0
        inv = 1.0

    valid = ~np.isnan(y)
    if not np.any(valid):
        return y.copy()

    y_rad = y * factor

    x_comp = np.cos(y_rad)
    y_comp = np.sin(y_rad)

    if np.sum(valid) == 1:
        filled = np.full_like(y_rad, y_rad[valid][0])
        return (filled * inv) if deg else filled

    try:
        x_filled = nan_interp(x_comp)
        y_filled = nan_interp(y_comp)
    except Exception:
        return y.copy()

    angle = np.arctan2(y_filled, x_filled)
    if deg:
        angle = angle * inv
        angle = ((angle + 180) % 360) - 180

    return angle


def calc_r2(y, y_hat):
    """ R-squared between observed y and predicted y_hat. """

    y_mean = np.mean(y)
    sst = np.sum((y - y_mean) ** 2)
    sse = np.sum((y - y_hat) ** 2)
    r_squared = 1 - (sse / sst)
    return r_squared


def mask_non_nan(arrays):
    """ Return arrays masked to frames where all inputs are non-NaN.

    Parameters
    ----------
    arrays : list of array-like

    Returns
    -------
    masked_arrays : list of np.ndarray
    mask : np.ndarray of bool
    """

    arrays = [np.asarray(a) for a in arrays]

    mask = np.ones_like(arrays[0], dtype=bool)
    for a in arrays:
        mask &= ~np.isnan(a)

    masked_arrays = [a[mask] for a in arrays]

    return masked_arrays, mask


def interp_short_gaps(x, max_gap=5):
    """ Fill NaN gaps shorter than max_gap via linear interpolation.

    Parameters
    ----------
    x : array-like
        1D signal with NaN gaps.
    max_gap : int
        Maximum gap length (frames) to fill; longer gaps are left as NaN.

    Returns
    -------
    x_interp : np.ndarray
    """

    x = np.asarray(x, dtype=float)
    isnan = np.isnan(x)

    if not np.any(isnan):
        return x.copy()

    x_interp = x.copy()
    n = len(x)

    i = 0
    while i < n:
        if isnan[i]:
            start = i
            while i < n and isnan[i]:
                i += 1
            end = i

            gap_len = end - start
            if gap_len <= max_gap:
                left = start - 1
                right = end if end < n else None

                if left >= 0 and right is not None:
                    x_interp[start:end] = np.interp(
                        np.arange(start, end),
                        [left, right],
                        [x_interp[left], x_interp[right]]
                    )
        else:
            i += 1

    return x_interp


def interp_short_gaps_circ(x, max_gap=5):
    """ Fill NaN gaps in a degree-valued array with circular interpolation.

    Parameters
    ----------
    x : array-like
        1D angular signal in degrees with NaN gaps.
    max_gap : int
        Maximum gap length (frames) to fill.

    Returns
    -------
    x_interp : np.ndarray
    """

    x = np.asarray(x, dtype=float)
    isnan = np.isnan(x)

    if not np.any(isnan):
        return x.copy()

    x_interp = x.copy()
    n = len(x)

    i = 0
    while i < n:
        if isnan[i]:
            start = i
            while i < n and isnan[i]:
                i += 1
            end = i

            gap_len = end - start
            if gap_len <= max_gap:
                left = start - 1
                right = end if end < n else None

                if left >= 0 and right is not None:
                    a0 = x_interp[left]
                    a1 = x_interp[right]

                    # Shortest signed angular step across the gap.
                    delta = ((a1 - a0 + 180) % 360) - 180

                    t = (np.arange(start, end) - left) / (right - left)

                    x_interp[start:end] = (a0 + t * delta) % 360
        else:
            i += 1

    return x_interp


def angular_diff_deg(angles):
    """ Circular difference between successive angles in degrees.

    Parameters
    ----------
    angles : array-like

    Returns
    -------
    diffs : np.ndarray, length N-1
        Each value is in (-180, 180].
    """

    angles = np.asarray(angles)
    diffs = np.diff(angles)
    diffs = (diffs + 180) % 360 - 180

    return diffs


def step_interp(x, y, x_new):
    """ Zero-order-hold (step) interpolation.

    At each new x position, returns the y value from the most recent known x.

    Parameters
    ----------
    x : array-like
        Known x positions (sorted ascending).
    y : array-like
        Known y values.
    x_new : array-like
        New x positions to evaluate.

    Returns
    -------
    result : list
    """

    result = []
    j = 0

    for xn in x_new:
        while j + 1 < len(x) and xn >= x[j + 1]:
            j += 1
        result.append(y[j])

    return result


def bootstrap_stderr(data, n_boot=5000):
    """ Bootstrap standard error of the median.

    Parameters
    ----------
    data : array-like
    n_boot : int
        Number of bootstrap resamples.

    Returns
    -------
    se : float
    """

    data = np.asarray(data)
    n = len(data)
    boot_medians = np.empty(n_boot)

    for i in range(n_boot):
        sample = np.random.choice(data, size=n, replace=True)
        boot_medians[i] = np.median(sample)

    se = np.std(boot_medians, ddof=1)

    return se


def array_to_pil(arr):
    """ Convert a numpy array to a PIL Image, normalizing to uint8.

    Parameters
    ----------
    arr : np.ndarray or PIL.Image.Image
        Grayscale (H, W), RGB (H, W, 3), or RGBA (H, W, 4) array.

    Returns
    -------
    PIL.Image.Image
    """

    if isinstance(arr, Image.Image):
        return arr

    a = np.asarray(arr)
    if a.dtype != np.uint8:
        try:
            amin = float(np.nanmin(a))
            amax = float(np.nanmax(a))
        except Exception:
            amin, amax = 0.0, 1.0
        if amax == amin:
            a = np.zeros_like(a, dtype=np.uint8)
        else:
            a = ((a - amin) / (amax - amin) * 255.0).astype(np.uint8)

    if a.ndim == 2:
        return Image.fromarray(a, mode='L')
    if a.ndim == 3 and a.shape[2] == 3:
        return Image.fromarray(a, mode='RGB')
    if a.ndim == 3 and a.shape[2] == 4:
        return Image.fromarray(a, mode='RGBA')
    return Image.fromarray(a)
