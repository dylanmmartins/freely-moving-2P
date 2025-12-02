# -*- coding: utf-8 -*-
"""
Miscillaneous helper functions.

Functions
---------
split_xyl(xyl)
    Split the xyl dataframe into x, y, and likelihood dataframes.
apply_liklihood_thresh(x, l, threshold=0.99)
    Apply a likelihood threshold to a dataframe.
str_to_bool(value)
    Parse strings to read argparse flag entries in as bool.

Author: DMM, 2024
"""

import sys
import os
import pandas as pd
import numpy as np

import fm2p

def split_xyl(xyl):
    """ Split the xyl dataframe into x, y, and likelihood dataframes.
    
    Parameters
    ----------
    xyl : pd.DataFrame
        Dataframe containing x, y, and likelihood data.
    
    Returns
    -------
    x_vals : pd.DataFrame
        Dataframe containing x values.
    y_vals : pd.DataFrame
        Dataframe containing y values.
    l_vals : pd.DataFrame
        Dataframe containing likelihood values.
    """

    names = list(xyl.columns.values)

    x_locs = []
    y_locs = []
    l_locs = []

    # seperate the lists of point names into x, y, and likelihood
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
    """ Apply a likelihood threshold to a dataframe.

    Parameters
    ----------
    x : pd.DataFrame
        Dataframe containing x or y values.
    l : pd.DataFrame
        Dataframe containing likelihood values.
    threshold : float, optional
        Likelihood threshold to apply. The default is 0.99.
    
    Returns
    -------
    x_vals : pd.DataFrame
        Dataframe containing x or y values with likelihood threshold applied. Values
        below the reshold are set to NaN.
    """

    thresh_arr = (l>threshold).astype(float).values
    x_vals1 = x.copy().values

    x_vals2 = pd.DataFrame((x_vals1 * thresh_arr), columns=x.columns)
    x_vals2[x_vals2==0.] = np.nan

    x_vals = x_vals2.copy()

    return x_vals


def str_to_bool(value):
    """ Parse strings to read argparse flag entries in as bool.
    
    Parameters
    ----------
    value : str
        Input value.

    Returns
    -------
    bool
        Input value as a boolean.
    """

    if isinstance(value, bool):
        return value
    
    if value.lower() in {'False', 'false', 'f', '0', 'no', 'n'}:
        return False
    
    elif value.lower() in {'True', 'true', 't', '1', 'yes', 'y'}:
        return True
    
    raise ValueError(f'{value} is not a valid boolean value')


def make_default_cfg():
    internals_config_path = os.path.join(fm2p.up_dir(__file__, 1), 'internals.yaml')
    cfg = fm2p.read_yaml(internals_config_path)

    return cfg

def to_dict_of_arrays(df):
    seriesdict = {}
    for key in df.keys():
        seriesdict[key] = df[key].to_numpy()
    return seriesdict

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def fix_dict_dtype(d, totype):
    
    for k,v in d.items():
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
    # 'items' must be a list of arrays or list-like objects

    if any([type(arr)!=np.ndarray for arr in items]):
        items = [np.array(arr) for arr in items]

    shapes = [arr.shape for arr in items]
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError('All input arrays must have the same shape.')
    
    assert items[0].ndim == 2

    # backward-compatible behavior: if circular is not requested, simply
    # remove columns where any item has a NaN (original behavior).
    # If circular interpolation is requested, interpolate NaNs on the
    # circular items (per-row) using sin/cos interpolation and then
    # apply the same column mask to return synchronized columns.
    
    # Support calling signature nan_filt(items, circular=False) in the
    # future; detect if user passed a circular flag by looking for
    # keyword-like invocation. To remain backward compatible we check
    # whether a trailing boolean was provided inside 'items'.
    circular = False
    # If last element of items is the literal True/False and not an array,
    # treat it as the circular flag and remove it from items list.
    if len(items) > 0 and isinstance(items[-1], (bool, np.bool_)):
        circular = bool(items[-1])
        items = items[:-1]

    # Alternatively, allow circular to be a list/tuple specifying per-item
    # circular flags by passing it as the second argument to the function in
    # future. But for now, also support passing a list as the last element.
    per_item_circular = None
    if len(items) > 0 and isinstance(items[-1], (list, tuple, np.ndarray)):
        # if last element is list-like of bools and its length matches items
        cand = items[-1]
        if len(cand) == len(items)-1:
            # this means user passed items + per_item_circular list
            per_item_circular = [bool(x) for x in cand]
            items = items[:-1]

    n_items = len(items)
    if per_item_circular is None:
        per_item_circular = [circular] * n_items

    # Ensure arrays again (in case we sliced off flag)
    items = [np.array(arr) for arr in items]

    # Perform circular-aware interpolation where requested
    for idx, arr in enumerate(items):
        if not per_item_circular[idx]:
            # do not interpolate: leave as-is
            continue

        # arr is 2D: iterate rows and interpolate across columns
        for r in range(arr.shape[0]):
            row = arr[r, :]
            if np.all(np.isnan(row)):
                # nothing to do
                continue

            # interpolate sin/cos components to respect angular wrapping
            cos_row = np.cos(np.deg2rad(row))
            sin_row = np.sin(np.deg2rad(row))

            # nan_interp expects 1D numpy arrays
            try:
                cos_filled = nan_interp(cos_row)
                sin_filled = nan_interp(sin_row)
            except Exception:
                # If interpolation fails (e.g., not enough valid points),
                # skip and leave original row unchanged
                continue

            angle = np.rad2deg(np.arctan2(sin_filled, cos_filled))
            # wrap to [-180, 180]
            angle = ((angle + 180) % 360) - 180
            arr[r, :] = angle

    # Now apply mask: keep only columns where all items are finite
    mask = ~np.isnan(np.vstack(items)).any(axis=0)
    items_out = [arr[:, mask] for arr in items]

    return items_out


def nan_interp(y):
    # interpolate linearly over NaNs, filling each position
    # base on https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    y_interp = y.copy()
    nan_mask, x = np.isnan(y), lambda z: z.nonzero()[0]
    y_interp[nan_mask] = np.interp(x(nan_mask), x(~nan_mask), y[~nan_mask])
    return y_interp


def nan_interp_circular(y, deg=True):
    """
    Interpolate 1D circular data with NaNs by interpolating sin/cos components.

    Parameters
    ----------
    y : array-like
        1D array of angular values. May contain NaNs. Values are expected in
        degrees by default and in the range [-180, 180]. If deg=False, input
        is assumed to be in radians and in range [-pi, pi].
    deg : bool, optional
        Whether the input (and output) are in degrees. Default True.

    Returns
    -------
    ndarray
        The input array with NaNs replaced by interpolated circular values.

    Notes
    -----
    This function converts angles to unit circle components, linearly
    interpolates each component over NaNs, then recomputes the angle with
    arctan2. If there are insufficient valid points to interpolate (e.g., all
    NaNs or a single valid point), the original array is returned unchanged
    (with NaNs left as-is for the all-NaN case, or filled with the single
    value for the single-point case).

    Example
    -------
    >>> a = np.array([170, np.nan, -170])
    >>> nan_interp_circular(a)
    array([170., 180., -170.])  # middle value interpolated correctly across wrap
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

    # mask of valid (non-NaN) entries
    valid = ~np.isnan(y)
    if not np.any(valid):
        # all NaNs: nothing to do
        return y.copy()

    y_rad = y * factor

    # convert to cos/sin components
    x_comp = np.cos(y_rad)
    y_comp = np.sin(y_rad)

    # if only one valid point, fill all NaNs with that angle
    if np.sum(valid) == 1:
        filled = np.full_like(y_rad, y_rad[valid][0])
        return (filled * inv) if deg else filled

    # interpolate components separately using nan_interp helper
    try:
        x_filled = nan_interp(x_comp)
        y_filled = nan_interp(y_comp)
    except Exception:
        # fallback: return copy
        return y.copy()

    # recompute angle and map to desired range
    angle = np.arctan2(y_filled, x_filled)
    if deg:
        angle = angle * inv
        # wrap to [-180, 180]
        angle = ((angle + 180) % 360) - 180

    return angle


def calc_r2(y, y_hat):
    # y is the ground truth/observed values
    # y_hat is prediction
    y_mean = np.mean(y)
    sst = np.sum((y - y_mean)**2)
    sse = np.sum((y - y_hat)**2)
    r_squared = 1 - (sse / sst)
    return r_squared


def mask_non_nan(arrays):
    # example usage:
    # [head_, eye_] = mask_non_nan([head, eye])

    arrays = [np.asarray(a) for a in arrays]

    mask = np.ones_like(arrays[0], dtype=bool)
    for a in arrays:
        mask &= ~np.isnan(a)

    masked_arrays = [a[mask] for a in arrays]

    return masked_arrays


def interp_short_gaps(x, max_gap=5):
    # Linearly interpolate over NaNs in a 1D array, but only for gaps shorter than `max_gap`.

    x = np.asarray(x, dtype=float)
    isnan = np.isnan(x)

    if not np.any(isnan):
        return x.copy()

    x_interp = x.copy()
    n = len(x)

    # Indices of non-NaN values
    not_nan_idx = np.where(~isnan)[0]
    if len(not_nan_idx) == 0:
        return x_interp  # all NaNs

    # Iterate through NaN runs
    i = 0
    while i < n:
        if isnan[i]:
            start = i
            while i < n and isnan[i]:
                i += 1
            end = i  # exclusive

            gap_len = end - start
            if gap_len <= max_gap:
                # Identify interpolation bounds
                left = start - 1
                right = end if end < n else None

                if left >= 0 and right is not None:
                    # Linear interpolation
                    x_interp[start:end] = np.interp(
                        np.arange(start, end),
                        [left, right],
                        [x_interp[left], x_interp[right]]
                    )
        else:
            i += 1

    return x_interp

def angular_diff_deg(angles):
    """
    Compute the difference between successive angles in degrees, correctly handling wraparound at 360 deg
    """
    
    angles = np.asarray(angles)
    diffs = np.diff(angles)
    diffs = (diffs + 180) % 360 - 180

    return diffs


def step_interp(x, y, x_new):
    """
    Step interpolation (zero-order hold).
    
    x: known x positions (sorted)
    y: known y values
    x_new: new x positions to evaluate
    
    Returns a list of y-values corresponding to x_new.
    """
    result = []
    j = 0  # index for original x
    
    for xn in x_new:
        # Advance j while xn is beyond the next known x
        while j + 1 < len(x) and xn >= x[j + 1]:
            j += 1
        result.append(y[j])
    
    return result