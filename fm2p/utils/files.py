# -*- coding: utf-8 -*-
"""
fm2p/utils/files.py

File read/write utilities for HDF5, YAML, and JSON.

Functions
---------
open_dlc_h5
    Read a DLC-format HDF5 output into a clean DataFrame.
write_h5
    Write a nested dict to an HDF5 file.
recursively_save_dict_contents_to_group
    Internal recursive helper for write_h5.
recursively_load_dict_contents_from_group
    Internal recursive helper for read_h5.
read_h5
    Read an HDF5 file back into a nested dict.
read_yaml
    Read a YAML config file.
write_yaml
    Write a dict to a YAML file.
write_group_h5
    Write a DataFrame split by a key column into a grouped HDF5.
get_group_h5_keys
    List top-level keys in a grouped HDF5.
read_group_h5
    Read one or more keys from a grouped HDF5 into a concatenated DataFrame.
find_hdf_overflow_columns
    Identify DataFrame columns with values outside int64 range.
fix_overflow_columns
    Clip overflow columns to int64 range.
normalize_for_hdf
    Normalize a DataFrame so all columns can be stored in HDF5.
read_json
    Read a JSON file into a dict.


DMM, December 2024
"""

import json
import yaml
import h5py
import datetime
import numpy as np
import pandas as pd

from .time import time2str


def open_dlc_h5(dlc_path, h5key=None):
    """ Read a DLC HDF5 output and return a clean DataFrame with flat column names.

    DLC stores keypoints as a multi-level column index
    (scorer, body_part, coord). This collapses it to '<body_part>_<coord>' strings.

    Parameters
    ----------
    dlc_path : str
        Path to the DLC .h5 file.
    h5key : str or None
        HDF5 key to read; None reads the default key.

    Returns
    -------
    pts : pd.DataFrame
        Keypoint coordinates with flattened column names.
    pt_loc_names : np.ndarray
        Array of the column names.
    """

    if h5key is None:
        pts = pd.read_hdf(dlc_path)
    else:
        pts = pd.read_hdf(dlc_path, key=h5key)
    pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
    pts = pts.rename(columns={pts.columns[n]: pts.columns[n].replace(' ', '_') for n in range(len(pts.columns))})
    pt_loc_names = pts.columns.values
    return pts, pt_loc_names


def write_h5(filename, dic):
    """ Write a nested dict to an HDF5 file.

    Parameters
    ----------
    filename : str
        Output path.
    dic : dict
        Arbitrarily nested dict; values must be numpy arrays, scalars, or strings.
    """

    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):
    """ Recursively write dict or list items into an HDF5 group. """

    if isinstance(dic, dict):
        iterator = dic.items()
    elif isinstance(dic, list):
        iterator = enumerate(dic)
    else:
        ValueError('Cannot save {} type'.format(type(dic)))
    for key, item in iterator:
        key = str(key)
        if isinstance(item, (np.ndarray, np.number, np.bool_, int, float, str, bytes, bool)):
            try:
                h5file[path + key] = item
            except TypeError:
                if isinstance(item, np.ndarray) and (item.dtype == object):
                    recursively_save_dict_contents_to_group(h5file, path + key + '/', item.item())
        elif isinstance(item, dict) or isinstance(item, list):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        elif isinstance(item, datetime.datetime):
            h5file[path + key] = time2str(item)
        else:
            raise ValueError('Cannot save {} type'.format(type(item)))


def recursively_load_dict_contents_from_group(h5file, path):
    """ Recursively read an HDF5 group into a nested dict. """

    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(
                h5file,
                path + key + '/'
            )

    return ans


def read_h5(filename, aslist=False):
    """ Read an HDF5 file into a nested dict (or list if aslist=True).

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.
    aslist : bool
        If True, convert the top-level dict to a list using integer keys as indices.

    Returns
    -------
    out : dict or list
    """

    with h5py.File(filename, 'r') as h5file:
        out = recursively_load_dict_contents_from_group(h5file, '/')
        if aslist:
            outl = [None for l in range(len(out.keys()))]
            for key, item in out.items():
                outl[int(key)] = item
            out = outl

        return out


def read_yaml(path):
    """ Read a YAML config file into a dict. """

    with open(path, 'r') as infile:
        try:
            contents = yaml.load(infile, Loader=yaml.FullLoader)
        except yaml.constructor.ConstructorError:
            infile.seek(0)
            contents = yaml.load(infile, Loader=yaml.UnsafeLoader)

    return contents


def write_yaml(path, contents):
    """ Write a dict to a YAML file. """

    with open(path, 'w') as outfile:
        yaml.dump(contents, outfile, default_flow_style=False)


def write_group_h5(df, savepath, repair_overflow=False):
    """ Write a DataFrame split by 'base_name' column into a grouped HDF5.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'base_name' column used as the split key.
    savepath : str
        Output HDF5 path.
    repair_overflow : bool
        If True, fix int64 overflow columns before writing.
    """

    if repair_overflow:
        df = normalize_for_hdf(fix_overflow_columns(df))
    split_key = 'base_name'
    split_list = df[split_key].unique()
    for i, sname in enumerate(split_list):
        print('Writing block {} of {} (key={})'.format(i + 1, len(split_list), sname))
        df[df[split_key] == sname].to_hdf(savepath, sname, mode='a')


def get_group_h5_keys(savepath):
    """ List top-level keys in a grouped HDF5 file. """

    with pd.HDFStore(savepath) as hdf:
        keys = [k.replace('/', '') for k in hdf.keys()]

    return keys


def read_group_h5(path, keys=None):
    """ Read one or more keys from a grouped HDF5 into a single concatenated DataFrame.

    Parameters
    ----------
    path : str
        Path to the HDF5.
    keys : str, list of str, or None
        Keys to read. None reads all keys.

    Returns
    -------
    df : pd.DataFrame
    """

    if type(keys) == str:
        df = pd.read_hdf(path, keys)
        return df
    if keys is None:
        keys = get_group_h5_keys(path)
    dfs = []
    for k in sorted(keys):
        _df = pd.read_hdf(path, k)
        dfs.append(_df)
    df = pd.concat(dfs)

    return df


def find_hdf_overflow_columns(df):
    """ Return column names whose values exceed the int64 range. """

    bad_cols = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_integer_dtype(s):
            if (s.min() < np.iinfo(np.int64).min) or (s.max() > np.iinfo(np.int64).max):
                bad_cols.append(col)
        elif pd.api.types.is_object_dtype(s):
            if s.apply(lambda x: isinstance(x, int) and (x > np.iinfo(np.int64).max or x < np.iinfo(np.int64).min)).any():
                bad_cols.append(col)

    return bad_cols


def fix_overflow_columns(df):
    """ Clip int64-overflow columns to the int64 range. """

    bad_cols = find_hdf_overflow_columns(df)
    if not bad_cols:
        return df
    df_fixed = df.copy()
    for col in bad_cols:
        print("Converting column '{}' to int64 (clipping values outside range).".format(col))
        df_fixed[col] = np.clip(df_fixed[col], np.iinfo(np.int64).min, np.iinfo(np.int64).max).astype('int64')

    return df_fixed


def normalize_for_hdf(df):
    """ Normalize a DataFrame so all columns and the index fit in HDF5.

    Converts int64-overflow columns and object columns containing dicts/lists
    to string representation so they can be stored without error.
    """

    INT64_MIN, INT64_MAX = np.iinfo(np.int64).min, np.iinfo(np.int64).max
    df_fixed = df.copy()
    try:
        if df_fixed.index.dtype == object or pd.api.types.is_integer_dtype(df_fixed.index):
            if getattr(df_fixed.index, 'min', lambda: 0)() < INT64_MIN or getattr(df_fixed.index, 'max', lambda: 0)() > INT64_MAX:
                print('Converting index to string (too large for int64).')
                df_fixed.index = df_fixed.index.astype(str)
    except Exception:
        df_fixed.index = df_fixed.index.astype(str)
    for col in df_fixed.columns:
        s = df_fixed[col]
        if pd.api.types.is_integer_dtype(s):
            if (s.min() < INT64_MIN) or (s.max() > INT64_MAX):
                print("Column '{}' too large -- converting to string.".format(col))
                df_fixed[col] = s.astype(str)
        elif pd.api.types.is_object_dtype(s):
            def clean(x):
                if isinstance(x, int) and not (INT64_MIN <= x <= INT64_MAX):
                    return str(x)
                if isinstance(x, (dict, list, tuple)):
                    return json.dumps(x)
                try:
                    str(x)
                    return x
                except Exception:
                    return repr(x)
            df_fixed[col] = s.map(clean).astype(str)
    return df_fixed


def read_json(file_path):
    """ Read a JSON file into a Python dict. """

    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
