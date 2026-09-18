# -*- coding: utf-8 -*-
"""
fm2p/print_model_stats.py

Print R^2 statistics from behavioral decoding (JSONs) and GLM predictions (HDF5).

Functions
---------
_r2_stats
    Mean, std and count over finite values.
load_decoding_r2
    Return (overall_r2, per_var_r2) from a decoding JSON (r_* fields are Pearson r).
load_glm_r2
    Return (light_r2, dark_r2) arrays pooled across all GLM HDF5 files and variables.
main
    Print decoding and GLM R^2 summaries for light and dark.


DMM, June 2026
"""

import glob
import json
import os

import h5py
import numpy as np

LIGHT_JSON = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'decode_across_areas_only50.json',
)
DARK_JSON = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'decode_across_areas_dark_only50.json',
)
BASE_DIR = '/home/dylan/Storage/freely_moving_data/_V1PPC'

GLM_COMPLETE_KEY = 'eyes_only'  # full model (all eye variables combined)
DEC_VARS = ('r_theta', 'r_phi', 'r_pitch', 'r_roll', 'r_yaw')


def _r2_stats(arr):
    """ Mean, std and count over finite values. """

    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float('nan'), float('nan'), 0
    return float(np.mean(arr)), float(np.std(arr)), len(arr)


def load_decoding_r2(path):
    """Return (overall_r2, per_var_r2) from a decoding JSON (r_* fields are Pearson r)."""

    with open(path) as f:
        data = json.load(f)
    overall = []
    per_var = {k: [] for k in DEC_VARS}
    for entry in data:
        for key in DEC_VARS:
            r = entry.get(key, float('nan'))
            if np.isfinite(r):
                r2 = r ** 2
                overall.append(r2)
                per_var[key].append(r2)
    return np.array(overall), {k: np.array(v) for k, v in per_var.items()}


def load_glm_r2():
    """Return (light_r2, dark_r2) arrays pooled across all GLM HDF5 files and variables."""

    pattern = os.path.join(BASE_DIR, '**', 'pytorchGLM_predictions_v09b.h5')
    h5_files = sorted(glob.glob(pattern, recursive=True))
    print('Found {} pytorchGLM_predictions_v09b.h5 files.'.format(len(h5_files)))

    light_r2, dark_r2 = [], []
    for path in h5_files:
        try:
            with h5py.File(path, 'r') as f:
                key_l = f'{GLM_COMPLETE_KEY}_trainLight_testLight_r2'
                key_d = f'{GLM_COMPLETE_KEY}_trainDark_testDark_r2'
                if key_l not in f or key_d not in f:
                    continue
                arr_l = f[key_l][()]
                arr_d = f[key_d][()]
                if arr_l.shape != arr_d.shape:
                    print('  Shape mismatch {}: light={} dark={}, skipping'.format(path, arr_l.shape, arr_d.shape))
                    continue
                light_r2.extend(arr_l.tolist())
                dark_r2.extend(arr_d.tolist())
        except Exception as e:
            print('  Error reading {}: {}'.format(path, e))

    return np.array(light_r2), np.array(dark_r2)


def main():
    """ Print decoding and GLM R^2 summaries for light and dark. """

    print('\n' + '=' * 60)
    print('MODEL PREDICTION STATISTICS')
    print('=' * 60)

    # Behavioral decoding from JSON files.
    print('\n--- Cross-validated behavioral decoding (Pearson r to R^2) ---')
    light_dec, light_dec_by_var = load_decoding_r2(LIGHT_JSON)
    dark_dec,  dark_dec_by_var  = load_decoding_r2(DARK_JSON)

    var_labels = {'r_theta': 'theta', 'r_phi': 'phi',
                  'r_pitch': 'pitch', 'r_roll': 'roll', 'r_yaw': 'yaw'}

    m, s, n = _r2_stats(light_dec)
    print('  Light (all vars): R^2 = {:.3f} +/- {:.3f}  (n={} session values)'.format(m, s, n))
    for key in DEC_VARS:
        m, s, n = _r2_stats(light_dec_by_var[key])
        if n > 0:
            print('    {:<6}: R^2 = {:.3f} +/- {:.3f}  (n={})'.format(var_labels[key], m, s, n))

    m, s, n = _r2_stats(dark_dec)
    print('  Dark  (all vars): R^2 = {:.3f} +/- {:.3f}  (n={} session values)'.format(m, s, n))
    for key in DEC_VARS:
        m, s, n = _r2_stats(dark_dec_by_var[key])
        if n > 0:
            print('    {:<6}: R^2 = {:.3f} +/- {:.3f}  (n={})'.format(var_labels[key], m, s, n))

    # GLM neural predictions from HDF5 files
    print('\n--- PyTorch GLM neural predictions (pytorchGLM_predictions_v09b.h5) ---')
    light_glm, dark_glm = load_glm_r2()

    m, s, n = _r2_stats(light_glm)
    print('  Light (trainLight-to-testLight): R^2 = {:.3f} +/- {:.3f}  (n={} cells)'.format(m, s, n))
    m, s, n = _r2_stats(dark_glm)
    print('  Dark  (trainDark-to-testDark):   R^2 = {:.3f} +/- {:.3f}  (n={} cells)'.format(m, s, n))

    print('=' * 60 + '\n')


if __name__ == '__main__':
    main()
