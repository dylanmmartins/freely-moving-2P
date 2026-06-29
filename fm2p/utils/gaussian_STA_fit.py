# -*- coding: utf-8 -*-
"""
fm2p/utils/gaussian_STA_fit.py

Fit 2D Gaussians to sparse-noise spike-triggered averages (STAs).

Best run only on cells already expected to have clean receptive fields --
fitting is slow enough that batch-running all cells is impractical.

Functions
---------
fit_gauss
    Fit a 2D Gaussian to a single STA frame.
within_pct
    Check whether two values agree within a given percentage.
gaus_eval
    Evaluate STA quality via Gaussian fit and split-half correlation.
gaussian_STA_fit
    Load an STA HDF5 file, fit Gaussians in parallel, and save results.


DMM, December 2025
"""

if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
    __package__ = 'fm2p.utils'

import os
import numpy as np
from tqdm import tqdm
import multiprocessing
from scipy.optimize import curve_fit

from .correlation import corr2_coeff
from .files import read_h5
from .gui_funcs import select_file


def fit_gauss(arr):
    """ Fit a tilted 2D Gaussian to arr and return centroid/shape parameters.

    Parameters
    ----------
    arr : np.ndarray, shape (ny, nx)
        STA frame (values in arbitrary units).

    Returns
    -------
    pos_fit : dict
        Keys: 'centroid', 'amplitude', 'baseline', 'tilt',
        'sigma_x', 'sigma_y', 'amp_baseline_ratio'.
    """

    ny, nx = arr.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    Xf = X.ravel()
    Yf = Y.ravel()
    Zf = arr.ravel()

    def gaussian2d(coords, A, x0, y0, sx, sy, B, Tx, Ty):
        x, y = coords
        g = A * np.exp(
            -(((x - x0) ** 2) / (2 * sx ** 2)
              + ((y - y0) ** 2) / (2 * sy ** 2))
        )
        # Linear tilt term lets the baseline vary across the field.
        tilt = Tx * (x - x0) + Ty * (y - y0)
        return g + B + tilt

    def fit_single_gaussian(initial_x0, initial_y0, is_positive=True):

        A0 = (arr.max() - arr.min()) * (1 if is_positive else -1)
        B0 = np.median(arr)
        sx0 = sy0 = min(nx, ny) / 4

        guess = (A0, initial_x0, initial_y0, sx0, sy0, B0, 0, 0)

        try:
            popt, _ = curve_fit(
                gaussian2d,
                (Xf, Yf),
                Zf,
                p0=guess,
                maxfev=20000
            )
        except RuntimeError:
            popt = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

        A, x0, y0, sx, sy, B, Tx, Ty = popt
        amp_baseline_ratio = A / B if B != 0 else np.inf

        return {
            'centroid': (x0, y0),
            'amplitude': A,
            'baseline': B,
            'tilt': (Tx, Ty),
            'sigma_x': sx,
            'sigma_y': sy,
            'amp_baseline_ratio': amp_baseline_ratio,
        }

    y_pos, x_pos = np.unravel_index(arr.argmax(), arr.shape)

    pos_fit = fit_single_gaussian(x_pos, y_pos, is_positive=True)

    return pos_fit


def within_pct(x1, x2, pct=15):
    """ Return True if x1 and x2 agree within pct percent of x2. """

    pct = pct / 100
    return abs(x1 - x2) <= pct * abs(x2)


def gaus_eval(STA, STA1, STA2):
    """ Evaluate STA quality: Gaussian fit on |STA| + split-half correlation.

    Parameters
    ----------
    STA : np.ndarray
        Full STA (averaged over all spikes).
    STA1 : np.ndarray
        First-half STA.
    STA2 : np.ndarray
        Second-half STA.

    Returns
    -------
    gauss_eval : dict
        Gaussian fit parameters plus 'corr2d' (split-half Pearson r).
    """

    corr = corr2_coeff(STA1, STA2)

    gauss_eval = fit_gauss(np.abs(STA))
    gauss_eval['corr2d'] = corr

    return gauss_eval


def gaussian_STA_fit(sparse_noise_sta_path):
    """ Load an STA HDF5, fit Gaussians in parallel, and save results.

    Parameters
    ----------
    sparse_noise_sta_path : str
        Path to an HDF5 file containing 'STA' with shape (N_cells, 768, 1360)
        (or a flat version that gets reshaped to that).
    """

    data = read_h5(sparse_noise_sta_path)

    STA = data['STA'].reshape(-1, 768, 1360)

    n_cells = np.size(STA, 0)

    n_proc = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=n_proc)

    print('Pool started with {} CPUs.'.format(n_proc))
    print('Fitting gaussian on splits and computing similarity metrics...')

    with tqdm(total=n_cells) as pbar:

        results = []

        def collect(res):
            results.append(res)
            pbar.update()

        param_mp = [pool.apply_async(fit_gauss, args=(STA[c],), callback=collect) for c in range(n_cells)]
        params_output = [result.get() for result in param_mp]

    centroids = np.zeros([n_cells, 2]) * np.nan
    amplitudes = np.zeros([n_cells]) * np.nan
    baselines = np.zeros([n_cells]) * np.nan
    sigmas = np.zeros([n_cells, 2]) * np.nan
    tilts = np.zeros([n_cells, 2]) * np.nan

    for c in range(len(params_output)):
        try:
            centroids[c, 0] = params_output[c]['centroid'][0]  # x
            centroids[c, 1] = params_output[c]['centroid'][1]  # y
            amplitudes[c] = params_output[c]['amplitude']
            baselines[c] = params_output[c]['baseline']
            sigmas[c, 0] = params_output[c]['sigma_x']
            sigmas[c, 1] = params_output[c]['sigma_y']
            tilts[c, 0] = params_output[c]['tilt'][0]
            tilts[c, 1] = params_output[c]['tilt'][1]
        except Exception:
            pass

    pool.close()

    savepath = os.path.join(os.path.split(sparse_noise_sta_path)[0], 'has_sparse_noise_STAs_v2.npz')
    print('Saving {}'.format(savepath))
    np.savez(
        savepath,
        centroids=centroids,
        amplitudes=amplitudes,
        baselines=baselines,
        sigmas=sigmas,
        tilts=tilts
    )


if __name__ == '__main__':

    hdf_path = select_file(
        'Select sparse noise preproc file.',
        [('HDF', '.h5'), ]
    )
    gaussian_STA_fit(hdf_path)
