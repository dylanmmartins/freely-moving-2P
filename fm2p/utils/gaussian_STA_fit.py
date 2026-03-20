# -*- coding: utf-8 -*-
"""
Fit 2D Gaussian to sparse noise spike-triggered averages and
extract parameters for review. This is slow, so it's best to
just run on the cells you already expect to have a good receptive
field.

Written Jan 2026, DMM
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
    pct = pct / 100
    return abs(x1 - x2) <= pct * abs(x2)


def gaus_eval(STA, STA1, STA2):

    corr = corr2_coeff(STA1, STA2)

    gauss_eval = fit_gauss(np.abs(STA))
    gauss_eval['corr2d'] = corr

    return gauss_eval


def gaussian_STA_fit(sparse_noise_sta_path):

    data = read_h5(sparse_noise_sta_path)

    STA = data['STA'].reshape(-1,768,1360)

    n_cells = np.size(STA, 0)

    n_proc = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=n_proc)

    print('  -> Pool started with {} CPUs.'.format(n_proc))
    
    print('  -> Fitting gaussian on splits and computing similarity metrics.'.format(n_proc))

    with tqdm(total=n_cells) as pbar:

        results = []
        def collect(res):
            results.append(res)
            pbar.update()
            
        param_mp = [pool.apply_async(fit_gauss, args=(STA[c]), callback=collect) for c in range(n_cells)]
        params_output = [result.get() for result in param_mp]

    centroids = np.zeros([n_cells, 2]) * np.nan
    amplitudes = np.zeros([n_cells]) * np.nan
    baselines = np.zeros([n_cells]) * np.nan
    sigmas = np.zeros([n_cells, 2]) * np.nan
    tilts = np.zeros([n_cells, 2]) * np.nan

    for c in range(len(params_output)):
        try:
            centroids[c,0] = params_output[c]['centroid'][0] # x
            centroids[c,1] = params_output[c]['centroid'][1] # y
            amplitudes[c] = params_output[c]['amplitude']
            baselines[c] = params_output[c]['baseline']
            sigmas[c,0] = params_output[c]['sigma_x']
            sigmas[c,1] = params_output[c]['sigma_y']
            tilts[c,0] = params_output[c]['tilt'][0]
            tilts[c,1] = params_output[c]['tilt'][1]
        except:
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
        [('HDF', '.h5'),]
    )

    gaussian_STA_fit(hdf_path)

