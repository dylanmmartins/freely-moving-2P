

import os
import numpy as np
from tqdm import tqdm
import multiprocessing
from scipy.optimize import curve_fit

import fm2p

def fit_gauss(arr):
    """ Fit both +/- 2D gaussian peaks to 2D array
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
        tilt = Tx * (x - x0) + Ty * (y - y0)
        return g + B + tilt

    def fit_single_gaussian(initial_x0, initial_y0, is_positive=True):
        """Fit a Gaussian around a given initial center.\
        """
        A0 = (arr.max() - arr.min()) * (1 if is_positive else -1)
        B0 = np.median(arr)
        sx0 = sy0 = min(nx, ny) / 4

        guess = (A0, initial_x0, initial_y0, sx0, sy0, B0, 0, 0)

        popt, _ = curve_fit(
            gaussian2d,
            (Xf, Yf),
            Zf,
            p0=guess,
            maxfev=20000
        )

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

    # find extreme points
    y_pos, x_pos = np.unravel_index(arr.argmax(), arr.shape)
    y_neg, x_neg = np.unravel_index(arr.argmin(), arr.shape)

    # fit pos & neg gaussians
    pos_fit = fit_single_gaussian(x_pos, y_pos, is_positive=True)
    neg_fit = fit_single_gaussian(x_neg, y_neg, is_positive=False)

    return {
        'positive': pos_fit,
        'negative': neg_fit
    }


def within_pct(x1, x2, pct=15):
    pct = pct / 100
    return abs(x1 - x2) <= pct * abs(x2)


def gaus_eval(STA, STA1, STA2):
    # here, STA, STA1, etc. are the 2D STAs for a single cell, not a 3D
    # stack for all cells.

    eval0 = fit_gauss(STA)
    eval1 = fit_gauss(STA1)
    eval2 = fit_gauss(STA2)
    
    has_RF = []

    for di, direc in enumerate(['positive','negative']):

        passes_centroid_test = False
        passes_ratio_test = False
        passes_corr_test = False

        # is centroid of each split within 15% of the full STA's?
        for i in range(2): # x or y
            tomatch = eval0[direc]['centroid'][i]
            c1 = within_pct(eval1[direc]['centroid'][i], tomatch, 15)
            c2 = within_pct(eval2[direc]['centroid'][i], tomatch, 15)
            if c1 and c2:
                passes_centroid_test = True

        # make sure amplitude to baseline ratio is within 10% of full STA
        tomatch = eval0[direc]['amplitude']
        r1 = eval1[direc]['amplitude'] / eval1[direc]['baseline']
        r2 = eval2[direc]['amplitude'] / eval2[direc]['baseline']
        c1 = within_pct(r1, tomatch, 15)
        c2 = within_pct(r2, tomatch, 15)
        if c1 and c2:
            passes_ratio_test = True

        # check 2d cross correlation with full STA
        c1 = fm2p.corr2_coeff(STA, STA1) >= 0.3
        c2 = fm2p.corr2_coeff(STA, STA2) >= 0.3
        if c1 and c2:
            passes_corr_test = True

        # check for both positive and negative and ensure that at least one has a conserved receptive field
        if passes_corr_test and passes_centroid_test and passes_ratio_test:
            has_RF.append(1)
        else:
            has_RF.append(0)

    return has_RF


def fit_dual_2d_gaussians(sparse_noise_sta_path):

    data = fm2p.read_h5(sparse_noise_sta_path)

    STA = data['STA'].reshape(-1,768,1360)
    STA1 = data['STA1'].reshape(-1,768,1360)
    STA2 = data['STA2'].reshape(-1,768,1360)

    n_cells = np.size(STA, 0)

    n_proc = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=n_proc)

    param_mp = [pool.apply_async(gaus_eval, args=(STA[c], STA1[c], STA2[c])) for c in range(n_cells)]
    params_output = [result.get() for result in param_mp] # returns list of tuples

    has_RFs = np.zeros([n_cells,2])
    for c, vals in enumerate(params_output):
        has_RFs[c,:] = vals    # pos, then neg

    pool.close()

    savepath = os.path.join(os.path.split(sparse_noise_sta_path)[0], 'has_sparse_noise_STAs.npy')
    print('Saving {}'.format(savepath))
    np.save(savepath, has_RFs)



if __name__ == '__main__':

    sn_path = fm2p.select_file(
        'Select sparse noise receptive field HDF file.',
        filetypes=[('HDF','.h5'),]
    )

    fit_dual_2d_gaussians(sn_path)