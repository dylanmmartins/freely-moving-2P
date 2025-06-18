# -*- coding: utf-8 -*-
"""
Fit a 3-feature GLM to predict the spike rate of a neuron, given beahvioral inputs.

Functions
---------
fit_GLM()
    Fit a GLM for a 1D value of y (i.e., single cell).


Example usage
-------------
    $ python -m fm2p.summarize_revcorr -v 01
or alternatively, leave out the -v flag and select the h5 file from a file dialog box, followed
by the version number in a text box.
    $ python -m fm2p.summarize_revcorr

Author: DMM, 2025
"""


from tqdm import tqdm
import numpy as np


def fit_GLM(X, y, usebias=True):
    """ Fit a GLM for a 1D value of y (i.e., single cell).
    
    """
    # y is the spike data for a single cell
    # X is a 2D array. for prediction using {pupil, retiocentric, egocentric}, there are
    # 3 features. So, shape should be {#frames, 3}.
    # w will be the bias followed 

    n_samples, n_features = X.shape

    if usebias:
        # Add bias (intercept) term: shape becomes (n_samples, num_features+1)
        # bias is inserted before any of the weights for individual behavior variables, so
        # X_aug should be {bias, w_p, w_r, w_e}
        X_aug = np.hstack([np.ones((n_samples, 1)), X])
    elif not usebias:
        X_aug = X
    
    # Closed-form solution: w = (X^T X)^(-1) X^T y
    XtX = X_aug.T @ X_aug
    Xty = X_aug.T @ y
    weights = np.linalg.inv(XtX) @ Xty
    
    return weights


def compute_y_hat(X, y, w):

    n_samples, n_features = X.shape

    # Was there a bias computed when the GLM was fit?
    if np.size(w)==n_features+1:
        usebias = True

    if usebias:
        # Add bias to the spike rate data
        X_aug = np.hstack([np.ones((n_samples, 1)), X])
    else:
        X_aug = X.copy()

    y_hat = X_aug @ w

    mse = np.mean((y - y_hat)**2)

    return y_hat, mse



def fit_pred_GLM(spikes, pupil, retino, ego, speed):
    # spikes for a whole dataset of neurons, shape = {#frames, #cells}

    # First, threshold all inputs by the animal's speed, i.e., drop
    # frames in which the animal is stationary
    use = speed > 1.5 # cm/sec

    spikes = spikes[use,:]
    pupil = pupil[use]
    retino = retino[use]
    ego = ego[use]

    nFrames, nCells = np.shape(spikes)
    X_shared = np.stack([pupil, retino, ego], axis=1)


    # Drop any frame for which one of the behavioral varaibles was NaN
    # At the end, need to compute y_hat and then add NaN indices back in so that temporal
    # structure of the origional recording is preseved.
    _keepFmask = ~np.isnan(X_shared).any(axis=1)
    X_shared_ = X_shared.copy()[_keepFmask,:]
    spikes_ = spikes.copy()[_keepFmask,:]


    # Make train/test split by splitting frames into 20 chunks,
    # shuffling the order of those chunks, and then grouping them
    # into two groups at a 75/25 ratio. Same timepoint split will
    # be used across all cells.
    ncnk = 20
    traintest_frac = 0.75
    _len = np.sum(use)
    cnk_sz = _len // ncnk
    _all_inds = np.arange(0,_len)
    chunk_order = np.arange(ncnk)
    np.random.shuffle(chunk_order)
    train_test_boundary = int(ncnk * traintest_frac)

    train_inds = []
    test_inds = []
    for cnk_i, cnk in enumerate(chunk_order):
        _inds = _all_inds[(cnk_sz*cnk) : ((cnk_sz*(cnk+1)))]
        if cnk_i < train_test_boundary:
            train_inds.extend(_inds)
        elif cnk_i >= train_test_boundary:
            test_inds.extend(_inds)
    train_inds = np.sort(np.array(train_inds))
    test_inds = np.sort(np.array(test_inds))

    # GLM weights for all cells
    w = np.zeros([
        nCells,
        np.size(X_shared)+1     # number of features + a bias term
    ]) * np.nan
    # Predicted spike rate for the test data
    y_hat = np.zeros([
        nCells,
        len(test_inds)
    ]) * np.nan
    # Mean-squared error for each cell
    mse = np.zeros(nCells) * np.nan


    for cell in tqdm(range(nCells)):

        X_train_c = X_shared_[train_inds, cell].copy()
        X_test_c = X_shared_[test_inds, cell].copy()

        y_train_c = spikes_[train_inds, cell].copy()
        y_test_c = spikes_[test_inds, cell].copy()

        w_c = fit_GLM(X_train_c, y_train_c)

        y_hat_c, mse_c = compute_y_hat(X_test_c, y_test_c, w_c)

        w[cell,:] = w_c.copy()
        y_hat[cell,:] = y_hat_c.copy()
        mse[cell] = mse_c

        # Initialize model as a GLM with a Tweedie distribution.
        # model = linear_model.TweedieRegressor(
        #     alpha=0.01,
        #     power=0,
        #     max_iter=3000,
        #     tol=1e-5,
        #     fit_intercept=False
        # )
        # modelfit = model.fit(X_train.T, y_train)
        # _gz = modelfit.coef_[0]
        # _ret = modelfit.coef_[1]
        # _ego = modelfit.coef_[2]
        # weights_gz.append(_gz)
        # weights_ret.append(_ret)
        # weights_ego.append(_ego)
        # w = np.array([_gz, _ret, _ego])
        # pred = w @ X_test
        # scoreval = calc_score(y_test, pred)


    result = {
        'y_test_hat': y_hat,
        'GLM_weights': w,
        'GLM_MSE': mse,
        'speeduse': use,
        'keepFmask': _keepFmask,
        'X': X_shared_,
        'y': spikes_,
        'train_inds': train_inds,
        'test_inds': test_inds
    }

    return result


