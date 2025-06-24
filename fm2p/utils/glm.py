# -*- coding: utf-8 -*-
"""
Fit a 3-feature GLM to predict the spike rate of a neuron, given beahvioral inputs.

Functions
---------
fit_GLM()
    Fit a GLM for a 1D value of y (i.e., single cell).



TODO: At some point, modify the GLM so that multiple frames (in perserved
presered temporal sequence) can be used to predict a single frame's firing
rate.

Author: DMM, 2025
"""


from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
from itertools import combinations


def fit_closed_GLM(X, y, usebias=True):
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



class GLM:
    def __init__(
            self,
            learning_rate=0.001, epochs=5000, l1_penalty=0.01, l2_penalty=0.01,
            bias=True, num_weights=3):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.use_bias = bias
        if bias is True:
            num_weights += 1
        self.weights = np.zeros(num_weights) # 3 parameters and a bias term

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _loss(self, y_true, y_pred):
        m = len(y_true)
        log_loss = -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
        l1 = self.l1_penalty * np.sum(np.abs(self.weights[1:]))
        l2 = self.l2_penalty * np.sum(self.weights[1:] ** 2)
        return log_loss + l1 + l2
    
    def _fit_no_bias(self, X, y):

        # if y is 1D
        if len(np.shape(y)) != 2:
            y = y[:,np.newaxis]

        # scale values
        scalerX = preprocessing.StandardScaler().fit(X)
        X_scaled = scalerX.transform(X)
        scalerY = preprocessing.StandardScaler().fit(y)
        y_scaled = scalerY.transform(y)

        m = len(y_scaled)

        for epoch in range(self.epochs):
            z = np.dot(X_scaled, self.weights)
            y_pred = self._sigmoid(z)

            gradient = np.dot(X_scaled.T, (y_pred - y_scaled.flatten())) / m
            # Apply L2 regularization (ridge)
            gradient[1:] += self.l2_penalty * 2 * self.weights
            # Apply L1 regularization (lasso) - subgradient method
            gradient[1:] += self.l1_penalty * np.sign(self.weights)

            self.weights -= self.learning_rate * gradient

        self.scalerX = scalerX
        self.scalerY = scalerY

    def fit(self, X, y):

        if not self.use_bias:
            self._fit_no_bias(X,y)
            return

        # if y is 1D
        if len(np.shape(y)) != 2:
            y = y[:,np.newaxis]

        # scale values
        scalerX = preprocessing.StandardScaler().fit(X)
        X_scaled = scalerX.transform(X)
        scalerY = preprocessing.StandardScaler().fit(y)
        y_scaled = scalerY.transform(y)

        X_bias = np.c_[np.ones(X_scaled.shape[0]), X_scaled]  # Add bias term
        m = len(y_scaled)

        for epoch in range(self.epochs):
            z = np.dot(X_bias, self.weights)
            y_pred = self._sigmoid(z)

            gradient = np.dot(X_bias.T, (y_pred - y_scaled.flatten())) / m
            # Apply L2 regularization (ridge)
            gradient[1:] += self.l2_penalty * 2 * self.weights[1:]
            # Apply L1 regularization (lasso) - subgradient method
            gradient[1:] += self.l1_penalty * np.sign(self.weights[1:])

            self.weights -= self.learning_rate * gradient

        self.scalerX = scalerX
        self.scalerY = scalerY

    def _predict(self, X):

        X_scaled = self.scalerX.transform(X)

        if X.shape[1] != 3:
            raise ValueError("Input X must have exactly 3 features for this GLM.")
        X_bias = np.c_[np.ones(X_scaled.shape[0]), X_scaled]
        y_hat = self._sigmoid(np.dot(X_bias, self.weights))[:, np.newaxis]

        y_hat_invtrans = self.scalerY.inverse_transform(y_hat)

        return y_hat_invtrans, y_hat
    
    def predict(self, X, y):
        # predict and score weights

        # if y is 1D
        if len(np.shape(y)) != 2:
            y = y[:,np.newaxis]

        y_scaled = self.scalerY.transform(y)

        # should be X_test and y_test as inputs
        y_hat_invtrans, y_hat = self._predict(X)
        
        mse = np.mean((y_scaled - y_hat)**2)
        mse_invtrans = np.mean((y - y_hat_invtrans)**2)

        return y_hat, y_hat_invtrans, mse, mse_invtrans

    # def predict_with_dropout(self, X, y):
        # Try every combination of weights being set to 0 so that the model performance with or without
        # behavioral measures can be compared. should i do a version that drops out the bias term? not sure
        # what the biological interpretation would be of this...

        # Number of weights excluding the bias term
        # nW = len(self.weights) - 1

        # How many combinations should I try?


    def get_weights(self):
        return self.weights



def fit_pred_GLM(spikes, pupil, retino, ego, speed, opts=None):
    # spikes for a whole dataset of neurons, shape = {#frames, #cells}


    if opts is None:
        learning_rate = 0.001
        epochs = 5000
        l1_penalty = 0.01
        l2_penalty = 0.01
        num_weights = 3
        use_bias = True
    elif opts is not None:
        learning_rate = opts['learning_rate']
        epochs = opts['epochs']
        l1_penalty = opts['l1_penalty']
        l2_penalty = opts['l2_penalty']
        num_weights = opts['num_weights']
        use_bias = opts['use_bias']


    # First, threshold all inputs by the animal's speed, i.e., drop
    # frames in which the animal is stationary
    speed = np.append(speed, speed[-1])
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
    _len = np.sum(_keepFmask)
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
    train_inds = np.sort(np.array(train_inds)).astype(int)
    test_inds = np.sort(np.array(test_inds)).astype(int)

    # GLM weights for all cells
    w = np.zeros([
        nCells,
        np.size(X_shared_,1)+1     # number of features + a bias term
    ]) * np.nan
    # Predicted spike rate for the test data
    y_hat = np.zeros([
        nCells,
        len(test_inds)
    ]) * np.nan
    # Mean-squared error for each cell
    mse = np.zeros(nCells) * np.nan

    # for scaled values (not the inverse transformed arrays)
    y_hat1 = np.zeros([
        nCells,
        len(test_inds)
    ]) * np.nan
    mse1 = np.zeros(nCells) * np.nan

    X_train = X_shared_[train_inds, :].copy()
    X_test = X_shared_[test_inds, :].copy()


    for cell in tqdm(range(nCells)):

        y_train_c = spikes_[train_inds, cell].copy()
        y_test_c = spikes_[test_inds, cell].copy()

        cell_model = GLM(
            learning_rate=learning_rate,
            epochs=epochs,
            l1_penalty=l1_penalty,
            l2_penalty=l2_penalty,
            use_bias=use_bias,
            num_weights=num_weights
        )

        cell_model.fit(X_train, y_train_c)

        y_hat_c, y_hat_cit, mse_c, mse_cit = cell_model.predict(X_test, y_test_c)

        w_c = cell_model.get_weights()

        w[cell,:] = w_c.copy()
        y_hat[cell,:] = y_hat_cit.copy().flatten()
        mse[cell] = mse_cit

        y_hat1[cell,:] = y_hat_c.copy().flatten()
        mse1[cell] = mse_c

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
        'y_test_hat': y_hat1,
        'GLM_weights': w,
        'GLM_MSE': mse1,
        'speeduse': use,
        'keepFmask': _keepFmask,
        'X': X_shared_,
        'y': spikes_,
        'train_inds': train_inds,
        'test_inds': test_inds,
        'y_test_hat_scaled': y_hat,
        'GLM_MSE_scaled': mse
    }

    return result


