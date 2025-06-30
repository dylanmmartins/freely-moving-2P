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
import multiprocessing


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
            learning_rate=0.001,
            epochs=5000,
            l1_penalty=0.01,
            l2_penalty=0.01,
        ):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty

        self.weights = None
        self.X_means = None
        self.X_stds = None

    def _mse(self, y, y_hat):
        return np.mean((y - y_hat)**2)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _softplus(self, z):
        return np.log1p(np.exp(-np.abs(z))) + np.maximum(z,0)
    
    def _zscore(self, X):

        X_z = np.zeros_like(X)
        savemeans = np.zeros(np.size(X,1))
        savestd = np.zeros(np.size(X,1))

        for feat in range(np.size(X,1)):
            mean_ = np.nanmean(X)
            std_ = np.nanstd(X)
            X_z[:, feat] = (X[:, feat] - mean_) / std_

        return X_z, savemeans, savestd
    
    def _apply_zscore(self, X):

        assert self.X_means is not None, 'Z score has not been computed, so it cannot yet be applied to novel arrays.'
        assert self.X_stds is not None, 'Z score has not been computed, so it cannot yet be applied to novel arrays.'

        X_z = np.zeros_like(X)

        for feat in range(np.size(X,1)):
            X_z[:, feat] = (X[:, feat] - self.X_means[feat]) / self.X_stds[feat]

        return X_z

    def _loss(self, y, y_hat):
        # 1e-8 is added for numerical stability to avoid log(0)

        # negative log-likelihood
        nll = np.mean(y_hat - y * np.log(y_hat + 1e-8))
        # L1 and L2 penalty terms
        l1 = self.l1_penalty * np.sum(np.abs(self.weights[1:]))
        l2 = self.l2_penalty * np.sum(self.weights[1:] ** 2)

        return nll + l1 + l2
    
    def fit(self, X, y, init=0.5, verbose=False):

        self.weights = np.ones(np.size(X,1)+1) * init

        # if y is 1D
        if len(np.shape(y)) != 2:
            y = y[:,np.newaxis]

        # scale values
        X_scaled, Xmeans, Xstds = self._zscore(X)
        self.X_means = Xmeans
        self.X_stds = Xstds

        # Add bias term
        X_bias = np.c_[np.ones(X_scaled.shape[0]), X_scaled]
        m = len(y)

        self.loss_history = np.zeros(self.epochs) * np.nan

        for epoch in range(self.epochs):
            z = np.dot(X_bias, self.weights)
            y_hat = self._softplus(z)

            # calculate loss
            lossval = self._loss(y, y_hat)
            self.loss_history[epoch] = lossval

            gradient = np.dot(X_bias.T, (y_hat - y.flatten())) / m
            # Apply L2 regularization (ridge)
            gradient[1:] += self.l2_penalty * 2 * self.weights[1:]
            # Apply L1 regularization (lasso) - subgradient method
            gradient[1:] += self.l1_penalty * np.sign(self.weights)[1:]

            self.weights -= self.learning_rate * gradient

            # explvar = self.score_explained_variance(y, y_hat)
            mse = self._mse(y, y_hat)

            if verbose and (epoch == 0):
                print('Initial pass:  loss={:.3}  MSE={:.3}'.format(lossval, mse))
            elif verbose and (epoch % 100 == 0):
                print('\rEpoch {}:  loss={:.3}  MSE={:.3}'.format(
                    epoch, lossval, mse), end='', flush=True)

    def _predict(self, X):

        assert self.X_means is not None
        assert self.X_stds is not None

        X_scaled = self._apply_zscore(X)

        X_bias = np.c_[np.ones(X_scaled.shape[0]), X_scaled]
        y_hat = self._softplus(np.dot(X_bias, self.weights))[:, np.newaxis]

        return y_hat

    def score_explained_variance(self, y, y_hat):
        # Similar to r^2 except that this will treat an offset as error, whereas
        # r^2 does not penalize for an offset, it just treats it as an intercept.
        # Could mult by 100 to get a percent. As currently written, max value is 1.0

        # Residual sum of squares
        ss_res = np.sum((y - y_hat)**2)
        # Total variance in y
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - ss_res / (ss_tot + 1e-8)
    
    def predict(self, X, y):
        # predict and score weights

        # if y is 1D
        if len(np.shape(y)) != 2:
            y = y[:,np.newaxis]

        # should be X_test and y_test as inputs
        y_hat = self._predict(X)
        
        mse = self._mse(y, y_hat)
        explained_variance = self.score_explained_variance(y, y_hat)

        return y_hat, mse, explained_variance

    # def predict_with_dropout(self, X, y):
        # Try every combination of weights being set to 0 so that the model performance with or without
        # behavioral measures can be compared. should i do a version that drops out the bias term? not sure
        # what the biological interpretation would be of this...

        # Number of weights excluding the bias term
        # nW = len(self.weights) - 1

        # How many combinations should I try?

    def get_weights(self):
        return self.weights
    
    def get_loss_history(self):
        return self.loss_history
    

def add_temporal_features(X, add_lags=1):

    nFrames, nFeats = np.shape(X)
    nFeatsOut = nFeats+(nFeats*add_lags)

    X_temporal = np.zeros([nFrames,nFeatsOut]) * np.nan

    i = 0
    for feat in range(nFeats):
        # Aligned data
        X_temporal[:,i] = X[:,feat].copy()
        i += 1

        for lag in range(1, add_lags+1):
            # Iterate through each lag position
            r = np.roll(X[:,feat].copy(), shift=-lag)
            r[lag:-lag] = np.nan
            X_temporal[:,i] = r
            i += 1

    assert i == nFeatsOut, 'Did not assign a temporal lag to each expected index.'

    return X_temporal

def multiprocess_model_fits(X_train, X_test,
                            y_train_c, y_test_c,
                            learning_rate, epochs, l1_penalty, l2_penalty):

    cell_model = GLM(
        learning_rate=learning_rate,
        epochs=epochs,
        l1_penalty=l1_penalty,
        l2_penalty=l2_penalty,
    )

    cell_model.fit(X_train, y_train_c, verbose=False)

    y_hat_c, mse_c, explvar_c = cell_model.predict(X_test, y_test_c)

    w_c = cell_model.get_weights()

    return (w_c, y_hat_c.flatten(), mse_c, explvar_c, cell_model.get_loss_history())


def fit_pred_GLM(spikes, pupil, retino, ego, speed, opts=None):
    # spikes for a whole dataset of neurons, shape = {#frames, #cells}

    if opts is None:
        learning_rate = 0.001
        epochs = 5000
        l1_penalty = 0.01
        l2_penalty = 0.01
        num_lags = 10
        use_multiprocess = True
    elif opts is not None:
        learning_rate = opts['learning_rate']
        epochs = opts['epochs']
        l1_penalty = opts['l1_penalty']
        l2_penalty = opts['l2_penalty']
        num_lags = opts['num_lags']
        use_multiprocess = opts['multiprocess']

    print('  -> Preening behavior data by speed and gaps in tracking.')

    # First, threshold all inputs by the animal's speed, i.e., drop
    # frames in which the animal is stationary
    speed = np.append(speed, speed[-1])
    use = speed > 1.5 # cm/sec

    spikes = spikes[use,:]
    pupil = pupil[use]
    retino = retino[use]
    ego = ego[use]

    _, nCells = np.shape(spikes)
    X_shared = np.stack([pupil, retino, ego], axis=1)

    # Drop any frame for which one of the behavioral varaibles was NaN
    # At the end, need to compute y_hat and then add NaN indices back in so that temporal
    # structure of the origional recording is preseved.
    _keepFmask = ~np.isnan(X_shared).any(axis=1)
    X_shared_ = X_shared.copy()[_keepFmask,:]
    spikes_ = spikes.copy()[_keepFmask,:]

    nFrames = np.sum(_keepFmask)

    print('     Of {} frames, {} are usable'.format(len(speed), nFrames))

    print('  -> Creating features for {} temporal lags.'.format(num_lags))

    # For each behavioral measure, add 9 previous time points so temporal
    # filters are learned. If it started w/ 3 features, will now have 30.
    if num_lags > 0:
        X_shared = add_temporal_features(X_shared, add_lags=num_lags)

    # Make train/test split by splitting frames into 20 chunks,
    # shuffling the order of those chunks, and then grouping them
    # into two groups at a 75/25 ratio. Same timepoint split will
    # be used across all cells.

    ncnk = 20
    traintest_frac = 0.75

    print('  -> Creating train/test split (nChunks={}, frac={})'.format(ncnk, traintest_frac))

    cnk_sz = nFrames // ncnk
    _all_inds = np.arange(0,nFrames)
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
        np.size(X_shared_,1)+1   # number of features + a bias term
    ]) * np.nan
    # Predicted spike rate for the test data
    y_hat = np.zeros([
        nCells,
        len(test_inds)
    ]) * np.nan
    # Mean-squared error for each cell
    mse = np.zeros(nCells) * np.nan
    explvar = np.zeros(nCells) * np.nan

    X_train = X_shared_[train_inds, :].copy()
    X_test = X_shared_[test_inds, :].copy()

    loss_histories = np.zeros([
        nCells,
        epochs
    ]) * np.nan

    if not use_multiprocess:

        for cell in tqdm(range(nCells)):

            y_train_c = spikes_[train_inds, cell].copy()
            y_test_c = spikes_[test_inds, cell].copy()

            cell_model = GLM(
                learning_rate=learning_rate,
                epochs=epochs,
                l1_penalty=l1_penalty,
                l2_penalty=l2_penalty,
            )

            cell_model.fit(X_train, y_train_c, verbose=True)

            y_hat_c, mse_c, explvar_c = cell_model.predict(X_test, y_test_c)

            w_c = cell_model.get_weights()

            w[cell,:] = w_c.copy()
            y_hat[cell,:] = y_hat_c.copy().flatten()
            mse[cell] = mse_c
            explvar[cell] = explvar_c
            loss_histories[cell,:] = cell_model.get_loss_history()

            X_scaled = cell_model._apply_zscore(X_shared_)

    elif use_multiprocess:

        n_proc = multiprocessing.cpu_count() - 1
        print('  -> Starting multiprocessing pool (number of workers: {}).'.format(n_proc))
        num_proc_batches = int(np.ceil(nCells / n_proc))
        print('     Models will be trained in {} batches of {} cells.'.format(num_proc_batches, n_proc))
        pool = multiprocessing.Pool(processes=n_proc)

        mp_param_set = [pool.apply_async(multiprocess_model_fits, args=(
                X_train,
                X_test,
                spikes_[train_inds, cell_num],
                spikes_[test_inds, cell_num],
                learning_rate,
                epochs,
                l1_penalty,
                l2_penalty
            )) for cell_num in range(nCells)]
        mp_outputs = [result.get() for result in mp_param_set]

        for mp_vals in mp_outputs:
            w[cell,:] = mp_vals[0]
            y_hat[cell,:] = mp_vals[1]
            mse[cell] = mp_vals[2]
            explvar[cell] = mp_vals[3]
            loss_histories[cell,:] = mp_vals[4]

            # Create a temporary model to apply z score to shared behavior data.
            # Useful for visualizations but not worth returning out of model since
            # it's shared across cells

            temp_model = GLM(
                learning_rate=learning_rate,
                epochs=epochs,
                l1_penalty=l1_penalty,
                l2_penalty=l2_penalty,
            )
            X_scaled, _, _ = temp_model._zscore(X_shared_)

        pool.close()

    print('  -> Across {} cells,'.format(nCells))
    print('     Mean R^2 = {}'.format(np.nanmean(explvar)))
    print('     Std of R^2 = {}'.format(np.nanstd(explvar)))
    print('     Mean MSE = {}'.format(np.nanmean(mse)))
    print('     Std of MSE = {}'.format(np.nanstd(mse)))

    result = {
        'GLM_weights': w,
        'speeduse': use,
        'keepFmask': _keepFmask,
        'X': X_shared_,
        'y': spikes_,
        'train_inds': train_inds,
        'test_inds': test_inds,
        'y_test_hat': y_hat,
        'MSE': mse,
        'explvar': explvar,
        'X_scaled': X_scaled,
        'loss_histories': loss_histories
    }

    return result
