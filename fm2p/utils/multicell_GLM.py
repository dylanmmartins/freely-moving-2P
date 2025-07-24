
import os
from tqdm import tqdm
import numpy as np
import multiprocessing
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import RobustScaler

import fm2p


class multicell_GLM:
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
    
    def _tanh(self, z):
        return np.tanh(z)
    
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
        nll = np.nanmean(y_hat - y * np.log(y_hat + 1e-8))
        # L1 and L2 penalty terms
        l1 = self.l1_penalty * np.sum(np.abs(self.weights[1:]))
        l2 = self.l2_penalty * np.sum(self.weights[1:] ** 2)

        return nll + l1 + l2
    
    def fit(self, X, y, init=0.5, verbose=False):

        self.weights = np.ones([
            np.size(X,0)+1,
            np.size(y,0)
        ]) * init

        # if y is 1D
        if len(np.shape(y)) != 2:
            y = y[:,np.newaxis]

        # Add bias term (in a sense, the baseline firing rate)
        X_bias = np.c_[np.ones(X.shape[1]), X.T]
        m = np.size(y,1)

        self.loss_history = np.zeros(self.epochs) * np.nan

        for epoch in range(self.epochs):
            z = np.dot(X_bias, self.weights)
            y_hat = self._tanh(z).T

            # calculate loss
            lossval = self._loss(y, y_hat)
            self.loss_history[epoch] = lossval

            gradient = np.dot(X_bias.T, (y_hat - y).T) / m
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

        X_bias = np.c_[np.ones(X.shape[1]), X.T]
        y_hat = self._tanh(np.dot(X_bias, self.weights))

        return y_hat
    

    def make_split(self, X, y, nanfilt=True):

        if nanfilt:
            mask1d = np.sum(np.isnan(y), 0) == 0
            # mask2d = np.broadcast_to(mask1d, X.shape)
            y = y[:,mask1d]
            X = X[:,mask1d]

        train_inds = []
        test_inds = []
        ss = ShuffleSplit(n_splits=1, test_size=0.25, random_state=20)
        for i, (train_index, test_index) in enumerate(ss.split(X.T)):
            # train_inds.append(np.array(sorted(train_index)))
            # test_inds.append(np.array(sorted(test_index)))
            train_inds = np.array(sorted(train_index))
            test_inds = np.array(sorted(test_index))

        self.train_inds = train_inds
        self.test_inds = test_inds
        self.nan_mask = mask1d

        X_train = X[:,train_inds]
        y_train = y[:,train_inds]

        X_test = X[:,test_inds]
        y_test = y[:,test_inds]

        return X_train, y_train, X_test, y_test
    

    def get_train_test_inds(self):
        return self.train_inds, self.test_inds, self.nan_mask


    def apply_transform(self, A):

        n_feats = np.size(A,0)

        assert n_feats == len(self.scalers)

        A_scale = np.zeros_like(A) * np.nan

        for feat in range(n_feats):
            A_scale[feat,:] = self.scalers[feat].transform(A[feat,:][:,np.newaxis]).flatten()

        return A_scale


    def fit_apply_transform(self, A):
        # a must be array-like. can be 1D or 2D. could be x or y, but
        # cannot apply to both.

        # for each feature along axis=0, train an independent scaler
        n_feats = np.size(A,0)

        self.scalers = []

        for feat in range(n_feats):
            self.scalers.append(RobustScaler().fit(A[feat,:][:,np.newaxis]))

        A_scale = self.apply_transform(A)

        return A_scale


    def apply_inverse_transform(self, A):

        n_feats = np.size(A,0)

        assert n_feats == len(self.scalers)

        A_invtran = np.zeros_like(A) * np.nan

        for feat in range(n_feats):
            A_invtran[feat,:] = self.scalers[feat].inverse_transform(A[feat,:][:,np.newaxis]).flatten()

        return A_invtran
    

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
        # assume that y is already scaled.

        # if y is 1D
        if len(np.shape(y)) != 2:
            y = y[:,np.newaxis]

        # should be X_test and y_test as inputs
        y_hat = self._predict(X)
        
        mse = self._mse(y, y_hat.T)
        explained_variance = self.score_explained_variance(y, y_hat.T)

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
    

def run_pupil_model(data):

    # 0.05
    learning_rate = 0.1
    epochs = 2500
    l1_penalty = 0
    l2_penalty = 0.001

    X = data['norm_spikes'].copy()

    theta = data['theta_interp'].copy()
    phi = data['phi_interp'].copy()
    y = np.vstack([theta, phi])

    model = fm2p.multicell_GLM(
        learning_rate=learning_rate,
        epochs=epochs,
        l1_penalty=l1_penalty,
        l2_penalty=l2_penalty,
    )

    y = model.fit_apply_transform(y)

    Xp_train, yp_train, Xp_test, yp_test = model.make_split(X, y)

    model.fit(Xp_train, yp_train, verbose=True)

    yp_hat, pupil_mse, pupil_explvar = model.predict(Xp_test, yp_test)

    pupil_weights = model.get_weights()
    train_inds, test_inds, nan_mask = model.get_train_test_inds()
    yp_test_unscaled = y.copy()[:,test_inds]

    # Test on the train data to see if it at least predicts that
    yp_train_hat, train_mse, explvar_train = model.predict(Xp_train, yp_train)

    results = {
        'weights': pupil_weights,
        'y_hat': model.apply_inverse_transform(yp_hat.T),
        'MSE': pupil_mse,
        'explvar': pupil_explvar,
        'loss_history': model.get_loss_history(),
        'X_train': Xp_train,
        'X_test': Xp_test,
        'y_train': model.apply_inverse_transform(yp_train),
        'y_test': model.apply_inverse_transform(yp_test),
        'y_test_unscaled': yp_test_unscaled,
        'y_hat_train': model.apply_inverse_transform(yp_train_hat.T),
        'MSE_train': train_mse,
        'explvar_train': explvar_train
    }

    return results

def run_retina_model(data):

    learning_rate = 0.05
    epochs = 2500
    l1_penalty = 0
    l2_penalty = 0.001

    X = data['norm_spikes'].copy()

    # retinocentric = data['retinocentric'].copy()
    # pillar_size = data['pillar_size'].copy()
    # y = np.vstack([retinocentric, pillar_size])
    retinocentric = data['retinocentric'].copy()

    # because it's radial, map the angle to the unit circiel so circular
    # continuity is built in.
    y = np.vstack([
        np.sin(np.deg2rad(retinocentric)),
        np.cos(np.deg2rad(retinocentric))
    ])
    y[0,:] = fm2p.convfilt(fm2p.nan_interp(y[0,:]), 3)
    y[1,:] = fm2p.convfilt(fm2p.nan_interp(y[1,:]), 3)

    model = fm2p.multicell_GLM(
        learning_rate=learning_rate,
        epochs=epochs,
        l1_penalty=l1_penalty,
        l2_penalty=l2_penalty,
    )

    y = model.fit_apply_transform(y)

    Xp_train, yp_train, Xp_test, yp_test = model.make_split(X, y)

    model.fit(Xp_train, yp_train, verbose=True)

    y_hat, pupil_mse, pupil_explvar = model.predict(Xp_test, yp_test)

    pupil_weights = model.get_weights()
    train_inds, test_inds, nan_mask = model.get_train_test_inds()
    yp_test_unscaled = y.copy()[:,test_inds]

    # Test on the train data to see if it at least predicts that
    yp_train_hat, train_mse, explvar_train = model.predict(Xp_train, yp_train)

    y_hat_invtran = model.apply_inverse_transform(y_hat.T)

    y_hat_retino = np.rad2deg(np.arctan2(y_hat_invtran[0,:], y_hat_invtran[1,:]))
    y_test_retino = np.rad2deg(np.arctan2(yp_test_unscaled[0,:], yp_test_unscaled[1,:]))

    results = {
        'weights': pupil_weights,
        'y_hat': model.apply_inverse_transform(y_hat.T),
        'MSE': pupil_mse,
        'explvar': pupil_explvar,
        'loss_history': model.get_loss_history(),
        'X_train': Xp_train,
        'X_test': Xp_test,
        'y_train': model.apply_inverse_transform(yp_train),
        'y_test': model.apply_inverse_transform(yp_test),
        'y_test_unscaled': yp_test_unscaled,
        'y_hat_train': model.apply_inverse_transform(yp_train_hat.T),
        'MSE_train': train_mse,
        'explvar_train': explvar_train,
        'y_hat_retino': y_hat_retino,
        'y_test_retino': y_test_retino,
        'retinocentric': retinocentric
    }

    return results

def run_body_model(data):

    learning_rate = 0.1
    epochs = 5000
    l1_penalty = 0
    l2_penalty = 0.001

    X = data['norm_spikes'].copy()

    egocentric = data['egocentric'].copy()
    distance = data['dist_to_pillar'].copy()
    y = np.vstack([
        np.sin(np.deg2rad(egocentric)),
        np.cos(np.deg2rad(egocentric)),
        distance
    ])
    y[0,:] = fm2p.convfilt(fm2p.nan_interp(y[0,:]), 3)
    y[1,:] = fm2p.convfilt(fm2p.nan_interp(y[1,:]), 3)

    model = fm2p.multicell_GLM(
        learning_rate=learning_rate,
        epochs=epochs,
        l1_penalty=l1_penalty,
        l2_penalty=l2_penalty,
    )

    y = model.fit_apply_transform(y)

    Xp_train, yp_train, Xp_test, yp_test = model.make_split(X, y)

    model.fit(Xp_train, yp_train, verbose=True)

    y_hat, pupil_mse, pupil_explvar = model.predict(Xp_test, yp_test)

    pupil_weights = model.get_weights()
    train_inds, test_inds, nan_mask = model.get_train_test_inds()
    yp_test_unscaled = y.copy()[:,test_inds]

    yp_train_hat, train_mse, explvar_train = model.predict(Xp_train, yp_train)

    y_hat_invtran = model.apply_inverse_transform(y_hat.T)

    y_hat_retino = np.rad2deg(np.arctan2(y_hat_invtran[0,:], y_hat_invtran[1,:]))
    y_test_retino = np.rad2deg(np.arctan2(yp_test_unscaled[0,:], yp_test_unscaled[1,:]))

    results = {
        'weights': pupil_weights,
        'y_hat': model.apply_inverse_transform(y_hat.T),
        'MSE': pupil_mse,
        'explvar': pupil_explvar,
        'loss_history': model.get_loss_history(),
        'X_train': Xp_train,
        'X_test': Xp_test,
        'y_train': model.apply_inverse_transform(yp_train),
        'y_test': model.apply_inverse_transform(yp_test),
        'y_test_unscaled': yp_test_unscaled,
        'y_hat_train': model.apply_inverse_transform(yp_train_hat.T),
        'MSE_train': train_mse,
        'explvar_train': explvar_train,
        'y_hat_ego': y_hat_retino,
        'y_test_ego': y_test_retino,
        'egocentric': egocentric
    }

    return results


def glm2():

    # preproc_path = r'Z:\Mini2P_data\250626_DMM_DMM037_ltdk\fm1\250626_DMM_DMM037_fm_01_preproc.h5'
    models = 'P'
    preproc_path = r'T:\Mini2P\250514_DMM_DMM046_LPaxons\fm1\250514_DMM_DMM046_fm_1_preproc.h5'

    data = fm2p.read_h5(preproc_path)

    all_model_results = {}

    if 'P' in models:
        all_model_results['pupil'] = run_pupil_model(data)

    if 'R' in models:
        all_model_results['retina'] = run_retina_model(data)

    if 'B' in models:
        all_model_results['body'] = run_body_model(data)
    

    savedir = os.path.split(preproc_path)[0]
    basename = os.path.split(preproc_path)[1][:-11]
    savepath = os.path.join(savedir, '{}_multicell_GLM_results_v2_LP_axonal_pupil_pred.h5'.format(basename))

    fm2p.write_h5(savepath, all_model_results)

    print('\nSaved {}'.format(savepath))


if __name__ == '__main__':

    glm2()