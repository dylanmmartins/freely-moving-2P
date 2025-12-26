
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
            model,
            learning_rate=0.001,
            epochs=5000,
            l1_penalty=0.,
            l2_penalty=0.,
        ):

        self.modeltype = model

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty

        self.weights = None
        self.X_means = None
        self.X_stds = None

    def _mse(self, y, y_hat):
        """ Mean squared error.
        """
        return np.mean((y - y_hat)**2)

    def _sigmoid(self, z):
        """ Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-z))
    
    def _softplus(self, z):
        """Softplus activation function.
        """
        return np.log1p(np.exp(-np.abs(z))) + np.maximum(z,0)
    
    def _tanh(self, z):
        """ Tanh activation function.
        """
        return np.tanh(z)
    
    def _zscore(self, X):
        """ Z-score for scaling input.
        """

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

    def _neg_log_likelihood_loss(self, y, y_hat):
        """ Loss function.

        Loss is calculated as negative log likleihood, with L1 and L2
        regularization terms. If the penalty parameters of the class are
        set to 0, no penalty is applied.
        """

        # Negative log-likelihood
        # 1e-8 is added for numerical stability to avoid log(0)
        nll = np.nanmean(y_hat - y * np.log(y_hat + 1e-8))
        # L1 and L2 penalty terms
        l1 = self.l1_penalty * np.sum(np.abs(self.weights[1:]))
        l2 = self.l2_penalty * np.sum(self.weights[1:] ** 2)

        return nll + l1 + l2
    
    def _cross_entropy_loss(self, y, y_hat):
        # for classification problems

        # avoid log(0)
        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

        cel = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        mcel = np.nanmean(cel)

        return mcel
    
    def _classification_fit(self, X, y, init=0, verbose=False, thresh=0.33):
        # categorical prediction
        
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

            y_hat = self._sigmoid(np.dot(X_bias, self.weights).T)
            y_hat = (y_hat>thresh).astype(int)

            # Calculate loss
            lossval = self._cross_entropy_loss(y, y_hat)
            self.loss_history[epoch] = lossval

            gradient = np.dot(X_bias.T, (y_hat - y).T) / m
            
            # Update weights
            self.weights -= self.learning_rate * gradient

            if verbose and (epoch == 0):
                print('Initial pass:  loss={:.3}'.format(lossval))
            elif verbose and (epoch % 100 == 0):
                print('\rEpoch {}:  loss={:.3}'.format(
                    epoch, lossval), end='', flush=True)        
        print('\n')

    
    def fit(self, X, y, init=0.5, verbose=False):
        """ Fit the model.
        """

        if self.modeltype == 'regress':
            self._regression_fit(X, y, init, verbose)
        elif self.modeltype == 'classify':
            self._classification_fit(X, y, init, verbose)


    def _regression_fit(self, X, y, init=0.5, verbose=False):

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

            y_hat = np.dot(X_bias, self.weights).T

            # Calculate loss
            lossval = self._neg_log_likelihood_loss(y, y_hat)
            self.loss_history[epoch] = lossval

            gradient = np.dot(X_bias.T, (y_hat - y).T) / m
            
            # Apply L2 regularization (ridge)
            gradient[1:] += self.l2_penalty * 2 * self.weights[1:]
            
            # Apply L1 regularization (lasso) - subgradient method
            gradient[1:] += self.l1_penalty * np.sign(self.weights)[1:]

            # Update weights
            self.weights -= self.learning_rate * gradient

            # explvar = self.score_explained_variance(y, y_hat)
            mse = self._mse(y, y_hat)

            if verbose and (epoch == 0):
                print('Initial pass:  loss={:.3}  MSE={:.3}'.format(lossval, mse))
            elif verbose and (epoch % 100 == 0):
                print('\rEpoch {}:  loss={:.3}  MSE={:.3}'.format(
                    epoch, lossval, mse), end='', flush=True)        
        print('\n')


    def _predict(self, X):

        X_bias = np.c_[np.ones(X.shape[1]), X.T]
        # remove this link function, which makes it behave more like a NN than a GLM. Instead,
        # return the identity.
        # y_hat = self._tanh(np.dot(X_bias, self.weights))
        y_hat = np.dot(X_bias, self.weights)

        return y_hat
    

    def make_split(self, X, y, nanfilt=True, test_size=0.25):
        # NaN at any position in X or y will cause issues. Mask out NaNs, which
        # cannot appear in spike data (X), but do show up in the behavior data (y)
        # because of gaps in tracking.
        if nanfilt:
            mask1d = np.sum(np.isnan(y), 0) == 0
            y = y[:,mask1d]
            X = X[:,mask1d]

        train_inds = []
        test_inds = []
        ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=20)
        for i, (train_index, test_index) in enumerate(ss.split(X.T)):
            train_inds = np.array(sorted(train_index))
            test_inds = np.array(sorted(test_index))

        self.train_inds = train_inds
        self.test_inds = test_inds
        self.nan_mask = mask1d

        X_train = X[:,train_inds]
        y_train = y[:,train_inds]

        X_test = X[:,test_inds]
        y_test = y[:,test_inds]

        # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

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

        if self.modeltype == 'regress':
            y_hat, err = self._reg_pred(X, y)
        elif self.modeltype == 'classify':
            y_hat, err = self._clas_pred(X, y)
        else:
            print('Unmatched model')

        return y_hat, err


    def _clas_pred(self, X, y, thresh=0.33):

        # if y is 1D
        if len(np.shape(y)) != 2:
            y = y[:,np.newaxis]
        
        X_bias = np.c_[np.ones(X.shape[1]), X.T]
        y_hat = self._sigmoid(np.dot(X_bias, self.weights))

        y_hat = (y_hat>thresh).astype(int)

        epsilon = 1e-15
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

        _err = self._cross_entropy_loss(y, y_hat)

        return y_hat, _err


    def _reg_pred(self, X, y):
        # predict and score weights
        # assume that y is already scaled.

        # if y is 1D
        if len(np.shape(y)) != 2:
            y = y[:,np.newaxis]

        # should be X_test and y_test as inputs
        y_hat = self._predict(X)
        
        mse = self._mse(y, y_hat.T)
        # explained_variance = self.score_explained_variance(y, y_hat.T)

        return y_hat, mse

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
    

def run_pupil_model(data, use_dFF=False, use_light=True):

    print('  -> Fitting pupil-prediction model.')

    learning_rate = 0.05
    epochs = 2500
    l1_penalty = 0.0001
    l2_penalty = 0.0001

    if not use_dFF:
        X = data['norm_spikes'].copy()
    elif use_dFF:
        X = data['denoised_dFF'].copy()
        for c in range(np.size(X, 0)):
            X[c,:] = X[c,:] - np.nanmin(X[c,:])
            X[c,:] = X[c,:] / np.nanmax(X[c,:])

    theta = data['theta_interp'].copy()
    phi = data['phi_interp'].copy()

    # movement speeds in  deg/sec
    # theta_full = np.rad2deg(data['theta'][data['eyeT_startInd']:data['eyeT_endInd']])
    # phi_full = np.rad2deg(data['phi'][data['eyeT_startInd']:data['eyeT_endInd']])
    # eyeT = data['eyeT'][data['eyeT_startInd']:data['eyeT_endInd']]
    # eyeT = eyeT - eyeT[0]
    # twopT = data['twopT']
    # dt = 1/60
    # dTheta_full = np.diff(theta_full) / dt
    # dPhi_full = np.diff(phi_full) / dt

    # dTheta = fm2p.interpT(dTheta_full, eyeT[:-1], twopT)
    # dPhi = fm2p.interpT(dPhi_full, eyeT[:-1], twopT)

    y = np.vstack([theta, phi])

    if data['ltdk'] and use_light is True:
        # if this is a light/dark recording, only fit on the light periods
        ltdk = data['ltdk_state_vec']
        y = y[:,ltdk]
        X = X[:,ltdk]
    elif data['ltdk'] and use_light is False:
        # fit on the dark periods
        ltdk = data['ltdk_state_vec']
        y = y[:,~ltdk]
        X = X[:,~ltdk]

    model = fm2p.multicell_GLM(
        model='regress',
        learning_rate=learning_rate,
        epochs=epochs,
        l1_penalty=l1_penalty,
        l2_penalty=l2_penalty,
    )

    y = model.fit_apply_transform(y)

    Xp_train, yp_train, Xp_test, yp_test = model.make_split(X, y)

    model.fit(Xp_train, yp_train, verbose=True)

    yp_hat, pupil_mse = model.predict(Xp_test, yp_test)

    pupil_weights = model.get_weights()
    train_inds, test_inds, nan_mask = model.get_train_test_inds()

    # predict on training data
    yp_train_hat, train_mse = model.predict(Xp_train, yp_train)

    results = {
        'weights': pupil_weights,
        'y_hat': model.apply_inverse_transform(yp_hat.T),
        'MSE': pupil_mse,
        'loss_history': model.get_loss_history(),
        'X_train': Xp_train,
        'X_test': Xp_test,
        'y_train': model.apply_inverse_transform(yp_train),
        'y_test': model.apply_inverse_transform(yp_test),
        'y_hat_train': model.apply_inverse_transform(yp_train_hat.T),
        'MSE_train': train_mse,
        'train_inds': train_inds,
        'test_inds': test_inds,
    }

    return results

def run_retina_model(data, use_dFF=False):

    print('  -> Fitting retina-prediction model.')

    learning_rate = 0.05
    epochs = 2500
    l1_penalty = 0
    l2_penalty = 0.001

    if not use_dFF:
        X = data['norm_spikes'].copy()
    elif use_dFF:
        X = data['denoised_dFF'].copy()
        for c in range(np.size(X, 0)):
            X[c,:] = X[c,:] - np.nanmin(X[c,:])
            X[c,:] = X[c,:] / np.nanmax(X[c,:])

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

    if data['ltdk']:
        # if this is a light/dark recording, only fit on the light periods
        ltdk = data['ltdk_state_vec']
        y = y[:,ltdk]
        X = X[:,ltdk]

    model = fm2p.multicell_GLM(
        model='regress',
        learning_rate=learning_rate,
        epochs=epochs,
        l1_penalty=l1_penalty,
        l2_penalty=l2_penalty,
    )

    y = model.fit_apply_transform(y)

    Xp_train, yp_train, Xp_test, yp_test = model.make_split(X, y)

    model.fit(Xp_train, yp_train, verbose=True)

    y_hat, pupil_mse = model.predict(Xp_test, yp_test)

    pupil_weights = model.get_weights()
    train_inds, test_inds, nan_mask = model.get_train_test_inds()
    yp_test_unscaled = y.copy()[:,test_inds]

    # Test on the train data to see if it at least predicts that
    yp_train_hat, train_mse = model.predict(Xp_train, yp_train)

    y_hat_invtran = model.apply_inverse_transform(y_hat.T)

    y_hat_retino = np.rad2deg(np.arctan2(y_hat_invtran[0,:], y_hat_invtran[1,:]))
    y_test_retino = np.rad2deg(np.arctan2(yp_test_unscaled[0,:], yp_test_unscaled[1,:]))

    results = {
        'weights': pupil_weights,
        'y_hat': model.apply_inverse_transform(y_hat.T),
        'MSE': pupil_mse,
        'loss_history': model.get_loss_history(),
        'X_train': Xp_train,
        'X_test': Xp_test,
        'y_train': model.apply_inverse_transform(yp_train),
        'y_test': model.apply_inverse_transform(yp_test),
        'y_test_unscaled': yp_test_unscaled,
        'y_hat_train': model.apply_inverse_transform(yp_train_hat.T),
        'MSE_train': train_mse,
        'y_hat_retino': y_hat_retino,
        'y_test_retino': y_test_retino,
        'theta': data['theta_interp'],
        'phi': data['phi_interp'],
        'train_inds': train_inds,
        'test_inds': test_inds,
        'theta_test': data['theta_interp'][test_inds],
        'phi_test': data['phi_interp'][test_inds],
        'theta_train': data['theta_interp'][train_inds],
        'phi_train': data['phi_interp'][train_inds]
    }

    return results


def run_body_model(data, use_dFF=False):

    print('  -> Fitting body-prediction model.')

    learning_rate = 0.1
    epochs = 2400
    l1_penalty = 0
    l2_penalty = 0.001

    if not use_dFF:
        X = data['norm_spikes'].copy()
    elif use_dFF:
        X = data['denoised_dFF'].copy()
        for c in range(np.size(X, 0)):
            X[c,:] = X[c,:] - np.nanmin(X[c,:])
            X[c,:] = X[c,:] / np.nanmax(X[c,:])

    egocentric = data['egocentric'].copy()
    distance = data['dist_to_pillar'].copy()
    y = np.vstack([
        np.sin(np.deg2rad(egocentric)),
        np.cos(np.deg2rad(egocentric)),
        distance
    ])
    # y[0,:] = fm2p.convfilt(fm2p.nan_interp(y[0,:]), 3)
    # y[1,:] = fm2p.convfilt(fm2p.nan_interp(y[1,:]), 3)

    if data['ltdk']:
        # if this is a light/dark recording, only fit on the light periods
        ltdk = data['ltdk_state_vec']
        y = y[:,ltdk]
        X = X[:,ltdk]

    model = fm2p.multicell_GLM(
        model='regress',
        learning_rate=learning_rate,
        epochs=epochs,
        l1_penalty=l1_penalty,
        l2_penalty=l2_penalty,
    )

    y = model.fit_apply_transform(y)

    Xp_train, yp_train, Xp_test, yp_test = model.make_split(X, y)

    model.fit(Xp_train, yp_train, verbose=True)

    y_hat, pupil_mse = model.predict(Xp_test, yp_test)

    pupil_weights = model.get_weights()
    train_inds, test_inds, nan_mask = model.get_train_test_inds()
    yp_test_unscaled = y.copy()[:,test_inds]

    yp_train_hat, train_mse = model.predict(Xp_train, yp_train)

    y_hat_invtran = model.apply_inverse_transform(y_hat.T)

    y_hat_retino = np.rad2deg(np.arctan2(y_hat_invtran[0,:], y_hat_invtran[1,:]))
    y_test_retino = np.rad2deg(np.arctan2(yp_test_unscaled[0,:], yp_test_unscaled[1,:]))

    results = {
        'weights': pupil_weights,
        'y_hat': model.apply_inverse_transform(y_hat.T),
        'MSE': pupil_mse,
        'loss_history': model.get_loss_history(),
        'X_train': Xp_train,
        'X_test': Xp_test,
        'y_train': model.apply_inverse_transform(yp_train),
        'y_test': model.apply_inverse_transform(yp_test),
        'y_test_unscaled': yp_test_unscaled,
        'y_hat_train': model.apply_inverse_transform(yp_train_hat.T),
        'MSE_train': train_mse,
        'y_hat_ego': y_hat_retino,
        'y_test_ego': y_test_retino,
        'egocentric': egocentric
    }

    return results


def drop_repeat_events(eventT, onset=True, win=0.020):
    duplicates = set([])
    for t in eventT:
        if onset:
            # keep first
            new = eventT[((eventT-t)<win) & ((eventT-t)>0)]
        else:
            # keep last
            new = eventT[((t-eventT)<win) & ((t-eventT)>0)]
        duplicates.update(list(new))
    thinned = np.sort(np.setdiff1d(eventT, np.array(list(duplicates)), assume_unique=True))
    return thinned


def run_movement_model(data, ind, use_dFF=False):
    # given a spike rate, try to predict bool array of eye movement onset times

    learning_rate = 0.1
    epochs = 3000

    if not use_dFF:
        X = data['norm_spikes'].copy()
    elif use_dFF:
        X = data['denoised_dFF'].copy()
        for c in range(np.size(X, 0)):
            X[c,:] = X[c,:] - np.nanmin(X[c,:])
            X[c,:] = X[c,:] / np.nanmax(X[c,:])

    theta = data['theta_interp'].copy()
    phi = data['phi_interp'].copy()

    # movement speeds in  deg/sec
    theta_full = np.rad2deg(data['theta'][data['eyeT_startInd']:data['eyeT_endInd']])
    phi_full = np.rad2deg(data['phi'][data['eyeT_startInd']:data['eyeT_endInd']])
    eyeT = data['eyeT'][data['eyeT_startInd']:data['eyeT_endInd']]
    eyeT = eyeT - eyeT[0]
    twopT = data['twopT']
    dt = 1/60
    dTheta = np.diff(theta_full) / dt
    dPhi = np.diff(phi_full) / dt
    # dTheta = fm2p.interpT(dTheta_full, eyeT[:-1], twopT)
    # dPhi = fm2p.interpT(dPhi_full, eyeT[:-1], twopT)

    all_theta_movements = drop_repeat_events(eyeT[np.where(np.abs(dTheta) > 60)[0]])
    all_phi_movements = drop_repeat_events(eyeT[np.where(np.abs(dPhi) > 60)[0]])
    # right_eye_movements = drop_repeat_events(eyeT[np.where(dTheta > 60)[0]])
    # left_eye_movements = drop_repeat_events(eyeT[np.where(dTheta < -60)[0]])

    all_theta_movement_inds = np.array([fm2p.find_closest_timestamp(twopT, t)[0] for t in all_theta_movements])
    mov0 = np.zeros(len(theta))
    mov0[all_theta_movement_inds] = 1
    mov0 = np.concatenate([(np.diff(mov0)>0), np.array([0])])

    all_phi_movement_inds = np.array([fm2p.find_closest_timestamp(twopT, t)[0] for t in all_phi_movements])
    mov1 = np.zeros(len(phi))
    mov1[all_phi_movement_inds] = 1
    mov1 = np.concatenate([(np.diff(mov1)>0), np.array([0])])

    # right_eye_movement_inds = np.array([fm2p.find_closest_timestamp(twopT, t)[0] for t in right_eye_movements])
    # mov1 = np.zeros(len(theta))
    # mov1[right_eye_movement_inds] = 1
    # mov1 = np.concatenate([np.array([0]), (np.diff(mov1)>0)])

    # left_eye_movement_inds = np.array([fm2p.find_closest_timestamp(twopT, t)[0] for t in left_eye_movements])
    # mov2 = np.zeros(len(theta))
    # mov2[left_eye_movement_inds] = 1
    # mov2 = np.concatenate([np.array([0]), (np.diff(mov2)>0)])

    # y = np.vstack([mov0, mov1])

    if ind == 0:
        y = mov0.copy()[np.newaxis,:]
        print('Found {} horizontal movements.'.format(np.sum(mov0)))
    elif ind == 1:
        y = mov1.copy()[np.newaxis,:]
        print('Found {} vertical movements.'.format(np.sum(mov1)))

    if data['ltdk']:
        # if this is a light/dark recording, only fit on the light periods
        ltdk = data['ltdk_state_vec']
        y = y[:,ltdk]
        X = X[:,ltdk]

    model = fm2p.multicell_GLM(
        model='classify',
        learning_rate=learning_rate,
        epochs=epochs
    )

    y = model.fit_apply_transform(y)

    Xp_train, yp_train, Xp_test, yp_test = model.make_split(X, y)

    model.fit(Xp_train, yp_train, verbose=True)

    y_hat, pupil_cel = model.predict(Xp_test, yp_test)

    pupil_weights = model.get_weights()
    train_inds, test_inds, nan_mask = model.get_train_test_inds()
    yp_test_unscaled = y.copy()[:,test_inds]

    yp_train_hat, train_cel = model.predict(Xp_train, yp_train)

    y_hat_invtran = model.apply_inverse_transform(y_hat.T)

    results = {
        'weights': pupil_weights,
        'y_hat': y_hat_invtran,
        'ce_loss': pupil_cel,
        'loss_history': model.get_loss_history(),
        'X_train': Xp_train,
        'X_test': Xp_test,
        'y_train': model.apply_inverse_transform(yp_train),
        'y_test': model.apply_inverse_transform(yp_test),
        'y_test_unscaled': yp_test_unscaled,
        'y_hat_train': model.apply_inverse_transform(yp_train_hat.T),
        'ce_loss_train': train_cel
    }

    return results


def fit_multicell_GLM(preproc_path=None, use_dFF=False):

    # preproc_path = r'K:\Mini2P\250627_DMM_DMM037_ltdk\fm5\250627_DMM_DMM037_fm_05_preproc.h5'
    models = 'P'
    #  = r'T:\Mini2P\250514_DMM_DMM046_LPaxons\fm1\250514_DMM_DMM046_fm_1_preproc.h5'

    data = fm2p.read_h5(preproc_path)

    all_model_results = {}

    if 'P' in models:
        all_model_results['pupil_light'] = run_pupil_model(data, use_dFF=use_dFF, use_light=True)
        all_model_results['pupil_dark'] = run_pupil_model(data, use_dFF=use_dFF, use_light=False)

    if 'M' in models:
        print('  -> Fitting theta saccade prediction model.')
        all_model_results['theta_mov'] = run_movement_model(data, 0, use_dFF=use_dFF)
        print('  -> Fitting phi saccade prediction model.')
        all_model_results['phi_mov'] = run_movement_model(data, 1, use_dFF=use_dFF)

    if 'R' in models:
        all_model_results['retina'] = run_retina_model(data, use_dFF=use_dFF)

    if 'B' in models:
        all_model_results['body'] = run_body_model(data, use_dFF=use_dFF)
    

    savedir = os.path.split(preproc_path)[0]
    basename = os.path.split(preproc_path)[1][:-11]
    savepath = os.path.join(savedir, '{}_multicell_GLM_results_v9_ltdk.h5'.format(basename))

    fm2p.write_h5(savepath, all_model_results)

    print('\nSaved {}'.format(savepath))


def run_all_GLMs(preproc_path):

    if preproc_path is None:
        preproc_path = fm2p.select_file(
            'Select preprocessing HDF file.',
            filetypes=[('HDF','.h5'),]
        )
        data = fm2p.read_h5(preproc_path)
    elif type(preproc_path) == str:
        data = fm2p.read_h5(preproc_path)
        
    print('  -> Loading data.')

    eyeT = data['eyeT'][data['eyeT_startInd']:data['eyeT_endInd']]
    eyeT = eyeT - eyeT[0]

    if 'dPhi' not in data.keys():
        phi_full = np.rad2deg(data['phi'][data['eyeT_startInd']:data['eyeT_endInd']])
        dPhi  = np.diff(fm2p.interp_short_gaps(phi_full, 5)) / np.diff(eyeT)
        dPhi = np.roll(dPhi, -2)
        data['dPhi'] = dPhi

    if 'dTheta' not in data.keys():
        data['dTheta'] = data['dEye'].copy()

    # interpolate dEye values to twop data
    dTheta = fm2p.interp_short_gaps(data['dTheta'])
    dTheta = fm2p.interpT(dTheta, data['eyeT1'], data['twopT'])
    dPhi = fm2p.interp_short_gaps(data['dPhi'])
    dPhi = fm2p.interpT(dPhi, data['eyeT1'], data['twopT'])

    # spikes = data['norm_spikes'].copy()

    ltdk = data['ltdk_state_vec'].copy()

    gaze = np.cumsum(data['dGaze'].copy())
    dGaze = data['dGaze'].copy()
    gazeT = data['eyeT_trim'] + (np.nanmedian(data['eyeT_trim']) / 2)
    gazeT = gazeT[:-1]
    gaze = fm2p.interpT(gaze, gazeT, data['twopT'])
    dGaze = fm2p.interpT(dGaze, gazeT, data['twopT'])

    # at some point, add in accelerations
    behavior_vars = {
        # head positions
        'yaw': data['head_yaw_deg'].copy(),
        'pitch': data['pitch_twop_interp'].copy(),
        'roll': data['roll_twop_interp'].copy(),
        # gaze
        'gaze': gaze,
        'dGaze': dGaze,
        # eye positions
        'theta': data['theta_interp'].copy(),
        'phi': data['phi_interp'].copy(),
        # eye speeds
        'dTheta': dTheta,
        'dPhi': dPhi,
        # head angular rotation speeds
        'gyro_x': data['gyro_x_twop_interp'].copy(),
        'gyro_y': data['gyro_y_twop_interp'].copy(),
        'gyro_z': data['gyro_z_twop_interp'].copy(),
        # head accelerations
        'acc_x': data['acc_x_twop_interp'].copy(),
        'acc_y': data['acc_y_twop_interp'].copy(),
        'acc_z': data['acc_z_twop_interp'].copy()
    }

    if len(behavior_vars['yaw']) > len(data['norm_spikes']):
        behavior_vars['yaw'] = behavior_vars['yaw'][:-1]

    dict_out = {}

    learning_rate = 0.05
    epochs = 2500
    l1_penalty = 0.0001
    l2_penalty = 0.0001

    for behavior_k, behavior_v in behavior_vars.items():

        print('  -> Fitting spike prediction GLM for {}'.format(behavior_k))

        for use_dFF in range(2):
            for use_lightperiods in range(2):

                if not use_dFF:
                    X = data['norm_spikes'].copy()
                    dFF_key = 'sps'
                elif use_dFF:
                    X = data['denoised_dFF'].copy()
                    for c in range(np.size(X, 0)):
                        X[c,:] = X[c,:] - np.nanmin(X[c,:])
                        X[c,:] = X[c,:] / np.nanmax(X[c,:])
                    dFF_key = 'dff'

                y = behavior_v.copy()[np.newaxis, :]

                ltdk = data['ltdk_state_vec']

                # interp over NaNs for small gaps only. for remaining 
                # would it be better to drop NaN positions without first interp over small gaps?
                # that would mess with temporal window more
                y = fm2p.interp_short_gaps_circ(y.flatten())
                _, nanmask = fm2p.mask_non_nan([y.flatten()])
                y = y.T[np.newaxis, nanmask]
                X = X[:,nanmask]
                ltdk = ltdk[nanmask]

                if data['ltdk'] and use_lightperiods==1:
                    # if this is a light/dark recording, only fit on the light periods
                    y = y[:,ltdk]
                    X = X[:,ltdk]
                    ltdk_key = 'lt'
                elif data['ltdk'] and use_lightperiods==0:
                    # fit on the dark periods
                    y = y[:,~ltdk]
                    X = X[:,~ltdk]
                    ltdk_key = 'dk'


                try:
                    model = fm2p.multicell_GLM(
                        model='regress',
                        learning_rate=learning_rate,
                        epochs=epochs,
                        l1_penalty=l1_penalty,
                        l2_penalty=l2_penalty,
                    )

                    y = model.fit_apply_transform(y)

                    Xp_train, yp_train, Xp_test, yp_test = model.make_split(X, y)

                    model.fit(Xp_train, yp_train, verbose=True)

                    yp_hat, pupil_mse = model.predict(Xp_test, yp_test)

                    pupil_weights = model.get_weights()
                    train_inds, test_inds, nan_mask = model.get_train_test_inds()

                    # predict on training data
                    yp_train_hat, train_mse = model.predict(Xp_train, yp_train)

                    single_model_results = {
                        'weights': pupil_weights,
                        'y_hat': model.apply_inverse_transform(yp_hat.T),
                        'MSE': pupil_mse,
                        'loss_history': model.get_loss_history(),
                        'X_train': Xp_train,
                        'X_test': Xp_test,
                        'y_train': model.apply_inverse_transform(yp_train),
                        'y_test': model.apply_inverse_transform(yp_test),
                        'y_hat_train': model.apply_inverse_transform(yp_train_hat.T),
                        'MSE_train': train_mse,
                        'train_inds': train_inds,
                        'test_inds': test_inds
                    }

                except:
                    single_model_results = {}

                model_key = '{}_{}_{}'.format(behavior_k, dFF_key, ltdk_key)
                dict_out[model_key] = single_model_results

    savepath = os.path.join(os.path.split(preproc_path)[0], 'GLM4_all_models_results.h5')
    fm2p.write_h5(savepath, dict_out)

    return dict_out

if __name__ == '__main__':

    fit_multicell_GLM()

