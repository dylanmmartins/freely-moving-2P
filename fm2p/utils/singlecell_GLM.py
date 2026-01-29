

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
import fm2p

class singlecell_GLM:
    def __init__(
            self,
            learning_rate=1e-3, # was 1e-5
            epochs=6000,
            l1_penalty=0.0,
            l2_penalty=0.0, # was 1.0
            distribution='poisson'
        ):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.distribution = distribution

        self.weights = None
        self.loss_history = None
        self.rmse = None
        self.n_cells = None
        
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None

    def _mse(self, y, y_hat):
        return np.nanmean((y - y_hat)**2)

    def _loss_old(self, y, y_hat, w):
        mse = np.nanmean((y - y_hat) ** 2)
        l1 = self.l1_penalty * np.sum(np.abs(w[1:]))
        l2 = self.l2_penalty * np.sum(w[1:] ** 2)
        return mse + l1 + l2
    
    def _loss(self, y, y_hat_or_z, w, input_is_z=False):
        # more efficient to pass the linear predictor 'z' (X@w) than y_hat to avoid log(exp(z))

        l1 = self.l1_penalty * np.sum(np.abs(w[1:]))
        l2 = self.l2_penalty * np.sum(w[1:] ** 2)

        if self.distribution == 'poisson':
            # poisson negnegative log likelihood
            # J = sum(y_hat - y * ln(y_hat))
            # since y_hat = exp(z), ln(y_hat) = z
            # J = sum(exp(z) - y * z)
            
            if input_is_z:
                z = y_hat_or_z
                term1 = np.exp(z)
                term2 = y * z
            else:
                # less numericaly stable
                term1 = y_hat_or_z
                term2 = y * np.log(y_hat_or_z + 1e-10)
                
            dev = np.nanmean(term1 - term2)
            return dev + l1 + l2

        else:
            # in this case, z == y_hat
            y_hat = y_hat_or_z
            mse = np.nanmean((y - y_hat) ** 2)
            return mse + l1 + l2
    
    def _fit_gradient_descent(self, X_norm, y_norm, verbose=False):

        n_frames, n_features = X_norm.shape
        
        rng = np.random.default_rng(42)

        weights = rng.normal(0, 0.001, size=n_features + 1)
        # init as mean firing rate
        avg_rate = np.nanmean(y_norm) + 1e-10
        weights[0] = np.log(avg_rate)
        
        X_bias = np.hstack([np.ones((n_frames, 1)), X_norm])
        
        loss_history = np.zeros(self.epochs)
        
        for epoch in range(self.epochs):
            # linear predictor
            z = X_bias @ weights

            if np.max(np.abs(z)) > 6:
                print('Must clip z. abs of val was {}'.format(np.max(np.abs(z))))
            z = np.clip(z, -20, 20) # started as 20
            y_hat = np.exp(z)

            loss_history[epoch] = self._loss(y_norm, z, weights, input_is_z=True)
            
            error = y_hat - y_norm
            gradient = (X_bias.T @ error) / n_frames
            
            gradient[1:] += self.l2_penalty * 2 * weights[1:] 
            gradient[1:] += self.l1_penalty * np.sign(weights[1:])
            
            weights -= self.learning_rate * gradient
            
        final_pred = X_bias @ weights
        final_rmse_norm = np.sqrt(np.nanmean((y_norm - final_pred)**2))
        
        return weights, loss_history, final_rmse_norm

    def fit(self, behavior, spikes, verbose=False):

        if behavior.shape[0] < behavior.shape[1]:
            X = behavior.T
        else:
            X = behavior.copy()

        if spikes.ndim == 1:
            spikes = spikes[:, np.newaxis]
        elif spikes.shape[0] < spikes.shape[1]:
            spikes = spikes.T

        self.X_mean = np.nanmean(X, axis=0)
        self.X_std = np.nanstd(X, axis=0)
        self.X_std[self.X_std == 0] = 1.0

        # mask1d = np.sum(~np.isnan(X), 1) == np.size(X, 1)
        # spikes = spikes[mask1d,:]
        # X = X[mask1d, :]

        X_norm = (X - self.X_mean) / self.X_std

        self.n_cells = spikes.shape[1]
        n_features = X.shape[1]
        
        self.y_mean = np.nanmean(spikes, axis=0)
        self.y_std = np.nanstd(spikes, axis=0)
        self.y_std[self.y_std == 0] = 1.0

        # y_mean_2d = self.y_mean.reshape(1, -1)
        # y_std_2d = self.y_std.reshape(1, -1)

        # Y_norm = (spikes - y_mean_2d) / y_std_2d

        self.weights = np.zeros((self.n_cells, n_features + 1))
        self.loss_history = np.zeros((self.n_cells, self.epochs))
        self.rmse = np.zeros(self.n_cells)

        iterator = range(self.n_cells)
        if verbose:
            iterator = tqdm(iterator, desc="Fitting Cells")

        # display(np.sum(np.isnan(X_norm), np.sum(np.isnan(spikes))))

        for c in iterator:

            y_cell_norm = spikes[:, c]
            
            w, h, r = self._fit_gradient_descent(X_norm, y_cell_norm, verbose=False)
            
            self.weights[c, :] = w
            self.loss_history[c, :] = h
            self.rmse[c] = r * self.y_std[c]

    def predict(self, X):

        if X.shape[0] == len(self.X_mean): 
            X = X.T

        X_norm = (X - self.X_mean) / self.X_std

        n_frames = X.shape[0]
        X_bias = np.hstack([np.ones((n_frames, 1)), X_norm])

        # need to use exp and not just the linear predictor
        z = X_bias @ self.weights.T
        y_hat = np.exp(z)
        
        return y_hat
    
    def get_model_summary(self):
        return {
            'n_cells': self.n_cells,
            'weights_shape': self.weights.shape if self.weights is not None else None,
            'mean_rmse_train': np.nanmean(self.rmse) if self.rmse is not None else None
        }


def main():

    fpath = '/home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings/cohort02_recordings/251031_DMM_DMM056_pos14/fm1/251031_DMM_DMM056_fm1_01_preproc.h5'
    data = fm2p.read_h5(fpath)

    eyeT = data['eyeT'][data['eyeT_startInd']:data['eyeT_endInd']]
    eyeT = eyeT - eyeT[0]

    if 'dPhi' not in data.keys():
        phi_full = np.rad2deg(data['phi'][data['eyeT_startInd']:data['eyeT_endInd']])
        dPhi  = np.diff(fm2p.interp_short_gaps(phi_full, 5)) / np.diff(eyeT)
        dPhi = np.roll(dPhi, -2)
        data['dPhi'] = dPhi

    if 'dTheta' not in data.keys():
        t = eyeT.copy()[:-1]
        data['eyeT1'] = t + (np.diff(eyeT) / 2)

        theta_full = np.rad2deg(data['theta'][data['eyeT_startInd']:data['eyeT_endInd']])
        dEye  = np.diff(fm2p.interp_short_gaps(theta_full, 5)) / np.diff(eyeT)
        data['dTheta'] = np.roll(dEye, -2) 

    twopT = data['twopT']

    base_behavior = np.vstack([
        fm2p.interp_short_gaps(data['theta_interp']),
        fm2p.interp_short_gaps(data['phi_interp']),
        fm2p.interp_short_gaps(fm2p.interpT(data['dTheta'], data['eyeT1'], twopT)),
        fm2p.interp_short_gaps(fm2p.interpT(data['dPhi'], data['eyeT1'], twopT))
    ])

    # only light condition
    base_behavior = base_behavior[:, data['ltdk_state_vec']]

    spikes = data['norm_spikes'][:, data['ltdk_state_vec']]

    mask1d = np.sum(~np.isnan(base_behavior), 0) == np.size(base_behavior, 0)
    spikes = spikes[:,mask1d]
    base_behavior = base_behavior[:,mask1d]

    n_lags = 16 # was 10 (got nan pred), then 0 (poor pred but no nans)
    # 16 lags would be 2 seconds
    lagged_features = []

    for lag in range(n_lags):

        shifted = np.roll(base_behavior, shift=lag, axis=1)
        shifted[:, :lag] = 0 
        lagged_features.append(shifted)

    if n_lags > 0:
        behavior_with_lags = np.vstack(lagged_features)
    else:
        behavior_with_lags = base_behavior

    model = singlecell_GLM()

    n_splits = 2 # was 5

    tscv = TimeSeriesSplit(n_splits=n_splits)

    n_cells = np.size(spikes, 0)

    cv_rmse = np.zeros([
        n_cells,
        n_splits
    ])

    if n_lags > 0:
        weights = np.zeros([
            n_cells,
            np.size(base_behavior, 0) * n_lags + 1,  # add one for the bias
            n_splits
        ])
    else:
        weights = np.zeros([
            n_cells,
            np.size(base_behavior, 0) + 1,
            n_splits
        ])

    for fold_num, (train_index, test_index) in tqdm(enumerate(tscv.split(base_behavior.T))):
        
        X_train, X_test = behavior_with_lags[:,train_index], behavior_with_lags[:,test_index]
        y_train, y_test = spikes[:,train_index], spikes[:,test_index]
        
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred = y_pred.T

        for c in range(n_cells):

            fold_rmse = np.sqrt(np.nanmean((y_test[c] - y_pred[c])**2))

            cv_rmse[c,fold_num] = fold_rmse

        weights[:,:,fold_num] = model.weights

    rmse_across_folds = np.nanmean(cv_rmse, axis=0)


if __name__ == '__main__':
    main()
