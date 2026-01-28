

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
import fm2p



class singlecell_GLM:
    def __init__(
            self,
            learning_rate=1e-5, # was 1e-5
            epochs=4000,
            l1_penalty=0.1,
            l2_penalty=0.5, # was 1.0
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
        return np.mean((y - y_hat)**2)

    def _loss_old(self, y, y_hat, w):
        mse = np.mean((y - y_hat) ** 2)
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
                
            dev = np.mean(term1 - term2)
            return dev + l1 + l2

        else:
            # in this case, z == y_hat
            y_hat = y_hat_or_z
            mse = np.mean((y - y_hat) ** 2)
            return mse + l1 + l2
    
    def _fit_gradient_descent(self, X_norm, y_norm, verbose=False):

        n_frames, n_features = X_norm.shape
        
        rng = np.random.default_rng(42)
        weights = rng.normal(0, 0.01, size=n_features + 1)
        
        X_bias = np.hstack([np.ones((n_frames, 1)), X_norm])
        
        loss_history = np.zeros(self.epochs)
        
        for epoch in range(self.epochs):
            # linear predictor
            z = X_bias @ weights

            z = np.clip(z, -20, 20)
            y_hat = np.exp(z)

            loss_history[epoch] = self._loss(y_norm, z, weights, input_is_z=True)
            
            error = y_hat - y_norm
            gradient = (X_bias.T @ error) / n_frames
            
            gradient[1:] += self.l2_penalty * 2 * weights[1:] 
            gradient[1:] += self.l1_penalty * np.sign(weights[1:])
            
            weights -= self.learning_rate * gradient
            
        final_pred = X_bias @ weights
        final_rmse_norm = np.sqrt(np.mean((y_norm - final_pred)**2))
        
        return weights, loss_history, final_rmse_norm

    def fit(self, behavior, spikes, verbose=False):

        if behavior.shape[0] < behavior.shape[1]: 
            X = behavior.T
        else:
            X = behavior.copy()

        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.X_std[self.X_std == 0] = 1.0
        X_norm = (X - self.X_mean) / self.X_std

        if spikes.ndim == 1:
            spikes = spikes[:, np.newaxis]
        elif spikes.shape[0] < spikes.shape[1]:
            spikes = spikes.T
            
        self.n_cells = spikes.shape[1]
        n_features = X.shape[1]
        
        self.y_mean = np.mean(spikes, axis=0)
        self.y_std = np.std(spikes, axis=0)
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

        # need to use exponentional and not just the linear predictor
        z = X_bias @ self.weights.T
        y_hat = np.exp(z)
        
        return y_hat

    def make_split(self, X, y, test_size=0.25):

        if X.shape[0] < X.shape[1]: 
            X = X.T 
        if y.ndim == 1: 
            y = y[:, np.newaxis]
        elif y.shape[0] < y.shape[1]: 
            y = y.T

        mask = ~np.any(np.isnan(X), axis=1) & ~np.any(np.isnan(y), axis=1)
        X = X[mask]
        y = y[mask]

        ss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, test_idx = next(ss.split(X))
        
        return (X[train_idx], y[train_idx],
                X[test_idx], y[test_idx])
    
    def get_model_summary(self):
        return {
            'n_cells': self.n_cells,
            'weights_shape': self.weights.shape if self.weights is not None else None,
            'mean_rmse_train': np.mean(self.rmse) if self.rmse is not None else None
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
        data['theta_interp'],
        data['phi_interp'],
        fm2p.interpT(data['dTheta'], data['eyeT1'], twopT),
        fm2p.interpT(data['dPhi'], data['eyeT1'], twopT)
    ])
    spikes = data['norm_spikes']

    n_lags = 15
    lagged_features = []

    for lag in range(n_lags):
        shifted = np.roll(base_behavior, shift=lag, axis=1)
        shifted[:, :lag] = 0 
        lagged_features.append(shifted)
    behavior_with_lags = np.vstack(lagged_features)

    scGLM = singlecell_GLM()
    
    X_train, y_train, X_test, y_test = scGLM.make_split(behavior_with_lags, spikes, test_size=0.25)
    
    print("Data Split Shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_test:  {y_test.shape}")
    
    print("\nFitting model...")
    scGLM.fit(X_train, y_train, verbose=False)
    
    y_pred = scGLM.predict(X_test)
    
    mse = np.mean((y_test - y_pred)**2, axis=0)
    
    rmse = np.sqrt(mse)
    
    ss_res = np.sum((y_test - y_pred)**2, axis=0)
    ss_tot = np.sum((y_test - np.mean(y_test, axis=0))**2, axis=0)
    ss_tot[ss_tot == 0] = 1.0
    r2 = 1 - (ss_res / ss_tot)
    
    print(f'\nTest Metrics (Average across {len(rmse)} cells):')
    print(f'  MSE:  {np.mean(mse):.4f}')
    print(f'  RMSE: {np.mean(rmse):.4f}')
    print(f'  R2:   {np.mean(r2):.4f}')

    best_cell = np.argmax(r2)
    print(f'\nBest Cell (Index {best_cell}):')
    print(f'  RMSE: {rmse[best_cell]:.4f}')
    print(f'  R2:   {r2[best_cell]:.4f}')

    plt.figure(figsize=(12, 4))
    plt.plot(y_test[:500, best_cell], 'k-', label='True Spikes', alpha=0.6)
    plt.plot(y_pred[:500, best_cell], 'r--', label='Predicted', alpha=0.8)
    plt.title(f"Cell {best_cell} Test Set Prediction (First 100 frames)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()