if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
    __package__ = 'fm2p.utils'

import numpy as np
from .files import read_h5
from .helper import interp_short_gaps
from .time import interpT
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm


class singlecell_GLM:
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
        self.loss_history = None
        self.rmse = None
        self.n_cells = None
        self.scalers = None

        # for calculating z score, hold onto the mean and std so it can be applied to novel data
        self.X_means = None
        self.X_stds = None
        self.y_means = None
        self.y_stds = None


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


    def _loss(self, y, y_hat, w):
        """ Loss function using MSE with L1 and L2 regularization.
        
        Uses mean squared error for stability with identity link function.
        If the penalty parameters of the class are set to 0, no penalty is applied.
        """
        # Mean squared error
        mse = np.mean((y - y_hat) ** 2)
        # L1 and L2 penalty terms (don't regularize bias)
        l1 = self.l1_penalty * np.sum(np.abs(w[1:]))
        l2 = self.l2_penalty * np.sum(w[1:] ** 2)

        return mse + l1 + l2
    
    
    def _fit_single(self, X, y, init=0.01, verbose=False):
        """Fit weights for a single cell using gradient descent.
        
        Parameters
        ----------
        X : ndarray
            Behavior variables of shape (nBehaviors, nFrames)
        y : ndarray
            Spike rate for single cell of shape (nFrames,)
        init : float
            Initial weight value
        verbose : bool
            Print progress
            
        Returns
        -------
        weights : ndarray
            Fitted weights of shape (nBehaviors+1,) including bias
        loss_history : ndarray
            Loss at each epoch
        rmse : float
            Final root mean squared error
        """
        # Ensure proper shapes: X should be (nBehaviors, nFrames), y should be (nFrames,)
        if X.ndim == 1:
            X = X[np.newaxis, :]
        if y.ndim != 1:
            y = y.ravel()
        
        n_behaviors = X.shape[0]
        n_frames = X.shape[1]
        
        assert y.shape[0] == n_frames, f"y has {y.shape[0]} frames but X has {n_frames}"
        
        # Normalize input features to prevent weight explosion
        X_mean = np.mean(X, axis=1, keepdims=True)
        X_std = np.std(X, axis=1, keepdims=True)
        X_std[X_std == 0] = 1.0  # Prevent division by zero
        X_norm = (X - X_mean) / X_std
        # Input X and y are assumed to be already normalized by fit()
        
        # Normalize target
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_std = max(y_std, 1e-8)  # Prevent division by zero
        y_norm = (y - y_mean) / y_std
        
        # Initialize weights: one per behavior variable + one bias term
        weights = np.ones(n_behaviors + 1) * init
        
        # Add bias term as first column: shape becomes (nFrames, nBehaviors+1)
        X_bias = np.c_[np.ones(n_frames), X_norm.T]
        X_bias = np.c_[np.ones(n_frames), X.T]
        
        loss_history = np.zeros(self.epochs) * np.nan
        
        for epoch in range(self.epochs):
            # Forward pass: (nFrames, nBehaviors+1) @ (nBehaviors+1,) -> (nFrames,)
            y_hat = np.dot(X_bias, weights)
            
            # Calculate loss
            lossval = self._loss(y_norm, y_hat, weights)
            lossval = self._loss(y, y_hat, weights)
            loss_history[epoch] = lossval
            
            # Gradient calculation
            y_diff = y_hat - y_norm  # Shape: (nFrames,)
            y_diff = y_hat - y  # Shape: (nFrames,)
            gradient = np.dot(X_bias.T, y_diff) / n_frames  # Shape: (nBehaviors+1,)
            
            # Apply L2 regularization (ridge) - don't regularize bias term
            gradient[1:] += self.l2_penalty * 2 * weights[1:]
            
            # Apply L1 regularization (lasso) - subgradient method
            gradient[1:] += self.l1_penalty * np.sign(weights[1:])
            
            # Clip gradient to prevent explosion
            gradient = np.clip(gradient, -10.0, 10.0)
            
            # Update weights
            weights -= self.learning_rate * gradient
            
            # Clip weights to prevent divergence
            weights = np.clip(weights, -100.0, 100.0)
            
            # Compute MSE for monitoring (using normalized data)
            mse = self._mse(y_norm, y_hat)
            mse = self._mse(y, y_hat)
            
            if verbose and (epoch == 0):
                print(f'Initial pass:  loss={lossval:.4f}  MSE={mse:.4f}')
            elif verbose and (epoch % 100 == 0):
                print(f'\rEpoch {epoch:4d}:  loss={lossval:.4f}  MSE={mse:.4f}', end='', flush=True)
        
        if verbose:
            print('\nTraining complete')
        
        # Final RMSE (in normalized space)
        final_y_hat = np.dot(X_bias, weights)
        rmse_norm = np.sqrt(self._mse(y_norm, final_y_hat))
        rmse_norm = np.sqrt(self._mse(y, final_y_hat))
        
        # Denormalize RMSE back to original scale
        rmse = rmse_norm * y_std
        rmse = rmse_norm # Caller handles scaling
        
        return weights, loss_history, rmse


    def fit(self, behavior, spikes, verbose=False):
        """Fit GLM weights to predict spike rate from behavior variables.
        
        Parameters
        ----------
        behavior : ndarray
            Behavior variables of shape (nBehaviors, nFrames) - typically (4, nFrames)
            for theta, phi, dTheta, dPhi
        spikes : ndarray
            Spike rate data. Can be:
            - 1D array of shape (nFrames,) for single cell
            - 2D array of shape (nCells, nFrames) for multiple cells
        verbose : bool
            Print training progress
        """
        # Ensure behavior has correct shape
        if behavior.ndim == 1:
            behavior = behavior[np.newaxis, :]
        
        # Handle both single cell (1D) and multiple cells (2D) input
        if spikes.ndim == 1:
            # Single cell case
            X = behavior.copy()
            y = spikes.copy()
            weights, loss_history, rmse = self._fit_single(X, y, verbose=verbose)
            
            # Store as 1D for single cell
            self.weights = weights
            self.loss_history = loss_history
            self.rmse = rmse
            self.n_cells = 1
            
        else:
            # Multiple cells case
            n_behaviors = behavior.shape[0]
            n_cells = spikes.shape[0]
            
            # Storage for all cells
            all_weights = np.zeros([n_cells, n_behaviors + 1]) * np.nan
            all_loss_histories = np.zeros([n_cells, self.epochs]) * np.nan
            all_rmse = np.zeros(n_cells) * np.nan
            
            X = behavior.copy()
            # Calculate Y statistics per cell
            self.y_means = np.mean(spikes, axis=1)
            self.y_stds = np.std(spikes, axis=1)
            self.y_stds[self.y_stds == 0] = 1e-8
            
            # Fit each cell independently
            for c in tqdm(range(n_cells), desc='Fitting cells'):
                y = spikes[c, :].copy()
                w, lh, rmse = self._fit_single(X, y, verbose=False)
                y = (spikes[c, :] - self.y_means[c]) / self.y_stds[c]
                w, lh, rmse = self._fit_single(X_norm, y, verbose=False)
                
                all_weights[c, :] = w
                all_loss_histories[c, :] = lh
                all_rmse[c] = rmse
            
            self.weights = all_weights
            self.loss_history = all_loss_histories
            self.rmse = all_rmse
            self.n_cells = n_cells


    def _predict(self, X):
        """Internal prediction method.
        
        Parameters
        ----------
        X : ndarray
            Behavior variables of shape (nBehaviors, nFrames)
            
        Returns
        -------
        y_hat : ndarray
            Predicted spike rates
            - shape (nFrames,) if single cell was fitted
            - shape (nFrames, nCells) if multiple cells were fitted
        """
        # Ensure proper shape
        if X.ndim == 1:
            X = X[np.newaxis, :]
        
        # Normalize X using stored stats
        if self.X_means is not None:
            X = (X - self.X_means) / self.X_stds

        n_frames = X.shape[1]
        
        # Add bias column: shape (nFrames, nBehaviors+1)
        X_bias = np.c_[np.ones(n_frames), X.T]
        
        if self.weights.ndim == 1:
            # Single cell: weights is (nBehaviors+1,)
            # Result shape: (nFrames,)
            y_hat = np.dot(X_bias, self.weights)
        else:
            # Multiple cells: weights is (nCells, nBehaviors+1)
            # Result shape: (nFrames, nCells)
            y_hat = np.dot(X_bias, self.weights.T)
            
        # Denormalize predictions
        if self.y_means is not None:
            if y_hat.ndim == 1:
                y_hat = y_hat * self.y_stds + self.y_means
            else:
                y_hat = y_hat * self.y_stds[np.newaxis, :] + self.y_means[np.newaxis, :]
        
        return y_hat
    

    def make_split(self, X, y, nanfilt=True, test_size=0.25):
        # NaN at any position in X or y will cause issues. Mask out NaNs, which
        # cannot appear in spike data (X), but do show up in the behavior data (y)
        # because of gaps in tracking.

        # if y is 1D
        if len(np.shape(y)) != 2:
            y = y[:,np.newaxis]

        if len(np.shape(X)) != 2:
            X = X[:,np.newaxis]


        if nanfilt:
            # Check for NaNs in both X and y along the frame dimension
            mask_y = np.sum(np.isnan(y), axis=0) == 0
            mask_X = np.sum(np.isnan(X), axis=0) == 0
            mask1d = mask_y & mask_X  # Only keep frames with no NaNs in either
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

        return X_train, y_train, X_test, y_test
    

    def get_train_test_inds(self):
        return self.train_inds, self.test_inds, self.nan_mask


    def apply_transform(self, A):
        """Apply fitted scalers to data.
        
        Parameters
        ----------
        A : ndarray
            Data of shape (nFeatures, nSamples) or (nSamples,)
            
        Returns
        -------
        A_scaled : ndarray
            Scaled data with same shape as input
            
        Raises
        ------
        AssertionError
            If scalers haven't been fitted or feature count doesn't match
        """
        if A.ndim == 1:
            A = A[np.newaxis, :]
        
        n_feats = A.shape[0]
        
        assert self.scalers is not None, "Scalers not fitted. Call fit_apply_transform first."
        assert n_feats == len(self.scalers), f"Expected {len(self.scalers)} features, got {n_feats}"
        
        A_scaled = np.zeros_like(A, dtype=float)
        
        for feat in range(n_feats):
            A_scaled[feat, :] = self.scalers[feat].transform(A[feat, :].reshape(-1, 1)).flatten()
        
        return A_scaled


    def fit_apply_transform(self, A):
        """Fit robust scalers to data and apply them.
        
        Trains one scaler per feature (row) and applies the transformation.
        Useful for normalizing behavior variables independently.
        
        Parameters
        ----------
        A : ndarray
            Data of shape (nFeatures, nSamples) or (nSamples,)
            
        Returns
        -------
        A_scaled : ndarray
            Scaled data with same shape as input
        """
        if A.ndim == 1:
            A = A[np.newaxis, :]
        
        n_feats = A.shape[0]
        self.scalers = []
        
        # Fit a RobustScaler for each feature
        for feat in range(n_feats):
            scaler = RobustScaler()
            scaler.fit(A[feat, :].reshape(-1, 1))
            self.scalers.append(scaler)
        
        # Apply scaling
        A_scaled = self.apply_transform(A)
        
        return A_scaled


    def apply_inverse_transform(self, A):

        n_feats = np.size(A,0)

        assert n_feats == len(self.scalers)

        A_invtran = np.zeros_like(A) * np.nan

        for feat in range(n_feats):
            A_invtran[feat,:] = self.scalers[feat].inverse_transform(A[feat,:][:,np.newaxis]).flatten()

        return A_invtran
    

    def score_explained_variance(self, y, y_hat):
        """Calculate explained variance (coefficient of determination, R^2).
        
        This is similar to R^2 but treats offsets as errors. Max value is 1.0.
        Can multiply by 100 to get a percentage.
        
        Parameters
        ----------
        y : ndarray
            True values of shape (nFrames,)
        y_hat : ndarray
            Predicted values of shape (nFrames,)
            
        Returns
        -------
        r2 : float
            Explained variance, range (-inf, 1.0]
        """
        # Ensure 1D
        y = y.ravel()
        y_hat = y_hat.ravel()
        
        # Residual sum of squares
        ss_res = np.sum((y - y_hat) ** 2)
        # Total sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        # Avoid division by zero
        if ss_tot < 1e-10:
            return 0.0 if ss_res < 1e-10 else -np.inf
        
        return 1.0 - (ss_res / ss_tot)


    def predict(self, X, y):
        """Make predictions and calculate error metrics.
        
        Parameters
        ----------
        X : ndarray
            Behavior variables of shape (nBehaviors, nFrames)
        y : ndarray
            True spike rates of shape (nFrames,) for single cell
            or (nCells, nFrames) for multiple cells
            
        Returns
        -------
        y_hat : ndarray
            Predicted spike rates, same shape as predictions from _predict
        metrics : dict
            Dictionary containing:
            - 'mse': mean squared error
            - 'rmse': root mean squared error
            - 'r2': explained variance (R^2)
        """
        # Ensure y is properly shaped
        if y.ndim == 1:
            y = y[np.newaxis, :]
        
        # Get predictions
        y_hat = self._predict(X)
        
        # Handle single vs multiple cells
        if y_hat.ndim == 1:
            # Single cell prediction: y_hat is (nFrames,)
            y_true = y.ravel()
            mse = self._mse(y_true, y_hat)
            rmse = np.sqrt(mse)
            r2 = self.score_explained_variance(y_true, y_hat)
        else:
            # Multiple cells prediction: y_hat is (nFrames, nCells)
            # Calculate metrics per cell
            n_cells = y_hat.shape[1]
            mse = np.zeros(n_cells)
            rmse = np.zeros(n_cells)
            r2 = np.zeros(n_cells)
            
            for c in range(n_cells):
                mse[c] = self._mse(y[c, :], y_hat[:, c])
                rmse[c] = np.sqrt(mse[c])
                r2[c] = self.score_explained_variance(y[c, :], y_hat[:, c])
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        return y_hat, metrics


    def get_weights(self):
        """Get fitted weights.
        
        Returns
        -------
        weights : ndarray
            Fitted weights of shape (nBehaviors+1,) for single cell
            or (nCells, nBehaviors+1) for multiple cells
        """
        return self.weights
    

    def get_loss_history(self):
        """Get training loss history.
        
        Returns
        -------
        loss_history : ndarray
            Loss at each epoch
        """
        return self.loss_history
    

    def get_error(self):
        """Get RMSE for fitted model.
        
        Returns
        -------
        rmse : float or ndarray
            Root mean squared error (scalar for single cell, array for multiple)
        """
        return self.rmse
    

    def get_model_summary(self):
        """Get a summary of the fitted model.
        
        Returns
        -------
        summary : dict
            Dictionary with model information
        """
        if self.weights is None:
            return {'status': 'Model not yet fitted'}
        
        summary = {
            'n_cells': self.n_cells,
            'n_behaviors': self.weights.shape[0] - 1 if self.weights.ndim == 1 else self.weights.shape[1] - 1,
            'rmse': self.rmse,
            'weights_shape': self.weights.shape,
        }
        
        return summary
    



def main():
    """Example usage of singlecell_GLM class."""
    
    data = read_h5('/home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings/cohort02_recordings/251031_DMM_DMM056_pos14/fm1/251031_DMM_DMM056_fm1_01_preproc.h5')

    eyeT = data['eyeT'][data['eyeT_startInd']:data['eyeT_endInd']]
    eyeT = eyeT - eyeT[0]

    if 'dPhi' not in data.keys():
        phi_full = np.rad2deg(data['phi'][data['eyeT_startInd']:data['eyeT_endInd']])
        dPhi  = np.diff(interp_short_gaps(phi_full, 5)) / np.diff(eyeT)
        dPhi = np.roll(dPhi, -2)
        data['dPhi'] = dPhi

    if 'dTheta' not in data.keys():
        t = eyeT.copy()[:-1]
        data['eyeT1'] = t + (np.diff(eyeT) / 2)

        theta_full = np.rad2deg(data['theta'][data['eyeT_startInd']:data['eyeT_endInd']])
        dEye  = np.diff(interp_short_gaps(theta_full, 5)) / np.diff(eyeT)
        data['dTheta'] = np.roll(dEye, -2)  # static offset correction

    twopT = data['twopT']
    behavior = np.vstack([
        data['theta_interp'],
        data['phi_interp'],
        interpT(data['dTheta'], data['eyeT1'], twopT),
        interpT(data['dPhi'], data['eyeT1'], twopT)
    ])
    spikes = data['norm_spikes']

    # Create GLM model
    scGLM = singlecell_GLM(learning_rate=0.001, epochs=5000, l1_penalty=0.01, l2_penalty=0.01)
    
    # Split data into train/test
    X_train, y_train, X_test, y_test = scGLM.make_split(behavior, spikes, test_size=0.25)
    
    print(f'Training shapes: X_train={X_train.shape}, y_train={y_train.shape}')
    print(f'Test shapes: X_test={X_test.shape}, y_test={y_test.shape}')
    
    # Fit model on training data
    scGLM.fit(X_train, y_train, verbose=False)
    
    # Evaluate on test data
    y_pred, metrics = scGLM.predict(X_test, y_test)
    
    print(f'\nModel Summary:')
    print(scGLM.get_model_summary())
    print(f'\nTest Metrics:')
    print(f'  MSE: {metrics["mse"]}')
    print(f'  RMSE: {metrics["rmse"]}')
    print(f'  R^2: {metrics["r2"]}')


if __name__ == '__main__':
    main()