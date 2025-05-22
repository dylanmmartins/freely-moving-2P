# -*- coding: utf-8 -*-
"""
Linear-nonlinear Poisson model.

Functions
---------
linear_nonlinear_poisson_model(param, X, Y, modelType, param_counts)
    Linear-nonlinear Poisson model.
fit_LNLP_model(A_input, dt, spiketrain, filter, modelType, param_counts, numFolds=10, ret_for_MP=True)
    Fit a linear-nonlinear-poisson model.
fit_all_LNLP_models(data_vars, data_bins, spikes, savedir):
    Fit all neurons to LNLP model for all model combinations.

Author: DMM, 2024
"""


import os
from tqdm import tqdm
import numpy as np
from scipy import sparse
from datetime import datetime
import multiprocessing
from itertools import combinations
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import factorial
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import fm2p


def linear_nonlinear_poisson_model(param, X, Y, modelType, param_counts):
    """ Linear-nonlinear poisson model.

    Parameters
    ----------
    param : array_like
        Array of parameters.
    X : array_like
        Behavioral variables one-hot encoded.
    Y : array_like
        Spike counts for a single cell.
    modelType : str
        Model type string. Must contain only the character P, R, E, and/or D.
    param_counts : array_like
        Number of parameters for each of the four model types.

    Returns
    -------
    f : float
        Objective function value.
    df : array_like
        Gradient of the objective function.
    hessian : array_like
        Hessian of the objective function.
    """

    # Compute the firing rate
    u = X @ param
    rate = np.exp(u)

    # Roughness regularizer weight
    b_P = 5e1
    b_R = 5e1
    b_E = 5e1
    # b_D = 5e1

    # Start computing the Hessian
    rX = np.multiply(rate[:, np.newaxis], X)
    hessian_glm = rX.T @ X

    # Initialize parameter-relevant variables
    J_P = 0
    J_P_g = np.array([])
    J_P_h = np.array([])
    J_R = 0
    J_R_g = np.array([])
    J_R_h = np.array([])
    J_E = 0
    J_E_g = np.array([])
    J_E_h = np.array([])
    # J_D = 0
    # J_D_g = np.array([])
    # J_D_h = np.array([])

    # Find the parameters
    numP, numR, numE = param_counts

    # param_P, param_R, param_E, param_D = fm2p.find_param(param, modelType, numP, numR, numE, numD)
    param_P, param_R, param_E = fm2p.find_param(param, modelType, numP, numR, numE)

    gradstack = []
    hessstack = []

    # Compute the contribution for f, df, and the hessian
    if param_P.size != 0:
        J_P, J_P_g, J_P_h = fm2p.rough_penalty(param_P, b_P)
        gradstack.extend(J_P_g.flatten())
        hessstack.append(J_P_h)

    if param_R.size != 0:
        J_R, J_R_g, J_R_h = fm2p.rough_penalty(param_R, b_R, circ=False)
        gradstack.extend(J_R_g.flatten())
        hessstack.append(J_R_h)

    if param_E.size != 0:
        J_E, J_E_g, J_E_h = fm2p.rough_penalty(param_E, b_E, circ=False)
        gradstack.extend(J_E_g.flatten())
        hessstack.append(J_E_h)

    # Compute f
    f = np.sum(rate - Y * u) + J_P + J_R + J_E

    # Gradient
    df = np.real(X.T @ (rate - Y) + gradstack)
    df = df.squeeze()

    # Hessian
    hessian = hessian_glm + sparse.block_diag(hessstack).toarray()

    return float(f), df, hessian



def fit_LNLP_model(A_input, dt, spiketrain, filter, modelType, param_counts, numFolds=10, ret_for_MP=True):
    """ Fit a linear-nonlinear-poisson model.

    Parameters
    ----------
    A_input : array_like
        Behavioral variables one-hot encoded.
    dt : float
        Time step.
    spiketrain : array_like
        Spike counts for a single cell.
    filter : array_like
        Filter for smoothing the spike train.
    modelType : str
        Model type string. Must contain only the character P, R, E, and/or D.
    param_counts : array_like
        Number of parameters for each of the four model types.
    numFolds : int, optional
        Number of folds for k-fold cross-validation. Default is 10.
    ret_for_MP : bool, optional
        If True, all results are returned as a single tuple. Otherwise, as individual
        array-like values. Default is True.
    lag : int, optional
        How much should spikes lag the behavior data, in units of milliseconds. Default
        is 255.

    Returns
    -------
    testFit : array_like
        Fit results for the test set. The array has the shape (10,6), where 10 is the number
        of k-fold iterations and 6 is the number of metrics, in order:
            1. Explained variance
            2. Correlation
            3. Log likelihood
            4. Mean squared error
            5. Number of spikes
            6. Number of time points
    trainFit : array_like
        Same as testFit, but for the training set.
    param_mean : array_like
        Mean parameter values across the k-fold iterations.
    param_stderr : array_like
        Standard error of the parameter values.
    predSpikes : array_like
        Predicted spike counts (for all cells).
    trueSpikes : array_like
        True spike counts (for all cells).
    """

    A = A_input.copy()

    # Index into the columns carrying parameters.
    if 'P' not in modelType:
        sc = 0                  # start column
        ec = param_counts[0]    # end column
        A[:, sc:ec] = np.zeros([np.size(A, 0), param_counts[0]]) * np.nan
    if 'R' not in modelType:
        sc = param_counts[0]
        ec = param_counts[0]+param_counts[1]
        A[:, sc:ec] = np.zeros([np.size(A, 0), param_counts[1]]) * np.nan
    if 'E' not in modelType:
        sc = param_counts[0]+param_counts[1]
        ec = param_counts[0]+param_counts[1]+param_counts[2]
        A[:, sc:ec] = np.zeros([np.size(A, 0), param_counts[2]]) * np.nan

    # Delete columns of all NaNs
    A = A[:, np.sum(np.isnan(A), axis=0)==0]

    # Divide the data up into 5*num_folds pieces
    numCol = np.size(A, 1)
    sections = numFolds * 5
    edges = np.round(np.linspace(0, len(spiketrain)-1, sections + 1)).astype(int)

    # Initialize matrices
    testFit = np.full((numFolds, 6), np.nan)
    trainFit = np.full((numFolds, 6), np.nan)
    paramMat = np.full((numFolds, numCol), np.nan)
    predSpikes = []
    trueSpikes = []

    # Perform k-fold cross validation
    for k in range(numFolds):

        # Get test data from edges
        fin_edge_index = k+4*numFolds+2
        if k == numFolds-1:
            fin_edge_index -= 1

        test_ind = np.concatenate([
            np.arange(edges[k], edges[k+2]),
            np.arange(edges[k+numFolds], edges[k+numFolds+2]),
            np.arange(edges[k+2*numFolds], edges[k+2*numFolds+2]),
            np.arange(edges[k+3*numFolds], edges[k+3*numFolds+2]),
            np.arange(edges[k+4*numFolds], edges[fin_edge_index])
        ])

        test_spikes = spiketrain[test_ind]
        smooth_spikes_test = np.convolve(test_spikes, filter, 'same') 
        smooth_fr_test = smooth_spikes_test / dt
        test_A = A[test_ind,:]

        # Training data
        train_ind = np.setdiff1d(np.arange(len(spiketrain)), test_ind)
        train_spikes = spiketrain[train_ind]
        smooth_spikes_train = np.convolve(train_spikes, filter, 'same')
        smooth_fr_train = smooth_spikes_train / dt
        train_A = A[train_ind,:]

        if k == 0:
            init_param = 1e-3 * np.random.randn(numCol)
        else:
            init_param = param

        if len(init_param) == 0:
            init_param = 1e-3 * np.random.randn(numCol)

        # Peform the fit
        res = minimize(
            fm2p.linear_nonlinear_poisson_model,
            init_param,
            args=(train_A, train_spikes, modelType, param_counts),
            method='Newton-CG',
            jac=True,
            hess='2-point',
            options={'disp': False, 'maxiter': 5000})
        
        param = res.x

        # Test data
        fr_hat_test = np.exp(test_A @ param) / dt
        smooth_fr_hat_test = np.convolve(fr_hat_test, filter, 'same') 

        sse = np.sum((smooth_fr_hat_test - smooth_fr_test) ** 2)
        sst = np.sum((smooth_fr_test - np.mean(smooth_fr_test)) ** 2)
        varExplain_test = 1 - (sse / sst)

        correlation_test = pearsonr(smooth_fr_test, smooth_fr_hat_test)[0]

        r = np.exp(test_A @ param)
        n = test_spikes
        meanFR_test = np.nanmean(test_spikes)

        log_llh_test_model = np.nansum(r - n * np.log(r) + np.log(factorial(n))) / np.sum(n)
        log_llh_test_mean = np.nansum(meanFR_test - n * np.log(meanFR_test) + np.log(factorial(n))) / np.sum(n)
        log_llh_test = (-log_llh_test_model + log_llh_test_mean)
        log_llh_test = log_llh_test / np.log(2)

        mse_test = np.nanmean((smooth_fr_hat_test - smooth_fr_test) ** 2)

        testFit[k, :] = [
            varExplain_test,
            correlation_test,
            log_llh_test,
            mse_test,
            np.sum(n),
            len(test_ind)
        ]

        # Train data
        fr_hat_train = np.exp(train_A @ param) / dt
        smooth_fr_hat_train = np.convolve(fr_hat_train, filter, 'same') 

        sse = np.sum((smooth_fr_hat_train - smooth_fr_train) ** 2)
        sst = np.sum((smooth_fr_train - np.mean(smooth_fr_train)) ** 2)
        varExplain_train = 1 - (sse / sst)

        correlation_train = pearsonr(smooth_fr_train, smooth_fr_hat_train)[0]

        r_train = np.exp(train_A @ param)
        n_train = train_spikes
        meanFR_train = np.nanmean(train_spikes)

        log_llh_train_model = np.nansum(r_train - n_train * np.log(r_train) + np.log(factorial(n_train))) / np.sum(n_train)
        log_llh_train_mean = np.nansum(meanFR_train - n_train * np.log(meanFR_train) + np.log(factorial(n_train))) / np.sum(n_train)
        log_llh_train = (-log_llh_train_model + log_llh_train_mean)
        log_llh_train = log_llh_train / np.log(2)

        mse_train = np.nanmean((smooth_fr_hat_train - smooth_fr_train) ** 2)

        trainFit[k, :] = [
            varExplain_train,
            correlation_train,
            log_llh_train,
            mse_train,
            np.sum(n_train),
            len(train_ind)
        ]

        paramMat[k, :] = param

        predSpikes.extend(r)
        trueSpikes.extend(test_spikes)

    param_mean = np.nanmean(paramMat, axis=0)

    if ret_for_MP is True:
        return (testFit, trainFit, param_mean, paramMat, np.array(predSpikes), np.array(trueSpikes))
    else:
        return testFit, trainFit, param_mean, paramMat, np.array(predSpikes), np.array(trueSpikes)


def fit_all_LNLP_models(data_vars, data_bins, spikes, savedir):
    """ Fit all neurons to LNLP model for all model combinations.

    Parameters
    ----------
    data_vars : tuple
        Tuple of data variables (pupil_data, ret_data, ego_data, dist_data).
    spikes : array_like
        Spike counts for all cells.
    savedir : str
        Directory to save the results.
    
    Returns
    -------
    all_model_results : dict
        Returns a dictionary of all model results for every model
        fit and every neuron. For each model key (e.g., 'PR'), the
        saved results include:
            1. testFit
            2. trainFit
            3. param_mean
            4. param_stderr
            5. predSpikes
            6. trueSpikes
        See output of function `fit_LNLP_model` for more details on
        each output.
    """
    pdf_path = os.path.join(savedir, 'model_fits.pdf')
    pdf = PdfPages(pdf_path)

    mapP, mapR, mapE = data_vars
    pupil_bins, ret_bins, ego_bins = data_bins

    # Visualize the ont-hot encoded maps of behavior variables
    fig, [ax1,ax2,ax3] = plt.subplots(1,3,dpi=300,figsize=(6,5))
    ax1.imshow(mapP, aspect=0.005)
    ax2.imshow(mapR, aspect=0.015)
    ax3.imshow(mapE, aspect=0.015)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax1.set_title('pupil')
    ax2.set_title('retinotopic')
    ax3.set_title('egocentric')
    fig.suptitle('One-hot encoded behavior variables')
    fig.tight_layout()
    pdf.savefig()
    plt.close()

    fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2,dpi=300,figsize=(4,3))
    ax1.plot(pupil_bins, np.mean(mapP, 0), color='tab:blue')
    ax2.plot(ret_bins, np.mean(mapR, 0), color='tab:orange')
    ax3.plot(ego_bins, np.mean(mapE, 0), color='tab:green')
    ax1.set_ylim([0,0.4])
    ax2.set_ylim([0,0.1])
    ax3.set_ylim([0,0.1])
    ax1.set_xlabel('pupil (deg)')
    ax2.set_xlabel('retinotopic (deg)')
    ax3.set_xlabel('egocentric (deg)')
    ax4.axis('off')
    fig.suptitle('Behavioral occupancy')
    fig.tight_layout()
    pdf.savefig()
    plt.close()

    A = np.concatenate([mapP, mapR, mapE], axis=1)
    A = A[np.sum(np.isnan(A), axis=1)==0, :]

    param_counts = [
        len(pupil_bins),
        len(ret_bins),
        len(ego_bins)
    ]

    # Generate all model combinations
    model_keys = []
    for count in np.arange(1,5):
        c_ = [''.join(x) for x in list(combinations(['P','R','E'], count))]
        model_keys.extend(c_)

    proc_cells = np.arange(np.size(spikes,0))

    all_model_results = {}

    n_proc = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=n_proc)

    amfstart = datetime.now()

    # Iterate through all models
    for mi, mk in tqdm(enumerate(model_keys)):

        # print('Fitting model for {}     (model {}/{})'.format(mk, mi+1, len(model_keys)))
        # mfstart = datetime.now()

        # Set up multiprocessing 
        param_mp = [
            pool.apply_async(
                fit_LNLP_model,
                args=(A, 0.05, spikes[ci,:], np.ones(8), mk, param_counts, 10, True)
            ) for ci in proc_cells
        ]

        # Get the values
        params_output = [result.get() for result in param_mp]

        # Iterate through results and organize into dict
        all_model_results[mk] = {}
        current_model_results = {}

        for ci, cell_fit in enumerate(params_output):

            testFit, trainFit, param_mean, paramMat, predSpikes, trueSpikes = cell_fit

            current_model_results[str(ci)] = {
                'testFit': testFit,
                'trainFit': trainFit,
                'param_mean': param_mean,
                'param_matrix': paramMat,
                'predSpikes': predSpikes,
                'trueSpikes': trueSpikes
            }

        all_model_results[mk] = current_model_results

        savepath = os.path.join(savedir, 'model_{}_results.h5'.format(mk))
        fm2p.write_h5(savepath, current_model_results)

        # mfend = datetime.now()
        # mf_timedelta = (mfend - mfstart).total_seconds() / 60.
        # print('  Time to fit: {} min'.format(int(mf_timedelta)))

    pdf.close()

    amfend = datetime.now()
    amf_timedelta = (amfend - amfstart).total_seconds() / 60.
    print('Time to fit all models:: {} min'.format(amf_timedelta))

    return all_model_results

