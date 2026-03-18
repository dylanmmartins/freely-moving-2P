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

from .LNP_helpers import find_param, rough_penalty
from .files import write_h5


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
        Model type string. Contains the 
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
    b_A = 5e1     # theta
    b_B = 5e1     # phi
    b_C = 5e1    # dTheta
    b_D = 5e1    # dPhi

    # Start computing the Hessian
    rX = np.multiply(rate[:, np.newaxis], X)
    hessian_glm = rX.T @ X

    # Initialize parameter-relevant variables
    J_A = 0
    J_A_g = np.array([])
    J_A_h = np.array([])
    J_B = 0
    J_B_g = np.array([])
    J_B_h = np.array([])
    J_C = 0
    J_C_g = np.array([])
    J_C_h = np.array([])
    J_D = 0
    J_D_g = np.array([])
    J_D_h = np.array([])

    numA, numB, numC, numD = param_counts

    param_A, param_B, param_C, param_D = find_param(param, modelType, numA, numB, numC, numD)

    gradstack = []
    hessstack = []

    # Compute the contribution for f, df, and the hessian
    if param_A.size != 0:
        J_A, J_A_g, J_A_h = rough_penalty(param_A, b_A)
        gradstack.extend(J_A_g.flatten())
        hessstack.append(J_A_h)

    if param_B.size != 0:
        J_B, J_B_g, J_B_h = rough_penalty(param_B, b_B, circ=False)
        gradstack.extend(J_B_g.flatten())
        hessstack.append(J_B_h)

    if param_C.size != 0:
        J_C, J_C_g, J_C_h = rough_penalty(param_C, b_C, circ=False)
        gradstack.extend(J_C_g.flatten())
        hessstack.append(J_C_h)

    if param_D.size != 0:
        J_D, J_D_g, J_D_h = rough_penalty(param_D, b_D, circ=False)
        gradstack.extend(J_D_g.flatten())
        hessstack.append(J_D_h)

    # Compute f
    f = np.sum(rate - Y * u) + J_A + J_B + J_C + J_D

    # Gradient
    df = np.real(X.T @ (rate - Y) + gradstack)
    df = df.squeeze()

    # Hessian
    hessian = hessian_glm + sparse.block_diag(hessstack).toarray()

    return float(f), df, hessian



def fit_LNLP_model(behavior_input, dt, spiketrain, filter, modelType, param_counts, numFolds=10, ret_for_MP=True):
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

    Ib = behavior_input.copy()

    # Index into the columns carrying parameters.
    if 'A' not in modelType:
        sc = 0                  # start column
        ec = param_counts[0]    # end column
        Ib[:, sc:ec] = np.zeros([np.size(Ib, 0), param_counts[0]]) * np.nan
    if 'B' not in modelType:
        sc = param_counts[0]
        ec = param_counts[0]+param_counts[1]
        Ib[:, sc:ec] = np.zeros([np.size(Ib, 0), param_counts[1]]) * np.nan
    if 'C' not in modelType:
        sc = param_counts[0]+param_counts[1]
        ec = param_counts[0]+param_counts[1]+param_counts[2]
        Ib[:, sc:ec] = np.zeros([np.size(Ib, 0), param_counts[2]]) * np.nan
    if 'D' not in modelType:
        sc = param_counts[0]+param_counts[1]+param_counts[2]
        ec = param_counts[0]+param_counts[1]+param_counts[2]+param_counts[3]
        Ib[:, sc:ec] = np.zeros([np.size(Ib, 0), param_counts[3]]) * np.nan

    # Delete columns of all NaNs
    Ib = Ib[:, np.sum(np.isnan(Ib), axis=0)==0]

    # Divide the data up into 5*num_folds pieces
    numCol = np.size(Ib, 1)
    sections = numFolds * 5
    edges = np.round(np.linspace(0, len(spiketrain)-1, sections + 1)).astype(int)

    # Initialize matrices
    testFit = np.full((numFolds, 6), np.nan) # will save out six parameters
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
        test_Ib = Ib[test_ind,:]

        # Training data
        train_ind = np.setdiff1d(np.arange(len(spiketrain)), test_ind)
        train_spikes = spiketrain[train_ind]
        smooth_spikes_train = np.convolve(train_spikes, filter, 'same')
        smooth_fr_train = smooth_spikes_train / dt
        train_Ib = Ib[train_ind,:]

        if k == 0:
            init_param = 1e-3 * np.random.randn(numCol)
        else:
            init_param = param

        if len(init_param) == 0:
            init_param = 1e-3 * np.random.randn(numCol)

        # Peform the fit
        res = minimize(
            linear_nonlinear_poisson_model,
            init_param,
            args=(train_Ib, train_spikes, modelType, param_counts),
            method='Newton-CG',
            jac=True,
            hess='2-point',
            options={'disp': False, 'maxiter': 5000})
        
        param = res.x

        # Test data
        fr_hat_test = np.exp(test_Ib @ param) / dt
        smooth_fr_hat_test = np.convolve(fr_hat_test, filter, 'same') 

        sse = np.sum((smooth_fr_hat_test - smooth_fr_test) ** 2)
        sst = np.sum((smooth_fr_test - np.mean(smooth_fr_test)) ** 2)
        varExplain_test = 1 - (sse / sst)

        correlation_test = pearsonr(smooth_fr_test, smooth_fr_hat_test)[0]

        r = np.exp(test_Ib @ param)
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
        fr_hat_train = np.exp(train_Ib @ param) / dt
        smooth_fr_hat_train = np.convolve(fr_hat_train, filter, 'same') 

        sse = np.sum((smooth_fr_hat_train - smooth_fr_train) ** 2)
        sst = np.sum((smooth_fr_train - np.mean(smooth_fr_train)) ** 2)
        varExplain_train = 1 - (sse / sst)

        correlation_train = pearsonr(smooth_fr_train, smooth_fr_hat_train)[0]

        r_train = np.exp(train_Ib @ param)
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


def get_colors():
    return [
        '#0d0887',
        '#9c179e',
        '#ed7953',
        '#f0f921'
    ]


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

    colors = get_colors()

    pdf_path = os.path.join(savedir, 'model_fits.pdf')
    pdf = PdfPages(pdf_path)

    mapA, mapB, mapC, mapD = data_vars
    A_bins, B_bins, C_bins, D_bins = data_bins

    # Visualize the ont-hot encoded maps of behavior variables
    fig, [ax1,ax2,ax3,ax4] = plt.subplots(1,4,dpi=300,figsize=(6,5))
    ax1.imshow(mapA, aspect=0.005)
    ax2.imshow(mapB, aspect=0.015)
    ax3.imshow(mapC, aspect=0.015)
    ax4.imshow(mapD, aspect=0.015)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')
    ax1.set_title('theta')
    ax2.set_title('phi')
    ax3.set_title('dTheta')
    ax4.set_title('dPhi')
    fig.suptitle('One-hot encoded behavior variables')
    fig.tight_layout()
    pdf.savefig()
    plt.close()

    fig, [[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2,dpi=300,figsize=(4,3))
    ax1.plot(A_bins, np.mean(mapA, 0), color=colors[0])
    ax2.plot(B_bins, np.mean(mapB, 0), color=colors[1])
    ax3.plot(C_bins, np.mean(mapC, 0), color=colors[2])
    ax4.plot(D_bins, np.mean(mapD, 0), color=colors[3])
    ax1.set_ylim([0,0.4])
    ax2.set_ylim([0,0.1])
    ax3.set_ylim([0,0.1])
    ax1.set_xlabel('theta (deg)')
    ax2.set_xlabel('phi (deg)')
    ax3.set_xlabel('dTheta (deg/s)')
    ax4.set_xlabel('dPhi (deg/s)')
    fig.suptitle('Behavioral occupancy')
    fig.tight_layout()
    pdf.savefig()
    plt.close()

    Ib = np.concatenate([mapA, mapB, mapC, mapD], axis=1)
    Ib = Ib[np.sum(np.isnan(Ib), axis=1)==0, :]

    param_counts = [
        len(A_bins),
        len(B_bins),
        len(C_bins),
        len(D_bins)
    ]

    # Generate all model combinations
    model_keys = []
    for count in np.arange(1,5):
        c_ = [''.join(x) for x in list(combinations(['A','B','C','D'], count))]
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
                args=(Ib, 0.05, spikes[ci,:], np.ones(8), mk, param_counts, 10, True)
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
        write_h5(savepath, current_model_results)

        # mfend = datetime.now()
        # mf_timedelta = (mfend - mfstart).total_seconds() / 60.
        # print('  Time to fit: {} min'.format(int(mf_timedelta)))

    pdf.close()

    amfend = datetime.now()
    amf_timedelta = (amfend - amfstart).total_seconds() / 60.
    print('Time to fit all models:: {} min'.format(amf_timedelta))

    return all_model_results

