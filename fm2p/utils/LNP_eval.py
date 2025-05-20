# -*- coding: utf-8 -*-
"""
Linear-nonlinear Poisson model evaluation.

This module contains functions for evaluating the performance of the linear-nonlinear
Poisson (LNP) model on neural data.

Functions
---------
add_scatter_col(ax, pos, vals)
    Add a scatter plot of values to a given axis at a specified position.
read_models(models_dir)
    Read model data from specified directory.
plot_model_LLHs(model_data, unit_num, test_only=False, fig=None, ax=None, tight_y_scale=False)
    Plot log likelihoods for different model types.
_get_best_model(model_data, uk, test_keys)
    Get the best model for a given cell based on the maximum log likelihood.
eval_models(model_data, unit_num, wilcoxon_thresh=0.05)
    Evaluate models for a given cell using the Wilcoxon signed-rank test.
plot_rank_test_results(model_data, test_results, unit_num, fig=None, axs=None)
    Plot the results of the Wilcoxon signed-rank test for model evaluation.
dictinds_to_arr(dic)
    Convert a dictionary with string keys to a numpy array.
plot_pred_spikes(model_data, unit_num, selected_models, fig=None, axs=None)
    Plot predicted spikes for different models.
calc_scaled_LNLP_tuning_curves(model_data=None, unit_num=0, ret_stderr=True, params=None, param_stderr=None)
    Calculate scaled tuning curves for the LNP model.
plot_scaled_LNLP_tuning_curves(predP, predR, predE, errP, errR, errE, pupil_bins, retino_bins, ego_bins,
        predP2=None, predR2=None, predE2=None, errP2=None, errR2=None, errE2=None,
        fig=None, axs=None)
    Plot scaled tuning curves for the LNP model.
calc_bootstrap_model_params(data_vars, var_bins, spikes, n_iter=30)
    Calculate bootstrap model parameters for the LNP model.
get_cells_best_LLHs(model_data)
    Get the best log likelihood for each cell in the model data.
determine_responsiveness_from_null(model_path, null_path, null_thresh=0.99)
    Determine responsiveness of cells based on log likelihood threshold from null model.
get_responsive_inds(model_data, LLH_threshold)
    Get indices of responsive cells based on log likelihood threshold.
get_responsive_inds_2(model1_data, model2_data, LLH_threshold, thresh2=None)
    Get indices of responsive cells based on log likelihood threshold for two models.

Author: DMM, 2024
"""


import os
import numpy as np
import scipy.stats
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 8
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

import fm2p


def add_scatter_col(ax, pos, vals):
    """ Add a scatter plot of values to a given axis at a specified position.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to which the scatter plot will be added.
    pos : float
        The x-coordinate position of the scatter plot.
    vals : array-like
        The values to be plotted on the y-axis.
    """

    ax.scatter(
        np.ones_like(vals)*pos + (np.random.rand(len(vals))-0.5)/10,
        vals,
        s=2, c='k'
    )
    ax.hlines(np.nanmean(vals), pos-.1, pos+.1, color='r')

    stderr = np.nanstd(vals) / np.sqrt(len(vals))
    ax.vlines(pos, np.nanmean(vals)-stderr, np.nanmean(vals)+stderr, color='r')


def read_models(models_dir):
    """ Read model data from specified directory.

    Each model type is stored in a separate HDF5 file, and the function reads
    the data into a single shared directory.
    
    Parameters
    ----------
    models_dir : str
        Directory containing model results.
    
    Returns
    -------
    model_data : dict
        Dictionary containing model results for each model type.
    """

    key_list = [
        'P','R','E',
        'PR','PE','RE',
        'PRE'
    ]

    model_data = {}
    for mk in key_list:
        model_data[mk] = fm2p.read_h5(os.path.join(
            os.path.join(models_dir, 'model_{}_results.h5'.format(mk))
        ))
    
    return model_data


def plot_model_LLHs(model_data, unit_num, test_only=False, fig=None, ax=None, tight_y_scale=False):
    """ Plot log likelihoods for different model types.
    
    Parameters
    ----------
    model_data : dict
        Dictionary containing model results for each model type.
    unit_num : int
        Unit number to plot.
    test_only : bool, optional
        If True, only plot test log likelihoods (not training log likelihoods). Default is False.
    fig : matplotlib.figure.Figure, optional
        Figure object to plot on. If None, a new figure is created. Default is None.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new axes is created. Default is None.
    tight_y_scale : bool, optional
        If True, set y-axis limits to the maximum log likelihood value. Default is False.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the plot.
    """

    uk = str(unit_num)

    key_list = [
        'P','R','E',
        'PR','PE','RE',
        'PRE'
    ]

    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(1,1, dpi=300, figsize=(6,2))

    # Horizontal line for zero log likeihood
    ax.hlines(0, -.5, len(key_list)+.5, color='k', linestyle='--', lw=1, alpha=0.3)

    # Iterate through each model
    for ki, mk in enumerate(key_list):

        # Get the log likelihood values for the current model
        llh_test = model_data[mk][uk]['testFit'][:,2]
        llh_train = model_data[mk][uk]['trainFit'][:,2]

        if not test_only:
            add_scatter_col(ax, ki, llh_train)
            add_scatter_col(ax, ki+0.3, llh_test)
            set_y_max = np.maxiumum(np.max(llh_train), np.max(llh_test))
        elif test_only:
            add_scatter_col(ax, ki, llh_test)
            set_y_max = np.max(llh_test)

    if not test_only:
        ax.set_xticks(np.arange(0, len(key_list))+0.15, labels=key_list)
    elif test_only:
        ax.set_xticks(np.arange(0, len(key_list)), labels=key_list)

    ax.set_ylabel('log likelihood (mean across k-folds)')
    ax.set_xlim([-0.5, len(key_list)+.5])

    if tight_y_scale:
        ax.set_ylim([0, set_y_max])

    fig.tight_layout()

    return fig


def _get_best_model(model_data, uk, test_keys):
    """ Get the best model for a given cell based on the maximum log likelihood.

    Parameters
    ----------
    model_data : dict
        Dictionary containing model results for each model type.
    uk : str
        Unit number as a string.
    test_keys : list
        List of model keys to evaluate.
    
    Returns
    -------
    best_model : str
        The model key with the highest average log likelihood.
    """

    model_ind = np.argmax([np.nanmean(model_data[mk][uk]['testFit'][:,2]) for mk in test_keys])
    best_model = test_keys[model_ind]
    
    return best_model


def eval_models(model_data, unit_num, wilcoxon_thresh=0.05):
    """ Evaluate models for a given cell using the Wilcoxon signed-rank test.
    
    Parameters
    ----------
    model_data : dict
        Dictionary containing model results for each model type.
    unit_num : int
        Unit number to evaluate.
    wilcoxon_thresh : float, optional
        Threshold for the Wilcoxon signed-rank test. Default is 0.05.

    Returns
    -------
    results : dict
        Dictionary containing the selected models, best model, and Wilcoxon test results.
    """

    uk = str(unit_num)

    all_1st_keys = ['P','R','E']
    all_2nd_keys = ['PR','PE','RE']

    best_1st_order = _get_best_model(model_data, uk, all_1st_keys)
    best_2nd_order = _get_best_model(model_data, uk, [mk for mk in all_2nd_keys if best_1st_order in mk])
    best_3rd_order = 'PRE'

    best_of_orders = [best_1st_order, best_2nd_order, best_3rd_order]

    test_12 = scipy.stats.wilcoxon(
        model_data[best_1st_order][uk]['testFit'][:,2],
        model_data[best_2nd_order][uk]['testFit'][:,2],
        alternative='less'
    ).pvalue
    test_23 = scipy.stats.wilcoxon(
        model_data[best_2nd_order][uk]['testFit'][:,2],
        model_data['PRE'][uk]['testFit'][:,2],
        alternative='less'
    ).pvalue

    wilcoxon_results = [0, test_12, test_23]

    best_model = np.nan

    results = {
        'sel_models': [best_1st_order, best_2nd_order, best_3rd_order],
        'best_model': best_model,
        'test_12': test_12,
        'test_23': test_23,
    }

    for k in range(3):

        res = wilcoxon_results[k]

        # If the p value is small, the current model is improved by adding
        # the additional variable, so we move on to compare to the best
        # model from the higher order.
        if (res < wilcoxon_thresh):
            pass
        # Otherwise, if the result is larger than threshold, this model is
        # not improved by adding more parameters, and we keep this model as
        # the best fit.
        elif (res > wilcoxon_thresh):
            
            best_model = best_of_orders[k-1]

            results['best_model'] = best_model

            # Return needs to happen in loop to return on the first model
            # to pass threshold, not the last.
            return results


def plot_rank_test_results(model_data, test_results, unit_num, fig=None, axs=None):
    """ Plot the results of the Wilcoxon signed-rank test for model evaluation.

    Parameters
    ----------
    model_data : dict
        Dictionary containing model results for each model type.
    test_results : dict
        Dictionary containing the results of the Wilcoxon signed-rank test.
    unit_num : int
        Unit number to plot.
    fig : matplotlib.figure.Figure, optional
        Figure object to plot on. If None, a new figure is created. Default is None.
    axs : list of matplotlib.axes.Axes, optional
        List of axes objects to plot on. If None, new axes are created. Default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the plot.
    """

    uk = str(unit_num)


    best_1st_order, best_2nd_order, best_3rd_order = test_results['sel_models']

    # log likelihood values
    allvals_tmp = np.concatenate([
        model_data[best_1st_order][uk]['testFit'][:,2],
        model_data[best_2nd_order][uk]['testFit'][:,2],
        model_data['PRE'][uk]['testFit'][:,2]
    ])
    set_min = np.nanmin(allvals_tmp) - np.nanmin(allvals_tmp)*0.01
    set_max = np.nanmax(allvals_tmp) + np.nanmax(allvals_tmp)*0.01
    set_min = np.round(set_min, 2)
    set_max = np.round(set_max, 2)
    set_mid = np.mean([set_min, set_max])

    test_12 = test_results['test_12']
    test_23 = test_results['test_23']

    if (fig is None) and (axs is None):
        fig, [ax1,ax2] = plt.subplots(1,2, figsize=(5,2), dpi=300)
    else:
        [ax1,ax2] = axs

    for ax in [ax1,ax2]:
        ax.set_xlim([set_min, set_max]); ax.set_ylim([set_min, set_max])
        ax.set_xticks([set_min, set_mid, set_max]); ax.set_yticks([set_min, set_mid, set_max])
        ax.plot([set_min, set_max],[set_min, set_max], color='k', linestyle='--', lw=1, alpha=0.3)
        ax.axis('square')

    ax1.scatter(
        model_data[best_1st_order][uk]['testFit'][:,2],
        model_data[best_2nd_order][uk]['testFit'][:,2],
        color='k', s=3.5
    )
    ax1.set_title('p={:.3f}'.format(test_12))
    ax1.set_xlabel('{} model LLH'.format(best_1st_order))
    ax1.set_ylabel('{} model LLH'.format(best_2nd_order))

    ax2.scatter(
        model_data[best_2nd_order][uk]['testFit'][:,2],
        model_data[best_3rd_order][uk]['testFit'][:,2],
        color='k', s=3.5
    )
    ax2.set_title('p={:.3f}'.format(test_23))
    ax2.set_xlabel('{} model LLH'.format(best_2nd_order))
    ax2.set_ylabel('{} model LLH'.format('PRE'))

    fig.suptitle('Wilcoxon signed-rank test (best={})'.format(test_results['best_model']))

    fig.tight_layout()

    return fig


def dictinds_to_arr(dic):
    """ Convert a dictionary with string keys to a numpy array.
    
    Parameters
    ----------
    dic : dict
        Dictionary with string keys representing indices and values.
    
    Returns
    -------
    arr : numpy.ndarray
        Numpy array with values from the dictionary, indexed by the integer keys.
    """

    maxkey = np.max([int(x) for x in dic.keys()])
    arr = np.zeros(maxkey)

    for i in range(maxkey):
        arr[i] = dic[str(i)]

    return arr


def plot_pred_spikes(model_data, unit_num, selected_models, fig=None, axs=None):
    """ Plot predicted spikes for different models.
    
    Parameters
    ----------
    model_data : dict
        Dictionary containing model results for each model type.
    unit_num : int
        Unit number to plot.
    selected_models : list
        List of model keys to plot.
    fig : matplotlib.figure.Figure, optional
        Figure object to plot on. If None, a new figure is created. Default is None.
    axs : list of matplotlib.axes.Axes, optional
        List of axes objects to plot on. If None, new axes are created. Default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the plot. 
    """

    uk = str(unit_num)

    _sp_true = model_data[selected_models[0]][uk]['trueSpikes']
    if type(_sp_true) == dict:
        _sp_true = dictinds_to_arr(_sp_true)

    setsz = len(_sp_true)
    fakeT = np.linspace(0, setsz*0.05, setsz)
    if (fig is None) and (axs is None):
        fig, axs = plt.subplots(4,1,figsize=(4,3), dpi=300)

    for mi, mk in enumerate(selected_models):
        _sp_vals = model_data[mk][uk]['predSpikes']
        if type(_sp_vals) == dict:
            _sp_vals = dictinds_to_arr(_sp_vals)
        axs[mi+1].plot(fakeT, _sp_vals, lw=0.5, alpha=0.9, label='mk')
    axs[0].plot(fakeT, _sp_true, 'k', lw=0.5, alpha=0.9, label='ground truth')
    
    ax1, ax2, ax3, ax4 = axs

    for ax in [ax1,ax2,ax3]:
        ax.set_xticks([])
    for ax in [ax1,ax2,ax3,ax4]:
        ax.set_xlim([0,120])
        ax.set_ylabel('norm sp')
        ax.set_ylim([0,1.5])
    fig.tight_layout()

    ax4.set_xlabel('time (sec)')
    fig.tight_layout()

    return fig


def calc_scaled_LNLP_tuning_curves(model_data=None, unit_num=0, ret_stderr=True,
                                   params=None, param_stderr=None):
    """ Calculate scaled tuning curves for the LNP model.
    
    Parameters
    ----------
    model_data : dict, optional
        Dictionary containing model results for each model type. Default is None.
    unit_num : int, optional
        Unit number to calculate tuning curves for. Default is 0.
    ret_stderr : bool, optional
        If True, return standard error of the tuning curves. Default is True.
    params : array-like, optional
        Parameters for the model. Default is None.
    param_stderr : array-like, optional
        Standard error of the parameters. Default is None.

    Returns
    -------
    predP : numpy.ndarray
        Predicted pupil tuning curve.
    predR : numpy.ndarray
        Predicted retino tuning curve.
    predE : numpy.ndarray
        Predicted ego tuning curve.
    """

    uk = str(unit_num)

    mk = 'PRE'

    if params is None:
        params = model_data[mk][uk]['param_mean']

    tuningP, tuningR, tuningE = fm2p.find_param(
        params,
        mk,
        10, 36, 36
    )
    
    # Scale factor to convert to units of spikes/sec
    scale_factor_P = np.nanmean(np.exp(tuningR)) * np.nanmean(np.exp(tuningE))
    scale_factor_R = np.nanmean(np.exp(tuningP)) * np.nanmean(np.exp(tuningE))
    scale_factor_E = np.nanmean(np.exp(tuningP)) * np.nanmean(np.exp(tuningR))
    
    # Compute model-derived response profile
    predP = np.exp(tuningP) * scale_factor_P
    predR = np.exp(tuningR) * scale_factor_R
    predE = np.exp(tuningE) * scale_factor_E

    if (param_stderr is None) and (ret_stderr is True):

        k = 10

        param_matrix = model_data[mk][uk]['param_matrix']

        k_tuningP = np.zeros([k,10])
        k_tuningR = np.zeros([k,36])
        k_tuningE = np.zeros([k,36])
        
        for k_i in range(k):

            ki_tuningP, ki_tuningR, ki_tuningE = fm2p.find_param(
                param_matrix[k_i,:],
                mk,
                10, 36, 36
            )

            k_tuningP[k_i,:] = np.exp(ki_tuningP) * scale_factor_P
            k_tuningR[k_i,:] = np.exp(ki_tuningR) * scale_factor_R
            k_tuningE[k_i,:] = np.exp(ki_tuningE) * scale_factor_E

        errP = np.std(k_tuningP, 0)
        errR = np.std(k_tuningR, 0)
        errE = np.std(k_tuningE, 0)

        return (predP, errP), (predR, errR), (predE, errE)

    return predP, predR, predE


def plot_scaled_LNLP_tuning_curves(predP, predR, predE,
                                   errP, errR, errE,
                                   pupil_bins, retino_bins, ego_bins,
                                   predP2=None, predR2=None, predE2=None,
                                   errP2=None, errR2=None, errE2=None,
                                   fig=None, axs=None):
    """ Plot scaled tuning curves for the LNP model.

    Parameters
    ----------
    predP : numpy.ndarray
        Predicted pupil tuning curve.
    predR : numpy.ndarray
        Predicted retino tuning curve.
    predE : numpy.ndarray
        Predicted ego tuning curve.
    errP : numpy.ndarray
        Standard error of the predicted pupil tuning curve.
    errR : numpy.ndarray
        Standard error of the predicted retino tuning curve.
    errE : numpy.ndarray
        Standard error of the predicted ego tuning curve.
    pupil_bins : numpy.ndarray
        Pupil bin edges.
    retino_bins : numpy.ndarray
        Retino bin edges.
    ego_bins : numpy.ndarray
        Ego bin edges.
    predP2 : numpy.ndarray, optional
        Second predicted pupil tuning curve. Optional, default is None.
    predR2 : numpy.ndarray, optional
        Second predicted retino tuning curve. Optional, default is None.
    predE2 : numpy.ndarray, optional
        Second predicted ego tuning curve. Optional, default is None.
    errP2 : numpy.ndarray, optional
        Standard error of the second predicted pupil tuning curve. Optional, default is None.
    errR2 : numpy.ndarray, optional
        Standard error of the second predicted retino tuning curve. Optional, default is None.
    errE2 : numpy.ndarray, optional
        Standard error of the second predicted ego tuning curve. Optional, default is None.
    fig : matplotlib.figure.Figure, optional
        Figure object to plot on. If None, a new figure is created. Default is None.
    axs : list of matplotlib.axes.Axes, optional
        List of axes objects to plot on. If None, new axes are created. Default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the plot.
    """

    if (fig is None) and (axs is None):
        fig, [ax1,ax2,ax3] = plt.subplots(1,3, figsize=(5,1.8), dpi=300)
    else:
        [ax1,ax2,ax3] = axs

    ax1.plot(pupil_bins, predP, color='tab:blue')
    ax1.fill_between(pupil_bins, predP-errP, predP+errP, color='tab:blue', alpha=0.3)
    ax1.set_xlabel('pupil (deg)')

    ax2.plot(retino_bins, predR, color='tab:orange')
    ax2.fill_between(retino_bins, predR-errR, predR+errR, color='tab:orange', alpha=0.3)
    ax2.set_xlabel('retino (deg)')

    ax3.plot(ego_bins, predE, color='tab:green')
    ax3.fill_between(ego_bins, predE-errE, predE+errE, color='tab:green', alpha=0.3)
    ax3.set_xlabel('ego (deg)')

    _setmax = np.max([np.max(x) for x in [
        predP,
        predR,
        predE
    ]]) * 1.1

    for ax in [ax1,ax2,ax3]:
        ax.set_ylim([
            0,
            _setmax * 1.1
        ])

    fig.suptitle('predicted tuning curves')
    fig.tight_layout()

    return fig


def calc_bootstrap_model_params(data_vars, var_bins, spikes, n_iter=30):
    """ Calculate bootstrap model parameters for the LNP model.
    
    Parameters
    ----------
    data_vars : list of numpy.ndarray
        List of data variables (pupil, retino, ego).
    var_bins : list of numpy.ndarray
        List of bin edges for each data variable (pupil, retino, ego).
    spikes : numpy.ndarray
        Spike counts for each trial.
    n_iter : int, optional
        Number of bootstrap iterations. Default is 30.

    Returns
    -------
    bootstrap_model_params : dict
        Dictionary containing the mean and standard error of the model parameters.
    """

    mk = 'PRED'
        
    pupil_data, ret_data, ego_data = data_vars

    pupil_bins, retino_bins, ego_bins = var_bins

    param_counts = [
        len(pupil_bins),
        len(retino_bins),
        len(ego_bins)
    ]

    mapP = fm2p.make_varmap(pupil_data, pupil_bins)
    mapR = fm2p.make_varmap(ret_data, retino_bins, circ=True)
    mapE = fm2p.make_varmap(ego_data, ego_bins, circ=True)

    A = np.concatenate([mapP, mapR, mapE], axis=1)
    A = A[np.sum(np.isnan(A), axis=1)==0, :]

    shufP = np.zeros([n_iter, len(pupil_bins)]) * np.nan
    shufR = np.zeros([n_iter, len(retino_bins)]) * np.nan
    shufE = np.zeros([n_iter, len(ego_bins)]) * np.nan

    print('Running bootstrap...')
    for it in tqdm(range(n_iter)):

        # Shuffle the order of the data for each iteration
        shuf_inds = np.random.choice(np.arange(np.size(A,0)), np.size(A,0), replace=False)
        
        A_shuf = A.copy()
        A_shuf = A_shuf[shuf_inds,:]

        spikes_shuf = spikes.copy()
        spikes_shuf = spikes_shuf[shuf_inds]

        # Fit the model to the shuffled data
        _, _, param_mean, param_stderr, _, _ = fm2p.fit_LNLP_model(
            A_shuf, 0.05, spikes_shuf, np.ones(1), mk, param_counts
        )
        
        # Calculate the tuning curves for the shuffled data
        predP, predR, predE, predD = fm2p.calc_scaled_LNLP_tuning_curves(
            params=param_mean,
            param_stderr=param_stderr,
            ret_stderr=False
        )

        shufP[it,:] = predP
        shufR[it,:] = predR
        shufE[it,:] = predE


    bootstrap_model_params = {}
    mkk = ['P', 'R', 'E', 'D']
    for p_i, p in enumerate([shufP, shufR, shufE]):

        # Calculate the mean and standard error of the model parameters. The standard error
        # does not need to be divided by the square root of the number of iterations because
        # it's calculated across bootstrapped values.
        mean_param = np.nanmean(p, axis=0)
        stderr_param = np.nanstd(p, axis=0) # / np.sqrt(n_iter)

        bootstrap_model_params[mkk[p_i]] = {
            'mean': mean_param,
            'stderr': stderr_param
        }

    return bootstrap_model_params


def get_cells_best_LLHs(model_data):
    """ Get the best log likelihood for each cell in the model data.
    
    Parameters
    ----------
    model_data : dict
        Dictionary containing model results for each model type.

    Returns
    -------
    all_best_LLHs : numpy.ndarray
        Array containing the best log likelihood for each cell.
    """

    # Get total number of cells (any model key is fine)
    num_cells = len(model_data['P'].keys())

    all_best_LLHs = np.zeros(num_cells) * np.nan

    for c in range(num_cells):

        # Get evaluation results
        eval_results = fm2p.eval_models(model_data, c)
        
        cstr = str(c)

        if eval_results is None:
            continue

        # Get the best model
        best_model = eval_results['best_model']
        
        # A few cells occasionally have NaN for best model because some k-folds have
        # NaN values for the fit. Could hunt this problem down later, but could be a
        # problem w/ the spike rates
        if (type(best_model) != str) and (np.isnan(best_model)):
            continue

        # Get the log likelihood for the best performing model
        best_LLH = model_data[best_model][cstr]['testFit'][:,2]

        # Calculate the average LLH across k-folds
        best_LLH = np.nanmean(best_LLH)

        all_best_LLHs[c] = best_LLH

    return all_best_LLHs


def determine_responsiveness_from_null(model_path, null_path, null_thresh=0.99):
    """ Determine responsiveness of cells based on log likelihood threshold from null model.

    Parameters
    ----------
    model_path : str or dict
        Path to the model data or the model data itself.
    null_path : str or dict
        Path to the null model data or the null model data itself.
    null_thresh : float, optional
        Threshold for the null model. Default is 0.99.

    Returns
    -------
    LLH_thresh : float
        Log likelihood threshold for responsiveness.
    fig : matplotlib.figure.Figure
        Figure object containing the plot of the log likelihood distributions.
    """

    # Read the data in from path
    if type(model_path)==str:
        model_data = fm2p.read_models(model_path)
        null_model_data = fm2p.read_models(null_path)
    else:
        model_data = model_path
        null_model_data = null_path

    # Get the best log likelihood for every cell
    model_LLHs = get_cells_best_LLHs(model_data)
    null_LLHs = get_cells_best_LLHs(null_model_data)

    use_bins = np.linspace(-0.2,0.3,30)
    show_bins = np.linspace(-0.2,0.3,29)

    hist1, _ = np.histogram(model_LLHs, bins=use_bins)
    hist2, _ = np.histogram(null_LLHs, bins=use_bins)

    # Determine LLH threshold for repsonsiveness based on the performance of the null distribution.
    # This is calculated from the cumulative sum of cells at binned LLH values in the shuffled data,
    # and a threshold is applied to determine the LLH value at which 99% time-shuffled cells (or
    # whatever is set as `null_thresh`) fail to meet criteria. This threshold can then be used to
    # filter the real data.
    LLH_thresh = show_bins[int(np.argwhere(np.cumsum(hist2/np.sum(hist2)) >= null_thresh)[0])]

    # Diagnostic plot of the two LLH distributions
    plot_max = np.nanmax(np.concatenate([hist1.copy(), hist2.copy()])) * 1.1

    fig, ax = plt.subplots(1,1, dpi=300, figsize=(2.5, 2))
    ax.plot(show_bins, hist2, color='k', label='shifted spikes')
    ax.plot(show_bins, hist1, color='tab:blue', label='data')
    ax.vlines(0, 0, plot_max, lw=0.75, ls='--', color='k', alpha=0.5)
    ax.vlines(LLH_thresh, 0, 210, color='tab:red', ls='--')
    ax.set_ylim([0, plot_max])
    ax.set_xlim([-0.2, 0.3])
    fig.tight_layout()

    return LLH_thresh, fig


def get_responsive_inds(model_data, LLH_threshold):
    """ Get indices of responsive cells based on log likelihood threshold.
    
    Parameters
    ----------
    model_data : dict
        Dictionary containing model results for each model type.
    LLH_threshold : float
        Threshold for log likelihood to determine responsiveness.

    Returns
    -------
    responsive_inds : numpy.ndarray
        Array containing indices of responsive cells.
    """
    
    model_LLHs = get_cells_best_LLHs(model_data)
    responsive_inds = np.where(model_LLHs>=LLH_threshold)[0]

    return responsive_inds


def get_responsive_inds_2(model1_data, model2_data, LLH_threshold, thresh2=None):
    """ Get indices of responsive cells based on log likelihood threshold for two models.

    Parameters
    ----------
    model1_data : dict
        Dictionary containing model results for the first model type.
    model2_data : dict
        Dictionary containing model results for the second model type.
    LLH_threshold : float
        Threshold for log likelihood to determine responsiveness for the first model.
    thresh2 : float, optional
        Threshold for log likelihood to determine responsiveness for the second model. Default is None.
        If None, uses `LLH_threshold`.

    Returns
    -------
    responsive_inds : numpy.ndarray
        Array containing indices of responsive cells for both models.
    """

    if thresh2 is None:
        thresh2 = LLH_threshold

    p1_responsive_inds = get_responsive_inds(model1_data, LLH_threshold)
    p2_responsive_inds = get_responsive_inds(model2_data, thresh2)

    responsive_inds = np.array([c for c in p1_responsive_inds if c in p2_responsive_inds])

    return responsive_inds


