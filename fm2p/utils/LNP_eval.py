

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

import ret2ego


def add_scatter_col(ax, pos, vals):
    ax.scatter(
        np.ones_like(vals)*pos + (np.random.rand(len(vals))-0.5)/10,
        vals,
        s=2, c='k')
    ax.hlines(np.nanmean(vals), pos-.1, pos+.1, color='r')

    stderr = np.nanstd(vals) / np.sqrt(len(vals))
    ax.vlines(pos, np.nanmean(vals)-stderr, np.nanmean(vals)+stderr, color='r')


def read_models(models_dir):

    key_list = [
        'P','R','E',
        'PR','PE','RE',
        'PRE'
    ]

    model_data = {}
    for mk in key_list:
        model_data[mk] = ret2ego.read_h5(os.path.join(
            os.path.join(models_dir, 'model_{}_results.h5'.format(mk))
        ))
    
    return model_data


def plot_model_LLHs(model_data, unit_num, test_only=False, fig=None, ax=None, tight_y_scale=False):

    uk = str(unit_num)

    key_list = [
        'P','R','E',
        'PR','PE','RE',
        'PRE'
    ]

    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(1,1, dpi=300, figsize=(6,2))
    ax.hlines(0, -.5, len(key_list)+.5, color='k', linestyle='--', lw=1, alpha=0.3)
    for ki, mk in enumerate(key_list):

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

    model_ind = np.argmax([np.nanmean(model_data[mk][uk]['testFit'][:,2]) for mk in test_keys])
    
    return test_keys[model_ind]


def eval_models(model_data, unit_num, wilcoxon_thresh=0.05):

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

    maxkey = np.max([int(x) for x in dic.keys()])
    arr = np.zeros(maxkey)

    for i in range(maxkey):
        arr[i] = dic[str(i)]

    return arr



def plot_pred_spikes(model_data, unit_num, selected_models, fig=None, axs=None):

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

    uk = str(unit_num)

    mk = 'PRE'

    if params is None:
        params = model_data[mk][uk]['param_mean']

    tuningP, tuningR, tuningE = ret2ego.find_param(
        params,
        mk,
        10, 36, 36
    )
    
    # scale factor
    # dt = 0.05
    scale_factor_P = np.nanmean(np.exp(tuningR)) * np.nanmean(np.exp(tuningE))
    scale_factor_R = np.nanmean(np.exp(tuningP)) * np.nanmean(np.exp(tuningE))
    scale_factor_E = np.nanmean(np.exp(tuningP)) * np.nanmean(np.exp(tuningR))
    
    # compute model-derived response profile
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

            ki_tuningP, ki_tuningR, ki_tuningE = ret2ego.find_param(
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

    if (fig is None) and (axs is None):
        fig, [ax1,ax2,ax3] = plt.subplots(1,3, figsize=(5,1.8), dpi=300)
    else:
        [ax1,ax2,ax3] = axs

    ax1.plot(np.rad2deg(pupil_bins), predP, color='tab:blue')
    ax1.fill_between(np.rad2deg(pupil_bins), predP-errP, predP+errP, color='tab:blue', alpha=0.3)
    # ax1.errorbar(np.rad2deg(pupil_bins), y=predP, yerr=errP, lw=1, color='tab:blue')#, alpha=0.3)
    ax1.set_xlabel('pupil (deg)')

    ax2.plot(np.rad2deg(retino_bins), predR, color='tab:orange')
    ax2.fill_between(np.rad2deg(retino_bins), predR-errR, predR+errR, color='tab:orange', alpha=0.3)
    # ax2.errorbar(np.rad2deg(retino_bins), y=predR, yerr=errR, lw=1, color='tab:orange')#, alpha=0.3)
    ax2.set_xlabel('retino (deg)')

    ax3.plot(np.rad2deg(ego_bins), predE, color='tab:green')
    ax3.fill_between(np.rad2deg(ego_bins), predE-errE, predE+errE, color='tab:green', alpha=0.3)
    # ax3.errorbar(np.rad2deg(ego_bins), y=predE, yerr=errE, lw=1, color='tab:green')#, alpha=0.3)
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

    mk = 'PRED'
        
    pupil_data, ret_data, ego_data = data_vars

    pupil_bins, retino_bins, ego_bins = var_bins

    param_counts = [
        len(pupil_bins),
        len(retino_bins),
        len(ego_bins)
    ]

    mapP = ret2ego.make_varmap(pupil_data, pupil_bins)
    mapR = ret2ego.make_varmap(ret_data, retino_bins, circ=True)
    mapE = ret2ego.make_varmap(ego_data, ego_bins, circ=True)

    A = np.concatenate([mapP, mapR, mapE], axis=1)
    A = A[np.sum(np.isnan(A), axis=1)==0, :]

    shufP = np.zeros([n_iter, len(pupil_bins)]) * np.nan
    shufR = np.zeros([n_iter, len(retino_bins)]) * np.nan
    shufE = np.zeros([n_iter, len(ego_bins)]) * np.nan

    print('Running bootstrap...')
    for it in tqdm(range(n_iter)):

        # shuffle the order of the data for each iteration
        shuf_inds = np.random.choice(np.arange(np.size(A,0)), np.size(A,0), replace=False)
        
        A_shuf = A.copy()
        A_shuf = A_shuf[shuf_inds,:]

        spikes_shuf = spikes.copy()
        spikes_shuf = spikes_shuf[shuf_inds]

        _, _, param_mean, param_stderr, _, _ = ret2ego.fit_LNLP_model(
            A_shuf, 0.05, spikes_shuf, np.ones(1), mk, param_counts
        )
        
        predP, predR, predE, predD = ret2ego.calc_scaled_LNLP_tuning_curves(
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

        mean_param = np.nanmean(p, axis=0)
        stderr_param = np.nanstd(p, axis=0) # / np.sqrt(n_iter)

        bootstrap_model_params[mkk[p_i]] = {
            'mean': mean_param,
            'stderr': stderr_param
        }

    return bootstrap_model_params



def get_cells_best_LLHs(model_data):

    # Get total number of cells (any model key is fine)
    num_cells = len(model_data['P'].keys())

    all_best_LLHs = np.zeros(num_cells) * np.nan

    for c in range(num_cells):

        # Get evaluation results
        eval_results = ret2ego.eval_models(model_data, c)
        
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

    # Read the data in from path
    if type(model_path)==str:
        model_data = ret2ego.read_models(model_path)
        null_model_data = ret2ego.read_models(null_path)
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
    
    model_LLHs = get_cells_best_LLHs(model_data)
    responsive_inds = np.where(model_LLHs>=LLH_threshold)[0]

    return responsive_inds


def get_responsive_inds_2(model1_data, model2_data, LLH_threshold, thresh2=None):

    if thresh2 is None:
        thresh2 = LLH_threshold

    p1_responsive_inds = get_responsive_inds(model1_data, LLH_threshold)
    p2_responsive_inds = get_responsive_inds(model2_data, thresh2)

    responsive_inds = np.array([c for c in p1_responsive_inds if c in p2_responsive_inds])

    return responsive_inds


