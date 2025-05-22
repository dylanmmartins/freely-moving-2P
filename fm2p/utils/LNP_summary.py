# -*- coding: utf-8 -*-
"""
Summary figures of the linear-nonlinear-Poisson model.

Functions
---------
tuning_curve(sps, x, x_range)
    Calculate tuning curve of neurons to a 1D variable.
plot_tuning(ax, var_cent, tuning, tuning_err, color, rad=True)
    Plot tuning curve of neurons to a 1D variable.
write_detailed_cell_summary(model_data, savepath, var_bins, preprocdata,
        null_data=None, responsive_inds=None, lag_val=0)
    Write a detailed cell summary of the model data.

Author: DMM, 2024
"""


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

import fm2p


def tuning_curve(sps, x, x_range):
    """ Calculate tuning curve of neurons to a 1D variable.

    Parameters
    ----------
    sps : np.array
        Spike data. Shape should be (n_cells, n_timepoints).
    x : np.array
        Variable data. Shape should be (n_cells, n_timepoints). The
        timepoints should match those for `sps`, either by interpolation
        or by binning.
    x_range : np.array
        Array of values to bin x into.
    
    Returns
    -------
    var_cent : np.array
        Array of values at the center of each bin. Shape is (n_bins,)
    tuning : np.array
        Array of mean spike counts for each bin. Shape is (n_cells, n_bins).
    tuning_err : np.array
        Array of standard error of the mean spike counts for each bin. Shape
        is (n_cells, n_bins).
    """

    n_cells = np.size(sps,0)

    scatter = np.zeros((n_cells, np.size(x,0)))

    tuning = np.zeros((n_cells, len(x_range)-1))
    tuning_err = tuning.copy()
    var_cent = np.zeros(len(x_range)-1)
    
    for j in range(len(x_range)-1):
        
        var_cent[j] = 0.5*(x_range[j] + x_range[j+1])
    
    for n in range(n_cells):
        
        scatter[n,:] = sps[n,:]
        
        for j in range(len(x_range)-1):
            
            usePts = (x>=x_range[j]) & (x<x_range[j+1])
            
            tuning[n,j] = np.nanmean(scatter[n, usePts])
            
            tuning_err[n,j] = np.nanstd(scatter[n, usePts]) / np.sqrt(np.count_nonzero(usePts))

    return var_cent, tuning, tuning_err


def plot_tuning(ax, var_cent, tuning, tuning_err, color, rad=True):
    """ Plot tuning curve of neurons to a 1D variable.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    var_cent : np.array
        Array of values at the center of each bin. Shape is (n_bins,)
    tuning : np.array
        Array of mean spike counts for each bin. Shape is (n_cells, n_bins).
    tuning_err : np.array
        Array of standard error of the mean spike counts for each bin. Shape
        is (n_cells, n_bins).
    color : str
        Color to use for the plot.
    rad : bool
        If True, convert var_cent from radians to degrees. Default is True.
    """
    
    if rad:
        usebins = np.rad2deg(var_cent)
    else:
        usebins = var_cent.copy()

    ax.plot(usebins, tuning[0], color=color)

    ax.fill_between(
        usebins,
        tuning[0]+tuning_err[0],
        tuning[0]-tuning_err[0],
        alpha=0.3, color=color
    )


def write_detailed_cell_summary(model_data, savepath, var_bins, preprocdata,
                       null_data=None, responsive_inds=None, lag_val=0):
    """ Write a detailed cell summary of the model data.

    Parameters
    ----------
    model_data : dict
        Dictionary containing the model data.
    savepath : str
        Path to save the summary PDF.
    var_bins : list
        List of bins for the variables to plot. Each element should be a
        numpy array of bin edges.
    preprocdata : dict
        Dictionary containing the preprocessed data. Should contain the
        following keys:
            - 'pupil_from_head': pupil position in degrees.
            - 'egocentric': egocentric position in degrees.
            - 'retinocentric': retinocentric position in degrees.
            - 'dist_to_center': distance to center in cm.
            - 'speed': running speed in cm/s.
            - 'oasis_spks': spike data.
    null_data : dict, optional
        Dictionary containing the null data. If None, the function will
        use the responsive threshold from the model data.
    responsive_inds : np.array, optional
        Array of indices of responsive cells. If None, the function will
        use the responsive threshold from the model data.
    lag_val : int, optional
        Lag value to use for the spike data. Default is 0.
    """
    
    pupil_bins, retino_bins, ego_bins = var_bins

    pupil = np.deg2rad(preprocdata['pupil_from_head'].copy())
    ego = np.deg2rad(preprocdata['egocentric'].copy())
    ret = np.deg2rad(preprocdata['retinocentric'].copy())
    dist = preprocdata['dist_to_center'].copy()

    speed = preprocdata['speed']
    speed = np.append(speed, speed[-1])
    use = speed > 1.5

    raw_spikes = preprocdata['oasis_spks'].copy()
    spikes = np.zeros_like(raw_spikes) * np.nan
    for i in range(np.size(raw_spikes,0)):
        spikes[i,:] = np.roll(raw_spikes[i,:], shift=lag_val)

    if (responsive_inds is None) and (null_data is not None):
        responsive_thresh, _ = fm2p.determine_responsiveness_from_null(model_data, null_data)

    elif (responsive_inds is None) and (null_data is None):
        responsive_inds = fm2p.get_responsive_inds(model_data, LLH_threshold=0.2)

    # If responsive inds still has not been defined, define it using the calculated
    # responsive threshold.
    if (responsive_inds is None) and (responsive_thresh is not None):
        responsive_inds = fm2p.get_responsive_inds(model_data, LLH_threshold=responsive_thresh)
    
    print('Using LLH threshold of {}'.format(responsive_thresh))

    pdf = PdfPages(savepath)

    # Break data into 10 chunks, randomly choose ten of them for each block.
    ncnk = 10
    _len = np.sum(use)
    cnk_sz = _len // ncnk

    _all_inds = np.arange(0,_len)

    chunk_order = np.arange(ncnk)
    np.random.shuffle(chunk_order)

    split1_inds = []
    split2_inds = []

    for cnk_i, cnk in enumerate(chunk_order[:(ncnk//2)]):
        _inds = _all_inds[(cnk_sz*cnk) : ((cnk_sz*(cnk+1)))]
        split1_inds.extend(_inds)

    for cnk_i, cnk in enumerate(chunk_order[(ncnk//2):]):
        _inds = _all_inds[(cnk_sz*cnk) : ((cnk_sz*(cnk+1)))]
        split2_inds.extend(_inds)

    split1_inds = np.array(np.sort(split1_inds))
    split2_inds = np.array(np.sort(split2_inds))

    for c_i in tqdm(responsive_inds):

        c_s = str(c_i)

        pupil_cent, pupil_tuning, pupil_err = tuning_curve(
            spikes[c_i,use][np.newaxis,:], pupil[use], pupil_bins)
        pupil2_cent, pupil2_tuning, pupil2_err = tuning_curve(
            spikes[c_i,use][np.newaxis,split1_inds], pupil[use][split1_inds], pupil_bins)
        pupil3_cent, pupil3_tuning, pupil3_err = tuning_curve(
            spikes[c_i,use][np.newaxis,split2_inds], pupil[use][split2_inds], pupil_bins)

        ret1_cent, ret1_tuning, ret1_err = tuning_curve(
            spikes[c_i,use][np.newaxis,:], ret[use], retino_bins)
        ret2_cent, ret2_tuning, ret2_err = tuning_curve(
            spikes[c_i,use][np.newaxis,split1_inds], ret[use][split1_inds], retino_bins)
        ret3_cent, ret3_tuning, ret3_err = tuning_curve(
            spikes[c_i,use][np.newaxis,split2_inds], ret[use][split2_inds], retino_bins)

        ego1_cent, ego1_tuning, ego1_err = tuning_curve(
            spikes[c_i,use][np.newaxis,:], ego[use], ego_bins)
        ego2_cent, ego2_tuning, ego2_err = tuning_curve(
            spikes[c_i,use][np.newaxis,split1_inds], ego[use][split1_inds], ego_bins)
        ego3_cent, ego3_tuning, ego3_err = tuning_curve(
            spikes[c_i,use][np.newaxis,split2_inds], ego[use][split2_inds], ego_bins)
        
        # running speed
        speed_bins = np.linspace(0,10,7)
        speed_cent, speed_tuning, speed_err = tuning_curve(
            spikes[c_i,use][np.newaxis,:], speed[use], speed_bins)
        
        # distance
        dist_bins = np.linspace(2.5,21.5,12)
        dist_cent, dist_tuning, dist_err = tuning_curve(
            spikes[c_i,use][np.newaxis,:], dist[use], dist_bins)

        fig = plt.figure(constrained_layout=False, figsize=(12,10), dpi=300)
        spec = gridspec.GridSpec(ncols=2, nrows=6, figure=fig)

        row0 = spec[0,0].subgridspec(1,3, wspace=0.35) # tuning curves
        row1 = spec[1,0].subgridspec(1,3, wspace=0.35) # split tuning curves
        row2 = spec[2,0].subgridspec(1,3, wspace=0.35) # tuning to other variables
        row3 = spec[3,0].subgridspec(1,3, wspace=0.35) # parameters
        row4 = spec[4,0].subgridspec(1,1, wspace=0.35) # LLH
        row5 = spec[5,0].subgridspec(1,2, wspace=0.35) # signed-rank

        # for row in [row0,row1,row2,row3,row4,row5]:
        #     row.update(wspace=0.5)

        col2 = spec[:,1].subgridspec(7,1)

        t1 = fig.add_subplot(row0[0,0])
        t2 = fig.add_subplot(row0[0,1])
        t3 = fig.add_subplot(row0[0,2])

        t4 = fig.add_subplot(row1[0,0])
        t5 = fig.add_subplot(row1[0,1])
        t6 = fig.add_subplot(row1[0,2])

        t7 = fig.add_subplot(row2[0,0])
        t8 = fig.add_subplot(row2[0,1])
        t9 = fig.add_subplot(row2[0,2])

        p1 = fig.add_subplot(row3[0,0])
        p2 = fig.add_subplot(row3[0,1])
        p3 = fig.add_subplot(row3[0,2])

        scatter = fig.add_subplot(row4[:,:])
        sr1 = fig.add_subplot(row5[0,0])
        sr2 = fig.add_subplot(row5[0,1])

        pred1 = fig.add_subplot(col2[0])
        pred2 = fig.add_subplot(col2[1])
        pred3 = fig.add_subplot(col2[2])
        pred4 = fig.add_subplot(col2[3])
        pred5 = fig.add_subplot(col2[4])
        pred6 = fig.add_subplot(col2[5])
        pred7 = fig.add_subplot(col2[6])

        plot_tuning(t1, pupil_cent, pupil_tuning, pupil_err, 'tab:blue')
        plot_tuning(t4, pupil2_cent, pupil2_tuning, pupil2_err, 'tab:red')
        plot_tuning(t4, pupil3_cent, pupil3_tuning, pupil3_err, 'tab:purple')

        plot_tuning(t2, ret1_cent, ret1_tuning, ret1_err, 'tab:orange')
        plot_tuning(t5, ret2_cent, ret2_tuning, ret2_err, 'tab:red')
        plot_tuning(t5, ret3_cent, ret3_tuning, ret3_err, 'tab:purple')

        plot_tuning(t3, ego1_cent, ego1_tuning, ego1_err, 'tab:green')
        plot_tuning(t6, ego2_cent, ego2_tuning, ego2_err, 'tab:red')
        plot_tuning(t6, ego3_cent, ego3_tuning, ego3_err, 'tab:purple')

        plot_tuning(t7, speed_cent, speed_tuning, speed_err, 'k', rad=False)
        plot_tuning(t8, dist_cent, dist_tuning, dist_err, 'k', rad=False)

        _set_max = 0
        for tuning in [pupil_tuning, pupil2_tuning, pupil3_tuning,
                    ret1_tuning, ret2_tuning, ret3_tuning,
                    ego1_tuning, ego2_tuning, ego3_tuning]:
            if np.max(tuning) > _set_max:
                _set_max = np.max(tuning)

        for ax in [t1,t2,t3,t4,t5,t6]:
            ax.set_ylim([0, _set_max])

        t1.set_xlabel('pupil tuning (deg)')
        t2.set_xlabel('retino tuning (deg)')
        t3.set_xlabel('ego tuning (deg)')
        t4.set_xlabel('pupil tuning (deg)')
        t5.set_xlabel('retino tuning (deg)')
        t6.set_xlabel('ego tuning (deg)')
        t7.set_xlabel('speed (cm/s)')
        t8.axis('off')
        t9.axis('off')

        for ax in [t2,t3,t5,t6,p2,p3]:
            ax.set_xlim([-180,180])
        for ax in [t1,t4,p1]:
            ax.set_xlim([0,100])
        for ax in [t1,t2,t3,t4,t5,t6,p1,p2,p3,t7]:
            ax.set_ylabel('sp/s')

        (predP, stderrP), (predR, stderrR), (predE, stderrE) = fm2p.calc_scaled_LNLP_tuning_curves(
                model_data, c_s, ret_stderr=True, params=None, param_stderr=None)

        fig = fm2p.plot_scaled_LNLP_tuning_curves(
                predP, predR, predE,
                stderrP, stderrR, stderrE,
                pupil_bins, retino_bins, ego_bins,
                fig=fig, axs=[p1,p2,p3]
        )
        eval_results = fm2p.eval_models(model_data, c_s)

        _len = len(model_data['P'][c_s]['predSpikes'])
        modelT = np.linspace(0, 0.05*_len, _len)

        pred_figs = [pred1,pred2,pred3,pred4,pred5,pred6,pred7]
        for model_i, model in enumerate(['P','R','E','PR','PE','RE','PRE']):

            ax = pred_figs[model_i]

            ax.plot(modelT, model_data[model][c_s]['trueSpikes'], color='k')
            ax.plot(modelT, model_data[model][c_s]['predSpikes'], color='tab:red')
            ax.set_xlim([0, 60])
            ax.set_ylim([0, np.max(model_data[model][c_s]['predSpikes'])])
            ax.set_title('model={} LLH={:.3}'.format(model, np.nanmean(model_data[model][c_s]['testFit'][:,2])))

        fig = fm2p.plot_model_LLHs(model_data, c_s, test_only=True, fig=fig, ax=scatter, tight_y_scale=True)
        scatter.set_ylabel('LLH')

        fig = fm2p.plot_rank_test_results(model_data, eval_results, c_s, fig=fig, axs=[sr1,sr2])

        fig.suptitle('cell {}; bestModel={}'.format(c_s, eval_results['best_model']))

        pdf.savefig(fig)
        plt.close()

    print('Closing PDF')
    pdf.close()

