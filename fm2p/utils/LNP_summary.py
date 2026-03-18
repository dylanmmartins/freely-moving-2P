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

from .LNP_eval import (
    determine_responsiveness_from_null,
    get_responsive_inds,
    calc_scaled_LNLP_tuning_curves,
    plot_model_LLHs,
    eval_models,
    plot_rank_test_results,
)
from .LNP_model import fit_LNLP_model


def tuning_curve(sps, x, x_range):
    """ Calculate tuning curve of neurons to a 1D variable. """
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
            if np.count_nonzero(usePts) > 0:
                tuning_err[n,j] = np.nanstd(scatter[n, usePts]) / np.sqrt(np.count_nonzero(usePts))
            else:
                tuning_err[n,j] = np.nan

    return var_cent, tuning, tuning_err


def plot_tuning(ax, var_cent, tuning, tuning_err, color, rad=True):
    """ Plot tuning curve. rad=True converts x-axis to degrees. """
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
    """ Write a detailed cell summary for models [A, B, C, D]. """
    
    # 1. Unpack bins (Expect 4 bins now)
    binsA, binsB, binsC, binsD = var_bins

    # 2. Prepare Variables (Map P,R,E,D to A,B,C,D)
    # A, B, C are angles (convert to rad), D is distance (keep linear)
    varA = np.deg2rad(preprocdata['pupil_from_head'].copy()) # A
    varB = np.deg2rad(preprocdata['retinocentric'].copy())   # B
    varC = np.deg2rad(preprocdata['egocentric'].copy())      # C
    varD = preprocdata['dist_to_center'].copy()              # D

    # 3. Process Speed and Spikes
    speed = preprocdata['speed']
    # Fix speed length if necessary (handling diff mismatch)
    if len(speed) < len(varA):
        speed = np.append(speed, speed[-1])
    use = speed > 1.5

    raw_spikes = preprocdata['oasis_spks'].copy()
    spikes = np.zeros_like(raw_spikes) * np.nan
    for i in range(np.size(raw_spikes,0)):
        spikes[i,:] = np.roll(raw_spikes[i,:], shift=lag_val)

    # 4. Determine Responsive Cells
    if (responsive_inds is None) and (null_data is not None):
        responsive_thresh, _ = determine_responsiveness_from_null(model_data, null_data)
    elif (responsive_inds is None) and (null_data is None):
        responsive_inds = get_responsive_inds(model_data, LLH_threshold=0.2)

    if (responsive_inds is None) and ('responsive_thresh' in locals()) and (responsive_thresh is not None):
        responsive_inds = get_responsive_inds(model_data, LLH_threshold=responsive_thresh)
    
    # Validation: Split data into chunks
    ncnk = 10
    _len = np.sum(use)
    cnk_sz = _len // ncnk
    _all_inds = np.arange(0,_len)
    chunk_order = np.arange(ncnk)
    np.random.shuffle(chunk_order)

    split1_inds = []
    split2_inds = []

    for cnk in chunk_order[:(ncnk//2)]:
        _inds = _all_inds[(cnk_sz*cnk) : ((cnk_sz*(cnk+1)))]
        split1_inds.extend(_inds)
    for cnk in chunk_order[(ncnk//2):]:
        _inds = _all_inds[(cnk_sz*cnk) : ((cnk_sz*(cnk+1)))]
        split2_inds.extend(_inds)

    split1_inds = np.array(np.sort(split1_inds))
    split2_inds = np.array(np.sort(split2_inds))

    # Setup PDF
    pdf = PdfPages(savepath)

    # --- Main Loop Over Cells ---
    for c_i in tqdm(responsive_inds):
        c_s = str(c_i)

        # -- Calculate Tuning Curves for A, B, C, D --
        # Store in dictionaries or lists to iterate easily
        # structure: [Full, Split1, Split2]
        tunings = {} 
        
        # Define vars and properties
        # Format: (Data, Bins, IsRadians)
        vars_list = [
            (varA, binsA, True),  # A
            (varB, binsB, True),  # B
            (varC, binsC, True),  # C
            (varD, binsD, False)  # D
        ]

        # Compute tuning for each variable
        for v_i, (data, bins, is_rad) in enumerate(vars_list):
            # Full data
            c, t, e = tuning_curve(spikes[c_i,use][np.newaxis,:], data[use], bins)
            # Split 1
            c2, t2, e2 = tuning_curve(spikes[c_i,use][np.newaxis,split1_inds], data[use][split1_inds], bins)
            # Split 2
            c3, t3, e3 = tuning_curve(spikes[c_i,use][np.newaxis,split2_inds], data[use][split2_inds], bins)
            
            tunings[v_i] = {
                'full': (c, t, e),
                's1': (c2, t2, e2),
                's2': (c3, t3, e3),
                'is_rad': is_rad
            }

        # Extra: Speed tuning (kept separate as in original)
        speed_bins = np.linspace(0,10,7)
        speed_cent, speed_tuning, speed_err = tuning_curve(
            spikes[c_i,use][np.newaxis,:], speed[use], speed_bins)

        # -- Setup Figure --
        fig = plt.figure(constrained_layout=False, figsize=(16, 10), dpi=300) # Widened figsize
        spec = gridspec.GridSpec(ncols=2, nrows=6, figure=fig, width_ratios=[3, 1])

        # Grid logic: The left column (spec[:,0]) is now split into 4 sub-columns for A, B, C, D
        row0 = spec[0,0].subgridspec(1, 4, wspace=0.35) # Tuning Curves (Blue)
        row1 = spec[1,0].subgridspec(1, 4, wspace=0.35) # Split Tuning (Red/Purple)
        row2 = spec[2,0].subgridspec(1, 4, wspace=0.35) # Scaled Model Curves
        row3 = spec[3,0].subgridspec(1, 4, wspace=0.35) # Parameters
        
        # These remain wider or different
        row4 = spec[4,0].subgridspec(1, 2, wspace=0.35) # Speed / LLH
        row5 = spec[5,0].subgridspec(1, 2, wspace=0.35) # Signed Rank

        col2 = spec[:,1].subgridspec(7, 1) # Predictions column

        # -- Plotting Tuning Curves --
        # Colors for A, B, C, D
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown']
        labels = ['A (pupil)', 'B (ret)', 'C (ego)', 'D (dist)']

        # Max Y for scaling
        _set_max = 0
        
        # Loop over A, B, C, D to plot Row 0 (Full) and Row 1 (Splits)
        for v_i in range(4):
            # Row 0: Full Tuning
            ax0 = fig.add_subplot(row0[0, v_i])
            cent, t, err = tunings[v_i]['full']
            plot_tuning(ax0, cent, t, err, colors[v_i], rad=tunings[v_i]['is_rad'])
            ax0.set_title(labels[v_i])
            if np.max(t) > _set_max: _set_max = np.max(t)

            # Row 1: Split Tuning
            ax1 = fig.add_subplot(row1[0, v_i])
            c2, t2, e2 = tunings[v_i]['s1']
            c3, t3, e3 = tunings[v_i]['s2']
            plot_tuning(ax1, c2, t2, e2, 'tab:red', rad=tunings[v_i]['is_rad'])
            plot_tuning(ax1, c3, t3, e3, 'tab:purple', rad=tunings[v_i]['is_rad'])
            
            if np.max(t2) > _set_max: _set_max = np.max(t2)
            if np.max(t3) > _set_max: _set_max = np.max(t3)

            # Row 2 (repurposed for Scaled Curves placeholders or additional vars)
            # In original code, Row 2 was "tuning to other variables" (Speed/Dist)
            # We moved Dist to D. So Row 2 can be empty or used for something else.
            # Let's put Speed in Row 4 for now and leave Row 2 blank or for params.
            
            # Formatting Axes
            for ax in [ax0, ax1]:
                if tunings[v_i]['is_rad']:
                    ax.set_xlim([-180, 180])
                # else: (Distance) auto-scale or set fixed range
            
            if v_i == 0:
                ax0.set_ylabel('sp/s')
                ax1.set_ylabel('sp/s')

        # Apply Y-Limit
        for row in [row0, row1]:
             for v_i in range(4):
                 ax = fig.add_subplot(row[0, v_i])
                 ax.set_ylim([0, _set_max])

        # -- Row 3: Scaled LNLP Tuning Curves (Model Preds) --
        # IMPORTANT: Updated fm2p call to expect 4 returns
        (predA, errA), (predB, errB), (predC, errC), (predD, errD) = calc_scaled_LNLP_tuning_curves(
                model_data, c_s, ret_stderr=True, params=None, param_stderr=None)
        
        # Prepare axes list for the plot function
        p_axs = [fig.add_subplot(row3[0, i]) for i in range(4)]
        
        # IMPORTANT: Updated fm2p plot call to pass 4 sets of data
        # Note: You must ensure fm2p.plot_scaled_LNLP_tuning_curves is updated to handle 4 args
        # or call a plotting function manually here. Assuming the former or manual plot:
        
        # Manual plot fallback for safety if fm2p isn't visible:
        for i, (p, e, b, rad_flag) in enumerate(zip(
            [predA, predB, predC, predD], 
            [errA, errB, errC, errD], 
            [binsA, binsB, binsC, binsD],
            [True, True, True, False]
        )):
            # Calc centers
            cents = 0.5 * (b[:-1] + b[1:])
            plot_tuning(p_axs[i], cents, p[np.newaxis,:], e[np.newaxis,:], 'k', rad=rad_flag)
            p_axs[i].set_ylim([0, _set_max]) # Match scale

        # -- Row 4: Speed Tuning & Scatter --
        t_spd = fig.add_subplot(row4[0,0])
        plot_tuning(t_spd, speed_cent, speed_tuning, speed_err, 'k', rad=False)
        t_spd.set_xlabel('Speed (cm/s)')

        scatter_ax = fig.add_subplot(row4[0,1])
        # Plot LLH Scatter (generic call)
        fig = plot_model_LLHs(model_data, c_s, test_only=True, fig=fig, ax=scatter_ax, tight_y_scale=True)
        scatter_ax.set_ylabel('LLH')

        # -- Row 5: Rank Tests --
        eval_results = eval_models(model_data, c_s)
        sr1 = fig.add_subplot(row5[0,0])
        sr2 = fig.add_subplot(row5[0,1])
        fig = plot_rank_test_results(model_data, eval_results, c_s, fig=fig, axs=[sr1,sr2])

        # -- Column 2: Time Series Predictions --
        # We have 7 slots. We will plot Single models (4) + Full Model (1) + 2 Combinations
        plot_models = ['A', 'B', 'C', 'D', 'AB', 'CD', 'ABCD']
        
        _len = len(model_data['A'][c_s]['predSpikes']) # Changed 'P' to 'A'
        modelT = np.linspace(0, 0.05*_len, _len)

        for i, model in enumerate(plot_models):
            ax = fig.add_subplot(col2[i])
            
            if model in model_data and c_s in model_data[model]:
                ax.plot(modelT, model_data[model][c_s]['trueSpikes'], color='k', lw=1)
                ax.plot(modelT, model_data[model][c_s]['predSpikes'], color='tab:red', lw=1)
                
                # Dynamic title with LLH
                llh_val = np.nanmean(model_data[model][c_s]['testFit'][:,2])
                ax.set_title(f'Model {model} | LLH={llh_val:.3f}', fontsize=8)
                ax.set_xlim([0, 60])
                ax.set_ylim([0, np.max(model_data[model][c_s]['predSpikes'])])
                ax.axis('off') # Cleaner look
            else:
                ax.text(0.5, 0.5, f'Model {model} not found', ha='center')

        fig.suptitle(f'Cell {c_s}; Best={eval_results["best_model"]}')

        pdf.savefig(fig)
        plt.close()

    print('Closing PDF')
    pdf.close()