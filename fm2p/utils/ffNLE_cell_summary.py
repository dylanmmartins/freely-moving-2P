
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, pathlib as _pl
if __package__ is None or __package__ == '':
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))
    __package__ = 'fm2p.utils'
import matplotlib as mpl
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 10
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

from .helper import interp_short_gaps
from .time import interpT
from .files import read_h5
from .paths import find
from .cmap import make_parula



# Set USE_RMSE = True to display importance as % increase in RMSE instead of
# % drop in R^2.  The conversion is done post-hoc from the saved R^2 importances
# so the full model does NOT need to be rerun.
USE_RMSE = False

# Set USE_SHUF_IDX = True to use calc_ablation_index (shuffle-normalised R^2 drop)
# instead of raw % drop in R^2.  Requires ffNLE to have been run with the updated
# compute_permutation_importance that saves {prefix}_ablation_index_{feat} arrays.
USE_SHUF_IDX = True



def get_shuf_index(y, h_hat):

    # calc r^2 of normal model
    r2_full = calculate_r2_numpy(y, h_hat)

    # shuffle y_hat and calc r^2 of shuffle against true
    n_shufs = 100
    r2_shufs = []
    for i in range(n_shufs):
        y_shuf = np.random.permutation(y)
        r2_shuf = calculate_r2_numpy(y_shuf, h_hat)
        r2_shufs.append(r2_shuf)

    mean_shuf = np.mean(r2_shufs)

    return r2_full, mean_shuf


def calc_ablation_index(y, y_hat, y_hat_partial):
        
    r2_full, r2_shuf_full = get_shuf_index(y, y_hat)
    r2_partial, r2_shuf_partial = get_shuf_index(y, y_hat_partial)
    full_signal = r2_full - r2_shuf_full
    partial_signal = r2_partial - r2_shuf_partial
    ablation_index = (full_signal - partial_signal) / (abs(full_signal) + 1e-8)

    return ablation_index


def make_earth_tones():

    colors = [
        '#2ECC71', '#82E0AA',
        '#FF9800', '#FFCC80',
        '#03A9F4', '#81D4FA',
        '#9C27B0', '#E1BEE7',
        '#FFEB3B', '#FFF59D'
    ]

    rgb_colors = [tuple(int(h.lstrip('#')[i:i+2], 16) / 255.0 for i in (0, 2, 4)) for h in colors]

    earth_map = LinearSegmentedColormap.from_list('earth_tones', rgb_colors, N=10)

    return earth_map

def get_equally_spaced_colormap_values(colormap_name, num_values):
    if not isinstance(num_values, int) or num_values <= 0:
        raise ValueError("num_values must be a positive integer.")
    if colormap_name == 'parula':
        cmap = make_parula()
    elif colormap_name == 'earth_tones':
        cmap = make_earth_tones()
    else:
        cmap = cm.get_cmap(colormap_name)
    normalized_positions = np.linspace(0, 1, num_values)
    colors = [cmap(pos) for pos in normalized_positions]
    return colors


def _plot_importance_row(ax_feat, ax_group, data, model_prefix, c, feature_names, colors,
                          nice_feature_names, group_keys, group_labels, hatch=None):

    r2_base = float(data.get(f'{model_prefix}_r2', data.get('full_r2', np.array([np.nan])))[c])

    if USE_SHUF_IDX:
        imp_prefix = f'{model_prefix}_ablation_index_'
        ylabel_str = 'Ablation Index'
    elif USE_RMSE:
        imp_prefix = f'{model_prefix}_importance_'
        ylabel_str = '% Drop in RMSE'
    else:
        imp_prefix = f'{model_prefix}_importance_'
        ylabel_str = '% Drop in R^2'

    importances = {}
    for k, v in data.items():
        if k.startswith(imp_prefix):
            feat_name = k[len(imp_prefix):]
            importances[feat_name] = v

    # Fall back to % drop R^2 if ablation index arrays are not present in this file
    if USE_SHUF_IDX and not importances:
        fallback_prefix = f'{model_prefix}_importance_'
        for k, v in data.items():
            if k.startswith(fallback_prefix):
                importances[k[len(fallback_prefix):]] = v
        ylabel_str = '% Drop in R^2 (fallback)'

    present = [f for f in feature_names if f in importances]
    feat_colors = [colors[feature_names.index(f)] for f in present]
    nice_present = [nice_feature_names[feature_names.index(f)] for f in present]
    values = [float(importances[feat][c]) for feat in present]

    bars = ax_feat.bar(nice_present, values, color=feat_colors,
                       hatch=hatch, edgecolor='k' if hatch else None, linewidth=0.5)
    heights = [bar.get_height() for bar in bars]
    if heights and max(heights) > 0:
        ax_feat.set_ylim([0, max(heights) * 1.1])
    for bar in bars:
        h = bar.get_height()
        if h <= 0:
            continue
        fmt = f'{h:.2f}' if USE_SHUF_IDX else f'{h:.1f}%'
        ax_feat.text(bar.get_x() + bar.get_width() / 2., h,
                     fmt, ha='center', va='bottom', fontsize=6)
    ax_feat.set_xticks(range(len(nice_present)), nice_present, rotation=90)
    ax_feat.set_ylabel(ylabel_str)

    group_vals_raw = []
    for gk in group_keys:
        if USE_SHUF_IDX:
            k = f'{model_prefix}_group_ablation_index_{gk}'
            if k not in data:
                k = f'{model_prefix}_group_importance_r2_{gk}'
        elif USE_RMSE:
            k = f'{model_prefix}_group_importance_rmse_{gk}'
        else:
            k = f'{model_prefix}_group_importance_r2_{gk}'
        if k in data:
            group_vals_raw.append(float(data[k][c]))
        else:
            group_vals_raw.append(0.0)

    # if USE_RMSE:
    #     group_vals = [float(v, r2_base) for v in group_vals_raw]
    # else:
    group_vals = group_vals_raw

    ax_group.bar(range(4), group_vals, color='black',
                 hatch=hatch, edgecolor='white' if hatch else None, linewidth=0.5)
    ax_group.set_xticks(range(4), group_labels, fontsize=7)
    ax_group.set_ylabel(ylabel_str)
    ax_group.set_title('group importance')
    ymax = max(group_vals) * 1.1 if group_vals and max(group_vals) > 0 else 1.0
    ymin = min(0, min(group_vals)) if group_vals else 0
    ax_group.set_ylim([ymin, ymax])
    for xi, v in enumerate(group_vals):
        if v > 0:
            fmt = f'{v:.2f}' if USE_SHUF_IDX else f'{v:.1f}%'
            ax_group.text(xi, v, fmt, ha='center', va='bottom', fontsize=6)

    return heights, group_vals



def make_cell_summary():

    # args = argparse.ArgumentParser()
    # args.add_argument('--dir', type='str', default=None)
    



    goodred = '#D96459'

    def calculate_r2_numpy(true, pred):
        true = np.array(true)
        pred = np.array(pred)
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        return 1 - (ss_res / ss_tot)

    basepath = '/home/dylan/Fast2/freely_moving_data/V1PPC/cohort03_recordings/260413_DMM_DMM065_pos13/fm2'
    pdata = read_h5(find('*_preproc.h5', basepath, MR=True))
    tdata = read_h5(os.path.join(basepath, 'eyehead_revcorrs_v06.h5'))
    data = read_h5(os.path.join(basepath, 'pytorchGLM_predictions_v09b.h5'))


    # Get behavior data
    eyeT = pdata['eyeT'][pdata['eyeT_startInd']:pdata['eyeT_endInd']]
    eyeT = eyeT - eyeT[0]
    twopT = pdata['twopT']

    if 'dPhi' not in pdata:
        phi_full = np.rad2deg(pdata['phi'][pdata['eyeT_startInd']:pdata['eyeT_endInd']])
        dPhi  = np.diff(interp_short_gaps(phi_full, 5)) / np.diff(eyeT)
        dPhi = np.roll(dPhi, -2)
        pdata['dPhi'] = dPhi

    if 'dTheta' not in pdata:
        if 'dEye' not in pdata:
            t = eyeT.copy()[:-1]
            t1 = t + (np.diff(eyeT) / 2)
            theta_full = np.rad2deg(pdata['theta'][pdata['eyeT_startInd']:pdata['eyeT_endInd']])
            dEye  = np.diff(interp_short_gaps(theta_full, 5)) / np.diff(eyeT)
            pdata['dTheta'] = np.roll(dEye, -2)
            pdata['eyeT1'] = t1
        else:
            pdata['dTheta'] = pdata['dEye'].copy()

    dTheta = interp_short_gaps(pdata['dTheta'])
    dTheta = interpT(dTheta, pdata['eyeT1'], twopT)
    dPhi = interp_short_gaps(pdata['dPhi'])
    dPhi = interpT(dPhi, pdata['eyeT1'], twopT)

    behavior_vars = {
        'theta': pdata['theta_interp'],
        'phi': pdata['phi_interp'],
        'dTheta': dTheta,
        'dPhi': dPhi,
        'pitch': pdata['pitch_twop_interp'],
        'roll': pdata['roll_twop_interp'],
        'yaw': pdata['head_yaw_deg'],
        'gyro_x': pdata['gyro_x_twop_interp'],
        'gyro_y': pdata['gyro_y_twop_interp'],
        'gyro_z': pdata['gyro_z_twop_interp'],
    }

    feature_names_hist = [
        'theta', 'phi', 'dTheta', 'dPhi',
        'pitch', 'roll', 'yaw', 'gyro_x', 'gyro_y', 'gyro_z'
    ]
    colors_hist = get_equally_spaced_colormap_values('earth_tones', len(feature_names_hist))

    _FPS = 60.0

    def _save_occupancy_fig(mask, label):
        speed = pdata.get('speed', np.ones(len(mask), dtype=float))
        n = min(len(mask), len(speed))
        total_min   = mask[:n].sum() / _FPS / 60
        moving_min  = (mask[:n] & (speed[:n] > 2.)).sum() / _FPS / 60

        fig, axs = plt.subplots(2, 5, figsize=(7, 3.5), dpi=300)
        axs = axs.flatten()
        for i, var_name in enumerate(feature_names_hist):
            ax = axs[i]
            if var_name in behavior_vars:
                var_data = behavior_vars[var_name]
                n_v = min(len(mask), len(var_data))
                masked = var_data[:n_v][mask[:n_v]]
                ax.hist(masked[~np.isnan(masked)], bins=100, density=True, color=colors_hist[i])
                ax.set_title(var_name)
            else:
                ax.axis('off')
            if var_name in ['dTheta', 'dPhi', 'gyro_x', 'gyro_y', 'gyro_z']:
                ax.set_xlim([-100, 100])
        fig.suptitle(f'occupancy ({label})  —  {total_min:.1f} min total, {moving_min:.1f} min moving (>2 cm/s)')
        fig.tight_layout()
        fname = f'model_results_occupancy_{label.lower()}.png'
        savename = os.path.join(os.path.split(basepath)[0], fname)
        print('saving {}'.format(savename))
        fig.savefig(savename)
        plt.close(fig)

    if 'ltdk_state_vec' in pdata:
        state = pdata['ltdk_state_vec'].astype(bool)
        _save_occupancy_fig(state,  'light')
        _save_occupancy_fig(~state, 'dark')
    else:
        all_mask = np.ones(len(twopT), dtype=bool)
        _save_occupancy_fig(all_mask, 'all')



    for c in np.argsort(data['full_r2'])[::-1][:20]:

        fig = plt.figure(figsize=(8.5, 11), constrained_layout=True, dpi=300)
        gs = fig.add_gridspec(nrows=5, ncols=3)

        light_y_true = data.get('full_trainLight_testLight_y_true', data.get('full_y_true'))
        light_y_hat  = data.get('full_trainLight_testLight_y_hat',  data.get('full_y_hat'))
        t = np.linspace(0, len(light_y_true[:,c])*(1/7.5), len(light_y_true[:,c])) / 60

        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(t, light_y_true[:,c], color='k', lw=1, label='$y$')
        ax1.plot(t, light_y_hat[:,c], color=goodred, lw=1, label='$\hat{y}$')
        ax1.set_xlim([0, np.max(t)])
        ax1.set_xlabel('time (min)')
        ax1.legend(fontsize=6, loc='upper left')
        ax1.set_ylabel('z-scored spike rate (spk/s)')

        feature_names = [
            'theta', 'dTheta', 'phi', 'dPhi',
            'pitch', 'gyro_y', 'roll', 'gyro_x', 'yaw', 'gyro_z'
        ]
        colors = get_equally_spaced_colormap_values('earth_tones', len(feature_names))
        swaps = [('gyro_x', 'dRoll'), ('gyro_y', 'dPitch'), ('gyro_z', 'dYaw')]
        nice_feature_names = list(feature_names)
        for swap in swaps:
            nice_feature_names = [swap[1] if x == swap[0] else x for x in nice_feature_names]
        group_keys   = ['position', 'velocity', 'eyes', 'head']
        group_labels = ['position\nonly', 'velocity\nonly', 'eyes\nonly', 'head\nonly']

        ax2 = fig.add_subplot(gs[1, :2])
        ax3 = fig.add_subplot(gs[1, 2])
        ax2.set_title('all inputs  (light)')
        _plot_importance_row(ax2, ax3, data, 'full_trainLight_testLight', c,
                            feature_names, colors, nice_feature_names,
                            group_keys, group_labels, hatch=None)

        ax_dark_trace = fig.add_subplot(gs[2, :])
        if 'full_trainDark_testDark_y_true' in data and 'full_trainDark_testDark_y_hat' in data:
            yt_d = data['full_trainDark_testDark_y_true'][:, c]
            yh_d = data['full_trainDark_testDark_y_hat'][:, c]
            t_d  = np.linspace(0, len(yt_d) * (1 / 7.5), len(yt_d)) / 60
            ax_dark_trace.plot(t_d, yt_d, color='k',      lw=1, label='$y$')
            ax_dark_trace.plot(t_d, yh_d, color=goodred,  lw=1, label='$\hat{y}$')
            ax_dark_trace.set_xlim([0, np.max(t_d)])
            ax_dark_trace.legend(fontsize=6, loc='upper left')
        ax_dark_trace.set_xlabel('time (min)')
        ax_dark_trace.set_ylabel('z-scored spike rate (spk/s)')

        dark_r2_val = np.nan
        if 'full_trainDark_testDark_r2' in data and c < len(data.get('full_trainDark_testDark_r2', [])):
            dark_r2_val = data['full_trainDark_testDark_r2'][c]
        if not np.isnan(dark_r2_val):
            ax_dark_trace.set_title(f'dark condition (R^2={dark_r2_val:.3f})')
        else:
            ax_dark_trace.set_title('dark condition  ($\hat{y}$ vs $y$)')

        ax2d = fig.add_subplot(gs[3, :2])
        ax3d = fig.add_subplot(gs[3, 2])
        ax2d.set_title('all inputs  (dark)')
        _plot_importance_row(ax2d, ax3d, data, 'full_trainDark_testDark', c,
                            feature_names, colors, nice_feature_names,
                            group_keys, group_labels, hatch='//')

        for ax_l, ax_d in [(ax2, ax2d), (ax3, ax3d)]:
            yl = ax_l.get_ylim()
            yd = ax_d.get_ylim()
            shared_max = max(yl[1], yd[1])
            shared_min = min(yl[0], yd[0])
            ax_l.set_ylim([shared_min, shared_max])
            ax_d.set_ylim([shared_min, shared_max])

        behavior_keys = np.array([
            'theta', 'dTheta', 'phi', 'dPhi',
            'pitch', 'gyro_y', 'roll', 'gyro_x', 'yaw', 'gyro_z'
        ])

        ax7 = fig.add_subplot(gs[4, 0])
        ax8 = fig.add_subplot(gs[4, 1])
        ax9 = fig.add_subplot(gs[4, 2])

        sharedmax = 0

        for i in [0, 2]:
            bkey = behavior_keys[i]
            ax7.plot(tdata['{}_1dbins'.format(bkey)], tdata['{}_1dtuning'.format(bkey)][c,:,1], color=colors[i], lw=2,
                    label='{} (MI={:.2})'.format(bkey, tdata['{}_l_mod'.format(bkey)][c]))
            ax7.fill_between(
                tdata['{}_1dbins'.format(bkey)],
                tdata['{}_1dtuning'.format(bkey)][c,:,1] + tdata['{}_1derr'.format(bkey)][c,:,1],
                tdata['{}_1dtuning'.format(bkey)][c,:,1] - tdata['{}_1derr'.format(bkey)][c,:,1],
                color=colors[i], alpha=0.2
            )
            ax7.hlines(
                np.percentile(tdata['{}_1dtuning'.format(bkey)][c,:,1], 10),
                tdata['{}_1dbins'.format(bkey)].min(),
                tdata['{}_1dbins'.format(bkey)].max(),
                ls='--', lw=1, color=colors[i]
            )
            singlemax = np.max(tdata['{}_1dtuning'.format(bkey)][c,:,1] + tdata['{}_1derr'.format(bkey)][c,:,1])
            if singlemax > sharedmax:
                sharedmax = singlemax
        ax7.legend(fontsize=6, loc='upper left')
        ax7.set_xlabel('deg')

        for i in [1, 3]:
            bkey = behavior_keys[i]
            ax9.plot(tdata['{}_1dbins'.format(bkey)], tdata['{}_1dtuning'.format(bkey)][c,:,1], color=colors[i], lw=2,
                    label='{} (MI={:.2})'.format(bkey, tdata['{}_l_mod'.format(bkey)][c]))
            ax9.fill_between(
                tdata['{}_1dbins'.format(bkey)],
                tdata['{}_1dtuning'.format(bkey)][c,:,1] + tdata['{}_1derr'.format(bkey)][c,:,1],
                tdata['{}_1dtuning'.format(bkey)][c,:,1] - tdata['{}_1derr'.format(bkey)][c,:,1],
                color=colors[i], alpha=0.2
            )
            ax9.hlines(
                np.percentile(tdata['{}_1dtuning'.format(bkey)][c,:,1], 10),
                tdata['{}_1dbins'.format(bkey)].min(),
                tdata['{}_1dbins'.format(bkey)].max(),
                ls='--', lw=1, color=colors[i]
            )
            singlemax = np.max(tdata['{}_1dtuning'.format(bkey)][c,:,1] + tdata['{}_1derr'.format(bkey)][c,:,1])
            if singlemax > sharedmax:
                sharedmax = singlemax
        ax9.legend(fontsize=6, loc='upper left')
        ax9.set_xlabel('deg/s')

        for i in [4, 6]:
            bkey = behavior_keys[i]
            ax8.plot(tdata['{}_1dbins'.format(bkey)], tdata['{}_1dtuning'.format(bkey)][c,:,1], color=colors[i], lw=2,
                    label='{} (MI={:.2})'.format(bkey, tdata['{}_l_mod'.format(bkey)][c]))
            ax8.fill_between(
                tdata['{}_1dbins'.format(bkey)],
                tdata['{}_1dtuning'.format(bkey)][c,:,1] + tdata['{}_1derr'.format(bkey)][c,:,1],
                tdata['{}_1dtuning'.format(bkey)][c,:,1] - tdata['{}_1derr'.format(bkey)][c,:,1],
                color=colors[i], alpha=0.2
            )
            ax8.hlines(
                np.percentile(tdata['{}_1dtuning'.format(bkey)][c,:,1], 10),
                tdata['{}_1dbins'.format(bkey)].min(),
                tdata['{}_1dbins'.format(bkey)].max(),
                ls='--', lw=1, color=colors[i]
            )
            singlemax = np.max(tdata['{}_1dtuning'.format(bkey)][c,:,1] + tdata['{}_1derr'.format(bkey)][c,:,1])
            if singlemax > sharedmax:
                sharedmax = singlemax
        ax8.set_xlabel('deg')
        ax8.legend(fontsize=6, loc='upper left')

        for i in [5, 7, 9]:
            bkey = behavior_keys[i]
            ax9.plot(tdata['{}_1dbins'.format(bkey)], tdata['{}_1dtuning'.format(bkey)][c,:,1], color=colors[i], lw=2,
                    label='{} (MI={:.2})'.format(bkey, tdata['{}_l_mod'.format(bkey)][c]))
            ax9.fill_between(
                tdata['{}_1dbins'.format(bkey)],
                tdata['{}_1dtuning'.format(bkey)][c,:,1] + tdata['{}_1derr'.format(bkey)][c,:,1],
                tdata['{}_1dtuning'.format(bkey)][c,:,1] - tdata['{}_1derr'.format(bkey)][c,:,1],
                color=colors[i], alpha=0.2
            )
            ax9.hlines(
                np.percentile(tdata['{}_1dtuning'.format(bkey)][c,:,1], 10),
                tdata['{}_1dbins'.format(bkey)].min(),
                tdata['{}_1dbins'.format(bkey)].max(),
                ls='--', lw=1, color=colors[i]
            )
            singlemax = np.max(tdata['{}_1dtuning'.format(bkey)][c,:,1] + tdata['{}_1derr'.format(bkey)][c,:,1])
            if singlemax > sharedmax:
                sharedmax = singlemax
        ax9.set_xlabel('deg/s')
        ax9.legend(fontsize=6, loc='upper left')
        ax9.set_xlim([-50, 50])

        for ax in [ax7, ax8, ax9]:
            ax.set_ylabel('norm. inf. spike rate')
            ax.set_ylim([0, sharedmax * 1.1])

        ax7.set_title('eye positions')
        ax8.set_title('head positions')
        ax9.set_title('velocities')

        _r2_arr  = data.get('full_trainLight_testLight_r2',    data.get('full_r2'))
        _cor_arr = data.get('full_trainLight_testLight_corrs', data.get('full_corrs'))
        r_rank = int(np.where(np.argsort(_r2_arr)[::-1] == c)[0]) + 1
        fig.suptitle('Cell {}, $R^2$={:.3}, corr={:.3}, r-rank={}/{}'.format(
            c,
            _r2_arr[c],
            _cor_arr[c],
            r_rank,
            len(_r2_arr)
        ))

        fig.tight_layout()
        savename = os.path.join(os.path.split(basepath)[0], 'model_results_cell_{}.png'.format(c))
        print('saving {}'.format(savename))
        fig.savefig(savename)



if __name__ == '__main__':
    make_cell_summary()