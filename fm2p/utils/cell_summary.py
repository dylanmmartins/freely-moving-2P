# -*- coding: utf-8 -*-

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages


from .files import read_h5
from .PETH import calc_PETHs



def make_filler_series(nc, ni):
    s = pd.Series(np.zeros(nc))
    for i in s.index.values:
        s.iloc[i] = (np.zeros(ni)*np.nan).astype(object)
    return s


rmse = lambda y, y_pred: ((sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred)) / len(y)) ** 0.5)


def make_pooled_dataset():

    tiled_positions = [1000, 500, 0, -500, -1000]

    tiled_GLM = [
        r'K:\Mini2P\250630_DMM_DMM037_ltdk\fm3\250630_DMM_DMM037_fm_03_multicell_GLM_results_v9_ltdk.h5',
        r'K:\Mini2P\250628_DMM_DMM037_ltdk\fm3\250628_DMM_DMM037_fm_03_multicell_GLM_results_v9_ltdk.h5',
        r'K:\Mini2P\250626_DMM_DMM037_ltdk\fm1\250626_DMM_DMM037_fm_01_multicell_GLM_results_v9_ltdk.h5',
        r'K:\Mini2P\250627_DMM_DMM037_ltdk\fm3\250627_DMM_DMM037_fm_02_multicell_GLM_results_v9_ltdk.h5',
        r'K:\Mini2P\250627_DMM_DMM037_ltdk\fm5\250627_DMM_DMM037_fm_05_multicell_GLM_results_v9_ltdk.h5'
    ]

    tiled_revcorr = [
        r'K:/Mini2P/250630_DMM_DMM037_ltdk\fm3\250630_DMM_DMM037_fm_03_revcorr_results_v3.h5',
        r'K:\Mini2P\250628_DMM_DMM037_ltdk\fm3\250628_DMM_DMM037_fm_03_revcorr_results_v3.h5',
        r'K:\Mini2P\250626_DMM_DMM037_ltdk\fm1\250626_DMM_DMM037_fm_01_revcorr_results_v3.h5',
        r'K:\Mini2P\250627_DMM_DMM037_ltdk\fm3\250627_DMM_DMM037_fm_02_revcorr_results_v3.h5',
        r'K:\Mini2P\250627_DMM_DMM037_ltdk\fm5\250627_DMM_DMM037_fm_05_revcorr_results_v3.h5'
    ]

    tiled_preproc = [
        r'K:/Mini2P/250630_DMM_DMM037_ltdk\fm3\250630_DMM_DMM037_fm_03_preproc.h5',
        r'K:\Mini2P\250628_DMM_DMM037_ltdk\fm3\250628_DMM_DMM037_fm_03_preproc.h5',
        r'K:\Mini2P\250626_DMM_DMM037_ltdk\fm1\250626_DMM_DMM037_fm_01_preproc.h5',
        r'K:\Mini2P\250627_DMM_DMM037_ltdk\fm3\250627_DMM_DMM037_fm_02_preproc.h5',
        r'K:\Mini2P\250627_DMM_DMM037_ltdk\fm5\250627_DMM_DMM037_fm_05_preproc.h5'
    ]


    merged_df = pd.DataFrame()

    for i in range(len(tiled_GLM)):
        
        glm_data = read_h5(tiled_GLM[i])
        revcorr_data = read_h5(tiled_revcorr[i])
        preproc_data = read_h5(tiled_preproc[i])

        peth_dict = calc_PETHs(preproc_data)

        df = pd.DataFrame()

        n_cells = np.size(glm_data['pupil_light']['weights'], 0) - 1

        df['theta_light_GLMweights'] = glm_data['pupil_light']['weights'][1:,0]
        df['phi_light_GLMweights'] = glm_data['pupil_light']['weights'][1:,1]
        df['theta_dark_GLMweights'] = glm_data['pupil_dark']['weights'][1:,0]
        df['phi_dark_GLMweights'] = glm_data['pupil_dark']['weights'][1:,1]

        df['theta_light_RMSE'] = rmse(glm_data['pupil_light']['y_test'][0,:], glm_data['pupil_light']['y_hat'][0,:])
        df['phi_light_RMSE'] = rmse(glm_data['pupil_light']['y_test'][1,:], glm_data['pupil_light']['y_hat'][1,:])
        df['theta_dark_RMSE'] = rmse(glm_data['pupil_dark']['y_test'][0,:], glm_data['pupil_dark']['y_hat'][0,:])
        df['phi_dark_RMSE'] = rmse(glm_data['pupil_dark']['y_test'][1,:], glm_data['pupil_dark']['y_hat'][1,:])
        
        for state in ['light', 'dark']:
            for key in ['distance_to_pillar', 'egocentric', 'phi', 'retinocentric', 'theta', 'yaw']:

                s = make_filler_series(n_cells, 12)
                for c in range(n_cells):
                    s.iloc[c] = (revcorr_data[state][key]['tuning_curve'][c,:]).astype(object)
                df['{}_{}_tuning_curve'.format(key,state)] = s

                s = make_filler_series(n_cells, 12)
                for c in range(n_cells):
                    s.iloc[c] = (revcorr_data[state][key]['tuning_stderr'][c,:]).astype(object)
                df['{}_{}_tuning_err'.format(key,state)] = s

                s = make_filler_series(n_cells, 12)
                for c in range(n_cells):
                    s.iloc[c] = (revcorr_data[state][key]['tuning_bins']).astype(object)
                df['{}_{}_tuning_bins'.format(key,state)] = s

        peth_keys = [
            'right_PETHs',
            'left_PETHs',
            'up_PETHs',
            'down_PETHs',
            'norm_right_PETHs',
            'norm_left_PETHs',
            'norm_up_PETHs',
            'norm_down_PETHs'
        ]
        for k in peth_keys:
            s = make_filler_series(n_cells, 12)
            for c in range(n_cells):
                s.iloc[c] = (peth_dict[k][c,:])
            df[k[:-1]] = s

        df['xloc'] = np.zeros(n_cells)
        df['yloc'] = np.zeros(n_cells)
        for c in range(n_cells):
            df['xloc'].iloc[c] = np.median(preproc_data['cell_x_pix'][str(c)])
            df['yloc'].iloc[c] = np.median(preproc_data['cell_y_pix'][str(c)])

        df['cell_num'] = df.index.values
        df['full_name'] = '_'.join([
            os.path.split(tiled_GLM[i])[1].split('_')[0],
            os.path.split(tiled_GLM[i])[1].split('_')[2],
            '_'.join(os.path.split(tiled_GLM[i])[1].split('_')[3:5]),
        ])
        df['rec_num'] = i
        df['rec_date'] = os.path.split(tiled_GLM[i])[1].split('_')[0]
        df['animal'] = os.path.split(tiled_GLM[i])[1].split('_')[2]
        df['rec_name'] = '_'.join(os.path.split(tiled_GLM[i])[1].split('_')[3:5])
        df['ML_offset'] = 0
        df['AP_offset'] = tiled_positions[i]

        df['merged_yloc'] = df['yloc'] + tiled_positions[i]
        # df['merged_yloc'] = df['yloc'] + tiled_positions[i]

        merged_df = pd.concat([merged_df, df], axis=0)

    merged_index = merged_df.reset_index()

    merged_index.to_hdf(r'K:\Mini2P\merged_DMM037_dataset.h5', 'k')

    return merged_index


def summarize_pooled_cells(merged_index):

    pdf = PdfPages(r'K:\Mini2P\DMM037_cell_sumary_v1.pdf')

    for ind in tqdm(merged_index.index.values):

        fig = plt.figure(figsize=(12,8), dpi=300)
        gs = gridspec.GridSpec(4, 4, figure=fig)

        ax_big = fig.add_subplot(gs[2:4, 2:4])
        ax0 = fig.add_subplot(gs[0,0])
        ax1 = fig.add_subplot(gs[0,1])
        ax_wide = fig.add_subplot(gs[0,2:4])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[1,1])
        ax4 = fig.add_subplot(gs[2,0])
        # ax5 = fig.add_subplot(gs[2,1])
        ax6 = fig.add_subplot(gs[3,0])
        # ax7 = fig.add_subplot(gs[3,1])

        psth_bins = np.arange(-15,16)*(1/7.49)

        ax0.plot(
            merged_index.loc[ind,'theta_light_tuning_bins'],
            merged_index.loc[ind,'theta_light_tuning_curve'],
            color='orange',
            lw=2,
            label='light'
        )
        ax0.fill_between(
            (merged_index.loc[ind,'theta_light_tuning_bins']).astype(np.float64),
            (merged_index.loc[ind,'theta_light_tuning_curve'] - merged_index.loc[ind,'theta_light_tuning_err']).astype(np.float64),
            (merged_index.loc[ind,'theta_light_tuning_curve'] + merged_index.loc[ind,'theta_light_tuning_err']).astype(np.float64),
            color='orange',
            alpha=0.3
        )
        ax0.plot(
            merged_index.loc[ind,'theta_dark_tuning_bins'],
            merged_index.loc[ind,'theta_dark_tuning_curve'],
            color='indigo',
            label='dark',
            lw=2
        )
        ax0.fill_between(
            (merged_index.loc[ind,'theta_dark_tuning_bins']).astype(np.float64),
            (merged_index.loc[ind,'theta_dark_tuning_curve'] - merged_index.loc[ind,'theta_dark_tuning_err']).astype(np.float64),
            (merged_index.loc[ind,'theta_dark_tuning_curve'] + merged_index.loc[ind,'theta_dark_tuning_err']).astype(np.float64),
            color='indigo',
            alpha=0.3
        )
        _setmax = np.nanmax([
            np.nanmax((merged_index.loc[ind,'theta_light_tuning_curve'] + merged_index.loc[ind,'theta_light_tuning_err']).astype(np.float64)),
            np.nanmax((merged_index.loc[ind,'theta_dark_tuning_curve'] + merged_index.loc[ind,'theta_dark_tuning_err']).astype(np.float64))
        ])
        ax0.set_ylim([0, _setmax*1.1])
        ax0.set_xlabel('theta (deg)')
        ax0.set_ylabel('norm spikes')
        ax0.legend()
        ax1.plot(
            psth_bins,
            merged_index.loc[ind,'norm_left_PETH'],
            color='tab:blue',
            label='left',
            lw=2
        )
        ax1.plot(
            psth_bins,
            merged_index.loc[ind,'norm_right_PETH'],
            color='tab:red',
            label='right',
            lw=2
        )
        ax1.legend()
        _setmax = np.nanmax([
            np.nanmax(np.abs(merged_index.loc[ind,'norm_right_PETH'])),
            np.nanmax(np.abs(merged_index.loc[ind,'norm_left_PETH']))
        ])
        ax1.set_ylim([-_setmax*1.1, _setmax*1.1])
        ax1.set_xlabel('time (sec)')
        ax1.set_ylabel('norm spikes')
        ax1.vlines(0, -_setmax*1.1, _setmax*1.1, color='k', alpha=0.4, ls='--')

        ax_wide.bar(0, merged_index.loc[ind,'theta_light_GLMweights'], width=0.5, color='orange')
        ax_wide.bar(0.5, merged_index.loc[ind,'theta_dark_GLMweights'], width=0.5, color='indigo')
        ax_wide.bar(1, merged_index.loc[ind,'phi_light_GLMweights'], width=0.5, color='orange')
        ax_wide.bar(1.5, merged_index.loc[ind,'phi_dark_GLMweights'], width=0.5, color='indigo')
        ax_wide.set_xticks([0.25, 1.25], labels=['theta','phi'])
        ax_wide.set_ylim([-0.5,0.5])
        ax_wide.hlines(0, -0.25, 1.75, ls='--', color='k')
        ax_wide.set_ylabel('GLM weight')

        ax2.plot(
            merged_index.loc[ind,'phi_light_tuning_bins'],
            merged_index.loc[ind,'phi_light_tuning_curve'],
            color='orange',
            lw=2
        )
        ax2.fill_between(
            (merged_index.loc[ind,'phi_light_tuning_bins']).astype(np.float64),
            (merged_index.loc[ind,'phi_light_tuning_curve'] - merged_index.loc[ind,'phi_light_tuning_err']).astype(np.float64),
            (merged_index.loc[ind,'phi_light_tuning_curve'] + merged_index.loc[ind,'phi_light_tuning_err']).astype(np.float64),
            color='orange',
            alpha=0.3
        )
        ax2.plot(
            merged_index.loc[ind,'phi_dark_tuning_bins'],
            merged_index.loc[ind,'phi_dark_tuning_curve'],
            color='indigo',
            lw=2
        )
        ax2.fill_between(
            (merged_index.loc[ind,'phi_dark_tuning_bins']).astype(np.float64),
            (merged_index.loc[ind,'phi_dark_tuning_curve'] - merged_index.loc[ind,'phi_dark_tuning_err']).astype(np.float64),
            (merged_index.loc[ind,'phi_dark_tuning_curve'] + merged_index.loc[ind,'phi_dark_tuning_err']).astype(np.float64),
            color='indigo',
            alpha=0.3
        )
        _setmax = np.nanmax([
            np.nanmax((merged_index.loc[ind,'phi_light_tuning_curve'] + merged_index.loc[ind,'phi_light_tuning_err']).astype(np.float64)),
            np.nanmax((merged_index.loc[ind,'phi_dark_tuning_curve'] + merged_index.loc[ind,'phi_dark_tuning_err']).astype(np.float64))
        ])
        ax2.set_ylim([0, _setmax*1.1])
        ax2.set_xlabel('phi (deg)')
        ax2.set_ylabel('norm spikes')
        ax3.plot(
            psth_bins,
            merged_index.loc[ind,'norm_down_PETH'],
            color='tab:blue',
            lw=2,
            label='down'
        )
        ax3.plot(
            psth_bins,
            merged_index.loc[ind,'norm_up_PETH'],
            color='tab:red',
            label='up',
            lw=2
        )
        ax3.set_xlabel('time (sec)')
        ax3.set_ylabel('norm spikes')
        _setmax = np.nanmax([
            np.nanmax(np.abs(merged_index.loc[ind,'norm_up_PETH'])),
            np.nanmax(np.abs(merged_index.loc[ind,'norm_right_PETH']))
        ])
        ax3.set_ylim([-_setmax*1.1, _setmax*1.1])
        ax3.legend()


        ax4.plot(
            merged_index.loc[ind,'retinocentric_light_tuning_bins'],
            merged_index.loc[ind,'retinocentric_light_tuning_curve'],
            color='orange',
            lw=2
        )
        ax4.fill_between(
            (merged_index.loc[ind,'retinocentric_light_tuning_bins']).astype(np.float64),
            (merged_index.loc[ind,'retinocentric_light_tuning_curve'] - merged_index.loc[ind,'retinocentric_light_tuning_err']).astype(np.float64),
            (merged_index.loc[ind,'retinocentric_light_tuning_curve'] + merged_index.loc[ind,'retinocentric_light_tuning_err']).astype(np.float64),
            color='orange',
            alpha=0.3
        )
        ax4.plot(
            merged_index.loc[ind,'retinocentric_dark_tuning_bins'],
            merged_index.loc[ind,'retinocentric_dark_tuning_curve'],
            color='indigo',
            lw=2
        )
        ax4.fill_between(
            (merged_index.loc[ind,'retinocentric_dark_tuning_bins']).astype(np.float64),
            (merged_index.loc[ind,'retinocentric_dark_tuning_curve'] - merged_index.loc[ind,'retinocentric_dark_tuning_err']).astype(np.float64),
            (merged_index.loc[ind,'retinocentric_dark_tuning_curve'] + merged_index.loc[ind,'retinocentric_dark_tuning_err']).astype(np.float64),
            color='indigo',
            alpha=0.3
        )
        _setmax = np.nanmax([
            np.nanmax((merged_index.loc[ind,'retinocentric_light_tuning_curve'] + merged_index.loc[ind,'retinocentric_light_tuning_err']).astype(np.float64)),
            np.nanmax((merged_index.loc[ind,'retinocentric_dark_tuning_curve'] + merged_index.loc[ind,'retinocentric_dark_tuning_err']).astype(np.float64))
        ])
        ax4.set_ylim([0, _setmax*1.1])
        ax4.set_xlabel('retinocentric (deg)')
        ax4.set_ylabel('norm spikes')

        ax6.plot(
            merged_index.loc[ind,'egocentric_light_tuning_bins'],
            merged_index.loc[ind,'egocentric_light_tuning_curve'],
            color='orange',
            lw=2
        )
        ax6.fill_between(
            (merged_index.loc[ind,'egocentric_light_tuning_bins']).astype(np.float64),
            (merged_index.loc[ind,'egocentric_light_tuning_curve'] - merged_index.loc[ind,'egocentric_light_tuning_err']).astype(np.float64),
            (merged_index.loc[ind,'egocentric_light_tuning_curve'] + merged_index.loc[ind,'egocentric_light_tuning_err']).astype(np.float64),
            color='orange',
            alpha=0.3
        )
        ax6.plot(
            merged_index.loc[ind,'egocentric_dark_tuning_bins'],
            merged_index.loc[ind,'egocentric_dark_tuning_curve'],
            color='indigo',
            lw=2
        )
        ax6.fill_between(
            (merged_index.loc[ind,'egocentric_dark_tuning_bins']).astype(np.float64),
            (merged_index.loc[ind,'egocentric_dark_tuning_curve'] - merged_index.loc[ind,'egocentric_dark_tuning_err']).astype(np.float64),
            (merged_index.loc[ind,'egocentric_dark_tuning_curve'] + merged_index.loc[ind,'egocentric_dark_tuning_err']).astype(np.float64),
            color='indigo',
            alpha=0.3
        )
        _setmax = np.nanmax([
            np.nanmax((merged_index.loc[ind,'egocentric_light_tuning_curve'] + merged_index.loc[ind,'egocentric_light_tuning_err']).astype(np.float64)),
            np.nanmax((merged_index.loc[ind,'egocentric_dark_tuning_curve'] + merged_index.loc[ind,'egocentric_dark_tuning_err']).astype(np.float64))
        ])
        ax6.set_ylim([0, _setmax*1.1])
        ax6.set_xlabel('egocentric (deg)')
        ax6.set_ylabel('norm spikes')

        showinds = merged_index.index.values
        showinds = [c for c in showinds if c != ind]
        for i in showinds:
            ax_big.plot(-merged_index.at[i, 'xloc'], merged_index.at[i, 'merged_yloc'], '.', color='k', alpha=0.3)
        ax_big.axis('equal')
        ax_big.plot(-merged_index.at[ind, 'xloc'], merged_index.at[ind, 'merged_yloc'], '*', color='tab:green', ms=10)
        ax_big.set_xlabel('medial / lateral')
        ax_big.set_ylabel('anterior / posterior')

        fig.suptitle('{} cell {}'.format(merged_index.loc[ind,'animal'], ind))

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    pdf.close()


if __name__ == '__main__':

    df = make_pooled_dataset()

    summarize_pooled_cells(df)