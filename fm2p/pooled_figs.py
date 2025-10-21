
import numpy as np
import pandas as pd
import os
import fm2p
import matplotlib as mpl
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 7
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_2D_hist(ax, celldata):
    x_vals = celldata['theta_interp']
    y_vals = celldata['phi_interp']
    rates = celldata['norm_spikes']
    n_x = 20
    n_y = 20

    x_bins = np.linspace(np.nanpercentile(x_vals, 5), np.nanpercentile(x_vals, 95), num=n_x+1)
    y_bins = np.linspace(np.nanpercentile(y_vals, 5), np.nanpercentile(y_vals, 95), num=n_y+1)

    # 2D histogram of firing rates
    heatmap = np.zeros((n_y, n_x))

    # Compute average firing rate in each bin
    for i in range(n_x):
        for j in range(n_y):
            in_bin = (x_vals >= x_bins[i]) & (x_vals < x_bins[i+1]) & \
                    (y_vals >= y_bins[j]) & (y_vals < y_bins[j+1])
            # if np.any(in_bin):
            heatmap[j, i] = np.nanmean(rates[in_bin])
            # else:
            #     heatmap[j, i] = np.nan  # or 0, depending on your needs

    ax.imshow(heatmap, origin='lower', 
            extent=[x_bins[0], x_bins[-1], y_bins[-1], y_bins[0]],
            aspect='auto', cmap='coolwarm', vmin=0, vmax=np.nanpercentile(heatmap.flatten(), 95))

def pooled_panels(data, celldata, cell, pdf):
        
    fig = plt.figure(figsize=(11,8.5), dpi=300)
    gs = gridspec.GridSpec(4, 4, figure=fig)

    ax_big = fig.add_subplot(gs[2:4, 2:4])
    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[0,1])
    ax_wide = fig.add_subplot(gs[0,2:4])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])
    ax4 = fig.add_subplot(gs[2,0])
    ax5 = fig.add_subplot(gs[2,1])
    ax6 = fig.add_subplot(gs[3,0])
    ax7 = fig.add_subplot(gs[3,1])
    ax9 = fig.add_subplot(gs[1,2])
    ax8 = fig.add_subplot(gs[1,3])

    psth_bins = np.arange(-15,16)*(1/7.49)

    ax0.plot(
        celldata['light_theta_tuning_bins'],
        celldata['light_theta_tuning_curve'],
        color='orange',
        lw=2,
        label='light'
    )
    ax0.fill_between(
        (celldata['light_theta_tuning_bins']).astype(np.float64),
        (celldata['light_theta_tuning_curve'] - celldata['light_theta_tuning_error']).astype(np.float64),
        (celldata['light_theta_tuning_curve'] + celldata['light_theta_tuning_error']).astype(np.float64),
        color='orange',
        alpha=0.3
    )
    ax0.plot(
        celldata['dark_theta_tuning_bins'],
        celldata['dark_theta_tuning_curve'],
        color='indigo',
        label='dark',
        lw=2
    )
    ax0.fill_between(
        (celldata['dark_theta_tuning_bins']).astype(np.float64),
        (celldata['dark_theta_tuning_curve'] - celldata['dark_theta_tuning_error']).astype(np.float64),
        (celldata['dark_theta_tuning_curve'] + celldata['dark_theta_tuning_error']).astype(np.float64),
        color='indigo',
        alpha=0.3
    )
    _setmax = np.max([
        np.max((celldata['light_theta_tuning_curve'] + celldata['light_theta_tuning_error']).astype(np.float64)),
        np.max((celldata['dark_theta_tuning_curve'] + celldata['dark_theta_tuning_error']).astype(np.float64))
    ])
    ax0.set_ylim([0, _setmax*1.1])
    ax0.set_xlabel('theta (deg)')
    ax0.set_ylabel('norm spikes')
    ax0.legend(frameon=False)

    ax1.plot(
        psth_bins,
        celldata['norm_left_PETHs_sps'],
        color='tab:blue',
        label='left',
        lw=2
    )
    ax1.plot(
        psth_bins,
        celldata['norm_right_PETHs_sps'],
        color='tab:red',
        label='right',
        lw=2
    )
    ax1.legend(frameon=False)
    _setmax = np.max([
        np.max(np.abs(celldata['norm_right_PETHs_sps'])),
        np.max(np.abs(celldata['norm_left_PETHs_sps']))
    ])
    ax1.set_ylim([-_setmax*1.1, _setmax*1.1])
    ax1.set_xlabel('time (sec)')
    ax1.set_ylabel('norm spikes')
    ax1.vlines(0, -_setmax*1.1, _setmax*1.1, color='k', alpha=0.4, ls='--')
    ax1.hlines(0, psth_bins[0], psth_bins[-1], color='k', alpha=0.4, ls='--')

    ax7.plot(
        psth_bins,
        celldata['norm_down_PETHs_dFF'],
        color='tab:blue',
        lw=2,
        label='down'
    )
    ax7.plot(
        psth_bins,
        celldata['norm_up_PETHs_dFF'],
        color='tab:red',
        label='up',
        lw=2
    )
    ax7.set_xlabel('time (sec)')
    ax7.set_ylabel('norm spikes')
    _setmax = np.max([
        np.max(np.abs(celldata['norm_up_PETHs_dFF'])),
        np.max(np.abs(celldata['norm_down_PETHs_dFF']))
    ])
    ax7.set_ylim([-_setmax*1.1, _setmax*1.1])
    ax7.legend(frameon=False)
    ax7.hlines(0, psth_bins[0], psth_bins[-1], color='k', alpha=0.4, ls='--')
    ax7.vlines(0, -_setmax*1.1, _setmax*1.1, color='k', alpha=0.4, ls='--')

    ax5.plot(
        psth_bins,
        celldata['norm_left_PETHs_dFF'],
        color='tab:blue',
        label='left',
        lw=2
    )
    ax5.plot(
        psth_bins,
        celldata['norm_right_PETHs_dFF'],
        color='tab:red',
        label='right',
        lw=2
    )
    ax5.legend(frameon=False)
    _setmax = np.max([
        np.max(np.abs(celldata['norm_right_PETHs_dFF'])),
        np.max(np.abs(celldata['norm_left_PETHs_dFF']))
    ])
    ax5.set_ylim([-_setmax*1.1, _setmax*1.1])
    ax5.set_xlabel('time (sec)')
    ax5.set_ylabel('norm spikes')
    ax5.vlines(0, -_setmax*1.1, _setmax*1.1, color='k', alpha=0.4, ls='--')
    ax5.hlines(0, psth_bins[0], psth_bins[-1], color='k', alpha=0.4, ls='--')

    ax2.plot(
        celldata['light_phi_tuning_bins'],
        celldata['light_phi_tuning_curve'],
        color='orange',
        lw=2
    )
    ax2.fill_between(
        (celldata['light_phi_tuning_bins']).astype(np.float64),
        (celldata['light_phi_tuning_curve'] - celldata['light_phi_tuning_error']).astype(np.float64),
        (celldata['light_phi_tuning_curve'] + celldata['light_phi_tuning_error']).astype(np.float64),
        color='orange',
        alpha=0.3
    )
    ax2.plot(
        celldata['dark_phi_tuning_bins'],
        celldata['dark_phi_tuning_curve'],
        color='indigo',
        lw=2
    )
    ax2.fill_between(
        (celldata['dark_phi_tuning_bins']).astype(np.float64),
        (celldata['dark_phi_tuning_curve'] - celldata['dark_phi_tuning_error']).astype(np.float64),
        (celldata['dark_phi_tuning_curve'] + celldata['dark_phi_tuning_error']).astype(np.float64),
        color='indigo',
        alpha=0.3
    )
    _setmax = np.max([
        np.max((celldata['light_phi_tuning_curve'] + celldata['light_phi_tuning_error']).astype(np.float64)),
        np.max((celldata['dark_phi_tuning_curve'] + celldata['dark_phi_tuning_error']).astype(np.float64))
    ])
    ax2.set_ylim([0, _setmax*1.1])
    ax2.set_xlabel('phi (deg)')
    ax2.set_ylabel('norm spikes')

    ax3.plot(
        psth_bins,
        celldata['norm_down_PETHs_sps'],
        color='tab:blue',
        lw=2,
        label='down'
    )
    ax3.plot(
        psth_bins,
        celldata['norm_up_PETHs_sps'],
        color='tab:red',
        label='up',
        lw=2
    )
    ax3.set_xlabel('time (sec)')
    ax3.set_ylabel('norm spikes')
    _setmax = np.max([
        np.max(np.abs(celldata['norm_up_PETHs_sps'])),
        np.max(np.abs(celldata['norm_down_PETHs_sps']))
    ])
    ax3.set_ylim([-_setmax*1.1, _setmax*1.1])
    ax3.legend(frameon=False)
    ax3.hlines(0, psth_bins[0], psth_bins[-1], color='k', alpha=0.4, ls='--')
    ax3.vlines(0, -_setmax*1.1, _setmax*1.1, color='k', alpha=0.4, ls='--')

    ax4.plot(
        celldata['light_retinocentric_tuning_bins'],
        celldata['light_retinocentric_tuning_curve'],
        color='orange',
        lw=2
    )
    ax4.fill_between(
        (celldata['light_retinocentric_tuning_bins']).astype(np.float64),
        (celldata['light_retinocentric_tuning_curve'] - celldata['light_retinocentric_tuning_error']).astype(np.float64),
        (celldata['light_retinocentric_tuning_curve'] + celldata['light_retinocentric_tuning_error']).astype(np.float64),
        color='orange',
        alpha=0.3
    )
    ax4.plot(
        celldata['dark_retinocentric_tuning_bins'],
        celldata['dark_retinocentric_tuning_curve'],
        color='indigo',
        lw=2
    )
    ax4.fill_between(
        (celldata['dark_retinocentric_tuning_bins']).astype(np.float64),
        (celldata['dark_retinocentric_tuning_curve'] - celldata['dark_retinocentric_tuning_error']).astype(np.float64),
        (celldata['dark_retinocentric_tuning_curve'] + celldata['dark_retinocentric_tuning_error']).astype(np.float64),
        color='indigo',
        alpha=0.3
    )
    _setmax = np.max([
        np.max((celldata['light_retinocentric_tuning_curve'] + celldata['light_retinocentric_tuning_error']).astype(np.float64)),
        np.max((celldata['dark_retinocentric_tuning_curve'] + celldata['dark_retinocentric_tuning_error']).astype(np.float64))
    ])
    ax4.set_ylim([0, _setmax*1.1])
    ax4.set_xlabel('retinocentric (deg)')
    ax4.set_ylabel('norm spikes')

    ax6.plot(
        celldata['light_egocentric_tuning_bins'],
        celldata['light_egocentric_tuning_curve'],
        color='orange',
        lw=2
    )
    ax6.fill_between(
        (celldata['light_egocentric_tuning_bins']).astype(np.float64),
        (celldata['light_egocentric_tuning_curve'] - celldata['light_egocentric_tuning_error']).astype(np.float64),
        (celldata['light_egocentric_tuning_curve'] + celldata['light_egocentric_tuning_error']).astype(np.float64),
        color='orange',
        alpha=0.3
    )
    ax6.plot(
        celldata['dark_egocentric_tuning_bins'],
        celldata['dark_egocentric_tuning_curve'],
        color='indigo',
        lw=2
    )
    ax6.fill_between(
        (celldata['dark_egocentric_tuning_bins']).astype(np.float64),
        (celldata['dark_egocentric_tuning_curve'] - celldata['dark_egocentric_tuning_error']).astype(np.float64),
        (celldata['dark_egocentric_tuning_curve'] + celldata['dark_egocentric_tuning_error']).astype(np.float64),
        color='indigo',
        alpha=0.3
    )
    _setmax = np.max([
        np.max((celldata['light_egocentric_tuning_curve'] + celldata['light_egocentric_tuning_error']).astype(np.float64)),
        np.max((celldata['dark_egocentric_tuning_curve'] + celldata['dark_egocentric_tuning_error']).astype(np.float64))
    ])
    ax6.set_ylim([0, _setmax*1.1])
    ax6.set_xlabel('egocentric (deg)')
    ax6.set_ylabel('norm spikes')

    ax8.plot(celldata['head_x'], celldata['head_y'], color='k', lw=1)
    sp_inds = celldata['norm_spikes']>np.percentile(celldata['norm_spikes'], 94)
    ax8.axis('equal')
    cmap = plt.cm.hsv(np.linspace(0,1,360))
    for i in np.where(sp_inds)[0]:
        if not np.isnan(celldata['head_yaw_deg'][i]):
            ax8.plot(celldata['head_x'][i], celldata['head_y'][i], '.', ms=4,
                    color=cmap[int(celldata['head_yaw_deg'][i])])
            
    ax_wide.plot(celldata['twopT'], celldata['norm_dFF'], color='k', lw=1)
    ax_wide.set_xlim([0,600])
    ax_wide.set_xticks(np.linspace(0,600,5), np.linspace(0,600/60,5))
    ax_wide.set_xlabel('time (min)')
    ax_wide.set_ylabel('dFF')

    for name in data.keys():
        if data[name]['animal'].decode("utf-8") == celldata['animal'].decode("utf-8"):
            ax_big.plot(
                - data[name]['cell_xloc'] - (data[name]['ML']*100),
                data[name]['cell_yloc'] + (data[name]['AP']*100),
                '.', color='k', alpha=0.3
            )
    ax_big.axis('equal')
    ax_big.plot(-celldata['cell_xloc'],  celldata['cell_yloc'], '*', color='tab:green', ms=10)
    ax_big.set_xlabel('medial / lateral')
    ax_big.set_ylabel('anterior / posterior')

    plot_2D_hist(ax9, celldata)
    ax9.set_xlabel('theta (deg)')
    ax9.set_ylabel('phi (deg)')
    ax9.axis('equal')

    ax0.set_title('theta tuning')
    ax1.set_title('pupil PETH (sp/s)')
    ax2.set_title('phi tuning')
    ax3.set_title('pupil PETH (sp/s)')
    ax4.set_title('retino. tuning')
    # ax5.set_title('ax5')
    # ax7.set_title('ax7')
    ax6.set_title('ego. tuning')
    ax_wide.set_title('calcium trace')
    ax8.set_title('allocentric spikes')
    ax9.set_title('theta/phi tuning')
    ax_big.set_title('cell coordinates')

    ax5.set_title('pupil PETH (dFF)')
    ax7.set_title('pupil PETH (dFF)')

    fig.suptitle(cell)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def pooled_figs():
    pooled_data_dir = r'K:\Mini2P\merged_V1PPC_dataset_w251020_v1.h5'
    data = fm2p.read_h5(pooled_data_dir)

    unique_recordings = []
    for cell, celldata in data.items():
        if celldata['rec_name'].decode("utf-8") not in unique_recordings:
            unique_recordings.append(celldata['rec_name'].decode("utf-8"))

    for rec in unique_recordings:

        print('  -> Writing pdf for {}'.format(rec))
        print('  -> Will check all cells from dataset to see which are in {}'.format(rec))

        pdf_path = os.path.join(os.path.split(pooled_data_dir)[0], 'pooled_cell_summary_{}.pdf'.format(rec))
        pdf = PdfPages(pdf_path)

        for cell, celldata in tqdm(data.items()):
            if celldata['rec_name'].decode("utf-8") == rec:
                pooled_panels(data, celldata, cell, pdf)

        pdf.close()

        print(' --> PDF written to {}'.format(pdf_path))


if __name__ == '__main__':
    pooled_figs()