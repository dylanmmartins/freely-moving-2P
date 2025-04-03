

import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import fm2p

def summarize_revcorr():

    parser = argparse.ArgumentParser()
    parser.add_argument('--preproc', type=str, default=None)
    parser.add_argument('-v', '--version', type=str, default='00')
    args = parser.parse_args()

    if args.preproc is None:
        h5_path = fm2p.select_file(
            title='Choose a preprocessing HDF file.',
            filetypes=[('H5','.h5'),]
        )
        versionnum = fm2p.get_string_input(
            title='Enter summary version number (e.g., 01).'
        )
    else:
        h5_path = args.preproc
        versionnum = args.version

    data = fm2p.read_h5(h5_path)
    
    spikes = data['oasis_spks'].copy()
    egocentric = data['egocentric'].copy()
    retinocentric = data['retinocentric'].copy()
    pupil = data['pupil_from_head'].copy()
    speed = data['speed'].copy()
    speed = np.append(speed, speed[-1])
    use = speed > 1.5

    ego_bins = np.linspace(-180, 180, 19)
    retino_bins = np.linspace(-180, 180, 19) # 20 deg bins
    pupil_bins = np.linspace(45, 95, 11) # 5 deg bins

    lag_vals = [-3,-2,-1,0,1,2,3,4,5]

    spiketrains = np.zeros([
        np.size(spikes,0),
        np.sum(use)
    ]) * np.nan
    
    # break data into 10 chunks, randomly choose ten of them for each block
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

    pupil_xcorr = np.zeros([np.size(spikes, 0), len(lag_vals)]) * np.nan
    retino_xcorr = np.zeros([np.size(spikes, 0), len(lag_vals)]) * np.nan
    ego_xcorr = np.zeros([np.size(spikes, 0), len(lag_vals)]) * np.nan

    pupil_tunings = np.zeros([np.size(spikes, 0), len(lag_vals), len(pupil_bins)-1]) * np.nan
    ret_tunings = np.zeros([np.size(spikes, 0), len(lag_vals), len(retino_bins)-1]) * np.nan
    ego_tunings = np.zeros([np.size(spikes, 0), len(lag_vals), len(ego_bins)-1]) * np.nan

    # axis 2: pupil, retino, ego
    # axis 3: modulation index, peak value
    all_mods = np.zeros([np.size(spikes,0), len(lag_vals), 3, 2]) * np.nan

    savepath, savename = os.path.split(h5_path)
    savename = '{}_revcorrRFs_v{}.pdf'.format(savename.split('_preproc')[0], versionnum)
    pdf = PdfPages(os.path.join(savepath, savename))

    ### BEHAVIORAL OCCUPANCY
    fig, [[ax1,ax2,ax3],[ax4,ax5,ax6]] = plt.subplots(2, 3, dpi=300, figsize=(5.5,3.5))

    ax1.hist(data['pupil_from_head'][use], bins=pupil_bins, color='tab:blue')
    ax1.set_xlabel('pupil (deg)')
    ax1.set_xlim([pupil_bins[0], pupil_bins[-1]])

    ax2.hist(data['retinocentric'][use], bins=retino_bins, color='tab:orange')
    ax2.set_xlabel('retinocentric (deg)')
    ax2.set_xlim([retino_bins[0], retino_bins[-1]])

    ax3.hist(data['egocentric'][use], bins=ego_bins, color='tab:green')
    ax3.set_xlabel('egocentric (deg)')
    ax3.set_xlim([ego_bins[0], ego_bins[-1]])

    ax4.hist(speed, bins=np.linspace(0,60,20), color='k')
    ax4.set_title('{:.4}% running time'.format((np.sum(use)/len(use))*100))
    ax4.set_xlabel('speed (cm/s)')

    ax5.plot(data['head_x'][use], data['head_y'][use], 'k.', ms=1, alpha=0.3)
    ax5.invert_yaxis()
    ax5.axis('equal')

    ax6.plot(data['theta_interp'][use], data['phi_interp'][use], 'k.', ms=1, alpha=0.3)
    ax6.set_xlabel('theta (deg)')
    ax6.set_ylabel('phi (deg)')

    running_frac = len(data['twopT'][use]) / len(data['twopT'])
    running_min = running_frac * (data['twopT'][-1] / 60)

    fig.suptitle('{} ({:.3}/{:.3} min)'.format(
        savename.split('_preproc')[0],
        running_min,
        data['twopT'][-1]/60)
    )
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close()


    ### FLUORESCENCE

    ops_path = os.path.join(savepath, 'suite2p\plane0\ops.npy')
    stat_path = os.path.join(savepath, 'suite2p\plane0\stat.npy')
    iscell_path = os.path.join(savepath, 'suite2p\plane0\iscell.npy')

    ops = np.load(ops_path, allow_pickle=True)
    stat = np.load(stat_path, allow_pickle=True)
    iscell = np.load(iscell_path)
    usecells = iscell[:,0]==1

    fig, ax1 = plt.subplots(1,1, figsize=(4,4), dpi=300)
    ax1.imshow(ops.item()['max_proj'], cmap='gray', vmin=0, vmax=500)
    ax1.axis('off')
    for cell in stat[usecells]:
        ax1.scatter(np.mean(cell['xpix'])-10, np.mean(cell['ypix'])-10, s=25, facecolors='none', edgecolors='r', alpha=0.5)
    ax1.set_title('n cells = {}'.format(np.sum(usecells)))
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close()

    scale = 200
    norm_dFF = data['norm_dFF']
    twopT = data['twopT']
    fig, ax = plt.subplots(1,1, figsize=(7,10), dpi=300)
    for cell in range(20):
        ax.plot(twopT, norm_dFF[cell,:] + (cell*scale), lw=1, alpha=0.9)
    ax.set_xlim([twopT[0], twopT[-1]])
    ax.set_yticks([])
    ax.set_xlabel('time (s)')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close()


    ### SUMMARIZE TUNING OF INDIVIDUAL CELLS
    for c_i in tqdm(range(np.size(spikes, 0))):

        fig, axs = plt.subplots(3, 9, dpi=300, figsize=(15,6))

        _maxtuning = 0

        for lag_ind, lag_val in enumerate(lag_vals):
            
            for cell_i in range(np.size(spikes,0)):
                spiketrains[cell_i,:] = np.roll(spikes[cell_i,:], shift=lag_val)[use]

            pupil_cent, pupil_tuning, pupil_err = fm2p.tuning_curve(
                spiketrains[c_i,:][np.newaxis,:],
                pupil[use],
                pupil_bins
            )
            ret_cent, ret_tuning, ret_err = fm2p.tuning_curve(
                spiketrains[c_i,:][np.newaxis,:],
                retinocentric[use],
                retino_bins
            )
            ego_cent, ego_tuning, ego_err = fm2p.tuning_curve(
                spiketrains[c_i,:][np.newaxis,:],
                egocentric[use],
                ego_bins
            )

            fm2p.plot_tuning(axs[0,lag_ind], pupil_cent, pupil_tuning, pupil_err, 'tab:blue', False)
            fm2p.plot_tuning(axs[1,lag_ind], ret_cent, ret_tuning, ret_err, 'tab:orange', False)
            fm2p.plot_tuning(axs[2,lag_ind], ego_cent, ego_tuning, ego_err, 'tab:green', False)

            lag_str = (1/7.49) * 1000 * lag_val

            Pmod, Ppeak = fm2p.calc_modind(pupil_cent, pupil_tuning, spiketrains[c_i,:])
            if np.isnan(Ppeak):
                axs[0,lag_ind].set_title('{:.4}ms\nmod={:.3}'.format(lag_str, Pmod))
            else:
                axs[0,lag_ind].set_title('{:.4}ms\nmod={:.3} peak={:.4}\N{DEGREE SIGN}'.format(lag_str, Pmod, Ppeak))
            
            Rmod, Rpeak = fm2p.calc_modind(ret_cent, ret_tuning, spiketrains[c_i,:])
            if np.isnan(Rpeak):
                axs[1,lag_ind].set_title('{:.4}ms\nmod={:.3}'.format(lag_str, Rmod))
            else:
                axs[1,lag_ind].set_title('{:.4}ms\nmod={:.3} peak={:.4}\N{DEGREE SIGN}'.format(lag_str, Rmod, Rpeak))
            
            Emod, Epeak = fm2p.calc_modind(ego_cent, ego_tuning, spiketrains[c_i,:])
            if np.isnan(Epeak):
                axs[2,lag_ind].set_title('{:.4}ms\nmod={:.3}'.format(lag_str, Emod))
            else:
                axs[2,lag_ind].set_title('{:.4}ms\nmod={:.3} peak={:.4}\N{DEGREE SIGN}'.format(lag_str, Emod, Epeak))

            all_mods[c_i, lag_ind, 0, :] = Pmod, Ppeak
            all_mods[c_i, lag_ind, 1, :] = Rmod, Rpeak
            all_mods[c_i, lag_ind, 2, :] = Emod, Epeak

            # axs[0,lag_ind].set_title('lag={:.3} ms'.format())
            axs[0,lag_ind].set_xlabel('pupil (deg)')
            axs[1,lag_ind].set_xlabel('retino (deg)')
            axs[2,lag_ind].set_xlabel('ego (deg)')

            for x in [
                np.nanmax(pupil_tuning+pupil_err),
                np.nanmax(ret_tuning+ret_err),
                np.nanmax(ego_tuning+ego_err)]:
                if x > _maxtuning:
                    _maxtuning = x

            axs[1,lag_ind].vlines([-75,75], 0, _maxtuning, color='k', alpha=0.3, ls='--', lw=1)

            pupil_tunings[c_i, lag_ind, :] = pupil_tuning
            ret_tunings[c_i, lag_ind, :] = ret_tuning
            ego_tunings[c_i, lag_ind, :] = ego_tuning

            _, pupil1_tuning, _ = fm2p.tuning_curve(
                spiketrains[c_i,:][np.newaxis,split1_inds],
                pupil[use][split1_inds],
                pupil_bins
            )
            _, pupil2_tuning, _ = fm2p.tuning_curve(
                spiketrains[c_i,:][np.newaxis,split2_inds],
                pupil[use][split2_inds],
                pupil_bins
            )
            pupil_xcorr[c_i, lag_ind], _ = scipy.stats.pearsonr(pupil1_tuning[0], pupil2_tuning[0])

            _, ret1_tuning, _ = fm2p.tuning_curve(
                spiketrains[c_i,:][np.newaxis,split1_inds],
                retinocentric[use][split1_inds],
                retino_bins
            )
            _, ret2_tuning, _ = fm2p.tuning_curve(
                spiketrains[c_i,:][np.newaxis,split2_inds],
                retinocentric[use][split2_inds],
                retino_bins
            )
            retino_xcorr[c_i, lag_ind], _ = scipy.stats.pearsonr(ret1_tuning[0], ret2_tuning[0])

            _, ego1_tuning, _ = fm2p.tuning_curve(
                spiketrains[c_i,:][np.newaxis,split1_inds],
                egocentric[use][split1_inds],
                ego_bins
            )
            _, ego2_tuning, _ = fm2p.tuning_curve(
                spiketrains[c_i,:][np.newaxis,split2_inds],
                egocentric[use][split2_inds],
                ego_bins
            )
            ego_xcorr[c_i, lag_ind], _ = scipy.stats.pearsonr(ego1_tuning[0], ego2_tuning[0])


        axs = axs.flatten()
        for ax in axs:
            ax.set_ylim([0, _maxtuning])
            ax.set_ylabel('sp/s')

        fig.suptitle('cell {}'.format(c_i))

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close()

    # Calculate the best lag value for each cell
    # response_peak = np.zeros([np.size(retino_xcorr,0), 3]) * np.nan
    # xcorr_lag_vals = np.arange(-num_cc_bins//2, (num_cc_bins//2)+1)
    # useccinds = np.zeros([np.size(retino_xcorr,0),3])

    # for c_i in range(np.size(retino_xcorr,0)):
    #     xcorr_across_lags = np.zeros([np.size(retino_xcorr,1),3]) * np.nan
        
    #     for lag_i in range(np.size(retino_xcorr,1)):
    #         xcorr_across_lags[lag_i,0] = xcorr_lag_vals[np.nanargmax(pupil_xcorr[c_i,lag_i,:])]
    #         xcorr_across_lags[lag_i,1] = xcorr_lag_vals[np.nanargmax(retino_xcorr[c_i,lag_i,:])]
    #         xcorr_across_lags[lag_i,2] = xcorr_lag_vals[np.nanargmax(ego_xcorr[c_i,lag_i,:])]
    #         if all_mods[c_i, lag_i, 0, 0] < 0.33:
    #             xcorr_across_lags[lag_i,0] = np.nan
    #         if all_mods[c_i, lag_i, 1, 0] < 0.33:
    #             xcorr_across_lags[lag_i,1] = np.nan
    #         if all_mods[c_i, lag_i, 2, 0] < 0.33:
    #             xcorr_across_lags[lag_i,2] = np.nan
        
    #     for i in range(3):
    #         # if is all NaNs (i.e., no lag had a cc that showed doubling of firing rate)
    #         if np.sum(np.isnan(xcorr_across_lags[:,i])) == len(xcorr_across_lags[:,i]):
    #             response_peak[c_i,i] = np.nan
    #         else:
    #             response_peak[c_i,i] = lag_vals[np.nanargmax(xcorr_across_lags[:,i])]
    #             useccinds[c_i,i] = 1

    # fig, [ax1,ax2,ax3] = plt.subplots(1,3, figsize=(6,2), dpi=300)
    # for c_i in range(np.size(pupil_xcorr,0)):
    #     if useccinds[c_i,0]:
    #         ax1.plot(np.arange(-num_cc_bins//2,num_cc_bins//2), pupil_xcorr[c_i,3,:], alpha=0.4, lw=1)
    #     if useccinds[c_i,1]:
    #         ax2.plot(np.arange(-num_cc_bins//2,num_cc_bins//2), retino_xcorr[c_i,3,:], alpha=0.4, lw=1)
    #     if useccinds[c_i,2]:
    #         ax3.plot(np.arange(-num_cc_bins//2,num_cc_bins//2), ego_xcorr[c_i,3,:], alpha=0.4, lw=1)
    # for ax in [ax1,ax2,ax3]:
    #     ax.set_xlim([-num_cc_bins//2,num_cc_bins//2])
    #     ax.set_ylim([-1,1])
    # ax1.set_title('pupil')
    # fig.suptitle('nanxcorr at 0 ms')
    # fig.tight_layout()
    # pdf.savefig(fig)
    # plt.close()

    fig, axs = plt.subplots(9,3, figsize=(6,10), dpi=300)
    _setmax = 0
    for lag_i, lagval in enumerate(lag_vals):
        lagval = (1/7.49) * 1000 * lagval

        h = axs[lag_i,0].hist(pupil_xcorr[:,lag_i], color='tab:blue', bins=np.linspace(-1,1,18))
        if np.nanmax(h[0]) > _setmax:
            _setmax = np.nanmax(h[0])
        h = axs[lag_i,1].hist(retino_xcorr[:,lag_i], color='tab:orange', bins=np.linspace(-1,1,18))
        if np.nanmax(h[0]) > _setmax:
            _setmax = np.nanmax(h[0])
        h = axs[lag_i,2].hist(ego_xcorr[:,lag_i], color='tab:green', bins=np.linspace(-1,1,18))
        if np.nanmax(h[0]) > _setmax:
            _setmax = np.nanmax(h[0])

        for col in range(3):    
            axs[lag_i,col].set_title('{:.4}ms'.format(lagval))
    axs[0,0].set_title('pupil, lag={:.4}ms'.format((1/7.49) * 1000 * lag_vals[0]))
    axs[0,1].set_title('retinocentric, lag={:.4}ms'.format((1/7.49) * 1000 * lag_vals[0]))
    axs[0,2].set_title('egocentric, lag={:.4}ms'.format((1/7.49) * 1000 * lag_vals[0]))

    axs = axs.flatten()
    for ax in axs:
        ax.set_xlim([-1,1])
        ax.set_ylim([0,_setmax*1.1])
        ax.vlines(0, 0, _setmax*1.1, color='tab:red', alpha=0.5, lw=1)
        ax.set_xlabel('xcorr')
        ax.set_ylabel('cells')
    fig.suptitle('cross correlation between tuning halves')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close()


    fig, [ax1,ax2,ax3] = plt.subplots(1,3, figsize=(6,2), dpi=300)
    _setmax = 0
    for lag_i, lagval in enumerate(lag_vals):
        ax1.bar(lag_i, np.sum(pupil_xcorr[:,lag_i]>0.5), width=1, color='tab:blue')
        ax2.bar(lag_i, np.sum(retino_xcorr[:,lag_i]>0.5), width=1, color='tab:orange')
        ax3.bar(lag_i, np.sum(ego_xcorr[:,lag_i]>0.5), width=1, color='tab:green')

        for x in [np.sum(pupil_xcorr[:,lag_i]>0.5), np.sum(retino_xcorr[:,lag_i]>0.5), np.sum(ego_xcorr[:,lag_i]>0.5)]:
            if x>_setmax:
                _setmax = x

    ax1.set_title('pupil')
    ax2.set_title('retinocentric')
    ax3.set_title('egocentric')
    for ax in [ax1,ax2,ax3]:
        ax.set_xlabel('lag (ms)')
        ax.set_ylabel('cells with xcorr>0.5')
        ax.set_ylim([0,_setmax])
        ax.set_xticks(np.arange(len(lag_vals)), labels=[int((1/7.49) * 1000 * l) for l in lag_vals], rotation=90)
    fig.suptitle('cells above xcorr thresh at each lag')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # fig, [ax1,ax2,ax3] = plt.subplots(1,3, figsize=(6.5,2), dpi=300)
    # for lag_i, lagval in enumerate(lag_vals):
    #     ax1.hist(response_peak[:,0]  * ((1/7.49) * 1000), color='tab:blue')
    #     ax2.hist(response_peak[:,1]  * ((1/7.49) * 1000), color='tab:orange')
    #     ax3.hist(response_peak[:,2]  * ((1/7.49) * 1000), color='tab:green')
    # for ax in [ax1,ax2,ax3]:
    #     ax.set_xlabel('peak lag (ms)')
    #     ax.set_ylabel('cells')
    #     ax.set_xlim([-500,600])
    # pdf.savefig(fig)
    # plt.close()


    pdf.close()


if __name__ == '__main__':
    summarize_revcorr()