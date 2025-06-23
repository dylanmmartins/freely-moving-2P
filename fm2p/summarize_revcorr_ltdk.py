# -*- coding: utf-8 -*-
"""
Summarize cell responses based on reverse correlation receptive fields.

Functions
---------
summarize_revcorr()
    Summarize cell responses based on reverse correlation receptive fields.

Example usage
-------------
    $ python -m fm2p.summarize_revcorr -v 01
or alternatively, leave out the -v flag and select the h5 file from a file dialog box, followed
by the version number in a text box.
    $ python -m fm2p.summarize_revcorr

Author: DMM, 2025
"""


import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import fm2p


def summarize_revcorr_ltdk():
    """ Summarize cell responses based on reverse correlation receptive fields.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--preproc', type=str, default=None)
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
    
    spikes = data['s2p_spks'].copy()
    egocentric = data['egocentric'].copy()
    retinocentric = data['retinocentric'].copy()
    pupil = data['pupil_from_head'].copy()
    speed = data['speed'].copy()
    speed = np.append(speed, speed[-1])
    use = speed > 1.5
    ltdk = data['ltdk_state_vec'].copy()

    # ego_bins = np.linspace(-180, 180, 19)
    # retino_bins = np.linspace(-180, 180, 19) # 20 deg bins
    # ego_bins = np.linspace(-180, 180, 37)
    # retino_bins = np.linspace(-180, 180, 37)
    # # pupil_bins = np.linspace(45, 95, 11) # 5 deg bins
    # pupil_bins = np.linspace(45, 110, 21)

    pupil_bins = np.linspace(45, 110, 21) # 3.25 deg bins (was 5 deg)
    retino_bins = np.linspace(-180, 180, 37) # 10. deg bins (was 20)
    ego_bins = np.linspace(-180, 180, 37)

    lag_vals = [-3,-2,-1,0,1,2,3,4,20]

    # Divide into two recordings: light and dark conditions
    spikes_lt = spikes.copy()[ltdk]
    spikes_dk = spikes.copy()[~ltdk]
    egocentric_lt = egocentric.copy()[ltdk]
    egocentric_dk = egocentric.copy()[~ltdk]
    retinocentric_lt = retinocentric.copy()[ltdk]
    retinocentric_dk = retinocentric.copy()[~ltdk]
    pupil_lt = pupil.copy()[ltdk]
    pupil_dk = pupil.copy()[~ltdk]
    speeduse_lt = use.copy()[ltdk]
    speeduse_dk = use.copy()[~ltdk]

    for state in range(0,2):

        # 0 is dark condition, 1 is light condition
        state = bool(state)

        if state == False:
            spikes_ = spikes_dk
            egocentric_ = egocentric_dk
            retinocentric_ = retinocentric_dk
            pupil_ = pupil_dk
            speeduse_ = speeduse_dk
            ltdkuse_ = ltdk.copy()

            statename = 'dark'

        elif state == True:
            spikes_ = spikes_lt
            egocentric_ = egocentric_lt
            retinocentric_ = retinocentric_lt
            pupil_ = pupil_lt
            speeduse_ = speeduse_lt
            ltdkuse_ = ~ltdk.copy()

            statename = 'light'

        spiketrains = np.zeros([
            np.size(spikes_,0),
            np.sum(speeduse_)
        ]) * np.nan
        
        # break data into 10 chunks, randomly choose ten of them for each block
        ncnk = 10
        _len = np.sum(speeduse_)
        cnk_sz = _len // ncnk
        _all_inds = np.arange(0,_len)
        chunk_order = np.arange(ncnk)
        np.random.shuffle(chunk_order)

        splits_inds = []

        for cnk in chunk_order:
            _inds = _all_inds[(cnk_sz*cnk) : ((cnk_sz*(cnk+1)))]
            splits_inds.append(_inds)

        pupil_tunings = np.zeros([
            np.size(spikes_, 0),
            len(lag_vals),
            len(pupil_bins)-1,
            2                       # {tuning, error}
        ]) * np.nan
        ret_tunings = np.zeros([
            np.size(spikes_, 0),
            len(lag_vals),
            len(retino_bins)-1,
            2
        ]) * np.nan
        ego_tunings = np.zeros([
            np.size(spikes_, 0),
            len(lag_vals),
            len(ego_bins)-1,
            2
        ]) * np.nan

        all_mods = np.zeros([
            np.size(spikes,0),
            len(lag_vals),
            3,                  # {pupil, retino, ego}
            2                   # {modulation index, peak value}
        ]) * np.nan

        savepath, savename = os.path.split(h5_path)
        savename = '{}_revcorrRFs_v{}_{}.pdf'.format(savename.split('_preproc')[0], versionnum, statename)
        pdf = PdfPages(os.path.join(savepath, savename))

        ### BEHAVIORAL OCCUPANCY
        fig, [[ax1,ax2,ax3],[ax4,ax5,ax6]] = plt.subplots(2, 3, dpi=300, figsize=(5.5,3.5))

        ax1.hist(pupil_[speeduse_], bins=pupil_bins, color='tab:blue')
        ax1.set_xlabel('pupil (deg)')
        ax1.set_xlim([pupil_bins[0], pupil_bins[-1]])

        ax2.hist(retinocentric_[speeduse_], bins=retino_bins, color='tab:orange')
        ax2.set_xlabel('retinocentric (deg)')
        ax2.set_xlim([retino_bins[0], retino_bins[-1]])

        ax3.hist(egocentric_[speeduse_], bins=ego_bins, color='tab:green')
        ax3.set_xlabel('egocentric (deg)')
        ax3.set_xlim([ego_bins[0], ego_bins[-1]])

        _showspeed = speed.copy()
        if state:
            _showspeed = _showspeed[ltdk]
        elif not state:
            _showspeed = _showspeed[~ltdk]
        ax4.hist(_showspeed, bins=np.linspace(0,60,20), color='k')
        ax4.set_title('{:.4}% running time'.format((np.sum(speeduse_)/len(ltdkuse_))*100))
        ax4.set_xlabel('speed (cm/s)')

        ax5.plot(data['head_x'][speeduse_], data['head_y'][speeduse_], 'k.', ms=1, alpha=0.3)
        ax5.invert_yaxis()
        ax5.axis('equal')

        ax6.plot(data['theta_interp'][speeduse_], data['phi_interp'][speeduse_], 'k.', ms=1, alpha=0.3)
        ax6.set_xlabel('theta (deg)')
        ax6.set_ylabel('phi (deg)')

        running_frac = len(data['twopT'][[speeduse_]]) / len(data['twopT'][ltdkuse_])
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
            ax1.scatter(np.mean(cell['xpix'])-10, np.mean(cell['ypix'])-10,
                        s=25, facecolors='none', edgecolors='r', alpha=0.5)
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

        cell_corrvals = np.zeros([
            np.size(spikes, 0),
            3,                      # {pupil, retino, ego}
            len(lag_vals)
        ]) * np.nan

        ### SUMMARIZE TUNING OF INDIVIDUAL CELLS
        for c_i in tqdm(range(np.size(spikes, 0))):

            fig, axs = plt.subplots(3, 9, dpi=300, figsize=(15,6))

            _maxtuning = 0

            for lag_ind, lag_val in enumerate(lag_vals):
                
                # for cell_i in range(np.size(spikes,0)):
                spiketrains[c_i,:] = np.roll(spikes[c_i,:], shift=lag_val)[speeduse_]

                pupil_cent, pupil_tuning, pupil_err = fm2p.tuning_curve(
                    spiketrains[c_i,:][np.newaxis,:],
                    pupil[speeduse_],
                    pupil_bins
                )
                ret_cent, ret_tuning, ret_err = fm2p.tuning_curve(
                    spiketrains[c_i,:][np.newaxis,:],
                    retinocentric[speeduse_],
                    retino_bins
                )
                ego_cent, ego_tuning, ego_err = fm2p.tuning_curve(
                    spiketrains[c_i,:][np.newaxis,:],
                    egocentric[speeduse_],
                    ego_bins
                )

                pupil_corr = fm2p.calc_tuning_reliability(
                    spiketrains[c_i, :][np.newaxis,:],
                    pupil[speeduse_],
                    pupil_bins,
                )
                retino_corr = fm2p.calc_tuning_reliability(
                    spiketrains[c_i, :][np.newaxis,:],
                    retinocentric[speeduse_],
                    retino_bins,
                )
                ego_corr = fm2p.calc_tuning_reliability(
                    spiketrains[c_i, :][np.newaxis,:],
                    egocentric[speeduse_],
                    ego_bins,
                )
                cell_corrvals[c_i, 0, lag_ind] = pupil_corr
                cell_corrvals[c_i, 1, lag_ind] = retino_corr
                cell_corrvals[c_i, 2, lag_ind] = ego_corr

                fm2p.plot_tuning(axs[0,lag_ind], pupil_cent, pupil_tuning, pupil_err, 'tab:blue', False)
                fm2p.plot_tuning(axs[1,lag_ind], ret_cent, ret_tuning, ret_err, 'tab:orange', False)
                fm2p.plot_tuning(axs[2,lag_ind], ego_cent, ego_tuning, ego_err, 'tab:green', False)

                lag_str = (1/7.49) * 1000 * lag_val

                Pmod, Ppeak = fm2p.calc_modind(pupil_cent, pupil_tuning, spiketrains[c_i,:])
                if np.isnan(Ppeak):
                    axs[0,lag_ind].set_title('{:.4}ms\nc={:.3}\nmod={:.3}'.format(lag_str, pupil_corr, Pmod))
                else:
                    axs[0,lag_ind].set_title('{:.4}ms\nc={:.3}\nmod={:.3} peak={:.4}\N{DEGREE SIGN}'.format(lag_str, pupil_corr, Pmod, Ppeak))
                
                Rmod, Rpeak = fm2p.calc_modind(ret_cent, ret_tuning, spiketrains[c_i,:])
                if np.isnan(Rpeak):
                    axs[1,lag_ind].set_title('{:.4}ms\nc={:.3}\nmod={:.3}'.format(lag_str, retino_corr, Rmod))
                else:
                    axs[1,lag_ind].set_title('{:.4}ms\nc={:.3}\nmod={:.3} peak={:.4}\N{DEGREE SIGN}'.format(lag_str, retino_corr, Rmod, Rpeak))
                
                Emod, Epeak = fm2p.calc_modind(ego_cent, ego_tuning, spiketrains[c_i,:])
                if np.isnan(Epeak):
                    axs[2,lag_ind].set_title('{:.4}ms\nc={:.3}\nmod={:.3}'.format(lag_str, ego_corr, Emod))
                else:
                    axs[2,lag_ind].set_title('{:.4}ms\nc={:.3}\nmod={:.3} peak={:.4}\N{DEGREE SIGN}'.format(lag_str, ego_corr, Emod, Epeak))

                all_mods[c_i, lag_ind, 0, :] = Pmod, Ppeak
                all_mods[c_i, lag_ind, 1, :] = Rmod, Rpeak
                all_mods[c_i, lag_ind, 2, :] = Emod, Epeak

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

            axs = axs.flatten()
            for ax in axs:
                ax.set_ylim([0, _maxtuning])
                ax.set_ylabel('sp/s')

            fig.suptitle('cell {}'.format(c_i))

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close()

        np.save(os.path.join(savepath, 'cell_corr_values.npy'), cell_corrvals)
        np.save(os.path.join(savepath, 'pupil_tunings.npy'), pupil_tunings)
        np.save(os.path.join(savepath, 'ret_tunings.npy'), ret_tunings)
        np.save(os.path.join(savepath, 'ego_tunings.npy'), ego_tunings)
        np.save(os.path.join(savepath, 'all_modinds.npy'), all_mods)

        # fig, axs = plt.subplots(3,3, dpi=300, figsize=(6,4))
        # axs = axs.flatten()
        # for lag_ind, lag_val in enumerate(lag_vals):
        #     hist = axs[lag_ind].hist(cell_pvals[:,0,lag_ind,0], color='k', bins=np.linspace(0,1,100))
        #     axs[lag_ind].set_xlabel('p value')
        #     axs[lag_ind].set_ylabel('cells')
        #     axs[lag_ind].set_title('lag = {} msec'.format(int(np.round((lag_val*(1/7.5))*1000,0))))
        #     axs[lag_ind].set_xlim([0,0.2])
        #     axs[lag_ind].vlines(0.05, 0, 40, ls='--', color='tab:red', lw=0.5)
        #     axs[lag_ind].set_ylim([0, np.max(hist[0])*1.1])
        # fig.suptitle('pupil')
        # fig.tight_layout()
        # pdf.savefig(fig)
        # plt.close()

        fig, axs = plt.subplots(3,3, dpi=300, figsize=(6,4.25))
        axs = axs.flatten()
        for lag_ind, lag_val in enumerate(lag_vals):
            hist = axs[lag_ind].hist(cell_corrvals[:,0,lag_ind], color='k', bins=np.linspace(-1,1,33))
            axs[lag_ind].set_xlabel('cc')
            axs[lag_ind].set_ylabel('cells')
            axs[lag_ind].set_title('lag = {} msec\n{}>0.5'.format(int(np.round((lag_val*(1/7.5))*1000,0)),np.sum(cell_corrvals[:,0,lag_ind]>0.5)))
            axs[lag_ind].set_xlim([-1,1])
            axs[lag_ind].vlines(0.5, 0, 40, ls='--', color='tab:red', lw=0.5)
            axs[lag_ind].set_ylim([0, np.max(hist[0])*1.1])
        fig.suptitle('pupil')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # fig, axs = plt.subplots(3,3, dpi=300, figsize=(6,4))
        # axs = axs.flatten()
        # for lag_ind, lag_val in enumerate(lag_vals):
        #     hist = axs[lag_ind].hist(cell_pvals[:,1,lag_ind,0], color='k', bins=np.linspace(0,1,100))
        #     axs[lag_ind].set_xlabel('p value')
        #     axs[lag_ind].set_ylabel('cells')
        #     axs[lag_ind].set_title('lag = {} msec'.format(int(np.round((lag_val*(1/7.5))*1000,0))))
        #     axs[lag_ind].set_xlim([0,0.2])
        #     axs[lag_ind].vlines(0.05, 0, 40, ls='--', color='tab:red', lw=0.5)
        #     axs[lag_ind].set_ylim([0, np.max(hist[0])*1.1])
        # fig.suptitle('retinocentric')
        # fig.tight_layout()
        # pdf.savefig(fig)
        # plt.close()

        fig, axs = plt.subplots(3,3, dpi=300, figsize=(6,4.25))
        axs = axs.flatten()
        for lag_ind, lag_val in enumerate(lag_vals):
            hist = axs[lag_ind].hist(cell_corrvals[:,1,lag_ind], color='k', bins=np.linspace(-1,1,33))
            axs[lag_ind].set_xlabel('cc')
            axs[lag_ind].set_ylabel('cells')
            axs[lag_ind].set_title('lag = {} msec\n{}>0.5'.format(int(np.round((lag_val*(1/7.5))*1000,0)),np.sum(cell_corrvals[:,1,lag_ind]>0.5)))
            axs[lag_ind].set_xlim([-1,1])
            axs[lag_ind].vlines(0.5, 0, 40, ls='--', color='tab:red', lw=0.5)
            axs[lag_ind].set_ylim([0, np.max(hist[0])*1.1])
        fig.suptitle('retinocentric')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # fig, axs = plt.subplots(3,3, dpi=300, figsize=(6,4))
        # axs = axs.flatten()
        # for lag_ind, lag_val in enumerate(lag_vals):
        #     hist = axs[lag_ind].hist(cell_pvals[:,2,lag_ind,0], color='k', bins=np.linspace(0,1,100))
        #     axs[lag_ind].set_xlabel('p value')
        #     axs[lag_ind].set_ylabel('cells')
        #     axs[lag_ind].set_title('lag = {} msec'.format(int(np.round((lag_val*(1/7.5))*1000,0))))
        #     axs[lag_ind].set_xlim([0,0.2])
        #     axs[lag_ind].vlines(0.05, 0, 40, ls='--', color='tab:red', lw=0.5)
        #     axs[lag_ind].set_ylim([0, np.max(hist[0])*1.1])
        # fig.suptitle('egocentric')
        # fig.tight_layout()
        # pdf.savefig(fig)
        # plt.close()

        fig, axs = plt.subplots(3,3, dpi=300, figsize=(6,4.25))
        axs = axs.flatten()
        for lag_ind, lag_val in enumerate(lag_vals):
            hist = axs[lag_ind].hist(cell_corrvals[:,2,lag_ind], color='k', bins=np.linspace(-1,1,33))
            axs[lag_ind].set_xlabel('cc')
            axs[lag_ind].set_ylabel('cells')
            axs[lag_ind].set_title('lag = {} msec\n{}>0.5'.format(int(np.round((lag_val*(1/7.5))*1000,0)),np.sum(cell_corrvals[:,2,lag_ind]>0.5)))
            axs[lag_ind].set_xlim([-1,1])
            axs[lag_ind].vlines(0.5, 0, 40, ls='--', color='tab:red', lw=0.5)
            axs[lag_ind].set_ylim([0, np.max(hist[0])*1.1])
        fig.suptitle('pupil')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots(1,1,figsize=(3,2.5),dpi=300)
        ax.vlines(0,0,65,lw=1,ls='--',color='k')
        ax.plot(
            [int(np.round((lag_val*(1/7.5))*1000,0)) for lag_val in lag_vals],
            [np.sum(cell_corrvals[:,0,lag_ind]>0.5) for lag_ind,_ in enumerate(lag_vals)],
            lw=2, color='tab:blue', label='pupil'
        )

        ax.plot(
            [int(np.round((lag_val*(1/7.5))*1000,0)) for lag_val in lag_vals],
            [np.sum(cell_corrvals[:,1,lag_ind]>0.5) for lag_ind,_ in enumerate(lag_vals)],
            lw=2, color='tab:orange', label='retino'
        )
        ax.plot(
            [int(np.round((lag_val*(1/7.5))*1000,0)) for lag_val in lag_vals],
            [np.sum(cell_corrvals[:,2,lag_ind]>0.5) for lag_ind,_ in enumerate(lag_vals)],
            lw=2, color='tab:green', label='ego'
        )
        ax.set_ylim([0,65])
        ax.set_ylabel('cells cc>0.5')
        ax.set_xlabel('lag (msec)')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close()

        pdf.close()


if __name__ == '__main__':

    summarize_revcorr_ltdk()