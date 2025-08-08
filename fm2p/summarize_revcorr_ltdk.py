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
    parser.add_argument('-p', '--path', type=str, default=None)
    parser.add_argument('-v', '--version', type=str, default='01')
    args = parser.parse_args()
    revcorr_path = args.path
    versionnum = args.version

    if preproc is None:
        revcorr_path = fm2p.select_file(
            title='Choose a revcorr HDF file.',
            filetypes=[('H5','.h5'),]
        )
        versionnum = fm2p.get_string_input(
            title='Enter summary version number (e.g., 01).'
        )

    print('  -> Making summary documents for {}'.format(revcorr_path))

    data = fm2p.read_h5(revcorr_path)
    
    retino_bins = np.linspace(-180, 180, 27) # 14 deg bins
    ego_bins = np.linspace(-180, 180, 27)

    lag_vals = [-3,-2,-1,0,1,2,3,4,20]

    # Divide into two recordings: light and dark conditions

    for state in range(0,2):

        # 0 is dark condition, 1 is light condition
        state = bool(state)

        if state == 0:
            statename = 'dark'
            speeduse_ = use.copy()[~ltdk]
            ltdkuse_ = (~ltdk.copy()) * use.copy()
            ltdk_plain_ = ~ltdk.copy()
            spikes_ = spikes.copy()[:,~ltdk]
            egocentric_ = egocentric.copy()[~ltdk]
            retinocentric_ = retinocentric.copy()[~ltdk]

        elif state == 1:
            statename = 'light'
            speeduse_ = use.copy()[ltdk]
            ltdkuse_ = (ltdk.copy()) * use.copy()
            ltdk_plain_ = ltdk.copy()
            spikes_ = spikes.copy()[:,ltdk]
            egocentric_ = egocentric.copy()[ltdk]
            retinocentric_ = retinocentric.copy()[ltdk]

        pupil_ = (theta.copy() - np.nanmean(theta))[ltdkuse_]

        pupil_bins = np.linspace(
            np.nanpercentile(pupil_, 5),
            np.nanpercentile(pupil_, 95),
            13
        )

        print('  -> Calculating tunings and making summary PDF for {} state.'.format(statename))

        spiketrains = np.zeros([
            np.size(spikes_, 0),
            np.sum(ltdkuse_)
        ]) * np.nan
        
        # break data into 10 chunks, randomly choose ten of them for each block
        ncnk = 10
        _len = np.sum(ltdkuse_)
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
            2,                   # {tuning, error}
            len(pupil_bins)-1                 
        ]) * np.nan
        ret_tunings = np.zeros([
            np.size(spikes_, 0),
            len(lag_vals),
            2,
            len(retino_bins)-1
        ]) * np.nan
        ego_tunings = np.zeros([
            np.size(spikes_, 0),
            len(lag_vals),
            2,
            len(ego_bins)-1,
        ]) * np.nan

        all_mods = np.zeros([
            np.size(spikes,0),
            len(lag_vals),
            3,                  # {pupil, retino, ego}
            2                   # {modulation index, peak value}
        ]) * np.nan

        savepath, savename = os.path.split(preproc)
        savename = '{}_revcorrRFs_v{}_{}.pdf'.format(savename.split('_preproc')[0], versionnum, statename)
        pdfsavepath = os.path.join(savepath, savename)
        pdf = PdfPages(pdfsavepath)

        ### BEHAVIORAL OCCUPANCY
        fig, [[ax1,ax2,ax3],[ax4,ax5,ax6]] = plt.subplots(2, 3, dpi=300, figsize=(5.5,3.5))

        ax1.hist(pupil_, bins=pupil_bins, color='tab:blue')
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
        ax4.set_title('{:.4}% running time'.format((np.sum(ltdkuse_)/len(ltdk_plain_))*100))
        ax4.set_xlabel('speed (cm/s)')

        ax5.plot(data['head_x'][ltdkuse_], data['head_y'][ltdkuse_], 'k.', ms=1, alpha=0.3)
        ax5.invert_yaxis()
        ax5.axis('equal')

        ax6.plot(data['theta_interp'][ltdkuse_], data['phi_interp'][ltdkuse_], 'k.', ms=1, alpha=0.3)
        ax6.set_xlabel('theta (deg)')
        ax6.set_ylabel('phi (deg)')

        running_frac = len(data['twopT'][ltdkuse_]) / len(data['twopT'][ltdk_plain_])
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
            np.size(spikes_, 0),
            3,                      # {pupil, retino, ego}
            len(lag_vals)
        ]) * np.nan

        ### SUMMARIZE TUNING OF INDIVIDUAL CELLS
        for c_i in tqdm(range(np.size(spikes_, 0))):

            fig, axs = plt.subplots(3, 9, dpi=300, figsize=(15,6))

            _maxtuning = 0

            for lag_ind, lag_val in enumerate(lag_vals):
                
                # Must into the sectiosn of the recording for this light/dark state AFTER
                # applying the temporal roll. otherwise, rolls will be non-continuous jumps in time.
                spiketrains[c_i,:] = np.roll(spikes[c_i,:], shift=lag_val)[ltdkuse_]

                pupil_cent, pupil_tuning, pupil_err = fm2p.tuning_curve(
                    spiketrains[c_i,:][np.newaxis,:],
                    pupil_,
                    pupil_bins
                )
                ret_cent, ret_tuning, ret_err = fm2p.tuning_curve(
                    spiketrains[c_i,:][np.newaxis,:],
                    retinocentric[ltdkuse_],
                    retino_bins
                )
                ego_cent, ego_tuning, ego_err = fm2p.tuning_curve(
                    spiketrains[c_i,:][np.newaxis,:],
                    egocentric[ltdkuse_],
                    ego_bins
                )

                pupil_corr = fm2p.calc_tuning_reliability(
                    spiketrains[c_i, :][np.newaxis,:],
                    pupil_,
                    pupil_bins,
                )
                retino_corr = fm2p.calc_tuning_reliability(
                    spiketrains[c_i, :][np.newaxis,:],
                    retinocentric[ltdkuse_],
                    retino_bins,
                )
                ego_corr = fm2p.calc_tuning_reliability(
                    spiketrains[c_i, :][np.newaxis,:],
                    egocentric[ltdkuse_],
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

                if lag_val != lag_vals[-1]:
                    for x in [
                        np.nanmax(pupil_tuning+pupil_err),
                        np.nanmax(ret_tuning+ret_err),
                        np.nanmax(ego_tuning+ego_err)]:
                        if x > _maxtuning:
                            _maxtuning = x

                axs[1,lag_ind].vlines([-75,75], 0, _maxtuning, color='k', alpha=0.3, ls='--', lw=1)

                pupil_tunings[c_i, lag_ind, 0, :] = np.squeeze(pupil_tuning.T)
                ret_tunings[c_i, lag_ind, 0, :] =  np.squeeze(ret_tuning.T)
                ego_tunings[c_i, lag_ind, 0, :] =  np.squeeze(ego_tuning.T)

                pupil_tunings[c_i, lag_ind, 1, :] = np.squeeze(pupil_err.T)
                ret_tunings[c_i, lag_ind, 1, :] =  np.squeeze(ret_err.T)
                ego_tunings[c_i, lag_ind, 1, :] =  np.squeeze(ego_err.T)


            axs = axs.flatten()
            for ax in axs:
                ax.set_ylim([0, _maxtuning])
                ax.set_ylabel('sp/s')

            fig.suptitle('cell {}'.format(c_i))

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close()

        tuning_data = {
            'cell_corrvals': cell_corrvals,
            'pupil_tunings': pupil_tunings,
            'retino_tunings': ret_tunings,
            'ego_tunings': ego_tunings,
            'all_mods': all_mods,
            'pupil_centers': pupil_cent,
            'retino_centers': ret_cent,
            'ego_centers': ego_cent
        }

        h5savepath = os.path.join(savepath, 'tuning_data_{}_v{}.h5'.format(statename, versionnum))
        fm2p.write_h5(h5savepath, tuning_data)
        print('  -> Saved {}'.format(h5savepath))

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

        print('  -> Closed {}'.format(pdfsavepath))


if __name__ == '__main__':

    summarize_revcorr_ltdk()