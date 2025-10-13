import numpy as np
import pandas as pd
import fm2p
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def make_filler_series(nc, ni):
    # nc is number of number of cells
    # ni is number of entries of the object
    s = pd.Series(np.zeros(nc))
    for i in s.index.values:
        s.iloc[i] = (np.zeros(ni)*np.nan).astype(object)
    return s


rmse = lambda y, y_pred: ((sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred)) / len(y)) ** 0.5)


def calc_heatmap(data, c):
    x_vals = data['theta_interp']
    y_vals = data['phi_interp']
    rates = data['norm_spikes'][c,:]
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

    return heatmap, x_bins, y_bins


def pool_recordings():

    rec_list = [
        {
            'dir': r'Y:\Mini2P_data\250616_DMM_DMM042_ltdk\fm2',
            'AP': 0,
            'ML': 0
        },
        {
            'dir': r'Y:\Mini2P_data\250618_DMM_DMM042_ltdk\fm2',
            'AP': -4,
            'ML': 4
        },
        {
            'dir': r'Y:\Mini2P_data\250619_DMM_DMM042_ltdk\fm3',
            'AP': -4,
            'ML': 4
        },
        {
            'dir': r'Y:\Mini2P_data\250626_DMM_DMM037_ltdk\fm1',
            'AP': 0,
            'ML': 0
        },
        {
            'dir': r'Y:\Mini2P_data\250626_DMM_DMM041_ltdk\fm1',
            'AP': 0,
            'ML': 0
        },
        {
            'dir': r'Y:\Mini2P_data\250627_DMM_DMM037_ltdk\fm3', 
            'AP': -4,
            'ML': 0
        },
        {
            'dir': r'Y:\Mini2P_data\250627_DMM_DMM037_ltdk\fm5',
            'AP': -8,
            'ML': 0
        },
        {
            'dir': r'Y:\Mini2P_data\250627_DMM_DMM041_ltdk\fm1',
            'AP': -4,
            'ML': 0
        },
        {
            'dir': r'Y:\Mini2P_data\250627_DMM_DMM042_ltdk\fm2',
            'AP': -4,
            'ML': 0
        },
        {
            'dir': r'Y:\Mini2P_data\250627_DMM_DMM042_ltdk\fm3',
            'AP': -8,
            'ML': 0
        },
        {
            'dir': r'Y:\Mini2P_data\250628_DMM_DMM037_ltdk\fm2',
            'AP': -4,
            'ML': -4
        },
        {
            'dir': r'Y:\Mini2P_data\250628_DMM_DMM037_ltdk\fm3',
            'AP': 4,
            'ML': 0
        },
        {
            'dir': r'Y:\Mini2P_data\250628_DMM_DMM041_ltdk\fm2',
            'AP': 4,
            'ML': 0
        },
        {
            'dir': r'Y:\Mini2P_data\250628_DMM_DMM041_ltdk\fm4',
            'AP': -4,
            'ML': -4
        },
        {
            'dir': r'Y:\Mini2P_data\250628_DMM_DMM042_ltdk\fm3',
            'AP': -4,
            'ML': -4
        },
        {
            'dir': r'Y:\Mini2P_data\250628_DMM_DMM042_ltdk\fm4',
            'AP': 4,
            'ML': 0
        },
        {
            'dir': r'Y:\Mini2P_data\250630_DMM_DMM037_ltdk\fm1',
            'AP': 0,
            'ML': -4
        },
        {
            'dir': r'Y:\Mini2P_data\250630_DMM_DMM037_ltdk\fm3',
            'AP': +8,
            'ML': 0
        },
        {
            'dir': r'Y:\Mini2P_data\250630_DMM_DMM041_ltdk\fm1',
            'AP': 8,
            'ML': 0
        },
        {
            'dir': r'Y:\Mini2P_data\250630_DMM_DMM041_ltdk\fm3',
            'AP': 0,
            'ML': -4
        }
    ]
    
    merged_dict = {}

    for i, v in tqdm(enumerate(rec_list)):

        dir_ = v['dir']
        AP = v['AP']
        ML = v['ML']

        base_name = dir_.split('\\')[-2][:-5]
        rnum = dir_.split('\\')[-1][-1]
        revcorr_path = '{}_fm_{:02d}_revcorr_results_v4.h5'.format(base_name, int(rnum))
        if not os.path.isfile(os.path.join(dir_, revcorr_path)):
            revcorr_path = '{}_fm_{:02d}_revcorr_results_v3.h5'.format(base_name, int(rnum))
            if not os.path.isfile(os.path.join(dir_, revcorr_path)):
                revcorr_path = '{}_fm_{:02d}_revcorr_results.h5'.format(base_name, int(rnum))
        preproc_path = '{}_fm_{:02d}_preproc.h5'.format(base_name, int(rnum))
        animal = base_name.split('_')[2]
        date = base_name.split('_')[0]
        
        revcorr_data = fm2p.read_h5(os.path.join(dir_, revcorr_path))
        preproc_data = fm2p.read_h5(os.path.join(dir_, preproc_path))

        peth_dict = fm2p.calc_PETHs(preproc_data)

        n_cells = np.size(preproc_data['norm_spikes'], 0)

        for c in range(n_cells):
            # create a dict per cell
            merged_dict['{}_cell{:03d}'.format(base_name, c)] = {}

            heatmap, xbins, ybins = calc_heatmap(preproc_data, i)
            merged_dict['{}_cell{:03d}'.format(base_name, c)]['th_ph_heatmap'] = heatmap
            merged_dict['{}_cell{:03d}'.format(base_name, c)]['th_ph_heatX'] = xbins
            merged_dict['{}_cell{:03d}'.format(base_name, c)]['th_ph_heatY'] = ybins

            merged_dict['{}_cell{:03d}'.format(base_name, c)]['animal'] = animal
            merged_dict['{}_cell{:03d}'.format(base_name, c)]['date'] = date
            merged_dict['{}_cell{:03d}'.format(base_name, c)]['rec_num'] = rnum
            merged_dict['{}_cell{:03d}'.format(base_name, c)]['AP'] = AP
            merged_dict['{}_cell{:03d}'.format(base_name, c)]['ML'] = ML

        for state in ['light', 'dark']:
            for key in ['distance_to_pillar', 'egocentric', 'phi', 'retinocentric', 'theta', 'yaw']:
                
                # s = make_filler_series(n_cells, 12)

                for c in range(n_cells):
                    try:
                        revcorr_tmp = revcorr_data[state][key]['tuning_curve'][c,:]
                    except KeyError:
                        try:
                            revcorr_tmp = revcorr_data[state][key]['tunings'][c,:]
                        except KeyError:
                            revcorr_tmp = np.ones(12) * np.nan
                            try:
                                merged_dict['{}_cell{:03d}'.format(base_name, c)]['{}_{}_tuning_curve'.format(state, key)] = revcorr_tmp
                            except:
                                pass
                            try:
                                merged_dict['{}_cell{:03d}'.format(base_name, c)]['{}_{}_tuning_bins'.format(state, key)] = revcorr_data[state][key]['tuning_bins']
                            except:
                                pass
                            try:
                                merged_dict['{}_cell{:03d}'.format(base_name, c)]['{}_{}_tuning_error'.format(state, key)] = revcorr_data[state][key]['tuning_stderr'][c,:]
                            except:
                                pass

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
            
            for c in range(n_cells):
                merged_dict['{}_cell{:03d}'.format(base_name, c)][k] = peth_dict[k][c,:]

        for c in range(n_cells):
            merged_dict['{}_cell{:03d}'.format(base_name, c)]['cell_xloc'] = np.median(preproc_data['cell_x_pix'][str(c)])
            merged_dict['{}_cell{:03d}'.format(base_name, c)]['cell_yloc'] = np.median(preproc_data['cell_y_pix'][str(c)])

        for c in range(n_cells):
            merged_dict['{}_cell{:03d}'.format(base_name, c)]['norm_spikes'] = (preproc_data['norm_spikes'][c,:])
            merged_dict['{}_cell{:03d}'.format(base_name, c)]['raw_spikes'] = (preproc_data['s2p_spks'][c,:])
            merged_dict['{}_cell{:03d}'.format(base_name, c)]['norm_dFF'] = (preproc_data['norm_dFF'][c,:])
            merged_dict['{}_cell{:03d}'.format(base_name, c)]['raw_F'] = (preproc_data['raw_F'][c,:])

        vars_to_add = [
            'ltdk_state_vec',
            'head_yaw_deg',
            'retinocentric',
            'egocentric',
            'head_x',
            'head_y',
            'longaxis',
            'theta_interp',
            'phi_interp',
            'twopT',
            'theta',
            'phi',
            'eyeT'
        ]
        for var in vars_to_add:
            for c in range(n_cells):
                try:
                    merged_dict['{}_cell{:03d}'.format(base_name, c)][var] = preproc_data[var]
                except:
                    pass

    fm2p.write_h5(
        r'T:\dylan\merged_V1PPC_dataset_w251012_v2.h5',
        merged_dict
    )



if __name__ == '__main__':

    pool_recordings()