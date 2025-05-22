""" Run GLM models.
ret2ego/utils/run_model.py

Functions
---------
fit_GLM_to_toy_data
    Fit GLM to toy data.
fit_GLM_to_neuron
    Fit GLM to neuron data.
fit_GLM_to_neuron_2feat
    Fit GLM to neuron data with 2 features.
fit_GLM_to_neuron_4feat
    Fit GLM to neuron data with 4 features.


Written by DMM, Oct 2023
"""

import numpy as np

import fmEphys as fme
import ret2ego as rte


def fit_GLM_to_toy_data(inputs):
    """Fit GLM to toy data.
    """

    # behavior data
    gaze = inputs['gazeA2P']
    retino = inputs['retinoA2P']
    ego = inputs['egoA2P']
    d = inputs['dA2P']

    # interpolate over nans
    gaze = fme.fill_NaNs(gaze)
    retino = fme.fill_NaNs(retino)
    ego = fme.fill_NaNs(ego)
    d = fme.fill_NaNs(d)

    use_d_bins = np.linspace(2,11,7)
    use_ego_bins = np.deg2rad(np.arange(-20,200,20))
    use_retino_bins = np.deg2rad(np.arange(-75, 85, 15))
    use_gaze_bins = np.deg2rad(np.arange(30,81,5))

    # make a receptive field for a toy cell
    rf1 = rte.make_empty_RF()
    rf1[2,3] = .8
    rf1[2,2] = 1.
    rf1[2,4] = .2
    rf1[3,3] = .75
    rf1[1,2] = .6
    rf1[3,2] = .22
    rf1[1,4] = .21
    rf1[1,3] = .33
    rf1[0,0] = .36
    rf1[0,2] = .37
    rf1[0,4] = .20
    rf1[0,3] = .39
    rf1 = rf1.T

    rf2 = rte.make_empty_RF().T.copy()

    rf3 = rte.make_empty_RF().T.copy()
    # rf3 = rte.flatten_RF(rf3)

    maxFr = 1

    spikes_rt = rte.pred_spike_train(rf1, retino, d, maxFR=1, use_ego_bins=use_retino_bins)
    spikes_eg = rte.pred_spike_train(rf2, ego, d, maxFR=1, use_ego_bins=use_ego_bins)
    spikes_gz = rte.pred_spike_train(rf3, gaze, d, maxFR=1, use_ego_bins=use_gaze_bins)
    spikes = (spikes_rt.copy() + spikes_eg.copy() + spikes_gz.copy()) * maxFr

    toy_data = {
        0: {
            'spikes': spikes,
            'retinoA2P': spikes_rt,
            'egoA2P': spikes_eg,
            'gazeA2P': spikes_gz,
        }
    }

    outputs = rte.fit_pred_GLM_3feat(toy_data)

    outputs['spikes_rt'] = spikes_rt
    outputs['spikes_eg'] = spikes_eg
    outputs['spikes_gz'] = spikes_gz

    w_sum = np.sum([
        outputs['weights_gz'][0],
        outputs['weights_ret'][0],
        outputs['weights_ego'][0]
    ])

    outputs['weights_ret'] /= w_sum
    outputs['weights_ego'] /= w_sum
    outputs['weights_gz'] /= w_sum

    outputs['rf_ret'] = rf1
    outputs['rf_eg'] = rf2
    outputs['rf_gz'] = rf3
    outputs['spikes'] = spikes

    hist1 = rte.calc_radhist(retino, d, spikes, use_retino_bins).T
    hist2 = rte.calc_radhist(ego, d, spikes, use_ego_bins).T
    hist3 = rte.calc_radhist(gaze, d, spikes, use_gaze_bins)
    # hist3 = rte.flatten_RF(hist3, RM=True).T

    max_hist = np.nanmax([np.nanmax(hist1), np.nanmax(hist2), np.nanmax(hist3)])
    hist1 /= max_hist
    hist2 /= max_hist
    hist3 /= max_hist

    hist1_ = hist1 * outputs['weights_ret']
    hist2_ = hist2 * outputs['weights_ego']
    hist3_ = hist3 * outputs['weights_gz']

    outputs['hist_rt'] = hist1
    outputs['hist_eg'] = hist2
    outputs['hist_gz'] = hist3

    outputs['w_hist_rt'] = hist1_
    outputs['w_hist_eg'] = hist2_
    outputs['w_hist_gz'] = hist3_
    
    return outputs


def fit_GLM_to_neuron(inputs):

    use_d_bins = np.linspace(2,11,30)
    use_ego_bins = np.deg2rad(np.linspace(-20, 200, 80))
    use_retino_bins = np.deg2rad(np.linspace(-75, 75, 70))
    use_gaze_bins = np.deg2rad(np.linspace(30, 90, 70))

    # behavior data
    gaze = inputs['gazeA2P']
    retino = inputs['retinoA2P']
    ego = inputs['egoA2P']
    d = inputs['dA2P']
    spikes = inputs['spikes']

    # Mask out NaNs
    mask = rte.mask_helper([gaze, retino, ego, d, spikes])

    gaze = gaze[mask]
    retino = retino[mask]
    ego = ego[mask]
    d = d[mask]
    spikes = spikes[mask]

    # calculate receptive fields
    xx_rt, yy_rt, rf_rt = rte.calc_KDE_radhist(retino, d, spikes, use_retino_bins, use_d_bins)
    xx_eg, yy_eg, rf_eg = rte.calc_KDE_radhist(ego, d, spikes, use_ego_bins, use_d_bins)
    xx_gz, yy_gz, rf_gz = rte.calc_KDE_radhist(gaze, d, spikes, use_gaze_bins, use_d_bins)

    # get individual spike rates from revcorr
    spikes_rt = rte.pred_spike_train(rf_rt, retino, d,
                                    xx_rt, yy_rt)
    
    spikes_eg = rte.pred_spike_train(rf_eg, ego, d,
                                    xx_eg, yy_eg)
    
    spikes_gz = rte.pred_spike_train(rf_gz, gaze, d,
                                    xx_gz, yy_gz)
    
    model_data = {
        0: {
            'spikes': spikes,
            'retinoA2P': spikes_rt,
            'egoA2P': spikes_eg,
            'gazeA2P': spikes_gz,
        }
    }

    outputs = rte.fit_pred_GLM_3feat(model_data)

    outputs['rf_rt'] = rf_rt
    outputs['rf_eg'] = rf_eg
    outputs['rf_gz'] = rf_gz
    
    outputs['spikes_rt'] = spikes_rt
    outputs['spikes_eg'] = spikes_eg
    outputs['spikes_gz'] = spikes_gz

    hist1_ = rf_rt * outputs['weights_ret']
    hist2_ = rf_eg * outputs['weights_ego']
    hist3_ = rf_gz * outputs['weights_gz']

    outputs['w_hist_rt'] = hist1_
    outputs['w_hist_eg'] = hist2_
    outputs['w_hist_gz'] = hist3_

    outputs['xx_rt'] = xx_rt
    outputs['xx_eg'] = xx_eg
    outputs['xx_gz'] = xx_gz
    outputs['yy_rt'] = yy_rt
    outputs['yy_eg'] = yy_eg
    outputs['yy_gz'] = yy_gz

    return outputs


def fit_GLM_to_neuron_2feat(inputs):

    use_d_bins = np.linspace(2,11,30)
    use_ego_bins = np.deg2rad(np.linspace(-20,200,80))
    use_retino_bins = np.deg2rad(np.linspace(-75, 75, 70))

    # behavior data
    # gaze = inputs['gazeA2P']
    retino = inputs['retinoA2P']
    ego = inputs['egoA2P']
    d = inputs['dA2P']

    mask = rte.mask_helper([retino, ego, d])
    # gaze = gaze[mask]
    retino = retino[mask]
    ego = ego[mask]
    d = d[mask]

    # spike data
    spikes = inputs['spikes']

    spikes = np.nan_to_num(spikes, copy=False)

    # calculate receptive fields
    xx_rt, yy_rt, rf_rt = rte.calc_KDE_radhist(retino, d, spikes, use_retino_bins)
    xx_eg, yy_eg, rf_eg = rte.calc_KDE_radhist(ego, d, spikes, use_ego_bins)
    # rf_gz = rte.flatten_RF(rf_gz, sum=True)

    max_rf = np.nanmax([
        np.nanmax(rf_rt),
        np.nanmax(rf_eg)
    ])
    
    rf_rt /= max_rf
    rf_eg /= max_rf

    # get individual spike rates from revcorr
    spikes_rt = rte.pred_spike_train(rf_rt, retino, d,
                                    xx_rt, yy_rt)
    
    spikes_eg = rte.pred_spike_train(rf_eg, ego, d,
                                    xx_eg, yy_eg)
    
    model_data = {
        0: {
            'spikes': spikes,
            'retinoA2P': spikes_rt,
            'egoA2P': spikes_eg,
        }
    }

    outputs = rte.fit_pred_GLM_2feat(model_data)

    w_sum = np.sum([
        outputs['weights_ret'][0],
        outputs['weights_ego'][0]
    ])
    w_sum = float(w_sum)

    outputs['weights_ret'] = outputs['weights_ret'] / w_sum
    outputs['weights_ego'] = outputs['weights_ego'] / w_sum

    outputs['rf_rt'] = rf_rt
    outputs['rf_eg'] = rf_eg
    
    outputs['spikes_rt'] = spikes_rt
    outputs['spikes_eg'] = spikes_eg

    hist1_ = rf_rt * outputs['weights_ret']
    hist2_ = rf_eg * outputs['weights_ego']

    outputs['w_hist_rt'] = hist1_
    outputs['w_hist_eg'] = hist2_

    outputs['xx_rt'] = xx_rt
    outputs['xx_eg'] = xx_eg
    outputs['yy_rt'] = yy_rt
    outputs['yy_eg'] = yy_eg

    return outputs


def fit_GLM_to_neuron_4feat(inputs):

    use_d_bins = np.linspace(2,11,30)
    use_ego_bins = np.deg2rad(np.linspace(-20,200,80))
    use_retino_bins = np.deg2rad(np.linspace(-75, 75, 70))
    use_gaze_bins = np.deg2rad(np.linspace(20,90,70))

    # behavior data
    gaze = inputs['gazeA2P']
    retino = inputs['retinoA2P']
    ego = inputs['egoA2P']
    dist = inputs['distA2P']

    # interpoalte nans
    gaze = fme.fill_NaNs(gaze)
    retino = fme.fill_NaNs(retino)
    ego = fme.fill_NaNs(ego)
    dist = fme.fill_NaNs(dist)

    mask = rte.mask_helper([gaze, retino, ego, dist])
    gaze = gaze[mask]
    retino = retino[mask]
    ego = ego[mask]
    dist = dist[mask]

    # spike data
    spikes = inputs['spikes']

    spikes = np.nan_to_num(spikes, copy=False)

    # calculate receptive fields
    rf_rt = rte.calc_KDE_tuning_curve(retino, spikes, use_retino_bins)
    rf_eg = rte.calc_KDE_tuning_curve(ego, spikes, use_ego_bins)
    rf_gz = rte.calc_KDE_tuning_curve(gaze, spikes, use_gaze_bins)
    rf_ds = rte.calc_KDE_tuning_curve(dist, spikes, use_d_bins)
    # rf_gz = rte.flatten_RF(rf_gz, sum=True)

    max_rf = np.nanmax([
        np.nanmax(rf_rt),
        np.nanmax(rf_eg),
        np.nanmax(rf_gz),
        np.nanmax(rf_ds)
    ])
    
    rf_rt /= max_rf
    rf_eg /= max_rf
    rf_gz /= max_rf
    rf_ds /= max_rf

    # get individual spike rates from revcorr RFs
    spikes_rt = rte.pred_spike_train_1D(rf_rt, retino, use_retino_bins)
    
    spikes_eg = rte.pred_spike_train_1D(rf_eg, ego, use_ego_bins)
    
    spikes_gz = rte.pred_spike_train_1D(rf_gz, gaze, use_gaze_bins)

    spikes_ds = rte.pred_spike_train_1D(rf_ds, dist, use_d_bins)
    
    model_data = {
        0: {
            'spikes': spikes,
            'retinoA2P': spikes_rt,
            'egoA2P': spikes_eg,
            'gazeA2P': spikes_gz,
            'distA2P': spikes_ds
        }
    }

    outputs = rte.fit_pred_GLM_4feat(model_data)

    w_sum = np.sum([
        outputs['weights_gz'][0],
        outputs['weights_ret'][0],
        outputs['weights_ego'][0],
        outputs['weights_dst'][0]
    ])
    w_sum = float(w_sum)

    outputs['weights_ret'] = outputs['weights_ret'] / w_sum
    outputs['weights_ego'] = outputs['weights_ego'] / w_sum
    outputs['weights_gz'] = outputs['weights_gz'] / w_sum
    outputs['weights_dst'] = outputs['weights_dst'] / w_sum

    outputs['rf_rt'] = rf_rt
    outputs['rf_eg'] = rf_eg
    outputs['rf_gz'] = rf_gz
    outputs['rf_ds'] = rf_ds
    
    outputs['spikes_rt'] = spikes_rt
    outputs['spikes_eg'] = spikes_eg
    outputs['spikes_gz'] = spikes_gz
    outputs['spikes_ds'] = spikes_ds

    hist1_ = rf_rt * outputs['weights_ret']
    hist2_ = rf_eg * outputs['weights_ego']
    hist3_ = rf_gz * outputs['weights_gz']
    hist4_ = rf_ds * outputs['weights_dst']

    outputs['w_hist_rt'] = hist1_
    outputs['w_hist_eg'] = hist2_
    outputs['w_hist_gz'] = hist3_
    outputs['w_hist_ds'] = hist4_

    outputs['xx_rt'] = use_retino_bins
    outputs['xx_eg'] = use_ego_bins
    outputs['xx_gz'] = use_gaze_bins
    outputs['xx_ds'] = use_d_bins

    return outputs