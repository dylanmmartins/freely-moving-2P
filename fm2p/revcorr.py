import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import fm2p


def calc_revcorr(preproc_path, axons=False):
    # for a recording that does not have light/dark periods

    data = fm2p.read_h5(preproc_path)

    spikes = data['norm_spikes'].copy()

    theta = data['theta_interp'].copy()
    phi = data['phi_interp'].copy()

    retinocentric = data['retinocentric'].copy()
    egocentric = data['egocentric'].copy()

    distance = data['dist_to_pillar'].copy()
    cdistance = data['dist_to_center'].copy()
    pillar_size = data['pillar_size'].copy()
    yaw = data['head_yaw_deg'][:-1].copy()

    speed = data['speed'].copy()
    speeduse = speed > 1.5

    retino_bins = np.linspace(-180, 180, 27)
    ego_bins = np.linspace(-180, 180, 27)
    yaw_bins = np.linspace(-180, 180, 27)
    theta_bins = np.linspace(
        np.nanpercentile(theta, 5),
        np.nanpercentile(theta, 95),
        13
    )
    phi_bins = np.linspace(
        np.nanpercentile(phi, 5),
        np.nanpercentile(phi, 95),
        13
    )
    dist_bins = np.linspace(
        np.nanpercentile(distance, 5),
        np.nanpercentile(distance, 95),
        13
    )
    cdist_bins = np.linspace(
        np.nanpercentile(cdistance, 5),
        np.nanpercentile(cdistance, 95),
        13
    )
    psize_bins = np.linspace(
        np.nanpercentile(pillar_size, 5),
        np.nanpercentile(pillar_size, 95),
        13
    )

    vardict = {
        'yaw': {
            'vec': yaw,
            'bins': yaw_bins
        },
        'theta': {
            'vec': theta,
            'bins': theta_bins
        },
        'phi': {
            'vec': phi,
            'bins': phi_bins
        },
        'retinocentric': {
            'vec': retinocentric,
            'bins': retino_bins
        },
        'egocentric': {
            'vec': egocentric,
            'bins': ego_bins
        },
        'distance_to_pillar': {
            'vec': distance,
            'bins': dist_bins
        }
        # 'distance_to_center': {
        #     'vec': cdistance,
        #     'bins': cdist_bins
        # },
        # 'pillar_size': {
        #     'vec': pillar_size,
        #     'bins': psize_bins
        # }
    }

    reliability_dict = {}
    
    for k,v in vardict.items():

        print('  -> Calculating reliability for tuning to: {}'.format(k))

        behavior = v['vec']
        bins = v['bins']

        add_dict = fm2p.calc_reliability_d(
            spikes[:,speeduse],
            behavior[speeduse],
            bins,
            10,
            100
        )

        tbins, tunings, errors = fm2p.tuning_curve(
            spikes[:,speeduse],
            behavior[speeduse],
            bins
        )
        add_dict['tuning_bins'] = tbins
        add_dict['tuning_curve'] = tunings
        add_dict['tuning_stderr'] = errors

        # In addition to the shufffle reliability metric, check the spectral slope of power across
        # frequencies; smooth curves will decay steeply while noisy curves will have a flat or shallow
        # decay. A clean curve will have a value of -2 to -5. Noisy will be -1 and above. Don't want to
        # be too strict, so I'm applying the threshold of -1.25, which seems to exclude the appropriate
        # curves while including clean responses.
        spec_val, spec_rel = fm2p.calc_spectral_noise(
            tunings,
            thresh=-1.25
        )
        add_dict['reliable_by_noise'] = spec_rel
        add_dict['spectral_noise'] = spec_val

        is_reliable = spec_rel.copy() * add_dict['reliable_by_shuffle'].copy()
        add_dict['is_reliable'] = is_reliable

        mod, is_modulated = fm2p.calc_multicell_modulation(
            tunings,
            spikes[:,speeduse],
            0.33
        )

        add_dict['modulation'] = mod
        add_dict['is_modulated'] = is_modulated

        reliability_dict[k] = add_dict

    savedir = os.path.split(preproc_path)[0]
    basename = os.path.split(preproc_path)[1][:-11]
    savepath = os.path.join(savedir, '{}_revcorr_results.h5'.format(basename))
    fm2p.write_h5(savepath, reliability_dict)

    print('Saved {}'.format(savepath))



def calc_revcorr_ltdk(preproc_path, restrict_by_deviation=False):
    # only for light/dark recordings

    data = fm2p.read_h5(preproc_path)


    spikes = data['norm_spikes'].copy()

    theta = data['theta_interp'].copy()
    phi = data['phi_interp'].copy()

    retinocentric = data['retinocentric'].copy()
    egocentric = data['egocentric'].copy()

    distance = data['dist_to_pillar'].copy()
    cdistance = data['dist_to_center'].copy()
    pillar_size = data['pillar_size'].copy()
    yaw = data['head_yaw_deg'].copy()[:-1]

    speed = data['speed'].copy()
    speeduse = speed > 1.5
    ltdk = data['ltdk_state_vec'].copy()
    
    if restrict_by_deviation:
        theta_cent = theta.copy()
        theta_cent = theta_cent - np.nanmedian(theta_cent)
        use_thdev = np.abs(theta_cent) > 5.


    retino_bins = np.linspace(-180, 180, 27)
    ego_bins = np.linspace(-180, 180, 27)
    yaw_bins = np.linspace(-180, 180, 27)
    theta_bins = np.linspace(
        np.nanpercentile(theta, 5),
        np.nanpercentile(theta, 95),
        13
    )
    phi_bins = np.linspace(
        np.nanpercentile(phi, 5),
        np.nanpercentile(phi, 95),
        13
    )
    dist_bins = np.linspace(
        np.nanpercentile(distance, 5),
        np.nanpercentile(distance, 95),
        13
    )
    cdist_bins = np.linspace(
        np.nanpercentile(cdistance, 5),
        np.nanpercentile(cdistance, 95),
        13
    )
    psize_bins = np.linspace(
        np.nanpercentile(pillar_size, 5),
        np.nanpercentile(pillar_size, 95),
        13
    )

    vardict = {
        'yaw': {
            'vec': yaw,
            'bins': yaw_bins
        },
        'theta': {
            'vec': theta,
            'bins': theta_bins
        },
        'phi': {
            'vec': phi,
            'bins': phi_bins
        },
        'retinocentric': {
            'vec': retinocentric,
            'bins': retino_bins
        },
        'egocentric': {
            'vec': egocentric,
            'bins': ego_bins
        },
        'distance_to_pillar': {
            'vec': distance,
            'bins': dist_bins
        }
        # 'distance_to_center': {
        #     'vec': cdistance,
        #     'bins': cdist_bins
        # },
        # 'pillar_size': {
        #     'vec': pillar_size,
        #     'bins': psize_bins
        # }

    }

    full_reliability_dict = {}


    for state in range(0,2):
        # 0 is dark condition, 1 is light condition
        state = bool(state)

        if state == 0:
            statename = 'dark'

            if restrict_by_deviation:
                use1 = (~ltdk.copy()) * speeduse.copy() * use_thdev
            use0 = (~ltdk.copy()) * speeduse.copy()

        elif state == 1:
            statename = 'light'
        
            if restrict_by_deviation:
                use1 = ltdk.copy() * speeduse.copy() * use_thdev
            use0 = ltdk.copy() * speeduse.copy()

        reliability_dict = {}

        print('  -> Analyzing {} periods.'.format(statename))

        for k,v in vardict.items():

            if restrict_by_deviation and (v!='theta') and (v!='phi'):
                use = use1
            else:
                use = use0

            print('  -> Calculating reliability for tuning to: {}'.format(k))

            behavior = v['vec']
            bins = v['bins']

            add_dict = fm2p.calc_reliability_d(
                spikes[:,use],
                behavior[use],
                bins,
                10,
                100,
                thresh=1.
            )

            tbins, tunings, errors = fm2p.tuning_curve(
                spikes[:,use],
                behavior[use],
                bins
            )
            add_dict['tuning_bins'] = tbins
            add_dict['tuning_curve'] = tunings
            add_dict['tuning_stderr'] = errors

            # In addition to the shufffle reliability metric, check the spectral slope of power across
            # frequencies; smooth curves will decay steeply while noisy curves will have a flat or shallow
            # decay. A clean curve will have a value of -2 to -5. Noisy will be -1 and above. Don't want to
            # be too strict, so I'm applying the threshold of -1.25, which seems to exclude the appropriate
            # curves while including clean responses.
            spec_val, spec_rel = fm2p.calc_spectral_noise(
                tunings,
                thresh=-1.25
            )
            add_dict['reliable_by_noise'] = spec_rel
            add_dict['spectral_noise'] = spec_val

            is_reliable = spec_rel.copy() * add_dict['reliable_by_shuffle'].copy()
            add_dict['is_reliable'] = is_reliable

            mod, is_modulated = fm2p.calc_multicell_modulation(
                tunings,
                spikes[:,speeduse],   # don't apply use, just speed
                0.33
            )

            add_dict['modulation'] = mod
            add_dict['is_modulated'] = is_modulated

            reliability_dict[k] = add_dict

        full_reliability_dict[statename] = reliability_dict

    # full_reliability_dict['behavior_inputs'] = vardict

    savedir = os.path.split(preproc_path)[0]
    basename = os.path.split(preproc_path)[1][:-11]
    savepath = os.path.join(savedir, '{}_revcorr_results_v3.h5'.format(basename))
    fm2p.write_h5(savepath, full_reliability_dict)

    print('Saved {}'.format(savepath))


def revcorr():

    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--cfg', type=str, default=None)
    args = parser.parse_args()
    cfg_path = args.cfg

    if cfg_path is None:
        cfg_path = fm2p.select_file(
            title='Choose a config yaml file.',
            filetypes=[('YAML', '*.yaml'),('YML', '*.yml'),]
        )

    cfg = fm2p.read_yaml(cfg_path)

    recording_names = fm2p.list_subdirs(cfg['spath'], givepath=False)
    num_recordings = len(recording_names)
    if (type(cfg['include_recordings']) == list) and (len(cfg['include_recordings']) > 0):
        num_specified_recordings = len(cfg['include_recordings'])
    elif (type(cfg['include_recordings']) == list) and (len(cfg['include_recordings']) == 0):
        num_specified_recordings = num_recordings

    if num_recordings != num_specified_recordings:
        recording_names = [x for x in recording_names if x in cfg['include_recordings']]

    n_rec = len(recording_names)

    for rec_i, rec in enumerate(recording_names):

        print('  -> Analyzing {} recording ({}/{}).'.format(rec, rec_i+1, n_rec))

        preproc_path = fm2p.find('*_preproc.h5', os.path.join(cfg['spath'], rec), MR=True)

        if cfg['ltdk']:
            calc_revcorr_ltdk(preproc_path)
        elif not cfg['ltdk']:
            calc_revcorr(preproc_path, axons=cfg['axons'])


if __name__ == '__main__':

    revcorr()

