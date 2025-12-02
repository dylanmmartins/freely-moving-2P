

import os
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import fm2p

plt.rcParams.update({'font.size': 6})


def summarize_session(preproc_path):

    data = fm2p.read_h5(preproc_path)

    usesuptitle = os.path.split(preproc_path)[1][:-11]

    pdf_savepath = os.path.join(
        os.path.split(preproc_path)[0],
        '{}_session_summary.pdf'.format(usesuptitle)
    )
    pdf = PdfPages(pdf_savepath)


    fig, axs = plt.subplots(5, 5, figsize=(10.5, 9.5), dpi=300)

    try:
        img = data['twop_mean_img'].copy()
        cmap = plt.cm.gray.copy()
        axs[0,0].imshow(img, cmap=cmap)
        for c in range(np.size(data['norm_spikes'], 0)):
            img[data['cell_y_pix'][str(c)], data['cell_x_pix'][str(c)]] = np.nan
        cmap.set_bad(color='gold')
        axs[0,1].imshow(img, cmap=cmap)
    except:
        mat = io.loadmat(fm2p.find('*registered_data.mat', os.path.split(preproc_path)[0], MR=True))
        proj_ind = int(np.argwhere(np.asarray(mat['data'][0].dtype.names)=='avg_projection')[0])
        proj = mat['data'].item()[proj_ind].copy()
        axs[0,0].imshow(proj, cmap='gray')

    axs[0,1].set_title('{} cells'.format(np.size(data['norm_spikes'], 0)),fontsize=6)
    axs[0,1].axis('off')
    axs[0,0].axis('off')

    axs[0,2].plot(data['head_x'], data['head_y'], 'k', lw=1)
    axs[0,2].set_xlabel('x (pxls)')
    axs[0,2].set_ylabel('y (pxls)')
    axs[0,2].set_title(
        'topdown {:.3}% tracked'.format(100 - (np.sum(np.isnan(data['head_x']) * np.isnan(data['head_y'])) / len(np.isnan(data['head_y']))) * 100)
    )

    try:
        _plot_light_speed = data['speed'][data['ltdk_state_vec']]
        _plot_dark_speed = data['speed'][~data['ltdk_state_vec']]
    except:
        _plot_light_speed = data['speed'][data['ltdk_state_vec'][:-1]]
        _plot_dark_speed = data['speed'][~data['ltdk_state_vec'][:-1]]

    axs[0,3].hist(_plot_light_speed, bins=np.linspace(0,100,40), color='k')
    axs[0,3].set_title('LIGHT {:.4}% spent running'.format(np.sum((_plot_light_speed>2)/ len(_plot_light_speed))*100), fontsize=6)
    axs[0,3].set_xlim([0,100])
    axs[0,3].set_xlabel('speed (cm/s)', fontsize=6)

    axs[0,4].hist(_plot_dark_speed, bins=np.linspace(0,100,40), color='k')
    axs[0,4].set_title('DARK {:.4}% spent running'.format(np.sum((_plot_dark_speed>2)/ len(_plot_dark_speed))*100), fontsize=6)
    axs[0,4].set_xlim([0,100])
    axs[0,4].set_xlabel('speed (cm/s)', fontsize=6)

    if 'dEye' in data.keys():
        dEye = data['dEye'][::10]
    else:
        dEye = data['dTheta'][::10]

    axs[1,0].scatter(data['dHead'][::10], dEye, s=1, color='k')
    axs[1,0].set_xlim([-600,600])
    axs[1,0].set_ylim([-600,600])
    axs[1,0].set_xlim([-600,600])
    axs[1,0].set_ylim([-600,600])
    axs[1,0].plot([-600,600], [600,-600], color='tab:red', ls='--', alpha=0.3, lw=1)
    axs[1,0].set_xlabel('dHead (deg/s)', fontsize=6)
    axs[1,0].set_ylabel('dEye (deg/s)', fontsize=6)

    likelihoods = []
    for k in [x for x in data.keys() if '_likelihood' in x and 'bar' not in x and 'nose' not in x]:
        likelihoods.append(data[k])
    likelihoods = np.array(likelihoods)
    likelihoods = likelihoods[:, data['eyeT_startInd'] : data['eyeT_endInd']]

    eye_interp_ltdk = fm2p.step_interp(data['twopT'], data['ltdk_state_vec'], data['eyeT_trim'])

    tracked_pts = np.sum(likelihoods[:, eye_interp_ltdk] > 0.6, 0)
    fracgoodframes = np.sum(tracked_pts >= 7) / np.size(tracked_pts)
    hist = axs[1,2].hist(tracked_pts+0.5, bins=8, color='k')
    axs[1,2].vlines(6.5, 0, np.max(hist[0]), color='tab:red', ls='--', lw=1)
    axs[1,2].set_title('LIGHT {:.3}% tracked'.format(fracgoodframes*100))
    axs[1,2].set_xlabel('tracked pts')

    tracked_pts = np.sum(likelihoods[:, ~np.array(eye_interp_ltdk)] > 0.6, 0)
    fracgoodframes = np.sum(tracked_pts >= 7) / np.size(tracked_pts)
    hist = axs[1,3].hist(tracked_pts+0.5, bins=8, color='k')
    axs[1,3].vlines(6.5, 0, np.max(hist[0]), color='tab:red', ls='--', lw=1)
    axs[1,3].set_title('DARK {:.3}% tracked'.format(fracgoodframes*100))
    axs[1,3].set_xlabel('tracked pts')

    for i in range(len(data['axes_rel_cent_x']['0'][0]))[::2][:-1]:
        axs[1,1].plot(
            (
                data['axes_rel_cent_x']['0'][0][i],
                data['axes_rel_cent_x']['0'][1][i]
            ),

            (
                data['axes_rel_cent_y']['0'][0][i],
                data['axes_rel_cent_y']['0'][1][i]
            ),
            lw=1
        )
    axs[1,1].plot(data['camcent'][0], data['camcent'][1], '*', color='tab:red', label='camera center')
    # axs[1,1].legend(frameon=False)
    axs[1,1].set_xlabel('theta (deg)')
    axs[1,1].set_ylabel('phi (deg)')

    axs[1,4].plot(data['imuT_trim'], data['gyro_z_trim'], color='tab:red')
    axs[1,4].set_xlabel('time (s)')
    axs[1,4].set_ylabel('gyro z (deg/s)')

    axs[2,0].plot(data['eyeT'] - data['eyeT'][0], label='eyeT')
    axs[2,0].plot(data['imuT_trim'] - data['imuT_trim'][0], label='imuT')
    axs[2,0].plot(data['twopT'], label='twopT')
    axs[2,0].legend(frameon=False)
    axs[2,0].set_title('total time = {:.3} min'.format(np.max(data['twopT'])/60), fontsize=6)
    axs[2,0].set_xlabel('frames')
    axs[2,0].set_ylabel('time (s)')

    axs[2,1].hist(np.diff(data['twopT']), color='tab:red')
    axs[2,1].set_xlabel('twop dT')

    axs[2,2].hist(np.diff(data['imuT_trim']), color='tab:red')
    axs[2,2].set_xlabel('imu dT')

    axs[2,3].hist(np.diff(data['eyeT_trim']), color='tab:red')
    axs[2,3].set_xlabel('eyecam dT')

    vars = ['head_yaw_deg', 'pitch_twop_interp','roll_twop_interp']
    is_running = np.append(np.array(data['speed']>2.), np.array(False))
    for i in range(3):
        if i > 0 :
            is_running = np.array(data['speed']>2.)
        var = vars[i]
        axs[3,i].hist(data[var],
            bins=np.linspace(
                np.nanpercentile(data[var], 5),
                np.nanpercentile(data[var], 95),
                30
            ),
            alpha=0.3,
            color='k', label='stationary'
        )
        if len(data[var]) > len(is_running):
            is_running1 = np.append(np.array(False), is_running)
        elif len(data[var]) < len(is_running):
            is_running1 = is_running[:-1]
            if len(data[var]) < len(is_running1):
                is_running1 = is_running1[:-1]
        else:
            is_running1 = is_running
        axs[3,i].hist(data[var][is_running1],
            bins=np.linspace(
                np.nanpercentile(data[var], 5),
                np.nanpercentile(data[var], 95),
                30
            ),
            color='k', label='running'
        )
        axs[3,i].legend(frameon=False, fontsize=6)

    vars = ['gyro_z_twop_interp', 'gyro_x_twop_interp', 'gyro_y_twop_interp']
    for i in range(3):
        if i > 0 :
            is_running = np.array(data['speed']>2.)
        var = vars[i]
        axs[4,i].hist(data[var],
            bins=np.linspace(
                np.nanpercentile(data[var], 5),
                np.nanpercentile(data[var], 95),
                30
            ),
            alpha=0.3,
            color='k', label='stationary'
        )
        if len(data[var]) > len(is_running):
            is_running1 = np.append(np.array(False), is_running)
        elif len(data[var]) < len(is_running):
            is_running1 = is_running[:-1]
            if len(data[var]) < len(is_running1):
                is_running1 = is_running1[:-1]
        else:
            is_running1 = is_running
        axs[4,i].hist(data[var][is_running1],
            bins=np.linspace(
                np.nanpercentile(data[var], 5),
                np.nanpercentile(data[var], 95),
                30
            ),
            color='k', label='running'
        )
        axs[4,i].legend(frameon=False, fontsize=6)

    try:
        use_speed = np.array(fm2p.step_interp(data['twopT'], data['speed'], data['eyeT_trim']))
    except:
        use_speed = np.array(fm2p.step_interp(data['twopT'], np.append(data['speed'],0), data['eyeT_trim']))
    is_running = np.array(use_speed>2.)
    axs[2,4].hist(data['dGaze'],
        bins=np.linspace(
            np.nanpercentile(data['dGaze'], 5),
            np.nanpercentile(data['dGaze'], 95),
            30
        ),
        alpha=0.3,
        color='k', label='stationary'
    )
    if len(data['dGaze']) > len(is_running):
        is_running1 = np.append(np.array(False), is_running)
    elif len(data['dGaze']) < len(is_running):
        is_running1 = is_running[:-1]
        if len(data['dGaze']) < len(is_running1):
            is_running1 = is_running1[:-1]
    else:
        is_running1 = is_running
    axs[2,4].hist(data['dGaze'][is_running1],
        bins=np.linspace(
            np.nanpercentile(data['dGaze'], 5),
            np.nanpercentile(data['dGaze'], 95),
            30
        ),
        color='k', label='running'
    )
    axs[2,4].legend(frameon=False, fontsize=6)


    use_theta = data['theta_trim']
    is_running = np.array(use_speed>2.)
    axs[3,3].hist(use_theta,
        bins=np.linspace(
            np.nanpercentile(use_theta, 5),
            np.nanpercentile(use_theta, 95),
            30
        ),
        alpha=0.3,
        color='k', label='stationary'
    )
    if len(use_theta) > len(is_running):
        is_running1 = np.append(np.array(False), is_running)
    elif len(use_theta) < len(is_running):
        is_running1 = is_running[:-1]
        if len(use_theta) < len(is_running1):
            is_running1 = is_running1[:-1]
    else:
        is_running1 = is_running
    axs[3,3].hist(use_theta[is_running1],
        bins=np.linspace(
            np.nanpercentile(use_theta, 5),
            np.nanpercentile(use_theta, 95),
            30
        ),
        color='k', label='running'
    )
    axs[3,3].legend(frameon=False, fontsize=6)

    use_phi = data['phi_trim']
    is_running = np.array(use_speed>2.)
    axs[3,4].hist(use_phi,
        bins=np.linspace(
            np.nanpercentile(use_phi, 5),
            np.nanpercentile(use_phi, 95),
            30
        ),
        alpha=0.3,
        color='k', label='stationary'
    )
    if len(use_phi) > len(is_running):
        is_running1 = np.append(np.array(False), is_running)
    elif len(use_phi) < len(is_running):
        is_running1 = is_running[:-1]
        if len(use_phi) < len(is_running1):
            is_running1 = is_running1[:-1]
    else:
        is_running1 = is_running
    axs[3,4].hist(use_phi[is_running1],
        bins=np.linspace(
            np.nanpercentile(use_phi, 5),
            np.nanpercentile(use_phi, 95),
            30
        ),
        color='k', label='running'
    )
    axs[3,4].legend(frameon=False, fontsize=6)

    if 'dTheta' not in data.keys():
        use_phi = data['dEye']
        is_running = np.array(use_speed>2.)
        axs[4,3].hist(use_phi,
            bins=np.linspace(
                np.nanpercentile(use_phi, 5),
                np.nanpercentile(use_phi, 95),
                30
            ),
            alpha=0.3,
            color='k', label='stationary'
        )
        if len(use_phi) > len(is_running):
            is_running1 = np.append(np.array(False), is_running)
        elif len(use_phi) < len(is_running):
            is_running1 = is_running[:-1]
            if len(use_phi) < len(is_running1):
                is_running1 = is_running1[:-1]
        else:
            is_running1 = is_running
        axs[4,3].hist(use_phi[is_running1],
            bins=np.linspace(
                np.nanpercentile(use_phi, 5),
                np.nanpercentile(use_phi, 95),
                30
            ),
            color='k', label='running'
        )
        axs[4,3].legend(frameon=False, fontsize=6)
    else:

        use_phi = data['dTheta']
        is_running = np.array(use_speed>2.)
        axs[4,3].hist(use_phi,
            bins=np.linspace(
                np.nanpercentile(use_phi, 5),
                np.nanpercentile(use_phi, 95),
                30
            ),
            alpha=0.3,
            color='k', label='stationary'
        )
        if len(use_phi) > len(is_running):
            is_running1 = np.append(np.array(False), is_running)
        elif len(use_phi) < len(is_running):
            is_running1 = is_running[:-1]
            if len(use_phi) < len(is_running1):
                is_running1 = is_running1[:-1]
        else:
            is_running1 = is_running
        axs[4,3].hist(use_phi[is_running1],
            bins=np.linspace(
                np.nanpercentile(use_phi, 5),
                np.nanpercentile(use_phi, 95),
                30
            ),
            color='k', label='running'
        )
        axs[4,3].legend(frameon=False, fontsize=6)

        use_phi = data['dPhi']
        is_running = np.array(use_speed>2.)
        axs[4,4].hist(use_phi,
            bins=np.linspace(
                np.nanpercentile(use_phi, 5),
                np.nanpercentile(use_phi, 95),
                30
            ),
            alpha=0.3,
            color='k', label='stationary'
        )
        if len(use_phi) > len(is_running):
            is_running1 = np.append(np.array(False), is_running)
        elif len(use_phi) < len(is_running):
            is_running1 = is_running[:-1]
            if len(use_phi) < len(is_running1):
                is_running1 = is_running1[:-1]
        else:
            is_running1 = is_running
        axs[4,4].hist(use_phi[is_running1],
            bins=np.linspace(
                np.nanpercentile(use_phi, 5),
                np.nanpercentile(use_phi, 95),
                30
            ),
            color='k', label='running'
        )
        axs[4,4].legend(frameon=False, fontsize=6)

    fig.suptitle(usesuptitle)

    fig.tight_layout()

    pdf.savefig(fig)
    pdf.close()

    print('Summary PDF saved to {}'.format(pdf_savepath))



if __name__ == '__main__':


    # preproc_path = fm2p.select_file(
    #     'Select preprocessing HDF file.',
    #     filetypes=[('HDF','.h5'),])
    # summarize_session(preproc_path)

    summarize_session('/home/dylan/Storage4/V1PPC_cohort02/251016_DMM_DMM061_pos18/fm1/251016_DMM_DMM061_fm_01_preproc.h5')

