import fm2p
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 10
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap


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
        cmap = fm2p.make_parula()
    elif colormap_name == 'earth_tones':
        cmap = make_earth_tones()
    else:
        cmap = cm.get_cmap(colormap_name)
    normalized_positions = np.linspace(0, 1, num_values)
    colors = [cmap(pos) for pos in normalized_positions]
    return colors

goodred = '#D96459'

def calculate_r2_numpy(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return 1 - (ss_res / ss_tot)


pdata = fm2p.read_h5('/home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings/cohort02_recordings/251021_DMM_DMM061_pos04/fm1/251021_DMM_DMM061_fm_01_preproc.h5')
tdata = fm2p.read_h5('/home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings/cohort02_recordings/251021_DMM_DMM061_pos04/fm1/eyehead_revcorrs_v4cent.h5')
data = fm2p.read_h5('/home/dylan//Storage/freely_moving_data/_V1PPC/cohort02_recordings/cohort02_recordings/251021_DMM_DMM061_pos04/fm1/pytorchGLM_predictions_v09b.h5')


for c in np.argsort(data['full_trainLight_testLight_r2'])[::-1][:20]:

    fig = plt.figure(figsize=(9, 12), constrained_layout=True, dpi=300)
    gs = fig.add_gridspec(nrows=5, ncols=3)

    t = np.linspace(0, len(data['full_y_true'][:,c])*(1/7.5), len(data['full_y_true'][:,c])) / 60

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, data['full_y_true'][:,c], color='k', lw=1, label='$y$')
    ax1.plot(t, data['full_y_hat'][:,c], color=goodred, lw=1, label='$\hat{y}$')
    ax1.set_xlim([0, np.max(t)])
    ax1.set_xlabel('time (min)')
    ax1.legend(fontsize=6, loc='upper left')
    ax1.set_ylabel('z-scored dF/F')

    ax2 = fig.add_subplot(gs[1, :2])
    model_key = 'full'
    importances = {}
    prefix = f'{model_key}_importance_'
    for k, v in data.items():
        if k.startswith(prefix):
            feat_name = k[len(prefix):]
            importances[feat_name] = v

    feature_names = [
        'theta',
        'dTheta',
        'phi',
        'dPhi',
        'pitch',
        'gyro_y',
        'roll',
        'gyro_x',
        'yaw',
        'gyro_z'
    ]
    feature_names = [f for f in feature_names if f in importances]
    colors = get_equally_spaced_colormap_values('earth_tones', len(feature_names))
    values = [importances[feat][c] for feat in feature_names]
    swaps = [('gyro_x', 'dRoll'), ('gyro_y', 'dPitch'), ('gyro_z', 'dYaw')]
    nice_feature_names = feature_names
    for swap in swaps:
        nice_feature_names = [swap[1] if x == swap[0] else x for x in nice_feature_names]
    bars = ax2.bar(nice_feature_names, values, color=colors)
    ax2.set_ylabel('Importance (Drop in R^2)')

    ax2.set_ylim([0, np.max([bar.get_height() for bar in bars])*1.1])
    for bar in bars:
        height = bar.get_height()
        if height <= 0:
            continue
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=6)
    ax2.set_xticks(range(len(nice_feature_names)), nice_feature_names, rotation=90)


    ax3 = fig.add_subplot(gs[1,2])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    ax6 = fig.add_subplot(gs[2, 2])

    useinds = [
        np.arange(0,10)[::2],
        np.arange(0,10)[::2] + 1,
        np.arange(0,4),
        np.arange(4,10)
    ]
    uselabels = [
        'position_only_trainLight_testLight',
        'velocity_only_trainLight_testLight',
        'eyes_only_trainLight_testLight',
        'head_only_trainLight_testLight',
    ]

    sharedmax = 0

    for i, ax in enumerate([ax3,ax4,ax5,ax6]):

        useinds_i = useinds[i]

        model_key = uselabels[i]
        importances = {}
        prefix = f'{model_key}_importance_'
        for k, v in data.items():
            if k.startswith(prefix):
                feat_name = k[len(prefix):]
                importances[feat_name] = v

        feature_names = np.array([
            'theta',
            'dTheta',
            'phi',
            'dPhi',
            'pitch',
            'gyro_y',
            'roll',
            'gyro_x',
            'yaw',
            'gyro_z'
        ])

        useinds_i = useinds_i[np.array([f in importances for f in feature_names[useinds_i]])]
        feature_names = feature_names[useinds_i]

        colors = get_equally_spaced_colormap_values('earth_tones', 10)
        values = [importances[feat][c] for feat in list(feature_names)]
        swaps = [('gyro_x', 'dRoll'), ('gyro_y', 'dPitch'), ('gyro_z', 'dYaw')]
        nice_feature_names = feature_names
        for swap in swaps:
            nice_feature_names = [swap[1] if x == swap[0] else x for x in nice_feature_names]
        bars = ax.bar(nice_feature_names, values, color=np.array(colors)[useinds_i])
        ax.set_ylabel('Importance (Drop in R^2)')

        heights = [bar.get_height() for bar in bars]
        if heights:
            singlemax = np.max(heights)
            if singlemax > sharedmax:
                sharedmax = singlemax

        for bar in bars:
            height = bar.get_height()
            if height <= 0:
                continue
            ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom' if height > 0 else 'top', fontsize=6)
        ax.set_xticks(range(len(nice_feature_names)), nice_feature_names, rotation=90)
        ax.set_title(uselabels[i].replace('_trainLight_testLight', '').replace('_', ' '))
        
    for i, ax in enumerate([ax3,ax4,ax5,ax6]):
        ax.set_ylim([0, sharedmax*1.1])


    behavior_keys = np.array([
        'theta',
        'dTheta',
        'phi',
        'dPhi',
        'pitch',
        'gyro_y',
        'roll',
        'gyro_x',
        'yaw',
        'gyro_z'
    ])

    ax7 = fig.add_subplot(gs[3,0])
    ax8 = fig.add_subplot(gs[3,1])
    ax9 = fig.add_subplot(gs[3,2])

    sharedmax = 0

    for i in [0,2]:
        bkey = behavior_keys[i]

        ax7.plot(tdata['{}_1dbins'.format(bkey)], tdata['{}_1dtuning'.format(bkey)][c,:,1], color=colors[i], lw=2,
                label='{} (MI={:.2})'.format(bkey, tdata['{}_l_mod'.format(bkey)][c]))
        ax7.fill_between(
            tdata['{}_1dbins'.format(bkey)],
            tdata['{}_1dtuning'.format(bkey)][c,:,1] + tdata['{}_1derr'.format(bkey)][c,:,1],
            tdata['{}_1dtuning'.format(bkey)][c,:,1] - tdata['{}_1derr'.format(bkey)][c,:,1],
            color=colors[i],
            alpha=0.2
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

    for i in [1,3]:
        bkey = behavior_keys[i]

        ax9.plot(tdata['{}_1dbins'.format(bkey)], tdata['{}_1dtuning'.format(bkey)][c,:,1], color=colors[i], lw=2,
                label='{} (MI={:.2})'.format(bkey, tdata['{}_l_mod'.format(bkey)][c]))
        ax9.fill_between(
            tdata['{}_1dbins'.format(bkey)],
            tdata['{}_1dtuning'.format(bkey)][c,:,1] + tdata['{}_1derr'.format(bkey)][c,:,1],
            tdata['{}_1dtuning'.format(bkey)][c,:,1] - tdata['{}_1derr'.format(bkey)][c,:,1],
            color=colors[i],
            alpha=0.2
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

    for i in [4,6]:
        bkey = behavior_keys[i]

        ax8.plot(tdata['{}_1dbins'.format(bkey)], tdata['{}_1dtuning'.format(bkey)][c,:,1], color=colors[i], lw=2,
                label='{} (MI={:.2})'.format(bkey, tdata['{}_l_mod'.format(bkey)][c]))
        ax8.fill_between(
            tdata['{}_1dbins'.format(bkey)],
            tdata['{}_1dtuning'.format(bkey)][c,:,1] + tdata['{}_1derr'.format(bkey)][c,:,1],
            tdata['{}_1dtuning'.format(bkey)][c,:,1] - tdata['{}_1derr'.format(bkey)][c,:,1],
            color=colors[i],
            alpha=0.2
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

    for i in [5,7,9]:
        bkey = behavior_keys[i]

        ax9.plot(tdata['{}_1dbins'.format(bkey)], tdata['{}_1dtuning'.format(bkey)][c,:,1], color=colors[i], lw=2,
                label='{} (MI={:.2})'.format(bkey, tdata['{}_l_mod'.format(bkey)][c]))
        ax9.fill_between(
            tdata['{}_1dbins'.format(bkey)],
            tdata['{}_1dtuning'.format(bkey)][c,:,1] + tdata['{}_1derr'.format(bkey)][c,:,1],
            tdata['{}_1dtuning'.format(bkey)][c,:,1] - tdata['{}_1derr'.format(bkey)][c,:,1],
            color=colors[i],
            alpha=0.2
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
    ax9.set_xlim([-50,50])

    for ax in [ax7,ax8,ax9]:
        ax.set_ylabel('norm. inf. spike rate')
        ax.set_ylim([0, sharedmax*1.1])

    ax7.set_title('eye positions')
    ax8.set_title('head positions')
    ax9.set_title('velocities')

    r_rank = int(np.where(np.argsort(data['full_r2'])[::-1] == c)[0]) + 1
    fig.suptitle('Cell {}, $R^2$={:.3}, corr={:.3}, r-rank={}/{}'.format(
        c,
        data['full_r2'][c],
        data['full_corrs'][c],
        r_rank,
        len(data['full_r2'])
    ))
    ax2.set_title('all inputs')
    for ax in [ax7,ax8,ax9]:
        ax.set_ylabel('norm. inf. spike rate')
        ax.set_ylim([0, sharedmax*1.1])

    ax10 = fig.add_subplot(gs[4,0])
    ax11 = fig.add_subplot(gs[4,1])
    ax12 = fig.add_subplot(gs[4,2])

    sharedmax = 0

    for i in [0,2]:
        bkey = behavior_keys[i]

        ax10.plot(tdata['{}_1dbins'.format(bkey)], tdata['{}_1dtuning'.format(bkey)][c,:,0], color=colors[i], lw=2,
                label='{} (MI={:.2})'.format(bkey, tdata['{}_d_mod'.format(bkey)][c]))
        ax10.fill_between(
            tdata['{}_1dbins'.format(bkey)],
            tdata['{}_1dtuning'.format(bkey)][c,:,0] + tdata['{}_1derr'.format(bkey)][c,:,0],
            tdata['{}_1dtuning'.format(bkey)][c,:,0] - tdata['{}_1derr'.format(bkey)][c,:,0],
            color=colors[i],
            alpha=0.2
        )
        ax10.hlines(
            np.percentile(tdata['{}_1dtuning'.format(bkey)][c,:,0], 10),
            tdata['{}_1dbins'.format(bkey)].min(),
            tdata['{}_1dbins'.format(bkey)].max(),
            ls='--', lw=1, color=colors[i]
        )
        singlemax = np.max(tdata['{}_1dtuning'.format(bkey)][c,:,0] + tdata['{}_1derr'.format(bkey)][c,:,0])
        if singlemax > sharedmax:
            sharedmax = singlemax
    ax10.legend(fontsize=6, loc='upper left')
    ax10.set_xlabel('deg')

    for i in [1,3]:
        bkey = behavior_keys[i]

        ax12.plot(tdata['{}_1dbins'.format(bkey)], tdata['{}_1dtuning'.format(bkey)][c,:,0], color=colors[i], lw=2,
                label='{} (MI={:.2})'.format(bkey, tdata['{}_d_mod'.format(bkey)][c]))
        ax12.fill_between(
            tdata['{}_1dbins'.format(bkey)],
            tdata['{}_1dtuning'.format(bkey)][c,:,0] + tdata['{}_1derr'.format(bkey)][c,:,0],
            tdata['{}_1dtuning'.format(bkey)][c,:,0] - tdata['{}_1derr'.format(bkey)][c,:,0],
            color=colors[i],
            alpha=0.2
        )
        ax12.hlines(
            np.percentile(tdata['{}_1dtuning'.format(bkey)][c,:,0], 10),
            tdata['{}_1dbins'.format(bkey)].min(),
            tdata['{}_1dbins'.format(bkey)].max(),
            ls='--', lw=1, color=colors[i]
        )
        singlemax = np.max(tdata['{}_1dtuning'.format(bkey)][c,:,0] + tdata['{}_1derr'.format(bkey)][c,:,0])
        if singlemax > sharedmax:
            sharedmax = singlemax
    ax12.legend(fontsize=6, loc='upper left')
    ax12.set_xlabel('deg/s')

    for i in [4,6]:
        bkey = behavior_keys[i]

        ax11.plot(tdata['{}_1dbins'.format(bkey)], tdata['{}_1dtuning'.format(bkey)][c,:,0], color=colors[i], lw=2,
                label='{} (MI={:.2})'.format(bkey, tdata['{}_d_mod'.format(bkey)][c]))
        ax11.fill_between(
            tdata['{}_1dbins'.format(bkey)],
            tdata['{}_1dtuning'.format(bkey)][c,:,0] + tdata['{}_1derr'.format(bkey)][c,:,0],
            tdata['{}_1dtuning'.format(bkey)][c,:,0] - tdata['{}_1derr'.format(bkey)][c,:,0],
            color=colors[i],
            alpha=0.2
        )
        ax11.hlines(
            np.percentile(tdata['{}_1dtuning'.format(bkey)][c,:,0], 10),
            tdata['{}_1dbins'.format(bkey)].min(),
            tdata['{}_1dbins'.format(bkey)].max(),
            ls='--', lw=1, color=colors[i]
        )
        singlemax = np.max(tdata['{}_1dtuning'.format(bkey)][c,:,0] + tdata['{}_1derr'.format(bkey)][c,:,0])
        if singlemax > sharedmax:
            sharedmax = singlemax
    ax11.set_xlabel('deg')
    ax11.legend(fontsize=6, loc='upper left')

    for i in [5,7,9]:
        bkey = behavior_keys[i]

        ax12.plot(tdata['{}_1dbins'.format(bkey)], tdata['{}_1dtuning'.format(bkey)][c,:,0], color=colors[i], lw=2,
                label='{} (MI={:.2})'.format(bkey, tdata['{}_d_mod'.format(bkey)][c]))
        ax12.fill_between(
            tdata['{}_1dbins'.format(bkey)],
            tdata['{}_1dtuning'.format(bkey)][c,:,0] + tdata['{}_1derr'.format(bkey)][c,:,0],
            tdata['{}_1dtuning'.format(bkey)][c,:,0] - tdata['{}_1derr'.format(bkey)][c,:,0],
            color=colors[i],
            alpha=0.2
        )
        ax12.hlines(
            np.percentile(tdata['{}_1dtuning'.format(bkey)][c,:,0], 10),
            tdata['{}_1dbins'.format(bkey)].min(),
            tdata['{}_1dbins'.format(bkey)].max(),
            ls='--', lw=1, color=colors[i]
        )
        singlemax = np.max(tdata['{}_1dtuning'.format(bkey)][c,:,0] + tdata['{}_1derr'.format(bkey)][c,:,0])
        if singlemax > sharedmax:
            sharedmax = singlemax
    ax12.set_xlabel('deg/s')
    ax12.legend(fontsize=6, loc='upper left')
    ax12.set_xlim([-50,50])

    for ax in [ax10,ax11,ax12]:
        ax.set_ylabel('norm. inf. spike rate')
        ax.set_ylim([0, sharedmax*1.1])

    ax7.set_title('eye positions')
    ax8.set_title('head positions')
    ax9.set_title('velocities')

    fig.tight_layout()
    savename = '/home/dylan/Desktop/model_results_cell_{}.png'.format(c)
    print('saving {}'.format(savename))
    fig.savefig(savename)