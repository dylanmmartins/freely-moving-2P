




import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

import fm2p


def calculate_r2_numpy(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return 1 - (ss_res / ss_tot)

def make_earth_tones():
    """ Create a custom categorical earth-tone colormap with 10 colors in pairs.

    The pairs are:
        1. Moss & Sage (Green)
        2. Clay & Sand (Brown)
        3. Slate & Sky (Blue-Grey)
        4. Rust & Peach (Red-Orange)
        5. Ochre & Straw (Yellow)
    """

    colors = [
        # '#4F6D53', '#94B59B', # Green
        # '#9C6B54', '#DBCBB8', # Brown
        # '#5C7080', '#A6BCC9', # Blue
        # '#B56357', '#E6B8AD', # Red
        # '#B0964F', '#E0D6A8'  # Yellow

        # '#5FA55A', '#98D696', # Green
        # '#C97B53', '#EBD4A9', # Brown
        # '#5690C2', '#A6D4EB', # Blue
        # '#D96459', '#F2BCAE', # Red
        # '#E3B536', '#F2E694'  # Yellow

        '#2ECC71', '#82E0AA', # Green
        '#FF9800', '#FFCC80', # Orange
        '#03A9F4', '#81D4FA', # Blue
        '#9C27B0', '#E1BEE7', # Purple
        '#FFEB3B', '#FFF59D'  # Yellow
    ]

    # Convert hex to RGB
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


h5_path = '/home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings/cohort02_recordings/251021_DMM_DMM061_pos04/fm1/pytorchGLM_predictions_v09b.h5'
data = fm2p.read_h5(h5_path)
pdf = PdfPages(os.path.join(os.path.split(h5_path)[0], 'ffNLE_figs_summary.pdf'))

tdata = fm2p.read_h5('/home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings/cohort02_recordings/251021_DMM_DMM061_pos04/fm1/eyehead_revcorrs_v4cent.h5')

fig, axs = plt.subplots(2, 2, dpi=300, figsize=(4.,4))
axs = axs.flatten()

y_labels = [
    'position only $R^2$',
    'velocity only $R^2$',
    'eyes only $R^2$',
    'head only $R^2$'
]
y_vals = [
    data['position_only_trainLight_testLight_r2'],
    data['velocity_only_trainLight_testLight_r2'],
    data['eyes_only_trainLight_testLight_r2'],
    data['head_only_trainLight_testLight_r2'],
]


for i, ax in enumerate(axs):
    ax.plot([-0.4,0.5], [-0.4,0.5], ls='--', color='tab:red', alpha=0.4)
    ax.plot(data['full_trainLight_testLight_r2'], y_vals[i], '.', ms=2, color='k')
    ax.set_xlabel('full model $R^2$')
    ax.set_ylabel(y_labels[i])
    ax.set_title('$R^2$={:.3}'.format(calculate_r2_numpy(data['full_trainLight_testLight_r2'], y_vals[i])))
    ax.axis('equal')
    ax.set_xlim([-.3,0.45])
    ax.set_ylim([-.3,0.45])
fig.tight_layout()
pdf.savefig(fig)
plt.close(fig)




fig, ax1 = plt.subplots(1, 1, dpi=300, figsize=(2,2))

ax1.plot([-1,1], [-1,1], ls='--', color='tab:red', alpha=0.4)
ax1.plot(data['full_trainLight_testLight_r2'], data['full_trainDark_testDark_r2'], '.', ms=2, color='k')
ax1.set_xlabel('light-train light-test $R^2$')
ax1.set_ylabel('dark-train dark-test $R^2$')
# ax1.set_title('$R^2$={:.3}'.format(calculate_r2_numpy(data['full_trainLight_testLight_r2'], data['full_trainDark_testDark_r2'])))
ax1.axis('equal')
ax1.set_xlim([-1 ,1])
ax1.set_ylim([-1,1])
# ax1.set_title('light-trained model')

fig.tight_layout()
pdf.savefig(fig)
plt.close(fig)

fig, [ax1, ax2] = plt.subplots(1, 2, dpi=300, figsize=(4,2))
# axs = axs.flatten()

ax1.plot([-0.4,0.5], [-0.4,0.5], ls='--', color='tab:red', alpha=0.4)
ax1.plot(data['full_trainLight_testLight_r2'], data['full_trainLight_testDark_r2'], '.', ms=2, color='k')
ax1.set_xlabel('light test $R^2$')
ax1.set_ylabel('dark test $R^2$')
ax1.set_title('$R^2$={:.3}'.format(calculate_r2_numpy(data['full_trainLight_testLight_r2'], data['full_trainLight_testDark_r2'])))
ax1.axis('equal')
ax1.set_xlim([-.5 ,0.5])
ax1.set_ylim([-.5,0.5])
ax1.set_title('light-trained model')

ax2.plot([-0.4,0.5], [-0.4,0.5], ls='--', color='tab:red', alpha=0.4)
ax2.plot(data['full_trainDark_testDark_r2'], data['full_trainDark_testLight_r2'], '.', ms=2, color='k')
ax2.set_ylabel('light test $R^2$')
ax2.set_xlabel('dark test $R^2$')
ax2.set_title('$R^2$={:.3}'.format(calculate_r2_numpy(data['full_trainDark_testDark_r2'], data['full_trainDark_testLight_r2'])))
ax2.axis('equal')
ax2.set_xlim([-.5,0.5])
ax2.set_ylim([-.5,0.5])
ax2.set_title('dark-trained model')

fig.tight_layout()
pdf.savefig(fig)
plt.close(fig)



fig, [ax1, ax2] = plt.subplots(1, 2, dpi=300, figsize=(4,2))
# axs = axs.flatten()

ax1.plot([-1,1], [-1,1], ls='--', color='tab:red', alpha=0.4)
ax1.plot(data['full_trainLight_testLight_corrs'], data['full_trainLight_testDark_corrs'], '.', ms=2, color='k')
ax1.set_xlabel('light test $R^2$')
ax1.set_ylabel('dark test $R^2$')
ax1.set_title('$R^2$={:.3}'.format(calculate_r2_numpy(data['full_trainLight_testLight_corrs'], data['full_trainLight_testDark_corrs'])))
ax1.axis('equal')
ax1.set_xlim([-1 ,1])
ax1.set_ylim([-1,1])
ax1.set_title('light-trained model')

ax2.plot([-1,1], [-1,1], ls='--', color='tab:red', alpha=0.4)
ax2.plot(data['full_trainDark_testDark_corrs'], data['full_trainDark_testLight_corrs'], '.', ms=2, color='k')
ax2.set_ylabel('light test $R^2$')
ax2.set_xlabel('dark test $R^2$')
ax2.set_title('$R^2$={:.3}'.format(calculate_r2_numpy(data['full_trainDark_testDark_corrs'], data['full_trainDark_testLight_corrs'])))
ax2.axis('equal')
ax2.set_xlim([-1,1])
ax2.set_ylim([-1,1])
ax2.set_title('dark-trained model')

fig.tight_layout()
pdf.savefig(fig)
plt.close(fig)


# List of features identified from the dictionary keys
features = [
    'dPhi', 'dTheta', 'gyro_x', 'gyro_y', 'gyro_z',
    'phi', 'pitch', 'roll', 'theta', 'yaw'
]

# Create a 4x5 grid: 
# Top 2 rows for Cross-Condition (L-D vs D-L)
# Bottom 2 rows for Same-Condition (L-L vs D-D)
fig, axs = plt.subplots(4, 5, dpi=300, figsize=(10,8), constrained_layout=True)
axs = axs.flatten()

for i, feature in enumerate(features):
    # ---------------------------------------------------------
    # PLOT SET 1: Cross-Condition Conservation (Rows 0 and 1)
    # X: Light-Train on Dark-Test
    # Y: Dark-Train on Light-Test
    # ---------------------------------------------------------
    ax_cross = axs[i]
    
    key_x_cross = f'full_trainLight_testDark_importance_{feature}'
    key_y_cross = f'full_trainDark_testLight_importance_{feature}'
    
    x_data_cross = data[key_x_cross]
    y_data_cross = data[key_y_cross]
    
    # Dynamic limits
    all_data_cross = np.concatenate([x_data_cross, y_data_cross])
    d_min, d_max = np.min(all_data_cross), np.max(all_data_cross)
    span = d_max - d_min
    lim_min = d_min - (span * 0.1)
    lim_max = d_max + (span * 0.1)
    
    ax_cross.plot([lim_min, lim_max], [lim_min, lim_max], ls='--', color='tab:red', alpha=0.4)
    ax_cross.plot(x_data_cross, y_data_cross, '.', ms=2, color='k')
    
    r2_cross = calculate_r2_numpy(x_data_cross, y_data_cross)
    
    ax_cross.set_xlabel('L-train D-test Imp.')
    ax_cross.set_ylabel('D-train L-test Imp.')
    ax_cross.set_title(f'{feature} (Cross)\n$R^2$={r2_cross:.3f}')
    ax_cross.axis('equal')
    ax_cross.set_xlim([lim_min, lim_max])
    ax_cross.set_ylim([lim_min, lim_max])

    # ---------------------------------------------------------
    # PLOT SET 2: Same-Condition Conservation (Rows 2 and 3)
    # X: Light-Train on Light-Test
    # Y: Dark-Train on Dark-Test
    # ---------------------------------------------------------
    ax_same = axs[i + 10] # Offset by 10 to hit the 3rd and 4th rows
    
    key_x_same = f'full_trainLight_testLight_importance_{feature}'
    key_y_same = f'full_trainDark_testDark_importance_{feature}'
    
    x_data_same = data[key_x_same]
    y_data_same = data[key_y_same]
    
    # Dynamic limits
    all_data_same = np.concatenate([x_data_same, y_data_same])
    d_min, d_max = np.min(all_data_same), np.max(all_data_same)
    span = d_max - d_min
    lim_min = d_min - (span * 0.1)
    lim_max = d_max + (span * 0.1)
    
    ax_same.plot([lim_min, lim_max], [lim_min, lim_max], ls='--', color='tab:red', alpha=0.4)
    ax_same.plot(x_data_same, y_data_same, '.', ms=2, color='k')
    
    r2_same = calculate_r2_numpy(x_data_same, y_data_same)
    
    ax_same.set_xlabel('L-train L-test Imp.')
    ax_same.set_ylabel('D-train D-test Imp.')
    ax_same.set_title(f'{feature} (Same)\n$R^2$={r2_same:.3f}')
    ax_same.axis('equal')
    ax_same.set_xlim([lim_min, lim_max])
    ax_same.set_ylim([lim_min, lim_max])

fig.tight_layout()
pdf.savefig(fig)
plt.close(fig)




# List of features
features = [
    'dPhi', 'dTheta', 'gyro_x', 'gyro_y', 'gyro_z',
    'phi', 'pitch', 'roll', 'theta', 'yaw'
]

fig, axs = plt.subplots(2, 5, figsize=(10,4), dpi=300, constrained_layout=True)
axs = axs.flatten()

for i, feature in enumerate(features):
    ax = axs[i]
    
    # Get the data for Same-Condition comparison
    light_vals = data[f'full_trainLight_testLight_importance_{feature}']
    dark_vals = data[f'full_trainDark_testDark_importance_{feature}']
    
    # Determine common bin edges for fair comparison
    all_vals = np.concatenate([light_vals, dark_vals])
    bins = np.linspace(np.min(all_vals), np.max(all_vals), 30)
    
    # Plot overlapping histograms
    ax.hist(light_vals, bins=bins, alpha=0.5, label='Light', color='tab:blue', density=True)
    ax.hist(dark_vals, bins=bins, alpha=0.5, label='Dark', color='tab:orange', density=True)
    
    # Calculate means to quantify the shift
    light_mean = np.mean(light_vals)
    dark_mean = np.mean(dark_vals)
    
    ax.axvline(light_mean, color='tab:blue', linestyle='--', linewidth=1)
    ax.axvline(dark_mean, color='tab:orange', linestyle='--', linewidth=1)
    
    ax.set_title(f'{feature}\nL_$\mu$={light_mean:.2f}, D_$\mu$={dark_mean:.2f}')
    ax.set_xlabel('feature importance')
    ax.set_ylabel('cells')
    if i == 0:
        ax.legend()

# plt.suptitle("Distribution of Importance Values: Light vs Dark", fontsize=16)
fig.tight_layout()
pdf.savefig(fig)
plt.close(fig)





if 'full_r2' in data:
    main_r2 = 'full_r2'
    main_corrs = 'full_corrs'
    main_y_true = 'full_y_true'
    main_y_hat = 'full_y_hat'
    main_imp_prefix = 'full_importance_'
elif 'full_trainLight_testLight_r2' in data:
    main_r2 = 'full_trainLight_testLight_r2'
    main_corrs = 'full_trainLight_testLight_corrs'
    main_y_true = None
    main_y_hat = None
    main_imp_prefix = 'full_trainLight_testLight_importance_'
else:
    raise KeyError("Could not find full model R2 scores (checked 'full_r2' and 'full_trainLight_testLight_r2')")

for c in np.argsort(data[main_r2])[::-1][:50]:

    fig = plt.figure(figsize=(9, 10), constrained_layout=True, dpi=300)
    gs = fig.add_gridspec(nrows=4, ncols=3)

    ax1 = fig.add_subplot(gs[0, :])
    if main_y_true and main_y_true in data:
        t = np.linspace(0, len(data[main_y_true][:,c])*(1/7.5), len(data[main_y_true][:,c])) / 60
        ax1.plot(t, data[main_y_true][:,c], color='k', lw=1, label='$y$')
        ax1.plot(t, data[main_y_hat][:,c], color=goodred, lw=1, label='$\hat{y}$')
        ax1.set_xlim([0, np.max(t)])
        ax1.set_xlabel('time (min)')
        ax1.legend()
        ax1.set_ylabel('z-scored dF/F')
    else:
        ax1.text(0.5, 0.5, 'Time series not available', ha='center', va='center')
        ax1.axis('off')

    ax2 = fig.add_subplot(gs[1, :2])
    importances = {}
    prefix = main_imp_prefix
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
    # ax2.set_title('trained in full')
    # cmap = cm.get_cmap('tab20')
    # colors = [cmap(i) for i in range(len(feature_names))][:6] + [cmap(i) for i in range(len(feature_names)+4)][8:]
    colors = get_equally_spaced_colormap_values('earth_tones', 10)
    values = [importances[feat][c] for feat in feature_names]
    swaps = [('gyro_x', 'dRoll'), ('gyro_y', 'dPitch'), ('gyro_z', 'dYaw')]
    nice_feature_names = feature_names
    for swap in swaps:
        nice_feature_names = [swap[1] if x == swap[0] else x for x in nice_feature_names]
    bars = ax2.bar(nice_feature_names, values, color=colors)
    ax2.set_ylabel('Importance (Drop in R²)')

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
        'position_only',
        'velocity_only',
        'eyes_only',
        'head_only'
    ]

    if 'position_only_importance_theta' not in data:
        uselabels = [l + '_trainLight_testLight' for l in uselabels]

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

        feature_names = feature_names[useinds_i]

        colors = get_equally_spaced_colormap_values('earth_tones', 10)
        values = [importances[feat][c] for feat in list(feature_names)]
        swaps = [('gyro_x', 'dRoll'), ('gyro_y', 'dPitch'), ('gyro_z', 'dYaw')]
        nice_feature_names = feature_names
        for swap in swaps:
            nice_feature_names = [swap[1] if x == swap[0] else x for x in nice_feature_names]
        bars = ax.bar(nice_feature_names, values, color=np.array(colors)[useinds_i])
        ax.set_ylabel('Importance (Drop in R²)')

        singlemax = np.max([bar.get_height() for bar in bars])
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
        ax.set_title(uselabels[i].replace('_', ' '))
        
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
    # ax10 = fig.add_subplot(gs[3,3])

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
    ax7.legend(fontsize=6, loc='lower left')
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
    ax9.legend(fontsize=6, loc='lower left')
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
    ax8.legend(fontsize=6, loc='lower left')

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
    ax9.legend(fontsize=6, loc='lower left')
    ax9.set_xlim([-50,50])

    for ax in [ax7,ax8,ax9]:
        ax.set_ylabel('norm. inf. spike rate')
        ax.set_ylim([0, sharedmax*1.1])

    ax7.set_title('eye positions')
    ax8.set_title('head positions')
    ax9.set_title('velocities')

    r_rank = int(np.where(np.argsort(data[main_r2])[::-1] == c)[0]) + 1
    fig.suptitle('Cell {}, $R^2$={:.3}, corr={:.3}, r-rank={}/{}'.format(
        c,
        data[main_r2][c],
        data[main_corrs][c],
        r_rank,
        len(data[main_r2])
    ))
    ax2.set_title('all inputs')

    fig.tight_layout()
    # fig.savefig('/home/dylan/Fast0/Dropbox/260216_modeling_update/model_results_cell_{}.png'.format(c))
    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# Accumulated Local Effects (ALE) Visualization
# =============================================================================

ale_features = [
    'theta', 'dTheta', 'phi', 'dPhi', 'pitch', 
    'gyro_y', 'roll', 'gyro_x', 'yaw', 'gyro_z'
]

# 1. Population Average ALE
fig_ale_pop, axs_ale_pop = plt.subplots(2, 5, figsize=(6,2.5), dpi=300, constrained_layout=True)
axs_ale_pop = axs_ale_pop.flatten()

# First pass: determine global ylims for population plots
pop_y_min = np.inf
pop_y_max = -np.inf

for feat in ale_features:
    k_L_v = f'full_trainLight_testLight_ale_{feat}_curve'
    k_D_v = f'full_trainDark_testDark_ale_{feat}_curve'
    
    if k_L_v in data:
        ale_L = data[k_L_v]
        mu_L = np.nanmean(ale_L, axis=1)
        sem_L = np.nanstd(ale_L, axis=1) / np.sqrt(ale_L.shape[1])
        if not np.all(np.isnan(mu_L)):
            pop_y_min = min(pop_y_min, np.nanmin(mu_L - sem_L))
            pop_y_max = max(pop_y_max, np.nanmax(mu_L + sem_L))
        
    if k_D_v in data:
        ale_D = data[k_D_v]
        mu_D = np.nanmean(ale_D, axis=1)
        sem_D = np.nanstd(ale_D, axis=1) / np.sqrt(ale_D.shape[1])
        if not np.all(np.isnan(mu_D)):
            pop_y_min = min(pop_y_min, np.nanmin(mu_D - sem_D))
            pop_y_max = max(pop_y_max, np.nanmax(mu_D + sem_D))

if pop_y_min == np.inf: pop_y_min, pop_y_max = -0.1, 0.1
y_range = pop_y_max - pop_y_min
if y_range == 0: y_range = 1.0
pop_ylims = [pop_y_min - 0.1*y_range, pop_y_max + 0.1*y_range]

for i, feat in enumerate(ale_features):
    ax = axs_ale_pop[i]
    
    # Construct keys for Light and Dark conditions
    k_L_c = f'full_trainLight_testLight_ale_{feat}_centers'
    k_L_v = f'full_trainLight_testLight_ale_{feat}_curve'
    k_D_c = f'full_trainDark_testDark_ale_{feat}_centers'
    k_D_v = f'full_trainDark_testDark_ale_{feat}_curve'
    
    has_L = k_L_v in data and k_L_c in data
    has_D = k_D_v in data and k_D_c in data
    
    if has_L:
        ale_L = data[k_L_v] # Shape: (n_bins, n_cells)
        cents_L = data[k_L_c]
        
        # Calculate mean and SEM across cells
        mu_L = np.nanmean(ale_L, axis=1)
        sem_L = np.nanstd(ale_L, axis=1) / np.sqrt(ale_L.shape[1])
        
        ax.plot(cents_L, mu_L, color='orange', label='Light')
        ax.fill_between(cents_L, mu_L - sem_L, mu_L + sem_L, color='orange', alpha=0.2)
        
    if has_D:
        ale_D = data[k_D_v]
        cents_D = data[k_D_c]
        
        mu_D = np.nanmean(ale_D, axis=1)
        sem_D = np.nanstd(ale_D, axis=1) / np.sqrt(ale_D.shape[1])
        
        ax.plot(cents_D, mu_D, color='purple', label='Dark')
        ax.fill_between(cents_D, mu_D - sem_D, mu_D + sem_D, color='purple', alpha=0.2)
    
    ax.set_title(feat)
    if i == 0:
        ax.legend(fontsize=8)
        ax.set_ylabel('Mean ALE (z-scored)')

    if feat.startswith('d') or 'gyro' in feat:
        ax.set_xlabel('deg/s')
    else:
        ax.set_xlabel('deg')
    
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_ylim(pop_ylims)

fig_ale_pop.suptitle('Population Average Accumulated Local Effects (ALE)')
pdf.savefig(fig_ale_pop)
plt.close(fig_ale_pop)

# 2. Per-Cell ALE for top cells
if main_r2 in data:
    top_cells_ale = np.argsort(data[main_r2])[::-1][:50]
    
    for c_idx in top_cells_ale:
        fig_cell, axs_cell = plt.subplots(2, 5, figsize=(6, 2.5), dpi=300, constrained_layout=True)
        axs_cell = axs_cell.flatten()
        
        # First pass: find min/max for scaling
        y_min, y_max = 0, 0
        for i, feat in enumerate(ale_features):
            k_L_v = f'full_trainLight_testLight_ale_{feat}_curve'
            k_D_v = f'full_trainDark_testDark_ale_{feat}_curve'
            
            vals = []
            if k_L_v in data: vals.append(data[k_L_v][:, c_idx])
            if k_D_v in data: vals.append(data[k_D_v][:, c_idx])
            
            if vals:
                curr_min = np.nanmin(np.concatenate(vals))
                curr_max = np.nanmax(np.concatenate(vals))
                if curr_min < y_min: y_min = curr_min
                if curr_max > y_max: y_max = curr_max
        
        y_range = y_max - y_min
        if y_range == 0: y_range = 1
        y_lims = [y_min - 0.1*y_range, y_max + 0.1*y_range]
        
        for i, feat in enumerate(ale_features):
            ax = axs_cell[i]
            k_L_c = f'full_trainLight_testLight_ale_{feat}_centers'
            k_L_v = f'full_trainLight_testLight_ale_{feat}_curve'
            k_D_c = f'full_trainDark_testDark_ale_{feat}_centers'
            k_D_v = f'full_trainDark_testDark_ale_{feat}_curve'
            
            if k_L_v in data and k_L_c in data:
                ax.plot(data[k_L_c], data[k_L_v][:, c_idx], color='orange', label='Light', linewidth=1)
            if k_D_v in data and k_D_c in data:
                ax.plot(data[k_D_c], data[k_D_v][:, c_idx], color='purple', label='Dark', linewidth=1)
                
            ax.set_title(feat, fontsize=6)
            ax.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_ylim(y_lims)
            ax.tick_params(axis='both', which='major', labelsize=5)
            if i == 0:
                ax.legend(fontsize=5)
                ax.set_ylabel('ALE', fontsize=5)

            if feat.startswith('d') or 'gyro' in feat:
                ax.set_xlabel('deg/s', fontsize=5)
            else:
                ax.set_xlabel('deg', fontsize=5)
        
        fig_cell.suptitle(f'Cell {c_idx} ALE (R2={data[main_r2][c_idx]:.2f})', fontsize=8)
        pdf.savefig(fig_cell)
        plt.close(fig_cell)

pdf.close()
