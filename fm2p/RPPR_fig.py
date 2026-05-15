

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import h5py
import fm2p
import matplotlib as mpl
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 7
import matplotlib.gridspec as gridspec
from fm2p.utils.ref_frame import calc_vor_eye_offset

tuning = fm2p.read_h5('/home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings/cohort02_recordings/251020_DMM_DMM056_pos08/fm1/eyehead_revcorrs_v06.h5')
encoding = fm2p.read_h5('/home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings/cohort02_recordings/251020_DMM_DMM056_pos08/fm1/pytorchGLM_predictions_v09b.h5')
retimg = np.load('/home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings/cohort02_recordings/251020_DMM_DMM056_pos08/fm1/251020_DMM_DMM056_fm_01_retinal_images.npz')
decoding = fm2p.read_h5('/home/dylan/Documents/Github/freely-moving-2P/decode_across_areas_only50.h5')

_preproc_h5 = '/home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings/cohort02_recordings/251020_DMM_DMM056_pos08/fm1/251020_DMM_DMM056_fm_01_preproc.h5'
with h5py.File(_preproc_h5, 'r') as _f:
    _eyeT      = _f['eyeT_trim'][:]
    _twopT     = _f['twopT'][:]
    _hx        = _f['head_x'][:]
    _hy        = _f['head_y'][:]
    _pfh_2p    = _f['pupil_from_head'][:]
    _hyaw_2p   = _f['head_yaw_deg'][:]
    _pillar_r  = float(_f['pillar_radius'][()])
    _arena_TL  = (float(_f['arenaTL']['x'][()]), float(_f['arenaTL']['y'][()]))
    _arena_TR  = (float(_f['arenaTR']['x'][()]), float(_f['arenaTR']['y'][()]))
    _arena_BL  = (float(_f['arenaBL']['x'][()]), float(_f['arenaBL']['y'][()]))
    _arena_BR  = (float(_f['arenaBR']['x'][()]), float(_f['arenaBR']['y'][()]))
    _pillar_cx = np.mean([float(_f['pillar_x'][str(k)][()]) for k in range(8)])
    _pillar_cy = np.mean([float(_f['pillar_y'][str(k)][()]) for k in range(8)])

_n = min(len(_pfh_2p), len(_hyaw_2p), len(_twopT))
_valid_x  = np.isfinite(_hx)
_valid_y  = np.isfinite(_hy)
_hx_eye   = np.interp(_eyeT, _twopT[_valid_x], _hx[_valid_x])
_hy_eye   = np.interp(_eyeT, _twopT[_valid_y], _hy[_valid_y])
_pfh_valid = np.isfinite(_pfh_2p[:_n])
_pfh_eye  = np.interp(_eyeT, _twopT[:_n][_pfh_valid], _pfh_2p[:_n][_pfh_valid])
_hyaw_eye = np.interp(_eyeT, _twopT[:_n], _hyaw_2p[:_n])



def load_results(h5_path: str) -> list:

    all_results = []
    with h5py.File(h5_path, 'r') as f:
        for key in f.keys():
            grp = f[key]
            rec = {}

            for sk in ('animal', 'pos', 'area', 'preproc_path'):
                if sk in grp.attrs:
                    v = grp.attrs[sk]
                    rec[sk] = v.decode() if isinstance(v, bytes) else str(v)

            for sk in ('area_id', 'n_cells', 'n_cells_total', 'n_blocks', 'n_folds'):
                if sk in grp.attrs:
                    rec[sk] = int(grp.attrs[sk])

            for sk in ('r_theta', 'r_phi', 'r_X0', 'r_Y0',
                       'r_pitch', 'r_roll', 'r_yaw',
                       'rmse_theta', 'rmse_phi', 'rmse_X0', 'rmse_Y0',
                       'rmse_pitch', 'rmse_roll', 'rmse_yaw'):
                rec[sk] = float(grp.attrs[sk]) if sk in grp.attrs else float('nan')

            arrays = {}
            for arr_key in ('gt_theta', 'gt_phi', 'gt_X0', 'gt_Y0',
                            'pred_theta', 'pred_phi', 'pred_X0', 'pred_Y0',
                            'gt_longaxis', 'gt_shortaxis', 'gt_ellipse_phi',
                            'gt_pitch', 'gt_roll', 'gt_yaw',
                            'pred_pitch', 'pred_roll', 'pred_yaw',
                            'valid_test', 'valid_pitch_roll', 'valid_yaw'):
                if arr_key in grp:
                    arrays[arr_key] = grp[arr_key][:]

            weights = {}
            if 'cell_weights' in grp:
                for wk in grp['cell_weights']:
                    weights[wk] = grp['cell_weights'][wk][:]
            arrays['weights'] = weights

            rec['_arrays'] = arrays
            rec['_vfs_pos'] = grp['vfs_cell_pos'][:] if 'vfs_cell_pos' in grp else np.zeros((0, 2))
            all_results.append(rec)

    print(f'Loaded {len(all_results)} results from {h5_path}')
    return all_results


def _scatter_col(ax, x_pos, vals, color, label=None):

    vals = np.array(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return
    jitter = (np.random.rand(len(vals)) - 0.5) * 0.4
    ax.scatter(np.ones(len(vals)) * x_pos + jitter, vals,
               s=4, c=color, alpha=0.7, zorder=3, label=label, ec=None)
    mn  = np.nanmean(vals)
    sem = np.nanstd(vals) / np.sqrt(len(vals))
    ax.hlines(mn, x_pos - 0.15, x_pos + 0.15, colors='tab:red', linewidths=1.5, zorder=4)
    ax.vlines(x_pos, mn - sem, mn + sem, colors='tab:red', linewidths=1.5, zorder=4)



fig = plt.figure(figsize=(5, 6), dpi=300)
gs = gridspec.GridSpec(5, 4, figure=fig)

ax_dict = {}

ax_dict['A'] = fig.add_subplot(gs[0:2, 0:2])
ax_dict['F'] = fig.add_subplot(gs[2, 0:2])
ax_dict['Z'] = fig.add_subplot(gs[2, 2:4]) 

ax_dict['A'].axis('off')

i = 0
tune1 = fig.add_subplot(gs[0, 2])
tune2 = fig.add_subplot(gs[0, 3])
tune3 = fig.add_subplot(gs[1, 2])
tune4 = fig.add_subplot(gs[1, 3])
tune1.plot(tuning['theta_1dbins'], tuning['theta_1dtuning'][i,:,1], color='k', lw=1)
tune1.fill_between(
    tuning['theta_1dbins'],
    tuning['theta_1dtuning'][i,:,1] - tuning['theta_1derr'][i,:,1],
    tuning['theta_1dtuning'][i,:,1] + tuning['theta_1derr'][i,:,1],
    alpha=0.3,
    color='k',
    ec=None
)
tune1.set_ylim([0, np.nanmax(tuning['theta_1dtuning'][i,:,1]+tuning['theta_1derr'][i,:,1])])
tune1.set_xlabel('horizontal pupil (deg)')
tune1.set_xlim([5,40])
tune1.set_xticks([5,15,25,35])

ii = 3
tune2.plot(tuning['phi_1dbins'], tuning['phi_1dtuning'][ii,:,1], color='k', lw=1)
tune2.fill_between(
    tuning['phi_1dbins'],
    tuning['phi_1dtuning'][ii,:,1] - tuning['phi_1derr'][ii,:,1],
    tuning['phi_1dtuning'][ii,:,1] + tuning['phi_1derr'][ii,:,1],
    alpha=0.3,
    color='k',
    ec=None
)
tune2.set_ylim([0, np.nanmax(tuning['phi_1dtuning'][ii,:,1]+tuning['phi_1derr'][ii,:,1])])
tune2.set_xlabel('vertical pupil (deg)')
tune2.set_xlim([5,52])
tune2.set_xticks([5,25,45])

iii = 3
tune3.plot(tuning['pitch_1dbins'], tuning['pitch_1dtuning'][iii,:,1], color='k', lw=1)
tune3.fill_between(
    tuning['pitch_1dbins'],
    tuning['pitch_1dtuning'][iii,:,1] - tuning['pitch_1derr'][iii,:,1],
    tuning['pitch_1dtuning'][iii,:,1] + tuning['pitch_1derr'][iii,:,1],
    alpha=0.3,
    color='k',
    ec=None
)
tune3.set_ylim([0, np.nanmax(tuning['pitch_1dtuning'][iii,:,1]+tuning['pitch_1derr'][iii,:,1])])
tune3.set_xlabel('head pitch (deg)')
tune3.set_xlim([-25,30])
tune3.set_xticks([-20,0,20])

iv = 8
tune4.plot(tuning['roll_1dbins'], tuning['roll_1dtuning'][iv,:,1], color='k', lw=1)
tune4.fill_between(
    tuning['roll_1dbins'],
    tuning['roll_1dtuning'][iv,:,1] - tuning['roll_1derr'][iv,:,1],
    tuning['roll_1dtuning'][iv,:,1] + tuning['roll_1derr'][iv,:,1],
    alpha=0.3,
    color='k',
    ec=None
)
tune4.set_ylim([0, np.nanmax(tuning['roll_1dtuning'][iv,:,1]+tuning['roll_1derr'][iv,:,1])])
tune4.set_xlabel('head roll (deg)')
tune4.set_xlim([-10,30])
tune4.set_xticks([-10,0,10,20,30])

tune1.set_ylabel('firing rate (Hz)')
tune3.set_ylabel('firing rate (Hz)')

ci = 6
behavior_vars = ['theta','phi','pitch','roll','yaw','dTheta','dPhi','dPitch','dRoll','dYaw']
for i in range(10):

    if behavior_vars[i] == 'dYaw':
        usebehavior = 'gyro_z'
    elif behavior_vars[i] == 'dPitch':
        usebehavior = 'gyro_y'
    elif behavior_vars[i] == 'dRoll':
        usebehavior = 'gyro_x'
    else:
        usebehavior = behavior_vars[i]

    ax_dict['Z'].bar(
        i,
        encoding['full_trainLight_testLight_importance_{}'.format(usebehavior)][ci]
    )
ax_dict['Z'].set_xticks(range(10), labels=behavior_vars, rotation=45)
ax_dict['Z'].set_ylabel('-$\Delta R^2$ (%)')

cii = 3
timestamps = np.arange(len(encoding['full_y_true'])) / 7.5
ax_dict['F'].plot(timestamps[:4000], encoding['full_y_true'][:4000, cii], color='k', lw=0.5, label='$y$')
ax_dict['F'].plot(timestamps[:4000], encoding['full_y_hat'][:4000, cii], color='tab:red', lw=0.5, label='$\hat{y}$')
ax_dict['F'].set_xlabel('time (sec)')
ax_dict['F'].set_ylabel('z-scored $\Delta$F/F')
ax_dict['F'].set_xlim([0, timestamps[4000]])
ax_dict['F'].legend(frameon=False, fontsize=5, ncol=2, bbox_to_anchor=(0.5, 1.15))

decode1 = fig.add_subplot(gs[3, 0])
decode2 = fig.add_subplot(gs[3, 1])
decode3 = fig.add_subplot(gs[3, 2:])

decode1.scatter(-decoding['DMM056_pos25_V1']['gt_phi'][::25], -decoding['DMM056_pos25_V1']['pred_phi'][::25], s=1, color='k')
decode1.plot([0,50], [0,50], '--', color='tab:red')
decode1.set_xlim([0,50])
decode1.set_ylim([0,50])
decode1.set_xlabel('vertical pupil (deg)')
decode1.set_ylabel('predicted vertical pupil (deg)')

decode2.scatter(decoding['DMM056_pos25_V1']['gt_pitch'][::40], decoding['DMM056_pos25_V1']['pred_pitch'][::40], s=1, color='k')
decode2.plot([-50,50], [-50,50], '--', color='tab:red')
decode2.set_xlim([-50,50])
decode2.set_ylim([-50,50])
decode2.set_xlabel('pitch (deg)')
decode2.set_ylabel('predicted pitch (deg)')

REGION_ORDER = ['V1', 'RL', 'AM', 'PM', 'A', 'AL', 'LM', 'P']
all_results = load_results('/home/dylan/Documents/Github/freely-moving-2P/decode_across_areas_only50.h5')
eye_vars   = ['rmse_theta', 'rmse_phi', 'rmse_X0', 'rmse_Y0']
eye_labels = [r'RMSE-idx $\theta$', r'RMSE-idx $\phi$',
                r'RMSE-idx $X_0$',    r'RMSE-idx $Y_0$']
head_vars   = ['rmse_pitch', 'rmse_roll', 'rmse_yaw']
head_labels = [r'RMSE-idx pitch', r'RMSE-idx roll', r'RMSE-idx yaw']
all_vars   = eye_vars + head_vars
area_data = {a: {v: [] for v in all_vars} for a in REGION_ORDER}
for _res in all_results:
    _a = _res.get('area', None)
    if _a in area_data:
        for _v in all_vars:
            _val = _res.get(_v, float('nan'))
            if np.isfinite(_val):
                area_data[_a][_v].append(_val)
areas_present = [a for a in REGION_ORDER if any(area_data[a][v] for v in all_vars)]
var = 'rmse_phi'
for xi, area in enumerate(areas_present):
    vals = area_data[area][var]
    _scatter_col(decode3, xi, vals, color='k')
decode3.set_xticks(range(len(areas_present)))
decode3.set_xticklabels(areas_present, fontsize=7)
decode3.set_xlim(-0.6, len(areas_present) - 0.4)
decode3.axhline(0, color='0.7', lw=0.8, ls='--')
decode3.set_ylabel('accuracy index')

ret1 = fig.add_subplot(gs[4, 0])
ret2 = fig.add_subplot(gs[4, 1])
ret3 = fig.add_subplot(gs[4, 2])
ret4 = fig.add_subplot(gs[4, 3])

# Top-down arena schematic
_arena_poly = np.array([_arena_TL, _arena_TR, _arena_BR, _arena_BL, _arena_TL])
ret1.plot(_arena_poly[:, 0], _arena_poly[:, 1], 'k-', lw=1.2, zorder=1)
ret1.add_patch(mpatches.Circle(
    (_pillar_cx, _pillar_cy), _pillar_r, color='k', fill=True, zorder=2))

_schematic_frames = [(1260, 'tab:red'), (2750, 'tab:blue'), (2730, 'tab:green')]
_arrow_len = 250  # pixels
_mouse_r   = 70   # pixels

for _fr, _col in _schematic_frames:
    _mx = _hx_eye[_fr]
    _my = _hy_eye[_fr]
    # head_yaw_deg is directly CW-from-North in image space
    _h_cw  = np.radians(_hyaw_eye[_fr])
    _ydx = np.sin(_h_cw) * _arrow_len
    _ydy = -np.cos(_h_cw) * _arrow_len
    ret1.plot([_mx, _mx + _ydx], [_my, _my + _ydy], color='0.6', lw=0.8, zorder=3)
    # gaze = head + stored pupil_from_head (includes anatomical ~65° offset)
    _pfh = _pfh_eye[_fr]
    if np.isfinite(_pfh):
        _g_cw = np.radians(_hyaw_eye[_fr] + _pfh)
        _dx = np.sin(_g_cw) * _arrow_len
        _dy = -np.cos(_g_cw) * _arrow_len
    else:
        _dx, _dy = _ydx, _ydy  # fallback: show head direction only
    ret1.add_patch(mpatches.Circle((_mx, _my), _mouse_r, color=_col, fill=True, zorder=4))
    ret1.annotate('', xy=(_mx + _dx, _my + _dy), xytext=(_mx, _my),
                  arrowprops=dict(arrowstyle='->', color=_col, lw=1), zorder=5)

_pad = 80
ret1.set_xlim(_arena_TL[0] - _pad, _arena_TR[0] + _pad)
ret1.set_ylim(_arena_BL[1] + _pad, _arena_TL[1] - _pad)  # inverted: large y at bottom
ret1.set_aspect('equal')
ret1.axis('off')

ret2.imshow(retimg['retinal_images'][1260], cmap='Reds')
ret2.axis('off')

ret3.imshow(retimg['retinal_images'][2750], cmap='Blues')
ret3.axis('off')

ret4.imshow(retimg['retinal_images'][2730], cmap='Greens')
ret4.axis('off')

fig.tight_layout()
fig.savefig('/home/dylan/Desktop/rppr_fig_v01.svg')
