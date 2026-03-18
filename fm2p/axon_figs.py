import os
import numpy as np
import matplotlib.pyplot as plt
import fm2p
import scipy.io as io
import matplotlib as mpl
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 7
import skimage
import scipy


def compute_kurtosis(traces):
    """
    Computes the excess kurtosis (Fisher's definition) of the traces.
    Normal distribution has kurtosis = 0.
    """
    if traces.ndim == 1:
        traces = traces[None, :]
    
    mean = np.mean(traces, axis=1, keepdims=True)
    std = np.std(traces, axis=1, keepdims=True)
    std[std < 1e-9] = 1.0
    
    fourth_moment = np.mean((traces - mean)**4, axis=1, keepdims=True)
    kurt = fourth_moment / (std**4)
    
    # Return excess kurtosis (Fisher)
    return (kurt - 3.0).flatten()


import matplotlib.cm as cm
def get_evenly_spaced_colors_from_cmap(num_colors: int = 10, cmap_name: str = 'jet'):
    """
    Generates a sequence of evenly spaced colors from a specified matplotlib colormap.

    Args:
        max_value: The upper bound of the data range to map colors from.
                   The colors will be spaced between 0 and this value.
        num_colors: The number of colors to generate in the sequence. Defaults to 10.
        cmap_name: The name of the matplotlib colormap to use. Defaults to 'jet'.

    Returns:
        A list of RGBA tuples, where each tuple represents a color.
    """
    max_value = 1

    if not isinstance(max_value, (int, float)) or max_value < 0:
        raise ValueError("max_value must be a non-negative number.")
    if not isinstance(num_colors, int) or num_colors <= 0:
        raise ValueError("num_colors must be a positive integer.")

    # Get the colormap
    cmap = cm.get_cmap(cmap_name)

    # Create an array of evenly spaced values between 0 and max_value
    # These values will be normalized by the colormap internally
    normalized_values = np.linspace(0, max_value, num_colors)

    # Map these values to colors using the colormap
    colors = [cmap(val / max_value) for val in normalized_values]

    return colors

import h5py
# preproc_path = '/home/dylan/Storage/freely_moving_data/LP/250514_DMM_DMM046_LPaxons/fm1/250514_DMM_DMM046_fm_1_preproc.h5'
preproc_path = '/home/dylan/Storage/freely_moving_data/_LGN/250923_DMM_DMM052_lgnaxons/fm1/250923_DMM_DMM052_fm_01_preproc.h5'
with h5py.File(preproc_path, 'r') as _f:
    dFF_out = _f['denoised_dFF'][:]  # (n_axons, n_frames)

plot_imgs = False
kept_groups = [[i] for i in range(dFF_out.shape[0])]

if plot_imgs:
    data = mat

    plt.figure(figsize=(4,4), dpi=300)
    img = skimage.exposure.adjust_gamma(data['data'][0]['avg_projection'][0], 1.5)
    img = img - np.min(img)
    plt.imshow(img, cmap='gray', vmax=700)
    plt.ylim([0,512])
    plt.axis('off')
    # plt.colorbar()
    plt.tight_layout()
    # plt.savefig('/home/dylan/Desktop/LGN_axons_blank.svg')


    plt.figure(dpi=300, figsize=(4,4))
    plt.imshow(img, cmap='gray', vmax=700)
    for c in range(np.size(data['data']['cellMasks'][0][0], 1)):
        pts = data['data']['cellMasks'][0][0][:,c][0]
        for i in range(np.size(pts, 0)):
            plt.plot(pts[i,0], pts[i,1], '.', color='tab:red', ms=0.25)
    plt.ylim([0,512])
    plt.axis('off')
    # plt.title('all ROIs')
    plt.tight_layout()
    # plt.savefig('/home/dylan/Desktop/LGN_axons_allROIs.svg')

    # plt.figure(dpi=300, figsize=(6,6))
    # for c in range(np.size(data['data']['cellMasks'][0][0], 1)):
    #     pts = data['data']['cellMasks'][0][0][:,c][0]
    #     for i in range(np.size(pts, 0)):
    #         plt.plot(pts[i,0], pts[i,1], 'k.', ms=1)

    plt.figure(dpi=300, figsize=(4,4))
    # for c in range(np.size(data['data']['cellMasks'][0][0], 1)):
    #     pts = data['data']['cellMasks'][0][0][:,c][0]
    #     for i in range(np.size(pts, 0)):
    #         plt.plot(pts[i,0], pts[i,1], '.', color='k', alpha=0.3, ms=1)
    colors = get_evenly_spaced_colors_from_cmap(len(kept_groups))
    np.random.shuffle(colors)
    plt.ylim([0,512])
    plt.imshow(img, cmap='gray', vmax=700)
    c_ = 0
    for gi, g in enumerate(kept_groups):
        if compute_kurtosis(dFF_out[gi,:]) < 2.:
            continue
        if len(g) > 1:
            for gx in g:
                pts = data['data']['cellMasks'][0][0][:,gx][0]
                for i in range(np.size(pts, 0)):
                    plt.plot(pts[i,0], pts[i,1], '.', color=colors[c_], ms=0.75)
        else:
            pts = data['data']['cellMasks'][0][0][:,g[0]][0]
            for i in range(np.size(pts, 0)):
                plt.plot(pts[i,0], pts[i,1], '.', color=colors[c_], ms=0.75)
        c_ += 1
    plt.axis('off')
    # plt.title('kurtosis > 2.0 (N={})'.format(c_))
    plt.tight_layout()
    # plt.savefig('/home/dylan/Desktop/LGN_axons_goodROIs.svg')
    # plt.invert_yaxis()

n_frames = np.size(dFF_out,1)

traces = np.zeros([len(kept_groups), n_frames])

c = 0
for gi, g in enumerate(kept_groups):
    if len(g) > 1:
        merged_dff = np.zeros(n_frames)
        for gx in g:
            if compute_kurtosis(dFF_out[gi,:]) < 2.:
                continue
            merged_dff += dFF_out[gi] / len(g)
        traces[c,:] = merged_dff
        c += 1
    else:
        if compute_kurtosis(dFF_out[gi,:]) < 2.:
            continue
        traces[c] = dFF_out[gi]
        c+= 1

n_cells = 10

time_range = None

t = np.arange(traces.shape[1]) / 7.5

# Select cells by kurtosis (using NaN-filled traces for stability)
# Higher kurtosis = more sparse/active transients relative to baseline noise
traces_clean = np.nan_to_num(traces)
kurt_global = np.nan_to_num(fm2p.compute_kurtosis(traces_clean), nan=-100)

# Auto-select window if needed
if time_range is None:
    # Sum normalized activity of chosen cells to find a busy window
    candidates = np.argsort(kurt_global)[::-1][:20]
    sub_traces = traces_clean[candidates]
    sub_std = np.std(sub_traces, axis=1, keepdims=True)
    sub_std[sub_std == 0] = 1.0
    sub_traces_norm = (sub_traces - np.mean(sub_traces, axis=1, keepdims=True)) / sub_std
    pop_activity = np.mean(sub_traces_norm, axis=0)
    
    dt = np.median(np.diff(t)) if len(t) > 1 else 1.0
    win_pts = int(60 / dt)
    if win_pts < len(pop_activity):
        # Rolling average to find densest activity
        kernel = np.ones(win_pts) / win_pts
        smoothed = np.convolve(pop_activity, kernel, mode='valid')
        best_idx = np.argmax(smoothed)
        time_range = [t[best_idx]-t[0], t[best_idx]-t[0] + 60]
    else:
        time_range = [0, t[-1]-t[0]]

t0 = t[0]
mask = (t >= t0 + time_range[0]) & (t <= t0 + time_range[1])

# Calculate kurtosis in the window to ensure we pick cells active in this segment
traces_local = traces_clean[:, mask]
kurt_local = np.nan_to_num(fm2p.compute_kurtosis(traces_local), nan=-100)

# Rank by both (sum of ranks) to find cells that are good globally and locally
rank_global = np.argsort(np.argsort(kurt_global))
rank_local = np.argsort(np.argsort(kurt_local))
combined_score = rank_global + rank_local
best_cells = np.argsort(combined_score)[::-1][:n_cells]

# best_cells = [0,1,2,3,4,5,6,7,8,9,10,11,12]

t_plot = t[mask] - t[mask][0]

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

for i, c_idx in enumerate(best_cells):
    y = traces[c_idx][mask]
    # Normalize to 0-1 for plotting stack (min-max normalization in window)
    y_min = np.nanmin(y)
    y_max = np.nanmax(y)
    denom = y_max - y_min
    y_norm = (y - y_min) / (denom + 1e-6)
    ax.plot(t_plot, y_norm + i * 1.1, color='k', lw=1)

ax.set_yticks([])
ax.set_xlabel('Time (s)')
# ax.set_title(f'Top {n_cells} cells by kurtosis (Window: {time_range[0]:.0f}-{time_range[1]:.0f}s)')
ax.spines['left'].set_visible(False)
fig.tight_layout()
fig.savefig('/home/dylan/Desktop/LGN_axons_demo_dFFs.svg')

# Generate tuning curve grids for theta, phi, dTheta, dPhi
revcorr_path = os.path.join(os.path.dirname(preproc_path), 'eyehead_revcorrs_v06.h5')
if os.path.exists(revcorr_path):
    print(f"Loading revcorr data from {revcorr_path}")
    revcorr_data = fm2p.read_h5(revcorr_path)
    
    vars_to_plot = ['theta', 'phi', 'dTheta', 'dPhi']
    cond_idx = 1  # 1 for light, 0 for dark
    pct_modulated = {}

    for var in vars_to_plot:
        if f'{var}_1dtuning' in revcorr_data:
            
            # Get data
            tuning = revcorr_data[f'{var}_1dtuning']
            bins = revcorr_data[f'{var}_1dbins']
            errs = revcorr_data.get(f'{var}_1derr', None)
            
            # Determine sorting metric (CV-MI/reliability)
            rel_key = f'{var}_l_rel'
            if rel_key in revcorr_data:
                metric = revcorr_data[rel_key]
            else:
                # Fallback: modulation amplitude
                if tuning.ndim == 3:
                    t = tuning[:,:,cond_idx]
                else:
                    t = tuning
                metric = np.max(t, axis=1) - np.min(t, axis=1)
            
            # Percentage of cells with modulation > 0.33
            n_mod = np.sum(np.nan_to_num(metric) > 0.15)
            pct_modulated[var] = (n_mod / len(metric)) * 100 if len(metric) > 0 else 0
            
            # Sort and select top 64
            sorted_inds = np.argsort(metric)[::-1]
            top_inds = sorted_inds[:64]
            
            fig, axs = plt.subplots(8, 8, figsize=(12, 12), dpi=300)
            axs = axs.flatten()
            
            for i, ax in enumerate(axs):
                if i < len(top_inds):
                    cell_idx = top_inds[i]
                    
                    if tuning.ndim == 3:
                        tc = tuning[cell_idx, :, cond_idx]
                        tc_err = errs[cell_idx, :, cond_idx] if errs is not None else None
                    else:
                        tc = tuning[cell_idx, :]
                        tc_err = errs[cell_idx, :] if errs is not None else None
                        
                    if len(bins) == len(tc) + 1:
                        centers = 0.5 * (bins[:-1] + bins[1:])
                    else:
                        centers = bins
                    
                    ax.plot(centers, tc, 'k-', lw=1)
                    if tc_err is not None:
                        ax.fill_between(centers, tc-tc_err, tc+tc_err, color='k', alpha=0.3, lw=0)
                    
                    ax.set_title(f'Cell {cell_idx}\nMod={metric[cell_idx]:.2f}', fontsize=6)
                    ax.tick_params(labelsize=6)
                    ax.set_ylim([0, np.max(tc+tc_err)*1.1])
                else:
                    ax.axis('off')
            
            fig.suptitle(f'{var} tuning (sorted by modulation)', fontsize=16)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path = f'/home/dylan/Desktop/LGN_axons_tuning_{var}.svg'
            fig.savefig(save_path)
            print(f"Saved {save_path}")
            plt.close(fig)

    # Bar plot of percent modulated
    if pct_modulated:
        fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
        plot_vars = [v for v in vars_to_plot if v in pct_modulated]
        vals = [pct_modulated[v] for v in plot_vars]
        
        ax.bar(plot_vars, vals, color='k', alpha=0.7)
        ax.set_ylabel('% Modulated (CV-MI > 0.33)')
        # ax.set_ylim([0, 100])
        ax.set_title('Modulation Prevalence')
        ax.set_ylim([0, 25])
        
        save_path = '/home/dylan/Desktop/LGN_axons_percent_modulated.svg'
        fig.tight_layout()
        fig.savefig(save_path)
        print(f"Saved {save_path}")
        plt.close(fig)

# Histogram of modulation index for LGN vs LP
lgn_revcorr_path = '/home/dylan/Storage/freely_moving_data/_LGN/250923_DMM_DMM052_lgnaxons/fm1/eyehead_revcorrs_v06.h5'
lp_revcorr_path = '/home/dylan/Storage/freely_moving_data/LP/250514_DMM_DMM046_LPaxons/fm1/eyehead_revcorrs_v06.h5'

if os.path.exists(lgn_revcorr_path) and os.path.exists(lp_revcorr_path):
    print("Loading LGN and LP revcorr data for comparison histogram.")
    lgn_data = fm2p.read_h5(lgn_revcorr_path)
    lp_data = fm2p.read_h5(lp_revcorr_path)

    vars_to_plot = ['theta', 'phi', 'dTheta', 'dPhi']
    cond_idx = 1  # 1 for light

    fig, axs = plt.subplots(1, 4, figsize=(8,2.5), dpi=300, sharey=True)
    axs = axs.flatten()

    for i, var in enumerate(vars_to_plot):
        ax = axs[i]
        
        # Process LGN data
        if f'{var}_1dtuning' in lgn_data:
            rel_key_lgn = f'{var}_l_rel'
            if rel_key_lgn in lgn_data:
                metric_lgn = lgn_data[rel_key_lgn]
                ax.hist(np.nan_to_num(metric_lgn), bins=np.linspace(0, 0.5, 25), density=False, histtype='step', color='tab:red', label=f'LGN (n={len(metric_lgn)})', linewidth=1.5)

        # Process LP data
        if f'{var}_1dtuning' in lp_data:
            rel_key_lp = f'{var}_l_rel'
            if rel_key_lp in lp_data:
                metric_lp = lp_data[rel_key_lp]
                ax.hist(np.nan_to_num(metric_lp), bins=np.linspace(0, 0.5, 25), density=False, histtype='step', color='tab:blue', label=f'LP (n={len(metric_lp)})', linewidth=1.5)

        ax.set_title(var)
        ax.set_xlabel('Modulation Index (CV-MI)')
        ax.set_ylabel('cells')
        # ax.legend()
        ax.set_xlim([0, 0.5])
        ax.set_ylim([0,50])

    # fig.suptitle('LGN vs LP Modulation Index Distributions (Light Condition)', fontsize=16)
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.tight_layout()
    save_path = '/home/dylan/Desktop/LGN_vs_LP_modulation_hist.svg'
    fig.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close(fig)
