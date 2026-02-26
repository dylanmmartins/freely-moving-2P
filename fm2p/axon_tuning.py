import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import fm2p
from fm2p.utils.topography import make_area_colors, get_region_for_points, get_cell_data, get_glm_keys, add_scatter_col



def get_metrics(rdata, var, cond='l'):
    """ Extract reliability and modulation metrics for a variable. """
    cond_key = 'light' if cond == 'l' else 'dark'
    reverse_map = {'dRoll': 'gyro_x', 'dPitch': 'gyro_y', 'dYaw': 'gyro_z'}
    
    use_key = var
    if var in reverse_map:
        use_key = reverse_map[var]

    var_data = None
    
    # Check for new structure (nested light/dark)
    if cond_key in rdata:
        curr_use_key = use_key
        if curr_use_key not in rdata[cond_key]:
            if var in reverse_map and reverse_map[var] in rdata[cond_key]:
                curr_use_key = reverse_map[var]
            elif var in rdata[cond_key]:
                curr_use_key = var
            else:
                return None, None, None, None, None, None # Variable not found
        var_data = rdata[cond_key][curr_use_key]

    # Check for flat dict structure
    elif (use_key in rdata and isinstance(rdata[use_key], dict)) or \
         (var in reverse_map and reverse_map[var] in rdata and isinstance(rdata[reverse_map[var]], dict)) or \
         (var in rdata and isinstance(rdata[var], dict)):
        
        if use_key in rdata and isinstance(rdata[use_key], dict):
            curr_use_key = use_key
        elif var in reverse_map and reverse_map[var] in rdata and isinstance(rdata[reverse_map[var]], dict):
            curr_use_key = reverse_map[var]
        else:
            curr_use_key = var
        var_data = rdata[curr_use_key]

    if var_data is not None:
        tuning = var_data.get('tuning_curve')
        bins = var_data.get('tuning_bins')
        mods = var_data.get('modulation', np.zeros(len(tuning)) if tuning is not None else None)
        rels = var_data.get('is_reliable', np.ones(len(tuning)) if tuning is not None else None)
        rel_vals = var_data.get('cohen_d_vals')
        errs = var_data.get('tuning_stderr')
        
        if 'is_modulated' in var_data and 'modulation' not in var_data:
             mods = np.zeros(len(tuning)) # Placeholder if only boolean is present

        return tuning, bins, mods, rels, errs, rel_vals

    # Fallback to old structure using get_cell_data logic (simplified)
    # This part assumes flat structure with specific naming conventions handled by get_cell_data
    # But get_cell_data returns (isrel, mod, peak). We need tuning curves for the plot.
    # So we might need to manually extract tuning/bins if get_cell_data doesn't give them.
    # For now, if var_data is None, we assume it might be the old flat array structure
    # which is handled inside the loop in plot_sorted_tuning_curves.
    # To unify, we would need to refactor plot_sorted_tuning_curves significantly.
    # For the comparison plots, we only need isrel and mod.
    
    return None, None, None, None, None, None



def plot_sorted_tuning_curves(pdf, data, animal_dirs, cond='l'):
    
    variables = ['theta', 'phi', 'dTheta', 'dPhi', 'pitch', 'yaw', 'roll', 'dPitch', 'dYaw', 'dRoll']
    cond_idx = 1 if cond == 'l' else 0
    cond_name = 'Light' if cond == 'l' else 'Dark'
    cond_key = 'light' if cond == 'l' else 'dark'
    
    reverse_map = {'dRoll': 'gyro_x', 'dPitch': 'gyro_y', 'dYaw': 'gyro_z'}

    for key in variables:
        cells = []
        
        use_key = key
        if key in reverse_map:
            use_key = reverse_map[key]

        for animal in animal_dirs:
            if animal not in data: continue
            if 'transform' not in data[animal]: continue
            
            for poskey in data[animal]['transform']:
                if (animal=='DMM056') and (cond=='d') and ((poskey=='pos15') or (poskey=='pos03')):
                    continue

                messentials = data[animal]['messentials'][poskey]
                rdata = messentials.get('rdata', {})
                
                # Try to get metrics using the helper
                tuning, bins, mods, rels, errs, rel_vals = get_metrics(rdata, key, cond)

                if tuning is None:
                    # Fallback to old structure logic if get_metrics returned None
                    curr_use_key = use_key
                    if key in reverse_map:
                        mapped = reverse_map[key]
                        if f'{mapped}_1dtuning' in rdata:
                            curr_use_key = mapped
                        elif f'{key}_1dtuning' in rdata:
                            curr_use_key = key
                        else:
                            curr_use_key = mapped

                    tuning_key = f'{curr_use_key}_1dtuning'
                    bins_key = f'{curr_use_key}_1dbins'
                    mod_key = f'{curr_use_key}_{cond}_mod'
                    rel_key = f'{curr_use_key}_{cond}_isrel'
                    rel_val_key = f'{curr_use_key}_{cond}_rel'
                
                    err_key = f'{curr_use_key}_1derr'
                    if err_key not in rdata:
                        err_key = f'{curr_use_key}_1dstderr'
                    
                    if tuning_key not in rdata or bins_key not in rdata or mod_key not in rdata or rel_key not in rdata:
                        continue
                    
                    tuning = rdata[tuning_key]
                    bins = rdata[bins_key]
                    mods = rdata[mod_key]
                    rels = rdata[rel_key]
                    
                    rel_vals = None
                    if rel_val_key in rdata:
                        rel_vals = rdata[rel_val_key]
                    
                    errs = None
                    if err_key in rdata:
                        errs = rdata[err_key]
                
                n_cells = tuning.shape[0]
                
                for c in range(n_cells):
                    if np.isnan(mods[c]): continue
                    if rels[c] == 0: continue
                    
                    if tuning is not None:
                        # New structure is always 2D (cells, bins) for a specific condition
                        t_curve = tuning[c, :]
                        t_err = errs[c, :] if errs is not None else np.zeros_like(t_curve)
                    else:
                        # Old structure might be 3D
                        if tuning.ndim == 3:
                            if tuning.shape[2] > cond_idx:
                                t_curve = tuning[c, :, cond_idx]
                                t_err = errs[c, :, cond_idx] if errs is not None else np.zeros_like(t_curve)
                            else:
                                continue
                        else:
                            t_curve = tuning[c, :]
                            t_err = errs[c, :] if errs is not None else np.zeros_like(t_curve)

                    cells.append({
                        'mod': mods[c],
                        'tuning': t_curve,
                        'err': t_err,
                        'bins': bins,
                        'id': f'{animal} {poskey} {c}',
                        'rel_val': rel_vals[c] if rel_vals is not None else np.nan
                    })
        
        if not cells:
            continue
            
        cells.sort(key=lambda x: x['mod'], reverse=True)
        top_cells = cells[:64]
        
        fig, axs = plt.subplots(8, 8, figsize=(16, 16), dpi=300)
        axs = axs.flatten()
        
        for i, ax in enumerate(axs):
            if i < len(top_cells):
                cell = top_cells[i]
                bin_edges = cell['bins']
                if len(bin_edges) == len(cell['tuning']) + 1:
                    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                else:
                    bin_centers = bin_edges
                
                ax.plot(bin_centers, cell['tuning'], 'k-')
                ax.fill_between(bin_centers, cell['tuning'] - cell['err'], cell['tuning'] + cell['err'], color='k', alpha=0.3)
                
                title_str = f"{cell['id']}\nMI={cell['mod']:.2f}"
                # if not np.isnan(cell['rel_val']):
                #     title_str += f" R={cell['rel_val']:.4f}"
                ax.set_title(title_str, fontsize=6)
                ax.tick_params(labelsize=6)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            else:
                ax.axis('off')
                
        fig.suptitle(f'Top 64 Modulated Cells for {key} ({cond_name})', fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)


def plot_lgn_lp_comparison(pdf, lgn_rdata, lp_rdata, cond='l'):
    variables = ['theta', 'phi', 'dTheta', 'dPhi', 'pitch', 'yaw', 'roll', 'dPitch', 'dYaw', 'dRoll']
    
    frac_reliable = {'LGN': [], 'LP': []}
    mod_indices = {'LGN': {}, 'LP': {}}
    
    for var in variables:
        # LGN
        _, _, lgn_mods, lgn_rels, _, _ = get_metrics(lgn_rdata, var, cond)
        if lgn_rels is not None:
            frac_reliable['LGN'].append(np.mean(lgn_rels))
            mod_indices['LGN'][var] = lgn_mods[lgn_rels == 1] if lgn_mods is not None else []
        else:
            frac_reliable['LGN'].append(0)
            mod_indices['LGN'][var] = []
            
        # LP
        _, _, lp_mods, lp_rels, _, _ = get_metrics(lp_rdata, var, cond)
        if lp_rels is not None:
            frac_reliable['LP'].append(np.mean(lp_rels))
            mod_indices['LP'][var] = lp_mods[lp_rels == 1] if lp_mods is not None else []
        else:
            frac_reliable['LP'].append(0)
            mod_indices['LP'][var] = []

    # 1. Bar plot of fraction reliable
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(variables))
    width = 0.35
    
    ax.bar(x - width/2, frac_reliable['LGN'], width, label='LGN', color='tab:blue')
    ax.bar(x + width/2, frac_reliable['LP'], width, label='LP', color='tab:orange')
    
    ax.set_ylabel('Fraction Reliable')
    ax.set_title(f'Reliability by Variable ({cond})')
    ax.set_xticks(x)
    ax.set_xticklabels(variables, rotation=45)
    ax.legend()
    
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    
    # 2. Histograms of modulation index
    fig, axs = plt.subplots(2, 5, figsize=(15, 6), dpi=300)
    axs = axs.flatten()
    
    for i, var in enumerate(variables):
        ax = axs[i]
        lgn_vals = mod_indices['LGN'][var]
        lp_vals = mod_indices['LP'][var]
        
        if len(lgn_vals) > 0:
            ax.hist(lgn_vals, bins=20, density=True, histtype='step', color='tab:blue', label='LGN', linewidth=1.5)
        if len(lp_vals) > 0:
            ax.hist(lp_vals, bins=20, density=True, histtype='step', color='tab:orange', label='LP', linewidth=1.5)
            
        ax.set_title(var)
        if i == 0:
            ax.legend(fontsize=8)
        if i >= 5:
            ax.set_xlabel('Modulation Index')
            
    fig.suptitle(f'Modulation Index Distributions ({cond})')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# TODO: show a bar plot of the fraction reliably responsive (LGN and LP next to one another)
# TODO: show a histogram of modulation index of all reliable cells (LGN and LP overlapping nas non-filled hist)



if __name__ == '__main__':

    lgn_path = '/home/dylan/Storage/freely_moving_data/_LGN/250915_DMM_DMM052_lgnaxons/fm1/250915_DMM_DMM052_fm_01_revcorr_results_v4.h5'
    lp_path = '/home/dylan/Storage/freely_moving_data/LP/250514_DMM_DMM046_LPaxons/fm1/250514_DMM_DMM046_fm_1_revcorr_results.h5'

    # Load data
    lgn_rdata = fm2p.read_h5(lgn_path)
    lp_rdata = fm2p.read_h5(lp_path)
    
    # Prepare data structure for plot_sorted_tuning_curves
    data = {}
    
    for name, rdata in [('LGN', lgn_rdata), ('LP', lp_rdata)]:
        # Determine number of cells
        n_cells = 0
        if 'light' in rdata.keys():
            for var in rdata['light']:
                if 'tuning_curve' in rdata['light'][var]:
                    n_cells = rdata['light'][var]['tuning_curve'].shape[0]
                    break
        else:
            # Check for flat dict structure
            for k in rdata.keys():
                if isinstance(rdata[k], dict) and 'tuning_curve' in rdata[k]:
                    n_cells = rdata[k]['tuning_curve'].shape[0]
                    break
        
        if n_cells == 0:
            for k in rdata.keys():
                if '1dtuning' in k:
                    n_cells = rdata[k].shape[0]
                    break
        
        if n_cells == 0:
            print(f"Could not determine number of cells for {name}.")
            continue

        # Dummy transform
        transform = np.zeros((n_cells, 4))
        
        data[name] = {
            'messentials': {
                'pos01': {
                    'rdata': rdata,
                    'model': {} 
                }
            },
            'transform': {
                'pos01': transform
            }
        }
    
    animal_dirs = list(data.keys())
    
    with PdfPages('LGN_LP_modulation_summary.pdf') as pdf:
        
        plot_sorted_tuning_curves(pdf, data, animal_dirs, cond='l')

        plot_lgn_lp_comparison(pdf, lgn_rdata, lp_rdata, cond='l')
    
    print("Done. Saved to LGN_LP_modulation_summary.pdf")