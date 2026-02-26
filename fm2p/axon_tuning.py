import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import fm2p
from fm2p.utils.topography import make_area_colors, get_region_for_points, get_cell_data, get_glm_keys, add_scatter_col



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
                
                # Check for new structure
                var_data = None
                if cond_key in rdata:
                    # New structure
                    # Check if variable exists
                    curr_use_key = use_key
                    if curr_use_key not in rdata[cond_key]:
                        # Try reverse map if not already tried
                        if key in reverse_map and reverse_map[key] in rdata[cond_key]:
                            curr_use_key = reverse_map[key]
                        elif key in rdata[cond_key]:
                            curr_use_key = key
                        else:
                            continue # Variable not found
                    
                    var_data = rdata[cond_key][curr_use_key]
                
                # Check for flat dict structure (no light/dark nesting)
                elif (use_key in rdata and isinstance(rdata[use_key], dict)) or \
                     (key in reverse_map and reverse_map[key] in rdata and isinstance(rdata[reverse_map[key]], dict)) or \
                     (key in rdata and isinstance(rdata[key], dict)):
                    
                    if use_key in rdata and isinstance(rdata[use_key], dict):
                        curr_use_key = use_key
                    elif key in reverse_map and reverse_map[key] in rdata and isinstance(rdata[reverse_map[key]], dict):
                        curr_use_key = reverse_map[key]
                    else:
                        curr_use_key = key
                    
                    var_data = rdata[curr_use_key]

                # Store variable data for this poskey
                pos_var_data = {}
                for var in variables:
                    cond_key = 'light' if cond == 'l' else 'dark'
                    isrel = None
                    
                    # Check for reverse mapping
                    use_var = var
                    reverse_map = {'dRoll': 'gyro_x', 'dPitch': 'gyro_y', 'dYaw': 'gyro_z'}
                    if var in reverse_map and reverse_map[var] in rdata:
                        if isinstance(rdata[reverse_map[var]], dict): # Check if it's the dict structure
                            use_var = reverse_map[var]

                    if cond_key in rdata and var in rdata[cond_key]:
                        if 'is_reliable' in rdata[cond_key][var]:
                            isrel = rdata[cond_key][var]['is_reliable']
                    elif use_var in rdata and isinstance(rdata[use_var], dict) and 'is_reliable' in rdata[use_var]:
                        isrel = rdata[use_var]['is_reliable']
                    else:
                        isrel, _, _ = get_cell_data(rdata, var, cond)

                if var_data is not None:
                    if 'tuning_curve' not in var_data: continue
                    
                    tuning = var_data['tuning_curve']
                    bins = var_data['tuning_bins']
                    
                    # Modulation
                    if 'modulation' in var_data:
                        mods = var_data['modulation']
                    elif 'is_modulated' in var_data:
                         mods = np.zeros(len(tuning)) # Placeholder
                    else:
                        mods = np.zeros(len(tuning))

                    # Reliability
                    if 'is_reliable' in var_data:
                        rels = var_data['is_reliable']
                    else:
                        rels = np.ones(len(tuning))

                    # Continuous reliability
                    rel_vals = None
                    if 'cohen_d_vals' in var_data:
                        rel_vals = var_data['cohen_d_vals']
                    
                    # Error
                    errs = None
                    if 'tuning_stderr' in var_data:
                        errs = var_data['tuning_stderr']

                else:
                    # Old structure logic
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
                    
                    if var_data is not None:
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




# TODO: show a bar plot of the fraction reliably responsive (LGN and LP next to one another)
# TODO: show a histogram of modulation index of all reliable cells (LGN and LP overlapping nas non-filled hist)



if __name__ == '__main__':

    # filepath = '/home/dylan/Storage/freely_moving_data/_LGN/250915_DMM_DMM052_lgnaxons/fm1/250915_DMM_DMM052_fm_01_revcorr_results_v4.h5'
    filepath = '/home/dylan/Storage/freely_moving_data/LP/250514_DMM_DMM046_LPaxons/fm1/250514_DMM_DMM046_fm_1_revcorr_results.h5'

    # Load data
    rdata = fm2p.read_h5(filepath)
    
    # Mock data structure
    animal = 'DMM052'
    poskey = 'pos01'
    
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
        print("Could not determine number of cells from rdata.")
        exit()
    
    # Dummy transform: all cells at (0,0)
    transform = np.zeros((n_cells, 4))
    
    data = {
        animal: {
            'messentials': {
                poskey: {
                    'rdata': rdata,
                    'model': {} 
                }
            },
            'transform': {
                poskey: transform
            }
        }
    }
    
    animal_dirs = [animal]
    
    # Mock labeled array and map
    # We want all cells to fall into a region, say V1 (id 5)
    labeled_array = np.zeros((10, 10), dtype=int)
    labeled_array[0, 0] = 5 # V1
    
    label_map = {5: 'V1', 2: 'RL', 3: 'AM', 4: 'PM'}
    
    with PdfPages('Pulvinar_modulation_summary.pdf') as pdf:
        plot_sorted_tuning_curves(pdf, data, animal_dirs, cond='l')
    
    print("Done. Saved to temp_modulation_summary.pdf")