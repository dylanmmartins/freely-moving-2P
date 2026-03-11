import fm2p
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 7
from matplotlib.collections import LineCollection


def plot_behavior_overview(data_input, save_path=None, time_range=None):
    """
    Create a summary figure of behavior: Eye (pos/vel), Head (trajectory, gyro, pitch/roll).
    
    Parameters
    ----------
    data_input : str or dict
        Path to preprocessed .h5 file or the loaded data dictionary.
    save_path : str, optional
        Path to save the figure (e.g. 'behavior.pdf'). If None, calls plt.show().
    time_range : tuple or list, optional
        [start_time, end_time] in seconds relative to recording start.
        If None, plots the entire recording.
    """
    if isinstance(data_input, str):
        data = fm2p.read_h5(data_input)
    else:
        data = data_input

    # Determine time base
    if 'twopT' in data:
        t = data['twopT']
    elif 'eyeT' in data:
        t = data['eyeT']
    else:
        # Fallback if no time vector found
        n = len(data.get('theta_interp', []))
        t = np.arange(n) / 60.0

    # Handle time range selection
    if time_range is not None:
        t0 = t[0]
        start_t = t0 + time_range[0]
        end_t = t0 + time_range[1]
        mask = (t >= start_t) & (t <= end_t)
    else:
        mask = np.ones_like(t, dtype=bool)

    t_seg = t[mask]
    if len(t_seg) == 0:
        print("No data in specified time range.")
        return

    t_plot = t_seg - t_seg[0] # Time from start of segment for plotting x-axis

    # Helper to get and slice data safely
    def get_data(key):
        if key in data:
            arr = data[key]
            if len(arr) == len(t):
                return arr[mask]
            # Handle slight mismatches if possible
            if len(arr) >= len(t):
                return arr[:len(t)][mask]
        return None

    theta = get_data('theta_interp')
    phi = get_data('phi_interp')
    
    if theta is not None:
        dt = np.median(np.diff(t_seg)) if len(t_seg) > 1 else 1.0
        # Calculate velocity from interpolated position to handle gaps/noise
        dTheta = np.gradient(fm2p.interp_short_gaps(theta), dt)
        dPhi = np.gradient(fm2p.interp_short_gaps(phi), dt)
    else:
        dTheta, dPhi = None, None

    # Trajectory: Prefer head_x/y (centroid of ears), then x/y (nose)
    x = get_data('head_x')
    y = get_data('head_y')
    if x is None:
        x = get_data('x')
        y = get_data('y')

    if x is not None and np.sum(~np.isnan(x)) > 1:
        x = fm2p.convfilt(fm2p.interp_short_gaps(x, max_gap=60), box_pts=60)
    if y is not None and np.sum(~np.isnan(y)) > 1:
        y = fm2p.convfilt(fm2p.interp_short_gaps(y, max_gap=60), box_pts=60)
    
    # IMU
    gx = get_data('gyro_x_twop_interp')
    gy = get_data('gyro_y_twop_interp')
    gz = get_data('gyro_z_twop_interp')
    
    # Accel
    ax_x = get_data('acc_x_twop_interp')
    ax_y = get_data('acc_y_twop_interp')
    ax_z = get_data('acc_z_twop_interp')
    
    pitch = get_data('pitch_twop_interp')
    roll = get_data('roll_twop_interp')

    # Setup Figure
    fig = plt.figure(figsize=(8, 4.5), constrained_layout=True, dpi=300)
    gs = fig.add_gridspec(3, 2)

    # Panel 1: Eye Position (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    if theta is not None:
        ax1.plot(t_plot, theta, label=r'$\theta$ (azimuth)', color='tab:blue')
        ax1.plot(t_plot, phi, label=r'$\phi$ (elevation)', color='tab:orange')
    ax1.set_ylabel('Degrees')
    ax1.set_xlim(0, 60)
    ax1.set_xticks(np.arange(0, 61, 10))

    # Panel 2: Eye Velocity (Middle Left)
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    if dTheta is not None:
        ax2.plot(t_plot, dTheta, label=r'$d\theta$', color='tab:blue', alpha=0.8)
        ax2.plot(t_plot, dPhi, label=r'$d\phi$', color='tab:orange', alpha=0.8)
    ax2.set_ylabel('Degrees/s')
    ax2.set_xlabel('Time (s)')

    # Panel 4: Gyroscope (Top Right)
    ax4 = fig.add_subplot(gs[0, 1], sharex=ax1)
    if gx is not None:
        ax4.plot(t_plot, gx, label='Gyro X', color='tab:green', alpha=0.8)
        ax4.plot(t_plot, gy, label='Gyro Y', color='tab:red', alpha=0.8)
        ax4.plot(t_plot, gz, label='Gyro Z', color='tab:purple', alpha=0.8)
    ax4.set_ylabel('deg/s')

    # Panel 5: Accelerometer (Middle Right)
    ax5 = fig.add_subplot(gs[1, 1], sharex=ax1)
    if ax_x is not None:
        ax5.plot(t_plot, ax_x, label='Acc X', color='tab:green', alpha=0.8)
        ax5.plot(t_plot, ax_y, label='Acc Y', color='tab:red', alpha=0.8)
        ax5.plot(t_plot, ax_z, label='Acc Z', color='tab:purple', alpha=0.8)
    ax5.set_ylabel('g')

    # Panel 6: Head Orientation (Bottom Right)
    ax6 = fig.add_subplot(gs[2, 1], sharex=ax1)
    if pitch is not None:
        ax6.plot(t_plot, pitch, label='Pitch', color='tab:brown')
        ax6.plot(t_plot, roll, label='Roll', color='tab:pink')
    ax6.set_ylabel('Degrees')
    ax6.set_xlabel('Time (s)')


    ax5.set_ylim([-2,2])
    ax4.set_ylim([-700,700])
    

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    return fig


def plot_example_cells(data_input, save_path=None, time_range=None, n_cells=10):
    """
    Plot dFF traces of example cells with high kurtosis.

    Parameters
    ----------
    data_input : str or dict
        Path to preprocessed .h5 file or the loaded data dictionary.
    save_path : str, optional
        Path to save the figure (e.g. 'cells.pdf'). If None, calls plt.show().
    time_range : list, optional
        [start, end] in seconds. If None, automatically selects a 60s window
        with high activity in the selected cells.
    n_cells : int
        Number of cells to plot.
    """
    if isinstance(data_input, str):
        data = fm2p.read_h5(data_input)
    else:
        data = data_input

    # Prefer dFF, fallback to denoised
    if 'norm_dFF' in data:
        traces = data['norm_dFF']
    elif 'denoised_dFF' in data:
        traces = data['denoised_dFF']
    else:
        print("No dFF traces found (norm_dFF or denoised_dFF).")
        return

    if 'twopT' in data:
        t = data['twopT']
    else:
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
    ax.set_title(f'Top {n_cells} cells by kurtosis (Window: {time_range[0]:.0f}-{time_range[1]:.0f}s)')
    ax.spines['left'].set_visible(False)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    return fig


def plot_stacked_tuning_curves(data_input, cell_list, variables=None, save_path=None, restrict_to_imu=True):
    """
    Plot stacked tuning curves for specified cells.
    
    Parameters
    ----------
    data_input : str or dict
        Path to pooled .h5 file or the loaded data dictionary.
    cell_list : list of dict
        List of cells to plot. Each item should be {'animal': str, 'pos': str, 'idx': int}.
    variables : list of str, optional
        Variables to plot (rows). Defaults to standard set.
    save_path : str, optional
        Path to save figure.
    """
    if variables is None:
        variables = ['pitch', 'roll', 'yaw', 'gyro_x', 'gyro_y', 'gyro_z']
        
    if isinstance(data_input, str):
        data = fm2p.read_h5(data_input)
    else:
        data = data_input

    if restrict_to_imu:
        filtered_list = []
        for cell_info in cell_list:
            animal = cell_info.get('animal')
            pos = cell_info.get('pos')
            try:
                rdata = data[animal]['messentials'][pos]['rdata']
                if 'gyro_z_1dtuning' in rdata:
                    filtered_list.append(cell_info)
            except (KeyError, TypeError):
                continue
        cell_list = filtered_list
        print(f"Restricted cell list to {len(cell_list)} cells with IMU data.")

    n_cells = len(cell_list)
    if n_cells == 0:
        print("No cells provided to plot_stacked_tuning_curves.")
        return

    n_vars = len(variables)
    
    fig, axs = plt.subplots(n_vars, n_cells, figsize=(n_cells * 1.5 + 1, n_vars * 1.2), 
                            dpi=300, constrained_layout=True)
    
    # Ensure axs is 2D array [row, col]
    if n_vars == 1 and n_cells == 1:
        axs = np.array([[axs]])
    elif n_vars == 1:
        axs = axs[np.newaxis, :]
    elif n_cells == 1:
        axs = axs[:, np.newaxis]

    # Map for variable name aliases
    alias_map = {
        'gyro_x': 'dRoll', 'gyro_y': 'dPitch', 'gyro_z': 'dYaw',
        'dRoll': 'gyro_x', 'dPitch': 'gyro_y', 'dYaw': 'gyro_z'
    }
    
    label_map = {
        'gyro_x': 'dRoll', 'gyro_y': 'dPitch', 'gyro_z': 'dYaw'
    }

    for i, cell_info in enumerate(cell_list):
        animal = cell_info.get('animal')
        pos = cell_info.get('pos')
        idx = cell_info.get('idx')
        
        try:
            rdata = data[animal]['messentials'][pos]['rdata']
        except (KeyError, TypeError):
            for j in range(n_vars):
                axs[j, i].axis('off')
            continue

        for j, var in enumerate(variables):
            ax = axs[j, i]
            
            keys = [var]
            if var in alias_map:
                keys.append(alias_map[var])
            
            tc = None
            bins = None
            
            for k in keys:
                t_key = f'{k}_1dtuning'
                b_key = f'{k}_1dbins'
                if t_key in rdata and b_key in rdata:
                    tc_arr = rdata[t_key]
                    bins = rdata[b_key]
                    if idx < tc_arr.shape[0]:
                        if tc_arr.ndim == 3:
                            ci = 1 if tc_arr.shape[2] > 1 else 0
                            tc = tc_arr[idx, :, ci]
                        else:
                            tc = tc_arr[idx, :]
                    break
            
            if tc is not None:
                if len(bins) == len(tc) + 1:
                    centers = 0.5 * (bins[:-1] + bins[1:])
                else:
                    centers = bins
                
                ax.plot(centers, tc, 'k-', lw=1)
                
                err = None
                for k in keys:
                    if f'{k}_1derr' in rdata:
                        err_arr = rdata[f'{k}_1derr']
                        if idx < err_arr.shape[0]:
                            if err_arr.ndim == 3:
                                ci = 1 if err_arr.shape[2] > 1 else 0
                                err = err_arr[idx, :, ci]
                            else:
                                err = err_arr[idx, :]
                        break
                
                if err is not None:
                    ax.fill_between(centers, tc-err, tc+err, color='k', alpha=0.2, lw=0)
                
                ax.set_ylim(bottom=0)
                
                if i == 0:
                    ylabel = label_map.get(var, var)
                    ax.set_ylabel(ylabel, rotation=0, ha='right', va='center', fontsize=8)
                if j == 0:
                    ax.set_title(f"{animal}\n{pos}\nCell {idx}", fontsize=6)
            else:
                ax.axis('off')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    return fig


if __name__ == '__main__':

    pooled_path = '/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites/pooled_260310a.h5'
    
    try:
        data = fm2p.read_h5(pooled_path)
        
        AREA_IDS = {'RL': 2, 'AM': 3, 'PM': 4, 'V1': 5, 'AL': 7, 'LM': 8, 'P': 9, 'A': 10}

        target_area = 'AM'
        cell_list = []
        
        alias_map = {
            'gyro_x': 'dRoll', 'gyro_y': 'dPitch', 'gyro_z': 'dYaw',
            'dRoll': 'gyro_x', 'dPitch': 'gyro_y', 'dYaw': 'gyro_z'
        }

        target_area_id = AREA_IDS.get(target_area)

        count = 0
        target_count = 8
        
        for animal in data.keys():
            if not isinstance(data[animal], dict): continue
            if 'messentials' not in data[animal]: continue
            
            messentials = data[animal]['messentials']
            for pos in messentials.keys():
                if not pos.startswith('pos'): continue
                
                pos_data = messentials[pos]

                if 'rdata' in pos_data:
                    rdata = pos_data['rdata']
                    if 'theta_1dtuning' in rdata:
                        n_cells_in_rec = rdata['theta_1dtuning'].shape[0]

                        candidates = []
                        if target_area_id is not None and 'visual_area_id' in pos_data:
                            area_ids = np.array(pos_data['visual_area_id'])
                            if len(area_ids) == n_cells_in_rec:
                                candidates = np.where(area_ids == target_area_id)[0]

                        if len(candidates) > 0 and 'gyro_z_1dtuning' in rdata:
                            # Take up to 2 cells per recording to get variety
                            rec_added = 0
                            for idx in candidates:
                                idx = int(idx)
                                # Check for NaNs
                                has_nan = False
                                for var in ['theta', 'phi', 'pitch', 'roll', 'dTheta', 'dPhi', 'gyro_z']:
                                    keys = [var]
                                    if var in alias_map:
                                        keys.append(alias_map[var])
                                    
                                    tc = None
                                    for k in keys:
                                        t_key = f'{k}_1dtuning'
                                        if t_key in rdata:
                                            tc_arr = rdata[t_key]
                                            if idx < tc_arr.shape[0]:
                                                if tc_arr.ndim == 3:
                                                    ci = 1 if tc_arr.shape[2] > 1 else 0
                                                    tc = tc_arr[idx, :, ci]
                                                else:
                                                    tc = tc_arr[idx, :]
                                            break
                                    
                                    if tc is None or np.any(np.isnan(tc)):
                                        has_nan = True
                                        break
                                if has_nan: continue
                                cell_list.append({'animal': animal, 'pos': pos, 'idx': idx})
                                count += 1
                                rec_added += 1
                                if rec_added >= 2: break
                                if count >= target_count: break

                                cell_list.append({'animal': animal, 'pos': pos, 'idx': idx})
                
                if count >= target_count: break
            if count >= target_count: break
        
        if len(cell_list) == 0:
            print(f"No cells found for area {target_area} (ID: {target_area_id}).")
        else:
            plot_stacked_tuning_curves(
                data,
                cell_list,
                variables=['theta', 'phi', 'pitch', 'roll', 'dTheta', 'dPhi', 'gyro_z'],
                save_path=f'stacked_tuning_curves_{target_area}.png'
            )
        
    except Exception as e:
        print(f"Could not run example: {e}")
