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


if __name__ == '__main__':

    plot_behavior_overview(
        '/home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings/cohort02_recordings/251021_DMM_DMM061_pos04/fm1/251021_DMM_DMM061_fm_01_preproc.h5',
        save_path = 'poster_plot_raw_data_demo.svg',
        time_range=[985,985+60]
    )
