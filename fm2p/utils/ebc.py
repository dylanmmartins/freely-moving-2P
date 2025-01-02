import numpy as np
import matplotlib.pyplot as plt

def calculate_egocentric_rate_map(trajectory_data, spike_times, boundaries, distance_bins, angle_bins):
    """
    Calculates the egocentric boundary cell rate map.

    Args:
        trajectory_data (np.array): Array of shape (n_time_bins, 3) containing 
                                    position (x, y) and head direction (theta) at each time bin.
        spike_times (np.array): Array of spike times.
        boundaries (np.array): Array of shape (n_boundary_points, 2) defining the environment boundaries.
        distance_bins (np.array): Array defining the distance bin edges.
        angle_bins (np.array): Array defining the angle bin edges (in radians).

    Returns:
        rate_map (np.array): 2D array representing the firing rate map in egocentric coordinates.
                             Rows correspond to distance bins, columns correspond to angle bins.
    """

    n_distance_bins = len(distance_bins) - 1
    n_angle_bins = len(angle_bins) - 1
    rate_map = np.zeros((n_distance_bins, n_angle_bins))
    occupancy_map = np.zeros((n_distance_bins, n_angle_bins))

    dt = trajectory_data[1,0] - trajectory_data[0,0]

    spike_indices = np.floor(spike_times / dt).astype(int)

    for t_idx, (time, x, y, theta) in enumerate(trajectory_data):
      
        # Calculate egocentric distance and angle to the nearest boundary
        min_distance = float('inf')
        min_angle = 0

        for bx, by in boundaries:
            dx = bx - x
            dy = by - y
            distance = np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx) - theta
            angle = np.arctan2(np.sin(angle), np.cos(angle))  # Normalize angle to [-pi, pi]

            if distance < min_distance:
                min_distance = distance
                min_angle = angle

        # Find the corresponding bins
        distance_bin_idx = np.digitize(min_distance, distance_bins) - 1
        angle_bin_idx = np.digitize(min_angle, angle_bins) - 1

        # Handle edge cases
        if 0 <= distance_bin_idx < n_distance_bins and 0 <= angle_bin_idx < n_angle_bins:
            occupancy_map[distance_bin_idx, angle_bin_idx] += 1
            if t_idx in spike_indices:
              rate_map[distance_bin_idx, angle_bin_idx] += 1

    # Calculate firing rate
    rate_map = np.divide(rate_map, occupancy_map, out=np.zeros_like(rate_map), where=occupancy_map!=0)
    
    return rate_map

def plot_rate_map(rate_map, distance_bins, angle_bins):
    """Plots the egocentric rate map."""
    
    plt.figure(figsize=(8, 6))
    extent = [angle_bins[0], angle_bins[-1], distance_bins[0], distance_bins[-1]]
    plt.imshow(rate_map, extent=extent, origin='lower', aspect='auto', cmap='viridis')
    plt.xlabel("Egocentric Angle (rad)")
    plt.ylabel("Egocentric Distance")
    plt.title("Egocentric Boundary Cell Rate Map")
    plt.colorbar(label="Firing Rate (Hz)")
    plt.show()


if __name__ == '__main__':
    # Generate sample data (replace with your actual data)
    time = np.arange(0, 10, 0.1)  # 10 seconds, 100 ms time bins
    x = np.cos(time) * 5  # Example trajectory
    y = np.sin(time) * 5
    theta = time  # Head direction
    trajectory_data = np.column_stack([time, x, y, theta])
    
    spike_times = np.array([0.5, 1.2, 2.8, 4.5, 6.1, 8.7])  # Example spike times

    # Define environment boundaries (example: square)
    boundaries = np.array([
        [-6, -6], [-6, 6], [6, 6], [6, -6]
    ])

    # Define bins
    distance_bins = np.arange(0, 10, 0.5)  # Distance bins from 0 to 10 cm, 0.5 cm width
    angle_bins = np.arange(-np.pi, np.pi, np.pi/12)  # Angle bins from -pi to pi, 15 degrees width

    # Calculate the rate map
    rate_map = calculate_egocentric_rate_map(trajectory_data, spike_times, boundaries, distance_bins, angle_bins)

    # Plot the rate map
    plot_rate_map(rate_map, distance_bins, angle_bins)