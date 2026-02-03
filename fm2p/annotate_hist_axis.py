
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fm2p
from skimage.measure import label, regionprops
from PIL import Image


def process_shapes_with_border_logic(image, points_to_check):

    label_map = {
        0: 'boundary',
        1: 'outside',
        2: 'RL',
        3: 'RLL',
        4: 'MMA',
        5: 'AM',
        6: 'PM',
        7: 'V1',
        8: 'MMP',
        9: 'AL',
        10: 'LM',
        11: 'P'
    }

    labeled_array = label(image, connectivity=1)

    border_labels = set()
    rows, cols = labeled_array.shape

    border_labels.update(np.unique(labeled_array[0, :]))      # Top row
    border_labels.update(np.unique(labeled_array[rows-1, :])) # Bottom row
    border_labels.update(np.unique(labeled_array[:, 0]))      # Left col
    border_labels.update(np.unique(labeled_array[:, cols-1])) # Right col

    all_labels = np.unique(labeled_array)
    all_labels = all_labels[all_labels != 0]

    results = []
    for i, (x, y) in enumerate(points_to_check):
        if 0 <= y < rows and 0 <= x < cols:
            # display(x, y)
            region_id = labeled_array[int(y), int(x)]
            category = label_map.get(region_id, "Unknown")
            results.append([i, y, x, region_id])
        else:
            results.append([i, y, x, -1])

    results = np.array(results)

    return results, labeled_array, label_map


def analyze_scatter_along_line(data, window_size=25):
    """
    data: list or array of tuples/rows format [(ind, x, y, value), ...]
    window_size: integer for the running average window
    """
    # 1. Prepare Data
    data = np.array(data)
    ids = data[:, 0]
    y = data[:, 1] # was x
    x = data[:, 2] # was y
    values = data[:, 3]

    # 2. Setup Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 12))
    
    # Scatter plot
    sc = ax1.scatter(x, y, c=values, cmap='coolwarm', vmin=-15, vmax=15)
    ax1.set_title("1. Click the START and END points of your line")
    ax1.set_xlabel("M/L axis")
    ax1.set_ylabel("A/P axis")
    ax1.invert_yaxis()
    plt.colorbar(sc, ax=ax1, label='Value')
    
    plt.tight_layout()
    plt.draw()

    # 3. Interactive Input (Blocking)
    # This will pause execution until the user clicks 2 times on the plot
    print("Please click two points on the plot to define the line...")
    points = plt.ginput(n=2, timeout=0)
    
    if len(points) < 2:
        print("Selection cancelled or incomplete.")
        return

    (x1, y1), (x2, y2) = points
    
    # Draw the selected line for visual confirmation
    ax1.plot([x1, x2], [y1, y2], 'r-', linewidth=2, label='Selected Axis')
    ax1.scatter([x1, x2], [y1, y2], color='red', s=100, zorder=5)
    ax1.legend()
    ax1.set_title("Line Selected. Calculating Projections...")
    fig.canvas.draw()

    # 4. Vector Math: Project points onto the line
    # Line Vector (L)
    L = np.array([x2 - x1, y2 - y1])
    
    # Vector from line start to every point (P)
    P_x = x - x1
    P_y = y - y1
    P = np.column_stack((P_x, P_y))
    
    # Project P onto L:  (P . L) / |L|
    # This gives the scalar distance 'd' along the line from the start point
    L_norm = np.linalg.norm(L)
    dot_products = P @ L  # Matrix multiplication for dot product
    distances = dot_products / L_norm

    # 5. Calculate Running Average
    # Create a DataFrame to sort easily by distance along the line
    df = pd.DataFrame({
        'distance': distances,
        'value': values
    })
    
    # Sort by distance so the rolling window makes sense spatially
    df = df.sort_values(by='distance')
    
    # Calculate rolling mean
    df['rolling_avg'] = df['value'].rolling(window=window_size, center=True).mean()

    # 6. Plot the Result
    ax2.plot(df['distance'], df['rolling_avg'], 'b-', linewidth=2, label=f'Running Avg (Window={window_size})')
    ax2.scatter(df['distance'], df['value'], alpha=0.3, color='gray', s=10, label='Projected Points')
    
    ax2.set_title("Running Average along Selected Axis")
    ax2.set_xlabel("Distance along line from Start Point")
    ax2.set_ylabel("Value")
    ax2.legend()
    ax2.grid(True)
    
    plt.show()


def main():

    img = Image.open('/home/dylan/Desktop/V1_HVAs_trace.png').convert("RGBA")
    img_array = np.array(img)

    data = fm2p.read_h5('/home/dylan/Fast1/freely_moving_data/pooled_datasets/pooled_260127.h5')

    animal_dirs = ['DMM037', 'DMM041', 'DMM042', 'DMM056', 'DMM061']
    # main_basepath = '/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites'

    key = 'phi'
    cond = 'l'

    h_hist_data = []
    v_hist_data = []

    for animal_dir in animal_dirs:
        
        kca_data = data[key][cond][animal_dir]['messentials']

        for poskey in data[key][cond][animal_dir]['transform'].keys():

            if (animal_dir=='DMM056') and (cond=='d') and ((poskey=='pos15') or (poskey=='pos03')):
                continue

            for c in range(np.size(data[key][cond][animal_dir]['messentials'][poskey]['rdata']['{}_{}_isrel'.format(key,cond)], 0)):
                cellx = data[key][cond][animal_dir]['transform'][poskey][c,2] # was 2
                celly = data[key][cond][animal_dir]['transform'][poskey][c,3] # was 3
                cellrel = data[key][cond][animal_dir]['messentials'][poskey]['rdata']['{}_{}_isrel'.format(key, cond)][c]
                cellmod = data[key][cond][animal_dir]['messentials'][poskey]['rdata']['{}_{}_mod'.format(key, cond)][c]

                # if cellrel:
                    # cellmod = data[key][cond][animal_dir]['messentials'][poskey]['rdata']['{}_{}_mod'.format(key, cond)][c]

                if cellrel and (cellmod > 0.33):
                    if cond == 'l':
                        condint = 1
                    elif cond == 'd':
                        condint = 0
                    cellpeak = data[key][cond][animal_dir]['messentials'][poskey]['rdata']['{}_1dbins'.format(key)][np.argmax(data[key][cond][animal_dir]['messentials'][poskey]['rdata']['{}_1dtuning'.format(key)][c,:,condint])]


                    h_hist_data.append([cellx, cellpeak])
                    v_hist_data.append([celly, cellpeak])

                    # h_hist_data.append([cellx, cellmod])
                    # v_hist_data.append([celly, cellmod])

    h_hist_data = np.array(h_hist_data)
    v_hist_data = np.array(v_hist_data)

    points = []
    for i in range(np.size(h_hist_data,0)):
        points.append((h_hist_data[i,0], v_hist_data[i,0]))

    area_num = 5 # 7 is V1, 5 is AM

    results, labeled_array, label_map = process_shapes_with_border_logic(img_array[:,:,0].clip(max=1), points)

    usedata = results[results[:,3] == area_num, :].copy()

    usedata[:,3] = h_hist_data[results[:,3] == area_num, 1].copy()

    analyze_scatter_along_line(usedata, window_size=100)


if __name__ == '__main__':
    main()
