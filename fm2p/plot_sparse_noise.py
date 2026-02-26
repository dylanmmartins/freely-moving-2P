

# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import fm2p

def visualize_sparse_noise_rf():
    # Path to the merged essentials file
    path = '/home/dylan/Storage/freely_moving_data/_V1PPC/mouse_composites/DMM056/DMM056_merged_essentials_v8.h5'
    
    # Read the merged data
    print(f"Reading {path}")
    data = fm2p.read_h5(path)
    
    animal_id = 'DMM056'
    # Directory where original recordings are stored
    cohort_dir = '/home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings/'
    
    all_real_stas = []
    
    # Get all position keys (e.g., 'pos01', 'pos02')
    keys = sorted([k for k in data.keys() if k.startswith('pos')])
    
    print(f"Found positions: {keys}")
    
    # Find all sparse_noise.h5 files in the cohort directory once to avoid repeated walking
    print("Searching for sparse_noise.h5 files...")
    try:
        all_sn_files = fm2p.find('sparse_noise.h5', cohort_dir)
    except FileNotFoundError:
        print("No sparse_noise.h5 files found in cohort directory.")
        return

    for pos_key in keys:
        pos_data = data[pos_key]
        
        # Check for sparse noise centroids/indices
        # 'sn_cents' contains [true_indices, centroid_x, centroid_y]
        if 'sn_cents' not in pos_data:
            continue
            
        sn_cents = pos_data['sn_cents']
        
        # Check if it's valid (not NaN)
        if np.all(np.isnan(sn_cents)):
            continue
            
        if sn_cents.ndim > 1 and sn_cents.shape[1] >= 1:
            true_indices = sn_cents[:, 0].astype(int)
        else:
            continue
            
        if len(true_indices) == 0:
            continue
            
        print(f"Position {pos_key} has {len(true_indices)} real RFs.")

        # Filter for the file corresponding to this animal and position
        # The path should contain the animal ID and the position key (e.g., .../DMM056_..._pos01/...)
        candidates = [f for f in all_sn_files if animal_id in f and pos_key in f]
        
        if not candidates:
            print(f"  No sparse noise file found for {pos_key}")
            continue
            
        # Use the most recent one if multiple found
        sn_path = fm2p.choose_most_recent(candidates)
        print(f"  Loading {sn_path}")
        
        try:
            sn_data = fm2p.read_h5(sn_path)
            
            if 'STA' in sn_data:
                sta_stack = sn_data['STA']
                
                # Iterate through the indices of cells labeled as 'real'
                for idx in true_indices:
                    if idx < len(sta_stack):
                        sta = sta_stack[idx]
                        
                        # Reshape if flattened (assuming 768x1360 based on review_STAs.py)
                        if sta.ndim == 1:
                            try:
                                sta = sta.reshape(768, 1360)
                            except ValueError:
                                print(f"    Shape mismatch for STA {idx}, size {sta.size}")
                                continue
                                
                        all_real_stas.append(sta)
                        
                        if len(all_real_stas) >= 24:
                            break
        except Exception as e:
            print(f"  Error processing {sn_path}: {e}")
            
        if len(all_real_stas) >= 24:
            break
    
    # Plotting
    if not all_real_stas:
        print("No receptive fields found to visualize.")
        return

    print(f"Plotting {len(all_real_stas)} receptive fields.")
    
    num_plot = min(len(all_real_stas), 24)
    cols = 6
    rows = int(np.ceil(num_plot / cols))
    
    fig, axs = plt.subplots(rows, cols, figsize=(16, 3*rows), dpi=300)
    axs = axs.flatten()
    
    for i in range(num_plot):
        ax = axs[i]
        sta = all_real_stas[i]
        
        vmax = np.max(np.abs(sta))
        ax.imshow(sta, cmap='coolwarm', vmin=-vmax, vmax=vmax)
        ax.axis('off')
        # ax.set_title(f'RF {i+1}')
        
    # Hide unused subplots
    for i in range(num_plot, len(axs)):
        axs[i].axis('off')
        
    plt.tight_layout()
    plt.show()
    save_path = os.path.join(os.path.dirname(path), 'sparse_noise_rfs_first24.png')
    plt.savefig(save_path)
    print(f"Figure saved to {save_path}")

if __name__ == '__main__':
    visualize_sparse_noise_rf()
