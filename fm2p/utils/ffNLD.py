# -*- coding: utf-8 -*-
"""
Neural decoding model to predict behavior from population activity.
Inverts the logic of ffNLE.py to perform decoding instead of encoding.

Author: DMM, 2025
"""

import torch
import numpy as np
import fm2p
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import matplotlib.cm as cm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BaseModel(nn.Module):
    def __init__(self, 
                    in_features, 
                    out_features, 
                    config,
                    ):
        super(BaseModel, self).__init__()

        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        self.activation_type = config['activation_type']
        self.loss_type = config.get('loss_type', 'mse')
        
        self.hidden_size = config.get('hidden_size', 0)
        self.dropout_p = config.get('dropout', 0.0)

        if self.hidden_size > 0:
            layers = [
                nn.Linear(self.in_features, self.hidden_size),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU()
            ]
            if self.dropout_p > 0:
                layers.append(nn.Dropout(p=self.dropout_p))
            layers.append(nn.Linear(self.hidden_size, self.out_features))
            self.Cell_NN = nn.Sequential(*layers)
        else:
            self.Cell_NN = nn.Sequential(nn.Linear(self.in_features, self.out_features, bias=True))

        self.activations = nn.ModuleDict({'SoftPlus':nn.Softplus(beta=0.5),
                                          'ReLU': nn.ReLU(),
                                          'Identity': nn.Identity(),
                                          'Sigmoid': nn.Sigmoid()})
        
        if self.config['initW'] == 'zero':
            for m in self.Cell_NN.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.uniform_(m.weight, a=-1e-6, b=1e-6)
                    if m.bias is not None:
                        m.bias.data.fill_(1e-6)
        elif self.config['initW'] == 'normal':
            for m in self.Cell_NN.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.normal_(m.weight, std=1/m.in_features)
        
        if isinstance(self.Cell_NN, nn.Sequential):
            self.Cell_NN[-1].bias.data.fill_(0.0)

        self.L1_alpha = config.get('L1_alpha')
        if self.L1_alpha != None:
            self.register_buffer('alpha',config['L1_alpha']*torch.ones(1))

    def forward(self, inputs):
        output = self.Cell_NN(inputs)
        if self.activation_type is not None and self.activation_type in self.activations:
            ret = self.activations[self.activation_type](output)
        else:
            ret = output
        return ret

    def loss(self, Yhat, Y): 
        if self.loss_type == 'poisson':
            loss_vec = torch.mean(Yhat - Y * torch.log(Yhat + 1e-8), axis=0)
        else:
            loss_vec = torch.mean((Yhat-Y)**2, axis=0)

        if self.L1_alpha != None:
            l1_reg0 = torch.stack([torch.linalg.vector_norm(NN_params, ord=1) for name, NN_params in self.Cell_NN.named_parameters() if '0.weight' in name])
            loss_vec = loss_vec + self.alpha*(l1_reg0)
        return loss_vec

    def get_weights(self):
        return {k: v.detach().cpu().numpy() for k, v in self.Cell_NN.state_dict().items()}

class DecodingModel(BaseModel):
    def __init__(self, in_features, out_features, config):
        super(DecodingModel, self).__init__(in_features, out_features, config)

def add_temporal_lags(X, lags):
    X_lagged = []
    for lag in lags:
        shifted = np.roll(X, shift=-lag, axis=0)
        if lag < 0: shifted[: -lag, :] = 0
        elif lag > 0: shifted[-lag :, :] = 0
        X_lagged.append(shifted)
    return np.concatenate(X_lagged, axis=1)

def setup_model_training(model, params, network_config):
    param_list = []
    for name, p in model.named_parameters():
        if ('weight' in name):
            param_list.append({'params':[p], 'lr':network_config['lr_w'], 'weight_decay':network_config['L2_lambda']})
        elif ('bias' in name):
            param_list.append({'params':[p], 'lr':network_config['lr_b']})

    if network_config['optimizer'].lower()=='adam':
        optimizer = optim.Adam(params=param_list)
    else:
        optimizer = optim.SGD(params=param_list)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    return optimizer, scheduler

def load_decoding_data(data_input, lags=None, device=device):
    if isinstance(data_input, (str, Path)):
        data = fm2p.read_h5(data_input)
    else:
        data = data_input

    # --- 1. Neural Data (Inputs) ---
    spikes = data.get('norm_dFF')
    if spikes is None:
        spikes = data.get('norm_spikes')
    if spikes is None:
        raise ValueError("Neural data (norm_dFF or norm_spikes) not found.")
    
    # Ensure (Time, Cells)
    if spikes.shape[0] < spikes.shape[1]: 
         spikes = spikes.T
    
    # Smoothing
    spikes_smoothed = np.zeros_like(spikes)
    for c in range(spikes.shape[1]):
        spikes_smoothed[:, c] = fm2p.convfilt(spikes[:, c], 10)
    
    X = spikes_smoothed
    n_cells = X.shape[1]

    # --- 2. Behavior Data (Targets) ---
    eyeT = data['eyeT'][data['eyeT_startInd']:data['eyeT_endInd']]
    eyeT = eyeT - eyeT[0]

    if 'dPhi' not in data.keys():
        phi_full = np.rad2deg(data['phi'][data['eyeT_startInd']:data['eyeT_endInd']])
        dPhi  = np.diff(fm2p.interp_short_gaps(phi_full, 5)) / np.diff(eyeT)
        dPhi = np.roll(dPhi, -2)
        data['dPhi'] = dPhi

    if 'dTheta' not in data.keys():
        theta_full = np.rad2deg(data['theta'][data['eyeT_startInd']:data['eyeT_endInd']])
        dTheta  = np.diff(fm2p.interp_short_gaps(theta_full, 5)) / np.diff(eyeT)
        dTheta = np.roll(dTheta, -2)
        data['dTheta'] = dTheta
        t = eyeT.copy()[:-1]
        t1 = t + (np.diff(eyeT) / 2)
        data['eyeT1'] = t1

    dTheta = fm2p.interp_short_gaps(data['dTheta'])
    dTheta = fm2p.interpT(dTheta, data['eyeT1'], data['twopT'])
    dPhi = fm2p.interp_short_gaps(data['dPhi'])
    dPhi = fm2p.interpT(dPhi, data['eyeT1'], data['twopT'])

    behaviors = {
        'theta': data.get('theta_interp'),
        'phi': data.get('phi_interp'),
        'yaw': data.get('head_yaw_deg'),
        'roll': data.get('roll_twop_interp'),
        'pitch': data.get('pitch_twop_interp'),
        'dTheta': dTheta,
        'dPhi': dPhi,
        'gyro_x': data.get('gyro_x_twop_interp'),
        'gyro_y': data.get('gyro_y_twop_interp'),
        'gyro_z': data.get('gyro_z_twop_interp')
    }
    
    behavior_names = [k for k, v in behaviors.items() if v is not None]
    Y_list = [behaviors[k] for k in behavior_names]
    
    min_len = min(len(arr) for arr in Y_list + [X])
    
    X = X[:min_len]
    Y_list = [arr[:min_len] for arr in Y_list]
    Y = np.stack(Y_list, axis=1)
    
    ltdk = data['ltdk_state_vec'][:min_len].copy()
    
    # Normalize X (Neural)
    X_mean = np.nanmean(X, axis=0)
    X_std = np.nanstd(X, axis=0)
    X_std[X_std == 0] = 1.0
    X = (X - X_mean) / X_std
    X[np.isnan(X)] = 0.0
    
    # Add lags to X
    if lags is not None:
        X = add_temporal_lags(X, lags)
        
    # Normalize Y (Behavior)
    Y_mean = np.nanmean(Y, axis=0)
    Y_std = np.nanstd(Y, axis=0)
    Y_std[Y_std == 0] = 1.0
    Y = (Y - Y_mean) / Y_std
    Y[np.isnan(Y)] = 0.0
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
    ltdk_tensor = torch.tensor(ltdk, device=device)
    
    return X_tensor, Y_tensor, behavior_names, ltdk_tensor, n_cells

def compute_cell_importance(model, X_test, Y_test, behavior_names, n_cells, lags, device=device):
    model.eval()
    X_np = X_test.cpu().numpy()
    Y_np = Y_test.cpu().numpy()
    
    with torch.no_grad():
        y_hat = model(X_test).cpu().numpy()
        
    baseline_r2 = np.zeros(len(behavior_names))
    for b in range(len(behavior_names)):
        ss_res = np.sum((Y_np[:, b] - y_hat[:, b]) ** 2)
        ss_tot = np.sum((Y_np[:, b] - np.mean(Y_np[:, b])) ** 2)
        baseline_r2[b] = 1 - (ss_res / (ss_tot + 1e-8))
        
    importances = np.zeros((n_cells, len(behavior_names)))
    
    n_lags = len(lags) if lags is not None else 1
    
    for c in tqdm(range(n_cells), desc="Computing Cell Importance"):
        X_shuff = X_np.copy()
        
        # Shuffle all lagged columns corresponding to this cell
        # Columns are ordered: [Lag1_features, Lag2_features, ...]
        # where LagK_features has columns for all cells in order.
        for l in range(n_lags):
            col_idx = c + (l * n_cells)
            np.random.shuffle(X_shuff[:, col_idx])
            
        X_shuff_tensor = torch.tensor(X_shuff, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            y_hat_shuff = model(X_shuff_tensor).cpu().numpy()
            
        for b in range(len(behavior_names)):
            ss_res = np.sum((Y_np[:, b] - y_hat_shuff[:, b]) ** 2)
            ss_tot = np.sum((Y_np[:, b] - np.mean(Y_np[:, b])) ** 2)
            shuff_r2 = 1 - (ss_res / (ss_tot + 1e-8))
            importances[c, b] = baseline_r2[b] - shuff_r2
            
    return importances, baseline_r2

def plot_decoding_results_pdf(results, save_path):
    behavior_names = results['decoding_behavior_names']
    baseline_r2 = results['decoding_baseline_r2']
    importances = results['decoding_cell_importances']
    y_true = results['y_true_test']
    y_pred = results['y_pred_test']
    
    with PdfPages(save_path) as pdf:
        # Page 1: Performance Summary
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(behavior_names, baseline_r2, color='skyblue', edgecolor='black')
        ax.set_ylabel('R² Score')
        ax.set_title('Decoding Performance (Baseline R²)')
        y_min = min(0, np.min(baseline_r2)) - 0.1
        y_max = max(1, np.max(baseline_r2)) + 0.1
        ax.set_ylim(y_min, y_max)
        plt.xticks(rotation=45, ha='right')
        ax.axhline(0, color='k', linewidth=0.8)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        # Page 2: Importance Heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        v_abs = np.max(np.abs(importances))
        if v_abs == 0: v_abs = 0.1
        im = ax.imshow(importances.T, aspect='auto', cmap='RdBu_r', vmin=-v_abs, vmax=v_abs)
        plt.colorbar(im, label='R² Drop (Importance)')
        ax.set_yticks(range(len(behavior_names)))
        ax.set_yticklabels(behavior_names)
        ax.set_xlabel('Cells')
        ax.set_title('Cell Importance for Behavior Decoding')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        
        # Page 3+: Time Series (first 1000 frames)
        n_plot = min(1000, y_true.shape[0])
        t = np.arange(n_plot)
        plots_per_page = 3
        n_pages = int(np.ceil(len(behavior_names) / plots_per_page))
        
        for p in range(n_pages):
            fig, axs = plt.subplots(plots_per_page, 1, figsize=(10, 12), sharex=True)
            if plots_per_page == 1: axs = [axs]
            start_idx = p * plots_per_page
            for i in range(plots_per_page):
                b_idx = start_idx + i
                if b_idx < len(behavior_names):
                    ax = axs[i]
                    ax.plot(t, y_true[:n_plot, b_idx], 'k', label='True', alpha=0.6, linewidth=1)
                    ax.plot(t, y_pred[:n_plot, b_idx], 'r--', label='Predicted', alpha=0.8, linewidth=1)
                    ax.set_title(f'{behavior_names[b_idx]} (R² = {baseline_r2[b_idx]:.3f})')
                    ax.set_ylabel('Normalized Value')
                    if i == 0: ax.legend(loc='upper right', fontsize='small')
                else:
                    axs[i].axis('off')
            if len(axs) > 0: axs[-1].set_xlabel('Time (frames)')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

def fit_decoding_model(data_input, save_dir=None):
    if isinstance(data_input, (str, Path)):
        if save_dir is None:
            save_dir = os.path.split(data_input)[0]
            
    data = fm2p.check_and_trim_imu_disconnect(data_input)
    
    config = {
        'activation_type': 'Identity',
        'loss_type': 'mse',
        'initW': 'normal',
        'optimizer': 'adam',
        'lr_w': 1e-3, 
        'lr_b': 1e-3,
        'L1_alpha': 1e-3,
        'Nepochs': 3000,
        'L2_lambda': 1e-3,
        'lags': np.arange(-10, 1, 1),
        'hidden_size': 256,
        'dropout': 0.3
    }
    
    X, Y, behavior_names, ltdk, n_cells = load_decoding_data(data, lags=config['lags'], device=device)
    
    n_samples = X.shape[0]
    n_chunks = 20
    indices = np.arange(n_samples)
    chunks = np.array_split(indices, n_chunks)
    chunk_indices = np.arange(n_chunks)
    np.random.seed(42)
    np.random.shuffle(chunk_indices)
    
    split_idx = int(0.8 * n_chunks)
    train_indices = np.sort(np.concatenate([chunks[i] for i in chunk_indices[:split_idx]]))
    test_indices = np.sort(np.concatenate([chunks[i] for i in chunk_indices[split_idx:]]))
    
    X_train, Y_train = X[train_indices], Y[train_indices]
    X_test, Y_test = X[test_indices], Y[test_indices]
    
    in_features = X.shape[1]
    out_features = Y.shape[1]
    
    model = DecodingModel(in_features, out_features, config).to(device)
    optimizer, scheduler = setup_model_training(model, {'ModelID': 0}, config)
    
    print("Training Decoding Model...")
    model.train()
    for epoch in range(config['Nepochs']):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = model.loss(outputs, Y_train)
        loss.sum().backward()
        optimizer.step()
        scheduler.step(loss.sum())
        
    importances, baseline_r2 = compute_cell_importance(
        model, X_test, Y_test, behavior_names, n_cells, config['lags'], device=device
    )
    
    model.eval()
    with torch.no_grad():
        y_hat_test = model(X_test).cpu().numpy()
    y_true_test = Y_test.cpu().numpy()

    print("\nBaseline R2 per behavior:")
    for b, name in enumerate(behavior_names):
        print(f"  {name}: {baseline_r2[b]:.4f}")
        
    results = {
        'decoding_baseline_r2': baseline_r2,
        'decoding_cell_importances': importances,
        'decoding_behavior_names': behavior_names,
        'decoding_model_weights': model.get_weights(),
        'y_true_test': y_true_test,
        'y_pred_test': y_hat_test
    }
    
    if save_dir:
        save_path = os.path.join(save_dir, 'neural_decoding_results.h5')
        fm2p.write_h5(save_path, results)
        print(f"Results saved to {save_path}")

        pdf_path = os.path.join(save_dir, 'decoding_summary.pdf')
        plot_decoding_results_pdf(results, pdf_path)
        print(f"Summary PDF saved to {pdf_path}")

    return results


def plot_region_summary(results_list, save_path):
    
    region_data = {} # {region: {behavior: [r2 values]}}
    
    for res in results_list:
        rname = res['region_name']
        r2s = res['decoding_baseline_r2']
        bnames = res['decoding_behavior_names']
        
        if rname not in region_data:
            region_data[rname] = {}
        
        for b, r2 in zip(bnames, r2s):
            if b not in region_data[rname]:
                region_data[rname][b] = []
            region_data[rname][b].append(r2)
            
    behaviors = sorted(list(set([b for r in region_data for b in region_data[r]])))
    regions = sorted(list(region_data.keys()))
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    bar_width = 0.8 / len(regions)
    x = np.arange(len(behaviors))
    
    colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(regions)))
    
    for i, rname in enumerate(regions):
        means = []
        sems = []
        for b in behaviors:
            vals = region_data[rname].get(b, [])
            if vals:
                means.append(np.mean(vals))
                sems.append(np.std(vals) / np.sqrt(len(vals)))
            else:
                means.append(0)
                sems.append(0)
        
        offset = (i - len(regions)/2) * bar_width + bar_width/2
        ax.bar(x + offset, means, yerr=sems, width=bar_width, label=rname, color=colors[i], capsize=2)
        
    ax.set_xticks(x)
    ax.set_xticklabels(behaviors, rotation=45, ha='right')
    ax.set_ylabel('Decoding $R^2$')
    ax.set_title('Behavior Decoding Performance by Visual Area')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    with PdfPages(save_path) as pdf:
        pdf.savefig(fig)
        plt.close(fig)


def run_analysis_from_topography(topo_path, save_dir=None):
    if save_dir is None:
        save_dir = os.path.dirname(topo_path)
        
    print(f"Loading topography data from {topo_path}")
    topo_data = fm2p.read_h5(topo_path)
    
    if 'raw_data_for_modeling' not in topo_data:
        print("No raw_data_for_modeling found in file.")
        return

    raw_data = topo_data['raw_data_for_modeling']
    
    label_map = {2: 'RL', 3: 'AM', 4: 'PM', 5: 'V1'}
    target_regions = [2, 3, 4, 5]
    
    all_results = []
    
    for rec_id, rec_data in tqdm(raw_data.items(), desc="Processing recordings"):
        if 'cell_regions' not in rec_data:
            print(f"Skipping {rec_id}: no cell_regions found.")
            continue
            
        regions = rec_data['cell_regions']
        unique_regions = np.unique(regions)
        
        for r in unique_regions:
            if r not in target_regions:
                continue
                
            region_name = label_map[r]
            cell_mask = (regions == r)
            n_cells_region = np.sum(cell_mask)
            
            if n_cells_region < 10:
                continue
                
            sub_data = rec_data.copy()
            
            # Handle spikes shape (n_cells, n_frames) or (n_frames, n_cells)
            # topography.py saves pdata['norm_spikes'] which is usually (n_cells, n_frames)
            spikes = sub_data['norm_spikes']
            if spikes.shape[0] == len(regions):
                sub_data['norm_spikes'] = spikes[cell_mask, :]
            elif spikes.shape[1] == len(regions):
                sub_data['norm_spikes'] = spikes[:, cell_mask]
            
            if 'norm_dFF' in sub_data:
                dff = sub_data['norm_dFF']
                if dff.shape[0] == len(regions):
                    sub_data['norm_dFF'] = dff[cell_mask, :]
                elif dff.shape[1] == len(regions):
                    sub_data['norm_dFF'] = dff[:, cell_mask]
            
            res = fit_decoding_model(sub_data, save_dir=None)
            res['region_name'] = region_name
            res['rec_id'] = rec_id
            all_results.append(res)
            
    pdf_path = os.path.join(save_dir, 'topography_decoding_summary.pdf')
    plot_region_summary(all_results, pdf_path)
    print(f"Summary saved to {pdf_path}")


if __name__ == '__main__':

    ### TEST ON SINGLE RECORDING
    # data = fm2p.read_h5(
    #     '/home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings/cohort02_recordings/251021_DMM_DMM061_pos04/fm1/251021_DMM_DMM061_fm_01_preproc.h5'
    # )
    # fit_decoding_model(data)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--topo', type=str,
        default='topography_analysis_results_v09e.h5',
        help='Path to topography analysis results HDF5'
    )
    args = parser.parse_args()

    if args.topo:
        run_analysis_from_topography(args.topo)
    else:
        ### BATCH PROCESS
        cohort_dir = '/home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings/cohort02_recordings/'
        # cohort_dir = '/home/dylan/Storage/freely_moving_data/_V1PPC/cohort01_recordings/'
        recordings = fm2p.find(
            '*fm*_preproc.h5',
            cohort_dir
        )
        print('Found {} recordings.'.format(len(recordings)))

        for ri, rec in tqdm(enumerate(recordings)):
            print('Fitting models for recordings {} of {} ({}).'.format(ri+1, len(recordings), rec))
            fit_decoding_model(rec)
    
    
