import torch
import numpy as np
import fm2p
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseModel(nn.Module):
    def __init__(self, 
                    in_features, 
                    N_cells,
                    config,
                    ):
        super(BaseModel, self).__init__()

        self.config = config
        self.in_features = in_features
        self.N_cells = N_cells
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
            layers.append(nn.Linear(self.hidden_size, self.N_cells))
            self.Cell_NN = nn.Sequential(*layers)
        else:
            self.Cell_NN = nn.Sequential(nn.Linear(self.in_features, self.N_cells,bias=True))

        self.activations = nn.ModuleDict({'SoftPlus':nn.Softplus(beta=0.5),
                                          'ReLU': nn.ReLU(),
                                          'Identity': nn.Identity()})
        
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

        self.L1_alpha = config['L1_alpha']
        if self.L1_alpha != None:
            self.register_buffer('alpha',config['L1_alpha']*torch.ones(1))

      
    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight,a=-1e-6,b=1e-6)
            m.bias.data.fill_(1e-6)
        
    def forward(self, inputs, pos_inputs=None):
        output = self.Cell_NN(inputs)
        if self.activation_type is not None and self.activation_type in self.activations:
            ret = self.activations[self.activation_type](output)
        else:
            ret = output
        return ret

    def loss(self,Yhat, Y): 

        if self.loss_type == 'poisson':
            loss_vec = torch.mean(Yhat - Y * torch.log(Yhat + 1e-8), axis=0)
        else:
            loss_vec = torch.mean((Yhat-Y)**2,axis=0)

        if self.L1_alpha != None:
            l1_reg0 = torch.stack([torch.linalg.vector_norm(NN_params,ord=1) for name, NN_params in self.Cell_NN.named_parameters() if '0.weight' in name])
            loss_vec = loss_vec + self.alpha*(l1_reg0)
        return loss_vec

    def get_weights(self):
        return {k: v.detach().cpu().numpy() for k, v in self.Cell_NN.state_dict().items()}


class PositionGLM(BaseModel):
    def __init__(self, 
                    in_features, 
                    N_cells, 
                    config,
                    device=device):
        super(PositionGLM, self).__init__(in_features, N_cells, config)
        
        self.L1_alpha = config.get('L1_alpha_m')
        if self.L1_alpha is None:
             self.L1_alpha = config.get('L1_alpha')

        if self.L1_alpha is not None:
            self.register_buffer('alpha', self.L1_alpha * torch.ones(1))

    def forward(self, inputs, pos_inputs=None):

        return super(PositionGLM, self).forward(inputs)
    

def arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--Nepochs',            type=int,         default=5000)
    args = parser.parse_args()
    return vars(args)


def add_temporal_lags(X, lags):
    """
    Add temporal lags to the input features.
    Args:
        X (np.array): Input features of shape (n_samples, n_features)
        lags (list): List of integers representing lags (e.g., [-2, -1, 0, 1, 2])
    Returns:
        X_lagged (np.array): Lagged features of shape (n_samples, n_features * len(lags))
    """
    X_lagged = []
    for lag in lags:
        shifted = np.roll(X, shift=-lag, axis=0)
        if lag < 0: shifted[: -lag, :] = 0
        elif lag > 0: shifted[-lag :, :] = 0
        X_lagged.append(shifted)
    return np.concatenate(X_lagged, axis=1)


def load_position_data(h5_path, modeltype='full', lags=None, device=device):

    data = fm2p.read_h5(h5_path)
    
    theta = data.get('theta_interp')
    phi = data.get('phi_interp')

    yaw = data['head_yaw_deg']

    roll = data.get('roll_twop_interp', np.zeros_like(theta))
    pitch = data.get('pitch_twop_interp', np.zeros_like(theta))

    gyro_x = data['gyro_x_twop_interp']
    gyro_y = data['gyro_y_twop_interp']
    gyro_z = data['gyro_z_twop_interp']

    eyeT = data['eyeT'][data['eyeT_startInd']:data['eyeT_endInd']]
    eyeT = eyeT - eyeT[0]

    if 'dPhi' not in data.keys():
        phi_full = np.rad2deg(data['phi'][data['eyeT_startInd']:data['eyeT_endInd']])
        dPhi  = np.diff(fm2p.interp_short_gaps(phi_full, 5)) / np.diff(eyeT)
        dPhi = np.roll(dPhi, -2)
        data['dPhi'] = dPhi

    if 'dTheta' not in data.keys() and 'dEye' not in data.keys():
        theta_full = np.rad2deg(data['theta'][data['eyeT_startInd']:data['eyeT_endInd']])
        dTheta  = np.diff(fm2p.interp_short_gaps(theta_full, 5)) / np.diff(eyeT)
        dTheta = np.roll(dTheta, -2)
        data['dTheta'] = dTheta

        t = eyeT.copy()[:-1]
        t1 = t + (np.diff(eyeT) / 2)
        data['eyeT1'] = t1

    elif 'dTheta' not in data.keys():
        data['dTheta'] = data['dEye'].copy()

    dTheta = fm2p.interp_short_gaps(data['dTheta'])
    dTheta = fm2p.interpT(dTheta, data['eyeT1'], data['twopT'])
    dPhi = fm2p.interp_short_gaps(data['dPhi'])
    dPhi = fm2p.interpT(dPhi, data['eyeT1'], data['twopT'])

    ltdk = data['ltdk_state_vec'].copy()
    
    min_len = min(
        len(theta),
        len(phi),
        len(yaw),
        len(roll),
        len(pitch),
        len(ltdk),
        len(dTheta),
        len(dPhi),
        len(gyro_x),
        len(gyro_y),
        len(gyro_z)
    )

    print(
        len(theta),
        len(phi),
        len(yaw),
        len(roll),
        len(pitch),
        len(ltdk),
        len(dTheta),
        len(dPhi),
        len(gyro_x),
        len(gyro_y),
        len(gyro_z)
    )
    
    spikes = data.get('norm_dFF')
    if spikes is None:
        raise ValueError("norm_spikes not found in HDF5 file.")
    
    for c in range(np.size(spikes, 0)):
        spikes[c,:] = fm2p.convfilt(spikes[c,:], 10)
        
    min_len = min(min_len, spikes.shape[1])
    
    theta = theta[:min_len]
    phi = phi[:min_len]
    yaw = yaw[:min_len]
    roll = roll[:min_len]
    pitch = pitch[:min_len]
    spikes = spikes[:, :min_len]
    spikes = spikes.T
    ltdk = ltdk[:min_len]
    dTheta = dTheta[:min_len]
    dPhi = dPhi[:min_len]
    gyro_x = gyro_x[:min_len]
    gyro_y = gyro_y[:min_len]
    gyro_z = gyro_z[:min_len]
    
    if modeltype == 'full':
        X = np.stack([theta, phi, yaw, roll, pitch, dTheta, dPhi, gyro_x, gyro_y, gyro_z], axis=1)
    elif modeltype == 'theta':
        X = np.stack([theta, dTheta], axis=1)
    elif modeltype == 'phi':
        X = np.stack([phi, dPhi], axis=1)
    elif modeltype == 'yaw':
        X = np.stack([yaw, gyro_z], axis=1)
    elif modeltype == 'roll':
        X = np.stack([roll, gyro_x], axis=1)
    elif modeltype == 'pitch':
        X = np.stack([pitch, gyro_y], axis=1)
    else:
        raise ValueError("Invalid modeltype. Choose from 'full', 'theta', 'phi', 'yaw', 'roll', 'pitch'.")
    
    X_mean = np.nanmean(X, axis=0)
    X_std = np.nanstd(X, axis=0)
    X_std[X_std == 0] = 1.0
    X = (X - X_mean) / X_std
    X[np.isnan(X)] = 0.0

    # Smooth inputs to reduce jitter in predictions
    # for i in range(X.shape[1]):
    #     X[:, i] = fm2p.convfilt(X[:, i], box_pts=10)

    if lags is not None:
        X = add_temporal_lags(X, lags)
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    if np.isnan(spikes).any():
        spikes[np.isnan(spikes)] = 0.0
    
    spikes_mean = np.nanmean(spikes, axis=0)
    spikes_std = np.nanstd(spikes, axis=0)
    spikes_std[spikes_std == 0] = 1.0
    spikes = (spikes - spikes_mean) / spikes_std

    print(f"Target (dF/F) stats (Z-scored) -- Mean: {np.nanmean(spikes):.4f}, Std: {np.nanstd(spikes):.4f}, Max: {np.nanmax(spikes):.4f}")
    Y_tensor = torch.tensor(spikes, dtype=torch.float32).to(device)
    
    return X_tensor, Y_tensor



def setup_model_training(model,params,network_config):

    check_names = []
    param_list = []
    if params['train_shifter']:
        param_list.append({'params': list(model.shifter_nn.parameters()),'lr': network_config['lr_shift'],'weight_decay':.0001})
    for name,p in model.named_parameters():
        if params['ModelID']<2:
            if ('Cell_NN' in name):
                if ('weight' in name):
                    param_list.append({'params':[p],'lr':network_config['lr_w'],'weight_decay':network_config['L2_lambda']})
                elif ('bias' in name):
                    param_list.append({'params':[p],'lr':network_config['lr_b']})
                check_names.append(name)
        elif (params['ModelID']==2) | (params['ModelID']==3):
            if ('posNN' in name):
                if ('weight' in name):
                    param_list.append({'params':[p],'lr':network_config['lr_w'],'weight_decay':network_config['L2_lambda_m']})
                elif ('bias' in name):
                    param_list.append({'params':[p],'lr':network_config['lr_b']})
                check_names.append(name)

    if network_config['optimizer'].lower()=='adam':
        optimizer = optim.Adam(params=param_list)
    else:
        optimizer = optim.SGD(params=param_list)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    return optimizer, scheduler


def train_position_model(h5_path, config, modeltype='full', save_path=None, device=device):

    lags = config.get('lags', None)

    X, Y = load_position_data(h5_path, modeltype=modeltype, lags=lags, device=device)
    
    n_samples = X.shape[0]
    n_train = int(0.8 * n_samples)
    
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_test, Y_test = X[n_train:], Y[n_train:]
    
    config['in_features'] = X.shape[1]
    config['Ncells'] = Y.shape[1]
    
    model = PositionGLM(config['in_features'], config['Ncells'], config, device=device)
    model.to(device)
    
    params = {'ModelID': 0, 'Nepochs': config.get('Nepochs', 1000), 'train_shifter': False}
    optimizer, scheduler = setup_model_training(model, params, config)
    
    model.train()
    print("Starting training...")
    for epoch in range(params['Nepochs']):
        optimizer.zero_grad()

        outputs = model(X_train)
        
        loss = model.loss(outputs, Y_train)

        loss.sum().backward()
        optimizer.step()
        
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss.sum())
            else:
                scheduler.step()
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{params['Nepochs']}, Loss: {loss.sum().item():.4f}")
            
    print("Training complete.")
    
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        
    return model, X_test, Y_test


def test_position_model(model, X_test, Y_test):
    """
    Test the PositionGLM model and return the loss.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        loss = model.loss(outputs, Y_test)
        mse = torch.mean((outputs - Y_test)**2).item()
        
    print(f"Test Loss: {loss.sum().item():.4f}")
    print(f"Avg MSE per cell: {mse:.4f}")
    return loss.sum().item()


if __name__ == '__main__':

    args = arg_parser()

    # h5_path = Path('/home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings/cohort02_recordings/251016_DMM_DMM061_pos18/fm1/251016_DMM_DMM061_fm_01_preproc.h5')
    base_path = '/home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings/cohort02_recordings/251021_DMM_DMM061_pos04/fm1'
    h5_path = Path(os.path.join(base_path, '251021_DMM_DMM061_fm_01_preproc.h5'))

    pos_config = {
        'activation_type': 'Identity',
        'loss_type': 'mse',
        'initW': 'normal',
        'optimizer': 'adam',
        'lr_w': 1e-2, 
        'lr_b': 1e-2,
        'L1_alpha': 1e-2,
        'Nepochs': args['Nepochs'],
        'L2_lambda': 1e-3,
        'lags': np.arange(-20,1,1), # Use past behavior (-N to 0) to predict current neural activity
        'hidden_size': 128,
        'dropout': 0.25
    }

    if h5_path.exists():
        print(f"Starting training on {h5_path}")
        model, X_test, y_test = train_position_model(h5_path, pos_config)

    else:
        print(f"File not found: {h5_path}")

    loss = test_position_model(model, X_test, y_test)

    model.eval()
    with torch.no_grad():
        y_hat = model(X_test)
    
    y_true = y_test.cpu().numpy()
    y_pred = y_hat.cpu().numpy()
    
    n_cells = y_true.shape[1]
    r2_scores = np.zeros(n_cells)
    
    for c in range(n_cells):
        ss_res = np.sum((y_true[:, c] - y_pred[:, c]) ** 2)
        ss_tot = np.sum((y_true[:, c] - np.mean(y_true[:, c])) ** 2)
        r2_scores[c] = 1 - (ss_res / (ss_tot + 1e-8))

    corrs = np.zeros(np.size(y_true,1))
    for c in range(np.size(y_true,1)):
        corrs[c] = fm2p.corrcoef(y_true[:,c], y_pred[:,c])
            
    best_cell_idx = np.argmax(r2_scores)
    print(f"Best cell index: {best_cell_idx}, R2: {r2_scores[best_cell_idx]:.4f}")
    
    plt.figure(figsize=(15, 5))
    plt.plot(y_true[:, best_cell_idx], 'k', label='True', linewidth=1)
    plt.plot(y_pred[:, best_cell_idx], 'r', label='Predicted', alpha=0.7, linewidth=1)
    plt.title(f'Best Cell (Index {best_cell_idx}) - R^2 = {r2_scores[best_cell_idx]:.3f}')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.hist(r2_scores, bins=20, edgecolor='k', alpha=0.7)
    plt.xlabel('R^2 Score')
    plt.ylabel('Count')
    plt.title('Distribution of R^2 Scores')
    plt.show()

    dict_out = {
        'y_hat': y_pred,
        'y_true': y_true,
        'r2_scores': r2_scores,
        'corrs': corrs,
        'model_weights': model.get_weights()
    }

    individual_model_keys = [
        'theta',
        'phi',
        'yaw',
        'roll',
        'pitch'
    ]

    for key in individual_model_keys:

        print('Fitting for individual model: {}'.format(key))

        model, X_test, y_test = train_position_model(h5_path, pos_config, modeltype=key)
        loss = test_position_model(model, X_test, y_test)

        model.eval()
        with torch.no_grad():
            y_hat = model(X_test)
        
        y_true = y_test.cpu().numpy()
        y_pred = y_hat.cpu().numpy()

        n_cells = y_true.shape[1]
        r2_scores = np.zeros(n_cells)
        
        for c in range(n_cells):
            ss_res = np.sum((y_true[:, c] - y_pred[:, c]) ** 2)
            ss_tot = np.sum((y_true[:, c] - np.mean(y_true[:, c])) ** 2)
            r2_scores[c] = 1 - (ss_res / (ss_tot + 1e-8))

        corrs = np.zeros(np.size(y_true,1))
        for c in range(np.size(y_true,1)):
            corrs[c] = fm2p.corrcoef(y_true[:,c], y_pred[:,c])

        dict_out['{}_r2'.format(key)] = r2_scores
        dict_out['{}_corrs'.format(key)] = corrs
        dict_out['{}_y_hat'.format(key)] = y_pred
        dict_out['{}_y_true'.format(key)] = y_true
        dict_out['{}_weights'.format(key)] = model.get_weights()

    fm2p.write_h5(os.path.join(base_path, 'pytorchGLM_predictions_nosmoothing.h5'), dict_out)