import torch
import numpy as np
from .cmap import make_parula
from .files import read_h5, write_h5
from .helper import interp_short_gaps
from .time import interpT
from .filter import convfilt
from .LNP_eval import add_scatter_col
from .imu import check_and_trim_imu_disconnect
from .correlation import corrcoef
from .paths import find
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

device = 'cuda' # torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def make_earth_tones():
    """ Create a custom categorical earth-tone colormap with 10 colors in pairs.

    The pairs are:
        1. Moss & Sage (Green)
        2. Clay & Sand (Brown)
        3. Slate & Sky (Blue-Grey)
        4. Rust & Peach (Red-Orange)
        5. Ochre & Straw (Yellow)
    """

    colors = [
        '#2ECC71', '#82E0AA', # Green
        '#FF9800', '#FFCC80', # Orange
        '#03A9F4', '#81D4FA', # Blue
        '#9C27B0', '#E1BEE7', # Purple
        '#FFEB3B', '#FFF59D'  # Yellow
    ]

    # Convert hex to RGB
    rgb_colors = [tuple(int(h.lstrip('#')[i:i+2], 16) / 255.0 for i in (0, 2, 4)) for h in colors]

    earth_map = LinearSegmentedColormap.from_list('earth_tones', rgb_colors, N=10)

    return earth_map


def get_equally_spaced_colormap_values(colormap_name, num_values):
    if not isinstance(num_values, int) or num_values <= 0:
        raise ValueError("num_values must be a positive integer.")
    if colormap_name == 'parula':
        cmap = make_parula()
    elif colormap_name == 'earth_tones':
        cmap = make_earth_tones()
    else:
        cmap = cm.get_cmap(colormap_name)
    normalized_positions = np.linspace(0, 1, num_values)
    colors = [cmap(pos) for pos in normalized_positions]
    return colors

goodred = '#D96459'


def load_position_data(
        data_input, modeltype='full',
        lags=None, use_abs=False, device=device):

    if isinstance(data_input, (str, Path)):
        data = read_h5(data_input)
    else:
        data = data_input

    feature_names = []
    theta = data.get('theta_interp')
    phi = data.get('phi_interp')

    yaw = data.get('head_yaw_deg')

    roll = data.get('roll_twop_interp')
    pitch = data.get('pitch_twop_interp')

    gyro_x = data.get('gyro_x_twop_interp')
    gyro_y = data.get('gyro_y_twop_interp')
    gyro_z = data.get('gyro_z_twop_interp')

    eyeT = data['eyeT'][data['eyeT_startInd']:data['eyeT_endInd']]
    eyeT = eyeT - eyeT[0]

    if 'dPhi' not in data.keys():
        phi_full = np.rad2deg(data['phi'][data['eyeT_startInd']:data['eyeT_endInd']])
        dPhi  = np.diff(interp_short_gaps(phi_full, 5)) / np.diff(eyeT)
        dPhi = np.roll(dPhi, -2)
        data['dPhi'] = dPhi

    if 'dTheta' not in data.keys():# and 'dEye' not in data.keys():
        theta_full = np.rad2deg(data['theta'][data['eyeT_startInd']:data['eyeT_endInd']])
        dTheta  = np.diff(interp_short_gaps(theta_full, 5)) / np.diff(eyeT)
        dTheta = np.roll(dTheta, -2)
        data['dTheta'] = dTheta

        t = eyeT.copy()[:-1]
        t1 = t + (np.diff(eyeT) / 2)
        data['eyeT1'] = t1

    # elif 'dTheta' not in data.keys():
    #     data['dTheta'] = data['dEye'].copy()

    dTheta = interp_short_gaps(data['dTheta'])
    # print(phi_full.shape, theta_full.shape, dTheta.shape)
    dTheta = interpT(dTheta, data['eyeT1'], data['twopT'])
    dPhi = interp_short_gaps(data['dPhi'])
    # print(dPhi.shape, data['eyeT1'].shape, data['twopT'].shape, eyeT.shape, dTheta.shape)
    dPhi = interpT(dPhi, data['eyeT1'], data['twopT'])

    # Calculate NaN mask on all relevant variables before any filtering/filling
    all_vars = [theta, phi, yaw, roll, pitch, dTheta, dPhi, gyro_x, gyro_y, gyro_z]
    nan_mask = np.isnan(np.stack([v[:min(len(v) for v in all_vars)] for v in all_vars], axis=1)).any(axis=1)
    
    valid_arrays = [x for x in [theta, phi, yaw, roll, pitch, ltdk, dTheta, dPhi, gyro_x, gyro_y, gyro_z] if x is not None]
    if not valid_arrays:
        raise ValueError("No valid data arrays found.")
    
    min_len = min(len(x) for x in valid_arrays)
    
    # spikes = data.get('norm_spikes')
    spikes = data.get('norm_dFF')
    if spikes is None:
        raise ValueError("norm_spikes not found in HDF5 file.")
    
    for c in range(np.size(spikes, 0)):
        spikes[c,:] = convfilt(spikes[c,:], 10)
        
    min_len = min(min_len, spikes.shape[1])
    
    if theta is not None: theta = theta[:min_len]
    if phi is not None: phi = phi[:min_len]
    if yaw is not None: yaw = yaw[:min_len]
    if roll is not None: roll = roll[:min_len]
    if pitch is not None: pitch = pitch[:min_len]
    spikes = spikes[:, :min_len]
    spikes = spikes.T
    ltdk = ltdk[:min_len]
    nan_mask = nan_mask[:min_len]
    if dTheta is not None: dTheta = dTheta[:min_len]
    if dPhi is not None: dPhi = dPhi[:min_len]
    if gyro_x is not None: gyro_x = gyro_x[:min_len]
    if gyro_y is not None: gyro_y = gyro_y[:min_len]
    if gyro_z is not None: gyro_z = gyro_z[:min_len]
    
    if modeltype == 'full':
        features = []
        names = []
        if theta is not None: features.append(theta); names.append('theta')
        if phi is not None: features.append(phi); names.append('phi')
        if yaw is not None: features.append(yaw); names.append('yaw')
        if roll is not None: features.append(roll); names.append('roll')
        if pitch is not None: features.append(pitch); names.append('pitch')
        if dTheta is not None: features.append(dTheta); names.append('dTheta')
        if dPhi is not None: features.append(dPhi); names.append('dPhi')
        if gyro_x is not None: features.append(gyro_x); names.append('gyro_x')
        if gyro_y is not None: features.append(gyro_y); names.append('gyro_y')
        if gyro_z is not None: features.append(gyro_z); names.append('gyro_z')
        
        X = np.stack(features, axis=1)
        feature_names = names
    elif modeltype == 'theta':
        X = np.stack([theta, dTheta], axis=1)
        feature_names = ['theta', 'dTheta']
    elif modeltype == 'theta_pos':
        X = theta[:, np.newaxis]
        feature_names = ['theta']
    elif modeltype == 'theta_vel':
        X = dTheta[:, np.newaxis]
        feature_names = ['dTheta']
    elif modeltype == 'phi':
        X = np.stack([phi, dPhi], axis=1)
        feature_names = ['phi', 'dPhi']
    elif modeltype == 'phi_pos':
        X = phi[:, np.newaxis]
        feature_names = ['phi']
    elif modeltype == 'phi_vel':
        X = dPhi[:, np.newaxis]
        feature_names = ['dPhi']
    elif modeltype == 'yaw':
        X = np.stack([yaw, gyro_z], axis=1)
        feature_names = ['yaw', 'gyro_z']
    elif modeltype == 'yaw_pos':
        X = yaw[:, np.newaxis]
        feature_names = ['yaw']
    elif modeltype == 'yaw_vel':
        X = gyro_z[:, np.newaxis]
        feature_names = ['gyro_z']
    elif modeltype == 'roll':
        X = np.stack([roll, gyro_x], axis=1)
        feature_names = ['roll', 'gyro_x']
    elif modeltype == 'roll_pos':
        X = roll[:, np.newaxis]
        feature_names = ['roll']
    elif modeltype == 'roll_vel':
        X = gyro_x[:, np.newaxis]
        feature_names = ['gyro_x']
    elif modeltype == 'pitch':
        X = np.stack([pitch, gyro_y], axis=1)
        feature_names = ['pitch', 'gyro_y']
    elif modeltype == 'pitch_pos':
        X = pitch[:, np.newaxis]
        feature_names = ['pitch']
    elif modeltype == 'pitch_vel':
        X = gyro_y[:, np.newaxis]
        feature_names = ['gyro_y']
    elif modeltype == 'velocity_only':
        features = []
        names = []
        if dTheta is not None: features.append(dTheta); names.append('dTheta')
        if dPhi is not None: features.append(dPhi); names.append('dPhi')
        if gyro_x is not None: features.append(gyro_x); names.append('gyro_x')
        if gyro_y is not None: features.append(gyro_y); names.append('gyro_y')
        if gyro_z is not None: features.append(gyro_z); names.append('gyro_z')
        X = np.stack(features, axis=1)
        feature_names = names
    elif modeltype == 'position_only':
        features = []
        names = []
        if theta is not None: features.append(theta); names.append('theta')
        if phi is not None: features.append(phi); names.append('phi')
        if yaw is not None: features.append(yaw); names.append('yaw')
        if roll is not None: features.append(roll); names.append('roll')
        if pitch is not None: features.append(pitch); names.append('pitch')
        X = np.stack(features, axis=1)
        feature_names = names
    elif modeltype == 'eyes_only':
        features = []
        names = []
        if theta is not None: features.append(theta); names.append('theta')
        if phi is not None: features.append(phi); names.append('phi')
        if dTheta is not None: features.append(dTheta); names.append('dTheta')
        if dPhi is not None: features.append(dPhi); names.append('dPhi')
        X = np.stack(features, axis=1)
        feature_names = names
    elif modeltype == 'head_only':
        features = []
        names = []
        if yaw is not None: features.append(yaw); names.append('yaw')
        if roll is not None: features.append(roll); names.append('roll')
        if pitch is not None: features.append(pitch); names.append('pitch')
        if gyro_x is not None: features.append(gyro_x); names.append('gyro_x')
        if gyro_y is not None: features.append(gyro_y); names.append('gyro_y')
        if gyro_z is not None: features.append(gyro_z); names.append('gyro_z')
        X = np.stack(features, axis=1)
        feature_names = names
    else:
        raise ValueError(f"Invalid modeltype: {modeltype}")
    
    X_mean = np.nanmean(X, axis=0)
    X_std = np.nanstd(X, axis=0)
    X_std[X_std == 0] = 1.0
    X = (X - X_mean) / X_std
    X[np.isnan(X)] = 0.0

    if use_abs:
        X = np.abs(X)

    if lags is not None:
        X = add_temporal_lags(X, lags)
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    if np.isnan(spikes).any():
        spikes[np.isnan(spikes)] = 0.0
    
    spikes_mean = np.nanmean(spikes, axis=0)
    spikes_std = np.nanstd(spikes, axis=0)
    spikes_std[spikes_std == 0] = 1.0
    spikes = (spikes - spikes_mean) / spikes_std

    # print(f"Target (dF/F) stats (Z-scored) -- Mean: {np.nanmean(spikes):.4f}, Std: {np.nanstd(spikes):.4f}, Max: {np.nanmax(spikes):.4f}")
    Y_tensor = torch.tensor(spikes, dtype=torch.float32).to(device)
    
    return X_tensor, Y_tensor, feature_names, torch.tensor(ltdk, device=device), torch.tensor(nan_mask, device=device), X_mean, X_std


def _detect_modalities(data):
    """Return a dict of booleans describing which behavioral signals are present."""
    has_eye = (data.get('theta_interp') is not None or data.get('phi_interp') is not None)
    has_head = (data.get('gyro_z_interp') is not None)
    return {'eye': has_eye, 'head': has_head}


def load_position_data_eyes_only(
        data_input, modeltype='eyes_only', lags=None, use_abs=False, device=device):
    """Like load_position_data but for recordings that have eye-tracking only (no head IMU).

    Supported modeltypes: 'eyes_only' / 'full', 'theta', 'theta_pos', 'theta_vel',
                          'phi', 'phi_pos', 'phi_vel'.
    """
    if isinstance(data_input, (str, Path)):
        data = read_h5(data_input)
    else:
        data = data_input

    theta = data.get('theta_interp')
    phi   = data.get('phi_interp')

    eyeT = data['eyeT'][data['eyeT_startInd']:data['eyeT_endInd']]
    eyeT = eyeT - eyeT[0]

    if 'dPhi' not in data:
        phi_full = np.rad2deg(data['phi'][data['eyeT_startInd']:data['eyeT_endInd']])
        dPhi = np.diff(interp_short_gaps(phi_full, 5)) / np.diff(eyeT)
        dPhi = np.roll(dPhi, -2)
        data['dPhi'] = dPhi

    if 'dTheta' not in data:
        theta_full = np.rad2deg(data['theta'][data['eyeT_startInd']:data['eyeT_endInd']])
        dTheta = np.diff(interp_short_gaps(theta_full, 5)) / np.diff(eyeT)
        dTheta = np.roll(dTheta, -2)
        data['dTheta'] = dTheta
        t = eyeT.copy()[:-1]
        data['eyeT1'] = t + (np.diff(eyeT) / 2)

    dTheta = interp_short_gaps(data['dTheta'])
    dTheta = interpT(dTheta, data['eyeT1'], data['twopT'])
    dPhi   = interp_short_gaps(data['dPhi'])
    dPhi   = interpT(dPhi,   data['eyeT1'], data['twopT'])

    if cond=='light':
        ltdk = data['ltdk_state_vec'].copy()
    elif cond=='dark':
        ltdk = ~data['ltdk_state_vec'].copy()

    spikes = data.get('norm_dFF')
    if spikes is None:
        raise ValueError("norm_dFF not found in data.")
    for c in range(spikes.shape[0]):
        spikes[c, :] = convfilt(spikes[c, :], 10)

    valid_arrays = [x for x in [theta, phi, dTheta, dPhi, ltdk] if x is not None]
    min_len = min(min(len(x) for x in valid_arrays), spikes.shape[1])

    if theta  is not None: theta  = theta[:min_len]
    if phi    is not None: phi    = phi[:min_len]
    dTheta = dTheta[:min_len]
    dPhi   = dPhi[:min_len]
    ltdk   = ltdk[:min_len]
    spikes = spikes[:, :min_len]
    spikes = spikes.T

    eye_vars = [v for v in [theta, phi, dTheta, dPhi] if v is not None]
    nan_mask = np.isnan(np.stack([v[:min_len] for v in eye_vars], axis=1)).any(axis=1)

    if modeltype in ('eyes_only', 'full'):
        features, names = [], []
        if theta  is not None: features.append(theta);  names.append('theta')
        if phi    is not None: features.append(phi);    names.append('phi')
        features.append(dTheta); names.append('dTheta')
        features.append(dPhi);   names.append('dPhi')
    elif modeltype == 'theta':
        features, names = [theta, dTheta], ['theta', 'dTheta']
    elif modeltype == 'theta_pos':
        features, names = [theta], ['theta']
    elif modeltype == 'theta_vel':
        features, names = [dTheta], ['dTheta']
    elif modeltype == 'phi':
        features, names = [phi, dPhi], ['phi', 'dPhi']
    elif modeltype == 'phi_pos':
        features, names = [phi], ['phi']
    elif modeltype == 'phi_vel':
        features, names = [dPhi], ['dPhi']
    else:
        raise ValueError(f"load_position_data_eyes_only: unsupported modeltype '{modeltype}'")

    X = np.stack(features, axis=1)
    feature_names = names

    X_mean = np.nanmean(X, axis=0)
    X_std  = np.nanstd(X,  axis=0)
    X_std[X_std == 0] = 1.0
    X = (X - X_mean) / X_std
    X[np.isnan(X)] = 0.0

    if use_abs:
        X = np.abs(X)
    if lags is not None:
        X = add_temporal_lags(X, lags)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    if np.isnan(spikes).any():
        spikes[np.isnan(spikes)] = 0.0
    spikes_mean = np.nanmean(spikes, axis=0)
    spikes_std  = np.nanstd(spikes,  axis=0)
    spikes_std[spikes_std == 0] = 1.0
    spikes = (spikes - spikes_mean) / spikes_std
    Y_tensor = torch.tensor(spikes, dtype=torch.float32).to(device)

    return X_tensor, Y_tensor, feature_names, torch.tensor(ltdk, device=device), torch.tensor(nan_mask, device=device), X_mean, X_std


def fit_test_ffNLE_eyes_only(data_input, save_dir=None):
    """Version of fit_test_ffNLE for recordings with eye-tracking but no head IMU signals."""

    if isinstance(data_input, (str, Path)):
        if save_dir is None:
            save_dir = os.path.split(data_input)[0]

    data = check_and_trim_imu_disconnect(data_input)
    base_path = save_dir

    pos_config = {
        'activation_type': 'Identity',
        'loss_type': 'mse',
        'initW': 'normal',
        'optimizer': 'adam',
        'lr_w': 1e-2,
        'lr_b': 1e-2,
        'L1_alpha': 1e-2,
        'Nepochs': 5000,
        'L2_lambda': 1e-3,
        'lags': np.arange(-10, 1, 1),
        'use_abs': False,
        'hidden_size': 128,
        'dropout': 0.25
    }

    dict_out = {}

    _single_var_candidates = [
        ('theta',  'theta_pos', 'theta_interp'),
        ('phi',    'phi_pos',   'phi_interp'),
        ('dTheta', 'theta_vel', None),
        ('dPhi',   'phi_vel',   None),
    ]
    model_runs = []
    for _sv_key, _sv_type, _sv_check in _single_var_candidates:
        if _sv_check is None or data.get(_sv_check) is not None:
            model_runs.append({'key': _sv_key, 'type': _sv_type, 'abs': False, 'Nepochs': 2000})

    model_runs.append({'key': 'eyes_only', 'type': 'eyes_only', 'abs': False})

    for run in model_runs:
        key   = run['key']
        mtype = run['type']
        use_abs = run['abs']

        current_config = pos_config.copy()
        current_config['use_abs'] = use_abs
        current_config['Nepochs'] = run.get('Nepochs', pos_config['Nepochs'])

        X_all, Y_all, feature_names, ltdk, nan_mask, X_feat_mean, X_feat_std = load_position_data_eyes_only(
            data, modeltype=mtype, lags=current_config.get('lags'),
            use_abs=use_abs, device=device
        )

        idx_light = get_strict_indices(ltdk, nan_mask, current_config.get('lags'), 1)
        idx_dark  = get_strict_indices(ltdk, nan_mask, current_config.get('lags'), 0)

        train_conditions = [
            {'name': 'Light', 'indices': idx_light},
            {'name': 'Dark',  'indices': idx_dark},
        ]

        for cond in train_conditions:
            cond_name   = cond['name']
            pool_indices = cond['indices']

            if len(pool_indices) < 100:
                print(f"Skipping {key} {cond_name}: too few samples ({len(pool_indices)})")
                continue

            print(f'Fitting model: {key} (type={mtype}, train={cond_name})')

            n_chunks = 10
            chunks = np.array_split(pool_indices, n_chunks)
            chunk_indices = np.arange(n_chunks)
            np.random.seed(42)
            np.random.shuffle(chunk_indices)
            split_pt  = int(0.8 * n_chunks)
            train_idx = np.sort(np.concatenate([chunks[i] for i in chunk_indices[:split_pt]]))
            val_idx   = np.sort(np.concatenate([chunks[i] for i in chunk_indices[split_pt:]]))

            model, _, _, _, train_inds, val_inds = train_position_model(
                (X_all, Y_all, feature_names, ltdk, nan_mask),
                current_config, modeltype=mtype,
                train_indices=train_idx, test_indices=val_idx, device=device
            )

            dict_out[f'{key}_train{cond_name}_weights']      = model.get_weights()
            dict_out[f'{key}_train{cond_name}_train_indices'] = train_inds
            dict_out[f'{key}_train{cond_name}_val_indices']   = val_inds
            dict_out[f'{key}_feature_names']                  = feature_names

            for test_name, test_idx in [('Light', idx_light), ('Dark', idx_dark)]:
                if len(test_idx) == 0:
                    continue

                X_test_sub = X_all[test_idx]
                Y_test_sub = Y_all[test_idx]

                model.eval()
                with torch.no_grad():
                    y_hat = model(X_test_sub)

                y_true_np = Y_test_sub.cpu().numpy()
                y_pred_np = y_hat.cpu().numpy()

                n_cells   = y_true_np.shape[1]
                r2_scores = np.zeros(n_cells)
                corrs     = np.zeros(n_cells)
                for c in range(n_cells):
                    ss_res = np.sum((y_true_np[:, c] - y_pred_np[:, c]) ** 2)
                    ss_tot = np.sum((y_true_np[:, c] - np.mean(y_true_np[:, c])) ** 2)
                    r2_scores[c] = 1 - (ss_res / (ss_tot + 1e-8))
                    corrs[c] = corrcoef(y_true_np[:, c], y_pred_np[:, c])

                prefix = f'{key}_train{cond_name}_test{test_name}'
                dict_out[f'{prefix}_r2']           = r2_scores
                dict_out[f'{prefix}_corrs']        = corrs
                dict_out[f'{prefix}_y_hat']        = y_pred_np
                dict_out[f'{prefix}_y_true']       = y_true_np
                dict_out[f'{prefix}_eval_indices'] = test_idx

                importances = compute_permutation_importance(
                    model, X_test_sub, Y_test_sub, feature_names, current_config.get('lags'), device=device
                )
                for feat, imp in importances.items():
                    dict_out[f'{prefix}_importance_{feat}'] = imp

                pdp_results = compute_pdp(
                    model, X_test_sub, feature_names, current_config.get('lags'), device=device,
                    X_mean=X_feat_mean, X_std=X_feat_std
                )
                for feat, res in pdp_results.items():
                    dict_out[f'{prefix}_pdp_{feat}_centers'] = res['centers']
                    dict_out[f'{prefix}_pdp_{feat}_curve']   = res['pdp']

    if base_path:
        h5_savepath = os.path.join(base_path, 'pytorchGLM_predictions_v09b.h5')
        write_h5(h5_savepath, dict_out)

        for cond_label, cond_key in [('Light', 'eyes_only_trainLight_testLight'),
                                      ('Dark',  'eyes_only_trainDark_testDark')]:
            if any(k.startswith(f'{cond_key}_importance_') for k in dict_out):
                corrs_sort = dict_out.get(f'{cond_key}_corrs')
                sorted_idx = np.argsort(corrs_sort)[::-1] if corrs_sort is not None else None
                pdf_path = os.path.join(base_path, f'feature_importance_v09b_{cond_label}.pdf')
                print(f'Generating {pdf_path}')
                plot_feature_importance(dict_out, model_key=cond_key, save_path=pdf_path, sorted_indices=sorted_idx)

    return dict_out


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


def train_position_model(
        data_input, config, modeltype='full', save_path=None, load_path=None,
        device=device, train_indices=None, test_indices=None):

    lags = config.get('lags', None)
    use_abs = config.get('use_abs', False)

    if isinstance(data_input, tuple):
        X, Y, feature_names, ltdk, nan_mask = data_input
    else:
        X, Y, feature_names, ltdk, nan_mask, _, _ = load_position_data(
            data_input, modeltype=modeltype, lags=lags, use_abs=use_abs, device=device)
    
    if train_indices is None or test_indices is None:
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
    
    train_idx = torch.tensor(train_indices, device=device)
    test_idx = torch.tensor(test_indices, device=device)
    
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    
    config['in_features'] = X.shape[1]
    config['Ncells'] = Y.shape[1]
    
    model = PositionGLM(config['in_features'], config['Ncells'], config, device=device)
    model.to(device)
    
    model_loaded = False
    if load_path and os.path.exists(load_path):
        print(f"Loading model from {load_path}")
        try:
            model.load_state_dict(torch.load(load_path))
            model_loaded = True
        except RuntimeError as e:
            print(f"Failed to load model (likely architecture mismatch): {e}\nTraining from scratch...")

    if not model_loaded:
        params = {'ModelID': 0, 'Nepochs': config.get('Nepochs', 1000), 'train_shifter': False}
        optimizer, scheduler = setup_model_training(model, params, config)
        
        model.train()
        # print("Starting training...")
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
            
            # if epoch % 100 == 0:
                # print(f"\rEpoch {epoch}/{params['Nepochs']}, Loss: {loss.sum().item():.4f}", end='', flush=True)
        # print('\n')
                
        # print("Training complete.")
        
        if save_path:
            torch.save(model.state_dict(), save_path)
            # print(f"Model saved to {save_path}")
        
    return model, X_test, Y_test, feature_names, train_indices, test_indices


def test_position_model(model, X_test, Y_test):
    """
    Test the PositionGLM model and return the loss.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        loss = model.loss(outputs, Y_test)
        mse = torch.mean((outputs - Y_test)**2).item()
        
    # print(f"Test Loss: {loss.sum().item():.4f}")
    # print(f"Avg MSE per cell: {mse:.4f}")
    return loss.sum().item()


def compute_permutation_importance(model, X_test, Y_test, feature_names, lags, device=device):

    model.eval()

    X_np = X_test.cpu().numpy()
    Y_np = Y_test.cpu().numpy()

    # Replace any residual NaN in the target (e.g. cells inactive in this condition)
    Y_np = np.nan_to_num(Y_np, nan=0.0)

    n_samples, n_inputs = X_np.shape
    n_lags = len(lags) if lags is not None else 1
    n_base_features = n_inputs // n_lags

    with torch.no_grad():
        y_hat = model(X_test).cpu().numpy()

    # Identify cells with zero variance in the test set — importance is undefined for them
    y_std = np.std(Y_np, axis=0)
    zero_var = y_std < 1e-6

    baseline_r2 = np.zeros(Y_np.shape[1])
    for c in range(Y_np.shape[1]):
        if zero_var[c]:
            baseline_r2[c] = 0.0
            continue
        ss_res = np.sum((Y_np[:, c] - y_hat[:, c]) ** 2)
        ss_tot = np.sum((Y_np[:, c] - np.mean(Y_np[:, c])) ** 2)
        baseline_r2[c] = 1 - (ss_res / (ss_tot + 1e-8))

    importances = {}

    for i, feat_name in enumerate(feature_names):
        X_shuff = X_np.copy()

        for l in range(n_lags):
            col_idx = i + (l * n_base_features)
            np.random.shuffle(X_shuff[:, col_idx])

        X_shuff_tensor = torch.tensor(X_shuff, dtype=torch.float32).to(device)

        with torch.no_grad():
            y_hat_shuff = model(X_shuff_tensor).cpu().numpy()

        shuff_r2 = np.zeros(Y_np.shape[1])
        for c in range(Y_np.shape[1]):
            if zero_var[c]:
                shuff_r2[c] = 0.0
                continue
            ss_res = np.sum((Y_np[:, c] - y_hat_shuff[:, c]) ** 2)
            ss_tot = np.sum((Y_np[:, c] - np.mean(Y_np[:, c])) ** 2)
            shuff_r2[c] = 1 - (ss_res / (ss_tot + 1e-8))

        importances[feat_name] = baseline_r2 - shuff_r2

    return importances


def plot_feature_importance(data, model_key=None, cell_idx=None, save_path=None, show=True, sorted_indices=None):

    if model_key is not None:

        importances = {}
        prefix = f'{model_key}_importance_'
        for k, v in data.items():
            if k.startswith(prefix):
                feat_name = k[len(prefix):]
                importances[feat_name] = v
        
        if not importances:
            print(f"No importance keys found for model '{model_key}' in data.")
            return
    else:
        importances = data

    feature_names = list(importances.keys())
    
    cmap = cm.get_cmap('tab20')
    colors = [cmap(i/len(feature_names)) for i in range(len(feature_names))]
    if hasattr(cmap, 'colors'):
        colors = [cmap.colors[i % 20] for i in range(len(feature_names))]
    else:
        colors = [cmap(i / 20) for i in range(len(feature_names))]
    colors = colors[:6] + colors[8:]
    
    if save_path and str(save_path).endswith('.pdf'):
        if model_key is None:
            print("model_key is required for PDF generation to sort by performance.")
            return

        corrs = data.get(f'{model_key}_corrs')
        if corrs is None:
            corrs = data.get(f'{model_key}_r2')
            
        if sorted_indices is None:
            if corrs is not None:
                sorted_indices = np.argsort(corrs)[::-1]
            else:
                n_cells = len(next(iter(importances.values())))
                sorted_indices = np.arange(n_cells)
            
        with PdfPages(save_path) as pdf:

            plt.figure(figsize=(5, 5), dpi=300)
            ax = plt.gca()
            for i, feat in enumerate(feature_names):
                vals = np.asarray(importances[feat]).flatten()
                add_scatter_col(ax, i, vals)#, color=colors[i])
            
            plt.ylabel('Importance (Drop in R²)', fontsize=12)
            plt.title(f'Feature Importance Population Summary ({model_key})', fontsize=14)
            plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right', fontsize=12)
            plt.axhline(0, color='black', linewidth=0.8)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            for i, c_idx in enumerate(sorted_indices):
                plot_feature_importance(importances, cell_idx=c_idx, save_path=None, show=False)
                title_str = f'Cell {c_idx} (Rank {i+1})'
                if corrs is not None:
                    title_str += f' - Corr: {corrs[c_idx]:.3f}'
                plt.suptitle(title_str, fontsize=12)
                pdf.savefig()
                plt.close()
        return

    if cell_idx is not None:

        n_cells = len(next(iter(importances.values())))
        if cell_idx >= n_cells:
            print(f"Cell index {cell_idx} out of bounds (n_cells={n_cells})")
            return

        values = [importances[feat][cell_idx] for feat in feature_names]
        
        plt.figure(figsize=(5, 4), dpi=300)
        bars = plt.bar(feature_names, values, color=colors, edgecolor='black')
        plt.ylabel('Importance (Drop in R²)', fontsize=12)
        # plt.title(f'Feature Importance for Cell {cell_idx}', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.axhline(0, color='black', linewidth=0.8)
        plt.grid(False)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}',
                     ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
                     
        plt.tight_layout()
        
    else:

        plt.figure(figsize=(5, 5), dpi=300)
        ax = plt.gca()
        for i, feat in enumerate(feature_names):
            vals = np.asarray(importances[feat]).flatten()
            add_scatter_col(ax, i, vals, color=colors[i])
            add_scatter_col(ax, i, vals, color=colors[i % len(colors)])
            
        plt.ylabel('Importance (Drop in R²)', fontsize=12)
        plt.title('Feature Importance Across All Cells', fontsize=14)
        plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right', fontsize=12)
        plt.axhline(0, color='black', linewidth=0.8)
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    elif show:
        plt.show()


def plot_feature_importance_full(data, importances, save_path=None, show=True):

    feature_names = list(importances.keys())
    
    cmap = cm.get_cmap('tab20')
    colors = [cmap(i/len(feature_names)) for i in range(len(feature_names))]
    
    corrs = data.get('corrs')
    if corrs is None:
        corrs = data.get('r2_scores')
            
    if corrs is not None:
        sorted_indices = np.argsort(corrs)[::-1]
    else:
        n_cells = len(next(iter(importances.values())))
        sorted_indices = np.arange(n_cells)
            
    if save_path and str(save_path).endswith('.pdf'):
            
        with PdfPages(save_path) as pdf:

            plt.figure(figsize=(5, 5), dpi=300)
            ax = plt.gca()
            for i, feat in enumerate(feature_names):
                vals = np.asarray(importances[feat]).flatten()
                add_scatter_col(ax, i, vals, color=colors[i])
            
            plt.ylabel('Importance (Drop in R²)', fontsize=12)
            plt.title(f'Feature Importance Population Summary (Full Model)', fontsize=14)
            plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right', fontsize=12)
            plt.axhline(0, color='black', linewidth=0.8)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            for i, c_idx in enumerate(sorted_indices):
                plot_feature_importance(importances, cell_idx=c_idx, save_path=None, show=False)
                plt.suptitle(f'Cell {c_idx} (Rank {i+1}) - Corr: {corrs[c_idx]:.3f}')
                pdf.savefig()
                plt.close()
    else:
        plot_feature_importance(importances, save_path=save_path, show=show)


def plot_feature_importance_full(data, importances, save_path=None, show=True):

    feature_names = list(importances.keys())
    
    cmap = cm.get_cmap('tab20')
    colors = [cmap(i/len(feature_names)) for i in range(len(feature_names))]
    
    corrs = data.get('corrs')
    if corrs is None:
        corrs = data.get('r2_scores')
            
    if corrs is not None:
        sorted_indices = np.argsort(corrs)[::-1]
    else:
        n_cells = len(next(iter(importances.values())))
        sorted_indices = np.arange(n_cells)
            
    if save_path and str(save_path).endswith('.pdf'):
            
        with PdfPages(save_path) as pdf:

            plt.figure(figsize=(5, 5), dpi=300)
            ax = plt.gca()
            for i, feat in enumerate(feature_names):
                vals = np.asarray(importances[feat]).flatten()
                add_scatter_col(ax, i, vals)
            
            plt.ylabel('Importance (Drop in R²)', fontsize=12)
            plt.title(f'Feature Importance Population Summary (Full Model)', fontsize=14)
            plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right', fontsize=12)
            plt.axhline(0, color='black', linewidth=0.8)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            for i, c_idx in enumerate(sorted_indices):
                plot_feature_importance(importances, cell_idx=c_idx, save_path=None, show=False)
                plt.suptitle(f'Cell {c_idx} (Rank {i+1}) - Corr: {corrs[c_idx]:.3f}')
                pdf.savefig()
                plt.close()
    else:
        plot_feature_importance(importances, save_path=save_path, show=show)


def plot_shuffled_comparison(model, X_test, Y_test, feature_names, lags, feature_to_shuffle, cell_idx, save_path=None, device=device, pdf=None):

    model.eval()
    
    X_np = X_test.cpu().numpy()
    Y_np = Y_test.cpu().numpy()
    
    with torch.no_grad():
        y_hat = model(X_test).cpu().numpy()
        
    n_lags = len(lags) if lags is not None else 1
    n_inputs = X_np.shape[1]
    n_base_features = n_inputs // n_lags
    
    if feature_to_shuffle not in feature_names:
        print(f"Feature {feature_to_shuffle} not found in {feature_names}")
        return

    feat_idx = feature_names.index(feature_to_shuffle)
    X_shuff = X_np.copy()
    
    for l in range(n_lags):
        col_idx = feat_idx + (l * n_base_features)
        np.random.shuffle(X_shuff[:, col_idx])
        
    X_shuff_tensor = torch.tensor(X_shuff, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        y_hat_shuff = model(X_shuff_tensor).cpu().numpy()
        
    plt.figure(figsize=(5, 6))
    plot_len = min(1000, Y_np.shape[0])
    t = np.arange(plot_len)
    
    plt.plot(t, Y_np[:plot_len, cell_idx], 'k', label='True Spikes', alpha=0.4, linewidth=1)
    plt.plot(t, y_hat[:plot_len, cell_idx], 'b', label='Baseline Pred', linewidth=1.5, alpha=0.8)
    plt.plot(t, y_hat_shuff[:plot_len, cell_idx], 'r--', label=f'Shuffled {feature_to_shuffle} Pred', linewidth=1.5, alpha=0.8)
    
    plt.title(f'Effect of Shuffling {feature_to_shuffle} on Cell {cell_idx}\n(Red line better than Blue = Negative Importance)')
    plt.xlabel('Time (frames)')
    plt.ylabel('Activity (z-scored)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    elif pdf:
        pdf.savefig()
        plt.close()
    else:
        plt.show()


def save_shuffled_comparison_pdf(model, X_test, Y_test, feature_names, lags, importances, corrs, save_path, device=device):
    sorted_indices = np.argsort(corrs)[::-1]
    
    with PdfPages(save_path) as pdf:

        for i, cell_idx in enumerate(sorted_indices):

            neg_feats = []
            for feat, imp in importances.items():
                if imp[cell_idx] < -0.05:
                    neg_feats.append((feat, imp[cell_idx]))
            
            neg_feats.sort(key=lambda x: x[1])
            
            for feat, imp in neg_feats:
                plot_shuffled_comparison(
                    model,
                    X_test,
                    Y_test,
                    feature_names,
                    lags,
                    feat,
                    cell_idx,
                    save_path=None,
                    device=device,
                    pdf=pdf
                )


def save_model_predictions_pdf(dict_out, save_path):
    
    if 'full_r2' in dict_out:
        r2 = dict_out['full_r2']
        sorted_indices = np.argsort(r2)[::-1]
    else:
        n_cells = dict_out['full_y_hat'].shape[1]
        sorted_indices = np.arange(n_cells)

    # all_keys = [k.replace('_y_hat', '') for k in dict_out.keys() if k.endswith('_y_hat')]
    normal_models = [
        'theta_y_hat',
        'phi_y_hat',
        'yaw_y_hat',
        'roll_y_hat',
        'pitch_y_hat',
        'dTheta_y_hat',
        'dPhi_y_hat',
        'gyro_z_y_hat',
        'gyro_x_y_hat',
        'gyro_y_y_hat'
    ]
    
    # normal_models = [k for k in all_keys if 'abs' not in k and k != 'full']
    # abs_models = [k for k in all_keys if 'abs' in k and k != 'full_abs']
    
    # has_full = 'full' in all_keys
    # has_full_abs = 'full_abs' in all_keys

    with PdfPages(save_path) as pdf:
        for cell_idx in tqdm(sorted_indices, desc="Generating Predictions PDF"):
            
            y_true = dict_out['full_y_true'][:, cell_idx]
            t = np.arange(len(y_true))
            
            fig = plt.figure(figsize=(5, 6), dpi=300)
            gs = fig.add_gridspec(3, 5)
            ax_main = fig.add_subplot(gs[0, :])
            
            ax_main.plot(t, y_true, 'k', alpha=0.5, label='True')
            # if has_full:
            y_pred_full = dict_out['full_y_hat'][:, cell_idx]
            ax_main.plot(t, y_pred_full, 'r', alpha=0.7, label='Full Model')
            r2_val = dict_out['full_r2'][cell_idx]
            ax_main.set_title(f'Cell {cell_idx} - Full Model (R2={r2_val:.3f})')
            ax_main.legend()
            
            for i, model in enumerate(normal_models):
                row = 1 + i // 4
                col = i % 4
                if row < 3:
                    ax = fig.add_subplot(gs[row, col])
                    y_pred = dict_out[f'{model}_y_hat'][:, cell_idx]
                    ax.plot(t, y_true, 'k', alpha=0.3)
                    ax.plot(t, y_pred, 'b', alpha=0.7)
                    r2_m = dict_out[f'{model}_r2'][cell_idx]
                    ax.set_title(f'{model} (R2={r2_m:.3f})', fontsize=8)
                    ax.axis('off')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            
            # fig = plt.figure(figsize=(12, 8), dpi=300)
            # gs = fig.add_gridspec(3, 4)
            # ax_main = fig.add_subplot(gs[0, :])
            
            # ax_main.plot(t, y_true, 'k', alpha=0.5, label='True')
            # if has_full_abs:
            #     y_pred_full = dict_out['full_abs_y_hat'][:, cell_idx]
            #     ax_main.plot(t, y_pred_full, 'r', alpha=0.7, label='Full Abs Model')
            #     r2_val = dict_out['full_abs_r2'][cell_idx]
            #     ax_main.set_title(f'Cell {cell_idx} - Full Abs Model (R2={r2_val:.3f})')
            # ax_main.legend()
            
            # for i, model in enumerate(abs_models):
            #     row = 1 + i // 4
            #     col = i % 4
            #     if row < 3:
            #         ax = fig.add_subplot(gs[row, col])
            #         y_pred = dict_out[f'{model}_y_hat'][:, cell_idx]
            #         ax.plot(t, y_true, 'k', alpha=0.3)
            #         ax.plot(t, y_pred, 'g', alpha=0.7)
            #         r2_m = dict_out[f'{model}_r2'][cell_idx]
            #         ax.set_title(f'{model} (R2={r2_m:.3f})', fontsize=8)
            #         ax.axis('off')
            
            # plt.tight_layout()
            # pdf.savefig(fig)
            # plt.close(fig)

def get_strict_indices(ltdk_tensor, nan_mask_tensor, lags, condition_val):
    """
    Returns indices where the timepoint AND all its lagged timepoints 
    match the condition (ltdk == condition_val) and are not NaN.
    """
    # Base validity: correct condition and not NaN
    valid_base = (ltdk_tensor == condition_val) & (~nan_mask_tensor)
    
    # Strict validity: must be valid for all lags
    # If lags = [-2, -1, 0], we need t, t+1, t+2 to be valid in the base array 
    # because X[t] is constructed from raw[t - lag].
    # Actually, add_temporal_lags shifts data. Row i of X contains raw[i - lag].
    # So for row i to be valid, raw[i - lag] must be valid for all lags.
    
    valid_strict = valid_base.clone()
    n_samples = len(valid_base)
    
    for lag in lags:
        # Check validity of the source timepoint for this lag
        # If lag is negative (past), we need raw[i - lag] (which is i + abs(lag))?
        # add_temporal_lags: shifted = np.roll(X, shift=-lag). 
        # If lag=-1, shift=1. X_lagged[i] comes from X[i-1].
        # So we need valid_base[i - lag] to be True.
        
        # Create shifted validity mask
        # If lag=-1, we need valid_base shifted by +1 (to align previous time to current row)
        # shift = -lag
        shift = -lag
        shifted_mask = torch.roll(valid_base, shifts=shift, dims=0)
        
        # Handle boundary effects of roll (zeros introduced in add_temporal_lags are effectively handled by masking)
        # But torch.roll wraps around. We must mask out wrapped parts.
        if shift > 0:
            shifted_mask[:shift] = False
        elif shift < 0:
            shifted_mask[shift:] = False
            
        valid_strict = valid_strict & shifted_mask
        
    return np.atleast_1d(torch.nonzero(valid_strict).squeeze().cpu().numpy())


def compute_pdp(model, X, feature_names, lags, device=device, n_bins=30, X_mean=None, X_std=None):
    """
    Compute Partial Dependence Plots (PDPs) for each feature.

    For each feature a grid of n_bins values is constructed from evenly-spaced
    quantiles of the observed data.  At each grid point every sample in X has
    its value of that feature replaced with the grid value; the model is run on
    the full modified dataset and the mean prediction across samples is recorded.
    This gives a tuning curve showing the model's expected output as a function
    of one feature, marginalised over the realistic joint distribution of all
    other features.

    The curve is mean-centred so it represents modulation around the baseline.

    grid_centers in the returned results are in original (un-z-scored) units
    when X_mean and X_std are provided, otherwise they are in z-scored units.

    Returns
    -------
    pdp_results : dict
        {feat_name: {'centers': ndarray (n_grid,),
                     'pdp':     ndarray (n_grid, N_cells)}}
    """
    model.eval()
    X_np = X.cpu().numpy()
    n_samples, n_inputs = X_np.shape
    n_lags = len(lags) if lags is not None else 1
    n_base_features = n_inputs // n_lags

    pdp_results = {}

    for i, feat_name in enumerate(feature_names):
        # Use only the lag-0 column (last lag) as the swept feature
        ref_col_idx = i + ((n_lags - 1) * n_base_features)
        feat_values = X_np[:, ref_col_idx]

        # Grid: evenly-spaced quantiles so all grid points span the data density
        quantiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(feat_values, quantiles)
        edges = np.unique(edges)
        if len(edges) < 2:
            continue
        grid = 0.5 * (edges[:-1] + edges[1:])

        pdp_curve = np.zeros((len(grid), model.N_cells))

        for k, v in enumerate(grid):
            # Replace the lag-0 column for all samples with this grid value
            X_mod = X_np.copy()
            X_mod[:, ref_col_idx] = v
            X_tensor = torch.tensor(X_mod, dtype=torch.float32).to(device)
            with torch.no_grad():
                pdp_curve[k] = model(X_tensor).cpu().numpy().mean(axis=0)

        # Mean-centre so the curve represents modulation around baseline
        pdp_curve -= pdp_curve.mean(axis=0, keepdims=True)

        # Convert grid to original units if stats are available
        centers = grid.copy()
        if X_mean is not None and X_std is not None:
            centers = centers * X_std[i] + X_mean[i]

        pdp_results[feat_name] = {
            'centers': centers,
            'pdp':     pdp_curve,
        }

    return pdp_results


def fit_test_ffNLE(data_input, save_dir=None):

    if isinstance(data_input, (str, Path)):
        if save_dir is None:
            save_dir = os.path.split(data_input)[0]

    if save_dir is None and not isinstance(data_input, (str, Path)):
        print("Warning: save_dir is None. Results will not be saved to disk.")

    data = check_and_trim_imu_disconnect(data_input)

    modalities = _detect_modalities(data)
    if not modalities['head']:
        print('  -> No head signals detected — using eyes-only model pipeline.')
        return fit_test_ffNLE_eyes_only(data, save_dir=save_dir)

    base_path = save_dir

    pos_config = {
        'activation_type': 'Identity',
        'loss_type': 'mse',
        'initW': 'normal',
        'optimizer': 'adam',
        'lr_w': 1e-2, 
        'lr_b': 1e-2,
        'L1_alpha': 1e-2,
        'Nepochs': 5000,
        'L2_lambda': 1e-3,
        'lags': np.arange(-10,1,1), # was -10 (1.33 sec)
        # 'lags': np.arange(0,1,1),
        'use_abs': False,
        'hidden_size': 128,
        'dropout': 0.25
    }

    print(f"Fitting full model")
    if base_path:
        full_model_path = os.path.join(base_path, 'full_model.pth')
    else:
        full_model_path = None
    model, X_test, y_test, feature_names, full_train_inds, full_test_inds = train_position_model(data, pos_config, save_path=full_model_path, load_path=full_model_path)

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
        corrs[c] = corrcoef(y_true[:,c], y_pred[:,c])
            
    best_cell_idx = np.argmax(r2_scores)
    # print(f"Best cell index: {best_cell_idx}, R2: {r2_scores[best_cell_idx]:.4f}")
    
    # plt.figure(figsize=(15, 5))
    # plt.plot(y_true[:, best_cell_idx], 'k', label='True', linewidth=1)
    # plt.plot(y_pred[:, best_cell_idx], 'r', label='Predicted', alpha=0.7, linewidth=1)
    # plt.title(f'Best Cell (Index {best_cell_idx}) - R^2 = {r2_scores[best_cell_idx]:.3f}')
    # plt.legend()
    # plt.show()
    
    # plt.figure(figsize=(8, 6))
    # plt.hist(r2_scores, bins=20, edgecolor='k', alpha=0.7)
    # plt.xlabel('R^2 Score')
    # plt.ylabel('Count')
    # plt.title('Distribution of R^2 Scores')
    # plt.show()

    dict_out = {
        'full_r2': r2_scores,
        'full_corrs': corrs,
        'full_y_hat': y_pred,
        'full_y_true': y_true,
        'full_weights': model.get_weights(),
        'full_train_indices': full_train_inds,
        'full_test_indices': full_test_inds,
        'full_feature_names': feature_names,
        'lags': pos_config.get('lags')
    }

    importances = compute_permutation_importance(model, X_test, y_test, feature_names, pos_config.get('lags'))
    for feat, imp in importances.items():
        dict_out[f'full_importance_{feat}'] = imp
        
    # fi_pdf_path = os.path.join(base_path, 'full_feature_importance.pdf')
    # plot_feature_importance_full(dict_out, importances, save_path=fi_pdf_path)
    
    # sc_pdf_path = os.path.join(base_path, 'full_shuffled_comparison.pdf')
    # save_shuffled_comparison_pdf(model, X_test, y_test, feature_names, pos_config.get('lags'), importances, corrs, sc_pdf_path, device=device)

    # existing_data = fm2p.read_h5(os.path.join(base_path, 'pytorchGLM_predictions_v03_add_shuff.h5'))
    # newdata = {**existing_data, **dict_out}
    # fm2p.write_h5(
    #     os.path.join(base_path, 'pytorchGLM_predictions_v03A_add_full_FI.h5'),
    #     newdata
    # )
    model_runs = []

    # Single-variable models — one per behavioral variable.
    # These are used to compute per-variable signal correlations in topography.py.
    # Each model receives only one variable (+ its temporal lags) as input.
    # check_key: the data dict key that must be non-None for the run to be added.
    _single_var_candidates = [
        ('theta',  'theta_pos',  'theta_interp'),
        ('phi',    'phi_pos',    'phi_interp'),
        ('dTheta', 'theta_vel',  None),   # dTheta computed from theta; always available
        ('dPhi',   'phi_vel',    None),   # dPhi computed from phi; always available
        ('yaw',    'yaw_pos',    'head_yaw_deg'),
        ('roll',   'roll_pos',   'roll_twop_interp'),
        ('pitch',  'pitch_pos',  'pitch_twop_interp'),
        ('gyro_x', 'roll_vel',   'gyro_x_twop_interp'),
        ('gyro_y', 'pitch_vel',  'gyro_y_twop_interp'),
        ('gyro_z', 'yaw_vel',    'gyro_z_twop_interp'),
    ]
    for _sv_key, _sv_type, _sv_check in _single_var_candidates:
        if _sv_check is None or data.get(_sv_check) is not None:
            model_runs.append({'key': _sv_key, 'type': _sv_type, 'abs': False, 'Nepochs': 2000})

    model_runs.append({'key': 'velocity_only', 'type': 'velocity_only', 'abs': False})
    model_runs.append({'key': 'position_only', 'type': 'position_only', 'abs': False})
    model_runs.append({'key': 'eyes_only', 'type': 'eyes_only', 'abs': False})
    model_runs.append({'key': 'head_only', 'type': 'head_only', 'abs': False})
    model_runs.append({'key': 'full', 'type': 'full', 'abs': False})

    for run in model_runs:
        key = run['key']
        mtype = run['type']
        use_abs = run['abs']

        current_config = pos_config.copy()
        current_config['use_abs'] = use_abs
        current_config['Nepochs'] = run.get('Nepochs', pos_config['Nepochs'])
        
        # Load data once per model type to get correct features
        X_all, Y_all, feature_names, ltdk, nan_mask, X_feat_mean, X_feat_std = load_position_data(
            data, modeltype=mtype, lags=current_config.get('lags'),
            use_abs=use_abs, device=device
        )
        
        # Define strict indices for Light and Dark
        idx_light = get_strict_indices(ltdk, nan_mask, current_config.get('lags'), 1)
        idx_dark = get_strict_indices(ltdk, nan_mask, current_config.get('lags'), 0)
        
        # Training conditions
        train_conditions = [
            {'name': 'Light', 'indices': idx_light},
            {'name': 'Dark', 'indices': idx_dark}
        ]
        
        for cond in train_conditions:
            cond_name = cond['name']
            pool_indices = cond['indices']
            
            if len(pool_indices) < 100:
                print(f"Skipping {key} {cond_name} training: too few samples ({len(pool_indices)})")
                continue
                
            print(f'Fitting model: {key} (type={mtype}, train={cond_name})')
            
            # Split pool into train/val (80/20)
            n_pool = len(pool_indices)
            n_chunks = 10
            chunks = np.array_split(pool_indices, n_chunks)
            chunk_indices = np.arange(n_chunks)
            np.random.seed(42)
            np.random.shuffle(chunk_indices)
            
            split_pt = int(0.8 * n_chunks)
            train_idx = np.sort(np.concatenate([chunks[i] for i in chunk_indices[:split_pt]]))
            val_idx = np.sort(np.concatenate([chunks[i] for i in chunk_indices[split_pt:]]))
            
            # Train model
            model, _, _, _, train_inds, val_inds = train_position_model(
                (X_all, Y_all, feature_names, ltdk, nan_mask), 
                current_config, modeltype=mtype, 
                train_indices=train_idx, test_indices=val_idx, device=device
            )
            
            dict_out[f'{key}_train{cond_name}_weights'] = model.get_weights()
            dict_out[f'{key}_train{cond_name}_train_indices'] = train_inds
            dict_out[f'{key}_train{cond_name}_val_indices'] = val_inds
            dict_out[f'{key}_feature_names'] = feature_names
            
            # Test on both Light and Dark (full valid sets)
            test_sets = [('Light', idx_light), ('Dark', idx_dark)]
            
            for test_name, test_idx in test_sets:
                if len(test_idx) == 0: continue
                
                X_test_sub = X_all[test_idx]
                Y_test_sub = Y_all[test_idx]
                
                model.eval()
                with torch.no_grad():
                    y_hat = model(X_test_sub)
                
                y_true_np = Y_test_sub.cpu().numpy()
                y_pred_np = y_hat.cpu().numpy()
                
                n_cells = y_true_np.shape[1]
                r2_scores = np.zeros(n_cells)
                corrs = np.zeros(n_cells)
                
                for c in range(n_cells):
                    ss_res = np.sum((y_true_np[:, c] - y_pred_np[:, c]) ** 2)
                    ss_tot = np.sum((y_true_np[:, c] - np.mean(y_true_np[:, c])) ** 2)
                    r2_scores[c] = 1 - (ss_res / (ss_tot + 1e-8))
                    corrs[c] = corrcoef(y_true_np[:,c], y_pred_np[:,c])
                
                prefix = f'{key}_train{cond_name}_test{test_name}'
                dict_out[f'{prefix}_r2'] = r2_scores
                dict_out[f'{prefix}_corrs'] = corrs
                dict_out[f'{prefix}_y_hat'] = y_pred_np
                dict_out[f'{prefix}_y_true'] = y_true_np
                dict_out[f'{prefix}_eval_indices'] = test_idx
                
                # Compute importance for all conditions
                importances = compute_permutation_importance(
                    model, X_test_sub, Y_test_sub, feature_names, current_config.get('lags'), device=device
                )
                for feat, imp in importances.items():
                    dict_out[f'{prefix}_importance_{feat}'] = imp

                # Compute PDP tuning curves
                pdp_results = compute_pdp(
                    model, X_test_sub, feature_names, current_config.get('lags'), device=device,
                    X_mean=X_feat_mean, X_std=X_feat_std
                )
                for feat, res in pdp_results.items():
                    dict_out[f'{prefix}_pdp_{feat}_centers'] = res['centers']
                    dict_out[f'{prefix}_pdp_{feat}_curve']   = res['pdp']
    
    if base_path:
        h5_savepath = os.path.join(base_path, 'pytorchGLM_predictions_v09b.h5')
        # print('Writing to {}'.format(h5_savepath))
        write_h5(h5_savepath, dict_out)

        # Generate Feature Importance PDFs
        light_key = 'full_trainLight_testLight'
        dark_key = 'full_trainDark_testDark'
        
        light_indices = None
        
        # Plot Light
        if any(k.startswith(f'{light_key}_importance_') for k in dict_out.keys()):
            # Determine sort order
            corrs = dict_out.get(f'{light_key}_corrs')
            if corrs is None:
                corrs = dict_out.get(f'{light_key}_r2')
            
            if corrs is not None:
                light_indices = np.argsort(corrs)[::-1]
            else:
                # Fallback
                for k in dict_out:
                    if k.startswith(f'{light_key}_importance_'):
                        n_cells = len(dict_out[k])
                        light_indices = np.arange(n_cells)
                        break
            
            if light_indices is not None:
                pdf_path = os.path.join(base_path, 'feature_importance_v09b_Light.pdf')
                print(f"Generating {pdf_path}")
                plot_feature_importance(dict_out, model_key=light_key, save_path=pdf_path, sorted_indices=light_indices)

        # Plot Dark
        if any(k.startswith(f'{dark_key}_importance_') for k in dict_out.keys()):
            pdf_path = os.path.join(base_path, 'feature_importance_v09b_Dark.pdf')
            print(f"Generating {pdf_path}")
            plot_feature_importance(dict_out, model_key=dark_key, save_path=pdf_path, sorted_indices=light_indices)
            
    return dict_out


AREA_IDS = {'V1': 5, 'RL': 2, 'AM': 3, 'PM': 4, 'AL': 7, 'LM': 8, 'P': 9, 'A': 10}


# Earth-tone colors (from ffNLE_figs.make_earth_tones) — one per variable in preferred order
_GOODRED = '#D96459'
_EARTH_HEX = [
    '#2ECC71', '#82E0AA',  # Green pair
    '#FF9800', '#FFCC80',  # Orange pair
    '#03A9F4', '#81D4FA',  # Blue pair
    '#9C27B0', '#E1BEE7',  # Purple pair
    '#FFEB3B', '#FFF59D',  # Yellow pair
]
_EARTH_COLORS = [tuple(int(h.lstrip('#')[i:i+2], 16) / 255.0 for i in (0, 2, 4)) for h in _EARTH_HEX]

# Canonical display order for variables
_VAR_ORDER = ['theta', 'dTheta', 'phi', 'dPhi', 'pitch', 'gyro_y', 'roll', 'gyro_x', 'yaw', 'gyro_z']


def _ordered_vars(all_vars):
    """Return vars sorted by _VAR_ORDER, with any extras appended."""
    ordered = [v for v in _VAR_ORDER if v in all_vars]
    extras  = [v for v in sorted(all_vars) if v not in _VAR_ORDER]
    return ordered + extras


def _var_color(var):
    """Return the earth-tone color assigned to a variable (consistent across panels)."""
    idx = _VAR_ORDER.index(var) if var in _VAR_ORDER else (hash(var) % len(_EARTH_COLORS))
    return _EARTH_COLORS[idx % len(_EARTH_COLORS)]


def _plot_cell_summary(animal, pos, cell_idx, rdata, model, save_path):
    """Generate a per-cell summary SVG: prediction trace, importances (full/dark/light), 1D tuning."""
    import matplotlib.gridspec as gridspec

    prefix_imp = 'full_importance_'
    vars_imp = set(k[len(prefix_imp):] for k in model.keys() if k.startswith(prefix_imp))
    vars_1d = set(k[:-len('_1dtuning')] for k in rdata.keys() if k.endswith('_1dtuning'))
    all_vars_set = vars_imp | vars_1d
    all_vars = _ordered_vars(all_vars_set)

    if not all_vars:
        print(f"No variables for {animal} {pos} cell {cell_idx}, skipping.")
        return

    n_vars = len(all_vars)
    bar_colors = [_var_color(v) for v in all_vars]

    corr = float(model['full_corrs'][cell_idx]) if 'full_corrs' in model else float('nan')
    r2   = float(np.asarray(model['full_r2'])[cell_idx]) if 'full_r2' in model else float('nan')

    fig = plt.figure(figsize=(max(5, 5), 18), dpi=150)
    gs = gridspec.GridSpec(5, n_vars, figure=fig, hspace=0.6, wspace=0.45)
    fig.suptitle(f'{animal}  {pos}  cell={cell_idx}  r={corr:.3f}  R²={r2:.3f}', fontsize=11)

    # --- Row 0: Predicted vs actual trace ---
    ax_pred = fig.add_subplot(gs[0, :])
    y_true_arr = model.get('full_y_true')
    y_hat_arr  = model.get('full_y_hat')
    if y_true_arr is not None and y_hat_arr is not None:
        y_true_arr = np.asarray(y_true_arr)
        y_hat_arr  = np.asarray(y_hat_arr)
        y_true_cell = y_true_arr[:, cell_idx] if y_true_arr.ndim == 2 else y_true_arr
        y_hat_cell  = y_hat_arr[:, cell_idx]  if y_hat_arr.ndim == 2  else y_hat_arr
        frames = np.arange(len(y_true_cell))
        ax_pred.plot(frames, y_true_cell, color='k',      lw=0.8, label='$y$')
        ax_pred.plot(frames, y_hat_cell,  color=_GOODRED, lw=0.8, label='$\\hat{y}$', alpha=0.85)
        ax_pred.legend(fontsize=7, loc='upper right')
        ax_pred.set_ylabel('Activity', fontsize=8)
    else:
        ax_pred.text(0.5, 0.5, 'y_true / y_hat not in model dict',
                     ha='center', va='center', transform=ax_pred.transAxes, fontsize=9)
    ax_pred.set_title('Predicted vs Actual', fontsize=9)
    ax_pred.tick_params(labelsize=6)

    def _draw_importance_row(ax, imp_prefix, title_label):
        names, vals = [], []
        colors_row = []
        for v in all_vars:
            k = f'{imp_prefix}_importance_{v}'
            if k in model:
                arr = np.asarray(model[k])
                val = float(arr[cell_idx]) if arr.ndim > 0 else float(arr)
                names.append(v)
                vals.append(max(val, 0.0))  # clip to 0 — only show positive importance
                colors_row.append(_var_color(v))
        if names:
            ax.bar(range(len(names)), vals, color=colors_row, edgecolor='k', lw=0.5)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
            ax.set_ylim(bottom=0)
        corr_key = f'{imp_prefix}_corrs'
        r2_key   = f'{imp_prefix}_r2'
        cond_r2  = float(np.asarray(model[r2_key])[cell_idx])   if r2_key   in model else float('nan')
        ax.set_ylabel('Importance (Drop R²)', fontsize=8)
        ax.set_title(f'{title_label}  (R²={cond_r2:.3f})', fontsize=9)
        ax.tick_params(labelsize=6)

    # --- Row 1: Full model importance ---
    ax_full = fig.add_subplot(gs[1, :])
    _draw_importance_row(ax_full, 'full', 'Feature Importance — Full model (all frames)')

    # --- Row 2: Dark importance ---
    ax_dark = fig.add_subplot(gs[2, :])
    _draw_importance_row(ax_dark, 'full_trainDark_testDark', 'Feature Importance — Dark')

    # --- Row 3: Light importance ---
    ax_light = fig.add_subplot(gs[3, :])
    _draw_importance_row(ax_light, 'full_trainLight_testLight', 'Feature Importance — Light')

    # --- Row 4: 1D tuning curves ---
    for col_i, v in enumerate(all_vars):
        ax = fig.add_subplot(gs[4, col_i])
        tc_key   = f'{v}_1dtuning'
        bins_key = f'{v}_1dbins'
        title_parts = [v]
        if tc_key in rdata and bins_key in rdata:
            tc   = np.asarray(rdata[tc_key])
            bins = np.asarray(rdata[bins_key])
            tc_cell = tc[cell_idx] if tc.ndim == 3 else tc  # (n_bins, 2)
            if tc_cell.ndim == 2 and tc_cell.shape[1] >= 2:
                ax.plot(bins, tc_cell[:, 0], color='steelblue', lw=1.5, label='dark')
                ax.plot(bins, tc_cell[:, 1], color='tomato',    lw=1.5, label='light')
            else:
                ax.plot(bins, tc_cell.ravel(), lw=1.5)
        for cond_chr, cond_key in [('d', 'd'), ('l', 'l')]:
            isrel_k = f'{v}_{cond_key}_isrel'
            mod_k   = f'{v}_{cond_key}_mod'
            if isrel_k in rdata:
                arr   = np.asarray(rdata[isrel_k])
                isrel = bool(arr[cell_idx]) if arr.ndim > 0 else bool(arr)
                mod_str = ''
                if mod_k in rdata:
                    marr    = np.asarray(rdata[mod_k])
                    mod_val = float(marr[cell_idx]) if marr.ndim > 0 else float(marr)
                    mod_str = f'={mod_val:.2f}'
                title_parts.append((f'{cond_chr}:rel{mod_str}' if isrel else f'{cond_chr}:nr'))
        ax.set_title('\n'.join([title_parts[0], '  '.join(title_parts[1:])]) if len(title_parts) > 1 else v,
                     fontsize=7)
        ax.tick_params(labelsize=6)
        if col_i == 0:
            ax.set_ylabel('Rate', fontsize=7)

    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close(fig)


def run_analysis_from_topography(
    pooled_path,
    visual_area='V1',
    save_dir=None,
    imu_only=False,
    n_cells_sample=30,
    rng_seed=0,
):
    """Randomly sample cells from a visual area across all recordings and save per-cell SVG summaries.

    Parameters
    ----------
    pooled_path : str
        Path to the pooled HDF5 file (from make_pooled_dataset / merge_animal_essentials).
    visual_area : str
        Target visual area, e.g. 'V1', 'AM', 'PM', 'RL', 'AL', 'LM', 'P'.
    save_dir : str, optional
        Directory to save SVG files (default: next to pooled_path).
    imu_only : bool
        If True, only include recordings that contain IMU (gyro) data.
    n_cells_sample : int
        Number of cells to sample (default 15).
    rng_seed : int
        Random seed for reproducible sampling.
    """
    if visual_area not in AREA_IDS:
        raise ValueError(f"Unknown area '{visual_area}'. Valid: {sorted(AREA_IDS)}")
    area_id = AREA_IDS[visual_area]

    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(pooled_path), f'ffNLE_summary_{visual_area}')
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading pooled data from {pooled_path}")
    pooled_data = read_h5(pooled_path)

    # Collect candidate cells: in target area, R²>0.2, both dark+light importance keys present
    # cell_list entries: (r2, animal, pos, local_cell_idx)
    cell_list = []

    for animal, animal_data in pooled_data.items():
        if not isinstance(animal_data, dict) or 'messentials' not in animal_data:
            continue
        for pos, pos_data in animal_data['messentials'].items():
            if not isinstance(pos_data, dict):
                continue
            rdata = pos_data.get('rdata', {})
            model = pos_data.get('model', {})
            visual_area_id = pos_data.get('visual_area_id')

            if visual_area_id is None or not isinstance(model, dict) or 'full_r2' not in model:
                continue
            if imu_only and not any('gyro' in k for k in rdata.keys()):
                continue

            # Both dark and light condition importance must exist (any one variable suffices to check)
            has_dark  = any(k.startswith('full_trainDark_testDark_importance_')  for k in model)
            has_light = any(k.startswith('full_trainLight_testLight_importance_') for k in model)
            if not (has_dark and has_light):
                continue

            area_arr = np.asarray(visual_area_id)
            r2_arr   = np.asarray(model['full_r2'])
            n_model  = r2_arr.shape[0] if r2_arr.ndim > 0 else 1
            cell_mask = (area_arr == area_id)
            if not np.any(cell_mask):
                continue

            for idx in np.where(cell_mask)[0]:
                if idx >= n_model:
                    continue  # cell not covered by model (area_id array larger than model output)
                r2_val = float(r2_arr[idx])
                if r2_val > 0.3:
                    cell_list.append((r2_val, animal, pos, int(idx)))

    print(f"Found {len(cell_list)} {visual_area} cells with R²>0.2 and both conditions "
          f"{'(IMU only)' if imu_only else ''}")
    if len(cell_list) == 0:
        print("No cells found — nothing to plot.")
        return

    # Sort by R^2 descending, take top n_cells_sample
    cell_list.sort(key=lambda x: x[0], reverse=True)
    sampled_full = cell_list[:n_cells_sample]
    # Shuffle the selected cells for variety in output order
    rng = np.random.default_rng(rng_seed)
    rng.shuffle(sampled_full)
    sampled = [(animal, pos, idx) for _, animal, pos, idx in sampled_full]
    sample_size = len(sampled)

    for sample_num, (animal, pos, cell_idx) in enumerate(sampled, start=1):
        pos_data = pooled_data[animal]['messentials'][pos]
        rdata = pos_data.get('rdata', {})
        model = pos_data.get('model', {})
        svg_path = os.path.join(save_dir, f'{animal}_{pos}_cell{cell_idx:04d}.svg')
        print(f"  [{sample_num}/{sample_size}] {animal} {pos} cell {cell_idx} -> {svg_path}")
        try:
            _plot_cell_summary(
                animal=animal,
                pos=pos,
                cell_idx=cell_idx,
                rdata=rdata,
                model=model,
                save_path=svg_path,
            )
        except Exception as e:
            print(f"    Error: {e}")

    print(f"Saved {sample_size} SVG figures to {save_dir}")


def ffNLE():

    parser = argparse.ArgumentParser()
    parser.add_argument('--pooled', type=str, default=None, help='Path to pooled HDF5 file')
    parser.add_argument('--area', type=str, default='all', help='Visual area to sample (default: V1)')
    parser.add_argument('--imu_only', action='store_true', help='Only use recordings with IMU data')
    parser.add_argument('--rec', type=str, default=None, help='Path to single recording HDF5')
    parser.add_argument('--cohort_dir', type=str,
                        default='/home/dylan/Storage/freely_moving_data/_V1PPC', help='Path to cohort directory')
    args = parser.parse_args()

    if args.pooled and args.area == 'all':

        for area in ['V1', 'AM', 'PM', 'RL', 'A']:
            run_analysis_from_topography(args.pooled, visual_area=area, imu_only=args.imu_only)
    if args.pooled:
        run_analysis_from_topography(args.pooled, visual_area=args.area, imu_only=args.imu_only)
    elif args.rec:
        fit_test_ffNLE(args.rec)
    else:
        # BATCH PROCESS
        # cohort_dir = '/home/dylan/Storage/freely_moving_data/_V1PPC/cohort01_recordings/'
        # cohort_dir = '/home/dylan/Storage/freely_moving_data/_V1PPC'
        recordings = find(
            '*fm*_preproc.h5',
            args.cohort_dir
        )
        recordings = sorted(recordings)
        print('Found {} recordings.'.format(len(recordings)))

        # recordings = recordings[30:]

        for ri, rec in enumerate(recordings):
            print('Fitting models for recordings {} of {} ({}).'.format(ri+1, len(recordings), rec))
            try:
                fit_test_ffNLE(rec)
            except Exception as e:
                print(f"Error processing {rec}: {e}")


if __name__ == '__main__':

    ffNLE()