import torch
import numpy as np
import fm2p
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseModel(nn.Module):
    def __init__(self, 
                    in_features, 
                    N_cells,
                    config,
                    ):
        super(BaseModel, self).__init__()
        r''' Base GLM Network
        Args: 
            in_feature: size of the input dimension
            N_cells: the number of cells to fit
            config: network configuration file with hyperparameters
        '''
        self.config = config
        self.in_features = in_features
        self.N_cells = N_cells
        self.activation_type = config['activation_type']
        self.loss_type = config.get('loss_type', 'mse')
        
        # Check for hidden size to create an MLP instead of a linear GLM
        self.hidden_size = config.get('hidden_size', 0)
        self.dropout_p = config.get('dropout', 0.0)

        if self.hidden_size > 0:
            layers = [
                nn.Linear(self.in_features, self.hidden_size),
                nn.ReLU()
            ]
            if self.dropout_p > 0:
                layers.append(nn.Dropout(p=self.dropout_p))
            layers.append(nn.Linear(self.hidden_size, self.N_cells))
            self.Cell_NN = nn.Sequential(*layers)
        else:
            self.Cell_NN = nn.Sequential(nn.Linear(self.in_features, self.N_cells,bias=True))

        # Beta=5 makes Softplus sharper (closer to ReLU), helping predict 0 and rapid peaks
        self.activations = nn.ModuleDict({'SoftPlus':nn.Softplus(beta=5),
                                          'ReLU': nn.ReLU(),})
        
        # Initialize weights for all linear layers
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

        # Initialize Regularization parameters
        self.L1_alpha = config['L1_alpha']
        if self.L1_alpha != None:
            self.register_buffer('alpha',config['L1_alpha']*torch.ones(1))

      
    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight,a=-1e-6,b=1e-6)
            m.bias.data.fill_(1e-6)
        
    def forward(self, inputs, pos_inputs=None):
        output = self.Cell_NN(inputs)
        if self.activation_type is not None:
            ret = self.activations[self.activation_type](output)
        else:
            ret = self.activations(output)
        return ret

    def loss(self,Yhat, Y): 
        """ Loss function for network with L1 regularization

        Args:
            Yhat (torch.Tensor): Prediction of target
            Y (torch.Tensor): Ground Truth of target

        Returns:
            loss_vec: Loss vector
        """
        if self.loss_type == 'poisson':
            loss_vec = torch.mean(Yhat - Y * torch.log(Yhat + 1e-8), axis=0)
        else:
            loss_vec = torch.mean((Yhat-Y)**2,axis=0)

        if self.L1_alpha != None:
            l1_reg0 = torch.stack([torch.linalg.vector_norm(NN_params,ord=1) for name, NN_params in self.Cell_NN.named_parameters() if '0.weight' in name])
            loss_vec = loss_vec + self.alpha*(l1_reg0)
        return loss_vec


class PositionGLM(BaseModel):
    def __init__(self, 
                    in_features, 
                    N_cells, 
                    config,
                    device=device):
        super(PositionGLM, self).__init__(in_features, N_cells, config)
        r''' Position GLM Network
        Args: 
            in_feature: size of the input dimension
            N_cells: the number of cells to fit
            config: network configuration file with hyperparameters
        '''
        # Use L1_alpha_m for position regularization if available, otherwise fallback to L1_alpha
        # This allows distinct regularization for position models vs visual models in the same config
        self.L1_alpha = config.get('L1_alpha_m')
        if self.L1_alpha is None:
             self.L1_alpha = config.get('L1_alpha')

        if self.L1_alpha is not None:
            self.register_buffer('alpha', self.L1_alpha * torch.ones(1))

    def forward(self, inputs, pos_inputs=None):
        # PositionGLM expects inputs to be the position features.
        # We ignore pos_inputs argument if passed, as the main inputs are position.
        return super(PositionGLM, self).forward(inputs)
    

def str_to_bool(value):
    """ Parse strings to read argparse flag entries in as bool.
    
    Parameters:
    value (str): input value
    
    Returns:
    bool
    """
    if isinstance(value, bool):
        return value
    if value.lower() in {'False', 'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'True', 'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def arg_parser(jupyter=False):
    parser = argparse.ArgumentParser(description=__doc__)
    ##### Directory Parameters #####
    parser.add_argument('--date_ani',           type=str, default='25') 
    parser.add_argument('--base_dir',           type=str, default='~/Research/SensoryMotorPred_Data/Testing')
    parser.add_argument('--fig_dir',            type=str, default='~/Research/SensoryMotorPred_Data/FigTesting')
    parser.add_argument('--data_dir',           type=str, default='~/Goeppert/nlab-nas/Dylan/freely_moving_ephys/ephys_recordings/')
    ##### Simulation Parameters ##### 
    parser.add_argument('--model_dt',           type=float,       default=1/7.50)
    parser.add_argument('--ds_vid',             type=int,         default=4)
    parser.add_argument('--Kfold',              type=int,         default=0)
    parser.add_argument('--ModRun',             type=str,         default='-1')
    parser.add_argument('--Nepochs',            type=int,         default=5000)
    parser.add_argument('--num_samples',        type=int,         default=25, help='number of samples for param search')
    parser.add_argument('--cpus_per_task',      type=float,       default=6)
    parser.add_argument('--gpus_per_task',      type=float,       default=.5)
    parser.add_argument('--load_ray',           type=str_to_bool, default=True)
    ##### Model Paremeters #####    
    parser.add_argument('--do_norm',            type=str_to_bool, default=True)
    parser.add_argument('--crop_input',         type=str_to_bool, default=True)
    parser.add_argument('--free_move',          type=str_to_bool, default=True)
    parser.add_argument('--thresh_cells',       type=str_to_bool, default=True)
    parser.add_argument('--fm_dark',            type=str_to_bool, default=False)
    parser.add_argument('--NoL1',               type=str_to_bool, default=False)
    parser.add_argument('--NoL2',               type=str_to_bool, default=False)
    parser.add_argument('--NoShifter',          type=str_to_bool, default=False)
    parser.add_argument('--do_shuffle',         type=str_to_bool, default=False)
    parser.add_argument('--use_spdpup',         type=str_to_bool, default=False)
    parser.add_argument('--only_spdpup',        type=str_to_bool, default=False)
    parser.add_argument('--train_shifter',      type=str_to_bool, default=False)
    parser.add_argument('--shifter_5050',       type=str_to_bool, default=False)
    parser.add_argument('--shifter_5050_run',   type=str_to_bool, default=False)
    parser.add_argument('--EyeHead_only',       type=str_to_bool, default=False)
    parser.add_argument('--EyeHead_only_run',   type=str_to_bool, default=False)
    parser.add_argument('--SimRF',              type=str_to_bool, default=False)

    if jupyter:
        args = parser.parse_args('')
    else:
        args = parser.parse_args()
    return vars(args)

def load_params(args,ModelID,file_dict=None,exp_dir_name=None,nKfold=0,debug=False):
    """ Set up params dictionary for loading data and model info.

    Args:
        args (dict): arguments from argparse
        ModelID (int): Model Idenfiter
        file_dict (dict): Dictionary with raw data files paths. Defaults to None and constructs
                        the dictionary assuming Niell Lab naming convension. 
        exp_dir_name (str, optional): Optional experiment directory name if using own data. Defaults to None.
        nKfold (int, optional): Kfold number for versioning. Defaults to 0.
        debug (bool, optional): debug=True does not create experiment directories. Defaults to False.

    Returns:
        params (dict): dictionary of parameters
        file_dict (dict): Dictionary with raw data files paths.
        exp (obj): Test_tube object for organizing files and tensorboard
    """
    # from test_tube import Experiment
    # ##### Check Stimulus Condition #####
    free_move = args['free_move']
    # if free_move:
    #     if args['fm_dark']:
    #         fm_dir = 'fm1_dark'
    #     else:
    #         fm_dir = 'fm1'
    #     stim_cond = fm_dir
    # else:
    #     fm_dir = 'fm1'
    #     stim_cond = 'hf1_wn' 
    stim_cond = 'fm1'

    # ##### Create directories and paths #####
    date_ani = args['date_ani']
    date_ani2 = '_'.join(date_ani.split('/'))
    data_dir = Path(args['data_dir']).expanduser() / date_ani / stim_cond 
    base_dir = Path(args['base_dir']).expanduser()
    save_dir_fm = base_dir / date_ani / 'fm1'
    # save_dir_hf = base_dir / date_ani / 'hf1_wn'
    save_dir_fm.mkdir(parents=True, exist_ok=True)
    # save_dir_hf.mkdir(parents=True, exist_ok=True)
    save_dir = (base_dir / date_ani / stim_cond)
    save_dir.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = (Path(args['fig_dir']).expanduser()/date_ani/stim_cond)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ##### Set up exp name #####
    # if exp_dir_name is None: 
    #     if args['shifter_5050']:
    #         exp_dir_name = 'shifter5050'
    #     elif args['EyeHead_only']:
    #         exp_dir_name = 'EyeHead_only'
    #     elif args['only_spdpup']:
    #         exp_dir_name = 'OnlySpdPupil'
    #     elif args['crop_input']:
    #         exp_dir_name = 'CropInputs'
    #     else:
    #         exp_dir_name = 'RevisionSims'
            
    # exp = Experiment(name='ModelID{}'.format(ModelID),
    #                     save_dir=save_dir / exp_dir_name, #'Shifter_TrTe_testing2', #'GLM_Network',#
    #                     debug=debug,
    #                     version=nKfold)

    save_model = save_dir / 'version_{}'.format(nKfold)
    save_model_Vis = save_dir /'vis_version_{}'.format(nKfold)
    save_dir_fm_exp = save_dir
    save_dir_fm_exp.mkdir(parents=True, exist_ok=True)
    # save_dir_hf_exp = save_dir_hf / save_dir
    # save_dir_hf_exp.mkdir(parents=True, exist_ok=True)
    save_model_shift = save_dir / 'shifter'
    save_model_shift.mkdir(parents=True, exist_ok=True)
    if args['train_shifter']:
        save_model = save_model_shift

    params = {
        ##### Data Parameters #####
        'data_dir':                 data_dir,
        'base_dir':                 base_dir,
        'exp_name_base':            base_dir.name,
        'free_move':                free_move,
        'stim_cond':                stim_cond,
        'save_dir':                 save_dir,
        'save_dir_fm':              save_dir_fm,
        'fig_dir':                  fig_dir,
        'save_model':               save_model,
        'save_model_Vis':           save_model_Vis,
        'save_model_shift':         save_model_shift,
        'date_ani2':                date_ani2,
        'model_dt':                 args['model_dt'],
        'quantiles':                [.05, .95],
        'thresh_cells':             args['thresh_cells'], 
        ##### Model Parameters #####
        'lag_list':                 [-2,-1,0,1,2], # List of which timesteps to include in model fit
        'Nepochs':                  args['Nepochs'], # Number of steps for training
        'do_shuffle':               args['do_shuffle'], # do_shuffle=True, shuffles spikes
        'do_norm':                  args['do_norm'], # z-scores inputs
        'train_shifter':            args['train_shifter'], # flag for training shifter network
        'thresh_shifter':           True,
        'ModelID':                  ModelID, # ModelID 1=Vis, 2=Add, 3=Mult., 4=HeadFixed
        'load_Vis' :                True if ModelID>1 else False,
        'LinMix':                   True if ModelID==2 else False,
        'Kfold':                    args['Kfold'], # Number of Kfolds. Default=1
        'NoL1':                     args['NoL1'], # Remove L1 regularization
        'NoL2':                     args['NoL2'], # Remove L2 regularization
        'position_vars':            ['th','phi','pitch','roll'], # ,'speed','eyerad'], # Which variables to use for position fits
        'use_spdpup':               args['use_spdpup'],
        'only_spdpup':              args['only_spdpup'],
        'EyeHead_only':             args['EyeHead_only'],
        'EyeHead_only_run':         args['EyeHead_only_run'],
        'SimRF':                    args['SimRF'],
        'NoShifter':                args['NoShifter'],
        'downsamp_vid':             args['ds_vid'], # Downsample factor. Default=4
        'shifter_5050':             args['shifter_5050'], # Shifter 50/50 controls
        'initW':                    'zero', # initialize W method. 'zero' or 'normal' 
        'optimizer':                'adam', # optimizer: 'adam' or 'sgd'
        'bin_length':               40,
        'shifter_train_size':       .9,
        'shift_hidden':             20,
        'shifter_5050_run':         args['shifter_5050_run'],
        'crop_input':               5 if args['crop_input']==True else 0,
    }

    params['nt_glm_lag'] = len(params['lag_list']) # number of timesteps for model fits
    params['data_name'] = 'testdataname' # '_'.join([params['date_ani2'],params['stim_cond']])
    # params['data_name_hf'] = '_'.join([params['date_ani2'],'hf1_wn'])
    # params['data_name_fm'] = '_'.join([params['date_ani2'],params['fm_dir']])
    params['data_name_hf'] = 'testdataname'
    params['data_name_fm'] = 'testdataname'

    ##### Saves yaml of parameters #####
    # if debug==False:
    #     params2=params.copy()
    #     for key in params2.keys():
    #         if isinstance(params2[key], Path):
    #             params2[key]=params2[key].as_posix()

    #     pfile_path = save_model / 'model_params.yaml'
    #     with open(pfile_path, 'w') as file:
    #         doc = yaml.dump(params2, file, sort_keys=True)

    # if file_dict is None:
    #     file_dict = {'cell': 0,
    #                 'drop_slow_frames': False,
    #                 'ephys': list(params['data_dir'].glob('*ephys_merge.json'))[0].as_posix(),
    #                 'ephys_bin': list(params['data_dir'].glob('*Ephys.bin'))[0].as_posix(),
    #                 'eye': list(params['data_dir'].glob('*REYE.nc'))[0].as_posix(),
    #                 'imu': list(params['data_dir'].glob('*imu.nc'))[0].as_posix() if params['stim_cond'] == params['fm_dir'] else None,
    #                 'mp4': True,
    #                 'name': params['date_ani2'] + '_control_Rig2_' + params['stim_cond'],
    #                 'probe_name': 'DB_P128-6',
    #                 'save': params['data_dir'].as_posix(),
    #                 'speed': list(params['data_dir'].glob('*speed.nc'))[0].as_posix() if params['stim_cond'] == 'hf1_wn' else None,
    #                 'stim_cond': 'light',
    #                 'top': list(params['data_dir'].glob('*TOP1.nc'))[0].as_posix() if params['stim_cond'] == params['fm_dir'] else None,
    #                 'world': list(params['data_dir'].glob('*world.nc'))[0].as_posix(), 
    #                 'ephys_csv': list(params['data_dir'].glob('*Ephys_BonsaiBoardTS.csv'))[0].as_posix()}


    return params



def make_network_config(params,single_trial=None,custom=False):
    """ Create Network Config dictionary for hyperparameter search

    Args:
        params (dict): key parameters dictionary
        single_trial (int): trial number for if testing single trial. If None, 
                            assumes using ray tune hyperparam search. Default=None
        custom (bool): If custom data ommit shifter params
    Returns:
        network_config (dict): dictionary with hyperparameters
    """
    network_config = {}
    network_config['in_features']   = params['nk']
    network_config['Ncells']        = params['Ncells']
    network_config['initW']         = params['initW']
    network_config['optimizer']         = params['optimizer']
    network_config['activation_type']   ='ReLU' # Default is ReLU, choose ReLu, SoftPlus, or None
    if custom == False:
        network_config['shift_in']      = params['shift_in']
        network_config['shift_hidden']  = params['shift_hidden']
        network_config['shift_out']     = params['shift_out']
        network_config['LinMix']        = params['LinMix']
        network_config['pos_features']  = params['pos_features']
        network_config['lr_shift']      = 1e-2
    network_config['lr_w']          = 1e-2 # was 1e-3
    network_config['lr_b']          = 1e-2 # was 1e-3
    # network_config['lr_m']          = 1e-4 # was 1e-3
    network_config['single_trial']  = single_trial
    if params['NoL1']:
        network_config['L1_alpha']  = None
        network_config['L1_alpha_m'] = None
    else:
        network_config['L1_alpha']  = .0001
        network_config['L1_alpha_m'] = None

    if params['NoL2']:
        network_config['L2_lambda']   = 0
        network_config['L2_lambda_m'] = 0
        initial_params={}
    else:
        if single_trial is not None:
            network_config['L2_lambda']   = 13 #np.logspace(-2, 3, 20)[-5]
            network_config['L2_lambda_m'] = 0 #np.logspace(-2, 3, 20)[-5]
            initial_params={}
        else:
            # network_config['L2_lambda']   = tune.grid_search(list(np.logspace(-2, 3,num=20)))
            # network_config['L2_lambda_m'] = tune.loguniform(1e-2, 1e3)
            network_config['L2_lambda_m'] = 0 #tune.loguniform(1e-2, 1e3)
            initial_params = [{'L2_lambda':.01},]
    return network_config, initial_params


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


def load_position_data(h5_path, lags=None, device=device):
    """
    Load and format data from the preprocessed HDF5 file.
    Extracts theta, phi, and head rotations (yaw, roll, pitch).
    """
    # Load the dictionary from the HDF5 file
    data = fm2p.read_h5(h5_path)
    
    # Extract features
    # Ensure we use the interpolated/aligned versions where available
    theta = data.get('theta_interp')
    phi = data.get('phi_interp')
    
    # Handle Yaw (Head Rotation)
    # 'upsampled_yaw' is typically created during IMU processing in preprocess()
    # 'head_yaw_deg' comes from top-down tracking
    # if 'upsampled_yaw' in data:
    #     yaw = data['upsampled_yaw']['igyro_corrected_deg']
    # elif 'head_yaw_deg' in data:
    #     yaw = data['head_yaw_deg']
    # else:
    #     # Fallback if no yaw data found, though expected for free moving
    #     yaw = np.zeros_like(theta)
    yaw = data['head_yaw_deg']
        
    # Handle Roll and Pitch (IMU data)
    # Assuming they are stored as 'roll' and 'pitch' in the dict if IMU was present
    # and aligned. If not, use zeros.
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
    
    # Check lengths and trim to minimum common length to ensure alignment
    # (preprocess usually aligns them, but safety check is good)
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
    
    # Extract Spikes
    spikes = data.get('norm_spikes')
    if spikes is None:
        raise ValueError("norm_spikes not found in HDF5 file.")
        
    min_len = min(min_len, spikes.shape[1])
    
    # Trim all arrays to the same length
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
    
    # Create input matrix: [Samples x Features]
    # Features: theta, phi, yaw, roll, pitch
    X = np.stack([theta, phi, yaw, roll, pitch, dTheta, dPhi, gyro_x, gyro_y, gyro_z], axis=1)
    
    # Normalize inputs (Z-score)
    X_mean = np.nanmean(X, axis=0)
    X_std = np.nanstd(X, axis=0)
    X_std[X_std == 0] = 1.0 # Avoid division by zero
    X = (X - X_mean) / X_std
    X[np.isnan(X)] = 0.0

    if lags is not None:
        X = add_temporal_lags(X, lags)
    
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    if np.isnan(spikes).any():
        spikes[np.isnan(spikes)] = 0.0
    
    print(f"Spikes stats -- Mean: {np.nanmean(spikes):.4f}, Std: {np.nanstd(spikes):.4f}, Max: {np.nanmax(spikes):.4f}")
    Y_tensor = torch.tensor(spikes, dtype=torch.float32).to(device)
    
    return X_tensor, Y_tensor



def setup_model_training(model,params,network_config):
    """Set up optimizer and scheduler for training

    Args:
        model (nn.Module): Network model to train
        params (dict): key parameters 
        network_config (diot): dictionary of hyperparameters

    Returns:
        optimizer: pytorch optimizer
        scheduler: learning rate scheduler
    """
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
    # print(check_names)
                
    if network_config['optimizer'].lower()=='adam':
        optimizer = optim.Adam(params=param_list)
    else:
        optimizer = optim.SGD(params=param_list)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(params['Nepochs']/5))
    return optimizer, scheduler


def train_position_model(h5_path, config, save_path=None, device=device):
    """
    Train the PositionGLM model using data from the HDF5 file.
    """
    lags = config.get('lags', None)
    # Load Data
    X, Y = load_position_data(h5_path, lags=lags, device=device)
    
    # Split into train and test (e.g., 80/20 split)
    n_samples = X.shape[0]
    n_train = int(0.8 * n_samples)
    
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_test, Y_test = X[n_train:], Y[n_train:]
    
    # Update config with dimensions
    config['in_features'] = X.shape[1]
    config['Ncells'] = Y.shape[1]
    
    # Initialize Model
    model = PositionGLM(config['in_features'], config['Ncells'], config, device=device)
    model.to(device)
    
    # Setup optimizer
    # Using ModelID=0 to indicate simple GLM structure (no shifter) in setup_model_training
    params = {'ModelID': 0, 'Nepochs': config.get('Nepochs', 1000), 'train_shifter': False}
    optimizer, scheduler = setup_model_training(model, params, config)
    
    # Training Loop
    model.train()
    print("Starting training...")
    for epoch in range(params['Nepochs']):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train)
        
        # Loss
        loss = model.loss(outputs, Y_train)
        
        # Backward
        loss.sum().backward()
        optimizer.step()
        
        if scheduler:
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
    # Parse arguments
    args = arg_parser()
    
    # Load parameters
    # Using ModelID=1 to load general parameters
    params = load_params(args, ModelID=1)

    # Direct path to the preprocessed HDF5 file
    h5_path = Path('/home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings/cohort02_recordings/251016_DMM_DMM061_pos18/fm1/251016_DMM_DMM061_fm_01_preproc.h5')

    # Create config dict for PositionGLM
    # Using appropriate values from params.py
    pos_config = {
        'activation_type': 'SoftPlus',
        'loss_type': 'poisson',
        'initW': 'normal', # Changed to normal to help find peaks faster
        'optimizer': params['optimizer'],
        'lr_w': 1e-2, # was 1e-3
        'lr_b': 1e-2, # was 1e-3
        'L1_alpha': 1e-4, # Regularization for position terms
        'Nepochs': params['Nepochs'],
        'L2_lambda': 1e-4, # Not sure if this is best
        'lags': np.arange(-20,21,1), # Reduced lags slightly to focus on immediate events
        'hidden_size': 128, # Increased capacity
        'dropout': 0.2 # Added dropout to prevent overfitting
    }

    # Run training
    if h5_path.exists():
        print(f"Starting training on {h5_path}")
        model, X_test, y_test = train_position_model(h5_path, pos_config)

    else:
        print(f"File not found: {h5_path}")

    loss = test_position_model(model, X_test, y_test)

    # --- Evaluation ---
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