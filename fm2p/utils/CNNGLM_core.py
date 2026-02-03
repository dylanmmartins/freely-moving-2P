
import tensorflow as tf
import numpy as np
import fm2p


class EncodingCNNGLM(tf.keras.Model):
    def __init__(self, n_neurons, n_predictors, fs=7.5, l2_scale=9.17e-7):
        """
        CNN-GLM Encoding Model based on Minderer et al., 2019.
        
        Args:
            n_neurons (int): Number of neurons to predict simultaneously.
            n_predictors (int): Number of task variable channels.
            fs (float): Sampling frequency in Hz (default 7.5).
            l2_scale (float): L2 regularization factor (paper used 9.17e-7).
        """
        super(EncodingCNNGLM, self).__init__()
        
        # --- Hyperparameters ---
        self.fs = fs
        self.n_neurons = n_neurons
        self.l2_reg = tf.keras.regularizers.L2(l2_scale)
        
        # Dimensions based on paper description
        snippet_duration_s = 4.0
        filter_duration_s = 0.6
        
        # Convert seconds to frames for your fs
        self.input_frames = int(snippet_duration_s * fs)
        self.filter_size = int(filter_duration_s * fs)
        
        self.conv1 = tf.keras.layers.Conv1D(
            32, self.filter_size, padding='same',
            kernel_initializer=self._custom_he_init(self.filter_size * n_predictors),
            kernel_regularizer=self.l2_reg
        )
        self.conv2 = tf.keras.layers.Conv1D(
            32, self.filter_size, padding='same',
            kernel_initializer=self._custom_he_init(self.filter_size * 32),
            kernel_regularizer=self.l2_reg
        )
        self.conv3 = tf.keras.layers.Conv1D(
            32, self.filter_size, padding='same',
            kernel_initializer=self._custom_he_init(self.filter_size * 32),
            kernel_regularizer=self.l2_reg
        )
        
        self.flatten = tf.keras.layers.Flatten()
        
        fan_in_dense1 = 32 * self.input_frames
        self.dense1 = tf.keras.layers.Dense(128,
                                            kernel_initializer=self._custom_he_init(fan_in_dense1),
                                            kernel_regularizer=self.l2_reg)
        
        self.dense_bottleneck = tf.keras.layers.Dense(n_predictors, 
                                                      kernel_initializer=self._custom_he_init(128),
                                                      kernel_regularizer=self.l2_reg)
        
        self.readout = tf.keras.layers.Dense(n_neurons, 
                                             kernel_initializer=self._custom_he_init(n_predictors),
                                             bias_initializer='zeros',
                                             kernel_regularizer=self.l2_reg)
        
        self.poly_readout = tf.keras.layers.Dense(n_neurons,
                                                  use_bias=False,
                                                  kernel_initializer='zeros',
                                                  name="poly_offset")

        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.dropout = tf.keras.layers.Dropout(0.5)


    def _custom_he_init(self, fan_in):
        """
        Implements the specific initialization from the paper:
        Truncated normal with stddev = 0.1 * sqrt(2/n).
        """
        stddev = 0.1 * np.sqrt(2.0 / fan_in)
        return tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=stddev)


    def call(self, inputs, training=False):
        """
        Args:
            inputs: tuple of (task_snippet, vrf_prediction, time_poly)
                - task_snippet: (batch, input_frames, n_predictors)
                - vrf_prediction: (batch, n_neurons) [Log-space prediction from visual model]
                - time_poly: (batch, 4) [Polynomial basis: 1, t, t^2, t^3]
        """
        task_snippet, vrf_prediction, time_poly = inputs
        
        # fwd pass
        x = self.conv1(task_snippet)
        x = self.leaky_relu(x)
        if training: x = self.dropout(x)
            
        x = self.conv2(x)
        x = self.leaky_relu(x)
        if training: x = self.dropout(x) # dropout to conv layers
            
        x = self.conv3(x)
        x = self.leaky_relu(x)
        if training: x = self.dropout(x)
        
        x = self.flatten(x)
        
        x = self.dense1(x)
        x = self.leaky_relu(x)
        if training: x = self.dropout(x) # dropout to first dense layer
            
        task_factors = self.dense_bottleneck(x)
        task_factors = self.leaky_relu(task_factors)
        
        cnn_log_pred = self.readout(task_factors)
        
        poly_offset = self.poly_readout(time_poly)
        
        total_log_pred = cnn_log_pred + vrf_prediction + poly_offset # prev transposed crf
        
        return tf.math.exp(total_log_pred)

def poisson_loss(y_true, y_pred):

    return tf.keras.losses.poisson(y_true, y_pred)


def create_encoding_dataset(
    predictor_data, 
    neural_data, 
    vrf_predictions=None, 
    fs=7.5, 
    batch_size=128, 
    shuffle=True
):
    """
    Creates a TensorFlow Dataset for the Minderer et al. (2019) model.
    
    Args:
        predictor_data (np.array): Shape (n_timepoints, n_predictors). 
                                   Continuous task variables (velocity, etc.).
        neural_data (np.array): Shape (n_timepoints, n_neurons).
                                Deconvolved fluorescence traces.
        vrf_predictions (np.array): Shape (n_timepoints, n_neurons).
                                    Optional. Log-space predictions from visual model.
                                    If None, zeros are used.
        fs (float): Sampling rate (default 7.5 Hz).
        batch_size (int): Batch size for training.
        shuffle (bool): Whether to shuffle the windows (set False for testing/validation).
        
    Returns:
        tf.data.Dataset: Yields ((X_task, X_vrf, X_poly), y_neural)
    """
    
    n_samples, n_neurons = neural_data.shape
    
    snippet_len_s = 4.0
    snippet_len_frames = int(snippet_len_s * fs)  # 30 frames
    
    target_delay_s = 3.0 
    target_offset_frames = int(target_delay_s * fs) # 22 frames
    
    max_start_idx = n_samples - snippet_len_frames

    t_norm = np.linspace(0, 1, n_samples)
    poly_features = np.stack([
        np.ones(n_samples), # Offset
        t_norm, # Linear
        t_norm**2,# Quadratic
        t_norm**3  # Cubic
    ], axis=1).astype(np.float32)


    if vrf_predictions is None:
        vrf_predictions = np.zeros_like(neural_data, dtype=np.float32)

    def data_generator():
        indices = np.arange(max_start_idx)
        if shuffle:
            np.random.shuffle(indices)

        for start_i in indices:

            x_task = predictor_data[start_i : start_i + snippet_len_frames]
            
            target_i = start_i + target_offset_frames
            
            y = neural_data[target_i] # (n_neurons,)
            x_vrf = vrf_predictions[target_i] # (n_neurons,)
            x_poly = poly_features[target_i] #(4,)
            
            yield (x_task, x_vrf, x_poly), y

    output_signature = (
        (
            tf.TensorSpec(shape=(snippet_len_frames, predictor_data.shape[1]), dtype=tf.float32), # X_task
            tf.TensorSpec(shape=(n_neurons,), dtype=tf.float32), # X_vrf
            tf.TensorSpec(shape=(4,), dtype=tf.float32) # X_poly
        ),
        tf.TensorSpec(shape=(n_neurons,), dtype=tf.float32) # y target
    )

    dataset = tf.data.Dataset.from_generator(
        data_generator, 
        output_signature=output_signature
    )
    
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


if __name__ == "__main__":
    
    data = fm2p.read_h5('/home/dylan/Fast0/Dropbox/251016_DMM_DMM056_pos13_LNLP_results//251016_DMM_DMM056_fm_01_preproc.h5')

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

    if 'dGaze' in data.keys():
        hasIMU = True
    else:
        hasIMU = False

    if hasIMU:
        gaze = np.cumsum(data['dGaze'].copy())
        dGaze = data['dGaze'].copy()
        gazeT = data['eyeT_trim'] + (np.nanmedian(data['eyeT_trim']) / 2)
        gazeT = gazeT[:-1]
        gaze = fm2p.interpT(gaze, gazeT, data['twopT'])
        dGaze = fm2p.interpT(dGaze, gazeT, data['twopT'])

        behavior_vars = {
            # head positions
            'yaw': data['head_yaw_deg'].copy(),
            'pitch': data['pitch_twop_interp'].copy(),
            'roll': data['roll_twop_interp'].copy(),
            # gaze
            # 'gaze': gaze,
            # 'dGaze': dGaze,
            # eye positions
            'theta': data['theta_interp'].copy(),
            'phi': data['phi_interp'].copy(),
            # eye speeds
            'dTheta': dTheta,
            'dPhi': dPhi,
            # head angular rotation speeds
            'gyro_x': data['gyro_x_twop_interp'].copy(),
            'gyro_y': data['gyro_y_twop_interp'].copy(),
            'gyro_z': data['gyro_z_twop_interp'].copy(),
            # head accelerations
            'acc_x': data['acc_x_twop_interp'].copy(),
            'acc_y': data['acc_y_twop_interp'].copy(),
            'acc_z': data['acc_z_twop_interp'].copy()
        }

        if len(behavior_vars['yaw']) > len(data['norm_spikes']):
            behavior_vars['yaw'] = behavior_vars['yaw'][:-1]

    elif not hasIMU:

        behavior_vars = {
            # eye positions
            'theta': data['theta_interp'].copy(),
            'phi': data['phi_interp'].copy(),
            # eye speeds
            'dTheta': dTheta,
            'dPhi': dPhi
        }

    raw_dff = data['raw_dFF'].copy().astype(np.float32)

    raw_task_vars = np.zeros([
        len(behavior_vars.keys()),
        len(behavior_vars['theta'])
    ])
    behavior_keys = []
    for vi, (key, var) in enumerate(behavior_vars.items()):
        raw_task_vars[vi,:] = var
        behavior_keys.append(key)

    use = (data['speed'] > 2.0) * (data['ltdk_state_vec']) * (np.sum(np.isnan(raw_task_vars), axis=0) == 0)
    raw_task_vars = raw_task_vars[:, use]
    # Apply same mask to neural data so predictor and neural lengths match
    raw_dff = raw_dff[:, use]

    raw_task_vars = raw_task_vars.T
    raw_dff = raw_dff.T

    N_NEURONS = np.size(raw_dff, 1)
    N_PREDICTORS = np.size(raw_task_vars, 1)
    FS = 7.5
    N_SAMPLES = np.size(raw_task_vars, 0)

    # Optional: Visual Receptive Field residuals (use None if you don't have them)
    # We will assume they are zero for this example
    raw_vrf = None 

    # 80/20 split
    split_idx = int(0.8 * N_SAMPLES)

    # dataset/window parameters (must match create_encoding_dataset)
    snippet_len_s = 4.0
    snippet_len_frames = int(snippet_len_s * FS)

    BATCH_SIZE = 128

    # compute number of windows for train/val so we can provide steps
    n_train_samples = split_idx
    train_windows = max(0, n_train_samples - snippet_len_frames)
    steps_per_epoch = max(1, int(np.ceil(train_windows / BATCH_SIZE)))

    n_val_samples = N_SAMPLES - split_idx
    val_windows = max(0, n_val_samples - snippet_len_frames)
    validation_steps = max(1, int(np.ceil(val_windows / BATCH_SIZE)))

    train_ds = create_encoding_dataset(
        predictor_data=raw_task_vars[:split_idx],
        neural_data=raw_dff[:split_idx],
        vrf_predictions=None, # Will default to zeros
        fs=FS,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_ds = create_encoding_dataset(
        predictor_data=raw_task_vars[split_idx:],
        neural_data=raw_dff[split_idx:],
        vrf_predictions=None,
        fs=FS,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = EncodingCNNGLM(n_neurons=N_NEURONS, n_predictors=N_PREDICTORS, fs=FS)
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.0027, beta_1=0.89, beta_2=0.9999)
    
    model.compile(optimizer=opt, loss=poisson_loss)
    
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        verbose=1,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
