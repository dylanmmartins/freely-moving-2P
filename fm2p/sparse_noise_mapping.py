import os

import fm2p

def sparse_noise_mapping():

    data_path = fm2p.select_file(
        'Select preprocessed HDF file.',
        filetypes=[('HDF','.h5'),]
    )

    sta_light_all, sta_dark_all, lag_axis, est_delay_frames = fm2p.calc_sparse_noise_STAs(
        data_path
    )

    dict_out = {
        'lightSTA': sta_light_all,
        'darkSTA': sta_dark_all,
        'lags': lag_axis,
        'delay': est_delay_frames
    }

    savepath = os.path.join(os.path.split(data_path)[0], 'sparse_noise_receptive_fields.h5')
    fm2p.write_h5(savepath, dict_out)


if __name__ == '__main__':
    sparse_noise_mapping()