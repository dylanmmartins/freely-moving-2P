# -*- coding: utf-8 -*-





import numpy as np
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing

import fm2p

def pred_pupil():

    data = fm2p.read_h5(r'Z:\Mini2P_data\250626_DMM_DMM037_ltdk\fm1\250626_DMM_DMM037_fm_01_preproc.h5')

    spikes = data['norm_spikes'].copy()
    egocentric = data['egocentric'].copy()
    retinocentric = data['retinocentric'].copy()
    pupil = data['pupil_from_head'].copy()
    speed = data['speed'].copy()
    speed = np.append(speed, speed[-1])
    use = speed > 1.5
    twopT = data['twopT'].copy()
    theta = data['theta_interp'].copy()
    phi = data['phi_interp'].copy()
    ltdk = data['ltdk_state_vec'].copy()


    X = spikes.copy()
    y_ = theta.copy()

    lag = 0 # in units of frames

    # Rolling spikes, so a positive value for `lag` means that spikes precede behavior
    # (not what we want). Negative values mean behavior precedes spikes (what we want).
    if lag != 0:
        y = np.roll(y_, lag)
    else:
        y = y_.copy()

    # Anywhere that y has a NaN, drop both y and X
    tokeep = ~np.isnan(y)
    X = X[:, tokeep].T
    y = y[tokeep]
    y = y[np.newaxis, :]
    y = y.T

    scalerY = preprocessing.RobustScaler().fit(y)
    y_scaled = scalerY.transform(y)

    train_inds = []
    test_inds = []
    ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
    for i, (train_index, test_index) in enumerate(ss.split(X)):
        train_inds.append(np.array(sorted(train_index)))
        test_inds.append(np.array(sorted(test_index)))

    dataout = {}

    for k in range(5):
        print('Training for fold={}'.format(k+1))

        tr = train_inds[k]
        te = test_inds[k]

        X_train = X.copy()[tr]
        y_train = y_scaled.copy()[tr]

        X_test = X.copy()[te]
        y_test = y_scaled.copy()[te]
        y_test_nontransform = y.copy()[te]

        model = GradientBoostingRegressor(
            n_estimators=1500, learning_rate=0.1, verbose=True
            ).fit(
                X_train,
                y_train
            )
        y_hat = model.predict(X_test)
        y_hat_train = model.predict(X_train)

        y_hat_train_it = scalerY.inverse_transform(y_hat_train[np.newaxis,:]).T
        y_hat_it = scalerY.inverse_transform(y_hat[np.newaxis,:]).T

        tr_err = r2_score(y_train, y_hat_train_it)
        te_err = r2_score(y_test, y_hat_it)

        print('{} train error: {:.6}   test error: {:.6}'.format(
            k,
            tr_err,
            te_err
        ))

        savedict = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'y_test_nontransform': y_test_nontransform,
            'y_hat': y_hat,
            'y_hat_train': y_hat_train,
            'y_hat_train_it': y_hat_train_it,
            'y_hat_it': y_hat_it
        }
    
        dataout['fold{}'.format(k+1)] = savedict
    
    fm2p.write_h5('C:/Users/dmartins/pupil_pred_outputs_lr0p1.h5', dataout)

if __name__ == '__main__':

    pred_pupil()
