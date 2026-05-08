

if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

import argparse
import glob
import json
import os

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kruskal
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib as mpl
mpl.rcParams['axes.spines.top']  = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['font.size']    = 7


DEFAULT_POOLED  = '/home/dylan/Fast2/pooled_260407a.h5'
DEFAULT_BASE    = '/home/dylan/Storage/freely_moving_data/_V1PPC'
DEFAULT_OUT_DIR = '.'
MIN_CELLS       = 50

ID_TO_NAME = {5: 'V1', 2: 'RL', 3: 'AM', 4: 'PM', 10: 'A', 7: 'AL', 8: 'LM', 9: 'P'}
REGION_ORDER = ['V1', 'RL', 'AM', 'PM', 'A', 'AL', 'LM', 'P']
COLORS = {
    'V1': '#1B9E77', 'RL': '#D95F02', 'AM': '#7570B3', 'PM': '#E7298A',
    'A':  '#1A5A34', 'AL': '#E6AB02', 'LM': '#A6761D', 'P':  '#666666',
}

_DECODE_LAGS   = 4
_DECODE_PCA    = 50
_DECODE_ALPHAS = [1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10., 30., 100., 300.,
                  1e3, 3e3, 1e4, 3e4, 1e5]


class EyeDecoder:

    def __init__(self, lags: int = _DECODE_LAGS, n_pca: int = _DECODE_PCA,
                 alphas=None):
        self.lags   = lags
        self.n_pca  = n_pca
        self.alphas = alphas or _DECODE_ALPHAS

    def load_data(self, h5_path: str) -> dict:
        data = {}
        with h5py.File(h5_path, 'r') as f:

            startInd = int(f['eyeT_startInd'][()])
            data['eyeT_startInd'] = startInd
            if 'eyeT_trim' in f:
                data['eyeT_trim'] = f['eyeT_trim'][:]
            elif 'eyeT' in f:

                raw = f['eyeT'][:].astype(float)
                data['eyeT_trim'] = raw[startInd:] - raw[startInd]
            else:
                raise KeyError(f'No eye time array found in {h5_path}')

            for key in ('twopT', 'light_onsets', 'dark_onsets',
                        'theta', 'phi', 'ellipse_phi',
                        'longaxis', 'shortaxis', 'X0', 'Y0'):
                if key in f:
                    data[key] = f[key][:]
                else:
                    raise KeyError(f'Required key {key!r} missing from {h5_path}')

            for neural_key in ('norm_spikes', 'norm_dFF'):
                if neural_key in f:
                    data['neural']     = f[neural_key][:]
                    data['neural_key'] = neural_key
                    break
            if 'neural' not in data:
                raise KeyError('Neither norm_spikes nor norm_dFF found.')

            n_twop = len(data['twopT'])
            def _load_head(candidates):
                for cand in candidates:
                    if cand not in f:
                        continue
                    item = f[cand]
                    if not hasattr(item, 'shape'):
                        continue
                    arr = item[:].astype(float)

                    if arr.ndim == 1 and abs(len(arr) - n_twop) <= 10:
                        return arr[:n_twop]
                return np.full(n_twop, np.nan)

            data['pitch_full'] = _load_head(('pitch_twop_interp',))
            data['roll_full']  = _load_head(('roll_twop_interp',))
            data['yaw_full']   = _load_head(('head_yaw_deg',))
            data['speed_full'] = _load_head(('speed',))
        return data

    def get_all_light_blocks(self, data: dict):
        eyeT_trim    = data['eyeT_trim']
        twopT        = data['twopT']
        light_onsets = data['light_onsets'].astype(int)
        dark_onsets  = data['dark_onsets'].astype(int)
        startInd     = data['eyeT_startInd']
        n_trim       = len(eyeT_trim)

        theta_trim = data['theta'][startInd:startInd + n_trim].astype(float)
        phi_trim   = data['phi'  ][startInd:startInd + n_trim].astype(float)

        blocks = []
        best_idx, best_pct = -1, -1.0
        for i in range(1, len(light_onsets)):
            lo  = int(light_onsets[i])
            nxt = dark_onsets[dark_onsets > lo]
            if len(nxt) == 0:
                continue
            nd   = int(nxt[0])
            mask = (eyeT_trim >= twopT[lo]) & (eyeT_trim <= twopT[nd])
            n_eye = int(mask.sum())
            if n_eye == 0:
                continue
            pct = 95.0 * int(np.sum(
                mask & np.isfinite(theta_trim) & np.isfinite(phi_trim))) / n_eye
            blocks.append({'lo': lo, 'nd': nd,
                           't_start': float(twopT[lo]), 't_end': float(twopT[nd]),
                           'pct': pct})
            if pct > best_pct:
                best_pct, best_idx = pct, len(blocks) - 1

        return blocks, best_idx

    def get_all_dark_blocks(self, data: dict):

        eyeT_trim    = data['eyeT_trim']
        twopT        = data['twopT']
        light_onsets = data['light_onsets'].astype(int)
        dark_onsets  = data['dark_onsets'].astype(int)
        startInd     = data['eyeT_startInd']
        n_trim       = len(eyeT_trim)
        n_twop       = len(twopT)

        theta_trim = data['theta'][startInd:startInd + n_trim].astype(float)
        phi_trim   = data['phi'  ][startInd:startInd + n_trim].astype(float)

        blocks = []
        best_idx, best_pct = -1, -1.0
        for i in range(len(dark_onsets)):
            lo = int(dark_onsets[i])
            if lo >= n_twop - 1:
                continue
            nxt = light_onsets[light_onsets > lo]
            if len(nxt) == 0:
                continue
            nd = min(int(nxt[0]), n_twop - 1)
            if lo >= nd:
                continue
            mask  = (eyeT_trim >= twopT[lo]) & (eyeT_trim <= twopT[nd])
            n_eye = int(mask.sum())
            if n_eye == 0:
                continue
            pct = 95.0 * int(np.sum(
                mask & np.isfinite(theta_trim) & np.isfinite(phi_trim))) / n_eye
            blocks.append({'lo': lo, 'nd': nd,
                           't_start': float(twopT[lo]), 't_end': float(twopT[nd]),
                           'pct': pct})
            if pct > best_pct:
                best_pct, best_idx = pct, len(blocks) - 1

        return blocks, best_idx

    def _neural_features(self, X: np.ndarray) -> np.ndarray:
        X = gaussian_filter1d(X.astype(float), sigma=1.5, axis=0)
        parts = [X]
        for lag in range(1, self.lags + 1):
            future = np.concatenate([X[lag:],   np.tile(X[-1:], (lag, 1))], axis=0)
            past   = np.concatenate([np.tile(X[:1], (lag, 1)), X[:-lag]], axis=0)
            parts += [future, past]
        return np.concatenate(parts, axis=1)

    def _make_pipeline(self, n_feat: int, n_samples: int = 10**9) -> Pipeline:
        n_pca = max(1, min(self.n_pca, n_feat - 1, n_samples - 1))
        return Pipeline([
            ('scaler', StandardScaler()),
            ('pca',    PCA(n_components=n_pca, whiten=True)),
            ('ridge',  RidgeCV(alphas=self.alphas, cv=5)),
        ])

    def _extract_cell_weights(self, pipe: Pipeline, n_cells: int) -> np.ndarray:

        # scaler = pipe.named_steps['scaler']
        pca    = pipe.named_steps['pca']
        ridge  = pipe.named_steps['ridge']

        w_feat = pca.components_.T @ ridge.coef_

        n_lag_slots = 1 + 2 * self.lags
        if w_feat.shape[0] != n_cells * n_lag_slots:
            return np.full(n_cells, np.nan)

        w_mat = w_feat.reshape(n_lag_slots, n_cells)
        return np.linalg.norm(w_mat, axis=0)

    def decode(self, data: dict, lo: int, nd: int,
               train_blocks=None, cell_mask=None) -> dict:

        twopT     = data['twopT']
        eyeT_trim = data['eyeT_trim']
        startInd  = data['eyeT_startInd']
        n_trim    = len(eyeT_trim)

        def eye_arr(key):
            return data[key][startInd:startInd + n_trim].astype(float)

        theta_e = eye_arr('theta')
        phi_e   = eye_arr('phi')
        X0_e    = eye_arr('X0')
        Y0_e    = eye_arr('Y0')
        la_e    = eye_arr('longaxis')
        sa_e    = eye_arr('shortaxis')
        ephi_e  = eye_arr('ellipse_phi')

        def to_2p(sig):
            return np.interp(twopT, eyeT_trim, sig)

        theta_2p = to_2p(theta_e)
        phi_2p   = to_2p(phi_e)
        X0_2p    = to_2p(X0_e)
        Y0_2p    = to_2p(Y0_e)
        la_2p    = to_2p(la_e)
        sa_2p    = to_2p(sa_e)
        ephi_2p  = to_2p(ephi_e)

        neural_full = data['neural']
        if cell_mask is not None:
            neural_full = neural_full[cell_mask, :]
        n_cells_used = neural_full.shape[0]

        bt    = theta_2p[lo:nd]
        bp    = phi_2p  [lo:nd]
        bX    = X0_2p   [lo:nd]
        bY    = Y0_2p   [lo:nd]
        bla   = la_2p   [lo:nd]
        bsa   = sa_2p   [lo:nd]
        bephi = ephi_2p [lo:nd]

        neural_T      = neural_full[:, lo:nd].T.astype(float)
        neural_T_feat = self._neural_features(neural_T)

        bspeed    = data['speed_full'][lo:nd]
        speed_ok  = ~np.isfinite(bspeed) | (bspeed > 2.0)

        valid_test = (  np.isfinite(bt)
                      & np.isfinite(bp)
                      & np.isfinite(bX)
                      & np.isfinite(bY)
                      & np.isfinite(neural_T_feat).all(axis=1)
                      & speed_ok)

        if train_blocks:
            X_parts, yt_parts, yp_parts, yX_parts, yY_parts = [], [], [], [], []
            for tb in train_blocks:
                t_lo, t_nd = tb['lo'], tb['nd']
                nt = self._neural_features(
                    neural_full[:, t_lo:t_nd].T.astype(float))
                bt_tr = theta_2p[t_lo:t_nd]
                bp_tr = phi_2p  [t_lo:t_nd]
                bX_tr = X0_2p   [t_lo:t_nd]
                bY_tr = Y0_2p   [t_lo:t_nd]
                sp_tr = data['speed_full'][t_lo:t_nd]
                v = (  np.isfinite(bt_tr) & np.isfinite(bp_tr)
                     & np.isfinite(bX_tr) & np.isfinite(bY_tr)
                     & np.isfinite(nt).all(axis=1)
                     & (~np.isfinite(sp_tr) | (sp_tr > 2.0)))
                if v.sum() > 0:
                    X_parts.append(nt[v])
                    yt_parts.append(bt_tr[v])
                    yp_parts.append(bp_tr[v])
                    yX_parts.append(bX_tr[v])
                    yY_parts.append(bY_tr[v])
            if X_parts:
                X_fit       = np.concatenate(X_parts)
                y_theta_fit = np.concatenate(yt_parts)
                y_phi_fit   = np.concatenate(yp_parts)
                y_X0_fit    = np.concatenate(yX_parts)
                y_Y0_fit    = np.concatenate(yY_parts)
            else:
                X_fit       = neural_T_feat[valid_test]
                y_theta_fit = bt[valid_test]; y_phi_fit = bp[valid_test]
                y_X0_fit    = bX[valid_test]; y_Y0_fit  = bY[valid_test]
        else:
            X_fit       = neural_T_feat[valid_test]
            y_theta_fit = bt[valid_test]; y_phi_fit = bp[valid_test]
            y_X0_fit    = bX[valid_test]; y_Y0_fit  = bY[valid_test]

        if X_fit.shape[0] < 20 or valid_test.sum() < 20:
            return None

        n_feat     = X_fit.shape[1]
        n_samp     = X_fit.shape[0]
        pipe_theta = self._make_pipeline(n_feat, n_samp).fit(X_fit, y_theta_fit)
        pipe_phi   = self._make_pipeline(n_feat, n_samp).fit(X_fit, y_phi_fit)
        pipe_X0    = self._make_pipeline(n_feat, n_samp).fit(X_fit, y_X0_fit)
        pipe_Y0    = self._make_pipeline(n_feat, n_samp).fit(X_fit, y_Y0_fit)

        pred_theta = pipe_theta.predict(neural_T_feat)
        pred_phi   = pipe_phi  .predict(neural_T_feat)
        pred_X0    = pipe_X0   .predict(neural_T_feat)
        pred_Y0    = pipe_Y0   .predict(neural_T_feat)

        def _r(a, b, mask):
            if mask.sum() < 2:
                return float('nan')
            c = np.corrcoef(a[mask], b[mask])
            return float(c[0, 1])

        def _rmse(a, b, mask):
            yt = a[mask].astype(float)
            yp = b[mask].astype(float)
            if len(yt) < 2:
                return float('nan')
            return float(np.sqrt(np.mean((yt - yp) ** 2)))

        def _rmse_circ(a, b, mask):
            yt = a[mask].astype(float)
            yp = b[mask].astype(float)
            if len(yt) < 2:
                return float('nan')
            diff = (yt - yp + 180.0) % 360.0 - 180.0
            return float(np.sqrt(np.mean(diff ** 2)))
        

        def _rmse_index(a, b, mask, circ=False):

            yt = a[mask].astype(float)
            yp = b[mask].astype(float)
            if len(yt) < 2:
                return float('nan')
            
            shuf_yp = np.random.permutation(yp)

            actual_rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
            shuf_rmse = float(np.sqrt(np.mean((yt - shuf_yp) ** 2)))

            idx = 1 - (actual_rmse / shuf_rmse)

            if circ:
                diff = (yt - yp + 180.0) % 360.0 - 180.0
                actual_rmse = float(np.sqrt(np.mean(diff ** 2)))

                shuf_diff = (yt - shuf_yp + 180.0) % 360.0 - 180.0
                shuf_rmse = float(np.sqrt(np.mean(shuf_diff ** 2)))

                idx = 1 - (actual_rmse / shuf_rmse)

            return idx
            

        r_theta = _r(bt, pred_theta, valid_test)
        r_phi   = _r(bp, pred_phi,   valid_test)
        r_X0    = _r(bX, pred_X0,    valid_test)
        r_Y0    = _r(bY, pred_Y0,    valid_test)

        rmse_theta = _rmse_index(bt, pred_theta, valid_test)
        rmse_phi   = _rmse_index(bp, pred_phi,   valid_test)
        rmse_X0    = _rmse_index(bX, pred_X0,    valid_test)
        rmse_Y0    = _rmse_index(bY, pred_Y0,    valid_test)

        weights = {
            'theta': self._extract_cell_weights(pipe_theta, n_cells_used),
            'phi':   self._extract_cell_weights(pipe_phi,   n_cells_used),
            'X0':    self._extract_cell_weights(pipe_X0,    n_cells_used),
            'Y0':    self._extract_cell_weights(pipe_Y0,    n_cells_used),
        }

        gt_pitch = data['pitch_full'][lo:nd]
        gt_roll  = data['roll_full' ][lo:nd]
        gt_yaw   = data['yaw_full'  ][lo:nd]

        def _build_train_head(signal_full):

            if train_blocks is None:
                return None, None
            Xp, yp = [], []
            for tb in train_blocks:
                t_lo, t_nd = tb['lo'], tb['nd']
                nt  = self._neural_features(neural_full[:, t_lo:t_nd].T.astype(float))
                sig = signal_full[t_lo:t_nd]
                sp_tr = data['speed_full'][t_lo:t_nd]
                v   = (np.isfinite(sig) & np.isfinite(nt).all(axis=1)
                       & (~np.isfinite(sp_tr) | (sp_tr > 2.0)))
                if v.sum() > 0:
                    Xp.append(nt[v]); yp.append(sig[v])
            if Xp:
                return np.concatenate(Xp), np.concatenate(yp)
            return None, None

        pred_pitch = np.full(nd - lo, np.nan)
        pred_roll  = np.full(nd - lo, np.nan)
        pred_yaw   = np.full(nd - lo, np.nan)
        r_pitch = r_roll = r_yaw = float('nan')
        rmse_pitch = rmse_roll = rmse_yaw = float('nan')
        gt_yaw_w = ((gt_yaw + 180) % 360) - 180

        vpr = np.isfinite(gt_pitch) & np.isfinite(gt_roll) & np.isfinite(neural_T_feat).all(axis=1) & speed_ok
        if vpr.sum() > 1:
            Xtp, ytp = _build_train_head(data['pitch_full'])
            Xtr, ytr = _build_train_head(data['roll_full'])
            if Xtp is None:
                Xtp, ytp = neural_T_feat[vpr], gt_pitch[vpr]
                Xtr, ytr = neural_T_feat[vpr], gt_roll [vpr]
            pp = self._make_pipeline(n_feat, Xtp.shape[0]).fit(Xtp, ytp)
            pr = self._make_pipeline(n_feat, Xtr.shape[0]).fit(Xtr, ytr)
            pred_pitch = pp.predict(neural_T_feat)
            pred_roll  = pr.predict(neural_T_feat)
            r_pitch    = _r   (gt_pitch, pred_pitch, vpr)
            r_roll     = _r   (gt_roll,  pred_roll,  vpr)
            rmse_pitch = _rmse(gt_pitch, pred_pitch, vpr)
            rmse_roll  = _rmse(gt_roll,  pred_roll,  vpr)
            weights['pitch'] = self._extract_cell_weights(pp, n_cells_used)
            weights['roll']  = self._extract_cell_weights(pr, n_cells_used)

        vy = np.isfinite(gt_yaw) & np.isfinite(neural_T_feat).all(axis=1) & speed_ok
        if vy.sum() > 1:
            yaw_rad = np.radians(gt_yaw)
            Xts, yts = _build_train_head(np.sin(np.radians(data['yaw_full'])))
            Xtc, ytc = _build_train_head(np.cos(np.radians(data['yaw_full'])))
            if Xts is None:
                Xts, yts = neural_T_feat[vy], np.sin(yaw_rad[vy])
                Xtc, ytc = neural_T_feat[vy], np.cos(yaw_rad[vy])
            ps = self._make_pipeline(n_feat, Xts.shape[0]).fit(Xts, yts)
            pc = self._make_pipeline(n_feat, Xtc.shape[0]).fit(Xtc, ytc)
            pred_yaw = np.degrees(np.arctan2(
                ps.predict(neural_T_feat), pc.predict(neural_T_feat)))
            r_yaw    = _r        (gt_yaw_w, pred_yaw, vy)
            rmse_yaw = _rmse_index(gt_yaw_w, pred_yaw, vy, circ=True)

            ws = self._extract_cell_weights(ps, n_cells_used)
            wc = self._extract_cell_weights(pc, n_cells_used)
            weights['yaw'] = 0.5 * (ws + wc)

        return dict(
            gt_theta=np.degrees(bt),           gt_phi=np.degrees(bp),
            gt_X0=bX,                          gt_Y0=bY,
            pred_theta=np.degrees(pred_theta), pred_phi=np.degrees(pred_phi),
            pred_X0=pred_X0,                   pred_Y0=pred_Y0,
            gt_longaxis=bla, gt_shortaxis=bsa, gt_ellipse_phi=bephi,
            gt_pitch=gt_pitch, gt_roll=gt_roll, gt_yaw=gt_yaw_w,
            pred_pitch=pred_pitch, pred_roll=pred_roll, pred_yaw=pred_yaw,
            valid_test=valid_test, valid_pitch_roll=vpr, valid_yaw=vy,
            r_theta=r_theta,     r_phi=r_phi,     r_X0=r_X0,     r_Y0=r_Y0,
            r_pitch=r_pitch,     r_roll=r_roll,   r_yaw=r_yaw,
            rmse_theta=rmse_theta, rmse_phi=rmse_phi, rmse_X0=rmse_X0, rmse_Y0=rmse_Y0,
            rmse_pitch=rmse_pitch, rmse_roll=rmse_roll, rmse_yaw=rmse_yaw,
            weights=weights,
        )

    def decode_kfold(self, data: dict, blocks: list, cell_mask=None) -> dict:

        if len(blocks) < 2:
            return None

        _r_keys  = ('r_theta', 'r_phi', 'r_X0', 'r_Y0',
                    'r_pitch', 'r_roll', 'r_yaw',
                    'rmse_theta', 'rmse_phi', 'rmse_X0', 'rmse_Y0',
                    'rmse_pitch', 'rmse_roll', 'rmse_yaw')
        _arr_keys = ('gt_theta', 'gt_phi', 'gt_X0', 'gt_Y0',
                     'pred_theta', 'pred_phi', 'pred_X0', 'pred_Y0',
                     'gt_longaxis', 'gt_shortaxis', 'gt_ellipse_phi',
                     'gt_pitch', 'gt_roll', 'gt_yaw',
                     'pred_pitch', 'pred_roll', 'pred_yaw',
                     'valid_test', 'valid_pitch_roll', 'valid_yaw')

        fold_rs   = {k: [] for k in _r_keys}
        cat_arrs  = {k: [] for k in _arr_keys}
        weight_folds = []
        n_valid   = 0

        for i, test_block in enumerate(blocks):
            train = [b for j, b in enumerate(blocks) if j != i]
            result = self.decode(data, test_block['lo'], test_block['nd'],
                                 train_blocks=train, cell_mask=cell_mask)
            if result is None:
                continue
            n_valid += 1
            for k in _r_keys:
                v = result.get(k, float('nan'))
                if np.isfinite(v):
                    fold_rs[k].append(v)
            for k in _arr_keys:
                if k in result:
                    cat_arrs[k].append(result[k])
            if result.get('weights'):
                weight_folds.append(result['weights'])

        if n_valid == 0:
            return None

        avg_r  = {k: float(np.mean(v)) if v else float('nan')
                  for k, v in fold_rs.items()}
        concat = {k: np.concatenate(parts)
                  for k, parts in cat_arrs.items() if parts}

        if weight_folds:
            avg_weights = {}
            for wk in weight_folds[0]:
                ws = [wf[wk] for wf in weight_folds
                      if wk in wf and wf[wk] is not None]
                if ws:
                    avg_weights[wk] = np.mean(ws, axis=0)
            concat['weights'] = avg_weights
        else:
            concat['weights'] = None

        return dict(**avg_r, n_folds=n_valid, **concat)


def find_preproc(base_dir: str, animal: str, pos: str, n_cells: int):

    pattern = os.path.join(base_dir, '**', f'*{animal}*{pos}*',
                           'fm*', '*preproc.h5')
    hits = glob.glob(pattern, recursive=True)
    hits = [h for h in hits
            if 'boundary' not in h
            and not os.path.basename(h).startswith('sn')]

    if len(hits) == 0:
        return None
    if len(hits) == 1:
        return hits[0]

    # Disambiguate by cell count
    matching = []
    for h in hits:
        try:
            with h5py.File(h, 'r') as f:
                for key in ('norm_spikes', 'norm_dFF'):
                    if key in f:
                        nc = f[key].shape[0]
                        break
                else:
                    continue
            if nc == n_cells:
                matching.append(h)
        except Exception:
            pass

    pool = matching if matching else hits
    return sorted(pool)[-1]  # latest date


def run_all(pooled_path: str, base_dir: str, out_dir: str,
            use_dark: bool = False, only50: bool = False) -> list:

    os.makedirs(out_dir, exist_ok=True)
    decoder = EyeDecoder()
    all_results = []

    with h5py.File(pooled_path, 'r') as pool:
        for animal in pool.keys():
            if animal == 'uniref':
                continue
            if 'messentials' not in pool[animal]:
                continue
            ess = pool[animal]['messentials']

            for pos in sorted(ess.keys()):
                if not pos.startswith('pos'):
                    continue
                grp = ess[pos]

                if 'visual_area_id' not in grp or 'vfs_cell_pos' not in grp:
                    continue

                va_ids       = grp['visual_area_id'][:]
                vfs_cell_pos = grp['vfs_cell_pos'][:]
                n_cells      = len(va_ids)

                named_ids = {aid for aid in np.unique(va_ids) if aid in ID_TO_NAME}
                if len(named_ids) == 0:
                    print(f'  {animal}/{pos}: no named areas, skipping.')
                    continue

                preproc_path = find_preproc(base_dir, animal, pos, n_cells)
                if preproc_path is None:
                    print(f'  {animal}/{pos}: preproc not found, skipping.')
                    continue

                print(f'\n=== {animal}/{pos} ===')
                print(f'  Preproc: {preproc_path}')
                print(f'  Areas: {sorted(ID_TO_NAME[i] for i in named_ids)}')

                try:
                    data = decoder.load_data(preproc_path)
                except Exception as e:
                    print(f'  ERROR loading data: {e}')
                    continue

                if data['neural'].shape[0] != n_cells:
                    print(f'  Cell count mismatch: pooled={n_cells}, '
                          f'preproc={data["neural"].shape[0]}, skipping.')
                    continue

                if use_dark:
                    blocks, _ = decoder.get_all_dark_blocks(data)
                    cond_label = 'dark'
                else:
                    blocks, _ = decoder.get_all_light_blocks(data)
                    cond_label = 'light'

                if len(blocks) < 2:
                    print(f'  Only {len(blocks)} {cond_label} block(s), '
                          f'need ≥2 for k-fold, skipping.')
                    continue

                print(f'  {len(blocks)} {cond_label} blocks -> '
                      f'{len(blocks)}-fold CV')

                for area_id in sorted(named_ids):
                    area_name = ID_TO_NAME[area_id]
                    cell_mask = (va_ids == area_id)
                    n_area    = int(cell_mask.sum())

                    if n_area < MIN_CELLS:
                        print(f'  {area_name}: only {n_area} cells '
                              f'(< {MIN_CELLS}), skip.')
                        continue

                    if only50 and n_area > MIN_CELLS:
                        rng    = np.random.default_rng()
                        chosen = rng.choice(np.where(cell_mask)[0],
                                            size=MIN_CELLS, replace=False)
                        cell_mask_used            = np.zeros(n_cells, dtype=bool)
                        cell_mask_used[chosen]    = True
                        n_area_used               = MIN_CELLS
                    else:
                        cell_mask_used = cell_mask
                        n_area_used    = n_area

                    print(f'  Decoding {area_name} ({n_area_used} cells, '
                          f'{len(blocks)}-fold)...')
                    result = decoder.decode_kfold(data, blocks,
                                                  cell_mask=cell_mask_used)
                    if result is None:
                        print(f'    Not enough valid frames, skipped.')
                        continue

                    print(f'    eye:  r_theta={result["r_theta"]:.3f}  '
                          f'r_phi={result["r_phi"]:.3f}  '
                          f'r_X0={result["r_X0"]:.3f}  '
                          f'r_Y0={result["r_Y0"]:.3f}  '
                          f'({result["n_folds"]}-fold mean)')
                    print(f'    eye (rmse): '
                          f'rmse_theta={result["rmse_theta"]:.3f}  '
                          f'rmse_phi={result["rmse_phi"]:.3f}  '
                          f'rmse_X0={result["rmse_X0"]:.3f}  '
                          f'rmse_Y0={result["rmse_Y0"]:.3f}')
                    print(f'    head: r_pitch={result["r_pitch"]:.3f}  '
                          f'r_roll={result["r_roll"]:.3f}  '
                          f'r_yaw={result["r_yaw"]:.3f}')
                    print(f'    head (rmse): '
                          f'rmse_pitch={result["rmse_pitch"]:.3f}  '
                          f'rmse_roll={result["rmse_roll"]:.3f}  '
                          f'rmse_yaw={result["rmse_yaw"]:.3f}')

                    all_results.append(dict(
                        animal         = animal,
                        pos            = pos,
                        area           = area_name,
                        area_id        = int(area_id),
                        n_cells        = n_area_used,
                        n_cells_total  = n_cells,
                        preproc_path   = preproc_path,
                        n_blocks       = len(blocks),
                        n_folds        = int(result['n_folds']),
                        r_theta        = float(result['r_theta']),
                        r_phi          = float(result['r_phi']),
                        r_X0           = float(result['r_X0']),
                        r_Y0           = float(result['r_Y0']),
                        r_pitch        = float(result['r_pitch']),
                        r_roll         = float(result['r_roll']),
                        r_yaw          = float(result['r_yaw']),
                        rmse_theta     = float(result['rmse_theta']),
                        rmse_phi       = float(result['rmse_phi']),
                        rmse_X0        = float(result['rmse_X0']),
                        rmse_Y0        = float(result['rmse_Y0']),
                        rmse_pitch     = float(result['rmse_pitch']),
                        rmse_roll      = float(result['rmse_roll']),
                        rmse_yaw       = float(result['rmse_yaw']),
                        _arrays        = result,
                        _vfs_pos       = vfs_cell_pos[cell_mask_used],
                    ))

    return all_results



def save_results(all_results: list, out_dir: str,
                 name: str = 'decode_across_areas'):
    
    h5_path   = os.path.join(out_dir, f'{name}.h5')
    json_path = os.path.join(out_dir, f'{name}.json')

    json_records = []

    with h5py.File(h5_path, 'w') as fout:
        for rec in all_results:
            key     = f'{rec["animal"]}_{rec["pos"]}_{rec["area"]}'
            arrays  = rec['_arrays']
            vfs_pos = rec['_vfs_pos']
            grp     = fout.require_group(key)

            for arr_key in ('gt_theta', 'gt_phi', 'gt_X0', 'gt_Y0',
                            'pred_theta', 'pred_phi', 'pred_X0', 'pred_Y0',
                            'gt_longaxis', 'gt_shortaxis', 'gt_ellipse_phi',
                            'gt_pitch', 'gt_roll', 'gt_yaw',
                            'pred_pitch', 'pred_roll', 'pred_yaw',
                            'valid_test', 'valid_pitch_roll', 'valid_yaw'):
                grp.create_dataset(arr_key, data=arrays[arr_key],
                                   compression='gzip')

            w = arrays['weights']
            wgrp = grp.require_group('cell_weights')
            for wk, wv in w.items():
                wgrp.create_dataset(wk, data=wv)

            grp.create_dataset('vfs_cell_pos', data=vfs_pos)

            for sk in ('animal', 'pos', 'area', 'area_id', 'n_cells',
                       'n_cells_total', 'preproc_path', 'n_blocks', 'n_folds',
                       'r_theta', 'r_phi', 'r_X0', 'r_Y0',
                       'r_pitch', 'r_roll', 'r_yaw',
                       'rmse_theta', 'rmse_phi', 'rmse_X0', 'rmse_Y0',
                       'rmse_pitch', 'rmse_roll', 'rmse_yaw'):
                grp.attrs[sk] = rec[sk]

            json_records.append({k: rec[k] for k in rec if not k.startswith('_')})

    with open(json_path, 'w') as jf:
        json.dump(json_records, jf, indent=2)

    print(f'\nSaved HDF5: {h5_path}')
    print(f'Saved JSON: {json_path}')
    return h5_path, json_path


def _scatter_col(ax, x_pos, vals, color, label=None):

    vals = np.array(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return
    jitter = (np.random.rand(len(vals)) - 0.5) * 0.4
    ax.scatter(np.ones(len(vals)) * x_pos + jitter, vals,
               s=8, c=color, alpha=0.7, zorder=3, label=label)
    mn  = np.nanmean(vals)
    sem = np.nanstd(vals) / np.sqrt(len(vals))
    ax.hlines(mn, x_pos - 0.15, x_pos + 0.15, colors='k', linewidths=1.5, zorder=4)
    ax.vlines(x_pos, mn - sem, mn + sem, colors='k', linewidths=1.5, zorder=4)


def make_diagnostic_pdf(all_results: list, pdf_path: str) -> None:

    eye_vars   = ['r_theta', 'r_phi', 'r_X0', 'r_Y0']
    eye_labels = [r'$r_\theta$', r'$r_\phi$', r'$r_{X_0}$', r'$r_{Y_0}$']
    head_vars   = ['r_pitch', 'r_roll', 'r_yaw']
    head_labels = [r'$r_{pitch}$', r'$r_{roll}$', r'$r_{yaw}$']
    all_vars   = eye_vars + head_vars

    area_data = {a: {v: [] for v in all_vars} for a in REGION_ORDER}
    for rec in all_results:
        area = rec['area']
        if area not in area_data:
            continue
        for v in all_vars:
            val = rec.get(v, float('nan'))
            try:
                if np.isfinite(float(val)):
                    area_data[area][v].append(float(val))
            except (TypeError, ValueError):
                pass

    areas_present = [a for a in REGION_ORDER if any(area_data[a][v]
                     for v in all_vars)]
    legend_patches = [mpatches.Patch(color=COLORS.get(a, 'k'), label=a)
                      for a in areas_present]

    def _box_page(pdf, variables, labels, title):
        fig, axes = plt.subplots(1, len(variables),
                                 figsize=(3 * len(variables), 4), dpi=300)
        if len(variables) == 1:
            axes = [axes]
        for ax, var, vlabel in zip(axes, variables, labels):
            groups = []
            for xi, area in enumerate(areas_present):
                vals = area_data[area][var]
                groups.append(np.array(vals, dtype=float))
                _scatter_col(ax, xi, vals, color=COLORS.get(area, 'k'))
            ax.set_xticks(range(len(areas_present)))
            ax.set_xticklabels(areas_present, fontsize=7)
            ax.set_ylabel(vlabel, fontsize=8)
            ax.set_xlim(-0.6, len(areas_present) - 0.4)
            ax.axhline(0, color='0.7', lw=0.8, ls='--')
            valid_g = [g[np.isfinite(g)] for g in groups if np.isfinite(g).sum() > 1]

        fig.suptitle(title, fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    with PdfPages(pdf_path) as pdf:

        _box_page(pdf, eye_vars, eye_labels, 'Eye-decoding r by visual area')


        _box_page(pdf, head_vars, head_labels, 'Head-decoding r by visual area')


        fig, axes = plt.subplots(1, len(eye_vars),
                                 figsize=(3.5 * len(eye_vars), 4), dpi=300)
        for ax, var, vlabel in zip(axes, eye_vars, eye_labels):
            for area in REGION_ORDER:
                recs = [r for r in all_results if r['area'] == area]
                if not recs:
                    continue
                xs = [r['n_cells'] for r in recs]
                ys = [r.get(var, float('nan')) for r in recs]
                ax.scatter(xs, ys, color=COLORS.get(area, 'k'),
                           s=25, alpha=0.8, zorder=3)
            ax.axhline(0, color='0.7', lw=0.8, ls='--')
            ax.set_xlabel('N cells in area', fontsize=7)
            ax.set_ylabel(vlabel, fontsize=7)
        fig.suptitle('N cells vs eye-decoding r', fontsize=9)
        fig.legend(handles=legend_patches, fontsize=6, frameon=False,
                   loc='upper right')
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, len(head_vars),
                                 figsize=(3.5 * len(head_vars), 4), dpi=300)
        for ax, var, vlabel in zip(axes, head_vars, head_labels):
            for area in REGION_ORDER:
                recs = [r for r in all_results if r['area'] == area]
                if not recs:
                    continue
                xs = [r['n_cells'] for r in recs]
                ys = [r.get(var, float('nan')) for r in recs]
                ax.scatter(xs, ys, color=COLORS.get(area, 'k'),
                           s=25, alpha=0.8, zorder=3)
            ax.axhline(0, color='0.7', lw=0.8, ls='--')
            ax.set_xlabel('N cells in area', fontsize=7)
            ax.set_ylabel(vlabel, fontsize=7)
        fig.suptitle('N cells vs head-decoding r', fontsize=9)
        fig.legend(handles=legend_patches, fontsize=6, frameon=False,
                   loc='upper right')
        fig.tight_layout()
        pdf.savefig(fig, dpi=150)
        plt.close(fig)


        plot_specs = [
            ('gt_theta',  'pred_theta',  'valid_test',       r'θ (°)',     'rmse_theta'),
            ('gt_phi',    'pred_phi',    'valid_test',        r'φ (°)',     'rmse_phi'),
            ('gt_pitch',  'pred_pitch',  'valid_pitch_roll',  'pitch (°)', 'rmse_pitch'),
            ('gt_roll',   'pred_roll',   'valid_pitch_roll',  'roll (°)',  'rmse_roll'),
            ('gt_yaw',    'pred_yaw',    'valid_yaw',         'yaw (°)',   'rmse_yaw'),
        ]
        ncols = len(plot_specs)
        for rec in all_results:
            arrays = rec['_arrays']
            color  = COLORS.get(rec['area'], 'steelblue')
            fig, axes = plt.subplots(1, ncols, figsize=(3 * ncols, 3), dpi=120)
            for ax, (gk, pk, vk, xlabel, rmse_key) in zip(axes, plot_specs):
                gt   = arrays[gk]
                pred = arrays[pk]
                mask = arrays[vk]
                rmse_val = rec.get(rmse_key, float('nan'))
                if np.isfinite(gt[mask]).any() and mask.sum() > 1:
                    ax.scatter(gt[mask], pred[mask], s=1, alpha=0.3,
                               color=color, rasterized=True)
                    lo_v = min(np.nanmin(gt[mask]), np.nanmin(pred[mask]))
                    hi_v = max(np.nanmax(gt[mask]), np.nanmax(pred[mask]))
                    ax.plot([lo_v, hi_v], [lo_v, hi_v], 'k--', lw=0.8, alpha=0.5)
                ax.set_xlabel(f'measured {xlabel}', fontsize=6)
                ax.set_ylabel(f'decoded {xlabel}', fontsize=6)
                rmse_str = f'{rmse_val:.3f}' if np.isfinite(rmse_val) else 'n/a'
                ax.set_title(f'RMSE={rmse_str}', fontsize=7)
                ax.tick_params(labelsize=6)
            fig.suptitle(f'{rec["animal"]} {rec["pos"]} — {rec["area"]} '
                         f'(n={rec["n_cells"]})', fontsize=8)
            fig.tight_layout()
            pdf.savefig(fig, dpi=100)
            plt.close(fig)

    print(f'Saved PDF: {pdf_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Decode eye/head variables per visual area (k-fold CV).')
    parser.add_argument('--pooled',   default=DEFAULT_POOLED,
                        help='Path to pooled_*.h5')
    parser.add_argument('--base_dir', default=DEFAULT_BASE,
                        help='Root directory for freely-moving recordings')
    parser.add_argument('--out_dir',  default=DEFAULT_OUT_DIR,
                        help='Output directory for results')
    parser.add_argument('--dark', action='store_true', default=False,
                        help='Use dark periods instead of light periods')
    parser.add_argument('--only50', action='store_true', default=False,
                        help='Subsample areas with >50 cells to exactly 50 '
                             'for a fair cell-count comparison')
    args = parser.parse_args()

    name = 'decode_across_areas'
    if args.dark:
        name += '_dark'
    if args.only50:
        name += '_only50'

    print(f'Pooled dataset : {args.pooled}')
    print(f'Recording base : {args.base_dir}')
    print(f'Output dir     : {args.out_dir}')
    print(f'Condition      : {"dark" if args.dark else "light"}')
    print(f'Cell sampling  : {"subsample to 50" if args.only50 else "all cells"}')

    all_results = run_all(args.pooled, args.base_dir, args.out_dir,
                          use_dark=args.dark, only50=args.only50)

    if not all_results:
        print('No results produced.')
        return

    h5_path, json_path = save_results(all_results, args.out_dir, name=name)

    pdf_path = os.path.join(args.out_dir, f'{name}_diagnostics.pdf')
    make_diagnostic_pdf(all_results, pdf_path)

    print(f'\nDone. {len(all_results)} area-recording combinations decoded.')
    print(f'  HDF5 : {h5_path}')
    print(f'  JSON : {json_path}')
    print(f'  PDF  : {pdf_path}')


if __name__ == '__main__':

    main()
