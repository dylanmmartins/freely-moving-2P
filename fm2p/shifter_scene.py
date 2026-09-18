# -*- coding: utf-8 -*-
"""
fm2p/shifter_scene.py

Shifter fit on the reconstructed scene: finds the camera rotation that makes
the render best predict the population, staged from a constant calibration
offset up to an eye-position-driven shifter.

Modeled on nielllab/pytorchGLM (Parker et al. 2022), with three changes forced
by this dataset.  Their visual input is a measured worldcam movie, so a shifter
only decides where in a real image the eye points.  Ours is rendered FROM pose,
so shifter and scene are both functions of the same variables and a shifter
with enough capacity can manufacture pose structure that looks visual: every
gain here is therefore reported against a shuffled-target null and a
shuffled-input null, not against the unshifted model alone.  Their readout is a
per-cell map over every pixel, which is the regime that already failed in
rf_from_reconstruction, so the readout here is a fixed low-order DCT basis with
a per-cell weight vector.  And their shift is a 2D affine, which is a poor
approximation of a camera rotation at 120 deg FOV; the render is a pinhole
projection, so a rotation is an exact homography and is applied as one.

What the fit buys, beyond a better encoding model: the egocentric and
retinocentric renders differ by a rotation driven by measured eye angle, so the
fitted gain of azimuth on eye theta answers the ego-versus-retino question as
one number with a confidence interval instead of a contest between two
negative-R2 models.  Gain near zero means egocentric, gain matching the
retinocentric render's own Jacobian means retinocentric.

One result is baked into the design and is worth knowing before reading any
output: a CONSTANT misaim is not identifiable.  A rotation applied identically
on every frame translates every receptive field by the same amount, and a
static linear readout absorbs it for free.  Checked on synthetic cells: a true
+8 deg offset leaves out-of-fold R2 unchanged and the fitted offset lands
anywhere in the search range.  What IS identifiable is the part of the rotation
that varies over time, because no static field can absorb that, and the eye
gain is exactly that part.  So the ladder's constant rung is a control, not a
measurement, and a fixed socket-angle error cannot be found this way; an eye or
head angle GAIN error can.

Read the recovery arms before believing any fitted gain.  recovery_gain_injected
rotates the scene by a known gain and asks whether the fit moves with it, which
measures how precisely THIS recording can locate a gain at all.
recovery_synthetic does the same against synthetic cells with known receptive
fields, separating a broken fitter from an uninformative dataset: on those,
a gain of 1.0 comes back as 0.94 even where R2 is only 0.055.

Classes
-------
ConstantShifter
    Single rotation shared by every frame: a degeneracy control, not a
    measurement, since a static readout absorbs a constant rotation.
LinearShifter
    Rotation linear in eye and head channels: calibration plus eye gain.
MLPShifter
    Niell lab architecture, one hidden layer; off by default.
_R2Accum
    Pools residual and total sums over folds, as cv_ridge does.

Functions
---------
_device
    Torch device, preferring CUDA when it is actually there.
rotation_matrices
    Camera rotations from azimuth, elevation and cyclotorsion in degrees.
sampling_grid
    grid_sample grid taking a cropped, rotated view back to source pixels.
warp_scene
    Apply per-frame rotations to the rendered movie.
dct_basis
    Low-order 2D DCT basis over the cropped field.
make_shifter
    Build one shifter by name.
needed_frames
    Frames that must be warped: the used ones plus every lag source.
trim_lag_boundary
    Drop used frames whose lag sources fall off the ends of the recording.
_lag_index_map
    Row positions of each lagged source frame within the warped-frame array.
ridge_solve
    Differentiable multi-output ridge.
_coefs
    Warp the scene by the shifter's rotation and project onto the basis.
_design
    Lagged design matrix, z-scored on ref_rows so one lam fits all columns.
_fit_eval
    Ridge on tr_rows, prediction on te_rows.
_pick_lam
    Penalty with the lowest inner-validation error, shared over cells.
_inner_split
    Split one training fold into inner-train and inner-validation blocks.
_rows
    Positions within the used-frame array for an absolute frame mask.
_empirical_gain
    Least-squares slope of each rotation output on each raw input channel.
grid_search_constant
    Exhaustive search over a constant rotation, scored on inner validation.
fit_shifter_fold
    Adam fit of one shifter on one training fold, readout profiled out.
_test_r2
    Freeze the shifter, refit the readout on the whole fold, score test.
run_ladder
    M0 through M2 (optionally M3) with pooled out-of-fold R2.
_median_gain
    Across-fold median of the fitted gain matrix, NaN if the model is absent.
null_shuffled_targets
    Refit everything against circularly shifted rates.
null_shuffled_inputs
    Refit with time-scrambled shifter inputs.
_rotate_scene
    Rotate the rendered movie in place of the renderer, same size and FOV.
recovery_gain_injected
    Fit a known eye gain injected into the real scene.
recovery_synthetic
    Fit a known eye gain against synthetic cells with known fields.
_load_session
    Load rates, both renders and the shifter channels onto a shared frame set.
_build_ctx
    Everything the fitting functions need for one render, on the device.
shifter_scene
    Top level driver: load, fit both renders, control, save, plot.
_compare_gains
    Azimuth-on-theta gain in each render, and what separates them.
_perm_p
    One-sided permutation p, floored at 1 / (n + 1).
print_verdict
    Plain-language readout of what the arms decided.
_sanitize
    Recursively drop Nones and unsaveable values so write_h5 can take it.
_save
    Write the whole result tree to h5.
_save_json
    Scalar summary, for grepping across sessions without opening the h5.
make_summary_pdf
    Figures for the whole run.
main
    CLI entry point.


DMM, September 2026
"""

if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

import argparse
import json
import os
import time

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.files import write_h5
from .rf_from_reconstruction import RET_FOV_DEG
from .ffNLE_scene import (load_scene, load_behavior_channels, block_folds,
                          _triangle_smooth, _fmt, SPEED_THRESH_CM_S)

# Scene-to-response offsets in 2P frames at 7.5 Hz, as in ffNLE_scene.
DEFAULT_LAGS = (0, 1, 2, 3)

# Lags used while the shifter itself is being fit.  pytorchGLM trains its
# shifter with a single lag and freezes it before adding the rest; fewer lags
# here means fewer readout parameters competing with the 3-12 that matter.
SHIFTER_LAGS = (0, 1)

# Ridge penalties, swept once per fold on the unshifted model and then held
# fixed so that only the rotation changes between models.
LAM_GRID = np.logspace(-1, 5, 13)

# Cropped field the readout sees, degrees.  Smaller than the rendered 120 deg
# so a rotation never samples outside the render and never pulls a
# shift-dependent black wedge into the input, which would be a pose signal
# dressed as a visual one.
DEFAULT_CROP_DEG = 60.0

# Output resolution of the crop.  32 px over 60 deg is about 1.9 deg per pixel,
# finer than any structure 25 DCT components can carry.
DEFAULT_RES_OUT = 32


def _device(name=None):
    """ Torch device, preferring CUDA when it is actually there. """

    if name:
        return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rotation_matrices(daz, del_deg, droll):
    """ Camera rotations from azimuth, elevation and cyclotorsion.

    Camera frame matches retinal_pixel_angles: x right, y up, z forward.  The
    returned matrix maps a direction in the CORRECTED camera frame to the same
    direction expressed in the ORIGINAL render frame, which is what the
    resampler needs.  Positive daz rotates the camera to the animal's right,
    positive del_deg rotates it up, positive droll is counter-clockwise
    cyclotorsion seen from behind the eye.

    Parameters
    ----------
    daz, del_deg, droll : (B,) torch.Tensor
        Degrees.

    Returns
    -------
    R : (B, 3, 3) torch.Tensor
    """

    a = torch.deg2rad(daz)
    e = torch.deg2rad(del_deg)
    r = torch.deg2rad(droll)
    ca, sa = torch.cos(a), torch.sin(a)
    ce, se = torch.cos(e), torch.sin(e)
    cr, sr = torch.cos(r), torch.sin(r)
    z = torch.zeros_like(ca)
    o = torch.ones_like(ca)

    # Ry: sends forward toward +x, so positive azimuth looks right.
    Ry = torch.stack([ca, z, sa, z, o, z, -sa, z, ca], dim=-1).reshape(-1, 3, 3)
    # Rx: sends forward toward +y, so positive elevation looks up.
    Rx = torch.stack([o, z, z, z, ce, se, z, -se, ce], dim=-1).reshape(-1, 3, 3)
    Rz = torch.stack([cr, -sr, z, sr, cr, z, z, z, o], dim=-1).reshape(-1, 3, 3)
    return Ry @ Rx @ Rz


def sampling_grid(R, res_in, fov_in_deg, res_out, fov_out_deg):
    """ grid_sample grid taking the cropped, rotated view back to source pixels.

    The render is a pinhole projection, so a camera rotation is an exact
    homography on the image plane: no small-angle approximation, which matters
    because at 120 deg FOV an affine shift is badly wrong at the edges and the
    edges are where the ablations said the little structure there is sits.

    Parameters
    ----------
    R : (B, 3, 3) torch.Tensor
        From rotation_matrices.
    res_in, res_out : int
        Source and output side lengths in pixels.
    fov_in_deg, fov_out_deg : float
        Source and output fields of view.

    Returns
    -------
    grid : (B, res_out, res_out, 2) torch.Tensor
        Normalized sampling coordinates for align_corners=False.
    """

    dev, dt = R.device, R.dtype
    f_out = (res_out / 2.0) / np.tan(np.radians(fov_out_deg / 2.0))
    f_in = (res_in / 2.0) / np.tan(np.radians(fov_in_deg / 2.0))

    u = torch.arange(res_out, device=dev, dtype=dt) + 0.5 - res_out / 2.0
    v = torch.arange(res_out, device=dev, dtype=dt) + 0.5 - res_out / 2.0
    V, U = torch.meshgrid(v, u, indexing='ij')

    # Ray direction per output pixel, in the corrected camera frame.
    d = torch.stack([U, -V, torch.full_like(U, float(f_out))], dim=-1)
    d_src = torch.einsum('bij,hwj->bhwi', R, d)
    x, y, zc = d_src[..., 0], d_src[..., 1], d_src[..., 2]

    # Rays that ended up behind the camera have no source pixel; park them far
    # outside the grid so padding handles them instead of aliasing to a
    # plausible-looking pixel.
    zc = torch.where(zc.abs() < 1e-6, torch.full_like(zc, 1e-6), zc)
    u_src = f_in * x / zc
    v_src = -f_in * y / zc
    gx = 2.0 * (u_src + res_in / 2.0) / res_in - 1.0
    gy = 2.0 * (v_src + res_in / 2.0) / res_in - 1.0
    bad = zc <= 0
    gx = torch.where(bad, torch.full_like(gx, 5.0), gx)
    gy = torch.where(bad, torch.full_like(gy, 5.0), gy)
    return torch.stack([gx, gy], dim=-1)


def warp_scene(S, R, res_out, fov_out_deg, fov_in_deg=RET_FOV_DEG,
               chunk=None):
    """ Apply per-frame rotations to the rendered movie and crop.

    Parameters
    ----------
    S : (T, res_in, res_in) torch.Tensor
    R : (B, 3, 3) torch.Tensor
        B is either T, or 1 for a rotation shared by every frame.
    res_out : int
    fov_out_deg, fov_in_deg : float
    chunk : int or None
        Frames per grid_sample call; None does the lot in one.

    Returns
    -------
    (T, res_out, res_out) torch.Tensor
    """

    T, res_in, _ = S.shape
    shared = (R.shape[0] == 1)
    if chunk is None or shared:
        grid = sampling_grid(R, res_in, fov_in_deg, res_out, fov_out_deg)
        if shared:
            grid = grid.expand(T, -1, -1, -1)
        out = F.grid_sample(S.unsqueeze(1), grid, mode='bilinear',
                            padding_mode='border', align_corners=False)
        return out.squeeze(1)

    parts = []
    for i in range(0, T, int(chunk)):
        j = min(i + int(chunk), T)
        grid = sampling_grid(R[i:j], res_in, fov_in_deg, res_out, fov_out_deg)
        parts.append(F.grid_sample(S[i:j].unsqueeze(1), grid, mode='bilinear',
                                   padding_mode='border',
                                   align_corners=False).squeeze(1))
    return torch.cat(parts, dim=0)


def dct_basis(res, n_dct):
    """ Orthonormal low-order 2D DCT-II basis over a res x res field.

    A fixed basis, not PCA: PCA refit at every candidate rotation would change
    the feature space along with the shift and confound the comparison the
    whole thing rests on.

    Parameters
    ----------
    res : int
    n_dct : int
        Components kept per axis, so the basis has n_dct ** 2 rows.

    Returns
    -------
    (n_dct ** 2, res * res) float32 ndarray
    """

    n = np.arange(res) + 0.5
    rows = []
    for k in range(n_dct):
        c = np.cos(np.pi * k * n / res)
        c = c / np.linalg.norm(c)
        rows.append(c)
    one_d = np.stack(rows, axis=0)

    basis = np.einsum('iy,jx->ijyx', one_d, one_d).reshape(n_dct * n_dct, -1)
    return basis.astype(np.float32)


class ConstantShifter(nn.Module):
    """ One rotation for every frame, 3 params.

    Not a calibration measurement, despite looking like one.  A rotation that
    is identical on every frame translates every receptive field by the same
    amount, and a static linear readout absorbs that for free: on synthetic
    cells with a true +8 deg misaim this rung leaves out-of-fold R2 unchanged
    and its fitted offset lands anywhere.  Kept as a control -- a real gain
    here means something is leaking, most likely at the crop edge -- and as an
    initializer for the rungs above.
    """

    n_params = 3

    def __init__(self, n_in, max_shift_deg):
        """ One learnable (dx, dy, rotation) shared by every frame. """

        super(ConstantShifter, self).__init__()
        self.max_shift_deg = float(max_shift_deg)
        self.raw = nn.Parameter(torch.zeros(3))

    def forward(self, shift_in):
        """ Rotation in degrees, (1, 3), broadcast over frames by warp_scene. """

        return self.max_shift_deg * torch.tanh(self.raw).unsqueeze(0)


class LinearShifter(nn.Module):
    """ Rotation linear in the shifter channels: calibration plus eye gain.

    This is the model that answers the ego-versus-retino question, since the
    coefficient of azimuth on eye theta is exactly the gain that separates the
    two renders.  Weights are unconstrained and only the total is squashed, so
    the reported gain is not pushed toward zero by the constraint.
    """

    def __init__(self, n_in, max_shift_deg):
        """ Linear map from eye/head inputs to shift, zero-initialized. """

        super(LinearShifter, self).__init__()
        self.max_shift_deg = float(max_shift_deg)
        self.lin = nn.Linear(n_in, 3)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        self.n_params = 3 * (n_in + 1)

    def forward(self, shift_in):
        """ Rotation in degrees, (T, 3). """

        return self.max_shift_deg * torch.tanh(self.lin(shift_in))


class MLPShifter(nn.Module):
    """ Niell lab shifter: Linear(n_in, hidden), Softplus, Linear(hidden, 3). """

    def __init__(self, n_in, max_shift_deg, hidden=20):
        """ One-hidden-layer map from eye/head inputs to shift, output zero-initialized. """

        super(MLPShifter, self).__init__()
        self.max_shift_deg = float(max_shift_deg)
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.Softplus(),
                                 nn.Linear(hidden, 3))
        nn.init.uniform_(self.net[0].weight, a=-1e-3, b=1e-3)
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)
        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(self, shift_in):
        """ Rotation in degrees, (T, 3). """

        return self.max_shift_deg * torch.tanh(self.net(shift_in))


def make_shifter(kind, n_in, max_shift_deg, device, hidden=20):
    """ Build one shifter by name. """

    if kind == 'constant':
        m = ConstantShifter(n_in, max_shift_deg)
    elif kind == 'linear':
        m = LinearShifter(n_in, max_shift_deg)
    elif kind == 'mlp':
        m = MLPShifter(n_in, max_shift_deg, hidden=hidden)
    else:
        raise ValueError('Unknown shifter {!r}.'.format(kind))
    return m.to(device)


def needed_frames(use, lags):
    """ Frames that have to be warped: the used ones plus every lag source.

    A lag source does not have to be a model sample itself, only renderable.
    Requiring otherwise (which is what closing use under its own lags would do)
    cascades and eats the recording, since frame t-1 is excluded exactly
    because ITS lag source is out of range.

    Parameters
    ----------
    use : (n_2p,) bool
    lags : sequence of int

    Returns
    -------
    (n_2p,) bool
    """

    need = use.copy()
    for L in lags:
        src = np.roll(use, -int(L))
        if L > 0:
            src[-int(L):] = False
        elif L < 0:
            src[:-int(L)] = False
        need |= src
    return need


def trim_lag_boundary(use, lags):
    """ Drop used frames whose lag sources fall off either end of the recording.

    The only genuine constraint on lags once sources are allowed to sit outside
    use: costs a handful of frames instead of the whole session.
    """

    n = len(use)
    idx = np.arange(n)
    ok = np.ones(n, dtype=bool)
    for L in lags:
        ok &= (idx - int(L) >= 0) & (idx - int(L) < n)
    return use & ok


def _lag_index_map(use, need, lags, device):
    """ Row positions of each lagged source frame within the warped-frame array.

    Parameters
    ----------
    use : (n_2p,) bool
        Model samples, in order.
    need : (n_2p,) bool
        Frames actually warped; a superset of use from needed_frames.
    lags : sequence of int
    device : torch.device

    Returns
    -------
    list of (n_used,) torch.LongTensor
        Indices into the need-ordered array, one per lag, aligned to use order.
    """

    use_idx = np.flatnonzero(use)
    need_idx = np.flatnonzero(need)
    pos = np.full(len(use), -1, dtype=np.int64)
    pos[need_idx] = np.arange(len(need_idx))
    rows = []
    for L in lags:
        r = pos[use_idx - int(L)]
        if np.any(r < 0):
            raise RuntimeError('Lag {} escapes the warped-frame set.'.format(L))
        rows.append(torch.as_tensor(r, device=device))
    return rows


def ridge_solve(X, Y, lam):
    """ Multi-output ridge in closed form, differentiable through the solve.

    Profiling the readout out like this leaves the optimizer only the 3 to 143
    shifter parameters, which is far better conditioned than pytorchGLM's joint
    SGD over shifter and per-cell weights when the stimulus carries on the order
    of ten independent samples.

    Parameters
    ----------
    X : (n, p) torch.Tensor
    Y : (n, C) torch.Tensor
    lam : float

    Returns
    -------
    B : (p, C) torch.Tensor
    xm : (p,) torch.Tensor
    ym : (C,) torch.Tensor
    """

    xm = X.mean(dim=0)
    ym = Y.mean(dim=0)
    Xc = X - xm
    A = Xc.T @ Xc
    A = A + lam * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    B = torch.linalg.solve(A, Xc.T @ (Y - ym))
    return B, xm, ym


class _R2Accum:
    """ Pools residual and total sums over folds, as cv_ridge does. """

    def __init__(self, n_cells, device):
        """ Running sums for per-cell R^2 across batches. """

        self.ss_res = torch.zeros(n_cells, dtype=torch.float64, device=device)
        self.sy = torch.zeros(n_cells, dtype=torch.float64, device=device)
        self.syy = torch.zeros(n_cells, dtype=torch.float64, device=device)
        self.n = 0
        self.folds = []

    def add(self, Yte, pred):
        """ Fold in one test block. """

        d = (Yte - pred).double()
        self.ss_res += (d ** 2).sum(dim=0)
        self.sy += Yte.double().sum(dim=0)
        self.syy += (Yte.double() ** 2).sum(dim=0)
        self.n += Yte.shape[0]
        ssr = (d ** 2).sum(dim=0)
        sst = ((Yte - Yte.mean(dim=0)).double() ** 2).sum(dim=0)
        self.folds.append((1.0 - ssr / torch.clamp(sst, min=1e-12)).cpu().numpy())

    def result(self):
        """ Pooled per-cell R2 and the per-fold matrix. """

        sst = self.syy - self.sy ** 2 / max(self.n, 1)
        r2 = 1.0 - self.ss_res / torch.where(sst > 0, sst, torch.full_like(sst, np.nan))
        return r2.cpu().numpy(), np.asarray(self.folds)


def _coefs(shifter, ctx, shift_in=None):
    """ Warp the scene by the shifter's rotation and project onto the basis. """

    si = ctx['shift_in'] if shift_in is None else shift_in
    dxy = shifter(si)
    R = rotation_matrices(dxy[:, 0], dxy[:, 1], dxy[:, 2])
    W = warp_scene(ctx['S'], R, ctx['res_out'], ctx['crop_deg'],
                   chunk=ctx.get('chunk'))
    return W.reshape(W.shape[0], -1) @ ctx['basis'].T


def _design(coefs, lag_rows, ref_rows):
    """ Lagged design matrix, z-scored on ref_rows so one lam fits all columns. """

    X = torch.cat([coefs[r] for r in lag_rows], dim=1)
    mu = X[ref_rows].mean(dim=0)
    sd = X[ref_rows].std(dim=0)
    return (X - mu) / torch.clamp(sd, min=1e-6)


def _fit_eval(X, Y, tr_rows, te_rows, lam):
    """ Ridge on tr_rows, prediction on te_rows. """

    B, xm, ym = ridge_solve(X[tr_rows], Y[tr_rows], lam)
    return (X[te_rows] - xm) @ B + ym


def _pick_lam(X, Y, inner_tr, inner_va, lam_grid):
    """ Penalty with the lowest inner-validation error, shared over cells. """

    best, best_lam = np.inf, float(lam_grid[0])
    for lam in lam_grid:
        pred = _fit_eval(X, Y, inner_tr, inner_va, float(lam))
        err = float(torch.mean((Y[inner_va] - pred) ** 2))
        if err < best:
            best, best_lam = err, float(lam)
    return best_lam


def _inner_split(twopT, use, train_mask, fold_block_s, embargo_s, seed):
    """ Split one training fold into inner-train and inner-validation blocks.

    Reuses block_folds so the inner split carries the same embargo as the outer
    one: a shifter tuned against a validation block sitting inside its training
    set's autocorrelation would be tuned against nothing.
    """

    _, tr_in, va_in = block_folds(twopT, use & train_mask, n_folds=4,
                                  block_s=fold_block_s, embargo_s=embargo_s,
                                  seed=seed)
    return tr_in[0], va_in[0]


def _rows(mask, use, device):
    """ Positions within the used-frame array for an absolute frame mask. """

    use_idx = np.flatnonzero(use)
    sel = np.flatnonzero(mask[use_idx])
    return torch.as_tensor(sel, device=device, dtype=torch.long)


def _empirical_gain(shifter, shift_in, chans_deg):
    """ Least-squares slope of each rotation output on each raw input channel.

    Reported in degrees of rotation per degree of input rather than per unit of
    the z-scored channel the network actually sees, so the azimuth-on-theta
    entry can be read straight off against the ego and retino predictions.

    Parameters
    ----------
    shifter : nn.Module
    shift_in : (T, n_in) torch.Tensor
        z-scored channels, as fed to the shifter.
    chans_deg : (T, n_in) ndarray
        Same channels in degrees.

    Returns
    -------
    (3, n_in) ndarray
    """

    with torch.no_grad():
        out = shifter(shift_in)
        if out.shape[0] == 1:
            out = out.expand(shift_in.shape[0], -1)
        out = out.detach().cpu().numpy()

    A = np.column_stack([np.ones(len(chans_deg)), np.asarray(chans_deg)])
    good = np.isfinite(A).all(axis=1) & np.isfinite(out).all(axis=1)
    if good.sum() < 10:
        return np.full((3, A.shape[1] - 1), np.nan)
    coef, _, _, _ = np.linalg.lstsq(A[good], out[good], rcond=None)
    return coef[1:].T


def grid_search_constant(ctx, inner_tr, inner_va, lam, az_grid, el_grid,
                         roll_grid):
    """ Exhaustive search over a constant rotation, scored on inner validation.

    Cheap enough to be exhaustive, and the surface is worth more than the
    argmax: a flat surface says the recording cannot locate a shift at all, and
    no amount of gradient descent will change that.

    Returns
    -------
    best : (3,) ndarray
        Degrees, azimuth, elevation, roll.
    surface : (n_az, n_el, n_roll) ndarray
        Inner-validation mean squared error.
    """

    dev = ctx['device']
    Y = ctx['Y']
    surface = np.full((len(az_grid), len(el_grid), len(roll_grid)), np.nan)

    with torch.no_grad():
        for ia, a in enumerate(az_grid):
            for ie, e in enumerate(el_grid):
                for ir, r in enumerate(roll_grid):
                    par = torch.tensor([[float(a), float(e), float(r)]],
                                       device=dev, dtype=torch.float32)
                    R = rotation_matrices(par[:, 0], par[:, 1], par[:, 2])
                    W = warp_scene(ctx['S'], R, ctx['res_out'], ctx['crop_deg'],
                                   chunk=ctx.get('chunk'))
                    C = W.reshape(W.shape[0], -1) @ ctx['basis'].T
                    X = _design(C, ctx['lag_rows_fit'], inner_tr)
                    pred = _fit_eval(X, Y, inner_tr, inner_va, lam)
                    surface[ia, ie, ir] = float(
                        torch.mean((Y[inner_va] - pred) ** 2))

    ia, ie, ir = np.unravel_index(np.nanargmin(surface), surface.shape)
    best = np.array([az_grid[ia], el_grid[ie], roll_grid[ir]], dtype=float)
    return best, surface


def fit_shifter_fold(kind, ctx, inner_tr, inner_va, lam, n_steps=300, lr=0.05,
                     seed=0, init_const=None, verbose=False):
    """ Adam fit of one shifter on one training fold, readout profiled out.

    The loss is inner-validation error after solving the readout in closed form
    on inner-train, so the shifter is selected on data the readout never saw.
    Best-on-validation parameters are kept rather than the last step's.

    Returns
    -------
    shifter : nn.Module
    trace : (n_steps,) ndarray
        Inner-validation MSE per step.
    """

    torch.manual_seed(int(seed))
    sh = make_shifter(kind, ctx['n_in'], ctx['max_shift'], ctx['device'],
                      hidden=ctx.get('hidden', 20))

    # Start the eye-driven shifters from the constant fit: it costs nothing and
    # keeps them out of the flat region tanh has far from zero.
    if init_const is not None:
        raw = np.arctanh(np.clip(np.asarray(init_const, dtype=float)
                                 / ctx['max_shift'], -0.99, 0.99))
        with torch.no_grad():
            if isinstance(sh, ConstantShifter):
                sh.raw.copy_(torch.as_tensor(raw, dtype=torch.float32,
                                             device=ctx['device']))
            elif isinstance(sh, LinearShifter):
                sh.lin.bias.copy_(torch.as_tensor(raw, dtype=torch.float32,
                                                  device=ctx['device']))
            elif isinstance(sh, MLPShifter):
                sh.net[2].bias.copy_(torch.as_tensor(raw, dtype=torch.float32,
                                                     device=ctx['device']))

    opt = torch.optim.Adam(sh.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(n_steps // 4, 1),
                                            gamma=0.5)
    Y = ctx['Y']
    trace = np.full(int(n_steps), np.nan)
    best_err, best_state = np.inf, None

    for step in range(int(n_steps)):
        opt.zero_grad()
        C = _coefs(sh, ctx)
        X = _design(C, ctx['lag_rows_fit'], inner_tr)
        pred = _fit_eval(X, Y, inner_tr, inner_va, lam)
        loss = torch.mean((Y[inner_va] - pred) ** 2)
        loss.backward()
        opt.step()
        sched.step()

        trace[step] = float(loss)
        if trace[step] < best_err:
            best_err = trace[step]
            best_state = {k: v.detach().clone() for k, v in sh.state_dict().items()}
        if verbose and (step % 50 == 0):
            print('      step {:3d}  inner val MSE {}...'.format(
                step, _fmt(trace[step], 5)))

    if best_state is not None:
        sh.load_state_dict(best_state)
    return sh, trace


def _test_r2(sh, ctx, train_rows, test_rows, lam_final, acc):
    """ Freeze the shifter, refit the readout on the whole fold, score test. """

    with torch.no_grad():
        C = _coefs(sh, ctx)
        X = _design(C, ctx['lag_rows'], train_rows)
        pred = _fit_eval(X, ctx['Y'], train_rows, test_rows, lam_final)
        acc.add(ctx['Y'][test_rows], pred)
    return pred


def run_ladder(ctx, twopT, use, train_masks, test_masks,
               models=('none', 'constant', 'linear'), az_grid=None,
               el_grid=None, roll_grid=None, n_steps=300, lr=0.05,
               fold_block_s=120.0, embargo_s=40.0, seed=0, verbose=True):
    """ M0 through M2 (optionally M3) with pooled out-of-fold R2.

    Nested by construction, so the gain at each rung is the answer to one
    question: M1 over M0 is whether the render is mis-aimed, M2 over M1 is
    whether eye position moves the image the way the render says it should.

    Everything is fit inside the training fold, with the shifter selected on an
    inner validation block that carries the same embargo as the outer split.

    Returns
    -------
    dict of model name to result dict
    """

    dev = ctx['device']
    az_grid = np.arange(-20, 20.1, 5.0) if az_grid is None else az_grid
    el_grid = np.arange(-20, 20.1, 5.0) if el_grid is None else el_grid
    roll_grid = np.array([-10.0, 0.0, 10.0]) if roll_grid is None else roll_grid

    out = {m: {'acc': _R2Accum(ctx['Y'].shape[1], dev), 'params': [],
               'gain': [], 'trace': [], 'lam': []} for m in models}
    surfaces = []

    for k, (tr_mask, te_mask) in enumerate(zip(train_masks, test_masks)):
        if tr_mask.sum() < 100 or te_mask.sum() < 20:
            continue
        itr_mask, iva_mask = _inner_split(twopT, use, tr_mask, fold_block_s,
                                          embargo_s, seed)
        train_rows = _rows(tr_mask, use, dev)
        test_rows = _rows(te_mask, use, dev)
        itr = _rows(itr_mask, use, dev)
        iva = _rows(iva_mask, use, dev)
        if len(itr) < 50 or len(iva) < 20:
            if verbose:
                print('    Fold {}: inner split too small, skipping...'.format(k + 1))
            continue

        # One penalty per fold, picked on the unshifted model and then held
        # fixed: if lam moved with the rotation the ladder would be comparing
        # two things at once.
        sh0 = make_shifter('constant', ctx['n_in'], ctx['max_shift'], dev)
        with torch.no_grad():
            C0 = _coefs(sh0, ctx)
            lam_fit = _pick_lam(_design(C0, ctx['lag_rows_fit'], itr), ctx['Y'],
                                itr, iva, LAM_GRID)
            lam_final = _pick_lam(_design(C0, ctx['lag_rows'], train_rows),
                                  ctx['Y'], itr, iva, LAM_GRID)

        best_const = None
        for m in models:
            t0 = time.time()
            if m == 'none':
                sh = sh0
                trace = np.array([])
            elif m == 'constant':
                best_const, surf = grid_search_constant(ctx, itr, iva, lam_fit,
                                                        az_grid, el_grid, roll_grid)
                surfaces.append(surf)
                sh, trace = fit_shifter_fold('constant', ctx, itr, iva, lam_fit,
                                             n_steps=max(n_steps // 3, 50), lr=lr,
                                             seed=seed + k, init_const=best_const)
                with torch.no_grad():
                    best_const = sh(ctx['shift_in']).detach().cpu().numpy().ravel()[:3]
            else:
                sh, trace = fit_shifter_fold(m, ctx, itr, iva, lam_fit,
                                             n_steps=n_steps, lr=lr,
                                             seed=seed + k, init_const=best_const)

            _test_r2(sh, ctx, train_rows, test_rows, lam_final, out[m]['acc'])
            with torch.no_grad():
                dxy = sh(ctx['shift_in']).detach().cpu().numpy()
            out[m]['params'].append(dxy.mean(axis=0))
            out[m]['gain'].append(_empirical_gain(sh, ctx['shift_in'],
                                                  ctx['chans_deg']))
            out[m]['trace'].append(trace)
            out[m]['lam'].append([lam_fit, lam_final])
            if verbose:
                r2_so_far = out[m]['acc'].folds[-1]
                print('    Fold {} {:9s} median R2 {}  shift az {} el {} roll {}'
                      '  ({:.0f}s)...'.format(
                          k + 1, m, _fmt(np.nanmedian(r2_so_far)),
                          _fmt(dxy.mean(axis=0)[0], 1), _fmt(dxy.mean(axis=0)[1], 1),
                          _fmt(dxy.mean(axis=0)[2], 1), time.time() - t0))

    res = {}
    for m in models:
        r2, folds = out[m]['acc'].result()
        res[m] = {
            'r2': r2,
            'r2_folds': folds,
            'median_r2': float(np.nanmedian(r2)),
            'shift_deg': np.asarray(out[m]['params'], dtype=float),
            'gain_deg_per_deg': np.asarray(out[m]['gain'], dtype=float),
            'lam': np.asarray(out[m]['lam'], dtype=float),
        }
    if surfaces:
        res['constant']['surface'] = np.asarray(surfaces)
        res['constant']['az_grid'] = np.asarray(az_grid, dtype=float)
        res['constant']['el_grid'] = np.asarray(el_grid, dtype=float)
        res['constant']['roll_grid'] = np.asarray(roll_grid, dtype=float)
    return res


def _median_gain(res, model='linear'):
    """ Across-fold median of the fitted gain matrix, NaN if the model is absent. """

    g = res.get(model, {}).get('gain_deg_per_deg', None)
    if g is None or not len(g):
        return None
    return np.nanmedian(np.asarray(g), axis=0)


def null_shuffled_targets(ctx, twopT, use, train_masks, test_masks, models,
                          n_null=8, min_shift=200, seed=0, verbose=True,
                          **kw):
    """ Refit the whole ladder against circularly shifted rates.

    This is the null that matters.  A shifter has three to twelve free
    parameters and a scene that is itself a function of pose, so it will always
    improve the training fit; the question is whether it improves the held-out
    fit by more than it does when the rates no longer line up with anything.
    Shifts are taken on the full-length trace so the calcium autocorrelation
    survives.

    Returns
    -------
    dict of model name to (n_null,) array of median R2
    """

    n_2p = ctx['Y_full'].shape[0]
    rng = np.random.default_rng(seed)
    shifts = rng.integers(min_shift, max(n_2p - min_shift, min_shift + 1),
                          size=int(n_null))
    out = {m: [] for m in models}
    for j, sh in enumerate(shifts):
        idx = (ctx['use_idx'] - int(sh)) % n_2p
        ctx_j = dict(ctx)
        ctx_j['Y'] = ctx['Y_full'][torch.as_tensor(idx, device=ctx['device'])]
        r = run_ladder(ctx_j, twopT, use, train_masks, test_masks,
                       models=models, seed=seed + j, verbose=False, **kw)
        for m in models:
            out[m].append(r[m]['median_r2'])
        if verbose:
            print('    null {}/{}: {}'.format(
                j + 1, len(shifts),
                '  '.join(['{} {}'.format(m, _fmt(out[m][-1])) for m in models])))
    return {m: np.asarray(v, dtype=float) for m, v in out.items()}


def null_shuffled_inputs(ctx, twopT, use, train_masks, test_masks, models,
                         n_null=8, min_shift=200, seed=0, verbose=True, **kw):
    """ Refit with time-scrambled shifter inputs.

    Separates the eye and head channels mattering from three extra free time
    series mattering: the shifter keeps its capacity, the inputs keep their
    marginal statistics, only their alignment to the scene is destroyed.
    """

    n_used = ctx['shift_in'].shape[0]
    rng = np.random.default_rng(seed)
    shifts = rng.integers(min_shift, max(n_used - min_shift, min_shift + 1),
                          size=int(n_null))
    out = {m: [] for m in models}
    for j, sh in enumerate(shifts):
        ctx_j = dict(ctx)
        ctx_j['shift_in'] = torch.roll(ctx['shift_in'], int(sh), dims=0)
        ctx_j['chans_deg'] = np.roll(ctx['chans_deg'], int(sh), axis=0)
        r = run_ladder(ctx_j, twopT, use, train_masks, test_masks,
                       models=models, seed=seed + j, verbose=False, **kw)
        for m in models:
            out[m].append(r[m]['median_r2'])
        if verbose:
            print('    null {}/{}: {}'.format(
                j + 1, len(shifts),
                '  '.join(['{} {}'.format(m, _fmt(out[m][-1])) for m in models])))
    return {m: np.asarray(v, dtype=float) for m, v in out.items()}


def _rotate_scene(S, daz, del_deg, droll, res_in, fov_in_deg=RET_FOV_DEG):
    """ Rotate the rendered movie in place of the renderer, same size and FOV.

    Scalar angles give a constant rotation; a (T,) tensor of azimuths gives a
    per-frame one, which is how a known eye gain is injected.  Border padding
    at the edges is harmless while the injected angle plus the fitted range
    stays inside the margin the crop leaves.
    """

    dev = S.device
    if np.isscalar(daz):
        daz = torch.full((1,), float(daz), device=dev)
    daz = daz.to(dev).float()
    zer = torch.zeros_like(daz)
    R = rotation_matrices(daz, zer + float(del_deg), zer + float(droll))
    return warp_scene(S, R, res_in, fov_in_deg, fov_in_deg=fov_in_deg,
                      chunk=4096)


def recovery_gain_injected(ctx, twopT, use, train_masks, test_masks,
                           gains=(-0.6, -0.3, 0.0, 0.3, 0.6), chan='theta',
                           shift_chans=('theta', 'phi', 'pitch'), seed=0,
                           verbose=True, **kw):
    """ Rotate the scene by a known EYE GAIN and ask whether the fit follows.

    The arm that gates everything else, and it injects a gain rather than a
    constant rotation for a reason worth stating plainly: a constant misaim is
    not identifiable here.  A rotation that is the same on every frame is
    exactly a translation of every cell's receptive field, and a static linear
    readout absorbs it for free -- verified on synthetic cells, where a true
    +8 deg misaim leaves out-of-fold R2 unchanged and the fitted offset lands
    anywhere.  A rotation that VARIES with eye position cannot be absorbed by a
    static field, so the gain is the identifiable part, and it happens to be
    the part the ego-versus-retino question is about.

    Injecting gain g adds g to the rotation the model must supply, so the
    fitted gain should fall by exactly g: slope -1.

    Returns
    -------
    dict with 'gains', 'fitted_gain' (n_gains, n_folds), 'slope', 'r'
    """

    it = list(shift_chans).index(chan)
    th = torch.as_tensor(ctx['chans_deg'][:, it], dtype=torch.float32,
                         device=ctx['device'])
    S0 = ctx['S']
    res_in = S0.shape[1]

    fitted = []
    for g in gains:
        ctx_g = dict(ctx)
        if g != 0:
            ctx_g['S'] = _rotate_scene(S0, float(g) * th, 0.0, 0.0, res_in)
        r = run_ladder(ctx_g, twopT, use, train_masks, test_masks,
                       models=('linear',), seed=seed, verbose=False, **kw)
        gm = r['linear']['gain_deg_per_deg']
        vals = gm[:, 0, it] if len(gm) else np.array([np.nan])
        fitted.append(vals)
        if verbose:
            print('    Injected gain {:+.2f}: fitted {} (median over '
                  'folds)...'.format(g, _fmt(np.nanmedian(vals), 2)))

    n = max(len(a) for a in fitted)
    arr = np.full((len(gains), n), np.nan)
    for i, a in enumerate(fitted):
        arr[i, :len(a)] = a

    x = np.asarray(gains, dtype=float)
    y = np.nanmedian(arr, axis=1)
    good = np.isfinite(y)
    if good.sum() >= 3:
        slope = float(np.polyfit(x[good], y[good], 1)[0])
        rr = float(np.corrcoef(x[good], y[good])[0, 1])
    else:
        slope, rr = np.nan, np.nan
    return {'gains': x, 'fitted_gain': arr, 'slope': slope, 'r': rr,
            'chan': chan}


def recovery_synthetic(ctx, twopT, use, train_masks, test_masks,
                       true_gain=1.0, chan='theta',
                       shift_chans=('theta', 'phi', 'pitch'), n_syn=200,
                       rho_grid=(0.5, 0.2, 0.1, 0.05, 0.02), seed=0,
                       verbose=True, **kw):
    """ Same test against synthetic cells with known Gaussian fields.

    Separates a broken fitter from an uninformative dataset: these cells really
    do see the scene through a known eye gain, so failing to recover it here is
    the machinery's fault.  The noise level at which recovery breaks is the
    precision this design can ever have, and it is the number to compare the
    real cells' R2 against before believing any fitted gain.

    Returns
    -------
    dict with 'rho', 'fitted_gain', 'true_gain', 'median_r2'
    """

    dev = ctx['device']
    rng = np.random.default_rng(seed)
    res = ctx['res_out']
    it = list(shift_chans).index(chan)
    th = torch.as_tensor(ctx['chans_deg'][:, it], dtype=torch.float32,
                         device=dev)
    zer = torch.zeros_like(th)

    # Cells see the scene through the true gain; the model is handed the
    # untouched render, so the answer it has to find is true_gain.
    R = rotation_matrices(float(true_gain) * th, zer, zer)
    W = warp_scene(ctx['S'], R, res, ctx['crop_deg'], chunk=4096)
    Wf = W.reshape(W.shape[0], -1)[ctx['rows_use']]

    yy, xx = np.meshgrid(np.arange(res), np.arange(res), indexing='ij')
    fields = []
    for _ in range(int(n_syn)):
        cx, cy = rng.uniform(0.2, 0.8, size=2) * res
        sig = rng.uniform(0.06, 0.18) * res
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig ** 2))
        g = g - g.mean()
        fields.append((g / (np.linalg.norm(g) + 1e-9)).ravel())
    Fm = torch.as_tensor(np.asarray(fields, dtype=np.float32).T, device=dev)

    sig_t = Wf @ Fm
    sig_t = (sig_t - sig_t.mean(dim=0)) / torch.clamp(sig_t.std(dim=0), min=1e-6)

    fitted, med_r2 = [], []
    for rho in rho_grid:
        # Noise scaled so the explainable fraction of variance is rho.
        nsd = float(np.sqrt(max(1.0 / max(rho, 1e-6) - 1.0, 0.0)))
        gcpu = torch.Generator().manual_seed(int(seed))
        Ysyn = sig_t + torch.randn(sig_t.shape, generator=gcpu).to(dev) * nsd
        Ysyn = (Ysyn - Ysyn.mean(dim=0)) / torch.clamp(Ysyn.std(dim=0), min=1e-6)

        ctx_s = dict(ctx)
        ctx_s['Y'] = Ysyn
        ctx_s['Y_full'] = Ysyn
        r = run_ladder(ctx_s, twopT, use, train_masks, test_masks,
                       models=('linear',), seed=seed, verbose=False, **kw)
        gm = r['linear']['gain_deg_per_deg']
        fitted.append(float(np.nanmedian(gm[:, 0, it])) if len(gm) else np.nan)
        med_r2.append(r['linear']['median_r2'])
        if verbose:
            print('    Explainable {:.2f}: fitted gain {} (true {:+.2f}), '
                  'median R2 {}...'.format(rho, _fmt(fitted[-1], 2), true_gain,
                                           _fmt(med_r2[-1])))

    return {'rho': np.asarray(rho_grid, dtype=float),
            'fitted_gain': np.asarray(fitted, dtype=float),
            'true_gain': float(true_gain),
            'median_r2': np.asarray(med_r2, dtype=float)}


def _load_session(ego_npz, retino_npz, h5_path, spike_key, smooth_s, bin_px,
                  speed_thresh, light_only, shift_chans, lags, n_folds,
                  fold_block_s, embargo_s, max_cells, seed, verbose=True):
    """ Load rates, both renders and the shifter channels onto a shared frame set.

    Both renders must see exactly the same frames or the ego-versus-retino gain
    is not a paired quantity.  Frames without a finite eye angle are dropped
    here even though the egocentric render does not need one: the shifter does.

    Returns
    -------
    dict of everything the conditions share
    """

    npz_paths = [('egocentric', ego_npz), ('retinocentric', retino_npz)]
    npz_paths = [(k, v) for k, v in npz_paths if v]
    if not npz_paths:
        raise ValueError('Need at least one npz path.')

    ref = npz_paths[0][1]
    stem, sep, _ = ref.rpartition('_retinal_images')
    if h5_path is None:
        h5_path = (stem + '_preproc.h5' if sep
                   else ref.replace('.npz', '_preproc.h5'))
    if h5_path.endswith('.npz') or not os.path.isfile(h5_path):
        raise FileNotFoundError(
            'Could not find the preproc file for {}; looked for {}. Pass '
            'h5_path explicitly.'.format(os.path.basename(ref), h5_path))

    print('Preproc: {}'.format(os.path.basename(h5_path)))
    with h5py.File(h5_path, 'r') as f:
        if spike_key not in f:
            raise KeyError('{!r} not in {}.'.format(spike_key,
                                                    os.path.basename(h5_path)))
        rate = np.asarray(f[spike_key][:], dtype=float)
        twopT = np.asarray(f['twopT'][:], dtype=float)
        speed = np.asarray(f['speed'][:], dtype=float) if 'speed' in f else None
        ltdk = (np.asarray(f['ltdk_state_vec'][:]).astype(bool)
                if 'ltdk_state_vec' in f else None)

    n_fr = min(rate.shape[1], len(twopT))
    rate, twopT = rate[:, :n_fr], twopT[:n_fr]
    if max_cells is not None:
        rate = rate[:int(max_cells)]
    Y = rate.T.copy()
    Y[~np.isfinite(Y)] = 0.0
    Y = (Y - Y.mean(axis=0)) / np.where(Y.std(axis=0) > 0, Y.std(axis=0), 1.0)
    dt_2p = float(np.nanmedian(np.diff(twopT)))
    if smooth_s and smooth_s > 0:
        Y = _triangle_smooth(Y, smooth_s, dt_2p)
        Y = (Y - Y.mean(axis=0)) / np.where(Y.std(axis=0) > 0, Y.std(axis=0), 1.0)
    print('  {} cells, {} 2P frames at {:.1f} Hz, target {!r} smoothed at '
          '{:g} s...'.format(Y.shape[1], n_fr, 1.0 / dt_2p, spike_key, smooth_s))

    scenes, coverage, metas = {}, {}, {}
    for label, path in npz_paths:
        S, cov, meta = load_scene(path, twopT, bin_px=bin_px, verbose=verbose)
        scenes[label], coverage[label], metas[label] = S, cov, meta
        metas[label]['path'] = os.path.basename(path)
    if metas.get('egocentric', {}).get('fixed_eye', 0):
        print('  NOTE: egocentric npz was rendered with fixed_eye, which also '
              'zeroes the anatomical socket. The ego arm then differs from the '
              'retino arm in where it looks as well as in eye movement, and '
              'the fitted gain is not a clean 1-df contrast. Re-render the ego '
              'arm with --fixed_pupil.')

    chans = load_behavior_channels(h5_path, n_fr)
    missing = [c for c in shift_chans if c not in chans]
    if missing:
        raise KeyError('Shifter channels {} not in the preproc; available: '
                       '{}'.format(missing, sorted(chans)))
    chans_deg = np.column_stack([chans[c][:n_fr] for c in shift_chans])

    use = np.ones(n_fr, dtype=bool)
    for label in scenes:
        use &= coverage[label]
    use &= np.isfinite(rate).all(axis=0)
    use &= np.isfinite(chans_deg).all(axis=1)
    if speed is not None and speed_thresh > 0:
        sp = np.asarray(speed, dtype=float)
        if len(sp) < n_fr:
            sp = np.concatenate([sp, np.full(n_fr - len(sp),
                                             sp[-1] if len(sp) else 0.0)])
        use &= sp[:n_fr] > speed_thresh
    if ltdk is not None and len(ltdk) >= n_fr and light_only:
        use &= ltdk[:n_fr]
        print('  Light-only and moving: {} frames...'.format(int(use.sum())))

    all_lags = sorted(set(list(lags) + list(SHIFTER_LAGS)))
    before = int(use.sum())
    use = trim_lag_boundary(use, all_lags)
    print('  {}/{} 2P frames usable ({:.1f} min), {} dropped at the lag '
          'boundary...'.format(int(use.sum()), n_fr, use.sum() * dt_2p / 60.0,
                               before - int(use.sum())))
    if use.sum() < 300:
        raise RuntimeError('Only {} usable frames.'.format(int(use.sum())))

    fold_id, train_masks, test_masks = block_folds(
        twopT, use, n_folds=n_folds, block_s=fold_block_s,
        embargo_s=embargo_s, seed=seed)

    return {'Y': Y, 'twopT': twopT, 'use': use, 'scenes': scenes,
            'metas': metas, 'chans_deg': chans_deg, 'shift_chans': shift_chans,
            'train_masks': train_masks, 'test_masks': test_masks,
            'fold_id': fold_id, 'dt_2p': dt_2p, 'h5_path': h5_path,
            'stem': stem if sep else ref.replace('.npz', '')}


def _build_ctx(S, sess, device, crop_deg, res_out, n_dct, lags, max_shift,
               hidden, chunk):
    """ Everything the fitting functions need for one render, on the device. """

    use = sess['use']
    all_lags = sorted(set(list(lags) + list(SHIFTER_LAGS)))
    need = needed_frames(use, all_lags)
    use_idx = np.flatnonzero(use)
    need_idx = np.flatnonzero(need)

    # Warp the need set, score on the use set: a lag source only has to be
    # renderable, not a sample in its own right.
    S_t = torch.as_tensor(S[need], dtype=torch.float32, device=device)
    Y_full = torch.as_tensor(sess['Y'], dtype=torch.float32, device=device)
    Y = Y_full[torch.as_tensor(use_idx, device=device)]

    # Lag sources are not required to pass the eye filter that use is, so they
    # can carry NaN pose.  Only their warp depends on it, so interpolate rather
    # than drop; a NaN here silently NaNs the whole shifter path.
    ch = np.array(sess['chans_deg'][need], dtype=float)
    n_filled = 0
    for j in range(ch.shape[1]):
        bad = ~np.isfinite(ch[:, j])
        n_filled += int(bad.sum())
        if not bad.any():
            continue
        good = ~bad
        ch[bad, j] = (np.interp(np.flatnonzero(bad), np.flatnonzero(good),
                                ch[good, j]) if good.any() else 0.0)
    if n_filled:
        print('  Filled {} non-finite shifter samples on lag-source '
              'frames...'.format(n_filled))

    mu, sd = ch.mean(axis=0), ch.std(axis=0)
    ch_z = (ch - mu) / np.where(sd > 0, sd, 1.0)

    basis = dct_basis(res_out, n_dct)
    return {
        'device': device,
        'S': S_t,
        'Y': Y,
        'Y_full': Y_full,
        'use_idx': use_idx,
        'basis': torch.as_tensor(basis, device=device),
        'res_out': int(res_out),
        'crop_deg': float(crop_deg),
        'chunk': chunk,
        'shift_in': torch.as_tensor(ch_z, dtype=torch.float32, device=device),
        'chans_deg': ch,
        'n_in': ch.shape[1],
        'max_shift': float(max_shift),
        'hidden': int(hidden),
        'lag_rows': _lag_index_map(use, need, lags, device),
        'lag_rows_fit': _lag_index_map(use, need, SHIFTER_LAGS, device),
        'rows_use': torch.as_tensor(
            np.searchsorted(need_idx, use_idx), device=device, dtype=torch.long),
    }


def shifter_scene(ego_npz, retino_npz=None, h5_path=None, out_h5=None,
                  spike_key='norm_spikes', smooth_s=1.2, bin_px=2,
                  crop_deg=DEFAULT_CROP_DEG, res_out=DEFAULT_RES_OUT, n_dct=5,
                  lags=DEFAULT_LAGS, n_folds=5, fold_block_s=120.0,
                  embargo_s=40.0, speed_thresh=SPEED_THRESH_CM_S,
                  light_only=True, shift_chans=('theta', 'phi', 'pitch'),
                  models=('none', 'constant', 'linear'), n_steps=300, lr=0.05,
                  n_null=8, max_cells=None, seed=0, hidden=20, chunk=2048,
                  do_recovery=True, make_pdf=True, device=None, verbose=True):
    """ Fit the ladder on both renders, run the controls, save and plot.

    Parameters
    ----------
    ego_npz, retino_npz : str
        Renders from get_retinal_image.  The ego arm should be the
        --fixed_pupil render, not --fixed_eye.
    crop_deg, res_out, n_dct : float, int, int
        Cropped field the readout sees, its resolution, and DCT components per
        axis.  n_dct 5 gives 25 spatial features per lag.
    models : sequence
        Rungs to run, from 'none', 'constant', 'linear', 'mlp'.
    n_null : int
        Circular shifts for each of the two nulls.

    Returns
    -------
    res : dict
        Everything written to the output h5.
    """

    dev = _device(device)
    print('Device: {}'.format(dev))
    t_start = time.time()

    sess = _load_session(ego_npz, retino_npz, h5_path, spike_key, smooth_s,
                         bin_px, speed_thresh, light_only, list(shift_chans),
                         lags, n_folds, fold_block_s, embargo_s, max_cells,
                         seed, verbose=verbose)
    if out_h5 is None:
        out_h5 = sess['stem'] + '_shifter_scene.h5'

    max_shift = (RET_FOV_DEG - crop_deg) / 2.0
    fit_kw = dict(n_steps=n_steps, lr=lr, fold_block_s=fold_block_s,
                  embargo_s=embargo_s)
    print('  Crop {:g} deg at {} px, {} DCT components per axis, shift clamped '
          'to {:g} deg...'.format(crop_deg, res_out, n_dct, max_shift))

    conditions = {}
    for label, S in sess['scenes'].items():
        print('Fitting {}...'.format(label))
        ctx = _build_ctx(S, sess, dev, crop_deg, res_out, n_dct, lags,
                         max_shift, hidden, chunk)
        cond = {}
        cond['ladder'] = run_ladder(ctx, sess['twopT'], sess['use'],
                                    sess['train_masks'], sess['test_masks'],
                                    models=models, seed=seed, verbose=verbose,
                                    **fit_kw)
        for m in models:
            print('  {:9s} median R2 {}'.format(
                m, _fmt(cond['ladder'][m]['median_r2'])))

        if n_null > 0:
            print('  [{}] shuffled-target null ({} shifts)...'.format(label, n_null))
            cond['null_targets'] = null_shuffled_targets(
                ctx, sess['twopT'], sess['use'], sess['train_masks'],
                sess['test_masks'], models, n_null=n_null, seed=seed,
                verbose=verbose, **fit_kw)
            shift_models = tuple(m for m in models if m != 'none')
            if shift_models:
                print('  [{}] shuffled-input null...'.format(label))
                cond['null_inputs'] = null_shuffled_inputs(
                    ctx, sess['twopT'], sess['use'], sess['train_masks'],
                    sess['test_masks'], shift_models, n_null=n_null, seed=seed,
                    verbose=verbose, **fit_kw)

        if do_recovery:
            print('  [{}] Injected eye-gain recovery...'.format(label))
            cond['recovery_gain'] = recovery_gain_injected(
                ctx, sess['twopT'], sess['use'], sess['train_masks'],
                sess['test_masks'], shift_chans=shift_chans, seed=seed,
                verbose=verbose, **fit_kw)
            print('  [{}] synthetic-cell recovery...'.format(label))
            cond['recovery_synthetic'] = recovery_synthetic(
                ctx, sess['twopT'], sess['use'], sess['train_masks'],
                sess['test_masks'], shift_chans=shift_chans, seed=seed,
                verbose=verbose, **fit_kw)

        with torch.no_grad():
            R0 = rotation_matrices(*[torch.zeros(1, device=dev)] * 3)
            C = warp_scene(ctx['S'], R0, res_out, crop_deg, chunk=chunk)
            cond['mean_crop'] = C.mean(dim=0).cpu().numpy()
        cond['meta'] = sess['metas'][label]
        conditions[label] = cond

        del ctx
        if dev.type == 'cuda':
            torch.cuda.empty_cache()

    res = {
        'conditions': conditions,
        'meta': {
            'h5_path': sess['h5_path'],
            'spike_key': spike_key,
            'smooth_s': float(smooth_s),
            'bin_px': int(bin_px),
            'crop_deg': float(crop_deg),
            'res_out': int(res_out),
            'n_dct': int(n_dct),
            'lags': np.asarray(lags),
            'shifter_lags': np.asarray(SHIFTER_LAGS),
            'shift_chans': list(shift_chans),
            'max_shift_deg': float(max_shift),
            'n_folds': int(n_folds),
            'fold_block_s': float(fold_block_s),
            'embargo_s': float(embargo_s),
            'n_frames_used': int(sess['use'].sum()),
            'minutes_used': float(sess['use'].sum() * sess['dt_2p'] / 60.0),
            'n_cells': int(sess['Y'].shape[1]),
            'n_null': int(n_null),
            'models': list(models),
        },
    }
    res['comparison'] = _compare_gains(conditions, list(shift_chans))

    print_verdict(res)
    _save(res, out_h5)
    _save_json(res, out_h5.replace('.h5', '.json'))
    if make_pdf:
        make_summary_pdf(res, out_h5.replace('.h5', '.pdf'))
    print('Done in {:.0f}s.'.format(time.time() - t_start))
    return res


def _compare_gains(conditions, shift_chans):
    """ Azimuth-on-theta gain in each render, and what separates them.

    The retinocentric render already contains the eye rotation, so its residual
    gain should sit near zero; the egocentric render does not, so its gain is
    the eye rotation the cells actually want.  Their difference is the render's
    own Jacobian measured on the same data, which is what makes the ego gain
    interpretable without deriving a sign convention by hand.
    """

    out = {}
    if 'theta' not in shift_chans:
        return out
    it = list(shift_chans).index('theta')
    for label, c in conditions.items():
        g = _median_gain(c['ladder'], 'linear')
        if g is None:
            continue
        folds = c['ladder']['linear']['gain_deg_per_deg']
        out[label + '_az_on_theta'] = float(g[0, it])
        out[label + '_az_on_theta_folds'] = np.asarray(folds[:, 0, it], dtype=float)
    a = out.get('egocentric_az_on_theta', np.nan)
    b = out.get('retinocentric_az_on_theta', np.nan)
    both = np.isfinite(a) and np.isfinite(b)
    out['jacobian_deg_per_deg'] = float(a - b) if both else np.nan
    # Cells want total rotation G; the ego fit returns a = G, the retino fit the
    # residual b = G - J.  So a - b is the render's own Jacobian J and a / J is
    # the fraction of it the cells actually want: 1 retinocentric, 0 egocentric.
    out['retino_fraction'] = (float(a / (a - b)) if both and abs(a - b) > 1e-6
                              else np.nan)
    return out


def _perm_p(obs, null):
    """ One-sided permutation p, floored at 1 / (n + 1). """

    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not len(null) or not np.isfinite(obs):
        return np.nan
    return float((1.0 + np.sum(null >= obs)) / (1.0 + len(null)))


def print_verdict(res):
    """ Say in plain terms what the arms decided, recovery first.

    Recovery leads because it gates everything else: a fitted rotation from a
    recording that cannot locate an injected one is not a measurement, and the
    ego-versus-retino gain built on it is unfalsifiable.
    """

    models = res['meta']['models']
    print('')
    print('Verdict')
    print('-------')
    for label, c in res['conditions'].items():
        print('  {}:'.format(label))

        rec = c.get('recovery_gain')
        if rec is not None:
            sl = rec['slope']
            print('    Injected eye-gain recovery: slope {} (r {}, perfect is '
                  '-1)  -- {}'.format(
                      _fmt(sl, 2), _fmt(rec['r'], 2),
                      'the fit tracks a known gain, so the fitted gain below '
                      'is a measurement' if np.isfinite(sl) and sl < -0.3 else
                      'the fit does NOT track a known gain, so the fitted gain '
                      'below is noise'))
        syn = c.get('recovery_synthetic')
        if syn is not None:
            ok = np.isfinite(syn['fitted_gain']) & (
                np.abs(syn['fitted_gain'] - syn['true_gain']) < 0.3)
            worst = (syn['rho'][ok].min() if ok.any() else np.nan)
            print('    Synthetic cells: recovers a gain of {:+.2f} down to an '
                  'explainable fraction of {}  -- {}'.format(
                      syn['true_gain'], _fmt(worst, 3),
                      'machinery works; the only question left is whether the '
                      'real cells sit above that line' if ok.any() else
                      'machinery fails even on cells that really see the scene'))

        prev = None
        for m in models:
            r2 = c['ladder'][m]['median_r2']
            line = '    {:9s} median R2 {}'.format(m, _fmt(r2))
            if prev is not None:
                line += '  gain over {} {}'.format(prev[0], _fmt(r2 - prev[1]))
            nt = c.get('null_targets', {}).get(m)
            if nt is not None:
                line += '  shuffled-target null {} (p {})'.format(
                    _fmt(np.nanmedian(nt)), _fmt(_perm_p(r2, nt), 3))
            ni = c.get('null_inputs', {}).get(m)
            if ni is not None:
                line += '  shuffled-input null {}'.format(_fmt(np.nanmedian(ni)))
            print(line)
            prev = (m, r2)

        for m in models:
            if m in ('none',):
                continue
            sd = c['ladder'][m]['shift_deg']
            if not len(sd):
                continue
            print('    {:9s} shift across folds: az {} +- {}  el {} +- {}  '
                  'roll {} +- {} deg'.format(
                      m, _fmt(np.nanmedian(sd[:, 0]), 1), _fmt(np.nanstd(sd[:, 0]), 1),
                      _fmt(np.nanmedian(sd[:, 1]), 1), _fmt(np.nanstd(sd[:, 1]), 1),
                      _fmt(np.nanmedian(sd[:, 2]), 1), _fmt(np.nanstd(sd[:, 2]), 1)))

    cm = res.get('comparison', {})
    if cm:
        print('  eye gain (azimuth per degree of theta):')
        for label in ('egocentric', 'retinocentric'):
            k = label + '_az_on_theta'
            if k in cm:
                f = cm[k + '_folds']
                print('    {:14s} {} (folds {})'.format(
                    label, _fmt(cm[k], 2),
                    ', '.join([_fmt(v, 2) for v in np.asarray(f).ravel()])))
        if np.isfinite(cm.get('jacobian_deg_per_deg', np.nan)):
            print('    Render Jacobian (ego minus retino) {}  -- the eye '
                  'rotation the render itself applies, measured on these '
                  'data'.format(_fmt(cm['jacobian_deg_per_deg'], 2)))
        if np.isfinite(cm.get('retino_fraction', np.nan)):
            rf = cm['retino_fraction']
            print('    Retinocentric fraction {}  -- {}'.format(
                _fmt(rf, 2),
                'the cells want the eye rotation the retino render applies'
                if rf > 0.5 else
                'the cells want less eye rotation than the retino render '
                'applies'))
    print('')


def _sanitize(obj):
    """ Recursively drop Nones and unsaveable values so write_h5 can take it. """

    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            s = _sanitize(v)
            if s is not None:
                out[str(k)] = s
        return out
    if isinstance(obj, (list, tuple)):
        if not len(obj):
            return None
        # h5py takes bytes but not numpy unicode arrays, and write_h5 swallows
        # the TypeError, so a list of channel names would vanish silently.
        if all(isinstance(v, str) for v in obj):
            return np.array([v.encode('ascii', 'replace') for v in obj])
        try:
            return np.asarray(obj)
        except (ValueError, TypeError):
            return None
    if obj is None:
        return None
    if isinstance(obj, (str, bytes, bool, int, float, np.ndarray, np.generic)):
        return obj
    return None


def _save(res, out_h5):
    """ Write the whole result tree to h5. """

    write_h5(out_h5, _sanitize(res))
    print('Saved {}.'.format(out_h5))


def _save_json(res, out_json):
    """ Scalar summary, for grepping across sessions without opening the h5. """

    def _med(x):
        """ NaN-median as float, None for empty input. """

        x = np.asarray(x, dtype=float)
        return float(np.nanmedian(x)) if x.size else None

    summary = {'meta': {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in res['meta'].items()},
               'conditions': {}}
    for label, c in res['conditions'].items():
        d = {}
        for m, r in c['ladder'].items():
            d[m] = {'median_r2': r['median_r2'],
                    'shift_deg': (np.nanmedian(r['shift_deg'], axis=0).tolist()
                                  if len(r['shift_deg']) else None),
                    'null_targets': _med(c.get('null_targets', {}).get(m, [])),
                    'null_inputs': _med(c.get('null_inputs', {}).get(m, []))}
        if 'recovery_gain' in c:
            d['recovery_gain_slope'] = c['recovery_gain']['slope']
        if 'recovery_synthetic' in c:
            syn = c['recovery_synthetic']
            d['recovery_synthetic_fitted'] = syn['fitted_gain'].tolist()
        summary['conditions'][label] = d
    summary['comparison'] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                             for k, v in res.get('comparison', {}).items()}
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print('Saved {}.'.format(out_json))


def make_summary_pdf(res, out_pdf):
    """ Figures for the whole run, recovery first for the same reason. """

    print('Building PDF {}...'.format(out_pdf))
    models = res['meta']['models']
    labels = list(res['conditions'].keys())
    colors = {'egocentric': '#1f77b4', 'retinocentric': '#d62728'}

    with PdfPages(out_pdf) as pdf:
        fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Recovery: can this recording locate an eye gain at all')

        ax = axs[0, 0]
        for label in labels:
            rec = res['conditions'][label].get('recovery_gain')
            if rec is None:
                continue
            ax.plot(rec['gains'], np.nanmedian(rec['fitted_gain'], axis=1), 'o-',
                    color=colors.get(label, 'k'), label='{} (slope {})'.format(
                        label, _fmt(rec['slope'], 2)))
        lim = ax.get_xlim()
        ax.plot(lim, [-l for l in lim], 'k--', lw=0.8, label='perfect recovery')
        ax.set_xlabel('injected gain (deg azimuth per deg theta)')
        ax.set_ylabel('fitted gain')
        ax.legend(fontsize=7)
        ax.set_title('injected eye gain, real cells')

        ax = axs[0, 1]
        for label in labels:
            syn = res['conditions'][label].get('recovery_synthetic')
            if syn is None:
                continue
            ax.semilogx(syn['rho'], syn['fitted_gain'], 'o-',
                        color=colors.get(label, 'k'), label=label)
            ax.axhline(syn['true_gain'], color='k', ls='--', lw=0.8)
        ax.set_xlabel('explainable fraction of variance')
        ax.set_ylabel('fitted gain')
        ax.legend(fontsize=7)
        ax.set_title('synthetic cells, known field and gain')

        ax = axs[1, 0]
        w = 0.35
        for i, label in enumerate(labels):
            c = res['conditions'][label]
            vals = [c['ladder'][m]['median_r2'] for m in models]
            ax.bar(np.arange(len(models)) + i * w, vals, w,
                   color=colors.get(label, 'k'), label=label)
            nt = c.get('null_targets')
            if nt:
                nul = [np.nanmedian(nt[m]) if m in nt else np.nan for m in models]
                ax.plot(np.arange(len(models)) + i * w, nul, 'k_', ms=14,
                        label='shuffled-target null' if i == 0 else None)
        ax.set_xticks(np.arange(len(models)) + w / 2)
        ax.set_xticklabels(models)
        ax.set_ylabel('median out-of-fold R2')
        ax.axhline(0, color='k', lw=0.5)
        ax.legend(fontsize=7)
        ax.set_title('model ladder')

        ax = axs[1, 1]
        ax.axis('off')
        meta = res['meta']
        txt = ['{} cells, {} frames ({:.1f} min)'.format(
            meta['n_cells'], meta['n_frames_used'], meta['minutes_used']),
            'crop {:g} deg at {} px, {} DCT/axis'.format(
                meta['crop_deg'], meta['res_out'], meta['n_dct']),
            'shift clamped to {:g} deg'.format(meta['max_shift_deg']),
            'shifter channels: {}'.format(', '.join(meta['shift_chans'])),
            '']
        cm = res.get('comparison', {})
        for k in ('egocentric_az_on_theta', 'retinocentric_az_on_theta',
                  'jacobian_deg_per_deg', 'retino_fraction'):
            if k in cm and np.isscalar(cm[k]):
                txt.append('{}: {}'.format(k, _fmt(cm[k], 2)))
        ax.text(0.0, 1.0, '\n'.join(txt), va='top', fontsize=8, family='monospace')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        for label in labels:
            c = res['conditions'][label]
            lad = c['ladder']
            if 'surface' not in lad.get('constant', {}):
                continue
            surf = lad['constant']['surface']
            az_g, el_g = lad['constant']['az_grid'], lad['constant']['el_grid']
            m_surf = np.nanmean(surf, axis=0)
            ir = np.unravel_index(np.nanargmin(m_surf), m_surf.shape)[2]

            fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
            fig.suptitle('{}: constant-rotation search'.format(label))
            im = axs[0].imshow(m_surf[:, :, ir].T, origin='lower', aspect='auto',
                               extent=[az_g[0], az_g[-1], el_g[0], el_g[-1]],
                               cmap='viridis_r')
            axs[0].set_xlabel('azimuth (deg)')
            axs[0].set_ylabel('elevation (deg)')
            axs[0].set_title('inner-val MSE, roll {:g} deg'.format(
                lad['constant']['roll_grid'][ir]))
            plt.colorbar(im, ax=axs[0])

            for m in models:
                sd = lad[m]['shift_deg']
                if not len(sd) or m == 'none':
                    continue
                axs[1].plot(sd[:, 0], sd[:, 1], 'o', label=m)
            axs[1].axhline(0, color='k', lw=0.5)
            axs[1].axvline(0, color='k', lw=0.5)
            axs[1].set_xlabel('azimuth (deg)')
            axs[1].set_ylabel('elevation (deg)')
            axs[1].set_title('fitted mean shift per fold (offset is not identifiable)')
            axs[1].legend(fontsize=7)

            axs[2].imshow(c['mean_crop'], origin='upper', cmap='gray')
            axs[2].set_title('mean cropped scene')
            axs[2].axis('off')
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        fig, axs = plt.subplots(1, max(len(labels), 2), figsize=(11, 4.5),
                                squeeze=False)
        fig.suptitle('Fitted gain, degrees of rotation per degree of input')
        for i, label in enumerate(labels):
            g = _median_gain(res['conditions'][label]['ladder'], 'linear')
            ax = axs[0, i]
            if g is None:
                ax.axis('off')
                continue
            im = ax.imshow(g, cmap='RdBu_r', vmin=-np.nanmax(np.abs(g)) - 1e-9,
                           vmax=np.nanmax(np.abs(g)) + 1e-9)
            ax.set_xticks(range(g.shape[1]))
            ax.set_xticklabels(res['meta']['shift_chans'], rotation=45, fontsize=7)
            ax.set_yticks(range(3))
            ax.set_yticklabels(['az', 'el', 'roll'], fontsize=7)
            ax.set_title(label, fontsize=9)
            plt.colorbar(im, ax=ax)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print('Saved {}.'.format(out_pdf))


def main():
    """ CLI entry point. """

    p = argparse.ArgumentParser(description='Shifter fit on the reconstructed scene')
    p.add_argument('--ego_npz', required=True,
                   help='Egocentric render; use the --fixed_pupil one')
    p.add_argument('--retino_npz', default=None)
    p.add_argument('--h5_path', default=None)
    p.add_argument('--out_h5', default=None)
    p.add_argument('--spike_key', default='norm_spikes')
    p.add_argument('--smooth_s', type=float, default=1.2)
    p.add_argument('--bin_px', type=int, default=2)
    p.add_argument('--crop_deg', type=float, default=DEFAULT_CROP_DEG)
    p.add_argument('--res_out', type=int, default=DEFAULT_RES_OUT)
    p.add_argument('--n_dct', type=int, default=5)
    p.add_argument('--lags', type=int, nargs='+', default=list(DEFAULT_LAGS))
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--fold_block_s', type=float, default=120.0)
    p.add_argument('--embargo_s', type=float, default=40.0)
    p.add_argument('--speed_thresh', type=float, default=SPEED_THRESH_CM_S)
    p.add_argument('--all_frames', action='store_true',
                   help='Keep dark frames as well as light')
    p.add_argument('--shift_chans', nargs='+', default=['theta', 'phi', 'pitch'])
    p.add_argument('--mlp', action='store_true',
                   help='Add the Niell lab MLP shifter as a fourth rung')
    p.add_argument('--n_steps', type=int, default=300)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--n_null', type=int, default=8)
    p.add_argument('--no_recovery', action='store_true')
    p.add_argument('--no_pdf', action='store_true')
    p.add_argument('--max_cells', type=int, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default=None)
    args = p.parse_args()

    models = ['none', 'constant', 'linear'] + (['mlp'] if args.mlp else [])
    shifter_scene(
        ego_npz=args.ego_npz, retino_npz=args.retino_npz, h5_path=args.h5_path,
        out_h5=args.out_h5, spike_key=args.spike_key, smooth_s=args.smooth_s,
        bin_px=args.bin_px, crop_deg=args.crop_deg, res_out=args.res_out,
        n_dct=args.n_dct, lags=tuple(args.lags), n_folds=args.n_folds,
        fold_block_s=args.fold_block_s, embargo_s=args.embargo_s,
        speed_thresh=args.speed_thresh, light_only=not args.all_frames,
        shift_chans=tuple(args.shift_chans), models=tuple(models),
        n_steps=args.n_steps, lr=args.lr, n_null=args.n_null,
        max_cells=args.max_cells, seed=args.seed, device=args.device,
        do_recovery=not args.no_recovery, make_pdf=not args.no_pdf)


if __name__ == '__main__':
    main()

# python -m fm2p.ffNLE_scene --ego_npz /home/dylan/Storage/freely_moving_data/V1PPC/cohort02_recordings/251016_DMM_DMM061_pos18/fm1/251016_DMM_DMM061_fm_01_retinal_images_fixed_pupil_ego.npz --retino_npz /home/dylan/Storage/freely_moving_data/V1PPC/cohort02_recordings/251016_DMM_DMM061_pos18/fm1/251016_DMM_DMM061_fm_01_retinal_images_retinocentric.npz --spike_key norm_spikes_s2p --h5_path /home/dylan/Storage/freely_moving_data/V1PPC/cohort02_recordings/251016_DMM_DMM061_pos18/fm1/251016_DMM_DMM061_fm_01_preproc.h5
