if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 7

from .utils.files import read_h5

REVCORR_PATH = '/home/dylan/Storage/freely_moving_data/_LGN/250923_DMM_DMM052_lgnaxons/fm1/eyehead_revcorrs_v06.h5'
# REVCORR_PATH = '/home/dylan/Storage/freely_moving_data/LP/250514_DMM_DMM046_LPaxons/fm1/eyehead_revcorrs_v06.h5'

# Condition index in the tuning array dim-2: 0=dark, 1=light
# Only used for Format A files (flat key structure with light/dark split).
COND_IDX = 1
COND_KEY = 'l'   # used to look up {var}_l_rel in Format A

# Variables in display order; panels skipped if absent in the file.
# gyro_x=dRoll, gyro_y=dPitch, gyro_z=dYaw
# Format A variables (eyehead_revcorrs_v06.h5)
VAR_ORDER_A = [
    'theta', 'phi',
    'dTheta', 'dPhi',
    'pitch', 'gyro_y',
    'roll',  'gyro_x',
    'yaw',   'gyro_z',
]

# Format B variables (revcorr_results.h5 - nested group format)
VAR_ORDER_B = [
    'theta', 'phi',
    'yaw',
    'egocentric', 'retinocentric',
    'distance_to_pillar',
]

VAR_LABEL = {
    'theta':              'θ',
    'phi':                'φ',
    'dTheta':             'dθ',
    'dPhi':               'dφ',
    'pitch':              'pitch',
    'gyro_y':             'dPitch',
    'roll':               'roll',
    'gyro_x':             'dRoll',
    'yaw':                'yaw',
    'gyro_z':             'dYaw',
    'egocentric':         'ego',
    'retinocentric':      'retino',
    'distance_to_pillar': 'dist',
}

# Earth-tone palette
VAR_COLOR = {
    'theta':              '#2ECC71',
    'dTheta':             '#82E0AA',
    'phi':                '#FF9800',
    'dPhi':               '#FFCC80',
    'pitch':              '#03A9F4',
    'gyro_y':             '#81D4FA',
    'roll':               '#9C27B0',
    'gyro_x':             '#E1BEE7',
    'yaw':                '#FFEB3B',
    'gyro_z':             '#FFF59D',
    'egocentric':         '#E74C3C',
    'retinocentric':      '#F1948A',
    'distance_to_pillar': '#8B4513',
}

N_CELLS_PER_PAGE = 8

# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(data):
    """Return 'A' for flat key structure or 'B' for nested group structure.

    Format A (eyehead_revcorrs_v06.h5): flat keys like ``{var}_1dtuning``,
    ``{var}_l_rel``; tuning shape ``(n_cells, n_bins, 2)`` for dark/light.

    Format B (revcorr_results.h5): nested dicts like ``data[var]['tuning_curve']``;
    shape ``(n_cells, n_bins)``; reliability via ``cohen_d_vals``.
    """
    for v in data.values():
        if isinstance(v, dict):
            return 'B'
    for k in data:
        if k.endswith('_1dtuning'):
            return 'A'
    return 'A'


def get_present_vars(data, fmt):
    """Return the ordered list of variables available in *data*."""
    if fmt == 'A':
        return [v for v in VAR_ORDER_A
                if f'{v}_1dtuning' in data and f'{v}_1dbins' in data]
    else:
        return [v for v in VAR_ORDER_B
                if v in data and isinstance(data[v], dict) and 'tuning_curve' in data[v]]


def get_n_cells(data, present_vars, fmt):
    if fmt == 'A':
        return np.array(data[f'{present_vars[0]}_1dtuning']).shape[0]
    else:
        return np.array(data[present_vars[0]]['tuning_curve']).shape[0]


def get_title_metric(data, var, cell_idx, fmt):
    """Return (metric_value, metric_label) for the column title."""
    if fmt == 'A':
        key = f'{var}_{COND_KEY}_rel'
        val = float(np.array(data[key])[cell_idx]) if key in data else float('nan')
        return val, 'cvMI'
    else:
        grp = data.get(var, {})
        arr = grp.get('cohen_d_vals', None)
        val = float(np.array(arr)[cell_idx]) if arr is not None else float('nan')
        return val, 'cohenD'


def get_tuning(data, var, cell_idx, fmt):
    """Return (centers, tc, err_or_None) for a single cell."""
    if fmt == 'A':
        tc_arr = np.array(data[f'{var}_1dtuning'])   # (n_cells, n_bins[, 2])
        bins   = np.array(data[f'{var}_1dbins'])

        if tc_arr.ndim == 3:
            tc = tc_arr[cell_idx, :, COND_IDX] if tc_arr.shape[2] > COND_IDX else tc_arr[cell_idx, :, 0]
        else:
            tc = tc_arr[cell_idx]

        err = None
        err_key = f'{var}_1derr'
        if err_key in data:
            err_arr = np.array(data[err_key])
            if err_arr.ndim == 3:
                err = err_arr[cell_idx, :, COND_IDX] if err_arr.shape[2] > COND_IDX else err_arr[cell_idx, :, 0]
            else:
                err = err_arr[cell_idx]

    else:  # Format B
        grp  = data[var]
        tc_arr = np.array(grp['tuning_curve'])   # (n_cells, n_bins)
        bins   = np.array(grp['tuning_bins'])
        tc     = tc_arr[cell_idx]

        err = None
        if 'tuning_stderr' in grp:
            err = np.array(grp['tuning_stderr'])[cell_idx]

    if len(bins) == len(tc) + 1:
        centers = 0.5 * (bins[:-1] + bins[1:])
    else:
        centers = bins

    return centers, tc, err

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

data = read_h5(REVCORR_PATH)

fmt = detect_format(data)
present_vars = get_present_vars(data, fmt)

if not present_vars:
    raise RuntimeError(f'No expected variables found in {REVCORR_PATH} (format={fmt})')

n_cells = get_n_cells(data, present_vars, fmt)
print(f'Detected format {fmt}. Found {n_cells} cells, variables: {present_vars}')

# Pre-compute MI arrays for all variables (used for per-page sorting)
mi_cache = {
    v: np.array([get_title_metric(data, v, i, fmt)[0] for i in range(n_cells)])
    for v in present_vars
}

# With IMU data (many variables) make one page per variable, each sorted by
# that variable's MI.  Otherwise make 5 sequential pages sorted by the first
# variable's MI.
has_imu = len(present_vars) > 4
if has_imu:
    sort_vars = present_vars[:10]   # up to 10 pages
else:
    sort_vars = [present_vars[0]] * 5

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

for page, sort_var in enumerate(sort_vars):
    mi_vals = mi_cache[sort_var]
    cell_order = np.argsort(np.where(np.isnan(mi_vals), -np.inf, mi_vals))[::-1]

    if has_imu:
        page_cells = cell_order[:N_CELLS_PER_PAGE]
    else:
        page_cells = cell_order[page * N_CELLS_PER_PAGE : (page + 1) * N_CELLS_PER_PAGE]

    if len(page_cells) == 0:
        break

    n_cols = len(page_cells)
    n_rows = len(present_vars)

    fig, axs = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 1.5 + 0.5, n_rows * 1.2),
        dpi=300,
        constrained_layout=True,
    )

    # Ensure 2-D indexing
    if n_rows == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif n_rows == 1:
        axs = axs[np.newaxis, :]
    elif n_cols == 1:
        axs = axs[:, np.newaxis]

    for col, cell_idx in enumerate(page_cells):

        metric_val, metric_label = get_title_metric(data, sort_var, cell_idx, fmt)

        for row, var in enumerate(present_vars):
            ax = axs[row, col]

            centers, tc, err = get_tuning(data, var, cell_idx, fmt)

            color = VAR_COLOR.get(var, 'k')
            ax.plot(centers, tc, '-', color=color, lw=1)

            if err is not None:
                ax.fill_between(centers, tc - err, tc + err,
                                color=color, alpha=0.2, lw=0)

            ax.set_ylim(bottom=0)

            if col == 0:
                ax.set_ylabel(VAR_LABEL.get(var, var),
                              rotation=0, ha='right', va='center', fontsize=7)
            else:
                ax.set_yticklabels([])

            if row == 0:
                sort_label = VAR_LABEL.get(sort_var, sort_var)
                ax.set_title(f'cell {cell_idx}\n{sort_label} {metric_label}={metric_val:.2f}', fontsize=6)

            ax.tick_params(labelsize=5)

    out_path = f'axon_tuning_page{page + 1}_{sort_var}.svg'
    fig.savefig(out_path)
    plt.close(fig)
    print(f'Saved: {out_path}')
