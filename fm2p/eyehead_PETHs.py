# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .utils.files import read_h5, write_h5
from .utils.gui_funcs import select_file
from .utils.PETH import (
    get_discrete_spike_times,
    calc_eye_head_movement_times,
    calc_binned_PETH,
    calc_cont_PETH,
)


def eyehead_PETHs(preproc_path=None):
    """Calculate PETHs around gaze shifts and compensatory eye movements.

    Spike times are inferred from the dF/F trace using MCMC deconvolution
    (fMCSI).  PETHs are computed in two forms:

    * **Histogram** - binned spike counts divided by bin width, in spikes/s.
    * **KDE** - Gaussian kernel density estimate, normalised to spikes/s.

    Results are written to an HDF5 file in the same directory as the input.

    Parameters
    ----------
    preproc_path : str or None
        Path to a ``*_preproc.h5`` file produced by ``fm2p.preprocess()``.
        Opens a file-chooser dialog when None.

    Returns
    -------
    savepath : str
        Path of the HDF5 file that was written.
    """
    if preproc_path is None:
        preproc_path = select_file(
            'Select preprocessing HDF file.',
            filetypes=[('HDF', '.h5')]
        )
    data = read_h5(preproc_path)

    # --- frame rate -------------------------------------------------------
    twopT = data['twopT']
    fs = float(1.0 / np.nanmedian(np.diff(twopT)))

    # --- dF/F trace -------------------------------------------------------
    dFF = data['norm_dFF']

    # --- spike inference (fMCSI only) -------------------------------------
    print('  -> Inferring spike times via MCMC deconvolution...')
    spike_times = get_discrete_spike_times(dFF, fs=fs)

    # --- movement event times (requires IMU) ------------------------------
    if 'imuT_trim' not in data or 'gyro_z_trim' not in data:
        raise KeyError(
            'eyehead_PETHs requires IMU data (imuT_trim, gyro_z_trim). '
            'Run preprocess with cfg["imu"]=True.'
        )

    print('  -> Computing eye/head movement event times...')
    saccade_dict = calc_eye_head_movement_times(data)

    # --- PETH parameters --------------------------------------------------
    window       = [-0.75, 0.75]   # seconds around event
    bin_size     = 0.020           # 20 ms histogram bins
    kde_sigma    = 0.040           # 40 ms Gaussian sigma
    kde_res      = 0.005           # 5 ms KDE resolution

    event_keys = ['gaze_left', 'gaze_right', 'comp_left', 'comp_right']

    # --- pack spike times for storage (NaN-padded 2-D array) --------------
    n_cells  = len(spike_times)
    max_sp   = max((len(st) for st in spike_times), default=1)
    sp_array = np.full((n_cells, max_sp), np.nan, dtype=np.float64)
    for i, st in enumerate(spike_times):
        sp_array[i, :len(st)] = st

    dict_out = {
        'spike_times':    sp_array,
        'window':         np.array(window),
        'hist_bin_size':  np.float64(bin_size),
        'kde_sigma':      np.float64(kde_sigma),
        'kde_resolution': np.float64(kde_res),
    }

    # store the event-time vectors too
    for key in event_keys:
        if key in saccade_dict:
            dict_out[key] = saccade_dict[key]

    # --- compute PETHs ----------------------------------------------------
    for key in event_keys:
        events = saccade_dict.get(key, np.array([]))
        if len(events) == 0:
            print(f'  -> {key}: no events, skipping.')
            continue

        print(f'  -> {key}: {len(events)} events.')

        # histogram PETH (spikes/s)
        mean_hist, stderr_hist, time_hist = calc_binned_PETH(
            spike_times, events, window=window, bin_size=bin_size
        )
        dict_out[f'{key}_hist_mean']   = mean_hist    # (n_cells, n_bins)
        dict_out[f'{key}_hist_stderr'] = stderr_hist
        dict_out[f'{key}_hist_time']   = time_hist    # (n_bins,)

        # KDE PETH (spikes/s)
        mean_kde, stderr_kde, time_kde = calc_cont_PETH(
            spike_times, events, window=window,
            sigma=kde_sigma, resolution=kde_res
        )
        dict_out[f'{key}_kde_mean']   = mean_kde      # (n_cells, n_points)
        dict_out[f'{key}_kde_stderr'] = stderr_kde
        dict_out[f'{key}_kde_time']   = time_kde      # (n_points,)

    # --- save HDF5 --------------------------------------------------------
    basedir  = os.path.dirname(preproc_path)
    basename = os.path.splitext(os.path.basename(preproc_path))[0]
    savepath = os.path.join(basedir, f'{basename}_eyehead_PETHs.h5')
    print(f'  -> Writing {savepath}')
    write_h5(savepath, dict_out)

    # --- save PDF ---------------------------------------------------------
    pdf_path = os.path.join(basedir, f'{basename}_eyehead_PETHs.pdf')
    print(f'  -> Writing {pdf_path}')
    _save_peth_pdf(dict_out, pdf_path, title=basename)

    return savepath


def _save_peth_pdf(dict_out, pdf_path, title=''):
    """Write a multi-page PDF of per-cell PETHs (histogram, spikes/s)."""
    event_keys = ['gaze_left', 'gaze_right', 'comp_left', 'comp_right']
    colors = {'gaze_left': 'steelblue', 'gaze_right': 'tomato',
              'comp_left': 'steelblue',  'comp_right': 'tomato'}

    # determine n_cells from the first available PETH
    n_cells = None
    for key in event_keys:
        k = f'{key}_hist_mean'
        if k in dict_out:
            n_cells = dict_out[k].shape[0]
            break
    if n_cells is None:
        return

    n_gl = len(dict_out.get('gaze_left',  []))
    n_gr = len(dict_out.get('gaze_right', []))
    n_cl = len(dict_out.get('comp_left',  []))
    n_cr = len(dict_out.get('comp_right', []))
    header = (f'{title}   |   '
              f'gaze L={n_gl} R={n_gr}   comp L={n_cl} R={n_cr}')

    cells_per_page = 10
    n_pages = int(np.ceil(n_cells / cells_per_page))

    with PdfPages(pdf_path) as pdf:
        for page in range(n_pages):
            start = page * cells_per_page
            end   = min(start + cells_per_page, n_cells)
            page_cells = range(start, end)

            fig, axs = plt.subplots(cells_per_page, 2, figsize=(8.5, 11))
            fig.suptitle(header, fontsize=8)

            for row, c in enumerate(page_cells):
                ax_g = axs[row, 0]
                ax_c = axs[row, 1]

                for key, ax in [('gaze_left',  ax_g), ('gaze_right', ax_g),
                                 ('comp_left',  ax_c), ('comp_right', ax_c)]:
                    mk = f'{key}_hist_mean'
                    tk = f'{key}_hist_time'
                    if mk in dict_out:
                        lbl = key.split('_')[1]
                        ax.plot(dict_out[tk], dict_out[mk][c],
                                color=colors[key], linewidth=0.8, label=lbl)

                for ax in (ax_g, ax_c):
                    ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
                    ax.set_ylabel(f'C{c}', fontsize=6)
                    ax.tick_params(labelsize=5)
                    if row < len(page_cells) - 1:
                        ax.tick_params(labelbottom=False)
                    else:
                        ax.set_xlabel('Time (s)', fontsize=6)

            ax_g.legend(fontsize=5, loc='upper right')

            axs[0, 0].set_title('Gaze shifts', fontsize=7)
            axs[0, 1].set_title('Compensatory', fontsize=7)

            # hide unused rows
            for row in range(len(page_cells), cells_per_page):
                axs[row, 0].axis('off')
                axs[row, 1].axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)


def batch_eyehead_PETHs(root_dir):
    """Run eyehead_PETHs on every *preproc.h5 file under root_dir.

    Skips recordings that already have a ``*_eyehead_PETHs.h5`` output.
    Results (HDF5 + PDF) are saved alongside each preproc file.

    Parameters
    ----------
    root_dir : str
        Top-level directory to search recursively.
    """
    pattern = os.path.join(root_dir, '**', '*preproc.h5')
    candidates = sorted(glob.glob(pattern, recursive=True))

    # exclude files that are themselves eyehead outputs
    candidates = [p for p in candidates if '_eyehead_PETHs' not in p]

    print(f'Found {len(candidates)} preproc file(s) under {root_dir}')

    done, skipped, failed = 0, 0, 0
    for path in candidates:
        basedir  = os.path.dirname(path)
        basename = os.path.splitext(os.path.basename(path))[0]
        out_h5   = os.path.join(basedir, f'{basename}_eyehead_PETHs.h5')

        if os.path.exists(out_h5):
            print(f'  [skip] {os.path.basename(path)} (output exists)')
            skipped += 1
            continue

        print(f'\n[{done+1}/{len(candidates)-skipped}] {path}')
        try:
            eyehead_PETHs(path)
            done += 1
        except KeyError as e:
            print(f'  [skip] missing key: {e}')
            skipped += 1
        except Exception as e:
            print(f'  [error] {e}')
            failed += 1

    print(f'\nDone. processed={done}  skipped={skipped}  errors={failed}')


if __name__ == '__main__':
    
    # batch_eyehead_PETHs('/home/dylan/Storage/freely_moving_data/_V1PPC')

    eyehead_PETHs(
        '/home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings/cohort02_recordings/251020_DMM_DMM056_pos08/fm1/251020_DMM_DMM056_fm_01_preproc.h5'
    )
