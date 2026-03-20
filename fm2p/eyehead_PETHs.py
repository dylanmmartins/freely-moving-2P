# -*- coding: utf-8 -*-

if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

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
    calc_dff_peth,
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

    # --- spike inference (fMCSI only) -------------------------------------
    # Pass raw F and Fneu so fMCSI computes dF/F internally using its own
    # 8th-percentile baseline — the scale for which its prior was calibrated.
    raw_F    = data['raw_F']
    raw_Fneu = data['raw_Fneu']
    print('  -> Inferring spike times via MCMC deconvolution...')
    spike_times = get_discrete_spike_times(raw_F, raw_Fneu, fs=fs)
    n_spikes = np.array([len(st) for st in spike_times])
    rec_dur  = twopT[-1] - twopT[0]
    mean_rate = np.mean(n_spikes) / rec_dur
    print(f'  -> Spike times: mean {np.mean(n_spikes):.0f} spikes/cell  '
          f'(mean rate {mean_rate:.3f} Hz)  '
          f'time range [{min(st.min() for st in spike_times if len(st)):.1f}, '
          f'{max(st.max() for st in spike_times if len(st)):.1f}] s')

    # --- movement event times (requires IMU) ------------------------------
    if 'imuT_trim' not in data or 'gyro_z_trim' not in data:
        raise KeyError(
            'eyehead_PETHs requires IMU data (imuT_trim, gyro_z_trim). '
            'Run preprocess with cfg["imu"]=True.'
        )

    # --- gyro units check (idea 5) ----------------------------------------
    gyro = data['gyro_z_trim']
    gyro_p99 = float(np.nanpercentile(np.abs(gyro), 99))
    print(f'  -> Gyro check: 99th-percentile |gyro_z_trim| = {gyro_p99:.3f}')
    if gyro_p99 < 5.0:
        print(f'  *** gyro_z_trim looks like rad/s (max ~{gyro_p99:.2f}).')
        print(f'  *** Head-movement thresholds (60/240) are in deg/s — converting.')
        import copy
        data = copy.copy(data)
        data['gyro_z_trim'] = gyro * (180.0 / np.pi)
    elif gyro_p99 > 500.0:
        print(f'  *** gyro_z_trim 99th pctile={gyro_p99:.1f} — confirmed deg/s.')
    else:
        print(f'  *** gyro_z_trim 99th pctile={gyro_p99:.1f} — ambiguous; assuming deg/s.')

    print('  -> Computing eye/head movement event times...')
    saccade_dict = calc_eye_head_movement_times(data)
    for _k in ['gaze_left', 'gaze_right', 'comp_left', 'comp_right']:
        _ev = saccade_dict.get(_k, np.array([]))
        _rng = f'[{_ev.min():.1f}, {_ev.max():.1f}] s' if len(_ev) else 'n/a'
        print(f'     {_k}: {len(_ev)} events  {_rng}')

    # --- PETH parameters --------------------------------------------------
    # bin_size and kde_sigma must be >= 1 frame interval to avoid frame-rate
    # artifacts.  At 7.5 Hz the frame interval is ~133 ms; sub-frame resolution
    # produces periodic zig-zag artefacts at the imaging frame rate.
    frame_interval = 1.0 / fs
    window         = [-0.75, 0.75]       # seconds around event
    bin_size       = max(0.020, frame_interval)      # >= 1 frame
    kde_sigma      = max(0.050, 1.2 * frame_interval)  # >= 1.2 frames
    kde_res        = frame_interval / 4.0            # 4 points per frame
    print(f'  -> PETH params: bin={bin_size*1000:.0f} ms  '
          f'kde_sigma={kde_sigma*1000:.0f} ms  '
          f'kde_res={kde_res*1000:.0f} ms  (frame={frame_interval*1000:.0f} ms)')

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

    # --- timing sanity check (idea 4) ------------------------------------
    all_event_times = np.concatenate([
        saccade_dict[k] for k in event_keys if k in saccade_dict and len(saccade_dict[k])
    ])
    sp_t_min = min((st.min() for st in spike_times if len(st)), default=0.0)
    sp_t_max = max((st.max() for st in spike_times if len(st)), default=0.0)
    ev_t_min = float(all_event_times.min()) if len(all_event_times) else 0.0
    ev_t_max = float(all_event_times.max()) if len(all_event_times) else 0.0
    eyeT_raw = data.get('eyeT', np.array([0.0, 0.0]))
    startInd = int(data.get('eyeT_startInd', 0))
    endInd   = int(data.get('eyeT_endInd', len(eyeT_raw)))
    eyeT_trim_start = float(eyeT_raw[startInd])
    eyeT_trim_end   = float(eyeT_raw[endInd - 1])
    print(f'  -> Timing check:')
    print(f'       twopT range      : [{twopT[0]:.3f}, {twopT[-1]:.3f}] s  (dur {rec_dur:.1f} s)')
    print(f'       spike times      : [{sp_t_min:.3f}, {sp_t_max:.3f}] s')
    print(f'       event times      : [{ev_t_min:.3f}, {ev_t_max:.3f}] s')
    print(f'       eyeT[startInd]   : {eyeT_trim_start:.3f} s  (raw, absolute clock time)')
    print(f'       imuT_trim[0]     : {float(data["imuT_trim"][0]):.3f} s')
    # Check alignment by whether spike times and event times actually overlap
    overlap_start = max(sp_t_min, ev_t_min)
    overlap_end   = min(sp_t_max, ev_t_max)
    overlap_frac  = max(0.0, overlap_end - overlap_start) / max(rec_dur, 1.0)
    if overlap_frac < 0.5:
        print(f'  *** WARNING: spike times and event times overlap by only '
              f'{100*overlap_frac:.0f}% of recording — likely a time-reference mismatch.')
    else:
        print(f'       overlap          : {100*overlap_frac:.0f}% of recording — OK')
        if abs(eyeT_trim_start) > 1.0:
            print(f'       (eyeT uses absolute clock timestamps; zeroing to startInd is correct)')

    # --- dF/F PETHs (diagnostic — no spike inference) ---------------------
    print('  -> Computing dF/F PETHs (diagnostic)...')
    Fc   = raw_F - 0.7 * raw_Fneu
    F0   = np.percentile(Fc, 8, axis=1, keepdims=True)
    dff  = (Fc - F0) / np.abs(F0)

    for key in event_keys:
        events = saccade_dict.get(key, np.array([]))
        if len(events) == 0:
            continue
        mean_dff, stderr_dff, time_dff = calc_dff_peth(
            dff, twopT, events, window=window
        )
        dict_out[f'{key}_dff_mean']   = mean_dff
        dict_out[f'{key}_dff_stderr'] = stderr_dff
        dict_out[f'{key}_dff_time']   = time_dff

    # --- norm_spikes PETHs (frame-rate spike rate, bypasses fMCSI) ---------
    if 'norm_spikes' in data:
        print('  -> Computing norm_spikes PETHs...')
        ns = data['norm_spikes']   # (n_cells, n_frames), already in spikes/s units
        for key in event_keys:
            events = saccade_dict.get(key, np.array([]))
            if len(events) == 0:
                continue
            mean_ns, stderr_ns, time_ns = calc_dff_peth(
                ns, twopT, events, window=window
            )
            dict_out[f'{key}_ns_mean']   = mean_ns
            dict_out[f'{key}_ns_stderr'] = stderr_ns
            dict_out[f'{key}_ns_time']   = time_ns

    # --- save HDF5 --------------------------------------------------------
    basedir  = os.path.dirname(preproc_path)
    basename = os.path.splitext(os.path.basename(preproc_path))[0]
    savepath = os.path.join(basedir, f'{basename}_eyehead_PETHs.h5')
    print(f'  -> Writing {savepath}')
    write_h5(savepath, dict_out)

    # --- save histogram PDF -----------------------------------------------
    pdf_path = os.path.join(basedir, f'{basename}_eyehead_hist_PETHs.pdf')
    print(f'  -> Writing {pdf_path}')
    _save_peth_pdf(dict_out, pdf_path, title=basename, peth_type='hist')

    # --- save KDE PDF -----------------------------------------------------
    kde_pdf_path = os.path.join(basedir, f'{basename}_eyehead_kde_PETHs.pdf')
    print(f'  -> Writing {kde_pdf_path}')
    _save_peth_pdf(dict_out, kde_pdf_path, title=basename, peth_type='kde')

    # --- save dF/F diagnostic PDF -----------------------------------------
    diag_pdf_path = os.path.join(basedir, f'{basename}_eyehead_dff_diagnostic.pdf')
    print(f'  -> Writing {diag_pdf_path}')
    _save_peth_pdf(dict_out, diag_pdf_path, title=basename, peth_type='dff')

    # --- save norm_spikes PDF (if available) ------------------------------
    if 'norm_spikes' in data:
        ns_pdf_path = os.path.join(basedir, f'{basename}_eyehead_ns_PETHs.pdf')
        print(f'  -> Writing {ns_pdf_path}')
        _save_peth_pdf(dict_out, ns_pdf_path, title=basename, peth_type='ns')

    return savepath


def _peak_modulation(peth, time_axis):
    """Peak absolute change from pre-event baseline (-0.5 to -0.1 s), in spikes/s."""
    baseline_mask = (time_axis >= -0.5) & (time_axis < -0.1)
    if not np.any(baseline_mask):
        baseline_mask = time_axis < 0
    baseline = np.mean(peth[baseline_mask])
    return float(np.max(np.abs(peth - baseline)))


def _save_peth_pdf(dict_out, pdf_path, title='', peth_type='hist'):
    """Write a multi-page PDF of per-cell PETHs in spikes/s, sorted by modulation.

    Parameters
    ----------
    peth_type : 'hist' or 'kde'
        Which PETH to plot.
    """
    event_keys = ['gaze_left', 'gaze_right', 'comp_left', 'comp_right']
    colors = {'gaze_left': 'steelblue', 'gaze_right': 'tomato',
              'comp_left': 'steelblue',  'comp_right': 'tomato'}

    mean_key_tmpl   = '{key}_' + peth_type + '_mean'
    stderr_key_tmpl = '{key}_' + peth_type + '_stderr'
    time_key_tmpl   = '{key}_' + peth_type + '_time'

    # determine n_cells from the first available PETH
    n_cells = None
    for key in event_keys:
        k = mean_key_tmpl.format(key=key)
        if k in dict_out:
            n_cells = dict_out[k].shape[0]
            break
    if n_cells is None:
        return

    n_gl = len(dict_out.get('gaze_left',  []))
    n_gr = len(dict_out.get('gaze_right', []))
    n_cl = len(dict_out.get('comp_left',  []))
    n_cr = len(dict_out.get('comp_right', []))
    type_label = {'hist': 'Histogram', 'kde': 'KDE', 'dff': 'dF/F', 'ns': 'Norm spikes'}.get(peth_type, peth_type)
    units      = {'hist': 'spikes/s', 'kde': 'spikes/s', 'dff': 'dF/F', 'ns': 'spikes/s'}.get(peth_type, peth_type)
    header = (f'{title}  [{type_label}]   |   '
              f'gaze L={n_gl} R={n_gr}   comp L={n_cl} R={n_cr}')

    # --- sort cells by peak modulation (max |Δrate| vs baseline) ---------
    mod_scores = np.zeros(n_cells)
    for c in range(n_cells):
        best = 0.0
        for key in ['gaze_left', 'gaze_right']:
            mk = mean_key_tmpl.format(key=key)
            tk = time_key_tmpl.format(key=key)
            if mk in dict_out:
                best = max(best, _peak_modulation(dict_out[mk][c], dict_out[tk]))
        mod_scores[c] = best

    sort_order = np.argsort(mod_scores)[::-1]  # most modulated first

    cells_per_page = 10
    n_pages = int(np.ceil(n_cells / cells_per_page))

    with PdfPages(pdf_path) as pdf:
        # --- page 0: population mean ----------------------------------------
        fig, axs_pop = plt.subplots(2, 2, figsize=(8.5, 6))
        fig.suptitle(f'{header}\n[population mean ± SEM, n={n_cells} cells]', fontsize=8)
        ax_map = {'gaze_left':  axs_pop[0, 0], 'gaze_right': axs_pop[0, 1],
                  'comp_left':  axs_pop[1, 0], 'comp_right': axs_pop[1, 1]}
        for key, ax in ax_map.items():
            mk = mean_key_tmpl.format(key=key)
            sk = stderr_key_tmpl.format(key=key)
            tk = time_key_tmpl.format(key=key)
            if mk in dict_out:
                t  = dict_out[tk]
                mn = np.mean(dict_out[mk], axis=0)
                se = np.std(dict_out[mk], axis=0) / np.sqrt(n_cells)
                ax.plot(t, mn, color=colors[key], linewidth=1.2)
                ax.fill_between(t, mn - se, mn + se,
                                color=colors[key], alpha=0.25, linewidth=0)
            ax.axvline(0, color='k', linestyle='--', linewidth=0.7)
            ax.set_title(key.replace('_', ' '), fontsize=8)
            ax.set_xlabel('Time (s)', fontsize=7)
            ax.set_ylabel(units, fontsize=7)
            ax.tick_params(labelsize=6)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

        for page in range(n_pages):
            start = page * cells_per_page
            end   = min(start + cells_per_page, n_cells)
            page_cells = sort_order[start:end]

            fig, axs = plt.subplots(cells_per_page, 2, figsize=(8.5, 11))
            fig.suptitle(header, fontsize=8)

            for row, c in enumerate(page_cells):
                ax_g = axs[row, 0]
                ax_c = axs[row, 1]

                for key, ax in [('gaze_left',  ax_g), ('gaze_right', ax_g),
                                 ('comp_left',  ax_c), ('comp_right', ax_c)]:
                    mk = mean_key_tmpl.format(key=key)
                    sk = stderr_key_tmpl.format(key=key)
                    tk = time_key_tmpl.format(key=key)
                    if mk in dict_out:
                        t    = dict_out[tk]
                        mn   = dict_out[mk][c]
                        lbl  = key.split('_')[1]
                        ax.plot(t, mn, color=colors[key], linewidth=0.8, label=lbl)
                        if sk in dict_out:
                            se = dict_out[sk][c]
                            ax.fill_between(t, mn - se, mn + se,
                                            color=colors[key], alpha=0.2, linewidth=0)

                for ax in (ax_g, ax_c):
                    ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
                    mi_str = f'{mod_scores[c]:.3f}'
                    ax.set_ylabel(f'C{c}\n({mi_str})', fontsize=5)
                    ax.tick_params(labelsize=5)
                    if row < len(page_cells) - 1:
                        ax.tick_params(labelbottom=False)
                    else:
                        ax.set_xlabel('Time (s)', fontsize=6)

            axs[0, 0].legend(fontsize=5, loc='upper right')
            axs[0, 0].set_title(f'Gaze shifts ({type_label}, {units})', fontsize=7)
            axs[0, 1].set_title(f'Compensatory ({type_label}, {units})', fontsize=7)

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

    # eyehead_PETHs(
    #     '/home/dylan/Storage/freely_moving_data/_V1PPC/cohort02_recordings/cohort02_recordings/251016_DMM_DMM061_pos18/fm1/251016_DMM_DMM061_fm_01_preproc.h5'
    # )
