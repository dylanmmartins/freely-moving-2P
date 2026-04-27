
if __package__ is None or __package__ == '':
    import sys as _sys, pathlib as _pl
    _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
    __package__ = 'fm2p'

import os
import sys
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

from .utils.files import read_h5, write_h5


_CELL_TIME_KEYS = {
    'raw_F', 'norm_F', 'raw_Fneu', 'raw_dFF', 'norm_dFF',
    'denoised_dFF', 's2p_spks', 'oasis_spks', 'norm_spikes', 'dFF_transients',
}

_TWOP_TIME_KEYS = {
    'twopT',
    'head_yaw_deg', 'head_yaw', 'head_x', 'head_y',
    'x', 'y', 'lear_x', 'lear_y', 'rear_x', 'rear_y',
    'movement_yaw', 'movement_yaw_deg', 'x_displacement', 'y_displacement',
    'speed',
    'theta_interp', 'phi_interp',
    'egocentric', 'retinocentric', 'pupil_from_head', 'dist_to_center', 'pillar_abs',
    'ltdk_state_vec',
    'upsampled_yaw',
}

_EYE_TRIM_KEYS = {'eyeT_trim', 'theta_trim', 'phi_trim'}

# Raw full-length eye arrays that must be sliced per-recording using
# eyeT_startInd/eyeT_endInd before concatenation.
_EYE_RAW_KEYS = {'eyeT', 'phi', 'theta'}
_EYE_INDEX_KEYS = {'eyeT_startInd', 'eyeT_endInd'}

_FRAME_INDEX_KEYS = {'light_onsets', 'dark_onsets'}

_PER_REC_CELL_KEYS = {'raw_F0', 'norm_F0'}

_SHARED_CELL_KEYS = {'matlab_cellinds', 'cell_x_pix', 'cell_y_pix'}


def _classify(key, val, n_twop, n_cells, n_eye):

    if key in _EYE_RAW_KEYS:
        return 'eye_raw'

    if key in _EYE_INDEX_KEYS:
        return 'skip'

    if isinstance(val, dict):
        return 'scalar'
    if isinstance(val, (str, bytes)):
        return 'scalar'

    arr = val if isinstance(val, np.ndarray) else np.asarray(val)
    if arr.ndim == 0:
        return 'scalar'

    if key == 'spike_times':
        return 'spike_times'

    if key in _FRAME_INDEX_KEYS:
        return 'frame_index'

    if key in _SHARED_CELL_KEYS:
        return 'cell_only'

    if key in _PER_REC_CELL_KEYS:
        return 'prefix'

    if key in _CELL_TIME_KEYS:
        return 'twop_cell_time'

    if key in _TWOP_TIME_KEYS:
        return 'twop_time'

    if key in _EYE_TRIM_KEYS:
        return 'eye_time'

    if key.endswith('_raw') or key == 'imuT_raw':
        return 'prefix'

    if arr.ndim == 2:
        if n_cells > 0 and arr.shape[0] == n_cells and arr.shape[1] == n_twop:
            return 'twop_cell_time'
        return 'prefix'

    if arr.ndim == 1:
        n = len(arr)
        if n == n_twop or n == n_twop - 1:
            return 'twop_time'
        if key.endswith('_eye_interp') or (n_eye > 0 and n == n_eye):
            return 'eye_time'
        if n_cells > 0 and n == n_cells:
            return 'cell_only'
        if key.endswith('_trim'):
            return 'bonsai_trim'
        if key.endswith('_twop_interp'):
            return 'twop_time'
        return 'prefix'

    return 'prefix'


def _make_continuous_time(time_arrays):

    result = []
    offset = 0.0
    for t in time_arrays:
        t = np.asarray(t, dtype=float)
        if len(t) < 2:
            result.append(t - t[0] + offset)
            offset += 1.0
        else:
            dt = float(np.nanmedian(np.diff(t)))
            result.append(t - t[0] + offset)
            offset += float(t[-1] - t[0]) + dt
    return np.concatenate(result)



def merge_recordings(file_paths):

    print(f"Loading {len(file_paths)} recordings ...")
    datas, prefixes = [], []
    for fp in file_paths:
        print(f"  reading  {fp}")
        datas.append(read_h5(fp))
        prefixes.append(Path(fp).parent.name)

    dims = []
    for i, d in enumerate(datas):
        n_twop = int(len(d['twopT']))

        n_cells = 0
        for ck in ('raw_F', 's2p_spks', 'norm_F'):
            if ck in d:
                arr = np.asarray(d[ck])
                if arr.ndim == 2:
                    n_cells = int(arr.shape[0])
                    break

        n_eye = 0
        for ek in ('eyeT_trim', 'theta_trim', 'phi_trim'):
            if ek in d:
                n_eye = int(len(d[ek]))
                break

        dims.append({'n_twop': n_twop, 'n_cells': n_cells, 'n_eye': n_eye})
        print(f"  {prefixes[i]}: {n_twop} twop frames, {n_cells} cells, {n_eye} eye frames")

    cell_counts = [d['n_cells'] for d in dims]
    non_zero = [c for c in cell_counts if c > 0]
    if len(set(non_zero)) > 1:
        raise ValueError(
            "Cell counts differ across recordings — cannot merge twop data.\n"
            + "\n".join(f"  {p}: {c} cells" for p, c in zip(prefixes, cell_counts))
        )
    n_cells_global = non_zero[0] if non_zero else 0

    cumulative_frames = np.cumsum([0] + [d['n_twop'] for d in dims[:-1]], dtype=int)

    twop_durations = []
    for d in datas:
        t = d['twopT']
        dt = float(np.nanmedian(np.diff(t)))
        twop_durations.append(float(t[-1] - t[0]) + dt)
    cumulative_time = np.cumsum([0.0] + twop_durations[:-1])

    first_occurrence = {}  # key -> (recording_index, value)
    for i, d in enumerate(datas):
        for k, v in d.items():
            if k not in first_occurrence:
                first_occurrence[k] = (i, v)

    merged = {}

    for key, (fi, fv) in first_occurrence.items():
        d0 = dims[fi]
        cat = _classify(key, fv, d0['n_twop'], d0['n_cells'], d0['n_eye'])

        if cat == 'twop_cell_time':
            arrays = []
            missing = []
            for i, d in enumerate(datas):
                if key in d:
                    arrays.append(np.asarray(d[key]))
                else:
                    missing.append(prefixes[i])
            if missing:
                print(f"  WARNING: '{key}' absent in {missing}; saving available segments with prefix")
                for i, d in enumerate(datas):
                    if key in d:
                        merged[f"{prefixes[i]}_{key}"] = d[key]
            else:
                merged[key] = np.concatenate(arrays, axis=1)

        elif cat == 'twop_time':
            if key == 'twopT':
                merged[key] = _make_continuous_time([d['twopT'] for d in datas])
            else:
                arrays = []
                for i, d in enumerate(datas):
                    if key in d:
                        arrays.append(np.asarray(d[key]))
                    else:
                        ref_len = len(np.asarray(fv))
                        expected = dims[i]['n_twop'] if ref_len == d0['n_twop'] else dims[i]['n_twop'] - 1
                        arrays.append(np.full(expected, np.nan))
                merged[key] = np.concatenate(arrays)

        elif cat == 'eye_time':
            avail = [(i, d) for i, d in enumerate(datas) if key in d]
            if not avail:
                continue
            if key == 'eyeT_trim':
                merged[key] = _make_continuous_time([d[key] for _, d in avail])
            else:
                merged[key] = np.concatenate([np.asarray(d[key]) for _, d in avail])

        elif cat == 'bonsai_trim':
            avail = [(i, d) for i, d in enumerate(datas) if key in d]
            if not avail:
                continue
            if key == 'imuT_trim':
                merged[key] = _make_continuous_time([d[key] for _, d in avail])
            else:
                merged[key] = np.concatenate([np.asarray(d[key]) for _, d in avail])

        elif cat == 'spike_times':
            parts = []
            for i, d in enumerate(datas):
                if key not in d:
                    continue
                st = np.asarray(d[key], dtype=float).copy()
                mask = ~np.isnan(st)
                st[mask] += cumulative_time[i]
                parts.append(st)
            if parts:
                total_cols = sum(p.shape[1] for p in parts)
                result = np.full((n_cells_global, total_cols), np.nan)
                col = 0
                for p in parts:
                    result[:, col:col + p.shape[1]] = p
                    col += p.shape[1]
                merged[key] = result

        elif cat == 'frame_index':
            arrays = []
            for i, d in enumerate(datas):
                if key in d:
                    arrays.append(np.asarray(d[key], dtype=int) + cumulative_frames[i])
            if arrays:
                merged[key] = np.concatenate(arrays)

        elif cat == 'cell_only':
            merged[key] = fv

        elif cat == 'scalar':
            merged[key] = fv

        elif cat in ('eye_raw', 'skip'):
            pass  # handled below

        else:
            for i, d in enumerate(datas):
                if key in d:
                    merged[f"{prefixes[i]}_{key}"] = d[key]

    # Merge raw eye arrays (eyeT, phi, theta) by slicing each recording's
    # full array with its per-recording start/end indices, then concatenating.
    # eyeT is made monotonically increasing via _make_continuous_time.
    for key in _EYE_RAW_KEYS:
        trimmed = []
        for d in datas:
            if key not in d:
                continue
            arr = np.asarray(d[key], dtype=float)
            si = int(d.get('eyeT_startInd', 0))
            ei = int(d.get('eyeT_endInd', len(arr)))
            trimmed.append(arr[si:ei])
        if not trimmed:
            continue
        if key == 'eyeT':
            merged['eyeT'] = _make_continuous_time(trimmed)
        else:
            merged[key] = np.concatenate(trimmed)

    if 'eyeT' in merged:
        merged['eyeT_startInd'] = np.int64(0)
        merged['eyeT_endInd'] = np.int64(len(merged['eyeT']))

    return merged



class _MergeApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Merge Freely-Moving Recordings")
        self.resizable(True, True)
        self.selected_files = []
        self.result = None
        self._build_ui()

    def _build_ui(self):
        tk.Label(self, text="Selected recordings (in merge order):", anchor='w',
                 padx=10, pady=6).pack(fill='x')

        frame_list = tk.Frame(self, padx=10)
        frame_list.pack(fill='both', expand=True)
        sb = tk.Scrollbar(frame_list)
        sb.pack(side='right', fill='y')
        self.listbox = tk.Listbox(frame_list, width=90, height=12, yscrollcommand=sb.set)
        self.listbox.pack(side='left', fill='both', expand=True)
        sb.config(command=self.listbox.yview)

        frame_btns = tk.Frame(self, padx=10, pady=6)
        frame_btns.pack(fill='x')
        tk.Button(frame_btns, text="Add Recording",
                  command=self._add_file, width=20).pack(side='left', padx=4)
        tk.Button(frame_btns, text="Remove Last",
                  command=self._remove_last, width=14).pack(side='left', padx=4)

        frame_done = tk.Frame(self, padx=10, pady=10) #pady=(0, 12))
        frame_done.pack(fill='x')
        tk.Button(
            frame_done, text="This is the Final Recording",
            command=self._finish,
            bg='#27ae60', fg='white', activebackground='#1e8449',
            width=30, height=2,
        ).pack(side='left', padx=4)
        tk.Button(frame_done, text="Cancel",
                  command=self.destroy, width=10).pack(side='left', padx=4)

    def _add_file(self):
        path = filedialog.askopenfilename(
            title="Select preproc.h5 file",
            filetypes=[
                ("Preprocessed HDF5", "*preproc.h5"),
                ("HDF5 files", "*.h5"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.selected_files.append(path)
            self.listbox.insert('end', f"{len(self.selected_files):2d}.  {path}")

    def _remove_last(self):
        if self.selected_files:
            self.selected_files.pop()
            self.listbox.delete('end')

    def _finish(self):
        if not self.selected_files:
            messagebox.showwarning("No files", "Please add at least one recording first.")
            return
        self.result = list(self.selected_files)
        self.destroy()



def main():

    import matplotlib
    import os
    if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
        matplotlib.use('TkAgg')

    app = _MergeApp()
    app.mainloop()

    if not app.result:
        print("Cancelled.")
        return

    file_paths = app.result

    first_dir = Path(file_paths[0]).parent
    output_dir = first_dir.parent / 'fm0'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'merged_preproc.h5'

    print(f"\nMerging {len(file_paths)} recording(s) -> {output_path}\n")

    try:
        merged = merge_recordings(file_paths)
    except ValueError as exc:
        print(f"\nERROR: {exc}")
        messagebox.showerror("Merge Error", str(exc))
        return

    n_keys = len(merged)
    print(f"\nWriting {n_keys} keys to {output_path} …")
    write_h5(str(output_path), merged)
    print("Done.\n")

    messagebox.showinfo(
        "Merge Complete",
        f"Merged {len(file_paths)} recording(s) -> {n_keys} keys.\n\nSaved to:\n{output_path}",
    )


if __name__ == '__main__':

    main()

