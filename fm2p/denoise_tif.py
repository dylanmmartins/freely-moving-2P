# -*- coding: utf-8 -*-
"""
Subtract resonance scanner noise in mini2p image stacks.

Functions
---------
denoise_tif(tif_path=None, ret=False)
    Remove noise added into mini2p image stack by resonance scanner.

Author: DMM, 2025
"""


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import tifffile
import argparse
import os
from collections import deque
from matplotlib.backends.backend_pdf import PdfPages

import fm2p


def denoise_tif_1d(tif_path=None, ret=False, saveRA=False):
    """ Remove noise added into mini2p image stack by resonance scanner.

    The noise appears as hazy vertical banding which sweeps slowly along the x axis
    (they are not in static positions, and there are ~10 overlapping bands in the
    image for any given frame. they move both leftwards and rightwards. If ret is true,
    the function will return the image stack with a short (3 frame) rolling average applied.

    Processes the stack in two streaming passes so that only a handful of frames
    are in memory at any one time. Peak RAM use is O(N_frames * frame_width) for
    the 1-D noise signatures, plus a small per-frame working buffer.

    Parameters
    ----------
    tif_path : str
        Path to the tiff file to be denoised. If None, a file dialog will be opened
        to select the file.
    ret : bool
        Ignored (kept for API compatibility). The denoised stack is always written
        to disk rather than returned, to keep memory use low.
    saveRA : bool
        If True, write two additional tifs: one with a 3-frame rolling average (SRA)
        and one with a 12-frame rolling average (LRA). Default is False.
    """

    if tif_path is None:
        tif_path = fm2p.select_file(
            'Select tif stack.',
            filetypes=[('TIF', '*.tif'),('TIF','*.tiff'),]
        )

    print('Denoising {}'.format(tif_path))

    base_path = os.path.split(tif_path)[0]
    tif_name = os.path.split(tif_path)[1]
    tif_name_noext = os.path.splitext(tif_name)[0]
    pdf = PdfPages(os.path.join(base_path, 'denoising_figs.pdf'))

    nPix = 50

    # walk through pages once to build the per-frame 1-D noise
    # signature (mean of top/bottom nPix rows).  Only one frame is in RAM
    # at a time; the result is tiny: (N_frames, W) float32.
    print('Pass 1: computing noise signatures...')
    with tifffile.TiffFile(tif_path) as tif:
        n_frames = len(tif.pages)
        first_frame = tif.pages[0].asarray()
        H, W = first_frame.shape
        diag_f = min(500, n_frames - 1)

        mean_band = np.zeros((n_frames, W), dtype=np.float32)
        for f, page in enumerate(tqdm(tif.pages, desc='Reading edge pixels')):
            frame = page.asarray()
            band = np.concatenate([frame[:nPix, :], frame[-nPix:, :]], axis=0)
            mean_band[f] = band.mean(axis=0)

    fig = plt.figure(figsize=(6, 6), dpi=300)
    plt.imshow(mean_band, aspect='auto', cmap='gray')
    plt.xlabel('y pixels')
    plt.ylabel('time (frames)')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    print('Calculating noise pattern.')
    noise_sigs = np.empty_like(mean_band)
    for f in tqdm(range(n_frames)):
        noise_sigs[f] = fm2p.convfilt(mean_band[f], 5)
    del mean_band

    # stream through pages again, subtract noise, write output.
    # noise_sigs[f] has shape (W,) and broadcasts to (H, W) automatically.
    meanF = np.zeros(n_frames, dtype=np.float64)
    meanP = np.zeros(n_frames, dtype=np.float64)
    diag_raw = diag_noise_frame = diag_denoised = None
    noise_len = n_frames

    def process_frame(f, raw_frame):
        nonlocal diag_raw, diag_noise_frame, diag_denoised
        raw_f32 = raw_frame.astype(np.float32)
        # noise_sigs[f] is 1-D (W,); broadcasting subtracts it from every row
        denoised = np.clip(raw_f32 - noise_sigs[f] + 16, 0, 65535).astype(np.uint16)
        meanF[f] = denoised.mean()
        meanP[f] = noise_sigs[f].mean()
        if f == diag_f:
            diag_raw = raw_frame.copy()
            diag_noise_frame = np.broadcast_to(noise_sigs[f], (H, W)).copy()
            diag_denoised = denoised.copy()
        return denoised

    if not saveRA:
        savefilename = os.path.join(base_path, '{}_denoised.tif'.format(tif_name_noext))
        print('Pass 2: denoising → {}'.format(savefilename))
        with tifffile.TiffFile(tif_path) as tif, \
             tifffile.TiffWriter(savefilename, bigtiff=True) as writer:
            for f, page in enumerate(tqdm(tif.pages, desc='Denoising')):
                denoised = process_frame(f, page.asarray())
                writer.write(denoised, photometric='minisblack', contiguous=True)

    else:
        # Rolling-average outputs: keep a small sliding window deque for each.
        # Memory cost: (sra_window + lra_window) * H * W * 4 bytes — a few MB.
        sra_window, lra_window = 3, 12
        full_numF = n_frames
        sra_len = n_frames - sra_window + 1
        lra_len = n_frames - lra_window + 1

        s_savefilename = os.path.join(base_path, '{}_denoised_SRA.tif'.format(tif_name_noext))
        l_savefilename = os.path.join(base_path, '{}_denoised_LRA.tif'.format(tif_name_noext))
        print('Pass 2: denoising → SRA and LRA tifs...')

        sra_buf = deque(maxlen=sra_window)
        lra_buf = deque(maxlen=lra_window)

        with tifffile.TiffFile(tif_path) as tif, \
             tifffile.TiffWriter(s_savefilename, bigtiff=True) as sra_writer, \
             tifffile.TiffWriter(l_savefilename, bigtiff=True) as lra_writer:
            for f, page in enumerate(tqdm(tif.pages, desc='Denoising')):
                denoised = process_frame(f, page.asarray())
                d_f32 = denoised.astype(np.float32)
                sra_buf.append(d_f32)
                lra_buf.append(d_f32)

                # Write one SRA frame once the window is full; deque stays full thereafter
                if len(sra_buf) == sra_window:
                    sra_frame = np.stack(sra_buf).mean(axis=0).clip(0, 65535).astype(np.uint16)
                    sra_writer.write(sra_frame, photometric='minisblack', contiguous=True)

                if len(lra_buf) == lra_window:
                    lra_frame = np.stack(lra_buf).mean(axis=0).clip(0, 65535).astype(np.uint16)
                    lra_writer.write(lra_frame, photometric='minisblack', contiguous=True)

        sra_adjust = int((noise_len - sra_len) / 2)
        lra_adjust = int((noise_len - lra_len) / 2)
        frame_note = (
            'The full tif stack had {} frames. The denoised tif stack with a short running average '
            'has {} frames, and the one with a long running average has {} frames. When aligning '
            'the denoised stacks to other data streams, subtract diff/2 from the start and end. '
            'Adjust SRA by {} and LRA by {}.'
        ).format(full_numF, sra_len, lra_len, sra_adjust, lra_adjust)
        txt_savepath = os.path.join(base_path, 'note_on_denoised_tif_dims.txt')
        with open(txt_savepath, 'w') as file:
            file.write(frame_note)
        print(frame_note)

    # Diagnostic figure: one example frame captured during pass 2
    if diag_raw is not None:
        fig, axes = plt.subplots(1, 3, figsize=(5.5, 3), dpi=300)
        axes[0].imshow(diag_raw, cmap='gray', vmin=0, vmax=200)
        axes[1].imshow(diag_noise_frame, cmap='gray',
                       vmin=diag_noise_frame.min(), vmax=diag_noise_frame.max())
        axes[2].imshow(diag_denoised, cmap='gray', vmin=0, vmax=200)
        for ax in axes:
            ax.axis('off')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close()

    print('Calculating diagnostic statistics across recording.')
    fig, [ax1, ax2] = plt.subplots(1, 2, dpi=300, figsize=(8, 2.5))
    ax1.plot(meanF, color='k', lw=1)
    ax2.plot(meanP, color='k', lw=1)
    ax1.set_xlabel('frames')
    ax2.set_xlabel('frames')
    ax1.set_ylabel('frame mean pixel value')
    ax2.set_ylabel('frame mean pixel value')
    ax1.set_ylim([np.percentile(meanF, 0.1), np.percentile(meanF, 99.9)])
    ax2.set_ylim([np.percentile(meanP, 0.1), np.percentile(meanP, 99.9)])
    ax1.set_title('noise-corrected stack')
    ax2.set_title('putative noise pattern')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close()

    pdf.close()


def make_denoise_diagnostic_video(ra_img, noise_pattern, ra_newimg, vid_save_path, startF, endF):
    """ Make a diagnostic video of the array.

    Parameters
    ----------
    ra_img : np.ndarray
        Image stack (not denoised) image with short rolling average applied.
    noise_pattern : np.ndarray
        Noise pattern with the same dimensions as the ra_img.
    ra_newimg : np.ndarray
        Denoised image stack with short rolling average applied.
    vid_save_path : str
        Video save path.
    startF : int
        Starting frame.
    endF : int
        Ending frame.
    """

    # start/end crop value to align noise pattern with smoothed image stacks
    # important to do the smoothing after noise is subtracted instead of before!
    startEndFCrop = int((np.size(noise_pattern,0)-np.size(ra_img,0))/2)

    ra_img = fm2p.rolling_average(ra_img, 7)
    ra_newimg = fm2p.rolling_average(ra_newimg, 7)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(vid_save_path, fourcc, (7.5*8), (1650, 900))

    for f in tqdm(np.arange(startF, endF)):

        fig, [ax1,ax2,ax3] = plt.subplots(1,3, figsize=(5.5,3), dpi=300)
        ax1.imshow(ra_img[f,:,:], cmap='gray', vmin=0, vmax=200)
        ax2.imshow(noise_pattern[f+startEndFCrop,:,:], cmap='gray', vmin=-10, vmax=120)
        ax3.imshow(ra_newimg[f,:,:], cmap='gray', vmin=0, vmax=200)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        fig.suptitle('frame {}'.format(f))
        fig.tight_layout()

        fig.canvas.draw()
        frame_as_array = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        frame_as_array = frame_as_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        img = cv2.cvtColor(frame_as_array, cv2.COLOR_RGB2BGR)
        out_vid.write(img.astype('uint8'))

    out_vid.release()


def denoise_tif_2d(tif_path=None, ret=False, saveRA=False):
    """ Remove noise added into mini2p image stack by resonance scanner.

    The noise appears as hazy vertical banding which sweeps slowly along the x axis
    (they are not in static positions, and there are ~10 overlapping bands in the
    image for any given frame. they move both leftwards and rightwards. If ret is true,
    the function will return the image stack with a short (3 frame) rolling average applied.

    Processes the stack in two streaming passes so that only a handful of frames
    are in memory at any one time. Peak RAM use is O(N_frames * (frame_width +
    frame_height)) for the 1-D noise signatures, plus a small per-frame buffer.

    Parameters
    ----------
    tif_path : str
        Path to the tiff file to be denoised. If None, a file dialog will be opened
        to select the file.
    ret : bool
        Ignored (kept for API compatibility). The denoised stack is always written
        to disk rather than returned, to keep memory use low.
    saveRA : bool
        If True, write two additional tifs: one with a 3-frame rolling average (SRA)
        and one with a 12-frame rolling average (LRA). Default is False.
    """

    if tif_path is None:
        tif_path = fm2p.select_file(
            'Select tif stack.',
            filetypes=[('TIF', '*.tif'),('TIF','*.tiff'),]
        )

    print('Denoising {}'.format(tif_path))

    base_path = os.path.split(tif_path)[0]
    tif_name = os.path.split(tif_path)[1]
    tif_name_noext = os.path.splitext(tif_name)[0]
    pdf = PdfPages(os.path.join(base_path, 'denoising_figs.pdf'))

    nPix = 50

    # walk through pages once to build per-frame 1-D noise
    # signatures for both axes.  Memory: (N_frames, W) + (N_frames, H)
    # float32 — tens of MB even for very long recordings.
    print('Pass 1: computing noise signatures...')
    with tifffile.TiffFile(tif_path) as tif:
        n_frames = len(tif.pages)
        first_frame = tif.pages[0].asarray()
        H, W = first_frame.shape
        diag_f = min(500, n_frames - 1)

        mean_band_H = np.zeros((n_frames, W), dtype=np.float32)
        mean_band_V = np.zeros((n_frames, H), dtype=np.float32)

        for f, page in enumerate(tqdm(tif.pages, desc='Reading edge pixels')):
            frame = page.asarray()
            # Horizontal bands: average of top + bottom nPix rows → shape (W,)
            band_H = np.concatenate([frame[:nPix, :], frame[-nPix:, :]], axis=0)
            mean_band_H[f] = band_H.mean(axis=0)
            # Vertical bands: average of left + right nPix columns → shape (H,)
            band_V = np.concatenate([frame[:, :nPix], frame[:, -nPix:]], axis=1)
            mean_band_V[f] = band_V.mean(axis=1)

    fig = plt.figure(figsize=(6, 6), dpi=300)
    plt.imshow(mean_band_H, aspect='auto', cmap='gray')
    plt.xlabel('y pixels')
    plt.ylabel('time (frames)')
    plt.title('horizontal banded block')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    fig = plt.figure(figsize=(6, 6), dpi=300)
    plt.imshow(mean_band_V, aspect='auto', cmap='gray')
    plt.xlabel('y pixels')
    plt.ylabel('time (frames)')
    plt.title('vertical banded block')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    print('Calculating noise pattern.')
    noise_sigs_H = np.empty_like(mean_band_H)
    noise_sigs_V = np.empty_like(mean_band_V)
    for f in tqdm(range(n_frames)):
        noise_sigs_H[f] = fm2p.convfilt(mean_band_H[f], 5)
        noise_sigs_V[f] = fm2p.convfilt(mean_band_V[f], 5)
    del mean_band_H, mean_band_V

    # stream through pages, compute 2-D noise as the outer sum
    # of the two 1-D signatures, subtract, and write output TIFs.
    # noise[i,j] = noise_sigs_H[f][j] + noise_sigs_V[f][i]
    meanF = np.zeros(n_frames, dtype=np.float64)
    meanP = np.zeros(n_frames, dtype=np.float64)
    diag_raw = diag_noise_frame = diag_denoised = None
    noise_len = n_frames

    def process_frame(f, raw_frame):
        nonlocal diag_raw, diag_noise_frame, diag_denoised
        # Outer sum: (1,W) + (H,1) → (H,W); no temporary full-stack arrays
        noise = noise_sigs_H[f][np.newaxis, :] + noise_sigs_V[f][:, np.newaxis]
        denoised = np.clip(raw_frame.astype(np.float32) - noise + 16, 0, 65535).astype(np.uint16)
        meanF[f] = denoised.mean()
        # mean of outer sum = mean_H + mean_V
        meanP[f] = noise_sigs_H[f].mean() + noise_sigs_V[f].mean()
        if f == diag_f:
            diag_raw = raw_frame.copy()
            diag_noise_frame = noise.copy()
            diag_denoised = denoised.copy()
        return denoised, noise

    noise_savefilename = os.path.join(base_path, '{}_noise_pattern.tif'.format(tif_name_noext))

    if not saveRA:
        savefilename = os.path.join(base_path, '{}_denoised.tif'.format(tif_name_noext))
        print('Pass 2: denoising → {} and {}'.format(savefilename, noise_savefilename))
        with tifffile.TiffFile(tif_path) as tif, \
             tifffile.TiffWriter(savefilename, bigtiff=True) as writer, \
             tifffile.TiffWriter(noise_savefilename, bigtiff=True) as noise_writer:
            for f, page in enumerate(tqdm(tif.pages, desc='Denoising')):
                denoised, noise = process_frame(f, page.asarray())
                noise_clipped = np.clip(noise, 0, 65535).astype(np.uint16)
                writer.write(denoised, photometric='minisblack', contiguous=True)
                noise_writer.write(noise_clipped, photometric='minisblack', contiguous=True)

    else:
        sra_window, lra_window = 3, 12
        full_numF = n_frames
        sra_len = n_frames - sra_window + 1
        lra_len = n_frames - lra_window + 1

        s_savefilename = os.path.join(base_path, '{}_denoised_SRA.tif'.format(tif_name_noext))
        l_savefilename = os.path.join(base_path, '{}_denoised_LRA.tif'.format(tif_name_noext))
        print('Pass 2: denoising → SRA, LRA, and noise pattern tifs...')

        sra_buf = deque(maxlen=sra_window)
        lra_buf = deque(maxlen=lra_window)

        with tifffile.TiffFile(tif_path) as tif, \
             tifffile.TiffWriter(s_savefilename, bigtiff=True) as sra_writer, \
             tifffile.TiffWriter(l_savefilename, bigtiff=True) as lra_writer, \
             tifffile.TiffWriter(noise_savefilename, bigtiff=True) as noise_writer:
            for f, page in enumerate(tqdm(tif.pages, desc='Denoising')):
                denoised, noise = process_frame(f, page.asarray())
                noise_clipped = np.clip(noise, 0, 65535).astype(np.uint16)
                noise_writer.write(noise_clipped, photometric='minisblack', contiguous=True)

                d_f32 = denoised.astype(np.float32)
                sra_buf.append(d_f32)
                lra_buf.append(d_f32)

                if len(sra_buf) == sra_window:
                    sra_frame = np.stack(sra_buf).mean(axis=0).clip(0, 65535).astype(np.uint16)
                    sra_writer.write(sra_frame, photometric='minisblack', contiguous=True)

                if len(lra_buf) == lra_window:
                    lra_frame = np.stack(lra_buf).mean(axis=0).clip(0, 65535).astype(np.uint16)
                    lra_writer.write(lra_frame, photometric='minisblack', contiguous=True)

        sra_adjust = int((noise_len - sra_len) / 2)
        lra_adjust = int((noise_len - lra_len) / 2)
        frame_note = (
            'The full tif stack had {} frames. The denoised tif stack with a short running average '
            'has {} frames, and the one with a long running average has {} frames. When aligning '
            'the denoised stacks to other data streams, subtract diff/2 from the start and end. '
            'Adjust SRA by {} and LRA by {}.'
        ).format(full_numF, sra_len, lra_len, sra_adjust, lra_adjust)
        txt_savepath = os.path.join(base_path, 'note_on_denoised_tif_dims.txt')
        with open(txt_savepath, 'w') as file:
            file.write(frame_note)
        print(frame_note)

    # Diagnostic figure: one example frame captured during pass 2
    if diag_raw is not None:
        fig, axes = plt.subplots(1, 3, figsize=(5.5, 3), dpi=300)
        axes[0].imshow(diag_raw, cmap='gray', vmin=0, vmax=200)
        axes[1].imshow(diag_noise_frame, cmap='gray',
                       vmin=diag_noise_frame.min(), vmax=diag_noise_frame.max())
        axes[2].imshow(diag_denoised, cmap='gray', vmin=0, vmax=200)
        for ax in axes:
            ax.axis('off')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close()

    print('Calculating diagnostic statistics across recording.')
    fig, [ax1, ax2] = plt.subplots(1, 2, dpi=300, figsize=(8, 2.5))
    ax1.plot(meanF, color='k', lw=1)
    ax2.plot(meanP, color='k', lw=1)
    ax1.set_xlabel('frames')
    ax2.set_xlabel('frames')
    ax1.set_ylabel('frame mean pixel value')
    ax2.set_ylabel('frame mean pixel value')
    ax1.set_ylim([np.percentile(meanF, 0.1), np.percentile(meanF, 99.9)])
    ax2.set_ylim([np.percentile(meanP, 0.1), np.percentile(meanP, 99.9)])
    ax1.set_title('noise-corrected stack')
    ax2.set_title('putative noise pattern')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close()

    pdf.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-dim', '--dim', type=int, default=1)
    parser.add_argument('-makevid', '--makevid', type=fm2p.str_to_bool, default=False)
    args = parser.parse_args()

    if not args.makevid:
        if args.dim == 1:
            denoise_tif_1d()
        elif args.dim == 2:
            denoise_tif_2d()

    elif args.makevid:

        ra_img = fm2p.select_file(
            'Select the raw tif stack (not yet denoised).',
            filetypes=[('TIF','.tif'), ('TIFF','.tiff'),]
        )

        noise_pattern = fm2p.select_file(
            'Select the computed noise pattern tif stack.',
            filetypes=[('TIF','.tif'), ('TIFF','.tiff'),]
        )

        ra_newimg = fm2p.select_file(
            'Select the denoised image stack.',
            filetypes=[('TIF','.tif'), ('TIFF','.tiff'),]
        )

        vid_save_dir = fm2p.select_directory(
            'Select a save directory.'
        )
        vid_save_path = os.path.join(vid_save_dir, 'denoised_demo.avi')

        make_denoise_diagnostic_video(
            fm2p.load_tif_stack(ra_img),
            fm2p.load_tif_stack(noise_pattern),
            fm2p.load_tif_stack(ra_newimg),
            vid_save_path,
            0,
            3600
        )
