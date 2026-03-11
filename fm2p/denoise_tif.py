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
from matplotlib.backends.backend_pdf import PdfPages

from .utils.gui_funcs import select_file, select_directory
from .utils.img_stacks import load_tif_stack
from .utils.filter import convfilt, rolling_average
from .utils.helper import str_to_bool


def denoise_tif_1d(tif_path=None, ret=False, saveRA=False):
    """ Remove noise added into mini2p image stack by resonance scanner.
    
    The noise appears as hazy vertical banding which sweeps slowly along the x axis
    (they are not in static positions, and there are ~10 overlapping bands in the
    image for any given frame. they move both leftwards and rightwards. If ret is true,
    the function will return the image stack with a short (3 frame) rolling average applied.

    This is memory intensive and must be run on a computer with ~128 GB RAM for a video longer
    of 30k or more frames.

    Parameters
    ----------
    tif_path : str
        Path to the tiff file to be denoised. If None, a file dialog will be opened
        to select the file.
    ret : bool
        If True, return the denoised image stack with a short rolling average applied.
        Default is False.

    Returns
    -------
    sra_newimg : np.ndarray
        This is only returned if `ret` is True. The image array read in from the `tif_path`,
        denoised and with a rolling average applied to the array across the temporal axis.
    """

    if tif_path is None:
        tif_path = select_file(
            'Select tif stack.',
            filetypes=[('TIF', '*.tif'),('TIF','*.tiff'),]
        )

    print('Denoising {}'.format(tif_path))

    rawimg = load_tif_stack(tif_path)

    base_path = os.path.split(tif_path)[0]
    tif_name = os.path.split(tif_path)[1]
    pdf = PdfPages(os.path.join(base_path, 'denoising_figs.pdf'))

    nPix = 50
    band_block = np.concatenate([rawimg[:,:nPix,:], rawimg[:,-nPix:]], axis=1)
    mean_of_banded_block = np.mean(band_block, 1)

    fig = plt.figure(figsize=(6,6), dpi=300)
    plt.imshow(mean_of_banded_block, aspect='auto', cmap='gray')
    # plt.colorbar()
    plt.xlabel('y pixels')
    plt.ylabel('time (frames)')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    print('Calculating noise pattern.')
    # Take the section of the frame 
    f_size = np.shape(rawimg[0,:,:])
    noise_pattern = np.zeros_like(rawimg)
    for f in tqdm(range(np.size(noise_pattern,0))):
        frsn = convfilt(mean_of_banded_block[f,:],5)
        noise_pattern[f,:,:] = np.broadcast_to(frsn, f_size).copy()

    # Subtract the noise pattern from the raw image. Then, add back a
    # small amount of the signal so info doesn't get cut off for being
    # below the minimum of uint16 datatype. Clip the output so any
    # negative pixel values are discarded

    newimg = np.clip(
        np.subtract(rawimg, noise_pattern) + 16,
        0,
        None
    ).astype(np.uint16)

    f = 500
    fig, [ax1,ax2,ax3] = plt.subplots(1,3, figsize=(5.5,3), dpi=300)
    ax1.imshow(rawimg[f,:,:], cmap='gray', vmin=0, vmax=200)
    ax2.imshow(noise_pattern[f,:,:], cmap='gray',
               vmin=np.min(noise_pattern[f,:,:]),
               vmax=np.max(noise_pattern[f,:,:]))
    ax3.imshow(newimg[f,:,:], cmap='gray', vmin=0, vmax=200)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close()

    del rawimg

    print('Calculting diagnotic statistics across recording.')
    # mean across frames for output image data (is there drift over time or large jumps? it should not.)
    meanF = np.mean(newimg,axis=(1,2))
    # mean across frames for the oinse pattern (does it have large jumps or drift over time? it should)
    meanP = np.mean(noise_pattern,axis=(1,2))

    fig, [ax1,ax2] = plt.subplots(1,2, dpi=300, figsize=(8,2.5))
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

    noise_len = np.size(noise_pattern,0)

    del noise_pattern

    if not saveRA:

        newimg[newimg<np.iinfo(np.uint16).min] = np.iinfo(np.uint16).min
        newimg[newimg>np.iinfo(np.uint16).max] = np.iinfo(np.uint16).max

        tif_name_noext = os.path.splitext(tif_name)[0]
        savefilename = os.path.join(base_path, '{}_denoised.tif'.format(tif_name_noext))
        print('Writing {}'.format(savefilename))
        with tifffile.TiffWriter(savefilename, bigtiff=True) as savestack:
            savestack.write(
                data=newimg.astype(np.uint16),
                dtype=np.uint16,
                shape=newimg.shape,
                photometric='MINISBLACK'
            )

        pdf.close()

    elif saveRA:
        # Save two versions of the output video: one raw video, one with a small
        # rolling average, and one with a large rolling average.
        # For the small rolling average, apply a 400 msec smoothing window
        print('Calculating rolling average (short window).')
        sra_newimg = rolling_average(newimg, 3)

        full_numF = np.size(newimg,0)
        sra_len = np.size(sra_newimg,0)

        # Make sure corrected values are in the bounds of the data type that
        # will be used when the tif is written.
        sra_newimg[sra_newimg<np.iinfo(np.uint16).min] = np.iinfo(np.uint16).min
        sra_newimg[sra_newimg>np.iinfo(np.uint16).max] = np.iinfo(np.uint16).max

        tif_name_noext = os.path.splitext(tif_name)[0]
        s_savefilename = os.path.join(base_path, '{}_denoised_SRA.tif'.format(tif_name_noext))
        print('Writing {}'.format(s_savefilename))
        with tifffile.TiffWriter(s_savefilename, bigtiff=True) as savestack:
            savestack.write(
                data=sra_newimg.astype(np.uint16),
                dtype=np.uint16,
                shape=sra_newimg.shape,
                photometric='MINISBLACK'
            )
        del sra_newimg

        # For the large rolling average, apply a 1600 msec smoothing window (this
        # is probably only useful for visualization)
        print('Calculating rolling average (long window).')
        lra_newimg = rolling_average(newimg, 12)
        lra_len = np.size(lra_newimg,0)

        lra_newimg[lra_newimg<np.iinfo(np.uint16).min] = np.iinfo(np.uint16).min
        lra_newimg[lra_newimg>np.iinfo(np.uint16).max] = np.iinfo(np.uint16).max

        l_savefilename = os.path.join(base_path, '{}_denoised_LRA.tif'.format(tif_name_noext))
        print('Writing {}'.format(l_savefilename))
        with tifffile.TiffWriter(l_savefilename, bigtiff=True) as savestack:
            savestack.write(
                data=lra_newimg.astype(np.uint16),
                dtype=np.uint16,
                shape=lra_newimg.shape,
                photometric='MINISBLACK'
            )

        del newimg
        del lra_newimg

        pdf.close()

        frame_note = (
            'The full tif stack had {} frames. The denoised tif stack with a short running average '
            'has {} frames, and the one with a long running average has {} frames. When aligning '
            'the denoised stacks to other data streams, subtract diff/2 from the start and end. '
            'Adjust SRA by {} and LRA by {}.'
        )
        sra_adjust = int((noise_len-sra_len)/2)
        lra_adjust = int((noise_len-lra_len)/2)
        frame_note = frame_note.format(full_numF, sra_len, lra_len, sra_adjust, lra_adjust)
        txt_savepath = os.path.join(base_path, 'note_on_denoised_tif_dims.txt')
        with open(txt_savepath, 'w') as file:
            file.write(frame_note)
        print(frame_note)

        if ret:
            return sra_newimg


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

    ra_img = rolling_average(ra_img, 7)
    ra_newimg = rolling_average(ra_newimg, 7)

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

    This is memory intensive and must be run on a computer with ~128 GB RAM for a video longer
    of 30k or more frames.

    Parameters
    ----------
    tif_path : str
        Path to the tiff file to be denoised. If None, a file dialog will be opened
        to select the file.
    ret : bool
        If True, return the denoised image stack with a short rolling average applied.
        Default is False.

    Returns
    -------
    sra_newimg : np.ndarray
        This is only returned if `ret` is True. The image array read in from the `tif_path`,
        denoised and with a rolling average applied to the array across the temporal axis.
    """

    if tif_path is None:
        tif_path = select_file(
            'Select tif stack.',
            filetypes=[('TIF', '*.tif'),('TIF','*.tiff'),]
        )

    print('Denoising {}'.format(tif_path))

    rawimg = load_tif_stack(tif_path)

    base_path = os.path.split(tif_path)[0]
    tif_name = os.path.split(tif_path)[1]
    pdf = PdfPages(os.path.join(base_path, 'denoising_figs.pdf'))

    nPix = 50
    band_block_H = np.concatenate([rawimg[:, :nPix, :], rawimg[:, -nPix:, :]], axis=1)
    mean_of_banded_block_H = np.mean(band_block_H, 1)
    band_block_V = np.concatenate([rawimg[:, :, :nPix], rawimg[:, :, -nPix:]], axis=2)
    mean_of_banded_block_V = np.mean(band_block_V, 2)

    fig = plt.figure(figsize=(6,6), dpi=300)
    plt.imshow(mean_of_banded_block_H, aspect='auto', cmap='gray')
    # plt.colorbar()
    plt.xlabel('y pixels')
    plt.ylabel('time (frames)')
    plt.title('horizontal banded block')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    fig = plt.figure(figsize=(6,6), dpi=300)
    plt.imshow(mean_of_banded_block_V, aspect='auto', cmap='gray')
    # plt.colorbar()
    plt.xlabel('y pixels')
    plt.ylabel('time (frames)')
    plt.title('vertical banded block')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    print('Calculating noise pattern.')
    # Take the section of the frame 
    f_size = np.shape(rawimg[0,:,:])
    noise_pattern = np.zeros_like(rawimg)
    for f in tqdm(range(np.size(noise_pattern,0))):
        frsnH = convfilt(mean_of_banded_block_H[f,:],5)
        frsnV = convfilt(mean_of_banded_block_V[f,:],5)
        full_frsn = np.broadcast_to(frsnH, f_size).copy() + np.broadcast_to(frsnV, f_size).copy().T
        noise_pattern[f,:,:] = full_frsn

    # Subtract the noise pattern from the raw image. Then, add back a
    # small amount of the signal so info doesn't get cut off for being
    # below the minimum of uint16 datatype. Clip the output so any
    # negative pixel values are discarded

    newimg = np.clip(
        np.subtract(rawimg, noise_pattern) + 16,
        0,
        None
    ).astype(np.uint16)

    f = 500
    fig, [ax1,ax2,ax3] = plt.subplots(1,3, figsize=(5.5,3), dpi=300)
    ax1.imshow(rawimg[f,:,:], cmap='gray', vmin=0, vmax=200)
    ax2.imshow(noise_pattern[f,:,:], cmap='gray',
               vmin=np.min(noise_pattern[f,:,:]),
               vmax=np.max(noise_pattern[f,:,:]))
    ax3.imshow(newimg[f,:,:], cmap='gray', vmin=0, vmax=200)
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close()

    del rawimg

    print('Calculting diagnotic statistics across recording.')
    # mean across frames for output image data (is there drift over time or large jumps? it should not.)
    meanF = np.mean(newimg,axis=(1,2))
    # mean across frames for the oinse pattern (does it have large jumps or drift over time? it should)
    meanP = np.mean(noise_pattern,axis=(1,2))

    fig, [ax1,ax2] = plt.subplots(1,2, dpi=300, figsize=(8,2.5))
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

    noise_len = np.size(noise_pattern,0)

    noise_pattern[noise_pattern<np.iinfo(np.uint16).min] = np.iinfo(np.uint16).min
    noise_pattern[noise_pattern>np.iinfo(np.uint16).max] = np.iinfo(np.uint16).max

    tif_name_noext = os.path.splitext(tif_name)[0]
    savefilename = os.path.join(base_path, '{}_noise_pattern.tif'.format(tif_name_noext))
    print('Writing {}'.format(savefilename))
    with tifffile.TiffWriter(savefilename, bigtiff=True) as savestack:
        savestack.write(
            data=noise_pattern.astype(np.uint16),
            dtype=np.uint16,
            shape=noise_pattern.shape,
            photometric='MINISBLACK'
        )

    del noise_pattern

    if not saveRA:

        newimg[newimg<np.iinfo(np.uint16).min] = np.iinfo(np.uint16).min
        newimg[newimg>np.iinfo(np.uint16).max] = np.iinfo(np.uint16).max

        tif_name_noext = os.path.splitext(tif_name)[0]
        savefilename = os.path.join(base_path, '{}_denoised.tif'.format(tif_name_noext))
        print('Writing {}'.format(savefilename))
        with tifffile.TiffWriter(savefilename, bigtiff=True) as savestack:
            savestack.write(
                data=newimg.astype(np.uint16),
                dtype=np.uint16,
                shape=newimg.shape,
                photometric='MINISBLACK'
            )

        pdf.close()

    elif saveRA:
        # Save two versions of the output video: one raw video, one with a small
        # rolling average, and one with a large rolling average.
        # For the small rolling average, apply a 400 msec smoothing window
        print('Calculating rolling average (short window).')
        sra_newimg = rolling_average(newimg, 3)

        full_numF = np.size(newimg,0)
        sra_len = np.size(sra_newimg,0)

        # Make sure corrected values are in the bounds of the data type that
        # will be used when the tif is written.
        sra_newimg[sra_newimg<np.iinfo(np.uint16).min] = np.iinfo(np.uint16).min
        sra_newimg[sra_newimg>np.iinfo(np.uint16).max] = np.iinfo(np.uint16).max

        tif_name_noext = os.path.splitext(tif_name)[0]
        s_savefilename = os.path.join(base_path, '{}_denoised_SRA.tif'.format(tif_name_noext))
        print('Writing {}'.format(s_savefilename))
        with tifffile.TiffWriter(s_savefilename, bigtiff=True) as savestack:
            savestack.write(
                data=sra_newimg.astype(np.uint16),
                dtype=np.uint16,
                shape=sra_newimg.shape,
                photometric='MINISBLACK'
            )
        del sra_newimg

        # For the large rolling average, apply a 1600 msec smoothing window (this
        # is probably only useful for visualization)
        print('Calculating rolling average (long window).')
        lra_newimg = rolling_average(newimg, 12)
        lra_len = np.size(lra_newimg,0)

        lra_newimg[lra_newimg<np.iinfo(np.uint16).min] = np.iinfo(np.uint16).min
        lra_newimg[lra_newimg>np.iinfo(np.uint16).max] = np.iinfo(np.uint16).max

        l_savefilename = os.path.join(base_path, '{}_denoised_LRA.tif'.format(tif_name_noext))
        print('Writing {}'.format(l_savefilename))
        with tifffile.TiffWriter(l_savefilename, bigtiff=True) as savestack:
            savestack.write(
                data=lra_newimg.astype(np.uint16),
                dtype=np.uint16,
                shape=lra_newimg.shape,
                photometric='MINISBLACK'
            )

        del newimg
        del lra_newimg

        pdf.close()

        frame_note = (
            'The full tif stack had {} frames. The denoised tif stack with a short running average '
            'has {} frames, and the one with a long running average has {} frames. When aligning '
            'the denoised stacks to other data streams, subtract diff/2 from the start and end. '
            'Adjust SRA by {} and LRA by {}.'
        )
        sra_adjust = int((noise_len-sra_len)/2)
        lra_adjust = int((noise_len-lra_len)/2)
        frame_note = frame_note.format(full_numF, sra_len, lra_len, sra_adjust, lra_adjust)
        txt_savepath = os.path.join(base_path, 'note_on_denoised_tif_dims.txt')
        with open(txt_savepath, 'w') as file:
            file.write(frame_note)
        print(frame_note)

        if ret:
            return sra_newimg
        


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('-dim', '--dim', type=int, default=1)
    parser.add_argument('-makevid', '--makevid', type=str_to_bool, default=False)
    args = parser.parse_args()

    if not args.makevid:
        if args.dim == 1:
            denoise_tif_1d()
        elif args.dim == 2:
            denoise_tif_2d()

    elif args.makevid:

        ra_img = select_file(
            'Select the raw tif stack (not yet denoised).',
            filetypes=[('TIF','.tif'), ('TIFF','.tiff'),]
        )

        noise_pattern = select_file(
            'Select the computed noise pattern tif stack.',
            filetypes=[('TIF','.tif'), ('TIFF','.tiff'),]
        )

        ra_newimg = select_file(
            'Select the denoised image stack.',
            filetypes=[('TIF','.tif'), ('TIFF','.tiff'),]
        )

        vid_save_dir = select_directory(
            'Select a save directory.'
        )
        vid_save_path = os.path.join(vid_save_dir, 'denoised_demo.avi')

        make_denoise_diagnostic_video(
            load_tif_stack(ra_img),
            load_tif_stack(noise_pattern),
            load_tif_stack(ra_newimg),
            vid_save_path,
            0,
            3600
        )