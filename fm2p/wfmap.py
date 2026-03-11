""" Make overlay map from widefield sign map.
wfTools/plotmap.py


Written by DMM, Nov 2023
"""


import os
# import PySimpleGUI as sg
from scipy.io import loadmat
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter, zoom
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from .utils.gui_funcs import select_directory, select_file

# sg.theme('Default1')


def main():

    # get basepath
    print('Choose base directory.')
    base = select_directory('Choose base directory')
    
    print('Choose VFS .mat file.')
    v_path = select_file('Select VFS .mat file',
                      filetypes=(('.mat','.mat'),))
    
    print('Choose additional maps .mat file.')
    am_path = select_file('Select additional maps .mat file',
                      filetypes=(('.mat','.mat'),))
    
    print('Choose tiff reference image.')
    i_path = select_file('Select tiff reference image.',
                               filetypes=(('.tif','.tif'),))

    # get savepath
    print('Choose save directory.')
    savepath = select_directory('Choose save directory')
    
    # name = sg.popup_get_text('Enter animal name.')
    name = 'DMM070'
    
    matfile = loadmat(v_path)

    matfile2 = loadmat(am_path)

    im = Image.open(i_path)
    img = np.array(im)

    smlimg = zoom(img, 400 / 2048)
    smlimg = smlimg.astype(float)
    smlimg = 1-(smlimg-np.min(smlimg)) / 65535

    overlay = matfile['VFS_raw'].copy()

    h_over = matfile2['maps']['HorizontalRetinotopy'].copy()[0][0]
    v_over = matfile2['maps']['VerticalRetinotopy'].copy()[0][0]

    t2b = np.zeros([256, 4])
    t2b[:,3] = np.linspace(0, 1, 256)
    t2b = ListedColormap(t2b)

    plt.figure(figsize=(5,5),dpi=300)
    plt.imshow(smlimg, cmap=t2b)
    plt.axis('off')
    plt.title(name)
    plt.savefig(os.path.join(savepath, '{}_ref_img.png'.format(name)), dpi=300)

    plt.figure(figsize=(5,5),dpi=300)
    plt.imshow(gaussian_filter(overlay,1.5), cmap='jet')
    plt.imshow(smlimg, cmap=t2b)
    plt.axis('off')
    plt.title(name)
    plt.savefig(os.path.join(savepath, '{}_overlay.png'.format(name)), dpi=300)

    fig, [ax1,ax2] = plt.subplots(1,2, figsize=(6,3),dpi=300)

    im1 = ax1.imshow(gaussian_filter(h_over,2), cmap='jet')
    ax1.imshow(smlimg, cmap=t2b)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, label='horizontal')

    im2 = ax2.imshow(gaussian_filter(v_over,2), cmap='jet')
    ax2.imshow(smlimg, cmap=t2b)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, label='vertical')

    fig.suptitle(name)

    fig.tight_layout()
    fig.savefig(os.path.join(savepath, '{}_split_overlay.png'.format(name)), dpi=300)


if __name__=='__main__':
    
    main()