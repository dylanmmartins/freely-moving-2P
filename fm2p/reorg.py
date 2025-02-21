import os
import shutil
import json
import numpy as np
import PySimpleGUI as sg

sg.theme('Default1')

def popup_radio(title, prompt, options):
    layout = [
        [sg.Text(prompt)],
        [sg.Radio(option, group_id=1, key=option) for option in options],
        [sg.Button("OK"), sg.Button("Cancel")],
    ]
    window = sg.Window(title, layout)
    event, values = window.read()
    window.close()

    if event == "OK":
        for option in options:
            if values[option]:
                return option
    return None

def reorg():

    print('Choose session folder')
    main_dir = sg.popup_get_folder('Choose session folder',
                            no_window=True, keep_on_top=True)
    
    nRecs = int(sg.popup_get_text('How many recordings?'))

    session_name = os.path.split(main_dir)[1]

    all_props = {}

    for recnum in range(nRecs):

        print('Recording folder (n={}/{})'.format(recnum+1, nRecs))

        temp_rec_name = '{}_{:02}'.format(session_name, recnum+1)

        rec_name = sg.popup_get_text(
            'Accept recording name?', default_text=temp_rec_name)
        
        print('Choose recording subdirectory ({}/{})'.format(recnum+1, nRecs))
        recDir = sg.popup_get_folder(
            'Choose recording subdirectory ({}/{})'.format(recnum+1, nRecs),
            initial_folder=main_dir, no_window=True, keep_on_top=True)
        
        eyecam_option = popup_radio('Has eyecam?', 'Has eyecam?', ['True','False'])
        if eyecam_option == 'True':
            has_eyecam = True
        elif eyecam_option == 'False':
            has_eyecam = False
        elif eyecam_option == None:
            print('Invalid eyecam option. Defaulting to no eyecam.')
            has_eyecam = False

        # Rename subdirectory
        rdir = os.path.join(main_dir, rec_name)
        os.rename(recDir, rdir)
        recDir = rdir

        print('Choose topdown video')
        topvid = sg.popup_get_file(
            'Choose topdown video',
            initial_folder=rdir, no_window=True, file_types=(('MP4', '*.mp4'),))
        os.rename(topvid,
            os.path.join(recDir, '{}_topVid.mp4'.format(rec_name)))

        print('Choose topdown DLC hdf5 file')
        dlc_h5 = sg.popup_get_file(
            'Choose topdown DLC hdf5 file',
            initial_folder=rdir, no_window=True, file_types=(('HDF5', '*.h5'),))
        os.rename(dlc_h5,
            os.path.join(recDir, '{}_topdownTracking.h5'.format(rec_name)))
        
        print('Choose two photon tif stack')
        imgstack = sg.popup_get_file(
            'Choose two photon TIFF stack',
            initial_folder=rdir, no_window=True, file_types=(('TIF','*.tif'),('TIFF','*.tiff'),))
        os.rename(imgstack,
            os.path.join(recDir, '{}_twopData.tif'.format(rec_name)))
        
        print('Choose suite2p plane directory')
        s2p_dir = sg.popup_get_folder('Suite2p folder',
            initial_folder=recDir, no_window=True)
        
        F = np.load(os.path.join(s2p_dir, 'F.npy'))
        Fneu = np.load(os.path.join(s2p_dir, 'Fneu.npy'))
        spks = np.load(os.path.join(s2p_dir, 'spks.npy'))
        stat = np.load(os.path.join(s2p_dir, 'stat.npy'), allow_pickle=True)
        ops =  np.load(os.path.join(s2p_dir, 'ops.npy'), allow_pickle=True)
        iscell = np.load(os.path.join(s2p_dir, 'iscell.npy'))

        np.savez(os.path.join(recDir, '{}_twopData.npz'.format(rec_name)),
                F=F, Fneu=Fneu, spks=spks, stat=stat, ops=ops, iscell=iscell)
        
        if has_eyecam:
            
            print('Eyecam deinterlaced AVI file')
            eye_dAvi = sg.popup_get_file('Eyecam deinterlaced AVI file',
                                        initial_folder=recDir,
                                        no_window=True,
                                        file_types=(('AVI', '*.avi'),))
            os.rename(eye_dAvi,
                        os.path.join(recDir, '{}_eyeDeinterVid.avi'.format(rec_name)))
            
            print('Eyecam timestamps')
            eye_TS = sg.popup_get_file('Eyecam timestamps',
                                        initial_folder=recDir,
                                        no_window=True,
                                        file_types=(('CSV', '*.csv'),))
            os.rename(eye_TS,
                        os.path.join(recDir, '{}_eyeTS.csv'.format(rec_name)))
            
            print('Eyecam DLC HDF5 file')
            dlc_h5 = sg.popup_get_file('Eyecam DLC HDF5 file',
                                        initial_folder=recDir,
                                        no_window=True,
                                        file_types=(('HDF', '*.h5'),))
            os.rename(dlc_h5,
                        os.path.join(recDir, '{}_eyeTracking.h5'.format(rec_name)))
        
            print('Eyecam TTL voltage file')
            ttlV_csv = sg.popup_get_file('Eyecam TTL voltage file',
                                        initial_folder=recDir,
                                        no_window=True,
                                        file_types=(('CSV', '*.csv'),))
            os.rename(ttlV_csv,
                        os.path.join(recDir, '{}_eyeTTLvolts.h5'.format(rec_name)))

            print('Eyecam TTL timestamps file')
            ttlT_csv = sg.popup_get_file('Eyecam TTL timestamps file',
                                        initial_folder=recDir,
                                        no_window=True,
                                        file_types=(('CSV', '*.csv'),))
            os.rename(ttlT_csv,
                        os.path.join(recDir, '{}_eyeTTLTS.h5'.format(rec_name)))
        
        rec_props = {
            'rec_dir': rdir,
            'rec_name': rec_name,
            'rec_num': recnum+1,
            'top_vid': '{}_topVid.mp4'.format(rec_name),
            'top_dlc' : '{}_topdownTracking.h5'.format(rec_name),
            'twop_tiff': '{}_twopData.tif'.format(rec_name),
            'suite2p': '{}_twopData.npz'.format(rec_name)
        }

        if has_eyecam:
            rec_props['eye_vid'] = '{}_eyeDeinterVid.avi'.format(rec_name)
            rec_props['eye_dlc'] = '{}_eyeTracking.h5'.format(rec_name)
            rec_props['eye_TS'] = '{}_eyeTS.csv'.format(rec_name)
            rec_props['eye_TTLTS'] = '{}_eyeTTLTS.h5'.format(rec_name)
            rec_props['eye_TTLV'] = '{}_eyeTTLvolts.h5'.format(rec_name)

        all_props['R{:02}'.format(recnum+1)] = rec_props

    _savepath = os.path.join(main_dir, 'session_props.json')
    with open(_savepath, 'w') as f:
        json.dump(all_props, f, indent=4)
    print('File saved to {}'.format(_savepath))


if __name__ == '__main__':
    reorg()