"""
fm2p/preprocess.py
Preprocess recording, convering raw data to a set of several .h5 files.

DMM, 2024
"""


import os
import PySimpleGUI as sg

sg.theme('Default1')


import fm2p


def preprocess(rec_path=None, cfg=None):

    if rec_path is None:
        print('Choose recording path')
        rec_path = sg.popup_get_folder(
            'Choose recording path', 'Choose recording path',
            no_window=True, keep_on_top=True
        )

    print('Accept recording name')
    rec_name = os.path.split()

    if cfg is None:
        has_worldcam = False
        has_eyecam = False
        eye_DLC_project = 'T:/dlc_projects/freely_moving_eyecams_01-dylan-2024-12-04/config.yaml'
        top_DLC_project = 'T:/dlc_projects/freely_moving_topdown_tub_01-dylan-2024-12-04/config.yaml'
        worldcam_distortion_mtx_path = 'T:/camera_calibration/worldcam_01_calibration_mtx.npz'
    elif cfg is not None:
        has_worldcam = cfg['has_worldcam']
        has_eyecam = cfg['has_eyecam']
        eye_DLC_project = cfg['eye_DLC_project']
        top_DLC_project = cfg['top_DLC_project']
        worldcam_distortion_mtx_path = cfg['worldcam_distortion_mtx_path']

    # Deinterlace eyecam
    if has_eyecam:
        # raw_eyevid_path = fm2p.find('{}*eye.avi'.format(rec_name), rec_path, MR=True)
        eye_deinter_vid = fm2p.deinterlace(raw_eyevid_path)

    # Deinterlace worldcam
    if has_worldcam:
        # raw_worldvid_path = fm2p.find('{}*world.avi'.format(rec_name), rec_path, MR=True)
        world_deinter_vid = fm2p.deinterlace(raw_worldvid_path)
        _ = fm2p.undistort_video(world_deinter_vid, worldcam_distortion_mtx_path)

    # Run deeplabcut for eyecam
    if has_eyecam:
        fm2p.run_pose_estimation(eye_deinter_vid, eye_DLC_project)

    # Run deeplabcut for topdown camera
    # raw_topvid_path = fm2p.find('{}*top.avi'.format(rec_name), rec_path, MR=True)
    fm2p.run_pose_estimation(raw_topvid_path, top_DLC_project)

    fm2p.reorg()

    # Track pupil position from the eye camera
    if has_eyecam:
        reye_cam = fm2p.Eyecam(rec_path, rec_name)
        reye_cam.find_files()
        eye_xyl, ellipse_dict = reye_cam.track_pupil()

        # Load eye video as array
        eyevid_arr = fm2p.pack_video_frames(reye_cam.eye_avi)
        
        # Save eye cam tracking and video as .h5 file
        reye_cam.save_tracking(ellipse_dict, eye_xyl, eyevid_arr)

    # Track mouse position from top-down camera
    top_cam = fm2p.Topcam(rec_path, rec_name)
    top_cam.find_files()
    top_xyl, top_tracking_dict = top_cam.track_body()

    # Load top video as array
    topvid_arr = fm2p.pack_video_frames(top_cam.top_avi)

    # Save both as .h5
    top_cam.save_tracking(top_tracking_dict, top_xyl, topvid_arr)

    # Load processed two photon data from suite2p
    twop = fm2p.TwoP
    twop.find_files()
    twop_dict = twop.calc_dFF(neu_correction=0.7)
    twop.save_fluor(twop_dict)

