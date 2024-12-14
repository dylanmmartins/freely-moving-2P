

import fmr2e


def preprocess(rec_path, cfg=None):

    rec_name = ...

    if cfg is None:
        has_worldcam = False
        eye_DLC_project = ...
        top_DLC_project = ...
        worldcam_distortion_mtx_path = ...

    # Deinterlace eyecam
    raw_eyevid_path = fmr2e.find('{}*eye.avi'.format(rec_name), rec_path, MR=True)
    eye_deinter_vid = fmr2e.deinterlace(raw_eyevid_path)

    # Deinterlace worldcam
    if has_worldcam:
        raw_worldvid_path = fmr2e.find('{}*world.avi'.format(rec_name), rec_path, MR=True)
        world_deinter_vid = fmr2e.deinterlace(raw_worldvid_path)
        _ = fmr2e.undistort_video(world_deinter_vid, worldcam_distortion_mtx_path)

    # Run deeplabcut for eyecam
    fmr2e.run_pose_estimation(eye_deinter_vid, eye_DLC_project)

    # Run deeplabcut for topdown camera
    raw_topvid_path = fmr2e.find('{}*top.avi'.format(rec_name), rec_path, MR=True)
    fmr2e.run_pose_estimation(raw_topvid_path, top_DLC_project)

    # Track pupil position from the eye camera
    reye_cam = fmr2e.Eyecam(rec_path, rec_name)
    reye_cam.find_files()
    eye_xyl, ellipse_dict = reye_cam.track_pupil()

    # Load eye video as array
    eyevid_arr = fmr2e.pack_video_frames(reye_cam.eye_avi)
    
    # Save eye cam tracking and video as .h5 file
    reye_cam.save_tracking(ellipse_dict, eye_xyl, eyevid_arr)

    # Track mouse position from top-down camera
    top_cam = fmr2e.Topcam(rec_path, rec_name)
    top_cam.find_files()
    top_xyl, top_tracking_dict = top_cam.track_body()

    # Load top video as array
    topvid_arr = fmr2e.pack_video_frames(top_cam.top_avi)

    # Save both as .h5
    top_cam.save_tracking(top_tracking_dict, top_xyl, topvid_arr)

    # Load processed two photon data from suite2p
    twop = fmr2e.TwoP
    twop.find_files()
    twop_dict = twop.calc_dFF(neu_correction=0.7)
    twop.save_fluor(twop_dict)

