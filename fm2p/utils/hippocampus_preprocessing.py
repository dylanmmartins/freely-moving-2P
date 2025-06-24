
import os
import numpy as np
import fm2p


def hippocampal_preprocess(cfg_path):

    cfg = fm2p.read_yaml(cfg_path)

    recording_names = cfg['hp_inter_recs'] + cfg['hp_home_recs']

    for rnum, rname in enumerate(recording_names):

        full_rname = ...

        rpath = os.path.join(cfg['spath'], rname)

        # Topdown camera files
        possible_topdown_videos = fm2p.find('*.mp4', rpath, MR=False)
        topdown_video = fm2p.filter_file_search(possible_topdown_videos, toss=['labeled','resnet50'], MR=True)
    
        # Run pose estimation
        fm2p.run_pose_estimation(
            topdown_video,
            project_cfg=cfg['top_DLC_project'],
            filter=False
        )
    
        topdown_pts_path = fm2p.find('*DLC_resnet50_*freely_moving_topdown_06*.h5', rpath, MR=True)
    
        F_path = fm2p.find('F.npy', rpath, MR=True)
        Fneu_path = fm2p.find('Fneu.npy', rpath, MR=True)
        suite2p_spikes = fm2p.find('spks.npy', rpath, MR=True)
        iscell_path = fm2p.find('iscell.npy', rpath, MR=True)
    
        # Topdown behavior and obstacle/arena tracking
        top_cam = fm2p.Topcam(rpath, '', cfg=cfg)
        top_cam.add_files(
            top_dlc_h5=topdown_pts_path,
            top_avi=topdown_video
        )
        arena_dict = top_cam.track_arena(no_pillar=True)
        pxls2cm = arena_dict['pxls2cm']
        top_xyl, top_tracking_dict = top_cam.track_body(pxls2cm)
    
        F = np.load(F_path, allow_pickle=True)
        Fneu = np.load(Fneu_path, allow_pickle=True)
        spks = np.load(suite2p_spikes, allow_pickle=True)
        iscell = np.load(iscell_path, allow_pickle=True)
    
        twop_recording = fm2p.TwoP(rpath, '', cfg=cfg)
        twop_recording.add_data(
            F=F,
            Fneu=Fneu,
            spikes=spks,
            iscell=iscell
        )
        twop_dict = twop_recording.calc_dFF(neu_correction=0.7, oasis=False)

        twop_dt = 1./cfg['twop_rate']
        twopT = np.arange(0, np.size(twop_dict['s2p_spks'], 1)*twop_dt, twop_dt)
        twop_dict['twopT'] = twopT
        twop_dict['matlab_cellinds'] = np.arange(np.size(twop_dict['raw_F'],0))
    
        # For rare instances when scanimage acquired more 2P frames than topdown camera.
        # This is usually by ~1 frame, but this will handle larger discrepancies.
        _len_diff = np.size(learx) - np.size(twop_dict['s2p_spks'], 1)
        while _len_diff != 0:
            if _len_diff > 0:
                # top tracking is too long for spike data
                learx = learx[:-1]
                leary = leary[:-1]
                rearx = rearx[:-1]
                reary = reary[:-1]
                yaw = yaw[:-1]
            elif _len_diff < 0:
                # spike data is too long for top tracking
                twop_dict['twopT'] = twop_dict['twopT'][:-1]
                twop_dict['raw_F0'] = twop_dict['raw_F0'][:-1]
                twop_dict['raw_F'] = twop_dict['raw_F'][:-1]
                twop_dict['norm_F'] = twop_dict['norm_F'][:-1]
                twop_dict['raw_Fneu'] = twop_dict['raw_Fneu'][:-1]
                twop_dict['raw_dFF'] = twop_dict['raw_dFF'][:-1]
                twop_dict['norm_dFF'] = twop_dict['norm_dFF'][:-1]
                twop_dict['denoised_dFF'] = twop_dict['denoised_dFF'][:-1]
                twop_dict['s2p_spks'] = twop_dict['s2p_spks'][:-1]
            _len_diff = np.size(learx) - np.size(twop_dict['s2p_spks'], 1)

        headx = np.array([np.mean([rearx[f], learx[f]]) for f in range(len(rearx))])
        heady = np.array([np.mean([reary[f], leary[f]]) for f in range(len(reary))])
    
        sc = fm2p.SpatialCoding(cfg)
        sc.add_data(
            x=headx,
            y=heady,
            spikes=twop_dict['s2p_spks']
        )
        place_cell_dict = sc.calc_place_cells()
    
        preprocessed_dict = {
            **top_tracking_dict,
            **top_xyl.to_dict(),
            **arena_dict,
            **twop_dict,
            **place_cell_dict
        }

        _savepath = os.path.join(rpath, '{}_preproc.h5'.format(full_rname))
        print('Writing preprocessed data to {}'.format(_savepath))
        fm2p.write_h5(_savepath, preprocessed_dict)
    

    