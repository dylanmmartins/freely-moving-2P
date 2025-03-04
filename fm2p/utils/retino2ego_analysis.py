

import os
import yaml
import numpy as np
import pandas as pd

import fm2p


def retino2ego_analysis(cfg, rname):
    # Must run this function on each recording individually

    if type(cfg)==str:
        with open(cfg, 'r') as infile:
            cfg = yaml.load(infile, Loader=yaml.FullLoader)
    rcfg = cfg[rname]

    rpath = rcfg['rpath']
    eye_data = fm2p.read_h5(rcfg['eye_preproc_path'])
    top_data = fm2p.read_h5(rcfg['top_preproc_path'])
    twop_data = fm2p.read_h5(rcfg['twop_preproc_path'])
    eyeT_startInd = rcfg['eyeT_startInd']
    eyeT_endInd = rcfg['eyeT_endInd']

    # Two photon data
    sps = twop_data['s2p_spks']
    raw_dFF = twop_data['raw_dFF']
    twop_dt = 1./cfg['twop_rate']
    twopT = np.arange(0, np.size(sps, 1)*twop_dt, twop_dt)

    # Eyecam data
    eyeT = eye_data['eyeT'][eyeT_startInd:eyeT_endInd]
    theta_raw = eye_data['theta'][eyeT_startInd:eyeT_endInd]
    phi_raw = eye_data['phi'][eyeT_startInd:eyeT_endInd]
    puprad_raw = eye_data['longaxis'][eyeT_startInd:eyeT_endInd]

    # Interpolate eyecam data to twop and topdown data
    eyeT = eyeT.copy() - eyeT[0]

    theta = fm2p.interpT(theta_raw, eyeT, twopT, fill_consecutive=True)
    phi = fm2p.interpT(phi_raw, eyeT, twopT, fill_consecutive=True)
    puprad = fm2p.interpT(puprad_raw, eyeT, twopT, fill_consecutive=True)

    # Topcam data
    speed = top_data['speed']
    # Animal yaw is defined so that 0 deg is rightward in the raw video, 90 deg is downward in the raw
    # video, and angles wrap from 0 to 360 deg clockwise.
    yaw = top_data['head_yaw_deg']
    movement_yaw = top_data['movement_yaw_deg']
    top_x = top_data['x']
    top_y = top_data['y']
    top_xdisp = top_data['x_displacement']
    top_ydisp = top_data['y_displacement']
    arenaTL = (top_data['arenaTL']['x'], top_data['arenaTL']['y'])
    arenaTR = (top_data['arenaTR']['x'], top_data['arenaTR']['y'])
    arenaBL = (top_data['arenaBL']['x'], top_data['arenaBL']['y'])
    arenaBR = (top_data['arenaBR']['x'], top_data['arenaBR']['y'])
    pillarT = (top_data['pillarT']['x'], top_data['pillarT']['y'])
    pillarB = (top_data['pillarB']['x'], top_data['pillarB']['y'])
    pillarL = (top_data['pillarL']['x'], top_data['pillarL']['y'])
    pillarR = (top_data['pillarR']['x'], top_data['pillarR']['y'])
    pillar_radius = top_data['pillar_radius']
    pillar_centroid = (top_data['pillar_centroid']['x'], top_data['pillar_centroid']['y'])

    dist_to_pillar = np.zeros_like(top_x) * np.nan
    angle_to_pillar = np.zeros_like(top_x) * np.nan
    angle_pillar_spans = np.zeros_like(top_x) * np.nan
    for f in range(len(top_x)):

        # All geometry is handled using units of radians for angles and units of pixels for distances

        a_ = pillar_centroid[1] - top_y[f]
        b_ = pillar_centroid[0] - top_x[f]

        dist_to_pillar[f] = np.sqrt(b_**2 + a_**2)
        angle_to_pillar[f] = np.tan2(a_ / b_) # rad
        
        # Angle that pillar spans given the animal's current distance
        angle_pillar_spans[f] = np.tan2(pillar_radius / dist_to_pillar[f]) # rad

        # Horizontal gaze angle relative to the animal's head (instead of relative to the central axis
        # of the camera).
        theta_reltohead = theta[f] + cfg['eyecam_angular_offset']


if __name__ == '__main__':
    
    cfg = r'K:\Mini2P\250220_DMM_DMM042_pillar\preprocessed_config.yaml'
    rname = 'fm1'

    retino2ego_analysis(cfg, rname)