"""
Preprocessing and analysis for freely moving two-photon data.
DMM, 2024
"""

from fm2p.utils.helper import (
    split_xyl,
    apply_liklihood_thresh
)

from fm2p.utils.paths import (
    choose_most_recent,
    up_dir,
    find,
    filter_file_search,
    check_subdir,
    list_subdirs
)

from fm2p.utils.time import (
    read_timestamp_series,
    interp_timestamps,
    read_timestamp_file,
    find_closest_timestamp
)

from fm2p.utils.cameras import (
    deinterlace,
    flip_headcams,
    run_pose_estimation,
    pack_video_frames,
    compute_camera_distortion,
    undistort_video
)

from fm2p.utils.eyecam import (
    Eyecam
)

from fm2p.utils.files import (
    open_dlc_h5,
    write_h5,
    read_h5
)

from fm2p.utils.filter import (
    convfilt,
    nanmedfilt
)

from fm2p.utils.topcam import (
    Topcam
)

from fm2p.utils.twop import (
    TwoP
)

# from fm2p.reorg import reorg
from fm2p.preprocess import preprocess

from fm2p.utils.cmap import make_parula

from fm2p.utils.walls import calc_rays

from fm2p.utils.ebc import (
    calculate_egocentric_rate_map,
    calc_EBC,
    calc_show_rate_maps,
    plot_single_polar_ratemap,
    plot_allocentric_spikes,
    plot_egocentic_wall_positions
)

from fm2p.utils.behavior import (
    plot_yaw_distribution,
    plot_speed_distribution,
    plot_movement_yaw_distribution
)

from fm2p.utils.alignment import (
    align_eyecam_using_TTL
)