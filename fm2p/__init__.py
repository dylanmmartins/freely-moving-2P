# -*- coding: utf-8 -*-
"""
Preprocessing and analysis for freely moving two-photon data.

Author: DMM, written 2024-2026
"""

__version__ = "0.1"

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


from .utils.helper import (
    split_xyl,
    apply_liklihood_thresh,
    str_to_bool,
    make_default_cfg,
    to_dict_of_arrays,
    blockPrint,
    enablePrint,
    fix_dict_dtype,
    nan_filt,
    nan_interp,
    nan_interp_circular,
    calc_r2,
    mask_non_nan,
    interp_short_gaps,
    angular_diff_deg,
    step_interp,
    bootstrap_stderr,
    interp_short_gaps_circ,
    array_to_pil
)

from .utils.linalg import (
    make_U_triangular,
    make_L_triangular
)

from .utils.paths import (
    choose_most_recent,
    up_dir,
    find,
    filter_file_search,
    check_subdir,
    list_subdirs
)

from .utils.time import (
    read_timestamp_series,
    interp_timestamps,
    read_timestamp_file,
    time2str,
    str2time,
    time2float,
    interpT,
    find_closest_timestamp,
    fmt_now,
    read_scanimage_time
)

from .utils.cameras import (
    deinterlace,
    flip_headcams,
    run_pose_estimation,
    pack_video_frames,
    compute_camera_distortion,
    undistort_video,
    load_video_frame
)

from .utils.eyecam import (
    Eyecam,
    plot_pupil_ellipse_video
)

from .utils.files import (
    open_dlc_h5,
    write_h5,
    read_h5,
    read_yaml,
    write_yaml,
    get_group_h5_keys,
    read_group_h5,
    write_group_h5,
    read_json
)

from .utils.filter import (
    convfilt,
    sub2ind,
    nanmedfilt,
    convolve2d
)

from .utils.topcam import Topcam

from .utils.twop import (
    TwoP,
    calc_inf_spikes,
    normalize_axonal_spikes,
    zscore_spikes
)

from .preprocess import preprocess

from .utils.cmap import (
    make_parula,
    make_rainbow_legend
)

from .utils.walls import (
    Wall,
    closest_wall_per_ray,
    calc_rays
)

from .utils.ebc import (
    calculate_egocentric_rate_map,
    calc_EBC,
    calc_show_rate_maps,
    plot_single_polar_ratemap,
    plot_allocentric_spikes,
    plot_egocentic_wall_positions
)

from .utils.behavior import (
    plot_yaw_distribution,
    plot_speed_distribution,
    plot_movement_yaw_distribution
)

from .utils.alignment import (
    align_eyecam_using_TTL,
    align_lightdark_using_TTL,
    align_crop_IMU
)

from .utils.frame_annotation import (
    user_polygon_translation,
    place_points_on_image
)

from .utils.correlation import (
    corr2_coeff,
    nanxcorr,
    corrcoef,
    calc_cohen_d
)

from .utils.gui_funcs import (
    select_file,
    select_directory,
    get_string_input
)

from .utils.ref_frame import (
    angle_to_target,
    calc_reference_frames,
    visual_angle_degrees
)

from .utils.LNP_helpers import (
    rough_penalty,
    find_param,
    make_varmap
)

from .utils.LNP_model import (
    linear_nonlinear_poisson_model,
    fit_LNLP_model,
    fit_all_LNLP_models
)

from.utils.LNP_eval import (
    add_scatter_col,
    read_models,
    plot_model_LLHs,
    eval_models,
    plot_rank_test_results,
    plot_pred_spikes,
    calc_scaled_LNLP_tuning_curves,
    plot_scaled_LNLP_tuning_curves,
    calc_bootstrap_model_params,
    get_cells_best_LLHs,
    determine_responsiveness_from_null,
    get_responsive_inds,
    get_responsive_inds_2
)

from .fit_model import (
    fit_simple_GLM,
    fit_LNLP,
    fit_model
)

from .summarize_model_fit import summarize_model_fit

from .utils.LNP_summary import (
    tuning_curve,
    plot_tuning,
    write_detailed_cell_summary
)

from .utils.tuning import (
    tuning_curve,
    plot_tuning,
    calc_modind,
    calc_tuning_reliability,
    norm_tuning,
    plot_running_median,
    calc_reliability_d,
    calc_multicell_modulation,
    spectral_slope,
    calc_spectral_noise
)

from .utils.axons import (
    get_single_independent_axons,
    get_grouped_independent_axons,
    get_independent_axons
)

from .utils.glm import (
    GLM,
    fit_pred_GLM
)

from .deinter_dir import deinter_dir

from .utils.place_cells import (
    SpatialCoding,
    plot_place_cell_maps
)

from .utils.hippocampus_preprocessing import (
    hippocampal_preprocess
)

from .pred_pupil import pred_pupil

from .revcorr import (
    revcorr,
    calc_revcorr
)

from .utils.multicell_GLM import (
    multicell_GLM,
    fit_multicell_GLM,
    run_body_model,
    run_retina_model,
    run_pupil_model,
    run_all_GLMs
)

from .utils.boundary_tuning import (
    BoundaryTuning,
    calc_shfl_mean_resultant_mp,
    calc_MRL_mp,
    rate_map_mp,
    convert_bools_to_ints
)

from .boundary import boundary

from .utils.sensor_fusion import (
    Kalman,
    ImuOrientation
)

from .utils.imu import (
    read_IMU,
    detrend_gyroz_weighted_gaussian,
    detrend_gyroz_simple_linear,
    unwrap_degrees
)

from .utils.PETH import (
    calc_hist_PETH,
    norm_psth,
    norm_psth_paired,
    find_trajectory_initiation,
    get_event_offsets,
    get_event_onsets,
    balanced_index_resample,
    calc_PETHs,
    drop_nearby_events,
    drop_repeat_events,
    calc_PETH_mod_ind,
    drop_redundant_saccades,
    calc_eye_head_movement_times
)

from .utils.sparse_noise import (
    compute_calcium_sta_spatial,
    find_delay_frames,
    jaccard_topk,
    compute_split_STAs
)

from .utils.multiprocessing_helpers import (
    init_worker
)

from .split_suite2p import (
    split_suite2p,
    split_suite2p_npy,
    count_tif_frames
)

from .polar_revcorr import (
    polar_revcorr,
    polar_histogram2d,
    smooth_2d_rate_maps
)

from .sparse_noise_mapping import (
    sparse_noise_mapping,
    calc_sparse_noise_STAs,
    calc_sparse_noise_STA_reliability
)

from .summarize_session import summarize_session
from .eyehead_revcorr import eyehead_revcorr

from .gaussian_STA_fit import (
    fit_gauss,
    within_pct,
    gaus_eval,
    gaussian_STA_fit
)

from .register_tiled_locations import (
    register_tiled_locations
)

from .register_animals import (
    register_animals
)

from .merge_animal_essentials import merge_animal_essentials

from .utils.singlecell_GLM import singlecell_GLM