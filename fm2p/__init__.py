"""
Preprocessing and analysis for freely moving two-photon data.
DMM, 2024
"""

from .utils.helper import (
    split_xyl,
    apply_liklihood_thresh,
    str_to_bool,
    make_default_cfg,
    to_dict_of_arrays
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
    fmt_now
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
    write_yaml
)

from .utils.filter import (
    convfilt,
    sub2ind,
    nanmedfilt
)

from .utils.topcam import Topcam

from .utils.twop import (
    TwoP,
    calc_dFF1
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
    align_lightdark_using_TTL
)

from .utils.frame_annotation import (
    user_polygon_translation,
    place_points_on_image
)

from .utils.correlation import (
    corr2_coeff,
    nanxcorr
)

from .utils.gui_funcs import (
    select_file,
    select_directory,
    get_string_input
)

from .utils.ref_frame import (
    angle_to_target,
    calc_reference_frames
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
    plot_running_median
)

from .mapRF import mapRF
from .summarize_revcorr import summarize_revcorr
from .summarize_revcorr_ltdk import summarize_revcorr_ltdk

from .utils.axons import (
    get_independent_axons
)

from .utils.glm import (
    fit_closed_GLM,
    compute_y_hat,
    GLM,
    fit_pred_GLM
)

