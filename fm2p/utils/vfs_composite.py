# -*- coding: utf-8 -*-
"""
fm2p/utils/vfs_composite.py

Align individual animal VFS (visual field sign) maps to a shared reference
and compute mean composite maps.

Uses OpenCV (cv2) for template-matching-based alignment.  Do not import this
module outside of code that already requires cv2.

Functions
---------
list_animals
    List animal IDs present in an animals directory.
build_paths
    Construct .mat and widefield paths for a list of animal IDs.
load_maps_from_mat
    Load azimuth, altitude, and VFS maps from a .mat retinotopy file.
find_best_transforms
    Grid-search over rotation/scale to maximise template-matching correlation.
report_transform_ranges
    Print a summary of whether transform parameters hit search-grid limits.
align_single_vfs_to_reference
    Align one animal's VFS to the reference; return transform params + warped VFS.
compute_mean_vfs
    Average aligned VFS maps across animals.
compute_mean_azi_alt
    Average aligned azimuth and altitude maps across animals.
overlay_contours_on_transformed_vfs
    Display area contours over a transformed VFS.
plot_all_maps
    Tile reference and all animal VFS maps for inspection.
plot_vfs_overlays
    Overlay all aligned VFSs on one panel, plus per-animal panels.
plot_mean_vfs
    Plot (and optionally save) the mean composite VFS.
plot_mean_azi_alt
    Plot (and optionally save) mean azimuth and altitude maps.
save_transform_dict
    Serialise transform dict to JSON.
warp_with_transform
    Apply a stored transform dict to an image with cv2.warpAffine.


DMM, March 2026
"""

import os
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
import tifffile


def list_animals(animals_dir):
    """ Return sorted list of animal IDs in animals_dir. """
    return sorted([animal_id for animal_id in os.listdir(animals_dir)])


def build_paths(animals_dir, animal_ids, maps_path_end, widefield_path_end):
    """ Build full paths to retinotopy .mat files and widefield folders per animal. """
    maps_paths = [os.path.join(animals_dir, animal_id, maps_path_end) for animal_id in animal_ids]
    widefield_paths = [
        os.path.join(animals_dir, animal_id, widefield_path_end) for animal_id in animal_ids
    ]
    return maps_paths, widefield_paths


def load_maps_from_mat(mat_path):
    """ Load azimuth, altitude, and VFS maps from a retinotopy .mat file. """
    mat = loadmat(mat_path)
    azi_map = mat["maps"][0, 0]["HorizontalRetinotopy"]
    alt_map = mat["maps"][0, 0]["VerticalRetinotopy"]
    vfs_map = mat["maps"][0, 0]["VFS_raw"]
    return azi_map, alt_map, vfs_map


def plot_all_maps(
    animal_ids,
    maps_paths,
    reference_vfs_path,
    output_dir=None,
    save=False,
):
    """ Tile VFS maps for all animals alongside the reference map. """
    n_images = len(animal_ids) + 1
    n_cols = min(5, n_images)
    n_rows = int(np.ceil(n_images / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    ref_img = tifffile.imread(reference_vfs_path)
    axes[0].imshow(ref_img, cmap="jet")
    axes[0].set_title("Reference VFS (SAM028)")
    axes[0].axis("off")

    for idx, (animal_id, maps_path) in enumerate(zip(animal_ids, maps_paths), start=1):
        _, _, vfs_img = load_maps_from_mat(maps_path)
        ax = axes[idx]
        ax.imshow(vfs_img, cmap="jet")
        ax.set_title(animal_id)
        ax.axis("off")

    for ax in axes[n_images:]:
        ax.axis("off")

    plt.tight_layout()
    if save:
        if output_dir is None:
            raise ValueError("output_dir is required when save=True")
        save_path = os.path.join(output_dir, "vfs_reference_and_all.png")
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print('Saved: {}'.format(save_path))

    plt.show()
    return fig, axes


def _prepare_image(img, sigma=None):
    """ Mean-reduce, optional Gaussian smooth, then z-score an image. """
    img = np.asarray(img)
    if img.ndim > 2:
        img = np.mean(img, axis=0)
    img = img.astype(np.float32)
    if sigma is not None and sigma > 0:
        img = gaussian_filter(img, sigma=sigma)
    img = img - np.mean(img)
    std = np.std(img)
    if std > 0:
        img = img / std
    return img


def _to_display(img):
    """ Min-max normalise an image to [0, 1] for display. """
    img = np.asarray(img)
    if img.ndim > 2:
        img = np.mean(img, axis=0)
    img = img.astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)
    return (img - min_val) / (max_val - min_val + 1e-8)


def _warp(img, scale, rotation_deg, output_shape):
    """ Rotate and scale img around its centre using cv2.warpAffine. """
    h, w = output_shape
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, rotation_deg, scale)
    warped = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped, M


def _find_best_shift(ref_norm, moving_norm, max_translation):
    """ Template-match moving_norm within a padded ref_norm; return (dx, dy, corr). """
    pad = int(max_translation)
    ref_pad = cv2.copyMakeBorder(
        ref_norm, pad, pad, pad, pad, borderType=cv2.BORDER_CONSTANT, value=0
    )
    res = cv2.matchTemplate(ref_pad, moving_norm, method=cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    dx = max_loc[0] - pad
    dy = max_loc[1] - pad
    return dx, dy, max_val


def find_best_transforms(
    animal_ids,
    maps_paths,
    reference_vfs_path,
    vfs_sigma,
    rotation_degs,
    scale_factors,
    max_translation,
    verbose=True,
):
    """ Grid-search rotation/scale for each animal to maximise VFS-to-reference correlation.

    Returns transform_dict, aligned_vfs_dict, ref_norm, ref_disp, reference_vfs.
    """
    reference_vfs = tifffile.imread(reference_vfs_path)
    ref_norm = _prepare_image(reference_vfs, sigma=vfs_sigma)
    ref_disp = _to_display(reference_vfs)

    transform_dict = {}
    aligned_vfs_dict = {}

    for animal_id, maps_path in zip(animal_ids, maps_paths):
        if verbose:
            print('Finding best transform parameters for: {}'.format(animal_id))
        _, _, vfs_raw = load_maps_from_mat(maps_path)
        vfs_norm = _prepare_image(vfs_raw, sigma=vfs_sigma)

        best_corr = -np.inf
        best_params = None

        for rotation_deg in rotation_degs:
            for scale_factor in scale_factors:
                warped_norm, _ = _warp(vfs_norm, scale_factor, rotation_deg, ref_norm.shape)
                dx, dy, corr = _find_best_shift(ref_norm, warped_norm, max_translation=max_translation)

                if corr > best_corr:
                    best_corr = corr
                    best_params = (dx, dy, rotation_deg, scale_factor)

        best_dx, best_dy, best_rot, best_scale = best_params

        transform_dict[animal_id] = {
            "path": maps_path,
            "animal_id": animal_id,
            "dx": float(best_dx),
            "dy": float(best_dy),
            "rotation_deg": float(best_rot),
            "scale_factor": float(best_scale),
            "pearson_r": float(best_corr),
        }

        vfs_disp = _to_display(vfs_raw)
        h, w = ref_norm.shape
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, best_rot, best_scale)
        M[0, 2] += best_dx
        M[1, 2] += best_dy
        aligned = cv2.warpAffine(
            vfs_disp,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        aligned_vfs_dict[animal_id] = aligned

    return transform_dict, aligned_vfs_dict, ref_norm, ref_disp, reference_vfs


def report_transform_ranges(
    transform_dict,
    rotation_degs=np.arange(-8, 24.05, 0.5),
    scale_factors=np.arange(0.5, 1.505, 0.1),
    max_translation=200,
):
    """ Print range diagnostics for each transform parameter; flag when at search-grid boundary. """
    labels = list(transform_dict.keys())

    dx_vals = np.array([transform_dict[k]["dx"] for k in labels], dtype=float)
    dy_vals = np.array([transform_dict[k]["dy"] for k in labels], dtype=float)
    rot_vals = np.array([transform_dict[k]["rotation_deg"] for k in labels], dtype=float)
    scale_vals = np.array([transform_dict[k]["scale_factor"] for k in labels], dtype=float)
    pearson_vals = np.array([transform_dict[k]["pearson_r"] for k in labels], dtype=float)

    def _range_report(name, observed_vals, min_allowed, max_allowed, decimals=1, atol=1e-9):
        obs_min = float(np.min(observed_vals))
        obs_max = float(np.max(observed_vals))

        hit_min = np.isclose(obs_min, min_allowed, atol=atol) or (obs_min < min_allowed)
        hit_max = np.isclose(obs_max, max_allowed, atol=atol) or (obs_max > max_allowed)
        at_limit = hit_min or hit_max

        status = "at limit" if at_limit else "ok"

        fmt = '{{:.{}f}}'.format(decimals)
        print(
            '{} | input=[{}, {}] observed=[{}, {}] status = {}'.format(
                name,
                fmt.format(min_allowed), fmt.format(max_allowed),
                fmt.format(obs_min), fmt.format(obs_max),
                status
            )
        )

        return at_limit

    maxed_out = {
        "dx": _range_report("dx", dx_vals, -max_translation, max_translation, decimals=0),
        "dy": _range_report("dy", dy_vals, -max_translation, max_translation, decimals=0),
        "rotation": _range_report(
            "rotation",
            rot_vals,
            float(rotation_degs.min()),
            float(rotation_degs.max()),
            decimals=1,
        ),
        "scale": _range_report(
            "scale",
            scale_vals,
            float(scale_factors.min()),
            float(scale_factors.max()),
            decimals=1,
        ),
    }

    maxed_params = [name for name, is_maxed in maxed_out.items() if is_maxed]
    if maxed_params:
        print('\nShould expand range for: {}'.format(', '.join(maxed_params)))
    else:
        print('\nNo range expansion needed.')

    pearson_range = (float(pearson_vals.min()), float(pearson_vals.max()))
    print('\npearson_r observed=[{:.2f}, {:.2f}]'.format(pearson_range[0], pearson_range[1]))
    return maxed_out, pearson_range


def plot_vfs_overlays(
    aligned_vfs_dict,
    ref_disp,
    animal_ids,
    output_dir=None,
    save=False,
):
    """ Overlay all aligned VFSs on one panel plus individual per-animal panels. """
    n_animals = len(animal_ids)
    n_panels = n_animals + 1
    n_cols = min(5, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    ax0 = axes[0]
    if n_animals == 0:
        raise ValueError("No aligned VFSs found to plot.")
    alpha_all = 1 / n_animals
    for animal_id in animal_ids:
        ax0.imshow(aligned_vfs_dict[animal_id], cmap="jet", alpha=alpha_all)
    ax0.set_title("All VFS's overlayed (n={})".format(n_animals))
    ax0.axis("off")

    for idx, animal_id in enumerate(animal_ids, start=1):
        ax = axes[idx]
        ax.imshow(ref_disp, cmap="jet", alpha=0.5)
        ax.imshow(aligned_vfs_dict[animal_id], cmap="jet", alpha=0.5)
        ax.set_title(animal_id)
        ax.axis("off")

    for ax in axes[n_panels:]:
        ax.axis("off")

    plt.tight_layout()
    if save:
        if output_dir is None:
            raise ValueError("output_dir is required when save=True")
        save_path = os.path.join(output_dir, "vfs_overlays_aligned.png")
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print('Saved: {}'.format(save_path))

    plt.show()
    return fig, axes


def warp_with_transform(img, t, out_shape):
    """ Apply a stored transform dict (dx, dy, rotation_deg, scale_factor) to img. """
    h, w = out_shape
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, t["rotation_deg"], t["scale_factor"])
    M[0, 2] += t["dx"]
    M[1, 2] += t["dy"]
    warped = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped


def compute_mean_vfs(animal_ids, maps_paths, transform_dict, reference_vfs_path=None):
    """ Warp each animal's VFS to reference space and return the pixel-wise mean. """
    if len(animal_ids) == 0:
        raise ValueError("animal_ids is empty.")

    if reference_vfs_path is not None:
        target_shape = tifffile.imread(reference_vfs_path).shape[:2]
    else:
        _, _, sample_vfs = load_maps_from_mat(maps_paths[0])
        target_shape = sample_vfs.shape[:2]

    sum_vfs = np.zeros(target_shape, dtype=np.float64)

    for animal_id, maps_path in zip(animal_ids, maps_paths):
        if animal_id not in transform_dict:
            raise KeyError('Missing transform for {}'.format(animal_id))
        t = transform_dict[animal_id]

        _, _, vfs_raw = load_maps_from_mat(maps_path)
        vfs_warp = warp_with_transform(vfs_raw, t, target_shape)
        sum_vfs += vfs_warp

    num_images = len(animal_ids)
    mean_vfs = sum_vfs / num_images
    return mean_vfs.astype(np.float32), num_images


def plot_mean_vfs(
    animal_ids,
    maps_paths,
    transform_dict,
    reference_vfs_path=None,
    output_dir=None,
    save=False,
):
    """ Plot (and optionally save) the mean composite VFS tif. """
    mean_vfs, num_images = compute_mean_vfs(
        animal_ids=animal_ids,
        maps_paths=maps_paths,
        transform_dict=transform_dict,
        reference_vfs_path=reference_vfs_path,
    )

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(mean_vfs, cmap="jet")
    ax.set_title('Mean composite VFS (n={})'.format(num_images))
    ax.axis("off")

    plt.tight_layout()
    if save:
        if output_dir is None:
            raise ValueError("output_dir is required when save=True")
        save_path = os.path.join(output_dir, "mean_composite_vfs_uncompressed.tif")
        tifffile.imwrite(save_path, mean_vfs, compression=None)
        print('Saved: {}'.format(save_path))

    plt.show()
    return fig, ax, mean_vfs


def compute_mean_azi_alt(animal_ids, transform_dict, add_maps_paths, reference_vfs_path=None):
    """ Warp each animal's azimuth and altitude maps and return pixel-wise means. """
    if len(animal_ids) == 0:
        raise ValueError("animal_ids is empty.")

    if reference_vfs_path is not None:
        target_shape = tifffile.imread(reference_vfs_path).shape[:2]
    else:
        sample_azi, _, _ = load_maps_from_mat(add_maps_paths[0])
        target_shape = sample_azi.shape[:2]

    sum_azi = np.zeros(target_shape, dtype=np.float64)
    sum_alt = np.zeros(target_shape, dtype=np.float64)

    for animal_id, add_maps_path in zip(animal_ids, add_maps_paths):
        if animal_id not in transform_dict:
            raise KeyError('Missing transform for {}'.format(animal_id))
        t = transform_dict[animal_id]

        azi_map, alt_map, _ = load_maps_from_mat(add_maps_path)
        azi_warp = warp_with_transform(azi_map.astype(np.float32), t, target_shape)
        alt_warp = warp_with_transform(alt_map.astype(np.float32), t, target_shape)

        sum_azi += azi_warp
        sum_alt += alt_warp

    num_images = len(animal_ids)
    mean_azi = (sum_azi / num_images).astype(np.float32)
    mean_alt = (sum_alt / num_images).astype(np.float32)
    return mean_azi, mean_alt, num_images


def plot_mean_azi_alt(
    animal_ids,
    transform_dict,
    add_maps_paths,
    reference_vfs_path=None,
    output_dir=None,
    save=True,
):
    """ Plot (and optionally save) mean azimuth and altitude composite maps. """
    mean_azi, mean_alt, num_images = compute_mean_azi_alt(
        animal_ids=animal_ids,
        transform_dict=transform_dict,
        add_maps_paths=add_maps_paths,
        reference_vfs_path=reference_vfs_path,
    )

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(mean_azi, cmap="jet")
    axes[0].set_title('Mean composite aziPosMap (n={})'.format(num_images))
    axes[0].axis("off")

    axes[1].imshow(mean_alt, cmap="jet")
    axes[1].set_title('Mean composite altPosMap (n={})'.format(num_images))
    axes[1].axis("off")

    plt.tight_layout()
    if save:
        if output_dir is None:
            raise ValueError("output_dir is required when save=True")
        mean_azi_path = os.path.join(output_dir, "mean_composite_aziPosMap_uncompressed.tif")
        mean_alt_path = os.path.join(output_dir, "mean_composite_altPosMap_uncompressed.tif")
        tifffile.imwrite(mean_azi_path, mean_azi, compression=None)
        tifffile.imwrite(mean_alt_path, mean_alt, compression=None)
        print("Saved:")
        print(mean_azi_path)
        print(mean_alt_path)

    plt.show()
    return fig, axes, mean_azi, mean_alt


def save_transform_dict(transform_dict, output_dir, filename="vfs_composite_transforms.pkl"):
    """ Serialise transform dict to a JSON file in output_dir. """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)

    with open('save_path', 'w') as f:
        json.dump(transform_dict, f, indent=4)
        
    print('Saved: {}'.format(save_path))
    return save_path


def align_single_vfs_to_reference(
    additional_maps_path,
    reference_vfs_path,
    vfs_sigma=2,
    rotation_degs=np.arange(-8, 24.05, 0.5),
    scale_factors=np.arange(0.5, 1.505, 0.1),
    max_translation=200,
    verbose=True,
    animal_id=None,
):
    """ Align one animal's VFS to the reference; return transform_params and aligned VFS.

    Parameters
    ----------
    additional_maps_path : str
        Path to .mat retinotopy file for this animal.
    reference_vfs_path : str
        Path to reference VFS tif.
    verbose : bool
        Print best transform parameters when True.
    animal_id : str, optional
        Falls back to the parent directory name of additional_maps_path.
    """

    reference_vfs = tifffile.imread(reference_vfs_path)
    ref_norm = _prepare_image(reference_vfs, sigma=vfs_sigma)

    _, _, vfs_raw = load_maps_from_mat(additional_maps_path)
    vfs_norm = _prepare_image(vfs_raw, sigma=vfs_sigma)

    best_corr = -np.inf
    best_params = None

    for rotation_deg in rotation_degs:
        for scale_factor in scale_factors:
            warped_norm, _ = _warp(vfs_norm, scale_factor, rotation_deg, ref_norm.shape)
            dx, dy, corr = _find_best_shift(ref_norm, warped_norm, max_translation=max_translation)

            if corr > best_corr:
                best_corr = corr
                best_params = (dx, dy, rotation_deg, scale_factor)

    best_dx, best_dy, best_rot, best_scale = best_params

    if animal_id is None:
        animal_id = os.path.basename(os.path.dirname(additional_maps_path))
        print('Using animal_id={}. If incorrect, pass animal_id="<string>" argument.'.format(animal_id))

    transform_params = {
        "path": additional_maps_path,
        "animal_id": animal_id,
        "dx": float(best_dx),
        "dy": float(best_dy),
        "rotation_deg": float(best_rot),
        "scale_factor": float(best_scale),
        "pearson_r": float(best_corr),
        "reference_vfs_path": reference_vfs_path,
    }

    target_shape = reference_vfs.shape[:2]
    aligned_vfs = warp_with_transform(vfs_raw.astype(np.float32), transform_params, target_shape)

    if verbose:
        print(
            'Best transform | dx={:.0f}, dy={:.0f}, rot={:.1f}, scale={:.2f}, pearson_r={:.3f}'.format(
                best_dx, best_dy, best_rot, best_scale, best_corr
            )
        )

    return transform_params, aligned_vfs


def overlay_contours_on_transformed_vfs(
    additional_maps_path,
    transform_params,
    contours_path,
    reference_vfs_path=None,
    cmap="jet",
    contour_color="black",
    contour_alpha=0.9,
    contour_linewidth=2,
    labels=False,
    show=True,
):
    """ Display area contours from a JSON file over an aligned VFS map. """
    if reference_vfs_path is not None:
        target_shape = tifffile.imread(reference_vfs_path).shape[:2]
    else:
        _, _, vfs_raw = load_maps_from_mat(additional_maps_path)
        target_shape = vfs_raw.shape[:2]

    _, _, vfs_raw = load_maps_from_mat(additional_maps_path)
    aligned_vfs = warp_with_transform(vfs_raw.astype(np.float32), transform_params, target_shape)

    with open(contours_path, 'r') as file:
        contours = json.load(file)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(aligned_vfs, cmap=cmap)

    for area_name, coords in contours.items():
        if coords is None or len(coords) == 0:
            continue
        arr = np.asarray(coords)
        if arr.ndim != 2 or arr.shape[1] != 2:
            continue
        ax.plot(
            arr[:, 0],
            arr[:, 1],
            color=contour_color,
            alpha=contour_alpha,
            linewidth=contour_linewidth,
        )
        if labels:
            cx = float(np.nanmean(arr[:, 0]))
            cy = float(np.nanmean(arr[:, 1]))
            ax.text(
                cx,
                cy,
                str(area_name),
                color="white",
                fontsize=9,
                ha="center",
                va="center",
                bbox={"facecolor": "black", "edgecolor": "black", "boxstyle": "round,pad=0.2"},
            )

    ax.set_title('Aligned VFS with contours ({})'.format(transform_params['animal_id']))
    ax.axis("off")

    if show:
        plt.show()

    return fig, ax, aligned_vfs, contours
