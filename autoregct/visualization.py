import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    from skimage.measure import marching_cubes
    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False


def visualize_overlay(preop_mask, postop_mask, registered_preop_mask=None, metrics=None, output_path=None, show=True):
    if not _SKIMAGE:
        raise ImportError("scikit-image required: pip install scikit-image")

    fig = plt.figure(figsize=(16, 8))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    _render_3d(ax3d, preop_mask, postop_mask, registered_preop_mask)

    ax_metrics = fig.add_subplot(1, 2, 2)
    if metrics:
        _render_metrics(ax_metrics, metrics)
    else:
        ax_metrics.axis("off")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _render_3d(ax, preop_mask, postop_mask, registered_preop_mask=None):
    postop_arr = sitk.GetArrayFromImage(postop_mask)
    spacing = postop_mask.GetSpacing()

    _add_mesh(ax, postop_arr, spacing, color="#4A90D9", alpha=0.6)

    if registered_preop_mask is not None:
        preop_arr = sitk.GetArrayFromImage(registered_preop_mask)
        _add_mesh(ax, preop_arr, spacing, color="#E87040", alpha=0.3)
        # implant proxy: anything in post-op that wasn't in pre-op
        implant = np.logical_and(postop_arr > 0, preop_arr == 0).astype(np.uint8)
        if implant.sum() > 0:
            _add_mesh(ax, implant, spacing, color="#A0D468", alpha=0.9)
    else:
        preop_arr = sitk.GetArrayFromImage(preop_mask)
        _add_mesh(ax, preop_arr, spacing, color="#E87040", alpha=0.3)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("3D Overlay")

    patches = [
        mpatches.Patch(color="#4A90D9", label="Post-op femur"),
        mpatches.Patch(color="#E87040", label="Pre-op femur"),
        mpatches.Patch(color="#A0D468", label="Implant region"),
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=8)


def _add_mesh(ax, arr, spacing, color, alpha):
    # downsample 2x before marching cubes to avoid memory issues on large volumes
    down = arr[::2, ::2, ::2]
    sp = (spacing[0] * 2, spacing[1] * 2, spacing[2] * 2)
    try:
        verts, faces, _, _ = marching_cubes(down, level=0.5, spacing=sp)
    except (ValueError, RuntimeError):
        return
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    mesh.set_facecolor(color)
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)
    ax.set_xlim(0, arr.shape[2] * spacing[0])
    ax.set_ylim(0, arr.shape[1] * spacing[1])
    ax.set_zlim(0, arr.shape[0] * spacing[2])


def _render_metrics(ax, metrics):
    ax.axis("off")
    ax.set_title("Registration Metrics", fontsize=12, pad=10)
    labels = {
        "stem_angle_deg": ("Stem axis angle", "deg"),
        "centroid_offset_mm": ("Centroid offset", "mm"),
        "leg_length_change_mm": ("Leg length change", "mm"),
        "dice": ("Dice coefficient", ""),
        "coverage_ratio": ("Coverage ratio", ""),
        "transform_rotation_deg": ("Transform rotation", "deg"),
        "transform_translation_mm": ("Transform translation", "mm"),
    }
    y = 0.95
    for key, (label, unit) in labels.items():
        if key not in metrics:
            continue
        val = metrics[key]
        sign = "+" if key == "leg_length_change_mm" and val > 0 else ""
        ax.text(0.05, y, f"{label}: {sign}{val:.2f} {unit}".strip(), transform=ax.transAxes, fontsize=10, verticalalignment="top")
        y -= 0.10


def visualize_slices(preop_mask, postop_mask, registered_preop_mask=None, output_path=None, show=True):
    postop_arr = sitk.GetArrayFromImage(postop_mask)
    preop_arr = sitk.GetArrayFromImage(preop_mask)
    reg_arr = sitk.GetArrayFromImage(registered_preop_mask) if registered_preop_mask else None

    center = _center_slice(postop_arr)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, plane in zip(axes, ["axial", "coronal", "sagittal"]):
        post_sl = _get_slice(postop_arr, center, plane)
        pre_sl = _get_slice(preop_arr, center, plane)
        rgb = np.zeros((*post_sl.shape, 3), dtype=float)
        rgb[..., 2] = post_sl.astype(float)  # blue = post-op
        rgb[..., 0] = pre_sl.astype(float)   # red = pre-op
        if reg_arr is not None:
            rgb[..., 1] = _get_slice(reg_arr, center, plane).astype(float) * 0.5  # green = registered
        ax.imshow(rgb, origin="lower")
        ax.set_title(f"{plane.capitalize()} (slice {center})")
        ax.axis("off")

    plt.suptitle("Slice Overlay — Blue: post-op | Red: pre-op | Green: registered", fontsize=10)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _center_slice(arr):
    nz = np.argwhere(arr > 0)
    if len(nz) == 0:
        return arr.shape[0] // 2
    return int(nz[:, 0].mean())


def _get_slice(arr, idx, plane):
    idx = min(idx, arr.shape[0] - 1)
    if plane == "axial":
        return arr[idx, :, :]
    elif plane == "coronal":
        return arr[:, arr.shape[1] // 2, :]
    else:
        return arr[:, :, arr.shape[2] // 2]
