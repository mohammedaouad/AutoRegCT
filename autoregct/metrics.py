import numpy as np
import SimpleITK as sitk
from scipy.spatial.transform import Rotation

# thresholds for flagging: registration failure
_DICE_WARN = 0.4
_OFFSET_WARN = 30.0  # mm


def compute_all_metrics(preop_mask, postop_mask, transform):
    preop_pts = _mask_to_points(preop_mask)
    postop_pts = _mask_to_points(postop_mask)

    preop_axis, preop_centroid = _fit_axis(preop_pts)
    postop_axis, postop_centroid = _fit_axis(postop_pts)

    metrics = {
        "stem_angle_deg": _angle_between(preop_axis, postop_axis),
        "centroid_offset_mm": float(np.linalg.norm(preop_centroid - postop_centroid)),
        "leg_length_change_mm": _leg_length_change(preop_pts, postop_pts, postop_axis),
        "dice": _dice(preop_mask, postop_mask),
        "coverage_ratio": _coverage(preop_mask, postop_mask),
        "transform_rotation_deg": _rotation_magnitude(transform),
        "transform_translation_mm": _translation_magnitude(transform),
    }

    if metrics["dice"] < _DICE_WARN:
        print(f"  WARNING: Dice={metrics['dice']:.3f} is very low — registration may have failed.")
    if metrics["centroid_offset_mm"] > _OFFSET_WARN:
        print(f"  WARNING: centroid offset={metrics['centroid_offset_mm']:.1f}mm — check registration result.")

    return metrics


def _mask_to_points(mask):
    arr = sitk.GetArrayFromImage(mask)
    indices = np.argwhere(arr > 0)
    # sitk uses x,y,z ordering but numpy uses z,y,x - need to flip
    spacing = np.array(mask.GetSpacing())[::-1]
    origin = np.array(mask.GetOrigin())[::-1]
    return indices * spacing + origin


def _fit_axis(points):
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    axis = vh[0]
    return axis / np.linalg.norm(axis), centroid


def _angle_between(a, b):
    cos = np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0)
    return float(np.degrees(np.arccos(np.abs(cos))))


def _leg_length_change(preop_pts, postop_pts, axis):
    preop_proj = preop_pts @ axis
    postop_proj = postop_pts @ axis
    return float(
        (postop_proj.max() - postop_proj.min()) -
        (preop_proj.max() - preop_proj.min())
    )


def _dice(mask_a, mask_b):
    a = sitk.GetArrayFromImage(mask_a).astype(bool)
    b = sitk.GetArrayFromImage(mask_b).astype(bool)
    intersection = np.logical_and(a, b).sum()
    total = a.sum() + b.sum()
    if total == 0:
        return 1.0
    return float(2 * intersection / total)


def _coverage(preop_mask, postop_mask):
    preop = sitk.GetArrayFromImage(preop_mask).astype(bool)
    postop = sitk.GetArrayFromImage(postop_mask).astype(bool)
    if preop.sum() == 0:
        return 0.0
    return float(np.logical_and(preop, postop).sum() / preop.sum())


def _rotation_magnitude(transform):
    try:
        t = transform if not isinstance(transform, sitk.CompositeTransform) \
            else transform.GetNthTransform(0)
        r = Rotation.from_matrix(np.array(t.GetMatrix()).reshape(3, 3))
        return float(np.degrees(r.magnitude()))
    except Exception:
        return 0.0


def _translation_magnitude(transform):
    try:
        t = transform if not isinstance(transform, sitk.CompositeTransform) \
            else transform.GetNthTransform(0)
        return float(np.linalg.norm(np.array(t.GetTranslation())))
    except Exception:
        return 0.0


def format_metrics(metrics):
    lines = ["=== AutoRegCT Metrics ==="]
    lines.append(f"  Stem axis angle diff     : {metrics['stem_angle_deg']:.2f} deg")
    lines.append(f"  Centroid offset          : {metrics['centroid_offset_mm']:.2f} mm")
    lines.append(f"  Leg length change        : {metrics['leg_length_change_mm']:+.2f} mm")
    lines.append(f"  Dice coefficient         : {metrics['dice']:.4f}")
    lines.append(f"  Coverage ratio           : {metrics['coverage_ratio']:.4f}")
    lines.append(f"  Transform rotation       : {metrics['transform_rotation_deg']:.2f} deg")
    lines.append(f"  Transform translation    : {metrics['transform_translation_mm']:.2f} mm")
    return "\n".join(lines)
