import sys
import os
import numpy as np
import SimpleITK as sitk
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autoregct.registration import register_femur
from autoregct.metrics import compute_all_metrics
from autoregct.io import resample_to_reference, check_compatible


def make_ellipsoid(shape=(80, 80, 120), spacing=(1.5, 1.5, 1.5), center=None, radii=(14, 14, 38), angle_deg=0.0):
    if center is None:
        center = [s // 2 for s in shape]
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dz, dy, dx = z - center[0], y - center[1], x - center[2]
    if angle_deg != 0.0:
        rad = np.radians(angle_deg)
        dz, dx = dz * np.cos(rad) - dx * np.sin(rad), dz * np.sin(rad) + dx * np.cos(rad)
    arr = ((dz / radii[0])**2 + (dy / radii[1])**2 + (dx / radii[2])**2 <= 1.0).astype(np.uint8)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    return img


@pytest.fixture(scope="module")
def synthetic_pair():
    postop = make_ellipsoid(center=[40, 40, 60])
    preop = make_ellipsoid(center=[43, 38, 58], angle_deg=8.0)
    return preop, postop


@pytest.fixture(scope="module")
def registration_result(synthetic_pair):
    preop, postop = synthetic_pair
    registered, transform = register_femur(fixed_mask=postop, moving_mask=preop, verbose=False)
    return registered, transform, postop


def test_registered_mask_not_empty(registration_result):
    registered, _, _ = registration_result
    arr = sitk.GetArrayFromImage(registered)
    assert arr.sum() > 0, "Registered mask is empty — registration failed"


def test_rotation_recovered(registration_result):
    from autoregct.registration import get_transform_params
    from scipy.spatial.transform import Rotation
    _, transform, _ = registration_result
    matrix, _ = get_transform_params(transform)
    r = Rotation.from_matrix(matrix)
    angle = np.degrees(r.magnitude())
    assert abs(angle - 8.0) < 2.0, f"Expected ~8° rotation, got {angle:.2f}°"


def test_metrics_keys_present(registration_result):
    registered, transform, postop = registration_result
    metrics = compute_all_metrics(registered, postop, transform)
    expected_keys = [
        "stem_angle_deg", "centroid_offset_mm", "leg_length_change_mm",
        "dice", "coverage_ratio", "transform_rotation_deg", "transform_translation_mm"
    ]
    for key in expected_keys:
        assert key in metrics, f"Missing metric: {key}"


def test_dice_reasonable(registration_result):
    registered, transform, postop = registration_result
    metrics = compute_all_metrics(registered, postop, transform)
    assert metrics["dice"] > 0.5, f"Dice too low: {metrics['dice']:.4f}"


def test_resample_to_reference():
    img_a = make_ellipsoid(shape=(60, 60, 80), spacing=(2.0, 2.0, 2.0))
    img_b = make_ellipsoid(shape=(80, 80, 120), spacing=(1.5, 1.5, 1.5))
    resampled = resample_to_reference(img_a, img_b)
    assert resampled.GetSize() == img_b.GetSize()
    assert resampled.GetSpacing() == img_b.GetSpacing()
