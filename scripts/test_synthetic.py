import sys
import os
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autoregct.registration import register_femur, get_transform_params
from autoregct.metrics import compute_all_metrics, format_metrics
from autoregct.visualization import visualize_overlay, visualize_slices


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


def run(show=False):
    print("=== AutoRegCT Synthetic Test ===\n")

    print("[1/4] Generating masks...")
    postop = make_ellipsoid(center=[40, 40, 60])
    preop = make_ellipsoid(center=[43, 38, 58], angle_deg=8.0)

    print(f"  Pre-op voxels : {sitk.GetArrayFromImage(preop).sum()}")
    print(f"  Post-op voxels: {sitk.GetArrayFromImage(postop).sum()}")

    print("\n[2/4] Running registration...")
    registered, transform = register_femur(fixed_mask=postop, moving_mask=preop, verbose=True)

    matrix, translation = get_transform_params(transform)
    print(f"\n  Transform matrix:\n{matrix.round(4)}")
    print(f"  Translation (mm): {translation.round(2)}")

    reg_voxels = sitk.GetArrayFromImage(registered).sum()
    print(f"  Registered mask voxels: {reg_voxels}")
    if reg_voxels == 0:
        print("  WARNING: registered mask is empty — registration may have failed")

    print("\n[3/4] Computing metrics...")
    metrics = compute_all_metrics(registered, postop, transform)
    print(format_metrics(metrics))

    print("\n[4/4] Saving visualizations...")
    os.makedirs("test_output", exist_ok=True)
    visualize_slices(preop, postop, registered, output_path="test_output/slices.png", show=show)
    visualize_overlay(preop, postop, registered, metrics=metrics, output_path="test_output/overlay_3d.png", show=show)
    print("  Saved: test_output/slices.png")
    print("  Saved: test_output/overlay_3d.png")
    print("\nDone.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--show", action="store_true")
    args = p.parse_args()
    run(show=args.show)
