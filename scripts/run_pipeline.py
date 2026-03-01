import argparse
import json
import os
import sys

import SimpleITK as sitk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autoregct.io import load_volume, save_volume, resample_to_reference
from autoregct.segmentation import segment_femur, extract_largest_component
from autoregct.registration import register_femur, apply_transform_to_volume
from autoregct.metrics import compute_all_metrics, format_metrics
from autoregct.visualization import visualize_overlay, visualize_slices


def parse_args():
    p = argparse.ArgumentParser(description="AutoRegCT — automatic pre/post-op CT registration for THA")
    p.add_argument("--preop", required=True, help="Pre-op CT path (DICOM dir or .nii/.nii.gz)")
    p.add_argument("--postop", required=True, help="Post-op CT path (DICOM dir or .nii/.nii.gz)")
    p.add_argument("--side", required=True, choices=["left", "right"])
    p.add_argument("--out", default="./output")
    p.add_argument("--fast", action="store_true", help="TotalSegmentator fast mode")
    p.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--skip-seg", action="store_true", help="Reuse existing masks from --out")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-vis", action="store_true")
    p.add_argument("--save-volumes", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("[1/5] Loading CT volumes...")
    preop_ct = load_volume(args.preop)
    postop_ct = load_volume(args.postop)

    preop_seg_dir = os.path.join(args.out, "seg_preop")
    postop_seg_dir = os.path.join(args.out, "seg_postop")

    if args.skip_seg:
        print("[2/5] Loading existing segmentations...")
        preop_mask = sitk.ReadImage(os.path.join(preop_seg_dir, f"femur_{args.side}.nii.gz"))
        postop_mask = sitk.ReadImage(os.path.join(postop_seg_dir, f"femur_{args.side}.nii.gz"))
    else:
        print("[2/5] Segmenting femur (this takes a few minutes)...")
        preop_mask = segment_femur(preop_ct, side=args.side, output_dir=preop_seg_dir, fast=args.fast, device=args.device)
        postop_mask = segment_femur(postop_ct, side=args.side, output_dir=postop_seg_dir, fast=args.fast, device=args.device)
        preop_mask = extract_largest_component(preop_mask)
        postop_mask = extract_largest_component(postop_mask)
        save_volume(preop_mask, os.path.join(preop_seg_dir, f"femur_{args.side}.nii.gz"))
        save_volume(postop_mask, os.path.join(postop_seg_dir, f"femur_{args.side}.nii.gz"))

    print("[3/5] Registering pre-op femur to post-op space...")
    preop_mask_r = resample_to_reference(preop_mask, postop_mask, interpolator=sitk.sitkNearestNeighbor)
    registered_mask, transform = register_femur(fixed_mask=postop_mask, moving_mask=preop_mask_r, verbose=args.verbose)

    if args.save_volumes:
        preop_ct_r = resample_to_reference(preop_ct, postop_ct)
        registered_ct = apply_transform_to_volume(preop_ct_r, postop_ct, transform)
        save_volume(registered_ct, os.path.join(args.out, "registered_preop_ct.nii.gz"))
        save_volume(registered_mask, os.path.join(args.out, "registered_preop_mask.nii.gz"))

    print("[4/5] Computing metrics...")
    metrics = compute_all_metrics(registered_mask, postop_mask, transform)
    print(format_metrics(metrics))

    metrics_path = os.path.join(args.out, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {metrics_path}")

    if not args.no_vis:
        print("[5/5] Generating visualizations...")
        visualize_slices(preop_mask_r, postop_mask, registered_mask, output_path=os.path.join(args.out, "overlay_slices.png"), show=False)
        visualize_overlay(preop_mask_r, postop_mask, registered_mask, metrics=metrics, output_path=os.path.join(args.out, "overlay_3d.png"), show=False)
        print(f"  Saved visualizations to {args.out}/")
    else:
        print("[5/5] Skipping visualization.")

    print(f"\nDone. All outputs in: {args.out}")


if __name__ == "__main__":
    main()
