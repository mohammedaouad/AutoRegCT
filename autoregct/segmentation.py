import os
import subprocess
import tempfile
import SimpleITK as sitk

# label indices from TotalSegmentator v2 class list
FEMUR_LABELS = {
    "femur_right": 72,
    "femur_left": 73,
}


def segment_femur(image, side, output_dir=None, fast=False, device="cpu"):
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got '{side}'")

    label_value = FEMUR_LABELS[f"femur_{side}"]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_ct = os.path.join(tmp, "ct.nii.gz")
        sitk.WriteImage(image, tmp_ct)
        seg_dir = output_dir if output_dir else tempfile.mkdtemp()
        _run_totalsegmentator(tmp_ct, seg_dir, fast=fast, device=device)
        return _extract_femur_mask(seg_dir, side, label_value)


def _run_totalsegmentator(input_path, output_dir, fast, device):
    os.makedirs(output_dir, exist_ok=True)
    cmd = ["TotalSegmentator", "-i", input_path, "-o", output_dir, "--device", device]
    if fast:
        cmd.append("--fast")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"TotalSegmentator failed:\n{result.stderr}")


def _extract_femur_mask(seg_dir, side, label_value):
    label_name = f"femur_{side}"

    # TS output format changed between versions-check all known locations
    candidates = [
        os.path.join(seg_dir, f"{label_name}.nii.gz"),
        os.path.join(seg_dir, f"{label_name}.nii"),
        os.path.join(seg_dir, "segmentations", f"{label_name}.nii.gz"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return _binarize(sitk.ReadImage(path))

    # older TS versions output a single multilabel file
    multilabel = os.path.join(seg_dir, "segmentation.nii.gz")
    if os.path.exists(multilabel):
        vol = sitk.ReadImage(multilabel)
        mask = sitk.Equal(sitk.Cast(vol, sitk.sitkUInt8), label_value)
        return sitk.Cast(mask, sitk.sitkUInt8)

    raise FileNotFoundError(
        f"Could not find {label_name} mask in {seg_dir}. "
        f"Checked: {candidates + [multilabel]}"
    )


def _binarize(mask):
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    return sitk.BinaryThreshold(mask, lowerThreshold=1, upperThreshold=255, insideValue=1, outsideValue=0)


def extract_largest_component(mask):
    # removes small floating islands from segmentation noise
    connected = sitk.ConnectedComponent(mask)
    relabeled = sitk.RelabelComponent(connected, sortByObjectSize=True)
    return sitk.Cast(sitk.Equal(relabeled, 1), sitk.sitkUInt8)


def load_mask_from_file(path):
    # use this when you already have a segmentation and want to skip TotalSegmentator
    mask = sitk.ReadImage(path)
    return _binarize(mask)
