import streamlit as st
import numpy as np
import SimpleITK as sitk
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

# Real CT mode removed until pipeline is validated on actual THA data
# segmentation.py and run_pipeline.py still support it via CLI

from autoregct.registration import register_femur
from autoregct.metrics import compute_all_metrics
from autoregct.visualization import visualize_overlay, visualize_slices


st.set_page_config(page_title="AutoRegCT", page_icon="🦴", layout="wide")
st.title("🦴 AutoRegCT")
st.caption("Automatic pre/post-op CT registration — Total Hip Arthroplasty")
st.divider()


CASES = {
    "Case 1 — Standard THA (8° rotation, 3mm offset)": {
        "postop": {"center": [40, 40, 60], "radii": (14, 14, 38), "angle_deg": 0.0},
        "preop":  {"center": [43, 38, 58], "radii": (14, 14, 38), "angle_deg": 8.0},
        "note": "Baseline case. Closest to a real THA scenario. Registration should converge cleanly.",
        "metal_artifacts": False,
    },
    "Case 2 — Severe malrotation (20° rotation, 8mm offset)": {
        "postop": {"center": [40, 40, 60], "radii": (14, 14, 38), "angle_deg": 0.0},
        "preop":  {"center": [48, 36, 55], "radii": (14, 14, 38), "angle_deg": 20.0},
        "note": "Simulates a poorly positioned implant. Tests whether the optimizer still converges under harder conditions.",
        "metal_artifacts": False,
    },
    "Case 3 — Bone resection (5° rotation, larger pre-op femur)": {
        "postop": {"center": [40, 40, 60], "radii": (14, 14, 38), "angle_deg": 0.0},
        "preop":  {"center": [42, 40, 59], "radii": (17, 17, 44), "angle_deg": 5.0},
        "note": "Pre-op femur is larger; simulates bone removed during surgery. Lower Dice is expected here, not a registration failure.",
        "metal_artifacts": False,
    },
    "Case 4 — Metal artifact simulation (8° rotation + streak artifacts)": {
        "postop": {"center": [40, 40, 60], "radii": (14, 14, 38), "angle_deg": 0.0},
        "preop":  {"center": [43, 38, 58], "radii": (14, 14, 38), "angle_deg": 8.0},
        "note": "Same geometry as Case 1 but the post-op mask has simulated metal streak artifacts; high intensity spikes radiating from the implant center, mimicking what you see around metal in real post-op CT. Tests whether registration still converges with corrupted intensities.",
        "metal_artifacts": True,
    },
}


def make_ellipsoid(center, radii, angle_deg, shape=(80, 80, 120), spacing=(1.5, 1.5, 1.5)):
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dz, dy, dx = z - center[0], y - center[1], x - center[2]
    if angle_deg != 0.0:
        rad = np.radians(angle_deg)
        dz, dx = dz * np.cos(rad) - dx * np.sin(rad), dz * np.sin(rad) + dx * np.cos(rad)
    arr = ((dz / radii[0])**2 + (dy / radii[1])**2 + (dx / radii[2])**2 <= 1.0).astype(np.uint8)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    return img


def add_metal_artifacts(mask, n_streaks=12, streak_intensity=3, noise_level=0.15):
    arr = sitk.GetArrayFromImage(mask).astype(np.float32)
    shape = arr.shape
    center = np.array([shape[0] // 2, shape[1] // 2, shape[2] // 2])

    # create star-pattern streaks radiating from center in the axial plane
    # this mimics the beam hardening artifact pattern around metal implants
    for i in range(n_streaks):
        angle = (2 * np.pi * i) / n_streaks
        for r in range(max(shape)):
            z = int(center[0])
            y = int(center[1] + r * np.sin(angle))
            x = int(center[2] + r * np.cos(angle))
            if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
                arr[z, y, x] = min(arr[z, y, x] + streak_intensity * (1 - r / max(shape)), 4.0)

    # add some random noise on top to make it less clean
    noise = np.random.normal(0, noise_level, shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 4.0)

    result = sitk.GetImageFromArray(arr)
    result.SetSpacing(mask.GetSpacing())
    result.SetOrigin(mask.GetOrigin())
    result.SetDirection(mask.GetDirection())
    return result


def render_metrics(metrics):
    items = [
        ("Stem axis angle", "stem_angle_deg", "deg"),
        ("Centroid offset", "centroid_offset_mm", "mm"),
        ("Leg length change", "leg_length_change_mm", "mm"),
        ("Dice coefficient", "dice", ""),
        ("Coverage ratio", "coverage_ratio", ""),
        ("Transform rotation", "transform_rotation_deg", "deg"),
        ("Transform translation", "transform_translation_mm", "mm"),
    ]
    cols = st.columns(2)
    for i, (label, key, unit) in enumerate(items):
        if key not in metrics:
            continue
        val = metrics[key]
        if key == "leg_length_change_mm":
            display = f"{val:+.2f}"
        elif key in ("dice", "coverage_ratio"):
            display = f"{val:.4f}"
        else:
            display = f"{val:.2f}"
        with cols[i % 2]:
            col = cols[i % 2]
            col.metric(label, f"{display} {unit}".strip())


with st.sidebar:
    st.header("Settings")
    selected = st.radio("Select test case", list(CASES.keys()))
    st.info(CASES[selected]["note"])
    run = st.button("Run Pipeline", type="primary")


if run:
    case = CASES[selected]

    with st.spinner("Generating masks..."):
        postop = make_ellipsoid(**case["postop"])
        preop = make_ellipsoid(**case["preop"])

        if case["metal_artifacts"]:
            np.random.seed(42)
            postop_reg = add_metal_artifacts(postop)
        else:
            postop_reg = postop

    with st.spinner("Registering..."):
        # use artifact version for registration, clean mask for metrics
        registered, transform = register_femur(fixed_mask=postop_reg, moving_mask=preop, verbose=False)

    with st.spinner("Computing metrics..."):
        # compute metrics against clean post-op mask
        metrics = compute_all_metrics(registered, postop, transform)

    st.success("Done")

    if case["metal_artifacts"]:
        st.warning("Metal artifacts were added to the post-op mask before registration. Metrics are computed against the clean mask.")

    st.subheader("Metrics")
    render_metrics(metrics)
    st.divider()

    with st.spinner("Rendering visualizations..."):
        with tempfile.TemporaryDirectory() as tmp:
            slice_path = os.path.join(tmp, "slices.png")
            overlay_path = os.path.join(tmp, "overlay.png")
            # binarize postop_reg before visualization as artifact intensities above 1 break the overlay
            postop_vis = sitk.BinaryThreshold(
                sitk.Cast(postop_reg, sitk.sitkFloat32),
                lowerThreshold=0.5, upperThreshold=100.0,
                insideValue=1, outsideValue=0
            )
            postop_vis = sitk.Cast(postop_vis, sitk.sitkUInt8)
            visualize_slices(preop, postop_vis, registered, output_path=slice_path, show=False)
            visualize_overlay(preop, postop_vis, registered, metrics=metrics, output_path=overlay_path, show=False)

            st.subheader("Slice Overlay")
            st.image(slice_path, use_container_width=True)
            st.subheader("3D Overlay")
            st.image(overlay_path, use_container_width=True)

    st.divider()
    st.download_button("Download metrics (JSON)", json.dumps(metrics, indent=2), file_name="metrics.json", mime="application/json")

else:
    st.info("Select a test case in the sidebar and click **Run Pipeline** to start.")