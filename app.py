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
        "note": "Baseline case. Closest to a real THA scenario. Registration converge cleanly.",
    },
    "Case 2 — Severe malrotation (20° rotation, 8mm offset)": {
        "postop": {"center": [40, 40, 60], "radii": (14, 14, 38), "angle_deg": 0.0},
        "preop":  {"center": [48, 36, 55], "radii": (14, 14, 38), "angle_deg": 20.0},
        "note": "Simulates a poorly positioned implant. Tests whether the optimizer still converges under harder conditions.",
    },
    "Case 3 — Bone resection (5° rotation, larger pre-op femur)": {
        "postop": {"center": [40, 40, 60], "radii": (14, 14, 38), "angle_deg": 0.0},
        "preop":  {"center": [42, 40, 59], "radii": (17, 17, 44), "angle_deg": 5.0},
        "note": "Pre-op femur is larger; simulates bone removed during surgery. Lower Dice is expected here, not a registration failure.",
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

    with st.spinner("Registering..."):
        registered, transform = register_femur(fixed_mask=postop, moving_mask=preop, verbose=False)

    with st.spinner("Computing metrics..."):
        metrics = compute_all_metrics(registered, postop, transform)

    st.success("Done")
    st.subheader("Metrics")

    col1, col2 = st.columns(2)
    col1.metric("Stem axis angle", f"{metrics['stem_angle_deg']:.2f}°")
    col2.metric("Centroid offset", f"{metrics['centroid_offset_mm']:.2f} mm")
    col1.metric("Leg length change", f"{metrics['leg_length_change_mm']:+.2f} mm")
    col2.metric("Dice coefficient", f"{metrics['dice']:.4f}")
    col1.metric("Coverage ratio", f"{metrics['coverage_ratio']:.4f}")
    col2.metric("Transform rotation", f"{metrics['transform_rotation_deg']:.2f}°")
    col1.metric("Transform translation", f"{metrics['transform_translation_mm']:.2f} mm")

    st.divider()

    with st.spinner("Rendering visualizations..."):
        with tempfile.TemporaryDirectory() as tmp:
            slice_path = os.path.join(tmp, "slices.png")
            overlay_path = os.path.join(tmp, "overlay.png")
            visualize_slices(preop, postop, registered, output_path=slice_path, show=False)
            visualize_overlay(preop, postop, registered, metrics=metrics, output_path=overlay_path, show=False)

            st.subheader("Slice Overlay")
            st.image(slice_path, use_container_width=True)

            st.subheader("3D Overlay")
            st.image(overlay_path, use_container_width=True)

    st.divider()
    st.download_button("Download metrics (JSON)", json.dumps(metrics, indent=2), file_name="metrics.json", mime="application/json")

else:
    st.info("Select a test case in the sidebar and click **Run Pipeline** to start.")