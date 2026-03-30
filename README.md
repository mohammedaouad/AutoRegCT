# AutoRegCT

During my bachelor thesis I spent a lot of time aligning hip CT scans by hand in 3D Slicer and ImFusion. Pick landmarks manually, drag one scan over the other, read off measurements. It worked, but 20-30 minutes per patient is a lot of time, and the results were different depending on how carefully you placed each landmark. This project automates that.

Given a pre-op and post-op CT from the same THA patient, AutoRegCT segments the femur from both scans, registers the pre-op bone into the post-op space, and computes implant positioning metrics without any manual steps.

---

## What it does

1. Loads pre-op and post-op CT; DICOM folder or NIfTI file
2. Segments the femur using TotalSegmentator
3. Rigidly registers the pre-op femur to post-op space (SimpleITK, Mattes mutual information, 3-level pyramid)
4. Computes 7 positioning metrics from the aligned masks
5. Generates a 3D overlay and slice view

There's also a Streamlit GUI for running synthetic test cases directly in the browser.

---

## Installation
```bash
git clone https://github.com/mohammedaouad/AutoRegCT.git
cd AutoRegCT
```

**Windows** — just double-click `launch.bat`. It creates a virtual environment, installs everything, and opens the GUI. First run takes a few minutes.

**Manual:**
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Python 3.10+ required. TotalSegmentator downloads model weights (~1GB) on first use.

---

## CLI usage
```bash
python scripts/run_pipeline.py \
    --preop  /data/patient01/preop/ \
    --postop /data/patient01/postop/ \
    --side   right \
    --out    /data/patient01/results/
```

| Flag | Description |
|---|---|
| `--side` | `left` or `right` — required |
| `--fast` | TotalSegmentator fast mode, lower accuracy but much quicker |
| `--device` | `cpu` or `gpu` |
| `--skip-seg` | Reuse existing masks from a previous run |
| `--save-volumes` | Also write the registered CT volume to disk |
| `--verbose` | Print registration iteration logs |
| `--no-vis` | Skip visualization |

---

## Metrics

| Metric | What it measures |
|---|---|
| `stem_angle_deg` | Angle between pre-op and post-op femur axes |
| `centroid_offset_mm` | Distance between femur centroids after registration |
| `leg_length_change_mm` | Change in femoral length (LLD) along the mechanical axis |
| `dice` | Overlap between registered pre-op and post-op masks |
| `coverage_ratio` | Fraction of pre-op femur covered by post-op mask |
| `transform_rotation_deg` | Total rotation applied by the registration |
| `transform_translation_mm` | Total translation applied by the registration |

If Dice drops below >0.4 or centroid offset exceeds 30mm the pipeline prints a warning.

---

## Validation

No patient data yet, so I validated on synthetic data, two ellipsoids where one is shifted and tilted by a known amount to the other.

Three test cases are available in the GUI:

| Case | Setup | What to look at |
|---|---|---|
| Standard THA | 8° rotation, 3mm offset | Transform rotation should recover ~8° |
| Severe malrotation | 20° rotation, 8mm offset | Tests whether the optimizer still converges |
| Bone resection | 5° rotation, larger pre-op femur | Lower Dice is expected; the shapes changed, not the registration |

Result example: Case 1 - transform rotation **8.03°**, stem axis angle **0.91°** after registration.

Real CT validation is in progress using  private THA data.
---

## Running tests
```bash
pytest tests/
```

Covers: registered mask not empty, rotation recovered within 1° of ground truth, all 7 metric keys present, Dice above 0.5, resampling output shape correct.

---

## Stack

- **SimpleITK** — image I/O, resampling, rigid registration
- **TotalSegmentator** — femur segmentation (nnU-Net)
- **NumPy / SciPy** — SVD axis fitting, rotation math
- **scikit-image** — marching cubes for 3D mesh rendering
- **Matplotlib** — slice overlay and 3D visualization
- **Streamlit** — local browser GUI

---

## Notes

Registration uses Mattes mutual information (MMI) rather than mean squares, it handles intensity differences between pre and post-op scans better. The 3-level pyramid (4x → 2x → 1x) helps avoid local minima before refining at full resolution.

The implant region in the 3D overlay is `postop AND NOT registered_preop`. On real data this approximates where the prosthesis sits, though it also picks up bone remodeling and positioning differences.
