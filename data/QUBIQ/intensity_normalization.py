"""
QUBIQ intensity normalization (per task) + geometry-safe label alignment,
saved per *subtask*.

Output:
OUT_ROOT/{task_name}/{taskXX}/{split}/image_{k}.nii.gz
OUT_ROOT/{task_name}/{taskXX}/{split}/label_{k}_segYY.nii.gz

Special-case:
- If task_name == "brain-tumor": force everything to 2D by extracting the first slice
  from image.nii.gz (z=0). This avoids 3D direction (len=9) vs 2D direction (len=4)
  mismatch and prevents empty labels due to physical-space mismatch.
"""

import os
import glob
import re
import numpy as np
import SimpleITK as sitk


# -------------------------
# Config (edit these)
# -------------------------
IN_ROOT = "./training_data_v3_QC"                 # root that contains task folders
TASK_NAME = "prostate"           # run one task at a time
SPLIT = "Training"                  # "Training" or "Validation"
OUT_ROOT = "./tmp_normalized_qubiq" # output root

# Normalization strategy:
if TASK_NAME in ["prostate", "brain-tumor", "brain-growth"]:
    IS_CT = False
else:
    IS_CT = True

# CT window (HU)
# Common defaults; adjust per task if needed
CT_LIR = -125
CT_HIR = 275
CT_SCALE_TO_255 = True 

# For MRI tasks, z-score is usually safer than windowing.
# USE_ZSCORE = True
# CLIP_Z = 5.0  # optional clamp after zscore (helps outliers); set None to disable
HIST_CUT_TOP = 0.5  # optional: cut top X percent of intensity histogram for MRI (e.g. 0.5 => 99.5th percentile)


# -------------------------
# Helpers
# -------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def extract_first_slice_as_2d(img3d: sitk.Image) -> sitk.Image:
    """Extract z=0 slice from a 3D image as a 2D SimpleITK image."""
    if img3d.GetDimension() != 3:
        return img3d
    if img3d.GetSize()[2] < 1:
        raise RuntimeError("Invalid 3D image with zero slices.")
    extract = sitk.ExtractImageFilter()
    extract.SetIndex([0, 0, 0])
    extract.SetSize([img3d.GetSize()[0], img3d.GetSize()[1], 0])  # collapse z
    return extract.Execute(img3d)

def intensity_normalize(img: sitk.Image) -> sitk.Image:
    arr = sitk.GetArrayFromImage(img).astype(np.float32)

    if IS_CT:
        # HU window + normalize
        arr = np.clip(arr, CT_LIR, CT_HIR)
        if CT_SCALE_TO_255:
            arr = (arr - CT_LIR) / (CT_HIR - CT_LIR + 1e-8) * 255.0
        else:
            arr = (arr - CT_LIR) / (CT_HIR - CT_LIR + 1e-8)
    else:
        # MRI: optional histogram-top clipping to remove extreme bright outliers,
        # then z-score (preferred) or min-max scaling.
        if HIST_CUT_TOP is not None and HIST_CUT_TOP > 0.0:
            hir = float(np.percentile(arr, 100.0 - HIST_CUT_TOP))
            arr[arr > hir] = hir

        # if USE_ZSCORE:
        #     mu = float(arr.mean())
        #     sigma = float(arr.std()) + 1e-8
        #     arr = (arr - mu) / sigma
        #     if CLIP_Z is not None:
        #         arr = np.clip(arr, -CLIP_Z, CLIP_Z)
        # else:
        #     mn, mx = float(arr.min()), float(arr.max())
        #     arr = (arr - mn) / (mx - mn + 1e-8)

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out

def align_to_ref(ref: sitk.Image, moving: sitk.Image, is_label=True) -> sitk.Image:
    """
    Resample moving onto ref grid safely.
    Handles:
    - 2D ref + 2D moving
    - 2D ref + 3D(Z=1) moving (extract slice then resample)
    """
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear

    if ref.GetDimension() == 2 and moving.GetDimension() == 3 and moving.GetSize()[2] == 1:
        moving = extract_first_slice_as_2d(moving)

    if ref.GetDimension() == 2 and moving.GetDimension() == 2:
        return sitk.Resample(moving, ref, sitk.Transform(), interp, 0, moving.GetPixelID())

    if ref.GetDimension() == moving.GetDimension():
        return sitk.Resample(moving, ref, sitk.Transform(), interp, 0, moving.GetPixelID())

    # best-effort fallback (rare)
    return sitk.Resample(moving, ref, sitk.Transform(), interp, 0, moving.GetPixelID())

def parse_task_seg(filepath: str):
    """
    Parse 'task01_seg07.nii.gz' -> (task_id=1, rater_id=7)
    """
    base = os.path.basename(filepath)
    if base.startswith("case"):
        m = re.match(r"case(\d+)_seg(\d+)\.nii(\.gz)?$", base)
        return 1, int(m.group(2))
    m = re.match(r"task(\d+)_seg(\d+)\.nii(\.gz)?$", base)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


# -------------------------
# Main
# -------------------------
def main():
    in_split_root = os.path.join(IN_ROOT, TASK_NAME, SPLIT)
    case_dirs = sorted([p for p in glob.glob(os.path.join(in_split_root, "*")) if os.path.isdir(p)])
    if not case_dirs:
        raise RuntimeError(f"No case folders found under: {in_split_root}")

    reindex = 0
    for case_dir in case_dirs:
        img_path = os.path.join(case_dir, "image.nii.gz")
        if not os.path.exists(img_path):
            print(f"[SKIP] Missing image: {img_path}")
            continue

        # collect labels grouped by subtask
        label_paths = sorted(glob.glob(os.path.join(case_dir, "task*_seg*.nii*")))
        if not label_paths:
            label_paths = sorted(glob.glob(os.path.join(case_dir, "case*_seg*.nii*")))
        labels_by_task = {}  # task_id -> list of (rater_id, path)
        for lp in label_paths:
            parsed = parse_task_seg(lp)
            if parsed is None:
                continue
            task_id, rater_id = parsed
            labels_by_task.setdefault(task_id, []).append((rater_id, lp))

        if not labels_by_task:
            print(f"[WARN] No labels found in {case_dir}")
            reindex += 1
            continue

        # read image and define reference geometry
        img_obj = sitk.ReadImage(img_path)
        if TASK_NAME == "brain-tumor":
            img_ref = extract_first_slice_as_2d(img_obj)  # enforce 2D
        else:
            img_ref = img_obj

        img_norm = intensity_normalize(img_ref)

        # save per subtask folder
        for task_id, items in sorted(labels_by_task.items()):
            subtask_name = f"task{task_id:02d}"
            out_dir = os.path.join(OUT_ROOT, TASK_NAME, subtask_name, SPLIT)
            ensure_dir(out_dir)

            # save image into this subtask folder
            out_img_f = os.path.join(out_dir, f"image_{reindex}.nii.gz")
            sitk.WriteImage(img_norm, out_img_f, True)
            print(f"[OK] Saved {out_img_f}")

            # save labels for this subtask
            for rater_id, lp in sorted(items, key=lambda x: x[0]):
                seg_obj = sitk.ReadImage(lp)
                seg_aligned = align_to_ref(img_ref, seg_obj, is_label=True)

                arr = sitk.GetArrayFromImage(seg_aligned)
                uniq = np.unique(arr)
                if uniq.size == 1 and uniq[0] == 0:
                    print(f"[WARN] Empty label after align: {os.path.basename(lp)}")

                out_lb_f = os.path.join(out_dir, f"label_{reindex}_{rater_id}.nii.gz")
                sitk.WriteImage(seg_aligned, out_lb_f, True)
                print(f"[OK] Saved {out_lb_f}")

        reindex += 1

    print(f"\nDone. Processed {reindex} cases.")


if __name__ == "__main__":
    main()
