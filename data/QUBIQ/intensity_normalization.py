"""
QUBIQ intensity normalization (per task) + geometry-safe label alignment,
saved per *subtask*.

Output:
OUT_ROOT/{task_name}/{taskXX}/{split}/image_{k}.nii.gz
OUT_ROOT/{task_name}/{taskXX}/{split}/label_{k}_segYY.nii.gz
"""

import os
import glob
import re
import numpy as np
import SimpleITK as sitk


# -------------------------
# Config (edit these)
# -------------------------
TASK_NAME = "all"           # run one task at a time
OUT_ROOT = "./tmp_normalized_qubiq" # output root

# # Normalization strategy:
# IS_CT = None

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

    # If reference is 3D but moving is 2D, promote moving to 3D by repeating the single slice
    # so that Resample can operate without dimension mismatch.
    if ref.GetDimension() == 3 and moving.GetDimension() == 2:
        ref_z = ref.GetSize()[2]
        arr2 = sitk.GetArrayFromImage(moving)  # [y, x]
        arr3 = np.repeat(arr2[np.newaxis, ...], ref_z, axis=0)  # [z, y, x]
        moving3 = sitk.GetImageFromArray(arr3)
        moving3.CopyInformation(ref)
        moving = moving3

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
def main(task_name: str = None, split: str = None):
    """Process one task/split. If `task_name` or `split` is None the module
    defaults (`TASK_NAME`, `SPLIT`) are used.
    """
    # determine task and split to use
    task = task_name if task_name is not None else TASK_NAME
    split_name = split if split is not None else SPLIT

    # update IS_CT based on task (keep as global for intensity_normalize)
    global IS_CT
    if task in ["prostate", "brain-tumor", "brain-growth"]:
        IS_CT = False
    else:
        IS_CT = True

    in_split_root = os.path.join(IN_ROOT, task, split_name)
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
        # Keep original image dimensionality (do not force 2D).
        # Labels that are 2D or single-slice will be expanded below to match image depth.
        img_ref = img_obj

        img_norm = intensity_normalize(img_ref)

        # save per subtask folder
        for task_id, items in sorted(labels_by_task.items()):
            subtask_name = f"task{task_id:02d}"
            out_dir = os.path.join(OUT_ROOT, task, subtask_name, split_name)
            ensure_dir(out_dir)

            # save image into this subtask folder
            out_img_f = os.path.join(out_dir, f"image_{reindex}.nii.gz")
            sitk.WriteImage(img_norm, out_img_f, True)
            print(f"[OK] Saved {out_img_f}")

            # save labels for this subtask
            for rater_id, lp in sorted(items, key=lambda x: x[0]):
                seg_obj = sitk.ReadImage(lp)
                seg_aligned = align_to_ref(img_ref, seg_obj, is_label=True)

                # If the image is 3D but the aligned segmentation is 2D (or has only one slice),
                # duplicate the label along Z to match the image depth. This preserves geometric
                # information and avoids dropping labels.
                try:
                    ref_dim = img_ref.GetDimension()
                    seg_dim = seg_aligned.GetDimension()
                    if ref_dim == 3:
                        ref_z = img_ref.GetSize()[2]
                        if seg_dim == 2:
                            arr2 = sitk.GetArrayFromImage(seg_aligned)  # [y, x]
                            arr3 = np.repeat(arr2[np.newaxis, ...], ref_z, axis=0)  # [z, y, x]
                            seg_aligned = sitk.GetImageFromArray(arr3)
                            seg_aligned.CopyInformation(img_ref)
                            print(f"[INFO] Duplicated 2D label {os.path.basename(lp)} to {ref_z} slices")
                        else:
                            # seg_dim == 3
                            seg_z = seg_aligned.GetSize()[2]
                            if seg_z == 1 and ref_z > 1:
                                arr3 = sitk.GetArrayFromImage(seg_aligned)  # [1, y, x]
                                arr3_rep = np.repeat(arr3, ref_z, axis=0)
                                seg_aligned = sitk.GetImageFromArray(arr3_rep)
                                seg_aligned.CopyInformation(img_ref)
                                print(f"[INFO] Replicated single-slice label {os.path.basename(lp)} to {ref_z} slices")
                except Exception as e:
                    print(f"[WARN] Failed to duplicate/align label {lp}: {e}")

                arr = sitk.GetArrayFromImage(seg_aligned)
                uniq = np.unique(arr)
                if uniq.size == 1 and uniq[0] == 0:
                    print(f"[WARN] Empty label after align: {os.path.basename(lp)}")

                out_lb_f = os.path.join(out_dir, f"label_{reindex}_{rater_id}.nii.gz")
                sitk.WriteImage(seg_aligned, out_lb_f, True)
                print(f"[OK] Saved {out_lb_f}")

        reindex += 1

    print(f"\nDone. Processed {reindex} cases for task={task} split={split_name}.")


def process_all_tasks(IN_ROOT, SPLIT):
    """Discover all task folders under `IN_ROOT` and run `main` for each split
    folder found under the task (e.g., Training/Validation).
    """
    task_folders = sorted([p for p in glob.glob(os.path.join(IN_ROOT, "*")) if os.path.isdir(p)])
    if not task_folders:
        raise RuntimeError(f"No task folders found under IN_ROOT={IN_ROOT}")

    for tf in task_folders:
        task = os.path.basename(tf)
        # find splits inside task folder (directories)
        splits = sorted([os.path.basename(p) for p in glob.glob(os.path.join(tf, "*")) if os.path.isdir(p)])
        if not splits:
            # if no splits, try calling main() with default SPLIT
            try:
                main(task_name=task, split=SPLIT)
            except Exception as e:
                print(f"[WARN] Failed processing {task} with default split {SPLIT}: {e}")
            continue

        for sp in splits:
            try:
                main(task_name=task, split=sp)
            except Exception as e:
                print(f"[WARN] Failed processing {task}/{sp}: {e}")


if __name__ == "__main__":
    # If TASK_NAME is set to 'all' run over all tasks; otherwise run single task.
    if TASK_NAME == "all":
        for IN_ROOT in ['training_data_v3_QC', 'validation_data_qubiq2021_QC']:
            if 'training' in IN_ROOT.lower():
                SPLIT = "Training"
            elif 'validation' in IN_ROOT.lower():
                SPLIT = "Validation"
            process_all_tasks(IN_ROOT, SPLIT)
    else:
        raise NotImplementedError("Currently only TASK_NAME='all' is supported. To run a single task, set TASK_NAME to the desired task and adjust the main() function to call with that task and split directly.")
