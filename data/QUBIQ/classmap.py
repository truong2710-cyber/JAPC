import os
import glob
import json
import numpy as np
import SimpleITK as sitk

# -------------------------
# Config
# -------------------------
TASK_NAME = "all"
# Root that contains task folders (task/subtask/split/...)
IN_ROOT = "./qubiq_normalized"

MIN_TP = 1  # minimum pixels of class in a slice to count

# -------------------------
# Helpers
# -------------------------
def read_nii(path: str) -> np.ndarray:
    """Return array in (Z,Y,X) from NIfTI using SimpleITK."""
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def list_case_indices(data_dir: str):
    """Find all k from image_{k}.nii.gz"""
    img_paths = sorted(glob.glob(os.path.join(data_dir, "image_*.nii.gz")))
    ks = []
    for p in img_paths:
        base = os.path.basename(p)
        k = base.replace("image_", "").replace(".nii.gz", "")
        ks.append(k)
    return ks

# -------------------------
# Main
# -------------------------
def run_classmap(data_dir: str, out_dir: str, raters: list, label_names: list, min_tp: int = MIN_TP):
    ks = list_case_indices(data_dir)
    if not ks:
        print(f"[WARN] No images found in {data_dir} with pattern image_*.nii.gz")
        return

    # Initialize maps
    classmap_union = {name: {str(k): [] for k in ks} for name in label_names}
    classmap_per_rater = {
        r: {name: {str(k): [] for k in ks} for name in label_names}
        for r in raters
    }

    for k in ks:
        # load all existing rater labels for this case
        lb_vols = []
        for r in raters:
            # search for either .nii or .nii.gz
            lb_path = os.path.join(data_dir, f"label_{k}_{r}.nii.gz")
            if not os.path.exists(lb_path):
                lb_path = os.path.join(data_dir, f"label_{k}_{r}.nii")
            if os.path.exists(lb_path):
                lb_vols.append((r, read_nii(lb_path)))
            else:
                print(f"[WARN] missing {lb_path}")

        if not lb_vols:
            print(f"[SKIP] No labels for case {k}")
            continue

        # assume all raters have same shape
        n_slice = lb_vols[0][1].shape[0]

        for slc in range(n_slice):
            for cls, cls_name in enumerate(label_names):
                present_any = False
                for r, vol in lb_vols:
                    if np.any(vol[slc] == cls) and np.sum(vol[slc] == cls) >= min_tp:
                        classmap_per_rater[r][cls_name][str(k)].append(slc)
                        present_any = True
                if present_any:
                    classmap_union[cls_name][str(k)].append(slc)

        print(f"[OK] case {k} finished ({n_slice} slices)")

    # Save outputs
    os.makedirs(out_dir, exist_ok=True)
    out_union = os.path.join(out_dir, f"classmap_{min_tp}.json")
    with open(out_union, "w") as f:
        json.dump(classmap_union, f)

    for r in raters:
        out_r = os.path.join(out_dir, f"classmap_rater{r}_{min_tp}.json")
        with open(out_r, "w") as f:
            json.dump(classmap_per_rater[r], f)

    print("\nSaved:")
    print(" -", out_union)
    for r in raters:
        print(" -", os.path.join(out_dir, f"classmap_rater{r}_minTP{min_tp}.json"))

if __name__ == "__main__":
    # If TASK_NAME == 'all' iterate under IN_ROOT, handling task/subtask/split
    if TASK_NAME == "all":
        base_root = os.path.normpath(IN_ROOT)
        task_dirs = sorted([d for d in glob.glob(os.path.join(base_root, "*")) if os.path.isdir(d)])
        if not task_dirs:
            raise RuntimeError(f"No task folders found under: {base_root}")

        for td in task_dirs:
            task_name = os.path.basename(td)
            # subtask level (taskXX)
            subtask_dirs = sorted([d for d in glob.glob(os.path.join(td, "*")) if os.path.isdir(d)])
            if not subtask_dirs:
                subtask_dirs = [td]

            for st in subtask_dirs:
                subtask_name = os.path.basename(st)
                # split level (Training/Validation) under subtask
                split_dirs = sorted([d for d in glob.glob(os.path.join(st, "*")) if os.path.isdir(d)])
                if not split_dirs:
                    split_dirs = [st]

                for sd in split_dirs:
                    data_dir = sd
                    out_dir = sd  # keep outputs next to data

                    # auto-detect raters by scanning label files recursively
                    label_files = glob.glob(os.path.join(data_dir, "**", "label_*_*.nii*"), recursive=True)
                    rater_ids = set()
                    import re as _re
                    for lf in label_files:
                        m = _re.search(r"label_\d+_(\d+)\.nii(\.gz)?$", os.path.basename(lf))
                        if m:
                            try:
                                rater_ids.add(int(m.group(1)))
                            except ValueError:
                                continue

                    if not rater_ids:
                        print(f"[WARN] No label files found in {data_dir}; skipping")
                        continue

                    raters = sorted(rater_ids)

                    # construct label name: background + TASK_SUBTASKID
                    # TASK -> uppercase with non-alnum -> underscore
                    task_label = task_name.upper().replace('-', '_')
                    # extract digits from subtask (e.g., task01 -> 1)
                    sub_id = ""
                    m = _re.search(r"(\d+)", subtask_name)
                    if m:
                        sub_id = str(int(m.group(1)))
                    else:
                        sub_id = subtask_name
                    label_names = ["BGD", f"{task_label}_{sub_id}"]

                    print(f"\nProcessing {data_dir}\n - raters: {raters}\n - labels: {label_names}")
                    run_classmap(data_dir, out_dir, raters, label_names)
    else:
        raise NotImplementedError("Currently only TASK_NAME='all' is supported. To run a single task, set TASK_NAME to the desired task and adjust the main() function to call with that task and split directly.")