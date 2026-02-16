import os
import glob
import json
import numpy as np
import SimpleITK as sitk

# -------------------------
# Config
# -------------------------
DATA_DIR = "qubiq_normalized/brain-tumor/task01/Training/"
OUT_DIR = DATA_DIR

NUM_RATERS = 3  # number of raters per case 
RATERS = list(range(1, NUM_RATERS + 1))  # e.g. [1,2,3] for raters 1 to 3
MIN_TP = 1  # minimum pixels of class in a slice to count

# IMPORTANT: set your dataset's label names in index order.
# Example for your dataset (edit this to match your labels):
# 0=background, 1=pancreas, 2=kidney, 3=liver
LABEL_NAME = ["BGD", "BRAIN_TUMOR_1"]

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
def main():
    ks = list_case_indices(DATA_DIR)
    if not ks:
        raise RuntimeError(f"No images found in {DATA_DIR} with pattern image_*.nii.gz")

    # Initialize maps
    classmap_union = {name: {str(k): [] for k in ks} for name in LABEL_NAME}
    classmap_per_rater = {
        r: {name: {str(k): [] for k in ks} for name in LABEL_NAME}
        for r in RATERS
    }

    for k in ks:
        # load all existing rater labels for this case
        lb_vols = []
        for r in RATERS:
            lb_path = os.path.join(DATA_DIR, f"label_{k}_{r}.nii.gz")
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
            # union mask over raters (for union classmap)
            # (we check class-by-class, so no need to literally union volumes)
            for cls, cls_name in enumerate(LABEL_NAME):
                # per-rater
                present_any = False
                for r, vol in lb_vols:
                    if np.any(vol[slc] == cls) and np.sum(vol[slc] == cls) >= MIN_TP:
                        classmap_per_rater[r][cls_name][str(k)].append(slc)
                        present_any = True

                # union over raters
                if present_any:
                    classmap_union[cls_name][str(k)].append(slc)

        print(f"[OK] case {k} finished ({n_slice} slices)")

    # Save outputs
    out_union = os.path.join(OUT_DIR, f"classmap_{MIN_TP}.json")
    with open(out_union, "w") as f:
        json.dump(classmap_union, f)

    for r in RATERS:
        out_r = os.path.join(OUT_DIR, f"classmap_rater{r}_{MIN_TP}.json")
        with open(out_r, "w") as f:
            json.dump(classmap_per_rater[r], f)

    print("\nSaved:")
    print(" -", out_union)
    for r in RATERS:
        print(" -", os.path.join(OUT_DIR, f"classmap_rater{r}_minTP{MIN_TP}.json"))

if __name__ == "__main__":
    main()
