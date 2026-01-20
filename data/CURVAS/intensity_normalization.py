"""
Window + normalize CT volumes and reindex filenames for a multi-rater dataset.

Input structure:
./training_set/{id}/image.nii.gz
./training_set/{id}/annotation_1.nii.gz
./training_set/{id}/annotation_2.nii.gz
./training_set/{id}/annotation_3.nii.gz

Output structure (created if missing):
./tmp_normalized/image_{k}.nii.gz
./tmp_normalized/label_{k}_1.nii.gz
./tmp_normalized/label_{k}_2.nii.gz
./tmp_normalized/label_{k}_3.nii.gz
"""

import os
import glob
import SimpleITK as sitk
import numpy as np


# -------------------------
# Config
# -------------------------
IN_ROOT = "./training_set"
OUT_ROOT = "./tmp_normalized"

# Abdominal window (HU)
LIR = -125
HIR = 275

# If True: normalize AFTER clipping using fixed [LIR, HIR] -> [0,255]
# If False: normalize using per-volume min/max after clipping (still usually LIR/HIR)
USE_FIXED_WINDOW_FOR_NORM = True

# Expect these rater files
RATERS = [1, 2, 3]


# -------------------------
# Helpers
# -------------------------
def copy_spacing_ori(src: sitk.Image, dst: sitk.Image) -> sitk.Image:
    dst.SetSpacing(src.GetSpacing())
    dst.SetOrigin(src.GetOrigin())
    dst.SetDirection(src.GetDirection())
    return dst

def window_and_normalize_hu(img_obj: sitk.Image) -> sitk.Image:
    arr = sitk.GetArrayFromImage(img_obj).astype(np.float32)

    # Clip to abdominal window
    arr = np.clip(arr, LIR, HIR)

    # Normalize to [0, 255]
    if USE_FIXED_WINDOW_FOR_NORM:
        arr = (arr - LIR) / (HIR - LIR + 1e-8) * 255.0
    else:
        mn, mx = float(arr.min()), float(arr.max())
        arr = (arr - mn) / (mx - mn + 1e-8) * 255.0

    # Keep as float32 unless you explicitly want uint8
    # arr = arr.astype(np.uint8)  # optionally
    out = sitk.GetImageFromArray(arr)
    out = copy_spacing_ori(img_obj, out)
    return out


# -------------------------
# Main
# -------------------------
def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    # Find cases (directories under IN_ROOT)
    case_dirs = sorted([p for p in glob.glob(os.path.join(IN_ROOT, "*")) if os.path.isdir(p)])

    if not case_dirs:
        raise RuntimeError(f"No case folders found under: {IN_ROOT}")

    reindex = 0
    for case_dir in case_dirs:
        img_path = os.path.join(case_dir, "image.nii.gz")
        if not os.path.exists(img_path):
            print(f"[SKIP] Missing image: {img_path}")
            continue

        # Check rater label paths (allow missing with warning)
        label_paths = {r: os.path.join(case_dir, f"annotation_{r}.nii.gz") for r in RATERS}
        missing = [r for r, p in label_paths.items() if not os.path.exists(p)]
        if missing:
            print(f"[WARN] Case {os.path.basename(case_dir)} missing raters {missing}. "
                  f"Will still process existing ones.")

        # Load image, window+normalize
        img_obj = sitk.ReadImage(img_path)
        out_img_obj = window_and_normalize_hu(img_obj)

        out_img_fid = os.path.join(OUT_ROOT, f"image_{reindex}.nii.gz")
        sitk.WriteImage(out_img_obj, out_img_fid, True)
        print(f"[OK] Saved {out_img_fid}")

        # Copy each label unchanged (but you can also enforce geometry match if needed)
        for r in RATERS:
            lp = label_paths[r]
            if not os.path.exists(lp):
                continue

            seg_obj = sitk.ReadImage(lp)

            # Optional sanity: ensure label has same spatial meta as image
            # If your labels sometimes differ, you can copy spacing/origin/direction:
            seg_obj = copy_spacing_ori(img_obj, seg_obj)

            out_lb_fid = os.path.join(OUT_ROOT, f"label_{reindex}_{r}.nii.gz")
            sitk.WriteImage(seg_obj, out_lb_fid, True)
            print(f"[OK] Saved {out_lb_fid}")

        reindex += 1

    print(f"\nDone. Processed {reindex} cases into: {OUT_ROOT}")


if __name__ == "__main__":
    main()
