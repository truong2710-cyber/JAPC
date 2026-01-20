"""
Step 2 preprocessing (crop empty boundary + resample to 256x256 in-plane)
for a MULTI-RATER dataset already normalized in ./tmp_normalized/.

Input (from your step-1 output):
  ./tmp_normalized/
    image_{k}.nii.gz
    label_{k}_1.nii.gz
    label_{k}_2.nii.gz
    label_{k}_3.nii.gz

Output:
  ./curvas_CT_normalized/
    image_{k}.nii.gz
    label_{k}_1.nii.gz
    label_{k}_2.nii.gz
    label_{k}_3.nii.gz
"""

import os
import glob
import numpy as np
import SimpleITK as sitk

# -------------------------
# Config
# -------------------------
IN_FOLDER = "./tmp_normalized"
OUT_FOLDER = "./curvas_CT_normalized"

RATERS = [1, 2, 3]
BD_BIAS = 32          # crop border in axial plane (y/x dims)
TARGET_HW = 256       # target in-plane size after resampling


# -------------------------
# Helpers
# -------------------------
def copy_spacing_ori(src: sitk.Image, dst: sitk.Image) -> sitk.Image:
    dst.SetSpacing(src.GetSpacing())
    dst.SetOrigin(src.GetOrigin())
    dst.SetDirection(src.GetDirection())
    return dst

def resample_by_res(mov_img_obj: sitk.Image,
                    new_spacing,
                    interpolator=sitk.sitkLinear,
                    logging=True) -> sitk.Image:
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(mov_img_obj.GetDirection())
    resample.SetOutputOrigin(mov_img_obj.GetOrigin())

    mov_spacing = mov_img_obj.GetSpacing()  # (sx, sy, sz)
    mov_size = np.array(mov_img_obj.GetSize(), dtype=np.float64)  # (x, y, z)

    resample.SetOutputSpacing(new_spacing)

    # size scales inversely with spacing
    res_coe = np.array(mov_spacing, dtype=np.float64) / np.array(new_spacing, dtype=np.float64)
    new_size = mov_size * res_coe

    resample.SetSize([int(sz + 1) for sz in new_size])

    if logging:
        print(f"  Spacing: {mov_spacing} -> {tuple(new_spacing)}")
        print(f"  Size   : {tuple(mov_size.astype(int))} -> {tuple(new_size)}")

    return resample.Execute(mov_img_obj)

def resample_lb_by_res(mov_lb_obj: sitk.Image,
                       new_spacing,
                       interpolator=sitk.sitkLinear,
                       ref_img: sitk.Image = None,
                       logging=True) -> sitk.Image:
    """
    Label-safe resampling:
      - for each label value v: resample binary mask (label==v), round, then merge.
    """
    src_mat = sitk.GetArrayFromImage(mov_lb_obj)
    lbvs = np.unique(src_mat)

    if logging:
        print(f"  Label values: {lbvs}")

    out_vol = None
    _tar_curr_obj = None

    for idx, lbv in enumerate(lbvs):
        curr_bin = (src_mat == lbv).astype(np.float32)
        curr_obj = sitk.GetImageFromArray(curr_bin)
        curr_obj.CopyInformation(mov_lb_obj)

        _tar_curr_obj = resample_by_res(curr_obj, new_spacing, interpolator, logging=False)
        tar_bin = np.rint(sitk.GetArrayFromImage(_tar_curr_obj)).astype(np.int32)
        tar_lab = tar_bin * int(lbv)

        if out_vol is None:
            out_vol = tar_lab
        else:
            out_vol[tar_lab == lbv] = lbv

    out_obj = sitk.GetImageFromArray(out_vol.astype(src_mat.dtype))

    # Ensure geometry matches the resampled image
    if ref_img is not None:
        out_obj.CopyInformation(ref_img)
    else:
        # fall back to the last resampled binary object's spacing/origin/direction
        out_obj.SetSpacing(_tar_curr_obj.GetSpacing())
        out_obj.SetOrigin(_tar_curr_obj.GetOrigin())
        out_obj.SetDirection(_tar_curr_obj.GetDirection())

    return out_obj

def crop_axial_border(arr_zyx: np.ndarray, bd: int) -> np.ndarray:
    """
    arr is (z, y, x). Crop y and x borders by bd pixels.
    """
    if bd <= 0:
        return arr_zyx
    if arr_zyx.shape[1] <= 2*bd or arr_zyx.shape[2] <= 2*bd:
        raise ValueError(f"BD_BIAS={bd} too large for in-plane shape {arr_zyx.shape[1:]}")

    return arr_zyx[:, bd:-bd, bd:-bd]


# -------------------------
# Main
# -------------------------
def main():
    os.makedirs(OUT_FOLDER, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(IN_FOLDER, "image_*.nii.gz")))
    if not img_paths:
        raise RuntimeError(f"No images found in {IN_FOLDER} with pattern image_*.nii.gz")

    for img_path in img_paths:
        # parse index k from "image_{k}.nii.gz"
        base = os.path.basename(img_path)
        k = base.replace("image_", "").replace(".nii.gz", "")

        # check labels exist (at least one)
        label_paths = {r: os.path.join(IN_FOLDER, f"label_{k}_{r}.nii.gz") for r in RATERS}
        existing_raters = [r for r, p in label_paths.items() if os.path.exists(p)]
        if not existing_raters:
            print(f"[SKIP] No labels found for image_{k} (expected label_{k}_{{1,2,3}}.nii.gz)")
            continue

        print(f"\nProcessing case k={k} | image={img_path}")

        img_obj = sitk.ReadImage(img_path)

        # --- Crop image in array space (z,y,x), then restore geometry from original
        img_arr = sitk.GetArrayFromImage(img_obj)
        img_arr_crop = crop_axial_border(img_arr, BD_BIAS)

        cropped_img_obj = sitk.GetImageFromArray(img_arr_crop)
        cropped_img_obj = copy_spacing_ori(img_obj, cropped_img_obj)

        # Compute spacing factor to reach TARGET_HW given cropped size
        # We assume square in-plane after crop: y==x in numpy; in sitk size is (x,y,z)
        cropped_x, cropped_y, _ = cropped_img_obj.GetSize()
        if cropped_x != cropped_y:
            # still handle it by using per-dimension factors
            fac_x = cropped_x / float(TARGET_HW)
            fac_y = cropped_y / float(TARGET_HW)
        else:
            fac_x = fac_y = cropped_x / float(TARGET_HW)

        sx, sy, sz = img_obj.GetSpacing()
        new_spacing_img = [sx * fac_x, sy * fac_y, sz]

        # Resample image (linear)
        res_img_obj = resample_by_res(
            cropped_img_obj,
            new_spacing_img,
            interpolator=sitk.sitkLinear,
            logging=True
        )

        # Save resampled image
        out_img_path = os.path.join(OUT_FOLDER, f"image_{k}.nii.gz")
        sitk.WriteImage(res_img_obj, out_img_path, True)
        print(f"[OK] Saved {out_img_path}")

        # --- Process each rater label
        for r in existing_raters:
            seg_obj = sitk.ReadImage(label_paths[r])

            # Crop label in array space (z,y,x)
            lb_arr = sitk.GetArrayFromImage(seg_obj)
            lb_arr_crop = crop_axial_border(lb_arr, BD_BIAS)

            cropped_lb_obj = sitk.GetImageFromArray(lb_arr_crop)
            cropped_lb_obj = copy_spacing_ori(seg_obj, cropped_lb_obj)

            lsx, lsy, lsz = seg_obj.GetSpacing()
            new_spacing_lb = [lsx * fac_x, lsy * fac_y, lsz]

            # Resample label (label-safe), match geometry to resampled image
            res_lb_obj = resample_lb_by_res(
                cropped_lb_obj,
                new_spacing_lb,
                interpolator=sitk.sitkLinear,
                ref_img=res_img_obj,
                logging=True
            )

            out_lb_path = os.path.join(OUT_FOLDER, f"label_{k}_{r}.nii.gz")
            sitk.WriteImage(res_lb_obj, out_lb_path, True)
            print(f"[OK] Saved {out_lb_path}")

    print(f"\nDone. Outputs in: {OUT_FOLDER}")


if __name__ == "__main__":
    main()
