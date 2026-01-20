"""
Step 2 preprocessing:
- crop empty boundary in-plane (y/x) by BD_BIAS
- ALSO remove all-background slices on top/bottom along z (based on labels > 0)
- resample to 256x256 in-plane

Input:
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

    for lbv in lbvs:
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
        out_obj.CopyInformation(_tar_curr_obj)

    return out_obj


def find_nonempty_z_bounds_from_labels(label_arrays_zyx, foreground_fn=None):
    """
    label_arrays_zyx: list of np arrays (z,y,x)
    foreground_fn: optional function(arr)->bool mask; default: arr>0

    Returns (z0, z1) inclusive bounds of slices that contain any foreground
    across ALL provided labels. If no foreground found, returns (0, Z-1).
    """
    if foreground_fn is None:
        foreground_fn = lambda a: a > 0

    if not label_arrays_zyx:
        raise ValueError("label_arrays_zyx is empty; cannot determine z-bounds.")

    Z = label_arrays_zyx[0].shape[0]
    any_fg = np.zeros((Z,), dtype=bool)

    for arr in label_arrays_zyx:
        if arr.shape[0] != Z:
            raise ValueError("All label arrays must have the same z dimension.")
        fg = foreground_fn(arr)
        any_fg |= fg.reshape(Z, -1).any(axis=1)

    if not any_fg.any():
        # no foreground at all -> keep full z
        return 0, Z - 1

    z0 = int(np.argmax(any_fg))
    z1 = int(Z - 1 - np.argmax(any_fg[::-1]))
    return z0, z1


def crop_roi_xyz(img: sitk.Image, x0, y0, z0, sx, sy, sz) -> sitk.Image:
    """
    Crop using SITK ROI so origin/direction are handled correctly.
    index/size are in (x,y,z).
    """
    size = [int(sx), int(sy), int(sz)]
    index = [int(x0), int(y0), int(z0)]
    return sitk.RegionOfInterest(img, size=size, index=index)


# -------------------------
# Main
# -------------------------
def main():
    os.makedirs(OUT_FOLDER, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(IN_FOLDER, "image_*.nii.gz")))
    if not img_paths:
        raise RuntimeError(f"No images found in {IN_FOLDER} with pattern image_*.nii.gz")

    for img_path in img_paths:
        base = os.path.basename(img_path)
        k = base.replace("image_", "").replace(".nii.gz", "")

        label_paths = {r: os.path.join(IN_FOLDER, f"label_{k}_{r}.nii.gz") for r in RATERS}
        existing_raters = [r for r, p in label_paths.items() if os.path.exists(p)]
        if not existing_raters:
            print(f"[SKIP] No labels found for image_{k} (expected label_{k}_{{1,2,3}}.nii.gz)")
            continue

        print(f"\nProcessing case k={k} | image={img_path}")

        # Read image and all existing labels first (we need labels to crop z)
        img_obj = sitk.ReadImage(img_path)
        seg_objs = {r: sitk.ReadImage(label_paths[r]) for r in existing_raters}

        # Determine z-crop bounds from union of labels (foreground = label > 0)
        label_arrays = [sitk.GetArrayFromImage(seg_objs[r]) for r in existing_raters]  # (z,y,x)
        z0, z1 = find_nonempty_z_bounds_from_labels(label_arrays)
        print(f"  Z-crop (label foreground): z={z0}..{z1} (inclusive)")

        # Now compute full ROI crop indices/sizes in (x,y,z)
        x_size, y_size, z_size = img_obj.GetSize()  # SITK: (x,y,z)

        # In-plane crop by BD_BIAS + z crop by (z0,z1)
        x0 = BD_BIAS
        y0 = BD_BIAS
        z0_idx = z0

        new_x = x_size - 2 * BD_BIAS
        new_y = y_size - 2 * BD_BIAS
        new_z = (z1 - z0 + 1)

        if new_x <= 0 or new_y <= 0:
            raise ValueError(f"BD_BIAS={BD_BIAS} too large for in-plane size {(x_size, y_size)}")
        if new_z <= 0:
            raise ValueError(f"Computed invalid z crop: z0={z0}, z1={z1}")

        # Crop image and labels using ROI (preserves correct physical geometry)
        cropped_img_obj = crop_roi_xyz(img_obj, x0, y0, z0_idx, new_x, new_y, new_z)

        cropped_seg_objs = {}
        for r in existing_raters:
            # (Assumes label geometry matches the image; common in preprocessed datasets)
            cropped_seg_objs[r] = crop_roi_xyz(seg_objs[r], x0, y0, z0_idx, new_x, new_y, new_z)

        # Compute spacing factor to reach TARGET_HW given cropped size
        # Note: GetSize returns (x,y,z)
        cropped_x, cropped_y, _ = cropped_img_obj.GetSize()
        fac_x = cropped_x / float(TARGET_HW)
        fac_y = cropped_y / float(TARGET_HW)

        sx, sy, sz = img_obj.GetSpacing()
        new_spacing_img = [sx * fac_x, sy * fac_y, sz]

        # Resample image (linear)
        res_img_obj = resample_by_res(
            cropped_img_obj,
            new_spacing_img,
            interpolator=sitk.sitkLinear,
            logging=True
        )

        out_img_path = os.path.join(OUT_FOLDER, f"image_{k}.nii.gz")
        sitk.WriteImage(res_img_obj, out_img_path, True)
        print(f"[OK] Saved {out_img_path}")

        # Process each rater label
        for r in existing_raters:
            seg_obj = cropped_seg_objs[r]

            lsx, lsy, lsz = seg_obj.GetSpacing()
            new_spacing_lb = [lsx * fac_x, lsy * fac_y, lsz]

            res_lb_obj = resample_lb_by_res(
                seg_obj,
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
