"""
Step 2 preprocessing:
- crop empty boundary in-plane (y/x) by BD_BIAS
- remove all-background slices on top/bottom along z (based on labels > 0)
- resample to TARGET_HW x TARGET_HW in-plane
- ALSO resample z-spacing to TARGET_Z_SPACING (e.g., 3.0mm)

Input:
  ./tmp_normalized/
    image_{k}.nii.gz
    label_{k}_1.nii.gz
    label_{k}_2.nii.gz
    label_{k}_3.nii.gz

Output:
  ./qubiq_normalized/
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
TASK_NAME = "all"  # for naming output folder; no effect on processing
IN_FOLDER = "tmp_normalized_qubiq"
OUT_FOLDER = "qubiq_normalized"

BD_BIAS = 0          # crop border in axial plane (x/y dims)
TARGET_HW = 256       # target in-plane size after resampling
TARGET_Z_SPACING = None  # <-- NEW: desired z spacing (mm)

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

    # compute integer output size by rounding the float size (avoid forcing +1)
    resample.SetSize([max(1, int(np.round(sz))) for sz in new_size])

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
        return 0, Z - 1

    z0 = int(np.argmax(any_fg))
    z1 = int(Z - 1 - np.argmax(any_fg[::-1]))
    return z0, z1


def crop_roi_xyz(img: sitk.Image, x0, y0, z0, sx, sy, sz) -> sitk.Image:
    """
    Crop using SITK ROI so origin/direction are handled correctly.
    index/size are in (x,y,z).
    """
    # Handle 2D and 3D images: build size/index matching image dimension
    dim = img.GetDimension()
    if dim == 3:
        size = [int(sx), int(sy), int(sz)]
        index = [int(x0), int(y0), int(z0)]
    else:
        # 2D image: ignore z params
        size = [int(sx), int(sy)]
        index = [int(x0), int(y0)]
    return sitk.RegionOfInterest(img, size=size, index=index)


def ensure_image_is_3d(img: sitk.Image, z_spacing: float = None) -> sitk.Image:
    """Convert a 2D SimpleITK image to a 3D image with a single z-slice.

    Preserves x/y spacing, origin, and direction (extended to 3D). If the image
    is already 3D it is returned unchanged. The resulting z spacing is set to
    `z_spacing` if provided, otherwise 1.0.
    """
    if img.GetDimension() == 3:
        return img

    # Join single 2D image into a 3D volume with one slice
    img2d = img
    img3 = sitk.JoinSeries([img2d])

    # Set spacing: (sx, sy, z_spacing)
    sx, sy = img2d.GetSpacing()
    zsp = float(z_spacing) if z_spacing is not None else 1.0
    img3.SetSpacing((sx, sy, zsp))

    # Set origin: extend with 0 for z
    orig = img2d.GetOrigin()
    if len(orig) == 2:
        img3.SetOrigin((orig[0], orig[1], 0.0))
    else:
        img3.SetOrigin(orig + (0.0,))

    # Extend 2D direction to 3D: place 2D direction in top-left, leave z basis as identity
    dir2 = img2d.GetDirection()
    if len(dir2) == 4:
        dir3 = (dir2[0], dir2[1], 0.0,
                dir2[2], dir2[3], 0.0,
                0.0,       0.0,      1.0)
        img3.SetDirection(dir3)

    return img3

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

        # Read image and labels (need labels for z-crop)
        img_obj = sitk.ReadImage(img_path)
        seg_objs = {r: sitk.ReadImage(label_paths[r]) for r in existing_raters}

        # Ensure images/labels are 3D volumes (z=1 for originally 2D inputs)
        # Use the original z spacing when available, otherwise default to 1.0
        if img_obj.GetDimension() == 2:
            img_obj = ensure_image_is_3d(img_obj, z_spacing=(TARGET_Z_SPACING if TARGET_Z_SPACING is not None else 1.0))

        for r in list(seg_objs.keys()):
            if seg_objs[r].GetDimension() == 2:
                # set label z spacing to match image z spacing
                seg_objs[r] = ensure_image_is_3d(seg_objs[r], z_spacing=img_obj.GetSpacing()[2])

        # Determine z-crop bounds from union of labels (foreground = label > 0)
        # Ensure labels are represented as (z,y,x) arrays (expand 2D to z=1)
        label_arrays = []
        for r in existing_raters:
            arr = sitk.GetArrayFromImage(seg_objs[r])
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            label_arrays.append(arr)
        z0, z1 = find_nonempty_z_bounds_from_labels(label_arrays)
        print(f"  Z-crop (label foreground): z={z0}..{z1} (inclusive)")

        # Compute full ROI crop indices/sizes in (x,y,z)
        img_size = img_obj.GetSize()
        # SITK GetSize() returns (x,y,z) for 3D and (x,y) for 2D. Normalize to 3-tuple.
        if len(img_size) == 3:
            x_size, y_size, z_size = img_size
            img_dim = 3
        else:
            x_size, y_size = img_size
            z_size = 1
            img_dim = 2

        if BD_BIAS > 0:
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

            # Crop image and labels using ROI
            cropped_img_obj = crop_roi_xyz(img_obj, x0, y0, z0_idx, new_x, new_y, new_z)
            cropped_seg_objs = {r: crop_roi_xyz(seg_objs[r], x0, y0, z0_idx, new_x, new_y, new_z)
                                for r in existing_raters}
        else:
            cropped_img_obj = img_obj
            cropped_seg_objs = seg_objs

        # Compute in-plane spacing factor to reach TARGET_HW
        csize = cropped_img_obj.GetSize()
        if len(csize) == 3:
            if 'pancrea' in img_path:
                cropped_z, cropped_x, cropped_y = csize
            else:
                cropped_x, cropped_y, cropped_z = csize
        else:
            cropped_x, cropped_y = csize
            cropped_z = 1
        fac_x = cropped_x / float(TARGET_HW)
        fac_y = cropped_y / float(TARGET_HW)

        # ORIGINAL spacing of the CROPPED image (same as img_obj in practice)
        spacing = cropped_img_obj.GetSpacing()
        if len(spacing) == 3:
            if 'pancrea' in img_path:
                sz, sy, sx = spacing
            else:
                sx, sy, sz = spacing
        else:
            sx, sy = spacing
            sz = 1.0

        # Determine target spacing. For 2D images only produce x/y spacing. For 3D,
        # either resample z to TARGET_Z_SPACING or keep original z when ONLY_RESAMPLE_IN_PLANE.
        if len(spacing) == 2:
            # 2D image: only x/y spacing
            new_spacing_img = [sx * fac_x, sy * fac_y]
        else:
            if TARGET_Z_SPACING is None:
                if 'pancrea' in img_path:
                    new_spacing_img = [sz, sy * fac_y, sx * fac_x]
                else:
                    new_spacing_img = [sx * fac_x, sy * fac_y, sz]
            else:
                if 'pancrea' in img_path:
                    new_spacing_img = [float(TARGET_Z_SPACING), sy * fac_y, sx * fac_x]
                else:
                    new_spacing_img = [sx * fac_x, sy * fac_y, float(TARGET_Z_SPACING)]

        # Resample image (linear)
        res_img_obj = resample_by_res(
            cropped_img_obj,
            new_spacing_img,
            interpolator=sitk.sitkLinear,
            logging=True
        )
        out_img_path = os.path.join(OUT_FOLDER, f"image_{k}.nii.gz")
        if 'pancrea' in img_path:
            res_img_obj_permute = sitk.PermuteAxes(res_img_obj, order=[2, 1, 0])
            sitk.WriteImage(res_img_obj_permute, out_img_path, True)
        else:
            sitk.WriteImage(res_img_obj, out_img_path, True)
        print(f"[OK] Saved {out_img_path}")

        # Resample each rater label to the SAME spacing/geometry
        for r in existing_raters:
            seg_obj = cropped_seg_objs[r]

            # For labels, use the same target spacing as image (including z)
            res_lb_obj = resample_lb_by_res(
                seg_obj,
                new_spacing_img,  # <-- same [sz, sx*fac_x, sy*fac_y]
                interpolator=sitk.sitkLinear,
                ref_img=res_img_obj,  # ensures identical size/origin/direction
                logging=True
            )
            out_lb_path = os.path.join(OUT_FOLDER, f"label_{k}_{r}.nii.gz")
            if 'pancrea' in img_path:
                res_lb_obj_permute = sitk.PermuteAxes(res_lb_obj, order=[2, 1, 0])
                sitk.WriteImage(res_lb_obj_permute, out_lb_path, True)
            else:
                sitk.WriteImage(res_lb_obj, out_lb_path, True)
            print(f"[OK] Saved {out_lb_path}")
            
        del img_obj, cropped_img_obj, res_img_obj
        del seg_objs, cropped_seg_objs, res_lb_obj
        import gc; gc.collect()

    print(f"\nDone. Outputs in: {OUT_FOLDER}")


if __name__ == "__main__":
    # If TASK_NAME == 'all', iterate all task folders under the IN_FOLDER root and all splits
    if TASK_NAME == "all":
        # derive base input root from the configured IN_FOLDER (assumes IN_FOLDER = <root>/...)
        base_root = IN_FOLDER
        out_base = OUT_FOLDER

        task_dirs = sorted([d for d in glob.glob(os.path.join(base_root, "*")) if os.path.isdir(d)])
        if not task_dirs:
            raise RuntimeError(f"No task folders found under: {base_root}")

        for td in task_dirs:
            # handle hierarchy: task/subtask/split
            subtask_dirs = sorted([d for d in glob.glob(os.path.join(td, "*")) if os.path.isdir(d)])
            if not subtask_dirs:
                # no subtask level, treat td as holding split folders
                subtask_dirs = [td]

            for st in subtask_dirs:
                # find split folders under subtask (e.g., Training, Validation)
                split_dirs = sorted([d for d in glob.glob(os.path.join(st, "*")) if os.path.isdir(d)])
                if not split_dirs:
                    # no split level, treat st as the split folder
                    split_dirs = [st]

                for sd in split_dirs:
                    in_folder = sd
                    out_folder = os.path.join(out_base, os.path.relpath(sd, base_root))

                    print(f"\nProcessing folder: {in_folder} -> {out_folder}")

                    # auto-detect raters by scanning label files recursively under the split folder
                    label_files = glob.glob(os.path.join(in_folder, "**", "label_*_*.nii*"), recursive=True)
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
                        print(f"[WARN] No label files found in {in_folder}; skipping")
                        continue

                    rlist = sorted(rater_ids)

                    # set globals for main()
                    IN_FOLDER = in_folder
                    OUT_FOLDER = out_folder
                    NUM_RATERS = max(rlist)
                    RATERS = rlist

                    os.makedirs(OUT_FOLDER, exist_ok=True)
                    try:
                        main()
                    except Exception as e:
                        print(f"[ERROR] Processing {in_folder} failed: {e}")
    else:
        main()
