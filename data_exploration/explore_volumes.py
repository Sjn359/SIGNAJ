"""
explore_volumes.py
===================
Loads NIfTI volumes with nibabel, reports basic metadata, applies
modality-appropriate basic preprocessing, and saves slice visualizations.

Run against the SYNTHETIC placeholder files by default (see
make_synthetic_volumes.py). To run against real data once you have it
locally, just change the paths in `if __name__ == "__main__"` below --
everything else is written against real nibabel/NIfTI conventions and
should not need to change.

What "basic preprocessing" means here, and what it deliberately does NOT
attempt (be upfront about this in the meeting):
  - Intensity normalization: implemented (real, simple).
  - Resampling to isotropic spacing: implemented (real, simple linear
    interpolation via scipy.ndimage.zoom -- NOT what a production
    pipeline would use, see note below).
  - Skull stripping: NOT implemented. Real skull stripping needs a
    trained tool (HD-BET, SynthStrip, BET) or, for the synthetic MRI
    volume, we just use the fact that we generated it pre-masked. Faking
    this with a threshold would be misleading to show as "preprocessing."
  - Registration to a common template: NOT implemented. ATLAS ships
    pre-registered to MNI152; ISLES'24/AISD do not, and registering CT
    to MRI space (or vice versa) is a nontrivial step (rigid + often
    affine registration, e.g. ANTs or FSL FLIRT) that needs real data
    and real QA, not a skeleton demo.
"""

import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_and_report(path, modality_label):
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    zooms = img.header.get_zooms()[:3]

    print(f"\n--- {modality_label}: {os.path.basename(path)} ---")
    print(f"  shape (D,H,W):        {data.shape}")
    print(f"  voxel spacing (mm):   {tuple(round(z, 3) for z in zooms)}")
    print(f"  dtype:                {data.dtype}")
    print(f"  intensity min/max:    {data.min():.2f} / {data.max():.2f}")
    print(f"  intensity mean/std:   {data.mean():.2f} / {data.std():.2f}")
    print(f"  affine:\n{img.affine}")
    return img, data, zooms


def preprocess_mri(data):
    """Percentile clip + z-score, restricted to nonzero (brain) voxels."""
    brain_voxels = data[data > 0]
    lo, hi = np.percentile(brain_voxels, [1, 99])
    clipped = np.clip(data, lo, hi)
    mean, std = clipped[data > 0].mean(), clipped[data > 0].std()
    normalized = np.where(data > 0, (clipped - mean) / (std + 1e-8), 0.0)
    return normalized


def preprocess_ct(data, window_center=32, window_width=8):
    """
    Narrow "stroke window" HU clipping (roughly matching clinically-used
    narrow windows for subtle ischemia, e.g. center~32 HU / width~8 HU is
    in the range of what's reported in the stroke-CT literature -- this
    is a plausible starting point, not a validated clinical setting),
    then min-max scale to [0, 1].
    """
    lo, hi = window_center - window_width / 2, window_center + window_width / 2
    clipped = np.clip(data, lo, hi)
    normalized = (clipped - lo) / (hi - lo + 1e-8)
    return normalized


def resample_isotropic(data, zooms, target_spacing=1.0):
    """Naive linear-interpolation resampling to isotropic spacing.
    A production pipeline would use a proper medical-imaging resampler
    (e.g. SimpleITK / ANTs) with correct handling of the affine, not
    scipy.ndimage.zoom -- this is here to prove the shape math, nothing
    more."""
    factors = [z / target_spacing for z in zooms]
    resampled = zoom(data, factors, order=1)
    return resampled


def visualize_mid_slices(data, title, save_path, lesion_mask=None, zooms=(1.0, 1.0, 1.0)):
    """
    zooms=(dz, dy, dx) in mm. Anisotropic CT (e.g. 5mm slices, 0.45mm
    in-plane) will look wrong -- squashed brain -- if displayed with a
    naive 1:1 pixel aspect. We set the matplotlib aspect ratio from the
    actual voxel spacing so distances look physically correct even
    without resampling first.
    """
    D, H, W = data.shape
    dz, dy, dx = zooms
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    slices = [data[D // 2, :, :], data[:, H // 2, :], data[:, :, W // 2]]
    labels = ["axial (mid)", "coronal (mid)", "sagittal (mid)"]
    # matplotlib's imshow `aspect` = (spacing of the row/vertical axis) /
    # (spacing of the column/horizontal axis), for each of the three views
    aspects = [dx / dy, dx / dz, dy / dz]

    for ax, sl, lab, asp in zip(axes, slices, labels, aspects):
        ax.imshow(sl.T, cmap="gray", origin="lower", aspect=asp)
        ax.set_title(lab, fontsize=10)
        ax.axis("off")

    if lesion_mask is not None:
        mask_slices = [
            lesion_mask[D // 2, :, :],
            lesion_mask[:, H // 2, :],
            lesion_mask[:, :, W // 2],
        ]
        for ax, msl, asp in zip(axes, mask_slices, aspects):
            if msl.max() > 0:
                ax.contour(msl.T, colors="red", linewidths=1, origin="lower")
                ax.set_aspect(asp)

    fig.suptitle(title + "\n(SYNTHETIC placeholder data)", fontsize=11, color="darkred")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"  saved figure -> {save_path}")


def run(img_path, mask_path, modality_label, preprocess_fn, **preprocess_kwargs):
    img, data, zooms = load_and_report(img_path, modality_label)
    mask_img = nib.load(mask_path)
    mask = mask_img.get_fdata().astype(np.uint8)

    raw_fig = os.path.join(FIG_DIR, f"{modality_label}_raw.png")
    visualize_mid_slices(data, f"{modality_label} - raw", raw_fig, lesion_mask=mask, zooms=zooms)

    preprocessed = preprocess_fn(data, **preprocess_kwargs)
    prep_fig = os.path.join(FIG_DIR, f"{modality_label}_preprocessed.png")
    visualize_mid_slices(
        preprocessed, f"{modality_label} - preprocessed", prep_fig, lesion_mask=mask, zooms=zooms
    )

    resampled = resample_isotropic(preprocessed, zooms, target_spacing=1.0)
    print(f"  shape after preprocessing:        {preprocessed.shape}")
    print(f"  shape after isotropic resampling:  {resampled.shape}  (spacing -> 1.0mm iso)")

    return {
        "raw": data,
        "preprocessed": preprocessed,
        "resampled": resampled,
        "mask": mask,
        "zooms": zooms,
    }


if __name__ == "__main__":
    synth_dir = os.path.join(HERE, "synthetic_data")

    mri_result = run(
        img_path=os.path.join(synth_dir, "sub-fake01_ses-1_space-MNI152_T1w.nii.gz"),
        mask_path=os.path.join(
            synth_dir, "sub-fake01_ses-1_space-MNI152_label-L_desc-T1lesion_mask.nii.gz"
        ),
        modality_label="ATLAS-style_T1w_MRI",
        preprocess_fn=preprocess_mri,
    )

    ct_result = run(
        img_path=os.path.join(synth_dir, "sub-fake02_ncct.nii.gz"),
        mask_path=os.path.join(synth_dir, "sub-fake02_ncct_lesion_mask.nii.gz"),
        modality_label="AISD-ISLES-style_NCCT",
        preprocess_fn=preprocess_ct,
        window_center=32,
        window_width=8,
    )

    print("\nDone. Figures in:", FIG_DIR)
