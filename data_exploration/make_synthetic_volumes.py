"""
make_synthetic_volumes.py
==========================
IMPORTANT: This generates FAKE placeholder volumes. It does NOT download
or contain any real patient data from ATLAS, AISD, or ISLES'24.

Why this exists: this sandbox has no general internet egress (only
pypi/npm/github are reachable), and all three real datasets sit behind
either a registration wall (ISLES'24 -> grand-challenge.org, verified
account required), a data-use-agreement request form (ATLAS v2.0 ->
Sook-Lei Liew lab), or password-protected Baidu/Drive links (AISD).
So none of them are pullable from here today regardless of format.

What this script buys you instead: every downstream script
(explore_volumes.py) is written and tested against volumes with the
EXACT shape, spacing, and intensity-range conventions of the real
data, so when you run it against your real downloaded files later
(same nibabel calls, same file layout) it should just work unchanged.

Two volumes are generated, matching the two DIFFERENT modalities you'll
actually be working with -- see the note at the bottom of this file
about the ISLES'24 / AISD modality mismatch with the "all-MRI" framing.
"""

import numpy as np
import nibabel as nib
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "synthetic_data")
os.makedirs(OUT_DIR, exist_ok=True)


def _add_ellipsoid_lesion(volume, center, radii, value, rng):
    """Paint a soft-edged ellipsoidal blob into `volume` to stand in for a lesion."""
    D, H, W = volume.shape
    zz, yy, xx = np.meshgrid(
        np.arange(D), np.arange(H), np.arange(W), indexing="ij"
    )
    dist = (
        ((zz - center[0]) / radii[0]) ** 2
        + ((yy - center[1]) / radii[1]) ** 2
        + ((xx - center[2]) / radii[2]) ** 2
    )
    mask = dist <= 1.0
    # soft edge via a smooth falloff instead of a hard boundary
    falloff = np.clip(1.0 - dist, 0.0, 1.0)
    volume[mask] += value * falloff[mask]
    return mask.astype(np.uint8)


def make_atlas_style_mri(seed=0):
    """
    Mimics ATLAS v2.0: T1w, resampled to MNI152 space, ~1mm isotropic.
    Real ATLAS v2.0 volumes are shape ~(197, 233, 189) in MNI152NLin2009aSym
    space. Intensity is arbitrary T1 signal (not HU), roughly 0-1000 a.u.
    after scanner-specific scaling. We keep the same shape/spacing/dtype
    conventions here.
    """
    rng = np.random.default_rng(seed)
    shape = (197, 233, 189)  # matches ATLAS v2.0 MNI template shape
    voxel_spacing = (1.0, 1.0, 1.0)  # isotropic 1mm, as in MNI152 space

    # background: smooth low-frequency field (stand-in for tissue contrast)
    base = rng.normal(loc=400, scale=15, size=(8, 10, 8))
    volume = np.kron(base, np.ones((25, 24, 24)))[: shape[0], : shape[1], : shape[2]]
    volume = volume.astype(np.float32)
    volume += rng.normal(0, 10, size=shape).astype(np.float32)  # scanner noise
    volume = np.clip(volume, 0, None)

    # zero out a "skull-stripped background" border to mimic brain-only MNI volumes
    mask = np.zeros(shape, dtype=bool)
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    center = np.array(shape) / 2
    radii = np.array(shape) / 2.3
    brain_mask = (
        ((zz - center[0]) / radii[0]) ** 2
        + ((yy - center[1]) / radii[1]) ** 2
        + ((xx - center[2]) / radii[2]) ** 2
    ) <= 1.0
    volume *= brain_mask

    # synthetic lesion: T1 lesions are usually hypointense relative to surrounding tissue
    lesion_center = (100, 130, 60)
    lesion_mask = _add_ellipsoid_lesion(
        volume, lesion_center, radii=(10, 14, 12), value=-120, rng=rng
    )
    volume = np.clip(volume, 0, None) * brain_mask

    affine = np.diag([voxel_spacing[0], voxel_spacing[1], voxel_spacing[2], 1.0])
    img = nib.Nifti1Image(volume.astype(np.float32), affine)
    img.header["descrip"] = b"SYNTHETIC PLACEHOLDER - not real ATLAS data"
    mask_img = nib.Nifti1Image(lesion_mask, affine)
    mask_img.header["descrip"] = b"SYNTHETIC PLACEHOLDER - not real ATLAS data"

    img_path = os.path.join(OUT_DIR, "sub-fake01_ses-1_space-MNI152_T1w.nii.gz")
    mask_path = os.path.join(
        OUT_DIR, "sub-fake01_ses-1_space-MNI152_label-L_desc-T1lesion_mask.nii.gz"
    )
    nib.save(img, img_path)
    nib.save(mask_img, mask_path)
    return img_path, mask_path


def make_ct_style_ncct(seed=1):
    """
    Mimics AISD / ISLES'24 NCCT: anisotropic spacing (thick 5mm slices are
    typical for clinical stroke CT protocols), intensity in Hounsfield
    Units (HU). Brain parenchyma sits ~20-40 HU, CSF ~0-15 HU, skull/bone
    >700 HU, acute ischemic lesions are SUBTLY hypodense (a few HU lower
    than surrounding normal parenchyma -- this subtlety is exactly why
    NCCT stroke segmentation is hard, and why AISD/ISLES'24 use DWI-MRI
    as the annotation reference standard rather than trusting the CT
    alone).
    """
    rng = np.random.default_rng(seed)
    shape = (45, 512, 512)  # (slices, H, W): 5mm slice thickness -> few slices
    voxel_spacing = (5.0, 0.45, 0.45)  # thick axial slices, fine in-plane res

    volume = np.full(shape, fill_value=-1000, dtype=np.float32)  # air = -1000 HU

    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    center = np.array(shape) / 2
    radii = np.array([shape[0] / 2.2, shape[1] / 2.4, shape[2] / 2.4])
    skull_outer = (
        ((zz - center[0]) / radii[0]) ** 2
        + ((yy - center[1]) / radii[1]) ** 2
        + ((xx - center[2]) / radii[2]) ** 2
    ) <= 1.0
    inner_radii = radii * 0.92
    brain_region = (
        ((zz - center[0]) / inner_radii[0]) ** 2
        + ((yy - center[1]) / inner_radii[1]) ** 2
        + ((xx - center[2]) / inner_radii[2]) ** 2
    ) <= 1.0
    skull_shell = skull_outer & ~brain_region

    volume[skull_outer] = 30.0  # parenchyma baseline
    volume[skull_shell] = 750.0  # dense bone
    volume += rng.normal(0, 6, size=shape).astype(np.float32) * skull_outer

    # subtle hypodense lesion: only ~8 HU below normal parenchyma, deliberately
    # faint -- this is the realistic difficulty, not a stylized "obvious blob"
    lesion_center = (22, 260, 320)
    lesion_mask = _add_ellipsoid_lesion(
        volume, lesion_center, radii=(6, 22, 20), value=-8, rng=rng
    )

    affine = np.diag([voxel_spacing[0], voxel_spacing[1], voxel_spacing[2], 1.0])
    img = nib.Nifti1Image(volume.astype(np.float32), affine)
    img.header["descrip"] = b"SYNTHETIC PLACEHOLDER - not real AISD/ISLES data"
    mask_img = nib.Nifti1Image(lesion_mask, affine)
    mask_img.header["descrip"] = b"SYNTHETIC PLACEHOLDER - not real AISD/ISLES data"

    img_path = os.path.join(OUT_DIR, "sub-fake02_ncct.nii.gz")
    mask_path = os.path.join(OUT_DIR, "sub-fake02_ncct_lesion_mask.nii.gz")
    nib.save(img, img_path)
    nib.save(mask_img, mask_path)
    return img_path, mask_path


if __name__ == "__main__":
    mri_img, mri_mask = make_atlas_style_mri()
    ct_img, ct_mask = make_ct_style_ncct()
    print("Synthetic ATLAS-style MRI written to:", mri_img)
    print("Synthetic ATLAS-style lesion mask written to:", mri_mask)
    print("Synthetic AISD/ISLES-style NCCT written to:", ct_img)
    print("Synthetic AISD/ISLES-style lesion mask written to:", ct_mask)

# ---------------------------------------------------------------------------
# MODALITY NOTE (read this before the meeting):
# ISLES'24's inputs are NCCT + CTA + 4D CTP + perfusion maps -- it is a CT
# challenge, not MRI. AISD's segmentation *input* is also NCCT (DWI-MRI is
# only the reference standard used to draw the lesion masks, acquired
# separately within 24h). Of your three named datasets, only ATLAS v2.0 is
# actually MRI end-to-end. See the chat response for what this means for
# the SIGNAJ pitch.
# ---------------------------------------------------------------------------
