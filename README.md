# SIGNAJ

## 1. Data exploration (`data_exploration/`)

**What's real:** the nibabel loading, metadata reporting, HU-windowing /
z-score preprocessing, isotropic resampling, and orientation-correct
visualization code. These would work unchanged against real files.

**What's NOT real:** the volumes themselves. None of ISLES'24, ATLAS v2.0,
or AISD could be downloaded in this environment — ISLES'24 needs a verified
grand-challenge.org account, ATLAS v2.0 needs a data-use-agreement request
to the Liew lab, and AISD sits behind password-protected Baidu/Drive links.
So `make_synthetic_volumes.py` generates placeholder volumes matching each
dataset's real shape, voxel spacing, and intensity convention, and
`explore_volumes.py` runs against those. Point it at real files later and
it should need no changes.

**Important correction to flag before the meeting:** ISLES'24's inputs are
NCCT + CTA + 4D CTP (a CT challenge), and AISD's segmentation input is also
NCCT (DWI-MRI is only the annotation reference standard). Of the three
datasets, only ATLAS v2.0 is MRI end-to-end. Worth resolving how this
affects the "VLM for MRI lesion detection" framing before you're asked
about it live.

**Explicitly not attempted:** skull stripping (needs a trained tool —
HD-BET/SynthStrip — not something to fake), and registration to a common
template (ANTs/FSL FLIRT, needs real data + real QA).

## 2. Voxel Commitment Module (`vcm/`)

**What's real:** a working `nn.Module` — cross-attention-style logits over
a flattened 3D feature grid, Gumbel-Softmax with straight-through hard
commitment, voxel coordinate recovery, and a toy multi-step decoder loop
showing the module called once per diagnostic token. All shapes and
gradient flow are verified in `test_vcm.py` (passes).

**What's NOT real:** no training, no real encoder feeding it, no L_SCG /
L_HP loss terms wired up, no real diagnostic vocabulary. The attention
formulation (single dot-product query/key) is a first draft, not a
validated design choice — say so if asked.

## Why multimodal

Early framing of this proposal leaned on "MRI lesion detection" language.
While building `data_exploration/`, we confirmed that doesn't match the
actual dataset mix: ISLES 2024's inputs are NCCT + CTA + 4D CTP (a CT
challenge), and AISD's segmentation input is also NCCT (DWI-MRI is only
the annotation reference standard, not what the model would see). Of the
three target datasets, only ATLAS v2.0 is MRI end-to-end. So two of three
are CT, not MRI -- the framing needed to catch up to that.

In response, VCM (`vcm/voxel_commitment_module.py`) now takes an optional
modality tag ("CT" or "MRI") that's added as a small learned embedding to
the query before it computes attention logits over the 3D feature grid.
Concretely: the module no longer has to assume CT and MRI features look
alike -- it can shift what it attends to depending on which modality it's
looking at. `test_vcm.py` proves the mechanism works (same feature grid,
same query, different modality tag -> different attention output).

We think designing for both modalities from the start is a strength, not
a workaround. It's a direct, structural answer to the distribution-shift
concern already raised by a co-supervisor: rather than training on one
modality and hoping it generalizes to the other, or discovering the
mismatch after committing to an MRI-only framing, the architecture treats
modality as a first-class signal from day one. That said, this is still
an open design decision, not a closed one -- worth asking directly
whether CT-and-MRI-together is the right scope, or whether it should be
split differently.

Same honesty caveat as everywhere else in this doc: untrained, and one
shared additive embedding is the simplest first draft, not a validated
mechanism. `test_vcm.py` proves the module is *capable* of behaving
differently per modality -- it says nothing about whether this is the
*right* way to condition on modality. That needs real CT and MRI data.

## Natural next steps (not done, for the record)
- Wire L_SCG: compare `voxel_coord_norm` against a real ground-truth lesion
  centroid extracted from a lesion mask.
- Attach VCM to a real 3D encoder output instead of `torch.randn`.
- CT-vs-MRI dataset question above: the skeleton now designs for both
  modalities (see "Why multimodal"), but this is untested against real
  CT/MRI data -- that validation is still pending.
