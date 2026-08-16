"""
test_vcm.py
===========
Sanity checks only. This does NOT train anything -- it verifies:
  1. Shapes are correct end to end.
  2. The module is differentiable (loss.backward() runs, gradients
     reach the encoder-side feature grid and the module's own params).
  3. voxel_idx values are valid coordinates (within grid bounds).
  4. The multi-token loop composes correctly.

If this file passes, you know the module is MECHANICALLY sound. It
tells you nothing about whether it's the RIGHT design -- that needs
real data, a real loss, and real training.
"""

import torch
from voxel_commitment_module import VoxelCommitmentModule, DiagnosticTokenLoopDemo


def test_single_commitment_shapes():
    B, C, D, H, W = 2, 256, 8, 16, 16  # e.g. a 128^3 input downsampled 16x by a 3D encoder
    query_dim = 512  # decoder hidden size, independent of feature_dim on purpose

    feature_grid = torch.randn(B, C, D, H, W, requires_grad=True)
    query = torch.randn(B, query_dim)

    vcm = VoxelCommitmentModule(feature_dim=C, query_dim=query_dim, tau=1.0, hard=True)
    out = vcm(feature_grid, query)

    assert out["committed_feature"].shape == (B, C)
    assert out["voxel_idx"].shape == (B, 3)
    assert out["voxel_coord_norm"].shape == (B, 3)
    assert out["attn_weights"].shape == (B, D * H * W)

    # voxel indices must be valid grid coordinates
    assert (out["voxel_idx"][:, 0] < D).all() and (out["voxel_idx"][:, 0] >= 0).all()
    assert (out["voxel_idx"][:, 1] < H).all() and (out["voxel_idx"][:, 1] >= 0).all()
    assert (out["voxel_idx"][:, 2] < W).all() and (out["voxel_idx"][:, 2] >= 0).all()

    # normalized coords must be in [0, 1]
    assert (out["voxel_coord_norm"] >= 0).all() and (out["voxel_coord_norm"] <= 1).all()

    print("[PASS] single commitment shapes:")
    print(f"       committed_feature {out['committed_feature'].shape}, "
          f"voxel_idx {out['voxel_idx'].shape}, "
          f"voxel_coord_norm {out['voxel_coord_norm'].shape}")
    print(f"       example voxel_idx[0] = {out['voxel_idx'][0].tolist()} "
          f"(grid is {D}x{H}x{W})")
    return vcm, feature_grid, query, out


def test_gradient_flow():
    vcm, feature_grid, query, out = test_single_commitment_shapes()

    # fake downstream loss: pretend committed_feature feeds a linear head
    # predicting something, and we backprop a dummy target
    head = torch.nn.Linear(vcm.feature_dim, 10)
    logits = head(out["committed_feature"])
    target = torch.randint(0, 10, (logits.shape[0],))
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()

    assert feature_grid.grad is not None, "gradient did not reach the input feature grid"
    assert not torch.isnan(feature_grid.grad).any(), "NaN gradients in feature grid"
    assert vcm.key_proj.weight.grad is not None, "gradient did not reach VCM's own parameters"

    print("[PASS] gradient flow: loss.backward() reached feature_grid and VCM params, no NaNs")
    print(f"       loss value: {loss.item():.4f}")


def test_soft_vs_hard():
    """Hard mode should give a (near) one-hot distribution; soft mode should not."""
    B, C, D, H, W = 1, 32, 4, 4, 4
    feature_grid = torch.randn(B, C, D, H, W)
    query = torch.randn(B, C)

    vcm_hard = VoxelCommitmentModule(feature_dim=C, tau=0.5, hard=True)
    vcm_soft = VoxelCommitmentModule(feature_dim=C, tau=0.5, hard=False)
    # share weights so the comparison isolates the hard/soft difference
    vcm_soft.load_state_dict(vcm_hard.state_dict())

    hard_weights = vcm_hard(feature_grid, query)["attn_weights"]
    soft_weights = vcm_soft(feature_grid, query)["attn_weights"]

    hard_max = hard_weights.max().item()
    soft_max = soft_weights.max().item()

    print(f"[PASS] soft vs hard: hard max weight = {hard_max:.4f} "
          f"(expect ~1.0), soft max weight = {soft_max:.4f} (expect < 1.0)")
    assert hard_max > 0.99, "hard mode should be (near) one-hot"


def test_modality_changes_attention():
    """Same feature grid, same query, different modality tag -> different
    attention output. Proves the modality embedding actually reaches the
    attention logits, not just that the module runs without it.

    Uses soft mode (hard=False) so we compare distributions rather than
    argmax indices, and resets the RNG seed immediately before each
    forward call so the Gumbel noise draw is identical across calls --
    that isolates any difference in the output to the modality embedding
    itself, not to random chance.
    """
    B, C, D, H, W = 2, 32, 4, 4, 4
    query_dim = 32

    vcm = VoxelCommitmentModule(feature_dim=C, query_dim=query_dim, tau=0.5, hard=False)
    feature_grid = torch.randn(B, C, D, H, W)
    query = torch.randn(B, query_dim)

    torch.manual_seed(0)
    out_ct = vcm(feature_grid, query, modality="CT")
    torch.manual_seed(0)
    out_mri = vcm(feature_grid, query, modality="MRI")

    assert not torch.allclose(out_ct["attn_weights"], out_mri["attn_weights"]), (
        "attention output should differ between CT and MRI modality tags"
    )

    # Sanity companion: same modality + same seed -> identical output.
    # Confirms the difference above is attributable to the modality tag,
    # not some other source of nondeterminism.
    torch.manual_seed(0)
    out_ct_again = vcm(feature_grid, query, modality="CT")
    assert torch.allclose(out_ct["attn_weights"], out_ct_again["attn_weights"])

    print("[PASS] modality changes attention: CT and MRI tags on the same "
          "feature grid/query produce different attn_weights "
          "(max abs diff = "
          f"{(out_ct['attn_weights'] - out_mri['attn_weights']).abs().max().item():.4f}), "
          "while repeating the same tag with the same seed reproduces "
          "identical output.")


def test_multi_token_loop():
    B, C, D, H, W = 2, 64, 6, 8, 8
    hidden_dim = 128
    vocab_size = 50  # placeholder diagnostic-token vocabulary size
    num_tokens = 5

    feature_grid = torch.randn(B, C, D, H, W, requires_grad=True)
    model = DiagnosticTokenLoopDemo(
        feature_dim=C, hidden_dim=hidden_dim, vocab_size=vocab_size
    )
    out = model(feature_grid, num_tokens=num_tokens)

    assert out["token_logits"].shape == (B, num_tokens, vocab_size)
    assert out["voxel_idx_seq"].shape == (B, num_tokens, 3)
    assert out["voxel_coord_seq"].shape == (B, num_tokens, 3)

    # confirm the model can actually generate DIFFERENT commitments across
    # tokens (if every token committed to the same voxel, that would be a
    # red flag, not a feature)
    idx_seq = out["voxel_idx_seq"][0]  # (T, 3)
    unique_voxels = len(set(tuple(v.tolist()) for v in idx_seq))
    print(f"[PASS] multi-token loop: {num_tokens} tokens, "
          f"{unique_voxels}/{num_tokens} distinct voxel commitments "
          f"(random init, so distinctness isn't guaranteed each run -- "
          f"just checking the loop composes and runs)")

    target = torch.randint(0, vocab_size, (B, num_tokens))
    loss = torch.nn.functional.cross_entropy(
        out["token_logits"].reshape(-1, vocab_size), target.reshape(-1)
    )
    loss.backward()
    assert feature_grid.grad is not None
    print(f"       loop-level loss.backward() also reaches feature_grid. loss={loss.item():.4f}")


if __name__ == "__main__":
    print("=" * 70)
    test_gradient_flow()
    print("=" * 70)
    test_soft_vs_hard()
    print("=" * 70)
    test_modality_changes_attention()
    print("=" * 70)
    test_multi_token_loop()
    print("=" * 70)
    print("\nAll sanity checks passed. This confirms MECHANICAL correctness")
    print("only -- shapes, gradients, valid coordinates. No claim about")
    print("whether this is the right architecture until it sees real data.")
