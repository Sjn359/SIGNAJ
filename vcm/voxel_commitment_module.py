"""
voxel_commitment_module.py
===========================
SKELETON ONLY -- structure and forward-pass shapes, not trained, not
optimized, not validated against any real loss curve. Every design
choice below (dot-product attention for the logits, temperature=1.0,
straight-through Gumbel-Softmax) is a defensible FIRST DRAFT, not a
finished decision. Flag them as open questions in the meeting, don't
present them as settled.

WHAT THIS IS FOR
----------------
SIGNAJ's core claim is that each diagnostic token the model emits
(e.g. "hyperdensity", "acute infarct") should be traceable to ONE
specific 3D voxel in the scan, rather than being a free-floating
caption the model could hallucinate. The Voxel Commitment Module
(VCM) is the piece that enforces this: given a 3D feature grid from
an encoder and the decoder's current hidden state (its "intent" for
the token about to be generated), VCM produces a HARD (near one-hot)
commitment to a single spatial location, using Gumbel-Softmax with a
straight-through estimator so gradients still flow during training.

WHAT THIS IS *NOT*
-------------------
- Not attached to a real encoder. `feature_grid` here is a dummy
  tensor; a real version would come from a 3D CNN/ViT backbone
  applied to the preprocessed volumes (see ../data_exploration/).
- Not attached to L_SCG or L_HP (the SIGNAJ loss terms). Those need
  the committed voxel_coord_norm output compared against a real
  ground-truth lesion centroid/mask -- that's the natural next piece
  to build, not built here.
- The attention formulation (single dot-product query/key) is the
  simplest thing that could plausibly work, not a claim that it's the
  right inductive bias. A multi-head version, or a version that
  incorporates a distance/locality prior, are both open questions.
- Modality-conditioning (below) is structurally present but likewise a
  first draft: a single shared embedding table added to the query is
  the simplest thing that could work, not a claim that CT and MRI only
  differ by a query-side shift -- separate per-modality `key_proj`
  branches, or normalizing intensities upstream instead, are open
  alternatives. Untested against real CT/MRI data; `test_vcm.py` only
  proves the mechanism is *capable* of behaving differently per
  modality, not that it does so usefully.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Maps a human-readable modality tag to the embedding row VCM looks up.
# SIGNAJ's three target datasets are CT (ISLES 2024, AISD) and MRI
# (ATLAS v2.0) -- see README.md "Why multimodal" for why this exists.
MODALITY_TO_IDX = {"CT": 0, "MRI": 1}


class VoxelCommitmentModule(nn.Module):
    """
    Args:
        feature_dim:    channel dimension C of the 3D feature grid.
        query_dim:      dimension of the decoder hidden state used as the
                         "query" for this commitment step. Defaults to
                         feature_dim if not given.
        tau:            Gumbel-Softmax temperature. Higher = softer/more
                         exploratory early in training; anneal down over
                         training (not implemented here -- a training-loop
                         concern, not a module-structure concern).
        hard:           if True, forward pass returns a (near) one-hot
                         selection (straight-through); if False, returns
                         the soft attention distribution (useful for
                         debugging/visualizing what the module attends to
                         before it's forced to commit).
        num_modalities: size of the modality embedding table. Defaults to
                         len(MODALITY_TO_IDX) (currently CT, MRI).
    """

    def __init__(
        self,
        feature_dim: int,
        query_dim: int = None,
        tau: float = 1.0,
        hard: bool = True,
        num_modalities: int = len(MODALITY_TO_IDX),
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.query_dim = query_dim or feature_dim
        self.tau = tau
        self.hard = hard

        self.query_proj = nn.Linear(self.query_dim, feature_dim)
        self.key_proj = nn.Conv3d(feature_dim, feature_dim, kernel_size=1)
        # Small learned per-modality bias added to the query before it's
        # compared against keys -- lets VCM weigh voxel evidence
        # differently depending on whether it's looking at a CT or MRI
        # feature grid, instead of assuming the two look alike.
        self.modality_embed = nn.Embedding(num_modalities, feature_dim)

    def _modality_to_idx(self, modality, batch_size: int, device) -> torch.Tensor:
        """Normalize a modality tag (single string or per-sample list) to
        a (B,) LongTensor of embedding indices."""
        if isinstance(modality, str):
            modality = [modality] * batch_size
        assert len(modality) == batch_size, (
            f"modality list length {len(modality)} != batch size {batch_size}"
        )
        try:
            idx = [MODALITY_TO_IDX[m] for m in modality]
        except KeyError as e:
            raise ValueError(
                f"unknown modality {e}; expected one of {list(MODALITY_TO_IDX)}"
            )
        return torch.tensor(idx, device=device, dtype=torch.long)

    def forward(self, feature_grid: torch.Tensor, query: torch.Tensor, modality=None):
        """
        feature_grid: (B, C, D, H, W)  -- 3D feature volume from the encoder
        query:        (B, query_dim)   -- decoder state for the token about
                                           to be generated
        modality:     optional; a single modality string (e.g. "CT",
                      broadcast across the batch) or a list of B strings,
                      one per sample, drawn from MODALITY_TO_IDX. If None
                      (default), no modality-conditioning is applied and
                      behavior is identical to the modality-agnostic
                      version of this module.

        Returns a dict:
            committed_feature: (B, C)     -- fed to the LM head / next
                                              decoder step for this token
            voxel_idx:          (B, 3)    -- integer (d, h, w) commitment
            voxel_coord_norm:   (B, 3)    -- same, normalized to [0, 1],
                                              for comparison against a GT
                                              lesion coordinate (L_SCG)
            attn_weights:       (B, D*H*W)-- the (near) one-hot distribution
                                              over voxels, for inspection
        """
        B, C, D, H, W = feature_grid.shape
        assert C == self.feature_dim, f"expected feature_dim={self.feature_dim}, got {C}"
        N = D * H * W

        keys = self.key_proj(feature_grid)             # (B, C, D, H, W)
        keys_flat = keys.view(B, C, N)                   # (B, C, N)

        q = self.query_proj(query)                        # (B, C)
        if modality is not None:
            modality_idx = self._modality_to_idx(modality, B, feature_grid.device)
            q = q + self.modality_embed(modality_idx)      # (B, C)
        logits = torch.einsum("bc,bcn->bn", q, keys_flat)  # (B, N)
        logits = logits / (C ** 0.5)                        # scaled dot-product, standard practice

        attn_weights = F.gumbel_softmax(logits, tau=self.tau, hard=self.hard, dim=-1)  # (B, N)

        feat_flat = feature_grid.view(B, C, N)              # (B, C, N)
        committed_feature = torch.einsum("bn,bcn->bc", attn_weights, feat_flat)  # (B, C)

        voxel_flat_idx = attn_weights.argmax(dim=-1)          # (B,)  -- hard index, for reporting only
        d_idx = voxel_flat_idx // (H * W)
        h_idx = (voxel_flat_idx % (H * W)) // W
        w_idx = voxel_flat_idx % W
        voxel_idx = torch.stack([d_idx, h_idx, w_idx], dim=-1)  # (B, 3)

        denom = torch.tensor([D - 1, H - 1, W - 1], device=feature_grid.device, dtype=torch.float32)
        voxel_coord_norm = voxel_idx.float() / denom          # (B, 3), each in [0, 1]

        return {
            "committed_feature": committed_feature,
            "voxel_idx": voxel_idx,
            "voxel_coord_norm": voxel_coord_norm,
            "attn_weights": attn_weights,
        }


class DiagnosticTokenLoopDemo(nn.Module):
    """
    Toy decoder loop showing HOW VCM gets called once per diagnostic
    token during generation -- "before each diagnostic token" from the
    proposal, made concrete. The GRUCell here is a placeholder for
    whatever language decoder SIGNAJ actually uses; nothing about the
    decoder architecture is a real decision yet. All this proves is
    that VCM's inputs/outputs compose correctly across a multi-step
    sequence, and that gradients flow end to end.

    NOTE: does not yet thread a `modality` tag through to the VCM call
    below -- left out of scope for now, but the wiring is a one-line
    change (`self.vcm(feature_grid, hidden, modality=modality)`) if/when
    this demo needs to show a modality-aware generation loop.
    """

    def __init__(self, feature_dim, hidden_dim, vocab_size, tau=1.0, hard=True):
        super().__init__()
        self.vcm = VoxelCommitmentModule(feature_dim, query_dim=hidden_dim, tau=tau, hard=hard)
        self.decoder_cell = nn.GRUCell(feature_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, feature_grid, num_tokens):
        B = feature_grid.shape[0]
        device = feature_grid.device
        hidden = torch.zeros(B, self.hidden_dim, device=device)

        all_logits, all_voxel_idx, all_coords = [], [], []
        for _ in range(num_tokens):
            vcm_out = self.vcm(feature_grid, hidden)
            hidden = self.decoder_cell(vcm_out["committed_feature"], hidden)
            token_logits = self.lm_head(hidden)

            all_logits.append(token_logits)
            all_voxel_idx.append(vcm_out["voxel_idx"])
            all_coords.append(vcm_out["voxel_coord_norm"])

        return {
            "token_logits": torch.stack(all_logits, dim=1),      # (B, T, vocab)
            "voxel_idx_seq": torch.stack(all_voxel_idx, dim=1),  # (B, T, 3)
            "voxel_coord_seq": torch.stack(all_coords, dim=1),   # (B, T, 3)
        }
