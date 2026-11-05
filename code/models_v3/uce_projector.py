"""SAGA v3.0 -- UCE → Qwen projector.

Input: a single 1280-dim L2-normalised cell embedding from pretrained
UCE-33L (precomputed offline by `scripts_v3/precompute_uce_embeddings.py`).

Pipeline (per cell):

    UCE_emb (1280)
         |
         v
    expand MLP -> (M, hidden)        # M learned slots
         + positional embedding
         |
         v
    token mixer (small Transformer)  # let slots interact
         |
         v
    to_tokens linear -> (M, 5120)    # Qwen2.5-14B hidden size
         |
         v
    [injected at <CELL> positions in the Qwen prompt]

Auxiliary heads classify raw UCE or pooled projected tokens. The raw-UCE
branch does not backpropagate into the token-expansion path. The pooled
branch does. Parameter count depends on the selected architecture and heads;
constructor defaults are not the selected paper configuration.
"""

from __future__ import annotations
import torch
import torch.nn as nn


UCE_HIDDEN_DIM = 1280


class UCEProjectorV3(nn.Module):
    def __init__(
        self,
        uce_dim: int = UCE_HIDDEN_DIM,
        hidden: int = 512,
        num_tokens: int = 16,
        output_dim: int = 5120,           # Qwen2.5-14B hidden size
        num_mixer_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        n_cell_types: int = 44,
        n_organs: int = 8,
        n_hierarchy_fine: int = 0,
        n_hierarchy_mid: int = 0,
        n_hierarchy_organs: int = 0,
    ):
        super().__init__()
        self.uce_dim = uce_dim
        self.hidden = hidden
        self.num_tokens = num_tokens
        self.output_dim = output_dim

        self.expand = nn.Sequential(
            nn.Linear(uce_dim, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, num_tokens * hidden),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, hidden) * 0.02)
        self.in_norm = nn.LayerNorm(hidden)

        if num_mixer_layers > 0:
            enc = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=num_heads,
                dim_feedforward=hidden * 4, dropout=dropout,
                batch_first=True, activation="gelu", norm_first=True,
            )
            self.token_mixer = nn.TransformerEncoder(enc, num_layers=num_mixer_layers)
        else:
            self.token_mixer = nn.Identity()
        self.out_norm = nn.LayerNorm(hidden)
        self.to_tokens = nn.Linear(hidden, output_dim)

        # Auxiliary direct classifier on the raw UCE vector (BEFORE expansion).
        # This separate branch does not supervise the token-expansion path.
        self.aux_classifier = nn.Sequential(
            nn.LayerNorm(uce_dim),
            nn.Linear(uce_dim, hidden),
            nn.GELU(),
        )
        self.aux_ct_head = nn.Linear(hidden, n_cell_types)
        self.aux_org_head = nn.Linear(hidden, n_organs)

        # Auxiliary classifier on the actual projected token space. The raw-UCE
        # head is independent; this head can supervise the Qwen-facing
        # representation after token expansion when included in the loss.
        self.pooled_classifier = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, hidden),
            nn.GELU(),
        )
        self.pooled_ct_head = nn.Linear(hidden, n_cell_types)
        self.pooled_org_head = nn.Linear(hidden, n_organs)

        # Optional all-atlas hierarchy supervision. These heads consume the
        # actual Qwen-facing pooled Projector representation; they never read
        # raw UCE directly. Keeping distinct names also preserves compatibility
        # with legacy checkpoints whose pooled heads used the old TS vocab.
        self.hierarchy_classifier = None
        self.hierarchy_fine_head = None
        self.hierarchy_mid_head = None
        self.hierarchy_organ_head = None
        if n_hierarchy_fine > 0 and n_hierarchy_mid > 0 and n_hierarchy_organs > 0:
            self.hierarchy_classifier = nn.Sequential(
                nn.LayerNorm(output_dim),
                nn.Linear(output_dim, hidden),
                nn.GELU(),
            )
            self.hierarchy_fine_head = nn.Linear(hidden, n_hierarchy_fine)
            self.hierarchy_mid_head = nn.Linear(hidden, n_hierarchy_mid)
            self.hierarchy_organ_head = nn.Linear(hidden, n_hierarchy_organs)

    def encode(self, uce_emb: torch.Tensor):
        """uce_emb : (B, 1280) -> (B, M, hidden), aux_logits."""
        ref = self.expand[0].weight
        uce_emb = uce_emb.to(device=ref.device, dtype=ref.dtype)
        B = uce_emb.size(0)
        x = self.expand(uce_emb).view(B, self.num_tokens, self.hidden)
        x = self.in_norm(x + self.pos_embed)
        x = self.token_mixer(x)
        x = self.out_norm(x)

        h = self.aux_classifier(uce_emb)
        ct_logits = self.aux_ct_head(h)
        org_logits = self.aux_org_head(h)
        return x, (ct_logits, org_logits)

    def forward(self, uce_emb: torch.Tensor):
        """Returns (B, M, output_dim) cell tokens for Qwen injection."""
        x, _ = self.encode(uce_emb)
        return self.to_tokens(x)

    def forward_with_aux(self, uce_emb: torch.Tensor):
        """Returns (cell_tokens (B,M,output_dim), aux_logits=(ct, org))."""
        x, aux_logits = self.encode(uce_emb)
        cell_tokens = self.to_tokens(x)
        return cell_tokens, aux_logits

    def forward_with_aux_all(self, uce_emb: torch.Tensor):
        """Returns cell tokens plus raw-UCE and pooled-token auxiliary logits."""
        x, raw_aux_logits = self.encode(uce_emb)
        cell_tokens = self.to_tokens(x)
        pooled = cell_tokens.mean(dim=1)
        h = self.pooled_classifier(pooled)
        pooled_aux_logits = (
            self.pooled_ct_head(h),
            self.pooled_org_head(h),
        )
        hierarchy_logits = None
        if self.hierarchy_classifier is not None:
            hierarchy_hidden = self.hierarchy_classifier(pooled)
            hierarchy_logits = (
                self.hierarchy_fine_head(hierarchy_hidden),
                self.hierarchy_mid_head(hierarchy_hidden),
                self.hierarchy_organ_head(hierarchy_hidden),
            )
        return cell_tokens, {
            "raw": raw_aux_logits,
            "pooled": pooled_aux_logits,
            "pooled_features": pooled,
            "hierarchy": hierarchy_logits,
        }

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
