"""SingleCellLLMv3 -- minimal SAGA wrapper around UCE-precomputed cell
embeddings.

UCE is run offline. At training time we just look up a (1280,) vector per
cell, push it through ``UCEProjectorV3`` to get M cell tokens, inject those
at ``<CELL>`` placeholders in a Qwen prompt, and run frozen Qwen.

Qwen-aware readout / VAE recon are dropped (UCE already gives a SOTA cell
encoder, no need for self-distillation anymore). The only auxiliary loss
is a CE head sitting on the projector input (UCE 1280-d) for direct
supervision.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .uce_projector import UCEProjectorV3, UCE_HIDDEN_DIM


CELL_PLACEHOLDER = "<CELL>"


class SingleCellLLMv3(nn.Module):
    def __init__(
        self,
        llm_path: str,
        uce_dim: int = UCE_HIDDEN_DIM,
        hidden: int = 512,
        num_cell_tokens: int = 16,
        num_mixer_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        freeze_llm: bool = True,
        n_cell_types: int = 44,
        n_organs: int = 8,
        n_hierarchy_fine: int = 0,
        n_hierarchy_mid: int = 0,
        n_hierarchy_organs: int = 0,
        llm_dtype: torch.dtype = torch.bfloat16,
        llm_device_map: str | dict | None = "single",
        cell_token_mode: str | None = None,
    ):
        super().__init__()
        self.num_cell_tokens = num_cell_tokens
        self.llm_dtype = llm_dtype
        self.cell_token_mode = (cell_token_mode or os.environ.get(
            "SAGA_CELL_TOKEN_MODE", "resize")).lower()
        if self.cell_token_mode not in {"resize", "literal"}:
            raise ValueError(
                f"Unsupported cell_token_mode={self.cell_token_mode!r}; "
                "choose 'resize' or 'literal'."
            )
        if llm_device_map in ("local_rank", "ddp") and torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        added_cell_token = False
        if self.cell_token_mode == "resize":
            added = self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [CELL_PLACEHOLDER]}
            )
            added_cell_token = bool(added)
        if llm_device_map in (None, "", "single", "cuda:0", "0"):
            device_map = {"": 0}
        elif llm_device_map in ("local_rank", "ddp"):
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            device_map = {"": local_rank}
        elif llm_device_map == "auto":
            device_map = "auto"
        elif isinstance(llm_device_map, dict):
            device_map = llm_device_map
        else:
            raise ValueError(
                f"Unsupported llm_device_map={llm_device_map!r}. "
                "Use 'single', 'local_rank', 'ddp', or 'auto'."
            )
        self.is_parallelizable = device_map == "auto"
        self.model_parallel = device_map == "auto"
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path, torch_dtype=llm_dtype, device_map=device_map,
            trust_remote_code=True,
        )
        if self.cell_token_mode == "resize":
            # Old v3 projector checkpoints were trained with a real added
            # <CELL> token. Keep this mode available for checkpoint parity.
            self.llm.resize_token_embeddings(len(self.tokenizer))
        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False
            self.llm.enable_input_require_grads()

        hidden_size = self.llm.config.hidden_size
        self.projector = UCEProjectorV3(
            uce_dim=uce_dim, hidden=hidden, num_tokens=num_cell_tokens,
            output_dim=hidden_size, num_mixer_layers=num_mixer_layers,
            num_heads=num_heads, dropout=dropout,
            n_cell_types=n_cell_types, n_organs=n_organs,
            n_hierarchy_fine=n_hierarchy_fine,
            n_hierarchy_mid=n_hierarchy_mid,
            n_hierarchy_organs=n_hierarchy_organs,
        )
        device = self.input_device()
        self.projector.to(device=device, dtype=self.llm_dtype)

        if self.cell_token_mode == "resize":
            # Legacy/default path used by the successful 6/11 projector runs.
            # Projectors trained with this interface expect one dedicated
            # placeholder token per <CELL> slot.
            self.cell_token_ids = [self.tokenizer.convert_tokens_to_ids(CELL_PLACEHOLDER)]
            self.cell_token_id = self.cell_token_ids[0]
        else:
            # Memory-light path for small-GPU DDP experiments. Checkpoints
            # trained with resize mode are not compatible with this interface.
            self.cell_token_ids = self.tokenizer(
                CELL_PLACEHOLDER, add_special_tokens=False)["input_ids"]
            if not self.cell_token_ids:
                raise ValueError(f"Tokenizer produced no ids for {CELL_PLACEHOLDER!r}")
            self.cell_token_id = self.cell_token_ids[0] if len(self.cell_token_ids) == 1 else None
        print(
            f"[v3] cell_token_mode={self.cell_token_mode} "
            f"added_cell_token={added_cell_token} cell_token_ids={self.cell_token_ids}",
            flush=True,
        )
        if freeze_llm:
            self._mark_frozen_llm_ignored_by_ddp()

    # ----------------------------------------------------------- internals
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

    def _embed_layer(self):
        if hasattr(self.llm, "model") and hasattr(self.llm.model, "embed_tokens"):
            return self.llm.model.embed_tokens
        if hasattr(self.llm, "transformer") and hasattr(self.llm.transformer, "wte"):
            return self.llm.transformer.wte
        raise AttributeError("Cannot locate token embedding layer.")

    def input_device(self):
        return self._embed_layer().weight.device

    def _mark_frozen_llm_ignored_by_ddp(self):
        ignored = [
            name for name, _ in self.named_parameters()
            if name.startswith("llm.")
        ]
        ignored.extend(
            name for name, _ in self.named_buffers()
            if name.startswith("llm.")
        )
        # DDP reads this private-but-standard attribute before wrapping. It
        # prevents syncing the frozen Qwen states; only the projector is trained.
        self._ddp_params_and_buffers_to_ignore = ignored

    def _inject(self, text_embeds, input_ids, projected):
        for b in range(input_ids.size(0)):
            if self.cell_token_id is not None:
                positions = (input_ids[b] == self.cell_token_id).nonzero(as_tuple=True)[0]
            else:
                positions = self._cell_placeholder_positions(input_ids[b])
            n_pos = min(len(positions), self.num_cell_tokens)
            for i in range(n_pos):
                text_embeds[b, positions[i]] = projected[b, i]
        return text_embeds

    def _cell_placeholder_positions(self, input_ids_1d):
        needle = torch.tensor(self.cell_token_ids, device=input_ids_1d.device)
        n = int(needle.numel())
        if n == 0 or input_ids_1d.numel() < n:
            return input_ids_1d.new_empty((0,), dtype=torch.long)
        matches = []
        for start in range(0, input_ids_1d.numel() - n + 1):
            if torch.equal(input_ids_1d[start:start + n], needle):
                matches.append(start)
        if not matches:
            return input_ids_1d.new_empty((0,), dtype=torch.long)
        return torch.tensor(matches, device=input_ids_1d.device, dtype=torch.long)

    # ---------------------------------------------------------- public API
    def build_inputs_embeds(self, input_ids, uce_emb):
        embed_layer = self._embed_layer()
        text_embeds = embed_layer(input_ids).clone()
        projected = self.projector(uce_emb)
        return self._inject(text_embeds, input_ids, projected)

    def build_inputs_embeds_with_aux(self, input_ids, uce_emb):
        embed_layer = self._embed_layer()
        text_embeds = embed_layer(input_ids).clone()
        projected, aux_logits = self.projector.forward_with_aux(uce_emb)
        text_embeds = self._inject(text_embeds, input_ids, projected)
        return text_embeds, aux_logits

    def build_inputs_embeds_with_aux_all(self, input_ids, uce_emb):
        embed_layer = self._embed_layer()
        text_embeds = embed_layer(input_ids).clone()
        projected, aux_payload = self.projector.forward_with_aux_all(uce_emb)
        text_embeds = self._inject(text_embeds, input_ids, projected)
        return text_embeds, aux_payload

    def forward(self, input_ids, attention_mask, uce_emb, labels=None):
        inputs_embeds = self.build_inputs_embeds(input_ids, uce_emb)
        return self.llm(inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask, labels=labels)

    def forward_with_aux(self, input_ids, attention_mask, uce_emb, labels=None):
        inputs_embeds, aux_logits = self.build_inputs_embeds_with_aux(
            input_ids, uce_emb)
        out = self.llm(inputs_embeds=inputs_embeds,
                       attention_mask=attention_mask, labels=labels)
        return out, aux_logits

    def forward_with_aux_all(self, input_ids, attention_mask, uce_emb, labels=None):
        inputs_embeds, aux_payload = self.build_inputs_embeds_with_aux_all(
            input_ids, uce_emb)
        out = self.llm(inputs_embeds=inputs_embeds,
                       attention_mask=attention_mask, labels=labels)
        return out, aux_payload

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, uce_emb, **gen_kwargs):
        inputs_embeds = self.build_inputs_embeds(input_ids, uce_emb)
        return self.llm.generate(inputs_embeds=inputs_embeds,
                                 attention_mask=attention_mask, **gen_kwargs)

    # ------------------------------------------------------------ persist
    def save_projector(self, path):
        torch.save(self.projector.state_dict(), path)

    def load_projector(self, path, strict=True):
        sd = torch.load(path, map_location="cpu", weights_only=True)
        # Hierarchy heads are training-only auxiliary readouts.  Evaluation
        # models intentionally omit them, while retaining strict validation
        # for every tensor that contributes to the cell-token interface.
        hierarchy_prefixes = (
            "hierarchy_classifier.",
            "hierarchy_fine_head.",
            "hierarchy_mid_head.",
            "hierarchy_organ_head.",
        )
        ignored_hierarchy = [key for key in sd if key.startswith(hierarchy_prefixes)]
        if ignored_hierarchy:
            sd = {key: value for key, value in sd.items() if key not in ignored_hierarchy}
            print(
                f"[v3] projector inference load: ignored {len(ignored_hierarchy)} "
                "training-only hierarchy tensors",
                flush=True,
            )
        if strict:
            current = self.projector.state_dict()
            missing_keys = sorted(set(current) - set(sd))
            compatible_missing = all(
                key.startswith(("pooled_classifier.", "pooled_ct_head.", "pooled_org_head."))
                for key in missing_keys
            )
            if missing_keys and compatible_missing:
                missing, unexpected = self.projector.load_state_dict(sd, strict=False)
                print(
                    f"[v3] projector load legacy checkpoint: initialized "
                    f"{len(missing)} new pooled-head tensors",
                    flush=True,
                )
            else:
                missing, unexpected = self.projector.load_state_dict(sd, strict=True)
        else:
            missing, unexpected = self.projector.load_state_dict(sd, strict=False)
        if (missing or unexpected) and not strict:
            print(
                f"[v3] projector load non-strict: missing={len(missing)} "
                f"unexpected={len(unexpected)}",
                flush=True,
            )
        device = self.input_device()
        self.projector.to(device=device, dtype=self.llm_dtype)
