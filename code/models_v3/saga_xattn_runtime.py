"""Optional Flamingo-inspired gated cross-attention for SAGA.

The runtime patch leaves the existing SAGA source unchanged.  It subclasses the
loaded SingleCellLLMv3 class, registers trainable adapters under ``projector`` so
existing optimizer and projector checkpoint code can see them, and installs
forward hooks on evenly spaced frozen-Qwen decoder layers.
"""

from __future__ import annotations

import math
import os
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _replace_first_tensor(value: Any, replacement: torch.Tensor) -> Any:
    if torch.is_tensor(value):
        return replacement
    if isinstance(value, tuple):
        items = list(value)
        for index, item in enumerate(items):
            if _first_tensor(item) is not None:
                items[index] = _replace_first_tensor(item, replacement)
                return tuple(items)
    if isinstance(value, list):
        items = list(value)
        for index, item in enumerate(items):
            if _first_tensor(item) is not None:
                items[index] = _replace_first_tensor(item, replacement)
                return items
    raise TypeError(f"Unsupported decoder-layer output type: {type(value)!r}")


class GatedLowRankCrossAttention(nn.Module):
    def __init__(self, hidden_size: int, rank: int, num_heads: int) -> None:
        super().__init__()
        if rank % num_heads:
            raise ValueError(f"rank={rank} must be divisible by heads={num_heads}")
        self.hidden_size = hidden_size
        self.rank = rank
        self.num_heads = num_heads
        self.head_dim = rank // num_heads
        self.query_norm = nn.LayerNorm(hidden_size)
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.query = nn.Linear(hidden_size, rank, bias=False)
        self.key = nn.Linear(hidden_size, rank, bias=False)
        self.value = nn.Linear(hidden_size, rank, bias=False)
        self.output = nn.Linear(rank, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or memory.ndim != 3:
            raise ValueError(
                f"Expected [batch,tokens,hidden], got {tuple(hidden.shape)} and "
                f"{tuple(memory.shape)}"
            )
        if hidden.shape[-1] != self.hidden_size or memory.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Hidden mismatch: expected {self.hidden_size}, got "
                f"{hidden.shape[-1]} and {memory.shape[-1]}"
            )
        if memory.shape[0] != hidden.shape[0]:
            if hidden.shape[0] % memory.shape[0] != 0:
                raise ValueError(
                    f"Batch mismatch: hidden={hidden.shape[0]} memory={memory.shape[0]}"
                )
            repeats = hidden.shape[0] // memory.shape[0]
            memory = memory.repeat_interleave(repeats, dim=0)

        compute_dtype = self.query.weight.dtype
        hidden_compute = hidden.to(dtype=compute_dtype)
        memory_compute = memory.to(device=hidden.device, dtype=compute_dtype)
        query = self.query(self.query_norm(hidden_compute))
        key = self.key(self.memory_norm(memory_compute))
        value = self.value(self.memory_norm(memory_compute))

        batch, query_len, _ = query.shape
        memory_len = key.shape[1]
        query = query.view(batch, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, memory_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, memory_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
        attended = torch.matmul(attention, value)
        attended = attended.transpose(1, 2).contiguous().view(batch, query_len, self.rank)
        delta = self.output(attended).to(dtype=hidden.dtype)
        return hidden + torch.tanh(self.gate).to(dtype=hidden.dtype) * delta


class GatedCrossAttentionBank(nn.Module):
    def __init__(self, hidden_size: int, rank: int, heads: int, count: int) -> None:
        super().__init__()
        self.adapters = nn.ModuleList(
            GatedLowRankCrossAttention(hidden_size, rank, heads) for _ in range(count)
        )

    def gate_values(self) -> list[float]:
        return [float(torch.tanh(adapter.gate.detach()).cpu()) for adapter in self.adapters]


def _freeze_except_xattn(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.projector.saga_xattn.parameters():
        parameter.requires_grad = True

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    unexpected = [name for name in trainable if "saga_xattn" not in name]
    if not trainable or unexpected:
        raise RuntimeError(
            "SAGA-XAttn trainable-parameter audit failed: "
            f"count={len(trainable)} unexpected={unexpected[:20]}"
        )
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(
        f"[SAGA-XAttn trainable] tensors={len(trainable)} parameters={count} "
        f"sample={trainable[:8]}",
        flush=True,
    )


def _resolve_attr(root: Any, path: Sequence[str]) -> Any | None:
    value = root
    for name in path:
        if not hasattr(value, name):
            return None
        value = getattr(value, name)
    return value


def _decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    candidates: Iterable[Sequence[str]] = (
        ("llm", "model", "layers"),
        ("llm", "model", "model", "layers"),
        ("llm", "base_model", "model", "layers"),
        ("llm", "base_model", "model", "model", "layers"),
    )
    for path in candidates:
        layers = _resolve_attr(model, path)
        if isinstance(layers, (nn.ModuleList, list, tuple)) and len(layers):
            return layers
    raise AttributeError("Could not locate frozen-Qwen decoder layers")


def _llm_hidden_size(model: nn.Module) -> int:
    for path in (("llm", "config", "hidden_size"), ("llm", "config", "d_model")):
        value = _resolve_attr(model, path)
        if value is not None:
            return int(value)
    raise AttributeError("Could not determine Qwen hidden size")


def _selected_layer_indices(total: int, count: int) -> list[int]:
    count = max(1, min(count, total))
    indices = [round((slot + 1) * total / (count + 1)) - 1 for slot in range(count)]
    return sorted(set(max(0, min(total - 1, index)) for index in indices))


def attach_gated_cross_attention(
    model: nn.Module,
    *,
    rank: int,
    heads: int,
    layer_count: int,
    freeze_projector: bool,
) -> GatedCrossAttentionBank:
    if hasattr(model.projector, "saga_xattn"):
        return model.projector.saga_xattn

    layers = _decoder_layers(model)
    layer_indices = _selected_layer_indices(len(layers), layer_count)
    hidden_size = _llm_hidden_size(model)
    bank = GatedCrossAttentionBank(hidden_size, rank, heads, len(layer_indices))

    reference = next(model.llm.parameters())
    bank.to(device=reference.device, dtype=reference.dtype)
    model.projector.add_module("saga_xattn", bank)

    if freeze_projector:
        _freeze_except_xattn(model)

    model._saga_xattn_memory = None
    model._saga_xattn_layer_indices = layer_indices
    model._saga_xattn_handles = []

    def capture_memory(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        memory = _first_tensor(output)
        if memory is None:
            raise TypeError("Projector output contains no tensor")
        model._saga_xattn_memory = memory.detach() if freeze_projector else memory

    model._saga_xattn_handles.append(model.projector.register_forward_hook(capture_memory))

    for adapter_index, layer_index in enumerate(layer_indices):
        adapter = bank.adapters[adapter_index]

        def inject_memory(
            _module: nn.Module,
            _args: tuple[Any, ...],
            output: Any,
            adapter: GatedLowRankCrossAttention = adapter,
        ) -> Any:
            memory = model._saga_xattn_memory
            if memory is None:
                return output
            hidden = _first_tensor(output)
            if hidden is None:
                raise TypeError("Decoder-layer output contains no tensor")
            updated = adapter(hidden, memory)
            return _replace_first_tensor(output, updated)

        model._saga_xattn_handles.append(layers[layer_index].register_forward_hook(inject_memory))

    print(
        "[SAGA-XAttn] attached "
        f"layers={layer_indices} hidden={hidden_size} rank={rank} heads={heads} "
        f"freeze_projector={freeze_projector}",
        flush=True,
    )
    return bank


def install_runtime_patch() -> type[nn.Module]:
    # The project module is named ``sagav3_llm.py`` (without an underscore).
    import models_v3.sagav3_llm as saga_module

    current = saga_module.SingleCellLLMv3
    if getattr(current, "_saga_xattn_runtime_patch", False):
        return current

    enabled = _env_bool("SAGA_XATTN_ENABLED", True)
    rank = int(os.environ.get("SAGA_XATTN_RANK", "512"))
    heads = int(os.environ.get("SAGA_XATTN_HEADS", "8"))
    layer_count = int(os.environ.get("SAGA_XATTN_LAYERS", "4"))
    freeze_projector = _env_bool("SAGA_XATTN_FREEZE_PROJECTOR", True)
    log_every = int(os.environ.get("SAGA_XATTN_GATE_LOG_EVERY", "20"))

    class SingleCellLLMv3XAttn(current):  # type: ignore[misc, valid-type]
        _saga_xattn_runtime_patch = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._saga_xattn_forward_count = 0
            if enabled:
                attach_gated_cross_attention(
                    self,
                    rank=rank,
                    heads=heads,
                    layer_count=layer_count,
                    freeze_projector=freeze_projector,
                )

        def load_projector(self, *args: Any, **kwargs: Any) -> Any:
            # The Stage-I initialization checkpoint predates SAGA-XAttn.  Some
            # SAGA loaders use strict=True, so hide the new child only while
            # loading that legacy checkpoint.  Leave it attached for all new
            # checkpoints so adapter weights are restored normally.
            bank = None
            legacy_load = _env_bool("SAGA_XATTN_LEGACY_PROJECTOR_LOAD", False)
            if legacy_load and hasattr(self.projector, "saga_xattn"):
                bank = self.projector._modules.pop("saga_xattn")
            try:
                # Stage-I hierarchy checkpoints contain training-only
                # hierarchy_* heads.  Open-question inference builds the
                # reusable projector without those heads, while retaining the
                # XAttn child.  Filter only this known auxiliary namespace.
                if _env_bool("SAGA_XATTN_IGNORE_HIERARCHY_KEYS", False):
                    if not args:
                        raise TypeError("load_projector requires a checkpoint path")
                    payload = torch.load(args[0], map_location="cpu")
                    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
                    if not isinstance(state, dict):
                        raise TypeError("Projector checkpoint must contain a state dict")
                    state = {
                        key: value for key, value in state.items()
                        if not str(key).startswith("hierarchy_")
                    }
                    missing, unexpected = self.projector.load_state_dict(state, strict=True)
                    if missing or unexpected:
                        raise RuntimeError(
                            f"Projector load mismatch after hierarchy filtering: "
                            f"missing={missing}, unexpected={unexpected}"
                        )
                    result = None
                else:
                    result = super().load_projector(*args, **kwargs)
            finally:
                if bank is not None:
                    self.projector.add_module("saga_xattn", bank)
            if enabled and freeze_projector and hasattr(self.projector, "saga_xattn"):
                _freeze_except_xattn(self)
            return result

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            result = super().forward(*args, **kwargs)
            self._saga_xattn_forward_count += 1
            rank_id = int(os.environ.get("RANK", "0"))
            if (
                enabled
                and self.training
                and rank_id == 0
                and log_every > 0
                and self._saga_xattn_forward_count % log_every == 0
            ):
                values = self.projector.saga_xattn.gate_values()
                print(
                    f"[SAGA-XAttn gates forward={self._saga_xattn_forward_count}] "
                    + " ".join(f"g{index}={value:.6f}" for index, value in enumerate(values)),
                    flush=True,
                )
            return result

    SingleCellLLMv3XAttn.__name__ = "SingleCellLLMv3XAttn"
    saga_module.SingleCellLLMv3 = SingleCellLLMv3XAttn
    return SingleCellLLMv3XAttn
