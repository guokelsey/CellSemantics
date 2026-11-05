#!/usr/bin/env python3
"""Extract pooled Projector features with explicit architecture validation."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models_v3.uce_projector import UCEProjectorV3


def load_state(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and isinstance(obj.get("projector"), dict):
        obj = obj["projector"]
    elif isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(obj)}")
    clean = {}
    for key, value in obj.items():
        if not torch.is_tensor(value):
            continue
        while key.startswith(("module.", "projector.")):
            key = key.split(".", 1)[1]
        clean[key] = value
    return clean


def mixer_layer_indices(state: dict[str, torch.Tensor]) -> list[int]:
    indices = set()
    patterns = (
        r"(?:^|\.)(?:token_mixer|mixer|transformer)\.layers\.(\d+)(?:\.|$)",
        r"(?:^|\.)mixer_layers\.(\d+)(?:\.|$)",
    )
    for key in state:
        for pattern in patterns:
            match = re.search(pattern, key)
            if match:
                indices.add(int(match.group(1)))
    return sorted(indices)


def checkpoint_audit(path: Path, state: dict[str, torch.Tensor]) -> dict[str, Any]:
    indices = mixer_layer_indices(state)
    mixer_keys = sorted(key for key in state if "mixer" in key.lower() or "transformer" in key.lower())
    return {
        "checkpoint": str(path),
        "n_tensors": len(state),
        "mixer_layer_indices": indices,
        "detected_num_mixer_layers": max(indices) + 1 if indices else 0,
        "n_mixer_tensors": len(mixer_keys),
        "mixer_key_examples": mixer_keys[:12],
    }


def load_cache(path: Path) -> tuple[torch.Tensor, list[str] | None]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        features = obj.get("embeddings", obj.get("features"))
        cell_ids = obj.get("cell_ids", obj.get("ids", obj.get("cell_id")))
    else:
        features, cell_ids = obj, None
    if features is None or not torch.is_tensor(features):
        raise ValueError(f"No tensor embeddings/features in {path}")
    ids = [str(value) for value in cell_ids] if cell_ids is not None else None
    return features, ids


def optional_parameter(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in (
        "classifier", "_head", "aux", "hierarchy", "reconstruct", "decoder",
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projector-ckpt", required=True)
    parser.add_argument("--uce-cache")
    parser.add_argument("--out-features")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--audit-json")
    parser.add_argument("--input-dim", type=int, default=1280)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--num-cell-tokens", type=int, default=32)
    parser.add_argument("--output-dim", type=int, default=5120)
    parser.add_argument("--num-mixer-layers", type=int, default=0)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = Path(args.projector_ckpt)
    state = load_state(checkpoint)
    audit = checkpoint_audit(checkpoint, state)
    audit["requested_num_mixer_layers"] = args.num_mixer_layers
    audit["architecture_match"] = audit["detected_num_mixer_layers"] == args.num_mixer_layers
    if args.audit_json:
        Path(args.audit_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.audit_json).write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)
    if not audit["architecture_match"]:
        raise SystemExit(
            f"Checkpoint has {audit['detected_num_mixer_layers']} detected mixer layers; "
            f"requested {args.num_mixer_layers}"
        )
    if args.inspect_only:
        return
    if not args.uce_cache or not args.out_features:
        raise SystemExit("--uce-cache and --out-features are required unless --inspect-only is used")

    device = torch.device(args.device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    projector = UCEProjectorV3(
        uce_dim=args.input_dim,
        hidden=args.hidden,
        num_tokens=args.num_cell_tokens,
        output_dim=args.output_dim,
        num_mixer_layers=args.num_mixer_layers,
        num_heads=args.num_heads,
        dropout=0.0,
    )
    current = projector.state_dict()
    compatible = {
        key: value for key, value in state.items()
        if key in current and tuple(value.shape) == tuple(current[key].shape)
    }
    missing_interface = sorted(
        key for key in current if key not in compatible and not optional_parameter(key)
    )
    if missing_interface:
        raise RuntimeError(
            "Checkpoint is not compatible with the requested Projector interface; "
            f"missing interface tensors: {missing_interface[:20]}"
        )
    missing, unexpected = projector.load_state_dict(compatible, strict=False)
    print(
        f"[load-projector] compatible={len(compatible)} missing={len(missing)} "
        f"checkpoint_only={len(set(state) - set(compatible))}",
        flush=True,
    )
    projector.to(device=device, dtype=dtype).eval()

    features, cell_ids = load_cache(Path(args.uce_cache))
    chunks = []
    with torch.no_grad():
        for start in range(0, len(features), args.batch_size):
            stop = min(len(features), start + args.batch_size)
            batch = features[start:stop].to(device=device, dtype=dtype)
            projected = projector(batch)
            if isinstance(projected, tuple):
                projected = projected[0]
            if projected.ndim == 3:
                projected = projected.mean(dim=1)
            if projected.ndim != 2 or projected.shape[1] != args.output_dim:
                raise RuntimeError(f"Unexpected Projector output shape: {tuple(projected.shape)}")
            chunks.append(projected.float().cpu())
            if start == 0 or stop == len(features) or stop % 6400 == 0:
                print(f"[project] {stop}/{len(features)}", flush=True)
    output = torch.cat(chunks, dim=0)
    payload: dict[str, Any] = {
        "features": output,
        "metadata": {
            "projector_ckpt": str(checkpoint),
            "num_mixer_layers": args.num_mixer_layers,
            "pooling": "mean_over_cell_tokens",
            "dtype": "float32",
        },
    }
    if cell_ids is not None:
        payload["cell_ids"] = cell_ids
    out_path = Path(args.out_features)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"[save] {out_path} shape={tuple(output.shape)}", flush=True)


if __name__ == "__main__":
    main()
