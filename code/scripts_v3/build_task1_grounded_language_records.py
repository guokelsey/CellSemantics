#!/usr/bin/env python3
"""Build held-in Task1 language-grounding records from a training split.

The records reproduce the three open-question formats used at evaluation while
using only training-split cells.  They are preformatted ExtraTaskDatasetV3
examples, each pointing back to the existing UCE cache by source index.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch


FAMILIES = (
    "Immune",
    "Epithelial",
    "Stromal/muscle",
    "Endothelial/vascular",
    "Hematopoietic progenitor",
    "Pancreas/exocrine",
    "Other",
)


def records_from(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [row for row in payload["records"] if isinstance(row, dict)]
    raise ValueError(f"Unsupported record payload: {path}")


def label(record: dict[str, Any], key: str) -> str:
    answer = record.get("answer") if isinstance(record.get("answer"), dict) else {}
    for source in (record, answer):
        value = source.get(key)
        if value is not None and str(value).strip() and str(value).lower() != "nan":
            return str(value).strip()
    return ""


def family_for(value: str) -> str:
    text = value.lower().replace("_", " ")
    groups = {
        "Hematopoietic progenitor": ("hematopoietic", "progenitor", "stem cell", "erythroid", "erythroblast", "megakaryocyte"),
        "Immune": ("t cell", "b cell", "nk", "nkt", "macrophage", "monocyte", "dendritic", "plasma", "immune", "mast", "basophil", "neutrophil", "lymph"),
        "Epithelial": ("epithelial", "basal", "luminal", "club", "secretory", "goblet", "keratinocyte", "ciliated", "ductal"),
        "Stromal/muscle": ("fibroblast", "stromal", "smooth muscle", "myocyte", "pericyte", "mesenchymal", "stellate", "muscle"),
        "Endothelial/vascular": ("endothelial", "vascular", "capillary", "artery", "venous", "vein", "lymphatic"),
        "Pancreas/exocrine": ("acinar", "alpha", "beta", "delta", "gamma", "endocrine", "exocrine", "pancrea"),
    }
    for family, terms in groups.items():
        if any(term in text for term in terms):
            return family
    return "Other"


def prompt(task: str, tokens: int, wrong_family: str | None = None) -> str:
    cells = " ".join(["<CELL>"] * tokens)
    prefix = (
        "<|im_start|>system\nYou are a single-cell biology expert. "
        "Use the cell embedding tokens as the primary evidence.<|im_end|>\n"
        "<|im_start|>user\n"
        f"Cell embedding tokens: {cells}\n\n"
    )
    if task == "lineage_summary":
        instruction = (
            "Describe this single cell in natural language. Give a broad biological lineage or family, "
            "a likely cell identity, and one short rationale. Do not use a candidate list.\n\n"
            "Reply in three short fields:\nLineage: ...\nIdentity: ...\nRationale: ..."
        )
    elif task == "identity_rationale":
        instruction = (
            "Infer the likely biological identity of this cell from the cell embedding tokens. "
            "Use a short open-ended answer and include one biological reason.\n\n"
            "Reply in two short fields:\nIdentity: ...\nReason: ..."
        )
    elif task == "conflict_rationale":
        instruction = (
            f"A noisy upstream note says this cell is from the {wrong_family} family. "
            "This note may be wrong. Ignore it if the cell embedding disagrees. "
            "Describe the broad biological lineage, likely identity, and one reason.\n\n"
            "Reply in three short fields:\nLineage: ...\nIdentity: ...\nRationale: ..."
        )
    else:
        raise ValueError(task)
    return prefix + instruction + "<|im_end|>\n<|im_start|>assistant\n"


def target(task: str, family: str, identity: str) -> str:
    reason = f"The cell-state representation is consistent with {family.lower()} biology."
    if task == "identity_rationale":
        return f"Identity: {identity}\nReason: {reason}"
    return f"Lineage: {family}\nIdentity: {identity}\nRationale: {reason}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--embedding-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--label-key", default="mid_cell_type")
    parser.add_argument("--num-cell-tokens", type=int, default=32)
    parser.add_argument("--max-cells", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    source = records_from(Path(args.records))
    cache = torch.load(args.embedding_cache, map_location="cpu", weights_only=True)
    ids = [str(cell_id) for cell_id in cache["cell_ids"]]
    id_to_index = {cell_id: index for index, cell_id in enumerate(ids)}

    usable: list[dict[str, Any]] = []
    for record in source:
        cell_id = str(record.get("cell_id") or record.get("id") or "")
        identity = label(record, args.label_key) or label(record, "cell_type")
        source_index = record.get("source_index")
        if source_index is None:
            source_index = id_to_index.get(cell_id)
        if not cell_id or not identity or source_index is None:
            continue
        index = int(source_index)
        if index < 0 or index >= len(ids):
            continue
        item = dict(record)
        item["cell_id"] = cell_id
        item["source_index"] = index
        item["identity"] = identity
        item["family"] = family_for(identity)
        usable.append(item)
    if len(usable) > args.max_cells:
        usable = rng.sample(usable, args.max_cells)
    if not usable:
        raise SystemExit("No training records could be matched to the UCE cache")

    tasks = ("lineage_summary", "identity_rationale", "conflict_rationale")
    output: list[dict[str, Any]] = []
    for item in usable:
        for task_name in tasks:
            family = item["family"]
            wrong = rng.choice([candidate for candidate in FAMILIES if candidate != family])
            output.append({
                "task_type": "preformatted",
                "mixture_task": "task1",
                "source_dataset": "task1_grounded_language",
                "cell_id": item["cell_id"],
                "source_index": item["source_index"],
                "embedding_cache_path": str(Path(args.embedding_cache).resolve()),
                "prompt": prompt(task_name, args.num_cell_tokens, wrong),
                "target": target(task_name, family, item["identity"]),
                "grounding_task": task_name,
                "gold_family": family,
                "gold_identity": item["identity"],
                "wrong_family": wrong if task_name == "conflict_rationale" else "",
            })
    rng.shuffle(output)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output) + "\n")
    summary = {
        "records": str(Path(args.records).resolve()),
        "embedding_cache": str(Path(args.embedding_cache).resolve()),
        "n_input": len(source),
        "n_cells": len(usable),
        "n_records": len(output),
        "task_counts": dict(Counter(row["grounding_task"] for row in output)),
        "family_counts": dict(Counter(row["gold_family"] for row in output)),
        "seed": args.seed,
        "train_only": True,
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
