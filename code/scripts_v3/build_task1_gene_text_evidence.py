#!/usr/bin/env python3
"""Build label-free gene-text evidence for matched Task1 cells.

The sampled cells come from an existing CellSemantics answer CSV, ensuring that
continuous-token and gene-text interfaces are evaluated on identical cells.
Gene ranking uses only expression values.  TF-IDF document frequencies are
estimated from training-split cells and never use cell-type labels.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    raise ValueError(f"Unsupported records JSON: {path}")


def get_value(record: dict[str, Any], *keys: str) -> str:
    answer = record.get("answer") if isinstance(record.get("answer"), dict) else {}
    for key in keys:
        value = record.get(key, answer.get(key))
        if value is not None and str(value).strip() and str(value).lower() != "nan":
            return str(value).strip()
    return ""


def raw_cell_id(record: dict[str, Any]) -> str:
    value = get_value(record, "raw_cell_id", "obs_name", "barcode", "cell_id", "id")
    if "::" in value:
        value = value.split("::", 1)[1]
    return value


def tissue_and_obs_id(record: dict[str, Any]) -> tuple[str, str]:
    tissue = get_value(record, "tissue")
    value = raw_cell_id(record)
    if ":" in value:
        prefix, obs_id = value.split(":", 1)
        if not tissue:
            tissue = prefix
        return tissue, obs_id
    return tissue, value


def tissue_from_path(path: Path) -> str:
    if path.parent.name == "staged":
        return path.parent.parent.name
    name = path.stem
    for suffix in ("_uce_input", ".uce"):
        name = name.removesuffix(suffix)
    return name


def dense_row(matrix: Any, index: int) -> np.ndarray:
    row = matrix[index]
    if hasattr(row, "toarray"):
        row = row.toarray()
    return np.asarray(row).reshape(-1).astype(np.float64, copy=False)


def detection_frequency(matrix: Any, indices: list[int], n_genes: int, batch_size: int) -> np.ndarray:
    counts = np.zeros(n_genes, dtype=np.int64)
    for start in range(0, len(indices), batch_size):
        batch = matrix[indices[start:start + batch_size]]
        if hasattr(batch, "toarray"):
            batch = batch.toarray()
        values = np.asarray(batch)
        counts += np.count_nonzero(values > 0, axis=0)
        print(f"[idf] {min(start + batch_size, len(indices))}/{len(indices)}", flush=True)
    return counts


def top_indices(scores: np.ndarray, values: np.ndarray, k: int) -> np.ndarray:
    valid = np.flatnonzero(np.isfinite(scores) & (values > 0))
    if len(valid) <= k:
        return valid[np.argsort(scores[valid])[::-1]]
    chosen = valid[np.argpartition(scores[valid], -k)[-k:]]
    return chosen[np.argsort(scores[chosen])[::-1]]


def serialize(values: np.ndarray, genes: np.ndarray, idf: np.ndarray, k: int) -> dict[str, str]:
    expression_score = values
    specificity_score = np.log1p(np.maximum(values, 0)) * idf
    expression_idx = top_indices(expression_score, values, k)
    specificity_idx = top_indices(specificity_score, values, k)

    expressed = [str(genes[index]) for index in expression_idx]
    specific = [str(genes[index]) for index in specificity_idx]
    selected_values = values[expression_idx]
    if len(selected_values) >= 3:
        q1, q2 = np.quantile(selected_values, [1 / 3, 2 / 3])
    elif len(selected_values):
        q1 = q2 = float(np.median(selected_values))
    else:
        q1 = q2 = 0.0

    def level(value: float) -> str:
        if value <= q1:
            return "low"
        if value <= q2:
            return "medium"
        return "high"

    binned = [f"{genes[index]}:{level(float(values[index]))}" for index in expression_idx]
    return {
        "top_expressed": ", ".join(expressed),
        "specificity_weighted": ", ".join(specific),
        "expression_binned": ", ".join(binned),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-records", required=True)
    parser.add_argument("--train-records", required=True)
    parser.add_argument("--sample-answers", required=True)
    parser.add_argument("--h5ad", action="append", default=[])
    parser.add_argument("--h5ad-glob", default="")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--source-filter", default="TS")
    parser.add_argument("--obs-id-key", default="")
    parser.add_argument("--layer", default="X")
    parser.add_argument("--label-key", default="mid_cell_type")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--max-idf-cells", type=int, default=50000)
    parser.add_argument("--idf-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-matched", type=int, default=50)
    args = parser.parse_args()

    import anndata as ad

    rng = random.Random(args.seed)
    test_records = read_records(Path(args.test_records))
    train_records = read_records(Path(args.train_records))
    answers = pd.read_csv(args.sample_answers)
    if "cell_id" not in answers:
        raise ValueError("sample answer CSV must contain cell_id")
    sample_ids = list(dict.fromkeys(answers["cell_id"].astype(str)))

    by_cell_id = {get_value(record, "cell_id", "id"): record for record in test_records}
    selected_records = []
    for cell_id in sample_ids:
        record = by_cell_id.get(cell_id)
        if record is None:
            continue
        source = get_value(record, "source", "source_dataset", "dataset")
        if args.source_filter and source != args.source_filter:
            continue
        selected_records.append(record)

    h5ad_paths = [Path(path) for path in args.h5ad]
    if args.h5ad_glob:
        h5ad_paths.extend(Path(path) for path in sorted(glob.glob(args.h5ad_glob)))
    h5ad_paths = list(dict.fromkeys(path.resolve() for path in h5ad_paths))
    if not h5ad_paths:
        raise ValueError("Provide at least one --h5ad or a matching --h5ad-glob")

    shards: dict[str, dict[str, Any]] = {}
    genes = None
    for path in h5ad_paths:
        tissue = tissue_from_path(path)
        if tissue in shards:
            raise ValueError(f"Duplicate H5AD tissue key {tissue}: {path}")
        adata = ad.read_h5ad(path, backed="r")
        shard_genes = np.asarray(adata.var_names.astype(str))
        if genes is None:
            genes = shard_genes
        elif not np.array_equal(genes, shard_genes):
            raise ValueError(f"Gene order mismatch in {path}")
        obs_ids = (
            adata.obs[args.obs_id_key].astype(str).tolist()
            if args.obs_id_key else adata.obs_names.astype(str).tolist()
        )
        matrix = adata.X if args.layer == "X" else adata.layers[args.layer]
        shards[tissue] = {
            "path": str(path),
            "adata": adata,
            "matrix": matrix,
            "obs_to_row": {cell_id: index for index, cell_id in enumerate(obs_ids)},
        }
        print(f"[shard] tissue={tissue} n={adata.n_obs} genes={adata.n_vars} path={path}", flush=True)
    assert genes is not None

    matched = []
    for record in selected_records:
        tissue, obs_id = tissue_and_obs_id(record)
        shard = shards.get(tissue)
        index = None if shard is None else shard["obs_to_row"].get(obs_id)
        if index is not None:
            matched.append((record, tissue, index))
    if len(matched) < args.min_matched:
        examples = [tissue_and_obs_id(record) for record in selected_records[:10]]
        raise RuntimeError(
            f"Only {len(matched)} sampled cells matched H5AD; minimum={args.min_matched}. "
            f"Check tissue names and --obs-id-key. Example requested IDs: {examples}"
        )

    train_locations = []
    for record in train_records:
        source = get_value(record, "source", "source_dataset", "dataset")
        if args.source_filter and source != args.source_filter:
            continue
        tissue, obs_id = tissue_and_obs_id(record)
        shard = shards.get(tissue)
        index = None if shard is None else shard["obs_to_row"].get(obs_id)
        if index is not None:
            train_locations.append((tissue, index))
    train_locations = list(dict.fromkeys(train_locations))
    if len(train_locations) > args.max_idf_cells:
        train_locations = rng.sample(train_locations, args.max_idf_cells)
    if not train_locations:
        raise RuntimeError("No training-split cells matched H5AD; TF-IDF cannot be estimated without test leakage")

    locations_by_tissue: dict[str, list[int]] = {}
    for tissue, index in train_locations:
        locations_by_tissue.setdefault(tissue, []).append(index)
    df = np.zeros(len(genes), dtype=np.int64)
    for tissue, indices in sorted(locations_by_tissue.items()):
        indices = sorted(indices)
        print(f"[idf-shard] tissue={tissue} n={len(indices)}", flush=True)
        df += detection_frequency(
            shards[tissue]["matrix"], indices, len(genes), args.idf_batch_size,
        )
    idf = np.log((1 + len(train_locations)) / (1 + df)) + 1.0

    rows = []
    for record, tissue, index in matched:
        values = dense_row(shards[tissue]["matrix"], index)
        evidence = serialize(values, genes, idf, args.top_k)
        rows.append({
            "cell_id": get_value(record, "cell_id", "id"),
            "raw_cell_id": raw_cell_id(record),
            "source": get_value(record, "source", "source_dataset", "dataset"),
            "tissue": tissue,
            "gold_label": get_value(record, args.label_key, "cell_type"),
            "h5ad_row": int(index),
            "evidence": evidence,
        })
    order = list(range(len(rows)))
    shuffled = order.copy()
    while len(shuffled) > 1 and any(a == b for a, b in zip(order, shuffled)):
        rng.shuffle(shuffled)
    for row, shuffled_index in zip(rows, shuffled):
        row["shuffle_cell_id"] = rows[shuffled_index]["cell_id"]

    output = {
        "schema": "task1_gene_text_evidence_v1",
        "h5ad_shards": {tissue: shard["path"] for tissue, shard in sorted(shards.items())},
        "source_filter": args.source_filter,
        "layer": args.layer,
        "obs_id_key": args.obs_id_key or "obs_names",
        "top_k": args.top_k,
        "idf_train_cells": len(train_locations),
        "n_sample_answer_cells": len(sample_ids),
        "n_source_filtered_cells": len(selected_records),
        "n_matched_cells": len(rows),
        "label_free": True,
        "rows": rows,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "rows"}, indent=2))
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
