#!/usr/bin/env python3
"""Aggregate matched raw-gene and CellSemantics results across Projector seeds.

The deterministic gene-text outputs are shared across Projector seeds.  The
primary uncertainty for CellSemantics therefore has two levels: held-out cells
within a seed and independently trained Projectors across seeds.  This script
reports seed-level metrics and uses a hierarchical bootstrap that resamples
Projector seeds before resampling cells within each selected seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ENDPOINTS = (
    ("broad_family", "family_correct", "gold_family"),
    ("mid_label_exact", "mid_label_match", "gold_label"),
)


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def load_answers(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"cell_id", "task", "gold_label", "gold_family", "family_correct"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame["cell_id"] = frame["cell_id"].astype(str)
    frame["family_correct"] = as_bool(frame["family_correct"])
    if "mid_label_match" in frame:
        frame["mid_label_match"] = as_bool(frame["mid_label_match"])
    elif "label_token_overlap" in frame:
        frame["mid_label_match"] = frame["label_token_overlap"].astype(float).ge(1.0 - 1e-12)
    else:
        raise ValueError(f"{path} has neither mid_label_match nor label_token_overlap")
    return frame


def parse_seed_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected TAG=DIR for --seed-run, got {spec!r}")
    tag, directory = spec.split("=", 1)
    if not tag.strip() or not directory.strip():
        raise ValueError(f"Invalid --seed-run value: {spec!r}")
    return tag.strip(), Path(directory)


def macro_accuracy(frame: pd.DataFrame, metric: str, group_key: str) -> float:
    if frame.empty:
        return float("nan")
    return float(frame.groupby(group_key, observed=True)[metric].mean().mean())


def assert_matched(reference: pd.DataFrame, candidate: pd.DataFrame, name: str) -> None:
    keys = ["cell_id", "task"]
    ref_keys = set(map(tuple, reference[keys].astype(str).to_numpy()))
    candidate_keys = set(map(tuple, candidate[keys].astype(str).to_numpy()))
    if ref_keys != candidate_keys:
        raise ValueError(
            f"{name} is not matched to the reference: "
            f"missing={len(ref_keys - candidate_keys)} extra={len(candidate_keys - ref_keys)}"
        )


def merge_pair(left: pd.DataFrame, right: pd.DataFrame, metric: str, group_key: str) -> pd.DataFrame:
    columns = ["cell_id", "task", group_key, metric]
    merged = left[columns].merge(
        right[["cell_id", "task", metric]],
        on=["cell_id", "task"],
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No matched rows for paired comparison")
    return merged


def macro_difference(frame: pd.DataFrame, metric: str, group_key: str) -> float:
    left = f"{metric}_left"
    right = f"{metric}_right"
    grouped = frame.groupby(group_key, observed=True)[[left, right]].mean()
    return float(grouped[left].mean() - grouped[right].mean())


def hierarchical_bootstrap(
    pairs_by_seed: dict[str, pd.DataFrame],
    metric: str,
    group_key: str,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    tags = sorted(pairs_by_seed)
    seed_points = [macro_difference(pairs_by_seed[tag], metric, group_key) for tag in tags]
    point = float(np.mean(seed_points))
    estimates = np.empty(n_boot, dtype=float)
    for boot_i in range(n_boot):
        sampled_tags = rng.choice(tags, size=len(tags), replace=True)
        sampled_seed_points = []
        for tag in sampled_tags:
            frame = pairs_by_seed[str(tag)]
            indices = rng.integers(0, len(frame), size=len(frame))
            sampled_seed_points.append(macro_difference(frame.iloc[indices], metric, group_key))
        estimates[boot_i] = float(np.mean(sampled_seed_points))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return point, float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--seed-run", action="append", required=True, help="TAG=DIR; repeat for each seed")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260917)
    args = parser.parse_args()

    pilot_root = Path(args.pilot_root)
    pilot_summary_path = pilot_root / "results" / "interface_summary.csv"
    if not pilot_summary_path.exists():
        raise FileNotFoundError(pilot_summary_path)
    pilot_summary = pd.read_csv(pilot_summary_path)

    gene_runs: dict[str, dict[str, pd.DataFrame]] = {}
    for path in sorted(pilot_root.glob("gene_*_*.answers.csv")):
        one = pd.read_csv(path, nrows=1)
        if one.empty:
            continue
        serializer = str(one.iloc[0]["serializer"])
        control = str(one.iloc[0]["control"])
        if control not in {"correct", "shuffle"}:
            continue
        gene_runs.setdefault(serializer, {})[control] = load_answers(path)
    if "specificity_weighted" not in gene_runs:
        raise ValueError("Missing specificity_weighted gene-text results")

    seed_runs: dict[str, dict[str, pd.DataFrame]] = {}
    checkpoints: dict[str, str] = {}
    for spec in args.seed_run:
        tag, directory = parse_seed_spec(spec)
        if tag in seed_runs:
            raise ValueError(f"Duplicate seed tag: {tag}")
        correct_path = directory / "cellsemantics_correct.answers.csv"
        shuffle_path = directory / "cellsemantics_shuffle.answers.csv"
        correct = load_answers(correct_path)
        shuffle = load_answers(shuffle_path)
        assert_matched(correct, shuffle, f"{tag} shuffle")
        seed_runs[tag] = {"correct": correct, "shuffle": shuffle}
        summary_path = directory / "cellsemantics_correct.summary.json"
        if summary_path.exists():
            checkpoints[tag] = str(json.loads(summary_path.read_text()).get("projector_ckpt", ""))

    tags = sorted(seed_runs)
    if len(tags) < 2:
        raise ValueError("At least two independently trained Projector seeds are required")
    reference = seed_runs[tags[0]]["correct"]
    for tag in tags[1:]:
        assert_matched(reference, seed_runs[tag]["correct"], f"{tag} correct")
    for serializer, controls in gene_runs.items():
        if {"correct", "shuffle"}.difference(controls):
            raise ValueError(f"Incomplete controls for gene serializer {serializer}")
        assert_matched(reference, controls["correct"], f"gene {serializer} correct")
        assert_matched(reference, controls["shuffle"], f"gene {serializer} shuffle")

    seed_metric_rows = []
    for tag, controls in seed_runs.items():
        for control, frame in controls.items():
            for task, task_frame in frame.groupby("task", sort=True):
                for endpoint, metric, group_key in ENDPOINTS:
                    seed_metric_rows.append({
                        "seed": tag,
                        "task": task,
                        "control": control,
                        "endpoint": endpoint,
                        "n_cells": int(task_frame["cell_id"].nunique()),
                        "n_groups": int(task_frame[group_key].nunique()),
                        "macro_accuracy": macro_accuracy(task_frame, metric, group_key),
                        "micro_accuracy": float(task_frame[metric].mean()),
                    })
    seed_metrics = pd.DataFrame(seed_metric_rows)

    aggregate = (
        seed_metrics.groupby(["task", "control", "endpoint"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            macro_accuracy_mean=("macro_accuracy", "mean"),
            macro_accuracy_std=("macro_accuracy", "std"),
            macro_accuracy_min=("macro_accuracy", "min"),
            macro_accuracy_max=("macro_accuracy", "max"),
            micro_accuracy_mean=("micro_accuracy", "mean"),
            micro_accuracy_std=("micro_accuracy", "std"),
        )
    )

    rng = np.random.default_rng(args.bootstrap_seed)
    grounding_rows = []
    comparison_rows = []
    tasks = sorted(reference["task"].unique())
    for task in tasks:
        for endpoint, metric, group_key in ENDPOINTS:
            grounding_pairs = {}
            for tag, controls in seed_runs.items():
                left = controls["correct"].loc[lambda x: x["task"].eq(task)]
                right = controls["shuffle"].loc[lambda x: x["task"].eq(task)]
                grounding_pairs[tag] = merge_pair(left, right, metric, group_key)
            point, low, high = hierarchical_bootstrap(
                grounding_pairs, metric, group_key, args.bootstrap, rng,
            )
            grounding_rows.append({
                "representation": "CellSemantics-32",
                "task": task,
                "endpoint": endpoint,
                "n_seeds": len(tags),
                "n_cells_per_seed": int(reference.loc[reference["task"].eq(task), "cell_id"].nunique()),
                "correct_minus_shuffle_macro_mean": point,
                "ci95_low": low,
                "ci95_high": high,
                "bootstrap_type": "hierarchical_seed_then_cell",
            })

            for serializer, controls in sorted(gene_runs.items()):
                comparison_pairs = {}
                gene = controls["correct"].loc[lambda x: x["task"].eq(task)]
                for tag, seed_control in seed_runs.items():
                    cellsem = seed_control["correct"].loc[lambda x: x["task"].eq(task)]
                    comparison_pairs[tag] = merge_pair(cellsem, gene, metric, group_key)
                point, low, high = hierarchical_bootstrap(
                    comparison_pairs, metric, group_key, args.bootstrap, rng,
                )
                comparison_rows.append({
                    "left": "CellSemantics-32",
                    "right": f"GeneText-{serializer}",
                    "task": task,
                    "endpoint": endpoint,
                    "n_seeds": len(tags),
                    "n_cells_per_seed": int(gene["cell_id"].nunique()),
                    "cellsem_minus_gene_macro_mean": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "bootstrap_type": "hierarchical_seed_then_cell",
                })

    grounding = pd.DataFrame(grounding_rows)
    comparisons = pd.DataFrame(comparison_rows)

    # Main-figure source data: deterministic interfaces retain the original
    # cell-bootstrap estimates; CellSemantics is replaced by the seed mean.
    performance_source = pilot_summary.copy()
    performance_source = performance_source[~performance_source["representation"].eq("CellSemantics-32")].copy()
    cellsem_correct = aggregate[aggregate["control"].eq("correct")].copy()
    task_meta = {
        "broad_family": ("macro_family_accuracy", "macro_family_accuracy"),
        "mid_label_exact": ("macro_mid_label_accuracy", "macro_mid_label_accuracy"),
    }
    cellsem_wide_rows = []
    for task in sorted(cellsem_correct["task"].unique()):
        row = {
            "representation": "CellSemantics-32",
            "task": task,
            "control": "correct",
            "n": int(reference.loc[reference["task"].eq(task), "cell_id"].nunique()),
            "n_families": int(reference.loc[reference["task"].eq(task), "gold_family"].nunique()),
            "mean_evidence_tokens": 32.0,
            "n_projector_seeds": len(tags),
        }
        for endpoint, (output_col, _) in task_meta.items():
            value = cellsem_correct[
                cellsem_correct["task"].eq(task) & cellsem_correct["endpoint"].eq(endpoint)
            ].iloc[0]
            row[output_col] = value["macro_accuracy_mean"]
            row[f"{output_col}_seed_std"] = value["macro_accuracy_std"]
        micro_broad = cellsem_correct[
            cellsem_correct["task"].eq(task) & cellsem_correct["endpoint"].eq("broad_family")
        ].iloc[0]
        micro_mid = cellsem_correct[
            cellsem_correct["task"].eq(task) & cellsem_correct["endpoint"].eq("mid_label_exact")
        ].iloc[0]
        row["micro_accuracy"] = micro_broad["micro_accuracy_mean"]
        row["mid_label_accuracy"] = micro_mid["micro_accuracy_mean"]
        cellsem_wide_rows.append(row)
    performance_source = pd.concat(
        [performance_source, pd.DataFrame(cellsem_wide_rows)], ignore_index=True, sort=False,
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_metrics.to_csv(out / "cellsemantics_seed_level_metrics.csv", index=False)
    aggregate.to_csv(out / "cellsemantics_multiseed_summary.csv", index=False)
    grounding.to_csv(out / "cellsemantics_grounding_hierarchical_bootstrap.csv", index=False)
    comparisons.to_csv(out / "cellsemantics_vs_gene_hierarchical_bootstrap.csv", index=False)
    performance_source.to_csv(out / "main_figure_performance_source_data.csv", index=False)
    (out / "audit.json").write_text(json.dumps({
        "pilot_root": str(pilot_root),
        "projector_seeds": tags,
        "projector_checkpoints": checkpoints,
        "n_projector_seeds": len(tags),
        "n_matched_cells": int(reference["cell_id"].nunique()),
        "tasks": tasks,
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_type": "hierarchical_seed_then_cell",
        "note": "Gene-text outputs are deterministic shared comparators; Projector seeds and cells are resampled hierarchically.",
    }, indent=2) + "\n")

    print("== CellSemantics seed-level metrics ==")
    print(seed_metrics.sort_values(["endpoint", "task", "control", "seed"]).to_string(index=False))
    print("\n== CellSemantics seed aggregate ==")
    print(aggregate.sort_values(["endpoint", "task", "control"]).to_string(index=False))
    print("\n== CellSemantics grounding, hierarchical bootstrap ==")
    print(grounding.to_string(index=False))
    print("\n== CellSemantics versus gene text, hierarchical bootstrap ==")
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
