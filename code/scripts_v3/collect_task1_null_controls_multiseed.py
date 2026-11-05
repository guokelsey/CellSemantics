#!/usr/bin/env python3
"""Collect matched five-seed zero and deterministic text-only controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


TASKS = ("conflict_rationale", "identity_rationale", "lineage_summary")
ENDPOINTS = (
    ("broad_family", "family_correct", "gold_family"),
    ("mid_label_exact", "mid_label_match", "gold_label"),
)
FIELD_ALIASES = {
    "broad class": "broad_class",
    "revised class": "broad_class",
    "lineage": "broad_class",
    "best match": "identity",
    "likely identity": "identity",
    "identity": "identity",
    "evidence": "rationale",
    "supporting feature": "rationale",
    "rationale": "rationale",
    "reason": "rationale",
}
EXPECTED_FIELDS = {
    "lineage_summary": {"broad_class", "identity", "rationale"},
    "identity_rationale": {"identity", "rationale"},
    "conflict_rationale": {"broad_class", "identity", "rationale"},
}
CELL_LINE = re.compile(r"Cell embedding tokens:\s*(?:<CELL>\s*)+\n\s*\n", re.MULTILINE)


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes", "t"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_fields(answer: object) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in str(answer).splitlines():
        match = re.match(r"^\s*([A-Za-z][A-Za-z +/_-]{1,40})\s*:\s*(.*?)\s*$", line)
        if not match or not match.group(2):
            continue
        name = re.sub(r"\s+", " ", match.group(1).strip().lower())
        canonical = FIELD_ALIASES.get(name)
        if canonical and canonical not in fields:
            fields[canonical] = match.group(2).strip()
    return fields


def semantic_prompt(prompt: object) -> str:
    return CELL_LINE.sub("", str(prompt))


def load_answers(path: Path, expected_control: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "cell_id", "task", "gold_label", "gold_family", "control", "prompt",
        "prompt_style",
        "prompt_sha256", "pred_family", "family_correct", "answer",
        "wrong_family", "wrong_family_mentioned",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    if set(frame["control"].astype(str)) != {expected_control}:
        raise ValueError(f"{path} control values do not equal {expected_control!r}")
    if len(frame) != 500 * len(TASKS):
        raise ValueError(f"{path} rows={len(frame)} expected={500 * len(TASKS)}")
    if set(frame["task"].astype(str)) != set(TASKS):
        raise ValueError(f"{path} has unexpected tasks")
    if set(frame["prompt_style"].astype(str)) != {"paraphrase_v1"}:
        raise ValueError(f"{path} is not exclusively paraphrase_v1")
    frame["cell_id"] = frame["cell_id"].astype(str)
    frame["family_correct"] = as_bool(frame["family_correct"])
    frame["wrong_family_mentioned"] = as_bool(frame["wrong_family_mentioned"])
    if "mid_label_match" in frame:
        frame["mid_label_match"] = as_bool(frame["mid_label_match"])
    elif "label_token_overlap" in frame:
        frame["mid_label_match"] = frame["label_token_overlap"].astype(float).ge(1.0 - 1e-12)
    else:
        raise ValueError(f"{path} lacks mid-label scoring")
    parsed = frame["answer"].map(parse_fields)
    frame["answer_nonempty"] = frame["answer"].fillna("").astype(str).str.strip().ne("")
    frame["structured_parse_success"] = parsed.map(bool)
    frame["all_expected_fields"] = [
        EXPECTED_FIELDS[str(task)].issubset(set(fields))
        for task, fields in zip(frame["task"], parsed)
    ]
    normalized_pred = frame["pred_family"].fillna("").astype(str).str.strip().str.lower()
    frame["pred_family_resolved"] = ~normalized_pred.isin({"", "unknown", "nan"})
    frame["semantic_prompt"] = frame["prompt"].map(semantic_prompt)
    return frame


def indexed(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["cell_id", "task"]
    if frame.duplicated(keys).any():
        raise ValueError("duplicate cell_id/task rows")
    return frame.set_index(keys).sort_index()


def assert_matched(reference: pd.DataFrame, candidate: pd.DataFrame, name: str) -> None:
    ref = indexed(reference)
    other = indexed(candidate)
    if not ref.index.equals(other.index):
        raise ValueError(f"{name}: cell/task keys differ")
    for column in ("gold_label", "gold_family", "wrong_family"):
        if not ref[column].astype(str).equals(other[column].astype(str)):
            raise ValueError(f"{name}: {column} differs")


def assert_projector_prompt_match(reference: pd.DataFrame, candidate: pd.DataFrame, name: str) -> None:
    assert_matched(reference, candidate, name)
    ref = indexed(reference)
    other = indexed(candidate)
    if not ref["prompt_sha256"].astype(str).equals(other["prompt_sha256"].astype(str)):
        raise ValueError(f"{name}: exact prompts differ")


def assert_absent_prompt_match(reference: pd.DataFrame, absent: pd.DataFrame) -> None:
    assert_matched(reference, absent, "absent")
    ref = indexed(reference)
    other = indexed(absent)
    if not ref["semantic_prompt"].equals(other["semantic_prompt"]):
        raise ValueError("absent: prompts differ after removing only the cell-token line")


def macro_accuracy(frame: pd.DataFrame, metric: str, group_key: str) -> float:
    return float(frame.groupby(group_key, observed=True)[metric].mean().mean())


def merge_pair(left: pd.DataFrame, right: pd.DataFrame, metric: str, group_key: str) -> pd.DataFrame:
    columns = ["cell_id", "task", group_key, metric]
    return left[columns].merge(
        right[["cell_id", "task", metric]],
        on=["cell_id", "task"], suffixes=("_left", "_right"), validate="one_to_one",
    )


def macro_difference(frame: pd.DataFrame, metric: str, group_key: str) -> float:
    grouped = frame.groupby(group_key, observed=True)[
        [f"{metric}_left", f"{metric}_right"]
    ].mean()
    return float(grouped[f"{metric}_left"].mean() - grouped[f"{metric}_right"].mean())


def hierarchical_bootstrap(
    pairs_by_seed: dict[str, pd.DataFrame],
    metric: str,
    group_key: str,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    tags = sorted(pairs_by_seed)
    group_deltas: list[list[np.ndarray]] = []
    points = []
    for tag in tags:
        frame = pairs_by_seed[tag]
        delta_name = f"{metric}_left"
        right_name = f"{metric}_right"
        per_label = []
        for _, group in frame.groupby(group_key, sort=False, observed=True):
            delta = (
                group[delta_name].to_numpy(dtype=float)
                - group[right_name].to_numpy(dtype=float)
            )
            per_label.append(delta)
        group_deltas.append(per_label)
        points.append(float(np.mean([delta.mean() for delta in per_label])))
    point = float(np.mean(points))

    # This is a hierarchical bootstrap: resample Projectors and then cells
    # within each fixed gold-label stratum. Chunking keeps the 5,000-replicate
    # calculation fast without changing the estimator.
    chunk_size = 128
    estimates = np.empty(n_boot, dtype=float)
    for start in range(0, n_boot, chunk_size):
        stop = min(start + chunk_size, n_boot)
        size = stop - start
        per_seed = np.empty((len(tags), size), dtype=float)
        for seed_i, per_label in enumerate(group_deltas):
            means = np.zeros(size, dtype=float)
            for delta in per_label:
                indices = rng.integers(0, len(delta), size=(size, len(delta)))
                means += delta[indices].mean(axis=1)
            per_seed[seed_i] = means / len(per_label)
        sampled_seed_indices = rng.integers(0, len(tags), size=(size, len(tags)))
        estimates[start:stop] = per_seed.T[
            np.arange(size)[:, None], sampled_seed_indices
        ].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return point, float(low), float(high)


def parse_seed_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected TAG=DIR, got {spec!r}")
    tag, directory = spec.split("=", 1)
    return tag.strip(), Path(directory)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-run", action="append", required=True, help="TAG=existing correct/shuffle directory")
    parser.add_argument("--null-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260924)
    parser.add_argument("--inference-dtype", required=True)
    parser.add_argument("--reference-dtype", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    null_root = Path(args.null_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict[str, pd.DataFrame]] = {}
    input_files: list[dict[str, object]] = []
    for spec in args.seed_run:
        tag, directory = parse_seed_spec(spec)
        controls = {}
        paths: dict[str, Path] = {
            "correct": directory / "cellsemantics_correct.answers.csv",
            "shuffle": directory / "cellsemantics_shuffle.answers.csv",
        }
        if not args.preflight_only:
            paths["zero"] = null_root / "seeds" / f"seed{tag}" / "cellsemantics_zero.answers.csv"
        for control, path in paths.items():
            controls[control] = load_answers(path, control)
            input_files.append({"path": str(path), "sha256": sha256_file(path), "rows": len(controls[control])})
        assert_projector_prompt_match(controls["correct"], controls["shuffle"], f"seed{tag} shuffle")
        if "zero" in controls:
            assert_projector_prompt_match(controls["correct"], controls["zero"], f"seed{tag} zero")
        runs[tag] = controls

    tags = sorted(runs)
    reference = runs[tags[0]]["correct"]
    for tag in tags[1:]:
        assert_projector_prompt_match(reference, runs[tag]["correct"], f"seed{tag} correct")
    if args.preflight_only:
        payload = {
            "status": "PASS",
            "mode": "preflight",
            "projector_seeds": tags,
            "n_projector_seeds": len(tags),
            "n_cells": 500,
            "tasks": list(TASKS),
            "controls_checked": ["correct", "shuffle"],
            "prompt_style": "paraphrase_v1",
            "exact_prompt_match_within_and_across_seeds": True,
            "cell_task_label_wrong_family_match": True,
            "reference_correct_shuffle_dtype": args.reference_dtype,
            "inputs": input_files,
        }
        path = out_dir / "preflight_audit.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        print(f"[save] {path}")
        return

    absent_path = null_root / "deterministic" / "cellsemantics_absent.answers.csv"
    absent = load_answers(absent_path, "absent")
    input_files.append({"path": str(absent_path), "sha256": sha256_file(absent_path), "rows": len(absent)})
    assert_absent_prompt_match(reference, absent)

    seed_metric_rows = []
    for tag, controls in runs.items():
        for control, frame in controls.items():
            for task, task_frame in frame.groupby("task", sort=True):
                for endpoint, metric, group_key in ENDPOINTS:
                    seed_metric_rows.append({
                        "seed": tag, "task": task, "control": control, "endpoint": endpoint,
                        "n_cells": int(task_frame["cell_id"].nunique()),
                        "n_groups": int(task_frame[group_key].nunique()),
                        "macro_accuracy": macro_accuracy(task_frame, metric, group_key),
                        "micro_accuracy": float(task_frame[metric].mean()),
                    })
    seed_metrics = pd.DataFrame(seed_metric_rows)
    aggregate = (
        seed_metrics.groupby(["task", "control", "endpoint"], as_index=False)
        .agg(
            n_projector_seeds=("seed", "nunique"),
            macro_accuracy_mean=("macro_accuracy", "mean"),
            macro_accuracy_std=("macro_accuracy", "std"),
            macro_accuracy_min=("macro_accuracy", "min"),
            macro_accuracy_max=("macro_accuracy", "max"),
            micro_accuracy_mean=("micro_accuracy", "mean"),
            micro_accuracy_std=("micro_accuracy", "std"),
        )
    )

    absent_rows = []
    for task, task_frame in absent.groupby("task", sort=True):
        for endpoint, metric, group_key in ENDPOINTS:
            absent_rows.append({
                "task": task, "control": "absent", "endpoint": endpoint,
                "n_projector_seeds": 0,
                "macro_accuracy_mean": macro_accuracy(task_frame, metric, group_key),
                "macro_accuracy_std": np.nan,
                "macro_accuracy_min": np.nan,
                "macro_accuracy_max": np.nan,
                "micro_accuracy_mean": float(task_frame[metric].mean()),
                "micro_accuracy_std": np.nan,
            })
    source_data = pd.concat([aggregate, pd.DataFrame(absent_rows)], ignore_index=True)

    rng = np.random.default_rng(args.bootstrap_seed)
    contrast_rows = []
    comparisons = (
        ("correct_minus_shuffle", "correct", "shuffle"),
        ("correct_minus_zero", "correct", "zero"),
        ("correct_minus_absent", "correct", "absent"),
        ("zero_minus_absent", "zero", "absent"),
    )
    for task in TASKS:
        for endpoint, metric, group_key in ENDPOINTS:
            for name, left_name, right_name in comparisons:
                pairs = {}
                for tag, controls in runs.items():
                    left = controls[left_name].loc[lambda x: x["task"].eq(task)]
                    right_source = absent if right_name == "absent" else controls[right_name]
                    right = right_source.loc[lambda x: x["task"].eq(task)]
                    pairs[tag] = merge_pair(left, right, metric, group_key)
                point, low, high = hierarchical_bootstrap(
                    pairs, metric, group_key, args.bootstrap, rng,
                )
                contrast_rows.append({
                    "contrast": name, "task": task, "endpoint": endpoint,
                    "n_projector_seeds": len(tags), "n_cells_per_seed": 500,
                    "macro_difference_mean": point, "ci95_low": low, "ci95_high": high,
                    "bootstrap_type": "hierarchical_seed_then_within_gold_label_cell",
                })
    contrasts = pd.DataFrame(contrast_rows)

    audit_frames = []
    for tag, controls in runs.items():
        for control, frame in controls.items():
            one = frame.copy()
            one.insert(0, "seed", tag)
            audit_frames.append(one)
    absent_audit = absent.copy()
    absent_audit.insert(0, "seed", "deterministic")
    audit_frames.append(absent_audit)
    audit = pd.concat(audit_frames, ignore_index=True)
    audit_rows = []
    for (seed, control, task), group in audit.groupby(["seed", "control", "task"], sort=True):
        resolved = group.loc[group["pred_family_resolved"]]
        schema_complete = group.loc[group["all_expected_fields"]]
        audit_rows.append({
            "seed": seed, "control": control, "task": task, "n": len(group),
            "nonempty_rate": float(group["answer_nonempty"].mean()),
            "structured_parse_rate": float(group["structured_parse_success"].mean()),
            "all_expected_fields_rate": float(group["all_expected_fields"].mean()),
            "pred_family_resolved_rate": float(group["pred_family_resolved"].mean()),
            "family_accuracy": float(group["family_correct"].mean()),
            "n_pred_family_resolved": len(resolved),
            "family_accuracy_given_pred_family_resolved": (
                float(resolved["family_correct"].mean()) if len(resolved) else np.nan
            ),
            "n_all_expected_fields": len(schema_complete),
            "family_accuracy_given_all_expected_fields": (
                float(schema_complete["family_correct"].mean()) if len(schema_complete) else np.nan
            ),
            "wrong_family_mention_rate": (
                float(group["wrong_family_mentioned"].mean())
                if task == "conflict_rationale" else np.nan
            ),
        })
    answer_audit = pd.DataFrame(audit_rows)
    pred_distribution = (
        audit.groupby(["seed", "control", "task", "pred_family"], dropna=False)
        .size().rename("n").reset_index()
    )
    pred_distribution["fraction"] = pred_distribution["n"] / pred_distribution.groupby(
        ["seed", "control", "task"]
    )["n"].transform("sum")

    seed_metrics.to_csv(out_dir / "null_control_seed_level_metrics.csv", index=False)
    source_data.to_csv(out_dir / "null_control_source_data.csv", index=False)
    contrasts.to_csv(out_dir / "null_control_contrasts_hierarchical_bootstrap.csv", index=False)
    answer_audit.to_csv(out_dir / "null_control_answer_audit.csv", index=False)
    pred_distribution.to_csv(out_dir / "null_control_predicted_family_distribution.csv", index=False)
    payload = {
        "status": "PASS",
        "projector_seeds": tags,
        "n_projector_seeds": len(tags),
        "n_cells": 500,
        "tasks": list(TASKS),
        "controls": ["correct", "shuffle", "zero", "absent"],
        "exact_prompt_match_for_projector_controls": True,
        "semantic_prompt_match_for_absent": True,
        "absent_definition": "No cell placeholders and no Projector forward pass; frozen Qwen sees text only.",
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "inference_dtype": args.inference_dtype,
        "reference_correct_shuffle_dtype": args.reference_dtype,
        "dtype_matched": args.inference_dtype == args.reference_dtype,
        "inputs": input_files,
    }
    (out_dir / "audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(source_data.sort_values(["endpoint", "task", "control"]).to_string(index=False))
    print("\n== contrasts ==")
    print(contrasts.to_string(index=False))
    print("\n== answer audit ==")
    print(answer_audit.to_string(index=False))


if __name__ == "__main__":
    main()
