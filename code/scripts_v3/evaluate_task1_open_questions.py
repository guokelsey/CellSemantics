#!/usr/bin/env python
"""Evaluate open-ended Task1 biological questions for SAGA.

This benchmark is intentionally not exact ontology generation. It asks whether
the frozen decoder can use SAGA cell tokens to produce coarse, biologically
reasonable open responses:

1. lineage_summary: broad lineage/family + likely identity + short rationale
2. identity_rationale: likely cell identity with biological rationale
3. conflict_rationale: same as above, but with a misleading text hint

Historical automatic scoring uses lexical matching:
- family_correct: generated answer mentions the gold broad family
- wrong_family_mentioned: conflict answer repeats the injected wrong family
- label_token_overlap: lexical overlap with the gold mid/fine label
- rationale_keyword_hit: family-specific biological keyword appears

The goal is not to replace annotation-head accuracy, but to detect whether
open-ended frozen-Qwen readout has non-zero cell-conditioned signal.
These legacy matching rules are not a strict field parser or evidence of
gene-level mechanistic reasoning. See docs/protocol_scope.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.prompt_config import CELL_PLACEHOLDER
from models_v3.sagav3_llm import SingleCellLLMv3


FAMILY_KEYWORDS = {
    "Immune": [
        "immune", "lymphocyte", "t cell", "b cell", "nk", "nkt", "macrophage",
        "monocyte", "dendritic", "plasma", "mast", "basophil", "neutrophil",
        "antigen", "cytokine", "inflammatory",
    ],
    "Epithelial": [
        "epithelial", "basal", "luminal", "keratinocyte", "ciliated", "club",
        "secretory", "goblet", "ductal", "barrier", "mucosal", "lining",
    ],
    "Stromal/muscle": [
        "stromal", "fibroblast", "muscle", "myocyte", "smooth muscle",
        "myofibroblast", "mesenchymal", "matrix", "contractile", "collagen",
    ],
    "Endothelial/vascular": [
        "endothelial", "vascular", "capillary", "arterial", "venous",
        "lymphatic", "vessel", "angiogenic", "vasculature",
    ],
    "Hematopoietic progenitor": [
        "hematopoietic", "progenitor", "stem", "erythroid", "erythroblast",
        "megakaryocyte", "myeloid", "lymphoid", "precursor",
    ],
    "Pancreas/exocrine": [
        "pancreatic", "pancreas", "acinar", "ductal", "endocrine", "exocrine",
        "alpha", "beta", "delta", "gamma", "islet", "insulin", "glucagon",
    ],
    "Other": ["cell", "specialized", "tissue"],
}


def parse_torch_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[name.lower()]


def read_records(path: str | Path) -> list[dict[str, Any]]:
    obj = json.load(open(path))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("records"), list):
        return obj["records"]
    raise ValueError(f"Unsupported records JSON: {path}")


def load_uce_cache(path: str | Path) -> tuple[torch.Tensor, dict[str, int] | None]:
    d = torch.load(path, map_location="cpu")
    if isinstance(d, dict):
        x = d.get("embeddings", d.get("features"))
        ids = d.get("cell_ids", d.get("ids", d.get("cell_id")))
    else:
        x, ids = d, None
    if x is None:
        raise ValueError(f"No embeddings/features in {path}")
    id_to_row = None
    if ids is not None:
        ids = [str(v) for v in ids]
        id_to_row = {cid: i for i, cid in enumerate(ids)}
    return x.float(), id_to_row


def record_cell_id(record: dict[str, Any]) -> str:
    for k in ("cell_id", "id", "obs_id", "barcode"):
        if record.get(k) is not None:
            return str(record[k])
    return ""


def get_label(record: dict[str, Any], key: str) -> str:
    answer = record.get("answer", {})
    if not isinstance(answer, dict):
        answer = {}
    for source in (record, answer):
        v = source.get(key)
        if v is not None and str(v).strip() and str(v).lower() != "nan":
            return str(v).strip()
    if key == "mid_cell_type":
        return get_label(record, "cell_type")
    return ""


def label_family(label: object) -> str:
    s = str(label).lower().replace("_", " ")
    groups = {
        "Hematopoietic progenitor": [
            "hematopoietic", "progenitor", "stem cell", "erythroid", "erythroblast", "megakaryocyte",
        ],
        "Immune": [
            "t cell", "b cell", "nk", "nkt", "macrophage", "monocyte", "dendritic",
            "plasma", "immune", "mast", "basophil", "neutrophil", "lymph",
        ],
        "Epithelial": [
            "epithelial", "basal", "luminal", "club", "secretory", "goblet",
            "keratinocyte", "ciliated", "ductal",
        ],
        "Stromal/muscle": [
            "fibroblast", "stromal", "smooth muscle", "myocyte", "pericyte",
            "mesenchymal", "stellate", "muscle",
        ],
        "Endothelial/vascular": [
            "endothelial", "vascular", "capillary", "artery", "venous", "vein", "lymphatic",
        ],
        "Pancreas/exocrine": [
            "acinar", "alpha", "beta", "delta", "gamma", "endocrine", "exocrine", "pancrea",
        ],
    }
    for fam, words in groups.items():
        if any(w in s for w in words):
            return fam
    return "Other"


def normalize_text(x: object) -> str:
    s = str(x).lower()
    s = re.sub(r"[^a-z0-9+\-/ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def family_from_text(text: str) -> str:
    t = normalize_text(text)
    scores = {}
    for fam, words in FAMILY_KEYWORDS.items():
        if fam == "Other":
            continue
        scores[fam] = sum(1 for w in words if w in t)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Unknown"


def family_mentioned(fam: str, text: str) -> bool:
    t = normalize_text(text)
    return any(w in t for w in FAMILY_KEYWORDS.get(fam, []))


def label_token_overlap(gold_label: str, text: str) -> float:
    stop = {"cell", "cells", "of", "the", "and", "positive", "human", "lineage"}
    gold_tokens = [
        t for t in re.split(r"[^a-z0-9]+", gold_label.lower())
        if len(t) >= 3 and t not in stop
    ]
    if not gold_tokens:
        return 0.0
    answer = normalize_text(text)
    hit = sum(1 for t in set(gold_tokens) if t in answer)
    return hit / max(1, len(set(gold_tokens)))


def rationale_keyword_hit(gold_family: str, text: str) -> bool:
    t = normalize_text(text)
    words = FAMILY_KEYWORDS.get(gold_family, [])
    return any(w in t for w in words)


def make_prompt(
    task: str,
    num_cell_tokens: int,
    wrong_family: str | None = None,
    prompt_style: str = "canonical",
    include_cell_tokens: bool = True,
) -> str:
    cells = " ".join([CELL_PLACEHOLDER] * num_cell_tokens) if include_cell_tokens else ""
    if prompt_style == "canonical" and task == "lineage_summary":
        instruction = (
            "Describe this single cell in natural language. Give a broad biological lineage or family, "
            "a likely cell identity, and one short rationale. Do not use a candidate list.\n\n"
            "Reply in three short fields:\n"
            "Lineage: ...\nIdentity: ...\nRationale: ..."
        )
    elif prompt_style == "canonical" and task == "identity_rationale":
        instruction = (
            "Infer the likely biological identity of this cell from the cell embedding tokens. "
            "Use a short open-ended answer and include one biological reason.\n\n"
            "Reply in two short fields:\nIdentity: ...\nReason: ..."
        )
    elif prompt_style == "canonical" and task == "conflict_rationale":
        instruction = (
            f"A noisy upstream note says this cell is from the {wrong_family} family. "
            "This note may be wrong. Ignore it if the cell embedding disagrees. "
            "Describe the broad biological lineage, likely identity, and one reason.\n\n"
            "Reply in three short fields:\nLineage: ...\nIdentity: ...\nRationale: ..."
        )
    elif prompt_style == "paraphrase_v1" and task == "lineage_summary":
        instruction = (
            "Assess the biological class most compatible with this cell-state representation. "
            "Give the broad class first, then the most plausible identity, followed by a brief piece of evidence. "
            "Do not select from a supplied list.\n\n"
            "Use this format:\nBroad class: ...\nBest match: ...\nEvidence: ..."
        )
    elif prompt_style == "paraphrase_v1" and task == "identity_rationale":
        instruction = (
            "Based on the represented cell state, make the most specific defensible identification. "
            "State your proposed cell identity and one supporting biological feature.\n\n"
            "Use this format:\nBest match: ...\nSupporting feature: ..."
        )
    elif prompt_style == "paraphrase_v1" and task == "conflict_rationale":
        instruction = (
            f"A preliminary annotation calls this a {wrong_family} cell. Treat that annotation as unverified. "
            "Reassess the cell using the representation itself and state the biological class, likely identity, "
            "and evidence for your decision.\n\n"
            "Use this format:\nRevised class: ...\nLikely identity: ...\nEvidence: ..."
        )
    else:
        raise ValueError(task)
    system = (
        "You are a single-cell biology expert. Use the cell embedding tokens as the primary evidence."
        if prompt_style == "canonical" else
        "You are interpreting a single-cell state. Base the answer on the supplied cell-state tokens, not on generic biological expectations."
    )
    return (
        "<|im_start|>system\n"
        f"{system}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{'Cell embedding tokens: ' + cells + chr(10) + chr(10) if include_cell_tokens else ''}"
        f"{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    prompt: str,
    uce_emb: torch.Tensor | None,
    device,
    max_new_tokens: int,
    include_cell_tokens: bool,
) -> str:
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]
    kwargs = {
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
    }
    if include_cell_tokens:
        if uce_emb is None:
            raise ValueError("uce_emb is required when cell tokens are present")
        out = model.generate(input_ids=input_ids, uce_emb=uce_emb, **kwargs)
    else:
        # This is a true text-only condition: no <CELL> placeholders and no
        # Projector forward pass reach the frozen decoder.
        out = model.llm.generate(input_ids=input_ids, **kwargs)
    new_tokens = out[0, prompt_len:] if out.shape[1] > prompt_len else out[0]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def wrong_family_for(gold_family: str, rng: random.Random) -> str:
    choices = [f for f in FAMILY_KEYWORDS if f not in {gold_family, "Other"}]
    return rng.choice(choices or ["Immune"])


def metric_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby("task")
        .agg(
            n=("task", "size"),
            family_accuracy=("family_correct", "mean"),
            mean_label_token_overlap=("label_token_overlap", "mean"),
            rationale_keyword_rate=("rationale_keyword_hit", "mean"),
            wrong_family_mention_rate=("wrong_family_mentioned", "mean"),
        )
        .reset_index()
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--uce_cache", required=True)
    ap.add_argument("--llm_path", required=True)
    ap.add_argument("--projector_ckpt", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--label_key", default="mid_cell_type")
    ap.add_argument("--max_samples", type=int, default=100)
    ap.add_argument("--tasks", default="lineage_summary,identity_rationale,conflict_rationale")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--llm_dtype", default="fp32")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--control", choices=["correct", "zero", "shuffle", "absent"], default="correct",
                    help="Cell-token intervention. absent removes the placeholder segment and bypasses the Projector.")
    ap.add_argument("--prompt_style", choices=["canonical", "paraphrase_v1"], default="canonical",
                    help="Question wording. paraphrase_v1 changes instructions, field names and conflict framing.")
    ap.add_argument("--num_cell_tokens", type=int, default=32)
    ap.add_argument("--input_emb_dim", type=int, default=1280)
    ap.add_argument("--projector_hidden", type=int, default=1024)
    ap.add_argument("--num_mixer_layers", type=int, default=4)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--cell_token_mode", default="resize", choices=["resize", "literal"])
    args = ap.parse_args()

    # Keep sampled cells and conflict prompts invariant across intervention
    # conditions. Shuffle assignment is intentionally the only independent
    # random process, so correct/zero/shuffle/absent prompts can be audited.
    sample_rng = random.Random(args.seed)
    prompt_rng = random.Random(args.seed + 1)
    shuffle_rng = random.Random(args.seed + 2)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    pred_jsonl = out_prefix.with_suffix(".predictions.jsonl")
    answers_csv = out_prefix.with_suffix(".answers.csv")
    summary_csv = out_prefix.with_suffix(".summary.csv")
    summary_json = out_prefix.with_suffix(".summary.json")

    records = read_records(args.records)
    embeddings, id_to_row = load_uce_cache(args.uce_cache)
    valid = []
    for i, r in enumerate(records):
        cid = record_cell_id(r)
        if id_to_row is None:
            row = i if i < len(embeddings) else None
        else:
            row = id_to_row.get(cid)
        label = get_label(r, args.label_key)
        if row is not None and label:
            valid.append((i, row, cid, label))
    if args.max_samples > 0 and len(valid) > args.max_samples:
        valid = sample_rng.sample(valid, args.max_samples)

    print(
        f"[eval-open] records={len(records)} valid={len(valid)} tasks={args.tasks} "
        f"prompt_style={args.prompt_style}",
        flush=True,
    )
    dtype = parse_torch_dtype(args.llm_dtype)
    model = SingleCellLLMv3(
        llm_path=args.llm_path,
        uce_dim=args.input_emb_dim,
        hidden=args.projector_hidden,
        num_cell_tokens=args.num_cell_tokens,
        num_mixer_layers=args.num_mixer_layers,
        num_heads=args.num_heads,
        llm_dtype=dtype,
        cell_token_mode=args.cell_token_mode,
    )
    model.load_projector(args.projector_ckpt)
    model.eval()
    device = next(model.llm.parameters()).device
    tokenizer = model.tokenizer

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    rows = []
    t0 = time.time()
    with pred_jsonl.open("w") as jf, torch.no_grad():
        for sample_i, (record_i, row_i, cid, gold_label) in enumerate(valid, 1):
            gold_family = label_family(gold_label)
            wrong_family = wrong_family_for(gold_family, prompt_rng)
            if args.control == "zero":
                emb_cpu = torch.zeros_like(embeddings[row_i])
                control_row_i = -1
            elif args.control == "shuffle":
                _, control_row_i, _, _ = shuffle_rng.choice(valid)
                emb_cpu = embeddings[control_row_i]
            elif args.control == "absent":
                emb_cpu = None
                control_row_i = -2
            else:
                emb_cpu = embeddings[row_i]
                control_row_i = row_i
            emb = None if emb_cpu is None else emb_cpu.unsqueeze(0).to(device=device, dtype=dtype)
            for task in task_list:
                prompt = make_prompt(
                    task,
                    args.num_cell_tokens,
                    wrong_family=wrong_family,
                    prompt_style=args.prompt_style,
                    include_cell_tokens=args.control != "absent",
                )
                answer = generate_answer(
                    model, tokenizer, prompt, emb, device, args.max_new_tokens,
                    include_cell_tokens=args.control != "absent",
                )
                pred_family = family_from_text(answer)
                row = {
                    "sample_i": sample_i,
                    "record_i": record_i,
                    "cell_id": cid,
                    "task": task,
                    "gold_label": gold_label,
                    "gold_family": gold_family,
                    "control": args.control,
                    "prompt_style": args.prompt_style,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "embedding_row": int(row_i),
                    "control_embedding_row": int(control_row_i),
                    "wrong_family": wrong_family if task == "conflict_rationale" else "",
                    "pred_family": pred_family,
                    "family_correct": pred_family == gold_family,
                    "wrong_family_mentioned": family_mentioned(wrong_family, answer) if task == "conflict_rationale" else False,
                    "label_token_overlap": label_token_overlap(gold_label, answer),
                    "rationale_keyword_hit": rationale_keyword_hit(gold_family, answer),
                    "answer": answer,
                    "prompt": prompt,
                }
                rows.append(row)
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")
            if sample_i == 1 or sample_i % 10 == 0:
                partial = pd.DataFrame(rows)
                fam_acc = float(partial["family_correct"].mean()) if len(partial) else 0.0
                print(f"[progress] {sample_i}/{len(valid)} family_acc={fam_acc:.3f}", flush=True)

    df = pd.DataFrame(rows)
    summary = metric_summary(df)
    df.to_csv(answers_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    payload = {
        "records": args.records,
        "uce_cache": args.uce_cache,
        "llm_path": args.llm_path,
        "projector_ckpt": args.projector_ckpt,
        "label_key": args.label_key,
        "control": args.control,
        "prompt_style": args.prompt_style,
        "seed": args.seed,
        "llm_dtype": args.llm_dtype,
        "max_new_tokens": args.max_new_tokens,
        "num_cell_tokens": args.num_cell_tokens,
        "input_emb_dim": args.input_emb_dim,
        "projector_hidden": args.projector_hidden,
        "num_mixer_layers": args.num_mixer_layers,
        "num_heads": args.num_heads,
        "cell_token_mode": args.cell_token_mode,
        "do_sample": False,
        "n_cells": len(valid),
        "tasks": task_list,
        "elapsed_sec": time.time() - t0,
        "predictions_jsonl": str(pred_jsonl),
        "answers_csv": str(answers_csv),
        "summary_csv": str(summary_csv),
        "summary": summary.to_dict(orient="records"),
    }
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(f"[save] {summary_json}", flush=True)


if __name__ == "__main__":
    main()
