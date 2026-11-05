#!/usr/bin/env python3
"""Evaluate deterministic raw-gene text interfaces with frozen Qwen."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_task1_open_questions import (
    family_from_text,
    family_mentioned,
    label_family,
    label_token_overlap,
    rationale_keyword_hit,
    wrong_family_for,
)


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[name.lower()]


def instruction(task: str, wrong_family: str) -> str:
    if task == "lineage_summary":
        return (
            "Assess the biological class most compatible with this cell-state representation. Give the broad class first, "
            "then the most plausible identity, followed by a brief piece of evidence. Do not select from a "
            "supplied list.\n\nUse this format:\nBroad class: ...\nBest match: ...\nEvidence: ..."
        )
    if task == "identity_rationale":
        return (
            "Based on the represented cell state, make the most specific defensible identification. "
            "State your proposed cell identity and one supporting biological feature.\n\n"
            "Use this format:\nBest match: ...\nSupporting feature: ..."
        )
    if task == "conflict_rationale":
        return (
            f"A preliminary annotation calls this a {wrong_family} cell. Treat that annotation as unverified. "
            "Reassess the cell using the representation itself and state the biological class, likely "
            "identity, and evidence for your decision.\n\nUse this format:\nRevised class: ...\n"
            "Likely identity: ...\nEvidence: ..."
        )
    raise ValueError(task)


def make_prompt(task: str, evidence: str | None, wrong_family: str) -> tuple[str, str]:
    task_instruction = instruction(task, wrong_family)
    evidence_line = (
        f"Gene-expression evidence: {evidence}\n\n"
        if evidence is not None else
        "No molecular measurement is provided.\n\n"
    )
    prompt = (
        "<|im_start|>system\nYou are interpreting a single-cell state. Base the answer on the supplied "
        "molecular evidence, not on generic biological expectations.<|im_end|>\n"
        "<|im_start|>user\n" + evidence_line + task_instruction +
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    instruction_hash = hashlib.sha256(task_instruction.encode()).hexdigest()
    return prompt, instruction_hash


def generate_batch(model, tokenizer, prompts: list[str], device: torch.device, max_new_tokens: int) -> list[str]:
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_length = encoded["input_ids"].shape[1]
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    return [
        tokenizer.decode(row[prompt_length:], skip_special_tokens=True).strip()
        for row in generated
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-json", required=True)
    parser.add_argument("--llm-path", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--serializer", choices=["top_expressed", "specificity_weighted", "expression_binned"], required=True)
    parser.add_argument("--control", choices=["correct", "shuffle", "absent"], required=True)
    parser.add_argument("--tasks", default="lineage_summary,identity_rationale,conflict_rationale")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm-dtype", default="bf16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    payload = json.loads(Path(args.evidence_json).read_text())
    rows = payload["rows"]
    by_cell = {row["cell_id"]: row for row in rows}
    tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    rng = random.Random(args.seed + 1)

    dtype = dtype_from_name(args.llm_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_path, torch_dtype=dtype, trust_remote_code=True,
    ).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    work = []
    for row in rows:
        gold_family = label_family(row["gold_label"])
        wrong_family = wrong_family_for(gold_family, rng)
        if args.control == "absent":
            evidence = None
            evidence_cell_id = ""
        elif args.control == "shuffle":
            evidence_cell_id = row["shuffle_cell_id"]
            evidence = by_cell[evidence_cell_id]["evidence"][args.serializer]
        else:
            evidence_cell_id = row["cell_id"]
            evidence = row["evidence"][args.serializer]
        for task in tasks:
            prompt, instruction_hash = make_prompt(task, evidence, wrong_family)
            work.append((row, task, gold_family, wrong_family, evidence_cell_id, evidence, prompt, instruction_hash))

    output_rows = []
    started = time.time()
    for start in range(0, len(work), args.batch_size):
        batch = work[start:start + args.batch_size]
        answers = generate_batch(model, tokenizer, [item[6] for item in batch], device, args.max_new_tokens)
        for item, answer in zip(batch, answers):
            row, task, gold_family, wrong_family, evidence_cell_id, evidence, prompt, instruction_hash = item
            pred_family = family_from_text(answer)
            token_overlap = label_token_overlap(row["gold_label"], answer)
            output_rows.append({
                "cell_id": row["cell_id"],
                "raw_cell_id": row["raw_cell_id"],
                "source": row["source"],
                "task": task,
                "serializer": args.serializer,
                "control": args.control,
                "gold_label": row["gold_label"],
                "gold_family": gold_family,
                "evidence_cell_id": evidence_cell_id,
                "wrong_family": wrong_family if task == "conflict_rationale" else "",
                "pred_family": pred_family,
                "family_correct": pred_family == gold_family,
                "mid_label_match": token_overlap >= 1.0,
                "wrong_family_mentioned": family_mentioned(wrong_family, answer) if task == "conflict_rationale" else False,
                "label_token_overlap": token_overlap,
                "rationale_keyword_hit": rationale_keyword_hit(gold_family, answer),
                "evidence_tokens": len(tokenizer(evidence or "", add_special_tokens=False).input_ids),
                "prompt_tokens": len(tokenizer(prompt, add_special_tokens=False).input_ids),
                "instruction_sha256": instruction_hash,
                "answer": answer,
                "prompt": prompt,
            })
        completed = min(start + args.batch_size, len(work))
        if start == 0 or completed % 40 == 0 or completed == len(work):
            print(f"[progress] {completed}/{len(work)}", flush=True)

    frame = pd.DataFrame(output_rows)
    summary_rows = []
    for task, group in frame.groupby("task", sort=True):
        per_family = group.groupby("gold_family")["family_correct"].mean()
        summary_rows.append({
            "serializer": args.serializer,
            "control": args.control,
            "task": task,
            "n": len(group),
            "n_families": len(per_family),
            "micro_accuracy": group["family_correct"].mean(),
            "macro_family_accuracy": per_family.mean(),
            "mid_label_accuracy": group["mid_label_match"].mean(),
            "mean_label_token_overlap": group["label_token_overlap"].mean(),
            "rationale_keyword_rate": group["rationale_keyword_hit"].mean(),
            "mean_evidence_tokens": group["evidence_tokens"].mean(),
            "mean_prompt_tokens": group["prompt_tokens"].mean(),
        })

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(prefix.with_suffix(".answers.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(prefix.with_suffix(".summary.csv"), index=False)
    result = {
        "evidence_json": args.evidence_json,
        "llm_path": args.llm_path,
        "serializer": args.serializer,
        "control": args.control,
        "n_cells": len(rows),
        "n_answers": len(frame),
        "elapsed_sec": time.time() - started,
        "summary": summary_rows,
    }
    prefix.with_suffix(".summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"[save] {prefix.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
