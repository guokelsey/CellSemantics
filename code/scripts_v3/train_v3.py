"""SAGA v3.0 -- train UCE -> Qwen projector.

This is a stripped-down version of `scripts_v2/train_v2.py`: no Evo2, no
top-K memmap, no Stage-1 MGM, no readout/recon. Just::

    cell_id  ->  UCE 1280-d embedding (precomputed)
              ->  UCEProjectorV3   (M cell tokens)
              ->  Qwen2.5-14B (frozen)
              ->  CE on the SFT target (cell_type | organ)
              [+ aux CE on (cell_type, organ) directly from UCE 1280-d]

Diverse prompts + perturb augmentation are reused from
`models/prompt_config.py` so the same instruction-mixing strategy that
worked for v17 is in effect.

Usage::
    python scripts_v3/train_v3.py \\
        --sft_data data/phase1_sft_macrogene_train_split.json \\
        --uce_train data/uce_cache/saga_train_uce.pt \\
        --uce_val   data/uce_cache/saga_val_uce.pt \\
        --llm_path /path/to/Qwen2.5-14B-Instruct \\
        --output_dir checkpoints/v3_uce \\
        --epochs 3 --batch_size 4 --grad_accum 4 --lr 1e-4 \\
        --num_cell_tokens 16 --diverse_prompts
"""

from __future__ import annotations
import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import random
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset, Subset
from transformers import Trainer, TrainingArguments, TrainerCallback


def patch_accelerate_unwrap_model_compat() -> None:
    """Allow newer Transformers to run with older Accelerate."""
    try:
        import inspect
        from accelerate import Accelerator
    except Exception:
        return
    signature = inspect.signature(Accelerator.unwrap_model)
    if "keep_torch_compile" in signature.parameters:
        return
    original = Accelerator.unwrap_model

    def unwrap_model(self, model, keep_fp32_wrapper=True, keep_torch_compile=False):
        return original(self, model, keep_fp32_wrapper=keep_fp32_wrapper)

    Accelerator.unwrap_model = unwrap_model


patch_accelerate_unwrap_model_compat()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models_v3.sagav3_llm import SingleCellLLMv3, CELL_PLACEHOLDER
from models_v3.uce_projector import UCE_HIDDEN_DIM
from models.prompt_config import (
    make_classification_prompt, make_perturbed_classification_prompt,
    make_freeform_prompt, make_json_prompt, make_twoline_prompt,
    make_dropout_prompt, sample_diverse_mode, DEFAULT_DIVERSE_WEIGHTS,
    format_label, sample_task,
    TASK_CT_ORGAN, CELL_TYPES, ORGANS,
    # v18 evidence-driven tasks (now reused in v3)
    make_top_gene_prompt, format_top_genes_label,
    make_conflict_prompt, sample_wrong_label,
    make_gene_reference_prompt, format_gene_reference_label,
    make_trajectory_prediction_prompt, format_trajectory_label,
    make_grn_prediction_prompt, format_grn_label, format_grn_binary_label,
)
from scripts.step1_train_projector import (
    parse_rich_label, parse_target, fuzzy_match,
)


CT_TO_IDX = {c: i for i, c in enumerate(CELL_TYPES)}
ORG_TO_IDX = {o: i for i, o in enumerate(ORGANS)}


def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def stable_label_id(label: str) -> int:
    if not label:
        return -100
    digest = hashlib.sha1(str(label).strip().lower().encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % 2_000_000_000


def first_nonempty(record: dict, keys: tuple[str, ...], default: str = "") -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        value = str(value).strip()
        if value and value.lower() not in {"nan", "none", "null", "unknown"}:
            return value
    return default


def record_source_label(record: dict) -> str:
    return first_nonempty(
        record,
        (
            "source_dataset", "source", "dataset", "atlas", "study",
            "organ", "organ_tissue", "tissue",
        ),
        default="unknown_source",
    )


def record_mid_cell_type(record: dict, fine_cell_type: str) -> str:
    return first_nonempty(
        record,
        (
            "mid_cell_type", "coarse_cell_type", "cell_type_mid",
            "ann_level_2", "ann_level_3", "cell_type_superclass",
        ),
        default=fine_cell_type,
    )


def build_hierarchy_label_maps(records_path: str) -> dict[str, dict[str, int]]:
    """Build contiguous hierarchy vocabularies from the training split only."""
    payload = json.load(open(records_path))
    records = payload.get("records", []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"Unsupported hierarchy records format: {records_path}")
    labels = {"fine": set(), "mid": set(), "organ": set()}
    for record in records:
        if not isinstance(record, dict):
            continue
        fine, organ = parse_rich_label(str(record.get("target_text") or ""))
        fine = first_nonempty(record, ("cell_type", "celltype"), default=fine)
        organ = first_nonempty(
            record, ("organ", "organ_tissue", "tissue"), default=organ
        )
        mid = record_mid_cell_type(record, fine)
        for key, value in (("fine", fine), ("mid", mid), ("organ", organ)):
            value = str(value or "").strip()
            if value and value.lower() not in {"nan", "none", "null", "unknown"}:
                labels[key].add(value)
    maps = {
        key: {label: idx for idx, label in enumerate(sorted(values))}
        for key, values in labels.items()
    }
    if any(not mapping for mapping in maps.values()):
        sizes = {key: len(mapping) for key, mapping in maps.items()}
        raise ValueError(f"Empty hierarchy vocabulary from {records_path}: {sizes}")
    return maps


def env_rank() -> int:
    return int(os.environ.get("RANK", "-1"))


def is_env_rank0() -> bool:
    return env_rank() in (-1, 0)


def write_json(path: str | os.PathLike, payload: dict) -> None:
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    rank = os.environ.get("RANK", "single")
    tmp = f"{path}.rank{rank}.pid{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def append_jsonl(path: str | os.PathLike, payload: dict) -> None:
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(payload) + "\n")


def file_info(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"path": path, "exists": False}
    st = p.stat()
    return {
        "path": str(p),
        "exists": True,
        "bytes": int(st.st_size),
        "mtime": _dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
    }


def infer_embedding_dim(cache_path: str) -> int:
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    emb = cache["embeddings"]
    if emb.ndim != 2:
        raise ValueError(f"{cache_path} embeddings must be 2-D, got {tuple(emb.shape)}")
    return int(emb.shape[1])


def _resolve_vector_path(path: str, json_path: Path, repo_root: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    candidates = [
        Path.cwd() / p,
        repo_root / p,
        json_path.parent / p,
        json_path.parent / ".." / p,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return (repo_root / p).resolve()


def _load_vector_from_record(record: dict, json_path: Path, repo_root: Path) -> torch.Tensor:
    for key in ("embedding", "cell_embedding", "uce_embedding", "macrogene_embedding"):
        if key in record:
            return torch.as_tensor(record[key], dtype=torch.float32).view(-1)
    cache_path = record.get("embedding_cache_path") or record.get("cache_path")
    if cache_path:
        path = _resolve_vector_path(
            str(cache_path), json_path=json_path, repo_root=repo_root
        )
        cache = _EXTRA_EMBED_CACHE.get(str(path))
        if cache is None:
            try:
                payload = torch.load(
                    path, map_location="cpu", weights_only=True, mmap=True
                )
            except TypeError:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            cache = torch.as_tensor(payload["embeddings"]).float()
            _EXTRA_EMBED_CACHE[str(path)] = cache
        source_index = int(record["source_index"])
        return cache[source_index].view(-1)
    embed_path = record.get("embed_path") or record.get("embedding_path")
    if not embed_path:
        raise KeyError("Extra task record has no embedding field or embed_path")
    path = _resolve_vector_path(str(embed_path), json_path=json_path, repo_root=repo_root)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".npy":
        return torch.from_numpy(np.load(path)).float().view(-1)
    if path.suffix == ".npz":
        payload = np.load(path)
        key = "embedding" if "embedding" in payload else payload.files[0]
        return torch.from_numpy(payload[key]).float().view(-1)
    return torch.load(path, map_location="cpu", weights_only=True).float().view(-1)


_EXTRA_EMBED_CACHE: dict[str, torch.Tensor] = {}


def _load_extra_task_records(path: str) -> list[dict]:
    payload = json.load(open(path))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("records"), list):
            records = payload["records"]
        else:
            records = []
            for key, value in payload.items():
                if isinstance(value, dict):
                    item = dict(value)
                    item.setdefault("cell_id", key)
                    records.append(item)
    else:
        raise ValueError("extra task JSON must be a list, a dict of records, or {'records': [...]}")
    out = [r for r in records if isinstance(r, dict)]
    if not out:
        raise ValueError(f"No usable extra task records found in {path}")
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CellSFTDatasetV3(Dataset):
    """SFT dataset that pulls per-cell embedding vectors from a torch cache.

    The cache is a dict {"cell_ids": [...], "embeddings": (N, D) tensor}.
    It can be produced from UCE h5ad inference, or directly from split SFT JSON
    via `scripts_v3/build_embedding_cache_from_sft.py`.
    """

    def __init__(
        self,
        sft_data_path: str,
        tokenizer,
        uce_cache_path: str,
        max_length: int = 768,
        num_cell_tokens: int = 16,
        perturbation_dict: dict | None = None,
        perturb_prob: float = 0.0,
        perturb_seed: int = 2026,
        diverse_prompts: bool = False,
        diverse_weights: tuple = DEFAULT_DIVERSE_WEIGHTS,
        topk_prefix: str | None = None,
        top_gene_prob: float = 0.0,
        gene_reference_prob: float = 0.0,
        trajectory_prediction_prob: float = 0.0,
        grn_prediction_prob: float = 0.0,
        grn_answer_mode: str = "reasoning",
        conflict_prob: float = 0.0,
        top_gene_k: int = 5,
        trajectory_data_path: str | None = None,
        grn_data_path: str | None = None,
        hierarchy_label_maps: dict[str, dict[str, int]] | None = None,
    ):
        super().__init__()
        with open(sft_data_path) as f:
            self.samples = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_cell_tokens = num_cell_tokens
        self.hierarchy_label_maps = hierarchy_label_maps or {}

        # ---------------- UCE cache ----------------
        cache = torch.load(uce_cache_path, map_location="cpu", weights_only=True)
        self.cell_ids = list(cache["cell_ids"])
        self.uce_table = cache["embeddings"].float().contiguous()  # (N, 1280)
        self.cid_to_row = {cid: i for i, cid in enumerate(self.cell_ids)}
        n_match = sum(1 for s in self.samples if s["cell_id"] in self.cid_to_row)
        print(f"[v3-ds] sft={len(self.samples):,}  uce_table={tuple(self.uce_table.shape)}  "
              f"matched={n_match} ({n_match/max(len(self.samples),1)*100:.1f}%)",
              flush=True)

        # ---------------- perturb / diverse prompt config ----------------
        self.perturbation_dict = perturbation_dict or {}
        self.perturb_prob = float(perturb_prob)
        self._rng = random.Random(perturb_seed)
        if self.perturb_prob > 0 and self.perturbation_dict:
            covered = sum(1 for c in CELL_TYPES if c in self.perturbation_dict
                          and self.perturbation_dict[c])
            print(f"[v3-ds] perturbation ENABLED prob={self.perturb_prob:.2f} "
                  f"covered={covered}/{len(CELL_TYPES)}", flush=True)
        self.diverse_prompts = diverse_prompts
        self.diverse_weights = tuple(diverse_weights)
        if diverse_prompts:
            print(f"[v3-ds] DIVERSE prompts ENABLED weights="
                  f"{self.diverse_weights}", flush=True)

        # ---------------- evidence-driven tasks ----------------
        # The v3 model only sees a single 1280-d UCE vector per cell, but we
        # can still ask it to predict per-cell gene evidence. The named
        # gene_reference task and the legacy top_gene task both look up the
        # top-K HGNC symbols from the v2-format memmap
        # (`data/cells_v2_{split}_topk_*`). Conflict task does not need extra
        # data because the true cell_type/organ label is already in `samples`.
        self.top_gene_prob = float(top_gene_prob)
        self.gene_reference_prob = float(gene_reference_prob)
        self.trajectory_prediction_prob = float(trajectory_prediction_prob)
        self.grn_prediction_prob = float(grn_prediction_prob)
        self.grn_answer_mode = str(grn_answer_mode or "reasoning")
        self.conflict_prob = float(conflict_prob)
        self.top_gene_k = int(top_gene_k)
        self._topk_idx_mm = None
        self._topk_meta = None
        self.trajectory_by_cell: dict = {}
        self.grn_by_cell: dict = {}
        if trajectory_data_path and self.trajectory_prediction_prob > 0:
            with open(trajectory_data_path) as f:
                raw_traj = json.load(f)
            if isinstance(raw_traj, dict):
                self.trajectory_by_cell = raw_traj
            elif isinstance(raw_traj, list):
                self.trajectory_by_cell = {
                    str(item["cell_id"]): item
                    for item in raw_traj
                    if isinstance(item, dict) and "cell_id" in item
                }
            else:
                raise ValueError("trajectory_data_path must contain a dict or list")
            n_traj = sum(1 for s in self.samples
                         if str(s["cell_id"]) in self.trajectory_by_cell)
            print(f"[v3-ds] TRAJECTORY task ENABLED  "
                  f"trajectory_prediction_prob={self.trajectory_prediction_prob:.2f}  "
                  f"coverage={n_traj}/{len(self.samples)}", flush=True)
        elif self.trajectory_prediction_prob > 0:
            print("[v3-ds] WARN: trajectory_prediction_prob > 0 but no "
                  "trajectory_data_path was provided; trajectory task disabled",
                  flush=True)
            self.trajectory_prediction_prob = 0.0

        if grn_data_path and self.grn_prediction_prob > 0:
            with open(grn_data_path) as f:
                raw_grn = json.load(f)
            if isinstance(raw_grn, dict):
                self.grn_by_cell = raw_grn
            elif isinstance(raw_grn, list):
                self.grn_by_cell = {
                    str(item["cell_id"]): item
                    for item in raw_grn
                    if isinstance(item, dict) and "cell_id" in item
                }
            else:
                raise ValueError("grn_data_path must contain a dict or list")
            n_grn = sum(1 for s in self.samples
                        if str(s["cell_id"]) in self.grn_by_cell)
            print(f"[v3-ds] GRN task ENABLED  "
                  f"grn_prediction_prob={self.grn_prediction_prob:.2f}  "
                  f"coverage={n_grn}/{len(self.samples)}", flush=True)
        elif self.grn_prediction_prob > 0:
            print("[v3-ds] WARN: grn_prediction_prob > 0 but no "
                  "grn_data_path was provided; GRN task disabled",
                  flush=True)
            self.grn_prediction_prob = 0.0

        use_gene_reference = self.gene_reference_prob > 0 or self.top_gene_prob > 0
        if topk_prefix and use_gene_reference:
            try:
                with open(topk_prefix + "_meta.json") as f:
                    self._topk_meta = json.load(f)
                self._topk_K = self._topk_meta["top_k"]
                self._topk_N = self._topk_meta["n_cells"]
                self._topk_vocab = self._topk_meta["vocab"]
                self._topk_cell_to_row = self._topk_meta["cell_id_to_row"]
                self._topk_idx_mm = np.memmap(
                    topk_prefix + "_topk_idx.bin",
                    dtype="int32", mode="r",
                    shape=(self._topk_N, self._topk_K),
                )
                n_cov = sum(1 for s in self.samples
                            if s["cell_id"] in self._topk_cell_to_row)
                print(f"[v3-ds] EVIDENCE tasks ENABLED  "
                      f"top_gene_prob={self.top_gene_prob:.2f} (k={self.top_gene_k})  "
                      f"gene_reference_prob={self.gene_reference_prob:.2f}  "
                      f"conflict_prob={self.conflict_prob:.2f}  "
                      f"topk_table_cov={n_cov}/{len(self.samples)}",
                      flush=True)
            except FileNotFoundError as e:
                print(f"[v3-ds] WARN: top-gene memmap not found ({e}); "
                      f"will use top_genes/reference_genes fields in SFT JSON "
                      f"when present", flush=True)
        elif self.conflict_prob > 0:
            print(f"[v3-ds] EVIDENCE tasks (conflict only)  "
                  f"conflict_prob={self.conflict_prob:.2f}", flush=True)

    def __len__(self):
        return len(self.samples)

    def _build_prompt_and_target(self, ct: str, organ: str,
                                 top_gene_names: list[str] | None = None,
                                 trajectory_target=None,
                                 trajectory_context: str = "",
                                 grn_target=None,
                                 grn_context=None):
        n = self.num_cell_tokens

        # ---------------- named v3 tasks (priority) ----------------
        if (self.grn_prediction_prob > 0 and grn_target
                and self._rng.random() < self.grn_prediction_prob):
            return (
                make_grn_prediction_prompt(
                    n, grn_context=grn_context, rng=self._rng,
                    answer_mode=self.grn_answer_mode),
                format_grn_binary_label(grn_target)
                if self.grn_answer_mode == "binary"
                else format_grn_label(grn_target),
            )
        if (self.trajectory_prediction_prob > 0 and trajectory_target
                and self._rng.random() < self.trajectory_prediction_prob):
            return (
                make_trajectory_prediction_prompt(
                    n, context=trajectory_context, rng=self._rng),
                format_trajectory_label(trajectory_target),
            )
        if (self.gene_reference_prob > 0 and top_gene_names
                and self._rng.random() < self.gene_reference_prob):
            return (
                make_gene_reference_prompt(n, k=self.top_gene_k, rng=self._rng),
                format_gene_reference_label(top_gene_names, k=self.top_gene_k),
            )

        # ---------------- v18 evidence-driven tasks (legacy alias) ------------
        if (self.top_gene_prob > 0 and top_gene_names
                and self._rng.random() < self.top_gene_prob):
            return (
                make_top_gene_prompt(n, k=self.top_gene_k, rng=self._rng),
                format_top_genes_label(top_gene_names, k=self.top_gene_k),
            )
        if self.conflict_prob > 0 and self._rng.random() < self.conflict_prob:
            wct, worg = sample_wrong_label(ct, organ, rng=self._rng)
            return (
                make_conflict_prompt(n, wrong_cell_type=wct, wrong_organ=worg,
                                     rng=self._rng),
                format_label(ct, organ, TASK_CT_ORGAN),
            )

        if self.diverse_prompts:
            mode = sample_diverse_mode(self._rng, self.diverse_weights)
            if mode == "freeform":
                return make_freeform_prompt(n, rng=self._rng), \
                       format_label(ct, organ, TASK_CT_ORGAN)
            if mode == "json":
                return make_json_prompt(n), format_label(ct, organ, TASK_CT_ORGAN)
            if mode == "twoline":
                return make_twoline_prompt(n), format_label(ct, organ, TASK_CT_ORGAN)
            if mode == "dropout":
                return make_dropout_prompt(n), format_label(ct, organ, TASK_CT_ORGAN)
            task = sample_task()
            if self.perturb_prob > 0 and self._rng.random() < self.perturb_prob:
                perturb_text = self._maybe_get_perturbation(ct)
                if perturb_text:
                    return make_perturbed_classification_prompt(
                        n, task=task, rng=self._rng,
                        perturbation_text=perturb_text,
                    ), format_label(ct, organ, task)
                return make_perturbed_classification_prompt(
                    n, task=task, rng=self._rng,
                ), format_label(ct, organ, task)
            return make_classification_prompt(n, task=task), \
                   format_label(ct, organ, task)
        # non-diverse path
        task = sample_task()
        if self.perturb_prob > 0 and self._rng.random() < self.perturb_prob:
            perturb_text = self._maybe_get_perturbation(ct)
            if perturb_text:
                return make_perturbed_classification_prompt(
                    n, task=task, rng=self._rng,
                    perturbation_text=perturb_text,
                ), format_label(ct, organ, task)
            return make_perturbed_classification_prompt(
                n, task=task, rng=self._rng,
            ), format_label(ct, organ, task)
        return make_classification_prompt(n, task=task), \
               format_label(ct, organ, task)

    def _maybe_get_perturbation(self, cell_type: str) -> str:
        if self.perturb_prob <= 0 or not self.perturbation_dict:
            return ""
        candidates = self.perturbation_dict.get(cell_type, [])
        if not candidates:
            return ""
        return self._rng.choice(candidates)

    def __getitem__(self, idx):
        s = self.samples[idx]
        cid = s["cell_id"]
        row = self.cid_to_row.get(cid, 0)
        uce = self.uce_table[row].clone()                        # (D,)

        ct, organ = parse_rich_label(s["target_text"])

        # Look up top-K gene SYMBOLS for this cell. Prefer genes already stored
        # in the split JSON; fall back to the optional top-K memmap.
        top_gene_names: list[str] = []
        for key in ("gene_reference", "reference_genes", "top_genes", "top_gene_names"):
            if key in s and s[key]:
                value = s[key]
                if isinstance(value, str):
                    top_gene_names = [g.strip() for g in value.split(",") if g.strip()]
                else:
                    top_gene_names = [str(g) for g in value]
                break
        if (not top_gene_names and self._topk_idx_mm is not None
                and (self.top_gene_prob > 0 or self.gene_reference_prob > 0)):
            tk_row = self._topk_cell_to_row.get(cid)
            if tk_row is not None:
                idx_arr = self._topk_idx_mm[tk_row, :self.top_gene_k]
                for gi in idx_arr:
                    if gi < 0:
                        break
                    top_gene_names.append(self._topk_vocab[int(gi)])

        trajectory_record = self.trajectory_by_cell.get(str(cid), {})
        trajectory_target = None
        trajectory_context = ""
        if trajectory_record:
            if isinstance(trajectory_record, dict):
                trajectory_context = str(trajectory_record.get("context", ""))
                trajectory_target = (
                    trajectory_record.get("target")
                    or trajectory_record.get("label")
                    or trajectory_record
                )
            else:
                trajectory_target = trajectory_record
        grn_record = self.grn_by_cell.get(str(cid), {})
        grn_target = None
        grn_context = None
        if grn_record:
            if isinstance(grn_record, dict):
                grn_context = grn_record.get("context", grn_record)
                grn_target = (
                    grn_record.get("answer")
                    or grn_record.get("response")
                    or grn_record.get("formatted_label")
                    or grn_record
                )
            else:
                grn_context = grn_record
                grn_target = grn_record
        prompt, target = self._build_prompt_and_target(
            ct, organ, top_gene_names=top_gene_names if top_gene_names else None,
            trajectory_target=trajectory_target,
            trajectory_context=trajectory_context,
            grn_target=grn_target,
            grn_context=grn_context,
        )
        full = prompt + " " + target + self.tokenizer.eos_token
        enc = self.tokenizer(full, max_length=self.max_length,
                             padding="max_length", truncation=True,
                             return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        prompt_enc = self.tokenizer(prompt, max_length=self.max_length,
                                    truncation=True, return_tensors="pt")
        prompt_len = prompt_enc["input_ids"].shape[1]
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        ct_idx = torch.tensor(CT_TO_IDX.get(ct, -100), dtype=torch.long)
        org_idx = torch.tensor(ORG_TO_IDX.get(organ, -100), dtype=torch.long)
        ct_hash = torch.tensor(stable_label_id(ct), dtype=torch.long)
        org_hash = torch.tensor(stable_label_id(organ), dtype=torch.long)
        mid_ct_hash = torch.tensor(
            stable_label_id(record_mid_cell_type(s, ct)), dtype=torch.long
        )
        source_hash = torch.tensor(stable_label_id(record_source_label(s)), dtype=torch.long)
        mid_ct = record_mid_cell_type(s, ct)
        hierarchy_fine_idx = torch.tensor(
            self.hierarchy_label_maps.get("fine", {}).get(ct, -100), dtype=torch.long
        )
        hierarchy_mid_idx = torch.tensor(
            self.hierarchy_label_maps.get("mid", {}).get(mid_ct, -100), dtype=torch.long
        )
        hierarchy_organ_idx = torch.tensor(
            self.hierarchy_label_maps.get("organ", {}).get(organ, -100), dtype=torch.long
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "uce_emb": uce,
            "ct_idx": ct_idx,
            "org_idx": org_idx,
            "ct_hash": ct_hash,
            "org_hash": org_hash,
            "mid_ct_hash": mid_ct_hash,
            "source_hash": source_hash,
            "hierarchy_fine_idx": hierarchy_fine_idx,
            "hierarchy_mid_idx": hierarchy_mid_idx,
            "hierarchy_organ_idx": hierarchy_organ_idx,
        }


class ExtraTaskDatasetV3(Dataset):
    """Independent Task2/Task3 samples mixed into projector training.

    Records do not need to share cell IDs with Task1 SFT. Each record must
    carry its own embedding (`embedding` or `embed_path`) and one task label.
    Supported task types:
      - trajectory / trajectory_prediction / cell_trajectory
      - grn / grn_prediction
      - classification / task1
      - preformatted / custom with explicit `prompt` and `target`
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 1024,
        num_cell_tokens: int = 32,
        repo_root: str | None = None,
        max_records: int | None = None,
        seed: int = 42,
        expected_dim: int | None = None,
        grn_answer_mode: str = "reasoning",
    ):
        super().__init__()
        self.data_path = str(data_path)
        self.json_path = Path(data_path).resolve()
        self.repo_root = Path(repo_root).resolve() if repo_root else self.json_path.parents[1]
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.num_cell_tokens = int(num_cell_tokens)
        self.grn_answer_mode = str(grn_answer_mode or "reasoning")
        self._rng = random.Random(seed)

        records = _load_extra_task_records(data_path)
        if max_records and len(records) > max_records:
            records = self._rng.sample(records, int(max_records))
        self.records = records
        counts: dict[str, int] = {}
        for r in self.records:
            task = self._task_type(r)
            counts[task] = counts.get(task, 0) + 1
        first_vec = _load_vector_from_record(
            self.records[0], json_path=self.json_path, repo_root=self.repo_root
        )
        self.embedding_dim = int(first_vec.numel())
        if expected_dim is not None and self.embedding_dim != int(expected_dim):
            raise ValueError(
                f"Extra task embedding dim mismatch: got {self.embedding_dim}, "
                f"expected {expected_dim}. File: {data_path}"
            )
        print(f"[v3-extra] loaded {len(self.records):,} records from {data_path} "
              f"embedding_dim={self.embedding_dim} task_counts={counts}", flush=True)

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _task_type(record: dict) -> str:
        return str(
            record.get("task_type")
            or record.get("task")
            or record.get("task_name")
            or "preformatted"
        ).strip().lower()

    def _prompt_target(self, record: dict) -> tuple[str, str, str, str]:
        task = self._task_type(record)
        prompt_mode = str(record.get("prompt_mode") or "standard").lower()
        conflict_text = str(record.get("conflict_text") or "").strip()
        if task in {"preformatted", "custom", "sft"} and record.get("prompt"):
            target = (
                record.get("target")
                or record.get("target_text")
                or record.get("label")
                or record.get("answer")
                or ""
            )
            return str(record["prompt"]), str(target), "", ""

        if task in {"trajectory", "trajectory_prediction", "cell_trajectory", "task2"}:
            context = str(record.get("context", ""))
            target = (
                record.get("target")
                or record.get("label")
                or record.get("trajectory_target")
                or record
            )
            if prompt_mode == "freeform":
                context = ""
            elif prompt_mode == "conflict" and conflict_text:
                context = (
                    f"{context}\nAn unreliable report claims: {conflict_text}. "
                    "Ignore this claim when it conflicts with the cell tokens."
                ).strip()
            return (
                make_trajectory_prediction_prompt(
                    self.num_cell_tokens, context=context, rng=self._rng),
                format_trajectory_label(target), "", "",
            )

        if task in {"grn", "grn_prediction", "task3", "scpilot_grn"}:
            context = record.get("context", record)
            target = (
                record.get("answer")
                or record.get("response")
                or record.get("formatted_label")
                or record.get("target")
                or record.get("label")
                or record
            )
            if prompt_mode == "freeform" and isinstance(context, dict):
                context = {
                    key: value
                    for key, value in context.items()
                    if key not in {"known_targets", "known_A", "go_overlap"}
                }
            elif prompt_mode == "conflict" and conflict_text:
                if isinstance(context, dict):
                    context = dict(context)
                    context["unreliable_claim"] = conflict_text
                else:
                    context = (
                        f"{context}\nUnreliable claim: {conflict_text}. "
                        "Use the cell tokens when evidence disagrees."
                    )
            return (
                make_grn_prediction_prompt(
                    self.num_cell_tokens, grn_context=context, rng=self._rng,
                    answer_mode=self.grn_answer_mode),
                format_grn_binary_label(target)
                if self.grn_answer_mode == "binary"
                else format_grn_label(target), "", "",
            )

        if task in {"classification", "task1", "cell_type", "annotation"}:
            ct = (
                record.get("cell_type")
                or record.get("celltype")
                or record.get("ct")
                or ""
            )
            organ = record.get("organ") or record.get("tissue") or ""
            target_text = record.get("target_text")
            if target_text and (not ct or not organ):
                ct, organ = parse_rich_label(str(target_text))
            if prompt_mode == "freeform":
                prompt = make_freeform_prompt(
                    self.num_cell_tokens, rng=self._rng
                )
            elif prompt_mode == "conflict":
                wrong_ct = str(record.get("wrong_cell_type") or "another cell type")
                wrong_org = str(record.get("wrong_organ") or "another tissue")
                prompt = make_conflict_prompt(
                    self.num_cell_tokens,
                    wrong_cell_type=wrong_ct,
                    wrong_organ=wrong_org,
                    rng=self._rng,
                )
            else:
                candidates = record.get("candidate_cell_types")
                if candidates:
                    cell_tokens = " ".join(
                        [CELL_PLACEHOLDER] * self.num_cell_tokens
                    )
                    candidate_text = ", ".join(map(str, candidates))
                    prompt = (
                        "<|im_start|>system\nYou are a single-cell biology expert."
                        "<|im_end|>\n<|im_start|>user\n"
                        f"Cell embedding tokens: {cell_tokens}\n\n"
                        f"Candidate cell types: {candidate_text}\n"
                        f"Candidate organ: {organ}\n\n"
                        "Identify the cell type and organ. Reply as: "
                        "cell_type | organ.<|im_end|>\n"
                        "<|im_start|>assistant\n"
                    )
                else:
                    prompt = make_freeform_prompt(
                        self.num_cell_tokens, rng=self._rng
                    )
            return prompt, format_label(str(ct), str(organ), TASK_CT_ORGAN), str(ct), str(organ)

        if record.get("prompt"):
            target = (
                record.get("target")
                or record.get("target_text")
                or record.get("label")
                or record.get("answer")
                or ""
            )
            return str(record["prompt"]), str(target), "", ""

        raise ValueError(
            f"Unsupported extra task_type={task!r}; provide prompt/target for custom tasks"
        )

    def __getitem__(self, idx):
        record = self.records[idx]
        uce = _load_vector_from_record(
            record, json_path=self.json_path, repo_root=self.repo_root
        )
        prompt, target, ct, organ = self._prompt_target(record)
        full = prompt + " " + target + self.tokenizer.eos_token
        enc = self.tokenizer(full, max_length=self.max_length,
                             padding="max_length", truncation=True,
                             return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        prompt_enc = self.tokenizer(prompt, max_length=self.max_length,
                                    truncation=True, return_tensors="pt")
        prompt_len = prompt_enc["input_ids"].shape[1]
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        ct_idx = torch.tensor(CT_TO_IDX.get(ct, -100), dtype=torch.long)
        org_idx = torch.tensor(ORG_TO_IDX.get(organ, -100), dtype=torch.long)
        ct_hash = torch.tensor(stable_label_id(ct), dtype=torch.long)
        org_hash = torch.tensor(stable_label_id(organ), dtype=torch.long)
        mid_ct_hash = torch.tensor(
            stable_label_id(record_mid_cell_type(record, ct)), dtype=torch.long
        )
        source_hash = torch.tensor(
            stable_label_id(record_source_label(record)), dtype=torch.long
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "uce_emb": uce,
            "ct_idx": ct_idx,
            "org_idx": org_idx,
            "ct_hash": ct_hash,
            "org_hash": org_hash,
            "mid_ct_hash": mid_ct_hash,
            "source_hash": source_hash,
            "hierarchy_fine_idx": torch.tensor(-100, dtype=torch.long),
            "hierarchy_mid_idx": torch.tensor(-100, dtype=torch.long),
            "hierarchy_organ_idx": torch.tensor(-100, dtype=torch.long),
        }


class HierarchicalMixtureDataset(Dataset):
    """Deterministic task/source-balanced virtual dataset.

    Hugging Face Trainer may still apply its normal distributed sampler. Each
    virtual index maps to a precomputed source/index pair, so DDP ranks receive
    disjoint examples while preserving the requested global mixture.
    """

    def __init__(
        self,
        task1_dataset,
        extra_dataset: ExtraTaskDatasetV3,
        epoch_samples: int,
        task_probs: tuple[float, float, float],
        source_alpha: float,
        seed: int,
    ):
        self.datasets = [task1_dataset, extra_dataset]
        self.epoch_samples = int(epoch_samples)
        if self.epoch_samples <= 0:
            raise ValueError("mixture_epoch_samples must be positive")

        probs = np.asarray(task_probs, dtype=np.float64)
        if probs.shape != (3,) or np.any(probs < 0) or probs.sum() <= 0:
            raise ValueError("task_mixture must contain three non-negative values")
        probs /= probs.sum()

        pools: dict[str, dict[str, list[int]]] = {
            "task1": {"gtex": list(range(len(task1_dataset)))},
            "task2": {},
            "task3": {},
        }
        for idx, record in enumerate(extra_dataset.records):
            task = extra_dataset._task_type(record)
            if task in {"classification", "task1", "cell_type", "annotation"}:
                task_key = "task1"
            elif task in {
                "trajectory", "trajectory_prediction", "cell_trajectory", "task2"
            }:
                task_key = "task2"
            elif task in {"grn", "grn_prediction", "task3", "scpilot_grn"}:
                task_key = "task3"
            else:
                task_key = str(record.get("mixture_task") or "")
            if task_key not in pools:
                continue
            source = str(
                record.get("source_dataset")
                or record.get("dataset")
                or record.get("source")
                or "extra"
            )
            pools[task_key].setdefault(source, []).append(idx)

        for task_key, prob in zip(("task1", "task2", "task3"), probs):
            if prob > 0 and not pools[task_key]:
                raise ValueError(f"requested {task_key} probability but pool is empty")

        rng = np.random.default_rng(seed)
        task_keys = np.asarray(["task1", "task2", "task3"])
        schedule = []
        realized_tasks = {"task1": 0, "task2": 0, "task3": 0}
        realized_sources: dict[str, int] = {}
        for task_key in rng.choice(task_keys, size=self.epoch_samples, p=probs):
            source_pools = pools[str(task_key)]
            sources = sorted(source_pools)
            sizes = np.asarray(
                [len(source_pools[source]) for source in sources],
                dtype=np.float64,
            )
            source_probs = np.power(sizes, float(source_alpha))
            source_probs /= source_probs.sum()
            source = str(rng.choice(sources, p=source_probs))
            local_idx = int(rng.choice(source_pools[source]))
            dataset_idx = 0 if source == "gtex" and task_key == "task1" else 1
            schedule.append((dataset_idx, local_idx))
            realized_tasks[str(task_key)] += 1
            source_key = f"{task_key}:{source}"
            realized_sources[source_key] = realized_sources.get(source_key, 0) + 1

        self.schedule = schedule
        self.realized_tasks = realized_tasks
        self.realized_sources = realized_sources
        print(
            f"[v3-mixture] samples={self.epoch_samples:,} "
            f"tasks={self.realized_tasks} sources={self.realized_sources}",
            flush=True,
        )

    def __len__(self):
        return len(self.schedule)

    def __getitem__(self, idx):
        dataset_idx, local_idx = self.schedule[idx]
        return self.datasets[dataset_idx][local_idx]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ProjectorTrainerV3(Trainer):
    def __init__(self, *a, aux_cls_weight: float = 1.0,
                 pooled_aux_weight: float = 0.0,
                 supcon_weight: float = 0.0,
                 mid_supcon_weight: float = 0.0,
                 fine_within_source_supcon: bool = False,
                 relational_distill_weight: float = 0.0,
                 hierarchy_weight: float = 0.0,
                 hierarchy_mid_weight: float = 0.5,
                 hierarchy_organ_weight: float = 0.2,
                 variance_weight: float = 0.0,
                 covariance_weight: float = 0.0,
                 supcon_temperature: float = 0.2,
                 compute_dtype: torch.dtype = torch.bfloat16,
                 log_interval: int = 20, **kw):
        super().__init__(*a, **kw)
        self.aux_cls_weight = aux_cls_weight
        self._use_aux = aux_cls_weight > 0.0
        self.pooled_aux_weight = pooled_aux_weight
        self.supcon_weight = supcon_weight
        self.mid_supcon_weight = mid_supcon_weight
        self.fine_within_source_supcon = fine_within_source_supcon
        self.relational_distill_weight = relational_distill_weight
        self.hierarchy_weight = hierarchy_weight
        self.hierarchy_mid_weight = hierarchy_mid_weight
        self.hierarchy_organ_weight = hierarchy_organ_weight
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.supcon_temperature = supcon_temperature
        self.compute_dtype = compute_dtype
        self._dbg_step = 0
        self._log_interval = log_interval

    def _move_model_to_device(self, model, device):
        pass

    @staticmethod
    def _ce_pair(ct_logits, org_logits, ct_idx, org_idx, device):
        valid_ct = ct_idx.ne(-100)
        valid_org = org_idx.ne(-100)
        ct_loss = (
            F.cross_entropy(ct_logits[valid_ct].float(), ct_idx[valid_ct])
            if valid_ct.any()
            else torch.tensor(0.0, device=device)
        )
        org_loss = (
            F.cross_entropy(org_logits[valid_org].float(), org_idx[valid_org])
            if valid_org.any()
            else torch.tensor(0.0, device=device)
        )
        return ct_loss + org_loss

    @staticmethod
    def _masked_ce(logits, labels, device):
        if logits is None or labels is None:
            return torch.tensor(0.0, device=device)
        valid = labels.ne(-100)
        if not valid.any():
            return torch.tensor(0.0, device=device)
        return F.cross_entropy(logits[valid].float(), labels[valid])

    def _supervised_contrastive_loss(self, features, labels):
        valid = labels.ne(-100)
        features = features[valid].float()
        labels = labels[valid]
        if features.size(0) < 2:
            return features.new_tensor(0.0)
        features = F.normalize(features, dim=1)
        logits = features @ features.T / self.supcon_temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        eye = torch.eye(features.size(0), dtype=torch.bool, device=features.device)
        positive = labels[:, None].eq(labels[None, :]) & ~eye
        return self._supcon_from_positive(logits, positive, eye)

    def _within_source_contrastive_loss(self, features, labels, sources):
        valid = labels.ne(-100) & sources.ne(-100)
        features = features[valid].float()
        labels = labels[valid]
        sources = sources[valid]
        if features.size(0) < 2:
            return features.new_tensor(0.0)
        features = F.normalize(features, dim=1)
        logits = features @ features.T / self.supcon_temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        eye = torch.eye(features.size(0), dtype=torch.bool, device=features.device)
        positive = (
            labels[:, None].eq(labels[None, :])
            & sources[:, None].eq(sources[None, :])
            & ~eye
        )
        return self._supcon_from_positive(logits, positive, eye)

    @staticmethod
    def _supcon_from_positive(logits, positive, eye):
        if not positive.any():
            return logits.new_tensor(0.0)
        exp_logits = torch.exp(logits) * (~eye).float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        per_sample = -(log_prob * positive.float()).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
        return per_sample[positive.any(dim=1)].mean()

    @staticmethod
    def _variance_loss(features, target_std=1.0, eps=1e-4):
        features = features.float()
        if features.size(0) < 2:
            return features.new_tensor(0.0)
        std = torch.sqrt(features.var(dim=0, unbiased=False) + eps)
        return F.relu(target_std - std).mean()

    @staticmethod
    def _covariance_loss(features):
        features = features.float()
        n, d = features.shape
        if n < 2:
            return features.new_tensor(0.0)
        centered = features - features.mean(dim=0, keepdim=True)
        cov = centered.T @ centered / (n - 1)
        off_diag = cov.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()
        return off_diag.pow(2).sum() / d

    @staticmethod
    def _relational_distillation_loss(projector_features, uce_features):
        projector_features = projector_features.float()
        uce_features = uce_features.float()
        n = projector_features.size(0)
        if n < 2:
            return projector_features.new_tensor(0.0)
        proj_norm = F.normalize(projector_features, dim=1)
        uce_norm = F.normalize(uce_features, dim=1)
        proj_sim = proj_norm @ proj_norm.T
        uce_sim = uce_norm @ uce_norm.T
        eye = torch.eye(n, dtype=torch.bool, device=projector_features.device)
        return F.smooth_l1_loss(proj_sim[~eye], uce_sim[~eye])

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        core_model = model.module if hasattr(model, "module") else model
        device = core_model.input_device()
        dtype = self.compute_dtype
        uce_emb = inputs.pop("uce_emb").to(device, dtype=dtype)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = inputs["labels"].to(device)
        ct_idx = inputs.get("ct_idx")
        org_idx = inputs.get("org_idx")
        ct_hash = inputs.get("ct_hash")
        org_hash = inputs.get("org_hash")
        mid_ct_hash = inputs.get("mid_ct_hash")
        source_hash = inputs.get("source_hash")
        hierarchy_fine_idx = inputs.get("hierarchy_fine_idx")
        hierarchy_mid_idx = inputs.get("hierarchy_mid_idx")
        hierarchy_organ_idx = inputs.get("hierarchy_organ_idx")
        if ct_idx is not None:
            ct_idx = ct_idx.to(device)
        if org_idx is not None:
            org_idx = org_idx.to(device)
        if ct_hash is not None:
            ct_hash = ct_hash.to(device)
        if org_hash is not None:
            org_hash = org_hash.to(device)
        if mid_ct_hash is not None:
            mid_ct_hash = mid_ct_hash.to(device)
        if source_hash is not None:
            source_hash = source_hash.to(device)
        if hierarchy_fine_idx is not None:
            hierarchy_fine_idx = hierarchy_fine_idx.to(device)
        if hierarchy_mid_idx is not None:
            hierarchy_mid_idx = hierarchy_mid_idx.to(device)
        if hierarchy_organ_idx is not None:
            hierarchy_organ_idx = hierarchy_organ_idx.to(device)

        outputs, aux_payload = core_model.forward_with_aux_all(
            input_ids=input_ids, attention_mask=attention_mask,
            uce_emb=uce_emb, labels=labels)
        raw_ct_logits, raw_org_logits = aux_payload["raw"]
        pooled_ct_logits, pooled_org_logits = aux_payload["pooled"]
        pooled_features = aux_payload["pooled_features"]
        hierarchy_logits = aux_payload.get("hierarchy")

        sft_loss = outputs.loss
        raw_aux_loss = torch.tensor(0.0, device=device)
        pooled_aux_loss = torch.tensor(0.0, device=device)
        supcon_loss = torch.tensor(0.0, device=device)
        mid_supcon_loss = torch.tensor(0.0, device=device)
        relational_distill_loss = torch.tensor(0.0, device=device)
        hierarchy_fine_loss = torch.tensor(0.0, device=device)
        hierarchy_mid_loss = torch.tensor(0.0, device=device)
        hierarchy_organ_loss = torch.tensor(0.0, device=device)
        variance_loss = torch.tensor(0.0, device=device)
        covariance_loss = torch.tensor(0.0, device=device)
        if ct_idx is not None and org_idx is not None:
            if self.aux_cls_weight > 0.0:
                raw_aux_loss = self._ce_pair(
                    raw_ct_logits, raw_org_logits, ct_idx, org_idx, device
                )
                if not torch.isfinite(raw_aux_loss):
                    print(
                        f"[WARN] non-finite raw aux loss at debug step {self._dbg_step + 1}; "
                        "dropping raw aux term for this batch",
                        flush=True,
                    )
                    raw_aux_loss = torch.tensor(0.0, device=device)

            if self.pooled_aux_weight > 0.0:
                pooled_aux_loss = self._ce_pair(
                    pooled_ct_logits, pooled_org_logits, ct_idx, org_idx, device
                )
            if self.supcon_weight > 0.0:
                contrast_ct = ct_hash if ct_hash is not None else ct_idx
                contrast_org = org_hash if org_hash is not None else org_idx
                if self.fine_within_source_supcon and source_hash is not None:
                    supcon_loss = self._within_source_contrastive_loss(
                        pooled_features, contrast_ct, source_hash
                    )
                else:
                    supcon_loss = self._supervised_contrastive_loss(pooled_features, contrast_ct)
                if supcon_loss.item() == 0.0:
                    supcon_loss = self._supervised_contrastive_loss(pooled_features, contrast_org)
            if self.mid_supcon_weight > 0.0:
                contrast_mid = mid_ct_hash if mid_ct_hash is not None else ct_hash
                if contrast_mid is not None:
                    mid_supcon_loss = self._supervised_contrastive_loss(
                        pooled_features, contrast_mid
                    )

        if self.variance_weight > 0.0:
            variance_loss = self._variance_loss(pooled_features)
        if self.covariance_weight > 0.0:
            covariance_loss = self._covariance_loss(pooled_features)
        if self.relational_distill_weight > 0.0:
            relational_distill_loss = self._relational_distillation_loss(
                pooled_features, uce_emb
            )
        if self.hierarchy_weight > 0.0:
            if hierarchy_logits is None:
                raise RuntimeError(
                    "hierarchy_weight > 0 but Projector hierarchy heads are disabled"
                )
            hierarchy_fine_loss = self._masked_ce(
                hierarchy_logits[0], hierarchy_fine_idx, device
            )
            hierarchy_mid_loss = self._masked_ce(
                hierarchy_logits[1], hierarchy_mid_idx, device
            )
            hierarchy_organ_loss = self._masked_ce(
                hierarchy_logits[2], hierarchy_organ_idx, device
            )
            hierarchy_loss = (
                hierarchy_fine_loss
                + self.hierarchy_mid_weight * hierarchy_mid_loss
                + self.hierarchy_organ_weight * hierarchy_organ_loss
            )
        else:
            hierarchy_loss = torch.tensor(0.0, device=device)

        total = (
            sft_loss
            + self.aux_cls_weight * raw_aux_loss
            + self.pooled_aux_weight * pooled_aux_loss
            + self.supcon_weight * supcon_loss
            + self.mid_supcon_weight * mid_supcon_loss
            + self.relational_distill_weight * relational_distill_loss
            + self.hierarchy_weight * hierarchy_loss
            + self.variance_weight * variance_loss
            + self.covariance_weight * covariance_loss
        )

        self._dbg_step += 1
        if self._dbg_step % self._log_interval == 0:
            print(f"[Loss step {self._dbg_step}] sft={sft_loss.item():.4f}  "
                  f"raw_aux={raw_aux_loss.item():.4f}  "
                  f"pooled_aux={pooled_aux_loss.item():.4f}  "
                  f"supcon={supcon_loss.item():.4f}  "
                  f"mid_supcon={mid_supcon_loss.item():.4f}  "
                  f"rel={relational_distill_loss.item():.4f}  "
                  f"hier_f={hierarchy_fine_loss.item():.4f}  "
                  f"hier_m={hierarchy_mid_loss.item():.4f}  "
                  f"hier_o={hierarchy_organ_loss.item():.4f}  "
                  f"var={variance_loss.item():.4f}  "
                  f"cov={covariance_loss.item():.4f}", flush=True)

        return (total, outputs) if return_outputs else total


class LogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        if logs and state.is_local_process_zero:
            loss = logs.get("loss"); lr = logs.get("learning_rate"); ep = logs.get("epoch")
            if loss is not None:
                print(f"[Step {state.global_step:>6d}] train_loss={loss:.4f} "
                      f"lr={lr:.2e} epoch={ep:.3f}", flush=True)


class RunStatusCallback(TrainerCallback):
    """Persist machine-readable status for long runs and future plotting."""

    def __init__(self, output_dir: str, model_ref=None, save_every_steps: int = 0,
                 save_latest_every_steps: int = 0):
        self.output_dir = output_dir
        self.model_ref = model_ref
        self.save_every_steps = int(save_every_steps or 0)
        self.save_latest_every_steps = int(save_latest_every_steps or 0)
        self.state_path = os.path.join(output_dir, "training_state.json")
        self.train_metrics_path = os.path.join(output_dir, "train_metrics.jsonl")
        self.checkpoint_events_path = os.path.join(output_dir, "checkpoint_events.jsonl")
        self._last_milestone_step = -1
        self._last_latest_step = -1
        self._state_extra: dict = {}

    def _save_projector_atomic(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.pid{os.getpid()}.tmp"
        self.model_ref.save_projector(tmp)
        os.replace(tmp, path)

    def _write_state(self, state, status: str, extra: dict | None = None) -> None:
        payload = {
            "status": status,
            "updated_at": now_iso(),
            "global_step": int(state.global_step),
            "epoch": float(state.epoch or 0.0),
            "max_steps": int(state.max_steps or 0),
            "best_metric": state.best_metric,
            "best_model_checkpoint": state.best_model_checkpoint,
            "is_local_process_zero": bool(state.is_local_process_zero),
        }
        payload.update(self._state_extra)
        if extra:
            self._state_extra.update(extra)
            payload.update(extra)
        write_json(self.state_path, payload)

    def on_train_begin(self, args, state, control, **kw):
        if state.is_local_process_zero:
            self._write_state(state, "running")

    def on_log(self, args, state, control, logs=None, **kw):
        if logs and state.is_local_process_zero:
            record = {"time": now_iso(), "step": int(state.global_step)}
            record.update({k: float(v) if isinstance(v, (int, float)) else v
                           for k, v in logs.items()})
            append_jsonl(self.train_metrics_path, record)
            self._write_state(state, "running", {"latest_train_log": record})

    def on_step_end(self, args, state, control, **kw):
        if not state.is_local_process_zero:
            return
        if state.global_step == 0:
            return
        step = int(state.global_step)
        extra: dict = {}

        if self.save_latest_every_steps > 0 and step != self._last_latest_step:
            if step % self.save_latest_every_steps == 0:
                latest_dir = os.path.join(self.output_dir, "latest")
                latest_path = os.path.join(latest_dir, "projector.pt")
                self._save_projector_atomic(latest_path)
                latest_meta = {
                    "time": now_iso(),
                    "step": step,
                    "epoch": float(state.epoch or 0.0),
                    "path": latest_path,
                    "kind": "latest",
                    "file": file_info(latest_path),
                }
                write_json(os.path.join(latest_dir, "metadata.json"), latest_meta)
                append_jsonl(self.checkpoint_events_path, latest_meta)
                self._last_latest_step = step
                extra.update({
                    "latest_projector_checkpoint": latest_path,
                    "latest_projector_step": step,
                })

        if self.save_every_steps > 0 and step != self._last_milestone_step:
            if step % self.save_every_steps == 0:
                ckpt_dir = os.path.join(self.output_dir, "projector_checkpoints",
                                        f"step_{step:06d}")
                ckpt_path = os.path.join(ckpt_dir, "projector.pt")
                self._save_projector_atomic(ckpt_path)
                milestone_meta = {
                    "time": now_iso(),
                    "step": step,
                    "epoch": float(state.epoch or 0.0),
                    "path": ckpt_path,
                    "kind": "milestone",
                    "file": file_info(ckpt_path),
                }
                write_json(os.path.join(ckpt_dir, "metadata.json"), milestone_meta)
                append_jsonl(self.checkpoint_events_path, milestone_meta)
                self._last_milestone_step = step
                extra.update({
                    "latest_milestone_projector_checkpoint": ckpt_path,
                    "latest_milestone_projector_step": step,
                })
                print(f"[checkpoint] projector saved -> {ckpt_path}", flush=True)

        if extra:
            self._write_state(state, "running", extra)

    def on_train_end(self, args, state, control, **kw):
        if state.is_local_process_zero:
            self._write_state(state, "completed")


# ---------------------------------------------------------------------------
# Validation callback
# ---------------------------------------------------------------------------

class ValidationCallbackV3(TrainerCallback):
    def __init__(self, model, val_data_path, uce_val_path,
                 num_cell_tokens=16, max_length=768,
                 eval_every_steps=500, max_eval_samples=200,
                 aux_cls_weight=1.0, diverse_prompts=False,
                 eval_conflict: bool = False,
                 compute_dtype: torch.dtype = torch.bfloat16):
        self.model_ref = model
        self.eval_every_steps = eval_every_steps
        self.max_eval_samples = max_eval_samples
        self.best_val_loss = float("inf")
        self.aux_cls_weight = aux_cls_weight
        self.use_aux = aux_cls_weight > 0.0
        self.diverse_prompts = diverse_prompts
        self.num_cell_tokens = num_cell_tokens
        self.eval_conflict = bool(eval_conflict)
        self.compute_dtype = compute_dtype
        self.val_metrics_path = None
        self.enabled = is_env_rank0()

        if not self.enabled:
            self._eval_ds = None
            self._eval_indices = []
            self.prompt = ""
            return

        self._eval_ds = CellSFTDatasetV3(
            sft_data_path=val_data_path, tokenizer=model.tokenizer,
            uce_cache_path=uce_val_path,
            max_length=max_length, num_cell_tokens=num_cell_tokens,
            diverse_prompts=False,
        )
        if len(self._eval_ds) > max_eval_samples:
            random.seed(42)
            self._eval_indices = random.sample(range(len(self._eval_ds)),
                                               max_eval_samples)
        else:
            self._eval_indices = list(range(len(self._eval_ds)))

        if diverse_prompts:
            self.prompt = make_freeform_prompt(num_cell_tokens, rng=random.Random(42))
            print(f"[Val] FIXED FREEFORM prompt for generation\n{self.prompt}",
                  flush=True)
        else:
            self.prompt = make_classification_prompt(num_cell_tokens,
                                                     task=TASK_CT_ORGAN)

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kw):
        if not state.is_local_process_zero:
            return
        if state.global_step == 0:
            return
        if not (state.global_step == 50 or
                state.global_step % self.eval_every_steps == 0):
            return

        model = self.model_ref
        was_training = model.training
        model.eval()
        device = model.input_device()
        tokenizer = model.tokenizer
        t0 = time.time()

        prompt_enc = tokenizer(self.prompt, return_tensors="pt")
        prompt_ids = prompt_enc["input_ids"].to(device)
        prompt_mask = prompt_enc["attention_mask"].to(device)
        prompt_len = prompt_ids.shape[1]

        total_sft = 0.0
        ct_correct = org_correct = n = 0
        ct_aux_correct = org_aux_correct = 0
        ct_aux_n = org_aux_n = 0
        # Conflict-prompt eval (does the model trust UCE tokens > wrong text?)
        ct_conflict_correct = org_conflict_correct = 0
        n_conflict = 0
        conflict_rng = random.Random(2027 + state.global_step)

        for idx_pos, di in enumerate(self._eval_indices):
            sample = self._eval_ds[di]
            uce_emb = sample["uce_emb"].unsqueeze(0).to(device, dtype=self.compute_dtype)
            ct_true = int(sample["ct_idx"].item())
            org_true = int(sample["org_idx"].item())
            gt_ct = CELL_TYPES[ct_true] if ct_true >= 0 else ""
            gt_org = ORGANS[org_true] if org_true >= 0 else ""
            gt_label = format_label(gt_ct, gt_org, TASK_CT_ORGAN)

            full_text = self.prompt + " " + gt_label + tokenizer.eos_token
            enc = tokenizer(full_text, max_length=sample["input_ids"].size(0),
                            truncation=True, padding="max_length",
                            return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            labels_t = input_ids.clone()
            labels_t[:, :prompt_len] = -100
            labels_t[attention_mask == 0] = -100

            out, aux_logits = model.forward_with_aux(
                input_ids=input_ids, attention_mask=attention_mask,
                uce_emb=uce_emb, labels=labels_t)
            ct_logits, org_logits = aux_logits

            sft_v = out.loss.item()
            if sft_v != sft_v:
                continue
            total_sft += sft_v
            if ct_true >= 0:
                if int(ct_logits.argmax(dim=-1).item()) == ct_true:
                    ct_aux_correct += 1
                ct_aux_n += 1
            if org_true >= 0:
                if int(org_logits.argmax(dim=-1).item()) == org_true:
                    org_aux_correct += 1
                org_aux_n += 1

            gen_ids = model.generate(
                input_ids=prompt_ids, attention_mask=prompt_mask,
                uce_emb=uce_emb, max_new_tokens=40, do_sample=False,
            )
            pla = prompt_ids.shape[1]
            new_tokens = gen_ids[0, pla:] if gen_ids.shape[1] > pla else gen_ids[0]
            pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            if idx_pos < 3:
                print(f"  [v {idx_pos}] gt='{gt_label}'  pred='{pred_text[:100]}'",
                      flush=True)

            pred_ct, pred_org = parse_target(pred_text)
            if not pred_ct:
                pred_ct = pred_text.strip()
            if fuzzy_match(gt_ct, pred_ct):
                ct_correct += 1
            if fuzzy_match(gt_org, pred_org):
                org_correct += 1
            n += 1

            # ---- Conflict-prompt eval: same UCE, with WRONG hint ----
            if self.eval_conflict and idx_pos < min(80, len(self._eval_indices)):
                if gt_ct and gt_org:
                    wct, worg = sample_wrong_label(gt_ct, gt_org, rng=conflict_rng)
                    cprompt = make_conflict_prompt(
                        self.num_cell_tokens, wrong_cell_type=wct,
                        wrong_organ=worg, rng=conflict_rng,
                    )
                    cenc = tokenizer(cprompt, return_tensors="pt")
                    cprompt_ids = cenc["input_ids"].to(device)
                    cprompt_mask = cenc["attention_mask"].to(device)
                    cgen_ids = model.generate(
                        input_ids=cprompt_ids, attention_mask=cprompt_mask,
                        uce_emb=uce_emb, max_new_tokens=40, do_sample=False,
                    )
                    cpla = cprompt_ids.shape[1]
                    cnew = cgen_ids[0, cpla:] if cgen_ids.shape[1] > cpla else cgen_ids[0]
                    cpred_text = tokenizer.decode(cnew, skip_special_tokens=True).strip()
                    cpred_ct, cpred_org = parse_target(cpred_text)
                    if not cpred_ct:
                        cpred_ct = cpred_text.strip()
                    if fuzzy_match(gt_ct, cpred_ct):
                        ct_conflict_correct += 1
                    if fuzzy_match(gt_org, cpred_org):
                        org_conflict_correct += 1
                    n_conflict += 1
                    if idx_pos < 2:
                        print(f"  [c {idx_pos}] wrong='{wct} | {worg}'  "
                              f"true='{gt_ct} | {gt_org}'  pred='{cpred_text[:80]}'",
                              flush=True)

        elapsed = time.time() - t0
        avg_sft = total_sft / max(n, 1)
        ct_acc = ct_correct / max(n, 1) * 100
        org_acc = org_correct / max(n, 1) * 100
        ct_aux_acc = ct_aux_correct / max(ct_aux_n, 1) * 100
        org_aux_acc = org_aux_correct / max(org_aux_n, 1) * 100
        val_record = {
            "time": now_iso(),
            "step": int(state.global_step),
            "sft": float(avg_sft),
            "ct_acc": float(ct_acc),
            "org_acc": float(org_acc),
            "ct_aux_acc": float(ct_aux_acc),
            "org_aux_acc": float(org_aux_acc),
            "n": int(n),
            "elapsed_sec": float(elapsed),
        }

        print(f"\n{'='*70}", flush=True)
        print(f"[Val @ step {state.global_step}]  sft={avg_sft:.4f}  "
              f"ct_acc={ct_acc:.1f}%  org_acc={org_acc:.1f}%  "
              f"(n={n}, {elapsed:.0f}s)", flush=True)
        print(f"  [Aux@UCE]  ct_aux_acc={ct_aux_acc:.1f}%  "
              f"org_aux_acc={org_aux_acc:.1f}%", flush=True)
        if self.eval_conflict and n_conflict > 0:
            ct_conflict_acc = ct_conflict_correct / n_conflict * 100
            org_conflict_acc = org_conflict_correct / n_conflict * 100
            val_record.update({
                "conflict_n": int(n_conflict),
                "conflict_ct_acc": float(ct_conflict_acc),
                "conflict_org_acc": float(org_conflict_acc),
            })
            print(f"  [Conflict prompt n={n_conflict}]  "
                  f"ct_acc={ct_conflict_acc:.1f}%  org_acc={org_conflict_acc:.1f}%  "
                  f"(higher = trusts cell tokens over wrong text)", flush=True)
        print(f"{'='*70}\n", flush=True)

        append_jsonl(os.path.join(args.output_dir, "val_metrics.jsonl"), val_record)
        write_json(os.path.join(args.output_dir, "latest_val_metrics.json"), val_record)

        val_loss = avg_sft
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_dir = os.path.join(args.output_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            model.save_projector(os.path.join(best_dir, "projector.pt"))
            print(f"  >> New best sft={val_loss:.4f} -> {best_dir}/", flush=True)
        if was_training:
            model.train()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    p = argparse.ArgumentParser()
    p.add_argument("--sft_data", required=True)
    p.add_argument("--uce_train", "--train_embedding_cache",
                   dest="uce_train", required=True)
    p.add_argument("--uce_val", "--val_embedding_cache",
                   dest="uce_val", required=True)
    p.add_argument("--llm_path", required=True)
    p.add_argument("--llm_device_map", default="single",
                   choices=["single", "local_rank", "ddp", "auto"],
                   help=(
                       "Use 'local_rank'/'ddp' with torchrun for true DDP. "
                       "Use 'auto' only to shard the frozen LLM when one GPU cannot fit it."
                   ))
    p.add_argument("--llm_dtype", default="bf16", choices=["bf16", "fp32"],
                   help="LLM/projector compute dtype. Use fp32 on H20 runtimes that SIGFPE in bf16.")
    p.add_argument("--cell_token_mode", choices=["resize", "literal"], default="resize",
                   help="'resize' adds a dedicated <CELL> token to Qwen and is "
                        "compatible with legacy successful checkpoints. 'literal' "
                        "avoids vocabulary resize for low-memory DDP, but requires "
                        "checkpoints trained with the same mode.")
    p.add_argument("--num_cell_tokens", type=int, default=16)
    p.add_argument("--input_emb_dim", type=int, default=None,
                   help="Input embedding dimension. If omitted, infer from --uce_train.")
    p.add_argument("--projector_hidden", type=int, default=512)
    p.add_argument("--num_mixer_layers", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--max_length", type=int, default=768)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--max_steps", type=int, default=-1,
                   help="Stop after this many optimizer steps. Use for true smoke runs.")
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Optional cap on Task1 SFT training samples before mixing extra tasks.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--aux_cls_weight", type=float, default=1.0)
    p.add_argument("--pooled_aux_weight", type=float, default=0.0,
                   help="CE weight for cell type/organ heads on mean-pooled projector tokens.")
    p.add_argument("--supcon_weight", type=float, default=0.0,
                   help="Supervised contrastive loss weight on mean-pooled projector tokens.")
    p.add_argument("--mid_supcon_weight", type=float, default=0.0,
                   help="Cross-source mid-level supervised contrastive loss weight.")
    p.add_argument("--fine_within_source_supcon", action="store_true",
                   help="Restrict fine-label SupCon positives to the same source/atlas.")
    p.add_argument("--relational_distill_weight", type=float, default=0.0,
                   help="Match batch-wise UCE cosine geometry with pooled projector cosine geometry.")
    p.add_argument("--hierarchy_weight", type=float, default=0.0,
                   help="Overall weight for fine+mid+organ CE on pooled Projector features.")
    p.add_argument("--hierarchy_mid_weight", type=float, default=0.5,
                   help="Mid-label CE ratio inside the hierarchy objective.")
    p.add_argument("--hierarchy_organ_weight", type=float, default=0.2,
                   help="Organ-label CE ratio inside the hierarchy objective.")
    p.add_argument("--variance_weight", type=float, default=0.0,
                   help="VICReg-style variance loss weight on mean-pooled projector tokens.")
    p.add_argument("--covariance_weight", type=float, default=0.0,
                   help="VICReg-style covariance loss weight on mean-pooled projector tokens.")
    p.add_argument("--supcon_temperature", type=float, default=0.2)
    p.add_argument("--eval_every_steps", type=int, default=500)
    p.add_argument("--max_eval_samples", type=int, default=200)
    p.add_argument("--disable_gradient_checkpointing", action="store_true",
                   help="Disable gradient checkpointing. Faster if memory allows.")
    p.add_argument("--diverse_prompts", action="store_true")
    p.add_argument("--diverse_weights", type=float, nargs=5,
                   default=list(DEFAULT_DIVERSE_WEIGHTS))
    p.add_argument("--perturb_dict_path", default=None)
    p.add_argument("--perturb_prob", type=float, default=0.0)
    # ---- v18 evidence-driven tasks (port from v2) ----
    p.add_argument("--train_topk_prefix", default=None,
                   help="Prefix of v2-format topk memmap (e.g. "
                        "data/cells_v2_train). Required for top-gene task.")
    p.add_argument("--top_gene_prob", type=float, default=0.0)
    p.add_argument("--gene_reference_prob", type=float, default=0.0,
                   help="Probability of the named gene-reference task. "
                        "Uses --train_topk_prefix for labels.")
    p.add_argument("--top_gene_k", type=int, default=5)
    p.add_argument("--trajectory_prediction_prob", "--cell_trajectory_prob",
                   dest="trajectory_prediction_prob", type=float, default=0.0,
                   help="Probability of the named cell-trajectory prediction task.")
    p.add_argument("--trajectory_data_path", "--cell_trajectory_data_path",
                   dest="trajectory_data_path", default=None,
                   help="JSON dict/list keyed by cell_id with cell-trajectory labels.")
    p.add_argument("--grn_prediction_prob", "--scpilot_grn_prob",
                   dest="grn_prediction_prob", type=float, default=0.0,
                   help="Probability of the named scPilot-style GRN TF-gene task.")
    p.add_argument("--grn_data_path", "--scpilot_grn_data_path",
                   dest="grn_data_path", default=None,
                   help="JSON dict/list keyed by cell_id with GRN TF-gene labels.")
    p.add_argument("--grn_answer_mode", choices=["reasoning", "binary"], default="reasoning",
                   help="Task3 target format: scPilot-style reasoning or strict 1/0 binary label.")
    p.add_argument("--extra_task_data_path", default=None,
                   help="JSON list/dict of independent Task2/Task3 samples. "
                        "Each record needs task_type plus embedding/embed_path "
                        "and target/context fields. These records do not need "
                        "to share cell_id with --sft_data.")
    p.add_argument("--extra_task_max_records", type=int, default=None,
                   help="Optional cap for independent extra-task records.")
    p.add_argument("--task_mixture", type=float, nargs=3, default=None,
                   metavar=("TASK1", "TASK2", "TASK3"),
                   help="Explicit hierarchical Task1/2/3 sampling weights.")
    p.add_argument("--mixture_epoch_samples", type=int, default=100000,
                   help="Virtual samples per mixture epoch.")
    p.add_argument("--source_temperature_alpha", type=float, default=0.5,
                   help="Within-task source weight exponent: 0=uniform, 1=proportional.")
    p.add_argument("--dataloader_num_workers", type=int, default=0,
                   help="Use 0 for shared large embedding caches.")
    p.add_argument("--conflict_prob", type=float, default=0.0)
    p.add_argument("--resume_projector", default=None)
    p.add_argument("--save_projector_every_steps", type=int, default=1000,
                   help="Save projector-only checkpoints every N optimizer steps. "
                        "Set 0 to disable.")
    p.add_argument("--save_latest_projector_every_steps", type=int, default=0,
                   help="Overwrite output_dir/latest/projector.pt every N optimizer "
                        "steps for resumability. Set 0 to disable.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    llm_dtype = torch.bfloat16 if args.llm_dtype == "bf16" else torch.float32

    if args.sft_data.endswith("_train_split.json"):
        train_json = args.sft_data
        val_json = args.sft_data.replace("_train_split.json", "_val_split.json")
        if not os.path.exists(val_json):
            raise FileNotFoundError(val_json)
    else:
        raise ValueError("sft_data should end with _train_split.json (use existing v17 splits)")

    hierarchy_label_maps = None
    if args.hierarchy_weight > 0.0:
        hierarchy_label_maps = build_hierarchy_label_maps(train_json)
        hierarchy_sizes = {
            key: len(mapping) for key, mapping in hierarchy_label_maps.items()
        }
        os.makedirs(args.output_dir, exist_ok=True)
        if is_env_rank0():
            write_json(
                os.path.join(args.output_dir, "hierarchy_label_encoder.json"),
                {
                    "source": train_json,
                    "label_to_idx": hierarchy_label_maps,
                    "n_classes": hierarchy_sizes,
                },
            )
        print(f"[v3] hierarchy vocab from train only: {hierarchy_sizes}", flush=True)

    input_emb_dim = args.input_emb_dim or infer_embedding_dim(args.uce_train)
    print(f"[v3] loading SingleCellLLMv3 ... input_emb_dim={input_emb_dim}", flush=True)
    print(f"[v3] cell_token_mode={args.cell_token_mode}", flush=True)
    print(f"[v3] llm_dtype={args.llm_dtype}", flush=True)
    model = SingleCellLLMv3(
        llm_path=args.llm_path,
        uce_dim=input_emb_dim,
        hidden=args.projector_hidden,
        num_cell_tokens=args.num_cell_tokens,
        num_mixer_layers=args.num_mixer_layers,
        num_heads=args.num_heads,
        llm_dtype=llm_dtype,
        llm_device_map=args.llm_device_map,
        cell_token_mode=args.cell_token_mode,
        n_hierarchy_fine=len(hierarchy_label_maps["fine"]) if hierarchy_label_maps else 0,
        n_hierarchy_mid=len(hierarchy_label_maps["mid"]) if hierarchy_label_maps else 0,
        n_hierarchy_organs=len(hierarchy_label_maps["organ"]) if hierarchy_label_maps else 0,
    )
    print(f"[v3] projector params = {model.projector.num_params()/1e6:.2f} M",
          flush=True)
    if args.resume_projector and os.path.exists(args.resume_projector):
        model.load_projector(args.resume_projector, strict=False)
        print(f"[v3] resumed projector from {args.resume_projector}", flush=True)

    perturb_dict = {}
    if args.perturb_dict_path and os.path.exists(args.perturb_dict_path):
        perturb_dict = json.load(open(args.perturb_dict_path))
        print(f"[v3] perturb dict: {len(perturb_dict)} cell_types", flush=True)

    train_ds = CellSFTDatasetV3(
        sft_data_path=train_json, tokenizer=model.tokenizer,
        uce_cache_path=args.uce_train,
        max_length=args.max_length, num_cell_tokens=args.num_cell_tokens,
        perturbation_dict=perturb_dict, perturb_prob=args.perturb_prob,
        diverse_prompts=args.diverse_prompts,
        diverse_weights=tuple(args.diverse_weights),
        topk_prefix=args.train_topk_prefix,
        top_gene_prob=args.top_gene_prob,
        gene_reference_prob=args.gene_reference_prob,
        trajectory_prediction_prob=args.trajectory_prediction_prob,
        grn_prediction_prob=args.grn_prediction_prob,
        grn_answer_mode=args.grn_answer_mode,
        conflict_prob=args.conflict_prob,
        top_gene_k=args.top_gene_k,
        trajectory_data_path=args.trajectory_data_path,
        grn_data_path=args.grn_data_path,
        hierarchy_label_maps=hierarchy_label_maps,
    )
    task1_train_dataset = train_ds
    if args.max_train_samples is not None and 0 <= int(args.max_train_samples) < len(train_ds):
        task1_train_dataset = Subset(train_ds, list(range(int(args.max_train_samples))))
        print(f"[v3] max_train_samples={args.max_train_samples:,}: "
              f"task1 {len(train_ds):,} -> {len(task1_train_dataset):,}",
              flush=True)

    extra_train_ds = None
    if args.extra_task_data_path:
        extra_train_ds = ExtraTaskDatasetV3(
            data_path=args.extra_task_data_path,
            tokenizer=model.tokenizer,
            max_length=args.max_length,
            num_cell_tokens=args.num_cell_tokens,
            max_records=args.extra_task_max_records,
            seed=args.seed,
            expected_dim=input_emb_dim,
            grn_answer_mode=args.grn_answer_mode,
        )
        if args.task_mixture is not None:
            train_dataset = HierarchicalMixtureDataset(
                task1_dataset=task1_train_dataset,
                extra_dataset=extra_train_ds,
                epoch_samples=args.mixture_epoch_samples,
                task_probs=tuple(args.task_mixture),
                source_alpha=args.source_temperature_alpha,
                seed=args.seed,
            )
            print(
                f"[v3] hierarchical mixture: task1={len(task1_train_dataset):,} "
                f"extra={len(extra_train_ds):,} virtual={len(train_dataset):,}",
                flush=True,
            )
        else:
            train_dataset = ConcatDataset([task1_train_dataset, extra_train_ds])
            print(f"[v3] mixed train dataset: task1={len(task1_train_dataset):,} "
                  f"extra={len(extra_train_ds):,} total={len(train_dataset):,}",
                  flush=True)
    else:
        train_dataset = task1_train_dataset

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = {
        "created_at": now_iso(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "argv": sys.argv,
        "args": vars(args),
        "paths": {
            "train_json": train_json,
            "val_json": val_json,
            "uce_train": args.uce_train,
            "uce_val": args.uce_val,
            "output_dir": args.output_dir,
            "train_topk_prefix": args.train_topk_prefix,
            "trajectory_data_path": args.trajectory_data_path,
            "grn_data_path": args.grn_data_path,
            "extra_task_data_path": args.extra_task_data_path,
        },
        "file_info": {
            "train_json": file_info(train_json),
            "val_json": file_info(val_json),
            "uce_train": file_info(args.uce_train),
            "uce_val": file_info(args.uce_val),
        },
        "data": {
            "train_samples": len(train_ds.samples),
            "task1_train_dataset_samples": int(len(task1_train_dataset)),
            "extra_task_samples": int(len(extra_train_ds) if extra_train_ds is not None else 0),
            "mixed_train_samples": int(len(train_dataset)),
            "train_matched_uce": int(sum(1 for s in train_ds.samples
                                         if s["cell_id"] in train_ds.cid_to_row)),
            "uce_train_shape": list(train_ds.uce_table.shape),
            "input_embedding_dim": int(input_emb_dim),
            "cell_token_mode": args.cell_token_mode,
            "cell_types": CELL_TYPES,
            "organs": ORGANS,
            "hierarchy_n_classes": (
                {key: len(value) for key, value in hierarchy_label_maps.items()}
                if hierarchy_label_maps else None
            ),
        },
        "tasks": {
            "top_gene_prob": args.top_gene_prob,
            "gene_reference_prob": args.gene_reference_prob,
            "trajectory_prediction_prob": args.trajectory_prediction_prob,
            "grn_prediction_prob": args.grn_prediction_prob,
            "grn_answer_mode": args.grn_answer_mode,
            "extra_task_data_path": args.extra_task_data_path,
            "extra_task_max_records": args.extra_task_max_records,
            "task_mixture": args.task_mixture,
            "mixture_epoch_samples": args.mixture_epoch_samples,
            "source_temperature_alpha": args.source_temperature_alpha,
            "max_steps": args.max_steps,
            "max_train_samples": args.max_train_samples,
            "aux_cls_weight": args.aux_cls_weight,
            "pooled_aux_weight": args.pooled_aux_weight,
            "supcon_weight": args.supcon_weight,
            "mid_supcon_weight": args.mid_supcon_weight,
            "fine_within_source_supcon": args.fine_within_source_supcon,
            "relational_distill_weight": args.relational_distill_weight,
            "hierarchy_weight": args.hierarchy_weight,
            "hierarchy_mid_weight": args.hierarchy_mid_weight,
            "hierarchy_organ_weight": args.hierarchy_organ_weight,
            "variance_weight": args.variance_weight,
            "covariance_weight": args.covariance_weight,
            "supcon_temperature": args.supcon_temperature,
            "conflict_prob": args.conflict_prob,
            "diverse_prompts": args.diverse_prompts,
            "diverse_weights": args.diverse_weights,
        },
        "visualization": {
            "train_metrics_jsonl": os.path.join(args.output_dir, "train_metrics.jsonl"),
            "val_metrics_jsonl": os.path.join(args.output_dir, "val_metrics.jsonl"),
            "latest_val_metrics_json": os.path.join(args.output_dir, "latest_val_metrics.json"),
            "training_state_json": os.path.join(args.output_dir, "training_state.json"),
            "best_projector": os.path.join(args.output_dir, "best", "projector.pt"),
            "final_projector": os.path.join(args.output_dir, "projector_final.pt"),
            "suggested_plots": [
                "train_loss_vs_step",
                "val_sft_vs_step",
                "val_cell_type_accuracy_vs_step",
                "val_organ_accuracy_vs_step",
                "aux_accuracy_vs_step",
                "checkpoint_timeline",
            ],
        },
    }
    if is_env_rank0():
        write_json(os.path.join(args.output_dir, "run_manifest.json"), manifest)
        write_json(os.path.join(args.output_dir, "visualization_payload.json"),
                   manifest["visualization"])
        print(f"[manifest] wrote {args.output_dir}/run_manifest.json", flush=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        logging_steps=10,
        save_strategy="no",
        bf16=(args.llm_dtype == "bf16"),
        gradient_checkpointing=(not args.disable_gradient_checkpointing),
        gradient_checkpointing_kwargs=None if args.disable_gradient_checkpointing else {"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        label_names=["labels"],
    )

    val_callback = ValidationCallbackV3(
        model=model, val_data_path=val_json, uce_val_path=args.uce_val,
        num_cell_tokens=args.num_cell_tokens, max_length=args.max_length,
        eval_every_steps=args.eval_every_steps,
        max_eval_samples=args.max_eval_samples,
        aux_cls_weight=args.aux_cls_weight,
        diverse_prompts=args.diverse_prompts,
        eval_conflict=(args.conflict_prob > 0),
        compute_dtype=llm_dtype,
    )
    trainer = ProjectorTrainerV3(
        model=model, args=training_args, train_dataset=train_dataset,
        aux_cls_weight=args.aux_cls_weight,
        pooled_aux_weight=args.pooled_aux_weight,
        supcon_weight=args.supcon_weight,
        mid_supcon_weight=args.mid_supcon_weight,
        fine_within_source_supcon=args.fine_within_source_supcon,
        relational_distill_weight=args.relational_distill_weight,
        hierarchy_weight=args.hierarchy_weight,
        hierarchy_mid_weight=args.hierarchy_mid_weight,
        hierarchy_organ_weight=args.hierarchy_organ_weight,
        variance_weight=args.variance_weight,
        covariance_weight=args.covariance_weight,
        supcon_temperature=args.supcon_temperature,
        compute_dtype=llm_dtype,
        callbacks=[
            LogCallback(),
            RunStatusCallback(args.output_dir, model_ref=model,
                              save_every_steps=args.save_projector_every_steps,
                              save_latest_every_steps=args.save_latest_projector_every_steps),
            val_callback,
        ],
    )

    print(f"\n{'='*70}\nSAGA v3.0  --  UCE -> Qwen SFT\n{'='*70}", flush=True)
    trainer.train()

    if trainer.is_world_process_zero():
        final_dir = args.output_dir
        os.makedirs(final_dir, exist_ok=True)
        model.save_projector(os.path.join(final_dir, "projector_final.pt"))
        print(f"Final saved to {final_dir}/projector_final.pt", flush=True)
        print(f"Best sft={val_callback.best_val_loss:.4f} at {final_dir}/best/",
              flush=True)


if __name__ == "__main__":
    main()
