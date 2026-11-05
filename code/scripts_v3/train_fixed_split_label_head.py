#!/usr/bin/env python
"""Train a dynamic label head on fixed train/val/test feature splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report


class Head(nn.Module):
    def __init__(self, dim: int, n_classes: int, kind: str, hidden: int, dropout: float):
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(dim, n_classes)
        elif kind == "mlp":
            self.net = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_classes),
            )
        else:
            raise ValueError(kind)

    def forward(self, x):
        return self.net(x)


def load_records(path: str) -> list[dict]:
    raw = json.load(open(path))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("records"), list):
        return raw["records"]
    raise ValueError(f"Unsupported records format: {path}")


def load_features(path: str) -> torch.Tensor:
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        x = data.get("features", data.get("embeddings"))
    else:
        x = data
    if x is None:
        raise ValueError(f"No features/embeddings in {path}")
    return x.float()


def get_label(record: dict, key: str) -> str:
    if key == "auto":
        answer = record.get("answer", {})
        answer = answer if isinstance(answer, dict) else {}
        for k in ("cell_type", "celltype", "mid_cell_type", "ann_level_3", "label"):
            v = record.get(k, answer.get(k))
            if v is not None and str(v).strip() and str(v).lower() != "nan":
                return str(v).strip()
        return ""
    if key == "organ":
        answer = record.get("answer", {})
        answer = answer if isinstance(answer, dict) else {}
        v = record.get("organ", record.get("tissue", answer.get("organ", "")))
        return str(v).strip()
    v = record.get(key, "")
    return str(v).strip()


def valid_label(x: str) -> bool:
    return bool(x) and x.lower() not in {"nan", "none", "null", "unknown"}


def normalize_features(x: torch.Tensor, center: bool, normalize: bool) -> torch.Tensor:
    x = x.float()
    if center:
        x = x - x.mean(dim=0, keepdim=True)
    if normalize:
        x = F.normalize(x, dim=1)
    return x.contiguous()


def metric_dict(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def focal_loss(logits, targets, weight=None, gamma=2.0, label_smoothing=0.0):
    ce = F.cross_entropy(logits, targets, weight=weight, reduction="none", label_smoothing=label_smoothing)
    pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
    return (((1.0 - pt) ** gamma) * ce).mean()


@torch.no_grad()
def predict(model, x, batch_size):
    model.eval()
    out = []
    for start in range(0, len(x), batch_size):
        out.append(model(x[start:start + batch_size]).argmax(1).detach().cpu())
    return torch.cat(out).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_records", required=True)
    p.add_argument("--val_records", required=True)
    p.add_argument("--test_records", required=True)
    p.add_argument("--train_features", required=True)
    p.add_argument("--val_features", required=True)
    p.add_argument("--test_features", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--label_key", default="cell_type")
    p.add_argument("--min_count", type=int, default=20)
    p.add_argument("--head", choices=["linear", "mlp"], default="mlp")
    p.add_argument("--loss", choices=["ce", "balanced_ce", "focal", "balanced_focal"], default="ce")
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--eval_batch_size", type=int, default=32768)
    p.add_argument("--center", action="store_true")
    p.add_argument("--no_normalize", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_records(args.train_records)
    val_records = load_records(args.val_records)
    test_records = load_records(args.test_records)
    x_train_all = load_features(args.train_features)
    x_val_all = load_features(args.val_features)
    x_test_all = load_features(args.test_features)
    if len(train_records) != len(x_train_all) or len(val_records) != len(x_val_all) or len(test_records) != len(x_test_all):
        raise SystemExit(
            f"length mismatch train={len(train_records)}/{len(x_train_all)} "
            f"val={len(val_records)}/{len(x_val_all)} test={len(test_records)}/{len(x_test_all)}"
        )

    train_labels_raw = np.asarray([get_label(r, args.label_key) for r in train_records])
    val_labels_raw = np.asarray([get_label(r, args.label_key) for r in val_records])
    test_labels_raw = np.asarray([get_label(r, args.label_key) for r in test_records])
    counts = Counter([x for x in train_labels_raw if valid_label(x)])
    keep = sorted([label for label, count in counts.items() if count >= args.min_count])
    label_to_id = {label: i for i, label in enumerate(keep)}
    if len(keep) < 2:
        raise SystemExit(f"not enough train labels after min_count={args.min_count}: {counts}")

    def select(records, labels_raw, x_all):
        idx = np.asarray([i for i, lab in enumerate(labels_raw) if lab in label_to_id], dtype=np.int64)
        y = np.asarray([label_to_id[labels_raw[i]] for i in idx], dtype=np.int64)
        return idx, y, x_all[idx]

    train_idx, y_train, x_train_cpu = select(train_records, train_labels_raw, x_train_all)
    val_idx, y_val, x_val_cpu = select(val_records, val_labels_raw, x_val_all)
    test_idx, y_test, x_test_cpu = select(test_records, test_labels_raw, x_test_all)
    if len(val_idx) == 0 or len(test_idx) == 0:
        raise SystemExit("val/test has zero closed-set rows under train label encoder")

    normalize = not args.no_normalize
    x_train_cpu = normalize_features(x_train_cpu, args.center, normalize)
    x_val_cpu = normalize_features(x_val_cpu, args.center, normalize)
    x_test_cpu = normalize_features(x_test_cpu, args.center, normalize)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train = x_train_cpu.to(device)
    x_val = x_val_cpu.to(device)
    x_test = x_test_cpu.to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    class_counts = np.bincount(y_train, minlength=len(keep))
    class_weight = None
    if args.loss in {"balanced_ce", "balanced_focal"}:
        class_weight = torch.tensor(class_counts.sum() / np.maximum(class_counts, 1) / len(class_counts), dtype=torch.float32, device=device)

    model = Head(x_train.shape[1], len(keep), args.head, args.hidden, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = {"epoch": -1, "val_macro_f1": -1.0, "state": None}
    stale = 0
    rng = np.random.default_rng(args.seed)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(y_train))
        total_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            batch = torch.tensor(order[start:start + args.batch_size], dtype=torch.long, device=device)
            logits = model(x_train[batch])
            target = y_train_t[batch]
            if args.loss in {"focal", "balanced_focal"}:
                loss = focal_loss(logits, target, weight=class_weight, gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
            else:
                loss = F.cross_entropy(logits, target, weight=class_weight, label_smoothing=args.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(batch)

        val_pred = predict(model, x_val, args.eval_batch_size)
        val_metrics = metric_dict(y_val, val_pred)
        if val_metrics["macro_f1"] > best["val_macro_f1"]:
            best = {"epoch": epoch, "val_macro_f1": val_metrics["macro_f1"], "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 25 == 0 or stale == 0:
            print(json.dumps({"epoch": epoch, "loss": total_loss / len(y_train), **{f"val_{k}": v for k, v in val_metrics.items()}}, sort_keys=True), flush=True)
        if stale >= args.patience:
            break

    model.load_state_dict(best["state"])
    val_pred = predict(model, x_val, args.eval_batch_size)
    test_pred = predict(model, x_test, args.eval_batch_size)
    val_metrics = metric_dict(y_val, val_pred)
    test_metrics = metric_dict(y_test, test_pred)

    id_to_label = {i: lab for lab, i in label_to_id.items()}
    summary = {
        "name": args.name,
        "label_key": args.label_key,
        "head": args.head,
        "loss": args.loss,
        "seed": args.seed,
        "n_classes": len(keep),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "train_total_records": len(train_records),
        "val_total_records": len(val_records),
        "test_total_records": len(test_records),
        "test_only_labels": sorted(set(test_labels_raw) - set(label_to_id)),
        "val_only_labels": sorted(set(val_labels_raw) - set(label_to_id)),
        "best_epoch": int(best["epoch"]),
        "val": val_metrics,
        "test": test_metrics,
        "features": {"train": args.train_features, "val": args.val_features, "test": args.test_features},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "label_encoder.json").write_text(json.dumps({"label_to_id": label_to_id, "id_to_label": id_to_label}, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "summary": summary}, out_dir / "best_head.pt")

    with open(out_dir / "test_predictions.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row", "cell_id", "gold", "pred"])
        writer.writeheader()
        for row, pred_id in zip(test_idx.tolist(), test_pred.tolist()):
            writer.writerow({
                "row": row,
                "cell_id": test_records[row].get("cell_id", ""),
                "gold": test_labels_raw[row],
                "pred": id_to_label[pred_id],
            })

    report = classification_report(
        y_test,
        test_pred,
        labels=list(range(len(keep))),
        target_names=keep,
        zero_division=0,
    )
    (out_dir / "test_classification_report.txt").write_text(report)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
