#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json

import numpy as np
from scipy.optimize import minimize


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = None
for p in [THIS_FILE.parent, *THIS_FILE.parent.parents]:
    if (p / "configs").is_dir() and ((p / "pipelines").is_dir() or (p / " pipelines").is_dir()):
        PROJECT_ROOT = p
        break
if PROJECT_ROOT is None:
    raise RuntimeError(f"Cannot locate project root from {THIS_FILE}")


def load_csv_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def rankdata_average(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        return None
    ranks = rankdata_average(y_score)
    sum_pos = float(np.sum(ranks[y_true == 1]))
    return float((sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def compute_auprc(y_true: np.ndarray, y_score: np.ndarray):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = int(np.sum(y_true == 1))
    if pos == 0:
        return None
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    recall = tp / pos
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def compute_brier(y_true: np.ndarray, y_prob: np.ndarray):
    return float(np.mean((np.asarray(y_prob, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)) ** 2))


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins=10):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    n = max(1, len(y_true))
    for i in range(int(n_bins)):
        lo = bins[i]
        hi = bins[i + 1]
        if i == int(n_bins) - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        ece += float(np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def compute_confusion(y_true: np.ndarray, y_prob: np.ndarray, threshold=0.5):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = (np.asarray(y_prob, dtype=np.float64) >= float(threshold)).astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"threshold": float(threshold), "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray):
    return {
        "num_rows": int(len(y_true)),
        "positive_ratio": float(np.mean(y_true)) if len(y_true) else None,
        "auroc": compute_auroc(y_true, y_prob),
        "auprc": compute_auprc(y_true, y_prob),
        "brier": compute_brier(y_true, y_prob) if len(y_true) else None,
        "ece": compute_ece(y_true, y_prob) if len(y_true) else None,
        "confusion_at_0p5": compute_confusion(y_true, y_prob, threshold=0.5) if len(y_true) else None,
    }


def platt_loss(params, scores, y, l2):
    a, b = params
    probs = np.clip(sigmoid(a * scores + b), 1e-9, 1.0 - 1e-9)
    nll = -np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs))
    reg = 0.5 * l2 * float(a * a + b * b)
    return float(nll + reg)


def platt_grad(params, scores, y, l2):
    a, b = params
    probs = sigmoid(a * scores + b)
    err = probs - y
    ga = float(np.mean(err * scores) + l2 * a)
    gb = float(np.mean(err) + l2 * b)
    return np.asarray([ga, gb], dtype=np.float64)


def fit_platt(scores: np.ndarray, y: np.ndarray, l2=1e-2, maxiter=500):
    x0 = np.asarray([1.0, 0.0], dtype=np.float64)
    result = minimize(
        fun=platt_loss,
        x0=x0,
        jac=platt_grad,
        args=(scores, y, l2),
        method="L-BFGS-B",
        options={"maxiter": int(maxiter)},
    )
    if not result.success:
        raise RuntimeError(f"platt optimization failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64), result


def main():
    ap = argparse.ArgumentParser(description="Fit Platt scaling on a baseline model's validation predictions.")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--task", required=True, choices=["pre", "post"])
    ap.add_argument("--label_col", default="y_true")
    ap.add_argument("--prob_col", default="p_hat")
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_manifest = load_json(model_dir / "run_manifest.json")
    prediction_rows = load_csv_rows(model_dir / "predictions.csv")
    by_split = {"train": [], "val": [], "test": []}
    for row in prediction_rows:
        split = str(row.get("split", ""))
        if split in by_split:
            by_split[split].append(row)

    def to_arrays(rows):
        y = np.asarray([int(float(r[args.label_col])) for r in rows], dtype=np.float64)
        p = np.asarray([float(r[args.prob_col]) for r in rows], dtype=np.float64)
        s = logit(p)
        return y, p, s

    y_train, p_train, s_train = to_arrays(by_split["train"])
    y_val, p_val, s_val = to_arrays(by_split["val"])
    y_test, p_test, s_test = to_arrays(by_split["test"])

    params, opt_result = fit_platt(s_val, y_val, l2=float(args.l2), maxiter=int(args.maxiter))
    a, b = params.tolist()
    p_train_cal = sigmoid(a * s_train + b)
    p_val_cal = sigmoid(a * s_val + b)
    p_test_cal = sigmoid(a * s_test + b)

    metrics = {
        "task": args.task,
        "uncalibrated": {
            "train": evaluate_probs(y_train, p_train),
            "val": evaluate_probs(y_val, p_val),
            "test": evaluate_probs(y_test, p_test),
        },
        "platt_calibrated": {
            "train": evaluate_probs(y_train, p_train_cal),
            "val": evaluate_probs(y_val, p_val_cal),
            "test": evaluate_probs(y_test, p_test_cal),
        },
        "platt_params": {"a": float(a), "b": float(b)},
        "optimizer": {
            "success": bool(opt_result.success),
            "status": int(opt_result.status),
            "message": str(opt_result.message),
            "nit": int(getattr(opt_result, "nit", -1)),
            "fun": float(getattr(opt_result, "fun", float("nan"))),
        },
    }

    out_rows = []
    for split_name, rows, p_uncal, p_cal in [
        ("train", by_split["train"], p_train, p_train_cal),
        ("val", by_split["val"], p_val, p_val_cal),
        ("test", by_split["test"], p_test, p_test_cal),
    ]:
        for row, pu, pc in zip(rows, p_uncal.tolist(), p_cal.tolist()):
            out_rows.append(
                {
                    "split": split_name,
                    "sample_uid": row["sample_uid"],
                    "sequence": row["sequence"],
                    "window_id": row["window_id"],
                    "y_true": row["y_true"],
                    "p_hat_uncal": float(pu),
                    "p_hat_cal": float(pc),
                }
            )

    calib_manifest = {
        "script_name": THIS_FILE.name,
        "task": args.task,
        "model_dir": str(model_dir),
        "source_run_manifest": str(model_dir / "run_manifest.json"),
        "source_legality_board_path": run_manifest.get("legality_board_path", ""),
        "source_legality_board_hash": run_manifest.get("legality_board_hash", ""),
        "calibration_method": "platt",
        "fit_split": "val",
        "outputs": {
            "metrics_json": str(out_dir / "calibration_metrics.json"),
            "predictions_csv": str(out_dir / "calibrated_predictions.csv"),
            "params_json": str(out_dir / "platt_params.json"),
            "run_manifest_json": str(out_dir / "calibration_manifest.json"),
        },
    }

    write_json(out_dir / "calibration_metrics.json", metrics)
    write_csv(
        out_dir / "calibrated_predictions.csv",
        ["split", "sample_uid", "sequence", "window_id", "y_true", "p_hat_uncal", "p_hat_cal"],
        out_rows,
    )
    write_json(out_dir / "platt_params.json", {"a": float(a), "b": float(b)})
    write_json(out_dir / "calibration_manifest.json", calib_manifest)

    before = metrics["uncalibrated"]["test"]
    after = metrics["platt_calibrated"]["test"]
    print(
        f"[RiskCalib] task={args.task} method=platt "
        f"test_Brier {before['brier']:.6f}->{after['brier']:.6f} "
        f"test_ECE {before['ece']:.6f}->{after['ece']:.6f}"
    )
    print(f"[RiskCalib] saved -> {out_dir}")


if __name__ == "__main__":
    main()
