#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import math

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


def log_loss_with_l2(theta, X, y, l2):
    z = X @ theta
    p = np.clip(sigmoid(z), 1e-9, 1.0 - 1e-9)
    nll = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    reg = 0.5 * l2 * float(np.sum(theta[1:] ** 2))
    return float(nll + reg)


def grad_log_loss_with_l2(theta, X, y, l2):
    p = sigmoid(X @ theta)
    grad = (X.T @ (p - y)) / max(1, X.shape[0])
    grad = np.asarray(grad, dtype=np.float64)
    grad[1:] += l2 * theta[1:]
    return grad


def fit_logistic_regression(X, y, l2=1e-2, maxiter=500):
    theta0 = np.zeros(X.shape[1], dtype=np.float64)
    result = minimize(
        fun=log_loss_with_l2,
        x0=theta0,
        jac=grad_log_loss_with_l2,
        args=(X, y, l2),
        method="L-BFGS-B",
        options={"maxiter": int(maxiter)},
    )
    if not result.success:
        raise RuntimeError(f"logistic optimization failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64), result


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
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean((y_prob - y_true) ** 2))


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


def safe_float(v, default=0.0):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(x):
        return float(default)
    return float(x)


def build_matrix(rows: list[dict], feature_columns: list[str]):
    X = np.asarray([[safe_float(row.get(col, 0.0), 0.0) for col in feature_columns] for row in rows], dtype=np.float64)
    return X


def build_labels(rows: list[dict], label_col: str):
    return np.asarray([int(float(row[label_col])) for row in rows], dtype=np.float64)


def parse_override_feature_columns(raw: str | None):
    if not raw:
        return None
    cols = [c.strip() for c in str(raw).split(",")]
    cols = [c for c in cols if c]
    return cols or None


def standardize(train_X: np.ndarray, other_Xs: list[np.ndarray]):
    mean = np.mean(train_X, axis=0)
    std = np.std(train_X, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    out = [(train_X - mean) / std]
    for X in other_Xs:
        out.append((X - mean) / std)
    return mean, std, out


def add_intercept(X: np.ndarray):
    return np.concatenate([np.ones((X.shape[0], 1), dtype=np.float64), X], axis=1)


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


def join_rule_scores(rows: list[dict], audit_rows_by_uid: dict, rule_col: str):
    scores = []
    for row in rows:
        uid = row["sample_uid"]
        audit = audit_rows_by_uid.get(uid, {})
        q_good = safe_float(audit.get(rule_col), 0.0)
        scores.append(float(np.clip(1.0 - q_good, 0.0, 1.0)))
    return np.asarray(scores, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description="Train a minimal interpretable initialization-risk baseline.")
    ap.add_argument("--task", required=True, choices=["pre", "post"])
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--label_col", default="Y_bad_v1")
    ap.add_argument("--model", default="logistic", choices=["logistic"])
    ap.add_argument(
        "--override_feature_columns",
        default="",
        help="Comma-separated feature subset. Must be a subset of manifest allowed_feature_columns.",
    )
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(Path(args.manifest).expanduser().resolve())
    train_rows = load_csv_rows(Path(args.train_csv).expanduser().resolve())
    val_rows = load_csv_rows(Path(args.val_csv).expanduser().resolve())
    test_rows = load_csv_rows(Path(args.test_csv).expanduser().resolve())

    allowed_feature_columns = list(manifest["allowed_feature_columns"])
    override_feature_columns = parse_override_feature_columns(args.override_feature_columns)
    if override_feature_columns is not None:
        unknown = [c for c in override_feature_columns if c not in allowed_feature_columns]
        if unknown:
            raise ValueError(
                f"override_feature_columns contains columns not allowed by manifest: {unknown}"
            )
        allowed_feature_columns = list(override_feature_columns)
    if not allowed_feature_columns:
        raise ValueError("No allowed_feature_columns remain after override.")
    label_columns = list(manifest.get("label_columns", []))
    if args.label_col not in label_columns:
        raise ValueError(f"label_col {args.label_col} not allowed by manifest: {label_columns}")

    X_train = build_matrix(train_rows, allowed_feature_columns)
    X_val = build_matrix(val_rows, allowed_feature_columns)
    X_test = build_matrix(test_rows, allowed_feature_columns)
    y_train = build_labels(train_rows, args.label_col)
    y_val = build_labels(val_rows, args.label_col)
    y_test = build_labels(test_rows, args.label_col)

    mean, std, scaled = standardize(X_train, [X_val, X_test])
    X_train_s, X_val_s, X_test_s = scaled
    X_train_i = add_intercept(X_train_s)
    X_val_i = add_intercept(X_val_s)
    X_test_i = add_intercept(X_test_s)

    theta, opt_result = fit_logistic_regression(X_train_i, y_train, l2=float(args.l2), maxiter=int(args.maxiter))
    p_train = sigmoid(X_train_i @ theta)
    p_val = sigmoid(X_val_i @ theta)
    p_test = sigmoid(X_test_i @ theta)

    rule_col = "Q_pre" if args.task == "pre" else "Q_post_geom_only"
    audit_csv = Path(args.manifest).expanduser().resolve().parent / "risk_dataset.csv"
    audit_rows = load_csv_rows(audit_csv)
    audit_rows_by_uid = {row["sample_uid"]: row for row in audit_rows}
    rule_train = join_rule_scores(train_rows, audit_rows_by_uid, rule_col)
    rule_val = join_rule_scores(val_rows, audit_rows_by_uid, rule_col)
    rule_test = join_rule_scores(test_rows, audit_rows_by_uid, rule_col)

    metrics = {
        "task": args.task,
        "label_col": args.label_col,
        "model": args.model,
        "allowed_feature_columns": allowed_feature_columns,
        "feature_override_active": override_feature_columns is not None,
        "feature_override_columns": list(override_feature_columns or []),
        "splits": {
            "train": evaluate_probs(y_train, p_train),
            "val": evaluate_probs(y_val, p_val),
            "test": evaluate_probs(y_test, p_test),
        },
        "rule_baseline": {
            "rule_col": rule_col,
            "train": evaluate_probs(y_train, rule_train),
            "val": evaluate_probs(y_val, rule_val),
            "test": evaluate_probs(y_test, rule_test),
        },
        "optimizer": {
            "success": bool(opt_result.success),
            "status": int(opt_result.status),
            "message": str(opt_result.message),
            "nit": int(getattr(opt_result, "nit", -1)),
            "fun": float(getattr(opt_result, "fun", float("nan"))),
        },
    }

    prediction_rows = []
    for split_name, rows, probs in [
        ("train", train_rows, p_train),
        ("val", val_rows, p_val),
        ("test", test_rows, p_test),
    ]:
        for row, prob in zip(rows, probs.tolist()):
            prediction_rows.append(
                {
                    "split": split_name,
                    "sample_uid": row["sample_uid"],
                    "sequence": row["sequence"],
                    "window_id": row["window_id"],
                    "y_true": row[args.label_col],
                    "p_hat": float(prob),
                }
            )

    coeff_rows = [{"feature": "__intercept__", "weight": float(theta[0])}]
    coeff_rows.extend(
        {"feature": feature, "weight": float(weight)}
        for feature, weight in zip(allowed_feature_columns, theta[1:].tolist())
    )

    scaler_rows = [
        {"feature": feature, "mean": float(mu), "std": float(sd)}
        for feature, mu, sd in zip(allowed_feature_columns, mean.tolist(), std.tolist())
    ]

    run_manifest = {
        "script_name": THIS_FILE.name,
        "task": args.task,
        "model": args.model,
        "train_csv": str(Path(args.train_csv).expanduser().resolve()),
        "val_csv": str(Path(args.val_csv).expanduser().resolve()),
        "test_csv": str(Path(args.test_csv).expanduser().resolve()),
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "label_col": args.label_col,
        "label_version": manifest.get("label_version", args.label_col),
        "label_scope": manifest.get("label_scope", "init_level_bad_event_proxy"),
        "task_scope": manifest.get("task_scope", ""),
        "allowed_claims": manifest.get("allowed_claims", []),
        "forbidden_claims": manifest.get("forbidden_claims", []),
        "proxy_bias_notes": manifest.get("proxy_bias_notes", []),
        "experimental_limit_flags": manifest.get("experimental_limit_flags", []),
        "label_population": manifest.get("label_population", ""),
        "coverage_summary": manifest.get("coverage_summary", {}),
        "allowed_feature_columns": allowed_feature_columns,
        "feature_override_active": override_feature_columns is not None,
        "feature_override_columns": list(override_feature_columns or []),
        "legality_board_path": manifest.get("legality_board_path", ""),
        "legality_board_hash": manifest.get("legality_board_hash", ""),
        "outputs": {
            "metrics_json": str(out_dir / "metrics.json"),
            "predictions_csv": str(out_dir / "predictions.csv"),
            "coefficients_csv": str(out_dir / "coefficients.csv"),
            "scaler_stats_csv": str(out_dir / "scaler_stats.csv"),
            "claim_constraints_json": str(out_dir / "claim_constraints.json"),
            "run_manifest_json": str(out_dir / "run_manifest.json"),
        },
    }
    claim_constraints = {
        "label_version": run_manifest["label_version"],
        "label_scope": run_manifest["label_scope"],
        "task_scope": run_manifest["task_scope"],
        "allowed_claims": run_manifest["allowed_claims"],
        "forbidden_claims": run_manifest["forbidden_claims"],
        "proxy_bias_notes": run_manifest["proxy_bias_notes"],
        "experimental_limit_flags": run_manifest["experimental_limit_flags"],
        "label_population": run_manifest["label_population"],
        "coverage_summary": run_manifest["coverage_summary"],
    }

    write_json(out_dir / "metrics.json", metrics)
    write_csv(out_dir / "predictions.csv", ["split", "sample_uid", "sequence", "window_id", "y_true", "p_hat"], prediction_rows)
    write_csv(out_dir / "coefficients.csv", ["feature", "weight"], coeff_rows)
    write_csv(out_dir / "scaler_stats.csv", ["feature", "mean", "std"], scaler_rows)
    write_json(out_dir / "claim_constraints.json", claim_constraints)
    write_json(out_dir / "run_manifest.json", run_manifest)

    print(f"[RiskBaseline] task={args.task} model={args.model}")
    print(f"[RiskBaseline] test AUROC={metrics['splits']['test']['auroc']} AUPRC={metrics['splits']['test']['auprc']} "
          f"Brier={metrics['splits']['test']['brier']} ECE={metrics['splits']['test']['ece']}")
    print(f"[RiskBaseline] rule_col={rule_col} rule_test_AUROC={metrics['rule_baseline']['test']['auroc']}")
    print(f"[RiskBaseline] saved -> {out_dir}")


if __name__ == "__main__":
    main()
