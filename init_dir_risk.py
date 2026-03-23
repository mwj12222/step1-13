#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


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


def safe_float(v, default=0.0):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(x):
        return float(default)
    return float(x)


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


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


def default_direction_map(task: str):
    if task == "pre":
        return {
            "front_p_static": -1,
            "front_p_band": +1,
            "front_coverage_ratio": -1,
            "front_kept_dyn_ratio": +1,
            "parallax_px_candidate": -1,
            "tri_points_candidate": -1,
            "pnp_success_rate": -1,
        }
    if task == "post":
        return {
            "cheirality_ratio": -1,
            "front_coverage_ratio": -1,
            "front_kept_dyn_ratio": +1,
            "front_p_band": +1,
            "front_p_static": -1,
            "parallax_px_candidate": -1,
            "pnp_success_rate": -1,
            "reproj_med_px": +1,
            "reproj_p90_px": +1,
            "tri_points_candidate": -1,
        }
    raise ValueError(f"Unsupported task: {task}")


def default_block_map(task: str):
    if task == "pre":
        return {
            "front": [
                "front_p_static",
                "front_p_band",
                "front_coverage_ratio",
                "front_kept_dyn_ratio",
            ],
            "candidate": [
                "parallax_px_candidate",
                "tri_points_candidate",
                "pnp_success_rate",
            ],
        }
    if task == "post":
        return {
            "front": [
                "front_p_static",
                "front_p_band",
                "front_coverage_ratio",
                "front_kept_dyn_ratio",
            ],
            "candidate": [
                "parallax_px_candidate",
                "tri_points_candidate",
                "pnp_success_rate",
            ],
            "geometry": [
                "reproj_med_px",
                "reproj_p90_px",
                "cheirality_ratio",
            ],
        }
    raise ValueError(f"Unsupported task: {task}")


def build_matrix(rows: list[dict], feature_columns: list[str], direction_map: dict[str, int]):
    raw = np.asarray(
        [[safe_float(row.get(col, 0.0), 0.0) for col in feature_columns] for row in rows],
        dtype=np.float64,
    )
    aligned = raw.copy()
    for j, col in enumerate(feature_columns):
        sign = int(direction_map.get(col, +1))
        if sign < 0:
            aligned[:, j] *= -1.0
    return raw, aligned


def build_labels(rows: list[dict], label_col: str):
    return np.asarray([int(float(row[label_col])) for row in rows], dtype=np.float64)


def build_block_contributions(X_aligned_std: np.ndarray, feature_columns: list[str], weights: np.ndarray, block_map: dict[str, list[str]]):
    feature_to_idx = {c: i for i, c in enumerate(feature_columns)}
    out = {}
    for block_name, cols in block_map.items():
        idxs = [feature_to_idx[c] for c in cols if c in feature_to_idx]
        if not idxs:
            out[block_name] = np.zeros(X_aligned_std.shape[0], dtype=np.float64)
            continue
        out[block_name] = X_aligned_std[:, idxs] @ weights[idxs]
    return out


def summarize_scores(y_true: np.ndarray, y_prob: np.ndarray):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    def q(x):
        return {
            "min": float(np.min(x)) if len(x) else None,
            "p10": float(np.percentile(x, 10)) if len(x) else None,
            "p50": float(np.percentile(x, 50)) if len(x) else None,
            "p90": float(np.percentile(x, 90)) if len(x) else None,
            "max": float(np.max(x)) if len(x) else None,
            "mean": float(np.mean(x)) if len(x) else None,
        }
    return {
        "all": q(y_prob),
        "good_y0": q(y_prob[y_true == 0]),
        "bad_y1": q(y_prob[y_true == 1]),
    }


def bootstrap_stability(
    X_train_std: np.ndarray,
    y_train: np.ndarray,
    feature_columns: list[str],
    block_map: dict[str, list[str]],
    l2: float,
    maxiter: int,
    n_bootstrap: int = 100,
    seed: int = 0,
):
    rng = np.random.default_rng(int(seed))
    rows = []
    block_feature_map = default_block_feature_map(feature_columns, block_map)
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, len(y_train), size=len(y_train))
        Xb = add_intercept(X_train_std[idx])
        yb = y_train[idx]
        theta, _ = fit_logistic_regression(Xb, yb, l2=l2, maxiter=maxiter)
        weights = theta[1:]
        row = {feature: float(weight) for feature, weight in zip(feature_columns, weights.tolist())}
        for block_name, feature_idxs in block_feature_map.items():
            block_weight = float(np.sum(np.abs(weights[feature_idxs]))) if feature_idxs else 0.0
            row[f"block_abs_{block_name}"] = block_weight
        rows.append(row)

    feature_rows = []
    for feature in feature_columns:
        vals = np.asarray([r[feature] for r in rows], dtype=np.float64)
        feature_rows.append(
            {
                "feature": feature,
                "mean_weight": float(np.mean(vals)),
                "std_weight": float(np.std(vals)),
                "sign_positive_ratio": float(np.mean(vals > 0.0)),
                "sign_negative_ratio": float(np.mean(vals < 0.0)),
                "sign_nonnegative_ratio": float(np.mean(vals >= 0.0)),
            }
        )

    block_rows = []
    for block_name in block_feature_map:
        vals = np.asarray([r[f"block_abs_{block_name}"] for r in rows], dtype=np.float64)
        block_rows.append(
            {
                "block": block_name,
                "mean_abs_weight": float(np.mean(vals)),
                "std_abs_weight": float(np.std(vals)),
            }
        )
    return feature_rows, block_rows


def default_block_feature_map(feature_columns: list[str], block_map: dict[str, list[str]]):
    feature_to_idx = {c: i for i, c in enumerate(feature_columns)}
    out = {}
    for block_name, cols in block_map.items():
        out[block_name] = [feature_to_idx[c] for c in cols if c in feature_to_idx]
    return out
