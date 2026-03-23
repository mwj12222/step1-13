#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np

from init_dir_risk import (
    add_intercept,
    bootstrap_stability,
    build_block_contributions,
    build_labels,
    build_matrix,
    default_block_map,
    default_direction_map,
    evaluate_probs,
    fit_logistic_regression,
    load_csv_rows,
    load_json,
    sigmoid,
    standardize,
    summarize_scores,
    write_csv,
    write_json,
)


THIS_FILE = Path(__file__).resolve()


def parse_override_feature_columns(raw: str | None):
    if not raw:
        return None
    cols = [c.strip() for c in str(raw).split(",")]
    cols = [c for c in cols if c]
    return cols or None


def join_rule_scores(rows: list[dict], audit_rows_by_uid: dict, rule_col: str):
    scores = []
    for row in rows:
        uid = row["sample_uid"]
        audit = audit_rows_by_uid.get(uid, {})
        q_good = float(audit.get(rule_col, 0.0) or 0.0)
        scores.append(float(np.clip(1.0 - q_good, 0.0, 1.0)))
    return np.asarray(scores, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description="Train a direction-aligned initialization-risk logistic baseline.")
    ap.add_argument("--task", required=True, choices=["pre", "post"])
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--label_col", default="Y_bad_v1")
    ap.add_argument("--override_feature_columns", default="")
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--bootstrap_runs", type=int, default=100)
    ap.add_argument("--bootstrap_seed", type=int, default=20260312)
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
            raise ValueError(f"override_feature_columns contains columns not allowed by manifest: {unknown}")
        allowed_feature_columns = list(override_feature_columns)
    if not allowed_feature_columns:
        raise ValueError("No allowed_feature_columns remain after override.")

    label_columns = list(manifest.get("label_columns", []))
    if args.label_col not in label_columns:
        raise ValueError(f"label_col {args.label_col} not allowed by manifest: {label_columns}")

    direction_map_full = default_direction_map(args.task)
    direction_map = {c: int(direction_map_full[c]) for c in allowed_feature_columns}
    block_map_full = default_block_map(args.task)
    block_map = {
        block: [c for c in cols if c in allowed_feature_columns]
        for block, cols in block_map_full.items()
    }
    block_map = {k: v for k, v in block_map.items() if v}

    train_raw, train_aligned = build_matrix(train_rows, allowed_feature_columns, direction_map)
    val_raw, val_aligned = build_matrix(val_rows, allowed_feature_columns, direction_map)
    test_raw, test_aligned = build_matrix(test_rows, allowed_feature_columns, direction_map)
    y_train = build_labels(train_rows, args.label_col)
    y_val = build_labels(val_rows, args.label_col)
    y_test = build_labels(test_rows, args.label_col)

    mean, std, scaled = standardize(train_aligned, [val_aligned, test_aligned])
    X_train_s, X_val_s, X_test_s = scaled
    X_train_i = add_intercept(X_train_s)
    X_val_i = add_intercept(X_val_s)
    X_test_i = add_intercept(X_test_s)

    theta, opt_result = fit_logistic_regression(X_train_i, y_train, l2=float(args.l2), maxiter=int(args.maxiter))
    p_train = sigmoid(X_train_i @ theta)
    p_val = sigmoid(X_val_i @ theta)
    p_test = sigmoid(X_test_i @ theta)
    weights = theta[1:]

    train_blocks = build_block_contributions(X_train_s, allowed_feature_columns, weights, block_map)
    val_blocks = build_block_contributions(X_val_s, allowed_feature_columns, weights, block_map)
    test_blocks = build_block_contributions(X_test_s, allowed_feature_columns, weights, block_map)

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
        "model": "dir_risk_logistic",
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

    coeff_rows = [{"feature": "__intercept__", "aligned_weight": float(theta[0]), "raw_direction": 0, "raw_implied_weight": float(theta[0])}]
    violation_count = 0
    for feature, weight in zip(allowed_feature_columns, weights.tolist()):
        raw_direction = int(direction_map[feature])
        raw_implied_weight = float(weight * raw_direction)
        if weight < 0.0:
            violation_count += 1
        coeff_rows.append(
            {
                "feature": feature,
                "aligned_weight": float(weight),
                "raw_direction": raw_direction,
                "raw_implied_weight": raw_implied_weight,
            }
        )

    direction_report = {
        "task": args.task,
        "model_name": "Main-DirRisk",
        "note": "Direction-aligned logistic baseline. This is not a true monotonic constrained model.",
        "direction_map": direction_map,
        "negative_aligned_weight_features": [r["feature"] for r in coeff_rows[1:] if r["aligned_weight"] < 0.0],
        "num_negative_aligned_weights": int(violation_count),
    }

    scaler_rows = [
        {"feature": feature, "mean": float(mu), "std": float(sd)}
        for feature, mu, sd in zip(allowed_feature_columns, mean.tolist(), std.tolist())
    ]

    feature_stability_rows, block_stability_rows = bootstrap_stability(
        X_train_s,
        y_train,
        allowed_feature_columns,
        block_map,
        l2=float(args.l2),
        maxiter=int(args.maxiter),
        n_bootstrap=int(args.bootstrap_runs),
        seed=int(args.bootstrap_seed),
    )

    block_weight_rows = []
    abs_weights = np.abs(weights)
    denom = float(np.sum(abs_weights)) if len(abs_weights) else 1.0
    for block_name, cols in block_map.items():
        idxs = [allowed_feature_columns.index(c) for c in cols]
        block_abs = float(np.sum(abs_weights[idxs])) if idxs else 0.0
        block_weight_rows.append(
            {
                "block": block_name,
                "num_features": len(cols),
                "abs_weight_sum": block_abs,
                "abs_weight_ratio": float(block_abs / denom) if denom > 1e-12 else 0.0,
            }
        )

    risk_score_summary = {
        "train": summarize_scores(y_train, p_train),
        "val": summarize_scores(y_val, p_val),
        "test": summarize_scores(y_test, p_test),
    }

    prediction_rows = []
    for split_name, rows, probs, raw_matrix, aligned_std, blocks in [
        ("train", train_rows, p_train, train_raw, X_train_s, train_blocks),
        ("val", val_rows, p_val, val_raw, X_val_s, val_blocks),
        ("test", test_rows, p_test, test_raw, X_test_s, test_blocks),
    ]:
        for i, (row, prob) in enumerate(zip(rows, probs.tolist())):
            out = {
                "split": split_name,
                "sample_uid": row["sample_uid"],
                "sequence": row["sequence"],
                "window_id": row["window_id"],
                "y_true": row[args.label_col],
                "p_hat": float(prob),
            }
            for block_name, vals in blocks.items():
                out[f"block_logit_{block_name}"] = float(vals[i])
            prediction_rows.append(out)

    run_manifest = {
        "script_name": THIS_FILE.name,
        "task": args.task,
        "model": "Main-DirRisk",
        "model_note": "Direction-aligned logistic baseline, not true monotonic constrained model.",
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
        "direction_map": direction_map,
        "block_map": block_map,
        "bootstrap_runs": int(args.bootstrap_runs),
        "bootstrap_seed": int(args.bootstrap_seed),
        "legality_board_path": manifest.get("legality_board_path", ""),
        "legality_board_hash": manifest.get("legality_board_hash", ""),
        "outputs": {
            "metrics_json": str(out_dir / "metrics.json"),
            "predictions_csv": str(out_dir / "predictions.csv"),
            "coefficients_csv": str(out_dir / "coefficients.csv"),
            "feature_direction_map_json": str(out_dir / "feature_direction_map.json"),
            "scaler_stats_csv": str(out_dir / "scaler_stats.csv"),
            "block_weight_summary_csv": str(out_dir / "block_weight_summary.csv"),
            "coefficient_stability_csv": str(out_dir / "coefficient_stability.csv"),
            "block_stability_csv": str(out_dir / "block_stability.csv"),
            "risk_score_summary_json": str(out_dir / "risk_score_summary.json"),
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
        "dir_risk_note": "Direction-aligned logistic baseline only; not a true monotonic constrained risk model.",
    }

    prediction_fieldnames = ["split", "sample_uid", "sequence", "window_id", "y_true", "p_hat"] + [f"block_logit_{b}" for b in block_map]
    write_json(out_dir / "metrics.json", metrics)
    write_csv(out_dir / "predictions.csv", prediction_fieldnames, prediction_rows)
    write_csv(out_dir / "coefficients.csv", ["feature", "aligned_weight", "raw_direction", "raw_implied_weight"], coeff_rows)
    write_json(out_dir / "feature_direction_map.json", direction_report)
    write_csv(out_dir / "scaler_stats.csv", ["feature", "mean", "std"], scaler_rows)
    write_csv(out_dir / "block_weight_summary.csv", ["block", "num_features", "abs_weight_sum", "abs_weight_ratio"], block_weight_rows)
    write_csv(
        out_dir / "coefficient_stability.csv",
        ["feature", "mean_weight", "std_weight", "sign_positive_ratio", "sign_negative_ratio", "sign_nonnegative_ratio"],
        feature_stability_rows,
    )
    write_csv(out_dir / "block_stability.csv", ["block", "mean_abs_weight", "std_abs_weight"], block_stability_rows)
    write_json(out_dir / "risk_score_summary.json", risk_score_summary)
    write_json(out_dir / "claim_constraints.json", claim_constraints)
    write_json(out_dir / "run_manifest.json", run_manifest)

    print(f"[DirRisk] task={args.task}")
    print(f"[DirRisk] test AUROC={metrics['splits']['test']['auroc']} AUPRC={metrics['splits']['test']['auprc']} "
          f"Brier={metrics['splits']['test']['brier']} ECE={metrics['splits']['test']['ece']}")
    print(f"[DirRisk] negative_aligned_weights={violation_count}")
    print(f"[DirRisk] saved -> {out_dir}")


if __name__ == "__main__":
    main()
