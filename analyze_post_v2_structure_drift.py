#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import math


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def safe_float(v):
    try:
        x = float(v)
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def mean(xs: list[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def std(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return 0.0 if xs else None
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    sx, sy = std(xs), std(ys)
    if sx in (None, 0.0) or sy in (None, 0.0):
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (len(xs) - 1)
    return cov / (sx * sy)


def brier(y_true: list[int], p_hat: list[float]) -> float | None:
    if not y_true:
        return None
    return sum((p - y) ** 2 for y, p in zip(y_true, p_hat)) / len(y_true)


def auc_roc(y_true: list[int], scores: list[float]) -> float | None:
    pos = [(s, y) for s, y in zip(scores, y_true) if y == 1]
    neg = [(s, y) for s, y in zip(scores, y_true) if y == 0]
    if not pos or not neg:
        return None
    total = 0.0
    for sp, _ in pos:
        for sn, _ in neg:
            if sp > sn:
                total += 1.0
            elif sp == sn:
                total += 0.5
    return total / (len(pos) * len(neg))


def best_signed_auc(y_true: list[int], scores: list[float]) -> tuple[float | None, str]:
    auc_pos = auc_roc(y_true, scores)
    auc_neg = auc_roc(y_true, [-s for s in scores])
    if auc_pos is None and auc_neg is None:
        return None, "na"
    if auc_neg is None or (auc_pos is not None and auc_pos >= auc_neg):
        return auc_pos, "positive"
    return auc_neg, "negative"


def standardized_mean_diff(a: list[float], b: list[float]) -> float | None:
    if not a or not b:
        return None
    ma, mb = mean(a), mean(b)
    sa, sb = std(a), std(b)
    pooled = math.sqrt(((sa or 0.0) ** 2 + (sb or 0.0) ** 2) / 2.0)
    if pooled == 0.0:
        return 0.0
    return (ma - mb) / pooled


def fmt(v, nd=4):
    if v is None or v == "":
        return "-"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def resolve_latest_dir(root: Path, pattern: str) -> Path:
    candidates = [p for p in root.glob(pattern) if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No directory matched {pattern} under {root}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def block_name(feature: str) -> str:
    if feature.startswith("front_"):
        return "front"
    if feature in ("parallax_px_candidate", "tri_points_candidate", "pnp_success_rate"):
        return "candidate"
    if feature in ("reproj_med_px", "reproj_p90_px", "cheirality_ratio"):
        return "geometry"
    return "other"


def main():
    ap = argparse.ArgumentParser(description="Analyze feature correlation, split drift, and sequence contribution for post_v2 structure shift.")
    ap.add_argument("--result_root", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    root = Path(args.result_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    task_manifest = load_json(root / "risk_dataset_post_v2_min_default_manifest.json")
    logistic_dir = resolve_latest_dir(root, "model_post_v2_min_default_logistic_*")
    dirrisk_dir = resolve_latest_dir(root, "model_post_v2_min_default_dir_risk_*")
    ablation_dir = resolve_latest_dir(root, "post_v2_ablations_*")
    logistic_metrics = load_json(logistic_dir / "metrics.json")
    dirrisk_metrics = load_json(dirrisk_dir / "metrics.json")
    predictions = load_csv_rows(logistic_dir / "predictions.csv")
    coeffs = load_csv_rows(logistic_dir / "coefficients.csv")
    ablation_summary = load_json(ablation_dir / "post_v2_ablation_summary.json")
    split_paths = task_manifest["output_split_csvs"]
    train_rows = load_csv_rows(Path(split_paths["train"]))
    val_rows = load_csv_rows(Path(split_paths["val"]))
    test_rows = load_csv_rows(Path(split_paths["test"]))
    all_rows = train_rows + val_rows + test_rows
    features = list(task_manifest["allowed_feature_columns"])

    # Correlation on train split
    corr_rows = []
    top_corr_rows = []
    for i, fa in enumerate(features):
        xa = [safe_float(r.get(fa)) for r in train_rows]
        for fb in features[i + 1:]:
            xb = [safe_float(r.get(fb)) for r in train_rows]
            pairs = [(a, b) for a, b in zip(xa, xb) if a is not None and b is not None]
            corr = pearson([a for a, _ in pairs], [b for _, b in pairs]) if pairs else None
            row = {
                "feature_a": fa,
                "feature_b": fb,
                "pearson_r_train": corr,
                "abs_r_train": abs(corr) if corr is not None else None,
                "num_pairs": len(pairs),
            }
            corr_rows.append(row)
    top_corr_rows = sorted(
        [r for r in corr_rows if r["abs_r_train"] is not None],
        key=lambda r: r["abs_r_train"],
        reverse=True,
    )[:15]

    # Split drift
    split_map = {"train": train_rows, "val": val_rows, "test": test_rows}
    drift_rows = []
    for feat in features:
        vals = {split: [safe_float(r.get(feat)) for r in rows] for split, rows in split_map.items()}
        vals = {k: [x for x in v if x is not None] for k, v in vals.items()}
        drift_rows.append(
            {
                "feature": feat,
                "train_mean": mean(vals["train"]),
                "val_mean": mean(vals["val"]),
                "test_mean": mean(vals["test"]),
                "train_std": std(vals["train"]),
                "val_std": std(vals["val"]),
                "test_std": std(vals["test"]),
                "smd_train_test": standardized_mean_diff(vals["train"], vals["test"]),
                "smd_train_val": standardized_mean_diff(vals["train"], vals["val"]),
                "smd_val_test": standardized_mean_diff(vals["val"], vals["test"]),
            }
        )
    top_drift_rows = sorted(
        [r for r in drift_rows if r["smd_train_test"] is not None],
        key=lambda r: abs(r["smd_train_test"]),
        reverse=True,
    )[:10]

    # Single-feature signal strength
    signal_rows = []
    for feat in features:
        row = {"feature": feat, "block": block_name(feat)}
        for split_name, split_rows in split_map.items():
            vals = []
            ys = []
            for r in split_rows:
                x = safe_float(r.get(feat))
                y = safe_float(r.get("Y_bad_v2_min_default"))
                if x is None or y is None:
                    continue
                vals.append(x)
                ys.append(int(y))
            auc_best, sign = best_signed_auc(ys, vals) if vals else (None, "na")
            row[f"{split_name}_label_pearson"] = pearson(vals, ys) if vals else None
            row[f"{split_name}_best_auc"] = auc_best
            row[f"{split_name}_best_auc_sign"] = sign
            row[f"{split_name}_num_rows"] = len(vals)
        signal_rows.append(row)

    top_test_signal_rows = sorted(
        [r for r in signal_rows if r["test_best_auc"] is not None],
        key=lambda r: r["test_best_auc"],
        reverse=True,
    )[:10]

    block_summary_rows = []
    for block in sorted({block_name(f) for f in features}):
        block_feats = [f for f in features if block_name(f) == block]
        drift_vals = [
            abs(r["smd_train_test"]) for r in drift_rows
            if r["feature"] in block_feats and r["smd_train_test"] is not None
        ]
        sig_vals = [
            r["test_best_auc"] for r in signal_rows
            if r["feature"] in block_feats and r["test_best_auc"] is not None
        ]
        block_summary_rows.append(
            {
                "block": block,
                "num_features": len(block_feats),
                "avg_abs_smd_train_test": mean(drift_vals),
                "max_abs_smd_train_test": max(drift_vals) if drift_vals else None,
                "avg_test_best_auc": mean(sig_vals),
                "max_test_best_auc": max(sig_vals) if sig_vals else None,
            }
        )

    # Sequence contribution from predictions
    pred_by_uid = {r["sample_uid"]: r for r in predictions}
    seq_rows = []
    for seq in sorted({r["sequence"] for r in all_rows}):
        rows = [r for r in all_rows if r["sequence"] == seq]
        seq_preds = [pred_by_uid[r["sample_uid"]] for r in rows if r["sample_uid"] in pred_by_uid]
        y = [int(r["Y_bad_v2_min_default"]) for r in rows]
        p = [float(pred_by_uid[r["sample_uid"]]["p_hat"]) for r in rows if r["sample_uid"] in pred_by_uid]
        y_for_pred = [int(r["Y_bad_v2_min_default"]) for r in rows if r["sample_uid"] in pred_by_uid]
        split = rows[0]["dataset_row_split"] if rows else ""
        feat_means = {feat: mean([safe_float(r.get(feat)) for r in rows if safe_float(r.get(feat)) is not None]) for feat in features}
        seq_rows.append(
            {
                "sequence": seq,
                "split": split,
                "rows": len(rows),
                "bad_ratio": sum(y) / len(y) if y else None,
                "mean_p_hat": mean(p),
                "brier": brier(y_for_pred, p),
                "auroc": auc_roc(y_for_pred, p),
                "top_feature_signature": json.dumps(feat_means, ensure_ascii=False),
            }
        )

    # A compact train-sequence signature table
    train_sequences = sorted({r["sequence"] for r in train_rows})
    train_seq_sig_rows = []
    train_feature_means = {feat: mean([safe_float(r.get(feat)) for r in train_rows if safe_float(r.get(feat)) is not None]) for feat in features}
    train_feature_stds = {feat: std([safe_float(r.get(feat)) for r in train_rows if safe_float(r.get(feat)) is not None]) or 0.0 for feat in features}
    for seq in train_sequences:
        rows = [r for r in train_rows if r["sequence"] == seq]
        drift_items = []
        for feat in features:
            vals = [safe_float(r.get(feat)) for r in rows if safe_float(r.get(feat)) is not None]
            if not vals:
                continue
            seq_mean = mean(vals)
            base_mean = train_feature_means[feat]
            base_std = train_feature_stds[feat]
            z = 0.0 if not base_std else (seq_mean - base_mean) / base_std
            drift_items.append((feat, z))
        drift_items.sort(key=lambda kv: abs(kv[1]), reverse=True)
        train_seq_sig_rows.append(
            {
                "sequence": seq,
                "rows": len(rows),
                "bad_ratio": sum(int(r["Y_bad_v2_min_default"]) for r in rows) / len(rows) if rows else None,
                "top_feature_drifts": json.dumps(
                    [{"feature": f, "z_train_mean": z} for f, z in drift_items[:5]],
                    ensure_ascii=False,
                ),
            }
        )

    coeff_map = {r["feature"]: safe_float(r["weight"]) for r in coeffs}
    summary = {
        "dataset": {
            "result_root": str(root),
            "label_scope": task_manifest["label_scope"],
            "coverage_summary": task_manifest["coverage_summary"],
            "split_sizes": {k: len(v) for k, v in split_map.items()},
            "split_bad_ratio": {
                k: sum(int(r["Y_bad_v2_min_default"]) for r in v) / len(v) if v else None
                for k, v in split_map.items()
            },
        },
        "main_result": {
            "test_auroc": logistic_metrics["splits"]["test"]["auroc"],
            "test_auprc": logistic_metrics["splits"]["test"]["auprc"],
            "test_brier": logistic_metrics["splits"]["test"]["brier"],
            "rule_test_auroc": logistic_metrics["rule_baseline"]["test"]["auroc"],
            "dirrisk_test_auroc": dirrisk_metrics["splits"]["test"]["auroc"],
        },
        "resolved_inputs": {
            "logistic_dir": str(logistic_dir),
            "dirrisk_dir": str(dirrisk_dir),
            "ablation_dir": str(ablation_dir),
        },
        "top_feature_correlations_train": top_corr_rows,
        "top_train_test_drift_features": top_drift_rows,
        "top_test_signal_features": top_test_signal_rows,
        "block_summary": block_summary_rows,
        "ablation_rows": ablation_summary["rows"],
        "sequence_contribution": seq_rows,
        "train_sequence_signatures": train_seq_sig_rows,
        "coefficients": coeff_map,
        "takeaways": [
            "先看 train 内特征相关性，判断 geometry/front/candidate 是否存在强共线性，从而影响 block 解释。",
            "再看 train-test 的 standardized mean difference，判断 geometry-only 掉队是结构变化还是分布漂移。",
            "再看单特征 test signal，判断 candidate_only 跑到第一是单特征可分性上升，还是多特征组合互相干扰。",
            "最后看各序列的 bad_ratio、预测难度和特征签名，定位是不是新增序列改写了原有结论。",
        ],
    }

    lines = []
    lines.append("# post_v2 结构漂移分析")
    lines.append("")
    lines.append("## 数据与主结果")
    lines.append("")
    lines.append(f"- split sizes: `{summary['dataset']['split_sizes']}`")
    lines.append(f"- split bad ratio: `{summary['dataset']['split_bad_ratio']}`")
    lines.append(f"- test AUROC: `{summary['main_result']['test_auroc']:.4f}`")
    lines.append(f"- rule test AUROC: `{summary['main_result']['rule_test_auroc']:.4f}`")
    lines.append("")
    lines.append("## Train 内最高相关特征对")
    lines.append("")
    for row in top_corr_rows[:8]:
        lines.append(f"- `{row['feature_a']}` vs `{row['feature_b']}`: r=`{fmt(row['pearson_r_train'])}`")
    lines.append("")
    lines.append("## Train/Test 漂移最大的特征")
    lines.append("")
    for row in top_drift_rows[:8]:
        lines.append(f"- `{row['feature']}`: SMD(train,test)=`{fmt(row['smd_train_test'])}`, train_mean=`{fmt(row['train_mean'])}`, test_mean=`{fmt(row['test_mean'])}`")
    lines.append("")
    lines.append("## Test 单特征信号最强的特征")
    lines.append("")
    for row in top_test_signal_rows[:8]:
        lines.append(
            f"- `{row['feature']}` ({row['block']}): best test AUROC=`{fmt(row['test_best_auc'])}`, "
            f"label_pearson(test)=`{fmt(row['test_label_pearson'])}`, best_sign=`{row['test_best_auc_sign']}`"
        )
    lines.append("")
    lines.append("## Block 级判断")
    lines.append("")
    for row in block_summary_rows:
        lines.append(
            f"- `{row['block']}`: avg|SMD|(train,test)=`{fmt(row['avg_abs_smd_train_test'])}`, "
            f"max|SMD|=`{fmt(row['max_abs_smd_train_test'])}`, avg best test AUROC=`{fmt(row['avg_test_best_auc'])}`, "
            f"max best test AUROC=`{fmt(row['max_test_best_auc'])}`"
        )
    lines.append("")
    lines.append("## 序列贡献")
    lines.append("")
    for row in seq_rows:
        lines.append(f"- `{row['sequence']}` ({row['split']}): rows=`{row['rows']}`, bad_ratio=`{fmt(row['bad_ratio'])}`, mean_p_hat=`{fmt(row['mean_p_hat'])}`, brier=`{fmt(row['brier'])}`, auroc=`{fmt(row['auroc'])}`")
    lines.append("")
    lines.append("## 批判性观察")
    lines.append("")
    test_sequences = [row["sequence"] for row in seq_rows if row["split"] == "test"]
    test_seq_text = ", ".join(f"`{s}`" for s in test_sequences) if test_sequences else "`<none>`"
    lines.append("- 如果几何特征与 front/candidate 在 train 内高度相关，那么旧的 block 主导叙事本来就可能被共线性放大。")
    lines.append("- 如果 train/test 在几何特征上出现明显漂移，而 front 特征漂移较小，那么 `geometry_only` 掉队更像分布漂移，不一定代表几何真的失效。")
    lines.append(f"- 当前 test 仍只由 {test_seq_text} 承担，所以这轮结果更适合写成 sequence-specific stress test，而不是稳定跨场景结论。")
    lines.append("")

    write_json(out_dir / "post_v2_structure_drift_analysis.json", summary)
    write_csv(
        out_dir / "post_v2_feature_correlation_train.csv",
        ["feature_a", "feature_b", "pearson_r_train", "abs_r_train", "num_pairs"],
        corr_rows,
    )
    write_csv(
        out_dir / "post_v2_top_feature_correlation_train.csv",
        ["feature_a", "feature_b", "pearson_r_train", "abs_r_train", "num_pairs"],
        top_corr_rows,
    )
    write_csv(
        out_dir / "post_v2_split_drift.csv",
        ["feature", "train_mean", "val_mean", "test_mean", "train_std", "val_std", "test_std", "smd_train_test", "smd_train_val", "smd_val_test"],
        drift_rows,
    )
    write_csv(
        out_dir / "post_v2_feature_signal.csv",
        [
            "feature", "block",
            "train_label_pearson", "train_best_auc", "train_best_auc_sign", "train_num_rows",
            "val_label_pearson", "val_best_auc", "val_best_auc_sign", "val_num_rows",
            "test_label_pearson", "test_best_auc", "test_best_auc_sign", "test_num_rows",
        ],
        signal_rows,
    )
    write_csv(
        out_dir / "post_v2_block_summary.csv",
        ["block", "num_features", "avg_abs_smd_train_test", "max_abs_smd_train_test", "avg_test_best_auc", "max_test_best_auc"],
        block_summary_rows,
    )
    write_csv(
        out_dir / "post_v2_sequence_contribution.csv",
        ["sequence", "split", "rows", "bad_ratio", "mean_p_hat", "brier", "auroc", "top_feature_signature"],
        seq_rows,
    )
    write_csv(
        out_dir / "post_v2_train_sequence_signatures.csv",
        ["sequence", "rows", "bad_ratio", "top_feature_drifts"],
        train_seq_sig_rows,
    )
    write_text(out_dir / "post_v2_structure_drift_analysis.md", "\n".join(lines))
    print(f"[post_v2_structure_drift] saved -> {out_dir}")


if __name__ == "__main__":
    main()
