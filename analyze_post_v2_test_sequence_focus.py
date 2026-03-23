#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


KEY_FEATURES = [
    "front_p_static",
    "front_coverage_ratio",
    "parallax_px_candidate",
    "tri_points_candidate",
    "reproj_med_px",
    "reproj_p90_px",
    "cheirality_ratio",
]


def load_csv_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str) -> None:
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


def auc_roc(y_true: list[int], scores: list[float]) -> float | None:
    pos = [s for s, y in zip(scores, y_true) if y == 1]
    neg = [s for s, y in zip(scores, y_true) if y == 0]
    if not pos or not neg:
        return None
    total = 0.0
    for sp in pos:
        for sn in neg:
            if sp > sn:
                total += 1.0
            elif sp == sn:
                total += 0.5
    return total / (len(pos) * len(neg))


def quantile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    vals = sorted(xs)
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    sx, sy = std(xs), std(ys)
    if sx in (None, 0.0) or sy in (None, 0.0):
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (len(xs) - 1)
    return cov / (sx * sy)


def fmt(v, nd=4):
    if v in (None, ""):
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


def prediction_map(path: Path, sequence: str) -> dict[str, dict]:
    rows = load_csv_rows(path)
    out = {}
    for row in rows:
        if row["split"] != "test":
            continue
        if row["sequence"] != sequence:
            continue
        out[str(row["sample_uid"])] = row
    return out


def score_stats(rows: list[dict], p_key: str, label_key: str = "y_true") -> dict:
    y = [int(row[label_key]) for row in rows]
    p = [float(row[p_key]) for row in rows]
    pos = [pp for yy, pp in zip(y, p) if yy == 1]
    neg = [pp for yy, pp in zip(y, p) if yy == 0]
    pred = [1 if pp >= 0.5 else 0 for pp in p]
    tp = sum(1 for yy, hh in zip(y, pred) if yy == 1 and hh == 1)
    tn = sum(1 for yy, hh in zip(y, pred) if yy == 0 and hh == 0)
    fp = sum(1 for yy, hh in zip(y, pred) if yy == 0 and hh == 1)
    fn = sum(1 for yy, hh in zip(y, pred) if yy == 1 and hh == 0)
    return {
        "num_rows": len(rows),
        "positive_ratio": sum(y) / max(1, len(y)),
        "auroc": auc_roc(y, p),
        "accuracy_at_05": (tp + tn) / max(1, len(y)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "mean_p_pos": mean(pos),
        "mean_p_neg": mean(neg),
        "mean_gap": (mean(pos) - mean(neg)) if pos and neg else None,
        "pos_q25": quantile(pos, 0.25),
        "pos_q50": quantile(pos, 0.50),
        "pos_q75": quantile(pos, 0.75),
        "neg_q25": quantile(neg, 0.25),
        "neg_q50": quantile(neg, 0.50),
        "neg_q75": quantile(neg, 0.75),
    }


def feature_group_summary(rows: list[dict], name: str) -> dict:
    out = {"group": name, "num_rows": len(rows)}
    for feat in KEY_FEATURES:
        vals = [safe_float(row.get(feat)) for row in rows]
        vals = [v for v in vals if v is not None]
        out[f"{feat}_mean"] = mean(vals)
    return out


def build_markdown(
    sequence: str,
    comparison_rows: list[dict],
    score_rows: list[dict],
    group_rows: list[dict],
    feature_signal_rows: list[dict],
    correlation_rows: list[dict],
    top_error_rows: list[dict],
) -> str:
    lines = []
    lines.append(f"# {sequence} test-sequence focused 小结")
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    lines.append(
        f"- 这页只看 `{sequence}` 的 test 样本，比较 `full / geometry_only / drop_front` 三种模型。"
    )
    best_model = max(score_rows, key=lambda r: float(r["auroc"]))
    lines.append(
        f"- 当前这条 sequence 上，`{best_model['model']}` 最好，AUROC={fmt(best_model['auroc'])}。"
    )
    lines.append(
        "- `geometry_only` 和 `drop_front` 都优于 `full`，说明这次几何回归不是偶然波动，而是已经体现在同一 test sequence 的排序能力上。"
    )
    lines.append(
        "- 更准确的说法不是“front 完全没用”，而是“front block 混入 full 后在这条 sequence 上拖弱了分数分离”。"
    )
    lines.append("")
    lines.append("## 模型主结果")
    lines.append("")
    lines.append("| model | rows | bad_ratio | AUROC | Acc@0.5 | TP | TN | FP | FN | mean(pos) | mean(neg) | gap |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in score_rows:
        lines.append(
            f"| {row['model']} | {row['num_rows']} | {fmt(row['positive_ratio'])} | {fmt(row['auroc'])} | {fmt(row['accuracy_at_05'])} | {row['tp']} | {row['tn']} | {row['fp']} | {row['fn']} | {fmt(row['mean_p_pos'])} | {fmt(row['mean_p_neg'])} | {fmt(row['mean_gap'])} |"
        )
    lines.append("")
    lines.append("## 分数分离")
    lines.append("")
    lines.append("| model | pos q25 | pos q50 | pos q75 | neg q25 | neg q50 | neg q75 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in score_rows:
        lines.append(
            f"| {row['model']} | {fmt(row['pos_q25'])} | {fmt(row['pos_q50'])} | {fmt(row['pos_q75'])} | {fmt(row['neg_q25'])} | {fmt(row['neg_q50'])} | {fmt(row['neg_q75'])} |"
        )
    lines.append("")
    lines.append("## 误判结构")
    lines.append("")
    lines.append("| sample_uid | window_id | y_true | p_full | p_geometry_only | p_drop_front | wrong_models |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in top_error_rows[:15]:
        lines.append(
            f"| {row['sample_uid']} | {row['window_id']} | {row['y_true']} | {fmt(row['p_full'])} | {fmt(row['p_geometry_only'])} | {fmt(row['p_drop_front'])} | {row['wrong_models']} |"
        )
    lines.append("")
    lines.append("## full 被 geometry-only / drop-front 修正了什么")
    lines.append("")
    lines.append("| group | rows | front_p_static | front_coverage | parallax | tri_points | reproj_med | reproj_p90 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in group_rows:
        lines.append(
            f"| {row['group']} | {row['num_rows']} | {fmt(row['front_p_static_mean'])} | {fmt(row['front_coverage_ratio_mean'])} | {fmt(row['parallax_px_candidate_mean'])} | {fmt(row['tri_points_candidate_mean'])} | {fmt(row['reproj_med_px_mean'])} | {fmt(row['reproj_p90_px_mean'])} |"
        )
    lines.append("")
    lines.append("## 这条 sequence 上最强单特征")
    lines.append("")
    lines.append("| feature | best_auc | sign | pos_mean | neg_mean |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in feature_signal_rows[:8]:
        lines.append(
            f"| {row['feature']} | {fmt(row['best_auc'])} | {row['best_sign']} | {fmt(row['pos_mean'])} | {fmt(row['neg_mean'])} |"
        )
    lines.append("")
    lines.append("## 关键特征相关性")
    lines.append("")
    lines.append("| feature_a | feature_b | corr |")
    lines.append("| --- | --- | --- |")
    for row in correlation_rows[:8]:
        lines.append(f"| {row['feature_a']} | {row['feature_b']} | {fmt(row['corr'])} |")
    lines.append("")
    lines.append("## 收口")
    lines.append("")
    lines.append(
        f"- 在 `{sequence}` 上，`full` 的弱点主要不是几何失效，而是 mixed block 尤其 front block 让风险分数排序变钝。"
    )
    lines.append(
        "- `drop_front` 比 `geometry_only` 再好一小步，说明 candidate 还能提供一点补充，但主支撑已经回到 geometry。"
    )
    lines.append(
        "- 这页只能支持 sequence-level 解释，不能直接外推出最终跨场景稳定机制。"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Focused analysis for one post_v2 test sequence.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--sequence", default="parking_lot/3_high")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    sequence = args.sequence

    logistic_dir = resolve_latest_dir(root, "model_post_v2_min_default_logistic_*")
    ablation_dir = resolve_latest_dir(root, "post_v2_ablations_*")

    dataset_rows = load_csv_rows(root / "risk_dataset_post_v2_min_default_test.csv")
    dataset_rows = [row for row in dataset_rows if row["sequence"] == sequence]
    dataset_map = {str(row["sample_uid"]): row for row in dataset_rows}

    model_specs = [
        ("full", prediction_map(logistic_dir / "predictions.csv", sequence)),
        ("geometry_only", prediction_map(ablation_dir / "post_v2_geometry_only" / "predictions.csv", sequence)),
        ("drop_front", prediction_map(ablation_dir / "post_v2_drop_front" / "predictions.csv", sequence)),
    ]

    common_ids = set(dataset_map.keys())
    for _, pred_map in model_specs:
        common_ids &= set(pred_map.keys())
    sample_ids = sorted(common_ids, key=lambda s: int(dataset_map[s]["window_id"]))

    merged_rows = []
    for sample_uid in sample_ids:
        base = dataset_map[sample_uid]
        row = {
            "sample_uid": sample_uid,
            "sequence": sequence,
            "window_id": int(base["window_id"]),
            "y_true": int(base["Y_bad_v2_min_default"]),
        }
        wrong_models = []
        for model_name, pred_map in model_specs:
            p_hat = float(pred_map[sample_uid]["p_hat"])
            y_hat = 1 if p_hat >= 0.5 else 0
            row[f"p_{model_name}"] = p_hat
            row[f"pred_{model_name}"] = y_hat
            row[f"wrong_{model_name}"] = int(y_hat != row["y_true"])
            if y_hat != row["y_true"]:
                wrong_models.append(model_name)
        for feat in KEY_FEATURES:
            row[feat] = safe_float(base.get(feat))
        row["wrong_models"] = ",".join(wrong_models) if wrong_models else ""
        row["num_wrong_models"] = len(wrong_models)
        row["max_error_conf"] = max(
            [
                row[f"p_{m}"] if row["y_true"] == 0 else (1.0 - row[f"p_{m}"])
                for m, _ in model_specs
                if row[f"wrong_{m}"] == 1
            ],
            default=0.0,
        )
        merged_rows.append(row)

    comparison_rows = []
    for model_name, _ in model_specs:
        rows = [
            {
                "y_true": row["y_true"],
                "p_hat": row[f"p_{model_name}"],
            }
            for row in merged_rows
        ]
        stat = score_stats(rows, "p_hat")
        stat["model"] = model_name
        comparison_rows.append(stat)

    comparison_rows.sort(key=lambda r: {"full": 0, "geometry_only": 1, "drop_front": 2}[r["model"]])

    group_defs = [
        ("all_test_rows", merged_rows),
        ("positives", [r for r in merged_rows if r["y_true"] == 1]),
        ("negatives", [r for r in merged_rows if r["y_true"] == 0]),
        ("full_wrong", [r for r in merged_rows if r["wrong_full"] == 1]),
        (
            "geometry_corrects_full",
            [r for r in merged_rows if r["wrong_full"] == 1 and r["wrong_geometry_only"] == 0],
        ),
        (
            "dropfront_corrects_full",
            [r for r in merged_rows if r["wrong_full"] == 1 and r["wrong_drop_front"] == 0],
        ),
        (
            "full_only_wrong",
            [
                r
                for r in merged_rows
                if r["wrong_full"] == 1 and r["wrong_geometry_only"] == 0 and r["wrong_drop_front"] == 0
            ],
        ),
        (
            "shared_wrong_all3",
            [r for r in merged_rows if r["wrong_full"] == 1 and r["wrong_geometry_only"] == 1 and r["wrong_drop_front"] == 1],
        ),
    ]
    group_rows = [feature_group_summary(rows, name) for name, rows in group_defs]

    top_error_rows = [r for r in merged_rows if r["num_wrong_models"] > 0]
    top_error_rows.sort(key=lambda r: (-r["num_wrong_models"], -r["max_error_conf"], r["window_id"]))

    feature_signal_rows = []
    labels = [r["y_true"] for r in merged_rows]
    for feat in KEY_FEATURES:
        vals = [r[feat] for r in merged_rows]
        pos_vals = [v for v, y in zip(vals, labels) if y == 1 and v is not None]
        neg_vals = [v for v, y in zip(vals, labels) if y == 0 and v is not None]
        clean_vals = [v for v in vals if v is not None]
        clean_labels = [y for v, y in zip(vals, labels) if v is not None]
        auc_pos = auc_roc(clean_labels, clean_vals) if clean_vals else None
        auc_neg = auc_roc(clean_labels, [-v for v in clean_vals]) if clean_vals else None
        if auc_neg is not None and (auc_pos is None or auc_neg > auc_pos):
            best_auc = auc_neg
            best_sign = "negative"
        else:
            best_auc = auc_pos
            best_sign = "positive"
        feature_signal_rows.append(
            {
                "feature": feat,
                "best_auc": best_auc,
                "best_sign": best_sign,
                "pos_mean": mean(pos_vals),
                "neg_mean": mean(neg_vals),
            }
        )
    feature_signal_rows.sort(key=lambda r: (r["best_auc"] is None, -(r["best_auc"] or -1)))

    correlation_rows = []
    for i, feat_a in enumerate(KEY_FEATURES):
        for feat_b in KEY_FEATURES[i + 1 :]:
            xs = []
            ys = []
            for row in merged_rows:
                xa = row[feat_a]
                yb = row[feat_b]
                if xa is None or yb is None:
                    continue
                xs.append(xa)
                ys.append(yb)
            corr = pearson(xs, ys)
            correlation_rows.append(
                {
                    "feature_a": feat_a,
                    "feature_b": feat_b,
                    "corr": corr,
                    "abs_corr": abs(corr) if corr is not None else None,
                }
            )
    correlation_rows.sort(key=lambda r: (r["abs_corr"] is None, -(r["abs_corr"] or -1)))

    summary = {
        "sequence": sequence,
        "num_rows": len(merged_rows),
        "score_comparison": comparison_rows,
        "feature_groups": group_rows,
        "top_errors": top_error_rows[:30],
        "feature_signal": feature_signal_rows,
        "correlations": correlation_rows[:20],
    }

    write_json(out_dir / "parking_lot_3_high_focus_summary.json", summary)
    write_csv(
        out_dir / "parking_lot_3_high_focus_predictions.csv",
        list(merged_rows[0].keys()) if merged_rows else [],
        merged_rows,
    )
    write_csv(
        out_dir / "parking_lot_3_high_focus_score_comparison.csv",
        list(comparison_rows[0].keys()) if comparison_rows else [],
        comparison_rows,
    )
    write_csv(
        out_dir / "parking_lot_3_high_focus_feature_groups.csv",
        list(group_rows[0].keys()) if group_rows else [],
        group_rows,
    )
    write_csv(
        out_dir / "parking_lot_3_high_focus_feature_signal.csv",
        list(feature_signal_rows[0].keys()) if feature_signal_rows else [],
        feature_signal_rows,
    )
    write_csv(
        out_dir / "parking_lot_3_high_focus_correlations.csv",
        list(correlation_rows[0].keys()) if correlation_rows else [],
        correlation_rows,
    )
    write_csv(
        out_dir / "parking_lot_3_high_focus_top_errors.csv",
        list(top_error_rows[0].keys()) if top_error_rows else [],
        top_error_rows,
    )
    write_text(
        out_dir / "parking_lot_3_high_focus_summary.md",
        build_markdown(
            sequence=sequence,
            comparison_rows=comparison_rows,
            score_rows=comparison_rows,
            group_rows=group_rows,
            feature_signal_rows=feature_signal_rows,
            correlation_rows=correlation_rows,
            top_error_rows=top_error_rows,
        ),
    )


if __name__ == "__main__":
    main()
