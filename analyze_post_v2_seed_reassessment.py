#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import math


FRONT_COLS = [
    "front_p_static",
    "front_p_band",
    "front_coverage_ratio",
    "front_kept_dyn_ratio",
]

CANDIDATE_COLS = [
    "parallax_px_candidate",
    "tri_points_candidate",
    "pnp_success_rate",
]

GEOMETRY_COLS = [
    "reproj_med_px",
    "reproj_p90_px",
    "cheirality_ratio",
]

SELECTED_FEATURES = FRONT_COLS + CANDIDATE_COLS + GEOMETRY_COLS


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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def sequence_block_summary(split_rows: list[dict]) -> list[dict]:
    seq_map: dict[str, list[dict]] = {}
    for row in split_rows:
        seq_map.setdefault(str(row["sequence"]), []).append(row)
    out = []
    summary_features = [
        "front_p_static",
        "front_coverage_ratio",
        "parallax_px_candidate",
        "tri_points_candidate",
        "reproj_med_px",
        "reproj_p90_px",
    ]
    for seq, rows in sorted(seq_map.items()):
        y = [int(row["Y_bad_v2_min_default"]) for row in rows]
        item = {
            "sequence": seq,
            "split": str(rows[0]["dataset_row_split"]),
            "num_rows": len(rows),
            "bad_ratio": sum(y) / max(1, len(y)),
        }
        for feat in summary_features:
            vals = [safe_float(row.get(feat)) for row in rows]
            vals = [v for v in vals if v is not None]
            item[f"{feat}_mean"] = mean(vals)
        out.append(item)
    return out


def per_sequence_model_rows(pred_rows: list[dict], model_name: str) -> list[dict]:
    seq_map: dict[str, list[dict]] = {}
    for row in pred_rows:
        seq_map.setdefault(str(row["sequence"]), []).append(row)
    out = []
    for seq, rows in sorted(seq_map.items()):
        y = [int(r["y_true"]) for r in rows]
        p = [float(r["p_hat"]) for r in rows]
        pos = [pp for yy, pp in zip(y, p) if yy == 1]
        neg = [pp for yy, pp in zip(y, p) if yy == 0]
        out.append(
            {
                "model": model_name,
                "sequence": seq,
                "split": str(rows[0]["split"]),
                "num_rows": len(rows),
                "positive_ratio": sum(y) / max(1, len(y)),
                "auroc": auc_roc(y, p),
                "mean_p_pos": mean(pos),
                "mean_p_neg": mean(neg),
                "mean_p_gap": (mean(pos) - mean(neg)) if pos and neg else None,
            }
        )
    return out


def selected_feature_signal(split_rows_map: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for feat in SELECTED_FEATURES:
        row = {"feature": feat}
        for split_name, split_rows in split_rows_map.items():
            vals = []
            ys = []
            for split_row in split_rows:
                x = safe_float(split_row.get(feat))
                y = safe_float(split_row.get("Y_bad_v2_min_default"))
                if x is None or y is None:
                    continue
                vals.append(x)
                ys.append(int(y))
            auc_best, sign = best_signed_auc(ys, vals) if vals else (None, "na")
            row[f"{split_name}_best_auc"] = auc_best
            row[f"{split_name}_best_auc_sign"] = sign
            row[f"{split_name}_mean"] = mean(vals)
            row[f"{split_name}_num_rows"] = len(vals)
        rows.append(row)
    return rows


def selected_feature_drift(split_rows_map: dict[str, list[dict]]) -> list[dict]:
    out = []
    for feat in SELECTED_FEATURES:
        vals = {}
        for split_name, split_rows in split_rows_map.items():
            xs = [safe_float(row.get(feat)) for row in split_rows]
            vals[split_name] = [x for x in xs if x is not None]
        out.append(
            {
                "feature": feat,
                "train_mean": mean(vals["train"]),
                "val_mean": mean(vals["val"]),
                "test_mean": mean(vals["test"]),
                "train_std": std(vals["train"]),
                "val_std": std(vals["val"]),
                "test_std": std(vals["test"]),
                "smd_train_val": standardized_mean_diff(vals["train"], vals["val"]),
                "smd_train_test": standardized_mean_diff(vals["train"], vals["test"]),
                "smd_val_test": standardized_mean_diff(vals["val"], vals["test"]),
            }
        )
    return out


def train_feature_correlations(train_rows: list[dict]) -> list[dict]:
    out = []
    feat_vals = {feat: [safe_float(row.get(feat)) for row in train_rows] for feat in SELECTED_FEATURES}
    for i, fa in enumerate(SELECTED_FEATURES):
        for fb in SELECTED_FEATURES[i + 1:]:
            pairs = [
                (a, b)
                for a, b in zip(feat_vals[fa], feat_vals[fb])
                if a is not None and b is not None
            ]
            corr = pearson([a for a, _ in pairs], [b for _, b in pairs]) if pairs else None
            out.append(
                {
                    "feature_a": fa,
                    "feature_b": fb,
                    "pearson_r_train": corr,
                    "abs_r_train": abs(corr) if corr is not None else None,
                    "num_pairs": len(pairs),
                }
            )
    out.sort(key=lambda row: row["abs_r_train"] if row["abs_r_train"] is not None else -1.0, reverse=True)
    return out


def build_markdown_summary(summary: dict) -> str:
    lines = []
    lines.append("# post_v2 9seq seed20260432 结构再评估")
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    for item in summary["headline"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 主结果")
    lines.append("")
    lines.append("| 模型 | test AUROC | test AUPRC | 备注 |")
    lines.append("| --- | --- | --- | --- |")
    for row in summary["main_result_rows"]:
        lines.append(f"| {row['name']} | {fmt(row['auroc'])} | {fmt(row['auprc'])} | {row['note']} |")
    lines.append("")
    lines.append("## 为什么这次几何又回来了")
    lines.append("")
    for item in summary["geometry_return_explanation"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 现在能说与不能说")
    lines.append("")
    for item in summary["claim_safe"]:
        lines.append(f"- Allowed: {item}")
    for item in summary["claim_forbidden"]:
        lines.append(f"- Forbidden: {item}")
    lines.append("")
    lines.append("## 下一步建议")
    lines.append("")
    for item in summary["next_steps"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def build_markdown_analysis(payload: dict) -> str:
    lines = []
    lines.append("# post_v2 9seq seed20260432 结构再评估分析")
    lines.append("")
    lines.append("## 包与 split")
    lines.append("")
    lines.append(f"- result_root: `{payload['result_root']}`")
    lines.append(f"- seed: `{payload['split_manifest']['seed']}`")
    lines.append(f"- sequence_to_split: `{payload['split_manifest']['sequence_to_split']}`")
    lines.append("")
    lines.append("## 新增 val/test 序列贡献")
    lines.append("")
    for row in payload["key_sequence_rows"]:
        lines.append(
            f"- `{row['sequence']}` ({row['split']}): rows={row['num_rows']}, bad_ratio={fmt(row['bad_ratio'])}, "
            f"front_p_static_mean={fmt(row['front_p_static_mean'])}, "
            f"parallax_mean={fmt(row['parallax_px_candidate_mean'])}, "
            f"reproj_p90_mean={fmt(row['reproj_p90_px_mean'])}"
        )
    lines.append("")
    lines.append("## full / geometry_only / drop_front 按序列表现")
    lines.append("")
    lines.append("| sequence | split | model | n | pos_ratio | AUROC | mean_p_pos | mean_p_neg |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["sequence_model_rows"]:
        lines.append(
            f"| {row['sequence']} | {row['split']} | {row['model']} | {row['num_rows']} | "
            f"{fmt(row['positive_ratio'])} | {fmt(row['auroc'])} | {fmt(row['mean_p_pos'])} | {fmt(row['mean_p_neg'])} |"
        )
    lines.append("")
    lines.append("## 选定特征的单特征信号")
    lines.append("")
    lines.append("| feature | train best AUC | val best AUC | test best AUC | test sign |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in payload["feature_signal_rows"]:
        lines.append(
            f"| {row['feature']} | {fmt(row['train_best_auc'])} | {fmt(row['val_best_auc'])} | "
            f"{fmt(row['test_best_auc'])} | {row['test_best_auc_sign']} |"
        )
    lines.append("")
    lines.append("## 选定特征的 split drift")
    lines.append("")
    lines.append("| feature | train_mean | val_mean | test_mean | SMD(train,test) |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in payload["feature_drift_rows"]:
        lines.append(
            f"| {row['feature']} | {fmt(row['train_mean'])} | {fmt(row['val_mean'])} | "
            f"{fmt(row['test_mean'])} | {fmt(row['smd_train_test'])} |"
        )
    lines.append("")
    lines.append("## 训练相关性 Top")
    lines.append("")
    lines.append("| feature_a | feature_b | pearson_r_train |")
    lines.append("| --- | --- | --- |")
    for row in payload["top_corr_rows"]:
        lines.append(f"| {row['feature_a']} | {row['feature_b']} | {fmt(row['pearson_r_train'])} |")
    lines.append("")
    lines.append("## 系数对比")
    lines.append("")
    lines.append("| feature | full | geometry_only | drop_front |")
    lines.append("| --- | --- | --- | --- |")
    for row in payload["coefficient_rows"]:
        lines.append(
            f"| {row['feature']} | {fmt(row['full_weight'])} | {fmt(row['geometry_only_weight'])} | {fmt(row['drop_front_weight'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Analyze post_v2 seed-specific reassessment focusing on full, geometry_only, and drop_front.")
    ap.add_argument("--result_root", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    root = Path(args.result_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_manifest = load_json(root / "dataset_split_manifest.json")
    task_manifest = load_json(root / "risk_dataset_post_v2_min_default_manifest.json")
    logistic_dir = resolve_latest_dir(root, "model_post_v2_min_default_logistic_*")
    ablation_dir = resolve_latest_dir(root, "post_v2_ablations_*")

    full_metrics = load_json(logistic_dir / "metrics.json")
    full_coeffs = {row["feature"]: safe_float(row["weight"]) for row in load_csv_rows(logistic_dir / "coefficients.csv")}
    geom_coeffs = {
        row["feature"]: safe_float(row["weight"])
        for row in load_csv_rows(ablation_dir / "post_v2_geometry_only" / "coefficients.csv")
    }
    drop_front_coeffs = {
        row["feature"]: safe_float(row["weight"])
        for row in load_csv_rows(ablation_dir / "post_v2_drop_front" / "coefficients.csv")
    }

    split_rows_map = {
        "train": load_csv_rows(root / "risk_dataset_post_v2_min_default_train.csv"),
        "val": load_csv_rows(root / "risk_dataset_post_v2_min_default_val.csv"),
        "test": load_csv_rows(root / "risk_dataset_post_v2_min_default_test.csv"),
    }
    all_rows = split_rows_map["train"] + split_rows_map["val"] + split_rows_map["test"]

    sequence_summary_rows = sequence_block_summary(all_rows)
    key_sequence_rows = [row for row in sequence_summary_rows if row["sequence"] in ("city_night/0_none", "parking_lot/3_high")]

    prediction_specs = {
        "full": ablation_dir / "post_v2_full" / "predictions.csv",
        "geometry_only": ablation_dir / "post_v2_geometry_only" / "predictions.csv",
        "drop_front": ablation_dir / "post_v2_drop_front" / "predictions.csv",
    }
    sequence_model_rows = []
    for model_name, pred_path in prediction_specs.items():
        sequence_model_rows.extend(per_sequence_model_rows(load_csv_rows(pred_path), model_name))
    sequence_model_rows.sort(key=lambda row: (row["sequence"], row["model"]))

    feature_signal_rows = selected_feature_signal(split_rows_map)
    feature_drift_rows = selected_feature_drift(split_rows_map)
    corr_rows = train_feature_correlations(split_rows_map["train"])
    top_corr_rows = corr_rows[:12]

    coeff_features = sorted(
        {k for k in full_coeffs if k != "__intercept__"}
        | {k for k in geom_coeffs if k != "__intercept__"}
        | {k for k in drop_front_coeffs if k != "__intercept__"}
    )
    coefficient_rows = []
    for feat in coeff_features:
        coefficient_rows.append(
            {
                "feature": feat,
                "full_weight": full_coeffs.get(feat),
                "geometry_only_weight": geom_coeffs.get(feat),
                "drop_front_weight": drop_front_coeffs.get(feat),
            }
        )

    main_result_rows = [
        {
            "name": "rule_q_post_geom_only",
            "auroc": full_metrics["rule_baseline"]["test"]["auroc"],
            "auprc": full_metrics["rule_baseline"]["test"]["auprc"],
            "note": "Current rule baseline on the same test split.",
        },
        {
            "name": "post_v2_full",
            "auroc": full_metrics["splits"]["test"]["auroc"],
            "auprc": full_metrics["splits"]["test"]["auprc"],
            "note": "Current full learning baseline.",
        },
        {
            "name": "post_v2_geometry_only",
            "auroc": next(row for row in load_csv_rows(ablation_dir / "post_v2_ablation_summary.csv") if row["name"] == "post_v2_geometry_only")["auroc"],
            "auprc": next(row for row in load_csv_rows(ablation_dir / "post_v2_ablation_summary.csv") if row["name"] == "post_v2_geometry_only")["auprc"],
            "note": "Geometry-only learning variant.",
        },
        {
            "name": "post_v2_drop_front",
            "auroc": next(row for row in load_csv_rows(ablation_dir / "post_v2_ablation_summary.csv") if row["name"] == "post_v2_drop_front")["auroc"],
            "auprc": next(row for row in load_csv_rows(ablation_dir / "post_v2_ablation_summary.csv") if row["name"] == "post_v2_drop_front")["auprc"],
            "note": "Remove front block, keep candidate + geometry.",
        },
    ]

    seq_model_map = {(row["sequence"], row["model"]): row for row in sequence_model_rows}
    test_signal_map = {row["feature"]: row for row in feature_signal_rows}
    summary = {
        "result_root": str(root),
        "headline": [
            "在 9seq seed20260432 这版更符合 val/test 扩样本目标的 split 上，`geometry_only` 和 `drop_front` 都超过了 `full`，说明当前后验结构重新朝几何主导方向收紧。",
            f"`drop_front` 是当前最优 learning variant，test AUROC={fmt(next(r for r in main_result_rows if r['name'] == 'post_v2_drop_front')['auroc'])}；`geometry_only` 也达到 {fmt(next(r for r in main_result_rows if r['name'] == 'post_v2_geometry_only')['auroc'])}；而 `full` 只有 {fmt(next(r for r in main_result_rows if r['name'] == 'post_v2_full')['auroc'])}。",
            f"但规则式 `Q_post_geom_only` 仍然更高，test AUROC={fmt(next(r for r in main_result_rows if r['name'] == 'rule_q_post_geom_only')['auroc'])}。所以这轮更像“几何回归 + full mixing 失效”，还不是 learning post 已经稳定超过规则式。",
        ],
        "main_result_rows": main_result_rows,
        "geometry_return_explanation": [
            f"两个新补进来的关键序列本身就更支持 geometry-heavy 变体。`city_night/0_none` 在 val 上，`geometry_only` 的 sequence-level AUROC={fmt(seq_model_map[('city_night/0_none', 'geometry_only')]['auroc'])}，高于 `full` 的 {fmt(seq_model_map[('city_night/0_none', 'full')]['auroc'])}；`parking_lot/3_high` 在 test 上，`geometry_only` 的 AUROC={fmt(seq_model_map[('parking_lot/3_high', 'geometry_only')]['auroc'])}，也高于 `full` 的 {fmt(seq_model_map[('parking_lot/3_high', 'full')]['auroc'])}。",
            f"`drop_front` 在 test 上进一步升到 {fmt(seq_model_map[('parking_lot/3_high', 'drop_front')]['auroc'])}，说明这次提升不是 candidate-only 回来了，而是“保留 geometry/candidate、删掉 front”更稳。",
            f"几何权重在 `full` 和 `drop_front` 之间几乎没变：`reproj_p90_px` 从 {fmt(full_coeffs.get('reproj_p90_px'))} 到 {fmt(drop_front_coeffs.get('reproj_p90_px'))}，`reproj_med_px` 从 {fmt(full_coeffs.get('reproj_med_px'))} 到 {fmt(drop_front_coeffs.get('reproj_med_px'))}。这说明增益主要来自去掉 front block 的噪声，而不是学出了全新的几何机制。",
            f"当前 split 的 test 单特征里，几何特征重新抬头：`reproj_med_px` best test AUC={fmt(test_signal_map['reproj_med_px']['test_best_auc'])}，`reproj_p90_px`={fmt(test_signal_map['reproj_p90_px']['test_best_auc'])}，都高于 `candidate_only` 里最强的 `tri_points_candidate`={fmt(test_signal_map['tri_points_candidate']['test_best_auc'])}。不过 `front_p_static` 单特征其实更高，达到 {fmt(test_signal_map['front_p_static']['test_best_auc'])}，但 `front_only` 整块仍然很差，说明问题不是 front 完全无信号，而是 front block 的组合方式在伤害模型。",
            f"训练相关性也支持这一点：`front_p_static` 与 `front_coverage_ratio` 的相关系数是 {fmt(next(r['pearson_r_train'] for r in top_corr_rows if r['feature_a'] == 'front_p_static' and r['feature_b'] == 'front_coverage_ratio'))}，`front_coverage_ratio` 与 `tri_points_candidate` 是 {fmt(next(r['pearson_r_train'] for r in top_corr_rows if r['feature_a'] == 'front_coverage_ratio' and r['feature_b'] == 'tri_points_candidate'))}。front block 既有自身相关，又会牵连 candidate，full mixing 更容易把有用的几何/候选信息搅乱。",
        ],
        "claim_safe": [
            "可以说在 9seq seed20260432 上，geometry-only 和 drop-front 都优于 full，当前更像 geometry-dominant with harmful front mixing。",
            "可以说这次几何回归与新增 val/test 序列的 sequence mix 明显相关，因为两条关键新序列都更偏向 geometry-heavy 变体。",
            "可以说 front block 现在不适合再被当成主判别层；它更像包含少量有用单特征，但当前 block 组合方式会拖累整体模型。",
        ],
        "claim_forbidden": [
            "不能说 geometry-dominated 已经被稳定证明，因为 test 仍只由 `parking_lot/3_high` 一条序列承担。",
            "不能说 learning post 已经超过规则式几何基线，因为当前 rule test AUROC 仍高于 full。",
            "不能把 `front_p_static` 的单特征强信号误写成 front block 整体有效，当前 `front_only` 明显失败。",
        ],
        "next_steps": [
            "下一步先针对 `full / geometry_only / drop_front` 再做一页 test-sequence focused 小结，只盯 `parking_lot/3_high` 的误判样本。",
            "在没解释清 front block 为什么伤害 full 之前，不继续推进更复杂的 post 学习器。",
            "如果继续扩样本，优先再补新的 test sequence，而不是继续加 train。当前最缺的是第二条真正的 post_v2 test 序列。",
        ],
    }

    payload = {
        "result_root": str(root),
        "split_manifest": split_manifest,
        "task_manifest": task_manifest,
        "summary": summary,
        "key_sequence_rows": key_sequence_rows,
        "sequence_model_rows": sequence_model_rows,
        "feature_signal_rows": feature_signal_rows,
        "feature_drift_rows": feature_drift_rows,
        "top_corr_rows": top_corr_rows,
        "coefficient_rows": coefficient_rows,
    }

    write_json(out_dir / "post_v2_seed20260432_reassessment.json", payload)
    write_csv(
        out_dir / "post_v2_seed20260432_sequence_model_comparison.csv",
        ["sequence", "split", "model", "num_rows", "positive_ratio", "auroc", "mean_p_pos", "mean_p_neg", "mean_p_gap"],
        sequence_model_rows,
    )
    write_csv(
        out_dir / "post_v2_seed20260432_sequence_feature_summary.csv",
        list(sequence_summary_rows[0].keys()) if sequence_summary_rows else ["sequence", "split", "num_rows", "bad_ratio"],
        sequence_summary_rows,
    )
    write_csv(
        out_dir / "post_v2_seed20260432_feature_signal.csv",
        list(feature_signal_rows[0].keys()) if feature_signal_rows else ["feature"],
        feature_signal_rows,
    )
    write_csv(
        out_dir / "post_v2_seed20260432_feature_drift.csv",
        list(feature_drift_rows[0].keys()) if feature_drift_rows else ["feature"],
        feature_drift_rows,
    )
    write_csv(
        out_dir / "post_v2_seed20260432_train_correlations.csv",
        list(top_corr_rows[0].keys()) if top_corr_rows else ["feature_a", "feature_b", "pearson_r_train", "abs_r_train", "num_pairs"],
        top_corr_rows,
    )
    write_csv(
        out_dir / "post_v2_seed20260432_coefficients.csv",
        ["feature", "full_weight", "geometry_only_weight", "drop_front_weight"],
        coefficient_rows,
    )
    write_text(out_dir / "post_v2_seed20260432_reassessment_summary.md", build_markdown_summary(summary))
    write_text(out_dir / "post_v2_seed20260432_reassessment_analysis.md", build_markdown_analysis(payload))
    print(f"[PostV2SeedReassessment] saved -> {out_dir}")


if __name__ == "__main__":
    main()
