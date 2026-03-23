#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json


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


def load_result_bundle(root: Path) -> dict:
    logistic_dir = resolve_latest_dir(root, "model_post_v2_min_default_logistic_*")
    dirrisk_dir = resolve_latest_dir(root, "model_post_v2_min_default_dir_risk_*")
    ablation_dir = resolve_latest_dir(root, "post_v2_ablations_*")
    return {
        "logistic_metrics": load_json(logistic_dir / "metrics.json"),
        "dirrisk_metrics": load_json(dirrisk_dir / "metrics.json"),
        "ablation_rows": load_csv_rows(ablation_dir / "post_v2_ablation_summary.csv"),
        "dataset_split": load_json(root / "dataset_split_manifest.json"),
        "task_manifest": load_json(root / "risk_dataset_post_v2_min_default_manifest.json"),
    }


def ablation_map(rows: list[dict]) -> dict[str, dict]:
    return {str(r["name"]): r for r in rows}


def build_main_comparison(old_bundle: dict, new_bundle: dict) -> list[dict]:
    old_log = old_bundle["logistic_metrics"]
    new_log = new_bundle["logistic_metrics"]
    old_dir = old_bundle["dirrisk_metrics"]
    new_dir = new_bundle["dirrisk_metrics"]
    rows = []
    specs = [
        (
            "post_v2_rule_q_post_geom_only",
            old_log["rule_baseline"]["test"],
            new_log["rule_baseline"]["test"],
        ),
        (
            "post_v2_logistic_full",
            old_log["splits"]["test"],
            new_log["splits"]["test"],
        ),
        (
            "post_v2_dir_risk",
            old_dir["splits"]["test"],
            new_dir["splits"]["test"],
        ),
    ]
    for name, old_t, new_t in specs:
        rows.append(
            {
                "name": name,
                "old_num_rows": old_t["num_rows"],
                "new_num_rows": new_t["num_rows"],
                "old_positive_ratio": old_t["positive_ratio"],
                "new_positive_ratio": new_t["positive_ratio"],
                "old_auroc": old_t["auroc"],
                "new_auroc": new_t["auroc"],
                "delta_auroc": float(new_t["auroc"]) - float(old_t["auroc"]),
                "old_auprc": old_t["auprc"],
                "new_auprc": new_t["auprc"],
                "old_brier": old_t["brier"],
                "new_brier": new_t["brier"],
                "old_ece": old_t["ece"],
                "new_ece": new_t["ece"],
            }
        )
    return rows


def build_ablation_shift(old_rows: list[dict], new_rows: list[dict]) -> list[dict]:
    old_map = ablation_map(old_rows)
    out = []
    for new_row in new_rows:
        name = str(new_row["name"])
        old_row = old_map.get(name, {})
        out.append(
            {
                "name": name,
                "old_auroc": old_row.get("auroc", ""),
                "new_auroc": new_row.get("auroc", ""),
                "delta_auroc": (
                    float(new_row["auroc"]) - float(old_row["auroc"])
                    if old_row.get("auroc", "") not in ("", None) else None
                ),
                "old_auprc": old_row.get("auprc", ""),
                "new_auprc": new_row.get("auprc", ""),
                "note": new_row.get("note", ""),
            }
        )
    out.sort(key=lambda r: float(r["new_auroc"]), reverse=True)
    return out


def build_markdown(summary: dict, main_rows: list[dict], ablation_rows: list[dict]) -> str:
    lines = []
    lines.append("# post_v2 结构结论再漂移")
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    for item in summary["headline_judgement"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 5 条到 7 条主结果变化")
    lines.append("")
    lines.append("| 模型 | 5 条 test AUROC | 7 条 test AUROC | 变化 |")
    lines.append("| --- | --- | --- | --- |")
    for row in main_rows:
        lines.append(
            f"| {row['name']} | {fmt(row['old_auroc'])} | {fmt(row['new_auroc'])} | {fmt(row['delta_auroc'])} |"
        )
    lines.append("")
    lines.append("## 7 条集的 block 排序")
    lines.append("")
    lines.append("| Ablation | 5 条 AUROC | 7 条 AUROC | 变化 |")
    lines.append("| --- | --- | --- | --- |")
    for row in ablation_rows:
        lines.append(
            f"| {row['name']} | {fmt(row['old_auroc'])} | {fmt(row['new_auroc'])} | {fmt(row['delta_auroc'])} |"
        )
    lines.append("")
    lines.append("## 为什么 candidate_only 跑到第一")
    lines.append("")
    for item in summary["candidate_explanation"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 批判性收口")
    lines.append("")
    for item in summary["critical_takeaways"]:
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


def main():
    ap = argparse.ArgumentParser(description="Summarize the second post_v2 structure drift after expanding from 5 to 7 sequences.")
    ap.add_argument("--old_root", required=True)
    ap.add_argument("--new_root", required=True)
    ap.add_argument("--drift_json", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    old_root = Path(args.old_root).expanduser().resolve()
    new_root = Path(args.new_root).expanduser().resolve()
    drift = load_json(Path(args.drift_json).expanduser().resolve())
    old_bundle = load_result_bundle(old_root)
    new_bundle = load_result_bundle(new_root)

    main_rows = build_main_comparison(old_bundle, new_bundle)
    ablation_rows = build_ablation_shift(old_bundle["ablation_rows"], new_bundle["ablation_rows"])

    top_signal = drift["top_test_signal_features"]
    block_summary = {row["block"]: row for row in drift["block_summary"]}
    seq_rows = drift["sequence_contribution"]
    test_rows = [row for row in seq_rows if row["split"] == "test"]
    train_rows = [row for row in seq_rows if row["split"] == "train"]

    test_seq = test_rows[0]["sequence"] if test_rows else "<none>"
    candidate_top = [r for r in top_signal if r["block"] == "candidate"][:2]
    geometry_top = [r for r in top_signal if r["block"] == "geometry"][:2]
    train_row_map = {row["sequence"]: row for row in train_rows}

    summary = {
        "old_dataset": {
            "coverage_summary": old_bundle["task_manifest"]["coverage_summary"],
            "test_num_rows": old_bundle["logistic_metrics"]["splits"]["test"]["num_rows"],
            "split": old_bundle["dataset_split"]["sequence_to_split"],
        },
        "new_dataset": {
            "coverage_summary": new_bundle["task_manifest"]["coverage_summary"],
            "test_num_rows": new_bundle["logistic_metrics"]["splits"]["test"]["num_rows"],
            "split": new_bundle["dataset_split"]["sequence_to_split"],
        },
        "headline_judgement": [
            f"从 5 条到 7 条后，post_v2 full logistic 的 test AUROC 从 {fmt(main_rows[1]['old_auroc'])} 进一步降到 {fmt(main_rows[1]['new_auroc'])}，说明结构结论还在继续漂移。",
            f"规则式 `Q_post_geom_only` 反而从 {fmt(main_rows[0]['old_auroc'])} 回升到 {fmt(main_rows[0]['new_auroc'])}，但学习式 full baseline 仍只小幅领先，说明“纯几何不够”仍成立，但优势已经很薄。",
            "7 条集上不只是 geometry-dominated 站不住，连 5 条集里的 `drop_candidate` 最优也被改写了；当前最优变成了 `candidate_only`。",
        ],
        "candidate_explanation": [
            f"test 单特征最强的两个特征都来自 candidate block：`{candidate_top[0]['feature']}` 的 best test AUROC 是 {fmt(candidate_top[0]['test_best_auc'])}，`{candidate_top[1]['feature']}` 是 {fmt(candidate_top[1]['test_best_auc'])}。这说明 `candidate_only` 第一不是偶然，而是 test 上的最强单特征信号本身就在 candidate 里。",
            f"geometry block 并没有完全失效，但它在 7 条集上的单特征上限更低：最强 geometry 特征 `{geometry_top[0]['feature']}` 的 best test AUROC 只有 {fmt(geometry_top[0]['test_best_auc'])}，低于 candidate 的峰值。",
            f"block summary 也支持这一点：candidate 的平均 test 单特征 AUROC 是 {fmt(block_summary['candidate']['avg_test_best_auc'])}，高于 geometry 的 {fmt(block_summary['geometry']['avg_test_best_auc'])}；但 geometry 的 train-test 平均漂移更大，avg|SMD|={fmt(block_summary['geometry']['avg_abs_smd_train_test'])}，高于 candidate 的 {fmt(block_summary['candidate']['avg_abs_smd_train_test'])}。",
            "train 内还存在明显共线性，尤其 `front_coverage_ratio` 与 `tri_points_candidate`、`reproj_med_px` 与 `reproj_p90_px` 相关较强。这意味着 full model 把 block 混在一起后，可能出现信息吸收和相互干扰，反而让 candidate-only 更干净。",
            f"sequence-level 上，7 条集的 test 完全由 `{test_seq}` 承担，而新增进入 train 的 `parking_lot/1_low` 本身就显示出更低的 `front_coverage_ratio`、更低的 `tri_points_candidate`、但更好的序列内 AUROC。这说明 sequence mix 变化正在把模型推向“candidate-stage observability 更敏感”的方向。",
        ],
        "critical_takeaways": [
            "这轮 7 条结果说明我们不能再沿用 5 条集的结构叙事。现在更稳的说法是：post_v2 仍可学，但其主导 block 对 sequence mix 非常敏感。",
            "当前 full、DirRisk、geometry-only 之间的差距都不大，说明这个任务在现阶段更像一个脆弱的 structure-sensitive benchmark，而不是已经稳定的主模型舞台。",
            "虽然 7 条比 5 条更厚，但 test 仍然只由单条 `city_night/1_low` 承担，所以这轮最多能支持“结构再漂移”判断，不能支持稳定泛化结论。",
        ],
        "claim_safe": [
            "可以说从 5 条到 7 条后，post_v2 的结构结论继续发生漂移，旧的 geometry-dominated 与 drop-candidate 最优都不再稳固。",
            "可以说 candidate-only 的领先有具体证据支持：candidate 特征在 test 上的单特征判别力最高，同时 geometry block 漂移更大。",
            "可以说当前最需要优先解释的是 sequence mix 与 feature drift，而不是急着升级模型复杂度。",
        ],
        "claim_forbidden": [
            "不能把这轮 7 条结果写成 candidate 已经被证明是 post_v2 的稳定主导信号。",
            "不能把当前 test 结果写成跨场景或跨 dynamic level 的稳健泛化结论。",
            "不能把 DirRisk 与 logistic 持平写成方向先验已经被验证有效。",
        ],
        "next_steps": [
            "先补一轮面向 7 条集的 sequence-specific summary，把 `city_night/1_low` 作为 test 的特殊性单独钉住。",
            "下一轮扩样本优先补新的 val/test sequence，而不是继续往 train 里堆风格相近的序列。",
            "在没解释清 sequence mix 之前，不继续推进 MonoRisk 或更复杂 post 模型。",
        ],
    }

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "post_v2_structure_redrift_summary.json", summary)
    write_csv(
        out_dir / "post_v2_structure_redrift_main_comparison.csv",
        [
            "name", "old_num_rows", "new_num_rows",
            "old_positive_ratio", "new_positive_ratio",
            "old_auroc", "new_auroc", "delta_auroc",
            "old_auprc", "new_auprc", "old_brier", "new_brier", "old_ece", "new_ece",
        ],
        main_rows,
    )
    write_csv(
        out_dir / "post_v2_structure_redrift_ablation_shift.csv",
        ["name", "old_auroc", "new_auroc", "delta_auroc", "old_auprc", "new_auprc", "note"],
        ablation_rows,
    )
    write_text(out_dir / "post_v2_structure_redrift_summary.md", build_markdown(summary, main_rows, ablation_rows))
    print(f"[post_v2_structure_redrift] saved -> {out_dir}")


if __name__ == "__main__":
    main()
