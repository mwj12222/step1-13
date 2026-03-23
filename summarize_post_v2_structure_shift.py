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


def f4(v):
    if v is None or v == "":
        return "-"
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)


def by_name(rows: list[dict]) -> dict[str, dict]:
    return {str(r.get("name", "")): r for r in rows}


def build_main_comparison(old_rows: list[dict], new_logistic: dict, new_dir: dict) -> list[dict]:
    old_map = by_name(old_rows)
    out = []
    mapping = [
        ("post_v2_rule_q_post_geom_only", "post_v2_rule_q_post_geom_only", "rule"),
        ("post_v2_logistic_full", "post_v2_logistic_full", "logistic"),
        ("post_v2_dir_risk", "post_v2_dir_risk", "dir_risk"),
    ]
    new_map = {
        "post_v2_rule_q_post_geom_only": {
            "auroc": new_logistic["rule_baseline"]["test"]["auroc"],
            "auprc": new_logistic["rule_baseline"]["test"]["auprc"],
            "brier": new_logistic["rule_baseline"]["test"]["brier"],
            "ece": new_logistic["rule_baseline"]["test"]["ece"],
            "num_rows": new_logistic["rule_baseline"]["test"]["num_rows"],
            "positive_ratio": new_logistic["rule_baseline"]["test"]["positive_ratio"],
        },
        "post_v2_logistic_full": {
            "auroc": new_logistic["splits"]["test"]["auroc"],
            "auprc": new_logistic["splits"]["test"]["auprc"],
            "brier": new_logistic["splits"]["test"]["brier"],
            "ece": new_logistic["splits"]["test"]["ece"],
            "num_rows": new_logistic["splits"]["test"]["num_rows"],
            "positive_ratio": new_logistic["splits"]["test"]["positive_ratio"],
        },
        "post_v2_dir_risk": {
            "auroc": new_dir["splits"]["test"]["auroc"],
            "auprc": new_dir["splits"]["test"]["auprc"],
            "brier": new_dir["splits"]["test"]["brier"],
            "ece": new_dir["splits"]["test"]["ece"],
            "num_rows": new_dir["splits"]["test"]["num_rows"],
            "positive_ratio": new_dir["splits"]["test"]["positive_ratio"],
        },
    }
    for old_name, new_name, source in mapping:
        old = old_map.get(old_name, {})
        new = new_map[new_name]
        out.append(
            {
                "name": new_name,
                "source": source,
                "old_num_rows": old.get("num_rows", ""),
                "new_num_rows": new["num_rows"],
                "old_positive_ratio": old.get("positive_ratio", ""),
                "new_positive_ratio": new["positive_ratio"],
                "old_auroc": old.get("auroc", ""),
                "new_auroc": new["auroc"],
                "delta_auroc": (
                    float(new["auroc"]) - float(old["auroc"])
                    if old.get("auroc", "") not in ("", None) else None
                ),
                "old_auprc": old.get("auprc", ""),
                "new_auprc": new["auprc"],
                "old_brier": old.get("brier", ""),
                "new_brier": new["brier"],
                "old_ece": old.get("ece", ""),
                "new_ece": new["ece"],
            }
        )
    return out


def build_markdown(summary: dict, comp_rows: list[dict], new_ablation_rows: list[dict]) -> str:
    lines = []
    lines.append("# post_v2 扩样本后结构结论变化")
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    for item in summary["headline_judgement"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 新旧主结果对比")
    lines.append("")
    lines.append("| 模型 | 旧样本 test AUROC | 新样本 test AUROC | 变化 |")
    lines.append("| --- | --- | --- | --- |")
    for row in comp_rows:
        lines.append(
            f"| {row['name']} | {f4(row['old_auroc'])} | {f4(row['new_auroc'])} | {f4(row['delta_auroc'])} |"
        )
    lines.append("")
    lines.append("## 旧结论与新证据")
    lines.append("")
    lines.append("**旧阶段正式结论**")
    for item in summary["legacy_conclusions"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("**新阶段观察**")
    for item in summary["new_observations"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 新 5 条集的 block 结果")
    lines.append("")
    lines.append("| Ablation | AUROC | AUPRC | 解释 |")
    lines.append("| --- | --- | --- | --- |")
    for row in new_ablation_rows:
        lines.append(
            f"| {row['name']} | {f4(row['auroc'])} | {f4(row['auprc'])} | {row['note']} |"
        )
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
    ap = argparse.ArgumentParser(description="Summarize post_v2 structure shift after sample expansion.")
    ap.add_argument("--old_summary_json", required=True)
    ap.add_argument("--old_main_csv", required=True)
    ap.add_argument("--new_root", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    old_summary = load_json(Path(args.old_summary_json).expanduser().resolve())
    old_main_rows = load_csv_rows(Path(args.old_main_csv).expanduser().resolve())

    new_root = Path(args.new_root).expanduser().resolve()
    new_logistic = load_json(new_root / "model_post_v2_min_default_logistic_5seq" / "metrics.json")
    new_dir = load_json(new_root / "model_post_v2_min_default_dir_risk_5seq" / "metrics.json")
    new_dir_feature = load_json(new_root / "model_post_v2_min_default_dir_risk_5seq" / "feature_direction_map.json")
    new_ablation = load_json(new_root / "post_v2_ablations_5seq" / "post_v2_ablation_summary.json")
    new_dataset_manifest = load_json(new_root / "dataset_manifest.json")
    new_task_manifest = load_json(new_root / "risk_dataset_post_v2_min_default_manifest.json")
    new_dataset_audit = load_json(new_root / "dataset_audit.json")

    comp_rows = build_main_comparison(old_main_rows, new_logistic, new_dir)
    ablation_rows = new_ablation["rows"]

    summary = {
        "legacy_dataset": {
            "label_scope": old_summary["dataset"]["label_scope"],
            "coverage_summary": old_summary["dataset"]["coverage_summary"],
            "old_test_rows": old_summary["results"]["logistic_test"]["num_rows"],
        },
        "new_dataset": {
            "label_scope": new_task_manifest["label_scope"],
            "coverage_summary": new_task_manifest["coverage_summary"],
            "new_test_rows": new_logistic["splits"]["test"]["num_rows"],
            "dynamic_level_split_coupling": new_dataset_audit.get("dynamic_level_split_coupling", {}),
        },
        "headline_judgement": [
            "扩样本后，post_v2_min_default 仍然可学，但 test AUROC 从 0.8750 降到 0.6294，说明旧小样本结论存在明显脆弱性。",
            "规则式 Q_post_geom_only 在新 5 条集上仍然不够，test AUROC 只有 0.5000；学习式 post baseline 仍然优于规则式基线。",
            "旧的“geometry-dominated + candidate 小幅纠错”叙事在更厚样本上不再稳固，必须收紧成“结构待重审”。",
        ],
        "legacy_conclusions": [
            "旧 4 条集正式小结把 post_v2 定位为更接近 accept 后短时稳定性的学习式 post 任务。",
            "旧阶段主结果为：rule 0.5312，logistic 0.8750，DirRisk 0.8750。",
            "旧阶段工作假设更接近：posterior geometry 主导，candidate 提供少量纠错。",
        ],
        "new_observations": [
            "新 5 条集主结果为：rule 0.5000，logistic 0.6294，DirRisk 0.6294。",
            "在新 5 条集上，geometry_only 只有 0.5102，front_only 却达到 0.6151，drop_geometry 也有 0.5912。",
            "drop_candidate 反而达到 0.6410，是当前 5 条集上表现最好的 ablation。",
            f"DirRisk 在新集上仍无额外增益，且仍有 {new_dir_feature['num_negative_aligned_weights']} 个方向违规项。",
        ],
        "critical_takeaways": [
            "扩样本确实把 post_v2 的证据链变厚了：test 从 8 条扩大到 128 条，v2 标签人口扩大到 393 条。",
            "扩样本后，旧的结构性结论被部分推翻，这不是坏事，恰恰说明此前的几何主导叙事受小样本影响较大。",
            "现在最稳的结论不是 geometry-dominated，而是：post_v2 在更厚数据上仍有可学习信号，但 feature-block 结构需要重新审查。",
            "当前 split 仍不理想：val 仍然很薄，dynamic-level coupling 只被部分缓解，不能把这轮结果写成最终泛化结论。",
        ],
        "claim_safe": [
            "可以说扩样本后学习式 post_v2 baseline 仍然优于规则式几何分数。",
            "可以说扩样本后旧结构结论发生漂移，说明需要重新审查 post_v2 的主导信号来源。",
            "可以说新的 5 条集为后续标签/特征/结构分析提供了更厚的证据基础。",
        ],
        "claim_forbidden": [
            "不能继续直接写 post_v2 明确是 geometry-dominated。",
            "不能把这轮 5 条结果写成跨场景稳健泛化结论。",
            "不能把 DirRisk 在新 5 条集上的无退化写成单调约束建模已成立。",
        ],
        "next_steps": [
            "先做一轮 feature correlation / split drift / sequence contribution 分析，解释 geometry-only 为什么显著掉队。",
            "在没解释清结构漂移前，不继续推进 MonoRisk 或更复杂 post 模型。",
            "下一轮样本扩容优先补 split 空缺，而不是平均地继续堆更多 2_mid。",
        ],
    }

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "post_v2_structure_shift_summary.json", summary)
    write_csv(
        out_dir / "post_v2_structure_shift_main_comparison.csv",
        [
            "name", "source", "old_num_rows", "new_num_rows",
            "old_positive_ratio", "new_positive_ratio",
            "old_auroc", "new_auroc", "delta_auroc",
            "old_auprc", "new_auprc", "old_brier", "new_brier", "old_ece", "new_ece",
        ],
        comp_rows,
    )
    write_csv(
        out_dir / "post_v2_structure_shift_new_ablation.csv",
        ["name", "num_features", "features", "auroc", "auprc", "brier", "ece", "positive_ratio", "num_rows", "note"],
        ablation_rows,
    )
    write_text(out_dir / "post_v2_structure_shift_summary.md", build_markdown(summary, comp_rows, ablation_rows))
    print(f"[post_v2_structure_shift] saved -> {out_dir}")


if __name__ == "__main__":
    main()
