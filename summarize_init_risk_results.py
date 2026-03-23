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


def fmt(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def row_from_metrics(name: str, task: str, source: str, metrics: dict, features: str, note: str = ""):
    return {
        "name": name,
        "task": task,
        "source": source,
        "num_rows": metrics.get("num_rows"),
        "positive_ratio": metrics.get("positive_ratio"),
        "auroc": metrics.get("auroc"),
        "auprc": metrics.get("auprc"),
        "brier": metrics.get("brier"),
        "ece": metrics.get("ece"),
        "features": features,
        "note": note,
    }


def load_dir_risk_report(root: Path, task: str):
    model_dir = root / f"model_{task}_dir_risk"
    metrics = load_json(model_dir / "metrics.json")
    direction = load_json(model_dir / "feature_direction_map.json")
    with open(model_dir / "block_weight_summary.csv", "r", encoding="utf-8") as f:
        block_rows = list(csv.DictReader(f))
    return {
        "metrics": metrics,
        "direction": direction,
        "block_rows": block_rows,
    }


def build_main_rows(root: Path):
    pre = load_json(root / "model_pre_logistic" / "metrics.json")
    post = load_json(root / "model_post_logistic" / "metrics.json")
    pre_platt = load_json(root / "model_pre_platt" / "calibration_metrics.json")
    post_platt = load_json(root / "model_post_platt" / "calibration_metrics.json")
    pre_dir = load_dir_risk_report(root, "pre")
    post_dir = load_dir_risk_report(root, "post")

    rows = [
        row_from_metrics(
            "pre_rule_q_pre",
            "pre",
            "rule",
            pre["rule_baseline"]["test"],
            pre["rule_baseline"]["rule_col"],
            "规则式 baseline，按 p_bad=1-Q_pre 对齐。",
        ),
        row_from_metrics(
            "pre_logistic_full",
            "pre",
            "logistic",
            pre["splits"]["test"],
            ",".join(pre["allowed_feature_columns"]),
            "当前最小可解释 pre baseline。",
        ),
        row_from_metrics(
            "pre_dir_risk",
            "pre",
            "dir_risk",
            pre_dir["metrics"]["splits"]["test"],
            ",".join(pre_dir["metrics"]["allowed_feature_columns"]),
            f"方向对齐版 pre baseline；test 不退化，但仍有 {pre_dir['direction']['num_negative_aligned_weights']} 个方向违规项。",
        ),
        row_from_metrics(
            "pre_logistic_platt",
            "pre",
            "platt",
            pre_platt["platt_calibrated"]["test"],
            ",".join(pre["allowed_feature_columns"]),
            "test 上校准变差，仅留档，不作为当前主结果。",
        ),
        row_from_metrics(
            "post_rule_q_post_geom_only",
            "post",
            "rule",
            post["rule_baseline"]["test"],
            post["rule_baseline"]["rule_col"],
            "规则式后验几何 baseline，按 p_bad=1-Q_post_geom_only 对齐。",
        ),
        row_from_metrics(
            "post_logistic_full",
            "post",
            "logistic",
            post["splits"]["test"],
            ",".join(post["allowed_feature_columns"]),
            "当前最小可解释 post baseline。",
        ),
        row_from_metrics(
            "post_dir_risk",
            "post",
            "dir_risk",
            post_dir["metrics"]["splits"]["test"],
            ",".join(post_dir["metrics"]["allowed_feature_columns"]),
            f"方向对齐版 post baseline；test 不退化，但仍有 {post_dir['direction']['num_negative_aligned_weights']} 个方向违规项。",
        ),
        row_from_metrics(
            "post_logistic_platt",
            "post",
            "platt",
            post_platt["platt_calibrated"]["test"],
            ",".join(post["allowed_feature_columns"]),
            "仅作 sanity check，不作为当前主推进点。",
        ),
    ]
    return rows


def build_ablation_rows(root: Path):
    ablation = load_json(root / "ablation_summary.json")["rows"]
    pre_candidate = load_json(root / "pre_candidate_ablation_summary.json")["rows"]
    seen = set()
    rows = []
    for row in ablation + pre_candidate:
        name = row["name"]
        if name in seen:
            continue
        seen.add(name)
        rows.append(row)
    return rows


def build_summary_json(root: Path, main_rows: list[dict], ablation_rows: list[dict]):
    audit = load_json(root / "dataset_audit.json")
    split = load_json(root / "dataset_split_manifest.json")
    dataset_manifest = load_json(root / "dataset_manifest.json")
    pre_dir = load_dir_risk_report(root, "pre")
    post_dir = load_dir_risk_report(root, "post")
    return {
        "dataset": {
            "num_rows": audit.get("num_rows"),
            "sample_types": audit.get("sample_types"),
            "schema_version_counts": audit.get("schema_version_counts"),
            "Y_bad_v1_ratio": audit.get("Y_bad_v1_ratio"),
            "label_version": "Y_bad_v1",
            "label_scope": "init_level_bad_event_proxy",
            "unknown_protocol_ratio": audit.get("unknown_protocol_ratio"),
            "dynamic_level_split_coupling": audit.get("dynamic_level_split_coupling", {}),
        },
        "claim_guardrails": dataset_manifest.get("claim_guardrails", {}),
        "split": {
            "split_mode": split.get("split_mode"),
            "sequence_to_split": split.get("sequence_to_split"),
            "split_stats": split.get("split_stats"),
        },
        "dir_risk": {
            "pre": {
                "num_negative_aligned_weights": pre_dir["direction"]["num_negative_aligned_weights"],
                "negative_aligned_weight_features": pre_dir["direction"]["negative_aligned_weight_features"],
                "block_weight_summary": pre_dir["block_rows"],
            },
            "post": {
                "num_negative_aligned_weights": post_dir["direction"]["num_negative_aligned_weights"],
                "negative_aligned_weight_features": post_dir["direction"]["negative_aligned_weight_features"],
                "block_weight_summary": post_dir["block_rows"],
            },
        },
        "main_results": main_rows,
        "ablations": ablation_rows,
        "takeaways": {
            "pre": "pre 任务当前以 candidate-driven 为主，front-only 明显失效，candidate-only 基本复现 full pre。",
            "post": "post 任务当前以 geometry-dominated 为主，geom-only 与 full post 基本等价。",
            "calibration": "pre 的 Platt 在 test 上变差，不采用；post 的 Platt 只保留为 sanity check。",
            "dir_risk": "DirRisk 说明方向先验可显式编码而不损失当前 test 性能，但仍出现方向违规项，因此当前只能称为 direction-aligned risk baseline，不能称为 true monotonic risk model。",
            "caution": "当前仅 4 个序列、88 条样本，strict-by-sequence split 与动态等级仍耦合，不能写成强泛化结论。",
        },
        "limitations": [
            "当前仅 4 个序列、88 条样本，single split 结果天然偏脆弱，动态等级与 split 存在耦合。",
            "Y_bad_v1 只是 initialization-level bad event proxy，不是系统级真实失败标签。",
            "pre 的事实主导项是 candidate 可观测性，而不是前端静态支撑直接主导，因此论文叙事必须收紧成 front-as-upstream-constraint。",
            "post 当前更像 geometry-dominated acceptor baseline，学术增量主要在整理和解释，不宜包装成当前阶段的主创新承载点。",
            "DirRisk 仍存在方向违规项，因此还不能称为 true monotonic constrained risk model。",
            "校准结果目前只能算探路，不能作为稳健概率输出的证据。",
            "尚未形成系统级门控收益闭环，当前完成的是风险建模链条成立，而不是系统收益已证实。",
        ],
        "next_priorities": [
            "补更强标签，从 Y_bad_v1 逐步过渡到更接近系统真实失败的坏事件标签。",
            "补更强协议，增加序列数量，削弱动态等级与 split 的耦合。",
            "补系统级收益，验证 accept/delay/reset 门控对真实初始化验收和后端鲁棒性的影响。",
        ],
        "claim_safe_conclusions": [
            "在冻结协议和任务级合法特征约束下，当前数据支持 pre/post 初始化风险建模链条的可行性验证。",
            "当前 pre 任务证据更支持 candidate-stage observability 作为直接主判别层，而 front 更适合作为上游约束和上下文。",
            "当前 post 任务更适合作为 geometry-dominated acceptor baseline，而不是整条工作的唯一主创新承载点。",
            "DirRisk 说明方向先验可以被显式编码而不损失当前 test 性能，但尚不足以支持 true monotonic constrained model 的主张。",
        ],
    }


def build_markdown(summary: dict):
    dataset = summary["dataset"]
    split = summary["split"]
    claim_guardrails = summary.get("claim_guardrails", {})
    dir_risk = summary["dir_risk"]
    main_rows = summary["main_results"]
    ablations = summary["ablations"]
    takeaways = summary["takeaways"]

    lines = []
    lines.append("# 初始化风险 baseline 实验小结")
    lines.append("")
    lines.append("## 数据与协议")
    lines.append("")
    lines.append(f"- 样本总数：`{dataset['num_rows']}`")
    lines.append(f"- 样本构成：`{dataset['sample_types']}`")
    lines.append(f"- schema：`{dataset['schema_version_counts']}`")
    lines.append(f"- `Y_bad_v1` 比例：`{fmt(dataset['Y_bad_v1_ratio'])}`")
    lines.append(f"- label：`{dataset['label_version']}` / `{dataset['label_scope']}`")
    lines.append(f"- unknown protocol 比例：`{fmt(dataset['unknown_protocol_ratio'])}`")
    lines.append(f"- split 模式：`{split['split_mode']}`")
    lines.append(f"- sequence -> split：`{split['sequence_to_split']}`")
    lines.append(f"- 动态等级与 split 耦合：`{dataset['dynamic_level_split_coupling']}`")
    lines.append("")
    lines.append("## Claim Guardrails")
    lines.append("")
    for item in claim_guardrails.get("allowed_claims", []):
        lines.append(f"- Allowed: {item}")
    for item in claim_guardrails.get("forbidden_claims", []):
        lines.append(f"- Forbidden: {item}")
    for item in claim_guardrails.get("proxy_bias_notes", []):
        lines.append(f"- Proxy bias: {item}")
    if claim_guardrails.get("experimental_limit_flags"):
        lines.append(f"- Limit flags: `{claim_guardrails['experimental_limit_flags']}`")
    lines.append("")
    lines.append("## 主结果")
    lines.append("")
    lines.append("| name | task | source | AUROC | AUPRC | Brier | ECE | note |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for row in main_rows:
        lines.append(
            f"| {row['name']} | {row['task']} | {row['source']} | {fmt(row['auroc'])} | "
            f"{fmt(row['auprc'])} | {fmt(row['brier'])} | {fmt(row['ece'])} | {row['note']} |"
        )
    lines.append("")
    lines.append("## DirRisk 说明")
    lines.append("")
    lines.append(
        f"- `pre DirRisk`：方向违规项 `"
        f"{dir_risk['pre']['num_negative_aligned_weights']}` 个，违规特征 "
        f"`{dir_risk['pre']['negative_aligned_weight_features']}`。"
    )
    lines.append(
        f"- `pre DirRisk` block 占比：`{dir_risk['pre']['block_weight_summary']}`。"
    )
    lines.append(
        f"- `post DirRisk`：方向违规项 `"
        f"{dir_risk['post']['num_negative_aligned_weights']}` 个，违规特征 "
        f"`{dir_risk['post']['negative_aligned_weight_features']}`。"
    )
    lines.append(
        f"- `post DirRisk` block 占比：`{dir_risk['post']['block_weight_summary']}`。"
    )
    lines.append("- 结论：DirRisk 可以作为比规则式 Q 更自然的学习式主线，但当前仍不应被表述为真正的单调约束风险模型。")
    lines.append("")
    lines.append("## 消融结论")
    lines.append("")
    lines.append("| name | AUROC | AUPRC | Brier | ECE |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in ablations:
        lines.append(
            f"| {row['name']} | {fmt(row.get('auroc'))} | {fmt(row.get('auprc'))} | "
            f"{fmt(row.get('brier'))} | {fmt(row.get('ece'))} |"
        )
    lines.append("")
    lines.append("## 阶段性判断")
    lines.append("")
    lines.append(f"- `pre`：{takeaways['pre']}")
    lines.append(f"- `post`：{takeaways['post']}")
    lines.append(f"- 校准：{takeaways['calibration']}")
    lines.append(f"- `DirRisk`：{takeaways['dir_risk']}")
    lines.append(f"- 风险提示：{takeaways['caution']}")
    lines.append("")
    lines.append("## 当前不足")
    lines.append("")
    for item in summary.get("limitations", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 下一步优先级")
    lines.append("")
    for item in summary.get("next_priorities", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Claim-Safe Conclusions")
    lines.append("")
    for item in summary.get("claim_safe_conclusions", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Summarize current init-risk baseline experiments into stable markdown/csv tables.")
    ap.add_argument("--result_root", required=True, help="Root directory containing audit dataset and model outputs.")
    ap.add_argument("--out_dir", required=True, help="Stable output directory for summary artifacts.")
    args = ap.parse_args()

    result_root = Path(args.result_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    main_rows = build_main_rows(result_root)
    ablation_rows = build_ablation_rows(result_root)
    summary = build_summary_json(result_root, main_rows, ablation_rows)
    markdown = build_markdown(summary)

    write_json(out_dir / "baseline_experiment_summary.json", summary)
    write_csv(out_dir / "baseline_main_results.csv", list(main_rows[0].keys()), main_rows)
    write_csv(out_dir / "baseline_ablation_results.csv", list(ablation_rows[0].keys()), ablation_rows)
    write_text(out_dir / "baseline_experiment_summary.md", markdown)

    print(f"[InitRiskSummary] saved -> {out_dir}")


if __name__ == "__main__":
    main()
