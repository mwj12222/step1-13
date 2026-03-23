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


def fmt(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def build_sequence_summary(rows: list[dict]) -> list[dict]:
    by_seq = {}
    for row in rows:
        if row.get("sample_type") != "step11":
            continue
        seq = str(row.get("sequence", ""))
        by_seq.setdefault(seq, []).append(row)
    out = []
    for seq, seq_rows in sorted(by_seq.items()):
        labeled = [r for r in seq_rows if str(r.get("Y_bad_v2_min_default", "")) in ("0", "1")]
        trigger_counts = {}
        for row in labeled:
            trig = str(row.get("Y_bad_v2_min_default_trigger", ""))
            trigger_counts[trig] = trigger_counts.get(trig, 0) + 1
        bad_rows = sum(int(row["Y_bad_v2_min_default"]) for row in labeled) if labeled else 0
        out.append(
            {
                "sequence": seq,
                "num_rows": len(seq_rows),
                "accept_rows": sum(1 for r in seq_rows if str(r.get("gate_post", "")) == "accept"),
                "labeled_rows": len(labeled),
                "bad_rows": int(bad_rows),
                "bad_ratio_labeled": float(bad_rows / max(1, len(labeled))) if labeled else None,
                "trigger_composition": json.dumps(trigger_counts, ensure_ascii=False),
            }
        )
    return out


def build_crosstab(rows: list[dict]) -> dict:
    table = {}
    for row in rows:
        if row.get("sample_type") != "step11":
            continue
        v2 = str(row.get("Y_bad_v2_min_default", ""))
        if v2 not in ("0", "1"):
            continue
        v1 = int(float(row.get("Y_bad_v1", 0)))
        key = f"v1_{v1}__v2_{v2}"
        table[key] = table.get(key, 0) + 1
    return table


def build_main_rows(metrics: dict) -> list[dict]:
    return [
        {
            "name": "post_v2_rule_q_post_geom_only",
            "task": "post_v2_min_default",
            "source": "rule",
            "num_rows": metrics["rule_baseline"]["test"]["num_rows"],
            "positive_ratio": metrics["rule_baseline"]["test"]["positive_ratio"],
            "auroc": metrics["rule_baseline"]["test"]["auroc"],
            "auprc": metrics["rule_baseline"]["test"]["auprc"],
            "brier": metrics["rule_baseline"]["test"]["brier"],
            "ece": metrics["rule_baseline"]["test"]["ece"],
            "features": metrics["rule_baseline"]["rule_col"],
            "note": "规则式后验几何分数在 Y_bad_v2_min_default 上的对照基线。",
        },
        {
            "name": "post_v2_logistic_full",
            "task": "post_v2_min_default",
            "source": "logistic",
            "num_rows": metrics["splits"]["test"]["num_rows"],
            "positive_ratio": metrics["splits"]["test"]["positive_ratio"],
            "auroc": metrics["splits"]["test"]["auroc"],
            "auprc": metrics["splits"]["test"]["auprc"],
            "brier": metrics["splits"]["test"]["brier"],
            "ece": metrics["splits"]["test"]["ece"],
            "features": ",".join(metrics["allowed_feature_columns"]),
            "note": "当前 Y_bad_v2_min_default 的最小 post learning baseline。",
        },
    ]


def load_dir_risk_report(root: Path) -> dict:
    model_dir = root / "model_post_v2_min_default_dir_risk"
    metrics = load_json(model_dir / "metrics.json")
    direction = load_json(model_dir / "feature_direction_map.json")
    with open(model_dir / "block_weight_summary.csv", "r", encoding="utf-8") as f:
        block_rows = list(csv.DictReader(f))
    return {
        "metrics": metrics,
        "direction": direction,
        "block_rows": block_rows,
    }


def build_summary_json(dataset_manifest: dict, task_manifest: dict, metrics: dict, dir_risk: dict, audit: dict, sequence_rows: list[dict], crosstab: dict):
    return {
        "dataset": {
            "label_version": task_manifest.get("label_version"),
            "label_scope": task_manifest.get("label_scope"),
            "label_population": task_manifest.get("label_population"),
            "coverage_summary": task_manifest.get("coverage_summary", {}),
            "dynamic_level_split_coupling": audit.get("dynamic_level_split_coupling", {}),
            "available_label_versions": dataset_manifest.get("available_label_versions", []),
        },
        "task": {
            "task_name": task_manifest.get("task_name"),
            "task_scope": task_manifest.get("task_scope"),
            "allowed_feature_columns": task_manifest.get("allowed_feature_columns", []),
            "allowed_claims": task_manifest.get("allowed_claims", []),
            "forbidden_claims": task_manifest.get("forbidden_claims", []),
            "proxy_bias_notes": task_manifest.get("proxy_bias_notes", []),
            "experimental_limit_flags": task_manifest.get("experimental_limit_flags", []),
        },
        "results": {
            "logistic_test": metrics["splits"]["test"],
            "rule_test": metrics["rule_baseline"]["test"],
            "dir_risk_test": dir_risk["metrics"]["splits"]["test"],
        },
        "dir_risk": {
            "num_negative_aligned_weights": dir_risk["direction"]["num_negative_aligned_weights"],
            "negative_aligned_weight_features": dir_risk["direction"]["negative_aligned_weight_features"],
            "block_weight_summary": dir_risk["block_rows"],
        },
        "sequence_summary": sequence_rows,
        "v1_vs_v2_crosstab": crosstab,
        "takeaways": [
            "Y_bad_v2_min_default currently applies only to post-accept windows with sufficient future horizon and should not be extended to pre-task training.",
            "Compared with Y_bad_v1, Y_bad_v2_min_default adds short-horizon instability semantics after acceptance.",
            "On this label, logistic remains learnable while the old Q_post_geom_only rule baseline weakens substantially.",
            "DirRisk-post on Y_bad_v2_min_default does not degrade test performance, but still contains aligned-direction violations and therefore remains a direction-aligned baseline rather than a true monotonic model.",
        ],
        "limitations": [
            "Y_bad_v2_min_default is still a proxy label rather than a full system-level reset/reinit/tracking-failure truth.",
            "The label only covers accepted windows with sufficient future horizon, which further shrinks the effective sample size.",
            "Current results still inherit the 4-sequence small-sample split and dynamic-level coupling limitations.",
            "DirRisk-post still shows direction violations, so current evidence does not justify monotonic constrained model claims on Y_bad_v2_min_default.",
        ],
        "claim_safe_conclusions": [
            "Current evidence supports Y_bad_v2_min_default as a more system-like post-accept short-horizon instability proxy than Y_bad_v1.",
            "Current evidence does not support system-level failure prediction or full downstream gate-benefit claims.",
            "Current evidence does not justify extending Y_bad_v2_min_default to pre-task modeling.",
            "Current evidence supports at most a direction-aligned post-risk baseline on Y_bad_v2_min_default, not a true monotonic constrained post model.",
        ],
    }


def build_markdown(summary: dict, main_rows: list[dict]) -> str:
    dataset = summary["dataset"]
    task = summary["task"]
    dir_risk = summary.get("dir_risk", {})
    lines = []
    lines.append("# post_v2_min_default 实验小结")
    lines.append("")
    lines.append("## 标签与覆盖")
    lines.append("")
    lines.append(f"- label：`{dataset['label_version']}` / `{dataset['label_scope']}`")
    lines.append(f"- label population：`{dataset['label_population']}`")
    lines.append(f"- coverage summary：`{dataset['coverage_summary']}`")
    lines.append(f"- available label versions：`{dataset['available_label_versions']}`")
    lines.append(f"- 动态等级与 split 耦合：`{dataset['dynamic_level_split_coupling']}`")
    lines.append("")
    lines.append("## 任务边界")
    lines.append("")
    lines.append(f"- task：`{task['task_name']}` / `{task['task_scope']}`")
    lines.append(f"- allowed features：`{task['allowed_feature_columns']}`")
    for item in task["allowed_claims"]:
        lines.append(f"- Allowed: {item}")
    for item in task["forbidden_claims"]:
        lines.append(f"- Forbidden: {item}")
    for item in task["proxy_bias_notes"]:
        lines.append(f"- Proxy bias: {item}")
    for item in task["experimental_limit_flags"]:
        lines.append(f"- Limit flag: `{item}`")
    lines.append("")
    lines.append("## 主结果")
    lines.append("")
    for row in main_rows:
        lines.append(
            f"- `{row['name']}`: AUROC=`{fmt(row['auroc'])}`, AUPRC=`{fmt(row['auprc'])}`, "
            f"Brier=`{fmt(row['brier'])}`, ECE=`{fmt(row['ece'])}`. {row['note']}"
        )
    if dir_risk:
        lines.append("")
        lines.append("## DirRisk 对照")
        lines.append("")
        lines.append(
            f"- 方向违规项数：`{dir_risk.get('num_negative_aligned_weights')}`；违规特征：`{dir_risk.get('negative_aligned_weight_features')}`"
        )
        lines.append(f"- block 权重摘要：`{dir_risk.get('block_weight_summary')}`")
    lines.append("")
    lines.append("## 按序列分布")
    lines.append("")
    for row in summary["sequence_summary"]:
        lines.append(
            f"- `{row['sequence']}`: labeled_rows=`{row['labeled_rows']}`, "
            f"bad_ratio=`{fmt(row['bad_ratio_labeled'])}`, triggers=`{row['trigger_composition']}`"
        )
    lines.append("")
    lines.append("## v1 × v2 交叉表")
    lines.append("")
    lines.append(f"- `{summary['v1_vs_v2_crosstab']}`")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    for item in summary["claim_safe_conclusions"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 当前不足")
    lines.append("")
    for item in summary["limitations"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Summarize post_v2_min_default formal results.")
    ap.add_argument("--result_root", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    root = Path(args.result_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    dataset_manifest = load_json(root / "dataset_manifest.json")
    dataset_audit = load_json(root / "dataset_audit.json")
    task_manifest = load_json(root / "risk_dataset_post_v2_min_default_manifest.json")
    metrics = load_json(root / "model_post_v2_min_default_logistic" / "metrics.json")
    dir_risk = load_dir_risk_report(root)
    risk_rows = load_csv_rows(root / "risk_dataset.csv")

    sequence_rows = build_sequence_summary(risk_rows)
    crosstab = build_crosstab(risk_rows)
    main_rows = build_main_rows(metrics)
    main_rows.insert(
        2,
        {
            "name": "post_v2_dir_risk",
            "task": "post_v2_min_default",
            "source": "dir_risk",
            "num_rows": dir_risk["metrics"]["splits"]["test"]["num_rows"],
            "positive_ratio": dir_risk["metrics"]["splits"]["test"]["positive_ratio"],
            "auroc": dir_risk["metrics"]["splits"]["test"]["auroc"],
            "auprc": dir_risk["metrics"]["splits"]["test"]["auprc"],
            "brier": dir_risk["metrics"]["splits"]["test"]["brier"],
            "ece": dir_risk["metrics"]["splits"]["test"]["ece"],
            "features": ",".join(dir_risk["metrics"]["allowed_feature_columns"]),
            "note": f"方向对齐版 post_v2 baseline；test 不退化，但仍有 {dir_risk['direction']['num_negative_aligned_weights']} 个方向违规项。",
        },
    )
    summary = build_summary_json(dataset_manifest, task_manifest, metrics, dir_risk, dataset_audit, sequence_rows, crosstab)

    write_json(out_dir / "post_v2_min_default_summary.json", summary)
    write_csv(
        out_dir / "post_v2_min_default_main_results.csv",
        ["name", "task", "source", "num_rows", "positive_ratio", "auroc", "auprc", "brier", "ece", "features", "note"],
        main_rows,
    )
    write_csv(
        out_dir / "post_v2_min_default_sequence_summary.csv",
        ["sequence", "num_rows", "accept_rows", "labeled_rows", "bad_rows", "bad_ratio_labeled", "trigger_composition"],
        sequence_rows,
    )
    write_json(out_dir / "post_v2_min_default_crosstab.json", crosstab)
    write_text(out_dir / "post_v2_min_default_summary.md", build_markdown(summary, main_rows))

    print(f"[V2Summary] saved -> {out_dir}")


if __name__ == "__main__":
    main()
