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


def build_main_rows(v1_summary: dict, v2_summary: dict) -> list[dict]:
    v1_post = next(row for row in v1_summary["main_results"] if row["name"] == "post_logistic_full")
    v1_rule = next(row for row in v1_summary["main_results"] if row["name"] == "post_rule_q_post_geom_only")
    v1_dir = next(row for row in v1_summary["main_results"] if row["name"] == "post_dir_risk")

    v2_main = load_json(v2_summary["source_files"]["main_results_json"])
    v2_log = next(row for row in v2_main if row["name"] == "post_v2_logistic_full")
    v2_rule = next(row for row in v2_main if row["name"] == "post_v2_rule_q_post_geom_only")
    v2_dir = next(row for row in v2_main if row["name"] == "post_v2_dir_risk")

    return [
        {
            "model": "rule_q_post_geom_only",
            "label_version": "Y_bad_v1",
            "label_scope": v1_summary["dataset"]["label_scope"],
            "num_rows": v1_rule["num_rows"],
            "positive_ratio": v1_rule["positive_ratio"],
            "auroc": v1_rule["auroc"],
            "auprc": v1_rule["auprc"],
            "brier": v1_rule["brier"],
            "ece": v1_rule["ece"],
            "note": "Y_bad_v1 下的后验几何规则式 baseline。",
        },
        {
            "model": "rule_q_post_geom_only",
            "label_version": "Y_bad_v2_min_default",
            "label_scope": v2_summary["dataset"]["label_scope"],
            "num_rows": v2_rule["num_rows"],
            "positive_ratio": v2_rule["positive_ratio"],
            "auroc": v2_rule["auroc"],
            "auprc": v2_rule["auprc"],
            "brier": v2_rule["brier"],
            "ece": v2_rule["ece"],
            "note": "Y_bad_v2_min_default 下的后验几何规则式 baseline。",
        },
        {
            "model": "post_logistic",
            "label_version": "Y_bad_v1",
            "label_scope": v1_summary["dataset"]["label_scope"],
            "num_rows": v1_post["num_rows"],
            "positive_ratio": v1_post["positive_ratio"],
            "auroc": v1_post["auroc"],
            "auprc": v1_post["auprc"],
            "brier": v1_post["brier"],
            "ece": v1_post["ece"],
            "note": "Y_bad_v1 下的最小 post learning baseline。",
        },
        {
            "model": "post_logistic",
            "label_version": "Y_bad_v2_min_default",
            "label_scope": v2_summary["dataset"]["label_scope"],
            "num_rows": v2_log["num_rows"],
            "positive_ratio": v2_log["positive_ratio"],
            "auroc": v2_log["auroc"],
            "auprc": v2_log["auprc"],
            "brier": v2_log["brier"],
            "ece": v2_log["ece"],
            "note": "Y_bad_v2_min_default 下的最小 post learning baseline。",
        },
        {
            "model": "post_dir_risk",
            "label_version": "Y_bad_v1",
            "label_scope": v1_summary["dataset"]["label_scope"],
            "num_rows": v1_dir["num_rows"],
            "positive_ratio": v1_dir["positive_ratio"],
            "auroc": v1_dir["auroc"],
            "auprc": v1_dir["auprc"],
            "brier": v1_dir["brier"],
            "ece": v1_dir["ece"],
            "note": "Y_bad_v1 下的方向对齐 post baseline。",
        },
        {
            "model": "post_dir_risk",
            "label_version": "Y_bad_v2_min_default",
            "label_scope": v2_summary["dataset"]["label_scope"],
            "num_rows": v2_dir["num_rows"],
            "positive_ratio": v2_dir["positive_ratio"],
            "auroc": v2_dir["auroc"],
            "auprc": v2_dir["auprc"],
            "brier": v2_dir["brier"],
            "ece": v2_dir["ece"],
            "note": "Y_bad_v2_min_default 下的方向对齐 post baseline。",
        },
    ]


def build_gain_summary(v1_summary: dict, v2_summary: dict) -> dict:
    v1_post = next(row for row in v1_summary["main_results"] if row["name"] == "post_logistic_full")
    v1_rule = next(row for row in v1_summary["main_results"] if row["name"] == "post_rule_q_post_geom_only")
    v1_dir = next(row for row in v1_summary["main_results"] if row["name"] == "post_dir_risk")
    v2_results = v2_summary["results"]
    v2_dir = v2_summary["dir_risk"]
    crosstab = v2_summary["v1_vs_v2_crosstab"]
    added_bad = int(crosstab.get("v1_0__v2_1", 0))
    kept_good = int(crosstab.get("v1_0__v2_0", 0))
    overlap_bad = int(crosstab.get("v1_1__v2_1", 0))
    return {
        "label_upgrade": {
            "from": {
                "label_version": v1_summary["dataset"]["label_version"],
                "label_scope": v1_summary["dataset"]["label_scope"],
                "positive_ratio": v1_summary["dataset"]["Y_bad_v1_ratio"],
            },
            "to": {
                "label_version": v2_summary["dataset"]["label_version"],
                "label_scope": v2_summary["dataset"]["label_scope"],
                "label_population": v2_summary["dataset"]["label_population"],
                "coverage_summary": v2_summary["dataset"]["coverage_summary"],
            },
        },
        "behavior_shift": {
            "v1_good_but_v2_bad": added_bad,
            "v1_good_and_v2_good": kept_good,
            "v1_bad_and_v2_bad": overlap_bad,
            "interpretation": "Y_bad_v2_min_default adds accepted short-horizon instability cases that Y_bad_v1 does not mark as bad.",
        },
        "metric_shift": {
            "rule_test_auroc_v1": v1_rule["auroc"],
            "rule_test_auroc_v2": v2_results["rule_test"]["auroc"],
            "logistic_test_auroc_v1": v1_post["auroc"],
            "logistic_test_auroc_v2": v2_results["logistic_test"]["auroc"],
            "dir_risk_test_auroc_v1": v1_dir["auroc"],
            "dir_risk_test_auroc_v2": v2_results["dir_risk_test"]["auroc"],
            "interpretation": "The old geometry rule degrades sharply on v2, while learning-based post models remain viable.",
        },
        "dir_risk": {
            "v1_negative_aligned_weights": v1_summary["dir_risk"]["post"]["num_negative_aligned_weights"],
            "v2_negative_aligned_weights": v2_dir["num_negative_aligned_weights"],
            "v2_negative_aligned_weight_features": v2_dir["negative_aligned_weight_features"],
            "interpretation": "DirRisk remains direction-aligned only; label upgrade does not remove the need for stronger monotonic constraints.",
        },
        "claim_safe_conclusions": [
            "Y_bad_v2_min_default makes the post task more system-like by focusing on accepted-window short-horizon instability rather than pure initialization-stage bad events.",
            "Under Y_bad_v2_min_default, the old Q_post_geom_only rule baseline weakens substantially, indicating that label upgrade exposes failures not captured by static posterior-geometry thresholds alone.",
            "Learning-based post models remain effective under Y_bad_v2_min_default, which supports continuing the main risk-model line on stronger labels.",
            "Current evidence still does not justify system-level failure prediction claims or extension of Y_bad_v2_min_default to pre-task modeling.",
        ],
        "limitations": [
            "The comparison is still based on a 4-sequence small-sample audit with strong dynamic-level split coupling.",
            "Y_bad_v2_min_default remains a proxy label restricted to accepted windows with sufficient future horizon.",
            "The upgraded label currently changes task population as well as semantics, so metric shifts should not be over-interpreted as pure label-only effects.",
        ],
    }


def build_markdown(summary: dict, rows: list[dict]) -> str:
    lines = []
    lines.append("# 标签升级收益小结")
    lines.append("")
    lines.append("## 升级对象")
    lines.append("")
    lines.append(
        f"- 从 `{summary['label_upgrade']['from']['label_version']}` / `{summary['label_upgrade']['from']['label_scope']}` "
        f"升级到 `{summary['label_upgrade']['to']['label_version']}` / `{summary['label_upgrade']['to']['label_scope']}`"
    )
    lines.append(f"- `Y_bad_v1` bad ratio：`{fmt(summary['label_upgrade']['from']['positive_ratio'])}`")
    lines.append(f"- `Y_bad_v2_min_default` coverage：`{summary['label_upgrade']['to']['coverage_summary']}`")
    lines.append("")
    lines.append("## 行为变化")
    lines.append("")
    bs = summary["behavior_shift"]
    lines.append(f"- `v1_good_but_v2_bad`：`{bs['v1_good_but_v2_bad']}`")
    lines.append(f"- `v1_good_and_v2_good`：`{bs['v1_good_and_v2_good']}`")
    lines.append(f"- `v1_bad_and_v2_bad`：`{bs['v1_bad_and_v2_bad']}`")
    lines.append(f"- 解读：{bs['interpretation']}")
    lines.append("")
    lines.append("## 主结果对照")
    lines.append("")
    for row in rows:
        lines.append(
            f"- `{row['model']}` @ `{row['label_version']}`: AUROC=`{fmt(row['auroc'])}`, "
            f"AUPRC=`{fmt(row['auprc'])}`, Brier=`{fmt(row['brier'])}`, ECE=`{fmt(row['ece'])}`. {row['note']}"
        )
    lines.append("")
    lines.append("## 关键判断")
    lines.append("")
    lines.append(f"- 规则式 baseline 变化：{summary['metric_shift']['interpretation']}")
    lines.append(
        f"- rule AUROC: `v1={fmt(summary['metric_shift']['rule_test_auroc_v1'])}` -> "
        f"`v2={fmt(summary['metric_shift']['rule_test_auroc_v2'])}`"
    )
    lines.append(
        f"- logistic AUROC: `v1={fmt(summary['metric_shift']['logistic_test_auroc_v1'])}` -> "
        f"`v2={fmt(summary['metric_shift']['logistic_test_auroc_v2'])}`"
    )
    lines.append(
        f"- DirRisk 方向违规项：`v1={summary['dir_risk']['v1_negative_aligned_weights']}` / "
        f"`v2={summary['dir_risk']['v2_negative_aligned_weights']}`"
    )
    lines.append("")
    lines.append("## Claim-Safe Conclusions")
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
    ap = argparse.ArgumentParser(description="Summarize post-task gains from upgrading label version from Y_bad_v1 to Y_bad_v2_min_default.")
    ap.add_argument("--v1_summary_json", required=True)
    ap.add_argument("--v2_summary_json", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    v1_summary = load_json(Path(args.v1_summary_json).expanduser().resolve())
    v2_summary = load_json(Path(args.v2_summary_json).expanduser().resolve())
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist a link back to the concrete result file used to construct v2 comparisons.
    v2_summary["source_files"] = {
        "main_results_json": str(Path(args.v2_summary_json).expanduser().resolve().parent / "post_v2_min_default_main_results.csv.json.placeholder")
    }
    # We rebuild rows from the markdown summary directory contents, so no external placeholder file is needed.
    # Replace with in-memory rows from v2 summary-compatible result files below.
    v2_main_rows = [
        {
            "name": "post_v2_rule_q_post_geom_only",
            "task": "post_v2_min_default",
            "source": "rule",
            "num_rows": v2_summary["results"]["rule_test"]["num_rows"],
            "positive_ratio": v2_summary["results"]["rule_test"]["positive_ratio"],
            "auroc": v2_summary["results"]["rule_test"]["auroc"],
            "auprc": v2_summary["results"]["rule_test"]["auprc"],
            "brier": v2_summary["results"]["rule_test"]["brier"],
            "ece": v2_summary["results"]["rule_test"]["ece"],
            "features": "Q_post_geom_only",
            "note": "Y_bad_v2_min_default 下的后验几何规则式 baseline。",
        },
        {
            "name": "post_v2_logistic_full",
            "task": "post_v2_min_default",
            "source": "logistic",
            "num_rows": v2_summary["results"]["logistic_test"]["num_rows"],
            "positive_ratio": v2_summary["results"]["logistic_test"]["positive_ratio"],
            "auroc": v2_summary["results"]["logistic_test"]["auroc"],
            "auprc": v2_summary["results"]["logistic_test"]["auprc"],
            "brier": v2_summary["results"]["logistic_test"]["brier"],
            "ece": v2_summary["results"]["logistic_test"]["ece"],
            "features": ",".join(v2_summary["task"]["allowed_feature_columns"]),
            "note": "Y_bad_v2_min_default 下的最小 post learning baseline。",
        },
        {
            "name": "post_v2_dir_risk",
            "task": "post_v2_min_default",
            "source": "dir_risk",
            "num_rows": v2_summary["results"]["dir_risk_test"]["num_rows"],
            "positive_ratio": v2_summary["results"]["dir_risk_test"]["positive_ratio"],
            "auroc": v2_summary["results"]["dir_risk_test"]["auroc"],
            "auprc": v2_summary["results"]["dir_risk_test"]["auprc"],
            "brier": v2_summary["results"]["dir_risk_test"]["brier"],
            "ece": v2_summary["results"]["dir_risk_test"]["ece"],
            "features": ",".join(v2_summary["task"]["allowed_feature_columns"]),
            "note": "Y_bad_v2_min_default 下的方向对齐 post baseline。",
        },
    ]
    # Provide the same shape expected by build_main_rows.
    v2_summary["source_files"] = {}
    v2_summary["main_rows_materialized"] = v2_main_rows

    def build_rows_from_summaries(v1, v2):
        v1_post = next(row for row in v1["main_results"] if row["name"] == "post_logistic_full")
        v1_rule = next(row for row in v1["main_results"] if row["name"] == "post_rule_q_post_geom_only")
        v1_dir = next(row for row in v1["main_results"] if row["name"] == "post_dir_risk")
        v2_rule = next(row for row in v2["main_rows_materialized"] if row["name"] == "post_v2_rule_q_post_geom_only")
        v2_log = next(row for row in v2["main_rows_materialized"] if row["name"] == "post_v2_logistic_full")
        v2_dir = next(row for row in v2["main_rows_materialized"] if row["name"] == "post_v2_dir_risk")
        return [
            {"model": "rule_q_post_geom_only", "label_version": "Y_bad_v1", "label_scope": v1["dataset"]["label_scope"], "num_rows": v1_rule["num_rows"], "positive_ratio": v1_rule["positive_ratio"], "auroc": v1_rule["auroc"], "auprc": v1_rule["auprc"], "brier": v1_rule["brier"], "ece": v1_rule["ece"], "note": "Y_bad_v1 下的后验几何规则式 baseline。"},
            {"model": "rule_q_post_geom_only", "label_version": "Y_bad_v2_min_default", "label_scope": v2["dataset"]["label_scope"], "num_rows": v2_rule["num_rows"], "positive_ratio": v2_rule["positive_ratio"], "auroc": v2_rule["auroc"], "auprc": v2_rule["auprc"], "brier": v2_rule["brier"], "ece": v2_rule["ece"], "note": "Y_bad_v2_min_default 下的后验几何规则式 baseline。"},
            {"model": "post_logistic", "label_version": "Y_bad_v1", "label_scope": v1["dataset"]["label_scope"], "num_rows": v1_post["num_rows"], "positive_ratio": v1_post["positive_ratio"], "auroc": v1_post["auroc"], "auprc": v1_post["auprc"], "brier": v1_post["brier"], "ece": v1_post["ece"], "note": "Y_bad_v1 下的最小 post learning baseline。"},
            {"model": "post_logistic", "label_version": "Y_bad_v2_min_default", "label_scope": v2["dataset"]["label_scope"], "num_rows": v2_log["num_rows"], "positive_ratio": v2_log["positive_ratio"], "auroc": v2_log["auroc"], "auprc": v2_log["auprc"], "brier": v2_log["brier"], "ece": v2_log["ece"], "note": "Y_bad_v2_min_default 下的最小 post learning baseline。"},
            {"model": "post_dir_risk", "label_version": "Y_bad_v1", "label_scope": v1["dataset"]["label_scope"], "num_rows": v1_dir["num_rows"], "positive_ratio": v1_dir["positive_ratio"], "auroc": v1_dir["auroc"], "auprc": v1_dir["auprc"], "brier": v1_dir["brier"], "ece": v1_dir["ece"], "note": "Y_bad_v1 下的方向对齐 post baseline。"},
            {"model": "post_dir_risk", "label_version": "Y_bad_v2_min_default", "label_scope": v2["dataset"]["label_scope"], "num_rows": v2_dir["num_rows"], "positive_ratio": v2_dir["positive_ratio"], "auroc": v2_dir["auroc"], "auprc": v2_dir["auprc"], "brier": v2_dir["brier"], "ece": v2_dir["ece"], "note": "Y_bad_v2_min_default 下的方向对齐 post baseline。"},
        ]

    rows = build_rows_from_summaries(v1_summary, v2_summary)
    summary = build_gain_summary(v1_summary, v2_summary)

    write_json(out_dir / "label_upgrade_gain_summary.json", summary)
    write_csv(
        out_dir / "label_upgrade_gain_main_results.csv",
        ["model", "label_version", "label_scope", "num_rows", "positive_ratio", "auroc", "auprc", "brier", "ece", "note"],
        rows,
    )
    write_text(out_dir / "label_upgrade_gain_summary.md", build_markdown(summary, rows))

    print(f"[LabelUpgradeSummary] saved -> {out_dir}")


if __name__ == "__main__":
    main()
