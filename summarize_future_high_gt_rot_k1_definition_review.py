#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUT_DIR = PROJECT_ROOT / "docs" / "research" / "init_risk_future_high_gt_rot_k1_definition_review_20260320"
DEFAULT_TAU_AUDIT_DIR = PROJECT_ROOT / "docs" / "research" / "init_risk_post_v2_tau_audit_20260320"
DEFAULT_FUTURE_HIGH_ROT_DIR = PROJECT_ROOT / "docs" / "research" / "init_risk_post_v2_future_high_gt_rot_shared_20260319"


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def fmt(v, nd: int = 4) -> str:
    if pd.isna(v):
        return "-"
    return f"{float(v):.{nd}f}"


def build_markdown(payload: dict) -> str:
    lines = []
    lines.append("# future_high_gt_rot@K=1 definition review")
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    for item in payload["headline"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 当前定义")
    lines.append("")
    lines.append("- `gate_post == accept`")
    lines.append("- `horizon_windows = 1`")
    lines.append("- 下一窗口 `gt_rot_med_deg > tau_gt_rot_med_deg` 即触发 `future_high_gt_rot`")
    lines.append("")
    lines.append("## 阈值对照")
    lines.append("")
    lines.append("| tau | labeled_rows | bad_ratio | future_high_gt_rot | both_wrong_counterfactual |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in payload["tau_rows"]:
        lines.append(
            f"| {row['tau']} | {row['labeled_rows']} | {fmt(row['bad_ratio_labeled'])} | {row['future_high_gt_rot']} | {row['both_wrong_counterfactual']} |"
        )
    lines.append("")
    lines.append("## 共享难例主来源画像")
    lines.append("")
    lines.append("| profile | rows | mean gt_rot_deg | mean parallax | mean reproj_p90 | mean Q_post_geom_only |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in payload["profile_rows"]:
        lines.append(
            f"| {row['profile']} | {row['rows']} | {fmt(row['gt_rot_med_deg'])} | {fmt(row['parallax_px_candidate'])} | {fmt(row['reproj_p90_px'])} | {fmt(row['Q_post_geom_only'])} |"
        )
    lines.append("")
    lines.append("## 当前建议")
    lines.append("")
    for item in payload["recommendations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize future_high_gt_rot@K=1 definition review.")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tau_audit_dir", type=Path, default=DEFAULT_TAU_AUDIT_DIR)
    parser.add_argument("--future_high_rot_dir", type=Path, default=DEFAULT_FUTURE_HIGH_ROT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tau_population = pd.read_csv(args.tau_audit_dir / "tau_population_summary.csv")
    tau_counter = pd.read_csv(args.tau_audit_dir / "tau_shared_wrong_counterfactual.csv")
    tau_rows = tau_population.merge(tau_counter[["tau", "both_wrong_counterfactual"]], on="tau", how="left")
    tau_rows = tau_rows[tau_rows["tau"].isin(["tau5.0", "tau7.5", "tau10.0"])]

    profile_rows = pd.read_csv(args.future_high_rot_dir / "future_high_gt_rot_profile_summary.csv").to_dict("records")

    headline = [
        "当前 `future_high_gt_rot@K=1` 的主要问题不在 horizon，而在旋转阈值过贴边界。",
        "现有共享难例主来源并不是“高旋转崩坏”，而是“轻到中度旋转触发的 next-window bad 样本”；它们几何分数仍高、重投影仍低，因此更像标签边界而不是纯 recoverability 缺口。",
        "`tau=7.5` 是当前最稳的第一步定义收紧：它保留 labeled_rows 不变，但明显削减 `future_high_gt_rot` 人口和 both-wrong；`tau=10.0` 更适合作为激进对照，而不是主线替换。",
    ]

    recommendations = [
        "先固定 `K=1` 不动，优先把 `tau_gt_rot_med_deg` 从 5.0 提到 7.5 做标签对照主候选。",
        "不要现在同时改 `K`、trigger 集和复合条件；否则我们会分不清是阈值纠偏有效，还是定义整体漂移。",
        "如果下一步还要继续审定义，最值钱的顺序应是：先看 `tau=7.5` 训练级重建，再决定是否要试 `tau=10.0` 或复合触发。",
        "只有当阈值收紧后，`future_high_gt_rot` 仍然主导共享难例，才值得继续试更复杂的复合定义。",
    ]

    payload = {
        "headline": headline,
        "tau_rows": tau_rows.to_dict("records"),
        "profile_rows": profile_rows,
        "recommendations": recommendations,
    }

    write_json(out_dir / "future_high_gt_rot_k1_definition_review.json", payload)
    write_csv(out_dir / "future_high_gt_rot_k1_tau_comparison.csv", tau_rows.to_dict("records"))
    (out_dir / "future_high_gt_rot_k1_definition_review.md").write_text(build_markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
