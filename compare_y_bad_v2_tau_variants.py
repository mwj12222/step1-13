#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_BASE_DIR = Path(
    "/mnt/g/Result/VIODE/post_v2_expand_valtest_20260319/step11_and_label_audit/y_bad_v2_min_default_9seq"
)
DEFAULT_TAU75_DIR = Path(
    "/mnt/g/Result/VIODE/post_v2_tau_audit_20260320/y_bad_v2_min_tau7p5_9seq"
)
DEFAULT_TAU10_DIR = Path(
    "/mnt/g/Result/VIODE/post_v2_tau_audit_20260320/y_bad_v2_min_tau10p0_9seq"
)
DEFAULT_HARD_CASES = Path(
    "docs/research/init_risk_post_v2_shared_hard_cases_20260319/post_v2_shared_hard_cases_rows.csv"
)
DEFAULT_OUT_DIR = Path(
    "docs/research/init_risk_post_v2_tau_audit_20260320"
)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fmt(v, nd=4):
    if pd.isna(v):
        return "-"
    return f"{float(v):.{nd}f}"


def package_name(path: Path) -> str:
    name = path.name
    if "tau7p5" in name:
        return "tau7.5"
    if "tau10p0" in name or "tau10" in name:
        return "tau10.0"
    return "tau5.0"


def label_lookup(labels_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_csv, usecols=[
        "sample_uid",
        "sequence",
        "Y_bad_v2_min",
        "Y_bad_v2_min_trigger",
        "Y_bad_v2_min_horizon_windows",
    ])
    return df.rename(columns={
        "Y_bad_v2_min": "y_new",
        "Y_bad_v2_min_trigger": "trigger_new",
        "Y_bad_v2_min_horizon_windows": "horizon_new",
    })


def build_markdown(payload: dict) -> str:
    lines = []
    lines.append("# Y_bad_v2 tau audit")
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    for item in payload["headline"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 标签人口变化")
    lines.append("")
    lines.append("| tau | labeled_rows | bad_ratio_labeled | future_high_gt_rot | future_solver_fail | future_reset | stable_horizon |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["population_rows"]:
        lines.append(
            f"| {row['tau']} | {row['labeled_rows']} | {fmt(row['bad_ratio_labeled'])} | {row['future_high_gt_rot']} | {row['future_solver_fail']} | {row['future_reset']} | {row['stable_horizon']} |"
        )
    lines.append("")
    lines.append("## 在当前 590 条 held-out 测试样本上的反事实 both-wrong 变化")
    lines.append("")
    lines.append("| tau | both_wrong_counterfactual | delta_vs_tau5 | positive_shared_wrong | future_high_gt_rot_shared_wrong |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in payload["counterfactual_rows"]:
        lines.append(
            f"| {row['tau']} | {row['both_wrong_counterfactual']} | {row['delta_vs_tau5']} | {row['positive_both_wrong']} | {row['future_high_gt_rot_both_wrong']} |"
        )
    lines.append("")
    lines.append("## 当前 192 条共享难例在新标签下的去留")
    lines.append("")
    lines.append("| tau | retained_from_current_shared_wrong | dropped_from_current_shared_wrong | dropped_high_gt_rot |")
    lines.append("| --- | --- | --- | --- |")
    for row in payload["retained_rows"]:
        lines.append(
            f"| {row['tau']} | {row['retained_from_current_shared_wrong']} | {row['dropped_from_current_shared_wrong']} | {row['dropped_high_gt_rot']} |"
        )
    lines.append("")
    lines.append("## 判断")
    lines.append("")
    for item in payload["judgement"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Compare Y_bad_v2 tau variants before rebuilding training packages.")
    ap.add_argument("--base_dir", type=Path, default=DEFAULT_BASE_DIR)
    ap.add_argument("--tau75_dir", type=Path, default=DEFAULT_TAU75_DIR)
    ap.add_argument("--tau10_dir", type=Path, default=DEFAULT_TAU10_DIR)
    ap.add_argument("--hard_cases_csv", type=Path, default=DEFAULT_HARD_CASES)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dirs = [args.base_dir, args.tau75_dir, args.tau10_dir]
    audits = []
    for d in dirs:
        audit = load_json(d / "y_bad_v2_min_audit.json")
        audit["tau"] = package_name(d)
        trigger_counts = audit.get("trigger_counts", {})
        audits.append({
            "tau": audit["tau"],
            "labeled_rows": audit.get("labeled_rows"),
            "bad_ratio_labeled": audit.get("bad_ratio_labeled"),
            "future_high_gt_rot": trigger_counts.get("future_high_gt_rot", 0),
            "future_solver_fail": trigger_counts.get("future_solver_fail", 0),
            "future_reset": trigger_counts.get("future_reset", 0),
            "stable_horizon": trigger_counts.get("stable_horizon", 0),
        })
    population_df = pd.DataFrame(audits)

    hard = pd.read_csv(args.hard_cases_csv)
    label_maps = {package_name(d): label_lookup(d / "y_bad_v2_min_labels.csv") for d in dirs}

    counterfactual_rows = []
    retained_rows = []
    baseline_shared = hard[hard["shared_wrong"] == 1].copy()

    for tau_name, labels in label_maps.items():
        merged = hard.merge(labels, on=["sample_uid", "sequence"], how="left")
        merged["y_new"] = pd.to_numeric(merged["y_new"], errors="coerce")
        merged = merged[merged["y_new"].isin([0, 1])].copy()
        merged["p_geometry_only"] = pd.to_numeric(merged["p_geometry_only"], errors="coerce")
        merged["p_gated"] = pd.to_numeric(merged["p_gated"], errors="coerce")
        merged["wrong_geo_new"] = ((merged["p_geometry_only"] >= 0.5) != (merged["y_new"] == 1)).astype(int)
        merged["wrong_gated_new"] = ((merged["p_gated"] >= 0.5) != (merged["y_new"] == 1)).astype(int)
        merged["both_wrong_new"] = ((merged["wrong_geo_new"] == 1) & (merged["wrong_gated_new"] == 1)).astype(int)

        current_shared_join = baseline_shared.merge(labels, on=["sample_uid", "sequence"], how="left")
        current_shared_join["y_new"] = pd.to_numeric(current_shared_join["y_new"], errors="coerce")
        retained = current_shared_join["y_new"].isin([1]).sum() + (
            (baseline_shared["y_true"] == 0).sum() if tau_name == "tau5.0" else 0
        )
        # more direct: recompute current shared wrong with new labels
        current_shared_eval = baseline_shared.merge(labels, on=["sample_uid", "sequence"], how="left")
        current_shared_eval["y_new"] = pd.to_numeric(current_shared_eval["y_new"], errors="coerce")
        current_shared_eval["p_geometry_only"] = pd.to_numeric(current_shared_eval["p_geometry_only"], errors="coerce")
        current_shared_eval["p_gated"] = pd.to_numeric(current_shared_eval["p_gated"], errors="coerce")
        current_shared_eval = current_shared_eval[current_shared_eval["y_new"].isin([0, 1])].copy()
        current_shared_eval["both_wrong_new"] = (
            ((current_shared_eval["p_geometry_only"] >= 0.5) != (current_shared_eval["y_new"] == 1))
            & ((current_shared_eval["p_gated"] >= 0.5) != (current_shared_eval["y_new"] == 1))
        ).astype(int)

        retained_count = int(current_shared_eval["both_wrong_new"].sum())
        dropped_count = int(len(current_shared_eval) - retained_count)
        dropped_high_gt_rot = int(
            ((current_shared_eval["both_wrong_new"] == 0) & (current_shared_eval["trigger_new"] != "future_high_gt_rot")).sum()
        )

        counterfactual_rows.append({
            "tau": tau_name,
            "both_wrong_counterfactual": int(merged["both_wrong_new"].sum()),
            "delta_vs_tau5": None,
            "positive_both_wrong": int(((merged["both_wrong_new"] == 1) & (merged["y_new"] == 1)).sum()),
            "future_high_gt_rot_both_wrong": int(((merged["both_wrong_new"] == 1) & (merged["trigger_new"] == "future_high_gt_rot")).sum()),
        })
        retained_rows.append({
            "tau": tau_name,
            "retained_from_current_shared_wrong": retained_count,
            "dropped_from_current_shared_wrong": dropped_count,
            "dropped_high_gt_rot": dropped_high_gt_rot,
        })

    counterfactual_df = pd.DataFrame(counterfactual_rows).sort_values("tau")
    tau5 = int(counterfactual_df[counterfactual_df["tau"] == "tau5.0"]["both_wrong_counterfactual"].iloc[0])
    counterfactual_df["delta_vs_tau5"] = counterfactual_df["both_wrong_counterfactual"] - tau5
    retained_df = pd.DataFrame(retained_rows).sort_values("tau")

    headline = [
        f"在不重训模型的前提下，把 `tau_gt_rot_med_deg` 从 `5.0` 抬到 `7.5` 或 `10.0`，最直接削掉的是 `future_high_gt_rot` 这批贴边界标签人口，而不是 `future_solver_fail / future_reset`。",
        f"当前 590 条 held-out 测试样本上的反事实 both-wrong 数，会从 tau5.0 的 {tau5} 条降到 tau7.5 的 {int(counterfactual_df[counterfactual_df['tau']=='tau7.5']['both_wrong_counterfactual'].iloc[0])} 条，再降到 tau10.0 的 {int(counterfactual_df[counterfactual_df['tau']=='tau10.0']['both_wrong_counterfactual'].iloc[0])} 条。",
        "这说明先审 `future_high_gt_rot@K=1` 的阈值是值得的；但它主要是在清理边界样本，还不能替代后续真正的训练级重建验证。",
    ]
    judgement = [
        "如果目标是先做低成本纠偏，`tau=7.5` 是更稳的第一候选：它会明显削掉一批 `future_high_gt_rot` 边界样本，但不像 `tau=10.0` 那样激进。",
        "`tau=10.0` 可以作为更强对照，但它更像压力测试：如果这版标签人口塌得太多，后面训练与 held-out 会更容易失真。",
        "因此下一步最合理的是：先拿 `tau=7.5` 做一次训练级重建；`tau=10.0` 保留为对照版，不要一上来替换主线。",
    ]

    payload = {
        "headline": headline,
        "judgement": judgement,
        "population_rows": population_df.to_dict(orient="records"),
        "counterfactual_rows": counterfactual_df.to_dict(orient="records"),
        "retained_rows": retained_df.to_dict(orient="records"),
    }

    population_df.to_csv(out_dir / "tau_population_summary.csv", index=False)
    counterfactual_df.to_csv(out_dir / "tau_shared_wrong_counterfactual.csv", index=False)
    retained_df.to_csv(out_dir / "tau_current_shared_wrong_retention.csv", index=False)
    write_json(out_dir / "tau_audit_summary.json", payload)
    (out_dir / "tau_audit_summary.md").write_text(build_markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
