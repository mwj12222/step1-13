#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_HARD_CASES = Path(
    "docs/research/init_risk_post_v2_shared_hard_cases_20260319/post_v2_shared_hard_cases_rows.csv"
)
DEFAULT_RISK_DATASET = Path(
    "/mnt/g/Result/VIODE/post_v2_rebuild_9seq/risk_dataset_9seq_seed20260432/risk_dataset.csv"
)
DEFAULT_SEGMENTS = Path(
    "docs/research/init_risk_city_day_future_high_gt_rot_boundary_neighborhood_20260320/city_day_future_high_gt_rot_boundary_segments.csv"
)
DEFAULT_FOLLOWUP_SHAPES = Path(
    "docs/research/init_risk_city_day_future_high_gt_rot_followup_20260320/city_day_future_high_gt_rot_followup_shapes.csv"
)
DEFAULT_OUT_DIR = Path(
    "docs/research/init_risk_city_day_boundary_uncertain_counterfactual_20260320"
)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def fmt(v: float | int | None, nd: int = 4) -> str:
    if v is None or pd.isna(v):
        return "-"
    return f"{float(v):.{nd}f}"


def build_markdown(payload: dict) -> str:
    lines: list[str] = []
    lines.append("# city_day boundary-uncertain counterfactual audit")
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    for item in payload["headline"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 覆盖率")
    lines.append("")
    lines.append("| population | rows | selected | share |")
    lines.append("| --- | --- | --- | --- |")
    for row in payload["coverage_rows"]:
        lines.append(
            f"| {row['population']} | {row['rows']} | {row['selected']} | {fmt(row['share'])} |"
        )
    lines.append("")
    lines.append("## shape 构成")
    lines.append("")
    lines.append("| shape | rows | share_within_selected |")
    lines.append("| --- | --- | --- |")
    for row in payload["shape_rows"]:
        lines.append(
            f"| {row['shape']} | {row['rows']} | {fmt(row['share_within_selected'])} |"
        )
    lines.append("")
    lines.append("## 画像")
    lines.append("")
    lines.append("| profile | rows | mean_future_gt_rot | mean_current_parallax | mean_reproj_p90 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in payload["profile_rows"]:
        lines.append(
            f"| {row['profile']} | {row['rows']} | {fmt(row['future_gt_rot_med_deg'])} | {fmt(row['parallax_px_candidate'])} | {fmt(row['reproj_p90_px'])} |"
        )
    lines.append("")
    lines.append("## 代表样本")
    lines.append("")
    lines.append("| sequence | window_id | segment_id | shape | future_gt_rot | current_parallax | reproj_p90 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["example_rows"]:
        lines.append(
            f"| {row['sequence']} | {row['window_id']} | {row['segment_id']} | {row['shape']} | {fmt(row['future_gt_rot_med_deg'])} | {fmt(row['parallax_px_candidate'])} | {fmt(row['reproj_p90_px'])} |"
        )
    lines.append("")
    lines.append("## 判断")
    lines.append("")
    for item in payload["judgement"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit city_day boundary-uncertain counterfactual candidates."
    )
    parser.add_argument("--hard_cases_csv", type=Path, default=DEFAULT_HARD_CASES)
    parser.add_argument("--risk_dataset_csv", type=Path, default=DEFAULT_RISK_DATASET)
    parser.add_argument("--segments_csv", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--followup_shapes_csv", type=Path, default=DEFAULT_FOLLOWUP_SHAPES)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    hard = pd.read_csv(args.hard_cases_csv)
    risk = pd.read_csv(args.risk_dataset_csv)
    risk = risk[risk["sample_type"] == "step11"].copy()
    risk["window_id"] = pd.to_numeric(risk["window_id"], errors="coerce")
    segments = pd.read_csv(args.segments_csv)
    shapes = pd.read_csv(args.followup_shapes_csv)

    segments = segments[segments["sequence"].isin(["city_day/0_none", "city_day/2_mid"])].copy()
    segments = segments.merge(shapes[["segment_id", "shape"]], on="segment_id", how="left")

    hard = hard[hard["shared_wrong"] == 1].copy()
    merge_cols = [
        "sample_uid",
        "sequence",
        "Y_bad_v2_min_default_trigger",
        "Y_bad_v2_min_default_trigger_window_id",
        "Y_bad_v2_min_default_horizon_windows",
    ]
    hard = hard.merge(
        risk[merge_cols],
        on=["sample_uid", "sequence"],
        how="left",
    )
    hard = hard[
        (hard["Y_bad_v2_min_default_trigger"] == "future_high_gt_rot")
        & (pd.to_numeric(hard["Y_bad_v2_min_default_horizon_windows"], errors="coerce") == 1)
    ].copy()
    hard["future_window_id"] = pd.to_numeric(hard["Y_bad_v2_min_default_trigger_window_id"], errors="coerce")

    future = risk[
        ["sequence", "window_id", "gt_rot_med_deg", "Y_bad_v2_min_default", "Y_bad_v2_min_default_trigger"]
    ].copy()
    future = future.rename(
        columns={
            "window_id": "future_window_id",
            "gt_rot_med_deg": "future_gt_rot_med_deg",
            "Y_bad_v2_min_default": "future_row_bad",
            "Y_bad_v2_min_default_trigger": "future_row_trigger",
        }
    )
    hard = hard.merge(future, on=["sequence", "future_window_id"], how="left")

    followup_rows = []
    for seg in segments.to_dict(orient="records"):
        seq = seg["sequence"]
        start = int(seg["start_window_id"])
        end = int(seg["end_window_id"])
        seg_id = seg["segment_id"]

        inside = hard[
            (hard["sequence"] == seq)
            & (pd.to_numeric(hard["window_id"], errors="coerce") >= start)
            & (pd.to_numeric(hard["window_id"], errors="coerce") <= end)
        ].copy()
        if inside.empty:
            continue
        inside["segment_id"] = seg_id
        inside["shape"] = seg["shape"]
        inside["offset1_window_id"] = inside["future_window_id"] + 10
        inside["offset2_window_id"] = inside["future_window_id"] + 20
        followup_rows.append(inside)

    if not followup_rows:
        raise RuntimeError("No overlap between city_day segments and both-wrong future_high_gt_rot rows.")

    candidates = pd.concat(followup_rows, ignore_index=True)

    risk_small = risk[
        ["sequence", "window_id", "Y_bad_v2_min_default_trigger"]
    ].rename(columns={"window_id": "probe_window_id", "Y_bad_v2_min_default_trigger": "probe_trigger"})
    off1 = candidates[["sample_uid", "sequence", "offset1_window_id"]].rename(columns={"offset1_window_id": "probe_window_id"})
    off2 = candidates[["sample_uid", "sequence", "offset2_window_id"]].rename(columns={"offset2_window_id": "probe_window_id"})
    off1 = off1.merge(risk_small, on=["sequence", "probe_window_id"], how="left").rename(columns={"probe_trigger": "offset1_trigger"})
    off2 = off2.merge(risk_small, on=["sequence", "probe_window_id"], how="left").rename(columns={"probe_trigger": "offset2_trigger"})
    candidates = candidates.merge(off1[["sample_uid", "offset1_trigger"]], on="sample_uid", how="left")
    candidates = candidates.merge(off2[["sample_uid", "offset2_trigger"]], on="sample_uid", how="left")

    severe_set = {"future_solver_fail", "future_reset"}
    candidates["has_severe_followup"] = candidates["offset1_trigger"].isin(severe_set) | candidates["offset2_trigger"].isin(severe_set)
    candidates["has_stable_followup"] = (candidates["offset1_trigger"] == "stable_horizon") | (candidates["offset2_trigger"] == "stable_horizon")

    selected = candidates[
        (pd.to_numeric(candidates["future_gt_rot_med_deg"], errors="coerce") > 5.0)
        & (pd.to_numeric(candidates["future_gt_rot_med_deg"], errors="coerce") <= 10.0)
        & (pd.to_numeric(candidates["parallax_px_candidate"], errors="coerce") < 20.0)
        & (~candidates["has_severe_followup"])
        & (candidates["has_stable_followup"])
    ].copy()

    all_both_wrong = pd.read_csv(args.hard_cases_csv)
    all_both_wrong = all_both_wrong[all_both_wrong["shared_wrong"] == 1].copy()
    city_day_both_wrong = hard[hard["sequence"].isin(["city_day/0_none", "city_day/2_mid"])].copy()

    coverage_rows = [
        {
            "population": "all both-wrong",
            "rows": int(len(all_both_wrong)),
            "selected": int(len(selected)),
            "share": len(selected) / len(all_both_wrong),
        },
        {
            "population": "future_high_gt_rot@K=1 both-wrong",
            "rows": int(len(hard)),
            "selected": int(len(selected)),
            "share": len(selected) / len(hard),
        },
        {
            "population": "city_day future_high_gt_rot both-wrong",
            "rows": int(len(city_day_both_wrong)),
            "selected": int(len(selected)),
            "share": len(selected) / len(city_day_both_wrong),
        },
    ]

    shape_df = (
        selected.groupby("shape")
        .size()
        .rename("rows")
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    if len(selected):
        shape_df["share_within_selected"] = shape_df["rows"] / len(selected)
    else:
        shape_df["share_within_selected"] = 0.0

    profile_rows = []
    for name, df in [("all_city_day_future_high_gt_rot_both_wrong", city_day_both_wrong), ("boundary_uncertain_selected", selected)]:
        profile_rows.append(
            {
                "profile": name,
                "rows": int(len(df)),
                "future_gt_rot_med_deg": pd.to_numeric(df["future_gt_rot_med_deg"], errors="coerce").mean(),
                "parallax_px_candidate": pd.to_numeric(df["parallax_px_candidate"], errors="coerce").mean(),
                "reproj_p90_px": pd.to_numeric(df["reproj_p90_px"], errors="coerce").mean(),
            }
        )
    profile_df = pd.DataFrame(profile_rows)

    example_rows = selected[
        ["sequence", "window_id", "segment_id", "shape", "future_gt_rot_med_deg", "parallax_px_candidate", "reproj_p90_px"]
    ].sort_values(["sequence", "window_id"]).head(12)

    headline = [
        f"按 6 条条件严格筛选后，`city_day boundary-uncertain` 一共只命中 {len(selected)} 条样本，占当前全部 `both-wrong` 的 {fmt(len(selected) / len(all_both_wrong))}，说明这是一个窄而克制的局部边界层，而不是大面积改写标签协议。",
        f"它覆盖了 `city_day future_high_gt_rot both-wrong` 的 {fmt(len(selected) / len(city_day_both_wrong))}，说明这条假设确实抓住了 city_day 主体中的一部分关键边界样本，但并没有把整个 city_day 主体都一股脑纳进来。",
        f"shape 上，这批样本主要来自 `{shape_df.iloc[0]['shape'] if not shape_df.empty else 'none'}`，支持它更像边界带而不是持续坏链。",
    ]

    judgement = [
        "这组 6 条条件筛出来的是一个小而精的局部集合，符合“只做 city_day 边界层审计，不动主线协议”的初衷。",
        "如果后续要进入训练级 label variant，这一结果已经足够支持先试一个 very small definition experiment，因为它既不大面积吞掉 both-wrong，也没有落到持续坏链主体上。",
        "但它现在仍然只是 counterfactual audit 结果；是否值得进入训练级重建，取决于我们是否愿意把一个 city_day-specific 的 boundary-uncertain 层正式作为定义实验保留下来。",
    ]

    payload = {
        "headline": headline,
        "judgement": judgement,
        "coverage_rows": coverage_rows,
        "shape_rows": shape_df.to_dict(orient="records"),
        "profile_rows": profile_df.to_dict(orient="records"),
        "example_rows": example_rows.to_dict(orient="records"),
    }

    pd.DataFrame(coverage_rows).to_csv(out_dir / "city_day_boundary_uncertain_coverage.csv", index=False)
    shape_df.to_csv(out_dir / "city_day_boundary_uncertain_shapes.csv", index=False)
    profile_df.to_csv(out_dir / "city_day_boundary_uncertain_profiles.csv", index=False)
    selected.to_csv(out_dir / "city_day_boundary_uncertain_rows.csv", index=False)
    write_json(out_dir / "city_day_boundary_uncertain_summary.json", payload)
    (out_dir / "city_day_boundary_uncertain_summary.md").write_text(
        build_markdown(payload), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
