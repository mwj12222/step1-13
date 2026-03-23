#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_SEGMENTS = Path(
    "docs/research/init_risk_city_day_future_high_gt_rot_boundary_neighborhood_20260320/city_day_future_high_gt_rot_boundary_segments.csv"
)
DEFAULT_HARD_CASES = Path(
    "docs/research/init_risk_post_v2_shared_hard_cases_20260319/post_v2_shared_hard_cases_rows.csv"
)
DEFAULT_RISK_DATASET = Path(
    "/mnt/g/Result/VIODE/post_v2_rebuild_9seq/risk_dataset_9seq_seed20260432/risk_dataset.csv"
)
DEFAULT_OUT_DIR = Path(
    "docs/research/init_risk_city_day_future_high_gt_rot_followup_20260320"
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


def classify_segment(df: pd.DataFrame) -> str:
    bad = {int(r["offset_idx"]): float(r["bad_rate"]) for _, r in df.iterrows()}
    high_gt = {int(r["offset_idx"]): float(r["future_high_gt_rot_rate"]) for _, r in df.iterrows()}
    severe = {int(r["offset_idx"]): float(r["severe_failure_rate"]) for _, r in df.iterrows()}
    if bad.get(0, 0) >= 0.75 and bad.get(1, 0) <= 0.5 and bad.get(2, 0) <= 0.5:
        return "short_boundary_jitter"
    if bad.get(0, 0) >= 0.75 and (bad.get(1, 0) >= 0.75 or bad.get(2, 0) >= 0.75):
        return "persistent_bad_chain"
    if severe.get(1, 0) > 0 or severe.get(2, 0) > 0:
        return "delayed_severe_instability"
    if high_gt.get(0, 0) >= 0.75 and high_gt.get(1, 0) >= 0.5:
        return "extended_rot_boundary"
    return "mixed_followup"


def build_markdown(payload: dict) -> str:
    lines: list[str] = []
    lines.append("# city_day future_high_gt_rot follow-up chain")
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    for item in payload["headline"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## offset 0/+1/+2 汇总")
    lines.append("")
    lines.append("| segment_id | offset | rows | bad_rate | future_high_gt_rot_rate | stable_horizon_rate | severe_failure_rate | dominant_trigger |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["followup_rows"]:
        lines.append(
            f"| {row['segment_id']} | {row['offset_idx']} | {row['rows']} | {fmt(row['bad_rate'])} | {fmt(row['future_high_gt_rot_rate'])} | {fmt(row['stable_horizon_rate'])} | {fmt(row['severe_failure_rate'])} | {row['dominant_trigger']} |"
        )
    lines.append("")
    lines.append("## 段级判型")
    lines.append("")
    lines.append("| segment_id | sequence | shape | offset0_bad | offset1_bad | offset2_bad |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in payload["segment_shapes"]:
        lines.append(
            f"| {row['segment_id']} | {row['sequence']} | {row['shape']} | {fmt(row['offset0_bad'])} | {fmt(row['offset1_bad'])} | {fmt(row['offset2_bad'])} |"
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
        description="Analyze 1-2 step follow-up status after future_high_gt_rot boundary windows."
    )
    parser.add_argument("--segments_csv", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--hard_cases_csv", type=Path, default=DEFAULT_HARD_CASES)
    parser.add_argument("--risk_dataset_csv", type=Path, default=DEFAULT_RISK_DATASET)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    segments = pd.read_csv(args.segments_csv)
    hard = pd.read_csv(args.hard_cases_csv)
    risk = pd.read_csv(args.risk_dataset_csv)
    risk = risk[risk["sample_type"] == "step11"].copy()
    risk["window_id"] = pd.to_numeric(risk["window_id"], errors="coerce")

    target_segments = segments[segments["sequence"].isin(["city_day/0_none", "city_day/2_mid"])].copy()

    label_cols = [
        "sample_uid",
        "sequence",
        "window_id",
        "Y_bad_v2_min_default",
        "Y_bad_v2_min_default_trigger",
        "Y_bad_v2_min_default_trigger_window_id",
    ]
    label_df = risk[label_cols].copy()

    hard_l = hard.merge(
        label_df[
            ["sample_uid", "sequence", "Y_bad_v2_min_default_trigger_window_id", "Y_bad_v2_min_default_trigger"]
        ],
        on=["sample_uid", "sequence"],
        how="left",
    )
    hard_l = hard_l[(hard_l["shared_wrong"] == 1)].copy()

    followup_rows = []
    segment_shape_rows = []

    for seg in target_segments.to_dict(orient="records"):
        seq = seg["sequence"]
        start = int(seg["start_window_id"])
        end = int(seg["end_window_id"])
        segment_id = seg["segment_id"]

        inside = hard_l[
            (hard_l["sequence"] == seq)
            & (pd.to_numeric(hard_l["window_id"], errors="coerce") >= start)
            & (pd.to_numeric(hard_l["window_id"], errors="coerce") <= end)
        ].copy()
        inside["trigger_window_id"] = pd.to_numeric(
            inside["Y_bad_v2_min_default_trigger_window_id"], errors="coerce"
        )
        inside = inside[inside["trigger_window_id"].notna()].copy()

        for offset in [0, 1, 2]:
            probes = inside[["sample_uid", "sequence", "trigger_window_id"]].copy()
            probes["probe_window_id"] = probes["trigger_window_id"] + offset * 10
            joined = probes.merge(
                risk[
                    [
                        "sequence",
                        "window_id",
                        "Y_bad_v2_min_default",
                        "Y_bad_v2_min_default_trigger",
                    ]
                ],
                left_on=["sequence", "probe_window_id"],
                right_on=["sequence", "window_id"],
                how="left",
            )
            trigger_counts = joined["Y_bad_v2_min_default_trigger"].fillna("missing").value_counts()
            dominant_trigger = trigger_counts.index[0] if len(trigger_counts) else "missing"
            rows = len(joined)
            followup_rows.append(
                {
                    "segment_id": segment_id,
                    "sequence": seq,
                    "offset_idx": offset,
                    "rows": int(rows),
                    "bad_rate": pd.to_numeric(joined["Y_bad_v2_min_default"], errors="coerce").mean(),
                    "future_high_gt_rot_rate": (joined["Y_bad_v2_min_default_trigger"] == "future_high_gt_rot").mean(),
                    "stable_horizon_rate": (joined["Y_bad_v2_min_default_trigger"] == "stable_horizon").mean(),
                    "severe_failure_rate": joined["Y_bad_v2_min_default_trigger"].isin(
                        ["future_solver_fail", "future_reset"]
                    ).mean(),
                    "dominant_trigger": dominant_trigger,
                }
            )

        seg_follow = pd.DataFrame([r for r in followup_rows if r["segment_id"] == segment_id]).sort_values("offset_idx")
        segment_shape_rows.append(
            {
                "segment_id": segment_id,
                "sequence": seq,
                "shape": classify_segment(seg_follow),
                "offset0_bad": seg_follow.loc[seg_follow["offset_idx"] == 0, "bad_rate"].iloc[0],
                "offset1_bad": seg_follow.loc[seg_follow["offset_idx"] == 1, "bad_rate"].iloc[0],
                "offset2_bad": seg_follow.loc[seg_follow["offset_idx"] == 2, "bad_rate"].iloc[0],
            }
        )

    followup_df = pd.DataFrame(followup_rows).sort_values(["segment_id", "offset_idx"])
    shape_df = pd.DataFrame(segment_shape_rows).sort_values(["sequence", "segment_id"])

    num_boundary = int((shape_df["shape"] == "short_boundary_jitter").sum())
    num_persistent = int(shape_df["shape"].isin(["persistent_bad_chain", "delayed_severe_instability", "extended_rot_boundary"]).sum())

    headline = [
        f"这批 city_day 主导段整体更像边界带而不是持续崩坏链：{num_boundary} 个段更接近 `short_boundary_jitter`，而只有 {num_persistent} 个段表现出明显持续 bad 或延迟升级特征。",
        "很多段在触发 future window 本身是 bad，但到后续 1-2 个窗口会部分或完全回到 stable_horizon，这支持“短时抖动边界”而不是“必然向后恶化”的解释。",
        "更激烈的持续坏链主要出现在少数局部段，而不是 city_day 主体模式；这说明对 city_day 类样本，全局提高阈值并不是最有针对性的修正。",
    ]

    judgement = [
        "city_day 主体模式更接近短时旋转边界带：future window 触发后，后续 1-2 个窗口并不会一致地保持 bad，更少直接升级成 solver/reset。",
        "因此 `future_high_gt_rot@K=1` 的 city_day 主导难例，确实更像标签边界问题，而不是已经进入稳定坏链的显性失稳。",
        "后续如果继续审定义，更值得对 city_day 这类 sequence-aware 边界模式做局部语义约束，而不是继续统一抬 tau 或扩大 horizon。",
    ]

    payload = {
        "headline": headline,
        "judgement": judgement,
        "followup_rows": followup_df.to_dict(orient="records"),
        "segment_shapes": shape_df.to_dict(orient="records"),
    }

    followup_df.to_csv(out_dir / "city_day_future_high_gt_rot_followup_rows.csv", index=False)
    shape_df.to_csv(out_dir / "city_day_future_high_gt_rot_followup_shapes.csv", index=False)
    write_json(out_dir / "city_day_future_high_gt_rot_followup_summary.json", payload)
    (out_dir / "city_day_future_high_gt_rot_followup_summary.md").write_text(
        build_markdown(payload), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
