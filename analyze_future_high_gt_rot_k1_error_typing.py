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
DEFAULT_CITY_DAY_SEGMENTS = Path(
    "docs/research/init_risk_city_day_future_high_gt_rot_boundary_neighborhood_20260320/city_day_future_high_gt_rot_boundary_segments.csv"
)
DEFAULT_CITY_DAY_SHAPES = Path(
    "docs/research/init_risk_city_day_future_high_gt_rot_followup_20260320/city_day_future_high_gt_rot_followup_shapes.csv"
)
DEFAULT_OUT_DIR = Path(
    "docs/research/init_risk_future_high_gt_rot_k1_error_typing_20260320"
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


def classify_rows(df: pd.DataFrame) -> pd.Series:
    # Label-boundary dominant:
    # light/mid future rotation + weak current parallax + clean current geometry +
    # stable return in +1/+2 and no severe upgrade. City-day boundary shapes get
    # an extra allowance because they were explicitly audited as boundary bands.
    boundary_base = (
        (pd.to_numeric(df["future_gt_rot_med_deg"], errors="coerce") < 10.0)
        & (pd.to_numeric(df["parallax_px_candidate"], errors="coerce") < 25.0)
        & (pd.to_numeric(df["reproj_p90_px"], errors="coerce") < 2.0)
        & df["has_stable_followup"]
        & (~df["has_severe_followup"])
    )
    boundary_city_day = (
        df["sequence"].isin(["city_day/0_none", "city_day/2_mid"])
        & df["shape"].isin(["short_boundary_jitter", "extended_rot_boundary", "mixed_followup"])
        & df["has_stable_followup"]
        & (~df["has_severe_followup"])
    )
    boundary = boundary_base | boundary_city_day

    # Recoverability-gap dominant:
    # stronger future rotation/high motion observability cases that look less like
    # boundary jitter and more like a missing recoverability cue.
    recover = (
        (
            (pd.to_numeric(df["future_gt_rot_med_deg"], errors="coerce") >= 10.0)
            & (pd.to_numeric(df["parallax_px_candidate"], errors="coerce") >= 25.0)
            & (pd.to_numeric(df["reproj_p90_px"], errors="coerce") < 2.0)
        )
        | (
            (pd.to_numeric(df["parallax_px_candidate"], errors="coerce") >= 50.0)
            & (pd.to_numeric(df["future_gt_rot_med_deg"], errors="coerce") >= 7.5)
        )
    ) & (~boundary)

    out = pd.Series("unresolved_mixed", index=df.index, dtype="object")
    out.loc[recover] = "recoverability_gap_dominant"
    out.loc[boundary] = "label_boundary_dominant"
    return out


def assign_city_day_segment_id(df: pd.DataFrame, seg_df: pd.DataFrame) -> list[str | None]:
    segment_ids: list[str | None] = []
    for row in df[["sequence", "window_id"]].to_dict(orient="records"):
        match = seg_df[
            (seg_df["sequence"] == row["sequence"])
            & (pd.to_numeric(seg_df["start_window_id"], errors="coerce") <= row["window_id"])
            & (pd.to_numeric(seg_df["end_window_id"], errors="coerce") >= row["window_id"])
        ]
        segment_ids.append(match["segment_id"].iloc[0] if len(match) else None)
    return segment_ids


def build_markdown(payload: dict) -> str:
    lines: list[str] = []
    lines.append("# future_high_gt_rot@K=1 shared wrong typing")
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    for item in payload["headline"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 分型规模")
    lines.append("")
    lines.append("| class | rows | row_share |")
    lines.append("| --- | --- | --- |")
    for row in payload["class_counts"]:
        lines.append(f"| {row['class']} | {row['rows']} | {fmt(row['row_share'])} |")
    lines.append("")
    lines.append("## 分型画像")
    lines.append("")
    lines.append("| class | rows | mean future_gt_rot | mean parallax | mean reproj_p90 | mean tri_points | stable_followup_rate | severe_followup_rate | city_day_rate |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["class_profiles"]:
        lines.append(
            f"| {row['class']} | {row['rows']} | {fmt(row['future_gt_rot_med_deg'])} | {fmt(row['parallax_px_candidate'])} | {fmt(row['reproj_p90_px'])} | {fmt(row['tri_points_candidate'])} | {fmt(row['stable_followup_rate'])} | {fmt(row['severe_followup_rate'])} | {fmt(row['city_day_rate'])} |"
        )
    lines.append("")
    lines.append("## 按 sequence x class")
    lines.append("")
    lines.append("| sequence | label_boundary_dominant | recoverability_gap_dominant | unresolved_mixed |")
    lines.append("| --- | --- | --- | --- |")
    for row in payload["sequence_breakdown"]:
        lines.append(
            f"| {row['sequence']} | {row.get('label_boundary_dominant', 0)} | {row.get('recoverability_gap_dominant', 0)} | {row.get('unresolved_mixed', 0)} |"
        )
    lines.append("")
    lines.append("## 典型样本")
    lines.append("")
    lines.append("| class | sequence | window_id | future_gt_rot | parallax | trigger_p1 | trigger_p2 | shape |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["example_rows"]:
        lines.append(
            f"| {row['class']} | {row['sequence']} | {row['window_id']} | {fmt(row['future_gt_rot_med_deg'])} | {fmt(row['parallax_px_candidate'])} | {row['trigger_p1']} | {row['trigger_p2']} | {row.get('shape', '-') if pd.notna(row.get('shape')) else '-'} |"
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
        description="Type future_high_gt_rot@K=1 shared hard cases into label-boundary vs recoverability-gap classes."
    )
    parser.add_argument("--hard_cases_csv", type=Path, default=DEFAULT_HARD_CASES)
    parser.add_argument("--risk_dataset_csv", type=Path, default=DEFAULT_RISK_DATASET)
    parser.add_argument("--city_day_segments_csv", type=Path, default=DEFAULT_CITY_DAY_SEGMENTS)
    parser.add_argument("--city_day_shapes_csv", type=Path, default=DEFAULT_CITY_DAY_SHAPES)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    hard = pd.read_csv(args.hard_cases_csv)
    risk = pd.read_csv(args.risk_dataset_csv)
    seg_df = pd.read_csv(args.city_day_segments_csv)
    shape_df = pd.read_csv(args.city_day_shapes_csv)

    current = risk[
        [
            "sample_uid",
            "sequence",
            "window_id",
            "Q_post_geom_only",
            "Y_bad_v2_min_default_trigger",
            "Y_bad_v2_min_default_trigger_window_id",
            "Y_bad_v2_min_default_horizon_windows",
            "gt_rot_med_deg",
        ]
    ].copy()

    rows = hard[hard["shared_wrong"] == 1].merge(
        current, on=["sample_uid", "sequence", "window_id"], how="left", suffixes=("", "_current")
    )
    rows = rows[
        (rows["Y_bad_v2_min_default_trigger"] == "future_high_gt_rot")
        & (pd.to_numeric(rows["Y_bad_v2_min_default_horizon_windows"], errors="coerce") == 1)
    ].copy()

    future = risk[
        [
            "sequence",
            "window_id",
            "gt_rot_med_deg",
            "Y_bad_v2_min_default",
            "Y_bad_v2_min_default_trigger",
        ]
    ].copy()
    future["window_id"] = pd.to_numeric(future["window_id"], errors="coerce")
    future = future.groupby(["sequence", "window_id"], as_index=False).first().rename(
        columns={
            "gt_rot_med_deg": "future_gt_rot_med_deg",
            "Y_bad_v2_min_default": "future_row_bad",
            "Y_bad_v2_min_default_trigger": "future_row_trigger",
        }
    )

    rows["future_window_id"] = pd.to_numeric(rows["Y_bad_v2_min_default_trigger_window_id"], errors="coerce")
    rows = rows.merge(
        future,
        left_on=["sequence", "future_window_id"],
        right_on=["sequence", "window_id"],
        how="left",
        suffixes=("", "_future"),
    )

    for offset in [1, 2]:
        probe = rows[["sequence", "future_window_id"]].copy()
        probe["probe_window_id"] = probe["future_window_id"] + 10 * offset
        probe_join = probe.merge(
            future.rename(
                columns={
                    "window_id": "probe_window_id",
                    "future_row_trigger": f"trigger_p{offset}",
                    "future_gt_rot_med_deg": f"future_rot_p{offset}",
                    "future_row_bad": f"bad_p{offset}",
                }
            ),
            on=["sequence", "probe_window_id"],
            how="left",
        )
        rows[f"trigger_p{offset}"] = probe_join[f"trigger_p{offset}"]
        rows[f"future_rot_p{offset}"] = probe_join[f"future_rot_p{offset}"]
        rows[f"bad_p{offset}"] = probe_join[f"bad_p{offset}"]

    rows["has_stable_followup"] = rows[["trigger_p1", "trigger_p2"]].isin(["stable_horizon"]).any(axis=1)
    rows["has_severe_followup"] = rows[["trigger_p1", "trigger_p2"]].isin(
        ["future_solver_fail", "future_reset"]
    ).any(axis=1)
    rows["segment_id"] = assign_city_day_segment_id(rows, seg_df)
    rows = rows.merge(shape_df[["segment_id", "shape"]], on="segment_id", how="left")
    rows["class"] = classify_rows(rows)
    rows["city_day_flag"] = rows["sequence"].isin(["city_day/0_none", "city_day/2_mid"]).astype(int)

    class_counts = (
        rows.groupby("class").size().rename("rows").reset_index().sort_values("rows", ascending=False)
    )
    class_counts["row_share"] = class_counts["rows"] / len(rows)

    profile = (
        rows.groupby("class")
        .agg(
            rows=("sample_uid", "size"),
            future_gt_rot_med_deg=("future_gt_rot_med_deg", "mean"),
            parallax_px_candidate=("parallax_px_candidate", "mean"),
            reproj_p90_px=("reproj_p90_px", "mean"),
            tri_points_candidate=("tri_points_candidate", "mean"),
            stable_followup_rate=("has_stable_followup", "mean"),
            severe_followup_rate=("has_severe_followup", "mean"),
            city_day_rate=("city_day_flag", "mean"),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )

    seq_break = (
        rows.groupby(["sequence", "class"]).size().unstack(fill_value=0).reset_index().sort_values("sequence")
    )

    example_rows = (
        rows.sort_values(["class", "sequence", "window_id"])
        .groupby("class", as_index=False)
        .head(6)[
            [
                "class",
                "sequence",
                "window_id",
                "future_gt_rot_med_deg",
                "parallax_px_candidate",
                "trigger_p1",
                "trigger_p2",
                "shape",
            ]
        ]
        .to_dict(orient="records")
    )

    label_rows = rows[rows["class"] == "label_boundary_dominant"]
    recover_rows = rows[rows["class"] == "recoverability_gap_dominant"]
    unresolved_rows = rows[rows["class"] == "unresolved_mixed"]

    headline = [
        f"这批 `future_high_gt_rot@K=1` 的 `both-wrong` 并不是单一来源：当前可较稳地分成 `label_boundary_dominant={len(label_rows)}`、`recoverability_gap_dominant={len(recover_rows)}`，仍有 `unresolved_mixed={len(unresolved_rows)}` 条暂时不能强行归类。",
        f"`label_boundary_dominant` 主体更像轻中度 future rotation + 偏低当前视差 + 后续可回到 `stable_horizon` 的边界正样本，city_day 占比达到 {fmt(label_rows['city_day_flag'].mean() if len(label_rows) else 0)}。",
        f"`recoverability_gap_dominant` 则更偏高 future rotation / 高 parallax 子集，平均 `future_gt_rot={fmt(recover_rows['future_gt_rot_med_deg'].mean() if len(recover_rows) else None)}`、`parallax={fmt(recover_rows['parallax_px_candidate'].mean() if len(recover_rows) else None)}`，更像当前缺 recoverability 锚点的样本。",
    ]

    judgement = [
        "当前主导部分仍然更像标签边界问题，而不是统一的几何失败：因为被归为 label-boundary 的样本通常具备较高几何质量、较低 reprojection、并且在后续 1-2 个窗口中出现 stable_horizon 回返。",
        "recoverability 缺口不是不存在，但它更像次级子集问题：主要集中在高视差、高旋转的 shared wrong 上，这类样本更可能受益于额外的可恢复性信号。",
        "因此下一步如果继续推进主线，更值钱的是优先审 label-boundary 主导部分，而不是急着把整个 `future_high_gt_rot@K=1` 统一改写。",
    ]

    payload = {
        "headline": headline,
        "judgement": judgement,
        "summary": {
            "total_rows": int(len(rows)),
            "label_boundary_rows": int(len(label_rows)),
            "recoverability_gap_rows": int(len(recover_rows)),
            "unresolved_rows": int(len(unresolved_rows)),
        },
        "class_counts": class_counts.to_dict(orient="records"),
        "class_profiles": profile.to_dict(orient="records"),
        "sequence_breakdown": seq_break.to_dict(orient="records"),
        "example_rows": example_rows,
    }

    rows.to_csv(out_dir / "future_high_gt_rot_k1_typed_rows.csv", index=False)
    class_counts.to_csv(out_dir / "future_high_gt_rot_k1_class_counts.csv", index=False)
    profile.to_csv(out_dir / "future_high_gt_rot_k1_class_profiles.csv", index=False)
    seq_break.to_csv(out_dir / "future_high_gt_rot_k1_sequence_breakdown.csv", index=False)
    write_json(out_dir / "future_high_gt_rot_k1_typing_summary.json", payload)
    (out_dir / "future_high_gt_rot_k1_typing_summary.md").write_text(
        build_markdown(payload), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
