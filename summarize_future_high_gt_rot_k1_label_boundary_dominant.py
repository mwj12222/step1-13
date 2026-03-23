#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_TYPED_ROWS = Path(
    "docs/research/init_risk_future_high_gt_rot_k1_error_typing_20260320/future_high_gt_rot_k1_typed_rows.csv"
)
DEFAULT_OUT_DIR = Path(
    "docs/research/init_risk_future_high_gt_rot_k1_label_boundary_dominant_20260320"
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


def future_rot_band(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "missing"
    v = float(v)
    if v < 7.5:
        return "5_to_7p5"
    if v < 10.0:
        return "7p5_to_10"
    if v < 15.0:
        return "10_to_15"
    return "ge_15"


def followup_pattern(row: pd.Series) -> str:
    p1 = row.get("trigger_p1")
    p2 = row.get("trigger_p2")
    if p1 == "stable_horizon" and p2 == "stable_horizon":
        return "stable_then_stable"
    if p1 == "stable_horizon" and p2 == "future_high_gt_rot":
        return "stable_then_rot"
    if p1 == "future_high_gt_rot" and p2 == "stable_horizon":
        return "rot_then_stable"
    if p1 == "future_high_gt_rot" and p2 == "future_high_gt_rot":
        return "rot_then_rot"
    if pd.isna(p1) or pd.isna(p2):
        return "missing_followup"
    return "other_followup"


def build_markdown(payload: dict) -> str:
    lines: list[str] = []
    lines.append("# future_high_gt_rot@K=1 label-boundary dominant finer summary")
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    for item in payload["headline"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 按 sequence")
    lines.append("")
    lines.append("| sequence | rows | row_share | mean future_gt_rot | mean parallax | stable_p1_rate | stable_any_followup_rate |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["sequence_summary"]:
        lines.append(
            f"| {row['sequence']} | {row['rows']} | {fmt(row['row_share'])} | {fmt(row['future_gt_rot_med_deg'])} | {fmt(row['parallax_px_candidate'])} | {fmt(row['stable_p1_rate'])} | {fmt(row['stable_any_followup_rate'])} |"
        )
    lines.append("")
    lines.append("## future rotation band x follow-up motif")
    lines.append("")
    lines.append("| future_rot_band | followup_pattern | rows | row_share | mean parallax |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in payload["cross_summary"]:
        lines.append(
            f"| {row['future_rot_band']} | {row['followup_pattern']} | {row['rows']} | {fmt(row['row_share'])} | {fmt(row['parallax_px_candidate'])} |"
        )
    lines.append("")
    lines.append("## city_day 主导段")
    lines.append("")
    lines.append("| segment_id | sequence | shape | rows | mean future_gt_rot | mean parallax | followup_pattern_mode |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["city_day_segment_summary"]:
        lines.append(
            f"| {row['segment_id']} | {row['sequence']} | {row['shape']} | {row['rows']} | {fmt(row['future_gt_rot_med_deg'])} | {fmt(row['parallax_px_candidate'])} | {row['followup_pattern_mode']} |"
        )
    lines.append("")
    lines.append("## 当前更克制的局部语义规律")
    lines.append("")
    for item in payload["local_rule"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 判断")
    lines.append("")
    for item in payload["judgement"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the finer structure of label-boundary dominant future_high_gt_rot@K=1 shared wrong samples."
    )
    parser.add_argument("--typed_rows_csv", type=Path, default=DEFAULT_TYPED_ROWS)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.typed_rows_csv)
    df = df[df["class"] == "label_boundary_dominant"].copy()
    df["future_rot_band"] = df["future_gt_rot_med_deg"].apply(future_rot_band)
    df["followup_pattern"] = df.apply(followup_pattern, axis=1)
    df["stable_p1_flag"] = (df["trigger_p1"] == "stable_horizon").astype(int)
    df["stable_any_followup_flag"] = df[["trigger_p1", "trigger_p2"]].isin(["stable_horizon"]).any(axis=1).astype(int)

    total_rows = len(df)

    seq = (
        df.groupby("sequence")
        .agg(
            rows=("sample_uid", "size"),
            future_gt_rot_med_deg=("future_gt_rot_med_deg", "mean"),
            parallax_px_candidate=("parallax_px_candidate", "mean"),
            stable_p1_rate=("stable_p1_flag", "mean"),
            stable_any_followup_rate=("stable_any_followup_flag", "mean"),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    seq["row_share"] = seq["rows"] / total_rows

    cross = (
        df.groupby(["future_rot_band", "followup_pattern"])
        .agg(
            rows=("sample_uid", "size"),
            parallax_px_candidate=("parallax_px_candidate", "mean"),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    cross["row_share"] = cross["rows"] / total_rows

    city_day = df[df["sequence"].str.startswith("city_day/")].copy()
    city_seg = (
        city_day.groupby(["segment_id", "sequence", "shape"], dropna=False)
        .agg(
            rows=("sample_uid", "size"),
            future_gt_rot_med_deg=("future_gt_rot_med_deg", "mean"),
            parallax_px_candidate=("parallax_px_candidate", "mean"),
            followup_pattern_mode=("followup_pattern", lambda s: s.mode().iloc[0] if len(s.mode()) else "NA"),
        )
        .reset_index()
        .sort_values(["rows", "future_gt_rot_med_deg"], ascending=[False, False])
    )

    seq_top = seq.iloc[0]["sequence"] if len(seq) else "-"
    cross_top = cross.iloc[0] if len(cross) else None
    city_day_seq = seq[seq["sequence"].str.startswith("city_day/")].copy()
    city_day_seq_names = city_day_seq["sequence"].tolist()
    city_day_seq_label = " + ".join(city_day_seq_names) if city_day_seq_names else "city_day"
    city_day_seq_rows = int(city_day["sample_uid"].shape[0]) if len(city_day) else 0

    headline = [
        f"这 `{total_rows}` 条 `label_boundary_dominant` 样本里，主体高度集中在 `{city_day_seq_label}`，合计 {city_day_seq_rows} 条，占比 {fmt(city_day_seq_rows / total_rows if total_rows else 0)}。",
        f"最主导的 follow-up 形态不是持续坏链，而是 `stable_then_stable` 与 `stable_then_rot / rot_then_stable` 这类边界摆动；其中最大单格是 `{cross_top['future_rot_band']} x {cross_top['followup_pattern']}`，共有 {int(cross_top['rows'])} 条。" if cross_top is not None else "当前没有可汇总样本。",
        f"因此当前更稳的局部语义规律已经比“统一抬阈值”更清楚：主体是 `{seq_top}` 这类 sequence 上、轻中度 future rotation、后续可回到 stable_horizon 的短时边界带，而不是一条统一的显性失稳链。",
    ]

    local_rule = [
        "如果一批 `future_high_gt_rot@K=1` 的 shared wrong 同时满足：future rotation 主要落在 `5_to_7p5` 或 `7p5_to_10`，当前 parallax 偏低到中等，且 `+1/+2` 窗口中至少一次回到 `stable_horizon`，那么它更像局部边界带，而不是统一意义上的强 bad。",
        "这条规律目前最适合被理解为 `city_day` 主导 pattern 下的 sequence-aware 语义提醒，而不是新的训练级标签定义。",
    ]

    judgement = [
        "这批边界主导样本内部已经出现相对稳定的局部规律：sequence 上以 city_day 为主，future rotation 上以轻中度为主，follow-up 上以 stable 回返或 stable/rot 摆动为主。",
        "它支持我们继续把 `future_high_gt_rot@K=1` 的主体问题看成标签边界，而不是直接看成 recoverability 缺口。",
        "但这条规律仍然应该停留在解释与定义审计层，当前还不足以直接进入训练级 label variant。",
    ]

    payload = {
        "headline": headline,
        "sequence_summary": seq.to_dict(orient="records"),
        "cross_summary": cross.to_dict(orient="records"),
        "city_day_segment_summary": city_seg.to_dict(orient="records"),
        "local_rule": local_rule,
        "judgement": judgement,
    }

    seq.to_csv(out_dir / "label_boundary_dominant_sequence_summary.csv", index=False)
    cross.to_csv(out_dir / "label_boundary_dominant_cross_summary.csv", index=False)
    city_seg.to_csv(out_dir / "label_boundary_dominant_city_day_segment_summary.csv", index=False)
    write_json(out_dir / "label_boundary_dominant_summary.json", payload)
    (out_dir / "label_boundary_dominant_summary.md").write_text(
        build_markdown(payload), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
