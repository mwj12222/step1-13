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
DEFAULT_OUT_DIR = Path(
    "docs/research/init_risk_city_day_future_high_gt_rot_boundary_neighborhood_20260320"
)

TARGET_SEQUENCES = {"city_day/0_none", "city_day/2_mid"}


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


def parallax_band(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "missing"
    v = float(v)
    if v < 12.693:
        return "lt_12p7"
    if v < 25:
        return "12p7_to_25"
    if v < 50:
        return "25_to_50"
    return "ge_50"


def future_rot_band(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "missing"
    v = float(v)
    if v < 7.5:
        return "5_to_7p5"
    if v < 10:
        return "7p5_to_10"
    if v < 15:
        return "10_to_15"
    return "ge_15"


def contiguous_segments(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame(rows)
    df = df.sort_values(["sequence", "window_id"]).copy()
    for sequence, group in df.groupby("sequence"):
        buf: list[dict] = []
        for row in group.to_dict("records"):
            if not buf:
                buf = [row]
                continue
            if int(row["window_id"]) - int(buf[-1]["window_id"]) == 10:
                buf.append(row)
            else:
                rows.append(_segment_summary(sequence, buf))
                buf = [row]
        if buf:
            rows.append(_segment_summary(sequence, buf))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["rows", "future_gt_rot_med_deg_mean"], ascending=[False, False])


def _segment_summary(sequence: str, rows: list[dict]) -> dict:
    seg = pd.DataFrame(rows)
    return {
        "sequence": sequence,
        "start_window_id": int(seg["window_id"].min()),
        "end_window_id": int(seg["window_id"].max()),
        "rows": int(len(seg)),
        "future_gt_rot_band_mode": seg["future_gt_rot_band"].mode().iloc[0],
        "parallax_band_mode": seg["parallax_band"].mode().iloc[0],
        "future_gt_rot_med_deg_mean": pd.to_numeric(seg["future_gt_rot_med_deg"], errors="coerce").mean(),
        "parallax_px_candidate_mean": pd.to_numeric(seg["parallax_px_candidate"], errors="coerce").mean(),
        "reproj_p90_px_mean": pd.to_numeric(seg["reproj_p90_px"], errors="coerce").mean(),
        "Q_post_geom_only_mean": pd.to_numeric(seg["Q_post_geom_only"], errors="coerce").mean(),
    }


def build_markdown(payload: dict) -> str:
    lines: list[str] = []
    lines.append("# city_day future_high_gt_rot boundary neighborhood")
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    for item in payload["headline"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 主导边界段")
    lines.append("")
    lines.append("| sequence | segment | rows | future_rot_band | parallax_band | mean_future_gt_rot | mean_current_parallax | mean_reproj_p90 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["segments"]:
        lines.append(
            f"| {row['sequence']} | {row['start_window_id']}-{row['end_window_id']} | {row['rows']} | {row['future_gt_rot_band_mode']} | {row['parallax_band_mode']} | {fmt(row['future_gt_rot_med_deg_mean'])} | {fmt(row['parallax_px_candidate_mean'])} | {fmt(row['reproj_p90_px_mean'])} |"
        )
    lines.append("")
    lines.append("## 段前/段内/段后画像")
    lines.append("")
    lines.append("| segment_id | relation | rows | bad_rate | mean_parallax | mean_gt_rot | mean_reproj_p90 | dominant_trigger |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["neighborhood_profiles"]:
        lines.append(
            f"| {row['segment_id']} | {row['relation']} | {row['rows']} | {fmt(row['bad_rate'])} | {fmt(row['parallax_px_candidate'])} | {fmt(row['gt_rot_med_deg'])} | {fmt(row['reproj_p90_px'])} | {row['dominant_trigger']} |"
        )
    lines.append("")
    lines.append("## 触发未来窗口自身的 trigger 分布")
    lines.append("")
    lines.append("| segment_id | future_row_trigger | rows | row_share |")
    lines.append("| --- | --- | --- | --- |")
    for row in payload["future_trigger_rows"]:
        lines.append(
            f"| {row['segment_id']} | {row['future_row_trigger']} | {row['rows']} | {fmt(row['row_share'])} |"
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
        description="Analyze city_day future_high_gt_rot both-wrong boundary neighborhoods."
    )
    parser.add_argument("--hard_cases_csv", type=Path, default=DEFAULT_HARD_CASES)
    parser.add_argument("--risk_dataset_csv", type=Path, default=DEFAULT_RISK_DATASET)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    hard = pd.read_csv(args.hard_cases_csv)
    risk = pd.read_csv(args.risk_dataset_csv)
    risk = risk[risk["sample_type"] == "step11"].copy()

    cur_cols = [
        "sample_uid",
        "sequence",
        "Y_bad_v2_min_default",
        "Y_bad_v2_min_default_trigger",
        "Y_bad_v2_min_default_trigger_window_id",
        "Y_bad_v2_min_default_horizon_windows",
        "Q_post_geom_only",
        "gt_rot_med_deg",
        "gate_post_reason",
    ]
    cur = risk[cur_cols].copy()
    future = risk[["sequence", "window_id", "gt_rot_med_deg", "Y_bad_v2_min_default_trigger", "Y_bad_v2_min_default"]].copy()
    future = future.rename(
        columns={
            "window_id": "future_window_id",
            "gt_rot_med_deg": "future_gt_rot_med_deg",
            "Y_bad_v2_min_default_trigger": "future_row_trigger",
            "Y_bad_v2_min_default": "future_row_bad",
        }
    )

    merged = hard.merge(cur, on=["sample_uid", "sequence"], how="left")
    merged = merged[
        (merged["shared_wrong"] == 1)
        & (merged["sequence"].isin(TARGET_SEQUENCES))
        & (merged["Y_bad_v2_min_default_trigger"] == "future_high_gt_rot")
        & (pd.to_numeric(merged["Y_bad_v2_min_default_horizon_windows"], errors="coerce") == 1)
    ].copy()
    merged["future_window_id"] = pd.to_numeric(
        merged["Y_bad_v2_min_default_trigger_window_id"], errors="coerce"
    )
    merged = merged.merge(future, on=["sequence", "future_window_id"], how="left")
    merged["parallax_band"] = merged["parallax_px_candidate"].apply(parallax_band)
    merged["future_gt_rot_band"] = merged["future_gt_rot_med_deg"].apply(future_rot_band)

    segments = contiguous_segments(merged).head(8).copy()
    if segments.empty:
        raise RuntimeError("No city_day future_high_gt_rot both-wrong segments found.")
    segments["segment_id"] = [
        f"{row.sequence}:{int(row.start_window_id)}-{int(row.end_window_id)}"
        for row in segments.itertuples()
    ]

    neighborhood_rows = []
    future_trigger_rows = []
    for seg in segments.to_dict(orient="records"):
        seq = seg["sequence"]
        start = int(seg["start_window_id"])
        end = int(seg["end_window_id"])
        seg_id = seg["segment_id"]

        neigh = risk[
            (risk["sequence"] == seq)
            & (pd.to_numeric(risk["window_id"], errors="coerce") >= start - 20)
            & (pd.to_numeric(risk["window_id"], errors="coerce") <= end + 20)
        ].copy()
        neigh["window_id"] = pd.to_numeric(neigh["window_id"], errors="coerce")
        neigh["relation"] = neigh["window_id"].apply(
            lambda x: "pre" if x < start else ("post" if x > end else "inside")
        )

        for relation, rel_df in neigh.groupby("relation"):
            trigger_mode = (
                rel_df["Y_bad_v2_min_default_trigger"].dropna().mode().iloc[0]
                if rel_df["Y_bad_v2_min_default_trigger"].dropna().shape[0]
                else "missing"
            )
            neighborhood_rows.append(
                {
                    "segment_id": seg_id,
                    "sequence": seq,
                    "relation": relation,
                    "rows": int(len(rel_df)),
                    "bad_rate": pd.to_numeric(rel_df["Y_bad_v2_min_default"], errors="coerce").mean(),
                    "parallax_px_candidate": pd.to_numeric(rel_df["parallax_px_candidate"], errors="coerce").mean(),
                    "gt_rot_med_deg": pd.to_numeric(rel_df["gt_rot_med_deg"], errors="coerce").mean(),
                    "reproj_p90_px": pd.to_numeric(rel_df["reproj_p90_px"], errors="coerce").mean(),
                    "Q_post_geom_only": pd.to_numeric(rel_df["Q_post_geom_only"], errors="coerce").mean(),
                    "dominant_trigger": trigger_mode,
                }
            )

        inside = merged[
            (merged["sequence"] == seq)
            & (pd.to_numeric(merged["window_id"], errors="coerce") >= start)
            & (pd.to_numeric(merged["window_id"], errors="coerce") <= end)
        ].copy()
        trigger_dist = (
            inside.groupby("future_row_trigger")
            .size()
            .rename("rows")
            .reset_index()
            .sort_values("rows", ascending=False)
        )
        total = trigger_dist["rows"].sum()
        trigger_dist["row_share"] = trigger_dist["rows"] / total if total else 0.0
        trigger_dist["segment_id"] = seg_id
        future_trigger_rows.extend(trigger_dist.to_dict(orient="records"))

    neighborhood_df = pd.DataFrame(neighborhood_rows).sort_values(["segment_id", "relation"])
    future_trigger_df = pd.DataFrame(future_trigger_rows).sort_values(["segment_id", "rows"], ascending=[True, False])

    headline = [
        "city_day 主导边界段不是孤立坏点，而是短连续段：`city_day/2_mid 440-480`、`city_day/0_none 210-240`、`city_day/0_none 440-470` 这些段都连续覆盖多个窗口。",
        "这些主导段的段内画像普遍是：中低当前视差、future_high_gt_rot 触发、几何分数仍高；更像 next-window 贴边界 bad，而不是当前窗口已经明显崩坏。",
        "未来窗口自身的 trigger 仍以 `future_high_gt_rot` 为主，说明这更像短时旋转边界带，而不是立即升级成 reset/solver_fail 的崩坏链。",
    ]

    judgement = [
        "city_day 的主导边界段表现出明显的“短连续边界带”结构：段前后并不是完全稳定/完全崩坏的二元切换，而是围绕 future_high_gt_rot 连续摆动。",
        "这支持我们把 `future_high_gt_rot@K=1` 主体问题看成 sequence-aware 标签边界，而不是简单提高统一阈值就能彻底解决。",
        "如果下一步继续审定义，更值得对 city_day 类 pattern 做局部语义收紧，而不是继续全局抬 tau 或扩大 horizon。",
    ]

    payload = {
        "headline": headline,
        "judgement": judgement,
        "segments": segments.to_dict(orient="records"),
        "neighborhood_profiles": neighborhood_df.to_dict(orient="records"),
        "future_trigger_rows": future_trigger_df.to_dict(orient="records"),
    }

    segments.to_csv(out_dir / "city_day_future_high_gt_rot_boundary_segments.csv", index=False)
    neighborhood_df.to_csv(out_dir / "city_day_future_high_gt_rot_boundary_neighborhood.csv", index=False)
    future_trigger_df.to_csv(out_dir / "city_day_future_high_gt_rot_boundary_future_trigger.csv", index=False)
    write_json(out_dir / "city_day_future_high_gt_rot_boundary_neighborhood_summary.json", payload)
    (out_dir / "city_day_future_high_gt_rot_boundary_neighborhood_summary.md").write_text(
        build_markdown(payload), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
