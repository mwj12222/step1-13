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
    "docs/research/init_risk_future_high_gt_rot_k1_both_wrong_segments_20260320"
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


def current_rot_band(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "missing"
    v = float(v)
    if v < 2:
        return "lt_2"
    if v < 5:
        return "2_to_5"
    if v < 10:
        return "5_to_10"
    return "ge_10"


def build_markdown(payload: dict) -> str:
    lines: list[str] = []
    lines.append("# future_high_gt_rot@K=1 both-wrong finer segments")
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    for item in payload["headline"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 主导 band")
    lines.append("")
    lines.append("| view | band | rows | row_share |")
    lines.append("| --- | --- | --- | --- |")
    for row in payload["dominant_bands"]:
        lines.append(
            f"| {row['view']} | {row['band']} | {row['rows']} | {fmt(row['row_share'])} |"
        )
    lines.append("")
    lines.append("## 未来旋转带 x 当前视差带")
    lines.append("")
    lines.append("| future_rot_band | parallax_band | rows | row_share | mean_future_gt_rot | mean_current_parallax | mean_q_geom |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["cross_top"]:
        lines.append(
            f"| {row['future_gt_rot_band']} | {row['parallax_band']} | {row['rows']} | {fmt(row['row_share'])} | {fmt(row['future_gt_rot_med_deg'])} | {fmt(row['parallax_px_candidate'])} | {fmt(row['Q_post_geom_only'])} |"
        )
    lines.append("")
    lines.append("## 序列 x band 主导项")
    lines.append("")
    lines.append("| sequence | future_rot_band | parallax_band | rows | row_share_within_sequence |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in payload["sequence_band_top"]:
        lines.append(
            f"| {row['sequence']} | {row['future_gt_rot_band']} | {row['parallax_band']} | {row['rows']} | {fmt(row['row_share_within_sequence'])} |"
        )
    lines.append("")
    lines.append("## 典型连续段")
    lines.append("")
    lines.append("| sequence | segment | rows | future_rot_band | parallax_band | mean_future_gt_rot | mean_current_parallax |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["top_segments"]:
        lines.append(
            f"| {row['sequence']} | {row['start_window_id']}-{row['end_window_id']} | {row['rows']} | {row['future_gt_rot_band_mode']} | {row['parallax_band_mode']} | {fmt(row['future_gt_rot_med_deg_mean'])} | {fmt(row['parallax_px_candidate_mean'])} |"
        )
    lines.append("")
    lines.append("## 判断")
    lines.append("")
    for item in payload["judgement"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


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
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze finer segments of future_high_gt_rot@K=1 both-wrong samples."
    )
    parser.add_argument("--hard_cases_csv", type=Path, default=DEFAULT_HARD_CASES)
    parser.add_argument("--risk_dataset_csv", type=Path, default=DEFAULT_RISK_DATASET)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    hard = pd.read_csv(args.hard_cases_csv)
    risk = pd.read_csv(args.risk_dataset_csv)

    cur_cols = [
        "sample_uid",
        "sequence",
        "Q_post_geom_only",
        "Y_bad_v2_min_default",
        "Y_bad_v2_min_default_trigger",
        "Y_bad_v2_min_default_trigger_window_id",
        "Y_bad_v2_min_default_horizon_windows",
        "gt_rot_med_deg",
    ]
    cur = risk[cur_cols].copy()
    future = risk[["sequence", "window_id", "gt_rot_med_deg", "Q_post_geom_only"]].copy()
    future["window_id"] = pd.to_numeric(future["window_id"], errors="coerce")
    future = (
        future.sort_values(["sequence", "window_id", "gt_rot_med_deg"], ascending=[True, True, False])
        .groupby(["sequence", "window_id"], as_index=False)
        .agg(
            gt_rot_med_deg=("gt_rot_med_deg", lambda s: pd.to_numeric(s, errors="coerce").dropna().iloc[0] if pd.to_numeric(s, errors="coerce").dropna().shape[0] else pd.NA),
            Q_post_geom_only=("Q_post_geom_only", lambda s: pd.to_numeric(s, errors="coerce").dropna().iloc[0] if pd.to_numeric(s, errors="coerce").dropna().shape[0] else pd.NA),
        )
    )
    future = future.rename(
        columns={
            "window_id": "future_window_id",
            "gt_rot_med_deg": "future_gt_rot_med_deg",
            "Q_post_geom_only": "future_Q_post_geom_only",
        }
    )

    merged = hard.merge(cur, on=["sample_uid", "sequence"], how="left")
    merged = merged[
        (merged["shared_wrong"] == 1)
        & (merged["Y_bad_v2_min_default_trigger"] == "future_high_gt_rot")
        & (pd.to_numeric(merged["Y_bad_v2_min_default_horizon_windows"], errors="coerce") == 1)
    ].copy()
    merged["future_window_id"] = pd.to_numeric(
        merged["Y_bad_v2_min_default_trigger_window_id"], errors="coerce"
    )
    merged = merged.merge(future, on=["sequence", "future_window_id"], how="left")

    merged["parallax_band"] = merged["parallax_px_candidate"].apply(parallax_band)
    merged["future_gt_rot_band"] = merged["future_gt_rot_med_deg"].apply(future_rot_band)
    merged["current_gt_rot_band"] = merged["gt_rot_med_deg"].apply(current_rot_band)

    total_rows = len(merged)

    future_band = (
        merged.groupby("future_gt_rot_band")
        .size()
        .rename("rows")
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    future_band["row_share"] = future_band["rows"] / total_rows

    parallax_band_df = (
        merged.groupby("parallax_band")
        .size()
        .rename("rows")
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    parallax_band_df["row_share"] = parallax_band_df["rows"] / total_rows

    cross = (
        merged.groupby(["future_gt_rot_band", "parallax_band"], dropna=False)
        .agg(
            rows=("sample_uid", "size"),
            future_gt_rot_med_deg=("future_gt_rot_med_deg", "mean"),
            parallax_px_candidate=("parallax_px_candidate", "mean"),
            Q_post_geom_only=("Q_post_geom_only", "mean"),
            reproj_p90_px=("reproj_p90_px", "mean"),
            tri_points_candidate=("tri_points_candidate", "mean"),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    cross["row_share"] = cross["rows"] / total_rows

    seq_band = (
        merged.groupby(["sequence", "future_gt_rot_band", "parallax_band"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    seq_total = merged.groupby("sequence").size().rename("sequence_rows").reset_index()
    seq_band = seq_band.merge(seq_total, on="sequence", how="left")
    seq_band["row_share_within_sequence"] = seq_band["rows"] / seq_band["sequence_rows"]
    seq_band = seq_band.sort_values(["sequence", "rows"], ascending=[True, False])
    seq_band_top = seq_band.groupby("sequence").head(3).reset_index(drop=True)

    segment_df = contiguous_segments(merged).head(15)

    dominant_bands = [
        {
            "view": "future_gt_rot",
            "band": future_band.iloc[0]["future_gt_rot_band"],
            "rows": int(future_band.iloc[0]["rows"]),
            "row_share": float(future_band.iloc[0]["row_share"]),
        },
        {
            "view": "current_parallax",
            "band": parallax_band_df.iloc[0]["parallax_band"],
            "rows": int(parallax_band_df.iloc[0]["rows"]),
            "row_share": float(parallax_band_df.iloc[0]["row_share"]),
        },
    ]

    headline = [
        f"`future_high_gt_rot@K=1` 的 `both-wrong` 细分后仍高度集中：总共 {total_rows} 条样本里，主导未来旋转带是 `{future_band.iloc[0]['future_gt_rot_band']}`（{int(future_band.iloc[0]['rows'])} 条，{fmt(future_band.iloc[0]['row_share'])}），主导当前视差带是 `{parallax_band_df.iloc[0]['parallax_band']}`（{int(parallax_band_df.iloc[0]['rows'])} 条，{fmt(parallax_band_df.iloc[0]['row_share'])}）。",
        f"最密集的联合区域不是极端高旋转，而是 `{cross.iloc[0]['future_gt_rot_band']} x {cross.iloc[0]['parallax_band']}`，共有 {int(cross.iloc[0]['rows'])} 条，平均未来旋转 {fmt(cross.iloc[0]['future_gt_rot_med_deg'])}、当前视差 {fmt(cross.iloc[0]['parallax_px_candidate'])}。",
        f"序列层面，`city_day/0_none` 和 `city_day/2_mid` 仍是主体，而且主导段都落在中低视差、轻中度 future rotation 区间；`city_night` 的困难段则相对更偏高旋转高视差。",
    ]

    judgement = [
        "这批 `both-wrong` 主体并不像“极端高旋转崩坏”，而更像轻到中度 future rotation 触发的边界型正样本，尤其集中在 city_day 端点。",
        "序列差异很明显：city_day 主导段更偏中低视差 + 轻中度 future rotation，city_night/parking_lot 则更偏高视差高旋转；这说明同一个 `future_high_gt_rot` 定义在不同 sequence 上语义并不完全一致。",
        "因此后续如果继续审定义，更值得优先看 `future_gt_rot` 的分带语义和 sequence-specific pattern，而不是只继续整体抬阈值。",
    ]

    payload = {
        "summary": {
            "rows": int(total_rows),
            "future_high_gt_rot_k1_both_wrong_rows": int(total_rows),
        },
        "headline": headline,
        "judgement": judgement,
        "dominant_bands": dominant_bands,
        "cross_top": cross.head(12).to_dict(orient="records"),
        "sequence_band_top": seq_band_top.to_dict(orient="records"),
        "top_segments": segment_df.to_dict(orient="records"),
    }

    future_band.to_csv(out_dir / "future_high_gt_rot_both_wrong_future_rot_bands.csv", index=False)
    parallax_band_df.to_csv(out_dir / "future_high_gt_rot_both_wrong_parallax_bands.csv", index=False)
    cross.to_csv(out_dir / "future_high_gt_rot_both_wrong_cross_bands.csv", index=False)
    seq_band_top.to_csv(out_dir / "future_high_gt_rot_both_wrong_sequence_band_top.csv", index=False)
    segment_df.to_csv(out_dir / "future_high_gt_rot_both_wrong_top_segments.csv", index=False)
    write_json(out_dir / "future_high_gt_rot_both_wrong_segments_summary.json", payload)
    (out_dir / "future_high_gt_rot_both_wrong_segments_summary.md").write_text(
        build_markdown(payload), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
