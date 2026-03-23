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
    "docs/research/init_risk_post_v2_future_high_gt_rot_shared_20260319"
)

USECOLS = [
    "sample_uid",
    "sequence",
    "window_id",
    "Y_bad_v2_min_default",
    "Y_bad_v2_min_default_trigger",
    "Y_bad_v2_min_default_trigger_window_id",
    "Y_bad_v2_min_default_horizon_windows",
    "Q_post",
    "Q_post_geom_only",
    "gt_rot_med_deg",
    "reproj_p90_px",
    "parallax_px_candidate",
    "tri_points_candidate",
    "front_p_static",
    "front_coverage_ratio",
]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fmt(v, nd: int = 4) -> str:
    if pd.isna(v):
        return "-"
    return f"{float(v):.{nd}f}"


def build_markdown(payload: dict) -> str:
    lines = []
    lines.append("# future_high_gt_rot shared hard cases")
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    for item in payload["headline"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 按 sequence 汇总")
    lines.append("")
    lines.append("| sequence | future_high_gt_rot rows | shared wrong | gated fixes geo | geo beats gated | shared wrong rate |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in payload["sequence_summary"]:
        lines.append(
            f"| {row['sequence']} | {row['future_high_gt_rot_rows']} | {row['shared_wrong_count']} | {row['gated_fixes_geo_count']} | {row['geo_beats_gated_count']} | {fmt(row['shared_wrong_rate'])} |"
        )
    lines.append("")
    lines.append("## 画像对比")
    lines.append("")
    lines.append("| profile | rows | mean gt_rot_deg | mean parallax | mean reproj_p90 | mean tri_points | mean Q_post_geom_only |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["profile_summary"]:
        lines.append(
            f"| {row['profile']} | {row['rows']} | {fmt(row['gt_rot_med_deg'])} | {fmt(row['parallax_px_candidate'])} | {fmt(row['reproj_p90_px'])} | {fmt(row['tri_points_candidate'])} | {fmt(row['Q_post_geom_only'])} |"
        )
    lines.append("")
    lines.append("## 典型共享难段")
    lines.append("")
    lines.append("| sequence | segment | rows | mean parallax | mean gt_rot_deg | mean Q_post_geom_only |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in payload["top_segments"]:
        lines.append(
            f"| {row['sequence']} | {row['start_window_id']}-{row['end_window_id']} | {row['num_rows']} | {fmt(row['parallax_px_candidate_mean'])} | {fmt(row['gt_rot_med_deg_mean'])} | {fmt(row['Q_post_geom_only_mean'])} |"
        )
    lines.append("")
    lines.append("## 判断")
    lines.append("")
    for item in payload["judgement"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def contiguous_segments(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    df = df.sort_values(["sequence", "window_id"]).copy()
    out = []
    for sequence, g in df.groupby("sequence"):
        current = []
        for row in g.to_dict("records"):
            if not current:
                current = [row]
                continue
            if row["window_id"] - current[-1]["window_id"] == 10:
                current.append(row)
            else:
                out.append(current)
                current = [row]
        if current:
            out.append(current)
    rows = []
    for seg in out:
        rows.append(
            {
                "sequence": seg[0]["sequence"],
                "start_window_id": int(seg[0]["window_id"]),
                "end_window_id": int(seg[-1]["window_id"]),
                "num_rows": len(seg),
                "parallax_px_candidate_mean": pd.to_numeric(pd.Series([r["parallax_px_candidate"] for r in seg]), errors="coerce").mean(),
                "gt_rot_med_deg_mean": pd.to_numeric(pd.Series([r["gt_rot_med_deg"] for r in seg]), errors="coerce").mean(),
                "Q_post_geom_only_mean": pd.to_numeric(pd.Series([r["Q_post_geom_only"] for r in seg]), errors="coerce").mean(),
            }
        )
    rows.sort(key=lambda x: (x["num_rows"], x["gt_rot_med_deg_mean"]), reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze future_high_gt_rot shared hard cases for post_v2.")
    parser.add_argument("--hard_cases_csv", type=Path, default=DEFAULT_HARD_CASES)
    parser.add_argument("--risk_dataset_csv", type=Path, default=DEFAULT_RISK_DATASET)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    hard = pd.read_csv(args.hard_cases_csv)
    risk = pd.read_csv(args.risk_dataset_csv, usecols=USECOLS)
    merged = hard.merge(risk, on=["sample_uid", "sequence"], how="left", suffixes=("", "_risk"))
    merged = merged[merged["Y_bad_v2_min_default_trigger"] == "future_high_gt_rot"].copy()
    merged["trigger_gap"] = (
        pd.to_numeric(merged["Y_bad_v2_min_default_trigger_window_id"], errors="coerce")
        - pd.to_numeric(merged["window_id_risk"], errors="coerce")
    )

    shared = merged[merged["shared_wrong"] == 1].copy()
    gated_fix = merged[merged["gated_fixes_geo"] == 1].copy()
    geo_beats = merged[merged["geo_beats_gated"] == 1].copy()

    seq_total = merged.groupby("sequence").size().rename("future_high_gt_rot_rows")
    seq_shared = shared.groupby("sequence").size().rename("shared_wrong_count")
    seq_fix = gated_fix.groupby("sequence").size().rename("gated_fixes_geo_count")
    seq_geo = geo_beats.groupby("sequence").size().rename("geo_beats_gated_count")
    seq = pd.concat([seq_total, seq_shared, seq_fix, seq_geo], axis=1).fillna(0).reset_index()
    for col in ["future_high_gt_rot_rows", "shared_wrong_count", "gated_fixes_geo_count", "geo_beats_gated_count"]:
        seq[col] = seq[col].astype(int)
    seq["shared_wrong_rate"] = seq["shared_wrong_count"] / seq["future_high_gt_rot_rows"]
    seq = seq.sort_values(["shared_wrong_rate", "shared_wrong_count"], ascending=[False, False])

    profile_rows = []
    for name, df in [
        ("all_future_high_gt_rot", merged),
        ("shared_wrong", shared),
        ("gated_fixes_geo", gated_fix),
        ("geo_beats_gated", geo_beats),
    ]:
        profile_rows.append(
            {
                "profile": name,
                "rows": int(len(df)),
                "gt_rot_med_deg": pd.to_numeric(df["gt_rot_med_deg"], errors="coerce").mean(),
                "parallax_px_candidate": pd.to_numeric(df["parallax_px_candidate"], errors="coerce").mean(),
                "reproj_p90_px": pd.to_numeric(df["reproj_p90_px"], errors="coerce").mean(),
                "tri_points_candidate": pd.to_numeric(df["tri_points_candidate"], errors="coerce").mean(),
                "Q_post_geom_only": pd.to_numeric(df["Q_post_geom_only"], errors="coerce").mean(),
            }
        )
    profile_df = pd.DataFrame(profile_rows)

    top_segments = contiguous_segments(shared)[:12]

    headline = [
        f"`future_high_gt_rot` 是共享难例的绝对主来源：总共 {len(merged)} 条这类样本里，有 {len(shared)} 条被 geometry 与 gated 同时判错，shared wrong rate 达到 {fmt(len(shared)/len(merged))}。",
        f"这类共享难例并不缺几何置信，它们平均 `Q_post_geom_only={fmt(profile_df.loc[profile_df['profile']=='shared_wrong','Q_post_geom_only'].iloc[0])}`、`reproj_p90={fmt(profile_df.loc[profile_df['profile']=='shared_wrong','reproj_p90_px'].iloc[0])}`，但平均 `gt_rot_deg` 只有 {fmt(profile_df.loc[profile_df['profile']=='shared_wrong','gt_rot_med_deg'].iloc[0])}，说明它更像“轻到中度旋转触发的贴边界坏样本”。",
        f"`gated` 修掉的同类样本则明显更极端：平均 `parallax={fmt(profile_df.loc[profile_df['profile']=='gated_fixes_geo','parallax_px_candidate'].iloc[0])}`、`gt_rot_deg={fmt(profile_df.loc[profile_df['profile']=='gated_fixes_geo','gt_rot_med_deg'].iloc[0])}`，这说明 gated parallax 主要对高运动子集有效。",
    ]

    judgement = [
        "`future_high_gt_rot` 共享难例更像标签边界问题而不是纯 recoverability 缺口：因为主体难例的旋转幅度并不高，当前几何与三角化质量也不差，却仍被定义为下一窗口 bad。",
        "recoverability 锚点仍有局部价值，但主要体现在更高 parallax、更高旋转的少数样本上；这解释了为什么 gated 能修掉一部分 future_high_gt_rot，但无法覆盖主体。",
        "如果下一步要继续收紧标签协议，最值得先审的是 `future_high_gt_rot@K=1` 的阈值和语义；如果要补特征，则应优先找短时可恢复性或姿态不稳定先兆，而不是再强化现有几何质量分数。",
    ]

    payload = {
        "headline": headline,
        "judgement": judgement,
        "summary": {
            "future_high_gt_rot_rows": int(len(merged)),
            "shared_wrong_rows": int(len(shared)),
            "gated_fix_rows": int(len(gated_fix)),
            "geo_beats_rows": int(len(geo_beats)),
        },
        "sequence_summary": seq.to_dict(orient="records"),
        "profile_summary": profile_df.to_dict(orient="records"),
        "top_segments": top_segments,
    }

    seq.to_csv(out_dir / "future_high_gt_rot_sequence_summary.csv", index=False)
    profile_df.to_csv(out_dir / "future_high_gt_rot_profile_summary.csv", index=False)
    pd.DataFrame(top_segments).to_csv(out_dir / "future_high_gt_rot_top_segments.csv", index=False)
    write_json(out_dir / "future_high_gt_rot_shared_summary.json", payload)
    (out_dir / "future_high_gt_rot_shared_summary.md").write_text(build_markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
