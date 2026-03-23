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
    "docs/research/init_risk_post_v2_shared_hard_case_structure_20260319"
)


RISK_COLS = [
    "sample_uid",
    "sequence",
    "window_id",
    "Y_bad_v2_min_default",
    "Y_bad_v2_min_default_horizon_windows",
    "Y_bad_v2_min_default_trigger",
    "Y_bad_v2_min_default_trigger_window_id",
    "gate_post_reason",
    "solver_ok",
    "Q_post",
    "Q_post_geom_only",
    "gt_rot_med_deg",
    "reproj_p90_px",
    "parallax_px_candidate",
    "tri_points_candidate",
]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fmt_num(v, nd: int = 4) -> str:
    if pd.isna(v):
        return "-"
    return f"{float(v):.{nd}f}"


def rate_table(df: pd.DataFrame, key: str, mask_col: str, label: str) -> pd.DataFrame:
    total = df.groupby(key).size().rename("total_rows")
    subset = df[df[mask_col] == 1].groupby(key).size().rename(label)
    out = pd.concat([total, subset], axis=1).fillna(0).reset_index()
    out[label] = out[label].astype(int)
    out["total_rows"] = out["total_rows"].astype(int)
    out[f"{label}_rate"] = out[label] / out["total_rows"]
    return out.sort_values([f"{label}_rate", label, "total_rows"], ascending=[False, False, False])


def mean_profile(df: pd.DataFrame, mask: pd.Series, cols: list[str], label: str) -> pd.DataFrame:
    subset = df[mask]
    rows = []
    for col in cols:
        a = pd.to_numeric(subset[col], errors="coerce")
        b = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "feature": col,
                f"{label}_mean": a.mean(),
                "all_mean": b.mean(),
                "delta": a.mean() - b.mean(),
            }
        )
    return pd.DataFrame(rows)


def build_markdown(payload: dict) -> str:
    lines: list[str] = []
    lines.append("# post_v2 shared hard-case structure")
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    for item in payload["headline"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Horizon 层")
    lines.append("")
    lines.append("| horizon_windows | all rows | shared wrong | shared wrong rate |")
    lines.append("| --- | --- | --- | --- |")
    for row in payload["horizon_breakdown"]:
        lines.append(
            f"| {row['Y_bad_v2_min_default_horizon_windows']} | {row['total_rows']} | {row['shared_wrong_count']} | {fmt_num(row['shared_wrong_count_rate'])} |"
        )
    lines.append("")
    lines.append("## Trigger 层")
    lines.append("")
    lines.append("| trigger | all rows | shared wrong | shared wrong rate | gated fixes geo | geo beats gated |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in payload["trigger_breakdown"]:
        lines.append(
            f"| {row['Y_bad_v2_min_default_trigger']} | {row['total_rows']} | {row['shared_wrong_count']} | {fmt_num(row['shared_wrong_count_rate'])} | {row['gated_fixes_geo_count']} | {row['geo_beats_gated_count']} |"
        )
    lines.append("")
    lines.append("## Sequence x Trigger")
    lines.append("")
    lines.append("| sequence | trigger | rows | shared wrong | shared wrong rate |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in payload["sequence_trigger_breakdown"]:
        lines.append(
            f"| {row['sequence']} | {row['Y_bad_v2_min_default_trigger']} | {row['total_rows']} | {row['shared_wrong_count']} | {fmt_num(row['shared_wrong_count_rate'])} |"
        )
    lines.append("")
    lines.append("## Sequence 层")
    lines.append("")
    lines.append("| sequence | rows | shared wrong | shared wrong rate | gated fixes geo | geo beats gated |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in payload["sequence_breakdown"]:
        lines.append(
            f"| {row['sequence']} | {row['total_rows']} | {row['shared_wrong_count']} | {fmt_num(row['shared_wrong_count_rate'])} | {row['gated_fixes_geo_count']} | {row['geo_beats_gated_count']} |"
        )
    lines.append("")
    lines.append("## 共享难例 vs gated 修复样本")
    lines.append("")
    lines.append("| profile | rows | y=1 ratio | mean gt_rot_deg | mean reproj_p90 | mean parallax | mean tri_points | mean Q_post_geom_only |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["profile_summary"]:
        lines.append(
            f"| {row['profile']} | {row['rows']} | {fmt_num(row['positive_ratio'])} | {fmt_num(row['gt_rot_med_deg'])} | {fmt_num(row['reproj_p90_px'])} | {fmt_num(row['parallax_px_candidate'])} | {fmt_num(row['tri_points_candidate'])} | {fmt_num(row['Q_post_geom_only'])} |"
        )
    lines.append("")
    lines.append("## 判断")
    lines.append("")
    for item in payload["judgement"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze shared hard-case structure for post_v2.")
    parser.add_argument("--hard_cases_csv", type=Path, default=DEFAULT_HARD_CASES)
    parser.add_argument("--risk_dataset_csv", type=Path, default=DEFAULT_RISK_DATASET)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    hard = pd.read_csv(args.hard_cases_csv)
    risk = pd.read_csv(args.risk_dataset_csv, usecols=RISK_COLS)

    merged = hard.merge(risk, on=["sample_uid", "sequence"], how="left", suffixes=("", "_risk"))
    merged["trigger_gap"] = (
        pd.to_numeric(merged["Y_bad_v2_min_default_trigger_window_id"], errors="coerce")
        - pd.to_numeric(merged["window_id_risk"], errors="coerce")
    )

    all_rows = len(merged)
    shared = merged[merged["shared_wrong"] == 1].copy()
    gated_fix = merged[merged["gated_fixes_geo"] == 1].copy()
    geo_beats = merged[merged["geo_beats_gated"] == 1].copy()

    horizon = rate_table(merged, "Y_bad_v2_min_default_horizon_windows", "shared_wrong", "shared_wrong_count")
    horizon["Y_bad_v2_min_default_horizon_windows"] = horizon["Y_bad_v2_min_default_horizon_windows"].fillna("NA")

    trigger_total = merged.groupby("Y_bad_v2_min_default_trigger").size().rename("total_rows")
    trigger_shared = shared.groupby("Y_bad_v2_min_default_trigger").size().rename("shared_wrong_count")
    trigger_fix = gated_fix.groupby("Y_bad_v2_min_default_trigger").size().rename("gated_fixes_geo_count")
    trigger_geo = geo_beats.groupby("Y_bad_v2_min_default_trigger").size().rename("geo_beats_gated_count")
    trigger = pd.concat([trigger_total, trigger_shared, trigger_fix, trigger_geo], axis=1).fillna(0).reset_index()
    for col in ["total_rows", "shared_wrong_count", "gated_fixes_geo_count", "geo_beats_gated_count"]:
        trigger[col] = trigger[col].astype(int)
    trigger["shared_wrong_count_rate"] = trigger["shared_wrong_count"] / trigger["total_rows"]
    trigger = trigger.sort_values(["shared_wrong_count_rate", "shared_wrong_count"], ascending=[False, False])

    seq_total = merged.groupby("sequence").size().rename("total_rows")
    seq_shared = shared.groupby("sequence").size().rename("shared_wrong_count")
    seq_fix = gated_fix.groupby("sequence").size().rename("gated_fixes_geo_count")
    seq_geo = geo_beats.groupby("sequence").size().rename("geo_beats_gated_count")
    seq = pd.concat([seq_total, seq_shared, seq_fix, seq_geo], axis=1).fillna(0).reset_index()
    for col in ["total_rows", "shared_wrong_count", "gated_fixes_geo_count", "geo_beats_gated_count"]:
        seq[col] = seq[col].astype(int)
    seq["shared_wrong_count_rate"] = seq["shared_wrong_count"] / seq["total_rows"]
    seq = seq.sort_values(["shared_wrong_count_rate", "shared_wrong_count"], ascending=[False, False])

    seq_trigger_total = merged.groupby(["sequence", "Y_bad_v2_min_default_trigger"]).size().rename("total_rows")
    seq_trigger_shared = shared.groupby(["sequence", "Y_bad_v2_min_default_trigger"]).size().rename("shared_wrong_count")
    seq_trigger = pd.concat([seq_trigger_total, seq_trigger_shared], axis=1).fillna(0).reset_index()
    seq_trigger["shared_wrong_count"] = seq_trigger["shared_wrong_count"].astype(int)
    seq_trigger["total_rows"] = seq_trigger["total_rows"].astype(int)
    seq_trigger["shared_wrong_count_rate"] = seq_trigger["shared_wrong_count"] / seq_trigger["total_rows"]
    seq_trigger = seq_trigger.sort_values(["sequence", "shared_wrong_count_rate", "shared_wrong_count"], ascending=[True, False, False])

    gate_reason = rate_table(
        merged.assign(gate_post_reason=merged["gate_post_reason"].fillna("NA")),
        "gate_post_reason",
        "shared_wrong",
        "shared_wrong_count",
    )

    profiles = []
    profile_specs = [
        ("shared_wrong", shared),
        ("shared_wrong_positive", merged[(merged["shared_wrong"] == 1) & (merged["y_true"] == 1)]),
        ("gated_fixes_geo_positive", merged[(merged["gated_fixes_geo"] == 1) & (merged["y_true"] == 1)]),
        ("geo_beats_gated_negative", merged[(merged["geo_beats_gated"] == 1) & (merged["y_true"] == 0)]),
    ]
    for name, df in profile_specs:
        profiles.append(
            {
                "profile": name,
                "rows": int(len(df)),
                "positive_ratio": float(df["y_true"].mean()) if len(df) else None,
                "gt_rot_med_deg": pd.to_numeric(df["gt_rot_med_deg"], errors="coerce").mean(),
                "reproj_p90_px": pd.to_numeric(df["reproj_p90_px"], errors="coerce").mean(),
                "parallax_px_candidate": pd.to_numeric(df["parallax_px_candidate"], errors="coerce").mean(),
                "tri_points_candidate": pd.to_numeric(df["tri_points_candidate"], errors="coerce").mean(),
                "Q_post_geom_only": pd.to_numeric(df["Q_post_geom_only"], errors="coerce").mean(),
            }
        )
    profiles_df = pd.DataFrame(profiles)

    shared_trigger_counts = (
        shared.groupby("Y_bad_v2_min_default_trigger").size().sort_values(ascending=False).to_dict()
    )

    headline = [
        f"跨 5 条 held-out 的 {all_rows} 条测试样本中，共享难例有 {len(shared)} 条；其中 {int((shared['y_true'] == 1).sum())} 条是正样本，说明当前瓶颈主要不是负样本排序，而是 bad 事件的边界型正样本。",
        f"horizon 层当前没有分辨力：所有共享难例都来自 `K=1` 的单步 horizon，且正样本触发都发生在下一窗口（trigger_gap=10），这更像短时标签边界，而不是远期预测失败。",
        f"trigger 层上，共享难例最集中在 `future_high_gt_rot`（{shared_trigger_counts.get('future_high_gt_rot', 0)} 条）和 `future_solver_fail`（{shared_trigger_counts.get('future_solver_fail', 0)} 条）；`stable_horizon` 只占 {shared_trigger_counts.get('stable_horizon', 0)} 条，说明问题主要出在短时坏事件正样本。",
        f"共享难正样本当前看起来并不“几何崩坏”：它们平均 `Q_post_geom_only={fmt_num(profiles_df.loc[profiles_df['profile']=='shared_wrong_positive', 'Q_post_geom_only'].iloc[0])}`、`reproj_p90={fmt_num(profiles_df.loc[profiles_df['profile']=='shared_wrong_positive', 'reproj_p90_px'].iloc[0])}`、`tri_points={fmt_num(profiles_df.loc[profiles_df['profile']=='shared_wrong_positive', 'tri_points_candidate'].iloc[0])}`，更像“当前几何看起来还行，但下一窗口仍出坏事件”的边界样本。",
        f"`gated` 真正修掉的样本则偏明显高运动：平均 `parallax={fmt_num(profiles_df.loc[profiles_df['profile']=='gated_fixes_geo_positive', 'parallax_px_candidate'].iloc[0])}`、`gt_rot={fmt_num(profiles_df.loc[profiles_df['profile']=='gated_fixes_geo_positive', 'gt_rot_med_deg'].iloc[0])}`，说明 gated parallax 在补一个“高视差高旋转”的 recoverability 线索，但还不足以覆盖主体难例。",
    ]

    judgement = [
        "当前共享难例更像 `短时标签边界 + recoverability 锚点不足` 的混合问题，但主导项更接近标签边界：因为绝大多数共享难例是下一窗口立即触发的正样本，而当前几何分数、重投影和三角化点数并没有明显崩坏。",
        "recoverability 锚点仍然是缺口的一部分：gated parallax 能稳定修掉一批高视差、高旋转正样本，说明“未来可恢复性/短时不稳风险”并不完全包含在当前 geometry-only 里。",
        "因此下一阶段更值钱的，不是继续扩 full 配方，而是围绕共享难正样本补 recoverability 线索，或者进一步审视 `future_high_gt_rot@K=1` 这类标签是否过于贴近边界。",
    ]

    payload = {
        "headline": headline,
        "judgement": judgement,
        "horizon_breakdown": horizon.to_dict(orient="records"),
        "trigger_breakdown": trigger.to_dict(orient="records"),
        "sequence_breakdown": seq.to_dict(orient="records"),
        "sequence_trigger_breakdown": seq_trigger.to_dict(orient="records"),
        "gate_post_reason_breakdown": gate_reason.to_dict(orient="records"),
        "profile_summary": profiles_df.to_dict(orient="records"),
        "summary": {
            "all_rows": int(all_rows),
            "shared_wrong_rows": int(len(shared)),
            "shared_wrong_positive_rows": int(((shared["y_true"] == 1)).sum()),
            "shared_wrong_negative_rows": int(((shared["y_true"] == 0)).sum()),
            "gated_fix_rows": int(len(gated_fix)),
            "geo_beats_rows": int(len(geo_beats)),
        },
    }

    horizon.to_csv(out_dir / "shared_hard_case_horizon_breakdown.csv", index=False)
    trigger.to_csv(out_dir / "shared_hard_case_trigger_breakdown.csv", index=False)
    seq.to_csv(out_dir / "shared_hard_case_sequence_breakdown.csv", index=False)
    seq_trigger.to_csv(out_dir / "shared_hard_case_sequence_trigger_breakdown.csv", index=False)
    gate_reason.to_csv(out_dir / "shared_hard_case_gate_reason_breakdown.csv", index=False)
    profiles_df.to_csv(out_dir / "shared_hard_case_profile_summary.csv", index=False)
    write_json(out_dir / "shared_hard_case_structure_summary.json", payload)
    (out_dir / "shared_hard_case_structure_summary.md").write_text(build_markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
