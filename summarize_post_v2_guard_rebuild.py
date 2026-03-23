#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import pandas as pd


DEFAULT_GUARD_ROOT = Path("/mnt/g/Result/VIODE/post_v2_guard_audit_20260320")
DEFAULT_TAU75_HOLDOUT_COMPARISON = Path(
    "docs/research/init_risk_post_v2_tau75_rebuild_20260320/post_v2_tau75_rebuild_holdout_comparison.csv"
)
DEFAULT_TAU75_SHARED_SUMMARY = Path(
    "docs/research/init_risk_post_v2_shared_hard_cases_tau7p5_20260320/post_v2_shared_hard_cases_summary.json"
)
DEFAULT_GUARD_SHARED_SUMMARY = Path(
    "docs/research/init_risk_post_v2_shared_hard_cases_guard12p693_20260320/post_v2_shared_hard_cases_summary.json"
)
DEFAULT_OUT_DIR = Path(
    "docs/research/init_risk_post_v2_guard_rebuild_20260320"
)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fmt(v, nd: int = 4) -> str:
    if pd.isna(v):
        return "-"
    return f"{float(v):.{nd}f}"


def load_guard_holdouts(root: Path) -> pd.DataFrame:
    rows = []
    pattern = str(root / "post_v2_holdout_*_guard12p693" / "post_v2_core4_holdout_summary.csv")
    for p in sorted(glob.glob(pattern)):
        df = pd.read_csv(p)
        seq = os.path.basename(os.path.dirname(p)).replace("post_v2_holdout_", "").replace("_guard12p693", "")
        parts = seq.split("_")
        if len(parts) >= 3:
            sequence = f"{parts[0]}_{parts[1]}/{parts[2]}"
        else:
            sequence = seq.replace("_", "/")
        rec = {"sequence_tag": seq, "sequence": sequence}
        for _, r in df.iterrows():
            rec[r["model"]] = r["auroc"]
        rows.append(rec)
    return pd.DataFrame(rows)


def build_markdown(payload: dict) -> str:
    lines = []
    lines.append("# post_v2 guard rebuild summary")
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    for item in payload["headline"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## held-out 对照")
    lines.append("")
    lines.append("| sequence | tau7.5 geometry | tau7.5 gated | guard geometry | guard gated | guard best model |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in payload["holdout_rows"]:
        lines.append(
            f"| {row['sequence']} | {fmt(row['tau75_geometry'])} | {fmt(row['tau75_gated'])} | {fmt(row['guard_geometry'])} | {fmt(row['guard_gated'])} | {row['guard_best_model']} |"
        )
    lines.append("")
    lines.append("## 共享难例对照")
    lines.append("")
    lines.append("| version | both_wrong | gated_fixes_geo | geo_beats_gated |")
    lines.append("| --- | --- | --- | --- |")
    for row in payload["shared_rows"]:
        lines.append(
            f"| {row['version']} | {row['both_wrong']} | {row['gated_fixes_geo']} | {row['geo_beats_gated']} |"
        )
    lines.append("")
    lines.append("## 判断")
    lines.append("")
    for item in payload["judgement"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize post_v2 guard-label rebuild against tau=7.5.")
    parser.add_argument("--guard_root", type=Path, default=DEFAULT_GUARD_ROOT)
    parser.add_argument("--tau75_holdout_comparison_csv", type=Path, default=DEFAULT_TAU75_HOLDOUT_COMPARISON)
    parser.add_argument("--tau75_shared_summary_json", type=Path, default=DEFAULT_TAU75_SHARED_SUMMARY)
    parser.add_argument("--guard_shared_summary_json", type=Path, default=DEFAULT_GUARD_SHARED_SUMMARY)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    tau75 = pd.read_csv(args.tau75_holdout_comparison_csv)
    guard = load_guard_holdouts(args.guard_root)

    merged = tau75.merge(guard, on="sequence", how="left")
    merged["guard_best_model"] = merged[["drop_front", "full_anchor", "geometry_plus_gated_parallax", "geometry_only"]].idxmax(axis=1)
    merged["guard_best_model"] = merged["guard_best_model"].replace(
        {
            "drop_front": "drop_front",
            "full_anchor": "full_anchor",
            "geometry_plus_gated_parallax": "gated",
            "geometry_only": "geometry",
        }
    )

    with open(args.tau75_shared_summary_json, "r", encoding="utf-8") as f:
        tau75_shared = json.load(f)
    with open(args.guard_shared_summary_json, "r", encoding="utf-8") as f:
        guard_shared = json.load(f)

    shared_rows = [
        {
            "version": "tau7.5",
            "both_wrong": int(tau75_shared["headline"][0].split("其中 both-wrong 共有 ")[1].split(" 条")[0]),
            "gated_fixes_geo": int(tau75_shared["headline"][1].split("样本有 ")[1].split(" 条")[0]),
            "geo_beats_gated": int(tau75_shared["headline"][1].split("样本有 ")[2].split(" 条")[0]),
        },
        {
            "version": "guard_parallax12p693",
            "both_wrong": int(guard_shared["headline"][0].split("其中 both-wrong 共有 ")[1].split(" 条")[0]),
            "gated_fixes_geo": int(guard_shared["headline"][1].split("样本有 ")[1].split(" 条")[0]),
            "geo_beats_gated": int(guard_shared["headline"][1].split("样本有 ")[2].split(" 条")[0]),
        },
    ]

    gated_best = int((merged["guard_best_model"] == "gated").sum())
    headline = [
        f"这版 guard 标签没有把 `geometry+gated_parallax` 变成更稳的 held-out 主赢家：在 5 条 held-out 里，guard 版 `gated` 只拿到 {gated_best}/5 次第一，而 `tau=7.5` 下它相对 geometry 的直接领先曾经是 2/5。",
        f"guard 标签确实继续清掉了一部分共享难例：both-wrong 从 {shared_rows[0]['both_wrong']} 降到 {shared_rows[1]['both_wrong']}，同时 `gated_fixes_geo` 从 {shared_rows[0]['gated_fixes_geo']} 升到 {shared_rows[1]['gated_fixes_geo']}。",
        "但更关键的是，任务结构本身被改写了：`drop_front` 成为 4/5 条 held-out 的第一名，说明这版 guard 更像在推动标签语义重排，而不是单纯让 gated 候选更稳。",
    ]
    judgement = [
        "recoverability-aware guard 在标签层是有选择性的，但训练级效果并没有沿着“扶正 geometry+gated_parallax”这条线收敛。",
        "它更像是在把 future_high_gt_rot 的主体样本重新分配到一种更适合 candidate+geometry 配方的任务上，因此不能直接当作 tau=7.5 的升级版。",
        "更稳的后续是：把这版 guard 保留为定义实验结果，不立刻切进主线标签协议。",
    ]

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    holdout_rows = []
    for _, row in merged.iterrows():
        holdout_rows.append(
            {
                "sequence": row["sequence"],
                "tau75_geometry": row["tau75_geometry"],
                "tau75_gated": row["tau75_gated"],
                "guard_geometry": row["geometry_only"],
                "guard_gated": row["geometry_plus_gated_parallax"],
                "guard_best_model": row["guard_best_model"],
            }
        )

    payload = {
        "headline": headline,
        "holdout_rows": holdout_rows,
        "shared_rows": shared_rows,
        "judgement": judgement,
    }

    pd.DataFrame(holdout_rows).to_csv(out_dir / "post_v2_guard_rebuild_holdout_comparison.csv", index=False)
    pd.DataFrame(shared_rows).to_csv(out_dir / "post_v2_guard_rebuild_shared_comparison.csv", index=False)
    write_json(out_dir / "post_v2_guard_rebuild_summary.json", payload)
    (out_dir / "post_v2_guard_rebuild_summary.md").write_text(build_markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
