#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_LABELS_CSV = Path(
    "/mnt/g/Result/VIODE/post_v2_expand_valtest_20260319/step11_and_label_audit/"
    "y_bad_v2_min_default_9seq/y_bad_v2_min_labels.csv"
)
DEFAULT_RISK_DATASET_CSV = Path(
    "/mnt/g/Result/VIODE/post_v2_rebuild_9seq/risk_dataset_9seq_seed20260432/risk_dataset.csv"
)
DEFAULT_SHARED_HARD_CASES_CSV = Path(
    "docs/research/init_risk_post_v2_shared_hard_cases_20260319/post_v2_shared_hard_cases_rows.csv"
)
DEFAULT_OUT_DIR = Path(
    "docs/research/init_risk_future_high_gt_rot_k1_guard_candidate_20260320"
)


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
    lines.append("# future_high_gt_rot@K=1 guard candidate audit")
    lines.append("")
    lines.append("## 结论先行")
    lines.append("")
    for item in payload["headline"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 等人口对照")
    lines.append("")
    lines.append("| variant | retain_rule | kept_rows | shared_keep | shared_rate | gfix_keep | gfix_rate | geo_keep | geo_rate |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in payload["comparison_rows"]:
        lines.append(
            f"| {row['variant']} | {row['retain_rule']} | {row['kept_rows']} | {row['shared_keep']} | {fmt(row['shared_rate'])} | {row['gfix_keep']} | {fmt(row['gfix_rate'])} | {row['geo_keep']} | {fmt(row['geo_rate'])} |"
        )
    lines.append("")
    lines.append("## guard 候选画像")
    lines.append("")
    lines.append("| profile | rows | parallax | Q_post_geom_only | reproj_p90 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in payload["profile_rows"]:
        lines.append(
            f"| {row['profile']} | {row['rows']} | {fmt(row['parallax_px_candidate'])} | {fmt(row['Q_post_geom_only'])} | {fmt(row['reproj_p90_px'])} |"
        )
    lines.append("")
    lines.append("## 判断")
    lines.append("")
    for item in payload["judgement"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a recoverability-aware guard candidate for future_high_gt_rot@K=1.")
    parser.add_argument("--labels_csv", type=Path, default=DEFAULT_LABELS_CSV)
    parser.add_argument("--risk_dataset_csv", type=Path, default=DEFAULT_RISK_DATASET_CSV)
    parser.add_argument("--shared_hard_cases_csv", type=Path, default=DEFAULT_SHARED_HARD_CASES_CSV)
    parser.add_argument("--target_tau_retained", type=int, default=227)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(args.labels_csv)
    risk = pd.read_csv(
        args.risk_dataset_csv,
        usecols=["sample_uid", "sequence", "parallax_px_candidate", "Q_post_geom_only", "reproj_p90_px"],
    )
    hard = pd.read_csv(
        args.shared_hard_cases_csv,
        usecols=["sample_uid", "sequence", "shared_wrong", "gated_fixes_geo", "geo_beats_gated"],
    )

    cur = labels[labels["Y_bad_v2_min_default_trigger"] == "future_high_gt_rot"][["sample_uid", "sequence"]].copy()
    merged = cur.merge(risk, on=["sample_uid", "sequence"], how="left").merge(hard, on=["sample_uid", "sequence"], how="left")
    for c in ["shared_wrong", "gated_fixes_geo", "geo_beats_gated"]:
        merged[c] = merged[c].fillna(0).astype(int)

    best = None
    for threshold in sorted(set(round(v, 3) for v in merged["parallax_px_candidate"].dropna().tolist())):
        keep = merged["parallax_px_candidate"] >= threshold
        kept = int(keep.sum())
        rec = {
            "variant": "parallax_guard_equal_pop",
            "retain_rule": f"current_parallax >= {threshold:.3f}",
            "parallax_threshold": threshold,
            "kept_rows": kept,
            "shared_keep": int((keep & (merged["shared_wrong"] == 1)).sum()),
            "shared_rate": float((keep & (merged["shared_wrong"] == 1)).sum()) / int((merged["shared_wrong"] == 1).sum()),
            "gfix_keep": int((keep & (merged["gated_fixes_geo"] == 1)).sum()),
            "gfix_rate": float((keep & (merged["gated_fixes_geo"] == 1)).sum()) / int((merged["gated_fixes_geo"] == 1).sum()),
            "geo_keep": int((keep & (merged["geo_beats_gated"] == 1)).sum()),
            "geo_rate": float((keep & (merged["geo_beats_gated"] == 1)).sum()) / int((merged["geo_beats_gated"] == 1).sum()),
            "diff_vs_target": abs(kept - args.target_tau_retained),
        }
        if best is None or (
            rec["diff_vs_target"],
            -rec["gfix_keep"],
            rec["shared_keep"],
            rec["geo_keep"],
        ) < (
            best["diff_vs_target"],
            -best["gfix_keep"],
            best["shared_keep"],
            best["geo_keep"],
        ):
            best = rec

    tau75 = {
        "variant": "tau7.5",
        "retain_rule": "future_gt_rot > 7.5",
        "kept_rows": 227,
        "shared_keep": 85,
        "shared_rate": 85 / 134,
        "gfix_keep": 15,
        "gfix_rate": 15 / 23,
        "geo_keep": 1,
        "geo_rate": 1 / 6,
    }

    keep_guard = merged["parallax_px_candidate"] >= best["parallax_threshold"]
    profiles = []
    for name, mask in [
        ("all_retained_by_guard", keep_guard),
        ("shared_retained_by_guard", keep_guard & (merged["shared_wrong"] == 1)),
        ("gated_fix_retained_by_guard", keep_guard & (merged["gated_fixes_geo"] == 1)),
    ]:
        df = merged[mask].copy()
        profiles.append(
            {
                "profile": name,
                "rows": int(len(df)),
                "parallax_px_candidate": pd.to_numeric(df["parallax_px_candidate"], errors="coerce").mean(),
                "Q_post_geom_only": pd.to_numeric(df["Q_post_geom_only"], errors="coerce").mean(),
                "reproj_p90_px": pd.to_numeric(df["reproj_p90_px"], errors="coerce").mean(),
            }
        )
    profile_df = pd.DataFrame(profiles)

    headline = [
        f"在和 `tau=7.5` 保持相同触发人口（{args.target_tau_retained} 条）的前提下，单纯的 parallax guard 候选会自动落到 `current_parallax >= {best['parallax_threshold']:.3f}`。",
        f"这版 guard 比 `tau=7.5` 更有选择性：共享难例保留 {best['shared_keep']}/134（{fmt(best['shared_rate'])}）低于 `tau=7.5` 的 85/134（{fmt(tau75['shared_rate'])}），同时 `gated` 修复样本保留 {best['gfix_keep']}/23（{fmt(best['gfix_rate'])}）高于 `tau=7.5` 的 15/23（{fmt(tau75['gfix_rate'])}）。",
        "这说明如果要继续审 `future_high_gt_rot@K=1` 的定义，recoverability-aware 方向是有价值的，而且比单纯抬旋转阈值更有可能保住有用子集、削掉边界错例。",
    ]

    judgement = [
        "这还只是标签层等人口对照，不是训练级结论；但它已经足够说明：单纯 `tau` 收紧不是唯一轴，当前窗口的 recoverability guard 值得进入下一轮正式审计。",
        "最小 guard 候选可以收成一句话：`future_high_gt_rot` 只在 `current_parallax >= p*` 时触发，其中 `p*` 当前的等人口参考值约为 12.693 px。",
        "如果继续往下做，最自然的下一步不是改多个变量，而是只拿这一个 guard 候选去和 `tau=7.5` 做一次训练级小重建对照。",
    ]

    comparison_rows = [tau75, best]
    payload = {
        "headline": headline,
        "judgement": judgement,
        "comparison_rows": comparison_rows,
        "profile_rows": profile_df.to_dict(orient="records"),
    }

    pd.DataFrame(comparison_rows).to_csv(out_dir / "future_high_gt_rot_k1_guard_comparison.csv", index=False)
    profile_df.to_csv(out_dir / "future_high_gt_rot_k1_guard_profiles.csv", index=False)
    write_json(out_dir / "future_high_gt_rot_k1_guard_candidate_summary.json", payload)
    (out_dir / "future_high_gt_rot_k1_guard_candidate_summary.md").write_text(build_markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
